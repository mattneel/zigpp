# Zig++ → AIR → metallib → GPU spike

End-to-end spike for [issue #11](https://github.com/mattneel/zigpp/issues/11): Zig device
code is compiled by Zig++, rewritten into Apple's AIR conventions, downgraded to the LLVM 14
bitcode format Apple's reader wants, wrapped in a `.metallib`, and run on an Apple GPU from a
Zig++ host program that talks to Metal through the Objective-C runtime. No SDK, no Objective-C
compiler, no Metal toolchain needed on the machine that does the compiling.

The design document with every convention this spike pinned is `doc/proposals/metal.md`.

## Status

Measured on an Apple M4 (macOS 26.6.2, Metal 4) with the Zig++ at `56013e0da` (LLVM 23.1.2):

| pipeline | result |
|---|---|
| Apple MSL → `xcrun metal` AIR → `llvm-downgrade 14.0` → `xcrun metallib` → GPU | `vadd` 0/4096 mismatches; `reduce` sums exact, counter 4 |
| Apple MSL → AIR → downgrade → the spike's own metallib writer → GPU | `vadd` 0/4096 mismatches |
| **Zig++ kernel → `air-rewrite` → `llvm-downgrade` → the spike's writer → the spike's host** | **`vadd` PASS, `reduce` PASS** |
| the `parsef` kernel (`std.fmt.parseFloat(f32, …)`) | module is valid AIR (`xcrun air-opt` accepts it), Apple's compiler service crashes compiling it — see "Open items" |

Files: `kernels.zig` + `air.zig` (the kernels and the AIR intrinsic declarations they use),
`kernels.spec` (kernel argument spec), `air-rewrite.cpp` (bitcode → AIR conventions),
`metallib.zig` (the container writer/reader), `host.zig` (the macOS host), `run.sh` (the
recipe below).

## The pipeline

Kernels are compiled for the **nvptx64 stand-in target**, whose LLVM address spaces are
already AIR's (`.global` = 1, `.constant` = 2, `.shared` = 3); `air-rewrite` then retypes the
entries to Apple's kernel conventions. That is the "least machinery" route: no new backend,
no new target, and every rewrite it performs is a rewrite the real target must emit.

```sh
# 1. kernels -> modern LLVM bitcode
#    ZIG_LIB_DIR must match the compiler binary's source tree (see Notes).
ZIG_LIB_DIR=/home/autark/src/zig/zig-amdgpu/lib /tmp/zig-amdgpu-final/bin/zig build-obj \
    test/standalone/metal_spike/kernels.zig -target nvptx64-cuda -fno-compiler-rt \
    -OReleaseFast -fno-emit-bin -femit-llvm-bc=kernels.bc

# 2. modern bitcode -> AIR conventions (triple, layout, !air.kernel, attributes, arg types)
air-rewrite kernels.bc -o kernels.air.bc --spec kernels.spec --opt O3

# 3. AIR -> LLVM 14 bitcode (typed pointers, what Apple's reader accepts)
llvm-downgrade kernels.air.bc --bitcode-version=14.0 -o kernels.air14.bc

# 4. bitcode -> .metallib (one library, one function group per kernel name)
metallib write --bitcode kernels.air14.bc --name vadd --name reduce --name parsef \
    --air 2.8 --metal 4.0 --format 1.2.9 --platform 26.0.0 --uuid -o kernels.metallib
```

`run.sh` runs exactly these four steps. On the Mac:

```sh
./host kernels.metallib
```

which prints, for the Zig++-produced library:

```
metal spike: /tmp/spike/k5_wrapped.metallib on Apple M4, registryID 0x1000003c9
--- PASS: vadd: 4096 f32, 64 threadgroups of 64 threads, c[i] == a[i] + b[i]
--- PASS: reduce: 4 threadgroups of 256 threads over 1024 f32, every group sum 1920 and counter 4
--- FAIL: parsef: newComputePipelineStateWithFunction:error: failed: Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries.
metal spike: FAIL: 2 passed, 1 failed, 0 skipped
```

## What had to be rewritten (the conventions the real target must emit)

`air-rewrite` performs, in order:

1. **Triple and data layout** — `air64_v28-apple-macosx26.0.0` and
   `e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-…-n8:16:32`
   (AIR 2.8 pointers are 64-bit in *every* address space, unlike the stand-in target's).
2. **Sanitizing** — drop `"target-cpu"`/`"target-features"` attributes, `nvvm.annotations`,
   unused target globals, the `%Target.*` runtime data; set the kernel calling convention to C;
   add `convergent nounwind`; normalize `air.*` declarations to Apple's parameter shapes.
3. **Kernel argument conventions** — buffer parameters keep their `addrspace(1)`/`addrspace(2)`
   pointers; each kernel's used builtins become **trailing value parameters** (`i32` for `uint`
   builtins) in a fixed order; GEPs are retyped to the buffer's element type from the spec.
4. **Metadata** — `!air.version`, `!air.language_version`, `!air.compile_options`,
   `!llvm.module.flags` (SDK version, `air.max_*`, wchar/frame-pointer) and `!air.kernel` with
   one node per kernel: `!air.kernel = !{!k1, !k2, !k3}` where
   `!k1 = !{ptr @vadd, !{}, !{…arg nodes…}}`. Apple's reader *rejects* an extra level of nesting
   here (`air-opt` reports `metadata AIKernelFunction is corrupted`) and rejects pointer
   arguments with an unknown pointee.
5. **Typed-pointer hints** — `!arg_eltypes` on declarations whose pointer parameters have no
   load/store/GEP to learn the pointee from (`air.atomic.global.add.u.i32` in the reduction),
   because the downgrader otherwise emits `{} addrspace(1)*` and Apple rejects that signature.
6. **LLVM middle end** — the O3 pipeline over the AIR module inlines the outlined std
   helpers, runs SROA/GlobalDCE and leaves exactly what the AIR backend sees.

The kernels themselves are written against raw AIR intrinsics declared in `air.zig`
(`air.wg.barrier`, `air.simd_sum.f32`, `air.atomic.global.add.u.i32`), i.e. the shapes
`std.gpu` must lower to for a Metal target; `kernels.spec` stands in for the type information
the real compiler derives from the Zig type system.

## Container facts the spike measured

* Apple's reader/compiler accepts a module that **exactly** follows the conventions above;
  anything off by a level (metadata nesting, missing pointee type) is rejected with a terse
  diagnostic or a compiler crash, so `air-rewrite`'s checks are not cosmetic.
* The writer stores the module bytes **verbatim**, including the 20-byte
  `0b17c0de`-prefixed module section header that `air-rewrite` (like Apple's `metal -c`) puts in
  front of the bitcode. A module stored *without* that header loaded for some modules but failed
  for others (`unable to copy bitcode for function`), so the spike keeps Apple's form.
* The function list is what `newFunctionWithName:` resolves against: three function groups with
  the same module bytes resolve `vadd`, `reduce` and `parsef` from one library.
* File version 1.2.9, platform 26.0.0, `TYPE` = 2 (kernel), SHA-256 of the module in `HASH`,
  `MDSZ` = module length, `VERS` = (AIR major, AIR minor, MSL major, MSL minor).

## Host facts the spike measured

* `objc_msgSend` through exactly-typed function pointers, `dispatch_data_create` for the library
  data, `MTLResourceStorageModeShared` buffers read back through `contents` after
  `waitUntilCompleted`, and `NSError**` on every `error:` selector — without it failures are
  silent nulls.
* The Metal *runtime* is all that is needed to load and run: the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain`, 688 MB) is only needed to produce reference
  AIR from MSL, and `xcrun air-opt` is a precise validator for emitted modules.

## Open items

1. **`parsef`** (`std.fmt.parseFloat(f32, …)`): the module passes `xcrun air-opt` and
   `xcrun metal-opt -O3`, and `vadd`/`reduce` compile from the same library, but creating the
   pipeline for `parsef` crashes Apple's compiler service
   (`XPC_ERROR_CONNECTION_INTERRUPTED`). Ruled out by experiment, each by editing the emitted
   module and re-running on the M4: the `fastcc` convention of the outlined helpers (→ `ccc`),
   the f64 slow path (`convert_slow` stubbed out), `llvm.umul.with.overflow.i64` (→ `mul` plus an
   explicit false overflow flag), `llvm.ctlz.i64` (→ `air.clz.i64`), all of
   `llvm.{ctlz,umax,umin,usub.sat}` (→ `air.{clz,max.u,min.u,sub_sat.u}`), and Apple's own
   `metal-opt -O3` over the module before wrapping. What is left as the difference to Apple's
   own frontend output: `llvm.memcpy.p0.p1.i64`/`llvm.memset.p0.i64` (the 32-byte staging copy
   out of device memory), the `%BiasedFp(f64)`-typed allocas, and `llvm.assume`. Next step: read
   the Metal compiler's crash report (`~/Library/Logs/DiagnosticReports`, not readable from this
   account) or bisect the inlined body of `@parsef`.
2. Whether Apple's AIR backend accepts `alloca`-heavy kernels at all, once (1) is answered.
3. `-fno-compiler-rt` was used for the stand-in compile to keep the module small; the real
   target bundles compiler-rt for kernels (works in this tree: 588 KB module measured with
   compiler-rt included).

## Notes

* `ZIG_LIB_DIR` must point at a lib tree that matches the compiler binary's source: pairing the
  `amdgpu`-tree compiler with master's lib produces `std.lang is corrupt` panics and bogus
  calling-convention diagnostics (measured). The mangled-but-working combination used here is
  `/tmp/zig-amdgpu-final/bin/zig` + `/home/autark/src/zig/zig-amdgpu/lib`.
* The stand-in target needs explicit calling conventions (`callconv(.nvptx_kernel)` on kernels,
  `.nvptx_device` on `air.*` declarations) because the default `extern`/`export` convention
  resolution misbehaves for GPU targets in this tree.
* File transfer to the Mac: base64 the gzipped file, write it with the remote `write_file`
  device in chunks, then `base64 -D -i x.b64 -o x.gz && gunzip x.gz`; the `-d` spelling of BSD
  `base64` is unreliable in the remote shell.
