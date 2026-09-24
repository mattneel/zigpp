# Zig++ → AIR → metallib → GPU spike

End-to-end spike: Zig device code written against raw AIR intrinsics, rewritten into Apple's
AIR conventions, wrapped into a `.metallib`, and executed on an Apple GPU.

| stage | file | what it does |
|---|---|---|
| 1 | `air-rewrite.cpp` | rewrites a Zig++ bitcode module (compiled for the nvptx64 stand-in target) into Apple's AIR conventions |
| 2 | the metallib writer (sibling task, same directory) | wraps the downgraded LLVM-14 module in a `.metallib` container |
| 3 | `host.zig` | macOS program that loads the container and runs the kernels on the GPU |

Kernel names are part of the contract and never change: `vadd`, `reduce`, `parsef`.

## 1. air-rewrite

```
air-rewrite <input.bc> -o <output.bc> --spec <spec> [--air 2.8] [--metal 4.0]
            [--deploy 26.0.0] [--sdk 26.5.0] [--opt O3|O0] [--ident <string>]
            [--source-name <path>]
```

`--spec` is the kernel argument spec documented at the top of `spike/kernels.spec`:
`kernel <name>` followed by one `buffer <index> <read|write|read_write> <air.arg_type_name>
<size> <align> <name>` or `builtin <index> <air builtin name> <air.arg_type_name> <name>`
line per argument, in IR parameter order. A module whose kernel signature disagrees with the
spec (argument count, non-pointer buffer argument, non-integer builtin argument, a
`ptx_kernel` function missing from the spec) is rejected.

Build (build directory outside the repository, `-DNDEBUG` is required because the LLVM
archives are a release build):

```sh
env PATH=/usr/bin:/bin cmake -B /tmp/metal-spike/air-rewrite-build \
    -S test/standalone/metal_spike -G Ninja \
    -DLLVM_DIR=/home/autark/src/zig/zigpp-bootstrap/out/host/lib/cmake/llvm \
    -DCMAKE_CXX_FLAGS=-DNDEBUG
env PATH=/usr/bin:/bin cmake --build /tmp/metal-spike/air-rewrite-build
```

Rewriting steps, in order:

1. Read the spec, check it against the module's kernels.
2. Set the AIR triple (`air64_v<major><minor>` for AIR >= 2.6, else `air64`, plus
   `-apple-macosx<deploy>`) and Apple's AIR data layout.
3. Sanitize: drop nvptx `target-cpu`/`target-features` and `!nvvm.annotations`, give kernels
   the C calling convention (the Zig input is `ptx_kernel`), `convergent nounwind`,
   `local_unnamed_addr`, spec-derived parameter attributes; internalize private/linkonce
   helper functions and threadgroup globals; strip parameter attributes from `air.*`
   declarations and normalize them to `convergent nounwind` (barrier/SIMD) resp. `nounwind`
   (atomics).
4. Attach the AIR metadata (`!air.version`, `!air.language_version`, `!air.compile_options`,
   `!air.kernel` with one kernel node per kernel, `!llvm.module.flags`, and
   `!air.source_file_name`/`!llvm.ident` when asked) plus the `!arg_eltypes` hints the
   typed-pointer downgrade needs.
5. Run the LLVM middle end: `default<O3>` (with an `optnone`-respecting pipeline) at `--opt
   O3`, or `GlobalDCE` + `StripDeadPrototypes` at `--opt O0` so the output stays close to the
   input.
6. Re-normalize the kernel signatures from the spec, retype buffer GEPs, verify, write.

`air-rewrite` writes an ordinary LLVM bitcode file, i.e. it *includes* the
`0b17c0de` module-section wrapper header (`llvm-bcanalyzer -dump` shows
`BITCODE_WRAPPER_HEADER` as the first record). The container step strips it; a bare module
works in the container too.

Verification (LLVM 23 tools, Apple references in `/tmp/metal-spike`):

```sh
air-rewrite spike/kernels_rf.bc -o spike/kernels_air.bc --spec spike/kernels.spec --opt O3
llvm-dis spike/kernels_air.bc -o spike/kernels_air.ll
llvm-downgrade --bitcode-version=14.0 spike/kernels_air.bc -o spike/kernels_air14.bc
```

Observed on the spike module:

* triple `air64_v28-apple-macosx26.0.0` and Apple's data layout;
* `define void @vadd(ptr addrspace(1) readonly %0, ptr addrspace(1) readonly %1,
  ptr addrspace(1) %2, i32 %3) local_unnamed_addr`, likewise `@reduce` and `@parsef`;
* `@kernels.scratch = internal unnamed_addr addrspace(3) global [8 x float] undef, align 4`;
* 4 defined functions: the three kernels plus `fmt.parse_float.convert_slow.convertSlow__func_7`
  (cost 3660, above the GPU inline threshold), and no `fmt.parse_float.*` helpers besides it —
  they are inlined into `@parsef`;
* no `target-features`, no `ptx_kernel`, no `double` instructions;
* `!air.kernel = !{!4, !11, !20}`, each `!{ptr @fn, !{}, !{argument nodes}}`, matching
  `vadd.ll`/`reduce.ll`/`ref2.ll:138`.

Conventions this tool adds that Apple's reference modules do not literally show (there is no
Apple compiler on Linux to check them against):

* the middle end runs with the *input* module's target machine (nvptx64, the stand-in target),
  because inline costs come from the target's cost model and the generic model leaves the
  outlined `fmt.parse_float.*` helpers un-inlined;
* kernel parameter attributes are re-derived from the spec after the pipeline (O3 infers
  `writeonly`/`nofree`/`captures` of its own), keeping only the access mode;
* buffer GEPs are retyped from Zig's `[N x i8]` form to the spec's element type (`float`,
  `i32`, `i8`); the rewrite only fires when the element sizes match, so addresses cannot
  change, and it makes the walks agree with `air.arg_type_name`/`air.arg_type_size`. It
  follows the parameter's own uses, which is what O3's SROA leaves behind: at `--opt O0` the
  Zig ABI's staging alloca still exists, so those GEPs keep the `[N x i8]` form;
* threadgroup (`addrspace(3)`) globals are `internal` (Zig emits `private`), matching
  `reduce.ll:8`;
* `!arg_eltypes` hints are emitted for kernel buffer parameters (from the spec) and for `air.*`
  declarations whose name encodes the element type (e.g. `air.atomic.global.add.u.i32`). The
  typed-pointer downgrade infers most pointees from loads/stores/GEPs, but a pointer that is
  only handed to an intrinsic (reduce's counter, and the declarations' own pointer parameters)
  has no element type in the IR and would otherwise become `{} addrspace(1)*`;
* `air.compile.fast_math_enable` is only emitted when the module's FP instructions actually
  carry fast-math flags (Zig's nvptx output does not), so the option cannot disagree with
  precise FP operations; `denorms_disable`/`framebuffer_fetch_enable` stay as driver defaults.

Remaining differences from Apple's reference modules: `air.compile_options` lacks
`fast_math_enable` (above); `SDK Version` is `[3 x i32] [26, 5, 0]` where the references print
`[2 x i32] [26, 5]`; `!llvm.ident` and `!air.source_file_name` are only emitted when
`--ident`/`--source-name` are given; the kernels carry no `approx-func-fp-math`,
`no-infs-fp-math`, `no-nans-fp-math`, `no-signed-zeros-fp-math`, `no-trapping-math`,
`unsafe-fp-math`, `min-legal-vector-width`, `no-builtins` or `stack-protector-buffer-size`
attributes; FP arithmetic is precise (`fadd float`, no `fast`) and buffer walks use the
kernel argument pointers directly rather than `!tbaa`/`!alias.scope`/`air-alias-scopes`
metadata; the `addrspace(3)` scratch GEP keeps Zig's `[4 x i8]` element type; `noredzone` is
left in place.

## 3. host.zig — macOS host

`host.zig` is the aarch64-macos program that loads a `.metallib` and runs the three spike kernels
on the Mac's GPU. It is cross-compiled from Linux with no SDK and no Objective-C compiler: it
resolves libobjc, libSystem and the Metal framework at run time with `dlopen`, and sends every
Objective-C message through `objc_msgSend` cast to the exact method signature (arm64 has no
variadic `objc_msgSend`).

```sh
# cross-compile on Linux
zig build-exe test/standalone/metal_spike/host.zig -target aarch64-macos -O ReleaseSafe \
    -femit-bin=/tmp/metal-spike/host-macos
# on the Mac
./host-macos --selftest          # checks libobjc/Metal/the device resolve; no metallib needed
./host-macos <path-to.metallib>  # runs vadd, reduce, parsef; prints --- PASS/FAIL/SKIP per test
```

A kernel that the `.metallib` does not contain is reported as `--- SKIP`, not a failure, so a
partial pipeline can be tested. Exit status is 0 when no test failed (skips are fine).
