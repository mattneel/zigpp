# Metal for `std.gpu`: a compiler backend for Apple GPUs

Issue: [#11](https://github.com/mattneel/zigpp/issues/11). Spike: `test/standalone/metal_spike/`,
branch `metal-spike`. This document is grounded in what the spike proved on an Apple M4
(macOS 26.6.2, Xcode 26.6, Metal toolchain 32023.883) and marks anything inferred or untested
as such.

## 1. The route

The one Metal.jl uses, and the only one that avoids a second source back end:

```
 Zig++ kernel source (.zig, std.gpu device API)
   │  zig build-obj -tar air64-macos -femit-llvm-bc
   ▼
 LLVM IR in AIR conventions        triple air64_v28-apple-macosx26.0.0, addrspace 1/2/3,
   │                               !air.kernel metadata, air.* intrinsics
   │  (in-process; spike: the `air-rewrite` tool)
   ▼
 modern LLVM bitcode
   │  llvm-downgrade --bitcode-version=14.0
   ▼
 LLVM-14-format bitcode, typed pointers
   │  (spike: the `metallib` writer)
   ▼
 .metallib container
   │  newLibraryWithData: (ObjC runtime, no SDK needed)
   ▼
 MTLComputePipelineState, dispatchThreads, GPU
```

Each stage is a place the compiler must eventually emit; each was exercised in the spike.
What the spike ran and what it measured is in §12.

## 2. AIR conventions

Primary sources: Apple's own toolchain (`xcrun metal -c`, `xcrun metal -S -emit-llvm`,
`xcrun metallib`), Apple-authored `.metallib` files shipped in macOS, and Metal.jl/GPUCompiler.jl
(MIT) for the platform-tuning details. Everything below was read out of those, not guessed.

### 2.1 Triple

`air64_v<AIR major><AIR minor>-apple-macosx<deployment target>`

Verified by compiling one kernel with `-mmacosx-version-min` from 11.0 to 26.0 on this machine:

| deployment target | triple | `!air.version` | `!air.language_version` |
| --- | --- | --- | --- |
| 11.0 | `air64_v23-apple-macosx11.0.0` | 2.3.0 | Metal 2.3.0 |
| 12.0 | `air64_v24-apple-macosx12.0.0` | 2.4.0 | Metal 2.4.0 |
| 13.0 | `air64_v25-apple-macosx13.0.0` | 2.5.0 | Metal 3.0.0 |
| 14.0 | `air64_v26-apple-macosx14.0.0` | 2.6.0 | Metal 3.1.0 |
| 15.0 | `air64_v27-apple-macosx15.0.0` | 2.7.0 | Metal 3.2.0 |
| 26.0 | `air64_v28-apple-macosx26.0.0` | 2.8.0 | Metal 4.0.0 |

The `_v<major><minor>` suffix exists from AIR 2.6 on; older toolchains use a bare `air64`
(GPUCompiler.jl `src/metal.jl:77-89`). `metal-opt`-style tools derive the expected AIR version
from the triple and complain when `!air.version` disagrees, so the two must be set together.

### 2.2 Data layout

```
e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32
```

Identical in Apple's bitcode, Apple's textual AIR and GPUCompiler.jl (`src/metal.jl:91-97`).
Note it has **no per-address-space pointer widths**: in AIR 2.8 pointers in *every* address
space are 64-bit, including threadgroup (AS 3). This differs from NVPTX (AS 3 = 32-bit) and
AMDGPU (AS 3 = 32-bit), and it matters when a kernel is compiled through a stand-in target
whose layout disagrees: the layout must be overwritten at AIR emission, and no `ptrtoint`
may survive from the stand-in.

### 2.3 Address spaces

| AIR AS | meaning | where it appears |
| --- | --- | --- |
| 0 | generic | function-local memory (`alloca`), flat pointers |
| 1 | device | kernel buffer arguments, textures |
| 2 | constant | `constant`-qualified buffers (MSL `constant T& [[buffer(n)]]`) |
| 3 | threadgroup | kernel threadgroup variables, `[[threadgroup(n)]]` buffers |
| 4 | thread | Metal.jl's TTI lists it; not observed from the MSL frontend |
| 5 | threadgroup_imageblock | tile memory (Metal 3 imageblocks) |
| 6 | ray | ray-tracing payloads |

Metal.jl passes the same hierarchy to its LLVM passes
(`gpucompiler/src/metal.jl:16-30`). Only 0, 1, 2 and 3 are exercised by the spike.

### 2.4 Kernel ABI

A kernel is a plain function returning `void` with C calling convention:

```llvm
define void @vadd(ptr addrspace(1) noundef readonly "air-buffer-no-alias" %0,
                  ptr addrspace(1) noundef readonly "air-buffer-no-alias" %1,
                  ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %2,
                  i32 noundef %3) local_unnamed_addr #0
```

* Buffers are pointer arguments in AS 1 (device) or AS 2 (constant) — never in the generic
  space; the host binds them with `setBuffer:offset:atIndex:` by `location_index`.
* **Thread positions and other builtins are value arguments appended after the buffers**, in the
  order the kernel's metadata lists them. `<2 x i32>`/`<3 x i32>` arguments are used when MSL
  declares a 2- or 3-dimensional builtin; the spike uses one `i32` per builtin.
* Everything a kernel needs from the *outside* arrives either as a buffer or as a builtin: AIR
  has no "kernel scalar argument". Zig kernels therefore have to pass lengths/scalars through
  buffers, or use a `constant` buffer, exactly as MSL does.
* `"air-buffer-no-alias"` is a string attribute Apple's frontend attaches to buffer parameters;
  it is a no-alias hint for the middle end, not a requirement.
* Kernels that contain barriers or SIMD-group operations carry `convergent`.
* Local (threadgroup) memory is a module-level global:

  ```llvm
  @scratch = internal unnamed_addr addrspace(3) global [8 x float] undef, align 4
  ```
  its size is static and per-kernel; the runtime allocates it from the pipeline's threadgroup
  memory budget (32 KiB on the devices relevant here). Dynamic threadgroup memory
  (`setThreadgroupMemoryLength:`) is a separate feature and is not needed if every kernel
  declares its threadgroup arrays at compile time.

### 2.5 Metadata

Module metadata (Apple's own vadd kernel, abridged to the load-bearing nodes):

```llvm
!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!0 = !{i32 2, !"SDK Version", [3 x i32] [i32 26, i32 5, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!air.version = !{!{i32 2, i32 8, i32 0}}
!air.language_version = !{!{!"Metal", i32 4, i32 0, i32 0}}
!air.compile_options = !{!{!"air.compile.denorms_disable"}, !{!"air.compile.fast_math_enable"},
                         !{!"air.compile.framebuffer_fetch_enable"}}
!llvm.ident = !{!{!"Apple metal version 32023.883 (metalfe-32023.883)"}}
```

`i32 7` is LLVM's "Max" module-flag behaviour; the `air.max_*` values are the device limits the
compiler promises. `!air.source_file_name = !{!{!"/path/to/file.metal"}}` appears when the
frontend records sources.

Kernel metadata is *module named metadata*, one node per kernel:

```llvm
!air.kernel = !{!K}
!K = !{ptr @vadd, !{}, !ARGS}     ; { function, stages, arguments }
!ARGS = !{!A0, !A1, !A2, !A3}
; buffer: index into the function's parameters, kind, then key/value pairs
!A0 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read",
        !"air.address_space", i32 1, !"air.arg_type_size", i32 4,
        !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float",
        !"air.arg_name", !"a"}
; builtin: no location index, just the name and the AIR type/name for reflection
!A3 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint",
        !"air.arg_name", !"gid"}
```

Access is one of `"air.read"`, `"air.write"`, `"air.read_write"`. `air.buffer_size` (u32) is
added when the size is known statically, `air.struct_type_info` for struct-typed buffers,
`air.texture`/`air.location_index`/`air.read`/`air.write` for textures, `air.sampler` for
samplers. Builtin names observed or used by Metal.jl: `air.thread_position_in_grid`,
`air.threadgroup_position_in_grid`, `air.thread_position_in_threadgroup`,
`air.thread_index_in_simdgroup`, `air.simdgroup_index_in_threadgroup`,
`air.threads_per_threadgroup`, `air.threads_per_simdgroup`, `air.simdgroup_per_threadgroup`,
`air.threadgroups_per_grid`.

TBAA (`!tbaa`), alias scopes (`air-alias-scope-arg(N)`, `air-alias-scopes(<kernel>)`) and the
`SDK Version` node are Apple frontend extras: the middle end uses them, the back end does not
require them. The spike's modules omit all three (see §12 for whether they loaded).

### 2.6 Intrinsics

All are ordinary `declare`d functions whose names start with `air.`; Apple's reader resolves them
by name. Signatures below are copied from Apple's own AIR for AIR 2.8 / Metal 4.0.

| operation | declaration |
| --- | --- |
| threadgroup barrier | `void @air.wg.barrier(i32 mem_flags, i32 barrier_id)` |
| SIMD-group reduction | `float @air.simd_sum.f32(float)` (`simd_max`, `simd_min`, prefix variants) |
| SIMD-group permutation | `float @air.simd_shuffle_down.f32(float, i16)` (`_up`, `_xor`, `air.simd_broadcast.f32`) |
| SIMD-group vote | `i1 @air.simd_all(i1)`, `i1 @air.simd_any(i1)`, `i32 @air.simd_ballot(i1)` |
| conversions | `float @air.convert.f.f32.u.i1(i1)` — `air.convert.<dst kind>.<dst type>.<src kind>.<src type>` |
| device atomic add | `i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32 value, i32 order, i32 scope, i1 volatile)` |
| threadgroup atomic add | `i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32 value, i32 order, i32 scope, i1 volatile)` |
| atomic load/store | `i32 @air.atomic.global.load.i32(ptr addrspace(1), i32 order, i32 scope, i1 volatile)` |
| atomic compare-exchange | `i32 @air.atomic.global.cmpxchg.weak.i32(ptr, ptr expected, i32 desired, i32 success_order, i32 failure_order, i32 scope, i1 volatile)` |
| fast math | `air.fast_sqrt.f32`, `air.fast_sin.f32`, `air.fma.f32`, … (Metal.jl `src/device/intrinsics/math.jl`) |
| logging | `air.os_log`, `air.os_log_*` (Metal.jl `src/device/intrinsics/output.jl`) |

Encodings that are *not* obvious from the name:

* barrier `mem_flags`: `0` = none, `1` = device, `2` = threadgroup, `3` = device|threadgroup
  (`mem_texture` = 4). Apple emits `air.wg.barrier(i32 2, i32 1)` for
  `threadgroup_barrier(mem_flags::mem_threadgroup)`.
* atomic `scope`: `1` = threadgroup, `2` = device; `order`: `0` = relaxed. The MSL frontend only
  accepts `memory_order_relaxed` until MSL 4.1, which is also what Zig's `@atomicRmw` with
  `.monotonic` maps to.
* shuffle/broadcast lane indices are **`i16`**, not `i32`.
* the last argument of every atomic is the "volatile" flag; Apple passes `true`.

## 3. Version selection

The AIR/Metal/metallib triple must be chosen per deployment target. Metal.jl's tables
(`src/version.jl`) match what Apple's toolchain does on this machine (§2.1), so the compiler
should ship the same table:

| macOS | AIR | MSL | metallib file version |
| --- | --- | --- | --- |
| 13 | 2.5 | 3.0 | 1.2.7 |
| 14 | 2.6 | 3.1 | 1.2.7 |
| 15 | 2.7 | 3.2 | 1.2.8 |
| 26 (Tahoe; reports as 16 on old SDKs) | 2.8 | 4.0 | 1.2.9 |
| 27 | 2.9 | 4.1 | 1.2.9 |

Rules:

* Emit the *host's* ceiling, like the offline compiler does with `-mmacosx-version-min`: the goal
  is to run on the machine doing the compiling, and both the Metal runtime and the ORC-less
  loader are backward compatible.
* Raising MSL raises the AIR floor (`air_floor(metal)` in Metal.jl `src/version.jl:113-127`):
  Metal 4.0 needs AIR ≥ 2.8. Keep AIR and MSL consistent with this table and with the triple.
* All AIR 2.8 features the spike uses (atomics with the `flags`-less signature, `air.simd_*`,
  `air.wg.barrier`) are present since AIR 2.5–2.7, so a compiler that targets macOS 13+ can emit
  the same intrinsic set for all of them; only the version *numbers* change.
* Probing at run time: `NSProcessInfo.operatingSystemVersion` (Objective-C) gives the host
  version; `std.Target`/the build system gives the deployment target. Apple's own rule for
  compatibility versions (macOS 26 reporting as 16 when built against an old SDK) needs the same
  normalization Metal.jl applies.

## 4. Bitcode downgrade

AIR predates opaque pointers: Apple's loader is an LLVM-14-era reader. The compiler therefore
emits modern bitcode and rewrites it into the LLVM 14 format — with typed pointers, and with
element types recovered for every pointer that needs one.

**Tool**: [llvm-downgrade](https://github.com/JuliaLLVM/llvm-downgrade) — LLVM's own
`BitcodeWriter`/`ValueEnumerator` from the 5.0, 7.0, 14.0, 15.0 and 18.1 releases, ported to
LLVM 23's C++ API, plus `ModuleRewriter*` passes that undo the opaque-pointer migration and a
C API. License: Apache-2.0 WITH LLVM-exception (the writers are LLVM-derived); the tool's own
driver is under the same terms. Zig++ must keep `LICENSE.TXT` with the vendored sources.

**Version**: `14.0`. Lower versions (5.0/7.0) cannot carry `bfloat` and have more restrictions;
15.0/18.0 exist but 14.0 is what Metal.jl ships in production ("Metal's metallib loader is a
backward-compatible reader that accepts real LLVM <= 15 bitcode", `gpucompiler/src/metal.jl:918`).

**C API** (`include/llvm-downgrade.h`): bitcode in, bitcode out.

```c
LLVMDGMemoryBufferRef out;
char *message;
if (LLVMDGDowngrade(data, len, 14, /*minor*/ 0, &out, &message)) { ... }
/* LLVMDGGetBufferStart(out), LLVMDGGetBufferSize(out), LLVMDGDisposeMemoryBuffer(out) */
```

Calls are serialized; buffers are independent and caller-owned.

**Typed pointer recovery**: the rewriter walks loads/stores/GEPs/allocas/atomics for pointee
types, reads `byval`/`sret` parameter attributes, and understands the standard LLVM intrinsics.
For *opaque* pointers passed to `air.*` functions whose element type cannot be inferred, the
compiler must attach the `!arg_eltypes` metadata to the *intrinsic declaration*:

```llvm
declare void @air.atomic.global.add.u.i32(ptr, i32, i32, i32, i1)
!arg_eltypes = !{!{i32 0, i32 0}}     ; parameter 0 is a pointer to i32
```

i.e. alternating `i32` parameter index and a null constant *of the pointee type* (the downgrader
reads the type, not the value). Metal.jl writes these for its big intrinsics
(`src/compiler/compilation.jl:449-475`); the spike's modules need them only where a pointer
argument has no load/store/GEP to infer from.

**Vendoring plan**: keep the fork unmodified under `lib/llvm-downgrade/` (or
`src/llvm-downgrade/` if it must be part of the compiler tree), build it with the same LLVM 23
sources Zig++ already builds (`build.zig` gains a static library target compiled by the C++
toolchain the LLVM build uses, so it links against LLVM's static component libraries), and link
it into `zig` itself. Then `Compilation.zig` calls it in-process, exactly where the object file
is produced. Alternatively, when Zig++ is built with `-Dllvm=system`, the library can be built
by CMake out of tree and linked in the same way.

Two build facts measured in the spike:

* The vendored sources call `Value::dump()` inside `#ifndef NDEBUG`, and a release LLVM has no
  definition of it: the library **must** be compiled with `-DNDEBUG` (or the LLVM build must
  enable `LLVM_ENABLE_DUMP`).
* Linking LLVM's static archives into a shared `libllvm_downgrade` needs those archives built
  with PIC. Zig++'s own LLVM build can simply link the writers into the compiler statically, which
  sidesteps the problem (`-DLLVMDG_BUILD_LIBRARY=OFF` in the spike).

## 5. `.metallib`

The container is small, fully reverse-engineered, and written by the *compiler*, not by the host:
the host hands the bytes to `newLibraryWithData:`.

Header, 88 bytes, little-endian:

| offset | size | field |
| --- | --- | --- |
| 0x00 | 4 | `"MTLB"` |
| 0x04 | 6 | file version: `u16 major` (bit 15 = macOS target), `u16 minor`, `u16 patch` |
| 0x0a | 1 | file type (bit 7 = stub; 0 = executable) — 4 bytes at 0x0a–0x0d in total |
| 0x0b | 1 | platform type (bit 7 = 64-bit; 1 = macOS) |
| 0x0c | 4 | platform version `u16 major`, `u8 minor`, `u8 patch` |
| 0x10 | 8 | file size |
| 0x18 | 16 | function list `(u64 offset, u64 size)` |
| 0x28 | 16 | public metadata `(offset, size)` |
| 0x38 | 16 | private metadata `(offset, size)` |
| 0x48 | 16 | module list `(offset, size)` |

Sections are concatenated without padding: function list at 88 (`u32 count`, then one *tag group*
per function), optional header-extension group, public metadata, private metadata, module bytes.

Tag group: `u32 size` (counts itself) then records `<4-byte tag><u16 value_size><value>`, closed
by the 4-byte token `ENDT`; `size = 4 + Σ records + 4`. A group with no tags is 8 bytes
(`u32 8` + `ENDT`).

Per-function tags, in the order Apple writes them: `NAME` (NUL-terminated), `TYPE` (`u8`, 2 =
kernel), `HASH` (SHA-256 of the stored module bytes), `OFFT` (3×`u64`: public, private and module
offsets, relative to their sections), `VERS` (4×`u16`: AIR major, AIR minor, MSL major, MSL
minor), `MDSZ` (`u64` module length), and optionally `SOFF`/`RFLT`. Module bytes are stored
verbatim; `MDSZ` segments them.

Header-extension tags (no size prefix, ended by `ENDT`): `HSRD`/`HSRC` (embedded sources),
`HDYN` (library name), `RLST` (reflection list), `SLST` (script list), `UUID`. A minimal
library needs none of them.

Writer design for the compiler (`src/metallib.zig`, no host dependencies):

1. hash the module bytes with SHA-256;
2. build the function group and the function list;
3. build an optional `UUID` tag — Apple's toolchain stamps a random v4 UUID, Metal.jl derives it
   from the module hash so output is reproducible; deriving is better for caching;
4. patch the four section offset/size pairs and the file size, then write, all fields explicitly
   little-endian (`std.mem.writeInt(..., .little)`).

Measured, and load-bearing:

* **Store the bitcode raw.** Apple's toolchain wraps a module in a 20-byte section header
  (`0b17c0de`, `u32 0`, `u32 0x14`, `u32 bitcode_size`, `i32 -1`) and stores that inside the
  metallib. A metallib built that way from *downgraded* bitcode failed to compile on the M4
  (`XPC_ERROR_CONNECTION_INTERRUPTED` from `newComputePipelineStateWithFunction:`, twice); the
  same bitcode stored raw loaded and ran. Metal.jl's writer also stores it raw. `MDSZ` is then
  just the bitcode length.
* Apple's own files insert an extra `04 00 00 00` after some groups (the next section's size
  field, left over from its writer). Omit it; our layout works.
* The metallib file version must not exceed what the host supports (§3); the runtime tolerates
  older versions, so emitting the *host's* version is right.
* No compression is involved anywhere in the container (the only compressed payloads are optional
  embedded sources, which are bzip2 tarballs).

## 6. Fitting the target into the compiler

### 6.1 `std.Target`

A new CPU architecture `air64` (Apple's own triple arch name), with:

* `isGPU() == true` (it participates in the GPU paths: no libc, kernels are exported functions,
  compiler-rt is bundled);
* 64-bit pointers, little-endian, no CPU features;
* OS `macos` (deployment target drives the version tables in §3);
* a new object format (`.metallib`) so `link.zig` does not try to hand it to a system linker;
* calling convention: the AIR-level kernel ABI is the C calling convention. Zig's `callconv(.kernel)`
  should map to a new `lang.CallingConvention` tag (e.g. `.metal_kernel`) whose LLVM calling
  convention is `ccc`, and `Target.cCallingConvention()` for `air64` should return the same, so
  that `extern`/`export` defaults and `callconv(.c)` are consistent.

Address spaces go in `llvmAddrSpaceInfo` (`src/codegen/llvm.zig:4465-4550`):

```zig
.air64 => &.{
    .{ .zig = .generic,  .llvm = 0 },   // flat / function-local
    .{ .zig = .global,   .llvm = 1 },   // device
    .{ .zig = .constant, .llvm = 2 },   // constant buffers
    .{ .zig = .shared,   .llvm = 3 },   // threadgroup
    .{ .zig = .local,    .llvm = 4 },   // per-thread; see note
},
```

`std.lang.AddressSpace` already has the vendor-neutral names (`global`, `constant`, `shared`,
`local`), so `std.gpu` can write `*addrspace(.global) f32` and have it mean device memory on
every GPU backend — on CUDA `.global` is AS 1, on HIP AS 1, on Metal AS 1. `.local` is the one
uncertain entry: Apple's frontend leaves function-local memory in AS 0 in every sample
inspected, while Metal.jl's TTI documents AS 4 as "thread". Either mapping works for
`alloca`-free kernels; keep it in one place so it can be flipped after testing an `alloca`-heavy
kernel.

### 6.2 Emitter

`src/codegen/llvm.zig` already has the pieces:

* triple/data layout come from `lib/std/zig/llvm/Builder.zig` (`triple()` and `datalayout()`);
  add the `air64` cases there, including the `air64_v<major><minor>` arch string and the
  `-apple-macosx<version>` OS component;
* kernel functions are exported functions with `callconv(.kernel)`; the module must mark them
  (module-level `!air.kernel`) and must **not** attach NVPTX/AMDGPU metadata;
* the `@builtin.target` / `@Target.<arch>.*` globals the backend emits for runtime target
  queries must be dropped for kernel modules (they are dead data, but they would bloat the
  module and can carry host-only types).

New work, in one place if possible:

1. **Kernel-argument discovery.** AIR needs the builtins a kernel actually uses to be appended
   parameters, in a fixed order, and the metadata to list every parameter. The natural
   implementation mirrors GPUCompiler: lower `@workItemId`/`@workGroupId`/`@workGroupSize`/
   `@workGroupBarrier` to calls to placeholder functions while building the kernel body, then run
   a module pass over each kernel that (a) collects the distinct builtins used, (b) rebuilds the
   kernel signature as `(buffers..., builtins...)`, (c) replaces placeholder calls with the new
   parameters, and (d) emits `!air.kernel` with the argument metadata.
   Buffer arguments come from the *declaration*: parameters whose type is a pointer in AS 1/2/3,
   with `air.location_index` assigned in parameter order and `air.read`/`air.write`/`air.read_write`
   derived from the pointer's `const`/mutable-ness (a `const` pointer is `air.read`; the spike
   takes this from an explicit spec until the type-based rule is implemented).
2. **Intrinsic lowering.** `std.gpu`'s device API already lowers to `llvm.nvvm.*`/`llvm.amdgcn.*`
   in `lib/std/gpu.zig`; for `air64` it must lower to the `air.*` names in §2.6. Barriers and
   builtins are arguments, not calls, so `syncThreads()` becomes `air.wg.barrier(2, 1)` and
   `threadIdx(dim)` becomes a read of the corresponding appended parameter. Atomics:
   `@atomicRmw`/`@cmpxchg*` must lower to `air.atomic.*` (AIR does not accept LLVM's `atomicrmw`
   in kernels — Apple's compiler never emits it). Zig's atomic *ordering* has no AIR equivalent
   beyond relaxed; `std.gpu`'s atomics are documented as relaxed/device-scope, which is exactly
   `(0, 2)`.
3. **compiler-rt.** Kernels must carry compiler-rt as on CUDA/HIP, compiled for `air64` with
   `callconv(.kernel)`/`.c` — and the device-specific parts (no `__assertfail`, no `vprintf`)
   need Metal equivalents (`air.os_log`) or must be omitted.
4. **Output.** With `air64`, the "object file" is the metallib, produced in-process:
   `emit_llvm_bc → AIR rewriter → llvm-downgrade → metallib writer`. `-femit-bin` and the
   linker stage are skipped (no system linker is involved); `--verbose-air` and `--verbose-metallib`
   should dump the intermediate bitcode for debugging, mirroring `--verbose-llvm-bc`.

### 6.3 What the emitter does not need

* no target machine (Apple's backend does instruction selection); GPUCompiler reports
  `llvm_machine(...) = nothing` for the same reason;
* no TBAA, no alias scopes, no debug info;
* no `.air` text output — bitcode is what the reader takes.

## 7. `std.gpu` device API mapping

`std.gpu` is already written against vendor-neutral names, so the Metal target mostly supplies a
new arm in each `switch (arch)`. The mapping:

| `std.gpu` | AIR |
| --- | --- |
| `threadIdx(.x/.y/.z)` | `air.thread_position_in_threadgroup` (component of `uint`/`uint2`/`uint3`) |
| `blockIdx(.x/.y/.z)` | `air.threadgroup_position_in_grid` |
| `blockDim(.x/.y/.z)` | `air.threads_per_threadgroup` |
| `globalId(.x)` | `air.thread_position_in_grid` |
| `gridDim(.x)` | `air.threadgroups_per_grid` |
| `syncThreads()` | `air.wg.barrier(2, 1)` (`mem_threadgroup`) |
| `laneId()` | `air.thread_index_in_simdgroup` |
| `warp_size` | 32 (Apple SIMD width) |
| `shflDown/Up/Xor/Broadcast` | `air.simd_shuffle_down/_up/_xor.f32`, `air.simd_broadcast.f32` — lane index is `i16` |
| `all/any/uniform/ballot/popcount` | `air.simd_all`, `air.simd_any`, (uniform = all^any of the ballot), `air.simd_ballot` |
| `warpReduceSum/Max/Min` | `air.simd_sum/_max/_min.<type>` (`air.simd_prefix_*` for scans) |
| `atomicAdd/Sub/Min/Max/Exchange/CAS` | `air.atomic.{global,local}.{add,sub,min,max,xchg,cmpxchg}.*`, order 0, scope 2 (device) or 1 (threadgroup) |
| `fast.sin/cos/...` | `air.fast_sin`, `air.fast_cos`, `air.fast_sqrt`, … |
| `print` | `air.os_log` (Metal 2.3+); no `vprintf` |
| `assertFail` | no `__assertfail` on Metal — needs `air.os_log` + an abort path; *unverified* |
| FMA | `air.fma.f32` (Apple GPUs have FMA) |

Threadgroup memory is *not* part of `std.gpu`'s API today; on Metal it should become a
`std.gpu` construct that lowers to an `addrspace(3)` global, because MRR/DRAM-backed device
buffers cannot substitute for it.

## 8. `std.gpu.metal` host API sketch

Shaped like `std.gpu.cuda`/`hip`: a struct-of-library-handles loaded at run time, no SDK, no
Objective-C compiler. The verified call sequence (§12) becomes:

```zig
pub const metal = struct {
    // dlopen: /usr/lib/libobjc.A.dylib, /usr/lib/libSystem.B.dylib,
    //         /System/Library/Frameworks/Metal.framework/Metal
    pub fn load() !void                       // resolves classes, selectors, objc_msgSend
    pub const Device = struct { id: objc_id, name: []const u8, ... };
    pub fn devices(allocator) ![]Device       // MTLCopyAllDevices / MTLCreateSystemDefaultDevice
    pub const Buffer = struct { id: objc_id, bytes: []u8 };   // MTLResourceStorageModeShared = unified memory
    pub fn createBuffer(len: usize) !Buffer
    pub fn destroyBuffer(buf: Buffer) void
    pub const Library = struct { id: objc_id };
    pub fn loadLibrary(bytes: []const u8) !Library            // newLibraryWithData: via dispatch_data_create
    pub const Pipeline = struct { id: objc_id };
    pub fn createPipeline(lib: Library, name: []const u8) !Pipeline
    pub fn launch(p: Pipeline, grid: [3]u32, block: [3]u32, args: anytype) !void
    pub fn synchronize() void

    // host-side helpers the driver needs
    fn dispatchData(bytes: []const u8) objc_id                // dispatch_data_create(ptr, len, NULL, NULL)
    fn nsString(s: []const u8) objc_id                        // +[NSString stringWithUTF8String:]
    fn errorMessage(err: objc_id) []const u8                  // -localizedDescription
};
```

Details that the spike established and that a naive binding gets wrong:

* `objc_msgSend` must be called through an exactly-typed function pointer per call site; there is
  no variadic `objc_msgSend` on arm64.
* `newLibraryWithData:` takes a `dispatch_data_t`; `dispatch_data_create` with a null queue and
  null destructor copies the bytes. `newLibraryWithURL:` is the alternative and needs Foundation.
* every `error:`-taking selector must be given an `NSError**`; without it, failures are silent
  nulls.
* buffers live in `MTLResourceStorageModeShared` (option 0): the CPU and GPU see the same
  memory, so results are read from `buffer.contents` after `waitUntilCompleted`, with no blit.
* `dispatchThreads:threadsPerThreadgroup:` takes two by-value `MTLSize` structs (3 × `u64`), and
  it is the driver — not the host — that materializes the AIR builtin parameters.
* launches are asynchronous; `waitUntilCompleted` (or a completion handler) is the
  synchronization primitive, mirroring `std.gpu.cuda`'s stream synchronize.

## 9. Limitations

* **No `f64`.** Apple GPUs have no double-precision arithmetic and reject `Float64` operations
  in kernels. Zig must reject or downgrade them at compile time for `air64`; `std.fmt.parseFloat(f32, …)`
  is usable because its f32 path is f32/u64-integer code (measured: zero `double` *instructions*
  in the compiled module).
* **No `i128`** (`@i128` arithmetic is not available on Apple GPUs; MSL has no 128-bit type).
* **Atomics**: 32-bit widths verified; 64-bit and float atomics depend on MSL 3.1+/Metal 4 and
  are *unverified* on this target. Orderings stronger than relaxed are not expressible through
  the MSL frontend before MSL 4.1, and `std.gpu`'s atomics are relaxed by definition.
* **Threadgroup memory** is static per kernel (an `addrspace(3)` global). The budget is 32 KiB on
  current Apple GPUs (16 KiB on some older families); the CI runner's paravirtual device reports
  32 KiB.
* **SIMD width 32**: `warp_size` is 32, and every SIMD-group operation assumes full groups.
  Threadgroup sizes that are not multiples of 32 still work (Metal's own reductions handle
  partial groups), but a `std.gpu`-written reduction must not assume 32 lanes per group blindly.
* **Address-space friction in `std`.** Ordinary Zig library code takes `[]const u8`
  (generic address space). Device memory is AS 1, which cannot implicitly convert to generic, so
  a kernel cannot hand a device slice to `std.fmt.parseFloat` directly: the spike stages the
  digits through a thread-local buffer first. The permanent options are (a) an address-space
  parameter on the affected `std` functions or a `std.gpu` slice type, or (b) `std` functions
  that take a generic-view pointer. This is a std-library task, not a backend one.
* **No host-side compiler.** Everything above runs at Zig++ compile time; the Mac only needs the
  Metal *runtime* (present in macOS), not the Metal toolchain — which matters because Xcode 26
  ships the Metal compiler as a separate downloadable component and Zig++ must not require it.
* **No `printf`/`assert` parity yet**: `air.os_log` exists, `__assertfail` does not.

## 10. Test plan

1. **Spike-level (this branch)**: three kernels — vector add; a 256-thread reduction with
   threadgroup memory, a barrier, a SIMD-group reduction and a device atomic; and a kernel that
   calls `std.fmt.parseFloat(f32, …)` — compiled by Zig++ for a stand-in target, rewritten to AIR,
   downgraded, wrapped, loaded through the Objective-C runtime and compared against the CPU.
2. **Compiler-level**: `zig build-obj -target air64-macos` emits a `.metallib`; the same three
   kernels then run with no spike tooling. CI checks that do not need a Mac:
   `llvm-downgrade` round-trip, `llvm-dis` metadata diff against the golden AIR shapes in this
   document, and a container lint (offsets, hashes, versions) of the emitted metallib.
3. **`macos-15` runner**: GitHub's `macos-15` and `macos-latest` runners expose a paravirtual
   Metal device ("Apple M1 (Virtual)", GPU family `mac2`, SIMD width 32, 32 KiB threadgroup
   memory) — enough for correctness, not for performance. `macos-14` has no Metal device.
4. **M4 hardware**: the full suite, including timing and the hardware-only features
   (`mac2` is absent there, so family-gated code paths are covered by both machines).
5. **`test/standalone/gpu`**: the existing suite (CUDA/HIP) gains a Metal backend in the same
   shape — per-backend build steps and a host binary — so the same kernels are exercised on
   Metal, CUDA and HIP from one source.

## 11. Staged implementation

1. **Spike** (this branch): prove AIR → downgrade → metallib → run, and pin the conventions in
   this document. *Done as far as §12 reports.*
2. **Target skeleton**: `air64` arch in `std.Target`, triple/data layout in `Builder.zig`,
   address spaces, calling convention, `.metallib` object format, and a `-femit-metallib` path
   that runs the downgrader and the writer on `--verbose-llvm-bc` output. At that point the
   spike's `air-rewrite` can be retired: its rewrite becomes a compiler pass.
3. **Kernel ABI pass**: builtins as arguments, `!air.kernel`, buffer metadata, `convergent`,
   and the removal of host-only globals.
4. **Device API**: `std.gpu` lowers to `air.*` for `air64`; compiler-rt included; `std.gpu`
   threadgroup-memory construct.
5. **Host API**: `std.gpu.metal` with the verified ObjC recipe; `test/standalone/gpu` gains the
   Metal arm.
6. **CI**: `macos-15` job plus the M4 for the full suite.

## 12. Spike results

*Everything in this section was executed; commands and outputs are in the branch's
`test/standalone/metal_spike/README.md`.*

Proven on the M4 (macOS 26.6.2):

| stage | what ran | result |
| --- | --- | --- |
| Apple MSL → AIR | `xcrun metal -c -O2 vadd.metal` | bitcode, AIR 2.8 / Metal 4.0 |
| AIR → downgrade | `llvm-downgrade --bitcode-version=14.0` | Apple's `metallib` accepted it |
| downgrade → container | `xcrun metallib` | loaded and ran |
| container → GPU | `newLibraryWithData:` via `objc_msgSend` | `vadd 0/4096 mismatches` |
| reduction | Apple's MSL reduction kernel | `reduce` threadgroup sums correct, counter = 4 |
| our own container | the spike's writer, raw module bytes | `vadd 0/4096 mismatches` |
| Zig++ kernels | `kernels.zig` → `air-rewrite` → `llvm-downgrade` → metallib writer | §12.1 |

Measured negative results worth knowing:

* a metallib whose module bytes are wrapped in Apple's `0b17c0de` section header (as Apple's own
  tool writes them) *failed* to compile once produced by our writer
  (`XPC_ERROR_CONNECTION_INTERRUPTED`), while the raw bitcode of the same module worked;
* `xcrun metallib` accepts a downgraded module that `xcrun metal` never saw, so the AIR
  conventions — not the frontend — are what the container and the runtime check;
* LLVM 23's `llvm-dis` reads the downgraded file but shows opaque pointers (the reader upgrades
  them); the typed-pointer form is what is on disk, in the LLVM 14 format;
* `xcrun air-validate` does not accept a raw AIR module file (it expects a Mach-O/AIR binary),
  so it is not a validation tool for emitted bitcode;
* Zig++'s `ZIG_LIB_DIR` must point at a lib tree matching the compiler binary's source; using
  the master lib with an `amdgpu`-tree compiler produces bogus `std.lang is corrupt` panics and
  wrong calling-convention diagnostics.

### 12.1 Zig++ kernels on the M4

*TBD: filled in from the spike run (see the branch README for the exact commands).*

## 13. Reproducing the spike

Linux side:

```sh
# 1. kernels -> modern bitcode (stand-in target, address spaces already AIR's)
ZIG_LIB_DIR=/home/autark/src/zig/zig-amdgpu/lib /tmp/zig-amdgpu-final/bin/zig build-obj \
    test/standalone/metal_spike/kernels.zig -target nvptx64-cuda -fno-compiler-rt -OReleaseFast \
    -fno-emit-bin -femit-llvm-bc=kernels.bc

# 2. modern bitcode -> AIR conventions (triple, layout, metadata, attributes)
air-rewrite kernels.bc -o kernels.air.bc --spec kernels.spec

# 3. AIR -> LLVM 14 bitcode for Apple's reader
llvm-downgrade kernels.air.bc --bitcode-version=14.0 -o kernels.air14.bc

# 4. bitcode -> .metallib
metallib write --bitcode kernels.air14.bc --name vadd --air 2.8 --metal 4.0 \
    --format 1.2.9 --platform 26.0.0 -o vadd.metallib
```

macOS side:

```sh
# host binary (built on Linux, signed ad hoc by Zig's Mach-O linker)
ZIG_LIB_DIR=… zig build-exe host.zig -target aarch64-macos -femit-bin=host

./host vadd.metallib       # runs the three kernels, compares against the CPU
```

Reference generation on macOS (needs the Metal Toolchain component):

```sh
xcodebuild -downloadComponent MetalToolchain      # 688 MB, one time
xcrun metal -c -O2 kernel.metal -o kernel.air     # Apple's AIR, for comparison
xcrun metal -S -emit-llvm kernel.metal            # textual AIR, typed pointers
xcrun metallib kernel.air -o kernel.metallib
```

## 14. Open questions

1. Does Apple's backend accept `alloca`/stack traffic and outlined `sret` functions in kernels?
   The spike's `std.fmt.parseFloat` kernel has both (Zig outlines parse helpers).
2. Does it accept the dead `f64` **type** in `parseFloat`'s slow path (`BiasedFp(f64)` as an
   `sret` type), or must the compiler prune value types the device cannot represent?
3. Are `SDK Version`, `air.max_*` and `w_frame-pointer` module flags required, or merely
   informative? The spike emits them; a negative test was not run.
4. Do `air.*` declarations need Apple's exact parameter attributes (`captures(none)`), or is the
   name plus the LLVM type enough?
5. What exactly does the runtime do with `air.arg_name`/`air.arg_type_name` — reflection only, or
   argument binding?
6. Threadgroup pointer width: AIR 2.8's layout says 64-bit for AS 3, but older AIR versions used
   32-bit threadgroup pointers; the target must not assume either across versions.
7. Whether `metallib` file version 1.2.9 is accepted by a macOS 15 runner (the spike only had an
   M4 on 26.6); the per-host table in §3 is the safe rule.
