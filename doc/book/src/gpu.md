# GPU Programming

`std.gpu` runs Zig++ on NVIDIA, AMD and Apple GPUs. This chapter is written
from the documentation comments in `lib/std/gpu.zig`, `lib/std/gpu/*.zig`, and
from the end-to-end test of everything described here,
[test/standalone/gpu](https://github.com/mattneel/zigpp/tree/master/test/standalone/gpu).

```zig
// kernels.zig
const gpu = @import("std").gpu;

export fn wave(data: [*]f32, amplitude: f32, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i < n) data[i] = amplitude * @sin(data[i]);
}
```

An exported function with the `.kernel` calling convention is a kernel. Kernels
can use the rest of the standard library as long as they avoid operating system
services: `std.fmt`, `std.json`, `std.mem`, `std.base64`, hash maps, array
lists, and the device allocators all work on the GPU, up to the limits that a
target has (see "What is not there yet" at the end of this chapter).

## The device API

The device-side functions in `std.gpu` are implemented for NVPTX, AMDGPU and
air64, the target of Apple GPUs. The indexing functions and `syncThreads` use
builtins that also exist for SPIR-V.

### Indexing

```zig
pub const Dim = enum(u2) { x, y, z };

pub inline fn threadIdx(comptime dim: Dim) u32
pub inline fn blockIdx(comptime dim: Dim) u32
pub inline fn blockDim(comptime dim: Dim) u32
pub inline fn gridDim(comptime dim: Dim) u32
pub inline fn globalId(comptime dim: Dim) u32
```

These are CUDA's indexing functions. `globalId(dim)` is
`blockIdx(dim) * blockDim(dim) + threadIdx(dim)`: the index of the calling
thread within the whole grid. `threadIdx`, `blockIdx`, and `blockDim` are
`@workItemId`, `@workGroupId`, and `@workGroupSize`. On Apple GPUs the hardware
provides `gridDim` and `globalId` directly, as the `air.threadgroups_per_grid`
and `air.thread_position_in_grid` builtins, one component per dimension that
the kernel asks for.

### Synchronization

```zig
pub inline fn syncThreads() void
```

Waits until every thread of the block has reached this call, and makes the
memory writes that each thread made before the call visible to the others.
Every thread of the block must reach the same call; anything else is undefined
behavior. It is `@workGroupBarrier()`.

### Warps

```zig
pub const warp_size: comptime_int
pub const WarpMask = @Int(.unsigned, warp_size);
pub inline fn laneId() u32
```

`warp_size` is 32, except on AMD: 64 before GFX10, and 32 from GFX10 unless
`wavefrontsize64` or `wavefrontsize32` selects otherwise. `laneId()` is the
index of the calling thread within its warp, from 0 to `warp_size - 1`.

An Apple GPU runs a thread in a SIMD group of 32 lanes, which is what a warp is
there: `warp_size` is 32, and `laneId()` is the thread's position in its SIMD
group. Apple's ballot is a 64-bit word with a bit per lane, of which the lanes
of the calling group are the low 32 bits, and that is what `ballot` returns.

```zig
pub inline fn shflDown(value: anytype, delta: u32) @TypeOf(value)
pub inline fn shflUp(value: anytype, delta: u32) @TypeOf(value)
pub inline fn shflXor(value: anytype, lane_mask: u32) @TypeOf(value)
pub inline fn shflBroadcast(value: anytype, src_lane: u32) @TypeOf(value)
```

Warp shuffles. Every thread of the warp that has not exited must call them
together, and a value read from a thread that has exited is undefined. `shflUp`
and `shflDown` read `delta` lanes below or above, and yield the caller's own
value when that lane is outside the warp; `shflXor` reads `laneId() ^
lane_mask`; `shflBroadcast` reads `src_lane` modulo `warp_size`. Shuffles
support values of at most 64 bits.

```zig
pub inline fn all(predicate: bool) bool
pub inline fn any(predicate: bool) bool
pub inline fn uniform(predicate: bool) bool
pub inline fn ballot(predicate: bool) WarpMask
pub inline fn popcount(predicate: bool) u32

pub fn warpReduceSum(value: anytype) @TypeOf(value)
pub fn warpReduceMax(value: anytype) @TypeOf(value)
pub fn warpReduceMin(value: anytype) @TypeOf(value)
```

Votes and reductions. `all` and `any` are true when every, or any, lane's
predicate is true; `uniform` is true when the predicate has the same value for
every lane; `ballot` has a set bit for each true lane, and `popcount` counts
them. The reductions return the result to every lane, and integer overflow is
checked like `+`, except on Apple GPUs: there each reduction of an `f32`, `i32`
or `u32` is one hardware instruction for the whole SIMD group, and the integer
form of the sum wraps rather than reporting an overflow the way the checked `+`
of the other targets does.

### Atomics

```zig
pub inline fn atomicAdd(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*)
pub inline fn atomicExchange(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*)
pub inline fn atomicCAS(ptr: anytype, expected: @TypeOf(ptr.*), new_value: @TypeOf(ptr.*)) @TypeOf(ptr.*)
pub inline fn atomicMin(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*)
pub inline fn atomicMax(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*)
```

Each returns the value the location held before the operation. They use relaxed
ordering, visible to the whole device, and `ptr` may be a global, shared, or
generic pointer. For other orderings, use `@atomicRmw` and `@cmpxchgStrong`.

On Apple GPUs, the operations are exactly the ones the Metal language has: only
the relaxed orderings, `.monotonic` and `.unordered`, exist, and a stronger one
is a compile error. The compare-exchange is the weak form that AIR has, which
`@cmpxchgStrong` and `@cmpxchgWeak` both lower to. An atomic through a pointer
without an address space, which is a pointer into function-local memory, is a
compile error as well: give it `addrspace(.global)` or `addrspace(.shared)`.

### Fast math

```zig
pub const fast = struct {
    pub inline fn sin(x: f32) f32
    pub inline fn cos(x: f32) f32
    pub inline fn sqrt(x: f32) f32
    pub inline fn fma(a: f32, b: f32, c: f32) f32
};
```

Hardware approximations: PTX `sin.approx.f32`, `cos.approx.f32`, and
`sqrt.approx.f32`, AMD `v_sin_f32` and `v_cos_f32`, or Apple's `air.fast_sin`,
`air.fast_cos`, and `air.fast_sqrt`. They are faster than `@sin`, `@cos`, and
`@sqrt`, and their error is bounded in absolute terms rather than relative to
the input. AMD's square root is not one of them: its instruction is accurate to
one unit in the last place, so `fast.sqrt` is a compile error there.

`fast.fma(a, b, c)` is `a * b + c` with a single rounding, the fused
multiply-add that each of the three GPUs has an instruction for. It is the one
function of the namespace whose result is exact: `air.fma.f32` on Apple GPUs,
and `@mulAdd` on the others, which is LLVM's `llvm.fma.f32`, `fma.rn.f32` on
PTX and `v_fma_f32` on AMD.

### Printing and panics

```zig
pub fn print(comptime fmt: []const u8, args: anytype) void
pub fn assertFail(message: []const u8) noreturn
```

`print` formats like `std.fmt` and writes to the host's standard output,
truncating at a 256-byte stack buffer. NVIDIA prints at the next host/device
synchronization. AMD writes into a 1 MiB buffer that `hip.Context.loadModule`
connects and `hip.Context.synchronize` drains, dropping output beyond that; a
code object loaded by another host prints nothing.

`assertFail` stops the launch and reports the message with the block and thread
that failed, truncated to 255 bytes. `std.debug.defaultPanic` calls it for
`.cuda` and `.amdhsa` targets, so a panic in a kernel reports its message to
the host. After a failed assertion, CUDA's next `Context.synchronize` returns
`error.Assert` and the context cannot run more kernels, while HIP's next
synchronization writes to stderr, returns `error.Assert`, and leaves the
context usable.

On Apple GPUs, both functions are compile errors, and the error names the
reason: Metal has neither `printf` nor `__assertfail`. The default panic
namespace of an `air64` module is therefore `std.debug.no_panic`: a failed
safety check traps without formatting a message, which stops the launch and
leaves its results unwritten.

### Allocators

```zig
pub const device_heap: std.mem.Allocator

pub fn BumpAllocator(comptime size: usize) type
```

`device_heap` is the CUDA device heap: memory that outlives a block, freed with
`free`. Its default size is 8 MiB, unless the host changes
`cuda.Limit.malloc_heap_size` before the heap is first used. It is implemented
with NVPTX-only syscalls, and is a compile error on other architectures.

`BumpAllocator(size)` carves allocations out of block-shared memory, which is
gone when the block finishes:

```zig
var heap: [16 * 1024]u8 addrspace(.shared) = undefined;

export fn kernel() callconv(.kernel) void {
    var bump = std.gpu.allocators.BumpAllocator(heap.len).init(&heap);
    if (std.gpu.threadIdx(.x) == 0) {
        var list: std.ArrayList(u32) = .empty;
        list.append(bump.allocator(), 42) catch return;
    }
}
```

Every thread of the block must call `init`, with the same buffer, before any
of them allocates: it waits at a barrier for the thread that writes the offset
of the unused memory. `used()` reports how much has been allocated,
allocations may race with each other, and only the most recent allocation can
be returned to the allocator. Buffers are limited to 4 GiB.

For memory used by a single thread, `std.heap.FixedBufferAllocator` needs no
synchronization. All three implement `std.mem.Allocator`, so the containers in
`std` work in kernels.

## Compiling kernels

For NVIDIA, compile the kernel module to PTX with `zig build-obj`, then load
the PTX with `cuda.Context.loadModule`:

```sh
zig build-obj -target nvptx64-cuda -mcpu=sm_75 -O ReleaseFast -fno-emit-bin -femit-asm=kernels.ptx kernels.zig
```

PTX for an older `-mcpu` runs on newer GPUs, because the driver compiles it for
the GPU that loads it.

For AMD, compile it to a code object with `zig build-lib -dynamic`, then load
it with `hip.Context.loadModule`:

```sh
zig build-lib -dynamic -target amdgcn-amdhsa -mcpu=gfx1036 -O ReleaseFast kernels.zig
```

A code object only runs on the architecture that `-mcpu` names;
`hip.Device.archName` reports it, as `gfx1036`, or with features as
`gfx90a:sramecc+:xnack-`, whose compiler spelling is
`-mcpu=gfx90a+sramecc-xnack`. On AMD, the first shared variable can live at
address 0, so index shared variables or `@addrSpaceCast` them to generic
pointers instead of building a shared pointer with `@ptrFromInt(0)`.

For Apple GPUs, compile the kernel module to the `.metallib` container that the
Metal runtime loads, then load the container with `metal.Context.loadModule`:

```sh
zig build-obj -target air64-macos -O ReleaseFast -femit-bin=kernels.metallib kernels.zig
```

The object file of the `air64-macos` target is the library itself, so no
linker, no Metal toolchain, and no macOS SDK are part of the pipeline: the GPU's
driver compiles the AIR of the module when the host builds a pipeline.
`-femit-llvm-bc` writes the downgraded AIR bitcode that goes into the
container, and `-femit-asm` and `-femit-llvm-ir` are errors, because there is
no AIR assembly and no textual IR that this compiler could write.

A Metal library carries the AIR version, the Metal language version, and the
container version of one macOS release, which the deployment target of the
compile selects: macOS 13 gives AIR 2.5, Metal 3.0, and container 1.2.7; 14
gives 2.6, 3.1, and 1.2.7; 15 gives 2.7, 3.2, and 1.2.8; 26 gives 2.8, 4.0, and
1.2.9; and 27 gives 2.9, 4.1, and 1.2.9. Name the release as the version of the
OS in the target, as in `-target air64-macos.26.0`, or as `.os_version_min` in a
`Target.Query`.

An Apple kernel's parameters are the ones a Metal host binds by index: a
pointer into the device, constant, or threadgroup address space is a buffer,
such as `[*]addrspace(.global) const f32`, which the host passes with
`setBuffer:offset:atIndex:`, and a scalar or an aggregate is passed as bytes
with `setBytes:length:atIndex:`, which the compiler loads through a `constant`
pointer. Threadgroup memory is a module-level variable in the `shared` address
space, whose size must be known at compile time, because the runtime takes it
from the pipeline: 32 KiB on current Apple GPUs, 16 KiB on some older families.
Dynamic threadgroup memory is not part of the API. The position of the calling
thread is not a parameter; the dispatch supplies it.

In a build script, compile kernels with `b.addObject` and embed
`getEmittedAsm()` for PTX, or `b.addLibrary` with `.linkage = .dynamic` and
embed `getEmittedBin()` for a code object:

```zig
const kernels = b.addObject(.{
    .name = "kernels",
    .root_module = b.createModule(.{
        .root_source_file = b.path("kernels.zig"),
        .target = kernel_target,
        .optimize = .fast,
    }),
});
exe.root_module.addAnonymousImport("kernels.ptx", .{
    .root_source_file = kernels.getEmittedAsm(),
});
```

An Apple container is an emitted binary as well: `b.addObject` with an
`air64-macos` target, whose `getEmittedBin()` is the `.metallib` to embed.

[test/standalone/gpu/build.zig](https://github.com/mattneel/zigpp/blob/master/test/standalone/gpu/build.zig)
builds all three kinds, in a debug and a fast variant: the CUDA and HIP images,
with the AMD code objects for a set of architectures given as `-Damdgpu-arch`
(comma-separated, default `gfx1030`), and the containers of `metal_kernels.zig`
for `air64-macos`, with a macOS 26 deployment target.

## The host APIs

`std.gpu.cuda` is the CUDA driver API and `std.gpu.hip` is the HIP runtime
library. They expose the same names, and a program switches between them by
changing the import. `std.gpu.metal` is the Metal framework, whose objects and
calls are Metal's own rather than a third copy of the other two:

| | `std.gpu.cuda` | `std.gpu.hip` | `std.gpu.metal` |
| --- | --- | --- | --- |
| Loads at run time | `libcuda.so.1` | `libamdhip64.so.7`, `.so.6`, `.so`, or `amdhip64_7.dll`, `_6.dll` | `Metal.framework`, `/usr/lib/libobjc.A.dylib`, `/usr/lib/libSystem.B.dylib` |
| Platform | Linux, with libc | Linux with libc, and Windows without it (`ntdll.LdrLoadDll`) | macOS, with libc |
| Toolkit needed to build | none | none | none |
| Module image | PTX, null-terminated: `[:0]const u8` | a code object for one architecture: `[]const u8` | a `.metallib`: `[]const u8` |
| Allocation failure | `error.OutOfDeviceMemory` | `error.OutOfMemory` | `error.OutOfMemory` |

The CUDA runtime library, NVRTC, events, and graphs are not part of
`std.gpu.cuda`; events, graphs, the memory pools of stream-ordered allocation,
and the rest of the runtime are not part of `std.gpu.hip`; and the rest of the
Metal framework, the Metal Shading Language compiler, and Metal's own runtime
libraries are not part of `std.gpu.metal`, which sends its messages to the
framework's Objective-C objects with `objc_msgSend`. All three require the
`Driver` to stay open, and not to move, while objects that point to it are in
use. The Metal framework is loaded with `dlopen`, which is part of libc, so a
program that uses `std.gpu.metal` links libc and needs neither the macOS SDK
nor an Objective-C compiler.

The CUDA and HIP namespaces export the same `Driver`, `Device`, `Context`,
`Module`, `Function`, `Buffer(T)`, `Stream`, `LaunchConfig`, `DevicePtr`,
`Version`, `ComputeCapability`, `Attribute`, `Limit`, `ModuleOptions`, `Error`,
and `OpenError`, and the sequence for a kernel launch is the same in both:

```zig
var driver = try cuda.Driver.open();
defer driver.close();

const context = try (try driver.device(0)).retainPrimaryContext();
defer context.release();

const module = try context.loadModule(@embedFile("kernels.ptx"), .{});
defer module.unload();

var data: [1000]f32 = undefined;
for (&data, 0..) |*x, i| x.* = @floatFromInt(i);

const buffer = try context.alloc(f32, data.len);
defer buffer.free();
try buffer.copyFromHost(&data);

const wave = try module.function("wave");
try wave.launch(cuda.LaunchConfig.linear(data.len, 256), .{ buffer, @as(f32, 2), @as(u32, data.len) });

try context.synchronize();
try buffer.copyToHost(&data);
```

`Driver.open()` returns `error.DriverNotFound` when the library is absent and
`error.IncompatibleDriver` when it lacks a function Zig++ needs.
`retainPrimaryContext` makes the context current. `Module.function` looks up a
kernel by the name it was exported with, and takes a null-terminated name.

`std.gpu.metal` is the same shape where Metal matches the other two, and its
own where it does not. `Driver.open` reports `error.FrameworkNotFound` on any
system other than macOS, `error.NoDevice` when the framework finds no GPU, and
`error.NotSupported` on an Intel Mac, where the `MTLSize` of a message passes
and returns by other rules than the arm64 ABI that the messages of this
namespace are built for. `Driver.device()` is the GPU of the machine, and
`Device.createContext` makes the context that holds the command queue, because
Metal has no primary context. `Context.loadModule` takes the bytes of a
`.metallib`; `Module.function` finds a kernel by the name it was exported with;
and `Function.pipeline` is where the Metal compiler runs, so it is the call
that reports a kernel the compiler refused, with what it said in
`Options.error_log`. `Pipeline.launch` encodes the dispatch:

```zig
var driver = try metal.Driver.open();
defer driver.close();

const context = try driver.device().createContext();
defer context.release();

const module = try context.loadModule(@embedFile("kernels.metallib"), .{});
defer module.unload();

const wave = try module.function("wave");
defer wave.release();
const pipeline = try wave.pipeline(.{});
defer pipeline.release();

var data: [1000]f32 = undefined;
for (&data, 0..) |*x, i| x.* = @floatFromInt(i);

const buffer = try context.alloc(f32, data.len);
defer buffer.free();
buffer.copyFromHost(&data);

try pipeline.launch(metal.LaunchConfig.linear(data.len, 256), .{ buffer, @as(f32, 2), @as(u32, data.len) });

try context.synchronize();
buffer.copyToHost(&data);
```

A `Buffer` of `std.gpu.metal` is unified memory: `values()` is the memory as a
slice of `T`, the same bytes the kernels read and write, and `copyFromHost`,
`copyToHost`, and `zero` copy and fill it with `@memcpy` and `@memset` rather
than calling into the framework, so no copy of the framework's is between the
host and the memory. A launch binds its arguments in parameter order: a `Buffer`
with `setBuffer:offset:atIndex:`, and anything else with
`setBytes:length:atIndex:`, at the index of the argument, from which the
compiler loads a scalar through a `constant` pointer; the position of the
calling thread comes from the dispatch. A threadgroup of no threads, or one
larger than the kernel allows, is `error.InvalidValue` before anything is
encoded, and a kernel that fails on the GPU is reported by `Context.synchronize`,
which is why the launch is asynchronous.

The errors of the framework are named after what happened rather than after the
numeric code: the codes of the `MTLLibraryErrorDomain` and
`MTLCommandBufferErrorDomain` become `error.InvalidLibrary`,
`error.CompileFailure`, `error.FunctionNotFound`, `error.PageFault`, and the
rest, and a failure of a domain or a code that this version does not know is
reported as the failure of the call. `Options.error_log` receives what the
framework said, as the domain, the code, and the message.

### Launch configuration

```zig
pub const Dim3 = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };
pub const LaunchConfig = struct {
    grid: Dim3 = .{},
    block: Dim3 = .{},
    shared_memory: u32 = 0,
    stream: ?Stream = null,
};
```

`shared_memory` is dynamic shared memory per block, on top of any the kernel
declares statically, and a null `stream` selects the null stream.
`LaunchConfig.linear(n, block_size)` covers `n` threads with blocks of
`block_size`, rounding the grid up; `block_size` must not be zero, and `n == 0`
produces no blocks, which `launch` rejects with `error.InvalidValue`.

`std.gpu.metal`'s `LaunchConfig` has `grid` and `block` only: Metal has no
dynamic shared memory, and a context has one command queue rather than a choice
of streams.

The launch arguments are passed in parameter order, one per kernel parameter:

- a `Buffer(T)` passes the device address it holds;
- a `DevicePtr` (an `enum(u64) { _ }`, a device memory address) passes
  unchanged;
- integers, floats, booleans, enums, vectors, and `extern` or `packed` structs
  pass by value.

A slice or a host pointer as a kernel argument, an untyped compile-time number,
a value with an unsupported layout, or a non-tuple argument list is a compile
error. Scalars need an explicit type: `@as(u32, data.len)`.

`std.gpu.metal` binds the arguments the same way, with one difference: it has no
`DevicePtr`, and every argument that is not a `Buffer` binds its bytes at the
index of the parameter.

```zig
pub const DevicePtr = enum(u64) { _ };

pub fn Buffer(comptime T: type) type
```

`Buffer(T)` has the methods `free`, `copyFromHost`, `copyToHost`, and `zero` in
all three namespaces, and CUDA's and HIP's have the fields `driver`, `ptr`, and
`len`. Copies may be shorter than the buffer, never longer. `Stream` has
`destroy` and `synchronize`, and HIP's stream synchronization does not itself
flush the GPU print and assert output: that happens at `Context.synchronize`.

### Attributes, limits, and error sets

`Attribute` and `Limit` are enums of the vendor's numeric codes: `Attribute`
has the same tags in `std.gpu.cuda` and `std.gpu.hip`, with different numeric
values, so use the tags. `Limit` is `stack_size`, `printf_fifo_size`, and
`malloc_heap_size`, and HIP's runtime reports `error.UnsupportedLimit` for
`printf_fifo_size`, which is not the buffer that `std.gpu.print` uses.

`OpenError` is `error{ DriverNotFound, IncompatibleDriver }` plus `Error`.
Beyond the difference in allocation failure, HIP's set has
`EccNotCorrectable`, `SetOnActiveProcess`, and no PTX-JIT or profiler codes;
CUDA's has `EccUncorrectable`, `PrimaryContextActive`, and the PTX and profiler
codes. Both have `error.Assert`, which is what a kernel assertion turns into at
the next synchronization.

## Running and testing on a GPU

[test/standalone/gpu](https://github.com/mattneel/zigpp/tree/master/test/standalone/gpu)
is the end-to-end test: a port of the examples of the
[ugpu](https://github.com/mattneel/ugpu) project, plus kernels that cover the
rest of `std.gpu`. Its host program computes the expected results on the CPU
and compares them with what the GPU produced, and each of the twenty test
groups — `vector_add`, `reduce`, `histogram`, `warp`, `matrix_mul`,
`convolution`, `stencil`, `stdlib`, `hashmap`, `base64`, `string_search`,
`json`, `dynamic`, `printf`, `hello_gpu`, `builtin_math`, `f128`,
`parse_float`, `device_heap`, and `bump_allocator` — reports its launches,
checks, and failures, and exits non-zero if any check fails.

```sh
cd test/standalone/gpu
zig build test
```

The CUDA variant is built on Linux, and the HIP variant on Linux and Windows.
Each kernel image is built in a debug and a fast variant, and the host program
runs once normally and once with `assert` as an argument, which launches a
kernel that indexes out of bounds and checks that the reported message names
the index, the length, and the failing thread.

The Metal variant is built wherever the suite is. `build.zig` compiles
`metal_kernels.zig` for `air64-macos` with the compiler under test, in a debug
and a fast variant, installs the two containers next to the host program
`gpu_metal_host`, and runs the host on each of them, so the suite exercises the
`air64` target on any host. The host runs a vector add, a reduction whose
partial sums go through threadgroup memory and a device atomic counter, a
kernel with scalar parameters, an index computed with a `usize` multiply (which
a debug build checks for overflow), the 128-bit products and overflow flags of
64-bit multiplies, and tables of integers and of structs in the constant address
space, checking the results against the CPU bit for bit and reporting the SIMD
width of the kernels;
`-Dmetallib=a.metallib,b.metallib` adds the containers that Apple's own `metal`
compiler or the spike built, to compare Apple's code with this one's. A machine
without the Metal framework skips every library it was given, a library without
one of the kernels skips its test, and a run with no library at all skips
everything: all three exit 0. The suite therefore builds its containers and its
host on any host, and only a Mac runs the kernels. On macOS, a framework that
will not open and a library that will not load are failures.

A machine without the driver or without a device is not a failure: the test
reports it and passes, for `error.DriverNotFound`, `error.NoDevice`, a driver
that sees no devices, or an AMD code object that matches no device (it suggests
the `-Damdgpu-arch` value to build for). Everything else fails.

CI runs it as part of the standalone tests:

```sh
zig build test-standalone -Dskip-non-native -Dskip-release
```

The GPU suite compiles its kernels for all three vendors on a runner with no
GPU driver, where the suite then skips itself.

## What is not there yet

- MLIR lowering for tensor cores and kernel fusion.
- Warp shuffles of values wider than 64 bits, and `std.gpu.allocators.device_heap`
  on architectures other than NVPTX, both of which are compile errors.
- On Apple GPUs, `f64` is a compile error, the atomics are the relaxed ones
  only, `print` and `assertFail` are compile errors, and threadgroup memory must
  be a static variable sized by the pipeline's 32 KiB. Metal has no generic
  address space, so a pointer into constant data (a string literal, a table)
  that meets a pointer to thread memory is a compile error naming the function,
  and a checked multiplication of `i128` values is supported only where both
  operands fit in 64 bits.
- On Apple GPUs, a program-scope constant that holds pointers, such as a table
  of strings or slices, is a compile error, because Apple's toolchain does not
  relocate the addresses inside constant data. `std.fmt.parseFloat` has such a
  table, so it does not compile for Apple GPUs yet
  ([#22](https://github.com/mattneel/zigpp/issues/22)).
- The AIR version, the Metal language version, and the container version of a
  Metal library are the ones of the macOS release row that the deployment target
  selects, not a choice of the program.
- On HIP, `ModuleOptions.error_log` is ignored, while CUDA's PTX JIT fills it in
  when a module fails to load.
