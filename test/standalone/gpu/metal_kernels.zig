//! The kernels of `metal_kernels.metal` in Zig, for the compiler under test to compile to the
//! `.metallib` that `metal_host.zig` runs on the GPU of a Mac through `std.gpu.metal`: the vector
//! add, the reduction and the kernel with scalar parameters, with the same names, the same
//! arguments in the same order, and the same binding of every argument.
//!
//! ```sh
//! zig build-obj -target air64-macos -O ReleaseFast -femit-bin=vadd.metallib metal_kernels.zig
//! ```
//!
//! The object file of the `air64` target *is* the library: no linker, no Metal toolchain and no
//! SDK are involved, because the driver compiles the AIR the module holds when the host builds the
//! pipeline. `build.zig` of this directory compiles this file with the compiler that runs the
//! build, in the two optimization modes of the other kernels of the suite, and passes the
//! containers to `metal_host.zig`.
//!
//! The kernels are separate from `kernels.zig` for two reasons. Most of that file needs parts of
//! the standard library that an Apple GPU does not have (f64 and f128 arithmetic, `print`, and
//! therefore the allocators and the parsers that report through it), and an AIR kernel argument
//! must be a pointer into the device or the constant address space, while the kernels of the other
//! vendors take generic pointers.
//!
//! An Apple GPU takes its threadgroup memory from the pipeline, so a kernel that uses it declares
//! it as a static variable in the `shared` address space, and the runtime finds its size when it
//! builds the pipeline.

const std = @import("std");
const gpu = std.gpu;

/// Kernels run on a GPU that has no `printf` and no `__assertfail`, and `std.gpu` makes both of
/// them compile errors. What a kernel that fails its own check needs is a trap, which is what
/// `no_panic` is: it stops the launch and leaves the results unwritten, which the host reports as
/// a mismatch. Without it a Debug build of these kernels would not compile, because the checks
/// that such a build inserts reach for the standard panic handler.
pub const panic = std.debug.no_panic;

/// Threads in a threadgroup of `reduce`: 256, like the block size of the reduction examples of the
/// CUDA and AMD kernels.
const reduce_block_size = 256;

/// The partial sums of `reduce`, one per SIMD group of the threadgroup. Every threadgroup of the
/// grid runs the kernel with its own copy, which the runtime allocates when it builds the
/// pipeline: the size of this array is the `threadgroupMemoryLength` of the dispatch, 32 bytes.
const reduce_simd_groups = reduce_block_size / gpu.warp_size;
var reduce_scratch: [reduce_simd_groups]f32 addrspace(.shared) = undefined;

/// `c[i] = a[i] + b[i]`: one thread per element of the three buffers, which a launch of 4096
/// threads covers exactly, so the kernel needs no length.
///
/// Every parameter is a pointer into device memory, which the host binds as a buffer at the index
/// of the parameter with `setBuffer:offset:atIndex:`; the position of the calling thread comes
/// from the dispatch, not from the host.
export fn vadd(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) f32,
) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    c[i] = a[i] + b[i];
}

/// Sums the elements of every threadgroup of `inbuf` into the element of `out` with the index of
/// that threadgroup, and adds one to `counter` once per threadgroup, so that the host can see that
/// every threadgroup ran.
///
/// The block-wide sum is the three steps of Apple's own reduction kernels: every thread holds the
/// sum of its own element, the SIMD group reduces its 32 lanes with one hardware instruction
/// (`warpReduceSum`, which is `air.simd_sum.f32` on this target), and the thread 0 of the
/// threadgroup adds the partial sums of the groups in threadgroup memory. A launch of 256 threads
/// per threadgroup covers a block exactly, so the kernel needs no length either.
export fn reduce(
    inbuf: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    counter: *addrspace(.global) u32,
) callconv(.kernel) void {
    const tid = gpu.threadIdx(.x);
    const block = gpu.blockIdx(.x);

    const value = inbuf[block * reduce_block_size + tid];
    const group_sum = gpu.warpReduceSum(value);

    if (gpu.laneId() == 0) reduce_scratch[tid / gpu.warp_size] = group_sum;
    gpu.syncThreads();

    if (tid == 0) {
        var total: f32 = 0;
        for (&reduce_scratch) |*partial| total += partial.*;
        out[block] = total;
        _ = gpu.atomicAdd(counter, 1);
    }
}

/// The count of `count` elements of `x` multiplied by `factor`: the kernel that takes scalars, and
/// that `metal_host.zig` runs to check the binding of an argument that is not a pointer.
///
/// A Zig parameter that is not a pointer is a value the host binds as bytes at the index of the
/// parameter, which is where the AIR of the kernel reads it from: the compiler turns such a
/// parameter into a pointer into the `constant` address space and loads it there. In the Metal
/// Shading Language that is `constant float& factor [[buffer(1)]]`, and both bind argument 1 of
/// the dispatch.
export fn scale(
    x: [*]addrspace(.global) f32,
    factor: f32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= count) return;
    x[i] = x[i] * factor;
}
