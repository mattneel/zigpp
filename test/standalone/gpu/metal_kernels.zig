//! The kernels of `metal_kernels.metal` in Zig, for the compiler under test to compile to the
//! `.metallib` that `metal_host.zig` runs on the GPU of a Mac through `std.gpu.metal`: the vector
//! add, the reduction and the kernel with scalar parameters, with the same names, the same
//! arguments in the same order, and the same binding of every argument.
//!
//! Three kernels have no counterpart in `metal_kernels.metal`, because what they test is what
//! this compiler does to a module before Apple's compiler sees it: `index2d`, `mulwide` and
//! `constant_tables` need the high half of 64-bit multiplies and program-scope constant tables,
//! which Apple's GPU compiler handles only in the form the backend rewrites them into.
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
//!
//! There is no `panic` declaration here: on `air64` the default panic namespace is
//! `std.debug.no_panic`, because the GPU has no `printf` and no `__assertfail` to report a message
//! with. A failed safety check traps, which stops the launch and leaves the results unwritten, and
//! the host reports a mismatch.

const std = @import("std");
const gpu = std.gpu;

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

/// One element per thread of a grid of threadgroups of `cols` threads: `out[i] = i`, with the
/// index `row * cols + col` computed in `usize`.
///
/// In a Debug build that multiply is checked for overflow, and the check needs the high 64 bits
/// of the product, which Apple's GPU compiler cannot produce itself: its compiler service dies
/// instead (issue #18). The compiler rebuilds the check out of 32-bit multiplies, and this is the
/// kernel that shows an ordinary `usize` multiply of a Debug kernel running.
export fn index2d(
    out: [*]addrspace(.global) u32,
    cols: u32,
) callconv(.kernel) void {
    const row: usize = gpu.blockIdx(.x);
    const col: usize = gpu.threadIdx(.x);
    const index = row * cols + col;
    out[index] = @intCast(index);
}

/// The multiplies of 64-bit integers that need the high half of the 128-bit product, for each
/// pair `a[i]`, `b[i]` of the first `count`: the high and the low half of the unsigned product,
/// the high half of the signed product, the wrapped product of `@mulWithOverflow`, and, in `flags`,
/// the overflow bit of `@mulWithOverflow` on `u64` (bit 0) and on `i64` (bit 1), plus bit 2 when
/// the two wrapped products differ, which they never should. The host compares each one bit for
/// bit with the CPU.
export fn mulwide(
    a: [*]addrspace(.global) const u64,
    b: [*]addrspace(.global) const u64,
    hi: [*]addrspace(.global) u64,
    lo: [*]addrspace(.global) u64,
    shi: [*]addrspace(.global) u64,
    wrapped: [*]addrspace(.global) u64,
    flags: [*]addrspace(.global) u32,
    count: u32,
) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= count) return;
    const x = a[i];
    const y = b[i];

    const product = @as(u128, x) * y;
    hi[i] = @truncate(product >> 64);
    lo[i] = @truncate(product);

    const sx: i64 = @bitCast(x);
    const sy: i64 = @bitCast(y);
    const signed_product = @as(i128, sx) * sy;
    shi[i] = @bitCast(@as(i64, @truncate(signed_product >> 64)));

    const unsigned_check = @mulWithOverflow(x, y);
    const signed_check = @mulWithOverflow(sx, sy);
    wrapped[i] = unsigned_check[0];
    const differ = @as(u64, @bitCast(signed_check[0])) != unsigned_check[0];
    flags[i] = @as(u32, unsigned_check[1]) | @as(u32, signed_check[1]) << 1 |
        @as(u32, @intFromBool(differ)) << 2;
}

/// The tables of `constant_tables`, which `metal_host.zig` has copies of.
pub const table_primes = [_]u32{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 };
pub const table_wide = [_]u64{
    0x0123456789abcdef, 0xfedcba9876543210, 0x8000000000000001, 0xffffffffffffffff,
    1,                  0,                  0x00000000ffffffff, 0xffffffff00000000,
};
pub const Step = struct { scale: u32, bias: u32, shift: u8 };
pub const table_steps = [_]Step{
    .{ .scale = 3, .bias = 7, .shift = 1 },
    .{ .scale = 5, .bias = 0, .shift = 0 },
    .{ .scale = 0xffff, .bias = 0xdeadbeef, .shift = 7 },
    .{ .scale = 1, .bias = 1, .shift = 31 },
    .{ .scale = 12345, .bias = 678, .shift = 3 },
};

/// Program-scope constants read at an index that each thread computes: tables of 32-bit and
/// 64-bit integers, and of structs, which a Debug build copies out of the table whole. They
/// live in the constant address space, where an Apple GPU keeps program-scope data (issue #18).
pub fn tableValue(i: u32) u32 {
    const wide = table_wide[i % table_wide.len];
    const step = table_steps[i % table_steps.len];
    const mixed = table_primes[i % table_primes.len] *% @as(u32, @truncate(wide >> @intCast(i % 64)));
    return (mixed +% step.scale *% i +% step.bias) >> @intCast(step.shift);
}

/// `out[i] = tableValue(i)` for the first `count` elements.
export fn constant_tables(out: [*]addrspace(.global) u32, count: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= count) return;
    out[i] = tableValue(i);
}
