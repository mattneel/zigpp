//! Zig++ Metal spike kernels.
//!
//! Compiled for the nvptx64-cuda stand-in target (`-fno-compiler-rt`), then rewritten to the
//! AIR conventions by `air-rewrite`. Address spaces and the argument order are already the AIR
//! ones: kernel builtins are plain trailing parameters, exactly as in Metal's AIR ABI.

const std = @import("std");
const air = @import("air.zig");

pub const panic = std.debug.no_panic;

/// Threadgroup memory for `reduce`: one SIMD-group partial sum per group.
/// Becomes an internal addrspace(3) global in AIR, sized by the backend.
pub var scratch: [8]f32 addrspace(.shared) = undefined;

/// `c[gid] = a[gid] + b[gid]` over `n` elements.
export fn vadd(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) f32,
    gid: u32,
) callconv(.nvptx_kernel) void {
    c[gid] = a[gid] + b[gid];
}

/// One threadgroup per 256-thread block; sums via SIMD-group reduction, threadgroup
/// memory, barriers, and a device-scope atomic counter.
export fn reduce(
    inbuf: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    counter: *addrspace(.global) u32,
    tgid: u32,
    tid: u32,
    lane: u32,
    sgid: u32,
) callconv(.nvptx_kernel) void {
    const v = inbuf[tgid * 256 + tid];
    const simd_sum = air.@"air.simd_sum.f32"(v);
    if (lane == 0) scratch[sgid] = simd_sum;
    air.@"air.wg.barrier"(air.mem_threadgroup, 1);
    if (tid == 0) {
        var total: f32 = 0;
        for (scratch) |x| total += x;
        out[tgid] = total;
        _ = air.@"air.atomic.global.add.u.i32"(counter, 1, 0, 2, true);
    }
    air.@"air.wg.barrier"(air.mem_threadgroup, 1);
}

/// Parses one decimal float per thread out of a fixed-stride text buffer with
/// `std.fmt.parseFloat`, i.e. standard-library code inside a kernel.
export fn parsef(
    text: [*]addrspace(.global) const u8,
    lengths: [*]addrspace(.global) const u32,
    out: [*]addrspace(.global) f32,
    gid: u32,
) callconv(.nvptx_kernel) void {
    const n = @min(lengths[gid], 32);
    // std.fmt.parseFloat takes a generic-address-space slice, so stage the digits in
    // thread-local memory first (see doc/proposals/metal.md, "standard library in kernels").
    var buf: [32]u8 = undefined;
    var i: u32 = 0;
    while (i < n) : (i += 1) buf[i] = text[gid * 32 + i];
    out[gid] = std.fmt.parseFloat(f32, buf[0..n]) catch 0;
}
