// The vector add, the reduction and the scalar kernel of `kernels.zig`, in the Metal Shading
// Language, for Apple's own compiler to build:
//
//     xcrun metal -c -O2 metal_kernels.metal -o metal_kernels.air
//     xcrun metallib metal_kernels.air -o reference.metallib
//
// The libraries that Zig++ builds for the `air64-macos` target hold the same kernels with the same
// arguments, in the same order, bound the same way: a pointer parameter is a buffer at its index,
// which the host binds with `setBuffer:offset:atIndex:`, a scalar parameter is the bytes of the
// value at its index, which the host binds with `setBytes:length:atIndex:`, and the position of the
// calling thread comes from the dispatch, so the host never passes it. `metal_host.zig` runs either
// library through `std.gpu.metal` and compares the results with the CPU.

#include <metal_stdlib>
using namespace metal;

// `c[i] = a[i] + b[i]`: one thread per element, launched as 64 threadgroups of 64 threads, which
// covers the 4096 elements of the test exactly.
kernel void vadd(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* c [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    c[gid] = a[gid] + b[gid];
}

// One threadgroup per 256-thread block: the sum of a threadgroup goes through a SIMD-group
// reduction, the partial sums of the eight SIMD groups of a threadgroup meet in threadgroup memory
// behind a barrier, and the first thread of the threadgroup adds the total to a device-scope atomic
// counter. Launched as four threadgroups of 256 threads over 1024 values.
kernel void reduce(device const float* inbuf [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   device atomic_uint* counter [[buffer(2)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];

    const float v = inbuf[tgid * 256 + tid];
    const float simd_sum_v = simd_sum(v);
    if (lane == 0) scratch[sgid] = simd_sum_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint i = 0; i < 8; i++) total += scratch[i];
        out[tgid] = total;
        atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The kernel with scalar parameters: the host binds `factor` at the index 1 and `count` at the
// index 2, in the order the kernel declares them, with `setBytes:length:atIndex:`, which is where a
// scalar parameter of an AIR kernel reads its value from as well.
kernel void scale(device float* x [[buffer(0)]],
                  constant float& factor [[buffer(1)]],
                  constant uint& count [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    x[gid] = x[gid] * factor;
}
