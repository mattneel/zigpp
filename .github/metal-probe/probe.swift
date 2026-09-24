// Checks what Metal compute a GitHub-hosted macOS runner offers: the device, its limits, and
// whether kernels compiled from Metal source at run time give correct results for plain
// arithmetic, threadgroup memory with barriers, SIMD-group operations, and device atomics.
import Foundation
import Metal

var failures = 0

func check(_ name: String, _ ok: Bool, _ detail: String) {
    print("\(ok ? "PASS" : "FAIL") \(name): \(detail)")
    if !ok { failures += 1 }
}

guard let device = MTLCreateSystemDefaultDevice() else {
    print("FAIL no Metal device")
    exit(1)
}
print("device: \(device.name)")
print("all devices: \(MTLCopyAllDevices().map { $0.name })")
print("unified memory: \(device.hasUnifiedMemory), low power: \(device.isLowPower), headless: \(device.isHeadless)")
let families: [(String, MTLGPUFamily)] = [
    ("apple6", .apple6), ("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9),
    ("mac2", .mac2), ("metal3", .metal3),
]
print("families: " + families.map { "\($0.0)=\(device.supportsFamily($0.1))" }.joined(separator: " "))
print("max threadgroup memory: \(device.maxThreadgroupMemoryLength) bytes")
print("max threads per threadgroup: \(device.maxThreadsPerThreadgroup)")
print("recommended max working set: \(device.recommendedMaxWorkingSetSize / (1 << 20)) MiB")

let source = """
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                device float* c [[buffer(2)]], constant uint& n [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    if (i < n) c[i] = a[i] + b[i];
}

kernel void reduce(device const uint* input [[buffer(0)]], device atomic_uint* total [[buffer(1)]],
                   constant uint& n [[buffer(2)]], threadgroup uint* partial [[threadgroup(0)]],
                   uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                   uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                   uint width [[threads_per_simdgroup]]) {
    uint value = gid < n ? input[gid] : 0;
    value = simd_sum(value);
    if (lane == 0) partial[lid / width] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint sum = 0;
        for (uint k = 0; k < threads / width; k++) sum += partial[k];
        atomic_fetch_add_explicit(total, sum, memory_order_relaxed);
    }
}

kernel void shuffle(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
    out[gid] = simd_shuffle_xor(gid, 1u);
}
"""

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: source, options: nil)
    check("compile Metal source at run time", true, "ok")
} catch {
    check("compile Metal source at run time", false, "\(error)")
    exit(1)
}

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

let queue = device.makeCommandQueue()!

func run(_ state: MTLComputePipelineState, groups: Int, threads: Int, threadgroupMemory: Int = 0,
         _ encode: (MTLComputeCommandEncoder) -> Void) -> Double {
    let commands = queue.makeCommandBuffer()!
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(state)
    encode(encoder)
    if threadgroupMemory > 0 { encoder.setThreadgroupMemoryLength(threadgroupMemory, index: 0) }
    encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error { print("command buffer error: \(error)") }
    return (commands.gpuEndTime - commands.gpuStartTime) * 1000
}

// Vector addition over a size that is not a multiple of the threadgroup size.
do {
    let n = (1 << 22) + 17
    let a = (0..<n).map { Float($0) }
    let b = (0..<n).map { Float($0) * 0.5 }
    let bufferA = device.makeBuffer(bytes: a, length: n * 4, options: .storageModeShared)!
    let bufferB = device.makeBuffer(bytes: b, length: n * 4, options: .storageModeShared)!
    let bufferC = device.makeBuffer(length: n * 4, options: .storageModeShared)!
    var count = UInt32(n)
    let state = pipeline("add")
    let ms = run(state, groups: (n + 255) / 256, threads: 256) { encoder in
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        encoder.setBytes(&count, length: 4, index: 3)
    }
    let c = bufferC.contents().bindMemory(to: Float.self, capacity: n)
    var wrong = 0
    for i in 0..<n where c[i] != a[i] + b[i] { wrong += 1 }
    check("vector add", wrong == 0, "\(n) elements, \(wrong) wrong, \(String(format: "%.3f", ms)) ms on the GPU, SIMD width \(state.threadExecutionWidth)")
}

// Threadgroup memory, a barrier, simd_sum, and a device atomic.
do {
    let n = 1_000_003
    let input = (0..<n).map { UInt32($0 % 1000) }
    let expected = input.reduce(UInt64(0)) { $0 + UInt64($1) }
    let bufferIn = device.makeBuffer(bytes: input, length: n * 4, options: .storageModeShared)!
    let bufferTotal = device.makeBuffer(length: 4, options: .storageModeShared)!
    memset(bufferTotal.contents(), 0, 4)
    var count = UInt32(n)
    let state = pipeline("reduce")
    let threads = 256
    _ = run(state, groups: (n + threads - 1) / threads, threads: threads, threadgroupMemory: threads * 4) { encoder in
        encoder.setBuffer(bufferIn, offset: 0, index: 0)
        encoder.setBuffer(bufferTotal, offset: 0, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
    }
    let total = bufferTotal.contents().load(as: UInt32.self)
    check("threadgroup reduction with simd_sum and atomics", UInt64(total) == expected, "got \(total), expected \(expected)")
}

// SIMD-group shuffle.
do {
    let n = 4096
    let bufferOut = device.makeBuffer(length: n * 4, options: .storageModeShared)!
    _ = run(pipeline("shuffle"), groups: n / 256, threads: 256) { encoder in
        encoder.setBuffer(bufferOut, offset: 0, index: 0)
    }
    let out = bufferOut.contents().bindMemory(to: UInt32.self, capacity: n)
    var wrong = 0
    for i in 0..<n where out[i] != UInt32(i ^ 1) { wrong += 1 }
    check("simd_shuffle_xor", wrong == 0, "\(n) threads, \(wrong) wrong")
}

print(failures == 0 ? "ALL PASSED" : "\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
