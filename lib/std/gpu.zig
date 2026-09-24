//! GPU programming: the functions that kernels call, allocators for kernels, and host-side
//! access to GPUs through vendor drivers.
//!
//! A kernel is an exported function with the `.kernel` calling convention, compiled for a GPU
//! target. Kernels can use the rest of the standard library, as long as they avoid operating
//! system services.
//!
//! ```zig
//! const gpu = @import("std").gpu;
//!
//! export fn vectorAdd(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
//!     const i = gpu.globalId(.x);
//!     if (i >= n) return;
//!     c[i] = a[i] + b[i];
//! }
//! ```
//!
//! For NVIDIA GPUs, compile kernels to PTX assembly:
//!
//! ```
//! zig build-obj -target nvptx64-cuda -mcpu=sm_75 -O ReleaseFast -fno-emit-bin -femit-asm=kernels.ptx kernels.zig
//! ```
//!
//! In a build script, use `std.Build.addObject` with an `nvptx64-cuda` target and
//! `Step.Compile.getEmittedAsm`. The host program loads the PTX with `cuda.Context.loadModule`
//! and launches kernels with `cuda.Function.launch`. PTX for an older `-mcpu` also runs on newer
//! GPUs, because the driver compiles it for the GPU when it is loaded.
//!
//! The device-side functions in this namespace are implemented for NVPTX. The indexing functions
//! use builtins that also exist for other GPU targets.

const std = @import("std.zig");
const builtin = @import("builtin");

pub const allocators = @import("gpu/allocators.zig");
pub const cuda = @import("gpu/cuda.zig");

test {
    _ = cuda;
}

const arch = builtin.cpu.arch;

/// A dimension of the thread, block, and grid index spaces.
pub const Dim = enum(u2) { x, y, z };

/// Index of the calling thread within its block, like CUDA's `threadIdx`.
pub inline fn threadIdx(comptime dim: Dim) u32 {
    return @workItemId(@backingInt(dim));
}

/// Index of the calling thread's block within the grid, like CUDA's `blockIdx`.
pub inline fn blockIdx(comptime dim: Dim) u32 {
    return @workGroupId(@backingInt(dim));
}

/// Number of threads in each block, like CUDA's `blockDim`.
pub inline fn blockDim(comptime dim: Dim) u32 {
    return @workGroupSize(@backingInt(dim));
}

/// Number of blocks in the grid, like CUDA's `gridDim`.
pub inline fn gridDim(comptime dim: Dim) u32 {
    return switch (arch) {
        .nvptx, .nvptx64 => switch (dim) {
            .x => nvvm.@"llvm.nvvm.read.ptx.sreg.nctaid.x"(),
            .y => nvvm.@"llvm.nvvm.read.ptx.sreg.nctaid.y"(),
            .z => nvvm.@"llvm.nvvm.read.ptx.sreg.nctaid.z"(),
        },
        else => unsupported("gridDim"),
    };
}

/// Index of the calling thread within the whole grid: `blockIdx(dim) * blockDim(dim) + threadIdx(dim)`.
pub inline fn globalId(comptime dim: Dim) u32 {
    return blockIdx(dim) * blockDim(dim) + threadIdx(dim);
}

/// Waits until every thread of the block has reached this call, and makes the memory writes that
/// each thread made before the call visible to the others, like CUDA's `__syncthreads`.
/// All threads of the block must reach the same call; calling it where only some threads of the
/// block go is undefined behavior.
pub inline fn syncThreads() void {
    switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.barrier.cta.sync.aligned.all"(0),
        else => unsupported("syncThreads"),
    }
}

/// Number of threads in a warp: the threads that execute together and that the shuffle, vote,
/// and warp reduction functions operate on.
pub const warp_size = 32;

/// Index of the calling thread within its warp, from 0 to `warp_size - 1`.
pub inline fn laneId() u32 {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.read.ptx.sreg.laneid"(),
        else => unsupported("laneId"),
    };
}

// The warp functions below operate on the full warp. Every thread of the warp that has not
// exited must call them together, and a value read from a thread that has exited is undefined.

/// Returns `value` from the thread `delta` lanes above the calling thread. Threads whose source
/// lane would be past the end of the warp get their own `value`. Supports any type of at most
/// 64 bits that `@bitCast` accepts.
pub inline fn shflDown(value: anytype, delta: u32) @TypeOf(value) {
    return shuffle(.down, value, delta);
}

/// Returns `value` from the thread `delta` lanes below the calling thread. Threads whose source
/// lane would be before the start of the warp get their own `value`.
pub inline fn shflUp(value: anytype, delta: u32) @TypeOf(value) {
    return shuffle(.up, value, delta);
}

/// Returns `value` from the thread in lane `laneId() ^ lane_mask`.
pub inline fn shflXor(value: anytype, lane_mask: u32) @TypeOf(value) {
    return shuffle(.bfly, value, lane_mask);
}

/// Returns `value` from the thread in lane `src_lane`.
pub inline fn shflBroadcast(value: anytype, src_lane: u32) @TypeOf(value) {
    return shuffle(.idx, value, src_lane);
}

/// Whether `predicate` is true for every thread of the warp.
pub inline fn all(predicate: bool) bool {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.all.sync"(full_mask, predicate),
        else => unsupported("all"),
    };
}

/// Whether `predicate` is true for any thread of the warp.
pub inline fn any(predicate: bool) bool {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.any.sync"(full_mask, predicate),
        else => unsupported("any"),
    };
}

/// Whether `predicate` has the same value for every thread of the warp.
pub inline fn uniform(predicate: bool) bool {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.uni.sync"(full_mask, predicate),
        else => unsupported("uniform"),
    };
}

/// A mask with bit `i` set when `predicate` is true for the thread in lane `i`.
pub inline fn ballot(predicate: bool) u32 {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.ballot.sync"(full_mask, predicate),
        else => unsupported("ballot"),
    };
}

/// Number of threads of the warp for which `predicate` is true.
pub inline fn popcount(predicate: bool) u32 {
    return @popCount(ballot(predicate));
}

/// Sum of `value` over all threads of the warp, returned to every thread.
/// Integer overflow is checked like `+`.
pub fn warpReduceSum(value: anytype) @TypeOf(value) {
    return warpReduce(.sum, value);
}

/// Maximum of `value` over all threads of the warp, returned to every thread.
pub fn warpReduceMax(value: anytype) @TypeOf(value) {
    return warpReduce(.max, value);
}

/// Minimum of `value` over all threads of the warp, returned to every thread.
pub fn warpReduceMin(value: anytype) @TypeOf(value) {
    return warpReduce(.min, value);
}

// Atomic read-modify-write operations with the semantics of their CUDA namesakes: relaxed
// ordering, visible to the whole device. `ptr` may point to global, shared, or generic memory.
// For other orderings, use `@atomicRmw` and `@cmpxchgStrong` directly.

/// Adds `operand` to `ptr.*` and returns the previous value.
pub inline fn atomicAdd(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*) {
    return @atomicRmw(@TypeOf(ptr.*), ptr, .Add, operand, .monotonic);
}

/// Stores `operand` to `ptr.*` and returns the previous value.
pub inline fn atomicExchange(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*) {
    return @atomicRmw(@TypeOf(ptr.*), ptr, .Xchg, operand, .monotonic);
}

/// Stores `new_value` to `ptr.*` if it equals `expected`. Returns the previous value, which
/// equals `expected` if and only if the store happened.
pub inline fn atomicCAS(ptr: anytype, expected: @TypeOf(ptr.*), new_value: @TypeOf(ptr.*)) @TypeOf(ptr.*) {
    return @cmpxchgStrong(@TypeOf(ptr.*), ptr, expected, new_value, .monotonic, .monotonic) orelse expected;
}

/// Stores the smaller of `ptr.*` and `operand` to `ptr.*` and returns the previous value.
pub inline fn atomicMin(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*) {
    return @atomicRmw(@TypeOf(ptr.*), ptr, .Min, operand, .monotonic);
}

/// Stores the larger of `ptr.*` and `operand` to `ptr.*` and returns the previous value.
pub inline fn atomicMax(ptr: anytype, operand: @TypeOf(ptr.*)) @TypeOf(ptr.*) {
    return @atomicRmw(@TypeOf(ptr.*), ptr, .Max, operand, .monotonic);
}

/// Hardware approximations of math functions, like CUDA's `__sinf`. They are much faster than
/// the builtins, which compute full-precision results, but have absolute rather than relative
/// error bounds, so they lose precision for results near zero and for large inputs.
/// The PTX ISA documents their exact error bounds.
pub const fast = struct {
    /// Approximates `@sin(x)` with PTX `sin.approx.f32`.
    pub inline fn sin(x: f32) f32 {
        return switch (arch) {
            .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.sin.approx.f"(x),
            else => unsupported("fast.sin"),
        };
    }

    /// Approximates `@cos(x)` with PTX `cos.approx.f32`.
    pub inline fn cos(x: f32) f32 {
        return switch (arch) {
            .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.cos.approx.f"(x),
            else => unsupported("fast.cos"),
        };
    }
};

/// Formats `args` like `std.fmt` and writes the text to the standard output of the host process.
/// The driver collects the output of all threads and writes it when the host synchronizes with
/// the device. Each call formats into a 256-byte buffer on the stack and truncates longer text.
/// In Debug builds, formatting needs more stack than the driver gives each thread by default;
/// the host raises the limit with `cuda.Context.setLimit(.stack_size, bytes)`.
pub fn print(comptime fmt: []const u8, args: anytype) void {
    switch (arch) {
        .nvptx, .nvptx64 => {
            var buffer: [256]u8 = undefined;
            const capacity = buffer.len - 1;
            const text = std.fmt.bufPrint(buffer[0..capacity], fmt, args) catch |err| switch (err) {
                error.NoSpaceLeft => buffer[0..capacity],
            };
            buffer[text.len] = 0;
            // `vprintf` takes its arguments as a buffer laid out like a C struct. The text is an
            // argument rather than the format so that '%' in it is printed as is.
            const Arguments = extern struct { text: [*:0]const u8 };
            const arguments: Arguments = .{ .text = buffer[0..text.len :0] };
            _ = nvptx_syscalls.vprintf("%s", &arguments);
        },
        else => unsupported("print"),
    }
}

/// Stops the kernel launch with `message`, like a failed `assert` in CUDA C++. The driver prints
/// the message with the block and the thread that stopped, and the launch fails: the host's next
/// `cuda.Context.synchronize` returns `error.Assert`, and the context cannot run kernels
/// afterwards. `std.debug.defaultPanic` calls this on CUDA, so safety checks and `@panic` in a
/// kernel report their message. The message is truncated to 255 bytes.
pub fn assertFail(message: []const u8) noreturn {
    switch (arch) {
        .nvptx, .nvptx64 => {
            var buffer: [256]u8 = undefined;
            const len = @min(message.len, buffer.len - 1);
            @memcpy(buffer[0..len], message[0..len]);
            buffer[len] = 0;
            // The driver prints "file:line: function: block: [..], thread: [..] Assertion
            // `message` failed.", and a panic has no location to put in the first three.
            nvptx_syscalls.__assertfail(buffer[0..len :0], "zig", 0, "panic", 1);
            @trap();
        },
        else => unsupported("assertFail"),
    }
}

const full_mask: u32 = 0xffff_ffff;

const ShuffleMode = enum { down, up, bfly, idx };

inline fn shuffle(comptime mode: ShuffleMode, value: anytype, lane_operand: u32) @TypeOf(value) {
    const T = @TypeOf(value);
    const bits = @bitSizeOf(T);
    if (bits > 64) @compileError("warp shuffles support values of at most 64 bits, found '" ++ @typeName(T) ++ "'");
    if (bits == 0) return value;
    const Bits = @Int(.unsigned, bits);
    const value_bits: Bits = @bitCast(value);
    if (bits <= 32) {
        return @bitCast(@as(Bits, @truncate(shuffle32(mode, value_bits, lane_operand))));
    }
    const wide: u64 = value_bits;
    const low = shuffle32(mode, @truncate(wide), lane_operand);
    const high = shuffle32(mode, @truncate(wide >> 32), lane_operand);
    return @bitCast(@as(Bits, @truncate(@as(u64, high) << 32 | low)));
}

inline fn shuffle32(comptime mode: ShuffleMode, value: u32, lane_operand: u32) u32 {
    switch (arch) {
        .nvptx, .nvptx64 => {
            // Bits 0-4 of `c` clamp the source lane and bits 8-12 select the segment of the warp;
            // zero selects the whole warp. `.up` clamps at lane 0, the others at the last lane.
            const c: u32 = switch (mode) {
                .up => 0,
                .down, .bfly, .idx => warp_size - 1,
            };
            return switch (mode) {
                .down => nvvm.@"llvm.nvvm.shfl.sync.down.i32"(full_mask, value, lane_operand, c),
                .up => nvvm.@"llvm.nvvm.shfl.sync.up.i32"(full_mask, value, lane_operand, c),
                .bfly => nvvm.@"llvm.nvvm.shfl.sync.bfly.i32"(full_mask, value, lane_operand, c),
                .idx => nvvm.@"llvm.nvvm.shfl.sync.idx.i32"(full_mask, value, lane_operand, c),
            };
        },
        else => unsupported("shuffle"),
    }
}

fn warpReduce(comptime op: enum { sum, min, max }, value: anytype) @TypeOf(value) {
    var result = value;
    comptime var lane_mask = warp_size / 2;
    inline while (lane_mask > 0) : (lane_mask /= 2) {
        const other = shflXor(result, lane_mask);
        result = switch (op) {
            .sum => result + other,
            .min => @min(result, other),
            .max => @max(result, other),
        };
    }
    return result;
}

fn unsupported(comptime name: []const u8) noreturn {
    @compileError("std.gpu." ++ name ++ " is not implemented for " ++ @tagName(arch));
}

/// LLVM intrinsics for NVPTX. LLVM gives these declarations the attributes of the intrinsics,
/// such as `convergent` for the barrier, shuffle, and vote operations.
const nvvm = struct {
    extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() u32;
    extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.y"() u32;
    extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.z"() u32;
    extern fn @"llvm.nvvm.read.ptx.sreg.laneid"() u32;
    extern fn @"llvm.nvvm.barrier.cta.sync.aligned.all"(barrier: u32) void;
    extern fn @"llvm.nvvm.shfl.sync.down.i32"(mask: u32, value: u32, delta: u32, c: u32) u32;
    extern fn @"llvm.nvvm.shfl.sync.up.i32"(mask: u32, value: u32, delta: u32, c: u32) u32;
    extern fn @"llvm.nvvm.shfl.sync.bfly.i32"(mask: u32, value: u32, lane_mask: u32, c: u32) u32;
    extern fn @"llvm.nvvm.shfl.sync.idx.i32"(mask: u32, value: u32, lane: u32, c: u32) u32;
    extern fn @"llvm.nvvm.vote.all.sync"(mask: u32, predicate: bool) bool;
    extern fn @"llvm.nvvm.vote.any.sync"(mask: u32, predicate: bool) bool;
    extern fn @"llvm.nvvm.vote.uni.sync"(mask: u32, predicate: bool) bool;
    extern fn @"llvm.nvvm.vote.ballot.sync"(mask: u32, predicate: bool) u32;
    extern fn @"llvm.nvvm.sin.approx.f"(x: f32) f32;
    extern fn @"llvm.nvvm.cos.approx.f"(x: f32) f32;
};

/// Functions that the CUDA driver provides to every PTX module.
const nvptx_syscalls = struct {
    extern fn vprintf(format: [*:0]const u8, arguments: ?*const anyopaque) i32;
    extern fn __assertfail(message: [*:0]const u8, file: [*:0]const u8, line: u32, function: [*:0]const u8, char_size: usize) void;
};
