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
//! For AMD GPUs, compile kernels to a code object, the shared library of machine code that the
//! HIP runtime loads:
//!
//! ```
//! zig build-lib -dynamic -target amdgcn-amdhsa -mcpu=gfx1036 -O ReleaseFast kernels.zig
//! ```
//!
//! In a build script, use `std.Build.addLibrary` with `.linkage = .dynamic` and an
//! `amdgcn-amdhsa` target, and `Step.Compile.getEmittedBin`. The host program loads the code
//! object with `hip.Context.loadModule` and launches kernels with `hip.Function.launch`. A code
//! object only runs on GPUs of the architecture that `-mcpu` names, which `hip.Device.archName`
//! reports.
//!
//! On AMD GPUs, shared memory starts at address 0, and the first shared variable of a kernel is
//! there. Zig takes address 0 for null: `@ptrFromInt` does not accept it, and an optional pointer
//! to that variable is null. Index shared variables instead, or `@addrSpaceCast` them to generic
//! pointers, which are never 0.
//!
//! The device-side functions in this namespace are implemented for NVPTX and AMDGPU. The
//! indexing functions and `syncThreads` use builtins that also exist for SPIR-V.

const std = @import("std.zig");
const builtin = @import("builtin");
const output_buffer = @import("gpu/output_buffer.zig");

pub const allocators = @import("gpu/allocators.zig");
pub const cuda = @import("gpu/cuda.zig");
pub const hip = @import("gpu/hip.zig");

test {
    _ = cuda;
    _ = hip;
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
        .amdgcn => blocks: {
            // The dispatch packet has the size of the grid in threads, which need not be a
            // multiple of the block size: the last block is partial.
            const packet = amdgcn.@"llvm.amdgcn.dispatch.ptr"();
            const threads = packet.grid_size[@backingInt(dim)];
            const block_size = packet.workgroup_size[@backingInt(dim)];
            break :blocks threads / block_size + @intFromBool(threads % block_size != 0);
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
    @workGroupBarrier();
}

/// Number of threads in a warp: the threads that execute together and that the shuffle, vote,
/// and warp reduction functions operate on. AMD calls a warp a wave, which has 64 threads before
/// GFX10 and 32 threads from GFX10 on, unless the `wavefrontsize64` or `wavefrontsize32` CPU
/// feature selects the other size.
pub const warp_size = switch (arch) {
    .amdgcn => if (builtin.cpu.has(.amdgcn, .wavefrontsize64))
        64
    else if (builtin.cpu.has(.amdgcn, .wavefrontsize32) or builtin.cpu.has(.amdgcn, .gfx10_insts))
        32
    else
        64,
    else => 32,
};

/// A mask with a bit for each thread of a warp, the one for lane `i` at bit `i`.
pub const WarpMask = @Int(.unsigned, warp_size);

/// Index of the calling thread within its warp, from 0 to `warp_size - 1`.
pub inline fn laneId() u32 {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.read.ptx.sreg.laneid"(),
        // Counts the lanes below the calling one, 32 lanes of the mask at a time.
        .amdgcn => lane: {
            const low = amdgcn.@"llvm.amdgcn.mbcnt.lo"(0xffff_ffff, 0);
            break :lane if (warp_size == 32) low else amdgcn.@"llvm.amdgcn.mbcnt.hi"(0xffff_ffff, low);
        },
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

/// Returns `value` from the thread in lane `src_lane`, taken modulo `warp_size`.
pub inline fn shflBroadcast(value: anytype, src_lane: u32) @TypeOf(value) {
    return shuffle(.idx, value, src_lane);
}

/// Whether `predicate` is true for every thread of the warp.
pub inline fn all(predicate: bool) bool {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.all.sync"(full_mask, predicate),
        .amdgcn => ballot(predicate) == ballot(true),
        else => unsupported("all"),
    };
}

/// Whether `predicate` is true for any thread of the warp.
pub inline fn any(predicate: bool) bool {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.any.sync"(full_mask, predicate),
        .amdgcn => ballot(predicate) != 0,
        else => unsupported("any"),
    };
}

/// Whether `predicate` has the same value for every thread of the warp.
pub inline fn uniform(predicate: bool) bool {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.uni.sync"(full_mask, predicate),
        .amdgcn => uniform: {
            const mask = ballot(predicate);
            break :uniform mask == 0 or mask == ballot(true);
        },
        else => unsupported("uniform"),
    };
}

/// A mask with the bit of each thread of the warp for which `predicate` is true.
pub inline fn ballot(predicate: bool) WarpMask {
    return switch (arch) {
        .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.vote.ballot.sync"(full_mask, predicate),
        .amdgcn => if (warp_size == 32)
            amdgcn.@"llvm.amdgcn.ballot.i32"(predicate)
        else
            amdgcn.@"llvm.amdgcn.ballot.i64"(predicate),
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
/// The PTX ISA and AMD's instruction set references document their error bounds.
pub const fast = struct {
    /// Approximates `@sin(x)` with PTX `sin.approx.f32` or AMD `v_sin_f32`.
    pub inline fn sin(x: f32) f32 {
        return switch (arch) {
            .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.sin.approx.f"(x),
            // LLVM lowers the intrinsic to `v_sin_f32`, scaling the angle to the turns that the
            // instruction takes.
            .amdgcn => amdgcn.@"llvm.sin.f32"(x),
            else => unsupported("fast.sin"),
        };
    }

    /// Approximates `@cos(x)` with PTX `cos.approx.f32` or AMD `v_cos_f32`.
    pub inline fn cos(x: f32) f32 {
        return switch (arch) {
            .nvptx, .nvptx64 => nvvm.@"llvm.nvvm.cos.approx.f"(x),
            .amdgcn => amdgcn.@"llvm.cos.f32"(x),
            else => unsupported("fast.cos"),
        };
    }
};

/// Formats `args` like `std.fmt` and writes the text to the standard output of the host process.
/// Each call formats into a 256-byte buffer on the stack and truncates longer text.
///
/// On NVIDIA GPUs, the driver collects the output of all threads and writes it when the host
/// synchronizes with the device. In Debug builds, formatting needs more stack than the driver
/// gives each thread by default; the host raises the limit with
/// `cuda.Context.setLimit(.stack_size, bytes)`.
///
/// On AMD GPUs, the text goes to a buffer that `hip.Context.loadModule` gives the code object,
/// and `hip.Context.synchronize` writes the text that the buffer collected. It holds 1 MiB of
/// text between synchronizations and drops the text beyond that. A code object that some other
/// host program loads prints nothing.
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
        .amdgcn => {
            var buffer: [256]u8 = undefined;
            const text = std.fmt.bufPrint(&buffer, fmt, args) catch |err| switch (err) {
                error.NoSpaceLeft => &buffer,
            };
            amdgpu_output.write(.print, text);
        },
        else => unsupported("print"),
    }
}

/// Stops the kernel launch with `message`, like a failed `assert` in CUDA C++, and reports the
/// message with the block and the thread that stopped. `std.debug.defaultPanic` calls this on
/// CUDA and AMDHSA, so safety checks and `@panic` in a kernel report their message. The message
/// is truncated to 255 bytes.
///
/// On NVIDIA GPUs, the driver prints the message and the launch fails: the host's next
/// `cuda.Context.synchronize` returns `error.Assert`, and the context cannot run kernels
/// afterwards.
///
/// On AMD GPUs, the wave of the calling thread stops, and the host's next
/// `hip.Context.synchronize` writes the message to standard error and returns `error.Assert`.
/// The context keeps working. Like `print`, this reports nothing when some other host program
/// loaded the code object; the wave still stops.
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
        .amdgcn => {
            amdgpu_output.write(.assert, message[0..@min(message.len, 255)]);
            amdgcn.@"llvm.amdgcn.endpgm"();
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
        .amdgcn => {
            // The same clamping as PTX: a source lane outside the warp is the calling lane.
            const lane = laneId();
            const source = switch (mode) {
                .down => if (lane_operand < warp_size - lane) lane + lane_operand else lane,
                .up => if (lane_operand <= lane) lane - lane_operand else lane,
                .bfly => if (lane ^ lane_operand < warp_size) lane ^ lane_operand else lane,
                .idx => lane_operand % warp_size,
            };
            // `ds_bpermute_b32` reads the value of the lane whose number is the address divided
            // by 4.
            return amdgcn.@"llvm.amdgcn.ds.bpermute"(source * 4, value);
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
/// such as `convergent` for the shuffle and vote operations.
const nvvm = struct {
    extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() u32;
    extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.y"() u32;
    extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.z"() u32;
    extern fn @"llvm.nvvm.read.ptx.sreg.laneid"() u32;
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

/// LLVM intrinsics for AMDGPU. LLVM gives these declarations the attributes of the intrinsics,
/// such as `convergent` for the lane permutation and the ballot.
const amdgcn = struct {
    extern fn @"llvm.amdgcn.dispatch.ptr"() *addrspace(.constant) const DispatchPacket;
    extern fn @"llvm.amdgcn.mbcnt.lo"(mask: u32, base: u32) u32;
    extern fn @"llvm.amdgcn.mbcnt.hi"(mask: u32, base: u32) u32;
    extern fn @"llvm.amdgcn.ds.bpermute"(address: u32, value: u32) u32;
    extern fn @"llvm.amdgcn.ballot.i32"(predicate: bool) u32;
    extern fn @"llvm.amdgcn.ballot.i64"(predicate: bool) u64;
    extern fn @"llvm.amdgcn.endpgm"() noreturn;
    extern fn @"llvm.sin.f32"(x: f32) f32;
    extern fn @"llvm.cos.f32"(x: f32) f32;
};

/// The start of `hsa_kernel_dispatch_packet_t`, the packet that the runtime launched the kernel
/// with.
const DispatchPacket = extern struct {
    header: u16,
    setup: u16,
    /// The size of a block in each dimension.
    workgroup_size: [3]u16,
    reserved0: u16,
    /// The number of threads of the whole grid in each dimension.
    grid_size: [3]u32,
};

/// The output buffer of `print` and `assertFail` on AMD GPUs; see `output_buffer`.
const amdgpu_output = struct {
    /// Null until the host points it at a buffer.
    var buffer: ?*output_buffer.Header = null;

    comptime {
        @export(&buffer, .{ .name = output_buffer.symbol });
    }

    fn write(kind: output_buffer.Kind, text: []const u8) void {
        const Record = output_buffer.Record;
        const header = buffer orelse return;
        const size = std.mem.alignForward(u64, @sizeOf(Record) + text.len, output_buffer.record_alignment);
        const offset = @atomicRmw(u64, &header.claimed, .Add, size, .monotonic);
        const capacity = header.capacity;
        if (offset > capacity or capacity - offset < @sizeOf(Record)) return;
        const len = @min(text.len, capacity - offset - @sizeOf(Record));
        const records: [*]u8 = @ptrCast(@as([*]output_buffer.Header, @ptrCast(header)) + 1);
        const record: *Record = @ptrCast(@alignCast(records + offset));
        record.* = .{
            .size = @intCast(size),
            .len = @intCast(len),
            .kind = kind,
            .block = .{ @workGroupId(0), @workGroupId(1), @workGroupId(2) },
            .thread = .{ @workItemId(0), @workItemId(1), @workItemId(2) },
        };
        @memcpy(records[offset + @sizeOf(Record) ..][0..len], text[0..len]);
    }
};
