//! Allocators for GPU kernels.
//!
//! * `device_heap` allocates global memory from the driver's device heap. Every thread can use
//!   it, and the memory stays allocated across kernel launches until it is freed.
//! * `BumpAllocator` allocates from an array in shared memory that all threads of a block use
//!   together. It is fast, and the memory is gone when the block finishes.
//! * For memory that only one thread uses, `std.heap.FixedBufferAllocator` works over any buffer,
//!   such as an array on the thread's stack or a slice of global memory.
//!
//! These are `std.mem.Allocator`s, so containers such as `std.ArrayList` and `std.HashMap` work
//! in kernels.

const std = @import("../std.zig");
const builtin = @import("builtin");
const gpu = std.gpu;
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

const arch = builtin.cpu.arch;

/// Allocates from the device heap with the `malloc` and `free` that the driver provides to
/// kernels. The heap is 8 MiB unless the host changes `cuda.Limit.malloc_heap_size` before
/// launching the first kernel that uses it.
pub const device_heap: Allocator = .{
    .ptr = undefined,
    .vtable = &.{
        .alloc = DeviceHeap.alloc,
        .resize = DeviceHeap.resize,
        .remap = DeviceHeap.remap,
        .free = DeviceHeap.free,
    },
};

const DeviceHeap = struct {
    /// The alignment of every block that the driver's `malloc` returns.
    const malloc_alignment: Alignment = .@"16";

    /// Blocks with a larger alignment are allocated with room to align them, and the address
    /// of the block from `malloc` is stored right before the aligned memory.
    const Header = usize;

    fn alloc(_: *anyopaque, len: usize, alignment: Alignment, _: usize) ?[*]u8 {
        if (alignment.compare(.lte, malloc_alignment)) return @ptrCast(syscalls.malloc(len));
        const padded_len = std.math.add(usize, len, alignment.toByteUnits() - 1 + @sizeOf(Header)) catch return null;
        const block: [*]u8 = @ptrCast(syscalls.malloc(padded_len) orelse return null);
        const aligned: [*]u8 = @ptrFromInt(alignment.forward(@intFromPtr(block) + @sizeOf(Header)));
        const header: *align(1) Header = @ptrCast(aligned - @sizeOf(Header));
        header.* = @intFromPtr(block);
        return aligned;
    }

    fn resize(_: *anyopaque, memory: []u8, _: Alignment, new_len: usize, _: usize) bool {
        return new_len <= memory.len;
    }

    fn remap(_: *anyopaque, memory: []u8, _: Alignment, new_len: usize, _: usize) ?[*]u8 {
        return if (new_len <= memory.len) memory.ptr else null;
    }

    fn free(_: *anyopaque, memory: []u8, alignment: Alignment, _: usize) void {
        if (alignment.compare(.lte, malloc_alignment)) return syscalls.free(memory.ptr);
        const header: *align(1) const Header = @ptrCast(memory.ptr - @sizeOf(Header));
        syscalls.free(@ptrFromInt(header.*));
    }
};

/// A bump allocator over an array in shared memory, shared by all threads of a block. Threads
/// may allocate concurrently: the offset of the unused memory is kept at the start of the array
/// and advanced atomically, so each thread can use its own copy of the allocator. Freeing the
/// most recent allocation returns its memory, and all memory is released when the block
/// finishes.
///
/// ```zig
/// var heap: [16 * 1024]u8 addrspace(.shared) = undefined;
///
/// export fn kernel() callconv(.kernel) void {
///     var bump = std.gpu.allocators.BumpAllocator(heap.len).init(&heap);
///     if (std.gpu.threadIdx(.x) == 0) {
///         var list: std.ArrayList(u32) = .empty;
///         list.append(bump.allocator(), 42) catch return;
///     }
/// }
/// ```
pub fn BumpAllocator(comptime size: usize) type {
    return struct {
        buffer: *addrspace(.shared) [size]u8,

        const Self = @This();

        /// Byte offset of the unused memory in `buffer`.
        const Offset = u32;

        comptime {
            if (size > std.math.maxInt(Offset)) @compileError("BumpAllocator buffers are limited to 4 GiB");
            if (size < @sizeOf(Offset) + @alignOf(Offset) - 1) @compileError("BumpAllocator buffer is too small");
        }

        /// Every thread of the block must call `init` with the same buffer, because it waits
        /// with `std.gpu.syncThreads` until the first thread has initialized the buffer.
        pub fn init(buffer: *addrspace(.shared) [size]u8) Self {
            const self: Self = .{ .buffer = buffer };
            if (gpu.threadIdx(.x) == 0 and gpu.threadIdx(.y) == 0 and gpu.threadIdx(.z) == 0) {
                const offset_ptr = self.offsetPtr();
                offset_ptr.* = @intCast(@intFromPtr(offset_ptr) + @sizeOf(Offset) - @intFromPtr(buffer));
            }
            gpu.syncThreads();
            return self;
        }

        pub fn allocator(self: *Self) Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = alloc,
                    .resize = resize,
                    .remap = remap,
                    .free = free,
                },
            };
        }

        /// Number of bytes of `buffer` in use, including the offset kept at its start.
        pub fn used(self: Self) usize {
            return @atomicLoad(Offset, self.offsetPtr(), .monotonic);
        }

        /// The offset lives in the first suitably aligned bytes of the buffer, since the buffer
        /// itself may have any alignment.
        fn offsetPtr(self: Self) *addrspace(.shared) Offset {
            return @ptrFromInt(std.mem.alignForward(usize, @intFromPtr(self.buffer), @alignOf(Offset)));
        }

        /// `buffer` in the generic address space, which the allocations are returned in.
        fn bytes(self: Self) *[size]u8 {
            return @addrSpaceCast(self.buffer);
        }

        /// Offset of `memory` within `buffer`.
        fn offsetOf(self: Self, memory: []u8) usize {
            return @intFromPtr(memory.ptr) - @intFromPtr(self.bytes());
        }

        fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, _: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const offset_ptr = self.offsetPtr();
            const base = @intFromPtr(self.bytes());
            var offset = @atomicLoad(Offset, offset_ptr, .monotonic);
            while (true) {
                const start = alignment.forward(base + offset) - base;
                const end = std.math.add(usize, start, len) catch return null;
                if (end > size) return null;
                // Build the result from the address rather than from `bytes()`: when the result
                // derives from a shared pointer, LLVM's NVPTX back end merges it with the `null`
                // of the failure path in the shared address space, and converting that `null` back
                // to a generic pointer yields the start of shared memory instead of null.
                offset = @cmpxchgWeak(Offset, offset_ptr, offset, @intCast(end), .monotonic, .monotonic) orelse
                    return @ptrFromInt(base + start);
            }
        }

        fn resize(ctx: *anyopaque, memory: []u8, _: Alignment, new_len: usize, _: usize) bool {
            if (new_len <= memory.len) return true;
            // Only the most recent allocation can grow.
            const self: *Self = @ptrCast(@alignCast(ctx));
            const start = self.offsetOf(memory);
            const new_end = std.math.add(usize, start, new_len) catch return false;
            if (new_end > size) return false;
            const end: Offset = @intCast(start + memory.len);
            return @cmpxchgStrong(Offset, self.offsetPtr(), end, @intCast(new_end), .monotonic, .monotonic) == null;
        }

        fn remap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
            return if (resize(ctx, memory, alignment, new_len, ret_addr)) memory.ptr else null;
        }

        fn free(ctx: *anyopaque, memory: []u8, _: Alignment, _: usize) void {
            // Only the most recent allocation can be returned.
            const self: *Self = @ptrCast(@alignCast(ctx));
            const start = self.offsetOf(memory);
            const end: Offset = @intCast(start + memory.len);
            _ = @cmpxchgStrong(Offset, self.offsetPtr(), end, @intCast(start), .monotonic, .monotonic);
        }
    };
}

/// Functions that the CUDA driver provides to every PTX module.
const syscalls = switch (arch) {
    .nvptx, .nvptx64 => struct {
        extern fn malloc(size: usize) ?*anyopaque;
        extern fn free(ptr: ?*anyopaque) void;
    },
    else => @compileError("std.gpu.allocators.device_heap is not implemented for " ++ @tagName(arch)),
};
