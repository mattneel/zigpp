//! Spike device API: raw AIR intrinsics, declared exactly as Apple's AIR 2.8 emits them.
//! In the finished compiler, std.gpu's device API lowers to these (see doc/proposals/metal.md).
//!
//! The stand-in target is nvptx64-cuda, whose LLVM address spaces happen to be the AIR
//! numbers: .global = 1 (device), .constant = 2, .shared = 3 (threadgroup).

/// Reduces `value` across the calling SIMD group; the result is returned to every lane.
pub extern fn @"air.simd_sum.f32"(value: f32) f32;

/// Threadgroup barrier. 2 = `mem_none|mem_threadgroup` memory flags, 1 = barrier id.
pub extern fn @"air.wg.barrier"(mem_flags: i32, barrier_id: i32) void;

/// Device-scope atomic add on i32, relaxed ordering. Returns the previous value.
pub extern fn @"air.atomic.global.add.u.i32"(
    ptr: *addrspace(.global) u32,
    value: u32,
    ordering: i32,
    scope: i32,
    volatile_: bool,
) u32;

pub const mem_threadgroup: i32 = 2;
