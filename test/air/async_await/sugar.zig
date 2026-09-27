//! The task forms as a program spells them. `test/air/async_await/hand.zig` is
//! the same program with §3.1's expansions written out by hand, and the two
//! must compile to the same AIR, instruction for instruction; `zig build
//! test-air` compares them (it needs a compiler built with
//! `-Ddebug-extensions`, because `--verbose-air` is gated on it).
const std = @import("std");
const Io = std.Io;

fn add(x: u32) u32 {
    return x + 1;
}

fn voidTask(out: *u32) void {
    out.* += 1;
}

fn awaitBinding(io: Io) u32 {
    const a = async(io) add(3);
    return await a;
}

fn concurrentBinding(io: Io) !u32 {
    const a = try concurrent(io) add(4);
    return await a;
}

fn cancelBinding(io: Io) u32 {
    const a = async(io) add(5);
    _ = cancel a;
    return 0;
}

fn voidBinding(io: Io, out: *u32) void {
    const a = async(io) voidTask(out);
    await a;
}

test "the task forms" {
    const io = std.testing.io;

    try std.testing.expectEqual(4, awaitBinding(io));
    try std.testing.expectEqual(5, concurrentBinding(io));
    try std.testing.expectEqual(0, cancelBinding(io));

    var out: u32 = 0;
    voidBinding(io, &out);
    try std.testing.expectEqual(1, out);
}
