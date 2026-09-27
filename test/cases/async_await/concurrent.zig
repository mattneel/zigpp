const std = @import("std");
const Io = std.Io;

fn add(x: u32) u32 {
    return x + 1;
}

fn slow(x: u32) u32 {
    std.atomic.spinLoopHint();
    return x + 2;
}

test "concurrent" {
    const io = std.testing.io;

    // `concurrent` promises a unit of concurrency, and can fail to place one.
    const a = try concurrent(io) slow(1);
    try std.testing.expectEqual(3, await a);

    const b = try concurrent(io) add(4);
    try std.testing.expectEqual(5, await b);
}

// run
// is_test=true
// target=native
//
