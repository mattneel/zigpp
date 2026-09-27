const std = @import("std");
const Io = std.Io;

fn add(x: u32) u32 {
    return x + 1;
}

fn fail(x: u32) error{Bad}!u32 {
    _ = x;
    return error.Bad;
}

fn noop() void {}

test "a task binding, awaited and joined" {
    const io = std.testing.io;

    // The future's result is the callee's return type, error union and all.
    const a = async(io) fail(1);
    try std.testing.expectError(error.Bad, await a);

    // A binding that is still live when its block ends is joined there, so the
    // block is the join boundary: leaving it waits for the task.
    {
        const b = async(io) add(2);
        try std.testing.expectEqual(3, await b);
    }

    // A result of `void` needs no discard.
    const c = async(io) noop();
    await c;

    // A binding whose result is an error union of a plain value.
    const d = async(io) add(3);
    try std.testing.expectEqual(4, await d);
}

// run
// is_test=true
// target=native
//
