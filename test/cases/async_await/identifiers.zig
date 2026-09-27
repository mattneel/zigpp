const std = @import("std");
const Io = std.Io;

// §2.2: the five words are not keywords. Every declaration and local that uses
// one of them keeps its meaning.
fn @"async"(x: u32) u32 {
    return x + 1;
}

fn await(x: u32) u32 {
    return x + 2;
}

fn cancel(x: u32) u32 {
    return x + 3;
}

const concurrent = 4;
const detach = 5;

fn log() void {}

test "identifiers named after the forms keep working" {
    const io = std.testing.io;

    // Declarations of those names, called as functions.
    try std.testing.expectEqual(2, @"async"(1));
    try std.testing.expectEqual(3, await(1));
    try std.testing.expectEqual(4, cancel(1));
    try std.testing.expectEqual(4, concurrent);
    try std.testing.expectEqual(5, detach);

    // `await(a)` is a call, and `await` next to an operator is a value.
    const await_val = await(1);
    try std.testing.expectEqual(5, await_val + 2);
    const p: ?u32 = await_val;
    try std.testing.expectEqual(3, p orelse 0);

    // Method calls keep their meaning. `Future.await` takes the future by
    // pointer, so the local is a `var`.
    var f = io.async(log, .{});
    f.await(io);

    // A spawn, for contrast, is the form the sugar recognises.
    const a = async(io) log();
    await a;
}

// run
// is_test=true
// target=native
//
