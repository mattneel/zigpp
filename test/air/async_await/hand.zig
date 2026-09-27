//! The hand-written expansion of every form `test/air/async_await/sugar.zig`
//! uses, exactly as §3.1 of `doc/proposals/async-await.md` spells it: the `Io`
//! in a slot of its own, the future in a frame slot, a flag, and a
//! `defer`/`errdefer` pair that joins and cancels behind the flag.
const std = @import("std");
const Io = std.Io;

fn add(x: u32) u32 {
    return x + 1;
}

fn voidTask(out: *u32) void {
    out.* += 1;
}

fn awaitBinding(io: Io) u32 {
    const a_io = io;
    var a = a_io.async(add, .{3});
    var a_live = true;
    defer {
        if (a_live) _ = a.await(a_io);
    }
    errdefer {
        if (a_live) _ = a.cancel(a_io);
    }
    a_live = false;
    return a.await(a_io);
}

fn concurrentBinding(io: Io) u32 {
    const a_io = io;
    var a = try a_io.concurrent(add, .{4});
    var a_live = true;
    defer {
        if (a_live) _ = a.await(a_io);
    }
    errdefer {
        if (a_live) _ = a.cancel(a_io);
    }
    a_live = false;
    return a.await(a_io);
}

fn cancelBinding(io: Io) u32 {
    const a_io = io;
    var a = a_io.async(add, .{5});
    var a_live = true;
    defer {
        if (a_live) _ = a.await(a_io);
    }
    errdefer {
        if (a_live) _ = a.cancel(a_io);
    }
    a_live = false;
    _ = a.cancel(a_io);
    return 0;
}

fn voidBinding(io: Io, out: *u32) void {
    const a_io = io;
    var a = a_io.async(voidTask, .{out});
    var a_live = true;
    defer {
        if (a_live) _ = a.await(a_io);
    }
    errdefer {
        if (a_live) _ = a.cancel(a_io);
    }
    a_live = false;
    a.await(a_io);
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
