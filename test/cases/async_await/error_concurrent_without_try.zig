const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    const a = concurrent(io) log();
    await a;
}

// error
// is_test=true
//
// :7:15: error: the result of `concurrent` is an error union; write `try concurrent(io) f(x)`
