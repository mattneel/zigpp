const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    const a = async(io) log();
    await a;
    await a;
}

// error
// is_test=true
//
// :9:11: error: task 'a' is already consumed
