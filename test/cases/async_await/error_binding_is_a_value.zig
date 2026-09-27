const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    const a = async(io) log();
    _ = a;
}

// error
// is_test=true
//
// :8:9: error: a spawned task is not a value; `await` it or `cancel` it
