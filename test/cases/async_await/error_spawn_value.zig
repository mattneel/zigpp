const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    _ = async(io) log();
}

// error
// is_test=true
//
// :7:9: error: a spawn is not a value; bind it: const f = async(io) f(x);
