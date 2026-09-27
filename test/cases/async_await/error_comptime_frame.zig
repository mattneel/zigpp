const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    comptime {
        const a = async(io) log();
        _ = a;
    }
}

// error
// is_test=true
//
// :8:19: error: a task needs a frame to join it; spawn inside a function body
