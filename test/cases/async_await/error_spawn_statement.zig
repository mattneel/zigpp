const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    async(io) log();
}

// error
// is_test=true
//
// :6:5: error: a spawn statement needs the frame's group, not yet implemented; bind the task: const f = async(io) f(x);
