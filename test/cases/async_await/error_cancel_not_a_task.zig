const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    const f = io.async(log, .{});
    _ = cancel f;
}

// error
// is_test=true
//
// :8:16: error: `cancel` works on a task spawned by the keywords; for a future value write `x.cancel(io)`
