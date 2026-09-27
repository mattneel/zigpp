const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    async() log();
}

// error
// is_test=true
//
// :7:5: error: the operand of `async` is the `Io` to spawn with: `async(io) f(x)`
