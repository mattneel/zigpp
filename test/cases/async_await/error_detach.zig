const std = @import("std");
const Io = std.Io;

fn log() void {}

fn handler(io: Io) void {
    detach(io) log();
}

// error
// is_test=true
//
// :7:5: error: `detach` needs the borrow checker's `owned` rule; compile this module with `-fborrow-check`
