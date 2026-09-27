const std = @import("std");
const Io = std.Io;

fn handler(io: Io) void {
    const x = 1;
    async(io) x;
}

// error
// is_test=true
//
// :7:15: error: the operand of `async` must be a call: `async(io) f(x)`
