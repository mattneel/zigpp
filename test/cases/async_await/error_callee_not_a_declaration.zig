const std = @import("std");
const Io = std.Io;

fn get() u32 {
    return 1;
}

fn handler(io: Io) void {
    async(io) get().g(2);
}

// error
// is_test=true
//
// :10:15: error: the callee of `async` must name a function; bind the receiver or the function first
