const std = @import("std");
const Io = std.Io;

fn log() void {}

const a = async(io) log();

// error
// is_test=true
//
// :6:11: error: a task needs a frame to join it; spawn inside a function body
