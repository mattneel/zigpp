const std = @import("std");
const Io = std.Io;

fn add(x: u32) u32 {
    return x + 1;
}

fn sleepAndAdd(io: Io, x: u32) Io.Cancelable!u32 {
    try io.sleep(.fromMilliseconds(100), .awake);
    return x + 1;
}

test "cancel" {
    const io = std.testing.io;

    // `cancel` asks the task to stop and then joins: the task's
    // `error.Canceled` comes back as the expression's value.
    const a = async(io) sleepAndAdd(io, 1);
    try std.testing.expectError(error.Canceled, cancel a);

    // A cancelled binding is consumed, so its disposal does nothing.
    const b = async(io) add(2);
    _ = cancel b;
}

// run
// is_test=true
// target=native
//
