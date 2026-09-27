const std = @import("std");
const Io = std.Io;

fn sleepAndAdd(io: Io, x: u32) Io.Cancelable!u32 {
    try io.sleep(.fromMilliseconds(60), .awake);
    return x + 1;
}

fn frame(io: Io, seen: *bool) error{Boom}!u32 {
    // A binding may not be used as a value, so the task is simply left live:
    // the error exit runs its `errdefer`, which cancels it and joins.
    const a = async(io) sleepAndAdd(io, 1);
    seen.* = true;
    return error.Boom;
}

test "an error exit cancels before it joins" {
    const io = std.testing.io;
    const start = Io.Clock.Timestamp.now(io, .awake);
    var seen = false;
    try std.testing.expectError(error.Boom, frame(io, &seen));
    try std.testing.expect(seen);
    // The frame did not wait out the task's sleep: `errdefer` cancelled it.
    const elapsed = start.untilNow(io);
    try std.testing.expect(elapsed.toMilliseconds() < 50);
}

// run
// is_test=true
// target=native
//
