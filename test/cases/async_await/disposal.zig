const std = @import("std");
const Io = std.Io;

fn sleepAndAdd(io: Io, x: u32, finished: *std.atomic.Value(bool)) Io.Cancelable!u32 {
    try io.sleep(.fromMilliseconds(60), .awake);
    finished.store(true, .monotonic);
    return x + 1;
}

fn frame(io: Io, seen: *bool, finished: *std.atomic.Value(bool)) !u32 {
    // A binding may not be used as a value, so the task is simply left live;
    // the error exit runs its `errdefer`, which cancels it and joins. The task
    // is spawned with `concurrent`, which never runs the call inline, so the
    // cancel always arrives before the sleep is over.
    const a = try concurrent(io) sleepAndAdd(io, 1, finished);
    seen.* = true;
    return error.Boom;
}

test "an error exit cancels before it joins" {
    const io = std.testing.io;

    var seen = false;
    var finished: std.atomic.Value(bool) = .init(false);
    try std.testing.expectError(error.Boom, frame(io, &seen, &finished));

    try std.testing.expect(seen);
    // The frame did not wait out the task's sleep: its `errdefer` cancelled it,
    // so the task never reached the store after the sleep.
    try std.testing.expect(!finished.load(.monotonic));
}

// run
// is_test=true
// target=native
//
