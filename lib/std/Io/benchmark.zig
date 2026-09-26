//! Throughput and latency of the `Io` task primitives, on the `Io.Threaded` or the `Io.Threadz`
//! of this tree.
//!
//!     zig run -OReleaseFast lib/std/Io/benchmark.zig -- [--io threaded|threadz] [--threads N]
//!         [--filter NAME] [--runs N]
//!
//! `--threads` is the number of threads the implementation may add to the calling one: the
//! `async_limit` of an `Io.Threaded`, the `thread_limit` of an `Io.Threadz`. The default is each
//! implementation's own, one less than the number of CPUs. Each benchmark runs `--runs` times (3
//! by default) and reports the best run.
//!
//! The tasks do a few dozen nanoseconds of work each, so what is measured is the cost of spawning,
//! scheduling, waking and joining them, not the work.

const builtin = @import("builtin");
const std = @import("std");
const Io = std.Io;

const Benchmark = struct {
    name: []const u8,
    /// What one operation is, for the report.
    unit: []const u8,
    /// Operations in one run, in a release build. Debug builds run 1/64 of it.
    ops: usize,
    run: *const fn (io: Io, ops: usize) anyerror!void,
};

const benchmarks = [_]Benchmark{
    .{ .name = "group-spawn", .unit = "task", .ops = 1 << 20, .run = groupSpawn },
    .{ .name = "fan-in", .unit = "task", .ops = 1 << 20, .run = fanIn },
    .{ .name = "fork-join", .unit = "task", .ops = 1 << 17, .run = forkJoin },
    .{ .name = "spawn-chain", .unit = "handoff", .ops = 1 << 18, .run = spawnChain },
    .{ .name = "async-await", .unit = "round trip", .ops = 1 << 17, .run = asyncAwait },
    .{ .name = "queue-ping-pong", .unit = "round trip", .ops = 1 << 17, .run = queuePingPong },
};

/// A few dozen nanoseconds of work that the optimizer cannot remove.
fn work(seed: usize) void {
    var x: u64 = seed | 1;
    for (0..32) |_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    std.mem.doNotOptimizeAway(x);
}

/// The main thread spawns every task into one group, then awaits the group.
fn groupSpawn(io: Io, ops: usize) !void {
    var group: Io.Group = .init;
    for (0..ops) |i| group.async(io, work, .{i});
    try group.await(io);
}

/// A quarter as many spawners as there are workers, each spawning its share of the tasks into one
/// group at the same time: the contention of spawning from several threads at once, with workers
/// left over to run what they spawn.
fn fanIn(io: Io, ops: usize) !void {
    const spawners = spawnerCount(io);
    var group: Io.Group = .init;
    var spawner_group: Io.Group = .init;
    for (0..spawners) |i| {
        const share = ops / spawners + @intFromBool(i < ops % spawners);
        spawner_group.async(io, spawnShare, .{ io, &group, share });
    }
    try spawner_group.await(io);
    try group.await(io);
}

fn spawnShare(io: Io, group: *Io.Group, share: usize) void {
    for (0..share) |i| group.async(io, work, .{i});
}

fn spawnerCount(io: Io) usize {
    _ = io;
    return @max(1, @min(64, added_threads / 4));
}

/// The threads the implementation under test may add to the calling one.
var added_threads: usize = undefined;

/// A binary tree of tasks: every inner node spawns its two children into a group of its own and
/// awaits it. Tasks spawn tasks from the workers, and inner nodes block in `await`.
fn forkJoin(io: Io, ops: usize) !void {
    const depth = std.math.log2_int(usize, ops);
    try forkJoinNode(io, depth);
}

fn forkJoinNode(io: Io, depth: usize) Io.Cancelable!void {
    if (depth == 0) return work(depth);
    var group: Io.Group = .init;
    group.async(io, forkJoinNode, .{ io, depth - 1 });
    group.async(io, forkJoinNode, .{ io, depth - 1 });
    try group.await(io);
}

/// Each task spawns the next one into the same group and returns: the latency of handing a task
/// from the worker that spawned it to the worker that runs it.
fn spawnChain(io: Io, ops: usize) !void {
    var group: Io.Group = .init;
    group.async(io, chainLink, .{ io, &group, ops });
    try group.await(io);
}

fn chainLink(io: Io, group: *Io.Group, remaining: usize) void {
    work(remaining);
    if (remaining > 1) group.async(io, chainLink, .{ io, group, remaining - 1 });
}

/// `io.async` then `await`, one at a time, from the main thread.
fn asyncAwait(io: Io, ops: usize) !void {
    for (0..ops) |i| {
        var future = io.async(workResult, .{i});
        std.mem.doNotOptimizeAway(future.await(io));
    }
}

fn workResult(seed: usize) usize {
    work(seed);
    return seed;
}

/// Two concurrent tasks pass a counter back and forth through two single-slot queues: the latency
/// of parking one task and waking another.
fn queuePingPong(io: Io, ops: usize) !void {
    var ping_buffer: [1]usize = undefined;
    var pong_buffer: [1]usize = undefined;
    var ping: Io.Queue(usize) = .init(&ping_buffer);
    var pong: Io.Queue(usize) = .init(&pong_buffer);
    var ponger = try io.concurrent(pongLoop, .{ io, &ping, &pong, ops });
    for (0..ops) |i| {
        try ping.putOne(io, i);
        std.debug.assert(try pong.getOne(io) == i);
    }
    try ponger.await(io);
}

fn pongLoop(io: Io, ping: *Io.Queue(usize), pong: *Io.Queue(usize), ops: usize) !void {
    for (0..ops) |_| try pong.putOne(io, try ping.getOne(io));
}

fn now(io: Io) i96 {
    return Io.Clock.awake.now(io).nanoseconds;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    var implementation: enum { threaded, threadz } = .threaded;
    var threads: ?usize = null;
    var filter: ?[]const u8 = null;
    var runs: usize = 3;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (i + 1 == args.len) usage();
        if (std.mem.eql(u8, arg, "--io")) {
            i += 1;
            implementation = std.meta.stringToEnum(@TypeOf(implementation), args[i]) orelse usage();
        } else if (std.mem.eql(u8, arg, "--threads")) {
            i += 1;
            threads = std.fmt.parseUnsigned(usize, args[i], 10) catch usage();
        } else if (std.mem.eql(u8, arg, "--filter")) {
            i += 1;
            filter = args[i];
        } else if (std.mem.eql(u8, arg, "--runs")) {
            i += 1;
            runs = @max(1, std.fmt.parseUnsigned(usize, args[i], 10) catch usage());
        } else usage();
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = Io.File.stdout().writer(init.io, &stdout_buffer);
    const out = &stdout_writer.interface;
    switch (implementation) {
        .threaded => {
            var threaded: Io.Threaded = .init(std.heap.smp_allocator, .{
                .async_limit = if (threads) |n| .limited(n) else null,
            });
            defer threaded.deinit();
            added_threads = @backingInt(threaded.async_limit);
            try out.print("Io.Threaded, async_limit {d}, {s}, best of {d}\n", .{
                added_threads, @tagName(builtin.mode), runs,
            });
            try runAll(threaded.io(), out, filter, runs);
        },
        .threadz => {
            if (Io.Threadz == void) usage();
            var threadz: Io.Threadz = undefined;
            try threadz.init(std.heap.smp_allocator, .{
                .backing_allocator_needs_mutex = false,
                .thread_limit = threads,
            });
            defer threadz.deinit();
            added_threads = threads orelse (std.Thread.getCpuCount() catch 1) - 1;
            try out.print("Io.Threadz, thread_limit {d}, {s}, best of {d}\n", .{
                added_threads, @tagName(builtin.mode), runs,
            });
            try runAll(threadz.io(), out, filter, runs);
        },
    }
}

fn runAll(io: Io, out: *Io.Writer, filter: ?[]const u8, runs: usize) !void {
    try out.flush();

    for (benchmarks) |b| {
        if (filter) |f| if (std.mem.indexOf(u8, b.name, f) == null) continue;
        const ops = if (builtin.mode == .debug) @max(1, b.ops / 64) else b.ops;
        var best: i96 = std.math.maxInt(i96);
        for (0..runs) |_| {
            const start = now(io);
            try b.run(io, ops);
            best = @min(best, now(io) - start);
        }
        const ns_per_op = @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(ops));
        try out.print("{s:<16} {d:>9} {s}s  {d:>9.1} ns/{s}  {d:>12.0} {s}s/s\n", .{
            b.name, ops, b.unit, ns_per_op, b.unit, 1e9 / ns_per_op, b.unit,
        });
        try out.flush();
    }
}

fn usage() noreturn {
    std.debug.print("usage: benchmark [--io threaded|threadz] [--threads N] [--filter NAME] [--runs N]\n", .{});
    std.process.exit(1);
}
