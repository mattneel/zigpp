//! Runs the kernels of `.metallib` files on the GPU of a Mac through `std.gpu.metal`.
//!
//! The kernels are the vector add and the reduction that `kernels.zig` of this directory writes for
//! NVIDIA and AMD GPUs, as the two compilers that can build them for an Apple GPU produce them:
//!
//! * `metal_kernels.metal` has the same kernels in the Metal Shading Language, which Apple's own
//!   compiler builds on a Mac that has the Metal toolchain of Xcode:
//!
//!   ```sh
//!   xcrun metal -c -O2 metal_kernels.metal -o metal_kernels.air
//!   xcrun metallib metal_kernels.air -o reference.metallib
//!   ```
//!
//! * Zig++ itself compiles `kernels.zig` for the `air64-macos` target into a `.metallib`. Until
//!   that target is finished, the library that the spike of the `metal-spike` branch builds with
//!   `test/standalone/metal_spike/run.sh` is the same kernels by the same route: Zig++ to bitcode,
//!   the AIR rewrites, the downgrade to the bitcode format that Apple's reader wants, and the
//!   container writer.
//!
//! The program is cross-compiled from Linux, where there is no Metal framework and no macOS SDK,
//! and runs on the Mac:
//!
//! ```sh
//! zig build-exe metal_host.zig -target aarch64-macos -lc -femit-bin=metal_host
//! ./metal_host reference.metallib zigpp.metallib
//! ```
//!
//! On a machine without the Metal framework -- the Linux machine that cross-compiles it -- every
//! test is skipped, and so is a run that was given no `.metallib` at all, which is how the build
//! step of the suite runs it: the exit status is 0 in both cases, so the program can be built and
//! run anywhere. A Mac always has the framework, so a Mac where it cannot be opened, a library that
//! does not load, and a kernel that does not agree with the CPU are failures there.
//!
//! Every test prints one line, `--- PASS:`, `--- FAIL:` or `--- SKIP:`, and the last line is the
//! summary, which starts with `PASS` or `FAIL`; the exit status is 1 when a test failed. A kernel
//! that a library does not have is skipped rather than failed, so a library that holds one of the
//! two kernels is still worth running.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const metal = std.gpu.metal;

/// The start of every line that is not a test verdict.
const prefix = "metal host: ";

/// The largest `.metallib` that this program reads. The libraries of the kernels are a few tens of
/// kilobytes; this only stops a wrong path (a directory, a disk image) from being read whole.
const max_metallib_size = 64 * 1024 * 1024;

/// The number of `f32` of the `vadd` test: one thread per element.
const vadd_len = 4096;

/// The number of threads of a threadgroup of the `vadd` test.
const vadd_block: u32 = 64;

/// The kernel `kernels.zig` and `metal_kernels.metal` both write: `c[i] = a[i] + b[i]`.
const vadd_name = "vadd";

/// The number of threads of a threadgroup of the `reduce` test, and the number of threadgroups.
const reduce_block: u32 = 256;
const reduce_groups: u32 = 4;
const reduce_len = reduce_block * reduce_groups;

/// The kernel of the reduction, which sums `reduce_block` values per threadgroup into `out` and
/// counts the threadgroups that ran in `counter`.
const reduce_name = "reduce";

/// The kernel of `metal_kernels.metal` that takes scalars, which the host binds with
/// `setBytes:length:atIndex:` at the index of the parameter: `x[i] *= factor` for the first
/// `count` elements. The kernels of the Metal Shading Language that Zig++ compiles take their
/// scalars the same way.
const scale_name = "scale";

/// The library of the reference kernels has no kernel of a name that the tests above use, so the
/// name of this one is not in any library: it is what the test of a missing function asks for.
const missing_name = "no_such_kernel";

/// The output of a run: one line for each test and the counts of the summary.
const Report = struct {
    out: *Io.Writer,
    passes: usize = 0,
    failures: usize = 0,
    skips: usize = 0,

    /// A test that ran and agreed with the CPU reference.
    fn pass(r: *Report, comptime format: []const u8, args: anytype) !void {
        r.passes += 1;
        try r.out.print("--- PASS: " ++ format ++ "\n", args);
    }

    /// A test that ran and disagreed, or a failure of the harness itself: a `.metallib` that does
    /// not load, a pipeline or a buffer that the framework refused. The line names the test, and
    /// what the framework said about a failure where it said anything.
    fn fail(r: *Report, comptime format: []const u8, args: anytype) !void {
        r.failures += 1;
        try r.out.print("--- FAIL: " ++ format ++ "\n", args);
    }

    /// A test that did not run: a library without the kernel, or a machine without the framework.
    fn skip(r: *Report, comptime format: []const u8, args: anytype) !void {
        r.skips += 1;
        try r.out.print("--- SKIP: " ++ format ++ "\n", args);
    }

    /// The last line of the run: `PASS` or `FAIL`, then the counts, so that the program that runs
    /// this one can grep for one word and still see a skip.
    fn summary(r: *Report) !void {
        try r.out.print(prefix ++ "{s}: {d} passed, {d} failed, {d} skipped\n", .{
            if (r.failures == 0) "PASS" else "FAIL",
            r.passes,
            r.failures,
            r.skips,
        });
    }
};

/// The driver of a run and the log buffer of its failures.
const Run = struct {
    driver: metal.Driver,
    /// What the framework said about a failure that it reported with an `NSError`: the domain, the
    /// code, and its message. Filled by every call that failed, and printed by the test that made
    /// the call.
    log: [512]u8 = @splat(0),

    /// What the framework last said about a failure, as a slice of `log`.
    fn message(run: *const Run) []const u8 {
        return std.mem.sliceTo(run.log[0..], 0);
    }

    /// A context of the GPU of this machine.
    fn context(run: *const Run) metal.Error!metal.Context {
        return run.driver.device().createContext();
    }
};

/// The bits of an `f32`. Every check of this program compares bits: a result that is one unit in
/// the last place off is off, and a NaN of the wrong sign is off as well.
fn floatBits(value: f32) u32 {
    return @bitCast(value);
}

/// The function of a library by name, with the report line of a library that does not have it.
/// Returns null when the library has no such function, and the caller reports a skip.
fn findFunction(module: metal.Module, report: *Report, name: [:0]const u8) !?metal.Function {
    return module.function(name) catch |err| switch (err) {
        error.FunctionNotFound => {
            try report.skip("{s}: the library has no function of that name", .{name});
            return null;
        },
        else => |other| {
            try report.fail("{s}: the library has no function to ask for: {s}", .{ name, @errorName(other) });
            return null;
        },
    };
}

/// The pipeline state of a function, with the report line of a kernel that the Metal compiler
/// refused. Returns null when the kernel did not compile.
fn compilePipeline(function: metal.Function, report: *Report, run: *Run, name: [:0]const u8) !?metal.Pipeline {
    return function.pipeline(.{ .error_log = &run.log }) catch |err| {
        try report.fail("{s}: the Metal compiler refused the kernel: {s}: {s}", .{ name, @errorName(err), run.message() });
        return null;
    };
}

/// `vadd`: one thread per element of three `vadd_len`-element buffers of `f32`, in threadgroups of
/// `vadd_block` threads.
///
/// The inputs are small integers, so every input and every sum is exactly representable as an
/// `f32` and the comparison is bit for bit: a kernel that rounded, fused or reordered anything
/// shows up as a mismatch.
fn testVadd(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, vadd_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, vadd_name)) orelse return;
    defer pipeline.release();

    var a: [vadd_len]f32 = undefined;
    var b: [vadd_len]f32 = undefined;
    var c: [vadd_len]f32 = @splat(0);
    for (&a, &b, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i % 13);
        y.* = @floatFromInt(i % 7);
    }

    // The buffers are the kernel arguments in order: a, b and c are the buffers 0, 1 and 2. The
    // thread position is not one of them: the dispatch supplies it.
    const buffer_a = try context.alloc(f32, vadd_len);
    defer buffer_a.free();
    const buffer_b = try context.alloc(f32, vadd_len);
    defer buffer_b.free();
    const buffer_c = try context.alloc(f32, vadd_len);
    defer buffer_c.free();
    buffer_a.copyFromHost(&a);
    buffer_b.copyFromHost(&b);

    try pipeline.launch(metal.LaunchConfig.linear(vadd_len, vadd_block), .{ buffer_a, buffer_b, buffer_c });
    try context.synchronize();
    buffer_c.copyToHost(&c);

    var mismatches: usize = 0;
    var first: usize = 0;
    for (c, a, b, 0..) |actual, x, y, i| {
        if (floatBits(actual) != floatBits(x + y)) {
            if (mismatches == 0) first = i;
            mismatches += 1;
        }
    }
    if (mismatches != 0) {
        return report.fail(
            "vadd: element {d}: expected {d} (0x{x:0>8}), got {d} (0x{x:0>8}); {d} of {d} elements differ",
            .{
                first,    a[first] + b[first], floatBits(a[first] + b[first]),
                c[first], floatBits(c[first]), mismatches,
                vadd_len,
            },
        );
    }
    try report.pass("vadd: {d} f32 in {d} threadgroups of {d} threads, c[i] == a[i] + b[i] (simd width {d})", .{
        vadd_len, (vadd_len + vadd_block - 1) / vadd_block, vadd_block, pipeline.threadExecutionWidth(),
    });
}

/// `reduce`: `reduce_groups` threadgroups of `reduce_block` threads over `reduce_len` values, with
/// the partial sums of a threadgroup in threadgroup memory and a device-scope atomic counter.
///
/// The inputs are the small integers 0 to 15, which makes every partial sum and every threadgroup
/// total exact in `f32` in any order: the reference is the plain sum of the values of the
/// threadgroup, and the comparison is bit for bit.
fn testReduce(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, reduce_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, reduce_name)) orelse return;
    defer pipeline.release();

    var input: [reduce_len]f32 = undefined;
    for (&input, 0..) |*value, i| value.* = @floatFromInt(i % 16);

    var expected: [reduce_groups]f32 = undefined;
    for (&expected, 0..) |*total, group| {
        total.* = 0;
        for (input[group * reduce_block ..][0..reduce_block]) |value| total.* += value;
    }

    // The counter starts at zero, and every threadgroup adds one to it, so the four threadgroups of
    // the dispatch must leave four. The dispatch is the same one that writes the sums, so a counter
    // that is still zero means that the kernel did not run at all.
    var out: [reduce_groups]f32 = @splat(0);
    var counters: [1]u32 = @splat(0);

    // The kernel arguments in order: inbuf, out and counter are the buffers 0, 1 and 2.
    const input_buffer = try context.alloc(f32, reduce_len);
    defer input_buffer.free();
    const out_buffer = try context.alloc(f32, reduce_groups);
    defer out_buffer.free();
    const counter_buffer = try context.alloc(u32, 1);
    defer counter_buffer.free();
    input_buffer.copyFromHost(&input);
    out_buffer.copyFromHost(&out);
    counter_buffer.copyFromHost(&counters);

    try pipeline.launch(metal.LaunchConfig.linear(reduce_len, reduce_block), .{
        input_buffer,
        out_buffer,
        counter_buffer,
    });
    try context.synchronize();
    out_buffer.copyToHost(&out);
    counter_buffer.copyToHost(&counters);

    var mismatches: usize = 0;
    var first: usize = 0;
    for (out, expected, 0..) |actual, want, group| {
        if (floatBits(actual) != floatBits(want)) {
            if (mismatches == 0) first = group;
            mismatches += 1;
        }
    }
    if (mismatches != 0) {
        return report.fail(
            "reduce: out[{d}]: expected {d} (0x{x:0>8}), got {d} (0x{x:0>8}); {d} of {d} sums differ; counter {d}, expected {d}",
            .{
                first,         expected[first],       floatBits(expected[first]),
                out[first],    floatBits(out[first]), mismatches,
                reduce_groups, counters[0],           reduce_groups,
            },
        );
    }
    if (counters[0] != reduce_groups) {
        return report.fail("reduce: the counter is {d}, expected {d}: every threadgroup adds one", .{
            counters[0], reduce_groups,
        });
    }
    try report.pass("reduce: {d} threadgroups of {d} threads over {d} f32, every sum {d}, counter {d}", .{
        reduce_groups, reduce_block, reduce_len, expected[0], counters[0],
    });
}

/// `scale`: the kernel of the reference library that takes scalars, which the host binds with
/// `setBytes:length:atIndex:` at the index of each parameter, in the order the kernel declares
/// them.
fn testScale(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, scale_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, scale_name)) orelse return;
    defer pipeline.release();

    const len = 8;
    const factor: f32 = 2.5;
    var values: [len]f32 = undefined;
    for (&values, 0..) |*value, i| value.* = @floatFromInt(i + 1);

    const buffer = try context.alloc(f32, len);
    defer buffer.free();
    buffer.copyFromHost(&values);

    // The scalar parameters are the arguments 1 and 2: the buffer is 0, `factor` is 1 and `count`
    // is 2, the places the kernel declares them in.
    try pipeline.launch(metal.LaunchConfig.linear(len, len), .{
        buffer,
        factor,
        @as(u32, len),
    });
    try context.synchronize();
    buffer.copyToHost(&values);

    for (values, 0..) |actual, i| {
        const expected = @as(f32, @floatFromInt(i + 1)) * factor;
        if (floatBits(actual) != floatBits(expected)) {
            return report.fail("scale: element {d}: expected {d}, got {d}", .{ i, expected, actual });
        }
    }
    try report.pass("scale: {d} f32 scaled by {d} through setBytes:length:atIndex:", .{ len, factor });
}

/// The bytes that are not a `.metallib`, for the test of a library that the framework cannot read.
const not_a_library = "these are not the bytes of the container of a library of kernels";

/// The error paths: a library that the framework cannot read, a kernel name that a library that it
/// can read does not have, and a launch whose threadgroup is larger than the kernel allows. Every
/// one of them must come back as an error, and none of them may crash the program.
fn testErrors(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    // A library of bytes that are not a `.metallib` reports an error, and what the framework said
    // about it goes into the log.
    var log: [512]u8 = @splat(0);
    if (context.loadModule(not_a_library, .{ .error_log = &log })) |library| {
        library.unload();
        return report.fail("errors: {d} bytes that are not a metallib loaded as a library", .{not_a_library.len});
    } else |err| switch (err) {
        error.InvalidLibrary, error.InternalError => {
            try report.pass("errors: bytes that are not a metallib report {s}: {s}", .{
                @errorName(err), std.mem.sliceTo(log[0..], 0),
            });
        },
        else => |other| return report.fail("errors: bytes that are not a metallib report {s}", .{@errorName(other)}),
    }

    // A library that loaded is one that the framework can read, so a name that it does not have is
    // the report of a kernel that the program asked for and the library does not hold.
    if (module.function(missing_name)) |function| {
        function.release();
        return report.fail("errors: the library has a function named {s}", .{missing_name});
    } else |err| switch (err) {
        error.FunctionNotFound => try report.pass("errors: a missing function name reports FunctionNotFound", .{}),
        else => |other| return report.fail("errors: a missing function name reports {s}", .{@errorName(other)}),
    }

    // A threadgroup larger than the kernel allows is rejected before anything is dispatched, so
    // that the framework, which reports such a dispatch only from a command buffer that failed,
    // never sees it.
    const function = (try findFunction(module, report, vadd_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, vadd_name)) orelse return;
    defer pipeline.release();

    const buffer_a = try context.alloc(f32, 1);
    defer buffer_a.free();
    const buffer_b = try context.alloc(f32, 1);
    defer buffer_b.free();
    const buffer_c = try context.alloc(f32, 1);
    defer buffer_c.free();

    const largest = pipeline.maxTotalThreadsPerThreadgroup();
    const block = largest * 2;
    if (pipeline.launch(.{ .grid = .{ .x = 1 }, .block = .{ .x = block } }, .{ buffer_a, buffer_b, buffer_c })) |_| {
        return report.fail("errors: a threadgroup of {d} threads was dispatched", .{block});
    } else |err| switch (err) {
        error.InvalidValue => try report.pass(
            "errors: a threadgroup of {d} threads reports InvalidValue (the kernel allows {d})",
            .{ block, largest },
        ),
        else => |other| return report.fail("errors: a threadgroup of {d} threads reports {s}", .{ block, @errorName(other) }),
    }

    // A threadgroup of no threads, which is what a block of no height makes of it, is not one
    // either.
    if (pipeline.launch(.{ .grid = .{ .x = 1 }, .block = .{ .x = 1, .y = 0 } }, .{
        buffer_a,
        buffer_b,
        buffer_c,
    })) |_| {
        return report.fail("errors: a threadgroup of no threads was dispatched", .{});
    } else |err| switch (err) {
        error.InvalidValue => try report.pass("errors: a threadgroup of no threads reports InvalidValue", .{}),
        else => |other| return report.fail("errors: a threadgroup of no threads reports {s}", .{@errorName(other)}),
    }
}

/// Reports what the GPU of the machine is, and what its framework says about the families of GPUs
/// that `std.gpu.metal` names.
fn reportDevice(run: *const Run, report: *Report) !void {
    const device = run.driver.device();

    var name_buffer: [256]u8 = undefined;
    const name = try device.name(&name_buffer);
    const largest = device.maxThreadsPerThreadgroup();

    // The families that this device is of, in the order of the enumeration. A device is of a
    // family and of the families before it on its line, so this lists several of them.
    var families: [256]u8 = undefined;
    var at: usize = 0;
    for (std.enums.values(metal.Family)) |family| {
        if (!device.supportsFamily(family)) continue;
        const family_name = @tagName(family);
        if (at + family_name.len + 1 >= families.len) break;
        @memcpy(families[at..][0..family_name.len], family_name);
        at += family_name.len;
        families[at] = ' ';
        at += 1;
    }

    try report.pass("device: {s}, at most {d}x{d}x{d} threads per threadgroup, families: {s}", .{
        name, largest.x, largest.y, largest.z, families[0..at],
    });
}

/// Runs the tests of the kernels of one `.metallib` on the GPU of this machine.
/// Runs the tests of the kernels of one `.metallib`, and returns whether all of them passed: a
/// library that does not load, and a kernel that does not agree with the CPU, are failures of the
/// run, not of the test of one library, so the caller is what reports the status.
fn runLibrary(out: *Io.Writer, run: *Run, path: []const u8, bytes: []const u8) !bool {
    var report: Report = .{ .out = out };

    const context = run.context() catch |err| {
        try report.fail("{s}: no context of the GPU: {s}", .{ path, @errorName(err) });
        try report.summary();
        return false;
    };
    defer context.release();

    const module = context.loadModule(bytes, .{ .error_log = &run.log }) catch |err| {
        try report.fail("{s}: the framework did not load the library: {s}: {s}", .{
            path, @errorName(err), run.message(),
        });
        try report.summary();
        return false;
    };
    defer module.unload();

    try out.print(prefix ++ "{s}: {d} bytes\n", .{ path, bytes.len });
    try testErrors(run, &report, context, module);
    try testVadd(run, &report, context, module);
    try testReduce(run, &report, context, module);
    try testScale(run, &report, context, module);
    try report.summary();
    return report.failures == 0;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();

    var stdout_writer = Io.File.stdout().writerStreaming(init.io, &.{});
    const out = &stdout_writer.interface;

    var report: Report = .{ .out = out };

    // The build step of the suite runs this program with the libraries that `-Dmetallib` names, and
    // with none of them by default: a run that was given no library has nothing to test, which is a
    // skip rather than a failure of the step.
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        try report.skip("no .metallib was given: pass the libraries to run, as in `{s} kernels.metallib`", .{args[0]});
        try report.summary();
        return;
    }

    var driver = metal.Driver.open() catch |err| {
        // The Metal framework is part of macOS, so a Mac where it cannot be opened is a machine to
        // fix, and this is a failure there. Any other operating system has nothing to run here,
        // which is how the Linux machine that cross-compiles this program is smoke-tested: its
        // tests are skipped, and the step that runs it passes.
        if (builtin.os.tag.isDarwin()) {
            try report.fail("no Metal framework: {s}", .{@errorName(err)});
            try report.summary();
            std.process.exit(1);
        }
        for (args[1..]) |path| {
            try report.skip("{s}: no Metal framework on this machine ({s})", .{ path, @errorName(err) });
        }
        try report.summary();
        return;
    };
    defer driver.close();

    var run: Run = .{ .driver = driver };
    try reportDevice(&run, &report);

    var failed = false;

    // Every library is run, whether an earlier one failed or not, and the status of the program is
    // the status of the run: a library that cannot be read, one that the framework refuses, and a
    // kernel that does not agree with the CPU all make it 1.
    for (args[1..]) |path| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, path, arena, .limited(max_metallib_size)) catch |err| {
            try report.fail("{s}: cannot read the file: {s}", .{ path, @errorName(err) });
            continue;
        };
        if (!try runLibrary(out, &run, path, bytes)) failed = true;
    }
    if (report.failures != 0) {
        // A library that could not be read, which is the failure of this report and not of the
        // tests of one library, gets a summary line of its own.
        try report.summary();
        failed = true;
    }
    if (failed) std.process.exit(1);
}
