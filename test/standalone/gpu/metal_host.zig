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
//! `metal_kernels.zig` also has kernels that only Zig++ builds, because what they test is what it
//! does to a module before Apple's compiler sees it: the checked `usize` multiply of a Debug build
//! and the 128-bit products and overflow flags of 64-bit multiplies (issue #18), the overflow
//! flags of adds and subtracts at every width (issue #26), and constant
//! data: tables of integers and of structs, a table of strings, `@errorName`,
//! `std.fmt.parseFloat`, and an allocator's vtable (issue #22).
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
//! that a library does not have is skipped rather than failed, so a library that holds some of the
//! kernels is still worth running.

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

/// The kernel of an index computed with a `usize` multiply, a checked one in a Debug build:
/// `out[row * cols + col] = row * cols + col`, over `index2d_rows` threadgroups of `index2d_cols`
/// threads.
const index2d_name = "index2d";
const index2d_rows: u32 = 8;
const index2d_cols: u32 = 32;

/// The kernel of the multiplies of 64-bit integers that need the high half of the product.
const mulwide_name = "mulwide";
const mulwide_block: u32 = 64;

/// The pairs of the `mulwide` test that sit on the edges of the unsigned and the signed
/// overflow checks; pseudo-random pairs follow them.
const mulwide_edges = [_][2]u64{
    .{ 0, 0 },
    .{ 5, 0 }, // a product of zero never overflows
    .{ 0, 5 },
    .{ 1, 1 },
    .{ 3, 1 << 63 }, // wraps to 2^63: an unsigned overflow that is not below either operand
    .{ 1 << 32, 1 << 32 }, // exactly 2^64
    .{ (1 << 32) - 1, (1 << 32) + 1 }, // 2^64 - 1: the largest product that fits
    .{ std.math.maxInt(u64), std.math.maxInt(u64) }, // signed: -1 * -1
    .{ std.math.maxInt(u64), 1 },
    .{ 1 << 62, 2 }, // 2^63: a signed overflow only
    .{ @bitCast(@as(i64, -(1 << 62))), 2 }, // -2^63: fits in an i64
    .{ 1 << 63, std.math.maxInt(u64) }, // signed: minInt(i64) * -1
    .{ 1 << 63, 1 }, // signed: minInt(i64) * 1
    .{ 0x123456789abcdef0, 0x0fedcba987654321 },
    .{ 0xffffffff00000000, 0x00000000ffffffff },
    .{ 0x8000000080000000, 0x7fffffff7fffffff },
};
const mulwide_len = 128;

/// The kernel of the overflow flags of adds and subtracts, and a copy of what it computes: two
/// bits per type (u64, i64, u32, i32, u16, i16, u8, i8), the flag of the sum and then the flag of
/// the difference. It runs on the pairs of `mulwide`.
const addsub_name = "addsub";
fn addSubFlags(x: u64, y: u64) u32 {
    var flags: u32 = 0;
    inline for (.{ u64, i64, u32, i32, u16, i16, u8, i8 }, 0..) |T, k| {
        const U = @Int(.unsigned, @bitSizeOf(T));
        const a: T = @bitCast(@as(U, @truncate(x)));
        const b: T = @bitCast(@as(U, @truncate(y)));
        const sum = @addWithOverflow(a, b);
        const difference = @subWithOverflow(a, b);
        flags |= (@as(u32, sum[1]) | @as(u32, difference[1]) << 1) << (2 * k);
    }
    return flags;
}

/// The kernel of program-scope constant tables, the tables themselves and the function of them
/// that it computes (copies of those of `metal_kernels.zig`, which this program cannot import:
/// its kernels are exported for the GPU), and its number of threads.
const constant_tables_name = "constant_tables";
const table_primes = [_]u32{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53 };
const table_wide = [_]u64{
    0x0123456789abcdef, 0xfedcba9876543210, 0x8000000000000001, 0xffffffffffffffff,
    1,                  0,                  0x00000000ffffffff, 0xffffffff00000000,
};
const Step = struct { scale: u32, bias: u32, shift: u8 };
const table_steps = [_]Step{
    .{ .scale = 3, .bias = 7, .shift = 1 },
    .{ .scale = 5, .bias = 0, .shift = 0 },
    .{ .scale = 0xffff, .bias = 0xdeadbeef, .shift = 7 },
    .{ .scale = 1, .bias = 1, .shift = 31 },
    .{ .scale = 12345, .bias = 678, .shift = 3 },
};
fn tableValue(i: u32) u32 {
    const wide = table_wide[i % table_wide.len];
    const step = table_steps[i % table_steps.len];
    const mixed = table_primes[i % table_primes.len] *% @as(u32, @truncate(wide >> @intCast(i % 64)));
    return (mixed +% step.scale *% i +% step.bias) >> @intCast(step.shift);
}
const constant_tables_len = 256;

/// The kernel of the table of strings, and a copy of the table.
const string_table_name = "string_table";
const table_words = [_][]const u8{ "zig", "plus", "plus", "metal", "", "constant", "address", "space" };
fn wordValue(i: u32) u32 {
    const word = table_words[i % table_words.len];
    var hash: u32 = 0;
    for (word) |byte| hash = hash *% 31 +% byte;
    return hash +% table_primes[i % table_primes.len] *% @as(u32, @intCast(word.len));
}

/// The kernel of `@errorName`, and the names of its errors in their order.
const error_names_name = "error_names";
const kernel_error_names = [_][]const u8{ "OutOfMemory", "InvalidCharacter", "Overflow", "EndOfStream" };
fn nameValue(name: []const u8) u32 {
    var hash: u32 = 0;
    for (name) |byte| hash = hash *% 31 +% byte;
    return hash +% (@as(u32, @intCast(name.len)) << 24);
}

/// The kernel of allocations through `std.mem.Allocator`, and what it computes.
const allocator_vtable_name = "allocator_vtable";
fn squareSum(n: u32) u32 {
    var sum: u32 = 0;
    for (0..n) |k| sum += @intCast(k * k);
    return sum;
}

/// The number of threads of the tests of constant data above.
const constant_data_len = 256;

/// The kernel of `std.fmt.parseFloat(f32, ...)` on the GPU, and the texts it parses: the edges
/// of `f32`, the paths of the parser (the fast path, Eisel-Lemire, hex floats, the special
/// values, and the slow path, which the halfway cases with more significant digits than a u64
/// holds take, and which reads the table of the powers of five as decimal strings), and texts
/// that are not numbers.
const parse_float_name = "parse_float";
const parse_float_block: u32 = 32;
const float_texts = [_][]const u8{
    "1.5",
    "-0.0",
    "0",
    "3.14159265358979323846",
    "1e10",
    "1e-10",
    "6.02214076e23",
    "3.4028235e38", // the largest f32
    "3.4028236e38",
    "1e39", // infinity
    "1.17549435e-38", // the smallest normal f32
    "1.4e-45", // the smallest subnormal f32
    "1e-50", // zero
    "0x1.8p1",
    "inf",
    "-inf",
    "nan",
    "123456789012345678901234567890",
    "1.000000059604644775390625", // halfway between 1 and the next f32: ties to even
    "1.00000005960464477539062500000000000001", // just above halfway
    "1.00000017881393432617187499", // just below the halfway point above 1 + 2^-23
    "16777217", // halfway between 2^24 and 2^24 + 2: ties to even
    "16777217.000000000000000000001", // just above halfway
    "0.1",
    "0.2",
    "0.3",
    "7.038531e-26",
    "abc",
    "",
    "1.2.3",
    "--1",
};

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

/// `index2d`: `index2d_rows` threadgroups of `index2d_cols` threads, each writing its own index,
/// which the kernel computes with a `usize` multiply: every element must hold its index.
fn testIndex2d(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, index2d_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, index2d_name)) orelse return;
    defer pipeline.release();

    const len = index2d_rows * index2d_cols;
    var out: [len]u32 = @splat(std.math.maxInt(u32));
    const buffer = try context.alloc(u32, len);
    defer buffer.free();
    buffer.copyFromHost(&out);

    try pipeline.launch(.{
        .grid = .{ .x = index2d_rows },
        .block = .{ .x = index2d_cols },
    }, .{ buffer, index2d_cols });
    try context.synchronize();
    buffer.copyToHost(&out);

    for (out, 0..) |actual, i| {
        if (actual != i) return report.fail("index2d: element {d}: expected {d}, got {d}", .{ i, i, actual });
    }
    try report.pass("index2d: {d} threadgroups of {d} threads, out[row * cols + col] with a usize multiply", .{
        index2d_rows, index2d_cols,
    });
}

/// `mulwide`: the halves of the products of 64-bit integers and the overflow flags of their
/// checked multiplies, on the edges in `mulwide_edges` and on pseudo-random pairs, each compared
/// bit for bit with the same arithmetic on the CPU.
fn testMulwide(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, mulwide_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, mulwide_name)) orelse return;
    defer pipeline.release();

    var a: [mulwide_len]u64 = undefined;
    var b: [mulwide_len]u64 = undefined;
    widePairs(&a, &b);

    var hi: [mulwide_len]u64 = @splat(0);
    var lo: [mulwide_len]u64 = @splat(0);
    var shi: [mulwide_len]u64 = @splat(0);
    var wrapped: [mulwide_len]u64 = @splat(0);
    var flags: [mulwide_len]u32 = @splat(0);
    const buffer_a = try context.alloc(u64, mulwide_len);
    defer buffer_a.free();
    const buffer_b = try context.alloc(u64, mulwide_len);
    defer buffer_b.free();
    const buffer_hi = try context.alloc(u64, mulwide_len);
    defer buffer_hi.free();
    const buffer_lo = try context.alloc(u64, mulwide_len);
    defer buffer_lo.free();
    const buffer_shi = try context.alloc(u64, mulwide_len);
    defer buffer_shi.free();
    const buffer_wrapped = try context.alloc(u64, mulwide_len);
    defer buffer_wrapped.free();
    const buffer_flags = try context.alloc(u32, mulwide_len);
    defer buffer_flags.free();
    buffer_a.copyFromHost(&a);
    buffer_b.copyFromHost(&b);

    try pipeline.launch(metal.LaunchConfig.linear(mulwide_len, mulwide_block), .{
        buffer_a,   buffer_b,       buffer_hi,    buffer_lo,
        buffer_shi, buffer_wrapped, buffer_flags, @as(u32, mulwide_len),
    });
    try context.synchronize();
    buffer_hi.copyToHost(&hi);
    buffer_lo.copyToHost(&lo);
    buffer_shi.copyToHost(&shi);
    buffer_wrapped.copyToHost(&wrapped);
    buffer_flags.copyToHost(&flags);

    var unsigned_overflows: usize = 0;
    var signed_overflows: usize = 0;
    for (a, b, 0..) |x, y, i| {
        const product = @as(u128, x) * y;
        const sx: i64 = @bitCast(x);
        const sy: i64 = @bitCast(y);
        const signed_product = @as(i128, sx) * sy;
        const unsigned_check = @mulWithOverflow(x, y);
        const signed_check = @mulWithOverflow(sx, sy);
        const want_hi: u64 = @truncate(product >> 64);
        const want_lo: u64 = @truncate(product);
        const want_shi: u64 = @bitCast(@as(i64, @truncate(signed_product >> 64)));
        const want_flags = @as(u32, unsigned_check[1]) | @as(u32, signed_check[1]) << 1;
        unsigned_overflows += unsigned_check[1];
        signed_overflows += signed_check[1];
        if (hi[i] != want_hi or lo[i] != want_lo or shi[i] != want_shi or
            wrapped[i] != unsigned_check[0] or flags[i] != want_flags)
        {
            return report.fail(
                "mulwide: pair {d}, 0x{x} * 0x{x}: expected hi 0x{x} lo 0x{x} signed hi 0x{x} wrapped 0x{x} flags {b:0>3}, got hi 0x{x} lo 0x{x} signed hi 0x{x} wrapped 0x{x} flags {b:0>3}",
                .{ i, x, y, want_hi, want_lo, want_shi, unsigned_check[0], want_flags, hi[i], lo[i], shi[i], wrapped[i], flags[i] },
            );
        }
    }
    try report.pass("mulwide: {d} pairs of 64-bit integers, 128-bit products and checked multiplies ({d} unsigned and {d} signed overflows) bit for bit", .{
        mulwide_len, unsigned_overflows, signed_overflows,
    });
}

/// The pairs of the `mulwide` and `addsub` tests: the edges in `mulwide_edges`, then
/// pseudo-random pairs, every other one with a narrow second operand, which overflows less often.
fn widePairs(a: *[mulwide_len]u64, b: *[mulwide_len]u64) void {
    var prng: std.Random.DefaultPrng = .init(0x5eed_2026);
    const random = prng.random();
    for (a, b, 0..) |*x, *y, i| {
        if (i < mulwide_edges.len) {
            x.*, y.* = mulwide_edges[i];
        } else {
            x.* = random.int(u64);
            y.* = if (i % 2 == 0) random.int(u64) else random.int(u32);
        }
    }
}

/// `addsub`: the overflow flags of adds and subtracts at every width, signed and unsigned, of the
/// pairs of `mulwide`, compared bit for bit with the CPU.
fn testAddsub(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, addsub_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, addsub_name)) orelse return;
    defer pipeline.release();

    var a: [mulwide_len]u64 = undefined;
    var b: [mulwide_len]u64 = undefined;
    widePairs(&a, &b);
    var flags: [mulwide_len]u32 = @splat(0xdeadbeef);
    const buffer_a = try context.alloc(u64, mulwide_len);
    defer buffer_a.free();
    const buffer_b = try context.alloc(u64, mulwide_len);
    defer buffer_b.free();
    const buffer_flags = try context.alloc(u32, mulwide_len);
    defer buffer_flags.free();
    buffer_a.copyFromHost(&a);
    buffer_b.copyFromHost(&b);
    buffer_flags.copyFromHost(&flags);

    try pipeline.launch(metal.LaunchConfig.linear(mulwide_len, mulwide_block), .{
        buffer_a, buffer_b, buffer_flags, @as(u32, mulwide_len),
    });
    try context.synchronize();
    buffer_flags.copyToHost(&flags);

    var overflows: usize = 0;
    for (a, b, flags, 0..) |x, y, actual, i| {
        const want = addSubFlags(x, y);
        overflows += @popCount(want);
        if (actual != want) {
            return report.fail("addsub: pair {d}, 0x{x} and 0x{x}: expected flags 0b{b:0>16}, got 0b{b:0>16}", .{ i, x, y, want, actual });
        }
    }
    try report.pass("addsub: {d} pairs, the overflow flags of adds and subtracts of 8 to 64 bits, signed and unsigned ({d} overflows), bit for bit", .{
        mulwide_len, overflows,
    });
}

/// `constant_tables`: one thread per element, each reading tables of integers and of structs in
/// the constant address space, compared with the same reads on the CPU.
fn testConstantTables(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, constant_tables_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, constant_tables_name)) orelse return;
    defer pipeline.release();

    var out: [constant_tables_len]u32 = @splat(0);
    const buffer = try context.alloc(u32, constant_tables_len);
    defer buffer.free();
    buffer.copyFromHost(&out);

    try pipeline.launch(metal.LaunchConfig.linear(constant_tables_len, 64), .{
        buffer, @as(u32, constant_tables_len),
    });
    try context.synchronize();
    buffer.copyToHost(&out);

    for (out, 0..) |actual, i| {
        const expected = tableValue(@intCast(i));
        if (actual != expected) {
            return report.fail("constant_tables: element {d}: expected {d}, got {d}", .{ i, expected, actual });
        }
    }
    try report.pass("constant_tables: {d} reads of tables of integers and of structs in the constant address space", .{
        constant_tables_len,
    });
}

/// One thread per element of `constant_data_len` `u32`, which a kernel of constant data computes
/// from the thread's index, compared with `expected` of the index on the CPU.
fn runIndexed(
    run: *Run,
    report: *Report,
    context: metal.Context,
    module: metal.Module,
    name: [:0]const u8,
    expected: *const fn (u32) u32,
    what: []const u8,
) !void {
    const function = (try findFunction(module, report, name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, name)) orelse return;
    defer pipeline.release();

    var out: [constant_data_len]u32 = @splat(0xdeadbeef);
    const buffer = try context.alloc(u32, constant_data_len);
    defer buffer.free();
    buffer.copyFromHost(&out);

    try pipeline.launch(metal.LaunchConfig.linear(constant_data_len, 64), .{
        buffer, @as(u32, constant_data_len),
    });
    try context.synchronize();
    buffer.copyToHost(&out);

    for (out, 0..) |actual, i| {
        const want = expected(@intCast(i));
        if (actual != want) {
            return report.fail("{s}: element {d}: expected 0x{x:0>8}, got 0x{x:0>8}", .{ name, i, want, actual });
        }
    }
    try report.pass("{s}: {d} threads, {s}", .{ name, constant_data_len, what });
}

fn errorNameValue(i: u32) u32 {
    return nameValue(kernel_error_names[i % kernel_error_names.len]);
}

fn allocatorValue(i: u32) u32 {
    return squareSum(8 + i % 8);
}

/// `parse_float`: `std.fmt.parseFloat(f32, ...)` of every text of `float_texts` on the GPU,
/// compared bit for bit, errors included, with the same call on the CPU.
fn testParseFloat(run: *Run, report: *Report, context: metal.Context, module: metal.Module) !void {
    const function = (try findFunction(module, report, parse_float_name)) orelse return;
    defer function.release();
    const pipeline = (try compilePipeline(function, report, run, parse_float_name)) orelse return;
    defer pipeline.release();

    const count = float_texts.len;
    const text_len = comptime len: {
        var sum: usize = 0;
        for (float_texts) |text| sum += text.len;
        break :len sum;
    };
    var text: [text_len]u8 = undefined;
    var starts: [count]u32 = undefined;
    var lens: [count]u32 = undefined;
    var at: usize = 0;
    for (float_texts, &starts, &lens) |t, *start, *len| {
        @memcpy(text[at..][0..t.len], t);
        start.* = @intCast(at);
        len.* = @intCast(t.len);
        at += t.len;
    }

    var bits: [count]u32 = @splat(0xdeadbeef);
    var ok: [count]u32 = @splat(0xdeadbeef);
    const buffer_text = try context.alloc(u8, text_len);
    defer buffer_text.free();
    const buffer_starts = try context.alloc(u32, count);
    defer buffer_starts.free();
    const buffer_lens = try context.alloc(u32, count);
    defer buffer_lens.free();
    const buffer_bits = try context.alloc(u32, count);
    defer buffer_bits.free();
    const buffer_ok = try context.alloc(u32, count);
    defer buffer_ok.free();
    buffer_text.copyFromHost(&text);
    buffer_starts.copyFromHost(&starts);
    buffer_lens.copyFromHost(&lens);
    buffer_bits.copyFromHost(&bits);
    buffer_ok.copyFromHost(&ok);

    try pipeline.launch(metal.LaunchConfig.linear(count, parse_float_block), .{
        buffer_text, buffer_starts, buffer_lens, buffer_bits, buffer_ok, @as(u32, count),
    });
    try context.synchronize();
    buffer_bits.copyToHost(&bits);
    buffer_ok.copyToHost(&ok);

    var numbers: usize = 0;
    for (float_texts, bits, ok) |t, actual_bits, actual_ok| {
        const want_ok: u32, const want_bits: u32 = if (std.fmt.parseFloat(f32, t)) |value|
            .{ 1, floatBits(value) }
        else |_|
            .{ 0, 0 };
        numbers += want_ok;
        if (actual_ok != want_ok or actual_bits != want_bits) {
            return report.fail("parse_float: \"{s}\": expected ok {d} bits 0x{x:0>8}, got ok {d} bits 0x{x:0>8}", .{
                t, want_ok, want_bits, actual_ok, actual_bits,
            });
        }
    }
    try report.pass("parse_float: std.fmt.parseFloat(f32) of {d} texts, {d} numbers and {d} errors, bit for bit", .{
        count, numbers, count - numbers,
    });
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
    try testIndex2d(run, &report, context, module);
    try testMulwide(run, &report, context, module);
    try testAddsub(run, &report, context, module);
    try testConstantTables(run, &report, context, module);
    try runIndexed(run, &report, context, module, string_table_name, wordValue, "strings read through a table of slices in constant data");
    try runIndexed(run, &report, context, module, error_names_name, errorNameValue, "@errorName read through the table of error names");
    try runIndexed(run, &report, context, module, allocator_vtable_name, allocatorValue, "allocations through std.mem.Allocator and its vtable");
    try testParseFloat(run, &report, context, module);
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
