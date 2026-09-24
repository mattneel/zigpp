//! The host side of the `std.gpu` end-to-end test: it loads the two PTX images into the driver
//! and runs every kernel of `kernels.zig` on a GPU, checking the results against the values that
//! the same computation produces here, in the test process.
//!
//! The test skips itself, with a message and a successful exit, when there is no driver or no
//! device. Everything else that goes wrong is a test failure: the kernels that fail, the element
//! of the first mismatches, and the number of checks that passed are printed, and the process
//! exits with a nonzero status.

const std = @import("std");
const cuda = std.gpu.cuda;

pub fn main(_: std.process.Init) !void {
    var device_name_buffer: [256]u8 = undefined;
    var driver = cuda.Driver.open() catch |err| switch (err) {
        error.DriverNotFound, error.NoDevice => return skip(@errorName(err)),
        else => |open_error| {
            std.debug.print("gpu_cuda: cannot open the CUDA driver: {s}\n", .{@errorName(open_error)});
            std.process.exit(1);
        },
    };
    defer driver.close();

    const device_count = driver.deviceCount() catch |err| return fail("count the devices", err);
    if (device_count == 0) return skip("no device");
    const device = driver.device(0) catch |err| return fail("open the device", err);
    const name = device.name(&device_name_buffer) catch "unknown";
    const capability = device.computeCapability() catch cuda.ComputeCapability{ .major = 0, .minor = 0 };
    const version = driver.version() catch cuda.Version{ .major = 0, .minor = 0 };

    const context = device.retainPrimaryContext() catch |err| return fail("retain the primary context", err);
    defer context.release();

    // The kernels that format text and parse JSON need more stack and more buffered output than
    // the defaults of a context give them.
    context.setLimit(.stack_size, 256 * 1024) catch |err| switch (err) {
        error.UnsupportedLimit => {},
        else => return fail("set the stack size", err),
    };
    context.setLimit(.printf_fifo_size, 8 * 1024 * 1024) catch |err| switch (err) {
        error.UnsupportedLimit => {},
        else => return fail("set the printf buffer size", err),
    };
    context.setLimit(.malloc_heap_size, 16 * 1024 * 1024) catch |err| switch (err) {
        error.UnsupportedLimit => {},
        else => return fail("set the device heap size", err),
    };

    std.debug.print("gpu_cuda: {s}, CUDA {d}.{d}, compute capability {d}.{d}\n", .{
        name, version.major, version.minor, capability.major, capability.minor,
    });

    const images = [_]Image{
        .{ .name = "debug", .ptx = @embedFile("kernels_debug.ptx") },
        .{ .name = "fast", .ptx = @embedFile("kernels_fast.ptx") },
    };

    var failed = false;
    for (images) |image| {
        var error_log: [16 * 1024]u8 = undefined;
        @memset(&error_log, 0);
        const module = context.loadModule(image.ptx, .{ .error_log = &error_log }) catch |err| {
            std.debug.print("gpu_cuda: cannot load the {s} module: {s}\n{s}\n", .{
                image.name, @errorName(err), std.mem.sliceTo(&error_log, 0),
            });
            return std.process.exit(1);
        };
        defer module.unload();

        var runner: Runner = .{ .context = context, .module = module, .image = image.name };
        for (tests) |one| {
            runner.kernel = one.name;
            one.run(&runner);
            if (runner.device_faulted) {
                std.debug.print("gpu_cuda: {s} PTX: the device faulted, the rest of this image is not tested\n", .{image.name});
                break;
            }
        }
        if (runner.failures != 0) failed = true;
        std.debug.print("gpu_cuda: {s} PTX: {d} launches, {d} checks passed, {d} failed\n", .{
            image.name, runner.launches, runner.checks, runner.failures,
        });
    }

    if (failed) std.process.exit(1);
}

fn skip(reason: []const u8) void {
    std.debug.print("gpu_cuda: skipping, no GPU to test on: {s}\n", .{reason});
}

fn fail(what: []const u8, err: anyerror) void {
    std.debug.print("gpu_cuda: cannot {s}: {s}\n", .{ what, @errorName(err) });
    std.process.exit(1);
}

/// One PTX image of the kernels, embedded in this program by `build.zig`.
const Image = struct {
    name: []const u8,
    ptx: [:0]const u8,
};

/// A test of one example, or of one group of kernels: it launches the kernels and checks the
/// results.
const Test = struct {
    /// The name of the example or of the group, for the messages of failed launches.
    name: []const u8,
    run: *const fn (*Runner) void,
};

/// The state of a run of every kernel of one PTX image: the module to look the kernels up in, and
/// the counts of the summary.
const Runner = struct {
    context: cuda.Context,
    module: cuda.Module,
    /// The name of the PTX image, "debug" or "fast", for the messages.
    image: []const u8,
    /// The name of the kernel that the current test launches, for the messages of failed checks.
    kernel: []const u8 = "",
    checks: u64 = 0,
    failures: u64 = 0,
    launches: u64 = 0,
    /// The number of mismatches that were printed of the current kernel, so that a kernel that
    /// fails everywhere does not print thousands of lines.
    reported: u64 = 0,
    /// The kernel whose launch failed, if any: its results are not there to check, so the checks
    /// of that kernel are skipped, and the failure was reported when it happened.
    broken: ?[]const u8 = null,
    /// Whether the device faulted, which leaves the context unusable for the rest of the image.
    device_faulted: bool = false,

    /// Launches the kernel `name` and waits for it. A launch or a device failure is reported and
    /// counted here, and the checks of that kernel are skipped, because the kernel did not write
    /// its results.
    fn run(r: *Runner, name: [:0]const u8, config: cuda.LaunchConfig, args: anytype) void {
        r.kernel = name;
        r.reported = 0;
        const function = r.module.function(name) catch |err| return r.reportError(name, "find", err);
        function.launch(config, args) catch |err| return r.reportError(name, "launch", err);
        r.launches += 1;
        r.context.synchronize() catch |err| return r.reportError(name, "run", err);
    }

    /// Launches a one-dimensional kernel over `n` elements in blocks of `threads` threads.
    fn runLinear(r: *Runner, name: [:0]const u8, n: u32, threads: u32, args: anytype) void {
        r.run(name, cuda.LaunchConfig.linear(n, threads), args);
    }

    fn reportError(r: *Runner, name: []const u8, what: []const u8, err: anyerror) void {
        r.failures += 1;
        r.broken = name;
        // A kernel that fails on the device leaves the whole context unusable: every call after
        // it fails with the same error, and nothing of this image can be tested any more.
        if (err == error.IllegalAddress or err == error.IllegalInstruction) r.device_faulted = true;
        std.debug.print("gpu_cuda: {s} PTX: cannot {s} the kernel {s}: {s}\n", .{
            r.image, what, name, @errorName(err),
        });
    }

    /// Copies `values` into a new buffer of device memory.
    fn upload(r: *Runner, comptime T: type, values: []const T) !cuda.Buffer(T) {
        const buffer = try r.context.alloc(T, values.len);
        errdefer buffer.free();
        try buffer.copyFromHost(values);
        return buffer;
    }

    /// Checks one result against the value that the same computation gives here.
    fn expect(r: *Runner, index: usize, expected: anytype, actual: @TypeOf(expected)) void {
        if (r.skipChecks()) return;
        if (equalValues(@TypeOf(expected), expected, actual)) {
            r.checks += 1;
            return;
        }
        if (comptime @typeInfo(@TypeOf(expected)) == .float) {
            const Bits = @Int(.unsigned, @bitSizeOf(@TypeOf(expected)));
            r.reportMismatch(index, "{d} (0x{x})", .{ expected, @as(Bits, @bitCast(expected)) }, "{d} (0x{x})", .{
                actual, @as(Bits, @bitCast(actual)),
            });
        } else {
            r.reportMismatch(index, "{any}", .{expected}, "{any}", .{actual});
        }
    }

    /// Checks every element of `actual` against `expected`.
    fn expectSlice(r: *Runner, expected: anytype, actual: anytype) void {
        for (expected, actual, 0..) |expected_value, actual_value, index| {
            r.expect(index, expected_value, actual_value);
        }
    }

    /// Checks that the text that a kernel printed contains `part`.
    fn expectContains(r: *Runner, text: []const u8, part: []const u8) void {
        if (r.skipChecks()) return;
        if (std.mem.indexOf(u8, text, part) != null) {
            r.checks += 1;
            return;
        }
        r.failures += 1;
        r.reported += 1;
        if (r.reported > 10) return;
        std.debug.print("gpu_cuda: {s} PTX: kernel {s}: the output does not contain \"{s}\"\n", .{
            r.image, r.kernel, part,
        });
    }

    fn skipChecks(r: *Runner) bool {
        const broken = r.broken orelse return false;
        return std.mem.eql(u8, broken, r.kernel);
    }

    fn reportMismatch(r: *Runner, index: usize, comptime expected_fmt: []const u8, expected_args: anytype, comptime actual_fmt: []const u8, actual_args: anytype) void {
        r.failures += 1;
        r.reported += 1;
        if (r.reported > 10) return;
        std.debug.print("gpu_cuda: {s} PTX: kernel {s}: element {d}: expected " ++ expected_fmt ++ ", found " ++ actual_fmt ++ "\n", .{
            r.image, r.kernel, index,
        } ++ expected_args ++ actual_args);
    }
};

/// The grid of a 2D kernel that covers `x` by `y` elements in tiles of `tile` by `tile`.
fn grid2D(x: u32, y: u32, tile: u32) cuda.Dim3 {
    return .{ .x = (x + tile - 1) / tile, .y = (y + tile - 1) / tile, .z = 1 };
}

fn block2D(tile: u32) cuda.Dim3 {
    return .{ .x = tile, .y = tile, .z = 1 };
}

fn equalValues(comptime T: type, expected: T, actual: T) bool {
    if (comptime @typeInfo(T) == .float) {
        // Processors produce different NaN bit patterns for the same operation, so any NaN matches
        // any NaN; every other value must match bit for bit, including the sign of zero.
        if (std.math.isNan(expected)) return std.math.isNan(actual);
        const Bits = @Int(.unsigned, @bitSizeOf(T));
        return @as(Bits, @bitCast(expected)) == @as(Bits, @bitCast(actual));
    }
    return expected == actual;
}

// The output of the device through `std.gpu.print` is written to the standard output of this
// process, so a check of the text of a kernel reads it from a temporary file that the standard
// output is redirected into while that kernel runs.

extern "c" fn fflush(stream: ?*anyopaque) c_int;

const StdoutCapture = struct {
    path: [64:0]u8 = undefined,
    path_len: usize = 0,
    file: c_int = -1,
    saved: c_int = -1,
    active: bool = false,
    buffer: [64 * 1024]u8 = undefined,

    /// Redirects the standard output of the process into a temporary file.
    fn begin() !StdoutCapture {
        var capture: StdoutCapture = .{};
        const path = try std.mem.printSentinel(capture.path[0..capture.path.len], "/tmp/zig-gpu-cuda-{d}.txt", .{std.c.getpid()}, 0);
        capture.path_len = path.len;
        const path_pointer: [*:0]const u8 = @ptrCast(&capture.path);
        capture.file = std.c.open(path_pointer, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(c_uint, 0o600));
        if (capture.file < 0) return error.OpenFailed;
        capture.saved = std.c.dup(1);
        if (capture.saved < 0 or std.c.dup2(capture.file, 1) < 0) return error.RedirectFailed;
        capture.active = true;
        return capture;
    }

    /// Restores the standard output and returns the text that the device wrote to it. The host
    /// must have synchronized with the device first, because that is when the driver writes the
    /// buffered output of the kernels.
    fn end(capture: *StdoutCapture) []const u8 {
        // The driver writes the output with the C standard output stream, which buffers it.
        _ = fflush(null);
        if (std.c.dup2(capture.saved, 1) < 0) return "";
        capture.active = false;
        if (std.c.lseek(capture.file, 0, std.c.SEEK.SET) < 0) return "";
        var total: usize = 0;
        while (total < capture.buffer.len) {
            const count = std.c.read(capture.file, capture.buffer[total..].ptr, capture.buffer.len - total);
            if (count <= 0) break;
            total += @intCast(count);
        }
        return capture.buffer[0..total];
    }

    fn deinit(capture: *StdoutCapture) void {
        if (capture.active) {
            _ = fflush(null);
            _ = std.c.dup2(capture.saved, 1);
            capture.active = false;
        }
        if (capture.saved >= 0) _ = std.c.close(capture.saved);
        if (capture.file >= 0) _ = std.c.close(capture.file);
        if (capture.path_len != 0) _ = std.c.unlink(@ptrCast(&capture.path));
    }
};

// ---------------------------------------------------------------------------------------------
// The tests, one for every example of ugpu and one for every group of added kernels.
// ---------------------------------------------------------------------------------------------

const tests = [_]Test{
    .{ .name = "vector_add", .run = testVectorAdd },
    .{ .name = "reduce", .run = testReduce },
    .{ .name = "histogram", .run = testHistogram },
    .{ .name = "warp", .run = testWarp },
    .{ .name = "matrix_mul", .run = testMatrixMul },
    .{ .name = "convolution", .run = testConvolution },
    .{ .name = "stencil", .run = testStencil },
    .{ .name = "stdlib", .run = testStdlib },
    .{ .name = "hashmap", .run = testHashMap },
    .{ .name = "base64", .run = testBase64 },
    .{ .name = "string_search", .run = testStringSearch },
    .{ .name = "json", .run = testJson },
    .{ .name = "dynamic", .run = testDynamic },
    .{ .name = "printf", .run = testPrintf },
    .{ .name = "hello_gpu", .run = testHello },
    .{ .name = "builtin_math", .run = testBuiltinMath },
    .{ .name = "f128", .run = testF128 },
    .{ .name = "parse_float", .run = testParseFloat },
    .{ .name = "device_heap", .run = testDeviceHeap },
    .{ .name = "bump_allocator", .run = testBumpAllocator },
};

/// Sizes that are not multiples of a block, or of the tile of a 2D kernel, so that the kernels
/// take their guards against the end of the input.
const odd_size = 1000;
const less_odd_size = 320;

fn testVectorAdd(r: *Runner) void {
    const n = odd_size;
    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    for (&a, &b, 0..) |*a_value, *b_value, i| {
        a_value.* = @floatFromInt(i + 1);
        b_value.* = @floatFromInt(2 * i + 7);
    }

    const a_buffer = r.upload(f32, &a) catch |err| return r.reportError("vectorAdd", "upload to", err);
    defer a_buffer.free();
    const b_buffer = r.upload(f32, &b) catch |err| return r.reportError("vectorAdd", "upload to", err);
    defer b_buffer.free();
    const c_buffer = r.context.alloc(f32, n) catch |err| return r.reportError("vectorAdd", "allocate for", err);
    defer c_buffer.free();

    r.runLinear("vectorAdd", n, 256, .{ a_buffer, b_buffer, c_buffer, @as(u32, n) });

    var actual: [n]f32 = undefined;
    c_buffer.copyToHost(&actual) catch |err| return r.reportError("vectorAdd", "copy the results of", err);
    for (&a, &b, &actual) |a_value, b_value, actual_value| {
        r.expect(0, a_value + b_value, actual_value);
    }
}

fn testReduce(r: *Runner) void {
    const n = odd_size;
    const blocks = (n + 255) / 256;
    var input: [n]f32 = undefined;
    for (&input, 0..) |*value, i| value.* = @floatFromInt((i * 7 + 3) % 13);

    const input_buffer = r.upload(f32, &input) catch |err| return r.reportError("sumReduce", "upload to", err);
    defer input_buffer.free();
    const output_buffer = r.context.alloc(f32, blocks) catch |err| return r.reportError("sumReduce", "allocate for", err);
    defer output_buffer.free();

    // The elements of a block are small integers, so both the order in which the kernel adds them
    // and the order here give the same sum.
    r.runLinear("sumReduce", n, 256, .{ input_buffer, output_buffer, @as(u32, n) });
    var sums: [blocks]f32 = undefined;
    output_buffer.copyToHost(&sums) catch |err| return r.reportError("sumReduce", "copy the results of", err);
    for (sums, 0..) |sum, block| {
        var expected: f32 = 0;
        for (input[block * 256 .. @min((block + 1) * 256, n)]) |value| expected += value;
        r.expect(block, expected, sum);
    }

    r.runLinear("maxReduce", n, 256, .{ input_buffer, output_buffer, @as(u32, n) });
    var maxima: [blocks]f32 = undefined;
    output_buffer.copyToHost(&maxima) catch |err| return r.reportError("maxReduce", "copy the results of", err);
    for (maxima, 0..) |maximum, block| {
        var expected: f32 = 0;
        for (input[block * 256 .. @min((block + 1) * 256, n)]) |value| expected = @max(expected, value);
        r.expect(block, expected, maximum);
    }
}

fn testHistogram(r: *Runner) void {
    const n = odd_size;
    const num_bins = 16;
    var data: [n]u32 = undefined;
    for (&data, 0..) |*value, i| value.* = @intCast((i * 7 + 3) % 23);

    const data_buffer = r.upload(u32, &data) catch |err| return r.reportError("histogram", "upload to", err);
    defer data_buffer.free();
    const bins_buffer = r.context.alloc(u32, num_bins) catch |err| return r.reportError("histogram", "allocate for", err);
    defer bins_buffer.free();

    var counts: [num_bins]u32 = @splat(0);
    for (data) |value| {
        if (value < num_bins) counts[value] += 1;
    }

    r.context.synchronize() catch |err| return r.reportError("histogram", "synchronize with", err);
    bins_buffer.zero() catch |err| return r.reportError("histogram", "clear the bins of", err);
    r.runLinear("histogram", n, 256, .{ data_buffer, bins_buffer, @as(u32, n), @as(u32, num_bins) });
    var bins: [num_bins]u32 = undefined;
    bins_buffer.copyToHost(&bins) catch |err| return r.reportError("histogram", "copy the results of", err);
    r.expectSlice(&counts, &bins);

    bins_buffer.zero() catch |err| return r.reportError("histogramShared", "clear the bins of", err);
    r.runLinear("histogramShared", n, 256, .{ data_buffer, bins_buffer, @as(u32, n), @as(u32, num_bins) });
    bins_buffer.copyToHost(&bins) catch |err| return r.reportError("histogramShared", "copy the results of", err);
    r.expectSlice(&counts, &bins);

    // Pairs, with values of the x and the y coordinate at and above their numbers of bins, which
    // the kernel leaves out.
    const num_bins_x = 7;
    const num_bins_y = 5;
    var x: [n]u32 = undefined;
    var y: [n]u32 = undefined;
    for (&x, &y, 0..) |*x_value, *y_value, i| {
        x_value.* = @intCast((i * 5 + 7) % 9);
        y_value.* = @intCast((i * 11 + 2) % 6);
    }
    var bin_counts: [num_bins_x * num_bins_y]u32 = @splat(0);
    for (x, y) |x_value, y_value| {
        if (x_value < num_bins_x and y_value < num_bins_y) bin_counts[y_value * num_bins_x + x_value] += 1;
    }

    const x_buffer = r.upload(u32, &x) catch |err| return r.reportError("histogram2D", "upload to", err);
    defer x_buffer.free();
    const y_buffer = r.upload(u32, &y) catch |err| return r.reportError("histogram2D", "upload to", err);
    defer y_buffer.free();
    const pairs_buffer = r.context.alloc(u32, num_bins_x * num_bins_y) catch |err| return r.reportError("histogram2D", "allocate for", err);
    defer pairs_buffer.free();

    r.runLinear("histogram2D", n, 256, .{ x_buffer, y_buffer, pairs_buffer, @as(u32, n), @as(u32, num_bins_x), @as(u32, num_bins_y) });
    var pair_counts: [num_bins_x * num_bins_y]u32 = undefined;
    pairs_buffer.copyToHost(&pair_counts) catch |err| return r.reportError("histogram2D", "copy the results of", err);
    r.expectSlice(&bin_counts, &pair_counts);
}

fn testWarp(r: *Runner) void {
    // A multiple of the warp size, so that every warp of the grid is full, and not a multiple of
    // the block size, so that the last block is partial and several blocks run.
    const n = less_odd_size;
    const threads = 256;
    const warps_per_block = threads / 32;
    const blocks = (n + threads - 1) / threads;

    var input: [n]u32 = undefined;
    for (&input, 0..) |*value, i| value.* = @intCast((i * 13 + 5) % 100 + 1);

    const input_buffer = r.upload(u32, &input) catch |err| return r.reportError("warpSumKernel", "upload to", err);
    defer input_buffer.free();
    const output_buffer = r.context.alloc(u32, blocks * warps_per_block) catch |err| return r.reportError("warpSumKernel", "allocate for", err);
    defer output_buffer.free();

    // The threads past the end of the input take part in the warp functions with a zero value, so
    // the results of the last warps are the sum, the vote, or the shuffle of the elements that
    // exist and of those zeros.
    const warpValue = struct {
        fn of(index: u32) u32 {
            return if (index < n) (index * 13 + 5) % 100 + 1 else 0;
        }
    }.of;

    r.runLinear("warpSumKernel", n, threads, .{ input_buffer, output_buffer, @as(u32, n) });
    var results: [blocks * warps_per_block]u32 = undefined;
    output_buffer.copyToHost(&results) catch |err| return r.reportError("warpSumKernel", "copy the results of", err);
    for (results, 0..) |result, warp| {
        var expected: u32 = 0;
        for (0..32) |lane| expected += warpValue(@intCast(warp * 32 + lane));
        r.expect(warp, expected, result);
    }

    r.runLinear("warpMaxKernel", n, threads, .{ input_buffer, output_buffer, @as(u32, n) });
    output_buffer.copyToHost(&results) catch |err| return r.reportError("warpMaxKernel", "copy the results of", err);
    for (results, 0..) |result, warp| {
        var expected: u32 = 0;
        for (0..32) |lane| expected = @max(expected, warpValue(@intCast(warp * 32 + lane)));
        r.expect(warp, expected, result);
    }

    r.runLinear("warpMinKernel", n, threads, .{ input_buffer, output_buffer, @as(u32, n) });
    output_buffer.copyToHost(&results) catch |err| return r.reportError("warpMinKernel", "copy the results of", err);
    for (results, 0..) |result, warp| {
        var expected: u32 = std.math.maxInt(u32);
        for (0..32) |lane| expected = @min(expected, warpValue(@intCast(warp * 32 + lane)));
        r.expect(warp, expected, result);
    }

    r.runLinear("ballotKernel", n, threads, .{ input_buffer, output_buffer, @as(u32, n) });
    output_buffer.copyToHost(&results) catch |err| return r.reportError("ballotKernel", "copy the results of", err);
    for (results, 0..) |result, warp| {
        var expected: u32 = 0;
        for (0..32) |lane| {
            if (warpValue(@intCast(warp * 32 + lane)) > 100) expected |= @as(u32, 1) << @intCast(lane);
        }
        r.expect(warp, expected, result);
    }

    r.runLinear("checkDivergence", n, threads, .{ input_buffer, output_buffer, @as(u32, n) });
    output_buffer.copyToHost(&results) catch |err| return r.reportError("checkDivergence", "copy the results of", err);
    for (results, 0..) |result, warp| {
        const first = warpValue(@intCast(warp * 32)) > 50;
        var agree = true;
        for (0..32) |lane| {
            if ((warpValue(@intCast(warp * 32 + lane)) > 50) != first) agree = false;
        }
        r.expect(warp, @as(u32, @intFromBool(agree)), result);
    }

    const broadcast_buffer = r.context.alloc(u32, n) catch |err| return r.reportError("shuffleBroadcastKernel", "allocate for", err);
    defer broadcast_buffer.free();
    r.runLinear("shuffleBroadcastKernel", n, threads, .{ input_buffer, broadcast_buffer, @as(u32, n) });
    var broadcast: [n]u32 = undefined;
    broadcast_buffer.copyToHost(&broadcast) catch |err| return r.reportError("shuffleBroadcastKernel", "copy the results of", err);
    for (&input, &broadcast, 0..) |value, result, i| {
        _ = value;
        // Every element of a warp gets the value of the first lane of that warp.
        r.expect(i, input[i / 32 * 32], result);
    }

    const shuffled = r.context.alloc(u32, n * 3) catch |err| return r.reportError("shuffleKernel", "allocate for", err);
    defer shuffled.free();
    r.runLinear("shuffleKernel", n, threads, .{ input_buffer, shuffled, @as(u32, n) });
    // The kernel writes three arrays of `n` elements into one buffer of `n * 3`, which the host
    // checks one after the other.
    var shuffle_results: [n * 3]u32 = undefined;
    shuffled.copyToHost(&shuffle_results) catch |err| return r.reportError("shuffleKernel", "copy the results of", err);
    for (0..3) |which| {
        r.kernel = "shuffleKernel";
        for (0..n) |i| {
            const lane = i % 32;
            const source = switch (which) {
                0 => if (lane + 1 < 32) i + 1 else i,
                1 => if (lane >= 1) i - 1 else i,
                else => i ^ 1,
            };
            r.expect(i, warpValue(@intCast(source)), shuffle_results[which * n + i]);
        }
    }
}

fn testMatrixMul(r: *Runner) void {
    const m = 20;
    const n = 22;
    const k = 18;

    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    for (&a, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast((i * 3 + 1) % 7)) - 3);
    for (&b, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast((i * 5 + 2) % 5)) - 2);

    var expected: [m * n]f32 = undefined;
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |i| sum += a[row * k + i] * b[i * n + col];
            expected[row * n + col] = sum;
        }
    }

    const a_buffer = r.upload(f32, &a) catch |err| return r.reportError("matrixMulNaive", "upload to", err);
    defer a_buffer.free();
    const b_buffer = r.upload(f32, &b) catch |err| return r.reportError("matrixMulNaive", "upload to", err);
    defer b_buffer.free();
    const c_buffer = r.context.alloc(f32, m * n) catch |err| return r.reportError("matrixMulNaive", "allocate for", err);
    defer c_buffer.free();

    var actual: [m * n]f32 = undefined;
    const naive_config: cuda.LaunchConfig = .{ .grid = grid2D(n, m, 16), .block = block2D(16) };
    r.run("matrixMulNaive", naive_config, .{ a_buffer, b_buffer, c_buffer, @as(u32, m), @as(u32, n), @as(u32, k) });
    c_buffer.copyToHost(&actual) catch |err| return r.reportError("matrixMulNaive", "copy the results of", err);
    r.expectSlice(&expected, &actual);

    r.run("matrixMulTiled", naive_config, .{ a_buffer, b_buffer, c_buffer, @as(u32, m), @as(u32, n), @as(u32, k) });
    c_buffer.copyToHost(&actual) catch |err| return r.reportError("matrixMulTiled", "copy the results of", err);
    r.expectSlice(&expected, &actual);

    // A block of 32 columns of threads and 8 rows: the kernel tiles the output in 32 by 32
    // elements, and a block of 32 by 32 threads needs more registers than the debug build of the
    // kernel has.
    const large_config: cuda.LaunchConfig = .{ .grid = grid2D(n, m, 32), .block = .{ .x = 32, .y = 8, .z = 1 } };
    r.run("matrixMulLargeTile", large_config, .{ a_buffer, b_buffer, c_buffer, @as(u32, m), @as(u32, n), @as(u32, k) });
    c_buffer.copyToHost(&actual) catch |err| return r.reportError("matrixMulLargeTile", "copy the results of", err);
    r.expectSlice(&expected, &actual);
}

fn testConvolution(r: *Runner) void {
    const width = 34;
    const height = 30;
    const filter_size = 5;

    var input: [width * height]f32 = undefined;
    for (&input, 0..) |*value, i| value.* = @floatFromInt((i * 7 + 3) % 10);
    var filter: [filter_size * filter_size]f32 = undefined;
    for (&filter, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast((i * 2 + 1) % 3)) - 1);

    const input_buffer = r.upload(f32, &input) catch |err| return r.reportError("convolution2D", "upload to", err);
    defer input_buffer.free();
    const filter_buffer = r.upload(f32, &filter) catch |err| return r.reportError("convolution2D", "upload to", err);
    defer filter_buffer.free();
    const output_buffer = r.context.alloc(f32, width * height) catch |err| return r.reportError("convolution2D", "allocate for", err);
    defer output_buffer.free();

    const config: cuda.LaunchConfig = .{ .grid = grid2D(width, height, 16), .block = block2D(16) };
    r.run("convolution2D", config, .{
        input_buffer, output_buffer, filter_buffer, @as(u32, width), @as(u32, height), @as(u32, filter_size),
    });
    var actual: [width * height]f32 = undefined;
    output_buffer.copyToHost(&actual) catch |err| return r.reportError("convolution2D", "copy the results of", err);
    for (0..height) |y| {
        for (0..width) |x| {
            var sum: f32 = 0;
            for (0..filter_size) |fy| {
                for (0..filter_size) |fx| {
                    const source_y = @as(i64, @intCast(y)) + @as(i64, @intCast(fy)) - 2;
                    const source_x = @as(i64, @intCast(x)) + @as(i64, @intCast(fx)) - 2;
                    if (source_x >= 0 and source_x < width and source_y >= 0 and source_y < height) {
                        const source = @as(usize, @intCast(source_y)) * width + @as(usize, @intCast(source_x));
                        sum += input[source] * filter[fy * filter_size + fx];
                    }
                }
            }
            r.expect(y * width + x, sum, actual[y * width + x]);
        }
    }

    // The transposition of the same input.
    const transpose_config: cuda.LaunchConfig = .{ .grid = grid2D(width, height, 16), .block = block2D(16) };
    const transposed_buffer = r.context.alloc(f32, width * height) catch |err| return r.reportError("transpose", "allocate for", err);
    defer transposed_buffer.free();
    r.run("transpose", transpose_config, .{ input_buffer, transposed_buffer, @as(u32, width), @as(u32, height) });
    var transposed: [width * height]f32 = undefined;
    transposed_buffer.copyToHost(&transposed) catch |err| return r.reportError("transpose", "copy the results of", err);
    for (0..width) |row| {
        for (0..height) |col| {
            r.expect(row * height + col, input[col * width + row], transposed[row * height + col]);
        }
    }

    // The exclusive prefix sums of every block of the input.
    const n = odd_size;
    var scan_input: [n]f32 = undefined;
    for (&scan_input, 0..) |*value, i| value.* = @floatFromInt((i * 11 + 4) % 9);

    const scan_input_buffer = r.upload(f32, &scan_input) catch |err| return r.reportError("prefixSum", "upload to", err);
    defer scan_input_buffer.free();
    const scan_output_buffer = r.context.alloc(f32, n) catch |err| return r.reportError("prefixSum", "allocate for", err);
    defer scan_output_buffer.free();
    r.runLinear("prefixSum", n, 256, .{ scan_input_buffer, scan_output_buffer, @as(u32, n) });
    var scan_actual: [n]f32 = undefined;
    scan_output_buffer.copyToHost(&scan_actual) catch |err| return r.reportError("prefixSum", "copy the results of", err);
    for (0..n) |i| {
        var expected: f32 = 0;
        for (scan_input[i - i % 256 .. i]) |value| expected += value;
        r.expect(i, expected, scan_actual[i]);
    }
}

fn testStencil(r: *Runner) void {
    const n = odd_size;
    var input: [n]f32 = undefined;
    for (&input, 0..) |*value, i| value.* = @floatFromInt((i * 5 + 2) % 9);

    const input_buffer = r.upload(f32, &input) catch |err| return r.reportError("stencil1D", "upload to", err);
    defer input_buffer.free();
    const output_buffer = r.context.alloc(f32, n) catch |err| return r.reportError("stencil1D", "allocate for", err);
    defer output_buffer.free();

    const alpha: f32 = 1;
    const beta: f32 = -2;
    const gamma: f32 = 1;
    r.runLinear("stencil1D", n, 256, .{ input_buffer, output_buffer, @as(u32, n), alpha, beta, gamma });
    var actual: [n]f32 = undefined;
    output_buffer.copyToHost(&actual) catch |err| return r.reportError("stencil1D", "copy the results of", err);
    for (actual, 0..) |result, i| {
        const expected = if (i == 0 or i == n - 1)
            input[i]
        else
            alpha * input[i - 1] + beta * input[i] + gamma * input[i + 1];
        r.expect(i, expected, result);
    }

    // The 2D stencils, whose edges are the elements of the input copied to the output.
    const width = 34;
    const height = 30;
    var image: [width * height]f32 = undefined;
    for (&image, 0..) |*value, i| value.* = @floatFromInt((i * 3 + 5) % 9);

    const image_buffer = r.upload(f32, &image) catch |err| return r.reportError("stencil2DLaplace", "upload to", err);
    defer image_buffer.free();
    const image_output_buffer = r.context.alloc(f32, width * height) catch |err| return r.reportError("stencil2DLaplace", "allocate for", err);
    defer image_output_buffer.free();
    var image_actual: [width * height]f32 = undefined;

    const config: cuda.LaunchConfig = .{ .grid = grid2D(width, height, 16), .block = block2D(16) };
    r.run("stencil2DLaplace", config, .{ image_buffer, image_output_buffer, @as(u32, width), @as(u32, height) });
    image_output_buffer.copyToHost(&image_actual) catch |err| return r.reportError("stencil2DLaplace", "copy the results of", err);
    for (0..height) |y| {
        for (0..width) |x| {
            const index = y * width + x;
            const expected = if (x == 0 or x == width - 1 or y == 0 or y == height - 1) image[index] else blk: {
                const center = image[index];
                const left = image[index - 1];
                const right = image[index + 1];
                const top = image[index - width];
                const bottom = image[index + width];
                break :blk left + right + top + bottom - 4 * center;
            };
            r.expect(index, expected, image_actual[index]);
        }
    }

    r.run("stencil2D9Point", config, .{ image_buffer, image_output_buffer, @as(u32, width), @as(u32, height) });
    image_output_buffer.copyToHost(&image_actual) catch |err| return r.reportError("stencil2D9Point", "copy the results of", err);
    for (0..height) |y| {
        for (0..width) |x| {
            const index = y * width + x;
            const expected = if (x == 0 or x == width - 1 or y == 0 or y == height - 1) image[index] else blk: {
                const center = image[index];
                const north = image[index - width];
                const south = image[index + width];
                const east = image[index + 1];
                const west = image[index - 1];
                const north_east = image[index - width + 1];
                const north_west = image[index - width - 1];
                const south_east = image[index + width + 1];
                const south_west = image[index + width - 1];
                break :blk 0.25 * (north_east + north_west + south_east + south_west) +
                    0.5 * (north + south + east + west) -
                    3 * center;
            };
            r.expect(index, expected, image_actual[index]);
        }
    }
}

fn testStdlib(r: *Runner) void {
    // The square root of a negative element is a NaN on both sides, which the checks of the
    // floats of this test treat as equal.
    const n = 96;
    var input: [n]f32 = undefined;
    for (&input, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast(i % 17)) - 8);

    const input_buffer = r.upload(f32, &input) catch |err| return r.reportError("mathOps", "upload to", err);
    defer input_buffer.free();
    const output_buffer = r.context.alloc(f32, n) catch |err| return r.reportError("mathOps", "allocate for", err);
    defer output_buffer.free();

    r.runLinear("mathOps", n, 32, .{ input_buffer, output_buffer, @as(u32, n) });
    var actual: [n]f32 = undefined;
    output_buffer.copyToHost(&actual) catch |err| return r.reportError("mathOps", "copy the results of", err);
    for (input, actual) |value, result| {
        const expected = std.math.sqrt(value) + @abs(value) + @min(@max(value, -1), 1);
        r.expect(0, expected, result);
    }

    var bits_input: [n]u32 = undefined;
    for (&bits_input, 0..) |*value, i| {
        value.* = switch (i % 6) {
            0 => 0,
            1 => 1,
            2 => std.math.maxInt(u32),
            3 => 1 << 31,
            4 => 0x5555_aaaa,
            else => @as(u32, @intCast(i)) *% 2654435761,
        };
    }
    const bits_input_buffer = r.upload(u32, &bits_input) catch |err| return r.reportError("bitOps", "upload to", err);
    defer bits_input_buffer.free();
    const bits_output_buffer = r.context.alloc(u32, n) catch |err| return r.reportError("bitOps", "allocate for", err);
    defer bits_output_buffer.free();
    r.runLinear("bitOps", n, 32, .{ bits_input_buffer, bits_output_buffer, @as(u32, n) });
    var bits_actual: [n]u32 = undefined;
    bits_output_buffer.copyToHost(&bits_actual) catch |err| return r.reportError("bitOps", "copy the results of", err);
    for (bits_input, bits_actual) |value, result| {
        r.expect(0, @as(u32, @popCount(value)) + @clz(value) + @ctz(value), result);
    }

    // Four elements per thread: the input has to be a multiple of four.
    const vector_n = 1024;
    var va: [vector_n]f32 = undefined;
    var vb: [vector_n]f32 = undefined;
    for (&va, &vb, 0..) |*a_value, *b_value, i| {
        a_value.* = @as(f32, @floatFromInt(i % 5)) * 0.5;
        b_value.* = @as(f32, @floatFromInt(i % 3)) + 0.25;
    }
    const va_buffer = r.upload(f32, &va) catch |err| return r.reportError("vectorOps", "upload to", err);
    defer va_buffer.free();
    const vb_buffer = r.upload(f32, &vb) catch |err| return r.reportError("vectorOps", "upload to", err);
    defer vb_buffer.free();
    const vector_output = r.context.alloc(f32, vector_n) catch |err| return r.reportError("vectorOps", "allocate for", err);
    defer vector_output.free();
    r.runLinear("vectorOps", vector_n, 64, .{ va_buffer, vb_buffer, vector_output, @as(u32, vector_n) });
    var vector_actual: [vector_n]f32 = undefined;
    vector_output.copyToHost(&vector_actual) catch |err| return r.reportError("vectorOps", "copy the results of", err);
    for (va, vb, vector_actual) |a_value, b_value, result| {
        r.expect(0, a_value + b_value + a_value * b_value, result);
    }

    var scalar_input: [n]f32 = undefined;
    for (&scalar_input, 0..) |*value, i| value.* = @floatFromInt(@as(i32, @intCast(i % 11)) - 5);
    const scalar_input_buffer = r.upload(f32, &scalar_input) catch |err| return r.reportError("comptimeOps", "upload to", err);
    defer scalar_input_buffer.free();
    r.runLinear("comptimeOps", n, 32, .{ scalar_input_buffer, output_buffer, @as(u32, n) });
    output_buffer.copyToHost(&actual) catch |err| return r.reportError("comptimeOps", "copy the results of", err);
    for (scalar_input, actual) |value, result| {
        r.expect(0, @max(value * 2, 1) + @min(value * 2, 1), result);
    }
}

fn testHashMap(r: *Runner) void {
    const n = 20;
    const threads = 8;
    var keys: [n]u32 = undefined;
    var values: [n]u32 = undefined;
    for (&keys, &values, 0..) |*key, *value, i| {
        key.* = @intCast(i % 7);
        value.* = @intCast(i * 3);
    }

    const keys_buffer = r.upload(u32, &keys) catch |err| return r.reportError("hashMapKernel", "upload to", err);
    defer keys_buffer.free();
    const values_buffer = r.upload(u32, &values) catch |err| return r.reportError("hashMapKernel", "upload to", err);
    defer values_buffer.free();
    const results_buffer = r.context.alloc(u32, n) catch |err| return r.reportError("hashMapKernel", "allocate for", err);
    defer results_buffer.free();

    r.runLinear("hashMapKernel", n, threads, .{ keys_buffer, values_buffer, @as(u32, n), results_buffer });
    var results: [n]u32 = undefined;
    results_buffer.copyToHost(&results) catch |err| return r.reportError("hashMapKernel", "copy the results of", err);
    // Every block builds the map of all of its pairs before it looks any of them up, so a key
    // that appears twice in a block keeps the value of its last pair.
    for (keys, 0..) |key, i| {
        const block_start = i / threads * threads;
        const block_end = @min(block_start + threads, n);
        var expected: u32 = 0xdead_beef;
        for (keys[block_start..block_end], values[block_start..block_end]) |block_key, block_value| {
            if (block_key == key) expected = block_value;
        }
        r.expect(i, expected, results[i]);
    }

    const inserted_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("hashMapStressKernel", "allocate for", err);
    defer inserted_buffer.free();
    const wrong_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("hashMapStressKernel", "allocate for", err);
    defer wrong_buffer.free();
    // More pairs than the shared heap holds: the map grows several times, then the allocator
    // refuses a growth, and the pairs that the map took must still be there.
    const stress_count = 5000;
    r.runLinear("hashMapStressKernel", 1, 1, .{ @as(u32, stress_count), inserted_buffer, wrong_buffer });
    var inserted: [1]u32 = undefined;
    inserted_buffer.copyToHost(&inserted) catch |err| return r.reportError("hashMapStressKernel", "copy the results of", err);
    var wrong: [1]u32 = undefined;
    wrong_buffer.copyToHost(&wrong) catch |err| return r.reportError("hashMapStressKernel", "copy the results of", err);
    r.expect(0, true, inserted[0] > 100 and inserted[0] < stress_count);
    r.expect(0, @as(u32, 0), wrong[0]);

    const string_n = 20;
    const string_values_buffer = r.upload(u32, values[0..string_n]) catch |err| return r.reportError("stringHashMapKernel", "upload to", err);
    defer string_values_buffer.free();
    const string_results_buffer = r.context.alloc(u32, string_n) catch |err| return r.reportError("stringHashMapKernel", "allocate for", err);
    defer string_results_buffer.free();
    r.runLinear("stringHashMapKernel", 1, threads, .{ string_values_buffer, @as(u32, string_n), string_results_buffer });
    var string_results: [string_n]u32 = undefined;
    string_results_buffer.copyToHost(&string_results) catch |err| return r.reportError("stringHashMapKernel", "copy the results of", err);
    for (values[0..string_n], string_results, 0..) |value, result, i| {
        r.expect(i, value, result);
    }
}

fn testBase64(r: *Runner) void {
    const chunk_size = 12;
    const n = 36;
    const threads = n / chunk_size;
    var input: [n]u8 = undefined;
    for (&input, 0..) |*value, i| value.* = @truncate(i * 7 + 1);

    // The text of the chunks, computed with the codec of the standard library here.
    const Encoder = std.base64.standard.Encoder;
    var expected_text: [Encoder.calcSize(n)]u8 = undefined;
    var expected_lengths: [threads]u32 = @splat(0);
    for (0..threads) |chunk| {
        const encoded = Encoder.encode(expected_text[chunk * 16 ..][0..16], input[chunk * chunk_size ..][0..chunk_size]);
        expected_lengths[chunk] = @intCast(encoded.len);
    }

    const input_buffer = r.upload(u8, &input) catch |err| return r.reportError("base64EncodeKernel", "upload to", err);
    defer input_buffer.free();
    const text_buffer = r.context.alloc(u8, Encoder.calcSize(n)) catch |err| return r.reportError("base64EncodeKernel", "allocate for", err);
    defer text_buffer.free();
    const lengths_buffer = r.context.alloc(u32, threads) catch |err| return r.reportError("base64EncodeKernel", "allocate for", err);
    defer lengths_buffer.free();

    r.runLinear("base64EncodeKernel", threads, threads, .{ input_buffer, text_buffer, lengths_buffer, @as(u32, n), @as(u32, chunk_size) });
    var text: [Encoder.calcSize(n)]u8 = undefined;
    text_buffer.copyToHost(&text) catch |err| return r.reportError("base64EncodeKernel", "copy the results of", err);
    var lengths: [threads]u32 = undefined;
    lengths_buffer.copyToHost(&lengths) catch |err| return r.reportError("base64EncodeKernel", "copy the results of", err);
    r.expectSlice(&expected_text, &text);
    r.expectSlice(&expected_lengths, &lengths);

    // The decoding of that text, chunk by chunk, has to give the input back.
    const decode_chunk = 12;
    const decode_threads = Encoder.calcSize(n) / decode_chunk;
    const text_input_buffer = r.upload(u8, &expected_text) catch |err| return r.reportError("base64DecodeKernel", "upload to", err);
    defer text_input_buffer.free();
    const decoded_buffer = r.context.alloc(u8, n) catch |err| return r.reportError("base64DecodeKernel", "allocate for", err);
    defer decoded_buffer.free();
    const decoded_lengths_buffer = r.context.alloc(u32, decode_threads) catch |err| return r.reportError("base64DecodeKernel", "allocate for", err);
    defer decoded_lengths_buffer.free();

    r.runLinear("base64DecodeKernel", @intCast(decode_threads), @intCast(decode_threads), .{
        text_input_buffer, decoded_buffer, decoded_lengths_buffer, @as(u32, @intCast(Encoder.calcSize(n))), @as(u32, decode_chunk),
    });
    var decoded: [n]u8 = undefined;
    decoded_buffer.copyToHost(&decoded) catch |err| return r.reportError("base64DecodeKernel", "copy the results of", err);
    r.expectSlice(&input, &decoded);

    // The round trip of one thread, which is also where the kernel prints its result.
    const round_n = 32;
    var round_input: [round_n]u8 = undefined;
    for (&round_input, 0..) |*value, i| value.* = @truncate(i * 3 + 9);
    const round_buffer = r.upload(u8, &round_input) catch |err| return r.reportError("base64RoundTripKernel", "upload to", err);
    defer round_buffer.free();
    const round_trip_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("base64RoundTripKernel", "allocate for", err);
    defer round_trip_buffer.free();

    var capture = StdoutCapture.begin() catch |err| return r.reportError("base64RoundTripKernel", "capture the output of", err);
    defer capture.deinit();
    r.runLinear("base64RoundTripKernel", 1, 1, .{ round_buffer, @as(u32, round_n), round_trip_buffer });
    const printed = capture.end();
    var round_trip: [1]u32 = undefined;
    round_trip_buffer.copyToHost(&round_trip) catch |err| return r.reportError("base64RoundTripKernel", "copy the results of", err);
    r.expect(0, @as(u32, 1), round_trip[0]);
    if (round_trip[0] == 1) r.expectContains(printed, "round-trip successful!");
}

fn testStringSearch(r: *Runner) void {
    const haystack = "the quick brown fox jumps over the lazy dog, the dog sleeps";
    const needles_text = [_][]const u8{ "the", "dog", "cat", "quick", "zzz" };
    const n = needles_text.len;

    var needles: [n][64]u8 = undefined;
    var needle_lens: [n]u32 = undefined;
    for (&needles, &needle_lens, needles_text) |*needle, *length, text| {
        @memset(needle, 0);
        @memcpy(needle[0..text.len], text);
        length.* = @intCast(text.len);
    }

    const haystack_buffer = r.upload(u8, haystack) catch |err| return r.reportError("stringSearchKernel", "upload to", err);
    defer haystack_buffer.free();
    const needles_buffer = r.upload([64]u8, &needles) catch |err| return r.reportError("stringSearchKernel", "upload to", err);
    defer needles_buffer.free();
    const lens_buffer = r.upload(u32, &needle_lens) catch |err| return r.reportError("stringSearchKernel", "upload to", err);
    defer lens_buffer.free();
    const results_buffer = r.context.alloc(u32, n) catch |err| return r.reportError("stringSearchKernel", "allocate for", err);
    defer results_buffer.free();

    r.runLinear("stringSearchKernel", n, 32, .{
        haystack_buffer, @as(u32, haystack.len), needles_buffer, lens_buffer, results_buffer, @as(u32, n),
    });
    var results: [n]u32 = undefined;
    results_buffer.copyToHost(&results) catch |err| return r.reportError("stringSearchKernel", "copy the results of", err);
    for (needles_text, results) |text, result| {
        const expected = if (std.mem.indexOf(u8, haystack, text)) |position| @as(u32, @intCast(position)) else std.math.maxInt(u32);
        r.expect(0, expected, result);
    }

    r.runLinear("stringCountKernel", n, 32, .{
        haystack_buffer, @as(u32, haystack.len), needles_buffer, lens_buffer, results_buffer, @as(u32, n),
    });
    results_buffer.copyToHost(&results) catch |err| return r.reportError("stringCountKernel", "copy the results of", err);
    for (needles_text, results) |text, result| {
        r.expect(0, @as(u32, @intCast(std.mem.count(u8, haystack, text))), result);
    }

    // The functions of `std.mem` over one text, whose results the kernel writes one by one.
    const text = "Hello brave new GPU world";
    const text_buffer = r.upload(u8, text) catch |err| return r.reportError("stringUtilsKernel", "upload to", err);
    defer text_buffer.free();
    const utils_buffer = r.context.alloc(u32, 6) catch |err| return r.reportError("stringUtilsKernel", "allocate for", err);
    defer utils_buffer.free();
    r.runLinear("stringUtilsKernel", 1, 1, .{ text_buffer, @as(u32, text.len), utils_buffer });
    var utils: [6]u32 = undefined;
    utils_buffer.copyToHost(&utils) catch |err| return r.reportError("stringUtilsKernel", "copy the results of", err);
    r.expect(0, @as(u32, @intFromBool(std.mem.startsWith(u8, text, "Hello"))), utils[0]);
    r.expect(1, @as(u32, @intFromBool(std.mem.endsWith(u8, text, "GPU"))), utils[1]);
    r.expect(2, @as(u32, @intFromBool(std.mem.containsAtLeast(u8, text, 1, "the"))), utils[2]);
    r.expect(3, @as(u32, @intFromBool(std.mem.eql(u8, text[0..5], "Hello"))), utils[3]);
    r.expect(4, @as(u32, @intCast(std.mem.indexOfScalar(u8, text, ' ').?)), utils[4]);
    r.expect(5, @as(u32, @intCast(std.mem.lastIndexOfScalar(u8, text, ' ').?)), utils[5]);

    // The patterns of several texts.
    const patterns = [_][]const u8{ "GPU", "CUDA", "parallel", "fast", "compute" };
    const texts = [_][]const u8{
        "GPU and CUDA are fast",
        "parallel compute is fast and parallel",
        "GPU",
        "nothing here",
        "fast fast fast",
        "CUDA GPU parallel compute fast",
    };
    const count = texts.len;
    var text_records: [count][128]u8 = undefined;
    var text_lens: [count]u32 = undefined;
    for (&text_records, &text_lens, texts) |*record, *length, source| {
        @memset(record, 0);
        @memcpy(record[0..source.len], source);
        length.* = @intCast(source.len);
    }
    const records_buffer = r.upload([128]u8, &text_records) catch |err| return r.reportError("multiPatternKernel", "upload to", err);
    defer records_buffer.free();
    const text_lens_buffer = r.upload(u32, &text_lens) catch |err| return r.reportError("multiPatternKernel", "upload to", err);
    defer text_lens_buffer.free();
    const counts_buffer = r.context.alloc(u32, count) catch |err| return r.reportError("multiPatternKernel", "allocate for", err);
    defer counts_buffer.free();
    r.runLinear("multiPatternKernel", count, 32, .{ records_buffer, text_lens_buffer, counts_buffer, @as(u32, count) });
    var counts: [count]u32 = undefined;
    counts_buffer.copyToHost(&counts) catch |err| return r.reportError("multiPatternKernel", "copy the results of", err);
    for (texts, counts) |source, result| {
        var expected: u32 = 0;
        for (patterns) |pattern| expected += @intCast(std.mem.count(u8, source, pattern));
        r.expect(0, expected, result);
    }

    // Every thread searches a part of the haystack, and a part overlaps the one before it, so
    // that a needle that starts at the end of a part is still found.
    var long_haystack: [200]u8 = undefined;
    const phrase = "the lazy dog sleeps. ";
    for (&long_haystack, 0..) |*byte, i| byte.* = phrase[i % phrase.len];
    const long_buffer = r.upload(u8, &long_haystack) catch |err| return r.reportError("optimizedSearchKernel", "upload to", err);
    defer long_buffer.free();
    const needle = "dog";
    const needle_buffer = r.upload(u8, needle) catch |err| return r.reportError("optimizedSearchKernel", "upload to", err);
    defer needle_buffer.free();
    const thread_count = 8;
    const search_buffer = r.context.alloc(u32, thread_count) catch |err| return r.reportError("optimizedSearchKernel", "allocate for", err);
    defer search_buffer.free();
    r.runLinear("optimizedSearchKernel", thread_count, 8, .{
        long_buffer, @as(u32, long_haystack.len), needle_buffer, @as(u32, needle.len), @as(u32, thread_count), search_buffer,
    });
    var positions: [thread_count]u32 = undefined;
    search_buffer.copyToHost(&positions) catch |err| return r.reportError("optimizedSearchKernel", "copy the results of", err);
    const chunk_size = long_haystack.len / thread_count;
    for (positions, 0..) |position, gid| {
        const start = gid * chunk_size;
        const end = if (gid == thread_count - 1) long_haystack.len else @min((gid + 1) * chunk_size + needle.len, long_haystack.len);
        const expected = if (std.mem.indexOf(u8, long_haystack[start..end], needle)) |found| @as(u32, @intCast(start + found)) else std.math.maxInt(u32);
        r.expect(gid, expected, position);
    }
}

fn testJson(r: *Runner) void {
    const people = [_][]const u8{
        "{\"name\": \"ann\", \"age\": 30, \"score\": 91.5}",
        "{\"name\":\"bo\",\"age\":7,\"score\":0.25}",
        "{\"name\": \"cy\", \"age\": 44, \"score\": 12.5, \"city\": \"Oslo\"}",
        "{ this is not json",
        "{\"name\": \"dee\", \"age\": 68, \"score\": 7.75, \"nested\": {\"x\": 1, \"y\": [1, 2, 3]}}",
    };
    const n = people.len;
    var texts: [n][256]u8 = undefined;
    var text_lens: [n]u32 = undefined;
    for (&texts, &text_lens, people) |*record, *length, source| {
        @memset(record, 0);
        @memcpy(record[0..source.len], source);
        length.* = @intCast(source.len);
    }

    const texts_buffer = r.upload([256]u8, &texts) catch |err| return r.reportError("jsonParseKernel", "upload to", err);
    defer texts_buffer.free();
    const text_lens_buffer = r.upload(u32, &text_lens) catch |err| return r.reportError("jsonParseKernel", "upload to", err);
    defer text_lens_buffer.free();
    const ages_buffer = r.context.alloc(u32, n) catch |err| return r.reportError("jsonParseKernel", "allocate for", err);
    defer ages_buffer.free();
    const scores_buffer = r.context.alloc(f32, n) catch |err| return r.reportError("jsonParseKernel", "allocate for", err);
    defer scores_buffer.free();

    // Four threads of a block can parse at the same time, because every one of them has its own
    // heap in shared memory.
    const json_threads = 4;
    r.runLinear("jsonParseKernel", n, json_threads, .{ texts_buffer, text_lens_buffer, @as(u32, n), ages_buffer, scores_buffer });
    var ages: [n]u32 = undefined;
    ages_buffer.copyToHost(&ages) catch |err| return r.reportError("jsonParseKernel", "copy the results of", err);
    var scores: [n]f32 = undefined;
    scores_buffer.copyToHost(&scores) catch |err| return r.reportError("jsonParseKernel", "copy the results of", err);
    const expected_ages = [_]u32{ 30, 7, 44, std.math.maxInt(u32), 68 };
    const expected_scores = [_]f32{ 91.5, 0.25, 12.5, -1, 7.75 };
    r.expectSlice(&expected_ages, &ages);
    r.expectSlice(&expected_scores, &scores);

    // Nested objects and a slice of them.
    const games = [_][]const u8{
        "{\"level\": 3, \"players\": [{\"id\": 11, \"health\": 90.5, \"position\": {\"x\": 1.5, \"y\": 2.5, \"z\": 3.5}}, {\"id\": 12, \"health\": 50, \"position\": {\"x\": 0, \"y\": 0, \"z\": 0}}], \"timestamp\": 1700000000}",
        "{\"level\": 9, \"players\": [], \"timestamp\": 0}",
        "{\"level\": 4, \"players\": [{\"id\": 7, \"health\": 1, \"position\": {\"x\": -1, \"y\": -2, \"z\": -3}}], \"timestamp\": 12}",
    };
    const game_count = games.len;
    var game_texts: [game_count][512]u8 = undefined;
    var game_lens: [game_count]u32 = undefined;
    for (&game_texts, &game_lens, games) |*record, *length, source| {
        @memset(record, 0);
        @memcpy(record[0..source.len], source);
        length.* = @intCast(source.len);
    }
    const game_texts_buffer = r.upload([512]u8, &game_texts) catch |err| return r.reportError("jsonParseNestedKernel", "upload to", err);
    defer game_texts_buffer.free();
    const game_lens_buffer = r.upload(u32, &game_lens) catch |err| return r.reportError("jsonParseNestedKernel", "upload to", err);
    defer game_lens_buffer.free();
    const levels_buffer = r.context.alloc(u32, game_count) catch |err| return r.reportError("jsonParseNestedKernel", "allocate for", err);
    defer levels_buffer.free();
    const players_buffer = r.context.alloc(u32, game_count) catch |err| return r.reportError("jsonParseNestedKernel", "allocate for", err);
    defer players_buffer.free();
    const ids_buffer = r.context.alloc(u32, game_count) catch |err| return r.reportError("jsonParseNestedKernel", "allocate for", err);
    defer ids_buffer.free();
    r.runLinear("jsonParseNestedKernel", game_count, json_threads, .{
        game_texts_buffer, game_lens_buffer, @as(u32, game_count), levels_buffer, players_buffer, ids_buffer,
    });
    var levels: [game_count]u32 = undefined;
    levels_buffer.copyToHost(&levels) catch |err| return r.reportError("jsonParseNestedKernel", "copy the results of", err);
    var player_counts: [game_count]u32 = undefined;
    players_buffer.copyToHost(&player_counts) catch |err| return r.reportError("jsonParseNestedKernel", "copy the results of", err);
    var first_ids: [game_count]u32 = undefined;
    ids_buffer.copyToHost(&first_ids) catch |err| return r.reportError("jsonParseNestedKernel", "copy the results of", err);
    r.expectSlice(&[_]u32{ 3, 9, 4 }, &levels);
    r.expectSlice(&[_]u32{ 2, 0, 1 }, &player_counts);
    r.expectSlice(&[_]u32{ 11, 0, 7 }, &first_ids);

    // An array of objects, parsed by one thread with a bump allocator in shared memory.
    const array_text = "[{\"name\": \"aa\", \"age\": 1, \"score\": 1}, {\"name\": \"bb\", \"age\": 2, \"score\": 2}, {\"name\": \"cc\", \"age\": 3, \"score\": 3}]";
    const array_buffer = r.upload(u8, array_text) catch |err| return r.reportError("jsonArrayKernel", "upload to", err);
    defer array_buffer.free();
    const array_count_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("jsonArrayKernel", "allocate for", err);
    defer array_count_buffer.free();
    const array_age_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("jsonArrayKernel", "allocate for", err);
    defer array_age_buffer.free();
    const array_total_buffer = r.context.alloc(u64, 1) catch |err| return r.reportError("jsonArrayKernel", "allocate for", err);
    defer array_total_buffer.free();
    r.runLinear("jsonArrayKernel", 1, 1, .{ array_buffer, @as(u32, array_text.len), array_count_buffer, array_age_buffer, array_total_buffer });
    var array_count: [1]u32 = undefined;
    array_count_buffer.copyToHost(&array_count) catch |err| return r.reportError("jsonArrayKernel", "copy the results of", err);
    var array_age: [1]u32 = undefined;
    array_age_buffer.copyToHost(&array_age) catch |err| return r.reportError("jsonArrayKernel", "copy the results of", err);
    var array_total: [1]u64 = undefined;
    array_total_buffer.copyToHost(&array_total) catch |err| return r.reportError("jsonArrayKernel", "copy the results of", err);
    r.expect(0, @as(u32, 3), array_count[0]);
    r.expect(0, @as(u32, 1), array_age[0]);
    r.expect(0, @as(u64, 6), array_total[0]);
}

fn testDynamic(r: *Runner) void {
    const n = 100;
    var input: [n]u32 = undefined;
    for (&input, 0..) |*value, i| value.* = @intCast((i * 7 + 3) % 100);

    var expected: [2 * n]u32 = @splat(0);
    var expected_count: u32 = 0;
    for (input) |value| {
        if (value > 50) {
            expected[expected_count] = value;
            expected[expected_count + 1] = value * 2;
            expected_count += 2;
        }
    }

    const input_buffer = r.upload(u32, &input) catch |err| return r.reportError("arrayListKernel", "upload to", err);
    defer input_buffer.free();
    const output_buffer = r.context.alloc(u32, n) catch |err| return r.reportError("arrayListKernel", "allocate for", err);
    defer output_buffer.free();
    const count_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("arrayListKernel", "allocate for", err);
    defer count_buffer.free();

    r.runLinear("arrayListKernel", 1, 8, .{ input_buffer, output_buffer, count_buffer, @as(u32, n) });
    var actual: [n]u32 = undefined;
    output_buffer.copyToHost(&actual) catch |err| return r.reportError("arrayListKernel", "copy the results of", err);
    var count: [1]u32 = undefined;
    count_buffer.copyToHost(&count) catch |err| return r.reportError("arrayListKernel", "copy the results of", err);
    r.expect(0, expected_count, count[0]);
    r.expectSlice(expected[0..expected_count], actual[0..expected_count]);

    // The length of a message formatted on the device, checked against the same formatting here.
    const format_n = 64;
    var values: [format_n]u32 = undefined;
    for (&values, 0..) |*value, i| value.* = @truncate(i * 12345 + 7);
    const values_buffer = r.upload(u32, &values) catch |err| return r.reportError("formatStringsKernel", "upload to", err);
    defer values_buffer.free();
    const lengths_buffer = r.context.alloc(u32, format_n) catch |err| return r.reportError("formatStringsKernel", "allocate for", err);
    defer lengths_buffer.free();
    r.runLinear("formatStringsKernel", format_n, format_n, .{ values_buffer, lengths_buffer, @as(u32, format_n) });
    var lengths: [format_n]u32 = undefined;
    lengths_buffer.copyToHost(&lengths) catch |err| return r.reportError("formatStringsKernel", "copy the results of", err);
    for (values, lengths, 0..) |value, length, i| {
        const expected_length = std.fmt.count("Thread {d} processed value: {d} (hex: 0x{x})", .{ i, value, value });
        r.expect(i, @as(u32, @intCast(expected_length)), length);
    }
}

fn testPrintf(r: *Runner) void {
    // One thread, so that the order of the lines is the order of the calls of that thread.
    var capture = StdoutCapture.begin() catch |err| return r.reportError("printfKernel", "capture the output of", err);
    defer capture.deinit();
    r.runLinear("printfKernel", 1, 1, .{});
    const printed = capture.end();

    r.expectContains(printed, "Hello from thread 0 in block 0 (global id: 0)\n");
    r.expectContains(printed, "=== Kernel Launch Summary ===\n");
    r.expectContains(printed, "Grid size: 1 blocks\n");
    r.expectContains(printed, "Block size: 1 threads\n");
    r.expectContains(printed, "Total threads: 1, 100% of them printed\n");
    // A '%' in the text of a message is not part of a format, which the static buffer of
    // `std.gpu.print` makes possible.
    r.expectContains(printed, "100%");

    var capture_math = StdoutCapture.begin() catch |err| return r.reportError("mathPrintfKernel", "capture the output of", err);
    defer capture_math.deinit();
    r.runLinear("mathPrintfKernel", 1, 1, .{@as(f32, 1.5)});
    const printed_math = capture_math.end();

    r.expectContains(printed_math, "Input: 1.5\n");
    r.expectContains(printed_math, "  sin: ");
    r.expectContains(printed_math, "  cos: ");
    // The square root is the same on both sides: it is one of the functions that the hardware
    // rounds correctly.
    var expected_line: [64]u8 = undefined;
    const line = std.fmt.bufPrint(&expected_line, "  sqrt: {d}\n", .{std.math.sqrt(@as(f32, 1.5))}) catch |err|
        return r.reportError("mathPrintfKernel", "format the expected line of", err);
    r.expectContains(printed_math, line);

    // The values of a computation and the line that the kernel prints for one of them.
    const n = 6;
    var input: [n]f32 = undefined;
    for (&input, 0..) |*value, i| value.* = @floatFromInt(i + 1);
    const input_buffer = r.upload(f32, &input) catch |err| return r.reportError("debugComputeKernel", "upload to", err);
    defer input_buffer.free();
    const output_buffer = r.context.alloc(f32, n) catch |err| return r.reportError("debugComputeKernel", "allocate for", err);
    defer output_buffer.free();

    r.runLinear("debugComputeKernel", n, n, .{ input_buffer, output_buffer, @as(u32, n) });
    var actual: [n]f32 = undefined;
    output_buffer.copyToHost(&actual) catch |err| return r.reportError("debugComputeKernel", "copy the results of", err);
    for (input, actual) |value, result| {
        r.expect(0, value * value + 2 * value + 1, result);
    }

    var capture_debug = StdoutCapture.begin() catch |err| return r.reportError("debugComputeKernel", "capture the output of", err);
    defer capture_debug.deinit();
    r.runLinear("debugComputeKernel", 1, 1, .{ input_buffer, output_buffer, @as(u32, 1) });
    const printed_debug = capture_debug.end();

    var expected_debug: [64]u8 = undefined;
    const debug_line = std.fmt.bufPrint(&expected_debug, "[0] input {d} output {d}\n", .{ @as(f32, 1), @as(f32, 4) }) catch |err|
        return r.reportError("debugComputeKernel", "format the expected line of", err);
    r.expectContains(printed_debug, debug_line);
}

fn testHello(r: *Runner) void {
    const out_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("helloKernel", "allocate for", err);
    defer out_buffer.free();
    out_buffer.zero() catch |err| return r.reportError("helloKernel", "clear", err);
    r.runLinear("helloKernel", 1, 1, .{ out_buffer, @as(usize, 1) });
    var out: [1]u32 = undefined;
    out_buffer.copyToHost(&out) catch |err| return r.reportError("helloKernel", "copy the results of", err);
    r.expect(0, @as(u32, 42), out[0]);
}

/// The device and this process both compute the builtin math functions with the same
/// compiler-rt code, so their results must be identical, bit for bit. The inputs cover typical
/// arguments, large arguments, random bit patterns across every exponent, arguments right next
/// to multiples of pi/2, and the range where `@exp` is finite. A different math library on
/// either side, or a float operation that the PTX assembler contracts into a fused multiply-add,
/// changes some of these results.
fn testBuiltinMath(r: *Runner) void {
    testBuiltinMathSweep(f32, r, "builtinMathF32Kernel");
    testBuiltinMathSweep(f64, r, "builtinMathF64Kernel");
}

fn testBuiltinMathSweep(comptime T: type, r: *Runner, comptime kernel: [:0]const u8) void {
    const class_len = 1 << 17;
    const n = 5 * class_len;
    const gpa = std.heap.page_allocator;
    const inputs = gpa.alloc(T, n) catch |err| return r.reportError(kernel, "allocate the inputs of", err);
    defer gpa.free(inputs);
    const results = gpa.alloc(T, n * 8) catch |err| return r.reportError(kernel, "allocate the results of", err);
    defer gpa.free(results);

    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();
    for (inputs, 0..) |*x, i| {
        const u = random.float(T);
        x.* = switch (i / class_len) {
            0 => (2 * u - 1) * 2 * std.math.pi,
            1 => (2 * u - 1) * 1e6,
            2 => while (true) {
                const bits: T = @bitCast(random.int(@Int(.unsigned, @bitSizeOf(T))));
                if (std.math.isFinite(bits)) break bits;
            },
            3 => @as(T, @floatFromInt(random.intRangeAtMost(i32, -100_000, 100_000))) * (std.math.pi / 2.0) + (u - 0.5) * 1e-6,
            else => (2 * u - 1) * 750,
        };
    }

    const input_buffer = r.upload(T, inputs) catch |err| return r.reportError(kernel, "upload to", err);
    defer input_buffer.free();
    const output_buffer = r.context.alloc(T, n * 8) catch |err| return r.reportError(kernel, "allocate for", err);
    defer output_buffer.free();
    r.runLinear(kernel, n, 256, .{ input_buffer, output_buffer, @as(u32, n) });
    output_buffer.copyToHost(results) catch |err| return r.reportError(kernel, "copy the results of", err);
    for (inputs, 0..) |x, index| {
        const expected = mathResults(T, x);
        r.expectSlice(&expected, results[index * 8 ..][0..8]);
    }
}

/// The values of the builtin math functions of `x`, in the order that the kernels write them.
fn mathResults(comptime T: type, x: T) [8]T {
    return .{ @sin(x), @cos(x), @tan(x), @exp(x), @exp2(x), @log(x), @log2(x), @log10(x) };
}

fn testF128(r: *Runner) void {
    const a_values = [_]f64{ 0.5, 1.5, 2.25, 3.75, 1e10, 1e-5 };
    const b_values = [_]f64{ 2, 4, 0.5, 1.25, 3, 7 };
    const n = a_values.len;

    const a_buffer = r.upload(f64, &a_values) catch |err| return r.reportError("f128Kernel", "upload to", err);
    defer a_buffer.free();
    const b_buffer = r.upload(f64, &b_values) catch |err| return r.reportError("f128Kernel", "upload to", err);
    defer b_buffer.free();
    const output_buffer = r.context.alloc(f128, n * 4) catch |err| return r.reportError("f128Kernel", "allocate for", err);
    defer output_buffer.free();

    r.runLinear("f128Kernel", n, 32, .{ a_buffer, b_buffer, output_buffer, @as(u32, n) });
    var results: [n * 4]f128 = undefined;
    output_buffer.copyToHost(&results) catch |err| return r.reportError("f128Kernel", "copy the results of", err);
    for (a_values, b_values, 0..) |a, b, index| {
        const x: f128 = a;
        const y: f128 = b;
        r.expect(index * 4 + 0, x + y, results[index * 4 + 0]);
        r.expect(index * 4 + 1, x * y, results[index * 4 + 1]);
        r.expect(index * 4 + 2, x / y, results[index * 4 + 2]);
        r.expect(index * 4 + 3, @sqrt(x), results[index * 4 + 3]);
    }

    // The comparisons of `f128` values, and the narrowing of one of them.
    const compare_a = [_]f128{ 0.5, 1, 2, 1e4000, -1.5 };
    const compare_b = [_]f128{ 0.5, 3, 2, 1e4000, 2.5 };
    const compare_n = compare_a.len;
    const compare_a_buffer = r.upload(f128, &compare_a) catch |err| return r.reportError("f128CompareKernel", "upload to", err);
    defer compare_a_buffer.free();
    const compare_b_buffer = r.upload(f128, &compare_b) catch |err| return r.reportError("f128CompareKernel", "upload to", err);
    defer compare_b_buffer.free();
    const flags_buffer = r.context.alloc(u32, compare_n * 3) catch |err| return r.reportError("f128CompareKernel", "allocate for", err);
    defer flags_buffer.free();
    const narrowed_buffer = r.context.alloc(f64, compare_n) catch |err| return r.reportError("f128CompareKernel", "allocate for", err);
    defer narrowed_buffer.free();

    r.runLinear("f128CompareKernel", compare_n, 32, .{
        compare_a_buffer, compare_b_buffer, flags_buffer, @as(u32, compare_n), narrowed_buffer,
    });
    var flags: [compare_n * 3]u32 = undefined;
    flags_buffer.copyToHost(&flags) catch |err| return r.reportError("f128CompareKernel", "copy the results of", err);
    var narrowed: [compare_n]f64 = undefined;
    narrowed_buffer.copyToHost(&narrowed) catch |err| return r.reportError("f128CompareKernel", "copy the results of", err);
    for (compare_a, compare_b, 0..) |x, y, index| {
        var expected: u32 = 0;
        if (x < y) expected |= 1;
        if (x == y) expected |= 2;
        if (x > y) expected |= 4;
        r.expect(index * 3 + 0, expected, flags[index * 3 + 0]);
        r.expect(index * 3 + 1, @as(u32, @intFromBool(@as(f32, @floatCast(x)) == @as(f32, @floatCast(y)))), flags[index * 3 + 1]);
        r.expect(index * 3 + 2, @as(u32, @intFromBool(@as(f64, @floatCast(x)) == 0.5)), flags[index * 3 + 2]);
        r.expect(index, @as(f64, @floatCast(x)), narrowed[index]);
    }
}

fn testParseFloat(r: *Runner) void {
    const texts = [_][]const u8{
        "3.14159",
        "0.5",
        "-2.5e-3",
        "1e10",
        "123.45678901234567890",
        "1.7976931348623157e308",
        "1e-4900",
        "1e4000",
        "0",
        "not a number",
    };
    const n = texts.len;
    var records: [n][32]u8 = undefined;
    var lengths: [n]u32 = undefined;
    for (&records, &lengths, texts) |*record, *length, source| {
        @memset(record, 0);
        @memcpy(record[0..source.len], source);
        length.* = @intCast(source.len);
    }

    const records_buffer = r.upload([32]u8, &records) catch |err| return r.reportError("parseFloatKernel", "upload to", err);
    defer records_buffer.free();
    const lengths_buffer = r.upload(u32, &lengths) catch |err| return r.reportError("parseFloatKernel", "upload to", err);
    defer lengths_buffer.free();
    const f32_results = r.context.alloc(f32, n) catch |err| return r.reportError("parseFloatKernel", "allocate for", err);
    defer f32_results.free();
    const f64_results = r.context.alloc(f64, n) catch |err| return r.reportError("parseFloatKernel", "allocate for", err);
    defer f64_results.free();
    const f128_results = r.context.alloc(f128, n) catch |err| return r.reportError("parseFloatKernel", "allocate for", err);
    defer f128_results.free();

    r.runLinear("parseFloatKernel", n, 32, .{ records_buffer, lengths_buffer, f32_results, f64_results, f128_results, @as(u32, n) });
    var actual32: [n]f32 = undefined;
    f32_results.copyToHost(&actual32) catch |err| return r.reportError("parseFloatKernel", "copy the results of", err);
    var actual64: [n]f64 = undefined;
    f64_results.copyToHost(&actual64) catch |err| return r.reportError("parseFloatKernel", "copy the results of", err);
    var actual128: [n]f128 = undefined;
    f128_results.copyToHost(&actual128) catch |err| return r.reportError("parseFloatKernel", "copy the results of", err);

    for (texts, 0..) |text, index| {
        // The same parser runs here and on the device, so the results must be equal, bit for bit.
        r.expect(index, std.fmt.parseFloat(f32, text) catch std.math.nan(f32), actual32[index]);
        r.expect(index, std.fmt.parseFloat(f64, text) catch std.math.nan(f64), actual64[index]);
        const expected: f128 = std.fmt.parseFloat(f128, text) catch std.math.nan(f128);
        const expected_bits: u128 = @bitCast(expected);
        const actual_bits: u128 = @bitCast(actual128[index]);
        r.expect(index, expected_bits, actual_bits);
    }
}

fn testDeviceHeap(r: *Runner) void {
    const list_len_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("deviceHeapKernel", "allocate for", err);
    defer list_len_buffer.free();
    const list_sum_buffer = r.context.alloc(u64, 1) catch |err| return r.reportError("deviceHeapKernel", "allocate for", err);
    defer list_sum_buffer.free();
    const aligned_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("deviceHeapKernel", "allocate for", err);
    defer aligned_buffer.free();

    // The kernel and the driver share the heap that `malloc` in a kernel allocates from, so this
    // also checks that the limit of the context is the size of that heap.
    r.runLinear("deviceHeapKernel", 1, 1, .{ list_len_buffer, list_sum_buffer, aligned_buffer });
    var list_len: [1]u32 = undefined;
    list_len_buffer.copyToHost(&list_len) catch |err| return r.reportError("deviceHeapKernel", "copy the results of", err);
    var list_sum: [1]u64 = undefined;
    list_sum_buffer.copyToHost(&list_sum) catch |err| return r.reportError("deviceHeapKernel", "copy the results of", err);
    var aligned: [1]u32 = undefined;
    aligned_buffer.copyToHost(&aligned) catch |err| return r.reportError("deviceHeapKernel", "copy the results of", err);

    const list_length = 500;
    r.expect(0, @as(u32, list_length), list_len[0]);
    var expected_sum: u64 = 0;
    for (0..list_length) |i| expected_sum += @as(u64, i) * i;
    r.expect(0, expected_sum, list_sum[0]);
    r.expect(1, @as(u32, 1), aligned[0]);
}

fn testBumpAllocator(r: *Runner) void {
    const intact_buffer = r.context.alloc(u32, 256) catch |err| return r.reportError("bumpAllocatorKernel", "allocate for", err);
    defer intact_buffer.free();
    const violations_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("bumpAllocatorKernel", "allocate for", err);
    defer violations_buffer.free();
    const used_buffer = r.context.alloc(u32, 1) catch |err| return r.reportError("bumpAllocatorKernel", "allocate for", err);
    defer used_buffer.free();

    r.runLinear("bumpAllocatorKernel", 256, 256, .{ intact_buffer, violations_buffer, used_buffer });
    var intact: [256]u32 = undefined;
    intact_buffer.copyToHost(&intact) catch |err| return r.reportError("bumpAllocatorKernel", "copy the results of", err);
    var violations: [1]u32 = undefined;
    violations_buffer.copyToHost(&violations) catch |err| return r.reportError("bumpAllocatorKernel", "copy the results of", err);
    var used: [1]u32 = undefined;
    used_buffer.copyToHost(&used) catch |err| return r.reportError("bumpAllocatorKernel", "copy the results of", err);

    const expected_intact: [256]u32 = @splat(1);
    r.expectSlice(&expected_intact, &intact);
    r.expect(0, @as(u32, 0), violations[0]);
    // Every thread allocated four bytes, and the allocator keeps its offset at the start of the
    // heap, so the bytes in use are those of the allocations plus that offset.
    r.expect(0, true, used[0] >= 256 * @sizeOf(u32) and used[0] <= 4096);
}
