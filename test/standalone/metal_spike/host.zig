//! Host side of the Metal spike: runs the kernels of a `.metallib` on the GPU of a Mac and
//! checks their results against the same computation on the CPU.
//!
//! The program is cross-compiled from Linux, where there is no Objective-C compiler and no macOS
//! SDK, so it declares nothing at link time. It loads, with `dlopen`,
//!
//!   * `/usr/lib/libobjc.A.dylib`, for `objc_getClass`, `sel_registerName`, `objc_msgSend` and
//!     the autorelease pool,
//!   * `/usr/lib/libSystem.B.dylib`, for `dispatch_data_create`, which is how the bytes of a
//!     `.metallib` reach `newLibraryWithData:error:`, and
//!   * `/System/Library/Frameworks/Metal.framework/Metal`, for `MTLCreateSystemDefaultDevice`,
//!
//! and sends every Objective-C message through `objc_msgSend`. On arm64 there is no variadic
//! `objc_msgSend`: the arguments and the return value are in registers, and a call through a
//! pointer of the wrong type corrupts them silently. Every selector that this program sends
//! therefore has its own function pointer type, and `call` is the only place that casts.
//!
//! This is the prototype of a future `std.gpu.metal`: the sequence below -- load the runtime,
//! keep buffers in shared memory, build a compute pipeline, encode a dispatch, wait -- is the
//! shape that the standard library should have. What disappears then is the run-time lookup of
//! the libraries and the explicit `MsgSend` casts, which `@cImport` and the SDK replace.
//!
//! ```sh
//! host <path-to.metallib>   # run the spike kernels and check their results
//! host --selftest           # report what the Metal runtime has; needs no metallib
//! ```
//!
//! The tests print one line each, `--- PASS:`, `--- FAIL:` or `--- SKIP:`, and the last line is
//! the summary, which starts with `PASS` or `FAIL`. A kernel that the `.metallib` does not
//! contain is skipped rather than failed, so a pipeline that is still partial can be tested; on
//! a machine without the Metal runtime -- the Linux machine that cross-compiles this program --
//! everything is skipped, which is how the cross-compiled binary is smoke-tested without a Mac.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// The start of every line that is not a test verdict.
const prefix = "metal spike: ";

/// The kernels of the spike, in the order that the tests are reported. One test per kernel.
const kernel_names = [_][]const u8{ "vadd", "reduce", "parsef" };

/// The largest `.metallib` that this program reads. The libraries of the spike are a few
/// kilobytes; this only stops a wrong path (a directory, a disk image) from being read whole.
const max_metallib_size = 64 * 1024 * 1024;

/// The libraries that the program loads at run time, in the order that `--selftest` reports them.
const objc_library_path = "/usr/lib/libobjc.A.dylib";
const system_library_path = "/usr/lib/libSystem.B.dylib";
const metal_library_path = "/System/Library/Frameworks/Metal.framework/Metal";

/// `NSUInteger` on 64-bit Darwin. There is no header to import it from, so the Darwin types that
/// the messages below use are spelled out here.
const NSUInteger = u64;

/// `MTLResourceStorageModeShared`: the CPU and the GPU share the memory of the buffer, so the
/// tests read the results of a kernel in place, without a blit.
const resource_storage_mode_shared: NSUInteger = 0;

/// `MTLSize`: the width, height and depth of a grid or of a threadgroup.
const MTLSize = extern struct {
    width: NSUInteger,
    height: NSUInteger,
    depth: NSUInteger,

    /// A one-dimensional size: every kernel of the spike is dispatched with one.
    fn linear(size: usize) MTLSize {
        return .{ .width = size, .height = 1, .depth = 1 };
    }
};

// ---------------------------------------------------------------- the symbols

/// `id objc_getClass(const char *name)`: the class object of a class, such as `NSString`.
const GetClass = *const fn (name: [*:0]const u8) callconv(.c) ?*anyopaque;

/// `SEL sel_registerName(const char *name)`: the selector of a method, such as
/// `"setBuffer:offset:atIndex:"`, registering it with the runtime the first time it is asked for.
const RegisterSelector = *const fn (name: [*:0]const u8) callconv(.c) ?*anyopaque;

/// `void *objc_autoreleasePoolPush(void)`.
const PoolPush = *const fn () callconv(.c) ?*anyopaque;

/// `void objc_autoreleasePoolPop(void *pool)`.
const PoolPop = *const fn (pool: ?*anyopaque) callconv(.c) void;

/// `dispatch_data_t dispatch_data_create(const void *buffer, size_t size, dispatch_queue_t
/// queue, dispatch_block_t destructor)`. With a null destructor libdispatch copies the buffer,
/// which is how the bytes of a `.metallib` reach the Metal runtime; this program keeps them
/// alive until the end of the run anyway.
const DispatchDataCreate = *const fn (
    buffer: ?*const anyopaque,
    size: usize,
    queue: ?*anyopaque,
    destructor: ?*anyopaque,
) callconv(.c) ?*anyopaque;

/// `id MTLCreateSystemDefaultDevice(void)`: the GPU of the machine, "system default" being the
/// discrete GPU on a Mac that has one, and the integrated GPU otherwise.
const CreateSystemDefaultDevice = *const fn () callconv(.c) ?*anyopaque;

// ----------------------------------------------------------- the message types
//
// `objc_msgSend` has no prototype: it takes the receiver, the selector, and then exactly the
// arguments of the method. Each type below is the signature of one shape of message, and `call`
// casts the address of `objc_msgSend` to it. The rules of the target that these follow:
//
//   * `?*anyopaque` stands for an Objective-C object, a `SEL`, or a `void *`: nothing here can
//     check what a message actually returns or takes, which is what the SDK would do.
//   * `NSUInteger` is `u64` and `NSInteger` is `i64`.
//   * `BOOL` is `i8`, not a Zig `bool`: on arm64 it is `signed char`, and a message that returns
//     it leaves the rest of the register undefined.
//   * an argument that is a struct of more than 16 bytes is passed by reference on arm64: the
//     caller makes a copy and passes a pointer to it. `MTLSize` is 24 bytes, so a by-value
//     `MTLSize` argument is a `*const MTLSize` here, and the method sees exactly what a C
//     compiler would have passed. (The copy is writable in the ABI; Metal only reads it.)

/// `- (NSString *)name`, `- (void *)contents`, `- (id)commandBuffer`, `- (id)computeCommandEncoder`,
/// `- (id)newCommandQueue`, `- (NSString *)localizedDescription`: one object in, one object out.
const MsgSendObject = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;

/// `- (void)endEncoding`, `- (void)commit`, `- (void)waitUntilCompleted`: no arguments, no value.
const MsgSendVoid = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void;

/// `- (id)newFunctionWithName:(NSString *)name`.
const MsgSendObjectWithObject = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;

/// `- (void)setComputePipelineState:(id<MTLComputePipelineState>)state`.
const MsgSendVoidWithObject = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) void;

/// `- (id)newLibraryWithData:(dispatch_data_t)data error:(NSError **)error` and
/// `- (id)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError **)error`:
/// the two messages that report why they failed. A null return without the `NSError` is a dead
/// end, which is why both of them are passed one and the message is printed.
const MsgSendObjectWithObjectError = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, *?*anyopaque) callconv(.c) ?*anyopaque;

/// `- (id)stringWithUTF8String:(const char *)text`, a class method of `NSString`.
const MsgSendObjectWithCString = *const fn (?*anyopaque, ?*anyopaque, [*:0]const u8) callconv(.c) ?*anyopaque;

/// `- (const char *)UTF8String`, the bytes of an `NSString`, which is how this program reads one.
const MsgSendCString = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?[*:0]const u8;

/// `- (unsigned long long)registryID`.
const MsgSendNSUInteger = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) NSUInteger;

/// `- (BOOL)supportsFamily:(MTLGPUFamily)family`.
const MsgSendI8WithI64 = *const fn (?*anyopaque, ?*anyopaque, i64) callconv(.c) i8;

/// `- (id)newBufferWithBytes:(const void *)bytes length:(NSUInteger)length
/// options:(MTLResourceOptions)options`: the buffer copies the bytes it is given, and the copy
/// is what the GPU reads and writes.
const MsgSendObjectWithBytes = *const fn (?*anyopaque, ?*anyopaque, ?*const anyopaque, NSUInteger, NSUInteger) callconv(.c) ?*anyopaque;

/// `- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index`.
const MsgSendVoidSetBuffer = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, NSUInteger, NSUInteger) callconv(.c) void;

/// `- (void)dispatchThreads:(MTLSize)threadsPerGrid
/// threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup`. The two sizes are past the third
/// argument because they are structs of more than 16 bytes; see above.
const MsgSendVoidDispatchThreads = *const fn (?*anyopaque, ?*anyopaque, *const MTLSize, *const MTLSize) callconv(.c) void;

/// The type of the value that a method with the signature `MsgSend` returns.
fn ReturnType(comptime MsgSend: type) type {
    return @typeInfo(@typeInfo(MsgSend).pointer.child).@"fn".return_type.?;
}

/// Sends one Objective-C message and returns what the method returns.
///
/// `receiver` is the object, `selector` is the `SEL` of the method, `args` are the arguments
/// after the selector with exactly the types of `MsgSend`, and `msg_send` is the address of
/// `objc_msgSend` that `loadRuntime` looked up. The cast to `MsgSend` is what makes the call
/// well formed: on arm64 the arguments go in registers and there is no variadic form of
/// `objc_msgSend` to fall back on, so the type of the function pointer is the whole ABI.
fn call(
    msg_send: *const anyopaque,
    comptime MsgSend: type,
    receiver: ?*anyopaque,
    selector: ?*anyopaque,
    args: anytype,
) ReturnType(MsgSend) {
    const method: MsgSend = @ptrCast(@alignCast(msg_send));
    return @call(.auto, method, .{ receiver, selector } ++ args);
}

// ---------------------------------------------------------------- the runtime

/// The Objective-C runtime, libdispatch and the Metal framework, as this program found them.
///
/// `loadRuntime` fills in as much as it can and leaves a field null when its library or symbol
/// is not on this machine, so that `--selftest` can report a partial runtime and the tests can
/// stop with a message that says what is missing, instead of crashing. Everything below is
/// optional; the helpers that unwrap a field are only reached once `missing` has returned null.
const Runtime = struct {
    objc_library: ?std.DynLib = null,
    system_library: ?std.DynLib = null,
    metal_library: ?std.DynLib = null,

    /// `objc_getClass`.
    get_class: ?GetClass = null,
    /// `sel_registerName`.
    register_selector: ?RegisterSelector = null,
    /// `objc_msgSend`, the address that `call` casts.
    msg_send: ?*const anyopaque = null,
    /// `objc_autoreleasePoolPush`: the messages that return an object without transferring
    /// ownership of it are autoreleased, and a thread without a pool leaks them and complains.
    /// The two pool functions are not required: a runtime without them costs a warning.
    pool_push: ?PoolPush = null,
    /// `objc_autoreleasePoolPop`.
    pool_pop: ?PoolPop = null,
    /// `dispatch_data_create`.
    dispatch_data_create: ?DispatchDataCreate = null,
    /// `MTLCreateSystemDefaultDevice`.
    create_system_default_device: ?CreateSystemDefaultDevice = null,
    /// The `NSString` class, which `newFunctionWithName:` takes and which arrives with
    /// Foundation, a framework that the Metal framework loads.
    string_class: ?*anyopaque = null,

    /// A selector for a method name, such as `"setBuffer:offset:atIndex:"`. The runtime keeps
    /// the selector, so registering the same name again is free.
    fn selector(rt: *const Runtime, name: [*:0]const u8) ?*anyopaque {
        return rt.register_selector.?(name);
    }

    /// Sends one message; see `call`.
    fn send(
        rt: *const Runtime,
        comptime MsgSend: type,
        receiver: ?*anyopaque,
        selector_arg: ?*anyopaque,
        args: anytype,
    ) ReturnType(MsgSend) {
        return call(rt.msg_send.?, MsgSend, receiver, selector_arg, args);
    }

    /// An `NSString` of a UTF-8 C string, with `[NSString stringWithUTF8String:]`. The string is
    /// autoreleased: `run` and `selfTest` cover their work with an autorelease pool.
    fn string(rt: *const Runtime, text: [*:0]const u8) ?*anyopaque {
        return rt.send(MsgSendObjectWithCString, rt.string_class.?, rt.selector("stringWithUTF8String:"), .{text});
    }

    /// The UTF-8 bytes of an `NSString`, or a phrase for a string that is not there.
    fn stringValue(rt: *const Runtime, object: ?*anyopaque) []const u8 {
        const string_object = object orelse return "(no string)";
        const utf8 = rt.send(MsgSendCString, string_object, rt.selector("UTF8String"), .{}) orelse
            return "(the string has no UTF-8 bytes)";
        return std.mem.span(utf8);
    }

    /// The message of an `NSError`, for the failure lines: what the runtime says went wrong.
    fn errorMessage(rt: *const Runtime, error_object: ?*anyopaque) []const u8 {
        const object = error_object orelse return "(the runtime returned no error object)";
        const description = rt.send(MsgSendObject, object, rt.selector("localizedDescription"), .{}) orelse
            return "(the error object has no description)";
        return rt.stringValue(description);
    }

    /// The `name` of a device, for the lines that say which GPU this is.
    fn deviceName(rt: *const Runtime, device: ?*anyopaque) []const u8 {
        return rt.stringValue(rt.send(MsgSendObject, device, rt.selector("name"), .{}));
    }

    /// The `registryID` of a device, the value that the system gives each GPU.
    fn registryID(rt: *const Runtime, device: ?*anyopaque) NSUInteger {
        return rt.send(MsgSendNSUInteger, device, rt.selector("registryID"), .{});
    }

    /// The first thing that could not be loaded, as a phrase for a line that names a test, or
    /// null when the runtime is complete.
    fn missing(rt: Runtime) ?[]const u8 {
        if (rt.objc_library == null) return "the Objective-C runtime, " ++ objc_library_path ++ ", is not available";
        if (rt.system_library == null) return "libdispatch, " ++ system_library_path ++ ", is not available";
        if (rt.metal_library == null) return "the Metal framework, " ++ metal_library_path ++ ", is not available";
        if (rt.get_class == null or rt.register_selector == null or rt.msg_send == null)
            return "the Objective-C runtime has no objc_getClass, sel_registerName or objc_msgSend";
        if (rt.dispatch_data_create == null) return "libdispatch has no dispatch_data_create";
        if (rt.create_system_default_device == null) return "the Metal framework has no MTLCreateSystemDefaultDevice";
        if (rt.string_class == null) return "the Objective-C runtime has no NSString class";
        return null;
    }

    /// Unloads the libraries. Nothing of the runtime may be used afterwards, which is why both
    /// `run` and `selfTest` do this last, when their report is written.
    fn close(rt: *Runtime) void {
        if (rt.metal_library) |*library| library.close();
        if (rt.system_library) |*library| library.close();
        if (rt.objc_library) |*library| library.close();
        rt.* = undefined;
    }
};

/// Loads the libraries and resolves every symbol that this program sends a message with.
///
/// Nothing here fails: a library or a symbol that is not there leaves its field null, which
/// `Runtime.missing` reports. The Metal framework is loaded before `NSString` is looked up,
/// because the class arrives with Foundation, one of the frameworks that Metal loads.
fn loadRuntime() Runtime {
    var rt: Runtime = .{};

    rt.objc_library = std.DynLib.open(objc_library_path) catch null;
    if (rt.objc_library) |*library| {
        rt.get_class = library.lookup(GetClass, "objc_getClass");
        rt.register_selector = library.lookup(RegisterSelector, "sel_registerName");
        rt.msg_send = library.lookup(*const anyopaque, "objc_msgSend");
        rt.pool_push = library.lookup(PoolPush, "objc_autoreleasePoolPush");
        rt.pool_pop = library.lookup(PoolPop, "objc_autoreleasePoolPop");
    }

    rt.system_library = std.DynLib.open(system_library_path) catch null;
    if (rt.system_library) |*library| {
        rt.dispatch_data_create = library.lookup(DispatchDataCreate, "dispatch_data_create");
    }

    rt.metal_library = std.DynLib.open(metal_library_path) catch null;
    if (rt.metal_library) |*library| {
        rt.create_system_default_device = library.lookup(CreateSystemDefaultDevice, "MTLCreateSystemDefaultDevice");
    }

    if (rt.get_class) |get_class| rt.string_class = get_class("NSString");

    return rt;
}

// ------------------------------------------------------------ the GPU objects

/// One `MTLBuffer` of shared memory: the CPU and the GPU see the same bytes.
const Buffer = struct {
    /// The `id<MTLBuffer>`.
    object: ?*anyopaque,
    /// The buffer's `contents`, as the CPU sees them: what `newBufferWithBytes:` copied in, and
    /// what the kernel reads and writes.
    contents: [*]u8,

    /// The values of type `T` that the buffer holds, `len` of them. The memory of a buffer is
    /// aligned far beyond any scalar, but `contents` is a byte pointer and nothing in the API
    /// promises more, so this keeps the alignment at 1: an unaligned load of an `f32` or a `u32`
    /// is one instruction, and a buffer can never trip the alignment check of a safe build.
    fn values(buffer: Buffer, comptime T: type, len: usize) []align(1) const T {
        const pointer: [*]align(1) const T = @ptrCast(buffer.contents);
        return pointer[0..len];
    }
};

/// The runtime and the GPU objects that the tests run their kernels with.
const Session = struct {
    /// The loaded runtime, which the caller keeps alive for the length of the run.
    rt: *const Runtime,
    device: ?*anyopaque,
    queue: ?*anyopaque,
    /// The `id<MTLLibrary>` of the `.metallib`.
    library: ?*anyopaque,

    /// The function object of a kernel, or null when the library does not have it: a pipeline
    /// that is still being built can be missing kernels, which the tests skip.
    fn function(session: *const Session, name: [*:0]const u8) ?*anyopaque {
        return session.rt.send(MsgSendObjectWithObject, session.library, session.rt.selector("newFunctionWithName:"), .{
            session.rt.string(name),
        });
    }

    /// The compute pipeline state of a function object, which is what a dispatch encodes. A null
    /// return, after a failure line, means the runtime refused to build it; the line carries the
    /// message of the `NSError` that the message wrote, which is the only reason to know why.
    fn newPipeline(session: *const Session, name: []const u8, function_object: ?*anyopaque, report: *Report) !?*anyopaque {
        var error_object: ?*anyopaque = null;
        const pipeline = session.rt.send(MsgSendObjectWithObjectError, session.device, session.rt.selector("newComputePipelineStateWithFunction:error:"), .{
            function_object,
            &error_object,
        }) orelse {
            try report.fail("{s}: newComputePipelineStateWithFunction:error: failed: {s}", .{ name, session.rt.errorMessage(error_object) });
            return null;
        };
        return pipeline;
    }

    /// A buffer of shared memory holding a copy of `bytes`, or null after a failure line that
    /// names the test and the size.
    fn buffer(session: *const Session, report: *Report, name: []const u8, bytes: []const u8) !?Buffer {
        const object = session.rt.send(MsgSendObjectWithBytes, session.device, session.rt.selector("newBufferWithBytes:length:options:"), .{
            @as(?*const anyopaque, @ptrCast(bytes.ptr)),
            @as(NSUInteger, bytes.len),
            resource_storage_mode_shared,
        }) orelse {
            try report.fail("{s}: cannot allocate a shared buffer of {d} bytes", .{ name, bytes.len });
            return null;
        };
        const contents = session.rt.send(MsgSendObject, object, session.rt.selector("contents"), .{}) orelse {
            try report.fail("{s}: a buffer of {d} bytes has no contents", .{ name, bytes.len });
            return null;
        };
        return .{ .object = object, .contents = @ptrCast(contents) };
    }

    /// Encodes and runs one dispatch and waits for the GPU: a command buffer, a compute command
    /// encoder, the pipeline state, one `setBuffer:` per kernel argument in argument order, and
    /// the grid. Returns false after a failure line when the runtime has no command buffer or
    /// encoder to give, which is the only step here that can fail on its own: a kernel that traps
    /// on the GPU leaves its results unwritten, which the checks of the test report as a
    /// mismatch.
    fn dispatch(
        session: *const Session,
        name: []const u8,
        report: *Report,
        pipeline: ?*anyopaque,
        buffers: []const ?*anyopaque,
        threads: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) !bool {
        const command_buffer = session.rt.send(MsgSendObject, session.queue, session.rt.selector("commandBuffer"), .{}) orelse {
            try report.fail("{s}: the command queue has no command buffer", .{name});
            return false;
        };
        const encoder = session.rt.send(MsgSendObject, command_buffer, session.rt.selector("computeCommandEncoder"), .{}) orelse {
            try report.fail("{s}: the command buffer has no compute command encoder", .{name});
            return false;
        };

        session.rt.send(MsgSendVoidWithObject, encoder, session.rt.selector("setComputePipelineState:"), .{pipeline});
        for (buffers, 0..) |buffer_object, index| {
            session.rt.send(MsgSendVoidSetBuffer, encoder, session.rt.selector("setBuffer:offset:atIndex:"), .{
                buffer_object,
                @as(NSUInteger, 0),
                @as(NSUInteger, @intCast(index)),
            });
        }
        // The builtins of the kernels -- the thread position in the grid -- are not buffers:
        // the dispatch is what supplies them.
        session.rt.send(MsgSendVoidDispatchThreads, encoder, session.rt.selector("dispatchThreads:threadsPerThreadgroup:"), .{
            &threads,
            &threads_per_threadgroup,
        });
        session.rt.send(MsgSendVoid, encoder, session.rt.selector("endEncoding"), .{});
        session.rt.send(MsgSendVoid, command_buffer, session.rt.selector("commit"), .{});
        session.rt.send(MsgSendVoid, command_buffer, session.rt.selector("waitUntilCompleted"), .{});
        return true;
    }
};

// ----------------------------------------------------------------- the report

/// The output of one run: one line for each test and the counts of the summary.
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

    /// A test that ran and disagreed, or a failure of the harness itself: a `.metallib` that
    /// does not load, a pipeline or a buffer that the runtime refused. The line names the test,
    /// and the message of the runtime's `NSError` where there was one.
    fn fail(r: *Report, comptime format: []const u8, args: anytype) !void {
        r.failures += 1;
        try r.out.print("--- FAIL: " ++ format ++ "\n", args);
    }

    /// A test that did not run because the `.metallib` does not have its kernel yet. A partial
    /// pipeline can be tested because this is not a failure.
    fn skip(r: *Report, comptime format: []const u8, args: anytype) !void {
        r.skips += 1;
        try r.out.print("--- SKIP: " ++ format ++ "\n", args);
    }

    /// The line for a kernel that the `.metallib` does not have.
    fn skipMissing(r: *Report, name: []const u8) !void {
        try r.skip("{s}: the metallib has no function of that name", .{name});
    }

    /// The last line of the run: `PASS` or `FAIL`, then the counts, so that the pipeline that
    /// runs this program can grep for one word and still see a skip.
    fn summary(r: *Report) !void {
        try r.out.print(prefix ++ "{s}: {d} passed, {d} failed, {d} skipped\n", .{
            if (r.failures == 0) "PASS" else "FAIL",
            r.passes,
            r.failures,
            r.skips,
        });
    }
};

/// Whether two `f32` have the same bits. Every check of this program compares bits: a result
/// that is one unit in the last place off is off, and a NaN of the wrong sign is off too.
fn sameBits(x: f32, y: f32) bool {
    return floatBits(x) == floatBits(y);
}

/// The bits of an `f32`, for the failure lines.
fn floatBits(value: f32) u32 {
    return @bitCast(value);
}

// ----------------------------------------------------------------- the tests

/// `vadd`: one thread per element of three 4096-element buffers of `f32`.
///
/// The inputs are small integers, so every input and every sum is exactly representable as an
/// `f32` and the comparison is bit for bit: a kernel that rounded, fused or reordered anything
/// would show up as a mismatch.
fn testVadd(session: *const Session, report: *Report) !void {
    const n = 4096;
    const threads_per_threadgroup: NSUInteger = 64;
    const threadgroups = (n + threads_per_threadgroup - 1) / threads_per_threadgroup;

    const function = session.function("vadd") orelse return report.skipMissing("vadd");
    const pipeline = (try session.newPipeline("vadd", function, report)) orelse return;

    var a: [n]f32 = undefined;
    var b: [n]f32 = undefined;
    var c: [n]f32 = @splat(0);
    for (&a, &b, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i % 13);
        y.* = @floatFromInt(i % 7);
    }

    // The buffers are the kernel arguments in order: a, b and c are buffer 0, 1 and 2.
    const buffer_a = (try session.buffer(report, "vadd", std.mem.sliceAsBytes(&a))) orelse return;
    const buffer_b = (try session.buffer(report, "vadd", std.mem.sliceAsBytes(&b))) orelse return;
    const buffer_c = (try session.buffer(report, "vadd", std.mem.sliceAsBytes(&c))) orelse return;

    const dispatched = try session.dispatch("vadd", report, pipeline, &.{
        buffer_a.object,
        buffer_b.object,
        buffer_c.object,
    }, .linear(n), .linear(threads_per_threadgroup));
    if (!dispatched) return;

    const results = buffer_c.values(f32, n);
    var mismatches: usize = 0;
    var first_index: usize = 0;
    var first_expected: f32 = 0;
    var first_actual: f32 = 0;
    for (results, a, b, 0..) |actual, x, y, i| {
        const expected = x + y;
        if (!sameBits(actual, expected)) {
            if (mismatches == 0) {
                first_index = i;
                first_expected = expected;
                first_actual = actual;
            }
            mismatches += 1;
        }
    }
    if (mismatches != 0) return report.fail(
        "vadd: element {d}: expected {d} (0x{x:0>8}), got {d} (0x{x:0>8}); {d} of {d} elements differ",
        .{ first_index, first_expected, floatBits(first_expected), first_actual, floatBits(first_actual), mismatches, n },
    );
    try report.pass("vadd: {d} f32, {d} threadgroups of {d} threads, c[i] == a[i] + b[i]", .{
        n, threadgroups, threads_per_threadgroup,
    });
}

/// `reduce`: four threadgroups of 256 threads over 1024 `f32`, with the partial sums of the
/// group in threadgroup memory and a device-scope atomic counter.
///
/// `air.simd_sum.f32` reduces 32 lanes on an Apple GPU, so the threadgroup memory of the kernel
/// holds eight partial sums. The inputs are the small integers 0 to 15, which makes every
/// partial sum and every group total exact in `f32` in any order: the reference is the plain sum
/// of the 256 values of the group, and the comparison is bit for bit.
fn testReduce(session: *const Session, report: *Report) !void {
    const group_size: NSUInteger = 256;
    const group_count = 4;
    const n = group_size * group_count;
    const expected_total: f32 = 1920;

    const function = session.function("reduce") orelse return report.skipMissing("reduce");
    const pipeline = (try session.newPipeline("reduce", function, report)) orelse return;

    var input: [n]f32 = undefined;
    for (&input, 0..) |*value, i| value.* = @floatFromInt(i % 16);

    var expected: [group_count]f32 = undefined;
    for (&expected, 0..) |*total, group| {
        total.* = 0;
        for (input[group * group_size ..][0..group_size]) |value| total.* += value;
    }

    // The counter starts at zero and every threadgroup adds one to it, so the four groups of the
    // dispatch must leave four. The dispatch is the same one that writes the sums, so an empty
    // counter means the kernel did not run at all.
    var out: [group_count]f32 = @splat(0);
    var counters: [1]u32 = @splat(0);

    // The kernel arguments in order: inbuf, out and counter are buffer 0, 1 and 2.
    const input_buffer = (try session.buffer(report, "reduce", std.mem.sliceAsBytes(&input))) orelse return;
    const out_buffer = (try session.buffer(report, "reduce", std.mem.sliceAsBytes(&out))) orelse return;
    const counter_buffer = (try session.buffer(report, "reduce", std.mem.sliceAsBytes(&counters))) orelse return;

    const dispatched = try session.dispatch("reduce", report, pipeline, &.{
        input_buffer.object,
        out_buffer.object,
        counter_buffer.object,
    }, .linear(n), .linear(group_size));
    if (!dispatched) return;

    const results = out_buffer.values(f32, group_count);
    const counter = counter_buffer.values(u32, 1)[0];

    var mismatches: usize = 0;
    var first_group: usize = 0;
    for (results, expected, 0..) |actual, want, group| {
        if (!sameBits(actual, want)) {
            if (mismatches == 0) first_group = group;
            mismatches += 1;
        }
    }
    if (mismatches != 0) return report.fail(
        "reduce: out[{d}]: expected {d} (0x{x:0>8}), got {d} (0x{x:0>8}); {d} of {d} group sums differ; counter {d}, expected {d}",
        .{
            first_group, expected[first_group], floatBits(expected[first_group]), results[first_group], floatBits(results[first_group]),
            mismatches,  group_count,           counter,                          group_count,
        },
    );
    if (counter != group_count) return report.fail(
        "reduce: the counter is {d}, expected {d}: every threadgroup adds one; out[0] = {d} (all group sums are {d})",
        .{ counter, group_count, results[0], expected_total },
    );
    try report.pass("reduce: {d} threadgroups of {d} threads over {d} f32, every group sum {d} and counter {d}", .{
        group_count, group_size, n, expected_total, counter,
    });
}

/// The numbers of the `parsef` test: a sign, a fraction, an exponent and a negative exponent.
const parse_numbers = [_][]const u8{ "3.25", "-0.5", "1e3", "0.125", "-12.5", "2e-3", "7", "1.5e2" };

/// The value of every number above, parsed on the CPU by the same standard-library function that
/// the kernel uses, so the two must agree bit for bit. The numbers are literals that parse, so
/// this is a compile-time constant: a number that did not parse, or did not fit a slot, would
/// stop the build.
const parse_expected: [parse_numbers.len]f32 = values: {
    var values: [parse_numbers.len]f32 = undefined;
    for (parse_numbers, 0..) |number, i| {
        if (number.len > 32) @compileError("a number of the parsef test does not fit in its 32-byte slot");
        values[i] = std.fmt.parseFloat(f32, number) catch unreachable;
    }
    break :values values;
};

/// `parsef`: eight threads, each parsing one decimal number out of a 32-byte slot of a text
/// buffer, with the length of the slot's text in a second buffer.
///
/// The kernel parses with `std.fmt.parseFloat`, in device code, from the same standard library
/// that `parse_expected` uses, so the results are compared bit for bit; the closest value that a
/// parser which is not correctly rounded produces for `"2e-3"` is one unit in the last place
/// off, which this comparison reports.
fn testParse(session: *const Session, report: *Report) !void {
    const slot_size = 32;
    const threads = parse_numbers.len;

    const function = session.function("parsef") orelse return report.skipMissing("parsef");
    const pipeline = (try session.newPipeline("parsef", function, report)) orelse return;

    var text: [threads * slot_size]u8 = @splat(0);
    var lengths: [threads]u32 = undefined;
    for (parse_numbers, 0..) |number, i| {
        @memcpy(text[i * slot_size ..][0..number.len], number);
        lengths[i] = @intCast(number.len);
    }

    var out: [threads]f32 = @splat(0);

    // The kernel arguments in order: text, lengths and out are buffer 0, 1 and 2. Every number
    // is shorter than its 32-byte slot, so the kernel parses the digits and stops at the length.
    const text_buffer = (try session.buffer(report, "parsef", &text)) orelse return;
    const lengths_buffer = (try session.buffer(report, "parsef", std.mem.sliceAsBytes(&lengths))) orelse return;
    const out_buffer = (try session.buffer(report, "parsef", std.mem.sliceAsBytes(&out))) orelse return;

    const dispatched = try session.dispatch("parsef", report, pipeline, &.{
        text_buffer.object,
        lengths_buffer.object,
        out_buffer.object,
    }, .linear(threads), .linear(threads));
    if (!dispatched) return;

    const results = out_buffer.values(f32, threads);
    var mismatches: usize = 0;
    var first_index: usize = 0;
    for (results, parse_expected, 0..) |actual, expected, i| {
        if (!sameBits(actual, expected)) {
            if (mismatches == 0) first_index = i;
            mismatches += 1;
        }
    }
    if (mismatches != 0) {
        const index = first_index;
        return report.fail(
            "parsef: \"{s}\": expected {d} (0x{x:0>8}), got {d} (0x{x:0>8}); {d} of {d} numbers differ",
            .{
                parse_numbers[index], parse_expected[index],     floatBits(parse_expected[index]),
                results[index],       floatBits(results[index]), mismatches,
                threads,
            },
        );
    }
    try report.pass("parsef: {d} decimal numbers ({s}, ...), parsed the same way as std.fmt.parseFloat", .{
        threads, parse_numbers[0],
    });
}

// --------------------------------------------------------------------- selftest

/// The GPU families that `--selftest` asks the device about. `supportsFamily:` takes the raw
/// numbers of `MTLGPUFamily`: 1007 is Apple7, and the numbers of the ticket for this program
/// (2005 for Metal 3, 3001 and 3002 for Metal 4) are not the ones in the Metal headers, where
/// Metal 3 is 5001 and Metal 4 is 5002 and 3001 and 3002 are Common1 and Common2. Both sets are
/// asked for, and every answer is printed with its number, so the report says what the machine
/// thinks of each of them.
const gpu_families = [_]struct { number: i64, name: []const u8 }{
    .{ .number = 1007, .name = "Apple7" },
    .{ .number = 1008, .name = "Apple8" },
    .{ .number = 1009, .name = "Apple9" },
    .{ .number = 1010, .name = "Apple10" },
    .{ .number = 2005, .name = "not in the headers" },
    .{ .number = 3001, .name = "Common1" },
    .{ .number = 3002, .name = "Common2" },
    .{ .number = 5001, .name = "Metal3" },
    .{ .number = 5002, .name = "Metal4" },
};

/// `host --selftest`: loads the libraries at run time and reports what it found, without a
/// `.metallib`.
///
/// This runs before the pipeline can produce a library, which is the point: it says whether the
/// three libraries, the symbols and the messages of this program all resolve, and what the
/// device says about its families. On a machine without the Metal runtime -- the Linux machine
/// that cross-compiles the program -- it reports what is missing and exits successfully, so the
/// binary can be smoke-tested anywhere; on a Mac, a missing piece is a failure to fix.
fn selfTest(out: *Io.Writer) !void {
    var runtime = loadRuntime();
    defer runtime.close();

    try out.print(prefix ++ "selftest on {s}\n", .{@tagName(builtin.os.tag)});

    const libraries = [_]struct { path: []const u8, loaded: bool }{
        .{ .path = objc_library_path, .loaded = runtime.objc_library != null },
        .{ .path = system_library_path, .loaded = runtime.system_library != null },
        .{ .path = metal_library_path, .loaded = runtime.metal_library != null },
    };
    for (libraries) |library| {
        try out.print(prefix ++ "library {s}: {s}\n", .{
            library.path, if (library.loaded) "loaded" else "unavailable",
        });
    }

    const symbols = [_]struct { name: []const u8, resolved: bool }{
        .{ .name = "objc_getClass", .resolved = runtime.get_class != null },
        .{ .name = "sel_registerName", .resolved = runtime.register_selector != null },
        .{ .name = "objc_msgSend", .resolved = runtime.msg_send != null },
        .{ .name = "objc_autoreleasePoolPush", .resolved = runtime.pool_push != null },
        .{ .name = "objc_autoreleasePoolPop", .resolved = runtime.pool_pop != null },
        .{ .name = "dispatch_data_create", .resolved = runtime.dispatch_data_create != null },
        .{ .name = "MTLCreateSystemDefaultDevice", .resolved = runtime.create_system_default_device != null },
        .{ .name = "NSString", .resolved = runtime.string_class != null },
    };
    for (symbols) |symbol| {
        try out.print(prefix ++ "symbol {s}: {s}\n", .{
            symbol.name, if (symbol.resolved) "resolved" else "not found",
        });
    }

    if (runtime.missing()) |reason| {
        try out.print(prefix ++ "selftest {s}: {s}\n", .{
            if (builtin.os.tag == .macos) "FAIL" else "PASS (no Metal on this machine)", reason,
        });
        if (builtin.os.tag == .macos) std.process.exit(1);
        return;
    }

    // The messages of the runtime that return an object without transferring ownership of it are
    // autoreleased: on a thread without a pool they leak, and the runtime warns about it.
    const pool = if (runtime.pool_push) |push| push() else null;
    defer if (runtime.pool_pop) |pop| pop(pool);

    // `MTLCreateSystemDefaultDevice` is the C function of the framework, and the device it
    // returns is what every message below is sent to.
    const device = runtime.create_system_default_device.?() orelse {
        try out.print(prefix ++ "selftest FAIL: MTLCreateSystemDefaultDevice returned nil: there is no GPU\n", .{});
        std.process.exit(1);
    };
    try out.print(prefix ++ "device: {s}, registryID 0x{x}\n", .{ runtime.deviceName(device), runtime.registryID(device) });

    // Each family is reported as it comes, so a number that this machine does not know is
    // visible as a `no` next to its number instead of as a failed call.
    for (gpu_families) |family| {
        const supported = runtime.send(MsgSendI8WithI64, device, runtime.selector("supportsFamily:"), .{family.number}) != 0;
        try out.print(prefix ++ "supportsFamily {d} ({s}): {s}\n", .{
            family.number, family.name, if (supported) "yes" else "no",
        });
    }

    try out.print(prefix ++ "selftest PASS\n", .{});
}

// ------------------------------------------------------------------- the run

/// The library object of a `.metallib`, built with `newLibraryWithData:error:`.
///
/// The bytes reach the runtime as a `dispatch_data_t`, which `dispatch_data_create` builds with
/// a null destructor: libdispatch copies the bytes then, so the file contents may go away as
/// soon as the message returns. The caller keeps them for the length of the run regardless;
/// what is copied is not something this program relies on.
fn loadLibrary(runtime: *const Runtime, device: ?*anyopaque, report: *Report, bytes: []const u8) !?*anyopaque {
    const data = runtime.dispatch_data_create.?(@ptrCast(bytes.ptr), bytes.len, null, null) orelse {
        try report.fail("dispatch_data_create returned no data for {d} bytes of metallib", .{bytes.len});
        return null;
    };
    var error_object: ?*anyopaque = null;
    const library = runtime.send(MsgSendObjectWithObjectError, device, runtime.selector("newLibraryWithData:error:"), .{
        data,
        &error_object,
    }) orelse {
        try report.fail("newLibraryWithData:error: failed for {d} bytes of metallib: {s}", .{
            bytes.len, runtime.errorMessage(error_object),
        });
        return null;
    };
    return library;
}

/// Runs the three tests against the kernels in `bytes`, the contents of the `.metallib` at
/// `path`, and prints the line of each test and the summary.
fn run(out: *Io.Writer, path: []const u8, bytes: []const u8) !void {
    var report: Report = .{ .out = out };

    var runtime = loadRuntime();
    defer runtime.close();

    if (runtime.missing()) |reason| {
        // Without the runtime there is nothing to run, and one reason covers all three tests. On
        // the Mac this is a machine to fix, so the run fails; anywhere else it is the expected
        // outcome, since the Metal runtime only exists on Darwin.
        for (kernel_names) |name| try report.skip("{s}: {s}", .{ name, reason });
        try report.summary();
        if (builtin.os.tag == .macos) std.process.exit(1);
        return;
    }

    // The messages that return an object without transferring ownership of it are autoreleased:
    // the `NSString` of a kernel name, the `NSError` of a failure, and the objects of the
    // framework's own bookkeeping. A thread without a pool leaks them and is warned about.
    const pool = if (runtime.pool_push) |push| push() else null;
    defer if (runtime.pool_pop) |pop| pop(pool);

    const device = runtime.create_system_default_device.?() orelse {
        try report.fail("MTLCreateSystemDefaultDevice returned nil: there is no GPU", .{});
        try report.summary();
        std.process.exit(1);
    };
    const queue = runtime.send(MsgSendObject, device, runtime.selector("newCommandQueue"), .{}) orelse {
        try report.fail("the device has no command queue", .{});
        try report.summary();
        std.process.exit(1);
    };
    const library = try loadLibrary(&runtime, device, &report, bytes) orelse {
        try report.summary();
        std.process.exit(1);
    };

    try out.print(prefix ++ "{s} on {s}, registryID 0x{x}\n", .{
        path, runtime.deviceName(device), runtime.registryID(device),
    });

    var session: Session = .{ .rt = &runtime, .device = device, .queue = queue, .library = library };
    try testVadd(&session, &report);
    try testReduce(&session, &report);
    try testParse(&session, &report);

    try report.summary();
    if (report.failures != 0) std.process.exit(1);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();

    var stdout_writer = Io.File.stdout().writerStreaming(init.io, &.{});
    const out = &stdout_writer.interface;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--selftest")) return selfTest(out);
    if (args.len != 2) {
        try out.print("usage: {s} <path-to.metallib>\n       {s} --selftest\n", .{ args[0], args[0] });
        std.process.exit(2);
    }

    const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, args[1], arena, .limited(max_metallib_size)) catch |err| {
        try out.print(prefix ++ "cannot read {s}: {s}\n", .{ args[1], @errorName(err) });
        std.process.exit(1);
    };

    try run(out, args[1], bytes);
}
