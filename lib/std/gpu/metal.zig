//! Host-side access to Apple GPUs through the Metal framework.
//!
//! The Metal framework, the Objective-C runtime, and libdispatch are loaded with `std.DynLib`
//! when they are needed, so a program that uses this namespace needs neither the macOS SDK nor an
//! Objective-C compiler to build, and no headers to import. Loading a library uses `dlopen`,
//! which is part of libc, so link the program with libc. The libraries are where macOS keeps
//! them, which is the system that this namespace covers: on any other operating system
//! `Driver.open` reports `error.FrameworkNotFound`, and on macOS without libc there is no `dlopen`
//! to load them with, which is a compile error.
//!
//! Where `std.gpu.cuda` loads PTX and `std.gpu.hip` loads a code object, this namespace loads a
//! `.metallib`: the container of the AIR of a library of kernels, which the Metal compiler turns
//! into the code of the device when a pipeline state is created. Zig++ compiles the kernels of a
//! program into that container, from the same source that it compiles for NVIDIA and AMD GPUs.
//!
//! The objects of this namespace line up with the ones of the two others like this:
//!
//! * a `Driver` is the loaded framework, as `cuda.Driver` is the loaded driver library, and
//!   `Driver.device` is the GPU of the machine, the one that macOS calls the system default;
//! * Metal has no primary context, so `Device.createContext` makes a `Context` of the device: the
//!   device and a command queue, which is what runs kernels and what `Context.synchronize` waits
//!   for;
//! * a `Module` is a loaded `.metallib`, a `Function` is a kernel of that module found by the name
//!   it was exported with, and a `Pipeline` is the compiled code of that kernel, which is what a
//!   launch dispatches;
//! * a `Buffer` is a region of memory that the CPU and the GPU share;
//! * the kernels of a launch run on the command queue of the context that loaded their module.
//!
//! Buffer memory is in unified memory (`MTLResourceStorageModeShared`): `Buffer.values` is the
//! memory as a slice of `T`, the same bytes that the kernels read and write, and
//! `Buffer.copyFromHost` and `Buffer.copyToHost` copy between that memory and a host slice, where
//! `std.gpu.cuda` and `std.gpu.hip` call into their driver instead. Launches are asynchronous, as
//! they are there: `Context.synchronize` waits until the kernels that were launched have finished,
//! and reports the error of a kernel that failed on the GPU, which no call before it can see.
//!
//! The lifetime rules are those of the two other namespaces: the driver must stay open, and the
//! context that loaded a module must stay alive while the modules, functions, pipeline states, and
//! buffers made from it are in use (a copy of a context is that same context, so a program may
//! keep one wherever it likes).
//!
//! ```zig
//! const std = @import("std");
//! const metal = std.gpu.metal;
//!
//! // The `.metallib` of a library of kernels, such as the output of
//! // `zig build-obj -target air64-macos -femit-bin=kernels.metallib`.
//! const library_bytes: []const u8 = @embedFile("kernels.metallib");
//!
//! pub fn main() !void {
//!     var driver = try metal.Driver.open();
//!     defer driver.close();
//!
//!     const context = try driver.device().createContext();
//!     defer context.release();
//!
//!     const module = try context.loadModule(library_bytes, .{});
//!     defer module.unload();
//!     const function = try module.function("add_one");
//!     defer function.release();
//!     const pipeline = try function.pipeline(.{});
//!     defer pipeline.release();
//!
//!     var data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//!     const buffer = try context.alloc(u32, data.len);
//!     defer buffer.free();
//!     buffer.copyFromHost(&data);
//!     try pipeline.launch(metal.LaunchConfig.linear(data.len, 4), .{ buffer, @as(u32, data.len) });
//!     try context.synchronize();
//!     buffer.copyToHost(&data);
//! }
//! ```

const std = @import("../std.zig");
const builtin = @import("builtin");
const launch_arguments = @import("launch_arguments.zig");
const testing = std.testing;

/// Whether the Metal framework can be loaded here: it comes with macOS, and `std.DynLib` loads a
/// library with `dlopen`, which is part of libc.
const driver_supported = builtin.os.tag == .macos and builtin.link_libc;

/// The errors that `Driver.open` reports, in addition to the errors of the framework.
pub const OpenError = error{
    /// The Metal framework, the Objective-C runtime, or libdispatch is not where this namespace
    /// looks for it, which is the case on every operating system other than macOS, or there is no
    /// GPU for the framework to run kernels on.
    FrameworkNotFound,
    /// The framework is loaded, but one of the libraries that this namespace calls into does not
    /// have a function that it uses.
    IncompatibleFramework,
} || Error;

/// The errors that the Metal framework reports, one name per failure that a caller can act on
/// differently.
///
/// The framework reports a failure as an `NSError`, in one of its domains and with a code of that
/// domain, and `errorFor` maps the ones worth telling apart. The names say what happened rather
/// than repeat the code: `error.InvalidLibrary` and `error.CompileFailure` come from
/// `Context.loadModule` and `Function.pipeline`, `error.FunctionNotFound` from `Module.function`,
/// and the rest from the command buffers that `Context.synchronize` waits for. A failure of a
/// domain or a code that this version of the standard library does not know is reported as the
/// failure of the call, so that a caller always has an error to act on.
pub const Error = error{
    /// `MTLCreateSystemDefaultDevice` found no GPU to run kernels on.
    NoDevice,
    /// This is an Intel Mac. The framework is there, but the messages that this namespace sends
    /// are the programs of the arm64 ABI, where a `MTLSize` of a message passes and returns by
    /// other rules than it does on `x86_64`: the wrong program would corrupt the call, so the
    /// namespace reports this instead of running kernels with it.
    NotSupported,
    /// The caller asked for something that the namespace cannot do: a name that does not fit the
    /// buffer that was given for it, an empty buffer, a grid of no threadgroups, or a threadgroup
    /// that is larger than the kernel of the launch allows.
    InvalidValue,
    /// The device cannot read the bytes that were given as a library: they are not a `.metallib`,
    /// they are a `.metallib` of a version that this macOS does not know, or the image of the
    /// library is not there.
    InvalidLibrary,
    /// The AIR of a module could not be compiled into the code of the device, which is what
    /// `Function.pipeline` does. `Options.error_log` receives what the compiler said about it.
    CompileFailure,
    /// The module has no function of the name that was asked for.
    FunctionNotFound,
    /// A command buffer stopped with an error that does not name a more specific one. The error of
    /// `Context.synchronize` for a GPU that failed in a way that this version does not know.
    CommandBufferError,
    /// The work of a command buffer took too long, and the GPU aborted it.
    Timeout,
    /// A kernel made an access to memory that the GPU cannot serve, which usually means a pointer
    /// past the end of a buffer. The kernel leaves its results unwritten, and the buffers that it
    /// wrote before the fault hold whatever it wrote.
    PageFault,
    /// Access to the GPU was revoked, because this program caused too many timeouts or hangs.
    AccessRevoked,
    /// This process may not use the GPU.
    NotPermitted,
    /// There was not enough memory for the work of a command buffer, or for an object that the
    /// framework was asked to create.
    OutOfMemory,
    /// A command buffer referred to a resource that was released before the command buffer ran,
    /// such as a buffer that `Buffer.free` freed while a launch that used it was in flight.
    InvalidResource,
    /// A limit on the internal resources of a pass was reached.
    Memoryless,
    /// The GPU was removed (an external GPU that was unplugged) before the work finished.
    DeviceRemoved,
    /// A kernel overflowed the stack that the GPU gives a thread.
    StackOverflow,
    /// A failure inside the framework that does not fit the other names.
    InternalError,
    /// A failure of this namespace or of the framework that the other names do not cover, such as
    /// a buffer of shared memory whose `contents` the framework did not give. The standard library
    /// reports it rather than leave a null pointer for a kernel to write through.
    Unexpected,
};

/// The options of the calls that can report what the framework said about a failure:
/// `Context.loadModule` and `Function.pipeline`.
pub const Options = struct {
    /// When the call fails, receives a line about the failure, with a null byte after it: the
    /// domain of the `NSError` that the framework reported, its code, and the message that the
    /// framework wrote for it. A line that does not fit is cut at the end of the buffer; a few
    /// hundred bytes hold any message that macOS writes, and a message that the framework did not
    /// write is a line that says so.
    error_log: ?[]u8 = null,
};

/// An Objective-C object pointer, an `id`. Nothing here can check what a message takes or returns,
/// so the fields and the parameters that are one say which object it is.
const Id = ?*anyopaque;

/// A selector: the name that the Objective-C runtime knows a method by.
const Sel = ?*anyopaque;

/// `NSUInteger` on 64-bit macOS.
const NSUInteger = u64;

/// `NSInteger` on 64-bit macOS.
const NSInteger = i64;

/// `BOOL`: a signed char, and not a Zig `bool`. macOS is one of the systems where the Objective-C
/// runtime defines it that way (the Apple systems that define it as `bool` are the ones that this
/// namespace does not cover), and it is a byte in a register that the rest of the register does not
/// extend, so the value is read as one.
const BOOL = i8;

/// `MTLSize`: the size of a grid or of a threadgroup in each of its three dimensions.
const MTLSize = extern struct {
    width: NSUInteger,
    height: NSUInteger,
    depth: NSUInteger,
};

/// `MTLResourceStorageModeShared`: the CPU and the GPU see the same memory. Every buffer of this
/// namespace is in this mode, which is what makes `Buffer.values` the bytes of the buffer.
const resource_storage_mode_shared: NSUInteger = 0;

/// `MTLCommandBufferStatusError`: the command buffer stopped with an error, which its `error`
/// property says more about. A command buffer that ran to the end is
/// `MTLCommandBufferStatusCompleted`, 4.
const command_buffer_status_error: NSUInteger = 5;

/// The libraries that this namespace loads: the Objective-C runtime, libdispatch, and the Metal
/// framework.
const objc_library_name: [:0]const u8 = "/usr/lib/libobjc.A.dylib";
const system_library_name: [:0]const u8 = "/usr/lib/libSystem.B.dylib";
const metal_library_name: [:0]const u8 = "/System/Library/Frameworks/Metal.framework/Metal";

/// Which of those libraries exports a function that this namespace calls.
const WhichLibrary = enum { objc, system, metal };

/// The loaded libraries, loaded and released together by a `Driver`.
const Libraries = struct {
    /// The Objective-C runtime, for `objc_getClass`, `sel_registerName`, `objc_msgSend`, and the
    /// autorelease pool.
    objc: std.DynLib,
    /// libdispatch, for `dispatch_data_create`: the bytes of a `.metallib` reach the framework as
    /// a `dispatch_data_t`.
    system: std.DynLib,
    /// The Metal framework, for `MTLCreateSystemDefaultDevice`.
    metal: std.DynLib,

    /// The address of a function of one of the libraries, or null when that library is not on this
    /// machine or does not export the name.
    fn lookup(libraries: *Libraries, comptime T: type, which: WhichLibrary, symbol: [:0]const u8) ?T {
        return switch (which) {
            .objc => libraries.objc.lookup(T, symbol),
            .system => libraries.system.lookup(T, symbol),
            .metal => libraries.metal.lookup(T, symbol),
        };
    }

    /// Opens the three libraries, or returns null when one of them is not on this machine. The
    /// ones that were opened are closed again.
    fn open() ?Libraries {
        var libraries: Libraries = undefined;
        libraries.objc = std.DynLib.open(objc_library_name) catch return null;
        libraries.system = std.DynLib.open(system_library_name) catch {
            libraries.objc.close();
            return null;
        };
        libraries.metal = std.DynLib.open(metal_library_name) catch {
            libraries.system.close();
            libraries.objc.close();
            return null;
        };
        return libraries;
    }

    /// Unloads the libraries. Nothing that came from them may be used afterwards.
    fn close(libraries: *Libraries) void {
        libraries.metal.close();
        libraries.system.close();
        libraries.objc.close();
    }
};

/// Pointers to the C functions that this namespace calls. The messages of the framework's objects
/// do not go through these: they are sent with `objc_msgSend`; see `Driver.send`.
const Functions = struct {
    /// `id objc_getClass(const char *name)`: the class object of a class, such as `NSString`.
    get_class: *const fn (name: [*:0]const u8) callconv(.c) Id,
    /// `SEL sel_registerName(const char *name)`: the selector of a method of that name, registered
    /// with the runtime the first time it is asked for.
    register_selector: *const fn (name: [*:0]const u8) callconv(.c) Sel,
    /// The address of `objc_msgSend`, which `send` calls through a function pointer whose type is
    /// the signature of the method. On arm64 there is no variadic `objc_msgSend` to fall back on:
    /// the arguments and the return value are in registers, and a call through a pointer of the
    /// wrong type corrupts them silently, which is why every message this namespace sends has a
    /// type of its own below.
    msg_send: *const anyopaque,
    /// `void *objc_autoreleasePoolPush(void)` and `void objc_autoreleasePoolPop(void *pool)`.
    /// Every message that returns an object without transferring ownership of it is autoreleased;
    /// a thread that sends one without a pool leaks the object and makes the runtime warn, so the
    /// calls of this namespace that can create such an object cover their work with a pool.
    pool_push: *const fn () callconv(.c) Id,
    pool_pop: *const fn (pool: Id) callconv(.c) void,
    /// `dispatch_data_t dispatch_data_create(const void *buffer, size_t size, dispatch_queue_t
    /// queue, dispatch_block_t destructor)`. With a null queue and a null destructor, libdispatch
    /// copies the buffer, and the copy is what the framework reads: the image of a module can be
    /// freed once the module is loaded.
    dispatch_data_create: *const fn (buffer: ?*const anyopaque, size: usize, queue: Id, destructor: Id) callconv(.c) Id,
    /// `id MTLCreateSystemDefaultDevice(void)`: the GPU of the machine, which is the discrete GPU
    /// on a Mac that has one and the integrated GPU otherwise. Unlike the other functions here,
    /// the device it returns is retained, so `Driver.close` releases it.
    create_system_default_device: *const fn () callconv(.c) Id,
};

/// The name of every function of `Functions`, in the same order, and the library that exports it.
const function_symbols = .{
    .{ .field = "get_class", .library = .objc, .symbol = "objc_getClass" },
    .{ .field = "register_selector", .library = .objc, .symbol = "sel_registerName" },
    .{ .field = "msg_send", .library = .objc, .symbol = "objc_msgSend" },
    .{ .field = "pool_push", .library = .objc, .symbol = "objc_autoreleasePoolPush" },
    .{ .field = "pool_pop", .library = .objc, .symbol = "objc_autoreleasePoolPop" },
    .{ .field = "dispatch_data_create", .library = .system, .symbol = "dispatch_data_create" },
    .{ .field = "create_system_default_device", .library = .metal, .symbol = "MTLCreateSystemDefaultDevice" },
};

comptime {
    const field_names = @typeInfo(Functions).@"struct".field_names;
    if (field_names.len != function_symbols.len) {
        @compileError("Functions and function_symbols must have a name for every function");
    }
    for (field_names, function_symbols) |field_name, entry| {
        if (!std.mem.eql(u8, field_name, entry.field)) {
            @compileError("function_symbols must name the fields of Functions, in order");
        }
    }
}

/// The messages that this namespace sends, registered with the runtime once by `Driver.open`.
/// `Driver.selectors` holds the `SEL` of each member.
const Selector = enum {
    // `NSObject`, which every object of the framework is.
    release,
    retain,
    // `MTLDevice`.
    name,
    max_threads_per_threadgroup,
    supports_family,
    new_command_queue,
    new_buffer,
    new_library,
    new_function,
    new_pipeline,
    // `MTLBuffer`.
    contents,
    // `MTLCommandQueue`.
    command_buffer,
    // `MTLCommandBuffer`.
    compute_command_encoder,
    commit,
    wait_until_completed,
    status,
    @"error",
    // `MTLComputeCommandEncoder`.
    set_pipeline,
    set_buffer,
    set_bytes,
    dispatch_threadgroups,
    end_encoding,
    // `MTLComputePipelineState`.
    max_total_threads_per_threadgroup,
    thread_execution_width,
    // `NSError`.
    domain,
    code,
    localized_description,
    // `NSString`.
    utf8_string,
    string_with_utf8_string,

    /// The method that the selector is the selector of, as the runtime spells it: the method's
    /// name, with a colon for every argument that a message with it takes after the selector.
    fn text(selector: Selector) [:0]const u8 {
        return switch (selector) {
            .release => "release",
            .retain => "retain",
            .name => "name",
            .max_threads_per_threadgroup => "maxThreadsPerThreadgroup",
            .supports_family => "supportsFamily:",
            .new_command_queue => "newCommandQueue",
            .new_buffer => "newBufferWithLength:options:",
            .new_library => "newLibraryWithData:error:",
            .new_function => "newFunctionWithName:",
            .new_pipeline => "newComputePipelineStateWithFunction:error:",
            .contents => "contents",
            .command_buffer => "commandBuffer",
            .compute_command_encoder => "computeCommandEncoder",
            .commit => "commit",
            .wait_until_completed => "waitUntilCompleted",
            .status => "status",
            .@"error" => "error",
            .set_pipeline => "setComputePipelineState:",
            .set_buffer => "setBuffer:offset:atIndex:",
            .set_bytes => "setBytes:length:atIndex:",
            .dispatch_threadgroups => "dispatchThreadgroups:threadsPerThreadgroup:",
            .end_encoding => "endEncoding",
            .max_total_threads_per_threadgroup => "maxTotalThreadsPerThreadgroup",
            .thread_execution_width => "threadExecutionWidth",
            .domain => "domain",
            .code => "code",
            .localized_description => "localizedDescription",
            .utf8_string => "UTF8String",
            .string_with_utf8_string => "stringWithUTF8String:",
        };
    }
};

// ---------------------------------------------------------------------- the messages
//
// `objc_msgSend` has no prototype: it takes the receiver, the selector, and then exactly the
// arguments of the method, and returns what the method returns. Each type below is the signature
// of one shape of message, and `call` casts the address of `objc_msgSend` to it. The rules of the
// ABI that these follow:
//
//   * `?*anyopaque` stands for an Objective-C object, a `SEL`, or a `void *`: nothing here can
//     check what a message takes or returns, which is what the SDK's headers would do;
//   * `NSUInteger` is `u64` and `NSInteger` is `i64`;
//   * `BOOL` is `i8`, not a Zig `bool`: on arm64 it is a signed char, and a message that returns
//     one leaves the rest of the register undefined;
//   * a struct of more than 16 bytes passes and returns by reference on arm64: the caller makes a
//     copy and passes a pointer to it, and a method that returns one writes it through a pointer
//     that the caller passed in a register. `MTLSize` is 24 bytes, so a by-value `MTLSize` here is
//     a `*const MTLSize`, and the method sees exactly what a C compiler would have passed. (The
//     copy of an argument is writable in the ABI; the methods here only read it.)

/// `- (NSString *)name`, `- (void *)contents`, `- (id)commandBuffer`, `- (id)computeCommandEncoder`,
/// `- (id)newCommandQueue`, `- (NSError *)error`, `- (NSString *)domain`, `- (id)retain`: one
/// object in, one object out.
const MsgSendObject = *const fn (Id, Sel) callconv(.c) Id;

/// `- (void)endEncoding`, `- (void)commit`, `- (void)waitUntilCompleted`, `- (void)release`: no
/// arguments and no value.
const MsgSendVoid = *const fn (Id, Sel) callconv(.c) void;

/// `- (NSUInteger)status`, `- (NSUInteger)maxTotalThreadsPerThreadgroup`,
/// `- (NSUInteger)threadExecutionWidth`: a number.
const MsgSendUnsigned = *const fn (Id, Sel) callconv(.c) NSUInteger;

/// `- (NSInteger)code` of an `NSError`.
const MsgSendInteger = *const fn (Id, Sel) callconv(.c) NSInteger;

/// `- (MTLSize)maxThreadsPerThreadgroup`: a struct of more than 16 bytes, which the method returns
/// through a pointer.
const MsgSendSize = *const fn (Id, Sel) callconv(.c) MTLSize;

/// `- (BOOL)supportsFamily:(MTLGPUFamily)family`.
const MsgSendBoolFamily = *const fn (Id, Sel, NSInteger) callconv(.c) BOOL;

/// `- (const char *)UTF8String`: the bytes of an `NSString`.
const MsgSendCString = *const fn (Id, Sel) callconv(.c) ?[*:0]const u8;

/// `- (id)newFunctionWithName:(NSString *)name`.
const MsgSendObjectWithObject = *const fn (Id, Sel, Id) callconv(.c) Id;

/// `+ (id)stringWithUTF8String:(const char *)text`, a class method of `NSString`; the class object
/// is the receiver.
const MsgSendObjectWithCString = *const fn (Id, Sel, [*:0]const u8) callconv(.c) Id;

/// `- (id)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options`.
const MsgSendObjectWithLength = *const fn (Id, Sel, NSUInteger, NSUInteger) callconv(.c) Id;

/// `- (id)newLibraryWithData:(dispatch_data_t)data error:(NSError **)error` and
/// `- (id)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError **)error`:
/// the two messages that say why they failed through an `NSError` that the caller passes the
/// address of. A message that returns null writes the error there, which is the only reason to
/// know why.
const MsgSendObjectWithObjectError = *const fn (Id, Sel, Id, *Id) callconv(.c) Id;

/// `- (void)setComputePipelineState:(id<MTLComputePipelineState>)state`.
const MsgSendVoidWithObject = *const fn (Id, Sel, Id) callconv(.c) void;

/// `- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index`.
const MsgSendVoidSetBuffer = *const fn (Id, Sel, Id, NSUInteger, NSUInteger) callconv(.c) void;

/// `- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index`: the
/// bytes are copied into the command buffer, so the memory of `bytes` can be a local that is gone
/// when the message returns.
const MsgSendVoidSetBytes = *const fn (Id, Sel, ?*const anyopaque, NSUInteger, NSUInteger) callconv(.c) void;

/// `- (void)dispatchThreadgroups:(MTLSize)threadgroupsPerGrid threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup`:
/// the two sizes are structs of more than 16 bytes, so they pass by reference; see above.
const MsgSendVoidDispatch = *const fn (Id, Sel, *const MTLSize, *const MTLSize) callconv(.c) void;

/// The type of the value that a message of the shape `MsgSend` returns.
fn ReturnType(comptime MsgSend: type) type {
    return @typeInfo(@typeInfo(MsgSend).pointer.child).@"fn".return_type.?;
}

/// Sends one Objective-C message and returns what the method returns.
///
/// `receiver` is the object, `selector` is the `SEL` of the method, `args` are the arguments after
/// the selector with exactly the types that `MsgSend` takes, and `msg_send` is the address of
/// `objc_msgSend` that `Driver.open` looked up. The cast to `MsgSend` is what makes the call well
/// formed: on arm64 the arguments go in registers and there is no variadic `objc_msgSend` to fall
/// back on, so the type of the function pointer is the whole ABI.
fn call(msg_send: *const anyopaque, comptime MsgSend: type, receiver: Id, selector: Sel, args: anytype) ReturnType(MsgSend) {
    const method: MsgSend = @ptrCast(@alignCast(msg_send));
    return @call(.auto, method, .{ receiver, selector } ++ args);
}

/// A loaded runtime: the libraries, the functions and selectors that this namespace uses, and the
/// GPU of the machine.
///
/// Everything else in this namespace comes from a driver: the driver must stay open while a
/// `Device`, `Context`, `Module`, `Function`, `Pipeline`, or `Buffer` from it is in use.
pub const Driver = struct {
    /// The loaded libraries, which `close` unloads.
    libraries: Libraries,
    /// The functions of those libraries that this namespace calls.
    functions: Functions,
    /// The `SEL` of every `Selector`, in the order of the enumeration.
    selectors: [selector_count]Sel,
    /// The `NSString` class: the strings of this namespace are made with it, and the class object
    /// belongs to the runtime and is never freed.
    string_class: Id,
    /// The `id<MTLDevice>` of the machine, retained by `open` and released by `close`.
    device_object: *anyopaque,

    /// The number of members of `Selector`, and so the length of `Driver.selectors`.
    const selector_count = @typeInfo(Selector).@"enum".field_names.len;

    /// Loads the libraries, resolves the functions and selectors that this namespace uses, and
    /// creates the device of the machine.
    ///
    /// `error.FrameworkNotFound` means that the libraries are not where this namespace looks for
    /// them, which is the case on every operating system other than macOS. A Mac whose framework
    /// has no GPU to give reports `error.NoDevice`, an Intel Mac reports `error.NotSupported`, and
    /// a target without libc, where there is no `dlopen` to load the libraries with, is a compile
    /// error.
    pub fn open() OpenError!Driver {
        if (comptime driver_supported) {
            var libraries = Libraries.open() orelse return error.FrameworkNotFound;
            errdefer libraries.close();

            var functions: Functions = undefined;
            inline for (function_symbols) |entry| {
                const pointer = libraries.lookup(@FieldType(Functions, entry.field), entry.library, entry.symbol);
                @field(functions, entry.field) = pointer orelse return error.IncompatibleFramework;
            }

            // The class objects and the selectors belong to the Objective-C runtime, and a
            // selector that the runtime does not have is registered by asking for it. `NSString`
            // arrives with Foundation, one of the frameworks that the Metal framework loads, which
            // is why the framework is loaded before the class is looked up.
            const string_class = functions.get_class("NSString") orelse return error.IncompatibleFramework;
            var selectors: [selector_count]Sel = undefined;
            inline for (std.enums.values(Selector)) |selector| {
                selectors[@backingInt(selector)] = functions.register_selector(selector.text());
            }

            const device_object = functions.create_system_default_device() orelse return error.NoDevice;
            return .{
                .libraries = libraries,
                .functions = functions,
                .selectors = selectors,
                .string_class = string_class,
                .device_object = device_object,
            };
        } else if (builtin.os.tag == .macos and builtin.cpu.arch == .aarch64) {
            @compileError("std.gpu.metal needs libc to load the Metal framework with dlopen; link the program with libc");
        } else if (builtin.os.tag == .macos) {
            // An Intel Mac has the framework, but not the programs of these messages.
            return error.NotSupported;
        } else {
            return error.FrameworkNotFound;
        }
    }

    /// Releases the device of the machine and unloads the libraries. The driver and everything
    /// that came from it must not be used afterwards, which is why this is the last call of a
    /// program that uses this namespace.
    pub fn close(driver: *Driver) void {
        if (comptime driver_supported) {
            const pool = driver.pushPool();
            _ = driver.send(MsgSendVoid, driver.device_object, .release, .{});
            driver.popPool(pool);
            // Inside the guard as well: `std.DynLib` is a library with no functions of its own on
            // a target that this namespace cannot load a library on, and `close` is not there.
            driver.libraries.close();
        }
        driver.* = undefined;
    }

    /// The GPU of the machine: the one that macOS calls the system default, which is the discrete
    /// GPU on a Mac that has one and the integrated GPU otherwise.
    ///
    /// The device object belongs to the driver, which makes it in `open` and releases it in
    /// `close`, so this is a handle onto it and needs no release of its own.
    pub fn device(driver: *const Driver) Device {
        return .{ .driver = driver, .object = driver.device_object };
    }

    /// Sends one Objective-C message to `receiver` and returns what the method returns.
    ///
    /// `MsgSend` is the shape of the message -- one of the types above -- and `selector` says
    /// which method of the receiver to send; `args` are the arguments that follow the selector,
    /// with exactly the types that `MsgSend` takes. Nothing checks either of them.
    fn send(
        driver: *const Driver,
        comptime MsgSend: type,
        receiver: Id,
        comptime selector: Selector,
        args: anytype,
    ) ReturnType(MsgSend) {
        return call(driver.functions.msg_send, MsgSend, receiver, driver.selectors[@backingInt(selector)], args);
    }

    /// An `NSString` of the UTF-8 bytes of a null-terminated string, with
    /// `+[NSString stringWithUTF8String:]`, or null for bytes that are not UTF-8. The string is
    /// autoreleased: send this from inside an autorelease pool, which every call of this namespace
    /// that makes a string covers itself with.
    fn string(driver: *const Driver, text: [*:0]const u8) Id {
        return driver.send(MsgSendObjectWithCString, driver.string_class, .string_with_utf8_string, .{text});
    }

    /// The UTF-8 bytes of an `NSString`, or an empty slice for a null object or a string that has
    /// no bytes of its own. The bytes belong to the string; an autoreleased string keeps them for
    /// as long as the pool of the call that read them.
    fn stringValue(driver: *const Driver, object: Id) []const u8 {
        const string_object = object orelse return "";
        const utf8 = driver.send(MsgSendCString, string_object, .utf8_string, .{}) orelse return "";
        return std.mem.span(utf8);
    }

    /// A thread's autorelease pool: the objects that the messages of a call create without giving
    /// the ownership of them away are released when the pool is popped.
    fn pushPool(driver: *const Driver) Id {
        return driver.functions.pool_push();
    }

    /// Pops the pool that `pushPool` pushed.
    fn popPool(driver: *const Driver, pool: Id) void {
        driver.functions.pool_pop(pool);
    }
};

/// A GPU that kernels can run on, from `Driver.device`.
pub const Device = struct {
    /// The driver that owns this device, which must outlive it.
    driver: *const Driver,
    /// The `id<MTLDevice>` of the device.
    object: *anyopaque,

    /// Copies the name of the device, such as "Apple M4", into `buffer` and returns it as a slice
    /// of `buffer`, without the null byte after it. A name that does not fit is cut to
    /// `buffer.len - 1` bytes; 256 bytes hold any name that macOS reports.
    ///
    /// The buffer must have room for at least one byte of a name and the null byte after it, and a
    /// shorter one reports `error.InvalidValue`.
    pub fn name(dev: Device, buffer: []u8) Error![]const u8 {
        if (buffer.len < 2) return error.InvalidValue;
        const pool = dev.driver.pushPool();
        defer dev.driver.popPool(pool);
        const text = dev.driver.stringValue(dev.driver.send(MsgSendObject, dev.object, .name, .{}));
        const len = @min(text.len, buffer.len - 1);
        @memcpy(buffer[0..len], text[0..len]);
        buffer[len] = 0;
        return buffer[0..len];
    }

    /// The largest threadgroup that the device can run, in each of the three dimensions, such as
    /// `{ .x = 1024, .y = 1024, .z = 64 }` on an Apple GPU. A threadgroup of a launch is at most
    /// this size, and a kernel can be smaller: `Pipeline.maxTotalThreadsPerThreadgroup` is the
    /// limit of one kernel, which is the smaller of the two that a launch has to respect.
    pub fn maxThreadsPerThreadgroup(dev: Device) Dim3 {
        const size = dev.driver.send(MsgSendSize, dev.object, .max_threads_per_threadgroup, .{});
        return .{ .x = @intCast(size.width), .y = @intCast(size.height), .z = @intCast(size.depth) };
    }

    /// Whether the device is of one of the families of `Family`. A device that is of a family is
    /// of the earlier families of the same line as well, so an Apple GPU of the M4 answers true for
    /// `apple9` and for every `apple` member before it.
    ///
    /// A program that uses a part of Metal that a family introduced asks this before it uses it:
    /// the device of a Mac that is too old runs the kernels of the families it has, and reports
    /// what it cannot do here rather than when a kernel is dispatched.
    pub fn supportsFamily(dev: Device, family: Family) bool {
        return dev.driver.send(MsgSendBoolFamily, dev.object, .supports_family, .{@backingInt(family)}) != 0;
    }

    /// Creates a context of this device: the command queue that runs kernels on it, with the
    /// objects that a kernel launch needs.
    ///
    /// Metal has no context object of its own, and nothing here is shared with another thread, so
    /// every call makes a context of its own; two contexts of one device run kernels of their own,
    /// one queue each. A device that cannot give a queue has no room for the work of a kernel,
    /// which is `error.OutOfMemory`, and so is a machine that is out of memory for the list of
    /// command buffers that the context has not waited for.
    pub fn createContext(dev: Device) Error!Context {
        const pending = try std.heap.page_allocator.create(std.ArrayList(Id));
        pending.* = .empty;
        errdefer std.heap.page_allocator.destroy(pending);
        const queue = dev.driver.send(MsgSendObject, dev.object, .new_command_queue, .{}) orelse
            return error.OutOfMemory;
        return .{ .driver = dev.driver, .device = dev, .queue = queue, .pending = pending };
    }
};

/// The families of GPUs that `Device.supportsFamily` asks about, one member for each
/// `MTLGPUFamily` that macOS does not deprecate.
///
/// `apple*` are the GPUs that Apple designs, in generations; `mac2` is the Mac family 2, which is
/// what the Intel and AMD GPUs of a Mac are; `common*` are the features that the families share;
/// and `metal3` and `metal4` are the families that a feature set of Metal names, which is what a
/// kernel that uses a newer part of the language asks about.
pub const Family = enum(NSInteger) {
    apple1 = 1001,
    apple2 = 1002,
    apple3 = 1003,
    apple4 = 1004,
    apple5 = 1005,
    apple6 = 1006,
    apple7 = 1007,
    apple8 = 1008,
    apple9 = 1009,
    apple10 = 1010,
    mac2 = 2002,
    common1 = 3001,
    common2 = 3002,
    common3 = 3003,
    metal3 = 5001,
    metal4 = 5002,
};

/// A device and the command queue that runs its kernels: the roles that a `std.gpu.cuda` or
/// `std.gpu.hip` context plays.
///
/// Metal has no context object of its own: buffers and modules come from the device, and the queue
/// is what a kernel is dispatched on and what `synchronize` waits for. A context owns the command
/// buffers of the launches that have not synchronized yet, and a copy of a context is that same
/// context: the copies share the queue and the command buffers of the launches, so one
/// `synchronize` waits for the work of all of them, and `release` releases the queue once.
///
/// A context is not thread-safe: two threads that launch on one context at the same time race for
/// the list of command buffers. Two contexts of the same device are independent of each other.
pub const Context = struct {
    /// The driver that owns this context, which must outlive it.
    driver: *const Driver,
    /// The device that this context runs kernels on.
    device: Device,
    /// The `id<MTLCommandQueue>` of the context, released by `release`.
    queue: Id,
    /// The command buffers that were committed and not yet waited for, oldest first. A command
    /// buffer that a queue hands out is autoreleased, so each one of these holds a retain of its
    /// own; `drain` waits for them and gives the retains back. The list is behind a pointer that
    /// `createContext` allocates and `release` frees, so that the copies of a context share it.
    pending: *std.ArrayList(Id),

    /// Allocates a buffer of `len` elements of `T` in unified memory: the memory that the CPU and
    /// the GPU share.
    ///
    /// The memory stays allocated until `Buffer.free`, and it belongs to this device. `len` must be
    /// at least 1, because a buffer of no elements is one that no kernel and no copy can be given.
    pub fn alloc(ctx: Context, comptime T: type, len: usize) Error!Buffer(T) {
        if (comptime @sizeOf(T) == 0) {
            @compileError("cannot allocate a Buffer of the zero-sized type '" ++ @typeName(T) ++ "'");
        }
        if (len == 0) return error.InvalidValue;
        const bytes = std.math.mul(usize, @sizeOf(T), len) catch return error.OutOfMemory;
        const object = ctx.driver.send(MsgSendObjectWithLength, ctx.device.object, .new_buffer, .{
            @as(NSUInteger, bytes),
            resource_storage_mode_shared,
        }) orelse return error.OutOfMemory;
        errdefer _ = ctx.driver.send(MsgSendVoid, object, .release, .{});
        const contents = ctx.driver.send(MsgSendObject, object, .contents, .{}) orelse return error.Unexpected;
        return .{ .driver = ctx.driver, .object = object, .contents = @ptrCast(contents), .len = len };
    }

    /// Loads a module from the bytes of a `.metallib`: the container of the AIR of a library of
    /// kernels, which Zig++ writes with `zig build-obj -target air64-macos -femit-bin`.
    ///
    /// `image` is the whole file, which needs no null byte on the end. The framework reads what it
    /// needs of the image while it loads the module, so the memory of the image can be freed
    /// afterwards.
    ///
    /// A module whose functions were not compiled yet is loaded: the Metal compiler turns the AIR
    /// of a function into the code of the device when `Function.pipeline` is called, and that is
    /// where a kernel that cannot be compiled is reported. Bytes that are not a `.metallib` that
    /// this macOS knows are `error.InvalidLibrary`, and `Options.error_log` receives what the
    /// framework said about them.
    pub fn loadModule(ctx: Context, image: []const u8, options: Options) Error!Module {
        const driver = ctx.driver;
        const pool = driver.pushPool();
        defer driver.popPool(pool);

        const data = driver.functions.dispatch_data_create(@ptrCast(image.ptr), image.len, null, null) orelse
            return error.OutOfMemory;
        defer _ = driver.send(MsgSendVoid, data, .release, .{});

        var error_object: Id = null;
        const library = driver.send(MsgSendObjectWithObjectError, ctx.device.object, .new_library, .{
            data,
            &error_object,
        }) orelse return failure(driver, options.error_log, error_object, error.InvalidLibrary);
        return .{ .context = ctx, .object = library };
    }

    /// Waits until every command buffer that this context committed has finished, and reports the
    /// error of the first one that failed. The command buffers are gone afterwards, whether they
    /// failed or not, and work that is committed later starts from an empty list.
    ///
    /// A launch is asynchronous: when `Pipeline.launch` returns, the kernels that it encoded may
    /// still be reading and writing the buffers of the launch, and a kernel that fails on the GPU
    /// reports its error here, not from the launch. A host that reads what a kernel wrote, or that
    /// writes memory that a kernel reads, waits here first.
    pub fn synchronize(ctx: Context) Error!void {
        if (ctx.drain()) |err| return err;
    }

    /// Frees the command buffers of the launches that were not synchronized yet, the memory of the
    /// list, and the command queue. The context and everything that came from it must not be used
    /// afterwards.
    ///
    /// The work that was still in flight is waited for, as `synchronize` does; an error of a
    /// kernel that failed is not reported again here, and was already reported by the synchronize
    /// that would have waited for it.
    pub fn release(ctx: Context) void {
        _ = ctx.drain();
        std.heap.page_allocator.destroy(ctx.pending);
        const pool = ctx.driver.pushPool();
        _ = ctx.driver.send(MsgSendVoid, ctx.queue, .release, .{});
        ctx.driver.popPool(pool);
    }

    /// Waits for every command buffer that this context committed and has not synchronized yet,
    /// giving back the retain that `Pipeline.launch` took of each one, and returns the error of the
    /// first one that failed. The list of command buffers is emptied either way.
    fn drain(ctx: Context) ?Error {
        const driver = ctx.driver;
        const pool = driver.pushPool();
        defer driver.popPool(pool);

        var failed: ?Error = null;
        for (ctx.pending.items) |command_buffer| {
            driver.send(MsgSendVoid, command_buffer, .wait_until_completed, .{});
            // The status of a command buffer that `waitUntilCompleted` returned for is
            // `MTLCommandBufferStatusCompleted`, or `MTLCommandBufferStatusError` with the reason
            // in the `NSError` of the buffer.
            const status = driver.send(MsgSendUnsigned, command_buffer, .status, .{});
            if (status == command_buffer_status_error and failed == null) {
                const error_object = driver.send(MsgSendObject, command_buffer, .@"error", .{});
                failed = failure(driver, null, error_object, error.CommandBufferError);
            }
            _ = driver.send(MsgSendVoid, command_buffer, .release, .{});
        }
        ctx.pending.clearAndFree(std.heap.page_allocator);
        return failed;
    }
};

/// A module of kernels, loaded from the bytes of a `.metallib` by `Context.loadModule`.
///
/// The module holds the AIR of its kernels, which the Metal compiler reads when a pipeline state is
/// created, so it must outlive the functions and the pipeline states that came from it and must be
/// unloaded after them.
pub const Module = struct {
    /// The context that loaded this module: its queue is what the kernels of the module run on,
    /// and it must outlive them. This is a copy of the context the module was loaded with, which is
    /// the same context as long as that one was not released.
    context: Context,
    /// The `id<MTLLibrary>` of the module, released by `unload`.
    object: Id,

    /// Unloads the module. The functions that came from it and the pipeline states made from them
    /// must not be used afterwards.
    ///
    /// Errors are not reported: a module that was loaded cannot fail to be unloaded, and a kernel
    /// that was still running on its code has already failed if it was going to.
    pub fn unload(module: Module) void {
        _ = module.context.driver.send(MsgSendVoid, module.object, .release, .{});
    }

    /// Looks up a kernel in the module by the name that it was written with and exported, such as
    /// "vadd".
    ///
    /// A module that has no function of that name reports `error.FunctionNotFound`, which is what
    /// a `.metallib` that was built from other kernels than the program expects answers.
    pub fn function(module: Module, name: [:0]const u8) Error!Function {
        const driver = module.context.driver;
        const pool = driver.pushPool();
        defer driver.popPool(pool);

        // A name that is not UTF-8 makes no `NSString` at all, and then no library can have it.
        const text = driver.string(name) orelse return error.FunctionNotFound;
        const object = driver.send(MsgSendObjectWithObject, module.object, .new_function, .{text}) orelse
            return error.FunctionNotFound;
        return .{ .context = module.context, .object = object };
    }
};

/// A kernel function of a module, from `Module.function`. Its pipeline state is the code that a
/// launch dispatches.
pub const Function = struct {
    /// The context of the module of this function, which must outlive it.
    context: Context,
    /// The `id<MTLFunction>` of the kernel, released by `release`.
    object: Id,

    /// Releases the function. The pipeline states that were made from it stay valid, and the module
    /// that it came from must be unloaded after it.
    pub fn release(function: Function) void {
        _ = function.context.driver.send(MsgSendVoid, function.object, .release, .{});
    }

    /// Creates the pipeline state of this kernel: the code of the device that the Metal compiler
    /// makes from the AIR of the module.
    ///
    /// This is the call that compiles a kernel, so it takes a while, and it is where the compiler
    /// reports a kernel that it cannot compile: `error.CompileFailure` with the message of the
    /// compiler in `Options.error_log`. Create the pipeline state once, when the kernels of a
    /// program are being set up, and launch it many times: a pipeline state holds the compiled code
    /// and is what a launch is encoded with.
    pub fn pipeline(function: Function, options: Options) Error!Pipeline {
        const driver = function.context.driver;
        const pool = driver.pushPool();
        defer driver.popPool(pool);

        var error_object: Id = null;
        const object = driver.send(MsgSendObjectWithObjectError, function.context.device.object, .new_pipeline, .{
            function.object,
            &error_object,
        }) orelse return failure(driver, options.error_log, error_object, error.CompileFailure);
        return .{ .context = function.context, .object = object };
    }
};

/// The compiled code of a kernel, from `Function.pipeline`. A launch is encoded with a pipeline
/// state, and the kernels that it encodes run on the command queue of the context that loaded the
/// module of the kernel.
pub const Pipeline = struct {
    /// The context that loaded the module of this kernel, which must outlive the pipeline: its
    /// queue is what the launches of this pipeline are committed to.
    context: Context,
    /// The `id<MTLComputePipelineState>` of the kernel, released by `release`.
    object: Id,

    /// Releases the pipeline state. The context and the module that it came from must be released
    /// and unloaded after it.
    pub fn release(pipeline: Pipeline) void {
        _ = pipeline.context.driver.send(MsgSendVoid, pipeline.object, .release, .{});
    }

    /// The largest number of threads that a threadgroup of this kernel can have: the limit of
    /// `LaunchConfig.block` for this kernel, which is at most the
    /// `Device.maxThreadsPerThreadgroup` of the device.
    pub fn maxTotalThreadsPerThreadgroup(pipeline: Pipeline) u32 {
        const threads = pipeline.context.driver.send(MsgSendUnsigned, pipeline.object, .max_total_threads_per_threadgroup, .{});
        return @intCast(threads);
    }

    /// The number of threads that the GPU runs together: the width of a SIMD group, which is what
    /// `std.gpu.warp_size` is for the Metal target, 32 on every Apple GPU. The warp functions of
    /// `std.gpu` operate on that many threads, and `std.gpu.laneId` is the position of the calling
    /// thread in its group.
    pub fn threadExecutionWidth(pipeline: Pipeline) u32 {
        const threads = pipeline.context.driver.send(MsgSendUnsigned, pipeline.object, .thread_execution_width, .{});
        return @intCast(threads);
    }

    /// Runs the kernel on the threads of `config`: one thread per thread position, in threadgroups
    /// of `config.block` threads, with `config.grid` threadgroups.
    ///
    /// `args` is a tuple with one element for every parameter of the kernel that the host binds, in
    /// the order that the kernel declares them: the builtins of a kernel, such as the position of
    /// the calling thread in the grid, are supplied by the dispatch and are not part of the tuple.
    /// An argument passes in one of these ways, and each one goes to the buffer index of its place
    /// in the tuple:
    /// * a `Buffer` binds its memory as the buffer index of the argument, which the kernel
    ///   receives in a pointer parameter such as `[*]addrspace(.global) f32`,
    /// * integers, floats, `bool`, enums, vectors, and `extern` and `packed` structs bind their
    ///   bytes at that index, with `setBytes:length:atIndex:`, which is where a scalar parameter
    ///   reads them from.
    ///
    /// Slices and host pointers cannot pass, because a kernel cannot read the memory of the host
    /// process: put the data in a `Buffer` and pass that. Neither can a `comptime_int` or a
    /// `comptime_float`, because the compiler has not chosen a type for it: write the type, as in
    /// `@as(u32, 256)`. Each of these is a compile error.
    ///
    /// The launch is asynchronous: it returns when the kernel is encoded and the command buffer
    /// that holds it is committed, and the kernel may not have run yet. `Context.synchronize`
    /// waits for it and reports a kernel that failed on the GPU; the buffers of the launch must not
    /// be freed, and the host must not read or write their memory, until then.
    pub fn launch(pipeline: Pipeline, config: LaunchConfig, args: anytype) Error!void {
        const field_types = comptime launch_arguments.argumentTypes(@TypeOf(args));
        if (config.grid.x == 0 or config.grid.y == 0 or config.grid.z == 0) {
            // A dispatch of no threadgroups is not one that the framework accepts.
            return error.InvalidValue;
        }
        if (config.block.x == 0 or config.block.y == 0 or config.block.z == 0) {
            // A threadgroup of no threads, such as one of no height, is not one that the framework
            // accepts either.
            return error.InvalidValue;
        }
        // A threadgroup that is larger than the kernel allows is the other dispatch that the
        // framework refuses, and one that is encoded anyway fails only when the command buffer
        // runs, so this is where the mistake is caught. The product is taken in 64 bits: a block
        // of 65536 threads in each dimension overflows a `u32`.
        const threads = @as(u64, config.block.x) * config.block.y * config.block.z;
        if (threads > pipeline.maxTotalThreadsPerThreadgroup()) return error.InvalidValue;

        const driver = pipeline.context.driver;
        const pool = driver.pushPool();
        defer driver.popPool(pool);

        const command_buffer = driver.send(MsgSendObject, pipeline.context.queue, .command_buffer, .{}) orelse
            return error.OutOfMemory;
        errdefer _ = driver.send(MsgSendVoid, command_buffer, .release, .{});
        // `-commandBuffer` hands out an autoreleased command buffer, and this one has to stay alive
        // until `Context.synchronize` waits for it, which is after the pool of this call is gone.
        _ = driver.send(MsgSendObject, command_buffer, .retain, .{});

        const encoder = driver.send(MsgSendObject, command_buffer, .compute_command_encoder, .{}) orelse
            return error.OutOfMemory;
        driver.send(MsgSendVoidWithObject, encoder, .set_pipeline, .{pipeline.object});
        inline for (field_types, 0..) |field_type, index| {
            if (comptime launch_arguments.isBuffer(Buffer, field_type)) {
                driver.send(MsgSendVoidSetBuffer, encoder, .set_buffer, .{
                    args[index].object,
                    @as(NSUInteger, 0),
                    @as(NSUInteger, index),
                });
            } else {
                const value: launch_arguments.ValueStorage(field_type, index, "a Buffer") = args[index];
                driver.send(MsgSendVoidSetBytes, encoder, .set_bytes, .{
                    @as(?*const anyopaque, @ptrCast(&value)),
                    @as(NSUInteger, @sizeOf(field_type)),
                    @as(NSUInteger, index),
                });
            }
        }
        const grid = MTLSize{ .width = config.grid.x, .height = config.grid.y, .depth = config.grid.z };
        const block = MTLSize{ .width = config.block.x, .height = config.block.y, .depth = config.block.z };
        driver.send(MsgSendVoidDispatch, encoder, .dispatch_threadgroups, .{ &grid, &block });
        driver.send(MsgSendVoid, encoder, .end_encoding, .{});
        driver.send(MsgSendVoid, command_buffer, .commit, .{});

        // The command buffer belongs to the context from here on: the error of a kernel that fails
        // on the GPU is what `Context.synchronize` reports.
        try pipeline.context.pending.append(std.heap.page_allocator, command_buffer);
    }
};

/// The size of a grid or of a threadgroup in each of the three dimensions.
pub const Dim3 = struct {
    /// The number of threadgroups, or of threads, in the x dimension.
    x: u32 = 1,
    /// The number of threadgroups, or of threads, in the y dimension.
    y: u32 = 1,
    /// The number of threadgroups, or of threads, in the z dimension.
    z: u32 = 1,
};

/// How `Pipeline.launch` arranges the threads of a kernel in a grid of threadgroups.
pub const LaunchConfig = struct {
    /// The number of threadgroups in each dimension of the grid.
    grid: Dim3 = .{},
    /// The number of threads in each dimension of a threadgroup. Every threadgroup has the same
    /// shape, and it holds at least one thread in each dimension. A threadgroup is at most
    /// `Pipeline.maxTotalThreadsPerThreadgroup` threads of the kernel, which is at most the
    /// `Device.maxThreadsPerThreadgroup` of the device; a launch whose `block` is not of that
    /// shape reports `error.InvalidValue` before anything is dispatched.
    block: Dim3 = .{},

    /// A one-dimensional launch of at least `n` threads, in threadgroups of `block_size` threads.
    ///
    /// The grid has as many threadgroups as it takes to cover `n` threads, rounded up, so every
    /// threadgroup is full and the threads of the last one that are past the end of the range must
    /// be skipped by the kernel, which compares its thread index against `n`.
    ///
    /// `block_size` must be at least 1. `n == 0` gives a grid of no threadgroups, which the
    /// framework does not accept, so do not launch when `n` is zero.
    pub fn linear(n: u32, block_size: u32) LaunchConfig {
        std.debug.assert(block_size != 0);
        return .{
            .grid = .{ .x = if (n == 0) 0 else (n - 1) / block_size + 1 },
            .block = .{ .x = block_size },
        };
    }
};

/// A region of memory that the CPU and the GPU share, from `Context.alloc`.
///
/// The memory is in unified memory, so `values` is the memory as a slice of `T`: the bytes that the
/// kernels of a launch on this buffer read and write, and the bytes that the host reads and writes
/// with no copy of the framework's in between. `copyFromHost` and `copyToHost` copy between that
/// memory and a host slice, and `zero` fills the memory with zero bytes, which is why none of the
/// three reports an error and none of them calls into the driver.
///
/// A kernel must not be reading or writing the memory while the host writes it: wait for the launch
/// that uses the buffer with `Context.synchronize` first. Writes that a kernel made to the memory
/// are visible to the host once the synchronize that waits for its launch has returned.
pub fn Buffer(comptime T: type) type {
    return struct {
        /// The driver that owns this memory, which must outlive the buffer.
        driver: *const Driver,
        /// The `id<MTLBuffer>` of the memory, released by `free`.
        object: Id,
        /// The bytes of the memory as the CPU sees them: the `contents` of the buffer, which a
        /// buffer of shared memory always has.
        contents: [*]u8,
        /// The number of `T` elements that the memory holds.
        len: usize,

        /// The type of the elements of the buffer.
        pub const Elem = T;

        /// The memory as a slice of `T`. The pointer is the `contents` of the buffer, which macOS
        /// gives aligned to the page size, and so to every scalar and vector that a kernel
        /// parameter can take; `@alignCast` reports it in a safe build if that ever changes.
        pub fn values(buffer: @This()) []T {
            return @as([*]T, @ptrCast(@alignCast(buffer.contents)))[0..buffer.len];
        }

        /// Frees the memory. The buffer must not be used afterwards, and the launches that used it
        /// must have been synchronized: a buffer that is freed while a command buffer that refers
        /// to it is still to run is what the framework describes as the usual cause of that command
        /// buffer failing with `error.InvalidResource`.
        pub fn free(buffer: @This()) void {
            _ = buffer.driver.send(MsgSendVoid, buffer.object, .release, .{});
        }

        /// Copies `src` into the buffer, starting at the first element. `src` may be shorter than
        /// the buffer, but not longer; a longer `src` fails an assertion.
        pub fn copyFromHost(buffer: @This(), src: []const T) void {
            std.debug.assert(src.len <= buffer.len);
            @memcpy(buffer.values()[0..src.len], src);
        }

        /// Copies the buffer into `dst`, starting at the first element. `dst` may be shorter than
        /// the buffer, but not longer; a longer `dst` fails an assertion.
        pub fn copyToHost(buffer: @This(), dst: []T) void {
            std.debug.assert(dst.len <= buffer.len);
            @memcpy(dst, buffer.values()[0..dst.len]);
        }

        /// Fills the whole buffer with zero bytes, whatever `T` is.
        pub fn zero(buffer: @This()) void {
            @memset(buffer.contents[0 .. buffer.len * @sizeOf(T)], 0);
        }
    };
}

/// The error for the domain and the code of an `NSError`, or null for a domain or a code that this
/// version of the standard library does not know, which the caller then reports as the failure of
/// its own call.
///
/// The codes are the members of `MTLLibraryError` and of `MTLCommandBufferError`, which the Metal
/// headers name; the names of the `Error` members say what each one means. The domains are the
/// strings that those headers declare, and an error of any other domain, such as one of the
/// compiler's own, is left to the caller.
fn errorFor(domain: []const u8, code: NSInteger) ?Error {
    if (std.mem.eql(u8, domain, "MTLLibraryErrorDomain")) {
        return switch (code) {
            1 => error.InvalidLibrary, // MTLLibraryErrorUnsupported
            2 => error.InternalError, // MTLLibraryErrorInternal
            3 => error.CompileFailure, // MTLLibraryErrorCompileFailure
            5 => error.FunctionNotFound, // MTLLibraryErrorFunctionNotFound
            6 => error.InvalidLibrary, // MTLLibraryErrorFileNotFound
            else => null,
        };
    }
    if (std.mem.eql(u8, domain, "MTLCommandBufferErrorDomain")) {
        return switch (code) {
            1 => error.InternalError, // MTLCommandBufferErrorInternal
            2 => error.Timeout, // MTLCommandBufferErrorTimeout
            3 => error.PageFault, // MTLCommandBufferErrorPageFault
            4 => error.AccessRevoked, // MTLCommandBufferErrorAccessRevoked
            7 => error.NotPermitted, // MTLCommandBufferErrorNotPermitted
            8 => error.OutOfMemory, // MTLCommandBufferErrorOutOfMemory
            9 => error.InvalidResource, // MTLCommandBufferErrorInvalidResource
            10 => error.Memoryless, // MTLCommandBufferErrorMemoryless
            11 => error.DeviceRemoved, // MTLCommandBufferErrorDeviceRemoved
            12 => error.StackOverflow, // MTLCommandBufferErrorStackOverflow
            else => null,
        };
    }
    return null;
}

/// The error that a failed call reports: the failure of the `NSError` that a message wrote, and,
/// when the caller gave one, a line about it in its `Options.error_log`.
///
/// `unknown` is what a failure of a domain or a code that this version of the standard library does
/// not know is reported as, which is the failure of the call site: a library that cannot be
/// loaded, a kernel that cannot be compiled, or a command buffer that failed.
fn failure(driver: *const Driver, log: ?[]u8, error_object: Id, unknown: Error) Error {
    const object = error_object orelse {
        if (log) |buffer| writeLog(buffer, &.{"the framework reported no error object"});
        return unknown;
    };
    const code = driver.send(MsgSendInteger, object, .code, .{});
    const domain = driver.stringValue(driver.send(MsgSendObject, object, .domain, .{}));
    if (log) |buffer| {
        var code_text: [24]u8 = undefined;
        const written = std.fmt.bufPrint(&code_text, "{d}", .{code}) catch code_text[0..0];
        const message = driver.stringValue(driver.send(MsgSendObject, object, .localized_description, .{}));
        writeLog(buffer, &.{ domain, " ", written, ": ", message });
    }
    return errorFor(domain, code) orelse unknown;
}

/// Writes the parts of a line into `log` and a null byte after them, cutting the line at the end of
/// the buffer. A log of no bytes gets no line, and needs none.
fn writeLog(log: []u8, parts: []const []const u8) void {
    var at: usize = 0;
    for (parts) |part| {
        if (at + 1 >= log.len) break;
        const len = @min(part.len, log.len - 1 - at);
        @memcpy(log[at..][0..len], part[0..len]);
        at += len;
    }
    if (log.len != 0) log[at] = 0;
}

test "metal: LaunchConfig.linear rounds the thread count up to whole threadgroups" {
    // No threads is an empty grid, with no threadgroups to launch.
    const none = LaunchConfig.linear(0, 128);
    try testing.expectEqual(@as(u32, 0), none.grid.x);
    try testing.expectEqual(@as(u32, 128), none.block.x);

    // A thread count that is a multiple of the size of a threadgroup fills the last one exactly.
    const exact = LaunchConfig.linear(512, 128);
    try testing.expectEqual(@as(u32, 4), exact.grid.x);
    try testing.expectEqual(@as(u32, 128), exact.block.x);
    try testing.expectEqual(@as(u32, 1), exact.grid.y);
    try testing.expectEqual(@as(u32, 1), exact.grid.z);
    try testing.expectEqual(@as(u32, 1), exact.block.y);
    try testing.expectEqual(@as(u32, 1), exact.block.z);

    // One thread more needs a whole threadgroup more.
    const partial = LaunchConfig.linear(513, 128);
    try testing.expectEqual(@as(u32, 5), partial.grid.x);

    const single = LaunchConfig.linear(1, 128);
    try testing.expectEqual(@as(u32, 1), single.grid.x);

    // One thread per threadgroup is one threadgroup per thread.
    const per_thread = LaunchConfig.linear(7, 1);
    try testing.expectEqual(@as(u32, 7), per_thread.grid.x);
    try testing.expectEqual(@as(u32, 1), per_thread.block.x);

    // The largest thread count is covered without overflowing the threadgroup count.
    const all = LaunchConfig.linear(0xffff_ffff, 32);
    const covered = @as(u64, all.grid.x) * 32;
    try testing.expect(covered >= 0xffff_ffff);
    try testing.expect(covered - 32 < 0xffff_ffff);
}

test "metal: the domain and the code of an NSError map to the members of Error" {
    // Every failure that the framework reports is a member of `MTLLibraryError` or of
    // `MTLCommandBufferError`, and the names in the comments are the ones of the Metal headers.
    const failures = [_]struct { domain: []const u8, code: NSInteger, expected: Error }{
        .{ .domain = "MTLLibraryErrorDomain", .code = 1, .expected = error.InvalidLibrary }, // Unsupported
        .{ .domain = "MTLLibraryErrorDomain", .code = 2, .expected = error.InternalError }, // Internal
        .{ .domain = "MTLLibraryErrorDomain", .code = 3, .expected = error.CompileFailure }, // CompileFailure
        .{ .domain = "MTLLibraryErrorDomain", .code = 5, .expected = error.FunctionNotFound }, // FunctionNotFound
        .{ .domain = "MTLLibraryErrorDomain", .code = 6, .expected = error.InvalidLibrary }, // FileNotFound
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 1, .expected = error.InternalError },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 2, .expected = error.Timeout },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 3, .expected = error.PageFault },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 4, .expected = error.AccessRevoked },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 7, .expected = error.NotPermitted },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 8, .expected = error.OutOfMemory },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 9, .expected = error.InvalidResource },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 10, .expected = error.Memoryless },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 11, .expected = error.DeviceRemoved },
        .{ .domain = "MTLCommandBufferErrorDomain", .code = 12, .expected = error.StackOverflow },
    };
    for (failures) |entry| {
        try testing.expectEqual(@as(?Error, entry.expected), errorFor(entry.domain, entry.code));
    }

    // A code of a library failure that is not one of the failures, a command buffer with no error,
    // and a domain that these calls do not report in are left to the call site that failed, which
    // then reports the failure of its own call: a library, a kernel, or a command buffer.
    try testing.expectEqual(@as(?Error, null), errorFor("MTLLibraryErrorDomain", 4)); // CompileWarning
    try testing.expectEqual(@as(?Error, null), errorFor("MTLCommandBufferErrorDomain", 0)); // None
    try testing.expectEqual(@as(?Error, null), errorFor("NSPOSIXErrorDomain", 1));
}

test "metal: open the framework and query the device" {
    var driver = Driver.open() catch |err| switch (err) {
        // No Metal framework, no GPU, or an Intel Mac, where the messages of this namespace are
        // not the programs of the ABI: there is nothing to test on such a machine.
        error.FrameworkNotFound, error.NoDevice, error.NotSupported => return error.SkipZigTest,
        else => |other| return other,
    };
    defer driver.close();

    const device = driver.device();

    var name_buffer: [256]u8 = undefined;
    const name = try device.name(&name_buffer);
    try testing.expect(name.len > 0);

    const largest = device.maxThreadsPerThreadgroup();
    try testing.expect(largest.x >= 1);
    try testing.expect(largest.y >= 1);
    try testing.expect(largest.z >= 1);

    // A GPU is of at least one of the families that this namespace names.
    const families = comptime std.enums.values(Family);
    var supported = false;
    for (families) |family| {
        if (device.supportsFamily(family)) supported = true;
    }
    try testing.expect(supported);

    var context = try device.createContext();
    defer context.release();
}

test "metal: a buffer in unified memory holds the bytes of the host" {
    var driver = Driver.open() catch |err| switch (err) {
        // No Metal framework, no GPU, or an Intel Mac, where the messages of this namespace are
        // not the programs of the ABI: there is nothing to test on such a machine.
        error.FrameworkNotFound, error.NoDevice, error.NotSupported => return error.SkipZigTest,
        else => |other| return other,
    };
    defer driver.close();

    var context = try driver.device().createContext();
    defer context.release();

    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const buffer = try context.alloc(f32, data.len);
    defer buffer.free();
    try testing.expectEqual(data.len, buffer.values().len);

    // The memory is the host's as well as the GPU's: what a copy writes is what `values` reads,
    // and what `values` writes is what a copy of it reads.
    buffer.copyFromHost(&data);
    try testing.expectEqualSlices(f32, &data, buffer.values());

    buffer.zero();
    for (buffer.values()) |value| try testing.expectEqual(@as(f32, 0), value);

    // A copy that is shorter than the buffer writes and reads the first elements only.
    buffer.copyFromHost(data[0..3]);
    var host = [_]f32{ 0, 0, 0 };
    buffer.copyToHost(&host);
    try testing.expectEqualSlices(f32, data[0..3], &host);
    try testing.expectEqual(@as(f32, 0), buffer.values()[3]);

    // A buffer of elements of another type is the same memory, read as another type.
    const words = try context.alloc(u32, 4);
    defer words.free();
    words.copyFromHost(&[_]u32{ 0x3f80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000 });
    const as_floats = @as([*]f32, @ptrCast(@alignCast(words.contents)))[0..4];
    try testing.expectEqual(@as(f32, 1), as_floats[0]);
    try testing.expectEqual(@as(f32, 2), as_floats[1]);
}

test "metal: bytes that are not a metallib report an error" {
    var driver = Driver.open() catch |err| switch (err) {
        // No Metal framework, no GPU, or an Intel Mac, where the messages of this namespace are
        // not the programs of the ABI: there is nothing to test on such a machine.
        error.FrameworkNotFound, error.NoDevice, error.NotSupported => return error.SkipZigTest,
        else => |other| return other,
    };
    defer driver.close();

    var context = try driver.device().createContext();
    defer context.release();

    // A `.metallib` starts with the four bytes "MTLB", and nothing that the framework can read
    // comes of anything else.
    const not_a_library = "this is not the container of a library of kernels";
    var log: [512]u8 = @splat(0);
    if (context.loadModule(not_a_library, .{ .error_log = &log })) |module| {
        module.unload();
        return error.TestUnexpectedResult;
    } else |err| switch (err) {
        error.InvalidLibrary, error.InternalError => {},
        else => |other| return other,
    }

    // The framework says what is wrong with the bytes.
    try testing.expect(std.mem.sliceTo(log[0..], 0).len > 0);

    // A library of no bytes is not a library either.
    if (context.loadModule("", .{})) |module| {
        module.unload();
        return error.TestUnexpectedResult;
    } else |err| switch (err) {
        error.InvalidLibrary, error.InternalError => {},
        else => |other| return other,
    }
}
