//! The arguments of a kernel launch, prepared for the driver APIs of `std.gpu.cuda` and
//! `std.gpu.hip`.
//!
//! Both drivers take the arguments of a launch as an array with one pointer per kernel
//! parameter, each of them pointing at the value to pass, so the code that turns the tuple that
//! `Function.launch` is called with into that array is the same for the two. The only thing that
//! this code needs to know about the namespace it works for is the `Buffer` type function of
//! that namespace: a `Buffer` passes to a kernel as the address of its device memory, and a
//! `Buffer` of one namespace does not belong in a launch of the other.

const std = @import("../std.zig");

/// The types of the elements of the tuple of kernel arguments, or a compile error when `args` is
/// not a tuple with one element per parameter.
pub fn argumentTypes(comptime Arguments: type) []const type {
    const info = @typeInfo(Arguments);
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError("the kernel arguments must be a tuple with one element per parameter, " ++
            "such as `.{ buffer, @as(u32, 256) }`, found '" ++ @typeName(Arguments) ++ "'");
    }
    return info.@"struct".field_types;
}

/// The type that holds the copy of a kernel argument that the driver reads. The address of that
/// copy is what the launch passes to the driver.
///
/// `Buffer` is the type function that makes a buffer type of the namespace, such as
/// `std.gpu.cuda.Buffer`.
pub fn ArgumentStorage(comptime Buffer: anytype, comptime T: type, comptime index: usize) type {
    if (comptime isBuffer(Buffer, T)) {
        return @TypeOf(@as(T, undefined).ptr);
    } else {
        return switch (@typeInfo(T)) {
            .comptime_int => @compileError(std.fmt.comptimePrint(
                "kernel parameter {d} is a comptime_int, which has no type that the driver could pass; write the type, such as `@as(u32, 256)`",
                .{index},
            )),
            .comptime_float => @compileError(std.fmt.comptimePrint(
                "kernel parameter {d} is a comptime_float, which has no type that the driver could pass; write the type, such as `@as(f32, 1.5)`",
                .{index},
            )),
            .pointer => @compileError(std.fmt.comptimePrint(
                "kernel parameter {d} is a host pointer or slice ('{s}'), and a kernel cannot read the memory of the host process; put the data in a Buffer and pass that, or pass a DevicePtr",
                .{ index, @typeName(T) },
            )),
            .int, .float, .bool, .vector, .@"enum" => T,
            .@"struct" => |info| if (info.layout == .@"extern" or info.layout == .@"packed")
                T
            else
                @compileError(std.fmt.comptimePrint(
                    "kernel parameter {d} has the type '{s}', which is not an extern or packed struct; only those have the layout that a kernel parameter needs",
                    .{ index, @typeName(T) },
                )),
            else => @compileError(std.fmt.comptimePrint(
                "kernel parameter {d} has the type '{s}', which cannot pass to a kernel; pass a Buffer, a DevicePtr, or a value of an integer, float, bool, enum, vector, extern struct, or packed struct type",
                .{ index, @typeName(T) },
            )),
        };
    }
}

/// The value of one kernel argument that the launch passes to the driver: the address of the
/// memory of a buffer, which the kernel receives in a pointer parameter, or the argument itself.
pub fn argumentValue(comptime Buffer: anytype, comptime T: type, comptime index: usize, arg: T) ArgumentStorage(Buffer, T, index) {
    if (comptime isBuffer(Buffer, T)) {
        return arg.ptr;
    } else {
        return arg;
    }
}

/// The tuple of `ArgumentStorage` types that holds the arguments of a launch. The elements are
/// contiguous, and the driver is passed the address of each one.
pub fn ArgumentTuple(comptime Buffer: anytype, comptime Arguments: type) type {
    const field_types = comptime argumentTypes(Arguments);
    var storage_types: [field_types.len]type = undefined;
    inline for (field_types, 0..) |field_type, index| {
        storage_types[index] = ArgumentStorage(Buffer, field_type, index);
    }
    return @Tuple(&storage_types);
}

/// Whether `T` is a `Buffer` of the namespace, which passes to a kernel as the address of its
/// device memory. `Buffer` is the type function that makes a buffer type of that namespace.
pub fn isBuffer(comptime Buffer: anytype, comptime T: type) bool {
    if (@typeInfo(T) != .@"struct") return false;
    if (!@hasDecl(T, "Elem") or @TypeOf(T.Elem) != type) return false;
    return T == Buffer(T.Elem);
}
