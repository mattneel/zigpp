const std = @import("std");
const builtin = @import("builtin");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const assert = std.debug.assert;
const types = @import("private_fields/types.zig");

const Local = struct {
    priv secret: u32,
    public: u32 = 0,

    fn reveal(l: Local) u32 {
        return l.secret;
    }
};

test "private fields are accessible by name in the declaring file" {
    var local: Local = .{ .secret = 1 };
    local.secret += 1;
    const ptr = &local.secret;
    ptr.* += 1;
    try expectEqual(3, local.reveal());
    try expectEqual(3, local.secret);
}

test "types with private fields are usable through their public interface" {
    var counter: types.Counter = .{};
    counter.step = 2;
    counter.increment();
    counter.increment();
    try expectEqual(4, counter.get());

    try expectEqual(7, types.Box(u8).init(7).get());
    try expectEqual(-3, types.ExternPoint.init(1, -3).getY());
    try expectEqual(5, types.reifiedB(.{ .a = 1 }));
}

test "private union fields" {
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_spirv) return error.SkipZigTest;

    try expectEqual(9, types.Shape.makeSquare(3).area());
    try expectEqual(0, types.Shape.makeEmpty().area());

    // Comparing against the tag, and `else` prongs, do not name the private field.
    const empty = types.Shape.makeEmpty();
    try expect(empty == .empty);
    const is_circle = switch (empty) {
        .circle => true,
        else => false,
    };
    try expect(!is_circle);

    // Neither does `inline else`, which is often used by generic code.
    const side: u32 = switch (types.Shape.makeSquare(4)) {
        inline else => |payload| if (@TypeOf(payload) == void) 0 else payload,
    };
    try expectEqual(4, side);
}

test "reflection includes private fields" {
    const counter = @typeInfo(types.Counter).@"struct";
    comptime assert(counter.field_names.len == 2);
    comptime assert(counter.field_attrs[0].@"priv");
    comptime assert(!counter.field_attrs[1].@"priv");
    comptime assert(@hasField(types.Counter, "count"));
    comptime assert(@FieldType(types.Counter, "count") == u32);

    const handle = @typeInfo(types.Handle).@"struct";
    comptime assert(handle.field_attrs[0].@"priv" and !handle.field_attrs[0].@"comptime");
    comptime assert(handle.field_attrs[1].@"priv" and handle.field_attrs[1].@"comptime");

    comptime assert(@typeInfo(types.Box(u8)).@"struct".field_attrs[0].@"priv");

    const shape = @typeInfo(types.Shape).@"union";
    comptime assert(!shape.field_attrs[0].@"priv");
    comptime assert(shape.field_attrs[1].@"priv");
    comptime assert(shape.field_attrs[2].@"priv");

    const flags = @typeInfo(types.PackedFlags).@"struct";
    comptime assert(!flags.field_attrs[0].@"priv");
    comptime assert(flags.field_attrs[1].@"priv");

    const bits = @typeInfo(types.PackedBits).@"union";
    comptime assert(!bits.field_attrs[0].@"priv");
    comptime assert(bits.field_attrs[1].@"priv");

    const point = @typeInfo(types.ExternPoint).@"struct";
    comptime assert(!point.field_attrs[0].@"priv");
    comptime assert(point.field_attrs[1].@"priv");

    const reified = @typeInfo(types.Reified).@"struct";
    comptime assert(!reified.field_attrs[0].@"priv");
    comptime assert(reified.field_attrs[1].@"priv");

    comptime assert(!@typeInfo(struct { u8 }).@"struct".field_attrs[0].@"priv");
}

test "reflection builtins bypass field privacy" {
    var counter: types.Counter = .{};
    @field(counter, "count") = 10;
    counter.increment();
    try expectEqual(11, @field(counter, "count"));
    try expectEqual(-1, @field(types.Handle.open(-1), "fd"));
    try expectEqual(@offsetOf(types.ExternPoint, "x") + 4, @offsetOf(types.ExternPoint, "y"));
}

test "@unionInit bypasses field privacy" {
    if (builtin.zig_backend == .stage2_sparc64) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_arm) return error.SkipZigTest; // TODO
    if (builtin.zig_backend == .stage2_spirv) return error.SkipZigTest;

    try expectEqual(16, @unionInit(types.Shape, "square", 4).area());
}

test "reification preserves field privacy relative to the reifying file" {
    const info = @typeInfo(types.Counter).@"struct";
    const Copy = @Struct(info.layout, info.backing_integer, info.field_names, info.field_types[0..2], info.field_attrs[0..2]);
    comptime assert(@typeInfo(Copy).@"struct".field_attrs[0].@"priv");
    // `Copy` is declared by the `@Struct` call in this file, so its private fields are accessible.
    var copy: Copy = .{ .count = 1 };
    copy.count += 2;
    try expectEqual(3, copy.count);
}
