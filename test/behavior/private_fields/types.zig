//! Types with private fields, used by `private_fields.zig`. Private fields can only be accessed
//! by name within this file.

pub const Counter = struct {
    priv count: u32 = 0,
    step: u32 = 1,

    pub fn increment(c: *Counter) void {
        c.count += c.step;
    }

    pub fn get(c: Counter) u32 {
        return c.count;
    }
};

pub const Handle = struct {
    priv fd: i32,
    priv comptime kind: u8 = 'h',

    pub fn open(fd: i32) Handle {
        return .{ .fd = fd };
    }
};

pub fn Box(comptime T: type) type {
    return struct {
        priv value: T,

        const Self = @This();

        pub fn init(value: T) Self {
            return .{ .value = value };
        }

        pub fn get(self: Self) T {
            return self.value;
        }
    };
}

pub const Shape = union(enum) {
    circle: u32,
    priv square: u32,
    priv empty,

    pub fn makeSquare(side: u32) Shape {
        return .{ .square = side };
    }

    pub fn makeEmpty() Shape {
        return .empty;
    }

    pub fn area(s: Shape) u32 {
        return switch (s) {
            .circle => |r| 3 * r * r,
            .square => |side| side * side,
            .empty => 0,
        };
    }
};

pub const PackedFlags = packed struct(u8) {
    visible: bool,
    priv reserved: u7 = 0,
};

pub const PackedBits = packed union {
    raw: u16,
    priv halves: packed struct(u16) { lo: u8, hi: u8 },
};

pub const ExternPoint = extern struct {
    x: i32,
    priv y: i32,

    pub fn init(x: i32, y: i32) ExternPoint {
        return .{ .x = x, .y = y };
    }

    pub fn getY(p: ExternPoint) i32 {
        return p.y;
    }
};

/// A reified struct type is declared in the file containing the `@Struct` call, so its private
/// field `b` is only accessible by name in this file.
pub const Reified = @Struct(.auto, null, &.{ "a", "b" }, &.{ u32, u32 }, &.{
    .{},
    .{ .@"priv" = true, .default_value_ptr = &@as(u32, 5) },
});

pub fn reifiedB(r: Reified) u32 {
    return r.b;
}
