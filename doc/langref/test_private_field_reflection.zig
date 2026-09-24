const std = @import("std");
const expect = std.testing.expect;
const Buffer = @import("test_private_fields.zig").Buffer;

test "reflection includes private fields" {
    const info = @typeInfo(Buffer).@"struct";
    try expect(std.mem.eql(u8, info.field_names[1], "len"));
    try expect(info.field_attrs[1].@"priv");

    // Builtins which name fields with strings are not subject to field privacy.
    var buf: Buffer = .{};
    try buf.append('z');
    try expect(@field(buf, "len") == 1);
}

// test
