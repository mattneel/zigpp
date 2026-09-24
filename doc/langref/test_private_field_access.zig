const Buffer = @import("test_private_fields.zig").Buffer;

test "access a private field from another file" {
    var buf: Buffer = .{};
    try buf.append('a');
    buf.len = 100;
}

// test_error=field 'len' of struct 'test_private_fields.Buffer' is private
