const std = @import("std");
const expect = std.testing.expect;

/// A fixed-capacity buffer. `len` is private, so code in other files cannot
/// break the invariant that `len` never exceeds the capacity of `bytes`.
pub const Buffer = struct {
    bytes: [16]u8 = undefined,
    priv len: usize = 0,

    pub fn append(buf: *Buffer, byte: u8) error{Overflow}!void {
        if (buf.len == buf.bytes.len) return error.Overflow;
        buf.bytes[buf.len] = byte;
        buf.len += 1;
    }

    pub fn slice(buf: *const Buffer) []const u8 {
        return buf.bytes[0..buf.len];
    }
};

test "private fields" {
    var buf: Buffer = .{};
    try buf.append('h');
    try buf.append('i');
    try expect(std.mem.eql(u8, buf.slice(), "hi"));

    // This test is in the file which declares `Buffer`, so it can access `len`.
    try expect(buf.len == 2);
}

// test
