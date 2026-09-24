//! minisign signature verification, the format the Zig Software Foundation
//! signs the downloads of ziglang.org with.
//!
//! A minisign public key is base64 of 2 bytes of algorithm ("Ed" for
//! Ed25519), 8 bytes of key id, and 32 bytes of Ed25519 public key. A
//! `.minisig` file has four lines: an untrusted comment, the base64 signature,
//! the trusted comment, and the base64 global signature. The signature line is
//! 2 bytes of algorithm ("ED" when the Ed25519 signature is over the
//! BLAKE2b-512 hash of the file, "Ed" when it is over the file itself), the
//! key id of the key that made it, and the 64-byte Ed25519 signature. The
//! global signature is a second Ed25519 signature over the Ed25519 signature
//! and the text of the trusted comment; checking it is what makes the trusted
//! comment -- the file name that is written there -- trustworthy.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Blake2b512 = std.crypto.hash.blake2.Blake2b512;
const ed25519 = std.crypto.sign.Ed25519;
const mem = std.mem;

/// The public key that the Zig Software Foundation signs upstream downloads
/// with, as ziglang.org publishes it.
pub const zig_public_key = "RWSGOq2NVecA2UPNdBUZykf1CCb147pkmdtYxgb3Ti+JO/wCYvhbAb/U";

pub const Error = error{
    /// The text is not in minisign's format, or the signature is of a kind
    /// that cannot be checked the way it was asked to be.
    Malformed,
    /// The signature was made by a key other than the one it was checked
    /// against.
    WrongKey,
    /// The file is not the one the signature is over.
    InvalidSignature,
    /// The global signature does not verify: the trusted comment is not the
    /// one that was signed together with the signature.
    InvalidGlobalSignature,
    /// The trusted comment does not name the file it was checked for.
    WrongFile,
    /// Reading the file to check it failed: the file, not the signature, is
    /// what needs attention.
    ReadFailed,
};

/// A minisign public key: 2 bytes of algorithm ("Ed"), 8 bytes of key id, and
/// 32 bytes of Ed25519 public key.
pub const PublicKey = struct {
    /// The id of the key, which every signature made with it repeats.
    key_id: [8]u8,
    key: ed25519.PublicKey,

    /// Parses the base64 that minisign writes in a `.pub` file and in the
    /// documentation of a key.
    pub fn fromBase64(text: []const u8) Error!PublicKey {
        var bytes: [2 + 8 + ed25519.PublicKey.encoded_length]u8 = undefined;
        if (!decodeExact(&bytes, text)) return error.Malformed;
        if (!mem.eql(u8, bytes[0..2], "Ed")) return error.Malformed;
        return .{
            .key_id = bytes[2..10].*,
            .key = ed25519.PublicKey.fromBytes(bytes[10..42].*) catch return error.Malformed,
        };
    }
};

/// A signature from a `.minisig` file.
pub const Signature = struct {
    /// Which form the Ed25519 signature takes.
    pub const Kind = enum {
        /// `ED`: over the BLAKE2b-512 hash of the file, which is what a large
        /// download wants: it can be hashed while it streams.
        hash,
        /// `Ed`: over the bytes of the file.
        message,
    };

    kind: Kind,
    /// The signature line as it was decoded: 2 bytes of algorithm, 8 bytes of
    /// key id, and the 64-byte Ed25519 signature. The global signature is over
    /// the last of those, the signature itself, followed by the trusted
    /// comment.
    line: [2 + 8 + ed25519.Signature.encoded_length]u8,
    /// The Ed25519 signature in `line`.
    signature: ed25519.Signature,
    /// The text after "trusted comment: ".
    trusted_comment: []const u8,
    global_signature: ed25519.Signature,

    /// Parses a `.minisig` file, which borrows `text`.
    pub fn parse(text: []const u8) Error!Signature {
        var lines = mem.splitScalar(u8, text, '\n');
        const untrusted = mem.trim(u8, lines.next() orelse return error.Malformed, whitespace);
        if (!mem.startsWith(u8, untrusted, "untrusted comment: ")) return error.Malformed;
        const signature_text = mem.trim(u8, lines.next() orelse return error.Malformed, whitespace);
        const trusted = lines.next() orelse return error.Malformed;
        if (!mem.startsWith(u8, trusted, "trusted comment: ")) return error.Malformed;
        const global_text = mem.trim(u8, lines.next() orelse return error.Malformed, whitespace);

        var line: [2 + 8 + ed25519.Signature.encoded_length]u8 = undefined;
        if (!decodeExact(&line, signature_text)) return error.Malformed;
        const kind: Kind = if (mem.eql(u8, line[0..2], "ED"))
            .hash
        else if (mem.eql(u8, line[0..2], "Ed"))
            .message
        else
            return error.Malformed;

        var global: [ed25519.Signature.encoded_length]u8 = undefined;
        if (!decodeExact(&global, global_text)) return error.Malformed;

        return .{
            .kind = kind,
            .line = line,
            .signature = ed25519.Signature.fromBytes(line[10..74].*),
            .trusted_comment = mem.trimEnd(u8, trusted["trusted comment: ".len..], "\r"),
            .global_signature = ed25519.Signature.fromBytes(global),
        };
    }

    /// The id of the key that made this signature.
    pub fn keyId(signature: Signature) [8]u8 {
        return signature.line[2..10].*;
    }

    /// The Ed25519 signature, the way it is written in the signature line.
    pub fn signatureBytes(signature: Signature) *const [ed25519.Signature.encoded_length]u8 {
        return signature.line[10..74];
    }

    /// Checks that `public_key` made this signature over the file whose
    /// BLAKE2b-512 hash is `hash`, and that the trusted comment names
    /// `filename`. Only a signature in the hashed form ("ED") is over a hash;
    /// `verifyReader` is the way to check a signature over the file itself.
    pub fn verifyHash(
        signature: Signature,
        public_key: PublicKey,
        hash: [Blake2b512.digest_length]u8,
        filename: []const u8,
    ) Error!void {
        if (signature.kind != .hash) return error.Malformed;
        try signature.verifyTrusted(public_key, filename);
        signature.signature.verify(&hash, public_key.key) catch return error.InvalidSignature;
    }

    /// Checks a signature of either form over `bytes`, the whole file.
    pub fn verifyBytes(
        signature: Signature,
        public_key: PublicKey,
        bytes: []const u8,
        filename: []const u8,
    ) Error!void {
        try signature.verifyTrusted(public_key, filename);
        switch (signature.kind) {
            .hash => {
                var hash: [Blake2b512.digest_length]u8 = undefined;
                Blake2b512.hash(bytes, &hash, .{});
                signature.signature.verify(&hash, public_key.key) catch return error.InvalidSignature;
            },
            .message => signature.signature.verify(bytes, public_key.key) catch return error.InvalidSignature,
        }
    }

    /// `verifyBytes`, for a file that is read in chunks from `reader`: a
    /// signature over the file itself cannot be checked against a hash, so the
    /// file has to be read again.
    pub fn verifyReader(
        signature: Signature,
        public_key: PublicKey,
        reader: *std.Io.Reader,
        filename: []const u8,
    ) Error!void {
        try signature.verifyTrusted(public_key, filename);
        var buffer: [16 * 1024]u8 = undefined;
        switch (signature.kind) {
            .hash => {
                var hasher: Blake2b512 = .init(.{});
                while (true) {
                    const n = reader.readSliceShort(&buffer) catch return error.ReadFailed;
                    if (n == 0) break;
                    hasher.update(buffer[0..n]);
                }
                var hash: [Blake2b512.digest_length]u8 = undefined;
                hasher.final(&hash);
                signature.signature.verify(&hash, public_key.key) catch return error.InvalidSignature;
            },
            .message => {
                var verifier = signature.signature.verifier(public_key.key) catch
                    return error.InvalidSignature;
                while (true) {
                    const n = reader.readSliceShort(&buffer) catch return error.ReadFailed;
                    if (n == 0) break;
                    verifier.update(buffer[0..n]);
                }
                verifier.verify() catch return error.InvalidSignature;
            },
        }
    }

    /// The part of checking a signature that does not look at the file: the
    /// key id, the global signature over the signature line and the trusted
    /// comment, and the name of the file in the trusted comment.
    fn verifyTrusted(signature: Signature, public_key: PublicKey, filename: []const u8) Error!void {
        if (!mem.eql(u8, signature.line[2..10], &public_key.key_id)) return error.WrongKey;

        // The global signature is over the signature line and the trusted
        // comment, so neither of them can be replaced without it failing.
        var verifier = signature.global_signature.verifier(public_key.key) catch
            return error.InvalidGlobalSignature;
        verifier.update(signature.signatureBytes());
        verifier.update(signature.trusted_comment);
        verifier.verify() catch return error.InvalidGlobalSignature;

        // Upstream's trusted comments say `file:<file name>`, which says what
        // the signature is for no matter where the file was downloaded from.
        if (!namesFile(signature.trusted_comment, filename)) return error.WrongFile;
    }
};

const whitespace = " \t\r";

/// Decodes base64 into exactly `dest.len` bytes.
fn decodeExact(dest: []u8, text: []const u8) bool {
    const decoder = &std.base64.standard.Decoder;
    const size = decoder.calcSizeForSlice(text) catch return false;
    if (size != dest.len) return false;
    decoder.decode(dest, text) catch return false;
    return true;
}

/// Whether a trusted comment names `filename`, as upstream's do:
/// `timestamp:1755707121\tfile:zig-x86_64-linux-0.15.1.tar.xz\thashed`.
fn namesFile(trusted_comment: []const u8, filename: []const u8) bool {
    var tokens = mem.tokenizeAny(u8, trusted_comment, " \t");
    while (tokens.next()) |token| {
        if (mem.cutPrefix(u8, token, "file:")) |name| return mem.eql(u8, name, filename);
    }
    return false;
}

const testing = std.testing;

/// The BLAKE2b-512 hash of zig-x86_64-linux-0.15.1.tar.xz, as downloaded from
/// ziglang.org, which `zig_0_15_1_minisig` is a signature over.
const zig_0_15_1_blake2b512 =
    "4e14bfe9fce7416ac398bd361c45194beae60d0e258bc01600e7c6df21a71b3b" ++
    "43198c9de9a26f807b4cda404541d44b1fc2d5f3b66ea78ac3a66780b4f3ee16";

/// The signature that ziglang.org publishes as
/// `zig-x86_64-linux-0.15.1.tar.xz.minisig`, verbatim.
const zig_0_15_1_minisig =
    "untrusted comment: signature from minisign secret key\n" ++
    "RUSGOq2NVecA2Wk3Npw2IGzIC8fMec6YOFiKRo3F7tmaEgdVL8wk06amCA64FxoC8iN6+FyllUz+mQteoOXYlEYFkZYUBG2Bngo=\n" ++
    "trusted comment: timestamp:1755707121\tfile:zig-x86_64-linux-0.15.1.tar.xz\thashed\n" ++
    "ynslZhi+qM0o9VFWdEWj0/Wsn1zuW4ubHyI6a4AeJW2DdLq04d0TPuMMWgnFBKcxfjhFhBUmaqr156dIlIoFDA==\n";

const zig_0_15_1_filename = "zig-x86_64-linux-0.15.1.tar.xz";

test "the real signature of zig-x86_64-linux-0.15.1.tar.xz verifies" {
    const public_key = try PublicKey.fromBase64(zig_public_key);
    const signature = try Signature.parse(zig_0_15_1_minisig);
    try testing.expectEqual(Signature.Kind.hash, signature.kind);

    var hash: [Blake2b512.digest_length]u8 = undefined;
    _ = try std.fmt.hexToBytes(&hash, zig_0_15_1_blake2b512);
    try signature.verifyHash(public_key, hash, zig_0_15_1_filename);

    // A tampered file is not the file that was signed, down to one bit.
    var tampered_hash = hash;
    tampered_hash[7] ^= 1;
    try testing.expectError(
        error.InvalidSignature,
        signature.verifyHash(public_key, tampered_hash, zig_0_15_1_filename),
    );

    // A trusted comment that names another file is not accepted either,
    // whatever the signature is over.
    try testing.expectError(
        error.WrongFile,
        signature.verifyHash(public_key, hash, "zig-x86_64-linux-0.15.2.tar.xz"),
    );

    // Another key's signature is not this key's signature.
    const other_key: PublicKey = .{
        .key_id = @splat(0x42),
        .key = public_key.key,
    };
    try testing.expectError(error.WrongKey, signature.verifyHash(other_key, hash, zig_0_15_1_filename));
}

test "the trusted comment and the signature are signed together" {
    const public_key = try PublicKey.fromBase64(zig_public_key);

    // An edited trusted comment fails the global signature, and so does an
    // edited signature line: neither can be replaced on its own.
    const Edit = struct { from: []const u8, to: []const u8 };
    for ([_]Edit{
        .{ .from = "timestamp:1755707121", .to = "timestamp:1755707122" },
        .{ .from = zig_0_15_1_filename, .to = "zig-x86_64-linux-0.15.0.tar.xz" },
        .{ .from = "RUSGOq2NVecA2Wk3", .to = "RUSGOq2NVecA2Wk2" },
    }) |edit| {
        var text = try testing.allocator.dupe(u8, zig_0_15_1_minisig);
        defer testing.allocator.free(text);
        const at = mem.indexOf(u8, text, edit.from).?;
        @memcpy(text[at..][0..edit.to.len], edit.to);

        // The text still parses: it is the signatures that say no.
        const signature = try Signature.parse(text);
        var hash: [Blake2b512.digest_length]u8 = undefined;
        _ = try std.fmt.hexToBytes(&hash, zig_0_15_1_blake2b512);
        try testing.expectError(
            error.InvalidGlobalSignature,
            signature.verifyHash(public_key, hash, zig_0_15_1_filename),
        );
    }
}

test "a wrong key id is caught before anything else" {
    // The public key of another minisign key, so that the key id differs.
    const other_key_base64 = "RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3";
    const other_key = try PublicKey.fromBase64(other_key_base64);
    const signature = try Signature.parse(zig_0_15_1_minisig);
    try testing.expect(!mem.eql(u8, &other_key.key_id, &signature.keyId()));

    var hash: [Blake2b512.digest_length]u8 = undefined;
    _ = try std.fmt.hexToBytes(&hash, zig_0_15_1_blake2b512);
    try testing.expectError(error.WrongKey, signature.verifyHash(other_key, hash, zig_0_15_1_filename));
}

test "minisign files that are not minisign files are rejected" {
    // The four lines, one at a time.
    try testing.expectError(error.Malformed, Signature.parse(""));
    try testing.expectError(error.Malformed, Signature.parse("untrusted comment: x\n"));
    try testing.expectError(error.Malformed, Signature.parse(
        "untrusted comment: x\nRUSGOq2NVecA2Wk3\n",
    ));
    try testing.expectError(error.Malformed, Signature.parse(
        "untrusted comment: x\nRUSGOq2NVecA2Wk3\ntrusted comment: x\n",
    ));
    // Lines in the wrong order, or with the wrong prefix.
    try testing.expectError(error.Malformed, Signature.parse(
        "RUSGOq2NVecA2Wk3\ntrusted comment: x\nRUSGOq2NVecA2Wk3\ny\n",
    ));
    try testing.expectError(error.Malformed, Signature.parse(
        "untrusted comment: x\nRUSGOq2NVecA2Wk3\ntrusted x\nRUSGOq2NVecA2Wk3\n",
    ));
    // Base64 that is not base64, or not the length of a signature.
    try testing.expectError(error.Malformed, Signature.parse(
        "untrusted comment: x\nnot base64! (really)\ntrusted comment: x\nAAAA\n",
    ));
    try testing.expectError(error.Malformed, Signature.parse(
        "untrusted comment: x\nAAAA\ntrusted comment: x\nAAAA\n",
    ));
    // The real signature line, with the algorithm bytes replaced.
    var text = try testing.allocator.dupe(u8, zig_0_15_1_minisig);
    defer testing.allocator.free(text);
    @memcpy(text[mem.indexOf(u8, text, "RUSG").?..][0..4], "AAAA");
    try testing.expectError(error.Malformed, Signature.parse(text));
}

test "a public key that is not a minisign key is rejected" {
    // Not base64.
    try testing.expectError(error.Malformed, PublicKey.fromBase64("not a key"));
    // Base64, but not 42 bytes.
    try testing.expectError(error.Malformed, PublicKey.fromBase64("AAAA"));
    // 42 bytes of base64 with the wrong algorithm.
    var wrong_algorithm: [42]u8 = @splat(0);
    wrong_algorithm[0..2].* = "ZZ".*;
    var encoded: [56]u8 = undefined;
    try testing.expectError(
        error.Malformed,
        PublicKey.fromBase64(std.base64.standard.Encoder.encode(&encoded, &wrong_algorithm)),
    );
}

/// A key to sign test vectors with: the real minisign algorithm, with key
/// material of our own, so that the tests never depend on anything but the
/// bytes above.
const TestKey = struct {
    key_pair: ed25519.KeyPair,
    key_id: [8]u8,

    fn init(seed: [ed25519.KeyPair.seed_length]u8) !TestKey {
        const key_pair = try ed25519.KeyPair.generateDeterministic(seed);
        const public_bytes = key_pair.public_key.toBytes();
        var id_hash: [Blake2b512.digest_length]u8 = undefined;
        Blake2b512.hash(&public_bytes, &id_hash, .{});
        return .{ .key_pair = key_pair, .key_id = id_hash[0..8].* };
    }

    fn publicKey(key: TestKey) PublicKey {
        return .{ .key_id = key.key_id, .key = key.key_pair.public_key };
    }

    /// The key as minisign writes it: base64 of "Ed", the key id, and the
    /// Ed25519 public key.
    fn publicKeyBase64(key: TestKey, out: *[56]u8) []const u8 {
        var bytes: [2 + 8 + ed25519.PublicKey.encoded_length]u8 = undefined;
        bytes[0..2].* = "Ed".*;
        bytes[2..10].* = key.key_id;
        bytes[10..42].* = key.key_pair.public_key.toBytes();
        return std.base64.standard.Encoder.encode(out, &bytes);
    }

    /// Signs `bytes` the way minisign does, and returns the `.minisig` file
    /// for it: an untrusted comment line, the signature line, the trusted
    /// comment line, and the global signature over the signature and the
    /// trusted comment.
    fn minisig(
        key: TestKey,
        arena: Allocator,
        bytes: []const u8,
        kind: Signature.Kind,
        trusted_comment: []const u8,
    ) ![]u8 {
        var digest: [Blake2b512.digest_length]u8 = undefined;
        var line: [2 + 8 + ed25519.Signature.encoded_length]u8 = undefined;
        line[0..2].* = if (kind == .hash) "ED".* else "Ed".*;
        line[2..10].* = key.key_id;
        line[10..74].* = (try key.key_pair.sign(switch (kind) {
            .hash => blk: {
                Blake2b512.hash(bytes, &digest, .{});
                break :blk digest[0..];
            },
            .message => bytes,
        }, null)).toBytes();

        const global_bytes = (try key.key_pair.sign(
            try mem.concat(arena, u8, &.{ line[10..], trusted_comment }),
            null,
        )).toBytes();

        const encoder = std.base64.standard.Encoder;
        return std.fmt.allocPrint(
            arena,
            "untrusted comment: signature from minisign secret key\n{s}\ntrusted comment: {s}\n{s}\n",
            .{
                encoder.encode(try arena.alloc(u8, encoder.calcSize(line.len)), &line),
                trusted_comment,
                encoder.encode(try arena.alloc(u8, encoder.calcSize(global_bytes.len)), &global_bytes),
            },
        );
    }
};

test "a signature of our own, in both forms, verifies" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const key = try TestKey.init(@splat(0x42));
    const filename = "zig-x86_64-linux-0.16.0-dev.1+abcdef012.tar.xz";
    const bytes = "the whole file, in memory, which a test file can be";

    // The key as minisign writes it parses back to the same key.
    var key_base64_buffer: [56]u8 = undefined;
    const key_base64 = key.publicKeyBase64(&key_base64_buffer);
    const parsed_key = try PublicKey.fromBase64(key_base64);
    try testing.expectEqualSlices(u8, &key.key_id, &parsed_key.key_id);
    const expected_public_bytes = key.key_pair.public_key.toBytes();
    const actual_public_bytes = parsed_key.key.toBytes();
    try testing.expectEqualSlices(u8, &expected_public_bytes, &actual_public_bytes);

    for ([_]Signature.Kind{ .hash, .message }) |kind| {
        const text = try key.minisig(arena, bytes, kind, "timestamp:1\tfile:" ++ filename ++ "\thashed");
        const signature = try Signature.parse(text);
        try testing.expectEqual(kind, signature.kind);

        var hash: [Blake2b512.digest_length]u8 = undefined;
        Blake2b512.hash(bytes, &hash, .{});
        try signature.verifyBytes(parsed_key, bytes, filename);

        var reader = std.Io.Reader.fixed(bytes);
        try signature.verifyReader(parsed_key, &reader, filename);

        // A signature over a hash is checked against the hash; a signature
        // over the file cannot be, and says so instead of pretending.
        switch (kind) {
            .hash => try signature.verifyHash(parsed_key, hash, filename),
            .message => try testing.expectError(error.Malformed, signature.verifyHash(parsed_key, hash, filename)),
        }

        // A tampered file is rejected, in both forms.
        var tampered_buffer: [64]u8 = undefined;
        @memcpy(tampered_buffer[0..bytes.len], bytes);
        tampered_buffer[bytes.len / 2] ^= 1;
        const tampered = tampered_buffer[0..bytes.len];
        try testing.expectError(error.InvalidSignature, signature.verifyBytes(parsed_key, tampered, filename));
        var tampered_reader = std.Io.Reader.fixed(tampered);
        try testing.expectError(error.InvalidSignature, signature.verifyReader(parsed_key, &tampered_reader, filename));
        if (kind == .hash) {
            Blake2b512.hash(tampered, &hash, .{});
            try testing.expectError(error.InvalidSignature, signature.verifyHash(parsed_key, hash, filename));
        }

        // A trusted comment that names another file is rejected.
        const other = try key.minisig(arena, bytes, kind, "timestamp:1\tfile:another.tar.xz\thashed");
        try testing.expectError(
            error.WrongFile,
            (try Signature.parse(other)).verifyBytes(parsed_key, bytes, filename),
        );
    }
}
