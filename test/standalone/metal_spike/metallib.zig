//! metallib: write and inspect Apple `.metallib` Metal library containers.
//!
//! Container format. Verified field by field against Apple's own `xcrun metallib`
//! output `/tmp/metal-spike/vadd.metallib`, the Apple-produced libraries
//! `/tmp/metal-spike/ref_default.metallib` and `metaljl/test/metallib/*.metallib`,
//! and against the writer in Metal.jl (`/tmp/metal-spike/metaljl/src/compiler/library.jl`),
//! which the layout below follows:
//!
//!   * header, 88 bytes, every integer little-endian:
//!       u32 magic "MTLB"
//!       u16 file_major      bit 15 set = macOS target               (library.jl:956-958)
//!       u16 file_minor
//!       u16 file_patch
//!       u8  file_type       bit 7 set = stub; 0 = FILE_EXECUTABLE   (library.jl:960-966)
//!       u8  platform_type   bit 7 set = 64-bit; 1 = PLATFORM_MACOS  (library.jl:968-975)
//!       u16 platform_major
//!       u8  platform_minor
//!       u8  platform_patch
//!       u64 file_size
//!       four (u64 offset, u64 size) pairs: function list, public metadata,
//!       private metadata, module list                              (library.jl:1004-1024)
//!   * function list at offset 88: u32 function_count, then one tag group per
//!     function, in function order. A group is `u32 group_size` -- which counts
//!     itself, the tag records and the trailing ENDT -- then records
//!     `<4-byte tag><u16 value_size><value>`, then the 4 ASCII bytes "ENDT".
//!     `RBUF`/`SBUF` use a u32 `value_size` (library.jl:208-224,427-466,484-527).
//!     The header's `function_list_size` excludes the leading u32 function_count
//!     (library.jl:1007-1011), so for a single kernel it equals the group size.
//!     Our function tags, in the order Xcode's own 1.2.5-1.2.9 fixtures use
//!     (e.g. metaljl/test/metallib/kernel.26.metallib):
//!       NAME  NUL-terminated kernel name, the NUL is part of the length
//!       TYPE  u8, 2 = PROGRAM_KERNEL                              (library.jl:59-68)
//!       HASH  SHA-256 over the stored module bytes
//!       OFFT  3 x u64: public, private and module offsets, relative to their sections
//!       VERS  4 x u16: air major, air minor, metal major, metal minor
//!       MDSZ  u64 module byte count                               (library.jl:891-920)
//!   * header extension between the function list and the public metadata: no size
//!     prefix, optional tags, always terminated by ENDT (library.jl:929-948,1044-1046).
//!     We write only `UUID` (u16 length 16, then two LE u64: high word, low word;
//!     library.jl:318-354), and only with --uuid; without it the extension is a bare
//!     ENDT. Section-pointer tags (HSRC/HSRD/HDYN/RLST) are printed when reading but
//!     never written: we have no embedded source, dynamic header or reflection data.
//!   * public and private metadata: one tag group per function, in function order.
//!     With a single kernel each is one empty group, written as `u32 8` + ENDT --
//!     the size counts itself and the ENDT exactly like every other group
//!     (library.jl:873-886), which is what Apple's Xcode fixtures contain.
//!     `metallib dump` also accepts the one variant found in Apple's vadd.metallib,
//!     where an empty group declares size 4, i.e. omits its ENDT from the count, and
//!     reports that as a note rather than a mismatch.
//!   * modules: the module bytes follow with no wrapper and no alignment,
//!     concatenated (library.jl:891-894,1054-1056). MDSZ and HASH are the byte count
//!     and hash of exactly those stored bytes, so nothing is transformed here.
//!     Store raw (unwrapped) bitcode: a module in Apple's 0x0b17c0de wrapper was
//!     reproducibly rejected by `newComputePipelineStateWithFunction` with
//!     XPC_ERROR_CONNECTION_INTERRUPTED on macOS 26.6.2, while the same module
//!     stored raw loaded and ran. `write` warns when the input looks wrapped.
//!   * UUID derivation: the first 16 bytes of SHA-256 over the module bytes,
//!     interpreted big-endian as u128, with the RFC 4122 version nibble (bits 76-79)
//!     stamped to 4 and the variant bits (bits 62-63) to binary 10 -- exactly
//!     `content_uuid` in library.jl:99-110.
//!   * sections are contiguous, in the order function list, header extension, public
//!     metadata, private metadata, module bytes; `file_size` is the total
//!     (library.jl:1027-1082).
//!
//! Usage:
//!   metallib write --bitcode <file> --name <kernel> [--name <kernel> ...]
//!                  [--air M.m] [--metal M.m] [--format M.m.p] [--platform M.m.p]
//!                  [--uuid] -o <out.metallib>
//!   metallib dump <file.metallib>
//!   metallib selftest [--dir <dir>] [--bitcode <file>] [ref.metallib ...]
//!
//! `--name` is repeatable: every name becomes its own function group, in the order
//! given, and all of them refer to the same module bytes (module offset 0, one
//! MDSZ, one HASH), because our kernels live in one LLVM module. Each group also
//! owns its own empty public/private metadata group, so those sections hold one
//! 8-byte group per function -- the shape Apple's ref_default.metallib uses for
//! its two functions. Verified on an M4 (macOS 26.6.2): a library written this way
//! with vadd/reduce/parsef resolved all three names with `newFunctionWithName:`
//! while an unknown name returned nil, so the runtime's lookup follows these NAME
//! tags and not the module's !air.kernel metadata.
//!
//! Build:
//!   ZIG_LIB_DIR=<lib> zig build-exe test/standalone/metal_spike/metallib.zig \
//!       -femit-bin=<out>/metallib

const std = @import("std");
const Io = std.Io;
const mem = std.mem;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const magic = "MTLB";
const header_size: usize = 88;
const kernel_type: u8 = 2; // PROGRAM_KERNEL, library.jl:60
const function_count_size: u64 = 4;
/// Sanity bound when reading a function count out of an untrusted file.
const max_functions: u32 = 65536;
/// Apple's bitcode wrapper: magic, u32 0, u32 0x14, u32 size, i32 -1.
const bitcode_wrapper_magic: u32 = 0x0b17c0de;

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

/// Collects notes and problems while parsing, so `dump` can report everything
/// that is wrong with a file instead of only the first error.
const Report = struct {
    notes: Io.Writer.Allocating,
    problems: Io.Writer.Allocating,
    oom: bool = false,

    fn init(gpa: Allocator) Report {
        return .{ .notes = .init(gpa), .problems = .init(gpa) };
    }

    fn deinit(r: *Report) void {
        r.notes.deinit();
        r.problems.deinit();
    }

    fn note(r: *Report, comptime fmt: []const u8, args: anytype) void {
        r.notes.writer.print(fmt ++ "\n", args) catch {
            r.oom = true;
        };
    }

    fn problem(r: *Report, comptime fmt: []const u8, args: anytype) void {
        r.problems.writer.print(fmt ++ "\n", args) catch {
            r.oom = true;
        };
    }

    fn hasProblems(r: *Report) bool {
        return r.problems.written().len != 0;
    }
};

fn printPrefixed(out: *Io.Writer, label: []const u8, text: []const u8) !void {
    var lines = mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        try out.print("{s}: {s}\n", .{ label, line });
    }
}

fn countLines(text: []const u8) usize {
    var lines = mem.splitScalar(u8, text, '\n');
    var n: usize = 0;
    while (lines.next()) |line| {
        if (line.len != 0) n += 1;
    }
    return n;
}

fn printHexBytes(out: *Io.Writer, bytes: []const u8, max: usize) !void {
    const n = @min(bytes.len, max);
    for (bytes[0..n]) |b| try out.print("{x:0>2}", .{b});
    if (n < bytes.len) try out.print("...", .{});
}

// ---------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------

const ParseError = error{ Mismatch, OutOfMemory };

/// Bounds-checked little-endian cursor over the file image.
const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(c: *Cursor, rep: *Report, comptime what: []const u8, n: usize) ParseError![]const u8 {
        if (n > c.bytes.len - c.pos) {
            rep.problem(
                "truncated input: {s} at offset {d} needs {d} byte(s), {d} remain",
                .{ what, c.pos, n, c.bytes.len - c.pos },
            );
            return error.Mismatch;
        }
        const result = c.bytes[c.pos..][0..n];
        c.pos += n;
        return result;
    }

    fn readU8(c: *Cursor, rep: *Report, comptime what: []const u8) ParseError!u8 {
        const b = try c.take(rep, what, 1);
        return b[0];
    }

    fn readU16(c: *Cursor, rep: *Report, comptime what: []const u8) ParseError!u16 {
        const b = try c.take(rep, what, 2);
        return mem.readInt(u16, b[0..2], .little);
    }

    fn readU32(c: *Cursor, rep: *Report, comptime what: []const u8) ParseError!u32 {
        const b = try c.take(rep, what, 4);
        return mem.readInt(u32, b[0..4], .little);
    }

    fn readU64(c: *Cursor, rep: *Report, comptime what: []const u8) ParseError!u64 {
        const b = try c.take(rep, what, 8);
        return mem.readInt(u64, b[0..8], .little);
    }

    fn section(c: *Cursor, rep: *Report, comptime what: []const u8) ParseError!Section {
        return .{
            .offset = try c.readU64(rep, what ++ " offset"),
            .size = try c.readU64(rep, what ++ " size"),
        };
    }
};

const Version = struct {
    major: u16,
    minor: u16,
    patch: u16 = 0,
};

const Section = struct {
    offset: u64,
    size: u64,
};

/// A [start, end) byte range inside the module list.
const Range = struct {
    start: u64,
    end: u64,

    fn lessThan(_: void, a: Range, b: Range) bool {
        return a.start < b.start;
    }
};

const Tag = struct {
    name: [4]u8,
    value: []const u8,
};

const Group = struct {
    /// The size field as stored; zero when the group has no size field at all,
    /// which is the case for the header extension.
    declared_size: u32,
    /// Offset of the size field (or of the first tag for the header extension).
    start: usize,
    /// Offset just past the terminating ENDT.
    end: usize,
    tags: []const Tag,
};

const Function = struct {
    name: []const u8,
    ty: u8,
    hash: [32]u8,
    offt_public_md: u64,
    offt_private_md: u64,
    offt_air_module: u64,
    vers: [4]u16,
    mdsz: u64,
    /// Offset of this function's reflection buffer inside the reflection list.
    rflt: ?u64,
    group_size: u32,
    pub_md_size: u64 = 0,
    priv_md_size: u64 = 0,
    tags: []const Tag,
};

const Module = struct {
    version: Version,
    is_macos: bool,
    file_type: u8,
    is_stub: bool,
    platform_type: u8,
    is_64bit: bool,
    platform_version: Version,
    file_size: u64,
    function_list: Section,
    public_md: Section,
    private_md: Section,
    module_list: Section,
    functions: []const Function,
    ext_tags: []const Tag,
    ext_size: u64,
    uuid: ?u128,
};

fn isWideTag(name: *const [4]u8) bool {
    return mem.eql(u8, name, "RBUF") or mem.eql(u8, name, "SBUF");
}

fn isSectionPointerTag(name: *const [4]u8) bool {
    return mem.eql(u8, name, "HSRD") or mem.eql(u8, name, "HSRC") or
        mem.eql(u8, name, "HDYN") or mem.eql(u8, name, "RLST");
}

fn isStandardFunctionTag(name: *const [4]u8) bool {
    const standard = [_][4]u8{ "NAME".*, "TYPE".*, "HASH".*, "OFFT".*, "VERS".*, "MDSZ".* };
    for (standard) |s| {
        if (mem.eql(u8, name, &s)) return true;
    }
    return false;
}

fn versionAtLeast(v: Version, major: u16, minor: u16, patch: u16) bool {
    if (v.major != major) return v.major > major;
    if (v.minor != minor) return v.minor > minor;
    return v.patch >= patch;
}

fn sectionInBounds(sec: Section, file_len: usize) bool {
    return sec.offset <= @as(u64, file_len) and sec.size <= @as(u64, file_len) - sec.offset;
}

/// A tag group with a leading u32 size: records until ENDT.
fn parseSizedGroup(
    arena: Allocator,
    c: *Cursor,
    rep: *Report,
    comptime what: []const u8,
    index: u32,
) ParseError!Group {
    const start = c.pos;
    const declared = try c.readU32(rep, what ++ " group size");
    var tags: std.ArrayList(Tag) = .empty;
    while (true) {
        const name_bytes = try c.take(rep, what ++ " tag name", 4);
        if (mem.eql(u8, name_bytes, "ENDT")) break;
        var name: [4]u8 = undefined;
        @memcpy(&name, name_bytes);
        const value_size: usize = if (isWideTag(&name))
            try c.readU32(rep, what ++ " tag value size")
        else
            try c.readU16(rep, what ++ " tag value size");
        const value = try c.take(rep, what ++ " tag value", value_size);
        try tags.append(arena, .{ .name = name, .value = value });
    }
    const consumed = c.pos - start;
    if (consumed != declared) {
        if (tags.items.len == 0 and @as(usize, declared) + 4 == consumed) {
            rep.note(
                "{s} group for function {d} at offset {d} declares size {d}, which omits its ENDT " ++
                    "(the variant in Apple's vadd.metallib; Xcode's other fixtures declare 8)",
                .{ what, index, start, declared },
            );
        } else {
            rep.problem(
                "{s} group for function {d} at offset {d} declares size {d} but its tags and ENDT occupy {d} byte(s)",
                .{ what, index, start, declared, consumed },
            );
        }
    }
    return .{
        .declared_size = declared,
        .start = start,
        .end = c.pos,
        .tags = try tags.toOwnedSlice(arena),
    };
}

/// The header extension: records until ENDT, with no size field (library.jl:929-948).
fn parseExtensionGroup(arena: Allocator, c: *Cursor, rep: *Report) ParseError!Group {
    const start = c.pos;
    var tags: std.ArrayList(Tag) = .empty;
    while (true) {
        const name_bytes = try c.take(rep, "header extension tag name", 4);
        if (mem.eql(u8, name_bytes, "ENDT")) break;
        var name: [4]u8 = undefined;
        @memcpy(&name, name_bytes);
        const value_size: usize = if (isWideTag(&name))
            try c.readU32(rep, "header extension tag value size")
        else
            try c.readU16(rep, "header extension tag value size");
        const value = try c.take(rep, "header extension tag value", value_size);
        try tags.append(arena, .{ .name = name, .value = value });
    }
    return .{
        .declared_size = 0,
        .start = start,
        .end = c.pos,
        .tags = try tags.toOwnedSlice(arena),
    };
}

fn parseFunction(arena: Allocator, c: *Cursor, rep: *Report, index: u32) ParseError!Function {
    const group = try parseSizedGroup(arena, c, rep, "function", index);
    var f = Function{
        .name = "<missing NAME tag>",
        .ty = 0,
        .hash = @splat(0),
        .offt_public_md = 0,
        .offt_private_md = 0,
        .offt_air_module = 0,
        .vers = .{ 0, 0, 0, 0 },
        .mdsz = 0,
        .rflt = null,
        .group_size = group.declared_size,
        .tags = group.tags,
    };
    var have_name = false;
    var have_mdsz = false;
    for (group.tags) |t| {
        if (mem.eql(u8, &t.name, "NAME")) {
            const nul = mem.indexOfScalar(u8, t.value, 0);
            f.name = try arena.dupe(u8, if (nul) |n| t.value[0..n] else t.value);
            have_name = true;
        } else if (mem.eql(u8, &t.name, "TYPE")) {
            if (t.value.len == 1) {
                f.ty = t.value[0];
            } else {
                rep.problem("function {d}: TYPE value is {d} byte(s), expected 1", .{ index, t.value.len });
            }
        } else if (mem.eql(u8, &t.name, "HASH")) {
            if (t.value.len == 32) {
                @memcpy(&f.hash, t.value);
            } else {
                rep.problem("function {d}: HASH value is {d} byte(s), expected 32", .{ index, t.value.len });
            }
        } else if (mem.eql(u8, &t.name, "OFFT")) {
            if (t.value.len == 24) {
                f.offt_public_md = mem.readInt(u64, t.value[0..8], .little);
                f.offt_private_md = mem.readInt(u64, t.value[8..16], .little);
                f.offt_air_module = mem.readInt(u64, t.value[16..24], .little);
            } else {
                rep.problem("function {d}: OFFT value is {d} byte(s), expected 24", .{ index, t.value.len });
            }
        } else if (mem.eql(u8, &t.name, "VERS")) {
            if (t.value.len == 8) {
                f.vers = .{
                    mem.readInt(u16, t.value[0..2], .little),
                    mem.readInt(u16, t.value[2..4], .little),
                    mem.readInt(u16, t.value[4..6], .little),
                    mem.readInt(u16, t.value[6..8], .little),
                };
            } else {
                rep.problem("function {d}: VERS value is {d} byte(s), expected 8", .{ index, t.value.len });
            }
        } else if (mem.eql(u8, &t.name, "MDSZ")) {
            if (t.value.len == 8) {
                f.mdsz = mem.readInt(u64, t.value[0..8], .little);
                have_mdsz = true;
            } else {
                rep.problem("function {d}: MDSZ value is {d} byte(s), expected 8", .{ index, t.value.len });
            }
        } else if (mem.eql(u8, &t.name, "RFLT")) {
            if (t.value.len == 8) {
                f.rflt = mem.readInt(u64, t.value[0..8], .little);
            } else {
                rep.problem("function {d}: RFLT value is {d} byte(s), expected 8", .{ index, t.value.len });
            }
        }
    }
    if (!have_name) rep.problem("function {d}: no NAME tag in the group at offset {d}", .{ index, group.start });
    if (!have_mdsz) rep.problem("function {d} ('{s}'): no MDSZ tag", .{ index, f.name });
    return f;
}

/// One tag group per function, in function order, filling the section exactly.
fn parseMetadataSection(
    arena: Allocator,
    bytes: []const u8,
    rep: *Report,
    comptime what: []const u8,
    sec: Section,
    functions: []Function,
) ParseError!void {
    if (!sectionInBounds(sec, bytes.len)) {
        rep.problem(
            "{s} section (offset {d}, size {d}) lies outside the {d}-byte input",
            .{ what, sec.offset, sec.size, bytes.len },
        );
        return error.Mismatch;
    }
    var c = Cursor{ .bytes = bytes, .pos = @intCast(sec.offset) };
    for (functions, 0..) |*f, i| {
        const group_start: u64 = c.pos;
        _ = try parseSizedGroup(arena, &c, rep, what, @intCast(i));
        const group_size: u64 = c.pos - @as(usize, @intCast(group_start));
        const offt = if (comptime mem.eql(u8, what, "public metadata")) f.offt_public_md else f.offt_private_md;
        if (offt != group_start - sec.offset) {
            rep.problem(
                "function '{s}': OFFT {s} offset is {d} but its group starts at {d} in the section",
                .{ f.name, what, offt, group_start - sec.offset },
            );
        }
        if (comptime mem.eql(u8, what, "public metadata")) {
            f.pub_md_size = group_size;
        } else {
            f.priv_md_size = group_size;
        }
    }
    if (@as(u64, c.pos) - sec.offset != sec.size) {
        rep.problem(
            "{s} section declares {d} byte(s) but its {d} group(s) occupy {d}",
            .{ what, sec.size, functions.len, @as(u64, c.pos) - sec.offset },
        );
    }
}

fn parseModule(arena: Allocator, bytes: []const u8, rep: *Report) ParseError!Module {
    if (bytes.len < header_size) {
        rep.problem(
            "input is {d} byte(s), shorter than the {d}-byte MTLB header",
            .{ bytes.len, header_size },
        );
        return error.Mismatch;
    }
    if (!mem.eql(u8, bytes[0..4], magic)) {
        rep.problem(
            "not a metallib: magic is {x:0>2}{x:0>2}{x:0>2}{x:0>2}, expected 'MTLB'",
            .{ bytes[0], bytes[1], bytes[2], bytes[3] },
        );
        return error.Mismatch;
    }

    var c = Cursor{ .bytes = bytes, .pos = 4 };
    var m: Module = undefined;

    const major_word = try c.readU16(rep, "file version major");
    m.version = .{
        .major = major_word & 0x7fff,
        .minor = try c.readU16(rep, "file version minor"),
        .patch = try c.readU16(rep, "file version patch"),
    };
    m.is_macos = major_word & 0x8000 != 0;

    const file_type = try c.readU8(rep, "file type");
    m.file_type = file_type & 0x7f;
    m.is_stub = file_type & 0x80 != 0;

    const platform_type = try c.readU8(rep, "platform type");
    m.platform_type = platform_type & 0x7f;
    m.is_64bit = platform_type & 0x80 != 0;

    m.platform_version = .{
        .major = try c.readU16(rep, "platform version major"),
        .minor = try c.readU8(rep, "platform version minor"),
        .patch = try c.readU8(rep, "platform version patch"),
    };

    m.file_size = try c.readU64(rep, "file size");
    m.function_list = try c.section(rep, "function list");
    m.public_md = try c.section(rep, "public metadata");
    m.private_md = try c.section(rep, "private metadata");
    m.module_list = try c.section(rep, "module list");

    if (m.file_size != @as(u64, bytes.len)) {
        rep.problem("file_size field is {d} but the input is {d} byte(s)", .{ m.file_size, bytes.len });
    }
    if (!sectionInBounds(m.function_list, bytes.len)) {
        rep.problem(
            "function list section (offset {d}, size {d}) lies outside the {d}-byte input",
            .{ m.function_list.offset, m.function_list.size, bytes.len },
        );
        return error.Mismatch;
    }

    // ------------------------------------------------------------ function list
    c.pos = @intCast(m.function_list.offset);
    const count = try c.readU32(rep, "function count");
    if (count > max_functions) {
        rep.problem("function count {d} exceeds the sanity limit of {d}", .{ count, max_functions });
        return error.Mismatch;
    }
    var functions: std.ArrayList(Function) = .empty;
    try functions.ensureTotalCapacity(arena, count);
    var index: u32 = 0;
    while (index < count) : (index += 1) {
        functions.appendAssumeCapacity(try parseFunction(arena, &c, rep, index));
    }
    const list_end = m.function_list.offset + function_count_size + m.function_list.size;
    if (@as(u64, c.pos) != list_end) {
        rep.problem(
            "function list ends at offset {d} but the header says it ends at {d}",
            .{ c.pos, list_end },
        );
    }
    // --------------------------------------------------------- header extension
    var ext = Group{ .declared_size = 0, .start = c.pos, .end = c.pos, .tags = &.{} };
    if (versionAtLeast(m.version, 1, 2, 3)) {
        ext = try parseExtensionGroup(arena, &c, rep);
        if (@as(u64, c.pos) != m.public_md.offset) {
            rep.problem(
                "header extension ends at offset {d} but the public metadata section starts at {d}",
                .{ c.pos, m.public_md.offset },
            );
        }
    } else if (@as(u64, c.pos) != m.public_md.offset) {
        rep.problem(
            "function list ends at offset {d} but the public metadata section starts at {d} " ++
                "(no header extension before file version 1.2.3)",
            .{ c.pos, m.public_md.offset },
        );
    }
    m.ext_tags = ext.tags;
    m.ext_size = ext.end - ext.start;

    // ----------------------------------------------------------------- metadata
    try parseMetadataSection(arena, bytes, rep, "public metadata", m.public_md, functions.items);
    try parseMetadataSection(arena, bytes, rep, "private metadata", m.private_md, functions.items);
    m.functions = try functions.toOwnedSlice(arena);

    // ----------------------------------------------------------- modules, hashes
    if (!sectionInBounds(m.module_list, bytes.len)) {
        rep.problem(
            "module list section (offset {d}, size {d}) lies outside the {d}-byte input",
            .{ m.module_list.offset, m.module_list.size, bytes.len },
        );
        return error.Mismatch;
    }
    var ranges: std.ArrayList(Range) = .empty;
    try ranges.ensureTotalCapacity(arena, m.functions.len);
    for (m.functions) |f| {
        if (f.offt_air_module > m.module_list.size or
            f.mdsz > m.module_list.size - f.offt_air_module)
        {
            rep.problem(
                "function '{s}': module range [{d}, {d}) lies outside the module list section of {d} byte(s)",
                .{ f.name, f.offt_air_module, f.offt_air_module + f.mdsz, m.module_list.size },
            );
            continue;
        }
        ranges.appendAssumeCapacity(.{
            .start = f.offt_air_module,
            .end = f.offt_air_module + f.mdsz,
        });
        const start: usize = @intCast(m.module_list.offset + f.offt_air_module);
        const module_bytes = bytes[start..][0..@intCast(f.mdsz)];
        var digest: [32]u8 = undefined;
        Sha256.hash(module_bytes, &digest, .{});
        if (!mem.eql(u8, &digest, &f.hash)) {
            rep.problem("function '{s}': HASH does not match SHA-256 over its module bytes", .{f.name});
        }
    }
    // The union of the function module ranges must cover the module list exactly.
    // Ranges are allowed to overlap: several function groups may refer to one
    // module (a library naming the kernels of a single LLVM module does exactly
    // that), so the sum of the MDSZ values may exceed the section size.
    const sorted = try arena.alloc(Range, ranges.items.len);
    @memcpy(sorted, ranges.items);
    std.mem.sortUnstable(Range, sorted, {}, Range.lessThan);
    var covered: u64 = 0;
    for (sorted) |r| {
        if (r.start > covered) {
            rep.problem(
                "module list byte(s) {d}..{d} are not covered by any function's module range",
                .{ covered, r.start },
            );
        }
        covered = @max(covered, r.end);
    }
    if (covered < m.module_list.size) {
        rep.problem(
            "module list byte(s) {d}..{d} are not covered by any function's module range",
            .{ covered, m.module_list.size },
        );
    }

    // --------------------------------------------------------------------- uuid
    m.uuid = null;
    for (m.ext_tags) |t| {
        if (mem.eql(u8, &t.name, "UUID")) {
            if (decodeUuid(t.value)) |id| {
                m.uuid = id;
            } else {
                rep.problem("UUID tag holds {d} byte(s), expected 16", .{t.value.len});
            }
        }
    }

    if (rep.oom) return error.OutOfMemory;
    if (rep.hasProblems()) return error.Mismatch;
    return m;
}

fn decodeUuid(value: []const u8) ?u128 {
    if (value.len != 16) return null;
    const high = mem.readInt(u64, value[0..8], .little);
    const low = mem.readInt(u64, value[8..16], .little);
    return (@as(u128, high) << 64) | low;
}

fn contentUuid(digest: [32]u8) u128 {
    var id = mem.readInt(u128, digest[0..16], .big);
    id = (id & ~(@as(u128, 0xf) << 76)) | (@as(u128, 0x4) << 76);
    id = (id & ~(@as(u128, 0x3) << 62)) | (@as(u128, 0x2) << 62);
    return id;
}

fn printUuid(out: *Io.Writer, id: u128) !void {
    const b = mem.toBytes(mem.nativeToBig(u128, id));
    try out.print(
        "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-" ++
            "{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}",
        .{
            b[0], b[1], b[2],  b[3],  b[4],  b[5],  b[6],  b[7],
            b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
        },
    );
}

fn fileTypeName(t: u8) []const u8 {
    return switch (t) {
        0 => "executable",
        1 => "core image",
        2 => "dynamic",
        3 => "symbol companion",
        else => "unknown",
    };
}

fn programTypeName(t: u8) []const u8 {
    return switch (t) {
        0 => "vertex",
        1 => "fragment",
        2 => "kernel",
        3 => "unqualified",
        4 => "visible",
        5 => "extern",
        6 => "intersection",
        255 => "none",
        else => "unknown",
    };
}

fn platformTypeName(t: u8) []const u8 {
    return switch (t) {
        0 => "unknown",
        1 => "macOS",
        2 => "iOS",
        3 => "tvOS",
        4 => "watchOS",
        5 => "bridgeOS",
        6 => "macCatalyst",
        7 => "iOS simulator",
        8 => "tvOS simulator",
        9 => "watchOS simulator",
        10 => "driverKit",
        11 => "xrOS",
        12 => "macOS simulator",
        else => "unknown",
    };
}

fn dumpModule(out: *Io.Writer, m: Module) !void {
    try out.print("magic           MTLB\n", .{});
    try out.print("file version    {d}.{d}.{d}{s}\n", .{
        m.version.major,
        m.version.minor,
        m.version.patch,
        if (m.is_macos) " (macOS target)" else "",
    });
    try out.print("file type       {d} ({s}){s}\n", .{
        m.file_type,
        fileTypeName(m.file_type),
        if (m.is_stub) " [stub]" else "",
    });
    try out.print("platform        {d} ({s}){s}, version {d}.{d}.{d}\n", .{
        m.platform_type,
        platformTypeName(m.platform_type),
        if (m.is_64bit) " [64-bit]" else "",
        m.platform_version.major,
        m.platform_version.minor,
        m.platform_version.patch,
    });
    try out.print("file size       {d}\n", .{m.file_size});
    try out.print("function list   offset {d} size {d}\n", .{ m.function_list.offset, m.function_list.size });
    try out.print("public md       offset {d} size {d}\n", .{ m.public_md.offset, m.public_md.size });
    try out.print("private md      offset {d} size {d}\n", .{ m.private_md.offset, m.private_md.size });
    try out.print("module list     offset {d} size {d}\n", .{ m.module_list.offset, m.module_list.size });
    try out.print("header ext      {d} byte(s), {d} tag(s)\n", .{ m.ext_size, m.ext_tags.len });
    for (m.ext_tags) |t| {
        if (mem.eql(u8, &t.name, "UUID")) {
            if (decodeUuid(t.value)) |id| {
                try out.print("  UUID          ", .{});
                try printUuid(out, id);
                try out.print("\n", .{});
            } else {
                try out.print("  UUID          malformed: {d} byte(s)\n", .{t.value.len});
            }
        } else if (isSectionPointerTag(&t.name) and t.value.len == 16) {
            const offset = mem.readInt(u64, t.value[0..8], .little);
            const size = mem.readInt(u64, t.value[8..16], .little);
            try out.print("  {s}          section offset {d} size {d}\n", .{ &t.name, offset, size });
        } else {
            try out.print("  {s}          {d} byte(s) ", .{ &t.name, t.value.len });
            try printHexBytes(out, t.value, 16);
            try out.print("\n", .{});
        }
    }
    try out.print("functions       {d}\n", .{m.functions.len});
    for (m.functions, 0..) |f, i| {
        const hash_hex = std.fmt.bytesToHex(f.hash, .lower);
        try out.print("  [{d}] {s}\n", .{ i, f.name });
        try out.print("      type              {d} ({s})\n", .{ f.ty, programTypeName(f.ty) });
        try out.print("      module size       {d}\n", .{f.mdsz});
        try out.print("      module hash       {s}\n", .{hash_hex[0..]});
        try out.print("      offsets           public {d} private {d} module {d}\n", .{
            f.offt_public_md, f.offt_private_md, f.offt_air_module,
        });
        try out.print("      VERS              air {d}.{d} metal {d}.{d}\n", .{
            f.vers[0], f.vers[1], f.vers[2], f.vers[3],
        });
        try out.print("      metadata groups   public {d} byte(s), private {d} byte(s)\n", .{
            f.pub_md_size, f.priv_md_size,
        });
        if (f.rflt) |rflt| {
            try out.print("      RFLT              reflection buffer at {d}\n", .{rflt});
        }
        for (f.tags) |t| {
            if (isStandardFunctionTag(&t.name)) continue;
            if (mem.eql(u8, &t.name, "RFLT")) continue; // reported above
            try out.print("      tag {s}            {d} byte(s) ", .{ &t.name, t.value.len });
            try printHexBytes(out, t.value, 16);
            try out.print("\n", .{});
        }
    }
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

const BuildOptions = struct {
    /// Raw, unwrapped bitcode: stored verbatim, so MDSZ and HASH are its byte
    /// count and hash. Every name refers to these bytes: one module in the module
    /// list, one function group per name, each pointing at module offset 0.
    module: []const u8,
    /// One function group per name, in order. Verified on an M4: a library whose
    /// function list names the three kernels of a single LLVM module resolves
    /// every one of them with `newFunctionWithName:`, an unknown name returns nil,
    /// and the lookup is driven by these NAME tags rather than by the module's
    /// !air.kernel metadata.
    names: []const []const u8,
    air: Version = .{ .major = 2, .minor = 8 },
    metal: Version = .{ .major = 4, .minor = 0 },
    file_version: Version = .{ .major = 1, .minor = 2, .patch = 9 },
    platform_version: Version = .{ .major = 26, .minor = 0, .patch = 0 },
    uuid: bool = false,
    file_type: u8 = 0, // FILE_EXECUTABLE
    platform_type: u8 = 1, // PLATFORM_MACOS
    is_64bit: bool = true,
    is_macos: bool = true,
};

const Image = struct {
    /// The whole file; the caller owns it.
    bytes: []u8,
    function_count: u64,
    function_list_offset: u64,
    /// As stored in the header: excludes the leading u32 function count, so it is
    /// the sum of the individual group sizes.
    function_list_size: u64,
    public_md_offset: u64,
    public_md_size: u64,
    private_md_offset: u64,
    private_md_size: u64,
    module_list_offset: u64,
    uuid: ?u128,
};

fn putInt(w: *Io.Writer, comptime T: type, value: T) !void {
    var buf: [@divExact(@bitSizeOf(T), 8)]u8 = undefined;
    mem.writeInt(T, &buf, value, .little);
    try w.writeAll(&buf);
}

fn putTagHeader(w: *Io.Writer, name: *const [4]u8, value_len: u16) !void {
    try w.writeAll(name);
    try putInt(w, u16, value_len);
}

/// The records of one function tag group, in the order Xcode's fixtures use.
/// `offt_public_md`/`offt_private_md` point at this function's own empty group in
/// the metadata sections; the module offset is always 0 because every function
/// group in our libraries refers to the same module bytes.
fn writeFunctionRecords(
    w: *Io.Writer,
    opts: BuildOptions,
    digest: [32]u8,
    name: []const u8,
    offt_public_md: u64,
    offt_private_md: u64,
) !void {
    try putTagHeader(w, "NAME", @intCast(name.len + 1));
    try w.writeAll(name);
    try w.writeByte(0);

    try putTagHeader(w, "TYPE", 1);
    try w.writeByte(kernel_type);

    try putTagHeader(w, "HASH", 32);
    try w.writeAll(&digest);

    try putTagHeader(w, "OFFT", 24);
    try putInt(w, u64, offt_public_md);
    try putInt(w, u64, offt_private_md);
    try putInt(w, u64, 0); // module offset

    try putTagHeader(w, "VERS", 8);
    try putInt(w, u16, opts.air.major);
    try putInt(w, u16, opts.air.minor);
    try putInt(w, u16, opts.metal.major);
    try putInt(w, u16, opts.metal.minor);

    try putTagHeader(w, "MDSZ", 8);
    try putInt(w, u64, opts.module.len);

    try w.writeAll("ENDT");
}

/// Byte size of the group `writeFunctionRecords` emits, size field and ENDT included.
fn functionGroupSize(name_len: usize) u64 {
    const name_record: u64 = 4 + 2 + (name_len + 1);
    const type_record: u64 = 4 + 2 + 1;
    const hash_record: u64 = 4 + 2 + 32;
    const offt_record: u64 = 4 + 2 + 24;
    const vers_record: u64 = 4 + 2 + 8;
    const mdsz_record: u64 = 4 + 2 + 8;
    return 4 + // the group size field itself
        name_record + type_record + hash_record + offt_record + vers_record + mdsz_record +
        4; // ENDT
}

/// Builds the whole library image: header, function list, header extension, the
/// two empty metadata groups, then the module bytes. The caller owns `bytes`.
fn buildLibrary(gpa: Allocator, opts: BuildOptions) !Image {
    var digest: [32]u8 = undefined;
    Sha256.hash(opts.module, &digest, .{});

    var function_list = Io.Writer.Allocating.init(gpa);
    defer function_list.deinit();
    try putInt(&function_list.writer, u32, @intCast(opts.names.len));
    var function_list_size: u64 = 0;
    for (opts.names, 0..) |name, i| {
        var records = Io.Writer.Allocating.init(gpa);
        defer records.deinit();
        // Function i owns metadata group i: 8 bytes into each metadata section,
        // exactly like Apple's ref_default.metallib, whose second function
        // declares OFFT (8, 8, ...).
        const offt = 8 * @as(u64, i);
        try writeFunctionRecords(&records.writer, opts, digest, name, offt, offt);
        // The group size counts itself, its records and its ENDT (library.jl:518-522).
        const group_size: u64 = records.written().len + 4;
        try putInt(&function_list.writer, u32, @intCast(group_size));
        try function_list.writer.writeAll(records.written());
        function_list_size += group_size;
    }

    var extension = Io.Writer.Allocating.init(gpa);
    defer extension.deinit();
    var uuid: ?u128 = null;
    if (opts.uuid) {
        uuid = contentUuid(digest);
        const id = uuid.?;
        var value: [16]u8 = undefined;
        mem.writeInt(u64, value[0..8], @truncate(id >> 64), .little);
        mem.writeInt(u64, value[8..16], @truncate(id), .little);
        try putTagHeader(&extension.writer, "UUID", 16);
        try extension.writer.writeAll(&value);
    }
    try extension.writer.writeAll("ENDT");

    // An empty group still counts its ENDT, so it is 8 bytes (library.jl:873-886),
    // and each metadata section holds one empty group per function.
    const metadata_size: u64 = 8 * opts.names.len;

    const function_list_offset: u64 = header_size;
    const public_md_offset: u64 = function_list_offset + function_list.written().len + extension.written().len;
    const private_md_offset: u64 = public_md_offset + metadata_size;
    const module_list_offset: u64 = private_md_offset + metadata_size;
    const file_size: u64 = module_list_offset + opts.module.len;

    var out = Io.Writer.Allocating.init(gpa);
    errdefer out.deinit();
    const w = &out.writer;

    try w.writeAll(magic);
    try putInt(w, u16, if (opts.is_macos) opts.file_version.major | 0x8000 else opts.file_version.major);
    try putInt(w, u16, opts.file_version.minor);
    try putInt(w, u16, opts.file_version.patch);
    try w.writeByte(opts.file_type);
    try w.writeByte(if (opts.is_64bit) opts.platform_type | 0x80 else opts.platform_type);
    try putInt(w, u16, opts.platform_version.major);
    try w.writeByte(@intCast(opts.platform_version.minor));
    try w.writeByte(@intCast(opts.platform_version.patch));
    try putInt(w, u64, file_size);
    try putInt(w, u64, function_list_offset);
    try putInt(w, u64, function_list_size);
    try putInt(w, u64, public_md_offset);
    try putInt(w, u64, metadata_size);
    try putInt(w, u64, private_md_offset);
    try putInt(w, u64, metadata_size);
    try putInt(w, u64, module_list_offset);
    try putInt(w, u64, opts.module.len);
    if (out.written().len != header_size) return error.BadHeaderSize;

    try w.writeAll(function_list.written());
    try w.writeAll(extension.written());
    var i: usize = 0;
    while (i < opts.names.len) : (i += 1) {
        try putInt(w, u32, 8);
        try w.writeAll("ENDT"); // public metadata: one empty group per function
    }
    i = 0;
    while (i < opts.names.len) : (i += 1) {
        try putInt(w, u32, 8);
        try w.writeAll("ENDT"); // private metadata: one empty group per function
    }
    try w.writeAll(opts.module);
    if (out.written().len != file_size) return error.BadFileSize;

    return .{
        .bytes = try out.toOwnedSlice(),
        .function_count = opts.names.len,
        .function_list_offset = function_list_offset,
        .function_list_size = function_list_size,
        .public_md_offset = public_md_offset,
        .public_md_size = metadata_size,
        .private_md_offset = private_md_offset,
        .private_md_size = metadata_size,
        .module_list_offset = module_list_offset,
        .uuid = uuid,
    };
}

// ---------------------------------------------------------------------------
// command line
// ---------------------------------------------------------------------------

fn bail(out: *Io.Writer, comptime fmt: []const u8, args: anytype) noreturn {
    out.print(fmt ++ "\n", args) catch {};
    out.flush() catch {};
    std.process.exit(1);
}

fn usage(out: *Io.Writer) void {
    out.print(
        \\usage:
        \\  metallib write --bitcode <file> --name <kernel> [--name <kernel> ...]
        \\                 [--air M.m] [--metal M.m] [--format M.m.p] [--platform M.m.p]
        \\                 [--uuid] -o <out.metallib>
        \\  metallib dump <file.metallib>
        \\  metallib selftest [--dir <dir>] [--bitcode <file>] [ref.metallib ...]
        \\
        \\--name is repeatable: one function group per name, all sharing the module bytes.
        \\defaults: --air 2.8 --metal 4.0 --format 1.2.9 --platform 26.0.0
        \\
    , .{}) catch {};
}

fn nextValue(args: []const []const u8, i: *usize, flag: []const u8, out: *Io.Writer) []const u8 {
    if (i.* + 1 >= args.len) bail(out, "missing value for {s}", .{flag});
    i.* += 1;
    return args[i.*];
}

fn parseVersion(text: []const u8, comptime components: usize) !Version {
    var parts = [3]u16{ 0, 0, 0 };
    var seen: usize = 0;
    var it = mem.splitScalar(u8, text, '.');
    while (it.next()) |part| {
        if (seen == 3) return error.TooManyComponents;
        parts[seen] = std.fmt.parseInt(u16, part, 10) catch return error.NotANumber;
        seen += 1;
    }
    if (seen != components) return error.WrongComponentCount;
    return .{ .major = parts[0], .minor = parts[1], .patch = parts[2] };
}

fn versionArg(
    args: []const []const u8,
    i: *usize,
    flag: []const u8,
    comptime components: usize,
    out: *Io.Writer,
) Version {
    const text = nextValue(args, i, flag, out);
    return parseVersion(text, components) catch bail(
        out,
        "invalid value for {s}: '{s}' (expected {d} dot-separated integers)",
        .{ flag, text, components },
    );
}

fn readInput(gpa: Allocator, io: Io, path: []const u8, out: *Io.Writer) []u8 {
    return Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |err| bail(
        out,
        "unable to read {s}: {t}",
        .{ path, err },
    );
}

fn writeOutput(io: Io, path: []const u8, data: []const u8, out: *Io.Writer) void {
    Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data }) catch |err| bail(
        out,
        "unable to write {s}: {t}",
        .{ path, err },
    );
}

fn noteWrappedModule(out: *Io.Writer, module: []const u8) !void {
    if (module.len < 4) return;
    if (mem.readInt(u32, module[0..4], .little) != bitcode_wrapper_magic) return;
    try out.print(
        "NOTE: the module starts with Apple's 0x0b17c0de bitcode wrapper. Store raw bitcode: " ++
            "wrapped modules were rejected by newComputePipelineStateWithFunction " ++
            "(XPC_ERROR_CONNECTION_INTERRUPTED) on macOS 26.6.2, raw modules loaded and ran.\n",
        .{},
    );
}

fn cmdWrite(gpa: Allocator, arena: Allocator, io: Io, out: *Io.Writer, args: []const []const u8) !void {
    var opts = BuildOptions{ .module = &.{}, .names = &.{} };
    var names: std.ArrayList([]const u8) = .empty;
    var bitcode_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (mem.eql(u8, arg, "--bitcode")) {
            bitcode_path = nextValue(args, &i, arg, out);
        } else if (mem.eql(u8, arg, "--name")) {
            const name = nextValue(args, &i, arg, out);
            if (name.len == 0 or name.len > 65534) {
                bail(out, "--name must be 1..65534 byte(s), got {d}", .{name.len});
            }
            for (names.items) |existing| {
                if (mem.eql(u8, existing, name)) bail(out, "duplicate --name '{s}'", .{name});
            }
            try names.append(arena, name);
        } else if (mem.eql(u8, arg, "--air")) {
            const v = versionArg(args, &i, arg, 2, out);
            opts.air = .{ .major = v.major, .minor = v.minor };
        } else if (mem.eql(u8, arg, "--metal")) {
            const v = versionArg(args, &i, arg, 2, out);
            opts.metal = .{ .major = v.major, .minor = v.minor };
        } else if (mem.eql(u8, arg, "--format")) {
            opts.file_version = versionArg(args, &i, arg, 3, out);
        } else if (mem.eql(u8, arg, "--platform")) {
            opts.platform_version = versionArg(args, &i, arg, 3, out);
        } else if (mem.eql(u8, arg, "--uuid")) {
            opts.uuid = true;
        } else if (mem.eql(u8, arg, "-o") or mem.eql(u8, arg, "--output")) {
            output_path = nextValue(args, &i, arg, out);
        } else {
            usage(out);
            bail(out, "unknown argument '{s}'", .{arg});
        }
    }

    const bitcode_path_final = bitcode_path orelse {
        usage(out);
        bail(out, "missing required option --bitcode <file>", .{});
    };
    if (names.items.len == 0) {
        usage(out);
        bail(out, "missing required option --name <kernel> (repeatable)", .{});
    }
    const dest = output_path orelse {
        usage(out);
        bail(out, "missing required option -o <out.metallib>", .{});
    };
    if (opts.platform_version.minor > 255 or opts.platform_version.patch > 255) {
        bail(out, "--platform minor and patch must fit in one byte", .{});
    }

    const module = readInput(gpa, io, bitcode_path_final, out);
    defer gpa.free(module);
    opts.module = module;
    opts.names = names.items;

    const image = try buildLibrary(gpa, opts);
    defer gpa.free(image.bytes);
    writeOutput(io, dest, image.bytes, out);

    // Self-check: the image we just produced must read back as a valid library.
    var report = Report.init(gpa);
    defer report.deinit();
    if (parseModule(arena, image.bytes, &report)) |_| {} else |err| switch (err) {
        error.Mismatch => {
            try printPrefixed(out, "NOTE", report.notes.written());
            try printPrefixed(out, "MISMATCH", report.problems.written());
            bail(out, "internal error: the library just written does not parse", .{});
        },
        else => |e| return e,
    }
    try printPrefixed(out, "NOTE", report.notes.written());

    var digest: [32]u8 = undefined;
    Sha256.hash(module, &digest, .{});
    const digest_hex = std.fmt.bytesToHex(digest, .lower);
    try out.print("bitcode         {s} ({d} byte(s), sha256 {s})\n", .{
        bitcode_path_final, module.len, digest_hex[0..],
    });
    try out.print("library         {s} ({d} byte(s))\n", .{ dest, image.bytes.len });
    try out.print("functions       {d}\n", .{image.function_count});
    try out.print("file version    {d}.{d}.{d}{s}\n", .{
        opts.file_version.major,
        opts.file_version.minor,
        opts.file_version.patch,
        if (opts.is_macos) " (macOS target)" else "",
    });
    try out.print("platform        {d}.{d}.{d} (type 0x{x:0>2})\n", .{
        opts.platform_version.major,
        opts.platform_version.minor,
        opts.platform_version.patch,
        if (opts.is_64bit) opts.platform_type | 0x80 else opts.platform_type,
    });
    if (image.uuid) |id| {
        try out.print("uuid            ", .{});
        try printUuid(out, id);
        try out.print("\n", .{});
    }
    try out.print("kernel          {s} TYPE {d} (kernel) VERS air {d}.{d} metal {d}.{d} MDSZ {d}\n", .{
        names.items[0],
        kernel_type,
        opts.air.major,
        opts.air.minor,
        opts.metal.major,
        opts.metal.minor,
        module.len,
    });
    for (names.items[1..]) |name| {
        try out.print("kernel          {s} TYPE {d} (kernel) VERS air {d}.{d} metal {d}.{d} MDSZ {d}\n", .{
            name,
            kernel_type,
            opts.air.major,
            opts.air.minor,
            opts.metal.major,
            opts.metal.minor,
            module.len,
        });
    }
    try out.print(
        "sections        function list {d}+{d}, public md {d}+{d}, private md {d}+{d}, module list {d}+{d}\n",
        .{
            image.function_list_offset,
            image.function_list_size,
            image.public_md_offset,
            image.public_md_size,
            image.private_md_offset,
            image.private_md_size,
            image.module_list_offset,
            module.len,
        },
    );
    if (names.items.len > 1) {
        try out.print(
            "NOTE: {d} function group(s) share one module: each declares OFFT module offset 0, " ++
                "MDSZ {d} and the same HASH, and owns its own empty metadata group at offset {d}.\n",
            .{ names.items.len, module.len, 8 },
        );
    }
    try noteWrappedModule(out, module);
    try out.print("OK\n", .{});
}

fn cmdDump(gpa: Allocator, arena: Allocator, io: Io, out: *Io.Writer, args: []const []const u8) !void {
    if (args.len != 1) bail(out, "usage: metallib dump <file.metallib>", .{});
    const path = args[0];

    const bytes = readInput(gpa, io, path, out);
    defer gpa.free(bytes);

    var report = Report.init(gpa);
    defer report.deinit();

    const module = parseModule(arena, bytes, &report) catch |err| switch (err) {
        error.Mismatch => {
            try out.print("metallib        {s} ({d} byte(s))\n", .{ path, bytes.len });
            try printPrefixed(out, "NOTE", report.notes.written());
            try printPrefixed(out, "MISMATCH", report.problems.written());
            bail(out, "FAILED {s}: {d} mismatch(es)", .{ path, countLines(report.problems.written()) });
        },
        else => |e| return e,
    };

    try out.print("metallib        {s} ({d} byte(s))\n", .{ path, bytes.len });
    try printPrefixed(out, "NOTE", report.notes.written());
    try dumpModule(out, module);
    try out.print("OK\n", .{});
}

// ---------------------------------------------------------------------------
// selftest
// ---------------------------------------------------------------------------

const Checks = struct {
    out: *Io.Writer,
    passed: usize = 0,
    failed: usize = 0,

    fn eqInt(c: *Checks, comptime what: []const u8, expected: anytype, actual: @TypeOf(expected)) !void {
        if (expected == actual) {
            c.passed += 1;
        } else {
            c.failed += 1;
            try c.out.print("FAIL: {s}: expected {d}, got {d}\n", .{ what, expected, actual });
        }
    }

    fn eqVers(c: *Checks, comptime what: []const u8, expected: [4]u16, actual: [4]u16) !void {
        if (mem.eql(u16, &expected, &actual)) {
            c.passed += 1;
        } else {
            c.failed += 1;
            try c.out.print("FAIL: {s}: expected {d}.{d}/{d}.{d}, got {d}.{d}/{d}.{d}\n", .{
                what,
                expected[0],
                expected[1],
                expected[2],
                expected[3],
                actual[0],
                actual[1],
                actual[2],
                actual[3],
            });
        }
    }

    fn eqBytes(c: *Checks, comptime what: []const u8, expected: []const u8, actual: []const u8) !void {
        if (mem.eql(u8, expected, actual)) {
            c.passed += 1;
        } else {
            c.failed += 1;
            try c.out.print("FAIL: {s}: expected {d} byte(s) ", .{ what, expected.len });
            try printHexBytes(c.out, expected, 16);
            try c.out.print(", got {d} byte(s) ", .{actual.len});
            try printHexBytes(c.out, actual, 16);
            try c.out.print("\n", .{});
        }
    }
};

/// Re-reads a written image and checks every field of its structure against what
/// was asked for, so the round trip covers the whole layout, not a sample of it.
fn checkRoundTrip(
    checks: *Checks,
    arena: Allocator,
    image: []const u8,
    module: []const u8,
    names: []const []const u8,
    expect_uuid: ?u128,
) !void {
    var report = Report.init(arena);
    defer report.deinit();

    const m = parseModule(arena, image, &report) catch |err| switch (err) {
        error.Mismatch => {
            checks.failed += 1;
            try checks.out.print("FAIL: the written image does not parse\n", .{});
            try printPrefixed(checks.out, "  MISMATCH", report.problems.written());
            return;
        },
        else => |e| return e,
    };
    try checks.eqInt("no notes when re-reading", @as(usize, 0), report.notes.written().len);

    var digest: [32]u8 = undefined;
    Sha256.hash(module, &digest, .{});

    // header
    try checks.eqInt("file version major", @as(u16, 1), m.version.major);
    try checks.eqInt("file version minor", @as(u16, 2), m.version.minor);
    try checks.eqInt("file version patch", @as(u16, 9), m.version.patch);
    try checks.eqInt("is macOS", @as(u8, 1), @intFromBool(m.is_macos));
    try checks.eqInt("file type", @as(u8, 0), m.file_type);
    try checks.eqInt("is stub", @as(u8, 0), @intFromBool(m.is_stub));
    try checks.eqInt("platform type", @as(u8, 1), m.platform_type);
    try checks.eqInt("is 64-bit", @as(u8, 1), @intFromBool(m.is_64bit));
    try checks.eqInt("platform major", @as(u16, 26), m.platform_version.major);
    try checks.eqInt("platform minor", @as(u16, 0), m.platform_version.minor);
    try checks.eqInt("platform patch", @as(u16, 0), m.platform_version.patch);
    try checks.eqInt("file size", @as(u64, image.len), m.file_size);

    // sections: contiguous, and each exactly where the documented layout puts it
    const extension_size: u64 = if (expect_uuid != null) 26 else 4;
    var expected_function_list_size: u64 = 0;
    for (names) |name| expected_function_list_size += functionGroupSize(name.len);
    const expected_metadata_size: u64 = 8 * @as(u64, names.len);
    try checks.eqInt("function list offset", @as(u64, 88), m.function_list.offset);
    try checks.eqInt("function list size", expected_function_list_size, m.function_list.size);
    try checks.eqInt(
        "public md offset",
        m.function_list.offset + function_count_size + m.function_list.size + extension_size,
        m.public_md.offset,
    );
    try checks.eqInt("public md size", expected_metadata_size, m.public_md.size);
    try checks.eqInt("private md offset", m.public_md.offset + expected_metadata_size, m.private_md.offset);
    try checks.eqInt("private md size", expected_metadata_size, m.private_md.size);
    try checks.eqInt("module list offset", m.private_md.offset + expected_metadata_size, m.module_list.offset);
    try checks.eqInt("module list size", @as(u64, module.len), m.module_list.size);
    try checks.eqInt("header extension size", extension_size, m.ext_size);

    // one function group per name, in order, all sharing module 0
    try checks.eqInt("function count", names.len, m.functions.len);
    for (names, 0..) |name, i| {
        if (i >= m.functions.len) break;
        const f = m.functions[i];
        const offt: u64 = 8 * @as(u64, i);
        try checks.eqBytes("function name", name, f.name);
        try checks.eqInt("function type", @as(u8, 2), f.ty);
        try checks.eqInt("function MDSZ", @as(u64, module.len), f.mdsz);
        try checks.eqBytes("function HASH", &digest, &f.hash);
        try checks.eqInt("OFFT public", offt, f.offt_public_md);
        try checks.eqInt("OFFT private", offt, f.offt_private_md);
        try checks.eqInt("OFFT module", @as(u64, 0), f.offt_air_module);
        try checks.eqVers("VERS air 2.8 metal 4.0", .{ 2, 8, 4, 0 }, f.vers);
        try checks.eqInt("public metadata group size", @as(u64, 8), f.pub_md_size);
        try checks.eqInt("private metadata group size", @as(u64, 8), f.priv_md_size);
        try checks.eqInt("RFLT absent", @as(u8, 0), @intFromBool(f.rflt != null));
        try checks.eqInt(
            "group size matches the header field",
            @as(u32, @intCast(functionGroupSize(name.len))),
            f.group_size,
        );
    }

    // the module bytes, verbatim
    const module_start: usize = @intCast(m.module_list.offset);
    try checks.eqBytes("module bytes", module, image[module_start..]);

    // uuid
    try checks.eqInt("header tag count", @as(usize, if (expect_uuid != null) 1 else 0), m.ext_tags.len);
    if (expect_uuid) |expected| {
        try checks.eqInt("UUID present", @as(u8, 1), @intFromBool(m.uuid != null));
        if (m.uuid) |got| {
            try checks.eqInt("UUID high word", @as(u64, @truncate(expected >> 64)), @as(u64, @truncate(got >> 64)));
            try checks.eqInt("UUID low word", @as(u64, @truncate(expected)), @as(u64, @truncate(got)));
        }
    } else {
        try checks.eqInt("UUID absent", @as(u8, 0), @intFromBool(m.uuid != null));
    }
}

fn syntheticBitcode(gpa: Allocator) ![]u8 {
    const bytes = try gpa.alloc(u8, synthetic_bitcode_hex.len / 2);
    errdefer gpa.free(bytes);
    _ = try std.fmt.hexToBytes(bytes, synthetic_bitcode_hex);
    return bytes;
}

/// Writes `module` to `<dir>/<stem>.bc`, builds a library naming it once per entry of
/// `names`, writes it to `<dir>/<stem>.metallib`, re-reads it from disk and checks
/// every field of the result. Returns the number of failed checks.
fn roundTripCase(
    gpa: Allocator,
    arena: Allocator,
    io: Io,
    out: *Io.Writer,
    dir: []const u8,
    stem: []const u8,
    module: []const u8,
    names: []const []const u8,
    with_uuid: bool,
) !usize {
    const module_path = try std.fs.path.join(arena, &.{ dir, try mem.concat(arena, u8, &.{ stem, ".bc" }) });
    const library_path = try std.fs.path.join(arena, &.{ dir, try mem.concat(arena, u8, &.{ stem, ".metallib" }) });
    writeOutput(io, module_path, module, out);
    try out.print("synthetic       {s} ({d} byte(s), sha256 ", .{ module_path, module.len });
    var digest: [32]u8 = undefined;
    Sha256.hash(module, &digest, .{});
    {
        const hex = std.fmt.bytesToHex(digest, .lower);
        try out.print("{s})\n", .{hex[0..]});
    }

    const image = try buildLibrary(gpa, .{ .module = module, .names = names, .uuid = with_uuid });
    defer gpa.free(image.bytes);
    writeOutput(io, library_path, image.bytes, out);
    try out.print("library         {s} ({d} byte(s), {d} function(s), uuid {s})\n", .{
        library_path, image.bytes.len, names.len, if (with_uuid) "on" else "off",
    });

    const reread = readInput(gpa, io, library_path, out);
    defer gpa.free(reread);

    var checks = Checks{ .out = out };
    try checkRoundTrip(
        &checks,
        arena,
        reread,
        module,
        names,
        if (with_uuid) contentUuid(digest) else null,
    );
    try out.print("round-trip      {d} field(s) OK, {d} failed\n", .{ checks.passed, checks.failed });
    return checks.failed;
}

fn cmdSelftest(
    gpa: Allocator,
    arena: Allocator,
    io: Io,
    environ: std.process.Environ,
    out: *Io.Writer,
    args: []const []const u8,
) !void {
    var dir: ?[]const u8 = null;
    var bitcode_path: ?[]const u8 = null;
    var refs: std.ArrayList([]const u8) = .empty;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (mem.eql(u8, arg, "--dir")) {
            dir = nextValue(args, &i, arg, out);
        } else if (mem.eql(u8, arg, "--bitcode")) {
            bitcode_path = nextValue(args, &i, arg, out);
        } else if (mem.startsWith(u8, arg, "-")) {
            usage(out);
            bail(out, "unknown argument '{s}'", .{arg});
        } else {
            try refs.append(arena, arg);
        }
    }
    const work_dir = dir orelse (environ.getAlloc(arena, "TMPDIR") catch "/tmp");

    var failed: usize = 0;

    // 1. reference libraries, when their paths were passed.
    for (refs.items) |path| {
        const bytes = readInput(gpa, io, path, out);
        defer gpa.free(bytes);
        var report = Report.init(gpa);
        defer report.deinit();
        const m = parseModule(arena, bytes, &report) catch |err| switch (err) {
            error.Mismatch => {
                failed += 1;
                try out.print("ref {s}: FAILED\n", .{path});
                try printPrefixed(out, "  NOTE", report.notes.written());
                try printPrefixed(out, "  MISMATCH", report.problems.written());
                continue;
            },
            else => |e| return e,
        };
        try out.print(
            "ref {s}: OK ({d} byte(s), file version {d}.{d}.{d}, {d} function(s), {d} header tag(s))\n",
            .{
                path,            bytes.len,       m.version.major, m.version.minor,
                m.version.patch, m.functions.len, m.ext_tags.len,
            },
        );
        try printPrefixed(out, "  NOTE", report.notes.written());
        for (m.functions) |f| {
            try out.print(
                "  {s}: type {d} ({s}) MDSZ {d} VERS air {d}.{d} metal {d}.{d} OFFT ({d},{d},{d})\n",
                .{
                    f.name,           f.ty,              programTypeName(f.ty), f.mdsz,
                    f.vers[0],        f.vers[1],         f.vers[2],             f.vers[3],
                    f.offt_public_md, f.offt_private_md, f.offt_air_module,
                },
            );
        }
    }

    // 2. a synthetic module, twice: once naming it with a single function and no
    //    UUID, once naming it three times with a UUID (three groups sharing one
    //    module, with different name lengths).
    const synthetic = if (bitcode_path) |p| readInput(gpa, io, p, out) else try syntheticBitcode(gpa);
    defer gpa.free(synthetic);

    failed += try roundTripCase(
        gpa,
        arena,
        io,
        out,
        work_dir,
        "metallib_selftest_single",
        synthetic,
        &.{"synthetic"},
        false,
    );
    failed += try roundTripCase(
        gpa,
        arena,
        io,
        out,
        work_dir,
        "metallib_selftest_shared",
        synthetic,
        &.{ "vadd", "reduce", "parsef" },
        true,
    );

    if (failed != 0) bail(out, "FAILED: {d} check(s)", .{failed});
    try out.print("PASS\n", .{});
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_writer.interface;
    defer out.flush() catch {};

    if (args.len < 2) {
        usage(out);
        bail(out, "missing command", .{});
    }

    const command = args[1];
    const rest = args[2..];
    if (mem.eql(u8, command, "write")) {
        try cmdWrite(gpa, arena, io, out, rest);
    } else if (mem.eql(u8, command, "dump")) {
        try cmdDump(gpa, arena, io, out, rest);
    } else if (mem.eql(u8, command, "selftest")) {
        try cmdSelftest(gpa, arena, io, init.minimal.environ, out, rest);
    } else {
        usage(out);
        bail(out, "unknown command '{s}'", .{command});
    }
    try out.flush();
}

// ---------------------------------------------------------------------------
// synthetic bitcode for `selftest`
// ---------------------------------------------------------------------------

/// A trivial LLVM 23 bitcode module, used by `selftest` when no --bitcode file is
/// given. Produced with LLVM 23.1.2 from:
///
///   $ cat synthetic.ll
///   ; synthetic module for metallib selftest
///   target triple = "air64_v28-apple-macosx26.0.0"
///   define void @synthetic(ptr addrspace(1) %a, i32 %gid) {
///   entry:
///     store i32 %gid, ptr addrspace(1) %a, align 4
///     ret void
///   }
///   $ llvm-as synthetic.ll -o synthetic.bc
///   $ sha256sum synthetic.bc
///   c8d726f6ab6e92f1ecfd10fe9bb5d52d410bb98957fcb5622526ee319b6a50a3
///
const synthetic_bitcode_hex =
    "dec0170b0000000014000000ac070000ffffffff4243c0de3514000005000000620c30244a59bea67dfbb56f0b51804c" ++ "01000000210c0000cf0100000b02210002000000230000000781239141c80449061032399201840c250508191e048b62" ++ "800c4502428a2384241784c820642870103096146464109160c942860c11098e38648448e29011224852800c19219602" ++ "64c80811243940468610cb01323284483254505420a3b8404672810c1932860f962b32641819490e3264c45872902123" ++ "469025101d3a74c8888e902144868c0432340000892000000b00000022660410b24282c910524282c99071c250480a09" ++ "2643c605423226080c1a01982c08e608c0a00cd10602000013b870480779b0033af8057b90033c688370800778608772" ++ "688376088771788779c00739b0033780033780038d10864c981f880cc95f168d14265054000000000000000018620109" ++ "010000000000000000000000900080c40681c2d3010000b118000000100100003308801cc4e11c6614013d88433884c3" ++ "8c4280077978077398710ce6000fed100ef4800e330c421ec2c11dcea11c6630053d88433884831bcc033dc8433d8c03" ++ "3dcc788c7470077b08077948877070077a700376788770208719cc110eec900ee1300f6e300fe3f00ef0500e3310c41d" ++ "de211cd8211dc2611e6630893bbc833bd04339b4033cbc833c84033bccf0147660077b68073768877268073780877090" ++ "8770600776280776f8057678877780875f08877118877298877998812ceef00eeee00ef5c00eec300362c8a11ce4a11c" ++ "cca11ce4a11cdc611cca211cc4811dca6106d6904339c84339984339c84339b8c33894433888033b94c32fbc833cfc82" ++ "3bd4033bb0c30cc7698770588772708374680778608774188774a08719ce530fee000ff2500ee4900ee3400fe1200eec" ++ "500e3320281ddcc11ec2411ed2211cdc811edce01ce4e11dea011e66185138b0433a9c833bcc50247660077b68073760" ++ "877778077898514cf4900ff0500e331e6a1eca611ce8211ddec11d7e011ee4a11ccc211df0610654858338ccc33bb043" ++ "3dd04339fcc23ce4433b88c33bb0c38cc50a877998877718877408077a28077298815ce3100eecc00ee5500ef33023c1" ++ "d2411ee4e117d8e11dde011e6648193bb0833db4831b84c3388c4339ccc33cb8c139c8c33bd4033ccc48b47108077660" ++ "0771088771588719dbc60eec600fede006f0200fe5300fe5200ff6500e6e100ee3300ee5300ff3e006e9e00ee4500ef8" ++ "3023e2ec611cc2811dd8e117ec211de6211dc4211dd8211de8211f66209d3bbc433db80339948339cc58bc7070077778" ++ "077a08077a488777708719cbe70eef300fe1e00ee9400fe9a00fe530c3010373a8077718875f988770708774a08774d0" ++ "87729881844139e0c338b0433d904339cc40c4a01dcaa11de0411edec11c662463300ee1c00eec300fe9400fe5304321" ++ "837518077348875fa0877c80877298b194013c8cc33c94c338d0433abc833bccc38cc50c48211542611ee6211dcec11d" ++ "528114664c67300eef200fefe006ef500ff4300fe9400ee5e006e6200fe1d00ee530a3408376680779088719521ab8c3" ++ "3b84033ba44338cc831b84033990833ccc033c84c33894c30c460dc6211cd8811dcaa11c7e811ef2011eca61c6b106ee" ++ "f00ee6200fe5500e33123618877080077aa8077928877998c1b44138b0033bbcc338fc023dbcc33a94833bcc68dc201d" ++ "da011ed8211dc6211de8c10de4a11ccc6186f206eef00ee6000fe3c00ee1300ff3308381837108077660875fa0877090" ++ "877328077a98f1c4413ab8033ba4833b94c32fa0433acc033dbc833ce4c38c4b0eca811dcce117e6a11cc6811ed2e11d" ++ "dce117e0411ede011eca411ee8211dca611e661473700ef5900ee430e3a18376288776708371088771408772f8057448" ++ "0777a08719591d88033bbcc338ac831bd4833ba48339bc833cb4433ad0433eb8013cc8c33b98433ab04339cc50ec601c" ++ "c2811dd8e11ce4211ce0011d00000000a9180000570000000b8a7060877438077758408cc33bb003392c78c8a11ce4a1" ++ "1ccca11ce4a11cdc611cca211cc4811dca61c1450ee5200fe5600ee5200fe5e00ee3500ee1200eec500ebff00ef2f00b" ++ "ee500fecc00e0b8c74380777780779288705873bbc833bb8433db0033b2c38dce11deac11dc8a11ccc6101340fe5000f" ++ "e1200fe1400fe5f00bf3400fef200fe1700ee5b0a0208772780778a087058739d4833b8c033b94033d2c68ce611cda80" ++ "1ee4211cdc611ed2811ed2e11ddc6141330ee6700ef5100ef2400ef4100ef2700ee5400f0b1878908772080776608777" ++ "188770a08772208705c7398c431bb0433ad843392cb0c6811dc2c11dcec10dc2411ec6c10dc2811ee8211cc6011dca81" ++ "1cc6211cd8811d161cf0400ff2100ef5400fe8b08058877130877458f08cc33bb8833d94833c9c4339b8c33894c338d0" ++ "833cb0c382891cca211cc6811ed2c11ec2811ed2e11ddca10de6211fda411cde811d0000d11000000600000007cc3ca4" ++ "833b9c033b94033da0833c94433890c301000000612800000900000023088222048305070200000004000000263018c8" ++ "700005d14c11660101000000000000007120000003000000320e10228400c80300000000000000005d0c000010000000" ++ "120394740000000073796e74686574696332332e312e3261697236345f7632382d6170706c652d6d61636f737832362e" ++ "302e306d5f617267732e6c6c00000000";
