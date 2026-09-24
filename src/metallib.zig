//! `.metallib`: the container the Metal runtime loads AIR bitcode from.
//!
//! `write` produces the image and `parse` reads it back, so tests can check a
//! whole library instead of a prefix of one. The layout is the one measured
//! against Apple's own `xcrun metallib` output: an image written this way
//! resolved its kernels with `newFunctionWithName:` on an M4 (macOS 26.6.2),
//! while the same module stored without Apple's module-section header was
//! rejected by `newComputePipelineStateWithFunction`.
//!
//! Shape of an image -- every integer little-endian, sections concatenated with
//! no padding:
//!
//!   * 88-byte header: `MTLB` magic, format version (major in bits 0-14,
//!     "targets macOS" in bit 15), file type 0 (executable), platform type 0x81
//!     (macOS, 64-bit), the deployment version (major `u16`, minor and patch one
//!     `u8` each), the total file size, then four (offset, size) pairs for the
//!     function list, public metadata, private metadata and module list. Section
//!     offsets are absolute; the pairs are in that order, 16 bytes each.
//!   * function list: a `u32` function count, then one tag group per function, in
//!     function order. A group is `u32 size` -- counting itself, its records and
//!     the trailing `ENDT` -- then `<4-byte tag><u16 value_size><value>` records,
//!     then the ASCII `ENDT`. The header's function-list size excludes the count,
//!     so for a single function it equals the group size.
//!   * records of a function, in Apple's order: `NAME` (NUL-terminated, the NUL
//!     is part of the value), `TYPE` (`u8`: 2 for a kernel, 0 for an ordinary
//!     function), `HASH` (SHA-256 over the stored module bytes), `OFFT` (three
//!     `u64`: public metadata, private metadata and module offsets, relative to
//!     their sections), `VERS` (four `u16`: AIR major, AIR minor, Metal major,
//!     Metal minor) and `MDSZ` (module byte count, module-section header
//!     included).
//!   * header extension: records with no size field in front, always terminated
//!     by `ENDT`. This writer emits only `UUID` (two `u64`, high word first,
//!     derived from the module hash so output is reproducible), and only when
//!     asked for; otherwise the extension is a bare `ENDT`. Apple's optional
//!     section-pointer records (`HSRC`/`HSRD`/`HDYN`/`RLST`) are skipped when
//!     reading and never written: we embed no source, dynamic library name or
//!     reflection data.
//!   * public and private metadata: one empty 8-byte group (`u32 8` + `ENDT`) per
//!     function, in function order, each function's `OFFT` pointing at its own.
//!   * module list: the module bytes verbatim, with no framing of their own; MDSZ
//!     segments them when a library stores several. Nothing is compressed: no
//!     part of the image is deflated, and HASH covers exactly the stored bytes.
//!
//! The extra `04 00 00 00` Apple's own writer leaves after some groups is not
//! reproduced; the layout without it loads and runs (measured on the M4).
//!
//! `write` stores the module bytes it is handed, so pass `wrapModule`'s output: a
//! module carrying Apple's 20-byte `0x0b17c0de` section header. `parse` accepts a
//! bare module but reports it through `Image.module_header`, because storing one
//! bare is a bug on the writer's side rather than a valid library.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// `MTLB` file magic.
const magic = "MTLB";
/// Size of the fixed header, `0x58`.
const header_size: usize = 88;
/// The `u32` function count in front of the function-list groups, which the
/// header's function-list size excludes.
const function_count_size: usize = 4;
/// The ASCII `ENDT` terminator that closes a tag group: four bytes, no size field
/// of its own.
const endt_size: usize = 4;
/// A group's own `u32` size field.
const group_size_size: usize = 4;
/// One empty metadata group: its size field, then `ENDT`. Empty groups count
/// their `ENDT`, which is what Xcode's fixtures contain.
const metadata_group_size: usize = 8;
/// `UUID` record: tag name, `u16` value size, two `u64`.
const uuid_record_size: usize = 4 + 2 + 16;
/// One record's tag name plus its `u16` value size.
const record_header_size: usize = 4 + 2;
/// TYPE value of a kernel; a non-kernel function is written as 0.
const kernel_type: u8 = 2;
const function_type: u8 = 0;
/// FILE_EXECUTABLE, PLATFORM_MACOS: what `write` stamps into every header.
const file_type_executable: u8 = 0;
const platform_type_macos: u8 = 1;
/// Apple's module-section header: magic, `u32 0`, `u32 0x14`, `u32 size`,
/// `i32 -1`.
const module_header_magic: u32 = 0x0b17c0de;
const module_header_size: usize = 20;
/// Sanity bound when reading a function count out of an untrusted file.
const max_functions: u32 = 65536;
/// Longest NAME value the `u16` value size can hold, NUL included, so the
/// longest name is one byte shorter.
const max_name_len: usize = std.math.maxInt(u16) - 1;

// ---------------------------------------------------------------------------
// the writer's API
// ---------------------------------------------------------------------------

/// Versions stamped into the container (from `std.Target.air64.versionsForMacos`
/// at the call site).
pub const Versions = struct {
    /// `!air.version` / VERS tag: AIR major, minor, patch. VERS holds two `u16`
    /// per language, so only `[0]` and `[1]` reach the file.
    air: [3]u16,
    /// `!air.language_version` / VERS tag: Metal language major, minor, patch,
    /// likewise with only the first two encoded.
    metal: [3]u16,
    /// Container file version (the u16 triple after "MTLB"); the major must fit
    /// the 15 bits below the "targets macOS" flag.
    format: [3]u16,
    /// Platform version stamped into the header (macOS major.minor.patch); minor
    /// and patch are one byte each in the header.
    platform: [3]u16,
};

/// One kernel (or visible library function) in the library's function list.
pub const Function = struct {
    /// The name the host resolves with `newFunctionWithName:`. 1 to 65534 bytes,
    /// with no NUL: the container stores its own NUL after the name, and an
    /// embedded one would truncate the name the host resolves.
    name: []const u8,
    /// True for a kernel (TYPE tag value 2), false for an ordinary function (TYPE 0).
    is_kernel: bool = true,
};

pub const WriteOptions = struct {
    /// Functions in function-list order. At least one is required; every one of
    /// them names the same module bytes, because our kernels live in one LLVM
    /// module (module offset 0, one MDSZ, one HASH).
    functions: []const Function,
    versions: Versions,
    /// Emit the UUID tag, derived from the module hash so output is reproducible.
    uuid: bool = true,
};

/// Builds a complete `.metallib` image whose module section holds `module` bytes
/// (already wrapped by `wrapModule`) and whose function list names every entry of
/// `options.functions`. The caller owns the returned bytes.
///
/// `module` is stored verbatim: MDSZ is its byte count and HASH its SHA-256, so
/// raw, unwrapped bitcode produces a library the runtime rejects. Nothing here
/// inspects or rewrites the module's contents.
pub fn write(gpa: Allocator, module: []const u8, options: WriteOptions) Allocator.Error![]u8 {
    const functions = options.functions;
    std.debug.assert(functions.len >= 1); // a library with no function leaves the module unreferenced
    std.debug.assert(functions.len <= std.math.maxInt(u32));
    std.debug.assert(options.versions.format[0] < 0x8000); // the high bit means "targets macOS"
    std.debug.assert(options.versions.platform[1] <= std.math.maxInt(u8));
    std.debug.assert(options.versions.platform[2] <= std.math.maxInt(u8));
    for (functions) |f| {
        std.debug.assert(f.name.len >= 1 and f.name.len <= max_name_len);
        std.debug.assert(std.mem.indexOfScalar(u8, f.name, 0) == null);
    }

    var digest: [32]u8 = undefined;
    Sha256.hash(module, &digest, .{});

    const list_size = functionListSize(functions);
    const extension_size = extensionSize(options.uuid);
    const metadata_size = metadataSize(functions.len);
    const public_md_offset = header_size + function_count_size + list_size + extension_size;
    const private_md_offset = public_md_offset + metadata_size;
    const module_list_offset = private_md_offset + metadata_size;

    const image = try gpa.alloc(u8, module_list_offset + module.len);
    errdefer gpa.free(image);
    var w = ImageWriter{ .bytes = image };

    w.writeAll(magic);
    w.writeInt(u16, options.versions.format[0] | 0x8000); // bit 15: targets macOS
    w.writeInt(u16, options.versions.format[1]);
    w.writeInt(u16, options.versions.format[2]);
    w.writeByte(file_type_executable);
    w.writeByte(platform_type_macos | 0x80); // bit 15: 64-bit
    w.writeInt(u16, options.versions.platform[0]);
    w.writeByte(@intCast(options.versions.platform[1]));
    w.writeByte(@intCast(options.versions.platform[2]));
    w.writeInt(u64, image.len);
    w.writeInt(u64, header_size);
    w.writeInt(u64, list_size);
    w.writeInt(u64, public_md_offset);
    w.writeInt(u64, metadata_size);
    w.writeInt(u64, private_md_offset);
    w.writeInt(u64, metadata_size);
    w.writeInt(u64, module_list_offset);
    w.writeInt(u64, module.len);
    std.debug.assert(w.pos == header_size);

    w.writeInt(u32, @intCast(functions.len));
    for (functions, 0..) |f, i| {
        // Function i owns metadata group i: 8 bytes into each metadata section.
        const metadata_offset = metadata_group_size * @as(u64, i);
        writeFunctionGroup(&w, f, options.versions, digest, module.len, metadata_offset);
    }

    // Header extension: the UUID of the module hash, then ENDT.
    if (options.uuid) {
        const id = contentUuid(digest);
        w.writeRecordHeader("UUID", 16);
        w.writeInt(u64, @truncate(id >> 64)); // high word first
        w.writeInt(u64, @truncate(id));
    }
    w.writeAll("ENDT");

    for (functions) |_| w.writeEmptyGroup(); // public metadata
    for (functions) |_| w.writeEmptyGroup(); // private metadata

    w.writeAll(module);
    std.debug.assert(w.pos == image.len);
    return image;
}

/// Apple's 20-byte module section header ("0b17c0de", u32 0, u32 0x14, u32 bitcode_size,
/// i32 -1) followed by `bitcode`. Metal's compiler stores the header inside the metallib;
/// stripping it made our modules fail to load (`doc/proposals/metal.md` section 5, "Keep the
/// module-section header").
pub fn wrapModule(gpa: Allocator, bitcode: []const u8) Allocator.Error![]u8 {
    std.debug.assert(bitcode.len <= std.math.maxInt(u32)); // the wrapper's size field is a u32
    const wrapped = try gpa.alloc(u8, module_header_size + bitcode.len);
    std.mem.writeInt(u32, wrapped[0..4], module_header_magic, .little);
    std.mem.writeInt(u32, wrapped[4..8], 0, .little);
    std.mem.writeInt(u32, wrapped[8..12], @intCast(module_header_size), .little);
    std.mem.writeInt(u32, wrapped[12..16], @intCast(bitcode.len), .little);
    std.mem.writeInt(i32, wrapped[16..20], -1, .little);
    @memcpy(wrapped[module_header_size..], bitcode);
    return wrapped;
}

/// Byte count of the function-list groups, which is what the header's
/// function-list size field holds; the function count is not part of it.
fn functionListSize(functions: []const Function) u64 {
    var total: u64 = 0;
    for (functions) |f| total += functionGroupSize(f.name.len);
    return total;
}

/// Byte count of the header extension: the optional UUID record and the ENDT.
fn extensionSize(uuid: bool) u64 {
    return (if (uuid) uuid_record_size else 0) + endt_size;
}

/// Byte count of one metadata section: one empty group per function.
fn metadataSize(function_count: usize) u64 {
    return metadata_group_size * @as(u64, function_count);
}

/// Size field of one function's tag group: the field itself, the NAME, TYPE,
/// HASH, OFFT, VERS and MDSZ records, and the ENDT.
fn functionGroupSize(name_len: usize) u64 {
    const name_record = record_header_size + name_len + 1; // the NUL is part of the value
    const type_record = record_header_size + 1;
    const hash_record = record_header_size + 32;
    const offt_record = record_header_size + 3 * 8;
    const vers_record = record_header_size + 4 * 2;
    const mdsz_record = record_header_size + 8;
    return group_size_size +
        name_record + type_record + hash_record + offt_record + vers_record + mdsz_record +
        endt_size;
}

/// The records of one function's tag group, in Apple's order. `metadata_offset`
/// points at this function's own empty group in both metadata sections, and the
/// module offset is 0 because every function names the module bytes at offset 0
/// of the module list.
fn writeFunctionGroup(
    w: *ImageWriter,
    f: Function,
    versions: Versions,
    digest: [32]u8,
    module_len: usize,
    metadata_offset: u64,
) void {
    w.writeInt(u32, @intCast(functionGroupSize(f.name.len)));
    w.writeRecordHeader("NAME", @intCast(f.name.len + 1));
    w.writeAll(f.name);
    w.writeByte(0);
    w.writeRecordHeader("TYPE", 1);
    w.writeByte(if (f.is_kernel) kernel_type else function_type);
    w.writeRecordHeader("HASH", 32);
    w.writeAll(&digest);
    w.writeRecordHeader("OFFT", 24);
    w.writeInt(u64, metadata_offset);
    w.writeInt(u64, metadata_offset);
    w.writeInt(u64, 0);
    w.writeRecordHeader("VERS", 8);
    w.writeInt(u16, versions.air[0]);
    w.writeInt(u16, versions.air[1]);
    w.writeInt(u16, versions.metal[0]);
    w.writeInt(u16, versions.metal[1]);
    w.writeRecordHeader("MDSZ", 8);
    w.writeInt(u64, module_len);
    w.writeAll("ENDT");
}

/// The UUID a module hash implies: its first 16 bytes read big-endian as a
/// `u128`, with the RFC 4122 version nibble set to 4 and the variant bits set to
/// binary 10. Apple's UUIDs are random; deriving one from the content keeps our
/// output reproducible.
fn contentUuid(digest: [32]u8) u128 {
    var id = std.mem.readInt(u128, digest[0..16], .big);
    id = (id & ~(@as(u128, 0xf) << 76)) | (@as(u128, 0x4) << 76);
    id = (id & ~(@as(u128, 0x3) << 62)) | (@as(u128, 0x2) << 62);
    return id;
}

/// Little-endian cursor over the image being assembled. The buffer is sized by
/// the arithmetic above and must end up exactly full: a size that disagrees with
/// the records panics on a slice bound here in safe builds, and the tests pin the
/// arithmetic directly.
const ImageWriter = struct {
    bytes: []u8,
    pos: usize = 0,

    fn writeAll(w: *ImageWriter, bytes: []const u8) void {
        @memcpy(w.bytes[w.pos..][0..bytes.len], bytes);
        w.pos += bytes.len;
    }

    fn writeByte(w: *ImageWriter, byte: u8) void {
        w.bytes[w.pos] = byte;
        w.pos += 1;
    }

    fn writeInt(w: *ImageWriter, comptime T: type, value: T) void {
        std.mem.writeInt(T, w.bytes[w.pos..][0..@sizeOf(T)], value, .little);
        w.pos += @sizeOf(T);
    }

    /// A record's `<4-byte tag><u16 value_size>` header.
    fn writeRecordHeader(w: *ImageWriter, tag: *const [4]u8, value_size: u16) void {
        w.writeAll(tag);
        w.writeInt(u16, value_size);
    }

    fn writeEmptyGroup(w: *ImageWriter) void {
        w.writeInt(u32, @intCast(metadata_group_size));
        w.writeAll("ENDT");
    }
};

// ---------------------------------------------------------------------------
// the reader's API
// ---------------------------------------------------------------------------

/// Everything the reader recovers from an image, for tests and diagnostics.
///
/// Every slice borrows the bytes passed to `parse`, including each function's
/// name and `module`; only `functions` is allocated, so those bytes must stay
/// alive for as long as the image is in use (which is natural: they are the
/// library itself).
pub const Image = struct {
    /// Container file version: the `u16` triple after the `MTLB` magic, with the
    /// "targets macOS" bit stripped from the major.
    format: [3]u16,
    /// The major word's high bit: the library targets macOS.
    is_macos: bool,
    /// 0 executable (what `write` stamps), 1 core image, 2 dynamic, 3 symbol
    /// companion.
    file_type: u8,
    /// The file-type byte's high bit: a stub library.
    is_stub: bool,
    /// 1 macOS, 2 iOS, 3 tvOS, 4 watchOS, ...; 0 in files that predate the field.
    platform_type: u8,
    /// The platform-type byte's high bit: 64-bit.
    is_64bit: bool,
    /// Deployment version from the header (major `u16`, minor and patch `u8`).
    platform: [3]u16,
    /// Total file size from the header; `parse` requires it to equal the input
    /// length.
    file_size: u64,
    /// The header's four (offset, size) pairs, in header order.
    sections: Sections,
    /// One entry per group of the function list, in function-list order.
    functions: []FunctionInfo,
    /// The UUID record's value, when the header extension carries one.
    uuid: ?u128,
    /// The module-list section: the module bytes verbatim, with no framing of
    /// their own. With one module (what `write` emits) this is the module.
    module: []const u8,
    /// True when `module` starts with Apple's 0x0b17c0de module-section header
    /// that `wrapModule` writes and the runtime requires. False means the module
    /// was stored bare, which is a bug on the writer's side: a library whose
    /// module lacked the header was rejected by
    /// `newComputePipelineStateWithFunction` ("unable to copy bitcode for
    /// function", macOS 26.6.2).
    module_header: bool,
};

/// One (offset, size) pair from the header: an absolute file offset and a byte
/// count.
pub const Section = struct {
    offset: u64,
    size: u64,
};

/// The header's section table.
pub const Sections = struct {
    /// The function tag groups; `size` excludes the leading function count.
    function_list: Section,
    /// One group per function, in function order.
    public_md: Section,
    /// One group per function, in function order.
    private_md: Section,
    /// The module bytes, concatenated and unframed.
    module_list: Section,
};

/// One function recovered from the function list.
pub const FunctionInfo = struct {
    /// NAME with the stored NUL stripped; a slice of the parsed bytes.
    name: []const u8,
    /// TYPE: true when the value is 2 (`PROGRAM_KERNEL`), false for 0 and for any
    /// other value a library of graphics functions might carry.
    is_kernel: bool,
    /// HASH: the SHA-256 that `parse` verified over the module bytes.
    hash: [32]u8,
    /// OFFT: where this function's public metadata, private metadata and module
    /// live, relative to their sections.
    offsets: Offsets,
    /// VERS: AIR major and minor, then 0 for the patch VERS does not carry.
    air: [3]u16,
    /// VERS: Metal major and minor, then 0 for the patch VERS does not carry.
    metal: [3]u16,
    /// MDSZ: byte count of the module bytes this function names.
    module_size: u64,
    /// The module bytes themselves, at `offsets.module` inside the module list.
    module_bytes: []const u8,
};

/// OFFT: three offsets, each relative to its own section.
pub const Offsets = struct {
    public_md: u64,
    private_md: u64,
    module: u64,
};

pub const ParseError = error{
    /// The input is not a well-formed image: wrong magic, truncated, a section or
    /// module range outside the file, a size that disagrees with what the file
    /// actually contains, or a HASH that is not the SHA-256 of its module.
    Mismatch,
    OutOfMemory,
};

/// Reads an image back: header fields, the function list with every record the
/// writer stamps, the module bytes, and the section table for diagnostics.
/// Anything that disagrees with the bytes actually present is `error.Mismatch`, a
/// HASH that does not match its module included. Two shapes found in other
/// writers' output are reported rather than rejected: a module stored without
/// Apple's header (`Image.module_header`), and an empty metadata group whose size
/// field omits its `ENDT` (Apple's vadd.metallib has one; Xcode's other fixtures
/// declare the full size, as `write` does).
pub fn parse(gpa: Allocator, bytes: []const u8) ParseError!Image {
    if (bytes.len < header_size) return error.Mismatch;
    if (!std.mem.eql(u8, bytes[0..4], magic)) return error.Mismatch;

    var c = Cursor{ .bytes = bytes, .pos = magic.len };
    const major_word = try c.readInt(u16);
    const is_macos = major_word & 0x8000 != 0;
    const format_major = major_word & 0x7fff;
    const format_minor = try c.readInt(u16);
    const format_patch = try c.readInt(u16);
    const format = [3]u16{ format_major, format_minor, format_patch };
    const file_type_byte = try c.readInt(u8);
    const platform_type_byte = try c.readInt(u8);
    const platform_major = try c.readInt(u16);
    const platform_minor = try c.readInt(u8);
    const platform_patch = try c.readInt(u8);
    const platform = [3]u16{ platform_major, platform_minor, platform_patch };
    const file_size = try c.readInt(u64);
    const function_list = try c.readSection();
    const public_md = try c.readSection();
    const private_md = try c.readSection();
    const module_list = try c.readSection();
    std.debug.assert(c.pos == header_size);
    const sections = Sections{
        .function_list = function_list,
        .public_md = public_md,
        .private_md = private_md,
        .module_list = module_list,
    };

    if (file_size != bytes.len) return error.Mismatch;
    if (!inBounds(function_list, bytes.len) or !inBounds(public_md, bytes.len) or
        !inBounds(private_md, bytes.len) or !inBounds(module_list, bytes.len))
    {
        return error.Mismatch;
    }
    const list_end = function_list.offset + function_count_size + function_list.size;
    if (function_list.offset < header_size) return error.Mismatch;
    if (list_end > bytes.len) return error.Mismatch;
    // The list follows the header, and the metadata sections and the modules
    // follow it in order.
    if (public_md.offset < list_end) return error.Mismatch;
    if (private_md.offset != public_md.offset + public_md.size) return error.Mismatch;
    if (module_list.offset != private_md.offset + private_md.size) return error.Mismatch;

    // ------------------------------------------------------------- function list
    c.pos = @intCast(function_list.offset);
    const function_count = try c.readInt(u32);
    if (function_count > max_functions) return error.Mismatch;
    const functions = try gpa.alloc(FunctionInfo, function_count);
    errdefer gpa.free(functions);
    for (functions) |*f| f.* = try parseFunction(&c);
    if (@as(u64, c.pos) != list_end) return error.Mismatch;

    // ---------------------------------------------------------- header extension
    // Only from format 1.2.3 on; before that the metadata directly follows the
    // function list. The extension has no size field: records, then ENDT.
    var uuid: ?u128 = null;
    if (versionAtLeast(format, 1, 2, 3)) {
        while (true) {
            const tag = try c.takeArray(4);
            if (std.mem.eql(u8, tag, "ENDT")) break;
            const value = try c.take(try c.readInt(u16));
            if (std.mem.eql(u8, tag, "UUID")) {
                if (value.len != 16) return error.Mismatch;
                const high = std.mem.readInt(u64, value[0..8], .little);
                const low = std.mem.readInt(u64, value[8..16], .little);
                uuid = (@as(u128, high) << 64) | low;
            }
        }
        if (@as(u64, c.pos) != public_md.offset) return error.Mismatch;
    } else if (public_md.offset != list_end) return error.Mismatch;

    try parseMetadataSection(&c, public_md, functions, .public_md);
    try parseMetadataSection(&c, private_md, functions, .private_md);

    // -------------------------------------------------------- modules and hashes
    const module_start: usize = @intCast(module_list.offset);
    const module = bytes[module_start..][0..@intCast(module_list.size)];
    for (functions) |*f| {
        if (f.offsets.module > module_list.size or
            f.module_size > module_list.size - f.offsets.module)
        {
            return error.Mismatch;
        }
        const start = module_start + @as(usize, @intCast(f.offsets.module));
        const end = start + @as(usize, @intCast(f.module_size));
        f.module_bytes = bytes[start..end];
        var digest: [32]u8 = undefined;
        Sha256.hash(f.module_bytes, &digest, .{});
        if (!std.mem.eql(u8, &digest, &f.hash)) return error.Mismatch;
    }
    // The function module ranges must cover the module list, but may overlap: a
    // library that names several kernels of one LLVM module has one group per
    // name, all with module offset 0 and the same MDSZ. Walking the union's
    // reachable prefix from 0 finds whether anything is left unreferenced.
    var covered: u64 = 0;
    while (true) {
        var extended = false;
        for (functions) |f| {
            const end = f.offsets.module + f.module_size;
            if (f.offsets.module <= covered and end > covered) {
                covered = end;
                extended = true;
            }
        }
        if (!extended) break;
    }
    if (covered != module_list.size) return error.Mismatch;

    return .{
        .format = format,
        .is_macos = is_macos,
        .file_type = file_type_byte & 0x7f,
        .is_stub = file_type_byte & 0x80 != 0,
        .platform_type = platform_type_byte & 0x7f,
        .is_64bit = platform_type_byte & 0x80 != 0,
        .platform = platform,
        .file_size = file_size,
        .sections = sections,
        .functions = functions,
        .uuid = uuid,
        .module = module,
        .module_header = module.len >= module_header_size and
            std.mem.readInt(u32, module[0..4], .little) == module_header_magic,
    };
}

/// Releases what `parse` allocated; the image borrows the parsed bytes, so it
/// must not be used afterwards.
pub fn deinit(image: *Image, gpa: Allocator) void {
    gpa.free(image.functions);
    image.* = undefined;
}

/// Which metadata section is being walked, and therefore which OFFT field each
/// group's position must match.
const MetadataKind = enum { public_md, private_md };

/// Decodes one function's tag group: its size field, the records the writer
/// stamps and the ENDT. NAME, TYPE, HASH, OFFT, VERS and MDSZ must all be
/// present with exactly the sizes the format gives them; records the reader does
/// not model are skipped, and the group must occupy exactly the size it declares.
///
/// RBUF and SBUF -- the only records whose value size is a `u32` rather than a
/// `u16` -- live in the reflection and script sections, which this reader does
/// not walk (it reads the function list, the header extension and the two
/// metadata sections), so every record it sees is `u16`-sized.
fn parseFunction(c: *Cursor) ParseError!FunctionInfo {
    const group_start = c.pos;
    const declared_size = try c.readInt(u32);
    var name: ?[]const u8 = null;
    var is_kernel = false;
    var hash: ?[32]u8 = null;
    var offsets: ?Offsets = null;
    var vers: ?[4]u16 = null;
    var module_size: ?u64 = null;
    while (true) {
        const tag = try c.takeArray(4);
        if (std.mem.eql(u8, tag, "ENDT")) break;
        const value = try c.take(try c.readInt(u16));
        if (std.mem.eql(u8, tag, "NAME")) {
            const nul = std.mem.indexOfScalar(u8, value, 0);
            name = value[0 .. nul orelse value.len];
        } else if (std.mem.eql(u8, tag, "TYPE")) {
            if (value.len != 1) return error.Mismatch;
            is_kernel = value[0] == kernel_type;
        } else if (std.mem.eql(u8, tag, "HASH")) {
            if (value.len != 32) return error.Mismatch;
            hash = value[0..32].*;
        } else if (std.mem.eql(u8, tag, "OFFT")) {
            if (value.len != 24) return error.Mismatch;
            offsets = .{
                .public_md = std.mem.readInt(u64, value[0..8], .little),
                .private_md = std.mem.readInt(u64, value[8..16], .little),
                .module = std.mem.readInt(u64, value[16..24], .little),
            };
        } else if (std.mem.eql(u8, tag, "VERS")) {
            if (value.len != 8) return error.Mismatch;
            vers = .{
                std.mem.readInt(u16, value[0..2], .little),
                std.mem.readInt(u16, value[2..4], .little),
                std.mem.readInt(u16, value[4..6], .little),
                std.mem.readInt(u16, value[6..8], .little),
            };
        } else if (std.mem.eql(u8, tag, "MDSZ")) {
            if (value.len != 8) return error.Mismatch;
            module_size = std.mem.readInt(u64, value[0..8], .little);
        }
    }
    if (c.pos - group_start != declared_size) return error.Mismatch;

    const v = vers orelse return error.Mismatch;
    return .{
        .name = name orelse return error.Mismatch,
        .is_kernel = is_kernel,
        .hash = hash orelse return error.Mismatch,
        .offsets = offsets orelse return error.Mismatch,
        .air = .{ v[0], v[1], 0 },
        .metal = .{ v[2], v[3], 0 },
        .module_size = module_size orelse return error.Mismatch,
        .module_bytes = &.{},
    };
}

/// Walks one metadata section: one size-prefixed group per function, in function
/// order, each starting where its OFFT says, filling the section exactly.
fn parseMetadataSection(
    c: *Cursor,
    section: Section,
    functions: []const FunctionInfo,
    comptime kind: MetadataKind,
) ParseError!void {
    const section_start: usize = @intCast(section.offset);
    c.pos = section_start;
    for (functions) |f| {
        const group_start = c.pos;
        try skipGroup(c);
        const offset: u64 = @intCast(group_start - section_start);
        const declared = switch (kind) {
            .public_md => f.offsets.public_md,
            .private_md => f.offsets.private_md,
        };
        if (declared != offset) return error.Mismatch;
    }
    if (@as(u64, c.pos - section_start) != section.size) return error.Mismatch;
}

/// Skips one size-prefixed group: its size field, its records, its ENDT.
fn skipGroup(c: *Cursor) ParseError!void {
    const start = c.pos;
    const declared_size = try c.readInt(u32);
    var records: usize = 0;
    while (true) {
        const tag = try c.takeArray(4);
        if (std.mem.eql(u8, tag, "ENDT")) break;
        _ = try c.take(try c.readInt(u16));
        records += 1;
    }
    const consumed = c.pos - start;
    if (consumed == declared_size) return;
    // Apple's vadd.metallib contains an empty group whose size field omits its
    // ENDT (declares 4, occupies 8); Xcode's other fixtures declare 8, as we do.
    if (records == 0 and consumed == @as(usize, declared_size) + endt_size) return;
    return error.Mismatch;
}

fn inBounds(section: Section, len: usize) bool {
    return section.offset <= len and section.size <= len - section.offset;
}

fn versionAtLeast(version: [3]u16, major: u16, minor: u16, patch: u16) bool {
    if (version[0] != major) return version[0] > major;
    if (version[1] != minor) return version[1] > minor;
    return version[2] >= patch;
}

/// Bounds-checked little-endian cursor. The invariant `pos <= bytes.len` holds
/// after every method here, and after every repositioning elsewhere in this file,
/// because those offsets are validated against the input first.
const Cursor = struct {
    bytes: []const u8,
    pos: usize,

    fn take(c: *Cursor, n: usize) ParseError![]const u8 {
        if (n > c.bytes.len - c.pos) return error.Mismatch;
        const taken = c.bytes[c.pos..][0..n];
        c.pos += n;
        return taken;
    }

    fn takeArray(c: *Cursor, comptime n: usize) ParseError!*const [n]u8 {
        return (try c.take(n))[0..n];
    }

    fn readInt(c: *Cursor, comptime T: type) ParseError!T {
        return std.mem.readInt(T, try c.takeArray(@sizeOf(T)), .little);
    }

    fn readSection(c: *Cursor) ParseError!Section {
        const offset = try c.readInt(u64);
        const size = try c.readInt(u64);
        return .{ .offset = offset, .size = size };
    }
};

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// A stand-in module payload. Real bitcode begins with LLVM's `BC\xc0\xde`
/// magic, which is all this needs to look like one; the container stores it
/// verbatim, hash and all.
const synthetic_bitcode = "BC\xc0\xde synthetic metallib payload, not a real module";

const test_versions = Versions{
    .air = .{ 2, 8, 0 },
    .metal = .{ 4, 0, 0 },
    .format = .{ 1, 2, 9 },
    .platform = .{ 26, 0, 0 },
};

/// Byte offset of the value of the first `tag` record in `image`, so a test can
/// corrupt one field the way a broken writer would.
fn recordValueOffset(image: []const u8, tag: *const [4]u8) usize {
    const at = std.mem.indexOf(u8, image, tag).?;
    return at + record_header_size;
}

fn readU16(image: []const u8, at: usize) u16 {
    return std.mem.readInt(u16, image[at..][0..2], .little);
}

fn readU32(image: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, image[at..][0..4], .little);
}

fn readU64(image: []const u8, at: usize) u64 {
    return std.mem.readInt(u64, image[at..][0..8], .little);
}

/// Checks a written image field by field: the raw header bytes at the offsets the
/// format fixes, the section arithmetic, the records of every function, and the
/// module bytes stored verbatim at the tail.
fn expectImageEquals(
    gpa: Allocator,
    image: []const u8,
    module: []const u8,
    functions: []const Function,
    versions: Versions,
    with_uuid: bool,
) !void {
    var parsed = try parse(gpa, image);
    defer deinit(&parsed, gpa);

    var digest: [32]u8 = undefined;
    Sha256.hash(module, &digest, .{});

    // Header, at the byte offsets the container fixes.
    try testing.expectEqualSlices(u8, "MTLB", image[0..4]);
    try testing.expectEqual(@as(u16, versions.format[0] | 0x8000), readU16(image, 0x04));
    try testing.expectEqual(versions.format[1], readU16(image, 0x06));
    try testing.expectEqual(versions.format[2], readU16(image, 0x08));
    try testing.expectEqual(@as(u8, 0), image[0x0a]);
    try testing.expectEqual(@as(u8, 0x81), image[0x0b]);
    try testing.expectEqual(versions.platform[0], readU16(image, 0x0c));
    try testing.expectEqual(@as(u8, @intCast(versions.platform[1])), image[0x0e]);
    try testing.expectEqual(@as(u8, @intCast(versions.platform[2])), image[0x0f]);
    try testing.expectEqual(@as(u64, @intCast(image.len)), readU64(image, 0x10));

    // The four section pairs: contiguity, the sizes the layout implies, and the
    // extension in the gap between the function list and the public metadata.
    const list_size = functionListSize(functions);
    const extension_size = extensionSize(with_uuid);
    const metadata_size = metadataSize(functions.len);
    try testing.expectEqual(@as(u64, header_size), readU64(image, 0x18));
    try testing.expectEqual(list_size, readU64(image, 0x20));
    try testing.expectEqual(
        readU64(image, 0x18) + function_count_size + list_size + extension_size,
        readU64(image, 0x28),
    );
    try testing.expectEqual(metadata_size, readU64(image, 0x30));
    try testing.expectEqual(readU64(image, 0x28) + metadata_size, readU64(image, 0x38));
    try testing.expectEqual(metadata_size, readU64(image, 0x40));
    try testing.expectEqual(readU64(image, 0x38) + metadata_size, readU64(image, 0x48));
    try testing.expectEqual(@as(u64, @intCast(module.len)), readU64(image, 0x50));

    // What the reader recovered from the same bytes.
    try testing.expectEqualSlices(u16, &versions.format, &parsed.format);
    try testing.expect(parsed.is_macos);
    try testing.expectEqual(@as(u8, 0), parsed.file_type);
    try testing.expect(!parsed.is_stub);
    try testing.expectEqual(@as(u8, 1), parsed.platform_type);
    try testing.expect(parsed.is_64bit);
    try testing.expectEqualSlices(u16, &versions.platform, &parsed.platform);
    try testing.expectEqual(@as(u64, @intCast(image.len)), parsed.file_size);
    try testing.expectEqual(@as(u64, header_size), parsed.sections.function_list.offset);
    try testing.expectEqual(list_size, parsed.sections.function_list.size);
    try testing.expectEqual(readU64(image, 0x28), parsed.sections.public_md.offset);
    try testing.expectEqual(metadata_size, parsed.sections.public_md.size);
    try testing.expectEqual(readU64(image, 0x38), parsed.sections.private_md.offset);
    try testing.expectEqual(metadata_size, parsed.sections.private_md.size);
    try testing.expectEqual(readU64(image, 0x48), parsed.sections.module_list.offset);
    try testing.expectEqual(@as(u64, @intCast(module.len)), parsed.sections.module_list.size);

    try testing.expectEqual(functions.len, parsed.functions.len);
    for (functions, 0..) |f, i| {
        const g = parsed.functions[i];
        try testing.expectEqualStrings(f.name, g.name);
        try testing.expectEqual(f.is_kernel, g.is_kernel);
        try testing.expectEqualSlices(u8, &digest, &g.hash);
        // Every function names the one module: offsets 0, one shared pair of
        // metadata groups each (function i owns group i).
        try testing.expectEqual(@as(u64, 0), g.offsets.module);
        try testing.expectEqual(metadata_group_size * @as(u64, i), g.offsets.public_md);
        try testing.expectEqual(metadata_group_size * @as(u64, i), g.offsets.private_md);
        try testing.expectEqualSlices(u16, &.{ versions.air[0], versions.air[1], 0 }, &g.air);
        try testing.expectEqualSlices(u16, &.{ versions.metal[0], versions.metal[1], 0 }, &g.metal);
        try testing.expectEqual(@as(u64, @intCast(module.len)), g.module_size);
        try testing.expectEqualSlices(u8, module, g.module_bytes);
        try testing.expectEqual(@as(u64, @intCast(image.len)), g.offsets.module + parsed.sections.module_list.offset + g.module_size);
        // OFFT points at this function's own empty group in each metadata section.
        const pub_group: usize = @intCast(parsed.sections.public_md.offset + g.offsets.public_md);
        const priv_group: usize = @intCast(parsed.sections.private_md.offset + g.offsets.private_md);
        try testing.expectEqual(@as(u32, @intCast(metadata_group_size)), readU32(image, pub_group));
        try testing.expectEqual(@as(u32, @intCast(metadata_group_size)), readU32(image, priv_group));
        try testing.expectEqualSlices(u8, "ENDT", image[pub_group + 4 ..][0..4]);
        try testing.expectEqualSlices(u8, "ENDT", image[priv_group + 4 ..][0..4]);
    }
    try testing.expectEqualSlices(u8, module, parsed.module);

    const extension_start: usize = @intCast(parsed.sections.public_md.offset - extension_size);
    if (with_uuid) {
        const id = contentUuid(digest);
        try testing.expectEqual(id, parsed.uuid.?);
        try testing.expectEqualSlices(u8, "UUID", image[extension_start..][0..4]);
        try testing.expectEqual(@as(u16, 16), readU16(image, extension_start + 4));
        try testing.expectEqual(@as(u64, @truncate(id >> 64)), readU64(image, extension_start + 6));
        try testing.expectEqual(@as(u64, @truncate(id)), readU64(image, extension_start + 14));
    } else {
        try testing.expectEqual(@as(?u128, null), parsed.uuid);
        try testing.expectEqualSlices(u8, "ENDT", image[extension_start..][0..4]);
    }
    try testing.expectEqual(module.len >= module_header_size and
        readU32(module, 0) == module_header_magic, parsed.module_header);
}

test "wrapModule writes Apple's 20-byte module-section header" {
    const gpa = testing.allocator;
    const wrapped = try wrapModule(gpa, synthetic_bitcode);
    defer gpa.free(wrapped);

    try testing.expectEqual(module_header_size + synthetic_bitcode.len, wrapped.len);
    try testing.expectEqual(@as(u32, 0x0b17c0de), std.mem.readInt(u32, wrapped[0..4], .little));
    try testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, wrapped[4..8], .little));
    try testing.expectEqual(@as(u32, 20), std.mem.readInt(u32, wrapped[8..12], .little));
    try testing.expectEqual(
        @as(u32, @intCast(synthetic_bitcode.len)),
        std.mem.readInt(u32, wrapped[12..16], .little),
    );
    try testing.expectEqual(@as(i32, -1), std.mem.readInt(i32, wrapped[16..20], .little));
    try testing.expectEqualSlices(u8, synthetic_bitcode, wrapped[module_header_size..]);

    // An empty payload still gets a well-formed header.
    const empty = try wrapModule(gpa, "");
    defer gpa.free(empty);
    try testing.expectEqual(module_header_size, empty.len);
    try testing.expectEqual(@as(u32, module_header_size), std.mem.readInt(u32, empty[8..12], .little));
    try testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, empty[12..16], .little));
}

test "write and parse round-trip a single function" {
    const gpa = testing.allocator;
    const functions = [_]Function{.{ .name = "synthetic" }};
    const image = try write(gpa, synthetic_bitcode, .{
        .functions = &functions,
        .versions = test_versions,
        .uuid = false, // the extension is then a bare ENDT
    });
    defer gpa.free(image);

    try expectImageEquals(gpa, image, synthetic_bitcode, &functions, test_versions, false);
    // A payload stored bare is reported, not rejected: the wrapper is required but
    // the reader is used on files other writers produced too.
    var parsed = try parse(gpa, image);
    defer deinit(&parsed, gpa);
    try testing.expect(!parsed.module_header);
}

test "write and parse round-trip several functions sharing one module" {
    const gpa = testing.allocator;
    const wrapped = try wrapModule(gpa, synthetic_bitcode);
    defer gpa.free(wrapped);

    const functions = [_]Function{
        .{ .name = "vadd" },
        .{ .name = "reduce" },
        .{ .name = "parsef" },
        .{ .name = "a_kernel_name_considerably_longer_than_the_others" },
    };
    // `uuid` defaults to on, derived from the module hash.
    const image = try write(gpa, wrapped, .{ .functions = &functions, .versions = test_versions });
    defer gpa.free(image);

    try expectImageEquals(gpa, image, wrapped, &functions, test_versions, true);
    var parsed = try parse(gpa, image);
    defer deinit(&parsed, gpa);
    try testing.expect(parsed.module_header);
    try testing.expectEqualStrings("a_kernel_name_considerably_longer_than_the_others", parsed.functions[3].name);
}

test "a name at the u16 limit round-trips" {
    const gpa = testing.allocator;
    // The NAME value size is a u16, so the longest name is one byte less, its NUL
    // included. This is the boundary that would truncate a long mangled symbol.
    const name = try gpa.alloc(u8, max_name_len);
    defer gpa.free(name);
    @memset(name, 'k');
    const functions = [_]Function{.{ .name = name }};
    const image = try write(gpa, synthetic_bitcode, .{ .functions = &functions, .versions = test_versions });
    defer gpa.free(image);

    try expectImageEquals(gpa, image, synthetic_bitcode, &functions, test_versions, true);
    var parsed = try parse(gpa, image);
    defer deinit(&parsed, gpa);
    try testing.expectEqualSlices(u8, name, parsed.functions[0].name);
    try testing.expectEqual(functionGroupSize(max_name_len), parsed.sections.function_list.size);
}

test "write stamps TYPE 0 for ordinary functions" {
    const gpa = testing.allocator;
    const functions = [_]Function{
        .{ .name = "kernel", .is_kernel = true },
        .{ .name = "helper", .is_kernel = false },
    };
    const image = try write(gpa, synthetic_bitcode, .{ .functions = &functions, .versions = test_versions });
    defer gpa.free(image);

    var parsed = try parse(gpa, image);
    defer deinit(&parsed, gpa);
    try testing.expect(parsed.functions[0].is_kernel);
    try testing.expect(!parsed.functions[1].is_kernel);

    // The raw TYPE records: `<tag><u16 1><value>`.
    const first = recordValueOffset(image, "TYPE");
    try testing.expectEqual(@as(u16, 1), readU16(image, first - 2));
    try testing.expectEqual(kernel_type, image[first]);
    const second = std.mem.indexOfPos(u8, image, first + 1, "TYPE").?;
    try testing.expectEqual(function_type, image[second + record_header_size]);
}

test "VERS carries the AIR and Metal major and minor versions only" {
    const gpa = testing.allocator;
    // Non-zero patches must not leak into the records, and the platform version's
    // minor and patch are one byte each in the header.
    const versions = Versions{
        .air = .{ 2, 8, 3 },
        .metal = .{ 4, 0, 7 },
        .format = .{ 1, 2, 9 },
        .platform = .{ 26, 1, 2 },
    };
    const functions = [_]Function{.{ .name = "k" }};
    const image = try write(gpa, synthetic_bitcode, .{ .functions = &functions, .versions = versions });
    defer gpa.free(image);

    try expectImageEquals(gpa, image, synthetic_bitcode, &functions, versions, true);

    const vers = recordValueOffset(image, "VERS");
    try testing.expectEqual(@as(u16, 8), readU16(image, vers - 2));
    try testing.expectEqual(@as(u16, 2), readU16(image, vers));
    try testing.expectEqual(@as(u16, 8), readU16(image, vers + 2));
    try testing.expectEqual(@as(u16, 4), readU16(image, vers + 4));
    try testing.expectEqual(@as(u16, 0), readU16(image, vers + 6));
}

test "parse rejects truncated images" {
    const gpa = testing.allocator;
    const functions = [_]Function{.{ .name = "synthetic" }};
    const image = try write(gpa, synthetic_bitcode, .{ .functions = &functions, .versions = test_versions });
    defer gpa.free(image);

    // Every prefix either lacks the header or contradicts its file size, its
    // section table, a group size or a module range. Nothing may read past the end.
    var len: usize = 0;
    while (len < image.len) : (len += 1) {
        try testing.expectError(error.Mismatch, parse(gpa, image[0..len]));
    }
}

test "parse rejects corrupt images" {
    const gpa = testing.allocator;
    const wrapped = try wrapModule(gpa, synthetic_bitcode);
    defer gpa.free(wrapped);
    const functions = [_]Function{.{ .name = "synthetic" }};
    const image = try write(gpa, wrapped, .{ .functions = &functions, .versions = test_versions });
    defer gpa.free(image);
    var parsed = try parse(gpa, image); // the pristine image is valid
    deinit(&parsed, gpa);

    const scratch = try gpa.dupe(u8, image);
    defer gpa.free(scratch);
    const name_value = recordValueOffset(image, "NAME");
    const hash_value = recordValueOffset(image, "HASH");
    const offt_value = recordValueOffset(image, "OFFT");
    const vers_value = recordValueOffset(image, "VERS");
    const mdsz_value = recordValueOffset(image, "MDSZ");

    const corruptions = [_]struct { what: []const u8, at: usize, value: u64, size: u8 }{
        .{ .what = "magic", .at = 0, .value = 'X', .size = 1 },
        .{ .what = "file size", .at = 0x10, .value = image.len + 1, .size = 8 },
        .{ .what = "function list offset", .at = 0x18, .value = 0, .size = 8 },
        .{ .what = "public metadata size", .at = 0x30, .value = metadata_group_size - 1, .size = 8 },
        .{ .what = "module list size", .at = 0x50, .value = wrapped.len + 1, .size = 8 },
        .{ .what = "function count", .at = header_size, .value = 0, .size = 4 },
        .{ .what = "function count over the sanity bound", .at = header_size, .value = std.math.maxInt(u16) + 1, .size = 4 },
        .{ .what = "group size", .at = header_size + function_count_size, .value = 1, .size = 4 },
        .{ .what = "NAME value size", .at = name_value - 2, .value = 0xffff, .size = 2 },
        .{ .what = "OFFT public metadata offset", .at = offt_value, .value = 1, .size = 8 },
        .{ .what = "VERS value size", .at = vers_value - 2, .value = 2, .size = 2 },
        .{ .what = "MDSZ", .at = mdsz_value, .value = std.math.maxInt(u64), .size = 8 },
    };
    for (corruptions) |corruption| {
        @memcpy(scratch, image);
        var value: [8]u8 = @splat(0);
        std.mem.writeInt(u64, &value, corruption.value, .little);
        @memcpy(scratch[corruption.at..][0..corruption.size], value[0..corruption.size]);
        try testing.expectError(error.Mismatch, parse(gpa, scratch));
    }

    // A flipped byte in the stored HASH no longer describes the module, and one in
    // the module no longer matches the stored HASH.
    @memcpy(scratch, image);
    scratch[hash_value] ^= 0xff;
    try testing.expectError(error.Mismatch, parse(gpa, scratch));
    @memcpy(scratch, image);
    scratch[image.len - 1] ^= 0xff;
    try testing.expectError(error.Mismatch, parse(gpa, scratch));
}

test "parse rejects garbage" {
    const gpa = testing.allocator;
    try testing.expectError(error.Mismatch, parse(gpa, ""));
    try testing.expectError(error.Mismatch, parse(gpa, "MTL"));
    var wrong_magic: [header_size]u8 = @splat(0);
    @memcpy(wrong_magic[0..4], "XTLB");
    try testing.expectError(error.Mismatch, parse(gpa, &wrong_magic));

    var prng = std.Random.DefaultPrng.init(0x6d6574616c6c6962);
    var buf: [256]u8 = undefined;
    var len: usize = 0;
    while (len <= buf.len) : (len += 8) {
        prng.random().bytes(buf[0..len]);
        try testing.expectError(error.Mismatch, parse(gpa, buf[0..len]));
    }
}

test "parse accepts the empty metadata group variant Apple writes" {
    const gpa = testing.allocator;
    const functions = [_]Function{.{ .name = "kernel" }};
    const image = try write(gpa, synthetic_bitcode, .{ .functions = &functions, .versions = test_versions });
    defer gpa.free(image);

    {
        var parsed = try parse(gpa, image);
        defer deinit(&parsed, gpa);
        try testing.expectEqualSlices(u8, "kernel", parsed.functions[0].name);
    }
    // Apple's vadd.metallib declares 4 for an empty group that occupies 8 bytes;
    // Xcode's other fixtures declare 8, as this writer does.
    const group: usize = @intCast(readU64(image, 0x28));
    try testing.expectEqual(@as(u32, @intCast(metadata_group_size)), readU32(image, group));
    std.mem.writeInt(u32, image[group..][0..4], @intCast(group_size_size), .little);

    var patched = try parse(gpa, image);
    defer deinit(&patched, gpa);
    try testing.expectEqualSlices(u8, "kernel", patched.functions[0].name);
    try testing.expectEqual(@as(u32, @intCast(group_size_size)), readU32(image, group));
}
