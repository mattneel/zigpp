//! The `air64` target's "linker". A Metal library needs no system linker: the LLVM backend has
//! already emitted the whole `.metallib` — the module wrapped in Apple's container — and this
//! backend only hands those bytes to the output file, so that the ordinary object pipeline can
//! produce `-femit-bin=foo.metallib` (`doc/proposals/metal.md` section 6.2 item 4). The SPIR-V
//! backend plays the same role for SPIR-V modules.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const Path = std.Build.Cache.Path;
const Zcu = @import("../Zcu.zig");
const Compilation = @import("../Compilation.zig");
const link = @import("../link.zig");

const Linker = @This();

base: link.File,
/// Whether the LLVM backend has handed the emitted library to `loadInput` yet.
written: bool = false,

pub fn open(
    arena: Allocator,
    comp: *Compilation,
    emit: Path,
    options: link.File.OpenOptions,
) !*Linker {
    return createEmpty(arena, comp, emit, options);
}

pub fn createEmpty(
    arena: Allocator,
    comp: *Compilation,
    emit: Path,
    options: link.File.OpenOptions,
) !*Linker {
    const io = comp.io;
    const linker = try arena.create(Linker);
    linker.* = .{ .base = .{
        .tag = .metallib,
        .comp = comp,
        .emit = emit,
        .gc_sections = options.gc_sections orelse false,
        .print_gc_sections = options.print_gc_sections,
        .stack_size = options.stack_size orelse 0,
        .allow_shlib_undefined = options.allow_shlib_undefined orelse false,
        .file = try emit.root_dir.handle.createFile(io, emit.sub_path, .{}),
        .build_id = options.build_id,
    } };
    return linker;
}

pub fn deinit(linker: *Linker) void {
    _ = linker;
}

pub fn loadInput(linker: *Linker, input: link.Input) !void {
    const comp = linker.base.comp;
    const io = comp.io;
    const diags = &comp.link_diags;
    const object = switch (input) {
        .object => |object| object,
        else => return diags.fail("air64 accepts Metal libraries as input, not archives or shared libraries", .{}),
    };

    const file = linker.base.file orelse return diags.fail("the output file is closed", .{});
    const stat = object.file.stat(io) catch |err|
        return diags.fail("failed to read {f}: {t}", .{ object.path, err });
    if (stat.size == 0) return diags.fail("{f} is empty", .{object.path});

    const gpa = comp.gpa;
    const bytes = try gpa.alloc(u8, stat.size);
    defer gpa.free(bytes);
    const n = object.file.readPositionalAll(io, bytes, 0) catch |err|
        return diags.fail("failed to read {f}: {t}", .{ object.path, err });
    if (n != bytes.len) return diags.fail("failed to read all of {f}", .{object.path});

    // A Metal library holds the module of one Zig compilation. Linking two of them together is
    // not something a Metal host could load, so refuse instead of silently keeping the first.
    if (linker.written) return diags.fail(
        "the air64 target links one Metal library; combine the kernels into one module",
        .{},
    );
    linker.written = true;

    var writer = file.writer(io, &.{});
    writer.interface.writeAll(bytes) catch |err| switch (err) {
        error.WriteFailed => return diags.fail("failed to write {f}: {t}", .{
            linker.base.emit, writer.err.?,
        }),
    };
    writer.end() catch |err|
        return diags.fail("failed to write {f}: {t}", .{ linker.base.emit, err });
}

pub fn flush(
    linker: *Linker,
    arena: Allocator,
    tid: Zcu.PerThread.Id,
    prog_node: std.Progress.Node,
) link.Error!void {
    _ = .{ linker, arena, tid, prog_node };
}
