const std = @import("std");

const Backend = enum { cuda, hip };

pub fn build(b: *std.Build) void {
    const test_step = b.step("test", "Test it");
    b.default_step = test_step;

    // The host programs are for the machine that runs the build unless `-Dtarget` says otherwise:
    // `zig build install -Dtarget=x86_64-windows` builds the AMD test for Windows, which WSL can
    // run, without running it.
    const host_target = b.standardTargetOptions(.{});
    // A code object only runs on the architecture that it was compiled for, so the AMD test
    // builds one for each of these, and skips itself on a GPU of any other architecture.
    const amdgpu_archs = b.option(
        []const u8,
        "amdgpu-arch",
        "Comma-separated AMD GPU architectures to build code objects for, as -mcpu names them (default: gfx1030)",
    ) orelse "gfx1030";

    // `std.gpu.cuda` loads the driver library with `dlopen`, which is only available on Linux.
    // `std.gpu.hip` loads the runtime with `dlopen` on Linux and with `ntdll` on Windows.
    const os = host_target.result.os.tag;
    if (os == .linux) addCudaTest(b, test_step, host_target);
    if (os == .linux or os == .windows) addHipTest(b, test_step, host_target, amdgpu_archs);

    // `std.gpu.metal` finds the Metal framework and the Objective-C runtime at run time, and only
    // on macOS. The host program is built and run everywhere: without a Mac, and without the
    // `.metallib` files of the kernels, every test of it is skipped.
    addMetalHost(b, test_step, host_target);
}

const images = [_]struct { name: []const u8, optimize: std.builtin.OptimizeMode }{
    // The same kernels in two optimization modes, to run the kernels as both the debug and the
    // release build of the standard library compile them.
    .{ .name = "debug", .optimize = .debug },
    .{ .name = "fast", .optimize = .fast },
};

fn addCudaTest(b: *std.Build, test_step: *std.Build.Step, host_target: std.Build.ResolvedTarget) void {
    // Kernels are compiled to PTX assembly for `std.gpu.cuda` to load. PTX for an older compute
    // capability runs on newer GPUs, because the driver compiles it for the device it is loaded
    // on, so one image covers every device.
    const kernel_target = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .cuda,
        .abi = .none,
        .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_75 },
    });

    const exe = addHost(b, host_target, .cuda);
    for (images) |image| {
        const kernels = b.addObject(.{
            .name = b.fmt("kernels_{s}", .{image.name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path("kernels.zig"),
                .target = kernel_target,
                .optimize = image.optimize,
            }),
        });
        exe.root_module.addAnonymousImport(b.fmt("kernels_{s}", .{image.name}), .{
            .root_source_file = kernels.getEmittedAsm(),
        });
    }
    addRuns(b, test_step, exe);
}

fn addHipTest(
    b: *std.Build,
    test_step: *std.Build.Step,
    host_target: std.Build.ResolvedTarget,
    amdgpu_archs: []const u8,
) void {
    const exe = addHost(b, host_target, .hip);

    // Kernels are compiled to code objects, the shared libraries of machine code for one AMD
    // architecture that `std.gpu.hip` loads. `code_objects.zig` lists them for `main.zig`.
    var source: std.ArrayList(u8) = .empty;
    var code_object_bins: std.ArrayList(struct { name: []const u8, bin: std.Build.LazyPath }) = .empty;
    source.appendSlice(b.allocator, "pub const all = .{\n") catch @panic("OOM");
    var archs = std.mem.tokenizeScalar(u8, amdgpu_archs, ',');
    while (archs.next()) |arch| {
        const query = std.Target.Query.parse(.{ .arch_os_abi = "amdgcn-amdhsa-none", .cpu_features = arch }) catch |err|
            std.debug.panic("invalid AMD GPU architecture '{s}': {s}", .{ arch, @errorName(err) });
        const kernel_target = b.resolveTargetQuery(query);
        for (images) |image| {
            const kernels = b.addLibrary(.{
                .linkage = .dynamic,
                .name = b.fmt("kernels_{s}_{s}", .{ image.name, arch }),
                .root_module = b.createModule(.{
                    .root_source_file = b.path("kernels.zig"),
                    .target = kernel_target,
                    .optimize = image.optimize,
                }),
            });
            code_object_bins.append(b.allocator, .{
                .name = b.fmt("{s}_{s}", .{ image.name, arch }),
                .bin = kernels.getEmittedBin(),
            }) catch @panic("OOM");
        }
        source.print(b.allocator,
            \\    .{{ .arch = "{0s}", .debug = @embedFile("debug_{0s}"), .fast = @embedFile("fast_{0s}") }},
            \\
        , .{arch}) catch @panic("OOM");
    }
    source.appendSlice(b.allocator, "};\n") catch @panic("OOM");
    const code_objects = b.createModule(.{
        .root_source_file = b.addWriteFiles().add("code_objects.zig", source.items),
    });
    for (code_object_bins.items) |code_object| {
        code_objects.addAnonymousImport(code_object.name, .{ .root_source_file = code_object.bin });
    }
    exe.root_module.addImport("code_objects", code_objects);
    addRuns(b, test_step, exe);
}

/// The Metal arm of this suite: `metal_host.zig` runs the vector add and the reduction through
/// `std.gpu.metal`, with the same kernels and the same arguments as the CUDA and HIP hosts, and
/// checks the results against the CPU. Its kernels are `.metallib` files that the step is given
/// with `-Dmetallib=a.metallib,b.metallib`, rather than objects of this directory, because the
/// container is what a Metal GPU is given: see the header of `metal_host.zig` for the two ways to
/// build one. The host program skips the tests of a library that it was not given, and every test
/// of a machine without the Metal framework, so the step passes with no libraries at all.
fn addMetalHost(b: *std.Build, test_step: *std.Build.Step, host_target: std.Build.ResolvedTarget) void {
    const exe = b.addExecutable(.{
        .name = "gpu_metal_host",
        .root_module = b.createModule(.{
            .root_source_file = b.path("metal_host.zig"),
            .target = host_target,
            .optimize = .debug,
            // The Metal framework and the Objective-C runtime are loaded with `dlopen`.
            .link_libc = true,
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    const metallibs = b.option(
        []const u8,
        "metallib",
        "Comma-separated .metallib files whose kernels the Metal host test runs on the GPU of a Mac",
    );
    if (metallibs) |list| {
        var paths = std.mem.tokenizeScalar(u8, list, ',');
        while (paths.next()) |path| run.addArg(path);
    }
    test_step.dependOn(&run.step);
}

fn addHost(b: *std.Build, host_target: std.Build.ResolvedTarget, backend: Backend) *std.Build.Step.Compile {
    const options = b.addOptions();
    options.addOption(Backend, "backend", backend);
    const exe = b.addExecutable(.{
        .name = b.fmt("gpu_{t}", .{backend}),
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = host_target,
            .optimize = .debug,
            // The driver libraries are loaded with `dlopen` on Linux.
            .link_libc = host_target.result.os.tag == .linux,
        }),
    });
    exe.root_module.addOptions("options", options);
    b.installArtifact(exe);
    return exe;
}

fn addRuns(b: *std.Build, test_step: *std.Build.Step, exe: *std.Build.Step.Compile) void {
    const run = b.addRunArtifact(exe);
    test_step.dependOn(&run.step);

    // A panic in a kernel leaves a CUDA context unusable, so it is checked in a process of its
    // own.
    const run_assert = b.addRunArtifact(exe);
    run_assert.addArg("assert");
    test_step.dependOn(&run_assert.step);
}
