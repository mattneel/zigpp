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
    // on macOS. The host program, and the kernels that the compiler under test builds for it, are
    // built and run everywhere: without a Mac every test of it is skipped.
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

/// The Metal arm of this suite: `metal_host.zig` runs the vector add, the reduction, and the
/// kernel with scalar parameters through `std.gpu.metal`, with the same kernels and the same
/// arguments as the CUDA and HIP hosts, and checks the results against the CPU.
///
/// Its kernels are `.metallib` files, not objects of this directory, because the container is what
/// a Metal GPU is given. The ones that this step always builds are the kernels of
/// `metal_kernels.zig`, compiled for the `air64` target by the compiler that runs the build: the
/// object file of that target *is* the library, so the compiler under test produces a loadable
/// container on any host, with no Metal toolchain, no linker and no SDK in the pipeline. The
/// containers of the two optimization modes of the rest of the suite are installed next to the
/// host, and each is run by it, so the step exercises the `air64` target wherever it runs.
///
/// A library of the Metal Shading Language, `metal_kernels.metal` compiled by Apple's own
/// compiler, or one that the spike builds, is what `-Dmetallib=a.metallib,b.metallib` feeds in, for
/// a comparison against Apple's code. Every run skips the tests of a machine without the Metal
/// framework, and the tests of a library that does not have a kernel, so the step passes with no
/// Mac and no library at all.
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

    const metallibs = b.option(
        []const u8,
        "metallib",
        "Comma-separated .metallib files whose kernels the Metal host test runs on the GPU of a Mac",
    );
    if (metallibs) |list| {
        const run_libraries = b.addRunArtifact(exe);
        var paths = std.mem.tokenizeScalar(u8, list, ',');
        while (paths.next()) |path| run_libraries.addArg(path);
        test_step.dependOn(&run_libraries.step);
    }

    // The kernels that the compiler under test builds: one container per optimization mode, to run
    // the kernels as both the debug and the release build of the standard library compile them.
    const kernel_target = b.resolveTargetQuery(.{
        .cpu_arch = .air64,
        .os_tag = .macos,
        // The AIR version, the Metal language version and the container version are one row per
        // macOS release, and the deployment target selects the row: macOS 26 is the release whose
        // toolchain emits the AIR 2.8, Metal 4.0 and container 1.2.9 that these kernels run as.
        .os_version_min = .{ .semver = .{ .major = 26, .minor = 0, .patch = 0 } },
    });
    for (images) |image| {
        const kernels = b.addObject(.{
            .name = b.fmt("metal_kernels_{s}", .{image.name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path("metal_kernels.zig"),
                .target = kernel_target,
                .optimize = image.optimize,
            }),
        });
        // An object has no installation procedure of its own, and the emitted file of this one is
        // the `.metallib` itself, so that a build for a Mac installs the containers next to the
        // host that runs them.
        const install_container = b.addInstallBinFile(
            kernels.getEmittedBin(),
            b.fmt("metal_kernels_{s}.metallib", .{image.name}),
        );
        b.getInstallStep().dependOn(&install_container.step);

        const run_kernels = b.addRunArtifact(exe);
        run_kernels.addFileArg(kernels.getEmittedBin());
        test_step.dependOn(&run_kernels.step);
    }
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
