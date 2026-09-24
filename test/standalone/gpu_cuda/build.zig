const std = @import("std");

pub fn build(b: *std.Build) void {
    const test_step = b.step("test", "Test it");
    b.default_step = test_step;

    // `std.gpu.cuda` loads the driver library with `dlopen`, which is only available on Linux.
    if (b.graph.host.result.os.tag != .linux) return;

    // Kernels are compiled to PTX assembly for `std.gpu.cuda` to load. PTX for an older compute
    // capability runs on newer GPUs, because the driver compiles it for the device it is loaded
    // on, so one image covers every device.
    const kernel_target = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .cuda,
        .abi = .none,
        .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_75 },
    });

    // The same kernels in two optimization modes, to run the kernels as both the debug and the
    // release build of the standard library compile them.
    const kernels_debug = b.addObject(.{
        .name = "kernels_debug",
        .root_module = b.createModule(.{
            .root_source_file = b.path("kernels.zig"),
            .target = kernel_target,
            .optimize = .debug,
        }),
    });
    const kernels_fast = b.addObject(.{
        .name = "kernels_fast",
        .root_module = b.createModule(.{
            .root_source_file = b.path("kernels.zig"),
            .target = kernel_target,
            .optimize = .fast,
        }),
    });

    const exe = b.addExecutable(.{
        .name = "gpu_cuda",
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = b.graph.host,
            .optimize = .debug,
            // The driver library is loaded with `dlopen`.
            .link_libc = true,
        }),
    });
    exe.root_module.addAnonymousImport("kernels_debug.ptx", .{
        .root_source_file = kernels_debug.getEmittedAsm(),
    });
    exe.root_module.addAnonymousImport("kernels_fast.ptx", .{
        .root_source_file = kernels_fast.getEmittedAsm(),
    });

    const run = b.addRunArtifact(exe);
    test_step.dependOn(&run.step);

    // A panic in a kernel leaves the context unusable, so it is checked in a process of its own.
    const run_assert = b.addRunArtifact(exe);
    run_assert.addArg("assert");
    test_step.dependOn(&run_assert.step);
}
