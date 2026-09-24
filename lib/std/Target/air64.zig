//! Apple's AIR (Apple Intermediate Representation), the GPU target whose LLVM IR the Metal
//! toolchain consumes (see `doc/proposals/metal.md`).
//!
//! This file is hand-written: there is no LLVM CPU model for AIR, so the only CPU model is the
//! generic one with no features. The version tables are measured facts from section 3 of the
//! proposal, and match Metal.jl's `src/version.jl`.

const std = @import("../std.zig");
const CpuFeature = std.Target.Cpu.Feature;
const CpuModel = std.Target.Cpu.Model;

pub const Feature = enum {};

pub const featureSet = CpuFeature.FeatureSetFns(Feature).featureSet;
pub const featureSetHas = CpuFeature.FeatureSetFns(Feature).featureSetHas;
pub const featureSetHasAny = CpuFeature.FeatureSetFns(Feature).featureSetHasAny;
pub const featureSetHasAll = CpuFeature.FeatureSetFns(Feature).featureSetHasAll;

pub const all_features: [0]CpuFeature = .{};

pub const cpu = struct {
    pub const generic: CpuModel = .{
        .name = "generic",
        .llvm_name = null,
        .features = featureSet(&.{}),
    };
};

/// The AIR version, Metal shading language version and `.metallib` container version used by
/// one macOS release's toolchain. All three are emitted together: a version triple is not
/// meaningful on its own, and the AIR version must agree with the one derived from the target
/// triple or Apple's `metal-opt`-style tools reject the module.
pub const Versions = struct {
    /// Goes into `!air.version`, and determines the `air64_v<major><minor>` triple suffix.
    air: [3]u16,
    /// Goes into `!air.language_version`.
    metal: [3]u16,
    /// The `.metallib` container's file version.
    metallib: [3]u16,
};

/// macOS major version -> AIR / Metal language / metallib container versions.
///
/// The rows are 13, 14, 15, 26 and 27, i.e. the macOS releases whose toolchains changed
/// anything. A version below the lowest row uses the 13 row, a version above the highest row
/// uses the 27 row, and a version between two rows uses the lower row: this maps 16 through 25
/// through the 15 row, as Metal.jl does.
///
/// Note that macOS 26 (Tahoe) reports its version as 16 when the caller was built against an
/// old SDK; callers observing `NSProcessInfo` must normalize such a version (add 10 to the
/// major version) before calling this function.
pub fn versionsForMacos(macos_major: u16) Versions {
    const macos_13 = Versions{ .air = .{ 2, 5, 0 }, .metal = .{ 3, 0, 0 }, .metallib = .{ 1, 2, 7 } };
    const macos_14 = Versions{ .air = .{ 2, 6, 0 }, .metal = .{ 3, 1, 0 }, .metallib = .{ 1, 2, 7 } };
    const macos_15 = Versions{ .air = .{ 2, 7, 0 }, .metal = .{ 3, 2, 0 }, .metallib = .{ 1, 2, 8 } };
    const macos_26 = Versions{ .air = .{ 2, 8, 0 }, .metal = .{ 4, 0, 0 }, .metallib = .{ 1, 2, 9 } };
    const macos_27 = Versions{ .air = .{ 2, 9, 0 }, .metal = .{ 4, 1, 0 }, .metallib = .{ 1, 2, 9 } };

    if (macos_major >= 27) return macos_27;
    if (macos_major >= 26) return macos_26;
    if (macos_major >= 15) return macos_15;
    if (macos_major >= 14) return macos_14;
    return macos_13;
}
