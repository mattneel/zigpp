//! Installs an exact compiler version into the global cache.
//!
//! `zig any <version>` and the automatic version dispatch of `src/main.zig`
//! (a project's `minimum_zig_version`) both need a compiler that is not the
//! one running. This file downloads that compiler, verifies it, unpacks it
//! into a temporary directory, and only then renames it to
//! `<global cache>/any/<version>/`, where the launcher runs `<version>/zig`.
//!
//! Versions come from one of three places:
//!
//! * A **Zig++ release**: a GitHub release of Zig++ whose version has `zigpp.`
//!   build metadata, for example `0.17.0-dev.2361+zigpp.5b96e6d21`. The release
//!   tag is `zigpp-` plus the version without its build metadata, and the
//!   release carries an `index.json` asset in the format of upstream's
//!   download index, which names the archive and its SHA-256.
//! * An **upstream tagged release**, for example `0.15.1`: listed in
//!   https://ziglang.org/download/index.json, which names the archive and its
//!   SHA-256.
//! * An **upstream dev build**, for example `0.16.0-dev.1234+abcdef012`: the
//!   archive URL is
//!   https://ziglang.org/builds/zig-<arch>-<os>-<version>.tar.xz (.zip on
//!   Windows).
//!
//! Every upstream archive is verified against the minisign signature that
//! upstream publishes next to it, which is signed with the Zig Software
//! Foundation's key; a tagged release is checked against the SHA-256 of the
//! download index as well. ziglang.org keeps only the recent dev builds, so an
//! archive that is no longer there is looked for on the community mirrors of
//! https://ziglang.org/download/community-mirrors.txt. The signature is the
//! same everywhere, which is what makes a mirror as trustworthy as the site.

const std = @import("std");
const builtin = @import("builtin");
const native_os = builtin.os.tag;
const Allocator = std.mem.Allocator;
const Blake2b512 = std.crypto.hash.blake2.Blake2b512;
const Dir = std.Io.Dir;
const File = std.Io.File;
const Graph = @import("Graph.zig");
const Io = std.Io;
const Minisign = @import("Minisign.zig");
const fatal = std.process.fatal;
const http = std.http;
const log = std.log;
const mem = std.mem;
const process = std.process;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The directory in the global cache that holds one directory per installed
/// version: `<global cache>/any/<version>/zig`. `src/main.zig` reads the same
/// layout when it dispatches to a version.
pub const versions_dir_name = "any";

/// The directory in the global cache for temporary files.
const tmp_dir_name = "tmp";

/// The GitHub repository that publishes the releases of Zig++.
const zigpp_repo = "mattneel/zigpp";

/// Zig++ marks its versions with this build metadata, e.g.
/// `0.17.0-dev.2361+zigpp.5b96e6d21`.
const zigpp_metadata_prefix = "zigpp.";

const upstream_index_url = "https://ziglang.org/download/index.json";
const upstream_builds_url = "https://ziglang.org/builds";

/// The public key that the Zig Software Foundation signs upstream downloads
/// with; `Minisign.zig` knows the format.
const upstream_public_key = Minisign.zig_public_key;

/// Upstream's community mirrors, one base URL per line. They mirror the
/// downloads of ziglang.org under the file's own name, and upstream asks
/// downloaders to say who they are in a query parameter.
const community_mirrors_url = "https://ziglang.org/download/community-mirrors.txt";
const mirror_source_query = "source=zigpp";

/// The host, named the way the download indexes name it: `<arch>-<os>`.
pub const Host = struct {
    arch: std.Target.Cpu.Arch,
    os: std.Target.Os.Tag,

    /// The host this compiler runs on. Installed versions are of course
    /// binaries for the host that downloads them.
    pub fn current() Host {
        return .{ .arch = builtin.cpu.arch, .os = native_os };
    }

    /// The key of this host in a download index, e.g. `x86_64-linux`.
    pub fn name(h: Host, arena: Allocator) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(arena, "{s}-{s}", .{ @tagName(h.arch), @tagName(h.os) });
    }

    pub fn exeName(h: Host) []const u8 {
        return if (h.os == .windows) "zig.exe" else "zig";
    }

    /// Windows archives are zip files; every other host uses .tar.xz.
    pub fn archiveExtension(h: Host) []const u8 {
        return if (h.os == .windows) "zip" else "tar.xz";
    }

    /// The name of the archive, which is also the name of the single
    /// top-level directory inside it: `zig-<arch>-<os>-<version>`.
    pub fn archiveStem(h: Host, arena: Allocator, version_str: []const u8) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(arena, "zig-{s}-{s}", .{ try h.name(arena), version_str });
    }
};

/// Where the archive of a version comes from, which the version string alone
/// determines.
pub const Source = union(enum) {
    /// A Zig++ release: a GitHub release that carries an `index.json` asset.
    zigpp_release: struct {
        tag: []const u8,
        index_url: []const u8,
    },
    /// A tagged upstream Zig release, listed in the upstream download index.
    upstream_release: struct {
        index_url: []const u8,
    },
    /// An upstream Zig dev build, whose archive URL is derived from the
    /// version. The upstream index lists only the current master, and a dev
    /// build is verified by its minisign signature, not by an index.
    upstream_dev: struct {
        tarball_url: []const u8,
    },
};

pub const SourceError = error{ InvalidVersion, OutOfMemory };

/// Derives where the archive of `version_str` comes from. Exact-version
/// semantics: the version string names the exact compiler, metadata included.
pub fn sourceFromVersion(arena: Allocator, host: Host, version_str: []const u8) SourceError!Source {
    // A Zig++ version is recognized by its build metadata, because a tagged
    // Zig++ release is named after the base version: `+zigpp.<commit>` cannot
    // be part of a tag.
    if (mem.indexOfScalar(u8, version_str, '+')) |plus| {
        if (mem.startsWith(u8, version_str[plus + 1 ..], zigpp_metadata_prefix)) {
            const tag = try std.fmt.allocPrint(arena, "zigpp-{s}", .{version_str[0..plus]});
            return .{ .zigpp_release = .{
                .tag = tag,
                .index_url = try std.fmt.allocPrint(
                    arena,
                    "https://github.com/{s}/releases/download/{s}/index.json",
                    .{ zigpp_repo, tag },
                ),
            } };
        }
    }

    const version = std.SemanticVersion.parse(version_str) catch return error.InvalidVersion;
    if (version.pre != null) {
        // A pre-release is a dev build: those are not listed in the upstream
        // download index, and their archive URL is derived from the version.
        return .{ .upstream_dev = .{
            .tarball_url = try std.fmt.allocPrint(
                arena,
                "{s}/zig-{s}-{s}.{s}",
                .{ upstream_builds_url, try host.name(arena), version_str, host.archiveExtension() },
            ),
        } };
    }
    // Tagged releases are listed in the upstream download index.
    return .{ .upstream_release = .{ .index_url = upstream_index_url } };
}

/// `<global cache>/any/<version>`, relative to the global cache directory.
pub fn versionSubPath(arena: Allocator, version_str: []const u8) Allocator.Error![]const u8 {
    return std.fmt.allocPrint(arena, "{s}/{s}", .{ versions_dir_name, version_str });
}

/// `<global cache>/any/<version>/zig[.exe]`.
fn installedExePath(arena: Allocator, global_cache: std.Build.Cache.Directory, host: Host, version_str: []const u8) Allocator.Error![]const u8 {
    return global_cache.join(arena, &.{ versions_dir_name, version_str, host.exeName() });
}

pub fn cmdAnyInstall(gpa: Allocator, graph: *Graph, args: []const []const u8) !void {
    const io = graph.io;
    const arena = graph.arena;

    var arg_i: usize = 0;
    const version_str = while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (mem.startsWith(u8, arg, "-")) {
            if (mem.eql(u8, arg, "-h") or mem.eql(u8, arg, "--help")) {
                try File.stdout().writeStreamingAll(io, usage_any_install);
                return process.cleanExit(io);
            }
            fatal("unrecognized parameter: {q}", .{arg});
        }
        break arg;
    } else fatal("expected a version, for example `zig any 0.15.1`", .{});

    if (arg_i + 1 != args.len) fatal("unexpected extra parameter: {q}", .{args[arg_i + 1]});
    _ = std.SemanticVersion.parse(version_str) catch
        fatal("invalid version {q}", .{version_str});

    const host = Host.current();
    const exe_path = try installedExePath(arena, graph.global_cache_root, host, version_str);

    const already_installed = if (Dir.cwd().access(io, exe_path, .{})) |_| true else |err| switch (err) {
        error.FileNotFound => false,
        else => |e| fatal("unable to check whether {s} is installed: {t}", .{ version_str, e }),
    };
    if (already_installed) return;

    var root_prog_node = std.Progress.start(io, .{
        .root_name = "Any",
    });
    defer root_prog_node.end();

    var http_client: http.Client = .{ .allocator = gpa, .io = io };
    defer http_client.deinit();
    try http_client.initDefaultProxies(arena, &graph.environ_map);

    const source = sourceFromVersion(arena, host, version_str) catch |err| switch (err) {
        error.InvalidVersion => fatal("invalid version {q}", .{version_str}),
        error.OutOfMemory => |e| return e,
    };
    const archive = try resolveArchive(arena, &http_client, root_prog_node, host, version_str, source);

    // The install happens in `installArchive`, which owns the temporary
    // directory: its defers delete it before it returns, and only then is the
    // failure reported. `fatal` exits without running any defer, so a failure
    // reported here cannot leave a half-downloaded version behind.
    var install: Install = .{
        .arena = arena,
        .gpa = gpa,
        .io = io,
        .http_client = &http_client,
        .root_prog_node = root_prog_node,
        .host = host,
        .version_str = version_str,
        .global_cache_root = graph.global_cache_root,
    };
    install.installArchive(archive) catch |err| install.reportFailure(err);
}

/// Why an install did not happen. Every one of these is returned out of the
/// temporary directory's scope, and reported after it is gone.
const Error = error{
    /// A source did not provide the archive. The download loop catches this to
    /// try the next source; `attempts` says what each source did.
    Unavailable,
    /// No source had the archive.
    NoSource,
    /// A downloaded archive did not match the SHA-256 its index published.
    ChecksumMismatch,
    /// A downloaded archive did not match the minisign signature upstream
    /// published for it.
    BadSignature,
    /// The archive is not an archive, or not the archive of this version.
    InvalidArchive,
    /// A version directory is there, but the compiler is not in it.
    BrokenInstall,
    /// Moving the unpacked version into place failed.
    RenameFailed,
    /// The file system said no.
    FileSystemFailed,
    OutOfMemory,
};

/// One install of one version: everything a download needs, and everything a
/// failure needs to explain itself.
const Install = struct {
    arena: Allocator,
    gpa: Allocator,
    io: Io,
    http_client: *http.Client,
    root_prog_node: std.Progress.Node,
    host: Host,
    version_str: []const u8,
    global_cache_root: std.Build.Cache.Directory,
    /// The version was installed from here, once one source worked out.
    source_url: ?[]const u8 = null,
    /// The name of the archive file of the version, set by `installArchive`.
    archive_name: ?[]const u8 = null,
    /// The community mirrors, once upstream's list of them was downloaded.
    mirrors: ?[]const []const u8 = null,
    /// What was tried and did not have the archive, in the order it was tried.
    attempts: std.ArrayList(Attempt) = .empty,
    /// The failure in the words of the code that failed, when `@errorName` of
    /// the error would not say enough.
    message: ?[]const u8 = null,

    /// Downloads, verifies, unpacks and publishes the archive of the version.
    fn installArchive(install: *Install, archive: Archive) Error!void {
        const io = install.io;
        const arena = install.arena;
        const version_str = install.version_str;

        // Install somewhere out of the way, and let the final rename publish
        // it: `<global cache>/any/<version>` either does not exist at all or
        // is a complete install, no matter how a download fails. The random
        // name is what keeps two processes that install the same version at
        // the same time out of each other's way.
        var random_bits: u64 = undefined;
        io.random(@ptrCast(&random_bits));
        const tmp_sub_path = try std.fmt.allocPrint(arena, "{s}/{s}-{s}", .{
            tmp_dir_name, version_str, std.fmt.hex(random_bits),
        });
        install.global_cache_root.handle.createDirPath(io, tmp_sub_path) catch |err| {
            install.message = try install.print("unable to create {s}/{s}: {t}", .{
                install.global_cache_root.path orelse ".", tmp_sub_path, err,
            });
            return error.FileSystemFailed;
        };
        defer install.global_cache_root.handle.deleteTree(io, tmp_sub_path) catch |err| {
            log.warn("unable to delete temporary directory {s}/{s}: {t}", .{
                install.global_cache_root.path orelse ".", tmp_sub_path, err,
            });
        };

        var tmp_dir = install.global_cache_root.handle.openDir(io, tmp_sub_path, .{ .iterate = true }) catch |err| {
            install.message = try install.print("unable to open {s}/{s}: {t}", .{
                install.global_cache_root.path orelse ".", tmp_sub_path, err,
            });
            return error.FileSystemFailed;
        };
        defer tmp_dir.close(io);

        const archive_name = try std.fmt.allocPrint(arena, "{s}.{s}", .{
            try install.host.archiveStem(arena, version_str), install.host.archiveExtension(),
        });
        install.archive_name = archive_name;

        try install.downloadArchive(archive, tmp_dir, archive_name);

        // The archive is unpacked from the file that was just verified, so its
        // handle is opened for reading again.
        var archive_file = tmp_dir.openFile(io, archive_name, .{}) catch |err| {
            install.message = try install.print("unable to open the downloaded {s}: {t}", .{ archive_name, err });
            return error.FileSystemFailed;
        };
        defer archive_file.close(io);

        try install.unpack(tmp_dir, archive_file);

        // The archive holds one top-level directory, which is the version's
        // root. Releases up to Zig 0.14 name it `zig-linux-x86_64-0.14.0` and
        // newer ones `zig-x86_64-linux-0.15.1`, so read the name from the
        // archive instead of predicting it.
        const unpacked_root = try install.unpackedRoot(tmp_dir, tmp_sub_path);
        const exe_rel_path = try std.fmt.allocPrint(arena, "{s}/{s}", .{
            unpacked_root, install.host.exeName(),
        });
        if (tmp_dir.access(io, exe_rel_path, .{})) |_| {} else |err| {
            install.message = try install.print("the archive of {s} has no {s}: {t}", .{
                version_str, exe_rel_path, err,
            });
            return error.InvalidArchive;
        }

        // The rename needs the directory it moves the version into to exist.
        install.global_cache_root.handle.createDirPath(io, versions_dir_name) catch |err| {
            install.message = try install.print("unable to create {s}: {t}", .{ versions_dir_name, err });
            return error.FileSystemFailed;
        };

        const version_sub_path = try versionSubPath(arena, version_str);
        switch (try install.renameIntoVersions(tmp_dir, unpacked_root, version_sub_path)) {
            .installed => log.info("installed {s} into {s}", .{ version_str, version_sub_path }),
            .already_installed => {
                // Another process installed this version while we were
                // downloading ours. A version only appears at its final path
                // after it was completely unpacked, so the other install is
                // what a user gets; make sure it is complete before saying so.
                try install.checkInstalled();
                log.info("version {s} was installed by another process", .{version_str});
            },
        }
    }

    /// The two ways publishing a version can end.
    const Published = enum { installed, already_installed };

    /// Renames the unpacked version into `<global cache>/any/<version>`. A
    /// plain rename does not replace a non-empty directory, so a version that
    /// is already there stops it, and which error says so depends on the
    /// platform: `DirNotEmpty` and `AccessDenied` both mean "another process
    /// got there first" (Windows says AccessDenied for a directory collision,
    /// and the BSDs and some network file systems have no atomic
    /// non-replacing rename at all -- which is why a plain rename is used, and
    /// why a collision is verified rather than trusted).
    fn renameIntoVersions(
        install: *Install,
        tmp_dir: Dir,
        unpacked_root: []const u8,
        version_sub_path: []const u8,
    ) Error!Published {
        Io.Dir.rename(
            tmp_dir,
            unpacked_root,
            install.global_cache_root.handle,
            version_sub_path,
            install.io,
        ) catch |err| switch (err) {
            error.DirNotEmpty, error.AccessDenied => return .already_installed,
            else => |e| {
                install.message = try install.print("unable to move {s} into {s}: {t}", .{
                    unpacked_root, version_sub_path, e,
                });
                return error.RenameFailed;
            },
        };
        return .installed;
    }

    /// Fails unless the compiler of this version is where the version
    /// directory is. A version directory without it is not an install -- a
    /// partial delete, or an antivirus that took `zig.exe` away -- and
    /// downloading the archive again would not fix it: the directory is in the
    /// way, and only the user can remove it.
    fn checkInstalled(install: *Install) Error!void {
        const io = install.io;
        const exe_path = try installedExePath(
            install.arena,
            install.global_cache_root,
            install.host,
            install.version_str,
        );
        if (Dir.cwd().access(io, exe_path, .{})) |_| return else |err| switch (err) {
            error.FileNotFound => {},
            else => |e| {
                install.message = try install.print("unable to check {s}: {t}", .{ exe_path, e });
                return error.FileSystemFailed;
            },
        }

        const dir_path = try install.versionDirPath();
        const dir_exists = if (Dir.cwd().access(io, dir_path, .{})) |_| true else |err| switch (err) {
            error.FileNotFound => false,
            else => |e| {
                install.message = try install.print("unable to check {s}: {t}", .{ dir_path, e });
                return error.FileSystemFailed;
            },
        };
        if (dir_exists) {
            install.message = try install.print(
                "{s} is not a complete install: it has no {s}; delete it to install {s} again",
                .{ dir_path, install.host.exeName(), install.version_str },
            );
        } else {
            install.message = try install.print("no compiler was installed at {s}", .{exe_path});
        }
        return error.BrokenInstall;
    }

    /// `<global cache>/any/<version>`, the directory an install publishes.
    fn versionDirPath(install: *Install) Error![]const u8 {
        return install.global_cache_root.join(install.arena, &.{
            versions_dir_name, install.version_str,
        });
    }

    /// Downloads the archive of the version from the first source that has it,
    /// verifying every byte that is accepted.
    fn downloadArchive(install: *Install, archive: Archive, dest_dir: Dir, archive_name: []const u8) Error!void {
        const sources = try install.archiveSources(archive);
        for (sources) |candidate| {
            log.debug("trying {s}", .{candidate.url});
            const downloaded = install.downloadFrom(candidate, dest_dir, archive_name) catch |err| switch (err) {
                error.Unavailable => continue,
                else => |e| return e,
            };
            if (downloaded != null) return;
        }
        return error.NoSource;
    }

    /// The sources to try, in order: the one that should have the archive
    /// first, then the community mirrors. A list of mirrors that cannot be
    /// downloaded is not fatal: the source that should have the archive has
    /// already failed, so the install is about to fail anyway, and the reason
    /// belongs in the list of what was tried.
    fn archiveSources(install: *Install, archive: Archive) Error![]const Candidate {
        const arena = install.arena;
        var sources: std.ArrayList(Candidate) = .empty;
        try sources.append(arena, .{
            .url = archive.url,
            .signature_url = archive.signature_url,
            .shasum = archive.shasum,
            .size = archive.size,
        });
        if (!archive.mirrors) return sources.items;

        // A mirror is asked for the file's own name, whatever directory the
        // primary source keeps it in.
        const filename = filenameFromUrl(archive.url);
        const mirrors = install.communityMirrors() catch |err| switch (err) {
            error.Unavailable => return sources.items,
            else => |e| return e,
        };
        for (mirrors) |mirror| {
            try sources.append(arena, .{
                .url = try install.print("{s}/{s}?{s}", .{ mirror, filename, mirror_source_query }),
                .signature_url = try install.print(
                    "{s}/{s}.minisig?{s}",
                    .{ mirror, filename, mirror_source_query },
                ),
                .shasum = archive.shasum,
                .size = archive.size,
            });
        }
        return sources.items;
    }

    /// The community mirrors of upstream's downloads, in the order to try
    /// them: shuffled, because no one mirror has everything, and because a
    /// single file is better spread over them.
    fn communityMirrors(install: *Install) Error![]const []const u8 {
        if (install.mirrors) |mirrors| return mirrors;

        var reason: []const u8 = undefined;
        const text = install.fetchSmall(community_mirrors_url, &reason) catch |err| switch (err) {
            error.Unavailable => return install.fail(community_mirrors_url, reason),
            else => |e| return e,
        };

        const parsed = try mirrorUrls(install.arena, text);
        if (parsed.len == 0) {
            return install.fail(community_mirrors_url, try install.print("it lists no mirrors", .{}));
        }
        // A copy, because shuffling a list that the parser returned would need
        // a mutable slice of it.
        const mirrors = try install.arena.dupe([]const u8, parsed);

        var seed: u64 = undefined;
        install.io.random(@ptrCast(&seed));
        var prng: std.Random.DefaultPrng = .init(seed);
        prng.random().shuffle([]const u8, mirrors);

        install.mirrors = mirrors;
        return mirrors;
    }

    /// Downloads, verifies and returns the archive from one source. Null means
    /// the source does not have the archive or could not deliver it, and the
    /// next source should be tried; the reason is in `attempts`. A download
    /// that does not verify is an error instead: another source would only
    /// hide that a file did not match the signature of it.
    fn downloadFrom(
        install: *Install,
        candidate: Candidate,
        dest_dir: Dir,
        archive_name: []const u8,
    ) Error!?Digests {
        const io = install.io;

        // The signature comes first: a source that has the archive but not the
        // signature that goes with it is no source at all, because nothing
        // would verify what it serves.
        var signature_text: ?[]const u8 = null;
        if (candidate.signature_url) |signature_url| {
            var reason: []const u8 = undefined;
            signature_text = install.fetchSmall(signature_url, &reason) catch |err| switch (err) {
                error.Unavailable => {
                    try install.recordFailure(candidate.url, try install.print(
                        "no minisign signature at {s}: {s}",
                        .{ signature_url, reason },
                    ));
                    return null;
                },
                else => |e| return e,
            };
        }

        var dest_file = dest_dir.createFile(io, archive_name, .{ .read = true }) catch |err| {
            install.message = try install.print("unable to create {s}: {t}", .{ archive_name, err });
            return error.FileSystemFailed;
        };
        defer dest_file.close(io);

        const digests = install.download(candidate, dest_file) catch |err| switch (err) {
            error.Unavailable => return null,
            else => |e| return e,
        };

        if (candidate.shasum) |expected| {
            if (!checksumMatches(expected, digests.sha256)) {
                install.message = try install.print(
                    "checksum mismatch for {s}: the index says {x}, the download is {x}; nothing was installed",
                    .{ candidate.url, expected, digests.sha256 },
                );
                return error.ChecksumMismatch;
            }
        }

        if (signature_text) |text| {
            const signature = (try install.parseSignature(candidate, text)) orelse return null;
            try install.verifySignature(signature, candidate, digests, dest_file);
        }

        install.source_url = candidate.url;
        return digests;
    }

    /// Streams `candidate.url` into `dest_file`, hashing it as it goes, and
    /// returns the digests of what was written. A source that does not have
    /// the file, or that cannot deliver it, leaves the reason in `attempts`
    /// and returns `error.Unavailable`: the next source may do better. A file
    /// that the file system refuses to take is an error instead: the next
    /// source would not help.
    fn download(install: *Install, candidate: Candidate, dest_file: File) Error!Digests {
        const io = install.io;
        const prog_node = install.root_prog_node.startFmt(0, "download {s}", .{install.archiveName()});
        defer prog_node.end();

        const uri = std.Uri.parse(candidate.url) catch |err|
            return install.fail(candidate.url, try install.print("invalid URL: {t}", .{err}));

        var req = install.http_client.request(.GET, uri, .{}) catch |err|
            return install.fail(candidate.url, try install.print("unable to request it: {t}", .{err}));
        defer req.deinit();

        req.sendBodiless() catch |err|
            return install.fail(candidate.url, try install.print("unable to request it: {t}", .{err}));

        // A redirect buffer makes redirects be followed: a mirror can be a
        // name that redirects to the host that really has the file.
        var redirect_buffer: [8 * 1024]u8 = undefined;
        var response = req.receiveHead(&redirect_buffer) catch |err| switch (err) {
            error.ReadFailed => return install.fail(candidate.url, try install.print(
                "unable to read the response: {t}",
                .{req.connection.?.getReadError().?},
            )),
            else => |e| return install.fail(candidate.url, try install.print(
                "bad HTTP response: {t}",
                .{e},
            )),
        };
        const head = &response.head;
        if (head.status != .ok) return install.fail(candidate.url, try install.print("HTTP {d} {s}", .{
            head.status, head.status.phrase() orelse "",
        }));

        const total_bytes = head.content_length orelse candidate.size;
        if (total_bytes) |total| prog_node.setEstimatedTotalItems(std.math.lossyCast(usize, total));

        var transfer_buffer: [64]u8 = undefined;
        var decompress: http.Decompress = undefined;
        const decompress_buffer = try install.arena.alloc(u8, head.content_encoding.minBufferCapacity());
        const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompress_buffer);

        var file_buffer: [32 * 1024]u8 = undefined;
        var file_writer = dest_file.writer(io, &file_buffer);

        var sha256: Sha256 = .init(.{});
        var blake2b: Blake2b512 = .init(.{});
        var downloaded: u64 = 0;
        var chunk: [64 * 1024]u8 = undefined;
        while (true) {
            const n = reader.readSliceShort(&chunk) catch return install.fail(
                candidate.url,
                try install.print("the download failed: {t}", .{response.bodyErr().?}),
            );
            if (n == 0) break;
            sha256.update(chunk[0..n]);
            blake2b.update(chunk[0..n]);
            file_writer.interface.writeAll(chunk[0..n]) catch {
                install.message = try install.print("unable to write {s}: {t}", .{
                    install.archiveName(), file_writer.err.?,
                });
                return error.FileSystemFailed;
            };
            downloaded += n;
            prog_node.setCompletedItems(std.math.lossyCast(usize, downloaded));
        }
        file_writer.interface.flush() catch {
            install.message = try install.print("unable to write {s}: {t}", .{
                install.archiveName(), file_writer.err.?,
            });
            return error.FileSystemFailed;
        };

        if (total_bytes) |total| {
            if (downloaded != total) return install.fail(candidate.url, try install.print(
                "the download stopped after {d} of {d} bytes",
                .{ downloaded, total },
            ));
        }

        var digests: Digests = .{ .sha256 = undefined, .blake2b512 = undefined };
        sha256.final(&digests.sha256);
        blake2b.final(&digests.blake2b512);
        return digests;
    }

    /// The minisign signature in `text`. Null means the source did not serve a
    /// minisign signature at all -- a proxy's error page, say -- which for this
    /// install is the same as not having the signature: no source was found
    /// yet, and the reason is recorded.
    fn parseSignature(
        install: *Install,
        candidate: Candidate,
        text: []const u8,
    ) Error!?Minisign.Signature {
        return Minisign.Signature.parse(text) catch |err| {
            try install.recordFailure(candidate.url, try install.print(
                "the signature at {s} is not a minisign signature: {t}",
                .{ candidate.signature_url.?, err },
            ));
            return null;
        };
    }

    /// Checks the archive against the minisign signature that upstream
    /// publishes for it, with the key that the Zig Software Foundation signs
    /// upstream downloads with. This is what makes a community mirror
    /// trustworthy, and it is the only check a dev build has: the upstream
    /// index publishes no SHA-256 for anything but the current master. A
    /// signature that does not verify is an error: trying another source would
    /// only hide that a file did not match the signature of it.
    fn verifySignature(
        install: *Install,
        signature: Minisign.Signature,
        candidate: Candidate,
        digests: Digests,
        file: File,
    ) Error!void {
        const public_key = Minisign.PublicKey.fromBase64(upstream_public_key) catch |err| {
            // The key is a constant in this file: it parses, or this build is
            // broken.
            install.message = try install.print(
                "the Zig Software Foundation public key of this build is not a minisign key: {t}",
                .{err},
            );
            return error.BadSignature;
        };

        switch (signature.kind) {
            // The download hashed the archive as it streamed it, which is
            // exactly what a signature in this form is over.
            .hash => signature.verifyHash(public_key, digests.blake2b512, install.archiveName()) catch |err|
                return install.badSignature(err, candidate),
            // A signature over the file itself needs the file, and it is read
            // again, in chunks, because the download did not keep it in
            // memory. Upstream's are all in the hashed form.
            .message => {
                var buffer: [16 * 1024]u8 = undefined;
                var reader = file.reader(install.io, &buffer);
                reader.seekTo(0) catch |err| {
                    install.message = try install.print("unable to read {s} to verify it: {t}", .{
                        install.archiveName(), err,
                    });
                    return error.FileSystemFailed;
                };
                signature.verifyReader(public_key, &reader.interface, install.archiveName()) catch |err|
                    return install.badSignature(err, candidate);
            },
        }
        log.info("verified the minisign signature of {s} with the Zig Software Foundation's key", .{
            install.archiveName(),
        });
    }

    /// The download did not match the signature that upstream published for
    /// it.
    fn badSignature(install: *Install, err: Minisign.Error, candidate: Candidate) Error {
        install.message = try install.print(
            "the minisign signature of {s} from {s} does not verify with the Zig Software Foundation's key: {t}; nothing was installed",
            .{ install.archiveName(), candidate.url, err },
        );
        return error.BadSignature;
    }

    /// Unpacks the archive into `dest_dir`.
    fn unpack(install: *Install, dest_dir: Dir, archive_file: File) Error!void {
        const io = install.io;
        var file_buffer: [16 * 1024]u8 = undefined;
        var file_reader = archive_file.reader(io, &file_buffer);
        file_reader.seekTo(0) catch |err| {
            install.message = try install.print("unable to read {s}: {t}", .{ install.archiveName(), err });
            return error.FileSystemFailed;
        };

        switch (install.host.os) {
            .windows => std.zip.extract(dest_dir, &file_reader, .{ .allow_backslashes = true }) catch |err| {
                install.message = try install.print("unable to unpack {s}: {t}", .{ install.archiveName(), err });
                return error.InvalidArchive;
            },
            else => {
                var decompress = std.compress.xz.Decompress.init(&file_reader.interface, install.gpa, &.{}) catch |err| {
                    install.message = try install.print("unable to decompress {s}: {t}", .{
                        install.archiveName(), err,
                    });
                    return error.InvalidArchive;
                };
                defer decompress.deinit();
                std.tar.extract(io, dest_dir, &decompress.reader, .{ .mode_mode = .executable_bit_only }) catch |err| {
                    install.message = try install.print("unable to unpack {s}: {t}", .{
                        install.archiveName(), err,
                    });
                    return error.InvalidArchive;
                };
            },
        }
    }

    /// The single directory that unpacking the archive created, which is
    /// everything but the downloaded archive file itself.
    fn unpackedRoot(install: *Install, tmp_dir: Dir, tmp_sub_path: []const u8) Error![]const u8 {
        const io = install.io;
        var root: ?[]const u8 = null;
        var iterator = tmp_dir.iterate();
        while (iterator.next(io) catch |err| {
            install.message = try install.print("unable to read {s}: {t}", .{ tmp_sub_path, err });
            return error.FileSystemFailed;
        }) |entry| {
            const stat = tmp_dir.statFile(io, entry.name, .{ .follow_symlinks = false }) catch |err| {
                install.message = try install.print("unable to inspect {s}/{s}: {t}", .{
                    tmp_sub_path, entry.name, err,
                });
                return error.FileSystemFailed;
            };
            if (stat.kind != .directory) continue; // the archive we downloaded

            if (root) |first| {
                install.message = try install.print(
                    "{s} unpacked into more than one directory: {s} and {s}",
                    .{ install.archiveName(), first, entry.name },
                );
                return error.InvalidArchive;
            }
            root = try install.arena.dupe(u8, entry.name);
        }
        if (root) |root_name| return root_name;
        install.message = try install.print("{s} unpacked into no directory", .{install.archiveName()});
        return error.InvalidArchive;
    }

    /// Downloads a small file into memory. `reason` is set, and
    /// `error.Unavailable` returned, when the source does not have it or
    /// cannot be reached; the caller decides what that means.
    fn fetchSmall(install: *Install, url: []const u8, reason: *[]const u8) Error![]const u8 {
        var body: std.Io.Writer.Allocating = .init(install.arena);
        const result = install.http_client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &body.writer,
        }) catch |err| {
            reason.* = try install.print("unable to download it: {t}", .{err});
            return error.Unavailable;
        };
        if (result.status != .ok) {
            reason.* = try install.print("HTTP {d} {s}", .{
                result.status, result.status.phrase() orelse "",
            });
            return error.Unavailable;
        }
        return body.written();
    }

    /// Records that a source did not provide the archive.
    fn recordFailure(install: *Install, url: []const u8, reason: []const u8) Error!void {
        try install.attempts.append(install.arena, .{ .url = url, .reason = reason });
        log.debug("{s} is not a source for {s}: {s}", .{ url, install.archiveName(), reason });
    }

    /// `recordFailure`, and the error that makes the download loop try the
    /// next source. An error value, so that callers that need the error (`the
    /// download loop`) return it directly, and callers that need null
    /// (`downloadFrom`) record the failure and return null.
    fn fail(install: *Install, url: []const u8, reason: []const u8) Error {
        install.recordFailure(url, reason) catch return error.OutOfMemory;
        return error.Unavailable;
    }

    /// The name of the archive that is being installed, e.g.
    /// `zig-x86_64-linux-0.15.1.tar.xz`.
    fn archiveName(install: *Install) []const u8 {
        return install.archive_name orelse install.version_str;
    }

    /// Formats `fmt` in the arena, for the messages of failures.
    fn print(install: *Install, comptime fmt: []const u8, args: anytype) Error![]const u8 {
        return std.fmt.allocPrint(install.arena, fmt, args);
    }

    /// Reports why the install did not happen, and exits. The temporary
    /// directory is already gone: `installArchive` deleted it before returning
    /// the error that got here, because `fatal` exits without running defers.
    fn reportFailure(install: *Install, err: Error) noreturn {
        if (err == error.NoSource) {
            var text: std.Io.Writer.Allocating = .init(install.arena);
            const writer = &text.writer;
            writer.print("no download source has {s} for version {s}:", .{
                install.archiveName(), install.version_str,
            }) catch fatal("no download source has {s}", .{install.archiveName()});
            for (install.attempts.items) |attempt|
                writer.print("\n  {s}: {s}", .{ attempt.url, attempt.reason }) catch {};
            writer.print(
                "\nset ZIG_ANY=off to run the compiler that was invoked instead of the pinned version",
                .{},
            ) catch {};
            fatal("{s}", .{text.written()});
        }
        if (install.message) |text| fatal("{s}", .{text});
        fatal("unable to install {s}: {t}", .{ install.version_str, err });
    }
};

/// A source that did not provide the archive, in the order it was tried.
const Attempt = struct {
    url: []const u8,
    reason: []const u8,
};

/// The digests of a downloaded archive, both computed in the one pass that
/// writes it into the cache.
const Digests = struct {
    /// SHA-256, which a download index publishes for a tagged release.
    sha256: [Sha256.digest_length]u8,
    /// BLAKE2b-512, which a minisign signature in its hashed form is over.
    blake2b512: [Blake2b512.digest_length]u8,
};

/// The archive of a version: where to look for it first, and what verifies it.
const Archive = struct {
    /// The URL of the archive on the source that should have it.
    url: []const u8,
    /// The URL of the `.minisig` that verifies it. Upstream publishes one next
    /// to every archive; a Zig++ release is verified by the SHA-256 of its
    /// index instead.
    signature_url: ?[]const u8,
    /// The SHA-256 that an index published for it: a tagged upstream release
    /// publishes one, a dev build does not.
    shasum: ?[Sha256.digest_length]u8,
    /// The size in bytes, when an index says, for download progress.
    size: ?u64,
    /// Whether the community mirrors are worth trying when the URL above does
    /// not have the archive. They mirror upstream's downloads, not Zig++'s.
    mirrors: bool,
};

/// One place an archive can come from, with what verifies it there.
const Candidate = struct {
    url: []const u8,
    signature_url: ?[]const u8,
    shasum: ?[Sha256.digest_length]u8,
    size: ?u64,
};

/// The mirror base URLs in upstream's list, which is one URL per line, in the
/// order to try them. Lines that are not URLs are ignored: the list is data
/// from the network, and a comment or a blank line must not fail an install.
fn mirrorUrls(arena: Allocator, text: []const u8) Allocator.Error![]const []const u8 {
    var mirrors: std.ArrayList([]const u8) = .empty;
    var lines = mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const url = mem.trim(u8, line, " \t\r");
        if (!mem.startsWith(u8, url, "http")) continue;
        try mirrors.append(arena, url);
    }
    return mirrors.items;
}

/// The name of the file that a URL names: the last path segment, without the
/// query string. A mirror is asked for the same name, and the trusted comment
/// of a minisign signature names it.
fn filenameFromUrl(url: []const u8) []const u8 {
    const path = url[0 .. mem.indexOfScalar(u8, url, '?') orelse url.len];
    const slash = mem.lastIndexOfScalar(u8, path, '/') orelse return path;
    return path[slash + 1 ..];
}

/// An archive is installed only if its SHA-256 is the one an index published.
/// A checksum is 32 bytes, all of them, compared in constant time.
fn checksumMatches(expected: [Sha256.digest_length]u8, actual: [Sha256.digest_length]u8) bool {
    return std.crypto.timing_safe.eql([Sha256.digest_length]u8, actual, expected);
}

/// The archive of a version as an upstream-style download index describes it.
const Tarball = struct {
    url: []const u8,
    shasum: [Sha256.digest_length]u8,
    size: ?u64,
};

fn resolveArchive(
    arena: Allocator,
    http_client: *http.Client,
    root_prog_node: std.Progress.Node,
    host: Host,
    version_str: []const u8,
    source: Source,
) !Archive {
    const host_name = try host.name(arena);
    switch (source) {
        .zigpp_release => |release| {
            const index = try fetchIndex(arena, http_client, root_prog_node, release.index_url);
            const entry = objectGet(index.document, "master") orelse fatal(
                "the index of Zig++ release {s} has no 'master' entry: {s} (HTTP {d})",
                .{ release.tag, release.index_url, index.status },
            );
            // The release names the exact version it published; asking for a
            // different one means asking for a compiler that does not exist.
            const index_version = stringField(entry, "version") orelse fatal(
                "the index of Zig++ release {s} has no 'version': {s} (HTTP {d})",
                .{ release.tag, release.index_url, index.status },
            );
            if (!mem.eql(u8, index_version, version_str)) fatal(
                "Zig++ release {s} publishes {s}, not {s}: they must match exactly",
                .{ release.tag, index_version, version_str },
            );
            const tarball = try archiveFromIndexEntry(
                arena,
                entry,
                host_name,
                version_str,
                release.index_url,
                index.status,
            );
            // A Zig++ release is a GitHub release: its index carries the
            // SHA-256, there is no minisign signature, and the community
            // mirrors of upstream Zig do not have it.
            return .{
                .url = tarball.url,
                .signature_url = null,
                .shasum = tarball.shasum,
                .size = tarball.size,
                .mirrors = false,
            };
        },

        .upstream_release => |release| {
            const index = try fetchIndex(arena, http_client, root_prog_node, release.index_url);
            const entry = objectGet(index.document, version_str) orelse fatal(
                "no version {s} in {s} (HTTP {d}); it lists: {s}",
                .{
                    version_str,
                    release.index_url,
                    index.status,
                    try joinKeysExcept(arena, index.document, &.{"master"}),
                },
            );
            const tarball = try archiveFromIndexEntry(
                arena,
                entry,
                host_name,
                version_str,
                release.index_url,
                index.status,
            );
            return .{
                .url = tarball.url,
                .signature_url = try std.fmt.allocPrint(arena, "{s}.minisig", .{tarball.url}),
                .shasum = tarball.shasum,
                .size = tarball.size,
                .mirrors = true,
            };
        },

        .upstream_dev => |dev| return .{
            .url = dev.tarball_url,
            .signature_url = try std.fmt.allocPrint(arena, "{s}.minisig", .{dev.tarball_url}),
            .shasum = null,
            .size = null,
            .mirrors = true,
        },
    }
}

const EntryError = error{
    /// The entry has no archive for this host.
    UnsupportedHost,
    /// The entry has no tarball URL, or no usable SHA-256.
    MissingTarball,
    MissingChecksum,
    MalformedChecksum,
};

/// The archive of `host_name` in an entry of an upstream-style download index,
/// e.g. `{"version": ..., "x86_64-linux": {"tarball": ..., "shasum": ...}}`.
fn archiveFromEntry(entry: std.json.Value, host_name: []const u8) EntryError!Tarball {
    const host_entry = objectGet(entry, host_name) orelse return error.UnsupportedHost;
    const url = stringField(host_entry, "tarball") orelse return error.MissingTarball;
    const shasum_text = stringField(host_entry, "shasum") orelse return error.MissingChecksum;
    var shasum: [Sha256.digest_length]u8 = undefined;
    _ = std.fmt.hexToBytes(&shasum, shasum_text) catch return error.MalformedChecksum;
    return .{ .url = url, .shasum = shasum, .size = sizeField(host_entry) };
}

/// `archiveFromEntry`, with errors that say what to do about them.
fn archiveFromIndexEntry(
    arena: Allocator,
    entry: std.json.Value,
    host_name: []const u8,
    version_str: []const u8,
    index_url: []const u8,
    index_status: http.Status,
) !Tarball {
    return archiveFromEntry(entry, host_name) catch |err| switch (err) {
        error.UnsupportedHost => fatal(
            "no build of {s} for {s} in {s} (HTTP {d}); it has builds for: {s}",
            .{
                version_str,
                host_name,
                index_url,
                index_status,
                try joinKeysExcept(arena, entry, &.{ "version", "date" }),
            },
        ),
        error.MissingTarball => fatal("{s} has no tarball for {s} on {s}", .{ index_url, version_str, host_name }),
        error.MissingChecksum => fatal("{s} has no shasum for {s} on {s}", .{ index_url, version_str, host_name }),
        error.MalformedChecksum => fatal("{s} has a shasum for {s} on {s} that is not a SHA-256", .{
            index_url, version_str, host_name,
        }),
    };
}

fn objectGet(value: std.json.Value, key: []const u8) ?std.json.Value {
    const object = switch (value) {
        .object => |object| object,
        else => return null,
    };
    return object.get(key);
}

fn stringField(value: std.json.Value, key: []const u8) ?[]const u8 {
    const field = objectGet(value, key) orelse return null;
    return switch (field) {
        .string => |string| string,
        else => null,
    };
}

fn sizeField(value: std.json.Value) ?u64 {
    const field = objectGet(value, "size") orelse return null;
    return switch (field) {
        .string => |string| std.fmt.parseInt(u64, string, 10) catch null,
        .integer => |integer| if (integer < 0) null else @intCast(integer),
        else => null,
    };
}

/// The keys of an object as a sorted, ", "-joined list, for the errors that
/// list what an index has.
fn joinKeysExcept(arena: Allocator, value: std.json.Value, skip: []const []const u8) Allocator.Error![]const u8 {
    const object = switch (value) {
        .object => |object| object,
        else => return arena.dupe(u8, ""),
    };

    var keys: std.ArrayList([]const u8) = .empty;
    for (object.keys()) |key| {
        for (skip) |skipped| {
            if (mem.eql(u8, skipped, key)) break;
        } else {
            try keys.append(arena, key);
        }
    }
    std.mem.sort([]const u8, keys.items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);
    return std.mem.join(arena, ", ", keys.items);
}

/// Downloads an index into memory and parses it, complaining about the HTTP
/// status if it cannot: an index that is not there means the version (or the
/// release that would have published it) is not there either.
const Index = struct {
    document: std.json.Value,
    /// The status of the successful download, which the errors that say what
    /// an index has also report.
    status: http.Status,
};

fn fetchIndex(
    arena: Allocator,
    http_client: *http.Client,
    root_prog_node: std.Progress.Node,
    url: []const u8,
) !Index {
    const prog_node = root_prog_node.startFmt(0, "download index {s}", .{url});
    defer prog_node.end();

    var body: std.Io.Writer.Allocating = .init(arena);
    const result = http_client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &body.writer,
    }) catch |err| fatal("unable to download {s}: {t}", .{ url, err });
    const status = result.status;
    if (status != .ok) fatal("HTTP {d} {s} from {s}", .{ status, status.phrase() orelse "", url });

    const document = std.json.parseFromSliceLeaky(std.json.Value, arena, body.written(), .{}) catch |err|
        fatal("unable to parse {s} as JSON: {t}", .{ url, err });
    return .{ .document = document, .status = status };
}

pub const usage_any_install =
    \\Usage: zig any-install <version>
    \\
    \\    Download an exact Zig++ or Zig version, verify it, and unpack it
    \\    into <global cache>/any/<version>, where `zig any` and the automatic
    \\    version dispatch find it.
    \\
    \\    An upstream Zig archive is verified against the minisign signature
    \\    that ziglang.org publishes next to it, with the Zig Software
    \\    Foundation's key; a tagged release is checked against the SHA-256 of
    \\    the upstream download index as well. An upstream archive that
    \\    ziglang.org no longer has is looked for on the community mirrors of
    \\    https://ziglang.org/download/community-mirrors.txt.
    \\
    \\Options:
    \\  -h, --help             Print this help and exit
    \\
    \\
;

const testing = std.testing;

const zigpp_version = "0.17.0-dev.2361+zigpp.5b96e6d21";

test "sourceFromVersion: Zig++ release tag and index URL" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const source = try sourceFromVersion(arena, .{ .arch = .x86_64, .os = .linux }, zigpp_version);
    switch (source) {
        .zigpp_release => |release| {
            try testing.expectEqualStrings("zigpp-0.17.0-dev.2361", release.tag);
            try testing.expectEqualStrings(
                "https://github.com/mattneel/zigpp/releases/download/zigpp-0.17.0-dev.2361/index.json",
                release.index_url,
            );
        },
        else => return error.TestUnexpectedResult,
    }
}

test "sourceFromVersion: a version without zigpp metadata is not a Zig++ version" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const host: Host = .{ .arch = .x86_64, .os = .linux };
    // The same shape as a Zig++ version, but the metadata names something else.
    const source = try sourceFromVersion(arena, host, "0.17.0-dev.2267+48abf1a34");
    switch (source) {
        .upstream_dev => |dev| try testing.expectEqualStrings(
            "https://ziglang.org/builds/zig-x86_64-linux-0.17.0-dev.2267+48abf1a34.tar.xz",
            dev.tarball_url,
        ),
        else => return error.TestUnexpectedResult,
    }

    // A tagged release, with no metadata at all.
    const release = try sourceFromVersion(arena, host, "0.15.1");
    switch (release) {
        .upstream_release => |upstream| try testing.expectEqualStrings(
            "https://ziglang.org/download/index.json",
            upstream.index_url,
        ),
        else => return error.TestUnexpectedResult,
    }

    try testing.expectError(error.InvalidVersion, sourceFromVersion(arena, host, "master"));
}

test "filenameFromUrl: the last path segment, without the query" {
    try testing.expectEqualStrings(
        "zig-x86_64-linux-0.15.1.tar.xz",
        filenameFromUrl("https://ziglang.org/download/0.15.1/zig-x86_64-linux-0.15.1.tar.xz"),
    );
    try testing.expectEqualStrings(
        "zig-x86_64-linux-0.15.1.tar.xz",
        filenameFromUrl("https://pkg.hexops.org/zig/zig-x86_64-linux-0.15.1.tar.xz?source=zigpp"),
    );
    try testing.expectEqualStrings(
        "zig-x86_64-linux-0.16.0-dev.1+abcdef012.tar.xz",
        filenameFromUrl("https://ziglang.org/builds/zig-x86_64-linux-0.16.0-dev.1+abcdef012.tar.xz"),
    );
}

test "Host: names, executables, and archive kinds" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const linux: Host = .{ .arch = .x86_64, .os = .linux };
    try testing.expectEqualStrings("x86_64-linux", try linux.name(arena));
    try testing.expectEqualStrings("zig", linux.exeName());
    try testing.expectEqualStrings("tar.xz", linux.archiveExtension());
    try testing.expectEqualStrings("zig-x86_64-linux-0.15.1", try linux.archiveStem(arena, "0.15.1"));

    const macos: Host = .{ .arch = .aarch64, .os = .macos };
    try testing.expectEqualStrings("aarch64-macos", try macos.name(arena));
    try testing.expectEqualStrings("tar.xz", macos.archiveExtension());

    // Windows releases are zip files, and their executable is zig.exe.
    const windows: Host = .{ .arch = .x86_64, .os = .windows };
    try testing.expectEqualStrings("x86_64-windows", try windows.name(arena));
    try testing.expectEqualStrings("zig.exe", windows.exeName());
    try testing.expectEqualStrings("zip", windows.archiveExtension());
    try testing.expectEqualStrings("zig-x86_64-windows-0.15.1", try windows.archiveStem(arena, "0.15.1"));

    // A dev build on Windows derives a zip URL.
    const dev = try sourceFromVersion(arena, windows, "0.16.0-dev.1234+abcdef012");
    switch (dev) {
        .upstream_dev => |d| try testing.expectEqualStrings(
            "https://ziglang.org/builds/zig-x86_64-windows-0.16.0-dev.1234+abcdef012.zip",
            d.tarball_url,
        ),
        else => return error.TestUnexpectedResult,
    }
}

test "archiveFromEntry" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena,
        \\{"master": {
        \\  "version": "0.15.1",
        \\  "date": "2025-08-19",
        \\  "x86_64-linux": {
        \\    "tarball": "https://ziglang.org/download/0.15.1/zig-x86_64-linux-0.15.1.tar.xz",
        \\    "shasum": "c61c5da6edeea14ca51ecd5e4520c6f4189ef5250383db33d01848293bfafe05",
        \\    "size": "53734456"
        \\  }
        \\}}
    , .{});

    const entry = objectGet(index, "master").?;
    const tarball = try archiveFromEntry(entry, "x86_64-linux");
    try testing.expectEqualStrings(
        "https://ziglang.org/download/0.15.1/zig-x86_64-linux-0.15.1.tar.xz",
        tarball.url,
    );
    try testing.expectEqual(@as(?u64, 53734456), tarball.size);
    var expected_shasum: [Sha256.digest_length]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected_shasum, "c61c5da6edeea14ca51ecd5e4520c6f4189ef5250383db33d01848293bfafe05");
    try testing.expectEqualSlices(u8, &expected_shasum, &tarball.shasum);

    try testing.expectError(error.UnsupportedHost, archiveFromEntry(entry, "aarch64-macos"));

    const malformed = try std.json.parseFromSliceLeaky(std.json.Value, arena,
        \\{"x86_64-linux": {"tarball": "https://example.com/zig.tar.xz", "shasum": "not a sha256"}}
    , .{});
    try testing.expectError(error.MalformedChecksum, archiveFromEntry(malformed, "x86_64-linux"));

    const missing = try std.json.parseFromSliceLeaky(std.json.Value, arena,
        \\{"x86_64-linux": {"tarball": "https://example.com/zig.tar.xz"}}
    , .{});
    try testing.expectError(error.MissingChecksum, archiveFromEntry(missing, "x86_64-linux"));
    const no_tarball = try std.json.parseFromSliceLeaky(std.json.Value, arena,
        \\{"x86_64-linux": {}}
    , .{});
    try testing.expectError(error.MissingTarball, archiveFromEntry(no_tarball, "x86_64-linux"));
    try testing.expectError(error.UnsupportedHost, archiveFromEntry(no_tarball, "aarch64-macos"));
}

test "checksumMatches: a tampered archive is rejected" {
    // The published digest of zig-x86_64-linux-0.15.1.tar.xz.
    var published: [Sha256.digest_length]u8 = undefined;
    _ = try std.fmt.hexToBytes(&published, "c61c5da6edeea14ca51ecd5e4520c6f4189ef5250383db33d01848293bfafe05");
    try testing.expect(checksumMatches(published, published));

    // Anything else is rejected, down to a difference in the last byte.
    var tampered = published;
    tampered[tampered.len - 1] ^= 1;
    try testing.expect(!checksumMatches(published, tampered));
    tampered = published;
    tampered[0] ^= 1;
    try testing.expect(!checksumMatches(published, tampered));

    const empty: [Sha256.digest_length]u8 = @splat(0);
    try testing.expect(!checksumMatches(published, empty));
    try testing.expect(!checksumMatches(empty, published));
}

test "joinKeysExcept" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const value = try std.json.parseFromSliceLeaky(std.json.Value, arena,
        \\{"version": "0.15.1", "date": "2025-08-19", "x86_64-linux": {}, "aarch64-macos": {}}
    , .{});
    try testing.expectEqualStrings("aarch64-macos, x86_64-linux", try joinKeysExcept(arena, value, &.{ "version", "date" }));
}

/// An `Install` for tests that only exercise its file-system parts: a global
/// cache in a temporary directory, and nothing else that is used. `init`
/// installs into `t` itself, because `t.install.arena` points at
/// `t.arena_instance` and must not move.
const TestInstall = struct {
    arena_instance: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    http_client: http.Client,
    install: Install,

    fn init(t: *TestInstall, version_str: []const u8) !void {
        const io = std.testing.io;
        t.tmp = std.testing.tmpDir(.{ .iterate = true });
        t.arena_instance = .init(testing.allocator);
        t.http_client = .{ .allocator = testing.allocator, .io = io };
        const arena = t.arena_instance.allocator();

        var path_buffer: [Dir.max_path_bytes]u8 = undefined;
        const path_len = try t.tmp.dir.realPath(io, &path_buffer);

        t.install = .{
            .arena = arena,
            .gpa = testing.allocator,
            .io = io,
            .http_client = &t.http_client,
            .root_prog_node = .none,
            .host = .{ .arch = builtin.cpu.arch, .os = native_os },
            .version_str = version_str,
            .global_cache_root = .{
                .path = try arena.dupe(u8, path_buffer[0..path_len]),
                .handle = t.tmp.dir,
            },
        };
    }

    fn deinit(t: *TestInstall) void {
        t.http_client.deinit();
        t.tmp.cleanup();
        t.arena_instance.deinit();
    }
};

test "renaming a version into the cache: a collision means already installed" {
    const io = std.testing.io;
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;

    // What `installArchive` has just before the rename: a version unpacked in
    // a temporary directory, with the compiler in it.
    const unpacked_root = "zig-x86_64-linux-0.15.1";
    try install.global_cache_root.handle.createDirPath(io, "tmp/0.15.1-abc/" ++ unpacked_root);
    try install.global_cache_root.handle.writeFile(io, .{
        .sub_path = "tmp/0.15.1-abc/" ++ unpacked_root ++ "/zig",
        .data = "the compiler",
    });
    var tmp_dir = try install.global_cache_root.handle.openDir(io, "tmp/0.15.1-abc", .{ .iterate = true });
    defer tmp_dir.close(io);
    try install.global_cache_root.handle.createDirPath(io, versions_dir_name);

    const version_sub_path = try versionSubPath(install.arena, install.version_str);
    try testing.expectEqual(
        Install.Published.installed,
        try install.renameIntoVersions(tmp_dir, unpacked_root, version_sub_path),
    );

    // The version is where the launcher looks for it, and the temporary
    // directory no longer has it: the rename moved it, it was not copied.
    const installed_rel_path = try std.fmt.allocPrint(install.arena, "{s}/{s}", .{
        version_sub_path, install.host.exeName(),
    });
    try testing.expectEqualStrings("the compiler", try install.global_cache_root.handle.readFileAlloc(
        io,
        installed_rel_path,
        install.arena,
        .limited(64),
    ));
    try Dir.cwd().access(io, try installedExePath(
        install.arena,
        install.global_cache_root,
        install.host,
        install.version_str,
    ), .{});
    try testing.expectError(error.FileNotFound, tmp_dir.access(io, unpacked_root ++ "/zig", .{}));

    // Another process unpacked and renamed the same version in the meantime:
    // that install stays, and the rename reports the collision.
    try tmp_dir.createDirPath(io, unpacked_root);
    try tmp_dir.writeFile(io, .{ .sub_path = unpacked_root ++ "/zig", .data = "the other compiler" });
    try testing.expectEqual(
        Install.Published.already_installed,
        try install.renameIntoVersions(tmp_dir, unpacked_root, version_sub_path),
    );
    try testing.expectEqualStrings("the compiler", try install.global_cache_root.handle.readFileAlloc(
        io,
        installed_rel_path,
        install.arena,
        .limited(64),
    ));
}

test "a version directory without its compiler is not an install" {
    const io = std.testing.io;
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;

    // Nothing is installed at all, so there is no directory to delete.
    try testing.expectError(error.BrokenInstall, install.checkInstalled());
    try testing.expect(mem.indexOf(u8, install.message.?, "no compiler was installed at") != null);

    // A directory without the compiler in it -- a partial delete, or an
    // antivirus that took the executable away -- is not an install, and the
    // failure names the directory that is in the way.
    install.message = null;
    const version_dir = try install.versionDirPath();
    try install.global_cache_root.handle.createDirPath(
        io,
        try versionSubPath(install.arena, install.version_str),
    );
    try testing.expectError(error.BrokenInstall, install.checkInstalled());
    try testing.expect(mem.indexOf(u8, install.message.?, version_dir) != null);
    try testing.expect(mem.indexOf(u8, install.message.?, "delete it") != null);

    // With the compiler in it, it is an install.
    install.message = null;
    try install.global_cache_root.handle.writeFile(io, .{
        .sub_path = try std.fmt.allocPrint(install.arena, "{s}/{s}", .{
            try versionSubPath(install.arena, install.version_str), install.host.exeName(),
        }),
        .data = "the compiler",
    });
    try install.checkInstalled();
    try testing.expectEqual(@as(?[]const u8, null), install.message);
}

test "unpackedRoot: the archive's own directory names the root" {
    const io = std.testing.io;
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;

    // Zig 0.14 and earlier name the root zig-<os>-<arch>-<version>; the
    // downloaded archive sits next to it and is not the root.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        try tmp.dir.createDir(io, "zig-linux-x86_64-0.14.0", .default_dir);
        try tmp.dir.writeFile(io, .{
            .sub_path = "zig-x86_64-linux-0.14.0.tar.xz",
            .data = "archive bytes",
        });
        try testing.expectEqualStrings(
            "zig-linux-x86_64-0.14.0",
            try install.unpackedRoot(tmp.dir, "tmp/0.14.0"),
        );
    }

    // Zig 0.15 and later say zig-<arch>-<os>-<version>.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        try tmp.dir.createDir(io, "zig-x86_64-linux-0.15.1", .default_dir);
        try tmp.dir.writeFile(io, .{
            .sub_path = "zig-x86_64-linux-0.15.1.tar.xz",
            .data = "archive bytes",
        });
        try testing.expectEqualStrings(
            "zig-x86_64-linux-0.15.1",
            try install.unpackedRoot(tmp.dir, "tmp/0.15.1"),
        );
    }

    // Two directories, or none, is not the archive of one version.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        try tmp.dir.createDir(io, "one", .default_dir);
        try tmp.dir.createDir(io, "two", .default_dir);
        try testing.expectError(error.InvalidArchive, install.unpackedRoot(tmp.dir, "tmp/0.15.1"));
        try testing.expect(mem.indexOf(u8, install.message.?, "more than one directory") != null);
    }
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        try testing.expectError(error.InvalidArchive, install.unpackedRoot(tmp.dir, "tmp/0.15.1"));
        try testing.expect(mem.indexOf(u8, install.message.?, "no directory") != null);
    }
}

test "the candidates of an archive: the primary source, then the mirrors" {
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;

    // A Zig++ release comes from GitHub, and from nowhere else.
    const zigpp: Archive = .{
        .url = "https://github.com/mattneel/zigpp/releases/download/zigpp-0.17.0-dev.2361/zig-x86_64-linux-0.17.0-dev.2361+zigpp.5b96e6d21.tar.xz",
        .signature_url = null,
        .shasum = null,
        .size = null,
        .mirrors = false,
    };
    const zigpp_candidates = try install.archiveSources(zigpp);
    try testing.expectEqual(@as(usize, 1), zigpp_candidates.len);
    try testing.expectEqualStrings(zigpp.url, zigpp_candidates[0].url);
    try testing.expectEqual(@as(?[]const u8, null), zigpp_candidates[0].signature_url);

    // An upstream archive comes from ziglang.org first, then from every
    // mirror in a random order, and every candidate knows where the signature
    // of the file it serves is.
    install.mirrors = &.{ "https://pkg.hexops.org/zig", "https://zig.squirl.dev" };
    const archive: Archive = .{
        .url = "https://ziglang.org/builds/zig-x86_64-linux-0.16.0-dev.1+abcdef012.tar.xz",
        .signature_url = "https://ziglang.org/builds/zig-x86_64-linux-0.16.0-dev.1+abcdef012.tar.xz.minisig",
        .shasum = null,
        .size = null,
        .mirrors = true,
    };
    const candidates = try install.archiveSources(archive);
    try testing.expectEqual(@as(usize, 3), candidates.len);
    try testing.expectEqualStrings(archive.url, candidates[0].url);
    try testing.expectEqualStrings(archive.signature_url.?, candidates[0].signature_url.?);

    const filename = "zig-x86_64-linux-0.16.0-dev.1+abcdef012.tar.xz";
    var seen_mirrors: usize = 0;
    for (candidates[1..]) |candidate| {
        if (mem.eql(u8, candidate.url, "https://pkg.hexops.org/zig/" ++ filename ++ "?source=zigpp")) {
            try testing.expectEqualStrings(
                "https://pkg.hexops.org/zig/" ++ filename ++ ".minisig?source=zigpp",
                candidate.signature_url.?,
            );
        } else if (mem.eql(u8, candidate.url, "https://zig.squirl.dev/" ++ filename ++ "?source=zigpp")) {
            try testing.expectEqualStrings(
                "https://zig.squirl.dev/" ++ filename ++ ".minisig?source=zigpp",
                candidate.signature_url.?,
            );
        } else {
            return error.TestUnexpectedResult;
        }
        try testing.expectEqual(archive.shasum, candidate.shasum);
        seen_mirrors += 1;
    }
    try testing.expectEqual(@as(usize, 2), seen_mirrors);
}

test "mirrorUrls: the community mirror list, as upstream publishes it" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    const text =
        \\https://pkg.hexops.org/zig
        \\https://zig.squirl.dev
        \\
        \\# a line that is not a mirror
        \\  https://zigmirror.com
        \\https://zig.chainsafe.dev
        \\
    ;
    const mirrors = try mirrorUrls(arena, text);
    try testing.expectEqual(@as(usize, 4), mirrors.len);
    try testing.expectEqualStrings("https://pkg.hexops.org/zig", mirrors[0]);
    try testing.expectEqualStrings("https://zig.squirl.dev", mirrors[1]);
    // A URL that has whitespace around it is still a URL.
    try testing.expectEqualStrings("https://zigmirror.com", mirrors[2]);
    try testing.expectEqualStrings("https://zig.chainsafe.dev", mirrors[3]);

    // A list with nothing usable in it is empty, not an error.
    try testing.expectEqual(@as(usize, 0), (try mirrorUrls(arena, "\n# nothing\n")).len);
    try testing.expectEqual(@as(usize, 0), (try mirrorUrls(arena, "")).len);
}

test "the community mirrors are downloaded once, and shuffled" {
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;

    // Without a download that could happen, the mirrors are what a download
    // has left there, and asking again answers with the same list.
    install.mirrors = &.{ "https://one.example", "https://two.example" };
    const mirrors = try install.communityMirrors();
    try testing.expectEqual(@as(usize, 2), mirrors.len);
    try testing.expectEqualStrings("https://one.example", mirrors[0]);
    try testing.expectEqual(mirrors.ptr, install.mirrors.?.ptr);
}

test "a signature that is not a signature is a source failure; one that does not verify is not" {
    const io = std.testing.io;
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;

    const candidate: Candidate = .{
        .url = "https://mirror.example/zig-x86_64-linux-0.15.1.tar.xz",
        .signature_url = "https://mirror.example/zig-x86_64-linux-0.15.1.tar.xz.minisig",
        .shasum = null,
        .size = null,
    };

    // A proxy's error page where a signature should be: this source has no
    // signature for the file, so it is not a source at all, and the next one
    // is tried.
    const not_a_signature = try install.parseSignature(candidate, "an error page, not a signature");
    try testing.expect(not_a_signature == null);
    try testing.expectEqual(@as(usize, 1), install.attempts.items.len);
    try testing.expectEqualStrings(candidate.url, install.attempts.items[0].url);
    try testing.expect(mem.indexOf(u8, install.attempts.items[0].reason, "not a minisign signature") != null);

    // The signature of zig-x86_64-linux-0.15.1.tar.xz, with its trusted
    // comment edited after the fact: it parses, and it does not verify, which
    // is an error instead of a reason to try somewhere else.
    const edited_comment_signature =
        "untrusted comment: signature from minisign secret key\n" ++
        "RUSGOq2NVecA2Wk3Npw2IGzIC8fMec6YOFiKRo3F7tmaEgdVL8wk06amCA64FxoC8iN6+FyllUz+mQteoOXYlEYFkZYUBG2Bngo=\n" ++
        "trusted comment: timestamp:1755707122\tfile:zig-x86_64-linux-0.15.1.tar.xz\thashed\n" ++
        "ynslZhi+qM0o9VFWdEWj0/Wsn1zuW4ubHyI6a4AeJW2DdLq04d0TPuMMWgnFBKcxfjhFhBUmaqr156dIlIoFDA==\n";
    const signature = (try install.parseSignature(candidate, edited_comment_signature)).?;
    var archive_file = try install.global_cache_root.handle.createFile(io, "not the archive", .{});
    defer archive_file.close(io);
    const digests: Digests = .{ .sha256 = @splat(0), .blake2b512 = @splat(0) };
    try testing.expectError(
        error.BadSignature,
        install.verifySignature(signature, candidate, digests, archive_file),
    );
    try testing.expect(mem.indexOf(u8, install.message.?, "does not verify") != null);
}

test "the installed compiler is found through a symlinked cache" {
    // Creating a symlink needs privileges on Windows, and the identity check
    // that uses this comparison is a POSIX loop guard.
    if (native_os == .windows) return error.SkipZigTest;
    const io = std.testing.io;
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;
    const arena = install.arena;

    // The compiler of the version, in the cache.
    try install.global_cache_root.handle.createDirPath(io, "any/0.15.1");
    try install.global_cache_root.handle.writeFile(io, .{
        .sub_path = "any/0.15.1/zig",
        .data = "the compiler",
    });
    const exe_path = try installedExePath(arena, install.global_cache_root, install.host, "0.15.1");

    // The same file, reached through a symlink to the cache directory. This is
    // what the version dispatch compares against its own executable, and the
    // comparison must not care which path the cache was reached by.
    const link_path = try std.fmt.allocPrint(arena, "{s}-link", .{install.global_cache_root.path.?});
    Dir.cwd().deleteTree(io, link_path) catch {};
    try Dir.cwd().symLink(io, install.global_cache_root.path.?, link_path, .{ .is_directory = true });
    defer Dir.cwd().deleteTree(io, link_path) catch {};
    const via_link = try std.fmt.allocPrint(arena, "{s}/{s}/{s}/{s}", .{
        link_path, versions_dir_name, "0.15.1", install.host.exeName(),
    });

    try testing.expect(std.zig.isSameFile(io, arena, exe_path, exe_path));
    try testing.expect(std.zig.isSameFile(io, arena, exe_path, via_link));
    try testing.expect(std.zig.isSameFile(io, arena, via_link, exe_path));

    // Another version's compiler is another file, and so is a version that is
    // not installed at all.
    try install.global_cache_root.handle.createDirPath(io, "any/0.15.2");
    try install.global_cache_root.handle.writeFile(io, .{
        .sub_path = "any/0.15.2/zig",
        .data = "the other compiler",
    });
    const other_exe = try installedExePath(arena, install.global_cache_root, install.host, "0.15.2");
    try testing.expect(!std.zig.isSameFile(io, arena, exe_path, other_exe));
    const missing_exe = try installedExePath(arena, install.global_cache_root, install.host, "0.15.0");
    try testing.expect(!std.zig.isSameFile(io, arena, exe_path, missing_exe));
}

test "a failed install leaves nothing in the cache's temporary directory" {
    const io = std.testing.io;
    var t: TestInstall = undefined;
    try t.init("0.15.1");
    defer t.deinit();
    const install = &t.install;

    // A source that cannot be asked at all: the temporary directory and the
    // archive file in it are created, the download fails, and the failure is
    // reported with the temporary directory already gone.
    try testing.expectError(error.NoSource, install.installArchive(.{
        .url = "not a URL",
        .signature_url = null,
        .shasum = null,
        .size = null,
        .mirrors = false,
    }));
    try testing.expectEqual(@as(usize, 1), install.attempts.items.len);
    try testing.expect(mem.indexOf(u8, install.attempts.items[0].reason, "invalid URL") != null);
    try expectNoLeftovers(io, install.global_cache_root.handle, tmp_dir_name);

    // The same when a source has no signature: a source whose archive nothing
    // verifies is no source, and it leaves no temporary tree either.
    install.attempts.clearRetainingCapacity();
    try testing.expectError(error.NoSource, install.installArchive(.{
        .url = "https://mirror.example/zig-x86_64-linux-0.15.1.tar.xz",
        .signature_url = "not a URL either",
        .shasum = null,
        .size = null,
        .mirrors = false,
    }));
    try testing.expectEqual(@as(usize, 1), install.attempts.items.len);
    try testing.expect(mem.indexOf(u8, install.attempts.items[0].reason, "no minisign signature") != null);
    try expectNoLeftovers(io, install.global_cache_root.handle, tmp_dir_name);
}

/// Fails when the directory `sub_path` exists and has anything in it: a failed
/// install must leave no temporary tree behind.
fn expectNoLeftovers(io: Io, dir: Dir, sub_path: []const u8) !void {
    var tmp = dir.openDir(io, sub_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => |e| return e,
    };
    defer tmp.close(io);
    var iterator = tmp.iterate();
    while (try iterator.next(io)) |entry| {
        std.debug.print("left over in {s}: {s}\n", .{ sub_path, entry.name });
        return error.TestUnexpectedResult;
    }
}
