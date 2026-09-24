# Installing

Zig++ is distributed as release archives, one per target. The
[Release workflow](https://github.com/mattneel/zigpp/blob/master/.github/workflows/release.yml)
publishes a release on every push to master, for these four targets:

| Target | Archive |
| --- | --- |
| x86_64-linux | `zig-x86_64-linux-<version>.tar.xz` |
| aarch64-linux | `zig-aarch64-linux-<version>.tar.xz` |
| aarch64-macos | `zig-aarch64-macos-<version>.tar.xz` |
| x86_64-windows | `zig-x86_64-windows-<version>.zip` |

The [Downloads](downloads.md) chapter lists the archives of every release with
their SHA-256 checksums, and [Versions and Releases](versions.md) explains what
`<version>` looks like.

There is no other way to install Zig++ yet: no package manager, and no
pre-built binary from anywhere but these releases. A Zig++ compiler built from
source is the other option; see [Building from Source](building-from-source.md).

## Unpack it anywhere

A Zig installation is two things: the `zig` executable, and the `lib/`
directory. At runtime, the executable searches up the file system for `lib/`,
relative to itself:

* `lib/`
* `lib/zig/`
* `../lib/`
* `../lib/zig/`
* and so on

In other words, you can unpack a release of Zig++ anywhere and begin using it
immediately, with no installation step, although the search also supports
installing it globally (`/usr/bin/zig` and `/usr/lib/zig/`).

A release archive holds:

```text
zig-x86_64-linux-0.17.0-dev.2361+zigpp.5b96e6d21/
    zig                 the compiler
    lib/                the standard library, builtin headers, and build system
    doc/langref.html    the language reference
    LICENSE
    README.md
```

On Windows the executable is `zig.exe`, and the archive is a `.zip`.

```sh
tar -xJf zig-x86_64-linux-0.17.0-dev.2361+zigpp.5b96e6d21.tar.xz
./zig-x86_64-linux-0.17.0-dev.2361+zigpp.5b96e6d21/zig version
```

```text
0.17.0-dev.2361+zigpp.5b96e6d21
```

## Verify the download

Each release has a `SHA256SUMS` asset: the SHA-256 of every archive and of
`index.json`, in the format `sha256sum` reads. Download the release's assets
into one directory, then check them:

```sh
sha256sum -c SHA256SUMS
```

If you downloaded only the archive for your target, check that one line and
skip the rest:

```sh
sha256sum -c --ignore-missing SHA256SUMS
```

```text
zig-aarch64-linux-0.17.0-dev.2361+zigpp.5b96e6d21.tar.xz: OK
zig-aarch64-macos-0.17.0-dev.2361+zigpp.5b96e6d21.tar.xz: OK
zig-x86_64-linux-0.17.0-dev.2361+zigpp.5b96e6d21.tar.xz: OK
zig-x86_64-windows-0.17.0-dev.2361+zigpp.5b96e6d21.zip: OK
index.json: OK
```

The same hashes, with the size of each archive, are in
[`index.json`](/download/index.json), which uses the format of
[ziglang.org's download index](https://ziglang.org/download/index.json), so
tools that read that one can read this one.

## Documentation for the version you have

* The language reference is `doc/langref.html` inside the archive, and this
  site serves the newest release's copy at [/langref.html](/langref.html).
* The standard library documentation is served by the compiler itself:

  ```sh
  zig std
  ```

  It generates the autodocs for the `lib/` next to the executable and opens a
  browser tab. This site serves the newest release's copy at
  [/std/](/std/index.html).

## Next

* [What Zig++ Adds](what-zigpp-adds.md) for the language features.
* [GPU Programming](gpu.md) to run Zig++ on an NVIDIA or AMD GPU.
* [Building from Source](building-from-source.md) for targets without a
  release, for the `zig2` bootstrap compiler, and for LLVM 23 development.
