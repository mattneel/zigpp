# Installing

Installing Zig++ is one line. On Linux and macOS:

```sh
curl -fsSL https://zigpp.lol/ppup | sh
```

On Windows, in PowerShell:

```powershell
irm https://zigpp.lol/ppup.ps1 | iex
```

Both run `ppup`, Zig++'s installer and toolchain manager. It unpacks the newest
release, makes it the default `zig`, installs `ppup` itself, and puts it on your
`PATH` so you can install more versions later. From then on:

```sh
zig version
```

```text
0.17.0-dev.2380+zigpp.add158e97
```

## ppup

| Command | What it does |
| --- | --- |
| `ppup` | Install the newest release as the default toolchain, install ppup, and add it to `PATH` |
| `ppup update` | Install the newest release and make it the default |
| `ppup install <version\|latest>` | Install a version, e.g. `ppup install 0.17.0-dev.2380+zigpp.add158e97` |
| `ppup default [<version>]` | Show the default toolchain, or make a version the default |
| `ppup list` | List the installed toolchains, marking the default |
| `ppup uninstall <version>` | Remove a toolchain; the default one refuses until another is the default |
| `ppup self update` | Replace ppup with the newest one from zigpp.lol |
| `ppup self uninstall` | Remove every toolchain, ppup, and its `PATH` entry |
| `ppup help` | List the commands, and `ppup --version` prints ppup's own version |

Where the toolchains live:

| | |
| --- | --- |
| Linux, macOS | `~/.zigpp`: `toolchains/<version>/` holds each release, `bin/zig` is the default compiler, `bin/ppup` is ppup |
| Windows | `%LOCALAPPDATA%\zigpp`: `toolchains\<version>\` holds each release, `current` is a directory junction to the default one, `bin\ppup.ps1` (and a `bin\ppup.cmd` shim) is ppup |

`ppup` adds two lines to the profile of your shell—`~/.zshrc`, `~/.bashrc`,
`~/.config/fish/conf.d/ppup.fish`, or `~/.profile`: a `# Zig++ (ppup)` comment
and the line that puts its `bin` directory on `PATH`.
`ppup self uninstall` removes them again. The environment variables
`PPUP_HOME` and `PPUP_NO_MODIFY_PATH=1` (the `--no-modify-path` option does the
same) move the installation and keep ppup out of your profile:

```sh
curl -fsSL https://zigpp.lol/ppup | sh -s -- --no-modify-path
```

```sh
PPUP_HOME=/opt/zigpp curl -fsSL https://zigpp.lol/ppup | sh
```

The installed `ppup` keeps to the home it is installed in, so `PPUP_HOME` is
needed only for the install.

Zig++ publishes releases for these four targets; anything else (an
`x86_64` Mac, say) fails with a message listing them:

| Target | Archive |
| --- | --- |
| x86_64-linux | `zig-x86_64-linux-<version>.tar.xz` |
| aarch64-linux | `zig-aarch64-linux-<version>.tar.xz` |
| aarch64-macos | `zig-aarch64-macos-<version>.tar.xz` |
| x86_64-windows | `zig-x86_64-windows-<version>.zip` |

The Windows build is `x86_64` only, so on ARM64 Windows ppup installs that one
and says that it runs under emulation. The
[Release workflow](https://github.com/mattneel/zigpp/blob/master/.github/workflows/release.yml)
publishes a release on every push to master, and the [Downloads](downloads.md)
chapter lists the archives of every release with their SHA-256 checksums;
[Versions and Releases](versions.md) explains what `<version>` looks like.

There is no package manager for Zig++ and no pre-built binary from anywhere but
these releases: `ppup`, or unpacking an archive by hand, are the two ways in. A
compiler built from source is the other option; see
[Building from Source](building-from-source.md).

## Unpack an archive anywhere

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
installing it globally (`/usr/bin/zig` and `/usr/lib/zig/`). This is all
`ppup` does: it unpacks `toolchains/<version>/` and points `bin/zig` at it.

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

`ppup` checks the same sums itself, and unpacks nothing that fails the check.

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
* [Any Zig Version](any-version.md) for pinning a version per project.
* [Building from Source](building-from-source.md) for targets without a
  release, for the `zig2` bootstrap compiler, and for LLVM 23 development.
