![Zig++: Zig with the features upstream said no to.](doc/book/src/zigpp-header.webp)

# Zig++

*Pronounced "zig peepee".*

[zigpp.lol](https://zigpp.lol/) is Zig++'s documentation: installing it, what it
adds to Zig, GPU programming with `std.gpu`, and how versions and releases work.

Zig++ is to Zig what TypeScript is to JavaScript: a superset that adds the
features people kept asking for. Every valid Zig program is a valid Zig++
program, unless it names something `priv` (write `@"priv"` instead).
TypeScript made the same promise, and everyone believed them too.

Like Zig, Zig++ is a general-purpose programming language and toolchain for
maintaining **robust**, **optimal**, and **reusable** software. Unlike Zig, it
has private fields, it will never drop LLVM, it welcomes AI, and its BDFL is
Matthew Neel. Zig++ is a fork of [Zig](https://ziglang.org/), and nearly all of
the compiler was written by upstream Zig contributors.

## What Zig++ Adds

- **Private fields.** A struct or union field marked `priv` can only be named
  from the file that declares its type. See "Private Fields" in the language
  reference (run `zig build langref`, then open `zig-out/doc/langref.html`) and
  [doc/langref/test_private_fields.zig](doc/langref/test_private_fields.zig).
  Upstream closed the [proposal](https://github.com/ziglang/zig/issues/9909) as
  not planned.
- **LLVM forever**, and a blessed path to GPUs: `std.gpu` runs Zig++ and its
  standard library on NVIDIA, AMD, and Apple GPUs. See [LLVM Is Forever](#llvm-is-forever).
- **AI in the toolchain.** See [AI Policy](#ai-policy).
- **Any Zig version, automatically.** A project's `build.zig.zon` can pin the
  exact compiler version it is built with, and `zig` runs that version instead
  of itself, downloading it on first use. See
  [Any Zig Version](#any-zig-version).
- **No versions.** Every push to master is a release, every release is kept
  forever, and you pin the build you use. See [Live at Head](#live-at-head).
- **A BDFL and one rule: talk about code.** See [Governance](#governance).

**Does Zig++ compile to Zig, the way TypeScript compiles to JavaScript?** No.
It compiles to machine code, C, WebAssembly, PTX, AMD GPU code objects, and
Metal libraries.

**Is Zig++ stable?** Zig++ follows semantic versioning exactly as closely as
TypeScript does.

**What version is Zig++?** None. Every push to master is a release, and every
release is kept forever: you pin the build you use. A build is named for the
upstream version it tracks, its commit height, and its commit, as in
`0.17.0-dev.2469+zigpp.04926fc36`. See [Live at Head](#live-at-head).

**When is 1.0?** 1.0 is a milestone, not a version. The milestone closes when
Zig++ gets its borrow checker, and the build that closes it ships like every
other build. There is no Zig++ 1.0.

**Can upstream Zig build Zig++?** No. Zig++ changed `std.lang.Type`, and an
upstream Zig binary cannot compile against it. Use the CMake build,
`bootstrap.c`, or an existing Zig++ binary; see
[Building from Source](#building-from-source).

## LLVM Is Forever

Upstream Zig plans to drop its dependency on the LLVM libraries. Zig++ will
never phase out LLVM. In `package.json` terms, LLVM stays in `dependencies`.

LLVM, and MLIR above it, are how Zig++ goes the final stretch on GPUs. The
blessed path lowers Zig++ directly to PTX, to AMD GPU code objects, and to Metal
libraries, with first-class GPU intrinsics, and its first leg works today:
`std.gpu`, a port of [ugpu](https://github.com/mattneel/ugpu) into the standard
library.

- Kernels are plain Zig functions. `std.gpu` has CUDA's indexing (`threadIdx`,
  `blockIdx`, `blockDim`, `gridDim`, `globalId`), `syncThreads` (the new
  `@workGroupBarrier` builtin), warp shuffles, votes and reductions, atomics,
  fast math approximations, and `print`.
- The standard library runs on the GPU: `std.fmt`, `std.json`, `std.mem`,
  `std.base64`, hash maps, and array lists, with allocators for shared memory
  and for the CUDA device heap in `std.gpu.allocators`. A panic in a kernel
  reports its message to the host.
- Every NVPTX and AMDGPU module carries the compiler-rt routines that it calls,
  so `@sin`, `@exp`, `@log`, `f128`, and float parsing work in kernels, with the
  same results as on the host, bit for bit. Upstream Zig crashes LLVM on `@sin`
  for NVPTX.
- `std.gpu.cuda` loads the CUDA driver at run time, so programs build without
  the CUDA toolkit, and launches kernels from the host. `std.gpu.hip` does the
  same with the HIP runtime of AMD GPUs, on Linux and on Windows, where it needs
  no libc.
- `std.gpu.metal` loads the Metal framework at run time and runs kernels that
  Zig++ compiles for the `air64` target into a `.metallib`, in process, with no
  Xcode, Metal toolchain, or macOS SDK. Apple GPUs have no `f64`, no `print`,
  and only relaxed atomics, which are compile errors in their kernels.

```zig
// kernels.zig
const gpu = @import("std").gpu;

export fn wave(data: [*]f32, amplitude: f32, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i < n) data[i] = amplitude * @sin(data[i]);
}
```

```zig
// main.zig
const std = @import("std");
const cuda = std.gpu.cuda;

pub fn main() !void {
    var driver = try cuda.Driver.open();
    defer driver.close();
    const context = try (try driver.device(0)).retainPrimaryContext();
    defer context.release();
    const module = try context.loadModule(@embedFile("kernels.ptx"), .{});
    defer module.unload();

    var data: [1000]f32 = undefined;
    for (&data, 0..) |*x, i| x.* = @floatFromInt(i);
    const buffer = try context.alloc(f32, data.len);
    defer buffer.free();
    try buffer.copyFromHost(&data);
    const wave = try module.function("wave");
    try wave.launch(.linear(data.len, 256), .{ buffer, @as(f32, 2), @as(u32, data.len) });
    try context.synchronize();
    try buffer.copyToHost(&data);
}
```

```sh
zig build-obj -target nvptx64-cuda -mcpu=sm_75 -O ReleaseFast -fno-emit-bin -femit-asm=kernels.ptx kernels.zig
zig build-exe -lc main.zig
```

PTX for `sm_75` runs on any newer GPU, because the driver compiles it for the
GPU when it loads the module. In a build script, compile kernels with
`b.addObject` and embed `getEmittedAsm()`.

For AMD GPUs, the same kernels compile to a code object, and the host program
uses `std.gpu.hip` in place of `std.gpu.cuda` with the same calls:

```sh
zig build-lib -dynamic -target amdgcn-amdhsa -mcpu=gfx1036 -O ReleaseFast kernels.zig
```

A code object only runs on the architecture that `-mcpu` names, which
`hip.Device.archName` reports for a GPU.
[test/standalone/gpu](test/standalone/gpu) builds the kernels both ways and
runs every ugpu example on NVIDIA and AMD GPUs.

For Apple GPUs, the kernels compile to a Metal library, and the host program
uses `std.gpu.metal`:

```sh
zig build-obj -target air64-macos -O ReleaseFast -femit-bin=kernels.metallib kernels.zig
```

The suite runs a vector add, a reduction, and a kernel with scalar arguments on
the GPU of a Mac, which CI's M4 provides.

Still to come: MLIR lowering for tensor cores and kernel fusion.

## Live at Head

Live at Head. Zig++ has no versions. Every push to master is a release, and
every release is kept forever. You pin the build you use, the way you pin every
dependency, and you upgrade when you are ready. Zig++ does not number its
releases; upstream does it for us.

1. **Every push to master is a release.**
2. **Builds are forever.** Every published build stays downloadable at a stable
   URL with its checksums, never deleted and never replaced.
3. **Master is append-only.** No force pushes, no rewritten history, no
   retagging. A pin is a commit, and commits do not move.
4. **Version identity comes from upstream tags.** Zig++ does not create bare
   semantic-version tags such as `1.0.0`. `zigpp-*` tags identify individual
   Zig++ builds and releases. They are immutable artifact locators and are
   excluded from version derivation. `zigpp-*` tags never move and never
   disappear.
5. **The root project's pin wins.** Dependencies state minimums, and a pin
   older than a dependency's minimum is an error that names the dependency and
   the build to move to. *The minimum check is tooling: planned.*
6. **Breakage is announced at the upgrade, not by a number.** Every commit that
   breaks existing code carries a `Breaking:` trailer saying what breaks and
   what to write instead, and a merge of upstream Zig carries one for each of
   upstream's breaking changes in the merged range. *Collecting them for an
   upgrade is tooling: planned.*
7. **There are no LTS branches.** Security fixes land on head, and advisories
   name the affected range of builds. *Advisories are tooling: planned.*
8. **Docs travel with the pin.** Every build carries its own std docs and
   language reference, and `zig std` serves them (the language reference from
   0.17.0-dev.2476 on).
9. **Upstream Zig and every vendored dependency are tracked continuously and
   pinned the same way.**
10. **1.0 is a milestone, not a version.** The build that closes it ships like
    every other build. There is no Zig++ 1.0.

"Live at Head" is Abseil's phrase, from Titus Winters' CppCon 2017 talk
["C++ as a 'Live at Head' Language"](https://www.youtube.com/watch?v=tISy7EJQPzI).
[Live at Head](https://zigpp.lol/live-at-head.html) in the book says how to pin
a build and how to upgrade, and where each piece of tooling stands.

## AI Policy

Upstream Zig bans LLMs from issues, patches, and bug tracker comments. Zig++
welcomes them. The first Zig++ language feature, private fields, was
implemented, tested, and documented by Claude, and the commit is signed that
way.

Zig++ has AI on both sides of compilation: as a runtime facility in the
programs it builds, and as a build-system facility in the build itself.
Neither is in `std` yet: both are packages today, and both are headed into the
standard library under the names `std.ai` and `std.Build.ai`.

`std.ai` will be [ai.zig](https://github.com/mattneel/ai.zig), the AI toolkit
for Zig: an implementation of the Vercel AI SDK v7 core built on `std.Io`,
with text generation and streaming, multi-step tool loops, structured output,
embeddings, reranking, reusable agents, and an MCP client.

`std.Build.ai` will be [Zigger](https://github.com/mattneel/zigger), AI code
generation as a Zig build step. Today it is a package that adds a
`zig build gen` step: it reads `SPEC.md`, runs the Claude CLI to implement it,
runs `zig build test`, feeds any failures back, and repeats until the tests
pass (up to 10 times by default). With `-Dtdd=true` it writes failing tests
first. The plan is to move this pipeline into `std.Build.ai`.

The goal: by September 2027, all new Zig++ code is written through the Zigger
pipeline.

## Governance

Upstream Zig is BDFN (Benevolent Dictator For Now). Zig++ is BDFL: Matthew Neel
is the Benevolent Dictator For Life and has final say on the design and
implementation of everything.

Zig++ has no Code of Conduct. It has one rule: talk about code. Issues, pull
requests, reviews, and comments are for the compiler, the language, the
standard library, and the tools. Everything else, politics included, is off
topic and will be closed.

Language proposals are welcome. Zig++ is made of them.

## Documentation

[zigpp.lol](https://zigpp.lol/) has this book along with the language reference
and the standard library documentation of the newest release.

If you are looking at this README file in a source tree, please refer to the
**Release Notes**, **Language Reference**, or **Standard Library
Documentation** corresponding to the version of Zig that you are using by
following the appropriate link on the
[download page](https://ziglang.org/download).

Otherwise, you're looking at a release of Zig, so you can find the language
reference at `doc/langref.html`, and the standard library documentation by
running `zig std`, which will open a browser tab.

## Installation

Installing Zig++ is one line. On Linux and macOS:

```sh
curl -fsSL https://zigpp.lol/ppup | sh
```

On Windows, in PowerShell:

```powershell
irm https://zigpp.lol/ppup.ps1 | iex
```

`ppup` unpacks the newest release, makes it the default `zig`, installs itself,
and then manages the toolchains you have: `ppup update`,
`ppup install <version>`, `ppup default <version>`, `ppup list`,
`ppup uninstall <version>`, and `ppup self uninstall` to undo everything. See
[Installing](https://zigpp.lol/installing.html) for the whole command list, and
[zigpp.lol/downloads.html](https://zigpp.lol/downloads.html) for the archives
and their SHA-256 checksums. To build it yourself, see
[Building from Source](#building-from-source).

Zig++ publishes a release on every push to master. The links below are for
upstream Zig.

 * [download a pre-built binary](https://ziglang.org/download/)
 * [install from a package manager](https://ziglang.org/learn/getting-started/#managers)
 * [bootstrap zig for any target](https://codeberg.org/ziglang/zig-bootstrap)

A Zig installation is composed of two things:

1. The Zig executable
2. The lib/ directory

At runtime, the executable searches up the file system for the lib/ directory,
relative to itself:

* lib/
* lib/zig/
* ../lib/
* ../lib/zig/
* (and so on)

In other words, you can **unpack a release of Zig anywhere**, and then begin
using it immediately. There is no need to install it globally, although this
mechanism supports that use case too (i.e. `/usr/bin/zig` and `/usr/lib/zig/`).

## Any Zig Version

A project can pin the exact compiler version it is built with:

```zig
// build.zig.zon
.minimum_zig_version = "0.15.1",
```

When it does, any `zig` command in that project runs that version instead of
the `zig` that was invoked. `zig init` writes the version that created the
project, so a new project keeps building with the compiler that made it.

The version names the exact compiler, so it can be a Zig++ release
(`0.17.0-dev.2361+zigpp.5b96e6d21`), an upstream Zig release (`0.15.1`), or an
upstream dev build (`0.16.0-dev.1234+abcdef012`). Zig++ downloads it into
`<global cache>/any/<version>/` the first time it is needed, verifies it,
unpacks it there, and runs it with the same arguments. An install is never
visible half-finished, and concurrent `zig` invocations that need the same
version end up with one install of it.

An upstream Zig release is verified against the SHA-256 of the upstream
download index, and every upstream archive against the minisign signature that
ziglang.org publishes next to it, which is signed with the Zig Software
Foundation's key. Zig++ releases are verified against the SHA-256 of their
GitHub release index.

ziglang.org keeps only the recent dev builds, so an archive it no longer has
-- an older dev build that a project pins, say -- is downloaded from the
[community mirrors](https://ziglang.org/download/community-mirrors.txt), whose
copy has to pass the same signature check.

Run a version explicitly, list what is installed, or turn the dispatch off:

```sh
zig any 0.15.1 version   # run that exact version
zig any list             # installed versions, one per line
ZIG_ANY=off zig version  # always the zig that was invoked
```

`zig any <version>` runs the version that was asked for, whatever the enclosing
project pins: the compiler it starts is told with `ZIG_ANY=off` not to dispatch
to the project's pin.

Installed versions live in `<global cache>/any/`, the `global_cache_dir` that
`zig env` reports. Zig++ releases come from
[github.com/mattneel/zigpp](https://github.com/mattneel/zigpp); upstream
versions come from [ziglang.org](https://ziglang.org/download), and from the
community mirrors when ziglang.org no longer has them.

## Building from Source

Ensure you have the required dependencies:

 * CMake >= 3.15
 * System C/C++ Toolchain
 * LLVM, Clang, LLD development libraries, version 23.x, compiled with the
   same system C/C++ toolchain.
   - If the system package manager lacks these libraries, or has them misconfigured,
     see below for how to build them from source.

Then it is the standard CMake build process:

```sh
mkdir build
cd build
cmake ..
make install
```

Use `CMAKE_PREFIX_PATH` if needed to help CMake find LLVM.

This produces `stage3/bin/zig` which is the Zig compiler built by itself.

## Building from Source without LLVM

In this case, the only system dependency is a C compiler.

```sh
cc -o bootstrap bootstrap.c
./bootstrap
```

This produces a `zig2` executable in the current working directory. This is a
"stage2" build of the compiler,
[without LLVM extensions](https://github.com/ziglang/zig/issues/16270), and is
therefore lacking these features:
- Release mode optimizations
- [Some ELF linking features](https://github.com/ziglang/zig/issues/17749)
- [Some COFF/PE linking features](https://github.com/ziglang/zig/issues/17751)
- [Some WebAssembly linking features](https://github.com/ziglang/zig/issues/17750)
- [Ability to create static archives from object files](https://github.com/ziglang/zig/issues/9828)
- [Ability to compile assembly files](https://github.com/ziglang/zig/issues/21169)
- Ability to compile C, C++, Objective-C, and Objective-C++ files

Even when built this way, Zig provides an LLVM backend that produces bitcode
files, which may be optimized and compiled into object files via separately
installed Clang. Similarly, Zig provides a C backend that produces C source
code, which may be optimized and compiled into object files via a separately
installed C compiler toolchain.

From here you can tinker with `zig2` or you can proceed to installation using
the build system as usual:

```sh
./zig2 build
```

Upstream Zig recommends not proceeding with this step until this issue is
resolved:

[completely eliminate dependency on LLVM library API calls](https://github.com/ziglang/zig/issues/25492)

**Zig++:** that issue will not be resolved here. LLVM is a permanent,
first-class dependency of Zig++; the LLVM-less `zig2` above exists only for
bootstrapping. See [LLVM Is Forever](#llvm-is-forever).

## Building from Source Using Prebuilt Zig

**Zig++:** the prebuilt Zig must be a Zig++ binary, such as `stage3/bin/zig`
from the CMake build. Upstream Zig binaries, including the one zig-bootstrap
produces, cannot build Zig++, because Zig++ changed `std.lang.Type`.
zig-bootstrap is still a good source of the LLVM, Clang, and LLD libraries.

Dependencies:

 * A recent prior build of Zig. The exact version required depends on how
   recently breaking changes occurred. If the language or std lib changed too
   much since this version, then this method of building from source will fail.
 * LLVM, Clang, and LLD libraries built using Zig.

The easiest way to obtain both of these artifacts is to use
[zig-bootstrap](https://codeberg.org/ziglang/zig-bootstrap), which creates the
directory `out/zig-$target-$cpu` and `out/$target-$cpu`, to be used as
`$ZIG_PREFIX` and `$LLVM_PREFIX`, respectively, in the following command:

```sh
"$ZIG_PREFIX/zig" build \
  -p stage3 \
  --search-prefix "$LLVM_PREFIX" \
  --zig-lib-dir "lib" \
  -Dstatic-llvm
```

Where `$LLVM_PREFIX` is the path that contains, for example,
`include/llvm/Pass.h` and `lib/libLLVMCore.a`.

This produces `stage3/bin/zig`. See `zig build -h` to learn about the options
that can be passed such as `-Drelease`.

## Building from Source on Windows

### Option 1: Use the Windows Zig Compiler Dev Kit

This one has the benefit that LLVM, LLD, and Clang are built in Release mode,
while your Zig build has the option to be a Debug build. It also works
completely independently from MSVC so you don't need it to be installed.

Determine the URL by
[looking at the CI script](https://codeberg.org/ziglang/zig/src/branch/master/ci/x86_64-windows-debug.ps1#L1-L4).
It will look something like this (replace `$VERSION` with the one you see by
following the above link):

```
https://ziglang.org/deps/zig+llvm+lld+clang-x86_64-windows-gnu-$VERSION.zip
```

This zip file contains:

 * An older Zig installation.
 * LLVM, LLD, and Clang libraries (.lib and .h files), version 16.0.1, built in Release mode.
 * zlib (.lib and .h files), v1.2.13, built in Release mode
 * zstd (.lib and .h files), v1.5.2, built in Release mode

#### Option 1a: CMake + [Ninja](https://ninja-build.org/)

Unzip the dev kit and then in cmd.exe in your Zig source checkout:

```bat
mkdir build
cd build
set DEVKIT=$DEVKIT
```

Replace `$DEVKIT` with the path to the folder that you unzipped after
downloading it from the link above. Make sure to use forward slashes (`/`) for
all path separators (otherwise CMake will try to interpret backslashes as
escapes and fail).

Then run:

```bat
cmake .. -GNinja -DCMAKE_PREFIX_PATH="%DEVKIT%" -DCMAKE_C_COMPILER="%DEVKIT%/bin/zig.exe;cc" -DCMAKE_CXX_COMPILER="%DEVKIT%/bin/zig.exe;c++" -DCMAKE_AR="%DEVKIT%/bin/zig.exe" -DZIG_AR_WORKAROUND=ON -DZIG_STATIC=ON -DZIG_USE_LLVM_CONFIG=OFF
```

 * Append `-DCMAKE_BUILD_TYPE=Release` for a Release build.
 * Append `-DZIG_NO_LIB=ON` to avoid having multiple copies of the lib/ folder.

Finally, run:

```bat
ninja install
```

You now have the `zig.exe` binary at `stage3\bin\zig.exe`.

#### Option 1b: zig build

Unzip the dev kit and then in cmd.exe in your Zig source checkout:

```bat
$DEVKIT\bin\zig.exe build -p stage3 --search-prefix $DEVKIT --zig-lib-dir lib -Dstatic-llvm -Duse-zig-libcxx -Dtarget=x86_64-windows-gnu
```

Replace `$DEVKIT` with the path to the folder that you unzipped after
downloading it from the link above.

Append `-Doptimize=ReleaseSafe` for a Release build.

**If you get an error building at this step**, it is most likely that the Zig
installation inside the dev kit is too old, and the dev kit needs to be
updated. In this case one more step is required:

 1. [Download the latest master branch zip file](https://ziglang.org/download/#release-master).
 2. Unzip, and try the above command again, replacing the path to zig.exe with
    the path to the zig.exe you just extracted, and also replace the lib\zig
    folder with the new contents.

You now have the `zig.exe` binary at `stage3\bin\zig.exe`.

### Option 2: Using CMake and Microsoft Visual Studio

This one has the benefit that changes to the language or build system won't
break your dev kit. This option can be used to upgrade a dev kit.

First, build LLVM, LLD, and Clang from source using CMake and Microsoft Visual
Studio (see below for detailed instructions).

Install [Build Tools for Visual Studio
2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).
Be sure to select "Desktop development with C++" when prompted.
 * You must additionally check the optional component labeled **C++ ATL for
   v142 build tools**.

Install [CMake](http://cmake.org).

Use [git](https://git-scm.com/) to clone the zig repository to a path with no spaces, e.g. `C:\Users\Andy\zig`.

Using the start menu, run **x64 Native Tools Command Prompt for VS 2019** and execute these commands, replacing `C:\Users\Andy` with the correct value.

```bat
mkdir C:\Users\Andy\zig\build-release
cd C:\Users\Andy\zig\build-release
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=C:\Users\Andy\llvm+clang+lld-20.0.0-x86_64-windows-msvc-release-mt -DCMAKE_BUILD_TYPE=Release
msbuild -p:Configuration=Release INSTALL.vcxproj
```

You now have the `zig.exe` binary at `bin\zig.exe` and you can run the tests:

```bat
bin\zig.exe build test
```

This can take a long time.

Note: In case you get the error "llvm-config not found" (or similar), make sure
that you have **no** trailing slash (`/` or `\`) at the end of the
`-DCMAKE_PREFIX_PATH` value.

## Building LLVM, LLD, and Clang from Source

### Windows

Install [CMake](https://cmake.org/), version 3.20.0 or newer.

[Download LLVM, Clang, and LLD sources](https://releases.llvm.org/download.html#23.0.0)
The downloads from llvm lead to the github release pages, where the source's
will be listed as : `llvm-23.X.X.src.tar.xz`, `clang-23.X.X.src.tar.xz`,
`lld-23.X.X.src.tar.xz`. Unzip each to their own directory. Ensure no
directories have spaces in them. For example:

 * `C:\Users\Andy\llvm-23.0.0.src`
 * `C:\Users\Andy\clang-23.0.0.src`
 * `C:\Users\Andy\lld-23.0.0.src`

Install [Build Tools for Visual Studio
2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).
Be sure to select "C++ build tools" when prompted.
 * You **must** additionally check the optional component labeled **C++ ATL for
   v142 build tools**. As this won't be supplied by a default installation of
   Visual Studio.
 * Full list of supported MSVC versions:
   - 2017 (version 15.8) (unverified)
   - 2019 (version 16.7)

Install [Python 3.9.4](https://www.python.org). Tick the box to add python to
your PATH environment variable.

#### LLVM

Using the start menu, run **x64 Native Tools Command Prompt for VS 2019** and execute these commands, replacing `C:\Users\Andy` with the correct value. Here is listed a brief explanation of each of the CMake parameters we pass when configuring the build

- `-Thost=x64` : Sets the windows toolset to use 64 bit mode.
- `-A x64` : Make the build target 64 bit .
- `-G "Visual Studio 16 2019"` : Specifies to generate a 2019 Visual Studio project, the best supported version.
- `-DCMAKE_INSTALL_PREFIX=""` : Path that llvm components will being installed into by the install project.
- `-DCMAKE_PREFIX_PATH=""` : Path that CMake will look into first when trying to locate dependencies, should be the same place as the install prefix. This will ensure that clang and lld will use your newly built llvm libraries.
- `-DLLVM_ENABLE_ZLIB=OFF` : Don't build llvm with ZLib support as it's not required and will disrupt the target dependencies for components linking against llvm. This only has to be passed when building llvm, as this option will be saved into the config headers.
- `-DCMAKE_BUILD_TYPE=Release` : Build llvm and components in release mode.
- `-DCMAKE_BUILD_TYPE=Debug` : Build llvm and components in debug mode.
- `-DLLVM_USE_CRT_RELEASE=MT` : Which C runtime should llvm use during release builds.
- `-DLLVM_USE_CRT_DEBUG=MTd` : Make llvm use the debug version of the runtime in debug builds.

##### Release Mode

```bat
mkdir C:\Users\Andy\llvm-23.0.0.src\build-release
cd C:\Users\Andy\llvm-23.0.0.src\build-release
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\Andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-release-mt -DCMAKE_PREFIX_PATH=C:\Users\Andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-release-mt -
DLLVM_ENABLE_ZLIB=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_USE_CRT_RELEASE=MT
msbuild /m -p:Configuration=Release INSTALL.vcxproj
```

##### Debug Mode

```bat
mkdir C:\Users\Andy\llvm-23.0.0.src\build-debug
cd C:\Users\Andy\llvm-23.0.0.src\build-debug
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-debug -
DLLVM_ENABLE_ZLIB=OFF -DCMAKE_PREFIX_PATH=C:\Users\andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-debug -DCMAKE_BUILD_TYPE=Debug -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="AVR" -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_USE_CRT_DEBUG=MTd
msbuild /m INSTALL.vcxproj
```

#### LLD

Using the start menu, run **x64 Native Tools Command Prompt for VS 2019** and execute these commands, replacing `C:\Users\Andy` with the correct value.

##### Release Mode

```bat
mkdir C:\Users\Andy\lld-23.0.0.src\build-release
cd C:\Users\Andy\lld-23.0.0.src\build-release
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\Andy\llvm+clang+lld-14.0.6-x86_64-windows-msvc-release-mt -DCMAKE_PREFIX_PATH=C:\Users\Andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-release-mt -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_CRT_RELEASE=MT
msbuild /m -p:Configuration=Release INSTALL.vcxproj
```

##### Debug Mode

```bat
mkdir C:\Users\Andy\lld-23.0.0.src\build-debug
cd C:\Users\Andy\lld-23.0.0.src\build-debug
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-debug -DCMAKE_PREFIX_PATH=C:\Users\andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-debug -DCMAKE_BUILD_TYPE=Debug -DLLVM_USE_CRT_DEBUG=MTd
msbuild /m INSTALL.vcxproj
```

#### Clang

Using the start menu, run **x64 Native Tools Command Prompt for VS 2019** and execute these commands, replacing `C:\Users\Andy` with the correct value.

##### Release Mode

```bat
mkdir C:\Users\Andy\clang-23.0.0.src\build-release
cd C:\Users\Andy\clang-23.0.0.src\build-release
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\Andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-release-mt -DCMAKE_PREFIX_PATH=C:\Users\Andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-release-mt -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_CRT_RELEASE=MT
msbuild /m -p:Configuration=Release INSTALL.vcxproj
```

##### Debug Mode

```bat
mkdir C:\Users\Andy\clang-23.0.0.src\build-debug
cd C:\Users\Andy\clang-23.0.0.src\build-debug
"c:\Program Files\CMake\bin\cmake.exe" .. -Thost=x64 -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=C:\Users\andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-debug -DCMAKE_PREFIX_PATH=C:\Users\andy\llvm+clang+lld-23.0.0-x86_64-windows-msvc-debug -DCMAKE_BUILD_TYPE=Debug -DLLVM_USE_CRT_DEBUG=MTd
msbuild /m INSTALL.vcxproj
```

### POSIX Systems

This guide will get you both a Debug build of LLVM, and/or a Release build of LLVM.
It intentionally does not require privileged access, using a prefix inside your home
directory instead of a global installation.

#### Release

This is the generally recommended approach.

```sh
cd ~/Downloads
git clone --depth 1 --branch release/23.x https://github.com/llvm/llvm-project llvm-project-23
cd llvm-project-23
git checkout release/23.x

mkdir build-release
cd build-release
cmake ../llvm \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm23-assert \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="lld;clang" \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -G Ninja
ninja install
```

#### Debug

This is occasionally needed when debugging Zig's LLVM backend. Here we build
the three projects separately so that LLVM can be in Debug mode while the
others are in Release mode.

```sh
cd ~/Downloads
git clone --depth 1 --branch release/23.x https://github.com/llvm/llvm-project llvm-project-23
cd llvm-project-23
git checkout release/23.x

# LLVM
mkdir llvm/build-debug
cd llvm/build-debug
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm23-debug \
  -DCMAKE_PREFIX_PATH=$HOME/local/llvm23-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -G Ninja
ninja install
cd ../..

# LLD
mkdir lld/build-debug
cd lld/build-debug
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm23-debug \
  -DCMAKE_PREFIX_PATH=$HOME/local/llvm23-debug \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_CXX_STANDARD=17 \
  -G Ninja
ninja install
cd ../..

# Clang
mkdir clang/build-debug
cd clang/build-debug
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm23-debug \
  -DCMAKE_PREFIX_PATH=$HOME/local/llvm23-debug \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_INCLUDE_TESTS=OFF \
  -G Ninja
ninja install
cd ../..
```

Then add to your Zig CMake line that you got from the README.md:
`-DCMAKE_PREFIX_PATH=$HOME/local/llvm23-debug` or
`-DCMAKE_PREFIX_PATH=$HOME/local/llvm23-assert` depending on whether you want
Debug or Release LLVM.


## Contributing

Zig++ is Free and Open Source Software. Bug reports, patches, and language
proposals are welcome from everyone, and so is AI. Read [AI Policy](#ai-policy)
and [Governance](#governance) first: Matthew Neel is BDFL, there is no Code of
Conduct, and the only topic is code.

Most of this compiler was written by upstream Zig contributors. If you want to
support them, [donate to the Zig Software Foundation](https://ziglang.org/zsf/).

### Make Software With Zig

One of the best ways you can contribute to Zig is to start using it for an
open-source personal project.

This leads to discovering bugs and helps flesh out use cases, which lead to
further design iterations of Zig. Importantly, each issue found this way comes
with real world motivations, making it straightforward to explain the reasoning
behind proposals and feature requests.

Ideally, such a project will help you to learn new skills and add something
to your personal portfolio at the same time.

### Talk About Zig

Another way to contribute is to write about Zig, speak about Zig at a
conference, or do either of those things for your project which uses Zig.

Programming languages live and die based on the pulse of their ecosystems. The
more people involved, the more we can build great things upon each other's
abstractions.

### AI Welcome

Use whatever tools you like for issues, patches, and code review. Better yet,
use the Zigger pipeline; see [AI Policy](#ai-policy).

### Find a Contributor Friendly Issue

The issue label
[Contributor Friendly](https://codeberg.org/ziglang/zig/issues?labels=741726&state=open)
exists to help you find issues that are **limited in scope and/or
knowledge of Zig internals.**

Please note that issues labeled
[Proposal: Proposed](https://codeberg.org/ziglang/zig/issues?labels=746937&state=open)
are still under consideration, and efforts to implement such a proposal have
a high risk of being wasted. If you are interested in a proposal which is
still under consideration, please express your interest in the issue tracker,
providing extra insights and considerations that others have not yet expressed.
The most highly regarded argument in such a discussion is a real world use case.

Language proposals are not accepted. Please do not open an issue proposing to
change the Zig language or syntax.

### Breaking Changes

Zig++ has no version numbers to warn people with, so the commit does. A commit
that breaks existing code carries a `Breaking:` trailer for each thing it
breaks, saying what breaks and what to write instead:

```text
Breaking: std.zig.Server.serveErrorBundle takes the message tag first. Write serveErrorBundle(.error_bundle, bundle).
```

A merge of upstream Zig carries a `Breaking:` trailer for each of upstream's
breaking changes in the merged range. See [Live at Head](#live-at-head).

### Editing Source Code

For a smooth workflow, when building from source, it is recommended to use
CMake with the following settings:

 * `-DCMAKE_BUILD_TYPE=Release` - to recompile zig faster.
 * `-GNinja` - Ninja is faster and simpler to use than Make.
 * `-DZIG_NO_LIB=ON` - Prevents the build system from copying the lib/
   directory to the installation prefix, causing zig use lib/ directly from the
   source tree instead. Effectively, this makes it so that changes to lib/ do
   not require re-running the install command to become active.

After configuration, there are two scenarios:

 1. Pulling upstream changes and rebuilding.
    - In this case use `git pull` and then `ninja install`. Expected wait:
      about 10 minutes.
 2. Building from source after making local changes.
    - In this case use `stage3/bin/zig build -p stage4 -Denable-llvm -Dno-lib`.
      Expected wait: about 20 seconds.

This leaves you with two builds of Zig:

 * `stage3/bin/zig` - an optimized master branch build. Useful for
   miscellaneous activities such as `zig fmt`, as well as for building the
   compiler itself after changing the source code.
 * `stage4/bin/zig` - a debug build that includes your local changes; useful
   for testing and eliminating bugs before submitting a patch.

To reduce time spent waiting for the compiler to build, try these techniques:

 * Omit `-Denable-llvm` if you don't need the LLVM backend.
 * Use `-Ddev=foo` to build with a reduced feature set for development of
   specific features. See `zig build -h` for a list of options.
 * Use `--watch -fincremental` to enable incremental compilation. This offers
   **near instant rebuilds**.

### Testing

```sh
stage4/bin/zig build test
```

This command runs the whole test suite, which does a lot of extra testing that
you likely won't always need, and can take upwards of 1 hour. This is what the
CI server runs when you make a pull request.

To save time, you can add the `--help` option to the `zig build` command and
see what options are available. One of the most helpful ones is
`-Dskip-release`. Adding this option to the command above, along with
`-Dskip-non-native`, will take the time down from around 2 hours to about 30
minutes, and this is a good enough amount of testing before making a pull
request.

Another example is choosing a different set of things to test. For example,
`test-std` instead of `test` will only run the standard library tests, and
not the other ones. Combining this suggestion with the previous one, you could
do this:

```sh
stage4/bin/zig build test-std -Dskip-release
```

This will run only the standard library tests in debug mode for all targets.
It will cross-compile the tests for non-native targets but not run them.

When making changes to the compiler source code, the most helpful test step to
run is `test-behavior`. When editing documentation it is `docs`. You can find
this information and more in the `zig build --help` menu.

#### Directly Testing the Standard Library with `zig test`

This command will run the standard library tests with only the native target
configuration and is estimated to complete in 3 minutes:

```sh
zig build test-std -Dno-matrix
```

However, one may also use `zig test` directly. From inside the `ziglang/zig` repo root:

```sh
zig test lib/std/std.zig --zig-lib-dir lib
```

You can add `--test-filter "some test name"` to run a specific test or a subset of tests.
(Running exactly 1 test is not reliably possible, because the test filter does not
exclude anonymous test blocks, but that shouldn't interfere with whatever
you're trying to test in practice.)

Note that `--test-filter` filters on fully qualified names, so e.g. it's possible to run only the `std.json` tests with:

```sh
zig test lib/std/std.zig --zig-lib-dir lib --test-filter "json."
```

If you used `-Dno-lib` and you are in a `build/` subdirectory, you can omit the
`--zig-lib-dir` argument:

```sh
stage3/bin/zig test ../lib/std/std.zig
```

#### Testing Non-Native Architectures with QEMU

The Linux CI server additionally has qemu installed and sets `-fqemu`.
This provides test coverage for, e.g. aarch64 even on x86_64 machines. It's
recommended for Linux users to install qemu and enable this testing option
when editing the standard library or anything related to a non-native
architecture.

QEMU packages provided by some system package managers (such as Debian) may be
a few releases old, or may be missing newer targets such as aarch64 and RISC-V.
[ziglang/qemu-static](https://codeberg.org/ziglang/qemu-static) offers static
binaries of the latest QEMU version.

##### Testing Non-Native libc Targets

Testing foreign architectures with dynamically linked libc is one step trickier.
This requires enabling `--libc-runtimes /path/to/libcs`. This path is obtained
by building glibc and musl for multiple architectures. This process for me took
an entire day to complete and takes up 65 GiB on my hard drive.

[Instructions for producing this path.](https://codeberg.org/ziglang/infra/src/branch/master/building-libcs.md)

It is understood that most contributors will not have these tests enabled. The
CI machines provide coverage for these.

#### Testing Windows from a Linux Machine with Wine

When developing on Linux, another option is available to you: `-fwine`.
This will enable running behavior tests and std lib tests with Wine. It's
recommended for Linux users to install Wine and enable this testing option
when editing the standard library or anything Windows-related.

#### Testing WebAssembly using wasmtime

If you have [wasmtime](https://wasmtime.dev/) installed, take advantage of the
`-fwasmtime` flag which will enable running WASI behavior tests and std
lib tests. It's recommended for all users to install wasmtime and enable this
testing option when editing the standard library and especially anything
WebAssembly-related.

### Improving Translate-C

`translate-c` is a feature provided by Zig that converts C source code into Zig
source code. It powers the `zig translate-c` command, allowing Zig code to not
only take advantage of function prototypes defined in C header files, but also
`static inline` functions written in C, and even some macros.

This feature used to work by using libclang API to parse and semantically
analyze C/C++ files, and then based on the provided AST and type information,
generating Zig AST, and finally using the mechanisms of `zig fmt` to render the
Zig AST to a file.

However, it is now based on [arocc](https://github.com/Vexu/arocc/), a
third-party C compiler written in Zig. Test coverage, bug reports, and official
implementation live in this repository: [ziglang/translate-c](https://codeberg.org/ziglang/translate-c/)

This package is currently vendored into the Zig source tree. The TranslateC
build step takes advantage of this to provide the ability to setup C
translation in one's build.zig script.

Please see the readme of the translate-c project for how to contribute. Once an
issue is resolved (and test coverage added) there, the changes can be
immediately backported to the zig compiler.

However, in the future, this build step will be removed in favor of explicit
dependency on the translate-c package via build system / package manager. At
that point, Zig will stop vendoring arocc.

### Autodoc

Autodoc is an interactive, searchable, single-page web application for browsing
Zig codebases.

An autodoc deployment looks like this:

```
index.html
main.js
main.wasm
sources.tar
```

* `main.js` and `index.html` are static files which live in a Zig installation
  at `lib/docs/`.
* `main.wasm` is compiled from the Zig files inside `lib/docs/wasm/`.
* `sources.tar` is all the zig source files of the project.

These artifacts are produced by the compiler when `-femit-docs` is passed.

#### Making Changes

The command `zig std` spawns an HTTP server that provides all the assets
mentioned above specifically for the standard library.

The server creates the requested files on the fly, including rebuilding
`main.wasm` if any of its source files changed, and constructing `sources.tar`,
meaning that any source changes to the documented files, or to the autodoc
system itself are immediately reflected when viewing docs.

This means you can test changes to Zig standard library documentation, as well
as autodocs functionality, by pressing refresh in the browser.

Prefixing the URL with `/debug` results in a debug build of `main.wasm`.

#### Debugging the Zig Code

While Firefox and Safari support are obviously required, I recommend Chromium
for development for one reason in particular:

[C/C++ DevTools Support (DWARF)](https://chromewebstore.google.com/detail/cc++-devtools-support-dwa/pdcpmagijalfljmkmjngeonclgbbannb)

This makes debugging Zig WebAssembly code a breeze.

#### The Sources Tarball

The system expects the top level of `sources.tar` to be the set of modules
documented. So for the Zig standard library you would do this:
`tar cf std.tar std/`. Don't compress it; the idea is to rely on HTTP
compression.

Any files that are not `.zig` source files will be ignored by `main.wasm`,
however, those files will take up wasted space in the tar file. For the
standard library, use the set of files that zig installs to when running `zig
build`, which is the same as the set of files that are provided on
ziglang.org/download.

If the system doesn't find a file named "foo/root.zig" or "foo/foo.zig", it
will use the first file in the tar as the module root.

You don't typically need to create `sources.tar` yourself, since it is lazily
provided by the `zig std` HTTP server as well as produced by `-femit-docs`.


## Testing Zig Code With LLDB

[@jacobly0](https://github.com/jacobly0) maintains a fork of LLDB with Zig support:

https://github.com/jacobly0/llvm-project/tree/lldb-zig

This fork only contains changes for debugging programs compiled by Zig's
self-hosted backends, i.e. `zig build-exe -fno-llvm ...`.

### Building

To build the LLDB fork, make sure you have
[prerequisites](https://lldb.llvm.org/resources/build.html#preliminaries)
installed, and then do something like:

```sh
$ cmake llvm -G Ninja -B build -DLLVM_ENABLE_PROJECTS="clang;lldb" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLDB_ENABLE_LIBEDIT=ON -DLLDB_ENABLE_PYTHON=ON
$ cmake --build build --target lldb --target lldb-server
```

(You may need to manually [configure
dependencies](https://lldb.llvm.org/resources/build.html#optional-dependencies)
if CMake can't find them.)

Once built, you can run `./build/bin/lldb` and so on.

### Pretty Printers

If you will be debugging the Zig compiler itself, or if you will be debugging
any project compiled with Zig's LLVM backend (not recommended with the LLDB
fork, prefer vanilla LLDB with a version that matches the version of LLVM that
Zig is using), you can get a better debugging experience by using
[`lldb/pretty_printers.py`](https://codeberg.org/ziglang/zig/src/branch/master/lib/lldb/pretty_printers.py)
which is included in Zig's installed lib dir.

Put this line in `~/.lldbinit`:

```
command script import /path/to/zig/lib/lldb/pretty_printers.py
```

If you will be debugging a Zig compiler built using Zig's self-hosted backends,
you will also want this line:

```
type category enable zig.compiler
```

If you will be using Zig's LLVM backend (again, not recommended with the LLDB
fork), you will also want these lines:

```
type category enable zig.lang
type category enable zig.std
```

If you will be debugging a Zig compiler built using Zig's LLVM backend without
using the LLDB fork, you will also want this line:

```
type category enable zig
```
