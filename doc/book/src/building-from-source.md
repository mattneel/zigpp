# Building from Source

Build Zig++ from source when you need a target that has no release, when you
are developing the compiler, or when you want to build it against your own LLVM
23 libraries. There are four routes, from the least to the most preparation:

1. **A Zig++ bootstrap devkit**, which brings LLVM, Clang, and LLD with it.
   This is what Zig++'s own CI and releases use.
2. **The CMake build**, against LLVM, Clang, and LLD 23.x you installed
   yourself.
3. **An existing Zig++ compiler**, which is the quickest route if you already
   have one.
4. **`bootstrap.c`**, which needs nothing but a C compiler, and produces a
   compiler that cannot use the LLVM backend.

## Requirements

Whichever route you take, the CMake build needs:

- CMake 3.15 or later;
- a system C/C++ toolchain;
- the LLVM, Clang, and LLD development libraries, version 23.x, built with the
  same system C/C++ toolchain.

Zig++ accepts LLVM 23.x: the CMake module that finds LLVM rejects anything
older than 23 and anything newer than 24, and it checks that the libraries it
found can target AMDGPU and NVPTX. `CMAKE_PREFIX_PATH` points the build at a
non-system LLVM:

```sh
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="$HOME/local/llvm23"
make install
```

This produces `stage3/bin/zig`, relative to the build directory: the Zig
compiler built by itself.

By default the build prefers the shared LLVM libraries. `-DZIG_STATIC=ON`
links them statically, which is what releases do; it is not compatible with
glibc. `-DZIG_SHARED_LLVM=ON` and `-DZIG_STATIC_LLVM=ON` choose explicitly, and
asking for both at once is an error.

## Building against an LLVM you built with Zig

If you have LLVM, Clang, and LLD libraries that were built *by* Zig, an
existing Zig++ compiler can build Zig++ against them:

```sh
"$ZIG_PREFIX/zig" build \
  -p stage3 \
  --search-prefix "$LLVM_PREFIX" \
  --zig-lib-dir "lib" \
  -Dstatic-llvm
```

`$LLVM_PREFIX` is the directory that holds `include/llvm/Pass.h` and
`lib/libLLVMCore.a`. The result is `stage3/bin/zig`.

The compiler that runs the build must be Zig++, not upstream Zig: Zig++
changed `std.lang.Type`, so an upstream Zig binary cannot compile against it.
Upstream's `zig-bootstrap` is still a good way to get the LLVM, Clang, and LLD
libraries, as long as you use a Zig++ compiler to build with them.

## Devkits

[zigpp-bootstrap](https://github.com/mattneel/zigpp-bootstrap) is
[zig-bootstrap](https://codeberg.org/ziglang/zig-bootstrap) for Zig++, and it
makes the devkits that Zig++'s CI and releases build with. A devkit is a
prebuilt tree of static LLVM, Clang, and LLD libraries, zlib and zstd, and a
Zig++ compiler, with the headers to build against.

It builds LLVM, Clang, and LLD 23.1.2 from the `llvmorg-23.1.2` release source,
checked against its SHA-256 and patched with the patches that zig-bootstrap
uses for LLVM 23, plus zlib 1.3.1 and zstd 1.5.2.

Host requirements:

- a C++ compiler that can build LLVM, Clang, and LLD: GCC 5.1 or later, or
  Clang;
- CMake 3.20 or later, and Ninja or another build system that CMake supports;
- `curl`, `tar` with xz support, `patch`, and `sha256sum` or `shasum`;
- a POSIX system (`sh`, `mkdir`, `cd`) and Python 3.

```sh
git clone --recursive https://github.com/mattneel/zigpp-bootstrap
cd zigpp-bootstrap
./build x86_64-linux-musl baseline
```

`<arch>-<os>-<abi>` is a Zig target and `<mcpu>` is a `-mcpu` value of Zig:
`baseline` for a generic CPU of the architecture, or `native`. The Zig++
distribution for the target appears in `out/zig-x86_64-linux-musl-baseline/`.

The first run builds LLVM, Clang, and LLD twice: once for the host, to build
Zig++ and the LLVM tools the cross build needs, and once for the target, with
Zig++ as the cross compiler. Later runs for other targets reuse the host build.
`CMAKE_GENERATOR=Ninja` and `CMAKE_BUILD_PARALLEL_LEVEL` tune it, `ZIG_SRC`
builds another Zig++ checkout instead of the submodule, and `ZIG_VERSION` sets
the version string it is built as.

To get just the libraries and compiler for a target, as one archive:

```sh
./devkit x86_64-linux-musl baseline
```

```text
out/devkit/zig+llvm+lld+clang-x86_64-linux-musl-0.17.0-dev.2361+zigpp.5b96e6d21.tar.xz
```

That archive holds the target's libraries and headers, `bin/zig`, and `lib/zig/`
for a computer of that target. On Windows, `./devkit x86_64-windows-gnu baseline`
packages the same thing as a `.zip`. Unpack it anywhere and build with it:

```sh
zig build -Dtarget=x86_64-linux-musl -Dstatic-llvm --search-prefix /path/to/devkit
```

`./publish` uploads every devkit of the version in `zig-version` to the
`devkit-<version>` release of zigpp-bootstrap, with a `SHA256SUMS` file, and
Zig++'s `.github/scripts/devkit.sh` names the version that CI downloads. So you
do not have to build a devkit to use one: download it, unpack it, and use it as
the `--search-prefix` above.

An upstream Zig devkit is not a Zig++ compiler, and it cannot build one. Use a
Zig++ devkit, whose LLVM is 23.1.2.

## How CI builds Zig++

Zig++'s CI does not install LLVM. `.github/scripts/bootstrap.sh` downloads the
devkit of the machine, `x86_64-linux-musl`, `x86_64-windows-gnu` (in Git Bash),
or `aarch64-macos-none`, into `~/deps`, where it stays for the next run. It then
builds the compiler with the newest Zig++ release, against this checkout's
`lib/`; on the Linux machine, that is:

```sh
ZIG_LIB_DIR="$PWD/lib" "$RELEASE/zig" build \
  --prefix build-bootstrap/stage3 \
  --search-prefix "$PREFIX" \
  -Dtarget=x86_64-linux-musl -Dcpu=baseline \
  -Denable-llvm -Dstatic-llvm -Duse-zig-libcxx \
  -Doptimize=ReleaseFast -Dstrip -Dno-lib
```

When there is no release, when `stage1/zig1.wasm` changed since the release's
commit, or when the release cannot build the checkout, it bootstraps from
source instead, on Linux and macOS: the CMake build, with the devkit's Zig++ as the C and C++
compiler and the devkit's libraries as the search prefix:

```sh
cmake .. \
  -DCMAKE_INSTALL_PREFIX=stage3 \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$PREFIX/bin/zig;cc;-target;x86_64-linux-musl;-mcpu=baseline" \
  -DCMAKE_CXX_COMPILER="$PREFIX/bin/zig;c++;-target;x86_64-linux-musl;-mcpu=baseline" \
  -DZIG_TARGET_TRIPLE=x86_64-linux-musl \
  -DZIG_TARGET_MCPU=baseline \
  -DZIG_STATIC=ON \
  -DZIG_NO_LIB=ON \
  -GNinja
ninja install
```

Either way, `lib/` stays in the checkout, so the scripts point `ZIG_LIB_DIR` at
this repository's `lib/`. The compiler that comes out is
`build-bootstrap/stage3/bin/zig`, and the release workflow cross-compiles every
target's archive with it.

## Building without LLVM

The only system dependency of this route is a C compiler:

```sh
cc -o bootstrap bootstrap.c
./bootstrap
```

This produces `zig2` in the current working directory: a stage2 build of the
compiler, reported as `0.17.0-dev.bootstrap`, without LLVM extensions, and
therefore lacking:

- release-mode optimizations;
- some ELF linking features, some COFF/PE linking features, and some WebAssembly
  linking features;
- the ability to create static archives from object files;
- the ability to compile assembly files;
- the ability to compile C, C++, Objective-C, and Objective-C++ files.

It still provides an LLVM backend that produces bitcode files, which a
separately installed Clang can optimize and compile, and a C backend that
produces C source. From there, the build system installs a compiler as usual:

```sh
./zig2 build
```

Zig++ is not going upstream's way on LLVM: LLVM is a permanent, first-class
dependency of Zig++, and the LLVM-less `zig2` above exists only to bootstrap
it. Upstream's project to
[completely eliminate the dependency on LLVM library API
calls](https://github.com/ziglang/zig/issues/25492) will not be resolved here,
and [LLVM is forever](what-zigpp-adds.md#llvm-is-forever) explains why.
