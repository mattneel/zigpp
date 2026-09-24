#!/bin/sh
#
# Builds Zig++ from source for the machine it runs on with the CMake build, the
# way upstream's ci/*-release.sh scripts start: bootstrap.c's zig2 builds the
# compiler with the LLVM 23.1.2 of zigpp-bootstrap's devkit for this host, into
# a static binary. An upstream Zig cannot build Zig++, so every other build
# starts from this one. Hosts: x86_64 and aarch64 Linux, and aarch64 macOS.
#
# The compiler is left at build-bootstrap/stage3/bin/zig, without lib/: point
# ZIG_LIB_DIR at the lib/ of this checkout to use it.
set -eux

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) TARGET=x86_64-linux-musl ;;
  Linux-aarch64) TARGET=aarch64-linux-musl ;;
  Darwin-arm64) TARGET=aarch64-macos-none ;;
  *) echo "no devkit for $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac
MCPU=baseline
PREFIX=$(.github/scripts/devkit.sh "$TARGET")
ZIG="$PREFIX/bin/zig"

# GitHub's macOS runners have 7.5 GB of memory, less than the 8 GB that
# compiling the compiler declares it may use, and zig build refuses to start a
# step that declares more than the machine has. macOS swaps instead of failing,
# so the budget is raised to the declaration there.
extra_build_args=""
if [ "$(uname -s)" = Darwin ]; then
  extra_build_args="--maxrss;8000000000"
fi

mkdir -p build-bootstrap
cd build-bootstrap

cmake .. \
  -DZIG_EXTRA_BUILD_ARGS="$extra_build_args" \
  -DCMAKE_INSTALL_PREFIX=stage3 \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ZIG;cc;-target;$TARGET;-mcpu=$MCPU" \
  -DCMAKE_CXX_COMPILER="$ZIG;c++;-target;$TARGET;-mcpu=$MCPU" \
  -DZIG_TARGET_TRIPLE="$TARGET" \
  -DZIG_TARGET_MCPU="$MCPU" \
  -DZIG_STATIC=ON \
  -DZIG_NO_LIB=ON \
  -GNinja

ninja install

stage3/bin/zig version
