#!/bin/sh
#
# Builds Zig++ for x86_64 Linux from source with the CMake build, the way
# upstream's ci/x86_64-linux-release.sh starts: bootstrap.c's zig2 builds the
# compiler with the LLVM 23.1.2 of zigpp-bootstrap's devkit, into a static
# binary that runs on any x86_64 Linux. An upstream Zig cannot build Zig++, so
# every other build starts from this one.
#
# The compiler is left at build-bootstrap/stage3/bin/zig, without lib/: point
# ZIG_LIB_DIR at the lib/ of this checkout to use it.
set -eux

TARGET=x86_64-linux-musl
MCPU=baseline
PREFIX=$(.github/scripts/devkit.sh "$TARGET")
ZIG="$PREFIX/bin/zig"

mkdir -p build-bootstrap
cd build-bootstrap

cmake .. \
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
