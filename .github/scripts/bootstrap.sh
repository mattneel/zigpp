#!/bin/sh
#
# Builds Zig++ from source for the machine it runs on: a static compiler with
# the LLVM 23.1.2 of zigpp-bootstrap's devkit for this host, which every other
# build starts from, because an upstream Zig cannot build Zig++. Hosts: x86_64
# and aarch64 Linux, and aarch64 macOS.
#
# The compiler of the newest Zig++ release builds it when it can. Otherwise it
# is built from source alone, the way upstream's ci/*-release.sh scripts start:
# CMake builds zig1 from stage1/zig1.wasm, zig1 translates the compiler to C,
# and the resulting zig2 builds the compiler. That takes about ten minutes
# longer, and it runs when there is no release, when stage1/zig1.wasm changed
# since the release's commit (the sign of a language change the release may not
# know), and when the release fails to build this source.
#
# The release comes from GITHUB_REPOSITORY (mattneel/zigpp by default) through
# gh, which needs a token in GH_TOKEN on CI.
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

# GitHub's macOS runners have 7.5 GB of memory, less than the 8 GB that
# compiling the compiler declares it may use, and zig build refuses to start a
# step that declares more than the machine has. macOS swaps instead of failing,
# so the budget is raised to the declaration there.
maxrss=""
if [ "$(uname -s)" = Darwin ]; then
  maxrss=8000000000
fi

# Builds the compiler with the newest release's, or fails. The shell does not
# stop at failing commands in a function called as a condition, hence the
# explicit returns.
from_release() {
  repo=${GITHUB_REPOSITORY:-mattneel/zigpp}
  arch=${TARGET%%-*}
  os=${TARGET#*-}
  os=${os%%-*}
  release=build-bootstrap/release

  tag=$(gh release view --repo "$repo" --json tagName --jq .tagName) || return 1
  rm -rf "$release" && mkdir -p "$release" || return 1
  gh release download "$tag" --repo "$repo" --dir "$release" \
    --pattern "zig-$arch-$os-*.tar.xz" || return 1
  archive=$(cd "$release" && echo zig-*.tar.xz)
  tar -xJf "$release/$archive" -C "$release" || return 1
  name=${archive%.tar.xz}
  version=${name#"zig-$arch-$os-"}

  if ! git diff --quiet "${version##*+zigpp.}" HEAD -- stage1/zig1.wasm; then
    echo "stage1/zig1.wasm changed since Zig++ $version" >&2
    return 1
  fi

  # The compiler must be built with the standard library of this checkout,
  # not the release's.
  ZIG_LIB_DIR="$PWD/lib" "$release/$name/zig" build \
    --prefix build-bootstrap/stage3 \
    --search-prefix "$PREFIX" \
    ${maxrss:+--maxrss "$maxrss"} \
    -Dtarget="$TARGET" \
    -Dcpu="$MCPU" \
    -Denable-llvm \
    -Dstatic-llvm \
    -Duse-zig-libcxx \
    -Doptimize=ReleaseFast \
    -Dstrip \
    -Dno-lib
}

from_source() {
  ZIG="$PREFIX/bin/zig"
  mkdir -p build-bootstrap
  cd build-bootstrap

  cmake .. \
    -DZIG_EXTRA_BUILD_ARGS="${maxrss:+--maxrss;$maxrss}" \
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
  cd ..
}

if ! from_release; then
  echo "building Zig++ from source alone" >&2
  from_source
fi

ZIG_LIB_DIR="$PWD/lib" build-bootstrap/stage3/bin/zig version
