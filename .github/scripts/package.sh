#!/bin/sh
#
# Builds a release of Zig++ for one target with the bootstrap compiler and packs
# it the way upstream packs its releases, into dist/:
#
#   zig-<arch>-<os>-<version>.tar.xz, or .zip for Windows, holding the zig
#   executable, lib/, doc/langref.html, LICENSE, and README.md.
#
#   .github/scripts/package.sh x86_64-linux-musl
#
# Cross compilation needs nothing but the target's LLVM libraries, so every
# target builds on the same Linux runner. Run bootstrap.sh first.
set -eux

target=$1
mcpu=${2:-baseline}

export ZIG_LIB_DIR="$PWD/lib"
zig="$PWD/build-bootstrap/stage3/bin/zig"
version=$("$zig" version)
prefix=$(.github/scripts/devkit.sh "$target")

arch=${target%%-*}
os=${target#*-}
os=${os%%-*}
name="zig-$arch-$os-$version"

rm -rf "dist/$name"
"$zig" build \
  --prefix "dist/$name" \
  --search-prefix "$prefix" \
  -Dflat \
  -Dtarget="$target" \
  -Dcpu="$mcpu" \
  -Denable-llvm \
  -Dstatic-llvm \
  -Duse-zig-libcxx \
  -Doptimize=ReleaseFast \
  -Dstrip \
  -Dversion-string="$version"

cd dist
case $os in
  windows) rm -f "$name.zip" && zip -qr "$name.zip" "$name" ;;
  *) rm -f "$name.tar.xz" && tar -cJf "$name.tar.xz" "$name" ;;
esac
rm -rf "$name"
ls -l
