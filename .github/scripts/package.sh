#!/bin/sh
#
# Builds a release of Zig++ for one target with the bootstrap compiler and packs
# it the way upstream packs its releases, into dist/:
#
#   zig-<arch>-<os>-<version>.tar.xz, or .zip for Windows, holding the zig
#   executable, lib/, doc/langref.html, LICENSE, and README.md.
#
#   LANGLREF=langref/doc/langref.html \
#   .github/scripts/package.sh x86_64-linux-musl
#
# Every target's archive holds the same language reference -- building one is
# the slowest step of a package and does not depend on the target -- so it is
# built once, by the bootstrap job, and named in LANGLREF; this script builds
# with -Dno-langref and puts the file where the install step would have.
#
# Cross compilation needs nothing but the target's LLVM libraries, so every
# target builds on the same Linux runner. Run bootstrap.sh first.
set -eux

target=$1
mcpu=${2:-baseline}

: "${LANGLREF:?LANGLREF must name the language reference to pack}"
test -f "$LANGLREF"

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
  -Dno-langref \
  -Dtarget="$target" \
  -Dcpu="$mcpu" \
  -Denable-llvm \
  -Dstatic-llvm \
  -Duse-zig-libcxx \
  -Doptimize=ReleaseFast \
  -Dstrip \
  -Dversion-string="$version"

# doc/langref.html of the -Dflat prefix is where the install step of a package
# without LANGLREF puts the reference.
mkdir -p "dist/$name/doc"
cp "$LANGLREF" "dist/$name/doc/langref.html"

# xz compresses on one core unless XZ_OPT asks for more, and the archives are
# written once here and downloaded many times, so the whole machine is worth
# the wait it saves.
XZ_OPT=-T0
export XZ_OPT

cd dist
case $os in
  windows) rm -f "$name.zip" && zip -qr "$name.zip" "$name" ;;
  *) rm -f "$name.tar.xz" && tar -cJf "$name.tar.xz" "$name" ;;
esac
rm -rf "$name"
ls -l
