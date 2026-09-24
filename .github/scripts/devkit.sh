#!/bin/sh
#
# Downloads the LLVM, Clang, and LLD libraries that upstream Zig builds its
# releases with, for one target, and prints the directory they are in.
#
#   .github/scripts/devkit.sh x86_64-linux-musl
#
# They are zig-bootstrap builds published at ziglang.org/deps, with a Zig
# compiler to build C and C++ with. The version is the one that upstream's
# ci/x86_64-linux-release.sh uses, so merging an upstream change of LLVM
# version moves these builds along with it.
set -eu

target=$1
deps=${DEPS_DIR:-"$HOME/deps"}

version=$(sed -n 's/^CACHE_BASENAME="zig+llvm+lld+clang-\$TARGET-\(.*\)"$/\1/p' ci/x86_64-linux-release.sh)
if [ -z "$version" ]; then
  echo "cannot find the devkit version in ci/x86_64-linux-release.sh" >&2
  exit 1
fi

name="zig+llvm+lld+clang-$target-$version"
case $target in
  *-windows-*) extension=zip ;;
  *) extension=tar.xz ;;
esac

if [ ! -d "$deps/$name" ]; then
  # Download and unpack next to the destination and move the result into
  # place, so that an interrupted download never leaves a partial devkit
  # where the next run would take it for a complete one.
  mkdir -p "$deps"
  work=$(mktemp -d "$deps/.devkit.XXXXXX")
  trap 'rm -rf "$work"' EXIT
  curl -sSfL --retry 5 --retry-all-errors -o "$work/$name.$extension" \
    "https://ziglang.org/deps/$name.$extension"
  case $extension in
    # The Windows devkit is a zip made on Windows: its paths use backslashes,
    # and its directories carry modes without the search bit, which unzip
    # applies and then cannot descend into. Python's zipfile ignores both.
    zip)
      python3 -c '
import os, shutil, sys, zipfile
archive, destination = sys.argv[1:3]
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        path = os.path.join(destination, info.filename.replace("\\", "/"))
        if path.endswith("/"):
            os.makedirs(path, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with z.open(info) as source, open(path, "wb") as target:
            shutil.copyfileobj(source, target)
' "$work/$name.$extension" "$work"
      ;;
    tar.xz) tar -xJf "$work/$name.$extension" -C "$work" ;;
  esac
  mv "$work/$name" "$deps/$name"
fi

echo "$deps/$name"
