#!/bin/sh
#
# Downloads the static LLVM, Clang, and LLD 23.1.2 libraries that Zig++ builds
# with, for one target, and prints the directory they are in.
#
#   .github/scripts/devkit.sh x86_64-linux-musl
#
# They are the devkits of https://github.com/mattneel/zigpp-bootstrap, which
# builds them from the LLVM release source, together with a Zig++ that can
# compile C and C++ for the target. `version` is the Zig++ that zigpp-bootstrap
# built them with; update it when it publishes new ones.
set -eu

version="0.17.0-dev.2364+zigpp.56013e0da"
repo="mattneel/zigpp-bootstrap"

target=$1
deps=${DEPS_DIR:-"$HOME/deps"}

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
  # GitHub keeps the "+" of the version in asset names and expects it
  # percent-encoded in their URLs.
  url_version=$(printf '%s' "$version" | sed 's/+/%2B/g')
  url_name=$(printf '%s' "$name" | sed 's/+/%2B/g')
  curl -sSfL --retry 5 --retry-all-errors -o "$work/$name.$extension" \
    "https://github.com/$repo/releases/download/devkit-$url_version/$url_name.$extension"
  case $extension in
    zip) (cd "$work" && unzip -q "$name.$extension") ;;
    tar.xz) tar -xJf "$work/$name.$extension" -C "$work" ;;
  esac
  mv "$work/$name" "$deps/$name"
fi

echo "$deps/$name"
