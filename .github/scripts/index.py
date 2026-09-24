#!/usr/bin/env python3
"""Writes the download index of a Zig++ release.

    .github/scripts/index.py <dist directory> <version> <release tag> <owner/repo>

The index has the format of upstream's https://ziglang.org/download/index.json,
so tools that read that index can read this one: "master" names the newest
build, with the tarball, SHA-256, and size for each target. The archives in
the dist directory are named zig-<arch>-<os>-<version>.tar.xz or .zip, and
their URLs are those of the assets of the release, where GitHub keeps the "+"
of the version and percent-encodes it.
"""

import datetime
import hashlib
import json
import os
import sys
import urllib.parse


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    dist, version, tag, repo = sys.argv[1:5]
    base = f"https://github.com/{repo}/releases/download/{urllib.parse.quote(tag)}"
    entry = {
        "version": version,
        "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
    }
    suffix = "-" + version
    for name in sorted(os.listdir(dist)):
        stem = name
        for extension in (".tar.xz", ".zip"):
            if stem.endswith(extension):
                stem = stem[: -len(extension)]
                break
        else:
            continue
        if not stem.startswith("zig-") or not stem.endswith(suffix):
            continue
        key = stem[len("zig-") : -len(suffix)]
        path = os.path.join(dist, name)
        entry[key] = {
            "tarball": f"{base}/{urllib.parse.quote(name)}",
            "shasum": sha256(path),
            "size": str(os.path.getsize(path)),
        }
    if len(entry) == 2:
        sys.exit(f"no archives of version {version} in {dist}")
    with open(os.path.join(dist, "index.json"), "w") as file:
        json.dump({"master": entry}, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
