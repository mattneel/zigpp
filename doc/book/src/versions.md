# Versions and Releases

## What a version says

```text
0.17.0-dev.2361+zigpp.5b96e6d21
└────┬───┘ └─┬─┘ └──────┬──────┘
     │       │          └─ build metadata: zigpp, and the commit it was built from
     │       └─ commits since the 0.16.0 tag
     └─ the upstream version it is based on
```

The base version is the last upstream Zig release that this commit descends
from: `0.17.0`. `0.16.0` is the tag that the commit count is measured from, so
`dev.2361` is 2361 commits after it. Both numbers come from `git describe`, and
the commit is its abbreviated hash. The base version and the tagged ancestor
are checked against each other, and a checkout where `git describe` cannot
answer is built as the plain base version.

The version is what `zig version` prints, what `zig env` reports, and what names
the archives of a release. It is also what tells two Zig++ builds apart: unlike
`0.17.0-dev`, it is impossible for two different Zig++ compilers to report the
same version.

The builder takes `-Dversion-string` to set it by hand, which the release
workflow does not need: it reads the version out of the compiler it just built.

## Rolling releases

The [Release
workflow](https://github.com/mattneel/zigpp/blob/master/.github/workflows/release.yml)
publishes a release on **every push to master**, for four targets:

| Target | Runner that checks it |
| --- | --- |
| x86_64-linux | ubuntu-24.04 |
| aarch64-linux | ubuntu-24.04-arm |
| aarch64-macos | macos-15 |
| x86_64-windows | windows-2025 |

The jobs are: build a bootstrap compiler from source, pack an archive for each
target, unpack and smoke-test each archive on a runner of that architecture,
and publish. The smoke test runs `zig version` and `zig env`, compiles and runs
a program that uses private fields, checks that naming a private field from
another file is rejected, and compiles and runs C with the Clang and libc the
archive carries.

A pull request that changes the release machinery builds the archives and
checks them, and publishes nothing.

The release's tag is the version without its build metadata —
`zigpp-0.17.0-dev.2361` — because `+` is legal in a tag but awkward in URLs.
`build.zig` and `CMakeLists.txt` exclude `zigpp-*` tags when they derive the
version, so the tag itself does not change it.

Each release carries:

- one archive per target, named `zig-<arch>-<os>-<version>.tar.xz`, or `.zip`
  for Windows;
- `index.json`, the download index described below;
- `SHA256SUMS`, the SHA-256 of every archive and of `index.json`.

Publishing the same version twice replaces the assets, so a publish that failed
halfway can be completed by re-running it.

## The download index

[/download/index.json](/download/index.json) collects every release in one
document, in the format of
[ziglang.org's own download index](https://ziglang.org/download/index.json), so
tools that read that one can read this one. `master` is the newest release, and
every release also has a key of its own:

```json
{
  "master": {
    "version": "0.17.0-dev.2361+zigpp.5b96e6d21",
    "date": "2026-09-24",
    "x86_64-linux": {
      "tarball": "https://github.com/mattneel/zigpp/releases/download/zigpp-0.17.0-dev.2361/zig-x86_64-linux-0.17.0-dev.2361%2Bzigpp.5b96e6d21.tar.xz",
      "shasum": "e3b0c442...",
      "size": "52345678"
    },
    "aarch64-linux": { "...": "..." },
    "aarch64-macos": { "...": "..." },
    "x86_64-windows": { "...": "..." }
  },
  "0.17.0-dev.2361+zigpp.5b96e6d21": { "...": "same as master" }
}
```

Each target key holds the URL of its archive, its SHA-256, and its size in
bytes. Every release has the same four in the same place, so a script that
wants the newest Linux build for its architecture reads
`.master["x86_64-linux"].tarball` and checks it against `.shasum`.

Because every push to master is a release, `master` in this index is a real
build, not a nightly snapshot: it is the newest published release.

## What the site serves

The standard library documentation at [/std/](/std/index.html) and the language
reference at [/langref.html](/langref.html) are generated from the newest
release, so they describe the compiler you download from
[Downloads](downloads.md) rather than an unreleased master.

Until the first release is published, both are stand-ins that say so, and the
Downloads chapter explains what a release will contain.
