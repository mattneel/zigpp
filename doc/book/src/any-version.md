# Any Zig Version

A project can pin the exact compiler it is built with, in its
`build.zig.zon`:

```zig
.minimum_zig_version = "0.15.1",
```

When it does, every `zig` command run in that project, or in any directory
below it, runs that version instead of the `zig` that was invoked: `zig build`,
`zig test`, `zig env`, and `zig cc` alike. The first time a version is needed,
Zig++ downloads it into the global cache and verifies it; after that, it only
reads the project's `build.zig.zon` and starts the installed compiler.

`zig init` writes the version of the compiler that ran it, so a new project
keeps building with the compiler that made it.

The version names the exact compiler, the way
[anyzig](https://github.com/marler8997/anyzig) reads it, not a lower bound. It
can be:

| Version | Where it comes from | Verified with |
| --- | --- | --- |
| a Zig++ release, `0.17.0-dev.2361+zigpp.5b96e6d21` | the GitHub release `zigpp-0.17.0-dev.2361` | the SHA-256 in the release's `index.json` |
| an upstream Zig release, `0.15.1` | [ziglang.org/download](https://ziglang.org/download/) | the SHA-256 in upstream's download index, and the minisign signature |
| an upstream dev build, `0.16.0-dev.1234+abcdef012` | `ziglang.org/builds` | the minisign signature |

Every upstream archive is checked against the minisign signature that
ziglang.org publishes next to it, made with the Zig Software Foundation's key,
and the signature has to name that archive. ziglang.org keeps only its recent
dev builds, so an archive it no longer has comes from one of the
[community mirrors](https://ziglang.org/download/community-mirrors.txt), tried
in a random order, whose copy has to pass the same check. An archive that fails
its check is never installed.

## Running a version by hand

```sh
zig any 0.15.1 version   # run that exact version, installing it if needed
zig any list             # the installed versions, one per line
ZIG_ANY=off zig version  # the zig that was invoked, whatever the project pins
```

`zig any <version>` takes any command after the version, the same way `zig`
does.

## Where versions go

Each version lives in `<global cache>/any/<version>/`, next to the packages
that `zig fetch` downloads; `zig env` prints the global cache directory. An
install downloads and unpacks in a temporary directory and moves the result
into place in one step, so a version is either complete or absent, even if the
download is interrupted. Two `zig` commands that need the same new version at
the same time end up with one install of it.

Deleting a version's directory uninstalls it.

## What the other version sees

The other version runs with the same arguments and the same environment,
except that `ZIG_LIB_DIR` is removed: it points at the standard library of the
`zig` that was invoked.

A compiler that runs from `<global cache>/any/<version>/` never dispatches to
that version again, so an installed Zig++ does not restart itself. Every other
`zig` still dispatches: a build step that runs `zig` from `PATH` gets the pinned
version too, and a project nested inside another one, pinning a different
version, gets its own. `zig any <version>` starts its compiler with
`ZIG_ANY=off`, so the version asked for runs even inside a project that pins
another one.

If no source has the version, `zig` says what it tried, and that `ZIG_ANY=off`
runs the compiler that was invoked instead.

A project created by a Zig++ that was never released, such as one you built
from source, pins a version that has nothing to download. Build it with
`ZIG_ANY=off`, or change `minimum_zig_version` to a released version.
