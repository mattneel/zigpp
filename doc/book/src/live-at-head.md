# Live at Head

Live at Head. Zig++ has no versions. Every push to master is a release, and
every release is kept forever. You pin the build you use, the way you pin every
dependency, and you upgrade when you are ready. Zig++ does not number its
releases; upstream does it for us.

## The rules

1. **Every push to master is a release.** The
   [Release workflow](https://github.com/mattneel/zigpp/blob/master/.github/workflows/release.yml)
   publishes one for every push, whatever the push changes, and no push's run
   cancels another's. [Versions and Releases](versions.md) says what a release
   holds.

2. **Builds are forever.** Every published build stays downloadable at a stable
   URL with its checksums. It is never deleted and never replaced: a release is
   published once, with all of its files, and never changed after. GitHub
   enforces this for every release published from 26 September 2026 on, which
   are immutable: their files and their tags cannot be changed. The twelve
   published before, 0.17.0-dev.2373 to 0.17.0-dev.2469, are kept by the same
   rule without that lock, because GitHub makes a release immutable only as it
   is published.

3. **Master is append-only.** No force pushes, no rewritten history, no
   retagging. A pin is a commit, and commits do not move. GitHub enforces this
   too: master refuses a push that is not a fast-forward and refuses to be
   deleted, and no tag in the repository can be moved or deleted.

4. **Version identity comes from upstream tags.** Zig++ does not create bare
   semantic-version tags such as `1.0.0`. `zigpp-*` tags identify individual
   Zig++ builds and releases. They are immutable artifact locators and are
   excluded from version derivation. **`zigpp-*` tags never move and never
   disappear.**

   A build's version is upstream's version in development, the number of
   commits since upstream's last release tag, and the commit it was built from:
   `0.17.0-dev.2469+zigpp.04926fc36`. `build.zig` derives it with
   `git describe --match "*.*.*" --exclude "zigpp-*"`, so only upstream's tags
   take part, and the build tag of that build is `zigpp-0.17.0-dev.2469`.

5. **The root project's pin wins.** Dependencies state minimums. A pin older
   than a dependency's minimum is an error that names the dependency and the
   build to move to. The root pin wins today: `zig` runs the build that the
   nearest `build.zig.zon` names and reads no dependency's pin. The minimum
   check is *tooling: planned*.

6. **Breakage is announced at the upgrade, not by a number.** Every commit that
   breaks existing code carries a `Breaking:` trailer saying what breaks and
   what to write instead:

   ```text
   Breaking: std.zig.Server.serveErrorBundle takes the message tag first. Write serveErrorBundle(.error_bundle, bundle).
   ```

   Merges of upstream Zig are the largest source of breakage, so the merge
   commit carries `Breaking:` trailers for upstream's changes in the merged
   range. The trailers are in force today. Collecting them into the download
   index, and printing them when a project upgrades, is *tooling: planned*.

7. **There are no LTS branches.** Security fixes land on head, and advisories
   name the affected range of builds. The advisory format, and a warning on
   every compile with an affected build, are *tooling: planned*.

8. **Docs travel with the pin.** The build you run serves its own std docs and
   language reference. `zig std` serves the standard library documentation of
   the build that runs it, from that build's own sources, and from
   0.17.0-dev.2476 on it serves that build's language reference at
   `/langref.html` too. Every release archive carries the language reference at
   `doc/langref.html`.

9. **Upstream Zig and every vendored dependency are tracked continuously and
   pinned the same way.** Upstream Zig is merged into master, never rebased
   onto, and the libraries Zig++ carries in `lib/` come with it. LLVM is pinned
   by the devkit Zig++ builds against. The packages Zig++ blesses follow the
   same rule: raylibz pins its raylib by commit, from a fork that merges
   raylib's master.

10. **1.0 is a milestone, not a version.** The build that closes it ships like
    every other build. There is no Zig++ 1.0, and the build system enforces it.
    `build.zig` derives every version from upstream's tags, so a bare `1.0.0`
    tag on master would stop every build from then on: the tagged commit fails
    with the first error, and every commit after it with the second.

    ```text
    error: zig version "0.17.0" does not match Git tag "1.0.0"
    error: zig version 0.17.0 must be greater than tagged ancestor "1.0.0"
    ```

## Pinning a build

A project pins the build it is built with in its `build.zig.zon`:

```zig
.minimum_zig_version = "0.17.0-dev.2469+zigpp.04926fc36",
```

Any `zig` command in the project then runs that build, downloading it the
first time; see [Any Zig Version](any-version.md). To install a build by
hand, `ppup install 0.17.0-dev.2469+zigpp.04926fc36`, or take its archive from
[Downloads](downloads.md) and check it against the release's `SHA256SUMS`.

## Upgrading today

1. Pick a newer build on [Downloads](downloads.md), or read
   `.master.version` from [/download/index.json](/download/index.json).
2. Read what broke between your build and that one. The version of each ends
   in its commit, so in a clone of Zig++:

   ```sh
   git log --grep='^Breaking:' --format='%h %s%n%(trailers:key=Breaking,valueonly,separator=%x0A)%n' 04926fc36..70dc1eaec
   ```

3. Move the pin, build, and fix what the notes name.

`zig upgrade` will do the first two steps for you. It is *tooling: planned*.

## Where the tooling stands

| Rule | Today |
| --- | --- |
| Every push to master is a release | Enforced by the Release workflow. |
| Builds are forever | Enforced: immutable releases from 26 September 2026 on. |
| Master is append-only | Enforced by the repository's rules. |
| Version identity comes from upstream tags | Enforced by `build.zig` and `CMakeLists.txt`. |
| The root project's pin wins | Works. Dependency minimums: *tooling: planned*. |
| `Breaking:` trailers | In force. Collected notes and `zig upgrade`: *tooling: planned*. |
| No LTS branches, advisories | No LTS branches. Advisories and compile warnings: *tooling: planned*. |
| Docs travel with the pin | Works: std docs always, the language reference from 0.17.0-dev.2476. |
| Upstream and vendored code tracked and pinned | By practice: merges of upstream, a pinned LLVM devkit. |
| 1.0 is a milestone | Enforced by `build.zig`. |

## Credit, and prior art

"Live at Head" is Abseil's phrase, from Titus Winters' CppCon 2017 talk
["C++ as a 'Live at Head' Language"](https://www.youtube.com/watch?v=tISy7EJQPzI).

- [Mach's nominated Zig versions](https://machengine.org/docs/nominated-zig/):
  every six weeks Mach nominates one nightly Zig for its whole ecosystem to
  pin.
- [Go's toolchain directive](https://go.dev/doc/toolchain): `go.mod` names the
  toolchain a module builds with, and the `go` command downloads and runs it
  when it is newer than itself.
- [Nix's `flake.lock`](https://nix.dev/manual/nix/stable/command-ref/new-cli/nix3-flake):
  every input pinned to a revision and a hash, moved only when you ask.
- [Dated Rust nightly toolchains](https://rust-lang.github.io/rustup/overrides.html#the-toolchain-file):
  a `rust-toolchain.toml` names a nightly by its date, and `rustup` installs
  exactly that build.
