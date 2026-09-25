# AI Policy and Governance

## AI policy

Upstream Zig bans LLMs from issues, patches, and bug tracker comments. Zig++
welcomes them. The first Zig++ language feature, private fields, was
implemented, tested, and documented by Claude, and the commit is signed that
way.

Zig++ is also building AI code generation into the build system. Today that is
[Zigger](https://github.com/mattneel/zigger), a package that adds a
`zig build gen` step: it reads `SPEC.md`, runs the Claude CLI to implement it,
runs `zig build test`, feeds any failures back, and repeats until the tests pass
(up to 10 times by default). With `-Dtdd=true` it writes failing tests first.
The plan is to make this pipeline part of `std.Build`.

The goal: by September 2027, all new Zig++ code is written through the Zigger
pipeline.

## Governance

Upstream Zig is BDFN (Benevolent Dictator For Now). Zig++ is BDFL: Matthew Neel
is the Benevolent Dictator For Life and has final say on the design and
implementation of everything.

Zig++ has no Code of Conduct. It has one rule: talk about code. Issues, pull
requests, reviews, and comments are for the compiler, the language, the
standard library, and the tools. Everything else, politics included, is off
topic and will be closed.

Language proposals are welcome. Zig++ is made of them.

## Contributing

The source, the issues, and the pull requests are at
[github.com/mattneel/zigpp](https://github.com/mattneel/zigpp). CI builds Zig++
once on x86_64 Linux and once on aarch64 macOS, then runs four jobs on each at
the same time:

```sh
# The standard library and behavior tests, with both back ends.
zig build test-fmt
zig test lib/std/std.zig -lc
zig test test/behavior.zig
zig test lib/std/std.zig -lc -fllvm
zig test test/behavior.zig -fllvm
# Zig++ against C that the Clang in Zig++ compiled.
zig build test-c-abi -Dskip-non-native -Dskip-release
zig build test-cases -Dskip-non-native -Denable-llvm
zig build test-standalone -Dskip-non-native -Dskip-release
```

The standalone tests include the [GPU suite](gpu.md): its kernels compile for
NVIDIA, AMD, and Apple GPUs, and it runs them wherever a GPU driver is present.
The macOS runners run the Metal kernels, and the suite skips itself on a machine
without a GPU driver.

A pull request that changes only documentation, ppup, or the other workflows
does not run CI.
[Building from Source](building-from-source.md) has the same bootstrap the CI
job uses, so a change can be tested against the compiler CI builds.
