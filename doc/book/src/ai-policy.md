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
with a bootstrap devkit and runs, in this order:

```sh
build-bootstrap/stage3/bin/zig build test-fmt
build-bootstrap/stage3/bin/zig test lib/std/std.zig -lc
build-bootstrap/stage3/bin/zig test test/behavior.zig
build-bootstrap/stage3/bin/zig build test-cases -Dskip-non-native -Denable-llvm
build-bootstrap/stage3/bin/zig build test-standalone -Dskip-non-native -Dskip-release
```

The standalone tests include the [GPU suite](gpu.md): its kernels compile for
NVIDIA and AMD GPUs, and it skips itself on a machine without a GPU driver.

[Building from Source](building-from-source.md) has the same bootstrap the CI
job uses, so a change can be tested against the compiler CI builds.
