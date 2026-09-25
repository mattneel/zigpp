# AI Policy and Governance

## AI policy

Upstream Zig bans LLMs from issues, patches, and bug tracker comments. Zig++
welcomes them. The first Zig++ language feature, private fields, was
implemented, tested, and documented by Claude, and the commit is signed that
way.

Zig++ has AI on both sides of compilation: as a runtime facility in the
programs it builds, and as a build-system facility in the build itself.
Neither is in `std` yet: both are packages today, and both are headed into the
standard library under the names `std.ai` and `std.Build.ai`.

`std.ai` will be [ai.zig](https://github.com/mattneel/ai.zig), the AI toolkit
for Zig: an implementation of the Vercel AI SDK v7 core built on `std.Io`,
with text generation and streaming, multi-step tool loops, structured output,
embeddings, reranking, reusable agents, and an MCP client.

`std.Build.ai` will be [Zigger](https://github.com/mattneel/zigger), AI code
generation as a Zig build step. Today it is a package that adds a
`zig build gen` step: it reads `SPEC.md`, runs the Claude CLI to implement it,
runs `zig build test`, feeds any failures back, and repeats until the tests
pass (up to 10 times by default). With `-Dtdd=true` it writes failing tests
first. The plan is to move this pipeline into `std.Build.ai`.

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
[github.com/mattneel/zigpp](https://github.com/mattneel/zigpp). CI runs on the
project's own machines: x86_64 Linux under WSL2 and x86_64 Windows, on a laptop
with an RTX 5090 Laptop GPU, and an M4 Mac. On each it builds Zig++ and runs:

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
The Linux machine runs the CUDA kernels on the RTX 5090 Laptop GPU, and the
Mac runs the Metal kernels on the M4; the suite skips what a machine has no
driver for.

A pull request that changes only documentation, ppup, or the other workflows
does not run CI, and neither does one from a fork: the machines run only
branches of the repository itself.
[Building from Source](building-from-source.md) has the same bootstrap the CI
job uses, so a change can be tested against the compiler CI builds.
