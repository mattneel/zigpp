# Downloads

Zig++ has no published release yet. The
[Release workflow](https://github.com/mattneel/zigpp/blob/master/.github/workflows/release.yml)
publishes one on every push to master, for x86_64-linux, aarch64-linux, aarch64-macos,
and x86_64-windows; the newest one appears here. The standard library documentation
and the language reference on this site come from that release, so they appear with it.

<details>
<summary>What a release contains</summary>

An archive is named `zig-<arch>-<os>-<version>.tar.xz`, or `.zip` on Windows. It holds the
`zig` executable, `lib/`, `doc/langref.html`, `LICENSE`, and `README.md`. The
[Installing](installing.html) chapter covers how the compiler finds the `lib/` it carries.

`SHA256SUMS` lists the SHA-256 of every archive and of `index.json`.
</details>

The machine-readable index of every release is at [download/index.json](download/index.json).
