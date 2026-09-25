# What Zig++ Adds

Zig++ is a superset of Zig: every valid Zig program is a valid Zig++ program,
unless it names something `priv` (write `@"priv"` instead). This chapter covers
what it adds to the language and to the toolchain.

## Private fields

Struct fields are public by default, meaning that they can be accessed from any
file. A field declared with the `priv` keyword is *private*: it can only be
accessed by name from within the file which declares the struct, just like a
declaration which is not marked `pub`. Private fields allow a type to protect
its invariants, and communicate which fields are implementation details rather
than part of its API.

```zig
/// A fixed-capacity buffer. `len` is private, so code in other files cannot
/// break the invariant that `len` never exceeds the capacity of `bytes`.
pub const Buffer = struct {
    bytes: [16]u8 = undefined,
    priv len: usize = 0,

    pub fn append(buf: *Buffer, byte: u8) error{Overflow}!void {
        if (buf.len == buf.bytes.len) return error.Overflow;
        buf.bytes[buf.len] = byte;
        buf.len += 1;
    }

    pub fn slice(buf: *const Buffer) []const u8 {
        return buf.bytes[0..buf.len];
    }
};
```

Code in the file that declares the type names `len` normally, including in
tests. Code in any other file cannot name it, whether to read it, write it,
take its address, call it, or specify its value in an initialization
expression:

```zig
const Buffer = @import("test_private_fields.zig").Buffer;

test "access a private field from another file" {
    var buf: Buffer = .{};
    try buf.append('a');
    buf.len = 100;
}

// error: field 'len' of struct 'test_private_fields.Buffer' is private
```

Private fields are still a part of every value of the type. If a private field
has no default value, the struct can only be initialized within the file which
declares it, typically by a public initialization function.

Field privacy only applies to syntax which names a field. `@typeInfo` reports
private fields along with all other fields, indicating their privacy with the
`@"priv"` field attribute, and builtins which name fields with strings, such as
`@field`, `@FieldType`, and `@offsetOf`, can be used with any field. This
allows generic code, such as formatting, comparison, hashing, and
serialization, to operate on all types:

```zig
const info = @typeInfo(Buffer).@"struct";
try expect(std.mem.eql(u8, info.field_names[1], "len"));
try expect(info.field_attrs[1].@"priv");

// Builtins which name fields with strings are not subject to field privacy.
var buf: Buffer = .{};
try buf.append('z');
try expect(@field(buf, "len") == 1);
```

A type created with `@Struct` or `@Union` has private fields where the
`@"priv"` field attribute is set; these fields are private to the file
containing the reification builtin call.

Fields of unions can be marked `priv` as well. Outside of the file which
declares the union, a private field cannot be accessed, initialized, or have
its payload captured by a `switch` prong which names it. The tag of a union is
not affected by field privacy, so it can still be compared against and switched
on, and `else` and `inline else` prongs can capture any payload.

Enum fields and tuple fields cannot be marked `priv`.

`priv` precedes `comptime` in a field declaration, so the grammar rule for a
container field is:

```text
ContainerField <- doc_comment? KEYWORD_priv? KEYWORD_comptime? (IDENTIFIER COLON)? TypeExpr ByteAlign? (EQUAL Expr)?
```

Because `priv` is a keyword, an identifier with that name is written
`@"priv"`. That is the whole of the incompatibility with Zig: a Zig program
that declares, or names, something `priv` needs quotes around it, and one that
expects to reach private fields from other files needs a different design.

Upstream closed the [proposal for private
fields](https://github.com/ziglang/zig/issues/9909) as not planned. Zig++'s
release smoke test compiles a struct with a private field, runs it, and checks
that naming that field from another file fails with a diagnostic that mentions
`private`.

The full rules, with the tests above as runnable examples, are in the language
reference: [Private Fields](/langref.html#Private-Fields).

## LLVM is forever

Upstream Zig plans to drop its dependency on the LLVM libraries, and to
[eliminate its dependency on the LLVM library API
calls](https://github.com/ziglang/zig/issues/25492). Zig++ will never phase out
LLVM. In `package.json` terms, LLVM stays in `dependencies`.

Zig++ builds with LLVM, Clang, and LLD 23.1.2. The CMake build requires the
23.x development libraries, and refuses a `llvm-config` older than 23 or newer
than 24.

The LLVM-less build path exists only to bootstrap: `cc -o bootstrap bootstrap.c
&& ./bootstrap` produces a `zig2` that is a stage2 build without LLVM
extensions, and it lacks release-mode optimizations, some ELF, COFF/PE, and
WebAssembly linking features, the ability to create static archives from object
files, the ability to compile assembly files, and the ability to compile C,
C++, Objective-C, and Objective-C++. It is enough to run `./zig2 build` and
produce a real Zig++ compiler. See
[Building from Source](building-from-source.md).

LLVM, and MLIR above it, are how Zig++ goes the final stretch on GPUs: the
blessed path lowers Zig++ directly to PTX, to AMD GPU code objects, and to Metal
libraries, with first-class GPU intrinsics.

## `std.gpu`

`std.gpu` is a port of [ugpu](https://github.com/mattneel/ugpu) into the
standard library. Kernels are plain Zig functions, and they can use the rest of
the standard library as long as they avoid operating system services:

- Kernels are exported functions with the `.kernel` calling convention, and
  `std.gpu` has CUDA's indexing (`threadIdx`, `blockIdx`, `blockDim`,
  `gridDim`, `globalId`), `syncThreads` (the new `@workGroupBarrier` builtin),
  warp shuffles, votes and reductions, atomics, fast math approximations, and
  `print`.
- The standard library runs on the GPU: `std.fmt`, `std.json`, `std.mem`,
  `std.base64`, hash maps, and array lists, with allocators for shared memory
  and for the CUDA device heap in `std.gpu.allocators`. A panic in a kernel
  reports its message to the host.
- Every NVPTX and AMDGPU module carries the compiler-rt routines that it calls,
  so `@sin`, `@exp`, `@log`, `f128`, and float parsing work in kernels, with
  the same results as on the host, bit for bit. Upstream Zig crashes LLVM on
  `@sin` for NVPTX.
- `std.gpu.cuda` loads the CUDA driver at run time, so programs build without
  the CUDA toolkit, and launches kernels from the host. `std.gpu.hip` does the
  same with the HIP runtime of AMD GPUs, on Linux and on Windows, where it
  needs no libc.
- `std.gpu.metal` loads the Metal framework at run time and runs kernels that
  Zig++ compiles for the `air64` target into a `.metallib`, in process, with no
  Xcode, Metal toolchain, or macOS SDK. Apple GPUs have no `f64`, no `print`,
  and only relaxed atomics, which are compile errors in their kernels.

[GPU Programming](gpu.md) covers the device API, the host APIs, and how to
build and run the kernels. Still to come: MLIR lowering for tensor cores and
kernel fusion.

## AI in the toolchain

Upstream Zig bans LLMs from issues, patches, and bug tracker comments. Zig++
welcomes them, and it is putting AI on both sides of compilation: in the
programs it builds and in the build itself; see
[AI Policy and Governance](ai-policy.md).
