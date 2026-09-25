# raylibz

A proposal for raylibz, the Zig++-sanctioned wrapper for [raylib](https://github.com/raysan5/raylib).
It ships as a package, not in std. Every claim about raylib is grounded in the `6.0` tag, the
newest release (published 2026-04-23). Every claim about this tree is grounded in `master` at
`ee991009e8`. The research commands and their results are in the pull request that opens this
document. Anything inferred rather than read or run is marked *[inferred]*.

## Summary

Nothing gets pixels on screen faster than raylib: one header, plain C99, and a function for
everything a game needs. raylibz is how Zig++ says so.
It is a thin, 1:1 wrapper over raylib's own translated headers, built with raylib's own build
script, consumed through the workflow Zig++ blesses for any C library. It adds nothing raylib does
not have, renames nothing beyond Zig casing, hides nothing, and never forks raylib.

## Thanks

To Ramon Santamaria (raysan) and raylib's contributors: thank you. raylib has taught a great many
people to program, and it keeps the promise its README makes: *raylib is a simple and easy-to-use
library to enjoy videogames programming.* It has done that since 2013, under the zlib license,
given away.

raylib is sustained by its users. If raylibz gets you a window, please consider
[sponsoring raysan](https://github.com/sponsors/raysan5) or supporting raylib on
[Patreon](https://www.patreon.com/raylib).

## Why a wrapper and not std

The standard library holds protocols and the runtime. raylib is neither: it is a framework, a peer
of what a program builds on, with its own windowing, input, audio and rendering backends. Vendoring
it into std would put every one of those backends into the compiler's test matrix. That would be
GLFW, RGFW, three generations of SDL, DRM, Win32, Android, OpenGL 1.1 through 4.3, GLES 2 and 3,
and a software renderer (`build.zig:545-586` in raylib 6.0). It would also tie raylib's release
cadence to Zig++'s.

raylibz is a package instead. That is also the point of it. It is the reference example of the
blessed workflow for consuming *any* C library from Zig++: fetch it, build it with its own build
script, translate its headers, and put a Zig face on the translation. The Zig++ book's chapter on
blessed packages will walk through raylibz as that example.

## The blessed workflow

1. **Fetch the pinned release.**
   `zig fetch --save https://github.com/raysan5/raylib/archive/refs/tags/6.0.tar.gz` records the
   URL, and the hash of the unpacked package, in `build.zig.zon`. The tarball's SHA-256, as
   downloaded for this document, is
   `2b3ee1e2120c7a0796b33062c7e9a694dd8a8caa56a96319ac8c8ecf54a90d0b`.
2. **Build raylib with raylib's own `build.zig`.** raylibz calls `b.dependency("raylib", ...)`
   and links `dependency.artifact("raylib")`. raylib's build script owns every platform choice. It
   exposes these options (`build.zig:508-540`), and raylibz forwards all of them unchanged:
   - `platform`: `glfw`, `rgfw`, `sdl`, `sdl2`, `sdl3`, `memory`, `win32`, `drm` or `android`.
   - `linux_display_backend`: `None`, `X11`, `Wayland` or `Both`.
   - `opengl_version`: `auto`, `gl_soft`, `gl_1_1`, `gl_2_1`, `gl_3_3`, `gl_4_3`, `gles_2` or
     `gles_3`.
   - The module switches `raudio`, `rmodels`, `rtext`, `rtextures` and `rshapes`.
   - `raygui`, `linkage`, `config`, `android_ndk` and `android_api_version`.
3. **Translate the headers.** This tree runs translate-c from `build.zig` with
   `b.addTranslateC(.{ .root_source_file, .target, .optimize })` (`lib/std/Build.zig:952`,
   `lib/std/Build/Step/TranslateC.zig:23-28`). The translator is the Aro-based tool in
   `lib/compiler/translate-c/`. raylib's own `build.zig` already runs this step. It publishes the
   translations of `raylib.h`, `rcamera.h`, `raymath.h` and `rlgl.h` as modules, each linked
   against the raylib artifact (`build.zig:588-603,620-623`). raylibz takes those modules as they
   are, so the C declarations raylibz wraps are exactly the ones raylib's own users get. For a
   library whose build script does not translate its headers, the consumer adds the same
   `addTranslateC` call itself. The book chapter shows both.
4. **Put the wrapper on top.** `raylibz` is a Zig module that imports the translation and
   re-exposes it under the rules of the next section.

**Step 2 does not work today.** raylib 6.0's `build.zig` is written for Zig 0.16.0's build API
(`build.zig.zon`: `minimum_zig_version = "0.16.0"`). Zig++ tracks upstream Zig's master, whose
build system has moved on, and this tree's `zig build` rejects the script:

- `b.findProgram(&.{"wayland-scanner"}, &.{})` (`build.zig:95`): `findProgram` now takes one
  options struct (`lib/std/Build.zig:1624`).
- `b.build_root.handle` (`build.zig:651,757`): the field is now `b.root`, a `Cache.Path`
  (`lib/std/Build.zig:37`).
- `b.getInstallPath` (`build.zig:721`): no longer exists.
- `LazyPath.getPath` in the `build.zig` of zemscripten (commit `3fa4b778`, lines 15, 27, 39 and
  49): no longer exists. zemscripten is a dependency raylib does not mark lazy, so every consumer
  compiles it.

The first compile reports the first two. With those changed in a scratch copy, the next compile
reports the other two. raylib's `master` has the same `findProgram` and `build_root` calls; its
`build.zig` last changed on 2026-08-12 (`77353e7f31`). raylibz does not patch, vendor or rewrite
raylib's build. The fix goes upstream, and it is Stage 1 of the plan.

One more fact about the tree, found while doing this research. Zig++'s version dispatch treats a
project's `minimum_zig_version` as the exact compiler to run (`src/main.zig:609-634`), as anyzig
does. So `zig build` inside raylib's own source tree runs Zig 0.16.0, not Zig++. Projects that
depend on raylib are unaffected: dispatch reads the manifest of the project being built, not those
of its dependencies. Running it also exposed a hang. The compiler's own `zig ld.lld` child
dispatched too, while the parent held the global cache to install 0.16.0. That is fixed in
`c0123d3a7f`. The research builds set `ZIG_ANY=off`.

### What translate-c makes of raylib 6.0

The three headers translate, and every public declaration of the three modules compiles apart
from those listed here. The check references each of the others, and compiles:
`zig test check.zig -lc -fno-emit-bin`.

| Header | Translated | Untranslatable, and what they are |
| --- | --- | --- |
| `raylib.h` (2,006 lines of Zig) | All 600 `RLAPI` functions as `extern fn`, including the variadic `TraceLog` and `TextFormat`. Every struct as an `extern struct`. Every enum as integer constants plus a `c_uint` type, e.g. `KEY_A: c_int = 65` and `KeyboardKey = c_uint`. All 26 colour macros, as `zeroInit(CLITERAL(Color), …)`. | 16 macros. 7 are the compiler's predefined suffix and segment macros, and 5 are the `va_*` macros of `stdarg.h`. The other 4 are raylib's allocator hooks `RL_MALLOC`, `RL_CALLOC`, `RL_REALLOC` and `RL_FREE`, which name `malloc` and friends without a header that declares them. None of them is raylib API. |
| `raymath.h` (5,154) | All 146 `RMAPI` functions, with their bodies, as Zig functions. | 83 macros: the same 7 predefined ones, `RMAPI` itself, and the machinery of glibc's `sys/cdefs.h` and `math.h` (`__REDIRECT`, `__THROW`, `__attribute_malloc__` and the like). 9 declarations depend on them. None of them is raymath API. |
| `rlgl.h` (852) | All 163 `RLAPI` functions. | 11 macros: the same 7 predefined ones and the same 4 allocator hooks. And one declaration that translates but does not compile: `#define TRACELOG(level, ...) (void)0` (`rlgl.h:130`) becomes `pub inline fn TRACELOG(level: anytype) anyopaque`, and Zig rejects an opaque return type. That is a translate-c bug, fixed in Zig++ as part of Stage 1. Upstream Zig does not take patches written by LLMs, so it is not offered there. |

## Design of the wrapper

| Area | Decision | Reason |
| --- | --- | --- |
| Coverage | 1:1 with `raylib.h`. Every one of its 600 functions is wrapped, re-exported unchanged, or listed in `not_wrapped.zig` with a reason. `tests/parity.zig` walks the translated module's function declarations and fails on any that is in none of the three. | raylib's API is the spec. A wrapper that silently drops a function has changed the spec. |
| Names | raylib's, in Zig case: `InitWindow` → `initWindow`, `IsKeyPressed` → `isKeyPressed`. Struct names are unchanged: `Vector2`, `Texture2D`, `Camera3D`. | Anyone who knows raylib already knows raylibz. Its cheatsheet stays the documentation. |
| Structs | `extern struct` mirrors, field for field. A comptime block asserts `@sizeOf`, `@alignOf` and every field's `@offsetOf` against the translated type. They cross the boundary with `@bitCast`. | A mirror can carry methods, which a translated type cannot. The asserts make a layout drift a compile error, not a crash. |
| Text | Text parameters take `[:0]const u8`, so a string literal passes as is. | raylib wants NUL-terminated strings, and Zig's literals already are. |
| Buffers | Pointer-and-count pairs become slices, in and out: `loadFileData(path) ?[]u8`, null where raylib returns `NULL`, rather than a pointer plus an `*c_int`. | The count is part of the value. |
| Loading | A load function that raylib pairs with an `Is*Valid` check returns `error{LoadFailed}!T`, having made that check. raylib 6.0 has 12 such checks, for shaders, images, textures, render textures, fonts, models, materials, model animations, waves, sounds, music and audio streams. `IsFileNameValid` is not one of them. | raylib signals a failed load with a value that fails its `Is*Valid` check. An error union makes the check impossible to forget, and costs exactly the call raylib's examples make anyway. |
| Constants | Every C enum becomes a Zig enum with the same values, non-exhaustive where raylib hands back raw integers (`getKeyPressed`). Field names are the constant names, Zig-cased: `KEY_A` → `.key_a`. Flag sets (`ConfigFlags`, `Gesture`) become `packed struct`s of bools, bit for bit. | The values are raylib's. Only the spelling changes. |
| Colours | The 26 colour macros are decls on `Color`: `Color.lightgray`, `Color.raywhite`. | Zig-cased raylib names, found where a colour is expected. |
| raymath | Methods on the vector and matrix types, named without the type prefix: `Vector2Add(a, b)` → `a.add(b)`, `MatrixRotateX` → `Matrix.rotateX`. Scalar helpers are free functions: `clamp`, `lerp`, `remap`. `Quaternion` is raylib's `typedef Vector4`, so the `Quaternion*` family stays as free functions with their raylib names: `quaternionAdd`. | Methods read the way raymath's own names do. Where one C type carries two families, a method name would be ambiguous. |
| The C layer | The translation is always reachable. `raylibz.c` is raylib's `raylib` module, and `raylibz.raymath` is raylib's `raymath` module, both raw. rlgl is re-exported raw as `raylibz.rlgl` and not wrapped. | Nothing is hidden. A program can always drop to raylib's own declarations. |
| State | No allocation, and no globals beyond raylib's own. Everything is `pub`. | raylibz is a spelling of raylib, not a layer with a life of its own. |

## Zig++ features, used only where they help

**`owned`, once [#7](https://github.com/mattneel/zigpp/issues/7)'s stage 2 lands.** Under `-fborrow-check`, a use of an object after an
`owned` call releases it is a compile error. The borrow checker proposal defines `owned` precisely
(`doc/proposals/borrow-checker.md`, §3.4): it goes on *a parameter of pointer or slice type*, and
means *this call takes ownership of the object the argument points into, and normally releases it.
The object dies at this call.* Passing a frame object to an `owned` parameter is itself rejected
(§9.10). Stage 2 of #7 implements `owned` as its only checked qualifier (§11). Against those rules,
raylib 6.0's release functions fall into three groups:

- **10 take memory raylib allocated, by pointer:** `UnloadFileData`, `UnloadFileText`,
  `UnloadRandomSequence`, `UnloadImageColors`, `UnloadImagePalette`, `UnloadFontData`,
  `UnloadCodepoints`, `UnloadTextLines`, `UnloadModelAnimations` and `UnloadWaveSamples`, plus
  `MemFree`. raylibz's wrappers take the slice the matching load returned, and they get `owned`.
  Using that slice after its unload becomes a compile error. This is exactly the case `owned` was
  designed for.
- **17 take a resource by value:** `UnloadTexture`, `UnloadImage`, `UnloadShader`, `UnloadSound`
  and 13 more. Each argument is a handle to a GPU or audio object, or a struct that holds
  raylib-allocated pointers, passed by value. `owned` cannot name these as #7 specifies it. The
  parameter is not a pointer, and taking the address of a local handle would pass a frame object.
  raylibz keeps raylib's by-value signatures rather than invent a pointer raylib does not have. The
  follow-up issue asks #7 whether handle values get a rule. That is a question for the borrow
  checker's design, not something raylibz decides.
- **2 take nothing:** `CloseWindow` and `CloseAudioDevice`. There is no argument to annotate.

Until stage 2 lands, every load/unload pair is documented in the words the annotation will encode:
*takes ownership of the memory `data` points into, and releases it; the memory dies at this call.*

**`priv` is not used.** raylib's structs are public by design, with every field documented in
`raylib.h`. raylibz respects that.

**Threadz, when it lands.** raylib's window, input and GL context belong to the thread that
created them, and on macOS that must be the process's main thread *[inferred from GLFW's
documented thread rules]*. Under `std.Io.Threadz`, the render loop is a *pinned* task, which never
leaves its worker (`doc/proposals/threadz.md`, Decisions: Affinity). It is pinned to the worker
that runs on the main thread. Everything else a game does can then run as ordinary tasks around
it.

## Versioning

raylibz's major and minor versions track raylib's: raylibz 6.0.x wraps raylib 6.0. The patch
version is the wrapper's own. The first release is 6.0.0.

raylibz's `build.zig.zon` names, as its `minimum_zig_version`, the Zig++ release it is tested with.
Zig++'s version dispatch runs that release inside raylibz's own tree. Projects that depend on
raylibz keep their own pin.

## Testing and CI

- **Build** on Linux, macOS and Windows, with the Zig++ release that `build.zig.zon` names.
- **Parity:** `tests/parity.zig` fails if any function of the translated `raylib.h` is neither
  wrapped, nor re-exported, nor listed in `not_wrapped.zig`. It also fails if `not_wrapped.zig`
  names a function that no longer exists.
- **Layout:** the comptime asserts of every mirrored struct, compiled on all three systems.
- **A window, for real:** on Linux, `examples/basic_window` runs under `xvfb` for 60 frames and
  exits, and CI checks that it exited cleanly.

## Non-goals

- **Forking raylib.** raylibz consumes raylib's releases as raysan publishes them. Any C-side fix
  goes upstream first, and raylibz waits for it rather than carry a patch.
- **Wrapping rlgl.** It is raylib's low-level GL layer. It is re-exported raw, for the programs
  that need it.
- **raygui** in the first release.
- **Hiding the C module.** `raylibz.c` is part of the API, permanently.

## Plan

1. **raylib builds with Zig++, and its headers translate clean.** raylib's `build.zig` and
   zemscripten's move to the build API that Zig++ ships, which is upstream Zig's next. That happens
   through pull requests to raysan5/raylib and zig-gamedev/zemscripten, in whatever form their
   maintainers prefer: both APIs at once, or in step with Zig's next release. raylibz pins the
   raylib commit or tag that carries it, and waits until then. In Zig++, translate-c emits valid
   Zig for `#define TRACELOG(level, ...) (void)0`.
   *Acceptance:* at the pinned raylib, `zig build` succeeds with Zig++ on Linux, macOS and Windows.
   The `raylib`, `raymath` and `rlgl` modules translate, and every public declaration outside the
   untranslatable macros compiles.
2. **The package and its raw layer.** Create mattneel/raylibz, with:
   - `build.zig` and `build.zig.zon` pinning raylib through `zig fetch --save`;
   - `raylibz.c`, `raylibz.raymath` and `raylibz.rlgl` from raylib's own modules;
   - `NOTICE` with raylib's license;
   - a README that credits raysan before anything else;
   - `examples/basic_window.zig`, a port of raylib's `core_basic_window` that keeps its zlib
     notice.

   *Acceptance:* the example builds on all three systems, and runs 60 frames under `xvfb` on Linux.
3. **The wrapper.** The rules of the design table over all 600 functions: names, mirrored structs
   with their layout asserts, text, slices, error unions, enums, flags and colours. Also
   `not_wrapped.zig`, `tests/parity.zig`, and a second example that loads and unloads a texture,
   the ownership pair the borrow checker will eventually see.
   *Acceptance:* `tests/parity.zig` passes with no function unaccounted for. The layout asserts
   compile on all three systems. Both examples build and run.
4. **raymath as methods.**
   *Acceptance:* every one of raymath's 146 functions is a method, a free function, or listed in
   `not_wrapped.zig`. A test calls each method and the translated C function on the same inputs
   and compares their results bit for bit.
5. **CI and 6.0.0.**
   *Acceptance:* CI builds on Linux, macOS and Windows, runs the parity and layout tests and the
   `xvfb` run, and passes. raylibz 6.0.0 is tagged.
6. **The book.** A chapter in the Zig++ book, under blessed packages, that walks through the
   workflow for someone who has never used translate-c, with raylibz as the example. And a
   CHANGELOG entry.
   *Acceptance:* following the chapter on a fresh machine, from installing Zig++ to a window,
   works as written.

Two follow-ups stay out of the stages, each with an issue of its own: `owned` on the unload
functions, blocked on #7, and a blessed package index behind `zig add raylibz`.

## Prior art: raylib-zig

[raylib-zig](https://github.com/raylib-zig/raylib-zig), by Nikolas Wipper (Not-Nik) and its
contributors, MIT-licensed, now in its own organisation, came first and does most of this well.
It tracks raylib 6.0, and it consumes raylib's own `build.zig` as a dependency. Its bindings are
generated from raylib's headers by `lib/generate_functions.py` and then tweaked by hand. Those
bindings:

- give raylib Zig-cased names and `[:0]const u8` strings;
- turn the `Load`/`Is*Valid` pairs into error unions;
- make C enums into Zig enums and `ConfigFlags` into a `packed struct`;
- put colours on `Color` and raymath on the vector types;
- keep the raw declarations reachable as `cdef`.

It also covers raygui, web builds through emscripten, and project templates. raylibz owes it the
shape of nearly every rule above.

raylibz differs in where the C declarations come from. raylib-zig keeps its own extern
declarations, `lib/raylib-ext.zig`, generated and then maintained. raylibz keeps none: it wraps
raylib's own translation, and proves the wrapper against it at compile time with the parity test
and the layout asserts. That is what makes raylibz the reference example of the workflow, rather
than a second set of bindings. raylib-zig's users have no reason to switch. raylibz exists so that
the Zig++ book can show how any C library is consumed, with raylib as the library.

---

This document quotes raylib's `raylib.h`, `rlgl.h`, `build.zig` and README, under raylib's license:

```text
Copyright (c) 2013-2026 Ramon Santamaria (@raysan5)

This software is provided "as-is", without any express or implied warranty. In no event
will the authors be held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial
applications, and to alter it and redistribute it freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not claim that you
  wrote the original software. If you use this software in a product, an acknowledgment
  in the product documentation would be appreciated but is not required.

  2. Altered source versions must be plainly marked as such, and must not be misrepresented
  as being the original software.

  3. This notice may not be removed or altered from any source distribution.
```
