# Blessed Packages

Zig++ blesses one way to use a C library from Zig++: fetch the library at a
pinned commit, build it with its own build script, translate its headers with
translate-c, and put a Zig face on the translation. A blessed package is a
package made that way, which Zig++ maintains and tests. The first is raylibz.

## raylibz

[raylibz](https://github.com/mattneel/raylibz) is the Zig++-sanctioned wrapper
for [raylib](https://www.raylib.com), the library for making games by Ramon
Santamaria (raysan). Thank you to raysan and raylib's contributors, who have
given raylib away under the zlib license since 2013. If raylibz gets you a
window, please consider [sponsoring raysan](https://github.com/sponsors/raysan5).

raylibz is a thin, one-to-one wrapper. Every function of `raylib.h` has
raylib's name in Zig case, `InitWindow` is `initWindow`, and raymath's
functions are methods on the vector and matrix types: `Vector2Add(a, b)` is
`a.add(b)`. The C declarations it wraps are raylib's own translation, and they
are always there as `raylibz.c`. raylib's
[cheatsheet](https://www.raylib.com/cheatsheet/cheatsheet.html) is raylibz's
documentation too.

## A window, from nothing

These steps take a machine with nothing on it to a raylib window, on Linux,
macOS or Windows.

1. **Install Zig++.** On Linux and macOS:

   ```sh
   curl -fsSL https://zigpp.lol/ppup | sh
   ```

   On Windows, in PowerShell:

   ```powershell
   irm https://zigpp.lol/ppup.ps1 | iex
   ```

   Then open a new terminal, so that `zig` is on your `PATH`. See
   [Installing](installing.md).

2. **On Linux, install the headers raylib's window layer builds against.** On
   Debian and Ubuntu:

   ```sh
   sudo apt install libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxext-dev libxfixes-dev libgl-dev
   ```

   macOS and Windows need nothing more.

3. **Make a project.**

   ```sh
   mkdir hello-raylib
   cd hello-raylib
   zig init
   ```

   `zig init` writes `build.zig`, `build.zig.zon` and `src/`. The
   `build.zig.zon` it writes pins the Zig++ build that made it, in
   `minimum_zig_version`, so the project keeps building with that build
   whichever `zig` runs it. See [Any Zig Version](any-version.md).

4. **Fetch raylibz, pinned to a commit.** raylibz has no tags. Pick a commit;
   the newest on [main](https://github.com/mattneel/raylibz/commits/main) is a
   good one:

   ```sh
   zig fetch --save https://github.com/mattneel/raylibz/archive/7822b0f143e6320307c4b759c275e8e82e1de90f.tar.gz
   ```

   That records the URL, and the hash of what it downloaded, in
   `build.zig.zon` under the name `raylibz`.

5. **Give your program the module.** In `build.zig`, after the line that
   starts `const exe = b.addExecutable(`, and after the call it opens has
   closed, add:

   ```zig
   const raylibz = b.dependency("raylibz", .{ .target = target, .optimize = optimize });
   exe.root_module.addImport("raylibz", raylibz.module("raylibz"));
   ```

6. **Write the program.** Replace everything in `src/main.zig` with:

   ```zig
   const rl = @import("raylibz");

   pub fn main() void {
       rl.initWindow(800, 450, "hello, raylib");
       defer rl.closeWindow();
       rl.setTargetFPS(60);

       while (!rl.windowShouldClose()) {
           rl.beginDrawing();
           defer rl.endDrawing();
           rl.clearBackground(rl.Color.raywhite);
           rl.drawText("Congrats! You created your first window!", 190, 200, 20, rl.Color.lightgray);
       }
   }
   ```

7. **Run it.**

   ```sh
   zig build run
   ```

   The first build takes a few minutes, because it compiles raylib. A window
   opens; close it, or press Escape.

## What just happened

- `zig build` fetched raylibz and, as raylibz's own dependency, raylib:
  [mattneel/raylib](https://github.com/mattneel/raylib), raysan's raylib with
  its build scripts ported to Zig++'s build API, following raylib's master.
- raylib's own `build.zig` compiled raylib's C with the Clang inside Zig++, and
  chose raylib's window layer: GLFW, on X11 on Linux. Every one of raylib's
  build options passes through raylibz unchanged, so
  `zig build run -Dplatform=rgfw` builds on RGFW instead.
- The same build script ran translate-c over `raylib.h`, `raymath.h` and
  `rlgl.h`. Zig++'s translator reads C headers and writes Zig declarations for
  them, and raylib's build script publishes those as modules.
- raylibz imported the translations and put the Zig face on them:
  `initWindow` takes a string literal as it is, `Color.raywhite` is a Zig
  value, and `loadTexture` returns `error{LoadFailed}!Texture2D`, where C
  returns a texture you must remember to check. raylibz's tests prove the face
  against the translation when it compiles: every struct's layout, every enum
  value, and that no raylib function is missing.
- raylib's own declarations stay one step away:
  `rl.c.InitWindow(800, 450, "hello")` calls the translation directly.

## Your own C library

A library whose build script does not translate its headers gets the same
treatment from your `build.zig`:

```zig
const mylib_h = b.addTranslateC(.{
    .root_source_file = b.path("vendor/mylib/include/mylib.h"),
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("mylib", mylib_h.createModule());
```

`@import("mylib")` is then the translated header. Link the library itself as
its build script produces it, for example with `exe.root_module.linkLibrary`
on the artifact of its package, and wrap what you use. raylibz's
[source](https://github.com/mattneel/raylibz) shows each of its rules at work.
