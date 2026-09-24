#!/bin/sh
#
# Checks an unpacked Zig++ release on the machine it was built for: the
# compiler reports the version it was built as, compiles and runs Zig++ code
# with private fields, rejects code that names a private field from another
# file, computes with f128 both at compile time and at run time, compiles and
# runs C with the Clang and libc it carries, and C++ with the libc++ it builds.
#
#   .github/scripts/smoke.sh <directory of the release> <expected version>
#
# The files it compiles go in a directory under the current one, and the
# compiler gets relative paths to them: a native Windows zig.exe cannot open
# the /tmp paths of the shell that the runner provides.
set -eux

dir=$1
expected=$2

exe=""
case $(uname -s) in
  MINGW* | MSYS* | CYGWIN*) exe=".exe" ;;
esac
zig="$dir/zig$exe"

test "$("$zig" version)" = "$expected"
"$zig" env

work=smoke-test
rm -rf "$work"
mkdir "$work"
trap 'rm -rf "$work"' EXIT

cat >"$work/counter.zig" <<'EOF'
pub const Counter = struct {
    priv count: u32 = 0,

    pub fn bump(counter: *Counter) u32 {
        counter.count += 1;
        return counter.count;
    }
};

test "private fields are visible in their own file" {
    var counter: Counter = .{};
    _ = counter.bump();
    try @import("std").testing.expectEqual(2, counter.bump());
    try @import("std").testing.expectEqual(2, counter.count);
}
EOF
"$zig" test "$work/counter.zig"

cat >"$work/intruder.zig" <<'EOF'
const Counter = @import("counter.zig").Counter;

export fn peek() u32 {
    var counter: Counter = .{};
    return counter.count;
}
EOF
if "$zig" build-obj "$work/intruder.zig" -femit-bin="$work/intruder.o" 2>"$work/intruder.log"; then
  echo "a private field was accessible from another file" >&2
  exit 1
fi
grep -i "private" "$work/intruder.log"

cat >"$work/hello.zig" <<'EOF'
const std = @import("std");

pub fn main() void {
    std.debug.print("hello from Zig++\n", .{});
}
EOF
"$zig" build-exe "$work/hello.zig" -femit-bin="$work/hello$exe"
"$work/hello$exe" 2>&1 | grep "hello from Zig++"

# The compiler evaluates float expressions at compile time in f128, with the
# compiler-rt routines linked into it, and a program calls them at run time.
# Both are calls whose calling convention LLVM decides, so a mismatch between
# LLVM and compiler-rt, as happened with f128 results on Windows in LLVM 23,
# shows up here as wrong numbers.
cat >"$work/float.zig" <<'EOF'
const std = @import("std");

test "f128 at compile time and at run time" {
    const sqrt2 = comptime @sqrt(@as(f128, 2.0));
    try std.testing.expectEqual(0x3fff6a09e667f3bcc908b2fb1366ea95, @as(u128, @bitCast(sqrt2)));

    var a: f128 = 1.5;
    var b: f128 = 2.25;
    std.mem.doNotOptimizeAway(&a);
    std.mem.doNotOptimizeAway(&b);
    try std.testing.expectEqual(3.75, a + b);
    try std.testing.expectEqual(3.375, a * b);
    try std.testing.expectEqual(2.0 / 3.0, a / b);
    try std.testing.expectEqual(sqrt2, @sqrt(a + a - 1.0));
}
EOF
"$zig" test "$work/float.zig"

cat >"$work/hello.c" <<'EOF'
#include <stdio.h>

int main(void) {
    printf("hello from C\n");
    return 0;
}
EOF
"$zig" cc "$work/hello.c" -o "$work/hello_c$exe"
"$work/hello_c$exe" | grep "hello from C"

# libc++ is built from source for the target, and it blocks a thread in
# atomic::wait with each operating system's own primitive: a futex on Linux,
# os_sync_wait_on_address on macOS, WaitOnAddress on Windows.
cat >"$work/wait.cpp" <<'EOF'
#include <atomic>
#include <cstdio>
#include <thread>

int main() {
    std::atomic<int> flag{0};
    std::thread notifier([&] {
        flag.store(1);
        flag.notify_one();
    });
    flag.wait(0);
    notifier.join();
    std::printf("hello from C++: %d\n", flag.load());
    return 0;
}
EOF
"$zig" c++ -std=c++20 "$work/wait.cpp" -o "$work/wait$exe"
"$work/wait$exe" | grep "hello from C++: 1"

echo "smoke test passed for $expected"
