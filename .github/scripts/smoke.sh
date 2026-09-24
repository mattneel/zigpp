#!/bin/sh
#
# Checks an unpacked Zig++ release on the machine it was built for: the
# compiler reports the version it was built as, compiles and runs Zig++ code
# with private fields, rejects code that names a private field from another
# file, and compiles and runs C with the Clang and libc it carries.
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

cat >"$work/hello.c" <<'EOF'
#include <stdio.h>

int main(void) {
    printf("hello from C\n");
    return 0;
}
EOF
"$zig" cc "$work/hello.c" -o "$work/hello_c$exe"
"$work/hello_c$exe" | grep "hello from C"

echo "smoke test passed for $expected"
