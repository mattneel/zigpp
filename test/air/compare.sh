#!/bin/sh
# Compares the AIR of the task forms as a program spells them with the AIR of
# §3.1's expansions written out by hand, instruction for instruction.
#
#   test/air/compare.sh <zig> <zig-lib-dir>
#
# `<zig>` must be a compiler built with `-Ddebug-extensions`, because
# `--verbose-air` is gated on it. The comparison ignores the compiler's own
# names for values (`%12`) and the coordinates of `dbg_stmt`, which differ
# between two files by construction; everything else must match, so a
# difference in an instruction, an operand, an order or a `dbg_stmt` count
# fails. The two files must define the same functions.
set -eu

zig=$1
lib=$2
dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/async_await
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# The functions both files must define, in a fixed order.
functions='awaitBinding concurrentBinding cancelBinding voidBinding test.the task forms'

dump() {
    file=$1
    base=$(basename "$file" .zig)
    ZIG_LIB_DIR=$lib "$zig" test "$file" -fno-emit-bin --verbose-air \
        --cache-dir "$tmp/cache-$base" >"$tmp/$base.raw" 2>&1 || true
    # The dump goes to stderr; a compile error means there is nothing to compare.
    if ! grep -q '^# Begin Function AIR' "$tmp/$base.raw"; then
        echo "no AIR was dumped for $file" >&2
        head -20 "$tmp/$base.raw" >&2
        exit 1
    fi
    for f in $functions; do
        awk -v want="$base.$f" -v base="$base" '
            /^# Begin Function AIR: / {
                name = $5
                sub(/:$/, "", name)
                gsub(/\//, ".", name)
                name = substr(name, length(name) - length(want) + 1)
                printing = (name == want)
                if (printing) print "# " f
                next
            }
            /^# End Function AIR: / { printing = 0; next }
            /^# (Total AIR|AIR |Liveness )/ { next }
            {
                if (!printing) next
                gsub(/%[0-9]+/, "%")
                gsub(/dbg_stmt\([0-9]+:[0-9]+\)/, "dbg_stmt()")
                print
            }
        ' "$tmp/$base.raw"
    done
}

dump "$dir/sugar.zig" >"$tmp/sugar.air"
dump "$dir/hand.zig" >"$tmp/hand.air"

# A comparison of nothing is not a comparison: every function must be there.
for f in $functions; do
    for base in sugar hand; do
        if ! grep -q "^# $f$" "$tmp/$base.air"; then
            echo "$base.air has no AIR for $f" >&2
            exit 1
        fi
    done
done
echo "compared $(grep -c '^# ' "$tmp/sugar.air") functions, $(grep -c '^  *%' "$tmp/sugar.air") instructions"

if diff -u "$tmp/hand.air" "$tmp/sugar.air"; then
    echo "the AIR of the sugared forms equals the hand-written expansion"
else
    echo "the sugared forms do not compile to the hand-written expansion" >&2
    exit 1
fi
