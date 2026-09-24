#!/bin/sh
# Spike pipeline: Zig++ kernels -> AIR -> LLVM 14 bitcode -> .metallib
#
# Run from anywhere; paths are resolved relative to this script. The result is one library
# per invocation containing all three kernels, ready for ./host on macOS.
#
# The stand-in target is nvptx64-cuda because its LLVM address spaces coincide with AIR's
# (.global = 1, .constant = 2, .shared = 3) and it makes Zig emit `ptx_kernel` entry
# functions that `air-rewrite` retypes to Apple's conventions. See doc/proposals/metal.md.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-/tmp/metal-spike/out}
mkdir -p "$out"

# Zig++ with LLVM 23.1.2. ZIG_LIB_DIR must be the lib tree the binary was built from:
# pairing this compiler with master's lib makes std.lang mismatch and mangles diagnostics.
ZIG=${ZIG:-/tmp/zig-amdgpu-final/bin/zig}
ZIG_LIB_DIR=${ZIG_LIB_DIR:-/home/autark/src/zig/zig-amdgpu/lib}
export ZIG_LIB_DIR

AIR_REWRITE=${AIR_REWRITE:-/tmp/metal-spike/air-rewrite-build/air-rewrite}
DOWNGRADE=${DOWNGRADE:-/tmp/metal-spike/downgrade/build/llvm-downgrade}
METALLIB=${METALLIB:-/tmp/metal-spike/metallib}

echo "== 1. kernels -> modern LLVM bitcode (stand-in target)"
"$ZIG" build-obj "$here/kernels.zig" -target nvptx64-cuda -mcpu=sm_52 \
    -fno-compiler-rt -OReleaseFast -fno-emit-bin -femit-llvm-bc="$out/kernels.bc"

echo "== 2. modern bitcode -> AIR conventions"
"$AIR_REWRITE" "$out/kernels.bc" -o "$out/kernels.air.bc" --spec "$here/kernels.spec" \
    --air 2.8 --metal 4.0 --deploy 26.0.0 --sdk 26.5.0 --opt O3

echo "== 3. AIR -> LLVM 14 bitcode (what Apple's reader wants)"
"$DOWNGRADE" "$out/kernels.air.bc" --bitcode-version=14.0 -o "$out/kernels.air14.bc"

echo "== 4. bitcode -> .metallib (one library, one function group per kernel)"
"$METALLIB" write --bitcode "$out/kernels.air14.bc" --name vadd --name reduce --name parsef \
    --air 2.8 --metal 4.0 --format 1.2.9 --platform 26.0.0 --uuid -o "$out/kernels.metallib"

echo "== done: $out/kernels.metallib"
ls -l "$out"
