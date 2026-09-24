export fn transcendental(x: [*]f32, y: [*]f64, h: [*]f16) callconv(.kernel) void {
    x[0] = @sin(x[1]) + @cos(x[2]) + @tan(x[3]) + @exp(x[4]) + @exp2(x[5]) + @log(x[6]) + @log2(x[7]) + @log10(x[8]);
    y[0] = @sin(y[1]) + @cos(y[2]) + @tan(y[3]) + @exp(y[4]) + @exp2(y[5]) + @log(y[6]) + @log2(y[7]) + @log10(y[8]);
    h[0] = @sin(h[1]) + @exp(h[2]);
}

export fn quad(q: [*]f128, i: [*]i128) callconv(.kernel) void {
    q[0] = @sqrt(q[1] * q[2] / q[3] - q[4]) + @round(q[5]);
    i[0] = @intFromFloat(q[6]);
}

// compile
// output_mode=Obj
// backend=llvm
// target=nvptx64-cuda
// emit_bin=false
// emit_asm=true
