//! The kernels of the `std.gpu` end-to-end test, ported from the examples of the `ugpu` project,
//! plus kernels that cover the parts of `std.gpu` and the standard library that the port to
//! `std.gpu` adds.
//!
//! `main.zig` compiles this file to PTX assembly twice, with `.debug` and with `.fast`, and runs
//! every kernel of both images on a GPU through `std.gpu.cuda`. The host compares the results
//! against values computed here in the test process.

const std = @import("std");
const gpu = std.gpu;
const allocators = gpu.allocators;

/// Threads in a block of the kernels that reduce over the whole block, or that use a shared array
/// with one element per thread. `main.zig` launches these kernels with that many threads.
const block_size = 256;

/// Threads in a tile of a 2D kernel: the tile of a matrix, an image, or a transposition.
const tile_size = 16;

/// Warp size, the number of threads that the shuffle and vote functions of `std.gpu` cover.
const warp_size = gpu.warp_size;

/// Bytes of the shared heaps of the kernels that use an allocator over shared memory.
const shared_heap_size = 32 * 1024;

/// The shared heap of the kernels that keep a hash map: the map itself, and the keys of the
/// string map.
var shared_heap: [shared_heap_size]u8 addrspace(.shared) = undefined;

/// The shared heap and the per-thread heaps of the kernels that parse JSON.
const json_threads = 4;
const json_heap_size = 8 * 1024;
var json_heaps: [json_threads][json_heap_size]u8 addrspace(.shared) = undefined;

/// The bytes of a buffer in shared memory, as a slice in the generic address space that the
/// allocators and containers of the standard library take.
fn sharedBytes(buffer: anytype) []u8 {
    return @as([*]u8, @ptrCast(@addrSpaceCast(buffer)))[0..buffer.len];
}

/// The heap of the thread `tid` of a JSON kernel.
fn jsonHeap(tid: u32) []u8 {
    return sharedBytes(&json_heaps[tid]);
}

// ---------------------------------------------------------------------------------------------
// examples/vector_add.zig: c[i] = a[i] + b[i]
// ---------------------------------------------------------------------------------------------

export fn vectorAdd(a: [*]const f32, b: [*]const f32, c: [*]f32, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;
    c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------------------------
// examples/reduce.zig: block-wide reductions in shared memory
// ---------------------------------------------------------------------------------------------

var sum_reduce_shared: [block_size]f32 addrspace(.shared) = undefined;

/// Sums the elements of every block of `input` into the element of `output` with the index of the
/// block. A block's result is the sum of the elements of that block, so `output` has as many
/// elements as the grid has blocks.
export fn sumReduce(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const tid = gpu.threadIdx(.x);
    const i = gpu.blockIdx(.x) * block_size + tid;

    sum_reduce_shared[tid] = if (i < n) input[i] else 0;
    gpu.syncThreads();

    var stride: u32 = block_size / 2;
    while (stride > 0) : (stride /= 2) {
        if (tid < stride) sum_reduce_shared[tid] += sum_reduce_shared[tid + stride];
        gpu.syncThreads();
    }

    if (tid == 0) output[gpu.blockIdx(.x)] = sum_reduce_shared[0];
}

var max_reduce_shared: [block_size]f32 addrspace(.shared) = undefined;

/// The largest element of every block of `input`, one result per block, like `sumReduce`. The
/// elements past the end of the input are the smallest `f32`, so that they cannot be the largest.
export fn maxReduce(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const tid = gpu.threadIdx(.x);
    const i = gpu.blockIdx(.x) * block_size + tid;

    max_reduce_shared[tid] = if (i < n) input[i] else -std.math.floatMax(f32);
    gpu.syncThreads();

    var stride: u32 = block_size / 2;
    while (stride > 0) : (stride /= 2) {
        if (tid < stride) {
            const left = max_reduce_shared[tid];
            const right = max_reduce_shared[tid + stride];
            max_reduce_shared[tid] = if (left > right) left else right;
        }
        gpu.syncThreads();
    }

    if (tid == 0) output[gpu.blockIdx(.x)] = max_reduce_shared[0];
}

// ---------------------------------------------------------------------------------------------
// examples/histogram.zig: atomic operations, and one histogram per block in shared memory
// ---------------------------------------------------------------------------------------------

/// Counts the occurrences of every value of `data` that is below `num_bins`, and adds them to
/// `bins`, which holds `num_bins` counters. The kernel adds to every element of `bins`, so the
/// host must fill it with zeros first, and may read it only after the launch.
export fn histogram(data: [*]const u32, bins: [*]u32, n: u32, num_bins: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;

    const value = data[i];
    if (value < num_bins) _ = gpu.atomicAdd(&bins[value], 1);
}

/// One histogram bin per element of the shared array, which is what a block needs for the
/// histogram of its own elements: the counters in shared memory take the atomic traffic of the
/// block, and each block adds its counters to `bins` once. `num_bins` must be at most the number
/// of threads in a block.
var histogram_shared_bins: [block_size]u32 addrspace(.shared) = undefined;

/// Like `histogram`, with the counters of a block counted in shared memory and added to `bins`
/// once per block.
export fn histogramShared(data: [*]const u32, bins: [*]u32, n: u32, num_bins: u32) callconv(.kernel) void {
    const tid = gpu.threadIdx(.x);
    const i = gpu.blockIdx(.x) * block_size + tid;

    if (tid < num_bins) histogram_shared_bins[tid] = 0;
    gpu.syncThreads();

    if (i < n) {
        const value = data[i];
        if (value < num_bins) _ = gpu.atomicAdd(&histogram_shared_bins[value], 1);
    }
    gpu.syncThreads();

    if (tid < num_bins) {
        const count = histogram_shared_bins[tid];
        if (count > 0) _ = gpu.atomicAdd(&bins[tid], count);
    }
}

/// Counts the pairs of `x` and `y` values: `bins[y * num_bins_x + x]` counts the pairs with
/// those coordinates, and pairs with a coordinate at or above its number of bins are left out.
export fn histogram2D(
    x: [*]const u32,
    y: [*]const u32,
    bins: [*]u32,
    n: u32,
    num_bins_x: u32,
    num_bins_y: u32,
) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;

    const x_value = x[i];
    const y_value = y[i];
    if (x_value < num_bins_x and y_value < num_bins_y) {
        _ = gpu.atomicAdd(&bins[y_value * num_bins_x + x_value], 1);
    }
}

// ---------------------------------------------------------------------------------------------
// examples/warp.zig: warp reductions, votes, and shuffles
// ---------------------------------------------------------------------------------------------

/// The index of the warp that the calling thread belongs to, in the whole grid.
inline fn warpId() u32 {
    return (gpu.blockIdx(.x) * gpu.blockDim(.x) + gpu.threadIdx(.x)) / warp_size;
}

/// The element of `input` with the index `i`, or zero when `i` is past the end of the input.
///
/// The warp functions of `std.gpu` operate on the whole warp, and reading the value of a thread
/// that has exited is undefined, so the kernels below let every thread of the block take part,
/// and threads past the end read zero instead of returning early. The results of a partial warp
/// are then the sum, vote, or shuffle of the elements that exist and of the zeros.
inline fn warpLoad(input: [*]const u32, i: u32, n: u32) u32 {
    return if (i < n) input[i] else 0;
}

/// Writes one sum per warp, taken with `std.gpu.warpReduceSum`, so `output` has one element for
/// every warp of the grid.
export fn warpSumKernel(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const sum = gpu.warpReduceSum(warpLoad(input, gpu.globalId(.x), n));
    if (gpu.laneId() == 0) output[warpId()] = sum;
}

/// Writes one maximum per warp, taken with `std.gpu.warpReduceMax`.
export fn warpMaxKernel(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const maximum = gpu.warpReduceMax(warpLoad(input, gpu.globalId(.x), n));
    if (gpu.laneId() == 0) output[warpId()] = maximum;
}

/// Writes one minimum per warp, taken with `std.gpu.warpReduceMin`.
export fn warpMinKernel(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const minimum = gpu.warpReduceMin(warpLoad(input, gpu.globalId(.x), n));
    if (gpu.laneId() == 0) output[warpId()] = minimum;
}

/// Writes the mask of the threads of every warp whose element is above 100, with the bit of a
/// lane set when its element is above 100: one mask per warp, in `output`.
export fn ballotKernel(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const mask = gpu.ballot(warpLoad(input, gpu.globalId(.x), n) > 100);
    if (gpu.laneId() == 0) output[warpId()] = mask;
}

/// Writes one flag per warp: whether every thread of the warp agrees on the test `value > 50`,
/// with `std.gpu.uniform`.
export fn checkDivergence(values: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const agree = gpu.uniform(warpLoad(values, gpu.globalId(.x), n) > 50);
    if (gpu.laneId() == 0) output[warpId()] = @intFromBool(agree);
}

/// Writes the value of lane 0 of the warp into every element of `output`.
export fn shuffleBroadcastKernel(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    const broadcast = gpu.shflBroadcast(warpLoad(input, i, n), 0);
    if (i < n) output[i] = broadcast;
}

/// Writes the results of the shuffles by one lane, one array after the other in `output`:
/// `output[i]`, `output[n + i]`, and `output[2 * n + i]` hold the values that `std.gpu.shflDown`,
/// `shflUp`, and `shflXor` return for the element of every thread. Threads whose source lane is
/// past the end of the warp, or before its start, keep their own value.
export fn shuffleKernel(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    const value = warpLoad(input, i, n);
    const down_value = gpu.shflDown(value, 1);
    const up_value = gpu.shflUp(value, 1);
    const xor_value = gpu.shflXor(value, 1);
    if (i < n) {
        output[i] = down_value;
        output[n + i] = up_value;
        output[2 * n + i] = xor_value;
    }
}

// ---------------------------------------------------------------------------------------------
// examples/matrix_mul.zig: c = a * b
// ---------------------------------------------------------------------------------------------

/// `c[row][col]` is the dot product of row `row` of `a` and column `col` of `b`, with every
/// thread computing one element.
export fn matrixMulNaive(
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    m: u32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const row = gpu.blockIdx(.y) * gpu.blockDim(.y) + gpu.threadIdx(.y);
    const col = gpu.blockIdx(.x) * gpu.blockDim(.x) + gpu.threadIdx(.x);
    if (row >= m or col >= n) return;

    var sum: f32 = 0;
    var i: u32 = 0;
    while (i < k) : (i += 1) sum += a[row * k + i] * b[i * n + col];
    c[row * n + col] = sum;
}

var matrix_mul_tile_a: [tile_size][tile_size]f32 addrspace(.shared) = undefined;
var matrix_mul_tile_b: [tile_size][tile_size]f32 addrspace(.shared) = undefined;

/// `c = a * b` with the tiles of the two matrices in shared memory, one tile of each per step of
/// the dot products. A block must have one thread for every element of a tile, so the block is
/// `tile_size` by `tile_size` threads.
export fn matrixMulTiled(
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    m: u32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const tx = gpu.threadIdx(.x);
    const ty = gpu.threadIdx(.y);
    const row = gpu.blockIdx(.y) * tile_size + ty;

    var sum: f32 = 0;
    var tile: u32 = 0;
    const tile_count = (k + tile_size - 1) / tile_size;
    while (tile < tile_count) : (tile += 1) {
        const col = gpu.blockIdx(.x) * tile_size + tx;
        const a_col = tile * tile_size + tx;
        matrix_mul_tile_a[ty][tx] = if (row < m and a_col < k) a[row * k + a_col] else 0;

        const b_row = tile * tile_size + ty;
        matrix_mul_tile_b[ty][tx] = if (b_row < k and col < n) b[b_row * n + col] else 0;
        gpu.syncThreads();

        var i: u32 = 0;
        while (i < tile_size) : (i += 1) {
            sum += matrix_mul_tile_a[ty][i] * matrix_mul_tile_b[i][tx];
        }
        gpu.syncThreads();
    }

    const col = gpu.blockIdx(.x) * tile_size + tx;
    if (row < m and col < n) c[row * n + col] = sum;
}

const large_tile_size = 32;
var matrix_mul_large_tile_a: [large_tile_size][large_tile_size]f32 addrspace(.shared) = undefined;
var matrix_mul_large_tile_b: [large_tile_size][large_tile_size]f32 addrspace(.shared) = undefined;

/// Fills the shared tiles of one step of the dot products of `matrixMulLargeTile`: the 32 by 32
/// elements of `a` in the rows of the block, and the ones of `b` in the columns of the block,
/// with zero where the matrices end. Every thread of the block loads its share.
fn loadLargeTiles(a: [*]const f32, b: [*]const f32, m: u32, n: u32, k: u32, tile: u32) void {
    const row_start = gpu.blockIdx(.y) * large_tile_size;
    const col_start = gpu.blockIdx(.x) * large_tile_size;
    const threads = gpu.blockDim(.x) * gpu.blockDim(.y);

    var index = gpu.threadIdx(.y) * gpu.blockDim(.x) + gpu.threadIdx(.x);
    while (index < large_tile_size * large_tile_size) : (index += threads) {
        const tile_y = index / large_tile_size;
        const tile_x = index % large_tile_size;

        const a_row = row_start + tile_y;
        const a_col = tile * large_tile_size + tile_x;
        matrix_mul_large_tile_a[tile_y][tile_x] = if (a_row < m and a_col < k) a[a_row * k + a_col] else 0;

        const b_row = tile * large_tile_size + tile_y;
        const b_col = col_start + tile_x;
        matrix_mul_large_tile_b[tile_y][tile_x] = if (b_row < k and b_col < n) b[b_row * n + b_col] else 0;
    }
}

/// `c = a * b` with tiles of 32 by 32 elements. A block has one column of threads for every
/// column of a tile, and as many rows as the launch gives it: every thread computes the elements
/// of the rows `ty`, `ty + blockDim.y`, and so on, of its tile. A block of 32 by 32 threads is
/// what the elements of a tile need, but it would not run when this kernel is compiled with
/// `.debug`, whose code needs more registers than a block of 1024 threads has, so the kernel
/// takes the height of its block as the launch says.
export fn matrixMulLargeTile(
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    m: u32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    const tx = gpu.threadIdx(.x);
    const ty = gpu.threadIdx(.y);
    const block_rows = gpu.blockDim(.y);
    const row_start = gpu.blockIdx(.y) * large_tile_size;
    const col = gpu.blockIdx(.x) * large_tile_size + tx;
    const row_steps = (large_tile_size + block_rows - 1) / block_rows;
    const tile_count = (k + large_tile_size - 1) / large_tile_size;

    // Every thread of the block steps through the same number of rows, so that all of them reach
    // the barriers below.
    var step: u32 = 0;
    while (step < row_steps) : (step += 1) {
        const row = row_start + step * block_rows + ty;

        var sum: f32 = 0;
        var tile: u32 = 0;
        while (tile < tile_count) : (tile += 1) {
            loadLargeTiles(a, b, m, n, k, tile);
            gpu.syncThreads();

            if (ty < large_tile_size) {
                var i: u32 = 0;
                while (i < large_tile_size) : (i += 1) {
                    sum += matrix_mul_large_tile_a[ty][i] * matrix_mul_large_tile_b[i][tx];
                }
            }
            gpu.syncThreads();
        }

        if (row < m and col < n) c[row * n + col] = sum;
    }
}

// ---------------------------------------------------------------------------------------------
// examples/convolution.zig: a convolution with a halo, a transposition, and a prefix sum
// ---------------------------------------------------------------------------------------------

const convolution_tile_size = 16;

/// The elements of the filter and the halo: a filter of up to 5 by 5 elements fits around a tile
/// of 16 by 16 output elements.
const convolution_halo = 2;
const convolution_extent = convolution_tile_size + 2 * convolution_halo;

var convolution_tile: [convolution_extent][convolution_extent]f32 addrspace(.shared) = undefined;

/// Loads the tile of `input` that the block `bx`, `by` convolves, and its halo, into
/// `convolution_tile`. Every thread of the block loads its share, and the elements outside the
/// input are zero.
fn loadConvolutionTile(input: [*]const f32, width: u32, height: u32, bx: u32, by: u32) void {
    const threads = gpu.blockDim(.x) * gpu.blockDim(.y);
    const width_signed: i32 = @intCast(width);
    const height_signed: i32 = @intCast(height);

    var index = gpu.threadIdx(.y) * gpu.blockDim(.x) + gpu.threadIdx(.x);
    while (index < convolution_extent * convolution_extent) : (index += threads) {
        const halo_x = index % convolution_extent;
        const halo_y = index / convolution_extent;
        const x = @as(i32, @intCast(bx * convolution_tile_size + halo_x)) - convolution_halo;
        const y = @as(i32, @intCast(by * convolution_tile_size + halo_y)) - convolution_halo;
        convolution_tile[halo_y][halo_x] = if (x >= 0 and x < width_signed and y >= 0 and y < height_signed)
            input[@as(u32, @intCast(y)) * width + @as(u32, @intCast(x))]
        else
            0;
    }
}

/// Convolves the image `input`, of `width` by `height` elements, with the `filter_size` by
/// `filter_size` elements of `filter`, which is at most 5 by 5, at every element of `output`.
/// Elements outside the image are zero. A block convolves a tile of `convolution_tile_size` by
/// `convolution_tile_size` elements.
export fn convolution2D(
    input: [*]const f32,
    output: [*]f32,
    filter: [*]const f32,
    width: u32,
    height: u32,
    filter_size: u32,
) callconv(.kernel) void {
    const out_x = gpu.blockIdx(.x) * convolution_tile_size + gpu.threadIdx(.x);
    const out_y = gpu.blockIdx(.y) * convolution_tile_size + gpu.threadIdx(.y);

    loadConvolutionTile(input, width, height, gpu.blockIdx(.x), gpu.blockIdx(.y));
    gpu.syncThreads();

    if (out_x >= width or out_y >= height) return;

    const tx = gpu.threadIdx(.x);
    const ty = gpu.threadIdx(.y);
    var sum: f32 = 0;
    var fy: u32 = 0;
    while (fy < filter_size) : (fy += 1) {
        var fx: u32 = 0;
        while (fx < filter_size) : (fx += 1) {
            sum += convolution_tile[ty + fy][tx + fx] * filter[fy * filter_size + fx];
        }
    }
    output[out_y * width + out_x] = sum;
}

var transpose_tile: [tile_size][tile_size + 1]f32 addrspace(.shared) = undefined;

/// Writes the transposition of the `width` by `height` elements of `input` into `output`: the
/// element of the row `y` and the column `x` of the input becomes the element of the row `x` and
/// the column `y` of the output. A block transposes a tile of `tile_size` by `tile_size`
/// elements, and the shared tile has one column more than a tile so that the reads and the writes
/// of a block do not conflict in the banks of the shared memory.
export fn transpose(input: [*]const f32, output: [*]f32, width: u32, height: u32) callconv(.kernel) void {
    const in_x = gpu.blockIdx(.x) * tile_size + gpu.threadIdx(.x);
    const in_y = gpu.blockIdx(.y) * tile_size + gpu.threadIdx(.y);

    transpose_tile[gpu.threadIdx(.y)][gpu.threadIdx(.x)] = if (in_x < width and in_y < height)
        input[in_y * width + in_x]
    else
        0;
    gpu.syncThreads();

    const out_x = gpu.blockIdx(.y) * tile_size + gpu.threadIdx(.x);
    const out_y = gpu.blockIdx(.x) * tile_size + gpu.threadIdx(.y);
    if (out_x < height and out_y < width) {
        output[out_y * height + out_x] = transpose_tile[gpu.threadIdx(.x)][gpu.threadIdx(.y)];
    }
}

var prefix_sum_shared: [block_size]f32 addrspace(.shared) = undefined;

/// The exclusive prefix sums of every block of `input`, in `output`: `output[i]` is the sum of
/// the elements of the block of `i` that are before `i`, and `input[0]` and `input[n]` give the
/// sums of the whole array. `output[i]` for `i` at or past the end of `input` is the sum of the
/// whole block of `i`.
///
/// The scan of a block is the work-efficient one of Blelloch: the block adds the elements of the
/// array up in a tree, and pushes the sums back down the tree.
export fn prefixSum(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const tid = gpu.threadIdx(.x);
    const i = gpu.blockIdx(.x) * block_size + tid;

    prefix_sum_shared[tid] = if (i < n) input[i] else 0;
    gpu.syncThreads();

    // Up-sweep: every element that is the root of a pair at this distance takes the sum of the
    // pair.
    var stride: u32 = 1;
    while (stride < block_size) : (stride *= 2) {
        if ((tid + 1) % (2 * stride) == 0) {
            prefix_sum_shared[tid] += prefix_sum_shared[tid - stride];
        }
        gpu.syncThreads();
    }

    // The last element holds the sum of the whole block; the down-sweep starts from zero there,
    // so that the tree holds the sums of the elements before each position.
    if (tid == block_size - 1) prefix_sum_shared[tid] = 0;
    gpu.syncThreads();

    // Down-sweep: the root of a pair takes the sum that reaches it, before it adds it to the
    // other element of the pair.
    stride = block_size / 2;
    while (stride > 0) : (stride /= 2) {
        if ((tid + 1) % (2 * stride) == stride) {
            const left = prefix_sum_shared[tid];
            prefix_sum_shared[tid] = prefix_sum_shared[tid + stride];
            prefix_sum_shared[tid + stride] += left;
        }
        gpu.syncThreads();
    }

    if (i < n) output[i] = prefix_sum_shared[tid];
}

// ---------------------------------------------------------------------------------------------
// examples/stencil.zig: 1D and 2D stencils
// ---------------------------------------------------------------------------------------------

/// `output[i]` is the weighted sum of `input[i - 1]`, `input[i]`, and `input[i + 1]`, and the
/// elements at the ends of the input are copied to the output.
export fn stencil1D(
    input: [*]const f32,
    output: [*]f32,
    n: u32,
    alpha: f32,
    beta: f32,
    gamma: f32,
) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;
    if (i == 0 or i >= n - 1) {
        output[i] = input[i];
        return;
    }
    output[i] = alpha * input[i - 1] + beta * input[i] + gamma * input[i + 1];
}

const stencil_halo = 1;
const stencil_extent = tile_size + 2 * stencil_halo;

var stencil_tile: [stencil_extent][stencil_extent]f32 addrspace(.shared) = undefined;

/// Loads the tile of `input` that the block `bx`, `by` reads, and its halo of one element, into
/// `stencil_tile`. Every thread of the block loads its share, and the elements outside the input
/// are zero.
fn loadStencilTile(input: [*]const f32, width: u32, height: u32, bx: u32, by: u32) void {
    const threads = gpu.blockDim(.x) * gpu.blockDim(.y);
    const width_signed: i32 = @intCast(width);
    const height_signed: i32 = @intCast(height);

    var index = gpu.threadIdx(.y) * gpu.blockDim(.x) + gpu.threadIdx(.x);
    while (index < stencil_extent * stencil_extent) : (index += threads) {
        const halo_x = index % stencil_extent;
        const halo_y = index / stencil_extent;
        const x = @as(i32, @intCast(bx * tile_size + halo_x)) - stencil_halo;
        const y = @as(i32, @intCast(by * tile_size + halo_y)) - stencil_halo;
        stencil_tile[halo_y][halo_x] = if (x >= 0 and x < width_signed and y >= 0 and y < height_signed)
            input[@as(u32, @intCast(y)) * width + @as(u32, @intCast(x))]
        else
            0;
    }
}

/// The 5-point Laplace operator of the image `input`, of `width` by `height` elements, at every
/// element of `output`, and the elements at the edges of the image are copied to the output. A
/// block covers a tile of `tile_size` by `tile_size` elements, and the tile of a block, with its
/// halo of one element, is in shared memory.
export fn stencil2DLaplace(input: [*]const f32, output: [*]f32, width: u32, height: u32) callconv(.kernel) void {
    const out_x = gpu.blockIdx(.x) * tile_size + gpu.threadIdx(.x);
    const out_y = gpu.blockIdx(.y) * tile_size + gpu.threadIdx(.y);

    loadStencilTile(input, width, height, gpu.blockIdx(.x), gpu.blockIdx(.y));
    gpu.syncThreads();

    if (out_x >= width or out_y >= height) return;

    const tx = gpu.threadIdx(.x) + stencil_halo;
    const ty = gpu.threadIdx(.y) + stencil_halo;
    if (out_x > 0 and out_x < width - 1 and out_y > 0 and out_y < height - 1) {
        const center = stencil_tile[ty][tx];
        const left = stencil_tile[ty][tx - 1];
        const right = stencil_tile[ty][tx + 1];
        const top = stencil_tile[ty - 1][tx];
        const bottom = stencil_tile[ty + 1][tx];
        output[out_y * width + out_x] = left + right + top + bottom - 4 * center;
    } else {
        output[out_y * width + out_x] = input[out_y * width + out_x];
    }
}

/// The 9-point stencil of the image `input`, of `width` by `height` elements, at every element of
/// `output`, and the elements at the edges of the image are copied to the output. The kernel
/// shares the tile in shared memory with `stencil2DLaplace`.
export fn stencil2D9Point(input: [*]const f32, output: [*]f32, width: u32, height: u32) callconv(.kernel) void {
    const out_x = gpu.blockIdx(.x) * tile_size + gpu.threadIdx(.x);
    const out_y = gpu.blockIdx(.y) * tile_size + gpu.threadIdx(.y);

    loadStencilTile(input, width, height, gpu.blockIdx(.x), gpu.blockIdx(.y));
    gpu.syncThreads();

    if (out_x >= width or out_y >= height) return;

    const tx = gpu.threadIdx(.x) + stencil_halo;
    const ty = gpu.threadIdx(.y) + stencil_halo;
    if (out_x > 0 and out_x < width - 1 and out_y > 0 and out_y < height - 1) {
        const center = stencil_tile[ty][tx];
        const north = stencil_tile[ty - 1][tx];
        const south = stencil_tile[ty + 1][tx];
        const east = stencil_tile[ty][tx + 1];
        const west = stencil_tile[ty][tx - 1];
        const north_east = stencil_tile[ty - 1][tx + 1];
        const north_west = stencil_tile[ty - 1][tx - 1];
        const south_east = stencil_tile[ty + 1][tx + 1];
        const south_west = stencil_tile[ty + 1][tx - 1];
        output[out_y * width + out_x] = 0.25 * (north_east + north_west + south_east + south_west) +
            0.5 * (north + south + east + west) -
            3 * center;
    } else {
        output[out_y * width + out_x] = input[out_y * width + out_x];
    }
}

// ---------------------------------------------------------------------------------------------
// examples/stdlib.zig: the standard library on the device
// ---------------------------------------------------------------------------------------------

/// `output[i]` is the square root of `input[i]`, plus its magnitude, plus the element clamped to
/// the range of -1 to 1.
export fn mathOps(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= n) return;

    const x = input[gid];
    const root = std.math.sqrt(x);
    const magnitude = @abs(x);
    const clamped = @min(@max(x, -1), 1);
    output[gid] = root + magnitude + clamped;
}

/// `output[i]` is the number of bits that are set in `input[i]`, plus the number of the leading
/// zeros, plus the number of the trailing zeros.
export fn bitOps(input: [*]const u32, output: [*]u32, n: u32) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= n) return;

    const x = input[gid];
    output[gid] = @as(u32, @popCount(x)) + @as(u32, @clz(x)) + @as(u32, @ctz(x));
}

/// `output[i]` is `a[i] + b[i] + a[i] * b[i]` for four elements at a time, loaded and stored as
/// vectors.
export fn vectorOps(a: [*]const f32, b: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid * 4 >= n) return;

    const index = gid * 4;
    const va: @Vector(4, f32) = .{ a[index], a[index + 1], a[index + 2], a[index + 3] };
    const vb: @Vector(4, f32) = .{ b[index], b[index + 1], b[index + 2], b[index + 3] };
    const result = va + vb + va * vb;
    output[index] = result[0];
    output[index + 1] = result[1];
    output[index + 2] = result[2];
    output[index + 3] = result[3];
}

/// The larger of two values of a type that the call instantiates, plus the smaller.
fn bounds(comptime T: type, x: T, y: T) T {
    return @max(x, y) + @min(x, y);
}

/// `output[i]` is `2 * input[i]`, at least 1.
export fn comptimeOps(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= n) return;

    const scale: f32 = 2;
    const scaled = input[gid] * scale;
    output[gid] = bounds(f32, scaled, 1);
}

// ---------------------------------------------------------------------------------------------
// examples/hashmap.zig: hash maps over a bump allocator in shared memory
// ---------------------------------------------------------------------------------------------

/// Builds a map of the keys and values of every block and looks every key up in it, which is
/// `results` when the key is in the map. Thread 0 does the work of the block; the map holds the
/// pairs of the block, so a key that appears twice in a block keeps the last value.
export fn hashMapKernel(
    keys: [*]const u32,
    values: [*]const u32,
    n: u32,
    results: [*]u32,
) callconv(.kernel) void {
    const threads = gpu.blockDim(.x);
    const block_start = gpu.blockIdx(.x) * threads;

    // Every thread of the block initializes the allocator, because `init` waits at a barrier;
    // only thread 0 then allocates from it.
    var bump = allocators.BumpAllocator(shared_heap_size).init(&shared_heap);
    if (gpu.threadIdx(.x) != 0) return;

    var map = std.AutoHashMap(u32, u32).init(bump.allocator());
    defer map.deinit();

    var i: u32 = 0;
    while (i < threads and block_start + i < n) : (i += 1) {
        map.put(keys[block_start + i], values[block_start + i]) catch break;
    }

    gpu.print("Block {d}: HashMap has {d} entries\n", .{ gpu.blockIdx(.x), map.count() });

    i = 0;
    while (i < threads and block_start + i < n) : (i += 1) {
        results[block_start + i] = map.get(keys[block_start + i]) orelse 0xdead_beef;
    }
}

/// Inserts the pairs `i` and `i * i` for `i` below `n` until the shared heap refuses a growth of
/// the map, reports how many pairs the map took in `inserted`, and counts the keys below that
/// whose value is wrong in `wrong`. The test passes more pairs than the heap holds, so the map
/// must keep working after the allocator refuses to let it grow.
export fn hashMapStressKernel(n: u32, inserted: *u32, wrong: *u32) callconv(.kernel) void {
    var bump = allocators.BumpAllocator(shared_heap_size).init(&shared_heap);
    if (gpu.globalId(.x) != 0) return;

    var map = std.AutoHashMap(u32, u32).init(bump.allocator());
    defer map.deinit();

    var count: u32 = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        map.put(i, i * i) catch break;
        count += 1;
    }
    inserted.* = count;

    var mismatches: u32 = 0;
    i = 0;
    while (i < count) : (i += 1) {
        if (map.get(i) != i * i) mismatches += 1;
    }
    wrong.* = mismatches;

    gpu.print("Stress test: inserted {d} entries\n", .{count});
}

/// The keys of the string map: one buffer of the shared heap for every thread, in which the
/// thread formats its key.
var string_keys: [block_size][64]u8 addrspace(.shared) = undefined;

/// Builds a map of the string keys that thread 0 formats, `key_0` to `key_<count - 1>`, to the
/// values of `values`, and looks every key up in it, which is `results`. `n` must be at most the
/// number of threads in a block.
export fn stringHashMapKernel(values: [*]const u32, n: u32, results: [*]u32) callconv(.kernel) void {
    // See `hashMapKernel`: the allocator is block-wide, so every thread initializes it.
    var bump = allocators.BumpAllocator(shared_heap_size).init(&shared_heap);
    if (gpu.threadIdx(.x) != 0) return;

    var map = std.StringHashMap(u32).init(bump.allocator());
    defer map.deinit();

    const count = @min(n, block_size);
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        // The keys live in the shared buffers, so they stay alive for as long as the map does.
        var fba = std.heap.FixedBufferAllocator.init(sharedBytes(&string_keys[i]));
        const key = std.fmt.allocPrint(fba.allocator(), "key_{d}", .{i}) catch break;
        map.put(key, values[i]) catch break;
    }

    gpu.print("StringHashMap: {d} string keys inserted\n", .{map.count()});

    i = 0;
    while (i < count) : (i += 1) {
        var fba = std.heap.FixedBufferAllocator.init(sharedBytes(&string_keys[i]));
        const key = std.fmt.allocPrint(fba.allocator(), "key_{d}", .{i}) catch break;
        results[i] = map.get(key) orelse 0xbad_bad;
    }
}

// ---------------------------------------------------------------------------------------------
// examples/base64.zig: the standard library's base64 codec
// ---------------------------------------------------------------------------------------------

/// Encodes every `chunk_size` elements of `input` on its own, and writes the text of the chunk of
/// every thread to `output` at the position of its chunk, and the length of that text to
/// `lengths`. `chunk_size` must be a multiple of 3, so that the chunks of text do not overlap.
export fn base64EncodeKernel(
    input: [*]const u8,
    output: [*]u8,
    lengths: [*]u32,
    input_len: u32,
    chunk_size: u32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    const start = gid * chunk_size;
    if (start >= input_len) return;
    const end = @min(start + chunk_size, input_len);

    var buffer: [512]u8 = undefined;
    const encoder = std.base64.standard.Encoder;
    const encoded = encoder.encode(&buffer, input[start..end]);

    const encoded_start: u32 = @intCast(encoder.calcSize(start));
    @memcpy(output[encoded_start .. encoded_start + encoded.len], encoded);
    lengths[gid] = @intCast(encoded.len);
}

/// Decodes every `chunk_size` elements of `input`, which is text of the base64 standard alphabet,
/// on its own, and writes the elements of the chunk of every thread to `output` at the position
/// of its chunk, and the number of them to `lengths`. `chunk_size` must be a multiple of 4, and
/// no chunk may contain the padding of the end of the text.
export fn base64DecodeKernel(
    input: [*]const u8,
    output: [*]u8,
    lengths: [*]u32,
    input_len: u32,
    chunk_size: u32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    const start = gid * chunk_size;
    if (start >= input_len) return;
    const end = @min(start + chunk_size, input_len);

    const decoder = std.base64.standard.Decoder;
    const chunk = input[start..end];
    const decoded_len = decoder.calcSizeForSlice(chunk) catch {
        lengths[gid] = 0;
        return;
    };
    var buffer: [512]u8 = undefined;
    decoder.decode(buffer[0..decoded_len], chunk) catch {
        lengths[gid] = 0;
        return;
    };

    const decoded_start: u32 = @intCast(decoder.calcSizeForSlice(input[0..start]) catch 0);
    @memcpy(output[decoded_start .. decoded_start + decoded_len], buffer[0..decoded_len]);
    lengths[gid] = @intCast(decoded_len);
}

/// Encodes the elements of `data` and decodes the text again, and writes whether the elements
/// that come back are the elements that went in, and how many there are.
export fn base64RoundTripKernel(data: [*]u8, n: u32, round_trip: *u32) callconv(.kernel) void {
    if (gpu.globalId(.x) != 0) return;

    var encode_buffer: [512]u8 = undefined;
    var decode_buffer: [512]u8 = undefined;

    const input = data[0..n];
    const encoder = std.base64.standard.Encoder;
    const encoded = encoder.encode(&encode_buffer, input);

    const decoder = std.base64.standard.Decoder;
    const decoded_len = decoder.calcSizeForSlice(encoded) catch {
        round_trip.* = 0;
        return;
    };
    decoder.decode(decode_buffer[0..decoded_len], encoded) catch {
        round_trip.* = 0;
        return;
    };

    const round_tripped = decode_buffer[0..decoded_len];
    const same = std.mem.eql(u8, round_tripped, input);
    round_trip.* = @intFromBool(same);

    gpu.print("encoded {d} bytes to {d} chars, decoded to {d} bytes: {s}\n", .{
        input.len,
        encoded.len,
        round_tripped.len,
        if (same) "round-trip successful!" else "round-trip failed",
    });
}

// ---------------------------------------------------------------------------------------------
// examples/string_search.zig: searching strings with std.mem
// ---------------------------------------------------------------------------------------------

/// The position of the first occurrence of the needle of every thread in `haystack`, or the
/// largest `u32` when the needle does not occur. `needles` holds fixed buffers of 64 bytes, and
/// `needle_lens` the length of the needle in each of them.
export fn stringSearchKernel(
    haystack: [*]const u8,
    haystack_len: u32,
    needles: [*]const [64]u8,
    needle_lens: [*]const u32,
    results: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= n) return;

    const hay = haystack[0..haystack_len];
    const needle = needles[gid][0..needle_lens[gid]];
    results[gid] = if (std.mem.indexOf(u8, hay, needle)) |position|
        @intCast(position)
    else
        std.math.maxInt(u32);

    gpu.print("Thread {d}: {d} occurrences\n", .{ gid, std.mem.count(u8, hay, needle) });
}

/// The number of occurrences of the needle of every thread in `haystack`, like
/// `stringSearchKernel`, which finds only the first one.
export fn stringCountKernel(
    haystack: [*]const u8,
    haystack_len: u32,
    needles: [*]const [64]u8,
    needle_lens: [*]const u32,
    results: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= n) return;

    const hay = haystack[0..haystack_len];
    const needle = needles[gid][0..needle_lens[gid]];
    const count = std.mem.count(u8, hay, needle);
    results[gid] = @intCast(count);

    gpu.print("Thread {d}: found {d} occurrences\n", .{ gid, count });
}

/// Writes the values of several standard library string functions over `input` into `results`,
/// which has an element for each of them:
/// 0: the input starts with "Hello"; 1: it ends with "GPU"; 2: it contains "the";
/// 3: its first five elements are "Hello"; 4: the first space; 5: the last space.
export fn stringUtilsKernel(input: [*]const u8, input_len: u32, results: [*]u32) callconv(.kernel) void {
    if (gpu.globalId(.x) != 0) return;
    const text = input[0..input_len];

    results[0] = @intFromBool(std.mem.startsWith(u8, text, "Hello"));
    results[1] = @intFromBool(std.mem.endsWith(u8, text, "GPU"));
    results[2] = @intFromBool(std.mem.containsAtLeast(u8, text, 1, "the"));
    results[3] = @intFromBool(std.mem.eql(u8, text[0..5], "Hello"));
    results[4] = if (std.mem.indexOfScalar(u8, text, ' ')) |position| @intCast(position) else std.math.maxInt(u32);
    results[5] = if (std.mem.lastIndexOfScalar(u8, text, ' ')) |position| @intCast(position) else std.math.maxInt(u32);

    gpu.print("startsWith('Hello'): {}, endsWith('GPU'): {}\n", .{
        results[0] != 0, results[1] != 0,
    });
}

/// The number of occurrences of the patterns "GPU", "CUDA", "parallel", "fast", and "compute" in
/// the text of every thread, which is a fixed buffer of 128 bytes of `texts` with the length in
/// `text_lens`.
export fn multiPatternKernel(
    texts: [*]const [128]u8,
    text_lens: [*]const u32,
    match_counts: [*]u32,
    n: u32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= n) return;

    const text = texts[gid][0..text_lens[gid]];
    const patterns = [_][]const u8{ "GPU", "CUDA", "parallel", "fast", "compute" };
    var matches: u32 = 0;
    for (patterns) |pattern| matches += @intCast(std.mem.count(u8, text, pattern));
    match_counts[gid] = matches;

    gpu.print("Thread {d}: {d} pattern matches\n", .{ gid, matches });
}

/// The position of the first occurrence of `needle` in `haystack`, found by searching a part of
/// the haystack in every thread, or the largest `u32` when no thread finds it. The threads search
/// overlapping parts, so a needle that starts at the end of a part is still found.
export fn optimizedSearchKernel(
    haystack: [*]const u8,
    haystack_len: u32,
    needle: [*]const u8,
    needle_len: u32,
    thread_count: u32,
    results: [*]u32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= thread_count) return;

    const chunk_size = haystack_len / thread_count;
    const start = gid * chunk_size;
    if (start >= haystack_len) return;
    const end = if (gid == thread_count - 1) haystack_len else @min((gid + 1) * chunk_size + needle_len, haystack_len);

    const hay = haystack[start..end];
    const target = needle[0..needle_len];
    results[gid] = if (std.mem.indexOf(u8, hay, target)) |position| @intCast(start + position) else std.math.maxInt(u32);
}

// ---------------------------------------------------------------------------------------------
// examples/json.zig: JSON parsing with std.json
// ---------------------------------------------------------------------------------------------

const Person = struct {
    name: []const u8,
    age: u32,
    score: f32,
};

const Position = struct {
    x: f32,
    y: f32,
    z: f32,
};

const Player = struct {
    id: u32,
    health: f32,
    position: Position,
};

const GameState = struct {
    level: u32,
    players: []Player,
    timestamp: u64,
};

/// Parses the JSON object of every thread into a `Person`, and writes its age and score, or the
/// largest `u32` and -1 when the text does not parse. Every thread parses into its own heap in
/// shared memory, so at most `json_threads` threads of a block can work.
export fn jsonParseKernel(
    texts: [*]const [256]u8,
    text_lens: [*]const u32,
    n: u32,
    ages: [*]u32,
    scores: [*]f32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    const tid = gpu.threadIdx(.x);
    if (gid >= n or tid >= json_threads) return;

    var fba = std.heap.FixedBufferAllocator.init(jsonHeap(tid));
    const parsed = std.json.parseFromSlice(Person, fba.allocator(), texts[gid][0..text_lens[gid]], .{
        .ignore_unknown_fields = true,
    }) catch {
        ages[gid] = std.math.maxInt(u32);
        scores[gid] = -1;
        return;
    };
    defer parsed.deinit();

    ages[gid] = parsed.value.age;
    scores[gid] = parsed.value.score;
    gpu.print("Thread {d}: {s} is {d}, score {d:.2}\n", .{
        gid, parsed.value.name, parsed.value.age, parsed.value.score,
    });
}

/// Parses the JSON object of every thread into a `GameState`, which has a slice of nested
/// objects, and writes the level and the number of players, or the largest `u32` and 0 when the
/// text does not parse.
export fn jsonParseNestedKernel(
    texts: [*]const [512]u8,
    text_lens: [*]const u32,
    n: u32,
    levels: [*]u32,
    player_counts: [*]u32,
    first_player_ids: [*]u32,
) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    const tid = gpu.threadIdx(.x);
    if (gid >= n or tid >= json_threads) return;

    var fba = std.heap.FixedBufferAllocator.init(jsonHeap(tid));
    const parsed = std.json.parseFromSlice(GameState, fba.allocator(), texts[gid][0..text_lens[gid]], .{}) catch {
        levels[gid] = std.math.maxInt(u32);
        player_counts[gid] = 0;
        first_player_ids[gid] = 0;
        return;
    };
    defer parsed.deinit();

    levels[gid] = parsed.value.level;
    player_counts[gid] = @intCast(parsed.value.players.len);
    first_player_ids[gid] = if (parsed.value.players.len > 0) parsed.value.players[0].id else 0;
    gpu.print("Thread {d}: level {d} with {d} players\n", .{
        gid, parsed.value.level, parsed.value.players.len,
    });
}

/// Parses a JSON array of objects into `[]const Person` and writes how many it holds, or zero
/// when the text does not parse.
export fn jsonArrayKernel(
    text: [*]const u8,
    text_len: u32,
    count: *u32,
    first_age: *u32,
    total_ages: *u64,
) callconv(.kernel) void {
    var bump = allocators.BumpAllocator(shared_heap_size).init(&shared_heap);
    if (gpu.globalId(.x) != 0) return;

    const parsed = std.json.parseFromSlice([]Person, bump.allocator(), text[0..text_len], .{}) catch {
        count.* = 0;
        first_age.* = 0;
        total_ages.* = 0;
        return;
    };
    defer parsed.deinit();

    count.* = @intCast(parsed.value.len);
    first_age.* = if (parsed.value.len > 0) parsed.value[0].age else 0;
    var total: u64 = 0;
    for (parsed.value) |person| total += person.age;
    total_ages.* = total;
    gpu.print("Parsed {d} people from a JSON array\n", .{parsed.value.len});
}

// ---------------------------------------------------------------------------------------------
// examples/dynamic.zig: growing containers, and formatting into shared buffers
// ---------------------------------------------------------------------------------------------

var dynamic_heap: [16 * 1024]u8 addrspace(.shared) = undefined;

/// Appends every element of `input` that is above 50, and twice every such element, to a list
/// that grows in shared memory, and writes the elements of the list to `output` with their count
/// in `count`.
export fn arrayListKernel(
    input: [*]const u32,
    output: [*]u32,
    count: *u32,
    n: u32,
) callconv(.kernel) void {
    // The allocator is block-wide, so every thread initializes it, and only thread 0 uses it.
    var bump = allocators.BumpAllocator(dynamic_heap.len).init(&dynamic_heap);
    if (gpu.threadIdx(.x) != 0) return;

    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(bump.allocator());

    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (input[i] > 50) {
            list.append(bump.allocator(), input[i]) catch break;
            list.append(bump.allocator(), input[i] * 2) catch break;
        }
    }

    for (list.items, 0..) |item, index| {
        if (index < n) output[index] = item;
    }
    count.* = @intCast(list.items.len);
}

/// The number of threads in a block of `formatStringsKernel`: one buffer of the shared heap for
/// every thread.
const format_threads = 64;
const format_buffer_size = 128;
var format_buffers: [format_threads][format_buffer_size]u8 addrspace(.shared) = undefined;

/// Formats a message of every thread into the buffer of the shared heap of that thread, and
/// writes the length of the message, which is
/// "Thread <id> processed value: <value> (hex: 0x<value>)".
///
/// The message goes into the shared buffer itself: `std.fmt.allocPrint` first allocates as many
/// bytes as the format has and grows that allocation, which needs several times the length of
/// the message in a heap that must hold it.
export fn formatStringsKernel(values: [*]const u32, output_lengths: [*]u32, n: u32) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    const tid = gpu.threadIdx(.x);
    if (gid >= n or tid >= format_threads) return;

    const message = std.fmt.bufPrint(sharedBytes(&format_buffers[tid]), "Thread {d} processed value: {d} (hex: 0x{x})", .{
        gid, values[gid], values[gid],
    }) catch {
        output_lengths[gid] = std.math.maxInt(u32);
        return;
    };
    output_lengths[gid] = @intCast(message.len);
}

// ---------------------------------------------------------------------------------------------
// examples/printf.zig: device output through std.gpu.print
// ---------------------------------------------------------------------------------------------

/// Prints the grid and the block that the kernel ran with, and the thread of every printing
/// thread: the first ten threads, and a percentage, which the format prints as is.
export fn printfKernel() callconv(.kernel) void {
    const tid = gpu.threadIdx(.x);
    const bid = gpu.blockIdx(.x);
    const gid = gpu.globalId(.x);

    if (gid < 10) {
        gpu.print("Hello from thread {d} in block {d} (global id: {d})\n", .{ tid, bid, gid });
    }
    if (gid == 0) {
        gpu.print("=== Kernel Launch Summary ===\n", .{});
        gpu.print("Grid size: {d} blocks\n", .{gpu.gridDim(.x)});
        gpu.print("Block size: {d} threads\n", .{gpu.blockDim(.x)});
        gpu.print("Total threads: {d}, {d}% of them printed\n", .{
            gpu.gridDim(.x) * gpu.blockDim(.x), @as(u32, 100),
        });
    }
}

/// Prints the sine, the cosine, and the square root of `x` from the thread 0 of every block.
export fn mathPrintfKernel(x: f32) callconv(.kernel) void {
    if (gpu.globalId(.x) != 0) return;
    gpu.print("Input: {d}\n", .{x});
    gpu.print("  sin: {d}\n", .{std.math.sin(x)});
    gpu.print("  cos: {d}\n", .{std.math.cos(x)});
    gpu.print("  sqrt: {d}\n", .{std.math.sqrt(x)});
}

/// Prints `x * x + 2 * x + 1` for the first five elements of `input`, and writes the results to
/// `output` for every element.
export fn debugComputeKernel(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const gid = gpu.globalId(.x);
    if (gid >= n) return;

    const value = input[gid];
    const result = value * value + 2 * value + 1;
    output[gid] = result;
    if (gid < 5) gpu.print("[{d}] input {d} output {d}\n", .{ gid, value, result });
}

// ---------------------------------------------------------------------------------------------
// examples/hello_gpu.zig
// ---------------------------------------------------------------------------------------------

/// Writes 42 into the first element of `out`, if there is one.
export fn helloKernel(out: [*]u32, len: usize) callconv(.kernel) void {
    if (len == 0) return;
    out[0] = 42;
}

/// Stores through a slice of `len` elements at the index of every thread, so in a Debug build the
/// threads past the end panic on the bounds check. Only the host's "assert" mode launches it,
/// because a failed assertion leaves the context unusable.
export fn outOfBoundsKernel(data: [*]u32, len: u32) callconv(.kernel) void {
    const elements = data[0..len];
    elements[gpu.globalId(.x)] = 1;
}

// ---------------------------------------------------------------------------------------------
// Added kernels: the builtin math functions for f32 and f64
// ---------------------------------------------------------------------------------------------

/// The values of the builtin math functions of `x`, in this order: sine, cosine, tangent, `e` to
/// the `x`, 2 to the `x`, the natural logarithm, the base-2 logarithm, and the base-10
/// logarithm.
fn mathResults(comptime T: type, x: T) [8]T {
    return .{
        @sin(x),
        @cos(x),
        @tan(x),
        @exp(x),
        @exp2(x),
        @log(x),
        @log2(x),
        @log10(x),
    };
}

/// Writes the eight values of `mathResults` for the element of every thread, one after the other,
/// so `output` has eight elements for every element of `input`.
export fn builtinMathF32Kernel(input: [*]const f32, output: [*]f32, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;

    const results = mathResults(f32, input[i]);
    for (results, 0..) |result, k| output[i * results.len + k] = result;
}

/// The same as `builtinMathF32Kernel` for `f64`.
export fn builtinMathF64Kernel(input: [*]const f64, output: [*]f64, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;

    const results = mathResults(f64, input[i]);
    for (results, 0..) |result, k| output[i * results.len + k] = result;
}

// ---------------------------------------------------------------------------------------------
// Added kernels: f128 arithmetic
// ---------------------------------------------------------------------------------------------

/// Writes four results of `f128` arithmetic for the pair of `f64` values of every thread: their
/// sum, their product, their quotient, and the square root of the first.
export fn f128Kernel(a: [*]const f64, b: [*]const f64, output: [*]f128, n: u32) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;

    const x: f128 = a[i];
    const y: f128 = b[i];
    output[i * 4 + 0] = x + y;
    output[i * 4 + 1] = x * y;
    output[i * 4 + 2] = x / y;
    output[i * 4 + 3] = @sqrt(x);
}

/// Compares the `f128` values of `a` and `b`, and narrows the first one, for the pair of every
/// thread. `results[i * 3 + 0]` has bit 0 set when `a[i] < b[i]` and bit 1 when they are equal,
/// `results[i * 3 + 1]` is `a[i]` narrowed to `f64`, and `results[i * 3 + 2]` is that narrowed to
/// `f32`.
export fn f128CompareKernel(a: [*]const f128, b: [*]const f128, results: [*]u32, n: u32, narrowed: [*]f64) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;

    const x = a[i];
    const y = b[i];

    var flags: u32 = 0;
    if (x < y) flags |= 1;
    if (x == y) flags |= 2;
    if (x > y) flags |= 4;
    results[i * 3 + 0] = flags;
    results[i * 3 + 1] = @intFromBool(@as(f32, @floatCast(x)) == @as(f32, @floatCast(y)));
    results[i * 3 + 2] = @intFromBool(@as(f64, @floatCast(x)) == 0.5);
    narrowed[i] = @floatCast(x);
}

// ---------------------------------------------------------------------------------------------
// Added kernels: std.fmt.parseFloat for f32, f64, and f128
// ---------------------------------------------------------------------------------------------

/// Parses the text of every thread as an `f32`, an `f64`, and an `f128`, and writes the three
/// results, or a NaN of the type when the text is not a number. `texts` holds fixed buffers of 32
/// bytes with the length of the text in `text_lens`.
export fn parseFloatKernel(
    texts: [*]const [32]u8,
    text_lens: [*]const u32,
    f32_results: [*]f32,
    f64_results: [*]f64,
    f128_results: [*]f128,
    n: u32,
) callconv(.kernel) void {
    const i = gpu.globalId(.x);
    if (i >= n) return;

    const text = texts[i][0..text_lens[i]];
    f32_results[i] = std.fmt.parseFloat(f32, text) catch std.math.nan(f32);
    f64_results[i] = std.fmt.parseFloat(f64, text) catch std.math.nan(f64);
    f128_results[i] = std.fmt.parseFloat(f128, text) catch std.math.nan(f128);
}

// ---------------------------------------------------------------------------------------------
// Added kernels: the allocators
// ---------------------------------------------------------------------------------------------

/// The elements that `deviceHeapKernel` appends to its list.
const device_heap_list_len = 500;

/// Grows a list of `u64` on the device heap of the driver, with `std.gpu.allocators.device_heap`,
/// and writes its length and the sum of its elements, then allocates memory aligned to 128 bytes,
/// which is more than the alignment of the heap blocks, fills it, and writes whether it came back
/// intact from the allocator.
export fn deviceHeapKernel(list_len: *u32, list_sum: *u64, aligned_ok: *u32) callconv(.kernel) void {
    if (gpu.globalId(.x) != 0) return;
    const allocator = allocators.device_heap;

    var list: std.ArrayList(u64) = .empty;
    defer list.deinit(allocator);

    var i: u32 = 0;
    while (i < device_heap_list_len) : (i += 1) {
        list.append(allocator, @as(u64, i) * i) catch {
            list_len.* = std.math.maxInt(u32);
            return;
        };
    }
    list_len.* = @intCast(list.items.len);

    var sum: u64 = 0;
    for (list.items) |item| sum +%= item;
    list_sum.* = sum;

    const aligned = allocator.alignedAlloc(u8, .@"64", 4096) catch {
        aligned_ok.* = 0;
        return;
    };
    defer allocator.free(aligned);

    var intact = true;
    for (aligned, 0..) |*byte, index| byte.* = @truncate(index * 7);
    for (aligned, 0..) |byte, index| {
        if (byte != @as(u8, @truncate(index * 7))) intact = false;
    }
    aligned_ok.* = @intFromBool(intact and @intFromPtr(aligned.ptr) % 64 == 0);
}

const bump_heap_size = 8 * 1024;
var bump_heap: [bump_heap_size]u8 addrspace(.shared) = undefined;
var bump_offsets: [block_size]u32 addrspace(.shared) = undefined;

/// Every thread of the block allocates one `u32` from the block-wide bump allocator in shared
/// memory at the same time, and writes its own thread index into the memory of its allocation.
/// One element of `intact` for every thread comes back as 1 when the allocation of that thread
/// still holds its own thread index after a barrier, `violations` is the number of pairs of
/// allocations that overlap, and `used` is the number of bytes of the heap in use afterwards.
export fn bumpAllocatorKernel(intact: [*]u32, violations: *u32, used: *u32) callconv(.kernel) void {
    const tid = gpu.threadIdx(.x);

    // Every thread must initialize the allocator before any of them allocates, because `init`
    // waits at a barrier for the thread that writes the offset of the unused memory.
    var bump = allocators.BumpAllocator(bump_heap.len).init(&bump_heap);

    const slot = bump.allocator().alloc(u32, 1) catch {
        intact[tid] = 0;
        return;
    };
    slot[0] = tid;
    const slot_shared: *addrspace(.shared) u32 = @addrSpaceCast(&slot[0]);
    bump_offsets[tid] = @intCast(@intFromPtr(slot_shared) - @intFromPtr(&bump_heap));
    gpu.syncThreads();

    // The allocations were made by different threads at the same time: none of them may cover
    // the thread index of another one.
    intact[tid] = @intFromBool(slot[0] == tid);

    if (tid == 0) {
        used.* = @intCast(bump.used());
        var overlaps: u32 = 0;
        for (0..gpu.blockDim(.x)) |a| {
            for (0..a) |b| {
                const distance = @as(i64, bump_offsets[a]) - @as(i64, bump_offsets[b]);
                if (distance > -@as(i64, @sizeOf(u32)) and distance < @sizeOf(u32)) overlaps += 1;
            }
        }
        violations.* = overlaps;
    }
}
