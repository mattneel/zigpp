//! The buffer that `std.gpu.print` and `std.gpu.assertFail` write to on AMD GPUs, and that
//! `std.gpu.hip` reads on the host. The HIP runtime has nothing like the `vprintf` and
//! `__assertfail` that the CUDA driver gives every PTX module, so the standard library brings its
//! own.
//!
//! A code object that calls either function exports a pointer to the buffer under the name
//! `symbol`, which is null until a host points it at one. `hip.Context.loadModule` points it at a
//! buffer on the device of the context, and `hip.Context.synchronize` writes the records of the
//! buffer to standard output and standard error and empties it. Without a host that sets the
//! pointer, the functions write nothing.
//!
//! The buffer is a `Header` followed by `Header.capacity` bytes of records. A thread claims the
//! space for a record by adding its size to `Header.claimed`, and then writes the record there.
//! Records that start past the end of the buffer are dropped, and the one that runs past the end
//! loses the text that does not fit.

/// The name that code objects export the pointer to the buffer as.
pub const symbol = "__zig_gpu_output";

/// The start of the buffer, which the records follow.
pub const Header = extern struct {
    /// The number of bytes of records that threads claimed since the host last emptied the
    /// buffer. It exceeds `capacity` when records were dropped.
    claimed: u64,
    /// The number of bytes that the records after the header can take.
    capacity: u64,
};

/// The start of a record, which its text follows. Records start at multiples of
/// `record_alignment` after the header.
pub const Record = extern struct {
    /// The number of bytes that the record claimed: this header, the text, and the padding that
    /// aligns the next record.
    size: u32,
    /// The number of bytes of text after this header, which is less than the text that was
    /// written when the record runs past the end of the buffer.
    len: u32,
    kind: Kind,
    /// `@workGroupId` of the thread that wrote the record in each dimension.
    block: [3]u32,
    /// `@workItemId` of the thread that wrote the record in each dimension.
    thread: [3]u32,
};

pub const Kind = enum(u32) {
    /// Text from `std.gpu.print`, for standard output.
    print,
    /// The message of `std.gpu.assertFail`, which fails the launch.
    assert,
    _,
};

pub const record_alignment = 8;
