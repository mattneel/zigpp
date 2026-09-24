//! Host-side access to AMD GPUs through the HIP runtime library.
//!
//! The runtime library is loaded when it is needed, so a program that uses this namespace needs
//! no HIP headers, import library, or toolkit to build. On Linux the shared object is loaded
//! with `std.DynLib`, which uses `dlopen` and so needs libc, so link the program with libc; on
//! Windows the DLL is loaded with `ntdll.LdrLoadDll`, which needs no C library. These are the
//! only operating systems that the HIP runtime comes for, and other ones report `Driver.open`
//! as a compile error.
//!
//! This is the part of the runtime that runs kernels: devices and their primary context, modules
//! of code objects, kernel functions, device memory, and streams. Events, graphs, the memory
//! pools of stream-ordered allocation, and the rest of the runtime are not part of it.
//!
//! A kernel is a code object: the ELF shared object that `zig build-lib -dynamic -target
//! amdgcn-amdhsa` produces, which holds machine code for one AMD architecture. Unlike the PTX
//! that the NVIDIA driver compiles for the device it is loaded on, a code object runs only on the
//! devices whose `Device.archName` it was built for, so a program that supports several
//! architectures builds one code object for each and loads the one that matches the device. The
//! name to give the compiler comes from `Device.archName`, such as "gfx1036" for the integrated
//! GPUs of some Ryzen processors, or `-mcpu=gfx90a+sramecc-xnack` for the "gfx90a:sramecc+:xnack-"
//! of an MI250.
//!
//! Functions return `Error`, which names the codes that the runtime reports, such as
//! `error.InvalidValue` and `error.OutOfMemory`. The objects of this namespace, `Device`,
//! `Context`, `Module`, `Function`, `Buffer`, and `Stream`, hold a pointer to the `Driver` they
//! came from: the driver must stay open, and must not move, while any of them is in use.
//!
//! ```zig
//! const std = @import("std");
//! const hip = std.gpu.hip;
//!
//! // The code object of a kernel, such as the output of
//! // `zig build-lib -dynamic -target amdgcn-amdhsa -mcpu=gfx1036`.
//! const code_object: []const u8 = @embedFile("kernel.co");
//!
//! pub fn main() !void {
//!     var driver = try hip.Driver.open();
//!     defer driver.close();
//!
//!     const device = try driver.device(0);
//!     const context = try device.retainPrimaryContext();
//!     defer context.release();
//!
//!     const module = try context.loadModule(code_object, .{});
//!     defer module.unload();
//!     const kernel = try module.function("add_one");
//!
//!     var data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//!     const buffer = try context.alloc(u32, data.len);
//!     defer buffer.free();
//!     try buffer.copyFromHost(&data);
//!     try kernel.launch(hip.LaunchConfig.linear(data.len, 4), .{ buffer, @as(u32, data.len) });
//!     try context.synchronize();
//!     try buffer.copyToHost(&data);
//! }
//! ```

const std = @import("../std.zig");
const builtin = @import("builtin");
const launch_arguments = @import("launch_arguments.zig");
const output_buffer = @import("output_buffer.zig");
const testing = std.testing;
const windows = std.os.windows;

/// Whether the runtime library can be loaded here. On Linux, `std.DynLib` loads shared libraries
/// with `dlopen`, which is part of libc; the pure-Zig ELF loader cannot load the vendor runtime.
/// Windows loads the DLL with `ntdll.LdrLoadDll`, which is part of the system.
const driver_supported = switch (builtin.os.tag) {
    .linux => builtin.link_libc,
    .windows => true,
    else => false,
};

/// Every function of the runtime API that this namespace uses returns one of these, a
/// `hipError_t`.
const Result = c_int;

/// The result of a driver call that succeeded, `hipSuccess`.
const success: Result = 0;

/// The errors that the runtime reports, one name per `hipError_t` code worth distinguishing.
///
/// The names follow the codes of the runtime: `error.InvalidValue` is `hipErrorInvalidValue`,
/// `error.NoBinaryForGpu` is `hipErrorNoBinaryForGpu`, and so on, and a code that the two
/// runtimes share has the name that it has in `std.gpu.cuda`. Where the runtime words a code
/// differently, the member says what the runtime calls it when the CUDA name would mislead:
/// `error.OutOfMemory` is the `error.OutOfDeviceMemory` of CUDA, `error.EccNotCorrectable` is its
/// `error.EccUncorrectable`, and `error.SetOnActiveProcess` is its `error.PrimaryContextActive`.
/// `error.Unknown` is `hipErrorUnknown`, the code the runtime uses for a failure that has no more
/// specific code, and `error.Unexpected` is a code that this version of the standard library does
/// not know.
///
/// The codes that only the CUDA driver has, such as the ones of its profiler and of its PTX JIT,
/// are not here: the runtime reports an image that it cannot load as `error.InvalidImage` or as
/// `error.InvalidKernelFile`.
pub const Error = error{
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    InvalidConfiguration,
    InvalidPitchValue,
    InvalidSymbol,
    InvalidDevicePointer,
    InvalidMemcpyDirection,
    InsufficientDriver,
    MissingConfiguration,
    PriorLaunchFailure,
    InvalidDeviceFunction,
    NoDevice,
    InvalidDevice,
    InvalidImage,
    InvalidContext,
    ContextAlreadyCurrent,
    MapFailed,
    UnmapFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoBinaryForGpu,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    EccNotCorrectable,
    UnsupportedLimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidKernelFile,
    InvalidGraphicsContext,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidHandle,
    IllegalState,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    SetOnActiveProcess,
    ContextIsDestroyed,
    Assert,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    LaunchFailure,
    CooperativeLaunchTooLarge,
    NotSupported,
    StreamCaptureUnsupported,
    StreamCaptureInvalidated,
    StreamCaptureMerge,
    StreamCaptureUnmatched,
    StreamCaptureUnjoined,
    StreamCaptureIsolation,
    StreamCaptureImplicit,
    CapturedEvent,
    StreamCaptureWrongThread,
    GraphExecUpdateFailure,
    RuntimeMemory,
    RuntimeOther,
    Unknown,
    Unexpected,
};

/// The errors that `Driver.open` reports, in addition to the errors of the runtime.
pub const OpenError = error{
    /// The runtime library was not found, usually because no AMD driver is installed.
    DriverNotFound,
    /// The runtime library is loaded, but it does not have the functions of the runtime API that
    /// this namespace uses.
    IncompatibleDriver,
} || Error;

/// Converts the `hipError_t` of a driver call into an error; `hipSuccess` is not an error.
fn check(result: Result) Error!void {
    if (result == success) return;
    return errorFor(result);
}

/// The error for a `hipError_t` code other than `hipSuccess`.
fn errorFor(result: Result) Error {
    return switch (result) {
        1 => error.InvalidValue,
        2 => error.OutOfMemory,
        3 => error.NotInitialized,
        4 => error.Deinitialized,
        5 => error.ProfilerDisabled,
        6 => error.ProfilerNotInitialized,
        7 => error.ProfilerAlreadyStarted,
        8 => error.ProfilerAlreadyStopped,
        9 => error.InvalidConfiguration,
        12 => error.InvalidPitchValue,
        13 => error.InvalidSymbol,
        17 => error.InvalidDevicePointer,
        21 => error.InvalidMemcpyDirection,
        35 => error.InsufficientDriver,
        52 => error.MissingConfiguration,
        53 => error.PriorLaunchFailure,
        98 => error.InvalidDeviceFunction,
        100 => error.NoDevice,
        101 => error.InvalidDevice,
        200 => error.InvalidImage,
        201 => error.InvalidContext,
        202 => error.ContextAlreadyCurrent,
        205 => error.MapFailed,
        206 => error.UnmapFailed,
        207 => error.ArrayIsMapped,
        208 => error.AlreadyMapped,
        209 => error.NoBinaryForGpu,
        210 => error.AlreadyAcquired,
        211 => error.NotMapped,
        212 => error.NotMappedAsArray,
        213 => error.NotMappedAsPointer,
        214 => error.EccNotCorrectable,
        215 => error.UnsupportedLimit,
        216 => error.ContextAlreadyInUse,
        217 => error.PeerAccessUnsupported,
        218 => error.InvalidKernelFile,
        219 => error.InvalidGraphicsContext,
        300 => error.InvalidSource,
        301 => error.FileNotFound,
        302 => error.SharedObjectSymbolNotFound,
        303 => error.SharedObjectInitFailed,
        304 => error.OperatingSystem,
        400 => error.InvalidHandle,
        401 => error.IllegalState,
        500 => error.NotFound,
        600 => error.NotReady,
        700 => error.IllegalAddress,
        701 => error.LaunchOutOfResources,
        702 => error.LaunchTimeout,
        704 => error.PeerAccessAlreadyEnabled,
        705 => error.PeerAccessNotEnabled,
        708 => error.SetOnActiveProcess,
        709 => error.ContextIsDestroyed,
        710 => error.Assert,
        712 => error.HostMemoryAlreadyRegistered,
        713 => error.HostMemoryNotRegistered,
        719 => error.LaunchFailure,
        720 => error.CooperativeLaunchTooLarge,
        801 => error.NotSupported,
        900 => error.StreamCaptureUnsupported,
        901 => error.StreamCaptureInvalidated,
        902 => error.StreamCaptureMerge,
        903 => error.StreamCaptureUnmatched,
        904 => error.StreamCaptureUnjoined,
        905 => error.StreamCaptureIsolation,
        906 => error.StreamCaptureImplicit,
        907 => error.CapturedEvent,
        908 => error.StreamCaptureWrongThread,
        910 => error.GraphExecUpdateFailure,
        999 => error.Unknown,
        1052 => error.RuntimeMemory,
        1053 => error.RuntimeOther,
        else => error.Unexpected,
    };
}

/// The version of the HIP runtime, such as `.{ .major = 6, .minor = 3, .patch = 42560 }` for HIP
/// 6.3.42560.
pub const Version = struct {
    /// The major version, the first number of the version: 6 for HIP 6.3.42560.
    major: u32,
    /// The minor version, the number after the first point: 3 for HIP 6.3.42560.
    minor: u32,
    /// The patch version, the last number of the version: 42560 for HIP 6.3.42560.
    patch: u32,
};

/// The compute capability of a device as HIP reports it: the major and minor version of the
/// instruction set it runs, such as `.{ .major = 10, .minor = 3 }` for a gfx1036 GPU.
///
/// AMD has no such version of its own: the runtime derives it from the architecture, whose
/// generation is the major version and whose revision is the minor one, so it names the same
/// architecture that `Device.archName` does. It is not what selects a code object, because a
/// capability covers several architectures: build and load the code object of the whole
/// `Device.archName` instead.
pub const ComputeCapability = struct {
    /// The major version, the generation of the architecture: 10 for gfx1036.
    major: u32,
    /// The minor version, the revision of the architecture: 3 for gfx1036.
    minor: u32,
};

/// The attributes of a device that `Device.attribute` reads, one name per
/// `hipDeviceAttribute*` value of the runtime.
///
/// An attribute that means the same thing as one of `std.gpu.cuda.Attribute` has the same name
/// here, but its value is the one that the HIP runtime uses, and the two runtimes number their
/// attributes differently. The attributes of the CUDA-compatible part of the enumeration come
/// first, and the ones that only AMD has, such as the clock that the device-side clock
/// instructions count, are above 10000.
///
/// An attribute that the runtime does not implement for the device reports
/// `error.InvalidValue` or `error.NotSupported` rather than a value.
pub const Attribute = enum(c_int) {
    /// Whether ECC support is enabled
    ecc_enabled = 0,
    /// The maximum size of the access policy window, in bytes
    max_access_policy_window_size = 1,
    /// Number of asynchronous engines
    async_engine_count = 2,
    /// Whether host memory can be mapped into the device address space
    can_map_host_memory = 3,
    /// Device can access host registered memory at the same virtual address as the CPU
    can_use_host_pointer_for_registered_mem = 4,
    /// Peak clock frequency in kilohertz
    clock_rate = 5,
    /// Compute mode that the device is currently in
    compute_mode = 6,
    /// Device supports compute preemption
    compute_preemption_supported = 7,
    /// Device can possibly execute multiple kernels concurrently
    concurrent_kernels = 8,
    /// Device can coherently access managed memory concurrently with the CPU
    concurrent_managed_access = 9,
    /// Device supports launching cooperative kernels
    cooperative_launch = 10,
    /// Device supports launching cooperative kernels on multiple devices
    cooperative_multi_device_launch = 11,
    /// Device can copy memory and run kernels at the same time. Deprecated: use the async engine
    /// count instead.
    gpu_overlap = 12,
    /// The host can directly access managed memory on the device without migration
    direct_managed_mem_access_from_host = 13,
    /// Device supports caching globals in L1
    global_l1_cache_supported = 14,
    /// Link between the device and the host supports all native atomic operations
    host_native_atomic_supported = 15,
    /// Device is integrated with host memory
    integrated = 16,
    /// Device is on a multi-GPU board
    multi_gpu_board = 17,
    /// Whether there is a run time limit on kernels
    kernel_exec_timeout = 18,
    /// Size of the L2 cache in bytes; 0 when the device has none
    l2_cache_size = 19,
    /// Device supports caching locals in L1
    local_l1_cache_supported = 20,
    /// The 8-byte locally unique identifier of the device, undefined on TCC and non-Windows
    /// platforms
    luid = 21,
    /// The LUID device node mask, undefined on TCC and non-Windows platforms
    luid_device_node_mask = 22,
    /// Major compute capability version number
    compute_capability_major = 23,
    /// Device can allocate managed memory on this system
    managed_memory = 24,
    /// Maximum number of blocks per multiprocessor
    max_blocks_per_multiprocessor = 25,
    /// Maximum block dimension X
    max_block_dim_x = 26,
    /// Maximum block dimension Y
    max_block_dim_y = 27,
    /// Maximum block dimension Z
    max_block_dim_z = 28,
    /// Maximum grid dimension X
    max_grid_dim_x = 29,
    /// Maximum grid dimension Y
    max_grid_dim_y = 30,
    /// Maximum grid dimension Z
    max_grid_dim_z = 31,
    /// Maximum 1D surface size
    maximum_surface1d_width = 32,
    /// Maximum dimensions of a 1D layered surface
    maximum_surface1d_layered_width = 33,
    /// Maximum dimensions (width, height) of a 2D surface
    maximum_surface2d_width = 34,
    /// Maximum dimensions of a 2D layered surface
    maximum_surface2d_layered_width = 35,
    /// Maximum dimensions (width, height, depth) of a 3D surface
    maximum_surface3d_width = 36,
    /// Maximum dimensions of a cubemap surface
    maximum_surfacecubemap_width = 37,
    /// Maximum dimensions of a cubemap layered surface
    maximum_surfacecubemap_layered_width = 38,
    /// Maximum size of a 1D texture
    maximum_texture1d_width = 39,
    /// Maximum dimensions of a 1D layered texture
    maximum_texture1d_layered_width = 40,
    /// Maximum number of elements in a 1D linear texture
    maximum_texture1d_linear_width = 41,
    /// Maximum size of a 1D mipmapped texture
    maximum_texture1d_mipmapped_width = 42,
    /// Maximum width of a 2D texture
    maximum_texture2d_width = 43,
    /// Maximum height of a 2D texture
    maximum_texture2d_height = 44,
    /// Maximum dimensions (width, height) of a 2D texture that gathers
    maximum_texture2d_gather = 45,
    /// Maximum dimensions of a 2D layered texture
    maximum_texture2d_layered = 46,
    /// Maximum dimensions (width, height, pitch) of a 2D texture over pitched memory
    maximum_texture2d_linear = 47,
    /// Maximum dimensions (width, height) of a 2D mipmapped texture
    maximum_texture2d_mipmapped = 48,
    /// Maximum width of a 3D texture
    maximum_texture3d_width = 49,
    /// Maximum height of a 3D texture
    maximum_texture3d_height = 50,
    /// Maximum depth of a 3D texture
    maximum_texture3d_depth = 51,
    /// Maximum dimensions of an alternate 3D texture
    maximum_texture3d_alternate = 52,
    /// Maximum dimensions of a cubemap texture
    maximum_texturecubemap_width = 53,
    /// Maximum dimensions of a cubemap layered texture
    maximum_texturecubemap_layered_width = 54,
    /// Maximum number of threads in each dimension of a block
    max_threads_dim = 55,
    /// Maximum number of threads per block
    max_threads_per_block = 56,
    /// Maximum number of resident threads per multiprocessor
    max_threads_per_multiprocessor = 57,
    /// Maximum pitch in bytes allowed by memory copies
    max_pitch = 58,
    /// Global memory bus width in bits
    global_memory_bus_width = 59,
    /// Peak memory clock frequency in kilohertz
    memory_clock_rate = 60,
    /// Minor compute capability version number
    compute_capability_minor = 61,
    /// Unique identifier of a group of devices on the same multi-GPU board
    multi_gpu_board_group_id = 62,
    /// Number of multiprocessors on the device
    multiprocessor_count = 63,
    /// Device supports coherently accessing pageable memory without registering it with the host
    pageable_memory_access = 65,
    /// Device accesses pageable memory through the page tables of the host
    pageable_memory_access_uses_host_page_tables = 66,
    /// PCI bus ID of the device
    pci_bus_id = 67,
    /// PCI device ID of the device
    pci_device_id = 68,
    /// PCI domain ID of the device
    pci_domain_id = 69,
    /// Maximum L2 persisting lines capacity in bytes
    max_persisting_l2_cache_size = 70,
    /// Maximum number of 32-bit registers available per block
    max_registers_per_block = 71,
    /// Maximum number of 32-bit registers available per multiprocessor
    max_registers_per_multiprocessor = 72,
    /// Shared memory reserved by the runtime for every block, in bytes
    reserved_shared_memory_per_block = 73,
    /// Maximum shared memory available per block in bytes
    max_shared_memory_per_block = 74,
    /// Maximum shared memory per block that is available to a kernel that asks for the
    /// non-portable amount
    max_shared_memory_per_block_optin = 75,
    /// Maximum shared memory available per multiprocessor in bytes, the CUDA-compatible
    /// attribute of the runtime
    max_shared_memory_per_multiprocessor = 76,
    /// Ratio of single precision performance (in floating-point operations per second) to double
    /// precision performance
    single_to_double_precision_perf_ratio = 77,
    /// Device supports stream priorities
    stream_priorities_supported = 78,
    /// Alignment requirement for surfaces
    surface_alignment = 79,
    /// Device is using the TCC driver model
    tcc_driver = 80,
    /// Alignment requirement for textures
    texture_alignment = 81,
    /// Pitch alignment requirement for textures over pitched memory
    texture_pitch_alignment = 82,
    /// Size of the constant memory of the device in bytes
    total_constant_memory = 83,
    /// Global memory available on the device in bytes
    total_global_memory = 84,
    /// Device shares a unified address space with the host
    unified_addressing = 85,
    /// Warp size in threads
    warp_size = 87,
    /// Device supports the stream-ordered memory allocator
    memory_pools_supported = 88,
    /// Device supports the virtual memory management API
    virtual_memory_management_supported = 89,
    /// Device supports host memory registration
    host_register_supported = 90,
    /// Handle types that the memory pools of the device support
    mempool_supported_handle_types = 91,
    /// Frequency in kilohertz of the timer that the device-side clock instructions count
    clock_instruction_rate = 10000,
    /// Maximum shared memory per multiprocessor in bytes, the attribute that the AMD runtime
    /// adds for it rather than the CUDA-compatible `max_shared_memory_per_multiprocessor`
    max_shared_memory_per_multiprocessor_amd = 10002,
    /// Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hdp_mem_flush_cntl = 10005,
    /// Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hdp_reg_flush_cntl = 10006,
    /// Device supports cooperative launch on multiple devices with unmatched functions
    cooperative_multi_device_unmatched_func = 10007,
    /// Device supports cooperative launch on multiple devices with unmatched grid dimensions
    cooperative_multi_device_unmatched_grid_dim = 10008,
    /// Device supports cooperative launch on multiple devices with unmatched block dimensions
    cooperative_multi_device_unmatched_block_dim = 10009,
    /// Device supports cooperative launch on multiple devices with unmatched shared memory
    cooperative_multi_device_unmatched_shared_mem = 10010,
    /// Whether the device is a Large Bar device
    is_large_bar = 10011,
    /// Revision of the GPU in this device
    asic_revision = 10012,
    /// Device supports waiting on a value in memory from a stream
    can_use_stream_wait_value = 10013,
    /// Device supports images
    image_support = 10014,
    /// Number of compute units that the device has when none of them are disabled
    physical_multiprocessor_count = 10015,
    /// Device supports fine-grained memory
    fine_grain_support = 10016,
    /// Constant frequency of the wall clock in kilohertz
    wall_clock_rate = 10017,
    _,
};

/// The properties that the runtime reports for a device, the `hipDeviceProp_tR0600` of the
/// headers, which `hipGetDevicePropertiesR0600` fills in.
///
/// Only `gcn_arch_name` and `name` are read; the rest of the fields are here to give the struct
/// the layout that the runtime expects, because it writes the whole struct. The function that
/// takes it is named `hipGetDevicePropertiesR0600` because AMD changes the layout of the struct
/// and keeps the old ones under versioned names; this one is the layout of the 6.0 headers and of
/// the 6.x runtimes.
const DeviceProperties = extern struct {
    name: [256]u8,
    /// The `hipUUID` of the header: 16 bytes of identifier.
    uuid: [16]u8,
    luid: [8]u8,
    luid_device_node_mask: c_uint,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: c_int,
    warp_size: c_int,
    mem_pitch: usize,
    max_threads_per_block: c_int,
    max_threads_dim: [3]c_int,
    max_grid_size: [3]c_int,
    clock_rate: c_int,
    total_const_mem: usize,
    major: c_int,
    minor: c_int,
    texture_alignment: usize,
    texture_pitch_alignment: usize,
    device_overlap: c_int,
    multi_processor_count: c_int,
    kernel_exec_timeout_enabled: c_int,
    integrated: c_int,
    can_map_host_memory: c_int,
    compute_mode: c_int,
    max_texture1d: c_int,
    max_texture1d_mipmap: c_int,
    max_texture1d_linear: c_int,
    max_texture2d: [2]c_int,
    max_texture2d_mipmap: [2]c_int,
    max_texture2d_linear: [3]c_int,
    max_texture2d_gather: [2]c_int,
    max_texture3d: [3]c_int,
    max_texture3d_alt: [3]c_int,
    max_texture_cubemap: c_int,
    max_texture1d_layered: [2]c_int,
    max_texture2d_layered: [3]c_int,
    max_texture_cubemap_layered: [2]c_int,
    max_surface1d: c_int,
    max_surface2d: [2]c_int,
    max_surface3d: [3]c_int,
    max_surface1d_layered: [2]c_int,
    max_surface2d_layered: [3]c_int,
    max_surface_cubemap: c_int,
    max_surface_cubemap_layered: [2]c_int,
    surface_alignment: usize,
    concurrent_kernels: c_int,
    ecc_enabled: c_int,
    pci_bus_id: c_int,
    pci_device_id: c_int,
    pci_domain_id: c_int,
    tcc_driver: c_int,
    async_engine_count: c_int,
    unified_addressing: c_int,
    memory_clock_rate: c_int,
    memory_bus_width: c_int,
    l2_cache_size: c_int,
    persisting_l2_cache_max_size: c_int,
    max_threads_per_multi_processor: c_int,
    stream_priorities_supported: c_int,
    global_l1_cache_supported: c_int,
    local_l1_cache_supported: c_int,
    shared_mem_per_multi_processor: usize,
    regs_per_multi_processor: c_int,
    managed_memory: c_int,
    is_multi_gpu_board: c_int,
    multi_gpu_board_group_id: c_int,
    host_native_atomic_supported: c_int,
    single_to_double_precision_perf_ratio: c_int,
    pageable_memory_access: c_int,
    concurrent_managed_access: c_int,
    compute_preemption_supported: c_int,
    can_use_host_pointer_for_registered_mem: c_int,
    cooperative_launch: c_int,
    cooperative_multi_device_launch: c_int,
    shared_mem_per_block_optin: usize,
    pageable_memory_access_uses_host_page_tables: c_int,
    direct_managed_mem_access_from_host: c_int,
    max_blocks_per_multi_processor: c_int,
    access_policy_max_window_size: c_int,
    reserved_shared_mem_per_block: usize,
    host_register_supported: c_int,
    sparse_hip_array_supported: c_int,
    host_register_read_only_supported: c_int,
    timeline_semaphore_interop_supported: c_int,
    memory_pools_supported: c_int,
    gpu_direct_rdma_supported: c_int,
    gpu_direct_rdma_flush_writes_options: c_uint,
    gpu_direct_rdma_writes_ordering: c_int,
    memory_pool_supported_handle_types: c_uint,
    deferred_mapping_hip_array_supported: c_int,
    ipc_event_supported: c_int,
    cluster_launch: c_int,
    unified_function_pointers: c_int,
    reserved: [63]c_int,
    hip_reserved: [32]c_int,
    gcn_arch_name: [256]u8,
    max_shared_memory_per_multi_processor: usize,
    clock_instruction_rate: c_int,
    /// The `hipDeviceArch_t` of the header: 17 one-bit fields in one `unsigned int`.
    arch: u32,
    hdp_mem_flush_cntl: ?*c_uint,
    hdp_reg_flush_cntl: ?*c_uint,
    cooperative_multi_device_unmatched_func: c_int,
    cooperative_multi_device_unmatched_grid_dim: c_int,
    cooperative_multi_device_unmatched_block_dim: c_int,
    cooperative_multi_device_unmatched_shared_mem: c_int,
    is_large_bar: c_int,
    asic_revision: c_int,
};

/// The names of the files that the runtime library comes as, most specific first: the versioned
/// name that the packages of a HIP 7 installation install, the one of HIP 6, and, on Linux, the
/// name that a package without a version installs.
const library_names = switch (builtin.os.tag) {
    .linux => .{ "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so" },
    .windows => .{
        std.unicode.utf8ToUtf16LeStringLiteral("amdhip64_7.dll"),
        std.unicode.utf8ToUtf16LeStringLiteral("amdhip64_6.dll"),
    },
    else => @compileError("std.gpu.hip is only available on Linux and Windows; on Linux it needs libc to load the runtime library with dlopen"),
};

/// The runtime library, loaded and unloaded by the operating system, and the way to find a
/// function in it.
const Library = struct {
    /// The handle of the library: a `std.DynLib` on Linux, the module handle that
    /// `ntdll.LdrLoadDll` returned on Windows.
    handle: Handle,

    /// The type of that handle.
    const Handle = switch (builtin.os.tag) {
        .linux => std.DynLib,
        .windows => *anyopaque,
        else => @compileError("std.gpu.hip is only available on Linux and Windows; on Linux it needs libc to load the runtime library with dlopen"),
    };

    /// Loads the first of `library_names` that the operating system can find, or null when it
    /// finds none of them.
    fn open() ?Library {
        inline for (library_names) |name| {
            switch (builtin.os.tag) {
                .linux => if (std.DynLib.openZ(name)) |library| {
                    return .{ .handle = library };
                } else |_| {},
                .windows => {
                    var handle: *anyopaque = undefined;
                    switch (windows.ntdll.LdrLoadDll(null, null, &.init(name), &handle)) {
                        .SUCCESS => return .{ .handle = handle },
                        else => {},
                    }
                },
                else => @compileError("std.gpu.hip is only available on Linux and Windows"),
            }
        }
        return null;
    }

    /// The address of the function that the library exports under `name`, or null when the
    /// library has no function of that name.
    fn lookup(library: *Library, comptime T: type, name: [:0]const u8) ?T {
        return switch (builtin.os.tag) {
            .linux => library.handle.lookup(T, name),
            .windows => address: {
                var address: *anyopaque = undefined;
                switch (windows.ntdll.LdrGetProcedureAddress(library.handle, &.init(name), 0, &address)) {
                    .SUCCESS => break :address @ptrCast(address),
                    else => break :address null,
                }
            },
            else => @compileError("std.gpu.hip is only available on Linux and Windows"),
        };
    }

    /// Unloads the library. The driver and everything created from it must not be used
    /// afterwards.
    fn close(library: Library) void {
        switch (builtin.os.tag) {
            .linux => {
                var handle = library.handle;
                handle.close();
            },
            .windows => _ = windows.ntdll.LdrUnloadDll(library.handle),
            else => @compileError("std.gpu.hip is only available on Linux and Windows"),
        }
    }
};

/// Pointers to the functions of the runtime API that this namespace uses. `Driver.open` resolves
/// them from the runtime library; every one of them returns a `hipError_t`.
const Functions = struct {
    init: *const fn (flags: c_uint) callconv(.c) Result,
    runtime_get_version: *const fn (version: *c_int) callconv(.c) Result,
    device_get_count: *const fn (count: *c_int) callconv(.c) Result,
    device_get: *const fn (device: *c_int, ordinal: c_int) callconv(.c) Result,
    device_get_name: *const fn (name: [*]u8, len: c_int, device: c_int) callconv(.c) Result,
    device_total_memory: *const fn (bytes: *usize, device: c_int) callconv(.c) Result,
    device_get_attribute: *const fn (
        value: *c_int,
        attribute: c_int,
        device: c_int,
    ) callconv(.c) Result,
    device_get_properties: *const fn (properties: *DeviceProperties, device: c_int) callconv(.c) Result,
    device_primary_ctx_retain: *const fn (context: *?*anyopaque, device: c_int) callconv(.c) Result,
    device_primary_ctx_release: *const fn (device: c_int) callconv(.c) Result,
    ctx_set_current: *const fn (context: ?*anyopaque) callconv(.c) Result,
    device_synchronize: *const fn () callconv(.c) Result,
    device_set_limit: *const fn (limit: c_int, value: usize) callconv(.c) Result,
    module_load_data_ex: *const fn (
        module: *?*anyopaque,
        image: *const anyopaque,
        option_count: c_uint,
        options: ?[*]const c_int,
        option_values: ?[*]?*anyopaque,
    ) callconv(.c) Result,
    module_load_data: *const fn (module: *?*anyopaque, image: *const anyopaque) callconv(.c) Result,
    module_unload: *const fn (module: ?*anyopaque) callconv(.c) Result,
    module_get_function: *const fn (
        function: *?*anyopaque,
        module: ?*anyopaque,
        name: [*:0]const u8,
    ) callconv(.c) Result,
    module_get_global: *const fn (
        address: *?*anyopaque,
        bytes: *usize,
        module: ?*anyopaque,
        name: [*:0]const u8,
    ) callconv(.c) Result,
    mem_alloc: *const fn (address: *?*anyopaque, bytes: usize) callconv(.c) Result,
    mem_free: *const fn (address: ?*anyopaque) callconv(.c) Result,
    memcpy_htod: *const fn (device: ?*anyopaque, host: *const anyopaque, bytes: usize) callconv(.c) Result,
    memcpy_dtoh: *const fn (host: *anyopaque, device: ?*anyopaque, bytes: usize) callconv(.c) Result,
    memset_d8: *const fn (device: ?*anyopaque, value: u8, count: usize) callconv(.c) Result,
    stream_create: *const fn (stream: *?*anyopaque) callconv(.c) Result,
    stream_destroy: *const fn (stream: ?*anyopaque) callconv(.c) Result,
    stream_synchronize: *const fn (stream: ?*anyopaque) callconv(.c) Result,
    launch_kernel: *const fn (
        function: ?*anyopaque,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory: c_uint,
        stream: ?*anyopaque,
        parameters: ?[*]?*anyopaque,
        extra: ?*anyopaque,
    ) callconv(.c) Result,
    error_name: *const fn (result: Result) callconv(.c) [*:0]const u8,
};

/// The name of every function in `Functions`, in the same order, as the runtime library exports
/// it. Unlike the CUDA driver, the HIP runtime does not export two names for a function that it
/// has changed: the plain name is the current one.
const function_symbols = .{
    .{ .field = "init", .symbol = "hipInit" },
    .{ .field = "runtime_get_version", .symbol = "hipRuntimeGetVersion" },
    .{ .field = "device_get_count", .symbol = "hipGetDeviceCount" },
    .{ .field = "device_get", .symbol = "hipDeviceGet" },
    .{ .field = "device_get_name", .symbol = "hipDeviceGetName" },
    .{ .field = "device_total_memory", .symbol = "hipDeviceTotalMem" },
    .{ .field = "device_get_attribute", .symbol = "hipDeviceGetAttribute" },
    .{ .field = "device_get_properties", .symbol = "hipGetDevicePropertiesR0600" },
    .{ .field = "device_primary_ctx_retain", .symbol = "hipDevicePrimaryCtxRetain" },
    .{ .field = "device_primary_ctx_release", .symbol = "hipDevicePrimaryCtxRelease" },
    .{ .field = "ctx_set_current", .symbol = "hipCtxSetCurrent" },
    .{ .field = "device_synchronize", .symbol = "hipDeviceSynchronize" },
    .{ .field = "device_set_limit", .symbol = "hipDeviceSetLimit" },
    .{ .field = "module_load_data_ex", .symbol = "hipModuleLoadDataEx" },
    .{ .field = "module_load_data", .symbol = "hipModuleLoadData" },
    .{ .field = "module_unload", .symbol = "hipModuleUnload" },
    .{ .field = "module_get_function", .symbol = "hipModuleGetFunction" },
    .{ .field = "module_get_global", .symbol = "hipModuleGetGlobal" },
    .{ .field = "mem_alloc", .symbol = "hipMalloc" },
    .{ .field = "mem_free", .symbol = "hipFree" },
    .{ .field = "memcpy_htod", .symbol = "hipMemcpyHtoD" },
    .{ .field = "memcpy_dtoh", .symbol = "hipMemcpyDtoH" },
    .{ .field = "memset_d8", .symbol = "hipMemsetD8" },
    .{ .field = "stream_create", .symbol = "hipStreamCreate" },
    .{ .field = "stream_destroy", .symbol = "hipStreamDestroy" },
    .{ .field = "stream_synchronize", .symbol = "hipStreamSynchronize" },
    .{ .field = "launch_kernel", .symbol = "hipModuleLaunchKernel" },
    .{ .field = "error_name", .symbol = "hipGetErrorName" },
};

comptime {
    const field_names = @typeInfo(Functions).@"struct".field_names;
    if (field_names.len != function_symbols.len) {
        @compileError("Functions and function_symbols must have a name for every function");
    }
    for (field_names, function_symbols) |field_name, entry| {
        if (!std.mem.eql(u8, field_name, entry.field)) {
            @compileError("function_symbols must name the fields of Functions, in order");
        }
    }
}

/// A loaded HIP runtime.
///
/// The runtime library is loaded and initialized by `open`, and stays loaded until `close`.
/// Everything else in this namespace is created from a driver and keeps a pointer to it: the
/// driver must stay open, and must not move, while a `Device`, `Context`, `Module`, `Function`,
/// `Buffer`, or `Stream` from it is in use.
pub const Driver = struct {
    /// The loaded runtime library, which must stay loaded while anything from this driver is in
    /// use.
    library: Library,
    /// Pointers to the functions of the runtime API that this namespace uses.
    functions: Functions,
    /// The device address of the output buffer of each device, by ordinal, or zero until a module
    /// that writes output is loaded on the device; see `Context.loadModule`.
    output_buffers: []std.atomic.Value(u64),

    /// Loads the runtime library, resolves the functions of the runtime API in it, and
    /// initializes the runtime.
    ///
    /// `error.DriverNotFound` means that there is no HIP runtime on this machine, and
    /// `error.IncompatibleDriver` means that the library is there but does not have the functions
    /// that this namespace uses.
    ///
    /// Loading the library needs either `dlopen` from libc or `ntdll`, so this only runs on
    /// Linux with libc linked and on Windows; every other target reports it as a compile error.
    pub fn open() OpenError!Driver {
        if (comptime driver_supported) {
            var library = Library.open() orelse return error.DriverNotFound;
            errdefer library.close();

            var functions: Functions = undefined;
            inline for (function_symbols) |entry| {
                const pointer = library.lookup(@FieldType(Functions, entry.field), entry.symbol);
                @field(functions, entry.field) = pointer orelse return error.IncompatibleDriver;
            }

            var driver: Driver = .{ .library = library, .functions = functions, .output_buffers = &.{} };
            try check(driver.functions.init(0));
            var count: c_int = 0;
            switch (driver.functions.device_get_count(&count)) {
                success => {},
                // `hipErrorNoDevice`, which is how the runtime reports zero devices.
                100 => count = 0,
                else => |result| return errorFor(result),
            }
            driver.output_buffers = std.heap.page_allocator.alloc(std.atomic.Value(u64), @intCast(count)) catch
                return error.OutOfMemory;
            @memset(driver.output_buffers, .init(0));
            return driver;
        } else if (builtin.os.tag != .linux and builtin.os.tag != .windows) {
            @compileError("std.gpu.hip is only available on Linux and Windows");
        } else {
            @compileError("std.gpu.hip needs libc to load the HIP runtime library with dlopen; link the program with libc");
        }
    }

    /// Frees the output buffers of the devices and unloads the runtime library. The driver and
    /// everything created from it must not be used afterwards.
    pub fn close(driver: *Driver) void {
        for (driver.output_buffers) |*output| {
            const address = output.load(.acquire);
            if (address != 0) _ = driver.functions.mem_free(@ptrFromInt(address));
        }
        std.heap.page_allocator.free(driver.output_buffers);
        driver.library.close();
        driver.* = undefined;
    }

    /// The version of the HIP runtime, such as `.{ .major = 6, .minor = 3, .patch = 42560 }` for
    /// HIP 6.3.42560.
    ///
    /// The runtime reports its version as one number,
    /// `major * 10_000_000 + minor * 100_000 + patch`, and these are its three parts. It is the
    /// version of the runtime library rather than of a driver for one device, so it says nothing
    /// about what a device supports: `Device.archName` is what says that, and what a code object
    /// must be built for.
    pub fn version(driver: *const Driver) Error!Version {
        var encoded: c_int = undefined;
        try check(driver.functions.runtime_get_version(&encoded));
        const value: u32 = @intCast(encoded);
        return .{
            .major = value / 10_000_000,
            .minor = value / 100_000 % 100,
            .patch = value % 100_000,
        };
    }

    /// The number of devices that kernels can run on, the highest ordinal accepted by `device`
    /// plus one.
    pub fn deviceCount(driver: *const Driver) Error!u32 {
        var count: c_int = undefined;
        try check(driver.functions.device_get_count(&count));
        return @intCast(count);
    }

    /// The device with the given ordinal, which is between 0 and `deviceCount() - 1`.
    pub fn device(driver: *const Driver, ordinal: u32) Error!Device {
        var handle: c_int = undefined;
        const ordinal_handle = std.math.cast(c_int, ordinal) orelse return error.InvalidDevice;
        try check(driver.functions.device_get(&handle, ordinal_handle));
        return .{ .driver = driver, .handle = handle };
    }
};

/// A GPU that kernels can run on, selected by ordinal from a `Driver`.
pub const Device = struct {
    /// The driver that owns this device, which must outlive it.
    driver: *const Driver,
    /// The handle of the device in the runtime API, a `hipDevice_t`.
    handle: c_int,

    /// Copies the name of the device, such as "AMD Radeon(TM) Graphics", into `buffer` and
    /// returns it as a slice of `buffer`, without the terminating null byte. A name that does not
    /// fit is truncated to `buffer.len - 1` bytes.
    ///
    /// The buffer must have room for at least one byte of the name and the null byte after it:
    /// the runtime does not accept a maximum length of zero, and `error.InvalidValue` reports a
    /// buffer that is too small.
    pub fn name(dev: Device, buffer: []u8) Error![]const u8 {
        if (buffer.len < 2) return error.InvalidValue;
        const len = std.math.cast(c_int, buffer.len - 1) orelse return error.InvalidValue;
        try check(dev.driver.functions.device_get_name(buffer.ptr, len, dev.handle));
        return std.mem.sliceTo(buffer, 0);
    }

    /// The amount of memory on the device, in bytes.
    pub fn totalMemory(dev: Device) Error!usize {
        var bytes: usize = undefined;
        try check(dev.driver.functions.device_total_memory(&bytes, dev.handle));
        return bytes;
    }

    /// Reads one of the attributes of the device, such as the number of threads in a block.
    pub fn attribute(dev: Device, attr: Attribute) Error!i32 {
        var value: c_int = undefined;
        try check(dev.driver.functions.device_get_attribute(&value, @backingInt(attr), dev.handle));
        return value;
    }

    /// The compute capability of the device as the runtime reports it, which names its
    /// architecture in the terms that CUDA uses.
    ///
    /// A code object cannot be built from this: the compute capability groups several
    /// architectures under one version, and a code object holds the code of one architecture
    /// only. Use `archName` for that, which is what the `-mcpu` of the compiler takes.
    pub fn computeCapability(dev: Device) Error!ComputeCapability {
        return .{
            .major = @intCast(try dev.attribute(.compute_capability_major)),
            .minor = @intCast(try dev.attribute(.compute_capability_minor)),
        };
    }

    /// Copies the architecture name of the device, such as "gfx1036" or
    /// "gfx90a:sramecc+:xnack-", into `buffer` and returns it as a slice of `buffer`, without the
    /// terminating null byte.
    ///
    /// This is the architecture that the code objects of the device must be built for. The part
    /// before the first ':' is the name that `-mcpu` takes, and the parts after it are settings of
    /// target features that the code object must match, which `-mcpu` takes as `+feature` and
    /// `-feature`: "gfx90a:sramecc+:xnack-" is `-mcpu=gfx90a+sramecc-xnack`. Unlike CUDA, which
    /// compiles PTX for the device at load time, every AMD code object runs on the architectures
    /// it was compiled for and on no others, and the runtime reports a code object that does not
    /// match the device as `error.NoBinaryForGpu`.
    ///
    /// The buffer must have room for the whole name and the null byte after it; a smaller buffer
    /// reports `error.InvalidValue` rather than a truncated name, which would select the wrong
    /// code object. 256 bytes is enough for any name that the runtime reports, and
    /// `error.Unexpected` reports a device whose properties carry no architecture name at all.
    pub fn archName(dev: Device, buffer: []u8) Error![]const u8 {
        var properties: DeviceProperties = undefined;
        try dev.readProperties(&properties);
        const arch_name = std.mem.sliceTo(&properties.gcn_arch_name, 0);
        if (arch_name.len == 0) return error.Unexpected;
        if (arch_name.len + 1 > buffer.len) return error.InvalidValue;
        @memcpy(buffer[0..arch_name.len], arch_name);
        buffer[arch_name.len] = 0;
        return buffer[0..arch_name.len];
    }

    /// Fills `properties` with what the runtime reports for the device, which `archName` reads.
    fn readProperties(dev: Device, properties: *DeviceProperties) Error!void {
        try check(dev.driver.functions.device_get_properties(properties, dev.handle));
    }

    /// Retains the primary context of the device and makes it current on the calling thread.
    ///
    /// The primary context is the one that the runtime keeps for the device; retaining it gives
    /// this thread a share of it. Every call must be paired with `Context.release`, and the
    /// context of a device must be the current one on a thread before the thread uses it.
    pub fn retainPrimaryContext(dev: Device) Error!Context {
        var handle: ?*anyopaque = null;
        try check(dev.driver.functions.device_primary_ctx_retain(&handle, dev.handle));
        const context: Context = .{ .driver = dev.driver, .handle = handle.?, .device = dev };
        errdefer context.release();
        try context.makeCurrent();
        return context;
    }
};

/// A limit on the resources that a context allocates, set with `Context.setLimit`.
pub const Limit = enum(c_int) {
    /// `hipLimitStackSize`: the size of the stack of a thread, in bytes, which device functions
    /// that call other functions use for their frames, up to 128K - 16.
    stack_size = 0,
    /// `hipLimitPrintfFifoSize`: the size of the buffer of the `printf` of HIP C++, in bytes, for
    /// every thread of a block. The AMD runtime does not support this limit and reports it as
    /// `error.UnsupportedLimit`. `std.gpu.print` does not use that buffer; see `Context.synchronize`.
    printf_fifo_size = 1,
    /// `hipLimitMallocHeapSize`: the size of the heap of the device-side `malloc` of HIP C++, in
    /// bytes, for the whole device.
    malloc_heap_size = 2,
    _,
};

/// A primary context of a device, retained by `Device.retainPrimaryContext`.
///
/// A context holds the device memory and modules of a thread, and must be the current one on the
/// thread that uses them.
pub const Context = struct {
    /// The driver that owns this context, which must outlive it.
    driver: *const Driver,
    /// The handle of the context in the runtime API, a `hipCtx_t`.
    handle: *anyopaque,
    /// The device that this context belongs to.
    device: Device,

    /// Releases this thread's share of the primary context, one for every call to
    /// `Device.retainPrimaryContext`. The context must not be used afterwards, and the memory and
    /// modules of the context stay alive while other threads still hold it.
    ///
    /// Errors are not reported, because the release cannot fail in a way that the caller can act
    /// on: the primary context of a valid device always exists.
    pub fn release(ctx: Context) void {
        _ = ctx.driver.functions.device_primary_ctx_release(ctx.device.handle);
    }

    /// Makes the context current on the calling thread, so that the runtime uses it for the
    /// memory, modules, and launches of that thread.
    pub fn makeCurrent(ctx: Context) Error!void {
        try check(ctx.driver.functions.ctx_set_current(ctx.handle));
    }

    /// Waits until every copy and every kernel launch in the context has finished, on every
    /// stream. The context must be current on the calling thread.
    ///
    /// Then writes what the kernels of the device wrote with `std.gpu.print` to standard output,
    /// and the messages of `std.gpu.assertFail` to standard error, in the order in which the
    /// threads wrote them. When a thread called `std.gpu.assertFail`, this returns
    /// `error.Assert`; the context can still run kernels afterwards. The output is written with
    /// `std.Options.debug_io`, like `std.debug.print`, and threads that synchronize the same
    /// device at the same time may write parts of each other's output.
    ///
    /// The runtime has no call of its own for this: `hipCtxSynchronize` is a CUDA compatibility
    /// entry point that the AMD runtime reports as `error.NotSupported`, so this waits for the
    /// current device with `hipDeviceSynchronize`, which is the device of the context.
    pub fn synchronize(ctx: Context) Error!void {
        const result = ctx.driver.functions.device_synchronize();
        const asserted = ctx.writeOutput() catch |err| {
            try check(result);
            return err;
        };
        try check(result);
        if (asserted) return error.Assert;
    }

    /// Sets a limit on a resource that the context allocates.
    ///
    /// A limit must be set before the resources it governs are used: the stack size and the
    /// printf buffer take effect for the next kernel launch, and the heap size for the next
    /// device-side allocation.
    pub fn setLimit(ctx: Context, limit: Limit, value: usize) Error!void {
        try check(ctx.driver.functions.device_set_limit(@backingInt(limit), value));
    }

    /// Loads a module from an image of a code object and makes its kernels available to this
    /// context.
    ///
    /// `image` is the whole ELF shared object of a code object, such as the output of
    /// `zig build-lib -dynamic -target amdgcn-amdhsa -mcpu=gfx1036`, which needs no null byte on
    /// the end. The runtime reads the image while it loads it; the memory of the image can be
    /// freed afterwards.
    ///
    /// The code object is machine code for one architecture, so it loads only in a context whose
    /// device is of the architecture that it was compiled for, and any other device reports
    /// `error.NoBinaryForGpu`. Compare `Device.archName`, which says what the device runs, with
    /// the `-mcpu` that built the code object before loading it.
    ///
    /// A code object that calls `std.gpu.print` or `std.gpu.assertFail` gets the output buffer
    /// of the device, which `synchronize` writes out: the first such module allocates it, 1 MiB
    /// of device memory, and `Driver.close` frees it.
    pub fn loadModule(ctx: Context, image: []const u8, options: ModuleOptions) Error!Module {
        var handle: ?*anyopaque = null;
        if (options.error_log) |log| {
            const log_size = std.math.cast(c_uint, log.len) orelse return error.InvalidValue;
            const option_names = [_]c_int{ jit_error_log_buffer, jit_error_log_buffer_size_bytes };
            // The value of an option that is not a pointer passes as the value itself, cast to a
            // pointer, so the size of the log passes as that number rather than its address.
            var option_values = [_]?*anyopaque{ log.ptr, @ptrFromInt(@as(usize, log_size)) };
            try check(ctx.driver.functions.module_load_data_ex(
                &handle,
                image.ptr,
                option_names.len,
                &option_names,
                &option_values,
            ));
        } else {
            try check(ctx.driver.functions.module_load_data(&handle, image.ptr));
        }
        const module: Module = .{ .driver = ctx.driver, .handle = handle.? };
        errdefer module.unload();
        try ctx.connectOutput(module);
        return module;
    }

    /// Points the pointer to the output buffer that a code object exports when it calls
    /// `std.gpu.print` or `std.gpu.assertFail` at the output buffer of the device.
    fn connectOutput(ctx: Context, module: Module) Error!void {
        var pointer: ?*anyopaque = null;
        var size: usize = 0;
        switch (ctx.driver.functions.module_get_global(&pointer, &size, module.handle, output_buffer.symbol)) {
            success => {},
            // `hipErrorNotFound`: the code object writes no output.
            500 => return,
            else => |result| return errorFor(result),
        }
        if (size != @sizeOf(u64)) return error.InvalidImage;
        const address = try ctx.outputBuffer() orelse return;
        try check(ctx.driver.functions.memcpy_htod(pointer, &address, @sizeOf(u64)));
    }

    /// The device address of the output buffer of the context's device, allocated by the first
    /// call, or null for a device whose ordinal `Driver.open` did not count.
    fn outputBuffer(ctx: Context) Error!?u64 {
        const output = ctx.outputSlot() orelse return null;
        const existing = output.load(.acquire);
        if (existing != 0) return existing;
        var allocation: ?*anyopaque = null;
        try check(ctx.driver.functions.mem_alloc(&allocation, @sizeOf(output_buffer.Header) + output_capacity));
        errdefer _ = ctx.driver.functions.mem_free(allocation);
        const header: output_buffer.Header = .{ .claimed = 0, .capacity = output_capacity };
        try check(ctx.driver.functions.memcpy_htod(allocation, &header, @sizeOf(output_buffer.Header)));
        const address: u64 = @intFromPtr(allocation orelse return error.OutOfMemory);
        // Another thread may have allocated one for the device at the same time.
        if (output.cmpxchgStrong(0, address, .acq_rel, .acquire)) |winner| {
            _ = ctx.driver.functions.mem_free(allocation);
            return winner;
        }
        return address;
    }

    fn outputSlot(ctx: Context) ?*std.atomic.Value(u64) {
        const ordinal = std.math.cast(usize, ctx.device.handle) orelse return null;
        if (ordinal >= ctx.driver.output_buffers.len) return null;
        return &ctx.driver.output_buffers[ordinal];
    }

    /// Writes the records of the output buffer of the context's device and empties the buffer.
    /// Returns whether one of them was the message of an assertion.
    fn writeOutput(ctx: Context) Error!bool {
        const Header = output_buffer.Header;
        const Record = output_buffer.Record;
        const functions = &ctx.driver.functions;
        const output = ctx.outputSlot() orelse return false;
        const address = output.load(.acquire);
        if (address == 0) return false;

        var header: Header = undefined;
        try check(functions.memcpy_dtoh(&header, @ptrFromInt(address), @sizeOf(Header)));
        if (header.claimed == 0) return false;
        const end = @min(header.claimed, header.capacity);
        const records = address + @sizeOf(Header);

        const io = std.Options.debug_io;
        var asserted = false;
        // A record has at most 256 bytes of text, so a chunk always holds the first record in it.
        var chunk: [4096]u8 = undefined;
        var offset: u64 = 0;
        read: while (offset < end and end - offset >= @sizeOf(Record)) {
            const len: usize = @intCast(@min(chunk.len, end - offset));
            try check(functions.memcpy_dtoh(&chunk, @ptrFromInt(records + offset), len));
            var at: usize = 0;
            while (at < len and len - at >= @sizeOf(Record)) {
                const record = std.mem.bytesToValue(Record, chunk[at..][0..@sizeOf(Record)]);
                // Not a record that a kernel wrote, such as memory that a faulting kernel
                // overwrote: nothing after it can be trusted either.
                if (record.size < @sizeOf(Record) or record.len > record.size - @sizeOf(Record)) break :read;
                const text_start = at + @sizeOf(Record);
                // The text goes on past the chunk: read the next chunk from this record on.
                if (record.len > len - text_start and offset + len < end) break;
                const text = chunk[text_start..][0..@min(record.len, len - text_start)];
                switch (record.kind) {
                    .print => std.Io.File.stdout().writeStreamingAll(io, text) catch {},
                    .assert => {
                        asserted = true;
                        var line_buffer: [512]u8 = undefined;
                        const line = std.fmt.bufPrint(&line_buffer, "block: [{d},{d},{d}], thread: [{d},{d},{d}] Assertion `{s}` failed.\n", .{
                            record.block[0],  record.block[1],  record.block[2],
                            record.thread[0], record.thread[1], record.thread[2],
                            text,
                        }) catch &line_buffer;
                        std.Io.File.stderr().writeStreamingAll(io, line) catch {};
                    },
                    _ => {},
                }
                at += record.size;
            }
            if (at == 0) break;
            offset += at;
        }

        const empty: u64 = 0;
        try check(functions.memcpy_htod(@ptrFromInt(address), &empty, @sizeOf(u64)));
        return asserted;
    }

    /// Allocates a buffer of `len` elements of `T` in device memory.
    ///
    /// The memory stays allocated until `Buffer.free`, and is only in this context. `len` must be
    /// at least 1: a request for an empty buffer succeeds with a null address, which no kernel
    /// and no copy can be given, so zero reports `error.InvalidValue` here.
    pub fn alloc(ctx: Context, comptime T: type, len: usize) Error!Buffer(T) {
        if (comptime @sizeOf(T) == 0) @compileError("cannot allocate a Buffer of the zero-sized type '" ++ @typeName(T) ++ "'");
        if (len == 0) return error.InvalidValue;
        const bytes = std.math.mul(usize, @sizeOf(T), len) catch return error.OutOfMemory;
        var address: ?*anyopaque = null;
        try check(ctx.driver.functions.mem_alloc(&address, bytes));
        const ptr: DevicePtr = @fromBackingInt(@intFromPtr(address orelse return error.OutOfMemory));
        return .{ .driver = ctx.driver, .ptr = ptr, .len = len };
    }

    /// Creates a stream: a queue of copies and kernel launches that other streams do not wait
    /// for. Work on the null stream, the one a `LaunchConfig` uses by default, is ordered
    /// against the work of every other stream.
    pub fn createStream(ctx: Context) Error!Stream {
        var handle: ?*anyopaque = null;
        try check(ctx.driver.functions.stream_create(&handle));
        return .{ .driver = ctx.driver, .handle = handle.? };
    }
};

/// The number of bytes of records that the output buffer of a device holds between two calls to
/// `Context.synchronize`, like the 1 MiB that the CUDA driver gives `printf` by default.
const output_capacity = 1 << 20;

/// The `hipJitOptionErrorLogBuffer` option of `hipModuleLoadDataEx`: a pointer to the buffer that
/// receives the errors of the loader.
const jit_error_log_buffer: c_int = 5;

/// The `hipJitOptionErrorLogBufferSizeBytes` option of `hipModuleLoadDataEx`: the size of that
/// buffer. Options whose value is not a pointer pass the value itself, as a pointer.
const jit_error_log_buffer_size_bytes: c_int = 6;

/// Options for `Context.loadModule`.
pub const ModuleOptions = struct {
    /// When the module fails to load, receives the messages that explain what is wrong with the
    /// image, if the runtime provides them.
    ///
    /// The AMD runtime accepts the options of `hipModuleLoadDataEx` and ignores them: it loads
    /// the image as `hipModuleLoadData` does, and leaves the buffer as it was, so the error of a
    /// failed load is only in the `Error` of the call.
    error_log: ?[]u8 = null,
};

/// A module of compiled code, loaded from an image of a code object by `Context.loadModule`.
pub const Module = struct {
    /// The driver that owns this module, which must outlive it.
    driver: *const Driver,
    /// The handle of the module in the runtime API, a `hipModule_t`.
    handle: *anyopaque,

    /// Unloads the module and its code from the device. Its `Function`s must not be used
    /// afterwards.
    ///
    /// Errors are not reported: the handle comes from a successful load, and the runtime only
    /// fails to unload a module that another thread is destroying at the same time.
    pub fn unload(module: Module) void {
        _ = module.driver.functions.module_unload(module.handle);
    }

    /// Looks up a kernel in the module by the name it was exported with.
    pub fn function(module: Module, name: [:0]const u8) Error!Function {
        var handle: ?*anyopaque = null;
        try check(module.driver.functions.module_get_function(&handle, module.handle, name.ptr));
        return .{ .driver = module.driver, .handle = handle.? };
    }
};

/// A kernel function of a module, launched by `launch`.
pub const Function = struct {
    /// The driver that owns this function, which must outlive it.
    driver: *const Driver,
    /// The handle of the function in the runtime API, a `hipFunction_t`.
    handle: *anyopaque,

    /// Runs the kernel on `config.grid` blocks of `config.block` threads.
    ///
    /// `args` is a tuple with one element for every parameter of the kernel, in order, such as
    /// `.{ buffer, @as(u32, count) }`. An argument passes in one of these ways:
    /// * a `Buffer` passes the address of its device memory, which the kernel receives in a
    ///   pointer parameter such as `[*]f32`,
    /// * a `DevicePtr` passes as it is,
    /// * integers, floats, `bool`, enums, vectors, and `extern` and `packed` structs pass by
    ///   value.
    ///
    /// Slices and host pointers cannot be passed, because kernels cannot read the memory of the
    /// host process: put the data in a `Buffer` and pass that, or pass a `DevicePtr`. Neither can
    /// a `comptime_int` or a `comptime_float`, because the compiler has not chosen a type for
    /// it: write the type, as in `@as(u32, 256)`. Each of these is a compile error.
    pub fn launch(function: Function, config: LaunchConfig, args: anytype) Error!void {
        const Arguments = @TypeOf(args);
        const field_types = comptime launch_arguments.argumentTypes(Arguments);
        if (comptime field_types.len == 0) return function.launchRaw(config, null);

        var storage: launch_arguments.ArgumentTuple(Buffer, Arguments) = undefined;
        var parameters: [field_types.len]?*anyopaque = undefined;
        inline for (field_types, 0..) |field_type, index| {
            storage[index] = launch_arguments.argumentValue(Buffer, field_type, index, args[index]);
            parameters[index] = @ptrCast(&storage[index]);
        }
        return function.launchRaw(config, @ptrCast(&parameters));
    }

    /// Passes the launch to the runtime with `parameters` holding the address of every argument,
    /// in the order that the kernel declares them.
    fn launchRaw(function: Function, config: LaunchConfig, parameters: ?[*]?*anyopaque) Error!void {
        try check(function.driver.functions.launch_kernel(
            function.handle,
            config.grid.x,
            config.grid.y,
            config.grid.z,
            config.block.x,
            config.block.y,
            config.block.z,
            config.shared_memory,
            if (config.stream) |stream| stream.handle else null,
            parameters,
            null,
        ));
    }
};

/// The size of a grid or a block in each of the three dimensions.
pub const Dim3 = struct {
    /// The number of blocks or threads in the x dimension.
    x: u32 = 1,
    /// The number of blocks or threads in the y dimension.
    y: u32 = 1,
    /// The number of blocks or threads in the z dimension.
    z: u32 = 1,
};

/// How `Function.launch` arranges the threads of a kernel in a grid of blocks.
pub const LaunchConfig = struct {
    /// The number of blocks in each dimension of the grid.
    grid: Dim3 = .{},
    /// The number of threads in each dimension of a block. All blocks have the same shape.
    block: Dim3 = .{},
    /// The number of bytes of dynamic shared memory for every block, in addition to the shared
    /// memory that the kernel declares itself. The kernel receives the address of the memory
    /// through the `extern .shared` array that it declares.
    shared_memory: u32 = 0,
    /// The stream to run the kernel on, or null for the null stream, which the runtime orders
    /// against the work of every other stream.
    stream: ?Stream = null,

    /// A one-dimensional launch of at least `n` threads, in blocks of `block_size` threads.
    ///
    /// The grid has as many blocks as it takes to cover `n` threads, rounded up, so the last
    /// block may be partial and the kernel must compare its thread index against `n`.
    /// `block_size` must be at least 1. `n == 0` gives a grid of zero blocks, which the runtime
    /// rejects with `error.InvalidValue`, so do not launch when `n` is zero.
    pub fn linear(n: u32, block_size: u32) LaunchConfig {
        std.debug.assert(block_size != 0);
        return .{
            .grid = .{ .x = if (n == 0) 0 else (n - 1) / block_size + 1 },
            .block = .{ .x = block_size },
        };
    }
};

/// The address of memory on a device, as a kernel receives it in a pointer parameter such as
/// `[*]f32`. A `Buffer` holds one of these, and `Function.launch` passes it without change.
pub const DevicePtr = enum(u64) {
    _,
};

/// A region of device memory that holds `len` elements of `T`, allocated by `Context.alloc`.
///
/// Kernels receive the address of the memory in a pointer parameter, and `Function.launch`
/// passes a buffer as its address, so a launch takes the buffer itself:
/// `.{ buffer, @as(u32, buffer.len) }`.
pub fn Buffer(comptime T: type) type {
    return struct {
        /// The driver that owns this memory, which must outlive the buffer.
        driver: *const Driver,
        /// The address of the memory, which the kernel receives in a pointer parameter.
        ptr: DevicePtr,
        /// The number of `T` elements that the memory holds.
        len: usize,

        /// The type of the elements of the buffer.
        pub const Elem = T;

        /// Frees the memory on the device. The buffer must not be used afterwards.
        ///
        /// Errors are not reported: the memory of a valid buffer can only fail to be freed if
        /// the runtime has already lost the context, and there is nothing left to free then.
        pub fn free(buffer: @This()) void {
            _ = buffer.driver.functions.mem_free(@ptrFromInt(@backingInt(buffer.ptr)));
        }

        /// Copies `src` into the buffer, starting at the first element. `src` may be shorter
        /// than the buffer, but not longer; a longer `src` fails an assertion.
        pub fn copyFromHost(buffer: @This(), src: []const T) Error!void {
            std.debug.assert(src.len <= buffer.len);
            try check(buffer.driver.functions.memcpy_htod(
                @ptrFromInt(@backingInt(buffer.ptr)),
                @ptrCast(src.ptr),
                @sizeOf(T) * src.len,
            ));
        }

        /// Copies the buffer into `dst`, starting at the first element. `dst` may be shorter
        /// than the buffer, but not longer; a longer `dst` fails an assertion.
        pub fn copyToHost(buffer: @This(), dst: []T) Error!void {
            std.debug.assert(dst.len <= buffer.len);
            try check(buffer.driver.functions.memcpy_dtoh(
                @ptrCast(dst.ptr),
                @ptrFromInt(@backingInt(buffer.ptr)),
                @sizeOf(T) * dst.len,
            ));
        }

        /// Fills the whole buffer with zero bytes, whatever `T` is.
        pub fn zero(buffer: @This()) Error!void {
            try check(buffer.driver.functions.memset_d8(@ptrFromInt(@backingInt(buffer.ptr)), 0, @sizeOf(T) * buffer.len));
        }
    };
}

/// A queue of copies and kernel launches, created by `Context.createStream`.
///
/// Streams run at the same time as each other, and the runtime decides the order of the work in
/// them; the null stream, which a `LaunchConfig` uses by default, is ordered against every
/// other stream.
pub const Stream = struct {
    /// The driver that owns this stream, which must outlive it.
    driver: *const Driver,
    /// The handle of the stream in the runtime API, a `hipStream_t`.
    handle: *anyopaque,

    /// Destroys the stream. The work queued on it must have finished, and the stream must not be
    /// used afterwards.
    ///
    /// Errors are not reported: a stream that was created cannot fail to be destroyed.
    pub fn destroy(stream: Stream) void {
        _ = stream.driver.functions.stream_destroy(stream.handle);
    }

    /// Waits until everything queued on the stream has finished. The output of `std.gpu.print`
    /// and `std.gpu.assertFail` stays in the output buffer until `Context.synchronize`.
    pub fn synchronize(stream: Stream) Error!void {
        try check(stream.driver.functions.stream_synchronize(stream.handle));
    }
};

/// The name that the runtime has for a driver status, such as "hipErrorNoBinaryForGpu", which
/// says what a failed call reported more precisely than the `Error` it maps to.
fn statusName(driver: *const Driver, result: Result) []const u8 {
    return std.mem.sliceTo(driver.functions.error_name(result), 0);
}

test "hip: LaunchConfig.linear rounds the thread count up to whole blocks" {
    // No threads is an empty grid, with no blocks to launch.
    const none = LaunchConfig.linear(0, 128);
    try testing.expectEqual(@as(u32, 0), none.grid.x);
    try testing.expectEqual(@as(u32, 128), none.block.x);

    // A thread count that is a multiple of the block size fills the last block exactly.
    const exact = LaunchConfig.linear(512, 128);
    try testing.expectEqual(@as(u32, 4), exact.grid.x);
    try testing.expectEqual(@as(u32, 128), exact.block.x);
    try testing.expectEqual(@as(u32, 1), exact.grid.y);
    try testing.expectEqual(@as(u32, 1), exact.grid.z);
    try testing.expectEqual(@as(u32, 1), exact.block.y);
    try testing.expectEqual(@as(u32, 1), exact.block.z);

    // One thread more needs a whole block more, and leaves it partial.
    const partial = LaunchConfig.linear(513, 128);
    try testing.expectEqual(@as(u32, 5), partial.grid.x);

    const single = LaunchConfig.linear(1, 128);
    try testing.expectEqual(@as(u32, 1), single.grid.x);

    // One thread per block is one block per thread.
    const per_thread = LaunchConfig.linear(7, 1);
    try testing.expectEqual(@as(u32, 7), per_thread.grid.x);
    try testing.expectEqual(@as(u32, 1), per_thread.block.x);

    // The largest thread count is covered without overflowing the block count.
    const all = LaunchConfig.linear(0xffff_ffff, 32);
    const covered = @as(u64, all.grid.x) * 32;
    try testing.expect(covered >= 0xffff_ffff);
    try testing.expect(covered - 32 < 0xffff_ffff);
}

test "hip: the errors of the runtime map to the members of Error" {
    if (!driver_supported) return error.SkipZigTest;

    var driver = Driver.open() catch |err| switch (err) {
        error.DriverNotFound, error.NoDevice => return error.SkipZigTest,
        else => |other| return other,
    };
    defer driver.close();

    // Every code of the runtime has one member of `Error`, named after it: the runtime's own
    // name for the code, with the "hipError" in front of it, is the name of the member.
    const codes = [_]struct { code: Result, expected: Error }{
        .{ .code = 1, .expected = error.InvalidValue },
        .{ .code = 2, .expected = error.OutOfMemory },
        .{ .code = 100, .expected = error.NoDevice },
        .{ .code = 101, .expected = error.InvalidDevice },
        .{ .code = 209, .expected = error.NoBinaryForGpu },
        .{ .code = 215, .expected = error.UnsupportedLimit },
        .{ .code = 710, .expected = error.Assert },
        .{ .code = 801, .expected = error.NotSupported },
        .{ .code = 999, .expected = error.Unknown },
    };
    inline for (codes) |entry| {
        const name = statusName(&driver, entry.code);
        try testing.expect(std.mem.startsWith(u8, name, "hipError"));
        try testing.expectEqualStrings(@errorName(entry.expected), name["hipError".len..]);
        try testing.expectEqual(entry.expected, errorFor(entry.code));
    }

    // A code that the runtime has no name for is not one that this namespace knows either.
    try testing.expectEqual(Error.Unexpected, errorFor(722));
}

test "hip: open the runtime and query the device" {
    if (!driver_supported) return error.SkipZigTest;

    var driver = Driver.open() catch |err| switch (err) {
        error.DriverNotFound, error.NoDevice => return error.SkipZigTest,
        else => |other| return other,
    };
    defer driver.close();

    // The runtime reports its version as major * 10_000_000 + minor * 100_000 + patch.
    const version = try driver.version();
    try testing.expect(version.major >= 1);
    try testing.expect(version.minor < 100);

    const device_count = driver.deviceCount() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => |other| return other,
    };
    if (device_count == 0) return error.SkipZigTest;
    const device = try driver.device(0);

    var name_buffer: [256]u8 = undefined;
    const name = try device.name(&name_buffer);
    try testing.expect(name.len > 0);

    // The architecture name is what a code object is built for, so it must be there.
    var arch_buffer: [256]u8 = undefined;
    const arch_name = try device.archName(&arch_buffer);
    try testing.expect(std.mem.startsWith(u8, arch_name, "gfx"));

    // `archName` reads the architecture out of the device properties, whose fields come before it
    // in the order of `hipDeviceProp_tR0600`; the name is the first of them, so it says whether
    // the properties were placed where the namespace expects them.
    var properties: DeviceProperties = undefined;
    try device.readProperties(&properties);
    try testing.expectEqualStrings(name, std.mem.sliceTo(&properties.name, 0));

    try testing.expect(try device.totalMemory() > 0);
    try testing.expect(try device.attribute(.warp_size) > 0);
    try testing.expect(try device.attribute(.max_threads_per_block) > 0);
    const capability = try device.computeCapability();
    try testing.expect(capability.major > 0);

    // All of the device, the modules, and the memory of a program live in the primary context of
    // its device, which is current on the threads that retain it.
    const context = try device.retainPrimaryContext();
    defer context.release();
    try context.makeCurrent();
    try context.setLimit(.stack_size, 1 << 13);
    try context.synchronize();

    // A stream is a second queue of work, and needs nothing but the context to create.
    const stream = try context.createStream();
    defer stream.destroy();
    try stream.synchronize();

    // An empty allocation is not one: the runtime answers it with a null address rather than an
    // error, which is not something a launch or a copy can be given.
    try testing.expectError(error.InvalidValue, context.alloc(u32, 0));

    // Memory of the device is allocated in the context, and comes back out of it the same way.
    var data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const buffer = try context.alloc(u32, data.len);
    defer buffer.free();
    try buffer.copyFromHost(&data);
    try buffer.zero();
    try buffer.copyToHost(&data);
    for (data) |element| try testing.expectEqual(@as(u32, 0), element);
}

test "hip: an image that is not a code object is rejected" {
    if (!driver_supported) return error.SkipZigTest;

    var driver = Driver.open() catch |err| switch (err) {
        error.DriverNotFound, error.NoDevice => return error.SkipZigTest,
        else => |other| return other,
    };
    defer driver.close();

    const device = driver.device(0) catch |err| switch (err) {
        error.NoDevice, error.InvalidDevice => return error.SkipZigTest,
        else => |other| return other,
    };
    const context = try device.retainPrimaryContext();
    defer context.release();

    // The runtime reads the ELF header of a code object, so an image that is not one is rejected
    // before any code runs, whatever the error log says: the AMD runtime ignores the options of
    // `hipModuleLoadDataEx` and does not put anything in the buffer.
    var not_a_code_object: [256]u8 = @splat(0);
    not_a_code_object[0] = 0x7f;
    var error_log: [1024]u8 = @splat(0);
    if (context.loadModule(&not_a_code_object, .{ .error_log = &error_log })) |module| {
        module.unload();
        std.debug.print("hip: the runtime loaded an image that is not a code object\n", .{});
        return error.TestUnexpectedResult;
    } else |err| switch (err) {
        error.InvalidImage, error.InvalidKernelFile => {},
        else => |other| return other,
    }
}
