//! Host-side access to NVIDIA GPUs through the CUDA driver API.
//!
//! The driver library, `libcuda.so.1`, is loaded with `std.DynLib` when it is needed, so a
//! program that uses this namespace needs no CUDA headers, import library, or toolkit to build.
//! Loading the library uses `dlopen`, which needs libc, so link the program with libc; this
//! namespace is available on Linux only.
//!
//! This is the part of the driver that runs kernels: devices and their primary context, modules
//! of PTX assembly, kernel functions, device memory, and streams. The CUDA runtime library,
//! NVRTC, events, and graphs are not part of it.
//!
//! Functions return `Error`, which names the codes that the driver reports, such as
//! `error.InvalidValue` and `error.OutOfDeviceMemory`. The objects of this namespace, `Device`,
//! `Context`, `Module`, `Function`, `Buffer`, and `Stream`, hold a pointer to the `Driver` they
//! came from: the driver must stay open, and must not move, while any of them is in use.
//!
//! ```zig
//! const std = @import("std");
//! const cuda = std.gpu.cuda;
//!
//! // PTX assembly of a kernel, such as the output of `zig build-obj -target nvptx64-cuda`.
//! const ptx: [:0]const u8 = @embedFile("kernel.ptx");
//!
//! pub fn main() !void {
//!     var driver = try cuda.Driver.open();
//!     defer driver.close();
//!
//!     const device = try driver.device(0);
//!     const context = try device.retainPrimaryContext();
//!     defer context.release();
//!
//!     const module = try context.loadModule(ptx, .{});
//!     defer module.unload();
//!     const kernel = try module.function("add_one");
//!
//!     var data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//!     const buffer = try context.alloc(u32, data.len);
//!     defer buffer.free();
//!     try buffer.copyFromHost(&data);
//!     try kernel.launch(cuda.LaunchConfig.linear(data.len, 4), .{ buffer, @as(u32, data.len) });
//!     try context.synchronize();
//!     try buffer.copyToHost(&data);
//! }
//! ```

const std = @import("../std.zig");
const builtin = @import("builtin");
const launch_arguments = @import("launch_arguments.zig");
const testing = std.testing;

/// Whether the driver library can be loaded here. On Linux, `std.DynLib` loads shared libraries
/// with `dlopen`, which is part of libc; the pure-Zig ELF loader cannot load the vendor driver.
const driver_supported = builtin.os.tag == .linux and builtin.link_libc;

/// The file name of the CUDA driver shared library on Linux.
const library_name: [:0]const u8 = "libcuda.so.1";

/// Every function of the driver API returns one of these, a `CUresult`.
const Result = c_int;

/// The result of a driver call that succeeded, `CUDA_SUCCESS`.
const success: Result = 0;

/// The errors that the driver reports, one name per `CUresult` code worth distinguishing.
///
/// The names follow the codes of the driver: `error.InvalidValue` is `CUDA_ERROR_INVALID_VALUE`,
/// `error.NoBinaryForGpu` is `CUDA_ERROR_NO_BINARY_FOR_GPU`, and so on. `error.OutOfDeviceMemory`
/// is `CUDA_ERROR_OUT_OF_MEMORY`, which the driver returns when it cannot allocate device memory
/// or other resources for a call. `error.Unknown` is `CUDA_ERROR_UNKNOWN`, the code the driver
/// uses for a failure that has no more specific code. `error.Unexpected` is a code that this
/// version of the standard library does not know.
pub const Error = error{
    InvalidValue,
    OutOfDeviceMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    StubLibrary,
    CallRequiresNewerDriver,
    DeviceUnavailable,
    NoDevice,
    InvalidDevice,
    DeviceNotLicensed,
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
    EccUncorrectable,
    UnsupportedLimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    JitCompilerNotFound,
    UnsupportedPtxVersion,
    JitCompilationDisabled,
    UnsupportedExecAffinity,
    UnsupportedDeviceSideSync,
    Contained,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidHandle,
    IllegalState,
    LossyQuery,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PrimaryContextActive,
    ContextIsDestroyed,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    LaunchFailed,
    CooperativeLaunchTooLarge,
    TensorMemoryLeak,
    NotPermitted,
    NotSupported,
    SystemNotReady,
    SystemDriverMismatch,
    CompatNotSupportedOnDevice,
    MpsConnectionFailed,
    MpsRpcFailure,
    MpsServerNotReady,
    MpsMaxClientsReached,
    MpsMaxConnectionsReached,
    MpsClientTerminated,
    CdpNotSupported,
    CdpVersionMismatch,
    StreamCaptureUnsupported,
    StreamCaptureInvalidated,
    StreamCaptureMerge,
    StreamCaptureUnmatched,
    StreamCaptureUnjoined,
    StreamCaptureIsolation,
    StreamCaptureImplicit,
    CapturedEvent,
    StreamCaptureWrongThread,
    Timeout,
    GraphExecUpdateFailure,
    ExternalDevice,
    InvalidClusterSize,
    FunctionNotLoaded,
    InvalidResourceType,
    InvalidResourceConfiguration,
    KeyRotation,
    StreamDetached,
    GraphRecaptureFailure,
    Unknown,
    Unexpected,
};

/// The errors that `Driver.open` reports, in addition to the errors of the driver.
pub const OpenError = error{
    /// The driver library was not found, usually because no driver is installed.
    DriverNotFound,
    /// The driver library is loaded, but it does not have the functions of the driver API that
    /// this namespace uses.
    IncompatibleDriver,
} || Error;

/// Converts the `CUresult` of a driver call into an error; `CUDA_SUCCESS` is not an error.
fn check(result: Result) Error!void {
    if (result == success) return;
    return errorFor(result);
}

/// The error for a `CUresult` code other than `CUDA_SUCCESS`.
fn errorFor(result: Result) Error {
    return switch (result) {
        1 => error.InvalidValue,
        2 => error.OutOfDeviceMemory,
        3 => error.NotInitialized,
        4 => error.Deinitialized,
        5 => error.ProfilerDisabled,
        6 => error.ProfilerNotInitialized,
        7 => error.ProfilerAlreadyStarted,
        8 => error.ProfilerAlreadyStopped,
        34 => error.StubLibrary,
        36 => error.CallRequiresNewerDriver,
        46 => error.DeviceUnavailable,
        100 => error.NoDevice,
        101 => error.InvalidDevice,
        102 => error.DeviceNotLicensed,
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
        214 => error.EccUncorrectable,
        215 => error.UnsupportedLimit,
        216 => error.ContextAlreadyInUse,
        217 => error.PeerAccessUnsupported,
        218 => error.InvalidPtx,
        219 => error.InvalidGraphicsContext,
        220 => error.NvlinkUncorrectable,
        221 => error.JitCompilerNotFound,
        222 => error.UnsupportedPtxVersion,
        223 => error.JitCompilationDisabled,
        224 => error.UnsupportedExecAffinity,
        225 => error.UnsupportedDeviceSideSync,
        226 => error.Contained,
        300 => error.InvalidSource,
        301 => error.FileNotFound,
        302 => error.SharedObjectSymbolNotFound,
        303 => error.SharedObjectInitFailed,
        304 => error.OperatingSystem,
        400 => error.InvalidHandle,
        401 => error.IllegalState,
        402 => error.LossyQuery,
        500 => error.NotFound,
        600 => error.NotReady,
        700 => error.IllegalAddress,
        701 => error.LaunchOutOfResources,
        702 => error.LaunchTimeout,
        703 => error.LaunchIncompatibleTexturing,
        704 => error.PeerAccessAlreadyEnabled,
        705 => error.PeerAccessNotEnabled,
        708 => error.PrimaryContextActive,
        709 => error.ContextIsDestroyed,
        710 => error.Assert,
        711 => error.TooManyPeers,
        712 => error.HostMemoryAlreadyRegistered,
        713 => error.HostMemoryNotRegistered,
        714 => error.HardwareStackError,
        715 => error.IllegalInstruction,
        716 => error.MisalignedAddress,
        717 => error.InvalidAddressSpace,
        718 => error.InvalidPc,
        719 => error.LaunchFailed,
        720 => error.CooperativeLaunchTooLarge,
        721 => error.TensorMemoryLeak,
        800 => error.NotPermitted,
        801 => error.NotSupported,
        802 => error.SystemNotReady,
        803 => error.SystemDriverMismatch,
        804 => error.CompatNotSupportedOnDevice,
        805 => error.MpsConnectionFailed,
        806 => error.MpsRpcFailure,
        807 => error.MpsServerNotReady,
        808 => error.MpsMaxClientsReached,
        809 => error.MpsMaxConnectionsReached,
        810 => error.MpsClientTerminated,
        811 => error.CdpNotSupported,
        812 => error.CdpVersionMismatch,
        900 => error.StreamCaptureUnsupported,
        901 => error.StreamCaptureInvalidated,
        902 => error.StreamCaptureMerge,
        903 => error.StreamCaptureUnmatched,
        904 => error.StreamCaptureUnjoined,
        905 => error.StreamCaptureIsolation,
        906 => error.StreamCaptureImplicit,
        907 => error.CapturedEvent,
        908 => error.StreamCaptureWrongThread,
        909 => error.Timeout,
        910 => error.GraphExecUpdateFailure,
        911 => error.ExternalDevice,
        912 => error.InvalidClusterSize,
        913 => error.FunctionNotLoaded,
        914 => error.InvalidResourceType,
        915 => error.InvalidResourceConfiguration,
        916 => error.KeyRotation,
        917 => error.StreamDetached,
        918 => error.GraphRecaptureFailure,
        999 => error.Unknown,
        else => error.Unexpected,
    };
}

/// The version of a CUDA driver, such as `.{ .major = 12, .minor = 4 }` for CUDA 12.4.
pub const Version = struct {
    /// The major version, the first number of the version: 13 for CUDA 13.4.
    major: u32,
    /// The minor version, the number after the point: 4 for CUDA 13.4.
    minor: u32,
};

/// The compute capability of a device: the major and minor version of the instruction set it
/// runs, such as `.{ .major = 12, .minor = 0 }` for an RTX 5090.
pub const ComputeCapability = struct {
    /// The major version, the first number of the compute capability: 12 for sm_120.
    major: u32,
    /// The minor version, the number after the point: 0 for sm_120.
    minor: u32,
};

/// The attributes of a device that `Device.attribute` reads, one name per
/// `CU_DEVICE_ATTRIBUTE_*` value of the driver.
pub const Attribute = enum(c_int) {
    /// Maximum number of threads per block
    max_threads_per_block = 1,
    /// Maximum block dimension X
    max_block_dim_x = 2,
    /// Maximum block dimension Y
    max_block_dim_y = 3,
    /// Maximum block dimension Z
    max_block_dim_z = 4,
    /// Maximum grid dimension X
    max_grid_dim_x = 5,
    /// Maximum grid dimension Y
    max_grid_dim_y = 6,
    /// Maximum grid dimension Z
    max_grid_dim_z = 7,
    /// Maximum shared memory available per block in bytes
    max_shared_memory_per_block = 8,
    /// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
    total_constant_memory = 9,
    /// Warp size in threads
    warp_size = 10,
    /// Maximum pitch in bytes allowed by memory copies
    max_pitch = 11,
    /// Maximum number of 32-bit registers available per block
    max_registers_per_block = 12,
    /// Typical clock frequency in kilohertz
    clock_rate = 13,
    /// Alignment requirement for textures
    texture_alignment = 14,
    /// The device can copy memory and run kernels at the same time. Deprecated: use the
    /// async engine count instead.
    gpu_overlap = 15,
    /// Number of multiprocessors on device
    multiprocessor_count = 16,
    /// Specifies whether there is a run time limit on kernels
    kernel_exec_timeout = 17,
    /// Device is integrated with host memory
    integrated = 18,
    /// Device can map host memory into CUDA address space
    can_map_host_memory = 19,
    /// Compute mode (See CUcomputemode for details)
    compute_mode = 20,
    /// Maximum 1D texture width
    maximum_texture1d_width = 21,
    /// Maximum 2D texture width
    maximum_texture2d_width = 22,
    /// Maximum 2D texture height
    maximum_texture2d_height = 23,
    /// Maximum 3D texture width
    maximum_texture3d_width = 24,
    /// Maximum 3D texture height
    maximum_texture3d_height = 25,
    /// Maximum 3D texture depth
    maximum_texture3d_depth = 26,
    /// Maximum 2D layered texture width
    maximum_texture2d_layered_width = 27,
    /// Maximum 2D layered texture height
    maximum_texture2d_layered_height = 28,
    /// Maximum layers in a 2D layered texture
    maximum_texture2d_layered_layers = 29,
    /// Alignment requirement for surfaces
    surface_alignment = 30,
    /// Device can possibly execute multiple kernels concurrently
    concurrent_kernels = 31,
    /// Device has ECC support enabled
    ecc_enabled = 32,
    /// PCI bus ID of the device
    pci_bus_id = 33,
    /// PCI device ID of the device
    pci_device_id = 34,
    /// Device is using TCC driver model
    tcc_driver = 35,
    /// Peak memory clock frequency in kilohertz
    memory_clock_rate = 36,
    /// Global memory bus width in bits
    global_memory_bus_width = 37,
    /// Size of L2 cache in bytes
    l2_cache_size = 38,
    /// Maximum resident threads per multiprocessor
    max_threads_per_multiprocessor = 39,
    /// Number of asynchronous engines
    async_engine_count = 40,
    /// Device shares a unified address space with the host
    unified_addressing = 41,
    /// Maximum 1D layered texture width
    maximum_texture1d_layered_width = 42,
    /// Maximum layers in a 1D layered texture
    maximum_texture1d_layered_layers = 43,
    /// Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    maximum_texture2d_gather_width = 45,
    /// Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    maximum_texture2d_gather_height = 46,
    /// Alternate maximum 3D texture width
    maximum_texture3d_width_alternate = 47,
    /// Alternate maximum 3D texture height
    maximum_texture3d_height_alternate = 48,
    /// Alternate maximum 3D texture depth
    maximum_texture3d_depth_alternate = 49,
    /// PCI domain ID of the device
    pci_domain_id = 50,
    /// Pitch alignment requirement for textures
    texture_pitch_alignment = 51,
    /// Maximum cubemap texture width/height
    maximum_texturecubemap_width = 52,
    /// Maximum cubemap layered texture width/height
    maximum_texturecubemap_layered_width = 53,
    /// Maximum layers in a cubemap layered texture
    maximum_texturecubemap_layered_layers = 54,
    /// Maximum 1D surface width
    maximum_surface1d_width = 55,
    /// Maximum 2D surface width
    maximum_surface2d_width = 56,
    /// Maximum 2D surface height
    maximum_surface2d_height = 57,
    /// Maximum 3D surface width
    maximum_surface3d_width = 58,
    /// Maximum 3D surface height
    maximum_surface3d_height = 59,
    /// Maximum 3D surface depth
    maximum_surface3d_depth = 60,
    /// Maximum 1D layered surface width
    maximum_surface1d_layered_width = 61,
    /// Maximum layers in a 1D layered surface
    maximum_surface1d_layered_layers = 62,
    /// Maximum 2D layered surface width
    maximum_surface2d_layered_width = 63,
    /// Maximum 2D layered surface height
    maximum_surface2d_layered_height = 64,
    /// Maximum layers in a 2D layered surface
    maximum_surface2d_layered_layers = 65,
    /// Maximum cubemap surface width
    maximum_surfacecubemap_width = 66,
    /// Maximum cubemap layered surface width
    maximum_surfacecubemap_layered_width = 67,
    /// Maximum layers in a cubemap layered surface
    maximum_surfacecubemap_layered_layers = 68,
    /// Maximum 2D linear texture width
    maximum_texture2d_linear_width = 70,
    /// Maximum 2D linear texture height
    maximum_texture2d_linear_height = 71,
    /// Maximum 2D linear texture pitch in bytes
    maximum_texture2d_linear_pitch = 72,
    /// Maximum mipmapped 2D texture width
    maximum_texture2d_mipmapped_width = 73,
    /// Maximum mipmapped 2D texture height
    maximum_texture2d_mipmapped_height = 74,
    /// Major compute capability version number
    compute_capability_major = 75,
    /// Minor compute capability version number
    compute_capability_minor = 76,
    /// Maximum mipmapped 1D texture width
    maximum_texture1d_mipmapped_width = 77,
    /// Device supports stream priorities
    stream_priorities_supported = 78,
    /// Device supports caching globals in L1
    global_l1_cache_supported = 79,
    /// Device supports caching locals in L1
    local_l1_cache_supported = 80,
    /// Maximum shared memory available per multiprocessor in bytes
    max_shared_memory_per_multiprocessor = 81,
    /// Maximum number of 32-bit registers available per multiprocessor
    max_registers_per_multiprocessor = 82,
    /// Device can allocate managed memory on this system
    managed_memory = 83,
    /// Device is on a multi-GPU board
    multi_gpu_board = 84,
    /// Unique id for a group of devices on the same multi-GPU board
    multi_gpu_board_group_id = 85,
    /// Link between the device and the host supports all native atomic operations
    host_native_atomic_supported = 86,
    /// Ratio of single precision performance (in floating-point operations per second) to double
    /// precision performance
    single_to_double_precision_perf_ratio = 87,
    /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
    pageable_memory_access = 88,
    /// Device can coherently access managed memory concurrently with the CPU
    concurrent_managed_access = 89,
    /// Device supports compute preemption.
    compute_preemption_supported = 90,
    /// Device can access host registered memory at the same virtual address as the CPU
    can_use_host_pointer_for_registered_mem = 91,
    /// Device supports launching cooperative kernels via cuLaunchCooperativeKernel
    cooperative_launch = 95,
    /// Maximum optin shared memory per block. That is shared memory that is available for dynamic
    /// allocation or static allocation (including architecture specific static shared memory) on
    /// this device but is not guaranteed to be portable.
    max_shared_memory_per_block_optin = 97,
    /// The device supports flushing remote writes, the CU_STREAM_WAIT_VALUE_FLUSH flag and
    /// the CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES mem op.
    can_flush_remote_writes = 98,
    /// Device supports host memory registration via cudaHostRegister.
    host_register_supported = 99,
    /// Device accesses pageable memory via the host's page tables.
    pageable_memory_access_uses_host_page_tables = 100,
    /// The host can directly access managed memory on the device without migration.
    direct_managed_mem_access_from_host = 101,
    /// Device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate,
    /// cuMemMap and related APIs
    virtual_memory_management_supported = 102,
    /// Device supports exporting memory to a posix file descriptor with
    /// cuMemExportToShareableHandle, if requested via cuMemCreate
    handle_type_posix_file_descriptor_supported = 103,
    /// Device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle, if
    /// requested via cuMemCreate
    handle_type_win32_handle_supported = 104,
    /// Device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle, if
    /// requested via cuMemCreate
    handle_type_win32_kmt_handle_supported = 105,
    /// Maximum number of blocks per multiprocessor
    max_blocks_per_multiprocessor = 106,
    /// Device supports compression of memory
    generic_compression_supported = 107,
    /// Maximum L2 persisting lines capacity setting in bytes.
    max_persisting_l2_cache_size = 108,
    /// Maximum value of CUaccessPolicyWindownum_bytes.
    max_access_policy_window_size = 109,
    /// Device supports specifying the GPUDirect RDMA flag with cuMemCreate
    gpu_direct_rdma_with_cuda_vmm_supported = 110,
    /// Shared memory reserved by CUDA driver per block in bytes
    reserved_shared_memory_per_block = 111,
    /// Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
    sparse_cuda_array_supported = 112,
    /// The device supports registering host memory as read-only, with the cuMemHostRegister
    /// flag CU_MEMHOSTERGISTER_READ_ONLY.
    read_only_host_register_supported = 113,
    /// External timeline semaphore interop is supported on the device
    timeline_semaphore_interop_supported = 114,
    /// Device supports using the cuMemAllocAsync and cuMemPool family of APIs
    memory_pools_supported = 115,
    /// Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see
    /// https://docs.nvidia.com/cuda/gpudirect-rdma for more information)
    gpu_direct_rdma_supported = 116,
    /// The returned attribute shall be interpreted as a bitmask, where the individual bits are
    /// described by the CUflushGPUDirectRDMAWritesOptions enum
    gpu_direct_rdma_flush_writes_options = 117,
    /// GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope
    /// indicated by the returned attribute. See CUGPUDirectRDMAWritesOrdering for the numerical
    /// values returned here.
    gpu_direct_rdma_writes_ordering = 118,
    /// Handle types supported with mempool based IPC
    mempool_supported_handle_types = 119,
    /// Indicates device supports cluster launch
    cluster_launch = 120,
    /// Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
    deferred_mapping_cuda_array_supported = 121,
    /// 64-bit operations are supported in cuStreamBatchMemOp and related MemOp APIs.
    can_use_64_bit_stream_mem_ops = 122,
    /// The mem op APIs support the CU_STREAM_WAIT_VALUE_NOR flag.
    can_use_stream_wait_value_nor = 123,
    /// Device supports buffer sharing with dma_buf mechanism.
    dma_buf_supported = 124,
    /// Device supports IPC Events.
    ipc_event_supported = 125,
    /// Number of memory domains the device supports.
    mem_sync_domain_count = 126,
    /// Device supports accessing memory using Tensor Map.
    tensor_map_access_supported = 127,
    /// Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or
    /// requested with cuMemCreate()
    handle_type_fabric_supported = 128,
    /// Device supports unified function pointers.
    unified_function_pointers = 129,
    /// NUMA configuration of a device: value is of type CUdeviceNumaConfig enum
    numa_config = 130,
    /// NUMA node ID of the GPU memory
    numa_id = 131,
    /// Device supports switch multicast and reduction operations.
    multicast_supported = 132,
    /// Indicates if contexts created on this device will be shared via MPS
    mps_enabled = 133,
    /// NUMA ID of the host node closest to the device. Returns -1 when system does not support
    /// NUMA.
    host_numa_id = 134,
    /// Device supports CIG with D3D12.
    d3d12_cig_supported = 135,
    /// The returned valued shall be interpreted as a bitmask, where the individual bits are
    /// described by the CUmemDecompressAlgorithm enum.
    mem_decompress_algorithm_mask = 136,
    /// The returned valued is the maximum length in bytes of a single decompress operation that is
    /// allowed.
    mem_decompress_maximum_length = 137,
    /// Device supports CIG with Vulkan.
    vulkan_cig_supported = 138,
    /// The combined 16-bit PCI device ID and 16-bit PCI vendor ID.
    gpu_pci_device_id = 139,
    /// The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.
    gpu_pci_subsystem_id = 140,
    /// Device supports HOST_NUMA location with the virtual memory management APIs like cuMemCreate,
    /// cuMemMap and related APIs
    host_numa_virtual_memory_management_supported = 141,
    /// Device supports HOST_NUMA location with the cuMemAllocAsync and cuMemPool family of APIs
    host_numa_memory_pools_supported = 142,
    /// Device supports HOST_NUMA location IPC between nodes in a multi-node system.
    host_numa_multinode_ipc_supported = 143,
    /// Device suports HOST location with the cuMemAllocAsync and cuMemPool family of APIs
    host_memory_pools_supported = 144,
    /// Device supports HOST location with the virtual memory management APIs like cuMemCreate,
    /// cuMemMap and related APIs
    host_virtual_memory_management_supported = 145,
    /// Device supports page-locked host memory buffer sharing with dma_buf mechanism.
    host_alloc_dma_buf_supported = 146,
    /// Link between the device and the host supports only some native atomic operations
    only_partial_host_native_atomic_supported = 147,
    /// Device supports atomic reduction operations in stream batch memory operations
    atomic_reduction_supported = 148,
    /// Device supports CIG streams with D3D12
    d3d12_cig_streams_supported = 151,
    /// Device supports mmap() of dmabuf file descriptors for CUDA device memory allocations
    dma_buf_mmap_supported = 152,
    /// Device supports unicast logical endpoints
    logical_endpoint_unicast_supported = 153,
    /// Device supports multicast logical endpoints
    logical_endpoint_multicast_supported = 154,
    /// Device supports counted operations via logical endpoints
    logical_endpoint_counted_ops_supported = 155,
    /// Device supports unicast logical endpoint access on the owner device
    logical_endpoint_unicast_access_on_owner_device_supported = 156,
};

/// Pointers to the functions of the driver API that this namespace uses. `Driver.open` resolves
/// them from the driver library; every one of them returns a `CUresult`.
const Functions = struct {
    init: *const fn (flags: c_uint) callconv(.c) Result,
    driver_get_version: *const fn (version: *c_int) callconv(.c) Result,
    device_get_count: *const fn (count: *c_int) callconv(.c) Result,
    device_get: *const fn (device: *c_int, ordinal: c_int) callconv(.c) Result,
    device_get_name: *const fn (name: [*]u8, len: c_int, device: c_int) callconv(.c) Result,
    device_total_memory: *const fn (bytes: *usize, device: c_int) callconv(.c) Result,
    device_get_attribute: *const fn (
        value: *c_int,
        attribute: c_int,
        device: c_int,
    ) callconv(.c) Result,
    device_primary_ctx_retain: *const fn (context: *?*anyopaque, device: c_int) callconv(.c) Result,
    device_primary_ctx_release: *const fn (device: c_int) callconv(.c) Result,
    ctx_set_current: *const fn (context: ?*anyopaque) callconv(.c) Result,
    ctx_synchronize: *const fn () callconv(.c) Result,
    ctx_set_limit: *const fn (limit: c_int, value: usize) callconv(.c) Result,
    module_load_data_ex: *const fn (
        module: *?*anyopaque,
        image: [*]const u8,
        option_count: c_uint,
        options: ?[*]const c_int,
        option_values: ?[*]?*anyopaque,
    ) callconv(.c) Result,
    module_unload: *const fn (module: ?*anyopaque) callconv(.c) Result,
    module_get_function: *const fn (
        function: *?*anyopaque,
        module: ?*anyopaque,
        name: [*:0]const u8,
    ) callconv(.c) Result,
    mem_alloc: *const fn (address: *u64, bytes: usize) callconv(.c) Result,
    mem_free: *const fn (address: u64) callconv(.c) Result,
    memcpy_htod: *const fn (device: u64, host: *const anyopaque, bytes: usize) callconv(.c) Result,
    memcpy_dtoh: *const fn (host: *anyopaque, device: u64, bytes: usize) callconv(.c) Result,
    memset_d8: *const fn (device: u64, value: u8, count: usize) callconv(.c) Result,
    stream_create: *const fn (stream: *?*anyopaque, flags: c_uint) callconv(.c) Result,
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
};

/// The name of every function in `Functions`, in the same order, as the driver library exports
/// it. The driver exports most of its functions under both a plain and a versioned name, and
/// current versions of `cuda.h` define the plain name to the versioned one; this uses the
/// versioned name, such as `cuMemAlloc_v2` for `cuMemAlloc`.
const function_symbols = .{
    .{ .field = "init", .symbol = "cuInit" },
    .{ .field = "driver_get_version", .symbol = "cuDriverGetVersion" },
    .{ .field = "device_get_count", .symbol = "cuDeviceGetCount" },
    .{ .field = "device_get", .symbol = "cuDeviceGet" },
    .{ .field = "device_get_name", .symbol = "cuDeviceGetName" },
    .{ .field = "device_total_memory", .symbol = "cuDeviceTotalMem_v2" },
    .{ .field = "device_get_attribute", .symbol = "cuDeviceGetAttribute" },
    .{ .field = "device_primary_ctx_retain", .symbol = "cuDevicePrimaryCtxRetain" },
    .{ .field = "device_primary_ctx_release", .symbol = "cuDevicePrimaryCtxRelease_v2" },
    .{ .field = "ctx_set_current", .symbol = "cuCtxSetCurrent" },
    .{ .field = "ctx_synchronize", .symbol = "cuCtxSynchronize" },
    .{ .field = "ctx_set_limit", .symbol = "cuCtxSetLimit" },
    .{ .field = "module_load_data_ex", .symbol = "cuModuleLoadDataEx" },
    .{ .field = "module_unload", .symbol = "cuModuleUnload" },
    .{ .field = "module_get_function", .symbol = "cuModuleGetFunction" },
    .{ .field = "mem_alloc", .symbol = "cuMemAlloc_v2" },
    .{ .field = "mem_free", .symbol = "cuMemFree_v2" },
    .{ .field = "memcpy_htod", .symbol = "cuMemcpyHtoD_v2" },
    .{ .field = "memcpy_dtoh", .symbol = "cuMemcpyDtoH_v2" },
    .{ .field = "memset_d8", .symbol = "cuMemsetD8_v2" },
    .{ .field = "stream_create", .symbol = "cuStreamCreate" },
    .{ .field = "stream_destroy", .symbol = "cuStreamDestroy_v2" },
    .{ .field = "stream_synchronize", .symbol = "cuStreamSynchronize" },
    .{ .field = "launch_kernel", .symbol = "cuLaunchKernel" },
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

/// A loaded CUDA driver.
///
/// The driver library is loaded and initialized by `open`, and stays loaded until `close`.
/// Everything else in this namespace is created from a driver and keeps a pointer to it: the
/// driver must stay open, and must not move, while a `Device`, `Context`, `Module`, `Function`,
/// `Buffer`, or `Stream` from it is in use.
pub const Driver = struct {
    /// The handle of the driver library, `libcuda.so.1`.
    library: std.DynLib,
    /// Pointers to the functions of the driver API that this namespace uses.
    functions: Functions,

    /// Loads the driver library, resolves the functions of the driver API in it, and initializes
    /// the driver.
    ///
    /// `error.DriverNotFound` means that there is no `libcuda.so.1` on this machine, and
    /// `error.IncompatibleDriver` means that the library does not have the functions that this
    /// namespace uses.
    ///
    /// Loading the library needs `dlopen`, so this only runs on Linux with libc linked; every
    /// other target reports it as a compile error.
    pub fn open() OpenError!Driver {
        if (comptime driver_supported) {
            var library = std.DynLib.openZ(library_name) catch return error.DriverNotFound;
            errdefer library.close();

            var functions: Functions = undefined;
            inline for (function_symbols) |entry| {
                const pointer = library.lookup(@FieldType(Functions, entry.field), entry.symbol);
                @field(functions, entry.field) = pointer orelse return error.IncompatibleDriver;
            }

            const driver: Driver = .{ .library = library, .functions = functions };
            try check(driver.functions.init(0));
            return driver;
        } else if (builtin.os.tag != .linux) {
            @compileError("std.gpu.cuda is only available on Linux, where the CUDA driver library is loaded with std.DynLib");
        } else {
            @compileError("std.gpu.cuda needs libc to load the CUDA driver library with dlopen; link the program with libc");
        }
    }

    /// Unloads the driver library. The driver and everything created from it must not be used
    /// afterwards.
    pub fn close(driver: *Driver) void {
        driver.library.close();
        driver.* = undefined;
    }

    /// The version of the driver, such as `.{ .major = 12, .minor = 4 }` for CUDA 12.4.
    pub fn version(driver: *const Driver) Error!Version {
        var encoded: c_int = undefined;
        try check(driver.functions.driver_get_version(&encoded));
        return .{
            .major = @intCast(@divTrunc(encoded, 1000)),
            .minor = @intCast(@divTrunc(@mod(encoded, 1000), 10)),
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
    /// The handle of the device in the driver API, a `CUdevice`.
    handle: c_int,

    /// Copies the name of the device, such as "NVIDIA GeForce RTX 5090", into `buffer` and
    /// returns it as a slice of `buffer`, without the terminating null byte. A name that does
    /// not fit is truncated to `buffer.len - 1` bytes.
    ///
    /// The buffer must have room for at least one byte of the name and the null byte after it:
    /// the driver does not accept a maximum length of zero, and `error.InvalidValue` reports a
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

    /// The compute capability of the device, which PTX assembly must target or be below.
    pub fn computeCapability(dev: Device) Error!ComputeCapability {
        return .{
            .major = @intCast(try dev.attribute(.compute_capability_major)),
            .minor = @intCast(try dev.attribute(.compute_capability_minor)),
        };
    }

    /// Retains the primary context of the device and makes it current on the calling thread.
    ///
    /// The primary context is the one that the driver keeps for the device; retaining it gives
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
    /// `CU_LIMIT_STACK_SIZE`: the size of the stack of a thread, in bytes, which device
    /// functions that call other functions use for their frames.
    stack_size = 0,
    /// `CU_LIMIT_PRINTF_FIFO_SIZE`: the size of the buffer that `std.gpu.print` writes to, in
    /// bytes, for every thread of a block.
    printf_fifo_size = 1,
    /// `CU_LIMIT_MALLOC_HEAP_SIZE`: the size of the heap that `std.gpu.allocators.device_heap`
    /// allocates from, in bytes, for the whole device.
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
    /// The handle of the context in the driver API, a `CUcontext`.
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

    /// Makes the context current on the calling thread, so that the driver uses it for the
    /// memory, modules, and launches of that thread.
    pub fn makeCurrent(ctx: Context) Error!void {
        try check(ctx.driver.functions.ctx_set_current(ctx.handle));
    }

    /// Waits until every copy and every kernel launch in the context has finished, on every
    /// stream. The context must be current on the calling thread.
    pub fn synchronize(ctx: Context) Error!void {
        try check(ctx.driver.functions.ctx_synchronize());
    }

    /// Sets a limit on a resource that the context allocates.
    ///
    /// A limit must be set before the resources it governs are used: the stack size and the
    /// printf buffer take effect for the next kernel launch, and the heap size for the next
    /// device-side allocation.
    pub fn setLimit(ctx: Context, limit: Limit, value: usize) Error!void {
        try check(ctx.driver.functions.ctx_set_limit(@backingInt(limit), value));
    }

    /// Loads a module from an image of PTX assembly and compiles it for the device of the
    /// context.
    ///
    /// `image` is the text of a PTX module that ends with a null byte, such as the output of
    /// `zig build-obj -target nvptx64-cuda`. The driver compiles PTX for older compute
    /// capabilities for the device it is loaded on, so one image can serve several devices.
    pub fn loadModule(ctx: Context, image: [:0]const u8, options: ModuleOptions) Error!Module {
        var handle: ?*anyopaque = null;
        if (options.error_log) |log| {
            const log_size = std.math.cast(c_uint, log.len) orelse return error.InvalidValue;
            var option_names = [_]c_int{ jit_error_log_buffer, jit_error_log_buffer_size_bytes };
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
            try check(ctx.driver.functions.module_load_data_ex(&handle, image.ptr, 0, null, null));
        }
        return .{ .driver = ctx.driver, .handle = handle.? };
    }

    /// Allocates a buffer of `len` elements of `T` in device memory.
    ///
    /// The memory stays allocated until `Buffer.free`, and is only in this context. `len` must
    /// be at least 1, because the driver does not allocate empty buffers.
    pub fn alloc(ctx: Context, comptime T: type, len: usize) Error!Buffer(T) {
        if (comptime @sizeOf(T) == 0) @compileError("cannot allocate a Buffer of the zero-sized type '" ++ @typeName(T) ++ "'");
        const bytes = std.math.mul(usize, @sizeOf(T), len) catch return error.OutOfDeviceMemory;
        var address: u64 = undefined;
        try check(ctx.driver.functions.mem_alloc(&address, bytes));
        return .{ .driver = ctx.driver, .ptr = @fromBackingInt(address), .len = len };
    }

    /// Creates a stream: a queue of copies and kernel launches that other streams do not wait
    /// for. Work on the null stream, the one a `LaunchConfig` uses by default, is ordered
    /// against the work of every other stream.
    pub fn createStream(ctx: Context) Error!Stream {
        var handle: ?*anyopaque = null;
        try check(ctx.driver.functions.stream_create(&handle, 0));
        return .{ .driver = ctx.driver, .handle = handle.? };
    }
};

/// The `CU_JIT_ERROR_LOG_BUFFER` option of `cuModuleLoadDataEx`: a pointer to the buffer that
/// receives the errors of the JIT compiler.
const jit_error_log_buffer: c_int = 5;

/// The `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES` option of `cuModuleLoadDataEx`: the size of that
/// buffer. Options whose value is not a pointer pass the value itself, as a pointer.
const jit_error_log_buffer_size_bytes: c_int = 6;

/// Options for `Context.loadModule`.
pub const ModuleOptions = struct {
    /// When the module fails to load, receives the null-terminated messages of the JIT compiler
    /// that explain what is wrong with the image, such as the line of a syntax error in PTX.
    /// Messages that do not fit in the buffer are cut off. Null discards the messages.
    error_log: ?[]u8 = null,
};

/// A module of compiled code, loaded from a PTX image by `Context.loadModule`.
pub const Module = struct {
    /// The driver that owns this module, which must outlive it.
    driver: *const Driver,
    /// The handle of the module in the driver API, a `CUmodule`.
    handle: *anyopaque,

    /// Unloads the module and its code from the device. Its `Function`s must not be used
    /// afterwards.
    ///
    /// Errors are not reported: the handle comes from a successful load, and the driver only
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
    /// The handle of the function in the driver API, a `CUfunction`.
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

    /// Passes the launch to the driver with `parameters` holding the address of every argument,
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
    /// The stream to run the kernel on, or null for the null stream, which the driver orders
    /// against the work of every other stream.
    stream: ?Stream = null,

    /// A one-dimensional launch of at least `n` threads, in blocks of `block_size` threads.
    ///
    /// The grid has as many blocks as it takes to cover `n` threads, rounded up, so the last
    /// block may be partial and the kernel must compare its thread index against `n`.
    /// `block_size` must be at least 1. `n == 0` gives a grid of zero blocks, which the driver
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
        /// the driver has already lost the context, and there is nothing left to free then.
        pub fn free(buffer: @This()) void {
            _ = buffer.driver.functions.mem_free(@backingInt(buffer.ptr));
        }

        /// Copies `src` into the buffer, starting at the first element. `src` may be shorter
        /// than the buffer, but not longer; a longer `src` fails an assertion.
        pub fn copyFromHost(buffer: @This(), src: []const T) Error!void {
            std.debug.assert(src.len <= buffer.len);
            try check(buffer.driver.functions.memcpy_htod(
                @backingInt(buffer.ptr),
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
                @backingInt(buffer.ptr),
                @sizeOf(T) * dst.len,
            ));
        }

        /// Fills the whole buffer with zero bytes, whatever `T` is.
        pub fn zero(buffer: @This()) Error!void {
            try check(buffer.driver.functions.memset_d8(@backingInt(buffer.ptr), 0, @sizeOf(T) * buffer.len));
        }
    };
}

/// A queue of copies and kernel launches, created by `Context.createStream`.
///
/// Streams run at the same time as each other, and the driver decides the order of the work in
/// them; the null stream, which a `LaunchConfig` uses by default, is ordered against every
/// other stream.
pub const Stream = struct {
    /// The driver that owns this stream, which must outlive it.
    driver: *const Driver,
    /// The handle of the stream in the driver API, a `CUstream`.
    handle: *anyopaque,

    /// Destroys the stream. The work queued on it must have finished, and the stream must not be
    /// used afterwards.
    ///
    /// Errors are not reported: a stream that was created cannot fail to be destroyed.
    pub fn destroy(stream: Stream) void {
        _ = stream.driver.functions.stream_destroy(stream.handle);
    }

    /// Waits until everything queued on the stream has finished.
    pub fn synchronize(stream: Stream) Error!void {
        try check(stream.driver.functions.stream_synchronize(stream.handle));
    }
};

// PTX assembly of a kernel that adds 7 to every element of a `u32` array: the hand-written
// equivalent of this kernel.
//
//     export fn add_constant(data: [*]u32, n: u32) callconv(.kernel) void {
//         const i = std.gpu.globalId(.x);
//         if (i >= n) return;
//         data[i] += 7;
//     }
//
// It targets sm_50, which the driver compiles for whatever device the module is loaded on, so
// the test runs without a toolchain that can compile for NVPTX.
const add_constant_ptx: [:0]const u8 =
    \\.version 6.0
    \\.target sm_50
    \\.address_size 64
    \\
    \\.visible .entry add_constant(
    \\    .param .u64 add_constant_param_0,
    \\    .param .u32 add_constant_param_1
    \\)
    \\{
    \\    .reg .pred %p<2>;
    \\    .reg .b32 %r<5>;
    \\    .reg .b64 %rd<3>;
    \\
    \\    mov.u32       %r1, %ctaid.x;
    \\    mov.u32       %r2, %ntid.x;
    \\    mov.u32       %r3, %tid.x;
    \\    mad.lo.u32    %r1, %r1, %r2, %r3;
    \\    ld.param.u32  %r2, [add_constant_param_1];
    \\    setp.ge.u32   %p1, %r1, %r2;
    \\    @%p1 bra      $L__done;
    \\    ld.param.u64  %rd1, [add_constant_param_0];
    \\    mul.wide.u32  %rd2, %r1, 4;
    \\    add.s64       %rd1, %rd1, %rd2;
    \\    ld.global.u32 %r4, [%rd1];
    \\    add.u32       %r4, %r4, 7;
    \\    st.global.u32 [%rd1], %r4;
    \\$L__done:
    \\    ret;
    \\}
;

test "cuda: LaunchConfig.linear rounds the thread count up to whole blocks" {
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

test "cuda: run a PTX kernel over a buffer" {
    if (!driver_supported) return error.SkipZigTest;

    var driver = Driver.open() catch |err| switch (err) {
        error.DriverNotFound, error.NoDevice => return error.SkipZigTest,
        else => |other| return other,
    };
    defer driver.close();

    // The driver reports its version as 1000 * major + 10 * minor.
    const driver_version = try driver.version();
    try testing.expect(driver_version.major >= 1);

    const device_count = driver.deviceCount() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => |other| return other,
    };
    if (device_count == 0) return error.SkipZigTest;
    const device = try driver.device(0);

    var name_buffer: [256]u8 = undefined;
    const name = try device.name(&name_buffer);
    try testing.expect(name.len > 0);

    // The kernel is PTX for sm_50, so it needs a device of at least that capability.
    const capability = try device.computeCapability();
    if (capability.major < 5) return error.SkipZigTest;
    try testing.expect(try device.totalMemory() > 0);
    try testing.expect(try device.attribute(.warp_size) > 0);

    const context = try device.retainPrimaryContext();
    defer context.release();
    try context.setLimit(.stack_size, 1 << 13);

    var error_log: [1024]u8 = undefined;
    const module = try context.loadModule(add_constant_ptx, .{ .error_log = &error_log });
    defer module.unload();
    const kernel = try module.function("add_constant");

    // 1000 threads are not a multiple of the block size, so the last block is partial and the
    // kernel skips the threads past the end.
    const len = 1000;
    const block_size = 256;
    var host: [len]u32 = undefined;
    for (&host, 0..) |*element, index| element.* = @intCast(index + 1);

    const buffer = try context.alloc(u32, len);
    defer buffer.free();
    try buffer.copyFromHost(&host);

    // A launch on the null stream, which `Context.synchronize` waits for.
    try kernel.launch(LaunchConfig.linear(len, block_size), .{ buffer, @as(u32, len) });
    try context.synchronize();
    @memset(&host, 0);
    try buffer.copyToHost(&host);
    for (host, 0..) |element, index| {
        try testing.expectEqual(@as(u32, @intCast(index + 1)) + 7, element);
    }

    // A launch on a stream of its own, after the driver has zeroed the buffer.
    const stream = try context.createStream();
    defer stream.destroy();
    try buffer.zero();
    var config = LaunchConfig.linear(len, block_size);
    config.stream = stream;
    try kernel.launch(config, .{ buffer, @as(u32, len) });
    try stream.synchronize();
    @memset(&host, 0);
    try buffer.copyToHost(&host);
    for (host) |element| try testing.expectEqual(@as(u32, 7), element);
}

test "cuda: malformed PTX reports an error with a JIT log" {
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

    const malformed_ptx: [:0]const u8 =
        \\.version 6.0
        \\.target sm_50
        \\this is not PTX;
    ;

    var error_log: [1024]u8 = @splat(0);
    if (context.loadModule(malformed_ptx, .{ .error_log = &error_log })) |module| {
        module.unload();
        return error.TestUnexpectedResult;
    } else |err| switch (err) {
        error.InvalidPtx, error.InvalidImage => {},
        else => |other| return other,
    }

    // The JIT compiler explains what is wrong with the image.
    const messages = std.mem.sliceTo(error_log[0..], 0);
    try testing.expect(messages.len > 0);
}
