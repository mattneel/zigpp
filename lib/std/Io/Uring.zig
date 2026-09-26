const addressFromPosix = Io.Threaded.addressFromPosix;
const addressToPosix = Io.Threaded.addressToPosix;
const addressUnixToPosix = Io.Threaded.addressUnixToPosix;
const Alignment = std.mem.Alignment;
const Allocator = std.mem.Allocator;
const Argv0 = Io.Threaded.Argv0;
const assert = std.debug.assert;
const builtin = @import("builtin");
const ChdirError = Io.Threaded.ChdirError;
const clockToPosix = Io.Threaded.clockToPosix;
const Csprng = Io.Threaded.Csprng;
const default_PATH = Io.Threaded.default_PATH;
const Dir = Io.Dir;
const Environ = Io.Threaded.Environ;
const errnoBug = Io.Threaded.errnoBug;
const Evented = @This();
const fallbackSeed = Io.Threaded.fallbackSeed;
const fd_t = linux.fd_t;
const File = Io.File;
const Io = std.Io;
const IoUring = linux.IoUring;
const iovec = std.posix.iovec;
const iovec_const = std.posix.iovec_const;
const linux = std.os.linux;
const linux_statx_request = Io.Threaded.linux_statx_request;
const LOCK = std.posix.LOCK;
const log = std.log.scoped(.@"io-uring");
const max_iovecs_len = Io.Threaded.max_iovecs_len;
const nanosecondsFromPosix = Io.Threaded.nanosecondsFromPosix;
const net = Io.net;
const PATH_MAX = linux.PATH_MAX;
const pathToPosix = Io.Threaded.pathToPosix;
const pid_t = linux.pid_t;
const PosixAddress = Io.Threaded.PosixAddress;
const posixAddressFamily = Io.Threaded.posixAddressFamily;
const posixSocketModeProtocol = Io.Threaded.posixSocketModeProtocol;
const process = std.process;
const recoverableOsBugDetected = Io.Threaded.recoverableOsBugDetected;
const setTimestampToPosix = Io.Threaded.setTimestampToPosix;
const splat_buffer_size = Io.Threaded.splat_buffer_size;
const statFromLinux = Io.Threaded.statFromLinux;
const statxKind = Io.Threaded.statxKind;
const std = @import("../std.zig");
const timestampFromPosix = Io.Threaded.timestampFromPosix;
const unexpectedErrno = std.posix.unexpectedErrno;
const UnixAddress = Io.Threaded.UnixAddress;
const winsize = std.posix.winsize;
const scheduler = @import("Threadz/scheduler.zig");
const Scheduler = scheduler.Scheduler(Evented);
const Fiber = Scheduler.Task;

backing_allocator_needs_mutex: bool,
backing_allocator_mutex: Io.Mutex,
/// Does not need to be thread-safe if not used elsewhere.
backing_allocator: Allocator,
log2_ring_entries: u4,
sched: Scheduler,
sync_limit: ?Io.Semaphore,

stderr_writer_initialized: bool = false,
stderr_mutex: Io.Mutex,
stderr_writer: File.Writer = .{
    .io = undefined,
    .interface = Io.File.Writer.initInterface(&.{}),
    .file = .stderr(),
    .mode = .streaming,
},
stderr_mode: Io.Terminal.Mode = .no_color,

environ_mutex: Io.Mutex,
environ_initialized: bool,
environ: Environ,

null_fd: CachedFd,
random_fd: CachedFd,

csprng_mutex: Io.Mutex,
csprng: Csprng,

/// The pool `io.blocking` calls run on. See `Dirty`.
dirty: Dirty = .{},

/// The ring the watchdog wakes parked workers from. See `Watchdog`.
watchdog: Watchdog = .{},

/// The pool `io.blocking` calls run on: an `Io.Threaded` instance this instance owns, started
/// with the first blocking call, and the group the jobs belong to. A job is one task of that
/// pool, which destroys itself when it finishes, so nothing awaits the group.
pub const Dirty = struct {
    lock: Io.Mutex = .init,
    /// `null` until the first blocking call starts the pool.
    threaded: ?*Io.Threaded = null,
    group: Io.Group = .init,

    fn deinit(d: *Dirty, ev: *Evented) void {
        const threaded = d.threaded orelse return;
        threaded.deinit();
        ev.allocator().destroy(threaded);
        d.threaded = null;
    }

    /// Runs `job` on the pool, starting it if it is not running yet. `false` if there is no
    /// thread to hand the job to: the pool is at its limit, or it could not be started.
    fn submit(d: *Dirty, ev: *Evented, job: *const DirtyJob, name: [:0]const u8) bool {
        const ev_io = ev.io();
        d.lock.lock(ev_io) catch return false; // canceled: the caller makes the call itself
        defer d.lock.unlock(ev_io);
        const threaded = d.threaded orelse started: {
            // The pool's threads allocate from the same allocator, so it is the allocator this
            // instance hands out, which is thread-safe.
            const threaded = ev.allocator().create(Io.Threaded) catch return false;
            // The pool's threads allocate through it too, so it is the allocator this instance
            // hands out, which is thread-safe.
            threaded.* = Io.Threaded.init(ev.allocator(), .{});
            d.threaded = threaded;
            break :started threaded;
        };
        const threaded_io = threaded.io();
        threaded_io.vtable.groupConcurrent(
            threaded_io.userdata,
            &d.group,
            @ptrCast(job),
            .of(*DirtyJob),
            name,
            DirtyJob.run,
        ) catch return false;
        return true;
    }
};

/// One `io.blocking` call: what the pool runs, and where the result goes. It lives on the
/// calling task's stack, which stays put while the task is parked, so the pool reads the
/// arguments and writes the result there without a copy.
pub const DirtyJob = struct {
    ev: *Evented,
    start: *const fn (context: *const anyopaque, result: *anyopaque) void,
    context: *const anyopaque,
    result: *anyopaque,
    /// 0 while the job runs, 1 once it has finished. This is the calling task's own word, on its
    /// stack: the pool copies the job, but the task parks on and the job's thread wakes this
    /// word.
    done: *std.atomic.Value(u32),

    /// On the pool's thread: makes the blocking call, then wakes the task.
    fn run(context: *const anyopaque) void {
        const job: *const DirtyJob = @ptrCast(@alignCast(context));
        job.start(job.context, job.result);
        @atomicStore(u32, &job.done.raw, 1, .release);
        // The task waits on this word on its worker's ring, as a futex wait of the kernel.
        // `futexWake` from a thread that is not a worker is a plain futex wake, which is what
        // wakes such a waiter.
        futexWake(@ptrCast(job.ev), &job.done.raw, 1);
    }
};

/// Runs `start` with `context` on a thread that may block, writing the result to `result` before
/// returning. See `Io.blocking`.
///
/// A task parks while a job runs, so the worker runs other tasks. The call is not cancelable
/// once it has started. If the pool is at its limit, or cannot be started, the call is made on
/// the calling thread, which holds its worker for as long as it blocks.
fn dirtyBlocking(
    userdata: ?*anyopaque,
    result: []u8,
    result_alignment: Alignment,
    context: []const u8,
    context_alignment: Alignment,
    name: [:0]const u8,
    start: *const fn (context: *const anyopaque, result: *anyopaque) void,
) void {
    charge(userdata);
    _ = result_alignment;
    _ = context_alignment;
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (Scheduler.Worker.currentOrNull() == null) {
        // A thread that is not one of the workers has no task to park.
        start(context.ptr, result.ptr);
        return;
    }
    var done: std.atomic.Value(u32) = .init(0);
    const job: DirtyJob = .{
        .ev = ev,
        .start = start,
        .context = context.ptr,
        .result = result.ptr,
        .done = &done,
    };
    if (!ev.dirty.submit(ev, &job, name)) {
        start(context.ptr, result.ptr);
        return;
    }
    // Until the job has run. Uncancelable: the call is not cancelable once it has started. The
    // word is on this task's stack, and a wake that arrives before the wait below does is not
    // lost: the wait returns at once when the word is not 0.
    futexWaitUncancelable(userdata, &done.raw, 0);
    // Acquire: the pool thread wrote the result before it set this word, so what it wrote is
    // visible here.
    assert(done.load(.acquire) != 0);
}

/// The ring the watchdog wakes parked workers from. A ring has one submitter, so the watchdog
/// thread owns this one, and starts it the first time it needs it.
pub const Watchdog = struct {
    /// `null` until the first wake.
    ring: ?IoUring = null,

    fn deinit(w: *Watchdog) void {
        if (w.ring) |*ring| ring.deinit();
        w.ring = null;
    }

    /// Wakes `to`, which is parked in `poll`. Called by the watchdog thread only.
    fn wake(w: *Watchdog, to: *Scheduler.Worker) void {
        const ring = w.get() orelse return;
        // Completions of earlier sends: a send that failed leaves one, and the ring must not
        // fill up with them.
        var cqes: [8]linux.io_uring_cqe = undefined;
        while (ring.cq_ready() != 0) _ = ring.copy_cqes(&cqes, 0) catch break;
        const sqe = ring.get_sqe() catch resubmit: {
            _ = ring.submit() catch return;
            break :resubmit ring.get_sqe() catch return;
        };
        sqe.* = .{
            .opcode = .MSG_RING,
            .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
            .ioprio = 0,
            .fd = to.backend.io_uring.fd,
            .off = @backingInt(Completion.Userdata.wakeup),
            .addr = @backingInt(linux.IORING_MSG_RING_COMMAND.DATA),
            .len = 0,
            .rw_flags = 0,
            .user_data = @backingInt(Completion.Userdata.wakeup),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        _ = ring.submit() catch return;
    }

    fn get(w: *Watchdog) ?*IoUring {
        if (w.ring) |*ring| return ring;
        w.ring = IoUring.init(1, linux.IORING_SETUP_SINGLE_ISSUER) catch |err| {
            std.log.scoped(.threadz).warn("unable to wake a stuck worker: {t}", .{err});
            return null;
        };
        return &w.ring.?;
    }
};

/// For the scheduler: wakes `to` from a thread that is not one of the workers, which is the
/// watchdog: its own ring sends the wake, since a ring may only be submitted to by its thread.
pub fn wakeForeign(ev: *Evented, to: *Scheduler.Worker) void {
    ev.watchdog.wake(to);
}

/// A worker's io_uring and the state that goes with it.
const Thread = struct {
    io_uring: IoUring,
    csprng: Csprng,
    /// The task that has acquired this ring, if any. See `Ring`.
    owner: RingOwner,

    fn current() *Thread {
        return &Scheduler.Worker.current().backend;
    }

    fn currentFiber(thread: *Thread) *Fiber {
        const worker: *Scheduler.Worker = @alignCast(@fieldParentPtr("backend", thread));
        return worker.currentTask();
    }

    fn enqueue(thread: *Thread) *linux.io_uring_sqe {
        while (true) return thread.io_uring.get_sqe() catch {
            thread.submit();
            continue;
        };
    }

    fn submit(thread: *Thread) void {
        _ = thread.io_uring.submit() catch |err| switch (err) {
            error.SignalInterrupt => {},
            else => |e| @panic(@errorName(e)),
        };
    }
};

const RingOwner = struct {
    task: ?*Fiber = null,
    /// Set while the owner is parked in `Ring.waitCqes`.
    waiting: bool = false,
    /// Completions of the owner's operations that `poll` took off the ring while the owner was
    /// not waiting, oldest first from `head`.
    queue: std.ArrayList(linux.io_uring_cqe) = .empty,
    head: usize = 0,
};

/// For the scheduler: a worker's state in this backend.
pub const Worker = Thread;

const CancelRegion = struct {
    fiber: *Fiber,
    status: Fiber.CancelStatus,
    fn init() CancelRegion {
        const fiber = Thread.current().currentFiber();
        return .{
            .fiber = fiber,
            .status = .{
                .requested = fiber.cancel_protection.check() == .unblocked,
                .awaiting = .nothing,
            },
        };
    }
    fn initBlocked() CancelRegion {
        return .{
            .fiber = Thread.current().currentFiber(),
            .status = .{ .requested = false, .awaiting = .nothing },
        };
    }
    fn deinit(cancel_region: *CancelRegion) void {
        if (cancel_region.status.requested) {
            @branchHint(.likely);
            _ = cancel_region.fiber.cancel_status.changeAwaiting(
                cancel_region.status.awaiting,
                .nothing,
            );
        }
        cancel_region.* = undefined;
    }
    fn await(cancel_region: *CancelRegion, awaiting: Fiber.CancelStatus.Awaiting) Io.Cancelable!void {
        if (!cancel_region.status.requested) {
            @branchHint(.unlikely);
            return;
        }
        const status: Fiber.CancelStatus = .{ .requested = true, .awaiting = awaiting };
        if (cancel_region.fiber.cancel_status.changeAwaiting(
            cancel_region.status.awaiting,
            status.awaiting,
        )) {
            @branchHint(.unlikely);
            // The request arrived first, so nothing is pending that it could cancel. Leave the
            // fiber awaiting nothing, since `deinit` does not run for an unrequested region.
            _ = cancel_region.fiber.cancel_status.changeAwaiting(status.awaiting, .nothing);
            cancel_region.fiber.cancel_protection.acknowledge();
            cancel_region.status = .unrequested;
            return error.Canceled;
        }
        cancel_region.status = status;
    }
    fn awaitIoUring(cancel_region: *CancelRegion) Io.Cancelable!*Thread {
        const thread: *Thread = .current();
        try cancel_region.await(.fromToken(@intCast(thread.io_uring.fd)));
        return thread;
    }
    fn completion(cancel_region: *const CancelRegion) Completion {
        return cancel_region.fiber.resultPointer(Completion).*;
    }
    fn errno(cancel_region: *const CancelRegion) linux.E {
        return cancel_region.completion().errno();
    }

    /// A system call made on the worker itself, which runs no other task until the call returns.
    /// Operations io_uring has no opcode for are made this way. So are the ones it can only
    /// complete on one of its kernel worker threads and that are quick in themselves: positional
    /// writes, `ftruncate`, `statx`, opens that create or truncate, `mkdirat`, `unlinkat`,
    /// `renameat`, `symlinkat` and `linkat`. Handing one of those to a kernel thread and back
    /// costs more than the call. `fsync` stays on the ring: it is slow in itself, and the worker
    /// runs other tasks meanwhile.
    const Sync = struct {
        cancel_region: CancelRegion,
        fn init(ev: *Evented) Io.Cancelable!Sync {
            if (ev.sync_limit) |*sync_limit| try sync_limit.wait(ev.io());
            return .{ .cancel_region = .init() };
        }
        fn initBlocked(ev: *Evented) Sync {
            if (ev.sync_limit) |*sync_limit| sync_limit.waitUncancelable(ev.io());
            return .{ .cancel_region = .initBlocked() };
        }
        fn deinit(sync: *Sync, ev: *Evented) void {
            sync.cancel_region.deinit();
            if (ev.sync_limit) |*sync_limit| sync_limit.post(ev.io());
        }

        const Maybe = union(enum) {
            cancel_region: CancelRegion,
            sync: Sync,

            fn deinit(maybe: *Maybe, ev: *Evented) void {
                switch (maybe.*) {
                    .cancel_region => |*cancel_region| cancel_region.deinit(),
                    .sync => |*sync| sync.deinit(ev),
                }
            }

            fn enterSync(maybe: *Maybe, ev: *Evented) Io.Cancelable!*Sync {
                switch (maybe.*) {
                    .cancel_region => |cancel_region| {
                        if (ev.sync_limit) |*sync_limit| try sync_limit.wait(ev.io());
                        maybe.* = .{ .sync = .{ .cancel_region = cancel_region } };
                    },
                    .sync => {},
                }
                return &maybe.sync;
            }

            fn leaveSync(maybe: *Maybe, ev: *Evented) void {
                switch (maybe.*) {
                    .cancel_region => {},
                    .sync => |sync| {
                        if (ev.sync_limit) |*sync_limit| sync_limit.post(ev.io());
                        maybe.* = .{ .cancel_region = sync.cancel_region };
                    },
                }
            }

            fn cancelRegion(maybe: *Maybe) *CancelRegion {
                return switch (maybe.*) {
                    .cancel_region => |*cancel_region| cancel_region,
                    .sync => |*sync| &sync.cancel_region,
                };
            }
        };
    };
};

const CachedFd = struct {
    once: Once,

    const Once = enum(fd_t) {
        uninitialized = -1,
        initializing = -2,
        /// fd
        _,

        fn fromFd(fd: fd_t) Once {
            return @fromBackingInt(@intCast(@as(u31, @intCast(fd))));
        }

        fn toFd(once: Once) fd_t {
            return @as(u31, @intCast(@backingInt(once)));
        }
    };

    const init: CachedFd = .{ .once = .uninitialized };

    fn close(cached_fd: *CachedFd) void {
        switch (cached_fd.once) {
            .uninitialized => {},
            .initializing => unreachable,
            _ => |fd| {
                assert(@backingInt(fd) >= 0);
                _ = linux.close(@backingInt(fd));
                cached_fd.* = .init;
            },
        }
    }

    fn open(
        cached_fd: *CachedFd,
        ev: *Evented,
        cancel_region: *CancelRegion,
        path: [*:0]const u8,
        flags: linux.O,
    ) File.OpenError!fd_t {
        var once = @atomicLoad(Once, &cached_fd.once, .monotonic);
        while (true) {
            switch (once) {
                .uninitialized => {},
                .initializing => try futexWait(
                    ev,
                    @ptrCast(&cached_fd.once),
                    @bitCast(@backingInt(once)),
                    .none,
                ),
                _ => |fd| {
                    @branchHint(.likely);
                    return fd.toFd();
                },
            }
            once = @cmpxchgWeak(
                Once,
                &cached_fd.once,
                .uninitialized,
                .initializing,
                .monotonic,
                .monotonic,
            ) orelse {
                errdefer {
                    @atomicStore(Once, &cached_fd.once, .uninitialized, .monotonic);
                    futexWake(ev, @ptrCast(&cached_fd.once), 1);
                }
                const fd = ev.openat(cancel_region, linux.AT.FDCWD, path, flags, 0) catch |err| switch (err) {
                    error.OperationUnsupported => return error.Unexpected, // TMPFILE unset.
                    else => |e| return e,
                };
                @atomicStore(Once, &cached_fd.once, .fromFd(fd), .monotonic);
                futexWake(ev, @ptrCast(&cached_fd.once), std.math.maxInt(u32));
                return fd;
            };
        }
    }
};

pub fn allocator(ev: *Evented) std.mem.Allocator {
    return if (ev.backing_allocator_needs_mutex) .{
        .ptr = ev,
        .vtable = &.{
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
        },
    } else ev.backing_allocator;
}

fn alloc(userdata: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    ev.backing_allocator_mutex.lockUncancelable(ev_io);
    defer ev.backing_allocator_mutex.unlock(ev_io);
    return ev.backing_allocator.rawAlloc(len, alignment, ret_addr);
}

fn resize(
    userdata: *anyopaque,
    memory: []u8,
    alignment: std.mem.Alignment,
    new_len: usize,
    ret_addr: usize,
) bool {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    ev.backing_allocator_mutex.lockUncancelable(ev_io);
    defer ev.backing_allocator_mutex.unlock(ev_io);
    return ev.backing_allocator.rawResize(memory, alignment, new_len, ret_addr);
}

fn remap(
    userdata: *anyopaque,
    memory: []u8,
    alignment: Alignment,
    new_len: usize,
    ret_addr: usize,
) ?[*]u8 {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    ev.backing_allocator_mutex.lockUncancelable(ev_io);
    defer ev.backing_allocator_mutex.unlock(ev_io);
    return ev.backing_allocator.rawRemap(memory, alignment, new_len, ret_addr);
}

fn free(userdata: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    ev.backing_allocator_mutex.lockUncancelable(ev_io);
    defer ev.backing_allocator_mutex.unlock(ev_io);
    return ev.backing_allocator.rawFree(memory, alignment, ret_addr);
}

/// Charges one operation to the calling task, if the calling thread is one of the workers: every
/// entry point of the vtable below is one Io operation the task made, and a task that makes
/// `Scheduler.budget` of them without parking yields at the next one. See `Scheduler.charge`.
///
/// An entry point that reaches another one is charged for both. That only makes a task that
/// loops on such an operation yield a little sooner.
inline fn charge(userdata: ?*anyopaque) void {
    Scheduler.charged(userdata);
}

pub fn io(ev: *Evented) Io {
    return .{
        .userdata = ev,
        .vtable = &.{
            .crashHandler = Scheduler.crashHandler,

            .async = Scheduler.async,
            .concurrent = Scheduler.concurrent,
            .await = Scheduler.await,
            .cancel = Scheduler.cancel,
            .blocking = dirtyBlocking,

            .groupAsync = Scheduler.groupAsync,
            .groupConcurrent = Scheduler.groupConcurrent,
            .groupAwait = Scheduler.groupAwait,
            .groupCancel = Scheduler.groupCancel,

            .recancel = Scheduler.recancel,
            .swapCancelProtection = Scheduler.swapCancelProtection,
            .checkCancel = Scheduler.checkCancel,

            .futexWait = futexWait,
            .futexWaitUncancelable = futexWaitUncancelable,
            .futexWake = futexWake,

            .operate = operate,
            .batchAwaitAsync = batchAwaitAsync,
            .batchAwaitConcurrent = batchAwaitConcurrent,
            .batchCancel = batchCancel,

            .dirCreateDir = dirCreateDir,
            .dirCreateDirPath = dirCreateDirPath,
            .dirCreateDirPathOpen = dirCreateDirPathOpen,
            .dirOpenDir = dirOpenDir,
            .dirStat = dirStat,
            .dirStatFile = dirStatFile,
            .dirAccess = dirAccess,
            .dirCreateFile = dirCreateFile,
            .dirCreateFileAtomic = dirCreateFileAtomic,
            .dirOpenFile = dirOpenFile,
            .dirClose = dirClose,
            .dirRead = dirRead,
            .dirRealPath = dirRealPath,
            .dirRealPathFile = dirRealPathFile,
            .dirDeleteFile = dirDeleteFile,
            .dirDeleteDir = dirDeleteDir,
            .dirRename = dirRename,
            .dirRenamePreserve = dirRenamePreserve,
            .dirSymLink = dirSymLink,
            .dirReadLink = dirReadLink,
            .dirSetOwner = dirSetOwner,
            .dirSetFileOwner = dirSetFileOwner,
            .dirSetPermissions = dirSetPermissions,
            .dirSetFilePermissions = dirSetFilePermissions,
            .dirSetTimestamps = dirSetTimestamps,
            .dirHardLink = dirHardLink,

            .fileStat = fileStat,
            .fileLength = fileLength,
            .fileClose = fileClose,
            .fileWritePositional = fileWritePositional,
            .fileWriteFileStreaming = fileWriteFileStreaming,
            .fileWriteFilePositional = fileWriteFilePositional,
            .fileReadPositional = fileReadPositional,
            .fileSeekBy = fileSeekBy,
            .fileSeekTo = fileSeekTo,
            .fileSync = fileSync,
            .fileIsTty = fileIsTty,
            .fileEnableAnsiEscapeCodes = fileEnableAnsiEscapeCodes,
            .fileSupportsAnsiEscapeCodes = fileIsTty,
            .fileSetLength = fileSetLength,
            .fileSetOwner = fileSetOwner,
            .fileSetPermissions = fileSetPermissions,
            .fileSetTimestamps = fileSetTimestamps,
            .fileLock = fileLock,
            .fileTryLock = fileTryLock,
            .fileUnlock = fileUnlock,
            .fileDowngradeLock = fileDowngradeLock,
            .fileRealPath = fileRealPath,
            .fileHardLink = fileHardLink,

            .fileMemoryMapCreate = fileMemoryMapCreate,
            .fileMemoryMapDestroy = fileMemoryMapDestroy,
            .fileMemoryMapSetLength = fileMemoryMapSetLength,
            .fileMemoryMapRead = fileMemoryMapRead,
            .fileMemoryMapWrite = fileMemoryMapWrite,

            .processExecutableOpen = processExecutableOpen,
            .processExecutablePath = processExecutablePath,
            .lockStderr = lockStderr,
            .tryLockStderr = tryLockStderr,
            .unlockStderr = unlockStderr,
            .processCurrentPath = processCurrentPath,
            .processSetCurrentDir = processSetCurrentDir,
            .processSetCurrentPath = processSetCurrentPath,
            .processReplace = processReplace,
            .processReplacePath = processReplacePath,
            .processSpawn = processSpawn,
            .processSpawnPath = processSpawnPath,
            .childWait = childWait,
            .childKill = childKill,

            .progressParentFile = progressParentFile,

            .now = now,
            .clockResolution = clockResolution,
            .sleep = sleep,

            .random = random,
            .randomSecure = randomSecure,

            .netListenIp = netListenIp,
            .netAccept = netAccept,
            .netBindIp = netBindIp,
            .netConnectIp = netConnectIp,
            .netListenUnix = netListenUnix,
            .netConnectUnix = netConnectUnix,
            .netSocketCreatePair = netSocketCreatePair,
            .netWriteFile = netWriteFile,
            .netClose = netClose,
            .netShutdown = netShutdown,
            .netInterfaceNameResolve = netInterfaceNameResolve,
            .netInterfaceName = netInterfaceName,
            .netLookup = netLookup,
        },
    };
}

pub const InitOptions = struct {
    backing_allocator_needs_mutex: bool = true,

    /// Maximum thread pool size (excluding the main thread).
    /// Defaults to one less than the number of logical CPU cores.
    thread_limit: ?usize = null,
    /// Maximum number of threads that may perform synchronous syscalls.
    sync_limit: Io.Limit = .unlimited,

    log2_ring_entries: u4 = 3,

    /// The stack reserved for each task, unless its spawn says otherwise. The OS commits the
    /// pages as the task touches them, and a guard page below the stack catches an overflow.
    stack_size: usize = scheduler.default_stack_size,

    /// Affects the following operations:
    /// * `processExecutablePath` on OpenBSD and Haiku.
    argv0: Argv0 = .empty,
    /// Affects the following operations:
    /// * `fileIsTty`
    /// * `processSpawn`, `processSpawnPath`, `processReplace`, `processReplacePath`
    environ: process.Environ = .empty,
};

/// Makes the calling thread the first worker, running as the main task. `ev` must stay put
/// until `deinit`, which must be called from the main task.
pub fn init(ev: *Evented, backing_allocator: Allocator, options: InitOptions) !void {
    ev.* = .{
        .backing_allocator_needs_mutex = options.backing_allocator_needs_mutex,
        .backing_allocator_mutex = .init,
        .backing_allocator = backing_allocator,
        .log2_ring_entries = options.log2_ring_entries,
        .sched = undefined,
        .sync_limit = if (options.sync_limit.toInt()) |sync_limit| .{ .permits = sync_limit } else null,

        .stderr_writer_initialized = false,
        .stderr_mutex = .init,
        .stderr_writer = .{
            .io = ev.io(),
            .interface = Io.File.Writer.initInterface(&.{}),
            .file = .stderr(),
            .mode = .streaming,
        },
        .stderr_mode = .no_color,

        .environ_mutex = .init,
        .environ_initialized = options.environ.block.isEmpty(),
        .environ = .{ .process_environ = options.environ },

        .null_fd = .init,
        .random_fd = .init,

        .csprng_mutex = .init,
        .csprng = .uninitialized,
        .dirty = .{},
        .watchdog = .{},
    };
    try ev.sched.init(backing_allocator, .{
        .workers = if (options.thread_limit) |thread_limit| 1 + thread_limit else null,
        .stack_size = options.stack_size,
    });
}

/// Called from the main task once every other task has been awaited. Returns on the thread that
/// called `init`.
pub fn deinit(ev: *Evented) void {
    ev.sched.deinit(ev.backing_allocator);
    ev.dirty.deinit(ev);
    ev.watchdog.deinit();
    ev.null_fd.close();
    ev.random_fd.close();
    ev.* = undefined;
}

pub const SpawnOptions = scheduler.SpawnOptions;
pub const Affinity = scheduler.Affinity;

/// `Io.concurrent`, with a stack size and an affinity for the task.
pub fn concurrentWith(
    ev: *Evented,
    options: SpawnOptions,
    function: anytype,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) Io.ConcurrentError!Io.Future(@typeInfo(@TypeOf(function)).@"fn".return_type.?) {
    return ev.sched.concurrentWith(options, function, args);
}

/// `Io.Group.concurrent`, with a stack size and an affinity for the task.
pub fn groupConcurrentWith(
    ev: *Evented,
    group: *Io.Group,
    options: SpawnOptions,
    function: anytype,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) Io.ConcurrentError!void {
    return ev.sched.groupConcurrentWith(group, options, function, args);
}

/// Starts no more than `n` workers, counting the thread that called `init`, and no more than
/// `InitOptions.thread_limit` allowed. Workers already running keep running.
pub fn setWorkerLimit(ev: *Evented, n: usize) void {
    ev.sched.setWorkerLimit(n);
}

/// The number of workers this instance may run, counting the thread that called `init`. A task
/// pinned to a worker has an index below this.
pub fn workerLimit(ev: *Evented) u32 {
    return ev.sched.limit.load(.monotonic);
}

/// What a program can read about this instance: see `Scheduler.Stats`, which is this type. The
/// stuck counter and the last stuck episode are the watchdog's; see `Scheduler`.
pub const Stats = Scheduler.Stats;

pub fn stats(ev: *Evented) Stats {
    return ev.sched.stats();
}

/// The `Evented` behind `any_io`, or `null` if `any_io` is another `Io` implementation.
pub fn fromIo(any_io: Io) ?*Evented {
    return if (any_io.vtable.operate == &operate) @ptrCast(@alignCast(any_io.userdata)) else null;
}

/// The bit of an SQE's `user_data` that makes its completion the ring owner's. Threadz's own
/// operations leave it clear.
pub const ring_owner_bit: u64 = 1 << 63;

/// A worker's io_uring, lent to a task pinned to that worker that drives operations of its own,
/// such as a server's event loop. The task queues SQEs on `uring` with `ring_owner_bit` set in
/// their `user_data`, and takes their completions with `copyCqes` and `waitCqes`. The worker
/// handles every other completion as before, so the owner and the worker's other tasks share
/// the ring, and the owner parks rather than blocking its worker.
pub const Ring = struct {
    ev: *Evented,
    worker: *Scheduler.Worker,

    /// The worker's ring, for SQEs, `submit`, and registering buffer rings. Its completions are
    /// taken with `copyCqes` and `waitCqes`, never with its own `copy_cqes`.
    pub fn uring(r: Ring) *IoUring {
        return &r.worker.backend.io_uring;
    }

    /// Submits the queued SQEs, then copies up to `cqes.len` completions of the owner's
    /// operations, oldest first, and returns how many. The worker's other completions found on
    /// the way are handled. Does not park.
    pub fn copyCqes(r: Ring, cqes: []linux.io_uring_cqe) usize {
        const thread = &r.worker.backend;
        const owner = &thread.owner;
        assert(owner.task == r.worker.currentTask());
        const queued = owner.queue.items[owner.head..];
        const n = @min(queued.len, cqes.len);
        @memcpy(cqes[0..n], queued[0..n]);
        owner.head += n;
        if (owner.head == owner.queue.items.len) {
            owner.queue.clearRetainingCapacity();
            owner.head = 0;
        }
        enterRing(thread, .nonblocking);
        return n + drain(r.ev, r.worker, cqes[n..]);
    }

    /// `copyCqes`, parking the task until at least one completion of its own has arrived. Not
    /// cancelable: a loop that waits here stops on an operation of its own, such as a
    /// `MSG_RING` from the task that stops it.
    pub fn waitCqes(r: Ring, cqes: []linux.io_uring_cqe) usize {
        assert(cqes.len != 0);
        while (true) {
            const n = r.copyCqes(cqes);
            if (n != 0) return n;
            r.worker.backend.owner.waiting = true;
            r.ev.sched.park();
        }
    }

    /// Ends the loan. Completions of operations still in flight are dropped, so the owner reaps
    /// or cancels its operations first.
    pub fn release(r: Ring) void {
        const owner = &r.worker.backend.owner;
        assert(owner.task == r.worker.currentTask());
        owner.task = null;
        owner.waiting = false;
        owner.queue.clearRetainingCapacity();
        owner.head = 0;
    }
};

/// Lends the calling task its worker's ring until `Ring.release`. The task must be pinned, since
/// the ring belongs to the worker; see `SpawnOptions`. A ring has one owner at a time.
pub fn acquireRing(ev: *Evented) error{ NotPinned, RingOwned, OutOfMemory }!Ring {
    const worker = Scheduler.Worker.current();
    const task = worker.currentTask();
    if (task.affinity != .pinned) return error.NotPinned;
    const owner = &worker.backend.owner;
    if (owner.task != null) return error.RingOwned;
    try owner.queue.ensureTotalCapacity(std.heap.page_allocator, worker.backend.io_uring.cq.cqes.len);
    owner.task = task;
    return .{ .ev = ev, .worker = worker };
}

pub const Completion = struct {
    result: i32,
    flags: u32,

    const Userdata = enum(usize) {
        unused,
        wakeup,
        futex_wake,
        close,
        cleanup,
        /// If bit 0 is 1, a pointer to the `context` field of `Io.Batch.Storage.Pending`.
        /// If bits 0 and 1 are 0, a `*Fiber`.
        _,
    };

    fn errno(completion: Completion) linux.E {
        return linux.errno(@as(isize, completion.result));
    }
};

/// For the scheduler: sets up worker `worker.index`'s ring. Workers after the first share the
/// first one's kernel worker pool, and start disabled until their own thread enables them.
pub fn workerInit(ev: *Evented, worker: *Scheduler.Worker) !void {
    const entries = @as(u16, 1) << ev.log2_ring_entries;
    worker.backend = .{
        .io_uring = if (worker.index == 0)
            try .init(entries, linux.IORING_SETUP_COOP_TASKRUN |
                linux.IORING_SETUP_TASKRUN_FLAG |
                linux.IORING_SETUP_SINGLE_ISSUER)
        else ring: {
            var params = std.mem.zeroInit(linux.io_uring_params, .{
                .flags = linux.IORING_SETUP_ATTACH_WQ |
                    linux.IORING_SETUP_R_DISABLED |
                    linux.IORING_SETUP_COOP_TASKRUN |
                    linux.IORING_SETUP_TASKRUN_FLAG |
                    linux.IORING_SETUP_SINGLE_ISSUER,
                .wq_fd = @as(u32, @intCast(ev.sched.workers[0].backend.io_uring.fd)),
            });
            break :ring try .init_params(entries, &params);
        },
        .csprng = .uninitialized,
        .owner = .{},
    };
}

/// For the scheduler: on the worker's own thread, which becomes the ring's only submitter.
pub fn workerStart(ev: *Evented, worker: *Scheduler.Worker) void {
    _ = ev;
    if (worker.index == 0) return;
    switch (linux.errno(linux.io_uring_register(worker.backend.io_uring.fd, .REGISTER_ENABLE_RINGS, null, 0))) {
        .SUCCESS => {},
        else => |err| @panic(@tagName(err)),
    }
}

/// For the scheduler.
pub fn workerDeinit(ev: *Evented, worker: *Scheduler.Worker) void {
    _ = ev;
    worker.backend.owner.queue.deinit(std.heap.page_allocator);
    worker.backend.io_uring.deinit();
}

/// For the scheduler: submits this worker's queued operations, then hands each task whose
/// operation finished back to the scheduler. `.block` first waits for at least one completion.
pub fn poll(ev: *Evented, worker: *Scheduler.Worker, mode: scheduler.PollMode) void {
    enterRing(&worker.backend, mode);
    _ = drain(ev, worker, &.{});
}

/// Submits the queued operations and, for `.block`, waits for a completion. A nonblocking call
/// enters the kernel only to submit, or to run the completion work the kernel has flagged:
/// completions it has already posted are in the ring.
fn enterRing(thread: *Thread, mode: scheduler.PollMode) void {
    const ring = &thread.io_uring;
    switch (mode) {
        .block => {},
        .nonblocking => if (ring.sq_ready() == 0 and @atomicLoad(u32, ring.sq.flags, .unordered) &
            (linux.IORING_SQ_TASKRUN | linux.IORING_SQ_CQ_OVERFLOW) == 0) return,
    }
    _ = ring.submit_and_wait(switch (mode) {
        .nonblocking => 0,
        .block => 1,
    }) catch |err| switch (err) {
        error.SignalInterrupt => {},
        else => |e| @panic(@errorName(e)),
    };
}

/// Handles the completions in the worker's ring, oldest first, and returns how many of the ring
/// owner's it copied to `owner_cqes`, which is where they go while there is room. Called from
/// `poll`, with no room, it leaves them to the owner: if the owner waits in `Ring.waitCqes`, the
/// first one stays in the ring and the owner is made runnable; if not, they move to its queue.
fn drain(ev: *Evented, worker: *Scheduler.Worker, owner_cqes: []linux.io_uring_cqe) usize {
    const thread = &worker.backend;
    const cq = &thread.io_uring.cq;
    var copied: usize = 0;
    while (true) {
        var head = cq.head.*;
        const tail = @atomicLoad(u32, cq.tail, .acquire);
        if (head == tail) return copied;
        while (head != tail) {
            const cqe = cq.cqes[head & cq.mask];
            const owned = cqe.user_data & ring_owner_bit != 0;
            if (owned and thread.owner.task != null and cqe.flags & linux.IORING_CQE_F_SKIP == 0) {
                const owner = &thread.owner;
                if (copied < owner_cqes.len) {
                    owner_cqes[copied] = cqe;
                    copied += 1;
                } else if (owner_cqes.len != 0 or owner.waiting) {
                    // Leave it for the owner, which is running or about to.
                    @atomicStore(u32, cq.head, head, .release);
                    if (owner.waiting) {
                        owner.waiting = false;
                        ev.sched.readyFromPoll(worker, owner.task.?);
                    }
                    return copied;
                } else owner.queue.append(std.heap.page_allocator, cqe) catch {
                    @atomicStore(u32, cq.head, head, .release);
                    return copied;
                };
                head +%= 1;
                continue;
            }
            // Consumed before it is handled: handling can submit, and the kernel can then post
            // completions only into room it knows about.
            head +%= 1;
            @atomicStore(u32, cq.head, head, .release);
            // An owner's completion with no owner left is dropped.
            if (!owned) handleCompletion(ev, worker, cqe);
        }
        @atomicStore(u32, cq.head, head, .release);
    }
}

fn handleCompletion(ev: *Evented, worker: *Scheduler.Worker, cqe: linux.io_uring_cqe) void {
    const thread = &worker.backend;
    if (cqe.flags & linux.IORING_CQE_F_SKIP != 0) return;
    switch (@as(
        Completion.Userdata,
        @fromBackingInt(@intCast(cqe.user_data)),
    )) {
        .unused => unreachable, // bad submission queued?
        .wakeup => {},
        .futex_wake => switch (Completion.errno(.{ .result = cqe.res, .flags = cqe.flags })) {
            .SUCCESS => recoverableOsBugDetected(), // success is skipped
            .INVAL => {}, // invalid futex_wait() on ptr done elsewhere
            .INTR, .CANCELED => recoverableOsBugDetected(), // `Completion.Userdata.futex_wake` is not cancelable
            .FAULT => {}, // pointer became invalid while doing the wake
            else => recoverableOsBugDetected(), // deadlock due to operating system bug
        },
        .close => switch (Completion.errno(.{ .result = cqe.res, .flags = cqe.flags })) {
            .BADF => recoverableOsBugDetected(), // Always a race condition.
            .INTR => {}, // This is still a success. See https://github.com/ziglang/zig/issues/2425
            else => {},
        },
        .cleanup => @panic("failed to notify another worker"),
        _ => if (@as(?*Fiber, ready_fiber: switch (@as(u2, @truncate(cqe.user_data))) {
            0b00 => {
                const ready_fiber: *Fiber = @ptrFromInt(cqe.user_data & ~@as(usize, 0b11));
                ready_fiber.resultPointer(Completion).* = .{
                    .result = cqe.res,
                    .flags = cqe.flags,
                };
                break :ready_fiber ready_fiber;
            },
            0b01 => {
                // Another worker asks this ring to cancel an operation it holds.
                thread.enqueue().* = .{
                    .opcode = .ASYNC_CANCEL,
                    .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
                    .ioprio = 0,
                    .fd = 0,
                    .off = 0,
                    .addr = cqe.user_data & ~@as(usize, 0b11),
                    .len = 0,
                    .rw_flags = 0,
                    .user_data = @backingInt(Completion.Userdata.wakeup),
                    .buf_index = 0,
                    .personality = 0,
                    .splice_fd_in = 0,
                    .addr3 = 0,
                    .resv = 0,
                };
                break :ready_fiber null;
            },
            0b10 => {
                const batch_userdata: *Io.Operation.Storage.Pending.Userdata =
                    @ptrFromInt(cqe.user_data & ~@as(usize, 0b11));
                const batch: *Io.Batch = @ptrFromInt(batch_userdata[0]);
                var next: usize = 0b00;
                batch_userdata[0..3].* = .{ next, @as(u32, @bitCast(cqe.res)), cqe.flags };
                while (true) {
                    next = @cmpxchgWeak(
                        usize,
                        @as(*usize, @ptrCast(&batch.userdata)),
                        next,
                        cqe.user_data,
                        .release,
                        .acquire,
                    ) orelse break;
                    batch_userdata[0] = next;
                }
                break :ready_fiber switch (@as(u2, @truncate(next))) {
                    0b00, 0b01 => @ptrFromInt(next & ~@as(usize, 0b11)),
                    0b10, 0b11 => null,
                };
            },
            0b11 => switch (Completion.errno(.{ .result = cqe.res, .flags = cqe.flags })) {
                .SUCCESS => unreachable, // no event count specified
                .TIME => {
                    const context: *usize = @ptrFromInt(cqe.user_data & ~@as(usize, 0b11));
                    const fiber = @atomicRmw(usize, context, .Add, 0b01, .acquire);
                    break :ready_fiber switch (@as(u2, @truncate(fiber))) {
                        else => unreachable, // timeout completed multiple times
                        0b00 => @ptrFromInt(fiber & ~@as(usize, 0b11)),
                        0b10 => null,
                    };
                },
                .CANCELED => null, // user data may have been invalidated
                else => |err| unexpectedErrno(err) catch null,
            },
        })) |ready_fiber| ev.sched.readyFromPoll(worker, ready_fiber),
    }
}

/// For the scheduler: makes `to`'s current or next blocking `poll` return.
pub fn wake(ev: *Evented, from: *Scheduler.Worker, to: *Scheduler.Worker) void {
    _ = ev;
    const thread = &from.backend;
    thread.enqueue().* = .{
        .opcode = .MSG_RING,
        .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
        .ioprio = 0,
        .fd = to.backend.io_uring.fd,
        .off = @backingInt(Completion.Userdata.wakeup),
        .addr = @backingInt(linux.IORING_MSG_RING_COMMAND.DATA),
        .len = 0,
        .rw_flags = 0,
        .user_data = @backingInt(Completion.Userdata.wakeup),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
    thread.submit();
}

/// For the scheduler: cancels the operation `fiber` waits for in the ring `token`, directly if
/// it is this worker's, or by asking the worker that owns it.
pub fn cancelOperation(ev: *Evented, from: *Scheduler.Worker, fiber: *Fiber, token: u31) void {
    _ = ev;
    const thread = &from.backend;
    thread.enqueue().* = if (thread.io_uring.fd == token) .{
        .opcode = .ASYNC_CANCEL,
        .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
        .ioprio = 0,
        .fd = 0,
        .off = 0,
        .addr = @intFromPtr(fiber),
        .len = 0,
        .rw_flags = 0,
        .user_data = @backingInt(Completion.Userdata.wakeup),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    } else .{
        .opcode = .MSG_RING,
        .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
        .ioprio = 0,
        .fd = token,
        .off = @intFromPtr(fiber) | 0b01,
        .addr = @backingInt(linux.IORING_MSG_RING_COMMAND.DATA),
        .len = 0,
        .rw_flags = 0,
        .user_data = @backingInt(Completion.Userdata.cleanup),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
    thread.submit();
}

/// Registers `fiber` to be resumed by the batch's next completion, once `fiber` is saved.
fn batchAwaitPending(s: *Scheduler, fiber: *Fiber, context: *anyopaque) void {
    const batch: *Io.Batch = @ptrCast(@alignCast(context));
    if (@cmpxchgStrong(
        ?*anyopaque,
        &batch.userdata,
        null,
        fiber,
        .release,
        .monotonic,
    )) |head| {
        assert(@as(u2, @truncate(@intFromPtr(head))) != 0b00);
        s.ready(.current(), fiber);
    }
}

/// A thread that is not one of the workers blocks in the kernel. io_uring's futex operations and
/// the futex system calls wake each other.
fn futexWaitForeign(ev: *Evented, ptr: *const u32, expected: u32, timeout: Io.Timeout) void {
    var timespec: linux.timespec = undefined;
    const timespec_ptr: ?*const linux.timespec = if (timeout.toDurationFromNow(ev.io())) |duration| ts: {
        const ns = @max(0, duration.raw.toNanoseconds());
        timespec = .{
            .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
            .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
        };
        break :ts &timespec;
    } else null;
    _ = linux.futex_4arg(ptr, .{ .cmd = .WAIT, .private = true }, expected, timespec_ptr);
}

fn futexWait(
    userdata: ?*anyopaque,
    ptr: *const u32,
    expected: u32,
    timeout: Io.Timeout,
) Io.Cancelable!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (Scheduler.Worker.currentOrNull() == null) return futexWaitForeign(ev, ptr, expected, timeout);
    const timespec: ?linux.kernel_timespec, const clock: Io.Clock, const timeout_flags: u32 = timespec: switch (timeout) {
        .none => .{
            null,
            .awake,
            linux.IORING_TIMEOUT_ABS,
        },
        .duration => |duration| {
            const ns = duration.raw.toNanoseconds();
            break :timespec .{
                .{
                    .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                    .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                },
                duration.clock,
                0,
            };
        },
        .deadline => |deadline| {
            const ns = deadline.raw.toNanoseconds();
            break :timespec .{
                .{
                    .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                    .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                },
                deadline.clock,
                linux.IORING_TIMEOUT_ABS,
            };
        },
    };
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    const thread = try cancel_region.awaitIoUring();
    thread.enqueue().* = .{
        .opcode = .FUTEX_WAIT,
        .flags = if (timespec) |_| linux.IOSQE_IO_LINK else 0,
        .ioprio = 0,
        .fd = @bitCast(linux.FUTEX2_FLAGS{ .size = .U32, .private = true }),
        .off = expected,
        .addr = @intFromPtr(ptr),
        .len = 0,
        .rw_flags = 0,
        .user_data = @intFromPtr(cancel_region.fiber),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = std.math.maxInt(u32),
        .resv = 0,
    };
    if (timespec) |*timespec_ptr| thread.enqueue().* = .{
        .opcode = .LINK_TIMEOUT,
        .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
        .ioprio = 0,
        .fd = 0,
        .off = 0,
        .addr = @intFromPtr(timespec_ptr),
        .len = 1,
        .rw_flags = timeout_flags | @as(u32, switch (clock) {
            .real => linux.IORING_TIMEOUT_REALTIME,
            else => 0,
            .boot => linux.IORING_TIMEOUT_BOOTTIME,
        }),
        .user_data = @backingInt(Completion.Userdata.wakeup),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
    ev.sched.park();
    switch (cancel_region.errno()) {
        .SUCCESS => {}, // notified by `wake()`
        .INTR, .CANCELED => {}, // caller's responsibility to retry
        .AGAIN => {}, // ptr.* != expect
        .INVAL => {}, // possibly timeout overflow
        .TIMEDOUT => unreachable,
        .FAULT => recoverableOsBugDetected(), // ptr was invalid
        else => recoverableOsBugDetected(),
    }
}

fn futexWaitUncancelable(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (Scheduler.Worker.currentOrNull() == null) return futexWaitForeign(ev, ptr, expected, .none);
    var cancel_region: CancelRegion = .initBlocked();
    defer cancel_region.deinit();
    const thread = cancel_region.awaitIoUring() catch |err| switch (err) {
        error.Canceled => unreachable, // blocked
    };
    thread.enqueue().* = .{
        .opcode = .FUTEX_WAIT,
        .flags = 0,
        .ioprio = 0,
        .fd = @bitCast(linux.FUTEX2_FLAGS{ .size = .U32, .private = true }),
        .off = expected,
        .addr = @intFromPtr(ptr),
        .len = 0,
        .rw_flags = 0,
        .user_data = @intFromPtr(cancel_region.fiber),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = std.math.maxInt(u32),
        .resv = 0,
    };
    ev.sched.park();
    switch (cancel_region.errno()) {
        .SUCCESS => {}, // notified by `wake()`
        .INTR, .CANCELED => {}, // caller's responsibility to retry
        .AGAIN => {}, // ptr.* != expect
        .INVAL => {}, // possibly timeout overflow
        .FAULT => recoverableOsBugDetected(), // ptr was invalid
        else => recoverableOsBugDetected(),
    }
}

fn futexWake(userdata: ?*anyopaque, ptr: *const u32, max_waiters: u32) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    // The kernel takes the count as an `int`, and wakes one waiter for a negative one.
    const n: u32 = @min(max_waiters, std.math.maxInt(i32));
    if (Scheduler.Worker.currentOrNull() == null) {
        _ = linux.futex_4arg(ptr, .{ .cmd = .WAKE, .private = true }, n, null);
        return;
    }
    const thread: *Thread = .current();
    thread.enqueue().* = .{
        .opcode = .FUTEX_WAKE,
        .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
        .ioprio = 0,
        .fd = @bitCast(linux.FUTEX2_FLAGS{ .size = .U32, .private = true }),
        .off = n,
        .addr = @intFromPtr(ptr),
        .len = 0,
        .rw_flags = 0,
        .user_data = @backingInt(Completion.Userdata.futex_wake),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = std.math.maxInt(u32),
        .resv = 0,
    };
    thread.submit();
}

fn operate(userdata: ?*anyopaque, operation: Io.Operation) Io.Cancelable!Io.Operation.Result {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    return switch (operation) {
        .file_read_streaming => |o| .{
            .file_read_streaming = ev.fileReadStreaming(
                &maybe_sync.cancel_region,
                o.file,
                o.data,
            ) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| e,
            },
        },
        .file_write_streaming => |o| .{
            .file_write_streaming = ev.fileWriteStreaming(
                &maybe_sync.cancel_region,
                o.file,
                o.header,
                o.data,
                o.splat,
            ) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| e,
            },
        },
        .device_io_control => |o| .{
            .device_io_control = try ev.deviceIoControl(try maybe_sync.enterSync(ev), o),
        },
        .net_receive => |o| .{
            .net_receive = r: {
                const opt_err, const n = ev.netReceive(&maybe_sync.cancel_region, o.socket_handle, o.message_buffer, o.data_buffer, o.flags);
                break :r .{
                    if (opt_err) |err| switch (err) {
                        error.Canceled => |e| return e,
                        else => |e| e,
                    } else null,
                    n,
                };
            },
        },
        .net_send => |o| .{
            .net_send = r: {
                const maybe_err, const count = ev.netSend(
                    &maybe_sync.cancel_region,
                    o.socket_handle,
                    o.messages,
                    o.flags,
                ) catch |err| switch (err) {
                    error.Canceled => |e| return e,
                    else => unreachable, // `netSend` only returns `error.Canceled`
                };
                break :r .{ maybe_err, count };
            },
        },
        .net_read => |o| .{
            .net_read = ev.netRead(&maybe_sync.cancel_region, o.socket_handle, o.data) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| e,
            },
        },
        .net_write => |o| .{
            .net_write = ev.netWrite(
                &maybe_sync.cancel_region,
                o.socket_handle,
                o.header,
                o.data,
                o.splat,
            ) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| e,
            },
        },
    };
}

fn fileReadStreaming(
    ev: *Evented,
    cancel_region: *CancelRegion,
    file: File,
    data: []const []u8,
) File.ReadStreamingError!usize {
    var iovecs_buffer: [max_iovecs_len]iovec = undefined;
    var i: usize = 0;
    for (data) |buf| {
        if (iovecs_buffer.len - i == 0) break;
        if (buf.len > 0) {
            iovecs_buffer[i] = .{ .base = buf.ptr, .len = buf.len };
            i += 1;
        }
    }
    const dest = iovecs_buffer[0..i];
    assert(dest[0].len > 0);

    const n = try ev.preadv(cancel_region, file.handle, dest, null);
    return if (n == 0) error.EndOfStream else n;
}

fn fileWriteStreaming(
    ev: *Evented,
    cancel_region: *CancelRegion,
    file: File,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
) File.Writer.Error!usize {
    var iovecs: [max_iovecs_len]iovec_const = undefined;
    var iovlen: iovlen_t = 0;
    addBuf(&iovecs, &iovlen, header);
    for (data[0 .. data.len - 1]) |bytes| addBuf(&iovecs, &iovlen, bytes);
    const pattern = data[data.len - 1];
    var backup_buffer: [splat_buffer_size]u8 = undefined;
    if (iovecs.len - iovlen != 0) switch (splat) {
        0 => {},
        1 => addBuf(&iovecs, &iovlen, pattern),
        else => switch (pattern.len) {
            0 => {},
            1 => {
                const splat_buffer = &backup_buffer;
                const memset_len = @min(splat_buffer.len, splat);
                const buf = splat_buffer[0..memset_len];
                @memset(buf, pattern[0]);
                addBuf(&iovecs, &iovlen, buf);
                var remaining_splat = splat - buf.len;
                while (remaining_splat > splat_buffer.len and iovecs.len - iovlen != 0) {
                    assert(buf.len == splat_buffer.len);
                    addBuf(&iovecs, &iovlen, splat_buffer);
                    remaining_splat -= splat_buffer.len;
                }
                addBuf(&iovecs, &iovlen, splat_buffer[0..@min(remaining_splat, splat_buffer.len)]);
            },
            else => for (0..@min(splat, iovecs.len - iovlen)) |_| {
                addBuf(&iovecs, &iovlen, pattern);
            },
        },
    };
    return ev.writev(cancel_region, file.handle, iovecs[0..iovlen]);
}

fn deviceIoControl(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    o: Io.Operation.DeviceIoControl,
) Io.Cancelable!i32 {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.ioctl(o.file.handle, @bitCast(o.code), @intFromPtr(o.arg));
        switch (linux.errno(rc)) {
            .SUCCESS => return @bitCast(@as(u32, @truncate(rc))),
            .INTR => {},
            else => |err| return -@as(i32, @backingInt(err)),
        }
    }
}

fn batchAwaitAsync(userdata: ?*anyopaque, batch: *Io.Batch) Io.Cancelable!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    ev.batchDrainSubmitted(&maybe_sync, batch, false) catch |err| switch (err) {
        error.ConcurrencyUnavailable => unreachable, // passed concurrency=false
        error.Canceled => |e| return e,
    };
    maybe_sync.leaveSync(ev);
    while (true) {
        batchDrainReady(batch) catch |err| switch (err) {
            error.Timeout => unreachable, // no timeout
        };
        if (batch.completed.head != .none or batch.pending.head == .none) return;
        ev.sched.yield(null, .{ .custom = .{ .context = batch, .run = batchAwaitPending } });
    }
}

fn batchAwaitConcurrent(
    userdata: ?*anyopaque,
    batch: *Io.Batch,
    timeout: Io.Timeout,
) Io.Batch.AwaitConcurrentError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    try ev.batchDrainSubmitted(&maybe_sync, batch, true);
    maybe_sync.leaveSync(ev);
    const timespec: linux.kernel_timespec, const clock: Io.Clock, const timeout_flags: u32 = while (true) {
        batchDrainReady(batch) catch |err| switch (err) {
            error.Timeout => unreachable, // no timeout
        };
        if (batch.completed.head != .none or batch.pending.head == .none) return;
        switch (timeout) {
            .none => ev.sched.yield(null, .{ .custom = .{ .context = batch, .run = batchAwaitPending } }),
            .duration => |duration| {
                const ns = duration.raw.toNanoseconds();
                break .{
                    .{
                        .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                        .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                    },
                    duration.clock,
                    0,
                };
            },
            .deadline => |deadline| {
                const ns = deadline.raw.toNanoseconds();
                break .{
                    .{
                        .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                        .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                    },
                    deadline.clock,
                    linux.IORING_TIMEOUT_ABS,
                };
            },
        }
    };
    {
        const thread = try maybe_sync.cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .TIMEOUT,
            .flags = 0,
            .ioprio = 0,
            .fd = 0,
            .off = 0,
            .addr = @intFromPtr(&timespec),
            .len = 1,
            .rw_flags = timeout_flags | @as(u32, switch (clock) {
                .real => linux.IORING_TIMEOUT_REALTIME,
                else => 0,
                .boot => linux.IORING_TIMEOUT_BOOTTIME,
            }),
            .user_data = @intFromPtr(&batch.userdata) | 0b11,
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
    }
    while (batch.completed.head == .none and batch.pending.head != .none) {
        ev.sched.yield(null, .{ .custom = .{ .context = batch, .run = batchAwaitPending } });
        batchDrainReady(batch) catch |err| switch (err) {
            error.Timeout => |e| return if (batch.completed.head == .none and
                batch.pending.head != .none) e,
        };
    }
    const thread = try maybe_sync.cancel_region.awaitIoUring();
    thread.enqueue().* = .{
        .opcode = .TIMEOUT_REMOVE,
        .flags = 0,
        .ioprio = 0,
        .fd = 0,
        .off = 0,
        .addr = @intFromPtr(&batch.userdata) | 0b11,
        .len = 0,
        .rw_flags = 0,
        .user_data = @intFromPtr(maybe_sync.cancel_region.fiber),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
    ev.sched.park();
    switch (maybe_sync.cancel_region.errno()) {
        .SUCCESS => return,
        .BUSY, .NOENT => {},
        else => |err| unexpectedErrno(err) catch {},
    }
    while (true) {
        batchDrainReady(batch) catch |err| switch (err) {
            error.Timeout => return,
        };
        ev.sched.yield(null, .{ .custom = .{ .context = batch, .run = batchAwaitPending } });
    }
}

/// If `concurrency` is false, `error.ConcurrencyUnavailable` is unreachable.
fn batchDrainSubmitted(
    ev: *Evented,
    maybe_sync: *CancelRegion.Sync.Maybe,
    batch: *Io.Batch,
    concurrency: bool,
) (Io.ConcurrentError || Io.Cancelable)!void {
    var index = batch.submitted.head;
    if (index == .none) return;
    const thread = try maybe_sync.cancelRegion().awaitIoUring();
    errdefer batch.submitted.head = index;
    while (index != .none) {
        const storage = &batch.storage[index.toIndex()];
        const next_index = storage.submission.node.next;
        if (@as(?Io.Operation.Result, result: switch (storage.submission.operation) {
            .file_read_streaming => |o| {
                const buffer = for (o.data) |buffer| {
                    if (buffer.len > 0) break buffer;
                } else break :result .{ .file_read_streaming = 0 };
                const fd = o.file.handle;
                storage.* = .{ .pending = .{
                    .node = .{ .prev = batch.pending.tail, .next = .none },
                    .tag = .file_read_streaming,
                    .userdata = undefined,
                } };
                thread.enqueue().* = .{
                    .opcode = .READ,
                    .flags = 0,
                    .ioprio = 0,
                    .fd = fd,
                    .off = std.math.maxInt(u64),
                    .addr = @intFromPtr(buffer.ptr),
                    .len = @min(buffer.len, 0xfffff000),
                    .rw_flags = 0,
                    .user_data = @intFromPtr(&storage.pending.userdata) | 0b10,
                    .buf_index = 0,
                    .personality = 0,
                    .splice_fd_in = 0,
                    .addr3 = 0,
                    .resv = 0,
                };
                break :result null;
            },
            .file_write_streaming => |o| {
                const buffer = buffer: {
                    if (o.header.len != 0) break :buffer o.header;
                    for (o.data[0 .. o.data.len - 1]) |buffer| {
                        if (buffer.len > 0) break :buffer buffer;
                    }
                    if (o.splat > 0) break :buffer o.data[o.data.len - 1];
                    break :result .{ .file_write_streaming = 0 };
                };
                const fd = o.file.handle;
                storage.* = .{ .pending = .{
                    .node = .{ .prev = batch.pending.tail, .next = .none },
                    .tag = .file_write_streaming,
                    .userdata = undefined,
                } };
                thread.enqueue().* = .{
                    .opcode = .WRITE,
                    .flags = 0,
                    .ioprio = 0,
                    .fd = fd,
                    .off = std.math.maxInt(u64),
                    .addr = @intFromPtr(buffer.ptr),
                    .len = @min(buffer.len, 0xfffff000),
                    .rw_flags = 0,
                    .user_data = @intFromPtr(&storage.pending.userdata) | 0b10,
                    .buf_index = 0,
                    .personality = 0,
                    .splice_fd_in = 0,
                    .addr3 = 0,
                    .resv = 0,
                };
                break :result null;
            },
            .device_io_control => |o| if (concurrency)
                return error.ConcurrencyUnavailable
            else
                .{ .device_io_control = try ev.deviceIoControl(try maybe_sync.enterSync(ev), o) },
            .net_receive => |o| {
                batchNetSubmit(thread, batch, storage, .net_receive, o.socket_handle, poll_mask_readable, .{
                    packHandleLen(o.socket_handle, o.data_buffer.len),
                    @intFromPtr(o.message_buffer.ptr),
                    packLenBits(o.message_buffer.len, @as(u8, @bitCast(o.flags))),
                    @intFromPtr(o.data_buffer.ptr),
                });
                break :result null;
            },
            .net_send => |o| {
                batchNetSubmit(thread, batch, storage, .net_send, o.socket_handle, poll_mask_writable, .{
                    @intCast(o.socket_handle),
                    @intFromPtr(o.messages.ptr),
                    packLenBits(o.messages.len, @as(u8, @bitCast(o.flags))),
                    0,
                });
                break :result null;
            },
            .net_read => |o| {
                batchNetSubmit(thread, batch, storage, .net_read, o.socket_handle, poll_mask_readable, .{
                    @intCast(o.socket_handle),
                    @intFromPtr(o.data.ptr),
                    o.data.len,
                    0,
                });
                break :result null;
            },
            .net_write => |o| {
                batchNetSubmit(thread, batch, storage, .net_write, o.socket_handle, poll_mask_writable, .{
                    packHandleLen(o.socket_handle, o.data.len),
                    @intFromPtr(o.header.ptr),
                    packLenBits(o.header.len, @intCast(@min(o.splat, net_write_splat_max))),
                    @intFromPtr(o.data.ptr),
                });
                break :result null;
            },
        })) |result| {
            switch (batch.completed.tail) {
                .none => batch.completed.head = index,
                else => |tail_index| batch.storage[tail_index.toIndex()].completion.node.next = index,
            }
            batch.completed.tail = index;
            storage.* = .{ .completion = .{ .node = .{ .next = .none }, .result = result } };
        } else {
            switch (batch.pending.tail) {
                .none => batch.pending.head = index,
                else => |tail_index| batch.storage[tail_index.toIndex()].pending.node.next = index,
            }
            batch.pending.tail = index;
            storage.pending.userdata[0] = @intFromPtr(batch);
        }
        index = next_index;
    }
    batch.submitted = .{ .head = .none, .tail = .none };
}

fn batchDrainReady(batch: *Io.Batch) Io.Timeout.Error!void {
    while (@atomicRmw(?*anyopaque, &batch.userdata, .Xchg, null, .acquire)) |head| {
        var next: usize = @intFromPtr(head);
        var timeout = false;
        while (cond: switch (@as(u2, @truncate(next))) {
            0b00 => if (timeout) return error.Timeout else false,
            0b01 => {
                assert(!timeout);
                return error.Timeout;
            },
            0b10 => true,
            0b11 => {
                assert(!timeout);
                timeout = true;
                break :cond true;
            },
        }) {
            const operation_userdata: *Io.Operation.Storage.Pending.Userdata =
                @ptrFromInt(next & ~@as(usize, 0b11));
            next = operation_userdata[0];
            const completion: Completion = .{
                .result = @bitCast(@as(u32, @intCast(operation_userdata[1]))),
                .flags = @intCast(operation_userdata[2]),
            };
            const pending: *Io.Operation.Storage.Pending =
                @fieldParentPtr("userdata", operation_userdata);
            const storage: *Io.Operation.Storage = @fieldParentPtr("pending", pending);
            const index: Io.Operation.OptionalIndex = .fromIndex(storage - batch.storage.ptr);
            assert(completion.flags & linux.IORING_CQE_F_SKIP == 0);
            switch (pending.node.prev) {
                .none => batch.pending.head = pending.node.next,
                else => |prev_index| batch.storage[prev_index.toIndex()].pending.node.next =
                    pending.node.next,
            }
            switch (pending.node.next) {
                .none => batch.pending.tail = pending.node.prev,
                else => |prev_index| batch.storage[prev_index.toIndex()].pending.node.prev =
                    pending.node.prev,
            }
            if (@as(?Io.Operation.Result, result: switch (pending.tag) {
                .file_read_streaming => .{
                    .file_read_streaming = switch (completion.errno()) {
                        .SUCCESS => @as(u32, @bitCast(completion.result)),
                        .INTR => 0,
                        .CANCELED => break :result null,
                        .INVAL => |err| errnoBug(err),
                        .FAULT => |err| errnoBug(err),
                        .AGAIN => error.WouldBlock,
                        .BADF => |err| errnoBug(err), // File descriptor used after closed
                        .IO => error.InputOutput,
                        .ISDIR => error.IsDir,
                        .NOBUFS => error.SystemResources,
                        .NOMEM => error.SystemResources,
                        .NOTCONN => error.SocketUnconnected,
                        .CONNRESET => error.ConnectionResetByPeer,
                        else => |err| unexpectedErrno(err),
                    },
                },
                .file_write_streaming => .{
                    .file_write_streaming = switch (completion.errno()) {
                        .SUCCESS => @as(u32, @bitCast(completion.result)),
                        .INTR => 0,
                        .CANCELED => break :result null,
                        .INVAL => |err| errnoBug(err),
                        .FAULT => |err| errnoBug(err),
                        .AGAIN => error.WouldBlock,
                        .BADF => error.NotOpenForWriting, // Can be a race condition.
                        .DESTADDRREQ => |err| errnoBug(err), // `connect` was never called.
                        .DQUOT => error.DiskQuota,
                        .FBIG => error.FileTooBig,
                        .IO => error.InputOutput,
                        .NOSPC => error.NoSpaceLeft,
                        .PERM => error.PermissionDenied,
                        .PIPE => error.BrokenPipe,
                        .CONNRESET => |err| errnoBug(err), // Not a socket handle.
                        .BUSY => error.DeviceBusy,
                        else => |err| unexpectedErrno(err),
                    },
                },
                .device_io_control => unreachable,
                .net_receive => net: {
                    const state = storage.pending.userdata[batch_net_state_offset .. batch_net_state_offset + batch_net_state_len].*;
                    const handle, const data_buffer_len = unpackHandleLen(state[0]);
                    const message_buffer_len, const flags_bits = unpackLenBits(state[2]);
                    const message_buffer = @as(
                        [*]net.IncomingMessage,
                        @ptrFromInt(state[1]),
                    )[0..message_buffer_len];
                    const data_buffer = @as([*]u8, @ptrFromInt(state[3]))[0..data_buffer_len];
                    if (completion.result < 0) switch (completion.errno()) {
                        .INTR, .CANCELED => break :net null,
                        else => |err| break :net .{ .net_receive = .{ netReceiveErrno(err), 0 } },
                    };
                    break :net .{ .net_receive = netReceiveSync(
                        handle,
                        message_buffer,
                        data_buffer,
                        @bitCast(@as(u8, @truncate(flags_bits))),
                    ) };
                },
                .net_send => net: {
                    const state = storage.pending.userdata[batch_net_state_offset .. batch_net_state_offset + batch_net_state_len].*;
                    const handle: net.Socket.Handle = @intCast(state[0]);
                    const messages_len, const flags_bits = unpackLenBits(state[2]);
                    const messages = @as(
                        [*]net.OutgoingMessage,
                        @ptrFromInt(state[1]),
                    )[0..messages_len];
                    if (completion.result < 0) switch (completion.errno()) {
                        .INTR, .CANCELED => break :net null,
                        else => |err| break :net .{ .net_send = .{ sendmsgErrno(err), 0 } },
                    };
                    break :net .{ .net_send = netSendSync(handle, messages, @bitCast(
                        @as(u8, @truncate(flags_bits)),
                    )) };
                },
                .net_read => net: {
                    const state = storage.pending.userdata[batch_net_state_offset .. batch_net_state_offset + batch_net_state_len].*;
                    const handle: net.Socket.Handle = @intCast(state[0]);
                    const data = @as([*][]u8, @ptrFromInt(state[1]))[0..state[2]];
                    if (completion.result < 0) switch (completion.errno()) {
                        .INTR, .CANCELED => break :net null,
                        else => |err| break :net .{ .net_read = readvErrno(err) },
                    };
                    break :net .{ .net_read = netReadSync(handle, data) };
                },
                .net_write => net: {
                    const state = storage.pending.userdata[batch_net_state_offset .. batch_net_state_offset + batch_net_state_len].*;
                    const handle, const data_len = unpackHandleLen(state[0]);
                    const header_len, const splat = unpackLenBits(state[2]);
                    const header = @as([*]const u8, @ptrFromInt(state[1]))[0..header_len];
                    const data = @as(
                        [*]const []const u8,
                        @ptrFromInt(state[3]),
                    )[0..data_len];
                    if (completion.result < 0) switch (completion.errno()) {
                        .INTR, .CANCELED => break :net null,
                        else => |err| break :net .{ .net_write = netWriteErrno(err) },
                    };
                    break :net .{ .net_write = netWriteSync(handle, header, data, splat) };
                },
            })) |result| {
                switch (batch.completed.tail) {
                    .none => batch.completed.head = index,
                    else => |tail_index| batch.storage[tail_index.toIndex()].completion.node.next =
                        index,
                }
                storage.* = .{ .completion = .{ .node = .{ .next = .none }, .result = result } };
                batch.completed.tail = index;
            } else {
                switch (batch.unused.tail) {
                    .none => batch.unused.head = index,
                    else => |tail_index| batch.storage[tail_index.toIndex()].unused.next = index,
                }
                storage.* = .{ .unused = .{ .prev = batch.unused.tail, .next = .none } };
                batch.unused.tail = index;
            }
        }
    }
}

fn batchCancel(userdata: ?*anyopaque, batch: *Io.Batch) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    batchDrainReady(batch) catch |err| switch (err) {
        error.Timeout => unreachable, // no timeout
    };
    var index = batch.pending.head;
    if (index == .none) return;
    var cancel_region: CancelRegion = .initBlocked();
    defer cancel_region.deinit();
    const thread = cancel_region.awaitIoUring() catch |err| switch (err) {
        error.Canceled => unreachable, // blocked
    };
    while (index != .none) {
        const pending = &batch.storage[index.toIndex()].pending;
        thread.enqueue().* = .{
            .opcode = .ASYNC_CANCEL,
            .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
            .ioprio = 0,
            .fd = 0,
            .off = 0,
            .addr = @intFromPtr(&pending.userdata) | 0b10,
            .len = 0,
            .rw_flags = 0,
            .user_data = @backingInt(Completion.Userdata.wakeup),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        index = pending.node.next;
    }
    while (true) {
        batchDrainReady(batch) catch |err| switch (err) {
            error.Timeout => unreachable, // no timeout
        };
        if (batch.pending.head == .none) return;
        // Parked until the next of the operations finishes, as they are canceled.
        ev.sched.yield(null, .{ .custom = .{ .context = batch, .run = batchAwaitPending } });
    }
}

fn dirCreateDir(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    permissions: Dir.Permissions,
) Dir.CreateDirError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.mkdirat(dir.handle, sub_path_posix, permissions.toMode()))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .PERM => return error.PermissionDenied,
            .DQUOT => return error.DiskQuota,
            .EXIST => return error.PathAlreadyExists,
            .FAULT => |err| return errnoBug(err),
            .LOOP => return error.SymLinkLoop,
            .MLINK => return error.LinkQuotaExceeded,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .NOTDIR => return error.NotDir,
            .ROFS => return error.ReadOnlyFileSystem,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirCreateDirPath(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    permissions: Dir.Permissions,
) Dir.CreateDirPathError!Dir.CreatePathStatus {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var it = Dir.path.componentIterator(sub_path);
    var status: Dir.CreatePathStatus = .existed;
    var component = it.last() orelse return error.BadPathName;
    while (true) {
        if (dirCreateDir(ev, dir, component.path, permissions)) |_| {
            status = .created;
        } else |err| switch (err) {
            error.PathAlreadyExists => {
                // stat the file and return an error if it's not a directory
                // this is important because otherwise a dangling symlink
                // could cause an infinite loop
                const kind = try ev.filePathKind(dir, component.path);
                if (kind != .directory) return error.NotDir;
            },
            error.FileNotFound => |e| {
                component = it.previous() orelse return e;
                continue;
            },
            else => |e| return e,
        }
        component = it.next() orelse return status;
    }
}

fn filePathKind(ev: *Evented, dir: Dir, sub_path: []const u8) !File.Kind {
    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        var statx_buf = std.mem.zeroes(linux.Statx);
        switch (linux.errno(linux.statx(
            dir.handle,
            sub_path_posix,
            linux.AT.NO_AUTOMOUNT | linux.AT.SYMLINK_NOFOLLOW,
            .{ .TYPE = true },
            &statx_buf,
        ))) {
            .SUCCESS => {
                if (!statx_buf.mask.TYPE) return error.Unexpected;
                return statxKind(statx_buf.mode);
            },
            .INTR => {},
            .ACCES => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .LOOP => |err| return errnoBug(err),
            .NAMETOOLONG => |err| return errnoBug(err),
            .NOENT => |err| return errnoBug(err),
            .NOMEM => return error.SystemResources,
            .NOTDIR => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirCreateDirPathOpen(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    permissions: Dir.Permissions,
    options: Dir.OpenOptions,
) Dir.CreateDirPathOpenError!Dir {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return dirOpenDir(ev, dir, sub_path, options) catch |err| switch (err) {
        error.FileNotFound => {
            _ = try dirCreateDirPath(ev, dir, sub_path, permissions);
            return dirOpenDir(ev, dir, sub_path, options);
        },
        else => |e| return e,
    };
}

fn dirOpenDir(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    options: Dir.OpenOptions,
) Dir.OpenError!Dir {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    return .{
        .handle = ev.openat(&cancel_region, dir.handle, sub_path_posix, .{
            .ACCMODE = .RDONLY,
            .DIRECTORY = true,
            .NOFOLLOW = !options.follow_symlinks,
            .CLOEXEC = true,
            .PATH = !options.iterate,
        }, 0) catch |err| switch (err) {
            error.IsDir => return errnoBug(.ISDIR),
            error.WouldBlock => return errnoBug(.AGAIN),
            error.FileTooBig => return errnoBug(.FBIG),
            error.NoSpaceLeft => return errnoBug(.NOSPC),
            error.DeviceBusy => return errnoBug(.BUSY), // EXCL unset.
            error.FileBusy => return errnoBug(.TXTBSY),
            error.PathAlreadyExists => return errnoBug(.EXIST), // Not creating.
            error.OperationUnsupported => return errnoBug(.OPNOTSUPP), // No TMPFILE, no locks.
            error.ReadOnlyFileSystem => return errnoBug(.ROFS), // Not creating.
            else => |e| return e,
        },
    };
}

fn dirStat(userdata: ?*anyopaque, dir: Dir) Dir.StatError!Dir.Stat {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.stat(&sync, dir.handle);
}

fn dirStatFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    options: Dir.StatFileOptions,
) Dir.StatFileError!File.Stat {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.statx(&sync, dir.handle, sub_path_posix, linux.AT.NO_AUTOMOUNT |
        @as(u32, if (options.follow_symlinks) 0 else linux.AT.SYMLINK_NOFOLLOW));
}

fn dirAccess(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    options: Dir.AccessOptions,
) Dir.AccessError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    const mode: u32 =
        @as(u32, if (options.read) linux.R_OK else 0) |
        @as(u32, if (options.write) linux.W_OK else 0) |
        @as(u32, if (options.execute) linux.X_OK else 0);
    const flags: u32 = if (options.follow_symlinks) 0 else linux.AT.SYMLINK_NOFOLLOW;

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.faccessat(dir.handle, sub_path_posix, mode, flags))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .PERM => return error.PermissionDenied,
            .ROFS => return error.ReadOnlyFileSystem,
            .LOOP => return error.SymLinkLoop,
            .TXTBSY => return error.FileBusy,
            .NOTDIR => return error.FileNotFound,
            .NOENT => return error.FileNotFound,
            .NAMETOOLONG => return error.NameTooLong,
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .IO => return error.InputOutput,
            .NOMEM => return error.SystemResources,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirCreateFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    flags: Dir.CreateFileOptions,
) File.OpenError!File {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const fd = ev.openatSync(try maybe_sync.enterSync(ev), dir.handle, sub_path_posix, .{
        .ACCMODE = if (flags.read) .RDWR else .WRONLY,
        .CREAT = true,
        .TRUNC = flags.truncate,
        .EXCL = flags.exclusive,
        .CLOEXEC = true,
    }, flags.permissions.toMode()) catch |err| switch (err) {
        error.OperationUnsupported => return error.Unexpected, // TMPFILE unset.
        else => |e| return e,
    };
    errdefer ev.closeAsync(fd);

    switch (flags.lock) {
        .none => {},
        .shared, .exclusive => try ev.flock(
            try maybe_sync.enterSync(ev),
            fd,
            flags.lock,
            if (flags.lock_nonblocking) .nonblocking else .blocking,
        ),
    }

    return .{ .handle = fd, .flags = .{ .nonblocking = false } };
}

fn dirCreateFileAtomic(
    userdata: ?*anyopaque,
    dir: Dir,
    dest_path: []const u8,
    options: Dir.CreateFileAtomicOptions,
) Dir.CreateFileAtomicError!File.Atomic {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    // Linux has O_TMPFILE, but linkat() does not support AT_REPLACE, so it's
    // useless when we have to make up a bogus path name to do the rename()
    // anyway.
    if (!options.replace) tmpfile: {
        const flags: linux.O = if (@hasField(linux.O, "TMPFILE")) .{
            .ACCMODE = .RDWR,
            .TMPFILE = true,
            .DIRECTORY = true,
            .CLOEXEC = true,
        } else if (@hasField(linux.O, "TMPFILE0") and !@hasField(linux.O, "TMPFILE2")) .{
            .ACCMODE = .RDWR,
            .TMPFILE0 = true,
            .TMPFILE1 = true,
            .DIRECTORY = true,
            .CLOEXEC = true,
        } else break :tmpfile;

        const dest_dirname = Dir.path.dirname(dest_path);
        if (dest_dirname) |dirname| {
            // This has a nice side effect of preemptively triggering EISDIR or
            // ENOENT, avoiding the ambiguity below.
            _ = dirCreateDirPath(ev, dir, dirname, .default_dir) catch |err| switch (err) {
                // None of these make sense in this context.
                error.IsDir,
                error.Streaming,
                error.DiskQuota,
                error.PathAlreadyExists,
                error.LinkQuotaExceeded,
                error.PipeBusy,
                error.FileTooBig,
                error.DeviceBusy,
                error.FileLocksUnsupported,
                error.FileBusy,
                => return error.Unexpected,

                else => |e| return e,
            };
        }

        var path_buffer: [PATH_MAX]u8 = undefined;
        const sub_path_posix = try pathToPosix(dest_dirname orelse ".", &path_buffer);

        var sync: CancelRegion.Sync = try .init(ev);
        defer sync.deinit(ev);
        return .{
            .file = .{
                .handle = ev.openatSync(
                    &sync,
                    dir.handle,
                    sub_path_posix,
                    flags,
                    options.permissions.toMode(),
                ) catch |err| switch (err) {
                    error.IsDir, error.FileNotFound, error.OperationUnsupported => {
                        // Ambiguous error code. It might mean the file system
                        // does not support O_TMPFILE. Therefore, we must fall
                        // back to not using O_TMPFILE.
                        break :tmpfile;
                    },
                    error.FileTooBig => return errnoBug(.FBIG),
                    error.DeviceBusy => return errnoBug(.BUSY), // O_EXCL not passed
                    error.PathAlreadyExists => return errnoBug(.EXIST), // Not creating.
                    else => |e| return e,
                },
                .flags = .{ .nonblocking = false },
            },
            .file_basename_hex = 0,
            .dest_sub_path = dest_path,
            .file_open = true,
            .file_exists = false,
            .close_dir_on_deinit = false,
            .dir = dir,
        };
    }

    if (Dir.path.dirname(dest_path)) |dirname| {
        const new_dir = if (options.make_path)
            dirCreateDirPathOpen(ev, dir, dirname, .default_dir, .{}) catch |err| switch (err) {
                // None of these make sense in this context.
                error.IsDir,
                error.Streaming,
                error.DiskQuota,
                error.PathAlreadyExists,
                error.LinkQuotaExceeded,
                error.PipeBusy,
                error.FileTooBig,
                error.FileLocksUnsupported,
                error.DeviceBusy,
                => return error.Unexpected,

                else => |e| return e,
            }
        else
            try dirOpenDir(ev, dir, dirname, .{});

        return ev.atomicFileInit(Dir.path.basename(dest_path), options.permissions, new_dir, true);
    }

    return ev.atomicFileInit(dest_path, options.permissions, dir, false);
}

fn atomicFileInit(
    ev: *Evented,
    dest_basename: []const u8,
    permissions: File.Permissions,
    dir: Dir,
    close_dir_on_deinit: bool,
) Dir.CreateFileAtomicError!File.Atomic {
    while (true) {
        var random_integer: u64 = undefined;
        random(ev, @ptrCast(&random_integer));
        const tmp_sub_path = std.fmt.hex(random_integer);
        const file = dirCreateFile(ev, dir, &tmp_sub_path, .{
            .permissions = permissions,
            .exclusive = true,
        }) catch |err| switch (err) {
            error.PathAlreadyExists => continue,
            error.DeviceBusy => continue,
            error.FileBusy => continue,

            error.IsDir => return error.Unexpected, // No path components.
            error.FileTooBig => return error.Unexpected, // Creating, not opening.
            error.FileLocksUnsupported => return error.Unexpected, // Not asking for locks.
            error.PipeBusy => return error.Unexpected, // Not opening a pipe.

            else => |e| return e,
        };
        return .{
            .file = file,
            .file_basename_hex = random_integer,
            .dest_sub_path = dest_basename,
            .file_open = true,
            .file_exists = true,
            .close_dir_on_deinit = close_dir_on_deinit,
            .dir = dir,
        };
    }
}

fn dirOpenFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    flags: Dir.OpenFileOptions,
) File.OpenError!File {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const fd = ev.openat(&maybe_sync.cancel_region, dir.handle, sub_path_posix, .{
        .ACCMODE = switch (flags.mode) {
            .read_only => .RDONLY,
            .write_only => .WRONLY,
            .read_write => .RDWR,
        },
        .NOCTTY = !flags.allow_ctty,
        .NOFOLLOW = !flags.follow_symlinks,
        .CLOEXEC = true,
        .PATH = flags.path_only,
    }, 0) catch |err| switch (err) {
        error.OperationUnsupported => return error.Unexpected, // TMPFILE unset.
        else => |e| return e,
    };
    errdefer ev.closeAsync(fd);

    if (!flags.allow_directory) {
        const is_dir = is_dir: {
            const s = ev.stat(try maybe_sync.enterSync(ev), fd) catch |err| switch (err) {
                // The directory-ness is either unknown or unknowable
                error.Streaming => break :is_dir false,
                else => |e| return e,
            };
            break :is_dir s.kind == .directory;
        };
        if (is_dir) return error.IsDir;
    }

    switch (flags.lock) {
        .none => {},
        .shared, .exclusive => try ev.flock(
            try maybe_sync.enterSync(ev),
            fd,
            flags.lock,
            if (flags.lock_nonblocking) .nonblocking else .blocking,
        ),
    }

    return .{ .handle = fd, .flags = .{ .nonblocking = false } };
}

fn dirClose(userdata: ?*anyopaque, dirs: []const Dir) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    for (dirs) |dir| ev.close(dir.handle);
}

fn dirRead(userdata: ?*anyopaque, dr: *Dir.Reader, buffer: []Dir.Entry) Dir.Reader.Error!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var buffer_index: usize = 0;
    while (buffer.len - buffer_index != 0) {
        if (dr.end - dr.index == 0) {
            // Refill the buffer, unless we've already created references to
            // buffered data.
            if (buffer_index != 0) break;
            var sync: CancelRegion.Sync = try .init(ev);
            defer sync.deinit(ev);
            if (dr.state == .reset) {
                ev.lseek(&sync, dr.dir.handle, 0, linux.SEEK.SET) catch |err| switch (err) {
                    error.Unseekable => return error.Unexpected,
                    else => |e| return e,
                };
                dr.state = .reading;
            }
            const n = while (true) {
                try sync.cancel_region.await(.nothing);
                const rc = linux.getdents64(dr.dir.handle, dr.buffer.ptr, @min(dr.buffer.len, std.math.maxInt(c_uint)));
                switch (linux.errno(rc)) {
                    .SUCCESS => break rc,
                    .INTR => {},
                    .BADF => |err| return errnoBug(err), // Dir is invalid or was opened without iteration ability.
                    .FAULT => |err| return errnoBug(err),
                    .NOTDIR => |err| return errnoBug(err),
                    // To be consistent across platforms, iteration
                    // ends if the directory being iterated is deleted
                    // during iteration. This matches the behavior of
                    // non-Linux, non-WASI UNIX platforms.
                    .NOENT => {
                        dr.state = .finished;
                        return 0;
                    },
                    // This can occur when reading /proc/$PID/net, or
                    // if the provided buffer is too small. Neither
                    // scenario is intended to be handled by this API.
                    .INVAL => return error.Unexpected,
                    .ACCES => return error.AccessDenied, // Lacking permission to iterate this directory.
                    else => |err| return unexpectedErrno(err),
                }
            };
            if (n == 0) {
                dr.state = .finished;
                return 0;
            }
            dr.index = 0;
            dr.end = n;
        }
        // Linux aligns the header by padding after the null byte of the name
        // to align the next entry. This means we can find the end of the name
        // by looking at only the 8 bytes before the next record. However since
        // file names are usually short it's better to keep the machine code
        // simpler.
        //
        // Furthermore, I observed qemu user mode to not align this struct, so
        // this code makes the conservative choice to not assume alignment.
        const linux_entry: *align(1) linux.dirent64 = @ptrCast(&dr.buffer[dr.index]);
        const next_index = dr.index + linux_entry.reclen;
        dr.index = next_index;
        const name_ptr: [*]u8 = &linux_entry.name;
        const padded_name = name_ptr[0 .. linux_entry.reclen - @offsetOf(linux.dirent64, "name")];
        const name_len = std.mem.findScalar(u8, padded_name, 0).?;
        const name = name_ptr[0..name_len :0];

        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;

        const entry_kind: File.Kind = switch (linux_entry.type) {
            linux.DT.BLK => .block_device,
            linux.DT.CHR => .character_device,
            linux.DT.DIR => .directory,
            linux.DT.FIFO => .named_pipe,
            linux.DT.LNK => .sym_link,
            linux.DT.REG => .file,
            linux.DT.SOCK => .unix_domain_socket,
            else => .unknown,
        };
        buffer[buffer_index] = .{
            .name = name,
            .kind = entry_kind,
            .inode = linux_entry.ino,
        };
        buffer_index += 1;
    }
    return buffer_index;
}

fn dirRealPath(userdata: ?*anyopaque, dir: Dir, out_buffer: []u8) Dir.RealPathError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.realPath(&sync, dir.handle, out_buffer);
}

fn dirRealPathFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    out_buffer: []u8,
) Dir.RealPathFileError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const fd = ev.openat(&maybe_sync.cancel_region, dir.handle, sub_path_posix, .{
        .CLOEXEC = true,
        .PATH = true,
    }, 0) catch |err| switch (err) {
        error.WouldBlock => return errnoBug(.AGAIN),
        error.OperationUnsupported => return errnoBug(.OPNOTSUPP), // Not asking for locks.
        error.ReadOnlyFileSystem => return errnoBug(.ROFS), // Not creating.
        else => |e| return e,
    };
    defer ev.closeAsync(fd);
    return ev.realPath(try maybe_sync.enterSync(ev), fd, out_buffer);
}

fn dirDeleteFile(userdata: ?*anyopaque, dir: Dir, sub_path: []const u8) Dir.DeleteFileError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.unlinkat(dir.handle, sub_path_posix, 0))) {
            .SUCCESS => return,
            .INTR => {},
            .PERM => return error.PermissionDenied,
            .ACCES => return error.AccessDenied,
            .BUSY => return error.FileBusy,
            .FAULT => |err| return errnoBug(err),
            .IO => return error.FileSystem,
            .ISDIR => return error.IsDir,
            .LOOP => return error.SymLinkLoop,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .NOMEM => return error.SystemResources,
            .ROFS => return error.ReadOnlyFileSystem,
            .EXIST => |err| return errnoBug(err),
            .NOTEMPTY => |err| return errnoBug(err), // Not passing AT.REMOVEDIR
            .ILSEQ => return error.BadPathName,
            .INVAL => |err| return errnoBug(err), // invalid flags, or pathname has . as last component
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirDeleteDir(userdata: ?*anyopaque, dir: Dir, sub_path: []const u8) Dir.DeleteDirError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.unlinkat(dir.handle, sub_path_posix, linux.AT.REMOVEDIR))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .PERM => return error.PermissionDenied,
            .BUSY => return error.FileBusy,
            .FAULT => |err| return errnoBug(err),
            .IO => return error.FileSystem,
            .ISDIR => |err| return errnoBug(err),
            .LOOP => return error.SymLinkLoop,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .NOMEM => return error.SystemResources,
            .ROFS => return error.ReadOnlyFileSystem,
            .EXIST => |err| return errnoBug(err),
            .NOTEMPTY => return error.DirNotEmpty,
            .ILSEQ => return error.BadPathName,
            .INVAL => |err| return errnoBug(err), // invalid flags, or pathname has . as last component
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirRename(
    userdata: ?*anyopaque,
    old_dir: Dir,
    old_sub_path: []const u8,
    new_dir: Dir,
    new_sub_path: []const u8,
) Dir.RenameError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var old_path_buffer: [PATH_MAX]u8 = undefined;
    var new_path_buffer: [PATH_MAX]u8 = undefined;

    const old_sub_path_posix = try pathToPosix(old_sub_path, &old_path_buffer);
    const new_sub_path_posix = try pathToPosix(new_sub_path, &new_path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.renameat(
        &sync,
        old_dir.handle,
        old_sub_path_posix,
        new_dir.handle,
        new_sub_path_posix,
        .{},
    ) catch |err| switch (err) {
        error.PathAlreadyExists => unreachable, // only with NOREPLACE
        else => |e| return e,
    };
}

fn dirRenamePreserve(
    userdata: ?*anyopaque,
    old_dir: Dir,
    old_sub_path: []const u8,
    new_dir: Dir,
    new_sub_path: []const u8,
) Dir.RenamePreserveError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var old_path_buffer: [PATH_MAX]u8 = undefined;
    var new_path_buffer: [PATH_MAX]u8 = undefined;

    const old_sub_path_posix = try pathToPosix(old_sub_path, &old_path_buffer);
    const new_sub_path_posix = try pathToPosix(new_sub_path, &new_path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.renameat(
        &sync,
        old_dir.handle,
        old_sub_path_posix,
        new_dir.handle,
        new_sub_path_posix,
        .{ .NOREPLACE = true },
    );
}

fn dirSymLink(
    userdata: ?*anyopaque,
    dir: Dir,
    target_path: []const u8,
    sym_link_path: []const u8,
    flags: Dir.SymLinkFlags,
) Dir.SymLinkError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = flags;

    var target_path_buffer: [PATH_MAX]u8 = undefined;
    var sym_link_path_buffer: [PATH_MAX]u8 = undefined;

    const target_path_posix = try pathToPosix(target_path, &target_path_buffer);
    const sym_link_path_posix = try pathToPosix(sym_link_path, &sym_link_path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.symlinkat(target_path_posix, dir.handle, sym_link_path_posix))) {
            .SUCCESS => return,
            .INTR => {},
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .ACCES => return error.AccessDenied,
            .PERM => return error.PermissionDenied,
            .DQUOT => return error.DiskQuota,
            .EXIST => return error.PathAlreadyExists,
            .IO => return error.FileSystem,
            .LOOP => return error.SymLinkLoop,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .ROFS => return error.ReadOnlyFileSystem,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirReadLink(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    buffer: []u8,
) Dir.ReadLinkError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var sub_path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &sub_path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.readlinkat(dir.handle, sub_path_posix, buffer.ptr, buffer.len);
        switch (linux.errno(rc)) {
            .SUCCESS => return @bitCast(rc),
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .FAULT => |err| return errnoBug(err),
            .INVAL => return error.NotLink,
            .IO => return error.FileSystem,
            .LOOP => return error.SymLinkLoop,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOTDIR => return error.NotDir,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirSetOwner(
    userdata: ?*anyopaque,
    dir: Dir,
    owner: ?File.Uid,
    group: ?File.Gid,
) Dir.SetOwnerError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    try ev.fchownat(
        &sync,
        dir.handle,
        "",
        owner orelse std.math.maxInt(linux.uid_t),
        group orelse std.math.maxInt(linux.gid_t),
        linux.AT.EMPTY_PATH,
    );
}

fn dirSetFileOwner(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    owner: ?File.Uid,
    group: ?File.Gid,
    options: Dir.SetFileOwnerOptions,
) Dir.SetFileOwnerError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    try ev.fchownat(
        &sync,
        dir.handle,
        sub_path_posix,
        owner orelse std.math.maxInt(linux.uid_t),
        group orelse std.math.maxInt(linux.gid_t),
        if (options.follow_symlinks) 0 else linux.AT.SYMLINK_NOFOLLOW,
    );
}

fn dirSetPermissions(
    userdata: ?*anyopaque,
    dir: Dir,
    permissions: Dir.Permissions,
) Dir.SetPermissionsError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    ev.fchmodat(
        &sync,
        dir.handle,
        "",
        permissions.toMode(),
        linux.AT.EMPTY_PATH,
    ) catch |err| switch (err) {
        error.NameTooLong => return errnoBug(.NAMETOOLONG),
        error.BadPathName => return errnoBug(.ILSEQ),
        error.ProcessFdQuotaExceeded => return errnoBug(.MFILE),
        error.SystemFdQuotaExceeded => return errnoBug(.NFILE),
        error.OperationUnsupported => return errnoBug(.OPNOTSUPP),
        else => |e| return e,
    };
}

fn dirSetFilePermissions(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    permissions: Dir.Permissions,
    options: Dir.SetFilePermissionsOptions,
) Dir.SetFilePermissionsError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    try ev.fchmodat(
        &sync,
        dir.handle,
        sub_path_posix,
        permissions.toMode(),
        if (options.follow_symlinks) 0 else linux.AT.SYMLINK_NOFOLLOW,
    );
}

fn dirSetTimestamps(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    options: Dir.SetTimestampsOptions,
) Dir.SetTimestampsError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var path_buffer: [PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);
    var cancel_region: CancelRegion.Sync = try .init(ev);
    defer cancel_region.deinit(ev);
    try ev.utimensat(
        &cancel_region,
        dir.handle,
        sub_path_posix,
        if (options.modify_timestamp != .now or options.access_timestamp != .now) &.{
            setTimestampToPosix(options.access_timestamp),
            setTimestampToPosix(options.modify_timestamp),
        } else null,
        if (options.follow_symlinks) 0 else linux.AT.SYMLINK_NOFOLLOW,
    );
}

fn dirHardLink(
    userdata: ?*anyopaque,
    old_dir: Dir,
    old_sub_path: []const u8,
    new_dir: Dir,
    new_sub_path: []const u8,
    options: Dir.HardLinkOptions,
) Dir.HardLinkError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var old_path_buffer: [PATH_MAX]u8 = undefined;
    var new_path_buffer: [PATH_MAX]u8 = undefined;

    const old_sub_path_posix = try pathToPosix(old_sub_path, &old_path_buffer);
    const new_sub_path_posix = try pathToPosix(new_sub_path, &new_path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.linkat(
        &sync,
        old_dir.handle,
        old_sub_path_posix,
        new_dir.handle,
        new_sub_path_posix,
        if (options.follow_symlinks) linux.AT.SYMLINK_FOLLOW else 0,
    );
}

fn fileStat(userdata: ?*anyopaque, file: File) File.StatError!File.Stat {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.stat(&sync, file.handle);
}

fn fileLength(userdata: ?*anyopaque, file: File) File.LengthError!u64 {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        var statx_buf = std.mem.zeroes(linux.Statx);
        switch (linux.errno(linux.statx(file.handle, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &statx_buf))) {
            .SUCCESS => {
                if (!statx_buf.mask.SIZE) return error.Unexpected;
                return statx_buf.size;
            },
            .INTR => {},
            .ACCES => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .LOOP => |err| return errnoBug(err),
            .NAMETOOLONG => |err| return errnoBug(err),
            .NOENT => |err| return errnoBug(err),
            .NOMEM => return error.SystemResources,
            .NOTDIR => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fileClose(userdata: ?*anyopaque, files: []const File) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    for (files) |file| ev.close(file.handle);
}

fn fileWritePositional(
    userdata: ?*anyopaque,
    file: File,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
    offset: u64,
) File.WritePositionalError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var iovecs: [max_iovecs_len]iovec_const = undefined;
    var iovlen: iovlen_t = 0;
    addBuf(&iovecs, &iovlen, header);
    for (data[0 .. data.len - 1]) |bytes| addBuf(&iovecs, &iovlen, bytes);
    const pattern = data[data.len - 1];
    var backup_buffer: [splat_buffer_size]u8 = undefined;
    if (iovecs.len - iovlen != 0) switch (splat) {
        0 => {},
        1 => addBuf(&iovecs, &iovlen, pattern),
        else => switch (pattern.len) {
            0 => {},
            1 => {
                const splat_buffer = &backup_buffer;
                const memset_len = @min(splat_buffer.len, splat);
                const buf = splat_buffer[0..memset_len];
                @memset(buf, pattern[0]);
                addBuf(&iovecs, &iovlen, buf);
                var remaining_splat = splat - buf.len;
                while (remaining_splat > splat_buffer.len and iovecs.len - iovlen != 0) {
                    assert(buf.len == splat_buffer.len);
                    addBuf(&iovecs, &iovlen, splat_buffer);
                    remaining_splat -= splat_buffer.len;
                }
                addBuf(&iovecs, &iovlen, splat_buffer[0..@min(remaining_splat, splat_buffer.len)]);
            },
            else => for (0..@min(splat, iovecs.len - iovlen)) |_| {
                addBuf(&iovecs, &iovlen, pattern);
            },
        },
    };

    // A positional write goes to a regular file or a block device, which io_uring writes on a
    // kernel worker thread unless the file system can write without blocking. ext4 and tmpfs
    // cannot: `pwritev2` with `RWF_NOWAIT` fails on both with `EOPNOTSUPP`. See
    // `CancelRegion.Sync`.
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return pwritevSync(&sync, file.handle, iovecs[0..iovlen], offset);
}

/// This is either usize or u32. Since, either is fine, let's use the same
/// `addBuf` function for both writing to a file and sending network messages.
const iovlen_t = @FieldType(linux.msghdr_const, "iovlen");

fn addBuf(v: []iovec_const, i: *iovlen_t, bytes: []const u8) void {
    // OS checks ptr addr before length so zero length vectors must be omitted.
    if (bytes.len == 0) return;
    if (v.len - i.* == 0) return;
    v[i.*] = .{ .base = bytes.ptr, .len = bytes.len };
    i.* += 1;
}

fn fileWriteFileStreaming(
    userdata: ?*anyopaque,
    file: File,
    header: []const u8,
    file_reader: *File.Reader,
    limit: Io.Limit,
) File.Writer.WriteFileError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const reader_buffered = file_reader.interface.buffered();
    if (header.len != 0 or reader_buffered.len != 0) {
        var cancel_region: CancelRegion = .init();
        defer cancel_region.deinit();
        const n = try ev.fileWriteStreaming(&cancel_region, file, header, &.{limit.slice(reader_buffered)}, 1);
        file_reader.interface.toss(n -| header.len);
        return n;
    }
    return copyFileRange(ev, file_reader, file, null, limit) catch |err| switch (err) {
        error.Unseekable => return error.Unimplemented, // the file is not a regular one
        else => |e| return e,
    };
}

fn fileWriteFilePositional(
    userdata: ?*anyopaque,
    file: File,
    header: []const u8,
    file_reader: *File.Reader,
    limit: Io.Limit,
    offset: u64,
) File.WriteFilePositionalError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const reader_buffered = file_reader.interface.buffered();
    if (header.len != 0 or reader_buffered.len != 0) {
        const n = try fileWritePositional(ev, file, header, &.{limit.slice(reader_buffered)}, 1, offset);
        file_reader.interface.toss(n -| header.len);
        return n;
    }
    return copyFileRange(ev, file_reader, file, offset, limit);
}

/// Copies from `file_reader`, whose buffer is empty, to `file` at `offset`, or at its position if
/// `offset` is null. io_uring has no `copy_file_range`, so it is a direct call. Returns
/// `error.Unimplemented` for a pair of files the kernel cannot copy between, so that the caller
/// reads and writes instead.
fn copyFileRange(
    ev: *Evented,
    file_reader: *File.Reader,
    file: File,
    offset: ?u64,
    limit: Io.Limit,
) File.WriteFilePositionalError!usize {
    if (file_reader.size) |size| if (size - file_reader.pos == 0) return error.EndOfStream;
    var len: usize = @min(@backingInt(limit), std.math.maxInt(i64) - (offset orelse 0));
    var off_in: i64 = undefined;
    const off_in_ptr: ?*i64 = switch (file_reader.mode) {
        .positional_simple, .streaming_simple => return error.Unimplemented,
        .positional => p: {
            len = @min(len, std.math.maxInt(i64) - file_reader.pos);
            off_in = @intCast(file_reader.pos);
            break :p &off_in;
        },
        .streaming => null,
        .failure => return error.ReadFailed,
    };
    var off_out: i64 = if (offset) |o| @intCast(o) else undefined;
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    const n: usize = while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.copy_file_range(
            file_reader.file.handle,
            off_in_ptr,
            file.handle,
            if (offset != null) &off_out else null,
            len,
            0,
        );
        switch (linux.errno(rc)) {
            .SUCCESS => break rc,
            .INTR => {},
            // The kernel cannot copy between these two files.
            .OPNOTSUPP, .INVAL, .NOSYS, .XDEV => return error.Unimplemented,
            .FBIG => return error.FileTooBig,
            .IO => return error.InputOutput,
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .OVERFLOW => |err| return errnoBug(err), // `len` keeps the offsets in range.
            .NXIO, .SPIPE => return error.Unseekable,
            .PERM => return error.PermissionDenied,
            .TXTBSY => return error.FileBusy,
            .ISDIR => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    };
    if (n == 0) {
        file_reader.size = file_reader.pos;
        return error.EndOfStream;
    }
    file_reader.pos += n;
    return n;
}

fn fileReadPositional(
    userdata: ?*anyopaque,
    file: File,
    data: []const []u8,
    offset: u64,
) File.ReadPositionalError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var iovecs_buffer: [max_iovecs_len]iovec = undefined;
    var i: usize = 0;
    for (data) |buf| {
        if (iovecs_buffer.len - i == 0) break;
        if (buf.len > 0) {
            iovecs_buffer[i] = .{ .base = buf.ptr, .len = buf.len };
            i += 1;
        }
    }
    if (i == 0) return 0;
    const dest = iovecs_buffer[0..i];
    assert(dest[0].len > 0);

    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    return ev.preadv(&cancel_region, file.handle, dest, offset) catch |err| switch (err) {
        error.SocketUnconnected => return errnoBug(.NOTCONN), // not a socket
        error.ConnectionResetByPeer => return errnoBug(.CONNRESET), // not a socket
        else => |e| return e,
    };
}

fn fileSeekBy(userdata: ?*anyopaque, file: File, offset: i64) File.SeekError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    try ev.lseek(&sync, file.handle, @bitCast(offset), linux.SEEK.CUR);
}

fn fileSeekTo(userdata: ?*anyopaque, file: File, offset: u64) File.SeekError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    try ev.lseek(&sync, file.handle, offset, linux.SEEK.SET);
}

fn fileSync(userdata: ?*anyopaque, file: File) File.SyncError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .FSYNC,
            .flags = 0,
            .ioprio = 0,
            .fd = file.handle,
            .off = 0,
            .addr = 0,
            .len = 0,
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            .BADF => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .ROFS => |err| return errnoBug(err),
            .IO => return error.InputOutput,
            .NOSPC => return error.NoSpaceLeft,
            .DQUOT => return error.DiskQuota,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fileIsTty(userdata: ?*anyopaque, file: File) Io.Cancelable!bool {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        var wsz: winsize = undefined;
        const rc = linux.ioctl(file.handle, linux.T.IOCGWINSZ, @intFromPtr(&wsz));
        switch (linux.errno(rc)) {
            .SUCCESS => return true,
            .INTR => {},
            else => return false,
        }
    }
}

fn fileEnableAnsiEscapeCodes(userdata: ?*anyopaque, file: File) File.EnableAnsiEscapeCodesError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (!try fileIsTty(ev, file)) return error.NotTerminalDevice;
}

fn fileSetLength(userdata: ?*anyopaque, file: File, length: u64) File.SetLengthError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.ftruncate(file.handle, @bitCast(length)))) {
            .SUCCESS => return,
            .INTR => {},
            .FBIG => return error.FileTooBig,
            .IO => return error.InputOutput,
            .PERM => return error.PermissionDenied,
            .TXTBSY => return error.FileBusy,
            .BADF => |err| return errnoBug(err), // Handle not open for writing.
            .INVAL => return error.NonResizable, // This is returned for /dev/null for example.
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fileSetOwner(
    userdata: ?*anyopaque,
    file: File,
    owner: ?File.Uid,
    group: ?File.Gid,
) File.SetOwnerError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    try ev.fchownat(
        &sync,
        file.handle,
        "",
        owner orelse std.math.maxInt(linux.uid_t),
        group orelse std.math.maxInt(linux.gid_t),
        linux.AT.EMPTY_PATH,
    );
}

fn fileSetPermissions(
    userdata: ?*anyopaque,
    file: File,
    permissions: File.Permissions,
) File.SetPermissionsError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    ev.fchmodat(
        &sync,
        file.handle,
        "",
        permissions.toMode(),
        linux.AT.EMPTY_PATH,
    ) catch |err| switch (err) {
        error.NameTooLong => return errnoBug(.NAMETOOLONG),
        error.BadPathName => return errnoBug(.ILSEQ),
        error.ProcessFdQuotaExceeded => return errnoBug(.MFILE),
        error.SystemFdQuotaExceeded => return errnoBug(.NFILE),
        error.OperationUnsupported => return errnoBug(.OPNOTSUPP),
        else => |e| return e,
    };
}

fn fileSetTimestamps(
    userdata: ?*anyopaque,
    file: File,
    options: File.SetTimestampsOptions,
) File.SetTimestampsError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    try ev.utimensat(
        &sync,
        file.handle,
        "",
        if (options.modify_timestamp != .now or options.access_timestamp != .now) &.{
            setTimestampToPosix(options.access_timestamp),
            setTimestampToPosix(options.modify_timestamp),
        } else null,
        linux.AT.EMPTY_PATH,
    );
}

fn fileLock(userdata: ?*anyopaque, file: File, lock: File.Lock) File.LockError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    ev.flock(&sync, file.handle, lock, .blocking) catch |err| switch (err) {
        error.WouldBlock => unreachable, // blocking
        else => |e| return e,
    };
}

fn fileTryLock(userdata: ?*anyopaque, file: File, lock: File.Lock) File.LockError!bool {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    ev.flock(&sync, file.handle, lock, switch (lock) {
        .none => .blocking,
        .shared, .exclusive => .nonblocking,
    }) catch |err| switch (err) {
        error.WouldBlock => return false,
        else => |e| return e,
    };
    return true;
}

fn fileUnlock(userdata: ?*anyopaque, file: File) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = .initBlocked(ev);
    defer sync.deinit(ev);
    ev.flock(&sync, file.handle, .none, .blocking) catch |err| switch (err) {
        error.Canceled => unreachable, // blocked
        error.WouldBlock => unreachable, // blocking
        error.SystemResources => return recoverableOsBugDetected(), // Resource deallocation.
        error.FileLocksUnsupported => return recoverableOsBugDetected(), // We already got the lock.
        error.Unexpected => return recoverableOsBugDetected(), // Resource deallocation must succeed.
    };
}

fn fileDowngradeLock(userdata: ?*anyopaque, file: File) File.DowngradeLockError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    ev.flock(&sync, file.handle, .shared, .nonblocking) catch |err| switch (err) {
        error.WouldBlock => return errnoBug(.AGAIN), // File was not locked in exclusive mode.
        error.SystemResources => return errnoBug(.NOLCK), // Lock already obtained.
        error.FileLocksUnsupported => return errnoBug(.OPNOTSUPP), // Lock already obtained.
        else => |e| return e,
    };
}

fn fileRealPath(userdata: ?*anyopaque, file: File, out_buffer: []u8) File.RealPathError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.realPath(&sync, file.handle, out_buffer);
}

fn fileHardLink(
    userdata: ?*anyopaque,
    file: File,
    new_dir: Dir,
    new_sub_path: []const u8,
    options: File.HardLinkOptions,
) File.HardLinkError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var new_path_buffer: [PATH_MAX]u8 = undefined;
    const new_sub_path_posix = try pathToPosix(new_sub_path, &new_path_buffer);

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return ev.linkat(
        &sync,
        file.handle,
        "",
        new_dir.handle,
        new_sub_path_posix,
        linux.AT.EMPTY_PATH | @as(u32, if (options.follow_symlinks) linux.AT.SYMLINK_FOLLOW else 0),
    );
}

fn fileMemoryMapCreate(
    userdata: ?*anyopaque,
    file: File,
    options: File.MemoryMap.CreateOptions,
) File.MemoryMap.CreateError!File.MemoryMap {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    const prot: linux.PROT = .{
        .READ = options.protection.read,
        .WRITE = options.protection.write,
        .EXEC = options.protection.execute,
    };
    const flags: linux.MAP = .{
        .TYPE = .SHARED_VALIDATE,
        .POPULATE = options.populate,
    };

    const page_align = std.heap.page_size_min;

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    const contents = while (true) {
        try sync.cancel_region.await(.nothing);
        const casted_offset = std.math.cast(i64, options.offset) orelse return error.Unseekable;
        const rc = linux.mmap(null, options.len, prot, flags, file.handle, casted_offset);
        switch (linux.errno(rc)) {
            .SUCCESS => break @as([*]align(page_align) u8, @ptrFromInt(rc))[0..options.len],
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .AGAIN => return error.LockedMemoryLimitExceeded,
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NFILE => return error.SystemFdQuotaExceeded,
            .NOMEM => return error.OutOfMemory,
            .PERM => return error.PermissionDenied,
            .OVERFLOW => return error.Unseekable,
            .BADF => |err| return errnoBug(err), // Always a race condition.
            .INVAL => |err| return errnoBug(err), // Invalid parameters to mmap()
            .OPNOTSUPP => |err| return errnoBug(err), // Bad flags with MAP.SHARED_VALIDATE on Linux.
            else => |err| return unexpectedErrno(err),
        }
    };
    return .{
        .file = file,
        .offset = options.offset,
        .memory = contents,
        .section = {},
    };
}

fn fileMemoryMapDestroy(userdata: ?*anyopaque, mm: *File.MemoryMap) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    const memory = mm.memory;
    if (memory.len == 0) return;
    switch (linux.errno(linux.munmap(memory.ptr, memory.len))) {
        .SUCCESS => {},
        else => |err| if (builtin.mode == .debug)
            std.log.err("failed to unmap {d} bytes at {*}: {t}", .{ memory.len, memory.ptr, err }),
    }
    mm.* = undefined;
}

fn fileMemoryMapSetLength(
    userdata: ?*anyopaque,
    mm: *File.MemoryMap,
    new_len: usize,
) File.MemoryMap.SetLengthError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    const page_size = std.heap.pageSize();
    const alignment: Alignment = .fromByteUnits(page_size);
    const page_align = std.heap.page_size_min;
    const old_memory = mm.memory;

    if (alignment.forward(new_len) == alignment.forward(old_memory.len)) {
        mm.memory.len = new_len;
        return;
    }
    const flags: linux.MREMAP = .{ .MAYMOVE = true };
    const addr_hint: ?[*]const u8 = null;
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    const new_memory = while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.mremap(old_memory.ptr, old_memory.len, new_len, flags, addr_hint);
        switch (linux.errno(rc)) {
            .SUCCESS => break @as([*]align(page_align) u8, @ptrFromInt(rc))[0..new_len],
            .INTR => {},
            .AGAIN => return error.LockedMemoryLimitExceeded,
            .NOMEM => return error.OutOfMemory,
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    };
    mm.memory = new_memory;
}

fn fileMemoryMapRead(userdata: ?*anyopaque, mm: *File.MemoryMap) File.ReadPositionalError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = mm;
}

fn fileMemoryMapWrite(userdata: ?*anyopaque, mm: *File.MemoryMap) File.WritePositionalError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = mm;
}

fn processExecutableOpen(
    userdata: ?*anyopaque,
    flags: Dir.OpenFileOptions,
) process.OpenExecutableError!File {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return dirOpenFile(ev, .{ .handle = linux.AT.FDCWD }, "/proc/self/exe", flags);
}

fn processExecutablePath(userdata: ?*anyopaque, out_buffer: []u8) process.ExecutablePathError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return dirReadLink(ev, .cwd(), "/proc/self/exe", out_buffer) catch |err| switch (err) {
        error.UnsupportedReparsePointType => unreachable, // Windows-only
        error.NetworkNotFound => unreachable, // Windows-only
        error.FileBusy => unreachable, // Windows-only
        else => |e| return e,
    };
}

fn lockStderr(userdata: ?*anyopaque, terminal_mode: ?Io.Terminal.Mode) Io.Cancelable!Io.LockedStderr {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    ev.stderr_mutex.lockUncancelable(ev_io);
    errdefer ev.stderr_mutex.unlock(ev_io);
    return ev.initLockedStderr(terminal_mode);
}

fn tryLockStderr(
    userdata: ?*anyopaque,
    terminal_mode: ?Io.Terminal.Mode,
) Io.Cancelable!?Io.LockedStderr {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    if (!ev.stderr_mutex.tryLock()) return null;
    errdefer ev.stderr_mutex.unlock(ev_io);
    return try ev.initLockedStderr(terminal_mode);
}

fn initLockedStderr(ev: *Evented, terminal_mode: ?Io.Terminal.Mode) Io.Cancelable!Io.LockedStderr {
    if (!ev.stderr_writer_initialized) {
        const ev_io = ev.io();
        const cancel_protection = Scheduler.swapCancelProtection(ev, .blocked);
        defer assert(Scheduler.swapCancelProtection(ev, cancel_protection) == .blocked);
        ev.scanEnviron() catch |err| switch (err) {
            error.Canceled => unreachable, // blocked
        };
        const NO_COLOR = ev.environ.exist.NO_COLOR;
        const CLICOLOR_FORCE = ev.environ.exist.CLICOLOR_FORCE;
        ev.stderr_mode = Io.Terminal.Mode.detect(
            ev_io,
            ev.stderr_writer.file,
            NO_COLOR,
            CLICOLOR_FORCE,
        ) catch |err| switch (err) {
            error.Canceled => unreachable, // blocked
        };
        ev.stderr_writer_initialized = true;
    }
    return .{
        .file_writer = &ev.stderr_writer,
        .terminal_mode = terminal_mode orelse ev.stderr_mode,
    };
}

fn unlockStderr(userdata: ?*anyopaque) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (ev.stderr_writer.err == null) ev.stderr_writer.interface.flush() catch {};
    if (ev.stderr_writer.err) |err| {
        switch (err) {
            error.Canceled => Thread.current().currentFiber().cancel_protection.recancel(),
            else => {},
        }
        ev.stderr_writer.err = null;
    }
    ev.stderr_writer.interface.end = 0;
    ev.stderr_writer.interface.buffer = &.{};
    ev.stderr_mutex.unlock(ev.io());
}

fn processCurrentPath(userdata: ?*anyopaque, buffer: []u8) process.CurrentPathError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.getcwd(buffer.ptr, buffer.len))) {
            .SUCCESS => return std.mem.findScalar(u8, buffer, 0).?,
            .INTR => {},
            .NOENT => return error.CurrentDirUnlinked,
            .RANGE => return error.NameTooLong,
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn processSetCurrentDir(userdata: ?*anyopaque, dir: Dir) process.SetCurrentDirError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (dir.handle == linux.AT.FDCWD) return;
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return fchdir(&sync, dir.handle);
}

fn processSetCurrentPath(userdata: ?*anyopaque, dir_path: []const u8) process.SetCurrentPathError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var path_buffer: [PATH_MAX]u8 = undefined;
    const dir_path_posix = try pathToPosix(dir_path, &path_buffer);
    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return chdir(&sync, dir_path_posix);
}

fn processReplace(userdata: ?*anyopaque, options: process.ReplaceOptions) process.ReplaceError {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    try ev.scanEnviron(); // for PATH
    const PATH = ev.environ.string.PATH orelse default_PATH;

    var arena_allocator = std.heap.ArenaAllocator.init(ev.allocator());
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

    const argv_buf = try arena.allocSentinel(?[*:0]const u8, options.argv.len, null);
    for (options.argv, 0..) |arg, i| argv_buf[i] = (try arena.dupeSentinel(u8, arg, 0)).ptr;

    const env_block = env_block: {
        const prog_fd: i32 = -1;
        if (options.environ_map) |environ_map| break :env_block try environ_map.createPosixBlock(arena, .{
            .zig_progress_fd = prog_fd,
        });
        break :env_block try ev.environ.process_environ.createPosixBlock(arena, .{
            .zig_progress_fd = prog_fd,
        });
    };

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return execv(&sync, options.expand_arg0, argv_buf.ptr[0].?, argv_buf.ptr, env_block, PATH);
}

fn processReplacePath(
    userdata: ?*anyopaque,
    dir: Dir,
    options: process.ReplaceOptions,
) process.ReplaceError {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    try ev.scanEnviron(); // for the child environment

    var arena_allocator = std.heap.ArenaAllocator.init(ev.allocator());
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

    const argv_buf = try arena.allocSentinel(?[*:0]const u8, options.argv.len, null);
    for (options.argv, 0..) |arg, i| argv_buf[i] = (try arena.dupeSentinel(u8, arg, 0)).ptr;

    const env_block = env_block: {
        const prog_fd: i32 = -1;
        if (options.environ_map) |environ_map| break :env_block try environ_map.createPosixBlock(arena, .{
            .zig_progress_fd = prog_fd,
        });
        break :env_block try ev.environ.process_environ.createPosixBlock(arena, .{
            .zig_progress_fd = prog_fd,
        });
    };

    var sync: CancelRegion.Sync = try .init(ev);
    defer sync.deinit(ev);
    return execvAt(&sync, dir, argv_buf.ptr[0].?, argv_buf.ptr, env_block);
}

/// Like `execvPath`, except that `file` is resolved relative to `dir`, and
/// `PATH` is never consulted, as documented for the `*Path` variants of
/// `spawn` and `replace`.
fn execvAt(
    sync: *CancelRegion.Sync,
    dir: Dir,
    file: [*:0]const u8,
    child_argv: [*:null]const ?[*:0]const u8,
    env_block: process.Environ.PosixBlock,
) process.ReplaceError {
    try sync.cancel_region.await(.nothing);
    switch (linux.errno(linux.execveat(dir.handle, file, child_argv, env_block.slice.ptr, .{
        .SYMLINK_NOFOLLOW = false,
        .EMPTY_PATH = false,
    }))) {
        .FAULT => |err| return errnoBug(err), // Bad pointer parameter.
        .@"2BIG" => return error.SystemResources,
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NAMETOOLONG => return error.NameTooLong,
        .NFILE => return error.SystemFdQuotaExceeded,
        .NOMEM => return error.SystemResources,
        .ACCES => return error.AccessDenied,
        .PERM => return error.PermissionDenied,
        .INVAL => return error.InvalidExe,
        .NOEXEC => return error.InvalidExe,
        .IO => return error.FileSystem,
        .LOOP => return error.FileSystem,
        .ISDIR => return error.IsDir,
        .NOENT => return error.FileNotFound,
        .NOTDIR => return error.NotDir,
        .TXTBSY => return error.FileBusy,
        .LIBBAD => return error.InvalidExe,
        else => |err| return unexpectedErrno(err),
    }
}

fn processSpawn(userdata: ?*anyopaque, options: process.SpawnOptions) process.SpawnError!process.Child {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const spawned = try ev.spawn(null, options);
    var cancel_region: CancelRegion = .initBlocked();
    defer cancel_region.deinit();
    defer ev.closeAsync(spawned.err_fd);

    // Wait for the child to report any errors in or before `execvpe`.
    var child_err: ForkBailError = undefined;
    ev.readAll(&cancel_region, spawned.err_fd, @ptrCast(&child_err)) catch |read_err| {
        switch (read_err) {
            error.Canceled => unreachable, // blocked
            error.EndOfStream => {
                // Write end closed by CLOEXEC at the time of the `execvpe` call,
                // indicating success.
            },
            else => {
                // Problem reading the error from the error reporting pipe. We
                // don't know if the child is alive or dead. Better to assume it is
                // alive so the resource does not risk being leaked.
            },
        }
        return .{
            .id = spawned.pid,
            .thread_handle = {},
            .stdin = spawned.stdin,
            .stdout = spawned.stdout,
            .stderr = spawned.stderr,
            .request_resource_usage_statistics = options.request_resource_usage_statistics,
        };
    };
    return child_err;
}

fn processSpawnPath(
    userdata: ?*anyopaque,
    dir: Dir,
    options: process.SpawnOptions,
) process.SpawnError!process.Child {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const spawned = try ev.spawn(dir, options);
    var cancel_region: CancelRegion = .initBlocked();
    defer cancel_region.deinit();
    defer ev.closeAsync(spawned.err_fd);

    // Wait for the child to report any errors in or before `execveat`.
    var child_err: ForkBailError = undefined;
    ev.readAll(&cancel_region, spawned.err_fd, @ptrCast(&child_err)) catch |read_err| {
        switch (read_err) {
            error.Canceled => unreachable, // blocked
            error.EndOfStream => {
                // Write end closed by CLOEXEC at the time of the `execveat` call,
                // indicating success.
            },
            else => {
                // Problem reading the error from the error reporting pipe. We
                // don't know if the child is alive or dead. Better to assume it is
                // alive so the resource does not risk being leaked.
            },
        }
        return .{
            .id = spawned.pid,
            .thread_handle = {},
            .stdin = spawned.stdin,
            .stdout = spawned.stdout,
            .stderr = spawned.stderr,
            .request_resource_usage_statistics = options.request_resource_usage_statistics,
        };
    };
    return child_err;
}

const prog_fileno = @max(linux.STDIN_FILENO, linux.STDOUT_FILENO, linux.STDERR_FILENO);

const Spawned = struct {
    pid: pid_t,
    err_fd: fd_t,
    stdin: ?File,
    stdout: ?File,
    stderr: ?File,
};
/// When `exec_dir` is provided, the executable path in `options.argv[0]` is
/// resolved relative to it (and `PATH` is not consulted), without affecting
/// the child's working directory.
fn spawn(ev: *Evented, exec_dir: ?Dir, options: process.SpawnOptions) process.SpawnError!Spawned {
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();

    // The child process does need to access (one end of) these pipes. However,
    // we must initially set CLOEXEC to avoid a race condition. If another thread
    // is racing to spawn a different child process, we don't want it to inherit
    // these FDs in any scenario; that would mean that, for instance, calls to
    // `poll` from the parent would not report the child's stdout as closing when
    // expected, since the other child may retain a reference to the write end of
    // the pipe. So, we create the pipes with CLOEXEC initially. After fork, we
    // need to do something in the new child to make sure we preserve the reference
    // we want. We could use `fcntl` to remove CLOEXEC from the FD, but as it
    // turns out, we `dup2` everything anyway, so there's no need!
    const pipe_flags: linux.O = .{ .CLOEXEC = true };

    const stdin_pipe = if (options.stdin == .pipe) try pipe2(pipe_flags) else undefined;
    errdefer if (options.stdin == .pipe) {
        ev.destroyPipe(stdin_pipe);
    };

    const stdout_pipe = if (options.stdout == .pipe) try pipe2(pipe_flags) else undefined;
    errdefer if (options.stdout == .pipe) {
        ev.destroyPipe(stdout_pipe);
    };

    const stderr_pipe = if (options.stderr == .pipe) try pipe2(pipe_flags) else undefined;
    errdefer if (options.stderr == .pipe) {
        ev.destroyPipe(stderr_pipe);
    };

    const any_ignore =
        options.stdin == .ignore or options.stdout == .ignore or options.stderr == .ignore;
    const dev_null_fd = if (any_ignore) try ev.null_fd.open(ev, &cancel_region, "/dev/null", .{
        .ACCMODE = .RDWR,
    }) else undefined;

    const prog_pipe: [2]fd_t = if (options.progress_node.index != .none) pipe: {
        // We use CLOEXEC for the same reason as in `pipe_flags`.
        const pipe = try pipe2(.{ .NONBLOCK = true, .CLOEXEC = true });
        _ = linux.fcntl(pipe[0], linux.F.SETPIPE_SZ, @as(u32, std.Progress.max_packet_len * 2));
        break :pipe pipe;
    } else .{ -1, -1 };
    errdefer ev.destroyPipe(prog_pipe);

    var arena_allocator = std.heap.ArenaAllocator.init(ev.allocator());
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

    // The POSIX standard does not allow malloc() between fork() and execve(),
    // and this allocator may be a libc allocator.
    // I have personally observed the child process deadlocking when it tries
    // to call malloc() due to a heap allocation between fork() and execve(),
    // in musl v1.1.24.
    // Additionally, we want to reduce the number of possible ways things
    // can fail between fork() and execve().
    // Therefore, we do all the allocation for the execve() before the fork().
    // This means we must do the null-termination of argv and env vars here.
    const argv_buf = try arena.allocSentinel(?[*:0]const u8, options.argv.len, null);
    for (options.argv, 0..) |arg, i| argv_buf[i] = (try arena.dupeSentinel(u8, arg, 0)).ptr;

    const env_block = env_block: {
        const prog_fd: i32 = if (prog_pipe[1] == -1) -1 else prog_fileno;
        if (options.environ_map) |environ_map| break :env_block try environ_map.createPosixBlock(arena, .{
            .zig_progress_fd = prog_fd,
        });
        break :env_block try ev.environ.process_environ.createPosixBlock(arena, .{
            .zig_progress_fd = prog_fd,
        });
    };

    // This pipe communicates to the parent errors in the child between `fork` and `execvpe`.
    // It is closed by the child (via CLOEXEC) without writing if `execvpe` succeeds.
    const err_pipe: [2]fd_t = try pipe2(.{ .CLOEXEC = true });
    errdefer ev.destroyPipe(err_pipe);

    try ev.scanEnviron(); // for PATH
    const PATH = ev.environ.string.PATH orelse default_PATH;

    const pid_result: pid_t = fork: {
        const rc = linux.fork();
        switch (linux.errno(rc)) {
            .SUCCESS => break :fork @intCast(rc),
            .AGAIN => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .NOSYS => return error.OperationUnsupported,
            else => |err| return unexpectedErrno(err),
        }
    };

    if (pid_result == 0) {
        defer comptime unreachable; // We are the child.
        // Note that the parent uring is no longer accessible, so we must no longer reference `ev`.
        var sync: CancelRegion.Sync = .{ .cancel_region = .initBlocked() };
        const err = setUpChild(&sync, .{
            .stdin_pipe = stdin_pipe[0],
            .stdout_pipe = stdout_pipe[1],
            .stderr_pipe = stderr_pipe[1],
            .dev_null_fd = dev_null_fd,
            .prog_pipe = prog_pipe[1],
            .argv_buf = argv_buf,
            .env_block = env_block,
            .PATH = PATH,
            .exec_dir = exec_dir,
            .spawn = options,
        });
        writeAllSync(&sync, err_pipe[1], @ptrCast(&err)) catch {};
        const exit = if (builtin.single_threaded) linux.exit else linux.exit_group;
        exit(1);
    }

    const pid: pid_t = @intCast(pid_result); // We are the parent.
    errdefer comptime unreachable; // The child is forked; we must not error from now on

    ev.closeAsync(err_pipe[1]); // make sure only the child holds the write end open

    if (options.stdin == .pipe) ev.closeAsync(stdin_pipe[0]);
    if (options.stdout == .pipe) ev.closeAsync(stdout_pipe[1]);
    if (options.stderr == .pipe) ev.closeAsync(stderr_pipe[1]);

    if (prog_pipe[1] != -1) ev.closeAsync(prog_pipe[1]);

    options.progress_node.setIpcFile(ev, .{ .handle = prog_pipe[0], .flags = .{ .nonblocking = true } });

    return .{
        .pid = pid,
        .err_fd = err_pipe[0],
        .stdin = switch (options.stdin) {
            .pipe => .{ .handle = stdin_pipe[1], .flags = .{ .nonblocking = false } },
            else => null,
        },
        .stdout = switch (options.stdout) {
            .pipe => .{ .handle = stdout_pipe[0], .flags = .{ .nonblocking = false } },
            else => null,
        },
        .stderr = switch (options.stderr) {
            .pipe => .{ .handle = stderr_pipe[0], .flags = .{ .nonblocking = false } },
            else => null,
        },
    };
}

pub const PipeError = error{
    SystemFdQuotaExceeded,
    ProcessFdQuotaExceeded,
} || Io.UnexpectedError;
pub fn pipe2(flags: linux.O) PipeError![2]fd_t {
    var fds: [2]fd_t = undefined;
    switch (linux.errno(linux.pipe2(&fds, flags))) {
        .SUCCESS => return fds,
        .INVAL => |err| return errnoBug(err), // Invalid flags
        .NFILE => return error.SystemFdQuotaExceeded,
        .MFILE => return error.ProcessFdQuotaExceeded,
        else => |err| return unexpectedErrno(err),
    }
}
fn destroyPipe(ev: *Evented, pipe: [2]fd_t) void {
    if (pipe[0] != -1) ev.closeAsync(pipe[0]);
    if (pipe[0] != pipe[1]) ev.closeAsync(pipe[1]);
}

/// Errors that can occur between fork() and execv()
const ForkBailError = process.SetCurrentDirError || ChdirError ||
    process.SpawnError || process.ReplaceError;
fn setUpChild(sync: *CancelRegion.Sync, options: struct {
    stdin_pipe: fd_t,
    stdout_pipe: fd_t,
    stderr_pipe: fd_t,
    dev_null_fd: fd_t,
    prog_pipe: fd_t,
    argv_buf: [:null]?[*:0]const u8,
    env_block: process.Environ.Block,
    PATH: []const u8,
    exec_dir: ?Dir,
    spawn: process.SpawnOptions,
}) ForkBailError {
    try setUpChildIo(
        sync,
        options.spawn.stdin,
        options.stdin_pipe,
        linux.STDIN_FILENO,
        options.dev_null_fd,
    );
    try setUpChildIo(
        sync,
        options.spawn.stdout,
        options.stdout_pipe,
        linux.STDOUT_FILENO,
        options.dev_null_fd,
    );
    try setUpChildIo(
        sync,
        options.spawn.stderr,
        options.stderr_pipe,
        linux.STDERR_FILENO,
        options.dev_null_fd,
    );

    switch (options.spawn.cwd) {
        .inherit => {},
        .dir => |cwd_dir| try fchdir(sync, cwd_dir.handle),
        .path => |cwd_path| {
            var cwd_path_buffer: [PATH_MAX]u8 = undefined;
            const cwd_path_posix = try pathToPosix(cwd_path, &cwd_path_buffer);
            try chdir(sync, cwd_path_posix);
        },
    }

    // Must happen after fchdir above, the cwd file descriptor might be
    // equal to prog_fileno and be clobbered by this dup2 call.
    if (options.prog_pipe != -1) try dup2(sync, options.prog_pipe, prog_fileno);

    if (options.spawn.gid) |gid| {
        switch (linux.errno(linux.setregid(gid, gid))) {
            .SUCCESS => {},
            .AGAIN => return error.ResourceLimitReached,
            .INVAL => return error.InvalidUserId,
            .PERM => return error.PermissionDenied,
            else => return error.Unexpected,
        }
    }

    if (options.spawn.uid) |uid| {
        switch (linux.errno(linux.setreuid(uid, uid))) {
            .SUCCESS => {},
            .AGAIN => return error.ResourceLimitReached,
            .INVAL => return error.InvalidUserId,
            .PERM => return error.PermissionDenied,
            else => return error.Unexpected,
        }
    }

    if (options.spawn.pgid) |pid| {
        switch (linux.errno(linux.setpgid(0, pid))) {
            .SUCCESS => {},
            .ACCES => return error.ProcessAlreadyExec,
            .INVAL => return error.InvalidProcessGroupId,
            .PERM => return error.PermissionDenied,
            else => return error.Unexpected,
        }
    }

    if (options.spawn.start_suspended) {
        switch (linux.errno(linux.kill(0, .STOP))) {
            .SUCCESS => {},
            .PERM => return error.PermissionDenied,
            else => return error.Unexpected,
        }
    }

    if (options.exec_dir) |exec_dir| return execvAt(
        sync,
        exec_dir,
        options.argv_buf.ptr[0].?,
        options.argv_buf.ptr,
        options.env_block,
    );
    return execv(
        sync,
        options.spawn.expand_arg0,
        options.argv_buf.ptr[0].?,
        options.argv_buf.ptr,
        options.env_block,
        options.PATH,
    );
}

fn setUpChildIo(
    sync: *CancelRegion.Sync,
    stdio: process.SpawnOptions.StdIo,
    pipe_fd: fd_t,
    std_fileno: i32,
    dev_null_fd: fd_t,
) !void {
    switch (stdio) {
        .pipe => try dup2(sync, pipe_fd, std_fileno),
        .close => _ = linux.close(std_fileno),
        .inherit => {},
        .ignore => try dup2(sync, dev_null_fd, std_fileno),
        .file => |file| try dup2(sync, file.handle, std_fileno),
    }
}

pub const DupError = error{
    ProcessFdQuotaExceeded,
    SystemResources,
} || Io.UnexpectedError || Io.Cancelable;
pub fn dup2(sync: *CancelRegion.Sync, old_fd: fd_t, new_fd: fd_t) DupError!void {
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.dup2(old_fd, new_fd))) {
            .SUCCESS => return,
            .BUSY, .INTR => {},
            .INVAL => |err| return errnoBug(err), // invalid parameters
            .BADF => |err| return errnoBug(err), // use after free
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NOMEM => return error.SystemResources,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn execv(
    sync: *CancelRegion.Sync,
    arg0_expand: process.ArgExpansion,
    file: [*:0]const u8,
    child_argv: [*:null]?[*:0]const u8,
    env_block: process.Environ.PosixBlock,
    PATH: []const u8,
) process.ReplaceError {
    const file_slice = std.mem.sliceTo(file, 0);
    if (std.mem.findScalar(u8, file_slice, '/') != null)
        return execvPath(sync, file, child_argv, env_block);

    // Use of PATH_MAX here is valid as the path_buf will be passed
    // directly to the operating system in posixExecvPath.
    var path_buf: [PATH_MAX]u8 = undefined;
    var it = std.mem.tokenizeScalar(u8, PATH, ':');
    var seen_eacces = false;
    var err: process.ReplaceError = error.FileNotFound;

    // In case of expanding arg0 we must put it back if we return with an error.
    const prev_arg0 = child_argv[0];
    defer switch (arg0_expand) {
        .expand => child_argv[0] = prev_arg0,
        .no_expand => {},
    };

    while (it.next()) |search_path| {
        const path_len = search_path.len + file_slice.len + 1;
        if (path_buf.len < path_len + 1) return error.NameTooLong;
        @memcpy(path_buf[0..search_path.len], search_path);
        path_buf[search_path.len] = '/';
        @memcpy(path_buf[search_path.len + 1 ..][0..file_slice.len], file_slice);
        path_buf[path_len] = 0;
        const full_path = path_buf[0..path_len :0].ptr;
        switch (arg0_expand) {
            .expand => child_argv[0] = full_path,
            .no_expand => {},
        }
        err = execvPath(sync, full_path, child_argv, env_block);
        switch (err) {
            error.AccessDenied => seen_eacces = true,
            error.FileNotFound, error.NotDir => {},
            else => |e| return e,
        }
    }
    if (seen_eacces) return error.AccessDenied;
    return err;
}
/// This function ignores PATH environment variable.
pub fn execvPath(
    sync: *CancelRegion.Sync,
    path: [*:0]const u8,
    child_argv: [*:null]const ?[*:0]const u8,
    env_block: process.Environ.PosixBlock,
) process.ReplaceError {
    try sync.cancel_region.await(.nothing);
    switch (linux.errno(linux.execve(path, child_argv, env_block.slice.ptr))) {
        .FAULT => |err| return errnoBug(err), // Bad pointer parameter.
        .@"2BIG" => return error.SystemResources,
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NAMETOOLONG => return error.NameTooLong,
        .NFILE => return error.SystemFdQuotaExceeded,
        .NOMEM => return error.SystemResources,
        .ACCES => return error.AccessDenied,
        .PERM => return error.PermissionDenied,
        .INVAL => return error.InvalidExe,
        .NOEXEC => return error.InvalidExe,
        .IO => return error.FileSystem,
        .LOOP => return error.FileSystem,
        .ISDIR => return error.IsDir,
        .NOENT => return error.FileNotFound,
        .NOTDIR => return error.NotDir,
        .TXTBSY => return error.FileBusy,
        .LIBBAD => return error.InvalidExe,
        else => |err| return unexpectedErrno(err),
    }
}

fn childWait(userdata: ?*anyopaque, child: *process.Child) process.Child.WaitError!process.Child.Term {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    defer ev.childCleanup(child);

    const pid = child.id.?;
    var info: linux.siginfo_t = undefined;
    while (true) {
        const thread = try maybe_sync.cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .WAITID,
            .flags = 0,
            .ioprio = 0,
            .fd = pid,
            .off = @intFromPtr(&info),
            .addr = 0,
            .len = @backingInt(linux.P.PID),
            .rw_flags = 0,
            .user_data = @intFromPtr(maybe_sync.cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = linux.W.EXITED |
                @as(i32, if (child.request_resource_usage_statistics) linux.W.NOWAIT else 0),
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (maybe_sync.cancel_region.errno()) {
            .SUCCESS => {
                if (child.request_resource_usage_statistics) {
                    const sync = try maybe_sync.enterSync(ev);
                    while (true) {
                        try sync.cancel_region.await(.nothing);
                        var rusage: linux.rusage = undefined;
                        switch (linux.errno(linux.waitid(
                            .PID,
                            pid,
                            &info,
                            linux.W.EXITED | linux.W.NOHANG,
                            &rusage,
                        ))) {
                            .SUCCESS => {
                                child.resource_usage_statistics.rusage = rusage;
                                break;
                            },
                            .INTR, .CANCELED => {},
                            .CHILD => |err| return errnoBug(err), // Double-free.
                            else => |err| return unexpectedErrno(err),
                        }
                    }
                }
                const status: u32 = @bitCast(info.fields.common.second.sigchld.status);
                const code: linux.CLD = @fromBackingInt(@intCast(info.code));
                return switch (code) {
                    .EXITED => .{ .exited = @truncate(status) },
                    .KILLED, .DUMPED => .{ .signal = @fromBackingInt(@intCast(status)) },
                    .TRAPPED, .STOPPED => .{ .stopped = @fromBackingInt(@intCast(status)) },
                    _, .CONTINUED => .{ .unknown = status },
                };
            },
            .INTR, .CANCELED => {},
            .CHILD => |err| return errnoBug(err), // Double-free.
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn childKill(userdata: ?*anyopaque, child: *process.Child) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var maybe_sync: CancelRegion.Sync.Maybe = .{ .sync = .initBlocked(ev) };
    defer maybe_sync.deinit(ev);
    defer ev.childCleanup(child);

    const pid = child.id.?;
    while (true) switch (linux.errno(linux.kill(pid, .TERM))) {
        .SUCCESS => break,
        .INTR => {},
        .PERM => return,
        .INVAL => |err| return errnoBug(err) catch {},
        .SRCH => |err| return errnoBug(err) catch {},
        else => |err| return unexpectedErrno(err) catch {},
    };
    maybe_sync.leaveSync(ev);

    var info: linux.siginfo_t = undefined;
    while (true) {
        const thread = maybe_sync.cancel_region.awaitIoUring() catch |err| switch (err) {
            error.Canceled => unreachable, // blocked
        };
        thread.enqueue().* = .{
            .opcode = .WAITID,
            .flags = 0,
            .ioprio = 0,
            .fd = pid,
            .off = @intFromPtr(&info),
            .addr = 0,
            .len = @backingInt(linux.P.PID),
            .rw_flags = 0,
            .user_data = @intFromPtr(maybe_sync.cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = linux.W.EXITED,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (maybe_sync.cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            .CHILD => |err| return errnoBug(err) catch {}, // Double-free.
            else => |err| return unexpectedErrno(err) catch {},
        }
    }
}

fn childCleanup(ev: *Evented, child: *process.Child) void {
    if (child.stdin) |*stdin| {
        ev.closeAsync(stdin.handle);
        child.stdin = null;
    }
    if (child.stdout) |*stdout| {
        ev.closeAsync(stdout.handle);
        child.stdout = null;
    }
    if (child.stderr) |*stderr| {
        ev.closeAsync(stderr.handle);
        child.stderr = null;
    }
    child.id = null;
}

fn progressParentFile(userdata: ?*anyopaque) std.Progress.ParentFileError!File {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const cancel_protection = Scheduler.swapCancelProtection(ev, .blocked);
    defer assert(Scheduler.swapCancelProtection(ev, cancel_protection) == .blocked);
    ev.scanEnviron() catch |err| switch (err) {
        error.Canceled => unreachable, // blocked
    };
    return ev.environ.zig_progress_file;
}

fn scanEnviron(ev: *Evented) Io.Cancelable!void {
    const ev_io = ev.io();
    try ev.environ_mutex.lock(ev_io);
    defer ev.environ_mutex.unlock(ev_io);
    if (ev.environ_initialized) return;
    ev.environ.scan(ev.allocator());
    ev.environ_initialized = true;
}

fn clockResolution(userdata: ?*anyopaque, clock: Io.Clock) Io.Clock.ResolutionError!Io.Duration {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    const clock_id = clockToPosix(clock);
    var timespec: linux.timespec = undefined;
    return switch (linux.errno(linux.clock_getres(clock_id, &timespec))) {
        .SUCCESS => .fromNanoseconds(nanosecondsFromPosix(&timespec)),
        .INVAL => return error.ClockUnavailable,
        else => |err| return unexpectedErrno(err),
    };
}

fn now(userdata: ?*anyopaque, clock: Io.Clock) Io.Timestamp {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    var tp: linux.timespec = undefined;
    switch (linux.errno(linux.clock_gettime(clockToPosix(clock), &tp))) {
        .SUCCESS => return timestampFromPosix(&tp),
        else => return .zero,
    }
}

fn sleep(userdata: ?*anyopaque, timeout: Io.Timeout) Io.Cancelable!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    const timespec: linux.kernel_timespec, const clock: Io.Clock, const timeout_flags: u32 = timespec: switch (timeout) {
        .none => .{
            .{
                .sec = std.math.maxInt(i64),
                .nsec = std.time.ns_per_s - 1,
            },
            .awake,
            linux.IORING_TIMEOUT_ABS,
        },
        .duration => |duration| {
            const ns = duration.raw.toNanoseconds();
            break :timespec .{
                .{
                    .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                    .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                },
                duration.clock,
                0,
            };
        },
        .deadline => |deadline| {
            const ns = deadline.raw.toNanoseconds();
            break :timespec .{
                .{
                    .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                    .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                },
                deadline.clock,
                linux.IORING_TIMEOUT_ABS,
            };
        },
    };
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    const thread = try cancel_region.awaitIoUring();
    thread.enqueue().* = .{
        .opcode = .TIMEOUT,
        .flags = 0,
        .ioprio = 0,
        .fd = 0,
        .off = 0,
        .addr = @intFromPtr(&timespec),
        .len = 1,
        .rw_flags = timeout_flags | @as(u32, switch (clock) {
            .real => linux.IORING_TIMEOUT_REALTIME,
            else => 0,
            .boot => linux.IORING_TIMEOUT_BOOTTIME,
        }),
        .user_data = @intFromPtr(cancel_region.fiber),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
    ev.sched.park();
    // Handles SUCCESS as well as clock not available and unexpected
    // errors. The user had a chance to check clock resolution before
    // getting here, which would have reported 0, making this a legal
    // amount of time to sleep.
}

fn random(userdata: ?*anyopaque, buffer: []u8) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var thread: *Thread = .current();
    if (!thread.csprng.isInitialized()) {
        @branchHint(.unlikely);
        var seed: [Csprng.seed_len]u8 = undefined;
        {
            const ev_io = ev.io();
            ev.csprng_mutex.lockUncancelable(ev_io);
            defer ev.csprng_mutex.unlock(ev_io);
            if (!ev.csprng.isInitialized()) {
                @branchHint(.unlikely);
                var cancel_region: CancelRegion = .initBlocked();
                defer cancel_region.deinit();
                ev.urandomReadAll(&cancel_region, &seed) catch |err| switch (err) {
                    error.Canceled => unreachable, // blocked
                    else => fallbackSeed(ev, &seed),
                };
                ev.csprng.rng = .init(seed);
                thread = .current();
            }
            ev.csprng.rng.fill(&seed);
        }
        if (!thread.csprng.isInitialized()) {
            @branchHint(.likely);
            thread.csprng.rng = .init(seed);
        } else thread.csprng.rng.addEntropy(&seed);
    }
    thread.csprng.rng.fill(buffer);
}

fn randomSecure(userdata: ?*anyopaque, buffer: []u8) Io.RandomSecureError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (buffer.len == 0) return;
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    ev.urandomReadAll(&cancel_region, buffer) catch |err| switch (err) {
        error.Canceled => |e| return e,
        else => return error.EntropyUnavailable,
    };
}

fn netListenIp(
    userdata: ?*anyopaque,
    address: *const net.IpAddress,
    options: net.IpAddress.ListenOptions,
) net.IpAddress.ListenError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const family = posixAddressFamily(address);
    const socket_fd = try ev.socket(&maybe_sync.cancel_region, family, .{
        .mode = options.mode,
        .protocol = options.protocol,
    });
    errdefer ev.closeAsync(socket_fd);

    if (options.reuse_address) {
        try ev.setsockopt(&maybe_sync.cancel_region, socket_fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, 1);
        if (@hasDecl(linux.SO, "REUSEPORT"))
            try ev.setsockopt(&maybe_sync.cancel_region, socket_fd, linux.SOL.SOCKET, linux.SO.REUSEPORT, 1);
    }

    var storage: PosixAddress = undefined;
    var addr_len = addressToPosix(address, &storage);
    try ev.bind(&maybe_sync.cancel_region, socket_fd, &storage.any, addr_len);
    try listen(try maybe_sync.enterSync(ev), socket_fd, options.kernel_backlog);
    maybe_sync.leaveSync(ev);
    try ev.getsockname(try maybe_sync.enterSync(ev), socket_fd, &storage.any, &addr_len);
    return .{ .handle = socket_fd, .address = addressFromPosix(&storage) };
}

/// This syscall does not block, but going through `CancelRegion.Sync` is how
/// the caller retains the ability to interrupt the task while it is inside a
/// syscall.
fn listen(sync: *CancelRegion.Sync, socket_fd: fd_t, backlog: u31) !void {
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.listen(socket_fd, backlog))) {
            .SUCCESS => return,
            .INTR => {},
            .ADDRINUSE => return error.AddressInUse,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .NOTSOCK => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn netAccept(
    userdata: ?*anyopaque,
    listen_handle: net.Socket.Handle,
    options: net.Server.AcceptOptions,
) net.Server.AcceptError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = options; // AcceptOptions is void on POSIX
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    var storage: PosixAddress = undefined;
    var addr_len: linux.socklen_t = @sizeOf(PosixAddress);
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .ACCEPT,
            .flags = 0,
            .ioprio = 0,
            .fd = listen_handle,
            .off = @intFromPtr(&addr_len),
            .addr = @intFromPtr(&storage.any),
            .len = 0,
            .rw_flags = linux.SOCK.CLOEXEC,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => return .{
                .handle = @intCast(completion.result),
                .address = addressFromPosix(&storage),
            },
            .INTR, .CANCELED => {},
            .AGAIN => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .CONNABORTED => return error.ConnectionAborted,
            .FAULT => |err| return errnoBug(err),
            .INVAL => return error.SocketNotListening,
            .NOTSOCK => |err| return errnoBug(err),
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NFILE => return error.SystemFdQuotaExceeded,
            .NOBUFS => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .OPNOTSUPP => |err| return errnoBug(err),
            .PROTO => return error.ProtocolFailure,
            .PERM => return error.BlockedByFirewall,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn netBindIp(
    userdata: ?*anyopaque,
    address: *const net.IpAddress,
    options: net.IpAddress.BindOptions,
) net.IpAddress.BindError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const family = posixAddressFamily(address);
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const socket_fd = try ev.socket(&maybe_sync.cancel_region, family, options);
    errdefer ev.closeAsync(socket_fd);
    var storage: PosixAddress = undefined;
    var addr_len = addressToPosix(address, &storage);
    try ev.bind(&maybe_sync.cancel_region, socket_fd, &storage.any, addr_len);
    if (options.allow_broadcast) try ev.setsockopt(&maybe_sync.cancel_region, socket_fd, linux.SOL.SOCKET, linux.SO.BROADCAST, 1);
    try ev.getsockname(try maybe_sync.enterSync(ev), socket_fd, &storage.any, &addr_len);
    return .{ .handle = socket_fd, .address = addressFromPosix(&storage) };
}

fn netConnectIp(
    userdata: ?*anyopaque,
    address: *const net.IpAddress,
    options: net.IpAddress.ConnectOptions,
) net.IpAddress.ConnectError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const family = posixAddressFamily(address);
    const socket_fd = try ev.socket(&maybe_sync.cancel_region, family, .{
        .mode = options.mode,
        .protocol = options.protocol,
    });
    errdefer ev.closeAsync(socket_fd);
    var storage: PosixAddress = undefined;
    var addr_len = addressToPosix(address, &storage);
    try connectIp(&maybe_sync.cancel_region, ev, socket_fd, &storage.any, addr_len, options.timeout);
    try ev.getsockname(try maybe_sync.enterSync(ev), socket_fd, &storage.any, &addr_len);
    return .{ .handle = socket_fd, .address = addressFromPosix(&storage) };
}

fn connectIp(
    cancel_region: *CancelRegion,
    ev: *Evented,
    socket_fd: fd_t,
    addr: *const linux.sockaddr,
    addr_len: linux.socklen_t,
    timeout: Io.Timeout,
) net.IpAddress.ConnectError!void {
    const timespec: ?linux.kernel_timespec, const timeout_flags: u32 = switch (timeout) {
        .none => .{ null, 0 },
        .duration => |duration| .{ .{
            .sec = @intCast(@divFloor(duration.raw.toNanoseconds(), std.time.ns_per_s)),
            .nsec = @intCast(@mod(duration.raw.toNanoseconds(), std.time.ns_per_s)),
        }, 0 },
        .deadline => |deadline| .{ .{
            .sec = @intCast(@divFloor(deadline.raw.toNanoseconds(), std.time.ns_per_s)),
            .nsec = @intCast(@mod(deadline.raw.toNanoseconds(), std.time.ns_per_s)),
        }, linux.IORING_TIMEOUT_ABS | @as(u32, switch (deadline.clock) {
            .real => linux.IORING_TIMEOUT_REALTIME,
            else => 0,
            .boot => linux.IORING_TIMEOUT_BOOTTIME,
        }) },
    };
    var timeout_storage = timespec;
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .CONNECT,
            .flags = if (timespec != null) linux.IOSQE_IO_LINK else 0,
            .ioprio = 0,
            .fd = socket_fd,
            .off = addr_len,
            .addr = @intFromPtr(addr),
            .len = 0,
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        if (timeout_storage) |*timespec_ptr| thread.enqueue().* = .{
            .opcode = .LINK_TIMEOUT,
            .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
            .ioprio = 0,
            .fd = 0,
            .off = 0,
            .addr = @intFromPtr(timespec_ptr),
            .len = 1,
            .rw_flags = timeout_flags,
            .user_data = @backingInt(Completion.Userdata.wakeup),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            // A linked timeout that expires cancels the `CONNECT` with `ETIME`.
            .TIME => return error.Timeout,
            .ADDRNOTAVAIL => return error.AddressUnavailable,
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .AGAIN, .INPROGRESS => return error.WouldBlock,
            // The kernel reports this when a linked timeout expiring races
            // with an in-flight connect.
            .ALREADY => return if (timespec != null) error.Timeout else error.ConnectionPending,
            .CONNREFUSED => return error.ConnectionRefused,
            .CONNRESET => return error.ConnectionResetByPeer,
            .HOSTUNREACH => return error.HostUnreachable,
            .NETUNREACH => return error.NetworkUnreachable,
            .TIMEDOUT => return error.Timeout,
            .ACCES => return error.AccessDenied,
            .NETDOWN => return error.NetworkDown,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .CONNABORTED => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .ISCONN => |err| return errnoBug(err),
            .NOENT => |err| return errnoBug(err),
            .NOTSOCK => |err| return errnoBug(err),
            .PERM => |err| return errnoBug(err),
            .PROTOTYPE => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn netListenUnix(
    userdata: ?*anyopaque,
    address: *const net.UnixAddress,
    options: net.UnixAddress.ListenOptions,
) net.UnixAddress.ListenError!net.Socket.Handle {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const socket_fd = ev.socket(&maybe_sync.cancel_region, linux.AF.UNIX, .{ .mode = .stream }) catch |err| switch (err) {
        error.ProtocolUnsupportedBySystem => return error.AddressFamilyUnsupported,
        error.ProtocolUnsupportedByAddressFamily => return error.AddressFamilyUnsupported,
        error.SocketModeUnsupported => return error.AddressFamilyUnsupported,
        error.OptionUnsupported => return error.Unexpected,
        else => |e| return e,
    };
    errdefer ev.closeAsync(socket_fd);

    var storage: UnixAddress = undefined;
    const addr_len = addressUnixToPosix(address, &storage);
    try bindUnix(&maybe_sync.cancel_region, ev, socket_fd, &storage.any, addr_len);
    try listen(try maybe_sync.enterSync(ev), socket_fd, options.kernel_backlog);
    return socket_fd;
}

fn bindUnix(
    cancel_region: *CancelRegion,
    ev: *Evented,
    socket_fd: fd_t,
    addr: *const linux.sockaddr,
    addr_len: linux.socklen_t,
) net.UnixAddress.ListenError!void {
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .BIND,
            .flags = 0,
            .ioprio = 0,
            .fd = socket_fd,
            .off = addr_len,
            .addr = @intFromPtr(addr),
            .len = 0,
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            .ACCES => return error.AccessDenied,
            .ADDRINUSE => return error.AddressInUse,
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .ADDRNOTAVAIL => return error.AddressUnavailable,
            .NOMEM => return error.SystemResources,

            .LOOP => return error.SymLinkLoop,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .ROFS => return error.ReadOnlyFileSystem,
            .PERM => return error.PermissionDenied,

            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .INVAL => |err| return errnoBug(err), // invalid parameters
            .NOTSOCK => |err| return errnoBug(err), // invalid `sockfd`
            .FAULT => |err| return errnoBug(err), // invalid `addr` pointer
            .NAMETOOLONG => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn netConnectUnix(
    userdata: ?*anyopaque,
    address: *const net.UnixAddress,
) net.UnixAddress.ConnectError!net.Socket.Handle {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const socket_fd = ev.socket(&maybe_sync.cancel_region, linux.AF.UNIX, .{ .mode = .stream }) catch |err| switch (err) {
        error.ProtocolUnsupportedByAddressFamily => return error.AddressFamilyUnsupported,
        error.OptionUnsupported => return error.Unexpected,
        else => |e| return e,
    };
    errdefer ev.closeAsync(socket_fd);
    var storage: UnixAddress = undefined;
    const addr_len = addressUnixToPosix(address, &storage);
    try connectUnix(&maybe_sync.cancel_region, ev, socket_fd, &storage.any, addr_len);
    return socket_fd;
}

fn connectUnix(
    cancel_region: *CancelRegion,
    ev: *Evented,
    socket_fd: fd_t,
    addr: *const linux.sockaddr,
    addr_len: linux.socklen_t,
) net.UnixAddress.ConnectError!void {
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .CONNECT,
            .flags = 0,
            .ioprio = 0,
            .fd = socket_fd,
            .off = addr_len,
            .addr = @intFromPtr(addr),
            .len = 0,
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .AGAIN, .INPROGRESS => return error.WouldBlock,
            .ACCES => return error.AccessDenied,

            .LOOP => return error.SymLinkLoop,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .ROFS => return error.ReadOnlyFileSystem,
            .PERM => return error.PermissionDenied,

            .CONNREFUSED => return error.ConnectionRefused,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .ISCONN => |err| return errnoBug(err),
            .NOTSOCK => |err| return errnoBug(err),
            .PROTOTYPE => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn netSocketCreatePair(
    userdata: ?*anyopaque,
    options: net.Socket.CreatePairOptions,
) net.Socket.CreatePairError![2]net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const family: linux.sa_family_t = switch (options.family) {
        .ip4 => linux.AF.INET,
        .ip6 => linux.AF.INET6,
    };
    const mode, const protocol = try posixSocketModeProtocol(family, options.mode, options.protocol);
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    var sockets: [2]fd_t = undefined;
    while (true) {
        try maybe_sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.socketpair(family, mode | linux.SOCK.CLOEXEC, protocol, &sockets))) {
            .SUCCESS => break,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .INVAL => return error.ProtocolUnsupportedBySystem,
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NFILE => return error.SystemFdQuotaExceeded,
            .NOBUFS => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .PROTONOSUPPORT => return error.ProtocolUnsupportedByAddressFamily,
            .PROTOTYPE => return error.SocketModeUnsupported,
            // e.g. socket pairs are only supported for `AF.UNIX` by Linux.
            .OPNOTSUPP => return error.OperationUnsupported,
            else => |err| return unexpectedErrno(err),
        }
    }
    errdefer {
        ev.closeAsync(sockets[0]);
        ev.closeAsync(sockets[1]);
    }
    var storages: [2]PosixAddress = undefined;
    var addr_lens: [2]linux.socklen_t = .{ @sizeOf(PosixAddress), @sizeOf(PosixAddress) };
    const sync = try maybe_sync.enterSync(ev);
    try ev.getsockname(sync, sockets[0], &storages[0].any, &addr_lens[0]);
    try ev.getsockname(sync, sockets[1], &storages[1].any, &addr_lens[1]);
    return .{
        .{ .handle = sockets[0], .address = addressFromPosix(&storages[0]) },
        .{ .handle = sockets[1], .address = addressFromPosix(&storages[1]) },
    };
}

/// Maps an `errno` from `sendmsg`; `INTR` and `CANCELED` are handled by the
/// callers.
fn sendmsgErrno(err: linux.E) Io.Operation.NetSend.Error {
    return switch (err) {
        .ACCES => error.AccessDenied,
        .ALREADY => error.FastOpenAlreadyInProgress,
        .CONNRESET => error.ConnectionResetByPeer,
        .MSGSIZE => error.MessageOversize,
        .NOBUFS => error.SystemResources,
        .NOMEM => error.SystemResources,
        .PIPE => error.SocketUnconnected,
        .AFNOSUPPORT => error.AddressFamilyUnsupported,
        .HOSTUNREACH => error.HostUnreachable,
        .NETUNREACH => error.NetworkUnreachable,
        .NOTCONN => error.SocketUnconnected,
        .TIMEDOUT => error.ConnectionTimedOut,
        .NETDOWN => error.NetworkDown,

        .BADF => |e| errnoBug(e), // File descriptor used after closed.
        .DESTADDRREQ => |e| errnoBug(e), // not connection-mode and no peer address is set
        .FAULT => |e| errnoBug(e), // invalid user space address
        .INVAL => |e| errnoBug(e), // invalid argument passed
        .ISCONN => |e| errnoBug(e), // connected already but a recipient was specified
        .NOTSOCK => |e| errnoBug(e), // does not refer to a socket
        .OPNOTSUPP => |e| errnoBug(e), // flags inappropriate for the socket type
        else => |e| unexpectedErrno(e),
    };
}

/// Maps an `errno` from `readv`; `INTR` and `CANCELED` are handled by the
/// callers. `.AGAIN` is an operating system bug because the file descriptors
/// managed by this implementation are blocking.
fn readvErrno(err: linux.E) Io.Operation.NetRead.Error {
    return switch (err) {
        .NOBUFS => error.SystemResources,
        .NOMEM => error.SystemResources,
        .NOTCONN => error.SocketUnconnected,
        .CONNRESET => error.ConnectionResetByPeer,
        .TIMEDOUT => error.ConnectionTimedOut,
        .NETDOWN => error.NetworkDown,

        .AGAIN => |e| errnoBug(e), // File descriptor is blocking.
        .BADF => |e| errnoBug(e), // File descriptor used after closed.
        .FAULT => |e| errnoBug(e),
        .INVAL => |e| errnoBug(e),
        else => |e| unexpectedErrno(e),
    };
}

/// Maps an `errno` from `sendmsg` on a stream; `INTR` and `CANCELED` are
/// handled by the callers.
fn netWriteErrno(err: linux.E) Io.Operation.NetWrite.Error {
    return switch (err) {
        .ALREADY => error.FastOpenAlreadyInProgress,
        .CONNRESET => error.ConnectionResetByPeer,
        .NOBUFS => error.SystemResources,
        .NOMEM => error.SystemResources,
        .PIPE => error.SocketUnconnected,
        .AFNOSUPPORT => error.AddressFamilyUnsupported,
        .HOSTUNREACH => error.HostUnreachable,
        .NETUNREACH => error.NetworkUnreachable,
        .NOTCONN => error.SocketUnconnected,
        .TIMEDOUT => error.ConnectionTimedOut,
        .NETDOWN => error.NetworkDown,

        .ACCES => |e| errnoBug(e),
        .AGAIN => |e| errnoBug(e), // File descriptor is blocking.
        .BADF => |e| errnoBug(e), // File descriptor used after closed.
        .DESTADDRREQ => |e| errnoBug(e), // connection-mode socket was never connected
        .FAULT => |e| errnoBug(e), // invalid user space address
        .INVAL => |e| errnoBug(e), // invalid argument passed
        .ISCONN => |e| errnoBug(e), // connected already but a recipient was specified
        .MSGSIZE => |e| errnoBug(e),
        .NOTSOCK => |e| errnoBug(e), // does not refer to a socket
        .OPNOTSUPP => |e| errnoBug(e), // flags inappropriate for the socket type
        else => |e| unexpectedErrno(e),
    };
}

/// Maps an `errno` from `recvmsg`; `INTR` and `CANCELED` are handled by the
/// callers.
fn netReceiveErrno(err: linux.E) net.Socket.ReceiveError {
    return switch (err) {
        .NFILE => error.SystemFdQuotaExceeded,
        .MFILE => error.ProcessFdQuotaExceeded,
        .NOBUFS => error.SystemResources,
        .NOMEM => error.SystemResources,
        .NOTCONN => error.SocketUnconnected,
        .PIPE => error.SocketUnconnected,
        .MSGSIZE => error.MessageOversize,
        .CONNRESET => error.ConnectionResetByPeer,
        .TIMEDOUT => error.ConnectionTimedOut,
        .NETDOWN => error.NetworkDown,
        // An ICMP port unreachable received for a previously sent datagram.
        .CONNREFUSED => error.PortUnreachable,

        .AGAIN => |e| errnoBug(e), // File descriptor is blocking.
        .BADF => |e| errnoBug(e), // File descriptor used after closed.
        .FAULT => |e| errnoBug(e),
        .INVAL => |e| errnoBug(e),
        .NOTSOCK => |e| errnoBug(e),
        .OPNOTSUPP => |e| errnoBug(e),
        else => |e| unexpectedErrno(e),
    };
}

fn netSend(
    ev: *Evented,
    cancel_region: *CancelRegion,
    handle: net.Socket.Handle,
    messages: []net.OutgoingMessage,
    flags: net.SendFlags,
) Io.Cancelable!struct { ?net.Socket.SendError, usize } {
    const posix_flags: u32 =
        @as(u32, if (@hasDecl(linux.MSG, "CONFIRM") and flags.confirm) linux.MSG.CONFIRM else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "DONTROUTE") and flags.dont_route) linux.MSG.DONTROUTE else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "EOR") and flags.eor) linux.MSG.EOR else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "OOB") and flags.oob) linux.MSG.OOB else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "FASTOPEN") and flags.fastopen) linux.MSG.FASTOPEN else 0) |
        linux.MSG.NOSIGNAL;

    var i: usize = 0;
    while (messages.len - i != 0) : (i += 1) {
        netSendOne(ev, cancel_region, handle, &messages[i], posix_flags) catch |err| switch (err) {
            error.Canceled => |e| if (i == 0) {
                return e;
            } else {
                // The messages sent so far are reported to the caller; the
                // cancelation request stays pending, so it is observed by the
                // next operation that awaits.
                return .{ null, i };
            },
            else => |e| return .{ e, i },
        };
    }
    return .{ null, i };
}

fn netSendOne(
    ev: *Evented,
    cancel_region: *CancelRegion,
    handle: net.Socket.Handle,
    message: *net.OutgoingMessage,
    posix_flags: u32,
) (Io.Cancelable || Io.Operation.NetSend.Error)!void {
    var addr: PosixAddress = undefined;
    var one_iovec: iovec_const = .{ .base = @constCast(message.data_ptr), .len = message.data_len };
    const msg: linux.msghdr_const = .{
        .name = &addr.any,
        .namelen = addressToPosix(message.address, &addr),
        .iov = (&one_iovec)[0..1],
        .iovlen = 1,
        // OS returns EINVAL if this pointer is invalid even if controllen is zero.
        .control = if (message.control.len == 0) null else @constCast(message.control.ptr),
        .controllen = @intCast(message.control.len),
        .flags = 0,
    };
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .SENDMSG,
            .flags = 0,
            .ioprio = 0,
            .fd = handle,
            .off = 0,
            .addr = @intFromPtr(&msg),
            .len = 1,
            .rw_flags = posix_flags,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => {
                message.data_len = @intCast(completion.result);
                return;
            },
            .INTR, .CANCELED => {},
            else => |err| return sendmsgErrno(err),
        }
    }
}

/// Direct syscall variant of `netSend`, for use once the kernel has reported
/// the socket to be writable, in contexts where awaiting the ring is not
/// possible.
fn netSendSync(
    handle: net.Socket.Handle,
    messages: []net.OutgoingMessage,
    flags: net.SendFlags,
) struct { ?Io.Operation.NetSend.Error, usize } {
    const posix_flags: u32 =
        @as(u32, if (@hasDecl(linux.MSG, "CONFIRM") and flags.confirm) linux.MSG.CONFIRM else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "DONTROUTE") and flags.dont_route) linux.MSG.DONTROUTE else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "EOR") and flags.eor) linux.MSG.EOR else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "OOB") and flags.oob) linux.MSG.OOB else 0) |
        @as(u32, if (@hasDecl(linux.MSG, "FASTOPEN") and flags.fastopen) linux.MSG.FASTOPEN else 0) |
        linux.MSG.NOSIGNAL;

    var i: usize = 0;
    while (messages.len - i != 0) : (i += 1) {
        netSendOneSync(handle, &messages[i], posix_flags) catch |err| return .{ err, i };
    }
    return .{ null, i };
}

fn netSendOneSync(
    handle: net.Socket.Handle,
    message: *net.OutgoingMessage,
    posix_flags: u32,
) Io.Operation.NetSend.Error!void {
    var addr: PosixAddress = undefined;
    var one_iovec: iovec_const = .{ .base = @constCast(message.data_ptr), .len = message.data_len };
    const msg: linux.msghdr_const = .{
        .name = &addr.any,
        .namelen = addressToPosix(message.address, &addr),
        .iov = (&one_iovec)[0..1],
        .iovlen = 1,
        // OS returns EINVAL if this pointer is invalid even if controllen is zero.
        .control = if (message.control.len == 0) null else @constCast(message.control.ptr),
        .controllen = @intCast(message.control.len),
        .flags = 0,
    };
    const rc: usize = while (true) {
        const rc = linux.sendmsg(handle, &msg, posix_flags);
        switch (linux.errno(rc)) {
            .INTR => {},
            else => break rc,
        }
    };
    switch (linux.errno(rc)) {
        .SUCCESS => message.data_len = @intCast(rc),
        else => |err| return sendmsgErrno(err),
    }
}

fn netRead(
    ev: *Evented,
    cancel_region: *CancelRegion,
    handle: net.Socket.Handle,
    data: [][]u8,
) net.Stream.Reader.Error!usize {
    var iovecs_buffer: [max_iovecs_len]iovec = undefined;
    var i: usize = 0;
    for (data) |buf| {
        if (iovecs_buffer.len - i == 0) break;
        if (buf.len != 0) {
            iovecs_buffer[i] = .{ .base = buf.ptr, .len = buf.len };
            i += 1;
        }
    }
    const dest = iovecs_buffer[0..i];
    assert(dest[0].len > 0);
    return ioUringReadv(ev, cancel_region, handle, dest);
}

fn netReadSync(handle: net.Socket.Handle, data: [][]u8) Io.Operation.NetRead.Result {
    var iovecs_buffer: [max_iovecs_len]iovec = undefined;
    var i: usize = 0;
    for (data) |buf| {
        if (iovecs_buffer.len - i == 0) break;
        if (buf.len != 0) {
            iovecs_buffer[i] = .{ .base = buf.ptr, .len = buf.len };
            i += 1;
        }
    }
    const dest = iovecs_buffer[0..i];
    assert(dest[0].len > 0);
    while (true) {
        const rc = linux.readv(handle, dest.ptr, dest.len);
        switch (linux.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => continue,
            else => |err| return readvErrno(err),
        }
    }
}

fn ioUringReadv(
    ev: *Evented,
    cancel_region: *CancelRegion,
    handle: net.Socket.Handle,
    dest: []const iovec,
) net.Stream.Reader.Error!usize {
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .READV,
            .flags = 0,
            .ioprio = 0,
            .fd = handle,
            .off = std.math.maxInt(u64),
            .addr = @intFromPtr(dest.ptr),
            .len = @intCast(dest.len),
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => return @intCast(@as(u32, @bitCast(completion.result))),
            .INTR, .CANCELED => {},
            else => |err| return readvErrno(err),
        }
    }
}

/// Builds the iovec array describing `header`, the literal `data` bytes, and
/// the `data` pattern repeated `splat` times. The returned slice refers to
/// `iovecs` and possibly `splat_buffer`.
fn netWriteIovecs(
    iovecs: *[max_iovecs_len]iovec_const,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
    splat_buffer: *[splat_buffer_size]u8,
) []const iovec_const {
    var iovlen: iovlen_t = 0;
    addBuf(iovecs, &iovlen, header);
    for (data[0 .. data.len - 1]) |bytes| addBuf(iovecs, &iovlen, bytes);
    const pattern = data[data.len - 1];
    if (iovecs.len - iovlen != 0) switch (splat) {
        0 => {},
        1 => addBuf(iovecs, &iovlen, pattern),
        else => switch (pattern.len) {
            0 => {},
            1 => {
                const memset_len = @min(splat_buffer.len, splat);
                const buf = splat_buffer[0..memset_len];
                @memset(buf, pattern[0]);
                addBuf(iovecs, &iovlen, buf);
                var remaining_splat = splat - buf.len;
                while (remaining_splat > splat_buffer.len and iovecs.len - iovlen != 0) {
                    assert(buf.len == splat_buffer.len);
                    addBuf(iovecs, &iovlen, splat_buffer);
                    remaining_splat -= splat_buffer.len;
                }
                addBuf(iovecs, &iovlen, splat_buffer[0..@min(remaining_splat, splat_buffer.len)]);
            },
            else => for (0..@min(splat, iovecs.len - iovlen)) |_| {
                addBuf(iovecs, &iovlen, pattern);
            },
        },
    };
    return iovecs[0..iovlen];
}

fn netWrite(
    ev: *Evented,
    cancel_region: *CancelRegion,
    handle: net.Socket.Handle,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
) net.Stream.Writer.Error!usize {
    var iovecs: [max_iovecs_len]iovec_const = undefined;
    var splat_backup_buffer: [splat_buffer_size]u8 = undefined;
    const used = netWriteIovecs(&iovecs, header, data, splat, &splat_backup_buffer);
    const msg: linux.msghdr_const = .{
        .name = null,
        .namelen = 0,
        .iov = used.ptr,
        .iovlen = used.len,
        .control = null,
        .controllen = 0,
        .flags = 0,
    };
    const posix_flags = linux.MSG.NOSIGNAL;
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .SENDMSG,
            .flags = 0,
            .ioprio = 0,
            .fd = handle,
            .off = 0,
            .addr = @intFromPtr(&msg),
            .len = 1,
            .rw_flags = posix_flags,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => return @intCast(completion.result),
            .INTR, .CANCELED => {},
            else => |err| return netWriteErrno(err),
        }
    }
}

/// Direct syscall variant of `netWrite`, for use once the kernel has reported
/// the socket to be writable, in contexts where awaiting the ring is not
/// possible.
fn netWriteSync(
    handle: net.Socket.Handle,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
) Io.Operation.NetWrite.Result {
    var iovecs: [max_iovecs_len]iovec_const = undefined;
    var splat_backup_buffer: [splat_buffer_size]u8 = undefined;
    const used = netWriteIovecs(&iovecs, header, data, splat, &splat_backup_buffer);
    const msg: linux.msghdr_const = .{
        .name = null,
        .namelen = 0,
        .iov = used.ptr,
        .iovlen = used.len,
        .control = null,
        .controllen = 0,
        .flags = 0,
    };
    while (true) {
        const rc = linux.sendmsg(handle, &msg, linux.MSG.NOSIGNAL);
        switch (linux.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => continue,
            else => |err| return netWriteErrno(err),
        }
    }
}

/// Direct syscall variant of `netReceive`, for use once the kernel has
/// reported the socket to be readable, in contexts where awaiting the ring is
/// not possible.
///
/// Receives as many messages as are immediately available. The first message
/// is received without `MSG_DONTWAIT` since the socket was reported to be
/// ready.
fn netReceiveSync(
    handle: net.Socket.Handle,
    message_buffer: []net.IncomingMessage,
    data_buffer: []u8,
    flags: net.ReceiveFlags,
) struct { ?net.Socket.ReceiveError, usize } {
    var data_i: usize = 0;
    for (message_buffer, 0..) |*message, message_i| {
        const remaining_data_buffer = data_buffer[data_i..];
        var storage: PosixAddress = undefined;
        var one_iovec: iovec = .{
            .base = remaining_data_buffer.ptr,
            .len = remaining_data_buffer.len,
        };
        var msg: linux.msghdr = .{
            .name = &storage.any,
            .namelen = @sizeOf(PosixAddress),
            .iov = (&one_iovec)[0..1],
            .iovlen = 1,
            .control = message.control.ptr,
            .controllen = @intCast(message.control.len),
            .flags = undefined,
        };
        const posix_flags: u32 =
            @as(u32, if (flags.oob) linux.MSG.OOB else 0) |
            @as(u32, if (flags.peek) linux.MSG.PEEK else 0) |
            @as(u32, if (flags.trunc) linux.MSG.TRUNC else 0) |
            linux.MSG.NOSIGNAL |
            @as(u32, if (message_i != 0) linux.MSG.DONTWAIT else 0);
        const rc: usize = while (true) {
            const rc = linux.recvmsg(handle, &msg, posix_flags);
            switch (linux.errno(rc)) {
                .INTR => {},
                else => break rc,
            }
        };
        switch (linux.errno(rc)) {
            .SUCCESS => {
                const data = remaining_data_buffer[0..@intCast(rc)];
                data_i += data.len;
                message.* = .{
                    .from = addressFromPosix(&storage),
                    .data = data,
                    .control = if (msg.control) |ptr| @as([*]u8, @ptrCast(ptr))[0..msg.controllen] else message.control,
                    .flags = .{
                        .eor = msg.flags & linux.MSG.EOR != 0,
                        .trunc = msg.flags & linux.MSG.TRUNC != 0,
                        .ctrunc = msg.flags & linux.MSG.CTRUNC != 0,
                        .oob = msg.flags & linux.MSG.OOB != 0,
                        .errqueue = msg.flags & linux.MSG.ERRQUEUE != 0,
                    },
                };
            },
            // All messages available have been received.
            .AGAIN => if (message_i != 0) return .{ null, message_i } else return .{
                // The socket was reported to be ready, and yet the first
                // message would have blocked.
                errnoBug(.AGAIN),
                0,
            },
            else => |err| return .{ netReceiveErrno(err), message_i },
        }
    }
    return .{ null, message_buffer.len };
}

/// Packs a length and a small value into one word, for storing in
/// `Io.Operation.Storage.Pending.Userdata`. The length of any real buffer is
/// far below the 16-bit-shift bound that is implied here.
fn packLenBits(len: usize, bits: u16) usize {
    assert(len <= std.math.maxInt(usize) >> 16);
    return len | (@as(usize, bits) << (@bitSizeOf(usize) - 16));
}

fn unpackLenBits(word: usize) struct { usize, u16 } {
    return .{
        word & (std.math.maxInt(usize) >> 16),
        @truncate(word >> (@bitSizeOf(usize) - 16)),
    };
}

/// Packs a socket handle and a length into one word, for storing in
/// `Io.Operation.Storage.Pending.Userdata`.
fn packHandleLen(handle: net.Socket.Handle, len: usize) usize {
    assert(handle >= 0);
    assert(len <= std.math.maxInt(u32));
    return @as(usize, @intCast(handle)) << 32 | @as(u32, @intCast(len));
}

fn unpackHandleLen(word: usize) struct { net.Socket.Handle, usize } {
    return .{ @intCast(word >> 32), @as(u32, @truncate(word)) };
}

/// The batch implementation of the network operations waits for the socket to
/// be reported ready by the kernel with a poll operation, and then performs
/// the operation itself with a syscall. This is the maximum useful value of
/// `Io.Operation.NetWrite.splat` for that purpose: `netWriteIovecs` can only
/// describe `max_iovecs_len` repetitions of a `splat_buffer_size`-byte pattern,
/// so any larger value has an identical effect.
const net_write_splat_max = max_iovecs_len * splat_buffer_size;

/// The first three words of the userdata are reserved for the scheduler.
const batch_net_state_offset = 3;
const batch_net_state_len = 4;

const poll_mask_readable = linux.POLL.IN | linux.POLL.ERR | linux.POLL.HUP;
const poll_mask_writable = linux.POLL.OUT | linux.POLL.ERR | linux.POLL.HUP;

/// Arms `storage` so that the operation is resumed when `handle` is reported
/// ready by the kernel, storing `state` in the storage for the completion
/// handler to consume.
fn batchNetSubmit(
    thread: *Thread,
    batch: *Io.Batch,
    storage: *Io.Operation.Storage,
    tag: Io.Operation.Tag,
    handle: net.Socket.Handle,
    poll_mask: u32,
    state: [batch_net_state_len]usize,
) void {
    storage.* = .{ .pending = .{
        .node = .{ .prev = batch.pending.tail, .next = .none },
        .tag = tag,
        .userdata = undefined,
    } };
    storage.pending.userdata[batch_net_state_offset .. batch_net_state_offset + batch_net_state_len].* = state;
    thread.enqueue().* = .{
        .opcode = .POLL_ADD,
        .flags = 0,
        .ioprio = 0,
        .fd = handle,
        .off = 0,
        .addr = 0,
        .len = 0,
        .rw_flags = poll_mask,
        .user_data = @intFromPtr(&storage.pending.userdata) | 0b10,
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
}

fn netReceive(
    ev: *Evented,
    cancel_region: *CancelRegion,
    handle: net.Socket.Handle,
    message_buffer: []net.IncomingMessage,
    data_buffer: []u8,
    flags: net.ReceiveFlags,
) struct { ?net.Socket.ReceiveError, usize } {
    var message_i: usize = 0;
    var data_i: usize = 0;

    while (true) {
        if (message_buffer.len - message_i == 0) return .{ null, message_i };
        const message = &message_buffer[message_i];
        const remaining_data_buffer = data_buffer[data_i..];
        var storage: PosixAddress = undefined;
        var iov: iovec = .{ .base = remaining_data_buffer.ptr, .len = remaining_data_buffer.len };
        var msg: linux.msghdr = .{
            .name = &storage.any,
            .namelen = @sizeOf(PosixAddress),
            .iov = (&iov)[0..1],
            .iovlen = 1,
            .control = message.control.ptr,
            .controllen = @intCast(message.control.len),
            .flags = undefined,
        };

        const thread = cancel_region.awaitIoUring() catch |err| return .{ err, message_i };
        thread.enqueue().* = .{
            .opcode = .RECVMSG,
            .flags = 0,
            .ioprio = 0,
            .fd = handle,
            .off = 0,
            .addr = @intFromPtr(&msg),
            .len = 0,
            .rw_flags = linux.MSG.NOSIGNAL |
                @as(u32, if (flags.oob) linux.MSG.OOB else 0) |
                @as(u32, if (flags.peek) linux.MSG.PEEK else 0) |
                @as(u32, if (flags.trunc) linux.MSG.TRUNC else 0),
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => {
                const data = remaining_data_buffer[0..@intCast(completion.result)];
                data_i += data.len;
                message.* = .{
                    .from = addressFromPosix(&storage),
                    .data = data,
                    .control = if (msg.control) |ptr| @as([*]u8, @ptrCast(ptr))[0..msg.controllen] else message.control,
                    .flags = .{
                        .eor = msg.flags & linux.MSG.EOR != 0,
                        .trunc = msg.flags & linux.MSG.TRUNC != 0,
                        .ctrunc = msg.flags & linux.MSG.CTRUNC != 0,
                        .oob = msg.flags & linux.MSG.OOB != 0,
                        .errqueue = msg.flags & linux.MSG.ERRQUEUE != 0,
                    },
                };
                message_i += 1;
                continue;
            },
            .AGAIN => unreachable,
            .INTR, .CANCELED => {},
            else => |err| return .{ netReceiveErrno(err), message_i },
        }
    }
}

fn netWriteFile(
    userdata: ?*anyopaque,
    socket_handle: net.Socket.Handle,
    header: []const u8,
    file_reader: *File.Reader,
    limit: Io.Limit,
) net.Stream.Writer.WriteFileError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();

    // Data the caller has already read from the file comes first, and it takes
    // up part of the limit.
    const buffered = limit.slice(file_reader.interface.buffered());
    var total: usize = 0;
    if (header.len != 0 or buffered.len != 0) {
        const n = try ev.netWrite(&cancel_region, socket_handle, header, &.{buffered}, 1);
        file_reader.interface.toss(n -| header.len);
        total = n;
        if (n != header.len + buffered.len) return n; // partial write; caller retries
    }
    const file_limit = @backingInt(limit) -| buffered.len;
    if (file_limit == 0) return total;
    const file_bytes = try spliceFileToSocket(
        ev,
        &cancel_region,
        file_reader,
        socket_handle,
        file_limit,
    );
    file_reader.pos += file_bytes;
    total += file_bytes;
    if (file_bytes == 0 and total == 0) {
        file_reader.size = file_reader.pos;
        return error.EndOfStream;
    }
    return total;
}

/// Copies up to `limit_bytes` (which are not already buffered by the reader)
/// from the file to the socket using `splice`.
///
/// `splice` requires one of its file descriptors to be a pipe, so the data
/// travels from the file into a pipe, and from the pipe into the socket.
///
/// The number of bytes that reached the socket is returned. Bytes that were
/// read into the pipe but not yet written to the socket are not counted, and
/// are re-read by the next call; this relies on `file_reader.pos` only being
/// advanced for bytes that reached the socket, which is possible by passing
/// the file offset by value rather than as a live file position.
fn spliceFileToSocket(
    ev: *Evented,
    cancel_region: *CancelRegion,
    file_reader: *File.Reader,
    socket_fd: net.Socket.Handle,
    limit_bytes: usize,
) net.Stream.Writer.WriteFileError!usize {
    if (file_reader.mode == .failure) return error.ReadFailed;
    const file_fd = file_reader.file.handle;
    const pipe_fds = pipe2(.{ .CLOEXEC = true }) catch |err| switch (err) {
        error.SystemFdQuotaExceeded, error.ProcessFdQuotaExceeded => return error.SystemResources,
        error.Unexpected => |e| return e,
    };
    defer destroyPipe(ev, pipe_fds);
    const pipe_read = pipe_fds[0];
    const pipe_write = pipe_fds[1];
    const pipe_capacity = pipeCapacity(pipe_read);

    var remaining = limit_bytes;
    var file_offset: ?u64 = switch (file_reader.mode) {
        .positional => @intCast(file_reader.pos),
        .streaming => null, // use the file position
        .streaming_simple, .positional_simple => return error.Unimplemented,
        .failure => unreachable, // checked above
    };
    var total: usize = 0;
    while (remaining != 0) {
        // Never ask for more than the pipe can hold, since the pipe is only
        // drained by this task, and filling a full pipe blocks the syscall.
        const chunk = @min(remaining, pipe_capacity);
        const fed = splice(ev, cancel_region, file_fd, file_offset, pipe_write, null, chunk) catch |err| switch (err) {
            // Nothing has been written yet, so the caller can fall back to a
            // read-based copy without any bytes being duplicated.
            error.Unimplemented => return if (total == 0) error.Unimplemented else total,
            else => |e| return if (total == 0) e else total,
        };
        if (fed == 0) break; // end of file
        if (file_offset) |*off| off.* += fed;
        // Only bytes that reach the socket are reported.
        var left = fed;
        while (left != 0) {
            const written = splice(ev, cancel_region, pipe_read, null, socket_fd, null, left) catch |err|
                return if (total == 0) err else total;
            if (written == 0) return total;
            left -= written;
            remaining -= written;
            total += written;
        }
    }
    return total;
}

fn pipeCapacity(pipe_read: fd_t) usize {
    // The minimum pipe capacity, used if the kernel does not report one.
    const default = 4096;
    const rc = linux.fcntl(pipe_read, linux.F.GETPIPE_SZ, 0);
    switch (linux.errno(rc)) {
        .SUCCESS => return if (rc == 0) default else @intCast(rc),
        else => return default,
    }
}

/// `off_in` and `off_out` are file offsets, or `null` to use the current file
/// position; pipes require `null`.
fn splice(
    ev: *Evented,
    cancel_region: *CancelRegion,
    fd_in: fd_t,
    off_in: ?u64,
    fd_out: fd_t,
    off_out: ?u64,
    len: usize,
) net.Stream.Writer.WriteFileError!usize {
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .SPLICE,
            .flags = 0,
            .ioprio = 0,
            .fd = fd_out,
            .off = off_out orelse std.math.maxInt(u64),
            .addr = off_in orelse std.math.maxInt(u64),
            .len = @intCast(len),
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = fd_in,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => return @intCast(@as(u32, @bitCast(completion.result))),
            .INTR, .CANCELED => {},
            else => |err| return spliceErrno(err),
        }
    }
}

fn spliceErrno(err: linux.E) net.Stream.Writer.WriteFileError {
    return switch (err) {
        // `splice` is not usable for this pair of file descriptors.
        .INVAL, .NOSYS, .OPNOTSUPP => error.Unimplemented,
        .NFILE, .MFILE => error.SystemResources,
        .NOBUFS, .NOMEM => error.SystemResources,
        .PIPE => error.SocketUnconnected,
        .NOTCONN => error.SocketUnconnected,
        .CONNREFUSED => error.SocketUnconnected,
        .CONNRESET => error.ConnectionResetByPeer,
        .TIMEDOUT => error.ConnectionTimedOut,
        .NETDOWN => error.NetworkDown,
        .NETUNREACH => error.NetworkUnreachable,
        .HOSTUNREACH => error.HostUnreachable,
        .AFNOSUPPORT => error.AddressFamilyUnsupported,

        .AGAIN => |e| errnoBug(e), // File descriptors are blocking.
        .BADF => |e| errnoBug(e), // File descriptor used after closed.
        .FAULT => |e| errnoBug(e),
        else => |e| unexpectedErrno(e),
    };
}

fn netClose(userdata: ?*anyopaque, sockets: []const net.Socket) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    for (sockets) |sock| ev.close(sock.handle);
}

fn netShutdown(
    userdata: ?*anyopaque,
    handle: net.Socket.Handle,
    how: net.ShutdownHow,
) net.ShutdownError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var cancel_region: CancelRegion = .init();
    defer cancel_region.deinit();
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .SHUTDOWN,
            .flags = 0,
            .ioprio = 0,
            .fd = handle,
            .off = 0,
            .addr = 0,
            .len = switch (how) {
                .recv => linux.SHUT.RD,
                .send => linux.SHUT.WR,
                .both => linux.SHUT.RDWR,
            },
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            .BADF, .NOTSOCK, .INVAL => |err| return errnoBug(err),
            .NOTCONN => return error.SocketUnconnected,
            .NOBUFS => return error.SystemResources,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn netInterfaceNameResolve(
    userdata: ?*anyopaque,
    name: *const net.Interface.Name,
) net.Interface.Name.ResolveError!net.Interface {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    const sock_fd = ev.socket(&maybe_sync.cancel_region, linux.AF.UNIX, .{ .mode = .dgram }) catch |err| switch (err) {
        error.ProcessFdQuotaExceeded => return error.SystemResources,
        error.SystemFdQuotaExceeded => return error.SystemResources,
        error.AddressFamilyUnsupported => return error.Unexpected,
        error.ProtocolUnsupportedBySystem => return error.Unexpected,
        error.ProtocolUnsupportedByAddressFamily => return error.Unexpected,
        error.SocketModeUnsupported => return error.Unexpected,
        error.OptionUnsupported => return error.Unexpected,
        else => |e| return e,
    };
    defer ev.closeAsync(sock_fd);

    var ifr: linux.ifreq = .{
        .ifrn = .{ .name = @bitCast(name.bytes) },
        .ifru = undefined,
    };
    ioctl(try maybe_sync.enterSync(ev), sock_fd, linux.SIOCGIFINDEX, &ifr) catch |err| switch (err) {
        error.NoDevice => return error.InterfaceNotFound,
        error.Canceled => |e| return e,
        error.Unexpected => |e| return e,
    };
    return .{ .index = @bitCast(ifr.ifru.ivalue) };
}

fn netInterfaceName(
    userdata: ?*anyopaque,
    interface: net.Interface,
) net.Interface.NameError!net.Interface.Name {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var maybe_sync: CancelRegion.Sync.Maybe = .{ .cancel_region = .init() };
    defer maybe_sync.deinit(ev);
    // `net.Interface.NameError` has no error for resource exhaustion.
    const sock_fd = ev.socket(&maybe_sync.cancel_region, linux.AF.UNIX, .{ .mode = .dgram }) catch |err| switch (err) {
        error.ProcessFdQuotaExceeded => return error.Unexpected,
        error.SystemFdQuotaExceeded => return error.Unexpected,
        error.SystemResources => return error.Unexpected,
        error.AddressFamilyUnsupported => return error.Unexpected,
        error.ProtocolUnsupportedBySystem => return error.Unexpected,
        error.ProtocolUnsupportedByAddressFamily => return error.Unexpected,
        error.SocketModeUnsupported => return error.Unexpected,
        error.OptionUnsupported => return error.Unexpected,
        else => |e| return e,
    };
    defer ev.closeAsync(sock_fd);

    var ifr: linux.ifreq = undefined;
    @memset(std.mem.asBytes(&ifr), 0);
    ifr.ifru.ivalue = @bitCast(interface.index);
    ioctl(try maybe_sync.enterSync(ev), sock_fd, linux.SIOCGIFNAME, &ifr) catch |err| switch (err) {
        error.NoDevice => return error.InterfaceNotFound,
        error.Canceled => |e| return e,
        error.Unexpected => |e| return e,
    };
    return .fromSliceUnchecked(std.mem.sliceTo(&ifr.ifrn.name, 0));
}

/// `arg` is the request-specific argument, e.g. an `ifreq`.
fn ioctl(
    sync: *CancelRegion.Sync,
    fd: fd_t,
    request: u32,
    arg: *anyopaque,
) (Io.Cancelable || Io.UnexpectedError || error{NoDevice})!void {
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.ioctl(fd, request, @intFromPtr(arg)))) {
            .SUCCESS => return,
            .INTR => {},
            // No such device; `SIOCGIFINDEX` uses it for "no such interface".
            .NODEV => return error.NoDevice,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn netLookup(
    userdata: ?*anyopaque,
    host_name: net.HostName,
    resolved: *Io.Queue(net.HostName.LookupResult),
    options: net.HostName.LookupOptions,
) net.HostName.LookupError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    defer resolved.close(ev.io());
    netLookupFallible(ev, host_name, resolved, options) catch |err| switch (err) {
        error.Closed => unreachable, // `resolved` must not be closed until `netLookup` returns
        else => |e| return e,
    };
}

/// Networking is available because this file is only compiled for Linux.
///
/// None of the I/O below happens on the calling thread: it all goes through
/// the `Io` interface, whose socket and file operations are serviced by io_uring.
fn netLookupFallible(
    ev: *Evented,
    host_name: net.HostName,
    resolved: *Io.Queue(net.HostName.LookupResult),
    options: net.HostName.LookupOptions,
) (net.HostName.LookupError || Io.QueueClosedError)!void {
    const ev_io = ev.io();
    const name = host_name.bytes;
    assert(name.len <= net.HostName.max_len);

    if (net.IpAddress.parseIp6(name, options.port)) |addr| {
        if (options.family == .ip4) return error.UnknownHostName;
        if (Io.Threaded.copyCanon(options.canonical_name_buffer, name)) |canon| {
            try resolved.putAll(ev_io, &.{
                .{ .address = addr },
                .{ .canonical_name = canon },
            });
        } else {
            try resolved.putOne(ev_io, .{ .address = addr });
        }
        return;
    } else |_| {}

    if (net.IpAddress.parseIp4(name, options.port)) |addr| {
        if (options.family == .ip6) return error.UnknownHostName;
        if (Io.Threaded.copyCanon(options.canonical_name_buffer, name)) |canon| {
            try resolved.putAll(ev_io, &.{
                .{ .address = addr },
                .{ .canonical_name = canon },
            });
        } else {
            try resolved.putOne(ev_io, .{ .address = addr });
        }
        return;
    } else |_| {}

    if (lookupHosts(ev, host_name, resolved, options)) return else |err| switch (err) {
        error.UnknownHostName => {},
        else => |e| return e,
    }

    // RFC 6761 Section 6.3.3
    // Name resolution APIs and libraries SHOULD recognize
    // localhost names as special and SHOULD always return the IP
    // loopback address for address queries and negative responses
    // for all other query types.

    // Check for equal to "localhost(.)" or ends in ".localhost(.)"
    const localhost = if (name[name.len - 1] == '.') "localhost." else "localhost";
    if (std.mem.endsWith(u8, name, localhost) and
        (name.len == localhost.len or name[name.len - localhost.len - 1] == '.'))
    {
        var results_buffer: [3]net.HostName.LookupResult = undefined;
        var results_index: usize = 0;
        if (options.family != .ip4) {
            results_buffer[results_index] = .{ .address = .{ .ip6 = .loopback(options.port) } };
            results_index += 1;
        }
        if (options.family != .ip6) {
            results_buffer[results_index] = .{ .address = .{ .ip4 = .loopback(options.port) } };
            results_index += 1;
        }
        if (options.canonical_name_buffer) |buf| {
            const canon_name = "localhost";
            const canon_name_dest = buf[0..canon_name.len];
            canon_name_dest.* = canon_name.*;
            results_buffer[results_index] = .{ .canonical_name = .{ .bytes = canon_name_dest } };
            results_index += 1;
        }
        try resolved.putAll(ev_io, results_buffer[0..results_index]);
        return;
    }

    return Io.Threaded.lookupDnsSearch(ev_io, host_name, resolved, options);
}

/// Same as `Io.Threaded.lookupHosts`, except that all I/O is routed through
/// this implementation's `Io` interface.
fn lookupHosts(
    ev: *Evented,
    host_name: net.HostName,
    resolved: *Io.Queue(net.HostName.LookupResult),
    options: net.HostName.LookupOptions,
) !void {
    const ev_io = ev.io();
    const file = Io.Dir.cwd().openFile(ev_io, "/etc/hosts", .{}) catch |err| switch (err) {
        error.FileNotFound,
        error.NotDir,
        error.AccessDenied,
        => return error.UnknownHostName,

        error.Canceled => |e| return e,

        else => {
            // Here we could add more detailed diagnostics to the results queue.
            return error.DetectingNetworkConfigurationFailed;
        },
    };
    defer file.close(ev_io);

    var line_buf: [512]u8 = undefined;
    var file_reader = file.reader(ev_io, &line_buf);
    return Io.Threaded.lookupHostsReader(
        ev_io,
        host_name,
        resolved,
        options,
        &file_reader.interface,
    ) catch |err| switch (err) {
        error.ReadFailed => switch (file_reader.err.?) {
            error.Canceled => |e| return e,
            else => {
                // Here we could add more detailed diagnostics to the results queue.
                return error.DetectingNetworkConfigurationFailed;
            },
        },
        error.Canceled,
        error.Closed,
        error.UnknownHostName,
        => |e| return e,
    };
}

fn bind(
    ev: *Evented,
    cancel_region: *CancelRegion,
    socket_fd: fd_t,
    addr: *const linux.sockaddr,
    addr_len: linux.socklen_t,
) !void {
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .BIND,
            .flags = 0,
            .ioprio = 0,
            .fd = socket_fd,
            .off = addr_len,
            .addr = @intFromPtr(addr),
            .len = 0,
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        switch (cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            .ACCES => return error.AccessDenied,
            .ADDRINUSE => return error.AddressInUse,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .INVAL => |err| return errnoBug(err), // invalid parameters
            .NOTSOCK => |err| return errnoBug(err), // invalid `sockfd`
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .ADDRNOTAVAIL => return error.AddressUnavailable,
            .FAULT => |err| return errnoBug(err), // invalid `addr` pointer
            .NOMEM => return error.SystemResources,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn chdir(sync: *CancelRegion.Sync, path: [*:0]const u8) ChdirError!void {
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.chdir(path))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .IO => return error.FileSystem,
            .LOOP => return error.SymLinkLoop,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOTDIR => return error.NotDir,
            .ILSEQ => return error.BadPathName,
            .FAULT => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn close(ev: *Evented, fd: fd_t) void {
    var cancel_region: CancelRegion = .initBlocked();
    defer cancel_region.deinit();
    const thread = cancel_region.awaitIoUring() catch |err| switch (err) {
        error.Canceled => unreachable, // blocked
    };
    thread.enqueue().* = .{
        .opcode = .CLOSE,
        .flags = 0,
        .ioprio = 0,
        .fd = fd,
        .off = 0,
        .addr = 0,
        .len = 0,
        .rw_flags = 0,
        .user_data = @intFromPtr(cancel_region.fiber),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
    ev.sched.park();
    switch (cancel_region.errno()) {
        .BADF => recoverableOsBugDetected(), // Always a race condition.
        .INTR => {}, // This is still a success. See https://github.com/ziglang/zig/issues/2425
        else => {},
    }
}

fn closeAsync(ev: *Evented, fd: fd_t) void {
    _ = ev;
    const thread: *Thread = .current();
    thread.enqueue().* = .{
        .opcode = .CLOSE,
        .flags = linux.IOSQE_CQE_SKIP_SUCCESS,
        .ioprio = 0,
        .fd = fd,
        .off = 0,
        .addr = 0,
        .len = 0,
        .rw_flags = 0,
        .user_data = @backingInt(Completion.Userdata.close),
        .buf_index = 0,
        .personality = 0,
        .splice_fd_in = 0,
        .addr3 = 0,
        .resv = 0,
    };
}

fn fchdir(sync: *CancelRegion.Sync, dir: fd_t) process.SetCurrentDirError!void {
    if (dir == linux.AT.FDCWD) return;
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.fchdir(dir))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .NOTDIR => return error.NotDir,
            .IO => return error.FileSystem,
            .BADF => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fchmodat(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    dir: fd_t,
    path: [*:0]const u8,
    mode: linux.mode_t,
    flags: u32,
) Dir.SetFilePermissionsError!void {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.fchmodat2(dir, path, mode, flags))) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .ACCES => return error.AccessDenied,
            .IO => return error.InputOutput,
            .LOOP => return error.SymLinkLoop,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOTDIR => return error.FileNotFound,
            .OPNOTSUPP => return error.OperationUnsupported,
            .PERM => return error.PermissionDenied,
            .ROFS => return error.ReadOnlyFileSystem,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fchownat(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    dir: fd_t,
    path: [*:0]const u8,
    owner: linux.uid_t,
    group: linux.gid_t,
    flags: u32,
) File.SetOwnerError!void {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.fchownat(dir, path, owner, group, flags))) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // likely fd refers to directory opened without `Dir.OpenOptions.iterate`
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .ACCES => return error.AccessDenied,
            .IO => return error.InputOutput,
            .LOOP => return error.SymLinkLoop,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOTDIR => return error.FileNotFound,
            .PERM => return error.PermissionDenied,
            .ROFS => return error.ReadOnlyFileSystem,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn flock(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    fd: fd_t,
    op: File.Lock,
    blocking: enum { blocking, nonblocking },
) (File.LockError || error{WouldBlock})!void {
    // The lock may be held through another open file description by a task on this worker, so the
    // worker must not wait for it in the kernel. A blocking lock sleeps and tries again, a
    // microsecond at first and twice as long each time, up to a millisecond.
    var retry: linux.kernel_timespec = .{ .sec = 0, .nsec = std.time.ns_per_us };
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.flock(fd, LOCK.NB | @as(i32, switch (op) {
            .none => LOCK.UN,
            .shared => LOCK.SH,
            .exclusive => LOCK.EX,
        })))) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err), // invalid parameters
            .NOLCK => return error.SystemResources,
            .AGAIN => {
                const thread = try sync.cancel_region.awaitIoUring();
                thread.enqueue().* = .{
                    .opcode = switch (blocking) {
                        .blocking => .TIMEOUT,
                        .nonblocking => .NOP,
                    },
                    .flags = 0,
                    .ioprio = 0,
                    .fd = 0,
                    .off = 0,
                    .addr = switch (blocking) {
                        .blocking => @intFromPtr(&retry),
                        .nonblocking => 0,
                    },
                    .len = switch (blocking) {
                        .blocking => 1,
                        .nonblocking => 0,
                    },
                    .rw_flags = 0,
                    .user_data = @intFromPtr(sync.cancel_region.fiber),
                    .buf_index = 0,
                    .personality = 0,
                    .splice_fd_in = 0,
                    .addr3 = 0,
                    .resv = 0,
                };
                ev.sched.park();
                switch (sync.cancel_region.errno()) {
                    .SUCCESS, .TIME, .INTR, .CANCELED => {},
                    else => unreachable,
                }
                switch (blocking) {
                    .blocking => retry.nsec = @min(retry.nsec * 2, std.time.ns_per_ms),
                    .nonblocking => return error.WouldBlock,
                }
            },
            .OPNOTSUPP => return error.FileLocksUnsupported,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn getsockname(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    socket_fd: fd_t,
    addr: *linux.sockaddr,
    addr_len: *linux.socklen_t,
) !void {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.getsockname(socket_fd, addr, addr_len))) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err), // invalid parameters
            .NOTSOCK => |err| return errnoBug(err), // always a race condition
            .NOBUFS => return error.SystemResources,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn linkat(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    old_dir: fd_t,
    old_path: [*:0]const u8,
    new_dir: fd_t,
    new_path: [*:0]const u8,
    flags: u32,
) File.HardLinkError!void {
    _ = ev;
    // allowed flags: https://man7.org/linux/man-pages/man2/linkat.2.html
    assert(flags & ~(@as(u32, linux.AT.SYMLINK_FOLLOW | linux.AT.EMPTY_PATH)) == 0);
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.linkat(old_dir, old_path, new_dir, new_path, flags))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .DQUOT => return error.DiskQuota,
            .EXIST => return error.PathAlreadyExists,
            .IO => return error.HardwareFailure,
            .LOOP => return error.SymLinkLoop,
            .MLINK => return error.LinkQuotaExceeded,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .NOTDIR => return error.NotDir,
            .PERM => return error.PermissionDenied,
            .ROFS => return error.ReadOnlyFileSystem,
            .XDEV => return error.CrossDevice,
            .ILSEQ => return error.BadPathName,
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn lseek(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    fd: fd_t,
    offset: u64,
    whence: u32,
) File.SeekError!void {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        var result: u64 = undefined;
        switch (linux.errno(switch (@sizeOf(usize)) {
            else => comptime unreachable,
            4 => linux.llseek(fd, offset, &result, whence),
            8 => linux.lseek(fd, @bitCast(offset), whence),
        })) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .INVAL => return error.Unseekable,
            .OVERFLOW => return error.Unseekable,
            .SPIPE => return error.Unseekable,
            .NXIO => return error.Unseekable,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn openat(
    ev: *Evented,
    cancel_region: *CancelRegion,
    dir: fd_t,
    path: [*:0]const u8,
    flags: linux.O,
    mode: linux.mode_t,
) !fd_t {
    var mut_flags = flags;
    if (@hasField(linux.O, "LARGEFILE")) mut_flags.LARGEFILE = true;
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .OPENAT,
            .flags = 0,
            .ioprio = 0,
            .fd = dir,
            .off = 0,
            .addr = @intFromPtr(path),
            .len = mode,
            .rw_flags = @bitCast(mut_flags),
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => return completion.result,
            .INTR, .CANCELED => {},
            else => |err| try openatError(err),
        }
    }
}

/// For an open that creates or truncates. See `CancelRegion.Sync`.
fn openatSync(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    dir: fd_t,
    path: [*:0]const u8,
    flags: linux.O,
    mode: linux.mode_t,
) !fd_t {
    _ = ev;
    var mut_flags = flags;
    if (@hasField(linux.O, "LARGEFILE")) mut_flags.LARGEFILE = true;
    while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.openat(dir, path, mut_flags, mode);
        switch (linux.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => {},
            else => |err| try openatError(err),
        }
    }
}

fn openatError(err: linux.E) !noreturn {
    switch (err) {
        .FAULT => return errnoBug(err),
        .INVAL => return error.BadPathName,
        .BADF => return errnoBug(err), // File descriptor used after closed.
        .ACCES => return error.AccessDenied,
        .FBIG => return error.FileTooBig,
        .OVERFLOW => return error.FileTooBig,
        .ISDIR => return error.IsDir,
        .LOOP => return error.SymLinkLoop,
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NAMETOOLONG => return error.NameTooLong,
        .NFILE => return error.SystemFdQuotaExceeded,
        .NODEV => return error.NoDevice,
        .NOENT => return error.FileNotFound,
        .SRCH => return error.FileNotFound, // Linux when opening procfs files.
        .NOMEM => return error.SystemResources,
        .NOSPC => return error.NoSpaceLeft,
        .NOTDIR => return error.NotDir,
        .PERM => return error.PermissionDenied,
        .EXIST => return error.PathAlreadyExists,
        .BUSY => return error.DeviceBusy,
        // This can be triggered by file locking and TMPFILE, but those
        // flags are mutually exclusive.
        .OPNOTSUPP => return error.OperationUnsupported,
        .AGAIN => return error.WouldBlock,
        .TXTBSY => return error.FileBusy,
        .NXIO => return error.NoDevice,
        .ROFS => return error.ReadOnlyFileSystem,
        .ILSEQ => return error.BadPathName,
        else => return unexpectedErrno(err),
    }
}

fn preadv(
    ev: *Evented,
    cancel_region: *CancelRegion,
    fd: fd_t,
    iov: []const iovec,
    offset: ?u64,
) File.Reader.Error!usize {
    if (iov.len == 0) return 0;
    const gather = iov.len > 1 or iov[0].len > 0xfffff000;
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = if (gather) .READV else .READ,
            .flags = 0,
            .ioprio = 0,
            .fd = fd,
            .off = offset orelse std.math.maxInt(u64),
            .addr = if (gather) @intFromPtr(iov.ptr) else @intFromPtr(iov[0].base),
            .len = @intCast(if (gather) iov.len else iov[0].len),
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => return @as(u32, @bitCast(completion.result)),
            .INTR, .CANCELED => {},
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .AGAIN => return error.WouldBlock,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed
            .IO => return error.InputOutput,
            .ISDIR => return error.IsDir,
            .NOBUFS => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .NOTCONN => return error.SocketUnconnected,
            .CONNRESET => return error.ConnectionResetByPeer,
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// A write at the file's position, on the ring. Positional writes are made on the worker; see
/// `CancelRegion.Sync`.
fn writev(
    ev: *Evented,
    cancel_region: *CancelRegion,
    fd: fd_t,
    iov: []const iovec_const,
) File.Writer.Error!usize {
    if (iov.len == 0) return 0;
    const scatter = iov.len > 1 or iov[0].len > 0xfffff000;
    while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = if (scatter) .WRITEV else .WRITE,
            .flags = 0,
            .ioprio = 0,
            .fd = fd,
            .off = std.math.maxInt(u64),
            .addr = if (scatter) @intFromPtr(iov.ptr) else @intFromPtr(iov[0].base),
            .len = @intCast(if (scatter) iov.len else iov[0].len),
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => return @as(u32, @bitCast(completion.result)),
            .INTR, .CANCELED => {},
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .AGAIN => return error.WouldBlock,
            .BADF => return error.NotOpenForWriting, // Can be a race condition.
            .DESTADDRREQ => |err| return errnoBug(err), // `connect` was never called.
            .DQUOT => return error.DiskQuota,
            .FBIG => return error.FileTooBig,
            .IO => return error.InputOutput,
            .NOSPC => return error.NoSpaceLeft,
            .PERM => return error.PermissionDenied,
            .PIPE => return error.BrokenPipe,
            .CONNRESET => |err| return errnoBug(err), // Not a socket handle.
            .BUSY => return error.DeviceBusy,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn readAll(
    ev: *Evented,
    cancel_region: *CancelRegion,
    fd: fd_t,
    buffer: []u8,
) (File.Reader.Error || error{EndOfStream})!void {
    var index: usize = 0;
    while (buffer.len - index != 0) {
        const len = try ev.preadv(cancel_region, fd, &.{
            .{ .base = buffer[index..].ptr, .len = buffer.len - index },
        }, null);
        if (len == 0) return error.EndOfStream;
        index += len;
    }
}

fn realPath(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    fd: fd_t,
    out_buffer: []u8,
) File.RealPathError!usize {
    _ = ev;
    var procfs_buf: [std.fmt.count("/proc/self/fd/{d}\x00", .{std.math.minInt(fd_t)})]u8 = undefined;
    const proc_path = std.mem.printSentinel(&procfs_buf, "/proc/self/fd/{d}", .{fd}, 0) catch
        unreachable;
    while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.readlink(proc_path, out_buffer.ptr, out_buffer.len);
        switch (linux.errno(rc)) {
            .SUCCESS => return rc,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .FAULT => |err| return errnoBug(err),
            .IO => return error.FileSystem,
            .LOOP => return error.SymLinkLoop,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOTDIR => return error.NotDir,
            .ILSEQ => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn renameat(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    old_dir: fd_t,
    old_path: [*:0]const u8,
    new_dir: fd_t,
    new_path: [*:0]const u8,
    flags: linux.RENAME,
) (Dir.RenameError || error{PathAlreadyExists})!void {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.renameat2(old_dir, old_path, new_dir, new_path, flags))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .PERM => return error.PermissionDenied,
            .BUSY => return error.FileBusy,
            .DQUOT => return error.DiskQuota,
            .ISDIR => return error.IsDir,
            .IO => return error.HardwareFailure,
            .LOOP => return error.SymLinkLoop,
            .MLINK => return error.LinkQuotaExceeded,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            // With NOREPLACE, the new path exists. Without, it is a directory that is not empty.
            .EXIST => return if (flags.NOREPLACE) error.PathAlreadyExists else error.DirNotEmpty,
            .NOTEMPTY => return error.DirNotEmpty,
            .ROFS => return error.ReadOnlyFileSystem,
            .XDEV => return error.CrossDevice,
            .ILSEQ => return error.BadPathName,
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn setsockopt(
    ev: *Evented,
    cancel_region: *CancelRegion,
    fd: fd_t,
    level: i32,
    opt_name: u32,
    option: u32,
) !void {
    const o: []const u8 = @ptrCast(&option);
    while (true) {
        const off: extern struct {
            cmd_op: linux.IO_URING_SOCKET_OP,
            pad: u32,
        } align(@alignOf(u64)) = .{
            .cmd_op = .SETSOCKOPT,
            .pad = 0,
        };
        const addr: extern struct { level: i32, opt_name: u32 } align(@alignOf(u64)) = .{
            .level = level,
            .opt_name = opt_name,
        };
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .URING_CMD,
            .flags = 0,
            .ioprio = 0,
            .fd = fd,
            .off = @as(*const u64, @ptrCast(&off)).*,
            .addr = @as(*const u64, @ptrCast(&addr)).*,
            .len = 0,
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = @intCast(o.len),
            .addr3 = @intFromPtr(o.ptr),
            .resv = 0,
        };
        ev.sched.park();
        switch (cancel_region.errno()) {
            .SUCCESS => return,
            .INTR, .CANCELED => {},
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .NOTSOCK => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn socket(
    ev: *Evented,
    cancel_region: *CancelRegion,
    family: linux.sa_family_t,
    options: net.IpAddress.BindOptions,
) error{
    AddressFamilyUnsupported,
    ProtocolUnsupportedBySystem,
    ProcessFdQuotaExceeded,
    SystemFdQuotaExceeded,
    SystemResources,
    ProtocolUnsupportedByAddressFamily,
    SocketModeUnsupported,
    OptionUnsupported,
    Unexpected,
    Canceled,
}!fd_t {
    const mode, const protocol = try posixSocketModeProtocol(family, options.mode, options.protocol);
    const socket_fd = while (true) {
        const thread = try cancel_region.awaitIoUring();
        thread.enqueue().* = .{
            .opcode = .SOCKET,
            .flags = 0,
            .ioprio = 0,
            .fd = family,
            .off = mode | linux.SOCK.CLOEXEC,
            .addr = 0,
            .len = protocol,
            .rw_flags = 0,
            .user_data = @intFromPtr(cancel_region.fiber),
            .buf_index = 0,
            .personality = 0,
            .splice_fd_in = 0,
            .addr3 = 0,
            .resv = 0,
        };
        ev.sched.park();
        const completion = cancel_region.completion();
        switch (completion.errno()) {
            .SUCCESS => break completion.result,
            .INTR, .CANCELED => {},
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .INVAL => return error.ProtocolUnsupportedBySystem,
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NFILE => return error.SystemFdQuotaExceeded,
            .NOBUFS => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .PROTONOSUPPORT => return error.ProtocolUnsupportedByAddressFamily,
            .PROTOTYPE => return error.SocketModeUnsupported,
            else => |err| return unexpectedErrno(err),
        }
    };
    errdefer ev.closeAsync(socket_fd);

    if (options.ip6_only) |ip6_only| {
        if (linux.IPV6 == void) return error.OptionUnsupported;
        try ev.setsockopt(cancel_region, socket_fd, linux.IPPROTO.IPV6, linux.IPV6.V6ONLY, @intFromBool(ip6_only));
    }

    return socket_fd;
}

fn stat(ev: *Evented, sync: *CancelRegion.Sync, fd: fd_t) Dir.StatError!Dir.Stat {
    return ev.statx(sync, fd, "", linux.AT.EMPTY_PATH) catch |err| switch (err) {
        error.BadPathName, error.NameTooLong => unreachable, // path is empty
        error.AccessDenied => return errnoBug(.ACCES),
        error.SymLinkLoop => return errnoBug(.LOOP),
        error.FileNotFound => return errnoBug(.NOENT),
        error.NotDir => return errnoBug(.NOTDIR),
        else => |e| return e,
    };
}

fn statx(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    dir: fd_t,
    path: [*:0]const u8,
    flags: u32,
) (Dir.StatError || Dir.PathNameError || error{ FileNotFound, NotDir, SymLinkLoop })!Dir.Stat {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        var statx_buf = std.mem.zeroes(linux.Statx);
        switch (linux.errno(linux.statx(dir, path, flags, linux_statx_request, &statx_buf))) {
            .SUCCESS => return statFromLinux(&statx_buf),
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .LOOP => return error.SymLinkLoop,
            .NAMETOOLONG => |err| return errnoBug(err),
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .NOMEM => return error.SystemResources,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn urandomReadAll(
    ev: *Evented,
    cancel_region: *CancelRegion,
    buffer: []u8,
) (File.OpenError || File.Reader.Error || error{EndOfStream})!void {
    return ev.readAll(cancel_region, try ev.random_fd.open(ev, cancel_region, "/dev/urandom", .{
        .ACCMODE = .RDONLY,
        .CLOEXEC = true,
    }), buffer);
}

fn utimensat(
    ev: *Evented,
    sync: *CancelRegion.Sync,
    dir: fd_t,
    path: [*:0]const u8,
    times: ?*const [2]linux.timespec,
    flags: u32,
) File.SetTimestampsError!void {
    _ = ev;
    while (true) {
        try sync.cancel_region.await(.nothing);
        switch (linux.errno(linux.utimensat(dir, path, times, flags))) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // always a race condition
            .FAULT => |err| return errnoBug(err),
            .INVAL => |err| return errnoBug(err),
            .ACCES => return error.AccessDenied,
            .PERM => return error.PermissionDenied,
            .ROFS => return error.ReadOnlyFileSystem,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn writeAllSync(sync: *CancelRegion.Sync, fd: fd_t, buffer: []const u8) File.Writer.Error!void {
    var index: usize = 0;
    while (buffer.len - index != 0) index += try writeSync(sync, fd, buffer[index..]);
}

fn writeSync(sync: *CancelRegion.Sync, fd: fd_t, buffer: []const u8) File.Writer.Error!usize {
    while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.write(fd, buffer.ptr, buffer.len);
        switch (linux.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => {},
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .AGAIN => return error.WouldBlock,
            .BADF => return error.NotOpenForWriting, // Can be a race condition.
            .DESTADDRREQ => |err| return errnoBug(err), // `connect` was never called.
            .DQUOT => return error.DiskQuota,
            .FBIG => return error.FileTooBig,
            .IO => return error.InputOutput,
            .NOSPC => return error.NoSpaceLeft,
            .PERM => return error.PermissionDenied,
            .PIPE => return error.BrokenPipe,
            .CONNRESET => |err| return errnoBug(err), // Not a socket handle.
            .BUSY => return error.DeviceBusy,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn pwritevSync(
    sync: *CancelRegion.Sync,
    fd: fd_t,
    iov: []const iovec_const,
    offset: u64,
) File.WritePositionalError!usize {
    if (iov.len == 0) return 0;
    while (true) {
        try sync.cancel_region.await(.nothing);
        const rc = linux.pwritev(fd, iov.ptr, iov.len, @bitCast(offset));
        switch (linux.errno(rc)) {
            .SUCCESS => return rc,
            .INTR => {},
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .DESTADDRREQ => |err| return errnoBug(err), // `connect` was never called.
            .CONNRESET => |err| return errnoBug(err), // Not a socket handle.
            .BADF => return error.NotOpenForWriting, // Can be a race condition.
            .AGAIN => return error.WouldBlock,
            .DQUOT => return error.DiskQuota,
            .FBIG => return error.FileTooBig,
            .IO => return error.InputOutput,
            .NOSPC => return error.NoSpaceLeft,
            .PERM => return error.PermissionDenied,
            .PIPE => return error.BrokenPipe,
            .BUSY => return error.DeviceBusy,
            .TXTBSY => return error.FileBusy,
            .NXIO, .SPIPE, .OVERFLOW => return error.Unseekable,
            else => |err| return unexpectedErrno(err),
        }
    }
}

test {
    _ = Fiber.CancelProtection;
}

test "Ring: the owner takes its completions while its worker runs other tasks" {
    const testing = std.testing;
    const ev = fromIo(testing.io) orelse return error.SkipZigTest;
    const Owner = struct {
        const nops = 100;

        /// The ring is small, so a full queue is submitted to make room.
        fn getSqe(ring: Ring) !*linux.io_uring_sqe {
            return ring.uring().get_sqe() catch |err| switch (err) {
                error.SubmissionQueueFull => {
                    _ = try ring.uring().submit();
                    return ring.uring().get_sqe();
                },
            };
        }

        /// Returns how many of its completions it saw, each once and in order.
        fn run(e: *Evented) !usize {
            const ring = try e.acquireRing();
            defer ring.release();
            for (0..nops) |i| {
                const sqe = try getSqe(ring);
                sqe.prep_nop();
                sqe.user_data = ring_owner_bit | i;
            }
            // Its own Io call parks it while the NOPs complete, so the worker's poll takes
            // them off the ring and queues them.
            try testing.io.sleep(.fromMilliseconds(2), .awake);
            // A timeout of its own makes `waitCqes` park it.
            const ts: linux.kernel_timespec = .{ .sec = 0, .nsec = 5 * std.time.ns_per_ms };
            const sqe = try getSqe(ring);
            sqe.prep_timeout(&ts, 0, 0);
            sqe.user_data = ring_owner_bit | nops;

            var cqes: [16]linux.io_uring_cqe = undefined;
            var next: u64 = 0;
            while (next <= nops) {
                for (cqes[0..ring.waitCqes(&cqes)]) |cqe| {
                    try testing.expectEqual(ring_owner_bit | next, cqe.user_data);
                    next += 1;
                }
            }
            return next;
        }
    };
    var owner = try ev.concurrentWith(.{ .affinity = .{ .pinned = 0 } }, Owner.run, .{ev});
    defer _ = owner.cancel(testing.io) catch {};
    // Meanwhile, tasks of the same worker keep using the ring.
    for (0..5) |_| try testing.io.sleep(.fromMilliseconds(1), .awake);
    try testing.expectEqual(Owner.nops + 1, try owner.await(testing.io));
}

test "Ring: one pinned owner at a time" {
    const testing = std.testing;
    const ev = fromIo(testing.io) orelse return error.SkipZigTest;
    const Acquire = struct {
        fn run(e: *Evented) !void {
            const ring = try e.acquireRing();
            ring.release();
        }

        fn twice(e: *Evented) !void {
            const ring = try e.acquireRing();
            defer ring.release();
            try testing.expectError(error.RingOwned, e.acquireRing());
        }
    };
    var sticky = try ev.concurrentWith(.{}, Acquire.run, .{ev});
    try testing.expectError(error.NotPinned, sticky.await(testing.io));
    var pinned = try ev.concurrentWith(.{ .affinity = .{ .pinned = 0 } }, Acquire.twice, .{ev});
    try pinned.await(testing.io);
}

/// Nanoseconds on the monotonic clock, read without making an `Io` call: a task in these tests
/// has to keep its worker, and an `Io` call would charge the budget and switch it out.
fn testNow() u64 {
    var tp: linux.timespec = undefined;
    assert(linux.errno(linux.clock_gettime(linux.CLOCK.MONOTONIC, &tp)) == .SUCCESS);
    return @as(u64, @intCast(tp.sec)) * std.time.ns_per_s + @as(u64, @intCast(tp.nsec));
}

/// A task the tests below queue behind one that will not let its worker go. It makes an `Io`
/// call that parks, so finishing it needs a worker of its own.
fn testQueuedTask(done: *std.atomic.Value(u32)) void {
    std.testing.io.sleep(.fromMicroseconds(200), .awake) catch {};
    _ = done.fetchAdd(1, .monotonic);
}

/// Runs `hostile` as a task pinned to worker 1, with the `queued` tasks it makes of its own, and
/// checks what the watchdog does about it: the tasks behind it finish while it still holds the
/// worker, the stuck counter moves, and the report names the task and the function it runs.
fn testWatchdog(
    ev: *Evented,
    comptime hostile: anytype,
    extra: anytype,
    queued: usize,
    limit_ns: u64,
    comptime expected_name: []const u8,
) !void {
    const testing = std.testing;
    if (workerLimit(ev) < 2) return error.SkipZigTest; // a worker to stick, and one to run elsewhere

    var done: std.atomic.Value(u32) = .init(0);
    var started: std.atomic.Value(bool) = .init(false);
    var group: Io.Group = .init;
    const args = .{ testing.io, &group, &started, &done, queued } ++ extra;
    var future = try ev.concurrentWith(.{ .affinity = .{ .pinned = 1 } }, hostile, args);
    defer future.cancel(testing.io);
    defer group.cancel(testing.io);

    const before = ev.stats().stuck_episodes;
    while (!started.load(.acquire)) try testing.io.sleep(.fromMicroseconds(50), .awake);
    const start = testNow();
    while (done.load(.acquire) != queued) {
        try testing.expect(testNow() - start < limit_ns);
        try testing.io.sleep(.fromMicroseconds(200), .awake);
    }
    try testing.expect(testNow() - start < limit_ns);
    const snapshot = ev.stats();
    try testing.expect(snapshot.stuck_episodes > before);
    const stuck = snapshot.last_stuck orelse return error.TestUnexpectedResult;
    // The same task and name are what the watchdog names in its log line.
    try testing.expectEqualStrings(expected_name, stuck.name);
    try testing.expect(stuck.ms >= stuck_after_ms);
    try testing.expect(stuck.id != 0);
    try testing.expectEqual(@as(u32, 1), stuck.worker);
    future.await(testing.io);
    try group.await(testing.io);
}

/// `Scheduler.stuck_after`, in milliseconds.
const stuck_after_ms = 100;

test "watchdog: a spinning task costs one worker, and nothing else waits" {
    const testing = std.testing;
    const ev = fromIo(testing.io) orelse return error.SkipZigTest;
    const S = struct {
        /// Spins for 500 ms without making an `Io` call: the worker has nothing to switch to, and
        /// the tasks this one queued behind itself wait for another worker to take them.
        fn spin(inner_io: Io, group: *Io.Group, started: *std.atomic.Value(bool), done: *std.atomic.Value(u32), queued: usize, spin_ns: u64) void {
            for (0..queued) |_| group.async(inner_io, testQueuedTask, .{done});
            started.store(true, .release);
            const until = testNow() + spin_ns;
            var x: u64 = 0;
            while (testNow() < until) {
                inline for (0..8) |i| x +%= i;
                std.mem.doNotOptimizeAway(x);
            }
        }
    };
    try testWatchdog(ev, S.spin, .{500 * std.time.ns_per_ms}, 32, 150 * std.time.ns_per_ms, "spin");
}

test "watchdog: a blocking syscall is reported like a spin" {
    const testing = std.testing;
    const ev = fromIo(testing.io) orelse return error.SkipZigTest;
    const S = struct {
        /// A raw nanosleep in a task that forgot `io.blocking`: the kernel holds the worker, and
        /// nothing of Threadz's runs on it until it returns.
        fn sleepRaw(inner_io: Io, group: *Io.Group, started: *std.atomic.Value(bool), done: *std.atomic.Value(u32), queued: usize, sleep_ns: u64) void {
            _ = inner_io;
            for (0..queued) |_| group.async(testing.io, testQueuedTask, .{done});
            started.store(true, .release);
            const ts: linux.timespec = .{ .sec = 0, .nsec = @intCast(sleep_ns) };
            _ = linux.nanosleep(&ts, null);
        }
    };
    try testWatchdog(ev, S.sleepRaw, .{300 * std.time.ns_per_ms}, 4, 150 * std.time.ns_per_ms, "sleepRaw");
}

test "io.blocking: 16 blocking calls run at once, holding no worker" {
    const testing = std.testing;
    const ev = fromIo(testing.io) orelse return error.SkipZigTest;
    const S = struct {
        /// A raw 200 ms nanosleep, the kind of call `io.blocking` is for.
        fn sleepRaw(ms: u64) void {
            const ts: linux.timespec = .{ .sec = 0, .nsec = @intCast(ms * std.time.ns_per_ms) };
            _ = linux.nanosleep(&ts, null);
        }

        fn blockingCall(done: *std.atomic.Value(u32)) void {
            testing.io.blocking(sleepRaw, .{@as(u64, 200)});
            _ = done.fetchAdd(1, .monotonic);
        }

        /// The progress of another task, to show that the workers run tasks while the calls
        /// block.
        fn progress(stop: *std.atomic.Value(bool), steps: *std.atomic.Value(u32)) void {
            while (!stop.load(.acquire)) {
                testing.io.sleep(.fromMilliseconds(1), .awake) catch {};
                _ = steps.fetchAdd(1, .monotonic);
            }
        }
    };

    var done: std.atomic.Value(u32) = .init(0);
    var group: Io.Group = .init;
    const calls = 16;
    for (0..calls) |_| group.async(testing.io, S.blockingCall, .{&done});
    var stop: std.atomic.Value(bool) = .init(false);
    var steps: std.atomic.Value(u32) = .init(0);
    var progress_future = try ev.concurrentWith(.{}, S.progress, .{ &stop, &steps });
    defer {
        stop.store(true, .release);
        progress_future.cancel(testing.io);
    }

    const start = testNow();
    while (done.load(.acquire) != calls) {
        try testing.expect(testNow() - start < 4 * calls * 200 * std.time.ns_per_ms);
        try testing.io.sleep(.fromMicroseconds(200), .awake);
    }
    const elapsed = testNow() - start;
    // `calls` calls of 200 ms each, in a fraction of the time they would take one after another.
    try testing.expect(elapsed < 1000 * std.time.ns_per_ms);
    // And a task made progress throughout: no worker was held by a blocking call.
    try testing.expect(steps.load(.monotonic) >= 20);
    stop.store(true, .release);
    progress_future.await(testing.io);
    try group.await(testing.io);
}

test "budget: a task that loops on Io without parking yields to its worker's queue" {
    const testing = std.testing;
    const ev = fromIo(testing.io) orelse return error.SkipZigTest;
    if (workerLimit(ev) < 2) return error.SkipZigTest;
    const S = struct {
        /// 10,000 `Io` operations that complete without parking. Every `Scheduler.budget` of
        /// them the task is switched out, so the other task on this worker runs first.
        fn loop(word: *std.atomic.Value(u32), finished: *std.atomic.Value(bool)) void {
            for (0..10_000) |_| testing.io.futexWake(u32, &word.raw, 1);
            finished.store(true, .release);
        }

        /// The other task on the worker: it runs only while the looper is switched out.
        fn other(inner_io: Io, finished: *std.atomic.Value(bool), before_finish: *std.atomic.Value(bool)) void {
            _ = inner_io;
            before_finish.store(!finished.load(.acquire), .release);
        }
    };
    var word: std.atomic.Value(u32) = .init(0);
    var finished: std.atomic.Value(bool) = .init(false);
    var before_finish: std.atomic.Value(bool) = .init(false);
    const options: SpawnOptions = .{ .affinity = .{ .pinned = 1 } };
    var looper = try ev.concurrentWith(options, S.loop, .{ &word, &finished });
    var other = try ev.concurrentWith(options, S.other, .{ testing.io, &finished, &before_finish });
    looper.await(testing.io);
    other.await(testing.io);
    try testing.expect(finished.load(.acquire));
    // The other task ran while the looper had not finished: it was switched out on the way.
    try testing.expect(before_finish.load(.acquire));
    try testing.expect(ev.stats().stuck_episodes == 0); // the budget, not the watchdog, yielded it
}
