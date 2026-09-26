//! Tasks on their own stacks, parked at every Io call and scheduled across a pool of workers,
//! with kqueue as the kernel's side: Darwin and the BSDs. See `std.Io.Threadz` for what a core
//! is and `lib/std/Io/Threadz/scheduler.zig` for the scheduler this one runs on.
//!
//! One kqueue per worker. Every operation that has to wait registers an event on the worker's
//! kqueue, switches the task away, and retries its system call when the event arrives:
//!
//! * a socket, a pipe or a terminal waits on `EVFILT.READ` or `EVFILT.WRITE`, oneshot, with the
//!   task in the event's `udata`, and the operation retries the call that would have blocked; a
//!   second task waiting on the same descriptor waits on the same event, and one that arrives
//!   makes all of them runnable, so that no task is left waiting on an event the kernel has
//!   already delivered;
//! * a connect returns `EINPROGRESS` and then waits for `EVFILT.WRITE`, after which `SO_ERROR`
//!   says whether it connected;
//! * `sleep` and every timeout is an `EVFILT.TIMER` with an ident of its own and `NOTE.NSECONDS`
//!   where the platform has it, in milliseconds where it does not, so an operation with a
//!   deadline waits on two events and whichever fires first ends the operation;
//! * `childWait` is `EVFILT.PROC` with `NOTE.EXIT` on the child, and `wait4` collects the status;
//! * `futexWait` is a table in this process: the task parks, and a wake takes the tasks waiting
//!   on its address out of the table and makes them runnable. A thread that is not one of the
//!   workers waits in the kernel instead, on `scheduler.Futex`, and a wake reaches both;
//! * `wake` is a trigger on the worker's own `EVFILT.USER` event, which any thread may set on any
//!   kqueue.
//!
//! Regular files, directories, `stat`, locks, mappings, and process spawn and exec are made on
//! the worker itself, as `Uring`'s synchronous paths are: kqueue has no event for them, and the
//! calls are quick. A regular file's `fsync` is not, and goes to the scheduler's pool, the same
//! one `io.blocking` runs on. `randomSecure` is `arc4random_buf` or `getentropy`.

const Evented = @This();

const addressFromPosix = Io.Threaded.addressFromPosix;
const addressToPosix = Io.Threaded.addressToPosix;
const addressUnixToPosix = Io.Threaded.addressUnixToPosix;
const Alignment = std.mem.Alignment;
const Allocator = std.mem.Allocator;
const Argv0 = Io.Threaded.Argv0;
const assert = std.debug.assert;
const builtin = @import("builtin");
const c = std.c;
const ChdirError = Io.Threaded.ChdirError;
const clockToPosix = Io.Threaded.clockToPosix;
const closeFd = Io.Threaded.closeFd;
const Csprng = Io.Threaded.Csprng;
const default_PATH = Io.Threaded.default_PATH;
const Dir = Io.Dir;
const Environ = Io.Threaded.Environ;
const errnoBug = Io.Threaded.errnoBug;
const fallbackSeed = Io.Threaded.fallbackSeed;
const fd_t = posix.fd_t;
const Fiber = Scheduler.Task;
const File = Io.File;
const Io = std.Io;
const iovec = std.posix.iovec;
const iovec_const = std.posix.iovec_const;
const log = std.log.scoped(.threadz);
const max_iovecs_len = Io.Threaded.max_iovecs_len;
const nanosecondsFromPosix = Io.Threaded.nanosecondsFromPosix;
const net = Io.net;
const PATH_MAX = std.fs.max_path_bytes;
const pathToPosix = Io.Threaded.pathToPosix;
const pid_t = posix.pid_t;
const PosixAddress = Io.Threaded.PosixAddress;
const posix = std.posix;
const posixAddressFamily = Io.Threaded.posixAddressFamily;
const posixSocketModeProtocol = Io.Threaded.posixSocketModeProtocol;
const process = std.process;
const recoverableOsBugDetected = Io.Threaded.recoverableOsBugDetected;
const scheduler = @import("Threadz/scheduler.zig");
const Scheduler = scheduler.Scheduler(Evented);
const setTimestampToPosix = Io.Threaded.setTimestampToPosix;
const splat_buffer_size = Io.Threaded.splat_buffer_size;
const statFromPosix = Io.Threaded.statFromPosix;
const std = @import("../std.zig");
const statusToTerm = Io.Threaded.statusToTerm;
const timestampFromPosix = Io.Threaded.timestampFromPosix;
const unexpectedErrno = std.posix.unexpectedErrno;
const UnixAddress = Io.Threaded.UnixAddress;

/// An event's filter is an `i16` on Darwin and FreeBSD, an `i32` on NetBSD, its `data` is an
/// `isize` on some and an `i64` on others, so everything that names a filter or reads a data
/// value goes through these.
const Filter = @FieldType(posix.Kevent, "filter");
const Data = @FieldType(posix.Kevent, "data");
const Udata = @FieldType(posix.Kevent, "udata");
const Ident = @FieldType(posix.Kevent, "ident");

/// The filters this core uses, in the platform's type for one.
const filt_read: Filter = std.c.EVFILT.READ;
const filt_write: Filter = std.c.EVFILT.WRITE;
const filt_timer: Filter = std.c.EVFILT.TIMER;
const filt_proc: Filter = std.c.EVFILT.PROC;
const filt_user: Filter = std.c.EVFILT.USER;

/// The one `EVFILT.USER` ident every worker's kqueue has: a trigger on it ends that worker's
/// blocking `kevent`. See `wake`.
const wake_ident: Ident = 1;

/// An event's registration adds it, and it goes away by itself once it has been delivered.
const register_flags = std.c.EV.ADD | std.c.EV.ONESHOT;

/// A task registers a descriptor, or a timer, and an operation with a deadline both: no
/// operation needs more.
const max_registrations = 2;

/// The events one `poll` reads. A worker handles what it gets and polls again, so this is a
/// batch size, not a limit on the work in flight.
const max_events = 64;

/// The buckets the tasks waiting in `futexWait` are hashed into.
const futex_buckets_len = 64;

/// `log2(futex_buckets_len)`, the bits of the hash a bucket index is taken from.
const futex_bucket_bits = std.math.log2_int(usize, futex_buckets_len);

/// Must be a thread-safe allocator.
backing_allocator: Allocator,
sched: Scheduler,

backing_allocator_needs_mutex: bool = true,
backing_allocator_mutex: Io.Mutex,

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

/// The tasks parked in `futexWait`, by the address they wait on. See `FutexBucket`.
futex_buckets: [futex_buckets_len]FutexBucket = @splat(.{}),

/// The `Csprng` `random` draws from, seeded on first use.
csprng_mutex: Io.Mutex,
csprng: Csprng,

/// The first argument, for the OpenBSD executable path, which has nothing else to go on. See
/// `InitOptions.argv0`.
argv0: Argv0,

/// The null device, opened once, for the streams a spawn was told to ignore. The child gets a
/// duplicate of the descriptor, so one open serves every spawn.
dev_null_lock: Scheduler.SpinLock = .{},
dev_null_file: ?File = null,

/// A worker's kqueue and the tasks parked on it. See the scheduler's backend contract.
pub const Worker = struct {
    /// The worker's kqueue. Every event its tasks wait on is registered here, and its
    /// `EVFILT.USER` event is how the worker is woken.
    kq_fd: posix.fd_t,
    /// The tasks this worker parked, oldest first, linked through their completion's `next`. The
    /// worker adds and removes entries, and a cancellation from another worker walks them and
    /// completes one, so both hold `parked_lock`.
    parked_lock: Scheduler.SpinLock = .{},
    parked: ?*Fiber = null,
    /// The ident of this worker's next timer event. A timer is the one event whose ident is not
    /// a descriptor or a process, so timers are numbered: an ident stays unique while any timer
    /// of this worker is live, which is what a completion and a cancellation need to delete the
    /// right one.
    next_timer: u32 = 1,
    /// The events `poll` reads into, reused.
    events: [max_events]posix.Kevent = undefined,
    /// This worker's generator for `random`, seeded on first use from the instance's own.
    csprng: Csprng = .uninitialized,
    /// What reads this worker's thread CPU time, recorded on its own thread, for
    /// `threadCpuTime`. Nothing where this platform cannot say.
    cpu: scheduler.CpuTime.Source = .{},
};

/// The core's own per-worker state, which the scheduler wraps: `Thread` is the name the code
/// below uses for it.
pub const Thread = Worker;

/// The worker the calling thread runs, for the code that needs it without the scheduler's.
fn currentWorker() *Worker {
    return &Scheduler.Worker.current().backend;
}

/// For the scheduler: sets up worker `worker.index`'s kqueue, with its own wake event on it.
pub fn workerInit(ev: *Evented, worker: *Scheduler.Worker) !void {
    _ = ev;
    const kq_fd = try createFileDescriptor();
    errdefer closeFd(kq_fd);
    worker.backend = .{ .kq_fd = kq_fd };
    const change: posix.Kevent = .{
        .ident = wake_ident,
        .filter = filt_user,
        .flags = std.c.EV.ADD | std.c.EV.CLEAR,
        .fflags = 0,
        .data = 0,
        .udata = 0,
    };
    keventChange(kq_fd, &.{change});
}

/// For the scheduler: on the worker's own thread, which is where a thread's clock is handed out.
pub fn workerStart(ev: *Evented, worker: *Scheduler.Worker) void {
    _ = ev;
    worker.backend.cpu = scheduler.CpuTime.acquire();
}

/// For the scheduler: after the worker's thread has run its last task.
pub fn workerDeinit(ev: *Evented, worker: *Scheduler.Worker) void {
    _ = ev;
    assert(worker.backend.parked == null); // a task was never awaited
    scheduler.CpuTime.release(worker.backend.cpu);
    closeFd(worker.backend.kq_fd);
}

/// For the scheduler: how much CPU time the worker's thread has used, in nanoseconds, or `null`
/// when it cannot be read. The watchdog tells a worker that is blocked from one that is computing
/// with it. Darwin and FreeBSD hand a thread's clock out; the other BSDs have no way to read
/// another thread's CPU time, so a worker there is always taken to be blocked, which is what the
/// compare-exchange path assumes. See the backend contract.
pub fn threadCpuTime(ev: *Evented, worker: *Scheduler.Worker) ?u64 {
    _ = ev;
    return scheduler.CpuTime.read(worker.backend.cpu);
}

/// For the scheduler: whether this platform can issue a barrier that makes this thread's earlier
/// stores visible to every other thread of the process after their own barriers. Darwin and the
/// BSDs have no `membarrier`, so there is none, and every taker of a slot for the next task pays
/// a compare-exchange while the watchdog takes no slot. See the scheduler's backend contract.
pub fn heavyBarrier(ev: *Evented) bool {
    _ = ev;
    return false;
}

/// For the scheduler: cancels the operation `task` waits for. The token it registered is
/// `worker.index + token_base`, which names the worker whose kqueue and futexes it is parked on:
/// this takes the task out of them and leaves it the outcome a canceled operation has. An
/// operation that has already completed is left alone, and its task sees the request at its next
/// cancel point.
pub fn cancelOperation(ev: *Evented, from: *Scheduler.Worker, task: *Fiber, token: u31) void {
    const owner = ev.sched.workerAt(token - token_base) orelse return;
    const backend = &owner.backend;
    var make_runnable = false;
    backend.parked_lock.lock();
    if (findParked(backend, task)) |parked| {
        const completion = parked.resultPointer(Completion);
        if (completion.state.cmpxchgStrong(.waiting, .canceled, .seq_cst, .monotonic) == null) {
            // It has not parked yet, so it takes itself out and makes itself runnable as it
            // parks: see `markParked`. Taking it out here would leave it waiting for an event
            // that has been deleted.
            completion.outcome = .canceled;
            completion.state.store(.idle, .monotonic);
            removeParked(backend, parked);
            deleteRegistrations(backend.kq_fd, completion);
            handOverEvents(ev, backend, parked, null);
        } else {
            assert(completion.state.load(.monotonic) == .parked);
            unpark(ev, backend, parked, null, .canceled);
            make_runnable = true;
        }
    }
    backend.parked_lock.unlock();
    if (make_runnable) ev.sched.ready(from, task);
}

/// For the scheduler: makes `to`'s blocking `kevent` return. A kqueue's events may be changed
/// from any thread, so this and `wakeForeign` are the same call; what the scheduler tells them
/// apart by is which worker the wake is credited to.
pub fn wake(ev: *Evented, from: *Scheduler.Worker, to: *Scheduler.Worker) void {
    _ = ev;
    _ = from;
    trigger(to);
}

/// For the scheduler: `wake` from a thread that is not one of the workers: the watchdog, and the
/// pool the scheduler runs blocking calls on.
pub fn wakeForeign(ev: *Evented, to: *Scheduler.Worker) void {
    _ = ev;
    trigger(to);
}

fn trigger(to: *Scheduler.Worker) void {
    const change: posix.Kevent = .{
        .ident = wake_ident,
        .filter = filt_user,
        .flags = 0,
        .fflags = std.c.NOTE.TRIGGER,
        .data = 0,
        .udata = 0,
    };
    keventChange(to.backend.kq_fd, &.{change});
}

/// For the scheduler: handles the events that arrived on this worker's kqueue, and passes each
/// task whose operation is ready, or whose timer has run out, back to the scheduler. `.block`
/// first waits for at least one event.
pub fn poll(ev: *Evented, worker: *Scheduler.Worker, mode: scheduler.PollMode) void {
    const backend = &worker.backend;
    const n = switch (mode) {
        .nonblocking => keventWait(backend.kq_fd, backend.events[0..], .{ .sec = 0, .nsec = 0 }),
        .block => keventWait(backend.kq_fd, backend.events[0..], null),
    };
    for (backend.events[0..n]) |event| handleEvent(ev, worker, event);
}

/// One event: this worker's own wake, or an operation's.
fn handleEvent(ev: *Evented, worker: *Scheduler.Worker, event: posix.Kevent) void {
    if (event.filter == filt_user) return; // a wake, this worker's own or another's
    const task: *Fiber = @ptrFromInt(event.udata);
    const fired: Registration = .{ .ident = event.ident, .filter = event.filter };
    const outcome: Completion.Outcome = if (event.filter == filt_timer) .timeout else .ready;
    const backend = &worker.backend;
    var waking: [max_events]*Fiber = undefined;
    var waking_len: usize = 0;
    backend.parked_lock.lock();
    if (findParked(backend, task)) |parked| {
        waking[waking_len] = parked;
        waking_len += 1;
        unpark(ev, backend, parked, fired, outcome);
        // Every task that waits on the same event: the kernel delivered it once, and it is gone
        // from the kqueue now, so they all retry the call it was for. The first of them to find
        // nothing to read or write registers the event again.
        if (outcome == .ready) {
            var node = backend.parked;
            while (node) |other| : (node = other.resultPointer(Completion).next) {
                if (waking_len == waking.len) break;
                const other_completion = other.resultPointer(Completion);
                if (other_completion.state.load(.monotonic) != .parked) continue;
                if (!waitsOn(other_completion, fired)) continue;
                waking[waking_len] = other;
                waking_len += 1;
            }
            for (waking[1..waking_len]) |other| unpark(ev, backend, other, fired, outcome);
        }
    }
    backend.parked_lock.unlock();
    for (waking[0..waking_len]) |each| ev.sched.readyFromPoll(worker, each);
}

/// Takes `task` out of this worker's parked list and out of the events it owns, and leaves it
/// `outcome`. `fired` is the event that arrived, which is gone from the kqueue already; the
/// task's other events are deleted, and one of them, which another parked task was waiting on
/// with this one, is handed over to that task. Called with the parked lock held.
fn unpark(
    ev: *Evented,
    backend: *Worker,
    task: *Fiber,
    fired: ?Registration,
    outcome: Completion.Outcome,
) void {
    const completion = task.resultPointer(Completion);
    const state = completion.state.load(.monotonic);
    assert(state == .parked or state == .waiting);
    completion.state.store(.idle, .monotonic);
    completion.outcome = outcome;
    removeParked(backend, task);
    deleteRegistrations(backend.kq_fd, completion);
    handOverEvents(ev, backend, task, fired);
    completion.count = 0;
}

/// The task in this worker's parked list, if any. Walks the list rather than following the task
/// the event names: an event for a task that is not parked any more — a cancellation or a timer
/// completed it, or the registration outlived it — is dropped instead of followed.
fn findParked(backend: *Worker, task: *Fiber) ?*Fiber {
    var node = backend.parked;
    while (node) |t| : (node = t.resultPointer(Completion).next) {
        if (t == task) return t;
    }
    return null;
}

fn removeParked(backend: *Worker, task: *Fiber) void {
    var link = &backend.parked;
    while (link.*) |t| {
        if (t == task) {
            link.* = t.resultPointer(Completion).next;
            t.resultPointer(Completion).next = null;
            return;
        }
        link = &t.resultPointer(Completion).next;
    }
    unreachable; // it was found in the list
}

/// What a parking operation reports: its task was canceled, or its deadline passed. The
/// operations whose own error set carries `error.Timeout` take the second one from here.
pub const ParkError = error{ Canceled, Timeout };

/// The cancelation state of one operation of the calling task: `init` at the start, `deinit` at
/// the end, and `check` before every system call the operation makes, which is where a cancel
/// request reaches it. The `park*` functions register what the operation waits for themselves.
const CancelRegion = struct {
    task: *Fiber,
    status: Fiber.CancelStatus,

    fn init() CancelRegion {
        const task = Scheduler.Worker.current().currentTask();
        return .{
            .task = task,
            .status = .{
                .requested = task.cancel_protection.check() == .unblocked,
                .awaiting = .nothing,
            },
        };
    }

    fn initBlocked() CancelRegion {
        return .{
            .task = Scheduler.Worker.current().currentTask(),
            .status = .{ .requested = false, .awaiting = .nothing },
        };
    }

    fn deinit(region: *CancelRegion) void {
        if (region.status.requested) {
            @branchHint(.likely);
            _ = region.task.cancel_status.changeAwaiting(region.status.awaiting, .nothing);
        }
        region.* = undefined;
    }

    /// The cancel point of an operation: returns `error.Canceled` when a request is pending, and
    /// leaves the task awaiting nothing.
    fn check(region: *CancelRegion) Io.Cancelable!void {
        return region.await(.nothing);
    }

    /// Arms the operation before the task parks: from here a cancel request reaches the task
    /// through `worker`'s kqueue, whose index is the token it parks under.
    fn arm(region: *CancelRegion, worker: *Scheduler.Worker) Io.Cancelable!void {
        return region.await(.fromToken(@intCast(worker.index + token_base)));
    }

    fn await(region: *CancelRegion, awaiting: Fiber.CancelStatus.Awaiting) Io.Cancelable!void {
        if (!region.status.requested) {
            @branchHint(.unlikely);
            return;
        }
        const status: Fiber.CancelStatus = .{ .requested = true, .awaiting = awaiting };
        if (region.task.cancel_status.changeAwaiting(region.status.awaiting, status.awaiting)) {
            @branchHint(.unlikely);
            // The request arrived first, so nothing is pending that it could cancel. Leave the
            // task awaiting nothing, since `deinit` does not run for an unrequested region.
            _ = region.task.cancel_status.changeAwaiting(status.awaiting, .nothing);
            region.task.cancel_protection.acknowledge();
            region.status = .unrequested;
            return error.Canceled;
        }
        region.status = status;
    }
};

/// The token a parked task registers with the scheduler is its worker's index plus this, so that
/// it names neither of the two tokens the scheduler reserves. See `Fiber.CancelStatus.Awaiting`.
const token_base: u31 = 2;

/// What a parked task left in the kernel, and how its wait ended.
pub const Completion = struct {
    /// How the wait ended. `none` while the task is parked.
    outcome: Outcome = .none,
    /// The events the task added to the worker's kqueue, which a completion and a cancellation
    /// delete so that none of them fires later for a task that is running again. A task that
    /// waits on an event another parked task added has it here too, with `owns` clear, and does
    /// not delete it.
    registrations: [max_registrations]Registration = @splat(.{}),
    /// How many of `registrations` are in use.
    count: u8 = 0,
    /// The park, a completion and a cancellation exchange this word, so that exactly one of them
    /// makes the task runnable. See `park`.
    state: std.atomic.Value(State) = .init(.idle),
    /// The next task in its worker's list of parked tasks. See `Worker.parked`.
    next: ?*Fiber = null,
    /// For a futex wait: the bucket the task is waiting in, and its node there, so that a wake
    /// and a cancellation take it out.
    futex_bucket: ?*FutexBucket = null,
    futex_node: ?*FutexWaiter = null,
    /// The kqueue the task registered its events on. A wake that does not come from an event —
    /// a futex table's, which no worker completes — leaves whatever did not fire to the task
    /// itself, and this is where they are.
    kq_fd: fd_t = -1,

    pub const Outcome = enum(u8) { none, ready, timeout, canceled };
    pub const State = enum(u8) { idle, waiting, parked, canceled };
};

/// What a task asked the kernel to watch: an ident and a filter, and for a timer how long it
/// waits. Deleting one needs the ident and the filter; a break of the wait needs the data too.
pub const Registration = struct {
    ident: Ident = 0,
    filter: Filter = 0,
    /// For a timer: how long it waits, in the platform's unit for one.
    fflags: u32 = 0,
    data: Data = 0,
    /// Whether this task's `EV.ADD` is the one the kernel holds. A task that waits on an event
    /// another parked task already added does not add it again, because `EV.ADD` for the same
    /// pair replaces the event and its `udata` with it; it is completed when that event is
    /// delivered, and it only deletes the pairs it added itself.
    owns: bool = false,
};

/// Parks the calling task: registers `registrations` on its worker's kqueue, switches away, and
/// returns how the wait ended. Only a worker completes a parked task, by the event it registered
/// or by a cancellation, so the task is on no run queue until then. `futex` is set for a wait in
/// the futex table, which is not a kernel event. A `region` makes the wait cancelable.
fn park(
    ev: *Evented,
    region: ?*CancelRegion,
    registrations: []const Registration,
    futex: ?FutexPark,
) Io.Cancelable!Completion.Outcome {
    const s = &ev.sched;
    const w = Scheduler.Worker.current();
    const task = w.currentTask();
    const completion = task.resultPointer(Completion);
    if (completion.state.load(.monotonic) != .idle) {
        std.debug.panic("park: task {d} state {t} outcome {t} count {d}", .{
            task.id,
            completion.state.load(.monotonic),
            completion.outcome,
            completion.count,
        });
    }
    assert(registrations.len <= max_registrations);
    completion.outcome = .none;
    completion.count = @intCast(registrations.len);
    @memcpy(completion.registrations[0..registrations.len], registrations);
    completion.futex_bucket = if (futex) |f| f.bucket else null;
    completion.futex_node = if (futex) |f| f.node else null;
    completion.kq_fd = w.backend.kq_fd;
    {
        // Publishing is one step, under the lock: a cancellation must not be able to take the
        // task out of the list before it is in it, nor leave a kernel event behind.
        w.backend.parked_lock.lock();
        defer w.backend.parked_lock.unlock();
        completion.state.store(.waiting, .monotonic);
        completion.next = w.backend.parked;
        w.backend.parked = task;
        register(ev, &w.backend, task);
    }
    if (region) |r| r.arm(w) catch |err| {
        assert(err == error.Canceled);
        // Nothing is armed, so nothing can reach this task any more: it takes itself out.
        w.backend.parked_lock.lock();
        defer w.backend.parked_lock.unlock();
        unpark(ev, &w.backend, task, null, .canceled);
        return error.Canceled;
    };
    s.yield(null, .{ .custom = .{ .context = completion, .run = markParked } });
    // Running again. A worker that completed the task took its events out already; a wake that
    // came through the futex table did not, because no worker is involved in it, so the events
    // that did not fire come out here, on whichever worker the task resumed on: any thread may
    // change any kqueue.
    deleteRegistrations(completion.kq_fd, completion);
    completion.kq_fd = -1;
    completion.futex_bucket = null;
    completion.futex_node = null;
    completion.state.store(.idle, .monotonic);
    switch (completion.outcome) {
        .ready => return .ready,
        .timeout => return .timeout,
        .canceled, .none => return error.Canceled,
    }
}

/// Runs once a parking task has switched away, which is the point from which a completion may
/// make it runnable: the exchange here and the one a cancellation makes are what keep a task
/// from ever being runnable twice.
fn markParked(s: *Sched, task: *Fiber, context: *anyopaque) void {
    const completion: *Completion = @ptrCast(@alignCast(context));
    if (completion.state.cmpxchgStrong(.waiting, .parked, .seq_cst, .monotonic) == null) return;
    // A cancellation got there first, and left this side of the exchange the wake: it cannot
    // make a task runnable that has not finished switching out.
    assert(completion.state.load(.monotonic) == .canceled);
    s.ready(.current(), task);
}

const Sched = Scheduler;

/// A futex wait, which is not a kernel event: the table it is in, and its node there.
const FutexPark = struct {
    bucket: *FutexBucket,
    node: *FutexWaiter,
};

/// Adds the events a parking task needs to the worker's kqueue, unless another parked task is
/// waiting on the same one already. Called with the parked lock held, so that a cancellation of
/// a task that is being published cannot delete an event before it is added.
fn register(ev: *Evented, backend: *Worker, task: *Fiber) void {
    _ = ev;
    const completion = task.resultPointer(Completion);
    var changes: [max_registrations]posix.Kevent = undefined;
    var count: usize = 0;
    for (completion.registrations[0..completion.count]) |*reg| {
        if (heldByOther(backend, task, reg.*)) continue;
        reg.owns = true;
        changes[count] = .{
            .ident = reg.ident,
            .filter = reg.filter,
            .flags = register_flags,
            .fflags = reg.fflags,
            .data = reg.data,
            .udata = @intFromPtr(task),
        };
        count += 1;
    }
    if (count > 0) keventChange(backend.kq_fd, changes[0..count]);
}

/// Whether a parked task other than `task` is waiting on this event already.
fn heldByOther(backend: *Worker, task: *Fiber, reg: Registration) bool {
    var node = backend.parked;
    while (node) |other| : (node = other.resultPointer(Completion).next) {
        if (other == task) continue;
        const other_completion = other.resultPointer(Completion);
        if (other_completion.state.load(.monotonic) != .parked) continue;
        if (waitsOn(other_completion, reg)) return true;
    }
    return false;
}

/// Gives the events a departing task owned to another parked task that was waiting on them with
/// it, or deletes them. `fired` is the event that arrived, if one did: it is gone from the
/// kqueue already and is not deleted, and the tasks that waited on it with this one are being
/// completed with it. Called with the parked lock held.
fn handOverEvents(
    ev: *Evented,
    backend: *Worker,
    task: *Fiber,
    fired: ?Registration,
) void {
    _ = ev;
    const completion = task.resultPointer(Completion);
    for (completion.registrations[0..completion.count]) |reg| {
        if (!reg.owns) continue;
        if (fired) |f| {
            if (f.ident == reg.ident and f.filter == reg.filter) continue;
        }
        const next = otherWaitingOn(backend, task, reg) orelse continue;
        const next_completion = next.resultPointer(Completion);
        next_completion.registrations[regIndex(next_completion, reg).?].owns = true;
        const change: posix.Kevent = .{
            .ident = reg.ident,
            .filter = reg.filter,
            .flags = register_flags,
            .fflags = reg.fflags,
            .data = reg.data,
            .udata = @intFromPtr(next),
        };
        keventChange(backend.kq_fd, &.{change});
    }
}

fn regIndex(completion: *Completion, reg: Registration) ?usize {
    for (completion.registrations[0..completion.count], 0..) |each, i| {
        if (each.ident == reg.ident and each.filter == reg.filter) return i;
    }
    return null;
}

/// A parked task other than the departing one that waits on this event.
fn otherWaitingOn(backend: *Worker, task: *Fiber, reg: Registration) ?*Fiber {
    var node = backend.parked;
    while (node) |other| : (node = other.resultPointer(Completion).next) {
        if (other == task) continue;
        const other_completion = other.resultPointer(Completion);
        if (other_completion.state.load(.monotonic) != .parked) continue;
        if (waitsOn(other_completion, reg)) return other;
    }
    return null;
}

/// Whether this completion has the event, whether it added it or not.
fn waitsOn(completion: *Completion, reg: Registration) bool {
    for (completion.registrations[0..completion.count]) |each| {
        if (each.ident == reg.ident and each.filter == reg.filter) return true;
    }
    return false;
}

/// Deletes the events a task added, leaving the ones other tasks are waiting on with it.
fn deleteRegistrations(kq_fd: posix.fd_t, completion: *Completion) void {
    var changes: [max_registrations]posix.Kevent = undefined;
    var count: usize = 0;
    for (completion.registrations[0..completion.count]) |reg| {
        if (!reg.owns) continue;
        changes[count] = .{
            .ident = reg.ident,
            .filter = reg.filter,
            .flags = std.c.EV.DELETE,
            .fflags = 0,
            .data = 0,
            .udata = 0,
        };
        count += 1;
    }
    if (count > 0) keventChange(kq_fd, changes[0..count]);
    completion.count = 0;
}

/// Parks the calling task until `fd` is ready for `filter` (`filt_read` or `filt_write`), until
/// `timeout` passes, or until the operation is canceled. On return the descriptor is ready and
/// the operation retries its call, which may still find nothing to do, in which case it parks
/// again.
fn parkFd(
    ev: *Evented,
    region: *CancelRegion,
    fd: posix.fd_t,
    filter: Filter,
    timeout: Io.Timeout,
) ParkError!void {
    var registrations: [max_registrations]Registration = undefined;
    const count = parkRegistrations(ev, @intCast(fd), filter, timeout, &registrations);
    switch (try park(ev, region, registrations[0..count], null)) {
        .ready => return,
        .timeout => return error.Timeout,
        .canceled, .none => return error.Canceled,
    }
}

/// Parks the calling task until `pid` exits or the timeout passes.
fn parkChild(
    ev: *Evented,
    region: *CancelRegion,
    pid: pid_t,
    timeout: Io.Timeout,
) ParkError!void {
    var registrations: [max_registrations]Registration = undefined;
    const count = parkRegistrations(ev, @intCast(pid), filt_proc, timeout, &registrations);
    switch (try park(ev, region, registrations[0..count], null)) {
        .ready => return,
        .timeout => return error.Timeout,
        .canceled, .none => return error.Canceled,
    }
}

/// Parks the calling task for `timeout`, which may be a duration or a deadline.
fn parkTimeout(ev: *Evented, region: *CancelRegion, timeout: Io.Timeout) Io.Cancelable!void {
    var registrations: [max_registrations]Registration = undefined;
    const count = parkRegistrations(ev, 0, 0, timeout, &registrations);
    assert(count == 1); // a timeout has a deadline
    switch (try park(ev, region, registrations[0..count], null)) {
        .timeout => return,
        .ready => unreachable, // only a timer was registered
        .canceled, .none => return error.Canceled,
    }
}

/// The events for an operation that waits on `ident` with `filter` and has `timeout`: the event
/// itself, and the timer that ends the wait when the timeout passes. A `timeout` of `.none` has
/// no timer, and a deadline that has passed has no event: the wait is over.
fn parkRegistrations(
    ev: *Evented,
    ident: Ident,
    filter: Filter,
    timeout: Io.Timeout,
    out: *[max_registrations]Registration,
) usize {
    var count: usize = 0;
    const duration: ?u64 = if (timeout.toDurationFromNow(ev.io())) |duration|
        @intCast(@max(0, duration.raw.toNanoseconds()))
    else
        null;
    if (filter != 0 and duration != 0) {
        out[count] = .{ .ident = ident, .filter = filter };
        count += 1;
    }
    if (duration) |ns| {
        if (ns > 0) {
            const w = Scheduler.Worker.current();
            const unit = timerUnit(ns);
            out[count] = .{
                .ident = w.backend.next_timer,
                .filter = filt_timer,
                .fflags = unit.fflags,
                .data = unit.data,
            };
            w.backend.next_timer += 1;
            count += 1;
        }
    }
    return count;
}

/// A timer's data and flags for a duration in nanoseconds: nanoseconds where the platform has a
/// flag for them, and milliseconds, the default unit, on OpenBSD and DragonFly, which do not.
fn timerUnit(ns: u64) struct { data: Data, fflags: u32 } {
    if (@hasDecl(std.c.NOTE, "NSECONDS")) {
        return .{ .data = @intCast(ns), .fflags = std.c.NOTE.NSECONDS };
    }
    return .{
        .data = @intCast(@max(1, (ns + std.time.ns_per_ms - 1) / std.time.ns_per_ms)),
        .fflags = 0,
    };
}

// Futexes

/// One bucket of the table the tasks waiting in `futexWait` are hashed into. The lock is held
/// while the bucket's list is walked or changed, and a task's node is on its own stack, so a
/// wake, a cancellation and a timer take the same node out under the same lock.
const FutexBucket = struct {
    lock: Scheduler.SpinLock = .{},
    head: ?*FutexWaiter = null,
};

/// One task waiting in `futexWait`: the address it waits on, and its link in the bucket.
const FutexWaiter = struct {
    next: ?*FutexWaiter,
    ptr: *const u32,
    task: *Fiber,
    /// Cleared, under the bucket's lock, by whoever takes this node out of the bucket.
    linked: bool = true,
};

/// The bucket for an address. Fibonacci hashing: the high bits of the golden-ratio multiple
/// spread the addresses of a slab better than the low ones, which vary least.
fn futexBucket(ev: *Evented, ptr: *const u32) *FutexBucket {
    const fibonacci_multiplier = 0x9E3779B97F4A7C15 >> (64 - @bitSizeOf(usize));
    const hashed = @intFromPtr(ptr) *% fibonacci_multiplier;
    comptime assert(std.math.isPowerOfTwo(futex_buckets_len));
    const index = hashed >> (@bitSizeOf(usize) - @as(usize, futex_bucket_bits));
    return &ev.futex_buckets[index];
}

/// Takes a node out of its bucket, under the bucket's lock and if it is still there.
fn futexUnlink(bucket: *FutexBucket, waiter: *FutexWaiter) void {
    if (!waiter.linked) return;
    var link = &bucket.head;
    while (link.*) |other| {
        if (other == waiter) {
            link.* = other.next;
            waiter.linked = false;
            return;
        }
        link = &other.next;
    }
    unreachable; // it was linked
}

fn futexWait(
    userdata: ?*anyopaque,
    ptr: *const u32,
    expected: u32,
    timeout: Io.Timeout,
) Io.Cancelable!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (Scheduler.Worker.currentOrNull() == null) {
        // A thread that is not one of the workers, the watchdog among them: it blocks in the
        // kernel, where a wake of the kernel reaches it.
        const timeout_ns: ?u64 = if (timeout.toDurationFromNow(ev.io())) |duration|
            @intCast(@max(0, duration.raw.toNanoseconds()))
        else
            null;
        scheduler.Futex.wait(ptr, expected, timeout_ns);
        return;
    }
    const w = Scheduler.Worker.current();
    const bucket = futexBucket(ev, ptr);
    var waiter: FutexWaiter = .{
        .next = null,
        .ptr = ptr,
        .task = w.currentTask(),
    };
    var registrations: [max_registrations]Registration = undefined;
    const count = parkRegistrations(ev, 0, 0, timeout, &registrations);
    var region: CancelRegion = .init();
    defer region.deinit();
    {
        bucket.lock.lock();
        defer bucket.lock.unlock();
        if (@atomicLoad(u32, ptr, .monotonic) != expected) return;
        waiter.next = bucket.head;
        bucket.head = &waiter;
    }
    const outcome = park(ev, &region, registrations[0..count], .{ .bucket = bucket, .node = &waiter }) catch |err| switch (err) {
        error.Canceled => |e| {
            bucket.lock.lock();
            futexUnlink(bucket, &waiter);
            bucket.lock.unlock();
            return e;
        },
    };
    bucket.lock.lock();
    futexUnlink(bucket, &waiter);
    bucket.lock.unlock();
    // A timeout is a return like a wake: the caller re-reads the word either way, and it is the
    // caller's business which of the two happened.
    assert(outcome != .canceled);
}

fn futexWaitUncancelable(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const w = Scheduler.Worker.current();
    const bucket = futexBucket(ev, ptr);
    var waiter: FutexWaiter = .{
        .next = null,
        .ptr = ptr,
        .task = w.currentTask(),
    };
    {
        bucket.lock.lock();
        defer bucket.lock.unlock();
        if (@atomicLoad(u32, ptr, .monotonic) != expected) return;
        waiter.next = bucket.head;
        bucket.head = &waiter;
    }
    const outcome = park(ev, null, &.{}, .{ .bucket = bucket, .node = &waiter }) catch unreachable;
    assert(outcome != .timeout);
    if (outcome == .canceled) unreachable; // not cancelable
    // A wake took the node out already, or left it to this side; either way it is out now.
    bucket.lock.lock();
    futexUnlink(bucket, &waiter);
    bucket.lock.unlock();
}

fn futexWake(userdata: ?*anyopaque, ptr: *const u32, max_waiters: u32) void {
    charge(userdata);
    if (max_waiters == 0) return;
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const bucket = futexBucket(ev, ptr);
    var woken: ?*FutexWaiter = null;
    {
        bucket.lock.lock();
        defer bucket.lock.unlock();
        var n: u32 = 0;
        var link = &bucket.head;
        while (link.*) |waiter| {
            if (n == max_waiters) break;
            if (waiter.ptr != ptr) {
                link = &waiter.next;
                continue;
            }
            link.* = waiter.next;
            waiter.linked = false;
            waiter.next = woken;
            woken = waiter;
            n += 1;
        }
    }
    // A wake of the kernel for the waiters this core does not park: a thread that is not one of
    // the workers waits on the futex itself.
    scheduler.Futex.wake(ptr, max_waiters);
    const from = Scheduler.Worker.currentOrNull();
    while (woken) |waiter| {
        woken = waiter.next;
        if (from) |w| ev.sched.ready(w, waiter.task) else ev.sched.readyFromForeign(waiter.task);
    }
}

// Time

/// Parks the calling task for `timeout`, which is what `Io.sleep` is.
fn sleep(userdata: ?*anyopaque, timeout: Io.Timeout) Io.Cancelable!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    if (Scheduler.Worker.currentOrNull() == null) {
        // A thread that is not one of the workers has no task to park: it sleeps in the kernel.
        const timeout_ns: ?u64 = if (timeout.toDurationFromNow(ev.io())) |duration|
            @intCast(@max(0, duration.raw.toNanoseconds()))
        else
            null;
        const ns = timeout_ns orelse return; // sleeping for no time at all
        var ts: posix.timespec = .{
            .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
            .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
        };
        while (posix.errno(posix.system.nanosleep(&ts, &ts)) == .INTR) {}
        return;
    }
    var region: CancelRegion = .init();
    defer region.deinit();
    return parkTimeout(ev, &region, timeout);
}

/// The clock's resolution, from the platform.
fn clockResolution(userdata: ?*anyopaque, clock: Io.Clock) Io.Clock.ResolutionError!Io.Duration {
    charge(userdata);
    var timespec: posix.timespec = undefined;
    return switch (posix.errno(posix.system.clock_getres(clockToPosix(clock), &timespec))) {
        .SUCCESS => .fromNanoseconds(nanosecondsFromPosix(&timespec)),
        .INVAL => return error.ClockUnavailable,
        else => |err| return unexpectedErrno(err),
    };
}

/// The clock's reading, from the platform. Any thread may call this, the watchdog among them.
fn now(userdata: ?*anyopaque, clock: Io.Clock) Io.Timestamp {
    _ = userdata;
    var tp: posix.timespec = undefined;
    switch (posix.errno(posix.system.clock_gettime(clockToPosix(clock), &tp))) {
        .SUCCESS => return timestampFromPosix(&tp),
        else => return .zero,
    }
}

// Children

/// Waits for a child to exit: an `EVFILT.PROC` event for its process, and `wait4` for the status,
/// which is the call the event makes worth a second try.
fn childWait(userdata: ?*anyopaque, child: *process.Child) process.Child.WaitError!process.Child.Term {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    defer childCleanup(userdata, child);
    const pid: pid_t = @intCast(child.id.?);
    var ru: posix.rusage = undefined;
    const ru_ptr: ?*posix.rusage = if (child.request_resource_usage_statistics) &ru else null;
    while (true) {
        try region.check();
        var status: c_int = undefined;
        const rc = posix.system.wait4(pid, &status, posix.W.NOHANG, ru_ptr);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) {
                    // Still running: wait for its process to exit, then ask again.
                    parkChild(ev, &region, pid, .none) catch |err| switch (err) {
                        error.Timeout => unreachable, // no timeout
                        error.Canceled => |e| return e,
                    };
                    continue;
                }
                if (ru_ptr) |p| child.resource_usage_statistics.rusage = p.*;
                return statusToTerm(@bitCast(status));
            },
            .INTR => continue,
            .CHILD => |err| return errnoBug(err), // reaped elsewhere
            .ACCES => return error.AccessDenied,
            .FAULT, .INVAL, .NOMEM => |err| return errnoBug(err),
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Kills a child, and waits for it: the process is gone by the time this returns.
fn childKill(userdata: ?*anyopaque, child: *process.Child) void {
    charge(userdata);
    const pid: pid_t = @intCast(child.id orelse return);
    while (true) switch (posix.errno(posix.system.kill(pid, posix.SIG.KILL))) {
        .SUCCESS => break,
        .INTR => continue,
        else => {
            childCleanup(@ptrCast(@alignCast(userdata)), child);
            return;
        },
    };
    var status: c_int = 0;
    while (true) switch (posix.errno(posix.system.waitpid(pid, &status, 0))) {
        .SUCCESS => break,
        .INTR => continue,
        else => break,
    };
    childCleanup(@ptrCast(@alignCast(userdata)), child);
}

/// `fsync` is the one regular-file call that can take long enough to matter: it goes to the
/// scheduler's pool, where the worker runs other tasks meanwhile.
fn fileSync(userdata: ?*anyopaque, file: File) File.SyncError!void {
    charge(userdata);
    const Sync = struct {
        fd: fd_t,

        fn start(context: *const anyopaque, result: *anyopaque) void {
            const fd: fd_t = @as(*const fd_t, @ptrCast(@alignCast(context))).*;
            const err: *?File.SyncError = @ptrCast(@alignCast(result));
            err.* = sync(fd);
        }

        fn sync(fd: fd_t) ?File.SyncError {
            while (true) switch (posix.errno(posix.system.fsync(fd))) {
                .SUCCESS => return null,
                .INTR => continue,
                .BADF => |err| return errnoBug(err), // File descriptor used after closed.
                .INVAL => |err| return errnoBug(err),
                .ROFS => |err| return errnoBug(err),
                .IO => return error.InputOutput,
                .NOSPC => return error.NoSpaceLeft,
                .DQUOT => return error.DiskQuota,
                else => |err| return unexpectedErrno(err),
            };
        }
    };
    var context: fd_t = file.handle;
    var err: ?File.SyncError = null;
    Scheduler.blocking(
        userdata,
        std.mem.asBytes(&err),
        .of(?File.SyncError),
        std.mem.asBytes(&context),
        .of(fd_t),
        "fsync",
        Sync.start,
    );
    if (err) |e| return e;
}

// Random

/// `random`: a `Csprng` per worker, seeded from `randomSecure` when it starts, and reseeded from
/// the instance's own generator when a new worker starts.
fn random(userdata: ?*anyopaque, buffer: []u8) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var thread: *Thread = currentWorker();
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
                randomSecure(userdata, &seed) catch |err| switch (err) {
                    error.Canceled => unreachable, // blocked
                    error.EntropyUnavailable => fallbackSeed(ev, &seed),
                };
                ev.csprng.rng = .init(seed);
                thread = currentWorker();
            }
            ev.csprng.rng.fill(&seed);
        }
        thread.csprng.rng = .init(seed);
    }
    thread.csprng.rng.fill(buffer);
}

/// `randomSecure`: the system's entropy. Any thread may call this.
fn randomSecure(userdata: ?*anyopaque, buffer: []u8) Io.RandomSecureError!void {
    _ = userdata;
    if (buffer.len == 0) return;
    switch (builtin.os.tag) {
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos, .freebsd, .netbsd, .openbsd, .dragonfly => c.arc4random_buf(buffer.ptr, buffer.len),
        else => @compileError("an entropy source for " ++ @tagName(builtin.os.tag) ++ " is not implemented"),
    }
}

/// Charges one operation to the calling task: see the scheduler's `charge`.
inline fn charge(userdata: ?*anyopaque) void {
    Scheduler.charged(userdata);
}

/// Closes a descriptor this instance owns, with the error thrown away: there is nothing left to
/// do about it here.
fn closeAsync(ev: *Evented, fd: fd_t) void {
    _ = ev;
    switch (posix.errno(posix.system.close(fd))) {
        .SUCCESS, .BADF, .INTR => {},
        else => {},
    }
}

/// Adds or deletes the events in `changes` on a kqueue. A change the kernel refuses — a
/// descriptor that was closed meanwhile, an event that is gone already — is left for the
/// operation's own system call to report, or to ignore, so the result is not read.
fn keventChange(kq_fd: posix.fd_t, changes: []const posix.Kevent) void {
    while (true) {
        const rc = posix.system.kevent(
            kq_fd,
            changes.ptr,
            @intCast(changes.len),
            undefined,
            0,
            &zero_timeout,
        );
        switch (posix.errno(rc)) {
            .SUCCESS, .NOENT, .BADF, .FAULT, .INVAL, .NOMEM, .ACCES, .SRCH => return,
            .INTR => continue,
            else => |err| {
                recoverableOsBugDetected();
                log.warn("unexpected kevent change error: {t}", .{err});
                return;
            },
        }
    }
}

/// Waits for events on a kqueue, up to `events.len` of them, for at most `timeout` (`null` means
/// until one arrives). Returns how many arrived.
fn keventWait(kq_fd: posix.fd_t, events: []posix.Kevent, timeout: ?posix.timespec) usize {
    while (true) {
        const rc = posix.system.kevent(
            kq_fd,
            undefined,
            0,
            events.ptr,
            @intCast(events.len),
            if (timeout) |*ts| ts else null,
        );
        switch (posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => continue, // a signal: look again, the timeout stands
            .BADF => unreachable, // the kqueue is this worker's own
            .FAULT => unreachable, // the buffers are ours
            .INVAL => unreachable,
            else => |err| {
                recoverableOsBugDetected();
                log.warn("unexpected kevent error: {t}", .{err});
                return 0;
            },
        }
    }
}

var zero_timeout: posix.timespec = .{ .sec = 0, .nsec = 0 };

// The instance's allocator, its public surface, and the entries whose work is one call on the
// worker: the executable's path, the working directory, the standard error writer, the terminal,
// the random device, and the `kevent` `Maker.Watch` uses.

/// `Maker.Watch` uses this to watch a directory tree on the BSDs, and `cli` uses it to wait for
/// its inputs: the raw `kevent` call, with this core's error mapping.
pub const CreateFileDescriptorError = error{
    /// The per-process limit on the number of open file descriptors has been reached.
    ProcessFdQuotaExceeded,
    /// The system-wide limit on the total number of open files has been reached.
    SystemFdQuotaExceeded,
} || Io.UnexpectedError;

pub fn createFileDescriptor() CreateFileDescriptorError!posix.fd_t {
    const rc = posix.system.kqueue();
    switch (posix.errno(rc)) {
        .SUCCESS => return @intCast(rc),
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NFILE => return error.SystemFdQuotaExceeded,
        else => |err| return posix.unexpectedErrno(err),
    }
}

pub const KEventError = error{
    /// The process does not have permission to register a filter.
    AccessDenied,
    /// The event could not be found to be modified or deleted.
    EventNotFound,
    /// No memory was available to register the event.
    SystemResources,
    /// The specified process to attach to does not exist.
    ProcessNotFound,
    /// `changelist` or `eventlist` was too long.
    Overflow,
};

pub fn kevent(
    kq: posix.fd_t,
    changelist: []const posix.Kevent,
    eventlist: []posix.Kevent,
    timeout: ?*const posix.timespec,
) KEventError!usize {
    while (true) {
        const rc = posix.system.kevent(
            kq,
            changelist.ptr,
            std.math.cast(c_int, changelist.len) orelse return error.Overflow,
            eventlist.ptr,
            std.math.cast(c_int, eventlist.len) orelse return error.Overflow,
            timeout,
        );
        switch (posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .ACCES => return error.AccessDenied,
            .FAULT => unreachable,
            .BADF => unreachable, // Always a race condition: the kqueue was closed.
            .INTR => continue,
            .INVAL => unreachable,
            .NOENT => return error.EventNotFound,
            .NOMEM => return error.SystemResources,
            .SRCH => return error.ProcessNotFound,
            else => unreachable,
        }
    }
}

pub fn allocator(ev: *Evented) Allocator {
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

fn alloc(userdata: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    ev.backing_allocator_mutex.lockUncancelable(ev_io);
    defer ev.backing_allocator_mutex.unlock(ev_io);
    return ev.backing_allocator.rawAlloc(len, alignment, ret_addr);
}

fn resize(
    userdata: *anyopaque,
    memory: []u8,
    alignment: Alignment,
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

fn free(userdata: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const ev_io = ev.io();
    ev.backing_allocator_mutex.lockUncancelable(ev_io);
    defer ev.backing_allocator_mutex.unlock(ev_io);
    return ev.backing_allocator.rawFree(memory, alignment, ret_addr);
}

pub const InitOptions = struct {
    /// Whether the backing allocator this instance is given needs a lock around it: it is used
    /// from every worker and from the pool.
    backing_allocator_needs_mutex: bool = true,

    /// The most workers, not counting the thread that calls `init` as worker 0. `null` means one
    /// per CPU, less one.
    thread_limit: ?usize = null,

    /// The stack reserved for each task, unless its spawn says otherwise. The OS commits the
    /// pages as the task touches them, and a guard page below the stack catches an overflow.
    stack_size: usize = scheduler.default_stack_size,

    /// Affects the following operations:
    /// * `processExecutablePath` on OpenBSD.
    argv0: Argv0 = .empty,
    /// Affects the following operations:
    /// * `fileIsTty`
    /// * `processSpawn`, `processSpawnPath`, `processReplace`, `processReplacePath`
    environ: process.Environ = .empty,
};

/// Makes the calling thread the first worker, running as the main task. `ev` must stay put until
/// `deinit`, which must be called from the main task.
pub fn init(ev: *Evented, backing_allocator: Allocator, options: InitOptions) !void {
    ev.* = .{
        .backing_allocator = backing_allocator,
        .backing_allocator_needs_mutex = options.backing_allocator_needs_mutex,
        .backing_allocator_mutex = .init,
        .sched = undefined,
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
        .csprng_mutex = .init,
        .csprng = .uninitialized,
        .argv0 = options.argv0,
    };
    errdefer ev.* = undefined;
    try ev.sched.init(backing_allocator, .{
        .workers = if (options.thread_limit) |thread_limit| 1 + thread_limit else null,
        .stack_size = options.stack_size,
    });
}

/// Called from the main task once every other task has been awaited. Returns on the thread that
/// called `init`.
pub fn deinit(ev: *Evented) void {
    ev.sched.deinit(ev.backing_allocator);
    if (ev.dev_null_file) |file| fileClose(ev, &.{file});
    ev.* = undefined;
}

/// The number of `iovec`s a message's header takes: an `int` on Darwin and the BSDs, a `u32` on
/// Linux, which is what `c.msghdr_const` says on each.
const iovlen_t = @FieldType(c.msghdr_const, "iovlen");

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

/// What a program can read about this instance: see `Scheduler.Stats`, which is this type.
pub const Stats = Scheduler.Stats;

pub fn stats(ev: *Evented) Stats {
    return ev.sched.stats();
}

/// The `Evented` behind `any_io`, or `null` if `any_io` is another `Io` implementation.
pub fn fromIo(any_io: Io) ?*Evented {
    return if (any_io.vtable.operate == &operate) @ptrCast(@alignCast(any_io.userdata)) else null;
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
            .blocking = Scheduler.blocking,

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

fn openDevNullFile(ev: *Evented) File.OpenError!File {
    ev.dev_null_lock.lock();
    defer ev.dev_null_lock.unlock();
    if (ev.dev_null_file) |file| return file;
    const file = try dirOpenFile(ev, .cwd(), "/dev/null", .{ .mode = .read_write });
    ev.dev_null_file = file;
    return file;
}

/// The environment, read once, for `PATH` and for the variables `Io.Terminal.Mode` decides by.
fn scanEnviron(ev: *Evented) Io.Cancelable!void {
    const ev_io = ev.io();
    try ev.environ_mutex.lock(ev_io);
    defer ev.environ_mutex.unlock(ev_io);
    if (ev.environ_initialized) return;
    ev.environ.scan(ev.allocator());
    ev.environ_initialized = true;
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
        const cancel_protection = Scheduler.swapCancelProtection(ev, .blocked);
        defer assert(Scheduler.swapCancelProtection(ev, cancel_protection) == .blocked);
        ev.scanEnviron() catch |err| switch (err) {
            error.Canceled => unreachable, // blocked
        };
        const NO_COLOR = ev.environ.exist.NO_COLOR;
        const CLICOLOR_FORCE = ev.environ.exist.CLICOLOR_FORCE;
        ev.stderr_mode = Io.Terminal.Mode.detect(
            ev.io(),
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
            error.Canceled => Scheduler.recancel(userdata),
            else => {},
        }
        ev.stderr_writer.err = null;
    }
    ev.stderr_writer.interface.end = 0;
    ev.stderr_writer.interface.buffer = &.{};
    ev.stderr_mutex.unlock(ev.io());
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

/// Whether a descriptor is a terminal. A terminal has no kqueue event this core needs, and the
/// call itself does not block. Everything that is not a terminal, a closed descriptor among it,
/// is simply not one.
fn fileIsTty(userdata: ?*anyopaque, file: File) Io.Cancelable!bool {
    charge(userdata);
    while (true) {
        if (c.isatty(file.handle) == 1) return true;
        switch (posix.errno(c._errno().*)) {
            .INTR => continue,
            else => return false,
        }
    }
}

fn fileEnableAnsiEscapeCodes(userdata: ?*anyopaque, file: File) File.EnableAnsiEscapeCodesError!void {
    if (!try fileIsTty(userdata, file)) return error.NotTerminalDevice;
}

fn childCleanup(userdata: ?*anyopaque, child: *process.Child) void {
    if (child.stdin) |stdin| {
        fileClose(userdata, &.{stdin});
        child.stdin = null;
    }
    if (child.stdout) |stdout| {
        fileClose(userdata, &.{stdout});
        child.stdout = null;
    }
    if (child.stderr) |stderr| {
        fileClose(userdata, &.{stderr});
        child.stderr = null;
    }
    child.id = null;
}

fn processCurrentPath(userdata: ?*anyopaque, buffer: []u8) process.CurrentPathError!usize {
    charge(userdata);
    if (c.getcwd(buffer.ptr, buffer.len)) |_| {
        return std.mem.findScalar(u8, buffer, 0).?;
    }
    switch (@as(c.E, @fromBackingInt(@intCast(c._errno().*)))) {
        .NOENT => return error.CurrentDirUnlinked,
        .RANGE => return error.NameTooLong,
        .FAULT => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err),
        else => |err| return unexpectedErrno(err),
    }
}

fn processSetCurrentDir(userdata: ?*anyopaque, dir: Dir) process.SetCurrentDirError!void {
    charge(userdata);
    if (dir.handle == c.AT.FDCWD) return;
    while (true) switch (posix.errno(posix.system.fchdir(dir.handle))) {
        .SUCCESS => return,
        .INTR => {},
        .ACCES => return error.AccessDenied,
        .NOTDIR => return error.NotDir,
        .IO => return error.FileSystem,
        .BADF => |err| return errnoBug(err), // File descriptor used after closed.
        else => |err| return unexpectedErrno(err),
    };
}

fn processSetCurrentPath(userdata: ?*anyopaque, dir_path: []const u8) process.SetCurrentPathError!void {
    charge(userdata);
    var path_buffer: [PATH_MAX]u8 = undefined;
    const dir_path_posix = try pathToPosix(dir_path, &path_buffer);
    while (true) switch (posix.errno(posix.system.chdir(dir_path_posix))) {
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
    };
}

/// The path of the running executable, however the platform reports it: Darwin has
/// `_NSGetExecutablePath`, FreeBSD and DragonFly a `sysctl`, NetBSD the same through
/// `KERN_PROC_ARGS`, and OpenBSD only the first argument, which may be a path or a name to look
/// for in `PATH`.
fn processExecutablePath(userdata: ?*anyopaque, out_buffer: []u8) process.ExecutablePathError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    switch (builtin.os.tag) {
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
            // _NSGetExecutablePath() returns a path that might be a symlink to the executable.
            var symlink_path_buf: [c.PATH_MAX + 1]u8 = undefined;
            var n: u32 = symlink_path_buf.len;
            if (c._NSGetExecutablePath(&symlink_path_buf, &n) != 0) return error.NameTooLong;
            const symlink_path = std.mem.sliceTo(&symlink_path_buf, 0);
            return dirRealPathFile(ev, .cwd(), symlink_path, out_buffer) catch |err| switch (err) {
                error.NetworkNotFound => unreachable, // Windows-only
                error.FileBusy => unreachable, // Windows-only
                else => |e| return e,
            };
        },
        .freebsd, .dragonfly, .netbsd => {
            var mib: [4]c_int = switch (builtin.os.tag) {
                .netbsd => .{ posix.CTL.KERN, posix.KERN.PROC_ARGS, -1, posix.KERN.PROC_PATHNAME },
                else => .{ posix.CTL.KERN, posix.KERN.PROC, posix.KERN.PROC_PATHNAME, -1 },
            };
            var out_len: usize = out_buffer.len;
            while (true) switch (posix.errno(posix.system.sysctl(&mib, mib.len, out_buffer.ptr, &out_len, null, 0))) {
                .SUCCESS => return out_len - 1, // the terminating NUL is not part of the path
                .INTR => {},
                .PERM => return error.PermissionDenied,
                .NOMEM => return error.SystemResources,
                .FAULT => |err| return errnoBug(err),
                .NOENT => |err| return errnoBug(err),
                else => |err| return unexpectedErrno(err),
            };
        },
        .openbsd => {
            // The best these systems can do is the first argument, which is a path when it
            // contains a `/` and otherwise a name to look for in `PATH`.
            const argv0 = std.mem.span(ev.argv0.value orelse return error.OperationUnsupported);
            if (std.mem.findScalar(u8, argv0, '/') == null) return error.OperationUnsupported;
            var resolved_buffer: [c.PATH_MAX]u8 = undefined;
            while (true) {
                if (c.realpath(argv0, &resolved_buffer)) |resolved| {
                    assert(resolved == &resolved_buffer);
                    const path = std.mem.sliceTo(&resolved_buffer, 0);
                    if (path.len > out_buffer.len) return error.NameTooLong;
                    @memcpy(out_buffer[0..path.len], path);
                    return path.len;
                }
                switch (@as(c.E, @fromBackingInt(@intCast(c._errno().*)))) {
                    .INTR => continue,
                    .ACCES => return error.AccessDenied,
                    .INVAL => |err| return errnoBug(err),
                    .IO => return error.InputOutput,
                    .LOOP => return error.SymLinkLoop,
                    .NAMETOOLONG => return error.NameTooLong,
                    .NOENT => return error.FileNotFound,
                    .NOTDIR => return error.NotDir,
                    .NOMEM => |err| return errnoBug(err),
                    else => |err| return unexpectedErrno(err),
                }
            }
        },
        else => |os| @compileError("executable path for " ++ @tagName(os) ++ " is not implemented"),
    }
}

/// The running executable, opened. The path may be a symlink to it, which does not matter here.
fn processExecutableOpen(
    userdata: ?*anyopaque,
    flags: Dir.OpenFileOptions,
) process.OpenExecutableError!File {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    switch (builtin.os.tag) {
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
            var symlink_path_buf: [c.PATH_MAX + 1]u8 = undefined;
            var n: u32 = symlink_path_buf.len;
            if (c._NSGetExecutablePath(&symlink_path_buf, &n) != 0) return error.NameTooLong;
            const symlink_path = std.mem.sliceTo(&symlink_path_buf, 0);
            return dirOpenFile(ev, .cwd(), symlink_path, flags);
        },
        else => {
            var path_buffer: [PATH_MAX]u8 = undefined;
            const path = try processExecutablePath(ev, &path_buffer);
            return dirOpenFile(ev, .cwd(), path_buffer[0..path], flags);
        },
    }
}

// Networking: sockets, addresses, names, and the messages that travel on them. Every operation
// that has to wait parks the calling task on the worker's kqueue with `parkFd` and retries its
// system call when the descriptor is reported ready, so each entry point is a loop around a
// call that would have blocked. The descriptors this core opens are nonblocking and
// close-on-exec, which is what makes the retry the way a call that finds nothing to do is
// resumed.

/// Whether this platform's `socket` and `accept4` refuse the close-on-exec and nonblocking
/// bits, so that a fresh descriptor has to be fixed up with `fcntl` instead: Darwin, which has
/// no `accept4` at all. `Io.Threaded` draws the same line, and `Haiku` is named there for the
/// same reason.
const socket_flags_unsupported = switch (builtin.os.tag) {
    .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos, .haiku => true,
    else => false,
};

/// The bits `socket`, `socketpair` and `accept4` take for the descriptor this core needs: the
/// close-on-exec bit and the nonblocking bit, where the platform names them.
fn socketCreationFlags() u32 {
    if (socket_flags_unsupported) return 0;
    var flags: u32 = 0;
    if (@hasDecl(posix.SOCK, "CLOEXEC")) flags |= posix.SOCK.CLOEXEC;
    if (@hasDecl(posix.SOCK, "NONBLOCK")) flags |= posix.SOCK.NONBLOCK;
    return flags;
}

/// Whether a descriptor this core opened still needs the bits `socketCreationFlags` could not
/// pass it: on Darwin, whose `socket` takes neither.
const needs_descriptor_fixup = socket_flags_unsupported or
    !@hasDecl(posix.SOCK, "CLOEXEC") or !@hasDecl(posix.SOCK, "NONBLOCK");

/// Gives a descriptor the bits its creation could not: close-on-exec, then nonblocking. The
/// nonblocking flag is a field of this platform's `O` type, so its bit is taken from there
/// rather than spelled out. `fcntl` cannot wait, so there is no cancel point in here; the
/// callers are the ones that check their region.
fn configureDescriptor(fd: fd_t) !void {
    while (true) {
        switch (posix.errno(posix.system.fcntl(fd, posix.F.SETFD, @as(usize, posix.FD_CLOEXEC)))) {
            .SUCCESS => break,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            else => |err| return unexpectedErrno(err),
        }
    }
    var flags: usize = while (true) {
        const rc = posix.system.fcntl(fd, posix.F.GETFL, @as(usize, 0));
        switch (posix.errno(rc)) {
            .SUCCESS => break @intCast(rc),
            .INTR => {},
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            else => |err| return unexpectedErrno(err),
        }
    };
    flags |= @as(usize, 1) << @bitOffsetOf(posix.O, "NONBLOCK");
    while (true) {
        switch (posix.errno(posix.system.fcntl(fd, posix.F.SETFL, flags))) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Opens a socket of `family` with `options`'s mode and protocol. The descriptor is nonblocking
/// and close-on-exec, which `socket` does itself where the bits are its own and `fcntl` does
/// afterwards where they are not (Darwin). The call cannot wait, so it is made without a cancel
/// point: the callers check their region before asking for a descriptor.
fn socket(
    ev: *Evented,
    family: posix.sa_family_t,
    options: struct { mode: net.Socket.Mode, protocol: ?net.Protocol },
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
    const flags = socketCreationFlags();
    const socket_fd = while (true) {
        const rc = posix.system.socket(family, mode | flags, protocol);
        switch (posix.errno(rc)) {
            .SUCCESS => break @as(fd_t, @intCast(rc)),
            .INTR => {},
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
    errdefer closeAsync(ev, socket_fd);
    if (needs_descriptor_fixup) try configureDescriptor(socket_fd);
    return socket_fd;
}

/// Sets an option whose value is one four-byte integer, which is the shape of every option this
/// core sets on its own sockets.
fn setsockopt(
    ev: *Evented,
    region: *CancelRegion,
    fd: fd_t,
    level: i32,
    opt_name: u32,
    option: u32,
) !void {
    _ = ev;
    const o: []const u8 = std.mem.asBytes(&option);
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.setsockopt(fd, level, opt_name, o.ptr, @intCast(o.len)))) {
            .SUCCESS => return,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .NOTSOCK => |err| return errnoBug(err), // only this core's sockets are given options
            .INVAL => |err| return errnoBug(err), // an option this core named wrongly
            .FAULT => |err| return errnoBug(err), // the option value is on this very stack
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Sets `IPV6.V6ONLY`, which has to happen before the socket is bound.
fn setIp6Only(ev: *Evented, region: *CancelRegion, fd: fd_t, ip6_only: bool) !void {
    if (posix.IPV6 == void) return error.OptionUnsupported;
    try setsockopt(ev, region, fd, posix.IPPROTO.IPV6, posix.IPV6.V6ONLY, @intFromBool(ip6_only));
}

/// Associates an internet address with a socket. An address of a family this socket is not of
/// fails with `AddressFamilyUnsupported`, and a port already in use with `AddressInUse`.
fn bind(
    ev: *Evented,
    region: *CancelRegion,
    socket_fd: fd_t,
    addr: *const posix.sockaddr,
    addr_len: posix.socklen_t,
) !void {
    _ = ev;
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.bind(socket_fd, addr, addr_len))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .ADDRINUSE => return error.AddressInUse,
            .ADDRNOTAVAIL => return error.AddressUnavailable,
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .NOMEM => return error.SystemResources,
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .INVAL => |err| return errnoBug(err), // the address and its length are this core's own
            .NOTSOCK => |err| return errnoBug(err), // the descriptor is one of this core's sockets
            .FAULT => |err| return errnoBug(err), // the address storage is on this very stack
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Associates a pathname address with a socket. Unlike an internet address, the path can fail
/// the way opening a file does.
fn bindUnix(
    ev: *Evented,
    region: *CancelRegion,
    socket_fd: fd_t,
    addr: *const posix.sockaddr,
    addr_len: posix.socklen_t,
) net.UnixAddress.ListenError!void {
    _ = ev;
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.bind(socket_fd, addr, addr_len))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .ADDRINUSE => return error.AddressInUse,
            .ADDRNOTAVAIL => return error.AddressUnavailable,
            .AFNOSUPPORT => return error.AddressFamilyUnsupported,
            .NOMEM => return error.SystemResources,
            .LOOP => return error.SymLinkLoop,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.NotDir,
            .ROFS => return error.ReadOnlyFileSystem,
            .PERM => return error.PermissionDenied,
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .INVAL => |err| return errnoBug(err), // the address and its length are this core's own
            .NOTSOCK => |err| return errnoBug(err), // the descriptor is one of this core's sockets
            .FAULT => |err| return errnoBug(err), // the address storage is on this very stack
            .NAMETOOLONG => |err| return errnoBug(err), // the path came from a `UnixAddress`
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Reads back the address a socket was bound to, so that a port the kernel chose for a bind of
/// port zero is reported. For a bound socket the kernel answers without waiting.
fn getsockname(
    ev: *Evented,
    region: *CancelRegion,
    socket_fd: fd_t,
    addr: *posix.sockaddr,
    addr_len: *posix.socklen_t,
) !void {
    _ = ev;
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.getsockname(socket_fd, addr, addr_len))) {
            .SUCCESS => return,
            .INTR => {},
            .NOBUFS => return error.SystemResources,
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .FAULT => |err| return errnoBug(err), // the address storage is on this very stack
            .INVAL => |err| return errnoBug(err), // the length this core passed is wrong
            .NOTSOCK => |err| return errnoBug(err), // only this core's own sockets are asked
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// The `SO.ERROR` of a socket, in errno numbers, which is how the kernel reports how a
/// nonblocking connect went: zero when it connected, and the errno of the failure otherwise.
fn getsockoptError(ev: *Evented, region: *CancelRegion, socket_fd: fd_t) !i32 {
    _ = ev;
    var value: i32 = 0;
    var len: posix.socklen_t = @sizeOf(i32);
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.getsockopt(
            socket_fd,
            posix.SOL.SOCKET,
            posix.SO.ERROR,
            &value,
            &len,
        ))) {
            .SUCCESS => return value,
            .INTR => {},
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .FAULT => |err| return errnoBug(err), // the value storage is on this very stack
            .INVAL => |err| return errnoBug(err), // the option this core asked for is not valid
            .NOTSOCK => |err| return errnoBug(err), // only this core's own sockets are asked
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Starts a socket listening, which is what makes its connections queue up for `netAccept`.
/// The call can fail when the address was taken between the bind and here.
fn listenSocket(
    ev: *Evented,
    region: *CancelRegion,
    socket_fd: fd_t,
    backlog: u31,
) !void {
    _ = ev;
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.listen(socket_fd, backlog))) {
            .SUCCESS => return,
            .INTR => {},
            .ADDRINUSE => return error.AddressInUse,
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .NOTSOCK => |err| return errnoBug(err), // only this core's own sockets are listened on
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Waits for `fd` with no deadline, which is every wait this core makes except a connect with a
/// timeout. A deadline-less park adds no timer to the kqueue, so the only way it can end
/// without the descriptor being ready is a cancel request; a timeout it does report is a wake
/// that nothing scheduled, and the wait is simply taken again.
fn parkFdIndefinitely(ev: *Evented, region: *CancelRegion, fd: fd_t, filter: Filter) Io.Cancelable!void {
    while (true) {
        parkFd(ev, region, fd, filter, .none) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            error.Timeout => continue,
        };
        return;
    }
}

/// Opens a listening socket for `address`: one of the address's family with the mode and
/// protocol of the options, with the address options applied, then bound and made to listen.
/// The address the socket ended up with is reported back, so a port of zero is replaced by the
/// port the kernel chose.
fn netListenIp(
    userdata: ?*anyopaque,
    address: *const net.IpAddress,
    options: net.IpAddress.ListenOptions,
) net.IpAddress.ListenError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    const family = posixAddressFamily(address);
    try region.check();
    const socket_fd = try socket(ev, family, .{
        .mode = options.mode,
        .protocol = options.protocol,
    });
    errdefer closeAsync(ev, socket_fd);

    if (options.reuse_address) {
        try setsockopt(ev, &region, socket_fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, 1);
        if (@hasDecl(posix.SO, "REUSEPORT"))
            try setsockopt(ev, &region, socket_fd, posix.SOL.SOCKET, posix.SO.REUSEPORT, 1);
    }

    var storage: PosixAddress = undefined;
    var addr_len = addressToPosix(address, &storage);
    try bind(ev, &region, socket_fd, &storage.any, addr_len);
    try listenSocket(ev, &region, socket_fd, options.kernel_backlog);
    try getsockname(ev, &region, socket_fd, &storage.any, &addr_len);
    return .{ .handle = socket_fd, .address = addressFromPosix(&storage) };
}

/// Takes the next connection that is waiting on a listening socket. The accepted descriptor
/// gets the close-on-exec and nonblocking bits the listening one has: `accept4` does that where
/// the platform has it, and Darwin, which does not, has the descriptor fixed up afterwards.
fn netAccept(
    userdata: ?*anyopaque,
    listen_handle: net.Socket.Handle,
    options: net.Server.AcceptOptions,
) net.Server.AcceptError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = options; // `AcceptOptions` is void on POSIX
    var region: CancelRegion = .init();
    defer region.deinit();
    var storage: PosixAddress = undefined;
    while (true) {
        var addr_len: posix.socklen_t = @sizeOf(PosixAddress);
        try region.check();
        const rc = if (socket_flags_unsupported)
            posix.system.accept(listen_handle, &storage.any, &addr_len)
        else
            posix.system.accept4(listen_handle, &storage.any, &addr_len, socketCreationFlags());
        switch (posix.errno(rc)) {
            .SUCCESS => {
                const fd: fd_t = @intCast(rc);
                errdefer closeAsync(ev, fd);
                if (needs_descriptor_fixup) try configureDescriptor(fd);
                return .{ .handle = fd, .address = addressFromPosix(&storage) };
            },
            .INTR => {},
            // Another task took the connection between this call and the one before it, or the
            // accept has to wait for the next one. Level-triggered, so a connection that is
            // already there is reported at once.
            .AGAIN => try parkFdIndefinitely(ev, &region, listen_handle, filt_read),
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .CONNABORTED => return error.ConnectionAborted,
            .FAULT => |err| return errnoBug(err), // the address storage is on this very stack
            .INVAL => return error.SocketNotListening,
            .NOTSOCK => |err| return errnoBug(err), // only this core's own sockets are accepted on
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NFILE => return error.SystemFdQuotaExceeded,
            .NOBUFS => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .OPNOTSUPP => |err| return errnoBug(err), // the listening socket is a stream socket
            .PROTO => return error.ProtocolFailure,
            .PERM => return error.BlockedByFirewall,
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Binds an address without listening: the socket is of the options' mode and protocol, the
/// address is associated with it, and the address it ended up with is reported.
fn netBindIp(
    userdata: ?*anyopaque,
    address: *const net.IpAddress,
    options: net.IpAddress.BindOptions,
) net.IpAddress.BindError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    const family = posixAddressFamily(address);
    try region.check();
    const socket_fd = try socket(ev, family, .{
        .mode = options.mode,
        .protocol = options.protocol,
    });
    errdefer closeAsync(ev, socket_fd);
    if (options.ip6_only) |ip6_only| try setIp6Only(ev, &region, socket_fd, ip6_only);
    var storage: PosixAddress = undefined;
    var addr_len = addressToPosix(address, &storage);
    try bind(ev, &region, socket_fd, &storage.any, addr_len);
    if (options.allow_broadcast)
        try setsockopt(ev, &region, socket_fd, posix.SOL.SOCKET, posix.SO.BROADCAST, 1);
    try getsockname(ev, &region, socket_fd, &storage.any, &addr_len);
    return .{ .handle = socket_fd, .address = addressFromPosix(&storage) };
}

/// Maps an `errno` from `connect` to the internet address's errors. `INTR` and `EAGAIN` are the
/// caller's own to handle.
fn connectIpErrno(err: posix.E) net.IpAddress.ConnectError {
    return switch (err) {
        .ACCES => error.AccessDenied,
        .ADDRNOTAVAIL => error.AddressUnavailable,
        .AFNOSUPPORT => error.AddressFamilyUnsupported,
        .AGAIN, .INPROGRESS => error.WouldBlock,
        .ALREADY => error.ConnectionPending,
        .CONNREFUSED => error.ConnectionRefused,
        .CONNRESET => error.ConnectionResetByPeer,
        .HOSTUNREACH => error.HostUnreachable,
        .NETUNREACH => error.NetworkUnreachable,
        .TIMEDOUT => error.Timeout,
        .NETDOWN => error.NetworkDown,
        .BADF => |errno| errnoBug(errno), // the descriptor was closed under this operation
        .CONNABORTED => |errno| errnoBug(errno), // the socket was never connected
        .FAULT => |errno| errnoBug(errno), // the address storage is on this very stack
        .ISCONN => |errno| errnoBug(errno), // the caller checks for this before asking again
        .NOENT => |errno| errnoBug(errno), // the protocol names a service that does not exist
        .NOTSOCK => |errno| errnoBug(errno), // the descriptor is one of this core's sockets
        .PERM => |errno| errnoBug(errno), // the address is one this core built itself
        .PROTOTYPE => |errno| errnoBug(errno), // the socket's mode and protocol disagree
        else => |errno| unexpectedErrno(errno),
    };
}

/// Maps an `errno` from `connect` to a pathname address's errors, which are the internet ones
/// plus the ways a path can fail.
fn connectUnixErrno(err: posix.E) net.UnixAddress.ConnectError {
    return switch (err) {
        .ACCES => error.AccessDenied,
        .AFNOSUPPORT => error.AddressFamilyUnsupported,
        .AGAIN, .INPROGRESS => error.WouldBlock,
        .CONNREFUSED => error.ConnectionRefused,
        .NETDOWN => error.NetworkDown,
        .LOOP => error.SymLinkLoop,
        .NOENT => error.FileNotFound,
        .NOTDIR => error.NotDir,
        .ROFS => error.ReadOnlyFileSystem,
        .PERM => error.PermissionDenied,
        .BADF => |errno| errnoBug(errno), // the descriptor was closed under this operation
        .FAULT => |errno| errnoBug(errno), // the address storage is on this very stack
        .INVAL => |errno| errnoBug(errno), // the address and its length are this core's own
        .ISCONN => |errno| errnoBug(errno), // the caller checks for this before asking again
        .NOTSOCK => |errno| errnoBug(errno), // the descriptor is one of this core's sockets
        .PROTOTYPE => |errno| errnoBug(errno), // the socket is not a stream socket
        else => |errno| unexpectedErrno(errno),
    };
}

/// Whether `timeout` has already passed, in which case no wait can be made for it: a park with
/// no time left registers neither the event nor a timer, so it would never be woken.
fn timeoutElapsed(ev: *Evented, timeout: Io.Timeout) bool {
    const duration = timeout.toDurationFromNow(ev.io()) orelse return false;
    return duration.raw.toNanoseconds() == 0;
}

/// Waits for a connection that is being made to end, and returns how it went as the errno the
/// kernel leaves in `SO.ERROR` (zero when it connected). The descriptor becomes writable when
/// the attempt is over, which is what the wait is for; the connection's own state is read before
/// every wait, so that a wait a finished attempt makes pointless is not made, and a deadline
/// that has passed is reported instead of a wait that could not be woken.
fn awaitConnect(
    ev: *Evented,
    region: *CancelRegion,
    socket_fd: fd_t,
    timeout: Io.Timeout,
) (Io.Cancelable || error{ Timeout, Unexpected })!i32 {
    while (true) {
        const so_error = try getsockoptError(ev, region, socket_fd);
        if (so_error == 0) return 0;
        const err: posix.E = @fromBackingInt(@intCast(so_error));
        switch (err) {
            // The attempt is still being made.
            .INPROGRESS, .AGAIN, .ALREADY => {},
            else => return so_error,
        }
        if (timeoutElapsed(ev, timeout)) return error.Timeout;
        parkFd(ev, region, socket_fd, filt_write, timeout) catch |err2| switch (err2) {
            error.Canceled => return error.Canceled,
            error.Timeout => return error.Timeout,
        };
    }
}

/// Connects an internet address. A nonblocking `connect` returns `EINPROGRESS` while the
/// kernel works on it; the descriptor becomes writable when that is over, and `SO.ERROR` then
/// says whether the connection was made.
fn connectIp(
    ev: *Evented,
    region: *CancelRegion,
    socket_fd: fd_t,
    addr: *const posix.sockaddr,
    addr_len: posix.socklen_t,
    timeout: Io.Timeout,
) net.IpAddress.ConnectError!void {
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.connect(socket_fd, addr, addr_len))) {
            .SUCCESS, .ISCONN => return, // a second connect on a connected socket is nothing to do
            .INTR => {},
            .INPROGRESS, .AGAIN, .ALREADY => {
                const so_error = awaitConnect(ev, region, socket_fd, timeout) catch |err| switch (err) {
                    error.Canceled => return error.Canceled,
                    error.Timeout => return error.Timeout,
                    error.Unexpected => |e| return e,
                };
                if (so_error == 0) return;
                return connectIpErrno(@fromBackingInt(@intCast(so_error)));
            },
            else => |err| return connectIpErrno(err),
        }
    }
}

/// Opens a socket of the address's family and connects it, with the options' mode, protocol and
/// timeout. The address it connected to is the local address of the socket, which is what the
/// caller gets.
fn netConnectIp(
    userdata: ?*anyopaque,
    address: *const net.IpAddress,
    options: net.IpAddress.ConnectOptions,
) net.IpAddress.ConnectError!net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    const family = posixAddressFamily(address);
    try region.check();
    const socket_fd = try socket(ev, family, .{
        .mode = options.mode,
        .protocol = options.protocol,
    });
    errdefer closeAsync(ev, socket_fd);
    var storage: PosixAddress = undefined;
    var addr_len = addressToPosix(address, &storage);
    try connectIp(ev, &region, socket_fd, &storage.any, addr_len, options.timeout);
    try getsockname(ev, &region, socket_fd, &storage.any, &addr_len);
    return .{ .handle = socket_fd, .address = addressFromPosix(&storage) };
}

/// Opens a listening socket on a pathname address, and leaves the file behind for the caller to
/// delete, as `bind` does.
fn netListenUnix(
    userdata: ?*anyopaque,
    address: *const net.UnixAddress,
    options: net.UnixAddress.ListenOptions,
) net.UnixAddress.ListenError!net.Socket.Handle {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    try region.check();
    const socket_fd = socket(ev, posix.AF.UNIX, .{ .mode = .stream, .protocol = null }) catch |err| switch (err) {
        error.ProtocolUnsupportedBySystem => return error.AddressFamilyUnsupported,
        error.ProtocolUnsupportedByAddressFamily => return error.AddressFamilyUnsupported,
        error.SocketModeUnsupported => return error.AddressFamilyUnsupported,
        error.OptionUnsupported => return error.Unexpected,
        else => |e| return e,
    };
    errdefer closeAsync(ev, socket_fd);
    var storage: UnixAddress = undefined;
    const addr_len = addressUnixToPosix(address, &storage);
    try bindUnix(ev, &region, socket_fd, &storage.any, addr_len);
    try listenSocket(ev, &region, socket_fd, options.kernel_backlog);
    return socket_fd;
}

/// Connects a pathname address, which for a unix socket is a rendezvous with the process that
/// is listening on the path. A nonblocking connect to a pathname goes through the same
/// `EINPROGRESS` and `SO.ERROR` as an internet one.
fn connectUnix(
    ev: *Evented,
    region: *CancelRegion,
    socket_fd: fd_t,
    addr: *const posix.sockaddr,
    addr_len: posix.socklen_t,
) net.UnixAddress.ConnectError!void {
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.connect(socket_fd, addr, addr_len))) {
            .SUCCESS, .ISCONN => return, // a second connect on a connected socket is nothing to do
            .INTR => {},
            .INPROGRESS, .AGAIN, .ALREADY => {
                // A pathname connect has no deadline to wait for, so a park with no time left
                // cannot be the answer; `WouldBlock` is what this address's set has for a wait
                // that ended without the connection being made.
                const so_error = awaitConnect(ev, region, socket_fd, .none) catch |err| switch (err) {
                    error.Canceled => return error.Canceled,
                    error.Timeout => return error.WouldBlock,
                    error.Unexpected => |e| return e,
                };
                if (so_error == 0) return;
                return connectUnixErrno(@fromBackingInt(@intCast(so_error)));
            },
            else => |err| return connectUnixErrno(err),
        }
    }
}

/// Opens a socket for the pathname address and connects it.
fn netConnectUnix(
    userdata: ?*anyopaque,
    address: *const net.UnixAddress,
) net.UnixAddress.ConnectError!net.Socket.Handle {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    try region.check();
    const socket_fd = socket(ev, posix.AF.UNIX, .{ .mode = .stream, .protocol = null }) catch |err| switch (err) {
        error.ProtocolUnsupportedByAddressFamily => return error.AddressFamilyUnsupported,
        error.SocketModeUnsupported => return error.AddressFamilyUnsupported,
        error.ProtocolUnsupportedBySystem => return error.AddressFamilyUnsupported,
        error.OptionUnsupported => return error.Unexpected,
        else => |e| return e,
    };
    errdefer closeAsync(ev, socket_fd);
    var storage: UnixAddress = undefined;
    const addr_len = addressUnixToPosix(address, &storage);
    try connectUnix(ev, &region, socket_fd, &storage.any, addr_len);
    return socket_fd;
}

/// Opens two sockets that are connected to each other. Both get the close-on-exec and
/// nonblocking bits, and both are asked for the address they ended up with.
fn netSocketCreatePair(
    userdata: ?*anyopaque,
    options: net.Socket.CreatePairOptions,
) net.Socket.CreatePairError![2]net.Socket {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    const family: posix.sa_family_t = switch (options.family) {
        .ip4 => posix.AF.INET,
        .ip6 => posix.AF.INET6,
    };
    const mode, const protocol = try posixSocketModeProtocol(family, options.mode, options.protocol);
    const flags = socketCreationFlags();
    var sockets: [2]fd_t = undefined;
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.socketpair(family, mode | flags, protocol, &sockets))) {
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
            // Socket pairs exist for `AF.UNIX` only, and only for the modes that have them.
            .OPNOTSUPP => return error.OperationUnsupported,
            else => |err| return unexpectedErrno(err),
        }
    }
    errdefer {
        closeAsync(ev, sockets[0]);
        closeAsync(ev, sockets[1]);
    }
    if (needs_descriptor_fixup) {
        try configureDescriptor(sockets[0]);
        try configureDescriptor(sockets[1]);
    }
    var storages: [2]PosixAddress = undefined;
    var addr_lens: [2]posix.socklen_t = .{ @sizeOf(PosixAddress), @sizeOf(PosixAddress) };
    try getsockname(ev, &region, sockets[0], &storages[0].any, &addr_lens[0]);
    try getsockname(ev, &region, sockets[1], &storages[1].any, &addr_lens[1]);
    return .{
        .{ .handle = sockets[0], .address = addressFromPosix(&storages[0]) },
        .{ .handle = sockets[1], .address = addressFromPosix(&storages[1]) },
    };
}

/// Maps an `errno` from `sendmsg`; `INTR` and `EAGAIN` are the caller's own to handle.
fn sendmsgErrno(err: posix.E) Io.Operation.NetSend.Error {
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
        .CONNREFUSED => error.ConnectionRefused,
        .BADF => |errno| errnoBug(errno), // the descriptor was closed under this operation
        .DESTADDRREQ => |errno| errnoBug(errno), // the caller always gives an address to send to
        .FAULT => |errno| errnoBug(errno), // the message and its vectors are on this very stack
        .INVAL => |errno| errnoBug(errno), // an argument this core built is wrong
        .ISCONN => |errno| errnoBug(errno), // a recipient was given for a connected socket
        .NOTSOCK => |errno| errnoBug(errno), // the descriptor is one of this core's sockets
        .OPNOTSUPP => |errno| errnoBug(errno), // a flag that this socket's mode does not take
        else => |errno| unexpectedErrno(errno),
    };
}

/// Maps an `errno` from `readv`; `INTR` and `EAGAIN` are the caller's own to handle.
fn readvErrno(err: posix.E) Io.Operation.NetRead.Error {
    return switch (err) {
        .NOBUFS => error.SystemResources,
        .NOMEM => error.SystemResources,
        .NOTCONN => error.SocketUnconnected,
        .CONNRESET => error.ConnectionResetByPeer,
        .TIMEDOUT => error.ConnectionTimedOut,
        .NETDOWN => error.NetworkDown,
        .ACCES => error.AccessDenied,
        .BADF => |errno| errnoBug(errno), // the descriptor was closed under this operation
        .FAULT => |errno| errnoBug(errno), // the vectors are on this very stack
        .INVAL => |errno| errnoBug(errno), // the vector count this core passed is wrong
        .NOTSOCK => |errno| errnoBug(errno), // the descriptor is one of this core's sockets
        else => |errno| unexpectedErrno(errno),
    };
}

/// Maps an `errno` from `sendmsg` on a stream; `INTR` and `EAGAIN` are the caller's own to
/// handle.
fn netWriteErrno(err: posix.E) Io.Operation.NetWrite.Error {
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
        .CONNREFUSED => error.ConnectionRefused,
        .ACCES => |errno| errnoBug(errno), // the socket was never connected to a peer
        .BADF => |errno| errnoBug(errno), // the descriptor was closed under this operation
        .DESTADDRREQ => |errno| errnoBug(errno), // a stream always has a peer
        .FAULT => |errno| errnoBug(errno), // the vectors are on this very stack
        .INVAL => |errno| errnoBug(errno), // an argument this core built is wrong
        .ISCONN => |errno| errnoBug(errno), // a recipient was given for a connected socket
        .MSGSIZE => |errno| errnoBug(errno), // a stream write is not a message
        .NOTSOCK => |errno| errnoBug(errno), // the descriptor is one of this core's sockets
        .OPNOTSUPP => |errno| errnoBug(errno), // a flag that a stream socket does not take
        .SOCKTNOSUPPORT => |errno| errnoBug(errno), // the socket's mode was checked when it was made
        else => |errno| unexpectedErrno(errno),
    };
}

/// Maps an `errno` from `recvmsg`; `INTR` and `EAGAIN` are the caller's own to handle.
fn netReceiveErrno(err: posix.E) net.Socket.ReceiveError {
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
        .CONNREFUSED => error.PortUnreachable, // an ICMP port unreachable for a datagram sent earlier
        .BADF => |errno| errnoBug(errno), // the descriptor was closed under this operation
        .FAULT => |errno| errnoBug(errno), // the vectors and the buffers are on this very stack
        .INVAL => |errno| errnoBug(errno), // an argument this core built is wrong
        .NOTSOCK => |errno| errnoBug(errno), // the descriptor is one of this core's sockets
        .OPNOTSUPP => |errno| errnoBug(errno), // a flag that this socket's mode does not take
        else => |errno| unexpectedErrno(errno),
    };
}

/// The `msg` flags a `net.SendFlags` asks for, of the ones this platform names, plus
/// `MSG.NOSIGNAL` where the platform has it: a peer that goes away must not signal the process
/// from inside a send.
fn sendFlags(flags: net.SendFlags) u32 {
    return @as(u32, if (@hasDecl(posix.MSG, "CONFIRM") and flags.confirm) posix.MSG.CONFIRM else 0) |
        @as(u32, if (@hasDecl(posix.MSG, "DONTROUTE") and flags.dont_route) posix.MSG.DONTROUTE else 0) |
        @as(u32, if (@hasDecl(posix.MSG, "EOR") and flags.eor) posix.MSG.EOR else 0) |
        @as(u32, if (@hasDecl(posix.MSG, "OOB") and flags.oob) posix.MSG.OOB else 0) |
        @as(u32, if (@hasDecl(posix.MSG, "FASTOPEN") and flags.fastopen) posix.MSG.FASTOPEN else 0) |
        @as(u32, if (@hasDecl(posix.MSG, "NOSIGNAL")) posix.MSG.NOSIGNAL else 0);
}

/// One `sendmsg` of one message, without waiting: the caller has had the socket reported
/// writable, and waits and calls again when it was not, which is what `error.WouldBlock` means
/// here.
fn netSendOneSync(
    handle: net.Socket.Handle,
    message: *net.OutgoingMessage,
    posix_flags: u32,
) (Io.Operation.NetSend.Error || error{WouldBlock})!void {
    var addr: PosixAddress = undefined;
    const one_iovec: iovec_const = .{
        .base = message.data_ptr,
        .len = message.data_len,
    };
    const msg: posix.msghdr_const = .{
        .name = &addr.any,
        .namelen = addressToPosix(message.address, &addr),
        .iov = (&one_iovec)[0..1],
        .iovlen = 1,
        // The kernel rejects a bad control pointer even when the length is zero.
        .control = if (message.control.len == 0) null else @constCast(message.control.ptr),
        .controllen = @intCast(message.control.len),
        .flags = 0,
    };
    while (true) {
        const rc = posix.system.sendmsg(handle, &msg, posix_flags);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                message.data_len = @intCast(rc);
                return;
            },
            .INTR => {},
            .AGAIN => return error.WouldBlock,
            else => |err| return sendmsgErrno(err),
        }
    }
}

/// Sends one message, waiting for the socket to become writable when it is full.
fn netSendOne(
    ev: *Evented,
    region: *CancelRegion,
    handle: net.Socket.Handle,
    message: *net.OutgoingMessage,
    posix_flags: u32,
) (Io.Cancelable || Io.Operation.NetSend.Error)!void {
    while (true) {
        try region.check();
        netSendOneSync(handle, message, posix_flags) catch |err| switch (err) {
            error.WouldBlock => {
                try parkFdIndefinitely(ev, region, handle, filt_write);
                continue;
            },
            else => |e| return e,
        };
        return;
    }
}

/// Sends each message in turn. The count of messages that were sent is reported with any error,
/// as the interface's own result wants: a message that was partially sent says how much of it
/// left by updating its `data_len`, and the ones after it are the caller's to send again.
fn netSend(
    ev: *Evented,
    region: *CancelRegion,
    handle: net.Socket.Handle,
    messages: []net.OutgoingMessage,
    flags: net.SendFlags,
) Io.Cancelable!struct { ?net.Socket.SendError, usize } {
    const posix_flags = sendFlags(flags);
    var i: usize = 0;
    while (messages.len - i != 0) : (i += 1) {
        netSendOne(ev, region, handle, &messages[i], posix_flags) catch |err| switch (err) {
            error.Canceled => |e| if (i == 0) return e else return .{ null, i },
            else => |e| return .{ e, i },
        };
    }
    return .{ null, i };
}

/// The vectors for `header`, the literal `data` bytes, and the last of `data` repeated `splat`
/// times. The returned slice refers to `iovecs`, and for a repeated one-byte pattern to
/// `splat_buffer`.
fn netWriteIovecs(
    iovecs: *[max_iovecs_len]iovec_const,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
    splat_buffer: *[splat_buffer_size]u8,
) []const iovec_const {
    var iovlen: usize = 0;
    addIovec(iovecs, &iovlen, header);
    for (data[0 .. data.len - 1]) |bytes| addIovec(iovecs, &iovlen, bytes);
    const pattern = data[data.len - 1];
    if (iovecs.len - iovlen != 0) switch (splat) {
        0 => {},
        1 => addIovec(iovecs, &iovlen, pattern),
        else => switch (pattern.len) {
            0 => {},
            1 => {
                const memset_len = @min(splat_buffer.len, splat);
                const buf = splat_buffer[0..memset_len];
                @memset(buf, pattern[0]);
                addIovec(iovecs, &iovlen, buf);
                var remaining = splat - buf.len;
                while (remaining > splat_buffer.len and iovecs.len - iovlen != 0) {
                    addIovec(iovecs, &iovlen, splat_buffer);
                    remaining -= splat_buffer.len;
                }
                addIovec(iovecs, &iovlen, splat_buffer[0..@min(remaining, splat_buffer.len)]);
            },
            else => for (0..@min(splat, iovecs.len - iovlen)) |_| {
                addIovec(iovecs, &iovlen, pattern);
            },
        },
    };
    return iovecs[0..iovlen];
}

/// One `sendmsg` of the vectors `header`, `data` and `splat` describe, built by
/// `netWriteIovecs`, without waiting: `error.WouldBlock` means the caller has to wait for the
/// socket to become writable and call again.
fn netWriteSync(
    handle: net.Socket.Handle,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
) (Io.Operation.NetWrite.Error || error{WouldBlock})!usize {
    var iovecs: [max_iovecs_len]iovec_const = undefined;
    var splat_buffer: [splat_buffer_size]u8 = undefined;
    const used = netWriteIovecs(&iovecs, header, data, splat, &splat_buffer);
    const msg: posix.msghdr_const = .{
        .name = null,
        .namelen = 0,
        .iov = used.ptr,
        .iovlen = @intCast(used.len),
        .control = null,
        .controllen = 0,
        .flags = 0,
    };
    const posix_flags: u32 = if (@hasDecl(posix.MSG, "NOSIGNAL")) posix.MSG.NOSIGNAL else 0;
    while (true) {
        const rc = posix.system.sendmsg(handle, &msg, posix_flags);
        switch (posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => {},
            .AGAIN => return error.WouldBlock,
            else => |err| return netWriteErrno(err),
        }
    }
}

/// Writes the header, the data and the last buffer repeated `splat` times to a socket, waiting
/// for it to become writable when it is full. A short write is reported as the count that left.
fn netWrite(
    ev: *Evented,
    region: *CancelRegion,
    handle: net.Socket.Handle,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
) net.Stream.Writer.Error!usize {
    while (true) {
        try region.check();
        return netWriteSync(handle, header, data, splat) catch |err| switch (err) {
            error.WouldBlock => {
                try parkFdIndefinitely(ev, region, handle, filt_write);
                continue;
            },
            else => |e| return e,
        };
    }
}

/// One `readv` of `data`'s buffers into the socket's data, without waiting:
/// `error.WouldBlock` means the caller has to wait for the socket and call again.
fn netReadSync(handle: net.Socket.Handle, data: [][]u8) (Io.Operation.NetRead.Error || error{WouldBlock})!usize {
    var iovecs: [max_iovecs_len]iovec = undefined;
    var i: usize = 0;
    for (data) |buf| {
        if (iovecs.len - i == 0) break;
        if (buf.len != 0) {
            iovecs[i] = .{ .base = buf.ptr, .len = buf.len };
            i += 1;
        }
    }
    const dest = iovecs[0..i];
    assert(dest.len > 0);
    while (true) {
        const rc = posix.system.readv(handle, dest.ptr, @intCast(dest.len));
        switch (posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => {},
            .AGAIN => return error.WouldBlock,
            else => |err| return readvErrno(err),
        }
    }
}

/// Reads into `data`'s buffers in order, waiting for the socket to become readable when it has
/// nothing to give. The number of bytes read is returned; zero means the peer closed.
fn netRead(
    ev: *Evented,
    region: *CancelRegion,
    handle: net.Socket.Handle,
    data: [][]u8,
) net.Stream.Reader.Error!usize {
    while (true) {
        try region.check();
        return netReadSync(handle, data) catch |err| switch (err) {
            error.WouldBlock => {
                try parkFdIndefinitely(ev, region, handle, filt_read);
                continue;
            },
            else => |e| return e,
        };
    }
}

/// Receives as many messages as are already there, without waiting: the first message is taken
/// as it is found, and `error.WouldBlock` from it means the socket has nothing yet. The count
/// of messages received is reported with any error.
fn netReceiveSync(
    handle: net.Socket.Handle,
    message_buffer: []net.IncomingMessage,
    data_buffer: []u8,
    flags: net.ReceiveFlags,
) (net.Socket.ReceiveError || error{WouldBlock})!usize {
    const posix_flags: u32 =
        @as(u32, if (flags.oob) posix.MSG.OOB else 0) |
        @as(u32, if (flags.peek) posix.MSG.PEEK else 0) |
        @as(u32, if (flags.trunc) posix.MSG.TRUNC else 0) |
        @as(u32, if (@hasDecl(posix.MSG, "NOSIGNAL")) posix.MSG.NOSIGNAL else 0);
    var message_i: usize = 0;
    var data_i: usize = 0;
    while (message_i != message_buffer.len) : (message_i += 1) {
        const message = &message_buffer[message_i];
        const remaining = data_buffer[data_i..];
        var storage: PosixAddress = undefined;
        var iov: iovec = .{ .base = remaining.ptr, .len = remaining.len };
        var msg: posix.msghdr = .{
            .name = &storage.any,
            .namelen = @sizeOf(PosixAddress),
            .iov = (&iov)[0..1],
            .iovlen = 1,
            .control = message.control.ptr,
            .controllen = @intCast(message.control.len),
            .flags = undefined,
        };
        const rc: isize = while (true) {
            const rc = posix.system.recvmsg(handle, &msg, posix_flags);
            switch (posix.errno(rc)) {
                .SUCCESS => break rc,
                .INTR => {},
                .AGAIN => {
                    if (message_i != 0) return message_i;
                    return error.WouldBlock;
                },
                else => |err| return netReceiveErrno(err),
            }
        };
        const data = remaining[0..@intCast(rc)];
        data_i += data.len;
        message.* = .{
            .from = addressFromPosix(&storage),
            .data = data,
            .control = if (msg.control) |ptr| @as([*]u8, @ptrCast(ptr))[0..msg.controllen] else message.control,
            .flags = .{
                .eor = (msg.flags & posix.MSG.EOR) != 0,
                .trunc = (msg.flags & posix.MSG.TRUNC) != 0,
                .ctrunc = (msg.flags & posix.MSG.CTRUNC) != 0,
                .oob = (msg.flags & posix.MSG.OOB) != 0,
                .errqueue = if (@hasDecl(posix.MSG, "ERRQUEUE")) (msg.flags & posix.MSG.ERRQUEUE) != 0 else false,
            },
        };
    }
    return message_buffer.len;
}

/// Receives the first message of a drain, waiting for the socket to become readable when it has
/// nothing for the caller yet.
fn netReceiveFirst(
    ev: *Evented,
    region: *CancelRegion,
    handle: net.Socket.Handle,
    message_buffer: []net.IncomingMessage,
    data_buffer: []u8,
    flags: net.ReceiveFlags,
) (Io.Cancelable || net.Socket.ReceiveError)!usize {
    while (true) {
        try region.check();
        const result = netReceiveSync(handle, message_buffer, data_buffer, flags);
        if (result) |count| return count else |err| {
            switch (err) {
                error.WouldBlock => try parkFdIndefinitely(ev, region, handle, filt_read),
                else => |e| return e,
            }
        }
    }
}

/// Receives datagrams. The first message may have to wait for the socket to become readable;
/// the ones after it are only taken if they are already there, which is what lets a caller
/// drain a socket with one wait. The count of messages received is reported with any error.
fn netReceive(
    ev: *Evented,
    region: *CancelRegion,
    handle: net.Socket.Handle,
    message_buffer: []net.IncomingMessage,
    data_buffer: []u8,
    flags: net.ReceiveFlags,
) struct { ?net.Socket.ReceiveError, usize } {
    if (message_buffer.len == 0) return .{ null, 0 };
    // `net.Socket.ReceiveError` is exactly the set `netReceiveFirst` reports, cancelation
    // included, so every error it can produce goes to the caller as it is.
    const first_count = netReceiveFirst(
        ev,
        region,
        handle,
        message_buffer[0..1],
        data_buffer,
        flags,
    ) catch |err| return .{ err, 0 };
    assert(first_count == 1);
    const rest = netReceiveSync(
        handle,
        message_buffer[1..],
        data_buffer[message_buffer[0].data.len..],
        flags,
    ) catch |err| switch (err) {
        // Nothing else was waiting, which is the end of this drain.
        error.WouldBlock => return .{ null, 1 },
        else => |e| return .{ e, 1 },
    };
    return .{ null, 1 + rest };
}

/// Whether the kernel can move a file's bytes to a socket without the process seeing them:
/// `sendfile` on Darwin and FreeBSD. A socket that it refuses is answered with
/// `error.Unimplemented`, and the bytes are copied through a buffer instead.
const have_sendfile = switch (builtin.os.tag) {
    .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos, .freebsd => true,
    else => false,
};

/// Whether the kernel can copy a whole file to a socket in one call: Darwin's `fcopyfile`.
const have_fcopyfile = switch (builtin.os.tag) {
    .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => true,
    else => false,
};

/// One flag for the process: a kernel call that refuses a pair of descriptors would refuse it
/// for every socket this core sends a file on, so it is turned off the first time and no
/// further call is made. `Io.Threaded.UseSendfile` and `UseFcopyfile` name only the disabling
/// value on the platforms that have no such call, so on those this is inert.
var use_sendfile: Io.Threaded.UseSendfile = .default;
var use_fcopyfile: Io.Threaded.UseFcopyfile = .default;

/// Sends the file's bytes to the socket with the kernel's own copy, for the platforms that have
/// one. Returns the number of bytes sent, zero at the end of the file, or
/// `error.Unimplemented` when the kernel will not move bytes between this pair of descriptors,
/// which the caller answers by copying them itself.
fn sendfileToSocket(
    ev: *Evented,
    region: *CancelRegion,
    file_reader: *File.Reader,
    socket_fd: net.Socket.Handle,
    offset: c.off_t,
    limit_bytes: usize,
) net.Stream.Writer.WriteFileError!usize {
    return switch (builtin.os.tag) {
        .freebsd => sendfileFreebsd(ev, region, file_reader, socket_fd, offset, limit_bytes),
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => sendfileDarwin(
            ev,
            region,
            file_reader,
            socket_fd,
            offset,
            limit_bytes,
        ),
        else => @compileError("no sendfile on " ++ @tagName(builtin.os.tag)),
    };
}

/// FreeBSD's `sendfile` takes the count to send and reports the count it sent through a pointer,
/// and sends as much as the socket will take. `EAGAIN` with nothing sent means the socket is
/// full, and `EBUSY` that the file changed under the call; both are answered by waiting and
/// calling again.
fn sendfileFreebsd(
    ev: *Evented,
    region: *CancelRegion,
    file_reader: *File.Reader,
    socket_fd: net.Socket.Handle,
    offset: c.off_t,
    limit_bytes: usize,
) net.Stream.Writer.WriteFileError!usize {
    const file_fd = file_reader.file.handle;
    var file_offset = offset;
    var total: usize = 0;
    while (total != limit_bytes) {
        try region.check();
        var sbytes: c.off_t = 0;
        switch (posix.errno(c.sendfile(
            file_fd,
            socket_fd,
            file_offset,
            limit_bytes - total,
            null,
            &sbytes,
            0,
        ))) {
            .SUCCESS => {},
            .INTR, .BUSY => continue,
            .AGAIN => if (sbytes == 0) {
                try parkFdIndefinitely(ev, region, socket_fd, filt_write);
                continue;
            },
            .NOTSOCK, .OPNOTSUPP, .NOSYS => {
                // Not a pair this kernel moves bytes between: the caller copies them, and no
                // other socket of this process asks the kernel again.
                @atomicStore(Io.Threaded.UseSendfile, &use_sendfile, .disabled, .monotonic);
                return error.Unimplemented;
            },
            .NOTCONN, .PIPE => return error.SocketUnconnected,
            .CONNRESET => return error.ConnectionResetByPeer,
            .TIMEDOUT => return error.ConnectionTimedOut,
            .NOBUFS, .NOMEM => return error.SystemResources,
            .NETDOWN => return error.NetworkDown,
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .FAULT => |err| return errnoBug(err), // the counts this core passed are its own
            .INVAL => |err| return errnoBug(err), // the offset and count this core passed are wrong
            else => |err| return unexpectedErrno(err),
        }
        const sent: usize = @intCast(sbytes);
        if (sent == 0) return total; // the end of the file
        file_offset += @intCast(sent);
        total += sent;
    }
    return total;
}

/// Darwin's `sendfile` takes the count to send in an in-out parameter, which it leaves as the
/// count it sent, and holds a limit of an `i32` before it refuses the call.
fn sendfileDarwin(
    ev: *Evented,
    region: *CancelRegion,
    file_reader: *File.Reader,
    socket_fd: net.Socket.Handle,
    offset: c.off_t,
    limit_bytes: usize,
) net.Stream.Writer.WriteFileError!usize {
    const file_fd = file_reader.file.handle;
    var file_offset = offset;
    var total: usize = 0;
    while (total != limit_bytes) {
        try region.check();
        var len: c.off_t = @intCast(@min(limit_bytes - total, std.math.maxInt(i32)));
        switch (posix.errno(c.sendfile(file_fd, socket_fd, file_offset, &len, null, 0))) {
            .SUCCESS => {},
            .INTR => continue,
            .AGAIN => if (len == 0) {
                try parkFdIndefinitely(ev, region, socket_fd, filt_write);
                continue;
            },
            .NOTSOCK, .OPNOTSUPP, .NOSYS => {
                // Not a pair this kernel moves bytes between: the caller copies them, and no
                // other socket of this process asks the kernel again.
                @atomicStore(Io.Threaded.UseSendfile, &use_sendfile, .disabled, .monotonic);
                return error.Unimplemented;
            },
            .NOTCONN, .PIPE => return error.SocketUnconnected,
            .CONNRESET => return error.ConnectionResetByPeer,
            .TIMEDOUT => return error.ConnectionTimedOut,
            .NOBUFS, .NOMEM => return error.SystemResources,
            .NETDOWN => return error.NetworkDown,
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .FAULT => |err| return errnoBug(err), // the count this core passed is its own
            .INVAL => |err| return errnoBug(err), // the count this core passed is out of range
            else => |err| return unexpectedErrno(err),
        }
        const sent: usize = @intCast(len);
        if (sent == 0) return total; // the end of the file
        file_offset += @intCast(sent);
        total += sent;
    }
    return total;
}

/// Copies a whole file to the socket with Darwin's `fcopyfile`, which is the one call that
/// takes the file from its beginning to the socket's position. It needs to know the size, and
/// the caller's limit has to be unlimited because the call has no limit of its own.
fn fcopyfileToSocket(
    region: *CancelRegion,
    file_reader: *File.Reader,
    socket_fd: net.Socket.Handle,
) net.Stream.Writer.WriteFileError!usize {
    if (@atomicLoad(Io.Threaded.UseFcopyfile, &use_fcopyfile, .monotonic) == .disabled) {
        return error.Unimplemented;
    }
    const size = file_reader.getSize() catch return error.Unimplemented;
    while (true) {
        try region.check();
        switch (posix.errno(c.fcopyfile(file_reader.file.handle, socket_fd, null, .{ .DATA = true }))) {
            .SUCCESS => return @intCast(size),
            .INTR => {},
            .OPNOTSUPP, .INVAL => {
                // Not a pair this kernel copies between: the caller sends the bytes itself, and
                // no other socket of this process asks the kernel again.
                @atomicStore(Io.Threaded.UseFcopyfile, &use_fcopyfile, .disabled, .monotonic);
                return error.Unimplemented;
            },
            .NOMEM => return error.SystemResources,
            .BADF => |err| return errnoBug(err), // a descriptor was closed under this operation
            .FAULT => |err| return errnoBug(err), // the state pointer this core passed is its own
            .NOTCONN, .PIPE => return error.SocketUnconnected,
            .CONNRESET => return error.ConnectionResetByPeer,
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Reads one chunk of the file for the copy loop, waiting for a descriptor that has nothing to
/// give (a pipe, a terminal; a regular file never does): a positional reader is read at the
/// position it has reached, which leaves the descriptor's own position alone, and a streaming
/// one through the descriptor, whose position the reader follows.
fn readFileChunk(
    ev: *Evented,
    region: *CancelRegion,
    file_reader: *File.Reader,
    file_fd: fd_t,
    buffer: []u8,
    offset: usize,
) (File.ReadStreamingError || File.Reader.Error || error{ReadFailed})!usize {
    const at: ?u64 = switch (file_reader.mode) {
        .positional, .positional_simple => file_reader.pos + offset,
        .streaming, .streaming_simple => null,
        .failure => return error.ReadFailed,
    };
    while (true) {
        try region.check();
        const rc = if (at) |absolute|
            posix.system.pread(file_fd, buffer.ptr, buffer.len, @intCast(absolute))
        else
            posix.system.read(file_fd, buffer.ptr, buffer.len);
        switch (posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => {},
            .AGAIN => try parkFdIndefinitely(ev, region, file_fd, filt_read),
            .BADF => return error.NotOpenForReading, // closed under the reader, or never open for reading
            .ISDIR => return error.IsDir,
            .NOBUFS, .NOMEM => return error.SystemResources,
            .CONNRESET => return error.ConnectionResetByPeer,
            .NOTCONN => return error.SocketUnconnected,
            .IO => return error.InputOutput,
            .INVAL => |err| return errnoBug(err), // the buffer and the offset are this file's own
            .FAULT => |err| return errnoBug(err), // the buffer is on this very stack
            .OVERFLOW => |err| return errnoBug(err), // the offset is one the file's own reader gave
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// The read side of a copy, narrowed to what a writer's contract carries: an error the file
/// side reported that a writer has no name for becomes `ReadFailed`, which is what an `Io.Reader`
/// implementation reports for a failure of its own.
const ReadChunkError = error{ ReadFailed, EndOfStream, Canceled, SystemResources, Unexpected };

fn readChunkError(err: (File.ReadStreamingError || File.Reader.Error || error{ReadFailed})) ReadChunkError {
    return switch (err) {
        error.ReadFailed => error.ReadFailed,
        error.EndOfStream => error.EndOfStream,
        error.Canceled => error.Canceled,
        error.SystemResources => error.SystemResources,
        error.Unexpected => error.Unexpected,
        else => error.ReadFailed,
    };
}

/// Copies up to `limit_bytes` of the file to the socket through a stack buffer, for the
/// platforms whose kernel has no call that moves file bytes to a socket and for the pairs of
/// descriptors such a call refuses. Only bytes that reached the socket are counted; the reader
/// is not advanced, which the caller does.
fn readFileToSocket(
    ev: *Evented,
    region: *CancelRegion,
    file_reader: *File.Reader,
    socket_fd: net.Socket.Handle,
    limit_bytes: usize,
) net.Stream.Writer.WriteFileError!usize {
    const file_fd = file_reader.file.handle;
    var buffer: [16 * 1024]u8 = undefined;
    var total: usize = 0;
    while (total != limit_bytes) {
        const chunk = buffer[0..@min(buffer.len, limit_bytes - total)];
        // A reader that cannot continue fails here rather than at the read, so that the
        // caller's fallback sees the same error it would have seen from the file itself.
        if (file_reader.mode == .failure) return error.ReadFailed;
        const read = readFileChunk(ev, region, file_reader, file_fd, chunk, total) catch |err|
            return readChunkError(err);
        if (read == 0) return total; // the end of the file
        var written: usize = 0;
        while (written != read) {
            try region.check();
            const result = netWriteSync(socket_fd, &.{}, &.{chunk[written..read]}, 1);
            if (result) |n| {
                if (n == 0) return total + written;
                written += n;
            } else |err| {
                switch (err) {
                    error.WouldBlock => try parkFdIndefinitely(ev, region, socket_fd, filt_write),
                    // Bytes that reached the socket are what the caller is told, even when the
                    // one after them failed: it sends the rest from where this stopped.
                    else => |e| return if (total + written == 0) e else total + written,
                }
            }
        }
        total += written;
    }
    return total;
}

/// Moves bytes of the file to the socket, taking the kernel's own copy where the platform has
/// one and copying through a buffer where it does not, or where the kernel refuses this pair of
/// descriptors. The reader's position is where the file's unread bytes begin, so a reader that
/// has buffered some of them continues from there.
fn fileToSocket(
    ev: *Evented,
    region: *CancelRegion,
    file_reader: *File.Reader,
    socket_fd: net.Socket.Handle,
    limit_bytes: usize,
) net.Stream.Writer.WriteFileError!usize {
    if (file_reader.mode == .failure) return error.ReadFailed;
    const unlimited = limit_bytes == @backingInt(Io.Limit.unlimited);
    // The whole file, from its beginning and with no limit to respect, is one `fcopyfile`.
    if (have_fcopyfile and unlimited and file_reader.pos == 0) {
        return fcopyfileToSocket(region, file_reader, socket_fd) catch |err| switch (err) {
            error.Unimplemented => readFileToSocket(ev, region, file_reader, socket_fd, limit_bytes),
            else => |e| return e,
        };
    }
    if (have_sendfile) {
        if (@atomicLoad(Io.Threaded.UseSendfile, &use_sendfile, .monotonic) != .disabled) {
            if (std.math.cast(c.off_t, file_reader.pos)) |offset| {
                return sendfileToSocket(ev, region, file_reader, socket_fd, offset, limit_bytes) catch |err| switch (err) {
                    error.Unimplemented => readFileToSocket(ev, region, file_reader, socket_fd, limit_bytes),
                    else => |e| return e,
                };
            }
        }
    }
    return readFileToSocket(ev, region, file_reader, socket_fd, limit_bytes);
}

/// Sends a header, the bytes the caller's reader has already buffered, and then the file
/// itself, up to `limit` bytes in all. The header and the buffered bytes go through `netWrite`;
/// the file is moved by the kernel where it can be, and copied through a buffer where it
/// cannot. A short write is the count the caller is told, so that the rest is sent from where
/// this stopped.
fn netWriteFile(
    userdata: ?*anyopaque,
    socket_handle: net.Socket.Handle,
    header: []const u8,
    file_reader: *File.Reader,
    limit: Io.Limit,
) net.Stream.Writer.WriteFileError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();

    // The bytes the interface has already read from the file come first, and they take up part
    // of the limit.
    const buffered = limit.slice(file_reader.interface.buffered());
    var total: usize = 0;
    if (header.len != 0 or buffered.len != 0) {
        const n = try netWrite(ev, &region, socket_handle, header, &.{buffered}, 1);
        file_reader.interface.toss(n -| header.len);
        total = n;
        if (n != header.len + buffered.len) return n; // a short write: the caller sends the rest
    }
    const file_limit = @backingInt(limit) -| buffered.len;
    if (file_limit == 0) return total;
    const file_bytes = try fileToSocket(ev, &region, file_reader, socket_handle, file_limit);
    file_reader.pos += file_bytes;
    total += file_bytes;
    if (file_bytes == 0 and total == 0) {
        file_reader.size = file_reader.pos;
        return error.EndOfStream;
    }
    return total;
}

/// Closes every socket. The result of a close is thrown away, since there is nothing left to do
/// about it here, and a descriptor this core owns is always closed by the task that owns it.
fn netClose(userdata: ?*anyopaque, sockets: []const net.Socket) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    for (sockets) |sock| closeAsync(ev, sock.handle);
}

/// Ends one direction of a connection, or both. The call does not wait, and a socket that was
/// shut down in the meantime is not an error the caller is told about here.
fn netShutdown(
    userdata: ?*anyopaque,
    handle: net.Socket.Handle,
    how: net.ShutdownHow,
) net.ShutdownError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    var region: CancelRegion = .init();
    defer region.deinit();
    const posix_how: i32 = switch (how) {
        .recv => posix.SHUT.RD,
        .send => posix.SHUT.WR,
        .both => posix.SHUT.RDWR,
    };
    while (true) {
        try region.check();
        switch (posix.errno(posix.system.shutdown(handle, posix_how))) {
            .SUCCESS => return,
            .INTR => {},
            .NOTCONN => return error.SocketUnconnected,
            .NOBUFS => return error.SystemResources,
            .BADF => |err| return errnoBug(err), // the descriptor was closed under this operation
            .NOTSOCK => |err| return errnoBug(err), // the descriptor is one of this core's sockets
            .INVAL => |err| return errnoBug(err), // the direction this core passed is one of three
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// The C library's "index of an interface by its name", which every Darwin and BSD C library
/// has. `std.c` declares `if_nametoindex` but not `if_indextoname`, so the other direction is
/// declared here; a name is written into a buffer of at least `IFNAMESIZE` bytes, and the
/// pointer it returns is that buffer or null.
extern "c" fn if_indextoname(ifindex: c_uint, ifname: [*]u8) ?[*:0]u8;

/// The index of an interface by its name. The ioctl fallback `Io.Threaded` uses for Linux has
/// nothing to switch to here: `std.c` has neither `ifreq` nor any `SIOCGIF*` for Darwin and the
/// BSDs, and their C libraries have the name lookups the callers want.
fn netInterfaceNameResolve(
    userdata: ?*anyopaque,
    name: *const net.Interface.Name,
) net.Interface.Name.ResolveError!net.Interface {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    var region: CancelRegion = .init();
    defer region.deinit();
    try region.check();
    const index = c.if_nametoindex(&name.bytes);
    if (index != 0) return .{ .index = @bitCast(index) };
    const err: posix.E = @fromBackingInt(@intCast(c._errno().*));
    switch (err) {
        // The name is not one of this machine's interfaces.
        .NXIO, .NODEV, .INVAL => return error.InterfaceNotFound,
        .NOBUFS, .NOMEM => return error.SystemResources,
        .ACCES, .PERM => return error.AccessDenied,
        else => |errno| return unexpectedErrno(errno),
    }
}

/// The name of an interface by its index. The name is at most `Name.max_len` bytes, which is
/// one less than the buffer the C library wants, so a shorter one is asked for first.
fn netInterfaceName(
    userdata: ?*anyopaque,
    interface: net.Interface,
) net.Interface.NameError!net.Interface.Name {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    var region: CancelRegion = .init();
    defer region.deinit();
    try region.check();
    var buffer: [c.IFNAMESIZE]u8 = undefined;
    const named = if_indextoname(@intCast(interface.index), &buffer) orelse {
        const err: posix.E = @fromBackingInt(@intCast(c._errno().*));
        switch (err) {
            .NXIO, .NODEV, .INVAL => return error.InterfaceNotFound,
            .NOBUFS => return error.NameTooLong,
            else => |errno| return unexpectedErrno(errno),
        }
    };
    return try .fromSlice(std.mem.sliceTo(named, 0));
}

/// Turns a host name into addresses, closing the queue when it is done so that the caller's
/// iteration ends. The work is `netLookupFallible`, whose queue is this instance's own `Io`.
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
        error.Closed => unreachable, // the queue is not closed until this call returns
        else => |e| return e,
    };
}

/// The lookup itself: a numeric address is its own answer, `/etc/hosts` is read first, then the
/// RFC 6761 localhost names are answered from the specification, and anything left goes to
/// `/etc/resolv.conf` and the DNS. Every file it reads is read through this instance's `Io`, so
/// a lookup is cancelable and does not hold a worker.
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

    // RFC 6761 Section 6.3.3: a name that is "localhost" or ends in ".localhost" is always the
    // loopback address, and never a question for a resolver.
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

/// Reads `/etc/hosts` through this instance's `Io` and looks the name up in it. A file that
/// cannot be read says the name is unknown rather than failing the lookup, since the name may
/// still be answered from the DNS.
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
            // The queue is where a caller of `lookup` reads diagnostics, so the reason a
            // configuration file could not be read only reaches it as this.
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
            else => return error.DetectingNetworkConfigurationFailed,
        },
        error.Canceled,
        error.Closed,
        error.UnknownHostName,
        => |e| return e,
    };
}

/// Reads into `data` from `file`'s position, as many bytes as the kernel has. A descriptor that
/// is not ready waits for its `EVFILT.READ` event and makes the call again, which is what a
/// terminal or a socket needs; a regular file never waits.
fn fileReadStreaming(
    ev: *Evented,
    region: *CancelRegion,
    file: File,
    data: []const []u8,
) File.ReadStreamingError!usize {
    var iovecs_buffer: [max_iovecs_len]iovec = undefined;
    var i: usize = 0;
    for (data) |buf| {
        if (iovecs_buffer.len - i == 0) break;
        if (buf.len == 0) continue;
        iovecs_buffer[i] = .{ .base = buf.ptr, .len = buf.len };
        i += 1;
    }
    // No room for data is not the end of the stream: it is a read that had nothing to do.
    if (i == 0) return 0;
    const dest = iovecs_buffer[0..i];
    while (true) {
        try region.check();
        const rc = posix.system.readv(file.handle, dest.ptr, @intCast(dest.len));
        switch (posix.errno(rc)) {
            .SUCCESS => return if (rc == 0) error.EndOfStream else @intCast(rc),
            .INTR => continue, // a signal: the call stands
            .AGAIN => {
                // A descriptor that is not ready, a terminal or a socket: wait for the event the
                // kernel has for it and make the call again. A spurious readiness is normal.
                parkFd(ev, region, file.handle, filt_read, .none) catch |err| switch (err) {
                    error.Timeout => unreachable, // no timeout was asked for
                    error.Canceled => |e| return e,
                };
            },
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .IO => return error.InputOutput,
            .ISDIR => return error.IsDir,
            .NOBUFS, .NOMEM => return error.SystemResources,
            .NOTCONN => return error.SocketUnconnected,
            .CONNRESET => return error.ConnectionResetByPeer,
            .ACCES => return error.AccessDenied,
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Writes `header` and then `data` to `file` at its position, `splat` repeating the last buffer
/// of `data`. A descriptor that is not ready to be written waits for its event, as a read does.
fn fileWriteStreaming(
    ev: *Evented,
    region: *CancelRegion,
    file: File,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
) File.Writer.Error!usize {
    var iovecs: [max_iovecs_len]iovec_const = undefined;
    var iovlen: usize = 0;
    addIovec(&iovecs, &iovlen, header);
    for (data[0 .. data.len - 1]) |bytes| addIovec(&iovecs, &iovlen, bytes);
    const pattern = data[data.len - 1];
    var backup_buffer: [splat_buffer_size]u8 = undefined;
    if (iovecs.len - iovlen != 0) switch (splat) {
        0 => {},
        1 => addIovec(&iovecs, &iovlen, pattern),
        else => switch (pattern.len) {
            0 => {},
            1 => {
                const splat_buffer = &backup_buffer;
                const memset_len = @min(splat_buffer.len, splat);
                const buf = splat_buffer[0..memset_len];
                @memset(buf, pattern[0]);
                addIovec(&iovecs, &iovlen, buf);
                var remaining_splat = splat - buf.len;
                while (remaining_splat > splat_buffer.len and iovecs.len - iovlen != 0) {
                    assert(buf.len == splat_buffer.len);
                    addIovec(&iovecs, &iovlen, splat_buffer);
                    remaining_splat -= splat_buffer.len;
                }
                addIovec(&iovecs, &iovlen, splat_buffer[0..@min(remaining_splat, splat_buffer.len)]);
            },
            else => for (0..@min(splat, iovecs.len - iovlen)) |_| {
                addIovec(&iovecs, &iovlen, pattern);
            },
        },
    };
    if (iovlen == 0) return 0;
    while (true) {
        try region.check();
        const rc = posix.system.writev(file.handle, iovecs[0..iovlen].ptr, @intCast(iovlen));
        switch (posix.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => continue, // a signal: the call stands
            .AGAIN => parkFd(ev, region, file.handle, filt_write, .none) catch |err| switch (err) {
                error.Timeout => unreachable, // no timeout was asked for
                error.Canceled => |e| return e,
            },
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .BADF => return error.NotOpenForWriting, // Can be a race condition.
            .DESTADDRREQ => |err| return errnoBug(err), // `connect` was never called.
            .CONNRESET => |err| return errnoBug(err), // Not a socket handle.
            .OVERFLOW => |err| return errnoBug(err), // No offset was given.
            .DQUOT => return error.DiskQuota,
            .FBIG => return error.FileTooBig,
            .IO => return error.InputOutput,
            .NOSPC => return error.NoSpaceLeft,
            .PERM => return error.PermissionDenied,
            .PIPE => return error.BrokenPipe,
            .BUSY => return error.DeviceBusy,
            .TXTBSY => return error.FileBusy,
            .NXIO => return error.NoDevice,
            .ACCES => return error.AccessDenied,
            .NOMEM, .NOBUFS => return error.SystemResources,
            else => |err| return unexpectedErrno(err),
        }
    }
}

/// Adds one buffer to a list of write vectors, if there is room for it. A vector of no bytes is
/// left out: the kernel checks the address of a vector before its length. This is the portable
/// core's `addBuf`, under a name of its own so that this file needs no agreement on one.
fn addIovec(v: []iovec_const, i: *usize, bytes: []const u8) void {
    if (bytes.len == 0) return;
    if (v.len - i.* == 0) return;
    v[i.*] = .{ .base = bytes.ptr, .len = bytes.len };
    i.* += 1;
}

/// Copies from `file_reader` to `file`, at its position, with the header written first. The copy
/// is a plain read and write here, in chunks of whatever the reader has buffered: this core has
/// no `copy_file_range` or `sendfile` to reach for, and a file-to-file copy of a few vectors per
/// chunk is what the loop below does with the reader's own buffer.
fn fileWriteFileStreaming(
    userdata: ?*anyopaque,
    file: File,
    header: []const u8,
    file_reader: *File.Reader,
    limit: Io.Limit,
) File.Writer.WriteFileError!usize {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    var written: usize = 0;
    var copied: usize = 0;
    {
        // What the reader has buffered goes out first, with the header in front of it.
        const buffered = file_reader.interface.buffered();
        if (header.len != 0 or buffered.len != 0) {
            const n = try fileWriteStreaming(ev, &region, file, header, &.{limit.slice(buffered)}, 1);
            file_reader.interface.toss(n -| header.len);
            written = n;
            copied = n -| header.len;
            // A header that did not go out whole leaves the rest to the caller, which still has
            // it: a short write here is not a reason to send what comes after it.
            if (written < header.len) return written;
        }
    }
    // The rest of the file is read through a buffer of this core's own, at the reader's
    // position: the reader's interface has no say in how much is behind its position, and asking
    // it to fill more than its own buffer would be a request it cannot grant.
    var buffer: [16 * 1024]u8 = undefined;
    while (copied < @backingInt(limit)) {
        if (file_reader.mode == .failure) {
            if (written == 0) return error.ReadFailed;
            break;
        }
        if (file_reader.size) |size| if (size == file_reader.pos) break;
        const want = @min(buffer.len, @backingInt(limit) - copied);
        const read_bytes = readFileChunk(
            ev,
            &region,
            file_reader,
            file_reader.file.handle,
            buffer[0..want],
            0,
        ) catch |err| {
            // What went out is this call's result; the caller sees the error next time.
            if (written != 0) break;
            return readChunkError(err);
        };
        if (read_bytes == 0) {
            if (written != 0) break;
            file_reader.size = file_reader.pos;
            return error.EndOfStream;
        }
        var sent: usize = 0;
        while (sent != read_bytes) {
            sent += try fileWriteStreaming(ev, &region, file, &.{}, &.{buffer[sent..read_bytes]}, 1);
        }
        file_reader.pos += read_bytes;
        written += read_bytes;
        copied += read_bytes;
    }
    // Nothing to write at all is the end of the stream; anything written is this call's result.
    if (written == 0) return error.EndOfStream;
    return written;
}

/// `fileWriteFileStreaming` for a writer that writes at `offset` instead of at the file's
/// position. The chunks are the same, written positionally, so that no seek is involved.
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
    var written: usize = 0;
    var copied: usize = 0;
    {
        const buffered = file_reader.interface.buffered();
        if (header.len != 0 or buffered.len != 0) {
            const n = try fileWritePositional(
                userdata,
                file,
                header,
                &.{limit.slice(buffered)},
                1,
                offset,
            );
            file_reader.interface.toss(n -| header.len);
            written = n;
            copied = n -| header.len;
            if (written < header.len) return written;
        }
    }
    var region: CancelRegion = .init();
    defer region.deinit();
    var buffer: [16 * 1024]u8 = undefined;
    while (copied < @backingInt(limit)) {
        if (file_reader.mode == .failure) {
            if (written == 0) return error.ReadFailed;
            break;
        }
        if (file_reader.size) |size| if (size == file_reader.pos) break;
        const want = @min(buffer.len, @backingInt(limit) - copied);
        const read_bytes = readFileChunk(
            ev,
            &region,
            file_reader,
            file_reader.file.handle,
            buffer[0..want],
            0,
        ) catch |err| {
            if (written != 0) break;
            return readChunkError(err);
        };
        if (read_bytes == 0) {
            if (written != 0) break;
            file_reader.size = file_reader.pos;
            return error.EndOfStream;
        }
        var sent: usize = 0;
        while (sent != read_bytes) {
            sent += try fileWritePositional(userdata, file, &.{}, &.{buffer[sent..read_bytes]}, 1, offset + copied);
        }
        file_reader.pos += read_bytes;
        written += read_bytes;
        copied += read_bytes;
    }
    if (written == 0) return error.EndOfStream;
    return written;
}

/// A device control request, made with the caller's own argument. The result is the kernel's,
/// which is a negative errno when the request failed, so no error is returned but a cancelation.
fn deviceIoControl(
    ev: *Evented,
    region: *CancelRegion,
    o: *const Io.Operation.DeviceIoControl,
) Io.Cancelable!i32 {
    _ = ev;
    while (true) {
        try region.check();
        const rc = c.ioctl(o.file.handle, @bitCast(o.code), @intFromPtr(o.arg));
        switch (c.errno(rc)) {
            .SUCCESS => return rc,
            .INTR => {},
            else => |err| return -@as(i32, @backingInt(err)),
        }
    }
}

/// One operation, with the cancel region of the calling task. Each operation makes its own
/// system calls on the worker and waits for the kernel's events where it has to, so a
/// cancelation reaches it at the next call, and every result that has room for `error.Canceled`
/// reports it here instead.
fn operate(userdata: ?*anyopaque, operation: Io.Operation) Io.Cancelable!Io.Operation.Result {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    return switch (operation) {
        .file_read_streaming => |o| .{
            .file_read_streaming = ev.fileReadStreaming(&region, o.file, o.data) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| e,
            },
        },
        .file_write_streaming => |o| .{
            .file_write_streaming = ev.fileWriteStreaming(
                &region,
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
            .device_io_control = try ev.deviceIoControl(&region, &o),
        },
        .net_receive => |o| .{
            .net_receive = r: {
                const opt_err, const n = ev.netReceive(
                    &region,
                    o.socket_handle,
                    o.message_buffer,
                    o.data_buffer,
                    o.flags,
                );
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
                    &region,
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
            .net_read = ev.netRead(&region, o.socket_handle, o.data) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => |e| e,
            },
        },
        .net_write => |o| .{
            .net_write = ev.netWrite(
                &region,
                o.socket_handle,
                o.header,
                o.data,
                o.splat,
            ) catch |err| switch (err) {
                error.Canceled => |e| return e,
                else => unreachable, // `netWrite` only returns `error.Canceled`
            },
        },
    };
}

// Batches
//
// A batch is the batch's own task's to walk: it holds the lists, and the stack of finished
// operations below. What a batch adds to this core is the way an operation that another task
// performs tells the task waiting for it: the finished operation is linked onto
// `Io.Batch.userdata`, and a task that published itself there while it waits is made runnable
// again. The low two bits of that word say what it holds, and the four cases are:
//
// * clear, with no pointer: nothing has finished and no task is waiting;
// * clear, with a task: that task is waiting for the batch, and a completion wakes it;
// * `batch_node`: the head of a stack of finished operations, linked through the first word of
//   each operation's storage, each tagged with the same bits;
// * `batch_timeout`: the marker a deadline leaves when it passes, which may be set alongside a
//   stack of finished operations.

/// A finished operation on the stack of `Io.Batch.userdata`.
const batch_node: usize = 0b10;

/// The marker a deadline leaves when the time it waited for has passed.
const batch_timeout: usize = 0b01;

/// The words of `Io.Operation.Storage.Pending.Userdata` a batched operation uses: the link of
/// the stack of finished operations, the task that performs the operation, and whether that
/// task was canceled, which takes the operation out of the iteration.
const batch_link_word = 0;
const batch_task_word = 1;
const batch_canceled_word = 2;

/// Waits for at least one of the batch's operations to finish. Each submitted operation is
/// performed on this task, one at a time, so no concurrency of any kind is needed for this, and
/// an operation left in flight by an earlier `awaitConcurrent` is waited for where it is.
fn batchAwaitAsync(userdata: ?*anyopaque, batch: *Io.Batch) Io.Cancelable!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    try batchRunSubmitted(userdata, &region, batch);
    while (true) {
        try region.check();
        batchDrainReady(ev, batch) catch |err| switch (err) {
            error.Timeout => {}, // no deadline belongs to this wait
        };
        if (batch.completed.head != .none or batch.pending.head == .none) return;
        // Nothing has finished yet and something is in flight, which another await started: wait
        // for the task that performs it to finish it.
        ev.sched.yield(null, .{ .custom = .{ .context = batch, .run = batchAwaitPending } });
    }
}

/// Waits for at least one of the batch's operations to finish, or for the deadline to pass.
/// Every submitted operation is performed on a task of its own as this is called, so they are
/// all in flight together.
fn batchAwaitConcurrent(
    userdata: ?*anyopaque,
    batch: *Io.Batch,
    timeout: Io.Timeout,
) Io.Batch.AwaitConcurrentError!void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var region: CancelRegion = .init();
    defer region.deinit();
    try batchSubmit(ev, &region, batch);
    batchDrainReady(ev, batch) catch |err| switch (err) {
        error.Timeout => {}, // a deadline of an earlier wait: nothing has finished
    };
    if (batch.completed.head != .none or batch.pending.head == .none) return;
    // The deadline is a task of its own, waiting for the time to pass while this task waits for
    // the operations: whichever of them finishes first is what ends the wait. A task the
    // scheduler will not make leaves the operations in flight, for the caller to cancel.
    var deadline: ?Io.Future(u8) = null;
    if (timeout != .none) {
        deadline = ev.sched.concurrentWith(.{}, batchDeadline, .{ ev, batch, timeout }) catch |err| return err;
    }
    defer if (deadline) |*future| {
        // This wait is over, so the deadline's is too, and whatever it left on the batch is
        // taken off here: a later wait must not be ended by a deadline of this one.
        _ = future.cancel(ev.io());
        batchDrainReady(ev, batch) catch |err| switch (err) {
            error.Timeout => {},
        };
    };
    while (true) {
        try region.check();
        batchDrainReady(ev, batch) catch |err| switch (err) {
            error.Timeout => |e| if (batch.completed.head == .none and batch.pending.head != .none) return e,
        };
        if (batch.completed.head != .none or batch.pending.head == .none) return;
        // Park until an operation finishes, or until the deadline's task says the time is up.
        ev.sched.yield(null, .{ .custom = .{ .context = batch, .run = batchAwaitPending } });
    }
}

/// Cancels every operation the batch has in flight, and waits for each one to finish, so that
/// the batch is ready to be iterated or reused when this returns. What had already finished is
/// left for the caller to iterate; an operation that finished before the cancel request reached
/// it is there too, and one that was canceled is absent.
fn batchCancel(userdata: ?*anyopaque, batch: *Io.Batch) void {
    charge(userdata);
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    batchDrainReady(ev, batch) catch |err| switch (err) {
        error.Timeout => {}, // a deadline that has no operation to cancel with it
    };
    var index = batch.pending.head;
    while (index != .none) {
        const pending = &batch.storage[index.toIndex()].pending;
        index = pending.node.next; // the list is walked as the tasks end, so it is read first
        const raw = pending.userdata[batch_task_word];
        // A pending operation was started, and the task that performs it is still to be freed.
        assert(raw != 0);
        var future: Io.Future(Io.Operation.Result) = .{
            .any_future = @ptrFromInt(raw),
            .result = undefined,
        };
        _ = future.cancel(ev.io());
    }
    // Every operation that was canceled pushed its completion as it ended: take them all off, so
    // that nothing is in flight and nothing is left on the stack, which is the state a ready
    // batch is in.
    batchDrainReady(ev, batch) catch |err| switch (err) {
        error.Timeout => {},
    };
    batch.userdata = null;
}

/// Runs the operations that are waiting to be submitted, each on its own task, so that they are
/// all in flight at once. An operation this cannot be done for is left where the caller put it,
/// and nothing else is started after it.
fn batchSubmit(
    ev: *Evented,
    region: *CancelRegion,
    batch: *Io.Batch,
) (Io.ConcurrentError || Io.Cancelable)!void {
    var index = batch.submitted.head;
    if (index == .none) return;
    // The first submission not yet taken off the list: an error leaves it, and the ones after
    // it, submitted, where `Batch.cancel` and `Batch.addAt` expect to find them.
    errdefer batch.submitted.head = index;
    while (index != .none) {
        try region.check();
        const storage = &batch.storage[index.toIndex()];
        const next_index = storage.submission.node.next;
        const operation = storage.submission.operation;
        switch (operation) {
            // With no buffer to read into or write from there is nothing to wait for, and the
            // result is this task's own: the operation never becomes one that is in flight.
            .file_read_streaming => |o| if (nothing: {
                for (o.data) |buffer| if (buffer.len != 0) break :nothing false;
                break :nothing true;
            }) {
                batchCompleteNow(batch, storage, index, .{ .file_read_streaming = 0 });
                index = next_index;
                continue;
            },
            .file_write_streaming => |o| if (nothing: {
                if (o.header.len != 0) break :nothing false;
                for (o.data[0 .. o.data.len - 1]) |buffer| if (buffer.len != 0) break :nothing false;
                if (o.splat == 0) break :nothing true;
                break :nothing o.data[o.data.len - 1].len == 0;
            }) {
                batchCompleteNow(batch, storage, index, .{ .file_write_streaming = 0 });
                index = next_index;
                continue;
            },
            // An ioctl is made on the worker, and one that has started cannot be cancelled, so a
            // batch whose operations have to be in flight cannot take one.
            .device_io_control => return error.ConcurrencyUnavailable,
            else => {},
        }
        // The task cannot run before this one switches away, so the record of the operation is
        // complete before anything reads it: see `batchOperation`.
        const future = ev.sched.concurrentWith(.{}, batchOperation, .{ ev, batch, index.toIndex(), operation }) catch |err| return err;
        storage.* = .{ .pending = .{
            .node = .{ .prev = batch.pending.tail, .next = .none },
            .tag = std.meta.activeTag(operation),
            .userdata = undefined,
        } };
        const node = &storage.pending.userdata;
        node[batch_link_word] = @intFromPtr(batch); // until a completion links it on the stack
        node[batch_task_word] = @intFromPtr(future.any_future.?);
        switch (batch.pending.tail) {
            .none => batch.pending.head = index,
            else => |tail_index| batch.storage[tail_index.toIndex()].pending.node.next = index,
        }
        batch.pending.tail = index;
        index = next_index;
    }
    batch.submitted = .{ .head = .none, .tail = .none };
}

/// Moves an operation that needed no waiting from the submission list to the completed list,
/// with the result it has already.
fn batchCompleteNow(
    batch: *Io.Batch,
    storage: *Io.Operation.Storage,
    index: Io.Operation.OptionalIndex,
    result: Io.Operation.Result,
) void {
    switch (batch.completed.tail) {
        .none => batch.completed.head = index,
        else => |tail_index| batch.storage[tail_index.toIndex()].completion.node.next = index,
    }
    storage.* = .{ .completion = .{ .node = .{ .next = .none }, .result = result } };
    batch.completed.tail = index;
}

/// One operation of a batch, on a task of its own, so that the batch's operations are in flight
/// together. A canceled operation is marked in its storage and its result is not read.
fn batchOperation(
    ev: *Evented,
    batch: *Io.Batch,
    index: u32,
    operation: Io.Operation,
) Io.Operation.Result {
    const result: ?Io.Operation.Result = operate(@as(?*anyopaque, ev), operation) catch |err| switch (err) {
        error.Canceled => null,
    };
    const node = &batch.storage[index].pending.userdata;
    // The push publishes the mark with the node, so the task that drains it sees both.
    node[batch_canceled_word] = @intFromBool(result == null);
    batchPush(ev, batch, node);
    return result orelse undefined;
}

/// Links a finished operation onto the batch's stack, and makes the task waiting for the batch
/// runnable if it had published itself there: it was replaced by this operation's node.
fn batchPush(ev: *Evented, batch: *Io.Batch, node: *Io.Operation.Storage.Pending.Userdata) void {
    node[batch_link_word] = 0b00; // the end of the stack, until the exchange below says more
    const push = @intFromPtr(node) | batch_node;
    var next: usize = 0b00;
    while (true) {
        next = @cmpxchgWeak(
            usize,
            @as(*usize, @ptrCast(&batch.userdata)),
            next,
            push,
            .release,
            .acquire,
        ) orelse break;
        node[batch_link_word] = next;
    }
    switch (@as(u2, @truncate(next))) {
        0b00 => if (next != 0) ev.sched.ready(Scheduler.Worker.current(), @ptrFromInt(next)),
        // A deadline passed, or a completion is already on the stack: the task waiting for the
        // batch was made runnable by whichever of them came first.
        batch_timeout, batch_node, batch_timeout | batch_node => {},
    }
}

/// Runs once the task waiting for a batch has switched away: it publishes itself as the task a
/// completion makes runnable, and re-runs itself at once when one has finished already, since
/// nothing else would.
fn batchAwaitPending(s: *Scheduler, task: *Fiber, context: *anyopaque) void {
    const batch: *Io.Batch = @ptrCast(@alignCast(context));
    if (@cmpxchgStrong(?*anyopaque, &batch.userdata, null, task, .release, .monotonic)) |head| {
        assert(@as(u2, @truncate(@intFromPtr(head))) != 0b00); // never this task itself
        s.ready(Scheduler.Worker.current(), task);
    }
}

/// The deadline of one wait, on a task of its own: it waits for the time to pass and then marks
/// the batch, which is what makes the task waiting for the batch runnable when no operation has
/// finished. Its result says nothing.
fn batchDeadline(ev: *Evented, batch: *Io.Batch, timeout: Io.Timeout) u8 {
    var region: CancelRegion = .init();
    defer region.deinit();
    parkTimeout(ev, &region, timeout) catch |err| switch (err) {
        // The wait ended first: this deadline has nothing to say about the batch.
        error.Canceled => return 0,
    };
    const previous = @atomicRmw(
        usize,
        @as(*usize, @ptrCast(&batch.userdata)),
        .Add,
        batch_timeout,
        .acquire,
    );
    switch (@as(u2, @truncate(previous))) {
        batch_timeout, batch_timeout | batch_node => unreachable, // one wait has one deadline
        // A task was waiting, and its task is the one this makes runnable: it finds the marker
        // where it is about to look for a completion.
        0b00 => if (previous != 0) ev.sched.ready(Scheduler.Worker.current(), @ptrFromInt(previous)),
        // An operation has finished already, and its completion made the task runnable.
        batch_node => {},
    }
    return 0;
}

/// Takes every finished operation off the batch's stack and puts it in the completed list, or
/// back in the unused list when it was canceled, and frees the task that performed it. Returns
/// `error.Timeout` when a deadline marker is found, which is the end of the wait whether or not
/// an operation finished in the same step.
fn batchDrainReady(ev: *Evented, batch: *Io.Batch) Io.Timeout.Error!void {
    while (@atomicRmw(?*anyopaque, &batch.userdata, .Xchg, null, .acquire)) |head| {
        var next: usize = @intFromPtr(head);
        var timeout = false;
        while (cond: switch (@as(u2, @truncate(next))) {
            0b00 => if (timeout) return error.Timeout else false, // the end of the stack
            batch_timeout => {
                assert(!timeout);
                return error.Timeout;
            },
            batch_node => true,
            batch_timeout | batch_node => {
                assert(!timeout);
                timeout = true;
                break :cond true;
            },
        }) {
            const node: *Io.Operation.Storage.Pending.Userdata =
                @ptrFromInt(next & ~(batch_node | batch_timeout));
            next = node[batch_link_word];
            const pending: *Io.Operation.Storage.Pending = @fieldParentPtr("userdata", node);
            const storage: *Io.Operation.Storage = @fieldParentPtr("pending", pending);
            const index: Io.Operation.OptionalIndex = .fromIndex(storage - batch.storage.ptr);
            switch (pending.node.prev) {
                .none => batch.pending.head = pending.node.next,
                else => |prev_index| batch.storage[prev_index.toIndex()].pending.node.next =
                    pending.node.next,
            }
            switch (pending.node.next) {
                .none => batch.pending.tail = pending.node.prev,
                else => |next_index| batch.storage[next_index.toIndex()].pending.node.prev =
                    pending.node.prev,
            }
            const completed: ?Io.Operation.Result = if (node[batch_canceled_word] != 0)
                null
            else
                batchReap(ev, node);
            if (completed) |result| {
                switch (batch.completed.tail) {
                    .none => batch.completed.head = index,
                    else => |tail_index| batch.storage[tail_index.toIndex()].completion.node.next =
                        index,
                }
                storage.* = .{ .completion = .{ .node = .{ .next = .none }, .result = result } };
                batch.completed.tail = index;
            } else {
                // A canceled operation is absent from the iteration, and its storage is free.
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

/// Takes a finished operation's result out of the task that performed it and frees that task.
/// The task is done, or about to be: it pushed its completion before it returned.
fn batchReap(ev: *Evented, node: *Io.Operation.Storage.Pending.Userdata) ?Io.Operation.Result {
    const raw = node[batch_task_word];
    if (raw == 0) return null; // no task was started: there is no result to take
    var future: Io.Future(Io.Operation.Result) = .{
        .any_future = @ptrFromInt(raw),
        .result = undefined,
    };
    return future.await(ev.io());
}

/// Runs the operations waiting to be submitted on this task, one at a time, each one to
/// completion, and puts each in the completed list. An operation that cannot be performed
/// leaves the rest of them submitted, for the next await or for `cancel`.
fn batchRunSubmitted(
    userdata: ?*anyopaque,
    region: *CancelRegion,
    batch: *Io.Batch,
) Io.Cancelable!void {
    var index = batch.submitted.head;
    if (index == .none) return;
    errdefer batch.submitted.head = index;
    var tail_index = batch.completed.tail;
    defer batch.completed.tail = tail_index;
    while (index != .none) {
        try region.check();
        const storage = &batch.storage[index.toIndex()];
        const submission = &storage.submission;
        const next_index = submission.node.next;
        const result = try operate(userdata, submission.operation);
        switch (tail_index) {
            .none => batch.completed.head = index,
            else => |last| batch.storage[last.toIndex()].completion.node.next = index,
        }
        storage.* = .{ .completion = .{ .node = .{ .next = .none }, .result = result } };
        tail_index = index;
        index = next_index;
    }
    batch.submitted = .{ .head = .none, .tail = .none };
}

/// The path of an open descriptor, as `Io.Threaded.realPathPosix` gets it: Darwin and DragonFly
/// ask `fcntl` for it, FreeBSD asks for the `kinfo_file` of the descriptor, and the rest of the
/// BSDs have no kernel call for it at all.
fn realPath(ev: *Evented, fd: c.fd_t, out_buffer: []u8) File.RealPathError!usize {
    _ = ev;
    switch (builtin.os.tag) {
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos, .dragonfly => {
            var buffer: [c.PATH_MAX]u8 = undefined;
            @memset(&buffer, 0);
            while (true) {
                switch (c.errno(c.fcntl(fd, c.F.GETPATH, &buffer))) {
                    .SUCCESS => break,
                    .INTR => {},
                    .ACCES => return error.AccessDenied,
                    .BADF => return error.FileNotFound,
                    .NOENT => return error.FileNotFound,
                    .NOMEM => return error.SystemResources,
                    .NOSPC => return error.NameTooLong,
                    .RANGE => return error.NameTooLong,
                    else => |err| return unexpectedErrno(err),
                }
            }
            const n = std.mem.findScalar(u8, &buffer, 0) orelse buffer.len;
            if (n > out_buffer.len) return error.NameTooLong;
            @memcpy(out_buffer[0..n], buffer[0..n]);
            return n;
        },
        .freebsd => {
            var k_file: c.kinfo_file = undefined;
            k_file.structsize = c.KINFO_FILE_SIZE;
            while (true) {
                switch (c.errno(c.fcntl(fd, c.F.KINFO, @intFromPtr(&k_file)))) {
                    .SUCCESS => break,
                    .INTR => {},
                    .BADF => return error.FileNotFound,
                    else => |err| return unexpectedErrno(err),
                }
            }
            const n = std.mem.findScalar(u8, &k_file.path, 0) orelse k_file.path.len;
            if (n == 0 or n > out_buffer.len) return error.NameTooLong;
            @memcpy(out_buffer[0..n], k_file.path[0..n]);
            return n;
        },
        // NetBSD and OpenBSD offer no way to name an open descriptor.
        else => return error.OperationUnsupported,
    }
}

fn fchown(fd: c.fd_t, owner: ?File.Uid, group: ?File.Gid) File.SetOwnerError!void {
    const uid = owner orelse std.math.maxInt(c.uid_t);
    const gid = group orelse std.math.maxInt(c.gid_t);
    while (true) switch (c.errno(c.fchown(fd, uid, gid))) {
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
    };
}

fn lseek(ev: *Evented, fd: c.fd_t, offset: u64, whence: i32) File.SeekError!void {
    _ = ev;
    while (true) switch (c.errno(c.lseek(fd, @bitCast(offset), whence))) {
        .SUCCESS => return,
        .INTR => {},
        .BADF => |err| return errnoBug(err), // File descriptor used after closed.
        .INVAL => return error.Unseekable,
        .OVERFLOW => return error.Unseekable,
        .SPIPE => return error.Unseekable,
        .NXIO => return error.Unseekable,
        else => |err| return unexpectedErrno(err),
    };
}

fn fchmod(ev: *Evented, fd: c.fd_t, mode: c.mode_t) File.SetPermissionsError!void {
    _ = ev;
    while (true) switch (c.errno(c.fchmod(fd, mode))) {
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
        .PERM => return error.PermissionDenied,
        .ROFS => return error.ReadOnlyFileSystem,
        else => |err| return unexpectedErrno(err),
    };
}

fn linkat(
    old_dir: c.fd_t,
    old_path: [*:0]const u8,
    new_dir: c.fd_t,
    new_path: [*:0]const u8,
    flags: u32,
) File.HardLinkError!void {
    while (true) switch (c.errno(c.linkat(old_dir, old_path, new_dir, new_path, flags))) {
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
    };
}

fn addConstBuf(v: []iovec_const, i: *iovlen_t, remaining: ?*usize, bytes: []const u8) void {
    if (v.len - i.* == 0) return;
    const len = @min(remaining.*, bytes.len);
    if (len == 0) return;
    v[i.*] = .{ .base = bytes.ptr, .len = len };
    i.* += 1;
    remaining.* -= len;
}

fn addBuf(
    comptime is_const: bool,
    vec: []if (is_const) iovec_const else iovec,
    vec_len: *iovlen_t,
    remaining: *Io.Limit,
    bytes: if (is_const) []const u8 else []u8,
) void {
    if (vec.len - vec_len.* == 0) return;
    const len = remaining.minInt(bytes.len);
    if (len == 0) return;
    vec[vec_len.*] = .{ .base = bytes.ptr, .len = len };
    vec_len.* += 1;
    remaining.* = remaining.subtract(len).?;
}

test {
    _ = Fiber.CancelProtection;
}

/// `Io.Threaded.pipe2`: the kernel's atomic `pipe2` where the platform has one, and a `pipe()`
/// with `F.SETFD`/`F.SETFL` where it does not (Darwin). The BSDs get the atomic call, so the
/// descriptors carry `CLOEXEC` before another thread can fork.
fn pipe2(flags: c.O) PipeError![2]c.fd_t {
    return Io.Threaded.pipe2(flags);
}

fn dup2(ev: *Evented, old_fd: c.fd_t, new_fd: c.fd_t) DupError!void {
    _ = ev;
    while (true) switch (c.errno(c.dup2(old_fd, new_fd))) {
        .SUCCESS => return,
        .BUSY, .INTR => {},
        .INVAL => |err| return errnoBug(err), // invalid parameters
        .BADF => |err| return errnoBug(err), // use after free
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NOMEM => return error.SystemResources,
        else => |err| return unexpectedErrno(err),
    };
}

fn execv(
    ev: *Evented,
    arg0_expand: process.ArgExpansion,
    file: [*:0]const u8,
    child_argv: [*:null]?[*:0]const u8,
    env_block: process.Environ.PosixBlock,
    PATH: []const u8,
) process.ReplaceError {
    const file_slice = std.mem.sliceTo(file, 0);
    if (std.mem.findScalar(u8, file_slice, '/') != null) return ev.execvPath(file, child_argv, env_block);

    // Use of PATH_MAX here is valid as the path_buf will be passed
    // directly to the operating system in posixExecvPath.
    var path_buf: [c.PATH_MAX]u8 = undefined;
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
        err = ev.execvPath(full_path, child_argv, env_block);
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
fn execvPath(
    ev: *Evented,
    path: [*:0]const u8,
    child_argv: [*:null]const ?[*:0]const u8,
    env_block: process.Environ.PosixBlock,
) process.ReplaceError {
    _ = ev;
    switch (c.errno(c.execve(path, child_argv, env_block.slice.ptr))) {
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
        else => |err| switch (builtin.os.tag) {
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => switch (err) {
                .BADEXEC => return error.InvalidExe,
                .BADARCH => return error.InvalidExe,
                else => return unexpectedErrno(err),
            },
            else => return unexpectedErrno(err),
        },
    }
}

fn spawn(ev: *Evented, options: process.SpawnOptions) process.SpawnError!Spawned {
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
    const pipe_flags: c.O = .{ .CLOEXEC = true };

    const stdin_pipe = if (options.stdin == .pipe) try pipe2(pipe_flags) else undefined;
    errdefer if (options.stdin == .pipe) {
        destroyPipe(stdin_pipe);
    };

    const stdout_pipe = if (options.stdout == .pipe) try pipe2(pipe_flags) else undefined;
    errdefer if (options.stdout == .pipe) {
        destroyPipe(stdout_pipe);
    };

    const stderr_pipe = if (options.stderr == .pipe) try pipe2(pipe_flags) else undefined;
    errdefer if (options.stderr == .pipe) {
        destroyPipe(stderr_pipe);
    };

    const any_ignore =
        options.stdin == .ignore or options.stdout == .ignore or options.stderr == .ignore;
    const dev_null_file = if (any_ignore) dev_null_file: {
        break :dev_null_file try openDevNullFile(ev);
    } else undefined;

    const prog_pipe: [2]c.fd_t = if (options.progress_node.index != .none)
        // We use CLOEXEC for the same reason as in `pipe_flags`.
        try pipe2(.{ .NONBLOCK = true, .CLOEXEC = true })
    else
        .{ -1, -1 };
    errdefer destroyPipe(prog_pipe);

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
    const err_pipe: [2]File = err_pipe: {
        const err_pipe = try pipe2(.{ .CLOEXEC = true });
        break :err_pipe .{
            .{ .handle = err_pipe[0], .flags = .{ .nonblocking = false } },
            .{ .handle = err_pipe[1], .flags = .{ .nonblocking = false } },
        };
    };
    errdefer fileClose(ev, &err_pipe);

    ev.scanEnviron() catch |err| switch (err) {
        error.Canceled => return error.Canceled,
    }; // for PATH
    const PATH = ev.environ.string.PATH orelse default_PATH;

    const pid_result: c.pid_t = fork: {
        const rc = c.fork();
        switch (c.errno(rc)) {
            .SUCCESS => break :fork @intCast(rc),
            .AGAIN => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .NOSYS => return error.OperationUnsupported,
            else => |err| return unexpectedErrno(err),
        }
    };

    if (pid_result == 0) {
        defer comptime unreachable; // We are the child.
        const err = ev.setUpChild(.{
            .stdin_pipe = stdin_pipe[0],
            .stdout_pipe = stdout_pipe[1],
            .stderr_pipe = stderr_pipe[1],
            .dev_null_fd = dev_null_file.handle,
            .prog_pipe = prog_pipe[1],
            .argv_buf = argv_buf,
            .env_block = env_block,
            .PATH = PATH,
            .spawn = options,
        });
        ev.writeAll(err_pipe[1], @ptrCast(&err)) catch {};
        c.exit(1);
    }

    const pid: c.pid_t = @intCast(pid_result); // We are the parent.
    errdefer comptime unreachable; // The child is forked; we must not error from now on

    fileClose(ev, err_pipe[1..2]); // make sure only the child holds the write end open

    if (options.stdin == .pipe) closeFd(stdin_pipe[0]);
    if (options.stdout == .pipe) closeFd(stdout_pipe[1]);
    if (options.stderr == .pipe) closeFd(stderr_pipe[1]);

    if (prog_pipe[1] != -1) closeFd(prog_pipe[1]);

    options.progress_node.setIpcFile(ev, .{ .handle = prog_pipe[0], .flags = .{ .nonblocking = true } });

    return .{
        .pid = pid,
        .err_pipe = err_pipe[0],
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

fn setUpChild(ev: *Evented, options: struct {
    stdin_pipe: c.fd_t,
    stdout_pipe: c.fd_t,
    stderr_pipe: c.fd_t,
    dev_null_fd: c.fd_t,
    prog_pipe: c.fd_t,
    argv_buf: [:null]?[*:0]const u8,
    env_block: process.Environ.Block,
    PATH: []const u8,
    spawn: process.SpawnOptions,
}) ForkBailError {
    try ev.setUpChildIo(
        options.spawn.stdin,
        options.stdin_pipe,
        c.STDIN_FILENO,
        options.dev_null_fd,
    );
    try ev.setUpChildIo(
        options.spawn.stdout,
        options.stdout_pipe,
        c.STDOUT_FILENO,
        options.dev_null_fd,
    );
    try ev.setUpChildIo(
        options.spawn.stderr,
        options.stderr_pipe,
        c.STDERR_FILENO,
        options.dev_null_fd,
    );

    switch (options.spawn.cwd) {
        .inherit => {},
        .dir => |cwd_dir| try processSetCurrentDir(ev, cwd_dir),
        .path => |cwd_path| try processSetCurrentPath(ev, cwd_path),
    }

    // Must happen after fchdir above, the cwd file descriptor might be
    // equal to prog_fileno and be clobbered by this dup2 call.
    if (options.prog_pipe != -1) try ev.dup2(options.prog_pipe, prog_fileno);

    if (options.spawn.gid) |gid| while (true) switch (c.errno(c.setregid(gid, gid))) {
        .SUCCESS => break,
        .INTR => {},
        .AGAIN => return error.ResourceLimitReached,
        .INVAL => return error.InvalidUserId,
        .PERM => return error.PermissionDenied,
        else => return error.Unexpected,
    };

    if (options.spawn.uid) |uid| while (true) switch (c.errno(c.setreuid(uid, uid))) {
        .SUCCESS => break,
        .INTR => {},
        .AGAIN => return error.ResourceLimitReached,
        .INVAL => return error.InvalidUserId,
        .PERM => return error.PermissionDenied,
        else => return error.Unexpected,
    };

    if (options.spawn.pgid) |pid| while (true) switch (c.errno(c.setpgid(0, pid))) {
        .SUCCESS => break,
        .INTR => {},
        .ACCES => return error.ProcessAlreadyExec,
        .INVAL => return error.InvalidProcessGroupId,
        .PERM => return error.PermissionDenied,
        else => return error.Unexpected,
    };

    if (options.spawn.start_suspended) while (true) switch (c.errno(c.kill(0, .STOP))) {
        .SUCCESS => break,
        .INTR => {},
        .PERM => return error.PermissionDenied,
        else => return error.Unexpected,
    };

    return ev.execv(
        options.spawn.expand_arg0,
        options.argv_buf.ptr[0].?,
        options.argv_buf.ptr,
        options.env_block,
        options.PATH,
    );
}

fn setUpChildIo(
    ev: *Evented,
    stdio: process.SpawnOptions.StdIo,
    pipe_fd: c.fd_t,
    std_fileno: i32,
    dev_null_fd: c.fd_t,
) !void {
    switch (stdio) {
        .pipe => try ev.dup2(pipe_fd, std_fileno),
        .close => closeFd(std_fileno),
        .inherit => {},
        .ignore => try ev.dup2(dev_null_fd, std_fileno),
        .file => |file| try ev.dup2(file.handle, std_fileno),
    }
}

const PipeError = error{
    SystemFdQuotaExceeded,
    ProcessFdQuotaExceeded,
} || Io.UnexpectedError;

fn destroyPipe(pipe: [2]c.fd_t) void {
    if (pipe[0] != -1) closeFd(pipe[0]);
    if (pipe[0] != pipe[1]) closeFd(pipe[1]);
}

const DupError = error{
    ProcessFdQuotaExceeded,
    SystemResources,
} || Io.UnexpectedError || Io.Cancelable;

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

const prog_fileno = @max(c.STDIN_FILENO, c.STDOUT_FILENO, c.STDERR_FILENO) + 1;

const Spawned = struct {
    pid: c.pid_t,
    err_pipe: File,
    stdin: ?File,
    stdout: ?File,
    stderr: ?File,
};

const ForkBailError = process.SetCurrentDirError || ChdirError ||
    process.SpawnError || process.ReplaceError;

fn dirCreateDir(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    permissions: Dir.Permissions,
) Dir.CreateDirError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    while (true) {
        switch (c.errno(c.mkdirat(dir.handle, sub_path_posix, permissions.toMode()))) {
            .SUCCESS => return,
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .PERM => return error.PermissionDenied,
            .DQUOT => return error.DiskQuota,
            .EXIST => return error.PathAlreadyExists,
            .LOOP => return error.SymLinkLoop,
            .MLINK => return error.LinkQuotaExceeded,
            .NAMETOOLONG => return error.NameTooLong,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .NOTDIR => return error.NotDir,
            .ROFS => return error.ReadOnlyFileSystem,
            .ILSEQ => return error.BadPathName,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .FAULT => |err| return errnoBug(err),
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
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var it = Dir.path.componentIterator(sub_path);
    var status: Dir.CreatePathStatus = .existed;
    var component = it.last() orelse return error.BadPathName;
    while (true) {
        if (dirCreateDir(ev, dir, component.path, permissions)) |_| {
            status = .created;
        } else |err| switch (err) {
            error.PathAlreadyExists => {
                // It is important to return an error if it's not a directory
                // because otherwise a dangling symlink could cause an infinite
                // loop.
                const fstat = try dirStatFile(ev, dir, component.path, .{});
                if (fstat.kind != .directory) return error.NotDir;
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

fn dirCreateDirPathOpen(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    permissions: Dir.Permissions,
    options: Dir.OpenOptions,
) Dir.CreateDirPathOpenError!Dir {
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
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    var flags: c.O = .{
        .ACCMODE = .RDONLY,
        .NOFOLLOW = !options.follow_symlinks,
        .DIRECTORY = true,
        .CLOEXEC = true,
    };

    // `Io.Threaded.dirOpenDirPosix`: a caller that promises not to iterate gets a bare path
    // descriptor where the platform has `O.PATH`, which on FreeBSD opens a directory the process
    // may not read. `openat`, `fstatat` and `fchdir` are allowed on such a descriptor, and
    // `Dir.Reader` is what iteration requires.
    if (@hasField(c.O, "PATH") and !options.iterate) flags.PATH = true;

    while (true) {
        const rc = c.openat(dir.handle, sub_path_posix, flags);
        switch (c.errno(rc)) {
            .SUCCESS => return .{ .handle = @intCast(rc) },
            .INTR => {},
            .INVAL => return error.BadPathName,
            .ACCES => return error.AccessDenied,
            .LOOP => return error.SymLinkLoop,
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NAMETOOLONG => return error.NameTooLong,
            .NFILE => return error.SystemFdQuotaExceeded,
            .NODEV => return error.NoDevice,
            .NOENT => return error.FileNotFound,
            .NOMEM => return error.SystemResources,
            .NOTDIR => return error.NotDir,
            .PERM => return error.PermissionDenied,
            .NXIO => return error.NoDevice,
            .ILSEQ => return error.BadPathName,
            .FAULT => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .BUSY => |err| return errnoBug(err), // O_EXCL not passed
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirStat(userdata: ?*anyopaque, dir: Dir) Dir.StatError!Dir.Stat {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return fileStat(ev, .{
        .handle = dir.handle,
        .flags = .{ .nonblocking = false },
    });
}

fn dirStatFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    options: Dir.StatFileOptions,
) Dir.StatFileError!File.Stat {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    const flags: u32 = if (options.follow_symlinks) 0 else c.AT.SYMLINK_NOFOLLOW;

    while (true) {
        var stat = std.mem.zeroes(c.Stat);
        switch (c.errno(c.fstatat(dir.handle, sub_path_posix, &stat, flags))) {
            .SUCCESS => return statFromPosix(&stat),
            .INTR => {},
            .INVAL => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .NOMEM => return error.SystemResources,
            .ACCES => return error.AccessDenied,
            .PERM => return error.PermissionDenied,
            .FAULT => |err| return errnoBug(err),
            .NAMETOOLONG => return error.NameTooLong,
            .LOOP => return error.SymLinkLoop,
            .NOENT => return error.FileNotFound,
            .NOTDIR => return error.FileNotFound,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn dirAccess(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    options: Dir.AccessOptions,
) Dir.AccessError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    const flags: u32 = if (options.follow_symlinks) 0 else c.AT.SYMLINK_NOFOLLOW;

    const mode: u32 =
        @as(u32, if (options.read) c.R_OK else 0) |
        @as(u32, if (options.write) c.W_OK else 0) |
        @as(u32, if (options.execute) c.X_OK else 0);

    while (true) switch (c.errno(c.faccessat(dir.handle, sub_path_posix, mode, flags))) {
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
    };
}

fn dirCreateFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    flags: Dir.CreateFileOptions,
) File.OpenError!File {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    const os_flags: c.O = .{
        .ACCMODE = if (flags.read) .RDWR else .WRONLY,
        .NONBLOCK = flags.lock == .none or flags.lock_nonblocking,
        .SHLOCK = flags.lock == .shared,
        .EXLOCK = flags.lock == .exclusive,
        .CREAT = true,
        .TRUNC = flags.truncate,
        .EXCL = flags.exclusive,
        .CLOEXEC = true,
    };

    const fd: c.fd_t = while (true) {
        const rc = c.openat(dir.handle, sub_path_posix, os_flags, flags.permissions.toMode());
        switch (c.errno(rc)) {
            .SUCCESS => break @intCast(rc),
            .INTR => {},
            .FAULT => |err| return errnoBug(err),
            .INVAL => return error.BadPathName,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
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
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .NOTDIR => return error.NotDir,
            .PERM => return error.PermissionDenied,
            .EXIST => return error.PathAlreadyExists,
            .BUSY => return error.DeviceBusy,
            .OPNOTSUPP => return error.FileLocksUnsupported,
            .AGAIN => return error.WouldBlock,
            .TXTBSY => return error.FileBusy,
            .ROFS => return error.ReadOnlyFileSystem,
            .NXIO => return error.NoDevice,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    };
    errdefer closeFd(fd);

    return .{
        .handle = fd,
        .flags = .{ .nonblocking = os_flags.NONBLOCK },
    };
}

fn dirCreateFileAtomic(
    userdata: ?*anyopaque,
    dir: Dir,
    dest_path: []const u8,
    options: Dir.CreateFileAtomicOptions,
) Dir.CreateFileAtomicError!File.Atomic {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
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

fn dirOpenFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    flags: Dir.OpenFileOptions,
) File.OpenError!File {
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    const os_flags: c.O = .{
        .ACCMODE = switch (flags.mode) {
            .read_only => .RDONLY,
            .write_only => .WRONLY,
            .read_write => .RDWR,
        },
        .NONBLOCK = flags.lock == .none or flags.lock_nonblocking,
        .SHLOCK = flags.lock == .shared,
        .EXLOCK = flags.lock == .exclusive,
        .NOFOLLOW = !flags.follow_symlinks,
        .NOCTTY = !flags.allow_ctty,
        .CLOEXEC = true,
    };

    const fd: c.fd_t = while (true) {
        const rc = c.openat(dir.handle, sub_path_posix, os_flags);
        switch (c.errno(rc)) {
            .SUCCESS => break @intCast(rc),
            .INTR => {},
            .FAULT => |err| return errnoBug(err),
            .INVAL => return error.BadPathName,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
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
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .NOTDIR => return error.NotDir,
            .PERM => return error.PermissionDenied,
            .EXIST => return error.PathAlreadyExists,
            .BUSY => return error.DeviceBusy,
            .OPNOTSUPP => return error.FileLocksUnsupported,
            .AGAIN => return error.WouldBlock,
            .TXTBSY => return error.FileBusy,
            .NXIO => return error.NoDevice,
            .ROFS => return error.ReadOnlyFileSystem,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    };
    errdefer closeFd(fd);

    if (!flags.allow_directory) {
        const is_dir = is_dir: {
            const stat = fileStat(ev, .{
                .handle = fd,
                .flags = .{ .nonblocking = false },
            }) catch |err| switch (err) {
                // The directory-ness is either unknown or unknowable
                error.Streaming => break :is_dir false,
                else => |e| return e,
            };
            break :is_dir stat.kind == .directory;
        };
        if (is_dir) return error.IsDir;
    }

    return .{
        .handle = fd,
        .flags = .{ .nonblocking = os_flags.NONBLOCK },
    };
}

fn dirClose(userdata: ?*anyopaque, dirs: []const Dir) void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    for (dirs) |dir| closeFd(dir.handle);
}

fn dirRead(userdata: ?*anyopaque, dr: *Dir.Reader, buffer: []Dir.Entry) Dir.Reader.Error!usize {
    return switch (builtin.os.tag) {
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => dirReadDarwin(userdata, dr, buffer),
        .freebsd, .netbsd, .dragonfly, .openbsd => dirReadBsd(userdata, dr, buffer),
        else => unreachable,
    };
}

/// `Io.Threaded.dirReadDarwin`: the reader's buffer carries a `seek` header for the
/// `getdirentries` call that refills it, and entries have `ino` and `reclen` fields.
fn dirReadDarwin(userdata: ?*anyopaque, dr: *Dir.Reader, buffer: []Dir.Entry) Dir.Reader.Error!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const Header = extern struct {
        seek: i64,
    };
    const header: *Header = @ptrCast(dr.buffer.ptr);
    const header_end: usize = @sizeOf(Header);
    if (dr.index < header_end) {
        // Initialize header.
        dr.index = header_end;
        dr.end = header_end;
        header.* = .{ .seek = 0 };
    }
    var buffer_index: usize = 0;
    while (buffer.len - buffer_index != 0) {
        if (dr.end - dr.index == 0) {
            // Refill the buffer, unless we've already created references to
            // buffered data.
            if (buffer_index != 0) break;
            if (dr.state == .reset) {
                ev.lseek(dr.dir.handle, 0, c.SEEK.SET) catch |err| switch (err) {
                    error.Unseekable => return error.Unexpected,
                    else => |e| return e,
                };
                dr.state = .reading;
            }
            const dents_buffer = dr.buffer[header_end..];
            const n: usize = while (true) {
                const rc = c.getdirentries(dr.dir.handle, dents_buffer.ptr, dents_buffer.len, &header.seek);
                switch (c.errno(rc)) {
                    .SUCCESS => break @intCast(rc),
                    .INTR => {},
                    .BADF => |err| return errnoBug(err), // Dir is invalid or was opened without iteration ability.
                    .FAULT => |err| return errnoBug(err),
                    .NOTDIR => |err| return errnoBug(err),
                    .INVAL => |err| return errnoBug(err),
                    else => |err| return unexpectedErrno(err),
                }
            };
            if (n == 0) {
                dr.state = .finished;
                return 0;
            }
            dr.index = header_end;
            dr.end = header_end + n;
        }
        const darwin_entry = @as(*align(1) c.dirent, @ptrCast(&dr.buffer[dr.index]));
        const next_index = dr.index + darwin_entry.reclen;
        dr.index = next_index;

        const name = @as([*]u8, @ptrCast(&darwin_entry.name))[0..darwin_entry.namlen];
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..") or (darwin_entry.ino == 0))
            continue;

        const entry_kind: File.Kind = switch (darwin_entry.type) {
            c.DT.BLK => .block_device,
            c.DT.CHR => .character_device,
            c.DT.DIR => .directory,
            c.DT.FIFO => .named_pipe,
            c.DT.LNK => .sym_link,
            c.DT.REG => .file,
            c.DT.SOCK => .unix_domain_socket,
            c.DT.WHT => .whiteout,
            else => .unknown,
        };
        buffer[buffer_index] = .{
            .name = name,
            .kind = entry_kind,
            .inode = darwin_entry.ino,
        };
        buffer_index += 1;
    }
    return buffer_index;
}

/// `Io.Threaded.dirReadBsd`: `getdents` fills the whole reader buffer, an entry's
/// record length is a field on all of the BSDs but DragonFly, which computes it,
/// and NetBSD and OpenBSD mark invalid entries with an inode of zero.
fn dirReadBsd(userdata: ?*anyopaque, dr: *Dir.Reader, buffer: []Dir.Entry) Dir.Reader.Error!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var buffer_index: usize = 0;
    while (buffer.len - buffer_index != 0) {
        if (dr.end - dr.index == 0) {
            // Refill the buffer, unless we've already created references to
            // buffered data.
            if (buffer_index != 0) break;
            if (dr.state == .reset) {
                ev.lseek(dr.dir.handle, 0, c.SEEK.SET) catch |err| switch (err) {
                    error.Unseekable => return error.Unexpected,
                    else => |e| return e,
                };
                dr.state = .reading;
            }
            const n: usize = while (true) {
                const rc = c.getdents(dr.dir.handle, dr.buffer.ptr, dr.buffer.len);
                switch (c.errno(rc)) {
                    .SUCCESS => break @intCast(rc),
                    .INTR => {},
                    .BADF => |err| return errnoBug(err), // Dir is invalid or was opened without iteration ability.
                    .FAULT => |err| return errnoBug(err),
                    .NOTDIR => |err| return errnoBug(err),
                    .INVAL => |err| return errnoBug(err),
                    // Introduced in freebsd 13.2: directory unlinked
                    // but still open. To be consistent, iteration ends
                    // if the directory being iterated is deleted
                    // during iteration.
                    .NOENT => {
                        dr.state = .finished;
                        return 0;
                    },
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
        const bsd_entry = @as(*align(1) c.dirent, @ptrCast(&dr.buffer[dr.index]));
        const next_index = dr.index +
            if (@hasField(c.dirent, "reclen")) bsd_entry.reclen else bsd_entry.reclen();
        dr.index = next_index;

        const name = @as([*]u8, @ptrCast(&bsd_entry.name))[0..bsd_entry.namlen];

        const skip_zero_fileno = switch (builtin.os.tag) {
            // fileno=0 is used to mark invalid entries or deleted files.
            .openbsd, .netbsd => true,
            else => false,
        };
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..") or
            (skip_zero_fileno and bsd_entry.fileno == 0))
        {
            continue;
        }

        const entry_kind: File.Kind = switch (bsd_entry.type) {
            c.DT.BLK => .block_device,
            c.DT.CHR => .character_device,
            c.DT.DIR => .directory,
            c.DT.FIFO => .named_pipe,
            c.DT.LNK => .sym_link,
            c.DT.REG => .file,
            c.DT.SOCK => .unix_domain_socket,
            c.DT.WHT => .whiteout,
            else => .unknown,
        };
        buffer[buffer_index] = .{
            .name = name,
            .kind = entry_kind,
            .inode = bsd_entry.fileno,
        };
        buffer_index += 1;
    }
    return buffer_index;
}

fn dirRealPath(userdata: ?*anyopaque, dir: Dir, out_buffer: []u8) Dir.RealPathError!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return ev.realPath(dir.handle, out_buffer);
}

fn dirRealPathFile(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    out_buffer: []u8,
) Dir.RealPathFileError!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    if (dir.handle == c.AT.FDCWD) {
        if (out_buffer.len < c.PATH_MAX) return error.NameTooLong;
        while (true) {
            if (c.realpath(sub_path_posix, out_buffer.ptr)) |redundant_pointer| {
                assert(redundant_pointer == out_buffer.ptr);
                return std.mem.findScalar(u8, out_buffer, 0) orelse out_buffer.len;
            }
            const err: c.E = @fromBackingInt(@intCast(c._errno().*));
            switch (err) {
                .INTR => {},
                .INVAL => return errnoBug(err),
                .BADF => return errnoBug(err),
                .FAULT => return errnoBug(err),
                .ACCES => return error.AccessDenied,
                .NOENT => return error.FileNotFound,
                .OPNOTSUPP => return error.OperationUnsupported,
                .NOTDIR => return error.NotDir,
                .NAMETOOLONG => return error.NameTooLong,
                .LOOP => return error.SymLinkLoop,
                .IO => return error.InputOutput,
                else => return unexpectedErrno(err),
            }
        }
    }

    var os_flags: c.O = .{
        .NONBLOCK = true,
        .CLOEXEC = true,
    };
    // `Io.Threaded.dirRealPathFilePosix` opens with `O.PATH` where the platform has it: on
    // FreeBSD the descriptor then names a file the process may not read, and `fcntl` is one of
    // the operations such a descriptor allows.
    if (@hasField(c.O, "PATH")) os_flags.PATH = true;

    const fd: c.fd_t = while (true) {
        const rc = c.openat(dir.handle, sub_path_posix, os_flags);
        switch (c.errno(rc)) {
            .SUCCESS => break @intCast(rc),
            .INTR => {},
            .FAULT => |err| return errnoBug(err),
            .INVAL => return error.BadPathName,
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
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
            .NOMEM => return error.SystemResources,
            .NOSPC => return error.NoSpaceLeft,
            .NOTDIR => return error.NotDir,
            .PERM => return error.PermissionDenied,
            .EXIST => return error.PathAlreadyExists,
            .BUSY => return error.DeviceBusy,
            .NXIO => return error.NoDevice,
            .ILSEQ => return error.BadPathName,
            else => |err| return unexpectedErrno(err),
        }
    };
    defer closeFd(fd);
    return ev.realPath(fd, out_buffer);
}

fn dirDeleteFile(userdata: ?*anyopaque, dir: Dir, sub_path: []const u8) Dir.DeleteFileError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    while (true) switch (c.errno(c.unlinkat(dir.handle, sub_path_posix, 0))) {
        .SUCCESS => return,
        .INTR => {},
        // Some systems return permission errors when trying to delete a
        // directory, so we need to handle that case specifically and
        // translate the error.
        .PERM => {
            // Don't follow symlinks to match unlinkat (which acts on symlinks rather than follows them).
            var st = std.mem.zeroes(c.Stat);
            while (true) switch (c.errno(c.fstatat(
                dir.handle,
                sub_path_posix,
                &st,
                c.AT.SYMLINK_NOFOLLOW,
            ))) {
                .SUCCESS => break,
                .INTR => {},
                else => return error.PermissionDenied,
            };
            if (st.mode & c.S.IFMT == c.S.IFDIR) return error.IsDir else return error.PermissionDenied;
        },
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
    };
}

fn dirDeleteDir(userdata: ?*anyopaque, dir: Dir, sub_path: []const u8) Dir.DeleteDirError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    while (true) switch (c.errno(c.unlinkat(dir.handle, sub_path_posix, c.AT.REMOVEDIR))) {
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
    };
}

fn dirRename(
    userdata: ?*anyopaque,
    old_dir: Dir,
    old_sub_path: []const u8,
    new_dir: Dir,
    new_sub_path: []const u8,
) Dir.RenameError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var old_path_buffer: [c.PATH_MAX]u8 = undefined;
    var new_path_buffer: [c.PATH_MAX]u8 = undefined;

    const old_sub_path_posix = try pathToPosix(old_sub_path, &old_path_buffer);
    const new_sub_path_posix = try pathToPosix(new_sub_path, &new_path_buffer);

    while (true) switch (c.errno(c.renameat(old_dir.handle, old_sub_path_posix, new_dir.handle, new_sub_path_posix))) {
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
        .EXIST => return error.DirNotEmpty,
        .NOTEMPTY => return error.DirNotEmpty,
        .ROFS => return error.ReadOnlyFileSystem,
        .XDEV => return error.CrossDevice,
        .ILSEQ => return error.BadPathName,
        .FAULT => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err),
        else => |err| return unexpectedErrno(err),
    };
}

fn dirRenamePreserve(
    userdata: ?*anyopaque,
    old_dir: Dir,
    old_sub_path: []const u8,
    new_dir: Dir,
    new_sub_path: []const u8,
) Dir.RenamePreserveError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    // Make a hard link then delete the original.
    try dirHardLink(ev, old_dir, old_sub_path, new_dir, new_sub_path, .{ .follow_symlinks = false });
    const prev = Scheduler.swapCancelProtection(ev, .blocked);
    defer _ = Scheduler.swapCancelProtection(ev, prev);
    dirDeleteFile(ev, old_dir, old_sub_path) catch {};
}

fn dirSymLink(
    userdata: ?*anyopaque,
    dir: Dir,
    target_path: []const u8,
    sym_link_path: []const u8,
    flags: Dir.SymLinkFlags,
) Dir.SymLinkError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = flags;

    var target_path_buffer: [c.PATH_MAX]u8 = undefined;
    var sym_link_path_buffer: [c.PATH_MAX]u8 = undefined;

    const target_path_posix = try pathToPosix(target_path, &target_path_buffer);
    const sym_link_path_posix = try pathToPosix(sym_link_path, &sym_link_path_buffer);

    while (true) switch (c.errno(c.symlinkat(target_path_posix, dir.handle, sym_link_path_posix))) {
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
    };
}

fn dirReadLink(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    buffer: []u8,
) Dir.ReadLinkError!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    var sub_path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &sub_path_buffer);
    while (true) {
        const rc = c.readlinkat(dir.handle, sub_path_posix, buffer.ptr, buffer.len);
        switch (c.errno(rc)) {
            .SUCCESS => return @intCast(rc),
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
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    return fchown(dir.handle, owner, group);
}

fn dirSetFileOwner(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    owner: ?File.Uid,
    group: ?File.Gid,
    options: Dir.SetFileOwnerOptions,
) Dir.SetFileOwnerError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);
    _ = ev;
    while (true) switch (c.errno(c.fchownat(
        dir.handle,
        sub_path_posix,
        owner orelse std.math.maxInt(c.uid_t),
        group orelse std.math.maxInt(c.gid_t),
        if (options.follow_symlinks) 0 else c.AT.SYMLINK_NOFOLLOW,
    ))) {
        .SUCCESS => return,
        .INTR => continue,
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
    };
}

fn dirSetPermissions(
    userdata: ?*anyopaque,
    dir: Dir,
    permissions: Dir.Permissions,
) Dir.SetPermissionsError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return ev.fchmod(dir.handle, permissions.toMode());
}

fn dirSetFilePermissions(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    permissions: Dir.Permissions,
    options: Dir.SetFilePermissionsOptions,
) Dir.SetFilePermissionsError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    const mode = permissions.toMode();
    const flags: u32 = if (options.follow_symlinks) 0 else c.AT.SYMLINK_NOFOLLOW;

    while (true) switch (c.errno(c.fchmodat(dir.handle, sub_path_posix, mode, flags))) {
        .SUCCESS => return,
        .INTR => {},
        .BADF => |err| return errnoBug(err),
        .FAULT => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err),
        .ACCES => return error.AccessDenied,
        .IO => return error.InputOutput,
        .LOOP => return error.SymLinkLoop,
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NAMETOOLONG => return error.NameTooLong,
        .NFILE => return error.SystemFdQuotaExceeded,
        .NOENT => return error.FileNotFound,
        .NOTDIR => return error.FileNotFound,
        .NOMEM => return error.SystemResources,
        .OPNOTSUPP => return error.OperationUnsupported,
        .PERM => return error.PermissionDenied,
        .ROFS => return error.ReadOnlyFileSystem,
        else => |err| return unexpectedErrno(err),
    };
}

fn dirSetTimestamps(
    userdata: ?*anyopaque,
    dir: Dir,
    sub_path: []const u8,
    options: Dir.SetTimestampsOptions,
) Dir.SetTimestampsError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var times_buffer: [2]c.timespec = undefined;
    const times = if (options.modify_timestamp == .now and options.access_timestamp == .now) null else p: {
        times_buffer = .{
            setTimestampToPosix(options.access_timestamp),
            setTimestampToPosix(options.modify_timestamp),
        };
        break :p &times_buffer;
    };

    const flags: u32 = if (options.follow_symlinks) 0 else c.AT.SYMLINK_NOFOLLOW;

    var path_buffer: [c.PATH_MAX]u8 = undefined;
    const sub_path_posix = try pathToPosix(sub_path, &path_buffer);

    while (true) switch (c.errno(c.utimensat(dir.handle, sub_path_posix, times, flags))) {
        .SUCCESS => return,
        .INTR => {},
        .BADF => |err| return errnoBug(err), // always a race condition
        .FAULT => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err),
        .ACCES => return error.AccessDenied,
        .PERM => return error.PermissionDenied,
        .ROFS => return error.ReadOnlyFileSystem,
        else => |err| return unexpectedErrno(err),
    };
}

fn dirHardLink(
    userdata: ?*anyopaque,
    old_dir: Dir,
    old_sub_path: []const u8,
    new_dir: Dir,
    new_sub_path: []const u8,
    options: Dir.HardLinkOptions,
) Dir.HardLinkError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var old_path_buffer: [c.PATH_MAX]u8 = undefined;
    var new_path_buffer: [c.PATH_MAX]u8 = undefined;

    const old_sub_path_posix = try pathToPosix(old_sub_path, &old_path_buffer);
    const new_sub_path_posix = try pathToPosix(new_sub_path, &new_path_buffer);

    const flags: u32 = if (options.follow_symlinks) c.AT.SYMLINK_FOLLOW else 0;
    return linkat(old_dir.handle, old_sub_path_posix, new_dir.handle, new_sub_path_posix, flags);
}

fn fileStat(userdata: ?*anyopaque, file: File) File.StatError!File.Stat {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    while (true) {
        var stat = std.mem.zeroes(c.Stat);
        switch (c.errno(c.fstat(file.handle, &stat))) {
            .SUCCESS => return statFromPosix(&stat),
            .INTR => {},
            .INVAL => |err| return errnoBug(err),
            .BADF => |err| return errnoBug(err), // File descriptor used after closed.
            .NOMEM => return error.SystemResources,
            .ACCES => return error.AccessDenied,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fileLength(userdata: ?*anyopaque, file: File) File.LengthError!u64 {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const stat = try fileStat(ev, file);
    return stat.size;
}

fn fileClose(userdata: ?*anyopaque, files: []const File) void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    for (files) |file| closeFd(file.handle);
}

fn fileWritePositional(
    userdata: ?*anyopaque,
    file: File,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
    offset: u64,
) File.WritePositionalError!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    var iovecs: [max_iovecs_len]iovec_const = undefined;
    var iovlen: iovlen_t = 0;
    var remaining: Io.Limit = .unlimited;
    addBuf(true, &iovecs, &iovlen, &remaining, header);
    for (data[0 .. data.len - 1]) |bytes| addBuf(true, &iovecs, &iovlen, &remaining, bytes);
    const pattern = data[data.len - 1];
    var backup_buffer: [splat_buffer_size]u8 = undefined;
    if (iovecs.len - iovlen != 0 and remaining != .nothing) switch (splat) {
        0 => {},
        1 => addBuf(true, &iovecs, &iovlen, &remaining, pattern),
        else => switch (pattern.len) {
            0 => {},
            1 => {
                const splat_buffer = &backup_buffer;
                const memset_len = @min(splat_buffer.len, splat);
                const buf = splat_buffer[0..memset_len];
                @memset(buf, pattern[0]);
                addBuf(true, &iovecs, &iovlen, &remaining, buf);
                var remaining_splat = splat - buf.len;
                while (remaining_splat > splat_buffer.len and iovecs.len - iovlen != 0 and remaining != .nothing) {
                    assert(buf.len == splat_buffer.len);
                    addBuf(true, &iovecs, &iovlen, &remaining, splat_buffer);
                    remaining_splat -= splat_buffer.len;
                }
                addBuf(true, &iovecs, &iovlen, &remaining, splat_buffer[0..@min(remaining_splat, splat_buffer.len)]);
            },
            else => for (0..@min(splat, iovecs.len - iovlen)) |_| {
                if (remaining == .nothing) break;
                addBuf(true, &iovecs, &iovlen, &remaining, pattern);
            },
        },
    };
    if (iovlen == 0) return 0;
    while (true) {
        const rc = c.pwritev(file.handle, &iovecs, iovlen, @bitCast(offset));
        switch (c.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => {},
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .DESTADDRREQ => |err| return errnoBug(err), // `connect` was never called.
            .CONNRESET => |err| return errnoBug(err), // Not a socket handle.
            .BADF => return error.NotOpenForWriting,
            .AGAIN => return error.WouldBlock,
            .DQUOT => return error.DiskQuota,
            .FBIG => return error.FileTooBig,
            .IO => return error.InputOutput,
            .NOSPC => return error.NoSpaceLeft,
            .PERM => return error.PermissionDenied,
            .PIPE => return error.BrokenPipe,
            .BUSY => return error.DeviceBusy,
            .TXTBSY => return error.FileBusy,
            .NXIO => return error.Unseekable,
            .SPIPE => return error.Unseekable,
            .OVERFLOW => return error.Unseekable,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fileReadPositional(
    userdata: ?*anyopaque,
    file: File,
    data: []const []u8,
    offset: u64,
) File.ReadPositionalError!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    var iovecs: [max_iovecs_len]iovec = undefined;
    var iovlen: iovlen_t = 0;
    var remaining: Io.Limit = .unlimited;
    for (data) |buf| addBuf(false, &iovecs, &iovlen, &remaining, buf);
    if (iovlen == 0) return 0;
    while (true) {
        const rc = c.preadv(file.handle, &iovecs, iovlen, @bitCast(offset));
        switch (c.errno(rc)) {
            .SUCCESS => return @intCast(rc),
            .INTR => {},
            .NXIO => return error.Unseekable,
            .SPIPE => return error.Unseekable,
            .OVERFLOW => return error.Unseekable,
            .NOBUFS => return error.SystemResources,
            .NOMEM => return error.SystemResources,
            .AGAIN => return error.WouldBlock,
            .IO => return error.InputOutput,
            .ISDIR => return error.IsDir,
            .NOTCONN => |err| return errnoBug(err), // not a socket
            .CONNRESET => |err| return errnoBug(err), // not a socket
            .INVAL => |err| return errnoBug(err),
            .FAULT => |err| return errnoBug(err),
            .BADF => return error.NotOpenForReading,
            else => |err| return unexpectedErrno(err),
        }
    }
}

fn fileSeekBy(userdata: ?*anyopaque, file: File, offset: i64) File.SeekError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return ev.lseek(file.handle, @bitCast(offset), c.SEEK.CUR);
}

fn fileSeekTo(userdata: ?*anyopaque, file: File, offset: u64) File.SeekError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return ev.lseek(file.handle, offset, c.SEEK.SET);
}

fn fileSetLength(userdata: ?*anyopaque, file: File, length: u64) File.SetLengthError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    const signed_len: i64 = @bitCast(length);
    if (signed_len < 0) return error.FileTooBig; // Avoid ambiguous EINVAL errors.

    while (true) switch (c.errno(c.ftruncate(file.handle, signed_len))) {
        .SUCCESS => return,
        .INTR => {},
        .FBIG => return error.FileTooBig,
        .IO => return error.InputOutput,
        .PERM => return error.PermissionDenied,
        .TXTBSY => return error.FileBusy,
        .BADF => |err| return errnoBug(err), // Handle not open for writing.
        .INVAL => return error.NonResizable, // This is returned for /dev/null for example.
        else => |err| return unexpectedErrno(err),
    };
}

fn fileSetOwner(
    userdata: ?*anyopaque,
    file: File,
    owner: ?File.Uid,
    group: ?File.Gid,
) File.SetOwnerError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    return fchown(file.handle, owner, group);
}

fn fileSetPermissions(
    userdata: ?*anyopaque,
    file: File,
    permissions: File.Permissions,
) File.SetPermissionsError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return ev.fchmod(file.handle, permissions.toMode());
}

fn fileSetTimestamps(
    userdata: ?*anyopaque,
    file: File,
    options: File.SetTimestampsOptions,
) File.SetTimestampsError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    var times_buffer: [2]c.timespec = undefined;
    const times = if (options.modify_timestamp == .now and options.access_timestamp == .now) null else p: {
        times_buffer = .{
            setTimestampToPosix(options.access_timestamp),
            setTimestampToPosix(options.modify_timestamp),
        };
        break :p &times_buffer;
    };

    while (true) switch (c.errno(c.futimens(file.handle, times))) {
        .SUCCESS => return,
        .INTR => {},
        .BADF => |err| return errnoBug(err), // always a race condition
        .FAULT => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err),
        .ACCES => return error.AccessDenied,
        .PERM => return error.PermissionDenied,
        .ROFS => return error.ReadOnlyFileSystem,
        else => |err| return unexpectedErrno(err),
    };
}

fn fileLock(userdata: ?*anyopaque, file: File, lock: File.Lock) File.LockError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    const operation: i32 = switch (lock) {
        .none => c.LOCK.UN,
        .shared => c.LOCK.SH,
        .exclusive => c.LOCK.EX,
    };
    while (true) switch (c.errno(c.flock(file.handle, operation))) {
        .SUCCESS => return,
        .INTR => {},
        .BADF => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err), // invalid parameters
        .NOLCK => return error.SystemResources,
        .AGAIN => |err| return errnoBug(err),
        .OPNOTSUPP => return error.FileLocksUnsupported,
        else => |err| return unexpectedErrno(err),
    };
}

fn fileTryLock(userdata: ?*anyopaque, file: File, lock: File.Lock) File.LockError!bool {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    const operation: i32 = switch (lock) {
        .none => c.LOCK.UN,
        .shared => c.LOCK.SH | c.LOCK.NB,
        .exclusive => c.LOCK.EX | c.LOCK.NB,
    };
    while (true) switch (c.errno(c.flock(file.handle, operation))) {
        .SUCCESS => return true,
        .INTR => {},
        .AGAIN => return false,
        .BADF => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err), // invalid parameters
        .NOLCK => return error.SystemResources,
        .OPNOTSUPP => return error.FileLocksUnsupported,
        else => |err| return unexpectedErrno(err),
    };
}

fn fileUnlock(userdata: ?*anyopaque, file: File) void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    while (true) switch (c.errno(c.flock(file.handle, c.LOCK.UN))) {
        .SUCCESS => return,
        .INTR => {},
        .AGAIN => return recoverableOsBugDetected(), // unlocking can't block
        .BADF => return recoverableOsBugDetected(), // File descriptor used after closed.
        .INVAL => return recoverableOsBugDetected(), // invalid parameters
        .NOLCK => return recoverableOsBugDetected(), // Resource deallocation.
        .OPNOTSUPP => return recoverableOsBugDetected(), // We already got the lock.
        else => return recoverableOsBugDetected(), // Resource deallocation must succeed.
    };
}

fn fileDowngradeLock(userdata: ?*anyopaque, file: File) File.DowngradeLockError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    const operation = c.LOCK.SH | c.LOCK.NB;
    while (true) switch (c.errno(c.flock(file.handle, operation))) {
        .SUCCESS => return,
        .INTR => {},
        .AGAIN => |err| return errnoBug(err), // File was not locked in exclusive mode.
        .BADF => |err| return errnoBug(err),
        .INVAL => |err| return errnoBug(err), // invalid parameters
        .NOLCK => |err| return errnoBug(err), // Lock already obtained.
        .OPNOTSUPP => |err| return errnoBug(err), // Lock already obtained.
        else => |err| return unexpectedErrno(err),
    };
}

fn fileRealPath(userdata: ?*anyopaque, file: File, out_buffer: []u8) File.RealPathError!usize {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    return ev.realPath(file.handle, out_buffer);
}

fn fileHardLink(
    userdata: ?*anyopaque,
    file: File,
    new_dir: Dir,
    new_sub_path: []const u8,
    options: File.HardLinkOptions,
) File.HardLinkError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = file;
    _ = new_dir;
    _ = new_sub_path;
    _ = options;
    return error.OperationUnsupported;
}

fn fileMemoryMapCreate(
    userdata: ?*anyopaque,
    file: File,
    options: File.MemoryMap.CreateOptions,
) File.MemoryMap.CreateError!File.MemoryMap {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    const prot: c.PROT = .{
        .READ = options.protection.read,
        .WRITE = options.protection.write,
        .EXEC = options.protection.execute,
    };
    const flags: c.MAP = .{
        .TYPE = .SHARED,
    };

    const page_align = std.heap.page_size_min;

    const contents = while (true) {
        const casted_offset = std.math.cast(i64, options.offset) orelse return error.Unseekable;
        const rc = c.mmap(null, options.len, prot, flags, file.handle, casted_offset);
        const err: c.E = if (rc != c.MAP_FAILED) .SUCCESS else @fromBackingInt(@intCast(c._errno().*));
        switch (err) {
            .SUCCESS => break @as([*]align(page_align) u8, @ptrCast(@alignCast(rc)))[0..options.len],
            .INTR => {},
            .ACCES => return error.AccessDenied,
            .AGAIN => return error.LockedMemoryLimitExceeded,
            .MFILE => return error.ProcessFdQuotaExceeded,
            .NFILE => return error.SystemFdQuotaExceeded,
            .NOMEM => return error.OutOfMemory,
            .PERM => return error.PermissionDenied,
            .OVERFLOW => return error.Unseekable,
            .BADF => return errnoBug(err), // Always a race condition.
            .INVAL => return errnoBug(err), // Invalid parameters to mmap()
            else => return unexpectedErrno(err),
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
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    const memory = mm.memory;
    if (memory.len == 0) return;
    switch (c.errno(c.munmap(memory.ptr, memory.len))) {
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
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;

    const page_size = std.heap.pageSize();
    const alignment: Alignment = .fromByteUnits(page_size);
    const old_memory = mm.memory;

    if (alignment.forward(new_len) == alignment.forward(old_memory.len)) {
        mm.memory.len = new_len;
        return;
    }
    return error.OperationUnsupported;
}

fn fileMemoryMapRead(userdata: ?*anyopaque, mm: *File.MemoryMap) File.ReadPositionalError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = mm;
}

fn fileMemoryMapWrite(userdata: ?*anyopaque, mm: *File.MemoryMap) File.WritePositionalError!void {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = mm;
}

fn processReplace(userdata: ?*anyopaque, options: process.ReplaceOptions) process.ReplaceError {
    const ev: *Evented = @ptrCast(@alignCast(userdata));

    if (!process.can_replace) return error.OperationUnsupported;

    ev.scanEnviron() catch |err| switch (err) {
        error.Canceled => {},
    }; // for PATH
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

    return ev.execv(options.expand_arg0, argv_buf.ptr[0].?, argv_buf.ptr, env_block, PATH);
}

fn processReplacePath(
    userdata: ?*anyopaque,
    dir: Dir,
    options: process.ReplaceOptions,
) process.ReplaceError {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = dir;
    _ = options;
    @panic("TODO processReplacePath");
}

fn processSpawn(userdata: ?*anyopaque, options: process.SpawnOptions) process.SpawnError!process.Child {
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    const spawned = try ev.spawn(options);
    defer fileClose(ev, &.{spawned.err_pipe});

    // Wait for the child to report any errors in or before `execvpe`.
    var child_err: ForkBailError = undefined;
    ev.readAll(spawned.err_pipe, @ptrCast(&child_err)) catch |read_err| {
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
    const ev: *Evented = @ptrCast(@alignCast(userdata));
    _ = ev;
    _ = dir;
    _ = options;
    @panic("TODO processSpawnPath");
}

/// Reads until the descriptor reaches end of stream, through the core's own streaming reader.
fn readAll(ev: *Evented, file: File, buffer: []u8) File.ReadStreamingError!void {
    _ = try file.readStreaming(ev.io(), &.{buffer});
}

/// Writes all of `bytes`, through the core's own streaming writer.
fn writeAll(ev: *Evented, file: File, buffer: []const u8) (File.Writer.Error || error{EndOfStream})!void {
    try file.writeStreamingAll(ev.io(), buffer);
}
