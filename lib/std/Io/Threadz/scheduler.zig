//! The scheduler the Threadz cores share. A task is a function running on its own stack, and a
//! worker is an OS thread that runs tasks one at a time. A task parks at every Io call that has
//! to wait: the backend submits the operation, the worker switches to the next runnable task, and
//! the completion makes the parked task runnable again. A worker blocks only when it has nothing
//! to run.
//!
//! Where a runnable task goes is its affinity, chosen at spawn:
//! * `sticky`, the default: it resumes on the worker that last ran it. Another worker takes it only
//!   from a worker with more than `steal_threshold` tasks waiting in its queue.
//! * `pinned`: it never leaves the worker it names. This is for tasks that own a worker's
//!   resources, such as a listener or a slab.
//! * `free`: it goes to the shared queue, and whichever worker gets there first runs it.
//!
//! A task's memory is one mapping: a guard page, then its stack, then the task itself with its
//! result and its arguments at the top, so an overflow faults instead of corrupting the task. The
//! OS commits stack pages as they are touched. Mappings of the default size are reused through a
//! small cache per worker and a shared pool, and unmapped beyond those.
//!
//! A task may complete `budget` Io operations without parking. The operation past that yields it
//! first: it goes to the back of its worker's queue, keeping its affinity, and then the operation
//! runs where the task landed. See `charge`, which every Threadz entry point calls.
//!
//! A worker whose current task does not switch out for `stuck_after` is stuck: it is running
//! something that neither parks nor yields, such as a spinning loop or a C call that blocks its
//! thread. One watchdog thread per instance samples every worker every `watchdog_interval` and,
//! for a stuck worker: makes its queued tasks takeable whatever their number, starts one
//! replacement worker so that parallelism stays, counts the episode, and names the task in one
//! log line. A task that owns its worker's resources, being pinned, waits for the worker instead;
//! see `Affinity.pinned`. A worker pays one store per switch for the watchdog, of the task it
//! runs, and one load of a flag when it takes the task in its slot; nothing else.
//!
//! A backend embeds the scheduler as its field `sched` and declares:
//! * `Worker`, its state for each worker, with `workerInit`, `workerStart` (called on the worker's
//!   own thread before it runs anything) and `workerDeinit`;
//! * `Completion`, what a finished operation leaves in its task's result slot;
//! * `poll(backend, worker, mode)`, which submits queued operations and passes each task whose
//!   operation finished to `readyFromPoll`. With `.block` it waits for at least one event.
//! * `wake(backend, from, to)`, which ends `to`'s current or next blocking `poll`;
//! * `wakeForeign(backend, to)`, the same from a thread that is not one of the workers, which is
//!   the watchdog. It must be safe to call from any thread;
//! * `threadCpuTime(backend, w) ?u64`, the CPU time in nanoseconds the worker's thread has used,
//!   or `null` when it cannot be read. The watchdog tells a worker that is blocked from one that
//!   is computing with it: see `watchdogWorker`. A backend that does not declare it leaves every
//!   stuck worker blocked, which is what a worker that cannot say otherwise is;
//! * `heavyBarrier(backend) bool`, whether the backend issued a barrier that makes this thread's
//!   earlier stores visible to every other thread of the process, after their own barriers. The
//!   watchdog takes a stuck worker's slot for the next task with it, and a worker that takes its
//!   own pays no locked instruction while the backend can issue one. On Linux it is
//!   `membarrier(PRIVATE_EXPEDITED)`, registered for once per process. A backend that does not
//!   declare it, or that says it cannot, has every taker of a slot pay a compare-exchange, and
//!   the watchdog takes the slot with the same exchange, which one of them wins;
//! * `cancelOperation(backend, from, task, token)`, which cancels the operation `task` is waiting
//!   for in the worker identified by `token`;
//! * `allocator(backend)`;
//! * `io(backend)`, the instance's `Io`, which the watchdog uses to wait between its samples and
//!   to read the clock. Its `now` has to work.

const builtin = @import("builtin");
const std = @import("../../std.zig");
const Io = std.Io;
const Alignment = std.mem.Alignment;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const linux = std.os.linux;
const posix = std.posix;
const page_size_min = std.heap.page_size_min;
const recoverableOsBugDetected = Io.Threaded.recoverableOsBugDetected;

const tracy = if (@hasDecl(@import("root"), "tracy")) @import("root").tracy else struct {
    const enable = false;
    inline fn fiberEnter(fiber: [*:0]const u8) void {
        _ = fiber;
    }
    inline fn fiberLeave() void {}
};

/// Where a task runs whenever it becomes runnable.
pub const Affinity = union(enum) {
    /// On the worker that last ran it. Another worker takes it only from a worker with more than
    /// `steal_threshold` tasks waiting.
    sticky,
    /// Only on the worker with this index, which is started if it is not running yet.
    pinned: u32,
    /// On whichever worker takes it first from the shared queue.
    free,
};

pub const SpawnOptions = struct {
    /// Bytes reserved for the task's stack. `null` means the instance's default.
    stack_size: ?usize = null,
    affinity: Affinity = .sticky,
};

/// The default stack reservation for a task.
pub const default_stack_size = 256 * 1024;

/// Workers with more than this many sticky tasks waiting let other workers take some.
pub const steal_threshold = 1;

/// The Io operations a task may complete without parking before the scheduler yields it. See
/// `Scheduler.charge`.
pub const budget = 128;

/// The watchdog samples every worker this often.
pub const watchdog_interval: u64 = 10 * std.time.ns_per_ms;

/// A worker whose current task has not switched out for this long is stuck. See `Scheduler`.
pub const stuck_after: u64 = 100 * std.time.ns_per_ms;

/// Worker 0 runs its scheduling loop on a stack of this size, since the thread's own stack belongs
/// to the main task. The other workers' threads get stacks of this size.
///
/// Empirically saw >128KB being used by the self-hosted backend to panic.
/// Empirically saw glibc complain about 256KB.
const idle_stack_size = 512 * 1024;

/// A worker goes through its scheduling loop, which polls the backend, after this many switches,
/// even if it always has another task to run.
const poll_interval = 61;
/// A worker runs the task in its LIFO slot this many times in a row, then takes from its queue.
const lifo_limit = 3;
/// The first worker to look for tasks spins through this many attempts before it parks.
const spin_limit = 128;
/// Every worker looking for tasks tries the other workers' queues this many times before it parks.
const search_rounds = 4;
/// Mappings of the default size each worker keeps for reuse.
const stack_cache_max = 16;
/// Mappings of the default size the pool keeps for reuse, per worker.
const stack_pool_max_per_worker = 16;
/// Room at the top of a default mapping for the task, its result and its arguments.
const default_header_size = 4096;
/// Mappings kept for reuse that are bigger than this give their stack pages back to the OS,
/// except the top `trim_keep` bytes.
const trim_threshold = 1024 * 1024;
const trim_keep = 256 * 1024;

pub const PollMode = enum { nonblocking, block };

/// ThreadSanitizer follows each task as a fiber of its own, so that it attributes accesses to the
/// task rather than to whichever thread ran it.
const tsan = struct {
    const enable = builtin.sanitize_thread;
    const Fiber = if (enable) *anyopaque else void;
    extern fn __tsan_get_current_fiber() *anyopaque;
    extern fn __tsan_create_fiber(flags: c_uint) *anyopaque;
    extern fn __tsan_destroy_fiber(fiber: *anyopaque) void;
    extern fn __tsan_switch_to_fiber(fiber: *anyopaque, flags: c_uint) void;
};

pub fn Scheduler(comptime Backend: type) type {
    return struct {
        workers: []Worker,
        /// `workers[0..started]` are running.
        started: std.atomic.Value(u32),
        /// Equal to `started`, or one more while a worker is being started.
        reserved: std.atomic.Value(u32),
        /// No more than this many workers are started.
        limit: std.atomic.Value(u32),
        /// The replacement workers the watchdog starts for stuck ones, one slot per worker. A
        /// slot holds a worker only once one has been started in it.
        spares: []Spare,
        shared: TaskQueue,
        idle: Idle,
        stopping: std.atomic.Value(bool),
        stack_size: usize,
        /// The length of a mapping for a task with the default stack size.
        mapping_len: usize,
        pool: StackPool,
        idle_stack: []align(page_size_min) u8,
        /// The pool `io.blocking` calls and a worker's fsyncs run on. See `Dirty`.
        dirty: Dirty = .{},
        /// The watchdog thread, started with the first worker past worker 0 and stopped by
        /// `deinit`.
        watchdog: std.Thread,
        watchdog_started: std.atomic.Value(bool),
        /// The watchdog waits on this between rounds, and `deinit` wakes it with it.
        watchdog_wait: std.atomic.Value(u32),
        /// Set while the watchdog waits with no timeout, with every worker parked: a worker that
        /// unparks then wakes it. See `watchdogWait` and `notifyWatchdogUnparked`.
        watchdog_sleeping: std.atomic.Value(bool),
        /// Rounds of samples the watchdog made. See `stats`.
        watchdog_rounds: std.atomic.Value(u64),
        /// Episodes in which a worker was found stuck. See `stats`.
        stuck_episodes: std.atomic.Value(u64),
        /// Episodes in which a worker was found computing. See `stats`.
        computing_episodes: std.atomic.Value(u64),
        /// Replacement workers the watchdog started for stuck workers. See `stats`.
        replacements: std.atomic.Value(u64),
        /// What the watchdog reported about the last stuck episode. See `stats`.
        report: StuckReport,
        /// The `live` counts of the replacement workers that have ended: a task spawned on one
        /// worker and ended on a replacement is counted on the replacement, which is gone by the
        /// time `deinit` adds the counts up. Only the watchdog writes it, once the replacement's
        /// thread has joined, and `deinit` reads it.
        reaped_live: isize,
        /// The thread that calls `init` runs as this task, on its own stack, followed by room for
        /// a completion.
        main_task_buffer: [@sizeOf(Task) + completion_space]u8 align(@alignOf(Task)),

        const Sched = @This();

        /// Whether the backend declares `heavyBarrier`. Without it, every taker of a slot for the
        /// next task pays a compare-exchange, and the watchdog takes none: see `takeNext`.
        const has_heavy_barrier = @hasDecl(Backend, "heavyBarrier");
        /// Whether the backend declares `threadCpuTime`. Without it, a worker that does not switch
        /// out is taken to be blocked, which is what has to be assumed of a worker that cannot
        /// say otherwise.
        const has_thread_cpu_time = @hasDecl(Backend, "threadCpuTime");
        /// Cleared when the backend says it cannot issue the barrier, which the watchdog finds
        /// out when it first needs one. Every taker of a slot then pays a compare-exchange: see
        /// `takeNext`.
        var heavy_barrier_available: std.atomic.Value(bool) = .init(true);

        const completion_space = std.mem.alignForward(usize, @sizeOf(Backend.Completion), @alignOf(Task));

        pub const Options = struct {
            /// The most workers, counting the thread that calls `init`. `null` means one per CPU.
            workers: ?usize = null,
            /// The default stack reservation for a task.
            stack_size: usize = default_stack_size,
        };

        pub const Task = struct {
            required_align: void align(16) = {},
            context: Io.fiber.Context,
            /// For a future, the task awaiting it, or `finished`. For a group member, its neighbors.
            /// Extern, so that no safety tag is read alongside `awaiter`, which other workers
            /// write atomically.
            link: extern union {
                awaiter: ?*Task,
                group: extern struct { prev: ?*Task, next: ?*Task },
            },
            status: union(enum) {
                queue_next: ?*Task,
                awaiting_group: Group,
            },
            cancel_status: CancelStatus,
            cancel_protection: CancelProtection,
            affinity: std.meta.Tag(Affinity),
            /// The worker that last ran it, or the one it is pinned to.
            home: u32,
            start: union(enum) {
                /// The main task, which runs on its thread's stack.
                main,
                future: *const fn (context: *const anyopaque, result: *anyopaque) void,
                group: struct {
                    group: Group,
                    start: *const fn (context: *const anyopaque) void,
                },
            },
            result_align: Alignment,
            /// The operations this task completed without parking since it last parked or
            /// yielded. See `Sched.charge`.
            ops: u32,
            context_bytes: [*]u8,
            /// Empty for the main task.
            mapping: []align(page_size_min) u8,
            /// This task's number, unique among the tasks every Threadz instance in the program
            /// made, in the order they were made. A log line and, later, a task dump name a
            /// task with it.
            id: u64,
            /// The name of the function this task runs, for the task names a log line reports.
            /// `Io.spawnedName` is where it comes from for a spawn through `Io`. The name of a
            /// task outlives it: it is a comptime string.
            name: [:0]const u8,
            tsan_fiber: tsan.Fiber,

            /// The first id of the next block of ids a worker takes. See `Task.id` and
            /// `Worker.nextTaskId`.
            var next_id: std.atomic.Value(u64) = .init(1);

            /// How many ids a worker takes from `next_id` at once.
            const id_block = 1024;

            pub const finished: ?*Task = @ptrFromInt(@alignOf(Task));

            pub const CancelStatus = packed struct(u32) {
                requested: bool,
                awaiting: Awaiting,

                pub const unrequested: CancelStatus = .{ .requested = false, .awaiting = .nothing };

                pub const Awaiting = enum(u31) {
                    nothing = std.math.maxInt(u31),
                    group = std.math.maxInt(u31) - 1,
                    /// A backend's token for the worker whose operation the task awaits.
                    _,

                    fn subWrap(lhs: Awaiting, rhs: Awaiting) Awaiting {
                        return @fromBackingInt(@intCast(@backingInt(lhs) -% @backingInt(rhs)));
                    }

                    pub fn fromToken(token: u31) Awaiting {
                        const awaiting: Awaiting = @fromBackingInt(token);
                        switch (awaiting) {
                            .nothing, .group => unreachable,
                            _ => return awaiting,
                        }
                    }

                    pub fn toToken(awaiting: Awaiting) u31 {
                        switch (awaiting) {
                            .nothing, .group => unreachable,
                            _ => return @backingInt(awaiting),
                        }
                    }
                };

                /// Returns whether cancelation was requested.
                pub fn changeAwaiting(
                    cancel_status: *CancelStatus,
                    old_awaiting: Awaiting,
                    new_awaiting: Awaiting,
                ) bool {
                    const old_cancel_status = @atomicRmw(CancelStatus, cancel_status, .Add, .{
                        .requested = false,
                        .awaiting = new_awaiting.subWrap(old_awaiting),
                    }, .monotonic);
                    assert(old_cancel_status.awaiting == old_awaiting);
                    return old_cancel_status.requested;
                }
            };

            pub const CancelProtection = packed struct {
                user: Io.CancelProtection,
                acknowledged: bool,

                pub const unblocked: CancelProtection = .{ .user = .unblocked, .acknowledged = false };

                pub fn check(cancel_protection: CancelProtection) Io.CancelProtection {
                    return @fromBackingInt(@intCast(@intFromBool(cancel_protection != unblocked)));
                }

                pub fn acknowledge(cancel_protection: *CancelProtection) void {
                    assert(!cancel_protection.acknowledged);
                    cancel_protection.acknowledged = true;
                }

                pub fn recancel(cancel_protection: *CancelProtection) void {
                    assert(cancel_protection.acknowledged);
                    cancel_protection.acknowledged = false;
                }

                test check {
                    try std.testing.expectEqual(Io.CancelProtection.unblocked, check(.unblocked));
                    try std.testing.expectEqual(Io.CancelProtection.blocked, check(.{
                        .user = .unblocked,
                        .acknowledged = true,
                    }));
                    try std.testing.expectEqual(Io.CancelProtection.blocked, check(.{
                        .user = .blocked,
                        .acknowledged = false,
                    }));
                    try std.testing.expectEqual(Io.CancelProtection.blocked, check(.{
                        .user = .blocked,
                        .acknowledged = true,
                    }));
                }
            };

            /// Like a `*Task`, but 2 bits smaller than a pointer (because the LSBs are always 0 due
            /// to alignment) so that those two bits can be used in a `packed struct`.
            pub const PackedPtr = enum(@Int(.unsigned, @bitSizeOf(usize) - 2)) {
                null = 0,
                all_ones = std.math.maxInt(@Int(.unsigned, @bitSizeOf(usize) - 2)),
                _,

                const Split = packed struct(usize) { low: u2, high: PackedPtr };
                pub fn pack(ptr: ?*Task) PackedPtr {
                    const split: Split = @bitCast(@intFromPtr(ptr));
                    assert(split.low == 0);
                    return split.high;
                }
                pub fn unpack(ptr: PackedPtr) ?*Task {
                    const split: Split = .{ .low = 0, .high = ptr };
                    return @ptrFromInt(@as(usize, @bitCast(split)));
                }
            };

            pub fn resultPointer(task: *Task, comptime Result: type) *Result {
                return @ptrCast(@alignCast(task.resultBytes(.of(Result))));
            }

            pub fn resultBytes(task: *Task, alignment: Alignment) [*]u8 {
                return @ptrFromInt(alignment.forward(@intFromPtr(task) + @sizeOf(Task)));
            }
        };

        pub const Worker = struct {
            backend: Backend.Worker,
            sched: *Sched,
            index: u32,
            thread: std.Thread,
            idle_context: Io.fiber.Context,
            current_context: *Io.fiber.Context,
            /// The task this worker runs next, which no other worker takes unless this one is
            /// stuck. It is atomic because a worker that is stuck has its slot taken from it;
            /// otherwise only this worker touches it.
            run_next: std.atomic.Value(?*Task),
            lifo_streak: u8,
            /// Runnable tasks pinned here, oldest first.
            pinned: TaskList,
            /// Runnable sticky tasks.
            local: LocalQueue,
            /// Tasks other workers made runnable here.
            inbox: Inbox,
            /// Set while blocked in `poll` with nothing to run. A worker that clears it wakes this one.
            parked: std.atomic.Value(bool),
            /// Set by `notify` when it wakes this worker to look for work: the worker counts in
            /// `Idle.searching` from then on, and takes the count over when it next looks.
            woken_to_search: std.atomic.Value(bool),
            /// Switches this worker made out of a task, the watchdog's second sample. See
            /// `running`.
            tick: std.atomic.Value(u32),
            steal_start: u32,
            stacks: StackCache,
            /// Tasks spawned here minus tasks that ended here. The sum over all workers counts
            /// the tasks alive.
            live: isize,
            /// The ids this worker gives the tasks it spawns: `ids.next` up to `ids.end`, a block
            /// taken from `Task.next_id`, so that a spawn takes an id without a locked instruction
            /// on a word every worker writes.
            ids: struct { next: u64 = 0, end: u64 = 0 } = .{},
            /// The task this worker is running, or `null` while it runs no task. This worker
            /// publishes it with a release store at every switch; the watchdog reads it to tell
            /// a running task from a worker that is looking for work, and to name a stuck task.
            running: std.atomic.Value(?*Task) align(std.atomic.cache_line),
            /// Set by the watchdog while this worker is stuck: its queued tasks are takeable
            /// whatever their number. Cleared, by the watchdog, once the worker switches again.
            stranded: std.atomic.Value(bool),
            /// Set while this worker is a replacement for a stuck one. See `Spare`.
            spare: ?*Spare,
            idle_tsan_fiber: tsan.Fiber,

            threadlocal var self: ?*Worker = null;

            /// The worker running on this thread. Not inline, and never cached across a switch:
            /// a task can resume on another thread after any Io call.
            pub noinline fn current() *Worker {
                return self orelse
                    @panic("std.Io.Threadz used from a thread that is not one of its workers");
            }

            /// `null` on a thread that is not a worker.
            pub noinline fn currentOrNull() ?*Worker {
                return self;
            }

            pub inline fn currentTask(w: *Worker) *Task {
                assert(w.current_context != &w.idle_context);
                return @alignCast(@fieldParentPtr("context", w.current_context));
            }

            /// An id for a task this worker spawns, from the block in `ids`, taking the next
            /// block from `Task.next_id` when this one runs out. Ids are unique in the program.
            fn nextTaskId(w: *Worker) u64 {
                if (w.ids.next == w.ids.end) {
                    @branchHint(.unlikely);
                    w.ids.next = Task.next_id.fetchAdd(Task.id_block, .monotonic);
                    w.ids.end = w.ids.next + Task.id_block;
                }
                defer w.ids.next += 1;
                return w.ids.next;
            }

            fn tsanFiberOf(w: *Worker, context: *Io.fiber.Context) *anyopaque {
                if (context == &w.idle_context) return w.idle_tsan_fiber;
                const task: *Task = @alignCast(@fieldParentPtr("context", context));
                return task.tsan_fiber;
            }
        };

        const TaskList = struct {
            head: ?*Task = null,
            tail: ?*Task = null,

            fn push(list: *TaskList, task: *Task) void {
                task.status = .{ .queue_next = null };
                if (list.tail) |tail| tail.status.queue_next = task else list.head = task;
                list.tail = task;
            }

            fn pop(list: *TaskList) ?*Task {
                const task = list.head orelse return null;
                list.head = task.status.queue_next;
                if (list.head == null) list.tail = null;
                task.status.queue_next = null;
                return task;
            }
        };

        /// The queue any worker takes from: free tasks, and sticky ones that overflowed a
        /// worker's queue.
        const TaskQueue = struct {
            lock: SpinLock align(std.atomic.cache_line) = .{},
            list: TaskList = .{},
            /// Written under `lock`, read without it to see whether there is anything to take.
            len: std.atomic.Value(u32) = .init(0),

            fn push(q: *TaskQueue, tasks: []const *Task) void {
                q.lock.lock();
                defer q.lock.unlock();
                for (tasks) |task| q.list.push(task);
                q.len.store(q.len.raw + @as(u32, @intCast(tasks.len)), .seq_cst); // see `notify`
            }

            fn pop(q: *TaskQueue) ?*Task {
                if (q.len.load(.seq_cst) == 0) return null;
                q.lock.lock();
                defer q.lock.unlock();
                const task = q.list.pop() orelse return null;
                q.len.store(q.len.raw - 1, .monotonic);
                return task;
            }
        };

        /// A worker's queue of runnable sticky tasks. Only that worker adds to it. It takes the
        /// oldest one at a time, and other workers take half of what exceeds `steal_threshold`.
        const LocalQueue = struct {
            head: std.atomic.Value(u32) align(std.atomic.cache_line) = .init(0),
            tail: std.atomic.Value(u32) align(std.atomic.cache_line) = .init(0),
            slots: [capacity]?*Task align(std.atomic.cache_line) = @splat(null),

            const capacity = 256;

            // A worker may read a slot for a task that another worker takes first, so every
            // access to the slots is atomic.
            fn load(q: *LocalQueue, index: u32) *Task {
                return @atomicLoad(?*Task, &q.slots[index % capacity], .unordered).?;
            }
            fn store(q: *LocalQueue, index: u32, task: *Task) void {
                @atomicStore(?*Task, &q.slots[index % capacity], task, .unordered);
            }

            fn len(q: *LocalQueue) u32 {
                const head = q.head.load(.seq_cst);
                const tail = q.tail.load(.seq_cst); // see `notify`
                return @min(tail -% head, capacity);
            }

            /// Owner only. Appends `task`. When the queue is full, moves its older half and `task`
            /// to `overflow` instead.
            fn push(q: *LocalQueue, task: *Task, overflow: *TaskQueue) void {
                while (true) {
                    const head = q.head.load(.acquire); // acquire: other workers are done with their slots
                    const tail = q.tail.raw;
                    if (tail -% head < capacity) {
                        q.store(tail, task);
                        q.tail.store(tail +% 1, .seq_cst); // release the slot; `.seq_cst`: see `notify`
                        return;
                    }
                    const n = (tail -% head) / 2;
                    var batch: [capacity / 2 + 1]*Task = undefined;
                    for (batch[0..n], 0..) |*b, i| b.* = q.load(head +% @as(u32, @intCast(i)));
                    // Take them the way another worker would. If one got there first, there is room now.
                    if (q.head.cmpxchgStrong(head, head +% n, .acq_rel, .monotonic) != null) continue;
                    batch[n] = task;
                    overflow.push(batch[0 .. n + 1]);
                    return;
                }
            }

            /// Owner only. Takes the oldest task.
            fn pop(q: *LocalQueue) ?*Task {
                var head = q.head.load(.acquire);
                while (head != q.tail.raw) {
                    const task = q.load(head);
                    head = q.head.cmpxchgWeak(head, head +% 1, .acq_rel, .acquire) orelse return task;
                }
                return null;
            }

            /// Owner of `q` only, and only while `q` is empty. Moves the older half of what
            /// `victim` has beyond `threshold`, rounded up, into `q` without publishing them,
            /// and returns how many. `threshold` is `steal_threshold`, or 0 for a worker that is
            /// stuck, whose tasks are all takeable.
            fn grab(q: *LocalQueue, victim: *LocalQueue, threshold: u32) u32 {
                const tail = q.tail.raw;
                while (true) {
                    const victim_head = victim.head.load(.acquire);
                    const victim_tail = victim.tail.load(.seq_cst); // see `notify`
                    const available = victim_tail -% victim_head;
                    if (available > capacity) continue; // `victim_head` was read before `victim_tail` moved on
                    if (available <= threshold) return 0;
                    const excess = available - threshold;
                    const n = excess - excess / 2;
                    for (0..n) |i| {
                        const offset: u32 = @intCast(i);
                        q.store(tail +% offset, victim.load(victim_head +% offset));
                    }
                    if (victim.head.cmpxchgWeak(victim_head, victim_head +% n, .acq_rel, .monotonic) == null) return n;
                }
            }
        };

        /// Tasks other workers made runnable for one worker, newest first. Any worker pushes; the
        /// owner takes them all at once.
        const Inbox = struct {
            head: std.atomic.Value(?*Task) align(std.atomic.cache_line) = .init(null),

            fn push(inbox: *Inbox, task: *Task) void {
                var head = inbox.head.load(.monotonic);
                while (true) {
                    task.status = .{ .queue_next = head };
                    head = inbox.head.cmpxchgWeak(head, task, .seq_cst, .monotonic) orelse return; // see `park`
                }
            }

            fn isEmpty(inbox: *Inbox) bool {
                return inbox.head.load(.seq_cst) == null; // see `park`
            }

            fn takeAll(inbox: *Inbox) ?*Task {
                if (inbox.head.load(.monotonic) == null) return null;
                return inbox.head.swap(null, .acquire);
            }
        };

        const Idle = struct {
            /// Workers looking through the other workers' queues.
            searching: std.atomic.Value(u32) align(std.atomic.cache_line) = .init(0),
            /// Workers blocked in `poll` with nothing to run, or about to be.
            parked: std.atomic.Value(u32) = .init(0),
            lock: SpinLock = .{},
            /// Indexes of parked workers that nobody has woken yet, under `lock`.
            stack: []u32,
            len: u32 = 0,
        };

        const FreeMapping = struct { next: ?*FreeMapping };

        /// A worker the watchdog starts to replace a stuck one. A spare is not counted against
        /// `limit`: it runs what was queued behind the stuck task, and it stops once it has
        /// nothing to run and the worker it replaced switches out again.
        pub const Spare = struct {
            /// Its worker, whose index is past every worker in `workers`.
            worker: Worker = undefined,
            thread: std.Thread = undefined,
            state: std.atomic.Value(State) = .init(.unused),
            /// The worker this spare replaces. Only the watchdog writes it, before the thread
            /// starts, and the spare reads it.
            assigned: std.atomic.Value(u32) = .init(0),

            pub const State = enum(u8) {
                /// No thread: another stuck worker may use this slot.
                unused,
                /// Its thread runs, whatever it is doing.
                running,
                /// Its thread has returned. The watchdog joins it and frees the slot.
                exited,
            };
        };

        /// What the watchdog keeps about the last stuck episode, for `stats`.
        const StuckReport = struct {
            /// Which kind it was: see `Stats.Kind`.
            kind: std.atomic.Value(Stats.Kind) = .init(.blocked),
            /// The task's number, or 0 before the first episode.
            id: std.atomic.Value(u64) = .init(0),
            /// The name of the function the task runs, or null before the first episode.
            name: std.atomic.Value(?[*:0]const u8) = .init(null),
            /// How long the task had not switched out, as the watchdog sampled it.
            ms: std.atomic.Value(u64) = .init(0),
            /// The worker it ran on.
            worker: std.atomic.Value(u32) = .init(0),
        };

        /// What a program can read about a Threadz instance: the gauges the runtime keeps. The
        /// rest of the observability plan, the task dump and the scheduler meters, is Threadz
        /// step 7.
        pub const Stats = struct {
            /// The most workers that may run, counting worker 0 and not replacements.
            worker_limit: u32,
            /// The workers running now: worker 0, the workers started for work, and the
            /// replacements the watchdog started for stuck ones.
            workers: u32,
            /// The replacements among `workers`.
            spares: u32,
            /// The workers stuck right now.
            stuck: u32,
            /// Episodes in which a worker was found blocked: its current task had not switched
            /// out for `stuck_after` and its thread used less than a fifth of the wall time, so
            /// it is in the kernel or in a call that blocks. The watchdog writes one log line per
            /// episode, names the task, and starts a replacement for the worker.
            stuck_episodes: u64,
            /// Episodes in which a worker was found computing: its current task had not switched
            /// out for `stuck_after` and its thread kept using its time, so it is running CPU
            /// work with no Io call in it. Its queued tasks become takeable, but nothing is
            /// replaced and nothing is logged, because that is what compiling is.
            computing_episodes: u64,
            /// The replacements the watchdog started for stuck workers.
            replacements: u64,
            /// Rounds of samples the watchdog made. It makes none while every worker is parked:
            /// it waits, without a timeout, until one of them runs a task again.
            watchdog_rounds: u64,
            /// The last stuck episode, or null if there has been none.
            last_stuck: ?Stuck = null,

            pub const Stuck = struct {
                /// Which kind of episode this was: a blocked worker or a computing one.
                kind: Kind,
                /// The task's number. See `Task.id`.
                id: u64,
                /// The name of the function the task runs.
                name: [:0]const u8,
                /// How long the task had not switched out, as the watchdog sampled it: at least
                /// `stuck_after`, and at most that plus one `watchdog_interval`.
                ms: u64,
                /// The worker it ran on.
                worker: u32,
            };

            /// The two kinds of stuck worker: see `stuck_episodes` and `computing_episodes`.
            pub const Kind = enum(u8) { blocked, computing };
        };

        /// What the watchdog keeps about one worker between samples.
        const Sample = struct {
            /// The task that worker runs, or null while it runs no task.
            task: ?*Task = null,
            /// Its switch count when the sample last saw it change.
            tick: u32 = 0,
            /// When it last saw this task with this switch count.
            since: i96 = 0,
            /// Whether this worker has been reported and has not switched out since.
            reported: bool = false,
            /// Set by the round that first found this worker over `stuck_after`, so that the next
            /// round can tell a blocked worker from a computing one by its thread's CPU time: see
            /// `watchdogWorker`.
            measuring: ?Measuring = null,

            const Measuring = struct {
                /// The CPU time of the worker's thread when the measurement started, or null when
                /// the backend cannot say.
                cpu: ?u64,
                /// When it was read, on the same clock as the rounds.
                at: i96,
            };
        };

        /// Mappings of the default size kept by one worker, linked through their top bytes.
        const StackCache = struct {
            head: ?*FreeMapping = null,
            len: u32 = 0,
        };

        /// Mappings of the default size any worker takes from, once its cache is empty or full.
        const StackPool = struct {
            lock: SpinLock = .{},
            head: ?*FreeMapping = null,
            len: u32 = 0,
        };

        /// The pool the calls that may block a worker for a long time run on: `io.blocking`, and
        /// a regular-file `fsync`. It is an `Io.Threaded` instance this one starts with the first
        /// such call, and the group its jobs belong to; one job is one task of that pool, and it
        /// destroys itself when it finishes, so nothing awaits the group. Every core on this
        /// scheduler shares the pool.
        pub const Dirty = struct {
            lock: Io.Mutex = .init,
            /// `null` until the first blocking call starts the pool.
            threaded: ?*Io.Threaded = null,
            group: Io.Group = .init,

            fn deinit(d: *Dirty, s: *Sched) void {
                const threaded = d.threaded orelse return;
                threaded.deinit();
                Backend.allocator(s.backendOf()).destroy(threaded);
                d.threaded = null;
            }

            /// Hands `job` to the pool, starting it if it is not running yet. `false` when there
            /// is no thread to hand it to: the pool is at its limit, or it could not be started,
            /// and the caller then makes the call itself.
            fn submit(d: *Dirty, s: *Sched, job: *const Job, name: [:0]const u8) bool {
                const io = Backend.io(s.backendOf());
                d.lock.lock(io) catch return false; // canceled: the caller makes the call itself
                defer d.lock.unlock(io);
                const threaded = d.threaded orelse started: {
                    // The pool's threads allocate through it too, so it is the allocator this
                    // instance hands out, which is thread-safe.
                    const threaded = Backend.allocator(s.backendOf()).create(Io.Threaded) catch return false;
                    threaded.* = Io.Threaded.init(Backend.allocator(s.backendOf()), .{});
                    d.threaded = threaded;
                    break :started threaded;
                };
                const threaded_io = threaded.io();
                threaded_io.vtable.groupConcurrent(
                    threaded_io.userdata,
                    &d.group,
                    @ptrCast(job),
                    .of(*Job),
                    name,
                    Job.run,
                ) catch return false;
                return true;
            }
        };

        /// One call handed to the pool: what it runs, where the result goes, and how the calling
        /// task is made runnable again. It lives on the calling task's stack, which stays put
        /// while the task is parked, so the pool reads the arguments and writes the result there
        /// without a copy.
        pub const Job = struct {
            sched: *Sched,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
            context: *const anyopaque,
            result: *anyopaque,
            /// The task waiting for the call.
            task: *Task,
            /// `waiting` until either the call has returned or the task has parked, whichever
            /// comes first, and then `done` or `parked`. Exactly one of the pool's thread and the
            /// parked task makes the task runnable, and this word is what tells them which: the
            /// two exchange it with the same expected value, so only one of them wins. A word of
            /// the caller's, since the pool reads the job through a const pointer. See `blocking`.
            state: *std.atomic.Value(State),

            pub const State = enum(u8) { waiting, parked, done };

            /// On the pool's thread: makes the call, then makes the task runnable.
            fn run(context: *const anyopaque) void {
                const job: *const Job = @ptrCast(@alignCast(context));
                job.start(job.context, job.result);
                if (job.state.cmpxchgStrong(.waiting, .done, .seq_cst, .monotonic) == null) {
                    @branchHint(.unlikely);
                    return; // the task has not parked yet, so it wakes itself when it does
                }
                assert(job.state.load(.monotonic) == .parked);
                job.sched.readyFromForeign(job.task);
            }
        };

        /// `Io.blocking`: `start` runs with `context` on a thread of the instance's pool, writing
        /// its result to `result`, and the calling task parks while it does, so the worker runs
        /// other tasks meanwhile. A call that has started is not cancelable. A thread that is not
        /// one of the workers has no task to park, and a job the pool will not take has no thread
        /// to run it: both make the call on the calling thread, which holds its worker for as long
        /// as it blocks.
        pub fn blocking(
            userdata: ?*anyopaque,
            result: []u8,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) void {
            _ = result_alignment;
            _ = context_alignment;
            charge(fromUserdata(userdata));
            const s = fromUserdata(userdata);
            const w = Worker.currentOrNull() orelse {
                start(context.ptr, result.ptr);
                return;
            };
            var state: std.atomic.Value(Job.State) = .init(.waiting);
            var job: Job = .{
                .sched = s,
                .start = start,
                .context = context.ptr,
                .result = result.ptr,
                .task = w.currentTask(),
                .state = &state,
            };
            if (!s.dirty.submit(s, &job, name)) {
                start(context.ptr, result.ptr);
                return;
            }
            // The task is switched away before either side can make it runnable: the `custom`
            // pending task runs once it is saved, and the exchange there decides who wakes it.
            s.yield(null, .{ .custom = .{ .context = &job, .run = blockingParked } });
        }

        /// Runs after a task that handed a job to the pool has switched away: see `Job.state`.
        fn blockingParked(s: *Sched, task: *Task, context: *anyopaque) void {
            const job: *Job = @ptrCast(@alignCast(context));
            if (job.state.cmpxchgStrong(.waiting, .parked, .seq_cst, .monotonic) != null) {
                @branchHint(.unlikely);
                // The call returned before this task was switched away, so the pool's thread left
                // the wake to this side, which is not running anything else yet.
                assert(job.state.load(.monotonic) == .done);
                s.ready(.current(), task);
            }
        }

        pub const SpinLock = struct {
            locked: std.atomic.Value(bool) = .init(false),

            pub fn lock(l: *SpinLock) void {
                var spins: u32 = 0;
                while (l.locked.swap(true, .acquire)) {
                    while (l.locked.load(.monotonic)) {
                        spins +%= 1;
                        if (spins < 64) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
                    }
                }
            }

            pub fn unlock(l: *SpinLock) void {
                l.locked.store(false, .release);
            }
        };

        pub inline fn backendOf(s: *Sched) *Backend {
            return @fieldParentPtr("sched", s);
        }

        /// The worker with this index: one of `workers`, or the replacement running in a spare
        /// slot. `null` for a spare that is not running.
        pub fn workerAt(s: *Sched, index: u32) ?*Worker {
            if (index < s.workers.len) return &s.workers[index];
            const spare = &s.spares[index - s.workers.len];
            if (spare.state.load(.acquire) != .running) return null;
            return &spare.worker;
        }

        fn fromUserdata(userdata: ?*anyopaque) *Sched {
            const backend: *Backend = @ptrCast(@alignCast(userdata));
            return &backend.sched;
        }

        pub fn mainTask(s: *Sched) *Task {
            return @ptrCast(&s.main_task_buffer);
        }

        /// Makes the calling thread worker 0, running as the main task. `s` must stay put until
        /// `deinit`, and the backend's fields must be ready for `workerInit`.
        pub fn init(s: *Sched, gpa: Allocator, options: Options) !void {
            const count: u32 = @intCast(@max(1, options.workers orelse std.Thread.getCpuCount() catch 1));
            const page = std.heap.pageSize();
            const workers = try gpa.alloc(Worker, count);
            errdefer gpa.free(workers);
            const spares = try gpa.alloc(Spare, count);
            errdefer gpa.free(spares);
            const idle_stack = try mapStack(idle_stack_size);
            errdefer posix.munmap(idle_stack);
            const idle_indexes = try gpa.alloc(u32, count);
            errdefer gpa.free(idle_indexes);
            s.* = .{
                .workers = workers,
                .started = .init(1),
                .reserved = .init(1),
                .limit = .init(count),
                .spares = spares,
                .shared = .{},
                .idle = .{ .stack = idle_indexes },
                .stopping = .init(false),
                .stack_size = options.stack_size,
                .mapping_len = std.mem.alignForward(usize, page + options.stack_size + default_header_size, page),
                .pool = .{},
                .idle_stack = idle_stack,
                .watchdog = undefined,
                .watchdog_started = .init(false),
                .watchdog_wait = .init(0),
                .watchdog_sleeping = .init(false),
                .watchdog_rounds = .init(0),
                .stuck_episodes = .init(0),
                .computing_episodes = .init(0),
                .replacements = .init(0),
                .report = .{},
                .reaped_live = 0,
                .main_task_buffer = undefined,
                .dirty = .{},
            };
            @memset(s.spares, .{});
            const main_task = s.mainTask();
            main_task.* = .{
                .context = undefined,
                .link = .{ .awaiter = null },
                .status = .{ .queue_next = null },
                .cancel_status = .unrequested,
                .cancel_protection = .unblocked,
                .affinity = .sticky,
                .home = 0,
                .start = .main,
                .result_align = .@"1",
                .ops = 0,
                .context_bytes = undefined,
                .mapping = &.{},
                .id = Task.next_id.fetchAdd(1, .monotonic),
                .name = "main task",
                .tsan_fiber = if (tsan.enable) tsan.__tsan_get_current_fiber(),
            };
            const w = &workers[0];
            w.* = s.newWorker(0);
            w.current_context = &main_task.context;
            const idle_end = @intFromPtr(idle_stack.ptr) + idle_stack.len;
            w.idle_context = switch (builtin.cpu.arch) {
                .aarch64, .riscv64 => .{ .sp = idle_end, .fp = @intFromPtr(s), .pc = @intFromPtr(&mainIdleEntry) },
                .x86_64 => .{ .rsp = idle_end, .rbp = @intFromPtr(s), .rip = @intFromPtr(&mainIdleEntry) },
                else => |arch| @compileError("unimplemented architecture: " ++ @tagName(arch)),
            };
            if (tsan.enable) w.idle_tsan_fiber = tsan.__tsan_create_fiber(0);
            errdefer if (tsan.enable) tsan.__tsan_destroy_fiber(w.idle_tsan_fiber);
            try Backend.workerInit(s.backendOf(), w);
            // Worker 0 runs on this thread, so its `workerStart` is this call: a backend records
            // what only a thread can know about itself, such as its id, there.
            Backend.workerStart(s.backendOf(), w);
            Worker.self = w;
            if (tracy.enable) tracy.fiberEnter(main_task.name);
        }

        fn newWorker(s: *Sched, index: u32) Worker {
            return .{
                .backend = undefined,
                .sched = s,
                .index = index,
                .thread = undefined,
                .idle_context = undefined,
                .current_context = undefined,
                .run_next = .init(null),
                .lifo_streak = 0,
                .pinned = .{},
                .local = .{},
                .inbox = .{},
                .parked = .init(false),
                .woken_to_search = .init(false),
                .tick = .init(0),
                .steal_start = index,
                .stacks = .{},
                .live = 0,
                .running = .init(null),
                .stranded = .init(false),
                .spare = null,
                .idle_tsan_fiber = undefined,
            };
        }

        /// Called by the main task once every other task has ended. Stops the workers and frees
        /// everything, then returns on the thread that called `init`.
        pub fn deinit(s: *Sched, gpa: Allocator) void {
            assert(Worker.current().currentTask() == s.mainTask());
            s.yield(null, .stop);
            // Worker 0 has switched back to the main task on the thread that called `init`. A
            // worker may have been starting one more meanwhile, which `.stop` did not wake. Once
            // it has, every worker is woken again: each one sees `stopping` and returns.
            while (s.reserved.load(.acquire) != s.started.load(.acquire)) std.Thread.yield() catch {};
            // The watchdog first: it samples the workers, and it may still be starting a
            // replacement for one of them.
            s.stopWatchdog();
            const started = s.started.load(.acquire);
            for (s.workers[1..started]) |*w| Backend.wake(s.backendOf(), &s.workers[0], w);
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) != .running) continue;
                Backend.wake(s.backendOf(), &s.workers[0], &spare.worker);
            }
            for (s.workers[1..started]) |*w| w.thread.join();
            var live: isize = 0;
            for (s.workers[0..started]) |*w| {
                live += w.live;
                s.unmapStacks(w);
                Backend.workerDeinit(s.backendOf(), w);
            }
            // The replacements: joined and freed whatever they were doing, with their tasks
            // counted in `live` like any other worker's.
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) == .unused) continue;
                if (spare.state.load(.acquire) == .running) {
                    spare.thread.join();
                    const w = &spare.worker;
                    live += w.live;
                    s.unmapStacks(w);
                    Backend.workerDeinit(s.backendOf(), w);
                } else {
                    s.reapSpare(spare);
                }
            }
            live += s.reaped_live;
            assert(live == 0); // a task was never awaited
            while (s.pool.head) |free| {
                s.pool.head = free.next;
                s.unmapFree(free);
            }
            s.dirty.deinit(s);
            Worker.self = null;
            if (tsan.enable) tsan.__tsan_destroy_fiber(s.workers[0].idle_tsan_fiber);
            posix.munmap(s.idle_stack);
            gpa.free(s.idle.stack);
            gpa.free(s.spares);
            gpa.free(s.workers);
            s.* = undefined;
        }

        fn unmapStacks(s: *Sched, w: *Worker) void {
            while (w.stacks.head) |free| {
                w.stacks.head = free.next;
                s.unmapFree(free);
            }
        }

        /// Starts no more workers than `n`, counting worker 0, and at most as many as `init` allowed.
        pub fn setWorkerLimit(s: *Sched, n: usize) void {
            s.limit.store(@intCast(std.math.clamp(n, 1, s.workers.len)), .monotonic);
        }

        // Switching

        pub const SwitchMessage = struct {
            contexts: Io.fiber.Switch,
            pending_task: PendingTask,

            /// Done by the context switched to, once the one switched away from is saved.
            pub const PendingTask = union(enum) {
                nothing,
                /// The task that switched away is runnable again.
                reschedule,
                /// The task that switched away waits for this one to finish.
                await: *Task,
                /// This task returned, so whoever awaits it may take its result.
                finished: *Task,
                group_await: Group,
                group_cancel: Group,
                /// The task that switched away returned and is gone.
                destroy,
                /// Every worker stops once it has nothing to run.
                stop,
                /// Something for the backend, done with the task that switched away.
                custom: struct {
                    context: *anyopaque,
                    run: *const fn (s: *Sched, task: *Task, context: *anyopaque) void,
                },
            };

            fn handle(message: *const SwitchMessage, s: *Sched) void {
                const w: *Worker = .current();
                w.current_context = message.contexts.new;
                const new: ?*Task = if (message.contexts.new != &w.idle_context)
                    @alignCast(@fieldParentPtr("context", message.contexts.new))
                else
                    null;
                if (new) |task| {
                    switch (task.affinity) {
                        .pinned => assert(task.home == w.index),
                        .sticky, .free => @atomicStore(u32, &task.home, w.index, .monotonic),
                    }
                    if (tracy.enable) tracy.fiberEnter(task.name.ptr);
                } else if (tracy.enable) tracy.fiberLeave();
                const old: ?*Task = if (message.contexts.old != &w.idle_context)
                    @alignCast(@fieldParentPtr("context", message.contexts.old))
                else
                    null;
                if (old) |task| {
                    // It parked or yielded: its budget starts over.
                    task.ops = 0;
                }
                // What the watchdog samples. Release: it names the task it publishes, and the
                // task's fields are the ones the queue handoff made visible to this worker.
                w.running.store(new, .release);
                // The first task this instance runs starts the watchdog, on any worker, worker 0
                // included: an instance with one worker still reports a task that is stuck, and
                // the replacement is the only way anything else runs.
                if (new != null and !s.watchdog_started.load(.monotonic)) s.startWatchdog();
                switch (message.pending_task) {
                    .nothing => {},
                    .reschedule => if (old) |task| s.readyYielded(w, task),
                    .await => |awaiting| {
                        const awaiter = old.?;
                        assert(awaiter.status.queue_next == null);
                        if (@atomicRmw(?*Task, &awaiting.link.awaiter, .Xchg, awaiter, .acq_rel) == Task.finished)
                            s.ready(w, awaiter);
                    },
                    .finished => |task| {
                        const maybe_awaiter = @atomicRmw(?*Task, &task.link.awaiter, .Xchg, Task.finished, .acq_rel);
                        if (maybe_awaiter) |awaiter| {
                            // The one switched to already, unless its affinity kept it elsewhere.
                            if (&awaiter.context != message.contexts.new) s.ready(w, awaiter);
                        }
                    },
                    .group_await => |group| {
                        const task = old.?;
                        if (group.await(s, task)) s.ready(w, task);
                    },
                    .group_cancel => |group| {
                        const task = old.?;
                        if (group.cancel(s, task)) s.ready(w, task);
                    },
                    .destroy => s.destroyTask(w, old.?),
                    .stop => s.stopAll(w),
                    .custom => |custom| custom.run(s, old.?, custom.context),
                }
            }
        };

        inline fn contextSwitch(message: *const SwitchMessage) *const SwitchMessage {
            return @fieldParentPtr("contexts", Io.fiber.contextSwitch(&message.contexts));
        }

        /// Switches from the current task to `maybe_next`, or to the next runnable task, then does
        /// `pending` once the current task is saved. Returns when the current task is resumed.
        pub fn yield(s: *Sched, maybe_next: ?*Task, pending: SwitchMessage.PendingTask) void {
            const w: *Worker = .current();
            const next_context: *Io.fiber.Context = if (maybe_next) |next|
                &next.context
            else if (s.nextLocal(w)) |next|
                &next.context
            else
                &w.idle_context;
            const message: SwitchMessage = .{
                .contexts = .{ .old = w.current_context, .new = next_context },
                .pending_task = pending,
            };
            if (tsan.enable) tsan.__tsan_switch_to_fiber(w.tsanFiberOf(next_context), 0);
            contextSwitch(&message).handle(s);
        }

        /// Parks the current task until something makes it runnable, such as the completion of an
        /// operation it submitted.
        pub fn park(s: *Sched) void {
            s.yield(null, .nothing);
        }

        /// Charges one Io operation to the calling task, and yields it first when it has
        /// completed `budget` operations without parking: the task goes to the back of its
        /// worker's queue, keeping its affinity, and then the operation runs where the task
        /// landed.
        ///
        /// Called at the Threadz entry points, for every operation that can return without
        /// parking. An operation that then parks resets the count as it switches out, so
        /// charging an operation that always parks changes nothing. Does nothing on a thread
        /// that is not one of the workers: those threads run no task, and the operations they
        /// may call block in the kernel rather than park.
        pub inline fn charge(s: *Sched) void {
            _ = s.chargeFetch();
        }

        /// `charge`, returning the worker the calling task runs on afterwards: the one it ran on
        /// unless the charge yielded it, or `null` on a thread that is not a worker. An entry
        /// point that needs the current worker takes this one rather than looking it up again,
        /// which is what keeps the scheduler's own operations, all of them charged, to one
        /// lookup. The lookup is `Worker.currentOrNull`, never an inline read of the
        /// threadlocal: a task may resume on another thread after a switch, and a compiler may
        /// reuse a threadlocal's address anywhere within one function.
        pub inline fn chargeFetch(s: *Sched) ?*Worker {
            const w = Worker.currentOrNull() orelse return null;
            // The idle context runs on the worker's own stack, not in a task.
            if (w.current_context == &w.idle_context) return w;
            const task = w.currentTask();
            if (task.ops < budget) {
                task.ops += 1;
                return w;
            } else {
                @branchHint(.unlikely);
                task.ops = 0;
                s.yield(null, .reschedule);
                return Worker.currentOrNull();
            }
        }

        /// The next task this worker can run without looking at other workers or polling, or
        /// `null` every `poll_interval` switches so that its scheduling loop polls.
        fn nextLocal(s: *Sched, w: *Worker) ?*Task {
            const tick = w.tick.load(.monotonic) +% 1;
            w.tick.store(tick, .monotonic);
            if (tick % poll_interval == 0) return null;
            return s.takeLocal(w);
        }

        fn takeLocal(s: *Sched, w: *Worker) ?*Task {
            if (takeNext(w)) |task| {
                if (w.lifo_streak < lifo_limit) {
                    w.lifo_streak += 1;
                    return task;
                }
                // Tasks that keep waking each other do not get to starve the queue.
                w.local.push(task, &s.shared);
                w.lifo_streak = 0;
            } else w.lifo_streak = 0;
            s.drainInbox(w);
            if (w.pinned.pop()) |task| return task;
            if (w.local.pop()) |task| return task;
            return s.shared.pop();
        }

        /// Takes the task in this worker's slot for the next task, if any.
        ///
        /// A worker takes its own slot with no locked instruction, because the only other taker
        /// is the watchdog, and only for a worker it has declared stuck, with a protocol this
        /// cheap path cannot lose a race with. The watchdog's half is `takeStuckSlot`; together
        /// they are the asymmetric Dekker argument:
        ///
        /// The watchdog samples this worker's `running` task and its `tick`, and declares it
        /// stuck only when both stay the same for `stuck_after`. It then sets `stranded`, calls
        /// `membarrier`, and re-reads `running` and `tick`. Those are the words this worker
        /// stores at every switch, before it can reach this point: `tick` in `nextLocal` on the
        /// way out of a task, and `running` in `handle`, which is also the store that says the
        /// worker is looking for work rather than running a task. If the re-read sees either
        /// changed, the worker switched, and the watchdog takes nothing. If it sees both
        /// unchanged, then this worker has not reached the load below yet, and it cannot see the
        /// old `stranded`: the membarrier returns only once every thread of the process has
        /// passed a memory barrier, so the store of `stranded` is visible to a load that comes
        /// after this thread's own barrier — which the barrier below makes sure of, for the
        /// compiler — and this worker takes the compare-exchange path instead of this one.
        /// Either way one taker wins.
        fn takeNext(w: *Worker) ?*Task {
            // Without a barrier from the backend, the watchdog never takes a slot, and the
            // compare-exchange is always the safe taker.
            if (comptime !has_heavy_barrier) return takeNextExclusive(w);
            const maybe_task = w.run_next.load(.monotonic) orelse return null;
            // The stores this worker made at this switch must not be reordered after the load
            // below: see the note above.
            asm volatile ("" ::: .{ .memory = true });
            if (w.stranded.load(.monotonic) or !heavy_barrier_available.load(.monotonic)) {
                @branchHint(.unlikely);
                return takeNextExclusive(w);
            }
            w.run_next.store(null, .monotonic);
            return maybe_task;
        }

        /// `takeNext`, with a compare-exchange, which only one taker wins. For a worker that the
        /// watchdog has declared stuck, and for a process where `membarrier` is not available,
        /// where `takeNext`'s cheaper path cannot be used.
        fn takeNextExclusive(w: *Worker) ?*Task {
            var maybe_task = w.run_next.load(.monotonic);
            while (maybe_task) |task| {
                if (w.run_next.cmpxchgStrong(task, null, .acq_rel, .acquire) == null) return task;
                maybe_task = w.run_next.load(.monotonic);
            }
            return null;
        }

        fn drainInbox(s: *Sched, w: *Worker) void {
            var reversed: ?*Task = null;
            var maybe_task = w.inbox.takeAll();
            while (maybe_task) |task| {
                maybe_task = task.status.queue_next;
                task.status.queue_next = reversed;
                reversed = task;
            }
            var pushed = false;
            while (reversed) |task| {
                reversed = task.status.queue_next;
                task.status.queue_next = null;
                switch (task.affinity) {
                    .pinned => w.pinned.push(task),
                    .sticky => {
                        w.local.push(task, &s.shared);
                        pushed = true;
                    },
                    .free => unreachable,
                }
            }
            if (pushed and w.local.len() > steal_threshold) s.notify(w);
        }

        /// A worker's scheduling loop, on its idle context: runs tasks, looks for more, and blocks
        /// in `poll` when there are none. Returns once the scheduler stops.
        fn schedulingLoop(s: *Sched, w: *Worker) void {
            while (true) {
                Backend.poll(s.backendOf(), w, .nonblocking);
                // A worker `notify` woke counts as searching already.
                const counted = w.woken_to_search.load(.monotonic) and w.woken_to_search.swap(false, .seq_cst);
                // The shared queue first now and then, so that workers whose tasks keep them busy
                // do not starve it.
                const task = if (s.shared.pop() orelse s.takeLocal(w)) |task| task: {
                    if (counted) s.stopSearching(w);
                    break :task task;
                } else s.search(w, counted) orelse {
                    if (s.stopping.load(.acquire)) return;
                    // A replacement worker that has nothing to run and is no longer wanted
                    // stops here: the stuck worker it replaced switches out again.
                    if (w.spare) |spare| if (!s.spareWanted(spare)) return;
                    s.parkWorker(w);
                    continue;
                };
                const message: SwitchMessage = .{
                    .contexts = .{ .old = &w.idle_context, .new = &task.context },
                    .pending_task = .nothing,
                };
                if (tsan.enable) tsan.__tsan_switch_to_fiber(task.tsan_fiber, 0);
                contextSwitch(&message).handle(s);
            }
        }

        fn mainIdleEntry() callconv(.naked) void {
            switch (builtin.cpu.arch) {
                .aarch64 => asm volatile (
                    \\ mov x0, fp
                    \\ mov fp, #0
                    \\ b %[mainIdle]
                    :
                    : [mainIdle] "X" (&mainIdle),
                ),
                .riscv64 => asm volatile (
                    \\ mv a0, fp
                    \\ mv fp, zero
                    \\ tail %[mainIdle]@plt
                    :
                    : [mainIdle] "X" (&mainIdle),
                ),
                .x86_64 => asm volatile (
                    \\ movq %%rbp, %%rdi
                    \\ xor %%ebp, %%ebp
                    \\ jmp %[mainIdle:P]
                    :
                    : [mainIdle] "X" (&mainIdle),
                ),
                else => |arch| @compileError("unimplemented architecture: " ++ @tagName(arch)),
            }
        }

        fn mainIdle(
            s: *Sched,
            contexts: *const Io.fiber.Switch,
        ) callconv(.withStackAlign(.c, @alignOf(Io.fiber.Context))) noreturn {
            const message: *const SwitchMessage = @fieldParentPtr("contexts", contexts);
            message.handle(s);
            s.schedulingLoop(&s.workers[0]);
            // Stopped. The main task finishes `deinit` on this thread, where it started.
            s.yield(s.mainTask(), .nothing);
            unreachable; // the idle context is not resumed after stopping
        }

        fn workerEntry(s: *Sched, index: u32) void {
            const w = &s.workers[index];
            Worker.self = w;
            w.current_context = &w.idle_context;
            if (tsan.enable) w.idle_tsan_fiber = tsan.__tsan_get_current_fiber();
            Backend.workerStart(s.backendOf(), w);
            s.schedulingLoop(w);
        }

        // Finding work

        /// Looks through the shared queue and the other workers' queues a few times, the first
        /// searcher spinning for longer. Returns `null` if it found nothing to run. `counted` says
        /// that `notify` counted this worker as searching when it woke it.
        fn search(s: *Sched, w: *Worker, counted: bool) ?*Task {
            const first = counted or s.idle.searching.fetchAdd(1, .seq_cst) == 0;
            const spins: u32 = if (first) spin_limit else 0;
            var attempt: u32 = 0;
            while (attempt < @max(spins, search_rounds)) : (attempt += 1) {
                // A spinning worker looks through every queue only now and then.
                const scan = attempt < search_rounds or attempt % 16 == 0;
                const found = s.shared.pop() orelse
                    (if (!w.inbox.isEmpty()) s.takeLocal(w) else null) orelse
                    (if (scan) s.steal(w) else null);
                if (found) |task| {
                    s.stopSearching(w);
                    return task;
                }
                if (s.stopping.load(.monotonic)) break;
                std.atomic.spinLoopHint();
            }
            _ = s.idle.searching.fetchSub(1, .seq_cst);
            return null;
        }

        /// `w` found a task while counted as searching. Workers queueing tasks wake nobody while
        /// one searches, so the last searcher to find something wakes another one if more is
        /// waiting.
        fn stopSearching(s: *Sched, w: *Worker) void {
            if (s.idle.searching.fetchSub(1, .seq_cst) == 1 and s.anyStealable(w)) s.notify(w);
        }

        /// Takes tasks from another worker whose queue has more than `steal_threshold`, or any
        /// it has at all while it is stuck: the older half of the excess, returning the last one
        /// taken to run. A stuck worker's slot for the next task and the tasks others made
        /// runnable for it are taken too.
        fn steal(s: *Sched, w: *Worker) ?*Task {
            const started = s.started.load(.acquire);
            w.steal_start +%= 1;
            for (0..started) |i| {
                const victim = &s.workers[(w.steal_start +% i) % started];
                if (victim == w) continue;
                if (s.stealFrom(w, victim)) |task| return task;
            }
            // Replacements are workers too, and one of them can be the stuck one: what it took
            // from a stuck worker before that is still anyone's to take.
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) != .running) continue;
                const victim = &spare.worker;
                if (victim == w) continue;
                if (s.stealFrom(w, victim)) |task| return task;
            }
            return null;
        }

        /// Takes a task from one victim worker, whichever of its queues has one to give: see
        /// `steal`.
        fn stealFrom(s: *Sched, w: *Worker, victim: *Worker) ?*Task {
            const stranded = victim.stranded.load(.acquire);
            if (stranded) {
                // What only the worker itself could take before it got stuck: the tasks others
                // made runnable for it. Its slot for the next task is the watchdog's to take,
                // because only a slot's own worker takes one without a locked instruction: see
                // `takeNext`.
                if (s.takeInbox(w, victim)) |task| return task;
            }
            const n = w.local.grab(&victim.local, if (stranded) 0 else steal_threshold);
            if (n == 0) return null;
            const tail = w.local.tail.raw;
            const task = w.local.load(tail +% n -% 1);
            if (n > 1) w.local.tail.store(tail +% n -% 1, .seq_cst);
            return task;
        }

        /// Takes the tasks others made runnable for stuck worker `victim`, which cannot take
        /// them itself. Its pinned tasks stay there: they own its resources, and they wait for
        /// the worker as they would anyway. Returns one task for `w` to run, with the rest on its
        /// own queue.
        fn takeInbox(s: *Sched, w: *Worker, victim: *Worker) ?*Task {
            var maybe_task: ?*Task = victim.inbox.takeAll();
            var taken: ?*Task = null;
            while (maybe_task) |task| {
                maybe_task = task.status.queue_next;
                switch (task.affinity) {
                    .free => unreachable, // `drainInbox` never sees one either
                    .pinned => victim.inbox.push(task),
                    .sticky => {
                        // Out of the inbox's chain, as `drainInbox` leaves every task it takes:
                        // a task that keeps a link it is no longer on is one `destroyTask` stops
                        // at, and one a later push could follow.
                        if (taken) |previous| {
                            previous.status = .{ .queue_next = null };
                            w.local.push(previous, &s.shared);
                        }
                        taken = task;
                    },
                }
            }
            const task = taken orelse return null;
            task.status = .{ .queue_next = null };
            if (w.local.len() > steal_threshold) s.notify(w);
            return task;
        }

        fn anyStealable(s: *Sched, w: *Worker) bool {
            if (s.shared.len.load(.seq_cst) != 0) return true;
            for (s.workers[0..s.started.load(.acquire)]) |*victim| {
                if (victim == w) continue;
                const threshold: u32 = if (victim.stranded.load(.acquire)) 0 else steal_threshold;
                if (victim.local.len() > threshold) return true;
                if (threshold == 0 and !victim.inbox.isEmpty()) return true;
            }
            return false;
        }

        /// Blocks in `poll` until an event arrives, having announced it and then looked once more
        /// for work: whoever makes work for this worker afterwards sees it parked and wakes it.
        fn parkWorker(s: *Sched, w: *Worker) void {
            w.parked.store(true, .seq_cst);
            _ = s.idle.parked.fetchAdd(1, .seq_cst);
            {
                s.idle.lock.lock();
                defer s.idle.lock.unlock();
                s.idle.stack[s.idle.len] = w.index;
                s.idle.len += 1;
            }
            if (w.inbox.isEmpty() and !s.anyStealable(w) and !s.stopping.load(.seq_cst)) {
                Backend.poll(s.backendOf(), w, .block);
            }
            w.parked.store(false, .seq_cst);
            _ = s.idle.parked.fetchSub(1, .seq_cst);
            s.idle.lock.lock();
            defer s.idle.lock.unlock();
            for (s.idle.stack[0..s.idle.len], 0..) |index, i| {
                if (index != w.index) continue;
                s.idle.len -= 1;
                s.idle.stack[i] = s.idle.stack[s.idle.len];
                break;
            }
            // About to run a task again, or to look for one: the watchdog is waiting with no
            // timeout if it saw every worker parked, and it has to sample this one. Announced
            // only after this worker stopped counting as parked, so that a watchdog that is
            // deciding right now uses its timeout instead of waiting for a wake.
            s.notifyWatchdogUnparked();
        }

        /// Wakes `target` if it is parked and nobody has woken it yet.
        fn wakeWorker(s: *Sched, from: *Worker, target: *Worker) void {
            if (!target.parked.load(.seq_cst)) return; // see `parkWorker`
            if (target.parked.cmpxchgStrong(true, false, .seq_cst, .monotonic) != null) return;
            Backend.wake(s.backendOf(), from, target);
        }

        /// `wakeWorker`, from a thread that is not one of the workers: the watchdog, or a
        /// replacement worker it started, which can send no ring wake from the worker the wake
        /// would come from.
        pub fn wakeWorkerFromForeign(s: *Sched, target: *Worker) void {
            if (!target.parked.load(.seq_cst)) return; // see `parkWorker`
            if (target.parked.cmpxchgStrong(true, false, .seq_cst, .monotonic) != null) return;
            Backend.wakeForeign(s.backendOf(), target);
        }

        /// Finds a worker for work any worker may take: wakes a parked one, or starts one, unless
        /// a worker is searching already. `from` is the worker the wake comes from, or `null` for
        /// a caller that is not a worker, which is the watchdog.
        ///
        /// Queueing work is a `.seq_cst` store and the loads here are `.seq_cst`. A worker about
        /// to park announces it with `.seq_cst` stores, then looks at the queues with `.seq_cst`
        /// loads. So either this sees the worker searching or parked, or the worker sees the work.
        fn notify(s: *Sched, from: ?*Worker) void {
            if (s.idle.searching.load(.seq_cst) != 0) return;
            if (s.idle.parked.load(.seq_cst) != 0) {
                // The worker woken here counts as searching from now on, so that the work queued
                // before it gets to look wakes no other worker.
                if (s.idle.searching.cmpxchgStrong(0, 1, .seq_cst, .monotonic) != null) return;
                while (true) {
                    const index = index: {
                        s.idle.lock.lock();
                        defer s.idle.lock.unlock();
                        if (s.idle.len == 0) {
                            // All being woken already.
                            _ = s.idle.searching.fetchSub(1, .seq_cst);
                            return;
                        }
                        s.idle.len -= 1;
                        break :index s.idle.stack[s.idle.len];
                    };
                    const target = s.workerAt(index) orelse {
                        _ = s.idle.searching.fetchSub(1, .seq_cst);
                        return;
                    };
                    target.woken_to_search.store(true, .seq_cst);
                    if (target.parked.cmpxchgStrong(true, false, .seq_cst, .monotonic) == null) {
                        if (from) |worker| Backend.wake(s.backendOf(), worker, target) else Backend.wakeForeign(s.backendOf(), target);
                        return;
                    }
                    // Awake already. If it took the count over meanwhile, it searches; if not,
                    // the count goes to the next parked worker.
                    if (!target.woken_to_search.swap(false, .seq_cst)) return;
                }
            }
            s.startWorker(s.reserved.load(.monotonic)) catch {};
        }

        /// Starts the worker with this index, which must be the next one.
        fn startWorker(s: *Sched, index: u32) error{ LimitReached, Busy, SystemResources }!void {
            if (index >= s.limit.load(.monotonic)) return error.LimitReached;
            // A start in progress has to finish first.
            if (s.started.load(.acquire) != index) return error.Busy;
            if (s.reserved.cmpxchgStrong(index, index + 1, .acquire, .monotonic) != null) return error.Busy;
            errdefer s.reserved.store(index, .release);
            // `deinit` waits for this start to finish, then wakes the workers started by then.
            if (s.stopping.load(.acquire)) return error.LimitReached;
            const w = &s.workers[index];
            w.* = s.newWorker(index);
            Backend.workerInit(s.backendOf(), w) catch |err| {
                std.log.scoped(.threadz).warn("unable to start worker: {t}", .{err});
                return error.SystemResources;
            };
            w.thread = std.Thread.spawn(.{
                .stack_size = idle_stack_size,
                .allocator = Backend.allocator(s.backendOf()),
            }, workerEntry, .{ s, index }) catch |err| {
                Backend.workerDeinit(s.backendOf(), w);
                std.log.scoped(.threadz).warn("unable to start worker: {t}", .{err});
                return error.SystemResources;
            };
            // A worker's fields are ready before it is counted as started.
            s.started.store(index + 1, .release);
        }

        /// Starts workers until the one with this index is running.
        fn ensureWorker(s: *Sched, index: u32) error{ConcurrencyUnavailable}!void {
            while (s.started.load(.acquire) <= index) {
                s.startWorker(s.started.load(.acquire)) catch |err| switch (err) {
                    error.Busy => std.Thread.yield() catch {},
                    error.LimitReached, error.SystemResources => return error.ConcurrencyUnavailable,
                };
            }
        }

        fn stopAll(s: *Sched, w: *Worker) void {
            s.stopping.store(true, .release);
            for (s.workers[0..s.started.load(.acquire)]) |*target| {
                if (target != w) Backend.wake(s.backendOf(), w, target);
            }
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) != .running) continue;
                Backend.wake(s.backendOf(), w, &spare.worker);
            }
        }

        // The watchdog

        /// What a program can read about this instance. See `Stats`.
        pub fn stats(s: *Sched) Stats {
            const started = s.started.load(.acquire);
            var spares: u32 = 0;
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) == .running) spares += 1;
            }
            var stuck: u32 = 0;
            for (s.workers[0..started]) |*w| {
                if (w.stranded.load(.acquire)) stuck += 1;
            }
            // The episode counts first: see `classifyStuck`.
            const stuck_episodes = s.stuck_episodes.load(.acquire);
            const computing_episodes = s.computing_episodes.load(.acquire);
            const name = s.report.name.load(.acquire);
            return .{
                .worker_limit = s.limit.load(.monotonic),
                .workers = started + spares,
                .spares = spares,
                .stuck = stuck,
                .stuck_episodes = stuck_episodes,
                .computing_episodes = computing_episodes,
                .replacements = s.replacements.load(.monotonic),
                .watchdog_rounds = s.watchdog_rounds.load(.monotonic),
                .last_stuck = if (name) |n| .{
                    .kind = s.report.kind.load(.monotonic),
                    .id = s.report.id.load(.monotonic),
                    .name = std.mem.span(n),
                    .ms = s.report.ms.load(.monotonic),
                    .worker = s.report.worker.load(.monotonic),
                } else null,
            };
        }

        /// Starts the watchdog thread, once: with the first task the instance makes or runs, on
        /// any worker, worker 0 included. An instance that never makes or runs a task pays
        /// nothing for one.
        pub fn startWatchdog(s: *Sched) void {
            if (s.watchdog_started.swap(true, .seq_cst)) return;
            s.watchdog = std.Thread.spawn(.{
                .stack_size = idle_stack_size,
                .allocator = Backend.allocator(s.backendOf()),
            }, watchdogEntry, .{s}) catch |err| {
                s.watchdog_started.store(false, .release);
                std.log.scoped(.threadz).warn("unable to start the watchdog: {t}", .{err});
                return;
            };
        }

        /// Stops the watchdog and joins it. Called by `deinit` before the workers it samples go
        /// away.
        fn stopWatchdog(s: *Sched) void {
            if (!s.watchdog_started.load(.acquire)) return;
            const io = Backend.io(s.backendOf());
            s.watchdog_wait.store(1, .release);
            Io.futexWake(io, u32, &s.watchdog_wait.raw, 1);
            s.watchdog.join();
        }

        /// Waits between two rounds of samples, or until `deinit` stops the watchdog, or until a
        /// worker that was parked runs a task again. With every worker parked there is nothing to
        /// sample, so the wait has no timeout then and an idle program wakes the watchdog no
        /// more than it runs tasks. The wait is an Io futex, so a backend that blocks foreign
        /// threads in the kernel sleeps here rather than spinning.
        fn watchdogWait(s: *Sched) void {
            const io = Backend.io(s.backendOf());
            if (s.allParked()) {
                // The flag is set before the word is cleared, and the loop looks at the workers
                // after both: a worker that unparks after that look sees the flag and sets and
                // wakes the word, which ends the wait; one that unparks before it ends the loop
                // there. A wait that returns for any other reason, such as a spurious wake,
                // waits again, so that an idle program has no rounds at all.
                s.watchdog_sleeping.store(true, .release);
                defer s.watchdog_sleeping.store(false, .release);
                s.watchdog_wait.store(0, .release);
                while (s.watchdog_wait.load(.acquire) == 0 and s.allParked() and !s.stopping.load(.acquire)) {
                    Io.futexWaitTimeout(io, u32, &s.watchdog_wait.raw, 0, .none) catch {};
                }
                return;
            }
            s.watchdog_wait.store(0, .release);
            Io.futexWaitTimeout(io, u32, &s.watchdog_wait.raw, 0, .{ .duration = .{
                .raw = .fromNanoseconds(watchdog_interval),
                .clock = .awake,
            } }) catch {};
        }

        /// Whether every worker that exists is parked, which is when the watchdog has nothing to
        /// sample. A worker that is looking for work is not parked, and it parks or finds work
        /// soon.
        fn allParked(s: *Sched) bool {
            var workers = s.started.load(.acquire);
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) == .running) workers += 1;
            }
            return s.idle.parked.load(.seq_cst) >= workers;
        }

        /// Wakes the watchdog, which is waiting with no timeout while every worker is parked and
        /// would not sample this worker: it is about to run a task again.
        fn notifyWatchdogUnparked(s: *Sched) void {
            if (!s.watchdog_sleeping.load(.acquire)) return;
            const io = Backend.io(s.backendOf());
            s.watchdog_wait.store(1, .release);
            Io.futexWake(io, u32, &s.watchdog_wait.raw, 1);
        }

        /// The watchdog: it samples every worker every `watchdog_interval`, and reports a worker
        /// whose current task has not switched out for `stuck_after`. See `openStuck` and
        /// `classifyStuck`.
        fn watchdogEntry(s: *Sched) void {
            const gpa = Backend.allocator(s.backendOf());
            const samples = gpa.alloc(Sample, s.workers.len + s.spares.len) catch |err| {
                std.log.scoped(.threadz).warn("unable to start the watchdog: {t}", .{err});
                return;
            };
            defer gpa.free(samples);
            @memset(samples, .{});
            while (true) {
                s.watchdogWait();
                if (s.stopping.load(.acquire)) return;
                s.watchdogRound(samples);
            }
        }

        fn watchdogRound(s: *Sched, samples: []Sample) void {
            _ = s.watchdog_rounds.fetchAdd(1, .monotonic);
            const now = Io.Timestamp.now(Backend.io(s.backendOf()), .awake).nanoseconds;
            const started = s.started.load(.acquire);
            for (s.workers[0..started], samples[0..started]) |*w, *sample| s.watchdogWorker(w, sample, now);
            for (s.spares, samples[started..][0..s.spares.len]) |*spare, *sample| {
                if (spare.state.load(.acquire) != .running) continue;
                s.watchdogWorker(&spare.worker, sample, now);
            }
            // Replacements that stopped: joined here, where the thread that joins is not the one
            // that would have to wait for it in `deinit`.
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) != .exited) continue;
                s.reapSpare(spare);
            }
        }

        /// Samples one worker. A task that switches out, or a worker that runs no task, is doing
        /// something: the sample starts over. A task that does not for `stuck_after` is stuck.
        fn watchdogWorker(s: *Sched, w: *Worker, sample: *Sample, now: i96) void {
            const task = w.running.load(.acquire);
            const tick = w.tick.load(.monotonic);
            if (task == null or task != sample.task or tick != sample.tick) {
                // It switches out again: it is not stuck, and what it had queued goes back to
                // being takeable only when the queue is over the threshold.
                if (sample.reported or sample.measuring != null) s.releaseStranded(w);
                sample.* = .{ .task = task, .tick = tick, .since = now };
                return;
            }
            if (sample.reported) return; // reported already; until it switches out, it is stuck
            const elapsed = now - sample.since;
            if (elapsed < stuck_after) return;
            if (sample.measuring) |measuring| {
                // A round later, still the same task and the same switch count. A thread that
                // used less than a fifth of the wall time since it was first found is waiting for
                // something outside the process — a syscall, a lock, a C call — and a replacement
                // keeps its worker's share of the parallelism; one that kept using its time is
                // computing, which is what compiling is, and nothing about it needs replacing.
                sample.reported = true;
                s.classifyStuck(w, task.?, @intCast(@divTrunc(elapsed, std.time.ns_per_ms)), .{
                    .cpu = measuring.cpu,
                    .cpu_now = if (comptime has_thread_cpu_time) Backend.threadCpuTime(s.backendOf(), w) else null,
                    .wall = now - measuring.at,
                });
                return;
            }
            // First found stuck: its queue opens now, whatever it is doing, so that nothing
            // queued behind it waits for the round that tells what that is.
            s.openStuck(w, task.?, tick);
            sample.measuring = .{
                .cpu = if (comptime has_thread_cpu_time) Backend.threadCpuTime(s.backendOf(), w) else null,
                .at = now,
            };
        }

        /// The worker switches out again: what was queued on it is takeable only when the queue
        /// is over the threshold, and a replacement running for it is released.
        fn releaseStranded(s: *Sched, w: *Worker) void {
            w.stranded.store(false, .release);
            for (s.spares) |*spare| {
                if (spare.state.load(.acquire) != .running) continue;
                if (spare.assigned.load(.monotonic) != w.index) continue;
                // A replacement stops once it has nothing to run, so it has to look at itself
                // again to notice that it is no longer wanted.
                s.wakeWorkerFromForeign(&spare.worker);
            }
        }

        /// Worker `w` has run task `task` for `stuck_after` without switching out: its queued
        /// tasks become takeable whatever their number, the one in its slot included, and the
        /// idle workers are woken to take them. Whether it also gets a replacement is decided a
        /// round later, by `classifyStuck`.
        fn openStuck(s: *Sched, w: *Worker, task: *Task, tick: u32) void {
            w.stranded.store(true, .release);
            if (comptime has_heavy_barrier) {
                // Make the store above visible to every worker that has not looked at it yet, so
                // that a worker which is switching right now either sees it and takes its slot
                // with a compare-exchange, or has already made both of the words the re-read
                // below checks say so. See `takeNext`.
                if (Backend.heavyBarrier(s.backendOf())) {
                    const current = w.running.load(.acquire);
                    if (current != null and current.? == task and w.tick.load(.monotonic) == tick)
                        s.takeStuckSlot(w);
                } else {
                    // The backend cannot issue the barrier, so from here on every taker of a slot
                    // pays a compare-exchange: see `takeNext`. This round takes no slot, though:
                    // a worker that loaded the flag before the store above is inside `takeNext`'s
                    // cheaper path, a plain load and store, and nothing orders its take against
                    // this thread's swap — that is the window the barrier and the re-read exist
                    // to close. The worker's next take is exclusive, and the slot is the
                    // watchdog's from then on.
                    heavy_barrier_available.store(false, .release);
                }
            } else {
                // No barrier at all: every taker of a slot pays a compare-exchange already, which
                // this swap is one of, so the slot is the watchdog's like any stranded queue.
                s.takeStuckSlot(w);
            }
            s.wakeIdle();
        }

        /// A round after `openStuck`, with the worker still in the same task: tells a blocked
        /// worker from a computing one by its thread's CPU time, names the task in `stats`, and
        /// counts the episode. A blocked worker also gets one replacement worker and one log
        /// line; a computing one keeps its worker, since it is using it, so that working around
        /// it does not raise the parallelism a program asked for with `-j` and a worker limit.
        fn classifyStuck(s: *Sched, w: *Worker, task: *Task, ms: u64, measurement: Measurement) void {
            const blocked: bool = if (measurement.cpu) |before| blk: {
                const after = measurement.cpu_now orelse break :blk true; // cannot say: blocked
                if (after <= before) break :blk true; // no time used at all: waiting for something
                break :blk (after - before) * 5 < measurement.wall; // less than a fifth of the wall
            } else true; // the backend cannot say: treat it as blocked
            const kind: Stats.Kind = if (blocked) .blocked else .computing;
            s.report.kind.store(kind, .monotonic);
            s.report.ms.store(ms, .monotonic);
            s.report.worker.store(w.index, .monotonic);
            s.report.id.store(task.id, .monotonic);
            s.report.name.store(task.name.ptr, .release);
            // The episode counts are released after the report and the replacement's count, and
            // `stats` acquires them first: a reader that sees an episode sees what it reported.
            if (!blocked) {
                _ = s.computing_episodes.fetchAdd(1, .release);
                return;
            }
            const replaced = s.startSpare(w);
            _ = s.stuck_episodes.fetchAdd(1, .release);
            std.log.scoped(.threadz).warn(
                "task {d} ({s}) has not switched out for {d} ms on worker {d}: it is blocked, its queued tasks are takeable, and {s}",
                .{ task.id, task.name, ms, w.index, if (replaced) "a replacement worker runs in its place" else "no replacement worker could be started" },
            );
        }

        /// What the deciding round measured: the thread's CPU time before and after, and the wall
        /// time between the two reads.
        const Measurement = struct {
            /// The CPU time when the worker was first found, or null when the backend cannot say.
            cpu: ?u64 = null,
            /// The CPU time a round later, or null.
            cpu_now: ?u64 = null,
            /// The wall time between the two reads.
            wall: i96 = 0,
        };

        /// Takes the task in stuck worker `w`'s slot for the next task, which only its own worker
        /// would take, and puts it in the shared queue, where every worker looks. Called by the
        /// watchdog, which is the only other taker of a slot: see `takeNext`.
        fn takeStuckSlot(s: *Sched, w: *Worker) void {
            const task = w.run_next.swap(null, .acquire) orelse return;
            s.shared.push(&.{task});
            s.notify(null);
        }

        /// Wakes every parked worker, so that work made takeable at once is taken at once.
        fn wakeIdle(s: *Sched) void {
            for (s.workers[0..s.started.load(.acquire)]) |*target| s.wakeWorkerFromForeign(target);
        }

        /// Starts a replacement worker for stuck worker `w`, unless it has one already or every
        /// slot is taken. A replacement is not counted against `limit`. Returns whether `w` has a
        /// replacement now.
        fn startSpare(s: *Sched, w: *Worker) bool {
            for (s.spares, 0..) |*spare, i| {
                switch (spare.state.load(.acquire)) {
                    .unused => {
                        spare.assigned.store(w.index, .monotonic);
                        s.startSpareThread(spare, @intCast(i)) catch {
                            spare.state.store(.unused, .release);
                            return false;
                        };
                        // Last: the worker, its backend and its thread are ready before any
                        // other thread reads this slot, and a spare that looks at itself while
                        // it is still starting sees the worker it was assigned to, not its own
                        // state.
                        spare.state.store(.running, .release);
                        return true;
                    },
                    .running => if (spare.assigned.load(.monotonic) == w.index) return true,
                    .exited => {},
                }
            }
            return false;
        }

        /// Starts the thread of spare `i`, and the worker it runs on: its index is past every
        /// worker in `workers`. The replacement is counted before its thread starts, so that
        /// anything the replacement runs happens after the count, for whoever reads it.
        fn startSpareThread(s: *Sched, spare: *Spare, i: u32) error{SystemResources}!void {
            const w = &spare.worker;
            w.* = s.newWorker(@intCast(s.workers.len + i));
            w.spare = spare;
            Backend.workerInit(s.backendOf(), w) catch return error.SystemResources;
            errdefer Backend.workerDeinit(s.backendOf(), w);
            _ = s.replacements.fetchAdd(1, .monotonic);
            errdefer _ = s.replacements.fetchSub(1, .monotonic);
            spare.thread = std.Thread.spawn(.{
                .stack_size = idle_stack_size,
                .allocator = Backend.allocator(s.backendOf()),
            }, spareEntry, .{ s, spare }) catch return error.SystemResources;
        }

        /// The thread of a replacement worker: `workerEntry`, and the thread returns once it has
        /// nothing to run and the worker it replaced switches out again.
        fn spareEntry(s: *Sched, spare: *Spare) void {
            const w = &spare.worker;
            Worker.self = w;
            w.current_context = &w.idle_context;
            if (tsan.enable) w.idle_tsan_fiber = tsan.__tsan_get_current_fiber();
            Backend.workerStart(s.backendOf(), w);
            s.schedulingLoop(w);
            // Tasks that arrived for this worker as it stopped: the shared queue, which any
            // worker takes from. Nothing pinned can be here: pinned tasks have a worker of their
            // own, not a replacement.
            var maybe_task = w.inbox.takeAll();
            var any = false;
            while (maybe_task) |task| {
                maybe_task = task.status.queue_next;
                task.status = .{ .queue_next = null };
                s.shared.push(&.{task});
                any = true;
            }
            if (any) s.notify(w);
            spare.state.store(.exited, .release);
        }

        /// Joins a replacement that has stopped and frees its worker, so that the slot may hold
        /// another one.
        fn reapSpare(s: *Sched, spare: *Spare) void {
            spare.thread.join();
            const w = &spare.worker;
            // No fiber to destroy: a worker past worker 0 uses the fiber its thread already has,
            // which ends with the thread. Only worker 0's idle fiber is one this scheduler made.
            s.reaped_live += w.live;
            s.unmapStacks(w);
            Backend.workerDeinit(s.backendOf(), w);
            spare.state.store(.unused, .release);
        }

        /// Whether a replacement worker still has a reason to run: the worker it replaces is
        /// still stuck.
        fn spareWanted(s: *Sched, spare: *Spare) bool {
            if (spare.state.load(.acquire) == .exited) return false;
            return s.workers[spare.assigned.load(.monotonic)].stranded.load(.acquire);
        }

        // Making tasks runnable

        /// Makes `task` runnable where its affinity says.
        pub fn ready(s: *Sched, w: *Worker, task: *Task) void {
            switch (task.affinity) {
                .free => {
                    s.shared.push(&.{task});
                    s.notify(w);
                },
                .sticky, .pinned => {
                    const home = @atomicLoad(u32, &task.home, .monotonic);
                    if (home == w.index) return s.readyHere(w, task, .next);
                    // A task that last ran on a replacement whose thread has stopped goes to the
                    // shared queue: there is no worker of its own any more.
                    const target = s.workerAt(home) orelse return s.readyAnywhere(w, task);
                    target.inbox.push(task);
                    s.wakeWorker(w, target);
                },
            }
        }

        /// For backends: `task`'s operation finished on `w`.
        pub fn readyFromPoll(s: *Sched, w: *Worker, task: *Task) void {
            switch (task.affinity) {
                .free => {
                    s.shared.push(&.{task});
                    s.notify(w);
                },
                .sticky, .pinned => if (@atomicLoad(u32, &task.home, .monotonic) == w.index)
                    s.readyHere(w, task, .later)
                else
                    s.ready(w, task),
            }
        }

        /// `task` goes to the shared queue, which any worker takes from. For a task that has no
        /// worker of its own, such as one that last ran on a replacement worker that stopped.
        /// `from` is the worker the wake comes from, or `null` for a caller that is not one.
        fn readyAnywhere(s: *Sched, from: ?*Worker, task: *Task) void {
            s.shared.push(&.{task});
            s.notify(from);
        }

        /// Makes `task` runnable from a thread that is not one of the workers. `ready` for a
        /// caller with no worker to wake from, such as the test backend's futex wake, whose
        /// waiters are tasks of this scheduler rather than waiters of the kernel.
        pub fn readyFromForeign(s: *Sched, task: *Task) void {
            switch (task.affinity) {
                .free => {
                    s.shared.push(&.{task});
                    s.notify(null);
                },
                .sticky, .pinned => {
                    const home = @atomicLoad(u32, &task.home, .monotonic);
                    const target = s.workerAt(home) orelse return s.readyAnywhere(null, task);
                    target.inbox.push(task);
                    s.wakeWorkerFromForeign(target);
                },
            }
        }

        fn readyYielded(s: *Sched, w: *Worker, task: *Task) void {
            switch (task.affinity) {
                .free => {
                    s.shared.push(&.{task});
                    s.notify(w);
                },
                .sticky, .pinned => s.readyHere(w, task, .later),
            }
        }

        /// `task` belongs to `w`. `.next` runs it next, `.later` after the tasks already waiting.
        fn readyHere(s: *Sched, w: *Worker, task: *Task, when: enum { next, later }) void {
            switch (task.affinity) {
                .pinned => return w.pinned.push(task),
                .sticky => {},
                .free => unreachable,
            }
            if (w.stranded.load(.monotonic)) {
                // The watchdog has found this worker stuck and it will not take anything soon, so
                // its slot is the wrong place for this task: a slot has one taker, and that taker
                // is the worker itself, which is what `takeStuckSlot` exists to work around. The
                // shared queue is where every worker looks, and one of them is woken for this.
                s.shared.push(&.{task});
                s.notify(null);
                return;
            }
            switch (when) {
                .next => {
                    task.status = .{ .queue_next = null };
                    // The slot holds one task: whichever got there first. A task that finds it
                    // taken goes to the queue behind the one already there, and a stealer that
                    // takes the slot while this worker is stuck takes nothing this worker has
                    // not put there itself.
                    if (w.run_next.load(.monotonic) == null) {
                        // Release: whoever takes this task from the slot, the watchdog or this
                        // worker, sees everything this worker knows about it.
                        w.run_next.store(task, .release);
                        return;
                    }
                    w.local.push(task, &s.shared);
                },
                .later => w.local.push(task, &s.shared),
            }
            if (w.local.len() > steal_threshold) s.notify(w);
        }

        /// Whether `w` may switch to `task` directly.
        fn runsHere(w: *Worker, task: *Task) bool {
            return switch (task.affinity) {
                .free => true,
                .sticky, .pinned => @atomicLoad(u32, &task.home, .monotonic) == w.index,
            };
        }

        // Spawning

        /// Makes a task for `start` and its arguments, on worker `w`, the calling task's. `name`
        /// is the name of the function it will run, for the task names a log line reports: see
        /// `Io.spawnedName`.
        fn spawn(
            s: *Sched,
            w: *Worker,
            options: SpawnOptions,
            result_len: usize,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: @FieldType(Task, "start"),
        ) Io.ConcurrentError!*Task {
            const page = std.heap.pageSize();
            const result_space = @max(result_len, @sizeOf(Backend.Completion)) + result_alignment.toByteUnits();
            const header_size = @sizeOf(Task) + result_space + context.len + context_alignment.toByteUnits() + 64;
            const mapping = if (options.stack_size == null and header_size <= default_header_size)
                s.takeMapping(w) catch return error.ConcurrencyUnavailable
            else
                mapStack(std.mem.alignForward(usize, page + (options.stack_size orelse s.stack_size) + header_size, page)) catch
                    return error.ConcurrencyUnavailable;
            errdefer s.releaseMapping(w, mapping);

            const home: u32 = switch (options.affinity) {
                .pinned => |index| index: {
                    try s.ensureWorker(index);
                    break :index index;
                },
                .sticky, .free => w.index,
            };

            const end = @intFromPtr(mapping.ptr) + mapping.len;
            const task: *Task = @ptrFromInt(std.mem.alignBackward(usize, end - result_space - @sizeOf(Task), @alignOf(Task)));
            const context_bytes: [*]u8 = @ptrFromInt(context_alignment.backward(@intFromPtr(task) - context.len));
            const sp = std.mem.alignBackward(usize, @intFromPtr(context_bytes), 16);
            task.* = .{
                .context = switch (builtin.cpu.arch) {
                    .aarch64, .riscv64 => .{ .sp = sp, .fp = @intFromPtr(task), .pc = @intFromPtr(&taskEntry) },
                    .x86_64 => .{ .rsp = sp - 8, .rbp = @intFromPtr(task), .rip = @intFromPtr(&taskEntry) },
                    else => |arch| @compileError("unimplemented architecture: " ++ @tagName(arch)),
                },
                .link = switch (start) {
                    .main => unreachable,
                    .future => .{ .awaiter = null },
                    .group => .{ .group = .{ .prev = null, .next = null } },
                },
                .status = .{ .queue_next = null },
                .cancel_status = .unrequested,
                .cancel_protection = .unblocked,
                .affinity = options.affinity,
                .home = home,
                .start = start,
                .result_align = result_alignment,
                .ops = 0,
                .context_bytes = context_bytes,
                .mapping = mapping,
                .id = w.nextTaskId(),
                // The name has to be a comptime string: a task reports it long after the memory
                // of whoever spawned it is gone. `Io.spawnedName` is one.
                .name = if (name.len == 0) "unnamed" else name,
                .tsan_fiber = if (tsan.enable) tsan.__tsan_create_fiber(0),
            };
            if (builtin.cpu.arch == .x86_64) @as(*usize, @ptrFromInt(sp - 8)).* = 0; // no return address
            @memcpy(context_bytes[0..context.len], context);
            w.live += 1;
            // A task that runs, or one that is made, starts the watchdog: an instance with one
            // worker still reports a task that is stuck, and the replacement is the only way
            // anything else runs. An instance that never makes or runs a task pays nothing.
            if (!s.watchdog_started.load(.monotonic)) s.startWatchdog();
            return task;
        }

        /// Queues a task that was just spawned.
        fn enqueueSpawned(s: *Sched, w: *Worker, task: *Task) void {
            switch (task.affinity) {
                .sticky => s.readyHere(w, task, .next),
                .pinned, .free => s.ready(w, task),
            }
        }

        fn taskEntry() callconv(.naked) void {
            switch (builtin.cpu.arch) {
                .aarch64 => asm volatile (
                    \\ mov x0, fp
                    \\ mov fp, #0
                    \\ b %[taskMain]
                    :
                    : [taskMain] "X" (&taskMain),
                ),
                .riscv64 => asm volatile (
                    \\ mv a0, fp
                    \\ mv fp, zero
                    \\ tail %[taskMain]@plt
                    :
                    : [taskMain] "X" (&taskMain),
                ),
                .x86_64 => asm volatile (
                    \\ movq %%rbp, %%rdi
                    \\ xor %%ebp, %%ebp
                    \\ jmp %[taskMain:P]
                    :
                    : [taskMain] "X" (&taskMain),
                ),
                else => |arch| @compileError("unimplemented architecture: " ++ @tagName(arch)),
            }
        }

        fn taskMain(
            task: *Task,
            contexts: *const Io.fiber.Switch,
        ) callconv(.withStackAlign(.c, @alignOf(Task))) noreturn {
            const message: *const SwitchMessage = @fieldParentPtr("contexts", contexts);
            const s = Worker.current().sched;
            message.handle(s);
            switch (task.start) {
                .main => unreachable,
                .future => |start| {
                    start(task.context_bytes, task.resultBytes(task.result_align));
                    // An awaiter parked already runs next here, if it may. Otherwise it is made
                    // runnable once this task is saved, by `.finished`.
                    const next = if (@atomicLoad(?*Task, &task.link.awaiter, .acquire)) |awaiter|
                        (if (runsHere(.current(), awaiter)) awaiter else null)
                    else
                        null;
                    s.yield(next, .{ .finished = task });
                },
                .group => |member| {
                    member.start(task.context_bytes);
                    const w: *Worker = .current();
                    const next = if (member.group.removeTask(task)) |awaiter| next: {
                        if (runsHere(w, awaiter)) break :next awaiter;
                        s.ready(w, awaiter);
                        break :next null;
                    } else null;
                    s.yield(next, .destroy);
                },
            }
            unreachable; // a task that returned is not resumed
        }

        fn destroyTask(s: *Sched, w: *Worker, task: *Task) void {
            assert(task.status.queue_next == null);
            if (tsan.enable) tsan.__tsan_destroy_fiber(task.tsan_fiber);
            w.live -= 1;
            s.releaseMapping(w, task.mapping);
        }

        // Stacks

        /// One mapping per task, the first page unreadable and unwritable: an access to it
        /// faults instead of silently corrupting whatever is below, and the platform maps the
        /// stack pages above it as they are touched. Every OS has its own way of asking for that:
        /// Linux overcommits unless `NORESERVE` says not to and likes `STACK`, FreeBSD and
        /// DragonFly take `STACK` without `NORESERVE`, OpenBSD requires `STACK`, NetBSD takes it,
        /// and Darwin has no `STACK` and refuses `NORESERVE` on a mapping it may later grow.
        fn mapStack(len: usize) error{OutOfMemory}![]align(page_size_min) u8 {
            const mapping = posix.mmap(null, len, .{ .READ = true, .WRITE = true }, switch (builtin.os.tag) {
                .linux => .{
                    .TYPE = .PRIVATE,
                    .ANONYMOUS = true,
                    .NORESERVE = true,
                    .STACK = true,
                },
                .freebsd, .netbsd, .openbsd, .dragonfly => .{
                    .TYPE = .PRIVATE,
                    .ANONYMOUS = true,
                    .STACK = true,
                },
                .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => .{
                    .TYPE = .PRIVATE,
                    .ANONYMOUS = true,
                },
                else => |os| @compileError("stack mapping for " ++ @tagName(os) ++ " is not implemented"),
            }, -1, 0) catch return error.OutOfMemory;
            guard(mapping[0..std.heap.pageSize()]);
            return mapping;
        }

        var guard_install: std.atomic.Value(bool) = .init(builtin.os.tag == .linux);

        /// Makes an access to `page` fault. Linux 6.13 guards a page without splitting the
        /// mapping, which keeps one mapping per task; older kernels, and every other OS, need the
        /// page protected by hand.
        fn guard(page: []align(page_size_min) u8) void {
            if (builtin.os.tag == .linux) {
                if (guard_install.load(.monotonic)) {
                    switch (linux.errno(linux.madvise(page.ptr, page.len, linux.MADV.GUARD_INSTALL))) {
                        .SUCCESS => return,
                        else => guard_install.store(false, .monotonic),
                    }
                }
                _ = linux.mprotect(page.ptr, page.len, .{});
                return;
            }
            _ = std.c.mprotect(@ptrCast(@alignCast(page.ptr)), page.len, .{});
        }

        fn takeMapping(s: *Sched, w: *Worker) error{OutOfMemory}![]align(page_size_min) u8 {
            if (w.stacks.head) |free| {
                w.stacks.head = free.next;
                w.stacks.len -= 1;
                return s.mappingOf(free);
            }
            pooled: {
                s.pool.lock.lock();
                defer s.pool.lock.unlock();
                const free = s.pool.head orelse break :pooled;
                s.pool.head = free.next;
                s.pool.len -= 1;
                return s.mappingOf(free);
            }
            return mapStack(s.mapping_len);
        }

        fn releaseMapping(s: *Sched, w: *Worker, mapping: []align(page_size_min) u8) void {
            if (mapping.len != s.mapping_len) return posix.munmap(mapping);
            if (mapping.len > trim_threshold) {
                const page = std.heap.pageSize();
                // Darwin keeps a freed page until the memory is needed again, which is what
                // `FREE` asks for; the other kernels drop the pages on `DONTNEED`.
                const advice: u32 = switch (builtin.os.tag) {
                    .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => posix.MADV.FREE,
                    else => posix.MADV.DONTNEED,
                };
                posix.madvise(@alignCast(mapping.ptr + page), mapping.len - page - trim_keep, advice) catch {};
            }
            const free: *FreeMapping = @ptrFromInt(@intFromPtr(mapping.ptr) + mapping.len - @sizeOf(FreeMapping));
            if (w.stacks.len < stack_cache_max) {
                free.next = w.stacks.head;
                w.stacks.head = free;
                w.stacks.len += 1;
                return;
            }
            pool: {
                s.pool.lock.lock();
                defer s.pool.lock.unlock();
                if (s.pool.len >= s.workers.len * stack_pool_max_per_worker) break :pool;
                free.next = s.pool.head;
                s.pool.head = free;
                s.pool.len += 1;
                return;
            }
            posix.munmap(mapping);
        }

        fn mappingOf(s: *Sched, free: *FreeMapping) []align(page_size_min) u8 {
            const end = @intFromPtr(free) + @sizeOf(FreeMapping);
            const base: [*]align(page_size_min) u8 = @ptrFromInt(end - s.mapping_len);
            return base[0..s.mapping_len];
        }

        fn unmapFree(s: *Sched, free: *FreeMapping) void {
            posix.munmap(s.mappingOf(free));
        }

        // The Io vtable functions the scheduler implements.
        /// `Io.blocking` for a backend with no pool for blocking calls: the call is made on the
        /// calling thread. See `Io.blocking`.
        pub fn blockingDirect(
            userdata: ?*anyopaque,
            result: []u8,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) void {
            charged(userdata);
            _ = result_alignment;
            _ = context_alignment;
            _ = name;
            start(context.ptr, result.ptr);
        }

        /// `charge`, from a vtable entry point's `userdata`.
        pub inline fn charged(userdata: ?*anyopaque) void {
            charge(fromUserdata(userdata));
        }

        pub fn async(
            userdata: ?*anyopaque,
            result: []u8,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) ?*Io.AnyFuture {
            return concurrent(userdata, result.len, result_alignment, context, context_alignment, name, start) catch {
                start(context.ptr, result.ptr);
                return null;
            };
        }

        pub fn concurrent(
            userdata: ?*anyopaque,
            result_len: usize,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) Io.ConcurrentError!*Io.AnyFuture {
            const s = fromUserdata(userdata);
            const w = s.chargeFetch() orelse Worker.current();
            return s.spawnFuture(w, .{}, result_len, result_alignment, context, context_alignment, name, start);
        }

        fn spawnFuture(
            s: *Sched,
            w: *Worker,
            options: SpawnOptions,
            result_len: usize,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) Io.ConcurrentError!*Io.AnyFuture {
            const task = try s.spawn(w, options, result_len, result_alignment, context, context_alignment, name, .{ .future = start });
            s.enqueueSpawned(w, task);
            return @ptrCast(task);
        }

        pub fn await(
            userdata: ?*anyopaque,
            future: *Io.AnyFuture,
            result: []u8,
            result_alignment: Alignment,
        ) void {
            const s = fromUserdata(userdata);
            var w = s.chargeFetch() orelse Worker.current();
            const awaiting: *Task = @ptrCast(@alignCast(future));
            if (@atomicLoad(?*Task, &awaiting.link.awaiter, .acquire) != Task.finished) {
                s.yield(null, .{ .await = awaiting });
                // The task may resume on another worker.
                w = .current();
            }
            @memcpy(result, awaiting.resultBytes(result_alignment));
            s.destroyTask(w, awaiting);
        }

        pub fn cancel(
            userdata: ?*anyopaque,
            future: *Io.AnyFuture,
            result: []u8,
            result_alignment: Alignment,
        ) void {
            const s = fromUserdata(userdata);
            s.requestCancel(@ptrCast(@alignCast(future)));
            await(userdata, future, result, result_alignment);
        }

        pub fn requestCancel(s: *Sched, task: *Task) void {
            const cancel_status = @atomicRmw(
                Task.CancelStatus,
                &task.cancel_status,
                .Or,
                .{ .requested = true, .awaiting = @fromBackingInt(@intCast(0)) },
                .acquire,
            );
            assert(!cancel_status.requested);
            switch (cancel_status.awaiting) {
                .nothing => {},
                .group => {
                    // The awaiter received a cancelation request while awaiting a group, so
                    // propagate the cancelation to the group.
                    if (task.status.awaiting_group.cancel(s, null)) {
                        task.status = .{ .queue_next = null };
                        s.ready(.current(), task);
                    }
                },
                _ => |awaiting| Backend.cancelOperation(s.backendOf(), .current(), task, awaiting.toToken()),
            }
        }

        pub fn groupAsync(
            userdata: ?*anyopaque,
            type_erased: *Io.Group,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque) void,
        ) void {
            return groupConcurrent(userdata, type_erased, context, context_alignment, name, start) catch {
                start(context.ptr);
            };
        }

        pub fn groupConcurrent(
            userdata: ?*anyopaque,
            type_erased: *Io.Group,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque) void,
        ) Io.ConcurrentError!void {
            const s = fromUserdata(userdata);
            const w = s.chargeFetch() orelse Worker.current();
            return s.spawnGroupMember(w, .{}, type_erased, context, context_alignment, name, start);
        }

        fn spawnGroupMember(
            s: *Sched,
            w: *Worker,
            options: SpawnOptions,
            type_erased: *Io.Group,
            context: []const u8,
            context_alignment: Alignment,
            name: [:0]const u8,
            start: *const fn (context: *const anyopaque) void,
        ) Io.ConcurrentError!void {
            const group: Group = .{ .ptr = type_erased };
            const task = try s.spawn(w, options, 0, .@"1", context, context_alignment, name, .{ .group = .{
                .group = group,
                .start = start,
            } });
            group.addTask(task);
            s.enqueueSpawned(w, task);
        }

        pub fn groupAwait(userdata: ?*anyopaque, type_erased: *Io.Group, initial_token: *anyopaque) Io.Cancelable!void {
            const s = fromUserdata(userdata);
            _ = initial_token;
            charge(s);
            s.yield(null, .{ .group_await = .{ .ptr = type_erased } });
        }

        pub fn groupCancel(userdata: ?*anyopaque, type_erased: *Io.Group, initial_token: *anyopaque) void {
            const s = fromUserdata(userdata);
            _ = initial_token;
            charge(s);
            s.yield(null, .{ .group_cancel = .{ .ptr = type_erased } });
        }

        pub fn recancel(userdata: ?*anyopaque) void {
            _ = userdata;
            Worker.current().currentTask().cancel_protection.recancel();
        }

        pub fn swapCancelProtection(userdata: ?*anyopaque, new: Io.CancelProtection) Io.CancelProtection {
            _ = userdata;
            const cancel_protection = &Worker.current().currentTask().cancel_protection;
            defer cancel_protection.user = new;
            return cancel_protection.user;
        }

        pub fn checkCancel(userdata: ?*anyopaque) Io.Cancelable!void {
            const s = fromUserdata(userdata);
            const w = s.chargeFetch() orelse Worker.current();
            const task = w.currentTask();
            switch (task.cancel_protection.check()) {
                .unblocked => {
                    const cancel_status = @atomicLoad(Task.CancelStatus, &task.cancel_status, .monotonic);
                    assert(cancel_status.awaiting == .nothing);
                    if (cancel_status.requested) {
                        @branchHint(.unlikely);
                        task.cancel_protection.acknowledge();
                        return error.Canceled;
                    }
                },
                .blocked => {},
            }
        }

        pub fn crashHandler(userdata: ?*anyopaque) void {
            _ = userdata;
            const w = Worker.currentOrNull() orelse std.process.abort();
            if (w.current_context == &w.idle_context) std.process.abort();
            const task = w.currentTask();
            @atomicStore(
                Task.CancelStatus,
                &task.cancel_status,
                .{ .requested = true, .awaiting = .nothing },
                .monotonic,
            );
            task.cancel_protection = .{ .user = .blocked, .acknowledged = true };
        }

        // Spawning with options, beyond what the `Io` interface can say.

        pub fn concurrentWith(
            s: *Sched,
            options: SpawnOptions,
            function: anytype,
            args: std.meta.ArgsTuple(@TypeOf(function)),
        ) Io.ConcurrentError!Io.Future(@typeInfo(@TypeOf(function)).@"fn".return_type.?) {
            const Result = @typeInfo(@TypeOf(function)).@"fn".return_type.?;
            const Args = @TypeOf(args);
            const TypeErased = struct {
                fn start(context: *const anyopaque, result: *anyopaque) void {
                    const args_casted: *const Args = @ptrCast(@alignCast(context));
                    const result_casted: *Result = @ptrCast(@alignCast(result));
                    result_casted.* = @call(.auto, function, args_casted.*);
                }
            };
            var future: Io.Future(Result) = undefined;
            future.any_future = try s.spawnFuture(
                .current(),
                options,
                @sizeOf(Result),
                .of(Result),
                @ptrCast(&args),
                .of(Args),
                Io.spawnedName(function),
                TypeErased.start,
            );
            return future;
        }

        pub fn groupConcurrentWith(
            s: *Sched,
            group: *Io.Group,
            options: SpawnOptions,
            function: anytype,
            args: std.meta.ArgsTuple(@TypeOf(function)),
        ) Io.ConcurrentError!void {
            const Args = @TypeOf(args);
            const TypeErased = struct {
                fn start(context: *const anyopaque) void {
                    const args_casted: *const Args = @ptrCast(@alignCast(context));
                    _ = @as(Io.Cancelable!void, @call(.auto, function, args_casted.*)) catch {};
                }
            };
            return s.spawnGroupMember(.current(), options, group, @ptrCast(&args), .of(Args), Io.spawnedName(function), TypeErased.start);
        }

        pub const Group = struct {
            ptr: *Io.Group,

            const List = packed struct(usize) {
                cancel_requested: bool,
                awaiter_delayed: bool,
                tasks: Task.PackedPtr,
            };
            fn listPtr(group: Group) *List {
                return @ptrCast(&group.ptr.token);
            }

            const Mutex = packed struct(u32) {
                locked: bool,
                contended: bool,
                shared2: u30,
            };
            fn mutexPtr(group: Group) *Mutex {
                return switch (comptime builtin.cpu.arch.endian()) {
                    .little => @ptrCast(&group.ptr.state),
                    .big => @ptrCast(@alignCast(
                        @as([*]u8, @ptrCast(&group.ptr.state)) + @sizeOf(usize) - @sizeOf(u32),
                    )),
                };
            }

            const Awaiter = packed struct(usize) {
                locked: bool,
                contended: bool,
                awaiter: Task.PackedPtr,
            };
            fn awaiterPtr(group: Group) *Awaiter {
                return @ptrCast(&group.ptr.state);
            }

            /// Spins rather than parking: it is taken by pending tasks run after a switch, where
            /// there may be no task to park, and it is never held across a switch.
            fn lock(group: Group) void {
                const mutex = group.mutexPtr();
                var spins: u32 = 0;
                while (true) {
                    const old_state = @atomicRmw(
                        Mutex,
                        mutex,
                        .Or,
                        .{ .locked = true, .contended = false, .shared2 = 0 },
                        .acquire,
                    );
                    if (!old_state.locked) {
                        @branchHint(.likely);
                        return;
                    }
                    while (@atomicLoad(Mutex, mutex, .monotonic).locked) {
                        spins +%= 1;
                        if (spins < 64) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
                    }
                }
            }

            fn unlock(group: Group) void {
                const mutex = group.mutexPtr();
                const old_state = @atomicRmw(
                    Mutex,
                    mutex,
                    .And,
                    .{ .locked = false, .contended = false, .shared2 = std.math.maxInt(u30) },
                    .release,
                );
                assert(old_state.locked);
            }

            fn addTask(group: Group, task: *Task) void {
                group.lock();
                defer group.unlock();
                const list_ptr = group.listPtr();
                const list = @atomicLoad(List, list_ptr, .monotonic);
                if (list.cancel_requested) task.cancel_status = .{ .requested = true, .awaiting = .nothing };
                const old_head = list.tasks.unpack();
                if (old_head) |head| head.link.group.prev = task;
                task.link.group.next = old_head;
                @atomicStore(List, list_ptr, .{
                    .cancel_requested = list.cancel_requested,
                    .awaiter_delayed = list.awaiter_delayed,
                    .tasks = .pack(task),
                }, .monotonic);
            }

            /// Returns the awaiter to resume, if `task` was the last member.
            fn removeTask(group: Group, task: *Task) ?*Task {
                group.lock();
                defer group.unlock();
                const list_ptr = group.listPtr();
                const list = @atomicLoad(List, list_ptr, .monotonic);
                if (task.link.group.next) |next| next.link.group.prev = task.link.group.prev;
                if (task.link.group.prev) |prev| {
                    prev.link.group.next = task.link.group.next;
                } else if (task.link.group.next) |new_head| {
                    @atomicStore(List, list_ptr, .{
                        .cancel_requested = list.cancel_requested,
                        .awaiter_delayed = list.awaiter_delayed,
                        .tasks = .pack(new_head),
                    }, .monotonic);
                } else if (@atomicLoad(Awaiter, group.awaiterPtr(), .monotonic).awaiter.unpack()) |awaiter| {
                    if (!awaiter.cancel_status.changeAwaiting(.group, .nothing) or list.cancel_requested) {
                        @atomicStore(List, list_ptr, .{
                            .cancel_requested = false,
                            .awaiter_delayed = false,
                            .tasks = .null,
                        }, .release);
                        assert(awaiter.status.awaiting_group.ptr == group.ptr);
                        awaiter.status = .{ .queue_next = null };
                        return awaiter;
                    }
                    // Race with `requestCancel`
                    @atomicStore(List, list_ptr, .{
                        .cancel_requested = false,
                        .awaiter_delayed = true,
                        .tasks = .null,
                    }, .monotonic);
                } else @atomicStore(List, list_ptr, .{
                    .cancel_requested = false,
                    .awaiter_delayed = false,
                    .tasks = .null,
                }, .release);
                return null;
            }

            /// Returns whether `awaiter` may continue at once.
            fn await(group: Group, s: *Sched, awaiter: *Task) bool {
                group.lock();
                defer group.unlock();
                if (@atomicLoad(List, group.listPtr(), .monotonic).tasks.unpack()) |_| {
                    if (group.registerAwaiter(awaiter) and awaiter.cancel_protection.check() == .unblocked) {
                        // The awaiter already had an unacknowledged cancelation request before
                        // attempting to await a group, so propagate the cancelation to the group.
                        assert(!group.cancelLocked(s, null));
                    }
                    return false;
                }
                return true;
            }

            fn cancel(group: Group, s: *Sched, maybe_awaiter: ?*Task) bool {
                group.lock();
                defer group.unlock();
                return group.cancelLocked(s, maybe_awaiter);
            }

            /// Assumes the mutex is held.
            fn cancelLocked(group: Group, s: *Sched, maybe_awaiter: ?*Task) bool {
                const list_ptr = group.listPtr();
                const list = @atomicRmw(
                    List,
                    list_ptr,
                    .Add,
                    .{ .cancel_requested = true, .awaiter_delayed = false, .tasks = .null },
                    .monotonic,
                );
                assert(!list.cancel_requested);
                if (list.tasks.unpack()) |head| {
                    var maybe_task: ?*Task = head;
                    while (maybe_task) |task| {
                        s.requestCancel(task);
                        maybe_task = task.link.group.next;
                    }
                    if (maybe_awaiter) |awaiter| _ = group.registerAwaiter(awaiter);
                    return false;
                }
                @atomicStore(
                    List,
                    list_ptr,
                    .{ .cancel_requested = false, .awaiter_delayed = false, .tasks = .null },
                    .release,
                );
                return if (maybe_awaiter) |_| true else list.awaiter_delayed;
            }

            /// Assumes the mutex is held.
            fn registerAwaiter(group: Group, awaiter: *Task) bool {
                assert(awaiter.status.queue_next == null);
                awaiter.status = .{ .awaiting_group = group };
                assert(@atomicRmw(
                    Awaiter,
                    group.awaiterPtr(),
                    .Add,
                    .{ .locked = false, .contended = false, .awaiter = .pack(awaiter) },
                    .monotonic,
                ).awaiter == .null);
                return awaiter.cancel_status.changeAwaiting(.nothing, .group);
            }
        };
    };
}

/// Whether this process's Darwin is one whose `__ulock_wait2` exists: the XNU in macOS 11 and
/// later has it, and the older `__ulock_wait` takes microseconds instead of nanoseconds. The
/// same test `Io.Threaded` makes.
const darwin_has_ulock_wait2 = builtin.os.version_range.semver.min.major >= 11;

/// How much CPU time another thread of this process has used, where the platform can say:
/// Darwin through Mach's `thread_info`, which counts the thread's user and system time, and Linux
/// through the clock named after a thread id. The BSDs' per-thread clocks can only be read by the
/// thread itself, so there a worker has no CPU time to report and is always taken to be blocked.
/// See the backend contract's `threadCpuTime`.
pub const CpuTime = struct {
    /// What a thread records about itself when it starts, for `read`. A plain struct rather than a
    /// union so that a core and the test backend may hold it on any platform; the field that does
    /// not apply stays zero.
    pub const Source = struct {
        /// The clock that names this thread, on Linux.
        clock: u32 = 0,
        /// The Mach thread port, on Darwin. It is a send right this process owns, and `release`
        /// gives it back.
        mach: u32 = 0,
    };

    /// Records what this thread has to hand out. Called on the thread itself.
    pub fn acquire() Source {
        return switch (builtin.os.tag) {
            .linux => .{ .clock = @bitCast((~linux.gettid() << 3) | 6) }, // the clock named by a thread id
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => .{ .mach = mach.mach_thread_self() },
            else => .{},
        };
    }

    /// Gives back what `acquire` took.
    pub fn release(source: Source) void {
        switch (builtin.os.tag) {
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
                if (source.mach != 0) _ = std.c.mach_port_deallocate(std.c.mach_task_self(), source.mach);
            },
            else => {},
        }
    }

    /// This thread's CPU time in nanoseconds, or `null` when it cannot be read.
    pub fn read(source: Source) ?u64 {
        switch (builtin.os.tag) {
            .linux => {
                if (source.clock == 0) return null;
                const clock_id: posix.clockid_t = @bitCast(source.clock);
                var tp: posix.timespec = undefined;
                switch (posix.errno(posix.system.clock_gettime(clock_id, &tp))) {
                    .SUCCESS => {},
                    else => return null,
                }
                return @as(u64, @intCast(@max(0, tp.sec))) * std.time.ns_per_s +
                    @as(u64, @intCast(@max(0, tp.nsec)));
            },
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
                if (source.mach == 0) return null;
                var info: std.c.thread_basic_info = std.mem.zeroes(std.c.thread_basic_info);
                var count: std.c.mach_msg_type_number_t = std.c.THREAD.BASIC.INFO_COUNT;
                if (std.c.thread_info(source.mach, std.c.THREAD.BASIC.INFO, @ptrCast(&info), &count) != 0) return null;
                if (count != std.c.THREAD.BASIC.INFO_COUNT) return null;
                const seconds = @as(i64, info.user_time.seconds) + @as(i64, info.system_time.seconds);
                const microseconds = @as(i64, info.user_time.microseconds) + @as(i64, info.system_time.microseconds);
                const total = seconds * std.time.ns_per_s + microseconds * std.time.ns_per_us;
                return if (total < 0) 0 else @intCast(total);
            },
            else => return null,
        }
    }

    /// Mach's own thread port, which the standard library does not bind.
    const mach = struct {
        extern "c" fn mach_thread_self() std.c.thread_t;
    };
};

/// A futex for a thread that is not one of the workers. A parked task is never woken through
/// this: the scheduler parks it and its backend wakes it. This is what a foreign thread blocks
/// in, which is the watchdog and any thread that called one of the instance's `Io` futex
/// functions, and what a backend that has no kernel object to wait in parks its own wake word
/// in. A wake has to reach the kernel's waiters and, in such a backend, its own, which is why
/// `wake` is not just the syscall: a backend's own waiters are woken through the scheduler.
///
/// Every platform has its own private operation for one address, and NetBSD's `futex(2)` is not
/// used by the standard library, which parks threads instead: see `netbsdWaitTable`.
pub const Futex = struct {
    /// Blocks the calling thread while `ptr.* == expected`, and returns when it is not, when a
    /// wake unparks it, or when `timeout_ns` nanoseconds have passed. A return says nothing about
    /// the word: the caller re-reads it, so a spurious one costs nothing.
    pub fn wait(ptr: *const u32, expected: u32, timeout_ns: ?u64) void {
        switch (builtin.os.tag) {
            .linux => {
                var ts_buffer: linux.timespec = undefined;
                const ts: ?*const linux.timespec = if (timeout_ns) |ns| ts: {
                    ts_buffer = .{
                        .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                        .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                    };
                    break :ts &ts_buffer;
                } else null;
                _ = linux.futex_4arg(ptr, .{ .cmd = .WAIT, .private = true }, expected, ts);
            },
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
                const c = std.c;
                // Zero means no timeout in either call. Darwin has two ulock waits: the older one
                // takes microseconds, and `__ulock_wait2`, XNU 7195.50.7.100.1 and later — macOS
                // 11 — takes nanoseconds, which is what `Io.Threaded` uses where it can.
                const flags: c.UL = .{ .op = .COMPARE_AND_WAIT, .NO_ERRNO = true };
                const status = if (darwin_has_ulock_wait2) c.__ulock_wait2(
                    flags,
                    ptr,
                    expected,
                    timeout_ns orelse 0,
                    0,
                ) else c.__ulock_wait(flags, ptr, expected, if (timeout_ns) |ns| @intCast(
                    std.math.clamp(@divFloor(ns, std.time.ns_per_us), 1, std.math.maxInt(u32)),
                ) else 0);
                if (status >= 0) return;
                switch (@as(c.E, @fromBackingInt(@intCast(-status)))) {
                    .INTR, .TIMEDOUT, .CANCELED, .AGAIN, .NOENT => return, // woken, or the wait is over
                    // The address was paged out, which darwin's own pthread code survives by
                    // returning: the caller re-reads the word and waits again.
                    .FAULT => return,
                    else => {
                        recoverableOsBugDetected();
                        return;
                    },
                }
            },
            .freebsd => {
                const c = std.c;
                var time_buffer: c._umtx_time = undefined;
                var size: usize = 0;
                var time: ?*const c._umtx_time = null;
                if (timeout_ns) |ns| {
                    time_buffer = .{
                        .timeout = .{
                            .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                            .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                        },
                        .flags = 0, // a duration, not an absolute time
                        .clockid = .MONOTONIC,
                    };
                    size = @sizeOf(c._umtx_time);
                    time = &time_buffer;
                }
                _ = c._umtx_op(
                    @intFromPtr(ptr),
                    @backingInt(c.UMTX_OP.WAIT_UINT_PRIVATE),
                    @as(c_ulong, expected),
                    size,
                    @intFromPtr(time),
                );
            },
            .openbsd => {
                const c = std.c;
                var ts_buffer: posix.timespec = undefined;
                const ts: ?*const posix.timespec = if (timeout_ns) |ns| ts: {
                    ts_buffer = .{
                        .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                        .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
                    };
                    break :ts &ts_buffer;
                } else null;
                _ = c.futex(
                    ptr,
                    c.FUTEX.WAIT | c.FUTEX.PRIVATE_FLAG,
                    @as(c_int, @bitCast(expected)),
                    ts,
                    null,
                );
            },
            .dragonfly => {
                const us: c_int = if (timeout_ns) |ns|
                    std.math.cast(c_int, ns / std.time.ns_per_us) orelse std.math.maxInt(c_int)
                else
                    0;
                _ = std.c.umtx_sleep(@ptrCast(ptr), @bitCast(expected), us);
            },
            .netbsd => netbsdWaitTable.wait(ptr, expected, timeout_ns),
            else => |os| @compileError("a futex for " ++ @tagName(os) ++ " is not implemented"),
        }
    }

    /// Wakes up to `max_waiters` threads blocked in `wait` on `ptr`. A wake with nobody waiting
    /// is a no-op, not an error.
    pub fn wake(ptr: *const u32, max_waiters: u32) void {
        if (max_waiters == 0) return;
        const n: c_int = @intCast(@min(max_waiters, std.math.maxInt(c_int)));
        switch (builtin.os.tag) {
            .linux => _ = linux.futex_3arg(
                ptr,
                .{ .cmd = .WAKE, .private = true },
                @intCast(@min(max_waiters, std.math.maxInt(i32))),
            ),
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
                const c = std.c;
                const flags: c.UL = .{
                    .op = .COMPARE_AND_WAIT,
                    .NO_ERRNO = true,
                    .WAKE_ALL = max_waiters > 1,
                };
                while (true) {
                    const status = c.__ulock_wake(flags, ptr, 0);
                    if (status >= 0) return;
                    switch (@as(c.E, @fromBackingInt(@intCast(-status)))) {
                        .INTR, .CANCELED => continue, // spurious
                        else => return,
                    }
                }
            },
            .freebsd => _ = std.c._umtx_op(
                @intFromPtr(ptr),
                @backingInt(std.c.UMTX_OP.WAKE_PRIVATE),
                @as(c_ulong, @intCast(n)),
                0,
                0,
            ),
            .openbsd => _ = std.c.futex(
                ptr,
                std.c.FUTEX.WAKE | std.c.FUTEX.PRIVATE_FLAG,
                @intCast(n),
                null,
                null,
            ),
            .dragonfly => _ = std.c.umtx_wakeup(@ptrCast(ptr), n),
            .netbsd => netbsdWaitTable.wake(ptr, max_waiters),
            else => |os| @compileError("a futex for " ++ @tagName(os) ++ " is not implemented"),
        }
    }
};

/// NetBSD parks threads rather than offering them a futex the standard library trusts, so a wait
/// is a node in a hashed table of waiters and a `_lwp_park`, and a wake removes the nodes for its
/// address from the table and unparks the threads, which is what `Io.Threaded` does for the same
/// reason. The flag a waiter carries is what makes a wake that arrives between the waiter's check
/// of the word and its park harmless: the waker sets it before unparking, and the waiter checks
/// it before every park, so the park it is about to make returns at once.
const netbsdWaitTable = struct {
    const Bucket = struct {
        lock: SpinLock = .{},
        /// The waiters for this bucket's addresses, newest first.
        head: ?*Waiter = null,

        const SpinLock = struct {
            locked: std.atomic.Value(bool) = .init(false),
            fn lock(l: *SpinLock) void {
                while (l.locked.swap(true, .acquire)) {
                    while (l.locked.load(.monotonic)) std.atomic.spinLoopHint();
                }
            }
            fn unlock(l: *SpinLock) void {
                l.locked.store(false, .release);
            }
        };
    };

    const Waiter = struct {
        next: ?*Waiter,
        address: usize,
        lwp: c_int,
        /// Set by a waker, once. The waiter owns the flip back to false.
        woken: std.atomic.Value(bool) = .init(false),
    };

    var buckets: [64]Bucket = @splat(.{});

    fn bucketFor(address: usize) *Bucket {
        // Fibonacci hashing: the high bits of the golden-ratio multiple spread addresses better
        // than the low ones, which are the ones a slab allocator varies least.
        const fibonacci_multiplier = 0x9E3779B97F4A7C15 >> (64 - @bitSizeOf(usize));
        const hashed = address *% fibonacci_multiplier;
        return &buckets[hashed >> (@bitSizeOf(usize) - @ctz(buckets.len))];
    }

    fn wait(ptr: *const u32, expected: u32, timeout_ns: ?u64) void {
        const bucket = bucketFor(@intFromPtr(ptr));
        var waiter: Waiter = .{
            .next = null,
            .address = @intFromPtr(ptr),
            .lwp = @bitCast(std.c._lwp_self()),
        };
        var ts_buffer: posix.timespec = undefined;
        const ts: ?*posix.timespec = if (timeout_ns) |ns| ts: {
            ts_buffer = .{
                .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
                .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
            };
            break :ts &ts_buffer;
        } else null;
        {
            bucket.lock.lock();
            defer bucket.lock.unlock();
            if (@atomicLoad(u32, ptr, .monotonic) != expected) return;
            waiter.next = bucket.head;
            bucket.head = &waiter;
        }
        while (!waiter.woken.swap(false, .acquire)) {
            switch (posix.errno(std.c._lwp_park(std.c.CLOCK.MONOTONIC, .{}, ts, 0, ptr, null))) {
                .SUCCESS, .ALREADY, .INTR => {},
                .TIMEDOUT => {
                    // A wake may have raced the timeout: if it took the node, it is about to
                    // unpark this thread, and the loop has to let it.
                    bucket.lock.lock();
                    defer bucket.lock.unlock();
                    var link = &bucket.head;
                    while (link.*) |other| {
                        if (other != &waiter) {
                            link = &other.next;
                            continue;
                        }
                        link.* = other.next;
                        return;
                    }
                },
                else => |err| std.debug.panic("_lwp_park: {t}", .{err}),
            }
        }
        // A waker took the node out of the table.
    }

    fn wake(ptr: *const u32, max_waiters: u32) void {
        const bucket = bucketFor(@intFromPtr(ptr));
        var unpark: [16]c_int = undefined;
        var len: usize = 0;
        bucket.lock.lock();
        var link = &bucket.head;
        var woken: u32 = 0;
        while (link.*) |waiter| {
            if (waiter.address != @intFromPtr(ptr) or woken == max_waiters) {
                link = &waiter.next;
                continue;
            }
            link.* = waiter.next;
            waiter.woken.store(true, .release);
            unpark[len] = waiter.lwp;
            len += 1;
            woken += 1;
            if (len == unpark.len) {
                // A waiter may be unparked once it is out of the table and flagged, so its
                // stack is the waker's to leave only after that: unpark under the lock.
                _ = std.c._lwp_unpark_all(@ptrCast(&unpark), len, ptr);
                len = 0;
            }
        }
        bucket.lock.unlock();
        if (len > 0) _ = std.c._lwp_unpark_all(@ptrCast(&unpark), len, ptr);
    }
};

/// A backend with no operations, for testing the scheduler alone. Workers block on a futex word,
/// and `Io` futexes park tasks in a list under one lock.
const TestBackend = struct {
    sched: Sched,
    gpa: Allocator,
    /// Whether this backend issues the barrier the watchdog takes a stuck worker's slot with.
    /// Tests turn it off to exercise what the scheduler does without one.
    heavy_barrier: std.atomic.Value(bool) = .init(true),
    futex_lock: Sched.SpinLock = .{},
    futex_waiters: ?*FutexWaiter = null,

    const Sched = Scheduler(TestBackend);

    pub const Completion = struct { result: i32 };
    pub const Worker = struct {
        wake_word: std.atomic.Value(u32),
        /// What reads this worker's thread CPU time, recorded on the worker's own thread, for
        /// `threadCpuTime`. Nothing where this platform cannot say, which leaves every stuck
        /// worker counted as blocked. See the backend contract.
        cpu: CpuTime.Source = .{},
    };

    const FutexWaiter = struct {
        ptr: *const u32,
        task: *Sched.Task,
        next: ?*FutexWaiter,
    };

    fn init(b: *TestBackend, gpa: Allocator, workers: usize) !void {
        b.* = .{ .sched = undefined, .gpa = gpa };
        // The commands of `membarrier` have to be registered for before they may be used. A
        // process that cannot register keeps the compare-exchange path, which these tests then
        // exercise.
        if (builtin.os.tag == .linux) {
            _ = linux.membarrier(linux.MEMBARRIER.REGISTER_PRIVATE_EXPEDITED, 0, 0);
        }
        try b.sched.init(gpa, .{ .workers = workers });
    }

    /// For the scheduler: a barrier that makes this thread's earlier stores visible to every
    /// other thread of the process after their own barriers, or `false` when this process cannot
    /// issue one. On Linux this is `membarrier`; no other platform has its equivalent, so a
    /// backend there says it has none, and every taker of a slot pays a compare-exchange.
    pub fn heavyBarrier(b: *TestBackend) bool {
        if (builtin.os.tag != .linux) return false;
        if (!b.heavy_barrier.load(.monotonic)) return false;
        return linux.errno(linux.membarrier(linux.MEMBARRIER.PRIVATE_EXPEDITED, 0, 0)) == .SUCCESS;
    }

    fn deinit(b: *TestBackend) void {
        b.sched.deinit(b.gpa);
    }

    fn io(b: *TestBackend) Io {
        return .{ .userdata = b, .vtable = &vtable };
    }

    const vtable: Io.VTable = v: {
        var v = Io.failing.vtable.*;
        v.async = Sched.async;
        v.concurrent = Sched.concurrent;
        v.await = Sched.await;
        v.cancel = Sched.cancel;
        v.groupAsync = Sched.groupAsync;
        v.groupConcurrent = Sched.groupConcurrent;
        v.groupAwait = Sched.groupAwait;
        v.groupCancel = Sched.groupCancel;
        v.recancel = Sched.recancel;
        v.swapCancelProtection = Sched.swapCancelProtection;
        v.checkCancel = Sched.checkCancel;
        v.blocking = Sched.blockingDirect;
        v.now = now;
        v.futexWait = futexWait;
        v.futexWaitUncancelable = futexWaitUncancelable;
        v.futexWake = futexWake;
        break :v v;
    };

    pub fn workerInit(b: *TestBackend, w: *Sched.Worker) !void {
        _ = b;
        w.backend = .{ .wake_word = .init(0) };
    }

    pub fn workerStart(b: *TestBackend, w: *Sched.Worker) void {
        _ = b;
        w.backend.cpu = CpuTime.acquire();
    }

    /// How much CPU time the worker's thread has used: see the backend contract.
    pub fn threadCpuTime(b: *TestBackend, w: *Sched.Worker) ?u64 {
        _ = b;
        return CpuTime.read(w.backend.cpu);
    }

    pub fn workerDeinit(b: *TestBackend, w: *Sched.Worker) void {
        _ = b;
        CpuTime.release(w.backend.cpu);
    }

    pub fn poll(b: *TestBackend, w: *Sched.Worker, mode: PollMode) void {
        _ = b;
        switch (mode) {
            .nonblocking => _ = w.backend.wake_word.swap(0, .acquire),
            .block => while (w.backend.wake_word.swap(0, .acquire) == 0) {
                Futex.wait(&w.backend.wake_word.raw, 0, null);
            },
        }
    }

    pub fn wake(b: *TestBackend, from: *Sched.Worker, to: *Sched.Worker) void {
        _ = b;
        _ = from;
        to.backend.wake_word.store(1, .release);
        Futex.wake(&to.backend.wake_word.raw, 1);
    }

    /// The scheduler's watchdog reads the clock to tell how long a task has held its worker.
    pub fn now(userdata: ?*anyopaque, clock: Io.Clock) Io.Timestamp {
        _ = userdata;
        var tp: posix.timespec = undefined;
        switch (posix.errno(posix.system.clock_gettime(Io.Threaded.clockToPosix(clock), &tp))) {
            .SUCCESS => {},
            else => return .zero,
        }
        return Io.Threaded.timestampFromPosix(&tp);
    }

    pub fn wakeForeign(b: *TestBackend, to: *Sched.Worker) void {
        _ = b;
        to.backend.wake_word.store(1, .release);
        Futex.wake(&to.backend.wake_word.raw, 1);
    }

    pub fn cancelOperation(b: *TestBackend, from: *Sched.Worker, task: *Sched.Task, token: u31) void {
        _ = b;
        _ = from;
        _ = task;
        _ = token;
        unreachable; // there are no operations to cancel
    }

    pub fn allocator(b: *TestBackend) Allocator {
        return b.gpa;
    }

    fn futexWait(userdata: ?*anyopaque, ptr: *const u32, expected: u32, timeout: Io.Timeout) Io.Cancelable!void {
        const b: *TestBackend = @ptrCast(@alignCast(userdata));
        if (Sched.Worker.currentOrNull() == null) {
            // A thread that is not one of the workers blocks in the kernel, as the real backends
            // do. The watchdog is one, and it waits with a timeout.
            const timeout_ns: ?u64 = if (timeout.toDurationFromNow(b.io())) |duration|
                @intCast(@max(0, duration.raw.toNanoseconds()))
            else
                null;
            Futex.wait(ptr, expected, timeout_ns);
            return;
        }
        assert(timeout == .none);
        try Sched.checkCancel(userdata);
        futexWaitUncancelable(userdata, ptr, expected);
    }

    pub fn futexWaitUncancelable(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
        const b: *TestBackend = @ptrCast(@alignCast(userdata));
        b.futex_lock.lock();
        if (@atomicLoad(u32, ptr, .monotonic) != expected) return b.futex_lock.unlock();
        var waiter: FutexWaiter = .{
            .ptr = ptr,
            .task = Sched.Worker.current().currentTask(),
            .next = b.futex_waiters,
        };
        b.futex_waiters = &waiter;
        // Unlocked once this task is saved, so a wake cannot make it runnable before that.
        b.sched.yield(null, .{ .custom = .{ .context = b, .run = unlockFutexes } });
    }

    fn unlockFutexes(s: *Sched, task: *Sched.Task, context: *anyopaque) void {
        _ = s;
        _ = task;
        const b: *TestBackend = @ptrCast(@alignCast(context));
        b.futex_lock.unlock();
    }

    pub fn futexWake(userdata: ?*anyopaque, ptr: *const u32, max_waiters: u32) void {
        const b: *TestBackend = @ptrCast(@alignCast(userdata));
        var woken: ?*FutexWaiter = null;
        {
            b.futex_lock.lock();
            defer b.futex_lock.unlock();
            var n: u32 = 0;
            var link = &b.futex_waiters;
            while (link.*) |waiter| {
                if (n == max_waiters) break;
                if (waiter.ptr != ptr) {
                    link = &waiter.next;
                    continue;
                }
                link.* = waiter.next;
                waiter.next = woken;
                woken = waiter;
                n += 1;
            }
        }
        // A wake of the kernel for the waiters this backend does not park: a thread that is not
        // one of the workers, such as the watchdog, waits on the futex itself.
        Futex.wake(ptr, max_waiters);
        // And the tasks this backend parks, which no wake of the kernel reaches: a caller that is
        // not one of the workers makes them runnable through the scheduler, since it has no
        // worker to wake them from.
        const from = Sched.Worker.currentOrNull();
        while (woken) |waiter| {
            woken = waiter.next;
            if (from) |w| b.sched.ready(w, waiter.task) else b.sched.readyFromForeign(waiter.task);
        }
    }
};

fn testBackend(workers: usize) !*TestBackend {
    if (!Io.fiber.supported or builtin.single_threaded) return error.SkipZigTest;
    // The OSes with a stack mapping, a futex for a thread that is not a worker, and a way to
    // read a thread's CPU time: see `mapStack`, `Futex` and `threadCpuTime`.
    switch (builtin.os.tag) {
        .linux,
        .driverkit,
        .ios,
        .maccatalyst,
        .macos,
        .tvos,
        .visionos,
        .watchos,
        .freebsd,
        .netbsd,
        .openbsd,
        .dragonfly,
        => {},
        else => return error.SkipZigTest,
    }
    const b = try std.testing.allocator.create(TestBackend);
    errdefer std.testing.allocator.destroy(b);
    try b.init(std.testing.allocator, workers);
    return b;
}

fn destroyTestBackend(b: *TestBackend) void {
    b.deinit();
    std.testing.allocator.destroy(b);
}

test "futures and groups across workers" {
    const b = try testBackend(8);
    defer destroyTestBackend(b);
    const io = b.io();

    const S = struct {
        fn fib(inner_io: Io, n: u32) u64 {
            if (n < 2) return n;
            var a = inner_io.async(fib, .{ inner_io, n - 1 });
            const c = fib(inner_io, n - 2);
            return a.await(inner_io) + c;
        }

        fn add(total: *std.atomic.Value(u64), n: u64) void {
            _ = total.fetchAdd(n, .monotonic);
        }
    };
    try std.testing.expectEqual(6765, S.fib(io, 20));

    var total: std.atomic.Value(u64) = .init(0);
    var group: Io.Group = .init;
    for (0..10_000) |i| group.async(io, S.add, .{ &total, i });
    try group.await(io);
    try std.testing.expectEqual(10_000 * 9_999 / 2, total.load(.monotonic));
}

test "pinned tasks never leave their worker" {
    const b = try testBackend(4);
    defer destroyTestBackend(b);
    const io = b.io();

    const S = struct {
        fn run(inner_io: Io, index: u32) anyerror!void {
            for (0..100) |_| {
                try std.testing.expectEqual(index, TestBackend.Sched.Worker.current().index);
                // Park, so that some other task runs meanwhile.
                var group: Io.Group = .init;
                group.async(inner_io, nop, .{});
                try group.await(inner_io);
            }
        }

        fn nop() void {}
    };
    var futures: [4]Io.Future(anyerror!void) = undefined;
    for (&futures, 0..) |*future, i| {
        future.* = try b.sched.concurrentWith(.{ .affinity = .{ .pinned = @intCast(i) } }, S.run, .{ io, @as(u32, @intCast(i)) });
    }
    for (&futures) |*future| try future.await(io);
    try std.testing.expect(b.sched.started.load(.monotonic) == 4);
}

test "arguments bigger than the default task header" {
    const b = try testBackend(2);
    defer destroyTestBackend(b);
    const io = b.io();

    const Big = [3000]u64;
    const S = struct {
        fn sum(big: Big) u64 {
            var total: u64 = 0;
            for (big) |x| total += x;
            return total;
        }
    };
    var big: Big = undefined;
    for (&big, 0..) |*x, i| x.* = i;
    var future = try io.concurrent(S.sum, .{big});
    try std.testing.expectEqual(3000 * 2999 / 2, future.await(io));
}

test "worker limit" {
    const b = try testBackend(8);
    defer destroyTestBackend(b);
    const io = b.io();
    b.sched.setWorkerLimit(2);

    const S = struct {
        fn spin(n: u64, sink: *std.atomic.Value(u64)) void {
            var x: u64 = 0;
            for (0..n) |i| x +%= i *% i;
            _ = sink.fetchAdd(x, .monotonic);
        }
    };
    var sink: std.atomic.Value(u64) = .init(0);
    var group: Io.Group = .init;
    for (0..1000) |_| group.async(io, S.spin, .{ 10_000, &sink });
    try group.await(io);
    try std.testing.expect(b.sched.started.load(.monotonic) <= 2);
}

/// Nanoseconds on the monotonic clock, without an Io call, for tests that must keep a worker.
fn testNow() u64 {
    var tp: posix.timespec = undefined;
    assert(posix.errno(posix.system.clock_gettime(posix.CLOCK.MONOTONIC, &tp)) == .SUCCESS);
    return @as(u64, @intCast(tp.sec)) * std.time.ns_per_s + @as(u64, @intCast(tp.nsec));
}

/// Sleeps this thread for at least `ns`. The tests below wait for the state they check rather
/// than for a fixed time, so this is only the step between two looks at it.
fn testSleep(ns: u64) void {
    var ts: posix.timespec = .{
        .sec = @intCast(@divFloor(ns, std.time.ns_per_s)),
        .nsec = @intCast(@mod(ns, std.time.ns_per_s)),
    };
    while (true) switch (posix.errno(posix.system.nanosleep(&ts, &ts))) {
        .SUCCESS => return,
        .INTR => continue,
        else => return,
    };
}

test "watchdog: with one worker, what a blocked task queued still runs" {
    const b = try testBackend(4);
    defer destroyTestBackend(b);
    b.sched.setWorkerLimit(1);
    const io = b.io();

    const S = struct {
        /// Queues one task behind itself, then blocks the only worker in a raw futex wait
        /// until that task has run, for ten seconds at most: with one worker, a computing task
        /// would hold it just as well, but a blocked one is what the replacement worker exists
        /// for, and the replacement is the only way that task runs. Waiting for the task rather
        /// than for a fixed time is what keeps the test true on a loaded machine.
        fn blocker(inner_io: Io, done: *std.atomic.Value(bool), word: *std.atomic.Value(u32), blocked: *std.atomic.Value(bool), release: *std.atomic.Value(u32), loop_ns: *std.atomic.Value(u64)) void {
            var future = inner_io.concurrent(short, .{ inner_io, done, word, blocked, release }) catch return;
            blocked.store(true, .release);
            // How long the loop really took, not how many steps it counted: a wait that returns
            // at once burns the ten seconds in microseconds, which is what a failure has to tell
            // apart from a wait that really waited.
            const started = testNow();
            var waited_ms: u32 = 0;
            while (release.load(.acquire) == 0 and waited_ms < 10_000) : (waited_ms += 100) {
                Futex.wait(&release.raw, 0, 100 * std.time.ns_per_ms);
            }
            loop_ns.store(testNow() -% started, .release);
            blocked.store(false, .release);
            future.await(inner_io);
        }

        /// Records whether the task that holds the only worker was still blocked on it, then
        /// releases it and wakes the main task.
        fn short(inner_io: Io, done: *std.atomic.Value(bool), word: *std.atomic.Value(u32), blocked: *std.atomic.Value(bool), release: *std.atomic.Value(u32)) void {
            done.store(blocked.load(.acquire), .release);
            release.store(1, .release);
            Futex.wake(&release.raw, 1);
            word.store(1, .release);
            inner_io.futexWake(u32, &word.raw, 1);
        }
    };

    var done: std.atomic.Value(bool) = .init(false);
    var word: std.atomic.Value(u32) = .init(0);
    var blocked: std.atomic.Value(bool) = .init(false);
    var release: std.atomic.Value(u32) = .init(0);
    var loop_ns: std.atomic.Value(u64) = .init(0);
    var future = try b.sched.concurrentWith(.{}, S.blocker, .{ io, &done, &word, &blocked, &release, &loop_ns });
    // The main task is the only worker's task: it parks, and the task the blocker queued wakes
    // it when it has run.
    io.futexWaitUncancelable(u32, &word.raw, 0);
    // Awaited before the checks, so that a failed check fails the test instead of leaving a task
    // that `deinit` finds never awaited.
    future.await(io);
    // The episode is counted by the watchdog, after the replacement started: wait for it. This
    // comes before the checks, because what the watchdog saw is what says whether a queued task
    // that did not run was waiting on a worker counted as blocked or as computing.
    var stats = b.sched.stats();
    var waited_ms: u32 = 0;
    while (stats.stuck_episodes == 0 and waited_ms < 10_000) : (waited_ms += 1) {
        testSleep(std.time.ns_per_ms);
        stats = b.sched.stats();
    }
    if (!done.load(.acquire)) {
        // The queued task did not run while the blocker held the only worker. What the watchdog
        // made of that worker, and what its thread's CPU time reads as now, is the difference
        // between a worker it never replaced and a replacement that did not take the task.
        const stuck = stats.last_stuck;
        std.debug.print(
            "one-worker watchdog: done=false blocked={} stuck={d} computing={d} replacements={d} kind={t} worker={d} stuck_ms={d} loop_ms={d} cpu_ns={?d}\n",
            .{
                blocked.load(.acquire),
                stats.stuck_episodes,
                stats.computing_episodes,
                stats.replacements,
                if (stuck) |st| st.kind else .blocked,
                if (stuck) |st| st.worker else 0,
                if (stuck) |st| st.ms else 0,
                loop_ns.load(.acquire) / std.time.ns_per_ms,
                TestBackend.threadCpuTime(b, &b.sched.workers[0]),
            },
        );
    }
    try std.testing.expect(done.load(.acquire)); // it ran while the blocker still held the worker
    try std.testing.expect(stats.stuck_episodes >= 1);
    try std.testing.expect(stats.replacements >= 1);
    try std.testing.expectEqual(TestBackend.Sched.Stats.Kind.blocked, stats.last_stuck.?.kind);
    try std.testing.expectEqualStrings("blocker", stats.last_stuck.?.name);
}

test "watchdog: an idle instance has no rounds" {
    const b = try testBackend(2);
    defer destroyTestBackend(b);
    const io = b.io();

    const S = struct {
        /// Parks itself on a word nobody sets until the end of the test.
        fn park(inner_io: Io, word: *std.atomic.Value(u32)) void {
            inner_io.futexWaitUncancelable(u32, &word.raw, 0);
        }

        /// Waits for the watchdog to fall asleep, which it does once every worker is parked,
        /// then counts its rounds over a window, then wakes the parked tasks and waits for the
        /// watchdog to sample again. Each wait is for the state it expects, not for a fixed
        /// time, which a loaded machine can outlast. A thread that is not one of the workers
        /// wakes the tasks through the Io interface: a wake of the kernel would not reach tasks
        /// this backend parks itself.
        fn measure(b2: *TestBackend, inner_io: Io, word: *std.atomic.Value(u32), result: *Result) void {
            const s = &b2.sched;
            var tries: u32 = 0;
            while (!s.watchdog_sleeping.load(.acquire) and tries < 10_000) : (tries += 1) testSleep(std.time.ns_per_ms);
            result.asleep = s.watchdog_sleeping.load(.acquire);
            const start = s.stats().watchdog_rounds;
            testSleep(200 * std.time.ns_per_ms);
            const end = s.stats().watchdog_rounds;
            result.during = end - start;
            word.store(1, .release);
            inner_io.futexWake(u32, &word.raw, std.math.maxInt(u32)); // every waiter
            tries = 0;
            while (s.stats().watchdog_rounds == end and tries < 10_000) : (tries += 1) testSleep(std.time.ns_per_ms);
            result.woken = s.stats().watchdog_rounds != end;
        }

        const Result = struct { asleep: bool = false, during: u64 = 0, woken: bool = false };
    };

    var word: std.atomic.Value(u32) = .init(0);
    var result: S.Result = .{};
    // One task parked on each worker, so that both workers are parked and there is nothing to
    // sample.
    var parked_here = try b.sched.concurrentWith(.{ .affinity = .{ .pinned = 1 } }, S.park, .{ io, &word });
    var parked_there = try b.sched.concurrentWith(.{ .affinity = .{ .pinned = 0 } }, S.park, .{ io, &word });
    const thread = try std.Thread.spawn(.{}, S.measure, .{ b, io, &word, &result });
    io.futexWaitUncancelable(u32, &word.raw, 0);
    thread.join();
    // Awaited before the checks, so that a failed check fails the test instead of leaving tasks
    // that `deinit` finds never awaited.
    parked_here.await(io);
    parked_there.await(io);
    try std.testing.expect(result.asleep); // every worker parked: the watchdog waits untimed
    try std.testing.expectEqual(0, result.during); // and has no rounds while they stay parked
    try std.testing.expect(result.woken); // until a worker unparks, which wakes it
}

test "watchdog: without a barrier, a stuck task's next task waits for it" {
    const b = try testBackend(2);
    defer destroyTestBackend(b);
    b.sched.setWorkerLimit(1);
    b.heavy_barrier.store(false, .release); // this backend cannot issue the barrier
    const io = b.io();

    const S = struct {
        /// Queues one task behind itself, then holds the only worker for 200 ms without making
        /// an Io call. With no barrier from the backend, the watchdog takes no slot, so that
        /// task waits for this one.
        fn spinner(inner_io: Io, spinning: *std.atomic.Value(bool), witnessed: *std.atomic.Value(bool), spin_ns: u64) void {
            var group: Io.Group = .init;
            group.async(inner_io, witness, .{ spinning, witnessed });
            spinning.store(true, .release);
            const until = testNow() + spin_ns;
            var x: u64 = 0;
            while (testNow() < until) {
                inline for (0..8) |i| x +%= i;
                std.mem.doNotOptimizeAway(x);
            }
            spinning.store(false, .release);
            group.await(inner_io) catch {};
        }

        /// Records whether the task that held the only worker was still holding it.
        fn witness(spinning: *std.atomic.Value(bool), witnessed: *std.atomic.Value(bool)) void {
            witnessed.store(spinning.load(.acquire), .release);
        }
    };

    var spinning: std.atomic.Value(bool) = .init(false);
    var witnessed: std.atomic.Value(bool) = .init(true);
    var future = try b.sched.concurrentWith(.{ .affinity = .{ .pinned = 0 } }, S.spinner, .{ io, &spinning, &witnessed, 200 * std.time.ns_per_ms });
    future.await(io);
    // The queued task ran, and it ran after the spinner let the worker go.
    try std.testing.expect(!witnessed.load(.acquire));
}

test "watchdog: a computing task starts no replacement" {
    const b = try testBackend(4);
    defer destroyTestBackend(b);
    const io = b.io();

    const S = struct {
        /// Spins for 300 ms without making an Io call: computing, not blocked. The idle workers
        /// can take whatever it queues, and nothing about it is worth a replacement thread or a
        /// log line.
        fn compute(inner_io: Io, word: *std.atomic.Value(u32), spin_ns: u64) void {
            const until = testNow() + spin_ns;
            var x: u64 = 0;
            while (testNow() < until) {
                inline for (0..8) |i| x +%= i;
                std.mem.doNotOptimizeAway(x);
            }
            word.store(1, .release);
            inner_io.futexWake(u32, &word.raw, std.math.maxInt(u32));
        }
    };

    const before = b.sched.stats();
    var word: std.atomic.Value(u32) = .init(0);
    var future = try b.sched.concurrentWith(.{ .affinity = .{ .pinned = 1 } }, S.compute, .{ io, &word, 300 * std.time.ns_per_ms });
    // The main task parks, so that worker 0 is idle too while the computing task runs.
    io.futexWaitUncancelable(u32, &word.raw, 0);
    const after = b.sched.stats();
    try std.testing.expect(after.computing_episodes > before.computing_episodes);
    try std.testing.expectEqual(before.stuck_episodes, after.stuck_episodes); // nothing blocked
    try std.testing.expectEqual(before.replacements, after.replacements); // and nothing replaced
    try std.testing.expectEqual(TestBackend.Sched.Stats.Kind.computing, after.last_stuck.?.kind);
    future.await(io);
}
