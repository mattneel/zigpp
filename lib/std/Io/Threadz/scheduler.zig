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
//! A backend embeds the scheduler as its field `sched` and declares:
//! * `Worker`, its state for each worker, with `workerInit`, `workerStart` (called on the worker's
//!   own thread before it runs anything) and `workerDeinit`;
//! * `Completion`, what a finished operation leaves in its task's result slot;
//! * `poll(backend, worker, mode)`, which submits queued operations and passes each task whose
//!   operation finished to `readyFromPoll`. With `.block` it waits for at least one event.
//! * `wake(backend, from, to)`, which ends `to`'s current or next blocking `poll`;
//! * `cancelOperation(backend, from, task, token)`, which cancels the operation `task` is waiting
//!   for in the worker identified by `token`;
//! * `allocator(backend)`.

const builtin = @import("builtin");
const std = @import("../../std.zig");
const Io = std.Io;
const Alignment = std.mem.Alignment;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const linux = std.os.linux;
const posix = std.posix;
const page_size_min = std.heap.page_size_min;

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
        shared: TaskQueue,
        idle: Idle,
        stopping: std.atomic.Value(bool),
        stack_size: usize,
        /// The length of a mapping for a task with the default stack size.
        mapping_len: usize,
        pool: StackPool,
        idle_stack: []align(page_size_min) u8,
        /// The thread that calls `init` runs as this task, on its own stack, followed by room for
        /// a completion.
        main_task_buffer: [@sizeOf(Task) + completion_space]u8 align(@alignOf(Task)),

        const Sched = @This();

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
            context_bytes: [*]u8,
            /// Empty for the main task.
            mapping: []align(page_size_min) u8,
            name: if (tracy.enable) [*:0]const u8 else void,
            tsan_fiber: tsan.Fiber,

            var next_name: u64 = 0;

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
            /// The task to run next, which no other worker takes.
            run_next: ?*Task,
            lifo_streak: u8,
            /// Runnable tasks pinned here, oldest first.
            pinned: TaskList,
            /// Runnable sticky tasks.
            local: LocalQueue,
            /// Tasks other workers made runnable here.
            inbox: Inbox,
            /// Set while blocked in `poll` with nothing to run. A worker that clears it wakes this one.
            parked: std.atomic.Value(bool),
            tick: u32,
            steal_start: u32,
            stacks: StackCache,
            /// Tasks spawned here minus tasks that ended here. The sum over all workers counts
            /// the tasks alive.
            live: isize,
            name_arena: if (tracy.enable) std.heap.ArenaAllocator.State else struct {},
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

            pub fn currentTask(w: *Worker) *Task {
                assert(w.current_context != &w.idle_context);
                return @alignCast(@fieldParentPtr("context", w.current_context));
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
            /// `victim` has beyond `steal_threshold`, rounded up, into `q` without publishing
            /// them, and returns how many.
            fn grab(q: *LocalQueue, victim: *LocalQueue) u32 {
                const tail = q.tail.raw;
                while (true) {
                    const victim_head = victim.head.load(.acquire);
                    const victim_tail = victim.tail.load(.seq_cst); // see `notify`
                    const available = victim_tail -% victim_head;
                    if (available > capacity) continue; // `victim_head` was read before `victim_tail` moved on
                    if (available <= steal_threshold) return 0;
                    const excess = available - steal_threshold;
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

        pub const SpinLock = struct {
            locked: std.atomic.Value(bool) = .init(false),

            fn lock(l: *SpinLock) void {
                var spins: u32 = 0;
                while (l.locked.swap(true, .acquire)) {
                    while (l.locked.load(.monotonic)) {
                        spins +%= 1;
                        if (spins < 64) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
                    }
                }
            }

            fn unlock(l: *SpinLock) void {
                l.locked.store(false, .release);
            }
        };

        pub fn backendOf(s: *Sched) *Backend {
            return @fieldParentPtr("sched", s);
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
            const idle_stack = try mapStack(idle_stack_size);
            errdefer posix.munmap(idle_stack);
            const idle_indexes = try gpa.alloc(u32, count);
            errdefer gpa.free(idle_indexes);
            s.* = .{
                .workers = workers,
                .started = .init(1),
                .reserved = .init(1),
                .limit = .init(count),
                .shared = .{},
                .idle = .{ .stack = idle_indexes },
                .stopping = .init(false),
                .stack_size = options.stack_size,
                .mapping_len = std.mem.alignForward(usize, page + options.stack_size + default_header_size, page),
                .pool = .{},
                .idle_stack = idle_stack,
                .main_task_buffer = undefined,
            };
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
                .context_bytes = undefined,
                .mapping = &.{},
                .name = if (tracy.enable) "main task",
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
                .run_next = null,
                .lifo_streak = 0,
                .pinned = .{},
                .local = .{},
                .inbox = .{},
                .parked = .init(false),
                .tick = 0,
                .steal_start = index,
                .stacks = .{},
                .live = 0,
                .name_arena = .{},
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
            const started = s.started.load(.acquire);
            for (s.workers[1..started]) |*w| Backend.wake(s.backendOf(), &s.workers[0], w);
            for (s.workers[1..started]) |*w| w.thread.join();
            var live: isize = 0;
            for (s.workers[0..started]) |*w| {
                live += w.live;
                while (w.stacks.head) |free| {
                    w.stacks.head = free.next;
                    s.unmapFree(free);
                }
                Backend.workerDeinit(s.backendOf(), w);
            }
            assert(live == 0); // a task was never awaited
            while (s.pool.head) |free| {
                s.pool.head = free.next;
                s.unmapFree(free);
            }
            Worker.self = null;
            if (tsan.enable) tsan.__tsan_destroy_fiber(s.workers[0].idle_tsan_fiber);
            posix.munmap(s.idle_stack);
            gpa.free(s.idle.stack);
            gpa.free(s.workers);
            s.* = undefined;
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
                if (message.contexts.new != &w.idle_context) {
                    const task: *Task = @alignCast(@fieldParentPtr("context", message.contexts.new));
                    switch (task.affinity) {
                        .pinned => assert(task.home == w.index),
                        .sticky, .free => @atomicStore(u32, &task.home, w.index, .monotonic),
                    }
                    if (tracy.enable) tracy.fiberEnter(task.name);
                } else if (tracy.enable) tracy.fiberLeave();
                const old: ?*Task = if (message.contexts.old != &w.idle_context)
                    @alignCast(@fieldParentPtr("context", message.contexts.old))
                else
                    null;
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

        /// The next task this worker can run without looking at other workers or polling, or
        /// `null` every `poll_interval` switches so that its scheduling loop polls.
        fn nextLocal(s: *Sched, w: *Worker) ?*Task {
            w.tick +%= 1;
            if (w.tick % poll_interval == 0) return null;
            return s.takeLocal(w);
        }

        fn takeLocal(s: *Sched, w: *Worker) ?*Task {
            if (w.run_next) |task| {
                w.run_next = null;
                if (w.lifo_streak < lifo_limit) {
                    w.lifo_streak += 1;
                    return task;
                }
                // Tasks that keep waking each other do not get to starve the queue.
                w.local.push(task, &s.shared);
            }
            w.lifo_streak = 0;
            s.drainInbox(w);
            if (w.pinned.pop()) |task| return task;
            if (w.local.pop()) |task| return task;
            return s.shared.pop();
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
                // The shared queue first now and then, so that workers whose tasks keep them busy
                // do not starve it.
                const task = s.shared.pop() orelse s.takeLocal(w) orelse s.search(w) orelse {
                    if (s.stopping.load(.acquire)) return;
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
        /// searcher spinning for longer. Returns `null` if it found nothing to run.
        fn search(s: *Sched, w: *Worker) ?*Task {
            const spins: u32 = if (s.idle.searching.fetchAdd(1, .seq_cst) == 0) spin_limit else 0;
            var attempt: u32 = 0;
            while (attempt < @max(spins, search_rounds)) : (attempt += 1) {
                // A spinning worker looks through every queue only now and then.
                const scan = attempt < search_rounds or attempt % 16 == 0;
                const found = s.shared.pop() orelse
                    (if (!w.inbox.isEmpty()) s.takeLocal(w) else null) orelse
                    (if (scan) s.steal(w) else null);
                if (found) |task| {
                    // Workers queueing tasks wake nobody while one searches, so the last searcher
                    // to find something wakes another one if more is waiting.
                    if (s.idle.searching.fetchSub(1, .seq_cst) == 1 and s.anyStealable(w)) s.notify(w);
                    return task;
                }
                if (s.stopping.load(.monotonic)) break;
                std.atomic.spinLoopHint();
            }
            _ = s.idle.searching.fetchSub(1, .seq_cst);
            return null;
        }

        /// Takes tasks from another worker whose queue has more than `steal_threshold`: the
        /// older half of the excess, returning the last one taken to run.
        fn steal(s: *Sched, w: *Worker) ?*Task {
            const started = s.started.load(.acquire);
            w.steal_start +%= 1;
            for (0..started) |i| {
                const victim = &s.workers[(w.steal_start +% i) % started];
                if (victim == w) continue;
                const n = w.local.grab(&victim.local);
                if (n == 0) continue;
                const tail = w.local.tail.raw;
                const task = w.local.load(tail +% n -% 1);
                if (n > 1) w.local.tail.store(tail +% n -% 1, .seq_cst);
                return task;
            }
            return null;
        }

        fn anyStealable(s: *Sched, w: *Worker) bool {
            if (s.shared.len.load(.seq_cst) != 0) return true;
            for (s.workers[0..s.started.load(.acquire)]) |*victim| {
                if (victim != w and victim.local.len() > steal_threshold) return true;
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
        }

        /// Wakes `target` if it is parked and nobody has woken it yet.
        fn wakeWorker(s: *Sched, from: *Worker, target: *Worker) void {
            if (!target.parked.load(.seq_cst)) return; // see `parkWorker`
            if (target.parked.cmpxchgStrong(true, false, .seq_cst, .monotonic) != null) return;
            Backend.wake(s.backendOf(), from, target);
        }

        /// Finds a worker for work any worker may take: wakes a parked one, or starts one, unless
        /// a worker is searching already.
        ///
        /// Queueing work is a `.seq_cst` store and the loads here are `.seq_cst`. A worker about
        /// to park announces it with `.seq_cst` stores, then looks at the queues with `.seq_cst`
        /// loads. So either this sees the worker searching or parked, or the worker sees the work.
        fn notify(s: *Sched, from: *Worker) void {
            if (s.idle.searching.load(.seq_cst) != 0) return;
            if (s.idle.parked.load(.seq_cst) != 0) {
                while (true) {
                    const index = index: {
                        s.idle.lock.lock();
                        defer s.idle.lock.unlock();
                        if (s.idle.len == 0) return; // all being woken already
                        s.idle.len -= 1;
                        break :index s.idle.stack[s.idle.len];
                    };
                    const target = &s.workers[index];
                    if (target.parked.cmpxchgStrong(true, false, .seq_cst, .monotonic) == null) {
                        return Backend.wake(s.backendOf(), from, target);
                    }
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
                    const target = &s.workers[home];
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
            switch (when) {
                .next => {
                    task.status = .{ .queue_next = null };
                    const displaced = w.run_next orelse {
                        w.run_next = task;
                        return;
                    };
                    w.run_next = task;
                    w.local.push(displaced, &s.shared);
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

        fn spawn(
            s: *Sched,
            options: SpawnOptions,
            result_len: usize,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            start: @FieldType(Task, "start"),
        ) Io.ConcurrentError!*Task {
            const w: *Worker = .current();
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
                .context_bytes = context_bytes,
                .mapping = mapping,
                .name = if (tracy.enable) name: {
                    var name_arena = w.name_arena.promote(std.heap.page_allocator);
                    defer w.name_arena = name_arena.state;
                    break :name std.fmt.allocPrintSentinel(
                        name_arena.allocator(),
                        "task {d}",
                        .{@atomicRmw(u64, &Task.next_name, .Add, 1, .monotonic)},
                        0,
                    ) catch return error.ConcurrencyUnavailable;
                },
                .tsan_fiber = if (tsan.enable) tsan.__tsan_create_fiber(0),
            };
            if (builtin.cpu.arch == .x86_64) @as(*usize, @ptrFromInt(sp - 8)).* = 0; // no return address
            @memcpy(context_bytes[0..context.len], context);
            w.live += 1;
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

        fn mapStack(len: usize) error{OutOfMemory}![]align(page_size_min) u8 {
            const mapping = posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{
                .TYPE = .PRIVATE,
                .ANONYMOUS = true,
                .NORESERVE = true,
                .STACK = true,
            }, -1, 0) catch return error.OutOfMemory;
            guard(mapping[0..std.heap.pageSize()]);
            return mapping;
        }

        var guard_install: std.atomic.Value(bool) = .init(builtin.os.tag == .linux);

        /// Makes an access to `page` fault. Linux 6.13 guards a page without splitting the
        /// mapping, which keeps one mapping per task; older kernels need a mapping of its own.
        fn guard(page: []align(page_size_min) u8) void {
            if (builtin.os.tag == .linux and guard_install.load(.monotonic)) {
                switch (linux.errno(linux.madvise(page.ptr, page.len, linux.MADV.GUARD_INSTALL))) {
                    .SUCCESS => return,
                    else => guard_install.store(false, .monotonic),
                }
            }
            switch (builtin.os.tag) {
                .linux => _ = linux.mprotect(page.ptr, page.len, .{}),
                else => _ = std.c.mprotect(page.ptr, page.len, .{}),
            }
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
                posix.madvise(@alignCast(mapping.ptr + page), mapping.len - page - trim_keep, posix.MADV.DONTNEED) catch {};
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

        pub fn async(
            userdata: ?*anyopaque,
            result: []u8,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) ?*Io.AnyFuture {
            return concurrent(userdata, result.len, result_alignment, context, context_alignment, start) catch {
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
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) Io.ConcurrentError!*Io.AnyFuture {
            const s = fromUserdata(userdata);
            return s.spawnFuture(.{}, result_len, result_alignment, context, context_alignment, start);
        }

        fn spawnFuture(
            s: *Sched,
            options: SpawnOptions,
            result_len: usize,
            result_alignment: Alignment,
            context: []const u8,
            context_alignment: Alignment,
            start: *const fn (context: *const anyopaque, result: *anyopaque) void,
        ) Io.ConcurrentError!*Io.AnyFuture {
            const task = try s.spawn(options, result_len, result_alignment, context, context_alignment, .{ .future = start });
            s.enqueueSpawned(.current(), task);
            return @ptrCast(task);
        }

        pub fn await(
            userdata: ?*anyopaque,
            future: *Io.AnyFuture,
            result: []u8,
            result_alignment: Alignment,
        ) void {
            const s = fromUserdata(userdata);
            const awaiting: *Task = @ptrCast(@alignCast(future));
            if (@atomicLoad(?*Task, &awaiting.link.awaiter, .acquire) != Task.finished)
                s.yield(null, .{ .await = awaiting });
            @memcpy(result, awaiting.resultBytes(result_alignment));
            s.destroyTask(.current(), awaiting);
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
            start: *const fn (context: *const anyopaque) void,
        ) void {
            return groupConcurrent(userdata, type_erased, context, context_alignment, start) catch {
                start(context.ptr);
            };
        }

        pub fn groupConcurrent(
            userdata: ?*anyopaque,
            type_erased: *Io.Group,
            context: []const u8,
            context_alignment: Alignment,
            start: *const fn (context: *const anyopaque) void,
        ) Io.ConcurrentError!void {
            const s = fromUserdata(userdata);
            return s.spawnGroupMember(.{}, type_erased, context, context_alignment, start);
        }

        fn spawnGroupMember(
            s: *Sched,
            options: SpawnOptions,
            type_erased: *Io.Group,
            context: []const u8,
            context_alignment: Alignment,
            start: *const fn (context: *const anyopaque) void,
        ) Io.ConcurrentError!void {
            const group: Group = .{ .ptr = type_erased };
            const task = try s.spawn(options, 0, .@"1", context, context_alignment, .{ .group = .{
                .group = group,
                .start = start,
            } });
            group.addTask(task);
            s.enqueueSpawned(.current(), task);
        }

        pub fn groupAwait(userdata: ?*anyopaque, type_erased: *Io.Group, initial_token: *anyopaque) Io.Cancelable!void {
            const s = fromUserdata(userdata);
            _ = initial_token;
            s.yield(null, .{ .group_await = .{ .ptr = type_erased } });
        }

        pub fn groupCancel(userdata: ?*anyopaque, type_erased: *Io.Group, initial_token: *anyopaque) void {
            const s = fromUserdata(userdata);
            _ = initial_token;
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
            _ = userdata;
            const task = Worker.current().currentTask();
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
                options,
                @sizeOf(Result),
                .of(Result),
                @ptrCast(&args),
                .of(Args),
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
            return s.spawnGroupMember(options, group, @ptrCast(&args), .of(Args), TypeErased.start);
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

/// A backend with no operations, for testing the scheduler alone. Workers block on a futex word,
/// and `Io` futexes park tasks in a list under one lock.
const TestBackend = struct {
    sched: Sched,
    gpa: Allocator,
    futex_lock: Sched.SpinLock = .{},
    futex_waiters: ?*FutexWaiter = null,

    const Sched = Scheduler(TestBackend);

    pub const Completion = struct { result: i32 };
    pub const Worker = struct { wake_word: std.atomic.Value(u32) };

    const FutexWaiter = struct {
        ptr: *const u32,
        task: *Sched.Task,
        next: ?*FutexWaiter,
    };

    fn init(b: *TestBackend, gpa: Allocator, workers: usize) !void {
        b.* = .{ .sched = undefined, .gpa = gpa };
        try b.sched.init(gpa, .{ .workers = workers });
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
        _ = w;
    }

    pub fn workerDeinit(b: *TestBackend, w: *Sched.Worker) void {
        _ = b;
        _ = w;
    }

    pub fn poll(b: *TestBackend, w: *Sched.Worker, mode: PollMode) void {
        _ = b;
        switch (mode) {
            .nonblocking => _ = w.backend.wake_word.swap(0, .acquire),
            .block => while (w.backend.wake_word.swap(0, .acquire) == 0) {
                _ = linux.futex_4arg(&w.backend.wake_word.raw, .{ .cmd = .WAIT, .private = true }, 0, null);
            },
        }
    }

    pub fn wake(b: *TestBackend, from: *Sched.Worker, to: *Sched.Worker) void {
        _ = b;
        _ = from;
        to.backend.wake_word.store(1, .release);
        _ = linux.futex_4arg(&to.backend.wake_word.raw, .{ .cmd = .WAKE, .private = true }, 1, null);
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
        const w: *Sched.Worker = .current();
        while (woken) |waiter| {
            woken = waiter.next;
            b.sched.ready(w, waiter.task);
        }
    }
};

fn testBackend(workers: usize) !*TestBackend {
    if (builtin.os.tag != .linux or !Io.fiber.supported or builtin.single_threaded) return error.SkipZigTest;
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
