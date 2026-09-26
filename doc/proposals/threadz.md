# Threadz

Green threads for Zig++: `std.Io.Threadz`, an `Io` implementation whose unit of work is a **task**
with its own stack, parked at Io calls and scheduled across a pool of OS workers. The docs say
"green threads" once, here, and "task" everywhere else.

Threadz is not new machinery. The tree already has the fiber switch (`Io/fiber.zig`), a
multi-threaded evented core on Linux (`Io.Uring`), a platform selector (`Io.Evented`), and
primitives that park through the vtable. Threadz is that machinery finished, given Zurtr's
scheduling policy and Zix's locality, and given the BEAM shapes on top. Everything after it in the
1.0 plan runs on it: pubsub fanout, driver pipelining, QUIC's timers and loss detection, jobs,
live sessions, and the `async`/`await` keywords, which lower to `Io.Future`.

The integration table below is grounded in the tree; the version with every `file:line` citation
is `threadz-integration.md`, built from the `StdIoMap`, `ZurtrRuntimeMap` and `ZixThreadingMap`
scout reports.

## Guiding star

1. **Zurtr and Zix decide first.** Zurtr's `runtime` supplies the policy: work stealing,
   completion delivery, structured scopes with failure policies. Zix supplies the rule "every
   byte owned, every thread deliberate": explicit worker counts, workers pinned physical-cores
   first, buffers with one owner, and the loops that produced its benchmark numbers never moving
   work between workers after accept. Threadz keeps every one of those properties, and the
   transport must not lose a benchmark by moving onto it.
2. **Where they are silent, do what the BEAM does.** Per-process heaps, bounded mailboxes with a
   policy, supervision on exit reasons, hot loading at quiescence, and a runtime that can always
   say what every task is doing.
3. **No compiler features in the first cut.** Zurtr and Zix run today without safepoints, and Zig
   has no garbage collector to demand them. Preemption is a flag for later, adopted if starvation
   shows up in practice rather than on principle.

## Integration with std

| Existing | In the tree today | Under Threadz |
| --- | --- | --- |
| The `Io` interface: `async`, `concurrent`, `await`, `cancel`, `group*`, `futexWait`/`futexWake`, `operate`, `batch*`, `sleep`, file, dir and net hooks (`Io.zig:51-244`) | One vocabulary. `Io.Evented` selects `Uring` on Linux, `Kqueue` on BSD and `Dispatch` on Darwin, wherever `fiber.zig` has a switch (`Io.zig:23-39`). | Unchanged. Threadz implements the vtable. `Io.Threadz` becomes the name of the platform selector; `Io.Evented` stays as an alias until nothing references it. |
| `std.Thread.Pool` | Does not exist. Upstream removed it; the compiler runs on `Io.Threaded` (`src/Zcu/PerThread.zig:62-69`). `std.Thread` is `spawn` and `getCpuCount`. | Nothing to rewrite. |
| `Io.Threaded` (19k lines) | One run queue for the whole pool: a linked list behind an `Io.Mutex` and `Io.Condition`, used LIFO (`Threaded.zig:31-49,1793-1807`). Workers spawned lazily and detached. When `async` hits its limit (CPU count − 1) the work runs on the caller. Blocked syscalls are cancelled with `SIGIO`, `tgkill` or `NtCancelSynchronousIoFile`. | Stays the OS-thread implementation and the dirty pool. **This is the justified rewrite**: per-worker queues with a LIFO slot and batch stealing in place of the single queue, syscall cancellation kept as is. The compiler and the build runner run on it, so "the compiler builds itself no slower" is the benchmark. |
| `Io.Mutex`, `Condition`, `Event`, `RwLock`, `Semaphore` | Every contended path ends at the vtable's `futexWait`/`futexWake`; none skips it. | Reused unchanged. Threadz implements the futex hooks as park and unpark. |
| `Io.Queue` / `TypeErasedQueue` | A mutex, a condition per waiter, intrusive waiter lists, a byte ring owned by the caller. Zero-capacity rendezvous, partial progress up to `min`, close, cancel. | API and semantics unchanged. A Vyukov ring is at most a fast path for the bounded case, added only if the scheduler benchmark puts the queue on the hot path. |
| `Group`, `Future`, `Select`, cancellation | `Select` is a `Group` plus a bounded `Queue`. Cancellation is `Canceled`, `recancel`, `swapCancelProtection`, `checkCancel`. | Unchanged. Zurtr's scopes and failure policies become `Io.Supervisor` on top of `Group`. The error is spelled `Canceled`, as std spells it. |
| `Io/fiber.zig` (323 lines) | A context switch for aarch64, riscv64 and x86_64: saves stack pointer, frame pointer and program counter, declares every other register clobbered. No stacks, no TLS handling, no Windows. | Reused as the switch. Added: Windows x64, which must also swap the TEB stack-base and stack-limit fields that `__chkstk` and structured exception handling read; and the rule that `threadlocal` is worker-local, since the switch does not move the TLS base. |
| `Io.Uring` (6.1k lines) | Already multi-threaded: a ring, a ready stack and a free-fiber queue per worker, CPU-count workers, idle workers stealing and waking each other with `MSG_RING`, futex waits through the ring. Fiber stacks are 60 MiB from the allocator, pooled, unguarded. Networking is unfinished: listen, accept, connect and DNS return `NetworkDown`; `net_write` panics (`Uring.zig:775-785,4986-5058`). `schedule` may resume a task on any worker. | Threadz's Linux core. Its scheduler is extracted into a module the other backends share. Stacks become guarded and lazily committed with a per-spawn size. Networking is finished. `schedule` gets the locality rule below. |
| `Io.Kqueue` (1.5k lines) | A stale draft: the futex, operate, batch and cancel hooks are missing from its vtable, and group operations panic. | Rewritten against the current vtable on the shared scheduler. The core for macOS and BSD. |
| `Io.Dispatch` (5k lines) | Built on GCD, which owns and sizes the threads. No per-worker queues, no networking. | Fails "every thread deliberate." Recommendation: leave it untouched until `Kqueue` reaches parity on Darwin, then retire it. Kept only if something turns up that needs GCD interop. |
| IOCP | Absent. | New, with the Windows fiber work. Last core to land. |
| Tests | `std.testing.io` is a `Threaded` instance. No evented backend is tested end to end. There is no scheduler benchmark. | The `Io/test.zig` contract tests run against Threadz on every platform it exists on. A scheduler benchmark is added and becomes the gate for the `Threaded` rewrite too. |

`Threaded` and `Threadz` share the interface, the primitives, the queue, and after the rewrite
the stealing policy. They differ in what a task is and what an Io call does to it: one blocks a
thread, the other parks a context and hands the thread to the next task. That is the one thing
that cannot be a configuration flag, and it is why they stay two constructors.

## Decisions

| Area | Decision | Reason |
| --- | --- | --- |
| Workers | One OS worker per core by default, count explicit, `-j` honored. Pinning physical-cores-first as an init option, on by default for servers. | Every thread deliberate. Today `-j` limits only `Threaded` (`src/main.zig:6634-6650`). |
| Affinity | Three per-task modes, set at spawn. **Sticky** (default): a task resumes on the worker that last ran it, and is stealable only when that worker's queue is over a threshold. **Pinned**: never moves; for tasks that own per-worker resources such as a listener or a slab. **Free**: goes to the global injection queue. | Zix's loops are pinned tasks; Zurtr's work is sticky; `Uring`'s `schedule` today is free, which is the missing locality rule. |
| Scheduling | Cooperative. A task runs until it parks at an Io call, awaits, or yields. A budget inside the implementation forces a yield after 128 consecutive operations without parking. | Zurtr today, plus Tokio's budget. No compiler involvement. |
| Runaway tasks | Work stealing contains a spinning task to one worker. A watchdog thread notices a worker that has not yielded in *N* ms, hands its run queue to a replacement worker, records a metric, and names the task. | Most of what preemption buys a server runtime, with nothing from the compiler. |
| Stacks | Pooled, guarded by one page, lazily committed by the OS, released with `MADV_DONTNEED` beyond the pool size. Reservation is per spawn: 256 KiB default for tasks, large (the current 60 MiB) for the compiler's own Sema work. No growth. | Growable stacks need pointer maps Zig does not have. The compiler's recursion is why `Uring` reserves 60 MiB today; lazy commit makes a big reservation cost only its mapping. One mapping per task meets `vm.max_map_count` near 65k tasks; that is a sysctl, and the docs say so. |
| Io calls | Sockets, timers, sleeps, futexes and file reads submit to the core and park the task. File operations the core can only finish on a kernel thread of its own (on Linux: positional writes, `statx`, `ftruncate`, opens that create or truncate, and directory changes) are made on the worker instead. | Upstream `Io` semantics; only the mechanism differs. The trip to the kernel thread and back costs more than the call: with every file write taking it, the compiler took 130 s to build itself at `-j8`, against 20-23 s with the writes made on the worker, as on `Threaded`. |
| Rings | A task pinned to a worker may borrow the worker's io_uring (`acquireRing`) and drive operations of its own on it. The SQEs it queues carry `ring_owner_bit` in `user_data`, and `Ring.waitCqes` parks it until one of their completions arrives. The worker handles every other completion as before. A ring has one owner at a time. | Zix's loops keep their multishot receives, provided buffer rings and batched submissions, which `Io` has no vocabulary for. A loop on a ring of its own would block its worker in the kernel. |
| Park, never block | Every wait inside Threadz ends at park. `Threaded`'s private helpers, including its own `Future` and `Group` waits, block on the OS futex; Threadz does not copy them. | A thread blocked inside a task is a worker lost. |
| Blocking | `io.blocking(fn, args)` runs on the dirty pool, which is an `Io.Threaded` instance. | Zix sets threads aside for blocking work; the BEAM calls them dirty schedulers. One implementation. |
| Extern calls | Not implicitly blocking. Long C calls use `io.blocking`; the watchdog covers the ones that forgot. | Per-call handoff would tax every FFI call for the few that block. |
| Structure | Every task belongs to exactly one `Io.Group`; the runtime is the root group; group exit joins; cancellation flows down. Data crossing a scope is `owned` or copied. | The borrow checker's frame rule makes this the only checkable shape. |
| Cancellation | `error.Canceled` at Io calls, through the existing `checkCancel` and protection API. Never unwinds a task. | Killing skips `defer`s, and shared memory makes that unsafe. |
| Memory | Each task gets an arena freed at task exit. Messages cross tasks by `owned` transfer or by copy into the receiver's arena. Zix's per-worker buffers either keep their task pinned or move to per-task ownership. | The BEAM's per-process heap, without a collector. |
| Mailboxes | `Io.Queue`, bounded, with a per-queue overflow policy: block, drop, or drop-and-resync. Tasks default to block; pubsub and live sessions to drop-and-resync. | Unbounded mailboxes are the BEAM's best-known footgun. |
| Task-local | `Io.Scoped(T)`, inherited down the group tree. `threadlocal` is documented as worker-local on Threadz. The compiler's per-thread IDs and the crash reporter, which is off in evented mode today, move to Scoped values and the task dump. | Tasks migrate at park points, and the switch does not move the TLS base. |
| Supervision | `Io.Supervisor`: a `Group` with a restart strategy (`one_for_one`, `one_for_all`, `rest_for_one`) applied to error returns, with OTP's restart intensity and period. Panics stay fatal. | Zurtr's failure policies, in std's vocabulary. Isolation does not transfer. |
| Hot swap | Code replacement takes effect once no task has the old code on its stack, checked at park points. | The BEAM's two-version rule at Io boundaries. ZEEX owns the rest. |
| Observability | Task IDs and names. A dump of every task's backtrace on `SIGQUIT`, which replaces the crash reporter under Threadz. "All tasks are asleep" deadlock detection. Scheduler gauges (run-queue depth, steals, handoffs, watchdog events) through the OpenTelemetry meter, with prometheuz as its exporter. | Go's two best-loved runtime features, and `observer`. Cheap now, miserable later. |
| Distribution | Out of scope. Task IDs reserve a node field. | QUIC is the transport; Zurtr owns it. |

## Do not port from Zurtr

The scout found three bugs in Zurtr's runtime that the move must leave behind: the
executor-shutdown test reads the executor after `deinit` has freed it (`task.zig:1375-1399`);
`next_seq` is incremented by concurrent submitters without a lock; and the completion queue the
docs call bounded is a growable list that drops completions when allocation fails. Zurtr also
steals naively: round-robin submission, one item stolen FIFO, no LIFO slot, no batch steal, no
pinning. The stealing policy comes from the table above, not from the current code.

## Later

- **Preemption.** Compiler-inserted safepoints behind a build flag, adopted if the watchdog metric
  shows starvation in real programs. The same prologue would carry a stack-limit check, removing
  the guard pages and the `vm.max_map_count` ceiling with them.
- **Growable stacks**, **isolation**, **distribution**, and the `async`/`await` syntax, which gets
  its own proposal and lowers onto this one.

## Plan

1. **`Threaded`'s queue.** Per-worker queues, LIFO slot, batch stealing, cancellation kept.
   Acceptance: `Io/test.zig` passes; the compiler builds itself no slower; the scheduler
   benchmark exists and is the gate from here on.
2. **The Linux core.** Extract `Uring`'s scheduler into the shared module; guarded lazily committed
   stacks with per-spawn size; the affinity rule in `schedule`; networking finished.
   Acceptance: `Io/test.zig` passes on Threadz; the compiler builds on Threadz with `-j` honored.
3. **Zix on Threadz.** The native loops become pinned tasks, one per worker, each with its
   `SO_REUSEPORT` listener and its worker's ring; buffers pinned with them or moved to arenas.
   Acceptance: parity with zix's recorded numbers measured on the same host with
   `localbench-isolate.sh` for `http1-uring`, `http1-ws-uring`, `http2-uring`,
   `http2-grpc-uring` and `http3-uring`. The recorded results are undated, so the baseline is
   re-measured first.
4. **Budget, blocking, watchdog.** A task counts the Io operations it completes without parking,
   and the operation past 128 yields it first, to the back of its worker's queue with its affinity
   kept; the constant is `scheduler.budget`, and every Threadz entry point charges, the count
   starting over at each park or yield. `io.blocking(fn, args)` is a new Io operation: `Threaded`,
   `Kqueue`, `Dispatch` and the failing implementation make the call on the calling thread, and
   Threadz runs it on a dirty pool, an `Io.Threaded` instance the instance owns and starts with
   the first such call, parking the calling task with the argument and result slots on its own
   stack; it is not cancelable once started. One watchdog thread per instance, started with the
   first task the instance makes or runs, samples every worker every 10 ms: a worker whose current
   task has not switched out for 100 ms is stuck, and its queued tasks become takeable by every
   other worker whatever their number. Two rounds ten milliseconds apart, and the worker thread's
   CPU time between them, tell the kinds apart: a thread that used less than a fifth of the wall
   time is blocked, in the kernel or in a call that never said it blocks, and gets one replacement
   worker, released when it switches again, and one `std.log` line scoped `.threadz` naming the
   task, by id and by the function it was spawned with, which is what tasks are named by now (see
   below); a thread that kept using its time is computing, which is what a compiler task is, and
   gets no replacement — `-j` and the worker limit mean what they say — and no log line.
   `Threadz.stats()` counts the episodes of each kind and says which the last one was. An instance whose worker limit is one still reports and replaces, and with
   every worker parked the watchdog sleeps until one unparks. A stuck worker's slot for the next
   task is the watchdog's to put in the shared queue, which is what lets a worker take its own
   slot with a plain load and store: the watchdog sets a flag, has the backend issue a
   process-wide barrier (`membarrier` on Linux, behind the scheduler's backend contract so that a
   core for another system can say it cannot), and re-reads the two words the worker stores at
   every switch; if it sees no change it takes the slot with an atomic exchange, and the worker
   takes it with a compare-exchange when it sees the flag or when its backend cannot issue a
   barrier. Pinned tasks stay put on a stuck worker. Acceptance: a spinning task costs one worker
   and nothing else waits; a blocking C call inside a task is detected and named.

   Shipped: the spin test queues 32 tasks behind a task that spins for 500 ms with no Io call and
   has them all finish 114-116 ms later with the stuck counter up and the log line naming the
   function; a task in a raw 300 ms `nanosleep` is reported the same way; 16 `io.blocking` calls
   of 200 ms each finish in 205 ms while other tasks run; and 10,000 operations that never park on
   a worker shared with one other task let that task run first. Task names come from
   `Io.spawnedName`, a comptime instantiation whose type name carries the function's declaration
   name, since the language has no reflection from a function value to its declaration; ids come
   from one counter for the program. Step 7 below adds the naming API on top.
5. **The other cores.** `Kqueue` rewritten on the shared scheduler for macOS and BSD; `Dispatch`
   retired once it reaches parity; IOCP and the Windows fiber work last.
6. **BEAM shapes.** Arenas, `Io.Scoped`, `Io.Supervisor`, overflow policies.
7. **Observability.** Dump, deadlock detection, scheduler metrics. This is where the OpenTelemetry
   plan attaches.
8. **Keywords.** In their own proposal.

## Acceptance for the whole

- Switching a program between `Threaded` and `Threadz` is one line and no semantic change.
- `Io/test.zig` passes on both implementations on every platform Threadz supports.
- The transport benchmarks do not regress by moving onto the runtime, measured on one host.
- The compiler builds itself no slower on the rewritten `Threaded`, and builds on Threadz.
- 50k tasks parked on queues fit in bounded memory, and the dump lists all of them.

## Open questions

1. Whether `Dispatch` has a reason to survive `Kqueue` parity.
2. The default stack reservation, measured against Zix's deepest call chain rather than guessed.
3. The sticky-steal threshold, chosen by the scheduler benchmark rather than by hand.
4. Whether IOCP is a 1.0 requirement or the first thing after it.
