# Threadz: the integration table, corrected against the tree

Built from three read-only scout reports, StdIoMap (std and the compiler), ZurtrRuntimeMap
(Z: = `~/src/zurtr`) and ZixThreadingMap (zix: = `~/src/zix`). Line numbers in this tree are for
master at `711f09ad45`, after the merge of upstream's master. The scouts read the code only; nothing
was built or run, and [inferred] marks a scout's inference rather than a line it read.

## What each existing piece is, and what it becomes

| Existing (this tree) | What it is today | Under Threadz |
| --- | --- | --- |
| `Io` interface: `async`, `concurrent`, `await`, `cancel`, `groupAsync/Concurrent/Await/Cancel`, `recancel`, `swapCancelProtection`, `checkCancel`, `futexWait`, `futexWaitUncancelable`, `futexWake`, `operate`, `batch*`, `sleep`, `now`, and the dir/file/process/net hooks (`lib/std/Io.zig:51-244`) | The one vocabulary. `Io.Evented` already selects `Uring` on Linux, `Kqueue` on the BSDs and `Dispatch` on Darwin wherever `fiber.supported` (`lib/std/Io.zig:23-39`). | Unchanged: Threadz implements the vtable. |
| `std.Thread.Pool` | **Does not exist.** It was removed upstream, and the compiler moved to `Io.Threaded` ("temporary workaround ... migrate from `std.Thread.Pool` to `std.Io.Threaded`", `src/Zcu/PerThread.zig:62-69`). `std.Thread` is now kernel threads, `spawn` and `getCpuCount` (`lib/std/Thread.zig:287-350`). | Nothing to rewrite. The rewrite target is `Io.Threaded`'s run queue (next row). |
| `Io.Threaded` (19,266 lines) | **A single `std.SinglyLinkedList` run queue behind an `Io.Mutex` and an `Io.Condition`.** Producers `prepend` and workers `popFirst`, so it is LIFO (`lib/std/Io/Threaded.zig:31-49,1793-1807,2115-2119`). Workers are spawned lazily and detached, with 16 MiB default stacks. The `async` limit is the CPU count minus 1, and at the limit `async` runs the work eagerly on the caller. The `concurrent` limit is unlimited (`:1575-1654,2065-2164`). Blocking syscalls are cancelled with `SIGIO`, `tgkill`, or `NtCancelSynchronousIoFile` (`:1260-1392`). | Stays the OS-thread implementation. **This is the justified rewrite:** replace the global LIFO list with per-worker queues and stealing, keeping the syscall-cancellation machinery. The compiler runs on `Threaded` unless it is built with `io_mode = .evented` (`src/main.zig:189-209`), so "the compiler builds itself no slower" remains the benchmark. The build runner is also `Threaded`: it uses `Group.async` per ready step, and `Select` for watch mode (`lib/compiler/Maker.zig:146-164,2421-2461,812-867`). |
| `Io.Mutex`, `Io.Condition`, `Io.Event` (the ResetEvent equivalent), `Io.RwLock`, `Io.Semaphore` (`lib/std/Io.zig:1705-2062`, `lib/std/Io/RwLock.zig`, `lib/std/Io/Semaphore.zig`) | **Every contended path goes through the vtable's `futexWait`/`futexWake`, and none bypasses it.** RwLock is built from a mutex and a semaphore; Semaphore from a mutex and a condition. | Reused unchanged. Under Threadz, `futexWait` parks the task, as `Uring` already does with `IORING_OP_FUTEX_WAIT` and a yield (`lib/std/Io/Uring.zig:1922-2059`). |
| `Threaded`'s private primitives | Its own worker condvar, mutex and event, and its `Future`/`Group` awaits, call `Thread.futexWait` directly (`lib/std/Io/Threaded.zig:2282-2309,2417-2430,18989-19112`). There is also a private `WaitGroup` (`:18955-18984`). | Not copied into Threadz: these block the OS thread. |
| `Io.Queue` / `TypeErasedQueue` (`lib/std/Io.zig:2066-2488`) | Built from an `Io.Mutex`, a stack-allocated `Io.Condition` per pending put or get, intrusive waiter lists, and a caller-owned byte ring. Supports zero-capacity rendezvous, partial progress up to `min`, close, and cancellation. | API unchanged. A Vyukov ring can only be a fast path for the bounded case, and it has to keep all of those semantics. Zurtr's `Queue(T)` owns its storage and exposes `push(bool)` and `pop(?T)`, so it is an internal primitive, not a drop-in replacement (Z:`src/runtime/mpmc.zig:61-168`). |
| `Io.Group`, `Io.Future`, `Io.Select`, cancellation (`lib/std/Io.zig:1288-1656,813-816`) | `Group` is an atomic token plus state owned by the implementation. `Select` is a `Group` plus a bounded `Queue`. The cancellation calls are `Cancelable`, `recancel`, `swapCancelProtection` and `checkCancel`. | Unchanged. Zurtr's `Scope`, `FailurePolicy` and `Failure` become `Io.Supervisor` and the other BEAM shapes on top. Zurtr's `error.Cancelled` becomes std's `error.Canceled`, and Zurtr's own flag check is replaced by `checkCancel`. |
| `Io/fiber.zig` (323 lines) | **The context switch only**, for aarch64, riscv64 and x86_64. It saves SP, FP and PC; every other register, including FP/SIMD, is declared clobbered. It does not switch the TLS base or allocate stacks (`lib/std/Io/fiber.zig:1-321`). | Reused as Threadz's switch. Still needed: [inferred] Windows x64, which has to swap the TIB's stack base and limit for `__chkstk` and SEH; and a TLS rule for tasks that migrate. |
| `Io.Uring` (6,110 lines) | **Multi-threaded already.** The main thread plus `thread_limit` workers (CPU count by default), each with its own io_uring, a ready linked stack and a free-fiber queue. Idle workers steal a bounded number of fibers and wake each other with `MSG_RING`. A completion arrives at the submitting ring's idle loop and then goes through `schedule`, which can move the fiber to another worker. Futexes go through the ring (`lib/std/Io/Uring.zig:94-118,794-818,927-1220,1922-2059`). Each fiber gets a stack of at least 60 MiB from the allocator; stacks are pooled, with no guard page and no lazy commit (`:55-57,251-320`). **The networking is unfinished:** IP listen, accept, connect and DNS return `NetworkDown`, `net_send` returns `NetworkDown`, `net_write` panics, and batched net operations panic (`:775-785,2096-2121,4986-5058`). | Threadz's Linux core. Replace the stacks with 256 KiB `mmap`ed, guard-paged, lazily committed and pooled ones; implement the networking; add a locality policy for where a completion's task runs (see below). |
| `Io.Kqueue` (1,502 lines) | **A stale draft.** Its vtable does not match the current `Io`: the futex, operate, batch and cancellation hooks are missing, and every group operation panics (`lib/std/Io/Kqueue.zig:616-662,743-808`). It has a kqueue per worker, ready stealing, and 4 MiB stacks that are freed rather than pooled. | Rewrite it against the current vtable, reusing Uring's scheduler shape with kqueue as the submission layer. This is the macOS/BSD core. |
| `Io.Dispatch` (4,987 lines) | libdispatch: GCD sizes and owns the threads, and there are no per-worker queues and no stealing. Futexes wait on a serial dispatch queue. Fiber stacks are 60 MiB and freed. Networking is unavailable (`lib/std/Io/Dispatch.zig:41-80,195-224,344-469,1642-1689`). | Does not meet "every thread deliberate". Your call: keep it as a separate `Io`, or retire it once Kqueue works. |
| IOCP | Absent. | New. |
| Tests | `std.testing.io` is a `Threaded` instance (`lib/std/testing.zig:21-24`). The contract tests are `lib/std/Io/test.zig` and `lib/std/Io/Threaded/test.zig`. There are no end-to-end evented tests: Uring's and Dispatch's test blocks only reference `CancelProtection` (`Uring.zig:6107-6110`, `Dispatch.zig:4984-4987`). There is no scheduler benchmark. | Run `Io/test.zig` against Threadz, and add a scheduler benchmark. |
| The compiler on an evented Io | `-j` caps only `Threaded`, not `Evented` (`src/main.zig:6634-6650`). The per-thread IDs rely on `threadlocal` slots (`src/Zcu/PerThread.zig:62-118`), and the crash reporter is disabled in evented mode (`src/crash_report.zig:1-4`). | Before the compiler can run on Threadz, `-j` has to apply to it and the threadlocal uses need auditing. |

## Zurtr's runtime (Z: = /home/autark/src/zurtr)

- **`mpmc.zig` (299 lines).** A Vyukov bounded MPMC FIFO: power-of-two capacity, a sequence number per cell, one cache line per cell, CAS with `acq_rel` and publication with `release`. It has four tests, including four producers and eight consumers exchanging 80k values (`Z:src/runtime/mpmc.zig:34-299`). It can move into std as an internal primitive.
- **`pool.zig` (503 lines) and `task.zig` (1,657 lines).**
  - Every task runs to completion on an OS-thread stack; neither file has green parking or implements `Io` (`Z:src/runtime/pool.zig:266-299`, `Z:src/runtime/task.zig:360-376`).
  - Submission is a global round-robin counter followed by a cyclic scan. Stealing takes one item, FIFO, from the next victim in order. There is no LIFO wake slot, no batch steal, and no pinning.
  - Workers park on an `Io.Condition` with a 5 ms timeout guard (`Z:src/runtime/pool.zig:179-205,300-328`, `Z:src/runtime/task.zig:305-406`).
  - These are reference behaviour for the scope and supervisor layer, not code that ports as-is.
- **Bugs not to carry over:**
  - The executor-shutdown test calls `abandonedCount()` after `deinit()` has destroyed the executor, a use-after-free (`Z:src/runtime/task.zig:1375-1399`).
  - `next_seq` is a plain `u64` incremented outside the lock, so concurrent submitters race on it (`Z:src/runtime/pool.zig:89-90,179-199`).
  - The completion queue is documented as bounded, but it is a growable `ArrayList` that drops completions when allocation fails (`Z:src/runtime/pool.zig:85-88,279-289`).
  - The waiter-with-a-lock test skips itself when it detects the deadlock (`Z:src/runtime/task.zig:1400-1467`).

## Zix (zix: = /home/autark/src/zix)

- **Benchmarked paths.** These are EPOLL and URING, for H1, WS, H2, gRPC and H3.
  - Zix drives its own loop per worker, one worker per allowed CPU (`workers=0` reads `sched_getaffinity`).
  - Each worker has its own `SO_REUSEPORT` listener and is pinned with `pinToCpu`, physical cores before SMT siblings.
  - Buffers are per worker: slabs, stream-slot pools, and the QUIC CID tables.
  - Nothing migrates after accept, and there is no shared queue (`zix:src/tcp/http1/dispatch/common.zig:285-385`, `epoll.zig:124-131,917-958`, `uring.zig:1480-1512,1760-1784`, `udp/http3/dispatch/uring.zig:540-565`).
- **Where `std.Io` is used.** It is injected only for auxiliary operations. The `.ASYNC` path is the only one that runs on `Io`:
  - H1, H2 and gRPC accept in one loop and call `io.async` per connection (`zix:src/tcp/http1/dispatch/async.zig:16-45`).
  - H3 in `.ASYNC` mode is a synchronous UDP receive loop (`zix:src/udp/http3/dispatch/common.zig:530-604`).
  - Zix has no kqueue or IOCP backend; non-Linux uses `.ASYNC` (`zix:docs/concurrency-en.md:258-279`).
- **Handler shapes.**
  - H1 handlers run inline in the loop.
  - H2 and gRPC keep their mux state across events and run route handlers inline.
  - H3 runs handlers inside the datagram driver, on a 4 KiB scratch stack arena.
  - A conversion to Threadz either keeps driving these state machines or redesigns suspension and buffer lifetimes (`zix:src/tcp/http1/core.zig:53-111,389-417`, `zix:src/tcp/http2/grpc/mux.zig:397-452`, `zix:src/udp/http3/core.zig:18-34,201-229`).
- **Recorded numbers.** The report files are undated (`zix:docs/benchmark/HttpArena-result-*.md`):
  - H1 URING: 4.19M RPS at 512 connections and 4.43M at 4096; pipelined 53.4M and 55.4M.
  - WS echo: 4.57M; pipelined 66.9M.
  - gRPC unary: 7.21M.
  - gRPC server streaming: 8.54M.

  The dated H2 and H3 numbers come from different hosts. The external `../HttpArena` checkout is absent.
- **How to reproduce.** Run `scripts/localbench-build.sh all --release fast`, then `localbench-validate.sh <entry>`, then `sudo -E localbench-isolate.sh <entry> --probe --sample-mem --summarize` for `http1-uring`, `http1-ws-uring`, `http2-uring`, `http2-grpc-uring` and `http3-uring`. For the full arena, use `httparena-benchmark-isolate.sh <fw> ../HttpArena --source local` (`zix:localbench/README-en.md`, `zix:scripts/httparena-benchmark-isolate.sh`).

## The three questions

1. **Thread-per-core or work stealing?**
   - Zix's numbers come from pinned, share-nothing loops, one per core.
   - Zurtr steals, but naively: a one-item FIFO steal, with round-robin submission.
   - `Io.Uring` already has the shape you describe: a ring per worker, stealing, and completions routed through `schedule`. What it lacks is a locality rule. Today a completion lands on the ring's worker and `schedule` may move it anywhere.
   - HttpArena parity also requires more than swapping `config.io`. That swap only touches zix's `.ASYNC` path, not the native loops the numbers come from. The native loops have to become Threadz tasks, with worker-owned buffers either pinned to the task's worker or moved to per-task ownership.
2. **Do the Io primitives park the task or block the worker?**
   - Park. Every public primitive's contended path ends at the vtable's futex hooks, and `Future`/`Group` waits end at `await`/`groupAwait`.
   - Threadz therefore parks everything by implementing those hooks, as `Uring` and `Dispatch` already do.
   - The only direct OS-futex blocking is inside `Threaded`'s private helpers, which Threadz must not copy.
   - Code that blocks outside `Io` (C libraries, raw syscalls) is what `io.blocking` and the watchdog are for.
3. **What is `fiber.zig`?** A 3-word context switch for aarch64, riscv64 and x86_64, with no stacks, no TLS and no Windows TIB handling.
   - `Uring` is multi-threaded with stealing.
   - `Kqueue` is a stale draft whose vtable does not match `Io`.
   - `Dispatch` runs on GCD's threads.
   - Stacks are 4-60 MiB from the allocator, with no guard pages.
   - Nothing evented is tested end to end, and `Uring` can't listen or accept yet.
   - So Threadz extends: it keeps `fiber.zig` and Uring's scheduler shape, and replaces the stacks, the Kqueue vtable, the networking gaps, and Threaded's run queue.
