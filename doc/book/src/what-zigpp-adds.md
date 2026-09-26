# What Zig++ Adds

Zig++ is a superset of Zig: every valid Zig program is a valid Zig++ program,
unless it names something `priv` (write `@"priv"` instead). This chapter covers
what it adds to the language and to the toolchain.

## Private fields

Struct fields are public by default, meaning that they can be accessed from any
file. A field declared with the `priv` keyword is *private*: it can only be
accessed by name from within the file which declares the struct, just like a
declaration which is not marked `pub`. Private fields allow a type to protect
its invariants, and communicate which fields are implementation details rather
than part of its API.

```zig
/// A fixed-capacity buffer. `len` is private, so code in other files cannot
/// break the invariant that `len` never exceeds the capacity of `bytes`.
pub const Buffer = struct {
    bytes: [16]u8 = undefined,
    priv len: usize = 0,

    pub fn append(buf: *Buffer, byte: u8) error{Overflow}!void {
        if (buf.len == buf.bytes.len) return error.Overflow;
        buf.bytes[buf.len] = byte;
        buf.len += 1;
    }

    pub fn slice(buf: *const Buffer) []const u8 {
        return buf.bytes[0..buf.len];
    }
};
```

Code in the file that declares the type names `len` normally, including in
tests. Code in any other file cannot name it, whether to read it, write it,
take its address, call it, or specify its value in an initialization
expression:

```zig
const Buffer = @import("test_private_fields.zig").Buffer;

test "access a private field from another file" {
    var buf: Buffer = .{};
    try buf.append('a');
    buf.len = 100;
}

// error: field 'len' of struct 'test_private_fields.Buffer' is private
```

Private fields are still a part of every value of the type. If a private field
has no default value, the struct can only be initialized within the file which
declares it, typically by a public initialization function.

Field privacy only applies to syntax which names a field. `@typeInfo` reports
private fields along with all other fields, indicating their privacy with the
`@"priv"` field attribute, and builtins which name fields with strings, such as
`@field`, `@FieldType`, and `@offsetOf`, can be used with any field. This
allows generic code, such as formatting, comparison, hashing, and
serialization, to operate on all types:

```zig
const info = @typeInfo(Buffer).@"struct";
try expect(std.mem.eql(u8, info.field_names[1], "len"));
try expect(info.field_attrs[1].@"priv");

// Builtins which name fields with strings are not subject to field privacy.
var buf: Buffer = .{};
try buf.append('z');
try expect(@field(buf, "len") == 1);
```

A type created with `@Struct` or `@Union` has private fields where the
`@"priv"` field attribute is set; these fields are private to the file
containing the reification builtin call.

Fields of unions can be marked `priv` as well. Outside of the file which
declares the union, a private field cannot be accessed, initialized, or have
its payload captured by a `switch` prong which names it. The tag of a union is
not affected by field privacy, so it can still be compared against and switched
on, and `else` and `inline else` prongs can capture any payload.

Enum fields and tuple fields cannot be marked `priv`.

`priv` precedes `comptime` in a field declaration, so the grammar rule for a
container field is:

```text
ContainerField <- doc_comment? KEYWORD_priv? KEYWORD_comptime? (IDENTIFIER COLON)? TypeExpr ByteAlign? (EQUAL Expr)?
```

Because `priv` is a keyword, an identifier with that name is written
`@"priv"`. That is the whole of the incompatibility with Zig: a Zig program
that declares, or names, something `priv` needs quotes around it, and one that
expects to reach private fields from other files needs a different design.

Upstream closed the [proposal for private
fields](https://github.com/ziglang/zig/issues/9909) as not planned. Zig++'s
release smoke test compiles a struct with a private field, runs it, and checks
that naming that field from another file fails with a diagnostic that mentions
`private`.

The full rules, with the tests above as runnable examples, are in the language
reference: [Private Fields](/langref.html#Private-Fields).

## LLVM is forever

Upstream Zig plans to drop its dependency on the LLVM libraries, and to
[eliminate its dependency on the LLVM library API
calls](https://github.com/ziglang/zig/issues/25492). Zig++ will never phase out
LLVM. In `package.json` terms, LLVM stays in `dependencies`.

Zig++ builds with LLVM, Clang, and LLD 23.1.2. The CMake build requires the
23.x development libraries, and refuses a `llvm-config` older than 23 or newer
than 24.

The LLVM-less build path exists only to bootstrap: `cc -o bootstrap bootstrap.c
&& ./bootstrap` produces a `zig2` that is a stage2 build without LLVM
extensions, and it lacks release-mode optimizations, some ELF, COFF/PE, and
WebAssembly linking features, the ability to create static archives from object
files, the ability to compile assembly files, and the ability to compile C,
C++, Objective-C, and Objective-C++. It is enough to run `./zig2 build` and
produce a real Zig++ compiler. See
[Building from Source](building-from-source.md).

LLVM, and MLIR above it, are how Zig++ goes the final stretch on GPUs: the
blessed path lowers Zig++ directly to PTX, to AMD GPU code objects, and to Metal
libraries, with first-class GPU intrinsics.

## `std.gpu`

`std.gpu` is a port of [ugpu](https://github.com/mattneel/ugpu) into the
standard library. Kernels are plain Zig functions, and they can use the rest of
the standard library as long as they avoid operating system services:

- Kernels are exported functions with the `.kernel` calling convention, and
  `std.gpu` has CUDA's indexing (`threadIdx`, `blockIdx`, `blockDim`,
  `gridDim`, `globalId`), `syncThreads` (the new `@workGroupBarrier` builtin),
  warp shuffles, votes and reductions, atomics, fast math approximations, and
  `print`.
- The standard library runs on the GPU: `std.fmt`, `std.json`, `std.mem`,
  `std.base64`, hash maps, and array lists, with allocators for shared memory
  and for the CUDA device heap in `std.gpu.allocators`. A panic in a kernel
  reports its message to the host.
- Every NVPTX and AMDGPU module carries the compiler-rt routines that it calls,
  so `@sin`, `@exp`, `@log`, `f128`, and float parsing work in kernels, with
  the same results as on the host, bit for bit. Upstream Zig crashes LLVM on
  `@sin` for NVPTX.
- `std.gpu.cuda` loads the CUDA driver at run time, so programs build without
  the CUDA toolkit, and launches kernels from the host. `std.gpu.hip` does the
  same with the HIP runtime of AMD GPUs, on Linux and on Windows, where it
  needs no libc.
- `std.gpu.metal` loads the Metal framework at run time and runs kernels that
  Zig++ compiles for the `air64` target into a `.metallib`, in process, with no
  Xcode, Metal toolchain, or macOS SDK. Apple GPUs have no `f64`, no `print`,
  and only relaxed atomics, which are compile errors in their kernels.

[GPU Programming](gpu.md) covers the device API, the host APIs, and how to
build and run the kernels. Still to come: MLIR lowering for tensor cores and
kernel fusion.

## `std.Io.Threadz`

`std.Io.Threadz` is an `Io` implementation whose unit of work is a task: a
function on its own stack that parks at every Io call that has to wait, while
its worker thread runs the next task. Code written against `std.Io` runs on it
unchanged, so moving a program between `Io.Threaded` and `Io.Threadz` changes
one declaration:

```zig
var threadz: std.Io.Threadz = undefined;
try threadz.init(gpa, .{});
defer threadz.deinit();
const io = threadz.io();
```

- The thread that calls `init` becomes the first worker and runs as the main
  task. More workers start as work arrives, one per CPU by default;
  `setWorkerLimit` lowers the count, and `-j` sets it in a compiler built with
  `-Dio-mode=evented`.
- Every task gets its own mapping with a guard page below the stack, and the OS
  commits the stack as the task touches it. Tasks reserve 256 KiB unless
  `InitOptions.stack_size` says otherwise.
- A task's affinity decides where it runs whenever it becomes runnable.
  `.sticky`, the default, resumes it on the worker that last ran it, and another
  worker takes it only from a worker with more than one task waiting.
  `.pinned` keeps it on one worker, for tasks that own that worker's
  resources. `.free` lets whichever worker is idle take it. `concurrentWith`
  and `groupConcurrentWith` take the affinity and the stack size of one task.
- A task may complete 128 `Io` operations that nobody has to wait for before
  the next one goes to the back of its worker's queue, with its affinity
  unchanged. A worker takes the task it will run next, and a stuck worker's
  slot for that task is the watchdog's to hand to the shared queue, so that a
  worker pays no locked instruction for its own: the watchdog has the backend
  issue a process-wide barrier, re-reads what that worker wrote at its last
  switch, and takes the slot only when the worker has not switched since.
  Counting operations without parking bounds what one task can take
  from a worker without ever blocking on it, and the count starts over at every
  park or yield. `scheduler.budget` is the 128.
- Every task has an id, unique among the tasks the program made, and a name:
  the function it was spawned with, from `Io.spawnedName`. A log line and, in
  step 7, the task dump use them.
- `Io.blocking(fn, args)` is for a call that blocks without making an `Io` call
  of its own, such as a C library call or a raw syscall. It returns `fn`'s
  result, errors and all. On Threadz it runs on a dirty pool, an `Io.Threaded`
  instance the Threadz instance starts with the first such call, while the
  calling task parks; the arguments and the result stay on the task's stack,
  where the pool thread reads them and writes the result back. Sixteen calls
  that each block for 200 ms finish in 205 ms, and the workers go on running
  tasks while they block. The call is not cancelable once it has started:
  a cancelation request takes effect at the calling task's next cancelation
  point after it returns. `Io.Threaded`, and the implementations with no pool
  of their own, make the call on the calling thread.
- One watchdog thread per instance, started with the first task the instance
  makes or runs, samples every worker every 10 ms. A worker whose current task
  has not switched out for 100 ms is stuck. Two rounds ten milliseconds apart,
  and the worker thread's CPU time between them, tell the kinds apart: a thread
  that used less than a fifth of the wall time is blocked, in the kernel or in
  a C call that never said it blocks, and its task is named in one `std.log`
  warning scoped `.threadz`, with its id, the function it was spawned with, and how
  long it has run; its queued tasks, including the ones others made runnable
  for it and the one in its slot for the next task, become takeable by every
  worker whatever their number; and one replacement worker is started for it,
  so that parallelism stays, which stops once it has nothing to run and the
  stuck worker switches out again. A thread that kept using its time is
  computing, which is what a compiler task is: its queued work is taken just
  the same, but no worker is replaced — `-j` and the worker limit mean what
  they say — and nothing is logged. An instance whose worker limit is one is
  covered for a blocked task, whose replacement is where the tasks behind it
  run; a computing task holds the only worker, which is what a limit of one
  means. Pinned tasks on a stuck worker wait for the worker, which they own the
  resources of. `Threadz.stats()` reads the counters: workers running,
  replacements, workers stuck now, the episodes of each kind, the rounds of
  samples the watchdog made, and the task and kind of the last episode. A
  worker pays one store per switch for the watchdog, of the task it runs, and
  one load of a flag when it takes the task in its slot. The watchdog waits
  without a timeout while every worker is parked, so an idle program is not
  woken to sample workers that are all asleep.
- Threads outside the pool can use the synchronization primitives, whose
  futexes they wait on in the kernel. Other `Io` calls from those threads are
  not supported yet.
- `io.arena()` returns an allocator whose memory belongs to the task: it is
  created at the first call, out of the implementation's own allocator, and
  released in one step when the task returns, whether the task freed it or not.
  It is where a task puts what it builds up and throws away, such as one
  request's parsed headers, and a value that outlives the task must not point
  into it: move what crosses a task boundary by `owned` transfer, into another
  allocator, or by copying it. The main thread has one too, released by
  `deinit`, and `io.blocking` runs on a thread of its own, which sees neither
  the calling task's arena nor its scopes.
- `io.Scoped(T)` is the task-local value. The key is declared once at container
  level, `const current = std.Io.Scoped(u32)`, bound around a call with
  `current.run(io, value, function, args)`, and read with `current.get(io)` -
  in that call, and in every task spawned inside it, a child of a child as much
  as a child, each by its own copy, until it returns. Outside the call the
  binding is gone. `threadlocal` cannot serve this on Threadz: a task migrates
  at a park point, and the fiber switch does not move the thread's TLS base, so
  `threadlocal` is worker-local here, not task-local, and `Io.Scoped` is what a
  per-task value uses. The compiler's per-thread ids and a logger's context are
  the intended movers.
- `std.Io.Supervisor` is a group of tasks with OTP's restart policy. Children
  are added with `add` - the function they run and its arguments, which are
  copied - before the first `run`. `one_for_one` restarts the child that
  failed, `one_for_all` cancels and restarts every child, and `rest_for_one`
  the child that failed and the ones added after it. A child that returns
  normally is not restarted, which is OTP's `transient` and the only restart
  type in this cut, and a panic stays fatal. More restarts than `intensity`
  within `period` - three in five seconds by default - cancel every child and
  return `error.RestartIntensityExceeded`, with the error it gave up on in
  `lastChildError`. Canceling `run` cancels the children and returns
  `error.Canceled`.
- `Io.Queue` takes its overflow policy at `initWithOptions`: `.block`, the
  default and the behaviour a queue always had; `.drop`, where a put that finds
  the queue full drops what does not fit, counts it, and never waits, for a
  mailbox that keeps the newest sample; and `.drop_and_resync`, which does the
  same and makes the next `get` return `error.Resync` once, for a consumer that
  must notice a drop rather than act on stale data. `droppedElements` counts
  what was dropped. The `get` error set has `error.Resync` in it, which only a
  `drop_and_resync` queue returns.

On Linux, Threadz runs on io_uring, with a ring per worker: sockets, DNS,
processes, timers, futexes, and file reads are ring operations. The file
operations io_uring can only finish on a kernel thread of its own, such as
writes to a file, `statx`, and creating a file, are made on the worker instead,
because the trip to that thread and back costs more than the call.

A task that drives io_uring itself, such as a server's event loop, can borrow
its worker's ring instead of making one of its own. `acquireRing` lends it to a
task pinned to that worker. The task queues SQEs on the ring with
`ring_owner_bit` set in their `user_data`, and `Ring.waitCqes` parks it until
one of its completions arrives. The worker handles every other completion as
before and runs its other tasks while the loop is parked. `fromIo` finds the
Threadz behind an `Io`, and `workerLimit` says how many workers a program may
pin tasks to.

[zix](https://github.com/mattneel/zix)'s io_uring servers for HTTP/1 and
WebSocket, HTTP/2, gRPC and HTTP/3 run on Threadz this way, each loop a task
pinned to its own worker, and keep their multishot receives, buffer rings and
batched submissions. On a 32-thread laptop, across the 45 load tiers of their
HttpArena profiles, they served at a geometric mean of 1.01 times the
throughput of the same servers on threads of their own, with single tiers from
0.83 to 1.23 times, inside the laptop's own run-to-run spread.

`zig build test-threadz` runs the `Io` tests on Threadz, and the test runner's
`--io=threadz` runs any test on it. The design, and the steps still to come
(the kqueue and IOCP cores, arenas and supervisors, the task dump and the
scheduler gauges, and the `async`/`await` keywords), are in
[the Threadz proposal](https://github.com/mattneel/zigpp/blob/master/doc/proposals/threadz.md).

## AI in the toolchain

Upstream Zig bans LLMs from issues, patches, and bug tracker comments. Zig++
welcomes them, and it is putting AI on both sides of compilation: in the
programs it builds and in the build itself; see
[AI Policy and Governance](ai-policy.md).
