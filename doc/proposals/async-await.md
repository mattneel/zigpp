# The `async` and `await` keywords

Status: **proposal, for the BDFL's review.** Nothing in this document is implemented; the branch it
lands on changes `doc/` only.

The BDFL's direction, 2026-09-25: *"async/await keywords as sugar over Io futures; structured task
groups that join before the frame exits (detached spawns take owned data)"*. Threadz's plan names
this work as its last step, "**Keywords.** In their own proposal" (`doc/proposals/threadz.md:117`),
after step 7, Observability.

This document answers seven questions the work asks for: the syntax of spawn, concurrent spawn,
await, cancel and a group form; the lowering to `std.Io`; the structured rule and what enforces it
now; cancellation with `defer`/`errdefer`; the behaviour on `Io.Threaded` and `Io.Threadz`;
migration and upstream compatibility; and a staged plan with acceptance per stage.

Every claim about this tree is cited by `file:line` at `master` `753de4bb58`, or measured with a
compiler of the same vintage (`0.17.0-dev.2480+zigpp.e95c60250`) against this tree's `lib/`; §11
lists the commands and the compiler flags. Claims about `blocking`, task names and the Threadz
scheduler cite branch `threadz-budget-watchdog` at `40dee169b6`, and say so where they appear.
Anything derived rather than read or measured is marked *[inferred]*.

## 0. The decisions

| # | Question | Decision |
| --- | --- | --- |
| 1 | Syntax | Five keywords, recognised only in expression position: `async(io) f(x)` (spawn, `Io.Future(R)`), `concurrent(io) f(x)` (spawn with real concurrency, `ConcurrentError!Io.Future(R)`), `async(io) f(x);` as a statement (a task of the frame's group, no handle), `detach(io) f(x);` (detached), and `await a` / `cancel a` (consume a task binding). No group *block*: the frame is the group. §2 |
| 2 | Where the `Io` comes from | An explicit operand at every spawn, always in parentheses. `await`/`cancel` take no operand: the compiler stores the spawn's `Io` in a generated slot next to the future, so the two cannot disagree. Alternatives (a name in scope; a block binding; an ambient `io`) are weighed in §2.3. |
| 3 | Keywords and the tokenizer | `async`, `await`, `cancel`, `concurrent` and `detach` are **contextual**: the tokenizer is unchanged (`lib/std/zig/tokenizer.zig:12-60`), the parser recognises the forms by one token of lookahead, and no program that compiles today changes meaning (measured, §2.2). `suspend`, `resume` and `nosuspend` keep exactly their present meaning: parsed, then rejected by Sema (§4.5). |
| 4 | Lowering | `io.async(f, .{x})`, `io.concurrent(f, .{x})`, `a.await(io_slot)`, `a.cancel(io_slot)`, `g.async(io, f, .{x})`, `io.detach(f, .{x})`. Every form's Zig expansion is given in §3.1; the ZIR is the ZIR of that expansion (§3.2). |
| 5 | Structured rule | A task never outlives the frame that spawned it. The compiler emits the disposal: a bound future is joined at the end of the block that binds it (`defer`), cancelled on error exits (`errdefer`); a spawn statement joins the frame's `Io.Group`, which is joined at the frame's exits and cancelled on error exits. §4.1 |
| 6 | Checked now, checked later | Now: the spawn grammar, the callee's return type against the form, the affinity of task bindings (no copies, no escapes, no other uses), the fact that `detach` needs a guarantee the compiler cannot yet give. Later (borrow checker): consumers on every path, futures as ordinary values, and the `owned` frame rule for `detach`'s arguments. §4.2-§4.4 |
| 7 | Cancellation | `cancel` requests cancellation and joins; it never unwinds a task (already true of `std.Io`, `doc/proposals/threadz.md:70`). Cancellation is delivered as `error.Canceled` from the *task's* next cancelation point, and arrives in the joiner as the task's result. Early returns cancel before they join, in that order (`errdefer` before `defer`, measured). §5 |
| 8 | `Threaded` and `Threadz` | No semantic difference: every form is defined by the `std.Io` vtable, and the same lowering passes on `--io=threaded` and `--io=threadz` (§6, §11.1). `async` may run the call inline on both; `concurrent` never inlines and fails with `error.ConcurrencyUnavailable` when it cannot place the task. |
| 9 | Migration | Additive. The 53 `X.async(io, …)`, 8 `io.async(…)`, 98 `.await(` and 87 `.cancel(` sites in `src/` and `lib/std/` keep compiling untouched; `Io.Group` and `Select` stay for groups whose lifetime is data, not a frame. Upstream Zig code compiles unchanged; the sugar is Zig++-only syntax, so a program that uses it does not build upstream. §7 |
| 10 | std additions the forms need | `Io.detach` and a vtable entry for it (stage 4); the task-name plumbing already on `threadz-budget-watchdog` (`name` on `async`/`concurrent`/`groupAsync`/`groupConcurrent`/`blocking`, `spawnedName`), which the sugar satisfies by naming its generated wrappers after the program's own spelling (§3.5). Nothing else. |
| — | Not decided here, still open | `Future` as an ordinary value (§9.1-§9.3), spawn options (affinity, stack size) in the syntax (§9.4), a timeout on `await` (§9.5), and how far the spawned callee's shape may reach (§9.6). |

## 1. What the tree already has

**The keywords are gone.** Upstream removed `async` and `await` in `40d11cc25a`, "remove `async` and
`await` keywords", whose message says "it is settled that there will not be `async`/`await` keywords
in the language". The removal touched the tokenizer, the AST, AstGen, the ZIR, the parser, Sema and
`BuiltinFn`. What is left in the tokenizer's keyword table are `nosuspend`, `resume` and `suspend`
(`lib/std/zig/tokenizer.zig:39,46,50`, tags at `:164,171,175`); `async` and `await` are ordinary
identifiers, which §2 depends on and §11.2 measures.

The three leftovers parse and lower, and stop at Sema:

* `nosuspend EXPR` and `suspend EXPR` are expressions (`lib/std/zig/Parse.zig:1918-1930`), and
  `nosuspend`/`suspend` statements are statements (`lib/std/zig/Parse.zig:1003-1018`);
* AstGen lowers them (`lib/std/zig/AstGen.zig:1115-1117`; `nosuspendExpr` at `:1211-1229`,
  `suspendExpr` at `:1230-1261`, which emits a `suspend_block`, `lib/std/zig/Zir.zig:337`);
* Sema refuses `suspend_block` outright (`src/Sema.zig:5335-5339` →
  `failWithUseOfAsync`, `src/Sema.zig:2527-2534`: *"async has not been implemented in the
  self-hosted compiler yet"*).

This proposal does not give those three keywords new meaning, and does not re-open upstream's
decision about *async functions*: it adds sugar over `std.Io`'s existing futures, which are values
of a struct type, not functions with a colour.

**The primitives the sugar must land on.**

| `std.Io` | Where | What matters here |
| --- | --- | --- |
| `Future(Result)` | `lib/std/Io.zig:1299-1337` | `{ any_future: ?*AnyFuture, result: Result }`. `await` and `cancel` are idempotent per object and not threadsafe; both return `Result`, and `cancel` is "equivalent to `await` but places a cancelation request" (`:1305-1310`). |
| `Io.async(io, f, args)` | `lib/std/Io.zig:2529-2554` | Returns `Future(R)` with no error. The call *may run before `async` returns* (`:2531-2534`). `Result` is `@typeInfo(@TypeOf(f)).@"fn".return_type.?` — the callee's return type, error union and all. |
| `Io.concurrent(io, f, args)` | `lib/std/Io.zig:2568-2593`; error at `:2555-2566` | Returns `ConcurrentError!Future(R)`; `ConcurrentError` is `error{ConcurrencyUnavailable}`. |
| `Group` | `lib/std/Io.zig:1341-1438` | `init`, `async`, `concurrent`, `await`, `cancel`; that is the whole API. `Group.async` requires a callee whose return type is coercible to `Cancelable!void` (`:1366-1375`; `Cancelable = error{Canceled}`, `:822-825`) and drops `error.Canceled` by design. `Group.await` returns `Cancelable!void` and still guarantees every member ran (`:1411-1428`); `Group.cancel` requests cancellation on all members and guarantees they ran (`:1430-1438`). There is **no** way to adopt an existing `Future` into a group. |
| `recancel`, `CancelProtection`, `checkCancel` | `lib/std/Io.zig:1442-1491` | `recancel` re-arms a cancelation request after a cancelation point returned `error.Canceled`; `swapCancelProtection(.blocked)` suspends cancelation delivery; `checkCancel` is a bare cancelation point. |
| The vtable contract | `lib/std/Io.zig:58-160` | The context a spawn is handed is **copied** by the implementation and then passed to `start` (`:73-75`, `:119-127`); `Threaded` does exactly that (`lib/std/Io/Threaded.zig:521`). |
| The single-threaded stubs | `lib/std/Io.zig:2845-2864` | `noAsync` runs the call inline and returns `null`; `failingConcurrent` returns `error.ConcurrencyUnavailable`. |

Two consequences shape the whole design:

1. **`std.Io` already has the semantics the keywords need**, including the two hard parts —
   *cancellation is a request delivered at the task's next cancelation point*, and *`Group.cancel`
   joins*. So the keywords are lowering, not runtime work. The one primitive missing is a detached
   spawn (§4.4).
2. **A future that is never awaited or cancelled is a leak**, not just a hazard: `Threaded`'s
   `Future` is heap-allocated at spawn (`lib/std/Io/Threaded.zig:668-700`) and freed only by `await`
   or `cancel` (`:2722`, `:2753`). A language that makes spawning easy owes an answer for disposal,
   and the answer is the structured rule of §4.

`Io.Threadz`'s runtime is `Io.Uring` on Linux (`lib/std/Io.zig:33-43`); on branch
`threadz-budget-watchdog` it is `lib/std/Io/Threadz/scheduler.zig`, whose `async` falls back to
running the call inline exactly as `Threaded` does (`scheduler.zig:2100-2113`), whose
`SpawnOptions` carries `stack_size` and `affinity` (`:75-79`), and whose tasks are named through
the vtable's new `name` parameter (`40dee169b6`) — `spawnedName` derives a task's name from the
type name of an instantiation carrying the function (`lib/std/Io.zig:282-306` on that branch,
`Spawned` at `:313-318`).
That branch is where the keywords' task names come from (§3.5).

## 2. Syntax

### 2.1 The forms

```zig
fn fetch(io: Io, url: []const u8) ![]u8;
fn log(msg: []const u8) void;
fn slow(io: Io, n: u32) Io.Cancelable!u32;

fn handler(io: Io, urls: []const []const u8) !void {
    // Spawn and bind: the future's result is `![]u8`, exactly the callee's return type.
    const page = async(io) fetch(urls[0]);

    // A spawn statement: no handle, a member of this frame's group.
    async(io) log("started");

    // Many of them, concurrently, joined when the frame returns.
    for (urls) |u| async(io) log(u);

    // Consume a binding. `await` joins; `cancel` requests cancellation and then joins.
    // Each binding is consumed once: `await page;` a second time would be an error.
    const body = try await page;

    // Real concurrency, or a typed error.
    const slow_future = try concurrent(io) slow(io, 10);
    const n = try await slow_future;

    // A binding whose result is not wanted is still consumed explicitly:
    const spare = async(io) fetch(urls[1]);
    _ = cancel spare;                        // asks it to stop, then joins, discarding the result

    // Detached: no handle, no join, arguments must be `owned` (stage 4).
    detach(io) log("this outlives the frame; its arguments must be owned");
}
```

Rules that fall out of the forms:

* **`async` and `concurrent` differ in what they promise.** `async` promises the result will be
  available after `await`; the call may have run before `async` returned. `concurrent` promises a
  unit of concurrency, and returns `error.ConcurrencyUnavailable` when the implementation cannot
  place one. That is exactly the difference between `Io.async` and `Io.concurrent`
  (`lib/std/Io.zig:2531-2534`, `lib/std/Io.zig:2568-2575`), which is why there are two keywords
  rather than one with a modifier.
* **A spawn statement's callee must be coercible to `Io.Cancelable!void`** — `void`, or an error
  union whose error set is a subset of `{error.Canceled}`. This is `Group.async`'s own rule
  (`lib/std/Io.zig:1366-1375`, measured in §11.4), and it is the honest one: a spawn statement has
  nowhere to put a result or an error, so the type system refuses the spawn instead of a wrapper
  dropping them. A callee that returns anything else must be bound: `const f = async(io) f(x);`.
* **`await a` and `cancel a` yield the task's `Result`**, as `Future.await`/`Future.cancel` do. A
  discarded value is discarded the way Zig discards any value: `_ = cancel a;`. The sugar does not
  get an exemption from that rule.
* **A task binding may be consumed at most once on a path, and a second consume is a compile error
  where the compiler can see it** (§4.2). Where it cannot, the generated code carries a flag and the
  second consume is `unreachable` — illegal behavior, checked in Debug and ReleaseSafe, exactly the
  shape Zig gives every such guarantee.
* **`async`/`concurrent`/`detach` are only legal inside a function body** (a frame that can carry
  the join). A spawn in a `comptime` block or in a container-level declaration's initialiser is an
  error: *"a task needs a frame to join it; spawn inside a function body"*. The frame is the unit
  the join belongs to (§4.1), so a spawn outside a frame has no join and is refused.

### 2.2 Contextual keywords: one token of lookahead, no tokenizer change

The five words stay identifiers. They are recognised as keywords only where the form they introduce
is what the program must have meant, and that is decided with **one token of lookahead**:

* `async`/`concurrent`/`detach` followed by `(`: the parenthesised expression is parsed as an
  ordinary call's arguments; if the token *after* it can begin a primary expression (an identifier
  or `@`), the parsed call is the **io operand** and what follows is the spawned call. Otherwise it
  was an ordinary call, and nothing changes.
* `await`/`cancel` not followed by `(`: the following expression is the operand. (Followed by `(`,
  they are ordinary calls, which is what keeps `f.await(io)` and every function named `cancel`
  working.)
* `detach` uses the first rule.

Why that is safe: a complete expression can never be followed by an identifier or `@` in Zig's
grammar, so every form above is a syntax error today, and no program that compiles today can change
meaning. Measured with this tree's compiler (§11.3):

```text
kw1.zig:4:19: error: expected ';' after statement       _ = async(io) f(1);
kw2.zig:3:15: error: expected ';' after statement       _ = async f(1);
kwsyn.zig:5:15: error: expected ';' after statement     _ = await x;
kwsyn.zig:5:16: error: expected ';' after statement     _ = cancel x;
kwsyn.zig:5:24: error: expected ';' after statement     _ = concurrent(io) f(1);
kwsyn.zig:5:20: error: expected ';' after statement     _ = detach(io) f(1);
```

and the identifiers that would be shadowed keep working (§11.2): `fn async(x: u8) u8`, `async(1)`,
`fn await(x: u8) u8`, `await(a)`, `p orelse 0`, `x catch 0`, `f.await(io)`, `Future.cancel(io)`.

Alternatives, and why not:

* **Reserve the words.** Making `async`, `await` and `cancel` keywords again would break exactly the
  code Zig++ has most of: `Future.await` (`lib/std/Io.zig:1322`), `Group.await` (`:1411`),
  `Select.await` (`:1602`), `Future.cancel` (`:1314`), `Group.cancel` (`:1430`), `Batch.cancel`
  (`:712`), plus 98 `.await(` and 87 `.cancel(` call sites in `src/` and `lib/std/` (§7). A keyword
  is a permanent tax on every program that ever named a variable `async`; one token of lookahead is
  a one-time cost in `Parse.zig`.
* **A sigil or a different spelling** (`spawn`, `go`, `@spawn`). `spawn` is a fair alternative for
  the first form, but it hides the fact that this is `Io.async`'s older name — and `async`/`await`
  are the words the BDFL asked for, the words the plan names (`doc/proposals/threadz.md:12,117`),
  and the words a reader of any other language already knows.

The parser's work is bounded and local: one `p.tokenTag(p.tok_i + 1)` for the two identifier forms,
and one token after the parsed operand for the three parenthesised forms. There is no scan to a
matching parenthesis and no backtracking. *[inferred: the token-level feasibility follows from
`Parse.zig`'s structure (`lib/std/zig/Parse.zig:1905-1940` for prefix expressions, `:995-1018` for
statements); the exact edit is stage 1's job.]*

### 2.3 Where the `Io` comes from

**Decision: an explicit operand at every spawn, `async(io) f(x)`, and no operand at the consume.**
The parenthesised operand is the `Io` value the spawn is made with, evaluated there, once, in
source order — and the compiler keeps it in a slot of its own, so `await a` uses the same `Io` the
spawn used. The user writes the runtime choice once, at the only place where it is chosen; the
consume cannot name a different one, because it cannot name one at all.

The alternatives, weighed:

* **A name in scope.** `async f(x)` with the compiler resolving an `io` declaration by name (or by
  type `Io`) in the nearest scope. Rejected: Zig has no implicit identifier resolution, and this
  would be the first; it breaks the moment the value is spelled `self.io`, `gpa_io` or `test_io`
  (the compiler's own code has all three: `src/Zcu/PerThread.zig:156`, `src/Compilation.zig:4390`,
  `lib/std/testing.zig:24`); and it hides the runtime choice at the one point where the program
  makes it.
* **A block binding.** `tasks (io) { … }`, with the block supplying the `Io` for everything inside,
  as well as the join boundary. Rejected as *the mechanism*, for two reasons. It adds a construct
  whose only job is to name a value the spawn site can name; and it makes the join boundary a
  syntactic block, when the frame already is the boundary the structured rule needs (§4.1) — a
  `tasks` block would need a rule for spawns *outside* it anyway. It remains a plausible future
  sugar for "many spawns, one `Io`, one join", and nothing in §3 or §4 would change if it were
  added (§9.7).
* **An ambient `io`** (`std.io`, `builtin.io`). Rejected: a global runtime contradicts Threadz's
  "every thread deliberate" (`doc/proposals/threadz.md:21-23`), cannot express two instances (the
  test runner builds one per run, `lib/std/testing.zig:23-46`), and makes single-threaded and
  evented builds a compile-time switch inside the language rather than a choice at the call.
* **`Future` carrying its `Io`.** Adding an `io: Io` field to `Future` (`lib/std/Io.zig:1299-1337`)
  would let `await` be spelled with no operand and no compiler slot. Rejected *for this proposal*:
  it changes a std type and its size for a fact the compiler already has at the spawn site, and it
  would still leave `Io.async`'s callers passing an `io` that must match. It is the right change
  the day futures become first-class values (§9.1).

The consume's `Io` is a compiler-managed slot: one per spawn site, assigned the operand's value
where the spawn appears, read by every `await`/`cancel` of that site. So `async(getIo()) f(1)`
evaluates `getIo()` once, at the spawn, in source order, and awaits it with that same value.

### 2.4 The spawned call

The operand of `async`/`concurrent`/`detach` is a **call expression**: the callee and the arguments
are the runtime's `function` and `args` (`lib/std/Io.zig:2529-2554`). Two restrictions on its shape,
both from what `Io.async` can accept as a function value (measured in §11.4):

1. **The callee must name a declaration**: an identifier (`f`), a field of a namespace
   (`ns.f`, `Type.f`), or a method (`a.b`). A callee of any other shape — `(expr).f`, `f(x).g`,
   a struct field of function-pointer type — is an error: *"the callee of `async` must name a
   function; bind the receiver or the function first"*, with the workaround in the message.
2. **A comptime argument is allowed; a runtime function pointer is not.** `async(io) gpa.alloc(u8, 16)`
   works: the comptime arguments are fixed inside the compiler's wrapper (§3.4). `async(io) ptr(x)`
   where `ptr` is a runtime `*const fn` cannot work, because Zig has no function value whose type
   `@TypeOf` can read at runtime — measured: *"cannot load comptime-only type 'fn (u32) u32'"*. The
   workaround, which the error message names, is a wrapper that takes the pointer as an argument
   (§3.4, case 5).

A method call binds its receiver: `async(io) a.b(x)` spawns `b` with `.{a, x}`. The receiver is
evaluated once, in the spawning task, and passed as the first argument, which is what `a.b(x)`
means today.

## 3. Lowering

### 3.1 Source to Zig

Each form has one expansion, and the expansion is Zig that a programmer could write today. This
section is the specification; §11.1 is the compiled, running version of it.

**`const a = async(io) f(x);`** — a task binding:

```zig
const a_io = io;                                  // the operand, evaluated here, once
var a = a_io.async(f, .{x});                      // Future(R); may already have run
var a_live = true;
defer {
    if (a_live) _ = a.await(a_io);                // normal exits join
}
errdefer {
    if (a_live) _ = a.cancel(a_io);               // error exits cancel, and `cancel` joins
}
```

`await a` is `a.await(a_io)` followed by `a_live = false`; `cancel a` is `a.cancel(a_io)` followed
by `a_live = false`. The slot's type is never written by the compiler — `var a = a_io.async(…)` is
an inferred allocation, as in the ZIR of §3.2 — so the sugar names no `std` type and depends on no
version of `Future`'s layout.

**`async(io) f(x);`** — a spawn statement, in a frame group:

```zig
// Once, in the frame's body, if the frame has any spawn statement:
var g = std.Io.Group.init;
defer g.await(io) catch |err| switch (err) {
    error.Canceled => io.recancel(),
};
errdefer g.cancel(io);

// Each statement, in place:
g.async(io, f, .{x});
```

**`concurrent(io) f(x)`** — `<operand>.concurrent(f, .{x})`, and the surrounding form's handling of the
error union is the program's: `const a = try concurrent(io) f(x);`.

**`cancel a`** — `a.cancel(a_io)`, evaluated where the form appears; **`await a`** — `a.await(a_io)`.

**`detach(io) f(x);`** — `io.detach(f, .{x})`, a std entry that does not exist yet; see §4.4 for why
dropping an unjoined `io.async` future in its place would be wrong on today's implementations.

Details that are part of the specification, not of the implementation:

* **The join's error is re-armed, not swallowed.** `Group.await` is a cancelation point that returns
  `error.Canceled` *after* the group finished (`lib/std/Io.zig:1411-1428`). A `defer` cannot return
  an error, so the generated code calls `recancel` in the error arm (`lib/std/Io.zig:1442-1452`):
  the frame's own cancelation request stays outstanding and is delivered at its next cancelation
  point, which is the semantics of the surrounding function, not of the sugar.
* **The generated order is the cancel-before-join order.** `errdefer`s run before `defer`s (LIFO;
  measured in §11.6), so on an error exit the generated cancels run first, clear their flags, and
  the joins that follow do nothing — and `Group.cancel` has already guaranteed the members ran
  (`lib/std/Io.zig:1430-1438`), so the frame's `g.await(io)` returns immediately.
* **The frame's group is emitted only for frames that have spawn statements**, found by a syntactic
  pre-scan of the function body. That is not an optimisation: the group's join needs an `Io` value
  and a frame with no spawns has no operand to take one from.
* **A group per frame, not per block.** A spawn statement inside a loop joins the frame's group, so
  the loop's tasks run concurrently and are joined when the frame returns. This is the whole reason
  a spawn statement exists separately from a binding: a binding is one live future per spawn site, a
  spawn statement is any number of tasks. `for (urls) |u| async(io) log(u);` is N tasks;
  `for (urls) |u| { const f = async(io) fetch(u); }` is one per iteration, joined at the iteration's
  end.
* **The compiler reaches `std` the way `@import("std")` does.** The generated calls name
  `std.Io.Group.init`, `Group.async`, `Io.async` and the rest through the compilation's own `std`
  module (`src/Zcu/PerThread.zig:2455`, the module `@import("std")` resolves to), so a file that
  never imports `std` can still spawn — and a `std` that is not this compiler's own is not a thing
  the keywords can be pointed at.
* **Inside `comptime`-known-false or `unreachable` tails**, the generated code is what Zig's own
  reachability analysis makes of it; nothing special is specified.

### 3.2 ZIR

The sugar adds no ZIR tags. What the keywords emit is the ZIR of §3.1, and §11.7 prints that ZIR
with Zig++'s own `dump-zir`. The shape, for `f.await(io)` and for a `defer`/`errdefer` pair:

```text
%42 = alloc_mut(%36)                        # the bound future's frame slot, address-stable
%112 = field_call(.auto, %57, "async", […])   # io.async(f, .{3, 4})
%122 = store_node(%42, %112)                   # the binding
%126 = field_call(.auto, %42, "await", […])   # f.await(io)
%135 = defer({ … field_call(… "await" …) })   # the disposal, emitted at the block's exit
%136 = defer({ … field_call(… "await" …) })   # the group's join, emitted at the frame's exit
%159 = defer({ … field_call(… "cancel" …) })  # the same bodies on the error exit
```

Two facts from that dump are worth having on the record, because the design leans on them:

* `defer` bodies are not instructions in the middle of the body; AstGen emits them at the exits
  (`lib/std/zig/AstGen.zig:2993-3016`, `:7983-8030`), and `errdefer` bodies only on error exits
  (`genDefers(…, .normal_and_error)` at `lib/std/zig/AstGen.zig:7998`, `:8028`). So "the compiler
  emits a `defer`/`errdefer` pair" is a statement about the ZIR the sugar's expansion produces, not
  a new mechanism: it is the same ZIR a programmer gets from writing those two statements.
* The bound future lives in a frame slot (`alloc_mut`), which is what makes `&f.result` — the
  pointer `Io.async` wrote into the implementation at spawn (`lib/std/Io.zig:2544-2551`) — stable
  for the future's whole life. A task binding is never a temporary.

### 3.3 Types

| Form | Type | Note |
| --- | --- | --- |
| `async(io) f(x)` | `Io.Future(R)` | `R` is the callee's declared return type *including* its error union: `async(io) gpa.alloc(u8, 16)` has type `Io.Future(Allocator.Error![]u8)`, and `try await` on it is the natural spelling (measured, §11.4 case 6). |
| `concurrent(io) f(x)` | `Io.ConcurrentError!Io.Future(R)` | `error.ConcurrencyUnavailable` is part of the type, so the program cannot forget it. |
| `async(io) f(x);` (statement) | the spawn has no value | requires `R` coercible to `Io.Cancelable!void`. |
| `await a`, `cancel a` | `R`, the callee's return type, which is the future's `Result` | `Future.await` is idempotent on one object (`lib/std/Io.zig:1322-1336`), but a *binding* holds one handle and may be consumed once (§4.2). |
| `detach(io) f(x);` | no value | requires the callee's parameters to be `owned`, and `R` to be `Cancelable!void`-coercible or ignored by `Io.detach`'s own contract (§4.4). |

`await` and `cancel` on a binding whose `Result` is `void` yield `void`, so `_ =` is not needed
there; on any other type, Zig's ordinary unused-value rule applies.

### 3.4 Callees `Io.async` cannot take

`Io.async`'s signature takes `function: anytype` and `args: std.meta.ArgsTuple(@TypeOf(function))`
(`lib/std/Io.zig:2529-2532`). `ArgsTuple` refuses a function with an `anytype` parameter
(`lib/std/meta.zig:773`), which is what every `comptime` parameter looks like in `@TypeOf`:

```text
error: cannot create ArgsTuple for function with an 'anytype' parameter
   callee.zig:33:22: note: generic function instantiated here
       var f2 = io.async(gen, .{ u32, 5 });
```

So the sugar cannot always hand the callee over as-is. When the callee has comptime parameters, or
is a runtime function pointer, the compiler emits a **wrapper**: a nested function whose parameters
are the runtime arguments, with the comptime arguments fixed inside its body. Measured
(§11.4 cases 4-6), the wrapper is exactly what a programmer writes by hand:

```zig
const Wrap = struct {
    fn alloc(a: std.mem.Allocator, n: usize) std.mem.Allocator.Error![]u8 {
        return a.alloc(u8, n);                 // u8 was a comptime argument of the source call
    }
    fn callPtr(p: *const fn (u32) u32, x: u32) u32 {
        return p(x);                           // a runtime function pointer rides in the arguments
    }
};

var f = io.async(Wrap.alloc, .{ std.testing.allocator, 4 });
```

Three properties of the wrapper make it the right answer rather than a workaround:

* **Arguments are still evaluated in the spawning task**, because they are the wrapper's parameters,
  passed through the same copied context every spawn uses.
* **Comptime arguments are baked in, not passed.** They cannot have runtime side effects, and
  re-analysing them inside the wrapper is a compile-time cost the compiler already pays for every
  instantiation.
* **The wrapper's return type is the callee's real return type**, so `Future(R)` and the error
  union survive intact.

The direct form — no wrapper — is used whenever `Io.async` can take the callee, which is every
non-generic function and every method whose receiver is not itself a call (§2.4). That keeps the
common case's expansion byte-for-byte the expansion a programmer writes.

### 3.5 Task names

`threadz-budget-watchdog` gives every spawn a name: the vtable's `async`, `concurrent`, `groupAsync`
and `groupConcurrent` take `name: [:0]const u8` (`lib/std/Io.zig:76-78`, `:90-91`, `:150-152`, `:163-165` on `40dee169b6`),
`spawnedName` derives it from the type name of an instantiation carrying the function
(`Io.zig:282-306`), and Threadz's scheduler uses it in its logs and its dump. A `#name;` that reads
`(function 'start')` — the compiler's thunk — would defeat the point. The sugar therefore names
**the wrapper after the program's own spelling** of the callee: the wrapper generated for
`async(io) gpa.alloc(u8, 16)` is a function declaration named `alloc`, so `spawnedName` prints
`alloc`; the direct form needs no help, since the callee's own name is already right. A method
spawn is named after the method (`async(io) self.reap()` names the task `reap`), which is what a
reader of the source can find. No std API change is needed for this; the compiler controls the
wrapper's name. *[inferred]*: the branch's `spawnedName` reads the name out of `@typeName`
(`lib/std/Io.zig:282-306` on `40dee169b6`), and no compiler here emits such a wrapper yet, so
stage 3's acceptance test is what turns this from a plan into a measurement.

## 4. The structured rule

> A task never outlives the frame that spawned it. Leaving the frame joins it — or cancels it, on an
> error exit — and leaving the frame is the *only* thing the program can do about a task it did not
> name.

### 4.1 What the compiler inserts

| Spawn | Disposal | Where |
| --- | --- | --- |
| `const a = async(io) f(x);` | `defer { if (a_live) _ = a.await(a_io); }` and `errdefer { if (a_live) _ = a.cancel(a_io); }` | the end of the block that declares `a`, i.e. `defer`'s own scope |
| `async(io) f(x);` | the frame's group: `defer g.await(io) catch …recancel…;` and `errdefer g.cancel(io);` | the frame's body block, so a spawn inside a loop accumulates in the frame's group |
| `detach(io) f(x);` | none: the task belongs to the runtime | — |

The frame is the join boundary because the frame is what holds the data a task may be reading. A
task's arguments are copied into its context (`lib/std/Io.zig:73-75`, `lib/std/Io/Threaded.zig:521`),
but a *pointer* argument points into the spawner's frame, and the only reason that is sound is that
the task cannot outlive the frame that holds the pointee. That is the frame rule of the borrow
checker (`doc/proposals/borrow-checker.md:302-330`) applied to a task instead of a pointer: the
frame rule says a pointer into a frame object must not be reachable at any exit from anything that
outlives the frame; a task holding it is exactly such a thing; the join removes it.

The block, not the frame, for a *binding*, because a name has a block's lifetime: `defer` is the
mechanism Zig already has for "this ends when the block does", and the disposal is a `defer` — its
body is emitted at the block's exits, so the join happens on `break`, `continue`, `return` and
`try` alike (`lib/std/zig/AstGen.zig:2993-3016`). A binding declared in a loop body is therefore
joined each iteration, which is the only sane meaning of one slot per spawn site.

For a *spawn statement* the frame is the boundary, and the group is what carries the many. A group
per frame also gives the frame's tasks one cancellation domain: a `SIGQUIT` dump, a supervisor, or
`Group.cancel` from the runtime reaches them as a set, which is the shape Threadz's structure
decision already asks for ("Every task belongs to exactly one `Io.Group`", `doc/proposals/threadz.md:69`).

### 4.2 What the compiler checks now, without the borrow checker

1. **Grammar.** The five forms are recognised as in §2.2, and only in expression position.
2. **Frame.** `async`/`concurrent`/`detach` inside a `comptime` block or outside a function body:
   *"a task needs a frame to join it; spawn inside a function body"*.
3. **Callee and type.** §2.4's callee shape and §3.3's return-type rules, with the errors of §3.4
   translated into a note that names the wrapper the compiler could not build.
4. **Affinity.** A task binding may appear only as the operand of `await` or `cancel`. Any other use
   — `_ = a;`, `a`, `a.field`, passing it to a function, storing it into an aggregate, returning it,
   comparing it — is *"a spawned task is not a value; `await` it or `cancel` it"*. This is a lexical
   check on a declaration, the same shape as `GenZir`'s tracking of the enclosing `suspend` and
   `nosuspend` nodes (`lib/std/zig/AstGen.zig:1211-1261`), and it is what makes the generated slot
   and flag sound: nothing else can hold a copy of the handle.
5. **Consumption.** `await a` requires `a` to be a task binding that is live at that point. Two
   consumes in one straight-line statement sequence are a compile error (*"task 'a' is already
   consumed"*). Consumes on different branches are allowed, and the generated flag makes the
   unreachable arm `unreachable` (§2.1).
6. **`detach` is refused outside a borrow-checked module**, and says so: *"`detach` needs the borrow
   checker's `owned` rule; compile this module with `-fborrow-check`"* (§4.4).
7. **`await`/`cancel` do not accept hand-written futures.** `await x` where `x` is not a task
   binding — a `Future` from `Io.async`, a `Select` member, an element of `pool.task_futures`
   (`src/Zcu.zig:5416`) — is an error that points at the method: *"`await` works on a task spawned
   by the keywords; for a future value write `x.await(io)`"*. This is the honest boundary of the
   first cut: the sugar's bookkeeping exists only for handles the compiler created.

### 4.3 What waits for the borrow checker

* **Path-sensitive consumption.** Today the flag makes a maybe-consumed binding safe at runtime and
  a same-path double consume a compile error in the cases the compiler can see. The check that a
  binding is consumed on *every* path where it can be live is the borrow checker's forward dataflow
  over AIR with per-root state and joins at merges (`doc/proposals/borrow-checker.md:362-388`).
  It buys better diagnostics, not safety, which is why it is not in the first cut.
* **Futures as ordinary values.** Storing a future in a slice, passing it to a helper, returning it
  from a function: all rejected by §4.2 rule 4 today. They become legal when `Future.await` and
  `Future.cancel` carry the `owned` qualifier the borrow checker reads
  (`doc/proposals/borrow-checker.md:446-527`) and `Future` is annotated `borrowed` where it hands
  out pointers, so a helper can be *checked* to consume its parameter. That is the stage that makes
  `await` accept any `Future` value, and it is also the stage that needs the `io` question of §9.1
  answered.
* **Races are not modelled, and this proposal does not model them either.** A task spawned with a
  `*T` argument runs concurrently with the frame that still holds the `*T`, and no existing check
  prevents either party from writing it. The borrow checker lists threads and atomics among its
  non-goals (`doc/proposals/borrow-checker.md:1599-1610`), and Threadz's structure row says data
  crossing a scope is `owned` or copied (`doc/proposals/threadz.md:69`) — the *rule* is stated, and
  the check is future work (§9.2). What this proposal guarantees is the lifetime, not the aliasing.

### 4.4 Detached spawns

`detach(io) f(x);` is the one form whose task *is* allowed to outlive the frame. Its data rule is the
BDFL's: **the arguments must be `owned`**. Concretely, for every argument:

* a value with no pointers is copied into the task's context, as for any spawn, and is fine;
* a pointer argument must be `owned` (the task takes the object, and releases it when it ends), or
  point at memory that is not this frame's (a `Global`, a `:static`/container-level object, or an
  object whose allocation outlives the frame and is not freed by it);
* the callee's return type must be coercible to `Io.Cancelable!void` (like a spawn statement) or
  `void`: there is no handle, so there is nowhere for a result to go.

That rule is exactly `owned`'s semantics in the borrow checker's proposal
(`doc/proposals/borrow-checker.md:446-527`) and exactly the frame rule's escape branch
(`:317-326`). Until the borrow checker runs, the compiler cannot check it — so the form is *refused*
outside a module compiled with `-fborrow-check` (`doc/proposals/borrow-checker.md:391-416`). A
compiler that accepts `detach` in an unchecked module would be promising something it cannot see;
refusing is the only honest option, and it makes the keyword's arrival a deliberate act (stage 4).

The lowering needs one std addition: **`Io.detach`** (and a `detach` vtable entry). Dropping the
future from an `io.async` in its place would be wrong on today's implementations: `Threaded`'s
future is heap-allocated and freed only by `await`/`cancel` (`lib/std/Io/Threaded.zig:668-700`,
`:2722`, `:2753`), so dropping it leaks the allocation and loses the task. A detached task must
belong to something the runtime owns — Threadz calls it the root group (`doc/proposals/threadz.md:69`),
and `Threaded` can implement it as a thread the pool owns. That is stage 4's work, with `Kqueue`,
`Dispatch` and the single-threaded stub following the same shape as the others.

### 4.5 The three leftover keywords

`suspend`, `resume` and `nosuspend` keep their present state: parsed, `suspend_block` lowered, Sema
refusing it (`src/Sema.zig:5335-5339`). This proposal gives them no new meaning, and the sugar does
not need them:

* there is no "suspend the current function" operation in the design: a task parks at `Io` calls
  (`doc/proposals/threadz.md:61,66`), and a *frame* is joined, not suspended;
* `nosuspend`'s purpose — "this code does not become an async function" — is a fact about async
  functions, which this language does not have: a spawn returns a `Future` value, and the spawning
  function's type is unchanged by it.

The stale message in `failWithUseOfAsync` ("async has not been implemented in the self-hosted
compiler yet") is about async *functions*; the day this proposal's stages land, the message should
say what `suspend` cannot mean here. That is a wording change for stage 1, not a semantic one.

## 5. Cancellation, `defer`, `errdefer`, and early returns

**Cancellation never unwinds a task.** It is a request, delivered as `error.Canceled` at the
requested task's next cancelation point (`lib/std/Io.zig:1305-1320`); `doc/proposals/threadz.md:70`
makes that a decision ("Never unwinds a task"), for the reason it gives: killing skips `defer`s.
`await` and `cancel` are the two ways to *join* a task after such a request, and the difference
between them is one line: `cancel` asks first.

| Situation | What the keywords do |
| --- | --- |
| `cancel a` in the middle of a block | `a.cancel(a_io)`: the task is asked to stop and the frame waits for it. The task's `error.Canceled`, if it returns one, comes back as the expression's value — measured in §11.5, where `expectError(error.Canceled, f.cancel(io))` holds. |
| The frame returns from a block with a live binding | The binding's `defer` joins it: the frame waits, the result is dropped. |
| The frame returns an error with a live binding | The binding's `errdefer` runs first (LIFO, measured §11.6), calls `cancel`, and joins. |
| The frame returns with tasks in its group | `errdefer g.cancel(io)` cancels the group and joins it; on a normal return `defer g.await(io)` joins. |
| The frame itself is cancelled while joining | `Group.await` propagates the request to the members, joins them, returns `error.Canceled` (`lib/std/Io.zig:1415-1427`), and the generated `recancel` re-arms the frame's own request so its enclosing function still sees it. |
| A `defer` in the frame returns from the function | Unaffected. The sugar's defers are ordinary defers at the positions §4.1 gives; they interleave with the program's own in the order the program wrote them. |
| The frame holds a future and calls `std.process.exit` | Nothing runs; this is not a language guarantee, it is the same fact as `defer`. |

**A task cannot be cancelled by a stranger.** There is no `cancel` on a name the program does not
hold: the only handle is the binding, and it cannot leave its block. Cancellation from outside a
frame comes from the runtime's own domains — a supervisor or the root group (§4.4) — not from this
syntax.

**The join is a cancelation point for the joiner.** That is `Group.await`'s and `Threaded.await`'s
own behaviour (`lib/std/Io.zig:1411-1428`; `lib/std/Io/Threaded.zig:2658-2724`, which forwards
the awaiter's request to the future and re-arms its own via `recancelInner`). So a frame waiting on
its children is itself interruptible, which is what makes nested cancellation work without any
unwinding.

## 6. `Io.Threaded` and `Io.Threadz`

The keywords add no implementation-specific semantics: every form is defined by the vtable and the
`std.Io` functions in §3.1, so the language cannot behave differently on the two runtimes. What
differs is what already differs for hand-written code, and the keywords inherit it exactly:

| | `Io.Threaded` | `Io.Threadz` (Linux: `Uring`, and `Threadz/scheduler.zig` on the branch) |
| --- | --- | --- |
| `async` may run the call inline | yes: `single_threaded`, allocation failure, or over `async_limit` (`lib/std/Io/Threaded.zig:2407-2436`) | yes: `concurrent` failure falls back to running it here (`scheduler.zig:2100-2113`) |
| `concurrent` inlines | never; returns `error.ConcurrencyUnavailable` (`lib/std/Io/Threaded.zig:2438-2454`) | never; `spawnFuture` propagates the failure (`scheduler.zig:2115-2127`) |
| What a spawn's arguments cost | copied into the future's allocation (`lib/std/Io/Threaded.zig:520-521`) | copied into the task's context, which is inside the task's allocation |
| A parked task holds | a worker, unless the call is handed to the pool | nothing: the worker takes the next task (`doc/proposals/threadz.md:66`) |
| Single-threaded builds | `noAsync` runs inline; `failingConcurrent` (`lib/std/Io.zig:2845-2864`) | `Threadz` is `void` where `fiber.zig` has no switch, so the same stubs apply (`lib/std/Io.zig:41-43`) |

Measured: the lowering file of §11.1 — the whole of §3.1, written by hand — passes all four of its
tests on `--io=threaded` and on `--io=threadz`, including the loop of statement spawns and the
cancel path; the only visible difference is the order of two concurrent prints, which is not
something the language specifies.

Requirements the keywords place on *both* implementations, beyond what they already do:

1. **A spawn's context is copied before the spawn returns.** Both already do it (`:521`,
   `scheduler.zig`'s `spawn`), and the sugar's binding slot relies on it: the compiler reuses the
   argument storage at the spawn site after the call.
2. **`Group.cancel` joins.** Already documented and implemented (`lib/std/Io.zig:1430-1438`;
   `Threaded.zig:2591-2615`), and the generated error-path disposal depends on it: it does not wait
   for anything after cancelling.
3. **Names.** The keywords pass the program's spelling (§3.5), which today's `Threaded` ignores and
   `threadz-budget-watchdog`'s Threadz reports. A dump that names `alloc` instead of `%5` is the
   difference between a usable and an unusable post-mortem; that is why §3.5 is a decision and not a
   detail.

## 7. Migration and upstream compatibility

**Nothing has to be rewritten.** The sugar is additive syntax over calls that already exist, and its
keywords are contextual, so:

* every `X.async(io, f, .{args})` (53 sites in `src/` and `lib/std/`), `io.async(f, .{args})` (8),
  `.await(` (98) and `.cancel(` (87) keeps compiling and keeps its meaning;
* `Future.await`, `Future.cancel`, `Group.await`, `Group.cancel`, `Batch.cancel`, `Select.await` and
  every declaration or local named `async`, `await`, `cancel`, `concurrent` or `detach` keep their
  names, because the tokenizer never learns the words (§2.2, measured §11.2);
* `Io.Group` and `Select` are not replaced. A group whose lifetime is *data* — a field of `Zcu`, a
  worker set that lives as long as a compilation — cannot be a frame-scoped group, and the compiler
  has several (`src/Zcu/PerThread.zig:156`, `src/Compilation.zig:4390-4640`). The keywords express
  the frame-scoped case; `std.Io.Group` remains the API for the other one.

**Migrating a site is a local, reviewable change.** The shape that migrates:

```zig
// before
var g: Io.Group = .init;
defer g.await(io) catch |err| { if (err != error.Canceled) return err; };
errdefer g.cancel(io);
g.async(io, workerUpdateFile, .{ comp, file });

// after
async(io) workerUpdateFile(comp, file);
```

and the shape that does not: groups held in a struct, groups whose members are added across several
functions, `Select`'s queue-of-results pattern (`lib/std/Io.zig:1499-1670`), and dynamic sets of
futures (`src/Zcu.zig:5416`, `pool.task_futures`). Those stay hand-written until stage 5 (§8).

**Upstream compatibility.** Zig++ tracks upstream continuously and merges it, never rebases onto it
(the book's Live at Head rule 9). The divergence this proposal creates is one-directional: Zig++
accepts five forms upstream rejects, and §2.2 measured that upstream programs that compile today
still mean the same thing here — because the forms are syntax errors there. What the keywords do
*not* do is make a program portable: a file using `async(io) f(x)` does not build with upstream Zig,
and this must be stated in the release notes of the stage that ships it, with a `Breaking:` trailer
on any commit that changes existing code (Live at Head rule 6).

The std surface this proposal touches, all additive:

| Addition | Why | Stage |
| --- | --- | --- |
| `Io.detach` + a `detach` vtable entry | a detached task needs an owner; dropping a future leaks on `Threaded` (§4.4) | 4 |
| Nothing else for the names | the compiler names its wrappers after the program's spelling (§3.5), so `spawnedName` needs no change | 3 |
| *Not* in this proposal | `Future` gains no field; `Io.async`/`concurrent`/`Group.*` keep their signatures; no vtable entry changes shape | — |

Zig++'s `std` also gains the test matrix §8 asks for, and a book chapter the day the sugar is
documented in the language reference. Both land in the stage that makes them true.

## 8. Stages

Each stage is a branch that lands on its own, with its acceptance run in CI before the next stage
starts. The stages are ordered so that each one is useful alone and none of them leaves the language
in a state where a keyword means something the compiler cannot check.

### Stage 1 — the front end, and the frame rule for bindings

Tokenizer untouched. `Parse.zig` recognises the five forms (§2.2). AstGen lowers bindings
(`async`, `concurrent`, `await`, `cancel`), emits the slots, the flags and the `defer`/`errdefer`
disposal of §3.1, and rejects: spawn statements, `detach`, non-binding operands of `await`/`cancel`,
non-call callees, bad return types, and a double consume in one statement sequence (§4.2).

*Acceptance*: a new `test/cases/async_await/` set, one file per accepted form and per rejection,
each rejection asserting the exact message; the accepted files run under `--io=threaded` and
`--io=threadz` (§11.1 is the model, and passes today); and an AIR-comparison test: for each form,
the AIR of the sugared file equals the AIR of the hand-written file of §3.1 instruction for
instruction, ignoring compiler-generated names — the same style of check the borrow checker's
proposal uses for its own promise of identical codegen (`doc/proposals/borrow-checker.md:627-670`).
`test/src/Cases.zig`'s manifest gains nothing: the sugar has no flag.

### Stage 2 — spawn statements and the frame group

The pre-scan for spawn statements, the frame's group, `g.async`'s typing rule, the group's
`defer`/`errdefer`, and the loop case. Error and early-return paths cancel before joining (§4.1,
§5).

*Acceptance*: a file that spawns N tasks in a loop and asserts they all ran (the hand-written shape
of §11.1 test 2, which is the model and passes today); a file whose error return cancels a spawn
statement's callee that would otherwise sleep past the exit, asserting the frame's wall clock does
not include the sleep; the AIR comparison of stage 1, extended to the group forms; both
implementations.

### Stage 3 — names

The generated wrappers carry the source's spelling (§3.5), so Threadz's dump and its task metrics
name tasks the way the program does. Depends on `threadz-budget-watchdog`'s `name` plumbing
(`40dee169b6`) being on master.

*Acceptance*: a test that spawns through the sugar on Threadz and asserts the name in
`stats()` (`scheduler.zig:1451` on that branch) or in the task dump is the source's callee name, for a plain function, a method and a comptime-generic
callee (the wrapper case).

### Stage 4 — `detach`, and the borrow checker's frame rule

`Io.detach`, the vtable entry, `Threaded`'s implementation (a pool-owned thread), Threadz's (the
root group), the stubs, and the gate of §4.4: `detach` accepted only where the module sets
`borrow_check` (`doc/proposals/borrow-checker.md:391-416`).

*Acceptance*: `detach` rejected in an unchecked module with the message of §4.2 rule 6, and accepted
in a checked one; the frame rule's own cases for its arguments — a spawn of `f(&local)` rejected, a
spawn of `f(owned_buffer)` accepted, `f(global)` accepted — matching the examples of
`doc/proposals/borrow-checker.md:1218-1247`; a detached task that outlives the spawning frame and
frees its owned argument on both implementations.

### Stage 5 — futures as ordinary values

The borrow checker's `owned`/`borrowed` annotations on `Future.await`, `Future.cancel` and the
`Io.async`/`concurrent` pairs; `await`/`cancel` accepting any `Future` value whose consumption the
checker can see; task bindings usable as values where ownership permits (§4.3).

*Acceptance*: a slice of futures awaited in a loop; a helper function that takes a `Future` by
`owned` parameter and joins it; the negative cases (a future stored into a frame object that
outlives it, a future passed to an unannotated callee that does not consume it); the whole suite of
stages 1-4 unchanged.

## 9. Open questions

1. **Does `Future` carry its `Io`?** §2.3 rejected the field for the sugar, because the compiler
   stores the spawn's operand already. Stage 5 changes the calculus: a future in a slice has no
   spawn site next to its use, so *something* must say which implementation awaits it — the field,
   or a rule that ties a `Future` to the `Io` of the call that made it, checked by the borrow
   checker. The field is the simpler answer and the one std can state in its own types; it costs
   one `Io` per future.
2. **Aliasing across a spawn.** A task waited on by a frame that keeps writing the same memory is a
   data race, and no check in this design sees it (§4.3). The borrow checker's qualifiers are the
   natural home for a rule — an argument to a spawn is, from the spawner's point of view, borrowed
   for as long as the task runs — and that rule is not designed yet. It is the largest known hole in
   this proposal.
3. **Should a `break`/`continue` out of a block with a live binding cancel instead of join?** §4.1
   joins on every non-error exit (`defer` semantics). Cancelling on `break` would match "I am
   leaving early" better; joining is what `defer` does for every other resource in the language.
   Decide with evidence from real code, not in the abstract.
4. **Spawn options.** Threadz's `SpawnOptions` has `stack_size` and `affinity`
   (`scheduler.zig:75-79`), and the affinity decision is per task (`doc/proposals/threadz.md:65`).
   The sugar has nowhere to put them. A second operand — `async(io, .{ .affinity = .pinned }) f(x)` —
   is the shape to weigh if the first users need pinning before a package API exists.
5. **A timeout on `await`.** `Io.operateTimeout` and `Batch.awaitConcurrent` have timeouts
   (`lib/std/Io.zig:567`, `:702`); `Future.await` does not. `await a timeout 100ms` is the kind of
   thing a language form makes pleasant and a method call makes awkward. It belongs to a later
   proposal with `Io.Timeout`, and it should not shape this one.
6. **How far the follower set should reach.** §2.2 requires the spawned call's callee to begin with
   an identifier or `@`. `(expr).f(x)`, a struct field of function-pointer type, and a callee whose
   receiver is itself a call are refused (§2.4). Widening is possible — the parser already knows the
   shape — and should be decided by what real code asks for.
7. **A `tasks` block for the group shape.** A block that binds one `Io` and one group for a set of
   spawns is a plausible sugar on top of stages 1-2, and nothing in §3.1 would change: it is the
   §3.1 lowering with the group's `Io` coming from the block's operand. Decide after the first users
   have written the frame-scoped form.
8. **Unbounded loops of spawn statements.** A spawn statement in an unbounded loop adds a member per
   iteration to a frame-scoped group; the tasks are joined, but they accumulate for the frame's
   lifetime. Production-quality guidance matters here (a bound, or a sub-frame), and it is a
   pressure-test of the per-frame group decision — the first thing to revisit if stage 2's users
   write it wrong.

## 10. What this proposal does not do

* It does not add async functions, function colours, or a suspension keyword: `async` returns a
  value of a struct type (`doc/proposals/threadz.md:29-31` keeps the compiler free of concurrency
  features, and this proposal is the exception the BDFL asked for, expressed entirely in std's
  existing types).
* It does not change `std.Io`'s existing functions or types, except for the additions of §7.
* It does not implement structured concurrency in the runtime: the runtime already has it (groups
  that join, cancellation that flows down, tasks that never get killed). The keywords are the
  language's spelling of it.
* It does not touch `Io.Scoped`, task-local values, supervision, or observability; those are
  Threadz's steps 6 and 7 (`doc/proposals/threadz.md:114-116`), and the keywords inherit whatever
  they decide.

## 11. Reproducing

The compiler is `0.17.0-dev.2480+zigpp.e95c60250` from this host's cache, pointed at a clone of this
tree at `753de4bb58`:

```sh
ZIG=~/.cache/zig/any/0.17.0-dev.2480+zigpp.e95c60250/zig
CLONE=/home/autark/src/zigpp-keywords          # doc-only clone; lib/ is this tree's
run() { ZIG_ANY=off ZIG_LIB_DIR=$CLONE/lib "$ZIG" "$@"; }
```

The example files are not committed: this branch changes `doc/` only. They were written for this
document, and each one reproduces with the command shown.

1. **The lowering, end to end** (§3.1). Four tests: a straight-line block with a binding and a spawn
   statement, a loop of statement spawns, an error exit that cancels before joining, and a cancelled
   task whose `error.Canceled` reaches the joiner.

   ```sh
   run test lowering.zig                                    # All 4 tests passed.
   run test lowering.zig --test-no-exec -femit-bin=lowering-test && ./lowering-test --io=threadz
   ```

2. **The identifiers** (§2.2). A file with declarations named `async` and `await`, called as
   functions, plus a function named `async`, a local used with `orelse`, and error handling directly
   after an identifier: compiles, and its calls mean what they meant before the change.

3. **The forms are syntax errors today** (§2.2). `_ = async(io) f(1);`, `_ = async f(1);`,
   `_ = await x;`, `_ = cancel x;`, `_ = concurrent(io) f(1);`, `_ = detach(io) f(1);` each produce
   `error: expected ';' after statement`.

4. **The callee rules** (§3.3, §3.4). `io.async(gen, .{u32, 5})` fails with
   `cannot create ArgsTuple for function with an 'anytype' parameter`; `io.async(global_fn.*, .{1})`
   fails with `cannot load comptime-only type 'fn (u32) u32'`; the wrapper forms (cases 4-6) compile and
   the test asserts their values; `Group.async` accepts a `void` and a
   `error{Canceled}!void` callee and rejects `error{Bad}!void` and `u32` with
   `expected type 'error{Canceled}!void'`.

5. **The cancellation path** (§5). A task that sleeps 10 ms, cancelled immediately, makes
   `f.cancel(io)` return `error.Canceled`; passes on both implementations.

6. **`errdefer` before `defer`** (§3.1, §5). A function with both, on the error path, prints
   `errdefer` then `defer`; on the success path, only `defer`.

7. **The ZIR** (§3.2). Compiled with a local cache, then dumped by Zig++ itself:

   ```sh
   run build-obj -fno-emit-bin air2.zig --cache-dir zc2
   run dump-zir zc2/z/*            # prints the ZIR of the hand-written lowering
   ```

   The quoted lines are from that dump. `dump-zir` is the compiler's own renderer
   (`src/main.zig:6140-6178`, `src/print_zir.zig`), and the cache file is keyed by file path
   and compiler version (`src/Zcu/PerThread.zig:490-497`), so the dump is reproducible for the same
   path.

8. **The line numbers.** `grep -rn "\.async(io" --include=*.zig src lib/std | wc -l` = 53;
   `grep -rnE "\bio\.async\(" … | wc -l` = 8; `grep -rnE "\.await\(" … | wc -l` = 98;
   `grep -rnE "\.cancel\(" … | wc -l` = 87. All at `753de4bb58`.

The one thing this document could not print is AIR. `--verbose-air` is gated on
`build_options.enable_debug_extensions` (`src/Zcu/PerThread.zig:2297`; `doc/proposals/borrow-checker.md:1777-1799`),
and the compiler used here is a release build, so §3.2 cites the ZIR the compiler does print, and
stage 1's acceptance asks for the AIR comparison in a build that has the extension. Nothing in this
document depends on AIR being printed.
