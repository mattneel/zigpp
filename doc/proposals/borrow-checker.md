# The borrow checker

Issue: [#7](https://github.com/mattneel/zigpp/issues/7). Branch `borrowck`. This is the design
proposal the issue's plan asks for: it answers every open question of issue #7 with a decision
and a reason, and it grounds every claim about the compiler in the source at `origin/master`
(`2887ca8eab`) with `file:line` citations. The AIR in §1, §6 and §9 was printed by Zig++ itself,
not written by hand; the commands and the compiler version are in §13. Anything inferred rather
than read out of the source is marked *[inferred]*.

The points issue #7 already decided are not re-opened here: it runs on AIR, per function, after
Sema; it checks per instantiation; aliasing starts from `noalias`; it is opt-in; it is compile
time only and emits exactly the same machine code; diagnostics point at the borrow, the
conflicting use and where the borrow is still live; the escape hatches (``@ptrCast``,
``@ptrFromInt``, ``@intFromPtr``, ``@fieldParentPtr``, `extern` calls) stay.

## 0. The decisions, in one table

| Open question (#7) | Decision | Where |
| --- | --- | --- |
| 1. How code opts in | A module option (`std.Build.Module.borrow_check`, `-fborrow-check` on the CLI), resolved through the module-option chain the compiler already has. Zero cost to code that does not set it: the language, the tokenizer and the parser are unchanged. | §3 |
| 1b. How annotations are spelled | Three *contextual* qualifiers in the parameter list — `owned`, `invalidates`, `borrowed` — recognised only where Zig's grammar today permits nothing but `comptime` and `noalias`. No keyword is added, reserved or otherwise. | §3.4 |
| 2. Checked code calling unchecked Zig | TypeScript's bargain: the boundary is *allowed*, not hardened. Unannotated callees are trusted under the default rules of §5.2, the bargain is recorded per compilation, and every use of an escape hatch is counted so the unchecked surface is visible. A hard boundary is rejected: until std is annotated it would leave the first cut unable to call std at all, including `Allocator.free` itself. | §5.2, §10.1 |
| 3. Lifetimes across calls | Nothing new is needed for the first cut: `*T` and `[]T` are mutable borrows, `*const T` and `[]const T` are shared borrows, `noalias` is a unique borrow, and a call result is a fresh object unless a parameter is marked `borrowed`. Lifetimes in signatures (§11 stage 4) are for *retention*, which the first cut does not check. | §5 |
| 4. `@fieldParentPtr` | A rule, not an escape hatch — except when the field pointer's provenance is already unknown. The recovered parent pointer is the same object as the field pointer, which is exactly what `std.Io`'s file reader/writer interfaces need, and the AIR already says so: `field_parent_ptr(ptr, field_index)` then a `ptr_cast`. | §6 |
| 5. std annotations | `owned` on the freeing operations (`Allocator.free`/`destroy`, every `deinit`, `rawFree`), `invalidates` on the operations whose own documentation already promises invalidation (`ArrayList.append`, `HashMap.put`, `toOwnedSlice`, …), `borrowed` on the accessors that hand out pointers into an object (`getPtr`, `Io.Reader.peek`, …). Container *growth* needs nothing else: `append` already takes `*Self`. The allocator's annotations ship with the first cut, not with stage 5, or the first cut would have nothing to check against. | §7 |
| — Non-goals and known unsoundness | §10 lists the non-goals (leak checking, index-sensitive invalidation, retention, races, comptime) and each piece of known unsoundness, with the reason it is acceptable for 1.0. | §10 |

The rest of this document is the argument for that table.

## 1. What a borrow is, in Zig terms

A borrow is not a type and not a syntax; it is a fact the checker derives about a *value* in a
function's AIR: **a pointer value, the memory object it points into, how it may be used, and the
region of the function over which it stays live.** Everything else in this proposal is derived
from those four things.

### 1.1 Borrows are values, and AIR already names them

The checker walks one `Air` per function (`Air.instructions` plus `Air.extra`,
`src/Air.zig:22-36`; the main block is decoded by `Air.getMainBody`, `src/Air.zig:1602-1605`).
Every AIR instruction whose result is a pointer, slice, optional of one, error union with one, or
an aggregate containing one is a *borrow-creating* instruction. The rules, per tag:

| AIR tag | what the result borrows |
| --- | --- |
| `arg` (`src/Air.zig:1303-1307`) | a *parameter object*: an object owned by the caller, of unknown extent. `noalias` is not here — it is in the function type (§3.4). |
| `alloc`, `ret_ptr` (`src/Air.zig:230,246`) | `alloc` a *frame object* of this function; `ret_ptr` the caller's result slot. |
| `call`, `call_always_tail`, `call_never_tail`, `call_never_inline` (`src/Air.zig:375-386`) | a *fresh object*, or the object of a parameter marked `borrowed` (§5.3). |
| `struct_field_ptr`, `struct_field_ptr_index_0..3`, `struct_field_val`, `agg_field_val` (`src/Air.zig:691-702`) | the same object as the operand, at `.field(k)`. |
| `ptr_elem_ptr`, `slice_elem_ptr`, `array_elem_val`, `slice_elem_val`, `ptr_elem_val` (`src/Air.zig:730-747`) | the same object as the operand aggregate, at `.elem`. |
| `slice`, `array_to_slice`, `slice_ptr`, `ptr_slice_ptr_ptr` (`src/Air.zig:710-750`) | the same object, at `.bytes`. |
| `ptr_add`, `ptr_sub` (`src/Air.zig:177-195`) | the same object, at `.bytes(offset)` when the offset is a constant. |
| `field_parent_ptr` (`src/Air.zig:913-915`) | the same object, at `.parent(field_index)` (§6). |
| `ptr_cast`, `bit_cast`, `addrspace_cast` (`src/Air.zig:290-314,951`) | the same object; `ptr_cast` is an escape hatch only when its operand is unknown (§10.2). |
| `optional_payload`, `optional_payload_ptr`, `optional_payload_ptr_set`, `wrap_optional` (`src/Air.zig:655-666`) | the same object as the optional's payload. |
| `unwrap_errunion_payload`, `unwrap_errunion_payload_ptr`, `wrap_errunion_payload`, `errunion_payload_ptr_set` (`src/Air.zig:667-687`) | the same object as the error union's payload. |
| `aggregate_init`, `union_init`, `select` (`src/Air.zig:891-901,802`) | the objects of the elements: an aggregate is a *set* of borrows. |
| `int_from_ptr` (`src/Air.zig:308-314`) | the same object, but the result is *opaque*: an integer carries the provenance but no path. |
| `ptr_from_int` | *unknown* — see §10.2. |
| `load` | whatever was stored at that path: a pointer read out of memory is a borrow of the object the memory belongs to, resolved by the store/load congruence of §1.4 rule 3. |
| everything else | no borrow. In particular a *use* is not a creation: `dbg_stmt`, `is_non_null`, `is_err`, `cmp_*` do not create borrows. |

`Air.Liveness` is not a borrow analysis and is not claimed to be one — it marks the last use of
each AIR value (`operandDies`) after a backward pass over the whole body
(`src/Air/Liveness.zig:1-6,827-860`). It is the substrate for §1.4, nothing more.

### 1.2 Origins: a root and a path

An *origin* is a pair `(root, path)`.

The root is one of:

| root | created by | what a free may do to it |
| --- | --- | --- |
| `Frame(alloc_i)` | `alloc` (`src/Air.zig:230`) | nothing: a frame object dies with the frame, and the checker's rule is that it must not escape (§2.3). |
| `Call(call_i)` | a pointer-valued `call*` (`src/Air.zig:375-386`) | it is the object a later `owned` call may free. |
| `Param(i)` | `arg` (`src/Air.zig:1303-1307`) | it is external: the caller owns it. |
| `Result` | `ret_ptr` (`src/Air.zig:246`) | it is the caller's slot. |
| `Global(nav)` | a reference to a container-level declaration — an interned pointer whose base is a `Nav` (`InternPool.Key.Ptr.BaseAddr`, `src/InternPool.zig:2434-2462`), or `runtime_nav_ptr` when it is resolved at runtime (`src/Air.zig:967`) | it is immortal for the duration of the program. |
| `Unknown` | `ptr_from_int`, `assembly` outputs, an unmodelled instruction | nothing is checked through it. |

The path is a short sequence of steps from `{ .field(index), .elem, .bytes(offset), .parent(index), .loaded }`.
Two pointers *overlap* when their roots are the same and one path is a prefix of the other, or
both paths end at the same `.loaded` site. Overlap is what the `noalias` check (§2.4) and the
`invalidates` rule (§2.5) use; it is deliberately a *syntactic* relation over the navigation
instructions AIR really contains, so that the checker never has to solve a points-to problem.

Two AIR facts make this cheaper than it sounds. First, `struct_field_ptr_index_N` is its own tag
with the field index in the tag (`src/Air.zig:696-699`), so field navigation is a table lookup.
Second, `Sema` already tracks a *comptime* base pointer and byte offset for interned pointer
values (`InternPool.Key.Ptr.BaseAddr`, `src/InternPool.zig:2432-2462`) and
`Sema.resolveComptimeKnownAllocPtr` follows `struct_field_ptr`, `ptr_elem_ptr` and `ptr_cast`
backwards through AIR to a known allocation for a different purpose
(`src/Sema.zig:3574-3638`). That is prior art for the runtime path walk, not a borrow check; the
checker generalises the walk and drops the "comptime-known" restriction.

### 1.3 Kinds

| kind | where it comes from | what the checker does with it |
| --- | --- | --- |
| shared | `*const T`, `[]const T`, `[*]const T` | nothing (any number may be live at once) |
| mutable | `*T`, `[]T`, `[*]T` | nothing by itself — Zig's `*T` promises nothing about what the callee does with the memory (§5.1) |
| unique | a `noalias` parameter, from the callee's own function type | at a call: the argument must not overlap any other argument (§2.4) |
| owned | a parameter marked `owned` (§5.3) | the call kills the object: every borrow into it is dead afterwards (§2.2) |
| unknown | provenance destroyed by an escape hatch | suppressed from the checks; the object it came from is marked *escaped* (§10.2) |

The kind is read off the *type* where the type carries it (`InternPool.Key.PtrType.Flags`:
`is_const`, `is_volatile`, `is_allowzero`, address space, size — `src/InternPool.zig:2052-2081`)
and off the *function type* where the function carries it: `noalias` is
`InternPool.Key.FuncType.noalias_bits` with `paramIsNoalias(i)`
(`src/InternPool.zig:2173-2194`), not an AIR instruction field — `Air.Inst.Data.arg` has only
`ty` and `zir_param_index` (`src/Air.zig:1303-1307`), and a real AIR dump shows exactly that:

```text
# Begin Function AIR: ex4.sum:
  %0 = arg([]const u8, 0)
  %1 = arg([]const u8, 1)
```

`sum` was declared `fn sum(noalias a: []const u8, b: []const u8) usize`, and the AIR does not
mention `noalias`: a checker that read only instruction payloads could not see it. It must read
the callee's function type, which is why §3.4 puts the new qualifiers in the same place.

### 1.4 The region of a borrow, and how `defer` disappears into it

The region of a borrow is the set of program points at which any value derived from it may still
be used. It is computed from three things:

1. **Liveness of AIR values.** For the value `V` created at instruction `i`, the region starts at
   `i`. Its end is the instruction at which `Air.Liveness.operandDies(V)` is set — the last use of
   `V` — or the end of the enclosing block if no use exists (`Air.Liveness.isUnused`,
   `src/Air/Liveness.zig:190-203`).
2. **Derivation.** If `W` is created from `V` (`struct_field_ptr(V)`, `slice(V, …)`, `ptr_cast(V)`,
   `optional_payload(V)`, …), then `W`'s region is part of `V`'s: the borrow ends at the *latest*
   end among its derived values. This is what makes "the borrow is still live here" in a
   diagnostic mean something concrete: it names the derived value that is still live.
3. **Memory.** A pointer stored into memory (`store`, `store_safe`, `memcpy`, an aggregate
   written to a field, an `optional_payload_ptr_set`) is *carried* by that path: a later `load`
   from the same path yields a value whose origin is the stored one. A store to a path ends the
   region of every borrow whose origin path is that path or a prefix of it and whose root is
   unchanged — unless the stored value is derived from the same root and reaches the same path, in
   which case the region continues.

Rule 3 is what lets the checker see a borrow survive a call that stores it: `std.Io`'s file
readers pass `&adapter.interface` down and the callee's vtable entry recovers the parent (§6.3),
and loops reload pointers from locals (in the dump in §9.16, `%10 = load(*Io.Reader…)` and
`%32 = load(usize, %26)` re-read the loop state on every iteration).

`defer` and `errdefer` need no rule of their own, because by the time the checker runs they are
not in the AIR at all. `AstGen` appends the deferred body to every exit path —
`GenZir.genDefers` walks the defer scopes innermost-first (`lib/std/zig/AstGen.zig:2993-3016`),
`ret` emits them before `addRet`/`.ret_err_value` (`lib/std/zig/AstGen.zig:7983-8030`), and
`blockExprStmts` covers fallthrough (`lib/std/zig/AstGen.zig:2641-2643`) — and Sema simply
analyses those ZIR bodies where they appear (`src/Sema.zig:1941-1956`). A function with
`defer gpa.free(buf);` and a loop has the free *before* each `ret`. Here is the real exit of the
worked example §9.1 (`fn fill(gpa: Allocator, n: usize) void`, allocate, fill, `defer
gpa.free(buf)`), printed by Zig++:

```text
  %51!= dbg_stmt(3:19)
  %52 = load(mem.Allocator, %5!)
  %55!= dbg_stmt(3:19)
  %56!= call(<fn (mem.Allocator, []u8) void, (function 'free__func_1')>, [%52!, %7!])
  %57!= ret_safe(@.void_value)
# End Function AIR: ex1.fill
```

Two things follow, and both matter later. The free is an ordinary `call` whose callee is the
*instantiated* `Allocator.free` (its AIR name is `free__func_1`, because `free` takes `anytype`
and every call creates its own instance, `src/Sema.zig:7085-7134`), and the `defer` body sits on
the exit path of the function it was written in, so "use after free through a `defer`" is
ordinary forward flow analysis over the AIR — no special case.

The early exit analysis is equally visible. In §9.2 the body is
`const buf = try gpa.alloc(u8, 8); errdefer gpa.free(buf); if (!ok) return error.Bad;`, and only
the error arm carries the free:

```text
    %30 = not(bool, %1!)
    %41!= cond_br(%30!, poi {
      %26!
      %31!= dbg_stmt(3:22)
      %32 = load(mem.Allocator, %5!)
      %35!= dbg_stmt(3:22)
      %36!= call(<fn (mem.Allocator, []u8) void, (function 'free__func_1')>, [%32!, %23!])
      %37!= dbg_stmt(4:14)
      %38!= call_never_tail(<noinline fn () void, (function 'returnError')>, [])
      %39!= ret_safe(<…!void, error.Bad>)
    }, poi {
      %5! %23!
      %40!= br(%29, @.void_value)
    })
```

The success arm branches to the rest of the function with the object still alive; the error arm
frees and returns. An `errdefer` is therefore not a second mechanism: it is a free on a subset of
the exits, and the checker's state at the merge point is the join over the predecessors that can
actually reach it.

### 1.5 What counts as a use

An instruction *uses* a borrow when it:

* reads or writes memory through it — `load`, `store`, `store_safe`, `memcpy`, `memmove`,
  `memset`, `memset_safe`, the `atomic_*` family, `cmpxchg_*`, `prefetch`,
  `legalize_vec_store_elem` (`src/Air.zig:614-626,806-872,1012-1018`);
* passes it to a call (`Air.unwrapCall`, `src/Air.zig:2302-2321`);
* returns it (`ret`, `ret_safe`, `ret_load`, `src/Air.zig:596-613`);
* stores it into memory (the store itself is the use that escapes it, §2.3);
* converts it to an integer (`int_from_ptr`) or reads its address into an aggregate
  (`aggregate_init` containing the pointer is a *navigation*; storing that aggregate is the use).

Navigation alone — `struct_field_ptr`, `ptr_elem_ptr`, `slice`, `ptr_cast`, `bit_cast` — is *not*
a use for the purposes of an error, because Zig has no undefined behaviour in computing an
address; the checker only moves the borrow. This keeps `&x.field` legal on a live borrow and
keeps the diagnostics about memory, not about arithmetic. `is_null`, `is_non_null`, `cmp_*` on
pointers are likewise not uses of memory, only of the pointer value; the checker still counts
them for liveness, because a pointer compared is still a pointer someone holds.

### 1.6 What counts as a free, and what counts as invalidating

AIR has no free instruction. `alloc` is a *stack* allocation (`src/Air.zig:228-230`), and nothing
in the instruction set releases heap memory: freeing is a `call`, and the callee is either
`Allocator.free` or something that wraps it. The only honest source of truth for "this call
releases memory" is the callee's own declaration, which is what the `owned` qualifier of §3.4
records:

```zig
pub fn free(self: Allocator, owned memory: anytype) void { … }   // std/mem/Allocator.zig
```

Semantics of a call with an `owned` parameter `i`:

* the object that argument `i` points into **dies** at this call, on this path;
* every borrow whose root and path overlap argument `i`'s is dead afterwards;
* a *use* of any of them later is `use of freed memory`; a second `free` of the same object is
  `double free`;
* if the argument's root is `Frame`, the call itself is the error: an allocator cannot free a
  frame object.

`invalidates` is the weaker contract, for the operations whose own std documentation already
promises it:

```zig
pub fn append(self: *Self, gpa: Allocator, item: T) Allocator.Error!void   // "Invalidates element pointers
                                                                            //  if additional memory is needed."
```

Semantics of a call with an `invalidates` parameter `i`: every borrow that overlaps argument `i`
*below* it (a strictly longer path, or a `.loaded` path) becomes **invalid**; a later use is
`use of invalidated memory`. The difference from `owned` is only in what the checker claims —
`owned` asserts the object was released, `invalidates` says pointers into it may have moved or
gone — and it changes the wording of the diagnostic, not the rule that a later use is an error.

The checker does not prove that a particular `append` grew the list: the std documentations say
"if additional memory is needed", and the checker takes the conditional as a *may*. Precision is
obtained the way std gets it — reserve first, then use the assume-capacity operations (§9.12).

### 1.7 On "allocator-owned memory"

Issue #7 says the first cut catches use after free "of allocator-owned memory". An allocator in
Zig is a `struct { ptr: *anyopaque, vtable: *const VTable }` (`lib/std/mem/Allocator.zig:22-24`)
and the compiler has no special knowledge of it: the type is an ordinary struct value, and
`Allocator.free` is an ordinary generic function. Nothing about `[]u8` says "this came from an
allocator", and nothing about a `*u8` parameter says it may not be freed. The checker therefore
does not try to reason about "allocator ownership" as a type property; it tracks the object a
pointer points into, and it learns that an operation releases that object from the callee's
`owned` annotation. `Allocator.free`'s annotation is the definition, and it is the same
mechanism any user library would use.

## 2. What the first cut checks

Four checks, all of them flow-sensitive over the CFG of one function's AIR, all of them reported
with the same three-party diagnostic (§8).

### 2.1 Use after free

For every use `U` (§1.5) of a borrow `B` whose kind is not `unknown`: if any path from `B`'s
creation to `U` contains a call with `owned` parameter whose argument overlaps `B`, report

```text
error: use of freed memory
```

with a note at `B`'s creation, a note at the freeing call, and a note at the end of `B`'s region.
The analysis is a *may* analysis: one freeing path is enough, because a use that *can* touch freed
memory is a bug on that path. This is deliberate, and it is the case that catches

```zig
if (cond) gpa.free(buf);
buf[0] = 1;                       // rejected: the free is on a path that reaches here
```

### 2.2 Double free

For a call with `owned` parameter whose argument overlaps an object that is already dead:
`error: double free`, with a note at the first free. Two shapes are checked: two explicit frees of
the same object, and a `defer`-registered free plus an explicit free (§9.3, §9.4). The second
shape is why the check runs after Sema: the deferred free is on the exit path, so the second free
is a use of a dead object like any other.

### 2.3 Dangling frame pointers

Frame objects (`alloc`, `src/Air.zig:230`) obey one rule: **a pointer into a frame object may not
be reachable, at any exit of the function, from anything that outlives the frame.** The checker
enforces it in three places:

1. **Returning.** A `ret`/`ret_safe`/`ret_load` whose value is derived from a `Frame` root of this
   function is an error, and so is returning an aggregate or an optional that *contains* one.
   Note that Sema already rejects the syntactic form of this: `return &x;` is
   `returning address of expired local variable 'x'` from AstGen
   (`lib/std/zig/AstGen.zig:8252`). The checker's job is the forms AstGen cannot see — through a
   pointer launder, through an aggregate, through an escape hatch, through a call.
2. **Storing into something that outlives the frame.** A `store` of a frame-derived pointer to a
   path whose root is `Param`, `Result`, `Global` or a *heap* object whose allocation predates the
   store marks that path *tainted*. The store is not itself an error; the error is reported at the
   **exit**, if the frame-derived pointer is still reachable from the tainted path. `h.p = &y;`
   with `h: *Holder` (§9.7) is reported at the exit with the note at the store; but
   `std.Io.TypeErasedQueue.putLocked`, which links a stack `Put` into `q.putters` and removes it
   before returning (`lib/std/Io.zig:2197-2211`), is accepted, because the taint is undone. This
   is the rule that makes the escape check useful in the presence of inverted control flow.
3. **Being captured by something that escapes.** The rules compose: if a frame-derived pointer is
   put into a heap object of this frame, the object is tainted, and returning *that object* is
   reported (§9.8). The same holds through `int_from_ptr`, which keeps the provenance so that
   returning `@intFromPtr(&x)` is still caught.

### 2.4 Broken `noalias` promises

`noalias` is a promise the *caller* makes: this argument does not alias any other pointer the
callee can reach. In Zig the promise is documented nowhere — the language reference's entry is
literally `TODO add documentation for noalias` (`doc/langref.html.in:7673-7678`) — and the
compiler enforces only its *type* rules (`Sema.checkParamType`: "non-pointer parameter declared
noalias", `src/Sema.zig:8776-8777`; at most 32 noalias parameters, `src/Sema.zig:20246-20254`;
`noalias` bits must be preserved through function-pointer coercions, `src/Sema.zig:29154-29168`).
Nothing checks the promise itself; breaking it is illegal behaviour, as issue #7 says.

This proposal defines the promise, and therefore the check:

> **Def. `noalias`.** A `noalias` argument does not overlap any other argument of the call, and it
> does not overlap any memory reachable from another argument for the duration of the call.

Which is checked: at a call whose callee's function type has `paramIsNoalias(k)`
(`src/InternPool.zig:2173-2194`, read through `Zcu.typeToFunc`, `src/Zcu.zig:4149-4152`), the
checker requires (a) argument `k` to be pairwise non-overlapping with every other pointer argument
whose root is known, and (b) no other live borrow into argument `k`'s object to be live across the
call. Overlap is the §1.2 relation. `sum(x, x)` is rejected (§9.13); `sum(x, y)` is accepted
(§9.14). `@memcpy`'s two operands are `noalias` by declaration
(`doc/langref.html.in:5218`), so overlapping slices of one array are rejected at compile time
where today the compiler only rejects them when the alias is comptime-known
(`src/Sema.zig:24844`, `"'@memcpy' arguments alias"`) and otherwise emits a runtime safety check
(`debug.FullPanic.memcpyAlias`, `lib/std/debug.zig:206-208`, seen in the AIR as
`%41!= memcpy(%9!, %21!)` followed by the `call_never_tail(… 'memcpyAlias')` panic path).

### 2.5 Invalidation

For every use `U` of a borrow `B` whose root/path overlaps an `invalidates` parameter of a call on
a path from `B`'s creation to `U`: `error: use of invalidated memory`, with a note at the call
(quoting the annotation), a note at `B`'s creation, and a note at `B`'s region end. §9.11 is the
required case, a slice of `ArrayList` items taken before an `append`; §9.12 is the reserve-first
version that is accepted.

### 2.6 How paths merge

The checker is a forward dataflow over the AIR control-flow graph, with a state per root
(`alive`, `dead`, `invalid`, `escaped`, `tainted`) and a lattice ordered by "worse":
`alive ⊑ invalid ⊑ dead`, and `escaped`/`tainted` as separate may-facts. AIR's control flow is
nested bodies in `Air.extra` (`Air.unwrapBlock`, `unwrapCondBr`, `unwrapSwitch`, `unwrapTry`,
`unwrapTryPtr`, `src/Air.zig:2245-2401`); the checker walks the same structure that
`Air.Liveness.analyzeBody`, `Air.Legalize.legalizeBody` and `Air.Verify.body` walk
(`src/Air/Liveness.zig:372-383`, `src/Air/Legalize.zig:362-371`, `src/Air/Verify.zig:64-103`).

| construct | AIR shape | the rule |
| --- | --- | --- |
| `defer`, `errdefer` | the deferred body is inlined on the exits (no AIR tag; §1.4) | nothing special: the free is on the paths it is written on. |
| `if`, `orelse`, `catch`, `try` | `cond_br` with two nested bodies (`src/Air.zig:1398-1412`), `@"try"`/`try_cold` whose error body terminates and whose success path continues (`src/Air.zig:512-529`) | analyse both bodies; at the join, the state is the join of the states of the predecessors that *can reach the join*. A body ending in `ret`/`unreach`/`trap`/a noreturn call does not reach it, so a free followed by a return does not make the continuation dead. |
| `switch` | `switch_br` / `loop_switch_br` with cases then else (`src/Air.zig:1414-1433`) | same join over all case bodies. |
| loops | `loop` + `repeat` + `br` back to the loop (`src/Air.zig:350-357`) | iterate to a fixpoint: the state at the loop head is the join of the state on entry and the state on every back edge. A pointer freed on one iteration is dead for the next, which is what §9.17 checks. |
| labeled `break` with a value | `br` to an outer `block` carrying the value (`src/Air.zig:1382-1385`) | the break's operands are uses at the `br`; the merged state at the target block is the join over all breaks. §9.18 breaks out of a loop after freeing. |
| `unreachable`, `trap` | `unreach`, `trap` (`src/Air.zig:629,362`), result type `noreturn` (`Air.typeOfIndex`, `src/Air.zig:1773-1798`) | a dead end: paths through it never merge, so a free before `unreachable` cannot make a later use dead. §9.19 has `if (n == 1) unreachable;` on one arm only. |
| noreturn calls | still a `call` (`src/Air.zig:374-380`), with Sema appending `.unreach` after it (`src/Sema.zig:7199-7214`) | same as `unreachable`. Note `Air.Verify.body` documents that instructions *after* a noreturn call can exist as a safety-check artefact (`src/Air/Verify.zig:437-461`); the checker treats that tail as an impossible-return path and does not let it merge. |
| `catch unreachable`, `orelse unreachable` | a `cond_br` whose error/null arm ends in a panic or `unreach` (`lib/std/zig/AstGen.zig:5851-5902`; `Sema.maybeErrorUnwrap`, `src/Sema.zig:12846-12901`) | the dead arm does not merge; the live arm is analysed normally. |
| inline assembly | `assembly` with clobbers and outputs (`src/Air.zig:247-248,1530-1562`) | an opaque use: every `escaped` object is marked invalid and no diagnostic is derived from it (§10.3). |

The join for a *dead* object is "dead if dead on some predecessor that reaches this point", and a
use whose state set contains `dead` is reported. The checker does not report a function that
merely *may* free nothing: an object that is never freed and never escapes is not an error, and
leaks are not reported at all (§10.4).

## 3. Opting in

### 3.1 The knob

**A module option.** `std.Build.Module` gains a nullable `borrow_check: ?bool`, the CLI gains
`-fborrow-check` / `-fno-borrow-check`, and the resolved boolean lands on the compiler's
`Module`. Only the function bodies whose *owner module* has the option set are checked. Nothing
else changes: no tokenizer entry, no reserved word, no parser rule, and unchecked modules do not
run the pass at all (the hook is guarded by the module flag, §4.1).

Why a module option, and not a per-file attribute or a per-function attribute:

* it is the smallest opt-in that can be *sound about itself*: annotations from any module are
  readable as facts, but only opted-in modules are checked, so `std` can carry annotations for
  years without being checkable itself;
* it composes with `build.zig`, which is where “this project wants the borrow checker” is already
  expressed (targets, optimization mode, single-threaded, stack checking);
* it costs existing code exactly nothing, which is the constraint issue #7 puts first. A per-file
  attribute (`//!borrow-check` style) would need parser-visible syntax in every file that wants it;
  a per-function attribute would need a new declaration-level qualifier. Neither is required to
  start.

Why not a pointer qualifier as the opt-in (a `borrowed *T` type): because it makes the *type
system* participate, and types are the one thing this design wants to leave alone. Zig's `*T` must
keep meaning exactly what it means today in unchecked code, and a checked module and an unchecked
module have to agree on what a pointer is, or a checked function could not call an unchecked one
at all. Keeping the check out of types is what makes the bargain of §5.2 possible.

### 3.2 The chain, as it exists today

The module-option mechanism is already there; the borrow checker adds one field to it. Read the
following as the change list:

| step | file | the existing analogue |
| --- | --- | --- |
| build API field | `lib/std/Build/Module.zig:29-45,213-234` | `single_threaded`, `stack_check`, `no_builtin` — nullable fields copied from `CreateOptions` in `Module.init` (`:268-282`) |
| build graph serialisation | `lib/std/Build/Serialize.zig:1223-1253` | `.single_threaded = .init(m.single_threaded)`, `.stack_check`, `.no_builtin` |
| protocol | `lib/std/Build/Configuration.zig:1721-1750` | the wire fields for those three |
| resolved options | `src/Module.zig:70-87` | `CreateOptions.Inherited` (all nullable), resolved local → parent → default in `Module.create` (`src/Module.zig:165-167,272-280,323-327`), stored concretely (`:420-436`) |
| CLI | `src/main.zig:1922-1925,2075-2090` | `-fstack-check`/`-fno-stack-check`, `-fsingle-threaded/…`, `-fbuiltin/…` are parsed into `mod_opts`; `buildOutputType` snapshots them per `-M` module at `src/main.zig:1580-1590,3626-3650`, and `createModule` passes `.inherited` to `Module.create` (`src/main.zig:4729-4739`) |
| cache key | `src/Compilation.zig:1268-1287` | `cache_helpers.addModule` hashes the resolved `single_threaded`, `stack_check`, `no_builtin`; `borrow_check` joins that set, so toggling it invalidates cached objects |
| test harness | `test/src/Cases.zig:734-755`, `:558-566` | `TestManifest.valid_keys` gains `borrow_check`; `Cases.lowerToBuildSteps` sets it on the module it builds, so a `test/cases` file can say `// borrow_check=true` |

### 3.3 What "checked" means for a module

* Every function body of that module that is *analysed* is checked: a non-generic function, and
  each instantiation of a generic one (§4.3).
* A body from another module that Sema *inlines* into a checked body is part of the checked
  body’s AIR and is therefore checked as part of it (`Sema.analyzeCall`’s inline path,
  `src/Sema.zig:7226-7470`; there is no AIR call to the callee). That is the right default: the
  check is about the code as compiled.
* `comptime`-evaluated code has no AIR and is not checked (`Sema.analyzeCall`’s comptime path
  resolves to a value, `src/Sema.zig:7280-7489`).
* Calling an unchecked function is allowed everywhere; what that means is §5.2.
* A checked function may be called from unchecked code; nothing is checked at the call site
  (the callee’s own body was checked when it was analysed).

### 3.4 How annotations are spelled: three contextual qualifiers

The checker needs facts that Zig’s types do not carry (§5.1). Three qualifiers supply them, in the
one position the grammar already reserves for parameter qualifiers — which today is exactly
`comptime` and `noalias`:

```text
ParamDecl <- doc_comment? (KEYWORD_comptime? KEYWORD_noalias? IDENTIFIER COLON TypeExpr
            | KEYWORD_comptime? KEYWORD_anytype)
```

becomes

```text
ParamQualifier <- KEYWORD_comptime | KEYWORD_noalias | "owned" | "invalidates" | "borrowed"
ParamDecl <- doc_comment? ParamQualifier* (IDENTIFIER COLON TypeExpr | KEYWORD_anytype)
```

`owned`, `invalidates` and `borrowed` are **contextual**: they are ordinary identifiers, and they
are recognised as qualifiers only when they appear immediately before a parameter name. That is
not a new reserved word, and it is not even a change to what existing code may say, because the
form is already a syntax error. Verified against Zig++’s own parser:

```text
$ zig build-obj -fno-emit-bin syn1.zig
syn1.zig:1:12: error: expected ',' after parameter
fn f(owned x: u8) void { _ = x; }

$ zig build-obj -fno-emit-bin syn2.zig      # fn f(owned: u8) void { _ = owned; }
$ echo $?
0
```

The second program still compiles: a parameter may still be *named* `owned`, and a declaration
may still be named `owned`, `invalidates` or `borrowed`. The cost to code that does not ask for
anything is zero — under the opt-in of §3.1 the qualifiers do not have to be used at all, and
outside it they do not have to be *avoided* either. This is the difference from `priv`, which
issue #7 allows naming as the fork’s one exception: `priv` is a keyword everywhere; these three
are nothing anywhere except in one syntactic position that Zig rejects today.

The qualifiers:

| qualifier | on | means | read by |
| --- | --- | --- | --- |
| `owned` | a parameter of pointer/slice type | this call takes ownership of the object the argument points into, and normally releases it. The object dies at this call. | §2.1, §2.2 |
| `invalidates` | a parameter of pointer type | this call may reallocate or release the storage behind the argument; pointers into it may be invalid afterwards. | §2.5 |
| `borrowed` | a parameter of pointer/slice type | if this function returns a pointer, slice, optional or error union containing one, the result may point into the object given for this parameter, and inherits its lifetime. | §5.3 |

`std`’s own documentation is the source of the wording, which is why these spellings and not
others: `Allocator.free` “free and invalidate a region of memory”
(`lib/std/mem/Allocator.zig:76-86`), `ArrayList.append` “invalidates element pointers if
additional memory is needed” (`lib/std/array_list.zig:1023-1028`), and an accessor like
`HashMapUnmanaged.getPtr` returns a pointer whose lifetime is the map’s
(`lib/std/hash_map.zig:1066-1078`).

Where they are stored: beside `noalias`, in the function type.
`InternPool.Key.FuncType.noalias_bits` is a `u32` bitmask with the accessor
`paramIsNoalias(i)` and presence flag `has_noalias_bits`
(`src/InternPool.zig:2179-2212,5243-5252,5556-5567,7036-7052`). The three new qualifiers follow
that shape exactly: `owned_bits`, `invalidates_bits`, `borrowed_bits`, one `u32` each plus their
own presence flags in the encoded type, and the same two limits `noalias` already has — at most
32 parameters per qualifier (`src/Sema.zig:20246-20254`) and identical bits required when a
function value is coerced to a function pointer (`src/Sema.zig:29154-29168,29692-29696`). The
plumbing is the `noalias` trail, end to end: parser slot in `lib/std/zig/Ast.zig`’s
`FnProto.Param` (`:2603-2609`, whose `comptime_noalias: ?TokenIndex` today carries both
qualifiers), bits collected in `AstGen.fnDeclInner` (`lib/std/zig/AstGen.zig:4160-4169`) and
passed through `GenZir.addFunc` (`:11625-11643,11679-11680`), read by `Sema.zirFuncFancy`
(`src/Sema.zig:25159-25187`) and validated in `Sema.funcCommon` (`:9084-9107,9144-9169`) with the
same “non-pointer parameter declared noalias”-style error for a non-pointer `owned` parameter
(`Sema.checkParamType`, `:8776-8777`). Generic instantiations rebuild the masks for the remaining
runtime parameters, which `Sema.analyzeCall` already does for `noalias`
(`src/Sema.zig:7092-7128`) — the new masks are added to that loop, or a call through a generic
`owned` parameter would lose its contract.

Reflection follows the `priv` precedent, which put its flag in `@typeInfo` rather than hiding it
(`doc/book/src/what-zigpp-adds.md`, “Private fields”): `std.lang.Type.Fn.ParamAttributes` today is
`struct { @"noalias": bool = false }` (`lib/std/lang.zig:881-887`), and gains
`@"owned"`, `@"invalidates"`, `@"borrowed"` booleans next to it, filled from `FuncType` in
`Sema.zirTypeInfo` (`src/Sema.zig:16267-16290`, where `param_attrs_fields` is built). Generic code
that reflects over signatures therefore sees the contracts, which is what will let std’s own
generic wrappers forward them.

## 4. Where it runs

### 4.1 The hook

**`Zcu.PerThread.analyzeFuncBody`, `src/Zcu/PerThread.zig:2288-2314`, immediately after
`analyzeFuncBodyInner` returns the function’s `Air` and before the codegen task is queued.**
That function is where the AIR exists and nothing else has seen it yet:

```zig
var air = try pt.analyzeFuncBodyInner(func_index, reason);
var air_owned = true;
defer if (air_owned) air.deinit(gpa);
…
if (comp.bin_file != null or zcu.llvm_object != null or dump_air or dump_llvm_ir) {
    …  // the codegen task, and with it the link queue
}
```

The guard the checker adds is the same module lookup codegen itself does
(`src/codegen.zig:161-165`): `zcu.navFileScope(func.owner_nav).mod.?` (`src/Zcu.zig:4468-4474`),
then `mod.borrow_check`. Properties of this spot, each of which the design depends on:

* the AIR here is exactly what Sema produced: not legalised by a backend (`Air.legalize` runs
  later, `src/Zcu/PerThread.zig:4512-4516`), so the instruction set the checker reasons about is
  the same on every target and no target-specific rewriting is in the way;
* it runs whether or not a binary is emitted — `-fno-emit-bin`, `zig build-obj`, a compile-error
  test, or an IDE analysis all reach it, because `ensureFuncBodyUpToDate` calls
  `analyzeFuncBody` for every outdated function independently of emission
  (`src/Zcu/PerThread.zig:2196-2234`);
* a rejection happens *before* the codegen task is enqueued, so a rejected function never reaches
  `codegen.generateFunction` (`src/codegen.zig:149-179`) or `Object.updateFunc`
  (`src/codegen/llvm.zig:1087-1097`) — there is no path from a diagnostic to a changed instruction
  stream (§4.4);
* the AIR is still owned by `analyzeFuncBody` (the `defer … air.deinit(gpa)` above), so the
  checker borrows it and allocates nothing that has to outlive the function.

Not in `runCodegenInner` (where liveness is computed today, `src/Zcu/PerThread.zig:4517`), because
that point exists only for functions that are being code-generated, is after backend legalisation,
and would leave non-emitting compilations unchecked.

Errors are reported the way every other per-function analysis reports them:
`Zcu.ErrorMsg` (`src/Zcu.zig:1262-1266`), notes attached with `Zcu.errNote`
(`src/Zcu.zig:3941-3955`), inserted under the function’s `AnalUnit` in `zcu.failed_analysis`
(`src/Zcu.zig:179-194`) by returning `error.AlreadyReported`, which
`ensureFuncBodyUpToDate` turns into `error.AnalysisFail`
(`src/Zcu/PerThread.zig:2232-2259`). §8 covers the messages.

### 4.2 What it consumes

| input | where from | used for |
| --- | --- | --- |
| the function’s AIR | `pt.analyzeFuncBodyInner` (`src/Zcu/PerThread.zig:3475-3535`) | the walk: instructions, nested bodies, `extra` payloads |
| the function instance’s type | `Zcu.typeToFunc` (`src/Zcu.zig:4149-4152`) on `Zcu.funcInfo(func_index).ty` | `noalias` and the three new qualifier masks, the return type |
| the callee of each `call` | `Air.unwrapCall` (`src/Air.zig:2302-2321`), then `ip.indexToKey` | the callee’s function type (annotations) and its identity for diagnostics |
| types | `InternPool`/`Type` | pointer kinds (§1.3), `Frame` vs heap roots, aggregate/optional/error-union structure |
| source positions | `dbg_stmt` instructions (`src/Air.zig:1328-1331`), the declaration | diagnostics |
| the intern pool | `ip` | nothing is interned by the checker except the strings of its messages |

The checker does **not** read ZIR. Everything it needs from the front end has already been
flattened into AIR and `InternPool` by the time it runs — which is what “it runs after Sema”
buys.

Liveness is computed by the checker itself, with the public entry point
`Air.Liveness.analyze(zcu, air, ip)` (`src/Air/Liveness.zig:142`), rather than by changing
`codegen.wantsLiveness` (`src/codegen.zig:92-99`), so the backends’ own decision to compute
liveness stays exactly as it is, and the check works with a backend that never wants it. The cost
is one extra liveness pass over a checked function (§4.5).

Source positions are the one place where the design has a sharp edge worth stating: AIR’s only
instruction-level position carrier is `dbg_stmt{line, column}` (`src/Air.zig:1328-1331`), which
Sema omits for `comptime` blocks and for **stripped** modules (`src/Sema.zig:5932`). The checker
therefore takes the position of the nearest preceding `dbg_stmt` inside the enclosing block, and
falls back to the declaration’s location (`Zcu.navSrcLoc`, `src/Zcu.zig:4437-4442`) when there is
none. Instruction-level locations become `LazySrcLoc{ .offset = .{ .byte_abs = … } }`, which
resolves to a one-byte span (`src/Zcu.zig:1340`), i.e. a single caret; declaration locations keep
their full span. A module compiled with `strip = true` therefore gets diagnostics at function
granularity, and §12 says so in the acceptance criteria.

### 4.3 Per instantiation, inline, comptime

* **Per instantiation.** AIR exists per *function instance*: `Sema.analyzeCall` interns one with
  `ip.getFuncInstance` for each distinct parameter/comptime-argument combination
  (`src/Sema.zig:7085-7134`, `src/InternPool.zig:9695-9764`), and
  `Zcu.ensureFuncBodyAnalysisQueued` schedules each instance’s body separately
  (`src/Zcu.zig:3629-3650`). The checker is called from that per-instance body analysis, so
  `fn f(comptime T: type, x: []T)` is checked once for `[]u8` and once for `[]u16`, and the
  annotations consulted are the *instance’s* function type, whose masks Sema rebuilt for that
  instantiation (the same reason `paramIsNoalias` must be read from the instance, not from the
  generic owner).
* **Generic code never instantiated** has no AIR at all and is therefore unchecked — the same
  laziness Zig already applies to analysis, and the same rule issue #7 decided.
* **`inline` calls** are inlined into the caller’s AIR by Sema (`src/Sema.zig:7226-7470`, with
  returns turned into branches at `:18465-18471`), so the inlined body is checked as part of the
  caller. If the same function also has a non-inline instantiation in a checked module, it is
  checked there too — the same body may be checked more than once, in the contexts it is
  compiled into, and that is the honest behaviour: the facts differ per call site.
* **`comptime` evaluation** leaves no AIR; `@evalBranchQuota` and friends are invisible to the
  checker, and pointers to comptime memory are immortal, so there is nothing to check.

### 4.4 Codegen

Nothing in codegen changes, and the design keeps it that way by construction:

* the checker runs before the codegen task is created (§4.1), so a rejected function is never
  lowered;
* the checker has no codegen hook: it neither rewrites AIR nor marks it; it does not call
  `Air.legalize` and does not touch `Air.Liveness`’s output for the backend (it computes its own);
* the two consumers of AIR are unchanged: `Object.updateFunc` → `FuncGen.genMainBody`
  (`src/codegen/llvm.zig:1087-1097`, `src/codegen/llvm/FuncGen.zig:343-345`) and
  `CodeGen.generate` → `genMainBody` (`src/codegen.zig:149-179`,
  `src/codegen/x86_64/CodeGen.zig:965-992`). Neither reads anything the checker writes, because
  the checker writes nothing;
* the only way the checker could change an object file is by *rejecting*; accepted programs
  compile to identical machine code, and §12 makes that a CI check (sha256 of the object with and
  without the flag).

### 4.5 Cost, and how it is measured

Per checked function the checker does: one liveness pass (`Air.Liveness.analyze`, comparable to
what the LLVM backend already spends when it wants liveness), one linear walk of the AIR with a
worklist over nested bodies, and a state map keyed by *root*, whose size is bounded by the number
of allocations, loads-through-pointers and calls in the function — not by the number of AIR
instructions.

Measurement plan, using machinery that already exists:

1. **Time**: add a `cpu_ns_borrow_ck` field to the per-declaration timing report. `TimeReport`
   already keys `decl_sema_info`/`decl_codegen_ns`/`decl_link_ns` by declaration
   (`src/Compilation.zig:751-785`), `runCodegenInner` already times itself with
   `comp.startTimer()` and accumulates into `tr.stats.cpu_ns_codegen`
   (`src/Zcu/PerThread.zig:4440-4449`), and `--time-report` prints the lot
   (`src/main.zig:981,1890`). The checker gets the same treatment; `-fno-borrow-check` on the same
   tree is the control.
2. **Memory**: the checker reports its own footprint in the same shape as the AIR dump’s size
   header (`# Total AIR+Liveness bytes: …`, printed by `src/Air/print.zig:12-53`), so a
   `--verbose-borrow-check` run shows bytes of state per function next to the AIR size it is
   derived from.
3. **What “cheap” means**: the phase must be invisible in the *unchecked* case (it is not run at
   all) and bounded in the checked case: the acceptance budget in §12 is ≤5% wall-clock on the
   compiler’s own build with the check forced on for all of `std` and the compiler’s modules,
   which is the worst case available and far above any module a user would check.

## 5. Lifetimes across calls

### 5.1 What Zig’s types already say

| type | what it implies to the checker |
| --- | --- |
| `*T`, `[]T`, `[*]T` | a mutable borrow, for the duration of the call. Says nothing about what the callee does to the memory behind it (§5.2). |
| `*const T`, `[]const T`, `[*]const T` | a shared borrow, same duration. |
| `noalias` on a parameter | additionally: the argument does not alias any other argument (§2.4). The only existing aliasing annotation Zig has. |
| `[*:sentinel]T`, `[N]T`, `[]T` with a sentinel | a slice/array, i.e. a length; no lifetime information. `AbsorbSentinel` on `Allocator.resize`/`remap`/`realloc` (`lib/std/mem/Allocator.zig:313-441`) is about the slice’s type, not its lifetime. |
| `volatile`, `allowzero`, `align(...)`, `addrspace(...)` | no lifetime information; they change how the pointer is used, not how long it lives (`InternPool.Key.PtrType.Flags`, `src/InternPool.zig:2052-2081`). |
| the return type | nothing at all today: `*T` out means the same as `*T` in. |
| a pointer *field* of a struct | no information; a loaded pointer is a pointer loaded from memory (§1.4). |

So the derived rules of §1 are all that the types give. Everything a call can *do* to an object,
beyond using it during the call, has to be stated — that is what `owned`, `invalidates` and
`borrowed` are for (§3.4).

### 5.2 The default for an unannotated function — issue #7’s open question 2

**Decision: TypeScript’s `any` bargain.** A call into a callee without annotations is *allowed*,
and is trusted under these defaults:

> **Default call rules.** For a call to a function with no relevant annotations, the callee
> (a) may read and write the memory behind every pointer argument, until the call returns;
> (b) does not free any argument’s object and does not invalidate any pointer into it;
> (c) does not retain any pointer argument beyond the call;
> (d) returns a *fresh* object if its result is a pointer, slice, optional or error union
> containing one — not a pointer into any argument.

`extern` functions are exactly this case, and so is every unannotated Zig function in every
module, including `std` before its annotations land.

**Why not a hard boundary.** A hard boundary — checked code may only call checked (or annotated)
code — is the sound choice, and it is unusable as a first cut: `Allocator.free` itself would be
across the boundary until std is annotated, so `use after free` — the headline check of issue #7 —
could not be detected at all in stage 2. The plan puts “std checked” at stage 5 (§11); a hard
boundary would invert the plan into “annotate all of std before the checker can check anything”,
which is a much larger piece of work before any of it can be exercised, and it would also make
checked code *depend* on the annotations being right before the checker itself has been tested
against anything.

**Why it is acceptable.** Four reasons, in the order they matter:

1. **The bargain is bounded and specific.** It is not “anything goes”: the checker still enforces
   everything it can derive structurally (frame escapes, double free of objects it knows, the
   `@memcpy`/`noalias` obligation, uses through values it has provenance for). What the bargain
   gives away is only the *effects of a call on objects the checker cannot see into*.
2. **It is the direction Zig already lives in.** Unchecked Zig is where illegal behaviour already
   lives; a checked module that calls `std.hash_map` is exactly as trustworthy as `std.hash_map`
   is, which is the same trust the program already places in it at run time.
3. **The escape hatch census makes it visible.** The checker counts, per compilation, every call
   that crossed an unchecked boundary and every use of `@ptrCast`/`@ptrFromInt`/`@intFromPtr`/
   `@fieldParentPtr`/`extern`, and `--verbose-borrow-check` prints them per module (§12.6). A
   project can watch that number go down as annotations land, and a reviewer can ask why a
   particular function has thirty of them.
4. **Annotations narrow it in practice.** std’s allocator and containers are the surface that
   almost every program’s memory safety runs through, and §7 annotates them in stages 2 and 3 —
   before the checker is asked to be useful, not after. The bargain’s remaining reach is user
   libraries and the long tail of std.

A strict mode is deliberately *not* in 1.0. It cannot be bootstrapped (nothing can be checked
until std is annotated), it would double the test matrix for the same feature, and it would tempt
the annotation surface to be designed to satisfy the mode rather than the language.

### 5.3 The annotations, with examples

```zig
const std = @import("std");

// The allocator's freeing operations take ownership of what they are given.
pub fn free(self: Allocator, owned memory: anytype) void { … }
pub fn destroy(self: Allocator, owned ptr: anytype) void { … }

// Growth may relocate; std already documents exactly this.
pub fn append(self: *Self, gpa: Allocator, item: T) Allocator.Error!void { … }   // self: invalidates *Self

// An accessor's result borrows the object it was taken from.
pub fn getPtr(self: Self, key: K) ?*V { … }                                      // self: borrowed Self
```

and, from a user module:

```zig
const Buf = struct {
    bytes: []u8,

    /// Takes the buffer; the caller must not touch it again.
    pub fn handOver(b: *Buf, owned bytes: []u8) void {
        b.bytes = bytes;
    }

    /// May grow; pointers into `bytes` do not survive it.
    pub fn grow(b: *Buf, gpa: std.mem.Allocator, owned n: usize) !void {
        _ = n;
        b.bytes = try gpa.realloc(b.bytes, …);
    }

    /// The result points into `b`.
    pub fn slice(self: borrowed *const Buf) []const u8 {
        return self.bytes;
    }
};
```

The three qualifiers, one at a time:

* **`owned`** — the callee takes the object. This is the only way a *checked* function can free
  memory, and it is why use-after-free and double-free are checkable at all (§1.6). It is also
  what makes “an allocator cannot free a frame object” a compile error rather than illegal
  behaviour.
* **`invalidates`** — the callee may move or release the storage behind the argument. This is the
  annotation the *containers* need, and it is the answer to issue #7’s open question 5 (§7).
* **`borrowed`** — the result points into the argument. This is the one place where a signature
  has to *relate* two of its own types, and the first cut keeps it deliberately weak: the result
  may point into *any* `borrowed` parameter (the union, if there are several), and the result’s
  lifetime is bounded by that object’s. Weak is the right direction for it: a `borrowed` parameter
  can only make the checker *report more*, never less, and a function that returns a fresh object
  does not need it.

What remains unsaid, and is therefore in §10: **retention** (a callee that stores a pointer
argument for later — `StringHashMap.put` keeps the key slice, `lib/std/hash_map.zig:55-68`) is not
expressible in this vocabulary. Stage 4 adds it (§11), because retention is what turns a
single-call rule into a real lifetime, and that is exactly issue #7’s “lifetimes in function
signatures”.

## 6. `@fieldParentPtr`

### 6.1 What AIR says

`@fieldParentPtr` lowers to one instruction, `field_parent_ptr` (`src/Air.zig:913-915`), with the
payload `FieldParentPtr{ field_ptr, field_index }` (`src/Air.zig:1459-1462`); at runtime the
backend computes the parent address by subtracting the field’s offset from the given field pointer
(`FuncGen.airFieldParentPtr`, `src/codegen/llvm/FuncGen.zig:2546-2567`). Here is the std.Io
pattern in miniature (§9.15), the way Zig++ prints it:

```text
# Begin Function AIR: ex10.Source.stream:
  %0 = arg(*Io.Reader, 0)
  %1!= arg(*Io.Writer, 1)
  %2!= arg(Io.Limit, 2)
  %3!= save_err_return_trace_index()
  %4!= dbg_stmt(2:31)
  %5 = field_parent_ptr(%0!, 0)
  %6 = ptr_cast(*ex10.Source, %5!)
```

from

```zig
fn stream(io_reader: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
    const self: *Source = @alignCast(@fieldParentPtr("interface", io_reader));
```

Note what the AIR shows: `field_parent_ptr` is a *derivation from the field pointer*, and the
subsequent `ptr_cast` to the parent type does not change the address. The parent pointer is the
same object as the field pointer, at a known offset.

### 6.2 The rule

**`@fieldParentPtr` is a rule, not an escape hatch**, and the rule is the one the AIR states:

> `field_parent_ptr(p, field_index)` has the origin of `p`, extended with
> `.parent(field_index)`, and the same kind as `p`. Uses of the result are checked against the
> object `p` points into, exactly as uses of `p` would be.

Three consequences:

* If `p` came from a checked value, the recovered parent is checked: freeing the parent's object
  while the parent pointer is live is a use after free; letting the parent (or the field) go out
  of frame while a recovered pointer escapes is a frame escape.
* The *caller* is where the promise lives. A function that recovers a parent from a parameter is
  checked under the `borrowed` rule (§5.3): the parameter is a borrow of the caller's object, so
  the recovered parent may only be used while that borrow is live. This is exactly what a vtable
  entry does when it is called through `interface.vtable.stream(&interface, …)`.
* A `field_parent_ptr` whose operand is already `unknown` (from `@ptrFromInt` or an unchecked
  boundary) produces `unknown`, i.e. it is in the escape-hatch ledger like its operand.

### 6.3 The `std.Io` pattern, worked through

The real code, from this tree:

```zig
// lib/std/Io/File.zig:563-565
pub fn reader(file: File, io: Io, buffer: []u8) Reader {
    return .init(file, io, buffer);
}

// lib/std/Io/File/Reader.zig:17-29
pub const Reader = struct {
    file: File,
    mode: Mode,
    interface: Io.Reader,
    …
};

// lib/std/Io/File/Reader.zig:199-201
fn stream(io_reader: *Io.Reader, w: *Io.Writer, limit: Io.Limit) Io.Reader.StreamError!usize {
    const r: *Reader = @alignCast(@fieldParentPtr("interface", io_reader));
    return streamMode(r, w, limit, r.mode);
}
```

The pointer graph is: a caller owns a `File.Reader` (usually a stack local or a field of the
caller's own object); `&reader.interface` is a pointer to a `Io.Reader` field; the vtable entry
receives that pointer and recovers `*Reader` from it. Under the rules above:

1. `&reader.interface` is a borrow of the `reader` object, at path `.field(interface)`, kind
   mutable (the field is not `const`), with the region of its last use — which is the call.
2. `stream`’s parameter `io_reader: *Io.Reader` is a mutable borrow for the duration of the call,
   so the recovered `r` may be used inside `stream` (§6.2).
3. Inside `stream`, `r.mode` and everything else is checked against `reader`’s object.
4. The caller may not deinit or move `reader` while an interface pointer into it is live: if it
   does, the interface pointer is a pointer into a dead object, and using it (which is what a call
   through the vtable does) is a use after free. This is the "interface pointer escapes the
   adapter" bug that the pattern invites, and it is now a compile error:
   §9.15 shows the accepted form and its variant with the reader left live across an
   invalidation.

What the rule does *not* verify: that there is really a parent of the right type at the computed
offset. `@fieldParentPtr("x", p)` where `p` points at a `u8` inside an object that has no such
field, or at an integer-derived address, computes a pointer outside the object. The type system
cannot check this, the AIR cannot check it (the offset is subtracted at runtime), and the checker
does not try: it inherits the origin, so a *wrong* field pointer whose origin is legitimate yields
a pointer with a legitimate-looking origin. §10.2 lists it as known unsoundness, together with the
rest of the escape hatches it is related to. Note that the *comptime* case is already checked
today: `Sema.zirFieldParentPtr` verifies an interned pointer’s `base_addr` is a `.field` of the
expected parent and reports `pointer value not based on parent struct` otherwise
(`src/Sema.zig:24189-24230`).

And the `std.Io` queue case, which is a different pattern in the same file: `TypeErasedQueue`
links a stack-resident `Put`/`Get` into its intrusive lists and removes it before returning, with
its own five `@fieldParentPtr("node", …)` recoveries (`lib/std/Io.zig:2119,2126,2175,2288,2355`,
`lib/std/Io.zig:2197-2211`). Under §2.3 that is a *tainted* store that is undone before the exit,
which is accepted; the frame pointer never survives the frame.

## 7. std

Annotations are needed exactly where std’s own documentation already makes a lifetime or
ownership promise. The tables below are the whole surface for the first cut; each row cites the
doc comment the annotation encodes.

### 7.1 `Allocator` (`lib/std/mem/Allocator.zig`)

| declaration | annotation | why |
| --- | --- | --- |
| `free(self, memory)` (`:448`) | `owned memory` | “Free an array allocated with `alloc`.” VTable side: “Free and invalidate a region of memory.” (`:76-86`) |
| `destroy(self, ptr)` (`:181`) | `owned ptr` | “`ptr` should be the return value of `create`” (`:179-180`) |
| `rawFree(a, memory, …)` (`:164`) | `owned memory` | the type-erased primitive under `free` |
| `alloc`, `allocWithOptions`, `alignedAlloc`, `create` (`:202,206,260,168`) | none | a pointer-valued result is fresh by default (§5.2); `create` says “Call `destroy` with the result to free the memory” |
| `resize(self, allocation, new_len)` (`:319`) | none, and see §10.4 | “It is guaranteed to not move the pointer” — so a use of the same pointer stays legal, even though “`new_len` may be zero, in which case the allocation is freed” (`:313-319`) |
| `remap(self, allocation, new_len)` (`:358`) | `invalidates allocation` | “The allocation may have same address, or may have been relocated.” (`:343-357`) |
| `realloc(self, old_mem, new_n)` (`:400`) | `invalidates old_mem` | it delegates to remap or alloc/copy/rawFree (`:428-441`) |

This table is why the `owned` mechanism has to exist by stage 2 and not stage 5: without it there
is no free to check.

### 7.2 Containers

`std.ArrayList(T)` is unmanaged here — `std.ArrayList(T) = array_list.Aligned(T, null)`
(`lib/std/std.zig:52-55`), with fields `items`, `capacity`, `pointer_stability` and no allocator.
That is convenient for checking: the operations that can move the backing allocation already take
`*Self`.

| container and operations | annotation | source of the promise |
| --- | --- | --- |
| `ArrayList`: `append`, `appendSlice`, `addOne`, `addManyAt`, `insert`, `ensureTotalCapacity`, `ensureUnusedCapacity`, `shrinkAndFree`, `clearAndFree`, `toOwnedSlice` | `invalidates self` | “Invalidates element pointers if additional memory is needed.” (`lib/std/array_list.zig:1023-1028,1105-1110,1400-1403,1344-1347,1383-1385`); `toOwnedSlice`: “…making deinit() safe but unnecessary to call. May invalidate element pointers.” (`:744-748`) |
| `ArrayList.deinit` | `owned self` | “Release all allocated memory.” (`:689-694`) |
| `ArrayList`: `items` (the field), `allocatedSlice`, `unusedCapacitySlice` | nothing | direct field access; the field’s own doc (“Pointers to elements in this slice are invalidated by various functions…”, `:639-649`) is what the `invalidates` rows above enforce |
| `ArrayList`: `appendAssumeCapacity`, `addOneAssumeCapacity`, `appendSliceAssumeCapacity`, `initBuffer` | nothing | “Never invalidates element pointers.” (`:1031-1045,1113-1115`) — the precision tool (§9.12) |
| `ArrayList`: `pop`, `swapRemove`, `orderedRemove`, `shrinkRetainingCapacity`, `clearRetainingCapacity`, `replaceRangeAssumeCapacity` | nothing in the first cut | they invalidate *specific* elements (`:1051-1055,1092-1096,1317-1326,1328-1333`); index-sensitive invalidation is §10.4 |
| `HashMapUnmanaged`: `put`, `getOrPut`, `fetchPut`, `ensureTotalCapacity`, `rehash` | `invalidates self` | “No order is guaranteed and any modification invalidates live iterators.” (`lib/std/hash_map.zig:495-505`); `rehash`: “any existing key/value pointers into the HashMap are invalidated” (`:1343-1352`) |
| `HashMapUnmanaged.deinit` | `owned self` | frees the map’s backing; explicitly not its keys or values (`:208-214`) |
| `HashMapUnmanaged.getPtr`, `get`, `iterator` | `borrowed self` | `getPtr` returns `&self.values()[idx]` (`:1066-1078`); the iterator yields pointers into slots (`:633-661`) |
| `MultiArrayList`: `append`, `insert`, `ensureTotalCapacity`, `ensureUnusedCapacity`, `setCapacity` | `invalidates self` | “Invalidates element pointers if additional memory is needed.” (`lib/std/multi_array_list.zig:515-517`); its `Slice` is “cached start pointers for each field” (`:73-82`) |
| `MultiArrayList.deinit`, `Slice.deinit` | `owned self` | `:221-225`, `:143-147` |
| `ArrayHashMapUnmanaged`: `put`, `ensureTotalCapacity`, `ensureUnusedCapacity`, `orderedRemoveAtMany` | `invalidates self` | “Entry pointers become invalid whenever this ArrayHashMap is modified, unless `ensureTotalCapacity`/`ensureUnusedCapacity` was previously used.” (`lib/std/array_hash_map.zig:104-112`) |
| `ArrayHashMapUnmanaged.entries`, `keys`, `values`, `iterator` | `borrowed self` | “Modifying the map may invalidate this array.” (`:249-274`) |
| `ArrayHashMapUnmanaged.deinit` | `owned self` | “does not free keys or values” (`:186-196`) |
| `Deque`, `PriorityQueue`, `PriorityDequeue` | `invalidates` on growth (`ensureTotalCapacity`, `pushFront`, `pushBack`, `push`), `owned` on `deinit`, `borrowed` on `iterator` | `lib/std/deque.zig:57-100,111-151`, `lib/std/priority_queue.zig:47-48,187-197`, `lib/std/priority_dequeue.zig:349-357,408-416` |
| `BufMap`, `BufSet` | `owned self` on `deinit` (they free their stored strings, `lib/std/buf_map.zig:20-30`, `lib/std/buf_set.zig:23-30`), `borrowed` on `getPtr` (“The returned pointer is invalidated if the map resizes.”, `lib/std/buf_map.zig:62-65`) | the one place where a container owns its *elements*; §10.5 records that the checker tracks the container, not the elements |

Note the shape of the answer to issue #7’s open question 5: invalidation needs an annotation only
where the *storage* moves and the operation is not already forced to be unique by its signature;
in practice that is one qualifier on the growth operations of the containers that own storage.
Everything else is either the field access the checker already understands or a `deinit` that
`owned` describes.

### 7.3 The I/O interfaces

| declaration | annotation | why |
| --- | --- | --- |
| `Io.Reader.peek`, `take`, `Io.Reader.buffer`-style accessors | `borrowed self` | `peek`: “Invalidates previously returned values from `peek`” (`lib/std/Io/Reader.zig:506-521`); `take`: “The data returned is invalidated by the next call to `take`, `peek`, `fill`…” (`:563-571`). A later call marked `invalidates self` kills the earlier `borrowed` result, which is exactly the documented rule. |
| `File.Reader.init`/`File.Writer.init` (`lib/std/Io/File/Reader.zig:69-83`, `File/Writer.zig:38-45`) | none; the buffer is a plain slice parameter | the adapter borrows the caller’s `buffer: []u8`; the interest is the caller’s use of the adapter, which `@fieldParentPtr` covers (§6.3) |
| `File.Reader.deinit`-style teardown, `Io.File.close` | `owned self` for the I/O object’s own storage; a closed file is not memory and is out of scope (§10.4) | `lib/std/Io/File/Reader.zig` |

### 7.4 Order of work

Stage 2 lands the `owned` annotations of §7.1 and nothing else, because use after free is the
headline check and the allocator is the only thing it needs. Stage 3 lands §7.2 and the
`borrowed`/`invalidates` surface needed for invalidation. Stage 5 annotates the rest of std *and*
turns the checker on for std itself — at which point std’s own idioms (an iterator held across a
mutation, `getPtr` across a `put`) become compile errors inside std, which is the point of
checking it (§11).

## 8. Diagnostics

### 8.1 Mechanism

Each error is one `Zcu.ErrorMsg`: `src_loc` (a `LazySrcLoc`, §4.2) plus `msg` plus `notes`, where
a note is the same struct (`Zcu.ErrorMsg`, `src/Zcu.zig:1262-1266`; notes are appended by
`Zcu.errNote`, `src/Zcu.zig:3941-3955`). Rendering is the existing path: `Compilation.getAllErrorsAlloc`
sorts and clones the failed analysis units (`src/Compilation.zig:3960-4003`),
`Compilation.addModuleErrorMsg` resolves each `LazySrcLoc` — primary and every note — into a
`SourceLocation` with `std.zig.findLineColumn` (`src/Compilation.zig:4226-4293`), deduplicates
repeated note locations by blanking their `source_line` (`:4281-4293`), and
`ErrorBundle.renderToWriter` prints them (`lib/std/zig/ErrorBundle.zig:178-335`). The shape it
produces is `path:line:column: error: message`, the source line, a caret line, then one
`path:line:column: note: message` block per located note — a display of the full output of this
path is in §9.6, which is real Zig++ output for an existing check.

The three parties issue #7 asks for are: the borrow, the conflicting use, and where the borrow is
still live. Every borrower diagnostic therefore has exactly one primary message (the *use*, or the
free when the free is the offending operation) and three notes: the borrow’s creation, the
operation that ended the borrow’s validity (free/invalidation/`noalias` argument/store), and the
end of the region. When two of them share a location the renderer collapses the repeated source
line, which is why the third note in the examples below is a bare `note:` line.

### 8.2 The messages

The message text below is normative: these are the strings the implementation must produce, with
the note order shown. The paths, lines and columns are the ones the matching example in §9 would
have. `§9.n` names the example that produces each message.

**Use after free — §9.3** (a slice freed by a `defer`-registered free and used on a later path):

```text
src/table.zig:44:20: error: use of freed memory
    return buf.len + 1;
           ^
src/table.zig:33:26: note: borrow of this object starts here, at the result of 'Allocator.alloc'
    const buf = try gpa.alloc(u8, n);
                         ^
src/table.zig:38:22: note: freed here, by this call to 'Allocator.free'
    defer gpa.free(buf);
                 ^
src/table.zig:44:20: note: the borrow is still live here
```

**Double free — §9.4** (an explicit free on top of a deferred one):

```text
src/cache.zig:29:14: error: double free
    gpa.free(buf);
             ^
src/cache.zig:28:20: note: already freed here
    defer gpa.free(buf);
                   ^
src/cache.zig:26:26: note: borrow of this object starts here, at the result of 'Allocator.alloc'
    const buf = try gpa.alloc(u8, 64);
                         ^
src/cache.zig:29:14: note: the borrow is still live here
```

**Dangling frame pointer — §9.7** (stored into a caller-owned struct, never undone):

```text
src/holder.zig:12:14: error: pointer to expired local variable escapes 'store'
    holder.p = &y;
             ^
src/holder.zig:11:13: note: 'y' is declared here
    var y: u8 = 0;
            ^
src/holder.zig:12:14: note: the store into 'holder.p' is still in place at the end of 'put', and 'holder' outlives this frame
```

**Use of invalidated memory — §9.11** (an item pointer taken before an `append`):

```text
src/rows.zig:18:11: error: use of invalidated memory
    item.* = 'c';
          ^
src/rows.zig:16:27: note: borrow of this object starts here
    const item = &list.items[0];
                          ^
src/rows.zig:17:10: note: invalidated here, by a call whose argument is declared 'invalidates' by 'ArrayList.append', which may reallocate
    list.append(gpa, 'b') catch return;
         ^
src/rows.zig:18:11: note: the borrow is still live here
```

**Broken `noalias` — §9.13** (the same object passed twice):

```text
src/sum.zig:7:12: error: noalias violation: this argument aliases argument 0 of 'sum'
    return sum(x, x);
               ^
src/sum.zig:2:17: note: 'sum' declares its parameter 0 as 'noalias'
fn sum(noalias a: []const u8, b: []const u8) usize {
                ^
src/sum.zig:7:9: note: 'x' is passed here for both, and they overlap
    return sum(x, x);
        ^
```

**Invalidating a stack buffer's allocator memory — §9.10** (a frame object handed to a freeing
call), for completeness since it is the cheapest check in the set:

```text
src/stack.zig:4:14: error: 'Allocator.free' cannot free a pointer into this frame
    gpa.free(&buf);
             ^
src/stack.zig:3:9: note: 'buf' is a local of this frame
    var buf: [16]u8 = undefined;
        ^
```

The message *text* is normative; the paths, lines and columns above are the ones the example in
§9 would have. Instruction-level positions resolve to a one-byte span (§4.2), which is why the
carets are single characters here while Zig’s own diagnostics against declarations show longer
spans.

## 9. Worked examples

Twenty programs, each small enough to check by hand. They compile today — the checker does not
exist yet — which is the point: every one of them is a valid Zig program, and the checker’s
contribution is to reject some of them *when the module opts in*. The AIR shown is printed by
Zig++ (§13), not invented.

### 9.1 Accepted — allocate, use, `defer` free

```zig
fn fill(gpa: std.mem.Allocator, n: usize) void {
    const buf = gpa.alloc(u8, n) catch return;
    defer gpa.free(buf);
    for (buf, 0..) |*b, i| b.* = @intCast(i);
}
```

The exit path is `%56!= call(… 'free__func_1' …)` then `%57!= ret_safe(@.void_value)` (§1.4). The
free is a use of `buf`; the borrow ends at it; nothing after it mentions the object; accepted.

### 9.2 Accepted — `errdefer`, and the success path keeps using the buffer

```zig
fn f(gpa: std.mem.Allocator, ok: bool) !void {
    const buf = try gpa.alloc(u8, 8);
    errdefer gpa.free(buf);
    if (!ok) return error.Bad;
    buf[0] = 1;
}
```

The free exists only on the error arm, which returns; the success arm merges with the rest of the
function with the object alive (§1.4). Accepted. Note that the `defer`/`errdefer` distinction
never appears in the checker: it is the AIR’s shape.

### 9.3 Rejected — use after free, through a `defer` in an inner scope

```zig
fn sliceOf(gpa: std.mem.Allocator, n: usize) usize {
    var buf: []u8 = undefined;
    {
        buf = gpa.alloc(u8, n) catch return 0;
        defer gpa.free(buf);
        buf[0] = 1;
    }                       // the deferred free runs here
    return buf.len + 1;     // rejected: 'buf' was freed on the way out of the inner scope
}
```

The real AIR, flattened — the inner scope’s `defer` body is just instructions between the store
and the use:

```text
  %32!= store_safe(%7, %13!)                        ; buf = <the allocation>
  …
  %42 = slice_elem_ptr(*u8, %34!, @.zero_usize)
  %43!= store_safe(%42!, @.one_u8)
  %44!= dbg_stmt(5:23)
  %45 = load(mem.Allocator, %5!)
  %47 = load([]u8, %7)
  %49!= dbg_stmt(5:23)
  %50!= call(<fn (mem.Allocator, []u8) void, (function 'free__func_1')>, [%45!, %47!])
  %52!= dbg_stmt(8:15)
  %53 = ptr_slice_len_ptr(*usize, %7!)
  %54 = load(usize, %53!)
```

(`%19 = call(… 'alloc__func_0' …)`, unwrapped by `%21 = unwrap_errunion_payload([]u8, %19!)` and
stored to the local `%7`.) The borrow is created at `%19`, carried by the store to `%7` (§1.4
rule 3), loaded back at `%47` for the free and again at `%53` for the use. The first note in §8.2
is the diagnostic. This is the case the design runs on AIR for: if the checker ran before Sema, the
`defer` would be a statement to interpret; here it is two adjacent instructions.

### 9.4 Rejected — double free, a `defer` plus an explicit free

```zig
fn f(gpa: std.mem.Allocator) void {
    const buf = gpa.alloc(u8, 8) catch return;
    defer gpa.free(buf);
    gpa.free(buf);              // rejected: already freed
}
```

Two `call`s to `free__func_0`/`free__func_1` on the same object, the second on a path that the
first reaches. Diagnostic `double free`, second message of §8.2.

### 9.5 Rejected — free on one branch only

```zig
fn g(gpa: std.mem.Allocator, cond: bool) u8 {
    const buf = gpa.alloc(u8, 4) catch return 0;
    if (cond) gpa.free(buf);
    return buf[0];              // rejected if 'cond', the object is dead
}
```

The *may* analysis of §2.1: the path through the free reaches the use, so the use is a bug on that
path. The `cond_br` in AIR splits the body; the join after it has `dead` in the state set.

### 9.6 Rejected today, by AstGen — a returned frame pointer

```zig
fn dangling() *u8 {
    var x: u8 = 0;
    return &x;
}
```

Zig++ already rejects the syntactic form, and its output is the format §8 must fit into — the
first real output in this document:

```text
ex3.zig:5:13: error: returning address of expired local variable 'x'
    return &x;
            ^
ex3.zig:4:9: note: declared runtime-known here
    var x: u8 = 0;
        ^
```

The check is AstGen’s (`lib/std/zig/AstGen.zig:8252`), it runs before Sema, and it only sees the
name in the `return`. What the borrow checker adds is every form this cannot see: through a
struct field (§9.7), through a pointer loaded from memory, through an aggregate or optional,
through `@intFromPtr`, through a call that returns the address. Only one of those is in the first
cut (§2.3), and all of them share this diagnostic’s wording so that the two checks read as one.

### 9.7 Rejected — a frame pointer stored into a caller’s struct

```zig
const Holder = struct { p: *u8 };

fn put(holder: *Holder) void {
    var y: u8 = 0;
    holder.p = &y;              // rejected: 'holder' outlives this frame
}
```

The store is real and visible in the AIR: `%6 = alloc(*u8)` is `y`, and the store’s destination is
rooted at the parameter:

```text
  %0 = arg(*ex3.Holder, 0)
  %2 = alloc(**ex3.Holder)
  %3!= store_safe(%2, %0!)
  %4 = ptr_cast(*const *ex3.Holder, %2!)
  %6 = alloc(*u8)
  %7!= store_safe(%6, @.zero_u8)
  %10 = load(*ex3.Holder, %4!)
  %11 = struct_field_ptr_index_0(**u8, %10!)
  %12!= store_safe(%11!, %6!)
  %13!= ret_safe(@.void_value)
```

The store at `%12` taints the path `(Param(0), .loaded, .field(0))`; nothing overwrites it, so at
`%13` the exit check fires (§2.3). Diagnostic: third message of §8.2.

### 9.8 Accepted — frame pointers that never leave the frame

```zig
fn build(a: u8, b: u8) usize {
    const x: u8 = a;
    const y: u8 = b;
    var views: std.ArrayList([]const u8) = .empty;
    defer views.deinit(std.heap.page_allocator);
    views.append(std.heap.page_allocator, &.{x}) catch return 0;
    views.append(std.heap.page_allocator, &.{y}) catch return 0;
    return views.items[0].len + views.items[1].len;
}
```

The slices point into this frame, and they are stored in a heap object — but the object is a local
of this frame, its tainted path dies with the frame, and no exit sees it, so §2.3 has nothing to
report. Swap the local `views` for a parameter and the same program is **rejected**, which is the
conservative answer and the honest one:

```zig
fn build(a: u8, b: u8, out: *std.ArrayList([]const u8)) usize {
    const x: u8 = a;
    const y: u8 = b;
    out.append(std.heap.page_allocator, &.{x}) catch return 0;   // rejected
    out.append(std.heap.page_allocator, &.{y}) catch return 0;   // rejected
    return out.items.len;
}
```

A pointer into this frame is left stored where the caller can see it, and the callee cannot know how
long the caller keeps `out`. The §11 stage-4 summaries are what will let this version be accepted
when `out`’s own lifetime is provably inside the call.

### 9.9 Rejected — the same slice, unwrapped from an optional after it was freed

```zig
fn f(maybe: ?[]u8, gpa: std.mem.Allocator) void {
    if (maybe) |s| gpa.free(s);
    const s2 = maybe.?;         // rejected: the optional still holds the freed slice
    s2[0] = 1;
}
```

Exactly the `optional_payload` path (§1.1): the free is on `%9 = optional_payload([]u8, %0)` and
the use is on a second `optional_payload` of the same `%0` — one root, one path, so the object is
dead. The `orelse`/`if` structure is `cond_br` with an `is_non_null` predicate, and the `.?` has
its own `unwrapNull` panic path; none of that changes the borrow.

### 9.10 Rejected — an allocator call on a frame object

```zig
fn bad(gpa: std.mem.Allocator) void {
    var buf: [16]u8 = undefined;
    gpa.free(&buf);             // rejected: not an allocation of 'gpa'
}
```

`free`’s parameter is `owned`, and the argument’s root is `Frame`, so the call is the error
(fifth message of §8.2). This compiles today: `free` takes `anytype` and casts internally
(`lib/std/mem/Allocator.zig:448-456`).

### 9.11 Rejected — a slice of `ArrayList` items taken before an `append`

```zig
fn f(list: *std.ArrayList(u8), gpa: std.mem.Allocator) void {
    const item = &list.items[0];
    list.append(gpa, 'b') catch return;
    item.* = 'c';               // rejected: the append may have reallocated
}
```

The AIR shows the pointer graph the checker walks:

```text
  %0 = arg(*array_list.Aligned(u8,null), 0)
  %1 = arg(mem.Allocator, 1)
  %7 = load(*array_list.Aligned(u8,null), %5)
  %8 = struct_field_ptr_index_0(*[]u8, %7!)
  %10 = load([]u8, %8!)
  %11 = slice_len(usize, %10)
  %18 = slice_elem_ptr(*u8, %10!, @.zero_usize)
  %19!= dbg_var_val(%18, "item")
  %23 = load(*array_list.Aligned(u8,null), %5!)
  %25 = call(<fn (*array_list.Aligned(u8,null), mem.Allocator, u8) error{OutOfMemory}!void, (function 'append')>, [%23!, %1!, <u8, 98>])
```

`item`’s origin is `(Param(0), .loaded, .field(0), .elem)` and the borrow was carried through the
`load` of the `items` field (rule 3 of §1.4) — that is what makes a *container field* checkable
without any annotation on the field. The `append` argument is the same root at a shorter path, and
`append`’s `self` is declared `invalidates`, so `item` is invalid at `%25` and used after it.
Diagnostic: fourth message of §8.2.

### 9.12 Accepted — the same function, with capacity reserved first

```zig
fn f(list: *std.ArrayList(u8), gpa: std.mem.Allocator) !void {
    try list.ensureUnusedCapacity(gpa, 1);
    const item = &list.items[0];
    list.appendAssumeCapacity('b');       // never invalidates element pointers
    item.* = 'c';
}
```

and its other accepted form, re-reading the field after the growth:

```zig
fn f(list: *std.ArrayList(u8), gpa: std.mem.Allocator) !void {
    try list.append(gpa, 'b');
    const item = &list.items[0];          // taken after the last append
    item.* = 'c';
}
```

The first form is the one std already recommends (`lib/std/array_list.zig:1031-1045`,
`:1383-1390`), and it is the reason `invalidates` is a *may*: the checker does not prove that this
particular `append` grew, so the program says so with `ensureUnusedCapacity`.

### 9.13 Rejected — `noalias` argument aliases another argument

```zig
fn sum(noalias a: []const u8, b: []const u8) usize {
    return a[0] + b[0];
}

fn f(x: []u8) usize {
    return sum(x, x);           // rejected: argument 1 aliases the noalias argument 0
}
```

Both arguments have origin `(Param(0), .bytes)`; the callee’s function type has bit 0 of
`noalias_bits` set (`InternPool.Key.FuncType.paramIsNoalias`, `src/InternPool.zig:2173-2194`), and
the AIR printing shows nothing about it — this check is only possible because the *type* carries
the promise. Diagnostic: fifth message of §8.2.

### 9.14 Accepted — `noalias` with distinct objects

```zig
fn f(a: []u8, b: []u8) usize {
    return sum(a, b);           // distinct roots
}
```

Also accepted: two slices of the same array that do not overlap — `sum(arr[0..2], arr[2..4])` —
because the paths differ at the `.bytes(0)`/`.bytes(2)` step (§1.2). And accepted, but with a
diagnostic-free note in the census: an argument whose provenance is unknown (§10.2) is not
checked against the `noalias` parameter, since nothing can be derived about it.

### 9.15 `@fieldParentPtr` — the interface pattern, accepted; the eager teardown, rejected

The miniature from §6.1:

```zig
const Source = struct {
    interface: std.Io.Reader,
    pos: usize,

    fn stream(io_reader: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *Source = @alignCast(@fieldParentPtr("interface", io_reader));
        _ = w;
        _ = limit;
        self.pos += 1;
        return 0;
    }
};
```

Accepted, and grounded in real AIR: `%5 = field_parent_ptr(%0!, 0)` then
`%6 = ptr_cast(*ex10.Source, %5!)` (§6.1) — the recovered parent is the same object as the field
pointer, so every use of `self` inside `stream` is checked against the caller’s object.

Rejected variant, the classic mistake:

```zig
fn run(reader: *FileReader, io: std.Io, buffer: []u8, out: *std.Io.Writer) !void {
    const iface = &reader.interface;
    // … 'reader' goes out of scope here in the real code; modelled as an explicit teardown:
    reader.deinit();                    // (or the frame ends)
    _ = try iface.stream(out, .unlimited);   // rejected: 'iface' points into a dead object
}
```

Whether the teardown is a `deinit` call (`owned self`) or the end of the frame, the interface
pointer is a pointer into the dead object and the vtable call is a use of it: `use of freed
memory`, with a note at `&reader.interface`.

### 9.16 Rejected — `@memcpy` with overlapping slices

```zig
fn f(buf: []u8) void {
    @memcpy(buf[0..4], buf[2..6]);   // rejected: dest and source alias
}
```

`@memcpy` declares both parameters `noalias` (`doc/langref.html.in:5218`), so §2.4 applies. The
AIR shows the derivation and the check the compiler *already* emits at run time:

```text
  %7 = slice_ptr([*]u8, %6)
  %8 = ptr_add([*]u8, %7!, @.zero_usize)
  %9 = ptr_cast(*[4]u8, %8!)
  …
  %19 = slice_ptr([*]u8, %18)
  %20 = ptr_add([*]u8, %19!, <usize, 2>)
  %21 = ptr_cast(*[4]u8, %20!)
  …
  %33 = cmp_gte(%29!, %31!)
  %34 = cmp_gte(%21, %32!)
  %35 = bit_or(%33!, %34!)
  %38!= block(void, {
    %39!= cond_br(%35!, likely {
      %40!= br(%38, @.void_value)
    }, cold {
      %36!= call(<fn () noreturn, (function 'memcpyAlias')>, [])
      %37!= unreach()
    })
  } %35!)
  %41!= memcpy(%9!, %21!)
```

Both ranges come from `(Param(0), .bytes(0))` and `(Param(0), .bytes(2))` with known lengths, so
the checker rejects the program at compile time — the compile-time version of the comparison
sequence above, and the generalisation of `Sema`’s existing comptime check
(`src/Sema.zig:24844`), which requires the alias to be comptime-known. `@memmove` is unaffected:
its operands are not `noalias` (it “permits overlap”, `src/Air.zig:822-849`).

### 9.17 Rejected — a free inside a loop

```zig
fn accumulate(gpa: std.mem.Allocator, n: usize) usize {
    var total: usize = 0;
    var buf = gpa.alloc(u8, 8) catch return 0;
    for (0..n) |i| {
        buf[0] = @intCast(i);
        total += buf[0];
        if (i == 1) gpa.free(buf);
    }
    return total + buf[0];      // rejected: freed on some iterations
}
```

The loop is `loop`/`repeat` (`src/Air.zig:350-357`), the state at the loop head is the join of the
entry state and every back edge, and the free is inside a `cond_br` body the back edge reaches
(§2.6), so `buf` is `dead ∪ alive` at the exit and at `return total + buf[0]`. Note that the use
inside the loop also comes after a *possible* free on the previous iteration — the same join, one
instruction earlier.

### 9.18 Rejected — labeled break out of a loop after a free

```zig
fn f(gpa: std.mem.Allocator, n: usize) void {
    const buf = gpa.alloc(u8, 8) catch return;
    outer: for (0..n) |i| {
        if (i == 2) {
            gpa.free(buf);
            break :outer;
        }
        buf[0] = @intCast(i);
    }
    buf[1] = 0;                 // rejected: the break path freed it
}
```

The break is a `br` to the outer `block` (`src/Air.zig:1382-1385`), which makes the free reach the
block’s continuation through the break edge. The same example without the labeled break — with
`return` instead — is *accepted*, because a return ends the path that frees.

### 9.19 Accepted — `unreachable` and noreturn calls end a path

```zig
fn f(gpa: std.mem.Allocator, n: usize) void {
    const buf = gpa.alloc(u8, n) catch return;
    defer gpa.free(buf);
    if (n == 0) return;         // exit: the free runs
    if (n == 1) unreachable;    // dead end: no free, and no continuation
    buf[0] = 1;                 // the object is alive on every path that gets here
}
```

`unreachable` is `unreach` with result type `noreturn` (`src/Air.zig:1773-1798`); the AIR for a
noreturn call is a `call` followed by `.unreach` (`Sema.analyzeCall`, `src/Sema.zig:7199-7214`).
Neither reaches the continuation, so the free on the `n == 1` path cannot make `buf` dead at
`buf[0] = 1`. The AIR also shows the tail §2.6 warns about — the safety-check instructions after
the noreturn call:

```text
      %46!= dbg_stmt(5:17)
      %47!= call(<fn () noreturn, (function 'reachedUnreachable')>, [])
      %48!= unreach()
    }, poi {
      %49!= br(%43, @.void_value)
    })
```

### 9.20 Accepted, and unchecked — the escape hatch

```zig
fn f(p: [*]u8, n: usize) []u8 {
    const q = @intFromPtr(p);
    return @as([*]u8, @ptrFromInt(q))[0..n];   // provenance destroyed and rebuilt
}
```

Nothing is reported. `@intFromPtr` is `int_from_ptr` and keeps the origin as an opaque value;
`@ptrFromInt` is `ptr_from_int` and produces an `unknown` root (§1.2), so the returned slice is
unchecked — but the *census* counts both, and a project can require that number to be zero outside
a declared module. The same applies to `extern` calls, `@ptrCast` from unknown provenance, and
`@constCast` (which drops the const-ness the shared/mutable distinction was read from). This is
the boundary of the bargain of §5.2, and it is on purpose: issue #7 keeps the escape hatches, and
a checker that pretended to check through them would be lying about its own result.

## 10. What it does not catch

Stated as non-goals (not planned for 1.0) and as known unsoundness (things the design cannot
promise), each with the reason it is acceptable at 1.0.

### 10.1 The unchecked boundary

A call into a function without annotations is trusted under §5.2. Therefore:

* a callee that frees an argument’s object without an `owned` annotation hides the free;
* a callee that reallocates storage behind an argument without `invalidates` hides the
  invalidation;
* a callee that stores a pointer argument for later (retention) is invisible until stage 4;
  `std.StringHashMap.put` keeping the key slice is the canonical example
  (`lib/std/hash_map.zig:55-68`);
* a frame pointer handed to an unchecked callee may be stored in a global inside the callee, which
  the caller-side check cannot see.

Why acceptable: this is the bargain of §5.2, taken deliberately and bounded by prose, by the
annotation surface in §7 and by the census. The alternative for 1.0 — a hard boundary — cannot
check use-after-free at all before std is annotated.

### 10.2 The escape hatches

* `@ptrFromInt` (`ptr_from_int`) creates an `unknown` root: everything reachable through it is
  unchecked. `@intFromPtr` (`int_from_ptr`) keeps the origin but without a path, so an object
  whose address is taken as an integer is *escaped*: the checker stops reporting on it (it cannot
  know what is done with the integer).
* `@ptrCast` from unknown provenance is unknown; `@ptrCast` from known provenance keeps it, since
  the address does not change (`Sema.ptrCastFull`, `src/Sema.zig:22414-22496`).
* `@constCast` removes `const`, so a shared borrow becomes a mutable one and the
  shared/mutable distinction written in the type is gone (§1.3).
* `@fieldParentPtr`’s offset arithmetic is unchecked: a wrong field pointer with a legitimate
  origin produces a legitimate-looking origin (§6.3).
* `extern` calls and `assembly` (§10.3) are opaque.

Why acceptable: these are exactly the constructs issue #7 keeps as escape hatches, they are
precisely the places Zig programmers already write down as “this is the unsafe part”, and the
alternative — checking through an integer-to-pointer conversion — is not possible in a
per-function, compile-time analysis.

### 10.3 Threads, atomics, volatile, assembly

No data-race detection, no ordering, no `volatile` semantics: the checker treats `atomic_*` and
`volatile` accesses as ordinary uses and writes, which is enough for the four checks and nothing
more. `noalias` is checked within one call’s arguments, not against what another thread holds. An
`assembly` block with more than trivial clobbers is treated as an opaque use: every *escaped*
object is marked invalid, no diagnostic is derived from it, and the census counts it.

Why acceptable: race freedom is a different analysis with a different implementation strategy
(whole-program, plus synchronization reasoning), and issue #7’s first cut does not name it.
`@memcpy`’s `noalias` obligation (§9.16) is the part of this area that is cheap, and it is checked.

### 10.4 What is simply not modelled

* **Leaks.** Zig has no destructors and the checker is not a leak checker: an object that is never
  freed and never used again is fine. Issue #7 asked for use after free and double free, not for
  “must free on every path”.
* **Index-sensitive invalidation.** `pop`, `swapRemove`, `orderedRemove`,
  `shrinkRetainingCapacity`, `clearRetainingCapacity`, `replaceRangeAssumeCapacity` invalidate
  *specific* elements or a suffix (`lib/std/array_list.zig:1051-1055,1092-1096,1317-1326,1328-1333`,
  `:992-1011`); the first cut does not annotate them, so a pointer to a removed element used later
  is not reported. Modelling it needs paths with indices and a range reasoning per element.
* **Element identity through a runtime index.** `slice_elem_ptr(s, i)` with a runtime `i` has path
  `.elem` without an index, so all elements of one slice share a path: the checker is
  object-granular there, which is conservative in the reporting direction (over-reports a removal
  once removals are modelled) and blind in the accepting direction (it cannot tell two elements
  apart).
* **Unions.** `union_init` and the `optional`/`errunion` payloads are tracked structurally, but tag
  changes (`set_union_tag`) are not: a borrow of a union field stays tracked after the tag
  switches. That over-reports rather than under-reports, and an escape hatch is available.
* **`Allocator.resize(a, 0)` and `remap(a, 0)`.** Both free (`lib/std/mem/Allocator.zig:313-319`,
  `:343-357`) without an `owned` annotation, so a use after such a call is not reported. The fix
  needs a value-dependent contract (`new_len == 0`), which is a language feature, not a checker
  feature; §7.1 chooses not to annotate them so that the far more common in-place `resize` stays
  accepted.
* **Transitive ownership.** `BufMap.deinit` frees its stored strings (`lib/std/buf_map.zig:20-30`);
  the checker models the container, not the elements inside it. A pointer that came from inside an
  owned value is only followed when the AIR’s own derivation makes it visible.
* **`std.Io`’s cancellable machinery.** The `TypeErasedQueue` case is accepted by the taint rule
  (§6.3), but the checker has not been run over the whole of `std.Io`; stage 5 is where that is
  measured rather than asserted.
* **Comptime and never-instantiated generics.** No AIR, nothing to check (§4.3).

### 10.5 Why this is enough for 1.0

The claim the 1.0 milestone makes is not “memory-safe Zig”. It is: **a module can ask to be checked, and
the checker will reject the four classes issue #7 names, on the code it can see, without changing a
byte of machine code and without changing what any existing Zig program means.** The non-goals
above are the boundary of that claim, the escape hatches are documented as the boundary-crossing
mechanism, and the boundary is *visible* (the census, and the annotations themselves). A stricter
system — lifetimes in the type system, aliasing as a type property — is a different language, and
it is the one thing issue #7’s constraints rule out by requiring every valid Zig program to stay a
valid Zig++ program.

## 11. Staged implementation

Mapped one-to-one onto the plan checkboxes in issue #7.

### Stage 1 — this document (plan: design proposal)

`doc/proposals/borrow-checker.md`. Worked through in §9 as the specification the rest of the
stages are tested against.

### Stage 2 — the intraprocedural checker (plan: use after free, double free, dangling stack pointers)

Lands:

1. the module option and its chain (§3.2) plus the `test/cases` manifest key;
2. the three contextual qualifiers (§3.4), with `owned` the only one implemented as a check;
3. the borrow model (§1) and the provenance walk over AIR;
4. the hook in `PerThread.analyzeFuncBody` (§4.1) with the checker as a new module
   (`src/Air/BorrowCheck.zig`, beside `Liveness.zig`/`Verify.zig`/`Legalize.zig`);
5. the three checks of §2.1, §2.2, §2.3 and the diagnostics of §8 with the census;
6. `Allocator`’s `owned` annotations and `remap`/`realloc`’s `invalidates` (§7.1) — the minimum
   std change that makes the headline checks testable.

Tests it adds:

* `test/cases/compile_errors/borrowck_uaf_defer.zig` (§9.3), `borrowck_double_free.zig` (§9.4),
  `borrowck_uaf_conditional.zig` (§9.5), `borrowck_frame_escape_field.zig` (§9.7),
  `borrowck_optional_payload_after_free.zig` (§9.9), `borrowck_free_frame_object.zig` (§9.10),
  each with `// borrow_check=true` and the expected `// error`/`// note` lines in the manifest —
  the convention `test/src/Cases.zig:775-808` already parses;
* `test/cases/borrowck_defer_free.zig` (§9.1), `borrowck_errdefer.zig` (§9.2),
  `borrowck_frame_local_list.zig` (§9.8), `borrowck_unreachable_exit.zig` (§9.19) as `// compile`
  cases: they must keep compiling;
* `test-unit` coverage for the checker’s own pieces (the same style as the compiler’s other source
  unit tests, `build.zig:618`); *[inferred]* — the compiler’s unit tests are registered in
  `src/`-adjacent test blocks, so the checker’s path/overlap/state-join helpers get direct tests.
* a census test: two modules, one checked one not, asserting the boundary counts.

Must not regress: `zig build test` and `zig build test-cases` on unmodified code (nothing can be
rejected without the flag), and the “same machine code” check (§12).

### Stage 3 — `noalias` and invalidation (plan: `noalias` checked, and invalidation)

Lands:

1. the `noalias` check of §2.4, including `@memcpy`/`@memmove` (§9.16);
2. `invalidates` and `borrowed` in the qualifier set, and their checks (§2.5, §5.3);
3. the container annotations of §7.2 and §7.3.

Tests: `borrowck_noalias_alias.zig` (§9.13, rejected), `borrowck_noalias_distinct.zig` (§9.14,
accepted), `borrowck_memcpy_overlap.zig` (§9.16, rejected), `borrowck_item_after_append.zig`
(§9.11, rejected), `borrowck_item_assume_capacity.zig` and `borrowck_item_retaken.zig` (§9.12,
accepted), `borrowck_getptr_after_put.zig`, `borrowck_take_twice.zig` (two `Io.Reader.take`
results, the first used after the second).

### Stage 4 — lifetimes in function signatures (plan: lifetimes in function signatures)

Lands:

1. a `retains` qualifier for the one thing §5.3 leaves out — a callee that stores a pointer
   argument — and its check: a retained argument’s borrow must be live for as long as the result
   (or for as long as the callee can hold it, bounded by the result’s use);
2. **summaries for checked callees**: when the callee is a function in a checked module, the
   checker derives `owned`/`invalidates`/`borrowed`/`retains` from the callee’s analysed body and
   caches it per function instance, so a checked library no longer needs annotations for its own
   internals and the bargain only applies at module boundaries;
3. the signature forms that summaries cannot express (a result that borrows *one of* several
   parameters depending on a runtime condition), which is the point at which a lifetime
   annotation in the signature becomes worth its syntax cost rather than an inference.

Tests: retention cases (a `put`-like function whose key must outlive the map; rejected), a
summary case (a checked callee with no annotations, proving the summary is used — the same file
with the callee in an unchecked module must behave differently), and the §9.8 `out` case above,
which should become accepted once `out`’s lifetime is provably inside the call.

### Stage 5 — std checked, and documented (plan: std checked, and the language reference documents it)

Lands:

1. the remaining annotations of §7 (the long tail, driven by running the checker over std);
2. `std` and the compiler’s own modules in the test matrix with `-fborrow-check`, with a
   consequence the reader should not be surprised by: std’s idioms that the checker rejects get
   fixed or escape-hatched with a documented reason;
3. the language reference: a “Borrow checking” chapter and the `noalias` section that is a `TODO`
   today (`doc/langref.html.in:7673-7678`) becomes the normative description of the promise §2.4
   defines, plus the three qualifiers and their `@typeInfo` reflection;
4. the book chapter (`doc/book/src/`) and the `doc/proposals` → `doc/book` promotion that the
   Metal proposal does for its own feature.

Tests: std compiles clean under `-fborrow-check` (a CI job, not a unit test), the census for std
is recorded as a baseline that may only shrink, and the langref’s own `test_borrow_checking.zig`-
style runnable examples.

### Stage 6 — the 1.0 milestone

The acceptance criteria of §12 all hold, and the four checks are on by default for new projects
through a `build.zig` template line, still off for existing code.

## 12. Acceptance criteria for 1.0

1. **The four checks exist and are exact on the code they see**: the twenty examples of §9 produce
   the listed verdicts, as `test/cases` files, in CI, for `x86_64-linux` and a second target.
2. **A program that does not opt in is untouched**: the full suite (`zig build test`) is green
   with no `borrow_check` anywhere; compile times for the compiler’s own build are unchanged
   within noise; `--time-report` shows no borrow-check phase.
3. **Checked code compiles to the same machine code**: for the §9 accepted examples and for a
   full build of the compiler, the objects emitted with `-fborrow-check` are byte-identical
   (sha256) to those without it. This is the operational meaning of “compile time only”, and it is
   checkable because the check runs before the codegen queue exists (§4.4).
4. **Diagnostics point at all three ends**: every rejection carries a primary message and notes
   for the borrow’s creation, the operation that ended it, and the region — the shapes of §8.2,
   asserted in the compile-error manifests (which compare rendered lines, including `note:` lines).
5. **std is annotated and checked**: `std` compiles clean with the check forced on, with a
   recorded census of escape-hatch uses and unchecked-boundary calls; the annotation surface is
   the tables of §7 with nothing else sprinkled in.
6. **The boundary is visible**: `--verbose-borrow-check` reports, per checked module, counts of
   unchecked-boundary calls and of each escape hatch, and the numbers are part of the release
   notes.
7. **Cost is bounded and measured**: ≤5% wall-clock on the compiler’s own build with the check on
   for every module, reported with `--time-report`; the per-function state footprint is reported
   in the shape of the AIR size header (§4.5).
8. **The spec in this document is the implementation’s spec**: any deviation (a check the
   implementation cannot do, a message it cannot produce) is either fixed or moved explicitly into
   §10 before 1.0, not left implicit.

## 13. Reproducing

Everything printed as AIR in this document came from a Zig++ built from the tree this document
lives in:

```sh
# 1. a compiler from this tree, with the debug extensions that --verbose-air needs
zig build -Ddebug-extensions=true --cache-dir /tmp/bck-cache --prefix /tmp/bck-out
#    -> 0.17.0-dev.2402+zigpp.2887ca8ea   (master is 2887ca8eab)

# 2. AIR of a worked example (std from this tree, so std.lang/std.Io match)
ZIG_LIB_DIR=$PWD/lib /tmp/bck-out/bin/zig build-obj -fno-emit-bin --verbose-air /tmp/bck/ex/ex1.zig
```

`--verbose-air` is gated on `build_options.enable_debug_extensions` (`build.zig:260`,
`src/Zcu/PerThread.zig:2297`), which is why a release build of the compiler prints nothing. Each
example in §9 is one small file; the ones whose AIR is quoted here were compiled exactly as above
and re-print identically on a second run. They are not committed — this proposal changes
documentation only; the file names used in the §8.2 messages (`src/table.zig`, `src/cache.zig`, …)
are the names the examples would have, not files in this tree. The `return &x` output in §9.6 is
from the same compiler. The parse-error output in §3.4 was produced with the released Zig++
(`/tmp/ppup-live/.zigpp/bin/zig`, version `0.17.0-dev.2373+zigpp.add158e97`) as well as with the
tree’s own build.

## 14. Open questions

Resolved by this document (each is a §0 row and a section):

* the opt-in mechanism and its cost (§3), including the annotation spelling that adds no reserved
  word (§3.4);
* the checked/unchecked boundary, decided as TypeScript’s bargain with a visible census (§5.2);
* what a signature says about its pointers, and what has to be annotated (§5);
* `@fieldParentPtr`: a rule, with the `Io` pattern worked through (§6);
* which std declarations need annotations and which need none (§7);
* the semantics of a borrow, a use, a free and a merge in AIR terms (§1, §2);
* the diagnostic shape and messages (§8).

Genuinely still open, in the sense that they are decisions deferred with a reason rather than
answered:

1. **How much of std the annotations change in stage 5**, measured rather than estimated; the
   table in §7 is the plan, and the first run of the checker over std is the measurement.
2. **Whether summaries (stage 4) can replace annotations inside a checked module**, and how much
   of the annotation surface they can retire. The design expects most of it; the implementation
   will say.
3. **Value-dependent contracts** (`Allocator.resize(a, 0)` frees, §10.4) — the only way to close
   that hole without a false positive on in-place `resize` is a contract that mentions a value,
   which is a language design question of its own.
4. **Index-sensitive invalidation**, the largest remaining precision gap: paths with indices,
   ranges of a slice, and the removal operations of §10.4.
5. **Retention across frames** (§5.3, stage 4): the qualifier is easy, the interaction with
   `std.Io`’s pending queues and with async/suspension is not, and it is where the next proposal
   in this area will start.
6. **Where the checker’s own implementation should live** — `src/Air/BorrowCheck.zig` as a peer of
   `Liveness.zig` is the plan, but the analysis needs a small state map and an error assembly that
   may argue for `src/Air/BorrowCheck/` with a `Verify.zig` of its own (the Liveness/Verify/
   precedent).
7. **Whether the three qualifiers should also be callable as declarations** — a file-scope contract
   for a function whose signature cannot carry it (for instance an `extern` function declared
   through a pointer). Not needed for the first cut; likely needed for stage 5.


