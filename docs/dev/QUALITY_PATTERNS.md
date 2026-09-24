# Quality patterns

Eleven failure classes, each extracted from a root-caused production failure,
and the rubrics they imply. They are the checklist for new code and new
claims, and other pages cite them by number (`#8`). Binding rulings live in
[decisions](../architecture/decisions.md) and module placement in
[layers](../architecture/layers.md); this page is how to judge work against
them.

> **Four-GPU rule.** Every GPU verification leg runs at P = 4; a P = 1-only
> verification never suffices for landing (unit and CPU cells are exempt).
> It is #2 and #9 as an operating rule. Procedure: `AGENT_PREAMBLE.md`.

## 1. Silent symmetry: bugs invisible to every invariant you check

A corruption of ψ*(−r) preserved norms, overlaps, ⟨T⟩ and ⟨V_H⟩ exactly, by
symmetry, while destroying every τ-dependent term. **A check that passes under
the bug is worse than none. For every symmetry assumption there must be an
observable that fails if it is wrong**, and where possible the assumption is
measured rather than inferred from flags. An exoneration against a reference
produced by the same code path is circular.

## 2. Scale thresholds: correctness depends on configuration

Remainder chunks at P > 1, the first nosym deck, a planner that picked one
chunk, window divisibility, a collective collapse at 144 ranks: none was
visible at fixture scale. **Gates sample the configuration lattice, not a
point**: P ∈ {1, small, production} on square meshes (nonsquare P refuses),
sym and nosym decks, remainder-inducing sizes, every compute mode. A suite that
always resolves to one path certifies nothing about the others.

## 3. Shadow accounting: parallel bookkeeping drifts

A memory planner with three independent coefficient errors, a capability probe
that checked symbols the build did not export, a dead duplicate of the band
slices that misled a live analysis. **One source of truth wherever possible;
where a model must exist (the planner), calibrate it against measurement and
treat model-versus-measured divergence as a bug.** Every unexplained GB has a
closed form.

## 4. The optimizer defeats intent: Python intent is not HLO behaviour

A traced-index slice hoisted by XLA into a full-stack all-gather larger than
the gather it replaced; donation inert under fused jits; a zero-copy
`device_put` of a dying host buffer that aborted. **For communication and
memory, the optimized HLO is the only ground truth**: a claim that a kernel
gathers or keeps only X is unverified until a trace shows it, and every new
distributed kernel lands with its collective table. Make a constraint
structural (slice inside `shard_map`, where the partitioner cannot hoist)
rather than fighting the optimizer.

## 5. Hidden framework cost: the stack below has unpriced O(P) costs

`device_put` onto a multi-process sharding all-gathers to assert equality
([the page](device_put_hidden_allgather.md)); glibc arena retention grows RSS
with FLOPs executed; an AOT cache compiled for another machine. **Below-the-API
costs are found only by byte-exact reconciliation of observed against modelled
resources.** "Overhead" is not a resolution: the delta has a closed form and a
line number.

### Compiled-object lifetime in services

A repeated service operation owns a persistent transformed callable. Build
`jit`, `shard_map`, `vmap`, `pmap` or autodiff wrappers at module scope or in
a cached builder keyed by the complete static signature; numerical inputs stay
traced operands, and output arrays or donated buffers are never cached. A
local transformation inside an already retained outer trace is fine.

Keys separate device placement and mesh axes, layouts, static shapes, dtypes,
operation flags and native context handles whenever the builder captures them.
Hashable `Mesh`/`NamedSharding` values compare by value, so an equivalent
reconstructed layout must hit the same builder; caches that do not retain a
mesh use the service's `mesh_key`. Test equivalent reconstructions, different
placements and changed input values before claiming reuse is correct.
Repeated equivalent plans reuse compiled kernels and provider warm-up; a
solver session may own its kernels for a fixed mathematical target, with that
lifetime documented.

## 6. Broken promise: approval at resolve, failure at call

A resolver approved a SLATE Cholesky on a mesh the call rejected; a tier
approved `distributed` while its body hard-coded ScaLAPACK; `pzheevd` returned
`INFO = 0` with garbage eigenvectors on short workspace. **A capability or
geometry check tests exactly what will execute: a returned backend name is a
promise its handler runs.** Numerical contracts need strict tests: eigenpair
residuals, not eigenvalue agreement.

**No process-local deadline inside a collective region: heartbeats name the
missing party or active rank and phase; the step supervisor owns whole-step
walltime.**

## 7. rc = 0: success codes are not evidence

A −136 eV gap ran to completion; a NaN-producing solver exited 0; a
half-written ζ was indistinguishable from a complete one; P ranks overwrote
one file cleanly. **Every stage boundary carries a cheap physical-invariant
gate** (finiteness, Hermiticity, sign and magnitude identities such as the
implied V_xc, written-versus-expected counts, completion markers that are
read) **and every CLI propagates failure.**

## 8. Environment is capability, not policy

A writer router that flipped when a package appeared in the venv; a
conditioning knob that lived only in an env var; a banner at import that
pinned every CLI to one process. **Physics- and routing-relevant choices change
only through declared inputs (the deck); the environment may grant capability
but never silently selects policy.** A capability whose appearance would change
behaviour announces that flip.
[Env-var registry](env_vars.md) · [gate contract](ffi_gate_contract.md).

## 9. Claim decay: every performance claim has scope conditions

"Distributed wins everywhere" held at P ≤ 64 and inverted at P = 144; "144×
smaller gather" was true of the design and false of the compiled artifact
until a structural fix. **Record each claim with its measured domain,
re-verify at every scale jump, and correct the record in place.**

## 10. Artifact provenance: data outlives the config that made it

Restart tensors reused across a changed band window produced −135 eV silently;
a `kin_ion.h5` from a corrupted-loader era poisoned every downstream QP
energy. **Every artifact carries its generating configuration as attributes,
and every consumer asserts compatibility at load.**

**Addendum: the observable must discriminate.** A healthy run was killed after
its `zeta_q.h5` size was read as "wedged", but file size looks the same in the
healthy and failed states under either writer. The discriminating liveness
signal was the progress cadence. Before acting on a health signal, ask what it
looks like in both states; if the answer is "the same", it is not a signal.

## 11. GPU data movement: the pass you did not write is still a pass

GPU heavy loops are bound by memory traffic: a CrI3 8×8 P = 4 ζ fit spent
40 % of device time in XLA data movement against 13 % in GEMM. Each rule below
compiled to a separate HBM round trip or to uncoalesced loads. Judge by the
optimized HLO (`--xla_dump_hlo_as_text`) and the per-kernel trace, before and
after.

- **Select between gathered candidates.** `where(p, x[i], y[j])` loads both
  candidates for every element. Concatenate the candidates and gather one row
  (`i` or `n + j`).
- **Einsums with a contraction of 2 or 4.** A spin sandwich
  `einsum('ac,bd,cxmdj->axmbj', U, conj(U), d)` lowers to many tiny cuBLAS
  GEMMs plus transposes. Written as `ns²` elementwise terms it fuses with the
  surrounding gathers (GEMM time 269 → 112 ms per batch). Not bitwise: the
  term order changes by an ulp, which the ζ solve amplifies to ~1e-7 in Σ.
- **A transpose between two opaque calls.** XLA cannot fuse into cuFFT or an
  FFI kernel, so a `moveaxis` between them is a full-size copy. Move the axis
  on the producer's input only when nothing else sits between the calls; the
  lasting cure is a consumer whose load takes the layout and phase.
- **Pad with zeros, then gather.** `take(concatenate([x, 0]), idx)` (or
  `take`'s default NaN fill) is a pass over every destination cell;
  `take(mode='fill', fill_value=0)` drops the pad (287 → 213 ms per batch,
  bitwise). A scatter into zeros is not the cure: XLA materialised a 47 GiB
  temporary.
- **Conjugation as its own pass.** `jnp.conj` of an opaque call's output is a
  copy; fold it into the producer (`conj(U X U†) = conj(U) conj(X) conj(U)†`).
  Inside an XLA fusion it is free.
- **Element gathers where rows would do.** A one-element gather over a
  permuted minor axis reads uncoalesced; keep the permuted axis major or make
  the permutation block-structured.
- **Chained takes.** `take(take(x, i), j)` is already one fused gather; a
  host-precomposed full-size int32 index adds its own bytes. Precompose only
  when it removes a materialised intermediate.

## Assessment rubric

For any new distributed-physics code, in order:

1. Which invariants would a wrong implementation still satisfy? Add a check
   that it would not. (#1)
2. Which configuration axes change its code path? Gate on the lattice. (#2)
3. What bookkeeping mirrors it? Delete the mirror or calibrate it. (#3)
4. What does its optimized HLO actually move? Trace before claiming. (#4, #5, #11)
5. What does its resolve-time check not test about its call? Close the gap. (#6)
6. What does it print or return when it silently produces garbage? Add the
   invariant gate. (#7)
7. What environmental accident would change its behaviour? Make that an
   input. (#8)
8. Under what conditions were its performance claims measured? Write them
   down. (#9)
9. What artifacts does it read that could postdate their config? Stamp and
   assert. (#10)

## Refusal-gate rubric

Classify a refusal by why the rejected value cannot run: **physics** (the
requested quantity or model does not exist), **implementation limit** (it
exists but this tree cannot compute it), **stale** (no reachable input can
satisfy the predicate), **duplicate** (another owner checks the same fact), or
**over-broad** (a valid input is rejected with the invalid ones). Keep physics
and live implementation gates; delete stale gates; route duplicates through one
owner; relax an over-broad predicate only with a positive result check plus a
negative control.

Every refusal names `GATE <id>`, the deck key or API argument and observed
value (`got:`), the accepted condition (`want:`), the consequence of continuing
(`why:`), and the caller's fix. A test that only sees the accepted case is not
gate coverage: each live gate needs a control that makes its real predicate
fire.
