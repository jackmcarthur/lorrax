# First-principles quality patterns — distilled from the 2026-07 campaign

*Every pattern below is extracted from at least one root-caused production
failure or audit finding on this branch (citations → SPEEDUP_SCORECARD.md
sections). The point is not the war stories — it is that these ten classes
predicted or explained every one of the ~14 failures, so they are the checklist
against which new code and new claims should be assessed.*

> **THE FOUR-GPU RULE — every GPU verification leg runs at P=4.** A P=1-only
> verification is never sufficient for landing; unit and CPU cells are exempt.
> The owner's rationale, verbatim: *"use four gpus for 100% of all testing so
> that never ever do we run something on one GPU and then learn it doesn't
> generalize later"* — which is this page's §2 and §9 stated as an operating
> rule rather than a lesson. See `AGENT_PREAMBLE.md` at the repository root.

## 1. The silent-symmetry class — bugs invisible to every invariant you check
The nosym ψ*(−r) corruption (Q) preserved norms, overlaps, ⟨T⟩ and ⟨V_H⟩
*exactly* — by symmetry — while destroying every τ-dependent term. Months of
validation passed because the checks were all symmetric under the bug.
**Principle: a check that passes under the bug is worse than no check (it
manufactures false confidence). For every symmetry assumption there must exist
an observable that FAILS if it is wrong** — and where possible, measure the
assumption itself instead of inferring it from flags (the load-time density
gate, U). Corollary: an "exoneration" is circular if the reference was produced
by the same code path (M's kin_ion exoneration).

## 2. The scale-threshold class — correctness is configuration-dependent
The remainder-chunk z_q bug (P>1 only), the nosym branch (first nosym deck
ever), the Stage-C gather (visible only when the planner picked 1 chunk), the
8×10 rectangular-mesh rematerialization, Gloo collapse at 144 ranks, the σ-window
divisibility, the 80 Ry projector question. None were visible at fixture scale.
**Principle: gates must sample the configuration LATTICE, not a point** —
P ∈ {1, small, production}, square AND rectangular meshes, sym AND nosym decks,
remainder-inducing sizes, both compute_modes. A suite that always resolves to
one path (GN-PPM had zero multi-device coverage) certifies nothing about the
others.

## 3. The shadow-accounting class — parallel bookkeeping always drifts
The memory planner carried three independent coefficient errors (Stage-C term
missing, F_tensor_write modeling the wrong tensor, loader terms absent); the
capability probe checked symbols the build didn't export; `Meta.band_ranges`
was a dead conflicting duplicate of `BandSlices` that misled a live analysis.
**Principle: single source of truth wherever possible; where a model MUST
exist (the planner), calibrate it against measurement continuously** (the probe
harness) **and treat model-vs-measured divergence as a first-class bug** — every
"mystery" GB this campaign eventually matched a closed-form expression to the
byte, which is the standard to hold.

## 4. The optimizer-defeats-intent class — Python intent ≠ HLO behavior
The per_q tier's traced-index slice was hoisted by XLA into a full-stack
all-gather (its gather became LARGER than the one it replaced); donation was
inert under fused jits; a zero-copy device_put of a dying host buffer became a
raw-buffer CHECK abort. **Principle: for communication and memory, the
optimized HLO is the only ground truth. Any claim of the form "this kernel
gathers/keeps only X" is unverified until a trace shows it** — hence the
standing probe practice: every new distributed kernel lands with its collective
table. The structural fix for defeated intent is to make the constraint
*structural* (slice inside shard_map where the partitioner cannot hoist), not
to fight the optimizer.

## 5. The hidden-framework-cost class — the stack below has unpriced O(P) costs
JAX's `device_put` runs a silent `assert_equal` all-gather on every
numpy→multi-process transfer (7.8 GB/rank at P=64, invisible in any profile not
taken); glibc's arena retention grew RSS proportional to FLOPs executed; the
container's AOT cache was machine-mismatched; the h5py wheel is silently
serial. **Principle: below-the-API costs are found only by byte-exact
reconciliation of observed vs modeled resources.** When measured ≠ modeled,
neither "the model is roughly right" nor "overhead" is an acceptable
resolution — the delta has a closed form and a line number.

### Compiled-object lifetime in services

A repeated service operation owns a persistent transformed callable. Build
`jit`, `shard_map`, `vmap`, `pmap` or autodiff wrappers at module scope or
inside a cached builder keyed by the complete static signature. Numerical
inputs remain traced operands; never cache output arrays or donated buffers.
A local transformation inside an already retained outer trace is safe.

Keys must separate device placement and mesh axes, layouts, static shapes,
dtypes, operation flags and native context handles whenever the builder
captures them. Hashable `Mesh` / `NamedSharding` values retain their devices
and compare by value; reconstructing an equivalent layout must hit the same
builder. Existing caches that do not retain a mesh use the service's
`mesh_key`. Test both equivalent reconstructions and different placements,
as well as changed input values, before claiming reuse is correct.

Planning alone is not an exemption: repeated equivalent plans must reuse
compiled kernels and provider warmup. A solver session may own its kernels
when its mathematical target is fixed for that session; document that
lifetime rather than describing every nested transform as a cache miss.

## 6. The broken-promise class — approval at resolve, failure at call
`resolve_backend` approved slate cholesky on a mesh the call rejects (L-1);
the distributed tier's gate approved 'distributed' while the body hard-coded
scalapack (GPU-fatal); MKL pzheevd returns INFO=0 with garbage eigenvectors on
short workspace. **Principle: a capability/geometry check must test exactly
what will execute — the promise contract** ("a returned backend name is a
promise its handler runs"). And numerical contracts need STRICT tests:
eigenpair residuals, not eigenvalue agreement; orthonormality alone certified
garbage.

**No process-local deadline inside a collective region: heartbeats name the missing party or active rank and phase; the step supervisor owns whole-step walltime.**

## 7. The rc=0 class — success codes are not evidence
The −136 eV gap ran to completion "successfully"; a NaN-producing solver bug
exited 0; a half-written ζ was indistinguishable from a complete one; P ranks
overwrote one output file cleanly. **Principle: every stage boundary carries a
cheap physical-invariant gate** (finiteness, hermiticity, sign/magnitude
identities like implied-Vxc, written-vs-expected float counts, completion
markers that are actually read) **and every CLI propagates failure** — the
guard that fires on garbage is worth more than the test that passes on health.
The first production outing of the implied-Vxc guard caught a real
double-count within hours of merging.

## 8. The env-coupled-behavior class — environment is capability, not policy
The former `slab_io=auto` flipped its writer when a package appeared in the
venv (that router is now deleted); the central conditioning knob (`zeta_rcond`)
lived only in an env var; JAX reads platform
env at import time (one banner call at module import silently pinned every CLI
to one process). **Principle: physics- and routing-relevant choices change
only via declared inputs (the input file); environment may grant capability
but must not silently select policy.** Where a capability's appearance would
change behavior, that flip is announced loudly and versioned (the router
prints its decision).

## 9. The claim-decay class — every performance claim has scope conditions
"Distributed wins everywhere" was true at P ≤ 64 and inverted at P = 144
(1.75× slower + Gloo-fatal); "144× smaller gather" was true of the design,
false of the compiled artifact, true again after the structural fix; the ≥8×
bands rule predicted ~2600 kept modes where 1676 were measured. **Principle:
record claims WITH their measured domain, re-verify at every scale jump, and
correct the record in place** (the scorecard's ⚠ banners) — a claims ledger
where refuted numbers stay visible-but-marked is what let three later
workstreams avoid planning on stale results.

## 10. The artifact-provenance class — data outlives the config that made it
Restart tensors reused across a changed band window produced −135 eV silently;
kin_ion.h5 from the corrupted-loader era poisoned every downstream eqp; the
120-band dipole still feeds a 160-band window's head. **Principle: every
artifact carries its generating configuration as attributes, and every consumer
asserts compatibility at load** — the pattern is now implemented for
isdf_tensors, zeta_q and kin_ion; dipole.h5 remains the known gap.

**Addendum (self-caught, 07-26): the observable must discriminate.** The
orchestrator killed a healthy 2406c run after reading a 16.7 MB zeta_q.h5 as
"wedged" — but under PHDF5_HOST the file is eagerly allocated to full size
before chunk 1, and under the allgather writer it sits at header-size for the
entire healthy fit: file size discriminates NOTHING in either direction. The
discriminating liveness signal is the LoopProgress cadence (whose absence,
structurally, means chunk 1 never completed). Before acting on any health
signal, ask: what would this observable look like in BOTH the healthy and the
failed state? If the answer is "the same", it is not a signal. (AC §1.)

---

## 11. GPU data movement — the pass you did not write is still a pass

On the GPU paths the heavy loops are bound by memory traffic, not FLOPs: the
route-G ζ fit on CrI3 8×8 P4 spent 40 % of its device time in XLA data
movement against 13 % in GEMM (sandbox `runs/runtime/fft_gemm_budget_20260924`:
kernel → `hlo_op` → `stack_frame_id` → source line).  Each rule is a pattern
that compiled to a separate HBM round trip or to uncoalesced loads, with the
measured case.  Judge by the optimized HLO (`--xla_dump_hlo_as_text`) and the
per-kernel trace, before and after — never by the Python.

- **Select between gathered candidates.** `where(p, x[i], y[j])` loads BOTH
  candidates for every element.  Concatenate the candidates and gather one
  row (`i` or `n + j`).  Measured: the typed pair-transpose unfold (c52b2c42).
- **An einsum with a contraction of 2 or 4.** A spin sandwich written as
  `einsum('ac,bd,cxmdj->axmbj', U, conj(U), d)` lowers to several tiny cuBLAS
  GEMMs plus transposes (512 launches per route-G batch).  Written as `ns²`
  elementwise terms it fuses with the surrounding gathers: GEMM time 269 →
  112 ms per batch, 4.1 s per CrI3 fit.  It is not bitwise: the ns-term sum
  order changes by an ulp, which the ζ fit's solve amplifies to ~1e-7 in Σ.
- **A transpose between two opaque calls.** XLA cannot fuse into cuFFT or an
  FFI kernel, so a `moveaxis` between them is a full-size copy.  Moving the
  axis on the producer's input helps only while nothing else sits between
  the calls: route-G plane FFT → k-conv cost 12.3 s per fit, moving the axis
  before the FFT recovered 1.5 s, and the phase-and-split copy that remains
  goes only when the consumer's load takes the phase and layout.  Same for
  the Σ spin-rotation FFI → k-conv klead transpose (3.8 ms per τ).
- **Pad with zeros, then gather.** `take(concatenate([x, 0]), idx)` (or
  `take`'s default NaN fill) is one pass over every destination cell, most of
  them empty; `take(mode='fill', fill_value=0)` drops the pad and the NaN
  select (route-G planes gather: 287 → 213 ms per batch, 2.2 s per CrI3 fit,
  bitwise).  A scatter into zeros is NOT the cure: XLA materialised a 47 GiB
  temporary for the route-G planes and ran out of memory.
- **Conjugation as its own pass.** `jnp.conj` of an opaque call's output is
  a copy; fold it into the producer (`conj(U X U†) = conj(U) conj(X)
  conj(U)†` exactly).  Measured: the χ0 Green pair, 6.85 ms of a 55 ms χ0
  node.  Inside an XLA fusion it is free: the route-G `where(anti, conj(d),
  d)` between two gathers cost nothing, and removing it changed no kernel.
- **Element gathers where rows would do.** A gather whose slice is one
  element over a permuted minor axis reads uncoalesced; keep the permuted
  axis major or make the permutation block-structured.  The typed operator
  unfold (`_apply_unfold_phase_and_trs_local`) streams ~0.5 TB/s for this
  reason (4.05 ms per Σ τ on 2.16 GB).
- **Chained takes.** `take(take(x, i), j)` is already ONE fused gather; a
  host-precomposed full-size int32 index ADDS its own bytes.  Precompose only
  when it removes a materialised intermediate.

## The assessment rubric these imply
For any new distributed-physics code, ask in order:
1. Which invariants would a *wrong* implementation still satisfy? Add a check
   that it wouldn't. (1)
2. Which configuration axes change its code path? Gate on the lattice, not the
   point. (2)
3. What bookkeeping mirrors it? Delete the mirror or calibrate it. (3)
4. What does its optimized HLO actually move? Trace before claiming. (4, 5)
5. What does its resolve-time check NOT test about its call? Close the gap. (6)
6. What does it print/return when it silently produces garbage? Add the
   invariant gate. (7)
7. What environmental accident would change its behavior? Make that an input.
   (8)
8. Under what conditions were its performance claims measured? Write them
   down. (9)
9. What artifacts does it read that could postdate their config? Stamp and
   assert. (10)

## Refusal-gate rubric

Classify a refusal by the reason the rejected value cannot run: **PHYSICS**
when the requested quantity/model does not exist, **IMPLEMENTATION LIMIT**
when it exists but this tree cannot compute it, **STALE** when no reachable
input can satisfy the predicate, **DUPLICATE** when another owner checks the
same fact, and **OVER-BROAD** when a valid input is rejected with the invalid
ones. Keep physics and live implementation gates. Delete stale gates, route
duplicates through one owner, and relax an over-broad predicate only with a
positive result check plus a negative control.

Every refusal names `GATE <id>`, the deck key or API argument and observed
value (`got:`), the accepted condition (`want:`), and the consequence of
continuing (`why:`). A test that only sees the accepted case is not gate
coverage; each live gate needs a control that makes its real predicate fire.

## Current standing of the codebase against this rubric
- **ζ-fit pipeline**: modeled, traced, lattice-gated, provenance-guarded —
  first-principles confidence. Failures here would be infra-class.
- **I/O**: collectives enumerated, writers sharded (overlay) or being ported
  (FFI), provenance attrs in place except dipole.h5.
- **Post-ζ (W/screening/Σ) at large μ**: the named residual — replicated W and
  the eager PPM pile are unmodeled beyond 606c (fix in flight, workstream AD);
  the failure line is predictable in advance, which is the rubric working.
- **GPU lattice dimension**: several fixes verified by construction only —
  open until rtx-dev gates run.
- **Infra (Gloo/coordination at P≥144)**: irreducible, priced (~10%/run),
  cheap to retry with restart granularity.
