# Design decisions

Dated, binding rulings from the code owner. Each entry states the decision,
its consequence for code, and what it licenses deleting. Newest first.
**These override older prose anywhere in the tree**, including every page the
[register](../index.md#register) names as an owner.

An entry records a *decision*. Whether the decision has been implemented is a
separate question and is stated per entry — an approved ruling that has not
landed is marked so, with the branch that carries it, because documenting an
unlanded change as live is how a tuning table becomes a lie.

## 2026-09-04 — One owner for every mesh-padded axis; producers pad and consumers strip

`runtime.padding` owns every mesh-divisibility divisor, carrier extent, pad,
mask, authentication, and strip operation. A producer carries a `PaddedAxis`
receipt beside its plain array; `file_io.tagged_arrays` serializes the same
logical/carrier/divisor receipt at a restart seam. A consumer never infers the
logical extent from the carrier shape and never computes `% p_x`, `% p_y`, a
mesh LCM, or a mesh round-up locally. See the authoritative contract and axis
inventory in [Mesh-padded axes](padding.md).

The Σ band-window refusal is retired. Dynamic Σ is accumulated on the one
square carrier derived from both projection specs, then QP/QSGW/SC and output
consumers strip by receipt. Thus 86 physical bands on a 4×4 mesh use an
88-band carrier with two exact-zero rows and columns. ζ band tails, centroid
families, q batches, and band/r chunks follow the same producer/consumer rule.

The only remaining physics refusals concern unsupported process topology or a
solve for which padding is not inert. Distributed and symmetry services may
authenticate an already-produced carrier at their provider boundary, but may
not plan its extent. A static census covers `src/` and service source, names
every exception with a disposition, and rejects every new local arithmetic or
refusal site. The deck doctor reports the same logical/carrier/divisor receipt
as the Sigma driver; an indivisible logical window is not a refusal.

This supersedes the incomplete 2026-08-22 helper ruling and the 2026-08-06
decision declining a common band-padding owner. Licenses deleting:
`assert_sharded_sigma_window_divides_mesh`, downstream post-hoc Σ pad/strip
pairs, local `mesh_padded`/`padded_extent` helpers, mesh-modulo error hints,
and duplicate divisor parameters whose only purpose was to restate the spec.

## 2026-09-01 — COHSEX with bispinors always carries the q→0 head; `head_correction = off` is debug-only

**Ruling (owner, 2026-09-01).** A static COHSEX calculation with
`bispinor = true` always includes the Γ-cell head corrections. No end-to-end
mode may require them off or drop them silently. `head_correction = off`
exists for brute-force k-grid convergence studies and debugging, never as a
production envelope requirement.

**Why.** The charge head alone is 0.66 eV on the CrI3 static gap at 6×6
(sandbox `reports/cri3_static_head_gap_attribution_2026-08-27`), decaying
only as `1/√N_k` in 2D. A "screened bispinor" mode that forbids it is not a
more complete calculation than the scalar route; it is a less complete one.

**Consequence for the tree, implemented at `34228021`.**
`full_static_cohsex` is the one packed screened-current mode and
`charge_hall_cubature` is a refused retired spelling. The Γ completion runs
under the default `head_correction = full`; it inserts `⟨D⟩` into the bare
operator and the charge `S^{00}`/wing head into the screened operator, with
the Hall term optional and diagnostic. `off` prints `WARNING -- DEBUG` and
is recorded in the run report. `no_local_fields` is refused on every
bispinor deck because the coupled solve has no scalar diagnostic head.
The old `required_head` condition and producerless `StaticGaugeHeadResponse`
seam are absent. Physics owner:
[Four-current heads and frequency](../theory/four-current-head-corrections.md);
wiring: [Four-current wiring](four_current_wiring.md).

## 2026-08-22 — One mesh-divisibility pad helper, and it returns a NAMED result

**Superseded by the 2026-09-04 all-axis receipt ruling above.** The historical
failure mode and parity evidence below remain the reason the result is named.

`runtime.padding.pad_axis(A, divisor, *, axis, fill=0.0)` is the single
implementation of the mesh-divisibility pad. It returns
`PadAxisResult(array, logical, padded)` and a caller reads the extent it
wants **by name**.

**What this replaces, and why the obvious repair was the wrong one.** The
tree carried two helpers whose second tuple element was the *opposite*
extent from the same slot: `runtime.padding.pad_axis_to` returned the
LOGICAL extent, `bse/bse_window._pad_axis_to_multiple` returned the PADDED
one. Both were spelled `A, n = helper(...)`. A call site copied from the
wrong neighbour compiles, runs, and is wrong **only when the extent was
not already a mesh multiple** — invisible on every mesh-divisible
validated run, and the BSE helper's own comment recorded a wrong answer
that had already come from exactly this. Unifying by *picking a
convention* — silently swapping one helper's single return — reintroduces
that bug rather than fixing it, which is why the register row filed under
`bse/common` says in as many words: "Do not swap a single-value return."

So the consolidation does two things at once. The arithmetic is
single-sourced, and the ambiguity is removed: neither extent can be taken
by accident because neither is positional any more. `fill` stays
keyword-only for the same class of reason — the BSE ε axis is the diagonal
of a diagonalisation and pads with a signed sentinel
(`bse_window.PAD_EPS_GUARD_RY`), and a positionally-supplied fill lets a
call site sign the guard by accident, putting pad transitions *below* the
optical onset.

Behaviour-preserving by construction: `pad_axis` is the old
`_pad_axis_to_multiple` body (`jnp.pad(..., constant_values=fill)`, same
`round_up`, same same-object return at zero pad), and every migrated call
site binds the extent the helper it replaced returned. Pinned by
`tests/test_pad_parity_gates.py::test_pad_axis_fill_is_keyword_only_and_signed`
(both extents differ on a non-divisible case, so the old single-value
return could not have expressed it) and by a source gate that fails if a
second helper reappears under either name.

**This partially satisfies the precondition** the 2026-08-06
`LORRAX_EXTRA_BAND_PAD` entry below set out. That entry declined the knob
because "the band pad has no single source" and named `bse_io`'s
hand-rolled helper as one of the three reasons. That reason is now gone —
one helper, one arithmetic, `fill` explicit. The other two stand: the five
`spec_divisor` sites still each do their own round-up, and
`gw/ppm_sigma.pad_sigma_window` still pads m and n independently, so a
knob honoured inside `pad_axis` would still not reach the Σ window. The
knob remains declined until those are routed through a
`padded_band_extent()` as that entry specifies.

Licenses deleting: `runtime.padding.pad_axis_to`,
`bse.bse_window._pad_axis_to_multiple` and its `bse_io` re-export, and any
prose describing the band pad as having two conventions.

## 2026-08-10 — `bse/bse_io.py` is split into four modules; the old name is a facade

`bse_io.py` had become the BSE's single choke point: restart discovery and
reading, band-window authority, q=0 head injection, coarse-to-fine
densification and restart policy all in one file that had reached 2917 lines,
with five separate lanes colliding in it or re-learning it inside two days. It
is now four modules, each holding one responsibility and stating one authority
rule at the top of its own file. **`bse_window`** owns which bands are in the
window and how its axes are padded — the window a run solved is the window
everything downstream names, and the signed `PAD_EPS_GUARD_RY` sentinel, the
pad masks and counts, the `--eqp` re-slice and the eigenvector writer's
declared window are all there because they are one convention.
**`bse_head`** owns the q=0 rank-1 Coulomb head: `vhead` goes on the exchange
tile unconditionally, `whead` goes on `W` only when a real screened `W0`
loaded, one spelling of that gate serves both loaders, and `defer_whead` is
kept deliberately distinct from it — "not yet, and not as a delta" is a
different statement from "not at all". **`bse_densify`** owns coarse to fine:
the interpolation is a zero-pad in real space, it is the identity on the
coarse sub-grid, there is exactly one sharded densifier, the interpolant is
shown only the smooth body, and C1's analytic per-fine-q head channel
re-attaches Γ afterwards. **`bse_loading`** owns reading a restart into a
bundle: a tensor is refused unless the file says its data was persisted, the
q-storage question is asked once and answered once, the SlabIO and serial
transports are held to bit equality, and whether a densification is pending is
resolved before the head injection because the answer changes what that
injection does.

`bse_io.py` remains as a compatibility facade that re-exports every name it
exported before, resolving to the same function objects, so **no consumer's
import anywhere in the tree changed**. Write new code against the module that
owns the behaviour. Retiring the facade — repointing the import sites and
deleting the file — is a separate decision that has **not** been taken, and one
in-package consumer (`bse/vq_interp.py`'s lazy `from . import bse_io`) is still
on it deliberately.

Older prose that points at `bse_io.py` for a specific function should be read
against this map: the pad helpers, `resolve_n_occ` and
`write_eigenvectors_stream` are in `bse_window`; `make_w_densifier`,
`pad_W_R_to_grid`, `resolve_w_head_densify` and `build_w_head_channel` are in
`bse_densify`; `_inject_q0_head` and the `vhead`/`whead_0freq` deck keys are in
`bse_head`; the two loaders, the sharded readers and `_find_restart_file` are
in `bse_loading`. In particular the 2026-08-06 `LORRAX_EXTRA_BAND_PAD` entry
below names `bse/bse_io.py` as the home of the hand-rolled
`_pad_axis_to_multiple`; that ruling stands unchanged and its precondition now
reads `bse_window` — and, since 2026-08-22, no longer names a hand-rolled
helper at all: that half of the precondition is closed by the one-`pad_axis`
entry above.

Behaviour-preserving by construction: every function and class moved verbatim,
only module headers are new, and the structural gates that had been pinned to
`bse_io.py` by path follow the code rather than the filename.

## 2026-08-10 — The long-range channel criterion is an energy cutoff with a two-shell floor

Asked and **ruled**, 2026-08-10. This settles the open decision the
2026-08-08 BSE performance campaign left behind, which that campaign's
consolidated list carries as B5 and which its own report called "the
physics call".

The owner's words were that the criterion for `vq_interp`'s long-range
channels "should probably be an energy cutoff, but I want it to reliably
capture the first at least 2 G shells, because we need to capture G=0,
which rolls by an umklapp vector at BZ boundaries." The ruling is
therefore both halves together, and neither is decoration:

> fit channel *n* if and only if `(n·|b₃|)² ≤ E_eff`, where
> `E_eff = max(E_cut, (2·|b₃|)²·(1 + margin))`.

**What it replaces.** The fitted channel set used to be read off the keys
of `DEG_B26P`, which is to say a hardcoded `|G_z| ≤ 3` — a fixed shell
*count* spanning a cell-dependent energy window. Because `|b₃| = 2π/c`
shrinks as a slab's vacuum grows, three shells cover 0.69 Ry on the MoS2
reference cell but only 0.077 Ry once the vacuum is tripled, while the
superset's own isotropic 6.63 Ry cutoff keeps widening. Stage 1 subtracts
the full-sphere `V_LR` and stage 3 adds back only the fitted channels, so
everything in between was weight that was subtracted and never returned:
**17.93 % of the long-range weight at 3× vacuum**, and silently, because
the null that looked like it was watching this was testing the sampled
form factors rather than the fitted model. Tying the count to an energy
makes it follow `1/|b₃|` instead of standing still.

**Why the floor is not redundant.** The head slot is `argmin_G |Q+G|`, and
at a zone boundary that is not `G=0` — it rolls onto a neighbouring
reciprocal-lattice vector. A criterion that captured only the literal
first shell would leave the rolled channel unfitted, its form factor
identically zero, and the head magnitude would be multiplied away without
a word. The floor makes that unreachable for any cutoff and any cell. Two
shells rather than one is also exactly what the bulk control needs, and
the two constraints agree: on Si the long-range weight lost is 48.4 % at a
zero-shell floor, 1.37 % at one shell, and 0.000 % at two, and a third
shell buys nothing because the superset itself stops at `|G_z| = 2`.

**The default is `E_CUT_FIT = 1.0` Ry, not the 0.5 Ry of the sketch.** The
default has to reproduce today's channel set on the MoS2 reference deck,
which pins it to `[0.691, 1.229)` Ry there, and it has to hold the
3×-vacuum loss under 1 %. 0.5 Ry fails both ends: it drops the `|G_z| = 3`
channel on the reference deck — making that deck *worse*, 0.24 % → 1.90 %
— and still strands 2.04 % at 3× vacuum. At 1.0 Ry the measured loss is
0.240 % on the reference deck (unchanged), and **17.93 % → 0.290 % at 3×
vacuum**, a 62-fold reduction. The extra channels enter at polynomial
degree zero, one complex coefficient each, so the fit's solve is unchanged
and the cost is paid only in the regime that needs it.

**Implemented and landed** on `fix/vqinterp-ecut-criterion-2026-08-10`,
in `src/bse/vq_interp.py::lr_fit_degrees` — the single site that decides
the fitted set, which the superset trim, the fit and the mini-BZ
head-slot guard all now read, so they cannot drift apart. Bit-identical
in production: on the real MoS2 reference deck the criterion returns
exactly `DEG_B26P`, and the fitted model agrees with today's rule to
`max|Δ| = 0.000e+00` at every Q tested, zone-boundary Q included.

Licenses deleting: any second opinion about which `|G_z|` channels the
long-range model fits. `DEG_B26P` remains the in-plane *degree ladder*
and must not be read as a channel set again.

## 2026-08-06 — There is deliberately no `LORRAX_EXTRA_BAND_PAD`

**Superseded 2026-09-04.** The band and Σ paths now share the same owner. The
historical reasons for declining a test knob remain useful, but no longer
describe the source tree.

Asked and **declined**, 2026-08-06 (ledger 0169), even though the band
axis is now the most-padded axis in the tree.

`LORRAX_EXTRA_MU_PAD` works because every μ extent funnels through the
single `runtime.padding.padded_mu_extent`, so one knob reaches all
eleven call sites. **The band pad has no single source.**
`spec_divisor` is shared by five sites (`wfn_loader`, `mtxel_sweep`,
`sc_iteration`, `qsgw_density`, `wfn_transforms`) but each does its own
round-up; `bse/bse_io.py` hand-rolls `_pad_axis_to_multiple` entirely
(and returns the PADDED extent where `pad_axis_to` returns the LOGICAL
one, from the same slot); `gw/ppm_sigma.pad_sigma_window` pads m and n
independently.

*(The second of those three was closed on 2026-08-22 — one `pad_axis`
with a named result; see that entry above. The other two stand and the
knob stays declined.)*

A knob honoured inside `pad_axis_to` would therefore reach **neither the
BSE band axis nor the Σ window** while reporting a green pad-flip run for
the band axis as a whole — a false all-clear, which is worse than no
check because it stops anyone else looking.

**Precondition for adding it:** single-source the band extent first
(a `padded_band_extent()` beside `padded_mu_extent()`, with `bse_io` and
`ppm_sigma` routed through it). The knob is cheap and correct after that
and manufactures a false all-clear before it.

*Moved here from `docs/dev/env_vars.md` on 2026-08-06: that page's register entry restricts it to spelling, default, class and parse grammar, and this is a ruling.  The registry row for `LORRAX_EXTRA_RANK_PAD` links here.*

## 2026-08-06 — `minimax` is the only screening method; `ctsp` is refused

`screening_method` accepts exactly one value, `minimax`, which is also its
default. Any other value raises at parse time.

The specific thing this closes: `screening_method = ctsp` **parsed,
normalised, and ran minimax**. The field was pure decoration — the spelling
never selected a different method, so every deck carrying it has been running
minimax all along and replacing the key (or deleting it) changes no result.
That is worse than an unsupported option, because the deck, the run log and
the provenance record all agreed on a method the code never had.

Implemented `c6b6aa0`; the refusal text says all of the above, so a deck
author reading the error does not have to come here. The one fixture whose
name asserted `ctsp` is renamed.

Licenses deleting: the `ctsp` spelling, its normalisation, and any prose
describing LORRAX as having two screening methods.

## 2026-08-05 — The Lustre stripe count is the aggregator count, so it is `nranks`

*(APPROVED. **Implemented on the Python side in this branch; the C++
writer still defaults to 16** — see the status note.)*

`LORRAX_PHDF5_STRIPE_COUNT` is not a filesystem-layout preference that
happens to affect speed. ROMIO sets `cb_nodes = min(striping_factor,
nranks)`, so **the stripe count IS the collective-buffering aggregator
count**. A fixed 16 therefore caps aggregation at 16 aggregators no matter
how many ranks write, which is exactly backwards for a design envelope of
hundreds of ranks. The default becomes `nranks`.

Why the existing measurement did not catch it: the sweep that chose 16 ran at
4 and 16 ranks, where `nranks <= 16` makes the two policies nearly the same
choice. A default tuned inside the region where it cannot be distinguished
from its replacement is not evidence for it.

Shape of the replacement: clamp to roughly [4, 128], ramp the striping unit
1 → 4 MiB with it, and **refuse a negative count** — a negative
`striping_factor` means "every OST on the filesystem", the
maximum-*contention* layout, and the current Python parse passes it straight
through.

**Status, re-verified 2026-08-06 after the merge.** `e5c9618`
(`feat/slab-io-stripe-nranks-2026-08-06`) **is now an ancestor of
`integration/2026-08-06`** — `merge-base --is-ancestor`, checked. It is
**not** on `origin/main`. *(This paragraph previously said the commit was
not an ancestor and that "both sites here still default to 16"; that was
true when written and is now wrong in both halves.)*

What actually landed, in `file_io/_slab_io_ffi.py`:

* `_stripe_policy(nranks)` is a pure function of the rank count — no env,
  no MPI, no filesystem — returning `count = clamp(nranks, 4, 128)` and a
  striping unit ramped 1 → 4 MiB with the rank count by exact integer
  comparison. `LORRAX_PHDF5_STRIPE_COUNT` unset now means *the policy*,
  not `16`.
* **The negative count refuses**, naming the measurement: `-1` means
  "every OST on the filesystem", which is the maximum-*contention* layout
  and measures like one — 0.105 GiB/s at 64 ranks / 32 GiB against 10.63
  for the policy (job 56389339).
* `_stripe_size_bytes` moved to sit beside `_stripe_count`, because the
  two were previously resolved in two modules with two different notions
  of "the default".

**Still open, and it is the same hazard the entry named:** the C++ writer
`src/ffi/cpp/phdf5/context.cc:463` carries the literal `"16"` as its
fallback, so with the variable unset the two writers now choose
*different* layouts — which is worse than both being wrong the same way.
One environment must mean one layout in every writer. Until that is fixed,
`LORRAX_PHDF5_STRIPE_COUNT` should be set explicitly for any run whose
numbers matter.

Licenses deleting: the fixed `16` default wherever it is still restated as
the Python-side value.

## 2026-08-05 — An allgather is a refusal, not a fallback

Any I/O route whose cost is "gather the whole global array onto one rank"
is **not a slow tier the system may fall back to**. The design envelope is
arrays that need hundreds of GPUs to hold, so a rank-0 gather does not buy
a slow run — it buys an out-of-memory some minutes later, behind a banner
nobody read. A slow correct path is a fallback; a path that cannot complete
at the design size is a refusal that has not been written yet.

Implemented 2026-08-06 in commit `0d8e50c`, as one rule covering every
route into the tier:

> `H5PY_ALLGATHER` is reachable at exactly one process, and nowhere else.

At P=1 the gather and the per-rank write are the *same operation*, so there
is nothing to refuse about; above P=1 both routes raise at parse time.

**SUPERSEDED the same day — the tier is deleted, not refused.** The rule
above was landed at seven separate doors over three sessions, each landing
reported as complete, and an eighth ungated route (a direct import of the
backend module from `gw/gw_init.py`) survived all seven. A tier that must
be refused at seven doors is not a tier; it is dead code wearing a safety
label. `H5PY_ALLGATHER` and `PHDF5_HOST` are both gone, along with the
`slab_io` deck key, the `use_ffi_io` boolean, the `SlabIOBackend` enum and
the `auto` router. There is one transport, so the contract holds by
construction. **Amended 2026-08-27:** one geometry now has a second
backend, and it is still not a choice — on an EMULATED mesh (`P == 1` with
more mesh cells than processes, `common.collectives.mesh_is_emulated`) the
phdf5 open refuses, and `SlabIO` constructs
`file_io._slab_io_serial._SerialBackend` instead. It is selected from a
predicate before any transport is built, never from a transport that
failed; it takes no deck key, no env var and no argument; and it refuses
above one process, off a CPU mesh, and outside `w`/`a`/`r`. The property
the paragraph above is defending — that no caller can *select* a tier —
is unchanged; what changed is that "one transport" is now "one transport
per geometry, with the geometry read from the mesh". See [`slab_io.md`](slab_io.md#tiers-history) for the
per-tier evidence, including the `nm -D` measurements that showed
`PHDF5_HOST`'s only selection condition to be false on every deployed
library.

Consequences already landed: the silent demotion from the parallel FFI
writer is deleted, as is the decline branch that read
`max(SLURM_JOB_NUM_NODES, SLURM_NNODES)`.

This refines the 2026-08-01 entry below, which licensed auto-demotion
"where the alternative is a different *service tier* (e.g. the h5py write
route on launches where MPI cannot bootstrap)". That licence no longer
extends to the allgather tier above one process: if MPI cannot bootstrap at
P>1, the correct behaviour is to refuse and say so, not to write the file a
way that cannot work at scale.

Licenses deleting: demotion branches into rank-0-gather I/O, and the
announcement machinery that existed only to narrate them.

## 2026-08-04 — Padding is SlabIO's business, not the caller's

A caller states LOGICAL shapes only. Physical padding, mesh divisibility and
pad rows are implementation details SlabIO owns and must never require the
caller to reason about.

* `write_slab(name, A, offset=...)` accepts any `A`. If the backend needs a
  mesh-divisible physical extent, SlabIO pads internally. `valid_shape`
  defaults to A's logical extent clipped to the dataset and becomes an
  OVERRIDE for the ragged-chunk case, not a routine argument.
* `read_slab(name, shape=...)` returns exactly `shape`. Any padded extent the
  backend needs is read and trimmed internally; a caller never sees a pad row.
* Bounds are tested ONCE, on the logical slab `offset + valid_shape`. That is
  a replicated quantity, so every rank reaches the same verdict. Testing a
  rank-local advanced offset is forbidden: it splits the ranks into those that
  refuse and those that enter the collective, which strands the communicator
  with no HDF5 error and no traceback (measured: 306 s hang at P=4, silent
  420 s timeout on the read path).
* No rank may skip a collective because of its own error. Record the error,
  participate in the collective teardown, then raise.
* `create_dataset` on an existing dataset: identical logical shape and dtype
  ⇒ reuse (idempotent). Anything else ⇒ REFUSE, naming both shapes. Never
  silently delete-and-recreate (data loss) and never silently write into the
  previous geometry (wrong physics, no symptom).

CLIPPING IS SILENT AND THAT IS INTENDED (owner, 2026-08-04). A logical slab
that overhangs the dataset is clipped to the dataset extent and writes a
slightly smaller version of the same array. No warning. The overhang case and
the ordinary pad-row case are indistinguishable from inside SlabIO, and the
owner does not want a diagnostic for a path that is robust — a warning nobody
can act on is noise that trains people to ignore the log. An EXPLICIT
`valid_shape` override still refuses, because there the caller has stated an
intent that can be contradicted.

Rationale: every SlabIO defect found on 2026-08-02/04 — the wholly-padded
rank refusal, its residual asymmetric-bounds twin in both directions, the
mode-ignoring context cache, the ragged `valid_shape` hazards — is the same
root cause: padding leaked into the caller's contract, so correctness
depended on every call site independently getting it right.

Consequences: the host backend's absent divisibility check stops being a
backend divergence, because divisibility is no longer part of the contract.
`ensure_dataset` gains a shape/dtype check, which will newly refuse `mode='a'`
reruns at a different mu — those were silently writing into the old geometry
and were never correct.

Licenses deleting: caller-side pad/unpad arithmetic that exists only to
satisfy SlabIO, and per-call-site `valid_shape` computation that merely
restates the logical extent.

## 2026-08-04 — The TRS veto is about k-grid FFT sums, and only those

Scope of the standing veto below ("No time-reversal-symmetry-based
reductions"), given by the owner 2026-08-04 because the wording read
absolutely and the code did the opposite:

* **FORBIDDEN** — using TRS to reduce sums over k for the **self-energy and
  related observables that are evaluated by FFT over the k-grid**. Those
  convolutions need the whole grid; halving them on an antiunitary
  identity is the case the veto exists for.
* **ALLOWED, AND PREFERRED** — TRS for every sum NOT strategically done by
  k-grid FFT. The charge density, the IBZ self-consistent update, matrix
  elements and k-weighted accumulations are all in this class.

So the standing entry is a statement about ONE class of sum, not about the
symmetry. It was never violated: `SymMaps.trs_allowed` defaulting to True,
`unfold_psi`'s TRS branch, and reading a `nrk=10` WFN are all outside the
forbidden class — unfolding 10 k to 16 is an EXPANSION, and ends in
full-grid work.

Why the wording was absolute: TRS is antiunitary, so the rule has two
halves — the `iσ_y·conj` spinor factor and the negation of the G list —
and they are applied in *different* places (`unfold_psi` and its caller).
Applying one without the other replaces ψ(r) by ψ*(−r), which is norm-,
orthogonality- and ⟨T⟩-preserving and therefore **invisible to every cheap
check**, while being wrong by O(100 eV) in V_loc/V_NL (the scorecard §Q
bug). The veto was a burn, not a physics objection.

Licenses: an IBZ-reduced self-consistent update on a TRS-reduced k-set
(`docs/dev/ibz_self_consistency_scaffold.md`); `gw.qsgw_density` consuming
`WfnLoader.kweights` on the IBZ. Does NOT license folding χ⁰, W or Σ over
±q.

## 2026-07-30 — D10: fixed-shape ngkmax G loading (ratified 2026-08-04)

G-vector counts differ per k-point. Every per-k kernel takes the loader's
own fixed `(n_k, ngkmax, 3)` table rather than a ragged slice back to each
k's `ngk`, so every k presents identical operand shapes, the kernel lowers
ONCE instead of once per distinct `ngk`, and the k sweep becomes dispatches
of a single executable that `collectives.sweep_local_k` can pipeline behind
one host readback. Padded-vs-ragged agreement is gated at 1e-12 relative,
not bit-exactness (`tests/test_kin_ion_padded_gvectors.py`,
`tests/test_psp_padded_gvectors.py`), because appending exact zeros does not
change a sum but does change XLA's reduction blocking.

PROVENANCE NOTE, recorded because it was the reason for ratifying late: this
decision was cited as binding in code (`gw/kin_ion_io.py`,
`common/collectives.py:938`), carried a named tolerance (`RTOL_D10`) and
gated two test files, but was never written in this register — so there was
nowhere to read it in order to sign off. Owner confirmed it 2026-08-04. A
code comment must not be able to mint an "owner decision"; cite an entry
here or do not use the phrase.

Licenses deleting: ragged per-k G slicing in production consumers.
`psp.dft_operators.generate_gvectors_k` stays ONLY as the reference route
the D10 gates compare against.

## 2026-08-01 — FFI backends are required, not optional

The FFI layer (FFT, GEMM, distributed linear algebra, parallel HDF5) is an
essential part of the build. Where a certified FFI path exists, the native
JAX fallback path is not maintained and may be deleted; a missing or
unloadable FFI library is a refusal at startup (with the library named),
not a silent demotion to a slower path. Auto-demotion remains only where
the alternative is a different *service tier* (e.g. the h5py write route
on launches where MPI cannot bootstrap), never a duplicate compute path.

The code must still run on one process: the required libraries must build
and load for P=1 (they do — the host library has no MPI hard dependency
for the FFT/GEMM handlers, and the fastloop gate runs the full chain at
P=1 with the FFI stack on).

Licenses deleting: the XLA fft path inside the flat-k consumers and its
gate arms; the ungated-vs-gated dual factories in `fft_helpers`; every
"if FFI enabled ... else jnp" fork where the FFI side is certified.
Does NOT license deleting: the vendor-portability fallbacks *inside* the
FFI handlers (plain-loop CBLAS, FFTW-vs-MKL resolution) — those are how
the required layer stays buildable everywhere; and BSE's XLA FFTs, which
have no FFI route yet.

## 2026-08-01 — Square process meshes only; nonsquare P REFUSES

Only square 2-D device meshes are supported. A device count that is not a
perfect square **refuses**, naming the two nearest square counts to
request. Launch scripts should request square counts.

Rationale: rectangular meshes complicate ScaLAPACK grid geometry and the
divisibility contracts for no measured benefit at our scales (ruled
2026-07-27).

AMENDED 2026-08-06. This entry previously said the resolver truncates to
`s = floor(sqrt(P))`, idles the surplus, "and does not refuse". **The
implementation refuses, and has since before the entry was written** —
`common/collectives.py:289-300`. The truncation safety net is deliberately
NOT implemented, because idle ranks cannot be made deadlock-free without
deep surgery:

* under the production transport `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`,
  communicator creation is `MPI_Comm_split`, collective over
  `MPI_COMM_WORLD` — a rank outside the mesh that never executes the
  warm-up jits leaves every in-mesh rank blocked in the split;
* `psum_replicate` and the k-sweep gathers assume mesh size == process
  count;
* `multihost_utils` barriers span the WORLD, so idle ranks would have to
  replay the entire driver control flow in lockstep while skipping every
  mesh-touching jit.

A refusal that names the square count is the deadlock-free form of the same
rule. The record is reconciled in favour of the implementation; the
"truncate" wording is withdrawn rather than kept alongside.

CONSEQUENCE, worth stating because it has already cost a plan: the reachable
device counts are 1, 4, 9, 16, 25, 36, … A scaling ladder cannot include
**32**, which is the natural next rung on a 4-GPU-per-node machine (8 nodes).
One has already been re-planned around it. Request 16 or 36; a run submitted
at 32 refuses at mesh resolution, before any work.

Licenses deleting: rectangular-mesh accommodation in the mesh resolver
and any divisibility contortions that exist only to serve non-square
grids. Does NOT license adding idle-rank truncation.

## 2026-09-05 — One in-memory centroid order (orbit-packed); files canonical

Every centroid axis a gwjax run computes on is in the orbit-packed order of
`common.grouped_layout` (whole symmetry orbits per X/Y shard, per-shard zero
pad suffixes), owned by `common.centroid_basis.PackedCentroidBasis` and
carried as `meta.mu_basis`; `meta.n_rmu_padded` is its packed extent.  No
kernel between the I/O seams converts between orders, and no second
in-memory order exists: the parent-k Green contraction, the ζ-fit tiles, the
τ chain, V, χ0, W and the static kernels all compute in it.  A conversion is
legal only where bytes cross a file boundary (readers pack, writers unpack;
`common/centroid_basis.py`), using an all-to-all round trip per sharded axis.
Files keep the canonical centroid-file order at logical extent, so restart
files and the MPA store are processor-grid agnostic. Canonical suffix padding
belongs only to I/O staging. The seam handles staging carriers both smaller
and larger than the packed extent; BSE, htransform and downfold read the
same logical files.

Consequences the pads impose: dense μ solves run at the whole packed extent
with C_q's physical mean diagonal on its pad slots (`meta.mu_solve_extent`),
and any
"logical prefix" test on a μ axis is a defect; use `meta.mu_active_mask`.
The test-only `LORRAX_EXTRA_MU_PAD` knob still sizes the canonical carrier
(I/O staging) and no longer changes the orbit-packed runtime extent.
`PackedCentroidBasis.solve_axis` supplies the solve receipt; the loader and
q-table construction consume that basis's extent without padding it again.
Bispinor charge and current families now use the same orbit-packed owner.
The mandatory parent plan requires closure under its computational typed actions.
For a nonclosed charge or current centroid set, the driver selects one
`SymMaps.trivial_view()` before building either packed basis or q policy.
The same parent kernels then consume loader-unfolded full-k states as parents
(`n_parent = nk`), and every q row is unreduced. A warning names the original
centroid file and recommends orbit-closed kmeans. The original loader keeps
its authenticated symmetry for G-sphere unfolding, file energies and file-wedge
serialization; the computational view does not change the WFN's physical
symmetry verdict. Historical centroid sets are not regenerated. The deleted
full-k kernels remain deleted. This owner ruling supersedes the earlier
nonclosed-centroid refusal (claims1282/1300); measured restoration is tracked
in the bispinor parent-route report.

## 2026-08-18 — ζ band chunks default to 16; zero opts into the planner

The no-key `band_chunk_size` value is 16, mesh-rounded and capped at the
logical ζ window.  This was selected from a pre-AOT Si 80 Ry P=4 measurement
of 33 ms steady z_q at bc16 versus 46 ms for full-window transport.  The
required final-tree A/B reversed that result: with the merged SM80 AOT
kernel, bc16 measured 31 ms and full-window 21 ms.  The owner retained bc16
as the no-key policy after that refutation; it is not a current performance
claim.

An explicit `band_chunk_size = 0` retains the full-window-first memory-planner
ladder as an opt-in.  A positive value retains its override semantics.  The
physics band window is unchanged in every mode; mesh pad bands are exact zero.

The ψ(r) cache remains one rectangular, all-P band-sharded `lax.scan` result.
At a 50-band window, bc16 therefore stores 64 slots.  A ragged tail would split
the cache/slice ABI into another compiled module family, so removing the 28%
pad is not licensed as a trivial accounting correction.  The memory model must
price the pad exactly and identify it in its documentation.

The tempting route-only Stage-C correction is not trivial.  Although
conv_kpair removes the old FFT/transpose/product chain inside the post-pair
operator, XLA's enclosing scan/custom-call live set still requires the
three-slot GPU BufferAssignment bound.  A two-slot trial admitted a bispinor
P=4 r-chunk with a 23.40 GB estimate, after which the executable requested a
31.985 GB arena and OOMed.  The conservative three-slot accounting therefore
stays until a compiled-memory query can replace it; no route-shaped estimate
is inferred from the CUDA kernel's internal scratch alone.

## Standing (recorded earlier, restated for one-page reference)

- Thousands-of-low-memory-processes is the scaling target: no N_mu^2-class
  object may be required to fit on one rank in the large-P limit
  (2026-07-27). Two plans per solve family: a local whole-tile plan and a
  distributed plan; execution schedules of a plan are not new plans.
- No time-reversal-symmetry-based reductions; no DFT-as-matmul (standing
  vetoes).
- Every performance claim carries a job id and an on-disk artifact.


## 2026-09-05 — covariant four-current parent route (owner-confirmed ruling)

Four-spinor transport uses the symmetry service's `diag(U₂, det(S) U₂)` action;
Lorentz blocks use its scalar centroid transport followed by `Λ ⊗ Λ`, with
`Λ = diag(1, polar time-odd Cartesian action)`. Vertices act after child unfold.
The experimental TT Ward proxy subtracts the Γ row on q-IBZ before Dyson and
star transport. Its full-q counterpart subtracts the transported contact, not
a constant unphased Γ matrix. The incumbent-versus-covariant MoS2 QP/sector
price remains an explicit acceptance measurement. The owner confirmed this
placement and rejected a centroid-diagonal contact: real-space locality does
not imply diagonality in the centroid representation.

The owner permits temporary `low_mem_bands` acceptance on the single
parent carrier: false warns that the full-k carrier no longer exists and
proceeds on parents. Refusal-by-name remains the owner's pending ruling.

The shared static Sigma consumer is `contract_lorentz_blocks`; packed X/SX/COH
and bare TT exchange call it with integer Lorentz indices. Persistent faces
are raw parents, vertices follow typed endpoint transport, and sectors sum
before band unfold. Distributed face GEMMs retain their necessary native
communication; no-HLO-collective evidence certifies additional symmetry
transport only, not absence of GEMM panel exchange.

SC density uses raw WFN IBZ rows, file k weights, and the symmetry service
for scalar-grid and polar-current projection. U/E select `kirr_fullids`; only
completed Hartree band matrices unfold. The transverse DFT parent bundle
rotates with the same iteration U/E as charge and reaches bare TT exchange.
`rho_from_wfns` accepts `sym` alongside `sym_perm` because scalar grid
pullbacks alone cannot project a vector current.

2026-09-05 BISP-ORCH: raw-parent transport no longer evaluates the former hardware/band-count profitability score. `parent_k_contraction_profitable` and `_resolve_parent_green_admission` are removed; exact typed transport is the required route under the one-order decision. Final carrier deletion is tracked by the bispinor parent-route report.

GW admission requires typed parent transport even for unreduced one-band decks.
`low_mem_bands` defaults to true; an explicit false prints a warning and uses
parents. Refusal-by-name remains pending. Non-RPA screening and old full-k
restart carriers refuse before sampling/loading; neither selects a fallback.

The bispinor parent route retires `LORRAX_FORCE_FULL_BZ` and explicit
`restart_q_storage = full`. Storage follows the computed q parents; a
naturally unreduced WFN still has a full q axis. Historical full-q readers
and the internal screening probe's full-q oracle remain shared functionality.

2026-09-06 BISP-ORCH: dynamic Sigma and invalid-mode static tails take their operands only from `parent_sigma_operands`; legacy/full-k and per-bracket carrier selection is retired. The Green GEMM boundary restores its declared two-axis sharding after eager arithmetic, including singleton meshes, without weakening the service guard. Scoped evidence: claims1036–1038 and1043; Mo dynamic selector regression is claim1025 (commit0b77e17c misnumbered it1024 but names the correct physical step). Broader carrier deletion remains tracked in the campaign report.


### 2026-09-06 — parent-only dynamic Sigma contraction

GW tau and static-limit spatial factories require the typed parent plan and canonical face shapes. Band brackets use masks over those resident parents. The retired full-k/split-channel projection wrappers and slicing loops have no production caller; the shared `common.contract_bands` primitive remains for BSE and its direct tests. Bracketed stage-timing remains explicitly unsupported.


### 2026-09-06 — remove the four-copy GW carrier

`Wavefunctions` no longer has psi_xn/psi_xr/psi_yr/psi_yn, their accessors or the old builder. Its persistent GW samples reside in `ParentGreenCarrier`. The two optional face fields remain for bounded head-star children and independent numerical oracles; `gw_init` does not populate full-k faces. Receipts inspect the two supported orientations and parent samples. The unused amplitude envelope and real/imaginary projection wrapper are retired; BSE/htransform shared raw-array services remain. The obsolete full-k persistence gate is replaced by `bispinor_parent_faces_gate.py` and `bispinor_parent_restart_gate.py`.


### 2026-09-06 — head attribution is opt-in

The existing `sigma_freq_debug_output` switch alone enables head-attribution diagnostics and their output. Physical Γ completion is independent of that switch. Each enabled endpoint-family class reuses its body Green function for the Γ-only attribution. The subsequently required S phase-2 producer materializes its class interaction stack before the compiled vertex scan; the earlier one-full-q-block residency statement no longer describes that consumer.


### 2026-09-06 — fresh GW fits require raw parents

`fit_zeta_to_h5` accepts only the typed parent plan and its two packed faces; full-k/single-axis fit operands and `low_mem_bands` dispatch are removed from the fit API. Charge reuse may omit fit-time inputs because it skips the fit. Shared downfold Cq and BSE/htransform Galerkin services remain separate. The outer `gw_jax.zeta_fit_transverse` interval measures current-fit wall time without summing overlapping channel timers.

- 2026-09-06 BISP parent route: `z_q_from_psi_sm` and `fit_one_rchunk` accept only typed parent faces and orbit-tile tables. Retired slab/full-k operands, normalization copies and layout/cache selectors are deleted. Direct NumPy q/band sums replace retired-kernel test comparisons; GW mesh parity covers square P1/P4 with explicit rectangular admission refusals. Shared Cq downfold and Galerkin consumers are retained.

- 2026-09-06 BISP parent route: the full-face Cq implementation is deleted; face operands require a typed parent plan. The existing rectangular downfold Gram remains under `c_q_from_psi_sm` and is tested against direct NumPy sums, including rectangular matrices. No shared BSE/htransform loader or Galerkin path is removed.

## 2026-09-06 — Covariant Gamma photon completion

The owner requires the Gamma completion itself to obey the full authenticated little group, even when incumbent fixed-main does not. Average the products of typed transported rank-four factors after the shared coupled head solve/cubature transaction. Keep fixed-main labelled as the incumbent and price the changed QP/sector rows separately. The one owner is `head_correction._photon_q0_factor_orbit`; `StaticPhotonQ0FactorCarrier.family_plans` is metadata needed to give optional diagnostic attribution exactly the same action. This extends the frozen interface under the new ruling and introduces no deck key or dense projector. One active orbit pair has O(group_size * 4 * packed_extent / sqrt(P)) factor storage; every quadratic update remains distributed on all P ranks. Evidence and remaining input-precision limit are in the symmetry register and campaign report.

### 2026-09-06 — parent tau antiunitary placement

Green functions now contract typed child faces for both static and dynamic weights. The symmetry owner unfolds each endpoint before the sole Green GEMM; energies, masks and signed complex-time weights follow the parent index without conjugation. The former parent-operator and transposed-partner Green branches are deleted. Child faces are transient and linear in centroid count: two complex128 faces occupy `32 * nk * nb * ns * M / P` bytes per rank; the quadratic Green result remains distributed over all P ranks. The dynamic Sigma factory hoists the child-face action outside its Green selector.

Dynamic band projection returns raw parent rows. The complex-linear band transpose follows the complete omega accumulation, once per result, rather than once per tau. Static invalid-pole completion uses the same typed unfold after its existing host gather. Quadrature weights are never conjugated. The nontrivial complex-time oracle rejects conjugating them. Numerical, communication and memory acceptance is recorded in the campaign report; the former identity-only tau fixture is insufficient for antiunitary communication claims.

### 2026-09-06 — BSE consumes raw-parent GW restart faces

The full landing gate exposed a downstream reader left behind by the GW carrier deletion. `bse_loading._unfold_bse_parent_faces` authenticates the WFN source, centroid content and parent-row mapping, then uses the existing packed basis and typed parent unfold for the selected valence/conduction bands. BSE receives its established full-k selected-band inputs; GW neither stores nor restores a full-k carrier. Legacy full-k files retain their reader. This consumer helper is a gate-driven correction to the frozen deletion list, not a second symmetry implementation. The parent seam uses the existing square-mesh plan; the single-device reference uses its existing local mesh.


### 2026-09-06 — adopted S phase-2 class residency

The ordered phase-2 integration adopts one compiled Sigma scan and one restore producer per endpoint-family class. The producer returns 1 CC, 3 CT/TC or 9 TT full-q blocks, with centroid axes distributed over all P ranks. This is a correction to the original one-full-q-block residency contract: a TT stack uses `9 * nk_tot * M_T_packed**2 * sizeof(complex128) / P` bytes per rank. It is transient per class, not an interaction cache; each executable shares one Green. Classes are submitted without intervening host fences, so a globally single-class memory lifetime does not follow from this source structure. The campaign report records the measured whole-process allocator peak and the lane gate supporting this explicitly requested integration.


### GW driver phase rulings retained during 2026-09-06 compaction

Original driver phase comments retain their rules, measurements and owner pointers here.

```text
	# Same factory the module-scope seam used, so the two cannot disagree.
	# ---- Stage timing: ONE table, and it sums to the wall -------------------
	# ``timing.reset()`` used to sit just above the ISDF call, which threw
	# away everything the prologue had already recorded.  Resetting HERE,
	# before any stage runs, is what lets the prologue appear at all.
	# ``_t_main`` is the wall this table is closed against
	# (``report(wall=...)``), so the printed rows plus ``(untimed)`` always
	# add up to the run — a reader can tell a complete accounting from a
	# partial one without doing arithmetic.
	#
	# The startup stack now runs ABOVE this reset (it runs above ``main()``),
	# so its own ``collective_warmup`` section is wiped here.  That is fine
	# and deliberate: ``initialize_communicator_stack`` measured every phase
	# itself and handed the numbers back in ``RUNTIME.facts['elapsed']``, and
	# the epilogue re-records them as a DECOMPOSITION of the pre-main span
	# rather than as extra rows.
	# Work done BEFORE main(): the module body's
	# ``initialize_communicator_stack()`` (env, jax.distributed, backend
	# init, mesh + clique warm-up) and every import under it.  Measured 75.0 s
	# to first output on a cold node vs 2.1 s warm — the largest single row in
	# a small run, and previously in no row at all.
	# Configuration parsing predates the scientific report file.  Its deck
	# echo is forensic detail, so it follows the driver's ONE debug switch.
	# Default/derivation provenance is different: it is part of the scientific
	# record, so retain just those lines and replay them once the report exists.
	# ---- Configuration ----
	# The two orthogonal physics axes are resolved + validated up front so
	# inconsistent (qp_solver × compute_mode × accumulation) combinations
	# fail before any heavy compute (see ``LorraxConfig.qp_solver``).
	# A mode may be DECLARED on the axis before its Σ stage exists (today:
	# ``mpa``).  Refusing here — before the WFN read, before ISDF, before
	# any allocation is spent — is the difference between an operator
	# learning in the first second and learning after the ζ fit.  The
	# refusal names the mode; a typo'd mode value never reaches this line
	# because ``config.compute_mode`` already raised on it.
		# Which route a bare-transverse deck takes is deck-visible, never
		# silent: the packed path and the incumbent charge-screened + Sigma^B
		# path are the SAME physics inside the envelope below (lane C gate,
		# reports/bisp_c_bare_as_packed_2026-09-01), but they differ in the
		# q->0 head mechanism, so the reader must be told which one ran and,
		# when it is the incumbent one, the first condition that decided it.
			# In production mode print0 sinks component chatter, so this goes
			# into the RUN RECORD: which of two physically equivalent-inside-
			# the-envelope routes ran is exactly the fact a later reader needs
			# (they differ in the q->0 head mechanism).
		# HEADS ARE ALWAYS ON (owner ruling 2026-09-01,
		# docs/architecture/decisions.md; TASTE.md row 20).  The packed route
		# already prints a boxed WARNING banner and a `Photon head` record
		# line when head_correction=off (gw.w_isdf, lane B).  The INCUMBENT
		# route printed only "no special Gamma-cell contribution" in the
		# component chatter that production mode sinks, so a headless
		# bispinor bulk / dynamic / x_only run reached eqp1.dat with no
		# DEBUG token anywhere in the run record (lane J section 3.c).
	# ---- The runtime is already up ----------------------------------------
	# ``RUNTIME`` was built by ``initialize_communicator_stack()`` at the top
	# of this module, above ``import jax``, because the JAX env defaults only
	# bind before jax reads them.  ``RUNTIME.mesh`` is THE run's square
	# ('x','y') mesh with every communicator it will need ALREADY created —
	# the warm-up is not optional and not the physics' job:
	#   * a mesh this process owns no device on is refused there, naming the
	#     caller, instead of surfacing a bare StopIteration deeper down;
	#   * ``warm_mesh_cliques`` (CPU/MPI) ran as well as ``nccl_warmup``
	#     (GPU/NCCL).  This driver used to call only the latter, so under
	#     ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` its MPI cliques were
	#     created incidentally, by whichever physics kernel happened to fire
	#     the first collective (``common/zeta_projection.py``,
	#     ``common/contract_bands.py`` each warm their own mesh).  That works
	#     only while those early programs stay small enough for XLA's
	#     SEQUENTIAL thunk executor; the parallel executor lands on the
	#     ``MPI_Is_thread_main`` refusal that killed the BSE TDA Lanczos
	#     (32 refusals at P=16, gate 7881216).
	# Do NOT call prepare_mesh() again here: a second Mesh object is a second
	# set of communicators and a second copy of every shape-keyed jit cache.
	# HOW MANY HDF5 LIBRARY INSTANCES IS THIS PROCESS CARRYING?  Measured
	# from /proc/self/maps, not asserted (audit A1 fix 3).  Two
	# instances — h5py's bundled libhdf5 and the FFI's cray
	# libhdf5_parallel — alternately touching one file is the standing
	# explanation for the metallic driver's iteration-3 rank-0 segfault,
	# and until now the condition was INFERRED from the deployed
	# artifacts' NEEDED entries rather than observed in the running
	# process.  Healthy inventory is diagnostic-only and quiet by default;
	# an unsafe condition is always printed.  ``file_io.hdf5_owner`` probes
	# again after each SC iteration's store cycle, with the paths that were
	# touched through both.
	# ---- System inputs: WFN, symmetry tables, ISDF centroids ----
	# This receipt is reporting evidence.  ``resolve_qgrid_symmetry_tables``
	# remains the one q-storage policy/table authority and deliberately
	# remeasures through the same symmetry_maps service at its execution seam.
	# The two user-facing execution dials belong in the first startup lines.
	# Their numeric memory receipt needs the loaded WFN symmetry and centroid
	# basis, so defer the longer architecture/method sections until now.
	# A self-consistent diagonal buffer evaluates a bounded number of real
	# states immediately outside the user-named nval/ncond window.  The named
	# counts remain the physical/full-matrix window; only Meta's execution
	# edges expand.  Other solvers and buffer_nbands=0 retain their exact
	# historical Meta construction.
	# THE in-memory centroid order (common.centroid_basis): whole symmetry
	# orbits per shard so every symmetry action is rank-local; files keep
	# the canonical order and convert at the I/O seam.  Each current family has
	# its own independently packed basis.
	# ``Meta`` describes the charge/CC carrier.  Spatial-current enablement
	# remains the independently parsed ``config.bispinor`` policy.
	# THE ``max`` IS NEVER SILENT.  Which of the two counts sized the ISDF ζ
	# fit is exactly the thing a reader of this log needs and cannot infer:
	# ``nband`` in the deck echo is already the max, so without this line a
	# split deck and an unsplit one at the larger count print the same
	# number.  Printed here, above the ζ fit, because this is where the
	# window it is about is decided.
	# RESOLVED, not requested: ``config.zeta_nband`` is a logical deck count
	# and the edge the fit gets is measured against the PADDED ``b4``.  This
	# banner used to print "sized for 700 bands" on a deck whose resolver and
	# memory planner were both acting on 160 (gw_init.resolve_zeta_fit_edge is
	# the one place that comparison is made).
	# ---- dynamic-Sigma logical/carrier receipt ----
	# Resolve the same square carrier the producer will use, early enough to
	# print the allocation shape before ζ fitting.  This is informational:
	# indivisible windows are ordinary exact-zero pads, not a geometry refusal.
	#
	# ``compute_mode = mpa`` reaches it whatever the deck says about layout,
	# because the MPA executor emits the sharded cube unconditionally.
		# The second refusal that used to live here -- sharded layout with
		# slab_io=h5py_allgather at P>1, which would have re-introduced the
		# full Σ_c(ω) cube gather inside the sigma_mnk.h5 writer -- is gone
		# with the tier.  It was door 7 of 7 and the only one that read
		# ``jax.process_count()`` raw instead of the launcher-aware count,
		# so it was also the weakest.  Nothing can select that writer now.
	# DFT eigenvalues on the Σ band window (Ry) — one fetch, reused by the
	# Σ_X diagnostic, the SC initial state, degeneracy averaging, and the
	# results writer.
	# Single resolver for every q→0 head sample we'll need this run; the
	# COHSEX static head, the W0 restart-flush head, and the PPM dynamic
	# head all read from the same plumbing (overrides → epshead → s_tensor)
	# so they share one cache.  See ``head_correction.HeadResolver``.
	# Optional BGW vcoul override (purely diagnostic — bit-reproducible BGW
	# comparisons).  Returns None when ``use_bgw_vcoul`` is False.
	# Everything from ``main()`` entry to here is the driver PROLOGUE:
	# config parse, the PHDF5 MPI pre-init, WFN + symmetry +
	# centroid reads, the head resolver.  (The mesh, its collective warm-up
	# and the compile cache are NOT here any more — they happen above
	# ``main()`` in ``initialize_communicator_stack`` and are reported as
	# their own rows.)  It is
	# executed exactly once and, on a cold node, it is the largest single row
	# in this table (75.0 s to first output, job 7881949) — so it is named.
	# ``timing.record`` rather than a ``with`` block deliberately: the block
	# above is another workstream's and must not be re-indented for a timer.
	# ISDF fitting or restart loading
	# The Σ kernels take the parent carrier whenever one exists: their G
	# contraction and band projection then run on the raw parents and the
	# band matrix is broadcast back to full k (gw.wavefunction_bundle.
	# sigma_face_kernel_kwargs); the q->0 head wings take the same view and
	# stream the children from the carrier.  Density, output and restart
	# consumers keep the primary bundle.
	# Bispinor: σ^B reads V^{i,j} tiles from v_q_bispinor.h5 and
	# samples ψ at the transverse-centroid Wfns bundle (None when
	# bispinor=False or centroids_file_current is unset).
	# LOUD guard (quality pattern #7): the Σ kernels' Σ^B fold-in is a
	# structural no-op when ``wfns_transverse``/``bispinor_v_q_path`` is
	# None — a bispinor run reaching Σ without them would exit rc=0 with
	# Σ^B silently dropped.  Both producer paths (fit + restart) raise
	# with specifics before this point; this is the last-line invariant.
	# ---- Screening: χ₀ → W = (1 − Vχ)⁻¹ V at every ω the Σ scheme needs ----
	# X_ONLY requests no screening at all.
		# The minimax τ-axis, solved on G's actual spectral range — shared
		# by every χ₀ build this run (static + probe W here, SC re-solves).
		# TIMED because it is the classic mis-attribution on this path: the
		# crossing-minimax solve costs ~95 s cold with no cache and no
		# shipped table (XPROF_TRACE_GUIDE §"Known LORRAX cost centers"),
		# and with no row of its own that 95 s reads as "GW startup".
			# The minimax service announces every served/uncertified rule with
			# ``warnings.warn`` in each process.  This request is deterministic
			# and collective, so keep its loud provenance warning once on the
			# output owner instead of repeating it P times.  The filter is local
			# to this call; exceptions and warnings from later rank-local work
			# remain untouched.
					# A metal's fundamental gap is not its smallest MEANINGFUL
					# transition; the smearing width is.  Insulating decks
					# carry no smearing and keep the incumbent interval.
	# One-shot and QSGW now share one response/finalization implementation.
	# Build the irreducible DFT tensor and its wings on the exact chi0 band
	# manifold before W, then retain only the tiny finalized head after W.
			# The DYNAMIC packed route keeps the scalar charge owner (its CC
			# channel is Sigma_x + Sigma_c(omega) on W_00), so it still needs
			# the direct DFT response its q->0 head samples are finalized
			# from.  Only the STATIC packed mode, whose packed Gamma-cell
			# completion replaces that machinery outright, skips it.
			# Every self-consistent mode builds its exact frequency plan and
			# response inside the map.  A pre-map response would exist only to
			# seed a restart artifact and could be mistaken for final physics.
		# An explicit MPA fit is validate-or-refuse and already owns its
		# finalized scalar head.  Keep the live plan above for authentication,
		# but do not allocate a direct response the reuse branch must discard.
			# ONE charge bundle: both shipped bispinor_gw values ride the
			# raw kinetic-balance carrier, so the scalar q->0 head/wings are
			# built on the run's own charge centroids.  The separate
			# source-Pauli head bundle went with the two retired
			# carrier-comparison modes (2026-09-01).
			# The Sigma view carries the parent carrier: on parents-only
			# storage the wings stream the children from it (qsgw_head).
	# SC solves W inside each map and persists only the final accepted map.
	# Do not perform a redundant DFT screening solve here: besides its cost,
	# that seed body used to survive long enough to be paired with a final head.
				# ONE selector, resolved in gw_config: full_static_cohsex screens
				# all sixteen blocks inside the packed Dyson solve; the
				# bare-transverse family declares chi_TT = chi_CT = 0 and screens
				# only CC, whose owner is the incumbent scalar screening model.
				# PHASE 3.  On the dynamic packed route the CHARGE block is
				# frequency dependent and is owned end to end by the scalar
				# Sigma_c machinery, so this run needs the mode's FULL role
				# set ({static, probe} for the plasmon-pole pair) rather than
				# the single static role the bare family's packed CC block is
				# assembled from -- and it must KEEP them past this block.
				# Sigma consumes packed block views directly.  Do not extract a
				# scalar W00 body solely to satisfy the legacy role mapping.
				# The DYNAMIC route keeps W_by_role: its charge channel is the
				# ordinary Sigma_c(omega) on those same role W's.
				# The run record (gwjax.out) must state the Gamma-cell status
				# of the packed mode in production mode, where print0 sinks
				# component chatter (owner ruling 2026-09-01: a headless
				# packed run is a DEBUG setting and must say so).
					# THE RUN RECORD MUST SAY WHICH SIGMA RAN.  A bispinor
					# plasmon-pole deck used to take the incumbent
					# charge-screened + Sigma^B route with no TT Gamma head;
					# it now takes the packed operator, and the current blocks
					# are an omega = 0 approximation inside a dynamic run.
					# Both facts are physics, so neither is left to a log.
			# Every chi0 call above blocks before returning.  Drop the
			# screening view's reference so it is not an unused jit operand
			# downstream.  The arrays themselves stay resident: the Sigma
			# view (wfns_sigma) holds the same carrier, priced as such.
				# The certified fit carries its already-folded scalar head and
				# Sigma reads that head directly.  Re-folding the current direct
				# response without resident W would be both impossible and a
				# second head path.
	# Persist W0_qmunu + q=0 head scalars to the ISDF restart file for
	# downstream consumers (BSE, future Σ-builders); no-op unless screened
	# and the restart file exists.
	# TIMED, and it was not.  The stage was measured at ~1.7 MB/s
	# aggregate and 2 h 55 m of total silence at c2406 (AF.4c), back when
	# this call gathered the whole (nq, μ, μ) W0 onto one rank on the
	# ``h5py_allgather`` backend to write it.  That backend is gone
	# (233a830d) and the write is SlabIO's per-rank tile path now, so the
	# number is history, not a prediction — yet the call still sat
	# between two timed stages with no
	# section of its own, so it appeared in the run's wall clock and in
	# NO row of the stage table.  Naming it is the precondition for
	# anyone attributing that wall time (the write path itself is
	# workstream AE/AF's; this is the instrument, not the fix).
	# ``sym``/``centroid_indices`` below are for the q-storage resolution
	# ONLY (see the callee): W0 must land on the same q-set V did, and the
	# way to be sure of that is to ask the same resolution point about the
	# same centroid set rather than to infer it from a shape.
	# q→0 head correction.  The bare-X head is the same physical quantity in
	# both COHSEX and PPM modes; gating this on ``not use_ppm_sigma`` was
	# the original ``Bare Σ_X missing q→0 head'' bug (skill compare/SKILL.md
	# §4i).  The SX/COH head pieces are also attached to the static
	# sig_sx/sig_coh in compute_cohsex_sigma, but for PPM those static values
	# are overwritten downstream (sig_sx ← sig_x, sig_c ← PPM-evaluated
	# correlation), so only the X-head survives — which is the piece needed.
			# The dynamic packed route's CC head is this scalar one; only the
			# static packed mode refuses it (its packed completion carries the
			# charge sector, and a scalar overlay would double count it).
		# A screened SC+FULL map always builds/folds its own head.  Supplying
		# and printing a direct DFT seed here would be false provenance even
		# though the map later replaces it.  OFF/NLF and unscreened X_ONLY keep
		# the direct/default route because that is the policy they consume.
					# MPA has no {0,probe} persistence grid; its one-shot
					# static contribution here is bare X from the direct sample.
	# ---- Σ_xc + V_H: ONE dispatch for every mode ----
	# The same ``compute_sigma_xc`` call the SC iteration map makes each
	# step — static COHSEX kernels for X_ONLY/COHSEX, the PPM pipeline
	# (fit → 4-branch τ-integration → analytic q→0 head → at-DFT interp)
	# for the dynamic modes, with the QSGW-symmetrised Σ_xc evaluated at
	# E_DFT (a one-shot full-matrix effective Hamiltonian, distinct from the
	# fixed-state diagonal G0W0 output; ``solve_qp`` re-evaluates for
	# fixed_point).
	# SC-iteration-1 ≡ this call, pinned by tests/test_invariance_gates.py
	# ::test_sc_iteration1_equals_one_shot.
	# SC runs skip it — the iteration map would re-do this work on iter 1.
	#
	# History note (kept here because it explains a specific decision and
	# is not yet captured anywhere else): the analytic q→0 head injected
	# at the end of ``compute_ppm_sigma_pipeline`` was re-added in
	# 2026-04-25 after being removed in 1542342 (Apr-10).  Magnitude is
	# ±W^c(0)/(2·V_cell·N_k) on-shell — ~1.24 eV/band on Si 4×4×4 60b.
	# See reports/mos2_kgrid_gnppm_head_convergence_2026-4-10/.
		# The one-shot path deliberately retains the full BZ.  Use the same
		# named k-set boundary as the SC map, here as a validation rather than
		# a selection, so both paths state what their Sigma tables contain.
		# Screening bodies have no consumer after Sigma.  In the photon mode
		# this drops the packed V/W pair at the exact lifetime boundary rather
		# than carrying O(N_gamma^2) arrays through QP/output post-processing.
		# Print bare Σ_X diagonal for ISDF quality assessment.  Apply
		# BGW-style degenerate-set averaging (mirrors Sigma/shiftenergy.f90)
		# unless disabled — without it, the QE basis-dependent splitting
		# within degenerate manifolds shows up as a few-meV spread across
		# symmetry-equivalent bands.
		# ── Σ stage gate ─────────────────────────────────────────────
		# Σ_x[n,n] = −Σ_{m∈occ} ⟨nm|V|mn⟩ is a negative-definite
		# quadratic form in a positive-semidefinite kernel: every
		# diagonal entry is strictly negative in a correct run, whatever
		# the system.  A positive one is a sign / conjugation /
		# band-index slip, not a convergence issue.  The magnitude
		# bracket is deliberately loose (bare exchange runs −40…−5 eV
		# for the production decks) — it exists to catch a units or
		# basis-normalisation slip, not to police physics.
	# ---- QP Hamiltonian: H_QP = (H_DFT - V_xc) + V_H + Σ_xc ----
	#
	# Static operators retain the layout selected by their producers through
	# the H build and diagonalisation.  Rank-0 text output consumes bounded
	# ``(nk,nb)`` diagonal mirrors; it never gathers a band-sharded
	# ``(nk,nb,nb)`` component merely to print its diagonal.
	# Provenance gate BEFORE the read.  In particular, refuse the retired
	# format that folded V_H into kin_ion: this driver always adds the live
	# G-space field.
	# TIMED as one row: the gate and slab read are one logical stage.
	# ---- update_H[Σ; qp_solver] — all branches yield ``sigma_total``
	# (Σ_xc + V_H, Ry, DFT basis, replicated) whose eigh gives E_qp/U_qp.
	# ``rotations_written`` is run_sc_driver's own report of whether it
	# wrote qp_wfn_rotations.h5; the writer below reads the fact rather
	# than re-deriving the predicate.
		# SC-QSGW: iterate ψ-rotation → χ₀ → W → Σ_xc (the same
		# compute_sigma_xc dispatch, mode-agnostic) to the fixed point;
		# the returned SigmaResult is already rotated back to the DFT
		# basis and its sigma_omega_h5_path points at the converged
		# single-write sigma_mnk.h5.  See ``sc_iteration.run_sc_driver``.
		# Executed once; it is the whole SC loop, so it is the run's biggest
		# row when it fires and must not hide inside ``(untimed)``.
			# The self-consistent map takes the Σ bundle: under parents-only
			# storage its carrier is the run's only ψ and the map's rotation
			# acts on it (wavefunction_bundle.rotate_wavefunctions).
			# SC's retained Sigma, its H/E/U and the mean-field operators used
			# by post-processing all stay on the loop's star wedge.  Output
			# writers alone unfold that complete result to their file wedge.
		# One-shot: ``one_shot_dft`` = Σ_xc was already QSGW-built at
		# E_DFT inside compute_sigma_xc (pass-through; also covers static
		# modes and the streamed-Σ_c stand-in); ``fixed_point`` = diagonal
		# on-shell solve + scissor + QSGW rebuild at the solved energies.
		# eqp0.dat/eqp1.dat are at-DFT in every case (written downstream
		# from ``sigma_c_at_dft_ev`` / the ω-grid diag, not from here).
	# Optional additional ladder: iterate ONLY the already-built, full-matrix
	# one-shot Sigma(omega).  This deliberately sits after the sole screening /
	# Sigma stage and calls neither dispatch, so write_eqp2 cannot accidentally
	# turn into a second GW calculation.  The callee rotates the fixed cube into
	# each updated QP basis before evaluating its off-diagonals.
	# ---- Post-Σ seam: bare locals from the SigmaResult ----
	# One extraction for SC and one-shot alike; PPM-only fields are None
	# in static modes.
	#
	# On the SC path the finalize rotates every matrix consumed here,
	# including the dynamic correlation cube, to the DFT output basis.  The
	# original cube was already persisted in the last map's QP compute basis,
	# where the QSGW ansatz is defined.  The separable analytic head diagonal is cheap
	# enough to rotate without materialising its dense matrix; SC returns that
	# as a separate DFT-basis diagnostic while leaving SigmaResult's
	# basis-of-computation field untouched.
	# The energies THIS Σ was evaluated at (E_DFT one-shot, the map's input
	# QP energies under SC).  Not degen-averaged below with the Σ channels:
	# it is a spectrum, not a self-energy component.
	# ---- BGW-style degenerate-set averaging at the H-build seam ----
	# (mirrors Sigma/shiftenergy.f90; see ``degen_average``).
		# Head-only debug columns must undergo the SAME DFT-degenerate-set
		# averaging as the Sigma components they decompose.  After the QP->DFT
		# diagonal transform an occupied-projector contribution need not be
		# identical in an arbitrary basis inside a degenerate manifold.
	# The PPM output below needs only bare exchange's diagonal.  Extract it
	# collectively while every rank is still in lockstep; never host-convert
	# the full band-sharded operator.
	# Σ_xc(E_DFT) diagonal (eV) — drives eqp_g0w0.dat (PPM one-shot
	# only).  Form it AFTER the one canonical conditioning seam above: forming
	# this sum before ``average_sigma_components`` made eqp_g0w0 retain raw,
	# unequal degenerate diagonals even while every other live text output used
	# the conditioned X/C pair.  With averaging disabled the seam is a no-op,
	# so this same expression deliberately preserves the raw red twin.
	# ---- Single H-build + diagonalization on the producer-selected layout ----
	# Gate the two inputs to the QP diagonalization *before* eigh: LAPACK
	# on a NaN-bearing matrix returns without complaining, and the garbage
	# then propagates into eqp0/eqp1/WFN_qp.h5 with rc=0.  ``kin_ion`` also
	# comes off disk (kin_ion.h5), so this doubles as the content check on
	# that interface.
	# REFUSALS, not warnings, and that distinction is the 2026-08-15 bcc-Fe
	# finding: an all-NaN Σ_c produced an all-NaN E_QP column, a NaN E_F and
	# a NaN scissor fit, and the run exited rc=0 in 883 s with only a warning
	# line to show for it (JID 57051742, CLAIMS 204).  The SC path catches
	# that on its SECOND map call, through
	# ``_solve_head_occupations -> OccupationState`` finiteness; a one-shot
	# run has no second map call, so this seam -- which both paths cross -- is
	# where the guard has to be.  ``LORRAX_ALLOW_NONFINITE_RESULT=1`` is the
	# named escape for forensics.
	# TIMED: nk independent (nb_sigma, nb_sigma) Hermitian eigensolves.  It is
	# one statement and normally seconds, but it is O(nk·nb³) and it is the
	# only dense LAPACK call on the post-Σ path, so it is the row that tells
	# you when the band window (not the physics) became the cost.
	# ---- One-shot WFN_qp.h5 dump (drop-in BSE / restart input).  SC
	# already wrote its own WFN_qp.h5 above via dump_qp_wfn_artifacts
	# (using state_final.H_qp_dft) — same physics, slightly different
	# numerics from the post-Σ-seam eigh path.  Skip the second write
	# in SC to avoid clobbering.
	# PPM mode: feed the writer the on-shell diag(Σ_c(E_DFT)) (Ry) so the
	# eqp0.dat "sigC" column reports dynamic correlation directly comparable
	# to BGW's (SX-X)+CH at Eo=E_DFT.  Off-diagonals stay zero — the full
	# Σ_c(ω, k, i, j) tensor is in sigma_mnk.h5 for callers that need them.
	# Σ_c diagonal on the ω-grid: feed the eqp1.dat writer's central-diff
	# Z-factor.  Pulled from the on-device sharded tensor when available.
			# Output-only diagonal curve.  The full persisted operator stays raw,
			# while this curve uses the SAME canonical group owner at every omega;
			# assemble_eqp consequently derives C(E_DFT) and Z from one function.
	# Per-state Gamma-cell attribution in the same conditioned DFT basis as
	# sigma_diag.dat.  The packed diagnostic owns CC/CT+TC/TT for its static
	# completion; the dynamic scalar owner contributes the X and C charge head.
	# ---- Output ----
	# Collectively extract only the bounded static diagonals before the
	# rank-0-only writers.  The full operators remain on their original mesh;
	# calling ``np.array`` on one here would either fail at multi-host P>1 or
	# silently reintroduce the full-matrix replication this layout avoids.
			# Dynamic sigma_diag owns sigXC as the direct per-state
			# sigX + sigC(E_DFT) interpolation, whereas the H operator above is
			# the QSGW-Hermitianized matrix.  Their diagonals can differ at the
			# micro-eV level because they use distinct interpolation kernels.  The
			# public Lorentz split is a decomposition of the old sigXC column, so
			# define its CC residual at that SAME output seam; CT+TC and TT remain
			# the independently computed static current contributions.
		# Canonical Σ-decomposition table.  Explicit debug requests it for all
		# modes; actual coupled q=0 completion auto-arms the same writer.
		# The QP-ladder half of sigma_mnk.h5's opt-in plotting appendix
		# (no-op unless ``write_qsgw_datasets``).  HERE and not at the Σ
		# seam because two of the three ladders need ``kin_ion``, which
		# ``compute_sigma_xc`` never sees; the QSGW cube itself was
		# already appended by whichever path wrote the file
		# (``qsgw_utils.write_qsgw_sigma_cube``).  Rank-0 and barrier-free
		# like every other writer in this block: eigenvalues are basis-
		# free, so this one seam is correct for the one-shot and the
		# self-consistent paths alike.
		# Degen averaging was applied once at the H-build seam upstream;
		# the writer just serializes the already-averaged Σ components.
			# THE symmetry object, not tables off it.  Every k-basis
			# decision is made where the data is written, through
			# ``symmetry_maps.reduce_full_bz_to_file_wedge``.
		# DECOMPOSE the pre-main span; do not add rows to it.  The entry
		# point timed its own phases (it happened before ``timing.reset()``,
		# so its own ``collective_warmup`` section was wiped), and the
		# remainder of ``_pre_main`` is the import storm — 75.0 s cold vs
		# 2.1 s warm, job 7881949.  Recording the phases AND the whole span
		# would double-count and break the table's "rows + (untimed) ==
		# wall" property, which is the only thing that lets a reader tell a
		# complete accounting from a partial one.
		# ``wall=`` closes the table: printed rows + ``(untimed)`` == the
		# whole PROCESS when /proc gave us the pre-main span, else main().
```
