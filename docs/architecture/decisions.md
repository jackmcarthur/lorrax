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


## Source contracts relocated during the 2026-09-06 compaction

### `src/gw/gw_config.py` — `<module>`

Unified configuration for LORRAX GW calculations.

``LorraxConfig`` is built once via :meth:`LorraxConfig.from_input_file`
from the ``[cohsex]`` section of ``cohsex.in`` and threaded through the
entire driver.  Its ~80 input keys are grouped into sub-dataclasses
along the same axes the input file's section comments already use:

    config.head        — q→0 Coulomb-head sources & overrides
    config.minimax     — screening-minimax target error / max nodes / table mode
    config.ppm         — PPM model + sigma quadrature + on-shell σ_c options
    config.sigma_grid  — ω-grid for Σ_c(ω) output
    config.sc          — self-consistency loop knobs (qp_solver = self_consistent)
    config.memory      — chunk sizing
    config.backend     — FFI/linalg backend selection
    config.debug       — debug-only flags & file paths
    config.bse         — BSE interpolation setup (htransform-driven)
    config.paths       — output filenames

The top-level ``LorraxConfig`` retains only system geometry
(``nval`` / ``ncond`` / ``nband`` / ``sys_dim``) and the orthogonal
mode flags (``compute_mode`` / ``qp_solver`` / etc.) that the
driver reads on the fast path.

Derived sub-objects (the math-internal ``MinimaxConfig`` from
``minimax_config.py``, one instance per quadrature consumer) and derived
data (the Σ_c(ω) grid) are constructed on demand via ``LorraxConfig``
properties.

### `src/gw/gw_config.py` — `env_float`

Canonical numeric env parse: unset/blank → default, bad → ANNOUNCE
(or, with ``refuse=True``, RAISE).

The same defect class as :func:`env_bool`, one type along.  A
``try: float(...) except: default`` leaves the user believing a knob is
in force when it is not — the exact failure the
``ISDF_CHUNK_TARGET_UTILIZATION`` parser used to commit.

``refuse=True`` is for knobs that GATE correctness rather than tune
performance (``LORRAX_FI_FSHOULDER_TOL``): running with the default while
the user believes a gate threshold is in force is itself the silent
failure, so garbage refuses loudly, naming the variable — the
announce-or-refuse doctrine's refuse half.

### `src/gw/gw_config.py` — `active_zeta_truncating_knobs`

``[(name, raw), ...]`` for every truncating knob currently in force.

Blank counts as unset (the r-chunk loop's own guard is
``if _max_rchunks and ...``, so ``""`` does not truncate).

### `src/gw/gw_config.py` — `ComputeMode`

The single axis describing what self-energy is computed.

Orthogonal to ``qp_solver`` (how QP energies are extracted from Σ):
any mode can be wrapped in the ``self_consistent`` QSGW loop — the
loop dispatches through the mode-agnostic
``sigma_dispatch.compute_sigma_xc`` (COHSEX and GN-PPM verified
end-to-end; see reports/gw_refactor_map_2026-07-01/
G0W0_SC_TOGGLE_DESIGN.md §4).

- ``X_ONLY`` — bare exchange Σ_X = -G·V (no screening, no correlation).
- ``COHSEX`` — static screened-exchange + Coulomb-hole.
- ``GN_PPM`` — dynamic Σ_c(ω) via GN plasmon-pole (probe at iω_p).
- ``HL_PPM`` — dynamic Σ_c(ω) via HL plasmon-pole (probe at real Ω).
- ``MPA`` — dynamic Σ_c(ω) from an n-pole multipole fit of W on a
  double-parallel sample grid in the complex-ω plane (complex poles
  Ω_p, residues B_p).  **DECLARED, NOT YET RUNNABLE** — see
  :data:`UNIMPLEMENTED_MODES` and
  :func:`refuse_unimplemented_compute_mode` below.

WHY THE VALUE IS SPELLED ``mpa`` AND NOT ``full_freq``.  Every value
on this axis names the *ansatz* for W's frequency dependence, not the
numerical machinery that follows from it: ``cohsex`` is "W at ω = 0",
``gn_ppm`` / ``hl_ppm`` are "one plasmon pole, fitted this way".  The
next member of that series is "n poles, fitted to a sampled W", whose
name in the literature is the multipole approximation, so ``mpa`` is
the spelling that keeps the axis reading as one list of ansätze.

``full_freq`` was the rejected alternative, and it was rejected for
two reasons rather than taste.  First, it names a *family* — contour
deformation, real-axis quadrature and MPA are all "full frequency" —
so a deck that set it would still have to say which one, which is a
second axis, which is precisely the thing the "single axis" wording
at the top of this docstring exists to prevent.  Second, it would
spend the good name: a genuinely numerical full-frequency Σ (no pole
model at all) is a plausible future member of this enum, and it
should be able to be called ``full_freq`` when it arrives instead of
finding the name already taken by a pole method.  The owner-facing
shorthand for this work is still "FF"; the deck key is ``mpa``.

### `src/gw/gw_config.py` — `ComputeMode.is_dynamic`

True when the mode builds a Σ_c(ω) grid: GN/HL-PPM and MPA.

The honest reading is "this run has an ω axis", which is what the
consumers of this property want to know (the σ-cube layout gate,
``qp_solver = fixed_point``'s ω-grid requirement, ``GWResults.
use_ppm``).  It is deliberately NOT the same question as "is this
a plasmon-pole model" — that one is :attr:`ppm_model`, and the
two questions differ for exactly one member, ``MPA``.

### `src/gw/gw_config.py` — `ComputeMode.ppm_model`

``'gn'`` for GN-PPM, ``'hl'`` for HL-PPM, else None.

None for MPA as well as for the static modes: MPA is dynamic but
is not a plasmon-pole model, so any site that means "which of the
two two-point PPM fits" must ask THIS and handle None, never
``is_dynamic`` with an ``else`` that assumes GN.

### `src/gw/gw_config.py` — `BispinorGWMode`

How the four-current photon channels enter the GW self-energy.

This is orthogonal to :class:`ComputeMode`: that enum selects the
frequency ansatz, while this one selects which Lorentz blocks are screened
and contracted.  ``bare_transverse`` is the historical charge-screened +
bare-TT behavior and remains the default.

TWO VALUES, and that is the whole grammar (owner ruling 2026-09-01,
``docs/architecture/decisions.md``; lane J's dial review,
``reports/bisp_j_architecture_review_2026-09-01/report.md`` section 2).
The three retired spellings -- ``charge_hall_cubature`` and the two
carrier-comparison modes ``pauli_reference_bare_transverse`` and
``isometric_kinetic_balance_bare_transverse`` -- are refused BY NAME in
:func:`coerce_bispinor_gw_mode`, never aliased: a mode value is the one
thing in the grammar that decides which physics runs, so a stale deck
must stop, not be silently re-pointed.

``full_static_cohsex`` is the ONE packed static mode: the sixteen-block
no-pair photon body, screened once at omega=0 under ``compute_mode =
cohsex``, plus the Gamma-cell completion (bare ``<D>`` into V, the charge
``S^{00}``/wing head into W, the Hall CT/TC term when a Hall artifact is
present).  The completion runs by default (owner ruling 2026-09-01,
``docs/architecture/decisions.md``); ``head_correction = off`` skips it
behind a DEBUG banner.  The former ``charge_hall_cubature`` spelling is
refused by :func:`coerce_bispinor_gw_mode` naming this mode.

### `src/gw/gw_config.py` — `announce_legacy_sigma_axis_keys`

Print one deprecation note per LEGACY self-energy-axis key the deck named.

Returns the keys announced, so a caller (or a test) can assert on them
rather than scraping the log.  Nothing is refused and nothing resolves
differently: this is the warning stage of the migration described on
:data:`LEGACY_SIGMA_AXIS_KEYS`.

### `src/gw/gw_config.py` — `SigmaChannel`

One term of Σ that a compute mode either builds or does not.

These are the channels the driver's outputs are written FROM — the
names on ``sigma_dispatch.SigmaResult`` and the operands of the QP
ladders in ``gw_output`` — not every intermediate a kernel touches.

- ``X`` — bare exchange Σ_x = −G·V.  Built by every mode; it needs no
  screening and every output that reports a Σ decomposition wants it.
- ``SX`` — static screened exchange Σ_SX = −G·W(0).
- ``COH`` — the Coulomb hole Σ_COH.  SX and COH are one pair in
  practice (a mode that builds one builds the other) but they are two
  datasets and two columns, so they are two channels here.
- ``C_OMEGA`` — dynamic correlation Σ_c(ω) on an ω grid, whatever
  analytic model produced it.

### `src/gw/gw_config.py` — `SigmaChannel.label`

How the channel is spelled in prose and in operator messages.

The enum VALUE stays a lowercase identifier because it is data —
it keys tables and appears in tests.  Messages an operator reads
want the physics spelling, and having both means neither has to
compromise.

### `src/gw/gw_config.py` — `coerce_compute_mode`

Accept a :class:`ComputeMode`, its ``.value``, or a bare string.

The writers reach this table holding whatever their caller handed
them — a resolved enum from ``config.compute_mode`` in the driver, a
plain string in a deck-echo path, an object carrying ``.value`` in a
unit test's stand-in config.  Normalising in ONE place is what lets
the table be the single answer rather than the third mode-string
hand-check in the tree.

An unrecognised spelling raises the same ValueError shape the config
parser raises, naming the legal set — a typo never resolves to a
default.

### `src/gw/gw_config.py` — `HeadCorrection`

Finite-grid treatment of the singular macroscopic ``q -> 0`` head.

``FULL`` is the physical default: an irreducible direct response is
completed with its microscopic head/body wings exactly once, while an
already micro-reducible response (the BSE resolvent) is used as-is.
``NO_LOCAL_FIELDS`` is the explicitly diagnostic epsilon-head value, and
``OFF`` removes the special Gamma-cell contribution so brute-force k-grid
convergence can be studied.  The diagram choice remains the orthogonal
:class:`ScreeningDiagrams` axis.

### `src/gw/gw_config.py` — `ScreeningDiagrams`

WHICH DIAGRAMS build the W that Σ consumes — the screening axis.

Orthogonal to :class:`ComputeMode` (which Σ *ansatz* is evaluated) and
to ``screening_method`` (how the χ₀ frequency integral is done).  Those
two say *at which frequencies* W is wanted and *how the quadrature is
taken*; this one says *which series* W sums.

- ``W_RPA`` — the random-phase approximation, ``W = (1 − Vχ₀)⁻¹V``.
  The only screening LORRAX had before 2026-08-15 and the default, so
  a deck that does not name this key is bit-identical to every deck
  written before it.
- ``W_BSE`` — ladder-corrected W: ``W(ω) − v = v (ω − H)⁻¹ v`` with the
  statically screened direct rung ``−W(0)`` in the kernel of ``H``.
  Two-stage by construction — the RPA ``W(0)`` of the first stage IS
  the ``W_R`` the ladder kernel consumes — which is why this value
  changes the dataflow rather than one solver call.
- ``W_RPA_RESOLVENT`` — the SAME resolvent identity
  ``W(ω) − v = v (ω − H)⁻¹ v`` evaluated with the RPA operator
  (``H_RPA``, the ladder's own ``include_w=False`` limit: the direct
  rung ``−W_R(0)`` is parameterized OUT of the ring matvec rather than
  rebuilt by a second matvec — ``bse.bse_ring_comm.
  build_bse_ring_matvec_full(..., include_W=False)`` — so this value
  exercises the same operator family as ``w_bse``, minus one term).
  DESIGNED to reproduce ``w_rpa``'s W to the minimax-quadrature floor,
  and CERTIFIED to do so on the spinor fixture the existing unit
  suite exercises (``tests/test_bse_w_ladder_identities.py``,
  ``tests/test_w_bse_wiring_closure.py`` — ``gnppm_debug``, nspinor=2,
  ~7e-12 agreement). **OPEN, MEASURED 2026-08-23 on a SCALAR
  (nspinor=1) system: it does NOT.** A direct q=0 tile probe against
  the incumbent ``W0_qmunu`` on a fresh scalar Si deck found a
  38-61%-relative, P-independent, window-size-insensitive
  disagreement (KNOWN_LORRAX_ISSUES.md, the
  ``bse_w_exact._build_rpa_resolvent`` row) — high shape-correlation
  (cos~0.999) but a non-uniform per-entry under-scaling, the
  signature of a missing/misapplied occupation or spin-degeneracy
  weight rather than a sign, operator, or solver-tolerance defect.
  THIS IS UPSTREAM OF ``include_w``: the same shared ring term backs
  ``w_bse`` too, and that feature's own first-ever scalar decks
  (``runs/Si_scalar/01_wbse_ab_2026-08-16``) never finished far
  enough to have compared against a reference, so the gap has been
  latent and undetected since that feature shipped. Do not read this
  value (or ``w_bse``) as certified-correct on a scalar mean field
  until that row closes; it exists to gate the resolvent machinery
  against the incumbent Dyson route on a diagram set simple enough to
  have an independent right answer, and on THIS run it correctly
  caught that the two disagree.

WHY AN ENUM AND NOT A BOOL.  ``ladder_screening = true`` would name the
one alternative that exists today and spend the axis: the resolvent
formalism admits more than one diagram set (TDA vs full symplectic,
test-charge vs test-electron), and each is a *value* on this axis, not
a second boolean beside it.  The same reasoning that spelled
``compute_mode`` as an enum of ansätze rather than ``use_ppm_sigma``
(see :class:`ComputeMode`'s docstring) applies here.

NOT EVERY COMBINATION IS SUPPORTED.  ``w_bse`` is refused at parse
time against ``x_only``, ``hl_ppm``, the self-consistent QP solver,
``mc_average_placement != off`` and a declared metal
(``mpa_material_class = metal``) — see
:func:`refuse_unsupported_screening_diagrams`, which carries the
reason for each.  INSULATORS ONLY is the one of those that a deck key
cannot always express: a metallic WFN on a deck that declares nothing
is refused at the stage instead, on the occupations themselves
(``gw.screening_bse``, the same ``w_bse_insulators_only`` id).
``w_rpa_resolvent`` is refused at parse time against ``x_only``, the
self-consistent QP solver, ``mc_average_placement != off`` and
``compute_mode = mpa`` — audited against ``w_bse``'s table, not
copied: the x_only / broadening / SC-loop / head-placement arguments
transfer (some by MECHANISM, some as an inherited infrastructure
risk; see ``_W_RPA_RESOLVENT_REFUSALS``' own per-row comments), and
``compute_mode = mpa`` is a NEW row here — MPA's ``wc_source`` seam
(``gw.screening_bse.make_ladder_wc_source``) has not been extended or
gated for the RPA-resolvent arm this session, unlike ``w_bse``, where
it is SUPPORTED.  INSULATORS ONLY has NO parse-time row for this
value: it is subsumed by the ``compute_mode = mpa`` refusal (a
declared metal requires ``compute_mode = mpa``, which is refused
unconditionally here, making a parallel deck-key predicate always
shadowed — see ``_W_RPA_RESOLVENT_REFUSALS``' comment at that site).
The certification and its enforcement survive in full through the
OTHER half w_bse already has: a metallic WFN on a deck that declares
nothing is refused at the stage, on the occupations themselves,
under the SAME ``{value}_insulators_only`` id pattern
(``gw.screening_bse``).

### `src/gw/gw_config.py` — `coerce_screening_diagrams`

Accept a :class:`ScreeningDiagrams`, its ``.value``, or a string.

Same shape and same reason as :func:`coerce_compute_mode`: the parser,
a hand-built stub config and a deck-echo path all reach the axis
holding different spellings of the same request, and normalising in
ONE place is what keeps the dispatch a single answer.  A typo raises
naming the legal set — it never resolves to the default.

### `src/gw/gw_config.py` — `refuse_unsupported_screening_diagrams`

Refuse the resolvent-diagram combinations v1 does not serve, at PARSE time.

Called from :meth:`LorraxConfig.from_input_file` once the record
exists, because every predicate here reads a RESOLVED axis
(``compute_mode`` and ``qp_solver`` are properties that fold in the
legacy flags) and re-deriving them beside the parse would be a second
opinion about the same question -- the shadow-accounting failure
class, QUALITY_PATTERNS #3.

NO-OP FOR ``w_rpa``, evaluated first and returning before any property
is touched: a default deck must not acquire a new parse-time
resolution -- and hence a new possible refusal -- from this function
existing.  ``w_bse`` and ``w_rpa_resolvent`` each carry their OWN
table (:data:`_RESOLVENT_REFUSAL_TABLES`) rather than one shared list,
because a shared table's ``doc`` text would have to describe both
operators at once -- which is exactly how the hl_ppm gate's dead-gate
incident happened (TASTE.md, "a gate pinned to a convention re-arms
itself"): a reused reference that does not resolve per call site reads
as evidence for a case it never measured.

### `src/gw/gw_config.py` — `explain_missing_channels`

The named-omission clause for channels ``mode`` does not build.

Phrased as a fragment so a writer can put it in parentheses after the
name of whatever it is declining to write, which is the shape the
QSGW appendix's line already had.

### `src/gw/gw_config.py` — `refuse_unimplemented_compute_mode`

Refuse a declared-but-not-yet-built compute mode, by name.

No-op for every mode whose Σ stage exists, so the call is free to sit
on the driver's fast path.  Raises :class:`NotImplementedError` —
distinct from the ``ValueError`` a *typo* gets from the parser,
because the two are different operator mistakes and deserve different
words: ``compute_mode = mpaa`` is "no such mode", ``compute_mode =
mpa`` is "that mode, not yet".

### `src/gw/gw_config.py` — `QPSolver`

How QP energies are extracted from Σ — orthogonal to ``compute_mode``.

The three states are mutually exclusive answers to the same physics
question, each naming a standard method:

- ``ONE_SHOT_DFT`` — one-shot full-matrix effective Hamiltonian (THE
  DEFAULT).  Σ is built once from the DFT inputs and evaluated at
  E_DFT; the QSGW-Hermitianised Σ_xc is diagonalised to produce
  ``E_qp_ry`` / ``qp_wfn_rotations.h5`` / ``WFN_qp.h5``.  This is
  distinct from the fixed-DFT-state diagonal ``eqp0.dat`` /
  ``eqp1.dat`` outputs.  No iteration of any kind.
- ``FIXED_POINT`` — one-shot Σ + diagonal on-shell solve
  E = h0 + ReΣ(E) for the QSGW-build evaluation energies
  (eigenvalue-only; Σ is never rebuilt).  Dynamic modes only — static
  Σ has no ω-grid to solve on.  ``sigma.sigma_at_dft_extrapolate`` is a
  sub-knob of this state (scissor for out-of-grid bands).
- ``SELF_CONSISTENT`` — full QSGW loop (:mod:`gw.sc_iteration`):
  Σ rebuilt each iteration from rotated ψ + the previous iteration's
  E.  Loop knobs live in :class:`SCConfig` (``config.sc``).

eqp0.dat / eqp1.dat keep the same formula in all three states; only
the provenance of Σ changes under ``SELF_CONSISTENT`` (converged Σ,
still evaluated at E_DFT — one more at-DFT Newton step from the SC
fixed point).

### `src/gw/gw_config.py` — `normalize_w_dyson_solver`

Normalise a ``w_dyson_solver`` spelling to one of the TWO plans.

Single source of the vocabulary — the parser and
``w_isdf._resolve_w_solve_fn`` both call this, so a spelling cannot
mean different things at parse time and solve time.

- ``local`` / ``auto`` / None → ``"local"`` (the q-parallel per-q
  dense LU; ``auto`` is a permanent back-compat alias).
- ``distributed`` → ``"distributed"`` (the 2-D-sharded stacked-GEMM
  backsolve through the distrib_la plan door).
- ``lu`` → ``"local"`` with a DeprecationWarning (it was the same
  route under its old name).
- ``lstsq`` → ``ValueError``: the SVD min-norm inner solve was
  REMOVED in the two-plan cleanup (2026-07-27) — old decks fail
  informatively instead of silently rerouting.

### `src/gw/gw_config.py` — `eigh_backend_choices`

The legal ``eigh_backend`` spellings — the RESOLVER's own list.

Read from :data:`distrib_la.BACKEND_CHOICES` so the parser and the
thing that actually dispatches cannot drift.  They HAD drifted:
this parser accepted only ``auto|off|cusolvermp|slate`` while the
resolver had grown ``distributed`` (the portable "spread ONE tile over
the mesh" spelling, and the ONLY eigh backend that exists on a host
mesh, where it means ScaLAPACK ``pzheevd``) and ``scalapack``.  The
effect was that the low-memory eigh could not be requested at all
through a GW input file on CPU — the very platform it is needed on.

``BACKEND_CHOICES`` is importable with NO ``.so`` on the machine — that
is a distrib_la door promise, precisely so a deck parser never needs
the FFI layer.  The literal fallback below covers the remaining case,
a tree whose ``services/`` is not on the path at all; it is pinned
equal to the door's list by ``tests/test_bse_setup_qchunk.py``.

:data:`EIGH_CHOICES_SOURCE` records WHICH of the two answered, because
a test comparing the two lists cannot: they are equal today, so the
comparison passes whether the live import ran or the except branch
caught it, and the drift this function exists to prevent would recur
with no signal at all.

### `src/gw/gw_config.py` — `LinalgResolution`

Internal execution profile selected by the one ``linalg`` deck dial.

The public vocabulary deliberately stops at layout.  These fields retain
the established implementation choices so existing stage code need not
become a second interpreter of ``local`` versus ``distributed``.

### `src/gw/gw_config.py` — `resolve_linalg`

Interpret ``linalg = local | distributed`` exactly once.

``distributed_lu='distributed'`` is an internal portable sentinel.  The
typed-config factory lowers it to cuSolverMp on CUDA and ScaLAPACK on CPU;
that is capability routing, not another interpretation of the deck dial.

### `src/gw/gw_config.py` — `distrib_la_batched_route_choices`

User-facing batch-route vocabulary from the ``distrib_la`` door.

``batch_reshard`` is the shipping default: it moves the batch axis onto
the device mesh and runs the service's local JAX kernel on whole
per-device matrices. ``auto`` explicitly restores the resolved
backend's scan/stacked-FFI route. Keep
this resolver beside :func:`eigh_backend_choices`: deck and CLI parsers
must not grow frozen copies of a service-owned vocabulary.

### `src/gw/gw_config.py` — `BandCountConflict`

Two band-count keys were set and they disagree.

Refusal, not coercion.  Every silent resolution of this case is wrong for
somebody: picking the umbrella throws away the specific request the deck
took the trouble to write, picking the specific makes the umbrella a lie
for the OTHER consumer, and picking the max or the min invents a run
nobody asked for.  So it is named, with both values quoted and the edit
that fixes it spelled out.

### `src/gw/gw_config.py` — `BandCounts`

The resolved χ and Σ band counts, and what the ISDF fit is sized by.

Constructed exactly once per run, by :func:`resolve_band_counts`.  The
only three numbers below this point are :attr:`chi`, :attr:`sigma` and
:attr:`isdf`; nothing downstream re-reads a deck key to get a band count.

Attributes
----------
chi, sigma : int
    The χ0/W band count and the Σ band-sum count, both fully resolved
    (never ``None``): a deck that names only the umbrella gets them equal
    to it, which is the whole of the bit-identity claim.
isdf : int
    ``max(chi, sigma)`` — the top of the band window the ψ is loaded over
    and therefore the window the ISDF ζ fit is built for.  The
    interpolation basis has to span the pair densities of whichever
    consumer reaches higher; sizing it by the smaller one would leave the
    larger consumer extrapolating in the ζ basis.
named : frozenset[str]
    Which of the four keys the DECK itself wrote.  Kept so consumers can
    distinguish "asked for this edge by name" (→ strict degeneracy check,
    the ``zeta_nband`` precedent) from "inherited it" (→ the grandfather
    clause), without re-parsing the deck.

### `src/gw/gw_config.py` — `BandCounts.describe`

The one line a run logs so the ``max`` is never silent.

Named in the brief that asked for the split: "log which count won the
``max`` and what the fit was built for.  A silent ``max`` is the kind
of thing that gets mis-debugged for a day."

``zeta_fit_edge`` IS THE RESOLVED EDGE, NOT THE DECK KEY.  Pass
``gw.gw_init.resolve_zeta_fit_edge(band_slices, config.zeta_nband)``
— the same value the fit, the window gates and the memory planner
act on.  ``None`` means "nothing narrows it", i.e. the fit really is
sized by :attr:`isdf`.

A BANNER PRINTS RESOLVED VALUES ONLY.  With ``nband=700`` and
``zeta_nband=160`` this line used to say "ISDF zeta fit sized for 700
bands ... the fit spans both" while the resolver and the memory
planner were both acting on 160 (the CrI3 rank floor fell 180 -> 84
on that key).  A startup-only run was then left with a materially
false provenance line and no way to tell.  Perlmutter smoke step
57236676.2,
``runs/CrI3/00_fm_331_991_700b_qsgw_gnppm_20260818/00_lorrax_smoke_p4/``.

### `src/gw/gw_config.py` — `resolve_band_counts`

Resolve the four band-count keys into one :class:`BandCounts`.

**THE ONLY PLACE THIS PRECEDENCE EXISTS.**  Four keys with two spellings
of the umbrella is exactly the shape that grows a second, disagreeing
resolution in a consumer six weeks later, so there is one function, it is
pure, and it is directly testable without a deck, a WFN or jax.

PRECEDENCE, in order:

1. ``nband`` is a TRANSITIONAL ALIAS of ``number_bands``.  Either
   spelling sets the umbrella.  Both set to DIFFERENT values → refuse
   (:class:`BandCountConflict`).
2. The umbrella supplies BOTH consumers.  A deck that names only it —
   every deck in the tree today — gets ``chi == sigma == umbrella``, and
   that is the bit-identity claim.
3. ``number_bands_chi`` / ``number_bands_sigma`` override their own
   consumer and nothing else.
4. Naming the umbrella AND a specific key with DIFFERENT values → refuse.
   "The umbrella overrides both" and "a specific key overrides its
   consumer" are both true and they contradict each other exactly here;
   this codebase has been bitten repeatedly by silent coercion, so the
   contradiction is reported rather than broken by fiat.  Naming them
   with the SAME value is redundant, not wrong, and is accepted.

Parameters
----------
params : dict
    A params dict from :func:`read_lorrax_input`, or any dict with the
    same keys.  Missing keys fall back to ``_DEFAULTS``.
deck_named : iterable of str, optional
    The keys the deck itself wrote.  Defaults to
    ``params[_DECK_NAMED_KEYS]`` and then to "every key whose value is
    not None", so a hand-made dict behaves sensibly.  This is what
    separates "set to 100" from "defaulted to 100": without it a deck
    that pinned ``number_bands = 100`` beside ``number_bands_chi = 248``
    would be indistinguishable from one that pinned neither, and rule 4
    could not fire.

### `src/gw/gw_config.py` — `resolve_band_extrapolation`

Resolve the two spellings into ``(enabled, explicit)``.

``use_band_extrapolation`` is the key; ``sigma_band_extrapolation`` is a
TRANSITIONAL alias kept so committed decks and fixtures do not break.
Both arrive tri-state: ``None`` means the deck did not name that spelling.

Returns
-------
enabled : bool
    Whether the feature is on.
explicit : bool
    Whether a deck NAMED either spelling.  This is not decoration -- it
    selects between two different behaviours on a non-PPM ``compute_mode``
    (``gw.sigma_dispatch``): a defaulted-on key AUTO-DISABLES with a
    recorded note so that staged / static runs stay usable, while an
    explicitly-named one REFUSES.  Silently ignoring a knob the operator
    wrote down is how a green A/B comes to measure nothing.

Raises
------
ValueError
    When both spellings are named and they DISAGREE.  Refusing by name
    rather than picking a winner: whichever precedence we chose, half the
    decks that hit it would silently get the other one, and the operator
    would have no signal.  Same migration shape as ``nband`` ->
    ``number_bands``.

### `src/gw/gw_config.py` — `sigma_stage_modes`

Every :class:`ComputeMode` this RUN will dispatch a Σ under, in order.

**THIS FUNCTION EXISTS BECAUSE A PREVIOUS ANALYSIS WAS WRONG, and the
correction is worth stating rather than silently applying.**  The
2026-08-16 SC-wiring branch concluded from a ``git log --all`` /
``git grep --all`` search that ``sc_stage_N_type`` "does not exist on any
branch", and mapped per-stage behaviour onto ``compute_mode`` as the only
available proxy.  The search was run in a single-branch checkout, where
``--all`` covers only FETCHED refs, so the null was a statement about that
checkout's remotes.  The keys are real: ``origin/feat/staged-sc-2026-08-15``
(98289d77) carries ``SC_STAGE_TYPES`` (``none | cohsex | gnppm | mpa``),
``SCStage(mode, cutoff_ev, max_iter)``, ``default_sc_ladder`` and
``resolve_sc_stages`` in this file, plus ``SCConfig.stages`` and
``run_staged_self_consistency`` in ``gw.sc_iteration``.

WHAT THE REAL INTERFACE CHANGES, AND WHAT IT DOES NOT.

* It does NOT invalidate the ``compute_mode`` seam.  Read against the real
  branch, ``run_staged_self_consistency`` rebuilds each stage's inputs with
  ``dataclasses.replace(config, compute_mode_raw=stage.mode.value)`` and
  passes ``stage.mode`` into ``compute_sigma_xc``, so during a stage the
  dispatched ``mode`` **is** that stage's mode.  A per-stage guard written
  against ``compute_mode`` therefore fires per stage already.  That part of
  the SC branch was accidentally right.
* It DOES invalidate the REFUSAL.  A refusal is a statement about the whole
  RUN, and under a ladder the stage in front of you is not the run.  With
  the guard written per-stage, an explicitly-named key would kill:
  ``sc_stage_1_type = cohsex, sc_stage_2_type = gnppm`` at stage 1 (before
  reaching the very stage that consumes the key), and the SHIPPED DEFAULT
  LADDER for ``compute_mode = mpa`` — ``(GN_PPM @5 meV, MPA @2 meV)`` — at
  stage 2, after paying for a full GN-PPM stage.  Both are runs that must
  work.  Hence this function, and hence the refusal below is asked about
  the LADDER rather than about one stage.

Parameters
----------
config
    A :class:`LorraxConfig`, or anything shaped like one.  Read entirely
    through ``getattr`` so it is correct **before** the staged-SC branch
    merges (no ``config.sc.stages`` → the deck's single ``compute_mode``)
    and **after** it merges (the resolved ladder), with no edit here.
fallback
    Mode to report when the config exposes neither a ladder nor a
    ``compute_mode`` — a hand-made namespace in a unit test, or a config
    whose ``compute_mode`` property refuses.  Callers pass the mode they
    are currently dispatching, which is the only honest answer available.

### `src/gw/gw_config.py` — `band_extrapolation_is_consumable`

Does ANY stage of this run reach the kernel that reads the key?

``ppm_model is not None`` is the exact predicate: the extrapolation is
wired into the two-point GN/HL plasmon-pole Σ_c kernel and nothing else.
Deliberately NOT ``is_dynamic`` — that is True for MPA, which is dynamic
and still does not consume this key.

### `src/gw/gw_config.py` — `_deck_key_line`

Locate ``key`` in the ``[cohsex]`` section; return ``"line N"``.

Returns ``"line ?"`` when the key cannot be found on a line of its own
(it can still have been parsed — configparser accepts continuations).

### `src/gw/gw_config.py` — `_print_deck_report`

Print one deck-hygiene report on rank 0.

``process_rank`` is jax-free-safe (lazy jax import inside, falls back
to 0 when jax is absent or uninitialised) — a downhill L1→L3 import,
function-scoped so this parser stays importable without the common
package fully initialised.

### `src/gw/gw_config.py` — `read_lorrax_input`

Parse a LORRAX input file ([cohsex] section) into a params dict.

Handles the QE-style K_POINTS block and strips it before INI parsing.
All keys use ``_DEFAULTS`` for fallback values — no duplicate definitions.

### `src/gw/gw_config.py` — `_normalize_placement`

Canonicalise ``mc_average_placement`` at deck-parse time.

Delegates to :func:`gw.head_channel.normalize_placement` so the deck
parser and the consumer cannot drift on what the mode names are, and so
a typo is a refusal at config time (with the valid list in the message)
rather than a silent ``off`` two stages later.  Imported lazily: this
module is imported by the CLI before jax is configured, and
``head_channel`` keeps its jax imports function-local for the same
reason, so the cost is one numpy-only module.

### `src/gw/gw_config.py` — `scalar_head_overrides_named`

Which scalar-head overrides this deck names, formatted for a message.

Empty for every deck that leaves them alone, which is the whole point:
the envelope's ``got``/``want`` used to be eight hand-written rows and
is now one line naming only what the deck actually set.

### `src/gw/gw_config.py` — `packed_static_envelope`

THE envelope of the packed static photon operator, as ONE table.

It used to be two: six conditions inside
:func:`packed_bare_transverse_route` and seventeen inside
:func:`refuse_unsupported_bispinor_gw`, five of them restated with
separately formatted ``got``/``want`` strings (lane J section 6.2,
quality pattern #3 -- shadow accounting).  A condition that is written
twice is a condition that will differ.

Yields ``(accepted, got, want, klass, why, derived_key)`` in the order
a reader should meet them.  ``derived_key`` names the deck key that
:meth:`LorraxConfig.from_input_file` SETS for this mode when the deck
did not name it (``None`` for a row the deck must satisfy itself), so
the promotion and the refusal read the same table instead of
re-deriving each other.  ``screened`` selects the packed SCREENED mode
(``full_static_cohsex``): the extra rows are the ones that only bite
when the fifteen current ``chi`` blocks and the packed Dyson solve are
actually built.  Material class is deliberately NOT a configuration
row: it is inferred once from the loaded WFN occupations and
:func:`validate_material_inputs` refuses every fractional-occupation
non-MPA run, including this COHSEX-only screened route, and is not a
row.  The distributed-plan row (``linalg = distributed``) is
shared because the packed response facade has that one plan even when
the bare route can skip the block-diagonal current solve.  ``sys_dim`` is
also deliberately NOT here -- the bare route treats it as a routing
condition while the screened mode refuses it only under
``head_correction = full`` (``GATE
static_bispinor_photon_head_slab_only``), and one row cannot honestly
say both.

### `src/gw/gw_config.py` — `packed_bare_transverse_route`

Is the bare-transverse family served by the packed photon path?

``bare_transverse`` IS the packed static mode with the fifteen current
blocks of ``chi`` set to zero: the packed Dyson equation is then block
diagonal, ``W_packed = diag(W_00, D_TT)`` with ``W_CT = 0``, and the
sixteen-block Sigma consumer returns the screened charge COHSEX in CC,
the bare Breit exchange ``Sigma^B`` in TT (``SX(D_TT) = X(D_TT)``,
``COH(D_TT - D_TT) = 0``) and zero in CT/TC -- the incumbent
``gw.sigma_x_bispinor`` result, block for block.  The Gamma completion
is the same :func:`gw.head_correction.complete_static_slab_photon_q0`
with the charge-only ``R(q)``, which returns ``diag(W^00_h, D_TT)`` and
so inserts BOTH the charge head and the bare ``<D_TT>`` that the
``bispinor_tt_head_correction`` overlay writes into the V tiles today.

The route is taken exactly inside the envelope that completion is
derived for, and NOWHERE else: outside it the incumbent
charge-screened + ``Sigma^B`` route is the only certified one, and it
is unchanged.  Returns ``(taken, reason)`` so the driver can print the
first unmet condition instead of switching physics in silence; the
predicate and its narration have one owner.

Not in the predicate on purpose: ``bispinor_tt_head_correction``.  Its
value must not move the route -- a deck that asks for the hand TT
overlay inside this envelope is REFUSED by
:func:`refuse_unsupported_bispinor_gw` (the completion already carries
that head, so honouring both would double count it).

### `src/gw/gw_config.py` — `packed_photon_screens_current`

Whether the packed response builds and screens the current blocks.

The ONE selector between the two packed static modes.  ``True`` for
``full_static_cohsex``: sixteen ``chi`` blocks and one packed Dyson
solve.  ``False`` for the bare-transverse family on the packed route:
``chi_TT = chi_CT = 0``, so the packed solve is skipped and the CC
block alone is screened by the incumbent scalar owner.

### `src/gw/gw_config.py` — `uses_static_photon_response`

Whether screening and Sigma use the packed 4x4 photon response.

Both packed static modes: ``full_static_cohsex`` always, and the
bare-transverse family inside :func:`packed_bare_transverse_route`'s
envelope.  :func:`packed_photon_screens_current` says which.

### `src/gw/gw_config.py` — `packed_photon_replaces_charge_sigma`

Does the packed operator own the WHOLE Sigma, charge channel included?

True only for ``compute_mode = cohsex``: the sixteen-block consumer
produces Sigma_X, Sigma_SX and Sigma_COH from the packed V/W and no
scalar charge Sigma, scalar q->0 head or scalar W role survives beside
it.

False on the DYNAMIC packed route, where the charge block is the
ordinary scalar ``Sigma_x + Sigma_c(omega)`` on the same ISDF ``W_00``
and the packed consumer contributes only the fifteen current blocks
(``gw.photon_sigma`` ``blocks = "current"``).  Every driver seam that
asks "may I skip the scalar charge machinery?" asks THIS, not
:func:`uses_static_photon_response` -- the difference is exactly the
four call sites in ``gw.gw_jax`` that install head samples, persist W0,
build ``static_head_terms`` and read the scalar ``W_by_role``.

### `src/gw/gw_config.py` — `uses_dynamic_packed_photon_route`

The packed four-current operator on a frequency-dependent Sigma.

``W_packed(omega) = diag(W_00(omega), W_TT, W_CT)``: the charge block
carries the run's plasmon-pole model, the current blocks are the
``omega = 0`` packed response.  See
``reports/bisp_n_dynamic_packed_2026-09-01/DESIGN.md`` for the block
algebra and the measured bound on what freezing the current blocks
costs.

### `src/gw/gw_config.py` — `uses_coupled_photon_head`

Whether the packed photon response runs its Gamma-cell completion.

True for either packed static mode under ``head_correction = full``
(the default; the completion needs the four literal-Gamma channel
vectors, which ``gw_init`` retains only when this is true).  False
under the DEBUG setting ``head_correction = off``, where the packed
V/W keep a zero q=Gamma, G=0 slot.  No third value reaches here:
``full_static_cohsex`` refuses ``no_local_fields`` in its envelope,
and a ``no_local_fields`` bare-transverse deck never takes the packed
route at all (:func:`packed_bare_transverse_route`).

### `src/gw/gw_config.py` — `incumbent_bispinor_head_record`

``(banner, run_record_line)`` for a bispinor deck on the INCUMBENT route.

Heads are always on (owner ruling 2026-09-01,
``docs/architecture/decisions.md``; TASTE.md row 20).  The packed route
has said so since lane B: a boxed ``WARNING -- DEBUG`` banner and a
``Photon head`` line naming the completion.  The incumbent route said
only "no special Gamma-cell contribution", in component chatter that
production mode sinks -- so a headless bispinor bulk / dynamic /
``x_only`` run reached ``eqp1.dat`` with no DEBUG token anywhere in the
run record (lane J section 3.c).

Returned rather than printed so the policy has ONE owner and a test can
read it without a driver.  ``banner`` is ``""`` when there is nothing
loud to say.  The caller is :mod:`gw.gw_jax`, and only for decks with
``uses_static_photon_response(config)`` false.

### `src/gw/gw_config.py` — `refuse_unsupported_bispinor_gw`

Validate four-current modes and require live direct fields for QSGW.

``head_correction`` has TWO values on a bispinor deck, ``full`` and
``off`` (owner ruling 2026-09-01, ``docs/architecture/decisions.md``;
TASTE.md row 20).  The third scalar value is refused here for EVERY
bispinor deck, not just the packed ones -- see the gate below.

### `src/gw/gw_config.py` — `refuse_unsupported_bispinor_tt_head_correction`

Refuse ``bispinor_tt_head_correction = true`` outside its envelope.

NOT REACHABLE FROM A DECK since 2026-09-01: the key is tombstoned in
``read_lorrax_input`` and :class:`HeadConfig` is built with ``False``,
so every parsed config returns on the first line.  This function is
the guard for a HAND-BUILT config (tests, tools, an embedded caller)
that sets the field itself, which is why the driver-entry call in
``gw.gw_init.prepare_isdf_and_wavefunctions`` remains and the
parser-altitude call was dropped.  Lane N deletes both with the
incumbent non-packed route.

Two named conditions, GATE ``bispinor_tt_head_unsupported``:

1. ``bispinor = false`` — the flag corrects a bare TT V-tile that a
   non-bispinor run never builds.
2. ``sys_dim not in (2, 3)`` — box truncation's q=Γ, G=0 slot is
   already finite (``vcoul.box_0d.Box0D._v_bare_per_q`` never zeros
   it), so there is no missing slot to substitute; the bispinor
   g-flat path also does not reach sys_dim=0 today
   (``gw.v_q_g_flat`` refuses sys_dim not in (2, 3) at its own
   entry), so this is a defensive, not merely a redundant, refusal.

### `src/gw/gw_config.py` — `HeadConfig`

q→0 Coulomb-head sources, BGW vcoul override, bare-cutoff knobs.

All Coulomb-at-small-q tweaks live here.  Σ head plumbing
(``wcoul0_*``, ``vhead``/``whead_*``) is consumed by
:class:`gw.head_correction.HeadResolver`; the BGW vcoul override is
purely diagnostic (matches BGW's per-G mini-BZ averaging exactly for
bit-reproducible comparisons).

### `src/gw/gw_config.py` — `ScreeningConfig`

χ₀ / W screening: method choice + minimax-quadrature knobs.

``method`` selects the chi0 frequency treatment, and minimax is the
ONLY one LORRAX implements (owner ruling 2026-08-06).  Nothing
downstream branches on this field, and that is deliberate -- there is
no second branch to take.  Its whole job is the ``__post_init__``
check below, which is what makes it honest: before that check the
field was pure decoration, so ``screening_method = ctsp`` parsed,
normalised, and ran minimax without a word.

``diagrams`` is a DIFFERENT axis and it does have a second branch:
``method`` says how the chi0 frequency integral is taken, ``diagrams``
says which series W sums (RPA, or the BSE ladder).  The fork lives in
``gw.screening.compute_screening_model`` and nowhere else.  Its
default is spelled here as well as in ``_DEFAULTS`` so a hand-built
config -- a tool, a test stub -- takes the SAME decision the parser
would; a fallback that disagreed with the registered default is the
defect the ``restart_q_storage`` note above describes.

### `src/gw/gw_config.py` — `DynamicSigmaConfig.parsed_omega_patches_ev`

The validated ``[(lo, hi), ...]`` patch list, or ``[]``.

Parsed from ``"lo:hi, lo:hi"``.  Patches must be well-formed
(hi > lo), ascending, and separated by at least one step —
overlapping or touching patches are a deck typo, refused rather
than silently merged.

### `src/gw/gw_config.py` — `MPAConfig.sample_plan`

Return the configured double-parallel frequency plan in Ry.

This is sampling geometry only.  In particular, constructing a
metallic plan does not claim that the occupation-weighted χ/Σ
evaluators needed to consume it have landed.

### `src/gw/gw_config.py` — `SCConfig`

Self-consistency loop knobs (read only when qp_solver=self_consistent).

Promoted from the ``LORRAX_SC_*`` env vars (NEXT_TARGETS #11); the
envs are still honored as deprecated overrides at config construction
(``from_input_file`` prints a note when one is active).

- ``max_iter`` / ``tol_ev``: loop length and RMS-ΔE convergence (eV).
- ``accelerator``: ``"rcrop"`` (Anderson-style restart-CROP, default —
  required for QSGW's typical 2-cycle Jacobian) or ``"linear"``
  (plain α-mixing, diagnostic).  rCROP makes TWO ``gw_iteration_map``
  calls per accelerator iteration (trial + residual).
- ``history_depth``: rCROP history (m=5 is BGW's QSGW default).
- ``mixing``: linear-mixing α (``accelerator="linear"`` only).
- ``dump_dir``: per-iteration E/U-history .npy dump dir (None = off).
- ``exact_degeneracy_tol_ev``: maximum splitting for the symmetric
  accidental-degeneracy average.  The default is 0.1 meV; physical SOC
  splittings above it remain distinct states.
- ``tail_fit``: ``"frontier"`` uses the lowest accidental-degeneracy
  conduction manifold for the energy-only sum-band tail;
  ``"all_conduction"`` is the historical affine-fit diagnostic control;
  ``"buffer_edges"`` fits the two tails only to their adjacent diagonal
  buffers.
- ``buffer_nbands``: number of extra valence and conduction states
  evaluated around the named nval/ncond SC window.  Zero is the exact
  historical path.
- ``buffer_mode``: treatment of those extra states: diagonal-only Sigma,
  one-sided cross-edge Sigma, or a carried previous-energy reference.
- ``eigh``: which eigh diagonalises the ``(nk, nb, nb)`` carry each
  iteration — ``"native"`` (k-sharded batch: one WHOLE ``(nb, nb)``
  tile per device), ``"distributed"`` (one tile spread over the mesh),
  or ``"auto"``.  A LAYOUT choice: it does not change the physics and
  it is deliberately not a side effect of ``density_self_consistent``,
  which is what used to select it.  Resolution lives in
  ``sc_iteration._resolve_sc_eigh``.

### `src/gw/gw_config.py` — `EQP2Config`

Fixed-Sigma eigenvalue self-consistency for the opt-in eqp2 file.

This is deliberately separate from :class:`SCConfig`: it does not
rebuild G, chi0, W, or Sigma.  It repeatedly evaluates and rotates the
one-shot full-matrix Sigma(omega) table, diagonalizes the resulting QP
Hamiltonian, and tests the worst eigenvalue change in eV.

### `src/gw/gw_config.py` — `MemoryConfig`

Per-device memory budget + chunk sizing + AOT chunk-chooser flag.

``memory_per_device_gb=0`` triggers GPU auto-detection at config
construction time.  ``chunk_target_utilization=0`` is the auto sentinel;
a positive ``ISDF_CHUNK_TARGET_UTILIZATION`` value overrides the
planner's spin-aware default after clamping to ``[0.85, 1.0]``.

### `src/gw/gw_config.py` — `BSEConfig`

BSE interpolation setup (htransform-driven fine-k wfn recovery).

See ``bandstructure.bse_setup.compute_wfns_fi``.  ``get_centroids_fi``
is the master gate; if False the rest is unused.

### `src/gw/gw_config.py` — `_validate_occupation_smearing`

Validate the occupation-smearing pair without classifying the WFN.

THE WIDTH CONVENTION, stated once, here, because two keys carry it.
``occ_smearing_width_ry`` and ``occ_broadening`` are the SAME width in
different units: BerkeleyGW's ``occ_broadening``, whose MP1 argument is
``(E - mu) / (2 * width)``.  The QE ``degauss`` is TWICE it.  A deck
that sets both and disagrees is refused below rather than silently
resolved, because the two ways of being wrong (halving or doubling the
smearing) are indistinguishable in the output.

### `src/gw/gw_config.py` — `resolve_mpa_sampling_alpha`

Resolve and report the MPA sampling exponent from WFN material class.

This runs only after occupations are loaded: fractional occupations select
2, while integer-occupation nonmetals select 1.  A deck value wins.

### `src/gw/gw_config.py` — `LorraxConfig`

Unified, immutable configuration for a LORRAX GW calculation.

Created once via :meth:`from_input_file` and threaded through the
entire driver.  Top-level fields are ``hot-path`` reads (system
geometry + the orthogonal mode flags); group sub-dataclasses
organise the remaining ~70 input keys along the same axes the
input file's section comments already use.

Access pattern::

    config.compute_mode           # -> ComputeMode enum
    config.head.wcoul0_source     # head plumbing
    config.ppm.omega_p            # PPM probe ω
    config.sigma.omega_grid_ev    # shared dynamic-Sigma frequency grid
    config.debug.sigma_freq_debug_output

See module docstring for the full grouping.  ``cohsex.in`` keys
are unchanged — input files written for prior versions still parse
(the factory unflattens the dict into sub-dataclasses).

### `src/gw/gw_config.py` — `LorraxConfig.occ_broadening_ry`

THE occupation-smearing width consumed at runtime, in Ry.

One width, one owner.  Every MP1 solve in the driver reads this
and nothing else, so the two deck keys that carry the width can
no longer feed different numbers into different stages.

CONVENTION — BerkeleyGW's, not QE's.  ``gw.efermi``'s MP1
argument is ``(E - mu) / (2 * width)`` (``_mp1_values``), the same
form BerkeleyGW uses (``Common/input_utils.f90:380``), so this
width is HALF the QE ``degauss``.  Measured, not asserted: at
``degauss = 0.02 Ry`` the sodium SOC deck's BGW arm reproduces
QE's own stored occupations to 7.1e-12 with ``occ_broadening =
0.13605693122994 eV = 0.01 Ry`` (CLAIMS 185), and LORRAX's mu
lands 6.2e-7 eV from QE's E_F at the same width (CLAIMS 180).
``OccupationState.smearing_width_ry`` — the field this feeds and
the one stamped into the MPA fit store — is the same quantity
under the same name.

SOURCE.  ``occ_smearing_width_ry`` when the deck declares it (the
metal path); otherwise ``occ_broadening`` converted from eV, which
is every insulating and pre-metal deck and is bit-for-bit what
those decks used before this key existed.  When both are set
``_validate_occupation_smearing`` has already refused any
disagreement beyond ``_OCC_WIDTH_RTOL``, so the branch cannot
change the physics of a deck that carries both — it only decides
which of two agreeing numbers is the exact one, and the deck's own
Ry value is the one that did not make a round trip through eV.

NOT A DIAL.  ``occ_broadening == 0`` remains the switch that
selects step occupations (``sc_iteration._solve_head_occupations``
and the metal V_H rebuild both read it as such); this property
answers "how wide", never "whether".

### `src/gw/gw_config.py` — `LorraxConfig.compute_mode`

Resolve ``compute_mode`` from explicit input or legacy flags.

``compute_mode = auto`` (the default) infers from
``do_screened`` / ``use_ppm_sigma`` / ``ppm.model``.  An explicit
setting overrides them; the legacy fields are still parsed for
back-compat but the enum is the load-bearing axis the driver
pivots on.

RESOLVING IS NOT PERMITTING.  This property answers "which mode
did the deck ask for", and it answers it for every member of the
enum including the ones whose Σ stage has not landed — the
refusal for those is
:func:`refuse_unimplemented_compute_mode`, called at driver
entry, so that config-only consumers (the deck echo, the layering
tests, an operator reading a config back) can name the mode
without tripping over it.  ``auto`` never infers an unimplemented
mode: the legacy flags it reads predate all of them.

### `src/gw/gw_config.py` — `LorraxConfig.qp_solver`

Resolve ``qp_solver`` from explicit input or legacy flags.

``qp_solver = auto`` (the default) resolves:

1. ``self_consistent = true`` → ``SELF_CONSISTENT`` (deprecated
   key, still honored);
2. else → ``ONE_SHOT_DFT`` — the one-shot full-matrix
   effective-Hamiltonian route is the default.
   (The deprecated ``sigma_at_dft_energies = true`` alias also
   lands here: its intended meaning — authoritative at-DFT QP
   evaluation — IS the default.)

An explicit setting overrides the legacy flags, mirroring how
``compute_mode`` absorbs ``do_screened`` / ``use_ppm_sigma``.

Validation (mutually inconsistent axis combinations):

- ``fixed_point`` × static mode → error (no ω-grid to solve on;
  a silent no-op would blur the axis).

### `src/gw/gw_config.py` — `LorraxConfig.omega_grid_ev`

Σ_c(ω) frequency grid in eV (length-stable single formula).

``n = floor((max−min)/step + 0.5) + 1`` — the Ry grid is derived
from this one by division so the two can never disagree in length
or accumulate independent float-step rounding.

With ``sigma_omega_patches_ev`` set, the grid is the union of the
patches, each built by the SAME length-stable formula, ascending
by the patch validation.  ``sigma_omega_min/max_ev`` are ignored
then — the patches ARE the grid.

### `src/gw/gw_config.py` — `LorraxConfig.from_input_file`

Parse input file and resolve runtime settings (memory, env vars).

Replaces ``read_cohsex_input`` + ``resolve_runtime_config`` +
path resolution in one call.  Returns a ``LorraxConfig`` with
sub-dataclasses fully populated.
unknown-key policy. ``runtime_platform`` is an injected ``cpu`` or
``gpu`` answer for a preflight that has no target device. With
``resolve_hardware=False``, an auto memory budget stays at its zero
sentinel and no device-memory probe is made; explicit deck budgets
remain resolved. Production callers use all defaults.


### Input parsing and envelope phase rulings (2026-09-06)

```text
        # --- Memory auto-detection ---
        # --- Chunk utilization from env ---
        # 0.0 (default) = auto: the planner uses its ns²-aware default
        # (higher for scalar, lower for bispinor's 4× pair density).  A
        # positive env value overrides it, clamped to [0.85, 1.0].
        # ``env_float`` announces a non-numeric value instead of swallowing
        # it — the bare ``except Exception`` here left the user believing a
        # utilization was in force when it was not.
        # Resolve the bundled metallic q0 contract before constructing any
        # typed group.  ``mc_average_vcoul_body`` defaults to true for every
        # historical deck, but BGW's noavg metal comparison requires false.
        # Only an EXPLICIT contradictory value refuses: an absent key is the
        # compatibility case this bundle exists to override, while an
        # explicit false is already compatible and remains visible in the
        # provenance line.
            # An explicit spelling of the shipping default must serialize to
            # the same LorraxConfig as an absent key.  ``raw_input_keys`` is
            # otherwise the one field that would distinguish them.
        # --- Build sub-dataclasses ---
            # NOT a deck key any more (tombstoned above).  The field stays
            # so the incumbent non-packed TT overlay owner
            # (gw.v_q_bispinor._make_per_q_v_builder_for_tile) keeps ONE
            # place to read, and so a hand-built config that sets it True
            # still meets refuse_unsupported_bispinor_tt_head_correction.
            # Lane N deletes the field with the incumbent route.
        # With patches, omega_min/max_ev ARE the patch hull.  Consumers
        # read these fields as "the Σ grid's reach" (the SC partition's
        # in-grid classification above all); leaving them at the deck
        # defaults silently scissored every band outside [-5, +5] on the
        # first patched run — Σ was computed on the deep clusters and
        # then never consulted (measured: arm 21, SC partition 2/48).
        # SC loop knobs.  The LORRAX_SC_* env vars are deprecated overrides
        # of the sc_* input keys (kept so existing sweep scripts run
        # unchanged); a note is printed whenever one is active.
            # No env override: the LORRAX_SC_* envs are deprecated and a
            # new knob must not add one.
        # Lower the one layout profile to platform libraries.  This is a
        # capability choice inside the already-resolved profile: GPU uses
        # cuSolverMp/cuBLASMp; CPU uses ScaLAPACK.
        # Same treatment, same reason, for the restart q-set.  Validated
        # here and NOT resolved here: ``auto`` resolves against the closure
        # answer, which needs the run's centroid set and its symmetry
        # tables, so the field below is the RAW request and
        # ``gw.restart_q_storage.resolve_restart_q_storage`` turns it into a
        # mode once those exist.  The ``_raw`` suffix says which kind it is — the
        # same convention ``compute_mode_raw`` / ``qp_solver_raw`` use.)
        # The ``or`` fallback must agree with ``_DEFAULTS`` — it is reached
        # only by a caller that built the params dict by hand and left the
        # key out, and a fallback that disagreed with the registered default
        # would make THAT caller silently take a different storage decision.
        # Same ``or`` caveat as above: this fallback is reached only by a
        # hand-built params dict and must agree with ``_DEFAULTS``.
        # BAND COUNTS.  ``read_lorrax_input`` already resolved them (once) and
        # left the answer in the params dict; a hand-made dict that never went
        # through the parser gets resolved here instead.  Either way there is
        # exactly one ``resolve_band_counts`` call per config.
        # ζ-fit window top.  Empty / unset collapse to None — "follow the
        # loaded window".  An EXPLICIT value is stored verbatim, INCLUDING one
        # that equals ``bands.isdf``.
        #
        # WHY IT IS NO LONGER ERASED HERE (2026-08-22).  This used to rewrite
        # ``zeta_nband == bands.isdf`` to None, reasoning that a redundant
        # restatement of the default must take the default path "pad and all".
        # It is not redundant, because ``bands.isdf`` is the LOGICAL count and
        # the edge the fit actually gets is ``BandSlices.b4`` — that count
        # ROUNDED UP to the world size.  On P=4 a scalar-Si deck with
        # ``nband = zeta_nband = 14`` silently fitted [0,16) and then refused,
        # correctly, because band 16 cuts a multiplet; the deck had asked for
        # 14 and no banner ever said otherwise (JID 57152792,
        # runs/Si_scalar/11_scalar_v_rootcause_20260817/).
        #
        # The collapse still exists — it just happens where the padded edge is
        # known, in ``gw.gw_init.resolve_zeta_fit_edge``, which is also the one
        # place the banner and the three fit-window consumers read.  A deck
        # whose ``nband`` already divides the world size is unchanged.
            # Top-level: system + mode flags
            # Build from a stable sequence.  Equal sets reached through an
            # absent key versus an explicit default can retain different
            # hash-table histories; pickling those frozensets then need not
            # be byte-identical even though the typed values compare equal.
            # Compatibility mirror only.  Every new head decision reads the
            # enum above; keeping this resolved bool prevents old consumers
            # from disagreeing with ``head_correction = off``.
            # Sub-dataclass groups
            # Parsed blocks
        # ``density_self_consistent`` is still an independent physics choice
        # for scalar QSGW, whose conventional fixed-density path remains the
        # default.  Bispinor QSGW has no corresponding safe fixed-density
        # treatment: both rho and the signed Dirac current J must follow the
        # evolving occupied orbitals.  Normalize the UNNAMED default here,
        # after the canonical qp_solver resolver exists, instead of duplicating
        # its legacy/explicit precedence logic.  An explicit false survives
        # unchanged and the gate below refuses it rather than overriding what
        # the user wrote.
        # Fresh physics is the global default.  A file existing in ``tmp``
        # is not permission to replace a live fit: only an explicit
        # ``restart = true`` enters the restart loader, whose provenance
        # gates authenticate the tensor set before use.
        # CROSS-KEY, and therefore after the record exists: the w_bse
        # refusals read resolved axes (compute_mode / qp_solver fold in the
        # legacy flags), and the honest way to ask which mode a deck chose
        # is to ask the resolver, not to re-derive it here.  A w_rpa deck
        # returns from this call before either property is touched.
        # ONE CANONICAL VOCABULARY FOR THE SELF-ENERGY AXIS, and a note for
        # the other one.  Same position and same reason as the two refusals
        # above: the announcement quotes the RESOLVED axes, which only the
        # record can answer.  Honoring a legacy key in silence beside a
        # canonical twin is how a tree ends up with two vocabularies for one
        # axis and no way to tell which one a run went through.
    # Locate [cohsex] section
    # Locate optional K_POINTS block
        # Strip K_POINTS from INI text
        # inline_comment_prefixes so 'key = off  # note' parses to 'off', not
        # 'off  # note' (the latter silently voided flags — a real footgun).
        # Legacy key check
        # RETIRED-KEY REPORT.  A key with an explicit legacy branch is
        # exempt from the unknown-key check below so one deck key never
        # draws two messages — but that exemption left
        # ``warnings.warn(..., DeprecationWarning)`` as the ONLY report,
        # and Python's default filter hides DeprecationWarning outside
        # ``__main__``.  A retired key was therefore parsed, matched,
        # ignored, and announced to nobody, which is exactly the failure
        # the unknown-key check exists to prevent.  Collect every hit and
        # print it through the same rank-0 reporter, in wording that keeps
        # "retired" (the key was real once, and here is what replaced it)
        # distinct from "unrecognized" (nothing ever read this).  The
        # DeprecationWarnings stay — they are what a library consumer
        # filters on.  Explicit refusal branches below own keys whose
        # replacement must be named rather than hidden in a generic error.
        # ``chunk_size`` (legacy band-chunk knob) was a no-op: its only
        # consumer wrote ``meta.chunk_size``, which nothing ever read —
        # chunk sizing is owned by the gflat planner.  Dropped 2026-07-09.
        # There is one sharded-slab transport and the deck does not select
        # it.  Refuse the deleted selectors by name: accepting and ignoring
        # them made stale decks look as though their requested HDF5 route was
        # still active.  The tombstones stay in ``_LEGACY_DECK_KEYS`` only so
        # strict unknown-key handling does not mask this specific message.
        # ``sigma_omega_accumulation`` was REMOVED (2026-08-14): host-tile
        # accumulation is the only mode, so the key steered nothing.  The
        # long-removed ``kij_stream`` VALUE keeps its dedicated refusal.
        # Deprecated qp_solver aliases (still honored via auto-resolution;
        # see ``LorraxConfig.qp_solver``).
        # REMOVED keys (owner-approved deletions, 2026-07-31; these behave
        # like any other unknown deck key — reported by the unknown-key
        # check below, never steering anything): ``isdf_memory_mode``
        # (two-plan W cleanup — the W Dyson solve is selected by
        # w_dyson_solver=local|distributed) and the legacy aliases
        # ``cusolvermp_charge``/``cusolvermp_lu`` (use the ``linalg`` dial).
        # --- Unknown-key check -----------------------------------------
        # Every key in the deck that is neither in ``_DEFAULTS`` nor
        # handled by one of the explicit legacy branches above is refused
        # in ONE aggregated error (key and line number).  Deck parsing is
        # always strict; there is no mode in which a typo is ignored.
        # Retired keys are exempt (they got their own report above).
        # configparser lower-cases option names (``optionxform = str.lower``),
        # so iterating ``section`` yields ``do_g0`` for a deck that writes
        # the documented ``do_G0`` -- the ONE non-lower-case key among the
        # 99 in _DEFAULTS.  Comparing the two raw made that key BOTH
        # honoured and unrecognised at the same time: ``section.get`` folds
        # the LOOKUP too, so ``do_G0 = false`` really did steer the run,
        # while this check reported it as an unknown key -- and, under
        # ``strict_keys``, REFUSED a valid deck outright.  Fold both sides
        # so recognition matches the lookup that already happens.
        # Build params from _DEFAULTS, overriding with parsed values
        # WHICH KEYS THE DECK ITSELF NAMED.  ``params`` cannot answer this
        # afterwards — a deck pinning a key to its default and a deck that
        # never mentions it produce the identical entry — and the difference
        # matters to anything that must speak only to decks that opted in.
        # Its first consumer is the ``restart_q_storage`` deprecation notice
        # (owner ruling 2026-08-08: the key is scheduled for deletion), which
        # must fire for a deck that pins it and stay silent for the other
        # ~forty, or it is noise nobody reads.  Recorded here, where the
        # answer is free, rather than re-parsed by each consumer.
                # Tri-state boolean (default None = unset); an explicit
                # value parses as bool.
                # Nullable float (vhead, whead_0freq, etc.)
    # Deck dial interpretation point (INVARIANTS row 19).  Every downstream
    # consumer reads this immutable record; no stage reinterprets ``linalg``.
    # --- Band counts: resolve ONCE, here ------------------------------
    # ``number_bands`` / ``number_bands_chi`` / ``number_bands_sigma`` /
    # ``nband`` collapse into two numbers plus their max, and this is the
    # only call to the resolver on the deck path.  Resolving here rather
    # than in ``LorraxConfig`` is what lets the params dict stay honest for
    # the tools that read it directly (``bandstructure.htransform``,
    # ``psp.get_DFT_mtxels``, ``gw.kin_ion_io``, ``file_io.epsreader``):
    # they ask for ``params["nband"]`` and must get the LOADED band extent,
    # which after the split is ``max(chi, sigma)`` — the same number they
    # always got on an unsplit deck.
    #
    # The mirror is why this is not idempotent and why the answer is
    # cached in ``params[_BAND_COUNTS]`` instead of being re-derived: after
    # the write-back, ``nband`` no longer says what the DECK said, so a
    # second ``resolve_band_counts`` on this dict would see an umbrella that
    # the deck never wrote.
    # Parse optional QE K_POINTS block
```
