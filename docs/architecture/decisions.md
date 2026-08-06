# Design decisions

Dated, binding rulings from the code owner. Each entry states the decision,
its consequence for code, and what it licenses deleting. Newest first.
These override older prose anywhere in the tree.

## 2026-08-05 — One Coulomb kernel; the mini-BZ sampler bug is fixed

Owner ruling: consolidate `v(q+G)` onto one formula and one Monte-Carlo
sampler, INCLUDING the physics change, and keep every feature that only one
of the ~10 copies had.

**The bug.** `gw/compute_vcoul.py::build_v_head_miniBZ_avg_3d` drew mini-BZ
samples as `randvals @ bvec.T`. The lattice period is the ROWS of `bvec`
(`q_cart = q_frac @ bvec` everywhere else in the tree); its columns are not in
general a period, so the parallelepiped being sampled was not a fundamental
domain and the Voronoi wrap that follows was not measure-preserving. The
mini-BZ average was then biased. `gw/coulomb/base.py:152` had the correct
`U @ bvec` and carries a comment about exactly this bug class — it was fixed
there in 2026-07 and never propagated back. Introduced 0c8083d (2026-04-05),
live and default-on (`mc_average_vcoul_body = True`, `gw_config.py:1142`;
gated to `sys_dim == 3` at `v_q_g_flat.py:540`).

**`allclose(bvec, bvec.T)` IS THE WRONG PREDICATE, and this correction
matters.** The audit that opened this work — and this entry's own first
draft — classified cells by whether `bvec` is symmetric. The estimator is
unbiased under the weaker condition that `bvec.T` is a SIGNED ROW-PERMUTATION
of `bvec`: then `u @ bvec.T` and `u @ bvec` are different points but the same
distribution on the same parallelepiped, so only the MC realisation changes.
Measured against a REJECTION ground truth (uniform on the mini-BZ Voronoi
cell by construction, sharing no code with the production sampler), max and
mean relative error over the 63 nonzero q of a 4x4x4 grid — Frontera job
7890650:

| cell | bvec.T a signed row-perm of bvec | deleted `bvec.T` | fixed `bvec` | MC self-noise |
|---|---|---|---|---|
| cubic | yes | 3.0e-03 | 3.0e-03 (identical) | 3.8e-03 |
| **Si WFN — the only 3D deck on this machine** | **yes** (perm (2,0,1)) | **3.7e-03** | 4.4e-03 | 2.2e-03 |
| fcc from `2pi inv(A).T` | no | **6.4e-01** / 8.4e-02 mean | 3.7e-03 | 2.2e-03 |
| hexagonal a=5.9 c=23 | no | **1.7e-01** / 6.5e-02 mean | 3.9e-03 | 3.7e-03 |
| triclinic (generic) | no | **2.3e-02** / 7.4e-03 mean | 2.6e-03 | 2.5e-03 |

So: the defect is real and up to 64% per-q on a genuinely skewed cell — a 3D
HEXAGONAL deck would have been wrong by ~6.5% in the mean — but the one 3D
deck that exists here happens to sit in the benign class. Both halves of that
have to be said. An earlier draft of this entry, and the commit messages of
the consolidation steps, cited a "4.25% fcc ibrav=2" figure for the Si deck;
that number belongs to an fcc `bvec` CONSTRUCTED as `2pi inv(A).T`, not to
the one pw2bgw wrote into this fixture. Corrected here.

**CORRECTION, same day (Frontera job 7890705).** Two things above are wrong,
in the same direction as the error they were correcting.

*The predicate is UNIMODULARITY, not "signed row-permutation".* The columns
of `bvec` span a fundamental domain of its ROW lattice — which is all the
Voronoi fold needs to stay measure-preserving — iff `M = bvec.T @ inv(bvec)`
is an INTEGER matrix with `|det M| = 1`. A signed row-permutation is a
special case. Row-permutation is a stricter test than the physics requires,
so it produces FALSE ALARMS: it files unbiased cells as biased.

*The `fcc from 2pi inv(A).T` row is one of those false alarms, and its
6.4e-01 is misattributed.* That cell has
`M = [[0,0,1],[1,1,1],[-1,0,0]]` — integer, `det = +1`, and NOT a
permutation. It is in the BENIGN class. Re-measured against a rejection
ground truth whose box is validated by `acceptance x box volume = |det
B_miniBZ|` (the 2026-08-05 draft's box CLIPPED 4-10% of the Voronoi cell on
skewed cells and manufactured a spurious 3e-02 "residual bias" in the FIXED
sampler), with the fold width held at `nmax=3` so the fill is isolated from
the separate `nmax=1` defect of the same deleted routine:

| cell | M integer, \|det\|=1 | deleted `bvec.T` at nmax=3 | self-noise |
|---|---|---|---|
| cubic | yes (M = I) | 3.9e-03 | 3.0e-03 |
| fcc from `2pi inv(A).T` | **yes** (M != I) | **5.0e-03** | 3.4e-03 |
| hexagonal a=5.9 c=9.4 | no | 1.5e-01 | 2.3e-03 |
| rhombohedral a=6.4 alpha=68 | no | 8.5e-02 | 2.5e-03 |
| triclinic 4.6/6.1/8.3, 62/78/104 | no | 6.5e-02 | 4.4e-03 |

(max over the 63 nonzero q of a 4x4x4 grid, relative; 150k accepted
rejection samples per cloud.) So on that fcc cell the deleted fill is
unbiased at the MC self-noise, and the 6.4e-01 in the table above is the
`nmax=1` fold, not `bvec.T`. The two defects were reproduced together and
reported as one.

What survives unchanged: the defect is real, it is up to 15% per-q on a
hexagonal cell from the fill ALONE, no deck in the tree was in the biased
class, and the Si-fixture conclusion (benign, 1.63 meV is re-seeding) is
untouched — a permutation is unimodular.

Gates: `tests/test_minibz_sampler_lattice_classes.py` (numerical, hexagonal
/ rhombohedral / triclinic + cubic control, tolerance set from self-noise
measured in the same run) and `tests/test_minibz_fill_wrap_convention.py`
(structural — `gw.coulomb.sampler.assert_fill_matches_wrap` refuses on the
fill path when the cloud is filled from one basis and folded against
another, self-tested at import, and an AST check that no fold site in the
tree can skip it).

**Measured Sigma impact** (Frontera job 7890626): bulk Si 4x4x4, `sys_dim = 3`,
`mc_average_vcoul_body` at its default, the two sampling lines as the only
difference, everything else byte-frozen:

    max |d sigTOT| = 1.63 meV     MAE = 0.147 meV
    max |d sigCOH| = 1.63 meV     max |d sigSX| = 0.12 meV
    VH and Eo bit-identical

Because this cell is in the benign class, that 1.63 meV is a RE-SEEDING
effect — a different Monte-Carlo realisation of an unbiased estimator — not a
bias being removed. It is nevertheless a real change to a pinned number
(noise floor for the comparison, arm A vs the previous GPU-generated
reference: 0.054 meV; the move is 30x that and above the gate's own 1 meV
atol), so `eqp_si_ref.dat` is re-pinned. It is NOT evidence that any past Si
number was wrong.

**Which past results are affected: NONE.** A filesystem-wide sweep of
Frontera (`/work2`, `/scratch1`, `/scratch2`, `/home1`: 2669 decks, 567 run
dirs) found that no deck anywhere sets `mc_average_vcoul_body`, so all run at
the default true; 2438 decks are `sys_dim = 2` and 152 leave it unset, where
the default is 2 (`gw_config.py:851`) — `sys_dim == 2` short-circuits the call
entirely. Every `sys_dim = 3` deck on the machine is one file, the
`si_cohsex_debug` fixture (79 identical copies), and its cell is in the
benign class above. Nothing needs re-running. The only artifact that moves is
the frozen reference, and it moves by MC reseeding.

**The new sampler is also more accurate.** Against the same ground truth on
the Si cell, head-table error drops from 4.0e-03 max / 9.2e-04 mean to
4.7e-04 / 1.6e-04 — about 8x — from `nmax` 1 -> 3 (BGW `ncell`), scrambled
Sobol instead of `RandomState(42)` uniform, and BGW's adaptive per-q sample
count. Cost on the Si 4x4x4 deck: 7.2 s for the whole 64-q table, one draw
reused across q (job 7890648).

**Kept, because exactly one copy had each.** The `slab_sr`
`-expm1(-K^2/4a^2)` channel (only `bse/vq_interp.py:319`); the full
{bulk, slab} x {full, lr, sr} product; the `argmin_G |q+G|^2` head-slot rule
for the BSE arbitrary-Q path; `bse.vq_interp.minibz_head_vlr`'s rank-invariant
`fold_in(key, global_slot)` distribution, which is why the sampler
distributes by default; and `make_eval_vq._body`'s `out_shardings` /
`with_sharding_constraint` structure, which the shared kernel is called
INSIDE, never around.

**Two guard tolerances, not one.** `TOL_QG_ZERO = 1e-12` identifies the exact
q=G=0 reciprocal-lattice slot. `TOL_MC_NAN = 1e-24` is a 0/0 guard on
Monte-Carlo draws, where a sample at |K|^2 ~ 1e-13 is a legitimate draw from
an integrable integrand and zeroing it biases the estimator low. They are not
interchangeable in either direction.

**REFUSAL: do not zero `v(G=0)` at `q != 0`.** Only `|q+G|^2 < TOL_QG_ZERO`
is zeroed. Zeroing the whole G=0 column is the natural tidy-up and it is
wrong: measured, it moves the BSE makeVq-vs-disk residual from ~1e-9 to 0.33
(`bse/vq_interp.py:325-328`). Guarded by
`tests/test_coulomb_kernel.py::test_G0_at_finite_q_is_NOT_zeroed`.

**Volume convention.** `q0_average` returns BARE (no `1/Omega_cell`).
`Bulk3D`/`Slab2D` were right; the `CoulombKernel` docstring claimed the
opposite and `Box0D` divided. `units` is now a required keyword on the kernel
with no default, and `get_kernel` ASSERTS `q0_units == "bare"` rather than
trusting a docstring.

**Pad policy is genuinely divergent and both sides are correct.**
`compute_v_q_per_G` leaves pad slots evaluated (the G-flat contract has
zeta-tilde = 0 there); `bse.vq_interp.v_sphere_padded` zeroes them (its `ZG`
carries junk in the pad columns). This is now an explicit `pad_policy`
parameter on both, defaulting to what each consumer needs. Silently picking
one corrupts whichever loses.

**Deleted.** The `gw/coulomb` per-dimension `v_qG` layer (never had a caller
from its creating commit d5b5119 to 2026-08-05), `Slab2D._vq_2d`,
`Slab2D.v_head_minibz_avg`, `psp/finite_q_head_interp.{v_head_3d,
v_head_2d_slab}`, and `build_v_head_miniBZ_avg_3d`.

**NOT merged, each for a reason.** `compute_vcoul_0d.compute_vcoul_box` (a
real-space Wigner-Seitz FFT — a different algorithm, not a truncation
factor); `file_io/read_bgw_vcoul.py` + `fill_v_grid_for_q` (a deliberate
independent cross-check, `use_bgw_vcoul`; merging it destroys its purpose);
`psp/dft_operators.solve_poisson` (DFT Hartree at q=0);
`v_q_bispinor`'s `eps_K2` (a projector guard on K-hat, a different quantity).

**OPEN, deliberately not decided here.** The head-SLOT rule in the GW driver
was left at Miller-(0,0,0). See the "Coulomb head slot" item in
`docs/dev/STATE.md`.

Licenses deleting: any remaining private spelling of `8pi/|q+G|^2`, any second
mini-BZ sampler, and the `nmax=1` / `RandomState(42)` / non-adaptive MC
settings.

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

## 2026-08-01 — Square process meshes only; nonsquare P truncates

Only square 2-D device meshes are supported. When a run is launched with
a nonsquare process count P, the mesh resolver uses s = floor(sqrt(P)),
builds the s x s mesh, and announces that P - s^2 processes are idle;
it does not build a rectangular mesh and does not refuse. Launch scripts
should request square counts; the truncation is the safety net, not the
recommendation.

Rationale: rectangular meshes complicate ScaLAPACK grid geometry and the
divisibility contracts for no measured benefit at our scales (ruled
2026-07-27; this entry adds the truncation default, 2026-08-01).

Licenses deleting: rectangular-mesh accommodation in the mesh resolver
and any divisibility contortions that exist only to serve non-square
grids.

## Standing (recorded earlier, restated for one-page reference)

- Thousands-of-low-memory-processes is the scaling target: no N_mu^2-class
  object may be required to fit on one rank in the large-P limit
  (2026-07-27). Two plans per solve family: a local whole-tile plan and a
  distributed plan; execution schedules of a plan are not new plans.
- No time-reversal-symmetry-based reductions; no DFT-as-matmul (standing
  vetoes).
- Every performance claim carries a job id and an on-disk artifact.
