# ρ and ⟨mk|V_H|nk⟩ on the 2-D process grid — design spec

Status: SPEC + OWNER RULINGS (see §4.1). Implementation in progress; §9 is the order.
Author's measurements: jobs 7885315/7885316 (b600/P=64 charge), 7885966
(b600 bispinor), and the code as of `eff3bb7`.

---

## 1. Why this exists

`gw/kin_ion_io.py` states its communication contract in four lines, and three
of them are already right:

| stage | partition | verdict |
|---|---|---|
| ρ(r) | over (k, band-chunk), one psum of `N_r` f64 | fine |
| V_H(r) = Poisson | replicated, zero collectives | fine |
| ⟨mk\|V_H\|nk⟩ | **over k only**, one all-gather | **the problem** |
| ψ load | process-local per (k, band) window | fine |

The matrix-element sweep is **1.05e12 of the 1.2e12 flops** at 12×12 — the
module says so itself — and it is the one stage partitioned over the axis that
does not scale.  Three walls follow, and they bite in this order:

**W1 — the per-k full-band FFT box.**  `load_kpoint_fftbox_local(wfn, meta,
ik, nb)` puts **every** band at one k into the FFT box on **one** rank:
`nb · nspinor · N_r · 16` bytes.

| deck | nb | nspinor | N_r | per-rank |
|---|---|---|---|---|
| b600 bispinor | 600 | 4 | 46 080 | **1.77 GB** |
| 12×12, nb=600 | 600 | 4 | 972 000 | **37 GB — OOM** |

This is a hard wall, not a slowdown, and it is reached before either of the
others matters.

**W2 — the replicated `(nk, nb, nb)` output.**  `gather_k_blocks` returns an
array "identical on every rank", and the driver's gspace V_H route *requires*
that (non-`owner_only`) mode.  Complex128:

| deck | size/rank |
|---|---|
| b600 (16 k, 600 b) | 92 MB |
| 12×12 (144 k, 600 b) | 829 MB |
| 12×12, nb=2000 | **9.2 GB** |
| dipole `(nk,3,nb,nb)`, 12×12, nb=2000 | **27 GB** |

A per-rank `nb²` object is the same class the scaling doctrine forbids for
`N_μ²`.

**W3 — idle ranks.**  Efficiency is `min(P, nk)/P`.  At b600/P=64 with nk=16,
**48 of 64 ranks are idle** through 87 % of the flops; the b600 report already
records it ("dipole and kin-ion cannot use more than 16 ranks on this deck").
At P=1024 with nk=144 it is 14 %.

Same pathology as `zeta_fit.cholesky`'s `P = n_q` ceiling.

---

## 2. What the kernel already is

`psp/get_DFT_mtxels._compute_local_V_k_jit` is already FFT + one GEMM:

```python
psi_r  = ifftn(psi_G) * scale            # nb·nspinor 3-D FFTs
phi_r  = psi_r * V_r
phi_G  = fftn(phi_r) * (deltaV*fft_norm) # nb·nspinor 3-D FFTs
psi_coeffs = psi_G[:, :, Gx, Gy, Gz]     # box -> sphere, NO FFT
vpsi       = phi_G[:, :, Gx, Gy, Gz]
V_loc = einsum('bsg,nsg->bn', conj(psi_coeffs), vpsi)
```

So "local GEMM over G against V(G)" is not a change of algorithm — it is the
contraction that is already there.  The work is re-sharding it.

**The asymmetry that makes this cheap.**  The `m` side never needs an FFT:
`psi_coeffs` is a straight gather of the *stored* G-sphere coefficients.  Only
the `n` side pays the ifft → ×V_r → fft round trip.  The current code obtains
both from the same full FFT box, which is why W1 is sized by `nb` rather than
by the n-block alone.

---

## 3. The physics constraint: there is no fusing ρ into V_H

V_H(r) is the Poisson solve of the **completed** ρ, which sums over all k and
all occupied bands.  So ρ → V_H is a global barrier: no schedule can start a
matrix element before every rank's ρ contribution is reduced.  "Fusing ρ and
V_H" therefore means *sharing one sharding discipline and one ψ-loading
pass structure across two sweeps separated by a barrier* — not merging them
into one kernel.  Any design claiming otherwise is wrong.

What can genuinely be shared: the mesh, the band-block map, the loader
call, and the FFT routing.

---

## 4. Self-consistency: what this must support

**Today's SC loop is fixed-density QSGW.**  `gw/sc_iteration.py` step 4 rotates
`(V_H + Σ_xc)` back to the DFT basis; `sigma_dispatch` applies
`hartree_basis_rotation` (`U†·V_H·U`), and `_hartree_cache` serves one V_H
built once from the DFT density.  The cache comment says so outright: "the SC
loop rotates the band basis, it does not rebuild ρ".

`U†·V_H[ρ]·U` is exact — but only as a **basis change of a fixed operator**.
It is not charge self-consistency, and it cannot become charge
self-consistency by any choice of U.  Real charge SC needs
`V_H[ρ_cur]` rebuilt from the current occupied orbitals
`ψ^cur_n = Σ_m U[k,m,n] ψ^DFT_m`, then expressed in whichever basis the caller
wants.

The `psi_rotation` seam for that exists in `build_valence_density_distributed`
and `compute_hartree_matrix` — and **no caller anywhere passes it**.  It is
plumbed, documented, and dead.

### 4.1 OWNER RULINGS (2026-08-02) — these are decided, not open

1. **Density-updating QSGW is the target.**  ρ is rebuilt each iteration;
   `psi_rotation` gets wired up.  So ρ's cost is paid per iteration, not once,
   and §5.2 must keep a k-only mode (band chunking is illegal once the
   rotation couples all `nmix` bands).
2. **The Hartree basis rotation stays** — it is how step *n*'s V_H becomes
   step *n+1*'s.  Rebuilding ρ does not remove it: Σ is computed in the
   current QP basis, so V_H must be rotated to match before it is added.
   The two are complementary, not alternatives.
3. **The rotation must be DISTRIBUTED**, through the linalg FFI, in the same
   2-D sharded stacked layout — not gathered to single procs.  Today it is
   `jnp.einsum('kpm,kpq,kqn->kmn', conj(U), v_h_ext, U)` on a **replicated**
   `(nk,nb,nb)`, which is W2 again and is incompatible with the sharded
   output of §5.1.  With V_H sharded `P('x','y')` the rotation is a
   distributed `U†·V·U` per k; the existing distributed-linalg plumbing
   (the ScaLAPACK/SLATE host targets already linked into the FFI) is the
   place to look, and the owner's instruction is explicitly to prefer the
   distributed form over saving a few seconds by gathering.
4. **ψ is carried as `psi_m_X,k(G)` and `psi_n_Y,k(G)`** — G-sphere, not the
   FFT box.  This is what removes W1, and it is also the natural input to the
   contraction, which already consumes sphere coefficients.
5. **V_q=0(G) is REGENERATED**, not stored or gathered.  It is a closed-form
   function of the reciprocal metric (plus the 2-D Ismail-Beigi truncation
   when `sys_dim == 2`); recomputing it on each rank costs nothing and
   removes a replicated array and a consistency hazard.  It MUST use the
   same truncation convention as `kin_ion`'s V_loc — the module docstring
   already warns that mixing conventions puts a large systematic error
   straight into H₀ where it cannot be told from a basis-convergence
   problem.
6. **Unify the k-weighting of the IBZ-only charge-density routine.**
   `rho_from_wfn_ibz` → `compute_valence_density` currently decides weights
   by the heuristic `use_kweights = (nk_local == len(wfn.kweights))`, falls
   back to uniform `1/nk_tot` otherwise, and is all-k-resident and
   replicated.  It must share ONE k-weighting convention (and the same
   sharded ψ layout) with `build_valence_density_distributed`.  That is both
   the correctness unification and, because it removes a fully replicated
   all-k sweep, the "a lot faster" the owner is after.

Consequences this design must honour:

1. **ρ must be cheap to rebuild**, because a density-updating loop pays it
   every iteration, not once.  This is the main reason to care about ρ's
   sharding at all.
2. **`invalidate_hartree_cache()` must be called at the top of each
   iteration** or the loop silently keeps iteration 0's V_H for ever.  The
   cache comment warns about exactly this.  A density-updating loop that
   forgets it produces a plausible, converged, wrong answer.
3. With `psi_rotation` supplied the band axis of ρ **cannot** be chunked
   (the mixing couples all `nmix` bands), so the ρ plan must keep a
   k-partitioned mode that does not assume band splitting.

---

## 5. The design

Mesh `('x','y')`, `P = px·py`, square (`resolve_mesh` enforces it).

### 5.1 ⟨mk|V_H|nk⟩ — 2-D band sharding, k outer

Bands `m` over `'x'` (`nb_x = ceil(nb/px)`), bands `n` over `'y'`
(`nb_y = ceil(nb/py)`).  For each k, **all P ranks** participate:

1. **m side (no FFT).**  Rank `(i,j)` loads the G-sphere coefficients for its
   m-block: `(nb_x, nspinor, nG)`.  Straight from the file layout; never
   boxed.
2. **n side (FFT, split over the full mesh).**  Rank `(i,j)` boxes and
   transforms only `nb/P` bands — the `i`-th slice of block `By_j` —
   ifft → ×V_r → fft → gather to sphere.
3. **All-gather along `'x'`** so every rank in column `j` holds the whole
   `vpsi` for `By_j`: `(nb_y, nspinor, nG)`.
4. **Local GEMM**, unchanged: `einsum('bsg,nsg->bn', conj(psi_m), vpsi_n)` →
   the `(nb_x, nb_y)` block this rank owns.  **No reduction** — G is
   contracted rank-locally.

Step 2's split is the point.  The naive version has each of the `px` ranks in
column `j` transform the whole of `By_j`, which is `px`-fold redundant FFT
work; splitting first and all-gathering the (much smaller) sphere-space result
makes FFT work `2nb/P` per rank — optimal, no redundancy — at the cost of one
all-gather of `nb_y·nspinor·nG·16` bytes per k (**9.4 MB** at b600/P=64).

Output is `(nk, nb, nb)` sharded `P('x','y')` per k.  **No rank ever forms a
full `(nb,nb)` tile.**

### 5.2 ρ — keep the existing plan, put it on the same mesh

ρ is already (k, band-chunk)-partitioned over all P with one `N_r` psum, and
that is optimal: both axes are free and the reduction payload is 1.4 MB.  The
change is cosmetic-but-useful — express the work map on the same `('x','y')`
mesh and the same band-block boundaries as §5.1, so one loader pass serves
both and the two sweeps cannot disagree about who owns which bands.

Do **not** shard ρ(r) itself.  It is `N_r` f64 (1.4 MB at 12×12); sharding it
buys 1.4 MB and costs an all-to-all.  The existing code says this and is
right.

### 5.3 Cost model

Per k, per rank — `G = nb²·nG·nspinor` the GEMM flops:

| | current | this design |
|---|---|---|
| ranks working | `min(P,nk)` | `P` |
| FFT work/rank | `2nb` | `2nb/P` |
| GEMM/rank | `G` | `G/P` |
| comm | none (one final all-gather) | one `x`-allgather, ~9.4 MB |
| peak box/rank | `nb·nspinor·N_r·16` | `(nb/P)·nspinor·N_r·16` |
| output/rank | `nk·nb²·16` | `nk·nb²·16/P` |

b600/P=64 (nk=16, nb=600, nspinor=4, N_r=46 080, nG=1964), per rank:

| | current | design |
|---|---|---|
| FFT box | 1.77 GB | **28 MB** |
| m-side sphere | (in the box) | 9.4 MB |
| output | 92 MB | **1.4 MB** |
| compute wall | `1·(2nb + G)` | `16·(2nb+G)/64` = **0.25×** |

12×12 / P=1024 / nb=600 / N_r=972 000: FFT box `37 GB → 36 MB`, output
`829 MB → 0.8 MB`.  W1 and W2 are both removed, not merely reduced.

### 5.4 Two plans, one family

`P ≤ nk` is exactly the regime where the current k-parallel plan is already
FFT-optimal and communication-free, and this design would add `nk`
sequential all-gathers for nothing.  So this follows the established
"two plans per solve family" pattern:

* **local (k-parallel)** — the existing path.  Resolve when `P ≤ nk` *and*
  the W1 box fits the per-device budget.
* **distributed (2-D band)** — §5.1.  Resolve when `P > nk` **or** when the
  full-band box would not fit.

The resolver must **announce which plan it took and why**, like the ζ solver
family does, and the W1 estimate must go through the memory planner (this
path is currently not covered — adjacent to open task #28).

---

### 5.5 How the sharding is expressed in JAX

The `x`-all-gather of §5.1 step 3 does not need to be hand-rolled.  Carry the
n-side over the **whole** mesh for the FFT and then constrain it to the
column layout; XLA inserts the collective:

```python
psi_n_r = ifftn_sharded(box(psi_n))                  # P(('x','y'), None, ...)
phi_G   = fftn_sharded(psi_n_r * V_r)
vpsi    = to_sphere(phi_G)                           # P(('x','y'), None, None)
vpsi_y  = with_sharding_constraint(vpsi, P('y',  None, None))   # the x-allgather
psi_m_x = with_sharding_constraint(psi_m, P('x',  None, None))
V_loc   = einsum('bsg,nsg->bn', conj(psi_m_x), vpsi_y)          # -> P('x','y')
```

FFT work is spread over all `P` (no `px`-fold redundancy), the GEMM is local,
and the only collective is the one re-shard.



`LORRAX_FFT_FFI` is documented as **required**, with "nothing to opt out to",
and `common/wfn_transforms.py` routes through it.  `psp/get_DFT_mtxels.py`
does not: it calls `jnp.fft` **eagerly at five sites** (lines 193, 359, 420,
515, 517), two of them inside the V_H kernel itself.

The FFI target is a *flat-k batched 3-D FFT*, and these shapes
(`(nb, nspinor, nx, ny, nz)`, batch over `nb·nspinor`) map onto it directly.
The flat-k FFI measured **3.78× on Σ**.  Routing this path through it is part
of "as fast as reasonably possible", and should be done **before** any
micro-tuning of the GEMM.

**Use the existing sharded helper, do not write a new one.**
`common.fft_helpers.make_sharded_ifftn_3d / make_sharded_fftn_3d(mesh,
in_spec, out_spec, norm=, axes=)` already wrap the FFI in a `shard_map` that
runs the transform locally per device with the three FFT axes replicated and
only batch dims sharded — exactly this kernel's shape.
`common.wfn_transforms._local_box_fft` shows the intended call pattern and the
jit-cache keying (`_sharding_key`, `_output_sharding`), and `to_rbox` already
implements "sphere → box → IFFT" for a sharded ψ, which is precisely the
n-side of §5.1.  The missing piece is the return leg (r → box FFT → sphere
gather); add it next to `to_rbox` rather than inline in `psp`.

---

## 7. Restart interaction — currently broken for bispinor

Two separate mechanisms, and they disagree:

* **restart state** (`file_io/tagged_arrays.py`) — `V_qmunu`, `S_qmunu`,
  `psi_full_y`, `enk_full`, `V0_noG0_munu`, `G0_mu_nu`, and (since
  2026-07-27) `psi_full_y_transverse`, with a stamped
  `n_rmu_transverse_logical` cross-checked on read.  **Bispinor-aware.**
* **ζ reuse** (`gw/gw_init.py`) — skips the ζ fit when `tmp/zeta_q.h5` is
  complete and provenance matches.  Guarded by
  `_reuse = (not cfg.bispinor) and _zeta_reuse_ok(...)`, i.e. **a bispinor
  run always refits**, because `transverse_wfn_data` (`psi_rmu_Y`,
  `psi_rmuT_X`, `meta`, `centroid_indices`) is in-memory only.

Measured cost of that refit at b600 bispinor (job 7885966): charge fit
181.85 s + transverse 135.91 s = **318 s of the 660 s GW wall, every run**.

The blocker is weaker than it looks.  `psi_rmu_Y` / `psi_rmuT_X` are ψ sampled
at centroids — recoverable from WFN + the centroid tables **without refitting
ζ**, and the same job measures that sampling at
`load_centroid_wfns` 7.04 s + `load_centroid_wfns_current` 2.15 s ≈ **9 s**.
So bispinor ζ reuse should be worth ~310 s per restarted run.

Required before it can be turned on:
1. extend `_zeta_fit_provenance` to cover the transverse channel (transverse
   centroid table hash, `n_rmu_T`, `transverse_zeta_solve`,
   `transverse_zeta_rcond`);
2. recompute `transverse_wfn_data` on the reuse path instead of returning
   `None` (today the reuse branch returns `None` for it, which would silently
   drop Σ^B — the exact failure 3d89885 fixed for the restart round-trip);
3. a round-trip gate: fit → reuse → outputs EXACT-0.

Note for the benchmark below: **the V_H path needs none of this.**  It needs
ψ, ρ and the Poisson solve — no ζ, no centroids, no V_q.  The restart work is
a separate deliverable the owner asked for, not a dependency.

---

## 8. Benchmark

`bench_rho_vh.py`: minimal WFN setup (or restart read), then **immediately**
the ρ + V_H work, nothing else.

Legs, each in its own process (peak RSS is monotone; one variant per process):

| leg | what |
|---|---|
| `local` | existing k-parallel plan |
| `dist` | §5.1 2-D band plan |
| `dist_ffi` | §5.1 with FFTs through the FFI |

Report per rank: `VmHWM` from `/proc/self/status` at exit; wall split into
`rho`, `poisson`, `mtxel_fft`, `mtxel_allgather`, `mtxel_gemm`; and the
achieved fraction of peak GEMM flops.

Correctness gate: `dist` vs `local` on `⟨mk|V_H|nk⟩` to **1e-12 relative**
(owner tolerance; the contraction is reassociated by the sharding, so
bit-identity is not required and should not be demanded).

Antipattern checks the benchmark must make explicit, not assume:
* one XLA compile for the whole k loop (fixed operand shapes — the D10
  `padded_gvectors` decision already gives this; the band blocks must not
  reintroduce ragged shapes);
* no host readback inside the k loop;
* no `.block_until_ready()` per k except at the deliberate pipeline boundary;
* the all-gather is along `'x'` only, not a global collective.

---

## 9. Implementation order

1. Route the five `jnp.fft` sites through the FFT FFI; re-measure.  (Standalone
   win, no API change.)
2. Land the `(nb_x, nb_y)`-blocked kernel behind the plan resolver, `local`
   default, `distributed` opt-in; gate at 1e-12 against `local`.
3. Keep the matrix live and sharded through its consumer. The former
   `v_hartree (nk,nb,nb)` artifact and its readers were removed; `gw_jax`
   now contracts the G-space field directly and rotates that live matrix.
4. Flip the resolver default once the b600 and 12×12 numbers are in.
5. Apply the same treatment to the two sibling sweeps that share the pattern:
   kin+ion (`kin_ion_io.py:741`) and dipole (`get_dipole_mtxels.py:919`,
   `(nk,3,nb,nb)` — 3× the W2 exposure).

Separately, and independently of the above: bispinor ζ reuse (§7), and the
`psi_rotation` wiring that turns fixed-density QSGW into charge
self-consistency (§4). There is no cross-run Hartree cache to invalidate.
