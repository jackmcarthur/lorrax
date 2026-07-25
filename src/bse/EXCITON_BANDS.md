# Arbitrary-Q exciton bandstructure — traps, gates, sizing

Status: working, 2026-07-22. Produces `E_S(Q)` on a continuous Q path from a
coarse-grid GW restart, off-grid points via the htransform Galerkin
interpolation. Reference figure and numbers at the bottom.

> **Before you trust any off-grid result, read "The two traps" below.** Both
> failures in this pipeline to date were silent: the run completes, the
> on-grid gates pass, and the bands are wrong by eV. On-grid agreement is
> *not* evidence that off-grid evaluation works.

## Pipeline

| Stage | Where | Notes |
|---|---|---|
| coarse ζ / V_qμν restart | `gw.gw_jax` | full-BZ ζ needed for `--vq-mode interp`; IBZ-only ζ works for `ongrid` |
| `V_Q` at arbitrary Q | `vq_interp.py` | F-scheme + b26p stencil; `build_vq_evaluator` → `eval_vq` |
| ψ_c(k+Q), ε_c(k+Q) | `bandstructure/htransform.py` + `bse_setup.py` | fH_R Fourier sum, Galerkin basis, per-q `eigh` |
| finite-Q TDA solve | `exciton_bands.py` | `H_Q = D_Q + V_Q − W`, one `lax.scan` over the Q path |

`--vq-mode ongrid` bypasses `vq_interp` entirely and uses the stored production
tile `V_qmunu[wrap(−Q)]`. Exact, but only for Q on the BSE grid — and it
therefore exercises none of the interpolation. Do not use it to validate
`interp`.

## The two traps

### 1. Splitting a contracted axis and summing the outer product

`streaming_galerkin_solve` forms

    Q[α, x] = Σ_{k,n} inv_s[α] · Uᴴ[α,(k,n)] · ψ[(k,n), x]
    G       = Q Qᴴ

The contraction runs over the pair index `(k,n)`; `x = (spinor, r)` is free.
So chunking `x` (the r axis) and summing `G` over chunks is exact — disjoint
column blocks. Chunking the **band** axis and summing `G` per chunk is **not**:
it drops every `bc ≠ bc'` cross term of `(Σ_bc Q_bc)(Σ_bc' Q_bc')ᴴ`. Band
chunks must be summed *into* `Q` before the outer product.

Getting it backwards is a silent wrong answer, not a crash. `G` stays
Hermitian positive-definite, the Cholesky succeeds, and the only symptom is
`ctilde` losing orthonormality — so `fH = Σ_n f(ε_n) c_n c_nᴴ` no longer has
eigenvalues `f(ε_n)` and recovered energies drift by ~1.7 eV.

Measured on the 2026-07-20 known-good case's own files (MoS2 12×12, 30 Ry,
n_μ=640, nb=40), on-grid max |Δε_c| vs the stored grid:

| source | on-grid \|Δε_c\| | ctilde ortho err |
|---|---:|---:|
| one band chunk (nb ≤ `band_chunk_size`) | 0.63 meV | — |
| two chunks, `G` summed per chunk | **1742.48 meV** | 4.9e-01 |
| two chunks, fixed (`Q` summed first) | 0.78 meV | 4.3e-04 |

Latent for months because `band_chunk_size` defaulted to 64 ≥ nb. It went live
when the chunk started being sized to the ψ box. **Any htransform
bandstructure produced with `nb > band_chunk_size` between those two changes
is suspect** — this includes the standalone single-particle driver, not just
the BSE path.

Detection: watch `ctilde[0] orthogonality error`. It moves five orders of
magnitude across this bug while `min-sval` barely moves at all.

### 2. On-grid gates cannot see off-grid breakage

At a mesh Q the "interpolated" `V_Q` reduces to the stored production tile, so
an interp-vs-ongrid comparison at mesh Q has **zero off-grid content**. During
the 2026-07-21 debugging this agreed to 0.026 meV while the off-grid path was
broken by eV.

Same failure on the ε side: an on-grid `|Δε_c|` gate returned **7.09 meV
bit-identical** across two different fH windows. A number that does not move
when you change the thing under test is not measuring it.

Corollary: every gate in the driver defaulted to on-grid (Γ-only gate,
`--vq-mode ongrid` pinning, on-grid window sweeps), which is why this survived
so long.

## How to gate off-grid correctly

**Do not gate on an absolute second difference of `E_1(Q)`.** `d²` does not
vanish for a correct calculation — real curvature plus kinks where sorted
branches cross. The reference calculation whose figure *is* smooth measures
43.5–61.4 meV max itself. A few-meV target in that metric is unreachable and
inviting fabrication.

Two things that do work:

1. **Against a reference calculation**, same metric, same path. Converged
   80 Ry run vs the 30 Ry known-good reference, off-grid |2nd diff| of `ε_c`
   max/mean (meV):

   | | Γ–M | Γ–K | on-grid |
   |---|---|---|---|
   | before the Gram fix | 4784 / 720 | 4785 / 671 | 311.24 |
   | after | **186 / 23.8** | **170 / 28.8** | **7.44** |
   | 30 Ry known-good reference | 187 / 22.5 | 185 / 27.6 | 0.78 |

2. **Reference-free symmetry test — the real bound.** Q and its point-group
   images must give identical `E_S`. `exciton_bands --extra-q` appends
   arbitrary Q to the *same* scan (one compile, no extra cost beyond the scan
   row), so pass the σ_v mirror images of a few off-grid path points. On the
   converged run: **ΔE_1 = −0.538 and +0.069 meV**, max over all 8 branches
   2.391 and 1.620 meV. That is the honest few-meV bound on interpolation
   error, and it needs no external reference.

## Sizing

**Interpolation capacity is `nk·nb < rank(ψ_μ)` — driven by nk, unrelated to
the per-band centroid guidance used for Σ/screening.** `nspinor·n_μ` is only
an upper bound on the ISDF column space: 2412 D3h-closed centroids on a
174,960-point grid give nominal 4824 but true rank **4570** (95%), so at 12×12
the ceiling is `nb < 31.74`, not 33.5. nb=28 passes, nb=32 fails. Measure the
rank; do not trust the nominal.

**The fH window must be contiguous from E_min upward, semicore included** —
htransform requires the Hamiltonian truncated into a subspace anchored at
E_min. Verify `nval` reaches E_min against the actual eigenvalues rather than
copying a band count from another run. Keep ≥2 guard bands above the BSE
conduction selection: a window with zero guards fails for an unrelated reason
(the top eigenvector lands in fH's null space where `f(ε) → 0`).

**640 centroids is enough for the BSE/interpolation basis.** The converged
quantity that matters is the GW *reference* (eqp energies), not the centroid
count. Reaching for n_μ=2412 here buys nothing and costs a memory fight.

## Memory

The eigh is **not** the constraint, despite appearances. Routing it through the
distributed FFI (`ffi.linalg.dispatch_eigh`, cuSOLVERMp/SLATE) is
numerically exact (6.19e-11 meV vs native at rank 4452) but **does not change
the high-water mark** — 15.677713664 GB/device, identical to the byte for both
backends, because the peak is reached before the eigh runs. It also costs
11–41× per q. Native batched is the default for that reason.

What actually OOMs at wide windows: `streaming_galerkin_solve`'s G
accumulation (5 × `(rank, nspinor, n_rtot/ndev)` c128) and `build_fH_R`. Those
are the next lever if wider windows are needed. `compute_wfns_fi`'s existing
`batch_size` reduces eigh memory linearly at zero wall cost and is exposed on
no CLI.

Allocator matters: `TF_GPU_ALLOCATOR=cuda_malloc_async` runs the full
E_min-anchored window where BFC@0.95 OOMs on the same hardware.

`LORRAX_SKIP_VQ_GATES=1` (read in `vq_interp.py`) drops the `run_gates` /
`run_nulls` batteries *and* the `keep_host_mirrors` replicated host tensors —
the latter is the real saving (a 58 GB alloc at 1496 centroids on 16 GPU).
Note the sandbox `run_shifter.sh` wrappers pass this via an unquoted
`${EXTRA_ENV:-}`, which word-splits into a command position and is silently
dropped; the job runs fine without your env.

## Reference run

MoS2, converged 80 Ry / 12×12 / 400-band G₀W₀ (direct gap 2.6356 eV at K),
8v8c TDA, 39 path Q + 2 symmetry-check Q, `--vq-mode interp`, 16 × A100-80GB,
one JAX process per GPU, `--px 4 --py 4`. `E_1(Γ) = 2.094357 eV`, binding
541 meV.

| stage | s | % |
|---|---:|---:|
| `load_bse` | 11.2 | 0.4 |
| `htransform_setup` | 21.0 | 0.8 |
| `htransform_psi_cQ` | 1386.3 | 55.7 |
| `vq_prepare` | 182.8 | 7.4 |
| `vq_eval` | 1.6 | 0.1 |
| `solve_scan_cold` (incl. ONE compile) | 879.0 | 35.3 |
| **total** | **2487.6** | |

Cost scales with Q count almost entirely through `htransform_psi_cQ`; the
solve is one compile plus ~21 s/Q. A denser path costs ~30 s/point, not 60.
`vq_prepare` is one-time, independent of Q count.

Keep the single-compile property: the whole path must go through one
`lax.scan`. A Python loop over Q recompiles per point and dominates everything
above.
