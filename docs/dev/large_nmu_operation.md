# Large-N_mu operation — running fully distributed

The page an agent reads to run LORRAX GW in the regime it exists for:
thousands of low-memory processes, where no `(n_mu·nspinor)²` tile fits on
one rank.  Every solve family carries exactly TWO plans:

* a **LOCAL** plan — whole per-q tiles, mesh-invariant bits, q-parallel
  scheduled over devices at P>1.  Default wherever it fits, because its
  output is bit-identical across process grids and device counts.
* a **DISTRIBUTED** plan — 2-D block-cyclic factorization over the whole
  mesh (ScaLAPACK on a host mesh, cuSOLVERMp on CUDA, via the `ffi.linalg`
  facade).  The only plan whose factorization work AND memory divide by P.
  Explicit opt-in, because a block-cyclic factorization is a different
  (equally valid) numerical gauge — agreement with the local plan is
  ~kappa·eps, not bit-exact.

`auto` never silently crosses that line: an explicit request that cannot
be honoured refuses at resolve time; an `auto` demotion is announced.
Conventions: mesh `(Px, Py)`, `P = Px·Py` ranks, `mu` = padded centroid
count, `r` = r-chunk size, `nq` = IBZ q count, all buffers complex128
(16 B).

## Per-stage plans, keys, and per-rank memory

| stage | key (deck unless noted) | LOCAL plan — per-rank scaling | DISTRIBUTED plan — per-rank scaling |
|---|---|---|---|
| zeta CCT build (`isdf/core.c_q_from_psi_sm`) | none (always sharded) | one 2-D-sharded shard_map; `C_q` at `P(None,'x','y')`: `nq·mu²/P` | same path (no second plan needed) |
| zeta charge factor (`isdf/core.factor_c_q`) | `charge_zeta_solve = rank_truncate` (default) + `distributed_cholesky = auto` → replicated whole-tile eigh pseudo-inverse, q-parallel at P>1 (`LORRAX_ZETA_QPARALLEL`): transient ≤ one q-batch replicated (4 GiB cap), compute `ceil(nq/P)·mu³` | `distributed_zeta_solve = distributed` → ScaLAPACK `pzheevd`, truncation on the replicated spectrum, C⁺ kept 2-D-sharded: `nq·mu²/P` stored, no O(mu²) replica anywhere; compute `nq·mu³/P`-class |
| zeta charge back-solve (per r-chunk, `solve_zeta`) | `distributed_zeta_solve = auto`: `replicated` gathers the whole factor, `nq·mu²·16` B/rank/r-chunk; `per_q` gathers one `(mu,mu)` tile at a time, `mu²·(1+1/Py)·16` B live (same total traffic) | `distributed` (same key): one stacked GEMM `C⁺@Z`, both operands 2-D-sharded; received bytes `nq·(mu²/Px + mu·r/Py)·16` per r-chunk, no whole tile ever |
| zeta transverse factor (bispinor mu_L=1,2,3; HOISTED 2026-08-01, once per channel) | `distributed_lu = auto` → per-q pivoted LU of the ridged LOGICAL tile via `lax.linalg.lu`, q-parallel at P>1 under the charge fold's policy; `(LU, piv)` stored, back-solve = `lu_solve` per r-chunk through the same replicated/per_q gather tiers — BIT-IDENTICAL to the old fused per-r-chunk `jnp.linalg.solve` | `distributed_lu = scalapack` (host) → `pXgetrf` ONCE at the logical extent, factors kept 2-D block-cyclic + per-rank ipiv threaded; `pXgetrs` per r-chunk (split FFI handlers).  `cusolvermp` (CUDA) still runs the FUSED per-r-chunk pair (hoist port pending) |
| zeta Z_q build (`z_q_from_psi_sm`) | none (always sharded) | streaming band-chunk scan inside one shard_map; carries `(nk, ns, r/Py, mu/Px, ns)` → `/P`; per-iter FFT box `nk·(band_chunk/P)·ns·n_rtot` | same path |
| zeta h5 write (G-flat accumulator + SlabIO) | `slab_io = auto` | accumulator `(nq_disk, mu/P, ngkmax)` → `/P`; `auto` → parallel-HDF5 FFI collective hyperslab write (no gather). The `h5py_allgather` fallback gathers the FULL `(nq_disk, mu, ngkmax)` tensor on rank 0 — announced, non-scaling; do not run large-mu with a demoted writer | same |
| W Dyson solve (`gw/w_isdf`) | `w_dyson_solver = auto` = `local`: q-parallel per-q dense LU, `ceil(nq/P)` whole `(mu,mu)` tiles per rank — a mu² tile per rank exists | `w_dyson_solver = distributed`: 2-D block-cyclic backsolve via `ffi.linalg` `solve_lu`, `nq·mu²/P`; refuses loudly, never downgrades |
| band-interpolation / BSE eigh (htransform fH_q, vq_interp C_q) | `eigh_backend = auto` = q-batched native eigh, one whole `(rank,rank)` matrix per device | `eigh_backend = distributed` (or `use_low_mem_eigh = true`): one tile spread over the mesh (`pzheevd` on host); square or 1-D mesh, `n` divisible by both axes |
| FFT / GEMM backends | env, not deck: `LORRAX_FFT_FFI` / `LORRAX_FFT_FFI_FUSED` / `LORRAX_BANDS_GEMM_FFI`, all default ON (REQUIRED since 2026-08-01 — a missing handler refuses at startup; explicit exports are redundant) | orthogonal to the plan choice (see `docs/dev/flat_k_fft_service.md`, `vendor_gemm_service.md`, `docs/dev/env_vars.md`) | same |
| transport | `config/frontera/mpi_transport_env.sh`: `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | required at distributed tiers (gloo banned there) | same |

Constraints common to every distributed backend (checked at RESOLVE time
by `ffi.linalg.resolve`, before any collective): host platform for
ScaLAPACK / CUDA for cuSOLVERMp, compiled handler present in
`LORRAX_FFI_HOST_SO`, one process per device covering the mesh, SQUARE or
1-D mesh (block-cyclic descriptors need `MB == NB`), and `mu_pad`
divisible by both mesh axes.  A rectangular mesh (e.g. 2×4) refuses
cleanly — plan node counts so the mesh is square.

## The fully-distributed deck, in one block

```
charge_zeta_solve      = rank_truncate    # default; the tier requires it
distributed_zeta_solve = distributed      # zeta factor+solve: nothing O(mu²) replicated
distributed_lu         = scalapack        # transverse channels, bispinor runs (CPU mesh)
w_dyson_solver         = distributed      # W Dyson backsolve
eigh_backend           = distributed      # only when one (rank,rank) tile no longer fits
slab_io                = auto             # verify the FFI writer engages (banner), not the allgather fallback
```

plus the launch env of `config/frontera/templates/gw_dev.sbatch`
(`srun --mpi=pmi2`, `impl=mpi`; the FFI stack is the required default since 2026-08-01 — no gate exports needed).

## Certified example invocations

* **Local tier, q-parallel factor** — MoS2 4×4 / 300 bands / 2979
  centroids / nq_ibz=10 / 8 nodes × 2 ranks (4×4 mesh), job 7885024:
  `zeta_fit.cholesky` 104.4 s (all-ranks control) → 11.8 s (fold, 8.9×),
  GW wall 335.9 → 214.6 s, eqp/sigma parity exact-0 vs control and vs the
  pre-fold baseline (job 7884656).  Bit-identity gate
  `tests/test_zeta_mesh_invariance.py::test_qparallel_execution_is_bit_identical_to_replicated`.
* **Distributed tier, same deck** — job 7885077 (same geometry,
  `distributed_zeta_solve = distributed`): rc=0, GW wall 222.1 s;
  `zeta_fit.cholesky` 24.25 s (pzheevd + explicit 2-D-sharded C⁺
  formation — 2.1× the q-parallel local factor at this size, EXPECTED:
  the tier buys memory shape and P-scaling, not wall time at 16 ranks);
  per-r-chunk back-solve 12.50 s vs 16.0 s local (one GEMM instead of
  two, no factor gather).  Parity vs the local tier AND vs the pre-fold
  baseline: max|Δ| = 8.6e-6 eV (eqp0/eqp1), 9.0e-6 eV
  (eqp_g0w0/sigma_diag), 4.9e-5 eV (sigma_mnk.h5) — the documented
  ~kappa·eps gauge difference (kappa ≤ 1/zeta_rcond = 1e8), 20×+ under
  physical significance (1e-3 eV); truncation active at real
  conditioning (n_keep ≈ 1995/2992 per q).  Per-rank MaxRSS (sacct,
  sampled) 6.0 GiB vs 4.6 GiB for the all-ranks control leg — same
  few-GiB class, no O(nq·mu²) gather term; largest single collective
  payload 485 MB (the `mu·r_chunk/Py` GEMM gather at r_chunk = 40544,
  announced as the documented over-budget floor; the C⁺-formation site
  chunked at 107 MB × 4).  Wall-time crossover: at mu = 10015 the
  all-ranks factor measured 4712 s on 64 ranks vs 236 s distributed
  (size-campaign figures quoted in the resolver refusal text,
  `isdf/core.py`).
* **Fixture gates** — 2×2/4×4 distributed-vs-replicated eqp max|Δ| =
  0.00e+00 at print precision; 2×4 refused at resolve time
  (`tests/test_zeta_mesh_invariance.py`, `tests/test_ffi_linalg_contract.py`).

## Auto-thresholds and their calibration

| threshold | value | what it decides | calibration |
|---|---|---|---|
| `_QPARALLEL_MIN_NQ_MU3` | 5e9 (module constant; `LORRAX_ZETA_QPARALLEL` overrides) | replicated charge factor executes q-parallel above it | 105.1 s redundant factor at nq·mu³ = 2.6e11 (job 7884656) → ~4e-10 s/unit on a 28-thread CLX rank; 5e9 ≈ 2 s, below which the two staged reshards + one compile outweigh the saving.  Mini-deck (2.6e8) stays below by design |
| `LORRAX_ZETA_GATHER_CAP_GIB` | 4 | `auto` back-solve tier: `replicated` under the cap, `per_q` above | live-bytes budget for the gathered factor; 12×12/mu=2016 stack (9.4 GB) lands on per_q |
| `LORRAX_ZETA_REPLICATE_CAP_GIB` | 4 | whether the charge factorization may run replicated at all (per-q-batch criterion for rank_truncate) | mu ceiling `sqrt(cap/16)` = 16384/batch; production 12×12 runs raise to 16 |
| `LORRAX_COLLECTIVE_CHUNK_MB` | 128 | max payload of ONE emitted collective in the distributed tier (host-level q-block loop, cannot be re-fused by XLA) | 1.15 GB single-shot AllGather fatal at P=144; 0.104 GB healthy on the same 144 ranks; at P=16 impl=mpi the cap is indistinguishable from unbounded.  A per-instruction transport cap, orthogonal to the 4 GiB live-bytes cap.  Note: once ONE q's collective exceeds the budget the bound is abandoned with a loud warning (q is the only split axis) |
| `slab_io = auto` probe | launcher PMI env, else a subprocess MPI_Init probe | FFI parallel-HDF5 vs announced allgather demotion | bare-launch demotion path is a standing regression test |
| `LORRAX_BANDS_GEMM_FFI` (default on — REQUIRED) | startup enforcement (`Gate.enforce`) | vendor batched GEMM; `=0` = announced uncertified XLA-dot debug opt-out | a missing handler refuses at startup naming the .so (decisions.md 2026-08-01) |

## Still replicated today (honest list)

These are the objects that do NOT yet divide by P; they bound the regime
until fixed.  File:line references as of this page's commit.

1. ~~htransform SVD family~~ — CLOSED 2026-08-01: the replicated
   `A = psi@centroids` gather + per-rank dense SVD is now a Gram-eigh of
   `A Aᴴ` (`nk·nb` square, N_mu-free) through the `ffi.linalg` eigh plan
   (deck key `eigh_backend`, same family as the fH_q eigh; `auto` = native
   replicated — the tile is N_mu-free and small — `distributed` = pzheevd),
   with `Vᴴ` and `B_at_mu` mu-sharded on `'y'` (`B_at_mu` fitted at the
   true centroid count).  Gauge-class change (eigenvector phases), NOT
   bit-exact: b300 `bandstructure.dat` parity measured exact-0 at file
   precision, fastloop both legs exact-0 (jobs 7885090, 7885093).  The
   remaining htransform mu-bound is item 2's single-axis psi sharding.
2. **psi-at-centroids single-axis sharding** — `(nk, nb, ns, N_mu)`
   sharded on ONE mesh axis (`common/wfn_transforms.py:1933,2002`), so
   per-rank memory is `~1/sqrt(P)`, in the zeta fit (both transposes),
   htransform, and the BSE psi stacks (`bse/bse_ring_comm.py:156`,
   `bse/exciton_bands.py:241`).
3. **zeta replicated/per_q back-solve gather** — `nq·mu²·16` B/rank per
   r-chunk unless `distributed_zeta_solve = distributed`.
4. **transverse LU factor tiles** — the LOCAL plan's hoisted `(LU, piv)`
   stack is gathered per q-tile by the replicated/per_q tiers exactly
   like the CCT it replaced (an `mu_T²` tile per gather); the per-r-chunk
   RE-FACTORING is gone on both plans since 2026-08-01 (see below), and
   `distributed_lu = scalapack` keeps the factors block-cyclic
   throughout.
5. **W Dyson local plan** — `ceil(nq/P)` whole `(mu,mu)` tiles per rank
   unless `w_dyson_solver = distributed`.
6. **dipole/kin-ion k-gathers** — FIXED: the CLI sweeps now gather
   `owner_only` (`common/collectives.py::gather_indexed_blocks_to_owner`):
   the `(nk, 3, nb, nb)` / `(nk, nb, nb)` table exists on rank 0 alone,
   assembled in `LORRAX_COLLECTIVE_CHUNK_MB`-bounded chunks.  The
   replicated mode remains the default for operand consumers
   (`sigma_dispatch`'s gspace V_H route).
7. **eigenvalue vectors** — `lambda (nq, mu)` replicated in the
   distributed zeta tier (ScaLAPACK's own contract); `Sigma_c(omega,k,m,n)`
   cube replicated under the default `sigma_omega_layout = replicated`.
8. **h5py_allgather writer fallback** — rank-0 full-tensor gather when
   the parallel-HDF5 probe demotes; announced.

## Transverse (bispinor) factor stage — CLOSED 2026-08-01

The transverse CCT is Hermitian INDEFINITE: no Cholesky, no eigh-based
rank truncation.  The factor is a per-q pivoted LU with a ridge
(`eps·|tr(C_log)|/n_log`, eps = 1e-12), historically fused into
`solve_zeta`'s per-r-chunk path.  Since 2026-08-01 `factor_c_q` HOISTS
it — one factorization per q per CHANNEL, `(factor, piv)` returned, and
`solve_zeta` only applies it per r-chunk — with the charge family's two
plans:

* **LOCAL** (`distributed_lu = auto`/`off`): `lax.linalg.lu` on the
  whole ridged LOGICAL tile, q-parallel over devices under the charge
  fold's policy (`LORRAX_ZETA_QPARALLEL` / nq·mu³ threshold), LU
  identity-re-embedded at the padded extent so the replicated/per_q
  gather tiers consume it unchanged; back-solve is
  `jax.scipy.linalg.lu_solve` — the identical arithmetic
  `jnp.linalg.solve` runs internally, so the hoist is a SCHEDULE, not a
  numerical route.  Gate `tests/test_transverse_factor_hoist.py`: exact
  bit equality vs the fused path across meshes, both gather tiers, both
  factor schedules, two r-chunks per factor, non-dividing nq + padded mu.
* **DISTRIBUTED** (`distributed_lu = scalapack`, host): `pXgetrf` ONCE
  at the logical extent, factors kept 2-D block-cyclic, per-rank ipiv
  threaded verbatim into `pXgetrs` per r-chunk (split FFI handlers
  `lorrax_scalapack_batched_getrf`/`_getrs`; an old host .so without
  them refuses at resolve time).  `pXgetrf` on the same tile is
  bit-identical whether or not the `getrs` follows immediately — the
  split-vs-fused contract cell `scalapack_getrf_getrs` pins exact
  equality (incl. factor reuse across two RHS) at 1x1 and on a 2x2
  process mesh.
* `cusolvermp` (CUDA) still runs the FUSED per-r-chunk getrf+getrs —
  the hoist port is pending; the CCT-passthrough + `lu_piv=None` fused
  branches in `solve_zeta` are preserved exactly for it.

The distributed route still requires the LOGICAL mu_T extent to divide
both mesh axes (pad-extent LU roundoff is amplified O(1) in near-null
transverse modes): pick transverse centroid counts divisible by the mesh
axes (the 4×4 deck now has `centroids_T_div136.txt`, 136 = %8==0) or
auto demotes (announced) to the local plan.

**Certified example (job 7885126, MoS2 4×4 bispinor deck, TLU host
.so):** bi4 (402c+143T, P=4) and bi16 (785c+275T, P=16) hoisted-vs-fused
EXACT-0 on eqp0/eqp1/eqp_g0w0/sigma_diag AND sigma_mnk.h5; scalapack
split-vs-fused (402c+136T, P=4) EXACT-0 likewise; distributed-vs-local
gauge delta 0.0 at .dat print precision, 1.1e-13 eV sigma_mnk.h5.
Measured effect at bi16: per-channel `zeta_fit.chunk.solve` 4.3–12.8 s →
1.7–2.8 s (per-r-chunk re-factorization gone), GW wall 214.3 → 201.2 s;
grows with n_rchunks·P at production mu_T.

## Pointers

`docs/dev/linalg_ffi.md` (facade + backend guards) ·
`docs/dev/env_vars.md` (every env knob, with calibration) ·
`docs/dev/mpi_collectives.md` (transport) · `docs/drivers.md` §gw (deck
keys) · `src/isdf/core.py` module comments (tier layout contracts,
collective payload chunking) · `src/gw/w_isdf.py` (W plan family).
