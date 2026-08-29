# Large-N_mu operation — running fully distributed

The page an agent reads to run LORRAX GW in the regime it exists for:
thousands of low-memory processes, where no `(n_mu·nspinor)²` tile fits on
one rank.  Dense solves have two storage plans:

* a **LOCAL** plan — whole per-q tiles, q-parallel scheduled over devices at
  P>1.  The ordinary replicated/per_q schedule is the mesh-invariant control.
* a **DISTRIBUTED** plan — 2-D block-cyclic factorization over the whole
  mesh (ScaLAPACK on a host mesh, cuSOLVERMp on CUDA, via the `distrib_la`
  facade).  The only plan whose factorization work AND memory divide by P.
  Explicit opt-in, because a block-cyclic factorization is a different
  (equally valid) numerical gauge — agreement with the local plan is
  ~kappa·eps, not bit-exact.

Transverse ridge solving also has a **local batch-reshard execution route**.
It starts and ends with the same 2-D face layout, but assigns complete q
matrices across ranks between two pairs of `all_to_all` operations.  It is a
local numerical plan, not a third factor format and not a provider call.

An explicit route request is never changed.  `auto` may select the local or
distributed plan after capability and capacity checks, and announces the
choice.
Conventions: mesh `(Px, Py)`, `P = Px·Py` ranks, `mu` = padded centroid
count, `r` = r-chunk size, and all buffers complex128 (16 B).  `nq` is the
stage's q extent; the coupled transverse admission model deliberately uses
full `Q=N_q^full`.

## Per-stage plans, keys, and per-rank memory

| stage | key (deck unless noted) | LOCAL plan — per-rank scaling | DISTRIBUTED plan — per-rank scaling |
|---|---|---|---|
| zeta CCT build (`isdf/core.c_q_from_psi_sm`) | none (always sharded) | one 2-D-sharded shard_map; `C_q` at `P(None,'x','y')`: `nq·mu²/P` | same path (no second plan needed) |
| zeta charge factor (`isdf/core.factor_c_q`) | `charge_zeta_solve = rank_truncate` (default) + `distributed_cholesky = auto` → replicated whole-tile eigh pseudo-inverse, q-parallel at P>1 (`LORRAX_ZETA_QPARALLEL`): transient ≤ one q-batch replicated (4 GiB cap), compute `ceil(nq/P)·mu³` | `distributed_zeta_solve = distributed` → distributed eigensolver (ScaLAPACK on CPU, cuSOLVERMp on CUDA), truncation on the replicated spectrum, C⁺ kept 2-D-sharded: `nq·mu²/P` stored, no O(mu²) replica anywhere; compute `nq·mu³/P`-class |
| zeta charge back-solve (per r-chunk, `solve_zeta`) | `distributed_zeta_solve = auto`: `replicated` gathers the whole factor, `nq·mu²·16` B/rank/r-chunk; `per_q` gathers one `(mu,mu)` tile at a time, `mu²·(1+1/Py)·16` B live (same total traffic) | `distributed` (same key): one stacked GEMM `C⁺@Z`, both operands 2-D-sharded; received bytes `nq·(mu²/Px + mu·r/Py)·16` per r-chunk, no whole tile ever |
| zeta transverse factor (bispinor μ1–3) | ridge/local: `lax.linalg.lu` is hoisted once per q and channel; replicated/per_q gather tiers apply `lu_solve`. Ridge/batch-reshard: face → q-batch → local `jnp.linalg.solve` → face, so LU is intentionally repeated per r-chunk. `rank_truncate`: local per-q eigh, explicit C⁺, one GEMM per r-chunk | ridge/distributed: `distrib_la.factor('solve_lu',...)` returns a 2-D-sharded token after one batched `getrf`; `solve(token,Z_q)` calls batched `getrs` per r-chunk. ScaLAPACK and cuSOLVERMp both implement the split, reusable token. `rank_truncate`: distributed eigensolver at the padded extent, inert pad modes removed, C⁺ stays 2-D-sharded |
| zeta Z_q build (`z_q_from_psi_sm`) | none (always sharded) | streaming band-chunk scan inside one shard_map; carries `(nk, ns, r/Py, mu/Px, ns)` → `/P`; per-iter FFT box `nk·(band_chunk/P)·ns·n_rtot` | same path |
| zeta h5 write (G-flat accumulator + SlabIO) | none (one transport) | accumulator `(nq_disk, mu/P, ngkmax)` → `/P`; SlabIO issues parallel-HDF5 collective hyperslab writes with no gather. The deleted h5py allgather path would have materialized the full tensor on rank 0, violating this tier's memory contract; there is no backend selector or demotion. | same |
| W Dyson solve (`gw/w_isdf`) | `w_dyson_solver = auto` = `local`: q-parallel per-q dense LU, `ceil(nq/P)` whole `(mu,mu)` tiles per rank — a mu² tile per rank exists | `w_dyson_solver = distributed`: 2-D block-cyclic backsolve via `distrib_la` `solve_lu`, `nq·mu²/P`; refuses loudly, never downgrades |
| **W ladder resolvent** (`bse/w_ladder`, reached from `gw/screening_bse`) | `screening_diagrams = w_bse` (the stage does not exist under `w_rpa`) | ONE CODE PATH SERVES BOTH PLANS, and this cell is the local one: the ring matvec at `P = 1` reduces its `ppermute` to a no-op loop over a single rank and its `shard_map` bodies to the local einsums, so there is no second implementation to keep in step. Per-rank high-water is MEASURED and is NOT the probe block: it is the replicated `O(max_iter^2)` GMRES workspace (`N_mu`-independent), then the direct rung's `T/T_R/U_R/U_q/U` chain, then the coupling rung's transient conduction-full trial buffer, then `X_full = (2, p_chunk, c, v, k)` and one `P('x','y')` output tile. `probe_chunk` trades dispatches for the probe block's memory. The itemised, dated numbers are the `bse/w_ladder` module docstring's scaling-envelope section — this row links there rather than restating them | same module, same call. Tiles never leave the mesh: the seed enters at `P(None,'y',None)`, the readout leaves through the reduce-scatter snapshot at `P('x','y')`, and the stacked wedge is `P(None,None,'x','y')` — the same class as `W_q` itself, so **no whole-`mu^2` per-rank object exists anywhere on this path**. The q-wedge is walked one q at a time and never batched; cost per `(q, z)` is (GMRES iterations) x (one ring matvec), and the whole `q x z` sweep dispatches ONE compiled program (operator structure baked into the cached block-GMRES engine, every q/z tensor a runtime arg, and `W_R` built once OUTSIDE the q loop — it depends on `k - k'`, not on q). GW-side assembly adds only the `+v`, the mu-pad and the SAME `unfold_isdf_operator` call the RPA W already makes |
| band-interpolation / BSE eigh (htransform fH_q, vq_interp C_q) | `eigh_backend = auto` = q-batched native eigh, one whole `(rank,rank)` matrix per device | `eigh_backend = distributed` (or `use_low_mem_eigh = true`): one tile spread over the mesh (`pzheevd` on host); square or 1-D mesh, `n` divisible by both axes |
| **BSE restart-bundle load** (`bse/bse_loading`) | NOT a deck key — selected by DEVICE COUNT at `bse/bse_jax.py:143` (`use_sharded = n_devices > 1 or not tda`). LOCAL = `_load_ring_subset`: whole-file `V_qmunu` + `W0_qmunu` + `psi_full_y`, `2·nq·mu²·16` B of host staging and `nq·mu²·16` B on the one device. **It REFUSES above one process** (`bse_loading::_load_ring_subset`, the `process_count() > 1` guard at its head), so it cannot silently become the P>1 path | `load_bse_data_from_restart_sharded`: per-rank `(mu,nu)` h5py hyperslabs, **no allgather anywhere**. `W_q` at `P('x','y',None,None,None)` = `nk·mu²/P`; `V_q0` at `P('x','y')` = `mu²/P`; `V_q_full` (opt-in `load_v_full`) a SECOND `nk·mu²/P`. Largest host staging buffer is exactly one rank's tile. ψ/`M` are single-axis (`1/px`, `1/py`) — see the honest list item 2 |
| **BSE matvec** (`bse_stack_matvec`; `bse_ring_comm` / `bse_simple` legacy) | No selector on the sharded path. `--matvec-kind` is a CLI flag, not a deck key, and it is **INERT on the sharded eigensolve** — every sharded solve builds `bse_stack_matvec` regardless (it still steers `bse.absorption_haydock`). LOCAL = `bse_serial.apply_bse_hamiltonian_single_device`: the `T` tensor `(b, mu, mu, ns, ns, nk)` whole on one device — the 1-device reference path only | `shard_map` whose body `lax.scan`s the trial axis, so exactly ONE `T` is alive regardless of stack width; `T`/`U` at `P(None,'x','y',None,None,None)` = `nk·ns²·mu²/P` per trial slot. Every collective is over pair-space `(c,v,k)` or one mu axis — **no collective carries a `mu²` payload**. (The flag's accepted values are `ring`/`gather`/`simple`; there has never been a `stack` value to pass, which is the other half of why this row was misleading.) |
| **BSE coarse→fine W densify** (`bse_io.make_w_densifier`) | none (always sharded) | one `jax.jit` with `out_shardings` pinned to `w_spec`, shard_map-interior FFTs both ways: per-rank peak is the local `(mu/px, nu/py, nk_fine)` tile. The eager `local_ifftn3` + `device_put` twin it replaced all-gathered the full tensor per rank; `tests/test_fft_shardmap_context.py` bans it | same path |
| **BSE arbitrary-Q exchange model** (`bse/vq_interp`) | none — **NO DISTRIBUTED PLAN EXISTS**; this is the BSE stage's live scaling defect. See the section below | — |
| FFT / GEMM backends | env, not deck: `LORRAX_FFT_FFI` / `LORRAX_FFT_FFI_FUSED` / `LORRAX_BANDS_GEMM_FFI`, all default ON (REQUIRED since 2026-08-01 — a missing handler refuses at startup; explicit exports are redundant) | orthogonal to the plan choice (see `docs/dev/flat_k_fft_service.md`, `vendor_gemm_service.md`, `docs/dev/env_vars.md`) | same |
| transport | `config/frontera/mpi_transport_env.sh`: `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` | required at distributed tiers (gloo banned there) | same |

Distributed backends check their contracts before any collective: CPU for
ScaLAPACK, CUDA for cuSOLVERMp, compiled handler present, one process per
device, and the matrix extent divisible by both mesh axes.  ScaLAPACK's
one-tile-per-rank descriptor supports square or 1-D meshes.  cuSOLVERMp
requires a true 2-D mesh; an explicit request on a 1-D mesh refuses.  Use a
square mesh for a portable deck.

## The fully-distributed deck, in one block

```
charge_zeta_solve      = rank_truncate    # default; the tier requires it
distributed_zeta_solve = distributed      # zeta factor+solve: nothing O(mu²) replicated
distributed_lu         = cusolvermp       # transverse ridge, CUDA true-2D mesh
# distributed_lu       = scalapack        # use this instead on a CPU mesh
w_dyson_solver         = distributed      # W Dyson backsolve
eigh_backend           = distributed      # only when one (rank,rank) tile no longer fits
```

plus the launch env of `config/frontera/templates/gw_dev.sbatch`
(`srun --mpi=pmi2`, `impl=mpi`; the FFI stack is the required default since 2026-08-01 — no gate exports needed).

> **`--mpi=pmi2` is a FRONTERA line. Do not carry it to Perlmutter.** It is
> right for Intel MPI under TACCs SLURM. Against Shifters Cray MPICH it
> makes MPI initialise as a **singleton** — every rank gets its own
> `MPI_COMM_WORLD` of size 1 — which under independent I/O can still produce
> a correct-looking file. Perlmutter needs `--mpi=cray_shasta`. This is the
> single most dangerous failure mode in the I/O subsystem;
> [`docs/architecture/slab_io.md`](../architecture/slab_io.md) owns it and
> carries the three-line world-size assertion every gate should run.

## Certified example invocations

* **Local tier, q-parallel factor** — MoS2 4×4 / 300 bands / 2979
  centroids / nq_ibz=10 / 8 nodes × 2 ranks (4×4 mesh), job 7885024:
  `zeta_fit.cholesky` 104.4 s (all-ranks control) → 11.8 s (fold, 8.9×),
  GW wall 335.9 → 214.6 s, eqp/sigma parity exact-0 vs control and vs the
  pre-fold baseline (job 7884656).  Bit-identity gate
  `tests/test_zeta_mesh_invariance.py::test_qparallel_execution_is_bit_identical_to_replicated`.
  **CEILING (measured 2026-08-01): the fold saturates at P = nq_ibz** —
  q is its only parallel axis, so every rank past nq idles for the whole
  factor stage.  At nq=10 / P=64 (b600 deck, job 7885316) 54 of 64 ranks
  idled for 53.7 s = 22.4 % of the GW wall; P=16→64 bought the stage
  nothing (already 1 q/rank at P=16), so its 4.57× growth was pure μ³.
  The announcement now states the ceiling whenever P > nq.
* **Distributed tier, same deck** — job 7885077 (same geometry,
  `distributed_zeta_solve = distributed`): rc=0, GW wall 222.1 s;
  `zeta_fit.cholesky` 24.25 s (pzheevd + explicit 2-D-sharded C⁺
  formation — 2.1× the q-parallel local factor at P/nq = 1.6);
  per-r-chunk back-solve 12.50 s vs 16.0 s local (one GEMM instead of
  two, no factor gather).  Parity vs the local tier AND vs the pre-fold
  baseline: max|Δ| = 8.6e-6 eV (eqp0/eqp1), 9.0e-6 eV
  (eqp_g0w0/sigma_diag), 4.9e-5 eV (sigma_mnk.h5) — the documented
  ~kappa·eps gauge difference (kappa ≤ 1/zeta_rcond = 1e8), 20×+ under
  physical significance (1e-3 eV); truncation active at real
  conditioning (n_keep ≈ 1995/2992 per q).  Largest single collective
  payload 485 MB (the `mu·r_chunk/Py` GEMM gather at r_chunk = 40544,
  announced as the documented over-budget floor; the C⁺-formation site
  chunked at 107 MB × 4).
  **Wall/memory framing, corrected 2026-08-01 (jobs 7885316 local vs
  7885323 distributed, μ=4775 / nq=10 / P=64, one deck key changed):**
  the wall crossover is governed by **P/nq, not μ alone** — at
  P/nq = 1.6 (b300/P=16) the distributed factor is 2.1× slower; at
  P/nq = 6.4 it is 1.64× FASTER (`zeta_fit.cholesky` 53.71 → 32.74 s,
  `chunk.solve` 10.20 → 6.68 s, GW wall minus startup 214.8 → 177.6 s =
  0.83×).  The μ=10015 figure quoted in the resolver refusal text
  (4712 s all-ranks vs 236 s distributed on 64 ranks) is the UNFOLDED
  all-ranks comparison, not the fold's crossover.  Memory: the tier
  removes the replicated `(nq, μ, μ)` back-solve gather, but per-rank
  peak moves ONLY when that gather is the binder — at b600/P=64 the
  binder was the `C_fit_one_rchunk` transient and peak VmHWM was
  unchanged (16.24 → 16.22 GiB) despite the announced 3.69 GB/rank
  gather being absent.  A memory claim for this tier must name the
  binder it relieves.  Parity at this shape: 1.1–1.5e-5 eV on all four
  .dat outputs at the run's own κ/q = 9.98e7 — the same κ·ε gauge
  class.
* **Fixture gates** — 2×2/4×4 distributed-vs-replicated eqp max|Δ| =
  0.00e+00 at print precision; 2×4 refused at resolve time
  (`tests/test_zeta_mesh_invariance.py`, `pytest -m distrib_la`).

## Auto-thresholds and their calibration

| threshold | value | what it decides | calibration |
|---|---|---|---|
| `_QPARALLEL_MIN_NQ_MU3` | 5e9 (module constant; `LORRAX_ZETA_QPARALLEL` overrides) | replicated charge factor executes q-parallel above it | 105.1 s redundant factor at nq·mu³ = 2.6e11 (job 7884656) → ~4e-10 s/unit on a 28-thread CLX rank; 5e9 ≈ 2 s, below which the two staged reshards + one compile outweigh the saving.  Mini-deck (2.6e8) stays below by design.  The fold SATURATES at P = nq (ranks past nq idle; announced whenever P > nq — jobs 7885316/7885323, see the certified-examples ceiling note); the transverse folds share the identical shape |
| `LORRAX_ZETA_GATHER_CAP_GIB` | 4 | `auto` back-solve tier: `replicated` under the cap, `per_q` above | live-bytes budget for the gathered factor; 12×12/mu=2016 stack (9.4 GB) lands on per_q |
| `LORRAX_ZETA_REPLICATE_CAP_GIB` | 4 | whether the charge factorization may run replicated at all (per-q-batch criterion for rank_truncate) | mu ceiling `sqrt(cap/16)` = 16384/batch; production 12×12 runs raise to 16 |
| `LORRAX_COLLECTIVE_CHUNK_MB` | 128 | max payload of ONE emitted collective in the distributed tier (host-level q-block loop, cannot be re-fused by XLA) | 1.15 GB single-shot AllGather fatal at P=144; 0.104 GB healthy on the same 144 ranks; at P=16 impl=mpi the cap is indistinguishable from unbounded.  A per-instruction transport cap, orthogonal to the 4 GiB live-bytes cap.  Note: once ONE q's collective exceeds the budget the bound is abandoned with a loud warning (q is the only split axis) |
| parallel-HDF5 availability | launcher PMI env, else a subprocess MPI_Init probe | FFI parallel-HDF5, or a refusal; there is no demotion | the bare-launch path asserts the refusal. At P>1 the FFI also compares `MPI_Comm_size(MPI_COMM_WORLD)` against `jax.process_count()` and refuses on a mismatch (`LORRAX_PHDF5_REQUIRE_MPI_WORLD`), because a PMI-flavour mismatch otherwise yields unsynchronised writers with rc=0 |
| `LORRAX_BANDS_GEMM_FFI` (default on — REQUIRED) | startup enforcement (`Gate.enforce`) | vendor batched GEMM; `=0` = announced uncertified XLA-dot debug opt-out | a missing handler refuses at startup naming the .so (decisions.md 2026-08-01) |

## Still replicated today (honest list)

These are the objects that do NOT yet divide by P; they bound the regime
until fixed.  File:line references as of this page's commit.

1. ~~htransform SVD family~~ — CLOSED 2026-08-01: the replicated
   `A = psi@centroids` gather + per-rank dense SVD is now a Gram-eigh of
   `A Aᴴ` (`nk·nb` square, N_mu-free) through the `distrib_la` eigh plan
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
   like the CCT it replaced (an `mu_T²` tile per gather).  The fully
   distributed ScaLAPACK and cuSOLVERMp plans keep reusable factors
   block-cyclic throughout.  The local batch-reshard route is the deliberate
   exception: it distributes complete q matrices across ranks and
   refactorizes them in each r-chunk to avoid provider-call overhead.
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
   `sharded` removes that residency (the layout's own `self_consistent`
   refusal was deleted 2026-08-05; since 2026-08-27 `self_consistent`
   itself is refused at driver entry beside a dynamic `compute_mode`, so
   the cube pairs with `one_shot_dft` / `fixed_point` only).

   IF A CUBE ROTATION IS EVER ADDED (e.g. to put `sigma_mnk.h5` and
   `WFN_qp.h5` in one basis), the sharded layout does NOT force the cube
   back onto every rank — but only if the output sharding is pinned.
   Measured at nomega=41/nk=16/nb=512 (cube 2624 MiB), job 7889790: the
   einsum `U Sigma U^dagger` with its output sharding LEFT FREE emits an
   all-reduce of the FULL 2624 MB cube and six full-cube buffers, identical
   at P=4 and P=16 — exactly the per-rank residency 712a866 removed, and it
   does not improve with P.  The SAME einsum with the output PINNED to the
   cube's own tiling emits two all-gathers of cube/p_axis and no full-cube
   buffer (transient 1.02 cube at 2x2, 0.51 at 4x4).  The difference is one
   `with_sharding_constraint`.  A two-half shard_map form that scans over
   the destination band block does better still: largest collective operand
   one tile, transient ~4 tiles (0.26 cube at 4x4).  See
   `tests/multi_device/sigma_omega_rotate_probe.py`.
8. **h5py_allgather writer fallback** — rank-0 full-tensor gather when
   the parallel-HDF5 probe demotes; announced.

## Transverse ridge routes and coupled μ1–3 scheduling

The transverse CCT is Hermitian indefinite.  The default ridge route solves
`(C_q + ridge·I) z = Z_q`; `rank_truncate` is the explicit alternative below.
All route boundaries use
`C_q[Q,M_T_x,M_T_y] = P(None,'x','y')` and
`Z_q[Q,M_T_x,R_y] = P(None,'x','y')`.

| route | factor/solve schedule | leading per-rank storage |
|---|---|---|
| local JAX | hoisted `lax.linalg.lu` once per q and channel; `lu_solve` per r-chunk through replicated/per_q gather tiers | sharded resident factor plus one whole `M_T²` gather tile |
| local batch-reshard | face → `P(('x','y'),None,None)` by two `all_to_all`s; each rank solves `ceil(Q/P)` complete matrices; inverse exchanges restore the face | measured floor `3·16·ceil(Q/P)·M_T·(M_T+R)`; refactorization per r-chunk is intentional |
| fully distributed | `distrib_la.factor('solve_lu',...)` runs batched `getrf` once per channel; `solve(FactorToken,Z_q)` runs batched `getrs` per r-chunk | factor `16·Q·M_T²/P`, each RHS/output `16·Q·M_T·R/P`, plus lower-order pivots |

The fully distributed token contract is implemented by both ScaLAPACK and
cuSOLVERMp.  Neither provider refactorizes per r-chunk, and neither gathers a
whole factor to one rank.  The batch-reshard route uses local JAX only; pad
`Q` to `P·ceil(Q/P)`, make `M_T` divisible by both mesh axes, and make `R`
divisible by `p_y`.

When all three transverse channels are fresh, `low_mem_bands=true`, and the
planner selects the face-Y cache, the coupled scheduler shares one
`Z_q[3,Q,M_T,R]` transform.  It solves and spills channels in the fixed order
μ1, μ2, μ3.  The three solves retain separate `Q` batches; production
does not flatten them to `3Q` or stack their outputs.  All three G-flat
accumulators are process-local host spills, with only the active
`P(None,('x','y'),None)` shard on the device.

The exact coupled increment over one transverse plan and the capacity gates
are in [`memory-model.md`](../architecture/memory-model.md#coupled-mu1-3-transverse-live-set).
The leading terms are `O(Q·M_T²/P + Q·M_T·R/P)` on the device and
`3·16·Q·M_T·N_G/P` bytes of host RAM per rank.  Admission enforces the
local route's measured factor-three operand floor, 50% allocator ceiling,
35% node-RAM spill ceiling, and the planner's fragmentation-safe device
budget.  If `auto` cannot fit the local route, it may use the fully
distributed token route.  An explicit local request is not changed; failure
falls back to sequential μ1→μ2→μ3.  Partial reuse is always sequential
and fits only missing channels; full reuse skips fitting.

## Transverse rank_truncate family

`transverse_zeta_solve = rank_truncate` ports the charge channel's
rank-truncating ζ solve to the transverse channels: per-q eigh of the
Hermitian INDEFINITE transverse CCT, drop |λ| < τ·|λ|_max
(τ = `transverse_zeta_rcond`), store the EXPLICIT truncated
pseudo-inverse C⁺ (no BBᴴ factor exists for an indefinite C⁺; explicit
C⁺ makes the per-r-chunk back-solve ONE GEMM).  The TRS-paired near-null
current modes the ridge merely lifts to its 1e-12 floor (κ~1e12) are
REMOVED, so κ_eff ≤ 1/τ by construction, and the per-q `n_keep` log is
the transverse basis-adequacy instrument.  Default remains `ridge`
(byte-identical legacy path) pending the calibration-driven flip.

Two plans, exactly the charge family's:

* **LOCAL** — replicated whole-tile eigh at the LOGICAL extent through
  the charge factor scaffolding (`_charge_factor_math` mode
  `'transverse_rank_truncate'`): batched under the replication cap,
  q-parallel over devices above the fold threshold, bit-identical
  across schedules/meshes/gather tiers by the same argument as the
  charge fold.  Valid at ANY centroid count on ANY mesh.  Carries the
  charge fold's P = nq saturation ceiling (per-channel nq is the same;
  announced whenever P > nq).
* **DISTRIBUTED** — `distributed_zeta_solve = distributed` (the same
  key as the charge tier; `distributed_lu` is an LU-family key and
  conflicts — refused at parse/resolve).  The distributed eigensolver
  (ScaLAPACK on CPU or cuSOLVERMp on CUDA) runs at the PADDED extent with
  the pad block ZEROED: pad eigenvalues are exactly 0, truncated at every
  τ, and never contaminate |λ|_max.  C⁺ is formed and kept 2-D-sharded
  (`_factor_c_q_distributed_rank_truncate(indefinite=True)`), back-solve
  is the charge tier's stacked 2-D GEMM.  Different (equally valid)
  gauge, ~κ_eff·ε vs the local plan — explicit opt-in, like the charge
  tier.

**Certified example (job 7885126, MoS2 4×4 bispinor deck, TLU host
.so):** bi4 (402c+143T, P=4) and bi16 (785c+275T, P=16) hoisted-vs-fused
EXACT-0 on eqp0/eqp1/eqp_g0w0/sigma_diag AND sigma_mnk.h5; scalapack
split-vs-fused (402c+136T, P=4) EXACT-0 likewise; distributed-vs-local
gauge delta 0.0 at .dat print precision, 1.1e-13 eV sigma_mnk.h5.
Measured effect at bi16: per-channel `zeta_fit.chunk.solve` 4.3–12.8 s →
1.7–2.8 s (per-r-chunk re-factorization gone), GW wall 214.3 → 201.2 s;
grows with n_rchunks·P at production mu_T.

## BSE stage — the plan family, and the one hole in it

Registered 2026-08-06.  Until then the BSE stage appeared in no plan
table, and the sandbox `INVARIANTS.md` row 6 recorded the gap as
"`bse_io._load_ring_subset` has no distributed counterpart".  **That
wording is wrong and is retracted here.**  `_load_ring_subset` is the
*single-device* full-file reader, it says so in its own docstring, and
it has REFUSED above one process since the BD.4 scorecard
(`bse_loading::_load_ring_subset`; the file was `bse_io.py` until the
2026-08-10 split).  Its distributed counterpart is
`load_bse_data_from_restart_sharded`, which has existed the whole time.

**Measured, MoS2 6x6 / mu=1496 / nq=36 / nb=200 / CUDA, Perlmutter job
56389339, cache-cold** (`bse_memprofile.py` census of every bundle
array's global shape, PartitionSpec and per-rank shard bytes, plus
`/proc/self/status` VmHWM):

| leg | loader | bundle GB/rank | `W_q` GB/rank | VmHWM GiB/rank |
|---|---|---|---|---|
| P=1 (1x1) | `_load_ring_subset` | 1.2468 | 1.2006 (whole) | 3.074 |
| P=4 (2x2) | sharded | 0.3342 | 0.3001 | 2.145-2.162 |
| P=16 (4x4) | sharded | 0.0909 | 0.0758 | 1.598-1.705 |

`W_q` divides by P exactly (1.2006 / 4 = 0.30015; at P=16 the global
grows to 1.2131 GB because `padded_mu_extent` rounds 1496 up to 1504,
then divides by 16).  The bundle total divides by 3.73x and 3.68x
instead of 4x for one reason only: ψ and `M` are single-axis sharded,
so they divide by `px` (= sqrt(P)) while `W_q` divides by `P`.  At P=16
they are 15% of the per-rank bundle and that fraction grows like
sqrt(P).

Per-rank scaling at the design envelope, from those shapes:

* `W_q`, `V_q0`, `V_q_full`, matvec `T`/`U`  →  `mu²·(...)/P`.  Sound.
* ψ_{c,v}^{X,Y} `(nk, nb, ns, mu)` and `M_{X,Y}` `(nk, nc, nv, mu)`
  →  `/px` = `/sqrt(P)`.  `M` is `nc·nv` times ψ and TWO copies are
  held for the whole run; it is the largest ψ-side object and the
  first thing to fix after the hole below.

### The hole: `bse/vq_interp` has no distributed plan

Everything above has two plans.  The arbitrary-Q exchange model has
one, and it is the local one.  `build_vq_evaluator` is on the
`exciton_bands` and `bse_k_grid` paths, so this is reachable from a
production deck, and it is the BSE stage's INVARIANTS-row-6 defect:

1. `zx["psi"] = fr["psi_full_y"][()]` in the ζ-transport reader: the
   WHOLE `(nk, nb, ns, mu)` ψ into host numpy on EVERY rank,
   unconditionally.  The module docstring quotes 3.7 GB at the MoS2
   reference.  Ungated.
2. `S_np` and `V_SRc_np`, two `(nq, mu, mu)` host mirrors, i.e. **two
   full `mu²·nq` tensors per process** — the evaluator's own docstring
   puts the pair at 26.8 GB/proc and names it as the node OOM.  Gated on
   `run_diagnostics` only.
3. `Fch`, `(nq, mu, nG)` host, **ungated**: allocated whether or not
   diagnostics run.
4. `_to_host(S_b/V_b/F_b)` gathers `(q_chunk, mu, mu)` onto every
   process.  Since 2026-08-11 `_to_host` is one line of delegation to
   `common.collectives.gather_to_host`, so the gather is the sanctioned
   L3 one and it announces itself; the memory it costs is unchanged and
   is what this row is still about.
5. `refit_vq` accepts `mesh_xy` and applies **no sharding constraint at
   all**: `C=(mu,mu)`, `Z=(mu,n_rp)`, `zeta=(mu,n_rtot)`,
   `ztG_box=(mu,n_rtot)` are whole-array device buffers on every rank.
   Its host fetches are no longer the problem — the last bare
   `device_get` on a μ-sharded array was replaced by `gather_to_host` at
   `93f8b572`, which is why `refit_vq` runs at P>1 at all — but nothing
   constrains the sharding of the arrays themselves.

The design for closing it is the charge-zeta family's, unchanged: keep
the `(q, mu, mu)` stacks 2-D-sharded on `('x','y')` end to end — `V_SRc`,
`C_q` and `P_R` already are on the device side — and delete the host mirrors rather
than gate them, replacing the diagnostics that consume them with
on-device reductions of the kind `exciton_bands._gate_stats_on_device`
already uses.  Item 1 is a per-rank
`(nk, nb_window, ns, mu/px)` hyperslab read through the same
`_read_psi_mu_sharded` the BSE loader uses.  Item 4 disappears with the
mirrors.  Item 5 needs the same `with_sharding_constraint` treatment
the W densifier got in 2026-07-31.

### BSE I/O: on the certified transport since 2026-08-06

`bse/` used to read its restart with plain serial `h5py` hyperslabs and
never touch `file_io.slab_io`.  Memory-wise that was always correct
(nothing larger than one rank's tile is ever materialised, and there is
no allgather); it just was not the parallel-HDF5 FFI collective read
that `GATES.md`'s `slab_io` row certifies.

`load_bse_data_from_restart_sharded` now goes through
`SlabIO.read_slab`.  The tile geometry is deliberately unchanged — the
port moves bytes, not contracts:

* `_MunuSlabPlan` restates the three on-disk V/W layouts (8-D legacy,
  6-D transitional, 3-D flat-q) as `(offset, shape, partition_spec)`;
  `_resolve_munu_reader` remains the single source of the layout facts
  and drives the serial path.
* `_slabio_read_munu` / `_slabio_read_psi` return exactly the shapes and
  PartitionSpecs the serial readers return at `trim=False`.
* `_read_bse_tensors` is the ONE place the transport is chosen, per
  load; `_bse_slab_io_backend` asks `gw_config.resolve_slab_io_backend`
  (extracted to module scope for this, since `bse/` builds no
  `LorraxConfig`).
* The serial readers remain and remain reachable — where no PHDF5 tier
  exists the router refuses above one process, and routing a
  single-process BSE through the allgather tier would buy nothing.

**Parity is bit equality**, not a tolerance: this selects the same
elements of the same datasets, so it is not a reduction-order change.
SHA-256 of each rank's own shard (no gather) of all 13 bundle arrays is
identical serial-vs-SlabIO on all 4 ranks, at `load_v_full` both False
and True.  The 8-D and 6-D layouts, which no deck has, were exercised by
writing one array of numbers in all three layouts at `N_mu=10` on a 2x2
mesh (so the mu pad is live): serial == SlabIO per layout, and
3-D == 6-D == 8-D per transport, on every rank.

**What the transport is actually worth, and what it is not.**  Measured
2026-08-06 (job 56389339 / 56393848), MoS2 6x6 restart, 1.2468 GiB of
logical payload, each leg on its own never-read copy of the file:

| where the file lives | P | serial reads | SlabIO reads | ROMIO `cb_nodes` |
|---|---|---|---|---|
| Lustre, stripe 4x1M | 4 | 1.98 s (0.62 GiB/s) | 1.41 s (0.88 GiB/s) | 4 |
| Lustre, stripe 16x1M | 16 | 1.09 s (1.13 GiB/s) | 0.53 s (2.33 GiB/s) | 16 |
| CFS (GPFS) | 16 | 3.89 s (0.32 GiB/s) | 3.31 s (0.37 GiB/s) | **1** |

The SlabIO leg at 16 ranks on Lustre reaches 2.33 GiB/s against the
2.919 GiB/s CLAIMS 69 certified for the phdf5 path at that rank count —
i.e. the port does reach the certified transport.  But the older "~0.1
GB/s, ~30x off" figure compared two different measurements: it was taken
on a **CFS-resident** restart, where ROMIO reports `cb_nodes = 1` and
neither transport can exceed ~0.37 GiB/s, against a Lustre 16-rank
number.  On matched filesystem and rank count the transport is worth
about 2x on the reads at P=16 and 1.4x at P=4.  **The larger lever is
where the restart file lives**: 2.33 vs 0.37 GiB/s, 6.3x, for the same
code.  A restart that a BSE run will read many times belongs on
`$SCRATCH` with `stripe_count = nranks`, not on CFS.

Whole-load wall barely moves (P=16: 4.33 s serial -> 3.61 s SlabIO;
P=4: 4.16 -> 3.70) because at this payload the load is dominated by
costs both legs share plus the collective `H5Fopen`, which the trace
prices at 0.69 s of the SlabIO leg at 16 ranks.  XLA compilation of the
read closures is 0.002-0.003 s and is not a term.

It IS a host-memory win, unlike the `read_q_slab` fix: VmHWM delta per
rank 1.198 -> 0.409 GiB at P=4 and 0.358 -> 0.300 at P=16, because the
serial reader stages each tile through a host numpy buffer before
`device_put` and the FFI path writes into the device buffer through a
pinned staging buffer.  Device peak is UNMEASURED — `memory_stats()`
returns `None` on the deployed jaxlib.

One byte-level defect on that path was found and fixed 2026-08-06:
`_read_vq0_sharded` read all `nq` q-tiles and used one, so `V_q0` cost
`nq`x its own size in disk traffic.  `_resolve_munu_reader` now returns
a `read_q_slab(q, ...)` single-q hyperslab for all three on-disk
layouts.  Load wall 12.4 s -> 7.4 s at P=4 (on the CFS-resident deck),
and the per-shard SHA-256 of all 13 bundle arrays is bit-identical
before and after on all four ranks — an element-selection change, so
exact equality is the right bar, not a gauge tolerance.  It is NOT a
measured memory win at that shape (VmHWM 2.145-2.162 -> 2.144-2.152
GiB, unchanged): the transient it removes is 0.32 GB/rank and is not
the binder.

### GW restart: the reader that has no sharded twin

`file_io.tagged_arrays.read_restart_state_from_h5` is the GW-side
equivalent of `_load_ring_subset` — a full-file `[:]` read of
`V_qmunu` / `S_qmunu` / `V0_noG0_munu` / `psi_full_y` on every rank —
and it is the ONLY reader of those datasets in `gw/`.  It is reached
from `gw_init.py:1593` on every `restart = true` deck.

Measured at P=4 before it was guarded (N_mu=1496, nq=36): it RAN and
returned correctly sharded tensors in 3.19 s, at a cost of **+1.53 GiB
of VmHWM on every rank**, silently.  The same read at the envelope
(N_mu=20000, nq=64) is 381.47 GiB per rank.  It now refuses above one
process and names `restart = false` — the chunked-ISDF producer, which
never materialises an N_mu^2-class object on one process — because
there is no sharded reader to name.

Building that reader is the open item: it is the `_MunuSlabPlan` +
`_slabio_read_munu` pattern above against the same three layouts, with
`P(None,'x','y')` for `V_qmunu` and `P(None,None,None,'y')` for
`psi_full_y`.  Until it exists, `restart = true` above one process has
no path.

## Pointers

`docs/dev/linalg_ffi.md` (facade + backend guards) ·
`docs/dev/env_vars.md` (every env knob, with calibration) ·
`docs/dev/mpi_collectives.md` (transport) · `docs/drivers.md` §gw (deck
keys) · `src/isdf/core.py` module comments (tier layout contracts,
collective payload chunking) · `src/gw/w_isdf.py` (W plan family).
