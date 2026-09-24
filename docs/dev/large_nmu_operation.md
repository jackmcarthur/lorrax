# Large-N_μ operation: running fully distributed

LORRAX targets thousands of low-memory processes, where no
`(N_μ·n_spinor)²` tile fits on one rank. Every dense solve therefore has two
storage plans:

* **local**: whole per-q tiles, scheduled q-parallel over devices at P > 1.
  Mesh-invariant; it is the numerical control.
* **distributed**: 2-D block-cyclic factorisation over the whole mesh through
  `distrib_la` (ScaLAPACK on a host mesh, cuSOLVERMp on CUDA). The only plan
  whose factorisation work and memory both divide by P. It is a different,
  equally valid numerical gauge: agreement with the local plan is `κ·ε`, not
  bit-exact.

The deck selects between them with one key, `linalg = local | distributed`;
the fields it resolves to are tabulated in [linalg_ffi.md](linalg_ffi.md#the-deck-dial).
An explicit distributed request refuses rather than downgrading.

Conventions: mesh `(P_x, P_y)`, `P = P_x·P_y`, `μ` = padded centroid count,
`Q` = the stage's q extent, `R` = the ζ fit's real-grid tile width, all
buffers complex128 (16 B).

## Per-stage plans and per-rank memory

| stage | `linalg = local` | `linalg = distributed` |
|---|---|---|
| ζ CCT `C_q` build | always 2-D sharded, `P(None,'x','y')`: `Q·μ²/P` | same |
| ζ charge factor (`rank_truncate`) | replicated whole-tile eigh pseudo-inverse, one q-batch at a time under `LORRAX_ZETA_REPLICATE_CAP_GIB`; q-parallel above `Q·μ³ ≥ 5e9`, compute `ceil(Q/P)·μ³` per rank | distributed eigensolver, truncation on the replicated spectrum, `C⁺` kept 2-D sharded: `Q·μ²/P` stored, compute `Q·μ³/P`-class |
| ζ back-solve (per tile) | at or below `LORRAX_ZETA_GATHER_CAP_GIB` the small factor stack may be gathered (`Q·μ²·16` B per rank per tile); above it each factor stays on its q owner and only the RHS moves, `2·ceil(Q/P)·μ·R·16` B | one stacked GEMM `C⁺·Z` with both operands 2-D sharded; received bytes `Q·(μ²/P_x + μ·R/P_y)·16` per tile |
| ζ transverse factor (ridge LU) | `lax.linalg.lu` once per q and channel, `lu_solve` per tile; or the batch-reshard route (below) | `distrib_la.factor('solve_lu')`: one batched `getrf` into a 2-D-sharded `FactorToken`, `solve(token, Z_q)` runs `getrs` per tile |
| ζ `Z_q` build and G-flat write | always sharded; SlabIO collective hyperslab writes, no gather | same |
| W Dyson solve (`gw/w_isdf`) | q-parallel per-q dense LU: `ceil(Q/P)` whole `(μ, μ)` tiles per rank | `plan('solve_lu', backend='distributed').batched`: `Q·μ²/P`, μ axes never leave `P(None,'x','y')` |
| W ladder resolvent (`bse/w_ladder`, `screening_diagrams = w_bse`) | one code path for both layouts; no whole-`μ²` per-rank object; seed `P(None,'y',None)`, readout `P('x','y')`. Scaling envelope: the module docstring | same |
| eigensolves (htransform `fH_q`, `vq_interp` `C_q`, SC) | q-batched native eigh, one whole `(n, n)` matrix per device | one tile spread over the mesh; square or 1-D mesh, `n` divisible by both axes |
| BSE restart load | `bse_loading.load_bse_data_from_restart_sharded` through SlabIO: per-rank `(μ, ν)` hyperslabs, no allgather; `W_q` at `P('x','y',None,None,None)` = `nk·μ²/P` | same |
| BSE matvec (`bse_stack_matvec`) | `shard_map` scanning the trial axis, one `T` alive; `T`/`U` at `P(None,'x','y',None,None,None)` = `nk·n_s²·μ²/P` per trial; no collective carries a `μ²` payload | same |
| BSE coarse→fine W densify (`bse_densify.make_w_densifier`) | one jit, `out_shardings` pinned, FFTs inside `shard_map`: peak one `(μ/P_x, ν/P_y, nk_fine)` tile | same |
| BSE arbitrary-Q exchange (`bse/vq_interp`) | **no distributed plan** (below) | — |

Distributed backends check platform, compiled handler, one process per
device, mesh geometry and divisibility before any collective
([distrib_la](../services/distrib_la.md#contract)). A square mesh is the
portable choice: ScaLAPACK needs square or 1-D, and cuSOLVERMp eigh needs
square.

The transverse ridge routes (local JAX, local batch-reshard, fully
distributed) and the coupled μ1–3 live set are specified with their capacity
equations in the [memory model](../architecture/memory-model.md#transverse-ridge-solve-routes);
their schedule is in [Face-ψ ζ fitting](../architecture/zeta_fit_face_psi_cct.md#coupled-current-schedule).

## Where the crossover is

The local charge factor's only parallel axis is q, so it saturates at
`P = Q`: every rank past `Q` idles for the whole factor stage, and the run
announces this whenever `P > Q`. The wall-time crossover between the plans is
therefore set by `P/Q`, not by μ alone. On MoS₂ 4×4 (μ ≈ 3000–4800, Q = 10) the
distributed factor is 2.1× slower than the q-parallel local factor at
`P/Q = 1.6` and 1.6× faster at `P/Q = 6.4`. The distributed plan relieves
per-rank peak memory only when the replicated `(Q, μ, μ)` gather is the
binding allocation; a memory claim for it must name the binder it removes.

## Thresholds

| knob | default | decides |
|---|---|---|
| `_QPARALLEL_MIN_NQ_MU3` (module constant; `LORRAX_ZETA_QPARALLEL` overrides) | 5e9 | the replicated charge factor executes q-parallel above it; below, two staged reshards and one compile outweigh the saving |
| `LORRAX_ZETA_GATHER_CAP_GIB` | 4 | the local back-solve's gather-versus-resident boundary (data movement only; numerically free) |
| `LORRAX_ZETA_REPLICATE_CAP_GIB` | 4 | whether the rank-truncating factor may run replicated at all; per q-batch, so μ ≤ `sqrt(cap/16)` |
| `LORRAX_COLLECTIVE_CHUNK_MB` | 128 | payload of one emitted collective in the distributed tier (host-level q-block loop XLA cannot re-fuse); a single q whose collective exceeds it is sent whole with a warning |

Spellings and grammar: [env_vars.md](env_vars.md). The ScaLAPACK
workspace and MKL-thread behaviour: [linalg_ffi.md](linalg_ffi.md#inside-the-scalapack-handlers).

## What does not yet divide by P

* **`bse/vq_interp`** (on the `exciton_bands` and `bse_k_grid` paths). `Fch`,
  the `(Q, μ, nG)` cleaned long-range form factors, is a host array on every
  process, unconditionally. With `run_diagnostics`, `S` and `V_SRc` add two
  `(Q, μ, μ)` host mirrors per process. `C_q` and `V_SRc` are already 2-D
  sharded on device; the fix is to keep the `(Q, μ, μ)` stacks sharded end to
  end, delete the host mirrors, and replace the diagnostics that read them with
  on-device reductions.
* **BSE ψ and `M` stacks.** `ψ_{c,v}` `(nk, nb, n_s, μ)` and
  `M` `(nk, n_c, n_v, μ)` are sharded on one mesh axis, so they divide by
  `√P` while `W_q` divides by P. Two `M` copies live for the whole run; they
  are the largest ψ-side object.

## Launch

The launcher, not the deck, owns the transport. CPU collectives run
`impl=mpi` ([MPI collectives](mpi_collectives.md),
[transports](../environment/transports.md)). The `srun --mpi=` flavour must
match the MPI stack (`pmi2` for Intel MPI on Frontera, `cray_shasta` on
Perlmutter); a mismatch gives every rank a private `MPI_COMM_WORLD`, and
[SlabIO](../architecture/slab_io.md) owns that failure and its refusal.
