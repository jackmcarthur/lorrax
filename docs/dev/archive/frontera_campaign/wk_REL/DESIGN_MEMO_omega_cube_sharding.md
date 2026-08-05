# DESIGN MEMO — Σ_c ω-cube sharded-consumer workstream (no code; handoff)

wk_REL, 2026-07-28.  Author context: Σ-stage perf campaign, branch
`fix/zq-band-gather-device-invariance` @ 9e6f7d0.  Cites: scorecard AK.9
(second named lever), jobs 7878038 / 7878092 / 7878110 / 7878233 / 7878276,
ladder runs run_L1_b256 (nb=256) and run_L3_b512_c5000 (nb=512, job 7878263).

## 1. Current state (measured)

Σ_c(ω,k,m,n) is accumulated as per-rank host tiles in the (m_X, n_Y)
sharding the reduce-scatter projector emits (16× smaller per rank at 8×8
than the full cube).  At stage end, ONE `process_allgather` reconstructs
the FULL cube on EVERY rank (`sigma.host_gather` row; the 4-per-branch
round trip was removed at dc30af4 — 4→1, parity exact-0).  The driver then
re-uploads it (`jnp.asarray(sigma_kij_host)`) as a replicated device array
for the consumers.

The reconstruction is P-INDEPENDENT and grows as n_ω·nk·nb²·16 B:

| deck | nb | full cube / rank | measured `sigma.host_gather` |
|---|---|---|---|
| AQ 4962c (job 7878233) | 128 | 172 MB | 0.24 s |
| run_L1_b256 (7878276)  | 256 | 688 MB | 0.83-3.9 s |
| run_L3_b512_c5000 (7878263) | 512 | **2751 MB** (HLO module_0962 `jit__identity_fn`, c128[41,16,512,512]) | 3.22 s |
| extrapolated | 1024 | ~11 GB | — |

Verified: L3 ran the committed tree (its log carries the post-9e6f7d0
`d2h_wait` rows); the 2751 MB collective is the SINGLE end-of-stage gather
(count=1, full-ω shape — not the old per-branch path).  The wall cost is
still small (3.2 s of 401 s at nb=512), but the **replicated host+device
RESIDENCY** (2×2.75 GB/rank at 512b, ~2×11 GB at 1024b) is what collides
with the scaling target (thousands of low-memory ranks; no rank may be
required to hold what sharding can split).

## 2. Consumers of the full cube (enumerated from code)

All in `src/gw/ppm_pipeline.py` unless noted; today each reads the
replicated `sigma_c_omega` (n_ω, nk, nb, nb):

1. `_inject_analytic_head` (:105) — ADDS the analytic q→0 head.  The head
   term is (n_ω, nk, diag-structured) and small; the add is elementwise →
   trivially shardable.  (Streamed KIJ_STREAM mode already RMWs the h5
   instead — an existing no-full-cube precedent, single-process only.)
2. `_eval_sigma_c_at_dft_energies` (:203) — takes
   `extract_sigma_diag_replicated(sigma_c_omega)`: only the (n_ω, nk, nb)
   DIAGONAL feeds the eqp interpolation/solve.  The diagonal of an
   (m_X, n_Y)-sharded cube lives on the mesh-diagonal ranks; a gather of
   the DIAGONAL is n_ω·nk·nb·16 B = 5.4 MB at nb=512 — nb² → nb.
3. `_write_sigma_omega_h5` (:273) — sigma_mnk.h5 writer, already routed
   through SlabIO (`backend=config.backend.slab_io`, :307; also
   gw_output.py:220).  SlabIO's sharded writers (zeta_q.h5 / V_qmunu.h5
   precedent) can consume per-rank tiles directly — no full cube needed.
4. `degen_average` / sigma_diag.dat / eqp0-1.dat writers (gw_output,
   sigma_output) — diag-derived, covered by (2)'s diagonal object.
5. `sc_iteration` — captures `sigma_c_omega` for the SC loop; needs the
   same layout decision applied recursively (out of scope for round 1).
6. Driver tail `jnp.asarray(sigma_kij_host)` (ppm_sigma.py) — the
   replicated re-upload; disappears when consumers accept tiles.

## 3. Proposal — two-plan doctrine (per the scaling-target note)

Plan A (default, unchanged): today's replicated fast path.  Correct at any
P, cheapest to reason about, required fallback for consumers not yet
ported.  Plan B (input-flag-gated, e.g. `sigma_omega_layout = sharded` in
gw.in — policy via declared input, never env; QUALITY_PATTERNS #8): the
cube stays as per-rank (m_X, n_Y) host tiles end-to-end:

- head injection: rank-local elementwise add on tiles (movement-free);
- diag/eqp path: gather ONLY the diagonal (nb² → nb bytes, 5.4 MB@512b);
- sigma_mnk.h5: SlabIO sharded write straight from tiles;
- `SigmaOmegaResult.sigma_c_kij` becomes a tile handle (small dataclass:
  tiles + index + sharding) with an explicit `.replicated()` escape hatch
  that reproduces Plan A on demand (the promise-contract seam, pattern #6).

Gates: every Plan-B consumer is data-movement-only relative to Plan A →
**bit-parity required** (sigma_diag/eqp0/eqp1/sigma_mnk max|diff|=0.0, the
campaign's standing suite) at nb=128 AND nb≥256, restart-gated (AC.4
harness; J.7 band-window rules).  Configuration-lattice coverage (pattern
#2): P ∈ {1, small, 64}, square + rectangular mesh, KIJ_HOST and
KIJ_STREAM, sym deck at minimum.  HLO/collective-table evidence that the
2751 MB `jit__identity_fn` module is GONE from the dump under Plan B
(pattern #4: the optimized HLO is the only ground truth).

Envelope statement: Plan B removes the last nb²-replicated Σ object; per-
rank Σ residency scales as n_ω·nk·(nb/p_x)·(nb/p_y)·16 B — flat in N_μ,
shrinking in P, linear in n_ω·nk — valid to n_atoms → hundreds, nb → 2048,
P → thousands, CPU and GPU alike.  Nothing in the design keys on today's
shapes.

## 4. Honest cost estimate

- Walltime won today: seconds (3.2 s @512b) — this workstream is a MEMORY/
  scaling play, not a speed play; say so in any claim.
- Effort: head-injection + diag path are small (≈1 day each with gates);
  SlabIO tile write is plumbing against an existing API (≈1-2 days);
  sc_iteration and KIJ_STREAM interplay are the long poles (the flag can
  refuse `self_consistent=true` in round 1 — broken-promise pattern #6
  demands the refusal be at resolve time).  Est. 1 week incl. lattice
  gates; the failure line (a consumer silently gathering — pattern #4's
  hoisted-gather class) is exactly what the collective table catches.
- Risk: LOW numerically (movement-only, bit-parity-gated); MODERATE in
  coverage (six consumers × the config lattice).

## 5. Pointer

Scorecard: appended to AY named-not-done.  Prior art: AK.9 ("deserves its
own workstream"), dc30af4 (4→1 gather), _MemoryTileSink.host_tiles()
(the tile seam Plan B extends), _AccumMode.KIJ_STREAM (no-full-cube
precedent), wk_REL/docs/sigma_perf_results.md (full campaign record).
