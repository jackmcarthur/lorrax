# ACARRY — two-axis shared-pole b carrier: accepted on both reference decks

Branch `lane/sp-acarry-2026-09-11`, tip `d08c7700`, over `cc6c184d` (carrier
implementation) and base `810c260bd8e006607eb5643430111403f11d248e`. On branch,
not landed: `git merge-base --is-ancestor d08c7700 origin/main` is false.

## What the change is

The Sigma face reader returns `b[parent,mu,spin,Kpad]` at `P(None,'x',None,'y')`
and `P(None,'y',None,'x')` — tiled on both mesh axes, as psi is. The base
returned `P(None,axis,None,None)`, mu on one axis with **K replicated**, so the
reader undid the two-dimensional sharding the writer had honoured. Poles stay
replicated, correctly: `[parent,Kcap]` is N_q*N_mu, not N_q*N_mu^2. Padding is
storage only; counts and the separate causal d mask every inactive column. Both
Sigma paths — resident all-parent and streamed bounded panels — consume the same
reader, which is why the defect was independent of `low_mem_bands`.

Endpoint transport stays the canonical spatial ring, now carrying only the
complementary-axis K tile, and capacity admission uses that layout. W contracts
through `build_G(layout="face")` and its eagerly warmed native GEMM plan. The
same-time transpose conjugates both factor faces, never d. Window-dependent
compact K slices are gone: shrinking a globally sharded pole axis redistributes
ownership. Admitted panel widths are fixed.

## Measured

Per-rank resident factor carrier, MEASURED on both decks with a bounded probe
that records what `read_shared_pole_faces` actually returns.

Si P4, base against tip on the same deck, both arms node-exclusive, all four
ranks identical within each arm: face [8,384,1,2010], global 98,795,520 B each.
Base `810c260b` (58209192.29) at `P(None,'x')` / `P(None,'y')` holds 98,795,520 B
per rank for the pair; tip (58209192.28) at `P(None,'x',None,'y')` /
`P(None,'y',None,'x')` holds 49,397,760 B. Measured reduction exactly 2.00x,
47.11 MiB per rank.

Na P16 (step 58209192.25, every one of 16 ranks identical): each face is [29,960,1,1592], global
709,140,480 B, `local 44,321,280 B` = global/16 exactly, at
`P(None,'x',None,'y')` and `P(None,'y',None,'x')`; poles stay replicated at
369,344 B. Per rank the pair is 88,642,560 B. The same reader at `810c260b`
returns `P(None,axis,None,None)`, i.e. global/4 per face and 354,570,240 B/rank
for the pair, so the reduction is 4.00x, about 253 MiB per rank.

| Gate | Si P4, job.step 58209192.10 | Na P16, job.step 58202457.3 |
|---|---|---|
| model `factor` + `poles2_ry2` bit-exact | yes, all 8 parents, max abs 0.0 | yes, all 29 parents, max abs 0.0 |
| identity / metadata / physical deck | equal | equal |
| Sigma max change (gate 2 meV) | 6.28e-12 meV | 0.01563 meV |
| quadrature rules certifying | 8 of 8 | 12 of 12 |
| parent invariants (5 each) | 8 parents, 0 failures | 29 parents, 0 failures |
| every-rank allocator peak | 4/4 ranks delta 0 B | 16/16 ranks delta 0 B |

The zero peak delta is expected, not a contradiction: `peak_bytes_in_use` is a
whole-run high-water mark, set on both decks during ISDF/screening roughly 200 s
before the synthesis runs, and 45 MiB (Si) / 236 MiB (Na) are 1.1% / 2.8% of it.
The peak gate can show non-increase but structurally cannot show this saving.

P4 planted multi-device gate at this tip, 58209192.5: store 10/10, all four Sigma
ranks PASS, services symmetry PASS. Its per-rank optimized HLO shows the
two-stage form: `pack_x` two all-to-alls at `replica_groups={{0,2},{1,3}}` (x),
`pack_y` two at `{{0,1},{2,3}}` (y), no all-gather anywhere, and the synthesis
kernel itself collective-free on every rank.

## Review items in `src/gw/mpa/sigma.py`

- **(a) FIXED.** Thirteen unconditional `timing.fence` calls, nine of them in
  `_integrate_sigma_batches`, which serves the incumbent elementwise-MPA route
  too. They now go through the module attribute `_band_fence`, a no-op that a
  measurement harness rebinds to `timing.fence` — the mechanism the campaign
  already uses for `timing.section` and `read_shared_pole_faces`. No dial, env
  var, cache, fast path or constant. The incumbent route binds `_unfenced`
  whatever is installed. Cost on the shared-pole route, Si P4 rank 0: 3.467 s of
  fence rows (3.184 s barrier + 0.283 s drain) in a 14.37 s tau sweep, an **upper
  bound** on recovery there since `rank_wait` absorbs rank skew that reappears at
  the next collective. On the INCUMBENT route the recovery is complete and
  measured: two interleaved A/B pairs on byte-identical decks, four consecutive
  node-exclusive steps on one node, give base `810c260b` (58209192.26, .31) 18
  fence rows and 3.834 / 3.639 s with 15.269 / 14.983 s tau sweeps, against tip
  (58209192.27, .32) with **zero** fence rows and 10.692 / 10.694 s sweeps -
  4.433 s in the means, 29.3%. The base pair's own spread is 0.287 s and the tip
  pair's is 0.002 s; mean fence rows 3.737 s leave a 0.696 s residual, mostly in
  `tau.kernel` (8.895 -> 8.321 s), which is the host/device overlap a per-tau-node
  `block_until_ready` was forbidding. With n=2 per arm that residual exceeds the
  observed spread; no distribution is claimed for it. Band
  measurement still works: candidate and reference show the
  same 158 fence rows and 6456 entries, which is also what licenses the peak
  comparison as one instrumentation regime rather than two.
- **(b) REFUTED at this tip.** `compact_kernels` existed at `810c260b:342-401`;
  `cc6c184d` removed it. The surviving per-panel `kernels` dict is pre-enumerated
  from at most two admitted widths; on both reference decks the cardinality is 1.
- **(c) FIXED.** `p2-p1` could collapse to zero (both ends are ceilinged byte
  counts) and divided with no guard inside a capacity planner; the affine price
  now refuses a non-positive slope and the column sizing is integer floor
  division, exact past 2**53. The store/current-map check compared an int byte
  count against a Python true division; it now compares the five geometry
  integers the ledger records.

## Open, and not this lane's to fix

`low_mem_bands = false` is unreachable on the shared-pole route at the base and
at this tip. `src/gw/wavefunction_bundle.py:313` maps it to `layout="legacy"`,
and `src/gw/response_bank.py:144-148` refuses a legacy carrier before Sigma.
Base refused in 25 s (58209192.9), this tip in 23 s (58209192.14), identically.
One clause of the owner's ruling therefore cannot be satisfied as written; the
two reachable Sigma face-read paths (resident, streamed) are gated instead, with
the streamed path covered by the planted P4 suite and
`tests/test_shared_pole_carrier.py:90-151` rather than by a reference deck.

Full evidence, scopes and what is not claimed:
`reports/shared_pole_push_2026-09-07/acarry/report.md` in the sandbox, and
`runs/frequency_integration_sandbox/361_acarry_20260911/`.
