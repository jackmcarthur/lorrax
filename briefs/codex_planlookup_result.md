# Lookup-first delivered planner result

Branch `perf/plan-lookup-2026-08-31` replaces the 21-step tolerance ladder
with shipped-table lookup.  Noncrossing windows use the minimax service's
selection order and physical `bound / x_min` rescaling, then pass the fitting
lattice, refined lattice, noise, and factor-growth gates.  Crossing windows
try matching HGL tables first and otherwise make one deterministic fixed-time
IRLS fit.  The cache epoch is 3; zero-time rules and unserved product support
refuse.  No explicit state--pole or direct fallback exists.

## Frozen Na evidence

Evidence input directory:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel`

Command: the prescribed CPU environment followed by
`python3 tools/benchmark_delivered_lookup.py`.

Achieved fitting-stage wall: **1.355172 s** for all six windows.  The four
noncrossing windows were served with 9, 11, 12, and 8 nodes.  The two crossing
windows used the single-fit fallback with 66 and 48 nodes.  Total cost was 154
window--tau pairs.

| Window | Achieved residual | Incumbent | Ratio | Noise gate |
|---|---:|---:|---:|---:|
| cond:resonant | 7.383451e-4 | 8.034575e-4 | 0.9190 | pass |
| cond:state_tail | 4.334706e-6 | 2.678331e-4 | 0.0162 | pass |
| cond:pole_tail | 8.318314e-5 | 8.268273e-5 | 1.0061 | pass |
| val:bulk | 7.700864e-6 | 2.245508e-5 | 0.3429 | pass |
| val:resonant | 5.919597e-4 | 1.076877e-3 | 0.5497 | pass |
| val:pole_tail | 2.438188e-6 | 6.440662e-4 | 0.0038 | pass |

Worst achieved/incumbent residual ratio is **1.0061**.  Every row also passes
`kappa_p99 * 6e-8 <= 0.05 * target`.

## Verification and remaining integration evidence

`tests/test_delivered_windows.py`: **13 passed** in 2.04 s in the prescribed
CPU environment.  The focused gates cover lookup-only noncrossing planning,
one-call crossing fallback, zero-time refusal, cache receipts, branch
independence, and removal of the tolerance ladder.

The full P=4 deck comparison is intentionally left to the execution lane.  It
must re-establish the campaign's 0.278609 meV maximum delivered-vs-pane result;
this lane establishes the requested fitting-stage wall and per-window
validation/noise evidence only.
