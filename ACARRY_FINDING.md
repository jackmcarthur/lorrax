# ACARRY — prepared carrier change; GPU acceptance pending

Heavy lane, base `810c260bd8e006607eb5643430111403f11d248e`, branch
`lane/sp-acarry-2026-09-11`. The owner required b to be sharded like ψ in
both low-memory and resident paths.

The common reader now returns `[parent,mu,spin,Kpad]` factors at
`P(None,'x',None,'y')` and `P(None,'y',None,'x')`. Replicated poles remain
replicated. Padding is storage only: counts and the separate causal d mask
all inactive columns. Both Σ paths consume this same reader.

Endpoint transport remains the canonical spatial ring, now carrying only
the complementary-axis K tile; capacity admission uses that layout. W
contracts through `build_G(layout="face")` and its eagerly warmed native
GEMM plan. Same-time transpose conjugates both factor faces, never d.
Window-dependent compact K slices are removed because they redistribute
sharded K ownership; admitted panel widths remain fixed. This trades some
GEMM work for a predictable carrier and needs measured sweep attribution.

CPU checks: 18 tests passed on four emulated CPU devices, covering the
reader's ragged column slices, two-axis residency, causal transpose, changed
endpoint tables, nonlocal symmetry, metadata, and both resident/streamed
synthesis orchestration. Evidence:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/361_acarry_20260911/cpu_03.log`.
No GPU model, peak, Sigma, timing or optimized-HLO acceptance is claimed.
The native face contraction uses the existing CUDA planned GEMM surface;
Frontera/CPU production is not certified by these emulated algebra checks.

SP-M2 job `58190627` remains PENDING (Priority); it has not run an ACARRY
step. Owed: planted P4 gate; all 8 Si and 29 Na parent bit-exactness; both
low_mem_bands modes; every-rank HLO and peak; Sigma <=2 meV; all rules
certifying; matched band measurements with control repeat.
