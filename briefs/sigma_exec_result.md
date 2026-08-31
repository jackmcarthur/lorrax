# Lane E — Sigma execution result

**Numbers first.** The accepted P=4/BFC@0.85 production baseline remains
**63.179 s Sigma / 86.956 s wall**, with **146 tau dispatches** for 115
logical `(window,tau)` pairs.  No execution optimization passed both memory
and numerical checks in this sprint.  The output reproduced the named panes
control at **0.195875573 meV max / 0.008833915 meV RMS** in
`sigma_c_kij_ev`, and was bit-identical to the prior batch-4 delivered arm.

The newly working stage-split P=4 measurement counted 147 calls including its
discarded prewarm: W phase **1.981 s**, G build **0.967 s**, fused GW FFT
convolution **20.082 s**, and band projection **1.015 s**.  Thus the fused
FFT convolution is **83.5%** of the four explicitly timed tau stages.  The
diagnostic Sigma wall was **40.652 s**, but it is deliberately blocking and
used a complete plan-cache hit, so that wall is not an A/B against the fused
63.179 s production baseline.

Cache-cold optimized HLO on the 2x2 mesh has exactly one module each for W
build, G build, fused GW convolution, and projection despite 147 calls: no
per-window kernel recompilation.  W/G/convolution contain **zero**
all-gathers, all-to-alls, or reductions; projection contains the expected
**two reduce-scatters**, **zero** all-gathers/all-to-alls, and two transposes.
The convolution is one `lorrax_mklfft_gw_conv` custom call.  This makes the
FFT handler, not collective placement or Python retracing, the measured hot
rung.

Pole batching was not a safe win.  Batch 8 failed before its sweep on a
measured **5.87 GiB** BFC allocation on all four A100 ranks.  Batch 6
completed and reduced dispatches **146 -> 105** (-28.1%), but changed the
delivered plan **115 -> 74** pairs, incurred a plan-cache miss, and regressed
end-to-end to **142.433 s Sigma / 157.342 s wall**.  Its control difference
also rose to **0.235789104 meV max / 0.009574672 meV RMS**.  Therefore neither
batch schedule is proposed as the execution fix.

One tight source fix did land: MPA now prewarms the documented stage-split
Python dispatcher by executing and blocking one discarded real-shape call,
matching the existing two-point PPM prewarm contract, instead of crashing on
the nonexistent `.lower()` method.  This is what enabled the breakdown and
the HLO evidence above; it does not alter the production kernel.

Verification: the required CPU gate passed **134/134** before the fix; the
post-fix gate adds `test_mpa_disk_pipeline.py` and is recorded in the evidence
log.  The real P=4 stage-split driver completed with unchanged batch-4
numbers.  Evidence:
`/pscratch/sd/j/jackm/wt_sigma_exec_2026-08-31/tmp/sigma_exec_p4_20260831`
(`combined_p4.log`, `batch6_p4.log`, `stage_p4/hlo_summary.{md,json}`, and
rank-specific cache-cold HLO dumps).  Branch:
`perf/sigma-exec-2026-08-31`.
