# Measure compression result

Branch `study/measure-compression-2026-08-31`; offline heavy lane; evidence:
`runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/measure_compression_audit.md`
and `measure_compression_baseline.json` beside it.

The exact raw-pair audit finds **no node penalty** from the 25x25
tail-refined lattice. The frozen Na valence crossing window (91,008 raw
pairs) and sign-definite window (76,800 pairs) each need a minimum of 7 ROQ
nodes when fitted on the 25-bin, 50-bin, 100-bin, or raw measure and accepted
only against raw error plus the noise gate. Node saving from changing the
compressor is therefore **no material saving: 0 nodes**. Compressed audits
took 2.1--3.0 s; raw fits took 81.5--105.1 s and are not an allowed fallback.

Compression can nevertheless flatter scores. On the eight frozen DEV-80 toy
crossing branch rules, 25-bin errors under-report raw by as much as 66.3%.
The independent 50-bin validation reduces the worst case to 24.1% but falsely
accepts 2/8 rules at the common 1e-3 target. On the two controlled Na fits,
the compressed scores are instead conservative by 0.087--5.85%; the frozen
Na incumbent's only optimism is 0.23% on the sign-definite window.

I did not replace the production lattice: tested mass-quantile, mixed
count/mass, mass-centroid, and higher-count variants were non-monotonic and
scored frozen rules anywhere from 0.925x to 93.1x raw, so none is a safe
Occam fix. The service test now pins determinism, roundoff mass conservation,
and the unchanged `(bins+1)^2` / `(2*bins+1)^2` payload bounds. The remaining
toy score risk needs a bounded certificate at the rule consumer; explicit raw
pair evaluation is both too slow and forbidden by the campaign ruling.
