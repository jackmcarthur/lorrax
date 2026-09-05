# Raw-parent Z performance: output-spin scan

The output-spin loop in `isdf.core._z_q_face_parent` now runs as a
`lax.scan` inside its existing local `shard_map`. This limits overlapping
full-k IFFT buffers and shortens cold compilation. Parent band sums, typed
symmetry actions, FFT helpers, canonical restore and Sigma are unchanged.

## Measured scope

Si: 64 full k, 8 raw parents, 836 canonical / 840 packed centroids,
two-component spinors, 80 loaded bands, 2768 slots per tile. P4, one rank
per A100, `cuda_async@0.85`, JAX 0.9.1; all continuation launches pinned
JID 57941637. Timing arms used the same node, nid003456, with cold compile
and no profiler. No CPU or P>4 performance claim is made.

| Measurement | Baseline 21 | Scan 22 | Scan repeat 19 |
|---|---:|---:|---:|
| First Z compile + execute | 5.666 s | 3.159 s | 3.382 s |
| Cold five-tile zeta stage | 16.7 s | 14.0 s | 14.3 s |
| Z executable peak memory | 8.39 GiB | 5.62 GiB | 5.62 GiB |
| Reported GPU arena high-water | 9.07 GB | 6.09 GB | 6.09 GB |

The final stage is 16% shorter and static peak Z memory is 33% smaller.
Warm whole-tile timing varied and the first warm speedup did not repeat.
The settled Z kernel itself was about 107–108 ms versus baseline 100 ms;
long runs with many tiles may trade warm work against cold compile and
memory savings. Whole-G0W0 acceleration is not established: cold scalar
quadrature planning dominates its wall time.

The scan introduces no collective. Existing projector band-loop traffic
is unchanged. Memory still scales as O(nk * mu * nr / P); the improvement
is a smaller overlapping live set, not a new asymptotic scaling law.

## Numerical evidence

Final step `lx-Xg4-093631-1927753-4721`: all ten requested gate groups,
500 collected, 493 passed, 6 skipped, 1 known strict xfail. Five skipped
real-process tau cases were separately 5/5 on P4, worst relative error
2.996e-16. The other skip is WSL-only; the xfail is the known CPU-emulated
NaN reduction. This is the focused lane gate, not a full-suite census.
The final tiled zeta is bit-identical to the parent-05 reference.

Cold quadrature planning chose different schedules: 474 nodes in the old
reference, 481 in untouched baseline 21, and 489 in final 22. Consequently,
printed-digit comparisons failed by up to 1.524 micro-eV even for untouched
source. Numerical-only step `lx-Xg4-094201-2023830-8777` copied the original
reference's quadrature cache to the unchanged deck's normal auto-cache
location. With all cache containment/error guards active and fresh zeta/W,
**eqp0 and eqp1 each matched all 224 reference rows to the printed digit**
(tolerance zero). Its zeta relative difference was 2.665e-16. No cached-run
wall time is used as a speed claim.

## Rejected candidate and provenance

Restore/crop + solve fusion gave a 16.6 s zeta stage versus baseline 16.7 s
and no arena reduction. It is not retained. Its standalone restore was
usually slower across mu=836/1672/3344 and nq=8/16/64. The eager crop's
centroid-axis replication remains registered in the sandbox issue ledger.

Evidence root:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Si/99_psi_irr_zeta_2026-09-05/perf/`.

- Baseline 21: `lx-Xg4-093104-1830041-6046`.
- Scan repeat 19: `lx-Xg4-092448-1753674-9414`.
- Restore-only 20: `lx-Xg4-092604-1760570-3964`.
- Final 22 and reference-rule 23: steps above, both exit zero.
- Per-variant `source.diff`, `tiles/driver_rank0.log`, HLO analyzer summaries,
  collection/JUnit logs, and approved eqp/HDF5 parser outputs preserve the
  exact scope. Early profiling and rejected experiments remain on disk.

The full sandbox report is `reports/psi_irr_perf_2026-09-05/report.md`;
claims 810 and 813 own the numerical verdicts. The obsolete canonical-pad
fixture was repaired separately in af86f0aa without changing its producer.
