# Delivered-pole census fast path

## Result

The Na 24-band P=4 cold census now takes **1.480791 s** (maximum rank time,
1.112106 s + 0.368685 s for the two four-pole batches), below the 1.5 s
requirement. The branch-invariant part of the instrumented host baseline took
70.753982 s, so the delivered speedup is 47.8x. On this base the old path
repeated that work for both causal branches and spent 135.685 s in the four
measure calls.

Evidence directory:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/codex_censusfast_24b`

## Profile and implementation

The persistent baseline replay is `baseline_profile_replay.log`, JID/step
`57754440.25`, source `942fe215`. For the two branch-invariant pole batches on
rank 0, the measured 70.754 s divides into 31.592 s in
`_axis_cloud_weights`, 14.626 s in `np.add.at`, 5.717 s transferring the two
6.30 GB local shards, 3.357 s in validity masks, 2.899 s in bounded
coordinates, 0.307 s in fixed collectives, and 12.256 s in the surrounding
host selection/allocation work. Thus the expensive seam was the host handling
of roughly 49.2 million values per pole, not the 15 KB interval collectives.

The new path keeps each large JAX shard resident. A jitted per-pole kernel
checks finiteness and pole validity, constructs 4K-value private histograms,
and reduces them to the existing mass/first-moment lattice. One tree transfer
returns the compact batch table; only that table is summed between processes.
The measured payload is 120,096 bytes per four-pole batch, or 30,024 bytes per
pole including three counters (30,000 bytes of moments). A one-entry weak
reference cache reuses the branch-invariant result for the second causal
branch. NumPy inputs retain the small host fallback. There is no direct-pair
path or state-by-pole escape hatch.

## Verification

- Full Na cold run: `final_cold.log`, JID/step `57754440.21`, source
  `1e0c786e`; exit 0, six windows, 137 `(window,tau)` pairs, 137 distinct tau,
  zero direct terms. The follow-up isolated capture, `measures_current.log`,
  JID/step `57754440.24`, measured 1.494213 s at the slowest rank and confirms
  the result is repeatable below 1.5 s. Both used P=4 and BFC at `.85`.
- Fingerprint compatibility: `measure_canonical_compare.txt` compares the
  saved old and new compact measures. All 3,496 cell/weight scalars match at
  the cache fingerprint's seven-significant-digit canonicalization (zero
  mismatches). The cold baseline and optimized plans both have six windows,
  137 pairs, and zero direct terms. This uses the brief's identical-window-
  census proof; derived envelope summation changes the internal cache hash
  below that canonical measure precision.
- Numerical artifact: `final_artifact_compare.txt` checks every HDF5 dataset
  finite. The largest old/new difference is
  `5.8594117435327605e-14` eV in `sigma_total_kij_ev`.
- Focused CPU gate:
  `tests/test_delivered_windows.py` — **7 passed** in 43.40 s.

Implementation commits are on `perf/census-fast-2026-08-31`; only the census
half of `src/gw/mpa/delivered_windows.py` changed.
