# BISP Sigma receipt analysis

Use the sandbox `tools/hlo/analyze_hlo_dump.py` and PERF2
`tools/profile_collective_census.py` for optimized HLO and native aggregate ranges.
Use the run-local `prof_s/00_tools/analyze_boundaries.py RUN` for the JSONL
boundary instrument. Its outer stage includes its children; do not sum them.

For a single warm occurrence beyond the native aggregate CSV, run
`prof_s/00_tools/extract_nsys_unit.py RUN PROGRAM --occurrence 1` against
Nsight's read-only SQLite export. It joins CUDA launch correlation IDs within
that module's NVTX thread/time scope and refuses an absent occurrence or zero
CUDA records. Compare its call count and durations against native CSV aggregates.
A graph without correlated node records needs native Nsight graph attribution;
never treat zero rows as zero execution. Kernel-duration sums, projected GPU
spans and host walls are different quantities; nested thunk spans overlap.

Science identity remains owned by `tools/eqp_ab.py --tol-uev 0` and the printed
Sigma rows; profiler analysis never substitutes for the P4 identity gate.

The canonical analyzer omits collective-permute-start. Run
prof_s/00_tools/census_async_starts.py RUN as a supplement, retaining both
outputs. It counts optimized synchronous operations or async starts once and
excludes done instructions. Native SendRecv is an independent cross-check.
Use prof_s/00_tools/gate_candidate.py BASE CANDIDATE to invoke the canonical
EQP parser and compare complete printed Sigma rows. Certificate-replay arms
are gated against the same selected schedule, not an original alternate rule.
