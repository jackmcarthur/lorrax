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
