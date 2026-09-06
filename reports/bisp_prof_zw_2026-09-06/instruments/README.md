# Run-local profiling instruments

These are snapshots of instruments executed from `runs/DEV/112_bisp_prof_zw_codex_2026-09-06/` or copied into a named profiling run. Several derive the sandbox root from that original directory depth; they are not a standalone installed tool suite. Use each immutable run's runner, manifest, source checksums and payload to reproduce its environment. Do not execute preparation or pinned-source batch scripts from this report directory.

Production changes are only in the three cited source commits. Run-local ablations are experimental harnesses and are not proposed as production abstractions. `common_control_batch.py` deliberately pins the two owned production files only while its guarded controls run and restores the saved source bytes in `finally`.
