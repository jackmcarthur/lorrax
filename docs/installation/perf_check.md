# Performance self-check

`scripts/perf_check.py` decides whether a run on this install, on this node, with
this launch line, will be as fast as the reference, and names the cause when it
will not. The Perlmutter installer runs it as its last step; run it again
whenever the install, the modules or the launch line change.

Why it exists: a published module (venv and source on `/global/common`) imported
fine on a login node and ran 2x slower on compute nodes. Nothing failed and every
gate passed: on compute nodes `/global/common` is a DVS mount where every
successful `open()` is a synchronous server round trip (3.5 ms against 0.24 ms on
Lustre `$SCRATCH`), and a cold LORRAX process opens about 10^4 install files
(CUDA and mathdx headers for each NVRTC build, libdevice for each XLA module,
every imported Python file).

## Running it

Use the `srun` line your production runs use; the check judges the install,
the node and the launch together.

```bash
source <prefix>/lorrax-env.sh            # or: module load <the LORRAX module>
srun -N 1 -n 4 -c 16 --threads-per-core=1 --cpu-bind=threads --gpus-per-node=4 \
    env OMP_NUM_THREADS=14 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh \
    python $LORRAX_ROOT/scripts/perf_check.py
```

| where | what happens | exit |
|---|---|---|
| login node, or not inside a job step | static checks only, the open latency there with the warning that login nodes look fast on every filesystem, and the `salloc`/`srun` command to run | 3 |
| a GPU job step | all six components, PASS or FAIL | 0 / 1 |
| a CPU job step (or `--cpu-only`) | components 1, 2 and 4; the verdict says it is not a GPU certification | 0 / 1 |

Exit 2 means the check itself broke. Every rank writes `rank<N>.json` and rank 0
writes `verdict.json` and `report.txt` under
`$SCRATCH/.cache/lorrax/perfcheck/<job>.<step>/` (or `--out`). The whole check
takes about two minutes on four A100s.

## What it checks, and the fix for each failure

| component | FAIL when | fix |
|---|---|---|
| 1 filesystem {#filesystem} | warm open()+close() median of ~500 install files, all ranks at once, ≥ 1.0 ms (PASS ≤ 0.5 ms); warm `os.walk`+`lstat` of the venv < 5000 entries/s; a read-only tree without `.pyc` | stage the runtime node-local, or install on a fast filesystem (Lustre `$SCRATCH`); `python -m compileall` at install, or `PYTHONPYCACHEPREFIX` on node-local storage |
| 2 caches | the compile or cubin cache is not writable; a second process misses the persistent compile cache; a cubin store without images for this GPU | point `SCRATCH` or `ISDF_JAX_CACHE_DIR` at a writable fast filesystem; rebuild the store (`scripts/build_cubin_store.py`) |
| 3 kernels | the FFI CUDA leg has no SASS for this GPU's `sm_XX`; `CUDA_FORCE_PTX_JIT` or `JAX_DISABLE_JIT` set; the NVRTC path builds no cubin | rebuild the FFI for this architecture; unset the variable |
| 4 binding | ranks share CPUs; a thread team larger than the rank's CPUs; fewer than two physical cores | `-c <cores per rank> --threads-per-core=1 --cpu-bind=threads`, `OMP_NUM_THREADS` = cores − 2 ([Perlmutter](../environment/machines/perlmutter.md#cpu)) |
| 5 libraries | an FFI leg refuses to open, its mapped closure differs from its record, two copies of a CUDA library are mapped | rebuild, or fix `LD_LIBRARY_PATH` |
| 6 benchmark | a metric of a GPU model with a reference is more than 15% (and more than its absolute slack) worse: cold compile of 36 fixed programs, the warm re-run, one NVRTC build, GEMM, FFT, host GEMM | follow the other components' diagnoses |

A thread team larger than the physical cores but within the logical CPUs (the
SMT siblings a mask without `--threads-per-core=1` contains) is a warning.

## Calibration

References and thresholds are constants at the top of the script. The table
below is what they were set from (Perlmutter, 4 ranks).

(filled from the validation runs)
