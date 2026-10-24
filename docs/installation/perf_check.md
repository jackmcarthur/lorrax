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

`lorrax-postinstall` (written by `install_deps.sh`) builds the NVRTC kernel
store and then runs this check with four ranks. When every rank has reported
and the verdict is a GPU PASS, rank 0 writes the verdict to
`runtime.install_verification_stamp(<GPU model>)`. Until that file exists, the
startup block of every GPU run warns INSTALL NOT VERIFIED. A FAIL removes the
file. Exit 2 means the check itself broke. Every rank writes `rank<N>.json` and rank 0
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

References and thresholds are constants at the top of the script. They were set
from the runs below (Perlmutter, 4 ranks, slowest rank, 2026-10-23). "Fast" is an
`install_deps.sh` install on Lustre `$SCRATCH` (JAX 0.12); "module" is the
published `lorrax_A` module on `/global/common` (JAX 0.9.1).

| install, node | warm open ms | venv walk /s | cold XLA compile, 36 programs | NVRTC build | verdict |
|---|---|---|---|---|---|
| fast, GPU node (80 GB A100), n = 3 | 0.015–0.157 | 20,234–20,670 | 7.63–7.97 s (spread 4.4%) | 5.8–6.6 s | PASS |
| fast, GPU node (40 GB A100), n = 3 | 0.015–0.155 | 20,535–20,907 | 7.68–8.02 s | 5.8–7.0 s | PASS; the first run is `lorrax-postinstall`'s, and it wrote the stamp |
| module, GPU node, n = 2 | 3.27–3.93 | 587–988 | 11.8–12.3 s (1.5–1.6x) | 16.5–32.6 s | FAIL: slow filesystem |
| fast, CPU node, n = 5 | 0.015–0.24 | — | — | — | PASS (CPU only) |
| module, CPU node, n = 5 | 1.37–4.6 | — | — | — | FAIL: slow filesystem |

Compute-node open medians from the filesystem survey: `/global/common`
(DVS) 3.5 ms, `$HOME` 10.6 ms, CFS 9.6 ms, `$SCRATCH` 0.24 ms, node-local
`/tmp` 0.005 ms. A login node reads every one of them at about 0.07 ms.

Injected faults on the fast install, GPU node:

| fault | verdict, first diagnosis |
|---|---|
| `OMP_NUM_THREADS=256` | FAIL, 4 binding: a 256-thread OpenMP team on 16 CPUs |
| `--cpu-bind=none` | FAIL, 4 binding: ranks share CPUs |
| `CUDA_FORCE_PTX_JIT=1` | FAIL, 3 kernels: CUDA_FORCE_PTX_JIT is set |
| `ISDF_JAX_CACHE_DIR` on a read-only tree | FAIL, 2 caches: compile cache not writable |
| empty NVRTC image cache | FAIL, 2 caches: cubin store missing, with the command that builds it |
| a `$HOME` directory first on `LD_LIBRARY_PATH` | FAIL, 1 filesystem: 2.4 ms per failed lookup on DVS |
| the docs `srun` line with `OMP_NUM_THREADS` unset | PASS: `select_gpu.sh` cuts each rank's CPU mask to one hardware thread per core and sets `OMP_NUM_THREADS` to the core count minus 2. Without that wrapper (the module), the check warns that the OpenMP team runs on SMT siblings; that cost about 8% of a stage (n = 1) |
