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

For an `install_deps.sh` install, `lorrax-postinstall` runs it the way every job
runs (through `lorrax-run`, so it judges the restored image); rerunning
`lorrax-postinstall` skips the kernel store once it is complete. By hand:

```bash
source <prefix>/lorrax-env.sh
srun -N 1 -n 4 --gpus-per-node=4 lorrax-run bash -c 'exec python $LORRAX_ROOT/scripts/perf_check.py'
# an --in-place install, a clone or a module: the tree's own select_gpu.sh
srun -N 1 -n 4 -c 16 --threads-per-core=1 --cpu-bind=threads --gpus-per-node=4 \
    env OMP_NUM_THREADS=14 $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh \
    python $LORRAX_ROOT/scripts/perf_check.py
```

| where | what happens | exit |
|---|---|---|
| login node, or a machine with a scheduler but not inside a job step | static checks only, the open latency there with the warning that login nodes can look fast on a filesystem that is slow on compute nodes, any slow filesystem it finds even there (printed as information), and how to run it: generic advice, the scheduler's commands, and a site line where the site is known | 3 |
| a GPU job step | all six components, PASS or FAIL | 0 / 1 |
| a CPU job step (or `--cpu-only`) | components 1, 2 and 4; the verdict says it is not a GPU certification | 0 / 1 |
| a workstation: no scheduler variables, no scheduler commands on `PATH`, not a login-named host | certifies where it runs (there is no login node to mislead it), one process or the ranks of `mpirun`; the verdict says it is a workstation | 0 / 1 |

A job step is recognised under Slurm (`SLURM_STEP_ID` and a rank), PBS and PBS
Pro (`PBS_JOBID`: the job's shell runs on a compute node), LSF (`LSB_JOBID`
plus a `jsrun`/`mpirun` rank, since a batch script may run on a launch node) and
Flux (`FLUX_JOB_ID` plus `FLUX_TASK_RANK`). Ranks come from Open MPI, Flux,
`jsrun`, PALS, MVAPICH2, Slurm or MPICH's variables, in that order. On a
workstation:

```bash
python scripts/perf_check.py                    # one process, all six components
mpirun -n 2 --map-by slot:PE=3 --bind-to core -x OMP_NUM_THREADS=1 \
    python scripts/perf_check.py --cpu-only     # two ranks
```

`lorrax-postinstall` (written by `install_deps.sh`) builds the NVRTC kernel
store and then runs this check with four ranks. When every rank has reported
and the verdict is a GPU PASS, rank 0 writes the verdict to
`runtime.install_verification_stamp(<GPU model>)`. Until that file exists, the
startup block of every GPU run warns INSTALL NOT VERIFIED. A FAIL removes the
file. Exit 2 means the check itself broke. Every rank writes `rank<N>.json` and rank 0
writes `verdict.json` and `report.txt` under
`$SCRATCH/.cache/lorrax/perfcheck/<job>.<step>/` (or `--out`). The whole check
takes about one minute on four A100s with a fast install, and many minutes on a
slow filesystem; rank 0 prints a line as each stage starts and ends, with the
seconds elapsed.

## What it checks, and the fix for each failure

| component | FAIL when | fix |
|---|---|---|
| 1 filesystem {#filesystem} | warm open()+close() median of ~500 install files, all ranks at once, ≥ 1.0 ms (PASS ≤ 0.5 ms), judged overall and for each group on its own (CUDA headers, mathdx headers, site-packages, source tree, libdevice; a group in the runtime image is named so), so a source tree on a slow mount fails even when the venv is fast; warm `os.walk`+`lstat` of the venv < 5000 entries/s; a read-only tree without `.pyc`; an `LD_LIBRARY_PATH` entry ≥ 1.0 ms per failed lookup (0.5 to 1.0 ms warns; a slow `PATH` entry warns) | depends on the launch: on an `install_deps.sh` install launched without it, launch with `lorrax-run`; under `lorrax-run` on a node without node-local room (the image ran from the shared copy), free `/tmp` or set `LORRAX_IMAGE_ROOTS`; otherwise move the install to a fast filesystem (on Perlmutter, reinstall with `install_deps.sh`). `python -m compileall` at install, or `PYTHONPYCACHEPREFIX` on node-local storage |
| 2 caches | the compile or cubin cache is not writable; its directory costs ≥ 1.0 ms per open or failed lookup (warm, slowest rank; a run looks up every compiled program and NVRTC image there); a second process misses the persistent compile cache; a cubin store without images for this GPU | `export LORRAX_CACHE_ROOT=<fast filesystem>` (or `ISDF_JAX_CACHE_DIR` for the compile cache) and rerun `lorrax-postinstall`; rebuild the store (`lorrax-postinstall`, or `scripts/build_cubin_store.py`) |
| 3 kernels | the FFI CUDA leg has no SASS for this GPU's `sm_XX`; `CUDA_FORCE_PTX_JIT` or `JAX_DISABLE_JIT` set; the NVRTC path builds no cubin | rebuild the FFI for this architecture; unset the variable |
| 4 binding | ranks share CPUs; a thread team larger than the rank's CPUs; fewer than two physical cores | a block of cores per rank and `OMP_NUM_THREADS` = cores − 2, in the launcher's words: `srun -c <n> --threads-per-core=1 --cpu-bind=threads`, `mpirun --map-by slot:PE=<n> --bind-to core`, `mpiexec --depth=<n> --cpu-bind depth` (PALS), `jsrun -c <n> -b packed:<n>`, `flux run -c <n>` ([Perlmutter](../environment/machines/perlmutter.md#cpu)) |
| 5 libraries | an FFI leg refuses to open, its mapped closure differs from its record, two copies of a CUDA library are mapped | rebuild, or fix `LD_LIBRARY_PATH` |
| 6 benchmark | a metric of a GPU model with a reference is more than 15% (and more than its absolute slack) worse: cold compile of 36 fixed programs, the warm re-run, GEMM, FFT, host GEMM. The one NVRTC build is judged by ratio alone: WARN above 1.5x the reference, FAIL above 2.5x, and the line says whether its headers came from the node-local copy (then the node's CPU is the cause) or from the install's filesystem | follow the other components' diagnoses |

A thread team larger than the physical cores but within the logical CPUs (the
SMT siblings a mask without `--threads-per-core=1` contains) is a warning.

The benchmark compiles with LORRAX's own XLA flags, as every driver's startup
sets them (`runtime.set_default_env`: the GPU leg's `xla_flag_defaults` in
`lxkit.LEGS`, on CUDA `--xla_gpu_autotune_level=0`, and the rematerialization
pass off), appended to the launch's `XLA_FLAGS`; a flag the launch already
sets wins. Component 6 prints the effective value. Without them, XLA's GEMM
autotuner spent 310 s on the first f64 8192² GEMM of a laptop GPU (4.1 s at
level 0).

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

The NVRTC build is one CPU thread compiling, so it moves with the node rather
than the install: four fast installs on one 80 GB node (after its kernel-store
build) took 7.0–7.3 s, 1.20–1.25x the reference from another node, while the
module's builds, which read about 3200 header files from DVS each, were 2.8x and
5.6x. Hence the bands: 1.5x warns, 2.5x fails. Each rank builds the probe kernel
into its own temporary image cache (deleted afterwards), so every run measures
it and none leaves it in your cache.

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
