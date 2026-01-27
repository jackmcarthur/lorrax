## Cluster config (“clone-and-go”)

The goal is: after cloning the repo, you fill out **one local config file** and then submit jobs without editing the `.sbatch` scripts.

### 1) Create your local config

From the repo root:

```bash
cp cluster_shifter/config.example cluster_shifter/config
```

Edit `cluster_shifter/config` and set:

- `ISDF_SLURM_ACCOUNT`
- `ISDF_CODE_HOST_PATH`
- `ISDF_VENV_HOST_PATH`
- `ISDF_DEFAULT_INPUT_BASENAME` (optional, default is fine)

`cluster_shifter/config` is ignored by git automatically.

### 2) Submit

Single node:

```bash
bash cluster_shifter/submit_singlenode.sh /pscratch/.../your_run_dir/cohsex_test.in
```

Multi node:

```bash
bash cluster_shifter/submit_multinode.sh /pscratch/.../your_run_dir/cohsex_test.in
```

If you omit the input argument, the submit script will:

- mount your current working directory as `/workspace/run`
- use `ISDF_DEFAULT_INPUT_BASENAME` as the input filename inside that directory

### Notes

- **Why wrappers?** Slurm directives like `#SBATCH --account` and `#SBATCH --volume` are
  parsed before the script runs, so we can’t read a config file inside the sbatch script
  to populate those. The wrappers solve that by passing `--account/--volume` on the
  `sbatch` command line.
- **Venv reuse:** the job creates/reuses a venv under:
  `${ISDF_VENV_HOST_PATH}/isdf_cohsex_py311` (mounted as `/workspace/venvroot/isdf_cohsex_py311`)
- **Output directory:** `cohsex_jax.py` writes `tmp/` under the **input file directory**,
  so put the input file in a writable run directory (recommended: `$PSCRATCH`).

### Interactive workflow (dedicated nodes)

This is handy for debugging and “manual runs” where you want to run multiple times without resubmitting jobs.

1) Get an interactive allocation (example: 1 node / 4 GPUs):

```bash
salloc -N 1 -C gpu -G 4 -q interactive -t 01:00:00 -A <YOUR_ACCOUNT>
```

2) Once allocated, run a one-time bootstrap to build/reuse the venv under your venv root (PSCRATCH):

```bash
# Example host paths (replace):
export CODE_HOST="$HOME/software/isdf_cohsex"
export VENV_HOST="$PSCRATCH/isdf_venvs"
mkdir -p "$VENV_HOST"

# Choose a run directory (recommended on PSCRATCH) and put your input file there.
export RUN_HOST="$PSCRATCH/isdf_tmp/run1"
mkdir -p "$RUN_HOST"

# Start shifter shell on one rank just to build the venv.
srun -n 1 --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,nccl-plugin \
  --volume="$CODE_HOST:/workspace/ISDF" \
  --volume="$VENV_HOST:/workspace/venvroot" \
  --volume="$RUN_HOST:/workspace/run" \
  shifter bash -lc "bash /workspace/ISDF/cluster_shifter/bootstrap_venv.sh"
```

3) Run your job with 1 process per GPU (example: 4 ranks), reading input from the run dir:

```bash
srun -n 4 --gpus-per-task=1 --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,nccl-plugin \
  --volume="$CODE_HOST:/workspace/ISDF" \
  --volume="$VENV_HOST:/workspace/venvroot" \
  --volume="$RUN_HOST:/workspace/run" \
  shifter bash -lc "
    source /workspace/venvroot/isdf_cohsex_py311/bin/activate
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    export HDF5_USE_FILE_LOCKING=FALSE
    python3 /workspace/ISDF/src/isdf/gw_isdf/cohsex_jax.py -i /workspace/run/<your_input>.in
  "
```

4) Multi-node interactive is the same idea; just request `-N <nodes>` and set JAX coordinator env vars (same as in `perlmutter_shifter_multinode.sbatch`).


