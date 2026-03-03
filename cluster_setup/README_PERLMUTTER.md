# ISDF on Perlmutter (NERSC)

This folder provides two supported workflows to run the JAX-based ISDF code on Perlmutter GPUs:

- UV + venv on the system nodes (simple, flexible for development)
- Shifter + NVIDIA JAX image (recommended for larger scale and faster cold starts)

## Prerequisites

- Request a GPU interactive session or submit via Slurm; avoid heavy work on login nodes.
- Know your project paths for code and scratch output.

## Workflow A: UV + Virtualenv (development-friendly)

1) Bootstrap UV and the project venv
```bash
bash scripts/perlmutter/uv_setup_perlmutter.sh
```
- By default the venv is created at `$SCRATCH/isdf-venv`.
- To place under /global/common/software (recommended for teams), set:
```bash
export PM_SOFTWARE_PREFIX=/global/common/software/<proj>/<subdir>
bash scripts/perlmutter/uv_setup_perlmutter.sh
```

2) Activate and sync dependencies
```bash
source $PM_SOFTWARE_PREFIX/isdf-venv/bin/activate
uv sync --frozen
```

3) Install JAX GPU wheels matching Perlmutter (CUDA 12)
```bash
uv pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4) Submit a job using the venv
```bash
sbatch scripts/perlmutter/slurm_uv_venv.sbatch
```
- The script enables recommended env vars:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`
  - `HDF5_USE_FILE_LOCKING=FALSE`

## Workflow B: Shifter + NVIDIA JAX image (recommended at scale)

1) Use the provided Slurm script (edit volume mounts as needed)
```bash
sbatch scripts/perlmutter/slurm_shifter_jax.sbatch
```
- Image: `nvcr.io/nvidia/jax:25.04-py3` (NGC)
- Shifter modules: `--module=gpu,nccl-plugin`
- Example code invocation is provided; adjust paths for your repository and inputs.

### Volume mounts: choosing the right paths

- If your repository is staged under Community File System for team use, set the volume like:
```bash
#SBATCH --volume="/global/common/software/<proj>/<subdir>/ISDF:/workspace/ISDF" \
  --volume="/pscratch/sd/<u>/<user>/isdf:/workspace/output"
```
  - Replace `<proj>` and `<subdir>` with your allocation and desired subdirectory.
  - Ensure `/pscratch/sd/<u>/<user>/isdf` exists (create it if needed).
- If your code currently lives in your home/scratch (e.g., `/global/homes/j/jackm/scratchperl/ISDF_test/isdf_cohsex`), you can test with:
```bash
#SBATCH --volume="/global/homes/j/jackm/scratchperl/ISDF_test/isdf_cohsex:/workspace/ISDF" \
  --volume="/pscratch/sd/j/jackm/isdf:/workspace/output"
```

### Why Shifter?
- Faster library load times and better cold-start performance at node counts > O(10).
- Isolation from host software drift; easier to reproduce environments.

## File system layout & I/O guidance

- Place code under Community File System: `/global/cfs/cdirs/<proj>/ISDF` (bind-mount for Shifter).
- Write outputs and temporary files to `$PSCRATCH` (fast, parallel Lustre).
- `$HOME` and `/global/common/software` are not writable from compute nodes; use them for read-only software stacks, not for runtime outputs.
- For HDF5 on non-$SCRATCH paths, set `HDF5_USE_FILE_LOCKING=FALSE`.

## Multi-GPU / Multi-node notes

- The code already constructs a 2D device mesh (x,y). Ensure you request enough GPUs per node and set `--gpus-per-node` appropriately.
- For runs spanning nodes, prefer the Shifter workflow. It tends to reduce startup overhead and improves MPI/NCCL behavior when properly configured.

## Interactive GPU session

Use an interactive allocation, then run with either Shifter or your UV venv.

### Shifter (interactive)

1) Request a GPU node (example: 1 node, 4 GPUs, 30 minutes):
```bash
salloc --nodes 1 --qos interactive -t 00:30:00 --constraint gpu --gpus 4 --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,nccl-plugin --account m4598 --volume="/global/homes/j/jackm/scratchperl/ISDF_test/isdf_cohsex:/workspace/ISDF" --volume="/pscratch/sd/j/jackm/isdf:/workspace/output"
```
(note that pscratch/sd/j/jackm/isdf must already be created)
2) Inside the allocation, run the code (adjust input path as needed):
```bash
srun -n 4 shifter python3 /workspace/ISDF/src/gw_isdf/gw_jax.py -i /workspace/ISDF/examples/cohsex_test/cohsex_test.in
```

- To use a team code location under `/global/common/software`, swap the first path in `--volume` accordingly.

### UV venv (interactive)

1) If not already created, bootstrap UV and the venv:
```bash
bash scripts/perlmutter/uv_setup_perlmutter.sh
```
2) Request a GPU node:
```bash
salloc -N 1 -C gpu -G 4 -q debug -t 00:30:00
```
3) Inside the allocation, activate and run:
```bash
source ${PM_SOFTWARE_PREFIX:-${SCRATCH:-$HOME}}/isdf-venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HDF5_USE_FILE_LOCKING=FALSE
srun -n 4 python3 src/gw_isdf/gw_jax.py -i examples/cohsex_test/cohsex_test.in
```

## Custom NVIDIA JAX-based image with uv

You can build a custom image that layers uv + your package on top of NVIDIA's JAX image. A sample `Dockerfile` is provided at the repo root under `ISDF_test/isdf_cohsex/Dockerfile`.

- This Dockerfile:
  - Uses `nvcr.io/nvidia/jax:25.04-py3` as base (CUDA 12, Python 3.12)
  - Installs `uv` and syncs project deps (excluding JAX which comes from base)
  - Builds MPICH and HDF5 (MPI-enabled) from source
  - Installs `mpi4py` and `h5py` with MPI support inside the uv venv
  - Installs your project in editable mode

Build and push (from repo root):
```bash
docker build -t docker.io/<user>/isdf-jax:pm .
docker push docker.io/<user>/isdf-jax:pm
```
Run on Perlmutter with Shifter:
```bash
salloc --nodes 1 --qos interactive -t 00:30:00 --constraint gpu --gpus 4 \
  --image=docker:<user>/isdf-jax:pm --module=gpu,nccl-plugin --account <acct>
# Use mpirun/srun as usual; mpich libs inside image will be swapped by Shifter
srun -n 4 shifter python3 -m gw_isdf.gw_jax -i /global/.../cohsex_test.in
```
Notes:
- Keep `--module=gpu` (and `nccl-plugin` if needed). Shifter swaps Cray MPICH at runtime for ABI-compatible MPI.
- For filesystem writes outside `$SCRATCH`, set `HDF5_USE_FILE_LOCKING=FALSE` (already default in image).

## Troubleshooting

- JAX doesn’t see GPUs:
  - Ensure you’re on a GPU node (`--constraint=gpu`, `--gpus-per-node`), and run inside srun.
  - With UV: verify JAX GPU wheels (`jaxlib` from CUDA12 URL) are installed in the active venv.
- OOM or fragmentation:
  - Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and tune `XLA_PYTHON_CLIENT_MEM_FRACTION`.
- HDF5 writes fail:
  - Export `HDF5_USE_FILE_LOCKING=FALSE` when writing to non-$SCRATCH filesystems.
- Shifter volume errors (e.g., failed to lstat /var/udiMount/...):
  - Use one `--volume` per mount.
  - Ensure all host paths exist on the compute node and use absolute paths.
  - Sanity-check mounts:
```bash
srun -n 1 shifter bash -lc 'ls -l /workspace/ISDF | head'
srun -n 1 shifter bash -lc 'touch /workspace/output/.write_test && ls -l /workspace/output/.write_test'
```

## Updating environments

- UV venv: re-run `uv sync` after dependency changes; the lockfile keeps versions consistent.
- Shifter image: if you need extra packages, build your own Docker image and push to a registry, then update `--image=...` in the Slurm script.

## Example commands

- Quick device check (inside allocation):
```bash
python -c "import jax; print(jax.devices()); import jax.numpy as jnp; print(jnp.ones((1,)).device())"
```

- Run test input (UV path):
```bash
srun python3 src/gw_isdf/gw_jax.py -i examples/cohsex_test/cohsex_test.in
```

## Best practices summary
- Use Shifter at larger scales; use UV venv for rapid iteration.
- Keep code on `/global/cfs/...`, outputs on `$PSCRATCH`.
- Use the CUDA12 JAX wheels; avoid mixing host CUDA with JAX’s bundled runtimes.
- Set recommended JAX env vars to manage GPU memory behavior. 