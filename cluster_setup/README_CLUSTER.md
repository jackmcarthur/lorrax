# Running COHSEX-JAX on NERSC Perlmutter with Shifter

This directory contains scripts for running `cohsex_jax` on NERSC Perlmutter using the NVIDIA JAX Shifter image.

## Quick Start

### Batch Job

1. Copy the batch script to your run directory:
   ```bash
   cp $HOME/software/isdf_cohsex/cluster_shifter/run_cohsex.slurm /path/to/your/run_dir/
   cd /path/to/your/run_dir/
   ```

2. Edit the script if needed (change `#SBATCH` options, `RUN_DIR`, `INPUT_FILE`)

3. Submit:
   ```bash
   sbatch run_cohsex.slurm
   ```

### Interactive Session

1. Allocate an interactive GPU node:
   ```bash
   salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu \
          --gpus=4 --account=m2651 --image=nvcr.io/nvidia/jax:25.04-py3
   ```

2. Run from your input directory:
   ```bash
   cd /path/to/your/run_dir
   
   # Run cohsex_jax
   srun -n 4 shifter --module=gpu --env=PYTHONPATH=$HOME/software/isdf_cohsex/src \
       python3 -m gw_isdf.gw_jax -i gw.inp
   
   # Or run any other module (e.g., generate centroids)
   srun -n 1 shifter --module=gpu --env=PYTHONPATH=$HOME/software/isdf_cohsex/src \
       python3 -m isdf.isdf_init.kmeans_isdf -i gw.inp
   ```

## Key Points

- **No venv needed**: The NVIDIA JAX container has all dependencies pre-installed
- **No volume mounts needed**: NERSC auto-mounts `$HOME` and `$PSCRATCH` in Shifter
- **`--module=gpu` is required**: This enables GPU access inside the container
- **Image**: `nvcr.io/nvidia/jax:25.04-py3` (NVIDIA's official JAX container)

## Running from a Separate Terminal (e.g., Cursor IDE)

If your `salloc` is running in one terminal but you want to run commands from another (like Cursor's integrated terminal), you need the job ID:

```bash
# Find your job ID
squeue -u $USER
# Shows: 48182279 urgent_gp interact ...

# Run with explicit --jobid and --image
srun --jobid=48182279 -n 4 shifter --module=gpu \
    --image=nvcr.io/nvidia/jax:25.04-py3 \
    --env=PYTHONPATH=$HOME/software/isdf_cohsex/src \
    python3 -m gw_isdf.gw_jax -i gw.inp
```

**Important**: When using `--jobid`, you MUST specify `--image=` explicitly (it's not inherited from the allocation).

## Convenience Wrapper

For a simpler interactive experience, add this to your `~/.bashrc`:

```bash
isdfrun() {
    local ntasks="${ISDF_NTASKS:-4}"
    srun -n "$ntasks" shifter --module=gpu \
        --env=PYTHONPATH=$HOME/software/isdf_cohsex/src \
        python3 "$@"
}
```

Then use:
```bash
isdfrun -m gw_isdf.gw_jax -i gw.inp
isdfrun -m isdf.isdf_init.kmeans_isdf -i gw.inp
```

## Multi-Node Jobs

For multi-node jobs, increase `--nodes` and `--ntasks-per-node`:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
```

This gives you 8 GPUs across 2 nodes (4 per node).
