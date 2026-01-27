## Shifter + Perlmutter (JAX base image) job scripts

This folder is a **minimal, fast-startup** way to run `isdf_cohsex` on Perlmutter using:

- **Shifter** + a **prebuilt NVIDIA JAX image** (no custom CUDA build),
- a **persistent Python venv on `$PSCRATCH`** to install Python deps like `h5py` (non-MPI),
- **multi-node JAX** via `jax.distributed.initialize()` (your code already supports this).

### What you edit before first use

Use the “clone-and-go” flow in `README_CLUSTER.md`:

- copy `config.example` → `config`
- fill in account + host paths
- submit via the `submit_*.sh` wrappers (no need to edit sbatch files)

### How the Python environment is handled

We **do not bake deps into the container**. Instead we:

- create/reuse a venv under `/workspace/output/venvs/isdf_cohsex_py311`
- install your project with `pip install -e /workspace/ISDF`
- rely on pip wheels for `h5py`, `scipy`, etc.

This keeps Shifter image startup fast. You pay the install cost **once** per venv path, then jobs reuse it.

### Network vs offline installs

If compute nodes have restricted network access, you can pre-stage wheels:

- On a login node, download wheels into a wheelhouse on `$PSCRATCH` or CFS:
  - `python -m pip download -d /pscratch/.../wheelhouse h5py scipy numpy matplotlib xmlschema xsdata mkdocs mkdocs-material mkdocstrings mkdocstrings-python`
- Then set `WHEELHOUSE=/workspace/output/wheelhouse` in the sbatch (see scripts) so installs use:
  - `pip install --no-index --find-links "$WHEELHOUSE" ...`

### Notes on parallel HDF5 (MPI-enabled h5py)

You asked to skip this for now (good call—it's messier).

If/when you want **MPI-enabled h5py**:

- You need **parallel HDF5** + **mpi4py** built against the MPI/HDF5 ABI that Shifter expects
  (commonly done by installing MPICH in-image and letting Shifter swap to Cray MPICH).
- Your older `Dockerfile`/`Dockerfile.multigpu` were doing exactly that.
- Practically: keep the same sbatch scripts here, but replace the base image with your custom
  image that provides `mpi4py` + parallel `h5py`, and then switch your IO patterns to collective writes.


