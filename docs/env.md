## Environment: Docker + uv

This repo uses one dependency spec (`pyproject.toml` + `uv.lock`) for all environments. You can develop locally with uv, or use the provided Dockerfiles for portable, GPU-ready images that you bind‑mount your source into.

### Dockerfiles

- `Dockerfile.gpu`: Single‑GPU dev image based on NVIDIA JAX. Installs uv deps and puts a virtualenv at `/opt/venv` so it isn’t shadowed by bind mounts. Intended to be used with `docker compose` and a bind mount of the repo to `/workspace/ISDF`.
- `Dockerfile.cpu`: CPU‑only dev image for laptops; installs `jax[cpu]`, FFTW, and project deps via uv.
- `Dockerfile`: Runtime image with MPI + parallel HDF5 (MPICH + HDF5 built from source). Uses uv to sync deps into `.venv` inside the repo. Good for HPC-style runs when you need MPI/HDF5.
- `Dockerfile.multigpu`: Like `Dockerfile` but targeted at multi‑GPU runs. MPICH/HDF5 included; uses `.venv` inside repo.

Notes:
- All Dockerfiles use uv to install Python deps from `uv.lock` (`uv sync --frozen`) for reproducibility.
- The GPU images rely on the base `nvcr.io/nvidia/jax` image for CUDA/JAX wheels compatibility.
- For performance and stable layers, `Dockerfile.gpu` keeps the venv in `/opt/venv` while you bind‑mount the repo to `/workspace/ISDF`.

### docker-compose

- `docker-compose.gpu.yaml` defines a service `isdf`:
  - Bind‑mounts the project: `.:/workspace/ISDF`
  - Mounts CUDA toolkit read‑only
  - Requests all GPUs and sets CUDA env vars
  - Runs `sleep infinity` so you can `docker compose exec` into it

Typical usage:
```bash
docker build -t isdf-gpu -f Dockerfile.gpu .
docker compose -f docker-compose.gpu.yaml up -d
docker compose -f docker-compose.gpu.yaml exec isdf bash
# inside container
python -c 'import jax; print(jax.devices())'
```

### uv environments

- Local dev (WSL/macOS/Linux):
  - `uv sync` to create `.venv`, then `uv run -- python -m pytest -q`
  - This uses the project‑local env; no Docker needed.
- In containers:
  - `Dockerfile.gpu` places the venv at `/opt/venv` to avoid bind‑mount shadowing.
  - If you use uv inside the container and want to target `/opt/venv`, call uv with `--active`, e.g. `uv --active run -- python ...`.

### Difficult dependencies

- `jax-finufft` depends on `cufinufft` for GPU NUFFT. This requires CUDA toolkits, FFTW, and compatible `cufinufft` builds. The current images install FFTW and rely on NVIDIA’s JAX base for CUDA. If you add `jax-finufft`, prefer building it in the image layer (not at runtime) so compiled artifacts are cached.
- MPI + parallel HDF5: the runtime/multigpu Dockerfiles build MPICH and HDF5 with MPI from source and then install `mpi4py` and `h5py` against those. This is needed for cluster/shifter environments where system MPI isn’t present or you want consistent versions.

### Bind‑mount model

- The container provides the toolchain (CUDA, NCCL, MPI, HDF5, Python deps).
- Your code is bind‑mounted to `/workspace/ISDF`. Edits are immediate; rebuilding the image is unnecessary unless you change dependencies or system libs.

## API docs (Markdown only)

- We use `pydoc-markdown` to generate Markdown under `docs/api/` from NumPy‑style docstrings.
- Generate:
```bash
uv add --group dev pydoc-markdown   # once per environment
uv run -- bash scripts/gen_api_docs.sh
```
- Browse the generated `docs/api/index.md` in your editor or on GitHub.
- Style guidance: see `docs/docstrings.md`. Focus docstrings on shapes, units, shardings, and dataflow (especially for JAX sharded arrays and mesh axis semantics).

