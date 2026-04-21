# Environment Setup & Configuration

**For AI agents**: This document covers dependencies, installation, JAX configuration, cluster usage, and troubleshooting. Read this when working on build/deployment issues. For code structure, see [`CODEBASE_COMPREHENSIVE.md`](CODEBASE_COMPREHENSIVE.md).

---

## Table of Contents

1. [Dependencies](#1-dependencies)
2. [Installation Methods](#2-installation-methods)
3. [JAX Configuration](#3-jax-configuration)
4. [Cluster Usage](#4-cluster-usage)
5. [Multi-Host/Multi-GPU Setup](#5-multi-hostmulti-gpu-setup)
6. [CUDA Compatibility](#6-cuda-compatibility)
7. [Build from Source](#7-build-from-source)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Dependencies

### 1.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **Python** | ≥3.12 | Language runtime |
| **JAX** | ≥0.9.0 | Array operations, auto-differentiation, GPU acceleration |
| **jaxlib** | ≥0.9.0 | JAX backend (CPU/GPU kernels) |
| **NumPy** | ≥2.3.1 | Array operations, I/O |
| **SciPy** | ≥1.16.0 | Linear algebra, FFTs |
| **h5py** | ≥3.14.0 | HDF5 file I/O |
| **cupy-cuda13x** | ≥13.6.0 | GPU arrays (phasing out for JAX) |

### 1.2 Optional Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| **jax-finufft** | Non-uniform FFT (NUFFT) | CPU-only in uv, use conda-forge for GPU |
| **finufft** | NUFFT backend | Dependency of jax-finufft |
| **matplotlib** | Plotting | Development |
| **pytest** | Unit tests | Development |

### 1.3 Build Dependencies

Only needed when building `jax-finufft`/`finufft` from source:

| Package | Version | Purpose |
|---------|---------|---------|
| **scikit-build-core** | ≥0.11.0 | CMake-based Python builds |
| **cmake** | ≥3.26 | Build system |
| **ninja** | ≥1.10 | Build backend |
| **pybind11** | ≥3.0.0 | Python-C++ bindings |

---

## 2. Installation Methods

### 2.1 Quick Start: uv (Recommended for Development)

**Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create environment and install dependencies**:
```bash
cd /path/to/lorrax
uv venv
uv sync --no-install-project --locked
```

**Activate environment**:
```bash
source .venv/bin/activate
```

**Run tools**:
```bash
uv run python -m gw.gw_jax -i cohsex.in
# or use console commands directly:
uv run gw_jax -i cohsex.in
```

**Note**: uv builds `jax-finufft` from source with **CPU-only** by default (see pyproject.toml lines 46-50). For GPU support, use conda-forge (§2.2).

---

### 2.2 conda-forge (GPU NUFFT Support)

**When to use**: If you need GPU-accelerated NUFFT (`jax-finufft` with cufinufft)

**Installation**:
```bash
conda create -n lorrax python=3.12
conda activate lorrax
conda install -c conda-forge jax-finufft cufinufft jax
pip install h5py scipy matplotlib
pip install -e .  # Install isdf package
```

**Why conda-forge?**
- Pre-built binaries for CUDA 12/13
- Avoids C++17/thrust compatibility issues (see §6)
- Much faster than building from source

---

### 2.3 Docker/Shifter (HPC Clusters)

**NERSC Perlmutter** (recommended for production):
See [`config/README.md`](../config/README.md)

**Dockerfiles** (in repo root):

| File | Purpose | Notes |
|------|---------|-------|
| `Dockerfile.gpu` | Single-GPU dev image (NVIDIA JAX base) | Venv at `/opt/venv` to avoid bind-mount shadowing |
| `Dockerfile.cpu` | CPU-only dev image | `jax[cpu]`, FFTW, project deps via uv |
| `Dockerfile` | Runtime with MPI + parallel HDF5 | MPICH/HDF5 built from source; `.venv` inside repo |
| `Dockerfile.multigpu` | Multi-GPU runs | Like `Dockerfile` but targeted at multi-GPU |

All Dockerfiles use `uv sync --frozen` for reproducibility. GPU images use `nvcr.io/nvidia/jax` base.

**docker-compose** (`docker-compose.gpu.yaml`):
```bash
docker build -t isdf-gpu -f Dockerfile.gpu .
docker compose -f docker-compose.gpu.yaml up -d
docker compose -f docker-compose.gpu.yaml exec isdf bash
# inside container:
python -c 'import jax; print(jax.devices())'
```

The compose service bind-mounts `.:/workspace/ISDF`, mounts CUDA toolkit read-only,
requests all GPUs, and runs `sleep infinity` for `exec` access.

**Bind-mount model**: The container provides the toolchain (CUDA, NCCL, MPI, HDF5,
Python deps). Code is bind-mounted to `/workspace/ISDF` — edits are immediate;
rebuild only when dependencies or system libs change.

---

## 3. JAX Configuration

### 3.1 Environment Variables

JAX behavior is controlled by environment variables set **before importing JAX**.

**Key variables** (set in `gw_jax.py` lines 9-13):

```bash
# Enable 64-bit precision (required for GW accuracy)
export JAX_ENABLE_X64=1

# Device priority: try CUDA first, fallback to CPU
export JAX_PLATFORMS="cuda,cpu"

# Disable memory pre-allocation (allows multiple processes per GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Use platform allocator (more efficient than BFC)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# CUDA async allocator (reduces fragmentation)
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

**For multi-host CPU testing** (no GPUs):
```bash
# Create 4 fake CPU "devices" for sharding tests
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
```

---

### 3.2 Device Selection

**Auto-detect** (default):
```python
import jax
devices = jax.devices()  # Returns all available GPUs or CPUs
```

**Force CPU**:
```python
import jax
jax.config.update("jax_platform_name", "cpu")
devices = jax.devices("cpu")
```

**Select specific GPUs**:
```bash
# Use only GPU 2 and 3
export CUDA_VISIBLE_DEVICES=2,3
python -m gw.gw_jax -i cohsex.in
```

---

### 3.3 Memory Management

**LORRAX default** (set by `module load lorrax`):
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

This makes XLA pull from the CUDA asynchronous mempool on demand
(`cudaMallocAsync`) rather than pre-allocating a fixed BFC pool. The
important consequence: **NCCL, cuSOLVERMp, and SLATE share one pool
with XLA**, so none of them can starve. An older default of
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` with BFC was observed to surface
as `NCCL error 1 unhandled cuda error` → `cusolverMpSyevd status=7`
on the Cray MPICH stack; `platform` + `PREALLOCATE=false` fixes it.

**If you need a hard cap** (e.g. to leave room for another process on
the same GPU), you can still use `XLA_PYTHON_CLIENT_MEM_FRACTION`, but
only with `XLA_PYTHON_CLIENT_ALLOCATOR=default` (BFC) — the platform
allocator ignores it.

**Monitor memory usage**:
```python
import jax
print(jax.local_devices()[0].memory_stats())
```

---

## 4. Cluster Usage

### 4.1 NERSC Perlmutter

LORRAX ships an Lmod module that wires up the Shifter container, Cray
MPICH, GPU affinity, and FFI bind-mounts so you never hand-write
`shifter --image=...` / `srun --mpi=...` flags. The full reference is
[`config/README.md`](../config/README.md); minimal flow:

```bash
# One-time: install the module (patches site-specific values via
# config/perlmutter/site_config.sh)
bash config/perlmutter/install.sh

# Every session: allocate + load + run
module load lorrax
lxalloc                                      # 1 node / 4 GPUs / 2 hours
lxrun python3 -u -m gw.gw_jax -i cohsex.in   # 4 GPUs, Cray MPICH stack
lxshell                                      # interactive container shell

# Single-GPU override:
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```

**Stack**: `--mpi=cray_shasta`, `--module=gpu,mpich`, one GPU per rank
via `select_gpu.sh` (`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`),
`LD_PRELOAD=libmpi_gtl_cuda.so.0` (CUDA-12 copy). SLATE FFI, cuSOLVERMp
FFI, and phdf5 (Cray HDF5) all share this single stack.

**Batch job** — the module's shell functions work inside `#SBATCH`
scripts; just `module load lorrax` at the top and use `lxrun`:

```bash
#!/bin/bash -l
#SBATCH -N 2 -C gpu -q regular -t 04:00:00 -A <account>
#SBATCH --ntasks-per-node=4 --gpus-per-node=4

module load lorrax
lxrun python3 -u -m gw.gw_jax -i cohsex.in
```

A template lives at [`config/perlmutter/run_gw.slurm`](../config/perlmutter/run_gw.slurm).

**Container**: `nvcr.io/nvidia/jax:25.04-py3`. Includes JAX + CUDA 12,
NumPy. Additional deps (h5py, scipy, matplotlib) are bind-mounted from
the per-user `LORRAX_SITE_PACKAGES` directory (see `config/README.md`).

---

### 4.2 Generic SLURM Cluster

**Single-node, 4 GPUs**:
```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -t 02:00:00

module load cuda/12.3 python/3.12
source /path/to/venv/bin/activate

export JAX_ENABLE_X64=1
export JAX_PLATFORMS="cuda,cpu"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m gw.gw_jax -i cohsex.in
```

---

## 5. Multi-Host/Multi-GPU Setup

### 5.1 JAX Distributed Initialization

**Automatic** (SLURM environment variables):
```python
import jax
jax.distributed.initialize()  # Auto-detects SLURM_PROCID, SLURM_NTASKS, etc.
```

**Manual** (non-SLURM):
```python
import jax
jax.distributed.initialize(
    coordinator_address="node001:12355",  # First node, arbitrary port
    num_processes=4,                      # Total MPI ranks
    process_id=0                          # This rank (0, 1, 2, 3)
)
```

**Environment variables** (if auto-detect fails):
```bash
export JAX_COORDINATOR_ADDRESS="node001:12355"
export JAX_NUM_PROCESSES=4
export JAX_PROCESS_INDEX=0  # Different for each rank
```

---

### 5.2 Multi-Node Execution

**2 nodes × 4 GPUs = 8 processes** via the LORRAX module:

```bash
#!/bin/bash -l
#SBATCH -N 2 -C gpu -q regular -t 04:00:00 -A <account>
#SBATCH --ntasks-per-node=4 --gpus-per-node=4

module load lorrax
LORRAX_NNODES=2 LORRAX_NGPU=8 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```

**GPU affinity caveat.** LORRAX deliberately avoids `--gpus-per-task=1`
and instead pins GPUs via `select_gpu.sh`
(`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`). `--gpus-per-task=1` breaks
JAX's distributed topology sync: each rank's `local_devices` call uses
narrow-view device ordinals but JAX's coordinator expects all ranks to
report from the same global ordinal space. With `select_gpu.sh` each
rank sees exactly one GPU as device 0, and callers pass
`local_device_ids=[0]` to `jax.distributed.initialize()` (auto-detected
via `len(CUDA_VISIBLE_DEVICES.split(","))==1` in the sandbox tests).

**JAX will**:
1. Auto-detect `SLURM_NTASKS=8`, `SLURM_PROCID=0..7` via Cray PMI.
2. Initialize the distributed backend (each rank contributes its
   one `local_device_ids=[0]`).
3. Shard arrays across all 8 GPUs.

---

### 5.3 Debugging Multi-Host

**Check JAX sees all devices**:
```python
import jax
print(f"Process {jax.process_index()}/{jax.process_count()}")
print(f"Devices: {jax.local_devices()}")
print(f"Global devices: {jax.devices()}")
```

**Expected output** (node 0, 4 GPUs):
```
Process 0/8
Devices: [cuda:0, cuda:1, cuda:2, cuda:3]
Global devices: [cuda:0, cuda:1, ..., cuda:7]  # All 8 GPUs across 2 nodes
```

**See also**: [`docs/advanced/jax_multihost.md`](advanced/jax_multihost.md) for deep dive

---

## 6. CUDA Compatibility

### 6.1 CUDA Version Requirements

| Component | CUDA 12.x | CUDA 13.x | Notes |
|-----------|-----------|-----------|-------|
| **JAX/jaxlib** | ✅ Supported | ✅ Supported | Use `jax[cuda12]` or `jax[cuda13]` |
| **cupy** | ✅ | ✅ | Use `cupy-cuda12x` or `cupy-cuda13x` |
| **jax-finufft (pip)** | ⚠️ CPU-only | ⚠️ CPU-only | GPU build fails (see §6.2) |
| **jax-finufft (conda)** | ✅ GPU | ✅ GPU | Pre-built binaries available |

---

### 6.2 jax-finufft GPU Build Issues (CUDA 13.0)

**Problem**: Building `jax-finufft` with GPU support fails on CUDA 13.0 due to:

1. **Missing CUFFT error codes** (`helper_cuda.h`):
   ```c
   case CUFFT_INCOMPLETE_PARAMETER_LIST:  // Doesn't exist in CUDA 13
   case CUFFT_PARSE_ERROR:                // Doesn't exist in CUDA 13
   case CUFFT_LICENSE_ERROR:              // Doesn't exist in CUDA 13
   ```

2. **thrust::binary_function removed**:
   ```cpp
   struct cmp : public thrust::binary_function<int, int, bool>  // Removed in C++17
   ```
   CUDA 13.0 defaults to C++17, but thrust removed deprecated templates.

**Workarounds**:
- **Option A**: Use conda-forge pre-built binaries (recommended)
- **Option B**: Use CPU-only NUFFT (default in pyproject.toml)
- **Option C**: Standard FFT (omit `q_grid` parameter)

**See**: [`JAX_FINUFFT_USAGE.md`](../JAX_FINUFFT_USAGE.md) for detailed build notes

---

### 6.3 Current pyproject.toml Configuration

**Lines 46-50** (CPU-only NUFFT):
```toml
[tool.uv.extra-build-variables.jax-finufft]
CMAKE_CUDA_COMPILER = "/usr/local/cuda/bin/nvcc"
CUDACXX             = "/usr/local/cuda/bin/nvcc"
CUDAHOME            = "/usr/local/cuda"
CMAKE_ARGS          = "-DJAX_FINUFFT_USE_CUDA=OFF -DCMAKE_CUDA_ARCHITECTURES=native"
```

**To enable GPU** (if you patch finufft):
```toml
CMAKE_ARGS = "-DJAX_FINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
```

---

## 7. Build from Source

### 7.1 Build jax-finufft (CPU-only)

**Already configured** in `pyproject.toml`. Build happens automatically:
```bash
uv sync
```

**Manual build** (if needed):
```bash
git clone https://github.com/flatironinstitute/jax-finufft
cd jax-finufft
git submodule update --init --recursive
pip install scikit-build-core cmake ninja
CMAKE_ARGS="-DJAX_FINUFFT_USE_CUDA=OFF" pip install -e .
```

---

### 7.2 Build finufft (GPU)

**Prerequisites**:
- CUDA Toolkit 12.x (13.x has compatibility issues)
- CMake ≥3.26
- ninja

**Steps**:
```bash
git clone https://github.com/flatironinstitute/finufft
cd finufft
git checkout v2.2.0  # Latest stable

mkdir build && cd build
cmake .. \
  -DFINUFFT_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70  # V100=70, A100=80, H100=90 \
  -GNinja

ninja
ninja install  # Installs to /usr/local by default
```

**Then install jax-finufft**:
```bash
CMAKE_ARGS="-DJAX_FINUFFT_USE_CUDA=ON" pip install jax-finufft
```

**Note**: This likely fails on CUDA 13.0. Use conda-forge instead.

---

## 8. Troubleshooting

### 8.1 Common Errors

#### "No GPU/TPU found"
```python
RuntimeError: No GPU/TPU found, falling back to CPU.
```

**Solutions**:
1. Check CUDA installation: `nvidia-smi`
2. Install `jax[cuda12]` or `jax[cuda13]`: `pip install 'jax[cuda13]'`
3. Check `CUDA_VISIBLE_DEVICES`: `echo $CUDA_VISIBLE_DEVICES`
4. Verify jaxlib has CUDA support: `python -c "import jax; print(jax.default_backend())"`

---

#### "jaxlib version mismatch"
```
RuntimeError: jaxlib version 0.4.20 is newer than supported jax version 0.4.19
```

**Solution**: Update both to matching versions
```bash
pip install --upgrade "jax[cuda13]" jaxlib
```

---

#### "Out of memory" (GPU)
```
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

**Solutions**:
1. Reduce chunk sizes in `cohsex.in`:
   ```
   chunk_bands = 4   # Reduce from 6
   chunk_q = 2       # Reduce from 3
   ```
2. Disable pre-allocation: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`
3. Use memory profiler: See [`MEMORY_MODEL.md`](MEMORY_MODEL.md)

---

#### "Import error: libcufft.so.11"
```
ImportError: libcufft.so.11: cannot open shared object file
```

**Solution**: CUDA libraries not in `LD_LIBRARY_PATH`
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Or install matching CUDA toolkit:
```bash
# For CUDA 12
pip install --upgrade "jax[cuda12_local]"
```

---

#### "finufft not found" (import error)
```
ModuleNotFoundError: No module named 'finufft'
```

**Solution**: Install finufft
```bash
# Option A: conda-forge
conda install -c conda-forge finufft

# Option B: pip (CPU-only)
pip install finufft

# Option C: Build from source (see §7.2)
```

---

### 8.2 Performance Debugging

#### Slow NUFFT performance
- **Expected**: CPU NUFFT is ~100× slower than GPU FFT
- **Solution**: Use conda-forge GPU build or standard FFT

#### Slow zeta fitting
- **Check**: `zeta_q.h5` size (10-100 GB causes disk I/O bottleneck)
- **Solution**: Use faster filesystem (NVMe, not NFS) or reduce `n_centroids`

#### GPU underutilization
- **Check**: `nvidia-smi dmon` during run
- **Common cause**: Small system size, GPU overhead dominates
- **Solution**: Use CPU for small systems (<50 bands, <10×10×1 k-grid)

---

### 8.3 Debugging Tools

**JAX debugging flags**:
```bash
export JAX_DEBUG_NANS=1         # Crash on NaN
export JAX_DISABLE_JIT=1        # Disable JIT for debugging
export JAX_LOG_COMPILES=1       # Log all compilations
export TF_CPP_MIN_LOG_LEVEL=0   # Verbose XLA logs
```

**Profile memory**:
```python
import jax
jax.profiler.start_trace("/tmp/jax_trace")
# ... run code ...
jax.profiler.stop_trace()
# View in chrome://tracing or TensorBoard
```

**See also**:
- [`docs/MEMORY_MODEL.md`](MEMORY_MODEL.md) — Memory budgets
- [`docs/MEMORY_MODEL.md`](MEMORY_MODEL.md) — Chunking constraints and bottleneck arrays
- [`docs/advanced/jax_multihost.md`](advanced/jax_multihost.md) — Multi-GPU debugging

---

## Next Steps

**For code structure**: See [`CODEBASE_COMPREHENSIVE.md`](CODEBASE_COMPREHENSIVE.md)
**For physics/theory**: See [`PHYSICS_COMPREHENSIVE.md`](PHYSICS_COMPREHENSIVE.md)
**For improvement suggestions**: See [`AGENT_TODO.md`](AGENT_TODO.md)
