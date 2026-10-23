# Frontier (OLCF)

Frontier runs LORRAX directly on its MI250X GPUs (`gfx90a`), using a ROCm
Python environment and two native libraries: the ROCm device leg and the host
leg. The Python environment, native builds and launcher must agree; they are
separate parts of the installation. No container or CUDA environment is used.

## 1. Select the Python environment {#environment}

Use the validated Frontier environment with JAX/jaxlib 0.9.1 and the
`rocm-sdk` 7.14.1 wheels. The existing installation used by the site scripts
is `$HOME/envs/lx-jax091-rocm7141`.

From the checkout:

```bash
source config/frontier/gpu_env.sh
```

The helper selects `LORRAX_OLCF_VENV` when set, otherwise the active
`VIRTUAL_ENV`, otherwise the default above. To choose another validated
ROCm environment, set `LORRAX_OLCF_VENV` before sourcing it. It activates that
environment, checks for the ROCm JAX plugin without starting a device client,
supplies the pinned host-library paths and ROCm launch setup, and selects
this checkout's Python source. Runtime adds the matching service packages.

### Fresh Python installation

The following is the Python 3.12 setup used by the validated environment.
Run it from the checkout on a login node. The AMD wheel index supplies the
ROCm libraries and JAX plugin; PyPI supplies the matching JAX frontend and
`jaxlib`. Keep these versions together.

```bash
module load cray-python/3.12.12
python3 -m venv "$HOME/envs/lx-jax091-rocm7141"
source "$HOME/envs/lx-jax091-rocm7141/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'rocm[libraries,device-gfx90a]==7.14.1'
python -m pip install --no-cache-dir \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  --extra-index-url https://pypi.org/simple \
  'jax_rocm7_plugin==0.9.1+rocm7.14.1' \
  'jax_rocm7_pjrt==0.9.1+rocm7.14.1'
python - <<'PYTHON'
from pathlib import Path
import subprocess
import sys
import tomllib

root = Path.cwd()
metadata = tomllib.loads((root / "pyproject.toml").read_text())
packages = sorted({path for pattern in metadata["tool"]["uv"]["workspace"]["members"]
                   for path in root.glob(pattern) if (path / "pyproject.toml").is_file()})
editable = [arg for path in [*packages, root] for arg in ("-e", str(path))]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                       "jax==0.9.1", "jaxlib==0.9.1", *editable, "pytest"])
PYTHON
python -m pip freeze > frontier-python-freeze.txt
source config/frontier/gpu_env.sh
```

The editable package list comes from the checkout's workspace metadata;
there is no second service roster to maintain. The helper and native build
scripts do not create this environment. Retain its package manifest and,
for an offline rebuild, the downloaded wheels with hashes. This recipe pins
the validated GPU stack; it is not a lock for every transitive dependency.

## 2. Build the native pair {#build}

### Existing dependency installs

With the validated environment and dependency installs already present, a
new checkout needs only its own two LORRAX builds:

```bash
bash config/frontier/build_ffi_host.sh
bash config/frontier/build_ffi_rocm.sh
```

The scripts select the pinned compiler/MPI/HDF5 stack themselves and use
`LORRAX_OLCF_VENV` for the JAX FFI headers. They write:

| Native leg | File | Provides |
|---|---|---|
| host | `src/ffi/cpp/build_host/liblorrax_ffi_host.so` | Host MPI/HDF5 operations, CPU linear algebra and FFTW |
| ROCm | `src/ffi/cpp/build_rocm/liblorrax_ffi_rocm.so` | GPU operations through HIP, rocBLAS/rocSOLVER, SLATE/ELPA and the ROCm FFT kernels |

A GPU process loads both. Each build also writes the library's
`<leg>.closure.json` and `PROVENANCE` records.

### First installation of the shared dependencies

Only when these dependency installs are absent or need replacing, build them
before the two commands above:

```bash
bash config/frontier/build_openblas_seq.sh
bash config/frontier/build_slate_hip.sh
bash config/frontier/build_slate_cpu.sh
```

| Shared dependency | Default prefix |
|---|---|
| Sequential, locking OpenBLAS | `$HOME/software/slate_builds/openblas/install` |
| SLATE with `gpu_backend=hip` | `$HOME/software/slate_builds/hip/install` |
| SLATE with `gpu_backend=none` | `$HOME/software/slate_builds/cpu/install` |
| ELPA | A content-keyed prefix under `$HOME/software/elpa_builds` |

`build_ffi_rocm.sh` builds ELPA if its matching completed prefix is absent;
otherwise it verifies and reuses that prefix. These dependency installs are
shared across checkouts. A new Python-only checkout does not require new
SLATE, OpenBLAS or ELPA builds. Do not use `--fresh` or remove those caches
merely to rebuild the two LORRAX libraries.

The site pins are in `config/frontier/ffi_mpi.sh` and the build scripts:

| Component | Tested version |
|---|---|
| Programming environment | `PrgEnv-gnu/8.7.0` |
| MPI | `cray-mpich/9.0.1` |
| Host BLAS/ScaLAPACK | `cray-libsci/25.09.0` |
| Parallel HDF5 | `cray-hdf5-parallel/1.14.3.7` |
| Host FFTW | `cray-fftw/3.3.10.11` |
| ROCm compile environment | `rocm/7.14.1` |

The ROCm module is used for compilation. At runtime, the JAX plugin and
native device leg use the **same wheel copies** of HIP, rocBLAS, rocSOLVER,
RCCL and rocFFT. The host leg uses the pinned MPI/HDF5/LibSci stack; the
device leg uses that same MPI/HDF5 with its HIP SLATE/OpenBLAS/ELPA stack.

## 3. Run {#run}

Inside an active GPU allocation:

```bash
cd /path/to/lorrax
source config/frontier/gpu_env.sh
srun -N 1 -n 4 -c 7 --gpus-per-task=1 --gpu-bind=closest \
  python -u -m gw.gw_jax -i /path/to/run/gw.in
```

Use one MPI process per visible GCD. Four processes form the tested 2 × 2
mesh on half a Frontier node. A complete node has eight GCDs, but eight
processes are not a square and are not a valid LORRAX mesh. Sixteen processes
form a 4 × 4 mesh across two complete nodes.

The launch helper prepares the wheel's unversioned HSA-library link before
`srun`. For a multi-node run, runtime applies the Frontier RCCL/OFI policy
from `config/frontier/site.toml`, including MPI initialization before the
first RCCL collective. No copied NVIDIA tuning or Perlmutter network setting
is needed. Keep the run directory on shared storage.

Check the startup report: ROCm/MI250X, four MPI processes and four devices,
a 2 × 2 mesh, and both native closures enforced. A checkout build also prints
`LEGACY-UNSEALED` and its file hash; that identifies a private build, not a
failed closure check. The environment helper does not itself certify the
native libraries or run a GPU calculation.

## 4. Verify linking and reuse {#build-reuse}

After building both legs, certify the pair inside an active Frontier
allocation with four free GCDs:

```bash
bash config/frontier/verify.sh
```

This command sources `gpu_env.sh`, verifies both recorded native closures
and the intended host FFTW, then compares native and XLA Fourier transforms
at four MPI processes. It refuses disabled acceptance checks. The host
build's `CMakeCache.txt` supplies the FFTW prefix; when reusing a copied
host leg without that cache, set the existing `LORRAX_FFTW3_STAGE` to its
verified FFTW prefix. No ROCm toolkit module or added library paths are
needed. A login-node check that skips runtime gates does not certify the
pair.

Each closure record names the exact library bytes and dependencies used by
its build. To inspect those records without running GPU transforms:

```bash
python -m lxkit.closure verify src/ffi/cpp/build_host/liblorrax_ffi_host.so
python -m lxkit.closure verify src/ffi/cpp/build_rocm/liblorrax_ffi_rocm.so
```

At startup the loader opens the recorded libraries and checks the process's
actual mappings. A missing, changed or duplicate MPI, HDF5, ROCm or BLAS
library refuses with `GATE native-closure`, naming the file and remedy. Fix
the conflicting installation or rebuild against the intended environment;
adding more library search paths does not repair an already mixed process.
[The native build contract](ffi-build.md#closure) explains the full checks.

Both native legs must share the pinned parallel HDF5 family. The normal
PyPI `h5py` wheel is supported: its private, hashed serial HDF5 libraries
form separate families, so no MPI-enabled `h5py` build is needed. The tested
environment has `h5py 3.16.0` with private HDF5 2.0.0 beside the native
pair's site HDF5 1.14.3.7. Runtime imports `h5py` before the native libraries
and enforces [file ownership](../architecture/slab_io.md#one-owner) between
the serial and collective handles; do not add library paths to combine them.

Reusing another checkout's native pair requires **both**:

1. Matching FFI sources, Frontier build configuration and JAX/jaxlib build
   environment. A matching ABI number alone is insufficient. The optional
   site source-key resolver records this match in `.lx_srckey`; it is site
   tooling, not a command installed by this repository.
2. Both libraries' recorded dependency closures still verify on this
   machine. A source-key match does not replace the closure check.

If either condition is unknown, build the two LORRAX legs in the new checkout
and retain the verified shared dependency installs. Rebuild after changes to
`src/ffi/cpp/`, the Frontier build inputs or the JAX FFI headers; revalidate
records after a wheel, MPI/HDF5 or shared-library update. A Python-only change
can reuse a verified matching pair. Do not stamp or override an older library
merely because it exists or has a closure JSON.

The machine wrapper runs `scripts/verify_ffi_build.sh` with the Frontier
pins before the four-process transform check; use the single wrapper
command for complete acceptance.
