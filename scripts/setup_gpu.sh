#!/usr/bin/env bash
set -euo pipefail

# One-shot GPU setup for ISDF (build image, start compose, install jax-finufft, open shell)
# Usage (from Ubuntu/WSL2, at repo root):
#   bash scripts/setup_gpu.sh

here_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${here_dir}/.." && pwd)"
cd "${repo_dir}"

echo "[setup] Verifying Docker..."
docker --version >/dev/null

echo "[setup] Building GPU image (isdf-gpu) ..."
docker build -f Dockerfile.gpu -t isdf-gpu .

echo "[setup] Starting docker-compose service..."
docker compose -f docker-compose.gpu.yaml down || true
docker compose -f docker-compose.gpu.yaml up -d

echo "[setup] Detecting GPU compute capability (host) ..."
arch_detect="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
if [[ -n "${arch_detect}" ]]; then
	arch_compact="$(echo "${arch_detect}" | tr -d '.')"
else
	arch_compact="86" # default Ampere
fi
# Map compute capabilities to supported CUDA architectures
build_arch="${arch_compact}"
if [[ "${arch_compact}" =~ ^[0-9]+$ ]]; then
  if (( arch_compact >= 120 )); then
    # Ada Lovelace (RTX 40xx/50xx series) - use 89 as highest widely supported
    build_arch="89"
  elif (( arch_compact >= 100 )); then
    # Unsupported very high CC, use forward-compatible PTX
    build_arch="90-virtual"
  fi
fi
echo "[setup] Using CMAKE_CUDA_ARCHITECTURES=${build_arch} (detected ${arch_compact})"

echo "[setup] Bootstrapping pip into image venv if missing ..."
docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc '
  if ! /opt/venv/bin/python -c "import pip" >/dev/null 2>&1; then \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    /opt/venv/bin/python /tmp/get-pip.py; \
  fi'

# echo "[setup] Ensuring JAX CUDA matches local toolkit (optional) ... (skipped; use base image JAX)"
# docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc '
#   /opt/venv/bin/python -m pip install -U "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'

echo "[setup] Installing cuFINUFFT (GPU runtime) ..."
docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc '
  # cuFINUFFT will be built as part of jax-finufft installation below
  echo "cuFINUFFT will be built with jax-finufft"
'

echo "[setup] Installing build deps for jax-finufft ..."
docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc '
  /opt/venv/bin/python -m pip install -U pip setuptools wheel scikit-build-core cmake ninja pybind11 setuptools_scm nanobind'

echo "[setup] Building jax-finufft cleanly with CUDA (CMAKE_CUDA_ARCHITECTURES=${build_arch}) ..."
docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc "
  set -eux; \
  export PATH=/usr/local/cuda/bin:\$PATH; \
  export CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS='-j1' TMPDIR=/var/tmp; \
  /opt/venv/bin/python -m pip install -v --no-build-isolation \
    -Ccmake.define.JAX_FINUFFT_USE_CUDA=ON \
    -Ccmake.define.CIBUILDWHEEL=ON \
    -Ccmake.define.CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -Ccmake.define.FINUFFT_STATIC_LINKING=OFF \
    -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=${build_arch} \
    git+https://github.com/flatironinstitute/jax-finufft.git"

echo "[setup] Aligning JAX CUDA plugin with jaxlib version ..."
docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc '
  set -euo pipefail; \
  V=$(/opt/venv/bin/python -c "import jaxlib; print(jaxlib.__version__)"); \
  /opt/venv/bin/python -m pip install -U "jax-cuda12-plugin==${V}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'

echo "[setup] Registering jax-finufft runtime lib path ..."
docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc '
  set -euo pipefail; \
  LIBDIR=$(/opt/venv/bin/python -c "import sysconfig, os; print(os.path.join(sysconfig.get_paths()[\"platlib\"], \"jax_finufft\", \"lib\"))"); \
  echo "$LIBDIR" > /etc/ld.so.conf.d/jax_finufft.conf; \
  ldconfig'

echo "[setup] Installing this project (editable) ..."
docker compose -f docker-compose.gpu.yaml exec -T isdf bash -lc '/opt/venv/bin/pip install -e .'

echo
echo "[setup] All set. Dropping you into the container shell (type 'exit' to leave)."
docker compose -f docker-compose.gpu.yaml exec isdf bash


