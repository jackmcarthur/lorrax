#!/bin/bash
set -euo pipefail

# Native uv setup for ISDF COHSEX (no Docker required)
# This script sets up the environment with pre-built packages to avoid compilation issues

echo "[native-setup] Setting up ISDF with uv (native, no Docker)..."

# Create and activate virtual environment
echo "[native-setup] Creating uv environment..."
uv venv --python 3.12
source .venv/bin/activate

# Install JAX with CUDA support from pre-built wheels
echo "[native-setup] Installing JAX with CUDA 12 support..."
uv pip install jax[cuda12_pip] jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install cupy for GPU array operations
echo "[native-setup] Installing CuPy..."
uv pip install cupy-cuda12x

# Install pre-built cufinufft if available, otherwise skip
echo "[native-setup] Installing cuFINUFFT..."
uv pip install cufinufft || echo "Warning: cuFINUFFT not available as wheel, will install from source later"

# Install other dependencies
echo "[native-setup] Installing other dependencies..."
uv pip install numpy scipy h5py matplotlib

# Try to install jax-finufft from pre-built wheel
echo "[native-setup] Attempting to install jax-finufft..."
uv pip install jax-finufft || echo "jax-finufft not available as wheel - will need to build from source"

# Install this project
echo "[native-setup] Installing ISDF project..."
uv pip install -e .

echo "[native-setup] Testing installation..."
python -c "
import jax
import jax.numpy as jnp
print('JAX devices:', jax.devices())
try:
    import cupy
    print('CuPy available:', cupy.cuda.is_available())
except ImportError:
    print('CuPy not available')
try:
    import cufinufft
    print('cuFINUFFT available')
except ImportError:
    print('cuFINUFFT not available')
try:
    from jax_finufft import nufft2
    print('jax-finufft available')
except ImportError:
    print('jax-finufft not available - may need manual installation')
"

echo "[native-setup] Setup complete! Activate with: source .venv/bin/activate"
