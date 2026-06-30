#!/bin/bash
# Rebuild jax-finufft and finufft with GPU (CUDA) support

set -e  # Exit on error

echo "========================================="
echo "Rebuilding FINUFFT with GPU support"
echo "========================================="

cd "$(dirname "$0")"

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Is CUDA installed?"
    echo "Expected at: /usr/local/cuda/bin/nvcc"
    exit 1
fi

echo "✓ Found nvcc at: $(which nvcc)"
nvcc --version

# Remove old builds
echo ""
echo "Removing old jax-finufft and finufft builds..."
uv pip uninstall -y jax-finufft finufft || true

# Clear UV cache to force rebuild
echo ""
echo "Clearing UV cache..."
rm -rf .venv/

# Reinstall with GPU support (will use new pyproject.toml with CUDA=ON)
echo ""
echo "========================================="
echo "Installing jax-finufft with CUDA support"
echo "========================================="
echo "This will take ~5-10 minutes to compile..."
echo ""

uv sync --reinstall-package jax-finufft --reinstall-package finufft

echo ""
echo "========================================="
echo "✓ GPU-accelerated FINUFFT installed!"
echo "========================================="
echo ""
echo "Run benchmark:"
echo "  cd ../tests_isdf/cohsex_prod"
echo "  source env.sh"
echo "  uv run python test_nufft_speed.py"

