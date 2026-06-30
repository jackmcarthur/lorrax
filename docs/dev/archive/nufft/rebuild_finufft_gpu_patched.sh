#!/bin/bash
# Rebuild jax-finufft with GPU support and CUDA 13 compatibility patch

set -e  # Exit on error

echo "========================================="
echo "Rebuilding FINUFFT with GPU (CUDA 13.0)"
echo "========================================="

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Is CUDA installed?"
    exit 1
fi

echo "✓ Found nvcc at: $(which nvcc)"
nvcc --version

# Remove old builds
echo ""
echo "Removing old jax-finufft and finufft..."
uv pip uninstall -y jax-finufft finufft 2>/dev/null || true

# Clear cache
echo "Clearing UV cache..."
rm -rf .venv/

# Install jax-finufft (will download source)
echo ""
echo "========================================="
echo "Downloading jax-finufft source..."
echo "========================================="
uv pip install --no-build-isolation --no-binary jax-finufft jax-finufft || true

# Find the downloaded source
CACHE_DIR="$HOME/.cache/uv/git-v0/checkouts"
FINUFFT_SRC=$(find "$CACHE_DIR" -type f -name "helper_cuda.h" -path "*/finufft/include/cufinufft/contrib/*" | head -n 1)

if [ -z "$FINUFFT_SRC" ]; then
    echo "ERROR: Could not find helper_cuda.h in UV cache"
    echo "Cache dir: $CACHE_DIR"
    exit 1
fi

echo "✓ Found helper_cuda.h at: $FINUFFT_SRC"

# Backup original
cp "$FINUFFT_SRC" "$FINUFFT_SRC.orig"

# Apply patch
echo ""
echo "========================================="
echo "Patching for CUDA 13.0 compatibility..."
echo "========================================="

# Use sed to add #ifdef guards around the deprecated error codes
sed -i '
/case CUFFT_INCOMPLETE_PARAMETER_LIST:/ {
    i\#ifdef CUFFT_INCOMPLETE_PARAMETER_LIST
    n
    a\#endif
}
/case CUFFT_PARSE_ERROR:/ {
    i\#ifdef CUFFT_PARSE_ERROR
    n
    a\#endif
}
/case CUFFT_LICENSE_ERROR:/ {
    i\#ifdef CUFFT_LICENSE_ERROR
    n
    a\#endif
}
' "$FINUFFT_SRC"

echo "✓ Patch applied"

# Show what changed
echo ""
echo "Patch diff:"
diff -u "$FINUFFT_SRC.orig" "$FINUFFT_SRC" || true

# Now rebuild with GPU support
echo ""
echo "========================================="
echo "Building jax-finufft with CUDA support"
echo "========================================="
echo "This will take ~5-10 minutes..."
echo ""

# Enable CUDA in config temporarily
cat > /tmp/pyproject_cuda.toml << 'EOF'
[tool.uv.extra-build-variables.jax-finufft]
CMAKE_CUDA_COMPILER = "/usr/local/cuda/bin/nvcc"
CUDACXX             = "/usr/local/cuda/bin/nvcc"
CUDAHOME            = "/usr/local/cuda"
CMAKE_ARGS          = "-DJAX_FINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"

[tool.uv.extra-build-variables.finufft]
CMAKE_CUDA_COMPILER = "/usr/local/cuda/bin/nvcc"
CUDACXX             = "/usr/local/cuda/bin/nvcc"
CUDAHOME            = "/usr/local/cuda"
CMAKE_ARGS          = "-DJAX_FINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
EOF

# Temporarily update pyproject.toml
cp pyproject.toml pyproject.toml.bak
# Merge the CUDA settings
python3 << 'PYTHON'
import sys
sys.path.insert(0, '.')
# Just set environment variables instead
import os
os.environ['CMAKE_CUDA_COMPILER'] = '/usr/local/cuda/bin/nvcc'
os.environ['CUDACXX'] = '/usr/local/cuda/bin/nvcc'
os.environ['CUDAHOME'] = '/usr/local/cuda'
os.environ['CMAKE_ARGS'] = '-DJAX_FINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native'
PYTHON

# Set environment variables for build
export CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
export CUDACXX="/usr/local/cuda/bin/nvcc"
export CUDAHOME="/usr/local/cuda"
export CMAKE_ARGS="-DJAX_FINUFFT_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"

# Rebuild
uv pip install --no-binary jax-finufft --no-build-isolation --force-reinstall jax-finufft

# Restore original pyproject.toml
mv pyproject.toml.bak pyproject.toml 2>/dev/null || true

echo ""
echo "========================================="
echo "✓ GPU-accelerated FINUFFT installed!"
echo "========================================="
echo ""
echo "Run benchmark:"
echo "  cd ../tests_isdf/cohsex_prod"
echo "  source env.sh"
echo "  uv run python test_nufft_speed.py"

