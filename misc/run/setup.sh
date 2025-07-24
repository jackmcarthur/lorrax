#!/bin/bash
set -e

# Upgrade pip to avoid older versions that may fail
python3 -m pip install --upgrade pip

# Install base Python dependencies
pip install --prefer-binary -r requirements.txt

# Optional GPU libraries (CuPy, FFTX, Spiral) are intentionally skipped to
# keep setup time under two minutes. The code will automatically fall back to
# NumPy implementations when these packages are unavailable.
