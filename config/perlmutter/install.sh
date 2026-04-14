#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LORRAX module installer for Perlmutter
#
# Run from the LORRAX repo root (or anywhere — the script finds itself):
#   bash config/perlmutter/install.sh
#
# What it does:
#   1. Reads site_config.sh for paths (edit that file first)
#   2. Patches the modulefile with resolved paths
#   3. Installs it into ~/modulefiles (or a shared location)
#   4. Adds `module use` to ~/.bashrc if not already present
#
# For shared group installation:
#   Edit site_config.sh to set LORRAX_MODULEFILE_DIR to a shared path,
#   e.g. /global/common/software/m2651/modulefiles
#   Then have users add:  module use /global/common/software/m2651/modulefiles
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")"
LORRAX_ROOT="$(dirname "$CONFIG_DIR")"

echo "LORRAX root: $LORRAX_ROOT"

# --- Load site config ---
CONFIG="$SCRIPT_DIR/site_config.sh"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: site_config.sh not found at $CONFIG"
    echo "Copy site_config.sh.example to site_config.sh and edit paths."
    exit 1
fi
source "$CONFIG"

# Resolve LORRAX_INSTALL_ROOT
if [ -z "${LORRAX_INSTALL_ROOT:-}" ]; then
    LORRAX_INSTALL_ROOT="$LORRAX_ROOT"
fi

# Validate
if [ ! -d "$LORRAX_INSTALL_ROOT/src/gw" ]; then
    echo "ERROR: LORRAX source not found at $LORRAX_INSTALL_ROOT/src/gw"
    echo "Set LORRAX_INSTALL_ROOT in site_config.sh"
    exit 1
fi

if [ ! -d "${LORRAX_SITE_PACKAGES:-}" ]; then
    echo "WARNING: LORRAX_SITE_PACKAGES directory not found: $LORRAX_SITE_PACKAGES"
    echo "The Shifter container will lack h5py, scipy, matplotlib."
    echo "See site_config.sh for instructions to build it."
fi

MODULEFILE_DIR="${LORRAX_MODULEFILE_DIR:-$HOME/modulefiles}"
IMAGE="${LORRAX_IMAGE:-nvcr.io/nvidia/jax:25.04-py3}"

# --- Install modulefile ---
SRC_MODULE="$CONFIG_DIR/modulefiles/lorrax/0.1.0.lua"
DST_DIR="$MODULEFILE_DIR/lorrax"
DST_MODULE="$DST_DIR/0.1.0.lua"

mkdir -p "$DST_DIR"

echo "Installing modulefile..."
echo "  Source:  $SRC_MODULE"
echo "  Target:  $DST_MODULE"

# Patch the template placeholders with resolved paths
sed \
    -e "s|@LORRAX_ROOT@|$LORRAX_INSTALL_ROOT|g" \
    -e "s|@LORRAX_SITE@|$LORRAX_SITE_PACKAGES|g" \
    -e "s|@LORRAX_IMAGE@|$IMAGE|g" \
    "$SRC_MODULE" > "$DST_MODULE"

echo "  Patched: LORRAX_ROOT  = $LORRAX_INSTALL_ROOT"
echo "  Patched: LORRAX_SITE  = $LORRAX_SITE_PACKAGES"
echo "  Patched: LORRAX_IMAGE = $IMAGE"

# --- Add module use to bashrc ---
BASHRC="$HOME/.bashrc"
MODULE_USE_LINE="module use $MODULEFILE_DIR"

if grep -qF "$MODULE_USE_LINE" "$BASHRC" 2>/dev/null; then
    echo "  ~/.bashrc already has: $MODULE_USE_LINE"
else
    echo "" >> "$BASHRC"
    echo "# LORRAX module support" >> "$BASHRC"
    echo "$MODULE_USE_LINE" >> "$BASHRC"
    echo "  Added to ~/.bashrc: $MODULE_USE_LINE"
fi

# --- Verify ---
echo ""
echo "Installation complete. To use:"
echo "  source ~/.bashrc       # or start a new shell"
echo "  module load lorrax"
echo "  module help lorrax"
echo ""
echo "Quick test (requires a SLURM GPU allocation):"
echo "  lxrun python3 -u -c \"import jax; print(jax.devices())\""
