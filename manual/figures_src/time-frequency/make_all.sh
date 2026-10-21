#!/usr/bin/env bash
# Regenerate docs/assets/figures/theory-timefreq-*.png (docs/theory/minimax.md).
# Needs the plotting venv (numpy, scipy, matplotlib, mpmath):
#   uv venv _tools/plotenv && VIRTUAL_ENV=_tools/plotenv uv pip install numpy scipy matplotlib mpmath
# and a read-only LORRAX checkout at $LORRAX_SRC (default ~/projects/lorrax_main);
# lorrax_minimax.py imports its services/minimax rule builders without writing bytecode.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../_tools/plotenv/bin/python
for f in fig_poles fig_si_spectrum fig_broadening fig_sampling fig_transform fig_exponentials fig_laplace_error fig_sigma_kinds fig_windows fig_sigma_error; do
  "$PY" -B "$f.py"
done
