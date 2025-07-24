# ISDF COHSEX Utilities

This repository contains a small collection of Python scripts implementing the Interpolative Separable Density Fitting approach used by BerkeleyGW.

## Layout

- `src/isdf/` – Python package with all driver routines and utilities.
  - `isdf_init/` – k-means point generation and charge density tools.
  - `gw_isdf/` – GW/COHSEX calculations (`cohsex_isdf`, `w_isdf`, `get_windows`).
  - `bse_isdf/` – Bethe-Salpeter driver and notes.
  - `common/` – shared helpers: wavefunction readers, symmetry maps, tagged arrays and GPU utilities.
- `examples/` – runnable directories for the test and production COHSEX setups.
- `tests/` – small regression tests.
- `misc/` – larger data files and archived scripts.
- `cohsex_isdf.py` – wrapper that launches the test example when run directly.

The main drivers are also available as console commands (`cohsex_isdf`, `kmeans_isdf`, `bse_isdf`) once the package is installed.

## Quick start

Install dependencies and run the unit tests:

```bash
uv pip install -e .
uv run -- python -m pytest -q
```

Run the sample COHSEX calculation:

```bash
uv run python cohsex_isdf.py
```
