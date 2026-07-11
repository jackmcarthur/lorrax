# Quickstart

This runs one complete static-COHSEX calculation end-to-end on a fresh clone — the bundled
regression fixture — with **no native (FFI) build**. It is the fastest way to confirm LORRAX
works on your machine.

!!! warning "Memory requirement"
    The fixture materializes a ~**17 GiB** array on a single device (LORRAX only tiles large
    arrays across a multi-GPU mesh). Run it on a **GPU with ≥ 24 GB**, on **multiple GPUs**,
    or on **CPU with ≥ ~20 GB RAM** (`JAX_PLATFORMS=cpu`). A smaller box will OOM with
    `RESOURCE_EXHAUSTED` — that is sizing, not a build failure. See
    [Generic cloud GPU](installation/cloud.md#two-things-decide-whether-it-runs-driver-and-memory).

## 1. Install (pure-JAX, CPU)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # one-time
git clone <lorrax-repo-url> && cd lorrax
uv sync                                            # editable install; puts src/ on sys.path
```

## 2. Run the bundled fixture

```bash
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
# console-script equivalent:
uv run gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

The fixture sets `use_ffi_io = false` and ships its own wavefunction
(`WFNsmall.h5`), centroids (`centroids_frac_60.txt`), `dipole.h5`, and `kin_ion.h5`, so it
needs nothing native. It writes `eqp_test.dat`; the reference is `eqp_ref.dat` in the same
directory.

## 3. Run the regression test

The same fixture is wrapped as a pytest regression (it shells out to the driver and is
CPU-capable):

```bash
uv run python -m pytest -q       # regression smoke test (CPU, ~1-2 min)
```

## Your first real calculation

For your own system you produce three preprocessing artifacts, then run GW:

1. **Centroids** — `python -m centroid.kmeans_cli <N> --seed 42` → `centroids_frac_<N>.txt`
2. **Dipoles** — `python -m psp.get_dipole_mtxels -i cohsex.in` → `dipole.h5`
3. **Kinetic + ionic** — `python -m gw.kin_ion_io -i cohsex.in` → `kin_ion.h5`
4. **GW** — `python -m gw.gw_jax -i cohsex.in`

On NERSC Perlmutter, `lxpre cohsex.in <N>` runs steps 1–3 in one command; see
[Perlmutter](installation/perlmutter.md).

## Where to next

- [Installation](installation/index.md) — GPU (CUDA 13), container, and from-source tracks
- [Generic cloud GPU](installation/cloud.md) — AWS / RunPod / Vast / Lambda, driver + memory sizing
- [FFI native libraries](installation/ffi-native-libs.md) — distributed `eigh`, sharded
  HDF5, SLATE
- [Theory overview](theory/overview.md) and [physics](theory/physics.md)
- [Codebase](architecture/codebase.md) and [memory model](architecture/memory-model.md)
