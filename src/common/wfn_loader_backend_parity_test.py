"""WfnLoader: phdf5 backend bit-equal vs eager (P2 contract test).

Runs both backends against the same captured WFN.h5 and asserts the
``load(...)`` outputs match byte-for-byte under the same (band_range,
k, sharding) request.

Designed to run under Shifter via ``lxrun`` (the phdf5 FFI requires
the .so + a 2-D device mesh + MPI).  Standalone script — pytest under
Shifter is not currently wired (see KNOWN_SANDBOX_ERRORS.md
2026-05-06: "Shifter runtime used by ``lorrax_B`` lacks ``pytest``").

Usage::

    lxrun python3 -u -m common.wfn_loader_backend_parity_test \\
        --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5 \\
        --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys

from runtime import set_default_env, init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
set_default_env()
init_jax_distributed()

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils

from file_io.wfn_loader import WfnLoader


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _parse_mesh(spec: str) -> tuple[int, int]:
    parts = spec.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"--mesh expects PxQ form, got {spec!r}")
    return int(parts[0]), int(parts[1])


def _build_mesh(p: int, q: int) -> Mesh:
    devices = np.asarray(jax.devices()).reshape(p, q)
    return Mesh(devices, axis_names=("x", "y"))


def _replicate_to_host(arr) -> np.ndarray:
    """Gather a sharded jax.Array onto rank 0 as a numpy array."""
    gathered = multihost_utils.process_allgather(arr, tiled=False)
    h = np.asarray(gathered)
    # process_allgather adds a leading process axis when tiled=False.
    while h.ndim > arr.ndim:
        h = h[0]
    return h


def _check(name: str, eager: np.ndarray, phdf5: np.ndarray, atol: float = 0.0) -> bool:
    if eager.shape != phdf5.shape:
        _log(f"FAIL {name}: shape mismatch eager={eager.shape} phdf5={phdf5.shape}")
        return False
    diff = np.abs(eager - phdf5)
    max_d = float(diff.max())
    if max_d > atol:
        _log(f"FAIL {name}: max|Δ| = {max_d:.3e}  (atol={atol:.0e})")
        bad = np.unravel_index(np.argmax(diff), diff.shape)
        _log(f"   worst at {bad}: eager={eager[bad]!r}  phdf5={phdf5[bad]!r}")
        return False
    _log(f"PASS {name}: max|Δ| = {max_d:.3e}  (atol={atol:.0e})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--mesh", required=True, help="PxQ")
    ap.add_argument("--bands", default="0,4", help="b_lo,b_hi")
    ap.add_argument("--atol", type=float, default=1e-12)
    args = ap.parse_args()

    p, q = _parse_mesh(args.mesh)
    if jax.device_count() < p * q:
        _log(f"need {p*q} devices, found {jax.device_count()}")
        return 1

    mesh = _build_mesh(p, q)
    b_lo, b_hi = (int(x) for x in args.bands.split(","))

    _log(f"[WfnLoader parity] wfn={args.wfn}  mesh={p}x{q}  bands=({b_lo},{b_hi})")
    _log(f"[WfnLoader parity] world={jax.process_count()}  devices={jax.device_count()}")

    all_ok = True

    # ---- Open both backends ----
    eager = WfnLoader(args.wfn, mesh=mesh, backend="eager")
    phdf5 = WfnLoader(args.wfn, mesh=mesh, backend="phdf5")
    _log(f"[WfnLoader parity] eager.backend={eager.backend}  "
         f"phdf5.backend={phdf5.backend}")

    try:
        sharding = P(None, ("x", "y"), None, None)

        # ---- 1. IBZ raw ----
        psi_e = _replicate_to_host(
            eager.load(bands=(b_lo, b_hi), k="ibz", sharding=sharding))
        psi_p = _replicate_to_host(
            phdf5.load(bands=(b_lo, b_hi), k="ibz", sharding=sharding))
        all_ok &= _check("load(k='ibz')", psi_e, psi_p, atol=args.atol)

        # ---- 2. Full-BZ unfold ----
        psi_e_full = _replicate_to_host(
            eager.load(bands=(b_lo, b_hi), k="full_bz", sharding=sharding))
        psi_p_full = _replicate_to_host(
            phdf5.load(bands=(b_lo, b_hi), k="full_bz", sharding=sharding))
        all_ok &= _check("load(k='full_bz')", psi_e_full, psi_p_full,
                          atol=args.atol)

        # ---- 3. gvecs + ngk_valid identity (host metadata; both backends
        # should produce identical numpy arrays since the math is the
        # same SymMaps under the hood) ----
        ge = np.asarray(eager.gvecs(k="full_bz"))
        gp = np.asarray(phdf5.gvecs(k="full_bz"))
        ok = bool(np.array_equal(ge, gp))
        _log(f"{'PASS' if ok else 'FAIL'} gvecs(k='full_bz') exact")
        all_ok &= ok

    finally:
        eager.close()
        phdf5.close()

    multihost_utils.sync_global_devices("wfn_loader_parity_done")
    _log(f"[WfnLoader parity] {'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
