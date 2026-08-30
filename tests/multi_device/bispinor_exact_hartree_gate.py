"""Real-device gate for the scalar/current exact-Hartree output boundary.

The independent four-current algebra is covered in
``test_sc_four_current_hartree.py``.  This gate uses a real bispinor WFN to
check the complete density -> Poisson/TT -> matrix-element transaction and
requires the new device-resident result to match the incumbent host boundary
without losing ``P(None,'x','y')``.

Environment: ``BISPINOR_HARTREE_DECK`` and optional
``BISPINOR_HARTREE_NB`` (default 32).
"""
from __future__ import annotations

import dataclasses
import os
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np                                             # noqa: E402
from jax.sharding import PartitionSpec as P                    # noqa: E402

from common import Meta                                        # noqa: E402
from common.four_current_model import (                        # noqa: E402
    resolve_four_current_representation,
)
from common.collectives import resolve_mesh                    # noqa: E402
from ffi import _services                                      # noqa: E402

_services.ensure_on_path()
import symmetry_maps                                           # noqa: E402
from wfn_loader import WfnLoader                               # noqa: E402
from gw.gw_config import read_lorrax_input                      # noqa: E402
from gw.kin_ion_io import compute_hartree_matrix                # noqa: E402


RTOL = 2.0e-12


def _shard_relative(got, reference):
    scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    worst = 0.0
    for shard in got.addressable_shards:
        worst = max(worst, float(np.max(np.abs(
            np.asarray(shard.data) - reference[shard.index]))) / scale)
    return worst


def _hermiticity_relative(value):
    a = np.asarray(value)
    scale = max(float(np.max(np.abs(a))), 1.0e-300)
    return float(np.max(np.abs(a - np.swapaxes(np.conj(a), -1, -2)))) / scale


def main():
    deck = os.environ["BISPINOR_HARTREE_DECK"]
    nb = int(os.environ.get("BISPINOR_HARTREE_NB", "32"))
    config = read_lorrax_input(deck)
    deck_dir = os.path.dirname(os.path.abspath(deck))
    wfn_path = config.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(deck_dir, wfn_path)

    mesh = resolve_mesh()
    wfn = WfnLoader(wfn_path)
    wfn.adopt_mesh(mesh)
    sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(
        wfn, sym, int(config["nval"]), int(config["ncond"]), nb, 0, True)
    if int(meta.nspinor) != 4:
        meta = dataclasses.replace(meta, nspinor=4, npol=4)
    representation = resolve_four_current_representation(
        True, config.get("bispinor_gw", "bare_transverse"))
    kwargs = dict(
        truncation_2d=(int(config["sys_dim"]) == 2), nb=nb, mesh=mesh,
        include_transverse=True,
        charge_nspinor=(None if representation.charge_bispinor
                        else int(wfn.nspinor)),
        bispinor_lift=(representation.current_lift or "raw"),
        print_fn=print,
    )

    host = compute_hartree_matrix(wfn, sym, meta, **kwargs)
    device = compute_hartree_matrix(
        wfn, sym, meta, return_sharded=True, **kwargs)
    failures = []
    for name in ("charge", "transverse"):
        ref = np.asarray(getattr(host, name))
        got = getattr(device, name)
        rel = _shard_relative(got, ref)
        herm = _hermiticity_relative(ref)
        spec = got.sharding.spec
        nonzero = float(np.max(np.abs(ref)))
        print(f"[bispinor-hartree] {name}: rel={rel:.3e} "
              f"herm={herm:.3e} max={nonzero:.6e} spec={spec}")
        if rel > RTOL:
            failures.append(f"{name} host/device rel {rel:.3e}")
        if herm > RTOL:
            failures.append(f"{name} hermiticity {herm:.3e}")
        if spec != P(None, "x", "y"):
            failures.append(f"{name} layout {spec}")
        if not np.isfinite(nonzero):
            failures.append(f"{name} is nonfinite")
        # This fixture is measured TR-symmetric.  Its correctly projected
        # periodic current may therefore be exact zero (the synthetic gate
        # owns nonzero-current algebra); only scalar charge must be nonzero.
        if name == "charge" and nonzero == 0.0:
            failures.append("charge is zero")

    diagnostics = {
        "current_g0_relative": float(device.current_g0_relative),
        "symmetry_movement": float(
            device.current_symmetry_relative_movement),
        "symmetry_residual": float(
            device.current_symmetry_relative_residual),
        "symmetry_tolerance": float(
            device.current_symmetry_relative_residual_tolerance),
    }
    print(f"[bispinor-hartree] diagnostics {diagnostics}")
    if (not all(np.isfinite(v) for v in diagnostics.values())
            or diagnostics["symmetry_residual"]
            > diagnostics["symmetry_tolerance"]):
        failures.append("current symmetry receipt invalid")
    if failures:
        raise AssertionError("; ".join(failures))
    print("[bispinor-hartree] VERDICT PASS")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finalize_process(rc)
