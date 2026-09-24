"""P=4 gate for the μ-batch symmetry modules (docs/architecture/zeta_fit_mubatch.md).

Four processes, one GPU each, 2x2 mesh.  Runs the receipt of
``tests/test_zeta_mubatch_sym_parity.py`` (whole-orbit μ batches x
orbit-closed rank r blocks x raw-parent projectors unfolded by
``isdf.core.parent_projector_kconv``, against the incumbent r-chunk kernel,
the full-BZ children and direct NumPy sums, then ζ and V_q) on the A-cubic
and glide fixtures with the tail arm the gate resolves on CUDA: the native
``conv_kparent`` when ``LORRAX_CONV_KPARENT_FFI`` is auto, which must then
be the arm that ran.  Pass/fail thresholds are the CPU test's.
Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/zeta_mubatch_sym_p4.py``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import test_zeta_mubatch_sym_parity as parity  # noqa: E402

TAG = "[zeta-mubatch-sym-p4]"


def main():
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    want_native = os.environ.get("LORRAX_CONV_KPARENT_FFI", "auto").lower() != "off"
    failed = []
    for case in parity._CASES:
        rng = np.random.default_rng(2026_09_23)
        if case == "acubic_ns1":
            fx, vertex = parity._acubic_fixture(mesh, rng), 0
        else:
            ns = 2 if case == "glide_ns2" else 4
            fx, vertex = parity._glide_fixture(mesh, rng, ns), int(case.endswith("_v1"))
        out = parity.parity_receipt(case, fx, mesh, vertex)
        if jax.process_index() == 0:
            print(f"{TAG} {json.dumps(out)}", flush=True)
        if want_native and out["tail_arm"] != "native":
            failed.append(f"{case}: tail arm {out['tail_arm']}, want native")
        for key in ("L2_projector_rel", "L3_Z_vs_rchunk_rel", "L3_Z_vs_dense_rel",
                    "L4_zeta_rel", "L4_vq_rel"):
            if key in out and not out[key] < parity._TOL:
                failed.append(f"{case}: {key}={out[key]:.3e}")
        for key in ("red_left_perm_rel", "red_right_perm_rel", "red_split_orbit_rel"):
            if not out[key] > parity._RED:
                failed.append(f"{case}: red twin {key}={out[key]:.3e} did not fire")
    if jax.process_index() == 0:
        print(f"{TAG} {'FAIL ' + '; '.join(failed) if failed else 'PASS'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_main_and_finalize(main)   # a failure keeps its traceback and a nonzero status
