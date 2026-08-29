"""Real P=4 gate for ordered GN residues and face-Sigma dispatch.

Run with one rank per GPU::

    lx run -N 1 -G 4 -n 4 ... python -u \
        tests/multi_device/gn_ppm_ordered_residue_gate.py --mesh 2x2
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_TESTS = os.path.join(_REPO, "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np  # noqa: E402


def _mesh(spec):
    import jax
    from jax.sharding import Mesh

    px, py = (int(value) for value in spec.split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _put(value, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    return jax.device_put(
        np.asarray(value), NamedSharding(mesh, P(None, "x", "y")))


def _gather(value):
    import jax

    if jax.process_count() == 1:
        return np.asarray(value)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


def _response(R_positive, R_negative, omega, z):
    return R_positive / (z - omega) - R_negative / (z + omega)


def _main():
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from gw.ppm_sigma import fit_ppm
    import symmetry_maps
    from symmetry_maps import q_negation_index

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="2x2")
    args = parser.parse_args()
    mesh = _mesh(args.mesh)
    target = NamedSharding(mesh, P(None, "x", "y"))

    rng = np.random.default_rng(20260828)
    nq, nmu = 3, 4
    raw = (
        rng.standard_normal((nq, nmu, nmu))
        + 1j * rng.standard_normal((nq, nmu, nmu))
    )
    R_positive = np.einsum(
        "qab,qcb->qac", raw, np.conj(raw), optimize=True)
    R_positive += 0.5 * np.eye(nmu)[None, ...]
    q_neg = q_negation_index((nq, 1, 1))
    R_negative = np.swapaxes(R_positive[q_neg], -1, -2)
    omega = np.full((nq, nmu, nmu), 1.7, dtype=np.float64)
    z = 1.1j
    W0 = _response(R_positive, R_negative, omega, 0.0j)
    W_probe = _response(R_positive, R_negative, omega, z)
    zero = np.zeros_like(W0)

    partner_rows = []
    original_pair = symmetry_maps.q_pair_transpose

    def _record_pair_rows(*args, **kwargs):
        rows = kwargs.get("q_rows")
        partner_rows.append(nq if rows is None else int(np.asarray(rows).size))
        return original_pair(*args, **kwargs)

    symmetry_maps.q_pair_transpose = _record_pair_rows
    try:
        fit = fit_ppm(
            _put(W0, mesh),
            _put(W_probe, mesh),
            _put(zero, mesh),
            z,
            mesh,
            fallback_omega=2.0,
            n_mu_logical=nmu,
            q_neg_index=q_neg,
            include_frequency_odd_response=True,
        )
    finally:
        symmetry_maps.q_pair_transpose = original_pair
    assert partner_rows and max(partner_rows) < nq, (
        "ordered fit requested a whole q-partner operator instead of "
        f"bounded rows: {partner_rows}")
    assert fit.B_negative_q is not None
    for name, value in (
        ("R_positive", fit.B_q),
        ("R_negative", fit.B_negative_q),
        ("Omega", fit.Omega_q),
    ):
        assert value.sharding == target, (
            f"{name} escaped P(None,x,y): {value.sharding}")

    got_positive = _gather(fit.B_q)
    got_negative = _gather(fit.B_negative_q)
    got_omega = _gather(fit.Omega_q)
    np.testing.assert_allclose(
        got_positive, R_positive, rtol=8.0e-13, atol=8.0e-13)
    np.testing.assert_allclose(
        got_negative, R_negative, rtol=8.0e-13, atol=8.0e-13)
    np.testing.assert_allclose(got_omega, omega, rtol=8.0e-13, atol=8.0e-13)
    np.testing.assert_allclose(
        -(got_positive + got_negative) / got_omega,
        W0,
        rtol=8.0e-13,
        atol=8.0e-13,
    )
    np.testing.assert_allclose(
        _response(got_positive, got_negative, got_omega, z),
        W_probe,
        rtol=8.0e-13,
        atol=8.0e-13,
    )
    # Reuse the existing real-mesh face finalizer gate.  Its ordered fixture
    # observes the production driver dispatching R+ to both conduction
    # branches and R- to both valence branches, for replicated and sharded
    # Sigma outputs, without duplicating that driver scaffold here.
    from test_ppm_sigma_face_sharded_tail import (
        check_face_sharded_tail_parity,
    )
    check_face_sharded_tail_parity(mesh)
    if jax.process_index() == 0:
        print(
            "PASS ordered GN P=4: R+/R- recovered, static/probe samples "
            "reconstructed, all pole fields P(None,x,y), bounded partner "
            f"q blocks {partner_rows}, and face Sigma dispatched R+/R-",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_main)
