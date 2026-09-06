"""Distributed face rotations agree with explicit NumPy band-column rotation."""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _TESTS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")

import argparse

import numpy as np
import jax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.wavefunction_bundle import (
    BandSlices, build_wavefunctions_face,
    rotate_wavefunctions,
)

PX = PY = 2

#: (ns, nk, n_rmu, nb_full, b0,b1,b2,b3, active_slice or None, seed).
#: active_slice=None exercises the DEFAULT (sigma window, a_lo=0);
#: an explicit slice exercises a NARROWER active window whose a_lo != 0
#: (the case an offset bug would show up in first, same reasoning
#: test_isdf_cq_face_parity.py's own "asym_lower" case documents).
_CASES = (
    ("ns1_default_active", dict(
        ns=1, nk=2, n_rmu=6, nb_full=10, edges=(0, 2, 3, 8),
        active_slice=None, seed=101)),
    ("ns2_default_active", dict(
        ns=2, nk=2, n_rmu=6, nb_full=10, edges=(0, 2, 3, 8),
        active_slice=None, seed=102)),
    ("ns4_narrow_offset_active", dict(
        ns=4, nk=3, n_rmu=6, nb_full=10, edges=(0, 2, 3, 8),
        active_slice=(2, 7), seed=104)),
    ("ns1_narrow_offset_active", dict(
        ns=1, nk=3, n_rmu=6, nb_full=10, edges=(0, 2, 3, 8),
        active_slice=(2, 7), seed=103)),
)


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _real_eigh_U(rng, nk, na):
    """A genuine eigenvector matrix per k — SAME mechanism an actual SC
    map's ``U_qp`` comes from (``sc_iteration.py``'s ``eigh_kshard``/
    ``distributed_eigh_bands``), not merely a random unitary."""
    A = _crand(rng, nk, na, na)
    H = A + np.conj(np.swapaxes(A, -1, -2))
    E, U = np.linalg.eigh(H)
    return E.astype(np.float64), U.astype(np.complex128)


def check_rotate_face_parity(mesh, *, ns, nk, n_rmu, nb_full, edges,
                             active_slice, seed):
    """Rotate complex eigenvector columns and preserve inactive bands exactly."""
    from jax.experimental import multihost_utils as mhu
    rng = np.random.default_rng(seed)
    slices = BandSlices.from_band_edges(*edges, nb_full + edges[0])
    psi = _crand(rng, nk, nb_full, ns, n_rmu)
    energies = np.sort(rng.standard_normal((nk, nb_full)), axis=1)
    a_lo, a_hi = (0, slices.sigma.stop) if active_slice is None else active_slice
    E_new, U = _real_eigh_U(rng, nk, a_hi - a_lo)
    with mesh:
        rep4 = NamedSharding(mesh, P(None, None, None, None))
        wfns = build_wavefunctions_face(
            jax.device_put(psi, rep4),
            jax.device_put(psi.conj().transpose(0, 3, 1, 2), rep4),
            enk_full=jax.device_put(energies, NamedSharding(mesh, P(None, None))),
            slices=slices, mesh_xy=mesh)
        out = rotate_wavefunctions(
            wfns, U, enk_active_new=E_new, efermi=0.0, mesh_xy=mesh,
            active_slice=None if active_slice is None else slice(a_lo, a_hi))
    jax.block_until_ready((out.psi_nmu, out.psi_mun))
    nmu = np.asarray(mhu.process_allgather(out.psi_nmu, tiled=True))
    mun = np.asarray(mhu.process_allgather(out.psi_mun, tiled=True))
    expected = psi.copy()
    expected[:, a_lo:a_hi] = np.einsum(
        "kmn,kmsr->knsr", U, psi[:, a_lo:a_hi])
    wrong = np.einsum("knm,kmsr->knsr", U, psi[:, a_lo:a_hi])
    assert np.max(np.abs(wrong - expected[:, a_lo:a_hi])) > 0.1
    np.testing.assert_allclose(nmu, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(mun, expected.transpose(0, 2, 3, 1),
                               rtol=1e-10, atol=1e-12)
    inactive = np.ones(nb_full, dtype=bool)
    inactive[a_lo:a_hi] = False
    np.testing.assert_array_equal(nmu[:, inactive], psi[:, inactive])
    energies[:, a_lo:a_hi] = E_new
    np.testing.assert_array_equal(np.asarray(out.enk), energies)
    np.testing.assert_array_equal(np.asarray(out.occ), energies <= 0.0)
    original = np.asarray(mhu.process_allgather(wfns.psi_nmu, tiled=True))
    np.testing.assert_array_equal(original, psi)


@pytest.mark.parametrize("name,kwargs", _CASES, ids=[c[0] for c in _CASES])
def test_rotate_wavefunctions_face_matches_numpy(name, kwargs):
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes for gemm_plan's cuBLASMp "
            f"GUARD 4 (got process_count={jax.process_count()}); run "
            f"`lx run -N 1 -G 4 -n 4 ... {__file__} --mesh 2x2` for the "
            f"real check (see this module's docstring)")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_rotate_face_parity(mesh, **kwargs)


def _cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2", help="PxQ process mesh")
    args = ap.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}")
    if jax.device_count() != px * py:
        p0(f"REFUSE: need exactly {px * py} devices for a {args.mesh} mesh; "
           f"got {jax.device_count()}")
        return 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    failures = 0
    for name, kwargs in _CASES:
        try:
            check_rotate_face_parity(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    p0(f"done: {len(_CASES) - failures}/{len(_CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
