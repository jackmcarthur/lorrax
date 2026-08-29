"""Algebra parity: ``wavefunction_bundle.rotate_wavefunctions`` on
``layout='legacy'`` vs ``layout='face'``, SAME ψ, SAME real (eigh-derived)
U, real multi-rank CUDA.

Report census row "QSGW orbital rotation" (``reports/gwjax_low_mem_bands_
audit_2026-08-22/report.md``); gates the ``low_mem_bands_self_consistent_
unported`` lift condition.

``layout='face'`` needs a real :func:`distrib_la.gemm_plan`, which needs
REAL multi-process CUDA (GUARD 4: cuBLASMp/cuSOLVERMp refuse whenever
``jax.process_count() < mesh.devices.size`` — see
``services/distrib_la/matmul_plan.py``).  Mirrors ``tests/
test_isdf_cq_face_parity.py``'s own ONE-SET-OF-CHECK-BODIES-TWO-CALLERS
convention (pytest cell skips naming why; CLI ``__main__`` runs for real):

    lx run -N 1 -G 4 -n 4 bash tmp/lm_qsgwrot_run_wrap.sh \\
        tests/test_qsgw_rotate_face_parity.py --mesh 2x2

U comes from a REAL small eigensolve (``jnp.linalg.eigh`` on a random
Hermitian ``H``), matching an actual SC map's own ``U_qp`` — not merely a
random unitary — per this task's own instruction.

Comparison exploits that legacy's ``psi_yr``/``psi_xn`` and face's
``psi_nmu``/``psi_mun`` share the SAME logical axis order
(``(nk,n,s,μ)``/``(nk,s,μ,n)`` respectively — see ``wavefunction_bundle``'s
own field comments): both ψ copies are built from the identical
``psi_rmu_Y``/``psi_rmuT_X`` pair via :func:`build_wavefunctions`/
:func:`build_wavefunctions_face`, so no re-derivation of the legacy/face
correspondence is needed here.
"""
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
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.wavefunction_bundle import (
    BandSlices, build_wavefunctions, build_wavefunctions_face,
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
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    rng = np.random.default_rng(seed)
    b0, b1, b2, b3 = edges
    b4 = nb_full + b0
    slices = BandSlices.from_band_edges(b0, b1, b2, b3, b4)

    # psi_rmu_Y / psi_rmuT_X: the SAME two centroid-sampled arrays both
    # build_wavefunctions and build_wavefunctions_face consume (their own
    # docstrings) -- un-conjugated ψ, and its conjugated+transposed twin.
    psi_full = _crand(rng, nk, nb_full, ns, n_rmu)     # (nk,nb,ns,n_rmu)
    psi_rmu_Y = jax.device_put(
        jnp.asarray(psi_full), NamedSharding(mesh, P(None, None, None, None)))
    psi_rmuT_X = jax.device_put(
        jnp.asarray(np.conj(psi_full).transpose(0, 3, 1, 2)),
        NamedSharding(mesh, P(None, None, None, None)))

    enk_full = jax.device_put(
        jnp.asarray(np.sort(rng.standard_normal((nk, nb_full)), axis=1)),
        NamedSharding(mesh, P(None, None)))

    if active_slice is None:
        a_lo, a_hi = 0, slices.sigma.stop
        active_slice_obj = None
    else:
        a_lo, a_hi = active_slice
        active_slice_obj = slice(a_lo, a_hi)
    nb_active = a_hi - a_lo
    E_new, U_np = _real_eigh_U(rng, nk, nb_active)
    U = jax.device_put(
        jnp.asarray(U_np), NamedSharding(mesh, P(None, None, None)))
    enk_active_new = jax.device_put(
        jnp.asarray(E_new), NamedSharding(mesh, P(None, None)))

    with mesh:
        wfns_legacy = build_wavefunctions(
            psi_rmu_Y, psi_rmuT_X, enk_full=enk_full, slices=slices,
            mesh_xy=mesh)
        wfns_face = build_wavefunctions_face(
            psi_rmu_Y, psi_rmuT_X, enk_full=enk_full, slices=slices,
            mesh_xy=mesh)

    out_legacy = rotate_wavefunctions(
        wfns_legacy, U, enk_active_new=enk_active_new, efermi=0.0,
        mesh_xy=mesh, active_slice=active_slice_obj)
    out_face = rotate_wavefunctions(
        wfns_face, U, enk_active_new=enk_active_new, efermi=0.0,
        mesh_xy=mesh, active_slice=active_slice_obj)
    jax.block_until_ready((out_legacy.psi_yr, out_legacy.psi_xn,
                           out_face.psi_nmu, out_face.psi_mun))

    from jax.experimental import multihost_utils as mhu

    def _gather(x):
        return np.asarray(mhu.process_allgather(x, tiled=True))

    # psi_yr (nk,n,s,μ_Y) vs psi_nmu (nk,n_X,s,μ_Y) -- SAME logical axes.
    yr = _gather(out_legacy.psi_yr)
    nmu = _gather(out_face.psi_nmu)
    # psi_xn (nk,s,μ_X,n) vs psi_mun (nk,s,μ_X,n_Y) -- SAME logical axes.
    xn = _gather(out_legacy.psi_xn)
    mun = _gather(out_face.psi_mun)

    assert yr.shape == nmu.shape, f"yr/nmu shape mismatch {yr.shape} {nmu.shape}"
    assert xn.shape == mun.shape, f"xn/mun shape mismatch {xn.shape} {mun.shape}"

    def _rel(a, b, label):
        absdiff = np.abs(a - b)
        ref = float(np.abs(a).max())
        max_abs = float(absdiff.max())
        max_rel = max_abs / max(ref, 1e-300)
        p0(f"  {label}: max|diff|={max_abs:.3e} (ref {ref:.3e}) "
           f"max|rel diff|={max_rel:.3e}")
        return max_rel

    rel_yr = _rel(yr, nmu, "psi_yr vs psi_nmu")
    rel_xn = _rel(xn, mun, "psi_xn vs psi_mun")
    p0(f"  ns={ns} nk={nk} n_rmu={n_rmu} nb_full={nb_full} "
       f"active=[{a_lo},{a_hi})")
    # Engine-parity bar (this repo's convention): relative, never
    # bit-exact -- a distributed cuBLASMp SUMMA GEMM and a rank-local
    # replicated-U einsum are different reductions in a different order.
    assert rel_yr < 1e-10, (
        f"rotate_wavefunctions layout='face' vs 'legacy' parity FAILED "
        f"on psi_yr/psi_nmu: max relative diff {rel_yr:.3e}")
    assert rel_xn < 1e-10, (
        f"rotate_wavefunctions layout='face' vs 'legacy' parity FAILED "
        f"on psi_xn/psi_mun: max relative diff {rel_xn:.3e}")

    # Bands OUTSIDE the active window must be UNCHANGED (pass-through) on
    # BOTH layouts -- the block-diag(U,I) embedding's own correctness
    # claim, checked directly rather than only inferred from the aggregate
    # relative diff above.
    inactive = np.ones(nb_full, dtype=bool)
    inactive[a_lo:a_hi] = False
    if inactive.any():
        orig_yr = _gather(wfns_legacy.psi_yr)
        assert np.array_equal(yr[:, inactive], orig_yr[:, inactive]), (
            "legacy: inactive bands changed by rotation")
        orig_nmu = _gather(wfns_face.psi_nmu)
        assert np.array_equal(nmu[:, inactive], orig_nmu[:, inactive]), (
            "face: inactive bands changed by rotation")


@pytest.mark.parametrize("name,kwargs", _CASES, ids=[c[0] for c in _CASES])
def test_rotate_wavefunctions_face_matches_legacy(name, kwargs):
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
