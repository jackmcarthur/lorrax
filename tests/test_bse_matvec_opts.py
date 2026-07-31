"""``LORRAX_BSE_MATVEC_OPT`` — the dial, its grammar, and its two levers.

The BSE stack matvec carries two optional rewrites (``bse.bse_stack_matvec``):

  ``densek``  the k-space convolution's ``ifftn -> *W_R -> fftn`` replaced by
              an explicit dense (nk x nk) DFT contraction.  Same linear map,
              different floating-point order.
  ``yhoist``  the two 'y'-axis collectives lifted out of the per-trial scan.
              Reassociates nothing.

Three things have to hold, and each one is worthless without the others:

1.  **The DFT matrices are the transform.**  ``_dft_matrices`` is checked
    against ``jnp.fft.ifftn/fftn(norm='ortho')`` on four k-grid shapes,
    INCLUDING the two mistakes that would otherwise pass a weaker test.
    The obvious control -- transpose IFT -- is VOID here, because the DFT
    matrix is symmetric, so transposing it changes nothing; that near-miss is
    why the controls below are a swap and a conjugation instead.  A W that is
    all-ones is likewise void: the transform pair is then its own inverse and
    every wrong-but-unitary substitution still round-trips to the identity.
    So the fixture uses a RANDOM W.

2.  **The rewrites do not move the operator.**  Every dial setting is applied
    to the same production matvec on the shared dense fixture and compared
    against the dense reference H.  ``yhoist`` must be bit-identical;
    ``densek`` must agree to fp64 round-off but is NOT required to be
    bit-identical, and asserting that it is would be asserting something
    false.

3.  **A misspelled dial refuses.**  ``LORRAX_FFT_FFI_FUSED`` on this codebase
    accepted ``=yes`` and silently ignored ``=Y``; a perf dial that degrades
    to the baseline under a typo makes every A/B built on it unfalsifiable.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_stack_matvec as SM  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the DFT matrices ARE the transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expect", [
    ("", frozenset()),
    ("   ", frozenset()),
    ("yhoist", frozenset({"yhoist"})),
    ("  YHOIST ", frozenset({"yhoist"})),
])
def test_matvec_opt_grammar_accepts(monkeypatch, raw, expect):
    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", raw)
    assert SM.matvec_opts() == expect


@pytest.mark.parametrize("raw", ["dense_k", "densek", "densek2", "yhoisted", "densek,fft",
                                 "true", "1", "on"])
def test_matvec_opt_grammar_refuses(monkeypatch, raw):
    """A token that is not an option must RAISE, never degrade to baseline."""
    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", raw)
    with pytest.raises(ValueError, match="LORRAX_BSE_MATVEC_OPT"):
        SM.matvec_opts()


def test_matvec_opt_unset_is_baseline(monkeypatch):
    monkeypatch.delenv("LORRAX_BSE_MATVEC_OPT", raising=False)
    assert SM.matvec_opts() == frozenset()


# ---------------------------------------------------------------------------
# 2. neither lever moves the operator
# ---------------------------------------------------------------------------
def _mesh():
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))


@pytest.mark.gpu
@pytest.mark.parametrize("opt,tol,label", [
    ("", 1e-9, "baseline"),
    ("yhoist", 0.0, "yhoist (must be BIT-identical)"),
])
def test_dial_preserves_the_operator(bse_dense_state, monkeypatch, opt, tol,
                                     label):
    harness.skip_unless_gpu(pytest)
    from test_bse_dense_reference import _build_dense_H, _relerr
    from test_bse_stack_matvec import _place, _random_stack

    data = bse_dense_state
    H, _, _, _ = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz

    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", opt)
    mesh = _mesh()
    sh, arr = _place(data, mesh)
    nt = 3
    X = _random_stack(nt, nc, nv, nk)
    with mesh:
        Xs = jax.lax.with_sharding_constraint(X, sh.X)
        mv = SM.build_bse_stack_matvec(mesh, nkx, nky, nkz, kernel="bse")
        HXs = np.asarray(mv(
            Xs, arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
            data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"],
            arr["M_X"], arr["M_Y"]))
    for t in range(nt):
        ref = H @ np.asarray(X)[t].reshape(-1)
        err = _relerr(HXs[t].reshape(-1), ref)
        assert err < 1e-9, f"{label} trial {t}: relerr {err:.3e} vs dense H"

    if tol == 0.0:
        # yhoist only re-orders WHICH collective moves the bytes; the
        # arithmetic and its association are untouched, so anything but a bit
        # match means it changed the computation.
        monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", "")
        with mesh:
            base_mv = SM.build_bse_stack_matvec(mesh, nkx, nky, nkz,
                                                kernel="bse")
            base = np.asarray(base_mv(
                jax.lax.with_sharding_constraint(X, sh.X),
                arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
                data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"],
                arr["M_X"], arr["M_Y"]))
        assert np.array_equal(base, HXs), \
            "yhoist changed a bit — it is supposed to move only collectives"
