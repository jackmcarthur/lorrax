"""Gate on the W-term's folded 1/N, against PRISTINE origin/main.

The matvec spells its inverse k-transform ``conj(fft(conj(x)))`` so both
transforms are unnormalised and the 1/nk is folded onto the (much smaller)
decode output.  That is a REASSOCIATION of the same products, so:

  * nk a power of two  -> every folded factor is exact, result is BITWISE equal;
  * otherwise          -> 1/nk is inexact and the result drifts at ~1e-16.

This gate asserts the right thing in each regime rather than claiming
bit-exactness everywhere.  It loads the pristine origin/main
``bse_stack_matvec.py`` from a detached baseline worktree alongside the
branch's, in ONE process, on the SAME mesh with the SAME inputs — both resolve
``.bse_ring_comm``/``common.*`` to the same files, so only the seam differs.

The stock fixture is MoS2 3x3x1 (nk = 9), which exercises the INEXACT arm; the
Si 4x4x4 record deck (nk = 64) exercises the bitwise arm end to end.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from test_bse_stack_matvec import _mesh, _place, _random_stack  # noqa: E402

#: Max relative deviation tolerated where nk is not a power of two.  Measured
#: ~1e-16 at nk=9; the 1e-10 w-omega precedent is four orders looser.
FOLD_RTOL = 1e-12

BASE_SRC = os.environ.get("FOLD_BASE_SRC", "")


def _load_base():
    name = "bse.bse_stack_matvec_BASE"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BASE_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["bse", "rpa"])
def test_fold_matches_main(bse_dense_state, kernel):
    harness.skip_unless_gpu(pytest)
    if not BASE_SRC or not os.path.exists(BASE_SRC):
        pytest.skip("FOLD_BASE_SRC not set to a pristine origin/main checkout")
    from bse.bse_stack_matvec import build_bse_stack_matvec as build_fold

    base = _load_base()
    assert "zfold" not in open(BASE_SRC).read(), "baseline is not pristine main"
    src_fold = open(os.path.join(os.path.dirname(__file__), "..", "src", "bse",
                                 "bse_stack_matvec.py")).read()
    assert "zfold" not in src_fold, "branch still carries zfold plumbing"
    assert "local_ifftn3" not in src_fold, "branch still calls local_ifftn3"

    data = bse_dense_state
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    pow2 = (nk & (nk - 1)) == 0

    mesh = _mesh()
    sh, arr = _place(data, mesh)
    X = _random_stack(4, nc, nv, nk)

    with mesh:
        Xs = jax.lax.with_sharding_constraint(X, sh.X)
        args = (arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
                data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"],
                arr["M_X"], arr["M_Y"])
        got = np.asarray(build_fold(mesh, nkx, nky, nkz, kernel=kernel)(Xs, *args))
        ref = np.asarray(base.build_bse_stack_matvec(
            mesh, nkx, nky, nkz, kernel=kernel)(Xs, *args))

    assert got.shape == ref.shape, (got.shape, ref.shape)
    d = np.abs(got - ref)
    sc = np.maximum(np.abs(ref), np.percentile(np.abs(ref), 1))
    maxrel = float((d / sc).max())
    print("\n[%s] nk=%d (%dx%dx%d) pow2=%s  n=%d  max_abs=%.6e  max_rel=%.6e  "
          "relL2=%.6e  array_equal=%s"
          % (kernel, nk, nkx, nky, nkz, pow2, got.size, d.max(), maxrel,
             np.linalg.norm(got - ref) / np.linalg.norm(ref),
             np.array_equal(got, ref)))

    if pow2:
        assert np.array_equal(got, ref), (
            "%s: nk=%d is a power of two, so the fold must be BITWISE equal to "
            "main; max abs %.6e" % (kernel, nk, d.max()))
    else:
        assert maxrel <= FOLD_RTOL, (
            "%s: nk=%d fold drift %.6e exceeds %.1e" % (kernel, nk, maxrel,
                                                        FOLD_RTOL))
