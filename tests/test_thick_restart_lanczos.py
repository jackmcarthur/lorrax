"""tests/test_thick_restart_lanczos.py — the thick-restart Lanczos solver.

Four claims, each with the twin that fails when the claim stops being true:

1. it finds the spectrum (twin: the arrowhead, dropped, degrades it);
2. it is FIXED SHAPE — the trace counts do not depend on ``n_restarts``
   (twin: a count that scaled with restarts would fail the equality);
3. the alpha-Hermiticity invariant detects a matvec that is not Hermitian;
4. it is shape-agnostic — the exciton ``(nc, nv, nk)`` layout works with no
   reshape, which is what lets the BSE caller hand it a sharded operand.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from solvers.thick_restart_lanczos import (      # noqa: E402
    TRACE_COUNTS,
    reset_trace_counts,
    thick_restart_lanczos_eig,
)


def _hermitian(n, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = 0.5 * (A + A.conj().T)
    # A band-edge cluster, so the test exercises the regime the BSE is in
    # rather than a comfortably separated spectrum.
    w, V = np.linalg.eigh(A)
    w[:4] = w[0] + np.array([0.0, 1e-4, 2e-4, 3e-4])
    return (V * w) @ V.conj().T


def _matvec_of(A):
    Aj = jnp.asarray(A)

    def mv(X):                      # (b, n) -> (b, n)
        return jnp.einsum('ij,bj->bi', Aj, X)
    return mv


def test_trlan_matches_dense_small():
    n, n_eig = 200, 6
    A = _hermitian(n)
    exact = np.sort(np.linalg.eigvalsh(A))[:n_eig]
    ev, vecs, aim = thick_restart_lanczos_eig(
        _matvec_of(A), (n,), n_eig=n_eig, m_max=60, n_keep=20, n_restarts=12)
    ev = np.asarray(ev)
    assert vecs.shape == (n_eig, n)
    assert float(aim) < 1e-10, f"Hermitian operand gave |Im alpha| = {aim}"
    assert np.abs(ev - exact).max() < 1e-9, (
        f"worst |dlam| = {np.abs(ev - exact).max():.3e}\n"
        f"  got   {ev}\n  exact {exact}")


def test_trlan_eigenvectors_have_small_residual():
    n, n_eig = 200, 4
    A = _hermitian(A_n := n)
    mv = _matvec_of(A)
    ev, X, _ = thick_restart_lanczos_eig(
        mv, (n,), n_eig=n_eig, m_max=60, n_keep=20, n_restarts=12)
    R = np.asarray(mv(X)) - np.asarray(X) * np.asarray(ev)[:, None]
    res = np.linalg.norm(R, axis=1)
    assert res.max() < 1e-6, f"residuals {res}"


def test_trlan_is_fixed_shape_independent_of_restart_count():
    """THE fixed-shape claim, stated so it can fail.

    A jagged solver traces its bodies once per distinct shape, so more
    restart cycles would mean more traces.  Here the restart cycle is one
    ``lax.fori_loop`` body at one shape, so the counts must be IDENTICAL for
    2 cycles and for 20.  ``compile_cache_stats()['compiles']`` cannot make
    this statement -- warm, it reads 0 either way.
    """
    n = 120
    A = _hermitian(n, seed=11)
    counts = {}
    for nr in (2, 20):
        reset_trace_counts()
        jax.block_until_ready(thick_restart_lanczos_eig(
            _matvec_of(A), (n,), n_eig=4, m_max=40, n_keep=14,
            n_restarts=nr)[0])
        counts[nr] = dict(TRACE_COUNTS)
    assert counts[2] == counts[20], (
        f"trace counts depend on n_restarts -> the loop is NOT fixed shape:\n"
        f"  2 restarts:  {counts[2]}\n 20 restarts:  {counts[20]}")
    # And pin the absolute numbers, so a future change that adds a shape is
    # caught rather than merely staying self-consistent.
    #
    # ``lanczos_step`` is 1, not 2: the cold cycle and the restart cycle pass
    # the SAME body function with the SAME carry avals to ``lax.fori_loop``,
    # so jax traces it once and reuses it for both trip counts.  The whole
    # iteration -- every Lanczos step of every cycle -- is one traced body.
    assert counts[2]['lanczos_step'] == 1, counts[2]
    assert counts[2]['restart'] == 2, counts[2]
    assert counts[2]['build_T_first'] == 1, counts[2]
    assert counts[2]['build_T_restart'] == 1, counts[2]


def test_trlan_arrowhead_is_load_bearing():
    """RED TWIN for the arrowhead.

    Dropping the arrow couplings is exactly the difference between
    thick-restart Lanczos and naively restarting Lanczos: the retained Ritz
    block stops coupling to the new Krylov directions.  If accuracy did NOT
    degrade, the arrowhead code would be dead and the method would not be
    what its docstring says it is.
    """
    n, n_eig = 200, 6
    A = _hermitian(n)
    exact = np.sort(np.linalg.eigvalsh(A))[:n_eig]
    kw = dict(n_eig=n_eig, m_max=50, n_keep=16, n_restarts=10)
    good, _, _ = thick_restart_lanczos_eig(_matvec_of(A), (n,), **kw)
    bad, _, _ = thick_restart_lanczos_eig(
        _matvec_of(A), (n,), _drop_arrowhead=True, **kw)
    e_good = np.abs(np.asarray(good) - exact).max()
    e_bad = np.abs(np.asarray(bad) - exact).max()
    assert e_good < 1e-9, f"the live arm should converge; got {e_good:.3e}"
    assert e_bad > 100 * e_good, (
        f"dropping the arrowhead did not degrade the answer "
        f"(good {e_good:.3e}, bad {e_bad:.3e}) -- the arrowhead is not "
        f"load bearing, so the method is not thick restart")


def test_trlan_alpha_hermiticity_detects_non_hermitian_matvec():
    """RED TWIN for the free invariant.

    <q, Hq> is real for Hermitian H at EVERY iteration, converged or not.
    A matvec that does not return H.q shows up in Im alpha immediately.
    """
    n = 120
    A = _hermitian(n, seed=5)
    B = A.copy()
    B[0, 1] += 0.5j            # break Hermiticity in one entry
    _, _, aim_ok = thick_restart_lanczos_eig(
        _matvec_of(A), (n,), n_eig=4, m_max=40, n_keep=14, n_restarts=4)
    _, _, aim_bad = thick_restart_lanczos_eig(
        _matvec_of(B), (n,), n_eig=4, m_max=40, n_keep=14, n_restarts=4)
    assert float(aim_ok) < 1e-10, aim_ok
    assert float(aim_bad) > 1e-6, (
        f"a non-Hermitian matvec was not detected: |Im alpha| = {aim_bad}")


def test_trlan_is_shape_agnostic_exciton_layout():
    """The BSE hands it (nc, nv, nk); nothing may reshape or gather that."""
    nc, nv, nk, n_eig = 2, 3, 8, 4
    n = nc * nv * nk
    A = _hermitian(n, seed=7)
    exact = np.sort(np.linalg.eigvalsh(A))[:n_eig]
    Aj = jnp.asarray(A)

    def mv(X):                                   # (b, nc, nv, nk)
        flat = X.reshape(X.shape[0], -1)
        return jnp.einsum('ij,bj->bi', Aj, flat).reshape(X.shape)

    ev, vecs, _ = thick_restart_lanczos_eig(
        mv, (nc, nv, nk), n_eig=n_eig, m_max=30, n_keep=10, n_restarts=8)
    assert vecs.shape == (n_eig, nc, nv, nk)
    assert np.abs(np.asarray(ev) - exact).max() < 1e-9


def test_trlan_rejects_incoherent_parameters():
    A = _hermitian(60, seed=9)
    with pytest.raises(ValueError, match="n_keep"):
        thick_restart_lanczos_eig(_matvec_of(A), (60,), n_eig=10, m_max=20,
                                  n_keep=25, n_restarts=2)
    with pytest.raises(ValueError, match="n_keep"):
        thick_restart_lanczos_eig(_matvec_of(A), (60,), n_eig=30, m_max=40,
                                  n_keep=20, n_restarts=2)
