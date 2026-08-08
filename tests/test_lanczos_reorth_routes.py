"""Gates for the Lanczos reorthogonalisation route dial (LORRAX_LANCZOS_REORTH).

The shipped route (``mgs``) reorthogonalises one basis vector per ``fori_loop``
trip, so on a sharded Krylov axis it issues ``max_iter (max_iter + 1) / 2``
separate all-reduces of a single complex scalar -- 20 100 of them for the
200-iteration Si BSE record deck, and the largest single GPU-time item in that
run.  The ``cgs2`` route computes every overlap as one matrix product and
issues ``2 * max_iter``.

Each cell below is paired with the failure it exists to catch:

* ``test_unknown_route_refuses`` -- RED TWIN for the dial itself.  If a
  misspelled token fell back to the default, an "optimised" A/B leg could
  silently be the baseline and every number taken from it would be void.
  (Same doctrine as ``bse.bse_stack_matvec.matvec_opts``.)
* ``test_mgs_trip_count_matches_the_shipped_loop`` -- RED TWIN for the
  *arithmetic* behind the 20 100.  The counting helper is re-derived here from
  the shipped loop's own bounds; if either drifts, the number this campaign
  quotes stops being the number the code executes.
* ``test_cgs2_removes_the_inner_sweep`` -- RED TWIN for the optimisation being
  real.  Structural, at jaxpr level, so it does not depend on an XLA version:
  ``mgs`` nests a ``while`` inside the Lanczos ``while``; ``cgs2`` must not.
  Fails if ``cgs2`` ever silently routes back to the sweep.
* ``test_routes_agree_full_reorth`` / ``test_routes_agree_partial_window`` --
  the accuracy gates.  The partial-window cell is the one with teeth for the
  mask: a wrong window (off-by-one, or "whole basis" instead of the window)
  changes the answer visibly at ``n_reorth = 3`` while being invisible at full
  reorth.
* ``test_cgs2_orthogonality`` -- the property the reorth exists to deliver.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from solvers import lanczos as LZ

try:                                    # jax >= 0.6 public home
    from jax.extend.core import ClosedJaxpr, Jaxpr
except ImportError:                     # pragma: no cover - older jax
    from jax._src.core import ClosedJaxpr, Jaxpr


# --------------------------------------------------------------------------
# the dial
# --------------------------------------------------------------------------

def test_default_route_is_the_shipped_sweep(monkeypatch):
    monkeypatch.delenv(LZ._REORTH_ENV, raising=False)
    assert LZ.reorth_kind() == "mgs"
    monkeypatch.setenv(LZ._REORTH_ENV, "")
    assert LZ.reorth_kind() == "mgs"


def test_env_selects_route_and_override_wins(monkeypatch):
    monkeypatch.setenv(LZ._REORTH_ENV, "cgs2")
    assert LZ.reorth_kind() == "cgs2"
    assert LZ.reorth_kind("mgs") == "mgs"          # explicit kwarg wins
    monkeypatch.setenv(LZ._REORTH_ENV, "  CGS2 ")  # tolerant of case/space
    assert LZ.reorth_kind() == "cgs2"


def test_unknown_route_refuses(monkeypatch):
    """RED TWIN: a misspelled dial must REFUSE, never silently run baseline."""
    monkeypatch.setenv(LZ._REORTH_ENV, "cgs")      # plausible near-miss
    with pytest.raises(ValueError, match="unknown reorthogonalisation route"):
        LZ.reorth_kind()
    with pytest.raises(ValueError):
        LZ.reorth_kind("classical")


# --------------------------------------------------------------------------
# the collective-count arithmetic this campaign quotes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("max_iter,n_reorth", [
    (200, 200), (200, 10), (50, 50), (50, 3), (7, 0), (1, 1),
])
def test_mgs_trip_count_matches_the_shipped_loop(max_iter, n_reorth):
    """RED TWIN: re-derive the trip count from the shipped loop's own bounds.

    ``lax.fori_loop(max(0, j - n_reorth), j + 1, ...)`` runs
    ``j + 1 - max(0, j - n_reorth)`` times, and every trip issues exactly one
    ``jnp.vdot`` -- hence one all-reduce.
    """
    expected = 0
    for j in range(max_iter):
        lo = max(0, j - n_reorth)
        hi = j + 1
        expected += max(0, hi - lo)
    assert LZ.mgs_trip_count(max_iter, n_reorth) == expected
    assert LZ.reorth_collective_count("mgs", max_iter, n_reorth) == expected
    assert LZ.reorth_collective_count("cgs2", max_iter, n_reorth) == 2 * max_iter


def test_full_reorth_is_the_triangular_number():
    """The campaign's headline: 200 iterations of full reorth = 20 100."""
    assert LZ.mgs_trip_count(200, 200) == 200 * 201 // 2 == 20100
    assert LZ.reorth_collective_count("cgs2", 200, 200) == 400


# --------------------------------------------------------------------------
# the structural claim
# --------------------------------------------------------------------------

def _sub_jaxprs(eqn):
    out = []
    for v in eqn.params.values():
        for it in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(it, ClosedJaxpr):
                out.append(it.jaxpr)
            elif isinstance(it, Jaxpr):
                out.append(it)
    return out


def _count_prim(jaxpr, name):
    """Occurrences of a primitive anywhere in a jaxpr, sub-jaxprs included."""
    n = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == name:
            n += 1
        for sub in _sub_jaxprs(eqn):
            n += _count_prim(sub, name)
    return n


def _diag_matvec(d):
    dj = jnp.asarray(d, dtype=jnp.complex128)
    return lambda v: dj * v


def test_cgs2_removes_the_inner_sweep():
    """RED TWIN: the batched route must not contain a per-vector loop.

    The Lanczos iteration itself lowers to ``scan`` (its bounds are static), so
    the ONLY ``while`` in the jaxpr is the reorthogonalisation sweep, whose
    bounds are traced.  ``mgs`` must have it; ``cgs2`` must have none.  Stated
    at jaxpr level so no XLA version can move it.
    """
    n, it = 32, 8
    mv = _diag_matvec(np.arange(1, n + 1, dtype=float))
    jx = {k: jax.make_jaxpr(
        lambda k=k: LZ.lanczos_eig_jit(mv, n, n_eig=4, max_iter=it,
                                       n_reorth=it, reorth=k))().jaxpr
        for k in ("mgs", "cgs2")}
    w_mgs = _count_prim(jx["mgs"], "while")
    w_cgs = _count_prim(jx["cgs2"], "while")
    assert _count_prim(jx["mgs"], "scan") >= 1, "outer Lanczos loop vanished"
    assert w_mgs >= 1, f"shipped route lost its per-vector sweep ({w_mgs} while)"
    assert w_cgs == 0, (
        f"cgs2 still carries a per-vector reorth loop "
        f"({w_cgs} while, mgs has {w_mgs})")


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------

def _degenerate_hermitian(n, seed=11):
    """Hermitian with an exactly 4-fold-degenerate low end (the reorth case)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    lam = np.linspace(1.0, 12.0, n)
    lam[:4] = 1.0
    lam[4:8] = 1.05
    lam = np.sort(lam)
    H = (Q * lam) @ Q.conj().T
    return jnp.asarray(0.5 * (H + H.conj().T), dtype=jnp.complex128), lam


def _run(H, n, n_eig, max_iter, n_reorth, kind):
    mv = lambda v: H @ v
    return jax.block_until_ready(LZ.lanczos_eig_jit(
        mv, n, n_eig=n_eig, max_iter=max_iter,
        n_reorth=n_reorth, reorth=kind, seed=3))


@pytest.mark.parametrize("n,max_iter", [(96, 48)])
def test_routes_agree_full_reorth(n, max_iter):
    H, lam = _degenerate_hermitian(n)
    ev_m, _ = _run(H, n, 8, max_iter, max_iter, "mgs")
    ev_c, _ = _run(H, n, 8, max_iter, max_iter, "cgs2")
    delta = float(np.max(np.abs(np.asarray(ev_m) - np.asarray(ev_c))))
    assert delta < 1e-9, f"full-reorth routes disagree by {delta:.3e}"
    # Both must still be solving the right problem.  A single-vector Lanczos
    # sees each DISTINCT eigenvalue once however high its multiplicity (the
    # Krylov space of one start vector meets each eigenspace in one direction),
    # so the reference is the distinct spectrum, and only the well-separated
    # bottom of it is converged at max_iter = n/2.
    distinct = np.unique(lam)
    for ev in (ev_m, ev_c):
        err = float(np.max(np.abs(np.asarray(ev)[:3] - distinct[:3])))
        assert err < 1e-8, f"lowest 3 Ritz values off by {err:.3e}"


def test_routes_agree_partial_window():
    """RED TWIN for the window mask: a wrong window shows up HERE, not at full.

    At ``n_reorth = 3`` the two routes must reproduce each other's *partial*
    reorthogonalisation exactly, including which basis vectors are skipped.  A
    mask that projected the whole basis (or was off by one) would give a
    visibly different, better-orthogonalised answer and fail this cell.
    """
    n, max_iter = 96, 40
    H, _ = _degenerate_hermitian(n)
    ev_m, _ = _run(H, n, 6, max_iter, 3, "mgs")
    ev_c, _ = _run(H, n, 6, max_iter, 3, "cgs2")
    delta = float(np.max(np.abs(np.asarray(ev_m) - np.asarray(ev_c))))
    assert delta < 1e-7, f"partial-window routes disagree by {delta:.3e}"


def test_cgs2_orthogonality():
    """The Krylov basis (rotated into the Ritz frame) stays orthonormal."""
    n, max_iter = 96, 48
    H, _ = _degenerate_hermitian(n)
    out = {}
    for kind in ("mgs", "cgs2"):
        _, V = _run(H, n, max_iter, max_iter, max_iter, kind)
        V = np.asarray(V)
        G = V.conj() @ V.T
        out[kind] = float(np.max(np.abs(G - np.eye(G.shape[0]))))
    assert out["cgs2"] < 1e-10, f"cgs2 orthogonality {out['cgs2']:.3e}"
    # cgs2 must not be materially worse than the shipped sweep.
    assert out["cgs2"] < max(1e-10, 20.0 * out["mgs"]), out


def test_block_routes_agree():
    """The same dial on the block path (``block_size > 1``)."""
    n, bs, max_iter = 96, 2, 20
    H, _ = _degenerate_hermitian(n)
    mvb = lambda Vb: (H @ Vb.T).T
    ev_m, _ = LZ.block_lanczos_eig_jit(
        mvb, n, n_eig=6, block_size=bs, max_iter=max_iter,
        n_reorth=max_iter, reorth="mgs", seed=5)
    ev_c, _ = LZ.block_lanczos_eig_jit(
        mvb, n, n_eig=6, block_size=bs, max_iter=max_iter,
        n_reorth=max_iter, reorth="cgs2", seed=5)
    delta = float(np.max(np.abs(np.asarray(ev_m) - np.asarray(ev_c))))
    assert delta < 1e-9, f"block routes disagree by {delta:.3e}"
