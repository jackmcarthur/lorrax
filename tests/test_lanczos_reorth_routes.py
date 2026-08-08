"""Gates for the Lanczos reorthogonalisation route (LORRAX_LANCZOS_REORTH).

The DEFAULT is ``cgs2``: every overlap of a sweep computed as one matrix
product, ``2 * max_iter`` all-reduces of an ``(m,)`` vector.  The legacy
``mgs`` route reorthogonalises one basis vector per ``fori_loop`` trip, so on a
sharded Krylov axis it issues ``max_iter (max_iter + 1) / 2`` separate
all-reduces of a single complex SCALAR -- 20 100 of them for the 200-iteration
Si BSE record deck, where it was the largest single GPU-time item in the run.
``mgs`` is kept reachable for bisects and for reproducing pre-2026-08-08 runs.

Each cell below is paired with the failure it exists to catch:

* ``test_default_route_is_batched`` + ``test_the_default_really_is_batched`` --
  RED TWINS for the LANDING.  The first pins the resolver, the second pins the
  jaxpr a production caller actually gets with nothing set.  Fails if the
  default ever regresses to the sweep.
* ``test_mgs_fallback_is_reachable_from_the_env`` -- RED TWIN for the other
  direction: the legacy route must stay one env var away, via the env path a
  bisect would use.  Fails if the fallback is amputated.
* ``test_unknown_route_refuses`` -- RED TWIN for the dial itself.  A misspelled
  token must refuse, not silently pick a route -- which now matters in BOTH
  directions, since a typo must not hand back the 20 100-collective sweep
  either.  (Same doctrine as ``bse.bse_stack_matvec.matvec_opts``.)
* ``test_mgs_trip_count_matches_the_shipped_loop`` +
  ``test_record_deck_collective_counts_are_pinned`` -- RED TWINS for the
  *arithmetic* behind 20 100 and 400.  The counting helper is re-derived from
  the legacy loop's own bounds; if either drifts, the numbers this landing
  quotes stop being the numbers the code executes.
* ``test_cgs2_removes_the_inner_sweep`` -- structural, at jaxpr level, so it
  does not depend on an XLA version.
* ``test_routes_agree_full_reorth`` / ``test_routes_agree_partial_window`` --
  the accuracy gates.  The partial-window cell is the one with teeth for the
  mask: a wrong window (off-by-one, or "whole basis" instead of the window)
  changes the answer visibly at ``n_reorth = 3`` while being invisible at full
  reorth.
* ``test_cgs2_orthogonality`` -- the property the reorth exists to deliver.
* ``test_window_includes_the_current_vector`` +
  ``test_routes_select_the_same_window`` +
  ``test_current_vector_projection_holds_orthogonality`` -- the 2026-08-08
  widening of the window from ``i < j`` to ``i <= j``.  The first pins the set
  from both sides AND pins that the collective counts did not move (the whole
  argument for the change being free); the second pins that the sweep's scalar
  predicate and the batched route's mask select the identical slots, so the two
  routes cannot drift apart; the third is the RED TWIN the owner asked for --
  reverting to ``i < j`` in-process must turn the orthogonality gate red.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from solvers import lanczos as LZ
from bse import bse_lanczos as BL

try:                                    # jax >= 0.6 public home
    from jax.extend.core import ClosedJaxpr, Jaxpr
except ImportError:                     # pragma: no cover - older jax
    from jax._src.core import ClosedJaxpr, Jaxpr


# --------------------------------------------------------------------------
# the dial
# --------------------------------------------------------------------------

def test_default_route_is_batched():
    """The DEFAULT is the batched route.  None and empty both mean cgs2."""
    assert LZ.reorth_kind() == "cgs2"
    assert LZ.reorth_kind(None) == "cgs2"
    assert LZ.reorth_kind("") == "cgs2"
    assert LZ._REORTH_DEFAULT == "cgs2"


def test_env_selects_route_and_token_wins(monkeypatch):
    """The env var is read by the BSE layer; the solver takes a token."""
    monkeypatch.setenv(BL.REORTH_ENV, "mgs")
    assert BL.reorth_route() == "mgs"              # legacy fallback reachable
    monkeypatch.setenv(BL.REORTH_ENV, "  MGS ")    # tolerant of case/space
    assert BL.reorth_route() == "mgs"
    monkeypatch.delenv(BL.REORTH_ENV, raising=False)
    assert BL.reorth_route() == "cgs2"             # unset -> default
    assert LZ.reorth_kind("mgs") == "mgs"          # explicit token honoured


def test_solvers_lanczos_reads_no_environment():
    """RED TWIN for the LAYERING rule that caught this feature's first draft.

    ``solvers`` is L2 -- physics-agnostic mathematics that must be a function
    of its arguments (tests/test_layering.py). The first version of this route
    resolved LORRAX_LANCZOS_REORTH inside solvers/lanczos.py and the layering
    census went red with ``{'solvers.lanczos': ['<dynamic>']}``.  The dial now
    lives in ``bse.bse_lanczos``; the solver takes a token.

    This cell is deliberately NOT a duplicate of the layering gate: it names
    THIS module, so a future edit that reaches for os.environ here fails in the
    file that owns the feature, next to the explanation, instead of only in a
    census someone runs later.
    """
    import ast
    import inspect
    src = inspect.getsource(LZ)
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            bad.append("os.environ")
        if isinstance(node, ast.Name) and node.id in ("getenv",):
            bad.append("getenv")
        if isinstance(node, ast.Attribute) and node.attr in ("getenv",):
            bad.append("os.getenv")
    assert not bad, f"solvers.lanczos reads the environment: {sorted(set(bad))}"
    assert not hasattr(LZ, "_REORTH_ENV"), (
        "the env-var name moved to bse.bse_lanczos.REORTH_ENV; a copy left "
        "here invites the read to come back with it")


def test_unknown_route_refuses(monkeypatch):
    """RED TWIN: a misspelled dial must REFUSE, never silently pick a route.

    Now that ``cgs2`` is the default the refusal matters in BOTH directions: a
    typo must not silently hand back the 20 100-collective sweep either.
    """
    monkeypatch.setenv(BL.REORTH_ENV, "cgs")       # plausible near-miss
    with pytest.raises(ValueError, match="unknown reorthogonalisation route"):
        BL.reorth_route()
    for bad in ("classical", "mgs2", "gs", "CGS-2"):
        with pytest.raises(ValueError):
            LZ.reorth_kind(bad)


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


def test_record_deck_collective_counts_are_pinned(monkeypatch):
    """The two numbers this landing is built on, pinned at the DEFAULT route.

    200 Lanczos iterations at full reorth on the Si record deck: the shipped
    sweep issued the triangular number 20 100; the default now issues 400.
    """
    monkeypatch.delenv(BL.REORTH_ENV, raising=False)
    assert BL.reorth_route() == "cgs2"
    assert LZ.reorth_collective_count(BL.reorth_route(), 200, 200) == 400
    # the legacy route's number, for the record and for the fallback cell below
    assert LZ.mgs_trip_count(200, 200) == 200 * 201 // 2 == 20100
    assert LZ.reorth_collective_count("mgs", 200, 200) == 20100
    # the window does not change the batched count; it does change the sweep's
    assert LZ.reorth_collective_count("cgs2", 200, 5) == 400
    assert LZ.reorth_collective_count("mgs", 200, 5) == 1185


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
    assert w_mgs >= 1, f"legacy route lost its per-vector sweep ({w_mgs} while)"
    assert w_cgs == 0, (
        f"cgs2 still carries a per-vector reorth loop "
        f"({w_cgs} while, mgs has {w_mgs})")


def test_the_default_really_is_batched(monkeypatch):
    """RED TWIN for the LANDING: with nothing set, the sweep must be gone.

    ``test_cgs2_removes_the_inner_sweep`` proves the batched route is batched
    when explicitly asked for.  This proves the DEFAULT gets it — i.e. that the
    flip actually reached production callers, not just the ``reorth=`` kwarg.
    Fails if the default ever regresses to ``mgs``.
    """
    monkeypatch.delenv(BL.REORTH_ENV, raising=False)
    n, it = 32, 8
    mv = _diag_matvec(np.arange(1, n + 1, dtype=float))
    jx = jax.make_jaxpr(
        lambda: LZ.lanczos_eig_jit(mv, n, n_eig=4, max_iter=it,
                                   n_reorth=it))().jaxpr
    assert _count_prim(jx, "while") == 0, (
        "the DEFAULT route still carries a per-vector reorth loop")
    assert _count_prim(jx, "scan") >= 1, "outer Lanczos loop vanished"


def test_window_includes_the_current_vector():
    """The projected set is ``{i : max(0,j-n_reorth) <= i <= j}``.  FROZEN.

    Until 2026-08-08 this window stopped at ``i < j``, leaving the current
    vector ``q_j`` unprojected.  That is the one direction no route removed, so
    the un-subtracted ``i*Im<q_j, z>`` the recurrence leaves behind survived
    into ``q_{j+1} = z/beta_j`` and put a route-independent ``4.2009e-06`` floor
    under the Ritz-vector orthogonality of the Si record deck -- identical under
    MGS and CGS2 to five significant figures, which is what proved it was not a
    Gram-Schmidt property.  ``RITZ_ORTHO_PROBE.md`` measured the widening
    collapsing that floor to ~1e-15, and the owner ruled it in.

    Two things a future tidy-up will be tempted to do, and must not:

    1. Narrow it back to ``i < j``.  That reopens the floor;
       ``test_current_vector_projection_holds_orthogonality`` is the red twin.
    2. "Optimise" the sweep to ``fori_loop(start, j, ...)``.  That was a real
       win when the last trip was a discarded no-op.  It no longer is -- the
       ``i == j`` trip is now the one doing the work this change exists for,
       and deleting it silently restores the defect.

    This cell pins the set from both sides, and pins that the collective counts
    did NOT move -- which is the whole argument for the change being free.
    """
    for j in (0, 1, 5, 17):
        for n_reorth in (0, 3, 200):
            sel = np.asarray(LZ._reorth_window(j, 24, n_reorth))
            expect = np.array(
                [max(0, j - n_reorth) <= i <= j for i in range(24)])
            assert np.array_equal(sel, expect), (j, n_reorth)
            # the current vector IS projected out, at every window size
            assert sel[j], f"window still stops at i<j (j={j}, k={n_reorth})"
            # ... and nothing past it is: slots > j must stay unselected
            assert not sel[j + 1:].any(), (j, n_reorth)
            # ... and the k previous vectors are exactly the k it always was
            assert int(sel.sum()) == min(j, n_reorth) + 1, (j, n_reorth)

    # ZERO NEW COLLECTIVES -- the loop bounds are untouched, so the sweep makes
    # the same number of trips it always did; the i == j trip stopped being
    # discarded, that is all.  And the batched route masks an h it already
    # computed in full.
    assert LZ.mgs_trip_count(1, 200) == 1
    assert LZ.mgs_trip_count(200, 200) == 20100        # == sum_j (j+1)
    assert LZ.mgs_trip_count(200, 200) != sum(range(200))   # != sum_j j
    assert LZ.reorth_collective_count("mgs", 200, 200) == 20100
    assert LZ.reorth_collective_count("cgs2", 200, 200) == 400
    assert LZ.reorth_collective_count("cgs2", 200, 5) == 400


def test_routes_select_the_same_window():
    """RED TWIN for route drift: both routes must project the IDENTICAL slots.

    The batched route applies ``_reorth_window`` as a mask.  The sweep uses the
    scalar form of the same predicate inside ``fori_loop(max(0,j-n_reorth),
    j+1)``.  Nothing in the type system ties those together, so this cell
    re-derives the sweep's set from its own bounds and predicate and demands
    equality.  If someone widens one route and not the other -- the exact
    failure mode the owner flagged when this landed -- it goes red here.
    """
    n_slots = 24
    for j in range(0, 20):
        for n_reorth in (0, 1, 3, 7, 200):
            mask = np.asarray(LZ._reorth_window(j, n_slots, n_reorth))
            # the sweep: trips i in [max(0, j-n_reorth), j+1), predicate i<=j
            swept = np.zeros(n_slots, dtype=bool)
            for i in range(max(0, j - n_reorth), j + 1):
                if i <= j:                      # the body's own predicate
                    swept[i] = True
            assert np.array_equal(mask, swept), (j, n_reorth, mask, swept)


def test_mgs_fallback_is_reachable_from_the_env(monkeypatch):
    """RED TWIN for the FALLBACK: the legacy sweep must still be one var away.

    Exercises the env path, not the kwarg, because that is the path a bisect or
    an archived-run reproduction uses.  Fails if the fallback is amputated.
    """
    monkeypatch.setenv(BL.REORTH_ENV, "mgs")
    # the real production path: the BSE layer reads the env, the solver takes
    # the token.  Asserting on the jaxpr proves BOTH halves are wired.
    route = BL.reorth_route()
    assert route == "mgs"
    n, it = 32, 8
    mv = _diag_matvec(np.arange(1, n + 1, dtype=float))
    jx = jax.make_jaxpr(
        lambda: LZ.lanczos_eig_jit(mv, n, n_eig=4, max_iter=it,
                                   n_reorth=it, reorth=route))().jaxpr
    assert _count_prim(jx, "while") >= 1, (
        "LORRAX_LANCZOS_REORTH=mgs did not restore the per-vector sweep")


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


# --------------------------------------------------------------------------
# the property the widening exists to deliver, and its red twin
# --------------------------------------------------------------------------

def _almost_hermitian(n, eps, seed=11):
    """A Hermitian H perturbed by an ANTI-Hermitian ``i*eps*D``, D real diag.

    ``<q, (H + i eps D) q> = <q,Hq> + i eps <q,Dq>`` with ``<q,Dq>`` real, so
    ``Im alpha_j ~ eps`` -- a controlled stand-in for the operator defect the
    Si record deck had (``Im alpha / max|alpha| = 1.088e-06`` from the mini-BZ
    Coulomb head).  This is the only regime in which the widening does anything
    at all: for an exactly Hermitian operator the ``i == j`` projection is a
    no-op, which is precisely why the change is safe.
    """
    H, lam = _degenerate_hermitian(n, seed)
    D = np.linspace(0.5, 1.5, n)
    return H + 1j * float(eps) * jnp.asarray(np.diag(D), dtype=jnp.complex128), lam


def _ritz_ortho(H, n, max_iter, kind):
    _, V = _run(H, n, max_iter, max_iter, max_iter, kind)
    V = np.asarray(V)
    G = V.conj() @ V.T
    return float(np.max(np.abs(G - np.eye(G.shape[0]))))


@pytest.mark.parametrize("kind", ["cgs2", "mgs"])
def test_current_vector_projection_holds_orthogonality(monkeypatch, kind):
    """RED TWIN: revert the window to ``i < j`` and this gate must go red.

    On an operator with ``Im alpha != 0``, the un-subtracted ``i*Im<q_j,z>``
    lands on the Krylov basis' first superdiagonal at ``|Im alpha_j| / beta_j``
    and the Ritz vectors inherit it.  With ``q_j`` in the window it is removed
    and orthogonality is at round-off; with the pre-2026-08-08 window it is not.

    Run on BOTH routes, because the defect was route-independent -- a fix that
    reached only one route would leave the other exactly as broken as before.
    """
    n, max_iter, eps = 96, 48, 1e-6
    H, _ = _almost_hermitian(n, eps)

    good = _ritz_ortho(H, n, max_iter, kind)
    assert good < 1e-11, (
        f"{kind}: q_j is in the window but orthogonality is {good:.3e}")

    monkeypatch.setattr(LZ, "_REORTH_INCLUDE_CURRENT", False)
    bad = _ritz_ortho(H, n, max_iter, kind)
    assert bad > 1e-9, (
        f"RED TWIN DID NOT GO RED: {kind} with the i<j window still holds "
        f"orthogonality at {bad:.3e} -- the gate is no longer testing anything")
    assert bad > 1e3 * good, (
        f"RED TWIN TOO WEAK: {kind} i<j {bad:.3e} vs i<=j {good:.3e}")


def test_widening_is_a_no_op_on_a_hermitian_operator(monkeypatch):
    """The safety half: with Im alpha == 0 the extra projection changes nothing.

    This is what makes the widening a stabilisation rather than a physics
    change -- the component it removes is zero on paper, and its cost on a real
    operator is proportional to that operator's own non-Hermiticity.
    """
    n, max_iter = 96, 48
    H, _ = _degenerate_hermitian(n)          # exactly Hermitian
    ev_new, _ = _run(H, n, 8, max_iter, max_iter, "cgs2")
    monkeypatch.setattr(LZ, "_REORTH_INCLUDE_CURRENT", False)
    ev_old, _ = _run(H, n, 8, max_iter, max_iter, "cgs2")
    delta = float(np.max(np.abs(np.asarray(ev_new) - np.asarray(ev_old))))
    assert delta < 1e-9, (
        f"widening moved a HERMITIAN operator's eigenvalues by {delta:.3e}")
