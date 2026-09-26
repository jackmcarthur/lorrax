"""Gates for the Lanczos reorthogonalisation route: CGS2, the only route.

Batched classical Gram-Schmidt, twice: every overlap of a sweep computed as
one matrix product, ``2 * max_iter`` all-reduces of an ``(m,)`` vector.
``LORRAX_LANCZOS_REORTH`` is retired and refuses by name.

* ``test_retired_env_refuses_by_name`` -- the retired dial cannot be set
  silently.
* ``test_the_default_really_is_batched`` -- structural, at jaxpr level: no
  per-vector reorth loop.
* ``test_record_deck_collective_counts_are_pinned`` -- the 400 all-reduces of
  the Si record deck (200 iterations).
* ``test_window_includes_the_current_vector`` +
  ``test_widening_is_a_no_op_on_a_hermitian_operator`` -- the 2026-08-08
  widening of the window from ``i < j`` to ``i <= j``: the set it selects,
  and that it moves no eigenvalue of a Hermitian operator.
* ``test_cgs2_full_reorth_solves_the_distinct_spectrum`` and
  ``test_cgs2_orthogonality`` -- accuracy and the property reorth delivers.
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


def test_retired_env_refuses_by_name(monkeypatch):
    """``LORRAX_LANCZOS_REORTH`` is retired: any value refuses, naming it."""
    for value in ("mgs", "cgs2"):
        monkeypatch.setenv("LORRAX_LANCZOS_REORTH", value)
        with pytest.raises(ValueError, match="LORRAX_LANCZOS_REORTH is retired"):
            BL.solve_bse_sharded({}, None, tda=True)


# --------------------------------------------------------------------------
# the collective-count arithmetic this campaign quotes
# --------------------------------------------------------------------------


def test_record_deck_collective_counts_are_pinned(monkeypatch):
    """200 Lanczos iterations on the Si record deck issue 400 reorth all-reduces.

    The count is two per iteration whatever the window.
    """
    assert LZ.reorth_collective_count(200) == 400


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


def test_the_default_really_is_batched(monkeypatch):
    """The Lanczos loop lowers to ``scan``; no per-vector ``while`` sweep remains.

    Stated at jaxpr level so no XLA version can move it.
    """
    n, it = 32, 8
    mv = _diag_matvec(np.arange(1, n + 1, dtype=float))
    jx = jax.make_jaxpr(
        lambda: LZ.block_lanczos_eig_jit(lambda V: mv(V[0])[None], n, n_eig=4,
                                         block_size=1, max_iter=it,
                                         n_reorth=it))().jaxpr
    assert _count_prim(jx, "while") == 0, (
        "the DEFAULT route still carries a per-vector reorth loop")
    assert _count_prim(jx, "scan") >= 1, "outer Lanczos loop vanished"


def test_window_includes_the_current_vector():
    """The projected set is ``{i : max(0,j-n_reorth) <= i <= j}``.  FROZEN.

    Until 2026-08-08 this window stopped at ``i < j``, leaving the current
    vector ``q_j`` unprojected, so the un-subtracted ``i*Im<q_j, z>`` the
    recurrence leaves behind survived into ``q_{j+1} = z/beta_j`` and put a
    ``4.2009e-06`` floor under the Ritz-vector orthogonality of the Si record
    deck.  ``RITZ_ORTHO_PROBE.md`` measured the widening collapsing that floor
    to ~1e-15, and the owner ruled it in.  That floor belonged to the retired
    single-vector kernel; the block kernel subtracts the full complex alpha,
    and the ``i == j`` slot stays in the window as a free re-projection.

    This cell pins the set from both sides, and pins that the collective count
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

    # ZERO NEW COLLECTIVES: the batched route masks an h it already computed
    # in full.
    assert LZ.reorth_collective_count(200) == 400


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


def _run(H, n, n_eig, max_iter, n_reorth):
    mv = lambda v: H @ v
    return jax.block_until_ready(LZ.block_lanczos_eig_jit(
        lambda V: mv(V[0])[None], n, n_eig=n_eig, block_size=1,
        max_iter=max_iter, n_reorth=n_reorth, seed=3))


@pytest.mark.parametrize("n,max_iter", [(96, 48)])
def test_cgs2_full_reorth_solves_the_distinct_spectrum(n, max_iter):
    H, lam = _degenerate_hermitian(n)
    ev_c, _ = _run(H, n, 8, max_iter, max_iter)
    # A single-vector Lanczos
    # sees each DISTINCT eigenvalue once however high its multiplicity (the
    # Krylov space of one start vector meets each eigenspace in one direction),
    # so the reference is the distinct spectrum, and only the well-separated
    # bottom of it is converged at max_iter = n/2.
    distinct = np.unique(lam)
    err = float(np.max(np.abs(np.asarray(ev_c)[:3] - distinct[:3])))
    assert err < 1e-8, f"lowest 3 Ritz values off by {err:.3e}"


def test_cgs2_orthogonality():
    """The Krylov basis (rotated into the Ritz frame) stays orthonormal."""
    n, max_iter = 96, 48
    H, _ = _degenerate_hermitian(n)
    _, V = _run(H, n, max_iter, max_iter, max_iter)
    V = np.asarray(V)
    G = V.conj() @ V.T
    err = float(np.max(np.abs(G - np.eye(G.shape[0]))))
    assert err < 1e-10, f"cgs2 orthogonality {err:.3e}"


# --------------------------------------------------------------------------
# the widening is safe on a Hermitian operator
# --------------------------------------------------------------------------


def test_widening_is_a_no_op_on_a_hermitian_operator(monkeypatch):
    """The safety half: with Im alpha == 0 the extra projection changes nothing.

    This is what makes the widening a stabilisation rather than a physics
    change -- the component it removes is zero on paper, and its cost on a real
    operator is proportional to that operator's own non-Hermiticity.
    """
    n, max_iter = 96, 48
    H, _ = _degenerate_hermitian(n)          # exactly Hermitian
    ev_new, _ = _run(H, n, 8, max_iter, max_iter)
    monkeypatch.setattr(LZ, "_REORTH_INCLUDE_CURRENT", False)
    ev_old, _ = _run(H, n, 8, max_iter, max_iter)
    delta = float(np.max(np.abs(np.asarray(ev_new) - np.asarray(ev_old))))
    assert delta < 1e-9, (
        f"widening moved a HERMITIAN operator's eigenvalues by {delta:.3e}")
