"""Gates for ``n_iter_done`` honesty (FIX_solver_robustness.md item 4).

Three of ``solve_bse_sharded``'s four routes run a FIXED iteration budget and
never measure a convergence point.  They used to return ``jnp.int32(max_iter)``
-- the budget wearing a measurement's name, indistinguishable from "converged
after exactly max_iter" to every caller.  They now return
``N_ITER_NOT_MEASURED``, and the one place the two meanings are allowed to meet
is ``iters_reported``.

Cells and the failure each one catches:

* ``test_fixed_routes_return_the_not_measured_sentinel`` + its red twin -- a
  future edit reinstating ``jnp.int32(max_iter)`` in a return position.
* ``test_iters_reported_maps_the_sentinel_onto_the_budget`` + its red twin --
  the sentinel leaking into arithmetic as -1 (``n_done * block_size`` was the
  live example).
* ``test_the_convergence_driven_route_still_measures`` -- the other direction:
  the one honest route must keep reporting a REAL count, so this is not
  "delete the number" dressed up as a fix.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from bse import bse_lanczos as BL
from solvers import lanczos as LZ


def _returns_with_third_element(src):
    """Every ``return a, b, c`` in ``src``, as the source text of ``c``."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            if len(node.value.elts) == 3:
                out.append(ast.unparse(node.value.elts[2]))
    return out


def test_fixed_routes_return_the_not_measured_sentinel():
    """No 3-tuple return in bse_lanczos may hand back the iteration BUDGET."""
    src = inspect.getsource(BL)
    thirds = _returns_with_third_element(src)
    assert thirds, "no 3-tuple returns found -- the detector has gone blind"
    offenders = [t for t in thirds if "max_iter" in t]
    assert not offenders, (
        f"a budget is being returned as n_iter_done: {offenders}")
    assert any("N_ITER_NOT_MEASURED" in t for t in thirds), thirds
    assert BL.N_ITER_NOT_MEASURED < 0, BL.N_ITER_NOT_MEASURED


def test_the_detector_would_catch_a_reinstated_budget():
    """RED TWIN for the cell above: the AST walk must fire on the old form."""
    bad = "def f():\n    return evs, evecs, jnp.int32(max_iter)\n"
    thirds = _returns_with_third_element(bad)
    assert thirds == ["jnp.int32(max_iter)"], thirds
    assert [t for t in thirds if "max_iter" in t], (
        "RED TWIN DID NOT GO RED: the detector no longer recognises the "
        "budget-as-measurement return it exists to catch")

    good = "def f():\n    return evs, evecs, jnp.int32(N_ITER_NOT_MEASURED)\n"
    assert not [t for t in _returns_with_third_element(good) if "max_iter" in t]


@pytest.mark.parametrize("budget", [1, 17, 200])
def test_iters_reported_maps_the_sentinel_onto_the_budget(budget):
    assert BL.iters_reported(BL.N_ITER_NOT_MEASURED, budget) == budget
    assert BL.iters_reported(jnp.int32(BL.N_ITER_NOT_MEASURED), budget) == budget
    # a real measurement passes through untouched
    assert BL.iters_reported(3, budget) == 3
    assert BL.iters_reported(jnp.int32(0), budget) == 0
    # RED TWIN: the bare int() the helper replaces would hand back -1, which
    # is what put "Krylov dim = -1 * block_size" one print away.
    assert int(BL.N_ITER_NOT_MEASURED) == -1
    assert BL.iters_reported(BL.N_ITER_NOT_MEASURED, budget) != -1, (
        "RED TWIN DID NOT GO RED: the helper is passing the sentinel through")


def test_the_convergence_driven_route_still_measures():
    """The honest route must keep reporting a REAL count, not the sentinel.

    This is the cell that stops "return -1 everywhere" counting as a fix.
    """
    n, bs, max_iter = 96, 2, 40
    rng = np.random.default_rng(5)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    lam = np.linspace(1.0, 40.0, n)
    H = jnp.asarray((Q * lam) @ Q.conj().T, dtype=jnp.complex128)
    H = 0.5 * (H + H.conj().T)
    mvb = lambda Vb: (H @ Vb.T).T

    _, _, n_it = jax.block_until_ready(LZ.block_lanczos_eig_jit_converged(
        mvb, n, n_eig=4, block_size=bs, max_iter=max_iter,
        rtol=1e-6, check_every=2, seed=5))
    n_it = int(n_it)
    assert n_it >= 0, f"the adaptive route returned the sentinel ({n_it})"
    assert 1 <= n_it <= max_iter, n_it
    # and it is a MEASUREMENT: a converging problem must stop before the budget
    assert n_it < max_iter, (
        f"the adaptive route ran its whole budget ({n_it}/{max_iter}); this "
        f"cell can no longer tell a measurement from a budget")
