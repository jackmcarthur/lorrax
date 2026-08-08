"""``_FEAST_RUNNER_CACHE`` must key on the operator and the data it bakes in.

``bse_feast._get_feast_runner`` returns a ``jax.jit`` closure that CLOSES OVER
``matvec`` and over the ten operand arrays it pulls out of ``data``
(``matvec_operands``).  Closed-over concrete arrays are compile-time constants
of the returned executable, not runtime arguments -- so a runner built for one
operator and one set of arrays computes that operator's answer and nothing
else, forever.

Before this test the cache key carried only the scalar knobs
``(n_quad, n_ritz, max_iter, tol, ry_to_ev, dtype, use_conjugate_symmetry)``.
Neither the operator nor the arrays appeared in it.  ``_FEAST_RUNNER_CACHE`` is
module-global and shared by two drivers that each build their OWN operator and
their own operand set -- ``bse_feast.run_feast_ritz`` (``build_bse_stack_matvec``
or ``build_bse_ring_matvec_full``) and ``bse_pseudopoles._feast_filter``
(``build_bse_ring_matvec``).  With matching scalars the second caller in a
process silently got back the FIRST caller's runner: wrong operator, wrong
arrays, rc=0, plausible-looking numbers.  That is the same failure family as
the recorded -161 eV silent route change in ``docs/services/distrib_la.md``.

The sibling ten lines up already gets this right: ``_get_gmres_solver`` keys on
``id(matvec)`` and stores the matvec beside the solver so the ``id()`` cannot be
recycled by a later object while the entry is live.  This file holds
``_get_feast_runner`` to the same contract, in BOTH directions:

  * a different operator, or a different operand set, must NOT share a runner
    -- and the numbers each returns must be its OWN operator's, checked against
    an independently built reference runner rather than merely "different from
    the other one";
  * an identical (operator, arrays, knobs) call MUST still hit the cache, or
    the fix would have traded a wrong answer for a recompile on every call,
    which is the entire reason the cache exists.

The operators here are deliberately trivial (a diagonal scaling): what is under
test is the cache key, not the physics.  Section 1's five cells are the red
twins -- every one of them FAILS on the old key.  Section 2's two cells pass on
BOTH keys, and that is their job: they are what stops "delete the cache" from
counting as a fix.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness  # noqa: F401  (puts src/ on sys.path)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_feast as BF  # noqa: E402


# --------------------------------------------------------------------------
# fixtures: toy operators + toy operand sets
# --------------------------------------------------------------------------

# The exact ten keys matvec_operands() reads, in its positional order.
_OPERAND_KEYS = ("psi_c_X", "psi_c_Y", "psi_v_X", "psi_v_Y",
                 "eps_c", "eps_v", "W_R", "V_q0", "M_X", "M_Y")

# Runner knobs, held FIXED across every operator below: that is the point.
# Under the old key these scalars WERE the key, so every call below collided
# on one cache entry.
_KNOBS = dict(n_quad=1, n_ritz=1, max_iter=2, tol=1e-10, ry_to_ev=13.6056980659,
              dtype=jnp.complex128, use_conjugate_symmetry=True)


def _make_matvec(scale: float):
    """A distinct callable object per call -- a distinct ``id()``, exactly as
    the two drivers' ``build_bse_*_matvec`` factories produce."""
    def matvec(x, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
               eps_c, eps_v, W_R, V_q0, M_X, M_Y):
        # Diagonal, and it READS an operand, so both halves of the key matter.
        return jnp.asarray(scale, dtype=x.dtype) * psi_c_X * x
    return matvec


def _make_data(fill: float) -> dict:
    """A fresh dict of freshly allocated arrays -- a distinct ``id()`` per
    array, as ``_build_gmres_data_fp32`` / ``ensure_W_R`` produce per driver."""
    return {k: jnp.full((1,), fill, dtype=jnp.complex128) for k in _OPERAND_KEYS}


def _call(runner):
    """Drive a runner on the shape it expects: ``X_batch`` is
    ``(n_ritz, 1, n)`` and ``diag_h`` is one rank lower than ``b`` so the
    preconditioner broadcasts (``_gmres_solve_core`` lines 185-186)."""
    X_batch = jnp.asarray([[[1.0 + 0.0j, 2.0 + 0.0j]]], dtype=jnp.complex128)
    z_nodes = jnp.asarray([0.7 + 0.3j], dtype=jnp.complex128)
    w_weights = jnp.asarray([0.25 + 0.1j], dtype=jnp.complex128)
    diag_h = jnp.asarray([0.1, 0.2], dtype=jnp.float64)
    filtered, _iters = runner(X_batch, z_nodes, w_weights, diag_h)
    return np.asarray(jax.device_get(filtered))


@pytest.fixture(autouse=True)
def _isolate_cache():
    """The cache is module-global; give each cell a clean one and put the
    process's own entries back afterwards."""
    saved = dict(BF._FEAST_RUNNER_CACHE)
    BF._FEAST_RUNNER_CACHE.clear()
    try:
        yield
    finally:
        BF._FEAST_RUNNER_CACHE.clear()
        BF._FEAST_RUNNER_CACHE.update(saved)


def _runner(matvec, data):
    return BF._get_feast_runner(matvec, data, **_KNOBS)


# --------------------------------------------------------------------------
# 1. THE RED TWINS -- every cell here fails on the old key
# --------------------------------------------------------------------------


def test_distinct_operators_do_not_share_a_runner():
    """RED ON THE OLD KEY.  Two operators, identical knobs, same arrays."""
    data = _make_data(1.0)
    r_a = _runner(_make_matvec(1.0), data)
    r_b = _runner(_make_matvec(3.0), data)
    assert r_a is not r_b, (
        "the cache returned operator A's runner for operator B: the key does "
        "not carry the matvec the closure bakes in")


def test_distinct_data_do_not_share_a_runner():
    """RED ON THE OLD KEY.  One operator, two operand sets.

    The operands are pulled out of ``data`` inside the jitted closure, so they
    are baked constants -- a second driver's arrays need a second entry.
    """
    matvec = _make_matvec(1.0)
    r_a = _runner(matvec, _make_data(1.0))
    r_b = _runner(matvec, _make_data(3.0))
    assert r_a is not r_b, (
        "the cache returned driver A's runner for driver B's arrays: the key "
        "does not carry the operands the closure bakes in")


def test_second_driver_gets_its_own_numbers():
    """RED ON THE OLD KEY -- the hazard in its numeric form.

    Mimics the real sequence: ``run_feast_ritz`` builds operator A and calls the
    cache, then ``bse_pseudopoles._feast_filter`` builds operator B with the same
    scalar knobs and calls the same module-global cache.  B's answer must be B's.
    Each arm is checked against an INDEPENDENTLY built reference runner (built
    alone in an empty cache), so the cell cannot be satisfied by merely returning
    something different from A.
    """
    mv_a, data_a = _make_matvec(1.0), _make_data(1.0)
    mv_b, data_b = _make_matvec(3.0), _make_data(5.0)

    BF._FEAST_RUNNER_CACHE.clear()
    ref_a = _call(_runner(mv_a, data_a))
    BF._FEAST_RUNNER_CACHE.clear()
    ref_b = _call(_runner(mv_b, data_b))

    # the two operators are genuinely distinguishable by their output
    assert not np.allclose(ref_a, ref_b), "fixture is void: A and B agree"

    # now the shared-cache sequence: feast first, then pseudopoles
    BF._FEAST_RUNNER_CACHE.clear()
    got_a = _call(_runner(mv_a, data_a))
    got_b = _call(_runner(mv_b, data_b))

    np.testing.assert_allclose(got_a, ref_a, rtol=0, atol=0)
    np.testing.assert_allclose(
        got_b, ref_b, rtol=0, atol=0,
        err_msg="the second driver got the first driver's operator: silent "
                "wrong answer, rc=0")


def test_cache_pins_the_operator_and_operand_references():
    """RED ON THE OLD KEY (the stored value was the bare runner).

    ``id()`` is unique only among LIVE objects.  Keying on ids without holding a
    reference lets a freed matvec's address be recycled by an unrelated later
    object, which resurrects the hazard through the back door.  This is exactly
    why ``_get_gmres_solver`` stores ``(matvec, solver)``.
    """
    matvec, data = _make_matvec(1.0), _make_data(1.0)
    runner = _runner(matvec, data)
    entries = list(BF._FEAST_RUNNER_CACHE.values())
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, tuple), (
        "the cache stores the bare runner, so nothing pins the objects whose "
        "id() the key is built from")
    assert any(v is matvec for v in entry), "cache does not pin the matvec"
    flat = [w for v in entry if isinstance(v, tuple) for w in v]
    assert any(w is data["psi_c_X"] for w in flat), "cache does not pin the operands"
    assert any(v is runner for v in entry), "cache does not hold the runner"


def test_equal_but_distinct_arrays_are_a_miss_not_a_wrong_hit():
    """RED ON THE OLD KEY.

    Two dicts of numerically EQUAL but separately allocated arrays are
    different baked constants as far as XLA is concerned.  A miss here costs one
    compile; a hit would be the hazard.  This records which side of that trade
    the key takes, and that the two runners then agree numerically.
    """
    matvec = _make_matvec(1.0)
    r1 = _runner(matvec, _make_data(1.0))
    r2 = _runner(matvec, _make_data(1.0))
    assert r1 is not r2
    np.testing.assert_allclose(_call(r1), _call(r2))


# --------------------------------------------------------------------------
# 2. THE GREEN SIDE -- the cache must still BE a cache.
#    Both cells pass on the old key too; that is what makes them controls.
# --------------------------------------------------------------------------


def test_identical_call_still_hits_the_cache():
    """The reuse contract.  Same operator object, same array objects, same
    knobs -> the SAME runner, no second trace.  Without this cell the fix could
    have been "never cache anything" and every FEAST iteration would recompile.
    """
    matvec, data = _make_matvec(1.0), _make_data(1.0)
    r1 = _runner(matvec, data)
    r2 = _runner(matvec, data)
    assert r1 is r2
    assert len(BF._FEAST_RUNNER_CACHE) == 1


def test_scalar_knobs_still_separate_entries():
    """The original key's job is preserved, not replaced."""
    matvec, data = _make_matvec(1.0), _make_data(1.0)
    knobs = dict(_KNOBS)
    r1 = BF._get_feast_runner(matvec, data, **knobs)
    knobs["n_ritz"] = _KNOBS["n_ritz"] + 1
    r2 = BF._get_feast_runner(matvec, data, **knobs)
    assert r1 is not r2
