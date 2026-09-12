"""``rcrop_nojit`` — the accelerator the SC loop actually runs.

``sc_accelerator = "rcrop"`` is the default (``gw_config.py``), the SC
loop reaches ``mixing.acceleration.rcrop_nojit`` from
``sc_iteration._run_rcrop``, and before this file nothing in ``tests/``
named it: the one automated SC gate
(``test_invariance_gates.py::test_sc_iteration1_equals_one_shot``) sets
``sc_max_iter = 1``, which returns on ``run_self_consistency``'s
one-shot fast path before an accelerator is constructed, and
``tests/bench/benchmark_synthetic.py`` imports ``rcrop``/``crop``, not
``rcrop_nojit`` (and ``bench`` is in ``norecursedirs``).

Everything here runs against a synthetic residual — no mesh, no
container fixture, no deck — so it belongs in the normal suite.

What is pinned:

* convergence to a nonlinear fixed point whose root is known in closed
  form;
* the circular history: the window handed to the α solve is the last
  ``min(stores, m)`` residuals in CHRONOLOGICAL order with the unfilled
  slots zeroed, asserted across the wrap;
* ``_solve_crop_alpha_stacked`` (the Gram path that replaced the flat
  QR) against a host least-squares reference.  On a NEAR-COLLINEAR
  window the two coefficient vectors genuinely differ — the relative
  ridge damps a near-null direction that the reference's SVD resolves —
  so the assertion there is on the OBJECTIVE ‖Fw·α‖, the quantity the
  iteration consumes, not on α;
* Σα = 1, the affine constraint that makes the update a combination of
  iterates rather than an arbitrary linear map;
* rank-agnosticism: rank 1, 2 and 4 operands take the same trajectory as
  the rank-3 SC carry;
* a negative control — a residual whose plain Picard step ``x + f(x)``
  DIVERGES, so a rewrite that degenerated the update to Picard fails
  here instead of merely converging more slowly.

Requires jax (``mixing.acceleration`` imports it at module scope), so
this runs in the container, not on a login node.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

jax = pytest.importorskip("jax")

import jax.numpy as jnp                                       # noqa: E402

from mixing import acceleration                               # noqa: E402
from mixing.acceleration import rcrop_nojit                   # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixed points
# ---------------------------------------------------------------------------

def _linear_diag(a, x_star):
    """``f(x) = a ⊙ (x − x*)`` — root at ``x*``, Jacobian ``diag(a)``.

    The Picard step is ``x + f(x)``, i.e. ``e ← (1 + a) ⊙ e``: entries
    with ``|1 + a| > 1`` make plain fixed-point iteration diverge in a
    2-cycle, the regime ``run_self_consistency``'s docstring names for
    QSGW on a dense band manifold.
    """
    a = jnp.asarray(a)
    x_star = jnp.asarray(x_star)

    def f(x):
        return a * (x - x_star)

    return f


def _nonlinear(a, x_star, eps):
    """``f(x) = a ⊙ d + eps·d²``, ``d = x − x*``.  Root still exactly ``x*``."""
    a = jnp.asarray(a)
    x_star = jnp.asarray(x_star)

    def f(x):
        d = x - x_star
        return a * d + eps * d * d

    return f


def _contracting(rng, shape):
    """Jacobian entries with ``|1 + a| ≤ 0.3`` — Picard alone converges."""
    return -(0.7 + 0.6 * rng.random(shape)).astype(np.complex128)


def _counted(f):
    """Wrap a residual so the test can replay the calls the solver made."""
    calls = []

    def g(x):
        y = f(x)
        calls.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return y

    return g, calls


def _norm(a):
    return float(np.sqrt(np.sum(np.abs(np.asarray(a)) ** 2)))


# ---------------------------------------------------------------------------
# 1.  Convergence on a known nonlinear fixed point
# ---------------------------------------------------------------------------

def test_converges_to_the_known_nonlinear_root():
    rng = np.random.default_rng(11)
    shape = (2, 3, 4)
    x_star = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex128)
    a = _contracting(rng, shape)
    x0 = x_star + (0.4 * (rng.normal(size=shape)
                          + 1j * rng.normal(size=shape))).astype(np.complex128)

    res = rcrop_nojit(_nonlinear(a, x_star, 0.1),
                      jnp.asarray(x0), m=5, maxit=40, tol=1e-11)

    assert res.converged
    assert float(res.residual_norms[-1]) <= 1e-11
    # The ROOT, not merely a small residual: catches a solver that
    # stalled on a different stationary point of the quadratic term.
    assert _norm(np.asarray(res.x) - x_star) < 1e-10
    assert float(res.residual_norms[-1]) < 1e-6 * float(res.residual_norms[0])


def test_early_exit_when_x0_is_already_the_root():
    x_star = jnp.asarray(np.ones((2, 3), dtype=np.complex128))
    res = rcrop_nojit(_linear_diag(np.full((2, 3), -0.5), x_star),
                      x_star, m=4, maxit=10, tol=1e-12)
    assert res.converged and res.iterations == 0


# ---------------------------------------------------------------------------
# 2.  The circular history, across the wrap
# ---------------------------------------------------------------------------

def _spy_windows(monkeypatch):
    """Record every ``Fw`` the accelerator's α solve is handed."""
    seen = []
    real = acceleration._solve_crop_alpha_stacked

    def spy(Fw):
        alpha = real(Fw)
        seen.append(np.asarray(Fw).copy())
        return alpha

    monkeypatch.setattr(acceleration, "_solve_crop_alpha_stacked", spy)
    return seen


def test_history_window_is_chronological_and_wraps(monkeypatch):
    """The window is the last ``min(stores, m)`` residuals, oldest first.

    Stored entries are ``x0`` plus one per completed iteration, so at the
    start of iteration ``it`` there are ``it + 1`` of them and the buffer
    has wrapped once ``it + 1 > m``.  ``rcrop_nojit`` writes at ``head``
    and rolls by ``(head − filled) % m`` to undo the wrap; this asserts
    the result entry by entry against the residuals the residual function
    actually returned, so an off-by-one in either index is fatal here.
    """
    m, maxit = 3, 6
    rng = np.random.default_rng(3)
    shape = (2, 3)
    x_star = rng.normal(size=shape).astype(np.complex128)
    a = -(0.3 + 0.4 * rng.random(shape)).astype(np.complex128)

    seen = _spy_windows(monkeypatch)
    fn, calls = _counted(_nonlinear(a, x_star, 0.05))
    # tol = 0 so the loop cannot stop on tolerance and the wrap is reached.
    rcrop_nojit(fn, jnp.asarray(x_star + 0.5), m=m, maxit=maxit, tol=0.0)

    nit = len(seen)
    assert nit >= m + 2, f"only {nit} iterations — the history never wrapped"
    # Call order is f(x0), then per iteration f(x_trial), f(x_new).
    assert len(calls) == 1 + 2 * nit
    stored = [calls[0][1]] + [calls[1 + 2 * i + 1][1] for i in range(nit)]

    for it in range(nit):
        Fw = seen[it]
        assert Fw.shape == (m + 1,) + shape
        filled = min(it + 1, m)
        chrono = stored[it + 1 - filled: it + 1]
        for j in range(filled):
            assert np.array_equal(Fw[j], chrono[j]), (
                f"iteration {it}: window slot {j} is not history entry "
                f"{it + 1 - filled + j}")
        for j in range(filled, m):
            assert not np.any(Fw[j]), "unfilled slot is not zeroed"
        # Last column is THIS iteration's trial residual.
        assert np.array_equal(Fw[m], calls[1 + 2 * it][1])


def test_history_depth_one_keeps_only_the_latest(monkeypatch):
    """``m = 1`` is the degenerate wrap: every iteration overwrites slot 0."""
    seen = _spy_windows(monkeypatch)
    rng = np.random.default_rng(5)
    shape = (4,)
    x_star = rng.normal(size=shape).astype(np.complex128)
    fn, calls = _counted(_nonlinear(np.full(shape, -0.6), x_star, 0.05))
    rcrop_nojit(fn, jnp.asarray(x_star + 0.4), m=1, maxit=4, tol=0.0)

    nit = len(seen)
    stored = [calls[0][1]] + [calls[1 + 2 * i + 1][1] for i in range(nit)]
    for it in range(nit):
        assert seen[it].shape == (2,) + shape
        assert np.array_equal(seen[it][0], stored[it])


# ---------------------------------------------------------------------------
# 3.  The α solve: Gram path against a host reference
# ---------------------------------------------------------------------------

def _alpha_reference(Fw):
    """Least squares in the affine coordinates, on the host, via SVD.

    Same formulation as ``_solve_crop_alpha_stacked`` — γ minimises
    ‖f_trial + F_prev γ‖ over the VALID (nonzero) history columns and
    ``α = [γ, 1 − Σγ]`` — but solved with ``lstsq`` instead of a
    normal-equation Gram plus ridge, so it is an independent answer to
    the same problem.
    """
    W = np.asarray(Fw)
    k = W.shape[0] - 1
    f_trial = W[k].reshape(-1)
    hist = W[:k].reshape(k, -1)
    valid = np.linalg.norm(hist, axis=1) > 1e-14
    gamma = np.zeros(k, dtype=np.complex128)
    if valid.any():
        F_prev = (hist[valid] - f_trial[None, :]).T          # (n, n_valid)
        sol, *_ = np.linalg.lstsq(F_prev, -f_trial, rcond=None)
        gamma[valid] = sol
    return np.concatenate([gamma, [1.0 - gamma.sum()]])


def _objective(Fw, alpha):
    """‖Σ_i α_i·Fw_i‖₂ — the quantity the update actually minimises."""
    W = np.asarray(Fw)
    return float(np.linalg.norm(np.asarray(alpha) @ W.reshape(W.shape[0], -1)))


def _window(rng, k, shape, filled):
    """A ``(k+1,) + shape`` window with ``filled`` nonzero history slots."""
    W = np.zeros((k + 1,) + shape, dtype=np.complex128)
    for j in range(filled):
        W[j] = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    W[k] = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return W


@pytest.mark.parametrize("filled", [1, 3, 5])
def test_alpha_matches_the_host_least_squares(filled):
    """Well-conditioned window: α itself must agree, not just its objective."""
    rng = np.random.default_rng(100 + filled)
    W = _window(rng, 5, (3, 4), filled)
    alpha = np.asarray(acceleration._solve_crop_alpha_stacked(jnp.asarray(W)))
    ref = _alpha_reference(W)

    assert np.allclose(alpha, ref, atol=1e-8, rtol=1e-6)
    # Unfilled slots contribute nothing, exactly.
    assert np.all(alpha[filled:5] == 0)
    assert abs(complex(alpha.sum()) - 1.0) < 1e-12


def test_alpha_is_pure_trial_on_an_empty_window():
    """``filled = 0`` is the seed iteration: α = [0, …, 0, 1]."""
    rng = np.random.default_rng(7)
    W = _window(rng, 5, (2, 2), 0)
    alpha = np.asarray(acceleration._solve_crop_alpha_stacked(jnp.asarray(W)))
    assert np.all(alpha[:5] == 0)
    assert abs(complex(alpha[5]) - 1.0) < 1e-14


def test_near_collinear_window_agrees_on_the_objective_not_on_alpha():
    """The window where the QR path and the Gram path answer differently.

    Two history columns separated by 1e-9 make ``F_prev`` numerically
    rank deficient.  The Gram squares that to 1e-18, below the relative
    ridge, so the Gram path damps the near-null direction; the
    reference's SVD keeps it (1e-9 is far above ``lstsq``'s cutoff) and
    resolves it with a coefficient of order 1e3.  Both are legitimate
    regularisations of an ill-posed problem.

    ``f_trial`` is built with only a 1e-6 component along that direction,
    which is the situation a converging CROP window is actually in — the
    residual differences go collinear while the residual itself stays in
    the well-conditioned span — so the two answers differ in α by orders
    of magnitude and in the OBJECTIVE by almost nothing.  The α gap is
    asserted too, so that the test cannot silently degrade into the
    well-conditioned case above.

    The four generators are ORTHONORMAL: with random ones the residual's
    unresolvable part (``b4``) leaks into the near-null direction and the
    exact solve buys a few percent of objective there, which would make
    the objective assertion below about the construction rather than
    about the solver.
    """
    rng = np.random.default_rng(21)
    shape = (4, 4)
    n = int(np.prod(shape))
    Q, _ = np.linalg.qr(
        (rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4))
         ).astype(np.complex128))
    b1, b2, b3, b4 = (Q[:, j].reshape(shape) for j in range(4))

    f_trial = -(2.0 * b1 + 0.5 * b3) + 1e-6 * b2 + 1e-3 * b4
    W = np.stack([f_trial + b1,                 # column c1 = b1
                  f_trial + b1 + 1e-9 * b2,     # column c2 = b1 + 1e-9·b2
                  f_trial + b3,                 # column c3 = b3
                  f_trial])                     # trial residual, last

    alpha = np.asarray(acceleration._solve_crop_alpha_stacked(jnp.asarray(W)))
    ref = _alpha_reference(W)

    assert abs(complex(alpha.sum()) - 1.0) < 1e-10
    obj_gram, obj_ref = _objective(W, alpha), _objective(W, ref)
    assert obj_ref > 0.0
    assert abs(obj_gram - obj_ref) / obj_ref < 1e-2, (
        f"objective moved: gram {obj_gram:.6e} vs reference {obj_ref:.6e}")
    assert float(np.abs(alpha - ref).max()) > 1e-4, (
        "window is not the ill-conditioned case this test is about")


# ---------------------------------------------------------------------------
# 4.  Rank-agnosticism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(24,), (4, 6), (2, 3, 2, 2)])
def test_same_trajectory_at_any_operand_rank(shape):
    """The history keeps the operand's shape; the answer must not notice.

    The rank-3 run is the reference (the SC carry is ``(nk, nb, nb)``).
    Every reduction in the accelerator runs over ALL operand axes, so a
    reshaped operand takes the same trajectory to float round-off; the
    reductions associate differently, which is why the residual history
    is compared with a tolerance and the iteration count is allowed to
    differ by the one step that a boundary crossing can cost.

    ``atol`` carries the tail: the trajectory converges to ~2e-12 in 13
    iterations, where a 1e-16 relative perturbation of the iterate is a
    percents-level relative change in the residual but still an absolute
    change of ~1e-16.  ``rtol`` carries the O(1) head.
    """
    ref_shape = (2, 3, 4)
    n = int(np.prod(shape))
    assert n == int(np.prod(ref_shape))

    rng = np.random.default_rng(31)
    flat_star = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(
        np.complex128)
    flat_a = _contracting(rng, n)
    flat_x0 = flat_star + 0.4 * rng.normal(size=n)

    def run(s):
        return rcrop_nojit(
            _nonlinear(flat_a.reshape(s), flat_star.reshape(s), 0.1),
            jnp.asarray(flat_x0.reshape(s)), m=4, maxit=40, tol=1e-11)

    ref, got = run(ref_shape), run(shape)

    assert ref.converged and got.converged
    assert abs(got.iterations - ref.iterations) <= 1
    k = min(len(ref.residual_norms), len(got.residual_norms))
    assert np.allclose(np.asarray(ref.residual_norms[:k]),
                       np.asarray(got.residual_norms[:k]),
                       rtol=1e-6, atol=1e-11)
    assert np.allclose(np.asarray(got.x).reshape(-1),
                       np.asarray(ref.x).reshape(-1), atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------
# 5.  Negative control: the update is not Picard
# ---------------------------------------------------------------------------

# ``|1 + a| > 1`` for every entry, so the plain fixed-point map diverges
# in a 2-cycle on all of them.
_DIVERGENT_A = np.array([-3.0, -2.6, -2.2, -3.4, -2.8, -2.4],
                        dtype=np.complex128)


def test_picard_diverges_on_this_fixed_point():
    """The control's control: without acceleration this problem blows up."""
    x_star = np.arange(1, 7, dtype=np.complex128)
    f = _linear_diag(_DIVERGENT_A, x_star)
    x = jnp.asarray(x_star + 1.0)
    r0 = _norm(f(x))
    for _ in range(12):
        x = x + f(x)
    assert _norm(f(x)) > 100.0 * r0


def test_rcrop_converges_where_picard_diverges():
    x_star = np.arange(1, 7, dtype=np.complex128)
    res = rcrop_nojit(_linear_diag(_DIVERGENT_A, x_star),
                      jnp.asarray(x_star + 1.0), m=6, maxit=30, tol=1e-10)
    assert res.converged, (
        f"did not converge in {res.iterations} iterations, "
        f"final residual {float(res.residual_norms[-1]):.3e}")
    assert _norm(np.asarray(res.x) - x_star) < 1e-9


def test_first_iterate_is_not_the_picard_step():
    """A rewrite that dropped the α solve would land exactly on ``x + f(x)``.

    On a scalar-multiple residual (``a = −3`` everywhere) the window is
    one-dimensional and the extrapolation is exact in a single iteration
    — α = [2/3, 1/3] carries ``x0`` and the trial point onto the root —
    while Picard doubles the error.  So the first iterate discriminates
    sharply rather than by a margin.
    """
    n = 5
    x_star = np.arange(n, dtype=np.complex128)
    x0 = jnp.asarray(x_star + 1.0)
    f = _linear_diag(np.full(n, -3.0, dtype=np.complex128), x_star)

    res = rcrop_nojit(f, x0, m=5, maxit=1, tol=0.0)
    picard = np.asarray(x0 + f(x0))

    # "Exact" up to the Gram solve's relative 1e-12 ridge, which perturbs
    # γ = 2/3 in the last places: measured ‖x − x*‖ = 4.47e-12 on this
    # window (n = 5, initial error 1 per component).
    assert _norm(np.asarray(res.x) - x_star) < 1e-10
    assert _norm(picard - x_star) > 1.9                   # e ← −2·e
    assert _norm(np.asarray(res.x) - picard) > 1.0


def test_crop_conditioning_guard_is_inert_when_the_gram_is_well_conditioned():
    """The guard must be BIT-IDENTICAL above the floor, not merely close.

    ``mixing.acceleration`` is shared with the incumbent MPA SC route, so a
    guard that perturbs every well-conditioned solve would move numbers on a
    path that has nothing to do with the defect it fixes.  The selector form
    exists precisely so that above the floor the returned array IS the
    historical expression's output, and this asserts that against a
    recomputation of that expression rather than against a tolerance.
    """
    import jax.numpy as jnp
    import numpy as np
    from mixing import acceleration

    rng = np.random.default_rng(20260911)
    k, n = 4, 37
    # A window whose differences are comfortably independent.
    Fw = jnp.asarray(rng.normal(size=(k + 1, n)) + 1j * rng.normal(size=(k + 1, n)))
    alpha = acceleration._solve_crop_alpha_stacked(Fw)

    # The historical expression, recomputed here in full.
    ax = tuple(range(1, Fw.ndim))
    bshape = (k,) + (1,) * (Fw.ndim - 1)
    f_trial, F_hist = Fw[-1], Fw[:k]
    valid = jnp.sqrt(jnp.sum(jnp.abs(F_hist) ** 2, axis=ax)) > 1e-14
    F_prev = jnp.where(valid.reshape(bshape), F_hist - f_trial[None], 0.0 + 0.0j)
    col = jnp.sqrt(jnp.sum(jnp.abs(F_prev) ** 2, axis=ax))
    scale = jnp.where(col > 0.0, col, 1.0)
    F_scaled = F_prev / scale.reshape(bshape)
    G = jnp.tensordot(jnp.conj(F_scaled), F_scaled, axes=(ax, ax))
    b = -jnp.tensordot(jnp.conj(F_scaled), f_trial,
                       axes=(ax, tuple(range(f_trial.ndim))))
    historical = jnp.linalg.solve(G + 1e-12 * jnp.eye(k, dtype=G.dtype), b) / scale
    historical = jnp.where(valid, historical, 0.0 + 0.0j)

    eig = np.asarray(jnp.linalg.eigvalsh(G), np.float64)
    assert eig[0] > acceleration._CROP_CONDITION_FLOOR * eig[-1], (
        "fixture is not well conditioned; the test would prove nothing")
    np.testing.assert_array_equal(np.asarray(alpha[:k]), np.asarray(historical))
    # The affine constraint still holds.
    np.testing.assert_allclose(complex(np.asarray(alpha).sum()), 1.0 + 0.0j,
                               rtol=0, atol=1e-12)


def test_crop_conditioning_guard_damps_a_near_collinear_window():
    """Below the floor the coefficients are bounded instead of amplified.

    Near convergence the residual differences become nearly collinear BY
    CONSTRUCTION, so this is the normal end state of a converging window,
    not a contrived input.  The unit-diagonal Gram's smallest eigenvalue
    falls toward the historical 1e-12 ridge, which is why that ridge never
    guarded anything.
    """
    import jax.numpy as jnp
    import numpy as np
    from mixing import acceleration

    rng = np.random.default_rng(7)
    k, n = 4, 29
    base = rng.normal(size=n) + 1j * rng.normal(size=n)
    f_trial = base * 1e-6
    # History differences collinear with `base` to one part in 1e10.
    rows = [f_trial + base * (1.0 + j) * (1.0 + 1e-10 * rng.normal(size=n))
            for j in range(k)]
    Fw = jnp.asarray(np.stack(rows + [f_trial]))

    alpha = np.asarray(acceleration._solve_crop_alpha_stacked(Fw))
    assert np.all(np.isfinite(alpha))
    # The damped branch keeps the combination bounded; the historical
    # expression on this window amplifies without bound.
    assert np.abs(alpha[:k]).sum() < 1.0e3, np.abs(alpha[:k]).sum()
    np.testing.assert_allclose(complex(alpha.sum()), 1.0 + 0.0j,
                               rtol=0, atol=1e-9)


def test_crop_conditioning_guard_ignores_unfilled_history_slots():
    """A YOUNG window is not an ill-conditioned one.

    An unfilled slot zeroes its whole column, so a window that has simply
    not filled yet has an exactly singular Gram by construction.  Reading
    that as ill-conditioning damps the first iterates and degenerates the
    update to the plain Picard step -- which is what
    ``test_first_iterate_is_not_the_picard_step`` caught on the first cut of
    this guard.  The condition test therefore judges the VALID block only.
    """
    import jax.numpy as jnp
    import numpy as np
    from mixing import acceleration

    rng = np.random.default_rng(31337)
    k, n = 4, 23
    rows = [rng.normal(size=n) + 1j * rng.normal(size=n) for _ in range(k + 1)]
    # Only the first history column is filled; the rest are unfilled slots.
    for j in range(1, k):
        rows[j] = np.zeros(n, complex)
    Fw = jnp.asarray(np.stack(rows))

    alpha = np.asarray(acceleration._solve_crop_alpha_stacked(Fw))
    # The unfilled slots carry no weight ...
    np.testing.assert_array_equal(alpha[1:k], np.zeros(k - 1, complex))
    # ... and the one real column is NOT damped away, so the update is not
    # the plain Picard step (which would be alpha = [0,0,0,0,1]).
    assert abs(alpha[0]) > 1e-6, alpha
    np.testing.assert_allclose(complex(alpha.sum()), 1.0 + 0.0j,
                               rtol=0, atol=1e-12)


# ``rcrop_nojit``'s call sequence into ``residual_fn``, which the fixtures
# below have to target precisely: ONE pre-loop call on x0, then per iteration
# a TRIAL call on ``x + f`` followed by an ACCEPTED call on the CROP-mixed
# ``x_new``.  So while nothing is rejected, calls 2, 4, 6 ... are trials and
# 1, 3, 5 ... are not -- but a REJECTED trial skips its accepted call, after
# which every subsequent call is another trial, so that parity cannot be
# relied on once rejections start.  Only the trial is protected: a refusal on
# the accepted map is a real refusal and must propagate, and getting that
# wrong is how the first cut of the P4 gate planted its failure on the wrong
# call.


def test_a_trial_that_fails_a_physics_gate_is_rejected_not_fatal():
    """A bad EXTRAPOLATION must not throw away the converged maps behind it.

    MEASURED (claim 2189): on Si shared-pole SC eleven maps converged to
    max|dE| 6.3e-4 eV and the twelfth trial -- an rCROP extrapolation --
    tripped ``GATE shared_pole_gram_valid`` at q=0, which aborted the run.
    The plain step from the last accepted point is a fine substitute for a
    refused extrapolation; the run continuing is the whole point.
    """
    import jax.numpy as jnp
    import numpy as np
    from mixing import acceleration

    # NONLINEAR on purpose: CROP solves an affine residual exactly in one
    # step, so an affine fixture converges before it ever reaches a later
    # trial and the plant never fires.
    target = jnp.asarray(np.arange(6, dtype=float) + 1.0) + 0j
    base = _nonlinear(-0.55 * jnp.ones(6), target, 0.35)
    calls = {"n": 0}

    def residual(x):
        calls["n"] += 1
        if calls["n"] == 4:                      # the it=1 TRIAL
            raise ValueError("GATE planted_trial_gate: got: planted; "
                             "want: a finite trial; why: test")
        return base(x)

    result = acceleration.rcrop_nojit(residual, jnp.zeros(6, complex),
                                      m=4, maxit=60, tol=1e-10)
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(target),
                               rtol=0, atol=1e-8)
    assert result.converged
    assert len(result.rejected_trials) == 1, result.rejected_trials
    row = result.rejected_trials[0]
    assert "GATE planted_trial_gate" in row["refusal"], row
    assert row["iteration"] == 1 and row["consecutive"] == 1, row


def test_a_refusal_on_the_ACCEPTED_map_still_propagates():
    """Only the trial is protected; the accepted map is the physics.

    A substituted plain step is an answer for a bad extrapolation. It is not
    an answer for a map that refuses on its own accepted input, and pretending
    otherwise would turn a physics refusal into a silent trajectory change.
    """
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from mixing import acceleration

    target = jnp.asarray(np.arange(6, dtype=float) + 1.0) + 0j
    base = _nonlinear(-0.55 * jnp.ones(6), target, 0.35)
    calls = {"n": 0}

    def residual(x):
        calls["n"] += 1
        if calls["n"] == 3:                      # the it=0 ACCEPTED call
            raise ValueError("GATE planted_accepted_gate: got: planted; "
                             "want: -; why: test")
        return base(x)

    with pytest.raises(ValueError, match="planted_accepted_gate"):
        acceleration.rcrop_nojit(residual, jnp.zeros(6, complex),
                                 m=4, maxit=60, tol=1e-10)


def test_consecutive_trial_rejections_still_refuse_loudly():
    """A rejected trial substitutes the plain step; a broken MAP must not.

    The fallback is only defensible while the refusal is a property of the
    extrapolation. If every trial refuses, the loop has to stop rather than
    spin on a substitute.
    """
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from mixing import acceleration

    target = jnp.asarray(np.arange(4, dtype=float) + 1.0) + 0j
    base = _nonlinear(-0.55 * jnp.ones(4), target, 0.35)
    calls = {"n": 0}

    def every_trial_refuses(x):
        # Refuse EVERYTHING after the pre-loop call.  A rejected trial
        # ``continue``s without making its accepted call, so once the
        # rejections start every subsequent call is another trial and the
        # even/odd table above no longer applies -- which is why this
        # fixture cannot use it.
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("GATE planted_always: got: planted; want: -; why: test")
        return base(x)

    with pytest.raises(RuntimeError) as caught:
        acceleration.rcrop_nojit(every_trial_refuses, jnp.zeros(4, complex),
                                 m=3, maxit=20, tol=1e-10)
    message = str(caught.value)
    assert "GATE sc_trial_rejections_exhausted" in message
    assert "GATE planted_always" in message
    assert str(acceleration._MAX_CONSECUTIVE_TRIAL_REJECTIONS) in message


def test_a_clean_run_reports_no_rejected_trials():
    """The new field must be inert for every run that never rejects one."""
    import jax.numpy as jnp
    import numpy as np
    from mixing import acceleration

    target = jnp.asarray(np.arange(5, dtype=float) + 1.0) + 0j
    result = acceleration.rcrop_nojit(_nonlinear(-0.55 * jnp.ones(5), target, 0.35),
                                      jnp.zeros(5, complex),
                                      m=3, maxit=40, tol=1e-10)
    assert result.rejected_trials == ()
    assert result.converged
