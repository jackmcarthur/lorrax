"""The Pade solve's conditioning, WITHOUT refitting the element to get it.

STAGING LOCATION: beside ``pade_fit`` and ``diagnostics``, whose seam this
module exists to close; the minimax-service design decides the final home
for all three together.

THE DEFECT THIS RETIRES, which ``fit_driver`` has admitted in its own
docstring since the driver landed.  ``mpa_store.write_fit_block``
requires ``condition`` and ``backward_error`` beside every block of
poles, because the Sigma stage's certification refuses poles that fail
its gates and a pole whose conditioning nobody recorded can only be
trusted, never refused.  ``pade_fit.fit_mpa_poles`` returns the
condition number and no backward error.  The only supplier of the
backward error was ``diagnostics.solve_conditioning`` — and that
function, to report a forward residual beside it, runs a COMPLETE
SECOND FIT of the same element: a second companion-root eigvals, a
second pair of residue least-squares solves, a second guard pass.

The driver then throws that second fit's three outputs away.  It reads
``cond`` and ``backward_error`` from ``solve_conditioning`` and takes
``residual`` and ``n_valid`` from the FIRST fit, which already returned
them.  So the second fit was computed for nobody, at a measured 55.2 of
the block's 120.1 microseconds per element -- measured on the production
W_c store, one A100, BFC@0.85, on the 78 960-element column block the
production walk actually takes.

WHAT THIS MODULE DOES INSTEAD.  :func:`solve_conditioning_only` is
``solve_conditioning`` with the second fit deleted and NOTHING ELSE
CHANGED: the same ``build_pade_system``, the same row equilibration, the
same ``_solve_normalised`` call with the same ``rcond``, the same
numerator and denominator, in the same order.  It is therefore not an
approximation of the shipped diagnostic — it is the same expression
tree, and the values it returns are bit-identical to the ones the
shipped path wrote.  That claim is a gate, not a hope:
``tests/test_mpa_fit_one_fit.py`` asserts byte-equality of ``cond`` and
``backward_error`` against ``diagnostics.solve_conditioning`` on the
same samples, and the production evidence checks it again on a real
column block.

THE ONE SOLVE THAT IS STILL PAID TWICE, and where it goes.  After this
module the block costs ONE fit and TWO solves of the Pade-in-z^2 system:
the fit's own, and the standalone equilibrated one here, which exists
only because the backward error needs the solution vector ``y`` and
``fit_mpa_poles`` does not return it.  A three-line change in
``pade_fit`` — reporting ``backward_error`` beside the ``cond_pade`` it
already computes — collapses that to one, and :func:`conditioning_for_block`
is written to TAKE IT THE MOMENT IT APPEARS: it asks the fit's own
diagnostics whether they carry a backward error and only solves again
when they do not.  The branch changes which arithmetic is performed and
not what is returned; both arms compute the same quantity from the same
system, one having kept it and the other recomputing it.  ``pade_fit``
is another lane's file this week, so the change is proposed there rather
than made here, and this module is the seam that makes the fold a
one-liner instead of a coordination.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import pade_fit

__all__ = [
    "conditioning_batched",
    "conditioning_for_block",
    "solve_conditioning_only",
]

#: The key ``pade_fit.fit_mpa_poles`` would carry if it reported the
#: backward error itself.  Named once, here, so the seam has one spelling.
FIT_BACKWARD_ERROR_KEY = "backward_error"

#: The key the fit already carries for the Pade solve's condition number.
#: ``diagnostics.solve_conditioning`` computes the same number from the
#: same system and calls it ``cond``; they are bit-identical because they
#: are the same call, which is why this module can read either.
FIT_CONDITION_KEY = "cond_pade"


def solve_conditioning_only(W_samples, z_samples, n_p, *, rcond=1.0e-13):
    """``cond``, ``sigma_max``, ``sigma_min``, ``backward_error``.

    ``diagnostics.solve_conditioning`` minus its ``fit_mpa_poles`` call —
    op for op identical in everything it still computes, so the four
    values are bit-identical to that function's.  What is missing is the
    three keys the driver never read: ``forward_residual``,
    ``rel_rms_residual`` and ``n_valid``, all three of which the fit
    itself returns (as ``max_abs_residual``, ``rel_rms_residual`` and
    ``n_valid``) to the caller who wants them.

    Pure function of ONE element's samples, in ``pade_fit``'s shape
    contract, so the same ``vmap`` applies; see
    :func:`conditioning_batched`.
    """

    pade_fit._require_x64()
    pade_fit._check_sample_support(W_samples, z_samples, n_p)
    n = int(n_p)
    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)

    A, rhs, _, _ = pade_fit.build_pade_system(w, z, n)
    row_norm = jnp.linalg.norm(A, axis=1)
    row_norm = jnp.where(row_norm > 0, row_norm, 1.0)
    A_n = A / row_norm[:, None]
    rhs_n = rhs / row_norm

    y, cond, s_max, s_min = pade_fit._solve_normalised(A, rhs, rcond)
    num = jnp.linalg.norm(A_n @ y - rhs_n)
    den = (jnp.linalg.norm(A_n) * jnp.linalg.norm(y)
           + jnp.linalg.norm(rhs_n))
    den = jnp.where(den > 0, den, 1.0)

    return {
        "cond": cond,
        "sigma_max": s_max,
        "sigma_min": s_min,
        "backward_error": num / den,
    }


def conditioning_batched(W_tile, z_samples, n_p, *, rcond=1.0e-13):
    """:func:`solve_conditioning_only`, vmapped over the element axis.

    ``W_tile`` is ``(n_elements, 2*n_p)``, the same tile
    ``pade_fit.fit_mpa_poles_batched`` takes.  Deliberately a plain
    ``jax.vmap`` and NOT ``jax.jit``: the probe measured that jitting
    this path buys about one percent (the block is one vmapped sequence,
    so eager dispatch overhead is already amortised over tens of
    thousands of elements) and costs bit-identity, which is the gate this
    restructure is held to.
    """

    tile = jnp.asarray(W_tile, dtype=jnp.complex128)
    if tile.ndim != 2:
        raise ValueError(
            f"GATE W_tile_rank: W_tile has shape {tuple(tile.shape)}. FALSE "
            "case: W_tile.ndim == 2, i.e. (n_elements, 2*n_p).")
    return jax.vmap(
        lambda row: solve_conditioning_only(row, z_samples, n_p,
                                            rcond=rcond))(tile)


def fit_reports_backward_error(fit_diag):
    """True when the fit already carried the backward error itself.

    The seam described in the module docstring, asked as a question about
    the object in hand rather than about a version number: a fit whose
    diagnostics carry ``backward_error`` has already done the solve this
    module would otherwise repeat.
    """

    return FIT_BACKWARD_ERROR_KEY in fit_diag


def conditioning_for_block(W_tile, z_samples, n_p, fit_diag, *,
                           rcond=1.0e-13):
    """``(diag, pade_solves_per_element)`` for one fitted block.

    ``fit_diag`` is the diagnostics pytree ``fit_mpa_poles_batched``
    returned for THIS tile.  If it already carries a backward error, this
    costs nothing and the block's Pade solve count is one; otherwise the
    equilibrated solve is run once more here and the count is two.  The
    count is returned rather than assumed because ``fit_driver``'s cost
    report states it, and a cost report that states a number the code
    stopped paying is worse than one that states none.

    The returned dict always carries ``cond`` and ``backward_error``,
    which are the two diagnostics ``mpa_store.write_fit_block`` requires.
    """

    if fit_reports_backward_error(fit_diag):
        return (
            {
                "cond": fit_diag[FIT_CONDITION_KEY],
                "backward_error": fit_diag[FIT_BACKWARD_ERROR_KEY],
            },
            1,
        )
    return conditioning_batched(W_tile, z_samples, n_p, rcond=rcond), 2
