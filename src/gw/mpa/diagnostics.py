"""Diagnostics for the MPA-W fit.

STAGING LOCATION; the minimax-service design decides the final home.

Why this module exists at all: the theory plan (``~/MPA_THEORY_PLAN.md``
section B, "Fit algebra") is explicit that the papers' fit statistics do
NOT transfer to LORRAX.  Their representability numbers -- the mean count
of corrected matrix elements and the relative standard deviation of the
extrapolated response, defined in the metals paper's supplemental section
III -- are plane-wave statistics collected in an orthonormal basis.  The
ISDF basis is nonorthogonal, so element magnitudes and the meaning of an
"element" both differ.  The plan therefore asks for four diagnostics that
stand on their own:

* condition number and backward error of the linear solve,
* held-out complex-frequency residuals,
* perturbation-refit propagation of certified sample-error vectors,

with the distributional cuts (diagonal vs off-diagonal, norm-resolved)
left to whatever reporting layer the design review lands, since they are
aggregations over elements rather than per-element quantities.

Everything here is a pure function over one element's samples, in the
same shape contract as ``pade_fit.fit_mpa_poles``, so the same ``vmap``
applies.
"""

import jax
import jax.numpy as jnp

from . import pade_fit


def solve_conditioning(W_samples, z_samples, n_p, *, rcond=1.0e-13):
    """Conditioning and backward error of the Pade-in-z^2 linear solve.

    Returns a dict with

    ``cond``
        2-norm condition number of the row-equilibrated cross-multiplied
        system.  This is the number that decides whether the companion
        roots mean anything: the theory plan's ranked risk 6 is precisely
        "Vandermonde/companion conditioning fails at any scheduled n_p",
        and the papers' ``n_p <= 15`` range is not evidence of safety.
    ``sigma_max``, ``sigma_min``
        The extreme singular values behind ``cond``.
    ``backward_error``
        ``||A y - rhs|| / (||A|| ||y|| + ||rhs||)`` on the equilibrated
        system -- the relative size of the perturbation to ``(A, rhs)``
        for which the computed ``y`` is the exact solution.  A small
        backward error with a large ``cond`` is the signature to report:
        the solve was done correctly and the ANSWER is still untrustworthy.
    ``forward_residual``
        ``max_j |W_model(z_j) - W_c(z_j)|`` of the finished fit, i.e.
        after roots, guards and the residue refit.  Kept alongside the
        backward error because they fail independently.
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

    Omega, B, diag = pade_fit.fit_mpa_poles(w, z, n, rcond=rcond)
    return {
        "cond": cond,
        "sigma_max": s_max,
        "sigma_min": s_min,
        "backward_error": num / den,
        "forward_residual": diag["max_abs_residual"],
        "rel_rms_residual": diag["rel_rms_residual"],
        "n_valid": diag["n_valid"],
    }


def default_holdout_indices(n_p):
    """The two samples held out by default: one per line, mid-line.

    A held-out point must probe INTERPOLATION, not extrapolation, so the
    endpoints (``omega = 0`` and ``omega = omega_m``) are deliberately not
    chosen: dropping either turns the test into an extrapolation test and
    reports a residual that says more about the partition edge than about
    the fit.  ``n_p // 2`` on each line is the mid-partition point.
    """

    n = int(n_p)
    if n < 2:
        raise ValueError(
            f"GATE holdout_support: n_p={n} leaves no poles after holding "
            "two samples out (the reduced fit uses n_p - 1 poles against "
            "2*n_p - 2 samples). FALSE case: int(n_p) >= 2.")
    k = n // 2
    return (k, n + k)


def holdout_residual(
    W_samples, z_samples, n_p, *, holdout=None, rcond=1.0e-13
):
    """Fit on ``2*n_p - 2`` samples, evaluate on the 2 held out.

    The counts work out exactly: dropping two samples leaves ``2*n_p - 2``
    points, which is the full sample support of an ``n_p - 1`` pole fit.
    So this is a genuine one-pole-lighter refit on a strict subset of the
    grid, not a rank-deficient version of the same fit.

    Returns a dict with the held-out sample indices, the model values
    there, the reference values, the absolute and relative errors, and
    the reduced fit's own diagnostics.  The relative error is normalised
    by ``max_j |W_c(z_j)|`` over the FULL grid, so it stays comparable
    across elements of wildly different magnitude -- the nonorthogonal
    ISDF basis makes per-point normalisation meaningless.
    """

    pade_fit._require_x64()
    pade_fit._check_sample_support(W_samples, z_samples, n_p)
    n = int(n_p)
    idx = default_holdout_indices(n) if holdout is None else tuple(holdout)
    if len(idx) != 2:
        raise ValueError(
            f"GATE holdout_pair: holdout={holdout!r} does not name exactly "
            "two samples. FALSE case: len(holdout) == 2 -- the reduced fit "
            "drops one pole, which frees exactly two sample slots.")
    i, j = int(idx[0]), int(idx[1])
    if not (0 <= i < 2 * n) or not (0 <= j < 2 * n) or i == j:
        raise ValueError(
            f"GATE holdout_distinct_in_range: holdout=({i}, {j}) is not a "
            f"pair of distinct indices in [0, {2 * n}). FALSE case: two "
            "distinct valid sample indices.")

    keep = [k for k in range(2 * n) if k not in (i, j)]
    keep_arr = jnp.asarray(keep, dtype=jnp.int32)
    held_arr = jnp.asarray([i, j], dtype=jnp.int32)

    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)

    Omega, B, diag = pade_fit.fit_mpa_poles(
        w[keep_arr], z[keep_arr], n - 1, rcond=rcond)
    model = pade_fit.eval_mpa_model(
        Omega, B, z[held_arr], valid=diag["valid"])
    ref = w[held_arr]
    scale = jnp.maximum(jnp.max(jnp.abs(w)), jnp.finfo(jnp.float64).tiny)

    return {
        "holdout_indices": held_arr,
        "model": model,
        "reference": ref,
        "abs_error": jnp.abs(model - ref),
        "max_abs_error": jnp.max(jnp.abs(model - ref)),
        "max_rel_error": jnp.max(jnp.abs(model - ref)) / scale,
        "reduced_cond": diag["cond_pade"],
        "reduced_n_valid": diag["n_valid"],
    }


def _match_poles(reference, moved, valid_ref, valid_moved):
    """Greedy nearest-neighbour pole matching; returns per-pole distance.

    Poles come back from ``fit_mpa_poles`` sorted by ``Re Omega``, so a
    perturbation small enough not to reorder them matches index-to-index.
    A perturbation large enough to REORDER them has, by definition,
    already moved a pole past its neighbour, and the index-wise distance
    is then the honest (large) number to report rather than something a
    matching heuristic could flatter.  Dead poles on either side are
    reported as ``inf`` movement so they cannot be silently ignored.
    """

    live = valid_ref & valid_moved
    dist = jnp.abs(moved - reference)
    return jnp.where(live, dist, jnp.inf), live


def perturbation_refit(
    W_samples,
    z_samples,
    n_p,
    error_vector,
    *,
    rcond=1.0e-13,
):
    """Refit under a certified sample-error vector; report the movement.

    This is the plan's "perturbation-refit tests that propagate certified
    sample-error vectors".  ``error_vector`` is the certified error of the
    ``W_c(z_j)`` evaluation -- one complex number per sample, in the same
    units as ``W_samples``, NOT a relative fraction.  The caller supplies
    it; this harness only propagates it.

    Returns a dict with the unperturbed and perturbed pole sets, the
    per-pole movement of ``Omega`` and ``B``, the maxima over live poles,
    and the movement of the fitted widths on their own -- the width is
    what the Sigma stage's crossing core and Laplace complements consume,
    so a fit that keeps ``Re Omega`` still while ``Gamma`` walks is a
    failure this harness must not average away.

    READING THE PER-POLE ARRAYS.  ``d_omega``, ``d_energy``, ``d_gamma``
    and ``d_residue`` carry ``inf`` on any pole that is dead on either
    side, so a pole the perturbation killed cannot be mistaken for a pole
    that did not move.  The ``max_*`` scalars are maxima over LIVE poles
    only and are therefore finite; read them together with
    ``valid_count_change``, which is nonzero exactly when the
    perturbation changed how many poles survived the guards.

    The plan's acceptance thresholds (under 2 meV maximum ladder-QP
    movement, under 5 meV at the 99th percentile of significant poles)
    are stated in QP energy, downstream of this function; they cannot be
    evaluated here and are deliberately not asserted here.
    """

    pade_fit._require_x64()
    pade_fit._check_sample_support(W_samples, z_samples, n_p)
    n = int(n_p)
    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)
    dw = jnp.asarray(error_vector, dtype=jnp.complex128)
    if tuple(jnp.shape(dw)) != (2 * n,):
        raise ValueError(
            f"GATE error_vector_shape: error_vector has shape "
            f"{tuple(jnp.shape(dw))} against {2 * n} samples. FALSE case: "
            "error_vector.shape == (2*n_p,) -- one certified complex error "
            "per sample, absolute and in the units of W_samples.")

    Om0, B0, d0 = pade_fit.fit_mpa_poles(w, z, n, rcond=rcond)
    Om1, B1, d1 = pade_fit.fit_mpa_poles(w + dw, z, n, rcond=rcond)

    d_omega, live = _match_poles(Om0, Om1, d0["valid"], d1["valid"])
    d_b, _ = _match_poles(B0, B1, d0["valid"], d1["valid"])
    d_gamma = jnp.where(
        live, jnp.abs(jnp.imag(Om1) - jnp.imag(Om0)), jnp.inf)
    d_energy = jnp.where(
        live, jnp.abs(jnp.real(Om1) - jnp.real(Om0)), jnp.inf)

    finite = jnp.where(live, d_omega, 0.0)
    return {
        "Omega_unperturbed": Om0,
        "Omega_perturbed": Om1,
        "B_unperturbed": B0,
        "B_perturbed": B1,
        "valid_both": live,
        "d_omega": d_omega,
        "d_energy": d_energy,
        "d_gamma": d_gamma,
        "d_residue": d_b,
        "max_d_omega": jnp.max(finite),
        "max_d_energy": jnp.max(jnp.where(live, d_energy, 0.0)),
        "max_d_gamma": jnp.max(jnp.where(live, d_gamma, 0.0)),
        "max_d_residue": jnp.max(jnp.where(live, d_b, 0.0)),
        "valid_count_change": d1["n_valid"] - d0["n_valid"],
        "perturbation_norm": jnp.linalg.norm(dw),
    }


def diagnostics_batched(fn, W_tile, z_samples, n_p, **kwargs):
    """``vmap`` any of the above over an ``(n_elements, 2*n_p)`` tile.

    ``fn`` is one of ``solve_conditioning``, ``holdout_residual`` or
    ``perturbation_refit``.  Extra keyword arguments are passed through
    unmapped, so a single certified error vector applies to every element
    -- which is the physically meaningful case, the error being a
    property of the quadrature that produced the samples, not of the
    element.
    """

    tile = jnp.asarray(W_tile, dtype=jnp.complex128)
    if tile.ndim != 2:
        raise ValueError(
            f"GATE W_tile_rank: W_tile has shape {tuple(tile.shape)}. FALSE "
            "case: W_tile.ndim == 2, i.e. (n_elements, 2*n_p).")
    return jax.vmap(lambda row: fn(row, z_samples, n_p, **kwargs))(tile)
