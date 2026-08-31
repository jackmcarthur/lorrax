"""Diagnostics for the MPA-W fit.

The authoritative fit contract is
``docs/theory/THEORY_mpa_implementation.md``.  The papers' fit statistics do
not transfer directly to LORRAX.  Their representability numbers -- the mean count
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

import jax.numpy as jnp

from . import pade_fit


#: The keys :func:`solve_conditioning` serves, and the ``fit_mpa_poles``
#: diagnostic each one IS.  Every entry is a quantity the fit computed on
#: its way to the poles; none needs a second fit.
_CONDITIONING_FROM_DIAG = {
    "cond": "cond_pade",
    "sigma_max": "sigma_max_pade",
    "sigma_min": "sigma_min_pade",
    "backward_error": "backward_error",
    "forward_residual": "max_abs_residual",
    "rel_rms_residual": "rel_rms_residual",
    "rsd_eq28": "rsd_eq28",
    "n_valid": "n_valid",
}


def solve_conditioning(
    W_samples,
    z_samples,
    n_p,
    *,
    rcond=1.0e-13,
    eig=pade_fit.EIG_MODES[0],
):
    """Conditioning and backward error of the DENOMINATOR solve.

    Follows ``solve`` (and ``affine``) so that the number stamped into a
    fit store is the conditioning of the algebra that actually produced
    that store's poles.  Reporting the Pade system's condition beside
    Loewner poles would be a measurement of a matrix nobody inverted.

    Returns a dict with

    ``cond``
        2-norm condition number of the matrix the chosen mode inverts:
        the row-equilibrated ``2n x 2n`` cross-multiplied system for
        the equilibrated, SVD-retained subspace of the ``n x n`` Loewner
        matrix ``L``.  A singular direction discarded by the declared
        ``rcond`` is not reported as if it produced a pole.  This is the
        number that decides whether the
        recovered poles mean anything: the theory plan's ranked risk 6 is
        precisely "Vandermonde/companion conditioning fails at any
        scheduled n_p", and the papers' ``n_p <= 15`` range is not
        evidence of safety.  It is also the number that measured the
        2026-08-10 rung-10 failure -- ``9.02e19`` on the shipped solve,
        past ``1/eps`` in double.
    ``sigma_max``, ``sigma_min``
        The extreme singular values behind ``cond``.
    ``backward_error``
        The relative size of the perturbation to the solved system for
        which the computed answer is the exact solution:
        ``||A y - rhs|| / (||A|| ||y|| + ||rhs||)`` on the equilibrated
        Pade system, and ``||L X - sL|| / (||L|| ||X|| + ||sL||)`` on the
        Loewner reduction.  A small backward error with a large ``cond``
        is the signature to report: the solve was done correctly and the
        ANSWER is still untrustworthy.  That pairing is exactly what the
        rung-10 measurement found (backward error 1.48e-12 at
        ``cond = 9.02e19``), and it is why the repair had to be a change
        of algebra rather than a more careful solve of the same system.
    ``forward_residual``
        ``max_j |W_model(z_j) - W_c(z_j)|`` of the finished fit, i.e.
        after roots, guards and the residue refit.  Kept alongside the
        backward error because they fail independently.

    Parameters
    ----------
    eig
        Which eigensolver the underlying fit uses; one of
        ``pade_fit.EIG_MODES``.

        PLUMBED, NOT ACCEPTED-AND-IGNORED, and the distinction is not
        stylistic.  It is true that ``cond``, ``sigma_max`` and
        ``sigma_min`` are properties of the MATRIX rather than of the
        diagonalization -- they come from the SVD, which no eigensolver
        choice touches -- and the same is true of ``backward_error``,
        which measures the reduction ``L X = sL`` and not its spectrum.
        On those four this argument genuinely makes no difference, and
        the production ladders confirm it: the condition medians are
        bit-identical between the two backends.

        But this function returns EIGHT fields, and the other four --
        ``forward_residual``, ``rel_rms_residual``, ``rsd_eq28`` and
        ``n_valid`` -- are computed from the finished fit, downstream of
        the root-finding, after the guards have run on the roots.  A
        version that swallowed ``eig`` would report the LAPACK path's
        residuals beside a jax_qr store's poles, which is the precise
        shape of the defect this module was already reorganised once to
        remove: a diagnostic that describes a fit nobody performed.

        This argument was missing while both fit entry points had it, so
        a caller forwarding one mode dict to the fit and to its
        diagnostics raised TypeError on the second.  The fix is here
        rather than at the call sites because the call sites were right.
    """

    pade_fit._require_x64()
    pade_fit._check_sample_support(W_samples, z_samples, n_p)
    n = int(n_p)
    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)

    # Every field is something ``fit_mpa_poles`` already computed on its
    # way to the poles; one fit, no second solve.
    _, _, diag = pade_fit.fit_mpa_poles(
        w, z, n, rcond=rcond, eig=eig)
    return {k: diag[v] for k, v in _CONDITIONING_FROM_DIAG.items()}


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
    W_samples, z_samples, n_p, *, holdout=None, rcond=1.0e-13,
    eig=pade_fit.EIG_MODES[0],
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
        w[keep_arr], z[keep_arr], n - 1, rcond=rcond, eig=eig)
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
    eig=pade_fit.EIG_MODES[0],
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

    kw = dict(rcond=rcond, eig=eig)
    Om0, B0, d0 = pade_fit.fit_mpa_poles(w, z, n, **kw)
    Om1, B1, d1 = pade_fit.fit_mpa_poles(w + dw, z, n, **kw)

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

# Restored 2026-08-15 after deletion on a zero-caller grep (cleanup commit
# 3cc20a93): this is an INSTRUMENT, not plumbing. It is the diagnostic that
# saw the 2026-08-10 rung-10 failure in its physical form (49% of |B| mass
# fleeing to >16 eV poles at n_p=10), and it distinguishes "the fit is
# worse" from "the fit is wrong". Instruments are exempt from zero-caller
# deletion; owner-relayed ruling, session 2026-08-15.
#: Hartree -> eV, for the width census only.  The fit itself never
#: converts units; ``Omega_p`` is in whatever unit ``z_samples`` was.
_HA_EV = 27.211386245988


def residue_width_census(
    Omega, B, valid=None, *, edges_ev=(4.0, 16.0), energy_unit="Ha"
):
    """WHERE THE RESIDUE MASS SITS, resolved by fitted width.

    An AGGREGATION over elements, unlike everything else in this module,
    and it is here rather than in a reporting script because it is the
    instrument that saw the 2026-08-10 rung-10 failure in its physical
    form: at ``n_p = 8`` the shipped fit put 1.5 % of ``sum_p |B_p|`` on
    poles wider than 16 eV, and at ``n_p = 10`` it put **49 %** there.
    Two extra poles bought no structure; the fit spent them on modes
    broader than the plasmon itself, which is what an ill-conditioned
    rational fit does with freedom the data cannot determine.  A
    conditioning number alone would not have said that, and a residual
    alone would not have said it either.

    Parameters
    ----------
    Omega, B
        ``(..., n_p)`` -- a batch of fitted pole sets and residues, e.g.
        straight off ``fit_mpa_poles_batched``.
    valid
        ``(..., n_p)`` bool; pruned poles carry ``B_p = 0`` already, so
        this only makes the intent explicit.
    edges_ev
        Width thresholds in eV.  ``Gamma = |Im Omega|``, the fitted
        half-width, on the same convention as the campaign's tables.
    energy_unit
        Unit of ``Omega``; ``"Ha"`` or ``"eV"``.

    Returns
    -------
    dict
        ``mass_fraction_above`` -- one entry per edge, the fraction of
        total ``|B|`` on poles wider than that edge; plus the live-mode
        count and the ``|B|``-weighted width percentiles the campaign's
        census tables quote.
    """

    if energy_unit not in ("Ha", "eV"):
        raise ValueError(
            f"GATE census_energy_unit: energy_unit={energy_unit!r} is not "
            "one of ('Ha', 'eV'). FALSE case: the caller names the unit "
            "Omega is in -- the census thresholds are in eV and the fit "
            "never converted anything.")
    to_ev = _HA_EV if energy_unit == "Ha" else 1.0

    om = jnp.asarray(Omega, dtype=jnp.complex128)
    b = jnp.asarray(B, dtype=jnp.complex128)
    live = jnp.ones(om.shape, dtype=bool) if valid is None else jnp.asarray(
        valid)
    mag = jnp.where(live, jnp.abs(b), 0.0)
    gamma_ev = jnp.abs(jnp.imag(om)) * to_ev
    total = jnp.sum(mag)
    total = jnp.where(total > 0, total, 1.0)

    return {
        "n_live": jnp.sum(live.astype(jnp.int32)),
        "mass_total": jnp.sum(mag),
        "mass_fraction_above": {
            float(e): jnp.sum(jnp.where(gamma_ev > e, mag, 0.0)) / total
            for e in edges_ev
        },
        "gamma_ev_p50": jnp.median(jnp.where(live, gamma_ev, jnp.nan)),
        "gamma_ev_weighted_mean": jnp.sum(mag * gamma_ev) / total,
    }
