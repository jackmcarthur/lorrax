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
        the ``n x n`` Loewner matrix ``L``.  This is the number that
        decides whether the
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


# Restored 2026-08-15 after deletion on a zero-caller grep (cleanup commit
# 3cc20a93): this is an INSTRUMENT, not plumbing. It is the diagnostic that
# saw the 2026-08-10 rung-10 failure in its physical form (49% of |B| mass
# fleeing to >16 eV poles at n_p=10), and it distinguishes "the fit is
# worse" from "the fit is wrong". Instruments are exempt from zero-caller
# deletion; owner-relayed ruling, session 2026-08-15.
#: Hartree -> eV, for the width census only.  The fit itself never
#: converts units; ``Omega_p`` is in whatever unit ``z_samples`` was.
_HA_EV = 27.211386245988

