"""The MPA-W fit kernel: 2*n_p complex samples of W_c -> n_p complex poles.

STAGING LOCATION; the minimax-service design decides the final home.
Landed ahead of that design review because the method is settled by
``~/MPA_THEORY_PLAN.md`` section B ("The fit stage").  Placement under
``src/gw/mpa`` is a parking spot, not a ruling.

The model
---------
``W_c`` is fitted DIRECTLY -- never the polarizability ``X``, whose
independently fitted pole sets would be scrambled by the Coulomb factors
(theory plan section B).  Elementwise in the ISDF basis, the model is the
even-in-z multipole form of the papers::

    W_c(z) = sum_p  2 * Omega_p * B_p / (z**2 - Omega_p**2)
           = sum_p  [ B_p/(z - Omega_p) - B_p/(z + Omega_p) ]

with ``Omega_p = a_p - i*Gamma_p``, ``a_p > 0``, ``Gamma_p > 0``.  The
``Gamma_p`` are fitted physical widths and are never rounded downstream;
the time-domain consumer reads ``W(tau) = sum_p B_p exp(-i Omega_p tau)``
(theory plan section A), which is why ``Re Omega_p > 0`` and
``Im Omega_p < 0`` are structural and not cosmetic: a leaked
``Im Omega_p > 0`` pole enters the tau stage as ``exp(+|Im Omega| tau)``.

Sources
-------
* ``~/MPA_THEORY_PLAN.md`` section B -- the binding LORRAX spec: fit
  ``W_c`` directly, normalised Pade-in-z^2 linear solve with z_max
  scaling, companion-matrix roots, all-2*n_p-point complex least-squares
  residues, the published guards, mandatory residue refit after any
  correction.
* Leon *et al.*, Phys. Rev. B **107**, 155130 (2023) -- markdown copy at
  ``~/projects/lorrax_perlmutter_salvage/reports/multipole/Multipole-W-metals.md``.
  Section II B Eq. (7) is the model; section II C Eq. (9) is the
  interpolation condition on ``2*n_p`` samples.
* Leon, Berland, Cardoso (2025), same directory,
  ``multipole-sigma-2025.md``, Supplemental section I -- the linear
  Pade solver (Eqs. S6, S9-S16) and the companion matrix (Eq. S8), given
  there for MPA-Sigma and used here in its even-in-z (Pade-in-z^2) form.
* The metals paper's SUPPLEMENTAL MATERIAL, section III, Eq. (S18) --
  the reflection guard.  NOT present in the markdown copies in this
  tree; retrieved from the paper library (library-rag record
  ``1095400688``, "SI.pdf", section "III. FREQUENCY REPRESENTABILITY OF
  COPPER"), which states the MPA generalisation of the GN
  unfulfilled-mode condition verbatim as::

      Omega_n = sqrt(Omega_n**2)            if Re[Omega_n**2] >= 0
              = sqrt(-(Omega_n**2)*)        if Re[Omega_n**2] <  0

  It is "a generalized condition that avoids reassigning the poles with
  a constant value", i.e. the fix for the pathology that made 48% of Cu
  matrix elements unfulfilled modes under GN-PPA.

The denominator solve, and why there are two of it
--------------------------------------------------
The model above fixes WHAT is fitted.  It does not fix the algebra that
finds the poles, and on 2026-08-10 that distinction stopped being
academic.  The papers' route -- cross-multiply, solve a ``2n_p x 2n_p``
system whose columns are powers of ``x``, take companion-matrix roots --
is a Vandermonde solve in disguise, and on the production Si deck its
condition number ran ``1.34e16`` at ``n_p = 8`` and ``9.02e19`` at
``n_p = 10``, past ``1/eps`` in double.  The symptom was not noise: the
guards went quiet (prune rate ``3.9e-08``, not one element pruned), the
backward error stayed at ``1.5e-12`` -- the solve was being done
correctly -- and yet the fit's sample residual ROSE from ``n_p = 8`` to
``n_p = 10`` while 49 % of the residue mass moved onto poles broader
than 16 eV.  A backward-stable solve of a problem that has itself gone
singular returns the exact answer to a nearby question, and the nearby
question was no longer this one.

So the pole-finding is selectable (:data:`SOLVE_MODES`) while everything
else -- the model, the guards, the canonical sort, the mandatory residue
refit, the returned ``(B_p, Omega_p)`` -- is shared and unchanged.  The
default is the Loewner pencil, which interpolates the same ``2n_p``
values with the same ``n_p`` poles and never forms a power of ``x``.
The published Pade route remains reachable, both because it is what the
papers specify and because ``solve="pade", affine=False`` reproduces the
shipped 2026-08-09 solve exactly, which is the only way to keep
exhibiting the disease this module was reconditioned to cure.

The guards, in order
--------------------
The two published corrections are exactly the two reflections that map a
fitted ``b_p = Omega_p**2`` into the closed fourth quadrant of the
``x = z**2`` plane, after which the principal square root lands in the
physical sector automatically.  They are applied in this order, and each
is separately switchable so its red twin can be exhibited:

1. ``reflection`` (published, Eq. S18).  ``Re b_p < 0`` means
   ``Gamma_p > a_p`` -- a pole further from the real axis than from the
   imaginary one, which the papers call nonphysical.  Repair
   ``b_p <- -conj(b_p)``: this flips ``Re b_p`` positive while PRESERVING
   ``Im b_p``, hence preserving time ordering.  Model-changing.
2. ``time_order``.  ``Im b_p = 2 * Re(Omega) * Im(Omega)``, so
   ``Im b_p > 0`` is the time-ordering violation.  Repair
   ``b_p <- conj(b_p)``: flips ``Im b_p`` negative while preserving
   ``Re b_p``.  Model-changing.
3. ``prune_coincident``.  Two poles closer than ``coincident_tol`` times
   the frequency scale are one pole fitted twice; the later one is
   dropped (its residue forced to zero) rather than left to make the
   residue system singular.
4. ``prune_out_of_range``.  Poles outside the admissible box -- see
   ``_guard_prune_out_of_range`` -- carry no support from the sampled
   data and are dropped.

MANDATORY RESIDUE REFIT.  Whenever ANY guard fires, the residues are
re-solved by the all-2*n_p-point complex least-squares problem with the
corrected poles held fixed.  Guards 1 and 2 move ``b_p``, so the
pre-guard residues are stale; guards 3 and 4 change which columns exist,
so the pre-guard residues are over-complete.  ``refit_after_guards=False``
exists ONLY so the tests can exhibit the stale-residue defect; it is not
a production setting.

Batching
--------
The production shape is ``(n_elements, 2*n_p)`` -- column tiles of
``W_q(mu, nu)`` flattened over the element axis.  ``fit_mpa_poles`` is a
pure function of one element's sample vector with no data-dependent
shapes anywhere, so ``jax.vmap`` over the leading axis works directly;
``fit_mpa_poles_batched`` is that vmap, provided so callers do not have
to rediscover the ``in_axes``.  All refusals are static-shape gates that
fire at trace time, so they are still loud under ``vmap`` and ``jit``.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Default guard configuration.  Values, not behaviour: every entry is
# read through ``_resolve_guards`` and may be overridden per call.
DEFAULT_GUARDS = {
    "reflection": True,
    "time_order": True,
    "prune_coincident": True,
    "prune_out_of_range": True,
    # Two poles within this fraction of the frequency scale are one pole.
    "coincident_tol": 1.0e-6,
    # Upper edge of the admissible |Re Omega| box, as a multiple of the
    # frequency scale (max |z_j|).  A pole above it is extrapolation.
    "range_factor_hi": 2.0,
    # |Omega| below this fraction of the scale is a numerical zero.
    "range_factor_lo": 1.0e-10,
    # The papers' "poles in the vicinity of the real frequency axis"
    # condition, quoted for MPA-Sigma as zeta/varsigma < -1 and
    # equivalent to |Im Omega| <= width_ratio_max * Re Omega.  Guard 1
    # already enforces this for every pole it touches; this entry catches
    # the residue of it after conjugation and pruning.
    "width_ratio_max": 1.0,
}

_GUARD_KEYS = tuple(DEFAULT_GUARDS)

#: The two denominator solves.  Both answer the SAME interpolation
#: problem -- ``n_p`` poles through ``2*n_p`` samples of ``W_c`` in
#: ``x = z**2`` -- and both return ``(B_p, Omega_p)`` in the identical
#: representation; they differ only in the algebra that gets there, and
#: therefore only in what finite precision does to the answer.
#:
#: ``"loewner"``
#:     The fixed-support Loewner pencil.  DEFAULT.  Never forms a power
#:     of ``x``, so the Vandermonde conditioning is simply absent.
#: ``"pade"``
#:     The published cross-multiplied Pade-in-z^2 system (sigma-paper
#:     Eqs. S9/S8), with the affine domain map of
#:     :func:`_affine_domain`.  ``affine=False`` on top of this is the
#:     shipped 2026-08-09 solve exactly, and is the red twin.
SOLVE_MODES = ("loewner", "pade")

#: Measured on 36,096 production ``W_c`` elements of the frozen Si
#: 4x4x4 deck (q in {0,1,21,63}, all 1128 rows, columns 0:8), reading
#: the same stores the papers-convergence protocol reads --
#: ``mpa_wcprod_0809`` for rungs 1/2/8 and ``mpa_np10_0810`` for rung 10.
#: ``cond`` is the conditioning of whatever matrix the mode inverts
#: (the row-equilibrated ``2n x 2n`` cross-multiplied system for
#: ``"pade"``, the ``n x n`` Loewner matrix for ``"loewner"``); ``<RSD>``
#: is [I] Eq. (28) averaged over elements; ``mass>16eV`` is the fraction
#: of ``sum_p |B_p|`` carried by poles wider than 16 eV.
#:
#: ===== ========================= ========= ========= =========
#: n_p   mode                      cond med  <RSD>     mass>16eV
#: ===== ========================= ========= ========= =========
#: 8     pade, affine=False        1.60e13   1.493e-3  0.79 %
#: 8     pade, affine=True         3.99e11   1.107e-3  0.66 %
#: 8     loewner                   1.14e07   1.100e-3  0.31 %
#: 10    pade, affine=False        1.11e16   1.979e-3  58.1 %
#: 10    pade, affine=True         5.68e13   6.286e-4  0.78 %
#: 10    loewner                   4.86e08   6.158e-4  0.03 %
#: ===== ========================= ========= ========= =========
#:
#: The shipped solve's ``<RSD>`` RISES from ``n_p = 8`` to ``n_p = 10``
#: while half its residue mass runs away onto modes broader than the
#: plasmon; both reconditioned modes restore the published falling
#: curve.  A CHEBYSHEV BASIS IN THE AFFINE VARIABLE, WITH COLLEAGUE-
#: MATRIX ROOTS, WAS BUILT AND MEASURED AND IS NOT SHIPPED: on this
#: geometry it is worse than the monomial basis it would replace
#: (``n_p = 10``: cond med 1.51e14, ``<RSD>`` 6.92e-4).  The reason is
#: geometric rather than incidental -- ``x = z**2`` maps the two
#: parallel sample lines onto two ARCS whose imaginary extent reaches
#: 1.37 in the affine variable, and a Chebyshev basis is
#: well-conditioned ON its interval and grows like ``rho**k`` off it,
#: with ``rho ~ 3`` here.  Chebyshev's advantage needs the samples to
#: lie on the interval, and the double-parallel protocol puts them
#: beside it.


def _resolve_guards(guards):
    """Merge a caller override onto ``DEFAULT_GUARDS``; refuse strays."""

    resolved = dict(DEFAULT_GUARDS)
    if guards is None:
        return resolved
    unknown = sorted(set(guards) - set(_GUARD_KEYS))
    if unknown:
        raise ValueError(
            f"GATE guard_keys_known: unknown guard key(s) {unknown}. "
            f"FALSE case: every key of ``guards`` is one of {_GUARD_KEYS}.")
    resolved.update(guards)
    return resolved


def _x64_is_on():
    """True when jax is configured for 64-bit.  Version-tolerant."""

    try:
        return bool(jax.config.read("jax_enable_x64"))
    except Exception:
        return bool(getattr(jax.config, "jax_enable_x64", False))


def _require_x64():
    if not _x64_is_on():
        raise RuntimeError(
            "GATE x64_enabled: jax is running without x64, so complex128 "
            "silently degrades to complex64 and the Pade solve returns "
            "noise. FALSE case: JAX_ENABLE_X64=1 in the environment BEFORE "
            "the first ``import jax`` (or "
            "``jax.config.update('jax_enable_x64', True)`` before any array "
            "is made).")


def _check_sample_support(W_samples, z_samples, n_p):
    """Static-shape gates.  Fire at trace time, so vmap-safe."""

    n = int(n_p)
    if n < 1:
        raise ValueError(
            f"GATE n_p_positive: n_p={n_p!r} is not a positive integer. "
            "FALSE case: int(n_p) >= 1.")

    w_shape = tuple(jnp.shape(W_samples))
    z_shape = tuple(jnp.shape(z_samples))
    if len(w_shape) != 1:
        raise ValueError(
            f"GATE W_samples_rank: W_samples has shape {w_shape}; this "
            "kernel fits ONE element and is vmapped for batches. FALSE "
            "case: W_samples.ndim == 1 (use fit_mpa_poles_batched, or "
            "jax.vmap, for an (n_elements, 2*n_p) tile).")
    if len(z_shape) != 1:
        raise ValueError(
            f"GATE z_samples_rank: z_samples has shape {z_shape}. FALSE "
            "case: z_samples.ndim == 1; the sample grid is shared by every "
            "element of a tile and is never batched.")
    if w_shape[0] != 2 * n or z_shape[0] != 2 * n:
        raise ValueError(
            f"GATE sample_support: n_p={n} demands exactly {2 * n} samples "
            f"but W_samples carries {w_shape[0]} and z_samples carries "
            f"{z_shape[0]} -- n_p exceeds (or undershoots) the sample "
            "support, and the Pade system is not square. FALSE case: "
            "W_samples.shape[0] == z_samples.shape[0] == 2*n_p.")


def _solve_normalised(A, rhs, rcond, *, equilibrate=True):
    """Truncated-SVD solve.  Returns ``(x, cond, s_max, s_min)``.

    "Normalised" in the plan's sense: the monic-Q normalisation is built
    into the Pade ansatz, and with ``equilibrate=True`` each row of the
    cross-multiplied system is scaled to unit 2-norm before the solve.
    Row equilibration does not change the exact solution of a SQUARE
    system -- it changes which solution finite precision finds, and the
    sampled ``W_c`` spans decades between the near line and the far line.

    It is NOT applied to the overdetermined residue system, where row
    scaling would silently reweight the least-squares objective; the plan
    asks for plain all-2*n_p-point complex least squares there.

    The solve is an explicit truncated-SVD pseudo-inverse rather than
    ``jnp.linalg.lstsq`` so that (a) the singular values are available
    for the diagnostics without a second factorisation and (b) there is
    no data-dependent rank in the output signature, which is what would
    break ``vmap``.
    """

    if equilibrate:
        row_norm = jnp.linalg.norm(A, axis=1)
        row_norm = jnp.where(row_norm > 0, row_norm, 1.0)
        A_n = A / row_norm[:, None]
        rhs_n = rhs / row_norm
    else:
        A_n = A
        rhs_n = rhs

    u, s, vh = jnp.linalg.svd(A_n, full_matrices=False)
    s_max = s[0]
    s_min = s[-1]
    cutoff = rcond * s_max
    s_inv = jnp.where(s > cutoff, 1.0 / jnp.where(s > 0, s, 1.0), 0.0)
    x = vh.conj().T @ (s_inv.astype(A_n.dtype) * (u.conj().T @ rhs_n))
    cond = jnp.where(s_min > 0, s_max / s_min, jnp.inf)
    return x, cond, s_max, s_min


def _x_normalisation(z_samples):
    """``x_max = max_j |z_j**2|`` -- the plan's z_max scaling, alone.

    Split out because ``x_max`` is the normalisation of the RESIDUE stage
    and of ``b_hat = Omega**2 / x_max``, and those are shared by every
    solve mode.  Only the DENOMINATOR stage varies between modes.
    """

    z = jnp.asarray(z_samples, dtype=jnp.complex128)
    x = z * z
    x_max = jnp.max(jnp.abs(x))
    x_max = jnp.where(x_max > 0, x_max, 1.0).astype(jnp.float64)
    return x, x_max


def _affine_domain(x):
    """Centre and half-width of the sample set's own real extent.

    THE MAP.  ``t = (x - centre) / scale`` with ``centre`` the midpoint
    and ``scale`` the half-width of ``[min_j Re x_j, max_j Re x_j]``.  It
    is read off the SAMPLE SET, so it is a property of the grid the file
    was evaluated on and never a tuned constant.

    WHY THE REAL EXTENT AND NOT THE MODULUS.  The shipped normalisation
    ``x_hat = x / max_j|x_j|`` is a pure scaling about the ORIGIN, and
    the origin is at the edge of the sampled set rather than inside it:
    on the double-parallel grid ``Re x`` runs from ``-varpi_2**2`` to
    ``omega_m**2 - varpi_1**2`` -- for the Si deck, ``[-1, +6.36] Ha^2``
    -- so ``x_hat`` lands the samples in ``[-0.14, +0.86]``, a set whose
    Chebyshev-like transfinite diameter is far smaller than 1 and whose
    monomial Vandermonde is correspondingly worse conditioned.  Centring
    first is one line and it is worth two orders of magnitude of
    conditioning at ``n_p = 10`` (measured; see the module note).

    Returns ``(centre, scale)``, both traced scalars.
    """

    lo = jnp.min(jnp.real(x))
    hi = jnp.max(jnp.real(x))
    centre = (0.5 * (hi + lo)).astype(jnp.float64)
    scale = (0.5 * (hi - lo)).astype(jnp.float64)
    scale = jnp.where(scale > 0, scale, 1.0)
    return centre, scale


def build_pade_system(W_samples, z_samples, n_p, *, affine=True):
    """Return ``(A, rhs, t, centre, scale)`` of the cross-multiplied system.

    With ``x = z**2`` and ``t = (x - centre)/scale``, the model
    ``P(t)/Q(t)`` with ``Q`` monic of degree ``n_p`` and ``P`` of degree
    ``n_p - 1`` gives the linear system in ``2*n_p`` unknowns
    ``[d_0..d_{n_p-1}, c_0..c_{n_p-1}]``::

        sum_k d_k t_j^k  -  W_j sum_k c_k t_j^k  =  W_j t_j^{n_p}

    which is the even-in-z form of the sigma paper's Eq. (S9).

    ``affine=True`` (the default) takes ``(centre, scale)`` from
    :func:`_affine_domain`.  ``affine=False`` is the SHIPPED
    normalisation -- ``centre = 0``, ``scale = x_max`` -- i.e. the plan's
    z_max scaling on its own, kept reachable because it is the red twin:
    the conditioning pathology this module was reconditioned to cure is
    only exhibitable by the solve that has it.  Without ANY rescaling the
    raw Vandermonde over ``x`` is hopeless well before ``n_p ~ 8``, so
    ``affine=False`` is a weaker normalisation and not an absent one.
    """

    n = int(n_p)
    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    x, x_max = _x_normalisation(z_samples)
    if affine:
        centre, scale = _affine_domain(x)
    else:
        centre, scale = jnp.zeros((), dtype=jnp.float64), x_max
    t = (x - centre.astype(jnp.complex128)) / scale.astype(jnp.complex128)

    vand = jnp.vander(t, n, increasing=True)
    A = jnp.concatenate([vand, -w[:, None] * vand], axis=1)
    rhs = w * t ** n
    return A, rhs, t, centre, scale


def _loewner_pencil(w, x_hat, n):
    """The fixed-support Loewner pencil ``(L, sL)``, ``n`` by ``n`` each.

    THE SUPPORT IS FIXED AND IT IS ALL 2*n_p SAMPLES.  The even-indexed
    samples are the left (row) support ``lambda_i`` and the odd-indexed
    ones the right (column) support ``mu_j``; because the grid is stored
    NEAR line first then FAR line, alternating indices puts half of each
    line on each side, which is the split that keeps both lines
    represented in both supports.  Nothing here is adaptive and no shape
    depends on the data -- this is a ``vmap``/``jit`` kernel like the
    rest of the module.

    THE MATRICES (Mayo & Antoulas, *Lin. Alg. Appl.* 425 (2007) 634,
    scalar case)::

        L[i, j]  = (f(lambda_i) - f(mu_j)) / (lambda_i - mu_j)
        sL[i, j] = (lambda_i f(lambda_i) - mu_j f(mu_j))
                   / (lambda_i - mu_j)

    The rational function of type ``(n-1, n)`` that interpolates all
    ``2n`` values has, as its poles, the eigenvalues of the pencil
    ``(sL, L)``.  That is exactly the MPA model's own type: ``W_c`` in
    the ``x = z**2`` variable is ``sum_p a_p/(x - x_p)``, strictly
    proper with ``n_p`` poles.  So the Loewner route solves the SAME
    interpolation problem as the Pade cross-multiplication and returns
    the same object; it differs only in never forming a power of ``x``,
    which is where the Vandermonde conditioning came from.

    ``lambda_i != mu_j`` for every pair because ``sampling`` already
    refuses a grid whose ``z_j**2`` collide (GATE
    ``distinct_squared_samples``), so no denominator here can vanish.
    """

    lam, mu = x_hat[0:2 * n:2], x_hat[1:2 * n:2]
    f_l, f_r = w[0:2 * n:2], w[1:2 * n:2]

    gap = lam[:, None] - mu[None, :]
    gap = jnp.where(jnp.abs(gap) > 0, gap, 1.0)
    L = (f_l[:, None] - f_r[None, :]) / gap
    sL = (lam[:, None] * f_l[:, None] - mu[None, :] * f_r[None, :]) / gap
    return L, sL


def _loewner_roots(w, x_hat, n, rcond):
    """Poles of the Loewner interpolant, in the ``b_hat = x/x_max`` plane.

    Returns ``(b_hat, cond, s_max, s_min)`` in the same shape contract as
    the companion route, so the two are drop-in for each other.

    THE PENCIL IS REDUCED RATHER THAN SOLVED AS A PENCIL.  The poles are
    the generalised eigenvalues of ``(sL, L)``; jax has no ``QZ``, so
    they are taken as the ordinary eigenvalues of ``L^+ sL`` with ``L^+``
    the same truncated-SVD pseudo-inverse the rest of the module uses.
    That is the identical spectrum whenever ``L`` is invertible, and the
    reported ``cond`` is ``cond(L)`` -- precisely the number that says
    whether it was.  The same ``n x n`` non-symmetric ``eigvals`` call
    ends the companion route, so this is a drop-in on the eigensolver
    side too.
    """

    L, sL = _loewner_pencil(w, x_hat, n)
    u, s, vh = jnp.linalg.svd(L, full_matrices=False)
    s_max = s[0]
    s_min = s[-1]
    s_inv = jnp.where(s > rcond * s_max, 1.0 / jnp.where(s > 0, s, 1.0), 0.0)
    L_pinv = vh.conj().T @ (s_inv.astype(L.dtype)[:, None] * u.conj().T)
    cond = jnp.where(s_min > 0, s_max / s_min, jnp.inf)
    return jnp.linalg.eigvals(L_pinv @ sL), cond, s_max, s_min


def _companion_roots(c_coeffs):
    """Roots of the monic ``Q(t) = t**n + sum_k c_k t**k``.

    Companion matrix exactly as sigma-paper Eq. (S8): ones on the
    subdiagonal, ``-c`` in the last column.  ``jnp.linalg.eigvals`` is
    non-symmetric; on CPU it is both jittable and vmappable, which is why
    the whole kernel can stay in jax here.  (A GPU deployment must
    revisit this -- non-symmetric ``eig`` has no GPU lowering -- but the
    fit stage is disk-staged and cheap, so that is a placement question
    for the design review, not a correctness one.)
    """

    n = c_coeffs.shape[0]
    comp = jnp.zeros((n, n), dtype=jnp.complex128)
    if n > 1:
        comp = comp.at[1:, :-1].set(jnp.eye(n - 1, dtype=jnp.complex128))
    comp = comp.at[:, -1].set(-c_coeffs)
    return jnp.linalg.eigvals(comp)


def _guard_reflection(b_hat):
    """Guard 1 -- metals-paper SI Eq. (S18).  Returns ``(b_hat, fired)``.

    ``Re b < 0`` <=> ``Gamma > a``: the pole is nonphysically overdamped.
    ``b <- -conj(b)`` flips ``Re b`` positive and leaves ``Im b`` alone,
    so a time-ordered pole stays time-ordered.
    """

    bad = jnp.real(b_hat) < 0.0
    return jnp.where(bad, -jnp.conj(b_hat), b_hat), bad


def _guard_time_order(b_hat):
    """Guard 2 -- forced time ordering.  Returns ``(b_hat, fired)``.

    ``Im b = 2 Re(Omega) Im(Omega)``, so with ``Re Omega >= 0`` the
    time-ordering requirement ``Re(Omega)*Im(Omega) < 0`` is exactly
    ``Im b <= 0``.  ``b <- conj(b)`` is the reflection that enforces it
    without touching ``Re b``, i.e. without undoing guard 1.
    """

    bad = jnp.imag(b_hat) > 0.0
    return jnp.where(bad, jnp.conj(b_hat), b_hat), bad


def _guard_prune_coincident(Omega, valid, scale, tol):
    """Guard 3 -- collapse coincident poles.  Returns ``(valid, fired)``.

    A pole is dropped when an EARLIER live pole sits within
    ``tol * scale`` of it.  Poles arrive sorted (ascending ``Re Omega``,
    then ``Im Omega``), so "earlier" is deterministic and the survivor of
    a cluster is its lowest-frequency member.  Written as a masked
    pairwise comparison rather than a loop so the shape is static.
    """

    n = Omega.shape[0]
    dist = jnp.abs(Omega[:, None] - Omega[None, :])
    earlier = jnp.tri(n, n, k=-1, dtype=bool)
    close = (dist <= tol * scale) & earlier & valid[None, :]
    dropped = jnp.any(close, axis=1) & valid
    return valid & ~dropped, dropped


def _guard_prune_out_of_range(Omega, valid, scale, cfg):
    """Guard 4 -- drop poles the sampled data cannot support.

    The admissible box, all edges configurable:

    * ``Re Omega > 0`` -- a nonpositive real part is not a positive
      excitation energy, and the tau consumer's ``exp(-i Omega tau)``
      convention assumes one.
    * ``|Omega| >= range_factor_lo * scale`` -- below this the pole is a
      numerical zero of the companion polynomial, not a mode.
    * ``Re Omega <= range_factor_hi * scale`` -- above the sampled span
      the fit is extrapolating and the residue is unconstrained.
    * ``|Im Omega| <= width_ratio_max * Re Omega`` -- the papers' "poles
      lie in the vicinity of the real frequency axis" condition.
    """

    lo = cfg["range_factor_lo"] * scale
    hi = cfg["range_factor_hi"] * scale
    bad = (
        (jnp.real(Omega) <= 0.0)
        | (jnp.abs(Omega) < lo)
        | (jnp.real(Omega) > hi)
        | (jnp.abs(jnp.imag(Omega)) > cfg["width_ratio_max"]
           * jnp.real(Omega))
        | ~jnp.isfinite(jnp.abs(Omega))
    )
    dropped = bad & valid
    return valid & ~dropped, dropped


def _residue_lstsq(x_hat, w, b_hat, valid, x_max, Omega, rcond):
    """All-2*n_p-point complex least-squares residues at fixed poles.

    Solves ``min_a || M a - W ||`` with ``M[j, p] = 1/(xh_j - bh_p)``
    over EVERY sample -- both lines, all ``2*n_p`` points -- which is the
    plan's "all-2n_p-point complex least-squares residues".  Dead columns
    are zeroed; the truncated-SVD pseudo-inverse then returns exactly
    zero for their coefficients (minimum-norm solution), so pruning needs
    no shape change and stays vmappable.
    """

    denom = x_hat[:, None] - b_hat[None, :]
    safe = jnp.where(jnp.abs(denom) > 0, denom, 1.0)
    M = jnp.where(valid[None, :], 1.0 / safe, 0.0 + 0.0j)
    a_hat, _, _, _ = _solve_normalised(M, w, rcond, equilibrate=False)
    a_hat = jnp.where(valid, a_hat, 0.0 + 0.0j)
    a = a_hat * x_max.astype(jnp.complex128)
    two_om = 2.0 * Omega
    two_om = jnp.where(jnp.abs(two_om) > 0, two_om, 1.0)
    B = jnp.where(valid, a / two_om, 0.0 + 0.0j)
    return B


def eval_mpa_model(Omega, B, z, valid=None):
    """Evaluate ``sum_p 2 Omega_p B_p / (z**2 - Omega_p**2)``.

    ``z`` broadcasts against a trailing pole axis, so this works for a
    single element's pole set against a vector of frequencies and, under
    ``vmap``, for a batch.
    """

    om = jnp.asarray(Omega, dtype=jnp.complex128)
    b = jnp.asarray(B, dtype=jnp.complex128)
    zz = jnp.asarray(z, dtype=jnp.complex128)
    if valid is not None:
        b = jnp.where(jnp.asarray(valid), b, 0.0 + 0.0j)
    denom = zz[..., None] ** 2 - om ** 2
    safe = jnp.where(jnp.abs(denom) > 0, denom, 1.0)
    terms = jnp.where(jnp.abs(denom) > 0, 2.0 * om * b / safe, 0.0 + 0.0j)
    return jnp.sum(terms, axis=-1)


def fit_mpa_poles(
    W_samples,
    z_samples,
    n_p,
    *,
    guards=None,
    refit_after_guards=True,
    rcond=1.0e-13,
    solve=SOLVE_MODES[0],
    affine=True,
):
    """Fit ``n_p`` MPA poles to one element's ``2*n_p`` samples of W_c.

    Parameters
    ----------
    W_samples
        ``(2*n_p,)`` complex128 -- ``W_c(z_j)`` for one ISDF element.
        Under ``jax.vmap`` this is the mapped axis.
    z_samples
        ``(2*n_p,)`` complex128 -- the sample grid, shared across a tile.
        Build it with ``sampling.double_parallel_grid``.
    n_p
        Number of poles; a Python int (static under jit/vmap).
    guards
        Optional override of ``DEFAULT_GUARDS``.  Setting any of the four
        boolean entries to ``False`` disables that guard -- for red-twin
        tests only.
    refit_after_guards
        Keep ``True``.  ``False`` returns the PRE-guard residues against
        the POST-guard poles, which is the stale-residue defect the
        theory plan calls out; it exists so the tests can prove the refit
        ran.
    rcond
        Relative singular-value cutoff for both least-squares solves.
    solve
        Which denominator algebra finds the poles; one of
        :data:`SOLVE_MODES`.  Both give ``(B_p, Omega_p)`` in the same
        representation and both feed the identical guards, canonical
        sort and residue refit -- the choice is numerical, not physical.
    affine
        ``"pade"`` mode only: use the affine domain map of
        :func:`_affine_domain`.  ``solve="pade", affine=False`` is the
        shipped 2026-08-09 solve exactly, and is how the conditioning
        pathology is exhibited.

    Returns
    -------
    ``(Omega_p, B_p, diag)``
        ``Omega_p`` ``(n_p,)`` complex128, ascending in ``Re Omega``;
        ``B_p`` ``(n_p,)`` complex128, zero on pruned poles;
        ``diag`` a dict of jax arrays (a pytree, so it survives ``vmap``)
        carrying ``valid``, the per-guard fire counts, the conditioning
        of the denominator solve, and the achieved sample residual.
    """

    _require_x64()
    _check_sample_support(W_samples, z_samples, n_p)
    cfg = _resolve_guards(guards)
    if solve not in SOLVE_MODES:
        raise ValueError(
            f"GATE solve_mode_known: solve={solve!r} is not one of "
            f"{SOLVE_MODES}. FALSE case: solve names a denominator "
            "algebra this kernel implements.")
    n = int(n_p)

    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)
    x, x_max = _x_normalisation(z)
    x_hat = x / x_max.astype(jnp.complex128)

    # --- Stages 1 and 2: the denominator solve and its roots, both of
    # which vary with ``solve`` and NOTHING ELSE DOES.  Every mode hands
    # the same object to stage 3: ``b_hat = Omega**2 / x_max``, the
    # scaled pole positions in the x = z**2 plane.  The branch is on a
    # Python string, so it is resolved at trace time and both arms are
    # static-shape -- this stays one jit/vmap kernel either way.
    if solve == "loewner":
        b_hat, cond, s_max, s_min = _loewner_roots(w, x_hat, n, rcond)
    else:
        A, rhs, _t, x_centre, x_scale = build_pade_system(
            w, z, n, affine=affine)
        coef, cond, s_max, s_min = _solve_normalised(A, rhs, rcond)
        # Roots come back in the t plane; ``b_hat`` lives in the x/x_max
        # plane.  Composing the two scalars BEFORE touching the roots
        # keeps ``affine=False`` bit-identical to the shipped solve,
        # where the composition is exactly ``1.0`` and ``0.0``.
        rescale = (x_scale / x_max).astype(jnp.complex128)
        offset = (x_centre / x_max).astype(jnp.complex128)
        b_hat = _companion_roots(coef[n:]) * rescale + offset

    # --- Stage 3: residues BEFORE any guard fires.  These are the
    # all-sample complex LS residues of the raw fit; they are what
    # ``refit_after_guards=False`` keeps.
    scale = jnp.sqrt(x_max).astype(jnp.float64)
    Omega_raw = jnp.sqrt(b_hat * x_max.astype(jnp.complex128))
    valid_all = jnp.ones((n,), dtype=bool)
    B_pre = _residue_lstsq(
        x_hat, w, b_hat, valid_all, x_max, Omega_raw, rcond)

    # --- Stage 4: the guards, in order.
    fired_reflection = jnp.zeros((n,), dtype=bool)
    fired_time_order = jnp.zeros((n,), dtype=bool)
    if cfg["reflection"]:
        b_hat, fired_reflection = _guard_reflection(b_hat)
    if cfg["time_order"]:
        b_hat, fired_time_order = _guard_time_order(b_hat)

    Omega = jnp.sqrt(b_hat * x_max.astype(jnp.complex128))

    # Canonical ordering: ascending Re Omega, then Im Omega.  Makes the
    # coincident-pruning survivor deterministic and the returned pole set
    # comparable against a reference without a matching step.
    order = jnp.lexsort((jnp.imag(Omega), jnp.real(Omega)))
    Omega = Omega[order]
    b_hat = b_hat[order]
    B_pre = B_pre[order]
    fired_reflection = fired_reflection[order]
    fired_time_order = fired_time_order[order]

    valid = jnp.ones((n,), dtype=bool)
    fired_coincident = jnp.zeros((n,), dtype=bool)
    fired_range = jnp.zeros((n,), dtype=bool)
    if cfg["prune_coincident"]:
        valid, fired_coincident = _guard_prune_coincident(
            Omega, valid, scale, cfg["coincident_tol"])
    if cfg["prune_out_of_range"]:
        valid, fired_range = _guard_prune_out_of_range(
            Omega, valid, scale, cfg)

    any_correction = (
        jnp.any(fired_reflection) | jnp.any(fired_time_order)
        | jnp.any(fired_coincident) | jnp.any(fired_range))

    # --- Stage 5: THE MANDATORY RESIDUE REFIT.  Any guard that fired
    # moved a pole or removed a column, so the pre-guard residues are
    # stale.  ``jnp.where`` rather than a Python branch: ``any_correction``
    # is a traced value under vmap, and the refit must be unconditional
    # in the graph.
    B_refit = _residue_lstsq(x_hat, w, b_hat, valid, x_max, Omega, rcond)
    B_stale = jnp.where(valid, B_pre, 0.0 + 0.0j)
    if refit_after_guards:
        B = jnp.where(any_correction, B_refit, B_stale)
        refit_performed = any_correction
    else:
        B = B_stale
        refit_performed = jnp.zeros((), dtype=bool)

    # --- Stage 6: achieved residual on the fitted samples.
    model = eval_mpa_model(Omega, B, z, valid=valid)
    resid = jnp.abs(model - w)
    w_scale = jnp.maximum(jnp.max(jnp.abs(w)), jnp.finfo(jnp.float64).tiny)

    diag = {
        "valid": valid,
        "n_valid": jnp.sum(valid.astype(jnp.int32)),
        "cond_pade": cond,
        "sigma_max_pade": s_max,
        "sigma_min_pade": s_min,
        "x_max": x_max,
        "n_reflected": jnp.sum(fired_reflection.astype(jnp.int32)),
        "n_time_order_flipped": jnp.sum(fired_time_order.astype(jnp.int32)),
        "n_pruned_coincident": jnp.sum(fired_coincident.astype(jnp.int32)),
        "n_pruned_out_of_range": jnp.sum(fired_range.astype(jnp.int32)),
        "any_correction": any_correction,
        "refit_performed": refit_performed,
        "max_abs_residual": jnp.max(resid),
        "rel_rms_residual": (
            jnp.sqrt(jnp.mean(resid ** 2)) / w_scale),
        # [I] Eq. (28) VERBATIM -- the papers' relative standard
        # deviation, which divides by 2*n_p - 1 rather than 2*n_p.  It is
        # carried BESIDE ``rel_rms_residual`` and not instead of it: the
        # shipped stores' numbers are the latter, and silently moving a
        # stamped quantity by a factor of sqrt(2n/(2n-1)) is how a
        # convergence table acquires a wobble nobody can attribute.
        "rsd_eq28": (
            jnp.sqrt(jnp.sum(resid ** 2) / (2.0 * n - 1.0)) / w_scale),
    }
    return Omega, B, diag


def fit_mpa_poles_batched(
    W_tile,
    z_samples,
    n_p,
    *,
    guards=None,
    refit_after_guards=True,
    rcond=1.0e-13,
    solve=SOLVE_MODES[0],
    affine=True,
):
    """``fit_mpa_poles`` vmapped over the leading (element) axis.

    ``W_tile`` is ``(n_elements, 2*n_p)`` -- a flattened column tile of
    ``W_q(mu, nu)``.  ``z_samples`` is shared and unmapped.  Returns
    ``(Omega, B, diag)`` with a leading ``n_elements`` axis on every leaf,
    bit-identical to looping ``fit_mpa_poles`` over the rows.

    ``solve`` and ``affine`` are static and shared across the tile, so
    the mapped function has one shape and one trace whichever mode is
    chosen; this stays a single ``jit``-able kernel.
    """

    tile = jnp.asarray(W_tile, dtype=jnp.complex128)
    if tile.ndim != 2:
        raise ValueError(
            f"GATE W_tile_rank: W_tile has shape {tuple(tile.shape)}. FALSE "
            "case: W_tile.ndim == 2, i.e. (n_elements, 2*n_p); reshape a "
            "(q, mu, nu, 2*n_p) tensor to two axes before calling.")

    def _one(w_row):
        return fit_mpa_poles(
            w_row, z_samples, n_p, guards=guards,
            refit_after_guards=refit_after_guards, rcond=rcond,
            solve=solve, affine=affine)

    return jax.vmap(_one)(tile)


def synthesize_w_samples(Omega, B, z_samples):
    """Host-side ``W_c(z_j)`` from a known pole set.  Test/validation aid.

    Uses the same model as ``eval_mpa_model`` but in numpy, so a test can
    build its reference without going through the code under test.
    """

    om = np.asarray(Omega, dtype=np.complex128)
    b = np.asarray(B, dtype=np.complex128)
    z = np.asarray(z_samples, dtype=np.complex128)
    if om.shape != b.shape:
        raise ValueError(
            f"GATE pole_residue_shapes: Omega has shape {om.shape} and B "
            f"has shape {b.shape}. FALSE case: Omega.shape == B.shape.")
    denom = z[:, None] ** 2 - om[None, :] ** 2
    if np.any(np.abs(denom) == 0.0):
        raise ValueError(
            "GATE synthesis_off_pole: a sample point sits exactly on a "
            "synthesized pole, so W_c is infinite there. FALSE case: no "
            "z_j**2 equals any Omega_p**2 -- give the poles a nonzero "
            "width, which the physical model has anyway.")
    return np.sum(2.0 * om[None, :] * b[None, :] / denom, axis=1)
