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


def build_pade_system(W_samples, z_samples, n_p):
    """Return ``(A, rhs, x_hat, x_max)`` of the cross-multiplied system.

    With ``x = z**2`` and ``x_hat = x / x_max``, ``x_max = max_j |z_j|**2``
    (the plan's z_max scaling), the model ``P(x)/Q(x)`` with ``Q`` monic
    of degree ``n_p`` and ``P`` of degree ``n_p - 1`` gives the linear
    system in ``2*n_p`` unknowns ``[d_0..d_{n_p-1}, c_0..c_{n_p-1}]``::

        sum_k d_k xh_j^k  -  W_j sum_k c_k xh_j^k  =  W_j xh_j^{n_p}

    which is the even-in-z form of the sigma paper's Eq. (S9).  Without
    the ``x_max`` rescaling the raw Vandermonde over ``x`` spanning
    ``[0, omega_m**2 + varpi_2**2]`` is hopeless at ``n_p ~ 8``.
    """

    n = int(n_p)
    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)
    x = z * z
    x_max = jnp.max(jnp.abs(x))
    x_max = jnp.where(x_max > 0, x_max, 1.0).astype(jnp.float64)
    x_hat = x / x_max.astype(jnp.complex128)

    vand = jnp.vander(x_hat, n, increasing=True)
    A = jnp.concatenate([vand, -w[:, None] * vand], axis=1)
    rhs = w * x_hat ** n
    return A, rhs, x_hat, x_max


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

    Returns
    -------
    ``(Omega_p, B_p, diag)``
        ``Omega_p`` ``(n_p,)`` complex128, ascending in ``Re Omega``;
        ``B_p`` ``(n_p,)`` complex128, zero on pruned poles;
        ``diag`` a dict of jax arrays (a pytree, so it survives ``vmap``)
        carrying ``valid``, the per-guard fire counts, the conditioning
        of the Pade solve, and the achieved sample residual.
    """

    _require_x64()
    _check_sample_support(W_samples, z_samples, n_p)
    cfg = _resolve_guards(guards)
    n = int(n_p)

    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)

    # --- Stage 1: the normalised Pade-in-z^2 linear solve.
    A, rhs, x_hat, x_max = build_pade_system(w, z, n)
    coef, cond, s_max, s_min = _solve_normalised(A, rhs, rcond)
    row_norm = jnp.linalg.norm(A, axis=1)
    row_norm = jnp.where(row_norm > 0, row_norm, 1.0)
    A_n, rhs_n = A / row_norm[:, None], rhs / row_norm
    backward_num = jnp.linalg.norm(A_n @ coef - rhs_n)
    backward_den = (
        jnp.linalg.norm(A_n) * jnp.linalg.norm(coef)
        + jnp.linalg.norm(rhs_n))
    backward_error = backward_num / jnp.where(
        backward_den > 0, backward_den, 1.0)
    c_coeffs = coef[n:]

    # --- Stage 2: companion-matrix roots.  b = Omega**2, scaled.
    b_hat = _companion_roots(c_coeffs)

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
        "backward_error": backward_error,
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
):
    """``fit_mpa_poles`` vmapped over the leading (element) axis.

    ``W_tile`` is ``(n_elements, 2*n_p)`` -- a flattened column tile of
    ``W_q(mu, nu)``.  ``z_samples`` is shared and unmapped.  Returns
    ``(Omega, B, diag)`` with a leading ``n_elements`` axis on every leaf,
    bit-identical to looping ``fit_mpa_poles`` over the rows.
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
            refit_after_guards=refit_after_guards, rcond=rcond)

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
