"""Build parent Green operators and transport them with typed local symmetry actions."""
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp

from common.contract_bands import merge_spin_centroid


def _build_G_face(psi_mun, psi_nmu, *, gemm, Gij=None, phases=None, mesh=None,
                  band_range=None, prepared_active_gemm=None):
    """Contract band-replicated faces locally or band-distributed faces with their GEMM plan.

    Returns the Green ``(nk, mu_X, s, nu_Y, s')``: centroid-major, the
    GEMM's own merged endpoint order split by a reshape.
    """
    if Gij is not None:
        raise NotImplementedError("Green faces support diagonal band weights, not dense Gij.")
    nk_, s_, mu_l_, n_ = psi_mun.shape
    nk_r_, n_r_, s_r_, mu_r_ = psi_nmu.shape
    if nk_r_ != nk_ or n_r_ != n_ or s_r_ != s_:
        raise ValueError(
            "build_G(layout='face'): left psi_mun and right psi_nmu must "
            "share (nk, nb, nspinor); got "
            f"{psi_mun.shape} and {psi_nmu.shape}.")
    A = merge_spin_centroid(psi_mun, 1, 2)          # (nk, mu*s, n) P(_,'x','y')
    if (phases is not None and band_range is None
            and prepared_active_gemm is None):
        w = phases.astype(A.dtype)                  # (nk, n)
        A = A * w[:, None, :]
    B = merge_spin_centroid(jnp.conj(psi_nmu), 2, 3)  # (nk, n, mu*s) P(_,'x','y')
    # Eager operations may erase singleton mesh axes before the GEMM boundary.
    from jax import lax
    in_sharding_a = getattr(gemm, "in_sharding_a", None)
    in_sharding_b = getattr(gemm, "in_sharding_b", None)
    if in_sharding_a is not None:
        A = lax.with_sharding_constraint(A, in_sharding_a)
    if in_sharding_b is not None:
        B = lax.with_sharding_constraint(B, in_sharding_b)
    if prepared_active_gemm is not None:
        if band_range is not None:
            raise ValueError(
                "prepared_active_gemm and band_range are mutually exclusive")
        if phases is None:
            raise ValueError("prepared_active_gemm requires per-band phases")
        G_flat = prepared_active_gemm(A, B, weights=phases)
    else:
        G_flat = (gemm(A, B) if band_range is None
                  else gemm.active_range(A, B, *band_range, weights=phases))
    # (nk, mu*s, nu*s), distributed over both centroid axes.  The merged
    # endpoint order is centroid-major (``mu*ns + s``, merge_spin_centroid's
    # own collective-free direction), so the split is a pure reshape: the
    # Green is stored ``(nk, mu_X, s, nu_Y, s')`` and is never transposed to
    # a spin-major order.  The parent-k unfold transports this order as-is.
    return G_flat.reshape(nk_, mu_l_, s_, mu_r_, s_)


def build_G(psi_xn, psi_yr, *, Gij=None, phases=None, layout='face',
           gemm=None, k_unfold_plan=None, right_k_unfold_plan=None, real_weights=None,
           band_range=None, prepared_active_gemm=None):
    """Build parent operators and transport both typed endpoints without processor exchange.

    The Green is centroid-major ``(nk, mu, s, nu, s')`` on parents and on
    full k alike; see :func:`_build_G_face`.
    """
    if layout not in ('face', 'axis'):
        raise ValueError("build_G requires canonical faces with layout=face or axis.")
    if gemm is None:
        raise ValueError("build_G requires a GEMM plan or typed parent plan-provided GEMM callable.")
    G = _build_G_face(psi_xn, psi_yr, gemm=gemm, Gij=Gij, phases=phases,
                      mesh=None if k_unfold_plan is None else k_unfold_plan.mesh_xy,
                      band_range=band_range,
                      prepared_active_gemm=prepared_active_gemm)
    if k_unfold_plan is None:
        return G
    transposed = None
    if np.any(np.asarray(k_unfold_plan.sym_idx) >= k_unfold_plan.n_sym_spatial):
        if (real_weights is True or phases is None
                or not jnp.issubdtype(phases.dtype, jnp.complexfloating)):
            transposed = jnp.conj(G)
        else:
            transposed = jax.lax.cond(
                (jnp.any(jnp.imag(phases) != 0) if real_weights is None
                 else ~jnp.asarray(real_weights)),
                lambda _: _build_G_face(jnp.conj(psi_xn), jnp.conj(psi_yr),
                                        gemm=gemm, Gij=Gij, phases=phases, mesh=k_unfold_plan.mesh_xy,
                                        band_range=band_range,
                                        prepared_active_gemm=prepared_active_gemm),
                lambda _: jnp.conj(G), operand=None)
    return k_unfold_plan.unfold_operator(
        G, operator_transpose=transposed, right_plan=right_k_unfold_plan)


def windowed_exp_iEt(E, t, E_min=None, E_max=None, *, e_ref=0.0):
    """``exp(-t·(E - e_ref))`` inside the energy window, EXACTLY zero outside.

        windowed_exp_iEt(E, t, E_min, E_max)
            = where((E > E_min) & (E <= E_max), exp(-t·(E - e_ref)), 0)

    THE POINT IS THAT THE WINDOW IS NEVER MATERIALISED.  The caller used to
    build a boolean array the shape of ``E``, keep it alive as a jit operand,
    and hand it in to be multiplied (or ``where``-ed) against the phases.
    Here the predicate is recomputed from ``E`` at the point of use, so it
    fuses with the ``exp`` into one elementwise loop and never becomes a
    buffer.  ``E`` is the immediate argument for exactly that reason.

    DELIBERATELY NO WEIGHT ARGUMENT.  Quadrature weights, α coefficients and
    every other float array stay OUTSIDE: they apply to the result, not to
    the window.  Threading them through here would put a second large float
    operand in the same fused loop and spend the register/cache headroom
    that makes the predicate+exp fusion free (owner ruling 2026-08-09).

    Window convention: HALF-OPEN AND CLOSED AT THE TOP, ``(E_min, E_max]``.
    Abutting windows still tile an energy axis without double-counting a band
    that lands exactly on a boundary; what the closed top additionally fixes is
    the DIRECTION in which such a band is assigned — downward, into the pane
    whose supremum it is.  That direction is not free: every certified
    quadrature rule in this core is built at max(Γ) over its own pane, so a
    pane that did not contain its supremum would evaluate a boundary pole under
    a rule that was never certified to cover it.  Decided and recorded in the
    catalog's ``bin_convention`` field (peer decision 2026-08-09); this helper
    landed as ``[lo, hi)`` and was flipped to match.

    It therefore now AGREES with ``ppm_windows.window_mask_B_bounds``, the
    Σ B-side Ω selector, which has always been ``(lo, hi]``.  The two sides of
    Σ used to assign a pole sitting exactly on a threshold in opposite
    directions; they no longer do, and ``tests/test_windowed_exp_iEt.py`` gates
    that agreement against the B-side helper itself rather than against a
    copied bound.

    Either bound may be ``None`` for a one-sided window; both ``None``
    returns the bare (unwindowed) phase factor.

    Parameters
    ----------
    E : array
        Band energies.  Any shape; the result has the same shape.
    t : scalar
        Complex evolution time.  Real ``t`` → imaginary-time evolution
        (χ₀ minimax quadrature); pure-imaginary ``t`` → real-time
        evolution (Σ_c).  The sign convention is the caller's, matching
        ``build_G_tau``: the exponent is ``-t·(E - e_ref)``.
    E_min, E_max : scalar or None
        Window bounds ``(E_min, E_max]``, in the SAME units and the SAME
        reference as ``E`` (i.e. NOT shifted by ``e_ref`` — ``e_ref`` moves
        only the phase origin, never the window).
    e_ref : scalar
        Energy reference subtracted from ``E`` before the exponential.

    Notes
    -----
    Bit-identity with the materialised form: for a 0/1 mask ``m``,
    ``where(m, exp, 0)`` and ``m * exp`` agree bit-for-bit on every lane
    (``1·x == x`` and ``0·x == 0`` exactly, for finite ``x``), and
    ``where`` additionally stays correct where ``exp`` overflows to inf
    on a lane the window excludes.
    """
    phase = jnp.exp(-t * (E - e_ref))
    if E_min is None and E_max is None:
        return phase
    if E_min is None:
        pred = E <= E_max
    elif E_max is None:
        pred = E > E_min
    else:
        pred = (E > E_min) & (E <= E_max)
    return jnp.where(pred, phase,
                     jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))


def _weighted_tau_phases(enk, t, *, e_ref=0.0, mask=None,
                         band_weight=None, E_min=None, E_max=None):
    """Build the exact per-band factors consumed by the Green contraction."""
    phases = windowed_exp_iEt(enk, t, E_min, E_max, e_ref=e_ref)
    if band_weight is not None:
        band_weight = jnp.reshape(band_weight, enk.shape)
        weight = band_weight.astype(jnp.result_type(phases, band_weight))
        # The selector is a support gate.  Inactive factors must be exact zero
        # even when the unused exponential overflows.
        phases = jnp.where(
            weight != 0.0, phases * weight,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    if mask is not None:
        mask = jnp.reshape(mask, enk.shape)
        phases = jnp.where(
            mask, phases,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    return phases


def _phase_band_interval(phases):
    """Return each parent's smallest half-open interval containing nonzeros."""
    nb = phases.shape[-1]
    index = jnp.arange(nb, dtype=jnp.int32)
    live = phases != 0
    lo = jnp.min(jnp.where(live, index, nb), axis=-1)
    hi = jnp.max(jnp.where(live, index + 1, 0), axis=-1)
    return jnp.minimum(lo, hi), hi


@jax.jit
def prepare_tau_band_range(enk, selector, e_ref, t_nodes, n_active):
    """Find one exact band interval shared by all active evolution times.

    ``t_nodes`` contains the same evolution times passed to
    :func:`build_G_tau`; an MPA caller therefore passes ``1j * nodes.t``.
    Its allocated length is fixed, while scalar ``n_active`` selects the
    prefix belonging to this planned window.  The loop retains two boolean
    arrays shaped like ``enk`` and never stores a time-by-band history.

    The returned ``lo`` and ``hi`` have one entry per parent.  ``invariant``
    is true exactly when every active time has those same enclosing
    intervals.  Support holes inside an interval may differ.  A false result
    tells the caller to keep the dynamic active-range path.  An empty active
    prefix is deliberately not considered invariant.

    A boolean ``selector`` has mask semantics.  Every other dtype has the
    signed/complex band-weight semantics used by :func:`build_G_tau`.
    Energy-window and explicit bracket restrictions are intentionally absent:
    their callers continue to use dynamic intervals.
    """
    if getattr(enk, "ndim", None) != 2:
        raise ValueError("prepare_tau_band_range: enk must have shape (parent, band)")
    if tuple(selector.shape) != tuple(enk.shape):
        raise ValueError("prepare_tau_band_range: selector must match enk.shape")
    if getattr(t_nodes, "ndim", None) != 1:
        raise ValueError("prepare_tau_band_range: t_nodes must be one-dimensional")
    n_active = jnp.asarray(n_active)
    if n_active.shape != () or not jnp.issubdtype(n_active.dtype, jnp.integer):
        raise TypeError("prepare_tau_band_range: n_active must be an integer scalar")

    max_nodes = int(t_nodes.shape[0])
    if max_nodes == 0:
        # fori_loop traces its body even for a zero upper bound.  Returning
        # before defining an indexed body keeps a valid empty plan from
        # tracing an impossible t_nodes[0] access.
        empty = jnp.zeros((enk.shape[0],), dtype=jnp.int32)
        return empty, empty, jnp.asarray(False)
    count_valid = (n_active >= 0) & (n_active <= max_nodes)
    count = jnp.clip(n_active, 0, max_nodes)
    union = jnp.zeros(enk.shape, dtype=jnp.bool_)
    intersection = jnp.ones(enk.shape, dtype=jnp.bool_)
    use_mask = jnp.issubdtype(selector.dtype, jnp.bool_)

    def one_node(i, state):
        any_live, every_live = state
        options = ({"mask": selector} if use_mask
                   else {"band_weight": selector})
        phases = _weighted_tau_phases(
            enk, t_nodes[i], e_ref=e_ref, **options)
        live = phases != 0
        return any_live | live, every_live & live

    union, intersection = jax.lax.fori_loop(
        0, count, one_node, (union, intersection))
    # Stack before reducing the band axis so a band-sharded execution needs
    # one min and one max collective after the loop, never one per time.
    (union_lo, common_lo), (union_hi, common_hi) = _phase_band_interval(
        jnp.stack((union, intersection), axis=0))
    invariant = (count_valid & (count > 0)
                 & jnp.all(union_lo == common_lo)
                 & jnp.all(union_hi == common_hi))
    return union_lo, union_hi, invariant


def build_G_tau(psi_xn, psi_yr, enk, t, *, e_ref=0.0, mask=None,
                band_weight=None, E_min=None, E_max=None,
                layout='face', gemm=None, k_unfold_plan=None, band_range=None,
                trim_zero_bands=False, prepared_active_gemm=None):
    """Contract phases exp(-t*(energy-reference)) with energy windows, identity masks and signed weights."""
    real_weights = not jnp.issubdtype(jnp.result_type(t), jnp.complexfloating)
    if not real_weights:
        real_weights = jnp.imag(t) == 0
    if band_weight is not None and jnp.issubdtype(
            jnp.result_type(band_weight), jnp.complexfloating):
        real_weights = False
    phases = _weighted_tau_phases(
        enk, t, e_ref=e_ref, mask=mask, band_weight=band_weight,
        E_min=E_min, E_max=E_max)
    if prepared_active_gemm is not None and band_range is not None:
        raise ValueError(
            "build_G_tau: prepared_active_gemm and band_range are mutually exclusive")
    if trim_zero_bands and prepared_active_gemm is None:
        # Exact support, separately for every parent k. No numerical cutoff:
        # interior holes retain their zero weights, while outer zero columns
        # never enter the distributed contraction.
        lo, hi = _phase_band_interval(phases)
        if band_range is not None:
            lo = jnp.maximum(lo, band_range[0])
            hi = jnp.minimum(hi, band_range[1])
        band_range = (jnp.minimum(lo, hi), hi)
    return build_G(
        psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm,
        k_unfold_plan=k_unfold_plan, real_weights=real_weights,
        band_range=band_range, prepared_active_gemm=prepared_active_gemm)
