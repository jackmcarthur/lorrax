"""Contract canonical centroid faces with one planned GEMM and optionally unfold the parent operator."""
import jax.numpy as jnp

from common.contract_bands import merge_spin_centroid, split_spin_centroid


def _build_G_face(psi_mun, psi_nmu, *, gemm, Gij=None, phases=None):
    """Form the weighted direct/conjugate outer product in the centroid-major GEMM order."""
    if Gij is not None:
        raise NotImplementedError(
            "build_G(layout='face'): an explicit dense Gij band-operator "
            "is not ported for this layout (report obstacle #4's named "
            "escape hatch — 'refuse that uncommon combination under "
            "low_mem_bands=true by name').  Production diagonal-occupation "
            "weights go through `phases` instead, which IS supported; a "
            "genuinely dense (non-diagonal) band operator would need its "
            "own face-sharded two-GEMM contract (like the projector's), "
            "not this identity/diagonal-weight path.")
    nk_, s_, mu_l_, n_ = psi_mun.shape
    nk_r_, n_r_, s_r_, mu_r_ = psi_nmu.shape
    if nk_r_ != nk_ or n_r_ != n_ or s_r_ != s_:
        raise ValueError(
            "build_G(layout='face'): left psi_mun and right psi_nmu must "
            "share (nk, nb, nspinor); got "
            f"{psi_mun.shape} and {psi_nmu.shape}.")
    A = merge_spin_centroid(psi_mun, 1, 2)          # (nk, mu*s, n) P(_,'x','y')
    if phases is not None:
        w = phases.astype(A.dtype)                  # (nk, n)
        A = A * w[:, None, :]
    B = merge_spin_centroid(jnp.conj(psi_nmu), 2, 3)  # (nk, n, mu*s) P(_,'x','y')
    # Eager operations may erase singleton mesh axes before the GEMM boundary.
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharding = NamedSharding(gemm.mesh, P(None, 'x', 'y'))
    A = lax.with_sharding_constraint(A, sharding)
    B = lax.with_sharding_constraint(B, sharding)
    G_flat = gemm(A, B)                              # (nk, mu*s, mu*s) P(_,'x','y')
    # Undo BOTH merges, restoring legacy's rectangular
    # (k, s, μ_left_X, s', μ_right_Y) axis order.  The historical
    # square path is the mu_l_ == mu_r_ specialization; accepting distinct
    # endpoint extents here lets the canonical G builder serve mixed C/T
    # response blocks without a second Green-function implementation.
    # The row merge sits at axis 1, the (still-merged) column pair shifts
    # from axis 2 to axis 3 once the first split inserts an axis.
    G = split_spin_centroid(G_flat, 1, s_, mu_l_)
    G = split_spin_centroid(G, 3, s_, mu_r_)
    return G


def build_G(psi_xn, psi_yr, *, Gij=None, phases=None, layout='face',
           gemm=None, k_unfold_plan=None):
    """Contract direct and conjugated faces, then apply the typed parent transport when supplied."""
    if layout != 'face':
        raise ValueError(f"build_G: layout must be 'face', got {layout!r}")
    if gemm is None:
        raise ValueError("build_G requires gemm=<distrib_la.GemmPlan> built outside the kernel.")
    G = _build_G_face(psi_xn, psi_yr, gemm=gemm, Gij=Gij, phases=phases)
    if k_unfold_plan is not None:
        # The contraction above remains the sole Green-function builder.
        # A CentroidKUnfoldPlan merely changes its k input from full children
        # to raw parents and transports the completed two-endpoint operator
        # to full k, in the run's packed centroid order.
        G = k_unfold_plan.unfold_operator(G)
    return G


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


def build_G_tau(psi_xn, psi_yr, enk, t, *, e_ref=0.0, mask=None,
                band_weight=None, E_min=None, E_max=None,
                layout='face', gemm=None, k_unfold_plan=None):
    """Contract phases exp(-t*(energy-reference)) with energy windows, identity masks and signed weights."""
    phases = windowed_exp_iEt(enk, t, E_min, E_max, e_ref=e_ref)
    if band_weight is not None:
        band_weight = jnp.reshape(band_weight, enk.shape)
        weight = band_weight.astype(phases.dtype)
        # A sparse tuple selector is an exact support gate, not merely a
        # post-hoc scale.  Complex delivered nodes can overflow the phase on
        # an UNSELECTED high-energy band even though the planner certified
        # every selected factor.  Multiplication would turn that lane into
        # ``0 * inf = nan`` and poison the whole G contraction.  ``where``
        # preserves the identical multiplication on every live lane and the
        # exact mathematical zero on every dead one.
        phases = jnp.where(
            weight != 0.0, phases * weight,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    if mask is not None:
        # mask gates phases per (k, n), so it must share enk's shape.  Some Σ
        # branches deliver it as (1, nk, nb) on a 1×1 processor mesh — the occ
        # array carries a leading nspin axis that a 2×2 mesh squeezes but a 1×1
        # mesh does not.  Reshape first so ``where`` can't broadcast phases up to
        # 3-D and violate build_G's 'ksxn,kn,knty' contract (crashed GN-PPM on 1 GPU).
        mask = jnp.reshape(mask, enk.shape)
        phases = jnp.where(mask, phases,
                           jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    if k_unfold_plan is not None:
        anti = k_unfold_plan.sym.operation_rows(k_unfold_plan.sym_idx)[2]
        if anti.any():
            # Equal endpoint faces give G^T = psi* f psi^T at the SAME
            # complex time; conjugating f would reverse the lifetime.
            parent = build_G(psi_xn, psi_yr, phases=phases,
                             layout=layout, gemm=gemm)
            transposed = build_G(jnp.conj(psi_xn), jnp.conj(psi_yr),
                                 phases=phases, layout=layout, gemm=gemm)
            return k_unfold_plan.unfold_operator(
                parent, operator_transpose=transposed)
    return build_G(
        psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm,
        k_unfold_plan=k_unfold_plan)
