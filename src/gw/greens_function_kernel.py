"""Build parent Green operators and transport them with typed local symmetry actions."""
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp

from common.contract_bands import merge_spin_centroid, split_spin_centroid


def _build_G_face(psi_mun, psi_nmu, *, gemm, Gij=None, phases=None, mesh=None):
    """Contract band-replicated faces locally or band-distributed faces with their GEMM plan."""
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
    if phases is not None:
        w = phases.astype(A.dtype)                  # (nk, n)
        A = A * w[:, None, :]
    B = merge_spin_centroid(jnp.conj(psi_nmu), 2, 3)  # (nk, n, mu*s) P(_,'x','y')
    # Eager operations may erase singleton mesh axes before the GEMM boundary.
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P
    A = lax.with_sharding_constraint(A, gemm.in_sharding_a)
    B = lax.with_sharding_constraint(B, gemm.in_sharding_b)
    G_flat = gemm(A, B)                              # (nk, mu*s, mu*s) P(_,'x','y')
    G = split_spin_centroid(G_flat, 1, s_, mu_l_)
    G = split_spin_centroid(G, 3, s_, mu_r_)
    return G


def build_G(psi_xn, psi_yr, *, Gij=None, phases=None, layout='face',
           gemm=None, k_unfold_plan=None, right_k_unfold_plan=None, real_weights=None):
    """Build parent operators and transport both typed endpoints without processor exchange."""
    if layout not in ('face', 'axis') or gemm is None:
        raise ValueError("build_G requires canonical faces and a GEMM plan or typed parent plan.")
    G = _build_G_face(psi_xn, psi_yr, gemm=gemm, Gij=Gij, phases=phases,
                      mesh=None if k_unfold_plan is None else k_unfold_plan.mesh_xy)
    if k_unfold_plan is None:
        return G
    transposed = None
    if np.any(np.asarray(k_unfold_plan.sym_idx) >= k_unfold_plan.n_sym_spatial):
        if phases is None or not jnp.issubdtype(phases.dtype, jnp.complexfloating):
            transposed = jnp.conj(G)
        else:
            transposed = jax.lax.cond(
                (jnp.any(jnp.imag(phases) != 0) if real_weights is None
                 else ~jnp.asarray(real_weights)),
                lambda _: _build_G_face(jnp.conj(psi_xn), jnp.conj(psi_yr),
                                        gemm=gemm, Gij=Gij, phases=phases, mesh=k_unfold_plan.mesh_xy),
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
    return build_G(
        psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm,
        k_unfold_plan=k_unfold_plan, real_weights=jnp.imag(t) == 0)
