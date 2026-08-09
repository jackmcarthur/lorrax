"""Unified Green's function builder for ISDF-basis GW.

  G_μν(k) = Σ_{ij} ψ_i(μ) W_ij ψ*_j(ν)

Convention: psi_xn is direct (μ side). psi_yr is conjugated (ν side).
This matches the tested COHSEX convention throughout gw_jax.py.
"""
import jax.numpy as jnp


def build_G(psi_xn, psi_yr, *, Gij=None, phases=None):
    """Build G_μν(k).  Returns (nk, s, μ_X, s, μ_Y) flat-k.

    psi_xn:  (nk, s, μ_X, nb) — direct (μ side)
    psi_yr:  (nk, nb, s, μ_Y) — conjugated internally (ν side)
    Gij:     (nk, nb, nb) or None — band-space matrix. None → identity.
    phases:  (nk, nb) complex or None — per-band weights. None → ones.
    """
    if Gij is not None and phases is not None:
        p = phases.astype(jnp.complex128)
        return jnp.einsum('ksxi,kij,kjty->ksxty',
            psi_xn, p[:, :, None] * Gij * p[:, None, :],
            jnp.conj(psi_yr), optimize=True)
    if Gij is not None:
        return jnp.einsum('ksxi,kij,kjty->ksxty',
            psi_xn, Gij, jnp.conj(psi_yr), optimize=True)
    if phases is not None:
        return jnp.einsum('ksxn,kn,knty->ksxty',
            psi_xn, phases.astype(jnp.complex128), jnp.conj(psi_yr), optimize=True)
    return jnp.einsum('ksxn,knty->ksxty',
        psi_xn, jnp.conj(psi_yr), optimize=True)


def windowed_exp_iEt(E, t, E_min=None, E_max=None, *, e_ref=0.0):
    """``exp(-t·(E - e_ref))`` inside the energy window, EXACTLY zero outside.

        windowed_exp_iEt(E, t, E_min, E_max)
            = where((E >= E_min) & (E < E_max), exp(-t·(E - e_ref)), 0)

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

    Window convention: HALF-OPEN ``[E_min, E_max)``, so abutting windows
    tile an energy axis without double-counting a band that lands exactly on
    a boundary.  Either bound may be ``None`` for a one-sided window; both
    ``None`` returns the bare (unwindowed) phase factor.

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
        Half-open window bounds, in the SAME units and the SAME reference
        as ``E`` (i.e. NOT shifted by ``e_ref`` — ``e_ref`` moves only the
        phase origin, never the window).
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
        pred = E < E_max
    elif E_max is None:
        pred = E >= E_min
    else:
        pred = (E >= E_min) & (E < E_max)
    return jnp.where(pred, phase,
                     jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))


def build_G_tau(psi_xn, psi_yr, enk, t, *, e_ref=0.0, mask=None,
                E_min=None, E_max=None):
    """G(t)_k(μ, ν) = Σ_n ψ_n(μ) · exp(-t · (e_n - e_ref)) · ψ_n*(ν).

    Unified time-evolution G builder shared by χ₀ (imaginary-time) and
    Σ (real-time).  The imaginary/real-time split is entirely in the
    caller's choice of ``t``:

        real    t → imaginary-time evolution  (χ₀ minimax quadrature).
        pure-i  t → real-time    evolution   (Σ_c ppm/minimax quadrature).

    psi_xn:  (nk, s, μ_X, nb)  direct (μ side), c128
    psi_yr:  (nk, nb, s, μ_Y)  conjugated internally (ν side), c128
    enk:     (nk, nb)           band energies, f64
    t:       scalar              complex evolution time
    e_ref:   scalar              energy reference subtracted from enk before phase
    E_min,
    E_max:   scalar or None      half-open energy window [E_min, E_max) applied to
                                 enk WITHOUT materialising a mask — see
                                 ``windowed_exp_iEt``.  This is the preferred
                                 spelling for any window that is an ENERGY RANGE.
    mask:    (nk, nb) bool or None   band-IDENTITY selector (Σ's ``mask_A``:
                                 which bands are valence vs conduction).  Kept
                                 as an array because band identity is NOT an
                                 energy interval in general — degenerate and
                                 padded bands break the equivalence.  An
                                 energy-range window must NOT be passed here;
                                 use E_min/E_max, which costs no buffer.

    Returns (nk, s, μ_X, s, μ_Y) flat-k.  Thin wrapper around ``build_G``
    with phases = windowed_exp_iEt(enk, t, E_min, E_max, e_ref=e_ref),
    optionally further gated by the band-identity ``mask``.
    """
    phases = windowed_exp_iEt(enk, t, E_min, E_max, e_ref=e_ref)
    if mask is not None:
        # mask gates phases per (k, n), so it must share enk's shape.  Some Σ
        # branches deliver it as (1, nk, nb) on a 1×1 processor mesh — the occ
        # array carries a leading nspin axis that a 2×2 mesh squeezes but a 1×1
        # mesh does not.  Reshape first so ``where`` can't broadcast phases up to
        # 3-D and violate build_G's 'ksxn,kn,knty' contract (crashed GN-PPM on 1 GPU).
        mask = jnp.reshape(mask, enk.shape)
        phases = jnp.where(mask, phases,
                           jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    return build_G(psi_xn, psi_yr, phases=phases)
