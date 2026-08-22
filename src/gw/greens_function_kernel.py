"""Unified Green's function builder for ISDF-basis GW.

  G_μν(k) = Σ_{ij} ψ_i(μ) W_ij ψ*_j(ν)

Two layouts, selected by the STATIC ``layout`` argument (never a traced
value — a Python string a caller passes once per kernel build, exactly
the ``Wavefunctions.layout`` tag; see ``reports/gwjax_low_mem_bands_audit_
2026-08-22/report.md``):

``layout='legacy'`` (default) — psi_xn is direct (μ side), psi_yr is
conjugated (ν side).  This matches the tested COHSEX convention
throughout gw_jax.py and is BYTE-IDENTICAL to the code this module
shipped before ``low_mem_bands`` existed: every legacy branch below is
the original body, unmoved and untouched.

``layout='face'`` — the two-face carrier's ``psi_mun``/``psi_nmu``
(``gw.wavefunction_bundle``, both un-conjugated, both full [b0,b4) band
extent — see obstacle #3, band windows are not legal face matrices, and
:func:`Wavefunctions.band_mask` is the bring-up substitute).  ``build_G``
computes ``(psi_mun * w) @ conj(psi_nmu)`` as ONE planned N,N GEMM
(``distrib_la.gemm_plan``, a ``GemmPlan`` the CALLER built once outside
its hot loop — never resolved/probed here) rather than a dense-band einsum,
because multi-rank cuBLASMp accepts only ``P(None,'x','y')`` matrices, and
``psi_mun``/``psi_nmu`` are exactly those two 2-D-sharded orientations.

GEMM-seam flatten order — a MEASURED correction to the audit report
------------------------------------------------------------------
The report's own verdict describes "flattening (s,mu) ... at the GEMM
seam" without specifying which of the two sits major.  ``psi_mun``/
``psi_nmu`` store the spinor axis ``s`` OUTER and the centroid axis
``mu``/``nu`` INNER (``wavefunction_bundle.PSI_MUN_SPEC`` /
``PSI_NMU_SPEC``: ``s`` is a REPLICATED axis, ``mu``/``nu`` is the
MESH-SHARDED one).  Merging them in THAT order — replicated axis major,
sharded axis minor — is **not** a free reshape: traced on an emulated 2x2
mesh, it lowers to a genuine ``all-to-all`` collective, one per rank, at
every G build.  Merging in the OTHER order — sharded axis major,
replicated axis minor (a ``jnp.swapaxes`` then reshape) — is a provably
free local reshape: zero collectives in the compiled HLO, bit-exact
against a plain NumPy transpose+reshape, and every rank's local shard
lands exactly where a genuinely chunked ``P(None,'x','y')`` array would
put it.  Both claims are pinned as a positive/negative test pair in
``tests/test_contract_bands.py``
(``test_merge_split_spin_centroid_roundtrip_and_no_collective``,
``test_merge_spin_centroid_storage_order_needs_an_all_to_all``).
:func:`common.contract_bands.merge_spin_centroid` is
this majority-order merge, shared with the face projector for the same
reason (:mod:`common.contract_bands`'s own two-GEMM projection needs the
identical fix on both O's spin pairs).  ns ∈ {1, 2} in every deck this
module has been run against; the merge (not a per-spin loop) keeps the
GENERAL spin-OFF-DIAGONAL G a spinor calculation needs — see the module's
own einsum below, which does NOT contract s against t.
"""
import jax.numpy as jnp

from common.contract_bands import merge_spin_centroid, split_spin_centroid


def _legacy_build_G(psi_xn, psi_yr, *, Gij=None, phases=None):
    """The exact pre-``low_mem_bands`` body.  UNTOUCHED — do not edit this
    function to add face-layout behaviour; add a sibling instead."""
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


def _face_build_G(psi_mun, psi_nmu, *, gemm, Gij=None, phases=None):
    """Face-layout G via one planned N,N GEMM.  See module docstring.

    psi_mun: (nk, s, μ, n)  P(None, None, 'x', 'y') — un-conjugated, direct.
    psi_nmu: (nk, n, s, μ)  P(None, 'x', None, 'y') — un-conjugated; this
             function applies the conjugate (matching legacy's psi_yr
             convention: the SECOND operand is conjugated internally).
    gemm:    a ``distrib_la.GemmPlan`` built ONCE by the caller (typically
             a kernel-factory closure, mirroring ``_Gv_fftn`` et al. in
             ``w_isdf.py``) with ``m=k=n=mu*ns``... no: ``m=n=mu*ns``,
             ``k=nb_full``, ``nq=nk``, matching this call's operand shapes
             exactly — see :func:`distrib_la.gemm_plan`.  Resolving or
             warming a plan HERE would defeat the entire point of a
             planned surface (it would dlopen/probe/compile on every G
             build, inside whatever ``lax.scan``/``jax.jit`` called this).
    """
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
    nk_, s_, mu_, n_ = psi_mun.shape
    A = merge_spin_centroid(psi_mun, 1, 2)          # (nk, mu*s, n) P(_,'x','y')
    if phases is not None:
        w = phases.astype(A.dtype)                  # (nk, n)
        A = A * w[:, None, :]
    B = merge_spin_centroid(jnp.conj(psi_nmu), 2, 3)  # (nk, n, mu*s) P(_,'x','y')
    G_flat = gemm(A, B)                              # (nk, mu*s, mu*s) P(_,'x','y')
    # Undo BOTH merges, restoring legacy's (k, s, μ_X, s', μ_Y) axis order.
    # The row merge sits at axis 1, the (still-merged) column pair shifts
    # from axis 2 to axis 3 once the first split inserts an axis.
    G = split_spin_centroid(G_flat, 1, s_, mu_)       # (nk, s, mu, mu*s)
    G = split_spin_centroid(G, 3, s_, mu_)            # (nk, s, mu, s', mu')
    return G


def build_G(psi_xn, psi_yr, *, Gij=None, phases=None, layout='legacy',
           gemm=None):
    """Build G_μν(k).

    ``layout='legacy'`` (default): returns (nk, s, μ_X, s, μ_Y) flat-k,
    exactly as before — ``psi_xn``/``psi_yr`` are the legacy 4-copy bundle
    fields, ``Gij``/``phases`` as documented on :func:`_legacy_build_G`.

    ``layout='face'``: ``psi_xn``/``psi_yr`` are actually ``psi_mun``/
    ``psi_nmu`` (the argument NAMES stay the SAME two slots — first operand
    direct, second conjugated internally — across both layouts on purpose,
    so a caller's call shape does not change shape when it switches
    layout).  ``gemm`` (a ``distrib_la.GemmPlan``) is then required.  See
    the module docstring for the GEMM-seam design and
    :func:`_face_build_G` for the operand contract.
    """
    if layout == 'legacy':
        return _legacy_build_G(psi_xn, psi_yr, Gij=Gij, phases=phases)
    if layout != 'face':
        raise ValueError(f"build_G: layout must be 'legacy' or 'face', got {layout!r}")
    if gemm is None:
        raise ValueError(
            "build_G(layout='face') requires gemm=<distrib_la.GemmPlan>, "
            "built ONCE outside this call (see distrib_la.gemm_plan and "
            "this module's docstring) — never resolved here.")
    return _face_build_G(psi_xn, psi_yr, gemm=gemm, Gij=Gij, phases=phases)


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
                layout='legacy', gemm=None):
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
    E_max:   scalar or None      energy window (E_min, E_max] — closed at the top —
                                 applied to enk WITHOUT materialising a mask; see
                                 ``windowed_exp_iEt``.  This is the preferred
                                 spelling for any window that is an ENERGY RANGE.
    mask:    (nk, nb) bool or None   band-IDENTITY selector (Σ's ``mask_A``:
                                 which bands are valence vs conduction).  Kept
                                 as an array because band identity is NOT an
                                 energy interval in general — degenerate and
                                 padded bands break the equivalence.  An
                                 energy-range window must NOT be passed here;
                                 use E_min/E_max, which costs no buffer.
    band_weight: (nk, nb) real or complex or None.  A physical linear
                 Green-function weight applied after the phase is formed.
                 Fractional-response callers use f or 1-f here.  The weight
                 stays out of windowed_exp_iEt so that helper remains the
                 fused phase/window primitive.  Weights are never
                 square-rooted or clipped.
    layout, gemm: forwarded verbatim to :func:`build_G` — see its
                 docstring.  Under ``layout='face'`` ``enk``/``mask``/
                 ``band_weight`` are expected at the FULL [b0,b4) extent
                 (``Wavefunctions.band_mask`` is the bring-up helper for
                 turning a logical band slice into this ``mask``, per
                 report §3): this function's own phase math is IDENTICAL
                 either way, only how the resulting weight reaches the ψ
                 operands differs.

    Returns (nk, s, μ_X, s, μ_Y) flat-k.  Thin wrapper around ``build_G``
    with phases = windowed_exp_iEt(enk, t, E_min, E_max, e_ref=e_ref),
    optionally further gated by the band-identity ``mask``.
    """
    phases = windowed_exp_iEt(enk, t, E_min, E_max, e_ref=e_ref)
    if band_weight is not None:
        band_weight = jnp.reshape(band_weight, enk.shape)
        phases = phases * band_weight.astype(phases.dtype)
    if mask is not None:
        # mask gates phases per (k, n), so it must share enk's shape.  Some Σ
        # branches deliver it as (1, nk, nb) on a 1×1 processor mesh — the occ
        # array carries a leading nspin axis that a 2×2 mesh squeezes but a 1×1
        # mesh does not.  Reshape first so ``where`` can't broadcast phases up to
        # 3-D and violate build_G's 'ksxn,kn,knty' contract (crashed GN-PPM on 1 GPU).
        mask = jnp.reshape(mask, enk.shape)
        phases = jnp.where(mask, phases,
                           jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    return build_G(psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm)
