"""Build parent Green operators and transport them with typed local symmetry actions."""
import dataclasses
from functools import partial

import jax

import numpy as np
import jax.numpy as jnp

from common.contract_bands import merge_spin_centroid


def face_green_product(A, B, mesh, phases, band_range, n_full=None, partner=False):
    """``A·diag(w)·B`` on band-distributed faces by the 2-D distributed GEMM.

    ``A`` ``(nq, M, N_b)`` and ``B`` ``(nq, N_b, N)`` are both
    ``P(None,'x','y')``.  ``w`` is the phase row, zero outside the per-row
    ``band_range``.  ``distrib_la.panel_matmul`` runs a batched SUMMA: band
    panels of at most ``N_b/p`` columns are all-gathered (A over y, B over
    x), every k in each exchange and each local GEMM, and multiplied into the
    rank's own output tile, so no reduction follows and no rank holds a
    band-complete panel.  Each panel's local product runs only over the
    rows' ``band_range`` (the local active-range GEMM).  The two live panels
    are bounded by one full-k Green tile, ``16·N_k·(M/p_x)·(N/p_y)`` bytes
    (``N_k = n_full``, the parents' full zone; ``nq`` when the faces are
    already at full k), which every Green-building stage reserves.
    ``partner``: also the conjugate-face Green ``conj(A)·diag(w)·conj(B)``
    from the same panel exchange (``_build_G_face(pair=True)``).

    Why this route (GEMM2D, A100-40GB, warm; ms per build, CrI3 8x8 N_b 144 /
    Fe 8³ N_b 120, valence windows):

    ==========================================  =============  =============
    route                                       P4             P16 (4x4)
    ==========================================  =============  =============
    band-complete gather (retired)              2.68 / 6.59    1.86 / 5.14
    batched SUMMA, panels <= N_b/p (this)       3.22 / 6.79    2.57 / 7.37
    two N_b/2 panels (all bands live: refused)  (same as this) 1.85 / 5.95
    cuBLASMp SUMMA, one call per k              6.68 / 10.95   6.88 / 5.11
    ==========================================  =============  =============

    CrI3 8x8 one-shot Σ τ sweep at P16: gather 2.30 s, cuBLASMp 2.57 s,
    this route 2.64 s at v1 (before the one-exchange partner pair).  cuBLASMp
    loses because it runs one SUMMA per k (nq calls of p broadcast rounds) and
    joins the XLA stream by events at entry and exit, so it overlaps nothing;
    this route moves the same bytes once per panel for every k.  Its
    prefetched gather does not overlap the GEMM either under XLA's default
    scheduler (``distrib_la.panel_matmul``'s lessons).  At P16 the exchange
    is most of a build (all-gathers 1.2 of 1.85 ms on CrI3).
    """
    from distrib_la import panel_matmul

    nq, m, nb = (int(v) for v in A.shape)
    n = int(B.shape[-1])
    weight = None if phases is None else phases.astype(A.dtype)
    bounds = None
    if band_range is not None:
        lo, hi = (jnp.broadcast_to(jnp.reshape(jnp.asarray(v, jnp.int32), (-1,)), (nq,))
                  for v in band_range)
        idx = jnp.arange(nb)[None, :]
        live = (idx >= lo[:, None]) & (idx < hi[:, None])
        weight = (live.astype(A.dtype) if weight is None
                  else jnp.where(live, weight, jnp.zeros((), A.dtype)))
        # The interval also bounds the gathered local product: the columns
        # outside it are zero, so skipping them is the same product.
        hi = jnp.clip(hi, 0, nb)
        bounds = jnp.stack([jnp.clip(lo, 0, hi), hi], axis=1)
    tile_bytes = green_panel_bytes(n_rows=int(n_full or nq), m=m, n=n, mesh=mesh)
    return panel_matmul(A, B, mesh=mesh, panel_bytes=tile_bytes, bounds=bounds, weights=weight,
                        partner=partner)


def green_panel_bytes(*, n_rows, m, n, mesh, room=None):
    """The transient band-panel budget of one Green build, per rank.

    One Green tile, ``16·n_rows·(m/p_x)·(n/p_y)`` (the stage reserves it), which
    bounds the two live SUMMA panels (each at most ``N_b/p_x`` bands, so the tile
    rarely binds); at most ``room`` when the caller's ledger has less beside its
    live stages; never below one contraction column ``16·n_rows·(m/p_x + n/p_y)``
    (``distrib_la.panel_matmul``'s floor).
    """
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    tile = 16 * int(n_rows) * (int(m) // px) * (int(n) // py)
    column = 16 * int(n_rows) * (int(m) // px + int(n) // py)
    return int(max(column, tile if room is None else min(tile, int(room))))


def _phases_on_face_carrier(phases, band_range, nb):
    """A Σ-sum phase row on the faces' band carrier (the one band carrier of a build).

    The faces carry every loaded band (max of the χ and Σ sums, padded); a Σ
    consumer's phases stop at its own band sum (``BandSlices.sigma_sum``).  The
    bands past that sum take exact-zero weight, and a given ``band_range`` ends
    at the phase row, so those bands are neither weighted nor contracted.
    """
    n_w = int(phases.shape[-1])
    if n_w > nb:
        raise ValueError(
            f"build_G: phases span {n_w} bands but the faces carry {nb}; a band "
            "sum cannot exceed the loaded band carrier.")
    phases = jnp.pad(phases, [(0, 0)] * (phases.ndim - 1) + [(0, nb - n_w)])
    if band_range is not None:
        lo, hi = band_range
        band_range = (lo, jnp.minimum(jnp.asarray(hi, jnp.int32), n_w))
    return phases, band_range


def green_right_operand(psi_nmu):
    """The Green GEMM's right operand ``conj(ψ_nmu)``, merged centroid-major ``(nk, n, ν·s)``.

    Every build forms it from its ``ψ_nmu`` (a transposed copy of the whole
    face); a caller that builds many Greens from one band-complete face forms
    it once and passes it as ``right`` (:func:`build_G_tau`).
    """
    return merge_spin_centroid(jnp.conj(psi_nmu), 2, 3)


def _build_G_face(psi_mun, psi_nmu, *, gemm, Gij=None, phases=None, mesh=None,
                  band_range=None, prepared_active_gemm=None, n_full=None, pair=False,
                  right=None):
    """Contract band-replicated faces locally or band-distributed faces with their GEMM plan.

    Returns the Green ``(nk, mu_X, s, nu_Y, s')``: centroid-major, the
    GEMM's own merged endpoint order split by a reshape.  ``pair`` (face
    route only): also the conjugate-face partner ``conj(A)·diag(w)·conj(B)``
    from the SAME panel exchange (each gathered panel conjugated before its
    own local GEMM; no conjugated tile).  ``right`` (local GEMMs): the right
    operand already formed (:func:`green_right_operand`); ``psi_nmu`` is then
    not read.
    """
    if Gij is not None:
        raise NotImplementedError("Green faces support diagonal band weights, not dense Gij.")
    nk_, s_, mu_l_, n_ = psi_mun.shape
    if right is not None:
        nk_r_, n_r_, merged_r_ = right.shape
        s_r_, mu_r_ = s_, merged_r_ // s_
    else:
        nk_r_, n_r_, s_r_, mu_r_ = psi_nmu.shape
    if nk_r_ != nk_ or n_r_ != n_ or s_r_ != s_:
        raise ValueError(
            "build_G(layout='face'): left psi_mun and right psi_nmu must "
            "share (nk, nb, nspinor); got "
            f"{psi_mun.shape} and {(nk_r_, n_r_, s_r_, mu_r_)}.")
    if phases is not None and int(phases.shape[-1]) != n_:
        phases, band_range = _phases_on_face_carrier(phases, band_range, n_)
    A = merge_spin_centroid(psi_mun, 1, 2)          # (nk, mu*s, n) P(_,'x','y')
    face = getattr(gemm, "backend", "local") != "local"
    if (phases is not None and band_range is None
            and prepared_active_gemm is None and not face):
        w = phases.astype(A.dtype)                  # (nk, n)
        A = A * w[:, None, :]
    # (nk, n, mu*s) P(_,'x','y')
    B = green_right_operand(psi_nmu) if right is None else right
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
    elif face:
        # The phases scale each panel's slice of A on its way into the gather.
        if pair:
            G_flat, partner = face_green_product(A, B, gemm.mesh, phases, band_range,
                                                 n_full=n_full, partner=True)
            return (G_flat.reshape(nk_, mu_l_, s_, mu_r_, s_),
                    partner.reshape(nk_, mu_l_, s_, mu_r_, s_))
        G_flat = face_green_product(A, B, gemm.mesh, phases, band_range, n_full=n_full)
    else:
        G_flat = (gemm(A, B) if band_range is None
                  else gemm.active_range(A, B, *band_range, weights=phases))
    # (nk, mu*s, nu*s), distributed over both centroid axes.  The merged
    # endpoint order is centroid-major (``mu*ns + s``, merge_spin_centroid's
    # own collective-free direction), so the split is a pure reshape: the
    # Green is stored ``(nk, mu_X, s, nu_Y, s')`` and is never transposed to
    # a spin-major order.  The parent-k unfold transports this order as-is.
    return G_flat.reshape(nk_, mu_l_, s_, mu_r_, s_)


@dataclasses.dataclass(frozen=True)
class ParentGreen:
    """The raw-parent Green and what its typed unfold needs.

    ``G`` ``(n_parent, mu, s, nu, s')`` centroid-major; ``transpose`` the
    partner an antiunitary row reads (the conjugate-face Green), ``None`` when
    the plan has no antiunitary row or when ``conj_partner``: weights known
    real, so the partner is ``conj(G)`` and the fused doors read it from ``G``
    on their load (``conj_partner=True``) instead of storing a second tile.
    ``k_unfold_plan.unfold_operator(G, operator_transpose=partner())`` is the
    full-k Green; ``ffi.fft.make_kconv_klead_unfold`` reads the pair directly.
    A pytree: ``G`` and ``transpose`` are leaves, ``conj_partner`` is static.
    """
    G: jax.Array
    transpose: jax.Array | None = None
    conj_partner: bool = False

    def partner(self):
        """The partner tile itself (``conj(G)`` materialized when ``conj_partner``)."""
        return jnp.conj(self.G) if self.conj_partner else self.transpose


jax.tree_util.register_pytree_node(
    ParentGreen, lambda p: ((p.G, p.transpose), p.conj_partner),
    lambda conj, leaves: ParentGreen(leaves[0], leaves[1], conj))


def has_antiunitary_rows(k_unfold_plan) -> bool:
    """Whether a raw-parent plan unfolds some full-k row antiunitarily (its Green then
    needs a partner, :class:`ParentGreen`)."""
    return bool(np.any(np.asarray(k_unfold_plan.sym_idx) >= k_unfold_plan.n_sym_spatial))


def build_G_parents(psi_xn, psi_yr, *, Gij=None, phases=None, layout='face', gemm=None,
                    k_unfold_plan, real_weights=None, band_range=None,
                    prepared_active_gemm=None, right=None) -> ParentGreen:
    """The parent Green and its antiunitary partner, before the typed unfold (see :class:`ParentGreen`).

    ``right`` (a local GEMM): the right operand formed once by the caller
    (:func:`green_right_operand`).  The partner ``conj(A)·diag(w)·conj(B)``
    is then ``conj(A·diag(conj w)·B)``, the same two operands at conjugate
    weights, so no conjugated face is formed.
    """
    if right is not None:
        if getattr(gemm, "backend", "local") != "local" or Gij is not None or phases is None:
            raise ValueError("build_G_parents(right=...) takes a local GEMM and band phases")

        def build(w):
            return _build_G_face(psi_xn, None, gemm=gemm, phases=w, mesh=k_unfold_plan.mesh_xy,
                                 band_range=band_range, prepared_active_gemm=prepared_active_gemm,
                                 n_full=k_unfold_plan.n_full, right=right)
        G = build(phases)
        if not has_antiunitary_rows(k_unfold_plan):
            return ParentGreen(G, None)
        if (real_weights is True
                or not jnp.issubdtype(phases.dtype, jnp.complexfloating)):
            return ParentGreen(G, None, conj_partner=True)
        if real_weights is not False:
            raise ValueError("build_G_parents(right=...) states real_weights")
        return ParentGreen(G, jnp.conj(build(jnp.conj(phases))))
    if layout not in ('face', 'axis'):
        raise ValueError("build_G requires canonical faces with layout=face or axis.")
    if gemm is None:
        raise ValueError("build_G requires a GEMM plan or typed parent plan-provided GEMM callable.")
    antiunitary = has_antiunitary_rows(k_unfold_plan)
    if (antiunitary and getattr(gemm, "backend", "local") != "local"
            and prepared_active_gemm is None and Gij is None and phases is not None
            and real_weights is not True
            and jnp.issubdtype(phases.dtype, jnp.complexfloating)):
        # The face route builds G and its conjugate-face partner from ONE panel
        # exchange (two weightings, w and w*); at real phases the partner is
        # conj(G) exactly, so no device predicate is needed.
        G, transposed = _build_G_face(psi_xn, psi_yr, gemm=gemm, phases=phases,
                                      mesh=k_unfold_plan.mesh_xy, band_range=band_range,
                                      n_full=k_unfold_plan.n_full, pair=True)
        return ParentGreen(G, transposed)
    G = _build_G_face(psi_xn, psi_yr, gemm=gemm, Gij=Gij, phases=phases,
                      mesh=k_unfold_plan.mesh_xy,
                      band_range=band_range,
                      prepared_active_gemm=prepared_active_gemm,
                      n_full=k_unfold_plan.n_full)
    transposed = None
    if antiunitary:
        if (real_weights is True or phases is None
                or not jnp.issubdtype(phases.dtype, jnp.complexfloating)):
            return ParentGreen(G, None, conj_partner=True)
        elif real_weights is False:
            transposed = _build_G_face(jnp.conj(psi_xn), jnp.conj(psi_yr),
                                       gemm=gemm, Gij=Gij, phases=phases, mesh=k_unfold_plan.mesh_xy,
                                       band_range=band_range,
                                       prepared_active_gemm=prepared_active_gemm,
                                       n_full=k_unfold_plan.n_full)
        else:
            transposed = jax.lax.cond(
                (jnp.any(jnp.imag(phases) != 0) if real_weights is None
                 else ~jnp.asarray(real_weights)),
                lambda _: _build_G_face(jnp.conj(psi_xn), jnp.conj(psi_yr),
                                        gemm=gemm, Gij=Gij, phases=phases, mesh=k_unfold_plan.mesh_xy,
                                        band_range=band_range,
                                        prepared_active_gemm=prepared_active_gemm,
                                        n_full=k_unfold_plan.n_full),
                lambda _: jnp.conj(G), operand=None)
    return ParentGreen(G, transposed)


def build_G(psi_xn, psi_yr, *, Gij=None, phases=None, layout='face',
           gemm=None, k_unfold_plan=None, right_k_unfold_plan=None, real_weights=None,
           band_range=None, prepared_active_gemm=None, conjugate=False):
    """Build parent operators and transport both typed endpoints without processor exchange.

    The Green is centroid-major ``(nk, mu, s, nu, s')`` on parents and on
    full k alike; see :func:`_build_G_face`.  ``conjugate=True`` returns
    ``conj(G)``, folded into the unfold's own pass when there is one.
    """
    if k_unfold_plan is None:
        if layout not in ('face', 'axis'):
            raise ValueError("build_G requires canonical faces with layout=face or axis.")
        if gemm is None:
            raise ValueError("build_G requires a GEMM plan or typed parent plan-provided GEMM callable.")
        G = _build_G_face(psi_xn, psi_yr, gemm=gemm, Gij=Gij, phases=phases,
                          band_range=band_range,
                          prepared_active_gemm=prepared_active_gemm)
        return jnp.conj(G) if conjugate else G
    pg = build_G_parents(psi_xn, psi_yr, Gij=Gij, phases=phases, layout=layout, gemm=gemm,
                         k_unfold_plan=k_unfold_plan, real_weights=real_weights,
                         band_range=band_range, prepared_active_gemm=prepared_active_gemm)
    return k_unfold_plan.unfold_operator(
        pg.G, operator_transpose=pg.partner(), right_plan=right_k_unfold_plan,
        conjugate=conjugate)


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


def build_G_tau(psi_xn, psi_yr, enk, t, *, e_ref=0.0, mask=None,
                band_weight=None, E_min=None, E_max=None,
                layout='face', gemm=None, k_unfold_plan=None, band_range=None,
                trim_zero_bands=False, prepared_active_gemm=None,
                conjugate=False, unfold=True, right_k_unfold_plan=None,
                real_weights=None, right=None):
    """Contract phases exp(-t*(energy-reference)) with energy windows, identity masks and signed weights.

    ``unfold=False`` returns the :class:`ParentGreen` pair instead of the
    full-k Green (the consumer does the typed unfold on its own load).
    ``right_k_unfold_plan`` transports the right endpoint of a two-family
    (charge x current) Green, as in :func:`build_G`.
    ``real_weights`` ``None`` derives from ``t`` whether the phases are real
    (a traced predicate for complex-typed ``t``); complex ``band_weight``
    always makes it ``False``. A Python bool is a static statement passed to
    :func:`build_G_parents`: ``False``
    builds an antiunitary partner as the conjugate-face GEMM with no device
    predicate, so a node loop does not stop on a host-read conditional; at a
    node whose phases are real that GEMM equals ``conj(G)``.
    ``right`` (``unfold=False``, a local GEMM): the right operand formed once
    from ``psi_yr`` (:func:`green_right_operand`); ``psi_yr`` is then not read.
    """
    if real_weights is None:
        real_weights = not jnp.issubdtype(jnp.result_type(t), jnp.complexfloating)
        if not real_weights:
            real_weights = jnp.imag(t) == 0
    # Complex band weights make the phases complex whatever the caller stated:
    # a static True is demoted, never trusted over the weights.
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
    if right is not None and unfold:
        raise ValueError("build_G_tau(right=...) returns the parent pair (unfold=False)")
    if not unfold:
        if conjugate:
            raise ValueError("build_G_tau(unfold=False) returns the parent pair; conjugate "
                             "applies to the unfolded Green only")
        return build_G_parents(
            psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm,
            k_unfold_plan=k_unfold_plan, real_weights=real_weights,
            band_range=band_range, prepared_active_gemm=prepared_active_gemm, right=right)
    return build_G(
        psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm,
        k_unfold_plan=k_unfold_plan, right_k_unfold_plan=right_k_unfold_plan,
        real_weights=real_weights,
        band_range=band_range, prepared_active_gemm=prepared_active_gemm,
        conjugate=conjugate)


# ---------------------------------------------------------------------------
# Output x blocks.  Σ = Σ_μ ψ*(μ) [G ⋆ W](μ, ν) ψ(ν) is linear in the output's
# μ rows, so a Σ consumer whose output tile would not fit stores and projects
# it in x blocks (mathdx mode 7's stored x block, every spin), each pass
# reading only its own pairs' Green and W; the Green itself stays whole at the
# parents.  ``d`` sizes the blocks: (ns/d)² of them, each ~(d/ns)² of the
# tile, laid on the face projector's slab pieces
# (``common.contract_bands.face_row_blocks``), so their projections move and
# multiply what the whole tile's does.  (Output spin blocks of the same size
# re-read the whole Green per block: the spin action mixes every source of a
# pair.)
# ---------------------------------------------------------------------------

def sigma_row_blocks(*, n_rmu, ns, d, mesh):
    """The x blocks ``(x0, bx, xs, xn)`` of a parent-row Σ tile stored at block size ``d`` (``None``: one whole tile at ``d = ns``)."""
    if int(d) == int(ns):
        return (None,)
    from common.contract_bands import face_row_blocks
    mx = int(n_rmu) // int(mesh.shape['x'])
    return face_row_blocks(mx, int(mesh.shape['y']), (int(ns) // int(d)) ** 2)


def _green_terms(*, n_parent, n_rmu, ns, n_band, mesh, n_right=None):
    """(T_p, M_axis): one parent Green tile ``16·n_parent·ns²·μ·ν/P``, and one Green build's
    two live SUMMA panels of both ψ orientations, at most ``N_b/p_x`` bands each,
    ``2·16·n_parent·ns·(N_b/p_x)·(μ/p_x + ν/p_y)`` (``face_green_product``), per rank.
    ``ν`` is ``n_right`` for a two-family Green, ``μ`` otherwise."""
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    nu = int(n_rmu) if n_right is None else int(n_right)
    tile = 16.0 * int(n_parent) * int(ns) ** 2 * (int(n_rmu) * nu) / (px * py)
    panels = (32.0 * int(n_parent) * int(ns) * (int(n_band) / px)
              * (int(n_rmu) / px + nu / py))
    return tile, panels


def sigma_spin_block(*, n_parent, n_rmu, ns, n_full, n_band, mesh, partner_tiles, plan=None):
    """The block size ``d`` (a divisor of ``ns``): a parent-row Σ convolution stores its output in (ns/d)² x blocks.

    New per rank beside what is live: the parent Green ``T_p``, ``partner_tiles`` more of
    it (1 when the antiunitary partner is its own GEMM, 0 when it is read as conj(G)),
    the widest stored x block ``T_p·xn·bx/μ_x`` (``(d/ns)²`` of it up to a row per slab
    piece, :func:`sigma_row_blocks`), the full-k W(τ) out of the k-convolution
    ``16·N_k·μ²/P``, and the panels of the Green and partner builds ``2·M_axis``.  The
    stored x block is the tiled part: the largest ``d`` whose block fits the fixed tile
    (``runtime.tiles.TILE_BYTES``) wins, else 1, so every process computes the same
    ``d`` from the shapes alone.  ``plan`` (a dict) receives ``d``, the pass's new
    bytes and the tile, for the compiled check of the window executable
    (``gw.mpa.sigma.SynthesisTau.admit``).
    """
    if int(ns) <= 1:
        return 1
    tile, panels = _green_terms(n_parent=n_parent, n_rmu=n_rmu, ns=ns, n_band=n_band,
                                mesh=mesh)
    w_tau = 16.0 * int(n_full) * int(n_rmu) ** 2 / (int(mesh.shape['x']) * int(mesh.shape['y']))
    from runtime.tiles import TILE_BYTES
    mx = int(n_rmu) // int(mesh.shape['x'])

    def frac(d):
        b = sigma_row_blocks(n_rmu=n_rmu, ns=ns, d=d, mesh=mesh)[0]
        return 1.0 if b is None else b[1] * b[3] / mx
    new = lambda d: ((1.0 + float(partner_tiles) + frac(d)) * tile + w_tau
                     + (1.0 + float(partner_tiles)) * panels)
    divisors = sorted((d for d in range(1, int(ns) + 1) if int(ns) % d == 0), reverse=True)
    d = next((d for d in divisors if frac(d) * tile <= TILE_BYTES), 1)
    from common.gpu_utils import record_stage_price
    record_stage_price(f"Sigma tau, sigma_spin_block d={d}/{int(ns)}", new(d),
                       section="sigma.tau_sweep")
    if plan is not None:
        plan.update(d=int(d), ns=int(ns), new=float(new(d)), tile=float(TILE_BYTES))
    return d


def chi0_door_scratch(*, kgrid, n_parent, n_rmu, ns, mesh, n_right=None):
    """Per-rank run-time scratch of one mathdx mode-11 call: its split arm's intermediate,
    the bound owned by ``ffi.fft.chi_unfold_scratch_bytes``;
    0 on the single pass and off the mathdx backend."""
    from ffi import fft as F
    if F.kconv_backend(mesh) != "mathdx":
        return 0
    tile, _ = _green_terms(n_parent=n_parent, n_rmu=n_rmu, ns=ns, n_band=0, mesh=mesh,
                           n_right=n_right)
    return F.chi_unfold_scratch_bytes(kgrid, ns, int(tile))


def price_chi0_node(*, n_parent, n_rmu, ns, n_full, n_out, n_band, mesh, partner, kgrid):
    """Record the price of one fused chi0 node (``w_isdf._get_chi_minimax_kernel_fused``).

    New per rank beside what is live: the valence and conduction parent Greens
    ``2·(1 + partner)·T_p`` (``partner`` when an antiunitary row reads a conjugate-face
    tile), the accumulator ``16·n_out·N_k·μ²/P``, the builds' panels
    ``2·(1 + partner)·M_axis`` (:func:`_green_terms`) and mode 11's run-time scratch
    (:func:`chi0_door_scratch`).  Nothing here is chunked: every term is a whole (μ, ν)
    tile, and ``distrib_la.panel_matmul`` bounds the panels itself.
    """
    tile, panels = _green_terms(n_parent=n_parent, n_rmu=n_rmu, ns=ns, n_band=n_band,
                                mesh=mesh)
    acc = 16.0 * int(n_out) * int(n_full) * int(n_rmu) ** 2 / (
        int(mesh.shape['x']) * int(mesh.shape['y']))
    from common.gpu_utils import record_stage_price
    new = (2.0 * (1.0 + float(bool(partner))) * (tile + panels) + acc
           + chi0_door_scratch(kgrid=kgrid, n_parent=n_parent, n_rmu=n_rmu, ns=ns, mesh=mesh))
    record_stage_price("chi0 node, price_chi0_node", new, section="chi.exec")
