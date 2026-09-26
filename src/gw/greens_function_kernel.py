"""Build parent Green operators and transport them with typed local symmetry actions."""
import dataclasses
from functools import partial

import jax

import numpy as np
import jax.numpy as jnp

from common.contract_bands import merge_spin_centroid


def face_band_gather_product(A, B, mesh, phases, band_range, n_full=None):
    """``A·diag(w)·B`` on band-distributed faces by gathered band panels.

    ``A`` ``(nq, M, N_b)`` and ``B`` ``(nq, N_b, N)`` are both
    ``P(None,'x','y')``.  ``w`` is the phase row, zero outside the per-row
    ``band_range``.  ``distrib_la.panel_matmul`` all-gathers the band panels
    (A over y, B over x) and multiplies them locally into the rank's own
    output tile, so no reduction follows.  Every Green-building stage
    reserves one full-k Green tile, ``16·N_k·(M/p_x)·(N/p_y)`` bytes
    (``N_k = n_full``, the parents' full zone; ``nq`` when the faces are
    already at full k), for these transient panels, so the panel count is
    derived from that reservation: one panel, the complete band extent
    (every band on every rank, for this call only), when it fits; otherwise
    interleaved band chunks, two live at a time (one prefetched), sized so
    that pair fits the tile.  Nothing band-complete outlives the call.
    """
    from distrib_la import panel_matmul

    nq, m, nb = (int(v) for v in A.shape)
    n = int(B.shape[-1])
    weight = None if phases is None else phases.astype(A.dtype)
    if band_range is not None:
        lo, hi = (jnp.reshape(jnp.asarray(v), (-1, 1)) for v in band_range)
        idx = jnp.arange(nb)[None, :]
        live = (idx >= lo) & (idx < hi)
        weight = (live.astype(A.dtype) if weight is None
                  else jnp.where(live, weight, jnp.zeros((), A.dtype)))
    if weight is not None:
        A = A * weight[:, None, :]
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    tile_bytes = A.dtype.itemsize * int(n_full or nq) * (m // px) * (n // py)
    return panel_matmul(A, B, mesh=mesh, panel_bytes=tile_bytes)


def _build_G_face(psi_mun, psi_nmu, *, gemm, Gij=None, phases=None, mesh=None,
                  band_range=None, prepared_active_gemm=None, n_full=None):
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
    elif getattr(gemm, "backend", "local") != "local":
        # A already carries the phases when there is no band range (above).
        G_flat = face_band_gather_product(
            A, B, gemm.mesh, None if band_range is None else phases, band_range,
            n_full=n_full)
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


def build_G_parents(psi_xn, psi_yr, *, Gij=None, phases=None, layout='face', gemm=None,
                    k_unfold_plan, real_weights=None, band_range=None,
                    prepared_active_gemm=None) -> ParentGreen:
    """The parent Green and its antiunitary partner, before the typed unfold (see :class:`ParentGreen`)."""
    if layout not in ('face', 'axis'):
        raise ValueError("build_G requires canonical faces with layout=face or axis.")
    if gemm is None:
        raise ValueError("build_G requires a GEMM plan or typed parent plan-provided GEMM callable.")
    G = _build_G_face(psi_xn, psi_yr, gemm=gemm, Gij=Gij, phases=phases,
                      mesh=k_unfold_plan.mesh_xy,
                      band_range=band_range,
                      prepared_active_gemm=prepared_active_gemm,
                      n_full=k_unfold_plan.n_full)
    transposed = None
    if np.any(np.asarray(k_unfold_plan.sym_idx) >= k_unfold_plan.n_sym_spatial):
        if (real_weights is True or phases is None
                or not jnp.issubdtype(phases.dtype, jnp.complexfloating)):
            return ParentGreen(G, None, conj_partner=True)

        def conjugate_build(_=None):
            return _build_G_face(jnp.conj(psi_xn), jnp.conj(psi_yr),
                                 gemm=gemm, Gij=Gij, phases=phases, mesh=k_unfold_plan.mesh_xy,
                                 band_range=band_range,
                                 prepared_active_gemm=prepared_active_gemm,
                                 n_full=k_unfold_plan.n_full)
        if real_weights is False:
            # Stated complex on the host: no runtime predicate.
            transposed = conjugate_build()
        else:
            # A traced predicate is a conditional the GPU runtime reads back
            # to the host (one synchronization per call).
            transposed = jax.lax.cond(
                (jnp.any(jnp.imag(phases) != 0) if real_weights is None
                 else ~jnp.asarray(real_weights)),
                conjugate_build, lambda _: jnp.conj(G), operand=None)
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
                real_phases=None):
    """Contract phases exp(-t*(energy-reference)) with energy windows, identity masks and signed weights.

    ``unfold=False`` returns the :class:`ParentGreen` pair instead of the
    full-k Green (the consumer does the typed unfold on its own load).
    ``right_k_unfold_plan`` transports the right endpoint of a two-family
    (charge x current) Green, as in :func:`build_G`.
    ``real_phases`` is the caller's host-side statement that ``Im t == 0``
    (a Σ Laplace window: ``t = τ``) or not (a crossing window); stated, the
    antiunitary partner is chosen at trace time (``conj(G)`` read on load,
    or the conjugate build).  ``None`` derives it from ``t`` on device, a
    conditional the GPU runtime reads back to the host at every call.
    Complex band weights make the phases complex whatever is stated.
    """
    real_weights = not jnp.issubdtype(jnp.result_type(t), jnp.complexfloating)
    if not real_weights:
        real_weights = (jnp.imag(t) == 0 if real_phases is None
                        else bool(real_phases))
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
    if not unfold:
        if conjugate:
            raise ValueError("build_G_tau(unfold=False) returns the parent pair; conjugate "
                             "applies to the unfolded Green only")
        return build_G_parents(
            psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm,
            k_unfold_plan=k_unfold_plan, real_weights=real_weights,
            band_range=band_range, prepared_active_gemm=prepared_active_gemm)
    return build_G(
        psi_xn, psi_yr, phases=phases, layout=layout, gemm=gemm,
        k_unfold_plan=k_unfold_plan, right_k_unfold_plan=right_k_unfold_plan,
        real_weights=real_weights,
        band_range=band_range, prepared_active_gemm=prepared_active_gemm,
        conjugate=conjugate)


# ---------------------------------------------------------------------------
# Output spin blocks.  Σ = Σ_ab ψ*_a [G ⋆ W]_ab ψ_b is linear in the output
# spin block, so a Σ consumer whose output tile would not fit stores and
# projects it in d x d blocks (mathdx mode 7's output spin block); the Green
# itself stays whole at the parents.
# ---------------------------------------------------------------------------

def sigma_spin_block(*, n_parent, n_rmu, ns, mesh, partner_tiles):
    """The output spin block ``d`` (a divisor of ``ns``) a parent-row Σ convolution stores per pass.

    Live per rank: the parent Green ``T_p = 16·n_parent·ns²·μ²/P``, ``partner_tiles`` more
    of it (1 when the antiunitary partner is its own GEMM, 0 when it is read as conj(G)),
    and the stored block ``T_p·(d/ns)²``; the largest ``d`` whose set fits the agreed
    device target (the minimum process budget times the spinor's fragmentation
    utilization) wins, else 1.  Every process computes the same ``d``.
    """
    if int(ns) <= 1:
        return 1
    P_ = int(mesh.shape['x']) * int(mesh.shape['y'])
    tile = 16.0 * int(n_parent) * int(ns) ** 2 * int(n_rmu) ** 2 / P_
    target = _device_target_bytes(ns)
    price = lambda d: (1.0 + float(partner_tiles) + (d / int(ns)) ** 2) * tile
    divisors = sorted((d for d in range(1, int(ns) + 1) if int(ns) % d == 0), reverse=True)
    d = next((d for d in divisors if price(d) <= target), 1)
    from common.gpu_utils import record_stage_price
    record_stage_price("Sigma tau, sigma_spin_block", price(d), section="sigma.tau_sweep")
    return d


def _device_target_bytes(ns: int) -> float:
    """The agreed per-process device target: the minimum process budget times the spinor's
    fragmentation utilization (the one target the Green-side planners share)."""
    from common.gpu_utils import (bfc_fragmentation_target_utilization,
                                  get_device_memory_gb,
                                  minimum_process_budget_gb)
    return (minimum_process_budget_gb(get_device_memory_gb()) * 1e9
            * bfc_fragmentation_target_utilization(int(ns)))


def chi_valence_chunks(*, n_parent, n_rmu, ns, n_full, n_out, n_val, mesh, partner):
    """Valence-band passes of the fused chi0 node (``w_isdf._get_chi_minimax_kernel_fused``).

    chi_tau = sum_ab conj(Gc'_ab) Gv'_ab is linear in Gv, so the valence Green may be
    built and accumulated in band chunks against one conduction Green.  Live per rank:
    ``(1 + partner)·T_p`` for Gc, the same over ``n`` for a Gv chunk, and the accumulator
    ``16·n_out·N_k·μ²/P``, with ``T_p = 16·n_parent·ns²·μ²/P``; the smallest ``n`` that fits
    the device target wins (every process computes the same ``n``), capped at ``n_val``.
    """
    P_ = int(mesh.shape['x']) * int(mesh.shape['y'])
    tile = 16.0 * int(n_parent) * int(ns) ** 2 * int(n_rmu) ** 2 / P_
    acc = 16.0 * int(n_out) * int(n_full) * int(n_rmu) ** 2 / P_
    target = _device_target_bytes(ns)
    side = (1.0 + float(bool(partner))) * tile
    price = lambda n: side * (1.0 + 1.0 / n) + acc
    n = next((n for n in range(1, max(1, int(n_val)) + 1) if price(n) <= target),
             max(1, int(n_val)))
    from common.gpu_utils import record_stage_price
    record_stage_price("chi0, chi_valence_chunks", price(n), section="chi.exec")
    return n

