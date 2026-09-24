"""Build parent Green operators and transport them with typed local symmetry actions."""
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
import jax.numpy as jnp

from common.contract_bands import merge_spin_centroid


def _face_band_gather_product(A, B, mesh, phases, band_range, n_full=None):
    """``A·diag(w)·B`` on band-distributed faces by gathered band panels.

    ``A`` ``(nq, M, N_b)`` and ``B`` ``(nq, N_b, N)`` are both
    ``P(None,'x','y')``.  ``w`` is the phase row, zero outside the per-row
    ``band_range``.  ``distrib_la.panel_matmul`` all-gathers the band panels
    (A over y, B over x) and multiplies them locally into the rank's own
    output tile, so no reduction follows.  The gathered panels are bounded
    by one full-k Green tile, ``16·N_k·(M/p_x)·(N/p_y)`` bytes (``N_k =
    n_full``, the parents' full zone; ``nq`` when the faces are already at
    full k), the unit the GW feasibility floor is counted in: one full band
    gather whenever that holds the whole band extent, streamed chunks
    otherwise.
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
        G_flat = _face_band_gather_product(
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


class ParentGreen(NamedTuple):
    """The raw-parent Green and what its typed unfold needs.

    ``G`` ``(n_parent, mu, s, nu, s')`` centroid-major; ``transpose`` the
    partner an antiunitary row reads (the conjugate-face Green, or ``conj(G)``
    for real weights), ``None`` when the plan has no antiunitary row.
    ``k_unfold_plan.unfold_operator(G, operator_transpose=transpose)`` is the
    full-k Green; ``ffi.fft.make_kconv_klead_unfold`` reads the pair directly.
    """
    G: jax.Array
    transpose: jax.Array | None


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
            transposed = jnp.conj(G)
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
        pg.G, operator_transpose=pg.transpose, right_plan=right_k_unfold_plan,
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
                conjugate=False, unfold=True):
    """Contract phases exp(-t*(energy-reference)) with energy windows, identity masks and signed weights.

    ``unfold=False`` returns the :class:`ParentGreen` pair instead of the
    full-k Green (the consumer does the typed unfold on its own load).
    """
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
        k_unfold_plan=k_unfold_plan, real_weights=real_weights,
        band_range=band_range, prepared_active_gemm=prepared_active_gemm,
        conjugate=conjugate)


# ---------------------------------------------------------------------------
# Spin-pair streaming (phase-1 item A4).  A contraction that is elementwise in
# the two spinor indices -- chi = sum_ab Gc_ab conj(Gv_ab), and
# Sigma_nm = sum_ab psi*_a [G_ab . W] psi_b with a spin-independent W -- never
# needs the ns^2 Green at once.  The typed unfold mixes spinor components, so
# the parents move to full k first (the psi action, whose Green equals the
# operator unfold of the parent Green) and each (a, b) block is one GEMM of
# the a and b spinor rows there.
# ---------------------------------------------------------------------------

def unfold_parent_faces(plan, psi_mun, psi_nmu, *, layout):
    """The raw-parent faces transported to full k by the plan's typed psi action.

    ``psi_mun`` ``(n_parent, s, mu, n)`` and ``psi_nmu`` ``(n_parent, n, s, mu)``
    at the ``layout`` specs in, the same orientations on ``plan.n_full`` rows
    out.  Collective-free: the packed centroid source map stays in its shard.
    """
    from common.shard_map import shard_map
    from common.wfn_layout import psi_specs

    nmu_spec, mun_spec = psi_specs(layout)
    return shard_map(
        lambda left, right: (
            plan.unfold_face(left, spin_axis=1, mu_axis=2, mesh_axis="x"),
            plan.unfold_face(right, spin_axis=2, mu_axis=3, mesh_axis="y")),
        mesh=plan.mesh_xy, in_specs=(mun_spec, nmu_spec),
        out_specs=(mun_spec, nmu_spec), check_vma=False)(psi_mun, psi_nmu)


def spin_pair_rows(psi_mun, psi_nmu, index, ns):
    """The spinor rows ``a, b = divmod(index, ns)`` of the two orientations."""
    return (jax.lax.dynamic_slice_in_dim(psi_mun, index // ns, 1, axis=1),
            jax.lax.dynamic_slice_in_dim(psi_nmu, index % ns, 1, axis=2))


def spin_pairs_needed(*, n_full, n_rmu, ns, mesh, live_green_tiles):
    """True when a whole-spin stage's live Greens exceed the device target.

    ``live_green_tiles`` counts the stage's concurrent ``G_tile =
    16·N_k·ns²·μ²/P``; the target is the agreed minimum device budget times
    the spinor's fragmentation utilization.  Every process must enter.
    """
    if int(ns) <= 1:
        return False
    from common.gpu_utils import (bfc_fragmentation_target_utilization,
                                  get_device_memory_gb,
                                  minimum_process_budget_gb)
    P_ = int(mesh.shape['x']) * int(mesh.shape['y'])
    g_tile = 16.0 * int(n_full) * int(ns) ** 2 * int(n_rmu) ** 2 / P_
    target = (minimum_process_budget_gb(get_device_memory_gb()) * 1e9
              * bfc_fragmentation_target_utilization(int(ns)))
    return float(live_green_tiles) * g_tile > target


def spin_block_sources(plan):
    """Host tables of the parent spin blocks behind each full-k spin block.

    The typed unfold gives ``G_ab(k) = Σ_cd U_k[a,c] conj(U_k[b,d]) T_k(G_cd)``
    with ``T_k`` the centroid transport of one parent block.  Returns
    ``(src_c, src_d, coef)``: for pair ``p = a·ns + b`` the ``(c, d)`` whose
    coefficient is nonzero at some full-k row (exact zeros only, as the
    incumbent rotation skips them), padded to a common count with zero
    coefficients, and ``coef[p, j, k] = U_k[a,c_j] conj(U_k[b,d_j])``.  A
    magnetic group whose spin actions are monomial needs two sources per
    block instead of ``ns²``.
    """
    U = np.asarray(plan.spin_action_full, dtype=np.complex128)
    n_full, ns = int(U.shape[0]), int(U.shape[-1])
    lists = []
    for a in range(ns):
        for b in range(ns):
            lists.append([(c, d) for c in range(ns) for d in range(ns)
                          if np.any(U[:, a, c] * np.conj(U[:, b, d]) != 0)])
    width = max(len(v) for v in lists)
    src_c = np.zeros((ns * ns, width), np.int32)
    src_d = np.zeros((ns * ns, width), np.int32)
    coef = np.zeros((ns * ns, width, n_full), np.complex128)
    for p, sources in enumerate(lists):
        a, b = divmod(p, ns)
        for j, (c, d) in enumerate(sources):
            src_c[p, j], src_d[p, j] = c, d
            coef[p, j] = U[:, a, c] * np.conj(U[:, b, d])
    return src_c, src_d, coef


def unfold_parent_spin_block(parent, plan, tables, index, *, conjugate=False):
    """One full-k spin block ``G_ab`` ``(n_full, mu, nu)`` of a parent Green.

    ``parent`` is the :class:`ParentGreen` pair, ``tables`` the
    :func:`spin_block_sources` triple, ``index = a·ns + b`` (traced).  Each
    source block is transported by the service's typed operator unfold with a
    unit spin action (its antiunitary rows read the partner's ``(d, c)``
    block), and the sources are summed with their coefficients.  No
    ``ns²`` full-k Green is formed.
    """
    from symmetry_maps import unfold_spin_centroid_operator

    src_c, src_d, coef = tables
    n_full = int(plan.n_full)
    unit = np.ones((n_full, 1, 1), dtype=np.complex128)
    trs = bool(np.any(np.asarray(plan.sym_idx) >= plan.n_sym_spatial))

    def block(op, c, d):
        op = jax.lax.dynamic_slice_in_dim(op, c, 1, axis=2)
        return jax.lax.dynamic_slice_in_dim(op, d, 1, axis=4)

    total = None
    for j in range(int(src_c.shape[1])):
        c, d = jnp.asarray(src_c)[index, j], jnp.asarray(src_d)[index, j]
        T = unfold_spin_centroid_operator(
            block(parent.G, c, d),
            operator_transpose=(block(parent.transpose, d, c)
                                if trs and parent.transpose is not None else None),
            conjugate=conjugate, irr_idx=plan.irr_idx, sym_idx=plan.sym_idx,
            sym_perm=plan.sym_perm, L_table=plan.L_table,
            k_irr_frac=plan.k_parent_frac, spin_action_full=unit,
            n_sym_spatial=plan.n_sym_spatial, mesh_xy=plan.mesh_xy,
            logical_centroid_extent=plan.n_centroid_packed, axis_local=True)
        w = jnp.asarray(coef)[index, j]
        w = jnp.conj(w) if conjugate else w
        term = w[:, None, None] * T.reshape(n_full, T.shape[1], T.shape[3])
        total = term if total is None else total + term
    return total
