"""Metric-aligned charge/current weights for ISDF centroid selection.

For the same left/right band windows used by candidate pruning, this module
builds the diagonal of the q=0 feature Gram and leaves the final square root
to the driver.  Band pairs are never materialised: each k point is first
contracted into two local spin-density matrices.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from common.bispinor_init import ALPHA_FS, NO_PAIR_DIRAC_CURRENT_MODEL
from common.gamma_matrices import gamma_apply, gamma_perm_phase


_GAMMA_MODES = ("charge", "transverse")
_PROJECTOR_BUDGET_BYTES = 8 * 1024 ** 3


@jax.jit
def _accumulate_subspace_density_matrices(
    density_left,
    density_right,
    psi_r,
    mask_left,
    mask_right,
):
    """Stream bands into ``D_L(r)`` and ``D_R(r)`` with one spin outer."""
    mask_left = jnp.asarray(mask_left, dtype=jnp.float64)
    mask_right = jnp.asarray(mask_right, dtype=jnp.float64)

    def add_state(carry, state_and_masks):
        D_left, D_right = carry
        psi_n, include_left, include_right = state_and_masks
        outer = psi_n[:, None, ...] * jnp.conj(psi_n[None, :, ...])
        return (
            D_left + include_left * outer,
            D_right + include_right * outer,
        ), None

    return jax.lax.scan(
        add_state,
        (density_left, density_right),
        (psi_r, mask_left, mask_right),
        unroll=1,
    )[0]


@jax.jit
def _charge_metric_diagonal(density_left, density_right, wavefunction_scale):
    """Return ``Tr(D_L D_R)`` in physical wavefunction normalisation."""
    out = jnp.real(jnp.sum(
        density_left * jnp.swapaxes(density_right, 0, 1), axis=(0, 1)))
    density_scale = jnp.asarray(wavefunction_scale, dtype=jnp.float64) ** 2
    return out * jnp.square(density_scale)


@jax.jit
def _transverse_metric_diagonal(
    density_left,
    density_right,
    wavefunction_scale,
):
    """Return ``sum_i Tr(D_L alpha_i D_R alpha_i) / alpha_fs^2``."""
    out = jnp.zeros(density_left.shape[-3:], dtype=jnp.float64)
    for mu in (1, 2, 3):
        perm, phase = gamma_perm_phase(mu)
        alpha_left = gamma_apply(density_left, perm, phase, axis=0)
        alpha_right = gamma_apply(density_right, perm, phase, axis=0)
        out = out + jnp.real(jnp.sum(
            alpha_left * jnp.swapaxes(alpha_right, 0, 1), axis=(0, 1)))
    current_scale = (
        jnp.asarray(wavefunction_scale, dtype=jnp.float64) ** 2 / ALPHA_FS)
    return out * jnp.square(current_scale)


@jax.jit
def _transverse_metric_diagonal_per_channel(
    density_left_by_channel,
    density_right_by_channel,
    wavefunction_scale,
):
    """``sum_i Tr(D_L^(i) alpha_i D_R^(i) alpha_i) / alpha_fs^2`` with channel
    i's projectors built from channel i's OWN carrier (the per-channel
    velocity balance; ``common.bispinor_init``).  With one shared carrier
    this is :func:`_transverse_metric_diagonal` term by term."""
    out = jnp.zeros(density_left_by_channel[0].shape[-3:], dtype=jnp.float64)
    for mu in (1, 2, 3):
        perm, phase = gamma_perm_phase(mu)
        alpha_left = gamma_apply(
            density_left_by_channel[mu - 1], perm, phase, axis=0)
        alpha_right = gamma_apply(
            density_right_by_channel[mu - 1], perm, phase, axis=0)
        out = out + jnp.real(jnp.sum(
            alpha_left * jnp.swapaxes(alpha_right, 0, 1), axis=(0, 1)))
    current_scale = (
        jnp.asarray(wavefunction_scale, dtype=jnp.float64) ** 2 / ALPHA_FS)
    return out * jnp.square(current_scale)


def feature_metric_diagonal_from_psi_r(
    psi_r,
    mask_left,
    mask_right,
    wavefunction_scale,
    *,
    gamma_mode: str,
):
    """Evaluate a charge or transverse Gram diagonal from one k carrier.

    Parameters
    ----------
    psi_r
        ``(nb, ns, nx, ny, nz)`` complex wavefunctions. ``ns=4`` is
        required for ``gamma_mode='transverse'``.
    mask_left, mask_right
        Unit/zero band-window masks of length ``nb``. They are fitting
        selectors, never occupations.
    wavefunction_scale
        ``sqrt(N_grid / Omega)`` for the unitary-FFT wavefunctions.
    gamma_mode
        ``'charge'`` for the identity vertex or ``'transverse'`` for the
        equally weighted three Dirac-current vertices.
    """
    mode = str(gamma_mode).strip().lower()
    if mode not in _GAMMA_MODES:
        raise ValueError(
            f"gamma_mode must be one of {_GAMMA_MODES}; got {gamma_mode!r}")
    if psi_r.ndim != 5:
        raise ValueError(
            "psi_r must have shape (nb,ns,nx,ny,nz); got "
            f"{psi_r.shape}")
    if mode == "transverse" and int(psi_r.shape[1]) != 4:
        raise ValueError(
            "gamma_mode='transverse' requires four-component bispinors; "
            f"got ns={int(psi_r.shape[1])}")
    left = jnp.asarray(mask_left, dtype=jnp.float64).reshape(-1)
    right = jnp.asarray(mask_right, dtype=jnp.float64).reshape(-1)
    if int(left.shape[0]) != int(psi_r.shape[0]) or left.shape != right.shape:
        raise ValueError(
            "left/right masks must have one entry per band; got "
            f"{left.shape}, {right.shape} for nb={int(psi_r.shape[0])}")
    ns = int(psi_r.shape[1])
    zero = jnp.zeros((ns, ns) + tuple(psi_r.shape[-3:]), dtype=psi_r.dtype)
    density_left, density_right = _accumulate_subspace_density_matrices(
        zero, jnp.zeros_like(zero), psi_r, left, right)
    if mode == "charge":
        return _charge_metric_diagonal(
            density_left, density_right, wavefunction_scale)
    return _transverse_metric_diagonal(
        density_left, density_right, wavefunction_scale)


@jax.jit
def _accumulate_grid_pullback(accumulator, field, pullback, member_weight):
    """Add one service-owned scalar pullback without a host field copy."""
    return accumulator + member_weight * field.reshape(-1)[pullback]


def _validated_range(band_range, nbands: int, name: str) -> tuple[int, int]:
    if len(band_range) != 2:
        raise ValueError(f"{name} must be a 0-based half-open pair")
    lo, hi = (int(v) for v in band_range)
    if lo < 0 or hi <= lo or hi > int(nbands):
        raise ValueError(
            f"{name} must be a nonempty subset of [0,{int(nbands)}); "
            f"got {(lo, hi)}")
    return lo, hi


def _window_mask(lo: int, canonical_mask, band_range) -> np.ndarray:
    canonical = np.asarray(canonical_mask, dtype=np.float64)
    bands = int(lo) + np.arange(canonical.size)
    range_lo, range_hi = band_range
    return canonical * ((bands >= range_lo) & (bands < range_hi))


def _quadrature_tables(wfn, sym):
    """Return authenticated star rows, their weights and the full-BZ quadrature.

    Two WFN k storages reach this function and they normalise DIFFERENTLY.
    The distinguishing fact is whether the raw WFN k axis is the irreducible
    wedge or the whole grid, i.e. ``wfn.nkpts`` against ``sym.nk_tot``:

    * **IBZ storage** (``wfn.nkpts < sym.nk_tot``): one stored row per star,
      and ``kweights`` already carries the WHOLE star's weight.  The stored
      parents must therefore cover the entire normalised weight, and each
      full-BZ member of a star takes ``w_parent / n_members``.
    * **Full-BZ storage** (``wfn.nkpts == sym.nk_tot``): every point of the
      grid is stored and ``kweights`` IS the full-BZ quadrature already.
      Each member keeps its own stored weight; the parents cover only
      ``n_parents / nk_tot`` of it by construction, so the IBZ precondition
      is not merely unnecessary here, it is false for every unfolded NSCF
      grid.  Requiring it refused every such WFN, including the repo's own
      ``tests/regression/bispinor_debug/WFN.h5`` (KNOWN_LORRAX_ISSUES.md,
      2026-09-01).

    The two branches coincide wherever they overlap: a full-BZ file whose
    stars are all singletons hits ``w_parent / 1 == w_member``, so the
    discriminant never changes an answer it did not have to change.

    Returns ``(parents_used, star_plan, full_weights)``.  ``star_plan[parent]``
    is ``(sym_rows, member_weights)``, aligned row by row, and is the ONLY
    place member weights are formed — the metric builder consumes it rather
    than re-deriving the same normalisation.  ``full_weights`` is the
    normalised full-BZ quadrature indexed by full-BZ k.
    """
    from symmetry_maps import star_tables_of

    parent_for_k, sym_row_for_k, _ = star_tables_of(sym)
    nk_full = int(sym.nk_tot)
    nk_raw = int(wfn.nkpts)
    if parent_for_k.shape != (nk_full,) or sym_row_for_k.shape != (nk_full,):
        raise ValueError(
            "SymMaps full-k tables disagree with nk_tot: "
            f"irr_idx_k={parent_for_k.shape}, sym_idx_k={sym_row_for_k.shape}, "
            f"nk_tot={nk_full}")
    if np.any(parent_for_k < 0) or np.any(parent_for_k >= nk_raw):
        raise ValueError(
            "SymMaps.irr_idx_k contains a row outside the raw WFN k axis "
            f"[0,{nk_raw})")
    parents_used = np.unique(parent_for_k)
    if parents_used.size == 0:
        raise ValueError("SymMaps contains no full-BZ parent rows")

    kweights = np.asarray(wfn.kweights, dtype=np.float64)
    if kweights.shape != (nk_raw,):
        raise ValueError(
            "WfnLoader.kweights must have one entry per raw WFN k row; "
            f"got {kweights.shape}, expected {(nk_raw,)}")
    weight_sum = float(kweights.sum())
    if (not np.all(np.isfinite(kweights)) or np.any(kweights < 0.0)
            or not np.isfinite(weight_sum) or weight_sum <= 0.0):
        raise ValueError(
            "WfnLoader.kweights must be finite, nonnegative, and have "
            f"positive sum; got sum={weight_sum}")
    kweights = kweights / weight_sum

    full_bz_storage = nk_raw == nk_full
    full_weights = np.empty(nk_full, dtype=np.float64)
    if full_bz_storage:
        # The raw axis IS the full BZ, so a parent must be its own star
        # representative.  If it is not, the two index spaces are not the
        # same axis and no weight assignment here would mean anything.
        fixed_points = parent_for_k[parents_used]
        if not np.array_equal(fixed_points, parents_used):
            raise ValueError(
                "full-BZ WFN storage (nkpts == nk_tot == "
                f"{nk_full}) requires SymMaps.irr_idx_k to map every star "
                "parent to itself, so that the full-BZ and raw WFN k axes "
                f"are the same axis; got irr_idx_k[{parents_used.tolist()}] "
                f"= {fixed_points.tolist()}")
        full_weights[:] = kweights
    else:
        used_weight = float(kweights[parents_used].sum())
        if not np.isclose(used_weight, 1.0, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError(
                "SymMaps full-BZ parents omit nonzero WFN quadrature weight: "
                f"selected normalized weight={used_weight:.17g}, want 1. "
                f"The raw WFN k axis ({nk_raw} rows) is read as the IBZ "
                f"because it is shorter than nk_tot={nk_full}; an IBZ file "
                "must store one row per star and carry the whole star weight "
                "on it")
        for parent in parents_used:
            member_rows = np.flatnonzero(parent_for_k == parent)
            full_weights[member_rows] = (
                float(kweights[parent]) / float(member_rows.size))

    if not np.isclose(full_weights.sum(), 1.0, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError(
            "expanded full-BZ quadrature weights do not sum to one: "
            f"sum={full_weights.sum():.17g}")

    star_plan = {}
    for parent in parents_used:
        member_rows = np.flatnonzero(parent_for_k == parent)
        star_plan[int(parent)] = (
            np.asarray(sym_row_for_k[member_rows], dtype=np.int32),
            np.asarray(full_weights[member_rows], dtype=np.float64))
    return parents_used, star_plan, full_weights


def full_k_quadrature_weights(wfn, sym) -> np.ndarray:
    """Normalised full-BZ quadrature weight of every unfolded k point.

    IBZ storage spreads each parent weight uniformly over its star; full-BZ
    storage passes the stored weights through.  See :func:`_quadrature_tables`.
    """
    return _quadrature_tables(wfn, sym)[2].copy()


def _projector_memory_plan(
    n_grid: int,
    ns: int,
    *,
    device_memory_bytes: int | None,
    n_carriers: int = 1,
) -> tuple[int, int]:
    """Return projector bytes/cap, refusing before an unbounded HBM attempt.

    ``n_carriers`` distinct four-spinor carriers hold one (D_L, D_R) pair
    each (three under the per-channel velocity balance)."""
    if min(n_grid, ns, n_carriers) <= 0:
        raise ValueError("metric memory plan requires positive grid/spin sizes")
    budget = _PROJECTOR_BUDGET_BYTES
    if device_memory_bytes is not None and int(device_memory_bytes) > 0:
        budget = min(budget, max(1024 ** 3, int(device_memory_bytes) // 4))
    bytes_per_point = (2 * int(n_carriers) * int(ns) ** 2
                       * np.dtype(np.complex128).itemsize)
    projector_bytes = bytes_per_point * int(n_grid)
    if projector_bytes > budget:
        raise MemoryError(
            "centroid feature metric projector carrier exceeds its preflight "
            f"cap: two {ns}x{ns} complex128 grids need "
            f"{projector_bytes / 2**30:.2f} GiB/rank, cap is "
            f"{budget / 2**30:.2f} GiB/rank. This path distributes IBZ "
            "parents but does not spatially shard one projector; reduce the "
            "FFT grid or use a future distributed-r-space metric backend.")
    return projector_bytes, budget


def build_feature_metric_diagonal(
    wfn,
    sym,
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    *,
    gamma_mode: str,
    dist_mesh=None,
    verbose: bool = True,
    bispinor_lifts: tuple[str, str, str] | None = None,
):
    """Build the q=0 feature-Gram diagonal on the WFN FFT grid.

    ``bispinor_lifts`` (transverse mode only) names the four-spinor carrier
    of each Lorentz label, ``resolve_four_current_representation(...).
    current_lift_for(a)``: ``None`` or three ``"raw"`` is the shipped
    sigma.p carrier shared by the three channels (one load per band window,
    byte-identical to the historical weight); distinct lifts load the
    two-spinor window once and lift it once per carrier through
    ``WfnLoader.lift``, and channel i's ``Tr(D_L alpha_i D_R alpha_i)`` is
    built from channel i's own projectors.

    The returned field is

    ``s(r)=sum_k w_k sum_{m in L,n in R}|Psi_m^dag Gamma Psi_n|^2``.

    ``Gamma=I`` for charge. For transverse current the three
    ``Gamma=alpha_i/alpha_fs`` channels are summed. The driver uses
    ``sqrt(s)`` as the Lloyd mass. In the one-k, one-component,
    equal-window limit this is exactly the historical band density; over
    multiple k points it is the norm of the k-stacked feature row.

    Bands stream into ``D_L`` and ``D_R`` together; neither an ``(n,m)``
    transition carrier nor an ``O(Nsym*Ngrid)`` pullback cache is formed.
    Parents are partitioned over processes and one final scalar-grid psum
    combines the result.  The two full-grid projector carriers are explicitly
    priced and capped at 8 GiB/rank (or one quarter of device memory); this
    parent-distributed path refuses rather than pretending those grids are
    spatially sharded.
    """
    from common.collectives import (process_rank_world, psum_replicate,
                                    single_device_mesh)
    from common.wfn_transforms import to_rbox
    from wfn_loader import IBZRows, WfnLoader, uniform_band_windows

    if not isinstance(wfn, WfnLoader):
        raise TypeError(
            "build_feature_metric_diagonal requires the driver's open "
            f"WfnLoader; got {type(wfn).__name__}")
    mode = str(gamma_mode).strip().lower()
    if mode not in _GAMMA_MODES:
        raise ValueError(
            f"gamma_mode must be one of {_GAMMA_MODES}; got {gamma_mode!r}")
    if mode == "transverse" and int(wfn.nspinor) != 2:
        raise ValueError(
            "gamma_mode='transverse' requires a two-component Pauli WFN; "
            f"got nspinor={int(wfn.nspinor)}")
    left_range = _validated_range(
        band_range_left, int(wfn.nbands), "band_range_left")
    right_range = _validated_range(
        band_range_right, int(wfn.nbands), "band_range_right")

    fft_grid = tuple(int(v) for v in wfn.fft_grid)
    n_grid = int(np.prod(fft_grid))
    cell_volume = float(wfn.cell_volume)
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError(
            f"WFN cell volume must be finite and positive, got {cell_volume}")
    wavefunction_scale = float(np.sqrt(n_grid / cell_volume))
    ns = 4 if mode == "transverse" else int(wfn.nspinor)
    lifts = ("raw", "raw", "raw")
    if bispinor_lifts is not None:
        if mode != "transverse":
            raise ValueError(
                "bispinor_lifts is only meaningful for gamma_mode='transverse'")
        lifts = tuple(str(l).strip().lower() for l in bispinor_lifts)
        if len(lifts) != 3:
            raise ValueError(
                f"bispinor_lifts needs one selector per Lorentz label; got "
                f"{bispinor_lifts!r}")
    distinct_lifts = tuple(dict.fromkeys(lifts))
    per_channel = mode == "transverse" and len(distinct_lifts) > 1
    if not per_channel and distinct_lifts != ("raw",):
        # One non-sigma.p carrier for all three channels is the retired
        # single sigma.v carrier (its alpha^a vertex carries the
        # (i/2) eps_abc [sigma^c, d_b V_SO] artifact); refuse it by name.
        raise ValueError(
            "bispinor_lifts must be three 'raw' (shared sigma.p carrier) or "
            "one distinct carrier per Lorentz label; got "
            f"{bispinor_lifts!r}")

    parents_used, star_plan, _ = _quadrature_tables(wfn, sym)

    rank, world = process_rank_world()
    if world > 1 and dist_mesh is None:
        raise ValueError(
            "build_feature_metric_diagonal requires dist_mesh at P>1 so the "
            f"wavefunction sweep is partitioned rather than repeated on {world} "
            "processes")

    union_lo = min(left_range[0], right_range[0])
    union_hi = max(left_range[1], right_range[1])
    bytes_per_band = 3 * ns * n_grid * np.dtype(np.complex128).itemsize
    chunk = max(1, min(
        union_hi - union_lo, (4 * 1024 ** 3) // max(1, bytes_per_band)))
    windows = uniform_band_windows(union_lo, union_hi, chunk)
    width = int(windows[0][1].shape[0])
    try:
        from common.gpu_utils import get_device_memory_gb
        device_memory_bytes = int(float(get_device_memory_gb()) * 1024 ** 3)
    except Exception:
        device_memory_bytes = None
    projector_bytes, projector_cap = _projector_memory_plan(
        n_grid, ns, device_memory_bytes=device_memory_bytes,
        n_carriers=len(distinct_lifts))
    n_rounds = -(-int(parents_used.size) // world)
    parents_local = [
        (int(parent), True) for parent in parents_used[rank::world]]
    while len(parents_local) < n_rounds:
        parents_local.append((int(parents_used[0]), False))
    if rank == 0:
        print(
            f"  {mode} metric plan: {len(parents_used)} parent(s) over "
            f"{world} rank(s); {2 * len(distinct_lifts)} full-grid projectors "
            f"{projector_bytes / 2**30:.2f} GiB/rank "
            f"(cap {projector_cap / 2**30:.2f}), WFN chunk <= 4.00 GiB, "
            "one scalar-grid psum"
            + ("" if not per_channel else
               f"; per-channel current carriers {lifts}"),
            flush=True)

    t0 = time.perf_counter()
    last_log = t0
    mesh = single_device_mesh()
    metric_local = jnp.zeros(n_grid, dtype=jnp.float64)
    real_parents_done = 0
    pullback_rows_done = 0
    for parent, include_parent in parents_local:
        k_spec = IBZRows((parent,))
        box_index = wfn.box_index(k=k_spec)
        zero = jnp.zeros((ns, ns) + fft_grid, dtype=jnp.complex128)
        # One (D_L, D_R) pair per DISTINCT carrier.  The shared-carrier case
        # is the historical single pair; the per-channel case lifts one
        # two-spinor window once per carrier.
        dens = {lift: (zero, jnp.zeros_like(zero)) for lift in distinct_lifts}
        for lo, canonical_mask in windows:
            left_mask = _window_mask(lo, canonical_mask, left_range)
            right_mask = _window_mask(lo, canonical_mask, right_range)
            if not include_parent:
                left_mask = np.zeros_like(left_mask)
                right_mask = np.zeros_like(right_mask)
            if per_channel:
                psi_2 = wfn.load_process_local(
                    bands=(lo, lo + width), k=k_spec)
                lifted = {
                    lift: wfn.lift(psi_2, k=k_spec, bispinor_lift=lift)
                    for lift in distinct_lifts}
                del psi_2
            else:
                lifted = {"raw": wfn.load_process_local(
                    bands=(lo, lo + width), k=k_spec,
                    bispinor=(mode == "transverse"))}
            for lift, psi_g in lifted.items():
                psi_r = to_rbox(
                    psi_g, box_index, fft_grid, mesh=mesh, norm="ortho")
                if int(psi_r.shape[0]) != 1 or int(psi_r.shape[2]) != ns:
                    raise ValueError(
                        "one-parent WfnLoader request returned incompatible "
                        f"shape {psi_r.shape}; expected k=1, ns={ns}")
                d_l, d_r = dens[lift]
                d_l, d_r = _accumulate_subspace_density_matrices(
                    d_l, d_r, psi_r[0],
                    jnp.asarray(left_mask), jnp.asarray(right_mask))
                d_l.block_until_ready()
                d_r.block_until_ready()
                dens[lift] = (d_l, d_r)
                del psi_r
            del lifted
        density_left, density_right = dens[distinct_lifts[0]]

        if mode == "charge":
            parent_metric = _charge_metric_diagonal(
                density_left, density_right, wavefunction_scale)
        elif per_channel:
            parent_metric = _transverse_metric_diagonal_per_channel(
                tuple(dens[lift][0] for lift in lifts),
                tuple(dens[lift][1] for lift in lifts),
                wavefunction_scale)
        else:
            parent_metric = _transverse_metric_diagonal(
                density_left, density_right, wavefunction_scale)
        if include_parent:
            # Weights come from _quadrature_tables, which is the one place
            # that knows whether this WFN stores the IBZ or the full BZ.
            star_sym_rows, star_weights = star_plan[parent]
            for row, member_weight in zip(star_sym_rows, star_weights):
                pullback = sym.fft_grid_pullback(
                    np.asarray([int(row)], dtype=np.int32),
                    fft_grid, validate=True)
                if pullback.shape != (1, n_grid):
                    raise ValueError(
                        "symmetry-service FFT-grid pullback has the wrong "
                        f"shape: {pullback.shape} != {(1, n_grid)}")
                pullback_dev = jnp.asarray(pullback[0], dtype=jnp.int32)
                metric_local = _accumulate_grid_pullback(
                    metric_local, parent_metric, pullback_dev,
                    jnp.asarray(float(member_weight), dtype=jnp.float64))
                metric_local.block_until_ready()
                pullback_rows_done += 1
                del pullback, pullback_dev
            real_parents_done += 1
        else:
            parent_metric.block_until_ready()
        del density_left, density_right, parent_metric, dens
        if verbose and time.perf_counter() - last_log > 5.0:
            last_log = time.perf_counter()
            print(
                f"    [{mode} metric] rank {rank}: "
                f"{real_parents_done}/{len(parents_local)} local parents "
                f"after {last_log - t0:.1f}s", flush=True)

    metric = np.asarray(metric_local, dtype=np.float64)
    if world > 1:
        metric = psum_replicate(metric, dist_mesh)
    metric = np.asarray(metric, dtype=np.float64).reshape(fft_grid)
    scale = float(np.max(np.abs(metric)))
    negative_tolerance = 256.0 * np.finfo(np.float64).eps * scale
    minimum = float(np.min(metric))
    if minimum < -negative_tolerance:
        raise FloatingPointError(
            f"{mode} feature-Gram diagonal is not PSD: min={minimum:.6e}, "
            f"relative={minimum / (scale or 1.0):.6e}")
    metric = np.maximum(metric, 0.0)
    if verbose:
        model = ("identity charge vertex" if mode == "charge" else
                 NO_PAIR_DIRAC_CURRENT_MODEL
                 + ("" if not per_channel else
                    f" with per-channel carriers {lifts}"))
        print(
            f"  {mode} feature-Gram diagonal ({model}, left={left_range}, "
            f"right={right_range}, unit band weights, {len(parents_used)} "
            f"parents/{world} ranks, one psum) built with "
            f"{pullback_rows_done} streamed local pullback row(s) in "
            f"{time.perf_counter() - t0:.2f}s", flush=True)
    return metric


__all__ = [
    "build_feature_metric_diagonal",
    "feature_metric_diagonal_from_psi_r",
    "full_k_quadrature_weights",
]
