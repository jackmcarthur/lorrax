"""Metric-aligned charge/current weights for ISDF centroid selection.

For the same left/right band windows used by candidate pruning, this module
builds the diagonal of the q=0 feature Gram and leaves the final square root
to the driver.  Band pairs are never materialised: each k point is contracted
into one band-summed spinor density matrix per window by the density scan of
:func:`gw.qsgw_density.rho_from_wfns`, with unit weights over the window.
"""

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from common.bispinor_init import ALPHA_FS, NO_PAIR_DIRAC_CURRENT_MODEL
from common.gamma_matrices import gamma_apply, gamma_perm_phase


_GAMMA_MODES = ("charge", "transverse")


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


@partial(jax.jit, static_argnames=("mode",))
def _accumulate_metric_rows(row_fields, density_left, density_right,
                            row_weights, *, mode):
    """``A_g += sum_k W[k, g] Tr(D_Lk Gamma D_Rk Gamma)`` for one k chunk.

    ``density_*`` are ``(K, ns, ns, nx, ny, nz)`` per-k spinor density
    matrices already in physical normalisation (scale 1 below);
    ``row_weights`` is ``(K, n_rows)``, the full-BZ member weight of each
    chunk parent under each distinct symmetry row; ``row_fields`` is
    ``(n_rows, nx*ny*nz)``.
    """
    kernel = (_charge_metric_diagonal if mode == "charge"
              else _transverse_metric_diagonal)
    fields = jax.vmap(lambda left, right: kernel(left, right, 1.0))(
        density_left, density_right)
    return row_fields + jnp.einsum(
        "kg,kr->gr", row_weights,
        fields.reshape(fields.shape[0], -1), optimize=True)


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


def _metric_chunk_plan(
    *,
    n_parents: int,
    n_bands: int,
    n_band_shards: int,
    ns: int,
    ngkmax: int,
    n_grid: int,
    n_windows: int,
    device_memory_bytes: int | None,
) -> tuple[int, int, int]:
    """Return ``(k_chunk, band_chunk, budget_bytes)`` for the metric scan.

    One quarter of the device is the budget, half for the scan's FFT box
    transient and half for the resident k chunk.  The density scan transforms
    every local band of one k at once, ``2 * nb/P * ns * N_grid`` complex
    values, so the band chunk bounds that transient.  A k chunk holds its
    ``psi(G)`` band shard plus the replicated per-k density matrices and their
    psum buffer.  The k chunk divides the parent count, so one executable
    serves every chunk.  A single k whose replicated matrices exceed the
    budget refuses: this route does not spatially shard ``D_k(r)``.
    """
    from runtime.padding import bounded_partition_tile

    c128 = np.dtype(np.complex128).itemsize
    budget = int(device_memory_bytes or 0) // 4
    if budget <= 0:
        budget = 2 * 1024 ** 3
    per_band = 2 * int(ns) * int(n_grid) * c128
    local_bands = max(1, (budget // 2) // per_band)
    band_chunk = int(min(int(n_bands), local_bands * int(n_band_shards)))
    local_chunk = -(-band_chunk // int(n_band_shards))
    per_k = (local_chunk * int(ns) * int(ngkmax) * c128
             + 2 * int(n_windows) * int(ns) ** 2 * int(n_grid) * c128)
    if per_k > budget // 2:
        raise MemoryError(
            "centroid feature metric: one k point's replicated "
            f"{ns}x{ns} density matrices need {per_k / 2**30:.2f} GiB/rank, "
            f"above the {budget / 2 / 2**30:.2f} GiB/rank chunk budget. "
            "This route band-shards psi but does not spatially shard D_k(r); "
            "reduce the FFT grid or add ranks.")
    k_chunk = bounded_partition_tile(
        int(n_parents), max(1, (budget // 2) // per_k), 1)
    return max(1, k_chunk), band_chunk, budget


def build_feature_metric_diagonal(
    wfn,
    sym,
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    *,
    gamma_mode: str,
    dist_mesh=None,
    verbose: bool = True,
):
    """Build the q=0 feature-Gram diagonal on the WFN FFT grid.

    The returned field is

    ``s(r)=sum_k w_k sum_{m in L,n in R}|Psi_m^dag Gamma Psi_n|^2
          =sum_k w_k Tr(D_Lk(r) Gamma D_Rk(r) Gamma)``,

    with ``D_Wk(r) = sum_{n in W} Psi_nk(r) Psi_nk(r)^dag`` the spinor
    density matrix of window W at k.  ``Gamma=I`` for charge. For transverse
    current the three ``Gamma=alpha_i/alpha_fs`` channels are summed and
    ``D`` is the ``4x4`` bispinor matrix. The driver uses ``sqrt(s)`` as the
    Lloyd mass. In the one-k, one-component, equal-window limit this is
    exactly the historical band density; over multiple k points it is the
    norm of the k-stacked feature row.

    ``D_Wk`` comes from :func:`gw.qsgw_density.rho_from_wfns` (the density
    scan's local route, ``per_k``) with unit weights over the window: bands
    are sharded over the whole mesh, every rank scans every stored parent,
    and one psum per chunk returns band-complete matrices.  Equal windows
    (every production deck) build one ``D`` and use it on both sides.  Each
    distinct symmetry row accumulates its weighted parent fields, and the
    FFT-grid pullback of that row is applied once at the end.
    """
    from common import timing
    from common.collectives import process_rank_world, single_device_mesh
    from common.wfn_layout import band_sphere_spec
    from gw.qsgw_density import rho_from_wfns
    from wfn_loader import IBZRows, WfnLoader

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
    ns = 4 if mode == "transverse" else int(wfn.nspinor)

    parents_used, star_plan, _ = _quadrature_tables(wfn, sym)

    rank, world = process_rank_world()
    if world > 1 and dist_mesh is None:
        raise ValueError(
            "build_feature_metric_diagonal requires dist_mesh at P>1 so the "
            f"wavefunction sweep is partitioned rather than repeated on {world} "
            "processes")
    mesh = single_device_mesh() if dist_mesh is None else dist_mesh

    # Full-BZ member weight of each parent under each distinct symmetry row.
    sym_rows = np.unique(np.concatenate(
        [star_plan[int(p)][0] for p in parents_used]))
    row_slot = {int(row): i for i, row in enumerate(sym_rows)}
    row_weights = np.zeros((parents_used.size, sym_rows.size), np.float64)
    for i, parent in enumerate(parents_used):
        for row, weight in zip(*star_plan[int(parent)]):
            row_weights[i, row_slot[int(row)]] += float(weight)

    windows = ((left_range,) if left_range == right_range
               else (left_range, right_range))
    union_lo = min(left_range[0], right_range[0])
    union_hi = max(left_range[1], right_range[1])
    try:
        from common.gpu_utils import get_device_memory_gb
        device_memory_bytes = int(float(get_device_memory_gb()) * 1e9)
    except Exception:
        device_memory_bytes = None
    k_chunk, band_chunk, budget = _metric_chunk_plan(
        n_parents=int(parents_used.size), n_bands=union_hi - union_lo,
        n_band_shards=int(mesh.devices.size), ns=ns,
        ngkmax=int(wfn.ngkmax), n_grid=n_grid, n_windows=len(windows),
        device_memory_bytes=device_memory_bytes)
    if rank == 0:
        print(
            f"  {mode} metric plan: {len(parents_used)} parent(s) in "
            f"chunks of {k_chunk}, bands [{union_lo},{union_hi}) in chunks "
            f"of {band_chunk} over {int(mesh.devices.size)} band shard(s); "
            f"{len(windows)} density matrix window(s), "
            f"{sym_rows.size} symmetry row(s); budget "
            f"{budget / 2**30:.2f} GiB/rank",
            flush=True)

    t0 = time.perf_counter()
    row_fields = jnp.zeros((sym_rows.size, n_grid), dtype=jnp.float64)
    for c0 in range(0, parents_used.size, k_chunk):
        chunk = tuple(int(p) for p in parents_used[c0:c0 + k_chunk])
        k_spec = IBZRows(chunk)
        box_index = wfn.box_index(k=k_spec)
        density = [None] * len(windows)
        for lo in range(union_lo, union_hi, band_chunk):
            hi = min(lo + band_chunk, union_hi)
            bands = np.arange(lo, hi)
            with timing.section("metric.load"):
                psi_g = wfn.load(bands=(lo, hi), k=k_spec,
                                 sharding=band_sphere_spec(),
                                 bispinor=(mode == "transverse"))
                psi_g.block_until_ready()
            for w, (w_lo, w_hi) in enumerate(windows):
                unit = ((bands >= w_lo) & (bands < w_hi)).astype(np.float64)
                if not unit.any():
                    continue
                part = rho_from_wfns(
                    psi_g, np.broadcast_to(unit, (len(chunk), unit.size)),
                    np.ones(len(chunk)), mesh=mesh, box_index=box_index,
                    fft_grid=fft_grid, cell_volume=cell_volume,
                    spin_degeneracy=1.0, return_spin_density_matrix=ns > 1,
                    per_k=True, memory_budget_bytes=budget)
                if ns == 1:                      # the 1x1 matrix is rho_k
                    part = part[:, None, None].astype(jnp.complex128)
                density[w] = part if density[w] is None else density[w] + part
            del psi_g
        row_fields = _accumulate_metric_rows(
            row_fields, density[0], density[-1],
            jnp.asarray(row_weights[c0:c0 + k_chunk]), mode=mode)
        del density
        if verbose:
            print(
                f"    [{mode} metric] {c0 + len(chunk)}/{len(parents_used)} "
                f"parents after {time.perf_counter() - t0:.1f}s", flush=True)

    metric_dev = jnp.zeros(n_grid, dtype=jnp.float64)
    for slot, row in enumerate(sym_rows):
        pullback = sym.fft_grid_pullback(
            np.asarray([int(row)], dtype=np.int32), fft_grid, validate=True)
        if pullback.shape != (1, n_grid):
            raise ValueError(
                "symmetry-service FFT-grid pullback has the wrong "
                f"shape: {pullback.shape} != {(1, n_grid)}")
        metric_dev = _accumulate_grid_pullback(
            metric_dev, row_fields[slot],
            jnp.asarray(pullback[0], dtype=jnp.int32),
            jnp.asarray(1.0, dtype=jnp.float64))
        del pullback
    metric = np.asarray(metric_dev, dtype=np.float64).reshape(fft_grid)
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
                 NO_PAIR_DIRAC_CURRENT_MODEL)
        print(
            f"  {mode} feature-Gram diagonal ({model}, left={left_range}, "
            f"right={right_range}, unit band weights, {len(parents_used)} "
            f"parents over {int(mesh.devices.size)} band shard(s), "
            f"{sym_rows.size} pullback row(s)) built in "
            f"{time.perf_counter() - t0:.2f}s", flush=True)
    return metric


__all__ = [
    "build_feature_metric_diagonal",
    "full_k_quadrature_weights",
]
