"""Transverse-current sampling weight for ISDF centroid selection.

The sampling importance is the positive, subspace-gauge-invariant quantity
``sum_{k,i,n,m} |<n k|alpha_i|m k>/alpha_fs|^2``.  Equivalently, at each
real-space point it is ``sum_i Tr(D alpha_i D alpha_i) / alpha_fs^2`` with
``D = sum_n |Psi_n><Psi_n|`` over the requested unit-weight band window.
It is neither the diagonal-only state importance nor the physical total
current magnitude ``sum_i |sum_n <n|alpha_i|n>|^2``.

Symmetry mapping, kinetic-balance lift, G-sphere placement, and FFT stay in
their existing owners.  At P>1 parent work is process partitioned, its band
windows stream locally, and one grid all-reduce combines the partial weights.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from common.bispinor_init import ALPHA_FS, NO_PAIR_DIRAC_CURRENT_MODEL
from common.gamma_matrices import gamma_apply, gamma_perm_phase


@jax.jit
def _accumulate_subspace_density_matrix(
    density_matrix,
    psi_r,
    band_mask,
):
    """Stream one fixed band window into local ``D_ab(r)``.

    The scan forms only one ``(4,4,nx,ny,nz)`` spin outer product at a time.
    There is no ``(nb,nb,...)`` state-pair carrier.
    """
    mask = jnp.asarray(band_mask, dtype=jnp.float64)

    def add_state(D, state_and_mask):
        psi_n, include = state_and_mask
        outer = (psi_n[:, None, ...]
                 * jnp.conj(psi_n[None, :, ...]))
        return D + include * outer, None

    return jax.lax.scan(
        add_state, density_matrix, (psi_r, mask), unroll=1)[0]


@jax.jit
def _transverse_importance_from_density_matrix(
    density_matrix,
    wavefunction_scale,
):
    """Return ``sum_i Tr(D alpha_i D alpha_i)`` with physical scaling."""
    out = jnp.zeros(density_matrix.shape[-3:], dtype=jnp.float64)
    for mu in (1, 2, 3):
        perm, phase = gamma_perm_phase(mu)
        alpha_D = gamma_apply(
            density_matrix, perm, phase, axis=0)
        out = out + jnp.real(jnp.sum(
            alpha_D * jnp.swapaxes(alpha_D, 0, 1), axis=(0, 1)))
    current_scale = (jnp.asarray(wavefunction_scale, dtype=jnp.float64) ** 2
                     / ALPHA_FS)
    # The exact expression is nonnegative.  Do not clamp a negative result:
    # that would let a wrong contraction masquerade as a sampling weight.
    return out * jnp.square(current_scale)


@jax.jit
def transverse_current_sampling_weight_from_psi_r(
    psi_r,
    band_mask,
    wavefunction_scale,
):
    """Gauge-invariant current importance for one complete band carrier.

    ``psi_r`` has shape ``(nb,4,nx,ny,nz)`` in the unitary-FFT convention.
    ``wavefunction_scale = sqrt(N_grid / Omega)`` converts it to physical
    unit-cell normalization before the bilinear is formed.  ``band_mask`` is
    strictly a fixed-window overlap mask: its included states are 1 and its
    already-counted overlap states are 0.  It is never an occupation.

    The function accumulates the 4x4 local subspace density matrix and then
    computes ``sum_i Tr(D alpha_i D alpha_i)``.  It is a focused public
    primitive for tests and small callers; the production builder invokes the
    two kernels separately so bands can be streamed in fixed-width windows.
    """
    if psi_r.ndim != 5 or psi_r.shape[1] != 4:
        raise ValueError(
            "transverse_current_sampling_weight_from_psi_r expects "
            "(nb,4,nx,ny,nz), got "
            f"{psi_r.shape}")
    mask = jnp.asarray(band_mask, dtype=jnp.float64).reshape(
        -1, 1, 1, 1)
    if int(mask.shape[0]) != int(psi_r.shape[0]):
        raise ValueError(
            "band_mask must have one unit/zero entry per state; got "
            f"{mask.shape[0]} for {psi_r.shape[0]} states")
    density_matrix = jnp.zeros(
        (4, 4) + tuple(psi_r.shape[-3:]), dtype=psi_r.dtype)
    density_matrix = _accumulate_subspace_density_matrix(
        density_matrix, psi_r, mask.reshape(-1))
    return _transverse_importance_from_density_matrix(
        density_matrix, wavefunction_scale)


@jax.jit
def _accumulate_grid_pullback(accumulator, field, pullback, member_weight):
    """Add one service-owned scalar pullback without a host field copy."""
    return accumulator + member_weight * field.reshape(-1)[pullback]


def build_current_density(
    wfn,
    sym,
    band_range: tuple[int, int],
    *,
    dist_mesh=None,
    verbose: bool = True,
):
    """Build the transverse-current sampling weight on the WFN FFT grid.

    Every band in the 0-based half-open ``band_range`` enters ``D`` with unit
    weight.  No WFN occupation or ``f_nk`` enters centroid fitting.  This is
    true for a valence-only range and for a range extended through requested
    conduction bands.  K points retain the WFN quadrature: a normalized
    stored-parent weight is divided equally over that parent's full-BZ star.

    The expensive lift and FFT are evaluated once per referenced raw WFN IBZ
    parent and streamed band window.  For a full-zone member ``k`` with parent
    ``p = irr_idx_k[k]`` and selected symmetry row ``s = sym_idx_k[k]``, the
    three alpha matrices transform as a polar vector (and are odd under time
    reversal), hence their summed subspace Frobenius norm is a scalar:

        sum_i Tr(D_k alpha_i D_k alpha_i)(r_new)
          = sum_i Tr(D_p alpha_i D_p alpha_i)(r_old).

    ``SymMaps.fft_grid_pullback`` supplies the exact nonsymmorphic
    ``r_new -> r_old`` gather for each typed row.  Its antiunitary current
    sign drops out of the square.  Thus the star accumulation is exactly the
    full-BZ WfnLoader-unfold result at fixed band index, without rebuilding
    any wavefunction, G-vector, FFT-box, or symmetry action here.

    At P>1, ``dist_mesh`` is required.  IBZ parents are divided round-robin
    over processes; their bands stream through one local 4x4 ``D(r)`` at a
    time.  All ranks execute the same number of same-shaped parent/window FFT
    rounds; surplus rounds reload the first parent's windows with all-zero
    overlap masks.  One final ``psum`` combines their full-grid partials.
    Parent-level distribution is deliberate: splitting a parent's bands over
    ranks would require a 16-field ``D`` all-reduce before the nonlinear trace
    and would replace one grid collective with one per parent.
    """
    from common.collectives import (process_rank_world, psum_replicate,
                                    single_device_mesh)
    from common.wfn_transforms import to_rbox
    from symmetry_maps import star_tables_of
    from wfn_loader import IBZRows, WfnLoader, uniform_band_windows

    if not isinstance(wfn, WfnLoader):
        raise TypeError(
            "build_current_density requires the driver's open WfnLoader; "
            f"got {type(wfn).__name__}")
    if int(wfn.nspinor) != 2:
        raise ValueError(
            "density-mode=current requires a two-component Pauli WFN so the "
            f"canonical kinetic-balance lift is defined; got nspinor="
            f"{int(wfn.nspinor)}")

    if len(band_range) != 2:
        raise ValueError(
            "band_range must be a 0-based half-open (lo, hi) pair; "
            f"got {band_range!r}")
    b_lo, b_hi = int(band_range[0]), int(band_range[1])
    if b_lo < 0 or b_hi <= b_lo or b_hi > int(wfn.nbands):
        raise ValueError(
            "band_range must be a nonempty subset of "
            f"[0,{int(wfn.nbands)}); got {(b_lo, b_hi)}")
    fft_grid = tuple(int(v) for v in wfn.fft_grid)
    n_grid = int(np.prod(fft_grid))
    cell_volume = float(wfn.cell_volume)
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError(
            f"WFN cell volume must be finite and positive, got {cell_volume}")
    wavefunction_scale = float(np.sqrt(n_grid / cell_volume))
    # Same fixed-shape window rule as the charge-density stream.  The 3x
    # factor prices the r-box plus the current kernel's two live equivalents.
    bytes_per_band = 3 * 4 * n_grid * np.dtype(np.complex128).itemsize
    nk_full = int(sym.nk_tot)
    parent_for_k, sym_row_for_k, _ = star_tables_of(sym)
    if parent_for_k.shape != (nk_full,) or sym_row_for_k.shape != (nk_full,):
        raise ValueError(
            "SymMaps full-k tables disagree with nk_tot: "
            f"irr_idx_k={parent_for_k.shape}, sym_idx_k={sym_row_for_k.shape}, "
            f"nk_tot={nk_full}")
    if (np.any(parent_for_k < 0)
            or np.any(parent_for_k >= int(wfn.nkpts))):
        raise ValueError(
            "SymMaps.irr_idx_k contains a row outside the raw WFN k axis "
            f"[0,{int(wfn.nkpts)})")
    parents_used = np.unique(parent_for_k)
    if parents_used.size == 0:
        raise ValueError("SymMaps contains no full-BZ parent rows")

    kweights = np.asarray(wfn.kweights, dtype=np.float64)
    if kweights.shape != (int(wfn.nkpts),):
        raise ValueError(
            "WfnLoader.kweights must have one entry per raw WFN k row; "
            f"got {kweights.shape}, expected {(int(wfn.nkpts),)}")
    weight_sum = float(kweights.sum())
    if (not np.all(np.isfinite(kweights)) or np.any(kweights < 0.0)
            or not np.isfinite(weight_sum) or weight_sum <= 0.0):
        raise ValueError(
            "WfnLoader.kweights must be finite, nonnegative, and have "
            f"positive sum; got sum={weight_sum}")
    kweights = kweights / weight_sum
    used_weight = float(kweights[parents_used].sum())
    if not np.isclose(used_weight, 1.0, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError(
            "SymMaps full-BZ parents omit nonzero WFN quadrature weight: "
            f"selected normalized weight={used_weight:.17g}, want 1")

    star_rows = {
        int(parent): np.asarray(
            sym_row_for_k[parent_for_k == parent], dtype=np.int32)
        for parent in parents_used
    }
    if any(rows.size == 0 for rows in star_rows.values()):
        raise ValueError("a selected WFN parent has an empty full-BZ star")

    rank, world = process_rank_world()
    if world > 1 and dist_mesh is None:
        raise ValueError(
            "build_current_density requires dist_mesh at P>1 so the "
            "wavefunction/current sweep is partitioned rather than repeated "
            f"on all {world} processes")

    span = b_hi - b_lo
    chunk = max(1, min(span, (4 * 1024 ** 3) // bytes_per_band))
    windows = uniform_band_windows(b_lo, b_hi, chunk)
    width = int(windows[0][1].shape[0])
    n_rounds = -(-int(parents_used.size) // world)
    parents_local = [(int(parent), True)
                     for parent in parents_used[rank::world]]
    while len(parents_local) < n_rounds:
        parents_local.append((int(parents_used[0]), False))

    t0 = time.perf_counter()
    last_log = t0
    mesh = single_device_mesh()
    sampling_weight_local = jnp.zeros(n_grid, dtype=jnp.float64)
    real_parents_done = 0
    pullback_rows_done = 0
    for parent, include_parent in parents_local:
        k_spec = IBZRows((parent,))
        box_index = wfn.box_index(k=k_spec)
        density_matrix = jnp.zeros(
            (4, 4) + fft_grid, dtype=jnp.complex128)
        for lo, canonical_mask in windows:
            band_mask = (np.asarray(canonical_mask, dtype=np.float64)
                         if include_parent else
                         np.zeros_like(canonical_mask, dtype=np.float64))
            psi_g = wfn.load_process_local(
                bands=(lo, lo + width), k=k_spec, bispinor=True)
            psi_r = to_rbox(
                psi_g,
                box_index,
                fft_grid,
                mesh=mesh,
                norm="ortho",
            )
            if int(psi_r.shape[0]) != 1:
                raise ValueError(
                    "one-parent WfnLoader request returned "
                    f"{int(psi_r.shape[0])} k rows")
            density_matrix = _accumulate_subspace_density_matrix(
                density_matrix, psi_r[0], jnp.asarray(band_mask))
            # One band box plus one 4x4 D at a time, independent of the total
            # requested band count.
            density_matrix.block_until_ready()
            del psi_g, psi_r

        parent_importance = _transverse_importance_from_density_matrix(
            density_matrix, wavefunction_scale)
        if include_parent:
            member_weight = (float(kweights[parent])
                             / float(star_rows[parent].size))
            # Stream exactly one N_grid pullback.  Blocking each addition
            # before resolving the next row prevents a latent
            # O(N_sym*N_grid) queue of device permutations.
            for row in star_rows[parent]:
                row_i = int(row)
                pullback = sym.fft_grid_pullback(
                    np.asarray([row_i], dtype=np.int32),
                    fft_grid,
                    validate=True,
                )
                if pullback.shape != (1, n_grid):
                    raise ValueError(
                        "symmetry-service FFT-grid pullback has the wrong "
                        f"shape: {pullback.shape} != {(1, n_grid)}")
                pullback_dev = jnp.asarray(
                    pullback[0], dtype=jnp.int32)
                sampling_weight_local = _accumulate_grid_pullback(
                    sampling_weight_local,
                    parent_importance,
                    pullback_dev,
                    jnp.asarray(member_weight, dtype=jnp.float64),
                )
                sampling_weight_local.block_until_ready()
                pullback_rows_done += 1
                del pullback, pullback_dev
            real_parents_done += 1
        else:
            parent_importance.block_until_ready()
        del density_matrix, parent_importance
        if verbose and time.perf_counter() - last_log > 5.0:
            last_log = time.perf_counter()
            print(f"    [transverse-current weight] rank {rank}: "
                  f"{real_parents_done}/{len(parents_local)} local parents "
                  f"after {last_log - t0:.1f}s", flush=True)

    sampling_weight_flat = np.asarray(
        sampling_weight_local, dtype=np.float64)
    if world > 1:
        sampling_weight_flat = psum_replicate(sampling_weight_flat, dist_mesh)
    sampling_weight = sampling_weight_flat.reshape(fft_grid)
    if verbose:
        dt = time.perf_counter() - t0
        distribution = (
            f"{len(parents_used)} parents / {world} ranks, one psum"
            if world > 1 else f"{len(parents_used)} serial parent(s)")
        print(
            "  transverse-current sampling weight "
            f"sum_(k,i,n,m)|<nk|alpha_i|mk>|^2 "
            f"({NO_PAIR_DIRAC_CURRENT_MODEL}, unit band weights, "
            "normalized WFN k weights + exact star pullback, "
            f"bands=[{b_lo},{b_hi}), chunk={chunk}, {distribution}) built "
            f"with {pullback_rows_done} streamed local pullback row(s) "
            f"in {dt:.2f}s; max={float(sampling_weight.max()):.3e}, "
            f"mean={float(sampling_weight.mean()):.3e}", flush=True)
    return sampling_weight


__all__ = [
    "build_current_density",
    "transverse_current_sampling_weight_from_psi_r",
]
