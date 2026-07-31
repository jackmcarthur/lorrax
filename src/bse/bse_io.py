"""I/O and padding utilities for BSE ISDF data."""
from __future__ import annotations

import glob
import os
from functools import partial
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime.padding import padded_mu_extent
from common.fft_helpers import local_fftn3, local_ifftn3

from .bse_serial import compute_pair_amplitude


_BARE_V_FALLBACK_WARNING = (
    "\n" + "=" * 72 + "\n"
    "BSE WARNING: W0_qmunu not ready (missing or W0_ready=False) -- falling\n"
    "back to BARE COULOMB V for the screened interaction W. This is NOT a\n"
    "physical BSE calculation: excitonic binding from screening is entirely\n"
    "absent. Re-run GW screening to produce a ready W0_qmunu before trusting\n"
    "these results.\n"
    + "=" * 72
)


def _generate_kpts_grid(nkx: int, nky: int, nkz: int) -> np.ndarray:
    """Monkhorst-Pack style k-point grid in crystal coords [0, 1), C-order.

    Returns ``(nk, 3)``.  Single consumer: ``write_eigenvectors_stream``.
    """
    kpts = []
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                kx = ix / nkx
                ky = iy / nky
                kz = iz / nkz if nkz > 0 else 0.0
                kpts.append([kx, ky, kz])
    return np.array(kpts, dtype=np.float64)


def write_eigenvectors_stream(
    output_file: str,
    eigenvalues: jax.Array,
    eigenvectors: jax.Array,
    n_val: int,
    n_cond: int,
    nkx: int,
    nky: int,
    nkz: int,
    n_write: int,
    use_tda: bool = True,
) -> None:
    # ``use_tda`` is written HONESTLY (was hardcoded 1).  TDA eigenvectors arrive
    # as ``(n_write, 1, nc, nv, nk)`` (resonant X only); full-BSE (non-TDA) as
    # ``(n_write, 2, nc, nv, nk)`` = the paired (X, Y) with X^H X - Y^H Y = +1.
    # For non-TDA the resonant X is written to ``eigenvectors`` (so the
    # sum-over-states absorption reads the dominant part) and the coupling Y to a
    # sibling ``eigenvectors_coupling`` dataset (both components persisted).
    # BGW eigenvectors.h5 stores eigenvalues in eV (header text in
    # ``eigenvalues.dat`` says "eig (eV)"; matches BGW's BSE/diag.f90
    # write path).  Our solvers return Ry — convert here so a downstream
    # consumer using BGW conventions reads the right number.
    RYD2EV = 13.6056980659
    eigenvalues = np.asarray(jax.device_get(eigenvalues[:n_write])) * RYD2EV
    kpts = _generate_kpts_grid(nkx, nky, nkz)
    nk = kpts.shape[0]
    ns = 1
    nQ = 1
    flavor = 2
    spin_kernel = 3
    bse_hamiltonian_size = ns * nk * n_val * n_cond
    evec_sz = bse_hamiltonian_size

    kpts_fortran = kpts.T.copy()
    exciton_Q_shifts = np.zeros((1, 3), dtype=np.float64)

    with h5py.File(output_file, "w") as f:
        f.create_group("mf_header")
        f.create_group("eps_header")
        f.create_group("bse_header")

        exciton_header = f.create_group("exciton_header")
        exciton_header.create_dataset("version", data=1)
        exciton_header.create_dataset("flavor", data=flavor)

        params = exciton_header.create_group("params")
        params.create_dataset("bse_hamiltonian_size", data=bse_hamiltonian_size)
        params.create_dataset("evec_sz", data=evec_sz)
        params.create_dataset("spin_kernel", data=spin_kernel)
        params.create_dataset("nevecs", data=n_write)
        params.create_dataset("ns", data=ns)
        params.create_dataset("nc", data=n_cond)
        params.create_dataset("nv", data=n_val)
        params.create_dataset("use_tda", data=1 if use_tda else 0)

        kpoints = exciton_header.create_group("kpoints")
        kpoints.create_dataset("nk", data=nk)
        kpoints.create_dataset("kpts", data=kpts_fortran)
        kpoints.create_dataset("nQ", data=nQ)
        kpoints.create_dataset("exciton_Q_shifts", data=exciton_Q_shifts.T)

        exciton_data = f.create_group("exciton_data")
        exciton_data.create_dataset("eigenvalues", data=eigenvalues)
        evec_dset = exciton_data.create_dataset(
            "eigenvectors",
            shape=(1, n_write, nk, n_cond, n_val, ns, 2),
            dtype=np.float64,
        )
        coupling_dset = None
        if not use_tda:
            coupling_dset = exciton_data.create_dataset(
                "eigenvectors_coupling",
                shape=(1, n_write, nk, n_cond, n_val, ns, 2),
                dtype=np.float64,
            )

        def _to_bgw(comp):
            # comp: (nc, nv, nk) -> BGW (nk, nc, nv, ns) layout.  BGW convention:
            # valence axis reversed (v=0 = highest valence, BSE/input_fi.f90:407);
            # our internal slice puts v=0 at deepest valence, so flip on write.
            c = np.transpose(comp, (2, 0, 1))[:, :, ::-1][..., None]
            return c.real, c.imag

        for i in range(n_write):
            vec = jax.device_get(eigenvectors[i])
            if use_tda:
                # (1, nc, nv, nk) sharded or (nc, nv, nk) unsharded -> resonant X.
                Xc = vec[0] if vec.ndim == 4 else vec
                Yc = None
            else:
                # (2, nc, nv, nk) = paired (X, Y) from the non-TDA solver.
                Xc, Yc = vec[0], vec[1]
            re, im = _to_bgw(Xc)
            evec_dset[0, i, :, :, :, :, 0] = re
            evec_dset[0, i, :, :, :, :, 1] = im
            if Yc is not None:
                re, im = _to_bgw(Yc)
                coupling_dset[0, i, :, :, :, :, 0] = re
                coupling_dset[0, i, :, :, :, :, 1] = im

    print(f"Wrote {n_write} eigenvectors to {output_file}"
          + ("" if use_tda else " (+ coupling Y)"))


def _pad_last_axis(x: jax.Array, target: int) -> jax.Array:
    pad = target - x.shape[-1]
    if pad <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, pad)
    return jnp.pad(x, pad_width, mode="constant")


def _pad_last_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[-2]
    pad1 = target - x.shape[-1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-2] = (0, max(0, pad0))
    pad_width[-1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")


def _pad_first_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[0]
    pad1 = target - x.shape[1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[0] = (0, max(0, pad0))
    pad_width[1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")


def _zeropad_R_axis(x: jax.Array, axis: int, n_fine: int) -> jax.Array:
    """Spectral zero-pad one real-space (R) axis from a coarse to a fine k-lattice.

    ``x`` is a real-space tensor whose ``axis`` is the R-lattice DUAL to a
    COARSE BZ axis of length ``n_c = x.shape[axis]`` (i.e. an ``ifft`` over that
    BZ axis already happened).  We embed the ``n_c`` coarse Fourier coefficients
    into ``n_fine`` slots so that an ``fft`` back to ``n_fine`` BZ points is the
    exact band-limited trigonometric interpolant of the coarse ``W(k)`` — which,
    because the coarse BZ points are a subset of the fine ones (``n_fine`` a
    multiple of ``n_c``), passes THROUGH the coarse samples exactly.

    Per-element R-lattice index map (the crux).  numpy/JAX FFT order maps array
    index ``i∈[0,N)`` to the physical lattice-vector rep ``n = fftfreq(N)*N``:

        n = i        for 0 ≤ i < ⌈N/2⌉     (the non-negative reps 0,1,…)
        n = i − N    for ⌈N/2⌉ ≤ i < N      (the negative reps …,−2,−1, wrapped)

    For even ``N`` the single Nyquist rep ``n = −N/2`` sits at ``i = N/2`` (it is
    in the negative branch here — kept single-sided, NOT split; sample agreement
    is exact regardless, and single-sided = "coarse WS cell, zero outside",
    which is precisely the requested zero-pad).  Embedding coarse→fine with
    ``s = ⌈n_c/2⌉ = (n_c+1)//2`` non-negative reps:

        coarse i_c ∈ [0, s)      (reps n = 0 … s−1)      → fine i_f = i_c
        coarse i_c ∈ [s, n_c)    (reps n = −(n_c−s) … −1) → fine i_f = i_c + (n_fine − n_c)

    i.e. the low block keeps its indices, the high (negative-freq) block slides
    to the TOP of the fine axis, and the gap ``i_f ∈ [s, n_fine − (n_c−s))`` is
    ZERO (the high-|R| coefficients absent from the coarse cell — incl. the
    fine-only Nyquist and the absent coarse ``+n_c/2``).  Realised as
    ``concat([low, zeros, high])`` — the textbook fft zero-insertion, exact for
    any parity of ``n_c``.

    NOTE: scale is applied ONCE by the caller (``pad_W_R_to_grid``), not here.
    """
    n_c = x.shape[axis]
    if n_fine == n_c:
        return x
    if n_fine < n_c or n_fine % n_c != 0:
        raise ValueError(
            f"_zeropad_R_axis: fine length {n_fine} must be a positive multiple "
            f"of the coarse length {n_c} (coarse BZ points ⊂ fine BZ points).")
    s = (n_c + 1) // 2
    lo = jax.lax.slice_in_dim(x, 0, s, axis=axis)
    hi = jax.lax.slice_in_dim(x, s, n_c, axis=axis)
    zshape = list(x.shape)
    zshape[axis] = n_fine - n_c
    zeros = jnp.zeros(zshape, dtype=x.dtype)
    return jnp.concatenate([lo, zeros, hi], axis=axis)


def pad_W_R_to_grid(W_R_coarse: jax.Array,
                    fine_grid: tuple[int, int, int]) -> jax.Array:
    """Zero-pad a coarse-k-grid real-space ``W_R`` onto a finer k-lattice.

    Enables a CHEAP coarse-grid W (e.g. GW/W on 6×6) to drive the BSE direct
    term on an arbitrarily FINE interpolated exciton sampling (e.g. 12×12):
    zero-padding in R is EXACT trigonometric interpolation in k that agrees
    with the coarse ``W(k)`` at the coarse sub-k-points that coincide with the
    fine grid (they do when each fine length is a multiple of the coarse one).

    Parameters
    ----------
    W_R_coarse : (..., ncx, ncy, ncz)
        Screened interaction on the real-space (R) lattice DUAL to a COARSE BZ
        grid, i.e. ``ifftn(W_q_coarse, axes=last-3, norm='ortho')``.  Leading
        axes (μ, ν, spin, …) are carried through untouched; only the trailing
        three (kx, ky, kz) R-axes are re-embedded.
    fine_grid : (nfx, nfy, nfz)
        Target BZ grid.  Each ``nf`` must be a positive multiple of the matching
        coarse length (so the coarse BZ ⊂ fine BZ).

    Returns
    -------
    W_R_fine : (..., nfx, nfy, nfz)
        Such that ``fftn(W_R_fine, norm='ortho')`` sampled at the fine BZ points
        coinciding with the coarse grid equals ``W_q_coarse`` bit-close, i.e.
        ``W_R_fine == ifftn(W_q_fine_interpolant, norm='ortho')`` — a drop-in for
        the matvec's ``W_R`` on the FINE grid.

    Scale.  ortho FFTs carry ``1/√N``, and ``N`` grows coarse→fine.  For the
    fine ``fft`` of the embedded coefficients to reproduce the coarse samples,
    the padded tensor is multiplied by ``√(∏nf / ∏nc)`` (= 2 for 6×6×1→12×12×1).

    Degenerate ``fine_grid == coarse shape``: returns ``W_R_coarse`` UNCHANGED
    (exact byte-identical no-op — scale is 1 and no axis is padded).
    """
    if W_R_coarse.ndim < 3:
        raise ValueError(
            f"pad_W_R_to_grid: expected trailing (kx,ky,kz) axes, got ndim="
            f"{W_R_coarse.ndim}")
    coarse_grid = tuple(int(s) for s in W_R_coarse.shape[-3:])
    fine_grid = tuple(int(s) for s in fine_grid)
    if fine_grid == coarse_grid:
        return W_R_coarse                         # exact no-op (fast path)
    out = W_R_coarse
    for ax, nf in zip((-3, -2, -1), fine_grid):
        out = _zeropad_R_axis(out, ax, nf)
    n_c_tot = coarse_grid[0] * coarse_grid[1] * coarse_grid[2]
    n_f_tot = fine_grid[0] * fine_grid[1] * fine_grid[2]
    scale = jnp.sqrt(jnp.asarray(n_f_tot / n_c_tot, dtype=out.real.dtype))
    return out * scale


def decimate_W_q_to_subgrid(W_q: jax.Array,
                            coarse_grid: tuple[int, int, int]) -> jax.Array:
    """Sub-sample a fine-grid ``W_q`` onto a coarse BZ sub-grid (every m-th q).

    ``W_q`` is ``(μ, ν, nfx, nfy, nfz)`` on a FINE BZ grid; returns the tiles at
    the coarse sub-grid BZ points ``q_j = j·(nf/nc)`` (which coincide with a
    coarse ``nc``-point grid since ``nf`` is a multiple of ``nc``).  This is the
    honest "what a coarse-grid W sampling looks like, in the SAME ISDF μ-basis"
    — used to drive ``pad_W_R_to_grid`` from a single fine restart (the q=0 tile,
    incl. its rank-1 head, is preserved because q=0 is on both grids).
    """
    nf = tuple(int(s) for s in W_q.shape[-3:])
    coarse_grid = tuple(int(s) for s in coarse_grid)
    for a, (f, c) in enumerate(zip(nf, coarse_grid)):
        if c <= 0 or f % c != 0:
            raise ValueError(
                f"decimate_W_q_to_subgrid: fine axis {a} length {f} must be a "
                f"positive multiple of coarse {c}.")
    rx, ry, rz = (nf[0] // coarse_grid[0], nf[1] // coarse_grid[1],
                  nf[2] // coarse_grid[2])
    return W_q[:, :, ::rx, ::ry, ::rz]


def _pad_axis_to_multiple(x: jax.Array, axis: int, multiple: int) -> tuple[jax.Array, int]:
    size = x.shape[axis]
    pad = (-size) % multiple
    if pad == 0:
        return x, size
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad)
    # Return the PADDED extent (size + pad), not the pre-pad size: every
    # consumer binds this to n_val_pad/n_cond_pad and relies on it being the
    # mesh-rounded value (e.g. bse_ring_comm's `n_cond_pad % px == 0` guard).
    # Previously wrong only when the band count was not already a mesh
    # multiple — invisible on all mesh-divisible validated runs.
    return jnp.pad(x, pad_width, mode="constant"), size + pad


def _get_local_mesh_coords(
    mesh_xy: Mesh, *, origin: str = "bse_io._get_local_mesh_coords",
) -> tuple[list[tuple[int, int]], int, int]:
    """This process's (x, y) coords in ``mesh_xy``, plus the grid shape.

    Refuses up front (``collectives._require_addressable``) a mesh on which
    THIS process owns no device.  Every shard-aware reader below funnels
    through here and then hands its process-local block to
    ``jax.make_array_from_process_local_data``, which on a zero-addressable
    mesh runs ``next(iter(addressable_shards.values()))``
    (``jax/_src/array.py:1017``) over an empty map and raises a BARE
    ``StopIteration('')`` — no message, no rank, no mention of a mesh.  At
    P=2 that lands as "rank 0 succeeded, rank 1 died with an empty
    exception", which is what makes it undiagnosable (job 7882420).

    The readers are called with the run's full 2-D mesh today, so every
    rank IS addressable and this is hardening rather than a live bug.  It
    is placed HERE, not at the three ``make_array_from_process_local_data``
    call sites, because the very next line — ``np.argwhere(...)[0]`` over a
    device this mesh does not contain — would raise an equally anonymous
    ``IndexError`` first, before the guard could speak.
    """
    from common.collectives import _require_addressable

    _require_addressable(mesh_xy, origin=origin)
    devices_2d = np.asarray(mesh_xy.devices)
    grid_x, grid_y = devices_2d.shape
    local_devices = list(jax.local_devices())
    local_coords = [tuple(np.argwhere(devices_2d == d)[0]) for d in local_devices]
    local_coords = sorted(local_coords, key=lambda c: c[0] * grid_y + c[1])
    return local_coords, grid_x, grid_y


def _get_local_axis_coords(local_coords: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    local_x = sorted({coord[0] for coord in local_coords})
    local_y = sorted({coord[1] for coord in local_coords})
    return local_x, local_y


def _assert_local_block(local_coords: list[tuple[int, int]], local_x: list[int], local_y: list[int]) -> None:
    expected = {(x, y) for x in local_x for y in local_y}
    actual = set(local_coords)
    if actual != expected:
        raise ValueError(
            "Local devices are not a full x/y block; shard-aware loader expects "
            "local device coords to form a Cartesian product."
        )


def _read_psi_mu_sharded(
    dset: h5py.Dataset,
    band_indices: np.ndarray,
    mu_per_shard: int,
    axis: str,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(
        mesh_xy, origin="bse_io._read_psi_mu_sharded")
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)

    n_rmu = dset.shape[3]
    nk = dset.shape[0]
    nspinor = dset.shape[2]

    local_mu = mu_per_shard * (len(local_x) if axis == "x" else len(local_y))
    local_psi = np.zeros((nk, len(band_indices), nspinor, local_mu), dtype=dtype)

    if axis == "x":
        coords = local_x
    else:
        coords = local_y

    for i, coord in enumerate(coords):
        mu_start = coord * mu_per_shard
        mu_end = min(mu_start + mu_per_shard, n_rmu)
        if mu_start >= n_rmu:
            continue
        slab = dset[:, band_indices, :, mu_start:mu_end]
        if slab.shape[3] < mu_per_shard:
            pad_mu = mu_per_shard - slab.shape[3]
            slab = np.pad(slab, ((0, 0), (0, 0), (0, 0), (0, pad_mu)), mode="constant")
        mu_off = i * mu_per_shard
        local_psi[:, :, :, mu_off:mu_off + mu_per_shard] = slab

    global_shape = (nk, len(band_indices), nspinor, n_rmu_pad)
    psi_sharding = NamedSharding(mesh_xy, P(None, None, None, axis))
    local_psi_jax = jax.device_put(local_psi)
    psi_global = jax.make_array_from_process_local_data(psi_sharding, local_psi_jax, global_shape)
    if trim and n_rmu_pad > n_rmu:
        psi_global = psi_global[..., :n_rmu]
    return psi_global


def _resolve_munu_reader(
    dset: h5py.Dataset,
    kgrid: Optional[tuple[int, int, int]] = None,
):
    """Resolve ``(n_rmu, n_rnu, nkx, nky, nkz, read_slab)`` for a V/W μν dataset.

    Single source of the on-disk layout shim both sharded readers need.
    Handles three layouts:

      * 8-D legacy ``(1, npol, npol, nkx, nky, nkz, μ, ν)`` — kgrid from shape.
      * 6-D transitional ``(1, npol, npol, nq, μ, ν)``.
      * 3-D flat-q ``(nq, μ, ν)``.

    For the 6-D/3-D layouts the caller must pass ``kgrid=(nkx, nky, nkz)`` (or
    the dataset must carry a ``'kgrid'`` attribute).  ``read_slab(mu0, mu1,
    nu0, nu1)`` returns the ``(μ, ν, nkx, nky, nkz)`` slab for that μ/ν block;
    q=0 is the ``(0, 0, 0)`` k-slice.
    """
    if dset.ndim == 8:
        n_rmu = int(dset.shape[6])
        n_rnu = int(dset.shape[7])
        nkx, nky, nkz = (int(s) for s in dset.shape[3:6])
        read_slab = lambda mu0, mu1, nu0, nu1: np.transpose(
            dset[0, 0, 0, :, :, :, mu0:mu1, nu0:nu1], (3, 4, 0, 1, 2))
        return n_rmu, n_rnu, nkx, nky, nkz, read_slab
    if dset.ndim == 6:
        n_rmu = int(dset.shape[-2])
        n_rnu = int(dset.shape[-1])
        nq = int(dset.shape[3])
        _flat = lambda mu0, mu1, nu0, nu1: np.asarray(
            dset[0, 0, 0, :, mu0:mu1, nu0:nu1])
    else:  # 3-D flat-q
        n_rmu = int(dset.shape[-2])
        n_rnu = int(dset.shape[-1])
        nq = int(dset.shape[0])
        _flat = lambda mu0, mu1, nu0, nu1: np.asarray(
            dset[:, mu0:mu1, nu0:nu1])
    if kgrid is not None:
        nkx, nky, nkz = (int(v) for v in kgrid)
    elif 'kgrid' in dset.attrs:
        nkx, nky, nkz = (int(v) for v in dset.attrs['kgrid'])
    else:
        raise ValueError(
            "_resolve_munu_reader: V/W dataset is flat-q but no kgrid was "
            "passed and no 'kgrid' attribute is present; the caller must "
            "resolve (nkx, nky, nkz) (from the top-level 'kgrid' dataset or "
            "the WFN) and pass it in.")
    if nkx * nky * nkz != nq:
        raise ValueError(f"kgrid {nkx}×{nky}×{nkz} ≠ nq={nq}")
    read_slab = (lambda mu0, mu1, nu0, nu1:
        _flat(mu0, mu1, nu0, nu1)
        .reshape(nkx, nky, nkz, mu1 - mu0, nu1 - nu0)
        .transpose(3, 4, 0, 1, 2))
    return n_rmu, n_rnu, nkx, nky, nkz, read_slab


def _read_vq0_sharded(
    dset: h5py.Dataset,
    mu_per_x: int,
    nu_per_y: int,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
    kgrid: Optional[tuple[int, int, int]] = None,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(
        mesh_xy, origin="bse_io._read_vq0_sharded")
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    n_rmu, n_rnu, _nkx, _nky, _nkz, read_slab = _resolve_munu_reader(dset, kgrid=kgrid)

    local_mu = mu_per_x * len(local_x)
    local_nu = nu_per_y * len(local_y)
    local_v = np.zeros((local_mu, local_nu), dtype=dtype)

    for ix, x_coord in enumerate(local_x):
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start >= n_rmu:
            continue
        for iy, y_coord in enumerate(local_y):
            nu_start = y_coord * nu_per_y
            nu_end = min(nu_start + nu_per_y, n_rnu)
            if nu_start >= n_rnu:
                continue
            # q=0 is the (0, 0, 0) k-slice of the normalized (μ, ν, k...) slab.
            slab = read_slab(mu_start, mu_end, nu_start, nu_end)[:, :, 0, 0, 0]
            if slab.shape[0] < mu_per_x or slab.shape[1] < nu_per_y:
                pad_mu = mu_per_x - slab.shape[0]
                pad_nu = nu_per_y - slab.shape[1]
                slab = np.pad(slab, ((0, pad_mu), (0, pad_nu)), mode="constant")
            mu_off = ix * mu_per_x
            nu_off = iy * nu_per_y
            local_v[mu_off:mu_off + mu_per_x, nu_off:nu_off + nu_per_y] = slab

    global_shape = (n_rmu_pad, n_rmu_pad)
    v_sharding = NamedSharding(mesh_xy, P("x", "y"))
    local_v_jax = jax.device_put(local_v)
    v_global = jax.make_array_from_process_local_data(v_sharding, local_v_jax, global_shape)
    if trim and (global_shape[0] > n_rmu or global_shape[1] > n_rnu):
        v_global = v_global[:n_rmu, :n_rnu]
    return v_global


def _read_wq_sharded(
    dset: h5py.Dataset,
    mu_per_x: int,
    nu_per_y: int,
    mesh_xy: Mesh,
    n_rmu_pad: int,
    dtype: np.dtype = np.complex128,
    trim: bool = True,
    kgrid: Optional[tuple[int, int, int]] = None,
) -> jax.Array:
    local_coords, grid_x, grid_y = _get_local_mesh_coords(
        mesh_xy, origin="bse_io._read_wq_sharded")
    local_x, local_y = _get_local_axis_coords(local_coords)
    _assert_local_block(local_coords, local_x, local_y)
    # Layout shim (8-D / 6-D / 3-D-flat-q) is single-sourced in
    # ``_resolve_munu_reader``.  Internal BSE work below stays in the
    # ``(μ, μ, nkx, nky, nkz)`` form the reader already produces.
    n_rmu, n_rnu, nkx, nky, nkz, _read_munu_slab = _resolve_munu_reader(dset, kgrid=kgrid)

    local_mu = mu_per_x * len(local_x)
    local_nu = nu_per_y * len(local_y)
    local_w = np.zeros((local_mu, local_nu, nkx, nky, nkz), dtype=dtype)

    for ix, x_coord in enumerate(local_x):
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start >= n_rmu:
            continue
        for iy, y_coord in enumerate(local_y):
            nu_start = y_coord * nu_per_y
            nu_end = min(nu_start + nu_per_y, n_rnu)
            if nu_start >= n_rnu:
                continue
            slab = _read_munu_slab(mu_start, mu_end, nu_start, nu_end)
            if slab.shape[0] < mu_per_x or slab.shape[1] < nu_per_y:
                pad_mu = mu_per_x - slab.shape[0]
                pad_nu = nu_per_y - slab.shape[1]
                slab = np.pad(slab, ((0, pad_mu), (0, pad_nu), (0, 0), (0, 0), (0, 0)), mode="constant")
            mu_off = ix * mu_per_x
            nu_off = iy * nu_per_y
            local_w[mu_off:mu_off + mu_per_x, nu_off:nu_off + nu_per_y, :, :, :] = slab

    global_shape = (n_rmu_pad, n_rmu_pad, nkx, nky, nkz)
    w_sharding = NamedSharding(mesh_xy, P("x", "y", None, None, None))
    local_w_jax = jax.device_put(local_w)
    w_global = jax.make_array_from_process_local_data(w_sharding, local_w_jax, global_shape)
    if trim and (global_shape[0] > n_rmu or global_shape[1] > n_rnu):
        w_global = w_global[:n_rmu, :n_rnu, :, :, :]
    return w_global


def _parse_grid_spec(spec) -> Optional[tuple[int, int, int]]:
    """Parse a ``bse_k_grid`` value → ``(nx, ny, nz)`` or ``None`` (unset).

    Accepts ``"NX NY NZ"``, ``"NX,NY,NZ"``, a 3-tuple/list, or an empty
    string / None (→ ``None``).  Single source for both the cohsex.in key and
    any driver flag.
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        parts = list(spec)
    else:
        s = str(spec).strip()
        if not s:
            return None
        parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"bse_k_grid must have 3 integers, got {spec!r}")
    return tuple(int(x) for x in parts)


def _resolve_bse_k_grid(bse_k_grid, input_file: Optional[str]):
    """Resolve the fine grid: explicit ``bse_k_grid`` arg wins; else read the
    ``bse_k_grid`` key from ``input_file`` (cohsex.in).  Returns a 3-tuple or
    ``None`` (feature off)."""
    fine = _parse_grid_spec(bse_k_grid)
    if fine is not None:
        return fine
    if input_file is not None and os.path.isfile(input_file):
        try:
            from gw.gw_config import read_lorrax_input
            return _parse_grid_spec(read_lorrax_input(input_file).get("bse_k_grid"))
        except Exception as exc:               # config parse must never crash load
            print(f"BSE: bse_k_grid config read failed ({exc}); feature off")
    return None


def _interpolate_bse_data_to_grid(
    data: dict,
    fine_grid: tuple[int, int, int],
    restart_file: str,
    input_file: str,
    mesh_xy: Mesh,
    *,
    log_fn=print,
) -> dict:
    """Interpolate the WHOLE coarse BSE ``data`` bundle onto ``fine_grid``.

    The single coarse→fine densification the ``bse_k_grid`` knob drives, living
    in the GENERAL init so EVERY solver consumes a fine-grid bundle unchanged.
    Reuses the exact machinery the exciton_bands Q-path uses — no second copy:

      * ψ_{v,c}(k) and QP ε_{v,c}(k) on the fine mesh ← ONE htransform fH
        (``bandstructure.htransform.initialize_wfns`` +
        ``bandstructure.bse_setup.compute_wfns_fi`` with ``kgrid_fi=fine_grid``);
        the fH is built over the full loaded band window and only the BSE
        sub-window [nval−n_val, nval+n_cond) is returned (interior, guarded).
      * V_Q exchange q=0 tile ← ``bse.vq_interp.build_vq_evaluator`` (the SAME
        builder exciton_bands calls) evaluated at Q=0 with the FINE mini-BZ
        head (``minibz_head_vlr(..., kgrid=fine_grid)``).  A Q=0 exciton's
        exchange is the single q=0 tile (dense in k,k'); the fine k-grid's
        exchange "q-set" is therefore just q=0.
      * W direct ← ``pad_W_R_to_grid`` (coarse→fine zero-pad in R = exact trig
        interpolation), fft'd back to a fine ``W_q`` so each solver's own
        ``ifftn(W_q)`` reproduces the padded ``W_R``.  This is the mechanism the
        exciton_bands ``--w-coarse-grid`` flag runs; ``bse_k_grid`` drives it
        automatically (the banner below is the "W-pad fired" proof).

    Band dims (n_val/n_cond and their mesh-pads), n_rmu(_pad), and the q=0 head
    projectors g0 are grid-INVARIANT and carried through untouched; only the k
    axis (and W's k-axes) change.  Returns a new bundle with the SAME keys.
    """
    from bandstructure import htransform as ht
    from bandstructure.bse_setup import compute_wfns_fi
    from gw.gw_config import read_lorrax_input

    from . import vq_interp

    coarse_grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    fine_grid = tuple(int(s) for s in fine_grid)
    for a, (f, c) in enumerate(zip(fine_grid, coarse_grid)):
        if c <= 0 or f <= 0 or f % c != 0:
            raise ValueError(
                f"bse_k_grid axis {a}: fine {f} must be a positive multiple of "
                f"the coarse restart grid {c} (coarse BZ ⊂ fine BZ).")
    n_val = int(data["n_val"]); n_cond = int(data["n_cond"])
    nv_pad = int(data["n_val_pad"]); nc_pad = int(data["n_cond_pad"])
    n_rmu = int(data["n_rmu"]); n_rmu_pad = int(data["n_rmu_pad"])
    nk_f = fine_grid[0] * fine_grid[1] * fine_grid[2]
    log_fn(f"[bse_k_grid] coarse {coarse_grid[0]}x{coarse_grid[1]}x{coarse_grid[2]}"
           f" → fine {fine_grid[0]}x{fine_grid[1]}x{fine_grid[2]} "
           f"({nk_f} k-pts); interpolating ψ/ε (htransform), V_Q (vq_interp), "
           f"W (zero-pad in R)")

    params = read_lorrax_input(input_file)

    # ── ψ_{v,c}(k_fine), ε_{v,c}(k_fine): ONE htransform fH ───────────────
    (wfn, sym, meta, _mesh, _S, ctilde, B_at_mu,
     enk_sigma) = ht.initialize_wfns(input_file, params, log_fn, mesh_xy=mesh_xy)
    nb_window = int(ctilde.shape[1])
    nval_in = int(params["nval"])          # window-relative CBM index
    b_min, b_max = nval_in - n_val, nval_in + n_cond
    if b_min < 0 or b_max > nb_window:
        raise ValueError(
            f"bse_k_grid BSE window [{b_min},{b_max}) escapes the htransform fH "
            f"window [0,{nb_window}); the input's nval/nband must load "
            f">= {n_val} valence and >= {n_cond} conduction guard bands.")
    bundle = compute_wfns_fi(
        ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
        kgrid_co=coarse_grid, kgrid_fi=fine_grid,
        band_window_fi=(b_min, b_max), mesh_xy=mesh_xy, log_fn=log_fn)

    x4 = NamedSharding(mesh_xy, P(None, None, None, "x"))
    y4 = NamedSharding(mesh_xy, P(None, None, None, "y"))
    rep = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=(x4, y4, x4, y4, rep, rep))
    def _split_pad(psi, enk):
        # psi (nk_f, n_val+n_cond, ns, n_mu); enk (nk_f, n_val+n_cond).
        psi_v = psi[:, :n_val]
        psi_c = psi[:, n_val:n_val + n_cond]
        psi_v = jnp.pad(psi_v, ((0, 0), (0, nv_pad - n_val), (0, 0),
                                (0, n_rmu_pad - psi.shape[3])))
        psi_c = jnp.pad(psi_c, ((0, 0), (0, nc_pad - n_cond), (0, 0),
                                (0, n_rmu_pad - psi.shape[3])))
        eps_v = jnp.pad(enk[:, :n_val], ((0, 0), (0, nv_pad - n_val)))
        eps_c = jnp.pad(enk[:, n_val:n_val + n_cond], ((0, 0), (0, nc_pad - n_cond)))
        return (jax.lax.with_sharding_constraint(psi_v, x4),
                jax.lax.with_sharding_constraint(psi_v, y4),
                jax.lax.with_sharding_constraint(psi_c, x4),
                jax.lax.with_sharding_constraint(psi_c, y4),
                eps_v, eps_c)

    psi_v_X, psi_v_Y, psi_c_X, psi_c_Y, eps_v, eps_c = _split_pad(
        bundle.psi_rmu_Y, bundle.enk_full)
    M_X = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, psi_v_X), x4)
    M_Y = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_Y, psi_v_Y), y4)

    # ── V_Q exchange q=0 tile on the fine grid (SAME vq_interp builder) ───
    # A Q=0 exciton's exchange kernel is DENSE in (k,k') through the ONE q=0
    # tile (bse_serial.apply_bse_hamiltonian_single_device); so the fine
    # k-grid's exchange q-set is just q=0.  We eval it with the FINE mini-BZ
    # head (the head magnitude scales with the mini-BZ cell area, which shrinks
    # coarse→fine); the body is the b26p/stencil model reproduced at the q=0
    # training point (run_nulls certifies the reproduction).
    vqm = vq_interp.build_vq_evaluator(
        restart_file, mesh_xy, n_rmu_pad, head_minibz_average=True,
        log_fn=log_fn)
    gstar, head_val = vq_interp.minibz_head_vlr(
        vqm.zx, vqm.prep, np.zeros(3), kgrid=fine_grid)
    V_q0 = vqm.eval_vq(jnp.zeros(3), vqm.prep["V_SRc"], vqm.pinvF,
                       vqm.coeffs_packed,
                       jnp.asarray(head_val, dtype=jnp.float64),
                       jnp.asarray(gstar, dtype=jnp.int32))
    V_q0 = 0.5 * (V_q0 + jnp.conj(V_q0).T)     # Hermitize (stencil residue)
    V_q0 = jax.device_put(V_q0, NamedSharding(mesh_xy, P("x", "y")))
    log_fn(f"[bse_k_grid] V_q0 exchange tile via vq_interp eval_vq(Q=0), "
           f"fine mini-BZ head <v_LR>={head_val:.4f} (gstar={gstar})")

    # ── W direct: coarse W_q → ifft(R) → zero-pad R → fine → fft back ─────
    W_sh = NamedSharding(mesh_xy, P("x", "y", None, None, None))
    W_R_coarse = local_ifftn3(data["W_q"], axes=(2, 3, 4), norm="ortho")
    W_R_fine = pad_W_R_to_grid(W_R_coarse, fine_grid)
    W_q_fine = local_fftn3(W_R_fine, axes=(2, 3, 4), norm="ortho")
    W_q_fine = jax.device_put(W_q_fine, W_sh)
    log_fn(f"[bse_k_grid] W zero-padded in R "
           f"{coarse_grid[0]}x{coarse_grid[1]}x{coarse_grid[2]}→"
           f"{fine_grid[0]}x{fine_grid[1]}x{fine_grid[2]} "
           f"(exact trig-interp; direct term now on the fine grid)")

    out = dict(data)
    out.update({
        "psi_c_X": psi_c_X, "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X, "psi_v_Y": psi_v_Y,
        "M_X": M_X, "M_Y": M_Y,
        "eps_c": eps_c, "eps_v": eps_v,
        "W_q": W_q_fine, "V_q0": V_q0,
        "V_q_full": None,                       # finite-q resolvent not a fine use
        "nkx": fine_grid[0], "nky": fine_grid[1], "nkz": fine_grid[2],
    })
    return out


def load_bse_data_from_restart_sharded(
    restart_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    fermi_energy: float = 0.0,
    mesh_xy: Optional[Mesh] = None,
    pad_bands: bool = True,
    use_nohead: bool = False,
    *,
    input_file: Optional[str] = None,
    cell_volume: Optional[float] = None,
    n_occ: Optional[int] = None,
    inject_head: bool = True,
    load_v_full: bool = False,
    bse_k_grid=None,
) -> dict:
    """Load BSE tensors from canonical gw_jax restart state (psi_full_y/enk_full).

    ``bse_k_grid`` (``(nx,ny,nz)`` / ``"nx ny nz"`` / ``None``) turns on
    fine-grid densification: when set and DIFFERENT from the coarse restart
    grid, the ENTIRE bundle (ψ, QP ε, V_Q exchange, W direct) is interpolated
    onto that grid via :func:`_interpolate_bse_data_to_grid` BEFORE returning,
    so every downstream solver runs on the fine grid transparently.  When
    ``None`` the value is read from the cohsex.in ``bse_k_grid`` key (via
    ``input_file``); unset or == the coarse grid → the coarse bundle is
    returned byte-identically (fast path untouched).

    ``inject_head=False`` returns the head-LESS V_q0 / W_q bodies exactly as
    stored on disk (the rank-1 q=0 head from vhead/whead is NOT added). Used by
    body-vs-body diagnostics such as the W(0) resolvent cross-check, where both
    sides must be head-less (bse_w_exact ``--compare-w0``).

    ``load_v_full=True`` additionally reads the FULL exchange tensor
    ``V_qmunu`` at every q as ``data['V_q_full']`` (μ, ν, nkx, nky, nkz),
    P('x','y',None,None,None) — same layout as ``W_q``.  The finite-q W_q
    resolvent (``bse_w_exact --compare-wq``) picks the tile
    ``V_q_full[:, :, qx, qy, qz]`` (NO head at q≠0) as the screening V and as
    the comparison target ``W_q[...,q] - V_q_full[...,q]``.  Default False keeps
    the q=0 path byte-identical.
    """
    if mesh_xy is None:
        raise ValueError("mesh_xy is required for sharded load")

    with h5py.File(restart_file, "r") as f:
        vq_key = "V_qmunu_nohead" if use_nohead and "V_qmunu_nohead" in f else "V_qmunu"
        if use_nohead and vq_key == "V_qmunu":
            print("Warning: requested --nohead but V_qmunu_nohead not found; using V_qmunu.")
        vq_dset = f[vq_key]
        wq_key = None
        if use_nohead and "W0_qmunu_nohead" in f and bool(f["W0_qmunu_nohead"].attrs.get("W0_ready", False)):
            wq_key = "W0_qmunu_nohead"
        elif "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            wq_key = "W0_qmunu"
        if wq_key is not None:
            wq_dset = f[wq_key]
        else:
            if use_nohead:
                print("Warning: requested --nohead but W0_qmunu_nohead not found/ready.")
            print(_BARE_V_FALLBACK_WARNING)
            wq_dset = vq_dset
        if "psi_full_y" not in f or "enk_full" not in f:
            raise ValueError(
                f"{restart_file} is missing canonical psi_full_y/enk_full datasets. "
                "Regenerate restart tensors with current gw_jax."
            )
        psi_full_dset = f["psi_full_y"]
        enk_full = np.asarray(f["enk_full"][:])

        # Same axis-shape compat shim as ``_load_per_axis_padded_w_block``.
        if vq_dset.ndim == 8:
            nkx, nky, nkz = (int(s) for s in vq_dset.shape[3:6])
            n_rmu = int(vq_dset.shape[6])
            n_rnu = int(vq_dset.shape[7])
        else:
            n_rmu = int(vq_dset.shape[-2])
            n_rnu = int(vq_dset.shape[-1])
            if 'kgrid' in vq_dset.attrs:
                nkx, nky, nkz = (int(v) for v in vq_dset.attrs['kgrid'])
            elif 'kgrid' in f:
                kgrid_vals = np.asarray(f['kgrid'][:]).reshape(-1)
                nkx, nky, nkz = (int(kgrid_vals[0]), int(kgrid_vals[1]),
                                  int(kgrid_vals[2]))
            elif input_file is not None:
                from file_io import WfnLoader as WFNReader
                _wfn = WFNReader(_parse_wfn_path(input_file))
                nkx, nky, nkz = (int(_wfn.kgrid[0]), int(_wfn.kgrid[1]),
                                  int(_wfn.kgrid[2]))
            else:
                raise ValueError(
                    "load_bse_data_from_restart_sharded: flat-q V_qmunu "
                    "needs kgrid; restart file has no top-level 'kgrid' "
                    "dataset and no input_file passed to fall back to WFN.")
        if n_rmu != n_rnu:
            raise ValueError("Expected square μ/ν dimensions in V_qmunu")

        n_occ = resolve_n_occ(
            enk_full, n_occ=n_occ, input_file=input_file,
            fermi_energy=fermi_energy if fermi_energy != 0.0 else None,
        )
        nb_total = int(enk_full.shape[1])
        n_val_available = int(n_occ)
        n_cond_available = nb_total - int(n_occ)
        if n_val > n_val_available:
            print(f"Warning: requested {n_val} valence bands but only {n_val_available} available; using {n_val_available}")
        if n_cond > n_cond_available:
            print(f"Warning: requested {n_cond} conduction bands but only {n_cond_available} available; using {n_cond_available}")
        n_val = min(n_val, n_val_available)
        n_cond = min(n_cond, n_cond_available)
        if n_val == 0 or n_cond == 0:
            raise ValueError(
                f"No valence ({n_val_available}) or conduction ({n_cond_available}) bands "
                f"resolved (n_occ={n_occ}, total={nb_total})."
            )
        val_indices = np.arange(n_occ - n_val, n_occ)
        cond_indices = np.arange(n_occ, n_occ + n_cond)

        eps_v = jnp.asarray(enk_full[:, val_indices])
        eps_c = jnp.asarray(enk_full[:, cond_indices])

        _, grid_x, grid_y = _get_local_mesh_coords(
            mesh_xy, origin="bse_io.load_bse_data_from_restart_sharded")
        # Disk stores the LOGICAL μ extent; re-pad to the ONE in-memory
        # convention (mesh-product round-up, runtime.padding).
        n_rmu_pad = padded_mu_extent(n_rmu, grid_x * grid_y)
        mu_per_x = n_rmu_pad // grid_x
        nu_per_y = n_rmu_pad // grid_y

        psi_v_X = _read_psi_mu_sharded(psi_full_dset, val_indices, mu_per_x, "x", mesh_xy, n_rmu_pad, trim=False)
        psi_c_X = _read_psi_mu_sharded(psi_full_dset, cond_indices, mu_per_x, "x", mesh_xy, n_rmu_pad, trim=False)

        if pad_bands:
            psi_v_X, n_val_pad = _pad_axis_to_multiple(psi_v_X, axis=1, multiple=grid_y)
            psi_c_X, n_cond_pad = _pad_axis_to_multiple(psi_c_X, axis=1, multiple=grid_x)
            eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=grid_y)
            eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=grid_x)
        else:
            n_val_pad = int(psi_v_X.shape[1])
            n_cond_pad = int(psi_c_X.shape[1])
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v_X, NamedSharding(mesh_xy, P(None, None, None, "y")))
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c_X, NamedSharding(mesh_xy, P(None, None, None, "y")))

        V_q0 = _read_vq0_sharded(vq_dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad,
                                 trim=False, kgrid=(nkx, nky, nkz))
        W_q = _read_wq_sharded(wq_dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad,
                               trim=False, kgrid=(nkx, nky, nkz))

        # Full-q exchange tensor for the finite-q W_q resolvent: read every V
        # tile with the SAME (μ, ν, nkx, nky, nkz) reader as W_q (no head; head
        # is a q=0-only rank-1 piece).  V_q_full[:, :, 0, 0, 0] == V_q0 (the
        # head-less q=0 read) — a self-check the finite-q harness asserts.
        V_q_full = (_read_wq_sharded(vq_dset, mu_per_x, nu_per_y, mesh_xy,
                                     n_rmu_pad, trim=False, kgrid=(nkx, nky, nkz))
                    if load_v_full else None)

        # ── q=0 head: load G0_mu_nu, dual-shard X/Y, inject as rank-1 ────
        # On the (μ,ν)-sharded V_q0 and W_q tensors, the rank-1 update
        # ``conj(g0_X[μ_loc]) * g0_Y[ν_loc]`` is local on every proc when
        # g0 is held in TWO copies — one P("x") on μ, one P("y") on ν.
        # Source priority: cohsex.in overrides → restart vhead/whead.
        if "G0_mu_nu" in f:
            G0_full = np.asarray(f["G0_mu_nu"][:], dtype=np.complex128)
            if G0_full.size < n_rmu_pad:
                G0_pad = np.zeros((n_rmu_pad,), dtype=np.complex128)
                G0_pad[:G0_full.size] = G0_full
                G0_full = G0_pad
            # Process-local (AA.1): G0_full is host numpy read identically
            # on every rank; plain device_put would fire the hidden
            # assert_equal all-gather (twice).  LORRAX_CHECK_REPLICA=1
            # re-arms it.
            from common.collectives import device_put_process_local
            g0_X = device_put_process_local(G0_full,
                                            NamedSharding(mesh_xy, P("x")))
            g0_Y = device_put_process_local(G0_full,
                                            NamedSharding(mesh_xy, P("y")))
            vhead_restart = (complex(f["vhead"][()])
                             if "vhead" in f else None)
            whead_restart = (jnp.asarray(f["whead"][:], dtype=jnp.complex128)
                             if "whead" in f else None)
        else:
            g0_X = g0_Y = None
            vhead_restart = whead_restart = None

    if g0_X is not None and inject_head:
        vhead, whead, cell_volume = _resolve_head_params(
            input_file, vhead_restart, whead_restart, cell_volume)

        if cell_volume is not None and (vhead is not None or whead is not None):
            from gw.head_correction import apply_q0_head_rank1_sharded
            V_q0, W_q = apply_q0_head_rank1_sharded(
                V_q0, W_q, g0_X, g0_Y, vhead, whead, cell_volume,
                omega_index=0)
            v_str = (f"vhead={complex(vhead).real:.3f}"
                     if vhead is not None else "vhead=skipped")
            w_str = (f"whead[0]={complex(whead[0]).real:.3f}"
                     if whead is not None else "whead=skipped")
            print(f"BSE-sharded: q=0 head injected (rank-1, dual-sharded G0, "
                  f"V_cell={cell_volume:.2f}): {v_str}, {w_str}")
        else:
            # §16.5 latent-bug guard: G0_mu_nu is present and inject_head is
            # True, but the rank-1 head was NOT injected because vhead/whead
            # both resolve to None (no cohsex.in override, no restart scalars)
            # and/or cell_volume is unknown.  Previously silent → a head-LESS
            # q=0 exchange tile with no trace.  Recompute is NON-TRIVIAL here
            # (the loader has no wfn/meta/sym/S_cart to rebuild <v>_mBZ), so
            # warn loudly and name the fix (add vhead/whead to the restart or
            # a cohsex.in override).
            import warnings
            reasons = []
            if cell_volume is None:
                reasons.append("cell_volume unknown (WFN not passed)")
            if vhead is None and whead is None:
                reasons.append("vhead/whead both unresolved "
                               "(no cohsex.in override, no restart scalars)")
            msg = (
                "BSE q=0 head NOT injected though G0_mu_nu is present and "
                f"inject_head=True: {', '.join(reasons)}.  The q=0 exchange "
                "tile is HEAD-LESS (missing the rank-1 (vhead/V_cell)·conj(g0)g0 "
                "term) — exciton binding energies will be under-bound at the "
                "zone centre.  FIX: add ``vhead``/``whead_0freq`` to cohsex.in, "
                "or write ``vhead``/``whead`` datasets into the restart.  "
                "(Recompute-from-WFN is not available at the loader.)")
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            print(f"BSE-sharded: [WARN] {msg}")

    # Hoisted exchange pair amplitudes M(k,c,v,μ) = Σ_s conj(ψ_c) ψ_v — the V-term
    # decode (M_X, μ on x) and encode (M_Y, ν on y) vertices. Precomputed ONCE here
    # (audit P3) so the per-iteration BSE matvec receives them as inputs instead of
    # rebuilding them from ψ every call. Peak-neutral; the between-matvec floor rises
    # by ~2·M/p (reports/bse_refactor_map_2026-07-15/archive/matvec_efficiency_audit).
    M_X = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, psi_v_X),
        NamedSharding(mesh_xy, P(None, None, None, "x")))
    M_Y = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_Y, psi_v_Y),
        NamedSharding(mesh_xy, P(None, None, None, "y")))

    data = {
        "psi_c_X": psi_c_X,
        "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X,
        "psi_v_Y": psi_v_Y,
        "M_X": M_X,
        "M_Y": M_Y,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
        "V_q_full": V_q_full,
        "g0_X": g0_X,
        "g0_Y": g0_Y,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "n_rmu": n_rmu,
        "n_rmu_pad": n_rmu_pad,
        "n_val": n_val,
        "n_cond": n_cond,
        "n_val_pad": n_val_pad,
        "n_cond_pad": n_cond_pad,
        "fermi_energy": fermi_energy,
    }

    # ── bse_k_grid coarse→fine densification (general BSE init) ───────────
    # Explicit arg wins; else read the ``bse_k_grid`` key from the cohsex.in.
    # None/unset or == the coarse grid → return the coarse bundle above
    # UNTOUCHED (fast path, byte-identical — the on-grid identity guarantee).
    fine_grid = _resolve_bse_k_grid(bse_k_grid, input_file)
    if fine_grid is not None and fine_grid != (nkx, nky, nkz):
        if input_file is None:
            raise ValueError(
                "bse_k_grid densification needs input_file (cohsex.in) to run "
                "the htransform ψ/ε and vq_interp V_Q interpolation.")
        data = _interpolate_bse_data_to_grid(
            data, fine_grid, restart_file, input_file, mesh_xy)
    return data


def read_bgw_eqp(eqp_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a BerkeleyGW ``eqp1.dat`` file."""

    kpts = []
    e_dft_blocks = []
    e_qp_blocks = []

    with open(eqp_file) as f:
        while True:
            header = f.readline()
            if not header:
                break
            stripped = header.strip()
            if not stripped:
                break
            if stripped.startswith("#"):
                continue
            parts = header.split()
            if len(parts) < 4:
                break
            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            n_bands = int(parts[3])
            kpts.append([kx, ky, kz])

            e_dft_k = []
            e_qp_k = []
            for _ in range(n_bands):
                cols = f.readline().split()
                e_dft_k.append(float(cols[2]))
                e_qp_k.append(float(cols[3]))
            e_dft_blocks.append(e_dft_k)
            e_qp_blocks.append(e_qp_k)

    kpts_ibz = np.array(kpts)
    max_band = max(len(b) for b in e_dft_blocks)
    n_kpts = len(kpts)
    e_dft_ibz = np.full((n_kpts, max_band), np.nan)
    e_qp_ibz = np.full((n_kpts, max_band), np.nan)
    for i in range(n_kpts):
        nb = len(e_dft_blocks[i])
        e_dft_ibz[i, :nb] = e_dft_blocks[i]
        e_qp_ibz[i, :nb] = e_qp_blocks[i]
    return kpts_ibz, e_dft_ibz, e_qp_ibz


def _parse_wfn_path(input_file: str) -> str:
    """Extract ``wfn_file`` from ``cohsex.in`` and resolve relative paths."""

    input_dir = os.path.dirname(os.path.abspath(input_file))
    wfn_file = "WFN.h5"
    with open(input_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            if key.strip() == "wfn_file":
                wfn_file = val.strip()
                break
    if not os.path.isabs(wfn_file):
        wfn_file = os.path.join(input_dir, wfn_file)
    return wfn_file


def resolve_n_occ(
    enk_full: np.ndarray,
    *,
    n_occ: Optional[int] = None,
    input_file: Optional[str] = None,
    fermi_energy: Optional[float] = None,
) -> int:
    """Determine n_occ (count of occupied bands) for BSE band slicing.

    Resolution order:

      1. **Explicit ``n_occ``** — caller knows; return as-is.
      2. **WFN.h5 ``ifmax``** via ``input_file`` (cohsex.in's ``wfn_file``
         entry). Reads ``mf_header/kpoints/ifmax`` directly — authoritative.
      3. **``mean_enk < fermi_energy``** if ``fermi_energy`` is explicitly
         passed (Ry). Caller's responsibility to pass a sane reference.

    Raises ``ValueError`` if none of the above resolves. The previous
    "auto-detect" heuristic (``mean_enk < 0`` or "largest gap") was
    silently broken for systems whose pseudopotential reference puts the
    valence well above zero (most QE setups, e.g. Si): it returned only
    the deepest semicore states. We now require an explicit source.

    Parameters
    ----------
    enk_full : (nk, nb) ndarray (Ry) — DFT eigenvalues per k. Used only
        when ``fermi_energy`` is given as a hint.
    n_occ : int, optional — explicit bypass.
    input_file : str, optional — cohsex.in / nscf.in path. Its
        ``wfn_file`` entry is followed to a WFN.h5 to read ``ifmax``.
    fermi_energy : float, optional (Ry) — explicit Fermi-level hint.
    """
    if n_occ is not None:
        return int(n_occ)

    if input_file is not None:
        try:
            from file_io import WfnLoader as WFNReader
            wfn_path = _parse_wfn_path(input_file)
            if os.path.exists(wfn_path):
                w = WFNReader(wfn_path)
                return int(w.nelec)
            else:
                print(f"  [resolve_n_occ] WFN.h5 not found at {wfn_path}; "
                      "trying fermi_energy hint next.")
        except Exception as e:
            print(f"  [resolve_n_occ] WFN.h5 lookup failed "
                  f"({type(e).__name__}: {e}); trying fermi_energy hint next.")

    if fermi_energy is not None:
        mean_enk = np.asarray(np.mean(enk_full, axis=0))
        nb = mean_enk.size
        n_occ_hint = int(np.sum(mean_enk < fermi_energy))
        if 1 <= n_occ_hint <= nb - 1:
            return n_occ_hint

    raise ValueError(
        "Could not determine n_occ. Pass `n_occ=` explicitly, or "
        "`input_file=` pointing to a cohsex.in / nscf.in whose `wfn_file` "
        "resolves to a valid WFN.h5 (where `mf_header/kpoints/ifmax` "
        "gives the count of occupied bands authoritatively)."
    )


def _parse_head_overrides(input_file: Optional[str]):
    """Extract ``vhead`` and ``whead_0freq`` overrides from cohsex.in.

    Returns ``(vhead, whead_0freq)`` where each is ``complex`` or ``None``
    if the key is absent / blank. These take precedence over any
    restart-file head values when assembling the q=0 rank-1 update.
    """
    if input_file is None or not os.path.isfile(input_file):
        return None, None
    vhead = None
    whead0 = None
    with open(input_file) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            key = key.strip().lower()
            val = val.strip()
            if not val:
                continue
            try:
                if key == "vhead":
                    vhead = complex(float(val))
                elif key == "whead_0freq":
                    whead0 = complex(float(val))
            except ValueError:
                continue
    return vhead, whead0


def _resolve_head_params(
    input_file: Optional[str],
    vhead_restart,
    whead_restart,
    cell_volume: Optional[float] = None,
):
    """Resolve ``(vhead, whead, cell_volume)`` for q=0 head injection.

    cohsex.in ``vhead``/``whead_0freq`` overrides take precedence over the
    restart-file head values; ``cell_volume`` (Bohr³) is pulled from the WFN
    when not supplied.  Any of the three may return ``None`` (head skipped).
    Single-sourced by both the sharded and single-device restart loaders.
    """
    vhead_in, whead0_in = _parse_head_overrides(input_file)
    vhead = vhead_in if vhead_in is not None else vhead_restart
    if whead0_in is not None:
        whead = jnp.asarray([whead0_in], dtype=jnp.complex128)
    else:
        whead = whead_restart
    if cell_volume is None and input_file is not None:
        try:
            from file_io import WfnLoader as WFNReader
            cell_volume = float(WFNReader(_parse_wfn_path(input_file)).cell_volume)
        except Exception as exc:
            print(f"BSE head: cell_volume unresolved ({exc}); skipping head")
            cell_volume = None
    return vhead, whead, cell_volume


def apply_eqp_corrections(
    enk_full: np.ndarray,
    eqp_file: str,
    input_file: Optional[str] = None,
    ry_to_ev: float = 13.6056980659,
) -> np.ndarray:
    """Apply BGW ``eqp1.dat`` corrections to full-BZ DFT eigenvalues."""

    _kpts_ibz, e_dft_ibz, e_qp_ibz = read_bgw_eqp(eqp_file)
    nk_ibz, nb_eqp = e_dft_ibz.shape
    nk_full, nb_full = enk_full.shape
    enk_qp = enk_full.copy()

    if input_file is not None:
        from file_io import WfnLoader as WFNReader
        from common.symmetry_maps import SymMaps

        wfn_path = _parse_wfn_path(input_file)
        wfn = WFNReader(wfn_path)
        sym = SymMaps(wfn)
        assert sym.nk_tot == nk_full
        assert nk_ibz == sym.nk_red

        for ik_full in range(nk_full):
            ik_ibz = sym.irr_idx_k[ik_full]
            for ib in range(min(nb_eqp, nb_full)):
                if not np.isnan(e_qp_ibz[ik_ibz, ib]):
                    enk_qp[ik_full, ib] = e_qp_ibz[ik_ibz, ib] / ry_to_ev
    else:
        enk_full_ev = enk_full * ry_to_ev
        tol_ev = 0.01
        matched = np.zeros(nk_full, dtype=bool)
        for ik_full in range(nk_full):
            best_ibz = -1
            best_err = np.inf
            for ik_ibz in range(nk_ibz):
                n_compare = min(nb_eqp, nb_full)
                mask = ~np.isnan(e_dft_ibz[ik_ibz, :n_compare])
                if not np.any(mask):
                    continue
                err = np.max(
                    np.abs(
                        enk_full_ev[ik_full, :n_compare][mask]
                        - e_dft_ibz[ik_ibz, :n_compare][mask]
                    )
                )
                if err < best_err:
                    best_err = err
                    best_ibz = ik_ibz
            if best_ibz >= 0 and best_err < tol_ev:
                matched[ik_full] = True
                for ib in range(min(nb_eqp, nb_full)):
                    if not np.isnan(e_qp_ibz[best_ibz, ib]):
                        enk_qp[ik_full, ib] = e_qp_ibz[best_ibz, ib] / ry_to_ev

    return enk_qp


def apply_eqp_and_reslice_bands(
    restart_file: str,
    eqp_file: str,
    input_file: Optional[str],
    n_val: int,
    n_cond: int,
    n_occ: Optional[int],
    grid_x: int,
    grid_y: int,
) -> tuple[jax.Array, jax.Array, int]:
    """Apply BGW ``eqp1.dat`` corrections and re-slice the BSE band window.

    Reads ``enk_full`` from ``restart_file``, applies the eqp corrections, then
    RE-resolves n_occ on the CORRECTED energies (QP shifts can move the gap)
    before slicing ``n_val``/``n_cond`` bands around the new n_occ.  The
    valence/conduction energy axes are padded to multiples of (grid_y, grid_x)
    to match the loader's psi band axes.

    ``n_val``/``n_cond`` must be the loader-CLAMPED counts (``data['n_val']`` /
    ``data['n_cond']``), not raw CLI requests, or the slice can run out of
    bounds.  Single-sourced by the sharded --eqp paths (bse_jax._preview_lanczos,
    davidson_absorption, absorption_haydock).

    Returns ``(eps_v_padded, eps_c_padded, n_occ_eff)``.
    """
    with h5py.File(restart_file, "r") as f:
        enk_full_np = np.asarray(f["enk_full"][:])
    enk_full_np = apply_eqp_corrections(enk_full_np, eqp_file, input_file=input_file)
    n_occ_eff = resolve_n_occ(enk_full_np, n_occ=n_occ, input_file=input_file)
    val_idx = np.arange(n_occ_eff - n_val, n_occ_eff)
    cond_idx = np.arange(n_occ_eff, n_occ_eff + n_cond)
    eps_v = jnp.asarray(enk_full_np[:, val_idx])
    eps_c = jnp.asarray(enk_full_np[:, cond_idx])
    eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=grid_y)
    eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=grid_x)
    return eps_v, eps_c, n_occ_eff


def _find_restart_file(input_file: str) -> str:
    """Locate ``isdf_tensors_<n_rmu>.h5`` — loudly when the choice is ambiguous.

    ``isdf_tensors_*.h5`` is namespaced by centroid count, so a run
    directory that has been used for a μ-sweep holds several.  This
    returned the first *lexicographically* sorted match, which is not the
    newest and not necessarily the one the GW run that produced this
    BSE's inputs wrote: ``isdf_tensors_1194.h5`` sorts before
    ``isdf_tensors_276.h5``.  Picking the wrong one is silent — the BSE
    kernel is built from a different ISDF basis than Σ was, every stage
    reports success, and the exciton spectrum is quietly wrong.  The
    ambiguity is now named in the log, and the *newest* file wins (the
    one the most recent GW run wrote), which is the intent everywhere
    this is called from.
    """
    input_dir = os.path.dirname(os.path.abspath(input_file))
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "tmp", "isdf_tensors_*.h5"))))
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "isdf_tensors_*.h5"))))
    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates:
        raise FileNotFoundError(f"Could not find canonical restart file isdf_tensors_*.h5 in {input_dir}")
    if len(candidates) > 1:
        # Newest wins; say so, and list what was passed over.
        chosen = max(candidates, key=os.path.getmtime)
        others = ", ".join(os.path.basename(p) for p in candidates
                           if p != chosen)
        print(
            f"  *** LORRAX SANITY: {len(candidates)} restart tensor files "
            f"match isdf_tensors_*.h5 in {input_dir} — they hold DIFFERENT "
            f"ISDF bases (different centroid counts).  Using the newest, "
            f"{os.path.basename(chosen)}; ignoring: {others}.  If that is "
            f"not the basis this BSE's eqp/Σ inputs were built with, the "
            f"exciton spectrum will be wrong with no other symptom — pass "
            f"the file explicitly or clean the run directory. ***",
            flush=True)
        return chosen
    return candidates[0]


def _load_ring_subset(
    restart_file: str,
    n_val: int,
    n_cond: int,
    px: int,
    py: int,
    eqp_file: Optional[str] = None,
    n_occ: Optional[int] = None,
    input_file: Optional[str] = None,
) -> dict:
    """Load a single-device BSE subset from canonical gw_jax restart state."""
    with h5py.File(restart_file, "r") as f:
        V_qmunu = jnp.asarray(f["V_qmunu"][:])
        if "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            W0_qmunu = jnp.asarray(f["W0_qmunu"][:])
        else:
            W0_qmunu = None
        # G0_mu_nu = ζ(q=0, μ, G=0) — rank-1 head projector. Persisted by the
        # GW writer; consumed below by the q=0 head-injection step.
        G0_mu_nu = jnp.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None
        # Restart-side scalar head fields (Phase B writer; may not exist yet).
        vhead_restart = complex(f["vhead"][()]) if "vhead" in f else None
        whead_restart = (jnp.asarray(f["whead"][:], dtype=jnp.complex128)
                         if "whead" in f else None)
        if "psi_full_y" not in f or "enk_full" not in f:
            raise ValueError(
                f"{restart_file} is missing canonical psi_full_y/enk_full datasets. "
                "Regenerate restart tensors with current gw_jax."
            )
        psi_full = jnp.asarray(f["psi_full_y"][:])
        enk_full_np = np.asarray(f["enk_full"][:])

    if eqp_file is not None:
        enk_full_np = apply_eqp_corrections(enk_full_np, eqp_file, input_file=input_file)

    enk_full = jnp.asarray(enk_full_np)

    # V_qmunu axis-shape compatibility shim:
    #   * legacy 8-D ``(1, npol, npol, nkx, nky, nkz, μ, μ)`` — read kgrid
    #     directly from the shape;
    #   * legacy 6-D ``(1, npol, npol, nq, μ, μ)`` — strip leading axes
    #     and read kgrid from the WFN;
    #   * new flat-q 3-D ``(nq, μ, μ)`` — read kgrid from the WFN.
    # Internal BSE machinery still wants ``W_q`` shaped as
    # ``(μ, μ, nkx, nky, nkz)``; we reshape after normalising V_qmunu.
    if V_qmunu.ndim == 8:
        nkx, nky, nkz = V_qmunu.shape[3:6]
        n_rmu = int(V_qmunu.shape[-1])
        V_qmunu_flat = jnp.asarray(V_qmunu)[0, 0, 0].reshape(
            -1, n_rmu, n_rmu)
        if W0_qmunu is not None:
            W0_qmunu_flat = jnp.asarray(W0_qmunu)[0, 0, 0].reshape(
                -1, n_rmu, n_rmu)
        else:
            W0_qmunu_flat = None
    else:
        if input_file is None:
            raise ValueError(
                "BSE: flat-q V_qmunu requires input_file to resolve "
                "kgrid from the WFN")
        from file_io import WfnLoader as WFNReader
        _wfn = WFNReader(_parse_wfn_path(input_file))
        nkx, nky, nkz = (int(_wfn.kgrid[0]), int(_wfn.kgrid[1]), int(_wfn.kgrid[2]))
        n_rmu = int(V_qmunu.shape[-1])
        if V_qmunu.ndim == 6:
            V_qmunu_flat = jnp.asarray(V_qmunu)[0, 0, 0]
            W0_qmunu_flat = (jnp.asarray(W0_qmunu)[0, 0, 0]
                              if W0_qmunu is not None else None)
        else:
            V_qmunu_flat = jnp.asarray(V_qmunu)  # already (nq, μ, μ)
            W0_qmunu_flat = (jnp.asarray(W0_qmunu)
                              if W0_qmunu is not None else None)
    V_qmunu = V_qmunu_flat
    W0_qmunu = W0_qmunu_flat
    nk = nkx * nky * nkz
    # Same μ re-pad convention as the sharded loader above.
    n_rmu_pad = padded_mu_extent(n_rmu, px * py)

    n_bands_total = enk_full.shape[1]
    n_occ = resolve_n_occ(enk_full_np, n_occ=n_occ, input_file=input_file)
    n_val_available = int(n_occ)
    n_cond_available = n_bands_total - n_occ
    if n_val > n_val_available:
        print(f"Warning: requested {n_val} valence bands but only {n_val_available} available; using {n_val_available}")
    if n_cond > n_cond_available:
        print(f"Warning: requested {n_cond} conduction bands but only {n_cond_available} available; using {n_cond_available}")
    n_val = min(n_val, n_val_available)
    n_cond = min(n_cond, n_cond_available)
    val_indices = jnp.arange(n_occ - n_val, n_occ)
    cond_indices = jnp.arange(n_occ, n_occ + n_cond)

    psi_v = psi_full[:, val_indices, :, :]
    psi_c = psi_full[:, cond_indices, :, :]
    eps_v = enk_full[:, val_indices]
    eps_c = enk_full[:, cond_indices]

    psi_v = _pad_last_axis(psi_v, n_rmu_pad)
    psi_c = _pad_last_axis(psi_c, n_rmu_pad)
    psi_v, n_val_pad = _pad_axis_to_multiple(psi_v, axis=1, multiple=py)
    psi_c, n_cond_pad = _pad_axis_to_multiple(psi_c, axis=1, multiple=px)
    eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=py)
    eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=px)
    # V_qmunu is now flat-q (nq, μ, μ) post-shim; q=0 is V_qmunu[0].
    V_q0 = V_qmunu[0]
    V_q0 = _pad_last_two_axes(V_q0, n_rmu_pad)
    if W0_qmunu is not None:
        W_src = W0_qmunu
    else:
        print(_BARE_V_FALLBACK_WARNING)
        W_src = V_qmunu
    # W_src: (nq, μ, μ).  Reshape flat-q → 3-D-k and transpose to the
    # ``(μ, μ, nkx, nky, nkz)`` layout the downstream BSE machinery
    # consumes.  This is the ONE place the 3-D-k form materialises
    # inside BSE; elsewhere we keep flat-q.
    W_q = W_src.reshape(nkx, nky, nkz, n_rmu, n_rmu).transpose(3, 4, 0, 1, 2)
    W_q = _pad_first_two_axes(W_q, n_rmu_pad)

    # ── q=0 head injection (rank-1 in (μ,ν) ISDF basis) ──────────────────
    # Runs AFTER the layout shim so it operates on the normalized q=0 slice:
    # V_q0 (μ,μ) and the (0,0,0) k-slice of W_q.  compute_vcoul zeroes the
    # G=G'=0 element of v(q=0); we reinstate the mini-BZ-averaged head as a
    # rank-1 update from G0_mu_nu = ζ(0,μ,G=0).  Single device → feed G0 as
    # both the μ- and ν-axis copy to the sharded rank-1 helper.  whead is
    # injected into W_q only when a real screened W0 is present (not the
    # bare-V fallback).  Source priority: cohsex.in overrides > restart-file.
    if G0_mu_nu is not None:
        vhead, whead, cell_volume = _resolve_head_params(
            input_file, vhead_restart, whead_restart)
        if cell_volume is None:
            print("BSE: head injection skipped — could not resolve cell_volume "
                  "(input_file required)")
        elif vhead is not None or whead is not None:
            from gw.head_correction import apply_q0_head_rank1_sharded
            g0_pad = _pad_last_axis(G0_mu_nu, n_rmu_pad)
            w0_ready = W0_qmunu is not None
            V_q0, W_head = apply_q0_head_rank1_sharded(
                V_q0, W_q if w0_ready else None, g0_pad, g0_pad,
                vhead, whead, cell_volume, omega_index=0)
            if w0_ready:
                W_q = W_head
            v_str = (f"vhead={complex(vhead).real:.3f}"
                     if vhead is not None else "vhead=skipped")
            w_str = (f"whead[0]={complex(whead[0]).real:.3f}"
                     if (whead is not None and w0_ready) else "whead=skipped")
            print(f"BSE: q=0 head injected (rank-1 in μν, V_cell={cell_volume:.2f}): "
                  f"{v_str}, {w_str}")

    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (1, n_cond_pad, n_val_pad, nk)) + 1j * jax.random.normal(
        key, (1, n_cond_pad, n_val_pad, nk)
    )

    return {
        "psi_c": psi_c,
        "psi_v": psi_v,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
        "X": X,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "nk": nk,
        "n_rmu_pad": n_rmu_pad,
    }
