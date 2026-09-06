"""G1.1 — synth bit-identity test for the Round-6 streaming-scan
z_q_from_psi_sm.

Round 6 / `round5_unified_plan.md` §5.4 G1: validate the rewritten
``z_q_from_psi_sm`` (commit ``f567aa0``: scan-inside-shard_map +
io_callback + all_gather) computes the same Z_q as a reference
implementation that builds ``psi_l_Y`` / ``psi_r_Y`` by direct
IFFT-and-slice on the full host tile, then does the global einsum.

Tolerance: ``rtol=1e-10, atol=1e-12`` per plan §5.2 (NOT bit-equal —
band-axis summation order differs; ULP-class drift expected).

Six sub-gates per the validation checklist in
``round6_discussion.md``:

* charge channel, single rchunk, multi-bc, no pseudobands
* G1.1a — single bc
* G1.1b — short final bc (last bc shorter than bpd_max)
* G1.1c — asymmetric L/R windows
* G1.1d — bispinor γ̃ ≠ I (vertex_mu_L = 1)
* G1.1e — pseudobands (norms != 1, pre-divided into psi_l_X / psi_r_X)
"""
from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from isdf import z_q_from_psi_sm
from common.gamma_matrices import gamma_double_contract
from common.wfn_transforms import to_rchunk


MESH = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
             axis_names=('x', 'y'))


# ---------------------------------------------------------------------------
# Minimal mock PsiGStore — only the surface area z_q_from_psi_sm touches.
# ---------------------------------------------------------------------------

class _MockPsiGStore:
    """Stand-in for :class:`PsiGStore` on a 1x1 mesh.

    Holds a single host tile ``(nk, nb_total, ns, ngkmax)`` and exposes
    the four attributes the new ``z_q_from_psi_sm`` reads:

    - ``_slice_local_tile_bc(x_idx, y_idx, bc_idx)``: per-iter slicer
      called via ``io_callback`` inside the scan body.  Mirrors
      :meth:`PsiGStore._slice_local_tile_bc` (psi_G_store.py:272-309).
    - ``_bpd_max``: padded uniform per-bc local band count (closed into
      ``ShapeDtypeStruct`` for the io_callback return spec).
    - ``_per_rank_shape``: ``(nk, nb_total, ns, ngkmax)`` — only the
      trailing ngkmax is read by ``z_q_from_psi_sm``.
    - ``g_index``: box-index tensor used for the IFFT scatter
      (``to_rchunk_inner`` consumer).
    - ``kvecs_frac``: per-k fractional kvecs for the Bloch phase.

    On a 1x1 mesh there's only one rank, so ``_host_tiles[(0, 0)]`` is
    the full tile (band sharding is a no-op).
    """

    def __init__(self, psi_G, g_index, kvecs_frac, band_chunk_ranges,
                 fft_grid=None):
        nk, nb_total, ns, ngkmax = psi_G.shape
        # Minimal ``meta`` stub — z_q_from_psi_sm reads ``store.meta.fft_grid``.
        class _MetaStub:
            pass
        self.meta = _MetaStub()
        self.meta.fft_grid = (tuple(int(s) for s in fft_grid)
                              if fft_grid is not None else (4, 4, 4))
        self.meta.nk_tot = int(nk)
        self.meta.nspinor = int(ns)
        self._host_tiles = {(0, 0): np.asarray(psi_G, dtype=np.complex128)}
        self.band_chunk_ranges = tuple(
            (int(lo), int(hi)) for (lo, hi) in band_chunk_ranges)
        offsets = [0]
        for (lo, hi) in self.band_chunk_ranges:
            offsets.append(offsets[-1] + (hi - lo))
        self._bc_band_offsets = tuple(offsets)
        bpd_per_bc = [
            (hi - lo) for (lo, hi) in self.band_chunk_ranges
        ]                                                  # p_xy=1 ⇒ no divide
        # Mirrors ``PsiGStore._bpd_per_bc`` (psi_G_store.py:147): per-bc
        # LOCAL band count, (hi-lo)//p — p_xy=1 here so no divide.  Read
        # by ``isdf.core``'s bc-compaction reorder (core.py:534).
        self._bpd_per_bc = tuple(bpd_per_bc)
        self._bpd_max = max(bpd_per_bc) if bpd_per_bc else 0
        self._per_rank_shape = (nk, nb_total, ns, ngkmax)
        self._g_index_dev = jax.device_put(
            np.asarray(g_index, dtype=np.int32),
            NamedSharding(MESH, P(*([None] * 4))))
        self._kvecs_frac_dev = jax.device_put(
            np.asarray(kvecs_frac, dtype=np.float64),
            NamedSharding(MESH, P(None, None)))

    def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx):
        x, y, bc = int(x_idx), int(y_idx), int(bc_idx)
        tile = self._host_tiles[(x, y)]
        b_lo = self._bc_band_offsets[bc]
        b_hi = self._bc_band_offsets[bc + 1]
        nk, _, ns, ngkmax = tile.shape
        out = np.zeros(
            (nk, self._bpd_max, ns, ngkmax), dtype=tile.dtype)
        out[:, : b_hi - b_lo, :, :] = tile[:, b_lo:b_hi, :, :]
        return out

    @property
    def local_band_chunk_shape(self):
        # Mirrors PsiGStore.local_band_chunk_shape: the per-device host
        # callback shape (nk, b_local, ns, ngkmax) that the r-chunk sources
        # read instead of the private _bpd_max/_per_rank_shape.
        nk, _, ns, ngkmax = self._per_rank_shape
        return (int(nk), int(self._bpd_max), int(ns), int(ngkmax))

    @property
    def g_index(self):
        return self._g_index_dev

    @property
    def kvecs_frac(self):
        return self._kvecs_frac_dev


# ---------------------------------------------------------------------------
# Reference Z_q via global einsum on pre-built psi_l_Y / psi_r_Y
# ---------------------------------------------------------------------------

def _reference_z_q(
    *,
    psi_l_X,                # (nk, n_rmu, nb_l, ns) — band-window L slice
    psi_r_X,                # (nk, n_rmu, nb_r, ns)
    psi_G,                  # (nk, nb_total, ns, ngkmax) full G-flat
    g_index,                # (nk, nx, ny, nz)
    kvecs_frac,             # (nk, 3)
    band_range_left,        # (L_lo, L_hi) global
    band_range_right,
    fft_grid,
    r_start,                # int
    r_chunk_size,           # int
    kgrid,                  # (nkx, nky, nkz)
    gamma_L=None,
    gamma_R=None,
):
    """Reference Z_q: build psi_l_Y / psi_r_Y by direct IFFT + slice
    on the full G-flat tile (no scan), do the global einsum + post-pair
    pipeline."""
    L_lo, L_hi = int(band_range_left[0]), int(band_range_left[1])
    R_lo, R_hi = int(band_range_right[0]), int(band_range_right[1])

    # psi_Y_full: (nk, nb_total, ns, n_zchunk) — direct call into the
    # production to_rchunk on the global G-flat (no chunking; the
    # reference does it in one shot).
    psi_Y_full = to_rchunk(
        jnp.asarray(psi_G),
        g_index,
        fft_grid,
        int(r_start),
        int(r_chunk_size),
        mesh=MESH,
        norm="ortho",
        kvecs_frac=jnp.asarray(kvecs_frac, dtype=jnp.float64),
    )

    psi_l_Y_ref = psi_Y_full[:, L_lo:L_hi, :, :]
    psi_r_Y_ref = psi_Y_full[:, R_lo:R_hi, :, :]

    # Pair-density einsums — same layout as the new body's per-iter
    # delta_P_l / delta_P_r, but over the FULL band axis at once.
    P_l = jnp.einsum(
        'kmna,knbr->karmb', psi_l_X, psi_l_Y_ref, optimize=True)
    P_r = jnp.einsum(
        'kmna,knbr->karmb', psi_r_X, psi_r_Y_ref, optimize=True)

    nkx, nky, nkz = kgrid
    nk = int(psi_l_X.shape[0])
    n_rmu = int(psi_l_X.shape[1])
    ns = int(psi_l_X.shape[3])
    n_zchunk = int(r_chunk_size)

    P_l_3d = P_l.reshape(nkx, nky, nkz, ns, n_zchunk, n_rmu, ns)
    P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0, 1, 2), norm='forward')
    P_l_R_conj = jnp.conj(P_l_R)

    P_r_3d = P_r.reshape(nkx, nky, nkz, ns, n_zchunk, n_rmu, ns)
    P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0, 1, 2), norm='forward')

    if gamma_L is None:
        perm_L = None
        phase_L = None
    else:
        perm_L, phase_L = gamma_L
    if gamma_R is None:
        perm_R = None
        phase_R = None
    else:
        perm_R, phase_R = gamma_R

    Z_R = gamma_double_contract(
        P_l_R_conj, P_r_R,
        perm_L=perm_L, phase_L=phase_L,
        perm_R=perm_R, phase_R=phase_R,
        spin_axes=(3, 6),
    )
    Z_q_3d = jnp.fft.fftn(Z_R, axes=(0, 1, 2), norm='forward')
    Z_q = jnp.transpose(
        Z_q_3d.reshape(nkx * nky * nkz, n_zchunk, n_rmu),
        (0, 2, 1))
    return Z_q


# ---------------------------------------------------------------------------
# Synth-WFN fixture helpers
# ---------------------------------------------------------------------------

def _build_synth(
    *,
    kgrid=(2, 2, 1),                # nk=4
    nb_total=8,
    ns=2,
    fft_grid=(4, 4, 4),
    n_rmu=4,
    n_zchunk=8,
    seed=0xABCDEF01,
):
    """Build self-consistent synth (psi_G, g_index, kvecs, psi_l_X-shape
    arrays) on a 1x1 mesh.

    psi_l_X / psi_r_X are random per-band-window slabs; psi_G is the
    same band axis filled with random complex numbers in valid G-sphere
    positions.  g_index uses the `to_rchunk` sentinel-zero contract.
    """
    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    nx, ny, nz = fft_grid
    rng = np.random.default_rng(seed)

    # ngkmax: number of G-vectors per k.  Use ~half of n_rtot (typical
    # cutoff ratio) so the box has both valid and empty cells.
    n_rtot = nx * ny * nz
    ngkmax = min(n_rtot // 2 + 1, n_rtot)

    # g_index: (nk, nx, ny, nz) int32.  Cells in the sphere map to
    # valid G-positions [0, ngkmax); cells outside map to the sentinel
    # `ngkmax` (which `_box_kernel` gathers from a zero-padded slot).
    # For the synth, we just take the first ngkmax cells in raster order
    # as "valid"; the rest are sentinel.
    g_index = np.full((nk, nx, ny, nz), ngkmax, dtype=np.int32)
    for k in range(nk):
        valid_cells = rng.choice(n_rtot, size=ngkmax, replace=False)
        for idx, cell in enumerate(sorted(valid_cells)):
            cx = cell // (ny * nz)
            cy = (cell // nz) % ny
            cz = cell % nz
            g_index[k, cx, cy, cz] = idx

    # psi_G: (nk, nb_total, ns, ngkmax) c128 random.
    psi_G = (rng.standard_normal((nk, nb_total, ns, ngkmax))
             + 1j * rng.standard_normal((nk, nb_total, ns, ngkmax)))
    psi_G = jnp.asarray(psi_G, dtype=jnp.complex128)

    # kvecs_frac: (nk, 3) random in [-0.5, 0.5).
    kvecs_frac = rng.uniform(-0.5, 0.5, size=(nk, 3)).astype(np.float64)

    # n_rmu fixed; psi_l_X / psi_r_X built by caller with the right
    # band-window widths.

    return dict(
        psi_G=psi_G, g_index=jnp.asarray(g_index), kvecs_frac=kvecs_frac,
        nk=nk, kgrid=kgrid, nb_total=nb_total, ns=ns,
        fft_grid=fft_grid, n_rmu=n_rmu, n_zchunk=n_zchunk,
        ngkmax=ngkmax, rng=rng,
    )


def _make_psi_X(synth, band_window, rng_seed):
    """psi_X for one (L or R) band window — independent random sample,
    shaped ``(nk, n_rmu, nb_window, ns)``."""
    nk = synth['nk']
    n_rmu = synth['n_rmu']
    ns = synth['ns']
    lo, hi = band_window
    nb = hi - lo
    rng = np.random.default_rng(rng_seed)
    arr = (rng.standard_normal((nk, n_rmu, nb, ns))
           + 1j * rng.standard_normal((nk, n_rmu, nb, ns)))
    return jnp.asarray(arr, dtype=jnp.complex128)


def _run_pair(
    synth, *,
    band_chunk_ranges, band_range_left, band_range_right,
    r_start=0,
    gamma_L=None, gamma_R=None,
    psi_l_X_seed=1, psi_r_X_seed=2,
):
    """Run the new ``z_q_from_psi_sm`` and the numpy reference on the
    same synth + return both Z_q tensors (numpy) for comparison."""
    store = _MockPsiGStore(
        synth['psi_G'], synth['g_index'], synth['kvecs_frac'],
        band_chunk_ranges, fft_grid=synth['fft_grid'])

    psi_l_X = _make_psi_X(synth, band_range_left, psi_l_X_seed)
    psi_r_X = _make_psi_X(synth, band_range_right, psi_r_X_seed)

    Z_q_new = z_q_from_psi_sm(
        psi_l_X, psi_r_X, store,
        band_chunk_ranges=band_chunk_ranges,
        band_range_left=band_range_left,
        band_range_right=band_range_right,
        r_start_dyn=int(r_start),
        r_chunk_size=int(synth['n_zchunk']),
        gamma_L=gamma_L, gamma_R=gamma_R,
        kgrid=synth['kgrid'], mesh_xy=MESH,
    )

    Z_q_ref = _reference_z_q(
        psi_l_X=psi_l_X, psi_r_X=psi_r_X,
        psi_G=synth['psi_G'], g_index=synth['g_index'],
        kvecs_frac=synth['kvecs_frac'],
        band_range_left=band_range_left,
        band_range_right=band_range_right,
        fft_grid=synth['fft_grid'],
        r_start=int(r_start),
        r_chunk_size=int(synth['n_zchunk']),
        kgrid=synth['kgrid'],
        gamma_L=gamma_L, gamma_R=gamma_R,
    )
    return np.asarray(Z_q_new), np.asarray(Z_q_ref)


# ---------------------------------------------------------------------------
# G1.1 (main) — charge channel, multi-bc, full band overlap
# ---------------------------------------------------------------------------

def test_g1_1_charge_multi_bc():
    """Main G1.1: nb_total=8, 2 bcs of 4 bands, L=R=(0,8) (full).
    Charge channel (gamma_L = gamma_R = None)."""
    synth = _build_synth()
    Z_new, Z_ref = _run_pair(
        synth,
        band_chunk_ranges=((0, 4), (4, 8)),
        band_range_left=(0, 8),
        band_range_right=(0, 8),
    )
    max_abs = float(np.max(np.abs(Z_new - Z_ref)))
    ref_norm = float(np.max(np.abs(Z_ref)))
    max_rel = max_abs / ref_norm if ref_norm > 0 else max_abs
    print(f"\nG1.1 charge multi-bc: |Z_ref|_max={ref_norm:.4e}, "
          f"max|Δ|={max_abs:.4e}, max rel={max_rel:.4e}")
    np.testing.assert_allclose(Z_new, Z_ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# G1.1a — single bc (n_bc = 1).  Scan runs one iter.
# ---------------------------------------------------------------------------

def test_g1_1a_single_bc():
    synth = _build_synth()
    Z_new, Z_ref = _run_pair(
        synth,
        band_chunk_ranges=((0, 8),),
        band_range_left=(0, 8),
        band_range_right=(0, 8),
    )
    max_rel = float(np.max(np.abs(Z_new - Z_ref))) / float(np.max(np.abs(Z_ref)))
    print(f"\nG1.1a single bc: max rel={max_rel:.4e}")
    np.testing.assert_allclose(Z_new, Z_ref, rtol=1e-10, atol=1e-12)


def test_hoisted_psi_r_cache_matches_streaming_path():
    """Cache construction changes lifetime and placement, not arithmetic."""
    from isdf.core import build_psi_r_cache_sm

    synth = _build_synth()
    band_chunks = ((0, 4), (4, 8))
    store = _MockPsiGStore(
        synth['psi_G'], synth['g_index'], synth['kvecs_frac'],
        band_chunks, fft_grid=synth['fft_grid'])
    psi_l_X = _make_psi_X(synth, (0, 8), 1)
    psi_r_X = _make_psi_X(synth, (0, 8), 2)

    streaming = z_q_from_psi_sm(
        psi_l_X, psi_r_X, store,
        band_chunk_ranges=band_chunks,
        band_range_left=(0, 8), band_range_right=(0, 8),
        r_start_dyn=0, r_chunk_size=int(synth['n_zchunk']),
        kgrid=synth['kgrid'], mesh_xy=MESH)
    cache = build_psi_r_cache_sm(store, mesh_xy=MESH)
    cached = z_q_from_psi_sm(
        psi_l_X, psi_r_X, store, cache,
        band_chunk_ranges=band_chunks,
        band_range_left=(0, 8), band_range_right=(0, 8),
        r_start_dyn=0, r_chunk_size=int(synth['n_zchunk']),
        kgrid=synth['kgrid'], mesh_xy=MESH)

    np.testing.assert_allclose(
        np.asarray(cached), np.asarray(streaming), rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# G1.1b — short final bc (last bc shorter than bpd_max).  Pad-band-zero
# math-neutrality test.
# ---------------------------------------------------------------------------

def test_g1_1b_short_final_bc():
    # nb_total=9 = 4 + 4 + 1; last bc has 1 band, bpd_max=4.
    synth = _build_synth(nb_total=9)
    Z_new, Z_ref = _run_pair(
        synth,
        band_chunk_ranges=((0, 4), (4, 8), (8, 9)),
        band_range_left=(0, 9),
        band_range_right=(0, 9),
    )
    max_rel = float(np.max(np.abs(Z_new - Z_ref))) / float(np.max(np.abs(Z_ref)))
    print(f"\nG1.1b short final bc: max rel={max_rel:.4e}")
    np.testing.assert_allclose(Z_new, Z_ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# G1.1c — asymmetric L/R windows (nb_L != nb_R, partial overlap with bcs).
# ---------------------------------------------------------------------------

def test_g1_1c_asymmetric_lr():
    # nb_total=8, L=(0,5), R=(3,8).  Overlap is [3,5).  bc 0=(0,4) hits
    # L only at [0,4) and R at [3,4); bc 1=(4,8) hits L at [4,5) and R
    # at [4,8).
    synth = _build_synth()
    Z_new, Z_ref = _run_pair(
        synth,
        band_chunk_ranges=((0, 4), (4, 8)),
        band_range_left=(0, 5),
        band_range_right=(3, 8),
    )
    max_rel = float(np.max(np.abs(Z_new - Z_ref))) / float(np.max(np.abs(Z_ref)))
    print(f"\nG1.1c asymmetric L/R: max rel={max_rel:.4e}")
    np.testing.assert_allclose(Z_new, Z_ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# G1.1d — bispinor γ̃ != I (vertex_mu_L = 1 → γ̃^1 = γ0·γ1).  ns=4 here.
# ---------------------------------------------------------------------------

def test_g1_1d_bispinor_gamma1():
    # ns=4 for bispinor.  Use the prod γ̃^1 perm/phase tuple.
    synth = _build_synth(ns=4)
    # γ̃^1 = γ0 · γ1 — pure-permutation matrix, computed from gamma_matrices.
    # Use the same `_gamma_perm_phase_mu` helper as production.
    from isdf.core import _gamma_perm_phase_mu
    perm, phase = _gamma_perm_phase_mu(1)
    Z_new, Z_ref = _run_pair(
        synth,
        band_chunk_ranges=((0, 4), (4, 8)),
        band_range_left=(0, 8),
        band_range_right=(0, 8),
        gamma_L=(perm, phase),
        gamma_R=(perm, phase),
    )
    max_rel = float(np.max(np.abs(Z_new - Z_ref))) / float(np.max(np.abs(Z_ref)))
    print(f"\nG1.1d bispinor γ̃^1: max rel={max_rel:.4e}")
    np.testing.assert_allclose(Z_new, Z_ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# G1.1e — pseudobands (norms != 1, applied as pre-divide on psi_l_X /
# psi_r_X per plan §5.3.4 / §6.3).  Round 6 production does this at the
# kernel call site, OUTSIDE z_q_from_psi_sm — so for the test we just
# verify the kernel's output scales linearly in psi_l_X / psi_r_X (a
# trivial bilinearity check; covers the "if we pre-divide upstream, the
# kernel doesn't break" contract).
# ---------------------------------------------------------------------------

def test_g1_1e_pseudobands_pre_divide():
    synth = _build_synth()
    # Pseudoband norms — one factor per band of L and R windows.
    rng = np.random.default_rng(0xCAFE)
    norms_l = rng.uniform(0.5, 2.0, size=8).astype(np.float64)
    norms_r = rng.uniform(0.5, 2.0, size=8).astype(np.float64)

    store = _MockPsiGStore(
        synth['psi_G'], synth['g_index'], synth['kvecs_frac'],
        ((0, 4), (4, 8)), fft_grid=synth['fft_grid'])
    psi_l_X_raw = _make_psi_X(synth, (0, 8), 11)
    psi_r_X_raw = _make_psi_X(synth, (0, 8), 12)
    # Pre-divide per plan §6.3 — production does this at the kernel call site.
    psi_l_X_scaled = psi_l_X_raw / jnp.asarray(
        norms_l)[None, None, :, None]
    psi_r_X_scaled = psi_r_X_raw / jnp.asarray(
        norms_r)[None, None, :, None]

    Z_new = z_q_from_psi_sm(
        psi_l_X_scaled, psi_r_X_scaled, store,
        band_chunk_ranges=((0, 4), (4, 8)),
        band_range_left=(0, 8), band_range_right=(0, 8),
        r_start_dyn=0, r_chunk_size=int(synth['n_zchunk']),
        kgrid=synth['kgrid'], mesh_xy=MESH,
    )
    # Reference: same pre-divide on psi_l_X / psi_r_X feeds the
    # reference einsum.  Mathematically identical to "divide psi_l_Y /
    # psi_r_Y after IFFT" by linearity of einsum.
    Z_ref = _reference_z_q(
        psi_l_X=psi_l_X_scaled, psi_r_X=psi_r_X_scaled,
        psi_G=synth['psi_G'], g_index=synth['g_index'],
        kvecs_frac=synth['kvecs_frac'],
        band_range_left=(0, 8), band_range_right=(0, 8),
        fft_grid=synth['fft_grid'],
        r_start=0, r_chunk_size=int(synth['n_zchunk']),
        kgrid=synth['kgrid'],
    )
    max_rel = float(np.max(np.abs(np.asarray(Z_new) - np.asarray(Z_ref)))) / \
              float(np.max(np.abs(np.asarray(Z_ref))))
    print(f"\nG1.1e pseudobands pre-divide: max rel={max_rel:.4e}")
    np.testing.assert_allclose(
        np.asarray(Z_new), np.asarray(Z_ref), rtol=1e-10, atol=1e-12)


# ===========================================================================
#  PsiGStore host-tile slicer contracts (merged from test_psi_g_store.py):
#  the pad-zero / band-window invariants the streaming kernel above relies on
# ===========================================================================
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.psi_G_store import _zero_user_band_pad_in_shard, PsiGStore


def test_zero_user_band_pad_in_shard_zeros_only_requested_pad_slots():
    data = np.ones((2, 4, 1, 1, 1, 1), dtype=np.complex128)
    data[:, :, 0, 0, 0, 0] *= np.arange(4)[None, :] + 1

    out = _zero_user_band_pad_in_shard(
        data,
        bc_range=(128, 192),
        shard_band_slice=slice(20, 24),
        user_band_stop=150,
    )

    assert out is not data
    np.testing.assert_array_equal(out[:, :2], data[:, :2])
    assert np.all(out[:, 2:] == 0.0)
    np.testing.assert_array_equal(data[:, :, 0, 0, 0, 0], [[1, 2, 3, 4], [1, 2, 3, 4]])


def test_zero_user_band_pad_in_shard_returns_original_when_no_pad_owned():
    data = np.ones((1, 3, 1, 1, 1, 1), dtype=np.complex128)

    out = _zero_user_band_pad_in_shard(
        data,
        bc_range=(64, 128),
        shard_band_slice=slice(0, 3),
        user_band_stop=150,
    )

    assert out is data


def test_zero_user_band_pad_in_shard_rejects_strided_band_slice():
    data = np.ones((1, 3, 1, 1, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="contiguous"):
        _zero_user_band_pad_in_shard(
            data,
            bc_range=(0, 64),
            shard_band_slice=slice(0, 6, 2),
            user_band_stop=4,
        )


# ---------------------------------------------------------------------------
# _slice_local_tile_bc — per-iter host-tile slicer used by the one-time
# ψ(r)-cache builder's io_callback (and the narrow no-cache reference path).
# Built without going through PsiGStore.__init__ so the test stays free of
# WfnLoader / Meta plumbing: subclasses PsiGStore, populates _host_tiles
# directly with a random multi-bc ψ(G) tile, and exercises the slicer.
# ---------------------------------------------------------------------------


class _FakePsiGStore(PsiGStore):
    """PsiGStore stub: skips loader/meta plumbing for unit tests.

    Caller hands in a single per-rank tile (shape ``(nk, nb_local, ns,
    ngkmax)``) — installed into ``_host_tiles`` for every (x, y) cell in
    the mesh — plus a ``band_chunk_ranges`` tuple that defines the
    bc-stacked band layout.  bc-band offsets and ``_bpd_max`` are
    derived from ``band_chunk_ranges`` exactly as production code does.
    """

    def __init__(self, *, mesh_xy, tile, band_chunk_ranges):
        self.mesh = mesh_xy
        self._dtype = tile.dtype
        from common.psi_G_store import _mesh_device_coords
        self._coords = _mesh_device_coords(mesh_xy)
        self._per_rank_shape = tuple(int(s) for s in tile.shape)
        self._host_tiles = {(x, y): tile.copy() for (x, y) in self._coords.values()}
        self._g_index_dev = None
        self._kvecs_frac_dev = None
        self.band_chunk_ranges = tuple(tuple(bc) for bc in band_chunk_ranges)
        # Mesh size = 1 here; bpd_per_bc[i] = (b_hi - b_lo) // 1 = bc_size.
        bpd_per_bc = [(b_hi - b_lo) for (b_lo, b_hi) in self.band_chunk_ranges]
        self._bpd_per_bc = tuple(bpd_per_bc)
        self._bpd_max = max(bpd_per_bc) if bpd_per_bc else 0
        offsets = [0]
        for bpd in bpd_per_bc:
            offsets.append(offsets[-1] + bpd)
        self._bc_band_offsets = tuple(offsets)

def test_slice_local_tile_bc_returns_correct_band_window():
    """Slicer returns each bc's bands of the host tile, padded to bpd_max."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, ns, ngkmax = 3, 2, 5
    band_chunks = ((0, 4), (4, 8), (8, 10))             # last bc short (bpd=2 vs 4)
    bpd_max = 4
    nb_local = 4 + 4 + 2
    rng = np.random.default_rng(11)
    tile = (rng.standard_normal((nk, nb_local, ns, ngkmax))
            + 1j * rng.standard_normal((nk, nb_local, ns, ngkmax))).astype(
        np.complex128)

    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, band_chunk_ranges=band_chunks)
    assert store._bpd_max == bpd_max

    # bc 0: bands [0, 4) — full
    out0 = store._slice_local_tile_bc(0, 0, 0)
    assert out0.shape == (nk, bpd_max, ns, ngkmax)
    np.testing.assert_array_equal(out0[:, :4, :, :], tile[:, 0:4, :, :])

    # bc 1: bands [4, 8) — full
    out1 = store._slice_local_tile_bc(0, 0, 1)
    np.testing.assert_array_equal(out1[:, :4, :, :], tile[:, 4:8, :, :])

    # bc 2: bands [8, 10) — short; pad rows must be exactly zero (math-neutral)
    out2 = store._slice_local_tile_bc(0, 0, 2)
    assert out2.shape == (nk, bpd_max, ns, ngkmax)
    np.testing.assert_array_equal(out2[:, :2, :, :], tile[:, 8:10, :, :])
    assert np.all(out2[:, 2:, :, :] == 0.0)             # pad-rows EXACTLY zero
    assert out2[:, 2:, :, :].dtype == np.complex128


def test_slice_local_tile_bc_traced_bc_idx_inputs():
    """Slicer accepts traced int32 scalars (the io_callback contract); resolved
    to Python ints inside the host fn."""
    import jax.numpy as jnp_local                       # noqa
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, ns, ngkmax = 2, 1, 3
    band_chunks = ((0, 2), (2, 4))
    tile = np.arange(nk * 4 * ns * ngkmax, dtype=np.complex128).reshape(
        nk, 4, ns, ngkmax)

    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, band_chunk_ranges=band_chunks)
    # Traced int32 scalars (what io_callback would pass).
    x_idx = np.int32(0)
    y_idx = np.int32(0)
    bc_idx = np.int32(1)
    out = store._slice_local_tile_bc(x_idx, y_idx, bc_idx)
    np.testing.assert_array_equal(out[:, :2, :, :], tile[:, 2:4, :, :])


def test_slice_local_tile_bc_rejects_out_of_range_bc():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, ns, ngkmax = 2, 1, 3
    tile = np.zeros((nk, 4, ns, ngkmax), dtype=np.complex128)
    store = _FakePsiGStore(mesh_xy=mesh, tile=tile,
                            band_chunk_ranges=((0, 2), (2, 4)))
    with pytest.raises(ValueError, match=r"bc_idx=2 not in"):
        store._slice_local_tile_bc(0, 0, 2)


def test_psi_G_device_full_property_removed():
    """Round 6 deleted the buggy lazy property + field.  Confirm absence."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    tile = np.zeros((1, 1, 1, 1), dtype=np.complex128)
    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, band_chunk_ranges=((0, 1),))
    assert not hasattr(store, "psi_G_device_full")
    assert not hasattr(store, "_psi_G_device_full")
