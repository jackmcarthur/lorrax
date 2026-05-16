"""End-to-end test for the new V_q orchestrator
:func:`gw.v_q_g_flat.compute_all_V_q_g_flat`.

We don't have a GPU + writer pipeline at unit-test time, so we
*synthesize* a G-flat ζ file directly:

1. Build a small per-q sphere via
   ``compute_per_q_bare_coulomb_components``.
2. Fill ζ̃ at the per-q sphere positions with random complex values,
   zero at pad slots.
3. Build a fake WFN.h5 + zeta_q_G HDF5 with the writer's metadata
   layout (zeta_layout='G_flat', gvec_components, ngk).
4. Open via ``ZetaLoader`` and call the orchestrator.
5. Compare to a hand-rolled reference: V_q[μ,ν] = Σ_G_logical
   conj(ζ̃_μ(G)) v(q+G) ζ̃_ν(G) on the per-q sphere.

This exercises the per-q gather + v(q+G) build + G-chunked contract
+ V_acc / g0_acc dynamic_update_slice + IBZ unfold (off, here — we
test full-BZ disk; IBZ correctness is covered by the writer's own
``test_zeta_loader.py`` and the unfold helpers' own tests).
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from common.coulomb_sphere import compute_per_q_bare_coulomb_components
from file_io.isdf_header import IsdfHeader, write_isdf_header
from file_io.mf_header import copy_mf_header
from file_io.zeta_loader import ZetaLoader
from gw.v_q_g_flat import compute_all_V_q_g_flat
from tests.test_mf_isdf_header_roundtrip import _make_fake_wfn


@pytest.fixture
def single_device_mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))


def _build_g_flat_zeta_file(tmp_path, *, fft_grid, kgrid, bvec, cutoff,
                             n_rmu, sys_dim, seed=0xC0DE):
    """Synthesise a G-flat zeta_q.h5 carrying random ζ̃ on the per-q sphere."""
    nkx, nky, nkz = kgrid
    n_q_full = nkx * nky * nkz

    # BGW wrap, fractional.
    q_int = np.array(
        [(qx, qy, qz) for qx in range(nkx)
         for qy in range(nky) for qz in range(nkz)],
        dtype=np.float64)
    kg = np.asarray(kgrid, dtype=np.float64)
    q_int_w = np.where(q_int > kg / 2, q_int - kg, q_int)
    q_frac = q_int_w / kg

    pkg = compute_per_q_bare_coulomb_components(
        fft_grid, bvec, q_frac, cutoff, sys_dim=sys_dim)
    gvec_components = pkg['gvec_components_padded']         # (n_q, 3, ngkmax)
    ngk_per_q = pkg['ngk_per_q']                            # (n_q,)
    ngkmax = pkg['ngkmax']

    # Random ζ̃ at logical sphere positions; zero at pad slots.
    rng = np.random.default_rng(seed)
    zeta_q_G = np.zeros((n_q_full, n_rmu, ngkmax), dtype=np.complex128)
    for q in range(n_q_full):
        nk = int(ngk_per_q[q])
        zeta_q_G[q, :, :nk] = (
            rng.standard_normal((n_rmu, nk))
            + 1j * rng.standard_normal((n_rmu, nk))
        ).astype(np.complex128)

    # Build the file: fake WFN header + isdf_header + zeta_q_G.
    wfn_path = str(tmp_path / "WFN.h5")
    out_path = str(tmp_path / "zeta_q.h5")
    _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, out_path, dst_mode='w')
    hdr = IsdfHeader.build(
        r_mu_fft_idx=(np.arange(3 * n_rmu, dtype=np.int32).reshape(n_rmu, 3)
                       % max(1, min(fft_grid))),
        fft_grid=fft_grid, density='scalar', vertex_mu_L=0,
        zeta_is_done=True, zeta_layout='G_flat',
        gvec_components=gvec_components,
        ngk_per_q=ngk_per_q,
        zeta_cutoff_ry=cutoff,
    )
    write_isdf_header(out_path, hdr, mode='a')
    with h5py.File(out_path, 'a') as f:
        f.create_dataset('zeta_q_G', data=zeta_q_G)

    return out_path, zeta_q_G, q_frac, gvec_components, ngk_per_q, ngkmax


def _reference_v_q(zeta_q_G, q_frac, gvec_components, ngk_per_q,
                    bvec, cell_volume, sys_dim, vcoul_cutoff):
    """One-shot reference V[q,μ,ν] via direct einsum on the logical sphere."""
    from gw.compute_vcoul import compute_v_q_per_G
    n_q = zeta_q_G.shape[0]
    n_rmu = zeta_q_G.shape[1]
    V_ref = np.zeros((n_q, n_rmu, n_rmu), dtype=np.complex128)
    v_table = compute_v_q_per_G(
        q_frac, gvec_components,
        bvec=bvec, cell_volume=cell_volume,
        sys_dim=sys_dim, vcoul_cutoff_ry=vcoul_cutoff,
    )
    for q in range(n_q):
        nk = int(ngk_per_q[q])
        z = zeta_q_G[q, :, :nk]
        v = v_table[q, :nk]
        V_ref[q] = np.einsum('mG,G,nG->mn', np.conj(z), v, z, optimize=True)
    return V_ref


def test_compute_all_V_q_g_flat_matches_einsum_reference(
        tmp_path, single_device_mesh):
    """Synthesise a G-flat ζ; run the orchestrator; compare to the
    direct einsum on the logical per-q sphere."""
    fft_grid = (6, 6, 8)
    kgrid = (2, 2, 1)
    bvec = np.diag([0.7, 0.7, 0.3])
    cutoff = 8.0
    cell_volume = 100.0
    n_rmu = 4
    sys_dim = 2

    path, zeta_disk, q_frac, gvec_comps, ngk, ngkmax = (
        _build_g_flat_zeta_file(
            tmp_path, fft_grid=fft_grid, kgrid=kgrid, bvec=bvec,
            cutoff=cutoff, n_rmu=n_rmu, sys_dim=sys_dim))

    V_ref = _reference_v_q(
        zeta_disk, q_frac, gvec_comps, ngk,
        bvec=bvec, cell_volume=cell_volume,
        sys_dim=sys_dim, vcoul_cutoff=cutoff,
    )

    with ZetaLoader(path, mesh=single_device_mesh) as loader:
        with single_device_mesh:
            V_q, g0 = compute_all_V_q_g_flat(
                loader,
                kgrid=kgrid, fft_grid=fft_grid,
                bvec=bvec, cell_volume=cell_volume,
                mesh_xy=single_device_mesh,
                sys_dim=sys_dim,
                bare_coulomb_cutoff_ry=cutoff,
                async_prefetch=False,
                sym=None,                # full-BZ iteration
                centroid_indices=None,
            )
            V_q = np.asarray(V_q)
            g0 = np.asarray(g0)

    # The orchestrator pads n_rmu up to mesh-product (here 1); on a single
    # device n_rmu_padded == n_rmu, so V_q is (n_q, n_rmu, n_rmu) directly.
    assert V_q.shape == V_ref.shape, (V_q.shape, V_ref.shape)
    np.testing.assert_allclose(V_q, V_ref, atol=1e-10, rtol=1e-10)

    # g0[μ,q] = ζ̃[q, μ, 0] (sphere convention: G=(0,0,0) at index 0).
    ref_g0 = zeta_disk[:, :, 0]
    np.testing.assert_allclose(g0, ref_g0, atol=1e-12, rtol=1e-12)


def test_compute_all_V_q_g_flat_async_matches_sync(
        tmp_path, single_device_mesh):
    """async_prefetch=True must produce identical V_q / g0 as sync."""
    fft_grid = (4, 4, 6)
    kgrid = (2, 2, 1)
    bvec = np.diag([0.8, 0.8, 0.3])
    cutoff = 6.0
    cell_volume = 50.0
    n_rmu = 3
    sys_dim = 2

    path, *_ = _build_g_flat_zeta_file(
        tmp_path, fft_grid=fft_grid, kgrid=kgrid, bvec=bvec,
        cutoff=cutoff, n_rmu=n_rmu, sys_dim=sys_dim, seed=0xFADE)

    with ZetaLoader(path, mesh=single_device_mesh) as loader:
        with single_device_mesh:
            V_sync, g0_sync = compute_all_V_q_g_flat(
                loader,
                kgrid=kgrid, fft_grid=fft_grid,
                bvec=bvec, cell_volume=cell_volume,
                mesh_xy=single_device_mesh,
                sys_dim=sys_dim,
                bare_coulomb_cutoff_ry=cutoff,
                async_prefetch=False,
            )
            V_async, g0_async = compute_all_V_q_g_flat(
                loader,
                kgrid=kgrid, fft_grid=fft_grid,
                bvec=bvec, cell_volume=cell_volume,
                mesh_xy=single_device_mesh,
                sys_dim=sys_dim,
                bare_coulomb_cutoff_ry=cutoff,
                async_prefetch=True,
            )

    np.testing.assert_allclose(
        np.asarray(V_async), np.asarray(V_sync),
        atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(g0_async), np.asarray(g0_sync),
        atol=1e-12, rtol=1e-12)


def test_compute_all_V_q_g_flat_rejects_r_space_loader(
        tmp_path, single_device_mesh):
    """r-space file → orchestrator raises a clear error."""
    from tests.test_zeta_loader import _build_zeta_h5
    out = _build_zeta_h5(tmp_path, n_q_disk=2)
    with ZetaLoader(out, mesh=single_device_mesh) as loader:
        with pytest.raises(ValueError, match="layout must be 'G_flat'"):
            compute_all_V_q_g_flat(
                loader,
                kgrid=(2, 3, 4),
                fft_grid=(8, 8, 8),
                bvec=np.eye(3),
                cell_volume=1.0,
                mesh_xy=single_device_mesh,
                sys_dim=2,
            )
