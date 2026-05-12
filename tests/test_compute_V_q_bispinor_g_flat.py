"""End-to-end test for the bispinor V_q orchestrator on the G-flat
on-disk format.

Synthesises four G-flat ζ files (charge + 3 transverse) with shared
geometry, calls
:func:`gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5`, and verifies
that each of the 7 unique tiles written to disk matches a hand-rolled
einsum reference V^{μ_L, ν_L}_q[μ, ν] = Σ_G conj(ζ_L) · v_q · ζ_R
on the per-q sphere.

Single-device mesh, full BZ (no IBZ reduction — the bispinor writer
always emits full-BZ ζ files in production via gw_init.fit_zeta).
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
from file_io.zeta_reader import ZetaReader
from gw.v_q_bispinor import (
    compute_V_q_bispinor_g_flat_to_h5,
    _make_per_q_v_builder_for_tile,
    UNIQUE_TILES, tile_dataset_name,
)
from tests.test_mf_isdf_header_roundtrip import _make_fake_wfn


@pytest.fixture
def single_device_mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_g_flat_zeta(tmp_path, name, *, fft_grid, kgrid, bvec, cutoff,
                        n_rmu, sys_dim, seed):
    """Synthesise one full-BZ G-flat zeta file (writer convention)."""
    nkx, nky, nkz = kgrid
    n_q_full = nkx * nky * nkz
    q_int = np.array(
        [(qx, qy, qz) for qx in range(nkx)
         for qy in range(nky) for qz in range(nkz)],
        dtype=np.float64)
    kg = np.asarray(kgrid, dtype=np.float64)
    q_int_w = np.where(q_int > kg / 2, q_int - kg, q_int)
    q_frac = q_int_w / kg

    pkg = compute_per_q_bare_coulomb_components(
        fft_grid, bvec, q_frac, cutoff, sys_dim=sys_dim)
    gvec_components = pkg['gvec_components_padded']
    ngk_per_q = pkg['ngk_per_q']
    ngkmax = pkg['ngkmax']

    rng = np.random.default_rng(seed)
    zeta_q_G = np.zeros((n_q_full, n_rmu, ngkmax), dtype=np.complex128)
    for q in range(n_q_full):
        nk = int(ngk_per_q[q])
        zeta_q_G[q, :, :nk] = (
            rng.standard_normal((n_rmu, nk))
            + 1j * rng.standard_normal((n_rmu, nk))
        ).astype(np.complex128)

    wfn_path = str(tmp_path / "WFN.h5")
    out_path = str(tmp_path / name)
    if not (tmp_path / "WFN.h5").exists():
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
    return out_path, zeta_q_G, q_frac, gvec_components, ngk_per_q


def _ref_tile_V(zeta_L_disk, zeta_R_disk, q_frac, gvec_components, ngk_per_q,
                 *, mu_L, nu_L, bvec, cell_volume, sys_dim, cutoff):
    """One-shot einsum reference for the (μ_L, ν_L) tile."""
    builder = _make_per_q_v_builder_for_tile(
        mu_L=mu_L, nu_L=nu_L,
        bvec=bvec, cell_volume=cell_volume,
        sys_dim=sys_dim, vcoul_cutoff_ry=cutoff)
    v_table = np.asarray(builder(q_frac, gvec_components))
    n_q = zeta_L_disk.shape[0]
    n_rmu_L = zeta_L_disk.shape[1]
    n_rmu_R = zeta_R_disk.shape[1]
    V_ref = np.zeros((n_q, n_rmu_L, n_rmu_R), dtype=np.complex128)
    for q in range(n_q):
        nk = int(ngk_per_q[q])
        zL = zeta_L_disk[q, :, :nk]
        zR = zeta_R_disk[q, :, :nk]
        vq = v_table[q, :nk]
        V_ref[q] = np.einsum('mG,G,nG->mn', np.conj(zL), vq, zR,
                              optimize=True)
    return V_ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bispinor_7_tiles_match_einsum_reference(tmp_path, single_device_mesh):
    """All 7 unique tiles agree with a per-q einsum reference."""
    fft_grid = (4, 4, 6)
    kgrid = (2, 2, 1)
    bvec = np.diag([0.7, 0.7, 0.3])
    cutoff = 6.0
    cell_volume = 100.0
    n_rmu_C = 4
    n_rmu_T = 3
    sys_dim = 2

    paths = {}
    disks = {}
    for label, name, n_rmu, seed in [
        ('C', 'zeta_q.h5', n_rmu_C, 0xC0DE),
        ('T1', 'zeta_q_mu1.h5', n_rmu_T, 0xD15C),
        ('T2', 'zeta_q_mu2.h5', n_rmu_T, 0xD15D),
        ('T3', 'zeta_q_mu3.h5', n_rmu_T, 0xD15E),
    ]:
        p, z, qf, gv, nk = _build_g_flat_zeta(
            tmp_path, name, fft_grid=fft_grid, kgrid=kgrid,
            bvec=bvec, cutoff=cutoff, n_rmu=n_rmu, sys_dim=sys_dim,
            seed=seed)
        paths[label] = p
        disks[label] = (z, qf, gv, nk)

    out_path = tmp_path / "v_q_bispinor.h5"
    with ZetaReader(paths['C'], mesh=single_device_mesh) as zC, \
         ZetaReader(paths['T1'], mesh=single_device_mesh) as zT1, \
         ZetaReader(paths['T2'], mesh=single_device_mesh) as zT2, \
         ZetaReader(paths['T3'], mesh=single_device_mesh) as zT3:
        with single_device_mesh:
            compute_V_q_bispinor_g_flat_to_h5(
                zeta_C_loader=zC,
                zeta_T_loaders=(zT1, zT2, zT3),
                output_h5_path=str(out_path),
                mesh_xy=single_device_mesh,
                kgrid=kgrid, fft_grid=fft_grid,
                bvec=bvec, cell_volume=cell_volume,
                sys_dim=sys_dim,
                n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T,
                bare_coulomb_cutoff_ry=cutoff,
                verbose=False,
            )

    # Re-derive q_frac / gvec_components / ngk on host for the references
    z_C, qf, gv, nk = disks['C']
    z_disks = {0: z_C,
                1: disks['T1'][0],
                2: disks['T2'][0],
                3: disks['T3'][0]}

    with h5py.File(out_path, 'r') as f:
        ds = list(f.keys())
        for mu_L, nu_L in UNIQUE_TILES:
            name = tile_dataset_name(mu_L, nu_L)
            assert name in ds, (name, ds)
            V_got = np.asarray(f[name][...])
            V_ref = _ref_tile_V(
                z_disks[mu_L], z_disks[nu_L], qf, gv, nk,
                mu_L=mu_L, nu_L=nu_L,
                bvec=bvec, cell_volume=cell_volume,
                sys_dim=sys_dim, cutoff=cutoff,
            )
            np.testing.assert_allclose(
                V_got, V_ref, atol=1e-10, rtol=1e-10,
                err_msg=f"tile (μ_L={mu_L}, ν_L={nu_L}) mismatch")

        # g0 only on the CC tile
        assert "V_qmunu_CC_g0" in ds, ds
        g0_got = np.asarray(f["V_qmunu_CC_g0"][...])
        # g0[q, μ] = ζ_C[q, μ, 0] (G=0 lives at sphere position 0)
        np.testing.assert_allclose(
            g0_got, z_C[:, :, 0], atol=1e-12, rtol=1e-12)
        # Layout metadata
        import json
        unique = [tuple(t) for t in json.loads(f.attrs['unique_tiles'])]
        assert set(unique) == set(UNIQUE_TILES)
        assert f['v_qmunu_format'][()].decode() == "bispinor_lorentz_v1"


def test_bispinor_CC_tile_matches_charge_orchestrator(
        tmp_path, single_device_mesh):
    """CC tile from the bispinor orchestrator == output of the charge-only
    orchestrator on the same ζ_C file."""
    from gw.v_q_g_flat import compute_all_V_q_g_flat

    fft_grid = (4, 4, 6)
    kgrid = (2, 2, 1)
    bvec = np.diag([0.7, 0.7, 0.3])
    cutoff = 6.0
    cell_volume = 100.0
    n_rmu_C = 3
    n_rmu_T = 3
    sys_dim = 2

    # Build only the C and 3 T files (T are dummies for the bispinor run).
    paths = {}
    disks = {}
    for label, name, n_rmu, seed in [
        ('C', 'zeta_q.h5', n_rmu_C, 0xC0DE),
        ('T1', 'zeta_q_mu1.h5', n_rmu_T, 0xD15C),
        ('T2', 'zeta_q_mu2.h5', n_rmu_T, 0xD15D),
        ('T3', 'zeta_q_mu3.h5', n_rmu_T, 0xD15E),
    ]:
        p, z, *rest = _build_g_flat_zeta(
            tmp_path, name, fft_grid=fft_grid, kgrid=kgrid,
            bvec=bvec, cutoff=cutoff, n_rmu=n_rmu, sys_dim=sys_dim,
            seed=seed)
        paths[label] = p
        disks[label] = z

    # Charge-only run via the dedicated orchestrator
    with ZetaLoader(paths['C'], mesh=single_device_mesh) as zC_loader:
        with single_device_mesh:
            V_charge, g0_charge = compute_all_V_q_g_flat(
                zC_loader,
                kgrid=kgrid, fft_grid=fft_grid,
                bvec=bvec, cell_volume=cell_volume,
                mesh_xy=single_device_mesh,
                n_rmu=n_rmu_C, sys_dim=sys_dim,
                bare_coulomb_cutoff_ry=cutoff,
                verbose=False, async_prefetch=False,
            )
    V_charge = np.asarray(V_charge)
    g0_charge = np.asarray(g0_charge)

    # Bispinor run
    out_path = tmp_path / "v_q_bispinor.h5"
    with ZetaReader(paths['C'], mesh=single_device_mesh) as zC, \
         ZetaReader(paths['T1'], mesh=single_device_mesh) as zT1, \
         ZetaReader(paths['T2'], mesh=single_device_mesh) as zT2, \
         ZetaReader(paths['T3'], mesh=single_device_mesh) as zT3:
        with single_device_mesh:
            compute_V_q_bispinor_g_flat_to_h5(
                zeta_C_loader=zC,
                zeta_T_loaders=(zT1, zT2, zT3),
                output_h5_path=str(out_path),
                mesh_xy=single_device_mesh,
                kgrid=kgrid, fft_grid=fft_grid,
                bvec=bvec, cell_volume=cell_volume,
                sys_dim=sys_dim,
                n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T,
                bare_coulomb_cutoff_ry=cutoff,
                verbose=False,
            )

    with h5py.File(out_path, 'r') as f:
        V_CC = np.asarray(f["V_qmunu_CC"][...])
        g0_CC = np.asarray(f["V_qmunu_CC_g0"][...])
    np.testing.assert_allclose(V_CC, V_charge, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(g0_CC, g0_charge, atol=1e-12, rtol=1e-12)
