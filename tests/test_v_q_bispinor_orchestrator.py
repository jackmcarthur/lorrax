"""End-to-end smoke test for :func:`gw.v_q_bispinor.compute_V_q_bispinor_to_h5`.

GPU-required (the underlying ``compute_V_q_tile`` does not support a CPU
backend).  We exercise the orchestrator on a tiny synthetic system —
mock ζ_C and ζ_T (μ_L=0..3) HDF5 files with random data of the right
layout — to verify that:

* All 7 unique tiles appear on disk under the expected dataset names.
* :class:`BispinorVqReader` returns:
    - direct reads for unique tiles,
    - ``conj(swapaxes(.., -1, -2))`` for Hermitian-redundant tiles,
    - zeros (sized correctly) for gauge-zero tiles.
* For tile (0, 0) the kernel result is bit-identical to a direct call
  to :func:`compute_V_q_tile` with the scalar charge v(K) — i.e. the
  bispinor orchestrator's CC tile is literally the existing scalar
  V_q.  This is the **thinness** invariant the user asked for.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pytest
import h5py

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _has_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def _write_mock_zeta(path: Path, shape, *, mesh, seed: int) -> None:
    """Write a small random ζ tensor to ``path`` with shape ``(nq, n_rtot, n_rmu)``.

    Layout matches what ``compute_V_q_tile`` expects from a SlabIO open
    in 'r' mode: full tensor in one dataset named ``'zeta_q'``, sharded
    across (None, None, ('x','y')).
    """
    from file_io.slab_io import SlabIO
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("zeta_q", shape=tuple(shape), dtype=np.complex128)
        io.write_slab("zeta_q", jnp.asarray(arr), global_shape=tuple(shape))


@pytest.fixture
def tiny_system(tmp_path):
    """A 2×2×1 k-grid, 1×1 mesh, 8×8×8 FFT, n_rmu=4 (charge) / 4 (transverse)."""
    if not _has_gpu():
        pytest.skip("GPU required for compute_V_q_tile")
    mesh = Mesh(np.array(jax.devices("gpu")[:1]).reshape(1, 1), ('x', 'y'))
    kgrid = (2, 2, 1)
    fft_grid = (8, 8, 8)
    n_rmu_C = 4
    n_rmu_T = 4
    n_rtot = int(np.prod(fft_grid))
    nq = int(np.prod(kgrid))
    bvec = np.diag([1.5, 1.5, 0.7]).astype(np.float64)
    cell_volume = float(np.linalg.det(bvec)) * 2 * np.pi
    # Mock ζ files: charge + 3 transverse channels.
    zeta_C_path = tmp_path / "zeta_C.h5"
    zeta_T_paths = tuple(tmp_path / f"zeta_T_{i}.h5" for i in (1, 2, 3))
    for path, n_rmu, seed in [
        (zeta_C_path, n_rmu_C, 1),
        (zeta_T_paths[0], n_rmu_T, 11),
        (zeta_T_paths[1], n_rmu_T, 12),
        (zeta_T_paths[2], n_rmu_T, 13),
    ]:
        _write_mock_zeta(path, (nq, n_rtot, n_rmu), mesh=mesh, seed=seed)

    # Reading-side coulomb kernel — same factory the scalar V_q uses.
    # Set vcoul_cutoff_ry so the factory builds a sphere index (the
    # kernel requires flat 1-D v_per_G; without a cutoff the factory
    # returns the full FFT-box (nx,ny,nz) which the kernel can't
    # broadcast against).  Production always sets this via cfg.bare_coulomb_cutoff.
    from gw.compute_vcoul import make_v_munu_chunked_kernel
    kernels = make_v_munu_chunked_kernel(
        *fft_grid, *kgrid, bvec, cell_volume, sys_dim=3,
        mc_average_vcoul_body=False,
        vcoul_cutoff_ry=200.0,    # large enough to encompass our 8³ box
    )
    return dict(
        mesh=mesh, kgrid=kgrid, fft_grid=fft_grid,
        n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T, n_rtot=n_rtot, nq=nq,
        bvec=bvec, cell_volume=cell_volume,
        zeta_C_path=zeta_C_path, zeta_T_paths=zeta_T_paths,
        coulomb_kernels=kernels, tmp_path=tmp_path,
    )


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
def test_orchestrator_writes_seven_tiles(tiny_system):
    """All 7 unique tiles must be present in the output HDF5."""
    from file_io.slab_io import SlabIO
    from gw.v_q_bispinor import (
        compute_V_q_bispinor_to_h5, UNIQUE_TILES, tile_dataset_name,
    )

    sysdict = tiny_system
    out_path = sysdict["tmp_path"] / "v_q_bispinor.h5"
    with SlabIO(sysdict["zeta_C_path"], mode="r", mesh=sysdict["mesh"]) as zc, \
         SlabIO(sysdict["zeta_T_paths"][0], mode="r", mesh=sysdict["mesh"]) as zt1, \
         SlabIO(sysdict["zeta_T_paths"][1], mode="r", mesh=sysdict["mesh"]) as zt2, \
         SlabIO(sysdict["zeta_T_paths"][2], mode="r", mesh=sysdict["mesh"]) as zt3:
        compute_V_q_bispinor_to_h5(
            zeta_C_io=zc,
            zeta_T_ios=(zt1, zt2, zt3),
            output_h5_path=out_path,
            coulomb_kernels=sysdict["coulomb_kernels"],
            mesh_xy=sysdict["mesh"],
            kgrid=sysdict["kgrid"],
            fft_grid=sysdict["fft_grid"],
            bvec=sysdict["bvec"],
            n_rmu_C=sysdict["n_rmu_C"],
            n_rmu_T=sysdict["n_rmu_T"],
            budget_bytes=4e9,            # plenty for tiny system
            verbose=False,
        )

    with h5py.File(out_path, "r") as f:
        for (mu_L, nu_L) in UNIQUE_TILES:
            name = tile_dataset_name(mu_L, nu_L)
            assert name in f, f"missing dataset {name}"
            ds = f[name]
            n_L = sysdict["n_rmu_C"] if mu_L == 0 else sysdict["n_rmu_T"]
            n_R = sysdict["n_rmu_C"] if nu_L == 0 else sysdict["n_rmu_T"]
            assert ds.shape == (sysdict["nq"], n_L, n_R), (
                f"{name} shape {ds.shape} ≠ expected "
                f"({sysdict['nq']}, {n_L}, {n_R})"
            )
        # CC g0 head must be present; TT tiles must NOT have a g0 dataset.
        assert "V_qmunu_CC_g0" in f
        for (mu_L, nu_L) in UNIQUE_TILES:
            if (mu_L, nu_L) == (0, 0):
                continue
            assert f"{tile_dataset_name(mu_L, nu_L)}_g0" not in f, (
                f"unexpected g0 dataset for TT tile ({mu_L}, {nu_L})"
            )


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
def test_reader_returns_zero_for_gauge_tiles(tiny_system):
    from file_io.slab_io import SlabIO
    from gw.v_q_bispinor import (
        compute_V_q_bispinor_to_h5, BispinorVqReader, ZERO_TILES,
    )

    sysdict = tiny_system
    out_path = sysdict["tmp_path"] / "v_q_bispinor.h5"
    with SlabIO(sysdict["zeta_C_path"], mode="r", mesh=sysdict["mesh"]) as zc, \
         SlabIO(sysdict["zeta_T_paths"][0], mode="r", mesh=sysdict["mesh"]) as zt1, \
         SlabIO(sysdict["zeta_T_paths"][1], mode="r", mesh=sysdict["mesh"]) as zt2, \
         SlabIO(sysdict["zeta_T_paths"][2], mode="r", mesh=sysdict["mesh"]) as zt3:
        compute_V_q_bispinor_to_h5(
            zeta_C_io=zc, zeta_T_ios=(zt1, zt2, zt3),
            output_h5_path=out_path,
            coulomb_kernels=sysdict["coulomb_kernels"],
            mesh_xy=sysdict["mesh"], kgrid=sysdict["kgrid"],
            fft_grid=sysdict["fft_grid"], bvec=sysdict["bvec"],
            n_rmu_C=sysdict["n_rmu_C"], n_rmu_T=sysdict["n_rmu_T"],
            budget_bytes=4e9, verbose=False,
        )

    with BispinorVqReader(out_path, sysdict["mesh"]) as r:
        for (mu_L, nu_L) in ZERO_TILES:
            V = np.asarray(r.get_tile(mu_L, nu_L))
            assert V.shape == (sysdict["nq"],
                               sysdict["n_rmu_C"] if mu_L == 0 else sysdict["n_rmu_T"],
                               sysdict["n_rmu_C"] if nu_L == 0 else sysdict["n_rmu_T"])
            np.testing.assert_array_equal(V, 0.0)


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
def test_reader_hermitian_pair_matches_companion_transpose(tiny_system):
    """``reader.get_tile(j, i)`` should equal
    ``conj(swapaxes(reader.get_tile(i, j), -1, -2))`` for the redundant
    pairs."""
    from file_io.slab_io import SlabIO
    from gw.v_q_bispinor import (
        compute_V_q_bispinor_to_h5, BispinorVqReader, HERMITIAN_PAIRS,
    )

    sysdict = tiny_system
    out_path = sysdict["tmp_path"] / "v_q_bispinor.h5"
    with SlabIO(sysdict["zeta_C_path"], mode="r", mesh=sysdict["mesh"]) as zc, \
         SlabIO(sysdict["zeta_T_paths"][0], mode="r", mesh=sysdict["mesh"]) as zt1, \
         SlabIO(sysdict["zeta_T_paths"][1], mode="r", mesh=sysdict["mesh"]) as zt2, \
         SlabIO(sysdict["zeta_T_paths"][2], mode="r", mesh=sysdict["mesh"]) as zt3:
        compute_V_q_bispinor_to_h5(
            zeta_C_io=zc, zeta_T_ios=(zt1, zt2, zt3),
            output_h5_path=out_path,
            coulomb_kernels=sysdict["coulomb_kernels"],
            mesh_xy=sysdict["mesh"], kgrid=sysdict["kgrid"],
            fft_grid=sysdict["fft_grid"], bvec=sysdict["bvec"],
            n_rmu_C=sysdict["n_rmu_C"], n_rmu_T=sysdict["n_rmu_T"],
            budget_bytes=4e9, verbose=False,
        )

    with BispinorVqReader(out_path, sysdict["mesh"]) as r:
        for redundant, companion in HERMITIAN_PAIRS.items():
            V_red = np.asarray(r.get_tile(*redundant))
            V_com = np.asarray(r.get_tile(*companion))
            np.testing.assert_allclose(
                V_red, np.conj(np.swapaxes(V_com, -1, -2)),
                atol=1e-12,
                err_msg=f"Hermitian relation failed for "
                        f"{redundant} vs {companion}",
            )


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
def test_cc_tile_matches_direct_compute_V_q_tile(tiny_system):
    """The orchestrator's CC tile must equal a direct call to
    ``compute_V_q_tile`` with the scalar charge v(K).  This is the
    **thinness** invariant: bispinor adds no logic to the CC path."""
    from file_io.slab_io import SlabIO
    from gw.v_q_bispinor import (
        compute_V_q_bispinor_to_h5, BispinorVqReader,
    )
    from gw.v_q_tile import compute_V_q_tile
    from gw.compute_vcoul import compute_all_V_q

    sysdict = tiny_system
    out_path = sysdict["tmp_path"] / "v_q_bispinor.h5"
    with SlabIO(sysdict["zeta_C_path"], mode="r", mesh=sysdict["mesh"]) as zc, \
         SlabIO(sysdict["zeta_T_paths"][0], mode="r", mesh=sysdict["mesh"]) as zt1, \
         SlabIO(sysdict["zeta_T_paths"][1], mode="r", mesh=sysdict["mesh"]) as zt2, \
         SlabIO(sysdict["zeta_T_paths"][2], mode="r", mesh=sysdict["mesh"]) as zt3:
        compute_V_q_bispinor_to_h5(
            zeta_C_io=zc, zeta_T_ios=(zt1, zt2, zt3),
            output_h5_path=out_path,
            coulomb_kernels=sysdict["coulomb_kernels"],
            mesh_xy=sysdict["mesh"], kgrid=sysdict["kgrid"],
            fft_grid=sysdict["fft_grid"], bvec=sysdict["bvec"],
            n_rmu_C=sysdict["n_rmu_C"], n_rmu_T=sysdict["n_rmu_T"],
            budget_bytes=4e9, verbose=False,
        )

    # Direct scalar V_q on the SAME zeta_C: must produce the same array.
    with SlabIO(sysdict["zeta_C_path"], mode="r", mesh=sysdict["mesh"]) as zc:
        V_scalar, g0_scalar = compute_all_V_q(
            zeta_io=zc,
            kgrid=sysdict["kgrid"], fft_grid=sysdict["fft_grid"],
            bvec=sysdict["bvec"], cell_volume=sysdict["cell_volume"],
            mesh_xy=sysdict["mesh"],
            n_rmu=sysdict["n_rmu_C"], n_rtot=sysdict["n_rtot"],
            sys_dim=3, mc_average_vcoul_body=False,
            budget_bytes=4e9, verbose=False,
        )

    with BispinorVqReader(out_path, sysdict["mesh"]) as r:
        V_cc = np.asarray(r.get_tile(0, 0))

    np.testing.assert_allclose(
        V_cc, np.asarray(V_scalar),
        atol=1e-10, rtol=1e-10,
        err_msg="bispinor CC tile diverges from scalar V_q — orchestrator "
                "is doing more than identity on the charge channel.",
    )
