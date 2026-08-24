"""Mesh-independent SlabIO restart contract for ``GalerkinBasis``."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

# The focused P4 gate is four processes with one visible GPU each.  JAX must
# join them before any module below touches its backend; ordinary P1 pytest
# runs retain their existing single-process startup.
os.environ.setdefault("JAX_ENABLE_X64", "1")
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax as _jax_boot

    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _kwargs = {"local_device_ids": [0]} if _visible and "," not in _visible else {}
    _jax_boot.distributed.initialize(**_kwargs)

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.sharding_fit import fit_sharding
from file_io.galerkin_basis import (
    BASIS_DATASET,
    CTILDE_DATASET,
    PROJECTOR_DATASET,
    GalerkinBasisStamp,
    read_galerkin_basis,
    write_galerkin_basis,
)
from file_io.slab_io import SlabIO
from isdf.galerkin import GalerkinBasis


def _mesh() -> Mesh:
    devices = jax.devices()
    if len(devices) >= 4:
        return Mesh(np.asarray(devices[:4]).reshape(2, 2), ("x", "y"))
    return Mesh(np.asarray(devices[:1]).reshape(1, 1), ("x", "y"))


def _artifact_path(tmp_path, name: str) -> Path:
    """One shared path in a multi-process step; pytest tmp paths at P1."""
    if jax.process_count() == 1:
        return tmp_path / name
    step = os.environ.get("SLURM_STEP_ID", "unknown").replace(".", "-")
    root = Path("/tmp") / f"lorrax-galerkin-basis-{step}"
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def _host(value) -> np.ndarray:
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(value))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


def _fixture(mesh: Mesh):
    nk, nb, ns, n_nodes = 2, 3, 2, 5
    rank_physical, rank_carrier = 3, 4
    ctilde_physical = (
        np.arange(nk * nb * rank_physical, dtype=np.float64).reshape(
            nk, nb, rank_physical)
        + 1j * 0.125
    ).astype(np.complex128)
    basis_physical = (
        np.arange(rank_physical * ns * n_nodes, dtype=np.float64).reshape(
            rank_physical, ns, n_nodes)
        - 1j * 0.25
    ).astype(np.complex128)
    projector_physical = (
        np.arange(rank_physical * nk * nb, dtype=np.float64).reshape(
            rank_physical, nk * nb)
        + 1j * 0.5
    ).astype(np.complex128)

    ctilde_carrier = np.pad(
        ctilde_physical, ((0, 0), (0, 0), (0, 1)))
    basis_carrier = np.pad(basis_physical, ((0, 1), (0, 0), (0, 0)))
    projector_carrier = np.pad(projector_physical, ((0, 1), (0, 0)))
    rep = NamedSharding(mesh, P())
    node_sharding = fit_sharding(
        mesh, P(None, None, "y"), basis_carrier.shape,
        "test.galerkin_basis_store(node-axis)")
    basis = GalerkinBasis(
        ctilde=jax.device_put(ctilde_carrier, rep),
        basis_at_nodes=jax.device_put(basis_carrier, node_sharding),
        projector=jax.device_put(projector_carrier, rep),
        rank_physical=rank_physical,
        band_range=(4, 7),
    )
    stamp = GalerkinBasisStamp(
        band_range=(4, 7),
        nk=nk,
        nb=nb,
        nspinor=ns,
        fft_grid=(3, 3, 3),
        kgrid=(2, 1, 1),
        bispinor=True,
        centroid_indices=np.asarray(
            [[0, 0, 0], [1, 0, 0], [2, 1, 0], [0, 2, 1], [2, 2, 2]],
            dtype=np.int32),
        wfn_path="/synthetic/WFN.h5",
        wfn_fingerprint="a" * 64,
        rtol=1.0e-8,
        rank_multiplier=0.0,
    )
    physical = (ctilde_physical, basis_physical, projector_physical)
    return basis, stamp, physical


@pytest.mark.mesh(4)
def test_galerkin_basis_roundtrip_stores_logical_extents_and_repads(tmp_path):
    mesh = _mesh()
    basis, stamp, physical = _fixture(mesh)
    path = _artifact_path(tmp_path, "galerkin_basis.h5")

    write_galerkin_basis(path, basis, stamp, mesh_xy=mesh)

    # Serial inspection happens only after the collective SlabIO owner closed.
    with h5py.File(path, "r") as h5:
        assert h5[CTILDE_DATASET].shape == (2, 3, 3)
        assert h5[BASIS_DATASET].shape == (3, 2, 5)
        assert h5[PROJECTOR_DATASET].shape == (3, 6)
        assert int(np.asarray(h5["galerkin_complete"])[0]) == 1
        np.testing.assert_array_equal(
            np.asarray(h5["galerkin_centroid_indices"]),
            stamp.centroid_indices)

    # P4: physical rank 3 -> mesh rank 4, then the two-row test pad -> 6.
    restored = read_galerkin_basis(
        path,
        mesh_xy=mesh,
        expected=stamp,
        require_projector=True,
        extra_rank_pad=2,
    )
    expected_carrier = 6 if mesh.size == 4 else 5
    assert restored.rank_physical == 3
    assert restored.rank_carrier == expected_carrier
    arrays = tuple(_host(value) for value in (
        restored.ctilde, restored.basis_at_nodes, restored.projector))
    np.testing.assert_array_equal(arrays[0][..., :3], physical[0])
    np.testing.assert_array_equal(arrays[1][:3], physical[1])
    np.testing.assert_array_equal(arrays[2][:3], physical[2])
    assert np.all(arrays[0][..., 3:] == 0)
    assert np.all(arrays[1][3:] == 0)
    assert np.all(arrays[2][3:] == 0)


@pytest.mark.mesh(4)
def test_galerkin_basis_refuses_mismatch_and_incomplete_payload(tmp_path):
    mesh = _mesh()
    basis, stamp, _ = _fixture(mesh)
    path = _artifact_path(tmp_path, "mismatch_galerkin_basis.h5")
    write_galerkin_basis(path, basis, stamp, mesh_xy=mesh)

    changed_centroids = stamp.centroid_indices.copy()
    changed_centroids[2, 1] += 1
    with pytest.raises(ValueError, match="centroid_indices"):
        read_galerkin_basis(
            path,
            mesh_xy=mesh,
            expected=replace(stamp, centroid_indices=changed_centroids),
        )
    with pytest.raises(ValueError, match="wfn_fingerprint"):
        read_galerkin_basis(
            path,
            mesh_xy=mesh,
            expected=replace(stamp, wfn_fingerprint="b" * 64),
        )

    partial = _artifact_path(tmp_path, "partial_galerkin_basis.h5")
    with SlabIO(partial, mode="w", mesh=mesh) as io:
        io.create_dataset(
            CTILDE_DATASET, shape=(2, 3, 3), dtype=np.complex128)
        io.create_dataset(
            BASIS_DATASET, shape=(3, 2, 5), dtype=np.complex128)
    with pytest.raises(ValueError, match="completion stamp"):
        read_galerkin_basis(
            partial, mesh_xy=mesh, expected=stamp)
