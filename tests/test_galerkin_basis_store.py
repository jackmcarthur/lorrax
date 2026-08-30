"""Focused collective lifecycle gate for ``isdf.galerkin.GalerkinBasis``."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax as _jax_boot
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    kwargs = {"local_device_ids": [0]} if visible and "," not in visible else {}
    _jax_boot.distributed.initialize(**kwargs)

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.sharding_fit import fit_sharding
from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME
from isdf.galerkin import GalerkinBasis, read_galerkin_basis, write_galerkin_basis


def _mesh() -> Mesh:
    devices = jax.devices()
    count = 4 if len(devices) >= 4 else 1
    return Mesh(np.asarray(devices[:count]).reshape(int(np.sqrt(count)), -1),
                ("x", "y"))


def _path(tmp_path: Path) -> Path:
    if jax.process_count() == 1:
        return tmp_path / "basis.h5"
    step = os.environ["SLURM_STEP_ID"].replace(".", "-")
    root = Path("/tmp") / f"lorrax-galerkin-basis-{step}"
    root.mkdir(parents=True, exist_ok=True)
    return root / "basis.h5"


def _host(value) -> np.ndarray:
    if jax.process_count() == 1:
        return np.asarray(value)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


def _fixture(mesh: Mesh):
    nk, nb, ns, nmu, physical, carrier = 2, 3, 2, 6, 3, 4
    c = (np.arange(nk * nb * physical).reshape(nk, nb, physical)
         + 0.125j).astype(np.complex128)
    nodes = (np.arange(physical * ns * nmu).reshape(physical, ns, nmu)
             - 0.25j).astype(np.complex128)
    factor = np.tril((np.arange(physical ** 2).reshape(physical, physical)
                      + 1 + 0.5j).astype(np.complex128))
    cpad = np.pad(c, ((0, 0), (0, 0), (0, carrier - physical)))
    npad = np.pad(nodes, ((0, carrier - physical), (0, 0), (0, 0)))
    fpad = np.eye(carrier, dtype=np.complex128)
    fpad[:physical, :physical] = factor
    fpad[:physical, physical:] = 0
    selected = (0, 2, 5)
    pivot = hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest()
    rep = NamedSharding(mesh, P())
    nsh = fit_sharding(mesh, P(None, None, "y"), npad.shape,
                       "test.galerkin.nodes")
    basis = GalerkinBasis(
        ctilde=jax.device_put(cpad, rep),
        basis_at_nodes=jax.device_put(npad, nsh), rank_physical=physical,
        band_range=(4, 7), selected_state_indices=selected,
        selection_factor=jax.device_put(fpad, rep), qrcp_seed=7,
        qrcp_eps=1e-3, qrcp_raw_rank=3, qrcp_search_rank=6,
        candidate_hash="b" * 64, pivot_hash=pivot)
    meta = SimpleNamespace(
        nk_tot=nk, nspinor=ns, fft_grid=(3, 3, 3), kgrid=(2, 1, 1))
    centroids = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [0, 2, 1], [2, 2, 2],
         [1, 1, 2]],
        dtype=np.int32)
    return basis, meta, centroids, (c, nodes, factor)


def _kwargs(mesh, meta, centroids):
    return dict(wfn=object(), meta=meta, centroid_indices=centroids,
                bispinor=True, rank_multiplier=20.0, qrcp_eps=1e-3,
                qrcp_seed=7, mesh_xy=mesh)


@pytest.mark.mesh(4)
def test_galerkin_basis_logical_roundtrip_and_provenance(tmp_path, monkeypatch):
    import common.parallel_transport as pt
    monkeypatch.setattr(pt, "wfn_fingerprint", lambda _wfn: "a" * 64)
    mesh = _mesh()
    basis, meta, centroids, physical = _fixture(mesh)
    path = _path(tmp_path)
    write_galerkin_basis(path, basis, **_kwargs(mesh, meta, centroids))
    with pytest.raises(FileExistsError, match="immutable Galerkin"):
        write_galerkin_basis(path, basis, **_kwargs(mesh, meta, centroids))
    with h5py.File(path, "r") as h5:
        assert h5["galerkin_ctilde"].shape == (2, 3, 3)
        assert h5["galerkin_basis_at_nodes"].shape == (3, 2, 6)
        assert h5["galerkin_selection_factor"].shape == (3, 3)
        assert int(np.asarray(h5["galerkin_complete"])[0]) == 1
        scheme = bytes(np.asarray(
            h5["galerkin_transform_scheme"], dtype=np.uint8)).decode()
        assert scheme == FULL_BLOCH_TRANSFORM_SCHEME

    restored = read_galerkin_basis(
        path, band_range=(4, 7), extra_rank_pad=2,
        **_kwargs(mesh, meta, centroids))
    carrier = 6 if mesh.size == 4 else 5
    arrays = tuple(_host(value) for value in (
        restored.ctilde, restored.basis_at_nodes, restored.selection_factor))
    np.testing.assert_array_equal(arrays[0][..., :3], physical[0])
    np.testing.assert_array_equal(arrays[1][:3], physical[1])
    np.testing.assert_array_equal(arrays[2][:3, :3], physical[2])
    assert np.all(arrays[0][..., 3:] == 0) and np.all(arrays[1][3:] == 0)
    np.testing.assert_array_equal(arrays[2][3:, 3:], np.eye(carrier - 3))
    changed = centroids.copy()
    changed[2, 1] += 1
    with pytest.raises(ValueError, match="centroid_hash"):
        read_galerkin_basis(
            path, band_range=(4, 7), **_kwargs(mesh, meta, changed))
    with h5py.File(path, "r+") as h5:
        encoded = np.frombuffer(
            "pre-paired-k-g", dtype=np.uint8).astype(np.int32)
        del h5["galerkin_transform_scheme"]
        h5.create_dataset("galerkin_transform_scheme", data=encoded)
    with pytest.raises(ValueError, match="transform_scheme"):
        read_galerkin_basis(
            path, band_range=(4, 7), **_kwargs(mesh, meta, centroids))
