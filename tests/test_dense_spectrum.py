"""Reference archive guards, independent of any HDF5 payload on disk.

Runtime read/FFT parity belongs to the named compute-node archive pilot.
These cases cover pre-I/O metadata refusal and exact ragged carrier masks.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import h5py
import numpy as np
import pytest

from file_io import dense_spectrum as ds

_ORIGINAL_READ_METADATA = ds._read_metadata


class _Source:
    path = "/bounded/source/WFN.h5"
    nspinor = 1
    ngkmax = 3
    kpoints = np.asarray([[0., 0., 0.], [.5, 0., 0.]])
    kweights = np.asarray([.5, .5])
    kgrid = np.asarray([2, 1, 1])
    occs = np.asarray([[[1., 0.], [1., 0.]]])
    num_electrons = 2.

    def ngk_valid(self, *, k):
        assert k == "ibz"
        return np.asarray([2, 3])

    def get_gvec_nk(self, parent):
        return np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], np.int32)[:parent + 2]


class _IO:
    def __init__(self, path, *, mode, mesh):
        assert mode == "r"
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def archive(monkeypatch):
    source = _Source()
    bindings = {source.path: "0" * 64}
    for name in ("data-file-schema.xml", "charge-density.hdf5", "Si.upf", "run_dense_h.py", "qp_wfn.py"):
        bindings[f"/bounded/source/{name}"] = "1" * 64
    attrs = dict(schema=ds.SCHEMA, finalized=1, complete_native_basis=1,
                 nspinor=1, energy_units="Ry",
                 coefficient_convention="band,spinor,source-QE-G,real-imag",
                 dense_h_source_wfn=source.path,
                 source_sha256_bindings=json.dumps(bindings))
    data = dict(ngk=source.ngk_valid(k="ibz"), basis_dimensions=np.asarray([2, 3]),
                kpoints_crystal=source.kpoints.copy(), kweights=source.kweights.copy(),
                kgrid=source.kgrid.copy(), source_occupations=source.occs.copy(),
                num_electrons=np.asarray(source.num_electrons), io_committed=1)
    parents = [dict(energies_ry=np.arange(parent + 2, dtype=float),
                    gvecs=source.get_gvec_nk(parent), checks=np.zeros(3),
                    coefficient_shape=(parent + 2, 1, parent + 2, 2),
                    coefficient_dtype=np.dtype(np.float64)) for parent in range(2)]
    monkeypatch.setattr(ds, "_read_metadata", lambda path: (attrs, data, parents))
    monkeypatch.setattr(ds, "_sha256", lambda path:
        "3" * 64 if str(Path(path).resolve()) == "/bounded/reference.h5"
        else bindings.get(str(Path(path).resolve()), ""))
    monkeypatch.setattr(ds, "SlabIO", _IO)
    return source, attrs, data, parents


def _open(archive):
    return ds.DenseSpectrumReader("/bounded/reference.h5", archive[0],
                                 mesh=SimpleNamespace(size=4), expected_source_sha256="0" * 64,
                                 expected_archive_sha256="3" * 64)


def test_native_dimensions_survive_and_reader_owns_only_archive(archive):
    with _open(archive) as reader:
        np.testing.assert_array_equal(reader.basis_dimensions, [2, 3])
        assert not reader.basis_dimensions.flags.writeable
        handle = reader._io
    assert handle.closed and reader._io is None
    reader.close()


@pytest.mark.parametrize("attribute", ["finalized", "complete_native_basis"])
def test_incomplete_archive_is_refused_before_transport(archive, attribute):
    archive[1][attribute] = 0
    with pytest.raises(ValueError, match="finalized"):
        _open(archive)


def test_uncommitted_dataset_metadata_refuses(archive):
    archive[2]["io_committed"] = 0
    with pytest.raises(ValueError, match="committed"):
        _open(archive)


def test_changed_seed_binding_and_changed_metadata_refuse(archive):
    with pytest.raises(ValueError, match="SHA256"):
        ds.DenseSpectrumReader("/bounded/reference.h5", archive[0],
            mesh=SimpleNamespace(size=4), expected_source_sha256="2" * 64,
            expected_archive_sha256="3" * 64)
    archive[2]["basis_dimensions"] = np.asarray([2, 4])
    with pytest.raises(ValueError, match="basis_dimensions"):
        _open(archive)


@pytest.mark.parametrize("failure", ["missing_native_band", "wrong_g_order", "nonfinite_energy", "failed_eigenpairs"])
def test_native_parent_negative_controls(archive, failure):
    parent = archive[3][1]
    if failure == "missing_native_band":
        parent["coefficient_shape"] = (2, 1, 3, 2)
    elif failure == "wrong_g_order":
        parent["gvecs"] = parent["gvecs"][::-1]
    elif failure == "nonfinite_energy":
        parent["energies_ry"][0] = np.nan
    else:
        parent["checks"] = np.asarray([0., .01, 0.])
    with pytest.raises(ValueError, match="native parent"):
        _open(archive)


def test_absent_dft_reconstruction_bindings_refuse(archive):
    archive[1]["source_sha256_bindings"] = json.dumps({archive[0].path: "0" * 64})
    with pytest.raises(ValueError, match="density/XML/UPF"):
        _open(archive)


def test_archive_checksum_is_required_even_when_header_is_complete(archive):
    with pytest.raises(ValueError, match="archive SHA256"):
        ds.DenseSpectrumReader("/bounded/reference.h5", archive[0],
            mesh=SimpleNamespace(size=4), expected_source_sha256="0" * 64,
            expected_archive_sha256="4" * 64)


def test_missing_native_group_is_a_named_refusal(archive, monkeypatch):
    def missing(path):
        raise KeyError("k00001/coefficients")
    monkeypatch.setattr(ds, "_read_metadata", missing)
    with pytest.raises(ValueError, match="metadata/groups.*k00001"):
        _open(archive)


@pytest.mark.parametrize("layout", ["attribute_only", "dataset_zero", "wrong_dataset_shape"])
def test_actual_hdf5_commit_layout_negative_controls(tmp_path, layout):
    path = tmp_path / "native_layout.h5"
    with h5py.File(path, "w") as handle:
        # This deliberately reproduces the original faulty synthetic fixture.
        handle.attrs["lorrax_io_committed"] = 1
        if layout == "dataset_zero":
            handle.create_dataset("lorrax_io_committed", data=np.asarray([0], np.int32))
        elif layout == "wrong_dataset_shape":
            handle.create_dataset("lorrax_io_committed", data=np.asarray(1, np.int32))
    with pytest.raises(ValueError, match="commit"):
        ds._read_metadata(path)


def test_actual_hdf5_dataset_receipt_is_read(archive, tmp_path):
    path = tmp_path / "native_complete.h5"
    _, attrs, data, parents = archive
    with h5py.File(path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        handle.create_dataset("lorrax_io_committed", data=np.asarray([1], np.int32))
        for key, value in data.items():
            if key != "io_committed":
                handle.create_dataset(key, data=value)
        for ik, parent in enumerate(parents):
            group = handle.create_group(f"k{ik:05d}")
            for key in ("energies_ry", "gvecs", "checks"):
                group.create_dataset(key, data=parent[key])
            group.create_dataset("coefficients", shape=parent["coefficient_shape"], dtype=np.float64)
    # The fixture stubs metadata at the resource guard. Read this real file
    # with the original metadata owner to exercise the actual disk layout.
    _, actual_data, _ = _ORIGINAL_READ_METADATA(path)
    assert actual_data["io_committed"] == 1


def test_ragged_tile_masks_do_not_invent_high_empty_bands():
    shape, live, valid, local_bytes = ds.native_band_geometry(
        3, 3, 1, 4, 4, 2, 5, pad_to=5, max_local_read_bytes=1024)
    assert shape == (8, 1, 4, 2) and live == 1 and local_bytes == 128
    np.testing.assert_array_equal(valid, [[True, False, False, False, False, False, False, False]])
    _, live, valid, _ = ds.native_band_geometry(
        3, 3, 1, 4, 4, 3, 5, max_local_read_bytes=1024)
    assert live == 0 and not valid.any()


def test_tile_memory_and_native_count_guards_precede_io():
    with pytest.raises(ValueError, match="explicit reference bound"):
        ds.native_band_geometry(100, 100, 1, 100, 4, 0, 100,
                                max_local_read_bytes=64)
    with pytest.raises(ValueError, match="invalid native dimension"):
        ds.native_band_geometry(4, 3, 1, 4, 4, 0, 4,
                                max_local_read_bytes=1024)
