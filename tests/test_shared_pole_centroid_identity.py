"""Store guards bind off-grid scientific coordinates independently of padding."""
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest


POINTS = np.asarray([[.1234567, .27, .39], [.6234567, .73, .89]])


def _basis(points, kind="fractional", *, typed=True):
    fields = dict(canonical_indices=np.asarray(points), n_logical=len(points),
                  mesh_xy=SimpleNamespace(shape={"x": 1, "y": 1}))
    if typed:
        fields["coordinate_kind"] = kind
    return SimpleNamespace(**fields)


def _header(basis):
    """Real metadata owner on a tiny identity/TR q table, no tensor payload."""
    from file_io import shared_pole_store as store

    n = basis.n_logical
    qt = store.QirrTables(
        irr_idx_q=np.zeros(1, np.int32), sym_idx_q=np.zeros(1, np.int32),
        q_irr_frac=np.zeros((1, 3)), sym_perm=np.tile(np.arange(n), (2, 1)),
        L_table=np.zeros((2, n, 3), np.int32), n_sym_spatial=1)
    sym = SimpleNamespace(
        trs_allowed=True, active_symmetry_rows=np.arange(2, dtype=np.int32),
        operation_typing_source="centroid guard fixture",
        operation_rows=lambda rows: (np.tile(np.eye(3, dtype=np.int32), (len(rows), 1, 1)),
                                     np.zeros((len(rows), 3)), rows >= 1),
        spinor_action=lambda rows, nspinor: np.ones((len(rows), nspinor, nspinor), complex))
    meta = SimpleNamespace(mu_basis=basis, nspinor=2, nspinor_wfnfile=2,
                           nkx=1, nky=1, nkz=1, fft_grid=(8, 8, 8))
    identity = {name: "fixture" for name in store._IDENTITY_KEYS}
    tables = dict(qirr=qt, q_irr_full_idx=np.zeros(1, np.int32), sym=sym)
    header = store._metadata(meta, tables, {"version": "centroid-guard-fixture"},
                             identity, ordered=False)
    return meta, header


def test_fractional_metadata_uses_the_canonical_coordinate_owner():
    from file_io import shared_pole_store as store
    from file_io.wfn_basis import centroid_table_md5, centroid_table_fingerprint_scheme

    meta, header = _header(_basis(POINTS))
    assert header["centroid_digest"] == centroid_table_md5(POINTS, coordinate_kind="fractional")
    assert header["centroid_coordinate_kind"] == "fractional"
    assert header["centroid_fingerprint_scheme"] == centroid_table_fingerprint_scheme("fractional")
    assert store._check_basis(meta, header) is meta.mu_basis
    # Endianness and equivalent host dtype do not invent a new physical table.
    equivalent = _basis(POINTS.astype(">f8"))
    assert store._check_basis(meta, header, equivalent) is equivalent


@pytest.mark.parametrize("change", ["subgrid_move", "reordered"])
def test_same_integer_cell_fractional_change_refuses(change):
    from file_io import shared_pole_store as store

    meta, header = _header(_basis(POINTS))
    moved = POINTS.copy()
    if change == "subgrid_move":
        moved[0, 0] += 1e-7
    else:
        moved = moved[::-1]
    # This is the precise collision of the old store guard, not a shape change.
    assert np.array_equal(POINTS.astype("<i4"), moved.astype("<i4"))
    assert _header(_basis(moved))[1]["centroid_digest"] != header["centroid_digest"]
    with pytest.raises(ValueError, match="centroid or spin identity changed"):
        store._check_basis(meta, header, _basis(moved))


@pytest.mark.parametrize("typed", [False, True])
def test_legacy_fft_header_and_sha256_bytes_are_unchanged(typed):
    from file_io import shared_pole_store as store

    points = np.asarray([[0, 1, 2], [3, 4, 5]], np.int64)
    basis = _basis(points, "fft_indices", typed=typed)
    meta, header = _header(basis)
    digest = hashlib.sha256(np.asarray(points, dtype="<i4").tobytes()).hexdigest()
    assert store._centroid_basis_metadata(basis) == {"centroid_digest": digest}
    assert header["centroid_digest"] == digest
    assert "centroid_coordinate_kind" not in header
    assert "centroid_fingerprint_scheme" not in header
    assert store._check_basis(meta, header) is basis


@pytest.mark.parametrize("stored_kind", ["fft_indices", "fractional"])
def test_equal_numeric_coordinates_with_a_changed_kind_refuse(stored_kind):
    from file_io import shared_pole_store as store

    points = np.asarray([[0, 0, 0], [1, 1, 1]])
    meta, header = _header(_basis(points, stored_kind))
    other = "fractional" if stored_kind == "fft_indices" else "fft_indices"
    with pytest.raises(ValueError, match="coordinate kind changed"):
        store._check_basis(meta, header, _basis(points, other))


def test_old_integer_cast_fractional_header_has_no_implicit_fallback():
    from file_io import shared_pole_store as store

    meta, header = _header(_basis(POINTS))
    header.pop("centroid_coordinate_kind")
    header.pop("centroid_fingerprint_scheme")
    header["centroid_digest"] = hashlib.sha256(POINTS.astype("<i4").tobytes()).hexdigest()
    with pytest.raises(ValueError, match="coordinate kind changed"):
        store._check_basis(meta, header)


@pytest.mark.parametrize("scheme", [None, "int64-c-order-md5-v1"])
def test_fractional_header_requires_the_precise_scheme(scheme):
    from file_io import shared_pole_store as store

    meta, header = _header(_basis(POINTS))
    if scheme is None:
        header.pop("centroid_fingerprint_scheme")
    else:
        header["centroid_fingerprint_scheme"] = scheme
    with pytest.raises(ValueError, match="centroid or spin identity changed"):
        store._check_basis(meta, header)


def test_factor_reader_refuses_wrong_fractional_basis_before_capacity_or_payload(monkeypatch):
    from file_io import shared_pole_store as store

    meta, header = _header(_basis(POINTS))
    header["finalized"] = True
    def forbidden(*args, **kwargs):
        raise AssertionError("wrong-point reader reached tensor/capacity work")
    monkeypatch.setattr(store, "_capacity", forbidden)
    moved = POINTS.copy(); moved[0, 1] += 1e-7
    with pytest.raises(ValueError, match="centroid or spin identity changed"):
        store.read_shared_pole_faces(SimpleNamespace(read_slab=forbidden), (0, 1),
                                     meta=meta, header=header, basis=_basis(moved))


def test_legacy_fft_photon_metadata_fields_are_unchanged():
    from file_io import shared_pole_store as store

    points = np.asarray([[0, 1, 2], [3, 4, 5]], np.int32)
    bases = (_basis(points, "fft_indices"), _basis(points + 1, "fft_indices"))
    expected = {"photon_centroid_digests": [hashlib.sha256(
        b.canonical_indices.astype("<i4").tobytes()).hexdigest() for b in bases]}
    assert store._photon_centroid_metadata(bases) == expected
    store.check_photon_centroid_bases(expected, bases)


@pytest.mark.parametrize("endpoint", [0, 1])
def test_photon_fractional_endpoint_move_refuses(endpoint):
    from file_io import shared_pole_store as store

    bases = (_basis(POINTS), _basis(POINTS + .03))
    header = store._photon_centroid_metadata(bases)
    store.check_photon_centroid_bases(header, bases)
    moved = bases[endpoint].canonical_indices.copy(); moved[0, 2] += 1e-7
    assert np.array_equal(moved.astype("<i4"), bases[endpoint].canonical_indices.astype("<i4"))
    changed = list(bases); changed[endpoint] = _basis(moved)
    with pytest.raises(ValueError, match="centroid endpoint identity changed"):
        store.check_photon_centroid_bases(header, changed)


def test_mixed_photon_endpoint_kind_and_missing_fractional_typing_refuse():
    from file_io import shared_pole_store as store

    charge = _basis([[0, 0, 0], [1, 1, 1]], "fft_indices")
    current = _basis(POINTS)
    header = store._photon_centroid_metadata((charge, current))
    assert len(header["photon_centroid_identities"]) == 2
    store.check_photon_centroid_bases(header, (charge, current))
    changed = (charge, _basis(POINTS.astype(np.int32), "fft_indices"))
    with pytest.raises(ValueError, match="centroid endpoint identity changed"):
        store.check_photon_centroid_bases(header, changed)
    header.pop("photon_centroid_identities")
    with pytest.raises(ValueError, match="centroid endpoint identity changed"):
        store.check_photon_centroid_bases(header, (charge, current))


def test_actual_photon_bank_reader_refuses_before_coulomb_payload(monkeypatch):
    from file_io import shared_pole_store as store
    from gw import response_bank

    bases = (_basis(POINTS), _basis(POINTS + .03))
    header = dict(store._photon_centroid_metadata(bases),
                  photon_layout={"packed_extent": 8})
    monkeypatch.setattr(store, "validate_shared_pole_bank", lambda *args, **kwargs: header)
    def forbidden(*args, **kwargs):
        raise AssertionError("wrong-point photon bank reached the Coulomb payload")
    monkeypatch.setattr(response_bank, "resource_digest", forbidden)
    moved = POINTS.copy(); moved[0, 0] += 1e-7
    with pytest.raises(ValueError, match="centroid endpoint identity changed"):
        response_bank.compute_photon_bank(None, None, None, None,
            mesh_xy=None, sym=None, mu_bases=(_basis(moved), bases[1]),
            layout=SimpleNamespace(packed_extent=8), occupation_state=None,
            sample_plan=None, bank_io={"path": "no-tensor-read", "identity": {}})
