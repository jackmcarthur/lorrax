from types import SimpleNamespace

import numpy as np
import pytest

from gw.four_current_head import FrequencyResolvedFourCurrentHead


def _source(nw=1, storage=4):
    response = FrequencyResolvedFourCurrentHead(
        omega_ry=np.arange(nw, dtype=np.complex128) * 1j,
        Q0_direct=np.zeros((nw, 4, 4), dtype=np.complex128),
        H_linear=np.zeros((nw, 2, 4, 4), dtype=np.complex128),
        S_direct=np.zeros((nw, 2, 2, 4, 4), dtype=np.complex128),
    )
    return SimpleNamespace(
        response=response,
        energy_scaled_d1_raw=np.zeros(
            (2, 4, 1, storage, storage), dtype=np.complex128),
        hamiltonian_config_operator_fingerprint="sha256:" + "1" * 64,
        source_fingerprint="sha256:" + "2" * 64,
        parallel_transport_schema_version=4,
        parallel_transport_polar_rcond=1.0e-10,
        parallel_transport_coefficient_frame="source_pauli_coefficient_frame_v1",
        parallel_transport_derivative_axes=(0, 1),
        wfn_fingerprint="3" * 64,
        band_start=0, band_stop=3, nk_tot=1,
        charge_ward_residual=0.0, ordered_curvature_residual=0.0,
        q2_symmetry_residual=0.0,
    )


def test_four_current_head_writer_has_one_frequency_axis_and_one_provenance(
        tmp_path, monkeypatch):
    import file_io.static_gauge_head as persistence
    import gw.static_gauge_response as response_module

    created, written = {}, {}

    class RecordingSlabIO:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def create_dataset(self, name, *, shape, dtype):
            created[name] = (tuple(shape), np.dtype(dtype))

        def write_slab(self, name, value):
            written[name] = np.asarray(value)

    source = _source()
    monkeypatch.setattr(persistence, "SlabIO", RecordingSlabIO)
    monkeypatch.setattr(
        response_module, "require_photon_head_source",
        lambda value, _mesh: value)
    monkeypatch.setattr(
        persistence, "_publish_completed_partial", lambda *_a, **_k: None)

    persistence.write_frequency_resolved_four_current_head(
        tmp_path / "head.h5", source, mesh_xy=object())

    assert set(created) == {
        "omega_ry", "Q0_direct", "H_linear", "S_direct",
        "energy_scaled_d1_raw",
        "provenance_utf8_i32",
    }
    assert created["energy_scaled_d1_raw"][0] == (2, 4, 1, 3, 3)
    provenance = persistence._decode_i32_text(
        written["provenance_utf8_i32"], field_name="provenance",
        encoding="utf-8")
    assert "frequency_resolved_four_current_head" in provenance
    assert "response_sample" not in provenance
    assert "role" not in provenance
    assert "availability" not in provenance


def test_four_current_head_writer_refuses_noncausal_dynamic_rows(
        tmp_path, monkeypatch):
    import file_io.static_gauge_head as persistence
    import gw.static_gauge_response as response_module

    monkeypatch.setattr(
        response_module, "require_photon_head_source",
        lambda value, _mesh: value)
    with pytest.raises(ValueError, match="causal response kernel"):
        persistence.write_frequency_resolved_four_current_head(
            tmp_path / "head.h5", _source(nw=2), mesh_xy=object())
