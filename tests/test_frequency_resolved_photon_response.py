import numpy as np
from gw.four_current_head import FrequencyResolvedFourCurrentHead


def test_photon_source_fingerprint_is_input_lineage_only():
    import inspect
    from gw.static_gauge_response import _source_fingerprint

    common = dict(
        operator_fingerprint="sha256:" + "1" * 64,
        wfn_fingerprint="2" * 64,
        band_start=0,
        band_stop=4,
        nk_tot=1,
        parallel_transport_schema_version=4,
        parallel_transport_polar_rcond=1.0e-10,
        parallel_transport_coefficient_frame=(
            "source_pauli_coefficient_frame_v1"),
        parallel_transport_derivative_axes=(0, 1),
    )
    assert "response" not in inspect.signature(_source_fingerprint).parameters
    baseline = _source_fingerprint(**common)
    changed = dict(common, band_stop=5)
    assert _source_fingerprint(**changed) != baseline
    changed = dict(common, parallel_transport_derivative_axes=(1, 0))
    assert _source_fingerprint(**changed) != baseline


def test_frequency_lookup_requires_exact_stored_value():
    try:
        FrequencyResolvedFourCurrentHead(
            omega_ry=np.asarray((0.0 + 0.0j,), dtype=np.complex128),
            Q0_direct=np.zeros((1, 4, 4), dtype=np.complex128),
            H_linear=np.zeros((1, 2, 4, 4), dtype=np.complex128),
            S_direct=np.zeros((1, 2, 2, 4, 4), dtype=np.complex128),
        ).index(0.0 + 3.0j)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("static-only source was accepted as a probe")

    assert "no exact response" in message
