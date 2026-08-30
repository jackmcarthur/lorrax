"""Focused state matrix for restart ``psi/E`` QP provenance."""
from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _wfn(path, *, qp: bool, fingerprint: str) -> None:
    from file_io.qp_wfn import QP_WFN_ATTR, QP_WFN_SCHEME

    with h5py.File(path, "w") as h5:
        h5.attrs["test_fingerprint"] = fingerprint
        if qp:
            h5.attrs[QP_WFN_ATTR] = QP_WFN_SCHEME
            h5.attrs["qp_wfn_band_start"] = 2
            h5.attrs["qp_wfn_band_stop"] = 6
            h5.attrs["qp_wfn_source"] = "WFN.h5"


def _record(path) -> dict:
    from common.parallel_transport import WFN_FINGERPRINT_SCHEME
    from file_io.qp_wfn import QP_STATE_SOURCE_SCHEMA, read_qp_wfn_stamp

    with h5py.File(path, "r") as h5:
        fingerprint = str(h5.attrs["test_fingerprint"])
    return {
        "schema": QP_STATE_SOURCE_SCHEMA,
        "wfn_fingerprint_scheme": WFN_FINGERPRINT_SCHEME,
        "wfn_fingerprint": fingerprint,
        "qp_wfn_stamp": read_qp_wfn_stamp(path),
    }


def _restart(path, record=None) -> None:
    from file_io.qp_wfn import (
        QP_STATE_SOURCE_DATASET,
        encode_qp_state_source_provenance,
    )

    with h5py.File(path, "w") as h5:
        if record is not None:
            h5.create_dataset(
                QP_STATE_SOURCE_DATASET,
                data=encode_qp_state_source_provenance(record))


@pytest.mark.parametrize(
    "legacy,source_qp,selected_qp,same_content,same_path,eqp,error",
    [
        (True, False, False, True, False, None, None),
        (True, False, False, True, False, "eqp1.dat", "predates"),
        (True, False, True, True, False, None, "stamped QP WFN"),
        (False, False, False, True, False, "eqp1.dat", None),
        (False, True, True, True, False, None, None),
        (False, True, True, True, False, "eqp1.dat", "DFT band LABELS"),
        (False, False, False, False, False, None, "different canonical"),
        (False, False, False, False, True, None, "different canonical"),
    ],
    ids=(
        "legacy-no-external", "legacy-eqp", "legacy-stamped-wfn",
        "matched-dft-eqp", "matched-qp", "qp-double-eqp",
        "different-dft-state", "same-path-replaced-wfn",
    ),
)
def test_restart_qp_state_matrix(
        tmp_path, legacy, source_qp, selected_qp, same_content, same_path,
        eqp, error):
    from file_io import qp_wfn

    source = tmp_path / "WFN-source.h5"
    selected = source if same_path else tmp_path / "WFN-selected.h5"
    restart = tmp_path / "restart.h5"
    _wfn(source, qp=source_qp, fingerprint="a" * 64)
    state = None if legacy else _record(source)
    _wfn(selected, qp=selected_qp,
         fingerprint=("a" if same_content else "b") * 64)
    _restart(restart, state)

    def selected_record(path):
        return _record(path)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            qp_wfn, "_qp_state_source_from_path", selected_record)
        def call():
            return qp_wfn.refuse_conflicting_qp_state_sources(
                wfn_path=str(selected), eqp_file=eqp,
                state_artifact_path=str(restart),
                where="focused state matrix")
        if error is None:
            assert call() is None
        else:
            with pytest.raises(ValueError, match=error):
                call()


def test_record_roundtrip_and_malformed_refusal(tmp_path):
    from file_io.qp_wfn import (
        QP_STATE_SOURCE_DATASET,
        read_qp_state_source_provenance,
    )

    wfn = tmp_path / "WFN.h5"
    restart = tmp_path / "restart.h5"
    _wfn(wfn, qp=False, fingerprint="c" * 64)
    expected = _record(wfn)
    _restart(restart, expected)
    assert read_qp_state_source_provenance(restart) == expected

    with h5py.File(restart, "w") as h5:
        h5.create_dataset(QP_STATE_SOURCE_DATASET, data=np.bytes_(b"{bad"))
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        read_qp_state_source_provenance(restart)


def test_qp_state_provenance_reuses_bound_canonical_fingerprint(
        tmp_path, monkeypatch):
    from common import parallel_transport
    from file_io.qp_wfn import qp_state_source_provenance_from_binding

    source = tmp_path / "WFN.h5"
    _wfn(source, qp=False, fingerprint="d" * 64)

    wfn = type("PathBearingWfn", (), {"path": str(source)})()
    monkeypatch.setattr(
        parallel_transport, "wfn_fingerprint", lambda _wfn: "d" * 64)
    binding = parallel_transport.bind_wfn_fingerprint(wfn)

    def unexpected_rescan(_wfn_value):
        raise AssertionError("precomputed canonical WFN fingerprint rescanned")

    monkeypatch.setattr(
        parallel_transport, "wfn_fingerprint", unexpected_rescan)
    record = qp_state_source_provenance_from_binding(
        wfn, wfn_fingerprint_binding=binding)
    assert record["wfn_fingerprint"] == "d" * 64

    with pytest.raises(ValueError, match="different loaded WFN object"):
        qp_state_source_provenance_from_binding(
            type("PathBearingWfn", (), {"path": str(source)})(),
            wfn_fingerprint_binding=binding)
    with pytest.raises(TypeError, match="bind_wfn_fingerprint"):
        qp_state_source_provenance_from_binding(
            wfn, wfn_fingerprint_binding="d" * 64)


def test_restart_authentication_scans_stamped_wfn_once_and_legacy_zero(
        tmp_path, monkeypatch):
    from common import parallel_transport
    from file_io.qp_wfn import (
        authenticate_restart_qp_state_source_for_wfn,
    )

    selected = tmp_path / "WFN.h5"
    restart = tmp_path / "restart.h5"
    _wfn(selected, qp=False, fingerprint="e" * 64)
    wfn = type("PathBearingWfn", (), {"path": str(selected)})()
    scans = []

    def canonical_scan(value):
        scans.append(value)
        return "e" * 64

    monkeypatch.setattr(
        parallel_transport, "wfn_fingerprint", canonical_scan)
    _restart(restart, _record(selected))
    record, binding = authenticate_restart_qp_state_source_for_wfn(
        wfn=wfn, state_artifact_path=str(restart), where="focused restart")
    assert record == _record(selected)
    assert parallel_transport.fingerprint_from_binding(binding, wfn) == "e" * 64
    assert scans == [wfn]

    scans.clear()
    _restart(restart)
    record, binding = authenticate_restart_qp_state_source_for_wfn(
        wfn=wfn, state_artifact_path=str(restart), where="focused restart")
    assert (record, binding) == (None, None)
    assert scans == []
