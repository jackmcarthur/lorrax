"""W0 q-wedge provenance must refuse before the restart dataset is mutated."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_mismatched_q_capture_refuses_before_slabio_opens(monkeypatch):
    pytest.importorskip("jax")
    from file_io import slab_io, tagged_arrays
    from gw import restart_q_storage

    slab_opens = []

    class _ForbiddenSlabIO:
        def __init__(self, *_args, **_kwargs):
            slab_opens.append(True)
            raise AssertionError("SlabIO opened before q-capture validation")

    def _mismatch(*_args, **_kwargs):
        raise ValueError("capture/resolution mismatch sentinel")

    monkeypatch.setattr(slab_io, "SlabIO", _ForbiddenSlabIO)
    monkeypatch.setattr(restart_q_storage, "assert_capture_matches", _mismatch)
    capture = SimpleNamespace(
        X_ibz=np.zeros((1, 1, 1), dtype=np.complex128),
        n_rmu_logical=1,
    )
    qirr = SimpleNamespace(
        store_wedge=True, capture=capture, resolution=object())

    with pytest.raises(ValueError, match="mismatch sentinel"):
        tagged_arrays.write_w0_qmunu_to_h5(
            "must_not_be_opened.h5",
            np.ones((2, 1, 1), dtype=np.complex128),
            n_rmu_logical=1,
            mesh=None,
            qirr=qirr,
        )

    assert slab_opens == []
