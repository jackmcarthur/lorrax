"""Deferred charge payloads keep real file receipts incomplete until V writes."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


@pytest.mark.parametrize("write_file,augmented,vertices", [
    (False, True, (0,)), (True, False, (0,)),
    (True, True, (1,)), (True, True, (1, 2, 3)),
])
def test_defer_refuses_an_owner_without_a_physical_charge_writer(write_file, augmented, vertices):
    from gw.isdf_fitting import _validate_deferred_zeta_write
    with pytest.raises(ValueError, match="augmented charge fit"):
        _validate_deferred_zeta_write(
            defer=True, write_file=write_file, augmented=augmented, vertices=vertices)


@pytest.mark.parametrize("defer,trunc,completed,pending", [
    (False, (), True, None), (True, (), False, True),
    (False, (("partial_fit", 1),), False, None),
    (True, (("partial_fit", 1),), False, False),
])
def test_completion_waits_for_the_deferred_physical_payload(tmp_path, monkeypatch,
                                                          defer, trunc, completed, pending):
    import jax
    from gw.isdf_fitting import _finish_zeta_fit_files, _validate_deferred_zeta_write
    # A real isdf_header receipt is authoritative; a pending object alone must
    # not certify a payload that has not been streamed into the file.
    path = tmp_path / "zeta.h5"
    with h5py.File(path, "w") as stream:
        stream.create_group("isdf_header").create_dataset("zeta_is_done", data=np.bool_(False))
    zeta = SimpleNamespace()
    channel = SimpleNamespace(write=not defer, output_file=str(path))
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    _validate_deferred_zeta_write(defer=defer, write_file=True, augmented=True, vertices=(0,))
    _finish_zeta_fit_files({0: zeta}, [channel], defer=defer, trunc=trunc)
    with h5py.File(path) as stream:
        assert bool(stream["isdf_header/zeta_is_done"][()]) is completed
        assert "zeta_q_G" not in stream
    assert getattr(zeta, "pending_zeta_write", None) is pending
