"""The completion gate's verdict is rank 0's, and peers never open the files.

``gw_jax._published_artifacts`` gates the completion line on every required
artifact being present and committed. Each rank used to open the files
itself; on a 4-node QSGW run rank 9 of 16 failed the gate while rank 0's
manifest (the only one logged) was clean. The manifest is now read once on
rank 0 and handed to every rank (``common.collectives.rank0_read``). Here a
peer whose own h5py open would fail (a lock, a short read) still returns
rank 0's verdict, and the single-process verdicts are unchanged.

Host-side file metadata, no device code: a CPU unit cell, exempt from the
four-GPU rule under its unit-and-CPU-cell clause.
"""
import h5py
import numpy as np


def test_published_artifacts_rank0(tmp_path, monkeypatch):
    from gw import gw_jax
    import common.collectives as collectives

    good, bad = tmp_path / "good.h5", tmp_path / "bad.h5"
    for path, flag in ((good, 1), (bad, 0)):
        with h5py.File(path, "w") as f:
            f["lorrax_io_committed"] = np.array([flag], dtype=np.int32)
    missing = str(tmp_path / "missing.h5")
    assert gw_jax._published_artifacts((("Sigma", str(good), True),))
    assert not gw_jax._published_artifacts((("Sigma", str(bad), True),))
    assert not gw_jax._published_artifacts((("Sigma", missing, True),))
    assert gw_jax._published_artifacts((("tensors", missing, False),
                                        ("Sigma", str(good), True)))

    # A peer rank: rank 0's read arrives, its own open would fail.
    stages = []

    def rank0_read(path, *, stage, read):
        stages.append(stage)
        return {"ok": True, "lines": [f"    Sigma: committed=1 ({path})"]}

    def unopenable(*args, **kwargs):
        raise OSError("unable to lock file")
    monkeypatch.setattr(collectives, "rank0_read", rank0_read)
    monkeypatch.setattr(h5py, "File", unopenable)
    assert gw_jax._published_artifacts((("Sigma", str(good), True),))
    assert stages == ["gw.published_artifacts"]
