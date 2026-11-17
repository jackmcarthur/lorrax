"""A run's tmp/ and its large collective writes off Cray DVS (CPU, no device; milliseconds).

Job 61163888 wrote 440 GB of tmp/zeta_q.h5 to CFS, which Perlmutter's compute nodes mount
through DVS at stripe width 1, at about 70 MB/s, and timed out.  ``file_io.paths.dvs_reason``
names DVS from the mount table or the site profile (a login node sees CFS as GPFS); the run
refuses a tmp/ there when it writes the restart tensors
(``file_io.restart_bundle.refuse_tmp_on_dvs``), ``file_io.slab_io`` refuses a collective dataset
over 1 GB there, and ``lx doctor`` fails the deck the run would refuse.
"""
import logging
import pathlib
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

MOUNTS = (("/", "rootfs"), ("/global/cfs", "dvs"), ("/global/cfs/cdirs/fast", "lustre"),
          ("/pscratch", "lustre"))
NO_SITE = {}
PERLMUTTER = tomllib.loads(
    (pathlib.Path(__file__).parents[1] / "config" / "perlmutter" / "site.toml").read_text())


def test_dvs_from_the_mount_table():
    from file_io.paths import dvs_reason
    assert "DVS mount /global/cfs" in dvs_reason("/global/cfs/cdirs/m1/run/tmp", mounts=MOUNTS,
                                                 site=NO_SITE)
    # The longest mount point holding the path decides.
    assert dvs_reason("/global/cfs/cdirs/fast/tmp", mounts=MOUNTS, site=NO_SITE) is None
    assert dvs_reason("/pscratch/sd/j/u/run/tmp", mounts=MOUNTS, site=NO_SITE) is None
    assert dvs_reason("/global/cfsx/tmp", mounts=MOUNTS, site=NO_SITE) is None


def test_dvs_from_the_site_profile_where_the_login_node_sees_gpfs():
    from file_io.paths import dvs_reason
    login = (("/", "rootfs"), ("/global/cfs", "gpfs"), ("/pscratch", "lustre"))
    site = dict(PERLMUTTER, site="perlmutter")
    why = dvs_reason("/global/cfs/cdirs/m4598/run/tmp", mounts=login, site=site)
    assert why is not None and "compute_dvs_roots" in why
    assert dvs_reason("/global/homes/j/u/run/tmp", mounts=login, site=site) is not None
    assert dvs_reason("/pscratch/sd/j/u/run/tmp", mounts=login, site=site) is None


def _cfg(write):
    return SimpleNamespace(write_restart_tensors=write)


def test_run_refuses_a_dvs_tmp_that_takes_the_restart_tensors(monkeypatch, caplog):
    import file_io.paths as paths
    from file_io.restart_bundle import refuse_tmp_on_dvs
    monkeypatch.setattr(paths, "dvs_reason", lambda p: f"{p} is on the DVS mount /global/cfs")
    with pytest.raises(RuntimeError, match="GATE tmp_on_dvs.*write_restart_tensors = false"):
        refuse_tmp_on_dvs("/global/cfs/run/tmp", _cfg(True))
    with caplog.at_level(logging.WARNING):
        refuse_tmp_on_dvs("/global/cfs/run/tmp", _cfg(False))
    assert any("DVS" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(paths, "dvs_reason", lambda p: None)
    refuse_tmp_on_dvs("/pscratch/run/tmp", _cfg(True))


def test_slab_io_refuses_a_large_collective_dataset_on_dvs(monkeypatch):
    import contextlib
    from file_io import slab_io
    from file_io.io_timing import DISABLED
    from file_io.paths import DVS_COLLECTIVE_LIMIT
    from file_io.slab_io import SlabIO
    # The handle's transport and journal are stand-ins: the refusal comes before either.
    monkeypatch.setattr(slab_io._journal, "op_scope", lambda *a, **k: contextlib.nullcontext())
    created = []
    io = object.__new__(SlabIO)
    io.path, io.mode, io._stack, io._timing = "/global/cfs/run/tmp/zeta_q.h5", "w", "ffi", DISABLED
    io._backend = SimpleNamespace(create_dataset=lambda name, **kw: created.append(name), fh=None)
    io._dvs = "/global/cfs/run/tmp/zeta_q.h5 is on the DVS mount /global/cfs"
    side = int(np.sqrt(DVS_COLLECTIVE_LIMIT / 16)) + 2
    with pytest.raises(RuntimeError, match="GATE collective_write_on_dvs"):
        io.create_dataset("zeta", shape=(side, side), dtype=np.complex128)
    assert created == []
    io.create_dataset("energies", shape=(4, 1000), dtype=np.float64)
    io._dvs = None
    io.create_dataset("zeta", shape=(side, side), dtype=np.complex128)
    assert created == ["energies", "zeta"]


def test_doctor_fails_the_deck_the_run_would_refuse(monkeypatch, tmp_path):
    import file_io.paths as paths
    from lxkit.deck_doctor import DeckDoctorError, _tmp_filesystem_lines
    deck = tmp_path / "gnppm.in"
    monkeypatch.setattr(paths, "dvs_reason", lambda p: f"{p} is under /global/cfs")
    with pytest.raises(DeckDoctorError) as exc:
        _tmp_filesystem_lines(_cfg(True), deck)
    assert exc.value.rule == "LX-DECK-TMP-DVS"
    assert _tmp_filesystem_lines(_cfg(False), deck)[0].startswith("TMP_FILESYSTEM WARNING")
    monkeypatch.setattr(paths, "dvs_reason", lambda p: None)
    assert _tmp_filesystem_lines(_cfg(True), deck)[0].startswith("TMP_FILESYSTEM OK")
