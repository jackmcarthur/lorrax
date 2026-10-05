"""The per-rank streamed tier's capacity probe on toy stores (CPU only, seconds).

1. Records are page-aligned, never padded to the filesystem block (16 MiB on GPFS).
2. A store whose bytes fallocate already took is not promised a second time;
   a sparse (Lustre) store stays promised until it is written.
3. The Lustre quota room is measured against the hard limit, so a user inside
   the soft-limit grace period still gets a capacity check, not a negative room.
4. A W-bank field created after initialization (a line panel) that the disk
   refuses is held in host memory, so the map runs on.
"""
import mmap
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

TILE = 64                      # local tile 64 x 64 complex128 = 64 KiB per record


def _mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _bank(root, label, n_out, kind="file"):
    from file_io import _slab_io_rank as R
    return R.StreamedBank(_mesh(), root=root, label=label, kind=kind, n_out=n_out, q=1,
                          segments=((TILE, TILE, ((0, 0, 0, 0, TILE, TILE),)),), tile=(TILE, TILE))


def _no_lfs(monkeypatch):
    import shutil
    from file_io import _slab_io_rank as R
    which = shutil.which
    monkeypatch.setattr(R.shutil, "which", lambda name: None if name == "lfs" else which(name))


def test_records_align_to_the_page_not_the_filesystem_block(tmp_path, monkeypatch):
    statvfs = os.statvfs

    def gpfs(path):
        real = statvfs(path)
        return SimpleNamespace(f_bsize=16 << 20, f_frsize=real.f_frsize, f_bavail=real.f_bavail)
    _no_lfs(monkeypatch)
    monkeypatch.setattr(os, "statvfs", gpfs)
    bank = _bank(tmp_path, "align", 2)
    try:
        assert bank.fits
        assert bank.align == mmap.PAGESIZE
        assert bank.records == (16 * TILE * TILE,)
    finally:
        bank.release()


def test_reserved_bytes_are_not_promised_twice(tmp_path, monkeypatch):
    from file_io import _slab_io_rank as R
    _no_lfs(monkeypatch)
    root = str(tmp_path)

    def allocated():
        total = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}").startswith(root):
                    total += os.fstat(int(fd)).st_blocks * 512
            except OSError:
                pass
        return total
    per_store = 16 * 16 * TILE * TILE          # 16 outputs of 64 KiB on one device
    cap = int(2.5 * per_store)                 # room for two stores, not three
    statvfs = os.statvfs
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(
        f_bsize=4096, f_frsize=4096, f_bavail=(cap - allocated()) // 4096)
        if str(path).startswith(root) else statvfs(path))
    a = _bank(root, "a", 16)
    b = None
    try:
        assert a.fits
        if allocated() < per_store:
            pytest.skip("fallocate does not allocate under the test's tmp_path")
        assert a.promised == 0                 # its bytes left statvfs at once
        b = _bank(root, "b", 16)
        assert b.fits                          # 1.5 stores free; old code promised a twice
    finally:
        for bank in (a, b):
            if bank is not None:
                bank.release()
    # A sparse store (no fallocate, the Lustre case) stays promised until written.
    monkeypatch.setattr(R, "_reserve", lambda fd, nbytes: False)
    sparse = _bank(root, "sparse", 16)
    try:
        assert sparse.fits and sparse.promised == per_store
    finally:
        sparse.release()
    assert R._PROMISED[0] == 0


def test_quota_room_is_under_the_hard_limit(tmp_path, monkeypatch):
    import subprocess
    from file_io import _slab_io_rank as R
    tib = 1 << 30                                                  # lfs reports KiB
    used, soft, hard = 21 * tib, 20 * tib, 30 * tib                # in the grace period
    monkeypatch.setattr(R.shutil, "which", lambda name: "/usr/bin/lfs")
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout=f"{tmp_path} {used}* {soft} {hard} 6d 1 0 0 -\n", stderr=""))
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(
        f_bsize=4096, f_frsize=4096, f_bavail=(100 * tib * 1024) // 4096))
    assert R._free_bytes(tmp_path) == (hard - used) * 1024


def test_late_bank_field_refused_by_the_disk_is_held_in_host_memory(tmp_path, monkeypatch):
    from file_io import _slab_io_rank as R
    from file_io.shared_pole_store import ResidentBankPayload
    _no_lfs(monkeypatch)
    nq, ns, d = 2, 2, TILE
    dense = 16 * nq * ns * d * d
    written = [0]
    write = R._Store.write

    def counted(self, source, offset):
        written[0] += len(source)
        return write(self, source, offset)
    # Lustre without fallocate: the quota counts bytes as they are written.
    monkeypatch.setattr(R._Store, "write", counted)
    monkeypatch.setattr(R, "_reserve", lambda fd, nbytes: False)
    monkeypatch.setattr(R, "_free_bytes", lambda directory: dense + 4096 - written[0])
    mesh = _mesh()
    pay = ResidentBankPayload(mesh, carrier=d, label="bank.h5", memory_kind="file", root=tmp_path)
    try:
        pay.create_dataset("Wc", shape=(nq, ns, d, d), dtype=np.complex128)
        pay.write_attr("header_json", b"{}")       # initialization is done
        assert pay.fits
        put = lambda shape: jax.device_put(jnp.ones(shape, jnp.complex128),
                                           NamedSharding(mesh, P(None, None, "x", "y")))
        pay.write_slab("Wc", put((nq, ns, d, d)), offset=(0, 0, 0, 0))
        panel = (nq, 3, d, 8)                      # a line sample's panels, made at first write
        with pytest.warns(RuntimeWarning, match="held in host memory"):
            pay.create_dataset("line_charge_003", shape=panel, dtype=np.complex128)
        assert pay._fields["line_charge_003"].kind == "host"
        pay.write_slab("line_charge_003", put(panel), offset=(0, 0, 0, 0))
        with pay:
            back = pay.read_slab("line_charge_003", shape=panel, offset=(0, 0, 0, 0),
                                 dtype=np.complex128, partition_spec=P(None, None, "x", "y"))
        assert np.array_equal(np.asarray(back), np.ones(panel))
    finally:
        pay.release()
