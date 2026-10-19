"""runtime.vendor_kernels on the host: the store fold, and no-ops on a leg that declares nothing.

Plain sqlite3 files in tmp_path stand in for rocFFT's cache (its table, read back from
the writer itself); no GPU, no vendor library.
"""
import sqlite3

from runtime import vendor_kernels as vk

_SCHEMA = ("CREATE TABLE cache_v1 (kernel_name TEXT NOT NULL, arch TEXT NOT NULL, "
           "code BLOB NOT NULL, PRIMARY KEY (kernel_name, arch))")


def _db(path, rows):
    c = sqlite3.connect(path)
    c.execute(_SCHEMA)
    c.executemany("insert into cache_v1 values (?, ?, ?)", rows)
    c.commit()
    c.close()


def _rows(path):
    c = sqlite3.connect(path)
    try:
        return sorted(c.execute("select kernel_name, arch from cache_v1"))
    finally:
        c.close()


def test_fold_creates_then_extends_the_snapshot(tmp_path):
    snap = tmp_path / "rocfft_kernel_cache.db"
    w1 = tmp_path / "w1.db"
    _db(w1, [("fft_len_15", "gfx90a", b"a"), ("fft_len_18", "gfx90a", b"b")])
    assert vk.fold_into_snapshot(w1, snap) == 2
    assert not w1.exists() and _rows(snap) == [("fft_len_15", "gfx90a"), ("fft_len_18", "gfx90a")]
    w2 = tmp_path / "w2.db"
    _db(w2, [("fft_len_18", "gfx90a", b"b"), ("fft_len_20", "gfx90a", b"c")])
    assert vk.fold_into_snapshot(w2, snap) == 1            # the duplicate is ignored
    assert [r[0] for r in _rows(snap)] == ["fft_len_15", "fft_len_18", "fft_len_20"]
    assert not list(tmp_path.glob("*.tmp.*"))


def test_empty_or_absent_writer_writes_nothing(tmp_path):
    snap = tmp_path / "s.db"
    assert vk.fold_into_snapshot(tmp_path / "absent.db", snap) == 0
    w = tmp_path / "w.db"
    _db(w, [])
    assert vk.fold_into_snapshot(w, snap) == 0
    assert not snap.exists() and not w.exists()


def test_host_leg_arms_nothing(monkeypatch):
    monkeypatch.delenv("ROCFFT_RTC_CACHE_PATH", raising=False)
    assert vk.arm_runtime_kernel_stores(say=lambda *_: None) == []
    assert vk.start_blas_host_warmup(say=lambda *_: None) == {}
    assert "ROCFFT_RTC_CACHE_PATH" not in __import__("os").environ
