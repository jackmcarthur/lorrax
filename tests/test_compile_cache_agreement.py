"""Unit tests for the multiprocess persistent-compile-cache agreement layer.

These need no devices and no ``jax.distributed``: the agreement protocol is
pure Python over a key-value client, so a fake in-memory client plus one
thread per "rank" reproduces it exactly.  The properties under test are the
safety invariants the whole design rests on:

  * every rank ends up with the SAME agreed set (that is what keeps the
    hit/miss pattern identical and therefore keeps XLA:GPU's collective
    autotune exchange from deadlocking — scorecard AG);
  * an entry only one rank can see is dropped by ALL ranks;
  * the ``LORRAX_JAX_CACHE_FORCE_DIVERGE`` positive-control hook really does
    force a drop rather than silently doing nothing;
  * the directory scan ignores partial/temp files.
"""
from __future__ import annotations

import os
import threading

import pytest

from common import jax_compile_cache as jcc


# ---------------------------------------------------------------------------
# a fake coordination-service KV client
# ---------------------------------------------------------------------------
class FakeKV:
    """In-memory stand-in for jax's DistributedRuntimeClient."""

    def __init__(self, n_proc: int):
        self._d: dict[str, bytes] = {}
        self._cv = threading.Condition()
        self._barrier = threading.Barrier(n_proc, timeout=30)
        self.n_set = 0
        self.n_get = 0

    def key_value_set_bytes(self, key, value):
        with self._cv:
            self._d[key] = bytes(value)
            self.n_set += 1
            self._cv.notify_all()

    def blocking_key_value_get_bytes(self, key, timeout_ms):
        deadline = timeout_ms / 1000.0
        with self._cv:
            self.n_get += 1
            if not self._cv.wait_for(lambda: key in self._d, timeout=deadline):
                raise TimeoutError(f"no value for {key!r}")
            return self._d[key]

    def wait_at_barrier(self, name, timeout_in_ms=None):
        self._barrier.wait()


def _run_agreement(tmp_dirs, n_proc, env=None):
    """Run the real protocol with one thread per rank; return per-rank results."""
    kv = FakeKV(n_proc)
    out: dict[int, object] = {}
    old_env = {k: os.environ.get(k) for k in (env or {})}
    if env:
        os.environ.update(env)

    def work(p):
        try:
            out[p] = jcc._agree_on_entries(tmp_dirs[p], n_proc, p, 20.0,
                                           client=kv)
        except BaseException as exc:  # noqa: BLE001
            out[p] = exc

    threads = [threading.Thread(target=work, args=(p,)) for p in range(n_proc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=40)
    if env:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for p in range(n_proc):
        assert p in out, f"rank {p} never finished (deadlock in the protocol)"
        if isinstance(out[p], BaseException):
            raise out[p]
    return out


def _make_entries(d, keys, size=16):
    for k in keys:
        (d / f"{k}{jcc._CACHE_SUFFIX}").write_bytes(b"x" * size)


# ---------------------------------------------------------------------------
def test_local_entry_keys_ignores_partial_and_temp_files(tmp_path):
    _make_entries(tmp_path, ["jit_a-1111", "jit_b-2222"])
    # a zero-length file (a torn write from a pre-AH, non-atomic writer)
    (tmp_path / f"jit_c-3333{jcc._CACHE_SUFFIX}").write_bytes(b"")
    # our own in-flight temp file
    (tmp_path / f".jit_d-4444{jcc._CACHE_SUFFIX}.tmp.123.abcd").write_bytes(b"x")
    # jax's atime sidecar
    (tmp_path / "jit_a-1111-atime").write_bytes(b"12345678")
    assert jcc._local_entry_keys(tmp_path) == ["jit_a-1111", "jit_b-2222"]


def test_all_ranks_agree_when_the_directory_is_shared(tmp_path):
    keys = [f"jit_m{i}-{i:064x}" for i in range(5)]
    shared = tmp_path / "np4"
    shared.mkdir()
    _make_entries(shared, keys)
    res = _run_agreement({p: shared for p in range(4)}, 4)
    for p in range(4):
        n_seen, agreed = res[p]
        assert n_seen == 5
        assert agreed == frozenset(keys)


def test_an_entry_one_rank_cannot_see_is_dropped_by_everyone(tmp_path):
    """The core safety property: NO rank may hit what some rank would miss."""
    keys = [f"jit_m{i}-{i:064x}" for i in range(5)]
    dirs = {}
    for p in range(4):
        d = tmp_path / f"rank{p}view"
        d.mkdir()
        # rank 2 is missing the last two entries
        _make_entries(d, keys[:-2] if p == 2 else keys)
        dirs[p] = d
    res = _run_agreement(dirs, 4)
    expected = frozenset(keys[:-2])
    for p in range(4):
        n_seen, agreed = res[p]
        assert n_seen == 5
        assert agreed == expected, f"rank {p} disagreed: {sorted(agreed)}"


def test_force_diverge_hook_really_forces_a_drop(tmp_path):
    """The positive control must not be a no-op."""
    keys = sorted(f"jit_m{i}-{i:064x}" for i in range(6))
    shared = tmp_path / "np4"
    shared.mkdir()
    _make_entries(shared, keys)
    res = _run_agreement({p: shared for p in range(4)}, 4,
                         env={"LORRAX_JAX_CACHE_FORCE_DIVERGE": "2"})
    expected = frozenset(keys[:-2])
    for p in range(4):
        n_seen, agreed = res[p]
        assert n_seen == 6
        assert agreed == expected


def test_empty_cold_cache_agrees_on_nothing(tmp_path):
    shared = tmp_path / "np8"
    shared.mkdir()
    res = _run_agreement({p: shared for p in range(8)}, 8)
    for p in range(8):
        n_seen, agreed = res[p]
        assert n_seen == 0
        assert agreed == frozenset()


def test_agreement_cost_is_o1_rpcs_per_rank(tmp_path):
    """Guard against a per-module protocol creeping back in.

    The whole point of the snapshot design is that agreement is O(1) RPCs
    per rank per RUN, not O(modules).
    """
    keys = [f"jit_m{i}-{i:064x}" for i in range(200)]
    shared = tmp_path / "np8"
    shared.mkdir()
    _make_entries(shared, keys)
    kv = FakeKV(8)
    out = {}

    def work(p):
        out[p] = jcc._agree_on_entries(shared, 8, p, 20.0, client=kv)

    ts = [threading.Thread(target=work, args=(p,)) for p in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=40)
    assert len(out) == 8
    # p0: 1 keylist set + 1 final set;  p!=0: 1 mask set  -> 9 sets total
    assert kv.n_set == 9
    # p0: P-1 mask gets;  p!=0: 1 keylist get + 1 final get -> 7 + 14 = 21
    assert kv.n_get == 21


def test_missing_client_raises_rather_than_hanging(monkeypatch):
    """No coordination client -> a clean raise, which ensure_() turns into
    'cache off, here is why'.  It must never fall through to a shared cache
    without agreement."""
    import sys
    import types

    fake = types.ModuleType("jax._src.distributed")
    fake.global_state = types.SimpleNamespace(client=None)
    monkeypatch.setitem(sys.modules, "jax._src.distributed", fake)
    with pytest.raises(RuntimeError):
        jcc._agree_on_entries("/nonexistent", 4, 0, 1.0)


def test_bitmask_helpers_roundtrip():
    m = jcc._mask_bytes(19)
    assert len(m) == 3
    for i in (0, 7, 8, 18):
        jcc._set_bit(m, i)
    assert [i for i in range(19) if jcc._bit(m, i)] == [0, 7, 8, 18]


# ---------------------------------------------------------------------------
# THE CACHE OBJECT MUST FOLLOW THE DIRECTORY WE CHOOSE
# ---------------------------------------------------------------------------
# MEASURED, Perlmutter 2026-08-07, centroid deck at P=4: the run reported
# `cache_probes=149 hits=0 vetoed=149 agreed=622/622` and never warmed up on
# consecutive runs.  `jax._src.compilation_cache._cache` is bound ONCE, at the
# first compile, from `jax_compilation_cache_dir` as it read then; the
# `lorrax_J070` modulefile exports `JAX_COMPILATION_CACHE_DIR=$SCRATCH/
# .jax_cache` into the container, and the mesh warm-up compiles before
# `ensure_jax_compile_cache` runs.  A later `config.update` does NOT rebind it,
# so the agreement listed one directory while JAX used another: every probe
# vetoed, every write a no-op on an already-present key, `enabled=True`.
#
# These two cells are the pair.  The first is the property; the second is the
# case where it comes out FALSE — without the rebind the cache object keeps the
# directory it was bound to, which is the defect verbatim.

class _FakeCache:
    """Stands in for jax's LRUCache: all `bound_cache_dir` reads is `.path`."""

    def __init__(self, path):
        self.path = path
        self._path = path


def _fake_cc(monkeypatch, bound_to):
    """Bind jax's REAL compilation_cache module to ``bound_to``.

    The real module, not a stand-in in ``sys.modules``: ``bound_cache_dir``
    reaches it with ``from jax._src import compilation_cache``, which resolves
    the ATTRIBUTE on the already-imported parent package, so a sys.modules
    substitution would not be seen and the cell would test nothing.
    """
    cc = pytest.importorskip("jax._src.compilation_cache")
    monkeypatch.setattr(
        cc, "_cache", _FakeCache(bound_to) if bound_to is not None else None,
        raising=False)
    return cc


def test_bound_cache_dir_reports_the_directory_jax_actually_uses(
        monkeypatch, tmp_path):
    """The reported binding is READ OFF the live cache object, not inferred.

    `_STATE.dir` is what this module ASKED for.  They are different questions
    and the whole defect lived in the gap between them.
    """
    _fake_cc(monkeypatch, str(tmp_path / "elsewhere"))
    assert jcc.bound_cache_dir() == str(tmp_path / "elsewhere")

    # ...and an unbound cache reports "" rather than guessing.
    _fake_cc(monkeypatch, None)
    assert jcc.bound_cache_dir() == ""


@pytest.mark.parametrize("bound_name", ["elsewhere", "ours"])
def test_any_inherited_binding_is_rebuilt_through_the_atomic_factory(
        monkeypatch, tmp_path, bound_name):
    """THE FIX, including the same-path stock-cache red twin.

    `reset_cache()` is the only thing that makes JAX rebuild its cache object
    through the patched ``get_file_cache`` factory.  A same-path object must be
    reset too: otherwise its directory looks correct while its writes remain
    stock-JAX non-atomic and unmeasured.
    """
    inherited = str(tmp_path / ".jax_cache")
    ours = str(tmp_path / "lorrax_jax_cache" / "np4")
    bound_to = ours if bound_name == "ours" else inherited

    # RED TWIN FIRST: without a reset the stale binding survives and does NOT
    # pass through the patched atomic factory.  The same-path arm is the subtle
    # case: path equality cannot prove which cache class owns writes.
    _fake_cc(monkeypatch, bound_to)
    assert jcc.bound_cache_dir() == bound_to

    # THE FIX: production resets both arms.  The next `_get_cache` rebuilds
    # from `jax_compilation_cache_dir`, after ensure has pointed it at ours,
    # and therefore invokes LORRAX's patched factory.
    jcc._reset_bound_cache_for_atomic_writer(tmp_path / "lorrax_jax_cache" /
                                              "np4", proc_idx=1)
    assert jcc.bound_cache_dir() == ""


def test_bound_stock_cache_reset_failure_is_loud(monkeypatch, tmp_path):
    """Atomicity cannot degrade to a healthy-looking stock cache object."""
    ours = str(tmp_path / "lorrax_jax_cache" / "np4")
    cc = _fake_cc(monkeypatch, ours)

    def _fail_reset():
        raise RuntimeError("constructed reset failure")

    monkeypatch.setattr(cc, "reset_cache", _fail_reset)
    with pytest.raises(RuntimeError, match="stock non-atomic writes"):
        jcc._reset_bound_cache_for_atomic_writer(
            tmp_path / "lorrax_jax_cache" / "np4", proc_idx=0)
    assert jcc.bound_cache_dir() == ours


def test_the_summary_line_names_a_disagreeing_binding(monkeypatch, tmp_path,
                                                      capsys):
    """A silent disagreement is the failure mode; the report must say it."""
    monkeypatch.setattr(jcc._STATE, "dir", str(tmp_path / "ours"),
                        raising=False)
    _fake_cc(monkeypatch, str(tmp_path / "elsewhere"))
    jcc._report_impl()
    assert "BOUND-ELSEWHERE" in capsys.readouterr().out

    # RED TWIN: when they AGREE the line must stay quiet, or the marker is
    # noise and stops meaning anything.
    _fake_cc(monkeypatch, str(tmp_path / "ours"))
    jcc._report_impl()
    assert "BOUND-ELSEWHERE" not in capsys.readouterr().out
