"""Gates for the ``jit__multi_slice`` rank-dependent cache key.

WHAT IS BEING GATED.  ``jax/_src/array.py::shard_device_array`` — the path a
single-device ``jax.Array`` takes when it is handed to a multi-device sharding
— passes THIS RANK'S shard offsets to ``ArrayImpl._multi_slice``, which is
``jit(static_argnums=(1,2,3))``.  The offsets are therefore baked into the jit
signature, every rank compiles a different program, and every rank holds a
different persistent-compile-cache key.  Since JAX writes cache entries from
process 0 only, a warm P>1 run then has rank 0 HITTING while its peers MISS
and compile — the divergent hit/miss pattern that deadlocks XLA:GPU's
collective autotune (``docs/dev/env_vars.md``, scorecard AG).

WHY THESE TESTS NEED NEITHER A CLUSTER NOR FOUR DEVICES.  The divergence is in
the STATIC ARGUMENTS, not in the devices: rank r differs from rank s only by
the index tuple it passes.  So each case constructs the four ranks' index
tuples with JAX's own ``as_slice_indices`` — the same function
``shard_device_array`` uses — and compiles the slicer once per tuple, dropping
the in-process jit cache in between so that each iteration measures what a
SEPARATE PROCESS would measure.  The keys recorded are the real ones, spied
off ``jax._src.cache_key.get``.

Every case carries its red twin: the same measurement with the patch backed
out is the pre-fix state, and it must show the defect.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax._src import array as jax_array
from jax._src import cache_key as jax_cache_key
from jax._src.array import ArrayImpl

from common import jax_compile_cache as jcc

# One rank's worth of a length-8 leading axis split four ways — the shape of
# the production one-GPU-per-process launch, where a rank's addressable index
# map holds exactly one shard.
_SHAPE = (8, 5)
_NRANKS = 4


@pytest.fixture
def uninstalled():
    """Run with the patch backed out, and restore whatever was there."""
    had = getattr(ArrayImpl, "_lorrax_shard_slice_installed", False)
    orig = getattr(ArrayImpl, "_lorrax_shard_slice_orig", None)
    if had:
        ArrayImpl._multi_slice = orig
        del ArrayImpl._lorrax_shard_slice_installed
    yield
    if had:
        ArrayImpl._multi_slice = ArrayImpl.__dict__["_lorrax_shard_slice_orig"]
        jcc._install_shard_slice_patch()


def _rank_slice_args(rank: int):
    """The ``(start, limit, removed)`` triple rank ``rank`` would pass.

    Built the way ``shard_device_array`` builds it: from an index tuple, via
    JAX's own ``as_slice_indices``.
    """
    per = _SHAPE[0] // _NRANKS
    idx = (slice(rank * per, (rank + 1) * per), slice(0, _SHAPE[1]))
    return jax_array.as_slice_indices(np.zeros(_SHAPE), idx)


def _keys_and_values(tmp_path, monkeypatch):
    """Compile the slicer once per rank; return (keys, shard values, n_entries).

    The persistent cache is pointed at ``tmp_path`` and its size/time floors
    dropped to zero, so every compiled module is a real write and the entry
    count is a direct measurement of gate (d).
    """
    seen: list[str] = []
    orig_get = jax_cache_key.get

    def _spy(*a, **kw):
        k = orig_get(*a, **kw)
        seen.append(k)
        return k

    monkeypatch.setattr(jax_cache_key, "get", _spy)

    host = np.arange(int(np.prod(_SHAPE)), dtype=np.float32).reshape(_SHAPE)
    x = jnp.asarray(host)

    tmp_path.mkdir(parents=True, exist_ok=True)
    keys, values = [], []
    knobs = {
        "jax_compilation_cache_dir": str(tmp_path),
        "jax_persistent_cache_min_compile_time_secs": 0.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
    }
    # These flags carry contextmanagers, so `config.read` refuses them; the
    # attribute is the supported way to read one.
    saved = {k: getattr(jax.config, k) for k in knobs}
    for k, v in knobs.items():
        jax.config.update(k, v)
    # The cache object is built ONCE from the directory as it read then, and
    # only reset_cache() rebinds it (this file's own §"rebound JAX's compile
    # cache" note) — so a test that merely updates the config would measure
    # some other directory.
    from jax._src import compilation_cache as _cc
    _cc.reset_cache()
    try:
        for rank in range(_NRANKS):
            st, li, rm = _rank_slice_args(rank)
            # Each rank is its own process in the real run and computes its key
            # from scratch; drop the in-process jit cache so this loop does too.
            jax.clear_caches()
            del seen[:]
            shards = x._multi_slice((st,), (li,), (rm,))
            jax.block_until_ready(shards)
            assert seen, (
                "no cache key was computed — the persistent cache did not "
                "engage, so this case is measuring nothing")
            keys.append(seen[-1])
            values.append(np.asarray(shards[0]))
    finally:
        for k, v in saved.items():
            jax.config.update(k, v)
        _cc.reset_cache()

    entries = [p for p in tmp_path.iterdir() if p.is_file()]
    return keys, values, len(entries)


def test_patch_is_installed_by_default():
    """The production default arms the canonicalization."""
    jcc._install_shard_slice_patch()
    assert getattr(ArrayImpl, "_lorrax_shard_slice_installed", False)
    assert ArrayImpl.__dict__["_multi_slice"].__name__ == \
        "_canonical_multi_slice"


def test_one_cache_key_across_four_ranks(tmp_path, monkeypatch):
    """GATE: with the patch, four ranks name ONE ``jit__multi_slice`` key."""
    jcc._install_shard_slice_patch()
    keys, _values, n_entries = _keys_and_values(tmp_path, monkeypatch)

    assert len(set(keys)) == 1, (
        f"expected one key for four ranks, got {len(set(keys))}: {set(keys)}")
    # GATE (d): the one program actually persists.
    assert n_entries == 1, (
        f"expected exactly one cache entry on disk, found {n_entries}")


def test_red_twin_four_keys_without_the_patch(tmp_path, monkeypatch,
                                              uninstalled):
    """RED TWIN: stock JAX gives four ranks four keys and four entries.

    This is the registered defect, reproduced.  If this case ever goes green
    on its own, JAX has changed ``_multi_slice`` and the patch above should be
    re-examined rather than kept on faith.
    """
    keys, _values, n_entries = _keys_and_values(tmp_path, monkeypatch)

    assert len(set(keys)) == _NRANKS, (
        f"expected the defect — four distinct keys — got {len(set(keys))}")
    assert all(k.startswith("jit__multi_slice-") for k in keys), keys
    assert n_entries == _NRANKS


def test_shards_bit_identical_patched_vs_stock(tmp_path, monkeypatch):
    """GATE: the canonicalization does not change what any rank computes."""
    jcc._install_shard_slice_patch()
    _k, patched, _n = _keys_and_values(tmp_path / "on", monkeypatch)

    orig = ArrayImpl.__dict__["_lorrax_shard_slice_orig"]
    installed = ArrayImpl.__dict__["_multi_slice"]
    ArrayImpl._multi_slice = orig
    try:
        _k2, stock, _n2 = _keys_and_values(tmp_path / "off", monkeypatch)
    finally:
        ArrayImpl._multi_slice = installed

    host = np.arange(int(np.prod(_SHAPE)), dtype=np.float32).reshape(_SHAPE)
    per = _SHAPE[0] // _NRANKS
    for rank in range(_NRANKS):
        want = host[rank * per:(rank + 1) * per, :]
        # bit-identical, not close: these are copies, nothing arithmetic
        # happens to the values on either path
        assert np.array_equal(stock[rank], want)
        assert np.array_equal(patched[rank], want)
        assert patched[rank].dtype == stock[rank].dtype


def test_squeeze_path_preserved(tmp_path, monkeypatch):
    """``removed_dims`` (an integer index in the sharding map) still squeezes.

    ``as_slice_indices`` emits a removed dim for every INTEGER entry in the
    index tuple, and the canonical slicer has to reproduce the squeeze, not
    just the slice.
    """
    jcc._install_shard_slice_patch()
    host = np.arange(int(np.prod(_SHAPE)), dtype=np.float32).reshape(_SHAPE)
    x = jnp.asarray(host)

    st, li, rm = jax_array.as_slice_indices(np.zeros(_SHAPE), (2, slice(0, 5)))
    assert rm == (0,), rm
    (out,) = x._multi_slice((st,), (li,), (rm,))
    assert out.shape == (5,)
    assert np.array_equal(np.asarray(out), host[2, :])


def test_fallback_announces_and_stays_correct(monkeypatch):
    """A shape the canonical slicer declines must fall back LOUDLY, not quietly.

    The file's own doctrine: a compatibility path nobody can see in the log is
    indistinguishable from the bug it replaced.
    """
    jcc._install_shard_slice_patch()
    jcc._COMPAT_SAID.clear()
    said: list[str] = []
    monkeypatch.setattr(jcc, "_say", lambda m: said.append(m))
    monkeypatch.setattr(jcc._STATE, "proc_idx", 0)

    def _boom(*_a, **_kw):
        raise RuntimeError("constructed refusal")

    monkeypatch.setattr(jcc, "_canon_slice_jit", _boom)

    host = np.arange(int(np.prod(_SHAPE)), dtype=np.float32).reshape(_SHAPE)
    x = jnp.asarray(host)
    st, li, rm = _rank_slice_args(1)
    (out,) = x._multi_slice((st,), (li,), (rm,))

    # correct anyway — the fallback is JAX's own slicer
    assert np.array_equal(np.asarray(out), host[2:4, :])
    assert any("shard slicer declined" in m for m in said), said
    assert any("PER RANK" in m for m in said), said


def test_index_dtype_comes_from_the_global_shape():
    """The offset dtype must not be sized from the rank's own offsets.

    Sizing it from ``max(local offsets)`` would put the rank back into the jit
    signature through the dtype — the same defect one level down.
    """
    assert jcc._slice_index_dtype((8, 5)) is np.int32
    assert jcc._slice_index_dtype((2 ** 31 + 4, 3)) is np.int64
    # every rank of one run sees the same global shape, hence the same dtype
    assert jcc._slice_index_dtype((8, 5)) is jcc._slice_index_dtype((8, 5))
