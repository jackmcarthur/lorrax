"""rank0_read_broadcast: only rank 0 opens the store; every rank gets its result or its refusal."""
import numpy as np
import pytest

import common.collectives as collectives
from file_io import mpa_store


class _KV:
    def __init__(self):
        self.store = {}

    def key_value_set(self, key, value):
        self.store[key] = value

    def blocking_key_value_get(self, key, timeout_ms):
        return self.store[key]


def _as_rank(monkeypatch, rank, client):
    from jax._src.distributed import global_state
    monkeypatch.setattr(collectives, "process_count", lambda: 2)
    monkeypatch.setattr(collectives, "process_rank", lambda: rank)
    monkeypatch.setattr(global_state, "client", client)
    # Same per-occurrence key on both "ranks": replay the counter.
    monkeypatch.setattr(collectives, "_IO_CONTROL_OCCURRENCE", 0)


def test_rank0_reads_and_other_ranks_never_open(monkeypatch):
    kv = _KV()
    ledger = {"blocks_done": np.ones((4, 3), bool), "n_p": 1, "complete": True}
    _as_rank(monkeypatch, 0, kv)
    got0 = mpa_store.rank0_read_broadcast(lambda: ledger, path="/x/store.h5", stage="t")

    def must_not_open():
        raise AssertionError("rank 1 opened the store")

    _as_rank(monkeypatch, 1, kv)
    got1 = mpa_store.rank0_read_broadcast(must_not_open, path="/x/store.h5", stage="t")
    for got in (got0, got1):
        np.testing.assert_array_equal(got["blocks_done"], ledger["blocks_done"])
        assert got["n_p"] == 1 and got["complete"] is True


def test_rank0_refusal_is_raised_on_every_rank(monkeypatch):
    kv = _KV()

    def refuse():
        raise ValueError("MPA Sigma requires a finalized pole fit store")

    for rank in (0, 1):
        _as_rank(monkeypatch, rank, kv)
        with pytest.raises(ValueError, match="finalized pole fit store"):
            mpa_store.rank0_read_broadcast(refuse, path="/x/store.h5", stage="t")
