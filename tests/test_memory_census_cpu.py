"""lxkit.memory on CPU: prices, receipts and the instrument's readings (MEMORY_MODEL_AUDIT §5)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax                                                             # noqa: E402
import jax.numpy as jnp                                                # noqa: E402
import pytest                                                          # noqa: E402

from common import timing                                              # noqa: E402
from lxkit import memory                                               # noqa: E402

MB = 1 << 20


@pytest.fixture(autouse=True)
def _fresh():
    memory.reset()
    timing.reset()
    yield
    memory.reset()


def test_price_terms_and_ranks():
    p = memory.Price(operands=memory.Term(spread=16e9, source="shardings"),
                     xla=memory.Term(fixed=1e9, source="memory_analysis"), ranks=4)
    assert p.bytes() == 5e9 and p.bytes(16) == 2e9
    assert p.unsourced() == []
    assert memory.Price(vendor=memory.Term(fixed=1.0)).unsourced() == ["vendor"]
    both = p + memory.priced(2e9, "LAPACK heevd", term="vendor", ranks=4)
    assert both.terms()["vendor"] == 2e9 and both.bytes() == 7e9


def test_booked_section_prints_one_receipt(capsys):
    jnp.zeros(4).block_until_ready()                  # the backend is up
    memory.book(memory.priced(64 * MB, "test closed form"), stage="t.stage")
    with timing.section("t.outer"):
        with timing.section("t.stage"):
            x = jnp.ones((16 * MB,), jnp.uint8)
            x.block_until_ready()
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("memory t.stage:")]
    assert len(lines) == 1, out
    assert "priced 0.07 GB (operands 0.00 GB, xla 0.07 GB, vendor 0.00 GB, outside 0.00 GB)" in lines[0]
    assert "| measured live " in lines[0] and "| γ " in lines[0]
    (row,) = memory.receipts()
    assert row["path"] == ("t.outer", "t.stage") and row["measured"] >= 16 * MB
    assert memory.pending() == {}
    del x


def test_book_inside_open_section_meters_from_the_booking():
    jnp.zeros(4).block_until_ready()
    with timing.section("t.own"):
        held = jnp.ones((32 * MB,), jnp.uint8).block_until_ready()
        del held
        memory.book(memory.priced(1 * MB, "after the release"))
    (row,) = memory.receipts()
    assert row["stage"] == "t.own" and row["measured"] < 32 * MB


class _Arena:
    """A BFC arena whose peak never resets (ROCm)."""
    def __init__(self):
        self.use, self.peak = 0, 0

    def set(self, use):
        self.use = use
        self.peak = max(self.peak, use)

    def memory_stats(self):
        return {"bytes_in_use": self.use, "peak_bytes_in_use": self.peak}


def test_bfc_first_raise_is_exact_and_a_lower_stage_is_a_lower_bound(monkeypatch):
    arena = _Arena()
    monkeypatch.setattr(memory, "_backend_up", lambda: True)
    monkeypatch.setattr(memory, "_sync", lambda: None)
    monkeypatch.setitem(memory._INST, "source", "bfc")
    monkeypatch.setitem(memory._INST, "device", arena)
    monkeypatch.setitem(memory._INST, "mark", 0)
    monkeypatch.setitem(memory._INST, "poll", 0)
    with memory.stage("raise", memory.priced(10e9, "test")):
        arena.set(9e9)
        arena.set(1e9)
    with memory.stage("below", memory.priced(5e9, "test")):
        arena.set(4e9)                                # under the 9 GB run mark
        with memory._INST_LOCK:                       # what the sampler would have polled
            memory._INST["poll"] = max(memory._INST["poll"], 4e9)
        arena.set(1e9)
    rows = {r["stage"]: r for r in memory.receipts()}
    assert rows["raise"]["measured"] == 9e9 and rows["raise"]["exact"]
    assert rows["below"]["measured"] == 4e9 and not rows["below"]["exact"]
    line = memory.receipt_line("below", rows["below"]["price"], 4e9, False, "bfc")
    assert "| measured >=4.00 GB | γ 0.80" in line
