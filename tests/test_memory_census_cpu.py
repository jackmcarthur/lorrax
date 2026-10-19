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
    monkeypatch.setitem(memory._I, "source", "bfc")
    monkeypatch.setitem(memory._I, "device", arena)
    monkeypatch.setitem(memory._I, "mark", 0)
    monkeypatch.setitem(memory._I, "poll", 0)
    with memory.stage("raise", memory.priced(10e9, "test")):
        arena.set(9e9)
        arena.set(1e9)
    with memory.stage("below", memory.priced(5e9, "test")):
        arena.set(4e9)                                # under the 9 GB run mark
        with memory._LOCK:                       # what the sampler would have polled
            memory._I["poll"] = max(memory._I["poll"], 4e9)
        arena.set(1e9)
    rows = {r["stage"]: r for r in memory.receipts()}
    assert rows["raise"]["measured"] == 9e9 and rows["raise"]["exact"]
    assert rows["below"]["measured"] == 4e9 and not rows["below"]["exact"]
    line = memory.receipt_line("below", rows["below"]["price"], 4e9, False, "bfc")
    assert "| measured >=4.00 GB | γ 0.80" in line


# ---- the admission policy (owner 2026-10-19) and the sizing estimate -----------------------
GB = 1e9


def _run(budget=10 * GB, limit=12 * GB):
    memory.set_budget(budget, limit_bytes=limit, per_node=4, platform="NVIDIA A100-SXM4-80GB")


def test_admit_takes_the_fastest_route_that_fits():
    _run()
    routes = (("local", memory.priced(11 * GB, "t")), ("mesh", memory.priced(4 * GB, "t")),
              ("tiny", memory.priced(1 * GB, "t")))
    assert memory.admit("s", routes, 10 * GB) == "mesh"


def test_admit_within_the_limit_warns_and_continues():
    _run()
    with pytest.warns(RuntimeWarning, match="memory over budget at s: local needs 11.00 GB"):
        assert memory.admit("s", (("local", memory.priced(11 * GB, "t")),), 10 * GB) == "local"
    assert memory._DOOMED == {}


def test_admit_over_the_limit_banners_and_never_raises(capsys):
    _run()
    big = memory.Price(operands=memory.Term(spread=16 * 11 * GB, source="faces"), ranks=4)
    with pytest.warns(RuntimeWarning, match="will not fit"):
        assert memory.admit("big stage", big, 10 * GB, runs_in="t.big") == "chosen"
    out = capsys.readouterr().out
    assert "!! MEMORY: big stage WILL NOT FIT" in out and "short by 32.00 GB" in out
    assert "minimum to complete: P = 16 (4 nodes of NVIDIA A100-SXM4-80GB)" in out
    with timing.section("t.big"):                     # repeated at stage entry
        pass
    assert "!! MEMORY: big stage WILL NOT FIT" in capsys.readouterr().out


def test_sizing_finds_exactly_sixteen():
    stage = {"s": (("faces", memory.Price(operands=memory.Term(spread=16 * 9 * GB, source="f"),
                                          ranks=4)),)}
    got = memory.sizing(stage, limit_bytes=10 * GB, per_node=4, platform="A100")
    assert got["p"] == 16 and got["nodes"] == 4 and got["binding"] == "s"
    assert got["line"] == "minimum to complete: P = 16 (4 nodes of A100); binding stage s"


def test_sizing_says_no_p_suffices_for_a_replicated_stage():
    stage = {"replicated": (("local", memory.priced(12 * GB, "whole matrix", term="operands")),)}
    got = memory.sizing(stage, limit_bytes=10 * GB, per_node=4)
    assert got["p"] is None and got["binding"] == "replicated"
    assert got["line"].startswith("minimum to complete: no P suffices: replicated (local) holds "
                                  "12.00 GB/rank of operands that no mesh divides")
    assert "replicated operands sharded over the mesh" in got["line"]


def test_vendor_rows_and_strict_range(monkeypatch):
    from lxkit import LEGS
    term = memory.vendor(LEGS["cpu"], "lapack", "heevd", 64, 1, 16)
    assert term.fixed == 16 * (2 * 64 + 64 ** 2) + 8 * (1 + 5 * 64 + 2 * 64 ** 2) + 4 * (3 + 5 * 64)
    assert "LAPACK" in term.source
    monkeypatch.setattr(memory, "STRICT", True)
    with pytest.raises(RuntimeError, match="GATE memory_unpriced: .*outside the measured"):
        memory.vendor(LEGS["CUDA"], "cusolvermp", "solve_lu", 30000, 16, 16, unit=1.0)
    memory.remember("cusolvermp", "solve_lu", 30000, 16, 16, 7e9, "test query")
    assert memory.vendor(LEGS["CUDA"], "cusolvermp", "solve_lu", 30000, 16, 16).fixed == 7e9
    memory._VENDOR.clear()


# ---- the census (memory.STRICT) -------------------------------------------------------------

def test_census_flags_unpriced_allocations_and_missing_provenance(monkeypatch):
    jnp.zeros(4).block_until_ready()
    monkeypatch.setattr(memory, "STRICT", True)
    keep = []
    with timing.section("t.unpriced"):
        keep.append(jnp.ones((160 * MB,), jnp.uint8).block_until_ready())
    memory.book(memory.priced(100 * MB, "test"), stage="t.priced")
    with timing.section("t.priced"):
        with timing.section("t.child"):                # inside a priced stage: fine
            keep.append(jnp.ones((80 * MB,), jnp.uint8).block_until_ready())
    with timing.section("t.small"):
        keep.append(jnp.ones((8 * MB,), jnp.uint8).block_until_ready())
    memory.admit("t.route", (("a", memory.Price(xla=memory.Term(1.0))),), 2.0)
    got = memory.findings()
    assert len(got) == 2, got
    assert got[0].startswith("unpriced: t.unpriced allocated ")
    assert got[1] == "no provenance: t.route (a): xla"
    del keep


def test_run30k_plan_census_is_clean_under_strict():
    """Every price the run30k plan books or admits has provenance and every measured row is
    used inside its range (the replay of the 16-device plan, tests/replays)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, LORRAX_MEMORY_STRICT="1", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=16")
    run = subprocess.run([sys.executable, str(root / "tests/replays/run30k_plan_census.py")],
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    assert '"refusals": []' in run.stdout
    assert "CENSUS []" in run.stdout, [l for l in run.stdout.splitlines() if "CENSUS" in l]
