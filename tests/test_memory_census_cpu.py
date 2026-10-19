"""lxkit.memory on CPU through its public API: prices, receipts, admission, sizing, census."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax.numpy as jnp                                                # noqa: E402
import pytest                                                          # noqa: E402

from common import timing                                              # noqa: E402
from lxkit import LEGS, memory                                         # noqa: E402

MB, GB = 1 << 20, 1e9


@pytest.fixture(autouse=True)
def _fresh():
    memory.reset()
    timing.reset()
    yield


@pytest.fixture
def calibrate(monkeypatch):
    """``LORRAX_MEMORY_RECEIPTS=calibrate``: synced, exact, every section read."""
    monkeypatch.setattr(memory, "CALIBRATE", True)


def _run(budget=10 * GB, limit=12 * GB):
    memory.set_budget(budget, limit_bytes=limit, per_node=4, platform="NVIDIA A100-SXM4-80GB")


def test_price_terms_and_ranks():
    p = memory.Price(operands=memory.Term(spread=16e9, source="shardings"),
                     xla=memory.Term(fixed=1e9, source="memory_analysis"), ranks=4)
    assert p.bytes() == 5e9 and p.bytes(16) == 2e9 and p.unsourced() == []
    assert memory.Price(vendor=memory.Term(fixed=1.0)).unsourced() == ["vendor"]
    both = p + memory.priced(2e9, "LAPACK heevd", term="vendor", ranks=4)
    assert both.terms()["vendor"] == 2e9 and both.bytes() == 7e9


def test_a_stage_prices_the_sum_of_its_parts_and_prints_one_receipt(calibrate, capsys):
    jnp.zeros(4).block_until_ready()                  # the backend is up
    memory.book(memory.priced(32 * MB, "model"), stage="t.stage")
    memory.book(memory.priced(16 * MB, "model"), stage="t.stage")     # same part: the larger
    memory.book(memory.priced(32 * MB, "eigh"), stage="t.stage")      # another part: added
    with timing.section("t.outer"):
        with timing.section("t.stage"):
            x = jnp.ones((16 * MB,), jnp.uint8).block_until_ready()
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("memory t.stage:")]
    assert len(lines) == 1 and "priced 0.07 GB" in lines[0] and "| measured live " in lines[0]
    (row,) = memory.receipts()
    assert row["measured"] >= 16 * MB and memory.findings() == []
    del x


def test_admit_takes_the_fastest_route_that_fits():
    _run()
    routes = (("local", memory.priced(11 * GB, "t")), ("mesh", memory.priced(4 * GB, "t")),
              ("tiny", memory.priced(1 * GB, "t")))
    assert memory.admit("s", routes, 10 * GB) == "mesh"
    assert memory.admit("s", routes, 10 * GB, detail=True)["route"] == "mesh"


def test_admit_within_the_limit_warns_and_continues(capsys):
    _run()
    with pytest.warns(RuntimeWarning, match="memory over budget at s: local needs 11.00 GB"):
        assert memory.admit("s", (("local", memory.priced(11 * GB, "t")),), 10 * GB) == "local"
    assert "!! MEMORY" not in capsys.readouterr().out


def test_admit_over_the_limit_banners_repeats_at_entry_and_never_raises(capsys):
    _run()
    big = memory.Price(operands=memory.Term(spread=16 * 11 * GB, source="faces"), ranks=4)
    with pytest.warns(RuntimeWarning, match="priced over the limit"):
        assert memory.admit("big stage", big, 10 * GB, runs_in="t.big") == "chosen"
    out = capsys.readouterr().out
    assert "!! MEMORY: big stage is PRICED OVER the allocator limit" in out and "short by 32.00 GB" in out
    assert "minimum to complete: P = 16 (4 nodes of NVIDIA A100-SXM4-80GB)" in out
    with timing.section("t.big"):
        pass
    assert "!! MEMORY: big stage is PRICED OVER" in capsys.readouterr().out


def test_host_and_tile_rooms_neither_banner_nor_size():
    _run()
    with pytest.warns(RuntimeWarning, match="memory over budget at host"):
        memory.admit("host", memory.priced(50 * GB, "t", term="operands"), 1 * GB, device=False)
    assert memory.sizing()["line"] == "minimum to complete: no priced stage"


def test_sizing_takes_every_square_mesh_and_names_the_least_headroom():
    spread = lambda gb: memory.Price(operands=memory.Term(spread=16 * gb * GB, source="f"), ranks=4)
    got = memory.sizing({"a": (("f", spread(5)),), "b": (("f", spread(4.4)),)},
                        limit_bytes=10 * GB, per_node=4, platform="A100")
    assert got["p"] == 9 and got["nodes"] == 3 and got["binding"] == "a"
    assert got["line"] == "minimum to complete: P = 9 (3 nodes of A100); binding stage a"


def test_sizing_says_no_p_suffices_for_a_replicated_stage():
    got = memory.sizing({"replicated": (("local", memory.priced(12 * GB, "whole matrix",
                                                                 term="operands")),)},
                        limit_bytes=10 * GB, per_node=4)
    assert got["p"] is None and got["binding"] == "replicated"
    assert got["line"].startswith("minimum to complete: no P suffices: replicated holds "
                                  "12.00 GB/rank of operands that no mesh divides")


def test_vendor_rows_and_strict_range(monkeypatch):
    """Measured rows remain only where no compile or query exists: no heevd row on any leg
    (XLA holds the local solver's workspace in the compiled program)."""
    assert not [k for leg in LEGS.values() for k, _ in leg.vendor_rows if k[1] == "heevd"]
    monkeypatch.setattr(memory, "STRICT", True)
    with pytest.raises(RuntimeError, match="GATE memory_unpriced: .*outside the measured"):
        memory.vendor(LEGS["CUDA"], "cusolvermp", "solve_lu", 30000, 16, 16, unit=1.0)
    with pytest.raises(RuntimeError, match="GATE memory_unpriced: .*no measured row"):
        memory.vendor(LEGS["ROCM"], "rocsolver", "heevd", 64, 1, 16)


def test_production_instrument_reads_priced_stages_only_on_cpu(monkeypatch):
    """Default receipts on the CPU leg: an unpriced section reads nothing; a priced one twice."""
    monkeypatch.setattr(memory, "CALIBRATE", False)
    x = jnp.ones((256, 256)).block_until_ready()
    reads = memory.instrument_cost()[0]
    for _ in range(20):
        with timing.section("hot.loop"):
            x = x + 1
    assert memory.instrument_cost()[0] == reads
    memory.book(memory.priced(1e6, "test"), stage="priced")
    with timing.section("priced"):
        x = x + 1
    assert memory.instrument_cost()[0] == reads + 2


def test_census_flags_unpriced_allocations_and_missing_provenance(calibrate, monkeypatch):
    jnp.zeros(4).block_until_ready()
    monkeypatch.setattr(memory, "STRICT", True)
    keep = []
    with timing.section("t.unpriced"):
        keep.append(jnp.ones((160 * MB,), jnp.uint8).block_until_ready())
    memory.book(memory.priced(100 * MB, "test"), stage="t.priced")
    with timing.section("t.priced"):
        with timing.section("t.child"):                # inside a priced stage: fine
            keep.append(jnp.ones((80 * MB,), jnp.uint8).block_until_ready())
    memory.admit("t.route", (("a", memory.Price(xla=memory.Term(1.0))),), 2.0)
    got = memory.findings()
    assert len(got) == 2 and got[0].startswith("unpriced: t.unpriced allocated ")
    assert got[1] == "no provenance: t.route (a): xla"
    del keep


@pytest.fixture(scope="module")
def run30k():
    """The replay of run30k's 16-device plan under STRICT (tests/replays), run once."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, LORRAX_MEMORY_STRICT="1", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=16")
    run = subprocess.run([sys.executable, str(root / "tests/replays/run30k_plan_census.py")],
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    return run.stdout


def _sizing(stdout):
    return next(l for l in stdout.splitlines() if l.startswith("SIZING "))


def test_run30k_plan_census_is_clean_under_strict(run30k):
    """Every price the run30k plan books or admits has provenance and every measured row is
    used inside its range."""
    assert '"refusals": []' in run30k
    assert "CENSUS []" in run30k, [l for l in run30k.splitlines() if "CENSUS" in l]


@pytest.mark.xfail(strict=True, reason="compile-only prices on XLA:CPU over-price the A100 programs (the checked eigh chain 10.0 L against 7.4-8.0 L measured on 2x2, the face program 47.9 against 31.29 GB), so sizing on the CPU replay asks P = 25 (passivity_held 83.8 GB at P = 16, its eigh stacks 40.7 GB compiled on CPU); run30k ran at P = 16 (the GPU trace decides)")
def test_run30k_minimum_p_is_the_run(run30k):
    """run30k completed at P = 16 (4 A100-80GB nodes, 66.02 GB peak of 77.89)."""
    assert "minimum to complete: P = 16 (4 nodes of NVIDIA A100-SXM4-80GB)" in _sizing(run30k)


def test_run30k_binds_on_passivity_held(run30k):
    assert "binding stage spole.passivity_held" in _sizing(run30k)
