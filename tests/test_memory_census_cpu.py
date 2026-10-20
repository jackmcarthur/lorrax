"""lxkit.memory on CPU through its public API: prices, receipts, admission and its record,
sizing, the run budget, census.  A deck above the model's budget never raises it; a BFC arena's
``bytes_limit`` is the limit (cuda_async's a release threshold); the provider contexts are
charged under ``linalg = local``; each route decision prints once and lands in the Routes table;
a cuFFT pricing failure prices the scratch conservatively."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from types import SimpleNamespace as NS                               # noqa: E402

import jax.numpy as jnp                                                # noqa: E402
import pytest                                                          # noqa: E402

from common import timing                                              # noqa: E402
from lxkit import LEGS, memory                                         # noqa: E402

MB, GB = 1 << 20, 1e9
P = memory.priced


@pytest.fixture(autouse=True)
def _fresh():
    memory.reset()
    timing.reset()
    memory.set_budget(30 * GB, limit_bytes=36 * GB)
    yield
    memory.reset()


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


def test_census_flags_missing_provenance(calibrate, monkeypatch):
    """An unpriced allocation is the trace's residual (lxkit.memory_residuals); the census lists
    terms without provenance."""
    monkeypatch.setattr(memory, "STRICT", True)
    memory.admit("t.route", (("a", memory.Price(xla=memory.Term(1.0))),), 2.0)
    assert memory.findings() == ["no provenance: t.route (a): xla"]


def test_deck_above_budget_runs_at_budget():
    rule = memory.resolve(0.0, "local", total=64 * GB)
    with pytest.warns(RuntimeWarning, match=f"running at {rule / GB:.2f} GB"):
        assert memory.resolve(72 * GB, "local", total=64 * GB) == rule
    assert memory.resolve(10 * GB, "distributed", total=64 * GB) == 10 * GB
    assert memory.run()["linalg"] == "distributed"


def test_enforced_limit_only_for_bfc(monkeypatch):
    stats = {"bytes_limit": 30 * 10**9}
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    assert not memory.enforced(stats)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    assert memory.enforced(stats)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR")
    assert memory.enforced(stats)


def test_fastest_that_fits_prints_once(capsys):
    for _ in range(3):
        assert memory.admit("W Dyson 8 x 64^2", (("local", P(1 * GB, "t")),
                                                 ("distributed", P(0, "t"))), 30 * GB) == "local"
    assert memory.admit("s", (("local", P(40 * GB, "t")), ("mesh", P(4 * GB, "t"))), 30 * GB,
                        quiet=True, detail=True)["route"] == "mesh"
    out = [line for line in capsys.readouterr().out.splitlines() if "route " in line]
    assert out == ["  route W Dyson 8 x 64^2: local (1.00 GB/rank of room 30.00 GB)"]
    (row,) = memory.routes()
    assert row["count"] == 3 and row["warning"] is None


def test_provider_contexts_are_charged_under_local_linalg():
    face = lambda: P(28 * GB, "t") + memory.Price(outside=memory.contexts())
    with pytest.warns(RuntimeWarning, match="memory over budget at site"):
        assert memory.admit("site", (("local", P(40 * GB, "t")), ("distributed", face)),
                            30 * GB) == "distributed"
    assert "distributed 31.60 GB/rank over room" in memory.routes()[0]["reason"]
    memory.set_budget(30 * GB, linalg="distributed")
    assert memory.admit("again", (("local", P(40 * GB, "t")), ("distributed", face)),
                        30 * GB) == "distributed"
    assert memory.routes()[-1]["warning"] is None


def test_lazy_price_probe_forced_quiet():
    calls = []
    assert memory.admit("fits", (("a", P(1, "t")), ("b", lambda: calls.append(1) or P(0, "t"))),
                        2) == "a" and calls == []

    def refused():
        raise ValueError("1-D mesh")
    assert memory.admit("probe", (("local", P(5 * GB, "t")), ("distributed", P(0, "t"))),
                        1 * GB, probe=refused) == "local"
    assert "1-D mesh" in memory.routes()[-1]["warning"]
    assert memory.admit("SC eigh", (), 0, forced="distributed") == "distributed"
    n = len(memory.routes())
    assert memory.admit("constructor", (("local", P(5 * GB, "t")), ("face", P(0, "t"))),
                        4 * GB, quiet=True, detail=True)["route"] == "face"
    assert len(memory.routes()) == n


def test_report_lists_routes():
    from gw.production_report import GWProductionReport
    for _ in range(2):
        memory.admit("W Dyson 8 x 64^2", (("local", P(1 * GB, "t")), ("d", P(0, "t"))), 30 * GB)
    lines = []
    GWProductionReport._routes(NS(heading=lambda t: lines.append(f"== {t}"), emit=lines.append))
    assert lines == ["== Routes",
                     "  route W Dyson 8 x 64^2: local (1.00 GB/rank of room 30.00 GB) [x2]"]


def test_w_dyson_route(monkeypatch):
    import distrib_la
    import gw.w_isdf as w
    from common import gpu_utils
    monkeypatch.setattr(gpu_utils, "device_budget_bytes", lambda: 31.6 * GB)
    monkeypatch.setattr(distrib_la, "resolve_backend", lambda *a, **k: "cusolvermp")
    mesh = NS(size=4)
    assert w._dyson_route("local", 64, 8, mesh) == ("local", None)
    assert w._dyson_route("local", 64, 8, NS(size=1)) == ("local", None)
    route, room = w._dyson_route("distributed", 64, 8, mesh)
    assert route == "distributed" and room == int(31.6 * GB - 16.0 * 64 * 64 * 3 * 8 / 4)
    # n = 4096, 16 parents: local 3.2 + 3.8 GB fits 31.6.
    assert w._dyson_route("local", 4096, 16, mesh) == ("local", None)
    # n = 6144, 16 parents: local 15.7 GB > 12; distributed 7.5 + 3.6 GB of contexts fits.
    monkeypatch.setattr(gpu_utils, "device_budget_bytes", lambda: 12 * GB)
    assert w._dyson_route("local", 6144, 16, mesh)[0] == "distributed"
    assert memory.routes()[-1]["warning"] is None


def test_cufft_failures_price_conservatively(monkeypatch, capsys):
    from runtime import aot_memory as aot
    stats = NS(temp_size_in_bytes=100, argument_size_in_bytes=50, output_size_in_bytes=30,
               alias_size_in_bytes=0)
    monkeypatch.setattr(aot, "_fft_plan_api", lambda platform: "cufft")
    drifted = NS(memory_analysis=lambda: stats, as_text=lambda: "%x = c128[8] fft(%y), weird")
    peak = aot.aot_kernel_peak_bytes(drifted, platform="cuda")
    assert peak.cufft_measured is False and peak.cufft_scratch == 130 and peak.total == 180 + 130
    text = ("%fft.0 = c128[2,4,4,4]{3,2,1,0} fft(%x.1), fft_type=FFT, fft_length={4,4,4}")

    def refuse(specs, api="cufft"):
        raise aot.CufftQueryError("libcufft absent")
    monkeypatch.setattr(aot, "query_cufft_workspace_bytes", refuse)
    parsed = NS(memory_analysis=lambda: stats, as_text=lambda: text)
    peak = aot.aot_kernel_peak_bytes(parsed, platform="cuda")
    assert peak.cufft_measured is False and peak.cufft_scratch == 2 * 2 * 64 * 16
    assert "UNAVAILABLE" in capsys.readouterr().out
