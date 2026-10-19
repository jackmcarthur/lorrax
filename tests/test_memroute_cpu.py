"""The device budget, route admission and its record (CPU): ``lxkit.memory``.

* ``memory_per_device_gb`` above the model's budget never raises it; below, it lowers it.
* A BFC arena's ``bytes_limit`` is the limit; cuda_async's is a release threshold.
* ``admit``: the fastest route that fits; the provider contexts charged under
  ``linalg = local``; an unavailable later route keeps the first, warned; forced/quiet.
* Each decision prints once on rank 0 and lands in the run's Routes table.
* The W Dyson plan admits local by its whole-matrix price.
* A cuFFT pricing failure warns and prices the scratch conservatively.
"""
from types import SimpleNamespace as NS

import pytest

from lxkit import memory

GB = 1e9
P = memory.priced


@pytest.fixture(autouse=True)
def _fresh():
    memory.reset()
    memory.set_budget(30 * GB, limit_bytes=36 * GB)
    yield
    memory.reset()


def test_deck_above_budget_runs_at_budget():
    rule = memory.resolve(0.0, "local", total=64 * GB)
    with pytest.warns(RuntimeWarning, match=f"running at {rule / GB:.2f} GB"):
        assert memory.resolve(72 * GB, "local", total=64 * GB) == rule
    assert memory.resolve(10 * GB, "distributed", total=64 * GB) == 10 * GB
    assert memory.run()["linalg"] == "distributed"


def test_enforced_limit_only_for_bfc(monkeypatch):
    stats = {"bytes_limit": 30 * 10**9}
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    assert not memory._enforced(stats)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    assert memory._enforced(stats)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR")
    assert memory._enforced(stats)


def test_fastest_that_fits_prints_once(capsys):
    for _ in range(3):
        assert memory.admit("W Dyson 8 x 64^2", (("local", P(1 * GB, "t")),
                                                 ("distributed", P(0, "t"))), 30 * GB) == "local"
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
                        4 * GB, quiet=True) == "face"
    assert len(memory.routes()) == n and memory.last_decision()["route"] == "face"


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
