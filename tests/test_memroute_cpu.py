"""The device budget cap, the local-route admission and its record (CPU).

* ``memory_per_device_gb`` above the memory rule never raises the budget (72 on a
  40 GB card runs at the rule); below it, it lowers it.
* A caller-owned BFC arena caps the rule; the runtime's cuda_async pool does not.
* ``admit_local``: local when it fits; the distributed route otherwise, with the
  provider contexts charged under ``linalg = local``; the route less over, warned,
  when neither fits; an unavailable distributed route keeps local, warned.
* Each decision prints once on rank 0 and lands in the run's Routes table.
* The W Dyson plan (charge and photon) admits local by its whole-matrix price.
* A cuFFT pricing failure warns and prices the scratch conservatively.
"""
from types import SimpleNamespace as NS

import pytest

from common import gpu_utils

GB = 1e9


@pytest.fixture(autouse=True)
def _fresh_routes(monkeypatch):
    monkeypatch.setattr(gpu_utils, "_ROUTES", {})
    monkeypatch.setattr(gpu_utils, "_LAST_DECISION", {})
    monkeypatch.setattr(gpu_utils, "_RUN_LINALG", "local")
    monkeypatch.setattr(gpu_utils, "_RUN_DEVICE_BUDGET_GB", None)
    yield


def _rule(monkeypatch, gb):
    monkeypatch.setattr(gpu_utils, "get_device_memory_gb", lambda linalg="local": gb)
    monkeypatch.setattr(gpu_utils, "minimum_process_budget_gb", lambda local: float(local))


def test_deck_above_rule_runs_at_rule(monkeypatch):
    _rule(monkeypatch, 31.6)
    with pytest.warns(RuntimeWarning, match="running at 31.60 GB"):
        assert gpu_utils.resolve_device_budget_gb(72.0, "local") == 31.6
    assert gpu_utils.device_budget_bytes() == 31.6 * GB


def test_deck_below_rule_lowers_it(monkeypatch):
    _rule(monkeypatch, 31.6)
    assert gpu_utils.resolve_device_budget_gb(20.0, "distributed") == 20.0
    assert gpu_utils._RUN_LINALG == "distributed"
    assert gpu_utils.resolve_device_budget_gb(0.0, "local") == 31.6


def test_enforced_pool_limit_only_for_bfc(monkeypatch):
    device = NS(memory_stats=lambda: {"bytes_limit": 30 * 10**9})
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    assert gpu_utils._enforced_pool_limit(device) == 0
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    assert gpu_utils._enforced_pool_limit(device) == 30 * 10**9
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR")
    assert gpu_utils._enforced_pool_limit(device) == 30 * 10**9


def test_local_when_it_fits_and_prints_once(capsys):
    for _ in range(3):
        assert gpu_utils.admit_local("W Dyson 8 x 64^2", 1 * GB, 30 * GB)
    out = [line for line in capsys.readouterr().out.splitlines() if "route " in line]
    assert out == ["  route W Dyson 8 x 64^2: local (1.00 GB/rank of room 30.00 GB)"]
    (row,) = gpu_utils.routes()
    assert row["route"] == "local" and row["count"] == 3 and row["warning"] is None


def test_distributed_layout_charges_no_contexts():
    gpu_utils._RUN_LINALG = "distributed"
    assert not gpu_utils.admit_local("site", 40 * GB, 30 * GB, distributed_bytes=28 * GB)
    assert gpu_utils.routes()[0]["route"] == "distributed"


def test_local_layout_charges_provider_contexts():
    # 28 GB fits 30 alone, but not with the 3 GB of contexts the local rule never withheld.
    assert gpu_utils.admit_local("site", 40 * GB, 30 * GB, distributed_bytes=28 * GB) is False
    row = gpu_utils.routes()[0]
    assert row["route"] == "distributed" and row["warning"] is not None
    assert "linalg = distributed" in row["warning"] and "provider contexts 3.00 GB" in row["reason"]
    # Both over: the distributed route runs, warned (no fail-open onto one rank).
    assert gpu_utils.admit_local("other", 31 * GB, 30 * GB, distributed_bytes=29 * GB) is False
    assert gpu_utils.routes()[1]["route"] == "distributed" and gpu_utils.routes()[1]["warning"]
    # A route that opens no vendor context (an XLA program) is not charged.
    assert gpu_utils.admit_local("xla", 40 * GB, 30 * GB, distributed_bytes=28 * GB,
                                 provider=False) is False
    assert gpu_utils.routes()[2]["warning"] is None


def test_lazy_distributed_price_and_probe():
    calls = []
    assert gpu_utils.admit_local("fits", 1, 2, distributed_bytes=lambda: calls.append(1) or 0)
    assert calls == []

    def refused():
        raise ValueError("1-D mesh")
    assert gpu_utils.admit_local("probe", 5 * GB, 1 * GB, probe=refused) is True
    row = gpu_utils.routes()[-1]
    assert row["route"] == "local" and "1-D mesh" in row["warning"]


def test_forced_and_quiet():
    assert gpu_utils.admit_local("SC eigh", 0, 0, forced="distributed",
                                 names=("native", "distributed")) is False
    assert gpu_utils.routes()[0]["reason"] == "linalg = distributed"
    assert gpu_utils.admit_local("constructor", 5 * GB, 4 * GB, names=("local", "face"),
                                 quiet=True) is False
    assert len(gpu_utils.routes()) == 1
    assert gpu_utils.last_decision()["route"] == "face"


def test_budget_resolution_starts_a_new_record(monkeypatch):
    gpu_utils.admit_local("first run", 1, 2)
    _rule(monkeypatch, 31.6)
    gpu_utils.resolve_device_budget_gb(0.0, "local")
    assert gpu_utils.routes() == []


def test_report_lists_routes():
    from gw.production_report import GWProductionReport
    gpu_utils.admit_local("W Dyson 8 x 64^2", 1 * GB, 30 * GB)
    gpu_utils.admit_local("W Dyson 8 x 64^2", 1 * GB, 30 * GB)
    lines = []
    fake = NS(heading=lambda title: lines.append(f"== {title}"), emit=lines.append)
    GWProductionReport._routes(fake)
    assert lines == ["== Routes",
                     "  route W Dyson 8 x 64^2: local (1.00 GB/rank of room 30.00 GB) [x2]"]


def test_w_dyson_route(monkeypatch):
    import gw.w_isdf as w
    monkeypatch.setattr(gpu_utils, "device_budget_bytes", lambda: 31.6 * GB)
    import distrib_la
    monkeypatch.setattr(distrib_la, "resolve_backend", lambda *a, **k: "cusolvermp")
    mesh = NS(size=4)
    # (route, room): the room rides with the distributed plan (ae374ca4c), None for local / P1.
    assert w._dyson_route("local", 64, 8, mesh) == ("local", None)
    assert w._dyson_route("local", 64, 8, NS(size=1)) == ("local", None)
    route, room = w._dyson_route("distributed", 64, 8, mesh)
    assert route == "distributed" and room == int(31.6 * GB - 16.0 * 64 * 64 * 3 * 8 / 4)
    # n = 8192, 64 parents at P4: whole matrices 16 n^2 (3*16 + 2) = 53.7 GB/rank beside
    # 51.5 GB of faces; the distributed 52.1 + 3 GB of contexts is over too, but less.
    assert w._dyson_route("local", 8192, 64, mesh)[0] == "distributed"
    assert gpu_utils.routes()[-1]["warning"]
    # n = 4096, 16 parents: local 3.2 + 3.8 GB fits 31.6.
    assert w._dyson_route("local", 4096, 16, mesh) == ("local", None)
    # n = 6144, 16 parents: local 7.2 + 8.5 = 15.7 GB > 12 GB; distributed 7.5 + 3 GB of contexts fits.
    monkeypatch.setattr(gpu_utils, "device_budget_bytes", lambda: 12 * GB)
    assert w._dyson_route("local", 6144, 16, mesh)[0] == "distributed"
    row = gpu_utils.routes()[-1]
    assert "provider contexts" in row["reason"] and row["warning"] is None


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
