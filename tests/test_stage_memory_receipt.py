"""The per-stage device-memory receipt: own peaks per timing section, and γ = peak / price.

A scripted pool stands in for the CUDA pool high-water mark
(``runtime.xla_memory.pool_high_water``), so the arithmetic is checked without a GPU.
"""
import re
from types import SimpleNamespace

import pytest

import runtime.xla_memory as xla_memory
from common import gpu_utils, timing
from gw.production_report import GWProductionReport


class _Pool:
    """cudaMallocAsync's USED_MEM_HIGH: the max since the last reset; a reset sets it to now."""

    def __init__(self):
        self.cur = self.high = 0

    def alloc(self, n):
        self.cur += n
        self.high = max(self.high, self.cur)

    def free(self, n):
        self.cur -= n

    def read(self, *, reset=False):
        high = self.high
        if reset:
            self.high = self.cur
        return high


@pytest.fixture
def pool(monkeypatch):
    p = _Pool()
    monkeypatch.setattr(xla_memory, "pool_high_water", p.read)
    monkeypatch.setattr(xla_memory, "_POOL", dict(xla_memory._POOL, source="pool"))
    return p


def test_a_stage_below_an_earlier_high_water_mark_keeps_its_own_peak(pool):
    """The case peak_bytes_in_use cannot see: a later, smaller stage."""
    c = timing.TimingCollector()
    pool.alloc(1)
    with c.section("big"):
        pool.alloc(20)
        pool.free(20)
    with c.section("small"):
        pool.alloc(5)
        with c.section("inner"):
            pool.alloc(3)
            pool.free(3)
        pool.free(5)
    peaks = {r["path"]: (r["peak"], r["peak_self"]) for r in c.records()}
    assert peaks[("big",)] == (21, 21)
    assert peaks[("small",)] == (9, 6)          # inner's 9 counts in small, not in its self
    assert peaks[("small", "inner")] == (9, 9)
    c.gather_peaks()
    small = next(r for r in c.records() if r["path"] == ("small",))
    assert small["peak_ranks"] == [9.0] and small["peak_self_ranks"] == [6.0]


def test_negative_control_without_reset_the_small_stage_reads_the_big_one(pool, monkeypatch):
    """A cumulative mark (the fallback) must show the defect the reset removes."""
    monkeypatch.setattr(xla_memory, "pool_high_water",
                        lambda *, reset=False: pool.high)
    c = timing.TimingCollector()
    with c.section("big"):
        pool.alloc(20)
        pool.free(20)
    with c.section("small"):
        pool.alloc(5)
        pool.free(5)
    assert next(r for r in c.records() if r["path"] == ("small",))["peak"] == 20


def _row(path, seconds, peak, peak_self):
    return dict(name=path[-1], path=path, inclusive=seconds, peak=peak, peak_self=peak_self,
                peak_ranks=[peak, peak - 1e8], peak_self_ranks=[peak_self, peak_self - 1e8])


def test_stage_table_prints_gamma_per_row_or_no_planner(tmp_path, monkeypatch):
    rows = [
        _row(("gw_jax.screening",), 20.0, 20.59e9, 2e9),
        _row(("gw_jax.screening", "gw_jax.chi0_W"), 8.0, 12.03e9, 1e9),
        _row(("gw_jax.screening", "gw_jax.chi0_W", "chi.compile"), 3.0, 10.30e9, 10.30e9),
        _row(("gw_jax.screening", "gw_jax.chi0_W", "chi.exec"), 4.0, 12.03e9, 12.03e9),
        _row(("gw_jax.screening", "gw_jax.chi0_W_probe"), 8.0, 20.59e9, 1e9),
        _row(("gw_jax.screening", "gw_jax.chi0_W_probe", "chi.compile"), 3.0, 10.30e9, 10.30e9),
        _row(("gw_jax.screening", "gw_jax.chi0_W_probe", "chi.exec"), 4.0, 20.59e9, 20.59e9),
        _row(("gw_jax.sigma",), 30.0, 23.30e9, 1e9),
        _row(("gw_jax.sigma", "sigma.hartree"), 5.0, 11.58e9, 11.58e9),
        _row(("gw_jax.sigma", "sigma.tau_sweep"), 20.0, 23.30e9, 23.30e9),
        _row(("gw_jax.sigma", "sigma.tau_sweep", "tau.setup"), 2.0, 16.63e9, 16.63e9),
    ]
    monkeypatch.setattr(gpu_utils, "_STAGE_PRICES", [
        # priced during the compile; judged against the same role's chi.exec
        dict(stage="chi0 static", bytes=9.95e9, section="chi.exec",
             path=("gw_jax.screening", "gw_jax.chi0_W", "chi.compile")),
        dict(stage="chi0 probe", bytes=18.5e9, section="chi.exec",
             path=("gw_jax.screening", "gw_jax.chi0_W_probe", "chi.compile")),
        # priced in tau.setup; judged against the enclosing sweep
        dict(stage="Sigma tau", bytes=12.85e9, section="sigma.tau_sweep",
             path=("gw_jax.sigma", "sigma.tau_sweep", "tau.setup")),
    ])
    monkeypatch.setattr(gpu_utils, "_RUN_DEVICE_BUDGET_GB", 20.0)
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(str(path), runtime=SimpleNamespace(process_index=0),
                                debug=False, stdout=lambda line: None)
    report.timings(rows, wall=60.0)
    report.finish()
    text = path.read_text()
    table = text.split("MAJOR-STAGE DEVICE MEMORY")[1]
    assert "budget memory_per_device_gb = 20.00 GB; run peak 23.30 GB (OVER)" in table
    line = lambda label: next(l for l in table.splitlines() if l.strip().startswith(label))
    assert "20.59 /  20.49" in line("chi0") and "18.50" in line("chi0")
    assert " 1.11  " in line("chi0") and line("chi0").endswith("gw_jax.chi0_W_probe > chi.exec")
    assert " 1.81  " in line("Sigma tau other")                     # the sweep's own 23.30
    assert "16.63" in line("Sigma tau setup") and "no planner" in line("Sigma tau setup")
    assert "no planner" in line("Sigma Hartree")
    assert "gw_jax.chi0_W > chi.exec peak 12.03 / 11.93 GB; γ 1.21" in table
    # The sandbox parsers' stage-row pattern still reads the timing table and
    # never mistakes a memory row for a stage.
    stage_row = re.compile(r"^\s{2}(?P<name>\S.*?)\s{2,}(?P<wall>[\d.Ee+-]+)\s+"
                           r"(?P<fraction>[\d.Ee+-]+)%\s*$")
    assert not any(stage_row.match(l) for l in table.splitlines())
    assert {"chi0", "Sigma tau other", "total run"} <= {
        m["name"] for m in map(stage_row.match, text.splitlines()) if m}


def test_no_peaks_no_table(tmp_path):
    """A CPU run (no pool) keeps the report as it was."""
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(str(path), runtime=SimpleNamespace(process_index=0),
                                debug=False, stdout=lambda line: None)
    report.timings([dict(name="gw_jax.sigma", path=("gw_jax.sigma",), inclusive=1.0)], wall=1.0)
    report.finish()
    assert "DEVICE MEMORY" not in path.read_text()
