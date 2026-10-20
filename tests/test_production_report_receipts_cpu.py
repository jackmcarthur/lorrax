"""The end-of-run report on real receipts (C7 crashed there: production_report read a key the
receipts no longer carry, after the physics had finished)."""
from types import SimpleNamespace as NS

import jax.numpy as jnp

from common import timing
from lxkit import memory


def test_end_of_run_report_reads_receipts(tmp_path):
    from gw.production_report import GWProductionReport
    memory.reset(run=True)
    timing.reset()
    memory.set_budget(4e9)
    with timing.section("gw_jax.isdf"):
        with timing.section("spole.gram_reduction"):
            memory.book(memory.priced(1e6, "a compiled figure"))
            jnp.ones((256, 256)).block_until_ready()
    timing.gather_peaks()
    assert memory.receipts() and "measured_ranks" in memory.receipts()[0]
    out = tmp_path / "report.txt"
    report = GWProductionReport(str(out), runtime=NS(process_index=0), debug=False,
                                stdout=lambda *a, **k: None)
    report.timings(timing.records(), wall=1.0)
    report.warnings()
    report.close()
    text = out.read_text()
    assert "priced stages (receipts" in text and "memory spole.gram_reduction: priced" in text
