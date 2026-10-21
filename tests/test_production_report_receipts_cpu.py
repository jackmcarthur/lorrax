"""The end-of-run report on real receipts (C7 crashed there: production_report read a key the
receipts no longer carry, after the physics had finished)."""
from types import SimpleNamespace as NS

import jax.numpy as jnp

from common import timing
from lxkit import memory


def _records(caplog, key):
    return [r.lorrax["record"] for r in caplog.records
            if getattr(r, "lorrax", {}).get("record_key") == key]


def test_end_of_run_report_reads_receipts(tmp_path, caplog):
    from gw.production_report import GWProductionReport
    caplog.set_level("DEBUG", logger="lorrax")
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
    # The main log keeps the table, the memory summary and one pointer line ...
    assert "\nTIMING\n" in text and "MAJOR-STAGE" not in text
    assert "  per-stage memory, priced stages and notes on fused clocks: report.json " \
           "(timing, memory)" in text
    assert "priced stages (receipts" not in text and "stage walls:" not in text
    # ... and the receipts, the per-stage table and the clock notes are sidecar records.
    (priced,) = _records(caplog, "memory.priced")
    assert any(row["line"].startswith("memory spole.gram_reduction: priced")
               for row in priced["stages"])
    (stages,) = _records(caplog, "memory.stages")
    assert stages["rows"] and {"stage", "peak_max_gb", "host_gb", "set_by"} <= set(stages["rows"][0])
    (notes,) = _records(caplog, "timing.notes")
    assert any(note.startswith("stage walls: ") for note in notes["notes"])
