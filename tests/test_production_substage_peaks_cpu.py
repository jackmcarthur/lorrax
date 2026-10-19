"""The memory table's constructor sub-stage rows: each section's own device peak, so a leg
measures gram_reduction, passivity_held and the head sample terms against their prices."""
import numpy as np

from gw.production_report import _substage_peak_lines


def test_substage_rows_take_each_sections_own_peak_max_over_calls_and_ranks():
    paths = [("gw_jax.screening", "spole.constructor"),
             ("gw_jax.screening", "spole.constructor", "spole.gram_reduction"),
             ("gw_jax.screening", "spole.constructor", "spole.gram_reduction"),
             ("gw_jax.screening", "spole.constructor", "spole.passivity_held"),
             ("gw_jax.persist_w0", "head.sample_terms")]
    rows = [{"peak": 66.0e9}, {"peak_ranks": [40.1e9, 39.0e9]}, {"peak_ranks": [41.5e9, 38.0e9]},
            {"peak_ranks": [66.01e9, 65.0e9]}, {"peak": None}]
    lines = _substage_peak_lines(paths, rows)
    assert lines[0].startswith("  constructor sub-stages")
    assert lines[1] == ("    spole.gram_reduction: peak 41.50 / 39.00 GB; "
                        "gw_jax.screening > spole.constructor > spole.gram_reduction")
    assert lines[2].startswith("    spole.passivity_held: peak 66.01 / 65.00 GB")
    assert len(lines) == 3                        # head.sample_terms unmeasured: no row
    assert _substage_peak_lines(paths[:1], rows[:1]) == []
