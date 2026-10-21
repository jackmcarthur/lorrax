"""Golden tests of the SC loop's main-log text (``gw.sc_report``) and of where
the SC driver's lines go (``gw.sc_iteration._record_sc`` / ``_report_sc``).

Pure functions from values to lines: no device work, no driver run.
"""
import logging

from common.report_blocks import stage_detail
from gw import sc_report as r


def _lines(rows):
    out = []
    for label, value in rows:
        out += stage_detail(label, value)
    return out


def _bsc_run():
    """The two maps of the hsuite bispinor QSGW run (bsc.out, baseline 67e94f0)."""
    run = r.start_run(mode="mpa", max_iter=2, tol_ev=3.0, accelerator="anderson", depth=20)
    m0 = run.map(0)
    m0.input, m0.rebuilt = r.input_label("dft_seed"), ("W", "Sigma", "V_H")
    m0.role, m0.max_abs_ev, m0.worst_k, m0.worst_band = "initial", 3.892961096, 6, 4
    m0.cutoff_ev, m0.rms_ev, m0.n_bands, m0.converged = 3.0, 1.48295, 5, False
    m0.gap_ev, m0.wall_s = 0.474034726, 88.1
    m0.stage_s = {"W response": 78.2, "Sigma rule refit": 0.0, "Sigma rule plan": 0.9,
                  "Sigma tau sweep": 3.9, "Sigma exchange": 2.0, "Sigma Hartree": 0.0}
    run.change(0, "grid", "Sigma grid grown to [-2.75, +7.25] eV for 25 states")
    m1 = run.map(1)
    m1.input, m1.rebuilt = r.input_label("accepted_input_map"), ("W", "Sigma", "V_H")
    m1.role, m1.max_abs_ev, m1.worst_k, m1.worst_band = "accepted_input_map", 1.914016183, 6, 4
    m1.cutoff_ev, m1.rms_ev, m1.n_bands, m1.converged = 3.0, 0.750533, 5, True
    m1.gap_ev, m1.wall_s = 0.818788572, 51.3
    m1.stage_s = {"W response": 47.1, "Sigma rule refit": 0.1, "Sigma tau sweep": 3.0}
    run.change(None, "rules", "Sigma rules: 3 of 10 windows rebuilt")
    run.change(1, "w_ceiling", "W pole ceiling re-planned")
    run.verdict = r.verdict_text(converged=True, stalled=False, one_map=False, n_maps=2,
                                 max_abs_ev=1.914016183, cutoff_ev=3.0)
    return run


def test_loop_and_map_begin_rows():
    run = _bsc_run()
    assert stage_detail("SC loop", r.loop_value(**run.loop)) == [
        "      SC loop        anderson mixing (history 20), at most 2 maps, stops at "
        "max|dE| < 3 eV"]
    assert stage_detail("SC map 0", r.map_begin_value(run.map(0))) == [
        "      SC map 0       input DFT states; rebuilds W, Sigma and V_H"]
    assert r.loop_value(mode="mpa", max_iter=1, tol_ev=0.002, accelerator="anderson",
                        depth=20) == "one map, no mixing; max|dE| reported against 0.002 eV"


def test_map_result_rows_golden():
    run = _bsc_run()
    assert _lines(r.map_result_rows(run.map(0))) == [
        "      max|dE|        3.892961 eV at k=6, band 4 (cutoff 3.000000 eV): not converged",
        "      RMS |dE|       1.482950 eV over 5 bands",
        "      gap            0.474035 eV",
        "      changes        Sigma grid grown to [-2.75, +7.25] eV for 25 states",
        "      wall           88.1 s (W response 78.2, Sigma 6.8, other 3.1)",
    ]
    assert _lines(r.map_result_rows(run.map(1)))[3:] == [
        "      changes        Sigma rules: 3 of 10 windows rebuilt; W pole ceiling re-planned",
        "      wall           51.3 s (W response 47.1, Sigma 3.1, other 1.1)",
    ]


def test_metal_and_one_shot_rows():
    facts = r.MapFacts(call=0, role="one_shot", max_abs_ev=0.5, worst_k=0, worst_band=6,
                       cutoff_ev=1.5, rms_ev=0.2, n_bands=8, converged=False, metal=True,
                       gap_ev=1.5e-8)
    rows = dict(r.map_result_rows(facts))
    assert rows["gap"] == "none (metal)"
    assert rows["max|dE|"].endswith(": one map, no convergence test")
    assert "wall" not in rows and "changes" not in rows


def test_convergence_table_golden():
    lines = _lines(_bsc_run().convergence_lines())
    assert lines == [
        "      SC convergence max|dE| over 5 non-scissored bands, cutoff 3.000000 eV",
        "        map  input           max|dE| eV  at k, band      RMS eV      gap eV   wall s",
        "          0  DFT states        3.892961        6, 4    1.482950    0.474035     88.1",
        "          1  Anderson          1.914016        6, 4    0.750533    0.818789     51.3",
        "      verdict        CONVERGED after 2 maps (max|dE| 1.914016 eV < 3.000000 eV)",
    ]
    assert max(map(len, lines)) <= 100
    # The harness reads "SC iteration: call=N ... max|dE|=" from the main log;
    # the table must not add a match.
    assert not any("SC iteration:" in line or "max|dE|=" in line for line in lines)


def test_verdict_texts():
    common = dict(n_maps=3, max_abs_ev=0.723399, cutoff_ev=1.5)
    assert r.verdict_text(converged=False, stalled=False, one_map=False, **common) == (
        "NOT CONVERGED after 3 maps (max|dE| 0.723399 eV, cutoff 1.500000 eV)")
    assert r.verdict_text(converged=False, stalled=True, one_map=False, **common) == (
        "NOT CONVERGED: stalled at a residual floor after 3 maps (max|dE| 0.723399 eV)")
    assert r.verdict_text(converged=False, stalled=False, one_map=True, **common) == (
        "one map, convergence not requested (max|dE| 0.723399 eV)")


def test_last_run_and_final():
    run = _bsc_run()
    assert r.last_run() is run
    assert run.final.call == 1 and run.converged is None
    assert r.convergence_lines(r.SCRun()) == []


def test_plain_reason_drops_gate():
    assert r.plain_reason("GATE pt_head_window_hybridized: sc_head_update=parallel_transport: "
                          "the window dips") == ("sc_head_update=parallel_transport: "
                                                 "the window dips")
    assert r.plain_reason("links incomplete") == "links incomplete"


def test_w_model_row():
    from gw.shared_pole_screening import _bank_samples, w_model_row
    assert w_model_row({"W": [4, 6, 6]}, 22, 7.84) == (
        "shared-pole, 4-6 poles per q on 3 q, from 22 frequency samples; 7.8 s")
    assert w_model_row({"CC": [10, 11], "TT": [31, 33]}, 22, 78.2) == (
        "shared-pole sectors, poles per q CC 10-11, TT 31-33, from 22 frequency samples; 78.2 s")
    assert w_model_row({"W": []}, None, 0.4, reused="committed") == (
        "shared-pole, reused, no bank built; 0.4 s")
    bank = {"batches": [dict(sample=0), dict(sample=1), dict(sample=1)]}
    assert _bank_samples(bank) == 2 and _bank_samples({"samples": bank}) == 2
    assert _bank_samples(None) is None


def test_rows_are_main_log_blocks(caplog):
    from common.log import MainLogHandler
    handler = MainLogHandler(lambda *a: None, lambda text, fields: None)
    with caplog.at_level(logging.DEBUG, logger="lorrax"):
        r.emit_row("SC map 0", "input DFT states")
    rec = [x for x in caplog.records if x.name == "lorrax.gw.sc_report"][-1]
    assert rec.levelno == logging.INFO and handler.accepts(rec)
    assert rec.lorrax == {"block": "sc", "label": "SC map 0"}


def test_sc_lines_go_to_the_rank_log_and_the_iteration_line_to_the_report(caplog):
    from types import SimpleNamespace
    from gw import sc_iteration
    printed = []
    inputs = SimpleNamespace(record_fn=printed.append, print_fn=printed.append)
    with caplog.at_level(logging.INFO, logger="lorrax"):
        sc_iteration._record_sc(inputs, "    SC semicore = dft: 16 coarse (k,label) hold")
        sc_iteration._report_sc(inputs, "    SC iteration: call=0000 role=initial max|dE|=1e+00 eV")
    # The REQUIRED_LINES receipt is INFO on the rank log, never the report.
    assert [x.getMessage() for x in caplog.records if x.name == "lorrax.gw.sc_iteration"] == [
        "    SC semicore = dft: 16 coarse (k,label) hold"]
    assert printed == ["    SC iteration: call=0000 role=initial max|dE|=1e+00 eV"]
