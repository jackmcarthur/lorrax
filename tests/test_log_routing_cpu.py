"""Golden tests of LORRAX's output routing on ``logging`` (``common.log``).

The three handlers a driver session installs, fed fixed records:

* the main-log handler (rank 0) turns WARNING records into the pass-2 message
  blocks, prints progress and stage lines verbatim, and other block-tagged
  INFO lines as detail rows under the current stage;
* the rank stream handler writes ``%(message)s`` to the stream the process
  held when the session opened, leaves out on rank 0 what the main log
  mirrors there, and renders messages as the main log does on other ranks;
* the JSON sidecar keeps each ``record`` extra under its ``record_key``, and
  writes no file without one.

Python warnings, ``LoopProgress`` and the stage cadence of ``common.timing``
reach them with byte-identical text (nested stage traces at DEBUG), and a
whole ``RunSession`` shows the progress bar and the stage banners in its
main log.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import types
import warnings

import pytest

from common import log
from common import report_blocks as blocks
from common.production_report import ProductionReport

_BAR = re.compile(r"^\[ \d\d:\d\d:\d\d \| [\u2588\u2591]{10} \| +\d+% \] ")
_STAGE = re.compile(r"^\[stage \d\d:\d\d:\d\d\] ")


def _record(name, level, text, fields=None):
    rec = logging.LogRecord(name, level, __file__, 1, text, None, None)
    if fields is not None:
        rec.lorrax = fields
    return rec


def _report(tmp_path, lines):
    runtime = types.SimpleNamespace(process_index=0, facts={})
    return ProductionReport(str(tmp_path / "t.out"), runtime=runtime, debug=False,
                            stdout=lines.append, driver="t")


@pytest.fixture(autouse=True)
def _no_debug(monkeypatch):
    import runtime
    monkeypatch.delenv("LORRAX_DEBUG_PRINT", raising=False)
    runtime.set_verbose(False)
    yield
    runtime.set_verbose(False)
    for uninstall in list(log._SESSION):
        uninstall()


# ── extras ───────────────────────────────────────────────────────────────────
def test_logger_names_and_extras():
    assert log.get_logger("gw.sigma").name == "lorrax.gw.sigma"
    assert log.message("x_id", found=("dipole.h5", "bands 1-24"), needs="bands 1-30",
                       fix="rerun", lines=("a",)) == {"lorrax": {
        "id": "x_id", "found": ("dipole.h5", "bands 1-24"), "needs": "bands 1-30",
        "fix": "rerun", "lines": ("a",)}}
    assert log.block("progress") == {"lorrax": {"block": "progress"}}
    assert log.block("isdf", label="zeta plan") == {"lorrax": {
        "block": "isdf", "label": "zeta plan"}}
    assert log.record(n=3) == {"lorrax": {"record": {"n": 3}}}
    assert log.record("sigma.quadrature", n=3) == {"lorrax": {
        "record_key": "sigma.quadrature", "record": {"n": 3}}}
    assert log.fields_of(_record("lorrax.x", logging.INFO, "t")) == {}


# ── main-log handler ─────────────────────────────────────────────────────────
def test_main_log_prints_a_tagged_warning_as_its_message_block(tmp_path):
    lines = []
    report = _report(tmp_path, lines)
    handler = log.MainLogHandler(report.log_message, report.progress)
    handler.handle(_record("lorrax.centroid", logging.WARNING,
                           "centroids not placed for the zeta fit",
                           log.message("centroid_window_short")["lorrax"]
                           | {"found": ("c.txt", "selected on left 1-3"),
                              "needs": "left window up to band 5",
                              "fix": "python -m centroid.kmeans_cli 6"}))
    head = "WARNING  centroids not placed for the zeta fit"
    tag = "[centroid_window_short]"
    assert lines == [
        head + " " * (100 - len(head) - len(tag)) + tag,
        "         c.txt: selected on left 1-3    this run needs: left window up to band 5",
        "         fix: python -m centroid.kmeans_cli 6",
    ]
    assert [(m.kind, m.id) for m in report.messages()] == [
        ("WARNING", "centroid_window_short")]


def test_main_log_maps_an_untagged_warning_through_the_id_table(tmp_path):
    lines = []
    report = _report(tmp_path, lines)
    handler = log.MainLogHandler(report.log_message, report.progress)
    handler.handle(_record("py.warnings", logging.WARNING,
                           "RuntimeWarning: GATE some_gate: the thing is off"))
    handler.handle(_record("py.warnings", logging.ERROR, "UserWarning: plain words"))
    assert [(m.kind, m.id, m.text) for m in report.messages()] == [
        ("WARNING", "some_gate", "the thing is off"),
        ("ERROR", "unclassified", "plain words")]
    assert lines[0].startswith("WARNING  the thing is off") and lines[0].endswith("[some_gate]")
    assert not any("GATE" in line for line in lines)


def test_main_log_takes_block_info_and_leaves_the_rest(tmp_path):
    lines = []
    report = _report(tmp_path, lines)
    handler = log.MainLogHandler(report.log_message, report.log_detail)
    banner = "[stage 12:10:58] -> screening (chi0 -> W)"
    bar = "[ 14:44:32 | █████░░░░░ |  50% ] W role 1 / 2 · ETA 1 s"
    for rec in (_record("lorrax.stage", logging.INFO, banner, {"block": "stage"}),
                _record("lorrax.progress", logging.INFO, bar, {"block": "progress"}),
                _record("lorrax.isdf", logging.INFO,
                        "route G, 1 batch of 6 centroids, Z store on the host",
                        log.block("isdf", label="zeta plan")["lorrax"]),
                _record("lorrax.isdf", logging.INFO, "=> zeta and V(q) on 6 centroids",
                        {"block": "isdf"}),
                _record("lorrax.x", logging.INFO, "plain info"),
                _record("lorrax.x", logging.DEBUG, "tagged debug", {"block": "x"})):
        handler.handle(rec)
    assert lines == [
        banner,
        bar,
        "      zeta plan      route G, 1 batch of 6 centroids, Z store on the host",
        "      => zeta and V(q) on 6 centroids",
    ]
    assert report._stage == "screening"
    assert not log.MainLogHandler(report.log_message).accepts(
        _record("lorrax.stage", logging.INFO, banner, {"block": "stage"}))


# ── rank stream handler ──────────────────────────────────────────────────────
def test_rank_stream_on_rank_zero_leaves_out_what_the_main_log_mirrors():
    out = io.StringIO()
    main = log.MainLogHandler(lambda *a: None, lambda *a: None)
    handler = log.RankStreamHandler(out, debug=False, exclude=main.accepts)
    for rec in (_record("lorrax.stage", logging.INFO, "[stage 00:00:00] -> a",
                        {"block": "stage"}),
                _record("lorrax.x", logging.WARNING, "w"),
                _record("lorrax.x", logging.INFO, "info line"),
                _record("lorrax.x", logging.DEBUG, "debug line")):
        if rec.levelno >= handler.level:
            handler.handle(rec)
    assert out.getvalue() == "info line\n"


def test_rank_stream_on_another_rank_renders_messages_as_the_main_log(tmp_path):
    out = io.StringIO()
    report = _report(tmp_path, [])
    handler = log.RankStreamHandler(out, debug=True, render=report.render_message)
    handler.handle(_record("py.warnings", logging.WARNING,
                           "RuntimeWarning: GATE lr_x: got 3"))
    handler.handle(_record("lorrax.x", logging.DEBUG, "debug line"))
    text = out.getvalue().splitlines()
    assert text[0].startswith("WARNING  got 3") and text[0].endswith("[lr_x]")
    assert text[-1] == "debug line"
    assert report.messages() == []           # rendered, not counted


# ── JSON sidecar ─────────────────────────────────────────────────────────────
def test_sidecar_keeps_each_record_under_its_key(tmp_path):
    import numpy as np
    path = tmp_path / "t.json"
    handler = log.JsonSidecarHandler(str(path), report="t.out")
    handler.handle(_record("lorrax.x", logging.WARNING, "no record"))
    assert not path.exists()
    for rec in (
            _record("lorrax.gw", logging.DEBUG, "Sigma quadrature receipt",
                    log.record("sigma.quadrature", windows=11, eta_ev=0.25)["lorrax"]),
            _record("lorrax.mem", logging.DEBUG, "priced",
                    log.record("memory.receipts", stage="a", gb=np.float64(0.5))["lorrax"]),
            _record("lorrax.mem", logging.DEBUG, "priced",
                    log.record("memory.receipts", stage="b", gb=1.0)["lorrax"]),
            _record("lorrax.zeta", logging.DEBUG, "fit receipt",
                    log.record(rank=np.int64(7), shape=np.arange(3))["lorrax"])):
        handler.handle(rec)
        json.loads(path.read_text())                # a whole document after each record
    handler.close()
    doc = json.loads(path.read_text())
    assert doc["lorrax_sidecar"] == 1 and doc["report"] == "t.out" and "started" in doc
    assert doc["sigma"] == {"quadrature": {"windows": 11, "eta_ev": 0.25}}
    assert doc["memory"] == {"receipts": [{"stage": "a", "gb": 0.5},
                                          {"stage": "b", "gb": 1.0}]}
    (row,) = doc["records"]
    assert {k: row[k] for k in ("logger", "level", "message", "record")} == {
        "logger": "lorrax.zeta", "level": "DEBUG", "message": "fit receipt",
        "record": {"rank": 7, "shape": [0, 1, 2]}}
    assert [p.name for p in tmp_path.iterdir()] == ["t.json"]


# ── a session: warnings, progress, stage cadence ─────────────────────────────
def test_session_routes_warnings_progress_and_stage_lines(tmp_path):
    from common import timing
    from common.progress import LoopProgress
    console = io.StringIO()
    got = []
    saved_format, saved_stdout = warnings.formatwarning, sys.stdout
    uninstall = log.install_session(
        rank=0, debug=False, stream=console,
        on_message=lambda text, fields, level: got.append(("msg", text, level)),
        on_detail=lambda text, fields: got.append(("detail", text)),
        sidecar=str(tmp_path / "t.json"))
    try:
        sys.stdout = io.StringIO()          # the swap: the rank stream keeps its stream
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("GATE g: words", RuntimeWarning)
            warnings.warn("Some donated buffers were not usable: x", UserWarning)
        with LoopProgress(2, None, title="probe", item_name="step", enabled=True) as bar:
            bar.step()
            bar.step()
        timing.reset()
        with timing.section("t.outer", announce=True, label="outer"):
            with timing.section("t.inner", announce=True, label="inner substep"):
                pass
            try:
                with timing.section("t.bad", announce=True, label="bad substep"):
                    raise ValueError("GATE probe_gate: refused")
            except ValueError:
                pass
        log.get_logger("x").info("rank-log line")
    finally:
        sys.stdout = saved_stdout
        uninstall()
    assert got[0] == ("msg", "RuntimeWarning: GATE g: words", logging.WARNING)
    details = [text for kind, text, *_ in got if kind == "detail"]
    assert re.match(r"^Started probe at \d\d:\d\d:\d\d\.$", details[0])
    assert all(_BAR.match(line) for line in details[1:3])
    assert details[2].endswith("step 2 / 2 \u00b7 ETA 0 s")
    assert re.match(r"^Finished probe at \d\d:\d\d:\d\d\.  Elapsed: \d+ s\.$", details[3])
    # Plain top-level and nested traces are DEBUG; an exit naming an exception is not.
    assert re.search(r"\]   <- bad substep  \d+\.\d s  \[EXC\] ValueError: GATE probe_gate: "
                     r"refused$", details[4])
    assert not any("inner substep" in text or "outer" in text for text in details)
    assert len(got) == 6                    # the donation hint never reaches the log
    assert console.getvalue() == "rank-log line\n"
    assert not (tmp_path / "t.json").exists()
    assert warnings.formatwarning is saved_format
    assert log._FALLBACK in logging.getLogger("lorrax").handlers
    assert logging.getLogger("lorrax").propagate


def test_outside_a_session_progress_prints_to_stdout_and_propagates(capsys, caplog):
    from common.progress import LoopProgress
    caplog.set_level(logging.INFO, logger="lorrax.progress")
    with LoopProgress(1, None, title="probe", enabled=True) as bar:
        bar.step()
    out = capsys.readouterr().out.splitlines()
    assert [r.getMessage() for r in caplog.records] == out
    assert out[0].startswith("Started probe at ") and _BAR.match(out[1])
    assert all(log.fields_of(r) == {"block": "progress"} for r in caplog.records)


@pytest.mark.parametrize("rank, shown", [(0, ["status", "careful"]),
                                          (1, ["careful"])])
def test_outside_a_session_only_process_zero_prints_status(monkeypatch, capsys,
                                                            rank, shown):
    import runtime
    for name in runtime._WORLD_RANK_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_PROCID", str(rank))
    probe = log.get_logger("lorrax.probe")
    probe.debug("forensic")
    probe.info("status")
    probe.warning("careful")
    assert capsys.readouterr().out.splitlines() == shown


def test_band_window_snap_warns_once_with_its_id(caplog):
    import numpy as np
    from common.band_degeneracy import check_band_window, resolve_band_window
    caplog.set_level(logging.WARNING, logger="lorrax")
    e = np.array([[0.0, 0.5, 0.5, 1.0, 1.5, 2.0]])     # bands 1,2 degenerate
    assert resolve_band_window(e, 3, 1, 1, mode="snap", where="probe") == (2, 1)
    check_band_window(e, 2, 4, mode="snap", where="probe")
    assert [log.fields_of(r)["id"] for r in caplog.records] == [
        "band_window_snapped", "band_window_cuts_multiplet"]


def test_a_run_session_shows_progress_and_banners_in_its_main_log(tmp_path, capsys):
    """A non-GW driver's session: the bar and the stage banners are in the main
    log, a warning is printed and summarized, a receipt is in the sidecar."""
    from common import timing
    from common.progress import LoopProgress
    from runtime.run_session import RunSession
    runtime = types.SimpleNamespace(process_index=0, facts={})
    out = tmp_path / "probe.out"
    with RunSession(runtime, "probe", ProductionReport, str(out), stages=(),
                    driver="probe") as run:
        with timing.section("probe.solve", announce=True, label="solve"):
            with LoopProgress(1, run.report.legacy_print, title="solve",
                              enabled=True) as bar:
                bar.step()
        lg = log.get_logger("probe")
        lg.warning("the probe is short", extra=log.message(
            "probe_short", needs="a longer probe", fix="probe --longer"))
        lg.debug("receipt", extra=log.record("probe.receipt", n=1))
        print("incidental chatter")         # still swapped away in production
        run.complete()
    text = out.read_text().splitlines()
    # A plain top-level section's traces are DEBUG: the bar block alone.
    assert text[0].startswith("Started solve at ") and _BAR.match(text[1])
    assert text[2].startswith("Finished solve at ")
    assert text[3].startswith("WARNING  the probe is short") and text[3].endswith("[probe_short]")
    assert text[4:6] == ["         this run needs: a longer probe",
                         "         fix: probe --longer"]
    assert any(line.startswith("  WARNING [probe_short]") for line in text)
    assert text[-1].startswith("LORRAX probe COMPLETED ") and text[-1].endswith("1 warning")
    stdout = capsys.readouterr().out
    assert stdout.splitlines() == text      # the main log, mirrored once
    assert "incidental chatter" not in stdout
    sidecar = json.loads((tmp_path / "probe.json").read_text())
    assert sidecar["report"] == "probe.out" and sidecar["probe"] == {"receipt": {"n": 1}}
    assert blocks.warning_count(run.report.messages()) == 1


# ── verbose: --verbose, the deck key, LORRAX_DEBUG_PRINT ─────────────────────
def test_main_log_takes_debug_only_when_verbose(tmp_path):
    lines = []
    report = _report(tmp_path, lines)
    switch = {"on": False}
    handler = log.MainLogHandler(report.log_message, report.log_detail,
                                 on_debug=report.log_debug, verbose=lambda: switch["on"])
    report.progress("[stage 12:10:58] -> screening (chi0 -> W)")
    nested = _record("lorrax.stage", logging.DEBUG, "[stage 12:10:59]   -> W compile",
                     {"block": "stage"})
    receipt = _record("lorrax.gw", logging.DEBUG, "Sigma quadrature receipt\n  11 windows",
                      log.record("sigma.quadrature", windows=11, eta_ev=0.25)["lorrax"])
    handler.handle(nested)
    handler.handle(receipt)
    assert lines == ["[stage 12:10:58] -> screening (chi0 -> W)"]       # off: unchanged
    switch["on"] = True                     # read per record, as a deck parsed late is
    handler.handle(nested)
    handler.handle(receipt)
    assert lines[1:] == ["      [stage 12:10:59]   -> W compile",
                         "      Sigma quadrature receipt",
                         "        11 windows"]
    assert not any("eta_ev" in line for line in lines)                  # payload: sidecar only
    assert report._stage == "screening"     # a DEBUG stage trace names no stage
    assert not log.MainLogHandler(report.log_message, verbose=True).accepts(nested)


def test_rank_stream_shows_debug_while_the_switch_is_on():
    out = io.StringIO()
    switch = {"on": False}
    handler = log.RankStreamHandler(out, debug=lambda: switch["on"])
    handler.handle(_record("lorrax.x", logging.DEBUG, "hidden"))
    switch["on"] = True
    handler.handle(_record("lorrax.x", logging.DEBUG, "shown"))
    handler.handle(_record("lorrax.x", logging.INFO, "info"))
    assert out.getvalue() == "shown\ninfo\n"


def _session_run(tmp_path, *, verbose_late=False):
    """A RunSession whose run logs DEBUG, a receipt and legacy text."""
    import runtime
    from runtime.run_session import RunSession
    rt = types.SimpleNamespace(process_index=0, facts={})
    out = tmp_path / "probe.out"
    with RunSession(rt, "probe", ProductionReport, str(out), stages=(),
                    driver="probe") as run:
        if verbose_late:                    # after the session opened: a deck read late
            runtime.set_verbose(source="verbose = true in probe.in")
        run.report.begin()
        lg = log.get_logger("probe")
        lg.debug("plan line one\nplan line two")
        lg.debug("receipt", extra=log.record("probe.receipt", n=1))
        run.report.legacy_print("  Z store: host, 2 tiles")       # a retired keep prefix
        run.report.legacy_print("WARNING: kept as a message")
        lg.info("rank-log line")
        run.complete()
    return out.read_text().splitlines()


def test_session_without_verbose_keeps_debug_out_of_the_main_log(tmp_path, capsys,
                                                                 monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = _session_run(tmp_path)
    body = text[text.index(""):]            # past the header, which echoes (and
                                            # wraps) pytest's own argv
    assert not any(word in line for line in body
                   for word in ("plan line", "receipt", "Z store", "Verbose"))
    assert any(line.startswith("WARNING  kept as a message") for line in text)
    logs = next(line for line in text if line.startswith("Logs "))
    assert logs == "Logs             probe.out · probe.json"
    files = text.index("OUTPUT FILES AND INPUTS")
    assert any(re.fullmatch(r"  machine-readable record\s+written\s+probe\.json", line)
               for line in text[files:])
    stdout = capsys.readouterr().out
    assert "plan line" not in stdout and "rank-log line" in stdout


def test_session_with_verbose_prints_debug_under_the_stage(tmp_path, capsys):
    text = _session_run(tmp_path, verbose_late=True)
    assert ("Verbose          on (verbose = true in probe.in); DEBUG lines print under "
            "each stage") in text
    at = text.index("      plan line one")
    assert text[at:at + 4] == ["      plan line one", "      plan line two",
                               "      receipt", "        Z store: host, 2 tiles"]
    assert not any('"n": 1' in line for line in text)
    stdout = capsys.readouterr().out.splitlines()
    assert stdout.count("      plan line one") == 1          # mirrored once, not twice
    assert "plan line one" not in stdout
    sidecar = json.loads((tmp_path / "probe.json").read_text())
    assert sidecar["probe"] == {"receipt": {"n": 1}}


def test_verbose_switch_from_argv_deck_and_environment(tmp_path, monkeypatch):
    import runtime
    assert not runtime.debug_print_enabled() and runtime.verbose_source() is None
    argv = ["-i", "x.in", "--verbose", "--other"]
    runtime.read_verbose_switches(argv)
    assert argv == ["-i", "x.in", "--other"]                 # no parser sees the flag
    assert runtime.verbose_source() == "--verbose" and runtime.verbose_enabled()
    runtime.set_verbose(False)
    deck = tmp_path / "gw.in"
    deck.write_text("[cohsex]\n# verbose = false\nverbose = TRUE   # forensic\nnval = 3\n")
    runtime.read_verbose_switches(["--input", str(deck)])
    assert runtime.verbose_source() == "verbose = true in gw.in"
    runtime.set_verbose(False)
    deck.write_text("[cohsex]\nverbose = false\n")
    runtime.read_verbose_switches([f"--input={deck}"])
    runtime.read_verbose_switches(["-i", str(tmp_path / "missing.in")])
    assert not runtime.debug_print_enabled()
    monkeypatch.setenv("LORRAX_DEBUG_PRINT", "1")            # the alias
    assert runtime.verbose_source() == "LORRAX_DEBUG_PRINT=1"


def test_verbose_switch_leaves_a_pytest_command_line_alone(monkeypatch):
    import runtime
    monkeypatch.setattr(sys, "argv", ["/venv/bin/pytest", "--verbose", "-q"])
    monkeypatch.setattr(runtime, "_VERBOSE_ARGV_READ", [None])
    runtime.read_verbose_switches()
    assert sys.argv == ["/venv/bin/pytest", "--verbose", "-q"]
    assert not runtime.debug_print_enabled()
    monkeypatch.setattr(sys, "argv", ["gw.gw_jax", "-i", "deck.in", "--verbose"])
    runtime.read_verbose_switches()
    assert sys.argv == ["gw.gw_jax", "-i", "deck.in"]
    assert runtime.verbose_source() == "--verbose"


def test_refuse_bad_argv_takes_verbose_out_before_the_parser():
    import argparse
    from runtime.cli_seam import refuse_bad_argv
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    argv = ["-i", "deck.in", "--verbose"]
    refuse_bad_argv(parser, argv)                           # no SystemExit(2)
    assert argv == ["-i", "deck.in"]


def test_checkout_version_reads_the_probe(monkeypatch):
    import os
    import shutil
    from runtime import source_closure as sc
    monkeypatch.setattr(sc, "_VERSION_PROBE", {"started": False, "proc": None,
                                               "version": None})
    assert sc.checkout_version() is None                    # no probe: no Version row
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if shutil.which("git") is None or not os.path.exists(os.path.join(root, ".git")):
        pytest.skip("no git checkout here")
    sc.start_version_probe(sc.SourceClosureReceipt("source", root, "", ()))
    version = sc.checkout_version(timeout=30.0)
    assert re.fullmatch(r"LORRAX [0-9a-f]{9}( \+ local changes)?", version or ""), version
    assert sc.checkout_version() == version                 # resolved once
