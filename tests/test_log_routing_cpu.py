"""Golden tests of LORRAX's output routing on ``logging`` (``common.log``).

The three handlers a driver session installs, fed fixed records:

* the main-log handler (rank 0) turns WARNING records into the pass-2 message
  blocks, prints progress and stage lines verbatim, and other block-tagged
  INFO lines as detail rows under the current stage;
* the rank stream handler writes ``%(message)s`` to the stream the process
  held when the session opened, leaves out on rank 0 what the main log
  mirrors there, and renders messages as the main log does on other ranks;
* no JSON sidecar is written: a ``record`` extra is a DEBUG record's fields.

Python warnings, ``LoopProgress`` and the stage cadence of ``common.timing``
reach them with byte-identical text (nested stage traces at DEBUG), and a
whole ``RunSession`` shows the progress bar and the stage banners in its
main log.
"""

from __future__ import annotations

import io
import logging
import re
import sys
import types
import warnings


from common import log
from common import report_blocks as blocks
from common.production_report import ProductionReport

_BAR = re.compile(r"^\[ \d\d:\d\d:\d\d \| [\u2588\u2591]{10} \| +\d+% \] ")
# ── extras ───────────────────────────────────────────────────────────────────
# ── main-log handler ─────────────────────────────────────────────────────────
# ── rank stream handler ──────────────────────────────────────────────────────
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
        on_detail=lambda text, fields: got.append(("detail", text)))
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
    assert warnings.formatwarning is saved_format
    assert log._FALLBACK in logging.getLogger("lorrax").handlers
    assert logging.getLogger("lorrax").propagate


def test_a_run_session_shows_progress_and_banners_in_its_main_log(tmp_path, capsys):
    """A non-GW driver's session: the bar and the stage banners are in the main
    log, a warning is printed and summarized, and no JSON sidecar is written."""
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
    assert not (tmp_path / "probe.json").exists()
    assert blocks.warning_count(run.report.messages()) == 1


# ── verbose: --verbose, the deck key, LORRAX_DEBUG_PRINT ─────────────────────