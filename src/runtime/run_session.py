"""One driver invocation's session: report, stdout, logging, timing, refusal, files.

Every LORRAX driver ran the same bookkeeping around its physics.  It opened a
rank-zero report, installed :class:`ProductionStdout`, reset the timing
collector, split the pre-``main`` span into the startup call's own phases, and
at the end printed the stage table, the file table and the completion line and
restored stdout.  :class:`RunSession` is that sequence, written once::

    with RunSession(RUNTIME, "kin_ion", Report, report_path,
                    stages=STAGES, **labels) as run:
        ...physics, reporting through run.report...
        run.complete(files=[("mean-field matrices", "written", out_path)])

A launcher whose P tasks would each form a one-process world refuses at
startup (``runtime.init_jax_distributed``).  A refusal raised inside the
``with`` block still closes the report: the stage table printed so far, the
retained warnings, a line naming the exception type and its ``[<id>]``, and
the closing line ``LORRAX <driver> ERROR ...``.  The exception, with its
``GATE <id>`` text, then propagates to ``runtime.run_main_and_finalize``
unchanged.

The output streams are :class:`RunStreams`: the stdout swap and the two
``lorrax`` logging handlers (``common.log``), main log on rank 0 and rank
stream on every rank (silent above rank 0 but for its warnings and errors,
on its stderr), with Python warnings captured as records.  No JSON sidecar
is written.  The handlers read the verbose switch
(``runtime.debug_print_enabled``: ``--verbose``, the deck's
``verbose = true``, ``LORRAX_DEBUG_PRINT=1``) on every record; verbose puts
DEBUG records in the rank log and, under the current stage, in the main log.  A driver that opens its report itself (GW,
whose report path comes from its deck) opens a :class:`RunStreams` beside it;
every other driver gets one from its session.

The report class is the driver's; this module imports none (it is L3 and a
report knows about bands).  It needs a constructor ``(path, *, runtime,
debug, stdout, **labels)``, a settable ``stdout``, and the methods
``legacy_print``, ``log_detail``, ``log_message``, ``render_message``,
``timings``, ``warnings``, ``files`` and ``finish(*, status, wall, error)``,
and optionally ``log_debug`` (verbose DEBUG rows)
(``common.production_report.ProductionReport`` has them).  A driver whose
report is a single text written at the end (``kmeans_cli``) passes
``report_cls=None``; the session then keeps the retained warnings' texts
(``<Category>: <message>``) in :attr:`RunSession.warnings` and emits through
:meth:`RunSession.emit`.

``stages`` names the major-stage table once, for completion and refusal
alike.  ``None`` hands :meth:`timings` the raw timing records, for a report
that partitions them itself (GW, htransform).  A tuple of ``(label, name,
...)`` rows sums those timing sections, and carries this rank's device peak
over them, after one ``runtime + imports`` row this module owns.  A zero-argument callable returns finished ``(label,
seconds)`` rows (``exciton_bands`` keeps its own two-column timers).
"""
from __future__ import annotations

import os
import sys
import time

import common.timing as timing
from common import log
from runtime.production_stream import ProductionStdout

__all__ = ["RunSession", "RunStreams", "published_artifacts"]

#: The row this module prepends to a declared stage table.
RUNTIME_ROW = "runtime + imports"


def _record_pre_main(runtime, prefix: str, pre_main: float | None) -> None:
    """Record the pre-``main`` span as the startup call's phases + imports.

    The startup call measured its own phases (``runtime.facts['elapsed']``).
    Those become ``<prefix>.runtime_stack.<phase>`` rows and the remainder of
    the process age at ``main()`` becomes ``<prefix>.imports``, so the table
    closes against the process wall without counting any span twice.
    """
    if pre_main is None:
        return
    try:
        phases = dict(runtime.facts.get("elapsed", {}) or {})
    except Exception:          # noqa: BLE001 — observability never kills a run
        phases = {}
    for phase, seconds in sorted(phases.items()):
        if phase != "total":
            timing.record(f"{prefix}.runtime_stack.{phase}", float(seconds))
    timing.record(f"{prefix}.imports",
                  max(pre_main - float(phases.get("total", 0.0)), 0.0))


def published_artifacts(rows, *, print_fn) -> bool:
    """Print each artifact's commit state; ``False`` if one is unpublished.

    A run that prints its completion line while an artifact is missing or
    still inside a collective write transaction is not a usable run (Fe
    10p/10q/10r, 2026-09-19).  ``rows`` are ``(label, path, required)``; a
    present file carrying the commit receipt must read 1.
    """
    from file_io.commit_state import read_commit_state

    ok = True
    print_fn("  published artifacts:")
    for label, path, required in rows:
        if path is None:
            print_fn(f"    {label}: not requested")
            continue
        path = str(path)
        if not os.path.exists(path):
            print_fn(f"    {label}: {'MISSING' if required else 'absent'} ({path})")
            ok = ok and not required
            continue
        try:
            committed = read_commit_state(path)
            state = "present" if committed is None else f"committed={committed}"
            ok = ok and committed in (None, 1)
        except Exception as exc:              # unreadable is not published
            state = f"unreadable ({type(exc).__name__})"
            ok = False
        print_fn(f"    {label}: {state} ({path})")
    return ok


class RunStreams:
    """One driver run's stdout and its ``lorrax`` logging handlers.

    :meth:`open` takes the stdout the process holds (the stream the harness
    captures as this rank's log), installs :class:`ProductionStdout` and then
    the handlers of ``common.log``: rank 0's main log into ``report``, the
    rank stream (rank 0's stdout; a rank above 0 prints only its warnings
    and errors, on its stderr), and Python warnings captured as records.  Without a report, a warning's text
    goes to ``warning_fn``.  :meth:`close` removes them in reverse order.
    """

    def __init__(self, runtime, *, debug: bool, report=None, warning_fn=None,
                 verbose=None) -> None:
        self.rank = int(runtime.process_index)
        self.debug = bool(debug)
        self.report = report
        self._warning_fn = warning_fn
        # The handlers' switch, read per record: a deck parsed after the open
        # still turns verbose on.  ``debug`` (the stdout swap) is fixed at open.
        self._verbose = verbose
        self._stdout = None
        self._uninstall = None

    def open(self) -> "RunStreams":
        from runtime import debug_print_enabled, rank0_print

        console = sys.stdout
        verbose = self._verbose if self._verbose is not None else debug_print_enabled
        # Warnings are logging records now; the swap keeps only its stdout role.
        self._stdout = ProductionStdout(debug=self.debug, rank=self.rank)
        self._stdout.install()
        report = self.report
        if report is not None:
            report.stdout = rank0_print if self.debug else self._stdout.emit
            sink = dict(on_message=report.log_message, on_detail=report.log_detail,
                        on_debug=getattr(report, "log_debug", None),
                        render=report.render_message)
        else:
            warning_fn = self._warning_fn

            def on_message(text, fields, levelno):
                if warning_fn is not None:
                    warning_fn(text)

            sink = dict(on_message=on_message)
        self._uninstall = log.install_session(
            rank=self.rank, debug=verbose, stream=console, **sink)
        return self

    def emit(self, text: str = "", *, end: str = "\n") -> None:
        """Rank-zero scientific output on the launcher's stdout."""
        if self.debug:
            from runtime import rank0_print
            rank0_print(text, end=end)
        else:
            self._stdout.emit(text, end=end)

    def close(self) -> None:
        if self._uninstall is not None:
            self._uninstall()
            self._uninstall = None
        if self._stdout is not None:
            self._stdout.close()
            self._stdout = None


class RunSession:
    """Report, stdout, logging, timing and refusal handling for one driver run."""

    def __init__(self, runtime, prefix: str, report_cls, report_path=None, *,
                 stages=None, **report_labels) -> None:
        self.runtime = runtime
        self.prefix = str(prefix)
        self._report_cls = report_cls
        self._report_path = report_path
        self._report_labels = report_labels
        self._stages = stages
        self.report = None
        self.warnings: list[str] = []
        self.debug = False
        self._streams = None
        self._t_main = None
        self._pre_main = None
        self._closed = False
        self._timed = False

    # ── entry ────────────────────────────────────────────────────────────
    def __enter__(self) -> "RunSession":
        from runtime import debug_print_enabled, rank0_print

        self._t_main = time.perf_counter()
        self._pre_main = timing.process_elapsed_s()
        timing.reset()
        _record_pre_main(self.runtime, self.prefix, self._pre_main)
        self.debug = bool(debug_print_enabled())
        if self._report_cls is not None:
            self.report = self._report_cls(
                self._report_path, runtime=self.runtime, debug=self.debug,
                stdout=rank0_print, **self._report_labels)
        self._streams = RunStreams(
            self.runtime, debug=self.debug, report=self.report,
            warning_fn=self.warnings.append).open()
        return self

    # ── clocks ───────────────────────────────────────────────────────────
    @property
    def main_wall(self) -> float:
        """Seconds since the session opened."""
        return time.perf_counter() - self._t_main

    @property
    def wall(self) -> float:
        """The process wall: the session's span plus the pre-``main`` span."""
        return self.main_wall + (self._pre_main or 0.0)

    def stage_rows(self):
        """What :meth:`timings` is handed; see the module docstring."""
        records = timing.records()
        if self._stages is None:
            return records
        if callable(self._stages):
            return tuple(self._stages())
        runtime_s = sum(
            float(row["inclusive"]) for row in records
            if str(row["name"]).startswith(f"{self.prefix}.runtime_stack."))
        runtime_s += timing.total(records, f"{self.prefix}.imports")

        def peak(names):
            """This rank's device pool high-water over the named sections (bytes)."""
            marks = [float(row["peak"]) for row in records
                     if str(row["name"]) in names and row.get("peak") is not None]
            return max(marks) if marks else None

        return ((RUNTIME_ROW, runtime_s),) + tuple(
            (label, timing.total(records, *names), peak(set(names)))
            for label, *names in self._stages)

    # ── output ───────────────────────────────────────────────────────────
    def emit(self, text: str = "", *, end: str = "\n") -> None:
        """Rank-zero scientific output on the launcher's stdout."""
        self._streams.emit(text, end=end)

    @staticmethod
    def file_rows(rows):
        """``(label, verb, path)`` rows; a missing path reads ``absent``."""
        return [(label, verb if path and os.path.exists(str(path))
                 else "absent", str(path)) for label, verb, path in rows]

    def _debug_timing_table(self, wall: float) -> None:
        if self.debug and int(self.runtime.process_index) == 0:
            timing.report(print_fn=(self.report.legacy_print
                                    if self.report is not None else print),
                          title="--- Timing (seconds) ---", wall=wall)

    def complete(self, *, files=(), published=(), stages=None) -> None:
        """Stage table, warnings, file table, completion line.

        ``stages`` replaces the table declared at open, for a driver whose
        early-exit mode runs other stages.  ``published`` rows ``(label, path, required)`` are checked by
        :func:`published_artifacts` before the completion line; an
        unpublished artifact refuses (``GATE artifacts_unpublished``) and the
        report closes as ERROR.  All ranks then leave together.
        """
        if stages is not None:
            self._stages = stages
        wall = self.wall
        self._debug_timing_table(wall)
        if self.report is not None:
            self.report.timings(self.stage_rows(), wall=wall)
            self._timed = True
            self.report.warnings()
            self.report.files(self.file_rows(files))
        if published and not published_artifacts(
                published, print_fn=(self.report.legacy_print
                                     if self.report is not None else print)):
            raise RuntimeError(
                "GATE artifacts_unpublished: got: a required artifact is "
                "missing or left uncommitted (manifest above); want: every "
                "listed artifact present and committed; why: a run that "
                "prints the completion line while a published artifact is "
                "missing or inside its write transaction is not a usable run")
        if self.report is not None:
            self.report.finish(wall=wall)
        self._closed = True
        from common.collectives import barrier
        barrier(f"{self.prefix}.report_written")

    # ── exit ─────────────────────────────────────────────────────────────
    def _report_refusal(self, exc: BaseException) -> None:
        wall = self.wall
        self._debug_timing_table(wall)
        if self.report is None:
            return
        if not self._timed:
            self.report.timings(self.stage_rows(), wall=wall)
        self.report.warnings()
        self.report.finish(status="ERROR", wall=wall, error=exc)

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            refused = exc is not None and not (
                isinstance(exc, SystemExit) and exc.code in (None, 0))
            if refused and not self._closed:
                self._closed = True
                try:
                    self._report_refusal(exc)
                except Exception:  # noqa: BLE001 — never mask the refusal
                    pass
        finally:
            if self._streams is not None:
                self._streams.close()
        return False
