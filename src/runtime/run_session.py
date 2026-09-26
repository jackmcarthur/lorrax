"""One driver invocation's session: report, stdout, timing, refusal, files.

Every LORRAX driver ran the same bookkeeping around its physics.  It opened a
rank-zero report, installed :class:`ProductionStdout` with the report as its
warning sink, reset the timing collector, split the pre-``main`` span into the
startup call's own phases, and at the end printed the stage table, the file
table and the completion line and restored stdout.  :class:`RunSession` is
that sequence, written once::

    with RunSession(RUNTIME, "kin_ion", Report, report_path,
                    stages=STAGES, **labels) as run:
        ...physics, reporting through run.report...
        run.complete(files=[("mean-field matrices", "written", out_path)])

A launcher whose P tasks each joined a one-process world refuses before the
report opens (:func:`refuse_split_launch`).  A refusal raised inside the
``with`` block still closes the report: the stage
table printed so far, the retained warnings and the line ``LORRAX ...
REFUSED (<ExceptionType>: GATE <id>)``.  The exception then propagates to
``runtime.run_main_and_finalize`` unchanged.

The report class is the driver's; this module imports none (it is L3 and a
report knows about bands).  It needs a constructor ``(path, *, runtime,
debug, stdout, **labels)``, a settable ``stdout``, and the methods
``legacy_print``, ``timings``, ``warnings``, ``files`` and ``finish``.  A
driver whose report is a single text written at the end (``kmeans_cli``)
passes ``report_cls=None``; the session then keeps the retained warnings in
:attr:`RunSession.warnings` and emits through :meth:`RunSession.emit`.

``stages`` names the major-stage table once, for completion and refusal
alike.  ``None`` hands :meth:`timings` the raw timing records, for a report
that partitions them itself (GW, htransform).  A tuple of ``(label, name,
...)`` rows sums those timing sections, after one ``runtime + imports`` row
this module owns.  A zero-argument callable returns finished ``(label,
seconds)`` rows (``exciton_bands`` keeps its own two-column timers).
"""
from __future__ import annotations

import os
import re
import time

import common.timing as timing
from runtime.production_stream import ProductionStdout

__all__ = ["RunSession", "published_artifacts", "refuse_split_launch"]

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


def refuse_split_launch(runtime) -> None:
    """Refuse a launcher's P tasks that each joined a one-process world.

    Then there is no world to partition over: every task computes the whole
    result, believes it is rank 0, and overwrites the same output file.
    Compared before any input or report file is opened.
    """
    from runtime import _resolve_proc_count

    advertised = _resolve_proc_count()
    world = int(runtime.process_count)
    if advertised > 1 and world <= 1:
        raise SystemExit(
            f"the launcher advertises {advertised} tasks (SLURM_NTASKS / "
            f"JAX_PROCESS_COUNT) but jax.distributed joined a world of "
            f"{world}.  Every task would redo the whole calculation and "
            f"overwrite the same output file.  Fix the distributed launch "
            f"(JAX_COORDINATOR_ADDRESS must be reachable from every "
            f"task) or run `-n 1`.")


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


class RunSession:
    """Report, stdout, timing and refusal handling for one driver run."""

    def __init__(self, runtime, prefix: str, report_cls, report_path=None, *,
                 stages=None, warning_fn=None, **report_labels) -> None:
        self.runtime = runtime
        self.prefix = str(prefix)
        self._report_cls = report_cls
        self._report_path = report_path
        self._report_labels = report_labels
        self._stages = stages
        self._warning_fn = warning_fn
        self.report = None
        self.warnings: list[str] = []
        self.debug = False
        self._stdout = None
        self._t_main = None
        self._pre_main = None
        self._closed = False
        self._timed = False

    # ── entry ────────────────────────────────────────────────────────────
    def __enter__(self) -> "RunSession":
        from runtime import debug_print_enabled, rank0_print

        refuse_split_launch(self.runtime)
        self._t_main = time.perf_counter()
        self._pre_main = timing.process_elapsed_s()
        timing.reset()
        _record_pre_main(self.runtime, self.prefix, self._pre_main)
        self.debug = bool(debug_print_enabled())
        if self._report_cls is not None:
            self.report = self._report_cls(
                self._report_path, runtime=self.runtime, debug=self.debug,
                stdout=rank0_print, **self._report_labels)
        sink = self._warning_fn or (
            self.report.legacy_print if self.report is not None
            else self.warnings.append)
        self._stdout = ProductionStdout(
            debug=self.debug, rank=int(self.runtime.process_index),
            warning_fn=sink)
        self._stdout.install()
        if self.report is not None:
            self.report.stdout = (rank0_print if self.debug
                                  else self._stdout.emit)
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
        return ((RUNTIME_ROW, runtime_s),) + tuple(
            (label, timing.total(records, *names))
            for label, *names in self._stages)

    # ── output ───────────────────────────────────────────────────────────
    def emit(self, text: str = "", *, end: str = "\n") -> None:
        """Rank-zero scientific output on the launcher's stdout."""
        if self.debug:
            from runtime import rank0_print
            rank0_print(text, end=end)
        else:
            self._stdout.emit(text, end=end)

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
        report closes as REFUSED.  All ranks then leave together.
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
            self.report.finish()
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
        gate = re.search(r"GATE (\w+)", str(exc))
        self.report.finish(
            status=f"REFUSED ({type(exc).__name__}"
            + (f": GATE {gate.group(1)}" if gate else "") + ")")

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
            if self._stdout is not None:
                self._stdout.close()
        return False
