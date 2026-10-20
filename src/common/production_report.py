"""The rank-zero main log every LORRAX driver report shares.

:class:`ProductionReport` owns the stream (process 0 writes the report file and
the live stdout), the retained warnings, and the blocks every driver prints the
same way: header, Devices, symmetry and sampling, the stage table, warnings,
the files table and the closing line.  A driver's report subclasses it only for
its own blocks (``gw.production_report``, ``bandstructure.production_report``,
``common.preprocessing_output``).  The text of each block is a pure function in
:mod:`common.report_blocks`.
"""

from __future__ import annotations

import dataclasses
import os
import re
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from common import report_blocks as blocks
from common.report_blocks import Message
from common.scientific_output import (
    abs_path,
    htransform_quality_lines,
    spectral_compression_lines,
)


def _command(argv: list[str]) -> tuple[str, ...]:
    """The command as typed: ``python -m <module> <args>``.

    A harness that runs ``main`` in process sets ``sys.argv[0]`` to the module
    name; ``python -m`` sets it to the module's file, and ``sys.orig_argv``
    keeps the interpreter flags and the module name.
    """
    head = argv[0] if argv else ""
    if head and os.sep not in head and not head.endswith(".py"):
        return ("python", "-m", *argv)
    orig = list(getattr(sys, "orig_argv", ()) or ())
    if "-m" in orig:
        return ("python", *orig[1:])
    return ("python", *argv)


def _job() -> str | None:
    """``Slurm job <id>[, step <s>]`` from the launcher, or ``None``."""
    job = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job:
        return None
    step = os.environ.get("SLURM_STEP_ID", "").strip()
    return f"Slurm job {job}" + (f", step {step}" if step not in ("", "0") else "")


def _node_count() -> int | None:
    """The launcher's node count, or ``None`` when no launcher states it."""
    for name in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        value = os.environ.get(name, "").strip()
        if value.isdigit():
            return int(value)
    return None


#: The log is ASCII plus ``·`` and ``†``: the spellings legacy texts use instead.
_ASCII = {"ζ": "zeta", "ψ": "psi", "μ": "mu", "Σ": "Sigma", "χ": "chi", "ω": "omega",
          "τ": "tau", "—": "-", "–": "-", "≥": ">=", "≤": "<=", "→": "->", "×": "x"}


def _ascii(text: str) -> str:
    return "".join(_ASCII.get(ch, ch) for ch in text)


def _bands(text: str) -> str:
    """Code windows ``left=(0, 3)`` as 1-based band ranges ``left 1-3``."""
    return re.sub(r"(\w+)=\((\d+), (\d+)\)",
                  lambda m: f"{m.group(1)} {int(m.group(2)) + 1}-{m.group(3)}", text)


def _legacy_message(text: str) -> tuple[Message, bool]:
    """A component's warning text (a ``warnings.warn`` or a printed line) as a
    message, and whether it is quiet: a fact another block prints in full.

    Known texts get their id and a plain first line; any other text is
    ``WARNING [unclassified]`` with the text itself.  ``GATE <id>`` in a text
    gives the id and is dropped from it.
    """
    body = _ascii(" ".join(str(text).split()))
    body = re.sub(r"^\w*Warning:\s*", "", body)
    body = re.sub(r"^(?:\[WARN\]|WARNING)[:\s]+", "", body, flags=re.IGNORECASE)
    if ("GLOBAL TIME-REVERSAL SYMMETRY IS BROKEN" in body
            or "two-component TRS check FAILED" in body):
        return Message("NOTE", "time_reversal_broken",
                       "time reversal is broken in the DFT reference",
                       lines=(body,)), True
    match = re.match(r"minimax: SOLVED .*?target (\S+) .*?-> (\d+) nodes, max_err (\S+?),", body)
    if match:
        return Message(
            "NOTE", "minimax_runtime_solve",
            f"{match.group(2)}-node chi0 time rule solved in this run",
            lines=(f"max error {match.group(3)}, target {match.group(1)}; node positions "
                   "can differ in the last digits between hosts (this machine's LAPACK)",)), False
    match = re.match(r"minimax: served .*?-> (\d+) nodes, max_err (\S+?),", body)
    if match and "UNCERTIFIED" in body:
        return Message(
            "NOTE", "minimax_catalog_uncertified",
            f"{match.group(1)}-node chi0 time rule served from the catalog, not certified",
            lines=(f"max error {match.group(2)}",)), False
    match = re.match(r"centroid file (\S+?): selection window (.*?) ends below the zeta left "
                     r"leg's top (\d+)\. The zeta fit still trains on (.*?); for best "
                     r"placement regenerate the centroids with (.*?)\.?$", body)
    if match:
        return Message(
            "WARNING", "centroid_window_short", "centroids not placed for the zeta fit",
            found=(os.path.basename(match.group(1)),
                   f"selected on {_bands(match.group(2))}"),
            needs=f"left window up to band {match.group(3)}",
            lines=(f"the zeta fit still trains on {_bands(match.group(4))}",),
            fix=f"python -m {match.group(5)}"), False
    match = re.match(r"Response rule rebuilt at a held map; failed reuse test: (.*)$", body)
    if match:
        return Message("NOTE", "response_rule_rebuilt",
                       "screening time rule rebuilt at a held map",
                       lines=(f"reuse test failed: {match.group(1)}",)), False
    match = re.match(r"Sigma reads past the W line support: (.*)$", body)
    if match:
        return Message("WARNING", "sigma_reads_past_w_support",
                       "Sigma reads W past its line support (model extrapolation)",
                       lines=(match.group(1),)), False
    match = re.match(r"only (\d+) conduction guard band\(s\) above the BSE "
                     r"selection\s*-?\s*(.*)$", body)
    if match:
        return Message("WARNING", "bse_guard_bands_few",
                       f"only {match.group(1)} conduction guard band(s) above the BSE "
                       "selection", lines=(match.group(2),)), False
    match = re.match(r"non-orbit-closed centroid set\(s\): (.*)$", body)
    if match:
        return Message("WARNING", "centroids_not_orbit_closed",
                       "the centroid set is not closed under the symmetry operations",
                       lines=(match.group(1),)), False
    gate = re.search(r"GATE (\w+):?\s*", body)
    if gate:
        rest = (body[:gate.start()] + body[gate.end():]).strip(" ;:()")
        return Message("WARNING", gate.group(1), rest or gate.group(1)), False
    return Message("WARNING", "unclassified", body), False


class ProductionReport:
    """One driver's main log, written only by process zero."""

    #: Short driver name of the title and the closing line ("gw", "bse", ...).
    driver = "lorrax"
    #: What the driver computes, for the title ("GW quasiparticles").
    purpose = ""
    #: Words that make an unclassified component line a retained warning.
    _WARNING_WORDS = (
        "WARNING", "[WARN]", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
        "NOT TRUSTWORTHY", "DEPRECATED",
    )
    #: Component lines the log keeps verbatim (receipts that are the run's record).
    _KEEP_PREFIXES: tuple[str, ...] = ()

    def __init__(self, path: str, *, runtime, debug: bool, stdout,
                 driver: str | None = None, purpose: str | None = None) -> None:
        self.path = abs_path(path)
        self.runtime = runtime
        self.debug = bool(debug)
        self.stdout = stdout
        if driver is not None:
            self.driver = str(driver)
        if purpose is not None:
            self.purpose = str(purpose)
        self.directory = os.getcwd()
        self.rank = int(getattr(runtime, "process_index", 0))
        # Every message, once per occurrence, and whether it was quiet (its fact
        # printed in full by another block).
        self._messages: list[tuple[Message, bool]] = []
        self._stage = "setup"
        self._warnings_emitted = False
        self._stream = None
        if self.rank == 0:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.path, "w", encoding="utf-8", buffering=1)

    # ── stream ───────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def emit(self, line: str = "") -> None:
        """Write one rank-zero line to both the report and live stdout."""
        if self.rank != 0:
            return
        text = str(line)
        self.stdout(text)
        if self._stream is not None:
            self._stream.write(text + "\n")

    def emit_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.emit(line)

    #: A top-level stage trace line of ``common.timing``: ``[stage hh:mm:ss] -> label``.
    _STAGE_ENTER = re.compile(r"^\[stage \d\d:\d\d:\d\d\] -> (.+)$")

    def progress(self, line: str) -> None:
        """Write one deliberately selected rank-zero progress line."""
        entered = self._STAGE_ENTER.match(str(line))
        if entered:
            # The summary's stage column: the stage's name without its detail.
            label = entered.group(1).split(" (")[0].split(" + ")[0].strip()
            self._stage = label[:20].rstrip()
        self.emit(line)

    def legacy_print(self, *args, sep: str = " ", **kwargs) -> None:
        """Sink component chatter unless driver debug is on.

        Receipts named by :attr:`_KEEP_PREFIXES` and the loop cadence of
        ``LoopProgress`` print; a line with a warning word becomes a message
        (:func:`_legacy_message`), printed here and counted in the WARNINGS
        summary; the rest reaches stdout only under debug, where a warning is
        printed raw and still counted.
        """
        text = sep.join(str(value) for value in args)
        if (text.startswith(self._KEEP_PREFIXES)
                or text.startswith("Started ") or text.startswith("Finished ")
                or (text.startswith("[ ") and " | " in text and " / " in text)):
            self.progress(text)
            return
        warned = any(word in text.upper() for word in self._WARNING_WORDS)
        if self.debug:
            self.stdout(text, **kwargs)
            if warned:
                self.warn(_legacy_message(text)[0], quiet=True)
            return
        if warned:
            message, quiet = _legacy_message(text)
            self.warn(message, quiet=quiet)

    def warn(self, message: Message, *, indent: int = 0, quiet: bool = False) -> None:
        """Print a message where it happens and count it for the WARNINGS summary.

        The only way a message reaches the summary.  ``quiet`` counts it without
        printing: its fact is printed in full by another block (the Devices
        memory row, the symmetry block's time-reversal NOTE).  A message without
        its own stage takes the stage the run is in.
        """
        message = self.count(message, quiet=quiet)
        if not quiet:
            self.emit_lines(blocks.message_lines(message, indent=indent))

    def count(self, message: Message, *, quiet: bool = False) -> Message:
        """Count a message a block has printed itself (Devices, SYMMETRY)."""
        if message.stage == "setup" and self._stage != "setup":
            message = dataclasses.replace(message, stage=self._stage)
        self._messages.append((message, bool(quiet)))
        return message

    def messages(self) -> list[Message]:
        """One entry per occurrence; a quiet one only when nothing printed its id."""
        printed = {(m.kind, m.id) for m, quiet in self._messages if not quiet}
        seen_quiet = set()
        kept = []
        for message, quiet in self._messages:
            key = (message.kind, message.id)
            if quiet:
                if key in printed or key in seen_quiet:
                    continue
                seen_quiet.add(key)
            kept.append(message)
        return kept

    def heading(self, title: str) -> None:
        self.emit_lines(blocks.heading(title))

    # ── shared blocks ────────────────────────────────────────────────────
    def begin(self, *, input_file: str | None = None) -> None:
        """The title and the run's facts (``input_file`` is in the command)."""
        self.emit_lines(blocks.header_lines(blocks.RunFacts(
            driver=self.driver, purpose=self.purpose,
            argv=_command(list(sys.argv)), directory=self.directory,
            started=datetime.now().astimezone(), host=socket.gethostname(),
            job=_job(), logs=(blocks.display_path(self.path, self.directory),),
            debug=self.debug)))

    def devices(self, *, mesh_role: str | None = None,
                extras: Iterable[tuple[str, str]] = ()) -> None:
        """GPUs, mesh, memory, CPU and threads from the runtime facts."""
        facts = dict(self.runtime.facts)
        facts.setdefault("n_nodes", _node_count())
        self.heading("Devices")
        self.emit_lines(blocks.devices_lines(facts, mesh_role=mesh_role,
                                             extras=tuple(extras)))
        memory = facts.get("device_memory") or {}
        warning = blocks.device_memory_message(memory) if memory.get("total") else None
        if warning is not None:                     # printed by devices_lines
            self.count(warning)

    def settings(self, rows: Iterable[blocks.Setting]) -> None:
        """SETTINGS: key, value, source and a one-line effect per row."""
        self.heading("Settings")
        self.emit_lines(blocks.settings_lines(list(rows)))

    @staticmethod
    def flag_source(*flags: str, argv=None) -> str:
        """``cli`` when the command line names one of ``flags``, else ``default``."""
        args = list(sys.argv[1:] if argv is None else argv)
        named = any(arg == flag or arg.startswith(flag + "=")
                    for arg in args for flag in flags)
        return "cli" if named else "default"

    def sampling(self, *, wfn, sym, centroids=None, consequence: str | None = None) -> None:
        """SYMMETRY (operations, time reversal and its NOTE, the operation table)
        and K POINTS.  ``consequence`` is the driver's line on what it does about
        time reversal."""
        receipt = getattr(wfn, "trs_reference", None)
        if receipt is None:
            receipt = getattr(wfn, "density_symmetry", None)
        tr_lines, note = blocks.time_reversal_lines(receipt, sym, consequence=consequence)
        if note is not None:
            tr_lines = tr_lines + blocks.message_lines(note, indent=2)
        self.heading("Symmetry")
        self.emit_lines(blocks.symmetry_lines(sym, centroids=centroids, tr_lines=tr_lines))
        if note is not None:                        # printed in the block above
            self.count(note)
        self.heading("K points")
        self.emit_lines(blocks.kpoint_lines(wfn, sym))

    def spectral_compression(self, receipt, *,
                             title: str = "Htransform spectral compression") -> None:
        """Print the one shared Galerkin truncation/compression receipt."""
        self.heading(title)
        self.emit_lines(spectral_compression_lines(receipt))

    def htransform_quality(self, receipt, *,
                           title: str = "Htransform interpolation quality") -> None:
        """Print the shared representation/locality receipt."""
        self.heading(title)
        self.emit_lines(htransform_quality_lines(receipt))

    def timings(self, stages: Iterable[tuple], *, wall: float) -> None:
        """The timing table and the memory summary.

        ``stages`` are ``(label, seconds)`` or ``(label, seconds, device peak
        bytes)`` rows (``RunSession.stage_rows``).  The peaks, the host mark and
        the budget are this rank's (process zero's); no collective is entered,
        so a refusal can print the table too.
        """
        from common import timing
        rows = []
        for stage in stages:
            label, seconds = stage[0], float(stage[1])
            peak = stage[2] if len(stage) > 2 else None
            rows.append(blocks.TimingRow(str(label), seconds, 0,
                                         None if peak is None else float(peak) / 1e9))
        peaked = [row for row in rows if row.device_gb is not None]
        top = max(peaked, key=lambda row: row.device_gb, default=None)
        hosts = [float(r["host"]) for r in timing.records() if r.get("host") is not None]
        facts = getattr(self.runtime, "facts", None) or {}
        budget = float((facts.get("device_memory") or {}).get("budget") or 0.0) / 1e9
        self.heading("Major-stage timing")
        self.emit_lines(blocks.timing_lines(
            rows, wall=float(wall), budget_gb=budget or None,
            device_peak=(top.device_gb, top.label) if top is not None else None,
            host_peak=(max(hosts) / 1e9, max(hosts) / 1e9) if hosts else None))

    def warnings(self) -> None:
        """The WARNINGS summary, once: one line per id."""
        if self._warnings_emitted:
            return
        self._warnings_emitted = True
        self.emit_lines(blocks.warnings_summary_lines(self.messages()))

    def files(self, rows: Iterable[tuple[str, str, str]]) -> None:
        """Inputs read and outputs written, relative to the run directory."""
        own = abs_path(self.path)
        self.heading("Output files and inputs")
        self.emit_lines(blocks.output_file_lines(
            [row for row in rows if abs_path(row[2]) != own],
            directory=self.directory))

    def finish(self, *, status: str = "COMPLETED", wall: float | None = None,
               error: BaseException | None = None) -> None:
        """Warnings, then the closing line; ``error`` names what stopped the run.

        The ERROR block above the closing line gives the exception's id and its
        found / needs / fix clauses; the exception keeps its full text (with
        ``GATE <id>``) on stderr, and the log never says GATE.
        """
        self.warnings()
        if wall is None:
            from common.timing import process_elapsed_s
            wall = process_elapsed_s() or 0.0
        if error is not None:
            self.emit()
            self.emit_lines(blocks.message_lines(blocks.message_from_exception(error)))
        self.emit()
        self.emit_lines(blocks.closing_lines(
            self.driver, status, ended=datetime.now().astimezone(), wall_s=wall,
            n_warnings=blocks.warning_count(self.messages())))
        self.close()


__all__ = ["ProductionReport"]
