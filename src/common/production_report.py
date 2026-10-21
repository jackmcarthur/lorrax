"""The rank-zero main log every LORRAX driver report shares.

:class:`ProductionReport` owns the stream (process 0 writes the report file and
the live stdout), the retained warnings, and the blocks every driver prints the
same way: header, Devices, symmetry and sampling, the stage table, warnings,
the files table and the closing line.  A driver's report subclasses it only for
its own blocks (``gw.production_report``, ``bandstructure.production_report``,
``common.preprocessing_output``).  The text of each block is a pure function in
:mod:`common.report_blocks`.

Lines reach the log through ``logging`` (``common.log``): messages, block rows,
and, when verbose (``--verbose``, the deck's ``verbose = true`` or
``LORRAX_DEBUG_PRINT=1``), DEBUG records as rows under the current stage
(:meth:`ProductionReport.log_debug`).  Component text still handed to
:meth:`ProductionReport.legacy_print` is a DEBUG record; only a line naming a
warning becomes a message.  The JSON sidecar ``<report stem>.json`` is named
in the header and, once a record has written it, in the files table.
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
from common.log import get_logger, record, session_sidecar, written_sidecar
from common.report_blocks import Message
from common.scientific_output import (
    abs_path,
    htransform_quality_lines,
    spectral_compression_lines,
)


#: Component text handed to :meth:`ProductionReport.legacy_print`: DEBUG.
_LEGACY = get_logger("legacy")
#: The report's own DEBUG lines and records (the density check, repeated messages).
_LOG = get_logger("report")
#: The files-table role of the JSON sidecar.
SIDECAR_ROLE = "machine-readable record"


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
        # ``<where> samples [a, b] eV, <rest>``: the window and its samples are
        # the found line, the one a repeat prints (``again (n)``).
        where = re.match(r"(near grid|coarse window \d+) samples (\[[^\]]*\] eV),\s*(.*)$",
                         match.group(1))
        if where:
            return Message("WARNING", "sigma_reads_past_w_support",
                           "Sigma reads W past its line support",
                           found=(where.group(1), f"samples {where.group(2)}"),
                           lines=(where.group(3),)), False
        return Message("WARNING", "sigma_reads_past_w_support",
                       "Sigma reads W past its line support",
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
    match = re.match(r"w_head_densify = c1: the injected head and the S tensor "
                     r"disagree by (\S+) on (\(.*?\)) \(whead=(\S+), integrand=(\S+) ", body)
    if match:
        return Message(
            "WARNING", "bse_head_densify_mismatch",
            "the injected W head and the S tensor disagree",
            found=(f"head on {match.group(2)}",
                   f"off by {match.group(1)} (whead {match.group(3)}, "
                   f"integrand {match.group(4)})"),
            needs="agreement within 5%; the head keeps the injected scale",
            fix="check vhead/whead_0freq overrides, head_minibz_average and "
                "wcoul0_source against the GW run"), False
    match = re.match(r"BSE q=0 head NOT injected though G0_mu_nu is present and "
                     r"inject_head=True: (.*?)\. The q=0 ", body)
    if match:
        return Message(
            "WARNING", "bse_head_not_injected",
            "the q=0 exchange tile has no head: binding energies are under-bound",
            found=("head inputs", match.group(1)),
            fix="add vhead/whead_0freq to cohsex.in, or write vhead/whead "
                "into the restart"), False
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
    #: Words that make a :meth:`legacy_print` line a retained warning.
    _WARNING_WORDS = (
        "WARNING", "[WARN]", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
        "NOT TRUSTWORTHY", "DEPRECATED",
    )

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
        # Per summary key of a message printed in full: [times printed, the
        # detail line printed last] (a repeat prints ``again (n)``).
        self._printed: dict[tuple, list] = {}
        self._stage = "setup"
        self._warnings_emitted = False
        self._stream = None
        if self.rank == 0:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.path, "w", encoding="utf-8", buffering=1)

    @property
    def sidecar_path(self) -> str:
        """The JSON sidecar beside this log: ``<report stem>.json``."""
        return os.path.splitext(self.path)[0] + ".json"

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
    #: A stage entry: a plain trace ``[stage HH:MM:SS] -> name`` or a manual
    #: stage banner ``[i/N] title`` with its clock at column 100.
    _STAGE_ENTER = re.compile(
        r"^(?:\[stage \d\d:\d\d:\d\d\] -> |\[\d+/\d+\] )(.+?)(?: {2,}\d\d:\d\d:\d\d)?$")

    def progress(self, line: str) -> None:
        """Write one deliberately selected rank-zero progress line."""
        entered = self._STAGE_ENTER.match(str(line))
        if entered:
            self._stage = self._stage_name(entered.group(1))
        self.emit(line)

    @staticmethod
    def _stage_name(label: str) -> str:
        """The summary's stage column: the stage's name without its detail."""
        label = str(label).split(" (")[0].split(" + ")[0]
        label = label.split(" and ")[0].strip()    # "ISDF basis and Coulomb metric"
        return label[:20].rstrip()

    def _current_stage(self) -> str:
        """Where a message happening now belongs: this thread's open stage
        (``timing.current_stage_label``), else the last stage banner printed."""
        try:
            from common.timing import current_stage_label
            label = current_stage_label()
        except Exception:          # noqa: BLE001 - output never fails a run
            label = None
        return self._stage_name(label) if label else self._stage

    def legacy_print(self, *args, sep: str = " ", **kwargs) -> None:
        """A ``print_fn`` for component code that has not moved to logging.

        A line with a warning word (:attr:`_WARNING_WORDS`) becomes a message
        (:func:`_legacy_message`), printed here and counted in the WARNINGS
        summary.  Every other line is a DEBUG record (logger
        ``lorrax.legacy``): in the rank log and, under the current stage, in
        the main log when verbose, and nowhere otherwise.  ``kwargs`` of
        ``print`` (``end``, ``flush``) are accepted and unused.
        """
        text = sep.join(str(value) for value in args)
        if any(word in text.upper() for word in self._WARNING_WORDS):
            message, quiet = _legacy_message(text)
            self.warn(message, quiet=quiet)
            return
        _LEGACY.debug(text)

    @staticmethod
    def message_of(text: str, fields, levelno: int) -> tuple[Message, bool]:
        """A logging record's message (``common.log``) and whether it is quiet.

        A record with an ``id`` extra is that message; any other (a captured
        ``warnings.warn``, ``<Category>: <text>``) goes through the id table
        (:func:`_legacy_message`).  ERROR and above is an ERROR.
        """
        error = levelno >= 40                         # logging.ERROR
        if fields.get("id"):
            found = fields.get("found")
            return Message(
                str(fields.get("kind") or ("ERROR" if error else "WARNING")),
                str(fields["id"]), str(text),
                found=(str(found[0]), str(found[1])) if found else None,
                needs=fields.get("needs"), lines=tuple(fields.get("lines", ())),
                fix=fields.get("fix")), bool(fields.get("quiet"))
        message, quiet = _legacy_message(text)
        if error:
            message = dataclasses.replace(message, kind="ERROR")
        return message, quiet

    def log_message(self, text: str, fields, levelno: int) -> None:
        """The main-log handler's entry: print and count one logged message."""
        message, quiet = self.message_of(text, fields, levelno)
        self.warn(message, quiet=quiet)

    def log_detail(self, text: str, fields) -> None:
        """The main-log handler's entry for a block-tagged INFO line.

        The ``progress`` and ``stage`` blocks are the log's cadence and print
        verbatim (:meth:`progress`); any other block is a detail row under the
        current stage (``report_blocks.stage_detail``).
        """
        if fields.get("block") in ("progress", "stage"):
            self.progress(text)
        else:
            self.emit_lines(blocks.stage_detail(fields.get("label"), text))

    def log_debug(self, text: str, fields) -> None:
        """The main-log handler's entry for a DEBUG record, when verbose.

        A row at indent 6 under the current stage, one per line of the
        message; a record's payload is in the JSON sidecar and is not
        printed.  The stage the summary names does not change.
        """
        if str(text).strip():
            self.emit_lines(blocks.stage_detail(None, text))

    def render_message(self, text: str, fields, levelno: int) -> list[str]:
        """The lines :meth:`log_message` prints, for another rank's own log."""
        return blocks.message_lines(self.message_of(text, fields, levelno)[0])

    def warn(self, message: Message, *, indent: int = 0, quiet: bool = False) -> None:
        """Print a message where it happens and count it for the WARNINGS summary.

        The only way a message reaches the summary.  ``quiet`` counts it without
        printing: its fact is printed in full by another block (the Devices
        memory row, the symmetry block's time-reversal NOTE).  A message without
        its own stage takes the stage the run is in.

        A WARNING or NOTE whose id was printed in full before prints one line,
        ``<text> again (<n>)  [id]``, and its first detail line when that
        differs from the one printed last for the id; the full block is a DEBUG
        line and a ``messages.repeats`` record.  An ERROR always prints in full.
        """
        message = self.count(message, quiet=quiet)
        if quiet:
            return
        full = blocks.message_lines(message, indent=indent)
        key = blocks._summary_key(message)
        detail = blocks.message_detail(message)
        seen = self._printed.get(key) if message.kind != "ERROR" else None
        if seen is None:
            if message.kind != "ERROR":
                self._printed[key] = [1, detail]
            self.emit_lines(full)
            return
        seen[0] += 1
        shown = detail if detail != seen[1] else None
        if shown is not None:
            seen[1] = detail
        self.emit_lines(blocks.message_repeat_lines(message, seen[0], indent=indent,
                                                    detail=shown))
        try:
            _LOG.debug("\n".join(line.strip() for line in full), extra=record(
                "messages.repeats", kind=message.kind, id=message.id, text=message.text,
                occurrence=seen[0], stage=message.stage, lines=list(full)))
        except Exception:          # noqa: BLE001 - output never fails a run
            pass

    def count(self, message: Message, *, quiet: bool = False) -> Message:
        """Count a message a block has printed itself (Devices, SYMMETRY)."""
        if message.stage == "setup":
            stage = self._current_stage()
            if stage != "setup":
                message = dataclasses.replace(message, stage=stage)
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
        """The title and the run's facts (``input_file`` is in the command).

        ``Version`` is the checkout's commit, when startup read it
        (``runtime.source_closure.checkout_version``); ``Logs`` names the
        JSON sidecar when this run's session writes one; ``Verbose`` says
        what turned verbose on, when it is on.
        """
        from runtime import verbose_source
        from runtime.source_closure import checkout_version
        logs = [blocks.display_path(self.path, self.directory)]
        if session_sidecar() == self.sidecar_path:
            logs.append(blocks.display_path(self.sidecar_path, self.directory))
        self.emit_lines(blocks.header_lines(blocks.RunFacts(
            driver=self.driver, purpose=self.purpose,
            argv=_command(list(sys.argv)), directory=self.directory,
            started=datetime.now().astimezone(), host=socket.gethostname(),
            job=_job(), logs=tuple(logs), version=checkout_version())))
        source = verbose_source()
        if source is not None:
            self.emit_lines(blocks.kv_lines(
                [("Verbose", f"on ({source}); DEBUG lines print under each stage")],
                indent=0))

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
            check = blocks.time_reversal_check(receipt)
            if check is not None:                   # the NOTE's figures: DEBUG + record
                _LOG.debug(check[0], extra=record("symmetry.trs_check", text=check[0],
                                                  **check[1]))
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
        self.heading("Timing")
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
        """Inputs read and outputs written, relative to the run directory.

        The JSON sidecar is the last row once a record of this run wrote it.
        """
        own = abs_path(self.path)
        rows = [row for row in rows if abs_path(row[2]) != own]
        sidecar = written_sidecar()
        if sidecar is not None and not any(abs_path(row[2]) == abs_path(sidecar)
                                           for row in rows):
            rows.append((SIDECAR_ROLE, "written", sidecar))
        self.heading("Output files and inputs")
        self.emit_lines(blocks.output_file_lines(rows, directory=self.directory))

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
