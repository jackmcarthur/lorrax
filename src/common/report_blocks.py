"""The text blocks of a LORRAX driver's main log: pure functions, facts in, lines out.

Every driver's log is built from the same blocks so that the same fact reads the
same way in every log.  A function here takes facts that are already resolved
(runtime facts, paths, clocks) and returns lines; it opens no file, reads no
environment and decides no physics.  The rank-zero stream that prints them is
:class:`common.production_report.ProductionReport`.

Layout grid (all blocks):

* titles, block headings and the closing line start at column 0;
* a block row is ``label value`` at indent 2, the label padded to 17;
* a right-aligned item (an ``[id]``, a wall) ends at column 100;
* a line holds at most 100 columns; only a command or a path may run over.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

#: Line width of the main log.
WIDTH = 100
#: Width of a block row's label column, separator included.
LABEL = 17
#: The rule above and below the title and above the closing line.
RULE = "=" * WIDTH

#: Width of the outputs table's role column: at least this, at most :data:`ROLE_MAX`.
ROLE_MIN = 24
ROLE_MAX = 44
#: ``state`` values of the outputs table that name an input; inputs print first.
INPUT_STATES = ("read", "reused", "checked")

#: Thread variables the Devices block names, in print order, with their short names.
_THREAD_NAMES = (
    ("OPENBLAS_NUM_THREADS", "OpenBLAS"),
    ("OMP_NUM_THREADS", "OMP"),
    ("MKL_NUM_THREADS", "MKL"),
    ("LORRAX_MKLBLAS_THREADS", "LORRAX MKL BLAS"),
)


# ── primitives ───────────────────────────────────────────────────────────────
def heading(title: str) -> list[str]:
    """A block heading: a blank line, the title in upper case, a dash rule."""
    text = str(title).upper()
    return ["", text, "-" * len(text)]


def kv_lines(rows: Sequence[tuple[str, str]], *, indent: int = 2,
             width: int = LABEL) -> list[str]:
    """``label value`` rows, the label padded to ``width`` after ``indent`` spaces."""
    pad = " " * indent
    return [f"{pad}{str(label):<{width - 1}} {value}".rstrip()
            for label, value in rows]


def table_lines(header: Sequence[str], rows: Sequence[Sequence[str]], *,
                widths: Sequence[int], align: str, indent: int = 2,
                legend: Sequence[tuple[str, str]] = ()) -> list[str]:
    """A fixed-column table: one space between columns, the last column unpadded.

    ``align`` holds one ``l`` or ``r`` per column.  A ``legend`` prints below the
    table as ``label value`` rows at indent ``indent + 2``.
    """
    if not (len(header) == len(widths) == len(align)):
        raise ValueError("table_lines: header, widths and align differ in length")
    last = len(widths) - 1

    def line(cells) -> str:
        parts = []
        for i, (cell, width, side) in enumerate(zip(cells, widths, align)):
            cell = str(cell)
            if i == last and side == "l":
                parts.append(cell)
            else:
                parts.append(f"{cell:>{width}}" if side == "r" else f"{cell:<{width}}")
        return (" " * indent + " ".join(parts)).rstrip()

    return ([line(header)] + [line(row) for row in rows]
            + kv_lines(legend, indent=indent + 2))


def right_align(left: str, right: str, width: int = WIDTH) -> str:
    """``left`` with ``right`` ending at column ``width`` (two spaces at least)."""
    if not right:
        return left
    gap = width - len(left) - len(right)
    return left + " " * max(gap, 2) + right


def fmt_wall(seconds: float) -> str:
    """A wall time: ``78.5 s`` below 100 s, ``18 min 39 s`` below an hour, else
    ``1 h 02 min``."""
    seconds = max(float(seconds), 0.0)
    if round(seconds, 1) < 100.0:
        return f"{seconds:.1f} s"
    whole = int(round(seconds))
    if whole < 3600:
        return f"{whole // 60} min {whole % 60:02d} s"
    minutes = int(round(seconds / 60.0))
    return f"{minutes // 60} h {minutes % 60:02d} min"


def fmt_clock(moment: datetime) -> str:
    """A date and time with its zone: ``2026-10-19 19:16:11 PDT``."""
    return moment.strftime("%Y-%m-%d %H:%M:%S %Z").rstrip()


def wrap_value(text: str, *, first_col: int, width: int = WIDTH) -> list[str]:
    """Wrap ``text`` between words for a value that starts at ``first_col``.

    The first line is returned bare (the caller puts its label before it); each
    continuation line is indented to ``first_col``.  A single word longer than
    the room runs over.
    """
    room = max(width - first_col, 1)
    lines: list[str] = []
    current = ""
    for word in str(text).split():
        if current and len(current) + 1 + len(word) > room:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    lines.append(current)
    return [lines[0]] + [" " * first_col + line for line in lines[1:]]


def display_path(path, directory: str) -> str:
    """``path`` relative to ``directory`` when it lies under it, else absolute."""
    if not path or str(path) == "-":
        return "-"
    full = os.path.abspath(os.fspath(path))
    root = os.path.abspath(os.fspath(directory))
    if full == root or full.startswith(root.rstrip(os.sep) + os.sep):
        return os.path.relpath(full, root)
    return full


# ── header ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RunFacts:
    """What the header states about one driver invocation."""

    driver: str                       # "gw", "bse", ...: the title and the closing line
    purpose: str                      # "GW quasiparticles"
    argv: tuple[str, ...]             # the command, interpreter first
    directory: str                    # the run directory; outputs print relative to it
    started: datetime                 # tz-aware
    host: str
    job: str | None = None            # "Slurm job 60021274" or "Slurm job 59988848, step 6"
    logs: tuple[str, ...] = ()        # the main log, relative to ``directory``
    debug: bool = False
    version: str | None = None        # printed only when given
    native_bundle: str | None = None  # printed only when given


def title_lines(driver: str, purpose: str) -> list[str]:
    """The title between two rules: ``LORRAX <driver> · <purpose>``."""
    title = f"LORRAX {driver}" + (f" · {purpose}" if purpose else "")
    return [RULE, title, RULE]


def header_lines(run: RunFacts) -> list[str]:
    """The title and the run's facts, one per line, labels flush left."""
    command = wrap_value(" ".join(run.argv), first_col=LABEL)
    rows = []
    if run.version:
        rows.append(("Version", run.version))
    if run.native_bundle:
        rows.append(("Native bundle", run.native_bundle))
    rows.append(("Command", command[0]))
    rows.append(("Directory", run.directory))
    rows.append(("Started", fmt_clock(run.started)))
    rows.append(("Host", run.host + (f", {run.job}" if run.job else "")))
    if run.logs:
        rows.append(("Logs", ", ".join(run.logs)))
    if run.debug:
        rows.append(("Debug", "on (LORRAX_DEBUG_PRINT=1)"))
    lines = title_lines(run.driver, run.purpose)
    for label, value in rows:
        lines += kv_lines([(label, value)], indent=0)
        if label == "Command":
            lines += command[1:]
    return lines


# ── devices ──────────────────────────────────────────────────────────────────
def devices_lines(facts: Mapping, *, mesh_role: str | None,
                  extras: Sequence[tuple[str, str]] = ()) -> list[str]:
    """The Devices block's rows from the runtime facts (``runtime.facts``).

    ``facts`` may carry ``n_nodes``; without it the rank row gives the count
    alone.  ``mesh_role`` names what the two mesh axes hold; ``None`` omits the
    mesh row.  ``extras`` are a driver's own ``(label, value)`` rows, printed last.
    """
    p = int(facts.get("process_count", 1))
    nodes = facts.get("n_nodes")
    ranks = str(p) + (f" on {int(nodes)} node{'' if int(nodes) == 1 else 's'}"
                      if nodes else "")
    ndev = int(facts.get("n_devices", 0))
    nlocal = int(facts.get("n_local_devices", 0))
    kind = str(facts.get("device_kind", "unknown"))
    backend = str(facts.get("backend", "unknown")).lower()
    per_rank = f", {nlocal} per rank" if nlocal else ""
    if backend in ("gpu", "cuda", "rocm"):
        devices = ("GPUs", f"{ndev} x {kind}{per_rank}")
    else:
        devices = ("GPUs", f"none; {ndev} {backend.upper()} devices{per_rank}")
    rows = [("MPI ranks", ranks), devices]
    if mesh_role is not None:
        mesh = tuple(int(v) for v in facts.get("mesh_shape", (1, 1)))
        rows.append(("mesh", f"{mesh[0]} x {mesh[1]} ({mesh_role})"))
    threads = facts.get("threads") or {}
    affinity = threads.get("affinity")
    rows.append(("CPU", f"{affinity} cores per rank" if affinity is not None
                 else "unknown cores per rank"))
    # A runtime note names the library it applied the count to; the row keeps
    # the count and why (``6 (auto: physical cores of the affinity mask)``).
    named = [f"{short} " + re.sub(r";\s*applied to [^)]*", "", str(threads[key]))
             for key, short in _THREAD_NAMES if threads.get(key) not in (None, "")]
    rows.append(("host threads", ", ".join(named) if named else "runtime defaults"))
    lines = kv_lines(rows)
    memory = facts.get("device_memory") or {}
    if memory.get("total"):
        outside = memory.get("outside_max")
        lines += kv_lines((
            ("device memory", f"card {memory['total'] / 1e9:.2f} GB, XLA pool "
                              f"{memory['reservation'] / 1e9:.2f} GB, "
                              f"{memory['headroom'] / 1e9:.2f} GB outside the pool"),
            ("memory budget", f"{memory['budget'] / 1e9:.2f} GB per device"),
            ("after warm-up", "unread" if outside is None else
                              f"{outside / 1e9:.2f} GB outside the pool (max over ranks)"),
        ))
        warning = device_memory_message(memory)
        if warning is not None:
            lines += message_lines(warning, indent=2)
    precision = "FP64 / complex128" if facts.get("x64") else "FP32 / complex64"
    collectives = ("NCCL" if backend in ("gpu", "cuda", "rocm") else
                   str((facts.get("collectives") or {}).get("impl", "local")).upper())
    lines += kv_lines((("JAX", f"{facts.get('jax_version', 'unknown')} / jaxlib "
                               f"{facts.get('jaxlib_version', 'unknown')}, {precision}, "
                               f"{collectives} collectives"),
                       *extras))
    return lines


def device_memory_message(memory: Mapping) -> "Message | None":
    """``WARNING [device_memory_outside_pool]`` when the runtime's rule fires."""
    # The 0.75-of-headroom rule and its text are the runtime's own.
    from runtime import device_memory_summary
    warning = device_memory_summary(memory)[1]
    if not warning:
        return None
    return Message("WARNING", "device_memory_outside_pool",
                   "memory outside the XLA pool is high", lines=(warning,), stage="setup")


# ── messages: ERROR, WARNING, NOTE ───────────────────────────────────────────
#: The three message kinds, in the order the summary lists them.
KINDS = ("ERROR", "WARNING", "NOTE")
#: Column of a message's text, relative to its indent: ``WARNING`` plus two spaces.
TEXT_COL = 9


@dataclass(frozen=True)
class Message:
    """One ERROR, WARNING or NOTE: plain words, an id for scripts, then a fix.

    ``found`` is ``(what, value)`` (``("dipole.h5", "bands 1-24")``); ``needs``
    is what this run needs; ``lines`` are free lines; ``fix`` prints last.
    ``stage`` names where it happened, for the WARNINGS summary.
    """

    kind: str
    id: str
    text: str
    found: tuple[str, str] | None = None
    needs: str | None = None
    lines: tuple[str, ...] = ()
    fix: str | None = None
    stage: str = "setup"


def _indented(column: int, text: str) -> list[str]:
    """``text`` wrapped between words, every line starting at ``column``."""
    wrapped = wrap_value(text, first_col=column)
    return [" " * column + wrapped[0]] + wrapped[1:]


def message_lines(msg: Message, *, indent: int = 0) -> list[str]:
    """A message block: kind and text, ``[id]`` ending at column 100, then the
    ``found / needs`` line, the free lines and ``fix:``, all at the text column."""
    if msg.kind not in KINDS:
        raise ValueError(f"message_lines: kind {msg.kind!r} is not one of {KINDS}")
    tag = f"[{msg.id}]"
    column = indent + TEXT_COL
    head = wrap_value(msg.text, first_col=column, width=WIDTH - len(tag) - 2)
    lines = [right_align(" " * indent + f"{msg.kind:<{TEXT_COL}}" + head[0], tag)]
    lines += head[1:]
    parts = ([f"{msg.found[0]}: {msg.found[1]}"] if msg.found else []) + (
        [f"this run needs: {msg.needs}"] if msg.needs else [])
    if parts:
        joined = " " * column + "    ".join(parts)
        if len(joined) <= WIDTH:
            lines.append(joined)
        else:
            for part in parts:
                lines += _indented(column, part)
    for line in msg.lines:
        lines += _indented(column, line)
    if msg.fix:
        lines += _indented(column, f"fix: {msg.fix}")
    return lines


def _summary_key(msg: Message) -> tuple:
    """One summary line per kind and id; an unclassified message keeps its text."""
    return (msg.kind, msg.id, msg.text if msg.id == "unclassified" else "")


def warning_count(messages: Sequence[Message]) -> int:
    """The closing line's count: distinct WARNING lines of the summary."""
    return len({_summary_key(m) for m in messages if m.kind == "WARNING"})


def warnings_summary_lines(messages: Sequence[Message]) -> list[str]:
    """The WARNINGS block: one line per id, with its stage and count.

    ``messages`` holds one entry per occurrence; each was printed in full where
    it happened.  With nothing to report the block is the line ``WARNINGS  none``.
    """
    if not messages:
        return ["", "WARNINGS  none"]
    groups: dict[tuple, list] = {}
    for msg in messages:
        entry = groups.setdefault(_summary_key(msg), [msg, 0])
        entry[1] += 1
    ordered = sorted(groups.values(), key=lambda entry: KINDS.index(entry[0].kind))
    counts = []
    for kind, noun in (("ERROR", "error"), ("WARNING", "warning"), ("NOTE", "note")):
        n = sum(1 for msg, _ in ordered if msg.kind == kind)
        if n:
            counts.append(f"{n} {noun}{'' if n == 1 else 's'}")
    id_width = max(27, *(len(msg.id) + 3 for msg, _ in ordered))
    stage_width = min(max(5, *(len(msg.stage) for msg, _ in ordered)), 24)
    lines = heading("Warnings") + [
        "  " + ", ".join(counts) + "; each was printed in full where it happened"]
    for msg, count in ordered:
        left = (f"  {msg.kind:<7} {'[' + msg.id + ']':<{id_width}} "
                f"{msg.stage[:stage_width]:<{stage_width}}   x{count:<3}")
        text = " ".join(str(msg.text).split())
        room = WIDTH - len(left)
        if len(text) > room:
            text = text[:max(room - 3, 0)].rstrip() + "..."
        lines.append((left + text).rstrip())
    return lines


#: Clauses of a refusal's text, in LORRAX's ``GATE <id>: lead; got: ...; want:
#: ...; fix: ...; why: ...`` grammar.
_CLAUSE = re.compile(r"(?:^|;\s+)(got|want|fix|why|doc):\s*", re.IGNORECASE)


def message_from_exception(error: BaseException) -> Message:
    """The ERROR block for an exception that stopped the run.

    The id is the exception's ``GATE <id>`` name (``unclassified`` without one);
    the text never says GATE.  ``got`` and ``want`` clauses become the found and
    needs line, ``fix`` the fix line, ``why`` a free line.  The full message
    stays on stderr.
    """
    raw = " ".join(str(error).split())
    gate = re.search(r"GATE (\w+):?\s*", raw)
    ident = gate.group(1) if gate else "unclassified"
    body = raw[gate.end():] if gate else raw
    pieces = _CLAUSE.split(body)
    lead = pieces[0].strip().rstrip(";").strip()
    clauses = {key.lower(): value.strip().rstrip(".;").strip()
               for key, value in zip(pieces[1::2], pieces[2::2])}
    kind_name = type(error).__name__
    if len(lead) > 300:
        lead = lead[:297].rstrip() + "..."
    lines = ([f"why: {clauses['why']}"] if clauses.get("why") else []) + [
        f"{kind_name}; the full message is on stderr"]
    return Message(
        "ERROR", ident, lead or f"the run stopped with {kind_name}",
        found=("found", clauses["got"]) if clauses.get("got") else None,
        needs=clauses.get("want") or None, lines=tuple(lines),
        fix=clauses.get("fix") or None, stage="")


# ── symmetry, time reversal, k points ───────────────────────────────────────
#: Decimals of a fractional translation and of a k coordinate or weight.
TAU_DIGITS = 3
K_DIGITS = 5


def _clean(values, digits: int):
    """Display values with round-to-zero results made positive zero."""
    array = np.asarray(values, dtype=np.float64).copy()
    array[np.abs(array) < 0.5 * 10.0 ** (-digits)] = 0.0
    return array


def _matrix_text(matrix) -> str:
    rows = np.asarray(matrix, dtype=np.int64).reshape(3, 3)
    return "[" + "; ".join(" ".join(f"{int(v):2d}" for v in row) for row in rows) + "]"


def _used_as(active: set, row: int, n_sym: int) -> str:
    """How the run uses operation ``row``: alone, composed with time reversal, both."""
    alone, with_tr = row in active, row + n_sym in active
    return ("both" if alone and with_tr else "unitary" if alone
            else "with TR" if with_tr else "unused")


def _qe_antiunitary(sym, n_sym: int):
    """QE's ``t_rev`` bit per operation, or ``None`` without a QE schema."""
    binding = getattr(sym, "qe_symmetry_binding", None)
    if binding is None:
        return None
    flags = np.asarray(getattr(sym, "qe_operation_antiunitary", binding.antiunitary),
                       dtype=bool).reshape(-1)
    if flags.shape != (n_sym,):
        raise ValueError(f"symmetry_lines: QE operation typing has shape {flags.shape}, "
                         f"expected ({n_sym},)")
    return flags


def time_reversal_lines(receipt, sym, *, consequence: str | None = None
                        ) -> tuple[list[str], "Message | None"]:
    """The ``time reversal`` row and, when it is broken, its NOTE.

    ``receipt`` is the WFN's time-reversal check (``WfnLoader.trs_reference``
    or ``density_symmetry``; ``None`` when none was retained).  Only its
    verdict and ``SymMaps.trs_allowed`` are quoted; nothing is rechecked.
    ``consequence`` is one line a driver adds (what the run does about it).
    The NOTE is returned, not printed: the report prints and counts it.
    """
    n_sym = int(np.asarray(sym.Rinv_grid).shape[0])
    qe_tr = _qe_antiunitary(sym, n_sym)
    n_anti = 0 if qe_tr is None else int(np.count_nonzero(qe_tr))
    allowed = bool(sym.trs_allowed)
    note = None
    method = getattr(receipt, "method", "real-space-density") if receipt is not None else None
    if receipt is None:
        value = ("used" if allowed else "not used") + " (no density check retained)"
    elif method == "occupied-density-subspace" and receipt.trs_basis == "not-2c":
        value = (("used" if allowed else "not used")
                 + f" (nspin {int(receipt.nspin)}, nspinor {int(receipt.nspinor)}: "
                 "no 2-component check)")
    else:
        if method == "occupied-density-subspace":
            holds, conclusive = bool(receipt.trs_holds), bool(receipt.conclusive)
            residual = ("n/a" if receipt.subspace_residual is None
                        else f"{float(receipt.subspace_residual):.1e}")
            evidence = ", ".join(f"{kind} {int(count)}"
                                 for kind, count in receipt.evidence_counts)
            check = (f"2-component density check: basis {receipt.trs_basis}, residual "
                     f"{residual}, coverage {100.0 * float(receipt.trs_coverage):.2f}%, "
                     + (evidence or "no independent evidence"))
        else:
            holds, conclusive = bool(receipt.trs_holds), True
            m_rel = "n/a" if receipt.m_rel is None else f"{float(receipt.m_rel):.1e}"
            check = (f"density check: basis {receipt.trs_basis}, |m|/|rho| {m_rel}, "
                     f"coverage {100.0 * float(receipt.trs_coverage):.2f}%")
        if holds:
            value = ("used" if allowed else "not used") + (
                "" if conclusive else " (the density check is inconclusive)")
        else:
            value = "not a symmetry alone: " + (
                f"used only inside the {n_anti} t_rev = 1 operations" if n_anti
                else "k and -k are sampled apart")
            note = Message(
                "NOTE", "time_reversal_broken",
                (f"QE marks {n_anti} of the {n_sym} operations t_rev = 1 (magnetic)"
                 if n_anti else "the DFT reference breaks time reversal (magnetic)"),
                lines=(check, f"k and -k pair only through those {n_anti} operations"
                       if n_anti else "k and -k are not paired by time reversal"))
    lines = kv_lines([("time reversal", value)])
    if consequence:
        lines += _indented(LABEL + 2, consequence)
    return lines, note


def symmetry_lines(sym, *, centroids=None, tr_lines: Sequence[str] = ()) -> list[str]:
    """The SYMMETRY block's rows and the operation table (always in full).

    ``centroids`` is a loaded centroid set with a ``closure`` verdict (only for
    drivers that read centroids); ``tr_lines`` are :func:`time_reversal_lines`'
    row and NOTE, printed above the table.
    """
    rotations = np.asarray(sym.Rinv_grid)
    n_sym = int(rotations.shape[0])
    tau = np.asarray(sym.translations, dtype=np.float64)[:n_sym] / (2.0 * np.pi)
    fractional = int(np.count_nonzero(
        np.any(np.abs(tau - np.rint(tau)) > 1e-7, axis=1)))
    inversion = [f"S{int(r) + 1:02d}" for r in np.flatnonzero(
        np.all(rotations == -np.eye(3, dtype=rotations.dtype), axis=(1, 2)))]
    active = {int(r) for r in np.asarray(sym.active_symmetry_rows,
                                         dtype=np.int32).reshape(-1)}
    used = [_used_as(active, i, n_sym) for i in range(n_sym)]
    n_used = sum(u != "unused" for u in used)
    qe_tr = _qe_antiunitary(sym, n_sym)
    rows = [("operations", f"{n_sym}, {fractional} with fractional translations; "
             + (f"inversion {', '.join(inversion)}" if inversion else "no inversion"))]
    usage = f"all {n_sym} used" if n_used == n_sym else f"{n_used} of {n_sym} used"
    if qe_tr is None:
        diagnostic = str(getattr(sym, "qe_symmetry_diagnostic",
                                 "no authenticated QE data-file-schema.xml"))
        rows.append(("QE types", f"unknown ({diagnostic}); {usage}"))
    else:
        n_anti = int(np.count_nonzero(qe_tr))
        rows.append(("QE types", f"{n_sym - n_anti} unitary, {n_anti} composed with time "
                                 f"reversal; {usage}"))
    if centroids is not None:
        verdict = centroids.closure
        n_ops, n_bad = int(verdict.n_sym), int(verdict.n_violating)
        sites = int(verdict.n_centroids)
        rows.append(("centroid orbit",
                     f"closed: {n_ops - n_bad}/{n_ops} operations map the {sites} sites "
                     "onto themselves" if bool(verdict.closed) else
                     f"NOT closed: {n_bad}/{n_ops} operations move the {sites}-site set; "
                     f"worst residual {float(verdict.worst_residual):.1e} at "
                     f"S{int(verdict.worst_op) + 1:02d}"))
    lines = kv_lines(rows) + list(tr_lines)
    lines.append(f"  {'op':<5} {'QE-TR':<6} {'used as':<9} {'R^-1':<32} "
                 f"{'tau':<8} (r' = R^-1 r + tau)")
    shown = _clean(tau, TAU_DIGITS)
    for i, (rotation, shift) in enumerate(zip(rotations, shown)):
        flag = "-" if qe_tr is None else ("yes" if bool(qe_tr[i]) else "no")
        lines.append(f"  S{i + 1:02d}   {flag:<6} {used[i]:<9} {_matrix_text(rotation):<32} "
                     f"({shift[0]: .{TAU_DIGITS}f} {shift[1]: .{TAU_DIGITS}f} "
                     f"{shift[2]: .{TAU_DIGITS}f})")
    return lines


def kpoint_lines(wfn, sym, *, extra: Mapping[str, Sequence] = {},
                 max_rows: int = 100) -> list[str]:
    """The K POINTS block: the grid, then the stored list in crystal coordinates.

    A stored set that is the whole grid is one line and no list.  ``extra``
    adds columns (``{"ngk": [...]}``).  A list longer than ``max_rows`` prints
    20 rows and the count of the rest.
    """
    grid = tuple(int(v) for v in np.asarray(wfn.kgrid).reshape(-1)[:3])
    shift = _clean(np.asarray(wfn.shift, dtype=np.float64).reshape(-1)[:3], K_DIGITS)
    shift_text = ("unshifted" if not np.any(shift) else
                  "shift (" + ", ".join(f"{v:.3f}" for v in shift) + ")")
    n_full, n_red = int(sym.nk_tot), int(sym.nk_red)
    head = f"{grid[0]} x {grid[1]} x {grid[2]}, {shift_text}: "
    if n_red == n_full:
        return kv_lines([("k grid", head + f"{n_full} k, all stored (no IBZ reduction)")])
    lines = kv_lines([("k grid", head + f"{n_full} k in the BZ, {n_red} in the IBZ")])
    weights = np.asarray(wfn.kweights, dtype=np.float64)
    if float(np.sum(weights)) != 0.0:
        weights = weights / float(np.sum(weights))
    points = _clean(wfn.kpoints, K_DIGITS)
    weights = _clean(weights, K_DIGITS)
    columns = {name: [str(v) for v in values] for name, values in extra.items()}
    widths = {name: max([len(name)] + [len(v) for v in values]) + 3
              for name, values in columns.items()}
    lines.append(f"{'ik':>7}{'kx':>11}{'ky':>11}{'kz':>11}{'weight':>11}"
                 + "".join(f"{name:>{widths[name]}}" for name in columns))
    count = len(points)
    shown = range(count) if count <= max_rows else range(20)
    for i in shown:
        p = points[i]
        lines.append(f"{i + 1:>7}{p[0]:>11.{K_DIGITS}f}{p[1]:>11.{K_DIGITS}f}"
                     f"{p[2]:>11.{K_DIGITS}f}{weights[i]:>11.{K_DIGITS}f}"
                     + "".join(f"{columns[name][i]:>{widths[name]}}" for name in columns))
    if count > max_rows:
        lines.append(f"{'':7}[... {count - 20} more k points]")
    return lines


# ── timing and memory ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class TimingRow:
    """One row of the timing table: a stage (depth 0) or one of its parts (depth 1)."""

    label: str
    seconds: float
    depth: int = 0
    device_gb: float | None = None


def timing_lines(rows: Sequence[TimingRow], *, wall: float, compile: str | None = None,
                 notes: Sequence[str] = (), budget_gb: float | None = None,
                 device_peak: tuple[float, str] | None = None,
                 host_peak: tuple[float, float] | None = None,
                 n_ranks: int | None = None) -> list[str]:
    """The timing table with its device-peak column, then the memory summary.

    ``rows`` close against ``wall`` (the caller adds the ``other`` row); a
    ``total run`` row closes the table.  ``device_gb`` is the row's pool
    high-water, the largest over ranks when they were gathered (``n_ranks``).
    Below the table: the compile line, the driver's ``notes`` (each wrapped at
    100 columns), then ``budget``, ``device peak`` (GB and
    the row that set it) and ``host peak`` (largest and smallest GB per rank),
    each only when given.  Fractions are of ``wall``.
    """
    width = max([36] + [len(row.label) + 2 * row.depth for row in rows])
    peaks = [row.device_gb for row in rows if row.device_gb is not None]

    def line(label: str, depth: int, seconds: float, gb: float | None) -> str:
        fraction = 100.0 * float(seconds) / wall if wall else 0.0
        text = (f"  {' ' * (2 * depth)}{label:<{width - 2 * depth}}"
                f"{float(seconds):>11.2f}{fraction:>10.2f}%")
        return text + (f"{gb:>12.2f}" if gb is not None else "")

    lines = [f"  {'stage':<{width}}{'wall (s)':>11}{'fraction':>11}"
             + (f"{'device GB':>12}" if peaks else "")]
    lines += [line(row.label, row.depth, row.seconds, row.device_gb) for row in rows]
    lines.append(f"  {'total run':<{width}}{float(wall):>11.2f}{100.0:>10.2f}%"
                 + (f"{max(peaks):>12.2f}" if peaks else ""))
    if compile:
        lines.append(f"  {compile}")
    for note in notes:
        wrapped = wrap_value(note, first_col=4)
        lines += ["  " + wrapped[0]] + wrapped[1:]
    over = f", max over {n_ranks} ranks" if n_ranks and n_ranks > 1 else ""
    if budget_gb:
        lines += kv_lines([("budget", f"{budget_gb:.2f} GB per device (memory_per_device_gb)")])
    if device_peak is not None:
        verdict = ("" if not budget_gb else
                   " (within the budget)" if device_peak[0] <= budget_gb else
                   " (OVER the budget)")
        lines += kv_lines([("device peak",
                            f"{device_peak[0]:.2f} GB in {device_peak[1]}{over}{verdict}")])
    if host_peak is not None:
        high, low = host_peak
        lines += kv_lines([("host peak", f"{high:.2f} GB per rank{over}" + (
            f" (min {low:.2f} GB)" if over else ""))])
    return lines


# ── settings ────────────────────────────────────────────────────────────────
#: Where a setting's value came from.
SOURCES = ("deck", "cli", "default", "derived", "WFN", "QE", "bundle", "env")
#: Column of a setting's effect (``key`` 24, ``value`` 15, ``source`` 8, gaps).
EFFECT_COL = 52


@dataclass(frozen=True)
class Setting:
    """One row of SETTINGS: the key, its value, where it came from, what it does."""

    key: str
    value: str
    source: str
    effect: str


def settings_lines(rows: Sequence[Setting]) -> list[str]:
    """The SETTINGS table, ``key / value / source / effect``, BerkeleyGW style.

    The caller passes only what the user set, what was derived, and defaults
    that change the result.  An effect wraps to at most two lines at its
    column (52, or further right when a value is longer than 15, up to 20); a
    value longer than 20 pushes its own source and effect right.
    """
    for row in rows:
        if row.source not in SOURCES:
            raise ValueError(f"settings_lines: source {row.source!r} of {row.key} is not "
                             f"one of {SOURCES}")
    # The value column widens to the longest value up to 20, moving the effect
    # column with it, so every effect starts in one column.
    value_width = min(max([15] + [len(row.value) for row in rows]), 20)
    effect_col = EFFECT_COL + value_width - 15
    lines = [f"  {'key':<24} {'value':<{value_width}} {'source':<8} effect"]
    for row in rows:
        left = f"  {row.key:<24} {row.value:<{value_width}} {row.source:<8} "
        column = max(len(left), effect_col)
        effect = wrap_value(row.effect, first_col=column)
        if len(effect) > 2:
            tail = " ".join(part.strip() for part in effect[1:])
            room = WIDTH - column
            effect = [effect[0], " " * column + (
                tail if len(tail) <= room else tail[:room - 3].rstrip() + "...")]
        lines.append((left.ljust(column) + effect[0]).rstrip())
        lines += effect[1:]
    return lines


# ── outputs ──────────────────────────────────────────────────────────────────
def output_file_lines(rows: Sequence[tuple[str, str, str]], *, directory: str,
                      next_step: str | None = None) -> list[str]:
    """The files table: ``role / state / path``, inputs first, paths relative.

    ``rows`` are ``(role, state, path)``.  A row whose state is in
    :data:`INPUT_STATES` is an input and prints before the outputs; the order is
    otherwise the caller's.  The caller leaves out the log itself.
    """
    ordered = sorted(rows, key=lambda row: row[1] not in INPUT_STATES)
    width = min(max([ROLE_MIN] + [len(str(role)) for role, _, _ in ordered]), ROLE_MAX)
    state_width = max([11] + [len(str(state)) + 1 for _, state, _ in ordered])
    lines = table_lines(
        ("role", "state", "path"),
        [(role, state, display_path(path, directory)) for role, state, path in ordered],
        widths=(width, state_width, 0), align="lll")
    if next_step:
        lines.append(f"  {'next':<{LABEL - 1}} {next_step}")
    return lines


# ── closing ──────────────────────────────────────────────────────────────────
def closing_lines(driver: str, status: str, *, ended: datetime, wall_s: float,
                  n_warnings: int) -> list[str]:
    """The rule and the one closing line; the warning count only when nonzero."""
    if status not in ("COMPLETED", "ERROR"):
        raise ValueError(f"closing_lines: status {status!r} is not COMPLETED or ERROR")
    line = f"LORRAX {driver} {status} {fmt_clock(ended)} · {fmt_wall(wall_s)}"
    if n_warnings:
        line += f" · {int(n_warnings)} warning{'' if n_warnings == 1 else 's'}"
    return [RULE, line]


__all__ = [
    "INPUT_STATES",
    "KINDS",
    "LABEL",
    "Message",
    "RULE",
    "RunFacts",
    "SOURCES",
    "Setting",
    "TEXT_COL",
    "TimingRow",
    "WIDTH",
    "closing_lines",
    "device_memory_message",
    "devices_lines",
    "display_path",
    "message_from_exception",
    "message_lines",
    "warning_count",
    "warnings_summary_lines",
    "fmt_clock",
    "fmt_wall",
    "header_lines",
    "heading",
    "kpoint_lines",
    "kv_lines",
    "output_file_lines",
    "settings_lines",
    "symmetry_lines",
    "time_reversal_lines",
    "timing_lines",
    "right_align",
    "table_lines",
    "title_lines",
    "wrap_value",
]
