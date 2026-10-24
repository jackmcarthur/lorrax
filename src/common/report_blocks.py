"""The text blocks of a LORRAX driver's main log: pure functions, facts in, lines out.

Every driver's log is built from the same blocks so that the same fact reads the
same way in every log.  A function here takes facts that are already resolved
(runtime facts, paths, clocks) and returns lines; it opens no file, reads no
environment and decides no physics.  The rank-zero stream that prints them is
:class:`common.production_report.ProductionReport`.

Layout grid (all blocks):

* titles, block headings and the closing line start at column 0;
* two blank lines above each block heading, a ``=`` rule above and below each
  ``[i/N]`` stage banner (``common.timing``);
* a block row is ``label value`` at indent 2, the label padded to 17;
* a table's header row is underlined, one dash run per column
  (:func:`underline`);
* a number in a table column has one fixed format, right-aligned
  (:func:`fixed`: a sign column, two digits, two decimals);
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
def heading(title: str, suffix: str = "") -> list[str]:
    """A block heading: two blank lines, the title in upper case (``suffix``,
    a file name, as written), a dash rule."""
    text = str(title).upper() + (f" {suffix}" if suffix else "")
    return ["", "", text, "-" * len(text)]


def underline(header: str) -> str:
    """The rule under a table's header row: a dash run under each column title
    (a title of words one space apart is one column), blanks between."""
    return re.sub(r"\S+(?: \S+)*", lambda m: "-" * len(m.group()), str(header)).rstrip()


def fixed(value: float, *, digits: int = 2, decimals: int = 2) -> str:
    """One fixed-width number for a table column: a sign column, ``digits``
    places before the point (a blank for a leading zero), ``decimals`` after:
    ``+14.80``, ``- 7.40``, ``+ 0.25``.  A value too large runs wider."""
    value = float(value)
    sign = "-" if value < 0 else "+"
    return sign + f"{abs(value):{digits + 1 + decimals}.{decimals}f}"


def kv_lines(rows: Sequence[tuple[str, str]], *, indent: int = 2,
             width: int = LABEL) -> list[str]:
    """``label value`` rows, the label padded to ``width`` after ``indent`` spaces."""
    pad = " " * indent
    return [f"{pad}{str(label):<{width - 1}} {value}".rstrip()
            for label, value in rows]


def table_lines(header: Sequence[str], rows: Sequence[Sequence[str]], *,
                widths: Sequence[int], align: str, indent: int = 2,
                legend: Sequence[tuple[str, str]] = ()) -> list[str]:
    """A fixed-column table: one space between columns, the last column unpadded,
    the header row underlined (one dash run per column width).

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

    cells = [str(cell) for cell in header]
    rule_widths = [max(width, len(cell)) if i < last else len(cell)
                   for i, (cell, width) in enumerate(zip(cells, widths))]
    rule = (" " * indent + " ".join("-" * w for w in rule_widths)).rstrip()
    return ([line(header), rule] + [line(row) for row in rows]
            + kv_lines(legend, indent=indent + 2))


def right_align(left: str, right: str, width: int = WIDTH) -> str:
    """``left`` with ``right`` ending at column ``width`` (two spaces at least)."""
    if not right:
        return left
    gap = width - len(left) - len(right)
    return left + " " * max(gap, 2) + right


# A wall time (``78.5 s``, ``18 min 39 s``, ``1 h 02 min``) and a manual stage's
# banner and ``=>`` line are formatted in ``common.timing``, which prints them and
# sits below this module (tests/test_layering.py); they are the same functions.
from common.timing import fmt_wall, stage_banner, stage_result  # noqa: E402


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
    started: datetime                 # tz-aware
    job: str | None = None            # "Slurm job 60021274" or "Slurm job 59988848, step 6"
    logs: tuple[str, ...] = ()        # the main log, relative to the run directory
    version: str | None = None        # printed only when given


def title_lines(driver: str, purpose: str) -> list[str]:
    """The title between two rules: ``LORRAX <driver> · <purpose>``."""
    title = f"LORRAX {driver}" + (f" · {purpose}" if purpose else "")
    return [RULE, title, RULE]


def header_lines(run: RunFacts) -> list[str]:
    """The title, then ``Started`` right under it (as BerkeleyGW does), the
    version, the job and the logs, one per line, labels flush left.  No
    command, directory or host row: the directory is stated once, at the
    files table."""
    rows = [("Started", fmt_clock(run.started))]
    if run.version:
        rows.append(("Version", run.version))
    if run.job:
        rows.append(("Job", run.job))
    if run.logs:
        rows.append(("Logs", " · ".join(run.logs)))
    return title_lines(run.driver, run.purpose) + kv_lines(rows, indent=0)


# ── devices ──────────────────────────────────────────────────────────────────
def devices_lines(facts: Mapping, *, mesh_role: str | None,
                  extras: Sequence[tuple[str, str]] = ()) -> list[str]:
    """The Devices block's rows from the runtime facts (``runtime.facts``).

    ``facts`` may carry ``n_nodes``; without it the rank row gives the count
    alone.  ``mesh_role`` names what the two mesh axes hold; ``None`` omits the
    mesh row, which says how many GPUs the mesh leaves unused when it does not
    fill them.  The card's memory is in the GPUs row; ``memory budget`` is the
    one memory row.  FP64 is always on (``runtime.enforce_x64``), so the JAX
    row does not state it.  ``extras`` are a driver's own ``(label, value)``
    rows, printed last; math kernels are their own block
    (:func:`math_kernel_lines`).
    """
    p = int(facts.get("process_count", 1))
    nodes = facts.get("n_nodes")
    ranks = str(p) + (f" on {int(nodes)} node{'' if int(nodes) == 1 else 's'}"
                      if nodes else "")
    ndev = int(facts.get("n_devices", 0))
    nlocal = int(facts.get("n_local_devices", 0))
    kind = str(facts.get("device_kind", "unknown"))
    backend = str(facts.get("backend", "unknown")).lower()
    gpu = backend in ("gpu", "cuda", "rocm")
    per_rank = f", {nlocal} per rank" if nlocal else ""
    memory = facts.get("device_memory") or {}
    card = f" ({memory['total'] / 1e9:.2f} GB)" if memory.get("total") else ""
    if gpu:
        devices = ("GPUs", f"{ndev} x {kind}{card}{per_rank}")
    else:
        devices = ("GPUs", f"none; {ndev} {backend.upper()} devices{per_rank}")
    rows = [("MPI ranks", ranks), devices]
    if mesh_role is not None:
        mesh = tuple(int(v) for v in facts.get("mesh_shape", (1, 1)))
        unused = ndev - mesh[0] * mesh[1]
        noun = "GPU" if gpu else "device"
        rows.append(("mesh", f"{mesh[0]} x {mesh[1]} ({mesh_role})" + (
            f"; {unused} {noun}{'' if unused == 1 else 's'} unused" if unused > 0 else "")))
    threads = facts.get("threads") or {}
    affinity = threads.get("affinity")
    # A runtime note names the library it applied the count to and why; the
    # row keeps the count (``OpenBLAS 6 threads``).
    named = []
    for key, short in _THREAD_NAMES:
        value = threads.get(key)
        if value in (None, ""):
            continue
        count = re.match(r"\s*(\d+)", str(value))
        plain = re.sub(r";\s*applied to [^)]*", "", str(value))
        named.append(f"{short} {count.group(1)} threads" if count else f"{short} {plain}")
    # Physical cores and logical CPUs separately: under SMT a mask holding both
    # hardware threads of 16 cores is 32 logical CPUs, and OpenMP and XLA's CPU
    # pool size themselves from the logical count.
    physical = threads.get("physical_cores")
    if affinity is None:
        cpu = "unknown CPUs per rank"
    elif physical is None:
        cpu = f"{affinity} logical CPUs per rank (physical cores unknown: no core topology)"
    else:
        cpu = f"{physical} physical cores / {affinity} logical CPUs per rank"
    how = threads.get("rank_threads") or "threads not set by select_gpu.sh"
    rows.append(("CPU", cpu + (", " + ", ".join(named) if named else "") + f"; {how}"))
    lines = kv_lines(rows)
    if memory.get("total"):
        lines += kv_lines((("memory budget", f"{memory['budget'] / 1e9:.2f} GB per device"),))
        warning = device_memory_message(memory)
        if warning is not None:
            lines += message_lines(warning, indent=2)
    collectives = ("NCCL" if gpu else
                   str((facts.get("collectives") or {}).get("impl", "local")).upper())
    on = (f" on {ndev} GPU{'' if ndev == 1 else 's'}" if gpu else
          f" on {ndev} {backend.upper()} device{'' if ndev == 1 else 's'}")
    lines += kv_lines((("JAX", f"{facts.get('jax_version', 'unknown')} / jaxlib "
                               f"{facts.get('jaxlib_version', 'unknown')}{on}, "
                               f"{collectives} collectives"),
                       *extras))
    return lines


def math_kernel_lines(rows: Sequence[tuple[str, str]]) -> list[str]:
    """The MATH KERNELS block's rows: which engine runs each kind of kernel
    (dense linear algebra, k-convolutions, GEMMs), as ``(label, value)``."""
    return kv_lines(rows)


def memory_budget_message(peak_gb: float, budget_gb: float, *, where: str
                          ) -> "Message | None":
    """``WARNING [device_memory_near_budget]`` when a device peak reaches 90 %
    of the per-device memory budget, else ``None``."""
    if not budget_gb or peak_gb < 0.9 * float(budget_gb):
        return None
    return Message(
        "WARNING", "device_memory_near_budget",
        f"device memory reached {100.0 * peak_gb / budget_gb:.0f} % of the budget",
        found=("device peak", f"{peak_gb:.2f} GB in {where}"),
        needs=f"below 90 % of {budget_gb:.2f} GB per device",
        fix="raise memory_per_device_gb toward the card, or run on more GPUs")


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


def message_detail(msg: Message) -> str | None:
    """A message's first detail line, as :func:`message_lines` prints it first
    under the head (unwrapped): the ``found / needs`` line, else its first free
    line, else ``fix:``; ``None`` for a message of one line."""
    parts = ([f"{msg.found[0]}: {msg.found[1]}"] if msg.found else []) + (
        [f"this run needs: {msg.needs}"] if msg.needs else [])
    if parts:
        joined = "    ".join(parts)
        return joined if TEXT_COL + len(joined) <= WIDTH else parts[0]
    if msg.lines:
        return str(msg.lines[0])
    return f"fix: {msg.fix}" if msg.fix else None


def message_repeat_lines(msg: Message, count: int, *, indent: int = 0,
                         detail: str | None = None) -> list[str]:
    """A message whose id was printed in full before: ``<text> again (<count>)``
    with the ``[id]`` ending at column 100, then ``detail`` (wrapped, never cut)
    when given."""
    tag = f"[{msg.id}]"
    column = indent + TEXT_COL
    head = wrap_value(f"{msg.text} again ({int(count)})", first_col=column,
                      width=WIDTH - len(tag) - 2)
    lines = [right_align(" " * indent + f"{msg.kind:<{TEXT_COL}}" + head[0], tag)]
    lines += head[1:]
    if detail:
        lines += _indented(column, detail)
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
    it happened.  A text is never cut: it wraps, each continuation starting
    under the text column.  With nothing to report the block is the line
    ``WARNINGS  none``.
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
        wrapped = wrap_value(" ".join(str(msg.text).split()), first_col=len(left))
        lines.append((left + wrapped[0]).rstrip())
        lines += wrapped[1:]
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
        holds = bool(receipt.trs_holds)
        conclusive = (bool(receipt.conclusive) if method == "occupied-density-subspace"
                      else True)
        if holds:
            value = ("used" if allowed else "not used") + (
                "" if conclusive else " (the density check is inconclusive)")
        else:
            # One line: the k / -k fact is the row above, and the density
            # check's figures are DEBUG plus the ``symmetry.trs_check`` record
            # (:func:`time_reversal_check`, printed by the report).
            value = "not a symmetry alone: " + (
                f"used only inside the {n_anti} t_rev = 1 operations" if n_anti
                else "k and -k are sampled apart")
            qe = str(getattr(receipt, "trs_basis", "")).startswith("qe")
            note = Message(
                "NOTE", "time_reversal_broken",
                (f"QE marks {n_anti} of the {n_sym} operations t_rev = 1 (magnetic)"
                 if n_anti else "the DFT reference breaks time reversal "
                 + ("(magnetic QE run)" if qe else "(magnetic)")))
    lines = kv_lines([("time reversal", value)])
    if consequence:
        lines += _indented(LABEL + 2, consequence)
    return lines, note


def time_reversal_check(receipt) -> tuple[str, dict] | None:
    """The density check behind the ``time reversal`` row, as one line and its
    fields (the report's DEBUG line and ``symmetry.trs_check`` record), or
    ``None`` when no 2-component check was retained."""
    if receipt is None:
        return None
    method = getattr(receipt, "method", "real-space-density")
    if method == "occupied-density-subspace" and receipt.trs_basis == "not-2c":
        return None
    coverage = 100.0 * float(receipt.trs_coverage)
    fields = {"method": method, "basis": str(receipt.trs_basis),
              "holds": bool(receipt.trs_holds), "coverage_percent": coverage}
    if method == "occupied-density-subspace":
        residual = ("n/a" if receipt.subspace_residual is None
                    else f"{float(receipt.subspace_residual):.1e}")
        evidence = ", ".join(f"{kind} {int(count)}"
                             for kind, count in receipt.evidence_counts)
        fields.update(conclusive=bool(receipt.conclusive), residual=(
            None if receipt.subspace_residual is None else float(receipt.subspace_residual)),
            evidence={str(kind): int(count) for kind, count in receipt.evidence_counts})
        text = (f"2-component density check: basis {receipt.trs_basis}, residual "
                f"{residual}, coverage {coverage:.2f}%, "
                + (evidence or "no independent evidence"))
    else:
        m_rel = "n/a" if receipt.m_rel is None else f"{float(receipt.m_rel):.1e}"
        fields.update(m_rel=None if receipt.m_rel is None else float(receipt.m_rel))
        text = (f"density check: basis {receipt.trs_basis}, |m|/|rho| {m_rel}, "
                f"coverage {coverage:.2f}%")
    return text, fields


def symmetry_lines(sym, *, centroids=None, tr_lines: Sequence[str] = ()) -> list[str]:
    """The SYMMETRY block's rows and the table of every operation the run
    uses (an operation the file lists and the run does not use is counted in
    the ``QE types`` row, not tabulated).

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
    lines = kv_lines(rows) + list(tr_lines) + [""]
    header = (f"  {'op':<5} {'QE-TR':<6} {'used as':<9} {'R^-1':<32} "
              f"{'tau':<8} (r' = R^-1 r + tau)")
    lines += [header, underline(header)]
    shown = _clean(tau, TAU_DIGITS)
    for i, (rotation, shift) in enumerate(zip(rotations, shown)):
        if used[i] == "unused":         # the table lists the operations the run uses
            continue
        flag = "-" if qe_tr is None else ("yes" if bool(qe_tr[i]) else "no")
        lines.append(f"  S{i + 1:02d}   {flag:<6} {used[i]:<9} {_matrix_text(rotation):<32} "
                     f"({shift[0]: .{TAU_DIGITS}f} {shift[1]: .{TAU_DIGITS}f} "
                     f"{shift[2]: .{TAU_DIGITS}f})")
    return lines


def kpoint_lines(wfn, sym, *, extra: Mapping[str, Sequence] = {}) -> list[str]:
    """The K POINTS block: the grid, then every stored k point in crystal
    coordinates (no cap).  ``extra`` adds columns (``{"ngk": [...]}``)."""
    grid = tuple(int(v) for v in np.asarray(wfn.kgrid).reshape(-1)[:3])
    shift = _clean(np.asarray(wfn.shift, dtype=np.float64).reshape(-1)[:3], K_DIGITS)
    shift_text = ("unshifted" if not np.any(shift) else
                  "shift (" + ", ".join(f"{v:.3f}" for v in shift) + ")")
    n_full, n_red = int(sym.nk_tot), int(sym.nk_red)
    head = f"{grid[0]} x {grid[1]} x {grid[2]}, {shift_text}: "
    lines = kv_lines([("k grid", head + (
        f"{n_full} k, all stored (no IBZ reduction)" if n_red == n_full
        else f"{n_full} k in the BZ, {n_red} in the IBZ"))]) + [""]
    weights = np.asarray(wfn.kweights, dtype=np.float64)
    if float(np.sum(weights)) != 0.0:
        weights = weights / float(np.sum(weights))
    points = _clean(wfn.kpoints, K_DIGITS)
    weights = _clean(weights, K_DIGITS)
    columns = {name: [str(v) for v in values] for name, values in extra.items()}
    widths = {name: max([len(name)] + [len(v) for v in values]) + 3
              for name, values in columns.items()}
    header = (f"{'ik':>7}{'kx':>11}{'ky':>11}{'kz':>11}{'weight':>11}"
              + "".join(f"{name:>{widths[name]}}" for name in columns))
    lines += [header, underline(header)]
    for i in range(len(points)):
        p = points[i]
        lines.append(f"{i + 1:>7}{p[0]:>11.{K_DIGITS}f}{p[1]:>11.{K_DIGITS}f}"
                     f"{p[2]:>11.{K_DIGITS}f}{weights[i]:>11.{K_DIGITS}f}"
                     + "".join(f"{columns[name][i]:>{widths[name]}}" for name in columns))
    return lines


# ── stages ───────────────────────────────────────────────────────────────────
#: Indent of a stage's detail rows, and the width of their label column.
DETAIL_INDENT = 6
DETAIL_LABEL = 15


def stage_detail(label: str | None, value: str) -> list[str]:
    """A detail row under the current stage: indent 6, ``label`` padded to 15,
    the value at column 21 and wrapped at 100.  Without a label the value
    starts at the indent; each of its own lines is one row.  With a label and a
    value of several lines (a small table under its key) the first line follows
    the label and every further line is kept verbatim at column 21."""
    pad = " " * DETAIL_INDENT
    if label is None:
        return [(pad + line).rstrip() for line in str(value).splitlines() or [""]]
    column = DETAIL_INDENT + DETAIL_LABEL
    if "\n" in str(value):
        first, *rest = str(value).splitlines() or [""]
        return ([f"{pad}{str(label):<{DETAIL_LABEL - 1}} {first}".rstrip()]
                + [(" " * column + line).rstrip() for line in rest])
    wrapped = wrap_value(value, first_col=column)
    return [f"{pad}{str(label):<{DETAIL_LABEL - 1}} {wrapped[0]}".rstrip()] + wrapped[1:]


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

    header = (f"  {'stage':<{width}}{'wall (s)':>11}{'fraction':>11}"
              + (f"{'device GB':>12}" if peaks else ""))
    lines = [header, underline(header)]
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
    header = f"  {'key':<24} {'value':<{value_width}} {'source':<8} effect"
    lines = [header, underline(header)]
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


def flag_lines(rows: Sequence[tuple[str, str]]) -> list[str]:
    """The INPUT FLAGS block: a plain ``key value`` echo of what the input
    set, in its order; no source, no effect (the METHOD block says what the
    run does)."""
    width = max([24] + [len(str(key)) + 1 for key, _ in rows])
    return kv_lines([(str(key), str(value)) for key, value in rows], width=width + 1)


# ── outputs ──────────────────────────────────────────────────────────────────
def output_file_lines(rows: Sequence[tuple[str, str, str]], *, directory: str,
                      next_step: str | None = None) -> list[str]:
    """The files table: ``role / state / path``, inputs first, paths relative
    to ``directory``, which the line above the table names (the one place the
    log states the run directory).

    ``rows`` are ``(role, state, path)``.  A row whose state is in
    :data:`INPUT_STATES` is an input and prints before the outputs; the order is
    otherwise the caller's.  The caller leaves out the log itself.
    """
    ordered = sorted(rows, key=lambda row: row[1] not in INPUT_STATES)
    width = min(max([ROLE_MIN] + [len(str(role)) for role, _, _ in ordered]), ROLE_MAX)
    state_width = max([11] + [len(str(state)) + 1 for _, state, _ in ordered])
    lines = [f"  paths relative to {os.path.abspath(os.fspath(directory))}", ""]
    lines += table_lines(
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
    "fixed",
    "flag_lines",
    "math_kernel_lines",
    "memory_budget_message",
    "underline",
    "message_detail",
    "message_from_exception",
    "message_lines",
    "message_repeat_lines",
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
    "stage_banner",
    "stage_detail",
    "stage_result",
    "symmetry_lines",
    "time_reversal_check",
    "time_reversal_lines",
    "timing_lines",
    "right_align",
    "table_lines",
    "title_lines",
    "wrap_value",
]
