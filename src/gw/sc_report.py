"""The self-consistent loop in the main log: map rows, the convergence table,
and the run summary the RESULTS block reads.

The SC driver (:mod:`gw.sc_iteration`) feeds a :class:`SCRun` with facts it
already holds (the verdict, the map's stage seconds, what moved); this module
turns them into text and logs it.  Every text function is pure (values in,
lines out) and golden-tested in ``tests/test_sc_report_cpu.py``.

Main log: the loop's row, then one block per map (stage detail rows, indent
6, key padded to 15)::

      SC loop        anderson mixing (history 20), at most 2 maps, stops at max|dE| < 3 eV
      SC map 1       input Anderson iterate; rebuilds W, Sigma and V_H
      ... the map's notes, W model row and progress bars ...
      max|dE|        1.914016 eV at k=6, band 4 (cutoff 3.000000 eV): converged
      RMS |dE|       0.750533 eV over 5 bands
      gap            0.818789 eV
      changes        Sigma rules: 3 of 10 windows rebuilt; W pole ceiling re-planned
      wall           51.3 s (W response 47.1, Sigma 3.1, other 1.1)
    SC iteration: call=0001 role=accepted_input_map gap=... max|dE|=... (verbatim)

The ``SC iteration:`` line is the harness's (``tests/hsuite/chain.py`` reads
every ``max|dE|=`` of it from the main log); the driver prints it unchanged
after these rows.  At the loop's end :func:`convergence_lines` gives::

      SC convergence max|dE| over 5 non-scissored bands, cutoff 3.000000 eV
        map  input           max|dE| eV  at k, band      RMS eV      gap eV   wall s
          0  DFT states        3.892961        6, 4    1.482950    0.474035     88.1
          1  Anderson          1.914016        6, 4    0.750533    0.818789     51.3
      verdict        CONVERGED after 2 maps (max|dE| 1.914016 eV < 3.000000 eV)

Every map's facts also go to the JSON sidecar under ``sc.maps``, the loop's
settings under ``sc.loop`` and the verdict under ``sc.verdict``.

For the report (slice A's RESULTS): :func:`last_run` returns the latest
:class:`SCRun` of this process, or None when no SC loop ran (``run_sc_driver``
opens a new one, so read it only on a ``qp_solver = self_consistent`` run:
a process that runs several drivers keeps the last loop).  It carries the
per-map rows (:attr:`SCRun.maps`, :attr:`SCRun.final`), the verdict
(:attr:`SCRun.converged`, :attr:`SCRun.verdict`) and the eqp files the driver
wrote (:attr:`SCRun.eqp_files`); the final QP energies themselves are
``SCDriverResult.qp_energies_ry`` (the same numbers as those files).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from common.log import block, get_logger, record

log = get_logger(__name__)

__all__ = [
    "MapFacts", "SCRun", "convergence_lines", "emit_row", "emit_rows",
    "input_label", "last_run", "loop_value", "map_begin_value",
    "map_result_rows", "plain_reason", "record_map", "sc_stage_verdict", "start_run",
    "verdict_text",
]

#: The block name of the SC rows (``common.log.block``).
BLOCK = "sc"

#: What the input of a map is, by the driver's role / seed vocabulary.
_INPUT = {"dft_seed": "DFT states", "external_qp_seed": "QP seed",
          "initial": "DFT states", "one_shot": "DFT states",
          "accepted_input_map": "Anderson iterate"}


@dataclass
class MapFacts:
    """What one SC map did, as the main log and the sidecar report it."""

    call: int
    input: str = ""
    rebuilt: tuple = ()
    role: str = ""
    max_abs_ev: float | None = None
    worst_k: int | None = None
    worst_band: int | None = None
    cutoff_ev: float | None = None
    rms_ev: float | None = None
    n_bands: int | None = None
    converged: bool | None = None
    stalled: bool = False
    gap_ev: float | None = None
    metal: bool = False
    wall_s: float | None = None
    #: Seconds by stage label (``W response``, ``Sigma rule plan``, ...).
    stage_s: dict = field(default_factory=dict)
    #: What moved this map, by key (a later note of a key replaces it).
    changes: dict = field(default_factory=dict)
    matrix_residual_mev: float | None = None


@dataclass
class SCRun:
    """One SC loop of this process: settings, maps in call order, verdict."""

    loop: dict = field(default_factory=dict)
    maps: list = field(default_factory=list)
    converged: bool | None = None
    verdict: str = ""
    eqp_files: tuple = ()

    @property
    def final(self) -> MapFacts | None:
        """The last map with a verdict (the accepted map), or None."""
        done = [m for m in self.maps if m.max_abs_ev is not None]
        return done[-1] if done else None

    def map(self, call: int) -> MapFacts:
        """The facts of map ``call``, created on first use."""
        for facts in self.maps:
            if facts.call == int(call):
                return facts
        facts = MapFacts(call=int(call))
        self.maps.append(facts)
        return facts

    def current(self) -> MapFacts | None:
        """The map being evaluated (the last one begun)."""
        return self.maps[-1] if self.maps else None

    def change(self, call: int | None, key: str, text: str) -> None:
        """Note what moved on map ``call`` (the current map when None)."""
        facts = self.current() if call is None else self.map(call)
        if facts is not None:
            facts.changes[str(key)] = str(text)

    def convergence_lines(self) -> list[tuple[str | None, str]]:
        """:func:`convergence_lines` of this run."""
        return convergence_lines(self)


_RUNS: list[SCRun] = []


def start_run(**loop) -> SCRun:
    """Open the record of a new SC loop; :func:`last_run` returns it from now on."""
    run = SCRun(loop=dict(loop))
    _RUNS[:] = [run]
    return run


def last_run() -> SCRun | None:
    """The latest SC loop of this process (None when no SC loop ran)."""
    return _RUNS[-1] if _RUNS else None


# ── text ─────────────────────────────────────────────────────────────────────
def input_label(role: str) -> str:
    """What a map's input is, from the driver's role or seed name."""
    return _INPUT.get(str(role), str(role))


def _ev(value) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _joined(words) -> str:
    words = [str(w) for w in words]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def loop_value(*, mode: str, max_iter: int, tol_ev: float, accelerator: str,
               depth: int) -> str:
    """The ``SC loop`` row: how maps are mixed and when the loop stops
    (``mode`` is the SETTINGS block's ``compute_mode``; the sidecar keeps it)."""
    if int(max_iter) <= 1:
        return f"one map, no mixing; max|dE| reported against {float(tol_ev):g} eV"
    return (f"{accelerator} mixing (history {int(depth)}), at most {int(max_iter)} maps, "
            f"stops at max|dE| < {float(tol_ev):g} eV")


def map_begin_value(facts: MapFacts) -> str:
    """The ``SC map N`` row printed when the map starts."""
    text = f"input {facts.input or 'states'}"
    if facts.rebuilt:
        text += f"; rebuilds {_joined(facts.rebuilt)}"
    return text


def _verdict_word(facts: MapFacts) -> str:
    if facts.stalled:
        return "stalled, not converged"
    if facts.role == "one_shot":
        return "one map, no convergence test"
    return "converged" if facts.converged else "not converged"


#: How the wall row groups the map's stages: W, every Sigma part, the rest.
def _wall_value(facts: MapFacts) -> str:
    stages = dict(facts.stage_s)
    w = float(stages.pop("W response", 0.0))
    sigma = sum(float(v) for k, v in stages.items() if str(k).startswith("Sigma"))
    rest = float(facts.wall_s) - w - sigma
    from common.report_blocks import fmt_wall
    return (f"{fmt_wall(facts.wall_s)} (W response {w:.1f}, Sigma {sigma:.1f}, "
            f"other {max(rest, 0.0):.1f})")


def map_result_rows(facts: MapFacts) -> list[tuple[str, str]]:
    """The ``(label, value)`` rows closing a map, before its ``SC iteration:`` line."""
    rows = []
    if facts.max_abs_ev is not None:
        where = ("" if facts.worst_k is None else
                 f" at k={facts.worst_k}, band {facts.worst_band}")
        cutoff = ("" if facts.cutoff_ev is None else
                  f" (cutoff {_ev(facts.cutoff_ev)} eV)")
        rows.append(("max|dE|", f"{_ev(facts.max_abs_ev)} eV{where}{cutoff}: "
                                f"{_verdict_word(facts)}"))
    if facts.rms_ev is not None:
        over = "" if facts.n_bands is None else f" over {facts.n_bands} bands"
        rows.append(("RMS |dE|", f"{_ev(facts.rms_ev)} eV{over}"))
    if facts.metal:
        rows.append(("gap", "none (metal)"))
    elif facts.gap_ev is not None:
        rows.append(("gap", f"{_ev(facts.gap_ev)} eV"))
    if facts.changes:
        rows.append(("changes", "; ".join(facts.changes.values())))
    if facts.wall_s is not None:
        rows.append(("wall", _wall_value(facts)))
    return rows


def _table_row(facts: MapFacts) -> str:
    where = ("-" if facts.worst_k is None else f"{facts.worst_k}, {facts.worst_band}")
    gap = "metal" if facts.metal else _ev(facts.gap_ev)
    wall = "-" if facts.wall_s is None else f"{float(facts.wall_s):.1f}"
    inp = "Anderson" if facts.input == "Anderson iterate" else (facts.input or "-")
    return (f"  {facts.call:>3d}  {inp:<14}{_ev(facts.max_abs_ev):>12}  {where:>10}  "
            f"{_ev(facts.rms_ev):>10}  {gap:>10}  {wall:>7}")


def convergence_lines(run: SCRun) -> list[tuple[str | None, str]]:
    """The closing table as ``(label, value)`` rows; a ``None`` label is a table line.

    One table row per map with a verdict: map, input, max|dE| (the
    criterion, eV), the state attaining it, RMS |dE| (a diagnostic), the
    output gap and the map's wall.
    """
    maps = [m for m in run.maps if m.max_abs_ev is not None]
    if not maps:
        return []
    last = maps[-1]
    head = ("max|dE|" + ("" if last.n_bands is None else
                         f" over {last.n_bands} non-scissored bands")
            + ("" if last.cutoff_ev is None else f", cutoff {_ev(last.cutoff_ev)} eV"))
    rows: list[tuple[str | None, str]] = [("SC convergence", head)]
    rows.append((None, f"  {'map':>3}  {'input':<14}{'max|dE| eV':>12}  {'at k, band':>10}  "
                       f"{'RMS eV':>10}  {'gap eV':>10}  {'wall s':>7}"))
    rows += [(None, _table_row(m)) for m in maps]
    if run.verdict:
        rows.append(("verdict", run.verdict))
    return rows


def verdict_text(*, converged: bool, stalled: bool, one_map: bool, n_maps: int,
                 max_abs_ev: float, cutoff_ev: float) -> str:
    """The ``verdict`` row of the convergence table."""
    maps = f"{int(n_maps)} map{'s' if int(n_maps) != 1 else ''}"
    if one_map:
        return f"one map, convergence not requested (max|dE| {_ev(max_abs_ev)} eV)"
    if converged:
        return (f"CONVERGED after {maps} (max|dE| {_ev(max_abs_ev)} eV < "
                f"{_ev(cutoff_ev)} eV)")
    if stalled:
        return (f"NOT CONVERGED: stalled at a residual floor after {maps} "
                f"(max|dE| {_ev(max_abs_ev)} eV)")
    return (f"NOT CONVERGED after {maps} (max|dE| {_ev(max_abs_ev)} eV, "
            f"cutoff {_ev(cutoff_ev)} eV)")


def sc_stage_verdict(run: SCRun | None, *, n_fallback: int = 0) -> str:
    """The SC stage's ``=>`` result: ``CONVERGED after 3 maps, max|dE| 0.723 eV``
    (or ``NOT CONVERGED ...``), from the loop's verdict and its final map;
    ``N maps`` alone when no map carries a verdict."""
    maps = [] if run is None else [m for m in run.maps if m.max_abs_ev is not None]
    if not maps:
        n = int(n_fallback)
        return f"{n} map{'' if n == 1 else 's'}"
    n, final = len(maps), maps[-1]
    if final.role == "one_shot":
        return f"one map, no convergence test, max|dE| {float(final.max_abs_ev):.3f} eV"
    word = "CONVERGED" if run.converged else "NOT CONVERGED"
    return (f"{word} after {n} map{'' if n == 1 else 's'}, "
            f"max|dE| {float(final.max_abs_ev):.3f} eV")


_GATE = re.compile(r"^\s*GATE \w+:\s*")


def plain_reason(text: str) -> str:
    """A refusal reason without its leading ``GATE <id>:`` (the main log never says GATE)."""
    return _GATE.sub("", str(text), count=1)


# ── logging ──────────────────────────────────────────────────────────────────
def emit_row(label: str | None, value: str) -> None:
    """One SC row in the main log (and the rank log of every other rank)."""
    log.info(value, extra=block(BLOCK, label=label))


def emit_rows(rows) -> None:
    for label, value in rows:
        emit_row(label, value)


def record_map(facts: MapFacts) -> None:
    """The map's facts into the JSON sidecar, under ``sc.maps``."""
    log.debug(f"SC map {facts.call} facts", extra=record("sc.maps", **asdict(facts)))
