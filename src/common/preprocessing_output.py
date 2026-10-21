"""Shared production report for LORRAX's mean-field and BSE drivers.

This module owns presentation only.  Dipole, ``kin_ion``, BSE and the
exciton-band driver hand it facts from their canonical WFN, symmetry,
metadata and operator objects; it does not reopen files or reconstruct any
physical policy.  The stream, header, Devices, timing, warnings, files and
closing blocks are :class:`common.production_report.ProductionReport`'s.

Every block here is ``key   value`` rows at the two-space indent
(:func:`common.report_blocks.kv_lines`); implementation facts a user does not
act on (reader backend, matrix layout, writer) go to the JSON sidecar under
``environment``.  The RESULTS block is :func:`results_lines`, a pure function
of its rows.  :class:`StageSequence` opens the ``[i/N]`` manual stages of a
long driver body one after another (``common.timing.manual_stage``).
"""

from __future__ import annotations

import contextlib
from typing import Iterable, Sequence

from common.log import block, get_logger, record
from common.production_report import ProductionReport
from common.report_blocks import (LABEL, Setting, heading, kv_lines,
                                  wrap_value)

log = get_logger(__name__)


# ── pure blocks ──────────────────────────────────────────────────────────────
def kv_block_lines(rows: Sequence[tuple[str, str]], *, indent: int = 2,
                   width: int = LABEL) -> list[str]:
    """``label value`` rows; a long value wraps under itself, and a value with
    line breaks keeps each further line at the value column."""
    column = indent + width
    lines: list[str] = []
    for label, value in rows:
        first, *rest = str(value).splitlines() or [""]
        wrapped = wrap_value(first, first_col=column)
        lines += kv_lines([(label, wrapped[0])], indent=indent, width=width)
        lines += wrapped[1:]
        lines += [(" " * column + line).rstrip() for line in rest]
    return lines


def results_lines(rows: Sequence[tuple[str, str]], *,
                  table: Sequence[str] = ()) -> list[str]:
    """The RESULTS block: an optional preformatted ``table`` (lines at the
    two-space indent), a blank line, then ``label value`` rows."""
    table = list(table)
    return (heading("Results") + table + ([""] if table and rows else [])
            + kv_block_lines(rows))


def spin_text(*, nspin: int, nspinor: int, bispinor: bool = False) -> str:
    """The SYSTEM block's spin row: ``nspin 1, 2-component spinors``."""
    spinor = ("2-component spinors" if int(nspinor) == 2
              else "1-component (scalar) states")
    return (f"nspin {int(nspin)}, {spinor}"
            + (", bispinor (four-current) lift" if bispinor else ""))


def degeneracy_check_text(check) -> str:
    """The RESULTS row of ``psp.operator_checks.check_degeneracy_consistency``."""
    ry_to_mev = 13605.693122994
    if check is None:
        return "not run"
    n_manifolds = int(check.get("n_manifolds", 0))
    worst = float(check.get("max_split_ry", 0.0)) * ry_to_mev
    if check.get("clean"):
        return (f"consistent: {n_manifolds} degenerate manifolds, max split "
                f"{worst:.4f} meV")
    return (f"SPLIT: {int(check.get('n_split', 0))} of {n_manifolds} degenerate "
            f"manifolds, worst {worst:.4f} meV (see the WARNING above)")


def galerkin_rows(receipt) -> list[tuple[str, str]]:
    """The shared htransform Galerkin compression receipt as stage rows (the
    whole receipt is ``common.scientific_output.spectral_compression_lines``)."""
    if receipt.get("method") == "kmeans_nodes":
        n_nodes, n_cent = int(receipt["nodes"]), int(receipt["centroids"])
        source = ("the ISDF centroids" if n_nodes == n_cent
                  else f"k-means nodes for {n_cent} centroids")
        return [
            ("Galerkin basis", f"nodal rank {int(receipt['retained_rank'])} on "
                               f"{n_nodes} orbit-closed nodes ({int(receipt['orbits'])} "
                               f"orbits, {source}); "
                               f"{int(receipt['stacked_states'])} states fitted"),
            ("node metric", f"eps {float(receipt['metric_eps']):.2e} "
                            f"(max_k ||Phi^H W Phi - 1||_2), "
                            f"{int(receipt['zero_weight_nodes'])} zero-weight nodes"),
        ]
    metrics = receipt["compression"]
    return [
        ("Galerkin basis", f"rank {int(receipt['retained_rank'])} of numerical rank "
                           f"{int(receipt['numerical_rank'])} (ceiling "
                           f"{int(receipt['structural_ceiling'])}); retained weight "
                           f"{100.0 * float(metrics.retained_frobenius_weight):.2f}%"),
    ]


def locality_rows(receipt) -> list[tuple[str, str]]:
    """The shared htransform locality receipt as one stage row (the whole
    receipt is ``common.scientific_output.htransform_quality_lines``)."""
    shell = 100.0 * float(receipt["outer_shell_l2_fraction"])
    shell_max = 100.0 * float(receipt["outer_shell_max_over_r0"])
    return [("locality", f"outer shell {shell:.2f}% of ||f(H)_R||_F, largest "
                         f"shell {shell_max:.2f}% of R=0; row isometry "
                         f"{float(receipt['row_isometry_max']):.1e}")]


def kin_ion_result_rows(*, out_name: str, bands: str, nk_stored: int,
                        nk_full: int, soc: str, degeneracy) -> list[tuple[str, str]]:
    """The kin-ion RESULTS rows: what ``kin_ion.h5`` holds and the operator
    check (``degeneracy`` is the check's dict, ``None`` when it did not run)."""
    return [
        (out_name, f"T + V_loc + V_NL on bands {bands}, {int(nk_stored)} "
                   f"star-wedge k (the {int(nk_full)} full-BZ k are rebuilt on read)"),
        ("V_H", "not stored; gw_jax builds it live in G-space"),
        ("V_NL", soc),
        ("degeneracy test", degeneracy_check_text(degeneracy)),
    ]


def dipole_result_rows(*, out_name: str, bands: str, nk: int, velocity: str,
                       vnl: str, finite_q: str | None = None,
                       parallel_transport: str | None = None) -> list[tuple[str, str]]:
    """The dipole RESULTS rows: what ``dipole.h5`` holds, the velocity's
    source terms, and the optional finite-q and parallel-transport outcomes."""
    rows = [
        (out_name, f"q = 0 velocity <mk|v|nk> on bands {bands}, {int(nk)} k "
                   "(full BZ), with the band energies"),
        ("velocity", velocity),
        ("V_NL", vnl),
    ]
    if finite_q is not None:
        rows.append(("finite q", finite_q))
    if parallel_transport is not None:
        rows.append(("parallel gauge", parallel_transport))
    return rows


def parallel_transport_text(*, requested: bool, written: bool,
                            lines: Sequence[str] = (), dropped: str | None = None,
                            velocity_only: bool = False) -> str | None:
    """The dipole driver's parallel-transport outcome in one row value.

    ``lines`` are the validation block ``file_io.parallel_transport`` writes
    (kept whole in the JSON sidecar); its verdict and the head's relative L2
    are read back from the ``Covariant DFT velocity:`` line.  ``None`` when
    the run asked for no parallel-transport artifact.
    """
    import re
    if not requested:
        return None
    if not written:
        reason = str(dropped or "the link stage refused")
        if re.fullmatch(r"\w+", reason):
            return f"parallel_transport.h5 not written [{reason}]; see the NOTE above"
        return f"parallel_transport.h5 not written: {reason}"
    if velocity_only:
        return "parallel_transport.h5 written: DFT velocity only, no links"
    for line in lines:
        if line.startswith("Covariant DFT velocity: "):
            body = line[len("Covariant DFT velocity: "):]
            verdict = body.split(";")[0].strip()
            match = re.search(r"Fermi-surface diagonal\)=([0-9.eE+-]+)", body)
            return ("parallel_transport.h5 written; covariant velocity " + verdict
                    + (f", head relative L2 {match.group(1)}" if match else ""))
    return "parallel_transport.h5 written"


# ── stages ───────────────────────────────────────────────────────────────────
class _PlainStage:
    """Stand-in for ``timing.ManualStage`` over a plain timing section."""

    def __init__(self, section) -> None:
        self._section = section

    def result(self, text: str) -> None:
        pass

    def watch(self, *values) -> None:
        if hasattr(self._section, "watch"):
            self._section.watch(*values)


@contextlib.contextmanager
def _plain_stage(name: str):
    from common import timing
    with timing.section(name, announce=True) as section:
        yield _PlainStage(section)


def manual_stage(index: int, total: int, title: str, *, name: str | None = None):
    """``common.timing.manual_stage`` (the ``[i/N]`` banner and its ``=>``
    line); a plain announced timing section of the same name where that API
    is absent, so the TIMING table keeps its rows either way."""
    from common import timing
    manual = getattr(timing, "manual_stage", None)
    if manual is None:
        return _plain_stage(str(name or title))
    return manual(int(index), int(total), str(title), name=name)


class StageSequence(contextlib.AbstractContextManager):
    """Consecutive ``[i/N]`` manual stages opened in one long driver body.

    ``begin`` closes the open stage (its ``=>`` line) and opens the next; the
    sequence is entered beside the driver's ``RunSession`` (``with
    RunSession(...) as run, StageSequence(3) as stages:``), so an exception
    leaves the open stage through ``common.timing``'s own exit path and the
    driver body keeps its indentation.  Print-only: a stage is a timing
    section named ``name`` around code that runs unchanged.
    """

    def __init__(self, total: int) -> None:
        self.total = int(total)
        self._cm = None
        self.current = None

    def begin(self, index: int, title: str, *, name: str | None = None,
              total: int | None = None):
        self.end()
        self._cm = manual_stage(index, total or self.total, title, name=name)
        self.current = self._cm.__enter__()
        return self.current

    def result(self, text: str) -> None:
        """The open stage's ``=>`` text (the last call wins)."""
        if self.current is not None:
            self.current.result(str(text))

    def end(self) -> None:
        cm, self._cm, self.current = self._cm, None, None
        if cm is not None:
            cm.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, tb) -> bool:
        cm, self._cm, self.current = self._cm, None, None
        if cm is not None:
            cm.__exit__(exc_type, exc, tb)
        return False


def stage_row(label: str | None, value) -> None:
    """One ``key   value`` row under the current stage (main log and rank log)."""
    log.info(str(value), extra=block("detail", label=label))


class CapturedBlock:
    """A ``report`` stand-in that keeps a component's ``heading``/``emit``
    lines for the JSON sidecar instead of the main log."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def heading(self, title: str) -> None:
        self.lines.append(str(title).upper())

    def emit(self, line: str = "") -> None:
        self.lines.append(str(line))

    def progress(self, line: str) -> None:
        self.lines.append(str(line))


# ── the report ───────────────────────────────────────────────────────────────
class ScientificProductionReport(ProductionReport):
    """One rank-zero scientific stream for a LORRAX core driver."""

    def environment(self, *, wfn, lines: Iterable[str] = ()) -> None:
        """Implementation facts (reader, layouts, writer): JSON sidecar only."""
        log.debug("numerical environment", extra=record(
            "environment",
            wavefunction_reader=str(getattr(wfn, "backend", "unknown")),
            lines=[" ".join(str(line).split()) for line in lines]))

    def pseudopotentials(self, lines: Iterable[str]) -> None:
        """A PSEUDOPOTENTIALS block of preformatted lines
        (``psp.pseudos.pseudo_summary_lines``); the drivers here put the rows
        in :meth:`system` instead."""
        self.heading("Pseudopotentials")
        self.emit_lines(str(line) for line in lines)

    def system(self, *, natoms: int, species: Iterable[str], fft_grid,
               pseudopotentials: Sequence[tuple[str, str]] = (),
               rows: Sequence[tuple[str, str]] = ()) -> None:
        """SYSTEM: atoms and species, one pseudopotential row per species
        (``psp.pseudos.pseudo_summary_rows``), the FFT grid, then ``rows``."""
        self.heading("System")
        self.emit_lines(system_lines(
            natoms=natoms, species=species, fft_grid=fft_grid,
            pseudopotentials=pseudopotentials, rows=rows))

    def bands(self, rows: Sequence[tuple[str, str]]) -> None:
        """BAND WINDOWS: ``(label, value)`` rows."""
        self.heading("Band windows")
        self.emit_lines(kv_block_lines(rows))

    def results(self, rows: Sequence[tuple[str, str]], *,
                table: Sequence[str] = ()) -> None:
        """The RESULTS block (:func:`results_lines`)."""
        self.emit_lines(results_lines(rows, table=table))


def system_lines(*, natoms: int, species: Iterable[str], fft_grid,
                 pseudopotentials: Sequence[tuple[str, str]] = (),
                 rows: Sequence[tuple[str, str]] = ()) -> list[str]:
    """The SYSTEM block's rows (pure)."""
    grid = tuple(int(value) for value in fft_grid)
    names = ", ".join(str(v) for v in species) or "none"
    pseudo_rows = [("pseudopotential" if len(pseudopotentials) == 1
                    else f"UPF {element}", text)
                   for element, text in pseudopotentials]
    return kv_block_lines(
        [("atoms", f"{int(natoms)} ({names})")] + pseudo_rows
        + [("FFT grid", f"{grid[0]} x {grid[1]} x {grid[2]}")] + list(rows))


PreprocessingProductionReport = ScientificProductionReport


def soc_setting(resolved: str) -> Setting:
    """The SETTINGS row of the V_NL spin-orbit treatment, from the projector
    setup's own statement (``j-RESOLVED, from QE <spinorbit>``)."""
    text = " ".join(str(resolved).split())
    head, _, rest = text.partition(", ")
    value = head.split(" (")[0].strip().lower() or "none"
    return Setting("spin-orbit V_NL", value, "QE" if "QE" in text else "derived",
                   rest or text)


def soc_text(resolved: str) -> str:
    """The projector setup's spin-orbit statement as a row value:
    ``j-resolved, from QE <spinorbit>``."""
    row = soc_setting(resolved)
    return f"{row.value}, {row.effect}" if row.effect != row.value else row.value


__all__ = [
    "CapturedBlock",
    "PreprocessingProductionReport",
    "ScientificProductionReport",
    "StageSequence",
    "degeneracy_check_text",
    "dipole_result_rows",
    "galerkin_rows",
    "kin_ion_result_rows",
    "locality_rows",
    "kv_block_lines",
    "manual_stage",
    "parallel_transport_text",
    "results_lines",
    "soc_setting",
    "soc_text",
    "spin_text",
    "stage_row",
    "system_lines",
]
