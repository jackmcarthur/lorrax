"""tdagw/td_log.py — the TD-aGW run log, ``tdagw.out`` beside ``tdgw.h5``.

One file per run (``outdir/tdagw.out``, also with several drives), written by
process 0 through the driver's :class:`runtime.run_session.RunSession`
report, the mechanism every LORRAX driver's log uses (``gwjax.out``,
``bse.out``): :class:`TdReport` is a
:class:`common.preprocessing_output.ScientificProductionReport` whose
``emit`` writes the log only and whose ``echo`` writes stdout and the log.
Stdout keeps exactly the lines it had; the log holds every one of them (the
one-line chunk reports as rows of its progress table, which carry the same
samples, time and wall) and, in order:

provenance      date, host, git HEAD and dirty flag, source closure, FFI bundle,
                JAX, device mesh
resolved deck   every key, its value, set / default / unset, what the parser
                resolved, keys set but not used
system          header, window, k grid, occupations, QP range and gap, dipole route
self-energy     model, terms, N_mu, affinity, the measured wall of one dSigma call
integrator      method, step, corrections, predictor, chunk; the stiff split and the
                stability gate (sigma, static limit, probes, the chosen dt, cost)
progress        one fixed-column row per chunk (:data:`PROGRESS_COLUMNS`,
                :func:`parse_progress_row`)
summary         stdout's summary plus dSigma per fs, steady s/fs, the gate's share,
                the largest invariant drifts and the files written; then the
                session's stage table, warnings, file table and completion line

Nothing here touches the device: the gate's timing comes from its own
ahead-of-time compiled power iteration, and the progress rows from the
chunk's host samples (:data:`tdagw.propagate.PROGRESS`, ``COUNTERS``).
"""
from __future__ import annotations

import datetime
import os
import re
import socket
import sys
import textwrap

import numpy as np

from common.preprocessing_output import ScientificProductionReport
from common.scientific_output import architecture_lines, numerical_environment_lines
from tdagw import propagate as tp
from tdagw import stages, td_config, tdgw_io

__all__ = ["LOG_NAME", "PROGRESS_COLUMNS", "ROW_LENGTH", "TdReport", "log_path",
           "parse_progress_row", "progress_header", "progress_row_line"]

#: The log's file name, in ``[control.files] outdir``.
LOG_NAME = "tdagw.out"

#: The progress table: (header, width, format) per column, joined by two spaces; the
#: samples column is left-aligned, every other right-aligned; a missing value is "-".
PROGRESS_COLUMNS = (
    ("samples", 15, "{}"),
    ("t (fs)", 10, "{:.4f}"),
    ("wall (s)", 10, "{:.2f}"),
    ("s/fs", 9, "{:.3f}"),
    ("ETA (s)", 9, "{:.1f}"),
    ("dSigma", 8, "{:d}"),
    ("|P| (au)", 10, "{:.3e}"),
    ("n_c (/cell)", 11, "{:.3e}"),
    ("|E| (V/m)", 10, "{:.3e}"),
    ("dTr/Tr", 8, "{:.1e}"),
    ("dPur/Pur", 8, "{:.1e}"),
)
_ROW_KEYS = ("samples", "t_fs", "wall_s", "s_per_fs", "eta_s", "dsigma", "p_abs", "n_c",
             "e_abs", "d_trace", "d_purity")
_RULE = "  " + "-" * (sum(w for _, w, _ in PROGRESS_COLUMNS) + 2 * (len(PROGRESS_COLUMNS) - 1))
_T = tdgw_io.T_IN_ATTOSEC
_EV = tdgw_io.HA_IN_EV


def log_path(deck) -> str:
    """``outdir/tdagw.out`` of a parsed deck."""
    return os.path.join(str(deck.values["control.files.outdir"]), LOG_NAME)


def _fit(value, fmt: str, width: int) -> str:
    """``fmt`` of ``value``, or a shorter scientific form when it would not fit."""
    if value is None:
        return "-"
    text = fmt.format(value)
    for alt in ("{:.2e}", "{:.1e}", "{:.0e}"):
        if len(text) <= width or fmt in ("{}", "{:d}"):
            break
        text = alt.format(float(value))
    return text


def _cells(values) -> str:
    out = []
    for (head, width, fmt), value in zip(PROGRESS_COLUMNS, values):
        text = _fit(value, fmt, width)
        out.append(f"{text:<{width}}" if head == "samples" else f"{text:>{width}}")
    return "  " + "  ".join(out)


def progress_header() -> str:
    """The progress table's column header."""
    return "  " + "  ".join(f"{h:<{w}}" if h == "samples" else f"{h:>{w}}"
                            for h, w, _ in PROGRESS_COLUMNS)


def progress_row_line(row: dict) -> str:
    """One fixed-column table row of a ``stages._progress_row`` dict."""
    samples = f"{row['i0']}-{row['i1'] - 1}"
    return _cells([samples] + [row[k] for k in _ROW_KEYS[1:]])


#: The length of every table line.
ROW_LENGTH = 2 + sum(w for _, w, _ in PROGRESS_COLUMNS) + 2 * (len(PROGRESS_COLUMNS) - 1)


def parse_progress_row(line: str) -> dict:
    """The fields of one table row, read at the fixed columns ("-" reads None).

    Strict: the row has ROW_LENGTH characters, two spaces between columns,
    the samples column flush left and every other flush right, so a shifted
    or truncated row raises ValueError instead of reading a neighbour's
    digits.
    """
    if len(line) != ROW_LENGTH or not line.startswith("  "):
        raise ValueError(f"progress row of {len(line)} characters, want {ROW_LENGTH}: {line!r}")
    out, pos = {}, 2
    for key, (head, width, fmt) in zip(_ROW_KEYS, PROGRESS_COLUMNS):
        cell = line[pos:pos + width]
        sep = line[pos + width:pos + width + 2]
        pos += width + 2
        if (cell[0] == " ") if key == "samples" else (cell[-1] == " "):
            raise ValueError(f"column {head!r} misaligned: {cell!r}")
        if sep not in ("  ", ""):
            raise ValueError(f"no two-space gap after {head!r}: {sep!r}")
        text = cell.strip()
        if key == "samples":
            out[key] = tuple(int(x) for x in text.split("-"))
        elif text == "-":
            out[key] = None
        else:
            out[key] = int(text) if fmt == "{:d}" else float(text)
    return out


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _kgrid(hdr) -> tuple:
    if hdr.get("kgrid") is not None:
        return tuple(int(x) for x in np.asarray(hdr["kgrid"]).ravel())
    k = np.remainder(np.asarray(hdr["kpts"], float), 1.0)
    return tuple(len(np.unique(np.round(k[:, i], 8))) for i in range(3))


class TdReport(ScientificProductionReport):
    """The TD-aGW run log (``RunSession`` report): ``outdir/tdagw.out``, process 0.

    ``echo`` is the run's stdout (a line also goes to the log), ``emit`` the
    log alone, so the inherited closing blocks (stage table, warnings, file
    table, completion line) stay off stdout.  Between ``hold()`` and
    ``integrator()`` the echoed lines (the stiff split, the gate) go to
    stdout at once and to the log inside the integrator section, after the
    self-energy section that needs the gate's measured dSigma wall; a
    refusal on the way writes them before its status line.
    """

    def __init__(self, path, *, runtime, debug, stdout, deck) -> None:
        super().__init__(path, runtime=runtime, debug=debug, stdout=stdout,
                         driver_name="tdagw.tdgw_jax", calculation_name="TD-aGW propagation")
        self.deck = deck
        self._held = None
        self._rows = {}
        self._pending = True              # the self-energy / integrator sections not yet written
        if self.rank == 0:
            self.begin(input_file=deck.source)
            self.emit(f"Log            : {self.path} (stdout is a subset)")
            self._provenance()
            self._resolved_deck()

    # ── streams ──────────────────────────────────────────────────────────
    def emit(self, line: str = "") -> None:
        """One line to the log only (process 0)."""
        if self.rank == 0 and self._stream is not None:
            self._stream.write(str(line) + "\n")

    def echo(self, text: str = "", *, end: str = "\n") -> None:
        """The run's stdout text, also in the log (held lines wait, see the class)."""
        if self.rank != 0:
            return
        self.stdout(str(text))
        for line in str(text).split("\n"):
            if self._held is not None:
                self._held.append(line)
            else:
                self.emit(line)

    def hold(self) -> None:
        self._held = []

    # ── sections written at open ─────────────────────────────────────────
    def _provenance(self) -> None:
        f = self.runtime.facts
        self.heading("Provenance")
        self.emit(f"Started        : {_utc()} on {socket.gethostname()} (process 0 of "
                  f"{int(f.get('process_count', 1))})")
        from runtime.source_closure import ensure_source_closure
        rec = ensure_source_closure(print_fn=lambda *_: None)
        self.emit(f"Source         : {self._revision(rec.root)}")
        self.emit(f"Source closure : {rec.mode} closure, root {rec.root}, "
                  f"{len(rec.services)} services")
        self.emit(f"                 {rec.line()}")
        lib = f.get("ffi_library") or {}
        self.emit(f"FFI bundle     : {lib.get('platform', '?')} leg: "
                  + (lib.get("provenance") or lib.get("path")
                     or f"not loaded ({lib.get('reason', 'unknown')})"))
        for line in numerical_environment_lines(self.runtime):
            self.emit(line)
        for line in architecture_lines(self.runtime, mesh_role="k axis over every device"):
            self.emit(line)
        elapsed = (f.get("elapsed") or {}).get("total")
        if elapsed is not None:
            self.emit(f"Runtime startup: {float(elapsed):.1f} s")

    @staticmethod
    def _revision(root) -> str:
        """git HEAD and dirty flag of the running tree (the release commit when installed)."""
        from symmetry_maps import qirr_generator_commit
        rev = qirr_generator_commit()
        if rev.startswith("unknown:"):
            stamp = os.path.join(str(root), "SOURCE_COMMIT")
            if os.path.isfile(stamp):
                return f"release {open(stamp).read().strip()[:12]} (not a git checkout)"
            return f"git revision {rev}"
        dirty = rev.endswith("-dirty")
        return (f"git {rev[:-6] if dirty else rev} "
                + ("DIRTY (tracked files modified)" if dirty else "clean"))

    def _resolved_deck(self) -> None:
        rows = td_config.resolved_deck_rows(self.deck)
        self.heading("Resolved deck")
        width = max(len(k) for k, *_ in rows)
        self.emit(f"  {'key':<{width}}  {'status':<7}  value   (TDGW units: as, eV, V/m)")
        for key, value, status, notes in rows:
            self.emit(f"  {key:<{width}}  {status:<7}  {value}"
                      + (f"   ({'; '.join(notes)})" if notes else ""))
        keys = [r for r in rows if r[0] in td_config.TD_DEFAULTS]
        count = {s: sum(r[2] == s for r in keys) for s in ("set", "default", "unset")}
        unused = [r[0] for r in rows if any(n.startswith(("not used", "read, not used"))
                                           for n in r[3])]
        self.emit(f"  {len(keys)} keys: {count['set']} set, {count['default']} default, "
                  f"{count['unset']} unset; set but not used: "
                  + (", ".join(unused) if unused else "none")
                  + "; unknown keys refuse at parse (td_unknown_key)")

    # ── the run header ───────────────────────────────────────────────────
    def header(self, deck, system, model, runtime) -> None:
        """stdout: ``stages.describe``; the log: the system section (+ details)."""
        rows = stages.describe_rows(deck, system, model, runtime)
        if self.rank != 0:
            return
        self.stdout("\n".join(rows.values()))
        self._rows = rows
        hdr = system.hdr
        self.heading("System")
        for name, line in rows.items():
            if name not in ("integrator", "self_energy"):
                self.emit(line)
                if name == "window":
                    for extra in self._system_lines(deck, system):
                        self.emit(extra)
        if not system.receipts:
            self.emit(f"  dipole          : the TDGW header's r_inter (itype {hdr.get('itype')}"
                      + (", intraband connection from its overlaps" if hdr.get("olp") is not None
                         else "") + ")")

    @staticmethod
    def _system_lines(deck, system) -> list:
        hdr = system.hdr
        g = _kgrid(hdr)
        out = [f"  k grid          : {g[0]} x {g[1]} x {g[2]} ({system.nk} points, full zone; "
               f"TDGW uses no symmetry)"]
        e = np.asarray(hdr["e_kn"], float) * _EV
        try:
            n_val = stages._n_val(hdr)
        except ValueError:
            return out + [f"  QP energies     : {e.min():.4f} .. {e.max():.4f} eV (occupations "
                          f"are not the leading rows; no gap)"]
        out.append(f"  occupations     : {n_val} valence + {system.nb - n_val} conduction "
                   f"bands at every k")
        vbm, cbm = e[:, n_val - 1].max(), e[:, n_val].min()
        direct = (e[:, n_val] - e[:, n_val - 1]).min()
        out.append(f"  QP energies     : {e.min():.4f} .. {e.max():.4f} eV over the window; "
                   f"VBM {vbm:.4f}, CBM {cbm:.4f} eV: gap {cbm - vbm:.4f} eV "
                   f"(direct {direct:.4f} eV)")
        return out

    # ── after the split and the gate ─────────────────────────────────────
    def integrator(self, *, deck, system, model, cfg, gate, basis, resume) -> None:
        """The self-energy and integrator sections, then the progress table's header."""
        if self.rank != 0:
            return
        held, self._held = (self._held or []), None
        v = deck.values
        self.heading("Self-energy model")
        if "self_energy" in self._rows:
            self.emit(self._rows["self_energy"])
        terms = {"both": "Hartree + screened exchange", "sex": "screened exchange only",
                 "hartree": "Hartree only"}
        line = f"  model           : {v['lorrax.model']} (lorrax.model; calc_type {v['control.calc_type']})"
        if v["lorrax.model"] == "isdf":
            line += (f", terms {v['lorrax.isdf_terms']} ({terms[v['lorrax.isdf_terms']]}), "
                     f"TDA {'on' if v['lorrax.tda'] else 'off'}")
        self.emit(line)
        n_mu = (system.model or {}).get("n_rmu") if isinstance(system.model, dict) else None
        if n_mu is None:
            m = re.search(r"N_mu = (\d+)", model.name)
            n_mu = int(m.group(1)) if m else None
        if n_mu is not None:
            self.emit(f"  N_mu            : {int(n_mu)} interpolation points")
        self.emit("  affine in rho   : " + (
            "yes: dSigma[rho] - dSigma[rho_eq] is linear in rho - rho_eq, so the gate's "
            "linearized map and the stiff split are exact" if model.linear else
            "no (the stiff split refuses; the gate linearizes at rho_eq)"))
        if gate is not None and gate["power"]["dsigma"]:
            pw = gate["power"]
            self.emit(f"  one dSigma call : {gate['dsigma_wall_s']:.4g} s (the gate's power "
                      f"iteration: {pw['dsigma']} calls in {pw['exec_s']:.3f} s; its compile, "
                      f"{pw['compile_s']:.2f} s, excluded)")
        else:
            self.emit(f"  one dSigma call : not timed (no stability gate for "
                      f"{cfg.method}); the summary bounds it by the steady step wall")
        self.heading("Integrator and stability gate")
        if "integrator" in self._rows:
            self.emit(self._rows["integrator"])
        per = tp.dsigma_per_step(cfg.method, cfg.n_corr)
        if cfg.method in ("magnus2", "cfm4"):
            self.emit(f"  predictor       : {cfg.predictor}; frame {cfg.frame}; "
                      f"{per} dSigma calls per step")
        elif cfg.method == "dp54":
            self.emit(f"  controller      : SciPy RK45, rtol {cfg.rtol:g}, atol {cfg.atol:g}, "
                      f"max_step {cfg.max_step * _T:g} as; 6 dSigma calls per try")
        else:
            self.emit(f"  stages          : {per} dSigma calls per step")
        if cfg.method != "dp54":
            self.emit(f"  step            : dt = {cfg.dt * _T:.6g} as"
                      + (" (auto, chosen by the gate)" if v["lorrax.dt"] == "auto" else "")
                      + f", dt_measure / dt = {cfg.dt_measure / cfg.dt:.6g}, "
                      f"{per / (cfg.dt * _T) * 1000.0:.1f} dSigma calls per fs")
        self.emit(f"  chunk           : {cfg.chunk} samples per jitted call (one checkpoint, "
                  f"one progress row)")
        if cfg.remainder:
            self.emit("  remainder flow  : Gamma and/or intraband term live"
                      + (" (cfm4: Yoshida triple jump, 3x the exponentials)"
                         if cfg.method == "cfm4" else ""))
        if resume is not None:
            self.emit(f"  resumed         : sample {resume['i_next']}, step {resume['step']} of "
                      f"{resume['source']}")
        for line in held:
            self.emit(line)
        if basis is not None:
            mods = basis["moduli_ev"]
            n = basis["n_s"]
            self.emit(f"  stiff basis     : n_s {n} (|lambda| {mods[n - 1]:.4g} .. {mods[0]:.4g} eV "
                      f"above the cut {basis['cut_ev']:.4g} eV, next {mods[n]:.4g} eV), "
                      f"{basis['krylov']} Arnoldi steps, invariance residual "
                      f"{basis['residual_ev']:.2e} eV; "
                      + ("reused" if basis["reused"] else
                         f"{basis['dsigma']} dSigma calls, {basis['wall_s']:.2f} s"))
        if gate is not None:
            self._gate_block(gate)
        else:
            self.emit(f"  stability gate  : none for {cfg.method} ("
                      + ("adaptive: the controller bounds the local error)" if cfg.method == "dp54"
                         else "explicit: its step is not checked against the coupling)"))
        self._pending = False
        self.heading("Progress")
        ham = v["system.depopulation"] == 0.0 and v["system.dephasing"] == 0.0
        self.emit("  one row per chunk; t = the chunk's end state; |P|, n_c and |E| at its last "
                  "sample, max over drives; dSigma = calls so far (steps and samples);")
        self.emit("  dTr/Tr, dPur/Pur = largest |X - X(t0)| / X(t0) of Tr rho and Tr rho^2 over "
                  "the chunk's samples (" + (
                      "Gamma = 0: both are invariants of the exact flow)" if ham else
                      "Gamma != 0: Tr rho^2 decays by design, Tr rho is conserved)"))
        self.emit(progress_header())
        self.emit(_RULE)

    def _gate_block(self, g) -> None:
        slow = (f", slow map L(1 - Pi) with {g['stiff']} stiff modes split off"
                if g["stiff"] else "")
        self.emit("  gate verdict    :")
        self.emit(f"    sigma         : {g['sigma_ev']:.6g} eV (spectral radius of the "
                  f"linearized dSigma commutator map{slow}; {g['power_iterations']} power "
                  f"iterations, last change {100 * g['change']:.2g} %)")
        if g["frame"] == "whole_rho":
            self.emit("    static limit  : none (whole_rho frame: the probe decides)")
        elif g["sigma_ev"] > 0.0:
            self.emit(f"    static limit  : h sigma {g['margin']:g} <= {g['y_lim']:.4g} "
                      f"({g['method']}, {g['predictor']} predictor, n_corr {g['n_corr']}) -> "
                      f"dt <= {g['dt_max_as']:.4g} as; this step h sigma = {g['y']:.4g}")
        else:
            self.emit("    static limit  : none (sigma = 0: no coupling, any step is stable)")
        for q in g["probes"]:
            ok = q["growth"] <= 1.0 + g["probe_tolerance"] and not q.get("fast_unconverged")
            self.emit(f"    probe         : dt {q['dt_as']:.4g} as -> probe step "
                      f"{q['probe_dt_as']:.4g} as ({g['probe_factor']:g} dt), growth "
                      f"{q['growth']:.6f} per step over {g['probe_steps']} steps, field off: "
                      + ("pass" if ok else "FAIL") + f" (<= {1.0 + g['probe_tolerance']:g}); "
                      f"{q['wall_s']:.2f} s"
                      + (f"; closing solve unconverged on {q['fast_unconverged']} steps"
                         if q.get("fast_unconverged") else ""))
        self.emit(f"    chosen dt     : {g['dt_as']:.6g} as ("
                  + ("auto: the largest step that passed" if g["auto"] else "the deck's")
                  + ")")
        pw = g["power"]
        self.emit(f"    gate cost     : {g['dsigma']} dSigma calls, {g['wall_s']:.2f} s "
                  f"(power iteration {pw['dsigma']} calls {pw['exec_s']:.2f} s + compile "
                  f"{pw['compile_s']:.2f} s; {len(g['probes'])} probe(s))")

    # ── per chunk ────────────────────────────────────────────────────────
    def chunk(self, line: str, row: dict) -> None:
        """stdout: the one-line chunk report; the log: its table row."""
        if self.rank != 0:
            return
        self.stdout(line)
        self.emit(progress_row_line(row))
        if self._stream is not None:
            self._stream.flush()

    # ── the end ──────────────────────────────────────────────────────────
    def summary(self, deck, stats) -> None:
        """stdout: ``stages.summarize``; the log: the summary section (+ totals)."""
        text = stages.summarize(deck, stats)
        if self.rank != 0:
            return
        self.stdout(text)
        self.emit(_RULE)
        self.heading("Summary")
        for line in text.split("\n"):
            self.emit(line)
        for line in self._summary_lines(deck, stats):
            self.emit(line)

    def _summary_lines(self, deck, stats) -> list:
        fs = max(stats["simulated_fs"], 1e-30)
        rows = stats.get("rows") or []
        prop = int(stats["dsigma"]) + int(stats["sample_dsigma"])
        gate = int(stats.get("check_dsigma", 0))
        basis = int(stats.get("stiff_dsigma", 0))
        g = stats.get("gate")
        b = stats.get("basis")
        out = [f"  dSigma per fs   : {prop / fs:.1f} in steps and samples; "
               f"{(prop + gate + basis) / fs:.1f} with the gate and the stiff basis over this "
               f"run's {fs:.4g} fs"]
        steady = [r["s_per_fs"] for r in rows[1:]]
        if steady:
            out.append(f"  steady s/fs     : {np.mean(steady):.4g} s per fs (chunks 2-{len(rows)}; "
                       f"the first, {rows[0]['wall_s']:.2f} s, carries the compile)")
        elif rows:
            out.append(f"  steady s/fs     : one chunk only ({rows[0]['s_per_fs']:.4g} s per fs, "
                       f"compile included)")
        g_wall = (g or {}).get("wall_s", 0.0) + ((b or {}).get("wall_s", 0.0))
        total_calls = prop + gate + basis
        total_wall = stats["wall"] + g_wall
        out.append(f"  gate share      : {gate + basis} of {total_calls} dSigma calls "
                   f"({100.0 * (gate + basis) / max(total_calls, 1):.1f} %), "
                   f"{g_wall:.2f} of {total_wall:.2f} s ({100.0 * g_wall / max(total_wall, 1e-30):.1f} %)"
                   + ("; no gate for this method" if g is None else "")
                   + (f"; stiff basis {basis} calls" if b is not None else ""))
        if g is not None and g["power"]["dsigma"]:
            w = g["dsigma_wall_s"]
            out.append(f"  dSigma wall     : {w:.4g} s per call (gate) x {prop} calls = "
                       f"{w * prop:.2f} s, {100.0 * w * prop / max(stats['wall'], 1e-30):.0f} % "
                       f"of the propagation wall")
        elif steady and prop:
            out.append(f"  dSigma wall     : <= {np.mean(steady) / (prop / fs):.4g} s per call "
                       f"(steady s/fs over dSigma calls per fs; not timed alone without a gate)")
        dtr = [r["d_trace"] for r in rows if r["d_trace"] is not None]
        dpu = [r["d_purity"] for r in rows if r["d_purity"] is not None]
        if dtr:
            ham = (deck.values["system.depopulation"] == 0.0
                   and deck.values["system.dephasing"] == 0.0)
            out.append(f"  invariants      : max |dTr|/Tr {max(dtr):.2e}, max |dTr rho^2|/Tr rho^2 "
                       f"{max(dpu):.2e} over {rows[-1]['i1'] - rows[0]['i0']} samples ("
                       + ("Gamma = 0: both conserved by the exact flow)" if ham else
                          "Gamma != 0: only Tr rho is conserved)"))
        last = rows[-1] if rows else None
        if last is not None and (last["p_abs"] is not None or last["n_c"] is not None):
            out.append("  final sample    : "
                       + ", ".join(t for t in (
                           None if last["p_abs"] is None else f"|P| {last['p_abs']:.4e} au",
                           None if last["n_c"] is None else f"n_c {last['n_c']:.4e} per cell",
                           None if last["e_abs"] is None else f"|E| {last['e_abs']:.4e} V/m")
                                   if t))
        return out

    def timings(self, stages_rows, *, wall: float) -> None:
        self._flush_pending()
        super().timings(stages_rows, wall=wall)

    def finish(self, *, status: str = "completed") -> None:
        self._flush_pending()
        super().finish(status=status)

    def _flush_pending(self) -> None:
        """Before the closing blocks: what a refused run had not written yet.

        The held lines of a run that stopped before ``integrator`` (the split,
        the gate's verdict), then the refusal's own message: ``RunSession``
        closes a refused run from its ``__exit__``, where the exception is the
        one being handled.
        """
        if self.rank != 0:
            return
        if self._pending:
            self._pending = False
            held, self._held = (self._held or []), None
            if held or "integrator" in self._rows:
                self.heading("Integrator and stability gate (the run stopped here)")
                for name in ("self_energy", "integrator"):
                    if name in self._rows:
                        self.emit(self._rows[name])
                for line in held:
                    self.emit(line)
        exc = sys.exc_info()[1]
        if exc is not None and not getattr(self, "_refusal_written", False):
            self._refusal_written = True
            self.heading("Refusal")
            for line in textwrap.wrap(f"{type(exc).__name__}: {exc}", 110):
                self.emit(f"  {line}")
