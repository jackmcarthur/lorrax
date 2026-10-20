"""Rank-zero, human-readable GW calculation report.

This is the scientific output of ``gwjax``.  It intentionally does not echo
library inventories, per-rank messages, HDF5 implementation details or the
driver's historical diagnostic prose.  Those remain available behind the one
driver-wide ``LORRAX_DEBUG_PRINT`` switch and in the launcher's own log.
"""

from __future__ import annotations

import os

import numpy as np

from common.production_report import ProductionReport
from common.report_blocks import (
    LABEL, Message, Setting, TimingRow, kv_lines, timing_lines, wrap_value)
from common.scientific_output import band_range
from common.units import RYD_TO_EV
from . import quadrature_log
from .gw_config import qp_solver_semantics, sigma_requested_edges_ev


# Stable role labels for the driver's closing file table, at most about 40
# columns so the table stays aligned.  They name the energy definition (the
# diagonal on the fixed DFT states) rather than a method inferred from ``eqpN``.
EQP0_FILE_ROLE = "QP energies, zeroth order (eqp0)"
EQP1_FILE_ROLE = "QP energies, linearized (eqp1)"
QP_ROTATIONS_FILE_ROLE = "QP rotations, full matrix"
QP_WFN_FILE_ROLE = "QP wavefunctions, full matrix"



def _human_bytes(n_bytes: float) -> str:
    """GiB above one GiB, MiB below, three significant figures either way."""
    gib = float(n_bytes) / 2**30
    if gib >= 1.0:
        return f"{gib:.3g} GiB"
    return f"{float(n_bytes) / 2**20:.3g} MiB"

#: One-line effects of the GW settings (BerkeleyGW style), by resolved value.
_MODE_EFFECT = {
    "x_only": "bare exchange only: no screening",
    "cohsex": "static COHSEX: screened exchange + Coulomb hole at omega = 0",
    "gn_ppm": "Godby-Needs plasmon pole, one pole per element",
    "hl_ppm": "Hybertsen-Louie plasmon pole",
    "mpa": "full-frequency W, shared-pole (multipole) model",
}
_SOLVER_EFFECT = {
    "one_shot_dft": "Sigma at E_DFT; H(QP) diagonalized once",
    "self_consistent": "QSGW: G, W, Sigma rebuilt from each map's states",
}
_DIM_EFFECT = {
    3: "3D bulk: no Coulomb truncation",
    2: "2D slab: Coulomb truncated along z",
    1: "1D wire: Coulomb truncated in the plane",
    0: "0D: Coulomb truncated in every direction",
}
_HEAD_EFFECT = {
    "full": "q->0 head with local fields",
    "no_local_fields": "q->0 head without local fields",
    "off": "no q->0 cell term",
}
_SC_HEAD_EFFECT = {
    "parallel_transport": "head velocity with Sigma from the links",
    "dft_velocity": "head on the QP-rotated DFT velocity (dipole.h5)",
    "off": "the q->0 head is not updated between maps",
}
#: Deck-resolution notes (``[config provenance]`` lines) a settings row states.
_RESOLVED_BY_ROW = ("density_self_consistent was not named",
                    "density_self_consistent = false", "sc_head_update was not named",
                    "restart was not named", "mpa_sampling_alpha =", "linalg =")


def settings_rows(config, *, n_mu: int, n_q_irr: int, processes: int,
                  budget_gb: float | None = None, device: str | None = None) -> list:
    """The GW SETTINGS rows: what the deck set, what was derived, and the defaults
    that change the result, each with its one-line effect.

    ``n_mu``, ``n_q_irr`` and ``processes`` size the ``linalg`` row (the whole
    N_mu x N_mu matrices a rank holds under ``local``); ``budget_gb`` and
    ``device`` state the per-device memory budget.
    """
    named = frozenset(getattr(config, "raw_input_keys", ()) or ())

    def source(key, *, derived=False):
        return "deck" if key in named else ("derived" if derived else "default")

    mode, solver = config.compute_mode.value, config.qp_solver.value
    rows = [
        Setting("compute_mode", mode,
                source("compute_mode", derived=config.compute_mode_raw == "auto"),
                _MODE_EFFECT.get(mode, mode)),
        Setting("qp_solver", solver,
                source("qp_solver", derived=config.qp_solver_raw == "auto"),
                _SOLVER_EFFECT.get(solver, qp_solver_semantics(solver).description)),
    ]
    if config.compute_mode.needs_screening:
        diagrams = config.screening.diagrams.value
        rows.append(Setting("screening_diagrams", diagrams, source("screening_diagrams"), {
            "w_rpa": f"RPA Dyson series, {config.screening.method} chi0 time rules",
            "w_bse": "electron-hole ladder series in W"}.get(diagrams, diagrams)))
    dim = int(config.sys_dim)
    rows.append(Setting("sys_dim", str(dim), source("sys_dim"),
                        _DIM_EFFECT.get(dim, f"{dim}D")))
    head = config.head.correction.value
    rows.append(Setting("head_correction", head, source("head_correction"),
                        _HEAD_EFFECT.get(head, head)
                        + (f" ({config.head.wcoul0_source})" if head != "off" else "")))
    if config.bispinor:
        rows.append(Setting("bispinor", "true", source("bispinor"),
                            "charge + transverse current; bispinor_gw = "
                            + config.bispinor_gw.value))
    if solver == "self_consistent":
        dsc = bool(config.density_self_consistent)
        rows.append(Setting(
            "density_self_consistent", str(dsc).lower(),
            source("density_self_consistent", derived=dsc),
            "V_H rebuilt from each map's density" if dsc else
            "V_H fixed at the DFT density: a comparison mode, not production QSGW"))
        head_update = str(config.sc.head_update)
        rows.append(Setting("sc_head_update", head_update,
                            source("sc_head_update", derived=head_update != "off"),
                            _SC_HEAD_EFFECT.get(head_update, head_update)))
    elif not bool(config.no_degen_averaging):
        tol = float(config.degen_avg_tol_ry)
        rows.append(Setting("degen_avg_tol_ry", f"{tol:.3g}", source("degen_avg_tol_ry"),
                            f"reported diagonals averaged within "
                            f"{tol * RYD_TO_EV:.2g} eV"))
    rows.append(Setting("restart", str(bool(config.restart)).lower(), source("restart"),
                        "reuse the ISDF restart bundle" if config.restart else
                        "fresh ISDF fit; no restart bundle is read"))
    layout = str(config.backend.linalg).strip().lower()
    q_per_rank = (int(n_q_irr) + int(processes) - 1) // int(processes)
    rows.append(Setting(
        "linalg", layout, str(config.backend.linalg_provenance),
        f"{q_per_rank} whole N_mu^2 matrices per rank "
        f"({_human_bytes(q_per_rank * int(n_mu) ** 2 * 16)})"
        if layout == "local" else
        f"N_mu x N_mu matrices 2D-distributed (N_mu = {int(n_mu)})"))
    if budget_gb:
        rows.append(Setting(
            "memory_per_device_gb", f"{float(budget_gb):.2f}",
            source("memory_per_device_gb", derived=True),
            "the deck's budget per device" if "memory_per_device_gb" in named else
            f"card rule{' on ' + device if device else ''}"))
    rows.append(Setting("band_chunk_size", str(int(config.memory.band_chunk_size)),
                        source("band_chunk_size"),
                        "band panels gathered in chunks of at least this"))
    if mode == "mpa" or "mpa_sampling_alpha" in named:
        alpha_source = getattr(config.mpa, "sampling_alpha_provenance", "") or ""
        rows.append(Setting("mpa_sampling_alpha", f"{config.mpa.sampling_alpha:g}",
                            "deck" if "mpa_sampling_alpha" in named else "derived",
                            "MPA frequency sampling exponent"
                            + (f" ({alpha_source})" if alpha_source != "deck" else "")))
    return rows


def resolved_notes(lines) -> list[str]:
    """The deck-resolution notes no settings row states, without their tag."""
    notes = []
    for line in lines:
        text = " ".join(str(line).replace("[config provenance]", "").split())
        if text and not any(fragment in text for fragment in _RESOLVED_BY_ROW):
            notes.append(text)
    return notes


_SIGMA_KIND = {
    "crossing": "crossing",
    "sign_definite_positive": "d > 0",
    "sign_definite_negative": "d < 0",
}


def _chi_rule_line(row: dict) -> str:
    """One screening rule as a fixed-width report row (energies in Ry)."""
    kind = row["kind"]
    if kind == "direct":
        where = f"z {row['omega_ry']:.4g} + {row['varpi_ry']:.3g}i"
        return f"{kind:<7} {where:<34} {'-':>7}  {'exact':>9}  ordered-pair scan"
    if kind == "line":
        count = row["points"]
        where = (f"varpi {row['varpi_ry']:.4g}, F {row['freq_max_ry']:.4g}, "
                 f"{count} point{'' if count == 1 else 's'}")
        nodes = (f"{row['nodes']}x{row['sweeps']}" if row["sweeps"] > 1
                 else str(row["nodes"]))
        return (f"{kind:<7} {where:<34} {nodes:>7}  "
                f"{'<=' + format(row['target'], '.0e'):>9}  "
                f"Gauss-Legendre, {row['panels']} panels, "
                f"A {row['a_dim']:.0f}, kappa0 {row['kappa0']:.2f}")
    where = (f"x {row['x_min']:.3g}-{row['x_max']:.3g}, "
             f"R {row['x_max'] / row['x_min']:.3g}")
    if kind != "static":
        axis = "i" if kind == "imag" else ""
        where = f"omega {row['omega_ry']:.4g}{axis}, {where}"
    source = (f"{row['source']}, "
              f"{'certified' if row['certified'] else 'UNCERTIFIED'}")
    if row["note"]:
        source += f"; {row['note']}"
    return (f"{kind:<7} {where:<34} {row['nodes']:>7}  "
            f"{row['error']:>9.2e}  {source}")


def _sigma_rule_lines(geometry: dict, plans: int) -> list[str]:
    """The Sigma plan header and one row per product window (box in eta)."""
    eta = float(geometry["eta_ry"])
    count = int(geometry.get("n_windows", 0))
    head = f"Sigma rules: eta {eta * RYD_TO_EV:.4f} eV"
    if "eps" in geometry:
        head += f", eps {float(geometry['eps']):.1e}"
    head += f"; {count} window{'' if count == 1 else 's'}"
    if "window_tau_pairs" in geometry:
        head += f", {geometry['window_tau_pairs']} (window,tau) pairs"
    if "distinct_tau_count" in geometry:
        head += f", {geometry['distinct_tau_count']} distinct tau"
    if "plan_seconds" in geometry:
        head += f"; plan {float(geometry['plan_seconds']):.1f} s"
    if plans > 1:
        head += f" (last of {plans} plans)"
    lines = [head]
    windows = [window for branch in geometry.get("branches", ())
               for window in branch["windows"]]
    if not windows:
        return lines
    width = max(len("window"), *(len(w["name"]) for w in windows))
    lines.append(
        f"  {'window':<{width}}  {'kind':<8}  {'Re d / eta':>16}  "
        f"{'Im d / eta':>12}  {'nodes':>5}  {'sup/eps':>7}  {'kappa':>8}  "
        f"{'fit s':>6}  rule")
    for w in windows:
        lo, hi, gamma_lo, gamma_hi = (float(v) / eta for v in w["box_ry"])
        lines.append(
            f"  {w['name']:<{width}}  {_SIGMA_KIND.get(w['kind'], w['kind']):<8}  "
            f"{lo:+7.1f}..{hi:+7.1f}  {gamma_lo:5.2f}..{gamma_hi:5.2f}  "
            f"{w['node_count']:>5}  {w['sup_error'] / w['eps']:>7.3f}  "
            f"{w['kappa_max']:>8.2e}  {w['fit_seconds']:>6.1f}  "
            f"{w['rule_source']}")
    return lines


class _Band:
    """One stage row: its seconds and the timing nodes whose time it owns, the
    subtrees under ``plus`` less those under ``minus`` (paths).  Subtracting a band
    removes its subtrees, so a residual row owns exactly the time it reports."""

    __slots__ = ("seconds", "plus", "minus")

    def __init__(self, seconds=0.0, plus=(), minus=()):
        self.seconds, self.plus, self.minus = float(seconds), tuple(plus), tuple(minus)

    def __add__(self, other):
        return _Band(self.seconds + float(other), self.plus + getattr(other, "plus", ()),
                     self.minus + getattr(other, "minus", ()))

    __radd__ = __add__

    def __sub__(self, other):
        return _Band(self.seconds - float(other), self.plus,
                     self.minus + getattr(other, "plus", ()))

    def __float__(self):
        return self.seconds

    def floor(self):
        return _Band(max(self.seconds, 0.0), self.plus, self.minus)

    def owns(self, path):
        under = lambda roots: any(tuple(path[:len(r)]) == tuple(r) for r in roots)
        return under(self.plus) and not under(self.minus)


#: Sections whose own device peak the memory table prints beside the stage rows.
_SUBSTAGE_PEAKS = ("spole.gram_reduction", "spole.passivity_held", "head.sample_terms")


def _substage_peak_lines(paths, rows):
    """Lines of the memory table for :data:`_SUBSTAGE_PEAKS`: each section's own pool
    high-water (max over its calls, max / min over ranks) and where it ran."""
    sub = {}
    for path, row in zip(paths, rows):
        if path and path[-1] in _SUBSTAGE_PEAKS:
            peak = _rank_peaks(row, "peak")
            if np.any(np.isfinite(peak)):
                old = sub.get(path[-1])
                sub[path[-1]] = (peak if old is None else np.fmax(old[0], peak), path)
    if not sub:
        return []
    lines = ["  constructor sub-stages, each section's own pool high-water (max over its calls):"]
    for name in _SUBSTAGE_PEAKS:
        if name in sub:
            peak, path = sub[name]
            lines.append(f"    {name}: peak {_gb_pair(peak, 0)} GB; {' > '.join(path[-3:])}")
    return lines


def _rank_peaks(row, key):
    """Every rank's value of a record's peak (bytes; NaN = unmeasured)."""
    ranks = row.get(key + "_ranks")
    if ranks is not None:
        return np.asarray(ranks, dtype=np.float64)
    local = row.get(key)
    return np.asarray([np.nan if local is None else float(local)])


def _peak_reader(rows):
    """``(peak_of, n_ranks, host_run)`` for the timing records ``rows``.

    ``peak_of(band)`` is every rank's pool high-water over the timing nodes the
    stage row owns (bytes, NaN where unmeasured; ``None`` when no node was) and
    the path of the node that set it.  ``host_run`` is every rank's resident
    high-water over the run (bytes), or ``None`` when no host mark was read.
    """
    paths = [tuple(r.get("path", (r["name"],))) for r in rows]
    selfs = [_rank_peaks(r, "peak_self") for r in rows]
    hosts = [_rank_peaks(r, "host") for r in rows]
    n_ranks = max([1] + [len(v) for v in selfs + hosts])
    pad = lambda v: v if len(v) == n_ranks else np.full(n_ranks, np.nan)

    def peak_of(band):
        owned = [(p, pad(v)) for p, v in zip(paths, selfs) if band.owns(p)]
        owned = [(p, v) for p, v in owned if np.any(np.isfinite(v))]
        if not owned:
            return None, ()
        stack = np.asarray([v for _, v in owned])
        peak = np.asarray([np.nanmax(col) if np.any(np.isfinite(col)) else np.nan
                           for col in stack.T])
        return peak, owned[int(np.nanargmax(np.nanmax(stack, axis=1)))][0]

    marks = [pad(v) for v in hosts if np.any(np.isfinite(v))]
    host_run = np.nanmax(np.asarray(marks), axis=0) if marks else None
    return peak_of, n_ranks, host_run


#: The stage each timing row belongs to, by its label (first match wins); a row
#: that matches none is a stage of its own.
_STAGE_OF = (
    ("setup", ("runtime bring-up", "pre-main + imports", "input + run setup")),
    ("ISDF basis and Coulomb metric", ("zeta", "zeta transverse", "V(q)", "restart load",
                                       "ISDF setup + I/O")),
    ("Screening", ("minimax quadrature", "chi0", "W", "spole ", "bank", "bank ",
                   "screening support", "W persist + q0 head", "SC W response")),
    ("Self-energy", ("Sigma ", "SC Sigma ")),
    ("Quasiparticle bands", ("mean-field load", "QP solve + diagonalize", "result writes")),
)


def _group_stages(stages):
    """``[(stage, [(label, band), ...]), ...]`` in first-seen order; a row of no
    stage (``other driver work``) is ``(None, [row])``."""
    groups: list = []
    index: dict = {}
    for name, band in stages:
        stage = next((title for title, labels in _STAGE_OF
                      if any(name == label or (label.endswith(" ") and name.startswith(label))
                             for label in labels)), None)
        if stage is None:
            groups.append((None, [(name, band)]))
            continue
        if stage not in index:
            index[stage] = len(groups)
            groups.append((stage, []))
        groups[index[stage]][1].append((name, band))
    return groups


def _gb_pair(values, width=6):
    """"max / min" over ranks in GB, or "–" when no rank measured it."""
    if values is None or not np.any(np.isfinite(values)):
        return "–"
    return f"{np.nanmax(values) / 1e9:{width}.2f} / {np.nanmin(values) / 1e9:{width}.2f}"


class GWProductionReport(ProductionReport):
    """One clean GW report, owned and written only by process zero."""

    driver = "gw"
    purpose = "GW quasiparticles"
    _WARNING_WORDS = (
        "WARNING", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
        "NOT TRUSTWORTHY", "DEPRECATED",
    )
    _KEEP_PREFIXES = (
        "  Resident ψ ",
        # The μ-batch ζ fit's plan and its measured store/timing receipts are
        # the run's record of where Z lived and what the transfers cost; the
        # ladder resolvent's per-z receipts (wall, iterations, residual,
        # high-water, fingerprints) are the same kind of record.
        "  ISDF μ-batch plan", "  μ-batch ", "  Z store: ", "  [host mem] ",
        "  Zeta output: ", "  Ladder W(z) ",
        # Every box plan carries a durable policy and accepted-rule receipt,
        # including ordinary shared-pole runs with debug disabled.
        "Sigma quadrature receipt: ", "Response quadrature: ",
        "Sigma checkpoint: ", "Shared-pole constructor: ",
        "Shared-pole sector constructor: ", "  route ", "  WARNING: route ",
        # Fixed-SC quadrature identity is a physics invariant, not backend
        # chatter: retain its compact receipt so every map's exact node set
        # and zero-rebuild claim remain auditable after live stdout is gone.
        "  SC fixed quadrature: ", "    SC fixed quadrature recompute: ",
        "    SC fixed window: ",
    )

    def __init__(self, path: str, *, runtime, debug: bool, stdout) -> None:
        super().__init__(path, runtime=runtime, debug=debug, stdout=stdout)
        self._timings_emitted = False

    def settings_block(self, *, config, n_mu: int, n_q_irr: int, processes: int,
                       resolved=()) -> None:
        """SETTINGS, before any dense data is allocated: the GW rows, then the
        deck-resolution notes (``resolved``) that no row states."""
        facts = getattr(self.runtime, "facts", None) or {}
        budget = float((facts.get("device_memory") or {}).get("budget") or 0.0) / 1e9
        self.settings(settings_rows(
            config, n_mu=n_mu, n_q_irr=n_q_irr, processes=processes,
            budget_gb=budget or float(config.memory.per_device_gb) or None,
            device=facts.get("device_kind")))
        for note in resolved_notes(resolved):
            wrapped = wrap_value(note, first_col=LABEL + 2)
            self.emit_lines(kv_lines([("resolved", wrapped[0])]) + wrapped[1:])
        if (config.qp_solver.value == "self_consistent"
                and not bool(config.density_self_consistent)):
            self.warn(Message(
                "WARNING", "density_not_self_consistent",
                "V_H stays at the DFT density on every map",
                lines=("a comparison mode, not production QSGW",),
                fix="remove density_self_consistent = false from the deck"))

    def devices(self, *, config) -> None:
        """The shared Devices rows plus the GW engines this run calls."""
        f = self.runtime.facts
        platform = f.get("ffi_platform", str(f.get("backend", "unknown")).lower())
        extras = [("dense linalg", f"{config.backend.linalg} layout")]
        # The k-convolution router has no dial: it answers by device vendor
        # (nvidia-mathdx on CUDA, the host plan route on cpu, XLA elsewhere),
        # so the report names the engine it picks.
        kconv = {"CUDA": "nvidia-mathdx fused cuFFTDx kernels",
                 "cpu": "FFTW3-ABI host plan route"}.get(
                     platform, "XLA, jnp.fft k-axis transforms")
        extras.append(("k-convolutions", f"{platform}, {kconv} (zeta fit and Sigma)"))
        # Report only dials with a caller in this calculation.
        for dial in f.get("ffi_dials", ()):
            name = str(dial.get("env", ""))
            if name != "LORRAX_BANDS_GEMM_FFI":
                continue
            if platform not in tuple(dial.get("platforms", ())):
                state = "NATIVE"
                implementation = str(dial.get("off_label") or "JAX/XLA lowering")
            elif dial.get("enabled"):
                state = "ON"
                implementation = str(dial.get("target") or "platform FFI kernel")
            else:
                state = "OFF"
                implementation = str(dial.get("off_label") or "default lowering")
            extras.append(("G(tau) GEMM", f"{implementation} ({name} = {state})"))
        super().devices(mesh_role="centroid axes X x Y", extras=extras)

    def environment(self, *, wfn) -> None:
        self.heading("Numerical environment")
        self.emit(f"Wavefunctions  : {getattr(wfn, 'backend', 'unknown')} reader")

    @staticmethod
    def trs_consequence(*, config, sym, material_class="insulator") -> str | None:
        """The automatic time-reversal-dependent branch this run takes (no deck
        key overrides it), for the symmetry block's time-reversal row."""
        allowed = bool(sym.trs_allowed)
        mode = config.compute_mode.value
        if mode == "mpa":
            from .mpa.model import chi0_orientation_route
            return "MPA chi0: " + chi0_orientation_route(material_class, trs_allowed=allowed)
        if mode == "gn_ppm":
            return ("GN probe: " + ("ordered orientations with the time-odd channel"
                                    if not allowed else "symmetric completion")
                    + ", in its own pass")
        if mode == "hl_ppm":
            return ("HL probe: real-axis completion"
                    + ("; the broken-time-reversal limitation is kept" if not allowed else ""))
        return None

    def bands(self, *, config, wfn, band_slices, zeta_ranges) -> None:
        b = band_slices
        n_e = float(getattr(wfn, "num_electrons", np.nan))
        self.heading("Band spaces and energy coverage")
        self.emit(f"Electrons      : {n_e:.5f}; occupied-band boundary = {b.b2}")
        self.emit(f"Occupied bands : {band_range(b.b0, b.b2)}")
        self.emit(f"QP valence     : {band_range(b.b1, b.b2)}")
        self.emit(f"QP conduction  : {band_range(b.b2, b.b3)}")
        self.emit(f"QP matrix      : {band_range(b.b0, b.b3)}")
        # ``b4_logical`` is the physical loaded top (tagged_arrays.py); a
        # band-slices object without it (older producers, test stubs) means
        # no mesh padding, so the padded top is the logical top.
        b4_logical = int(getattr(b, "b4_logical", b.b4))
        logical_chi = min(int(b.b4_chi), b4_logical)
        logical_sigma = min(int(b.b4_sigma), b4_logical)
        self.emit(f"chi0/W sum     : {band_range(b.b0, logical_chi)}")
        self.emit(f"Sigma sum      : {band_range(b.b0, logical_sigma)}")
        self.emit(f"Loaded ISDF psi: {band_range(b.b0, b4_logical)}")
        if int(b.b4) != b4_logical:
            self.emit(
                f"Band carrier    : {band_range(b.b0, b.b4)} "
                f"({int(b.b4) - b4_logical} zero-pad rows)")
        self.emit(f"zeta fit       : left {band_range(*zeta_ranges[0])}; "
                  f"right {band_range(*zeta_ranges[1])}")

    def sigma_coverage(self, *, config, band_slices, enk_dft_ry,
                       sigma_result) -> None:
        """Report the dynamic grid against Sigma's resolved energy origin.

        The result owns this origin: for a metal it is the fixed-N chemical
        potential of the accepted map, while for an insulator it is the
        configured VBM or midgap.  Reconstructing it from ``wfn.efermi``
        would silently misreport every non-midgap calculation.
        """
        if not config.compute_mode.is_dynamic:
            return
        b = band_slices
        # The sampled grid the result was built on (grown past the requested
        # edges, or derived from the bands when they are unset).
        _grid = sigma_result.omega_grid_ev
        grid_lo, grid_hi = ((float(_grid[0]), float(_grid[-1])) if _grid is not None
                            else sigma_requested_edges_ev(config.sigma))
        ef_ev = float(sigma_result.efermi_dft_ev)
        provenance = (getattr(sigma_result, "omega_reference_provenance", None)
                      or config.sigma.fermi_reference)
        energies = np.asarray(enk_dft_ry, dtype=np.float64) * RYD_TO_EV
        rel = energies - ef_ev
        lv0, lv1 = b.b1 - b.b0, b.b2 - b.b0
        lc0, lc1 = b.b2 - b.b0, b.b3 - b.b0

        def span(i0, i1):
            if i1 <= i0:
                return None
            values = rel[:, i0:i1]
            return float(np.min(values)), float(np.max(values))

        val_span, cond_span = span(lv0, lv1), span(lc0, lc1)
        target = [v for pair in (val_span, cond_span) if pair is not None
                  for v in pair]

        self.heading("Dynamic Sigma energy coverage")
        self.emit(f"Energy origin   : E_F = {ef_ev:+.5f} eV ({provenance})")
        omega_grid = np.asarray(
            () if _grid is None else _grid, dtype=np.float64)
        grid_note = (f"; step={float(config.sigma.omega_step_ev):.5f} eV; "
                     f"{int(omega_grid.size)} points")
        if sigma_result.sigma_c_on_shell is not None:
            # On shell the grid is only the covered range; Sigma was not
            # sampled on its points (docs/theory/sigma-on-shell.md).
            grid_note += "; on shell: covered range only"
        self.emit(f"Sigma omega    : [{grid_lo:+.5f}, {grid_hi:+.5f}] eV "
                  f"relative to E_F{grid_note}")
        if val_span is not None:
            self.emit(f"DFT QP valence : [{val_span[0]:+.5f}, "
                      f"{val_span[1]:+.5f}] eV relative to E_F")
        if cond_span is not None:
            self.emit(f"DFT QP conduct.: [{cond_span[0]:+.5f}, "
                      f"{cond_span[1]:+.5f}] eV relative to E_F")
        if target:
            target_lo, target_hi = min(target), max(target)
            lower_margin = target_lo - grid_lo
            upper_margin = grid_hi - target_hi
            if lower_margin >= 0.0 and upper_margin >= 0.0:
                self.emit("Coverage status : COMPLETE")
                self.emit(f"Grid margins    : {lower_margin:.5f} eV below; "
                          f"{upper_margin:.5f} eV above requested DFT output bands")
            else:
                self.emit("Coverage status : INCOMPLETE")
                shortfalls = []
                if lower_margin < 0.0:
                    shortfalls.append(f"{-lower_margin:.5f} eV below")
                if upper_margin < 0.0:
                    shortfalls.append(f"{-upper_margin:.5f} eV above")
                self.emit("Grid shortfall  : " + "; ".join(shortfalls)
                          + " requested DFT output bands")
                coverage = getattr(sigma_result, "omega_coverage", None)
                policy_name = str(getattr(coverage, "policy", "unknown"))
                affected = ""
                if coverage is not None:
                    mask = np.asarray(getattr(coverage, "mask_kn", ()))
                    default_uncovered = (int(np.count_nonzero(~mask.astype(
                        bool))) if mask.size else 0)
                    n_uncovered = int(getattr(
                        coverage, "n_uncovered", default_uncovered))
                    if mask.size:
                        affected = (f"; Sigma(E_DFT) has {n_uncovered}/"
                                    f"{mask.size} out-of-grid cells")
                named = "; ".join(getattr(coverage, "uncovered_bands", ()))
                self.warn(Message(
                    "WARNING", "sigma_grid_incomplete",
                    "the dynamic Sigma grid does not cover every requested band",
                    found=("grid", "short by " + "; ".join(shortfalls) + affected.replace(
                        "; ", ", ", 1)),
                    lines=(f"out-of-range policy {policy_name}: those states use "
                           "Sigma(omega=0)" + (f" ({named})" if named else ""),),
                    fix="widen sigma_omega_min_ev / sigma_omega_max_ev, or add a "
                        "sigma_omega_patches_ev window over them"))

        state = "ON" if config.sigma.band_extrapolation else "OFF"
        estimator = (getattr(
            sigma_result, "band_extrapolation_estimator", None)
            or config.sigma.band_extrapolation_estimator)
        scheme = (getattr(sigma_result, "band_extrapolation_scheme", None)
                  or config.sigma.band_extrapolation_bracket_scheme)
        self.emit(f"Band tail      : {state} | estimator={estimator} | "
                  f"brackets={scheme}")
        counts = getattr(sigma_result, "band_extrapolation_counts", None)
        if counts:
            self.emit("Tail calculations: " + " / ".join(
                f"N{i}={int(value)}" for i, value in enumerate(counts, start=1))
                + " cumulative bands")
        if getattr(sigma_result, "sigma_c_at_dft_diag_unextrap_ev",
                   None) is not None:
            from file_io.sigma_output import RAW_SIGMA_NOTE
            self.emit("Raw Sigma_c    : sigC_raw, eqp0_raw, eqp1_raw "
                      "columns in sigma_diag.dat")
            self.emit("                 " + RAW_SIGMA_NOTE)

    def quadrature(self) -> None:
        """Every quadrature rule the run used: screening chi0, then Sigma."""
        rules = quadrature_log.chi_rules()
        geometry, plans = quadrature_log.sigma_plan()
        if not rules and geometry is None:
            return
        self.heading("Quadrature windows")
        if rules:
            self.emit("Screening chi0 rules (energies in Ry)")
            self.emit(f"  {'rule':<7} {'where':<34} {'nodes':>7}  "
                      f"{'max err':>9}  source")
            for row in rules:
                self.emit("  " + _chi_rule_line(row))
        if geometry is not None:
            for line in _sigma_rule_lines(geometry, plans):
                self.emit(line)

    def qp_gap(self, *, band_slices, e_dft_ry, e_qp_ry) -> None:
        """Summarize the fundamental gap without duplicating EQP tables."""
        b = band_slices
        dft = np.asarray(e_dft_ry, dtype=np.float64) * RYD_TO_EV
        qp = np.asarray(e_qp_ry, dtype=np.float64) * RYD_TO_EV
        iv = int(b.b2 - b.b0 - 1)
        ic = int(b.b2 - b.b0)
        self.heading("Fundamental gap")
        if (dft.ndim != 2 or qp.ndim != 2 or dft.shape != qp.shape
                or iv < 0 or ic >= dft.shape[1]):
            self.emit("Gap            : unavailable (no complete protected "
                      "valence/conduction pair in the QP result)")
            return
        dft_gap = float(np.min(dft[:, ic]) - np.max(dft[:, iv]))
        qp_gap = float(np.min(qp[:, ic]) - np.max(qp[:, iv]))
        if not np.isfinite(dft_gap) or not np.isfinite(qp_gap):
            self.emit("Gap            : unavailable (non-finite band edge)")
            return
        state = "insulating" if qp_gap > 0.0 else "metallic/overlapping"
        self.emit(f"DFT gap        : {dft_gap:.5f} eV")
        self.emit(f"Full-matrix effective-H gap: {qp_gap:.5f} eV ({state})")
        self.emit(f"Gap correction : {qp_gap - dft_gap:+.5f} eV relative to DFT")
        self.emit("State energies  : written to the EQP files listed below")

    def timings(self, records, *, wall: float) -> None:
        """Print accumulated, non-overlapping major scientific stages.

        ``gw_jax.isdf`` and ``gw_jax.screening`` are inclusive parent timers.
        Their named child rows are therefore subtracted from a residual setup
        row instead of being added beside the parent.  This keeps the report's
        invariant explicit: every displayed stage contributes to the wall
        exactly once, while retaining the zeta/V and chi0/W distinctions that
        operators use to diagnose scaling. Printed once: a refusal raised after
        the completed run's table (the artifact gate) does not repeat it.
        """
        if self._timings_emitted:
            return
        self._timings_emitted = True
        rows = list(records)

        def total(predicate):
            chosen = [row for row in rows if predicate(row)]
            return _Band(sum(float(row["inclusive"]) for row in chosen),
                         plus=[tuple(row.get("path", (row["name"],))) for row in chosen])

        def top_level(*names):
            wanted = set(names)
            return total(lambda r: r["name"] in wanted
                         and tuple(r.get("path", (r["name"],)))
                         == (r["name"],))

        def outer_prefixed(prefix, *, within=None, excluding=()):
            def selected(row):
                path = tuple(row.get("path", (row["name"],)))
                if not row["name"].startswith(prefix):
                    return False
                if within is not None and (not path or path[0] != within):
                    return False
                if any(str(parent).startswith(excluding) for parent in path[:-1]):
                    return False
                return not any(str(parent).startswith(prefix)
                               for parent in path[:-1])
            return total(selected)

        def partition(parent, prefixes, label):
            """Replace an inclusive band by disjoint descendants and residuals."""
            parents = [r for r in rows if r["name"] == parent]
            result = []
            for owner in parents:
                path = tuple(owner.get("path", (parent,)))
                selected = {tuple(r["path"]): r for r in rows
                            if tuple(r.get("path", ()))[:len(path)] == path
                            and len(r.get("path", ())) > len(path)
                            and r["name"].startswith(prefixes)}

                def visit(key, row, title):
                    children = [p for p in selected if p[:len(key)] == key
                                and len(p) > len(key)
                                and not any(q != p and len(q) > len(key)
                                            and p[:len(q)] == q for q in selected)]
                    for child in children:
                        name = selected[child]["name"].split(".", 1)[1]
                        visit(child, selected[child], label + " " + name)
                    residual = float(row["inclusive"]) - sum(
                        float(selected[p]["inclusive"]) for p in children)
                    result.append((title + (" other" if children else ""),
                                   _Band(residual, plus=(key,), minus=children).floor()))

                visit(path, owner, label)
            # A name may occur under the first-call probe and ordinary sweep.
            totals = {}
            for name, band in result:
                totals[name] = totals.get(name, 0.0) + band
            return list(totals.items())

        isdf_total = top_level("gw_jax.isdf")
        zeta = total(lambda r: r["name"] == "gw_jax.zeta_fit_chunked"
                     and tuple(r.get("path", ()))[:1] == ("gw_jax.isdf",))
        zeta_transverse = outer_prefixed(
            "gw_jax.zeta_fit_transverse", within="gw_jax.isdf")
        # The bispinor's CC and TT tiles (V_q_compute_cc / _tt) are V(q) too.
        v_q = outer_prefixed("gw_jax.V_q_compute", within="gw_jax.isdf")
        restart_load = total(
            lambda r: r["name"] == "gw_jax.restart_load"
            and tuple(r.get("path", ()))[:1] == ("gw_jax.isdf",))
        isdf_support = (
            isdf_total - zeta - zeta_transverse - v_q - restart_load).floor()

        screening_total = top_level("gw_jax.screening")
        # A shared-pole bank may call chi/W owners internally. Its inclusive
        # band already owns those seconds; do not print them a second time.
        chi0 = outer_prefixed("chi.", within="gw_jax.screening", excluding=("spole.",))
        w_screen = outer_prefixed("W.", within="gw_jax.screening", excluding=("spole.",))
        spole_rows = [r for r in rows if r["name"].startswith("spole.")
                      and tuple(r.get("path", ()))[:1] == ("gw_jax.screening",)
                      and not any(str(p).startswith("spole.")
                                  for p in r.get("path", ())[:-1])]
        # Preserve the constructor census names and their execution order.
        # Waits name the NEXT consumer; they drain PREVIOUS asynchronous work.
        screening_details = []
        for name in dict.fromkeys(r["name"] for r in spole_rows):
            label = name.removeprefix("spole.")
            if label.startswith("device_wait."):
                label = "wait before " + label.removeprefix("device_wait.")
            elif label.startswith("rank_wait."):
                continue  # all process alignment is one disjoint row below
            elif label == "passivity_held":
                label = "passivity + held (fused)"
            if name == "spole.bank" and any(r["name"].startswith("bank.") for r in rows):
                screening_details.extend(partition("spole.bank", ("bank.",), "bank"))
                continue
            screening_details.append(("spole " + label,
                total(lambda r: r in spole_rows and r["name"] == name)))
        screening_details.append(("spole rank synchronization",
            total(lambda r: r in spole_rows and r["name"].startswith("spole.rank_wait."))))
        screening_support = (screening_total - chi0 - w_screen
                             - sum(value for _, value in screening_details)).floor()

        # The dynamic-Sigma executor opens ``sigma.rule_plan`` (box-rule
        # building, reused within the run by box and tolerance) and ``sigma.tau_sweep`` (the
        # tau contraction) under gw_jax.sigma or, in a self-consistent run,
        # under gw_jax.sc_driver. Shared finalization and the fenced setup
        # phases below are disjoint from the plan and sweep.
        sigma_total = top_level("gw_jax.sigma", "gw_jax.sc_driver")
        # An SC map's W and its Sigma rule refit (nested in the plan) are
        # their own rows, so an SC table sums to the wall with named stages.
        sc_w_response = outer_prefixed("sc.w_response")
        sigma_refit = outer_prefixed("sigma.rule_refit")
        sigma_plan = outer_prefixed("sigma.rule_plan") - sigma_refit
        sigma_sweep = outer_prefixed("sigma.tau_sweep")
        sigma_details = [
            ("Sigma pending inputs", outer_prefixed("sigma.input_wait")
             + outer_prefixed("sigma.finalize_input_wait")),
            ("Sigma exchange", outer_prefixed("sigma.exchange")),
            ("Sigma Hartree", outer_prefixed("sigma.hartree")),
            ("Sigma model validation", outer_prefixed("sigma.model_validate")),
            ("Sigma capacity", outer_prefixed("sigma.capacity")),
            ("Sigma branches", outer_prefixed("sigma.branches")),
            ("Sigma census", outer_prefixed("sigma.census")),
            ("Sigma finalize + writes", outer_prefixed("gw_jax.dynamic_sigma_finalize")),
        ]
        sigma_other = (sigma_total - sc_w_response - sigma_refit
                       - sigma_plan - sigma_sweep
                       - sum(value for _, value in sigma_details)).floor()

        stages = [
            ("runtime bring-up", total(lambda r: r["name"].startswith(
                "gw_jax.runtime_stack."))),
            ("pre-main + imports", top_level("gw_jax.imports")),
            ("input + run setup", top_level("gw_jax.startup")),
            ("zeta", zeta),
            ("zeta transverse", zeta_transverse),
            ("V(q)", v_q),
            ("restart load", restart_load),
            ("ISDF setup + I/O", isdf_support),
            ("minimax quadrature", top_level("gw_jax.minimax_quadrature")),
            ("chi0", chi0),
            ("W", w_screen),
            *screening_details,
            ("spole other" if spole_rows else "screening support", screening_support),
            ("W persist + q0 head", top_level(
                "gw_jax.persist_w0", "gw_jax.static_head")),
            ("SC W response", sc_w_response),
            ("Sigma rule plan", sigma_plan),
            ("SC Sigma rule refit", sigma_refit),
            *(partition("sigma.tau_sweep", ("tau.",), "Sigma tau")
              if any(r["name"].startswith("tau.") for r in rows)
              else [("Sigma tau sweep", sigma_sweep)]),
            *sigma_details,
            ("Sigma other", sigma_other),
            ("mean-field load", top_level("gw_jax.kin_ion_load")),
            ("QP solve + diagonalize", top_level(
                "gw_jax.solve_qp", "gw_jax.qp_eigh")),
            ("result writes", top_level("gw_jax.output")),
        ]
        # A stage absent in a mode should not occupy a zero-valued report row.
        # Use the table's two-decimal display threshold so a floating-point
        # subtraction residual cannot survive the filter as ``0.00 s``.
        stages = [(name, band) for name, band in stages
                  if float(band) >= 0.005]
        accounted = sum(float(band) for _name, band in stages)
        stages.append(("other driver work", _Band(
            max(float(wall) - accounted, 0.0),
            plus=[tuple(r.get("path", (r["name"],))) for r in rows
                  if len(r.get("path", (r["name"],))) == 1],
            minus=[p for _name, band in stages for p in band.plus])))
        # The table: each stage row is the sum of its parts, its device column
        # the largest pool high-water over the timing nodes the rows own.
        peak_of, n_ranks, host_run = _peak_reader(rows)
        table = []
        for parent, children in _group_stages(stages):
            parts = []
            for name, band in children:
                peak, _setter = peak_of(band)
                parts.append(TimingRow(name, float(band), 1 if parent else 0,
                                       float(np.nanmax(peak)) / 1e9
                                       if peak is not None else None))
            if parent:
                gbs = [row.device_gb for row in parts if row.device_gb is not None]
                table.append(TimingRow(parent, sum(row.seconds for row in parts), 0,
                                       max(gbs) if gbs else None))
            table += parts
        leaves = [row for row in table if row.device_gb is not None
                  and (row.depth == 1 or row.label == "other driver work")]
        top = max(leaves, key=lambda row: row.device_gb, default=None)
        notes = []
        # The "runtime bring-up" row is the sum of these phases; state the
        # split so a slow bring-up names its phase (Frontier Fe V2 runs ending
        # 2026-10-15 20:33-21:06 UTC: 11 of 13 took 21-74 s, against 6.9-10.1 s
        # for 39 of 40 runs outside that window; three lanes, two nodes, and
        # no phase recorded anywhere).
        phases = [(str(r["name"])[len("gw_jax.runtime_stack."):], float(r["inclusive"]))
                  for r in rows if str(r["name"]).startswith("gw_jax.runtime_stack.")]
        if phases:
            notes.append("runtime bring-up (rank 0) = " + " + ".join(
                f"{phase} {seconds:.2f}" for phase, seconds in phases) + " s")
        if any(r["name"].startswith(("bank.", "tau.")) for r in rows):
            notes += [
                "bank real_time/laplace = incumbent G/FFT correlation plus bank coefficient "
                "carry; these fused clocks are not FFT-only or evidence of a bandwidth limit.",
                "Sigma tau other is the window executables: W(tau) synthesis, G/W convolution, "
                "projection and frequency fold in one program. No pure-GEMM or pure-FFT wall "
                "is inferred.",
                "device_wait drains preceding work; rank_wait measures process alignment."]
        from common.timing import stage_sync
        notes.append("stage walls: " + (
            "device-attributed (LORRAX_STAGE_SYNC=1: every stage waits for its device work)."
            if stage_sync() else
            "host-attributed (no timing-only device waits; device work lands in the first "
            "later stage that reads it; LORRAX_STAGE_SYNC=1 attributes it)."))
        if spole_rows:
            notes.append("spole bands: fenced host walls; wait-before rows drain prior "
                         "device/effect work, not the named consumer. Rank synchronization is "
                         "separate. Local passivity/held share one compiled call; their fused "
                         "wall is not split into invented timings.")
        from lxkit import memory
        budget = memory.run().get("budget", 0.0) / 1e9
        self.heading("Major-stage timing")
        self.emit_lines(timing_lines(
            table, wall=float(wall), notes=notes, budget_gb=budget or None,
            device_peak=(top.device_gb, top.label) if top is not None else None,
            host_peak=((float(np.nanmax(host_run)) / 1e9, float(np.nanmin(host_run)) / 1e9)
                       if host_run is not None else None),
            n_ranks=n_ranks))
        width = max([22] + [len(name) for name, _ in stages])     # the memory table's
        self._stage_memory(rows, stages, width=width, wall=wall)
        self._routes()

    def _routes(self) -> None:
        """Every memory-driven route the run took and why (``lxkit.memory.routes``)."""
        from lxkit import memory
        taken = memory.routes()
        if not taken:
            return
        self.heading("Routes")
        for row in taken:
            self.emit(f"  {row['line']}" + (f" [x{row['count']}]" if row["count"] > 1 else ""))

    def _stage_memory(self, rows, stages, *, width, wall) -> None:
        """The device peak of every major stage against its planner's price, and its host peak.

        A row's peak is the pool high-water mark over the timing nodes it owns
        (``common.timing``; max and min over ranks).  A priced stage's receipt
        (``lxkit.memory``) holds its own measured figure: the high-water from the
        price's booking to the stage's exit, after waits on the work around it.

        The host columns are the process resident high-water mark (``ru_maxrss``,
        which never falls) at the row's last exit and the rise the row's own
        intervals caused, each the max over ranks.  No planner prices the host.
        """
        from common import gpu_utils, timing
        from lxkit import memory
        paths = [tuple(r.get("path", (r["name"],))) for r in rows]
        selfs = [_rank_peaks(r, "peak_self") for r in rows]
        hosts = [_rank_peaks(r, "host") for r in rows]
        rises = [_rank_peaks(r, "host_rise") for r in rows]
        if not any(np.any(np.isfinite(v)) for v in selfs + hosts):
            return
        row_peak, n_ranks, _host_run = _peak_reader(rows)
        pad = lambda v: v if len(v) == n_ranks else np.full(n_ranks, np.nan)

        def row_host(band):
            """Max over ranks of the row's host mark and of the rise it caused (GB)."""
            mine = [i for i, p in enumerate(paths)
                    if band.owns(p) and np.any(np.isfinite(hosts[i]))]
            if not mine:
                return f"{'–':>7}  {'–':>6}"
            mark = np.nanmax([np.nanmax(hosts[i]) for i in mine])
            rise = np.nanmax(np.nansum([pad(rises[i]) for i in mine], axis=0))
            return f"{mark / 1e9:7.2f}  {rise / 1e9:+6.2f}"

        priced = [(row, np.where(m >= 0, m, np.nan)) for row in memory.receipts()
                  for m in (_rank_peaks(row, "measured"),)]
        # The budget, the run's device peak and the host peak are the timing
        # block's memory summary; this table is each stage's detail.
        self.heading("Major-stage device and host memory")
        self.emit(f"  per rank, pool high-water per stage; max / min over {n_ranks} "
                  f"rank(s); prices and γ: the priced stages below")
        if any(np.any(np.isfinite(v)) for v in hosts):
            self.emit(f"  host = resident high-water (ru_maxrss) per rank at the stage's last "
                      f"exit, max over ranks; rise = what the stage added to it; unpriced")
        holds = gpu_utils.host_holds()
        if holds:
            self.emit("  held on the host across stages, per rank: " + "; ".join(
                f"{name} {nbytes / 1e9:.2f} GB" for name, nbytes in holds.items()))
        self.emit(f"  {'stage':<{width}}  peak GB max / min  "
                  f"host GB    rise  set by")
        for name, band in stages:
            peak, setter = row_peak(band)
            host = row_host(band)
            if peak is None and "–" in host:
                continue
            self.emit(f"  {name:<{width}}  {_gb_pair(peak):>17}  {host}  "
                      f"{' > '.join(setter[-2:])}")
        # Constructor sub-stages with their own pool high-water (TimingNode.peak), so a leg
        # measures what the reduction, passivity and head prices are calibrated against.
        for line in _substage_peak_lines(paths, rows):
            self.emit(line)
        if priced:
            self.emit("  priced stages (receipts; measured max over ranks):")
            for row, peak in priced:
                worst = float(np.nanmax(peak)) if np.any(np.isfinite(peak)) else -1.0
                self.emit("    " + memory.receipt_line(row["stage"], row["price"], worst,
                                                       row["exact"], row["source"])
                          + f" [x{row['count']}]")
        self.emit("  " + memory.sizing()["line"])
        for finding in memory.findings() if memory.STRICT else ():
            self.emit("  memory census: " + finding)
        reads, seconds = timing.instrument_cost()
        self.emit(f"  instrument: {memory.instrument()['source'] or 'none'} "
                  f"({'calibrate' if memory.CALIBRATE else 'production'}); {reads} boundary "
                  f"reads on rank 0 took {seconds * 1e3:.1f} ms "
                  f"({100.0 * seconds / wall if wall else 0.0:.4f}% of the wall)")

__all__ = ["GWProductionReport"]
