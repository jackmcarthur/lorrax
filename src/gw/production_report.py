"""Rank-zero, human-readable GW calculation report.

This is the scientific output of ``gwjax``.  It intentionally does not echo
library inventories, per-rank messages, HDF5 implementation details or the
driver's historical diagnostic prose.  Those remain available behind the one
driver-wide ``LORRAX_DEBUG_PRINT`` switch and in the launcher's own log.
"""

from __future__ import annotations

import os
import re

import numpy as np

from common.production_report import ProductionReport
from common.report_blocks import (
    LABEL, Message, Setting, TimingRow, kv_lines, table_lines, timing_lines, wrap_value)
from common.scientific_output import band_range as _code_band_range
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


def band_range(lo: int, hi: int) -> str:
    """A zero-based half-open band interval as band numbers: ``1-3``, or ``3``
    for one band (``common.scientific_output.band_range`` otherwise)."""
    if int(hi) - int(lo) == 1:
        return str(int(hi))
    return _code_band_range(lo, hi)



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
#: Where the q->0 head's response comes from, by ``wcoul0_source``.
_HEAD_SOURCE = {
    "s_tensor": "from dipole.h5",
    "epshead": "from the epsilon head",
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
    head_source = str(config.head.wcoul0_source)
    rows.append(Setting("head_correction", head, source("head_correction"),
                        _HEAD_EFFECT.get(head, head)
                        + (" " + _HEAD_SOURCE.get(head_source, f"({head_source})")
                           if head != "off" else "")))
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
    elif not bool(config.no_degen_averaging) and "degen_avg_tol_ry" in named:
        # The default tolerance changes no result a user reads; a deck value is shown.
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
    if "band_chunk_size" in named:
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


def _ev(value_ry: float) -> str:
    """An energy in eV: three significant figures below 1 eV, two decimals above."""
    ev = float(value_ry) * RYD_TO_EV
    return f"{ev:.3g}" if abs(ev) < 1.0 else f"{ev:.2f}"


def chi_rule_detail_row(row: dict) -> str:
    """One screening rule as a stage-2 ``chi0 rules`` row (energies in eV)."""
    kind = row["kind"]
    if kind == "direct":
        where = f"z {_ev(row['omega_ry'])} + {_ev(row['varpi_ry'])}i eV"
        return f"{kind:<7} {where:<35}{'-':>5}   {'exact':>7}   ordered-pair scan"
    if kind == "line":
        count = row["points"]
        where = (f"varpi {_ev(row['varpi_ry'])} eV, F {_ev(row['freq_max_ry'])} eV, "
                 f"{count} point{'' if count == 1 else 's'}")
        nodes = (f"{row['nodes']}x{row['sweeps']}" if row["sweeps"] > 1
                 else str(row["nodes"]))
        return (f"{kind:<7} {where:<35}{nodes:>5}   "
                f"{'<=' + format(row['target'], '.0e'):>7}   Gauss-Legendre")
    where = (f"x {_ev(row['x_min'])}-{_ev(row['x_max'])} eV, "
             f"R {row['x_max'] / row['x_min']:.3g}")
    if kind != "static":
        omega = float(row["omega_ry"]) * RYD_TO_EV
        where = (f"omega {omega:.1f}{'i' if kind == 'imag' else ''} eV, "
                 f"x {_ev(row['x_min'])}-{_ev(row['x_max'])} eV")
    # The certificate flag is the ``screening.chi0_rules`` record's; the row
    # names where the rule came from.
    return f"{kind:<7} {where:<35}{row['nodes']:>5}   {row['error']:>7.1e}   {row['source']}"


def log_chi_rules() -> None:
    """The screening rules under stage 2 of the main log (``chi0 rules``)."""
    from common.log import block, get_logger, record
    rules = quadrature_log.chi_rules()
    if not rules or quadrature_log.shown("chi"):
        return
    log = get_logger(__name__)
    log.debug("chi0 rules", extra=record("screening.chi0_rules", rules=rules))
    # Unlabelled detail rows keep their spacing (a labelled value is
    # re-wrapped); the label column is written here, 15 wide.
    log.info(f"{'chi0 rules':<14} {'rule':<7} {'where':<35}{'nodes':>5}   "
             f"{'max err':>7}   source", extra=block("screening"))
    for row in rules:
        log.info(" " * 15 + chi_rule_detail_row(row), extra=block("screening"))
    quadrature_log.mark_shown("chi")


def sigma_rule_detail_lines(geometry: dict) -> tuple[str, list[str]]:
    """The ``Sigma rules`` head and the window table rows (Re d / eta only)."""
    eta = float(geometry["eta_ry"])
    count = int(geometry.get("n_windows", 0))
    head = f"eta {eta * RYD_TO_EV:.2f} eV"
    if "eps" in geometry:
        head += f", eps {float(geometry['eps']):.1e}"
    head += f"; {count} window{'' if count == 1 else 's'}"
    if "window_tau_pairs" in geometry:
        head += f", {geometry['window_tau_pairs']} (window, tau) pairs"
    if "distinct_tau_count" in geometry:
        head += f", {geometry['distinct_tau_count']} distinct tau"
    windows = [window for branch in geometry.get("branches", ())
               for window in branch["windows"]]
    if not windows:
        return head, []
    width = max(22, *(len(w["name"]) for w in windows))
    rows = [f"{'window':<{width}} {'kind':<8}  {'Re d / eta':>16}{'nodes':>9}"
            f"{'sup/eps':>9}{'kappa':>10}   rule"]
    for w in windows:
        lo, hi = (float(v) / eta for v in w["box_ry"][:2])
        rows.append(
            f"{w['name']:<{width}} {_SIGMA_KIND.get(w['kind'], w['kind']):<8}  "
            f"{lo:+7.1f}..{hi:+7.1f}{w['node_count']:>9}"
            f"{w['sup_error'] / w['eps']:>9.3f}{w['kappa_max']:>10.1e}   "
            f"{_rule_source(w['rule_source'])}")
    return head, rows


def _rule_source(source) -> str:
    """A Sigma window rule's origin in plain words, no digests or rule tags:
    ``built`` (fitted for this window), ``rebuilt`` (an SC map refitted it:
    ``rebuild:sc-fixed:...``) or ``reused`` (another window's or an earlier
    map's rule: ``plan:<digest>``, ``scope:<digest>``, ``hit:sc-fixed``,
    ``resume:sc-fixed``); ``init:<source>`` reads as its source."""
    source = str(source)
    while source.startswith("init:"):
        source = source[len("init:"):]
    if source.startswith("rebuild"):
        return "rebuilt"
    if source.startswith(("plan:", "scope:", "hit:", "resume:")):
        return "reused"
    if source == "analytic-line":
        return "analytic"
    return source.split(":", 1)[0]


def log_sigma_rules(geometry: dict) -> None:
    """The Sigma rule plan under stage 3 of the main log: the ``Sigma rules``
    row and the window table.  Only the run's first plan is shown there; a
    later plan (an SC map) is the end-of-run quadrature block's."""
    from common.log import block, get_logger
    if quadrature_log.sigma_plan()[1] != 1:
        return
    log = get_logger(__name__)
    head, rows = sigma_rule_detail_lines(geometry)
    log.info(head, extra=block("sigma", label="Sigma rules"))
    for row in rows:
        log.info(row, extra=block("sigma"))
    quadrature_log.mark_shown("sigma")


# ── stage rows ───────────────────────────────────────────────────────────────
def v_q_detail(*, n_q: int, n_mu: int, seconds: float | None, restart_file=None) -> str:
    """The stage-1 ``V(q)`` row: how many q, the matrix size, and its wall
    (or the restart file it was read from)."""
    text = f"{int(n_q)} q, {int(n_mu)} x {int(n_mu)} per q"
    if restart_file:
        return text + f", read from {os.path.basename(str(restart_file))}"
    return text + (f", {float(seconds):.2f} s" if seconds is not None else "")


def w_roles_detail(roles, *, n_q: int, trs_allowed: bool = True) -> str:
    """The stage-2 ``W roles`` row: each screened W the self-energy reads,
    with its frequency, and the q set the Dyson series ran on.

    ``roles`` are ``(role, omega_ry complex or None)``.
    """
    parts = []
    for role, omega in roles:
        notes = []
        if omega is not None:
            omega = complex(omega)
            notes.append("omega 0" if omega == 0 else
                         f"omega {omega.imag * RYD_TO_EV:.1f}i eV" if omega.real == 0.0 else
                         f"omega {omega.real * RYD_TO_EV:.1f} eV")
        if role == "probe" and not trs_allowed:
            notes.append("ordered orientations")
        parts.append(str(role) + (f" ({', '.join(notes)})" if notes else ""))
    return ", ".join(parts) + f"; Dyson on {int(n_q)} q"


def head_detail(config) -> str:
    """The stage-2 ``q->0 head`` row: the head correction and its source."""
    head = config.head.correction.value
    if head == "off":
        return "off: " + _HEAD_EFFECT["off"]
    source = str(config.head.wcoul0_source)
    return f"{head}, " + ("local fields, " if head == "full" else "no local fields, ") + (
        _HEAD_SOURCE.get(source, f"source {source}"))


def log_stage_row(label: str, value: str, *, stage: str = "detail") -> None:
    """One ``label value`` row under the current stage of the main log."""
    from common.log import block, get_logger
    get_logger(__name__).info(value, extra=block(stage, label=label))


# ── setup blocks: INPUT FILES, BAND WINDOWS ──────────────────────────────────
def input_file_lines(rows) -> list[str]:
    """The INPUT FILES table: ``file / bands / k grid / spinor / check``.

    ``rows`` are ``(file, bands, k grid, spinor, check)``; a cell no cheap
    header read gives is ``-``.  The check column is last and unpadded.
    """
    header = ("file", "bands", "k grid", "spinor", "check")
    widths = [max([len(header[i])] + [len(str(row[i])) for row in rows])
              for i in range(4)]
    widths = [max(w, minimum) for w, minimum in zip(widths, (22, 8, 15, 8))]
    return table_lines(header, [tuple(str(cell) for cell in row) for row in rows],
                       widths=(*widths, 0), align="lllll")


_FEATURE_FIT = re.compile(r"left=\((\d+),\s*(\d+)\),\s*right=\((\d+),\s*(\d+)\)")


def centroid_file_bands(path) -> str:
    """The bands a centroid file was selected on, ``1-3 x 1-7`` (left x right),
    from its ``# feature fit: left=(0, 3), right=(0, 7)`` header line; ``-``
    when the header does not say.  Only the leading comment lines are read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.startswith("#"):
                    break
                match = _FEATURE_FIT.search(line)
                if match and "feature fit" in line:
                    lo_l, hi_l, lo_r, hi_r = (int(v) for v in match.groups())
                    return f"{band_range(lo_l, hi_l)} x {band_range(lo_r, hi_r)}"
    except OSError:
        pass
    return "-"


def _same_path(a, b) -> bool:
    """Whether two paths name the same file (both unset counts as different)."""
    if not a or not b:
        return False
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def _k_cell(nk: int, n_stored: int, grid) -> str:
    """``5x5x1, 9 IBZ`` / ``5x5x1, 25 k``: the k set a file stores, on the
    run's grid when its full-zone count matches it."""
    text = f"{n_stored} IBZ" if int(n_stored) != int(nk) else f"{int(nk)} k"
    if grid is not None and int(np.prod(grid)) == int(nk):
        return f"{grid[0]}x{grid[1]}x{grid[2]}, " + text
    return text


def h5_input_cells(path, kind: str, *, grid=None) -> tuple[str, str, str]:
    """``(bands, k grid, spinor)`` of ``kin_ion.h5`` (``kind`` ``"kin_ion"``) or
    ``dipole.h5`` (``"dipole"``) from attributes and dataset shapes only (no
    dataset is read); a cell the header does not give is ``-``."""
    cells = ["-", "-", "-"]
    try:
        import h5py
        with h5py.File(path, "r") as h5:
            if kind == "kin_ion":
                attrs = h5["kin_ion"].attrs
                if "nb" in attrs:
                    cells[0] = band_range(0, int(attrs["nb"]))
                if "nk" in attrs:
                    cells[1] = _k_cell(int(attrs["nk"]), int(attrs.get("nrk", attrs["nk"])),
                                       grid)
                if "nspinor" in attrs:
                    cells[2] = str(int(attrs["nspinor"]))
            else:
                attrs = h5.attrs
                if "band_energies" in h5:
                    shape = h5["band_energies"].shape
                    cells[0] = band_range(0, int(shape[-1]))
                    cells[1] = _k_cell(int(shape[0]), int(shape[0]), grid)
                if "prov_nspinor" in attrs:
                    cells[2] = str(int(attrs["prov_nspinor"]))
    except Exception:                       # noqa: BLE001 - a header the report cannot read
        pass
    return tuple(cells)


def wavefunction_lines(name: str, *, wfn, sym, n_bands: int) -> list[str]:
    """The ``Wavefunctions (WFN.h5)`` detail block under INPUT FILES."""
    grid = tuple(int(v) for v in np.asarray(wfn.kgrid).reshape(-1)[:3])
    shift = np.asarray(getattr(wfn, "shift", (0, 0, 0)), dtype=np.float64).reshape(-1)[:3]
    shift_text = ("unshifted" if not np.any(np.abs(shift) > 1e-12) else
                  "shift (" + ", ".join(f"{v:.3f}" for v in shift) + ")")
    n_full, n_red = int(sym.nk_tot), int(sym.nk_red)
    stored = f"{n_red} k stored (IBZ)" if n_red != n_full else f"{n_full} k stored"
    electrons = float(getattr(wfn, "num_electrons", np.nan))
    nspinor = int(wfn.nspinor)
    spinor = "2-component" if nspinor == 2 else "scalar (1-component)"
    if not bool(getattr(sym, "trs_allowed", True)):
        spinor += ", time reversal broken"
    rows = [("k grid", f"{grid[0]} x {grid[1]} x {grid[2]}, {shift_text}, {stored}"),
            ("bands", f"{int(n_bands)} ({electrons:.5f} electrons)"),
            ("spinor", spinor)]
    fft = getattr(wfn, "fft_grid", None)
    if fft is not None:
        fft = [int(v) for v in np.asarray(fft).reshape(-1)[:3]]
        rows.append(("FFT grid", " x ".join(str(v) for v in fft)))
    rows.append(("reader", str(getattr(wfn, "backend", "unknown"))))
    return [f"  Wavefunctions ({name})"] + kv_lines(rows, indent=4, width=LABEL)


def band_window_lines(band_slices, zeta_ranges, *, n_electrons: float) -> list[str]:
    """The BAND WINDOWS rows: which bands each part of the calculation uses."""
    b = band_slices
    # ``b4_logical`` is the physical loaded top (tagged_arrays.py); a
    # band-slices object without it (older producers, test stubs) means
    # no mesh padding, so the padded top is the logical top.
    b4_logical = int(getattr(b, "b4_logical", b.b4))
    rows = [
        ("occupied", f"{band_range(b.b0, b.b2)} ({n_electrons:.5f} electrons)"),
        ("QP matrix", f"{band_range(b.b0, b.b3)} (valence {band_range(b.b1, b.b2)}, "
                      f"conduction {band_range(b.b2, b.b3)})"),
        ("chi0 / W sum", band_range(b.b0, min(int(b.b4_chi), b4_logical))),
        ("Sigma sum", band_range(b.b0, min(int(b.b4_sigma), b4_logical))),
        ("zeta fit", f"left {band_range(*zeta_ranges[0])} x "
                     f"right {band_range(*zeta_ranges[1])}"),
        ("loaded at r_mu", band_range(b.b0, b4_logical)),
    ]
    if int(b.b4) != b4_logical:
        rows.append(("band carrier", f"{band_range(b.b0, b.b4)} "
                                     f"({int(b.b4) - b4_logical} zero-pad rows)"))
    return kv_lines(rows)


# ── RESULTS ──────────────────────────────────────────────────────────────────
def read_eqp_table(path, *, kpoints=None):
    """A BerkeleyGW ``eqp0.dat`` / ``eqp1.dat`` file read back.

    ``(k_labels, bands, e_dft_ev, e_qp_ev)``: the 1-based index of each k in
    ``kpoints`` (the run's stored list, matched by coordinates; file order when
    a k matches none), the band numbers, and the (nk, nb) DFT and QP columns
    in eV.  ``None`` when the file is missing or not a complete table.
    """
    if not path or not os.path.exists(path):
        return None
    ks, blocks = [], []
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line in stream:
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                try:
                    int(parts[0])
                except ValueError:                  # a k header: kx ky kz nb
                    ks.append([float(v) for v in parts[:3]])
                    blocks.append([])
                    continue
                if blocks and len(parts) >= 4:      # spin band E_DFT E_QP
                    blocks[-1].append((int(parts[1]), float(parts[2]), float(parts[3])))
    except (OSError, ValueError):
        return None
    if not blocks or not blocks[0] or any(len(b) != len(blocks[0]) for b in blocks):
        return None
    bands = np.asarray([row[0] for row in blocks[0]], dtype=np.int64)
    e_dft = np.asarray([[row[1] for row in block] for block in blocks], dtype=np.float64)
    e_qp = np.asarray([[row[2] for row in block] for block in blocks], dtype=np.float64)
    labels = np.arange(1, len(ks) + 1)
    if kpoints is not None and len(kpoints):
        ref = np.asarray(kpoints, dtype=np.float64).reshape(-1, 3)
        for i, k in enumerate(np.asarray(ks)):
            delta = ref - k[None, :]
            delta -= np.rint(delta)
            hits = np.flatnonzero(np.max(np.abs(delta), axis=1) < 1e-5)
            if hits.size:
                labels[i] = int(hits[0]) + 1
    return labels, bands, e_dft, e_qp


def _k_labels(nk: int, sym):
    """1-based IBZ labels of an energy array's k rows: the stored list as is,
    or each full-zone k's star (``sym.irr_idx_k``); ``None`` when neither fits."""
    if sym is not None and int(nk) == int(getattr(sym, "nk_red", -1)):
        return np.arange(1, int(nk) + 1)
    irr = getattr(sym, "irr_idx_k", None)
    if irr is not None and len(np.asarray(irr)) == int(nk):
        return np.asarray(irr, dtype=np.int64) + 1
    return None


def _band_edge(energies, column, labels, *, top: bool):
    values = np.asarray(energies, dtype=np.float64)[:, column]
    if not np.all(np.isfinite(values)):
        return None
    i = int(np.argmax(values) if top else np.argmin(values))
    return float(values[i]), int(labels[i]) if labels is not None else i + 1


def results_lines(rows, *, valence_band: int, ef_ev: float | None,
                  reference: str | None, correction_label: str | None = None) -> list[str]:
    """The RESULTS table: each row's VBM and CBM (k, band) and gap, in eV
    relative to E_F, then the gap correction and the legend.

    ``rows`` are ``(label, energies (nk, nb) eV, first band number, k labels)``;
    the first row is the reference (DFT) and the last the one the gap
    correction is quoted for.  ``valence_band`` is the 1-based top valence
    band; the CBM is the band above it.  ``ef_ev`` ``None`` measures from the
    midgap of the reference row.  A row whose band window misses an edge, or
    whose edge energies are not finite, is left out.  ``correction_label``
    names the last row in the gap-correction line (its row label otherwise).
    """
    edges = []
    for label, energies, first, labels in rows:
        energies = np.asarray(energies, dtype=np.float64)
        iv, ic = int(valence_band) - int(first), int(valence_band) + 1 - int(first)
        if energies.ndim != 2 or iv < 0 or ic >= energies.shape[1]:
            continue
        vbm = _band_edge(energies, iv, labels, top=True)
        cbm = _band_edge(energies, ic, labels, top=False)
        if vbm is None or cbm is None:
            continue
        edges.append((label, vbm, cbm))
    if not edges:
        return ["  no complete valence / conduction pair in the QP result"]
    if ef_ev is None:
        ef_ev = 0.5 * (edges[0][1][0] + edges[0][2][0])
        reference = "midgap of the DFT edges"
    title = f"band edges, eV relative to E_F ({reference or 'reference'})"
    width = max(44, len(title) + 2, *(len(label) + 2 for label, _, _ in edges))
    lines = [f"  {title:<{width}}{'VBM (ik, band)':<19}{'CBM (ik, band)':<18}{'gap':>7}"]
    ref_v, ref_c = edges[0][1][1], edges[0][2][1]
    vb, cb = int(valence_band), int(valence_band) + 1
    for label, (ev, kv), (ec, kc) in edges:
        cell_v = f"{ev - ef_ev:+.3f} ({kv}, {vb})" + ("*" if kv != ref_v else "")
        cell_c = f"{ec - ef_ev:+.3f} ({kc}, {cb})" + ("*" if kc != ref_c else "")
        lines.append(f"  {label:<{width}}{cell_v:<19}{cell_c:<18}{ec - ev:>7.3f}")
    if len(edges) > 1:
        first_gap = edges[0][2][0] - edges[0][1][0]
        last_gap = edges[-1][2][0] - edges[-1][1][0]
        lines += kv_lines([("gap correction", f"{last_gap - first_gap:+.3f} eV "
                            f"({correction_label or edges[-1][0]} against {edges[0][0]})")])
    lines.append("  each row gives its own VBM and CBM; an edge whose k differs from "
                 f"the {edges[0][0]} edge's is marked *")
    return lines


def sc_loop_line(run) -> str:
    """The RESULTS ``SC loop`` value: the final map's max|dE| and the map count."""
    maps = [m for m in run.maps if m.max_abs_ev is not None]
    final = maps[-1]
    n = len(maps)
    text = f"final max|dE| {float(final.max_abs_ev):.6f} eV after {n} map{'s' if n != 1 else ''}"
    if final.cutoff_ev is not None:
        text += f" (cutoff {float(final.cutoff_ev):.6f} eV)"
    return text


def sc_results_lines(dft_row, qsgw_row, run, *, valence_band: int, ef_ev: float | None,
                     reference: str | None) -> list[str]:
    """RESULTS of a self-consistent run: the DFT row and the accepted map's
    ``QSGW (map N, converged|not converged)`` row, then the ``SC loop`` line.

    ``dft_row`` / ``qsgw_row`` are :func:`results_lines` rows; ``run`` is the
    :class:`gw.sc_report.SCRun` (its final map names the row and says whether
    the system is a metal).  A metal has no VBM / CBM: its rows carry
    ``metal`` in the gap column and one line says why no edges are given.
    """
    final = run.final
    verdict = ("one map" if final.role == "one_shot" else
               "converged" if run.converged else "not converged")
    label = f"QSGW (map {int(final.call)}, {verdict})"
    loop = kv_lines([("SC loop", sc_loop_line(run))])
    if not final.metal:
        lines = results_lines([dft_row, (label, *qsgw_row[1:])], valence_band=valence_band,
                              ef_ev=ef_ev, reference=reference, correction_label="QSGW")
        return lines[:-1] + loop + lines[-1:] if len(lines) > 1 else lines + loop
    width = 44
    lines = [f"  {'band gap':<{width}}{'gap':>7}"]
    lines += [f"  {name:<{width}}{'metal':>7}" for name in ("DFT", label)]
    where = ("" if ef_ev is None else
             f" (E_F = {float(ef_ev):.5f} eV" + (f", {reference}" if reference else "") + ")")
    lines += kv_lines([("band edges", "none: a metal has no VBM or CBM" + where)])
    return lines + loop


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
    def __init__(self, path: str, *, runtime, debug: bool, stdout) -> None:
        super().__init__(path, runtime=runtime, debug=debug, stdout=stdout)
        self._timings_emitted = False
        self._coverage_shown = False
        self._quadrature_heading = False

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

    def input_files(self, *, config, wfn, sym, centroids=None) -> None:
        """INPUT FILES: one row per core input this deck reads, then the
        wavefunction file's own facts.  Only what is cheap is shown: the WFN
        header, the centroid set's closure verdict and the bands its header
        names, the kin_ion / dipole windows from their HDF5 attributes and
        dataset shapes (rank zero, no dataset read), and whether the other
        files exist (their windows are checked where they are read)."""
        grid = tuple(int(v) for v in np.asarray(wfn.kgrid).reshape(-1)[:3])
        n_full, n_red = int(sym.nk_tot), int(sym.nk_red)
        n_bands = int(np.asarray(wfn.energies).shape[-1])
        k_short = (f"{grid[0]}x{grid[1]}x{grid[2]}, "
                   + (f"{n_red} IBZ" if n_red != n_full else f"{n_full} k"))
        rows = [(os.path.basename(config.paths.wfn_file), band_range(0, n_bands),
                 k_short, str(int(wfn.nspinor)), "OK  (source of everything below)")]
        header_reads = self.rank == 0          # only rank zero prints the table
        centroid_file = config.paths.centroids_file
        if centroids is not None:
            closed = bool(getattr(centroids, "orbit_closed", True))
            rows.append((os.path.basename(centroid_file),
                         centroid_file_bands(centroid_file) if header_reads else "-", "-", "-",
                         f"OK  {int(centroids.n_rmu)} sites, orbit closed" if closed else
                         f"WARNING  {int(centroids.n_rmu)} sites, orbit not closed"))
        current = getattr(config.paths, "centroids_file_current", None)
        if current and not _same_path(current, centroid_file):
            rows.append((os.path.basename(current),
                         centroid_file_bands(current) if header_reads else "-", "-", "-",
                         "found" if os.path.exists(current) else "MISSING"))
        kin_ion = config.paths.kin_ion_file
        found = os.path.exists(kin_ion)
        rows.append((os.path.basename(kin_ion),
                     *(h5_input_cells(kin_ion, "kin_ion", grid=grid)
                       if found and header_reads else ("-", "-", "-")),
                     "found" if found else "MISSING"))
        dipole = os.path.join(config.input_dir, "dipole.h5")
        if os.path.exists(dipole):
            rows.append(("dipole.h5", *(h5_input_cells(dipole, "dipole", grid=grid)
                                        if header_reads else ("-", "-", "-")), "found"))
        transport = config.paths.parallel_transport_file
        if transport and os.path.exists(transport):
            rows.append((os.path.basename(transport), "-", "-", "-", "found"))
        self.heading("Input files")
        self.emit_lines(input_file_lines(rows))
        self.emit()
        self.emit_lines(wavefunction_lines(
            os.path.basename(config.paths.wfn_file), wfn=wfn, sym=sym, n_bands=n_bands))

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
        """BAND WINDOWS: which bands each part of the calculation uses."""
        self.heading("Band windows")
        self.emit_lines(band_window_lines(
            band_slices, zeta_ranges,
            n_electrons=float(getattr(wfn, "num_electrons", np.nan))))

    def sigma_coverage(self, *, config, band_slices, enk_dft_ry,
                       sigma_result, final: bool = False) -> None:
        """The dynamic Sigma grid against Sigma's resolved energy origin, as
        stage-3 rows (``Sigma grid``, ``on shell``, ``coverage``, ``band tail``)
        and the ``sigma.coverage`` record.  Shown once: the one-shot shows it
        under its stage (the same result), so the end-of-run call is then a
        no-op.  ``final`` (the end-of-run call) prints the rows in the
        QUADRATURE WINDOWS section (a self-consistent run's final map).

        The result owns this origin: for a metal it is the fixed-N chemical
        potential of the accepted map, while for an insulator it is the
        configured VBM or midgap.  Reconstructing it from ``wfn.efermi``
        would silently misreport every non-midgap calculation.
        """
        if not config.compute_mode.is_dynamic or self._coverage_shown:
            return
        self._coverage_shown = True
        from common.log import block, get_logger, record
        log = get_logger(__name__)
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
        omega_grid = np.asarray(() if _grid is None else _grid, dtype=np.float64)
        on_shell = sigma_result.sigma_c_on_shell is not None
        step = float(config.sigma.omega_step_ev)
        rows = [("Sigma grid",
                 f"{grid_lo:+.2f} .. {grid_hi:+.2f} eV about E_F = {ef_ev:.5f} eV "
                 f"({provenance}), step {step:.2f} eV, {int(omega_grid.size)} points")]
        if on_shell:
            # On shell the grid is only the covered range; Sigma was not
            # sampled on its points (docs/theory/sigma-on-shell.md).
            rows.append(("on shell", "covered range only"))
        fields = dict(efermi_ev=ef_ev, reference=str(provenance),
                      grid_ev=[grid_lo, grid_hi], step_ev=step,
                      points=int(omega_grid.size), on_shell=bool(on_shell),
                      dft_valence_ev=val_span, dft_conduction_ev=cond_span)
        if target:
            target_lo, target_hi = min(target), max(target)
            lower_margin = target_lo - grid_lo
            upper_margin = grid_hi - target_hi
            fields.update(margin_below_ev=lower_margin, margin_above_ev=upper_margin)
            if lower_margin >= 0.0 and upper_margin >= 0.0:
                fields["status"] = "COMPLETE"
                rows.append(("coverage", f"COMPLETE: margins {lower_margin:.2f} eV below "
                             f"and {upper_margin:.2f} eV above the requested bands"))
            else:
                fields["status"] = "INCOMPLETE"
                shortfalls = []
                if lower_margin < 0.0:
                    shortfalls.append(f"{-lower_margin:.5f} eV below")
                if upper_margin < 0.0:
                    shortfalls.append(f"{-upper_margin:.5f} eV above")
                rows.append(("coverage", "INCOMPLETE: short by " + " and ".join(shortfalls)
                             + " the requested bands"))
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

        estimator = (getattr(
            sigma_result, "band_extrapolation_estimator", None)
            or config.sigma.band_extrapolation_estimator)
        scheme = (getattr(sigma_result, "band_extrapolation_scheme", None)
                  or config.sigma.band_extrapolation_bracket_scheme)
        counts = getattr(sigma_result, "band_extrapolation_counts", None)
        raw = getattr(sigma_result, "sigma_c_at_dft_diag_unextrap_ev", None) is not None
        fields.update(band_tail=bool(config.sigma.band_extrapolation),
                      tail_estimator=str(estimator), tail_brackets=str(scheme),
                      tail_counts=[int(v) for v in counts] if counts else None,
                      raw_columns=bool(raw))
        if config.sigma.band_extrapolation:
            tail = str(estimator)
            if counts:
                tail += " on bands " + " / ".join(str(int(v)) for v in counts)
            if raw:
                tail += "; raw (untailed) columns kept in the table"
        else:
            tail = "off: Sigma_c is the band sum truncated at the Sigma window"
        rows.append(("band tail", tail))
        log.debug("Sigma coverage", extra=record("sigma.coverage", **fields))
        if not final:
            for label, value in rows:
                log.info(value, extra=block("sigma", label=label))
            return
        if not self._quadrature_heading:
            self._quadrature_heading = True
            self.heading("Quadrature windows")
        for label, value in rows:
            wrapped = wrap_value(value, first_col=LABEL + 2)
            self.emit_lines(kv_lines([(label, wrapped[0])]) + list(wrapped[1:]))

    def quadrature(self, *, self_consistent: bool = False) -> None:
        """The quadrature rules no stage showed (a self-consistent run's final
        Sigma plan, or rules recorded outside the one-shot stages), in eV.

        A self-consistent run shows its chi0 rules at the start of its SC
        stage, so they are not repeated here (the ``screening.chi0_rules``
        record keeps every rule); its Sigma table is the final map's plan."""
        from common.log import get_logger, record
        rules = quadrature_log.chi_rules()
        geometry, plans = quadrature_log.sigma_plan()
        # A table its stage printed (one-shot: chi0 under Screening, the one
        # Sigma plan under Self-energy) is not repeated here.
        if quadrature_log.shown("chi") or self_consistent:
            if rules and self_consistent:
                get_logger(__name__).debug("chi0 rules (all maps)", extra=record(
                    "screening.chi0_rules_all_maps", rules=rules))
            rules = []
        if quadrature_log.shown("sigma") and plans <= 1:
            geometry = None
        if not rules and geometry is None:
            return
        self._quadrature_heading = True
        self.heading("Quadrature windows")
        if rules:
            self.emit(f"  {'chi0 rules':<15}{'rule':<7} {'where':<35}{'nodes':>5}   "
                      f"{'max err':>7}   source")
            for row in rules:
                self.emit("  " + " " * 15 + chi_rule_detail_row(row))
        if geometry is not None:
            head, rows = sigma_rule_detail_lines(geometry)
            if self_consistent:
                head = "final map: " + head
            elif plans > 1:
                head += f" (last of {plans} plans)"
            wrapped = wrap_value(head, first_col=LABEL + 2)
            self.emit_lines(kv_lines([("Sigma rules", wrapped[0])]) + list(wrapped[1:]))
            self.emit_lines("  " + row for row in rows)

    def results(self, *, band_slices, e_dft_ry, e_qp_ry, config, sym, wfn,
                sigma_result=None, sc_run=None) -> None:
        """RESULTS: the band edges of DFT, eqp0, eqp1 and the full-matrix H(QP),
        each relative to E_F, and the gap correction.

        The eqp0 / eqp1 rows are read back from the files this run wrote (rank
        zero, which wrote them); the DFT row is those files' DFT column, and
        the full-matrix row is ``e_qp_ry`` (the eigenvalues already in hand).
        ``sc_run`` (``gw.sc_report.last_run()``, a self-consistent run only)
        replaces the three QP rows with the accepted map's QSGW row
        (:func:`sc_results_lines`).
        """
        # Every rank reads its arrays to the host, as the gap block did; only
        # rank zero (which wrote the eqp files) reads them back and prints.
        dft = np.asarray(e_dft_ry, dtype=np.float64) * RYD_TO_EV
        qp = np.asarray(e_qp_ry, dtype=np.float64) * RYD_TO_EV
        if self.rank != 0:
            return
        b = band_slices
        valence = int(b.b2)                      # 1-based top valence band
        first = int(b.b0) + 1
        kpoints = np.asarray(wfn.kpoints, dtype=np.float64)
        rows: list = []
        dft_row = None
        for label, path in (("eqp0 (Sigma at E_DFT)", config.paths.eqp0_file),
                            ("eqp1 (Z-linearized)", config.paths.eqp1_file)):
            table = read_eqp_table(path, kpoints=kpoints)
            if table is None:
                continue
            labels, bands, e_dft, e_qp = table
            if dft_row is None:
                dft_row = ("DFT", e_dft, int(bands[0]), labels)
            rows.append((label, e_qp, int(bands[0]), labels))
        if dft_row is None:
            dft_row = ("DFT", dft, first, _k_labels(dft.shape[0], sym))
        rows.append(("full-matrix H(QP)", qp, first, _k_labels(qp.shape[0], sym)))
        rows.insert(0, dft_row)
        ef = getattr(sigma_result, "efermi_dft_ev", None)
        reference = (getattr(sigma_result, "omega_reference_provenance", None)
                     or getattr(getattr(config, "sigma", None), "fermi_reference", None))
        self.heading("Results")
        if sc_run is not None and sc_run.final is not None:
            # SC: eqp0 = eqp1 = the accepted map's spectrum (and e_qp_ry is the
            # same numbers), so one QSGW row against DFT.
            self.emit_lines(sc_results_lines(
                rows[0], rows[1], sc_run, valence_band=valence,
                ef_ev=None if ef is None else float(ef),
                reference=str(reference) if reference else None))
            return
        self.emit_lines(results_lines(
            rows, valence_band=valence,
            ef_ev=None if ef is None else float(ef),
            reference=str(reference) if reference else None))

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

        def outer(*names):
            """The named sections at any depth, less those nested in one of them
            (a stage section a manual stage now holds is no longer top-level)."""
            wanted = set(names)
            return total(lambda r: r["name"] in wanted and not any(
                str(parent) in wanted for parent in tuple(r.get("path", ()))[:-1]))

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
        # The minimax rule, the head response and the W persist run inside
        # the Screening stage of a one-shot run; their own rows own them.
        minimax = outer("gw_jax.minimax_quadrature")
        w_persist = outer("gw_jax.persist_w0", "gw_jax.static_head")
        inside = lambda band: _Band(sum(
            float(r["inclusive"]) for r in rows
            if tuple(r.get("path", ()))[:1] == ("gw_jax.screening",)
            and tuple(r.get("path", ())) in band.plus), plus=[
                p for p in band.plus if p[:1] == ("gw_jax.screening",)])
        screening_support = (screening_total - chi0 - w_screen
                             - inside(minimax) - inside(w_persist)
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
            ("minimax quadrature", minimax),
            ("chi0", chi0),
            ("W", w_screen),
            *screening_details,
            ("spole other" if spole_rows else "screening support", screening_support),
            ("W persist + q0 head", w_persist),
            ("SC W response", sc_w_response),
            ("Sigma rule plan", sigma_plan),
            ("SC Sigma rule refit", sigma_refit),
            # One row for the sweep; its setup / compile / window split is
            # the timing.sigma_tau record.
            ("Sigma tau sweep", sigma_sweep),
            *sigma_details,
            ("Sigma other", sigma_other),
            ("mean-field load", outer("gw_jax.kin_ion_load")),
            ("QP solve + diagonalize", outer("gw_jax.solve_qp", "gw_jax.qp_eigh")),
            ("result writes", outer("gw_jax.output")),
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
        from common.log import get_logger, record
        from lxkit import memory
        log = get_logger(__name__)
        # The clock notes, the Sigma tau split and the table itself are the
        # sidecar's; the main log keeps the table and one line pointing there.
        log.debug("timing notes: " + " | ".join(notes),
                  extra=record("timing.notes", notes=notes))
        if any(r["name"].startswith("tau.") for r in rows):
            split = partition("sigma.tau_sweep", ("tau.",), "Sigma tau")
            log.debug("Sigma tau split", extra=record("timing.sigma_tau", rows=[
                {"label": name, "seconds": float(band)} for name, band in split]))
        log.debug("timing table", extra=record("timing.stages", wall_s=float(wall), rows=[
            {"label": row.label, "depth": row.depth, "seconds": row.seconds,
             "device_gb": row.device_gb} for row in table]))
        budget = memory.run().get("budget", 0.0) / 1e9
        self.heading("Timing")
        self.emit_lines(timing_lines(
            table, wall=float(wall), budget_gb=budget or None,
            device_peak=(top.device_gb, top.label) if top is not None else None,
            host_peak=((float(np.nanmax(host_run)) / 1e9, float(np.nanmin(host_run)) / 1e9)
                       if host_run is not None else None),
            n_ranks=n_ranks))
        self._stage_memory(rows, stages, wall=wall)
        self._record_routes()
        sidecar = os.path.splitext(os.path.basename(self.path))[0] + ".json"
        self.emit(f"  per-stage memory, priced stages and notes on fused clocks: "
                  f"{sidecar} (timing, memory)")

    @staticmethod
    def _record_routes() -> None:
        """Every memory-driven route the run took and how often
        (``lxkit.memory.routes``), as the ``memory.routes`` record; a route
        that warned was printed where it happened."""
        from common.log import get_logger, record
        from lxkit import memory
        taken = memory.routes()
        if taken:
            get_logger(__name__).debug("routes taken", extra=record(
                "memory.routes", routes=taken))

    def _stage_memory(self, rows, stages, *, wall) -> None:
        """The device peak of every major stage against its planner's price, and
        its host peak, as JSON-sidecar records (``memory.stages``,
        ``memory.priced``, ``memory.sizing``); the main log's TIMING block keeps
        the budget, the device peak and the host peak.

        A row's peak is the pool high-water mark over the timing nodes it owns
        (``common.timing``; max and min over ranks).  A priced stage's receipt
        (``lxkit.memory``) holds its own measured figure: the high-water from the
        price's booking to the stage's exit, after waits on the work around it.

        The host columns are the process resident high-water mark (``ru_maxrss``,
        which never falls) at the row's last exit and the rise the row's own
        intervals caused, each the max over ranks.  No planner prices the host.
        """
        from common import gpu_utils, timing
        from common.log import get_logger, record
        from lxkit import memory
        log = get_logger(__name__)
        paths = [tuple(r.get("path", (r["name"],))) for r in rows]
        selfs = [_rank_peaks(r, "peak_self") for r in rows]
        hosts = [_rank_peaks(r, "host") for r in rows]
        rises = [_rank_peaks(r, "host_rise") for r in rows]
        if not any(np.any(np.isfinite(v)) for v in selfs + hosts):
            return
        row_peak, n_ranks, _host_run = _peak_reader(rows)
        pad = lambda v: v if len(v) == n_ranks else np.full(n_ranks, np.nan)
        gb = lambda value: (None if value is None or not np.isfinite(value)
                            else round(float(value) / 1e9, 4))

        def row_host(band):
            """Max over ranks of the row's host mark and of the rise it caused (bytes)."""
            mine = [i for i, p in enumerate(paths)
                    if band.owns(p) and np.any(np.isfinite(hosts[i]))]
            if not mine:
                return None, None
            mark = np.nanmax([np.nanmax(hosts[i]) for i in mine])
            rise = np.nanmax(np.nansum([pad(rises[i]) for i in mine], axis=0))
            return mark, rise

        table = []
        for name, band in stages:
            peak, setter = row_peak(band)
            mark, rise = row_host(band)
            if peak is None and mark is None:
                continue
            finite = peak is not None and np.any(np.isfinite(peak))
            table.append({
                "stage": name,
                "peak_max_gb": gb(np.nanmax(peak)) if finite else None,
                "peak_min_gb": gb(np.nanmin(peak)) if finite else None,
                "host_gb": gb(mark), "host_rise_gb": gb(rise),
                "set_by": " > ".join(setter[-2:])})
        log.debug("per-stage memory", extra=record(
            "memory.stages", ranks=n_ranks, rows=table,
            held_on_host_gb={name: gb(nbytes)
                             for name, nbytes in gpu_utils.host_holds().items()},
            # Constructor sub-stages with their own pool high-water
            # (TimingNode.peak): what the reduction, passivity and head prices
            # are calibrated against.
            substages=[line.strip() for line in _substage_peak_lines(paths, rows)]))
        priced = []
        for row in memory.receipts():
            peak = _rank_peaks(row, "measured")
            peak = np.where(peak >= 0, peak, np.nan)
            worst = float(np.nanmax(peak)) if np.any(np.isfinite(peak)) else -1.0
            priced.append({"line": memory.receipt_line(row["stage"], row["price"], worst,
                                                       row["exact"], row["source"]),
                           "count": int(row["count"])})
        if priced:
            log.debug("priced stages", extra=record("memory.priced", stages=priced))
        reads, seconds = timing.instrument_cost()
        log.debug("memory sizing", extra=record(
            "memory.sizing", minimum_to_complete=memory.sizing()["line"],
            findings=list(memory.findings()) if memory.STRICT else []))
        log.debug("memory instrument", extra=record(
            "memory.instrument", source=memory.instrument()["source"] or "none",
            mode="calibrate" if memory.CALIBRATE else "production",
            boundary_reads=int(reads), boundary_seconds=float(seconds),
            wall_fraction=float(seconds / wall) if wall else 0.0))


__all__ = ["GWProductionReport"]
