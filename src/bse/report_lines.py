"""Pure text blocks of the BSE and exciton-band main logs: facts in, lines out.

Nothing here reads a file, touches a device or decides physics; the drivers
(``bse.bse_jax``, ``bse.exciton_bands``) hand in host values they already hold.
The full htransform receipts behind :func:`galerkin_rows` and
:func:`locality_rows` go to the JSON sidecar; the main log keeps one row each.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from common.preprocessing_output import (galerkin_rows, kv_block_lines,
                                         locality_rows, results_lines)

#: Path points the exciton-band RESULTS table lists in full; a longer path
#: lists its labelled nodes and points to the ``.dat`` file.
PATH_ROWS_MAX = 40
#: Levels per row of the exciton-band table.
PATH_LEVELS_SHOWN = 4


# ── BSE ──────────────────────────────────────────────────────────────────────
#: The main log lists every root up to this many.  Past it, the first
#: ``BSE_ROOTS_FIRST`` and the ``BSE_ROOTS_BRIGHT`` brightest; every root is
#: in BerkeleyGW's ``eigenvalues*.dat`` and on the rank-0 stdout.
BSE_ROOTS_LISTED = 20
BSE_ROOTS_FIRST = 16
BSE_ROOTS_BRIGHT = 5

_STATE_ROW = re.compile(r"^  S(\d{3,})  ")


def exciton_energy_lines(energies_ev, states=None) -> list[str]:
    """The exciton table: ``  state       energy (eV)`` and one ``  S001  <E>``
    row per root (``states``: the 0-based roots to list, default all).  The
    row text is a harness contract (``tests/hsuite/chain.py``:
    ``^\\s*S\\d+\\s+([0-9.+-]+)\\s*$`` on the rank-0 stdout)."""
    energies = np.asarray(energies_ev, dtype=np.float64)
    lines = ["  state       energy (eV)"]
    for index in (range(energies.size) if states is None else states):
        lines.append(f"  S{int(index) + 1:03d}  {float(energies[int(index)]):15.5f}")
    return lines


def oscillator_lines(dipoles, n_k: int, states=None) -> list[str]:
    """Per-state ``|D_a|^2 / N_k`` (bohr^2 per cell) for ``D = <0|r_a|S>``
    (``states``: the 0-based roots to list, default all).

    Every row carries three numbers in e-format, so no row reads as an
    exciton-energy row."""
    d = np.asarray(dipoles, dtype=np.complex128)
    strength = np.abs(d) ** 2 / max(int(n_k), 1)
    lines = [f"  {'state':<7}{'|D_x|^2/N_k':>14}{'|D_y|^2/N_k':>14}"
             f"{'|D_z|^2/N_k':>14}   (bohr^2 per cell)"]
    for index in (range(strength.shape[0]) if states is None else states):
        lines.append(f"  S{int(index) + 1:03d}  "
                     + "".join(f"{float(v):14.4e}" for v in strength[int(index), :3]))
    return lines


def brightest_states(dipoles, count: int = BSE_ROOTS_BRIGHT) -> list[int]:
    """The 0-based roots of the ``count`` largest ``sum_a |D_a|^2``, brightest
    first (ties in root order)."""
    strength = np.sum(np.abs(np.asarray(dipoles, dtype=np.complex128)) ** 2, axis=1)
    order = sorted(range(strength.size), key=lambda i: (-float(strength[i]), i))
    return order[:int(count)]


def brightest_lines(energies_ev, dipoles, n_k: int, states) -> list[str]:
    """The brightest roots: the energy and ``|D_a|^2 / N_k`` of each."""
    energies = np.asarray(energies_ev, dtype=np.float64)
    strength = np.abs(np.asarray(dipoles, dtype=np.complex128)) ** 2 / max(int(n_k), 1)
    lines = [f"  {'state':<7}{'energy (eV)':>15}{'|D_x|^2/N_k':>14}{'|D_y|^2/N_k':>14}"
             f"{'|D_z|^2/N_k':>14}   (bohr^2 per cell)"]
    for i in states:
        lines.append(f"  S{int(i) + 1:03d}  {float(energies[int(i)]):15.5f}"
                     + "".join(f"{float(v):14.4e}" for v in strength[int(i), :3]))
    return lines


def bse_results_lines(energies_ev, *, gap=None, dipoles=None, n_k: int = 1,
                      helicity: Sequence[str] = (), eigenvectors: str | None = None,
                      roots_file: str | None = None, trim: bool = True,
                      extra_rows: Sequence[tuple[str, str]] = ()) -> list[str]:
    """The BSE RESULTS block.

    ``gap`` is ``(E_gap eV, k index 1-based, "QP" | "DFT")``, the lowest direct
    transition of the solved window; the binding row is ``E_gap - E(S1)``.
    ``dipoles`` (``(n_roots, 3)``) adds the oscillator table, ``helicity``
    the ``--helicity-table`` lines; ``eigenvectors`` says what was written,
    ``roots_file`` which file holds every root; ``extra_rows`` follow.

    With ``trim`` and more than :data:`BSE_ROOTS_LISTED` roots the tables list
    the first :data:`BSE_ROOTS_FIRST`, then the :data:`BSE_ROOTS_BRIGHT`
    brightest by ``sum_a |D_a|^2`` (when the dipoles are known).  The rank-0
    stdout carries the untrimmed block (``trim=False``).
    """
    energies_ev = np.asarray(energies_ev, dtype=np.float64)
    n_roots = int(energies_ev.size)
    states = bright = None
    if trim and n_roots > BSE_ROOTS_LISTED:
        states = range(BSE_ROOTS_FIRST)
        if dipoles is not None:
            bright = brightest_states(dipoles)
    table = exciton_energy_lines(energies_ev, states)
    if states is not None:
        table.append(f"  [the first {BSE_ROOTS_FIRST} of {n_roots} roots; all are in "
                     f"{roots_file or 'the rank-0 stdout'}"
                     + ("" if dipoles is not None else
                        "; the brightest need --dipole") + "]")
    if dipoles is not None:
        table += [""] + oscillator_lines(dipoles, n_k, states)
    if helicity:
        shown = None if states is None else set(states) | set(bright or ())
        table += [""] + [str(line) for line in helicity if shown is None
                         or not _STATE_ROW.match(str(line))
                         or int(_STATE_ROW.match(str(line)).group(1)) - 1 in shown]
    if bright is not None:
        table += ["", f"  brightest {len(bright)} of {n_roots} by "
                      "|D_x|^2 + |D_y|^2 + |D_z|^2"]
        table += brightest_lines(energies_ev, dipoles, n_k, bright)
    rows = []
    if gap is not None:
        gap_ev, ik, kind = gap
        rows.append((f"{kind} gap", f"{float(gap_ev):.5f} eV, the lowest direct "
                                    f"transition in the window (k {int(ik)})"))
        if energies_ev.size:
            rows.append(("binding (S1)", f"{float(gap_ev) - float(energies_ev[0]):.5f} eV "
                                         f"below the {kind} gap"))
    rows.append(("eigenvectors", eigenvectors or "not written (--write-eigs)"))
    if roots_file:
        rows.append(("all roots", f"{n_roots} in {roots_file}, BerkeleyGW's layout"))
    rows += [(str(label), str(value)) for label, value in extra_rows]
    return results_lines(rows, table=table)


def window_direct_gap(eps_v, eps_c, *, n_val: int, n_cond: int):
    """``(E_gap Ry, k index 1-based)``: the lowest direct transition of the
    window's physical bands (host arrays ``(nk, n_val_pad)``, ``(nk,
    n_cond_pad)``; the pad columns are dropped), or ``None``."""
    v = np.asarray(eps_v, dtype=np.float64)[:, :int(n_val)]
    c = np.asarray(eps_c, dtype=np.float64)[:, :int(n_cond)]
    if v.size == 0 or c.size == 0 or v.shape[0] != c.shape[0]:
        return None
    direct = c.min(axis=1) - v.max(axis=1)
    ik = int(np.argmin(direct))
    return float(direct[ik]), ik + 1


# ── exciton bands ────────────────────────────────────────────────────────────
def path_node_lines(Qpath, node_idx, node_labels) -> list[str]:
    """The labelled Q-path nodes, one line each (a stage row's table)."""
    points = np.asarray(Qpath, dtype=np.float64)
    lines = []
    for inode, (idx, label) in enumerate(zip(node_idx, node_labels), start=1):
        q = points[int(idx)] + 0.0
        q[np.abs(q) < 5e-6] = 0.0
        lines.append(f"N{inode:02d}  {label or '-':>3}  Q = ({q[0]: .5f} {q[1]: .5f} "
                     f"{q[2]: .5f})  path index {int(idx) + 1}")
    return lines


def exciton_band_table(path_ev, Qpath, node_idx, node_labels, *,
                       max_rows: int = PATH_ROWS_MAX,
                       shown: int = PATH_LEVELS_SHOWN) -> list[str]:
    """One row per path point (or per labelled node on a long path): the
    point, its label and its lowest exciton levels in eV."""
    levels = np.atleast_2d(np.asarray(path_ev, dtype=np.float64))
    points = np.asarray(Qpath, dtype=np.float64)
    labels = {int(i): (lbl or "") for i, lbl in zip(node_idx, node_labels)}
    n_q, n_eig = levels.shape
    rows = (range(n_q) if n_q <= max_rows
            else sorted(labels) or range(min(n_q, max_rows)))
    width = min(n_eig, shown)
    head = (f"  {'iQ':>4}  {'':>3}  {'Qx':>8} {'Qy':>8} {'Qz':>8}   "
            + "".join(f"{'E_' + str(j + 1):>10}" for j in range(width))
            + ("   ..." if n_eig > width else "") + "   (eV)")
    lines = [head]
    for iq in rows:
        q = points[iq] + 0.0
        q[np.abs(q) < 5e-6] = 0.0
        lines.append(f"  {iq + 1:>4}  {labels.get(iq, ''):>3}  {q[0]:8.5f} {q[1]:8.5f} "
                     f"{q[2]:8.5f}   " + "".join(f"{v:10.5f}" for v in levels[iq, :width]))
    if n_q > max_rows:
        lines.append(f"  [labelled nodes only; all {n_q} path points are in the .dat file]")
    return lines


def exciton_band_results_lines(path_ev, Qpath, node_idx, node_labels, *,
                               n_extra: int = 0, hermiticity: str,
                               certificate: str, files: str) -> list[str]:
    """The exciton-band RESULTS block: the per-Q table, then the lowest level,
    the energy range, the alpha-Hermiticity verdict, the refit certificate
    and the files written."""
    levels = np.atleast_2d(np.asarray(path_ev, dtype=np.float64))
    n_q, n_eig = levels.shape
    labels = {int(i): (lbl or "") for i, lbl in zip(node_idx, node_labels)}
    iq, _ = np.unravel_index(int(np.argmin(levels)), levels.shape)
    where = labels.get(int(iq)) or f"iQ {int(iq) + 1}"
    rows = [
        ("levels", f"{n_eig} per Q at {n_q} path points"
                   + (f" (+{int(n_extra)} extra Q, in the .dat file)" if n_extra else "")),
        ("lowest E_S", f"{float(levels.min()):.5f} eV at {where}"
                       + (f" (iQ {int(iq) + 1})" if labels.get(int(iq)) else "")),
        ("energy range", f"{float(levels.min()):.5f} .. {float(levels.max()):.5f} eV"),
        ("Hermiticity", hermiticity),
        ("refit check", certificate),
        ("written", files),
    ]
    return results_lines(rows, table=exciton_band_table(
        levels, Qpath, node_idx, node_labels))


__all__ = [
    "BSE_ROOTS_BRIGHT",
    "BSE_ROOTS_FIRST",
    "BSE_ROOTS_LISTED",
    "PATH_LEVELS_SHOWN",
    "PATH_ROWS_MAX",
    "brightest_lines",
    "brightest_states",
    "bse_results_lines",
    "exciton_band_results_lines",
    "exciton_band_table",
    "exciton_energy_lines",
    "galerkin_rows",
    "kv_block_lines",
    "locality_rows",
    "oscillator_lines",
    "path_node_lines",
    "window_direct_gap",
]
