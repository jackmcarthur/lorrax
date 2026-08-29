"""psp/pseudos.py — Pseudopotential loading, element lookup, atom assignments.

Extracted from get_DFT_mtxels.py to give a clean, minimal entry point
for any code that needs pseudopotentials.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Periodic table
# ---------------------------------------------------------------------------

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
    "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es "
    "Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Fl Lv Ts Og"
).split()


def symbol_to_Z(symbol: str) -> int | None:
    """Element symbol → atomic number, or None if not found."""
    try:
        return _SYMBOLS.index(symbol.strip().capitalize()) + 1
    except ValueError:
        return None


# Backwards-compatible alias
_symbol_to_Z = symbol_to_Z


# ---------------------------------------------------------------------------
# PSP loading
# ---------------------------------------------------------------------------

def load_pseudopotentials(work_dir: str = ".") -> dict:
    """Load all *.upf pseudopotential files from a directory.

    Returns dict mapping element symbol → parsed UPF object.
    """
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass

    upf_files = glob.glob(os.path.join(work_dir, "*.upf"))
    if not upf_files:
        upf_files = glob.glob(os.path.join(work_dir, "*.UPF"))
    if not upf_files:
        print(f"No pseudopotentials (*.upf) found in {work_dir}")
        return {}

    pseudos = {}
    for upf_file in sorted(upf_files):
        try:
            pseudo = load_upf(upf_file)
            pseudo = normalize_dataclass(pseudo)
            setattr(pseudo, "_source_path", upf_file)
            element = pseudo.pp_header.element.strip()
            pseudos[element] = pseudo
        except Exception as e:
            print(f"Warning: failed to load {os.path.basename(upf_file)}: {e}")
    return pseudos


# ---------------------------------------------------------------------------
# Atom ↔ pseudopotential assignment
# ---------------------------------------------------------------------------

@dataclass
class AtomPP:
    index: int
    atomic_number: int
    element: str
    position: np.ndarray
    pseudo: object | None


def build_atom_pp_assignments(
    atom_positions: np.ndarray,
    atom_types: np.ndarray,
    pseudos: dict,
) -> list[AtomPP]:
    """Map each atom in the cell to its pseudopotential."""
    z_to_pseudo: dict[int, object] = {}
    for element, pseudo in pseudos.items():
        Z = symbol_to_Z(element)
        if Z is not None:
            z_to_pseudo[Z] = pseudo

    assignments = []
    for i in range(atom_positions.shape[0]):
        Z = int(atom_types[i])
        pseudo = z_to_pseudo.get(Z)
        elem = (getattr(pseudo.pp_header, "element", str(Z)).strip()
                if pseudo is not None else str(Z))
        assignments.append(AtomPP(
            index=i, atomic_number=Z, element=elem,
            position=atom_positions[i], pseudo=pseudo))
    return assignments


def print_atomic_structure(wfn, pseudos: dict) -> None:
    """Print compact structure summary and pseudopotential table."""
    print(f"\nStructure: nat={wfn.nat}, volume={wfn.cell_volume:.4f} bohr³, "
          f"alat={wfn.alat:.6f} bohr")
    if not pseudos:
        print("Pseudopotentials: none found\n")
        return
    print("Pseudopotentials:")
    print(f"  {'Elem':<6} {'Z_val':>5} {'n_proj':>6}  file")
    for element, pseudo in pseudos.items():
        n_proj = pseudo.pp_header.number_of_proj
        z_val = pseudo.pp_header.z_valence
        fname = os.path.basename(getattr(pseudo, "_source_path", ""))
        print(f"  {element:<6} {z_val:>5} {n_proj:>6}  {fname}")
    print()


def _generator_tag(pseudo) -> str:
    """Short (≤15 char) generator/set tag for the report block, or "".

    Derived from PP_HEADER's ``generated`` attribute and the free-text
    PP_INFO body.  Only patterns this repo's staged sets actually carry
    are recognized (ONCVPSP with or without an adjacent version number;
    the PseudoDojo token); anything else yields "" and the tag is simply
    omitted — a guess here would be provenance fiction.
    """
    hdr = getattr(pseudo, "pp_header", None)
    info = getattr(getattr(pseudo, "pp_info", None), "value", None) or ""
    text = f"{getattr(hdr, 'generated', None) or ''} {info[:4000]}"
    m = re.search(r"ONCVPSP[-\s]*v?(\d+\.\d+\.\d+)", text)
    if m is None and "ONCVPSP" in text:
        # ONCVPSP's own PP_INFO prose puts the number two lines below the
        # name ("... version 3.3.0 08/16/2017 by D. R. Hamann").
        m = re.search(r"version\s+(\d+\.\d+\.\d+)", text)
        if m is None:
            return "ONCVPSP"
    if m is not None:
        return f"ONCVPSP-{m.group(1)}"[:15]
    if "PseudoDojo" in text:
        return "PseudoDojo"
    return ""


def pseudo_summary_lines(pseudos: dict) -> list[str]:
    """ONE log line per species for the Pseudopotentials report block.

        Si             : Si.upf — scalar-relativistic, z_val 4, NLCC yes (ONCVPSP-3.3.0)

    The SR/FR verdict follows the projector policy (radial/
    build_projectors_qe.pseudo_has_j_channels): a beta carrying an
    explicit jjj is what makes a file j-resolved — the PP_HEADER
    ``relativistic``/``has_so`` attributes are advisory, so when they
    disagree with the betas the disagreement is printed and the betas
    win.  The trailing parenthetical is :func:`_generator_tag`, omitted
    when nothing recognizable is in the file.
    """
    from psp.radial.build_projectors_qe import pseudo_has_j_channels

    lines: list[str] = []
    for element in sorted(pseudos):
        pseudo = pseudos[element]
        hdr = pseudo.pp_header
        fname = os.path.basename(getattr(pseudo, "_source_path", "?"))
        has_j = pseudo_has_j_channels(pseudo)
        rel_attr = getattr(hdr, "relativistic", None)
        rel_attr = getattr(rel_attr, "value", rel_attr)  # enum → str
        kind = "fully-relativistic" if has_j else "scalar-relativistic"
        header_says_full = str(rel_attr).strip().lower() == "full"
        if header_says_full != has_j and rel_attr is not None:
            kind += f' [header relativistic="{rel_attr}"; beta channels win]'
        nlcc = "yes" if getattr(pseudo, "pp_nlcc", None) is not None else "no"
        tag = _generator_tag(pseudo)
        lines.append(
            f"{element:<15}: {fname} — {kind}, "
            f"z_val {float(hdr.z_valence):g}, NLCC {nlcc}"
            + (f" ({tag})" if tag else ""))
    if not lines:
        lines.append("(none loaded)")
    return lines
