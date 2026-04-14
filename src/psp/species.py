"""psp/species.py — Extract radial data from UPF pseudopotentials.

One clean pass: UPF object → SpeciesData with all radial functions,
angular quantum numbers, and D-matrix.  No try/except, no string matching.
Fails hard on malformed UPF — that's a pseudopotential bug, not ours.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from psp.pseudos import symbol_to_Z


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SpeciesData:
    """All radial data for one atomic species, ready for Hankel transforms."""
    element: str
    z_valence: float
    z_atomic: int

    # Radial grid (shared by all channels)
    r: np.ndarray           # (n_r,) bohr
    rab: np.ndarray         # (n_r,) dr weights (QE log grid: rab = r * dx)

    # l=0 channels
    vloc_r: np.ndarray      # (n_r,) full local potential in Ry (includes -Z/r tail)
    rho_core_r: np.ndarray  # (n_r,) NLCC core charge (zeros if no NLCC)
    has_nlcc: bool

    # Nonlocal projectors: one entry per beta function
    n_proj: int
    beta_r: np.ndarray      # (n_proj, n_r) β(r)/r radial functions
    proj_l: np.ndarray      # (n_proj,) int — angular momentum per projector
    proj_j: np.ndarray      # (n_proj,) float — total angular momentum (SO)
    dij: np.ndarray         # (n_proj, n_proj) D-matrix in Ry
    nspinor: int            # 1 (scalar) or 2 (FR with SOC)


def extract_species(pseudos: dict, nspinor: int = 2) -> list[SpeciesData]:
    """Extract SpeciesData from all loaded pseudopotentials.

    Parameters
    ----------
    pseudos : dict from load_pseudopotentials (element → UPF object)
    nspinor : 1 for scalar-relativistic, 2 for fully-relativistic (SOC)
    """
    species = []
    for element, pseudo in pseudos.items():
        hdr = pseudo.pp_header
        mesh = pseudo.pp_mesh

        r = np.asarray(mesh.pp_r.value, dtype=np.float64)
        rab = np.asarray(mesh.pp_rab.value, dtype=np.float64)
        z_val = float(hdr.z_valence)
        z_at = symbol_to_Z(element)

        # Local potential
        vloc_r = np.asarray(pseudo.pp_local.value, dtype=np.float64)

        # NLCC core charge
        has_nlcc = bool(getattr(hdr, "core_correction", False))
        if has_nlcc and hasattr(pseudo, "pp_nlcc") and pseudo.pp_nlcc is not None:
            rho_core_r = np.asarray(pseudo.pp_nlcc.value, dtype=np.float64)
        else:
            rho_core_r = np.zeros_like(r)
            has_nlcc = False

        # Nonlocal projectors
        pp_nl = pseudo.pp_nonlocal
        n_proj = int(hdr.number_of_proj)
        beta_r = np.zeros((n_proj, len(r)), dtype=np.float64)
        proj_l = np.zeros(n_proj, dtype=np.int32)
        proj_j = np.zeros(n_proj, dtype=np.float64)

        for ip in range(n_proj):
            beta_obj = pp_nl.pp_beta[ip]
            raw = np.asarray(beta_obj.value, dtype=np.float64)
            # UPF stores β(r), we want β(r)/r for the Hankel transform
            safe_r = np.where(r > 0, r, 1.0)
            beta_r[ip] = np.where(r > 0, raw / safe_r, 0.0)
            proj_l[ip] = int(beta_obj.angular_momentum)
            proj_j[ip] = float(getattr(beta_obj, "total_angular_momentum",
                                        proj_l[ip] + 0.5))

        # D-matrix
        dij_flat = np.asarray(pp_nl.pp_dij.value, dtype=np.float64)
        dij = dij_flat.reshape(n_proj, n_proj)

        species.append(SpeciesData(
            element=element, z_valence=z_val, z_atomic=z_at,
            r=r, rab=rab,
            vloc_r=vloc_r, rho_core_r=rho_core_r, has_nlcc=has_nlcc,
            n_proj=n_proj, beta_r=beta_r, proj_l=proj_l, proj_j=proj_j,
            dij=dij, nspinor=nspinor,
        ))
    return species


def build_atom_species_map(crystal, species_list: list[SpeciesData]):
    """Map atoms in the crystal to species indices.

    Returns
    -------
    species_natoms : (n_species,) int
    species_tau : (n_species, max_atoms, 3) crystal coordinates
    """
    atom_types = np.asarray(crystal.atom_types, dtype=int)
    atom_crys = np.asarray(crystal.atom_crys, dtype=np.float64)

    z_to_idx = {sp.z_atomic: i for i, sp in enumerate(species_list)}
    n_species = len(species_list)

    atoms_per = [[] for _ in range(n_species)]
    for iat in range(len(atom_types)):
        idx = z_to_idx.get(int(atom_types[iat]))
        if idx is not None:
            atoms_per[idx].append(atom_crys[iat])

    max_atoms = max((len(a) for a in atoms_per), default=1)
    species_tau = np.zeros((n_species, max_atoms, 3), dtype=np.float64)
    species_natoms = np.zeros(n_species, dtype=np.int32)
    for i, ats in enumerate(atoms_per):
        species_natoms[i] = len(ats)
        for j, pos in enumerate(ats):
            species_tau[i, j] = pos

    return species_natoms, species_tau, max_atoms
