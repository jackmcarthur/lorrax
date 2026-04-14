"""psp/radial_tables.py — Build all radial Hankel tables from species data.

Single pass: SpeciesData → q-grid tables for V_loc_sr, ρ_core, and β projectors.
All transforms go through one Hankel function with different l.
Tables ship to GPU as JAX arrays via jnp.asarray.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, erf as scipy_erf

from psp.species import SpeciesData


# ---------------------------------------------------------------------------
# Core Hankel transform — the one function everything flows through
# ---------------------------------------------------------------------------

def hankel_l(l: int, r: np.ndarray, f_r: np.ndarray, q: np.ndarray,
             rab: np.ndarray) -> np.ndarray:
    """∫ dr r² f(r) j_l(qr) rab(r)  — Hankel transform with Simpson weights.

    Parameters
    ----------
    l : angular momentum (0, 1, 2, ...)
    r, f_r, rab : (n_r,) radial grid, function, and integration weights
    q : (n_q,) momentum grid

    Returns
    -------
    table : (n_q,)
    """
    n_r = len(r)
    # Simpson weights: 1/3, 4/3, 2/3, 4/3, ..., 4/3, 1/3
    sw = np.ones(n_r)
    sw[1:-1:2] = 4.0 / 3.0
    sw[2:-1:2] = 2.0 / 3.0
    sw[0] = sw[-1] = 1.0 / 3.0
    w = sw * rab

    qr = q[:, None] * r[None, :]                     # (n_q, n_r)
    j_l = spherical_jn(l, qr)                        # (n_q, n_r)
    return np.sum(j_l * (r * r * f_r)[None, :] * w[None, :], axis=1)


# ---------------------------------------------------------------------------
# Table builders — thin wrappers over hankel_l
# ---------------------------------------------------------------------------

def vloc_sr_table(sp: SpeciesData, q: np.ndarray) -> np.ndarray:
    """Short-range local potential: ∫ [V_loc(r) + Z·e²·erf(r)/r] j₀(qr) r² dr."""
    e2 = 2.0
    safe_r = np.where(sp.r > 0, sp.r, 1.0)
    erf_over_r = np.where(sp.r > 0, scipy_erf(sp.r) / safe_r, 2.0 / np.sqrt(np.pi))
    v_sr = sp.vloc_r + sp.z_valence * e2 * erf_over_r
    return hankel_l(0, sp.r, v_sr, q, sp.rab)


def core_charge_table(sp: SpeciesData, q: np.ndarray) -> np.ndarray:
    """NLCC core density: ∫ ρ_core(r) j₀(qr) r² dr."""
    return hankel_l(0, sp.r, sp.rho_core_r, q, sp.rab)


def projector_table(sp: SpeciesData, ip: int, q: np.ndarray) -> np.ndarray:
    """Beta projector form factor: ∫ (β(r)/r) j_l(qr) r² dr."""
    return hankel_l(int(sp.proj_l[ip]), sp.r, sp.beta_r[ip], q, sp.rab)


def alpha_z(sp: SpeciesData, vol: float) -> float:
    """G=0 local potential: (4π/Ω) ∫ r·[r·V_loc(r) + Z·e²] rab dr."""
    e2 = 2.0
    integrand = sp.r * (sp.r * sp.vloc_r + sp.z_valence * e2)
    # Simpson weights
    n_r = len(sp.r)
    sw = np.ones(n_r)
    sw[1:-1:2] = 4.0 / 3.0
    sw[2:-1:2] = 2.0 / 3.0
    sw[0] = sw[-1] = 1.0 / 3.0
    return float(4.0 * np.pi * np.sum(integrand * sw * sp.rab) / vol)


# ---------------------------------------------------------------------------
# Build all tables for all species in one pass
# ---------------------------------------------------------------------------

def build_all_tables(
    species_list: list[SpeciesData],
    q_max: float,
    n_q: int = 4000,
) -> dict:
    """Build Hankel tables for all species on a uniform q-grid.

    Returns dict with:
      q : (n_q,) uniform grid
      vloc : (n_species, n_q) V_loc short-range tables
      nlcc : (n_species, n_q) core density tables
      has_vloc : (n_species,) bool
      has_nlcc : (n_species,) bool
      proj_tables : list of (n_proj_s, n_q) per species
      proj_l : list of (n_proj_s,) int per species
      proj_j : list of (n_proj_s,) float per species
    """
    q = np.linspace(0.0, max(q_max, 1e-8), n_q)
    n_sp = len(species_list)

    vloc = np.zeros((n_sp, n_q), dtype=np.float64)
    nlcc = np.zeros((n_sp, n_q), dtype=np.float64)
    has_vloc = np.ones(n_sp, dtype=bool)
    has_nlcc = np.zeros(n_sp, dtype=bool)
    proj_tables = []

    for i, sp in enumerate(species_list):
        vloc[i] = vloc_sr_table(sp, q)
        if sp.has_nlcc:
            nlcc[i] = core_charge_table(sp, q)
            has_nlcc[i] = True

        # All projector tables for this species
        ptab = np.zeros((sp.n_proj, n_q), dtype=np.float64)
        for ip in range(sp.n_proj):
            ptab[ip] = projector_table(sp, ip, q)
        proj_tables.append(ptab)

    return dict(q=q, dq=float(q[1] - q[0]) if n_q > 1 else 1.0,
                vloc=vloc, nlcc=nlcc,
                has_vloc=has_vloc, has_nlcc=has_nlcc,
                proj_tables=proj_tables)
