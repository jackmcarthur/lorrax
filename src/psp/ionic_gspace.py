"""psp/ionic_gspace.py — Unified G-space setup for ionic potentials and core charge.

Provides jittable primitives that are lax.scan-composable:

1. species_structure_factor  — S_s(G) = Σ_{a∈s} e^{-2πi G·τ_a}
2. accumulate_species_on_G   — Σ_s pf_s · f_s(|G|) · S_s(G)  via lax.scan

These replace the Python loops over atoms/species in build_core_density
and build_local_ionic_potential_on_G_total.  The same interp_uniform_jax
primitive is reused by vnl_ops for per-k projector evaluation.

QE reference: setlocal.f90 (vloc accumulation), struct_fact.f90 (species
structure factors), set_rhoc.f90 (NLCC), init_us_2.f90 (VNL projectors).
All four use the same tabulate-on-q-grid → interpolate → structure-factor
→ accumulate → IFFT pipeline.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from psp.radial.radial_jax import (
    RadialTable,
    interp_uniform_jax,
    make_core_charge_table,
    make_local_sr_table,
    make_uniform_q_grid,
    radial_weights,
    spherical_hankel_table_jax,
)

# ---------------------------------------------------------------------------
# Primitive 1: species structure factors via lax.scan over atoms
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("max_atoms",))
def species_structure_factors(
    species_tau: jax.Array,
    species_natoms: jax.Array,
    G_crys_flat: jax.Array,
    max_atoms: int,
) -> jax.Array:
    """Compute S_s(G) = Σ_{a∈s} e^{-2πi G·τ_a} for each species.

    Uses nested lax.scan: outer over species, inner over atoms.
    Memory: O(N) per step, never materializes (n_sp × max_at × N).

    Parameters
    ----------
    species_tau : (n_species, max_atoms, 3) — padded crystal coords
    species_natoms : (n_species,) int — actual atom count per species
    G_crys_flat : (N, 3) float — integer Miller indices, flattened FFT grid
    max_atoms : static int — padding dimension

    Returns
    -------
    S : (n_species, N) complex128
    """
    N = G_crys_flat.shape[0]

    def _one_species(_, inputs):
        tau_padded, natoms = inputs   # (max_atoms, 3), scalar

        def _one_atom(S, i):
            phase = jnp.exp(-2j * jnp.pi * (G_crys_flat @ tau_padded[i]))
            return S + phase * (i < natoms), None

        S_sp, _ = jax.lax.scan(_one_atom, jnp.zeros(N, jnp.complex128),
                                jnp.arange(max_atoms))
        return None, S_sp

    _, S_all = jax.lax.scan(_one_species, None,
                             (species_tau, species_natoms))
    return S_all   # (n_species, N)


# ---------------------------------------------------------------------------
# Primitive 2: accumulate form-factors × structure-factors via lax.scan
# ---------------------------------------------------------------------------

@jax.jit
def accumulate_species_on_G(
    tables: jax.Array,
    prefactors: jax.Array,
    S_species: jax.Array,
    G_norm_flat: jax.Array,
    q0: float,
    dq: float,
) -> jax.Array:
    """Σ_s pf_s · interp(table_s, |G|) · S_s(G), accumulated via lax.scan.

    Parameters
    ----------
    tables : (n_species, n_q) — Hankel-transform tables on uniform q-grid
    prefactors : (n_species,) real — per-species scaling (e.g. 4π/Ω)
    S_species : (n_species, N) complex — species structure factors
    G_norm_flat : (N,) — |G| on FFT grid
    q0, dq : table grid origin and spacing

    Returns
    -------
    total_G : (N,) complex128
    """
    def _one(total, inputs):
        table, pf, S = inputs
        f_G = interp_uniform_jax(G_norm_flat, q0, dq, table)
        return total + pf * f_G * S, None

    total, _ = jax.lax.scan(
        _one,
        jnp.zeros(G_norm_flat.shape[0], dtype=jnp.complex128),
        (tables, prefactors, S_species),
    )
    return total


# ---------------------------------------------------------------------------
# G-grid setup (host-side, called once)
# ---------------------------------------------------------------------------

def build_fft_G_data(fft_grid, bvec, blat):
    """Compute |G| and integer Miller indices on the FFT grid.

    Returns
    -------
    G_crys_flat : (N, 3) int — Miller indices, row-major flattened
    G_norm_flat : (N,) float64 — |G| in Cartesian (bohr⁻¹)
    """
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_crys_flat = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.float64)

    B = float(blat) * np.asarray(bvec, dtype=np.float64)             # (3,3) rows = b_i
    G_cart_flat = G_crys_flat @ B                                     # (N, 3)
    G_norm_flat = np.sqrt(np.sum(G_cart_flat ** 2, axis=-1))          # (N,)
    return G_crys_flat, G_norm_flat


# ---------------------------------------------------------------------------
# Radial data extraction (host-side, called once per species)
# ---------------------------------------------------------------------------

def _extract_species_radial_data(pseudos, cell_volume):
    """Extract V_loc^SR and NLCC radial integrands from pseudos.

    Returns lists of (r, f_r, weights, z_valence, has_nlcc, rho_core_r)
    per species, plus species_key→index mapping.
    """
    from jax.scipy.special import erf as jax_erf

    species_data = []
    for elem, pseudo in pseudos.items():
        ppmesh = getattr(pseudo, "pp_mesh", None)
        if ppmesh is None:
            continue
        r = np.asarray(ppmesh.pp_r.value, dtype=np.float64)
        try:
            rab = np.asarray(ppmesh.pp_rab.value, dtype=np.float64)
        except Exception:
            rab = None

        z_val = float(getattr(getattr(pseudo, "pp_header", None), "z_valence", 0.0))

        # V_loc short-range integrand
        vloc_raw = getattr(getattr(pseudo, "pp_local", None), "value", None)
        has_vloc = vloc_raw is not None
        vloc_r = np.asarray(vloc_raw, dtype=np.float64) if has_vloc else np.zeros_like(r)

        # NLCC core charge
        hdr = getattr(pseudo, "pp_header", None)
        cc = getattr(hdr, "core_correction", None)
        has_nlcc = cc is not None and str(cc) == "UpfLogical.T"
        nlcc_obj = getattr(pseudo, "pp_nlcc", None)
        rho_core_r = (np.asarray(nlcc_obj.value, dtype=np.float64)
                      if has_nlcc and nlcc_obj is not None else np.zeros_like(r))

        species_data.append(dict(
            elem=elem, pseudo=pseudo, r=r, rab=rab,
            z_valence=z_val, vloc_r=vloc_r, has_vloc=has_vloc,
            rho_core_r=rho_core_r, has_nlcc=has_nlcc,
        ))
    return species_data


# ---------------------------------------------------------------------------
# High-level builder: V_loc(r) and ρ_core(r) in one pass
# ---------------------------------------------------------------------------

def build_ionic_and_core(
    wfn,
    pseudos: dict,
    fft_grid,
    *,
    truncation_2d: bool = False,
    n_q: int = 4000,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build V_loc(r) and ρ_core(r) from pseudopotentials on the FFT grid.

    Returns (V_loc_r, rho_core_r, rho_core_G_scaled).
    """
    from psp.species import extract_species, build_atom_species_map
    from psp.radial_tables import build_all_tables, alpha_z

    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    N = nx * ny * nz
    vol = float(wfn.cell_volume)
    bvec = np.asarray(wfn.bvec, dtype=np.float64)
    blat = float(wfn.blat)

    # ── G-grid ──
    G_crys_flat, G_norm_flat = build_fft_G_data(fft_grid, bvec, blat)
    q_max = float(np.max(G_norm_flat))

    # ── Species data + radial tables (all CPU, one pass) ──
    species_list = extract_species(pseudos, nspinor=int(getattr(wfn, 'nspinor', 2)))
    tables = build_all_tables(species_list, q_max, n_q)
    species_natoms, species_tau, max_atoms = build_atom_species_map(wfn, species_list)
    n_sp = len(species_list)
    q0, dq = 0.0, tables["dq"]

    # Alpha-Z: override vloc table at q=0 for QE-compatible G=0
    if not truncation_2d:
        for i, sp in enumerate(species_list):
            az = alpha_z(sp, vol)
            tables["vloc"][i, 0] = az * vol / (4.0 * np.pi)

    # ── Ship to GPU ──
    G_crys_j = jnp.asarray(G_crys_flat, dtype=jnp.float64)
    G_norm_j = jnp.asarray(G_norm_flat, dtype=jnp.float64)
    S_species = species_structure_factors(
        jnp.asarray(species_tau, dtype=jnp.float64),
        jnp.asarray(species_natoms, dtype=jnp.int32),
        G_crys_j, max_atoms)

    # ── NLCC accumulation ──
    nlcc_pf = jnp.asarray(
        np.where(tables["has_nlcc"], 4.0 * np.pi / vol, 0.0), dtype=jnp.float64)
    rho_core_G_flat = accumulate_species_on_G(
        jnp.asarray(tables["nlcc"]), nlcc_pf, S_species, G_norm_j, q0, dq)
    rho_core_G = rho_core_G_flat.reshape(nx, ny, nz)
    rho_core_r = jnp.real(jnp.fft.ifftn(rho_core_G)) * N

    n_nlcc_atoms = int(np.sum(species_natoms[tables["has_nlcc"]]))
    print(f"  Core density integral: {float(jnp.sum(rho_core_r)) * vol / N:.4f} e "
          f"(from NLCC, {n_nlcc_atoms} atoms)")

    rho_core_G_scaled = rho_core_G * N

    # ── V_loc short-range accumulation ──
    sqrtN = np.sqrt(float(N))
    vloc_pf = jnp.asarray(
        np.where(tables["has_vloc"], sqrtN * 4.0 * np.pi / vol, 0.0), dtype=jnp.float64)
    Vloc_sr_G_flat = accumulate_species_on_G(
        jnp.asarray(tables["vloc"]), vloc_pf, S_species, G_norm_j, q0, dq)

    # ── V_loc long-range Coulomb tail ──
    G2_flat = G_norm_j ** 2
    G2_safe = jnp.where(G2_flat == 0.0, 1.0, G2_flat)
    z_vals = jnp.asarray([sp.z_valence for sp in species_list], dtype=jnp.float64)
    rho_ion_G = -jnp.sum(z_vals[:, None] * S_species, axis=0)

    Vloc_lr_G_flat = (sqrtN * 4.0 * np.pi * 2.0 / vol) * (
        rho_ion_G / G2_safe * jnp.exp(-0.25 * G2_flat))

    if truncation_2d:
        B = blat * bvec
        G_cart_flat = G_crys_flat @ B
        Gxy = jnp.sqrt(G_cart_flat[:, 0] ** 2 + G_cart_flat[:, 1] ** 2)
        lz = np.pi / B[2, 2]
        cutoff = 1.0 - jnp.exp(-Gxy * lz) * jnp.cos(G_cart_flat[:, 2] * lz)
        Vloc_lr_G_flat = Vloc_lr_G_flat * jnp.where(G2_flat == 0.0, 0.0, cutoff)

    Vloc_lr_G_flat = Vloc_lr_G_flat.at[0].set(0.0)

    V_tot_G = (Vloc_sr_G_flat + Vloc_lr_G_flat).reshape(nx, ny, nz)
    V_loc_r = jnp.real(jnp.fft.ifftn(V_tot_G, norm="ortho"))

    return V_loc_r, rho_core_r, rho_core_G_scaled


__all__ = [
    "accumulate_species_on_G",
    "build_fft_G_data",
    "build_ionic_and_core",
    "species_structure_factors",
]
