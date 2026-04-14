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

from psp.radial_jax import (
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

    B = float(blat) * np.asarray(bvec, dtype=np.float64).T          # (3,3)
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

    Unified pipeline: all species share one JIT-compiled accumulation.

    Parameters
    ----------
    wfn : WFNReader or CrystalData
    pseudos : dict[str, UPF]
    fft_grid : (nx, ny, nz)
    truncation_2d : apply 2D Coulomb truncation to V_loc LR
    n_q : table grid size (shared for V_loc^SR + NLCC + VNL)

    Returns
    -------
    V_loc_r : (nx, ny, nz) float64 — local ionic potential in real space (Ry)
    rho_core_r : (nx, ny, nz) float64 — NLCC core density on FFT grid
    rho_core_G : (nx, ny, nz) complex128 — scaled G-space core density
    """
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    N = nx * ny * nz
    vol = float(wfn.cell_volume)
    bvec = np.asarray(wfn.bvec, dtype=np.float64)
    blat = float(wfn.blat)

    # ── G-grid data ──
    G_crys_flat, G_norm_flat = build_fft_G_data(fft_grid, bvec, blat)
    q_max = float(np.max(G_norm_flat))
    q_grid = make_uniform_q_grid(q_max, n_q)
    q0 = float(q_grid[0])
    dq = float(q_grid[1] - q_grid[0])

    # ── Extract species data and build atom→species mapping ──
    species_data = _extract_species_radial_data(pseudos, vol)
    from psp.get_DFT_mtxels import _symbol_to_Z
    z_to_species_idx = {}
    for i, sd in enumerate(species_data):
        z = _symbol_to_Z(sd["elem"])
        if z is not None:
            z_to_species_idx[z] = i

    atom_pos = np.asarray(wfn.atom_crys, dtype=np.float64)
    atom_types = np.asarray(wfn.atom_types, dtype=int)

    # Group atoms by species
    n_species = len(species_data)
    atoms_per_species = [[] for _ in range(n_species)]
    for iat in range(len(atom_pos)):
        sp_idx = z_to_species_idx.get(int(atom_types[iat]))
        if sp_idx is not None:
            atoms_per_species[sp_idx].append(atom_pos[iat])

    max_atoms = max((len(a) for a in atoms_per_species), default=1)
    max_atoms = max(max_atoms, 1)

    # Pad atom positions → (n_species, max_atoms, 3)
    species_tau = np.zeros((n_species, max_atoms, 3), dtype=np.float64)
    species_natoms = np.zeros(n_species, dtype=np.int32)
    for i, ats in enumerate(atoms_per_species):
        n = len(ats)
        species_natoms[i] = n
        for j in range(n):
            species_tau[i, j] = ats[j]

    # ── Hankel-transform all l=0 radial functions ──
    # V_loc^SR and NLCC tables, stacked for unified accumulation
    vloc_tables = np.zeros((n_species, n_q), dtype=np.float64)
    nlcc_tables = np.zeros((n_species, n_q), dtype=np.float64)
    has_vloc = np.zeros(n_species, dtype=bool)
    has_nlcc = np.zeros(n_species, dtype=bool)

    for i, sd in enumerate(species_data):
        r, rab = sd["r"], sd["rab"]
        weights_rab = radial_weights(r, rab, scheme="rab")
        weights_simp = radial_weights(r, rab, scheme="simpson_rab")

        if sd["has_vloc"]:
            tab = make_local_sr_table(r, sd["vloc_r"], sd["z_valence"], q_grid, weights_rab)
            vloc_tables[i] = tab.values
            has_vloc[i] = True

            # Alpha-Z: override q=0 entry so accumulator handles G=0 correctly.
            # In the old code, V_sr_on_grid[0,0,0] = sqrtN * alphaZ was set BEFORE
            # multiplication by structure factor. We achieve the same by setting
            # table[0] = alphaZ * vol / (4π) so that pf * table[0] = sqrtN * alphaZ.
            if not truncation_2d:
                e2 = 2.0
                z_val = sd["z_valence"]
                if rab is not None:
                    integrand = r * (r * sd["vloc_r"] + z_val * e2)
                    alphaZ = (4.0 * np.pi) * np.sum(integrand * rab) / vol
                else:
                    dr = float(r[1] - r[0]) if len(r) > 1 else 1.0
                    integrand = r * (r * sd["vloc_r"] + z_val * e2)
                    w = np.ones_like(r); w[0] = 0.5; w[-1] = 0.5
                    alphaZ = (4.0 * np.pi) * np.sum(integrand * w) * dr / vol
                # pf = sqrtN * 4π / vol, so table[0] = alphaZ * vol / (4π)
                # gives pf * table[0] = sqrtN * alphaZ  ✓
                vloc_tables[i, 0] = alphaZ * vol / (4.0 * np.pi)

        if sd["has_nlcc"]:
            tab = make_core_charge_table(r, sd["rho_core_r"], q_grid, weights_simp)
            nlcc_tables[i] = tab.values
            has_nlcc[i] = True

    # ── Move to JAX and run the jitted pipeline ──
    G_crys_j = jnp.asarray(G_crys_flat, dtype=jnp.float64)
    G_norm_j = jnp.asarray(G_norm_flat, dtype=jnp.float64)
    sp_tau_j = jnp.asarray(species_tau, dtype=jnp.float64)
    sp_nat_j = jnp.asarray(species_natoms, dtype=jnp.int32)

    # Structure factors: S_s(G) = Σ_a e^{-2πi G·τ_a}  (shared by V_loc and NLCC)
    S_species = species_structure_factors(sp_tau_j, sp_nat_j, G_crys_j, max_atoms)

    # ── NLCC accumulation ──
    nlcc_pf = jnp.asarray(
        np.where(has_nlcc, 4.0 * np.pi / vol, 0.0), dtype=jnp.float64
    )
    rho_core_G_flat = accumulate_species_on_G(
        jnp.asarray(nlcc_tables), nlcc_pf, S_species, G_norm_j, q0, dq,
    )
    rho_core_G = rho_core_G_flat.reshape(nx, ny, nz)
    rho_core_r = jnp.real(jnp.fft.ifftn(rho_core_G)) * N

    integral = float(jnp.sum(rho_core_r)) * vol / N
    print(f"  Core density integral: {integral:.4f} e "
          f"(from NLCC, {int(np.sum(species_natoms[has_nlcc]))} atoms)")

    # Scaled G-space coefficients for GGA gradient computation
    rho_core_G_scaled = rho_core_G * N

    # ── V_loc^SR accumulation ──
    sqrtN = np.sqrt(float(N))
    vloc_sr_pf = jnp.asarray(
        np.where(has_vloc, sqrtN * 4.0 * np.pi / vol, 0.0), dtype=jnp.float64
    )
    Vloc_sr_G_flat = accumulate_species_on_G(
        jnp.asarray(vloc_tables), vloc_sr_pf, S_species, G_norm_j, q0, dq,
    )

    # ── V_loc long-range Coulomb tail: -4πe²Z / G² × exp(-G²/4) × S ──
    G2_flat = G_norm_j ** 2
    G2_safe = jnp.where(G2_flat == 0.0, 1.0, G2_flat)
    e2 = 2.0

    # Ion charge density in G-space: ρ_ion(G) = -Σ_s Z_s S_s(G)
    z_vals = jnp.asarray([sd["z_valence"] for sd in species_data], dtype=jnp.float64)
    rho_ion_G_flat = -jnp.sum(z_vals[:, None] * S_species, axis=0)

    Vloc_lr_G_flat = (sqrtN * 4.0 * np.pi * e2 / vol) * (
        rho_ion_G_flat / G2_safe * jnp.exp(-0.25 * G2_flat)
    )

    # 2D truncation
    if truncation_2d:
        B = blat * bvec.T
        G_cart_flat = G_crys_flat @ B
        Gxy = jnp.sqrt(G_cart_flat[:, 0] ** 2 + G_cart_flat[:, 1] ** 2)
        lz = np.pi / B[2, 2]
        cutoff = 1.0 - jnp.exp(-Gxy * lz) * jnp.cos(G_cart_flat[:, 2] * lz)
        cutoff = jnp.where(G2_flat == 0.0, 0.0, cutoff)
        Vloc_lr_G_flat = Vloc_lr_G_flat * cutoff

    Vloc_lr_G_flat = Vloc_lr_G_flat.at[0].set(0.0)   # G=0 handled by alpha-Z

    # Total V_loc: IFFT to real space (ortho convention)
    V_tot_G = (Vloc_sr_G_flat + Vloc_lr_G_flat).reshape(nx, ny, nz)
    V_loc_r = jnp.real(jnp.fft.ifftn(V_tot_G, norm="ortho"))

    return V_loc_r, rho_core_r, rho_core_G_scaled


__all__ = [
    "accumulate_species_on_G",
    "build_fft_G_data",
    "build_ionic_and_core",
    "species_structure_factors",
]
