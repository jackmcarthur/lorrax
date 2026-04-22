#!/usr/bin/env python3
"""
DFT Hamiltonian matrix elements calculation.

This module computes all terms of the DFT Hamiltonian:
- Kinetic energy: <mk|T|nk> 
- Ionic potential: <mk|V_ion|nk>
- Hartree potential: <mk|V_H[n_v]|nk> 
- Nonlocal pseudopotential: <mk|V_NL|nk>

Also computes valence (n_v) and core (n_c) charge densities.
"""

from runtime import set_default_env
set_default_env()

import os
import argparse
import configparser
import re
import glob
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial
# Support both `python -m psp.get_DFT_mtxels` and direct script execution
try:
    from .normalize import normalize_dataclass
    from .load_upf import load_upf
    from ..io import WFNReader
    from ..common import symmetry_maps
    from ..common.load_wfns import read_Gvecs_to_devices
    from ..common import Meta
except ImportError:
    # Fallback for direct script execution: add project `src` to sys.path and use absolute imports
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.append(str(_Path(__file__).resolve().parents[2]))  # .../src
    from psp.upf.normalize import normalize_dataclass
    from psp.upf.load_upf import load_upf
    from file_io import WFNReader
    from common import symmetry_maps
    from common.load_wfns import read_Gvecs_to_devices
    from common import Meta
from psp.radial.build_projectors_qe import (
    build_local_ionic_potential_on_G_total,
)
from psp.dft_operators import vnl_matrix_from_kdata
from dataclasses import dataclass
import h5py
import psp.vnl_ops as vnl_ops

import common.timing as timing
# Lightweight device report (CPU-only by default)
try:
    devs = jax.devices()
    plat = devs[0].platform if devs else 'none'
    print(f"JAX: {len(devs)} {plat} devices")
except Exception:
    pass

# Import ISDF modules
 


def read_cohsex_input(filename: str) -> dict:
    """Parse input file, ignoring non-INI blocks like K_POINTS.

    We locate the [cohsex] section, strip any embedded K_POINTS {crystal_b}
    block before feeding to ConfigParser, and fall back to sensible defaults
    if the section is missing. This mirrors the robust parser used in
    `gw.gw_init.read_cohsex_input`.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find [cohsex] section bounds
    start = None
    for i, l in enumerate(lines):
        if l.strip().lower().startswith('[cohsex]'):
            start = i
            break
    if start is None:
        # fallback: find first INI-like section
        for i, l in enumerate(lines):
            if re.match(r"\s*\[.*\]", l):
                start = i
                break
    end = len(lines)

    # Locate optional K_POINTS block (count line + segments lines)
    kp_idx = None
    for i, l in enumerate(lines):
        if l.strip().lower().startswith('k_points'):
            kp_idx = i
            break
    kp_end = None
    if kp_idx is not None and kp_idx + 1 < len(lines):
        try:
            seg_count = int(lines[kp_idx + 1].strip().split()[0])
        except Exception:
            seg_count = 0
        kp_end = min(len(lines), kp_idx + 2 + max(seg_count, 0))

    if start is not None:
        for j in range(start + 1, len(lines)):
            if re.match(r"\s*\[.*\]", lines[j]):
                end = j
                break
        # Remove K_POINTS block inside the section before parsing
        if kp_idx is not None and (start <= kp_idx < end):
            section_lines = lines[start:kp_idx] + lines[(kp_end if kp_end is not None else kp_idx+1):end]
            ini_text = ''.join(section_lines)
        else:
            ini_text = ''.join(lines[start:end])
        parser = configparser.ConfigParser()
        parser.read_string(ini_text)
        section = parser["cohsex"] if "cohsex" in parser else parser[parser.sections()[0]]
        getb = section.getboolean
        get = section.get
        geti = section.getint
        return {
            "wfn_file": get("wfn_file", fallback="WFN.h5"),
            "nval": geti("nval", fallback=5),
            "ncond": geti("ncond", fallback=5),
            "nband": geti("nband", fallback=100),
            "bispinor": getb("bispinor", fallback=False),
            "ecutrho_eV": section.getfloat("ecutrho", fallback=None),
        }
    # Fallback defaults if no INI sections found
    return {
        "wfn_file": "WFN.h5",
        "nval": 5,
        "ncond": 5,
        "nband": 100,
        "bispinor": False,
        "ecutrho_eV": None,
    }


def get_bandranges(nv, nc, nband, nelec):
    """Return ranges of bands necessary for nonlocal potential calculation"""
    nvrange = [int(nelec - nv), int(nelec)]
    ncrange = [int(nelec), int(nelec + nc)]
    nsigmarange = [int(nelec - nv), int(nelec + nc)]
    n_fullrange = [0, int(nband)]
    n_valrange = [0, int(nelec)]
    return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


# ── Re-exports from psp.pseudos (canonical location) ──
from psp.pseudos import (                              # noqa: F401
    load_pseudopotentials,
    symbol_to_Z as _symbol_to_Z,
    AtomPP,
    build_atom_pp_assignments,
    print_atomic_structure,
)


def compute_valence_density(wfn_k, sym, wfn):
    """
    Compute valence charge density rho_v(r) from occupied valence wavefunctions.
        
    Returns:
        Valence charge density rho_v(r) on an ecutrho-based FFT grid if available
    """
    # Compute on configured rho grid if present, else fall back to 2x
    nk_local, nb_all, nspinor, nx, ny, nz = wfn_k.shape
    try:
        nx_pad, ny_pad, nz_pad = int(wfn.grid_rho[0]), int(wfn.grid_rho[1]), int(wfn.grid_rho[2])
    except Exception:
        nx_pad, ny_pad, nz_pad = nx, ny, nz

    same_grid = (nx_pad == nx) and (ny_pad == ny) and (nz_pad == nz)

    rho_val_local = jnp.zeros((nx_pad, ny_pad, nz_pad), dtype=jnp.float64)
    volume = jnp.asarray(wfn.cell_volume, dtype=jnp.float64)
    ngrid_pad = nx_pad * ny_pad * nz_pad
    scale_pad = jnp.sqrt(ngrid_pad / volume)
    
    # Get k-point weights - if looping over full mesh (sym.nk_tot), use 1/nk_tot
    # If looping over irreducible mesh with wfn.kweights, use those weights
    use_kweights = hasattr(wfn, 'kweights') and nk_local == len(wfn.kweights)
    if use_kweights:
        kweights = np.asarray(wfn.kweights, dtype=np.float64)
    else:
        # Assume equal weights for full mesh
        kweights = np.ones(nk_local, dtype=np.float64) / sym.nk_tot

    for ik in range(nk_local):
        nocc = int(wfn.nelec)
        nocc = min(nocc, nb_all)
        wk = float(kweights[ik])  # k-point weight

        if same_grid:
            psi_occ = wfn_k[ik, :nocc]  # (nocc, nspinor, nx, ny, nz)
            psi_r = jnp.fft.ifftn(psi_occ, axes=(-3, -2, -1), norm='ortho') * scale_pad
            rho_val_local += wk * jnp.sum(
                jnp.real(jnp.conj(psi_r) * psi_r), axis=(0, 1)
            )
        else:
            gvecs_k = np.asarray(sym.get_gvecs_kfull(wfn, ik))
            Gx = jnp.asarray(gvecs_k[:, 0], dtype=jnp.int32)
            Gy = jnp.asarray(gvecs_k[:, 1], dtype=jnp.int32)
            Gz = jnp.asarray(gvecs_k[:, 2], dtype=jnp.int32)

            for ispin in range(nspinor):
                C_src = wfn_k[ik, :nocc, ispin, :, :, :]

                def gather_one(arr3d):
                    return arr3d[Gx, Gy, Gz]

                C_occ = jax.vmap(gather_one, in_axes=0, out_axes=0)(C_src)

                def scatter_one(row):
                    buf = jnp.zeros((nx_pad, ny_pad, nz_pad), dtype=jnp.complex128)
                    return buf.at[Gx, Gy, Gz].set(row)

                psi_G_padded_batch = jax.vmap(scatter_one, in_axes=0, out_axes=0)(C_occ)
                psi_r_batch = jnp.fft.ifftn(psi_G_padded_batch, axes=(-3, -2, -1), norm='ortho') * scale_pad
                rho_val_local += wk * jnp.sum(jnp.real(psi_r_batch.conj() * psi_r_batch), axis=0)
    
    # With proper k-point weights included above, no division needed
    # (weights sum to 1 for irreducible mesh, or 1/nk_tot each for full mesh)
    rho_v = rho_val_local

    # Caller reports integrated charge if needed
    return rho_v

def compute_core_density(atom_positions, atom_types, pseudos, meta):
    """
    Compute core charge density rho_c(r) from atomic core states in pseudopotentials.
    
    Args:
        atom_positions: Atomic positions in crystal coordinates, shape (nat, 3)
        atom_types: Atom type indices, shape (nat,)
        pseudos: Dictionary mapping element names to pseudopotential objects
        meta: System metadata object
        
    Returns:
        Core charge density rho_c(r), shape (nx, ny, nz)
    """
    print("  Computing core charge density rho_c(r)...")
    
    # TODO: Implement the following steps:
    # 1. For each atom:
    #    a. Get core charge density from pseudopotential file (usually rho_core(r))
    #    b. Place at atomic position with proper structure factor
    #    c. Transform to real space grid
    # 2. Sum contributions from all atoms
    # 3. Apply proper normalization
    
    # Note: For norm-conserving pseudopotentials, core density is often
    # represented as a smooth function that reproduces the correct
    # integrated charge within some cutoff radius
    
    # Placeholder implementation
    nx, ny, nz = meta.fft_grid
    rho_core = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    
    return rho_core


def compute_hartree_potential_real(
    rho_valence_padded: jnp.ndarray,
    bdot: jnp.ndarray,
    bvec: jnp.ndarray | None = None,
    blat: float | None = None,
    truncation_2d: bool = False,
) -> jnp.ndarray:
    """Compute Hartree potential via the shared Poisson solver.
    
    Args:
        rho_valence_padded: Valence density on FFT grid
        bdot: Reciprocal lattice metric tensor (3x3)
        bvec: Reciprocal lattice vectors (3x3), needed if truncation_2d=True
        blat: Lattice constant (bohr), needed if truncation_2d=True  
        truncation_2d: If True, apply 2D slab truncation for Coulomb
    """
    rho_G = jnp.fft.fftn(rho_valence_padded, norm='ortho')
    V_H_r = poisson_potential_from_rhoG(rho_G, bdot, bvec, blat, truncation_2d)

    return V_H_r

# ── Re-exports from dft_operators (canonical location) ──
from psp.dft_operators import poisson_potential_from_rhoG  # noqa: F401
from psp.dft_operators import generate_gvectors_k          # noqa: F401


def compute_kinetic_k(wfn_k, Gk_crys, kpoint_crys, bdot, g_mask: jax.Array | None = None):
    """
    Compute kinetic energy matrix elements <mk|T|nk> for a single k-point.
    
    T = -∇² = |k+G|² in reciprocal space (Ry units)
    
    Args:
        wfn_k: Wavefunction coefficients for single k-point, shape (nb, nspinor, nx, ny, nz)
        Gk_crys: G-vectors in crystal coordinates, shape (nG, 3)
        kpoint_crys: k-point in crystal coordinates, shape (3,)
        bdot: reciprocal metric matrix, shape (3, 3)
        
    Returns:
        Kinetic energy matrix elements, shape (nb, nb)
    """
    G_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
    G_float = jnp.asarray(Gk_crys, dtype=jnp.float64)
    k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
    bdot = jnp.asarray(bdot, dtype=jnp.float64)
    g_mask_j = None if g_mask is None else jnp.asarray(g_mask, dtype=jnp.float64)
    return _compute_kinetic_k_jit(wfn_k, G_int, G_float, k_crys, bdot, g_mask_j)


@jax.jit
def _compute_kinetic_k_jit(
    wfn_k: jax.Array,
    G_int: jax.Array,
    G_float: jax.Array,
    k_crys: jax.Array,
    bdot: jax.Array,
    g_mask: jax.Array | None,
) -> jax.Array:
    K_crys = G_float + k_crys[None, :]
    T_G = jnp.einsum('gi,ij,gj->g', K_crys, bdot, K_crys, optimize=True)
    if g_mask is not None:
        T_G = T_G * g_mask
    Gx = G_int[:, 0]
    Gy = G_int[:, 1]
    Gz = G_int[:, 2]
    psi_G = wfn_k[:, :, Gx, Gy, Gz]
    T_psi = T_G[None, None, :] * psi_G
    return jnp.einsum('msg,nsg->mn', jnp.conj(psi_G), T_psi, optimize=True)

def compute_local_V_k(wfn_k, Gk_crys, V_r, cell_volume, g_mask: jax.Array | None = None):
    """
    Compute elements of a local potential (V_ion or V_H) <mk|V|nk> for a single k-point.
    
    Args:
        wfn_k: Wavefunction coefficients for single k-point, shape (nb, nspinor, nx, ny, nz)
        Gk_crys: G-vectors in crystal coordinates, shape (nG, 3)
        V_H_r: Real-space Hartree potential on the 2x FFT grid, shape (2*nx, 2*ny, 2*nz)
        
    Returns:
        Hartree potential matrix elements, shape (nb, nb)
    """
    V_r = jnp.asarray(V_r, dtype=jnp.complex128)
    Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
    volume = jnp.asarray(cell_volume, dtype=jnp.float64)
    g_mask_j = None if g_mask is None else jnp.asarray(g_mask, dtype=jnp.float64)
    return _compute_local_V_k_jit(wfn_k, Gx, Gy, Gz, V_r, volume, g_mask_j)


@jax.jit
def _compute_local_V_k_jit(
    wfn_k: jax.Array,
    Gx: jax.Array,
    Gy: jax.Array,
    Gz: jax.Array,
    V_r: jax.Array,
    volume: jax.Array,
    g_mask: jax.Array | None,
) -> jax.Array:
    psi_G = jnp.asarray(wfn_k, dtype=jnp.complex128)
    nb = psi_G.shape[0]
    nspinor = psi_G.shape[1]
    nx, ny, nz = psi_G.shape[-3:]

    ngrid = nx * ny * nz
    scale = jnp.sqrt(ngrid / volume)
    deltaV = volume / ngrid
    fft_norm = jnp.sqrt(ngrid)

    psi_r = jnp.fft.ifftn(psi_G, axes=(-3, -2, -1), norm='ortho') * scale
    phi_r = psi_r * V_r
    phi_G = jnp.fft.fftn(phi_r, axes=(-3, -2, -1), norm='ortho') * (deltaV * fft_norm)

    psi_coeffs = psi_G[:, :, Gx, Gy, Gz]
    vpsi = phi_G[:, :, Gx, Gy, Gz]
    if g_mask is not None:
        psi_coeffs = psi_coeffs * g_mask[None, None, :]
        vpsi = vpsi * g_mask[None, None, :]
    V_loc = jnp.einsum('bsg,nsg->bn', jnp.conj(psi_coeffs), vpsi, optimize=True)
    return V_loc * jnp.sqrt(1.0 / volume)

    # Legacy implementation removed in favor of unified vnl_ops / dft_operators path.
    raise NotImplementedError("compute_V_NL_k legacy path removed; use vnl_ops.build_vnl_kdata_from_kvec plus dft_operators.vnl_matrix_from_kdata.")


@timing.timed("psp.get_DFT_mtxels.get_H_matrix_elements", watch=True)
def get_H_matrix_elements(wfn, sym, pseudos, global_psi_G, meta, mesh_xy, n_valrange):
    """
    Compute nonlocal pseudopotential matrix elements <mk|V_NL|nk> for all k-points.
    
    This implementation distributes k-points across the XY processor grid and 
    computes V_NL elements for each k-point independently.
    
    Args:
        wfn: WFNReader object
        sym: SymMaps object  
        pseudos: Dictionary of loaded pseudopotentials
        global_psi_G: Global sharded wavefunction coefficients in G-space
        meta: System metadata
        mesh_xy: JAX device mesh for sharding
        n_valrange: Band range for all valence bands [0, nelec]
        
    Returns:
        Array of nonlocal potential matrix elements, shape (nk, nb, nb)
    """
    print("\nComputing DFT Hamiltonian (Ry units)...")
    
    # 1. Reshard wavefunctions to distribute k-points over XY grid
    k_xy_shard = NamedSharding(mesh_xy, P(('x','y'), None, None, None, None, None))
    print("  Resharding wavefunctions to device mesh")
    wfn_k_sharded = jax.lax.with_sharding_constraint(global_psi_G, k_xy_shard)
    
    # 2. Prepare atomic structure data (replicated on all devices)
    atom_positions = jnp.asarray(wfn.atom_crys, dtype=jnp.float64)  # Crystal coordinates
    atom_types = jnp.asarray(wfn.atom_types, dtype=jnp.int32)
    bvec = jnp.asarray(wfn.bvec, dtype=jnp.float64)
    kpoints = jnp.asarray(sym.unfolded_kpts, dtype=jnp.float64)
    
    print(f"\nSystem: {len(atom_positions)} atoms, {len(pseudos)} pseudopotential types")
    assignments = build_atom_pp_assignments(atom_positions, atom_types, pseudos)
    for ap in assignments:
        ppname = os.path.basename(getattr(ap.pseudo, '_source_path', 'N/A')) if ap.pseudo else 'None'
        print(f"  atom {ap.index}: Z={ap.atomic_number} elem={ap.element} pp={ppname}")

    species_payload: list[tuple[object, np.ndarray]] = []
    species_tmp: dict[int, dict[str, object]] = {}
    for ap in assignments:
        pseudo = ap.pseudo
        if pseudo is None:
            continue
        key = id(pseudo)
        entry = species_tmp.setdefault(key, {"pseudo": pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    for entry in species_tmp.values():
        positions = np.asarray(entry["positions"], dtype=float) if entry["positions"] else np.zeros((0, 3), dtype=float)
        species_payload.append((entry["pseudo"], positions))

    assignment_payload = [
        {
            "pseudo": ap.pseudo,
            "position": np.asarray(ap.position, dtype=float),
        }
        for ap in assignments
    ]

    # Precompute G and K scaffolding for all k-points
    print("  Precomputing G and K scaffolding for all k-points...")
    Gk_crys_all: list[jnp.ndarray] = []
    for i in range(sym.nk_tot):
        Gk_crys_i, _ = generate_gvectors_k(i, sym, wfn, meta)
        Gk_crys_all.append(jnp.asarray(Gk_crys_i, dtype=jnp.int32))
    print("  Done precomputing G scaffolding.")

    # Build fixed-size G pads for the local terms and a unified VNL setup once.
    nG_list: list[int] = []
    for Gk_crys_i in Gk_crys_all:
        nG_list.append(int(Gk_crys_i.shape[0]))
    nG_max = max(nG_list) if nG_list else 0
    Gk_crys_pad: list[jnp.ndarray] = []
    G_mask: list[jnp.ndarray] = []
    for i in range(sym.nk_tot):
        Gcur = jnp.asarray(Gk_crys_all[i], dtype=jnp.int32)
        nG = int(Gcur.shape[0])
        if nG < nG_max:
            pad = nG_max - nG
            Gpad = jnp.pad(Gcur, ((0, pad), (0, 0)))
            mask = jnp.concatenate(
                [jnp.ones((nG,), dtype=jnp.float64), jnp.zeros((pad,), dtype=jnp.float64)]
            )
        else:
            Gpad = Gcur
            mask = jnp.ones((nG_max,), dtype=jnp.float64)
        Gk_crys_pad.append(Gpad)
        G_mask.append(mask)
    vnl_setup = vnl_ops.build_vnl_setup(
        wfn,
        sym,
        meta,
        pseudos,
        nspinor=int(wfn.nspinor),
    )

    # 3. Compute valence charge density from occupied states
    V_H_r = None
    rho_valence = None
    print("\n  Computing valence charge density (ecutrho-based grid if provided)...")
    rho_valence = compute_valence_density(wfn_k_sharded, sym, wfn)
    print(f"    Valence density grid: {rho_valence.shape}")
    # Precompute Hartree potential V_H(r) on the rho grid using reciprocal metric bdot
    # Use 2D truncation by default for slab systems (matches ISDF/BerkeleyGW)
    bdot_j = jnp.asarray(wfn.bdot, dtype=jnp.float64)
    bvec_j = jnp.asarray(wfn.bvec, dtype=jnp.float64)
    V_H_r = compute_hartree_potential_real(
        rho_valence,
        bdot_j,
        bvec=bvec_j,
        blat=float(wfn.blat),
        truncation_2d=True,  # Match QE's assume_isolated='2D' behavior
    )
    deltaV = float(wfn.cell_volume) / float(np.prod(V_H_r.shape))
    print(f"    Hartree potential grid: {V_H_r.shape}")
    # Hartree energy: 1/2 ∫ rho(r) V_H(r) d^3r on the padded grid
    hartree_energy = 0.5 * float(jnp.sum(rho_valence * V_H_r)*deltaV)
    print(f"    Hartree energy (0.5 ∫ ρ V_H) = {hartree_energy:.6f} Ry")
    # Compute core density from pseudopotentials (disabled for performance)  
    # rho_core = compute_core_density(atom_positions, atom_types, pseudos, meta)
    # Build local ionic potential on rho grid via G-space and FFT (return total only)
    print("  Computing local ionic potential V_loc(r) on rho grid...")
    try:
        rho_grid = tuple(int(x) for x in getattr(wfn, 'grid_rho'))
    except Exception:
        rho_grid = (int(2*meta.fft_grid[0]), int(2*meta.fft_grid[1]), int(2*meta.fft_grid[2]))
    V_loc_r = build_local_ionic_potential_on_G_total(
        assignments=assignment_payload,
        species_groups=species_payload,
        fft_grid=rho_grid,
        bdot=np.asarray(wfn.bdot, dtype=float),
        cell_volume=float(wfn.cell_volume),
        bvec=np.asarray(wfn.bvec, dtype=float),
        blat=float(wfn.blat),
        truncation_2d=True,  # Match QE's assume_isolated='2D' behavior
    )
    V_loc_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    # 4. Execute DFT Hamiltonian calculation over k-points using precomputed G-vectors
    print("\n  Building H(k) on first k-point for debug...")
    H_list = []
    first_k_components = None
    for i in range(1):
        wfn_k = wfn_k_sharded[i]  # (nb, nspinor, nx, ny, nz)
        kpoint = kpoints[i]
        Gk_crys = Gk_crys_pad[i]

        T_k = compute_kinetic_k(
            wfn_k, Gk_crys, kpoint, bdot_j
        )
        V_ion_k = compute_local_V_k(wfn_k, Gk_crys, V_loc_r, wfn.cell_volume, G_mask[i])
        V_H_k = compute_local_V_k(wfn_k, Gk_crys, V_H_r, wfn.cell_volume, G_mask[i])
        kdata = vnl_ops.build_vnl_kdata_from_kvec(
            np.asarray(kpoint, dtype=float),
            np.asarray(Gk_crys_all[i], dtype=int),
            vnl_setup,
        )
        V_NL_k = vnl_matrix_from_kdata(wfn_k, Gk_crys_all[i], kdata)

        # Temporary debug prints: first 4x4 blocks (2 decimals, scientific)
        # (per-matrix debug prints removed)

        H_k = T_k + V_ion_k + V_H_k + V_NL_k
        H_list.append(H_k)

        # Save components for k=0 (first iter only)
        if i == 0:
            first_k_components = {
                'T': T_k,
                'V_ion': V_ion_k,
                'V_H': V_H_k,
                'V_NL': V_NL_k,
                'H_no_NL': T_k + V_ion_k + V_H_k,
            }

    H_sharded = jnp.stack(H_list, axis=0)
    
    print(f"  Completed: DFT Hamiltonian matrix shape {H_sharded.shape}")
    
    return H_sharded, rho_valence, first_k_components
    
@timing.timed("psp.get_DFT_mtxels.get_kin_ion", watch=True)
def get_kin_ion(
    global_psi_G,
    wfn,
    sym,
    pseudos,
    meta,
    mesh_xy,
    assignments=None,
    species_payload=None,
    include_hartree: bool = False,
    nb_limit: int | None = None,
    sys_dim: int = 3,
):
    """Return kinetic + ionic (+ optional Hartree) matrices for all k-points: shape (nk, nb, nb).

    When `include_hartree` is True the valence Hartree potential V_H is
    constructed from the occupied states and added to the returned matrices.

    Parameters
    ----------
    sys_dim : int
        System dimensionality (0, 2, or 3).  Controls Coulomb truncation
        for V_loc: ``truncation_2d=True`` only when ``sys_dim == 2``.
    """
    from psp.operator_checks import validate_operator_inputs
    ctx = validate_operator_inputs(
        pseudos=pseudos, wfn=wfn, sys_dim=sys_dim, caller="get_kin_ion",
    )
    # Reshard to ensure k is sharded across mesh as in main flow
    k_xy_shard = NamedSharding(mesh_xy, P(('x','y'), None, None, None, None, None))
    wfn_k_sharded = jax.lax.with_sharding_constraint(global_psi_G, k_xy_shard)

    # Structure setup
    with timing.section("psp.get_DFT_mtxels.get_kin_ion.structure_setup") as timer_struct:
        atom_positions = jnp.asarray(wfn.atom_crys, dtype=jnp.float64)
        atom_types = jnp.asarray(wfn.atom_types, dtype=jnp.int32)
        if assignments is None:
            assignments = build_atom_pp_assignments(atom_positions, atom_types, pseudos)
        if species_payload is None:
            tmp: dict[int, dict[str, object]] = {}
            for ap in assignments:
                pseudo = ap.pseudo
                if pseudo is None:
                    continue
                key = id(pseudo)
                entry = tmp.setdefault(key, {"pseudo": pseudo, "positions": []})
                entry["positions"].append(np.asarray(ap.position, dtype=float))
            species_payload = [
                (
                    e["pseudo"],
                    np.asarray(e["positions"], dtype=float) if e["positions"] else np.zeros((0, 3), dtype=float),
                )
                for e in tmp.values()
            ]
        timer_struct.watch(assignments, species_payload)

    # Precompute G scaffolding once; unified VNL setup reconstructs per-k K on device.
    with timing.section("psp.get_DFT_mtxels.get_kin_ion.k_scaffolding") as timer_kprep:
        Gk_crys_all: list[jnp.ndarray] = []
        for i in range(sym.nk_tot):
            Gk_crys_i, _ = generate_gvectors_k(i, sym, wfn, meta)
            Gk_crys_all.append(jnp.asarray(Gk_crys_i, dtype=jnp.int32))
        timer_kprep.watch(Gk_crys_all)

    # Build fixed-size G pads for local terms, and one shared VNL setup.
    nG_list = []
    for Gk_crys_i in Gk_crys_all:
        nG_list.append(int(Gk_crys_i.shape[0]))

    nG_max = max(nG_list) if nG_list else 0
    Gk_crys_pad: list[jnp.ndarray] = []
    G_mask: list[jnp.ndarray] = []
    for i in range(sym.nk_tot):
        Gcur = jnp.asarray(Gk_crys_all[i], dtype=jnp.int32)
        nG = int(Gcur.shape[0])
        if nG < nG_max:
            pad = nG_max - nG
            Gpad = jnp.pad(Gcur, ((0, pad), (0, 0)))
            mask = jnp.concatenate([jnp.ones((nG,), dtype=jnp.float64), jnp.zeros((pad,), dtype=jnp.float64)])
        else:
            Gpad = Gcur
            mask = jnp.ones((nG_max,), dtype=jnp.float64)
        Gk_crys_pad.append(Gpad)
        G_mask.append(mask)

    # Build unified VNL setup once for all k.
    with timing.section("psp.get_DFT_mtxels.get_kin_ion.plan_build"):
        vnl_setup = vnl_ops.build_vnl_setup(
            wfn,
            sym,
            meta,
            pseudos,
            nspinor=int(wfn.nspinor),
        )

    # Build V_loc on 2x grid once
    with timing.section("psp.get_DFT_mtxels.get_kin_ion.build_V_loc") as timer_vloc:
        V_loc_r = build_local_ionic_potential_on_G_total(
            assignments=[
                {"pseudo": ap.pseudo, "position": np.asarray(ap.position, dtype=float)} for ap in assignments
            ],
            species_groups=[
                (
                    sp[0],
                    (np.asarray(sp[1], dtype=float) if np.asarray(sp[1]).size > 0 else np.zeros((0, 3), dtype=float)),
                )
                for sp in species_payload
            ],
            fft_grid=tuple(int(x) for x in meta.fft_grid),
            bdot=np.asarray(wfn.bdot, dtype=float),
            cell_volume=float(wfn.cell_volume),
            bvec=np.asarray(wfn.bvec, dtype=float),
            blat=float(wfn.blat),
            truncation_2d=ctx.truncation_2d,
        )
        V_loc_r = jnp.asarray(V_loc_r, dtype=jnp.float64)
        # Avoid forcing device sync here; leave compute lazy

    V_H_r = None
    if include_hartree:
        if int(wfn.nelec) > wfn_k_sharded.shape[1]:
            raise ValueError(
                f"include_hartree=True requires at least nelec={wfn.nelec} bands (got {wfn_k_sharded.shape[1]})"
            )
        print("  Computing valence density and Hartree potential for kin+ion...")
        with timing.section("psp.get_DFT_mtxels.get_kin_ion.hartree") as timer_hartree:
            rho_valence = compute_valence_density(wfn_k_sharded, sym, wfn)
            bdot_j = jnp.asarray(wfn.bdot, dtype=jnp.float64)
            bvec_j = jnp.asarray(wfn.bvec, dtype=jnp.float64)
            V_H_r = compute_hartree_potential_real(
                rho_valence,
                bdot_j,
                bvec=bvec_j,
                blat=float(wfn.blat),
                truncation_2d=False,  # Set to True for 2D slab truncation matching ISDF
            )

    # Allocate output and compute per-k
    nk, nb_total = wfn_k_sharded.shape[0], wfn_k_sharded.shape[1]
    nb = int(nb_total)
    if nb_limit is not None:
        nb = max(1, min(nb, int(nb_limit)))
    kin_ion = np.zeros((nk, nb, nb), dtype=np.complex128)
    bdot_j = jnp.asarray(wfn.bdot, dtype=jnp.float64)
    for i in range(sym.nk_tot):
        with timing.section("psp.get_DFT_mtxels.get_kin_ion.k_loop") as timer_kloop:
            wfn_k = wfn_k_sharded[i, :nb]
            kpoint = jnp.asarray(sym.unfolded_kpts[i], dtype=jnp.float64)
            Gk_crys = Gk_crys_pad[i]
            with timing.section("psp.get_DFT_mtxels.get_kin_ion.compute_T"):
                T_k = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot_j, G_mask[i])
            with timing.section("psp.get_DFT_mtxels.get_kin_ion.compute_V_loc"):
                V_ion_k = compute_local_V_k(wfn_k, Gk_crys, V_loc_r, wfn.cell_volume, G_mask[i])
            with timing.section("psp.get_DFT_mtxels.get_kin_ion.compute_V_NL"):
                kdata = vnl_ops.build_vnl_kdata_from_kvec(
                    np.asarray(kpoint, dtype=float),
                    np.asarray(Gk_crys_all[i], dtype=int),
                    vnl_setup,
                )
                V_NL_k = vnl_matrix_from_kdata(wfn_k, Gk_crys_all[i], kdata)
            total_k = T_k + V_ion_k + V_NL_k
            if include_hartree:
                with timing.section("psp.get_DFT_mtxels.get_kin_ion.compute_V_H"):
                    V_H_k = compute_local_V_k(wfn_k, Gk_crys, V_H_r, wfn.cell_volume, G_mask[i])
                total_k = total_k + V_H_k
            kin_ion[i] = np.asarray(total_k)
            timer_kloop.watch(kin_ion[i])

    return kin_ion


def write_kin_ion_h5(kin_ion: np.ndarray, out_path: str = 'kin_ion.h5') -> None:
    """Write kin+ion array (nk, nb, nb) to an HDF5 file with dataset 'kin_ion'."""
    with h5py.File(out_path, 'w') as h5:
        dset = h5.create_dataset('kin_ion', data=kin_ion)
        dset.attrs['description'] = 'Kinetic + ionic (local + nonlocal) matrix elements, shape (nk, nb, nb)'




def main(argv=None):
    """Main function for nonlocal pseudopotential calculation."""
    print("="*60)
    print("Nonlocal Pseudopotential Matrix Elements Calculator")
    print("="*60)

    argp = argparse.ArgumentParser(description="Nonlocal pseudopotential V_NL calculator")
    argp.add_argument(
        "-i",
        "--input", 
        default="tests/cohsex_debug/cohsex_test.in",
        help="Input file",
    )
    args = argp.parse_args(argv)
    
    timing.reset()

    # Read input parameters
    print(f"\nReading input from: {args.input}")
    params = read_cohsex_input(args.input)
    
    # Resolve relative paths against the input file's directory
    input_dir = os.path.dirname(os.path.abspath(args.input))
    def _resolve_path(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)
    params["wfn_file"] = _resolve_path(params["wfn_file"])
    
    # Extract parameters
    nval = params["nval"]
    ncond = params["ncond"] 
    nband = params["nband"]
    bispinor = params["bispinor"]
    
    print(f"Parameters: nval={nval}, ncond={ncond}, nband={nband}, bispinor={bispinor}")
    
    # Load wavefunction file
    print(f"\nLoading wavefunction file: {os.path.basename(params['wfn_file'])}")
    with timing.section("psp.get_DFT_mtxels.load_wfn"):
        try:
            wfn = WFNReader(params["wfn_file"])
            print(f"  Success: {wfn.nkpts} k-points, {wfn.nbands} bands, {wfn.nelec} electrons")
        except Exception as e:
            print(f"  Error loading WFN file: {e}")
            return 1
    
    # Initialize symmetry mappings
    print("\nInitializing symmetry mappings...")
    with timing.section("psp.get_DFT_mtxels.symmetry"):
        sym = symmetry_maps.SymMaps(wfn)
    print(f"  Success: {sym.nk_tot} total k-points, {sym.nk_red} irreducible k-points")
    
    # Get band ranges
    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(
        nval, ncond, nband, wfn.nelec
    )
    print(f"Band ranges: valence={nvrange}, conduction={ncrange}, sigma={nsigmarange}")
    
    # Create system metadata
    print("\nCreating system metadata...")
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, 0, bispinor)  # n_rmu=0 for now
    print(f"  FFT grid: {meta.fft_grid}")
    print(f"  Spinor components: {meta.nspinor}")

    # If ecutrho provided, attach the rho FFT grid on the WFN object
    ecutrho_eV = params.get("ecutrho_eV", None)
    if ecutrho_eV is not None:
        ecutrho_ry = float(ecutrho_eV)  # field is actually Ry despite the name
        # Use 2× wavefunction grid as default rho grid
        setattr(wfn, 'grid_rho', tuple(2 * n for n in meta.fft_grid))
        print(f"  Using ecutrho = {ecutrho_ry:.3f} Ry; rho grid = {wfn.grid_rho}")
    
    # Set up JAX device mesh
    total_devices = jax.process_count() * jax.local_device_count()
    grid_x = int(np.sqrt(total_devices))
    while total_devices % grid_x != 0:
        grid_x -= 1
    grid_y = total_devices // grid_x
    devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
    mesh_xy = Mesh(devices_2d, ['x', 'y'])
    print(f"JAX device mesh: {grid_x}x{grid_y} = {total_devices} devices")
    
    # Load G-vectors and wavefunction coefficients
    print("\nLoading wavefunction coefficients to devices...")
    brange = (0, nsigmarange[1])  # Load all bands for now
    with timing.section("psp.get_DFT_mtxels.read_Gvecs") as timer_read:
        global_psi_G, nb_actual = read_Gvecs_to_devices(wfn, sym, brange, meta, bispinor, mesh_xy)
        timer_read.watch(global_psi_G)
    print(f"  Loaded {nb_actual} bands in G-space, shape: {global_psi_G.shape}")
    
    # Load pseudopotentials from working directory
    print("\nScanning for pseudopotential files...")
    with timing.section("psp.get_DFT_mtxels.load_pseudos"):
        pseudos = load_pseudopotentials(input_dir)
    
    # Print atomic structure information
    print_atomic_structure(wfn, pseudos)
    
    # Compute DFT Hamiltonian matrix elements
    print(f"\nComputing DFT Hamiltonian matrix elements...")
    H_DFT, rho_valence, k0 = get_H_matrix_elements(wfn, sym, pseudos, global_psi_G, meta, mesh_xy, n_valrange)
    print(f"  Hamiltonian matrix elements shape: {H_DFT.shape}")
    
    print(f"  Total electrons in system: {wfn.nelec}")
    # Report both grid-sum and ΔV-weighted integral of the valence density (rho grid)
    if rho_valence is not None:
        nx2, ny2, nz2 = rho_valence.shape
        Ntot = float(nx2 * ny2 * nz2)
        raw_sum = float(jnp.sum(rho_valence))
        deltaV = float(wfn.cell_volume) / Ntot
        charge_int = raw_sum * deltaV
        print(f"  Valence density (rho grid): Σ rho = {raw_sum:.6f}, ΔV = {deltaV:.6e}, ∫ rho d³r = {charge_int:.6f}")

    # Write requested k=0 diagonal elements: (band id, K, V_ion, V_H, K+I+H)
    try:
        if k0 is not None:
            T0 = np.asarray(k0['T'])
            VI0 = np.asarray(k0['V_ion'])
            VI_SR0 = np.asarray(k0.get('V_ion_SR', VI0*0))
            VI_LR0 = np.asarray(k0.get('V_ion_LR', VI0))
            VH0 = np.asarray(k0['V_H'])
            VNL0 = np.asarray(k0['V_NL'])
            HnoNL0 = np.asarray(k0['H_no_NL'])
            nb0 = T0.shape[0]
            m = min(26, nb0)
            out = np.zeros((m, 6), dtype=float)
            for b in range(m):
                out[b, 0] = b + 1  # 1-based band id
                out[b, 1] = np.real(T0[b, b])
                out[b, 2] = np.real(VI0[b, b])
                out[b, 3] = np.real(VH0[b, b])
                out[b, 4] = np.real(VNL0[b, b])
                out[b, 5] = np.real(HnoNL0[b, b] + VNL0[b, b])
            out_path = os.path.join(input_dir, 'k0_diag.txt')
            np.savetxt(
                out_path,
                out,
                fmt=['%4d','% .5f','% .5f','% .5f','% .5f','% .5f'],
                header='band_id  K(Ry)  V_ion(Ry)  V_H(Ry)  V_NL(Ry)  K+I+H+NL(Ry)'
            )
            print(f"\nWrote k=0 diagonals to {out_path}")

            # If a reference exists (k0_diag_check), compare K+I+H diagonals
            # Accept either k0_diag_check or k0_diag_check.txt
            ref_path = os.path.join(input_dir, 'k0_diag_check')
            if not os.path.exists(ref_path):
                alt = ref_path + '.txt'
                ref_path = alt if os.path.exists(alt) else ref_path
            if os.path.exists(ref_path):
                try:
                    ref = np.loadtxt(ref_path)
                    if ref.ndim == 1:
                        ref = ref.reshape(-1,)
                    # Accept either 1-col or 5-col (assume last col is K+I+H)
                    if ref.ndim == 2 and ref.shape[1] >= 1:
                        ref_vec = ref[:, -1]
                    else:
                        ref_vec = ref
                    mref = min(m, ref_vec.shape[0])
                    ours = out[:mref, 5]
                    diff = ours - ref_vec[:mref]
                    print(f"  k0_diag_check present; comparing first {mref} K+I+H+NL diagonals...")
                    print(f"    max|Δ|={np.max(np.abs(diff)):.5e}, rms|Δ|={np.sqrt(np.mean(diff**2)):.5e}")

                    # Constrained fit: fix a=1 for K and e=1 for V_NL; solve for b,d in ref - K - V_NL ≈ b*V_ion + d*V_H
                    Kcol   = out[:mref, 1]
                    VIcol  = out[:mref, 2]
                    VHcol  = out[:mref, 3]
                    VNLcol = out[:mref, 4]
                    y = ref_vec[:mref]
                    y_red = y - Kcol - VNLcol
                    X_red = np.stack([VIcol, VHcol], axis=1)
                    coeffs_red, residuals, rank, s = np.linalg.lstsq(X_red, y_red, rcond=None)
                    b, d = coeffs_red.tolist()
                    # Replace constrained fit with full LS: ref ≈ A*K + B*Vion + C*VH + D*VNL + E
                    ones = np.ones_like(Kcol)
                    X = np.stack([Kcol, VIcol, VHcol, VNLcol, ones], axis=1)
                    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                    A, B, C, D, E = coeffs.tolist()
                    y_fit = X @ coeffs
                    err_vec = y_fit - y
                    l2 = float(np.linalg.norm(err_vec))
                    rmse = float(np.sqrt(np.mean(err_vec**2)))
                    print("  Unconstrained fit: ref ≈ A*K + B*Vion + C*VH + D*VNL + E:")
                    print(f"    A={A:+.6f}  B={B:+.6f}  C={C:+.6f}  D={D:+.6f}  E={E:+.6f}")
                    print(f"    total_L2={l2:.6e}  rmse={rmse:.6e}")
                except Exception as e:
                    print(f"  Warning: failed to compare with k0_diag_check ({e})")

            # Vloc-only fit removed per user request
            
    except Exception as e:
        print(f"Warning: failed to write k=0 diagonals ({e})")

    print("\nDFT Hamiltonian calculation completed successfully!")
    print("="*60)

    # Optionally write kin_ion for this run
    try:
        kin_ion = get_kin_ion(global_psi_G, wfn, sym, pseudos, meta, mesh_xy)
        out_h5 = os.path.join(input_dir, 'kin_ion.h5')
        write_kin_ion_h5(kin_ion, out_h5)
        print(f"Wrote kin+ion matrices to {out_h5}")
    except Exception as e:
        print(f"Warning: failed to write kin_ion.h5 ({e})")

    timing.report(title="--- Timing (seconds) ---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
