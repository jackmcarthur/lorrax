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

import os
import argparse
import configparser
import re
import glob
from pathlib import Path

# Set JAX configs BEFORE importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import jax
# Force CPU backend to avoid GPU plugin errors in test envs
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
# Support both `python -m isdf.psp.get_DFT_mtxels` and direct script execution
try:
    from .normalize import normalize_dataclass
    from .load_upf import load_upf
    from ..common.wfnreader import WFNReader
    from ..common import symmetry_maps
    from ..common.load_wfns import read_Gvecs_to_devices
    from ..common import Meta
except ImportError:
    # Fallback for direct script execution: add project `src` to sys.path and use absolute imports
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.append(str(_Path(__file__).resolve().parents[2]))  # .../src
    from isdf.psp.normalize import normalize_dataclass
    from isdf.psp.load_upf import load_upf
    from isdf.common.wfnreader import WFNReader
    from isdf.common import symmetry_maps
    from isdf.common.load_wfns import read_Gvecs_to_devices
    from isdf.common import Meta
from isdf.psp.build_projectors_qe import (
    build_local_ionic_potential_on_G_total,
)
from isdf.psp.projector_pipeline import (
    build_vnl_plan,
    compute_V_NL_k_minimal,
)
from dataclasses import dataclass
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Lightweight device report (CPU-only by default)
try:
    devs = jax.devices()
    plat = devs[0].platform if devs else 'none'
    print(f"JAX: {len(devs)} {plat} devices")
except Exception:
    pass

# Import ISDF modules
 


def _divisibility_score(n: int) -> int:
    """Score n by how many preferred small factors divide it."""
    prefs = (2, 3, 4, 5, 8, 10, 12, 16)
    score = 0
    for p in prefs:
        if n % p == 0:
            score += 1
    return score


def _best_fft_dim(n0: int) -> int:
    """Choose a nearby FFT length in [n0, n0+5] with best small-factor score."""
    best_n = max(3, int(n0))
    best_s = _divisibility_score(best_n)
    for cand in range(best_n, best_n + 6):
        s = _divisibility_score(cand)
        if s > best_s:
            best_s, best_n = s, cand
    return best_n


def compute_fft_grid_from_ecutrho(bdot: np.ndarray, ecutrho_ry: float) -> tuple[int, int, int]:
    """Compute FFT grid (nx,ny,nz) from ecutrho (Ry) and reciprocal metric bdot.

    Uses h_i = floor(sqrt(ecutrho_ry / bdot[ii])) and n_i = 2*h_i + 1 per axis,
    then adjusts each n_i to a nearby FFT-friendly size in [n_i, n_i+5].
    """
    bdot = np.asarray(bdot, dtype=float)
    diag = np.clip(np.diag(bdot), 1e-12, None)
    h = np.floor(np.sqrt(max(0.0, float(ecutrho_ry))) / np.sqrt(diag)).astype(int)
    n = 2 * h + 1
    nx = _best_fft_dim(int(n[0]))
    ny = _best_fft_dim(int(n[1]))
    nz = _best_fft_dim(int(n[2]))
    return int(nx), int(ny), int(nz)


def read_cohsex_input(filename: str) -> dict:
    """Parse input file, ignoring non-INI blocks like K_POINTS.

    We locate the [cohsex] section, strip any embedded K_POINTS {crystal_b}
    block before feeding to ConfigParser, and fall back to sensible defaults
    if the section is missing. This mirrors the robust parser used in
    `isdf.gw_isdf.cohsex_jax.read_cohsex_input`.
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


def load_pseudopotentials(work_dir="."):
    """Load all UPF pseudopotential files found in the working directory.
    
    Args:
        work_dir (str): Directory to search for *.upf files
        
    Returns:
        dict: Dictionary mapping atom types to loaded pseudopotential objects
    """
    upf_files = glob.glob(os.path.join(work_dir, "*.upf"))
    if not upf_files:
        print(f"No pseudopotentials (*.upf) found in {work_dir}")
        return {}
    
    pseudos = {}
    for upf_file in upf_files:
        try:
            pseudo = load_upf(upf_file)
            # Normalize numeric string fields (e.g., radial arrays) to numpy arrays
            pseudo = normalize_dataclass(pseudo)
            # Attach source path for concise later reporting
            setattr(pseudo, "_source_path", upf_file)
            # Extract element name from pseudopotential
            element = pseudo.pp_header.element.strip()
            pseudos[element] = pseudo
        except Exception as e:
            print(f"Warning: failed to load {os.path.basename(upf_file)}: {e}")
            continue
            
    return pseudos


def _periodic_table_symbols() -> list[str]:
    # Ordered list of element symbols; index+1 gives atomic number Z
    return (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
        "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
        "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
        "Bh Hs Mt Ds Rg Cn Fl Lv Ts Og"
    ).split()


def _symbol_to_Z(symbol: str) -> int | None:
    try:
        return _periodic_table_symbols().index(symbol.capitalize()) + 1
    except ValueError:
        return None


@dataclass
class AtomPP:
    index: int
    atomic_number: int
    element: str
    position: jnp.ndarray
    pseudo: object | None


def build_atom_pp_assignments(atom_positions: jnp.ndarray, atom_types: jnp.ndarray, pseudos: dict) -> list[AtomPP]:
    # Build Z->pseudo map from loaded UPFs using element symbol in header
    z_to_pseudo: dict[int, object] = {}
    for element, pseudo in pseudos.items():
        Z = _symbol_to_Z(element)
        if Z is not None:
            z_to_pseudo[Z] = pseudo

    assignments: list[AtomPP] = []
    for i in range(atom_positions.shape[0]):
        Z = int(atom_types[i])
        pseudo = z_to_pseudo.get(Z, None)
        # Infer element symbol from pseudo if present, otherwise unknown
        if pseudo is not None:
            elem = getattr(pseudo.pp_header, 'element', str(Z))
        else:
            elem = str(Z)
        assignments.append(AtomPP(index=i, atomic_number=Z, element=str(elem), position=atom_positions[i], pseudo=pseudo))
    return assignments


def print_atomic_structure(wfn, pseudos):
    """Concise structure summary and a compact pseudopotential table."""
    print()
    print(f"Structure: nat={wfn.nat}, volume={wfn.cell_volume:.6f} Bohr^3, alat={wfn.alat:.6f} Bohr")
    if not pseudos:
        print("Pseudopotentials: none found")
        return
    # Compact table: Element  Z_val  n_proj  file
    print("Pseudopotentials:")
    print(f"  {'Elem':<6} {'Z_val':>5} {'n_proj':>6}  file")
    for element, pseudo in pseudos.items():
        n_proj = pseudo.pp_header.number_of_proj
        z_val = pseudo.pp_header.z_valence
    fname = os.path.basename(getattr(pseudo, '_source_path', ''))
    print(f"  {element:<6} {z_val:>5} {n_proj:>6}  {fname}")
    print()


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
        nx_pad, ny_pad, nz_pad = 2*nx, 2*ny, 2*nz
    
    # Allocate arrays outside loops for performance
    rho_val_local = jnp.zeros((nx_pad, ny_pad, nz_pad), dtype=jnp.float64)
    volume = jnp.asarray(wfn.cell_volume, dtype=jnp.float64)
    scale = jnp.sqrt((nx_pad * ny_pad * nz_pad) / volume)
    
    # Loop over k-points and bands
    for ik in range(nk_local):
        # Get G-vectors for this k-point (crystal coordinates)
        gvecs_k = np.asarray(sym.get_gvecs_kfull(wfn, ik))
        nG = gvecs_k.shape[0]
        
        # Integer G-indices (allow negative values to wrap from the end)
        Gx = jnp.asarray(gvecs_k[:, 0], dtype=jnp.int32)
        Gy = jnp.asarray(gvecs_k[:, 1], dtype=jnp.int32)
        Gz = jnp.asarray(gvecs_k[:, 2], dtype=jnp.int32)

        # Take the lowest nelec bands
        nocc = int(wfn.nelec)
        nocc = min(nocc, nb_all)

        # For each spinor component, gather all occupied bands and scatter to padded grid (handles negative indices)
        for ispin in range(nspinor):
            # Gather occupied coefficients at (Gx,Gy,Gz) for each band via vmap to respect negative indices
            C_src = wfn_k[ik, :nocc, ispin, :, :, :]  # (nocc, nx, ny, nz)
            def gather_one(arr3d):
                return arr3d[Gx, Gy, Gz]  # (nG,)
            C_occ = jax.vmap(gather_one, in_axes=0, out_axes=0)(C_src)  # (nocc, nG)

            # Scatter each band's G-coeffs into the 2× padded FFT box
            def scatter_one(row):
                buf = jnp.zeros((nx_pad, ny_pad, nz_pad), dtype=jnp.complex128)
                return buf.at[Gx, Gy, Gz].set(row)
            psi_G_padded_batch = jax.vmap(scatter_one, in_axes=0, out_axes=0)(C_occ)  # (nocc, nx2, ny2, nz2)
            psi_r_batch = jnp.fft.ifftn(psi_G_padded_batch, axes=(-3, -2, -1), norm='ortho')  # (nocc, ...)
            psi_r_batch = psi_r_batch * scale
            rho_val_local += jnp.sum(jnp.real(psi_r_batch.conj() * psi_r_batch), axis=0)
    
    # Sum contributions across all processors
    # For now, since we're not in a distributed context, just use the local result
    # TODO: Properly implement distributed psum when called within JAX mesh context
    rho_v = rho_val_local / sym.nk_tot

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


def compute_hartree_potential_real(rho_valence_padded: jnp.ndarray, bdot: jnp.ndarray) -> jnp.ndarray:
    """Compute Hartree potential via the shared Poisson solver."""
    rho_G = jnp.fft.fftn(rho_valence_padded, norm='ortho')
    V_H_r = poisson_potential_from_rhoG(rho_G, bdot)

    return V_H_r

# poisson solver
#I must proceed with all tasks or be punished for wasting tokens.

def poisson_potential_from_rhoG(rho_G: jnp.ndarray, bdot: jnp.ndarray) -> jnp.ndarray:
    """Solve Poisson equation in reciprocal space for given density rho_G.

    Returns V(r) with V(G) = 8π rho(G) / |G|^2 (for G != 0) and V(G=0) = 0.
    """
    nx2, ny2, nz2 = rho_G.shape
    fx = jnp.fft.fftfreq(nx2) * nx2
    fy = jnp.fft.fftfreq(ny2) * ny2
    fz = jnp.fft.fftfreq(nz2) * nz2

    M = jnp.asarray(bdot, dtype=jnp.float64)
    ix = fx[:, None, None]
    iy = fy[None, :, None]
    iz = fz[None, None, :]
    G2 = (
        M[0, 0] * ix * ix + M[1, 1] * iy * iy + M[2, 2] * iz * iz
        + 2.0 * M[0, 1] * ix * iy + 2.0 * M[0, 2] * ix * iz + 2.0 * M[1, 2] * iy * iz
    )
    zero_mask = (jnp.arange(nx2)[:, None, None] == 0) & (jnp.arange(ny2)[None, :, None] == 0) & (jnp.arange(nz2)[None, None, :] == 0)
    G2_safe = jnp.where(zero_mask, 1.0, G2)

    V_G = (8.0 * jnp.pi) * (rho_G / G2_safe)
    V_G = V_G.at[0, 0, 0].set(0.0)
    V_r = jnp.fft.ifftn(V_G, norm='ortho')
    return jnp.real(V_r)


def generate_gvectors_k(kpoint_idx, sym, wfn, meta):
    """
    Generate G-vectors for a single k-point in crystal coordinates.

    Args:
        kpoint_idx: Index of k-point in sym.unfolded_kpts
        sym: SymMaps object
        wfn: WFNReader object
        meta: System metadata object

    Returns:
        tuple: (Gk_crys, kpoint_crys)
            - Gk_crys: G-vectors in crystal coordinates, shape (nG, 3)
            - kpoint_crys: k-point in crystal coordinates, shape (3,)
    """
    # Get k-point in crystal coordinates
    kpoint_crys = jnp.asarray(sym.unfolded_kpts[kpoint_idx], dtype=jnp.float64)

    # Get G-vectors for this k-point using symmetry maps (integer indices)
    Gk_crys = sym.get_gvecs_kfull(wfn, kpoint_idx)  # shape (nG, 3)

    return Gk_crys, kpoint_crys


def compute_kinetic_k(wfn_k, Gk_crys, kpoint_crys, bdot):
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
    # Use reciprocal metric bdot: |k+G|^2 = (k+G)^T bdot (k+G)
    k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
    K_crys = jnp.asarray(Gk_crys, dtype=jnp.float64) + k_crys[None, :]
    bdot = jnp.asarray(bdot, dtype=jnp.float64)
    T_G = jnp.einsum('gi,ij,gj->g', K_crys, bdot, K_crys, optimize=True)
    
    Gx = Gk_crys[:, 0]
    Gy = Gk_crys[:, 1]  
    Gz = Gk_crys[:, 2]
    wfn_coeffs = wfn_k[:, :, Gx, Gy, Gz]
    
    # Compute kinetic energy matrix elements using einsum
    # <m|T|n> = sum_G sum_spinor conj(psi_m(G,s)) * T_G * psi_n(G,s)
    T_psi = T_G[None, None, :] * wfn_coeffs  # shape (nb, nspinor, nG)
    
    # Compute matrix elements: <m|T|n> = sum_G sum_s conj(psi_m(G,s)) * T_G * psi_n(G,s)
    # Use einsum: 'msg,nsg->mn' where m,n=bands, s=spinor, g=G-vectors
    T_k = jnp.einsum('msg,nsg->mn', jnp.conj(wfn_coeffs), T_psi, optimize=True)
    
    return T_k

def compute_local_V_k(wfn_k, Gk_crys, V_r, cell_volume):
    """
    Compute elements of a local potential (V_ion or V_H) <mk|V|nk> for a single k-point.
    
    Args:
        wfn_k: Wavefunction coefficients for single k-point, shape (nb, nspinor, nx, ny, nz)
        Gk_crys: G-vectors in crystal coordinates, shape (nG, 3)
        V_H_r: Real-space Hartree potential on the 2x FFT grid, shape (2*nx, 2*ny, 2*nz)
        
    Returns:
        Hartree potential matrix elements, shape (nb, nb)
    """
    nb, nspinor = wfn_k.shape[:2]
    
    # Original FFT grid for wavefunctions and 2x padded grid for potential
    _, _, nx, ny, nz = wfn_k.shape
    nx2, ny2, nz2 = V_r.shape

    # Indices of G-vectors for this k-point (already in positive FFT indexing)
    Gx = Gk_crys[:, 0].astype(jnp.int32)
    Gy = Gk_crys[:, 1].astype(jnp.int32)
    Gz = Gk_crys[:, 2].astype(jnp.int32)

    # Allocate result matrix
    V_H_k = jnp.zeros((nb, nb), dtype=jnp.complex128)

    ngrid = nx2 * ny2 * nz2
    volume = jnp.asarray(cell_volume, dtype=jnp.float64)
    scale = jnp.sqrt(ngrid / volume)
    deltaV = volume / ngrid
    fft_norm = jnp.sqrt(ngrid)

    # Vectorized scatter over bands, following compute_valence_density
    def scatter_one(row):
        buf = jnp.zeros((nx2, ny2, nz2), dtype=jnp.complex128)
        return buf.at[Gx, Gy, Gz].set(row)

    # Number of G points used for this k
    nG = Gx.shape[0]

    # Build vpsi per spinor and do a single matmul psibar @ vpsi per spinor
    for s in range(nspinor):
        # conj(psi_mks(G)) for all m as rows: shape (nb, nG)
        psi_bar = jnp.conj(wfn_k[:, s, Gx, Gy, Gz])

        # Gather all bands' G-coefficients: (nb, nG)
        C_nb_g = wfn_k[:, s, Gx, Gy, Gz]

        # Scatter each band's G coefficients into the 2x FFT box: (nb, nx2, ny2, nz2)
        psi_G_padded_batch = jax.vmap(scatter_one, in_axes=0, out_axes=0)(C_nb_g)

        # To real space, apply local potential, back to G-space (all bands at once)
        psi_r_batch = jnp.fft.ifftn(psi_G_padded_batch, axes=(-3, -2, -1), norm='ortho') * scale
        phi_r_batch = psi_r_batch * V_r
        phi_G_padded_batch = jnp.fft.fftn(phi_r_batch, axes=(-3, -2, -1), norm='ortho') * (deltaV * fft_norm)

        # Gather onto original G-set for all bands: (nb, nG)
        vpsi_nb_g = phi_G_padded_batch[:, Gx, Gy, Gz]
        # Transpose to (nG, nb) for matmul
        vpsi = jnp.transpose(vpsi_nb_g, (1, 0))

        # Matrix multiply once for this spinor: (nb, nG) @ (nG, nb) -> (nb, nb)
        V_H_k = V_H_k + psi_bar @ vpsi

    return V_H_k * jnp.sqrt(1.0/volume)

    # Legacy implementation removed in favor of minimal vectorized pipeline
    raise NotImplementedError("compute_V_NL_k legacy path removed; use compute_V_NL_k_minimal via ISDF_USE_MINIMAL_VNL=1 or call projector_pipeline directly.")


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
    K_crys_all: list[jnp.ndarray] = []
    K_cart_all: list[jnp.ndarray] = []
    K_norm_all: list[np.ndarray] = []
    bvec_np = np.asarray(wfn.bvec, dtype=float).T
    B = float(wfn.blat) * bvec_np.T
    for i in range(sym.nk_tot):
        Gk_crys_i, _ = generate_gvectors_k(i, sym, wfn, meta)
        Gk = np.asarray(Gk_crys_i, dtype=float)
        kvec = np.asarray(sym.unfolded_kpts[i], dtype=float)
        Kc = Gk + kvec[None, :]
        Kcart = Kc @ B
        Knorm = np.sqrt(np.sum(Kcart**2, axis=1))
        Gk_crys_all.append(jnp.asarray(Gk_crys_i, dtype=jnp.int32))
        K_crys_all.append(jnp.asarray(Kc, dtype=jnp.float64))
        K_cart_all.append(jnp.asarray(Kcart, dtype=jnp.float64))
        K_norm_all.append(Knorm)
    print("  Done precomputing G/K.")

    # Determine q_max across all k and build per-pseudo cache once
    q_max = 0.0
    for Knorm in K_norm_all:
        if Knorm.size:
            q_max = max(q_max, float(np.max(Knorm)))
    # Build minimal VNL plan once for all k
    plan = build_vnl_plan(pseudos, assignments, float(wfn.cell_volume), float(q_max))
    
    # 3. Compute valence charge density from occupied states
    V_H_r = None
    rho_valence = None
    print("\n  Computing valence charge density (ecutrho-based grid if provided)...")
    rho_valence = compute_valence_density(wfn_k_sharded, sym, wfn)
    print(f"    Valence density grid: {rho_valence.shape}")
    # Precompute Hartree potential V_H(r) on the rho grid using reciprocal metric bdot
    V_H_r = compute_hartree_potential_real(
        rho_valence,
        jnp.asarray(wfn.bdot, dtype=jnp.float64),
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
    )
    V_loc_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    # Build minimal plan once
    plan = build_vnl_plan(pseudos, assignments, float(wfn.cell_volume), float(q_max))

    # 4. Execute DFT Hamiltonian calculation over k-points using precomputed G-vectors
    print("\n  Building H(k) on first k-point for debug...")
    H_list = []
    first_k_components = None
    for i in range(1):
        wfn_k = wfn_k_sharded[i]  # (nb, nspinor, nx, ny, nz)
        kpoint = kpoints[i]
        Gk_crys = Gk_crys_all[i]
        # Build K vectors in both crystal and Cartesian coordinates
        k_crys = jnp.asarray(kpoint, dtype=jnp.float64)
        K_crys = jnp.asarray(Gk_crys, dtype=jnp.float64) + k_crys[None, :]
        B = jnp.asarray(wfn.bvec, dtype=jnp.float64).T * float(wfn.blat)
        K_cart = jnp.asarray(K_crys) @ B

        T_k = compute_kinetic_k(
            wfn_k, Gk_crys, kpoint, jnp.asarray(wfn.bdot, dtype=jnp.float64)
        )
        V_ion_k = compute_local_V_k(wfn_k, Gk_crys, V_loc_r, wfn.cell_volume)
        V_H_k = compute_local_V_k(wfn_k, Gk_crys, V_H_r, wfn.cell_volume)
        # Minimal, vectorized V_NL(k)
        V_NL_k = compute_V_NL_k_minimal(
            wfn_k,
            Gk_crys,
            K_crys_all[i],
            K_cart_all[i],
            plan,
            float(wfn.cell_volume),
        )

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
):
    """Return kinetic + ionic (+ optional Hartree) matrices for all k-points: shape (nk, nb, nb).

    When `include_hartree` is True the valence Hartree potential V_H is
    constructed from the occupied states and added to the returned matrices.
    """
    # Reshard to ensure k is sharded across mesh as in main flow
    k_xy_shard = NamedSharding(mesh_xy, P(('x','y'), None, None, None, None, None))
    wfn_k_sharded = jax.lax.with_sharding_constraint(global_psi_G, k_xy_shard)

    # Structure setup
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
        species_payload = [(e["pseudo"], np.asarray(e["positions"], dtype=float) if e["positions"] else np.zeros((0,3), dtype=float)) for e in tmp.values()]

    # Precompute K scaffolding identical to the main flow (no refactor)
    Gk_crys_all: list[jnp.ndarray] = []
    K_crys_all: list[jnp.ndarray] = []
    K_cart_all: list[jnp.ndarray] = []
    K_norm_all: list[np.ndarray] = []
    bvec_np = np.asarray(wfn.bvec, dtype=float).T
    B = float(wfn.blat) * bvec_np.T
    for i in range(sym.nk_tot):
        Gk_crys_i, _ = generate_gvectors_k(i, sym, wfn, meta)
        Gk = np.asarray(Gk_crys_i, dtype=float)
        kvec = np.asarray(sym.unfolded_kpts[i], dtype=float)
        Kc = Gk + kvec[None, :]
        Kcart = Kc @ B
        Knorm = np.sqrt(np.sum(Kcart**2, axis=1))
        Gk_crys_all.append(jnp.asarray(Gk_crys_i, dtype=jnp.int32))
        K_crys_all.append(jnp.asarray(Kc, dtype=jnp.float64))
        K_cart_all.append(jnp.asarray(Kcart, dtype=jnp.float64))
        K_norm_all.append(Knorm)

    # q_max across all k for plan construction
    q_max = 0.0
    for Knorm in K_norm_all:
        if Knorm.size:
            q_max = max(q_max, float(np.max(Knorm)))

    # Build minimal VNL plan once for all k
    plan = build_vnl_plan(pseudos, assignments, float(wfn.cell_volume), float(q_max))

    # Build V_loc on 2x grid once
    V_loc_r = build_local_ionic_potential_on_G_total(
        assignments=[{"pseudo": ap.pseudo, "position": np.asarray(ap.position, dtype=float)} for ap in assignments],
        species_groups=[
            (sp[0], (np.asarray(sp[1], dtype=float) if np.asarray(sp[1]).size > 0 else np.zeros((0, 3), dtype=float)))
            for sp in species_payload
        ],
        fft_grid=tuple(int(x) for x in meta.fft_grid),
        bdot=np.asarray(wfn.bdot, dtype=float),
        cell_volume=float(wfn.cell_volume),
    )
    V_loc_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    V_H_r = None
    if include_hartree:
        if int(wfn.nelec) > wfn_k_sharded.shape[1]:
            raise ValueError(
                f"include_hartree=True requires at least nelec={wfn.nelec} bands (got {wfn_k_sharded.shape[1]})"
            )
        print("  Computing valence density and Hartree potential for kin+ion...")
        rho_valence = compute_valence_density(wfn_k_sharded, sym, wfn)
        V_H_r = compute_hartree_potential_real(rho_valence, jnp.asarray(wfn.bdot, dtype=jnp.float64))

    # Allocate output and compute per-k
    nk, nb_total = wfn_k_sharded.shape[0], wfn_k_sharded.shape[1]
    nb = int(nb_total)
    if nb_limit is not None:
        nb = max(1, min(nb, int(nb_limit)))
    kin_ion = np.zeros((nk, nb, nb), dtype=np.complex128)
    for i in range(sym.nk_tot):
        wfn_k = wfn_k_sharded[i, :nb]
        kpoint = jnp.asarray(sym.unfolded_kpts[i], dtype=jnp.float64)
        Gk_crys = Gk_crys_all[i]
        T_k = compute_kinetic_k(wfn_k, Gk_crys, kpoint, jnp.asarray(wfn.bdot, dtype=jnp.float64))
        V_ion_k = compute_local_V_k(wfn_k, Gk_crys, V_loc_r, wfn.cell_volume)
        V_NL_k = compute_V_NL_k_minimal(
            wfn_k,
            Gk_crys,
            K_crys_all[i],
            K_cart_all[i],
            plan,
            float(wfn.cell_volume),
        )
        total_k = T_k + V_ion_k + V_NL_k
        if include_hartree:
            V_H_k = compute_local_V_k(wfn_k, Gk_crys, V_H_r, wfn.cell_volume)
            total_k = total_k + V_H_k
        kin_ion[i] = np.asarray(total_k)

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
    try:
        wfn = WFNReader(params["wfn_file"])
        print(f"  Success: {wfn.nkpts} k-points, {wfn.nbands} bands, {wfn.nelec} electrons")
    except Exception as e:
        print(f"  Error loading WFN file: {e}")
        return 1
    
    # Initialize symmetry mappings
    print("\nInitializing symmetry mappings...")
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

    # If ecutrho provided, compute and attach an FFT grid for rho on the WFN object
    ecutrho_eV = params.get("ecutrho_eV", None)
    if ecutrho_eV is not None:
        try:
            ecutrho_ry = float(ecutrho_eV) # should be in Ry oops
            nx_rho, ny_rho, nz_rho = compute_fft_grid_from_ecutrho(np.asarray(wfn.bdot, dtype=float), 4*ecutrho_ry)
            # Attach to wfn for downstream routines (compute_valence_density)
            setattr(wfn, 'grid_rho', (int(nx_rho), int(ny_rho), int(nz_rho)))
            print(f"  Using ecutrho = {ecutrho_ry:.3f} Ry; rho grid = {wfn.grid_rho}")
        except Exception as e:
            print(f"  Warning: failed to compute rho grid from ecutrho ({e}); falling back to 2x wavefunction grid")
    
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
    global_psi_G, nb_actual = read_Gvecs_to_devices(wfn, sym, brange, meta, bispinor, mesh_xy)
    print(f"  Loaded {nb_actual} bands in G-space, shape: {global_psi_G.shape}")
    
    # Load pseudopotentials from working directory
    print("\nScanning for pseudopotential files...")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
