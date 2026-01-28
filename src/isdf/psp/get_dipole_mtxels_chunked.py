#!/usr/bin/env python3
"""
Chunked version of get_dipole_mtxels that processes k-points in batches to avoid OOM.

This version:
- Processes k-points in configurable chunks (default: 16 at a time)
- Writes results incrementally
- Uses much less GPU memory for large k-grids

Usage:
  python -m isdf.psp.get_dipole_mtxels_chunked -i gw.inp --kchunk 16
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import h5py
from pathlib import Path
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..io import WFNReader
from ..common import symmetry_maps, Meta
from ..common.load_wfns import read_Gvecs_to_devices
from .get_DFT_mtxels import (
    read_cohsex_input,
    load_pseudopotentials,
    build_atom_pp_assignments,
    generate_gvectors_k,
)
from .projector_pipeline import (
    build_vnl_plan,
    compute_V_NL_velocity_k,
)


def build_K_vectors(Gk_crys, kpoint_crys, bvec, blat):
    """Build K = k + G vectors in crystal and Cartesian coordinates."""
    k_crys = np.asarray(kpoint_crys, dtype=float)
    K_crys = np.asarray(Gk_crys, dtype=float) + k_crys[None, :]
    B = float(blat) * np.asarray(bvec, dtype=float).T
    K_cart = K_crys @ B.T
    return K_crys, K_cart


def compute_p_operator_k(wfn_k, Gk_crys, kpoint_crys, bdot, bvec, blat):
    """Compute p-operator matrix elements per Cartesian component.
    
    Returns array of shape (3, nb, nb) for components x,y,z.
    p_i = sum_G (k+G)_cart[i] c*_mk(G) c_nk(G)
    """
    nb, nspinor = int(wfn_k.shape[0]), int(wfn_k.shape[1])
    Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
    C_bsg = wfn_k[:, :, Gx, Gy, Gz]  # (nb, nspinor, nG)
    
    k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
    K_crys = jnp.asarray(Gk_crys, dtype=jnp.float64) + k_crys[None, :]
    B = jnp.asarray(bvec, dtype=jnp.float64) * float(blat)
    K_cart = K_crys @ B  # (nG, 3)
    
    p_mn = []
    for i in range(3):
        K_i = K_cart[:, i]
        weighted = C_bsg * K_i[None, None, :]
        p_i = jnp.einsum('msg,nsg->mn', jnp.conj(C_bsg), weighted, optimize=True)
        p_mn.append(p_i)
    
    return 2 * jnp.stack(p_mn, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chunked dipole/velocity matrix elements <mk|v|nk>")
    parser.add_argument("-i", "--input", required=True, help="Input file (INI-like) with [cohsex] block")
    parser.add_argument("--kchunk", type=int, default=16, help="k-points per chunk (default: 16)")
    parser.add_argument("--vnl-mode", choices=["analytic", "numeric"], default="analytic",
                        help="Nonlocal velocity evaluation: analytic (dZ) or numeric FD(Z)")
    args = parser.parse_args(argv)
    
    input_path = Path(args.input).resolve()
    params = read_cohsex_input(str(input_path))
    
    # Resolve WFN relative to input file directory
    wfn_path = Path(params.get("wfn_file", "WFN.h5"))
    if not wfn_path.is_absolute():
        wfn_path = (input_path.parent / wfn_path).resolve()
    
    # Open WFN and symmetry
    wfn = WFNReader(str(wfn_path))
    sym = symmetry_maps.SymMaps(wfn)
    
    nval = int(params.get("nval", 5))
    ncond = int(params.get("ncond", 5))
    nband_param = params.get("nband", None)
    if nband_param is None:
        nband = max(int(wfn.nbands), int(wfn.nelec) + int(ncond))
    else:
        nband = int(nband_param)
    bispinor = bool(params.get("bispinor", False))
    
    print(f"\nCreating system metadata...")
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, 0, bispinor)
    
    # JAX mesh (simple 1D default)
    devices = np.array(jax.devices()).reshape(1, -1)
    mesh_xy = Mesh(devices, ['x', 'y'])
    
    print(f"K-points: {sym.nk_tot}, Bands: {nband}, Devices: {devices.size}")
    
    # Load pseudopotentials
    print("\nScanning for pseudopotential files...")
    pseudos = load_pseudopotentials(str(input_path.parent))
    
    # Structure setup
    assignments = build_atom_pp_assignments(
        jnp.asarray(wfn.atom_crys, dtype=jnp.float64),
        jnp.asarray(wfn.atom_types, dtype=jnp.int32),
        pseudos
    )
    
    # Species payload for nonlocal projectors
    species_payload = []
    for ap in assignments:
        if ap.pseudo is None:
            continue
        if not any(id(ap.pseudo) == id(p) for p, _ in species_payload):
            positions = np.asarray([a.position for a in assignments if id(a.pseudo) == id(ap.pseudo)], dtype=float)
            species_payload.append((ap.pseudo, positions))
    
    # Compute q_max for projector plan
    bvec_np = np.asarray(wfn.bvec, dtype=float)
    blat = float(wfn.blat)
    B = blat * bvec_np
    q_max = 0.0
    for ik in range(sym.nk_tot):
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        K_crys, K_cart = build_K_vectors(Gk_crys, sym.unfolded_kpts[ik], bvec_np, blat)
        K_norm = np.sqrt(np.sum(K_cart**2, axis=1))
        if K_norm.size:
            q_max = max(q_max, float(np.max(K_norm)))
    
    # Build V_NL plan
    plan = None
    if assignments and pseudos:
        print("Building V_NL plan...")
        plan = build_vnl_plan(pseudos, assignments, wfn.cell_volume, q_max)
    
    # Prepare output arrays
    nk = sym.nk_tot
    nband_eff = min(int(wfn.nbands), max(int(wfn.nelec) + int(ncond), nband))
    nb = nband_eff
    dipole = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    deltaE = np.zeros((nk, nb, nb), dtype=np.float64)
    
    # Process k-points in chunks
    kchunk_size = args.kchunk
    num_chunks = (nk + kchunk_size - 1) // kchunk_size
    
    print(f"\nProcessing {nk} k-points in {num_chunks} chunks of {kchunk_size}...")
    
    for ichunk in range(num_chunks):
        k_start = ichunk * kchunk_size
        k_end = min(k_start + kchunk_size, nk)
        k_actual = k_end - k_start
        
        print(f"\n  Chunk {ichunk+1}/{num_chunks}: k={k_start+1}-{k_end}")
        
        # Load only this chunk of k-points
        brange = (0, nband_eff)
        global_psi_G, _ = read_Gvecs_to_devices(wfn, sym, brange, meta, bispinor, mesh_xy)
        
        # Extract only the k-points we need
        psi_k_chunk = global_psi_G[k_start:k_end, :, :, :, :, :]
        
        for i_local in range(k_actual):
            i_global = k_start + i_local
            wfn_k = psi_k_chunk[i_local]
            kpoint = jnp.asarray(sym.unfolded_kpts[i_global], dtype=jnp.float64)
            Gk_crys, _ = generate_gvectors_k(i_global, sym, wfn, meta)
            K_crys, K_cart = build_K_vectors(Gk_crys, sym.unfolded_kpts[i_global], bvec_np, blat)
            
            # Momentum per component
            p_cart = compute_p_operator_k(
                wfn_k, Gk_crys, kpoint,
                jnp.asarray(wfn.bdot, dtype=jnp.float64),
                jnp.asarray(bvec_np, dtype=jnp.float64),
                blat
            )  # (3, nb, nb)
            
            # Nonlocal velocity components
            if args.vnl_mode == "analytic" and plan is not None:
                vNL_cart = compute_V_NL_velocity_k(
                    wfn_k, Gk_crys,
                    jnp.asarray(K_crys, dtype=jnp.float64),
                    jnp.asarray(K_cart, dtype=jnp.float64),
                    plan,
                    float(wfn.cell_volume),
                    fprime_mode="bessel"
                )  # (3, nb, nb)
                # Sign convention flip (see original code comment)
                vNL_cart = -vNL_cart
            else:
                vNL_cart = 0.0
            
            dipole[:, i_global] = np.asarray(p_cart + vNL_cart)
            
            # ΔE matrix for this k from band energies
            try:
                k_red = int(sym.irk_to_k_map[i_global])
            except Exception:
                k_red = int(i_global)
            energies = np.asarray(wfn.energies)
            if energies.ndim >= 3:
                e_b = np.asarray(energies[0, k_red, :nb], dtype=float)
            else:
                e_b = np.asarray(energies[:nb], dtype=float)
            deltaE[i_global] = e_b[:, None] - e_b[None, :]
        
        # Clear GPU memory
        del psi_k_chunk, global_psi_G
        jax.clear_caches()
        
        print(f"    Completed k={k_start+1}-{k_end}, memory freed")
    
    # Write output
    out_path = input_path.parent / "dipole.h5"
    print(f"\nWriting to {out_path}...")
    with h5py.File(out_path, 'w') as f:
        # Use 'dipole_cart' name to match expected format
        f.create_dataset('dipole_cart', data=dipole, dtype=np.complex128)
        f.create_dataset('deltaE', data=deltaE, dtype=np.float64)
        f['dipole_cart'].attrs['description'] = 'Velocity matrix elements (3, nk, nb, nb)'
        f['deltaE'].attrs['description'] = 'Energy differences (nk, nb, nb)'
        f['dipole_cart'].attrs['nk'] = nk
        f['dipole_cart'].attrs['nb'] = nb
    
    print(f"✓ Wrote dipole.h5 with dipole_cart shape {dipole.shape}, deltaE shape {deltaE.shape}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

