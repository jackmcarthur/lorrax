#!/usr/bin/env python3
"""
Chunked version of get_dipole_mtxels that processes k-points in batches to avoid OOM.

This version:
- Processes k-points in configurable chunks (default: 16 at a time)
- Writes results incrementally
- Uses much less GPU memory for large k-grids

Usage:
  python -m psp.get_dipole_mtxels_chunked -i gw.inp --kchunk 16
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import h5py
from pathlib import Path

from file_io import WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from .get_DFT_mtxels import (
    read_cohsex_input,
    load_pseudopotentials,
    generate_gvectors_k,
)
from .dft_operators import gather_psi_G_from_crys, momentum_matrix_k
import psp.vnl_ops as vnl_ops


def compute_p_operator_k(wfn_k, Gk_crys, kpoint_crys, bdot, bvec, blat):
    """Compute p-operator matrix elements per Cartesian component.
    
    Returns array of shape (3, nb, nb) for components x,y,z.
    p_i = sum_G (k+G)_cart[i] c*_mk(G) c_nk(G)
    """
    psi_G = gather_psi_G_from_crys(wfn_k, Gk_crys)
    G_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
    k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
    B = jnp.asarray(bvec, dtype=jnp.float64) * float(blat)
    return momentum_matrix_k(psi_G, G_int, k_crys, B)


def compute_vnl_velocity_cart(wfn_k, Gk_crys, kpoint_crys, vnl_setup):
    """Return dV_NL/dK_cart using the unified JAX VNL path."""
    kdata = vnl_ops.build_vnl_kdata_from_kvec(
        np.asarray(kpoint_crys, dtype=float),
        np.asarray(Gk_crys, dtype=int),
        vnl_setup,
        compute_dZ=True,
    )
    psi_G = gather_psi_G_from_crys(wfn_k, Gk_crys)
    return vnl_ops.vnl_velocity_matrix(psi_G, kdata.Z, kdata.dZ, kdata.E_super)


def compute_vnl_matrix_from_setup(wfn_k, Gk_crys, kpoint_crys, vnl_setup):
    """Return <m|V_NL(k)|n> using the unified JAX VNL path."""
    kdata = vnl_ops.build_vnl_kdata_from_kvec(
        np.asarray(kpoint_crys, dtype=float),
        np.asarray(Gk_crys, dtype=int),
        vnl_setup,
        compute_dZ=False,
    )
    psi_G = gather_psi_G_from_crys(wfn_k, Gk_crys)
    return vnl_ops.vnl_matrix(psi_G, kdata.Z, kdata.E_super)


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
    
    print(f"K-points: {sym.nk_tot}, Bands: {nband}, Devices: {jax.device_count()}")
    
    # Load pseudopotentials
    print("\nScanning for pseudopotential files...")
    pseudos = load_pseudopotentials(str(input_path.parent))
    
    # Build unified V_NL setup once; the custom table/JVP plumbing stays centralized here.
    print("Building unified V_NL setup...")
    vnl_setup = vnl_ops.build_vnl_setup(
        wfn,
        sym,
        meta,
        pseudos,
        nspinor=int(meta.nspinor),
    )
    
    # Prepare output arrays
    nk = sym.nk_tot
    nband_eff = min(int(wfn.nbands), max(int(wfn.nelec) + int(ncond), nband))
    nb = nband_eff
    dipole = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    deltaE = np.zeros((nk, nb, nb), dtype=np.float64)
    
    # Process k-points one at a time (per-k loading avoids 74 GiB bulk load)
    print(f"\nProcessing {nk} k-points individually...")

    energies = np.asarray(wfn.energies)

    for ik in range(nk):
        if ik % 16 == 0:
            print(f"  k={ik+1}/{nk}...")

        wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
        kpoint = jnp.asarray(sym.unfolded_kpts[ik], dtype=jnp.float64)
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        # Momentum per component
        p_cart = compute_p_operator_k(
            wfn_k, Gk_crys, kpoint,
            jnp.asarray(wfn.bdot, dtype=jnp.float64),
            jnp.asarray(wfn.bvec, dtype=jnp.float64),
            float(wfn.blat),
        )  # (3, nb, nb)

        # Nonlocal velocity components
        if args.vnl_mode == "analytic":
            vNL_cart = compute_vnl_velocity_cart(wfn_k, Gk_crys, kpoint, vnl_setup)
            vNL_cart = -vNL_cart
        else:
            B = np.asarray(wfn.bvec, dtype=float) * float(wfn.blat)
            Binv = np.linalg.inv(B)
            vNL_cart = np.zeros((3, nb, nb), dtype=np.complex128)
            K_cart_this = (np.asarray(Gk_crys, dtype=float) + np.asarray(kpoint, dtype=float)[None, :]) @ B
            K_med = float(np.median(np.linalg.norm(K_cart_this, axis=1))) if K_cart_this.size else 1.0
            h_base = 1e-5 * max(K_med, 1.0)
            for ic in range(3):
                delta = np.zeros((3,), dtype=float)
                delta[ic] = h_base
                delta_crys = delta @ Binv
                vp = compute_vnl_matrix_from_setup(
                    wfn_k, Gk_crys, np.asarray(kpoint, dtype=float) + delta_crys, vnl_setup,
                )
                vm = compute_vnl_matrix_from_setup(
                    wfn_k, Gk_crys, np.asarray(kpoint, dtype=float) - delta_crys, vnl_setup,
                )
                vNL_cart[ic] = - (vp - vm) / (2.0 * h_base)

        dipole[:, ik] = np.asarray(p_cart + vNL_cart)

        # ΔE matrix for this k from band energies
        try:
            k_red = int(sym.irk_to_k_map[ik])
        except Exception:
            k_red = int(ik)
        if energies.ndim >= 3:
            e_b = np.asarray(energies[0, k_red, :nb], dtype=float)
        else:
            e_b = np.asarray(energies[:nb], dtype=float)
        deltaE[ik] = e_b[:, None] - e_b[None, :]
        del wfn_k
    
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
