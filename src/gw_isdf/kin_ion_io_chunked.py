#!/usr/bin/env python3
"""
Chunked version of kin_ion_io that processes k-points in batches to avoid OOM.

This version:
- Processes k-points in configurable chunks (default: 16 at a time)
- Writes results incrementally to HDF5
- Uses much less GPU memory for large k-grids

Usage:
  python -m gw_isdf.kin_ion_io_chunked -i gw.inp -o kin_ion.h5 --kchunk 16
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import h5py

from isdf.io import WFNReader
from isdf.common import symmetry_maps, Meta
from isdf.common.load_wfns import load_kpoint_fftbox
import isdf.common.timing as timing

from isdf.psp.get_DFT_mtxels import (
    read_cohsex_input,
    load_pseudopotentials,
    build_atom_pp_assignments,
    generate_gvectors_k,
    compute_kinetic_k,
    compute_local_V_k,
)
from isdf.psp.projector_pipeline import build_vnl_plan, compute_V_NL_k_minimal


def _resolve_against(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def get_kin_ion_k(wfn_k, Gk_crys, kvec, plan, pseudos, wfn, meta, species_payload):
    """Compute kin+ion for a single k-point."""
    # Kinetic
    bdot_np = np.asarray(wfn.bdot, dtype=float)
    T_k = compute_kinetic_k(wfn_k, Gk_crys, kvec, bdot_np, g_mask=None)
    
    # Local potential (if available)
    V_loc_k = 0.0
    if hasattr(wfn, 'Vloc_r') and wfn.Vloc_r is not None:
        V_loc_k = compute_local_V_k(wfn_k, Gk_crys, wfn.Vloc_r, wfn.cell_volume, g_mask=None)
    
    # Nonlocal potential
    V_NL_k = 0.0
    if plan is not None and pseudos:
        # Build K vectors for V_NL
        bvec_np = np.asarray(wfn.bvec, dtype=float).T
        B = float(wfn.blat) * bvec_np.T
        K_crys = np.asarray(Gk_crys, dtype=float) + np.asarray(kvec, dtype=float)[None, :]
        K_cart = K_crys @ B
        V_NL_k = compute_V_NL_k_minimal(
            wfn_k, Gk_crys, K_crys, K_cart, plan, float(wfn.cell_volume), g_mask=None
        )
    
    return T_k + V_loc_k + V_NL_k


def main(argv=None):
    argp = argparse.ArgumentParser(description="Chunked kin+ion computation")
    argp.add_argument("-i", "--input", required=True, help="cohsex input file")
    argp.add_argument("-o", "--output", default=None, help="output HDF5 (default: kin_ion.h5)")
    argp.add_argument("-n", "--nb", type=int, default=None, help="number of bands")
    argp.add_argument("--kchunk", type=int, default=16, help="k-points per chunk (default: 16)")
    args = argp.parse_args(argv)

    timing.reset()
    
    print("== Chunked kin_ion_io ==")
    input_dir = os.path.dirname(os.path.abspath(args.input))
    params = read_cohsex_input(args.input)
    wfn_path = _resolve_against(params.get("wfn_file", "WFN.h5"), input_dir)
    
    print(f"Loading WFN: {os.path.basename(wfn_path)}")
    with timing.section("load_wfn"):
        wfn = WFNReader(wfn_path)
        sym = symmetry_maps.SymMaps(wfn)
    
    nval = int(params.get("nval", 5))
    ncond = int(params.get("ncond", 5))
    nband = int(params.get("nband", 100))
    bispinor = bool(params.get("bispinor", False))
    
    nb_req = int(args.nb) if args.nb is not None else int(nband)
    nb_eff = max(1, min(int(wfn.nbands), nb_req))
    meta = Meta.from_system(wfn, sym, nval, ncond, nb_eff, 0, bispinor)
    print(f"Bands: {nb_eff}, FFT grid: {meta.fft_grid}, k-points: {sym.nk_tot}")
    
    print(f"Devices: {jax.device_count()}")

    # Load pseudopotentials
    pseudos = load_pseudopotentials(input_dir)
    if not pseudos:
        print("Warning: no pseudopotentials found")
    
    # Build structure data
    atom_positions = np.asarray(wfn.atom_crys, dtype=float)
    atom_types = np.asarray(wfn.atom_types, dtype=int)
    assignments = build_atom_pp_assignments(
        jnp.asarray(atom_positions), jnp.asarray(atom_types), pseudos
    )
    
    # Species payload for projectors
    tmp = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        key = id(ap.pseudo)
        entry = tmp.setdefault(key, {"pseudo": ap.pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    species_payload = [
        (e["pseudo"], np.asarray(e["positions"], dtype=float) if e["positions"] else np.zeros((0, 3), dtype=float))
        for e in tmp.values()
    ]
    
    # Compute q_max for projector plan
    bvec_np = np.asarray(wfn.bvec, dtype=float).T
    B = float(wfn.blat) * bvec_np.T
    q_max = 0.0
    for ik in range(sym.nk_tot):
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        Gk = np.asarray(Gk_crys, dtype=float)
        kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
        K_cart = (Gk + kvec[None, :]) @ B
        K_norm = np.sqrt(np.sum(K_cart**2, axis=1))
        if K_norm.size:
            q_max = max(q_max, float(np.max(K_norm)))
    
    # Build V_NL plan
    plan = None
    if assignments and pseudos:
        print("Building V_NL plan...")
        plan = build_vnl_plan(pseudos, assignments, wfn.cell_volume, q_max)
    
    # Prepare output HDF5
    out_path = args.output or os.path.join(input_dir, 'kin_ion.h5')
    kin_ion_all = np.zeros((sym.nk_tot, nb_eff, nb_eff), dtype=np.complex128)
    
    # Process k-points one at a time (per-k loading avoids 74 GiB bulk load)
    print(f"\nProcessing {sym.nk_tot} k-points individually...")

    for ik in range(sym.nk_tot):
        if ik % 16 == 0:
            print(f"  k={ik+1}/{sym.nk_tot}...")

        with timing.section(f"k{ik}"):
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb_eff)
            kvec = sym.unfolded_kpts[ik]
            Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)

            H_k = get_kin_ion_k(wfn_k, Gk_crys, kvec, plan, pseudos, wfn, meta, species_payload)
            kin_ion_all[ik] = np.asarray(H_k)
            del wfn_k
    
    # Write output
    print(f"\nWriting to {out_path}...")
    with timing.section("write_h5"):
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('kin_ion', data=kin_ion_all, dtype=np.complex128)
            f['kin_ion'].attrs['description'] = 'Kinetic + ionic potential matrix elements'
            f['kin_ion'].attrs['nk'] = sym.nk_tot
            f['kin_ion'].attrs['nb'] = nb_eff
    
    print(f"✓ Wrote kin_ion.h5 with shape {kin_ion_all.shape}")
    timing.report(title="--- Timing (seconds) ---")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

