#!/usr/bin/env python3
"""
Chunked kin+ion computation: T + V_loc + V_NL for all k-points.

Processes k-points one at a time to avoid OOM on large k-grids.
Hartree (V_H) is excluded; it enters via the self-energy in the GW step.

Usage:
  python -m gw.kin_ion_io_chunked -i gw.inp -o kin_ion.h5 [--sys_dim 3]
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

from file_io import WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
import common.timing as timing

from psp.get_DFT_mtxels import (
    read_cohsex_input,
    load_pseudopotentials,
    build_atom_pp_assignments,
    build_local_ionic_potential_on_G_total,
    generate_gvectors_k,
    compute_kinetic_k,
    compute_local_V_k,
)
from psp.projector_pipeline import build_vnl_plan, compute_V_NL_k_minimal
from psp.operator_checks import validate_operator_inputs


def _resolve_against(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, plan, wfn, g_mask=None):
    """Compute T + V_loc + V_NL for a single k-point.

    Parameters
    ----------
    wfn_k : (nb, nspinor, nx, ny, nz) — wavefunctions in FFT box
    Gk_crys : (nG, 3) int — G-vector indices for this k
    kvec : (3,) float — k-point in crystal coords
    V_loc_r : (nx, ny, nz) — local ionic potential on FFT grid
    plan : VNL plan from build_vnl_plan (or None to skip V_NL)
    wfn : WFNReader (for bdot, bvec, blat, cell_volume)
    g_mask : (nG,) float or None — mask for padded G-vectors
    """
    bdot_np = np.asarray(wfn.bdot, dtype=float)
    T_k = compute_kinetic_k(wfn_k, Gk_crys, kvec, bdot_np, g_mask=g_mask)
    V_loc_k = compute_local_V_k(
        wfn_k, Gk_crys, V_loc_r, wfn.cell_volume, g_mask=g_mask
    )

    V_NL_k = 0.0
    if plan is not None:
        bvec_np = np.asarray(wfn.bvec, dtype=float).T
        B = float(wfn.blat) * bvec_np.T
        K_crys = np.asarray(Gk_crys, dtype=float) + np.asarray(kvec, dtype=float)[None, :]
        K_cart = K_crys @ B
        V_NL_k = compute_V_NL_k_minimal(
            wfn_k, Gk_crys, K_crys, K_cart, plan,
            float(wfn.cell_volume), g_mask=g_mask,
        )

    return T_k + V_loc_k + V_NL_k


def main(argv=None):
    argp = argparse.ArgumentParser(description="Chunked kin+ion computation")
    argp.add_argument("-i", "--input", required=True, help="cohsex / GW input file")
    argp.add_argument("-o", "--output", default=None, help="output HDF5 (default: kin_ion.h5)")
    argp.add_argument("-n", "--nb", type=int, default=None, help="number of bands")
    argp.add_argument("--sys_dim", type=int, default=None,
                      help="system dimensionality: 0, 2, or 3 (overrides input file)")
    argp.add_argument("--pseudo_dir", default=None,
                      help="directory containing *.upf files (default: input file dir)")
    args = argp.parse_args(argv)

    timing.reset()
    print("== Chunked kin_ion_io ==")
    input_dir = os.path.dirname(os.path.abspath(args.input))

    # ---- parse input ----
    params = read_cohsex_input(args.input)
    wfn_path = _resolve_against(params.get("wfn_file", "WFN.h5"), input_dir)

    # sys_dim: CLI flag > input file > default 3
    sys_dim = args.sys_dim
    if sys_dim is None:
        sys_dim = int(params.get("sys_dim", 3))

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
    nx, ny, nz = meta.fft_grid
    print(f"Bands: {nb_eff}, FFT grid: {meta.fft_grid}, k-points: {sym.nk_tot}")
    print(f"sys_dim: {sys_dim}")
    print(f"Devices: {jax.device_count()}")

    # ---- load pseudopotentials ----
    pseudo_dir = args.pseudo_dir or input_dir
    pseudos = load_pseudopotentials(pseudo_dir)
    if not pseudos:
        # Also try the QE subdirectory (common sandbox layout)
        for fallback in [os.path.join(input_dir, '..', 'qe', 'scf'),
                         os.path.join(input_dir, '..', 'qe', 'nscf')]:
            pseudos = load_pseudopotentials(fallback)
            if pseudos:
                print(f"Found pseudopotentials in {fallback}")
                break

    # ---- validate (will raise if pseudos missing or sys_dim invalid) ----
    ctx = validate_operator_inputs(
        pseudos=pseudos, wfn=wfn, sys_dim=sys_dim,
        caller="kin_ion_io_chunked",
    )
    print(f"Pseudopotentials: {list(ctx.pseudos.keys())}")
    print(f"Coulomb truncation: {'2D slab' if ctx.truncation_2d else '3D bulk'}")

    # ---- build structure data ----
    atom_positions = np.asarray(wfn.atom_crys, dtype=float)
    atom_types = np.asarray(wfn.atom_types, dtype=int)
    assignments = build_atom_pp_assignments(
        jnp.asarray(atom_positions), jnp.asarray(atom_types), pseudos
    )
    species_tmp = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        key = id(ap.pseudo)
        entry = species_tmp.setdefault(key, {"pseudo": ap.pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    species_payload = [
        (e["pseudo"], np.asarray(e["positions"], dtype=float)
         if e["positions"] else np.zeros((0, 3), dtype=float))
        for e in species_tmp.values()
    ]

    # ---- build V_loc on the FFT grid (k-independent) ----
    print("Building V_loc...")
    with timing.section("build_V_loc"):
        V_loc_r = build_local_ionic_potential_on_G_total(
            assignments=[
                {"pseudo": ap.pseudo, "position": np.asarray(ap.position, dtype=float)}
                for ap in assignments
            ],
            species_groups=species_payload,
            fft_grid=(nx, ny, nz),
            bdot=np.asarray(wfn.bdot, dtype=float),
            cell_volume=float(wfn.cell_volume),
            bvec=np.asarray(wfn.bvec, dtype=float),
            blat=float(wfn.blat),
            truncation_2d=ctx.truncation_2d,
        )
        V_loc_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    # ---- compute q_max for V_NL plan ----
    bvec_np = np.asarray(wfn.bvec, dtype=float).T
    B = float(wfn.blat) * bvec_np.T
    q_max = 0.0
    for ik in range(sym.nk_tot):
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        Gk = np.asarray(Gk_crys, dtype=float)
        kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
        K_cart = (Gk + kvec[None, :]) @ B
        K_norm = np.sqrt(np.sum(K_cart ** 2, axis=1))
        if K_norm.size:
            q_max = max(q_max, float(np.max(K_norm)))

    plan = None
    if assignments and pseudos:
        print("Building V_NL plan...")
        plan = build_vnl_plan(pseudos, assignments, wfn.cell_volume, q_max)

    # ---- compute kin+ion per k-point ----
    out_path = args.output or os.path.join(input_dir, "kin_ion.h5")
    kin_ion_all = np.zeros((sym.nk_tot, nb_eff, nb_eff), dtype=np.complex128)

    print(f"\nProcessing {sym.nk_tot} k-points...")
    for ik in range(sym.nk_tot):
        if ik % 16 == 0:
            print(f"  k={ik + 1}/{sym.nk_tot}...")

        with timing.section(f"k{ik}"):
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb_eff)
            kvec = sym.unfolded_kpts[ik]
            Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)

            H_k = get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, plan, wfn)
            kin_ion_all[ik] = np.asarray(H_k)
            del wfn_k

    # ---- write output ----
    print(f"\nWriting to {out_path}...")
    with timing.section("write_h5"):
        with h5py.File(out_path, "w") as f:
            ds = f.create_dataset("kin_ion", data=kin_ion_all, dtype=np.complex128)
            ds.attrs["description"] = "T + V_loc + V_NL matrix elements"
            ds.attrs["nk"] = sym.nk_tot
            ds.attrs["nb"] = nb_eff
            ds.attrs["sys_dim"] = sys_dim
            ds.attrs["truncation_2d"] = ctx.truncation_2d
            ds.attrs["pseudopotentials"] = str(list(pseudos.keys()))

    print(f"Wrote kin_ion.h5: shape {kin_ion_all.shape}, sys_dim={sys_dim}")
    timing.report(title="--- Timing (seconds) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
