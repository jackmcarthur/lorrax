#!/usr/bin/env python3
"""kin+ion computation: T + V_loc + V_NL (+ V_H) for all k → kin_ion.h5.

**Default output is ``H_DFT − V_xc``**: kinetic + ionic (local and
nonlocal) *and* the mean-field Hartree potential, all evaluated exactly
on the plane-wave/FFT grid.  The file is stamped ``has_hartree=True`` and
the GW driver then skips its own ``sig_h`` so nothing is double counted
(see :func:`file_io.kin_ion.kin_ion_has_hartree`).

Why V_H belongs here and not in the ISDF Σ stage
------------------------------------------------
``H₀ = ⟨T+V_ion+V_NL⟩ + ⟨V_H⟩`` is a catastrophic cancellation: for MoS₂
the two terms are −502 eV and +461 eV and their sum is −42 eV.  The
first was always exact (plane waves); the second used to be an ISDF
centroid quadrature, so a few-percent basis error landed on H₀ as tens
of eV and wrecked every QP energy while every stage still reported
success.  Evaluating ⟨mk|V_H|nk⟩ here — the same ``compute_local_V_k``
route V_loc already takes — closes that cancellation analytically inside
one routine and makes eqp0 **independent of the centroid count**.

Every physical convention is inherited from the run's own input deck
(``sys_dim`` → Coulomb truncation, ``nval``/``ncond``/``nband`` → band
window, ``bispinor``, the WFN's FFT grid), and the resolved values are
stamped into the output so the generator and the GW run cannot silently
disagree.  ``--sys_dim`` may only *confirm* the deck, never contradict it.

Usage:
  python -m gw.kin_ion_io -i cohsex.in -o kin_ion.h5 [-n NB] [--no-hartree]
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

from file_io import WfnLoader as WFNReader
from common import symmetry_maps, Meta
from common.wfn_transforms import load_kpoint_fftbox
import common.timing as timing

from psp.pseudos import load_pseudopotentials, build_atom_pp_assignments
from psp.dft_operators import generate_gvectors_k, vnl_matrix_from_kdata
from psp.radial.build_projectors_qe import build_local_ionic_potential_on_G_total
from gw.gw_config import read_lorrax_input as read_cohsex_input
from psp.get_DFT_mtxels import (
    compute_kinetic_k,
    compute_local_V_k,
    build_hartree_potential,
    spin_degeneracy_factor,
    valence_density_from_kpoint,
)
from psp.operator_checks import validate_operator_inputs
import psp.vnl_ops as vnl_ops


def _resolve_against(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, vnl_setup, wfn, g_mask=None,
                  V_H_r=None):
    """Compute T + V_loc + V_NL (+ V_H) for a single k-point.

    Parameters
    ----------
    wfn_k : (nb, nspinor, nx, ny, nz) — wavefunctions in FFT box
    Gk_crys : (nG, 3) int — G-vector indices for this k
    kvec : (3,) float — k-point in crystal coords
    V_loc_r : (nx, ny, nz) — local ionic potential on FFT grid
    vnl_setup : VNLSetup from vnl_ops.build_vnl_setup (or None to skip V_NL)
    wfn : WFNReader (for bdot, bvec, blat, cell_volume)
    g_mask : (nG,) float or None — mask for padded G-vectors
    V_H_r : (nx, ny, nz) or None — mean-field Hartree potential on the
        SAME FFT grid as ``V_loc_r``.  Folded in through the identical
        local-potential route, so H₀'s ~500 eV cancellation closes inside
        one exact routine instead of across two numerical schemes.
    """
    bdot_np = np.asarray(wfn.bdot, dtype=float)
    T_k = compute_kinetic_k(wfn_k, Gk_crys, kvec, bdot_np, g_mask=g_mask)
    V_loc_k = compute_local_V_k(
        wfn_k, Gk_crys, V_loc_r, wfn.cell_volume, g_mask=g_mask
    )

    V_NL_k = 0.0
    if vnl_setup is not None:
        kdata = vnl_ops.build_vnl_kdata_from_kvec(
            np.asarray(kvec, dtype=float),
            np.asarray(Gk_crys, dtype=int),
            vnl_setup,
        )
        V_NL_k = vnl_matrix_from_kdata(wfn_k, Gk_crys, kdata)

    H_k = T_k + V_loc_k + V_NL_k
    if V_H_r is not None:
        H_k = H_k + compute_local_V_k(
            wfn_k, Gk_crys, V_H_r, wfn.cell_volume, g_mask=g_mask
        )
    return H_k


def build_valence_density_chunked(wfn, sym, meta, nocc: int, print_fn=print):
    """Accumulate ρ_v(r) k-by-k on the ψ FFT box grid (bounded memory).

    Mirrors :func:`psp.get_DFT_mtxels.compute_valence_density` — same
    per-k quadrature helper — but never holds more than one k-point of
    ψ, which is what makes the 144-k / 400-band production decks fit on
    one node.  The unfolded full BZ carries uniform weights ``1/nk_tot``
    by construction (``SymMaps`` expands the IBZ to the full mesh), so
    no ``kweights`` lookup is needed here.
    """
    nx, ny, nz = meta.fft_grid
    rho = jnp.zeros((int(nx), int(ny), int(nz)), dtype=jnp.float64)
    f_spin = spin_degeneracy_factor(wfn)
    wk = 1.0 / float(sym.nk_tot)
    for ik in range(sym.nk_tot):
        if ik % 32 == 0:
            print_fn(f"    rho: k={ik + 1}/{sym.nk_tot}...")
        psi_k = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
        rho = rho + valence_density_from_kpoint(
            psi_k, nocc=nocc, weight=wk,
            cell_volume=float(wfn.cell_volume), spin_degeneracy=f_spin,
        )
        del psi_k
    return rho


def main(argv=None):
    argp = argparse.ArgumentParser(description="Chunked kin+ion computation")
    argp.add_argument("-i", "--input", required=True, help="cohsex / GW input file")
    argp.add_argument("-o", "--output", default=None, help="output HDF5 (default: kin_ion.h5)")
    argp.add_argument("-n", "--nb", type=int, default=None, help="number of bands")
    argp.add_argument("--sys_dim", type=int, default=None,
                      help="system dimensionality: 0, 2, or 3.  Must AGREE with "
                           "the input file when the file specifies it.")
    argp.add_argument("--pseudo_dir", default=None,
                      help="directory containing *.upf files (default: input file dir)")
    argp.add_argument("--hartree", dest="hartree", action="store_true", default=True,
                      help="fold the exact mean-field V_H into kin_ion (DEFAULT)")
    argp.add_argument("--no-hartree", dest="hartree", action="store_false",
                      help="legacy mode: T+V_loc+V_NL only; the GW run then adds "
                           "its own ISDF V_H (centroid-count dependent H0)")
    args = argp.parse_args(argv)

    timing.reset()
    print("== kin_ion_io ==")
    input_dir = os.path.dirname(os.path.abspath(args.input))

    # ---- parse input: the deck is the single source of truth ----
    # Everything physical (Coulomb truncation, band window, spinor
    # treatment, FFT grid) is inherited from the same file the GW run
    # reads.  A CLI flag may only confirm the deck, never silently
    # override it, so the generator and the run cannot disagree.
    params = read_cohsex_input(args.input)
    wfn_path = _resolve_against(params.get("wfn_file", "WFN.h5"), input_dir)

    sys_dim_file = params.get("sys_dim")
    if args.sys_dim is not None and sys_dim_file is not None and (
        int(args.sys_dim) != int(sys_dim_file)
    ):
        raise SystemExit(
            f"--sys_dim {args.sys_dim} contradicts sys_dim={int(sys_dim_file)} in "
            f"{os.path.basename(args.input)}.  kin_ion.h5 carries the Coulomb "
            "truncation convention for the whole run — fix the deck instead."
        )
    sys_dim = int(args.sys_dim if args.sys_dim is not None
                  else (sys_dim_file if sys_dim_file is not None else 3))

    print(f"Loading WFN: {os.path.basename(wfn_path)}")
    with timing.section("load_wfn"):
        wfn = WFNReader(wfn_path)
        sym = symmetry_maps.SymMaps(wfn)

    nval = int(params.get("nval", 5))
    ncond = int(params.get("ncond", 5))
    nband = int(params.get("nband", 100))
    bispinor = bool(params.get("bispinor", False))

    # Band window the GW run will actually ask for: ``load_kin_ion_submatrix``
    # reads [b_id_0, b_id_3) = [0, nelec + ncond).  Sizing the file below
    # that silently truncates the run's window, so it is a hard floor;
    # ``nband`` (the polarizability window) is the natural default.
    nb_window = int(wfn.nelec) + ncond
    nb_req = int(args.nb) if args.nb is not None else max(int(nband), nb_window)
    if nb_req < nb_window:
        raise SystemExit(
            f"Requested {nb_req} bands but the deck's sigma window needs "
            f"nelec+ncond = {int(wfn.nelec)}+{ncond} = {nb_window}."
        )
    nb_eff = max(1, min(int(wfn.nbands), nb_req))
    if nb_eff < nb_window:
        raise SystemExit(
            f"{os.path.basename(wfn_path)} only has {int(wfn.nbands)} bands but "
            f"the deck's sigma window needs {nb_window}."
        )
    meta = Meta.from_system(wfn, sym, nval, ncond, nb_eff, 0, bispinor)
    nx, ny, nz = meta.fft_grid
    # ρ (and hence V_H) lives on the ψ FFT box, which for a BGW WFN is
    # already the ecutrho grid — do NOT let a stale ``grid_rho`` attribute
    # push the density onto a different mesh than ``compute_local_V_k``.
    if getattr(wfn, 'grid_rho', None) is not None and (
        tuple(int(x) for x in wfn.grid_rho) != tuple(int(x) for x in meta.fft_grid)
    ):
        raise SystemExit(
            f"wfn.grid_rho={tuple(wfn.grid_rho)} != FFT box {tuple(meta.fft_grid)}"
        )
    print(f"Bands: {nb_eff} (deck nband={nband}, sigma window needs {nb_window}), "
          f"FFT grid: {meta.fft_grid}, k-points: {sym.nk_tot}")
    print(f"sys_dim: {sys_dim}   bispinor: {bispinor}   "
          f"nspin/nspinor: {int(getattr(wfn, 'nspin', 1))}/{int(wfn.nspinor)}")
    print(f"nval={nval} ncond={ncond} nelec(bands)={int(wfn.nelec)}")
    print(f"Hartree folded in: {args.hartree}")
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
        caller="kin_ion_io",
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

    vnl_setup = None
    if pseudos:
        print("Building unified V_NL setup...")
        vnl_setup = vnl_ops.build_vnl_setup(
            wfn,
            sym,
            meta,
            pseudos,
            nspinor=int(wfn.nspinor),
        )

    # ---- build the mean-field V_H on the same FFT grid (k-independent) ----
    # SAME Coulomb convention as V_loc above (``ctx.truncation_2d``, i.e.
    # the deck's sys_dim) — which is also QE's, since the DFT run that
    # produced E_DFT/vxc.dat used ``assume_isolated='2D'`` for a slab.
    V_H_r = None
    if args.hartree:
        nocc = int(wfn.nelec)
        if nocc > nb_eff:
            raise SystemExit(
                f"--hartree needs the {nocc} occupied bands but only "
                f"{nb_eff} were requested")
        print(f"\nBuilding valence density from {nocc} occupied bands...")
        with timing.section("build_V_H"):
            rho_r = build_valence_density_chunked(wfn, sym, meta, nocc)
            V_H_r = build_hartree_potential(
                rho_r, wfn,
                truncation_2d=ctx.truncation_2d,
                expected_electrons=spin_degeneracy_factor(wfn) * float(nocc),
            )
            V_H_r = jnp.asarray(V_H_r, dtype=jnp.float64)
            del rho_r

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

            H_k = get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, vnl_setup, wfn,
                                V_H_r=V_H_r)
            kin_ion_all[ik] = np.asarray(H_k)
            del wfn_k

    # ---- write output ----
    print(f"\nWriting to {out_path}...")
    desc = ("T + V_loc + V_NL + V_H matrix elements (H_DFT - V_xc)"
            if args.hartree else "T + V_loc + V_NL matrix elements")
    with timing.section("write_h5"):
        with h5py.File(out_path, "w") as f:
            ds = f.create_dataset("kin_ion", data=kin_ion_all, dtype=np.complex128)
            ds.attrs["description"] = desc
            ds.attrs["nk"] = sym.nk_tot
            ds.attrs["nb"] = nb_eff
            ds.attrs["sys_dim"] = sys_dim
            ds.attrs["truncation_2d"] = ctx.truncation_2d
            ds.attrs["pseudopotentials"] = str(list(pseudos.keys()))
            # ---- provenance: everything a consumer must agree with ----
            # ``has_hartree`` is the contract flag: when True the GW
            # driver MUST NOT add its own ``sig_h`` (double counting).
            ds.attrs["has_hartree"] = bool(args.hartree)
            ds.attrs["hartree_truncation_2d"] = bool(
                ctx.truncation_2d) if args.hartree else False
            ds.attrs["input_file"] = os.path.basename(args.input)
            ds.attrs["wfn_file"] = os.path.basename(wfn_path)
            ds.attrs["nval"] = nval
            ds.attrs["ncond"] = ncond
            ds.attrs["nband_input"] = nband
            ds.attrs["nelec_bands"] = int(wfn.nelec)
            ds.attrs["bispinor"] = bool(bispinor)
            ds.attrs["nspinor"] = int(wfn.nspinor)
            ds.attrs["fft_grid"] = np.asarray(meta.fft_grid, dtype=np.int32)

    print(f"Wrote {os.path.basename(out_path)}: shape {kin_ion_all.shape}, "
          f"sys_dim={sys_dim}, has_hartree={bool(args.hartree)}")
    if args.hartree:
        d0 = np.real(np.diagonal(kin_ion_all[0])) * 13.605693122994
        print("  H0 diag (eV), k=0, first 8 bands: "
              + "  ".join(f"{v:.4f}" for v in d0[:8]))
    timing.report(title="--- Timing (seconds) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
