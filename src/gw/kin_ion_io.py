#!/usr/bin/env python3
"""kin+ion computation: T + V_loc + V_NL (+ V_H) for all k → kin_ion.h5.

**By default this writes TWO datasets** and never mixes them:

``kin_ion``   (nk, nb, nb) — ``T + V_loc + V_NL``, **pristine**.
``v_hartree`` (nk, nb, nb) — the exact FFT-grid ⟨mk|V_H|nk⟩, Ry.

Keeping them apart is what makes one file serve every consumer: the same
``kin_ion.h5`` feeds a run that wants the exact V_H (``hartree_source=
stored``) and one that wants the ISDF quadrature (``isdf``), the VH
column of ``sigma_diag.dat`` stops reading 0.000 by construction, and a
QSGW band rotation gets the **full matrix** rather than a diagonal it
cannot transform.  At 12×12 / 80 bands the extra array is ~15 MB.

``--no-hartree`` skips V_H entirely (ionic-only file, ISDF route
mandatory).  ``--fold-hartree`` reproduces the legacy pre-``v_hartree``
format that added V_H *into* ``kin_ion`` and stamped ``has_hartree=True``;
it exists only to regenerate old artifacts bit-for-bit.

Which V_H source to use — read this before trusting an eqp number
-----------------------------------------------------------------
``H₀ = ⟨T+V_ion+V_NL⟩ + ⟨V_H⟩`` is a catastrophic cancellation: for MoS₂
the two terms are −502 eV and +461 eV and their sum is −42 eV, so V_H's
*relative* accuracy is H₀'s accuracy in absolute eV.

* **stored / gspace (exact FFT-grid).**  ⟨mk|V_H|nk⟩ through the same
  ``compute_local_V_k`` route V_loc takes: analytically exact and
  centroid-count independent.  Pinned to QE's ``kih.dat`` at rms
  1e-4 eV.  Built here by a serial, single-process, k-sequential
  density accumulation — that, not accuracy, is its limitation.
* **isdf (V_q[0] tile).**  Costs nothing extra, inherits the GW run's
  full P-way distribution, recomputable in-loop for QSGW.  Measured vs
  the exact route (MoS₂, scorecard §S.5): ~1 % on the occupied manifold
  at 6 centroids per ζ-fit band, 0.11–0.20 % from 9 c/band upward —
  where it **plateaus** (2.5× more centroids, and a 1e-12 rank cutoff
  instead of 1e-8, each buy <5 % relative).

The plateau is why the exact sources exist.  A 0.5 % V_H error is 2 eV
of H₀, and the VBM and CBM errors do not cancel: on the MoS₂ 12×12 at
606 centroids the ISDF route is 0.54 % rms over the occupied manifold
and that is +1.91 eV on the VBM and −2.25 eV on the CBM at K — **4.15 eV
on the band gap**.  50 meV QP energies need V_H to ~1e-4 relative, two
orders past the plateau.

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
from file_io.kin_ion import HARTREE_DATASET
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


def compute_hartree_matrix(wfn, sym, meta, *, truncation_2d: bool,
                           nb: int, print_fn=print):
    """The exact FFT-grid ⟨mk|V_H|nk⟩ for all k — **(nk, nb, nb) Ry**.

    SINGLE SOURCE for the two exact-V_H consumers: this CLI (which
    stores the result as ``kin_ion.h5``'s ``v_hartree`` dataset) and the
    driver's ``hartree_source=gspace`` route, which rebuilds it in
    memory.  Both must produce the same numbers or the ``stored`` and
    ``gspace`` sources would disagree, so there is exactly one
    implementation.

    Needs no pseudopotentials: ρ comes from ψ, V_H from the Poisson
    solve, and the matrix element from the same ``compute_local_V_k``
    route V_loc takes.  ``truncation_2d`` MUST be the run's own
    convention (deck ``sys_dim``); mixing it with V_loc's is a large
    systematic error inside a ~500 eV cancellation.

    Returns the FULL matrix, not the diagonal: a QSGW band rotation
    needs ⟨m|V_H|n⟩ off-diagonals to transform H₀ into the QP basis.
    """
    nocc = int(wfn.nelec)
    if nocc > nb:
        raise ValueError(
            f"V_H needs the {nocc} occupied bands but only {nb} were requested")
    print_fn(f"\nBuilding valence density from {nocc} occupied bands...")
    rho_r = build_valence_density_chunked(wfn, sym, meta, nocc,
                                          print_fn=print_fn)
    V_H_r = build_hartree_potential(
        rho_r, wfn,
        truncation_2d=bool(truncation_2d),
        expected_electrons=spin_degeneracy_factor(wfn) * float(nocc),
        print_fn=print_fn,
    )
    V_H_r = jnp.asarray(V_H_r, dtype=jnp.float64)
    del rho_r

    v_h_all = np.zeros((sym.nk_tot, nb, nb), dtype=np.complex128)
    for ik in range(sym.nk_tot):
        if ik % 32 == 0:
            print_fn(f"    V_H matrix: k={ik + 1}/{sym.nk_tot}...")
        wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        v_h_all[ik] = np.asarray(compute_local_V_k(
            wfn_k, Gk_crys, V_H_r, wfn.cell_volume, g_mask=None))
        del wfn_k
    return v_h_all


def build_argparser() -> argparse.ArgumentParser:
    """The CLI's argument parser.

    Split out of :func:`main` so the V_H defaults are pinnable by a unit
    test without running the generator.
    """
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
                      help="DEFAULT: also compute the exact FFT-grid ⟨mk|V_H|nk⟩ and "
                           "store it as the SEPARATE 'v_hartree' array (kin_ion "
                           "itself stays pristine T+V_loc+V_NL)")
    argp.add_argument("--no-hartree", dest="hartree", action="store_false",
                      help="skip V_H entirely: the file carries T+V_loc+V_NL only and "
                           "the GW run must supply V_H from the ISDF V_q[0] tile")
    argp.add_argument("--fold-hartree", dest="fold_hartree", action="store_true",
                      default=False,
                      help="LEGACY/compat: add V_H INTO the kin_ion values and stamp "
                           "has_hartree=True (the pre-'v_hartree' format).  Only for "
                           "reproducing old artifacts — the stored-array default is "
                           "strictly better (kin_ion stays reusable, QSGW gets the "
                           "full matrix, and the VH column stops reading 0.000).")
    return argp


def main(argv=None):
    args = build_argparser().parse_args(argv)

    timing.reset()
    print("== kin_ion_io ==")

    # ---- this generator is SINGLE-PROCESS; say so instead of racing ----
    # Every array below is built on device 0 through a 1x1 mesh
    # (``load_kpoint_fftbox`` passes ``sharding=None``), the k loop is a
    # host-side Python loop, and the output accumulator is a plain numpy
    # array written by whoever gets there.  Launched under ``srun -n P``
    # that is P processes doing identical work and then overwriting one
    # another's ``kin_ion.h5`` — a silently truncated or interleaved file
    # with rc=0.  Refuse loudly rather than produce it.  (Distributing it
    # is a real and worthwhile change — see the strong-scaling audit in
    # the scorecard — but it is not what this guard is for.)
    _nproc = int(os.environ.get(
        "JAX_PROCESS_COUNT",
        os.environ.get("JAX_NUM_PROCESSES",
                       os.environ.get("SLURM_NTASKS", "1"))))
    if _nproc > 1 and not os.environ.get("LORRAX_KIN_ION_ALLOW_MULTIPROC"):
        raise SystemExit(
            f"gw.kin_ion_io is a single-process generator but the launcher "
            f"advertises {_nproc} processes (SLURM_NTASKS / "
            f"JAX_PROCESS_COUNT).  Every rank would redo the whole "
            f"calculation and overwrite the same output file.  Run it with "
            f"one task (`srun -n 1` / `#SBATCH -n 1`).")

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
    v_h_all = None
    if args.hartree:
        with timing.section("build_V_H"):
            v_h_all = compute_hartree_matrix(
                wfn, sym, meta, truncation_2d=ctx.truncation_2d, nb=nb_eff)

    # ---- compute kin+ion per k-point ----
    # ``kin_ion`` stays PRISTINE (T + V_loc + V_NL) unless --fold-hartree
    # is given: V_H rides along as its own dataset so the same file can
    # feed a run that wants the exact V_H and one that wants the ISDF
    # quadrature, and so a QSGW rotation has the full ⟨m|V_H|n⟩ matrix.
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

            H_k = get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, vnl_setup, wfn)
            kin_ion_all[ik] = np.asarray(H_k)
            del wfn_k

    folded = bool(args.fold_hartree and v_h_all is not None)
    if folded:
        kin_ion_all = kin_ion_all + v_h_all

    # ---- write output ----
    print(f"\nWriting to {out_path}...")
    desc = ("T + V_loc + V_NL + V_H matrix elements (H_DFT - V_xc)"
            if folded else "T + V_loc + V_NL matrix elements")
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
            # ``has_hartree`` means the LEGACY fold-in: V_H is inside the
            # kin_ion VALUES and no consumer may add another.  It stays
            # False for the stored-array default, which is what makes the
            # new format safe for old readers (they see pristine kin_ion
            # and correctly supply their own ISDF V_H).
            ds.attrs["has_hartree"] = folded
            ds.attrs["hartree_truncation_2d"] = bool(
                ctx.truncation_2d) if folded else False
            ds.attrs["input_file"] = os.path.basename(args.input)
            ds.attrs["wfn_file"] = os.path.basename(wfn_path)
            ds.attrs["nval"] = nval
            ds.attrs["ncond"] = ncond
            ds.attrs["nband_input"] = nband
            ds.attrs["nelec_bands"] = int(wfn.nelec)
            ds.attrs["bispinor"] = bool(bispinor)
            ds.attrs["nspinor"] = int(wfn.nspinor)
            ds.attrs["fft_grid"] = np.asarray(meta.fft_grid, dtype=np.int32)
            if v_h_all is not None and not folded:
                vh = f.create_dataset(HARTREE_DATASET, data=v_h_all,
                                      dtype=np.complex128)
                vh.attrs["description"] = (
                    "<mk|V_H|nk> (Ry), exact FFT-grid mean-field Hartree; "
                    "NOT included in the 'kin_ion' dataset")
                vh.attrs["truncation_2d"] = bool(ctx.truncation_2d)
                vh.attrs["nocc"] = int(wfn.nelec)
                vh.attrs["fft_grid"] = np.asarray(meta.fft_grid, dtype=np.int32)

    _src = ("FOLDED into kin_ion (legacy)" if folded else
            (f"stored as '{HARTREE_DATASET}'" if v_h_all is not None
             else "absent (ISDF route required)"))
    print(f"Wrote {os.path.basename(out_path)}: kin_ion {kin_ion_all.shape}, "
          f"sys_dim={sys_dim}, V_H {_src}")
    if v_h_all is not None:
        d0 = np.real(np.diagonal(kin_ion_all[0])) * 13.605693122994
        v0 = np.real(np.diagonal(v_h_all[0])) * 13.605693122994
        print("  kin_ion diag (eV), k=0, first 8: "
              + "  ".join(f"{v:.4f}" for v in d0[:8]))
        print("  V_H     diag (eV), k=0, first 8: "
              + "  ".join(f"{v:.4f}" for v in v0[:8]))
    timing.report(title="--- Timing (seconds) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
