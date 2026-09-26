#!/usr/bin/env python3
"""Kinetic and ionic Hamiltonian preprocessing: ``kin_ion.h5``.

    ⟨mk|T + V_loc + V_NL|nk⟩  on the star wedge, one band-sharded k-scan
                              (``common.mtxel_sweep``), written through SlabIO
                              (``file_io.kin_ion.write_kin_ion``)

The file holds the pristine mean-field operator only.  The Hartree field is
``gw.hartree.direct_field_matrices``, built live by the GW driver.

Usage:
  python -m gw.kin_ion_io -i lorrax.in -o kin_ion.h5 [-n NB]
"""

import argparse
import os


def build_argparser() -> argparse.ArgumentParser:
    """The CLI's argument parser.

    Split out of :func:`main` so the CLI contract is pinnable by a unit
    test without running the generator, and kept ABOVE the startup call
    so ``--help`` can reach it without one (:mod:`runtime.cli_seam`).
    """
    argp = argparse.ArgumentParser(description="Chunked kin+ion computation")
    argp.add_argument("-i", "--input", required=True, help="cohsex / GW input file")
    argp.add_argument("-o", "--output", default=None, help="output HDF5 (default: kin_ion.h5)")
    argp.add_argument("--report-file", default=None,
                      help="human-readable report (default: kin_ion.out beside output)")
    argp.add_argument("-n", "--nb", type=int, default=None, help="number of bands")
    argp.add_argument("--sys_dim", type=int, default=None,
                      help="system dimensionality: 0, 2, or 3.  Must AGREE with "
                           "the input file when the file specifies it.")
    argp.add_argument("--pseudo_dir", default=None,
                      help="directory containing *.upf files (default: input file dir)")
    # SOC projector selection is automatic: QE metadata when available,
    # scalar for nspinor=1, otherwise a wavefunction measurement.
    return argp


if __name__ == "__main__":
    # Argv is answered before any runtime exists — runtime/cli_seam.py.
    from runtime.cli_seam import refuse_bad_argv
    refuse_bad_argv(build_argparser())


# ---- join the distributed world BEFORE anything touches XLA ------------
# THE startup call (runtime module docstring): env defaults, fail-fast
# hook, jax.distributed, CPU fallback, the run's clique-warmed ('x','y')
# mesh, compile cache, rank-0 report.  ``jax.distributed.initialize()``
# refuses to run once the XLA backend is up, and the import graph below
# (``psp.*``) reaches jax; ``runtime`` itself imports no jax, and the call
# is idempotent — which is what makes this safe under
# ``gw.sigma_dispatch``'s LAZY import of this module from inside an
# already-started driver: there it returns the existing stack.
from runtime import debug_print, initialize_communicator_stack  # noqa: E402
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import numpy as np
import jax.numpy as jnp
import h5py

from common import Meta
from common.gvec_fft_box import refuse_padded_gvecs_without_mask
from common.collectives import process_rank_world
import common.timing as timing
from common.preprocessing_output import PreprocessingProductionReport
from common.progress import LoopProgress
from common.scientific_output import (
    band_range, pseudopotential_file_rows,
)
from wfn_loader import WfnLoader                                    # noqa: E402
from file_io.kin_ion import write_kin_ion
from gw.gw_config import (
    BispinorGWMode,
    coerce_bispinor_gw_mode,
    read_lorrax_input as read_cohsex_input,
)
from psp.pseudos import load_pseudopotentials, build_atom_pp_assignments
from psp.dft_operators import padded_gvectors, vnl_matrix_from_kdata
from psp.radial.build_projectors_qe import build_local_ionic_potential_on_G_total
from psp.get_DFT_mtxels import compute_kinetic_k, compute_local_V_k
from psp.operator_checks import validate_operator_inputs
import psp.vnl_ops as vnl_ops
from runtime.run_session import RunSession                      # noqa: E402


def _resolve_against(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


# THE k-SET IS THE STAR WEDGE: one WFN row per symmetry orbit, and the
# full-BZ table is its star broadcast (a gather, conjugated on the
# time-reversed rows).  The derivation, the wedge choice and the traps in
# validating it: docs/architecture/symmetry_register.md §8.  The helpers are
# the symmetry service's and file_io.kin_ion's; the names below re-export
# them for gw.dynamic_sigma until the wave after ARCH wave 0 repoints it.
from symmetry_maps import star_wedge_rows                          # noqa: E402
from symmetry_maps import star_wedge_tables as star_tables         # noqa: E402
from file_io.kin_ion import (                                       # noqa: E402
    broadcast_star_wedge as broadcast_ibz_to_full_bz,
)
# The direct (Hartree) field is gw.hartree's; these names stay importable
# from here for gw.sigma_dispatch, psp.get_DFT_mtxels and the multi-device
# gates until the wave after ARCH wave 0.
from gw.hartree import (                                            # noqa: E402
    ExactHartreeMatrices,
    direct_field_matrices as compute_hartree_matrix,
    star_wedge_kspec as _wedge_sweep_kspec,
    wedge_density_occupations as _wedge_density_occupations,
)


# ---- artifact provenance ---------------------------------------------------
# A kin_ion.h5 with no stamp of WHAT it was made from is how a stale
# committed fixture survived a month of green tests.  WFN identity is owned
# by ``common.parallel_transport.wfn_fingerprint``; this module owns only its
# generator-commit stamp.

def _generator_commit() -> str:
    """Commit of the SOURCE TREE THIS MODULE RAN FROM — not the cwd's.

    The CLI is normally invoked from a scratch work directory that is not
    a checkout (or is a *different* one), so ``git rev-parse`` in the cwd
    names the wrong tree or nothing at all; ``git -C <src>`` anchors it to
    the file that is actually executing.  Falls back to
    ``'unknown:<reason>'`` rather than an empty string or a fake hash — a
    named reason is information, a blank attr is the failure mode this
    stamp exists to remove.

    PRICED, because a provenance stamp that costs real wall is a stamp
    someone will delete.  MEASURED from inside the shifter container with
    the device stack up (the tree on Lustre), against 0.03 s for the same
    commands from a login shell:

        bare fork of this process        0.181 s   (page tables of a live
                                                    JAX runtime)
        rev-parse --short HEAD           0.519 s
        status --porcelain -uno          0.625 s
        describe --always --dirty        3.583 s   <- rejected

    so this pair is ~1.1 s, once, on rank 0 — three orders below the
    generator run it stamps at any production shape, and it is charged to
    the ``write_h5`` timing section rather than hidden in ``(untimed)``.
    ``describe --dirty`` would have done it in one fork and costs 3× more:
    it walks the tag graph on top of the same worktree refresh.

    The ``-dirty`` suffix is not decoration.  A stamp that reads clean on
    an edited tree is worse than no stamp, because it is the exact claim
    the reader wanted to check.
    """
    import subprocess
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        rev = subprocess.run(
            ["git", "-C", src, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=60)
        if rev.returncode != 0:
            return "unknown:not-a-git-checkout"
        out = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", src, "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=60)
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except Exception as exc:                                  # noqa: BLE001
        return f"unknown:{type(exc).__name__}"


def get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, vnl_setup, wfn, g_mask=None,
                  V_H_r=None):
    """Compute T + V_loc + V_NL (+ V_H) for a single k-point.

    Parameters
    ----------
    wfn_k : (nb, nspinor, nx, ny, nz) — wavefunctions in FFT box
    Gk_crys : (nG, 3) int — G-vector indices for this k.  May be the
        k's own ``ngk`` rows or the ``ngkmax``-padded table, in which
        case ``g_mask`` is REQUIRED (pad rows are the FFT-box pad
        sentinel — a valid box index, so an absent mask double-counts
        that component rather than crashing).  Enforced by
        :func:`common.gvec_fft_box.refuse_padded_gvecs_without_mask`,
        which lives beside the routine that BUILDS the pad so the
        detector and the producer share one invariant.
    kvec : (3,) float — k-point in crystal coords
    V_loc_r : (nx, ny, nz) — local ionic potential on FFT grid
    vnl_setup : VNLSetup from vnl_ops.build_vnl_setup (or None to skip V_NL)
    wfn : WFNReader (for bdot, bvec, blat, cell_volume)
    g_mask : (nG,) float or None — 1 on physical G, 0 on pad columns.
    V_H_r : (nx, ny, nz) or None — mean-field Hartree potential on the
        SAME FFT grid as ``V_loc_r``.  Folded in through the identical
        local-potential route, so H₀'s ~500 eV cancellation closes inside
        one exact routine instead of across two numerical schemes.
    """
    Gk_np = np.asarray(Gk_crys, dtype=int)
    if g_mask is None:
        refuse_padded_gvecs_without_mask(
            Gk_np, getattr(wfn, "fft_grid", None),
            where="get_kin_ion_k: Gk_crys")
    bdot_np = np.asarray(wfn.bdot, dtype=float)
    T_k = compute_kinetic_k(wfn_k, Gk_crys, kvec, bdot_np, g_mask=g_mask)
    V_loc_k = compute_local_V_k(
        wfn_k, Gk_crys, V_loc_r, wfn.cell_volume, g_mask=g_mask
    )

    V_NL_k = 0.0
    if vnl_setup is not None:
        # Z is built ON THE SAME (padded) G-list, which is the contract
        # ``_build_vnl_kdata_core`` documents: Z at a pad row is finite
        # (it is evaluated at K = kvec) and the caller must mask before
        # contracting.  Masking ψ_G is sufficient and is what
        # ``vnl_matrix_from_kdata(mask=…)`` does — every contraction in
        # ``vnl_ops.vnl_matrix`` runs through ψ_G at least once.
        kdata = vnl_ops.build_vnl_kdata_from_kvec(
            np.asarray(kvec, dtype=float), Gk_np, vnl_setup)
        V_NL_k = vnl_matrix_from_kdata(wfn_k, Gk_crys, kdata, mask=g_mask)

    H_k = T_k + V_loc_k + V_NL_k
    if V_H_r is not None:
        H_k = H_k + compute_local_V_k(
            wfn_k, Gk_crys, V_H_r, wfn.cell_volume, g_mask=g_mask
        )
    return H_k


def _kin_ion_provenance(*, args, wfn, wfn_path, sym, meta, nb, nk_irr,
                        sys_dim, ctx, pseudos, nval, ncond, nband, bispinor,
                        vnl_setup, rank) -> dict:
    """The ``kin_ion`` dataset's attributes: what it is and what made it.

    ``nk`` is the LOGICAL full-BZ count every consumer means by nk; ``nrk``
    is the stored star-wedge row count.  ``soc`` and ``soc_provenance``
    record which V_NL projectors were built (``nspinor`` alone does not
    say).  ``wfn_fingerprint``, ``ngkmax`` and ``generator_commit`` say what
    the file was made from: ``wfn_file`` is a basename and identifies
    nothing.  ``generator_commit`` forks git (~1 s), so only rank 0, whose
    copy is the one SlabIO lands, computes it.
    """
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, wfn_fingerprint)
    return {
        "description": "T + V_loc + V_NL matrix elements",
        "nk": sym.nk_tot,
        "nb": nb,
        "sys_dim": sys_dim,
        "truncation_2d": ctx.truncation_2d,
        "pseudopotentials": str(list(pseudos.keys())),
        "input_file": os.path.basename(args.input),
        "wfn_file": os.path.basename(wfn_path),
        "nval": nval,
        "ncond": ncond,
        "nband_input": nband,
        "nelec_bands": int(wfn.nelec),
        "bispinor": bool(bispinor),
        "nspinor": int(wfn.nspinor),
        "soc": bool(vnl_setup.soc) if vnl_setup is not None else False,
        "soc_provenance": (vnl_setup.soc_provenance if vnl_setup is not None
                           else "no projector setup"),
        "fft_grid": np.asarray(meta.fft_grid, dtype=np.int32),
        "ngkmax": int(wfn.ngkmax),
        "wfn_fingerprint": wfn_fingerprint(wfn),
        "wfn_fingerprint_scheme": WFN_FINGERPRINT_SCHEME,
        "generator_commit": _generator_commit() if rank == 0 else "",
        "nrk": int(nk_irr),
        "k_set_computed": "ibz",
    }


#: The report's major-stage table: ``(label, timing sections...)``.
_STAGES = (
    ("wavefunction input", "load_wfn"),
    ("psi(G) sphere read", "load_psi_sphere"),
    ("local ionic potential", "build_V_loc"),
    ("nonlocal projectors", "build_V_NL"),
    ("T + ionic matrix", "kin_ion"),
    ("artifact write", "write_h5"),
)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    rank, world = process_rank_world()
    input_dir = os.path.dirname(os.path.abspath(args.input))
    out_path = args.output or os.path.join(input_dir, "kin_ion.h5")
    report_path = (os.path.abspath(args.report_file) if args.report_file else
                   os.path.join(os.path.dirname(os.path.abspath(out_path)),
                                "kin_ion.out"))
    with RunSession(RUNTIME, "kin_ion", PreprocessingProductionReport,
                    report_path, stages=_STAGES,
                    driver_name="gw.kin_ion_io",
                    calculation_name=(
                        "kinetic, ionic, and Hartree preprocessing")) as run:
        report = run.report
        print0 = report.legacy_print
        report.begin(input_file=args.input)
        report.architecture(mesh_role="band-matrix axes X x Y")

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

        print0(f"Loading WFN: {os.path.basename(wfn_path)}")
        # The module-top ``initialize_communicator_stack()`` already built and
        # clique-warmed the run's mesh; handing it to the loader is what lets
        # ``backend=auto`` pick the collective phdf5 read at P>1 instead of
        # the per-rank eager h5py read (scorecard BD.2 — htransform already
        # did this, dipole/kin-ion/kmeans did not).
        mesh_xy = RUNTIME.mesh
        with timing.section("load_wfn"):
            wfn = WfnLoader(wfn_path, mesh=mesh_xy)
            sym = wfn.symmetry()

        nval = int(params.get("nval", 5))
        ncond = int(params.get("ncond", 5))
        nband = int(params.get("nband", 100))
        bispinor = bool(params.get("bispinor", False))
        bispinor_gw_mode = coerce_bispinor_gw_mode(
            params.get("bispinor_gw", "bare_transverse"))
        if (bispinor_gw_mode is BispinorGWMode.FULL_STATIC_COHSEX
                and not bispinor):
            raise SystemExit(
                f"bispinor_gw={bispinor_gw_mode.value} requires "
                "bispinor=true; the selector does not enable spatial-current "
                "channels implicitly.")
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
        report.environment(wfn=wfn, lines=(
            "Matrix storage : distributed band blocks on the X x Y mesh",
            "Output writer  : SlabIO collective write from the band shards",
        ))
        report.sampling(wfn=wfn, sym=sym)
        print0(f"Bands: {nb_eff} (deck nband={nband}, sigma window needs {nb_window}), "
              f"FFT grid: {meta.fft_grid}, k-points: {sym.nk_tot}")
        print0(f"sys_dim: {sys_dim}   bispinor: {bispinor}   "
              f"bispinor_gw: {bispinor_gw_mode.value}   "
              f"nspin/nspinor: {int(getattr(wfn, 'nspin', 1))}/{int(wfn.nspinor)}")
        print0(f"nval={nval} ncond={ncond} nelec(bands)={int(wfn.nelec)}")

        # ---- load pseudopotentials ----
        pseudo_dir = args.pseudo_dir or input_dir
        pseudo_source = os.path.abspath(pseudo_dir)
        pseudos = load_pseudopotentials(pseudo_dir)
        if not pseudos:
            # Also try the QE subdirectory (common sandbox layout)
            for fallback in [os.path.join(input_dir, '..', 'qe', 'scf'),
                             os.path.join(input_dir, '..', 'qe', 'nscf')]:
                pseudos = load_pseudopotentials(fallback)
                if pseudos:
                    pseudo_source = os.path.abspath(fallback)
                    print0(f"Found pseudopotentials in {fallback}")
                    break

        # ---- validate (will raise if pseudos missing or sys_dim invalid) ----
        ctx = validate_operator_inputs(
            pseudos=pseudos, wfn=wfn, sys_dim=sys_dim,
            caller="kin_ion_io",
        )
        from psp.pseudos import pseudo_summary_lines
        report.pseudopotentials(pseudo_summary_lines(ctx.pseudos))
        print0(f"Coulomb truncation: {'2D slab' if ctx.truncation_2d else '3D bulk'}")

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
        print0("Building V_loc...")
        vloc_progress = LoopProgress(
            1, report.progress, title="local ionic potential construction",
            item_name="FFT-grid potential")
        vloc_progress.start()
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
        vloc_progress.step()
        vloc_progress.finish()

        vnl_setup = None
        if pseudos:
            print0("Building unified V_NL setup...")
            # Which PROJECTORS get built — j-resolved (spin-orbit) or j-averaged
            # (scalar-relativistic) — is resolved automatically inside
            # ``build_vnl_setup``: QE's <spinorbit> when the structure came from
            # a .save, nspinor=1 by force, and otherwise MEASURED against the
            # wavefunctions (see psp.vnl_ops.measure_soc_mode).  The choice is
            # upstream of the projector contraction and does not touch it.
            vnl_progress = LoopProgress(
                1, report.progress, title="nonlocal projector construction",
                item_name="projector setup")
            vnl_progress.start()
            with timing.section("build_V_NL"):
                vnl_setup = vnl_ops.build_vnl_setup(
                    wfn,
                    sym,
                    meta,
                    pseudos,
                    nspinor=int(wfn.nspinor),
                    print_fn=print0,
                )
            vnl_progress.step()
            vnl_progress.finish()

        k_spec, _, nk_irr = _wedge_sweep_kspec(wfn, sym)
        resolved_soc = (vnl_setup.soc_provenance if vnl_setup is not None
                        else "none (no projector setup)")
        report.pathways((
            "Mean-field H0  : pristine T + V_loc + V_NL",
            "Hartree V_H    : live G-space build in gw_jax",
            "Coulomb system : " + ("2D slab truncation" if ctx.truncation_2d
                                    else "3D bulk periodic"),
            f"SOC projectors : {resolved_soc}",
            f"k-space compute/storage: {nk_irr} star-wedge points; "
            f"{int(sym.nk_tot)} full-BZ points reconstructed on read",
        ))
        report.system(
            natoms=int(np.asarray(wfn.atom_crys).shape[0]),
            species=sorted(str(name) for name in ctx.pseudos),
            fft_grid=meta.fft_grid,
            lines=(
                f"Spin channels  : nspin={int(getattr(wfn, 'nspin', 1))}; "
                f"nspinor={int(wfn.nspinor)}; bispinor={bool(bispinor)}",
                f"System dimension: {int(sys_dim)}",
            ))
        report.bands((
            f"Electrons      : {float(getattr(wfn, 'num_electrons', wfn.nelec)):.5f}; "
            f"occupied-band boundary = {int(wfn.nelec)}",
            f"Matrix written : {band_range(0, nb_eff)}",
            f"Protected valence: {band_range(max(0, int(wfn.nelec) - nval), int(wfn.nelec))}",
            f"Protected conduction: {band_range(int(wfn.nelec), nb_window)}",
            f"Polarizability : {band_range(0, min(nb_eff, nband))}",
            f"WFN available  : {band_range(0, int(wfn.nbands))}",
        ))

        # ---- compute kin+ion: ONE k-scan, bands sharded over every rank -----
        # ``kin_ion`` stays pristine T + V_loc + V_NL.  The k-partitioned
        # route boxed a whole k's bands on one rank and stopped scaling at
        # P = nk.  Here the three terms are summed ON THE KET
        # (``sum_operators``) so ⟨m|T+V_loc+V_NL|n⟩ is ONE sweep with one
        # all-to-all and one slab GEMM, not three of each.
        #
        #   T       |k+G|² ψ            diagonal in G: applied on the G slab
        #   V_loc   F[V(r) F⁻¹ψ]        the only real-space excursion (band layout)
        #   V_NL    Z E Z† ψ            separable: c† E c on slab projections
        #
        # ``get_kin_ion_k`` is left in place — it is the per-k local-plan
        # kernel the sweep is gated against.
        from common.mtxel_sweep import (SweepGeometry, kinetic_operator,
                                        local_potential_operator, sum_operators,
                                        sweep_matrix_elements, vnl_operator)
        from common.wfn_layout import band_sphere_spec
        #
        # THE k-SET IS THE STAR WEDGE, and so is the WRITTEN table
        # (docs/architecture/symmetry_register.md §8: the conjugation the
        # time-reversed rows need, and why the WFN's own k-set is NOT the
        # wedge on every deck).  T, V_loc and
        # V_NL are built from the lattice and the atomic positions, so they are
        # exactly symmetric by construction and this is the sweep the argument
        # fits most cleanly.  No CONSUMER of ``kin_ion.h5`` sees the k-set
        # either: ``file_io.kin_ion`` unfolds on read and still hands back
        # ``(nk_tot, nb, nb)`` in full-BZ order.
        gtab = padded_gvectors(wfn, k=k_spec)
        # The band-sharded sphere read is this driver's largest single cost at
        # production size (VI3 12x12, 360 bands: ~53 s of a 75 s run at P16,
        # runs/runtime/mtxel_sweep_20260923 b01), so it is its own timed stage;
        # the sync keeps the device transfer inside it rather than in kin_ion.
        with timing.section("load_psi_sphere"):
            psi_G = wfn.load(bands=(0, nb_eff), k=k_spec,
                             sharding=band_sphere_spec())
            psi_G.block_until_ready()
        geom = SweepGeometry(mesh=mesh_xy, fft_grid=meta.fft_grid,
                             ngkmax=int(psi_G.shape[3]), nb=nb_eff,
                             ns=int(psi_G.shape[2]), nk=nk_irr,
                             cell_volume=float(wfn.cell_volume))
        terms = [kinetic_operator(geom, np.asarray(wfn.bdot, dtype=float)),
                 local_potential_operator(geom, V_loc_r)]
        if vnl_setup is not None:
            terms.append(vnl_operator(geom, vnl_setup))
        print0(f"\n⟨mk|T+V_loc+V_NL|nk⟩: one k-scan over {nk_irr} STAR-WEDGE "
               f"k-points (broadcast to {sym.nk_tot} full-BZ k), "
               f"{geom.nb} bands sharded over P={world}...")
        # ONE ``kin_ion`` timing section around the WHOLE sweep, count 1 — not
        # one per k.  A per-k section would time the dispatch and attribute the
        # compute to whoever happened to block next.
        matrix_progress = LoopProgress(
            1, report.progress, title="kinetic and ionic matrix construction",
            item_name="distributed band-matrix sweep")
        matrix_progress.start()
        with timing.section("kin_ion") as sweep:
            H_kin_ion = sweep_matrix_elements(
                psi_G, operator=sum_operators(*terms), geom=geom,
                gvecs=gtab.gvecs, gmask=gtab.mask,
                box_index=wfn.box_index(k=k_spec),
                # The WFN loader's paired k representative for these exact G rows,
                # for the same reason as the V_H sweep above.
                kvecs=gtab.kvecs)
            sweep.watch(H_kin_ion)
        matrix_progress.step()
        matrix_progress.finish()
        del psi_G

        # ---- write: SlabIO from the sweep's shards --------------------------
        # The star-wedge slab, not its full-BZ broadcast: the reader unfolds
        # (file_io.kin_ion).  No rank gathers the (n_orbits, nb, nb) table.
        print0(f"\nWriting to {out_path}...")
        write_progress = LoopProgress(
            1, report.progress, title="kinetic and ionic artifact write",
            item_name="output artifact")
        write_progress.start()
        with timing.section("write_h5"):
            write_kin_ion(
                out_path, H_kin_ion, mesh=mesh_xy, nb=nb_eff,
                star=star_tables(sym),
                attrs=_kin_ion_provenance(
                    args=args, wfn=wfn, wfn_path=wfn_path, sym=sym, meta=meta,
                    nb=nb_eff, nk_irr=nk_irr, sys_dim=sys_dim, ctx=ctx,
                    pseudos=pseudos, nval=nval, ncond=ncond, nband=nband,
                    bispinor=bispinor, vnl_setup=vnl_setup, rank=rank))
        del H_kin_ion
        write_progress.step()
        write_progress.finish()

        # ---- DOES THIS OPERATOR HAVE THE SYMMETRY OF THESE WAVEFUNCTIONS? ----
        # The detector that needs no metadata: on every degenerate manifold
        # of the WFN eigenvalues, T+V_loc+V_NL must be a multiple of the
        # identity (psp.operator_checks).  It reads the WRITTEN wedge rows'
        # manifold blocks, paired with the same WFN rows the sweep read.
        if rank == 0:
            from psp.operator_checks import check_degeneracy_consistency
            en = np.asarray(wfn.energies)
            en = en[0] if en.ndim == 3 else en          # (nk, nb), Ry
            with h5py.File(out_path, "r") as h5:
                check_degeneracy_consistency(
                    h5["kin_ion"], en[star_wedge_rows(sym)[0], :nb_eff],
                    label="kin_ion (T+V_loc+V_NL)", print_fn=print0)

        if rank == 0:
            print0(f"Wrote {os.path.basename(out_path)}: kin_ion "
                   f"{(int(nk_irr), nb_eff, nb_eff)} on the star wedge "
                   f"({sym.nk_tot / max(int(nk_irr), 1):.2f}x compression), "
                   f"sys_dim={sys_dim}; V_H is not stored.")
        run.complete(files=[
            ("mean-field matrices", "written", out_path),
            ("wavefunctions", "read", wfn_path),
            *pseudopotential_file_rows(pseudos, fallback=pseudo_source),
            ("input deck", "read", args.input),
        ])
    return 0


# ---------------------------------------------------------------------------
# Forwarding shims — NOT used by anything in this file.
# ---------------------------------------------------------------------------
# These four names used to be defined here; they are generic k-partition
# plumbing and now live in ``common.collectives``.  Two call sites still
# import them from this module (``gw.sigma_dispatch``,
# ``tests/test_sanity_gates_jax.py``), which are outside this workstream's
# file ownership — see requests R3/R4.  Delete this block once both move.
from common.collectives import (                       # noqa: E402,F401
    gather_indexed_blocks as _gather_indexed_blocks,
    psum_replicate as _psum_replicate,
    replicate_to_mesh,
    sweep_local_k,
)


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
