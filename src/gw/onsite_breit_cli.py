#!/usr/bin/env python3
"""The on-site Breit operator at the DFT level: ``onsite_breit.h5``.

    D[a, b]  = sum_k w_k sum_n f_nk P[k,a,n] conj(P[k,b,n])     DFT occupations, U = 1
    h        = h^dir[D] + h^x[D]                                  (keys of ONSITE_INCODE.md section 3)
    H_OS[k]  = P[k]^dagger h P[k]                                 (Ry, DFT band basis, full BZ)
    dH       = H_OS                    (onsite_breit_double_counting = onsite_only)
             = <m|alpha.A[J]|n> + H_OS (paw: the smooth transverse direct field, pseudo, all space)
    applied once:  E', U = eigh(diag(E_DFT) + Herm(dH)) per k;  D' = D[U, f(E')]

The field is ``gw.hartree.onsite_breit_field`` (the one assembly the GW
driver also calls); the smooth field is ``gw.hartree.direct_field_matrices``
with the carrier of ``gw.sigma_dispatch._compute_live_hartree``.  At the DFT
level Sigma carries no transverse exchange, so ``onsite_breit_exchange = paw``
refuses here by design.

Usage:
  python -m gw.onsite_breit_cli -i cohsex.in [--out onsite_breit.h5] [--no-apply]

The band window is the GW run's QP window [0, b3), b3 = nelec + ncond, with
``nval``/``ncond`` or ``number_bands_protected`` resolved as GW resolves them
(``LorraxConfig.with_band_request``, the helper ``gw.kin_ion_io`` uses).
"""

import argparse
import os


def build_argparser() -> argparse.ArgumentParser:
    """The CLI's argument parser, above the startup call (``runtime.cli_seam``)."""
    argp = argparse.ArgumentParser(description="On-site Breit operator at the DFT level")
    argp.add_argument("-i", "--input", required=True, help="cohsex / GW input file")
    argp.add_argument("--out", default=None,
                      help="output HDF5 (default: onsite_breit.h5 beside the deck)")
    argp.add_argument("--no-apply", action="store_true",
                      help="skip the one-shot application (eigh) and the moments after it")
    argp.add_argument("--report-file", default=None,
                      help="human-readable report (default: onsite_breit.out beside the output)")
    return argp


if __name__ == "__main__":
    # Argv is answered before any runtime exists — runtime/cli_seam.py.
    from runtime.cli_seam import refuse_bad_argv
    refuse_bad_argv(build_argparser())


# ---- join the distributed world BEFORE anything touches XLA ------------
from runtime import debug_print, initialize_communicator_stack  # noqa: E402
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import numpy as np                                                  # noqa: E402

import common.timing as timing                                      # noqa: E402
from common import Meta                                             # noqa: E402
from common.collectives import replicate_to_mesh                    # noqa: E402
from common.log import get_logger                                   # noqa: E402
from common.preprocessing_output import PreprocessingProductionReport  # noqa: E402
from common.units import RYD_TO_EV                                  # noqa: E402
from distrib_la import eigh_local                                   # noqa: E402
from file_io.onsite_breit_output import write_onsite_breit_h5       # noqa: E402
from gw.gw_config import LorraxConfig, read_lorrax_input            # noqa: E402
from gw.hartree import (direct_field_matrices, onsite_breit_field,  # noqa: E402
                        onsite_breit_setup_for)
from psp.onsite_breit import (onsite_band_matrix, onsite_density_matrix,  # noqa: E402
                              shell_moments)
from runtime.run_session import RunSession                          # noqa: E402
from wfn_loader import WfnLoader                                    # noqa: E402

log = get_logger(__name__)

#: The report's stage table: ``(label, timing sections...)``.
_STAGES = (
    ("wavefunction input", "load_wfn"),
    ("on-site setup", "onsite_setup"),
    ("projection, D, h, H_OS", "onsite_field"),
    ("smooth alpha.A[J] field", "vh_rho", "vh_transverse_field", "vh_matrix",
     "vh_transverse_matrix_boundary"),
    ("apply once", "onsite_apply"),
    ("artifact write", "write_h5"),
)
PIECES = ("dir_ae", "dir_ps", "x_ae", "x_ps")


def _resolve_against(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_run(deck: str, mesh):
    """``(config, wfn, sym, meta, nb)``: the deck, the WFN and the window [0, b3).

    ``b3 = nelec + ncond`` with the band request resolved as GW resolves it
    (``LorraxConfig.with_band_request``, as ``gw.kin_ion_io``), so nb equals
    the GW run's QP window top with ``b0 = 0`` (``common.meta.Meta.from_system``).
    """
    from gw.gw_config import _DECK_NAMED_KEYS, require_deck_keys
    input_dir = os.path.dirname(os.path.abspath(deck))
    params = read_lorrax_input(deck)
    require_deck_keys(params[_DECK_NAMED_KEYS], "band_request")
    wfn_path = _resolve_against(params.get("wfn_file", "WFN.h5"), input_dir)
    with timing.section("load_wfn"):
        wfn = WfnLoader(wfn_path, mesh=mesh)
        sym = wfn.symmetry()
    config = LorraxConfig.from_input_file(
        deck, print_fn=lambda *_a, **_k: None, resolve_hardware=False).with_band_request(
            wfn, print_fn=lambda *_a, **_k: None)
    nb = int(wfn.nelec) + int(config.ncond)
    if int(wfn.nbands) < nb:
        raise SystemExit(f"{os.path.basename(wfn_path)} has {int(wfn.nbands)} bands; the deck's "
                         f"QP window [0, {nb}) needs {nb}")
    meta = Meta.from_system(wfn, sym, int(config.nval), int(config.ncond), nb, 0,
                            bool(config.bispinor))
    log.info(f"onsite_breit_cli: {os.path.basename(wfn_path)}, window [0, {nb}) "
             f"(nelec {int(wfn.nelec)} + ncond {int(config.ncond)}), {int(sym.nk_tot)} full-BZ k")
    return config, wfn, sym, meta, nb


def require_setup(setup):
    """The driver computes nothing without the term: ``onsite_breit = false`` refuses."""
    if setup is None:
        raise ValueError(
            "GATE onsite_breit_keys: the deck does not switch the term on\n"
            "  got:  onsite_breit = false\n  want: onsite_breit = true with its keys\n"
            "  why:  gw.onsite_breit_cli computes only the on-site Breit field\n"
            "  fix:  set onsite_breit = true, onsite_breit_files, "
            "onsite_breit_double_counting, onsite_breit_exchange")
    return setup


def piece_band_matrices(field, mesh) -> dict:
    """``{piece: (nk, nb, nb) Ry}``: P^dagger h_piece P for each raw piece
    (``dir_ae, dir_ps, x_ae, x_ps``), so the regression can combine arms."""
    out = {}
    for name in PIECES:
        if all(pc.get(name) is None for pc in field.pieces):
            continue
        block = np.zeros_like(field.h_block)
        for sa, pc in zip(field.setup.shell_atoms, field.pieces):
            if pc.get(name) is not None:
                block[sa["start"]:sa["stop"], sa["start"]:sa["stop"]] = pc[name]
        out[name] = np.asarray(onsite_band_matrix(field.projections, block, mesh=mesh))
    return out


def smooth_direct_field(config, wfn, sym, meta, setup, nb, mesh):
    """``<m|alpha.A[J]|n>`` (nk, nb, nb) Ry for ``double_counting = paw``, else ``None``.

    The carrier arguments are ``gw.sigma_dispatch._compute_live_hartree``'s:
    four-current representation, ``charge_nspinor`` (the leading psi_L
    block on the diagnostic ``large`` charge lift) and ``bispinor_lift``.
    """
    if setup.double_counting != "paw":
        return None
    from dataclasses import replace
    from common.four_current_model import resolve_four_current_representation
    from file_io.qe_save_reader import dft_sys_dim
    representation = resolve_four_current_representation(config.bispinor, config.bispinor_gw)
    hartree_meta = replace(meta, nspinor=4, npol=4) if int(meta.nspinor) != 4 else meta
    exact = direct_field_matrices(
        wfn, sym, hartree_meta,
        truncation_2d=(dft_sys_dim(wfn, fallback=config.sys_dim) == 2),
        transverse_truncation_2d=(int(config.sys_dim) == 2),
        nb=nb, mesh=mesh, include_transverse=True,
        charge_nspinor=(int(wfn.nspinor) if representation.charge_lift == "large" else None),
        bispinor_lift=(representation.current_lift or "raw"),
        return_sharded=False)
    return np.asarray(exact.transverse, dtype=np.complex128)


def _shell_moment_rows(D, setup) -> list:
    """Per shell-atom ``(label, <L>.m, 2<S>.m, m)`` with m the shell's own spin direction."""
    rows = []
    for sa in setup.shell_atoms:
        mo = shell_moments(D[sa["start"]:sa["stop"], sa["start"]:sa["stop"]], sa["l"])
        n = np.linalg.norm(mo["S"])
        m = mo["S"] / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        rows.append((f"atom {sa['atom']} {sa['element']} {sa['label']}", mo, m))
    return rows


def _log_moment_change(D_before, D_after, setup, before_label: str) -> None:
    """Moments before and after, along each shell's own spin direction (before), and their change."""
    for (label, mb, m), (_, ma, _) in zip(_shell_moment_rows(D_before, setup),
                                          _shell_moment_rows(D_after, setup)):
        Lb, La = float(mb["L"] @ m), float(ma["L"] @ m)
        Sb, Sa = float(2 * mb["S"] @ m), float(2 * ma["S"] @ m)
        pct = lambda a, b: (100.0 * (a - b) / b) if abs(b) > 1e-12 else float("nan")  # noqa: E731
        log.info(f"onsite_breit applied once, {label} (m = ({m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}), "
                 f"before = {before_label}): <L>.m {Lb:+.6f} -> {La:+.6f} (d = {La - Lb:+.3e}, "
                 f"{pct(La, Lb):+.4f} %)  2<S>.m {Sb:+.6f} -> {Sa:+.6f} (d = {Sa - Sb:+.3e}, "
                 f"{pct(Sa, Sb):+.4f} %)  Tr D {mb['N']:.6f} -> {ma['N']:.6f}")


def _fixed_n_occupations(config, wfn, E_kn, kweights, *, label: str):
    """f of the states with energies E (nk, nb) Ry: the lowest nelec bands of an
    insulator; a metal's fixed-N state from ``gw.efermi`` with the deck's smearing;
    ``None`` (said in the log) when the deck declares no smearing family."""
    nk, nb = (int(s) for s in np.shape(E_kn))
    if bool(wfn.occupations_are_exact_integer):
        f = np.zeros((nk, nb), dtype=np.float64)
        f[:, :int(wfn.nelec)] = 1.0
        return f
    family = getattr(config, "occ_smearing_family", None)
    if family is None:
        log.info(f"onsite_breit applied once: metallic WFN and no occ_smearing_width_ry in the "
                 f"deck; the {label} occupations are NOT computed and the moments after the "
                 "application are skipped")
        return None
    from gw.efermi import OccupationState
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    state = OccupationState.solve_smearing(
        E_kn, kweights, float(wfn.num_electrons), float(config.occ_broadening_ry),
        state_capacity=spin_degeneracy_factor(wfn), family=str(family).strip().lower(),
        clamp_tol=float(config.occupation_clamp_tol))
    log.info(f"onsite_breit applied once: {label} fixed-N {family} occupations, "
             f"mu = {float(state.mu_ry) * RYD_TO_EV:+.6f} eV, kT = "
             f"{float(config.occ_broadening_ry) * RYD_TO_EV * 1e3:.3f} meV")
    return np.asarray(state.f_kn, dtype=np.float64)


def apply_once(config, wfn, sym, field, dH_total_ry, mesh):
    """One application of dH: per full-BZ k, ``E', U = eigh(diag(E_DFT) + Herm(dH))``;
    ``D' = sum_k w_k (P U) f(E') (P U)^dagger``.  Returns a dict of host arrays
    (``E_dft``, ``E_applied`` Ry, ``U``, ``f``, ``D_after``; the last two ``None``
    when a metal's occupations cannot be formed)."""
    nk, nb = (int(s) for s in np.shape(dH_total_ry)[:2])
    with timing.section("onsite_apply"):
        en = np.asarray(wfn.energies, dtype=np.float64)
        en = en[0] if en.ndim == 3 else en                                   # (nk_file, nb) Ry
        E_dft = en[np.asarray(sym.irr_idx_k, dtype=np.int64), :nb]           # full BZ, sym order
        dH = np.asarray(dH_total_ry, dtype=np.complex128)
        H = 0.5 * (dH + np.conj(np.swapaxes(dH, -1, -2)))                    # Herm(dH)
        H = H + E_dft[:, :, None] * np.eye(nb)[None]                         # diag(E_DFT) + Herm(dH)
        # Every rank holds the same host H: declared replicated, no communication.
        E_new, U = eigh_local(replicate_to_mesh(H, mesh))                    # ascending per k
        E_new = np.asarray(E_new)
        w = np.full(nk, 1.0 / nk, dtype=np.float64)
        f = _fixed_n_occupations(config, wfn, E_new, w, label="applied-spectrum")
        D_after = (None if f is None else
                   onsite_density_matrix(field.projections, w, f, field.setup, U=U, mesh=mesh))
        D_before, before = field.density_matrix, "the WFN occupations"
        if f is not None and not bool(wfn.occupations_are_exact_integer):
            # The same smearing on E_DFT (dH = 0), so the change is dH's alone.
            f0 = _fixed_n_occupations(config, wfn, E_dft, w, label="E_DFT reference")
            D_before = onsite_density_matrix(field.projections, w, f0, field.setup, mesh=mesh)
            before = "deck smearing at E_DFT (dH = 0)"
    shift = np.abs(E_new - E_dft).max() * RYD_TO_EV * 1e3
    log.info(f"onsite_breit applied once: {nk} k x {nb} bands, max|E' - E_DFT| = {shift:.5f} meV")
    if D_after is not None:
        _log_moment_change(np.asarray(D_before), np.asarray(D_after), field.setup, before)
    return {"E_dft": E_dft, "E_applied": E_new, "f": f,
            "D_before": np.asarray(D_before), "D_after": None if D_after is None else np.asarray(D_after)}


def write_artifact(out_path, *, field, dH_total, pieces, smooth, applied, sym, wfn, deck, nb):
    """``onsite_breit.h5`` through ``file_io.onsite_breit_output`` (rank 0 writes)."""
    from common.collectives import process_rank_world
    from common.parallel_transport import WFN_FINGERPRINT_SCHEME, wfn_fingerprint
    rank, _ = process_rank_world()
    extra = {f"dH_{k}_ev": v * RYD_TO_EV for k, v in pieces.items()}
    extra["P_kan"] = np.asarray(field.projections)
    extra["kirr_fullids"] = np.asarray(sym.kirr_fullids, dtype=np.int32)
    extra["irr_idx_k"] = np.asarray(sym.irr_idx_k, dtype=np.int32)
    if smooth is not None:
        extra["dH_smooth_ev"] = smooth * RYD_TO_EV
    if applied is not None:
        extra["E_dft_kn_ev"] = applied["E_dft"] * RYD_TO_EV
        extra["E_applied_kn_ev"] = applied["E_applied"] * RYD_TO_EV
        if applied["D_after"] is not None:
            extra["density_matrix_applied_ab"] = applied["D_after"]
            extra["density_matrix_reference_ab"] = applied["D_before"]
            extra["occupations_applied_kn"] = applied["f"]
    if rank == 0:
        from gw.kin_ion_io import _generator_commit
        commit = _generator_commit()
    else:
        commit = ""
    attrs = {"wfn_fingerprint": wfn_fingerprint(wfn), "wfn_fingerprint_scheme": WFN_FINGERPRINT_SCHEME,
             "generator_commit": commit, "input_file": os.path.basename(deck),
             "band_window": np.array([0, nb], dtype=np.int64),
             "applied_once": applied is not None,
             "dH_composition": "smooth alpha.A[J] + H_OS" if smooth is not None else "H_OS"}
    with timing.section("write_h5"):
        write_onsite_breit_h5(out_path, field=field, dH_kij_ry=dH_total,
                              kpoints=np.asarray(sym.unfolded_kpts, dtype=np.float64),
                              kset="full_bz", band_start_1based=1, attrs=attrs, extra=extra)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    input_dir = os.path.dirname(os.path.abspath(args.input))
    out_path = args.out or os.path.join(input_dir, "onsite_breit.h5")
    report_path = (os.path.abspath(args.report_file) if args.report_file else
                   os.path.join(os.path.dirname(os.path.abspath(out_path)), "onsite_breit.out"))
    mesh = RUNTIME.mesh
    with RunSession(RUNTIME, "onsite_breit", PreprocessingProductionReport, report_path,
                    stages=_STAGES, driver="onsite-breit",
                    purpose="on-site Breit operator at the DFT level") as run:
        run.report.begin(input_file=args.input)
        config, wfn, sym, meta, nb = load_run(args.input, mesh)
        # 1. setup: projector rows, kernels, keys (no Sigma: tt_exchange = False)
        setup = require_setup(onsite_breit_setup_for(config, wfn, input_dir=input_dir,
                                                     tt_exchange=False))
        # 2. D from the DFT occupations, h[D], H_OS = P^dagger h P
        field = onsite_breit_field(wfn, sym, setup, band_range=(0, nb), mesh=mesh)
        # 3. P^dagger h_piece P for dir_ae, dir_ps, x_ae, x_ps
        pieces = piece_band_matrices(field, mesh)
        # 4. paw: smooth <m|alpha.A[J]|n>; dH = smooth + H_OS (onsite_only: dH = H_OS)
        smooth = smooth_direct_field(config, wfn, sym, meta, setup, nb, mesh)
        dH_total = np.asarray(field.matrix) + (0.0 if smooth is None else smooth)
        # 5-6. eigh(diag(E_DFT) + Herm(dH)) per k; D' from (U, f'); moments before and after
        applied = None if args.no_apply else apply_once(config, wfn, sym, field, dH_total, mesh)
        # 7. onsite_breit.h5
        write_artifact(out_path, field=field, dH_total=dH_total, pieces=pieces, smooth=smooth,
                       applied=applied, sym=sym, wfn=wfn, deck=args.input, nb=nb)
        run.complete(files=[("on-site Breit field", "written", out_path),
                            ("wavefunctions", "read", wfn.path),
                            ("input deck", "read", args.input)])
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
