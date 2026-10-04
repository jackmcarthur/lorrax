"""htransform — the fH band-interpolation CLI (``bandstructure.fh_interp``).

    python -m bandstructure.htransform -i deck.in [--eqp-file eqp1.dat |
        --qp-rotations qp_wfn_rotations.h5] [-o bandstructure.dat]

Reads the deck's WFN and centroids, fits or reuses the Galerkin basis, runs
``fh_interp.h_transform`` on the deck's K_POINTS crystal_b path and writes
the interpolated bands; with ``get_centroids_fi`` it also builds the fine-k
BSE handoff (``bse_setup.compute_wfns_fi``).
"""
import os
import argparse

# THE startup call (runtime module docstring): env defaults, fail-fast
# hook, jax.distributed, CPU fallback, the run's clique-warmed ('x','y')
# mesh, compile cache, rank-0 report.  MUST run before this module's own
# `import jax` AND before `import numpy`: importing runtime is what sets
# OPENBLAS_THREAD_TIMEOUT, and OpenBLAS reads it in the constructor that
# runs with numpy (runtime.tune_blas_threading; tests/test_runtime_blas_env.py
# enforces this order).  This module is the CLI only; the library half is
# ``bandstructure.fh_interp``, which starts no runtime.
from runtime import debug_print, initialize_communicator_stack

import numpy as np                                                  # noqa: E402
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import jax

from common import timing
from common.units import RYD_TO_EV
from common.collectives import gather_to_host
# This driver reads a raw params dict rather than ``LorraxConfig``; use the
# parser-cached linalg profile rather than interpreting its public dial here.
from gw.gw_config import (
    distrib_la_batched_route_choices,
    eigh_backend_choices,
    linalg_resolution,
    read_cohsex_input,
    resolve_distrib_la_batched_route,
    resolve_eigh_backend,
)
from runtime.run_session import RunSession
from .production_report import HTransformProductionReport
from .fh_interp import (
    GALERKIN_BASIS_FILE, _build_mesh_xy, compact_galerkin_state,
    h_transform, initialize_kpath, initialize_wfns,
    resolve_qp_hamiltonian_state, setup_wfn_and_sym,
)
# The library half is bandstructure.fh_interp.  These names stay importable
# from the CLI module for open lanes until the wave after ARCH wave 0.
from .fh_interp import (                                             # noqa: F401
    build_R_grid_np, f_transform_eigs, outer_r_shell_mask,
    resolve_extra_rank_pad, resolve_galerkin_rank_multiplier,
    resolve_local_vbm_index, select_active_eigenpairs,
    validate_centroid_subset_idx,
)


def plot_bands(result):
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = result["kpath_data"]
    energies_sorted = result["energies_sorted"]
    gamma_exact = result["gamma_exact"]
    fermi_energy = result["fermi_energy"]
    nb_keep = result["nb_keep"]

    if kpath_frac is None or energies_sorted is None:
        raise RuntimeError("Plotting requires a K_POINTS {crystal_b} path in the input file")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    energies_ev = np.asarray(energies_sorted) * RYD_TO_EV
    gamma_exact_ev = (None if gamma_exact is None else
                      np.asarray(gamma_exact) * RYD_TO_EV)
    for band in range(nb_keep):
        ax.plot(x_path, energies_ev[:, band], lw=1.0, color='C0', alpha=0.9)

    x_ticks = x_path[np.asarray(node_indices, dtype=int)]
    labels = [(lbl or "") for lbl in node_labels]
    labelled_gamma = {
        int(idx) for idx, lbl in zip(node_indices, node_labels)
        if (lbl or "").strip() == "Gamma" or (lbl or "").strip() == "Γ"
    }
    for xpos in x_ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(x_ticks, labels)

    for pos_idx, idx in enumerate(gamma_positions or [0]):
        xpos = x_path[idx]
        label_exact = 'Exact Γ' if pos_idx == 0 else None
        label_ht = (('HT Γ' if idx in labelled_gamma else 'HT nearest Γ')
                    if pos_idx == 0 else None)
        if gamma_exact_ev is not None:
            ax.scatter(np.full(nb_keep, xpos), gamma_exact_ev, marker='o', facecolors='none', edgecolors='red', label=label_exact)
        ax.scatter(np.full(nb_keep, xpos), energies_ev[idx], marker='x', color='black', label=label_ht)

    ax.axhline(fermi_energy * RYD_TO_EV, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='$E_F$')
    ax.set_xlabel('k-path arc length (2π-scaled)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Hamiltonian-transform bands')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    plt.show() 


def plot_colored_bands(result, values, *, label, path, signed, e_fermi_ry,
                       clip=1.0):
    """Path bands colored by one band value on a linear scale clipped at ``clip``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _kf, x_path, node_indices, node_labels, _g = result["kpath_data"]
    energies = (np.asarray(result["energies_sorted"])
                + result["energy_reference_ry"] - e_fermi_ry) * RYD_TO_EV
    nq, nb = energies.shape
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(np.repeat(x_path[:nq], nb), energies.ravel(),
                    c=np.asarray(values)[:nq].ravel(), s=3,
                    cmap="coolwarm" if signed else "viridis",
                    vmin=-clip if signed else 0.0, vmax=clip)
    fig.colorbar(sc, ax=ax, label=label)
    ticks = x_path[np.asarray(node_indices, dtype=int)]
    for xpos in ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(ticks, [(lbl or "") for lbl in node_labels])
    ax.axhline(0.0, color='k', ls='--', lw=0.8)
    ax.set_xlim(x_path[0], x_path[nq - 1])
    ax.set_ylabel(r'$E - E_F$ (eV; $E_F$ of the coarse grid)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def add_band_character_arguments(parser):
    """The ``--color`` / ``--velocity`` / ``--moments-grid`` flags."""
    parser.add_argument(
        "--color", action="append", default=[],
        metavar="spin|orbital|orbital:[EL:]l",
        help="Color the path bands by <sigma.n> (n: the coarse-grid spin "
             "moment axis), by the modern-theory orbital moment m_n.n "
             "(orbital; needs --velocity), or by an atomic-orbital character, "
             "e.g. orbital:d or orbital:Fe:d (PP_PSWFC of the *.upf beside "
             "the deck); repeatable.  Writes bands_<color>.png and "
             "band_operators_path.npz beside the deck.")
    parser.add_argument(
        "--velocity", default=None, metavar="dipole.h5|dipole_qsgw.h5",
        help="The stored velocity of the WFN's own states, for --color "
             "orbital and the per-cell orbital moment: dipole.h5 "
             "(psp.get_dipole_mtxels) on a DFT WFN, an SC run's "
             "dipole_qsgw.h5 on its WFN_qp.h5.  Basis and WFN fingerprint "
             "are authenticated.")
    parser.add_argument(
        "--moments-grid", type=int, nargs=3, default=None,
        metavar=("NX", "NY", "NZ"),
        help="Interpolate H and n.sigma to this uniform grid and write "
             "moments.txt: sum_occ <sigma.n>, Fermi-Dirac at "
             "occ_smearing_width_ry with E_F re-found on the grid.")
    return parser


def band_character_and_moments(*, colors, moments_grid, velocity, wfn, sym,
                               wfn_path, params, ctilde, wfn_ctilde,
                               enk_sigma, meta, result, mesh, input_dir,
                               n_return_bands, a_band, energy_source, report):
    """Spin / orbital path coloring, coarse orbital totals, grid moments.

    ``ctilde`` builds H (QP-rotated under ``--qp-rotations``); the operator
    images take ``wfn_ctilde``, the unrotated rows of the WFN band basis
    the operators are computed in.

    1. ``<psi_kn|O|psi_km>`` on the coarse full BZ: sigma_a, the requested
       orbital-character projectors and, with ``velocity``, the stored
       velocity of the WFN's states (``orbital.stored_velocity``).
    2. Magnetization axis n from the WFN's QE schema
       (``bandstructure.orbital.magnetization_axis``).
    3. Path: the operators in the fH eigenvectors; color by <sigma.n>, the
       channel character, or n.m_n (``psp.orbital_response``; degenerate
       multiplets averaged).
    4. Coarse totals (``orbital.orbital_totals``), physical moments along n:
       orbital at the stored ceiling and extrapolated in 1/E_ceiling, spin
       -(g_e/2) sum_occ <sigma>, and their sum.
    5. Grid: H and sigma_a on ``moments_grid``; m = sum_q w sum_n f <sigma>.
    """
    if not colors and not moments_grid and velocity is None:
        return []
    from functools import partial
    import jax.numpy as jnp
    from gw.degen_average import average_within_degenerate_sets
    from psp.pseudos import load_pseudopotentials
    from .fh_interp import build_fH_R, require_newton_converged
    from .orbital import (band_operators, grid_moments, occupied_sums,
                          interpolate_band_operator, magnetization_axis,
                          orbital_totals, path_orbital_moments,
                          stored_velocity, _operator_R)

    report.heading("Band character and moments")
    log = report.emit
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    nk, nb_fit = int(ctilde.shape[0]), int(ctilde.shape[1])
    band_start = int(result["band_start"])
    # Electrons in the window: the count, less one per (spinor) band below.
    nelec = float(wfn.num_electrons) - band_start
    width = params.get("occ_smearing_width_ry")
    kT, kT_note = ((float(width), "occ_smearing_width_ry") if width is not None
                   else (1.0e-4, "no occ_smearing_width_ry: insulator step at"))
    channels = [c.split(":", 1)[1] for c in colors if c.startswith("orbital:")]
    for c in colors:
        if c not in ("spin", "orbital") and not c.startswith("orbital:"):
            raise ValueError(f"--color {c!r}: want spin, orbital or "
                             "orbital:[EL:]l")
    if "orbital" in colors and velocity is None:
        raise ValueError("--color orbital needs --velocity")
    if velocity is not None and not energy_source.startswith("the WFN's"):
        raise ValueError(
            "--velocity is the WFN's own velocity and pairs with its own "
            "energies; for QSGW run on the SC run's WFN_qp.h5 with its "
            "dipole_qsgw.h5, not --qp-rotations/--eqp-file (claim 3218)")
    pseudos = load_pseudopotentials(input_dir) if channels else {}
    if channels and not pseudos:
        raise FileNotFoundError(
            f"--color orbital:[EL:]l needs the *.upf beside the deck "
            f"({input_dir}) for the PP_PSWFC atomic functions")
    with mesh:
        ops, names = band_operators(
            wfn, (band_start, band_start + nb_fit), mesh, pseudos=pseudos,
            channels=channels)
    axis, axis_source = magnetization_axis(wfn)
    log(f"  [band operators] {len(names)} on {nk} coarse k x {nb_fit} "
        f"bands: {', '.join(names)}")
    log(f"  magnetization axis n = ({axis[0]:+.5f}, {axis[1]:+.5f}, "
        f"{axis[2]:+.5f}) from {axis_source}")

    def spin_lines(where, mu, m):
        return [f"  [{where}] E_F = {mu * RYD_TO_EV:.5f} eV (Fermi-Dirac kT = "
                f"{kT:.5f} Ry, {kT_note}; N = {nelec:g} electrons)",
                f"  [{where}] sum_occ <sigma> = ({m[0]:+.5f}, {m[1]:+.5f}, "
                f"{m[2]:+.5f}); along n: {float(axis @ m):+.5f}"]

    # Coarse-grid sums: a direct (uninterpolated) check of the grid route.
    diag = np.asarray(gather_to_host(
        jnp.diagonal(ops, axis1=2, axis2=3)))[:, :, :nb_fit].real
    E_k = np.asarray(gather_to_host(enk_sigma), dtype=np.float64).T
    mu_c, sigma_c, _ = occupied_sums(E_k, diag[:3], nelec, kT)
    for line in spin_lines(f"coarse {kgrid[0]}x{kgrid[1]}x{kgrid[2]}",
                           mu_c, sigma_c):
        log(line)

    files = []
    if velocity is not None:
        deps_tol = 1.4e-3 / RYD_TO_EV    # psp.orbital_magnetization's default
        parents, E_par, label = stored_velocity(
            velocity, wfn=wfn, wfn_path=wfn_path, sym=sym, mesh=mesh)
        mu, ceil, E_c, m = orbital_totals(
            parents, E_par, sym, nelec=float(wfn.num_electrons),
            width_ry=width, deps_tol_ry=deps_tol)
        fit = np.linalg.lstsq(np.c_[np.ones(len(ceil)), 1 / (E_c - mu)], m,
                              rcond=None)[0][0]
        spin = -0.5 * 2.00232 * sigma_c          # g_e = 2.00232
        lines = [
            f"Orbital moment (modern theory, coarse {kgrid[0]}x{kgrid[1]}x"
            f"{kgrid[2]}, energies: {energy_source})",
            f"  velocity: {velocity} = {label}; band_energies of that file",
            f"  mu = {mu * RYD_TO_EV:+.5f} eV ("
            + ("T = 0, midgap" if width is None else
               f"fixed-N Fermi-Dirac, k_B T = {float(width)} Ry") + "); "
            f"physical moments in mu_B per cell along n",
            f"  orbital, stored band ceiling {ceil[-1]} (E_c - mu = "
            f"{(E_c[-1] - mu) * RYD_TO_EV:.2f} eV): {float(axis @ m[-1]):+.6f}"
            f"  m = ({m[-1][0]:+.6f}, {m[-1][1]:+.6f}, {m[-1][2]:+.6f})",
            f"  orbital, 1/E_ceiling extrapolation over ceilings {ceil[0]}-"
            f"{ceil[-1]}: {float(axis @ fit):+.6f}",
            "  spin -(g_e/2) sum_occ <sigma> (g_e = 2.00232): "
            f"{float(axis @ spin):+.6f}",
            f"  spin + orbital at ceiling {ceil[-1]}: "
            f"{float(axis @ (spin + m[-1])):+.6f}",
            f"  spin + orbital extrapolated: {float(axis @ (spin + fit)):+.6f}"]
        for line in lines:
            log(line)
    if colors:
        kpath_frac = result["kpath_data"][0]
        if kpath_frac is None:
            raise ValueError("--color needs a K_POINTS {crystal_b} path")
        nq = int(kpath_frac.shape[0])
        with mesh:
            path_ops = gather_to_host(interpolate_band_operator(
                ops, wfn_ctilde, result["coeffs_on_path"],
                np.asarray(kpath_frac), kgrid, mesh))[:nq]
        expect = np.einsum('qaii->qai', path_ops).real
        extra = {}
        if "orbital" in colors:
            E_path = (np.asarray(gather_to_host(result["energies_sorted"]))
                      + result["energy_reference_ry"])[:nq]
            with mesh:
                m_path = path_orbital_moments(
                    parents, sym, slice(band_start, band_start + nb_fit),
                    wfn_ctilde, result["coeffs_on_path"], kpath_frac, kgrid,
                    mesh, E_path, deps_tol)
            m_n = average_within_degenerate_sets(
                np.einsum('a,qai->qi', axis, m_path), E_path, deps_tol)
            clip = float(np.percentile(np.abs(m_n), 95))
            log(f"  [orbital colors] n.m_n clipped at +-{clip:.4f} mu_B (95th "
                f"percentile; {np.mean(np.abs(m_n) > clip):.1%} of states)")
            extra = dict(orbital_moment_path=m_path, orbital_color=m_n,
                         deps_tol_ry=deps_tol)
        for c in colors:
            clip_c = 1.0
            if c == "spin":
                values = np.einsum('a,qai->qi', axis, expect[:, :3])
                label, signed = r"$\langle \sigma \cdot n \rangle$", True
            elif c == "orbital":
                values, signed, clip_c = m_n, True, clip
                label = (r"$m_n \cdot n$ ($\mu_B$; modern theory; clipped "
                         rf"at $\pm${clip:.3g})")
            else:
                values = expect[:, names.index(f"char_{c.split(':', 1)[1]}")]
                label, signed = f"{c.split(':', 1)[1]} character", False
            png = os.path.join(input_dir,
                               f"bands_{c.replace(':', '_')}.png")
            if jax.process_index() == 0:
                plot_colored_bands(result, values, label=label, path=png,
                                   signed=signed, e_fermi_ry=mu_c, clip=clip_c)
            files.append((f"bands colored by {c}", "written", png))
        npz = os.path.join(input_dir, "band_operators_path.npz")
        if jax.process_index() == 0:
            np.savez(npz, operator_names=np.asarray(names),
                     kpath_frac=np.asarray(kpath_frac),
                     x_path=result["kpath_data"][1],
                     energies_ev=result["energies_sorted"] * RYD_TO_EV,
                     energy_reference_ev=(result["energy_reference_ry"]
                                          - mu_c) * RYD_TO_EV,
                     spin_axis=axis, path_operators=path_ops, **extra)
        files.append(("path band operators", "written", npz))

    if moments_grid:
        grid = tuple(int(v) for v in moments_grid)
        with mesh:
            fH_R, f_params, _, _ = build_fH_R(
                ctilde, enk_sigma, kgrid, mesh, a_band_index=a_band)
            builders = [partial(_operator_R, ops[i:i + 1], wfn_ctilde,
                                kgrid, mesh) for i in range(3)]
            E, D, residual = grid_moments(
                fH_R, f_params, builders, kgrid, grid, n_return_bands, mesh)
        require_newton_converged(float(residual), where="moments grid")
        mu, m, f_top = occupied_sums(
            np.asarray(gather_to_host(E)), np.asarray(gather_to_host(D)),
            nelec, kT)
        lines = [
            f"Moments on the {grid[0]}x{grid[1]}x{grid[2]} htransform grid "
            f"(energies: {energy_source})",
            "  QE's sign: sum_occ <sigma> = n_up - n_down; the physical spin "
            "moment is -(g_e/2) mu_B sum_occ <sigma>",
            f"  magnetization axis n = ({axis[0]:+.5f}, {axis[1]:+.5f}, "
            f"{axis[2]:+.5f}) from {axis_source}",
        ] + spin_lines(f"grid {grid[0]}x{grid[1]}x{grid[2]}", mu, m) + [
            f"  max occupation of the top returned band = {f_top:.2e}",
        ]
        if f_top > 1e-6:
            lines.append("  WARNING: the returned window is not empty at "
                         "its top; raise ncond")
        stats = jax.local_devices()[0].memory_stats() or {}
        lines.append(f"  device peak (rank 0, whole run so far): "
                     f"{stats.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB")
        for line in lines:
            log(line)
        txt = os.path.join(input_dir, "moments.txt")
        if jax.process_index() == 0:
            with open(txt, "w", encoding="utf8") as fh:
                fh.write("\n".join(lines) + "\n")
        files.append(("grid moments", "written", txt))
    return files


def write_bands_to_file(output_path: str, energies_on_path, kpath_frac, x_path,
                        *, band_start: int = 0, nb_fit: int | None = None):
    if energies_on_path is None or kpath_frac is None or x_path is None:
        return
    # Same family as ``bse_io.write_eigenvectors_stream``: a WRITER must not
    # assume the layout of what it is handed.  ``energies_on_path`` comes
    # straight out of ``_post_kpath`` with the q axis tiled over the mesh.
    energies = gather_to_host(energies_on_path) * RYD_TO_EV
    kpoints = gather_to_host(kpath_frac)
    with open(output_path, 'w', encoding='utf8') as fh:
        fh.write('# idx_k idx_b kx ky kz s energy_eV\n')
        if nb_fit is not None:
            fh.write(
                f"# absolute_band_window=[{int(band_start)},"
                f"{int(band_start) + energies.shape[1]}) "
                f"fit_bands={int(nb_fit)} "
                f"guard_bands={int(nb_fit) - energies.shape[1]}\n")
        for ik in range(energies.shape[0]):
            for ib in range(energies.shape[1]):
                kx, ky, kz = kpoints[ik]
                s_coord = x_path[ik]
                fh.write(f"{ik:4d} {ib:4d} {kx: .8f} {ky: .8f} {kz: .8f} {s_coord: .8f} {energies[ik, ib]: .8f}\n")


def main(argv=None):
    parser = add_band_character_arguments(argparse.ArgumentParser(
        allow_abbrev=False, description="Hamiltonian interpolation driver"))
    parser.add_argument("-i", "--input", default="cohsex_test.in", help="Input file")
    parser.add_argument("-wfn", "--wfn-file", default=None, help="Override WFN file (e.g. WFN_qp.h5)")
    parser.add_argument("--plot", action="store_true", help="Show interpolated band plot")
    qp_group = parser.add_mutually_exclusive_group()
    qp_group.add_argument(
        "--eqp-file", default=None,
        help="Diagonal QP approximation: replace energies from eqp1.dat in "
             "the current WFN band labels; does not represent off-diagonal "
             "QP mixing. Mutually exclusive with --qp-rotations.")
    qp_group.add_argument(
        "--qp-rotations", default=None,
        help="Full QP Hamiltonian: consume matched U_mnk,E_qp from "
             "qp_wfn_rotations.h5 so f(H_QP)=U f(E_QP) U^H. The complete "
             "QP block must lie inside the fitted band window.")
    parser.add_argument("-o", "--output-file", default="bandstructure.dat",
                        help="Interpolated band table (relative paths are "
                             "resolved beside the input deck)")
    parser.add_argument("--report-file", default="htransform.out",
                        help="Human-readable calculation report (relative "
                             "paths are resolved beside the input deck)")
    parser.add_argument("--a-band", type=int, default=None,
                        help="Band index (0-based) whose bandwidth sets 'a'. "
                             "E.g. nval+ncond_keep-1. Default: top band.")
    parser.add_argument(
        "--guard-bands", type=int, default=4,
        help="Bands fitted above the requested nval+ncond output window. "
             "The f-transform is identically zero at the top of its own "
             "window, so standalone output requires interior returned bands. "
             "Default: 4 (the measured shoulder depth). Zero is retained only "
             "as a red/reproduction arm and will normally refuse.")
    parser.add_argument("--eigh-backend", default=None,
                        choices=eigh_backend_choices(),
                        help="Eigensolver for the fH_q eigendecomposition of "
                             "the get_centroids_fi handoff.  auto|off = the "
                             "q-batched native path; distributed|cusolvermp|"
                             "slate|scalapack spread ONE (rank, rank) tile "
                             "over the mesh through the distrib_la public API (wide "
                             "band windows).  ``distributed`` is the portable "
                             "spelling and the ONLY one that exists on a host "
                             "mesh, where it means ScaLAPACK pzheevd.  "
                             "A debugging override of the backend the deck's "
                             "``linalg`` dial resolves (default: the resolved "
                             "backend).")
    parser.add_argument(
        "--distrib-la-batched-route", default=None,
        choices=distrib_la_batched_route_choices(),
        help="A debugging override of the batch schedule the deck's "
             "``linalg`` dial resolves, for every Plan.batched call in this "
             "driver. auto preserves the backend's robust distributed route; "
             "batch_reshard moves q onto the mesh and runs whole-matrix local "
             "JAX linalg.")
    args = parser.parse_args(argv)
    input_dir = os.path.dirname(os.path.abspath(args.input))

    def _output_path(value: str) -> str:
        return value if os.path.isabs(value) else os.path.join(input_dir, value)

    output_path = _output_path(args.output_file)
    report_path = _output_path(args.report_file)
    with RunSession(RUNTIME, "htransform", HTransformProductionReport,
                    report_path) as run:
        report = run.report
        log = report.legacy_print
        if args.qp_rotations:
            _energy_source = (
                "full quasiparticle Hamiltonian from "
                f"{os.path.basename(args.qp_rotations)}")
        elif args.eqp_file:
            _energy_source = (
                "diagonal quasiparticle energies from "
                f"{os.path.basename(args.eqp_file)}")
        else:
            _energy_source = "the WFN's eigenvalues (QP for a WFN_qp.h5)"
        report.begin(input_file=args.input, output_file=output_path,
                     energy_source=_energy_source)
        report.architecture()

        params = read_cohsex_input(args.input)
        # Input file is the source of truth; CLI backend flags remain debug
        # overrides of the implementation selected by the resolved layout.
        eigh_backend = resolve_eigh_backend(params, override=args.eigh_backend)
        distrib_la_batched_route = resolve_distrib_la_batched_route(
            params, override=args.distrib_la_batched_route)
        use_low_mem_eigh = linalg_resolution(params).layout == "distributed"
        n_return_bands = int(params["nval"]) + int(params["ncond"])
    
        # Override WFN file if provided via CLI
        if args.wfn_file is not None:
            params["wfn_file"] = args.wfn_file
            log(f"Using WFN file from CLI: {args.wfn_file}")

        _wfn_path = (params["wfn_file"] if os.path.isabs(params["wfn_file"])
                     else os.path.join(input_dir, params["wfn_file"]))
        _qp_rotations_path = None
        if args.qp_rotations:
            _qp_rotations_path = (
                args.qp_rotations if os.path.isabs(args.qp_rotations)
                else os.path.join(input_dir, args.qp_rotations))
        _eqp_path = None
        if args.eqp_file:
            _eqp_path = (args.eqp_file if os.path.isabs(args.eqp_file)
                         else os.path.join(input_dir, args.eqp_file))
        from file_io.qp_wfn import refuse_conflicting_qp_state_sources
        refuse_conflicting_qp_state_sources(
            wfn_path=_wfn_path, eqp_file=_eqp_path,
            qp_rotations_file=_qp_rotations_path)

        # Resolve the concrete fine-k plan before the first setup/progress line.
        # Whole-state QRCP owns basis selection and has no Gram eigensolve.
        mesh_xy = _build_mesh_xy()
        wfn, sym = setup_wfn_and_sym(_wfn_path, mesh_xy=mesh_xy)
        from distrib_la import plan as _linalg_plan
        _fine_enabled = bool(params.get("get_centroids_fi", False))
        _fine_plan = (_linalg_plan(
            "eigh", mesh_xy, backend=eigh_backend, n=None,
            batched_route=distrib_la_batched_route)
            if _fine_enabled else None)
        report.environment(
            params=params, wfn=wfn,
            fine_plan=_fine_plan, fine_enabled=_fine_enabled)

        from common import sanity

        from common.progress import LoopProgress
        _setup_progress = LoopProgress(
            1, report.progress, title="wavefunction and Galerkin setup",
            item_name="stage", max_updates=1).start()
        _centroid_records = []
        _rank_records = []
        with timing.section("initialize_wfns"):
            wfn, sym, meta, mesh_xy, basis, enk_sigma = initialize_wfns(
                args.input, params, log, args.eqp_file,
                mesh_xy=mesh_xy, wfn_sym=(wfn, sym),
                n_guard_bands=args.guard_bands, progress_fn=report.progress,
                centroid_record_fn=_centroid_records.append,
                rank_record_fn=_rank_records.append,
                require_all_occupied=True,
                basis_path=GALERKIN_BASIS_FILE,
                distrib_la_batched_route=distrib_la_batched_route)
        _setup_progress.step()
        _setup_progress.finish()
        ctilde, B_at_mu = basis.ctilde, basis.basis_at_nodes
        qp_corrected_band_range = None
        if _qp_rotations_path is not None:
            (ctilde, enk_sigma,
             qp_corrected_band_range) = resolve_qp_hamiltonian_state(
                basis=basis, enk_sigma=enk_sigma, sym=sym, meta=meta,
                wfn=wfn,
                wfn_path=_wfn_path, qp_rotations_file=_qp_rotations_path,
                eqp_file=args.eqp_file, log_fn=log)
        # ── Galerkin-input gate ───────────────────────────────────────────
        # ``ctilde`` is the compact Galerkin coefficient table and ``enk_sigma``
        # is the band energies the whole
        # interpolation is anchored to — including, when ``--eqp-file`` is
        # given, energies read from a GW run that may itself have produced
        # garbage.  A −136 eV QP energy fed into htransform yields a
        # bandstructure.dat that is numerically finite, plots fine, and is
        # wrong.  Bracket it here, where the file name is still in scope.
        sanity.check_finite("htransform ctilde", ctilde, print_fn=log)
        sanity.check_finite("htransform band energies", enk_sigma, print_fn=log)
        # Bandwidth, not absolute energy: the zero of a pseudopotential
        # eigenvalue is convention-dependent, but the *spread* of a Σ-window
        # band set is not — 272 eV is far wider than any real
        # semicore-to-conduction window and so only fires on gross
        # corruption.  A subtler check (comparing --eqp-file energies against
        # the DFT ones they replace) is proposed but not implemented here;
        # see the workstream-O report.
        _enk = np.asarray(jax.device_get(enk_sigma), dtype=np.float64)
        if _enk.size:
            _spread = float(_enk.max() - _enk.min()) * RYD_TO_EV
            log(f"  E_nk spread: {_spread:.4f} eV over {_enk.size} states")
            sanity.check_in_range(
                "htransform E_nk bandwidth", np.array([_spread]),
                0.0, 20.0 * RYD_TO_EV, unit="eV", print_fn=log)

        # Band operators come from the WFN states, so their Galerkin image
        # C^T O C^* takes the unrotated rows; only H takes the QP rotation.
        ctilde, wfn_ctilde = (
            (compact_galerkin_state(ctilde, mesh_xy, log_fn=log),
             compact_galerkin_state(basis.ctilde, mesh_xy, log_fn=log))
            if _qp_rotations_path is not None
            and (args.color or args.moments_grid) else
            (compact_galerkin_state(ctilde, mesh_xy, log_fn=log),) * 2)
        del basis

        kpath_data = initialize_kpath(wfn, params)
        if len(_centroid_records) != 1:
            raise RuntimeError(
                "htransform initialize_wfns did not return exactly one centroid "
                f"closure record; got {len(_centroid_records)}.")
        if len(_rank_records) != 1:
            raise RuntimeError(
                "htransform initialize_wfns did not return exactly one Galerkin "
                f"rank record; got {len(_rank_records)}.")
        report.sampling(
            wfn=wfn, sym=sym, centroids=_centroid_records[0])
        _transform_progress = LoopProgress(
            1, report.progress, title="fH construction and path solution",
            item_name="stage", max_updates=1).start()
        _quality_records = []
        with mesh_xy, timing.section("h_transform"):
            result = h_transform(meta, ctilde, enk_sigma, wfn, kpath_data, log, mesh_xy,
                                 a_band_index=args.a_band,
                                 band_start=int(wfn.nelec) - int(params["nval"]),
                                 n_return_bands=n_return_bands,
                                 qp_corrected_band_range=qp_corrected_band_range,
                                 progress_fn=report.progress,
                                 quality_record_fn=_quality_records.append,
                                 sym=sym, return_coeffs=bool(args.color))
        _transform_progress.step()
        _transform_progress.finish()

        _centroid_path = params.get("centroids_file", "centroids_frac.txt")
        _centroid_path = (_centroid_path if os.path.isabs(_centroid_path) else
                          os.path.join(input_dir, _centroid_path))
        report.interpolation_space(
            params=params, wfn=wfn, meta=meta, result=result,
            enk_sigma_ry=enk_sigma, ctilde=ctilde,
            centroid_file=_centroid_path, energy_source=_energy_source,
            centroids=_centroid_records[0])
        report.spectral_compression(_rank_records[0])
        if len(_quality_records) != 1:
            raise RuntimeError(
                "htransform returned "
                f"{len(_quality_records)} interpolation-quality receipts; "
                "expected one")
        report.htransform_quality(_quality_records[0])
        report.path_summary(result=result)

        # Optional BSE interpolation handoff: fine-k wfns at coarse centroids.
        # Driven by ``get_centroids_fi`` + ``kgrid_fi`` + ``wfn_fi_{min,max}``;
        # see ``bandstructure.bse_setup.compute_wfns_fi`` for the contract.
        if params.get("get_centroids_fi", False):
            from .bse_setup import compute_wfns_fi
            b_min = int(params["wfn_fi_min"])
            b_max = int(params["wfn_fi_max"]) or int(ctilde.shape[1])
            # SPLASH RADIUS OF THE f-SHOULDER, named by
            # ``2026-08-11-fifth-wall-is-the-f-transform-shoulder.md`` §7 and
            # audited here.  ``wfn_fi_max`` unset DEFAULTS to the full band count
            # — zero guard bands, i.e. exactly the configuration that row
            # convicts.  ``compute_wfns_fi``'s f-shoulder gate is what refuses;
            # this warns first, in the vocabulary of the deck key the user would
            # have to change, so the refusal is not the first news of it.
            _n_guard = int(ctilde.shape[1]) - b_max
            if _n_guard < 4:
                log(f"  [warn] wfn_fi_max={b_max} leaves only {_n_guard} guard "
                    f"band(s) below the top of the htransform window "
                    f"({int(ctilde.shape[1])} bands)"
                    + (" — this is the ZERO-GUARD default (wfn_fi_max unset "
                       "means 'the whole window')" if not
                       int(params["wfn_fi_max"]) else "")
                    + f".  f(eps) is identically zero at and above "
                    f"max_k eps of the window's own top band, so the top of what "
                    f"you are asking BACK may be an arbitrary direction out of "
                    f"fH's null space.  Raise nband/ncond so the window extends "
                    f"above wfn_fi_max; the f-shoulder gate decides.")
            with mesh_xy, timing.section("wfns_fi"):
                wfns_fi = compute_wfns_fi(
                    ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                    kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
                    kgrid_fi=params["kgrid_fi"],
                    band_window_fi=(b_min, b_max),
                    mesh_xy=mesh_xy, a_band_index=args.a_band,
                    eigh_backend=eigh_backend,
                    use_low_mem_eigh=use_low_mem_eigh, log_fn=log,
                    distrib_la_batched_route=distrib_la_batched_route,
                )
            log(f"BSE setup: psi_rmu_Y={wfns_fi.psi_rmu_Y.shape} "
                f"P{wfns_fi.psi_rmu_Y.sharding.spec}, "
                f"psi_rmuT_X={wfns_fi.psi_rmuT_X.shape} "
                f"P{wfns_fi.psi_rmuT_X.sharding.spec}, "
                f"enk_full={wfns_fi.enk_full.shape}")

        _moment_files = band_character_and_moments(
                colors=args.color, moments_grid=args.moments_grid,
                velocity=args.velocity and _output_path(args.velocity),
                wfn=wfn, sym=sym, wfn_path=_wfn_path, params=params,
                ctilde=ctilde, wfn_ctilde=wfn_ctilde,
                enk_sigma=enk_sigma, meta=meta,
                result=result, mesh=mesh_xy, input_dir=input_dir,
                n_return_bands=n_return_bands, a_band=args.a_band,
                energy_source=_energy_source, report=report)

        if args.plot:
            plot_bands(result)

        # ── Writer gate ───────────────────────────────────────────────────
        # bandstructure.dat is the file downstream tooling and the regression
        # gate diff against ground truth; a NaN row silently changes the file
        # length rather than the exit code.
        sanity.check_finite(f"{os.path.basename(output_path)} energies",
                            result['energies_sorted'], print_fn=log)

        # Rank-0 writer gate: at P>1 every process reaches this line with the
        # same (replicated) energies and used to write the SAME shared-FS file
        # concurrently — a race that can interleave partial writes.  Same idiom
        # as the gw_output/gw_init writers.
        _write_progress = LoopProgress(
            1, report.progress, title="interpolated-band output",
            item_name="file", max_updates=1).start()
        if jax.process_index() == 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_bands_to_file(
                output_path,
                result['energies_sorted'],  # sorted & truncated to nb_keep, not raw eigenvalues
                kpath_data[0],
                kpath_data[1],
                band_start=result["band_start"],
                nb_fit=result["nb_fit"],
            )
        _write_progress.step()
        _write_progress.finish()

        # ── Outputs barrier, then CLOSE THE LOADER EXPLICITLY ─────────────
        # The block above is rank-0-only, so without this barrier the other
        # ranks arrive at the collective close while rank 0 is still writing
        # ``bandstructure.dat``.  The barrier is what makes every rank enter
        # the close at the same point; the close is the load-bearing half.
        #
        # ``initialize_wfns`` hands back a MESH-AWARE ``WfnLoader``
        # (``setup_wfn_and_sym`` -> ``WfnLoader(wfn_file, mesh=mesh_xy)``, and
        # this driver passes ``mesh_xy=None`` so ``initialize_wfns`` builds the
        # mesh itself), so at P>1 the loader picks the phdf5 backend and owns a
        # ``SlabIO`` whose ``close()`` runs an UNCONDITIONAL COLLECTIVE barrier
        # (``file_io/_slab_io_ffi.py``, ``_barrier("slab_io_ffi_close_attrs")``).
        # Left to ``WfnLoader.__del__``, that collective fires whenever the
        # garbage collector happens to drop the object during interpreter
        # shutdown — a moment no two ranks agree on, and rank 0's object graph
        # differs from the others' because of the writer block above.
        #
        # This is the defect measured and cured in ``bse.exciton_bands`` at
        # ``b3813d8f`` (FIX_exciton_exit_hang.md): three ranks parked in
        # ``__del__`` -> ``SlabIO.close`` -> ``sync_global_devices`` while the
        # fourth had already reached ``ffi.io._atexit_close_all`` ->
        # ``H5Fclose`` -> ``MPI_Barrier``; two disjoint collective domains,
        # neither ever satisfied, the payload complete and every output written,
        # and the step holding its GPUs at 4x100% CPU until the allocation died.
        # That report's §6 named THIS driver as the one remaining sibling with
        # the identical shape.  ``close()`` is idempotent and nulls the handles,
        # so the later ``__del__`` becomes a no-op on every rank; at P=1 both
        # the barrier and the SlabIO collective are already no-ops.
        from common.collectives import barrier
        barrier("htransform.outputs_written")
        wfn.close()
        _wfn_path = params["wfn_file"]
        _wfn_path = (_wfn_path if os.path.isabs(_wfn_path) else
                     os.path.join(input_dir, _wfn_path))
        _file_rows = [
            ("input deck", "read", args.input),
            ("DFT wavefunctions", "read", _wfn_path),
            ("ISDF centroids", "read", _centroid_path),
            ("Galerkin basis", "read or published",
             os.path.join(input_dir, GALERKIN_BASIS_FILE)),
        ]
        if args.eqp_file:
            _eqp_path = (args.eqp_file if os.path.isabs(args.eqp_file) else
                         os.path.join(input_dir, args.eqp_file))
            _file_rows.append(("QP energies", "read", _eqp_path))
        _file_rows.append(("interpolated bands", "written", output_path))
        run.complete(files=_file_rows + _moment_files)
    return 0


if __name__ == "__main__":
    # ``runtime.finalize_process``, not a bare ``SystemExit``: the explicit
    # ``wfn.close()`` above removes the collective from ``__del__``, and this
    # removes the SECOND unordered collective — ``ffi.io._atexit_close_all``,
    # whose ``H5Fclose`` on the restart/zeta contexts is collective too and
    # otherwise runs at whatever point each rank's interpreter teardown
    # reaches it.  ``finalize_process`` runs the effects barrier, the
    # distributed shutdown and the atexit hooks in ONE stated order on every
    # rank and then ends with ``os._exit``, so GC-driven ``__del__``s at
    # shutdown never run at all.  Same pattern as ``bse.exciton_bands``
    # (``b3813d8f``) and ``gw.gw_jax``, the sibling that has never hung.
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
