#!/usr/bin/env python3
"""Evaluate the DFT or current-QSGW interband plus Drude head spectrum.

This is a small end-to-end diagnostic, not a second head implementation.  It
loads the WFN and exact DFT velocity stage named by a LORRAX input file,
solves the fixed-electron BerkeleyGW-MP1 occupations, and calls
``gw.qsgw_head.head_s_tensor_sharded`` directly.  For one explicit 3D q0 it
writes

    chi00(q0,w) = q0.T @ S(w) @ q0
    epsinv00(q0,w) = 1 / (1 - 8*pi/|q0|^2 * chi00(q0,w))

The diagonal velocity and an explicit Fermi-surface quadrature supply the
ab-initio Drude tensor; no fitted plasma frequency is used.  With
``--dft-velocity-only`` this is strictly a DFT diagnostic.  Head/wing/body
local-field coupling and finite-q dispersion remain outside it.

Run on compute nodes through the normal LORRAX launcher, for example::

    lx run -N 4 -G 4 -n 16 --wait 600 python3 -u \
      "$LORRAX_CHECKOUT/tools/qsgw_head_spectrum.py" \
      -i /path/to/cohsex.in --q0-crystal 0 0 0.125 \
      --omega-max-ev 50 --omega-points 501 \
      -o /path/to/qsgw_head_spectrum.dat
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np


# Make ``python tools/qsgw_head_spectrum.py`` work without relying on the
# launcher's PYTHONPATH.  This precedes the runtime import; JAX must not be
# imported until initialize_communicator_stack has installed its environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Load a WFN + parallel_transport.h5, solve fixed-N MP1 "
            "occupations, and evaluate Lorrax's interband plus Drude "
            "S(omega)."
        ),
    )
    parser.add_argument("-i", "--input", required=True, help="LORRAX input file")
    parser.add_argument(
        "--q0-crystal",
        required=True,
        type=float,
        nargs=3,
        metavar=("QX", "QY", "QZ"),
        help="Nonzero q0 in reduced reciprocal-lattice coordinates",
    )
    parser.add_argument(
        "--omega-min-ev", type=float, default=0.0, help="First real frequency (eV)"
    )
    parser.add_argument(
        "--omega-max-ev", type=float, default=50.0, help="Last real frequency (eV)"
    )
    parser.add_argument(
        "--omega-points", type=int, default=501, help="Inclusive uniform-grid size"
    )
    parser.add_argument(
        "--eta-ev",
        type=float,
        default=None,
        help=(
            "Retarded broadening (eV); default is wcoul0_eta from the input "
            "file, whose native unit is Ry"
        ),
    )
    parser.add_argument(
        "--dft-velocity-only",
        action="store_true",
        help=(
            "read the completed exact-DFT velocity stage without requiring "
            "the Berry-connection/QSGW-rotation stage"
        ),
    )
    parser.add_argument(
        "--nbands",
        type=int,
        default=None,
        help=(
            "logical bands included in the head sum; default is the complete "
            "parallel-transport manifold"
        ),
    )
    parser.add_argument(
        "--drude-integration",
        choices=("none", "tetrahedron", "mp1"),
        default="none",
        help=(
            "Fermi-surface quadrature for the Drude tensor; tetrahedron "
            "integrates the linearly interpolated periodic grid without "
            "smearing, while mp1 samples -df/dE on the input k points"
        ),
    )
    parser.add_argument(
        "--electron-count",
        type=float,
        default=None,
        help=(
            "diagnostic fixed-N override; default is the physical weighted "
            "occupation count stored by the WFN"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="qsgw_head_spectrum.dat",
        help="Rank-0 text output",
    )
    args = parser.parse_args(argv)
    if args.omega_points < 1:
        parser.error("--omega-points must be >= 1")
    if args.nbands is not None and args.nbands < 1:
        parser.error("--nbands must be >= 1")
    if not np.isfinite(args.omega_min_ev) or not np.isfinite(args.omega_max_ev):
        parser.error("frequency endpoints must be finite")
    if args.omega_max_ev < args.omega_min_ev:
        parser.error("--omega-max-ev must be >= --omega-min-ev")
    if args.eta_ev is not None and (
        not np.isfinite(args.eta_ev) or args.eta_ev < 0.0
    ):
        parser.error("--eta-ev must be finite and >= 0")
    if args.electron_count is not None and (
        not np.isfinite(args.electron_count) or args.electron_count <= 0.0
    ):
        parser.error("--electron-count must be finite and positive")
    q0 = np.asarray(args.q0_crystal, dtype=np.float64)
    if not np.all(np.isfinite(q0)) or np.linalg.norm(q0) == 0.0:
        parser.error("--q0-crystal must be finite and nonzero")
    return args


def _write_spectrum(
    path: Path,
    *,
    rows: np.ndarray,
    input_file: Path,
    wfn_file: str,
    pt_file: str,
    q0_crystal: np.ndarray,
    q0_cart: np.ndarray,
    mu_ry: float,
    electron_count: float,
    occ_broadening_ev: float,
    eta_ry: float,
    nbands: int,
    nk: int,
    plasma_ev: float,
    drude_integration: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        (
            "lorrax_qsgw_head_spectrum schema=2",
            "physics=occupation-weighted interband plus ab-initio "
            "Fermi-surface Drude tensor",
            f"input={input_file}",
            f"wfn={wfn_file}",
            f"parallel_transport={pt_file}",
            "q0_crystal=" + " ".join(f"{x:.17g}" for x in q0_crystal),
            "q0_cart_bohr^-1=" + " ".join(f"{x:.17g}" for x in q0_cart),
            f"chemical_potential_ry={mu_ry:.17g}",
            f"chemical_potential_ev={mu_ry * _RYD_TO_EV:.17g}",
            f"mp1_occ_broadening_ev={occ_broadening_ev:.17g}",
            f"retarded_eta_ry={eta_ry:.17g}",
            f"electron_count_audit={electron_count:.17g}",
            f"nk_full={nk} nbands={nbands}",
            f"directional_plasma_frequency_ev={plasma_ev:.17g}",
            f"drude_integration={drude_integration}",
            "columns: omega_ev Re_chi00 Im_chi00 Re_epsinv00 Im_epsinv00",
        )
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    np.savetxt(temporary, rows, fmt="%.16e", header=header)
    os.replace(temporary, path)


_RYD_TO_EV = 13.605693122994


def _load_dft_velocity_stage(path, *, mesh, wfn, meta):
    """Load the exact DFT velocity stage needed by this diagnostic only."""
    from types import SimpleNamespace

    import h5py
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from common.parallel_transport import band_storage_extent, wfn_fingerprint
    from file_io.parallel_transport import (
        SCHEMA_VERSION,
        VELOCITY_DFT_DATASET,
    )
    from file_io.slab_io import SlabIO

    nb = int(meta.b_id_4_user)
    with h5py.File(path, "r") as raw:
        schema = int(raw["schema_version"][()])
        band_start = int(raw["band_start"][()])
        band_stop = int(raw["band_stop"][()])
        kgrid = np.asarray(raw["kgrid"][()], dtype=np.int32)
        reciprocal = np.asarray(
            raw["reciprocal_lattice_cart"][()], dtype=np.float64
        )
        fingerprint_raw = np.asarray(
            raw["wfn_fingerprint_utf8"][()], dtype=np.uint8
        )
    fingerprint = bytes(fingerprint_raw.tolist()).decode("ascii")
    expected_reciprocal = (
        np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat)
    )
    refusals = []
    if schema != int(SCHEMA_VERSION):
        refusals.append(f"schema_version={schema}, expected {SCHEMA_VERSION}")
    if (band_start, band_stop) != (0, nb):
        refusals.append(
            f"band manifold [{band_start},{band_stop}) != [0,{nb})"
        )
    if not np.array_equal(kgrid, np.asarray(wfn.kgrid, dtype=np.int32)):
        refusals.append("k grid differs from the current WFN")
    if not np.allclose(
        reciprocal, expected_reciprocal, rtol=0.0, atol=1.0e-13
    ):
        refusals.append("reciprocal lattice differs from the current WFN")
    if fingerprint != wfn_fingerprint(wfn):
        refusals.append("WFN fingerprint differs from the velocity artifact")
    if refusals:
        raise ValueError(
            f"{path}: refusing DFT velocity stage:\n  - "
            + "\n  - ".join(refusals)
        )
    nb_storage = band_storage_extent(mesh, nb)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        velocity = io.read_slab(
            VELOCITY_DFT_DATASET,
            shape=(3, int(meta.nk_tot), nb_storage, nb_storage),
            partition_spec=P(None, None, "x", "y"),
        )
    return SimpleNamespace(
        velocity_dft_cart=velocity,
        nb_logical=nb,
        reciprocal_lattice_cart=reciprocal,
    )


def _run(args: argparse.Namespace) -> int:
    # One bring-up, before any import that can initialize JAX.
    from runtime import initialize_communicator_stack

    runtime = initialize_communicator_stack()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from common import Meta, RYD_TO_EV
    from gw.fermi_surface import tetrahedron_delta_weights
    from common.wfn_transforms import get_enk_bandrange
    from ffi.common.ffi_loader import phdf5_init_mpi
    from file_io.paths import resolve_input_path
    from gw.efermi import mp1_negative_derivative, solve_mp1_occupations
    from gw.gw_config import LorraxConfig
    from gw.qsgw_head import (
        head_drude_tensor_sharded,
        head_samples_from_s,
        head_s_tensor_sharded,
        load_parallel_transport_head,
    )
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    from wfn_loader import WfnLoader
    import symmetry_maps

    rank = int(jax.process_index())

    def print0(*values, **kwargs):
        if rank == 0:
            kwargs.setdefault("flush", True)
            print(*values, **kwargs)

    input_file = Path(args.input).expanduser().resolve()
    config = LorraxConfig.from_input_file(str(input_file), print_fn=print0)
    if int(config.sys_dim) != 3:
        raise ValueError(
            "qsgw_head_spectrum currently defines the pointwise 8*pi/|q|^2 "
            f"comparison only for sys_dim=3; got sys_dim={config.sys_dim}."
        )
    width_ev = float(config.screening.occ_broadening_ev)
    if width_ev <= 0.0:
        raise ValueError(
            "qsgw_head_spectrum requires input key occ_broadening > 0 (eV)."
        )

    # The PT datasets use SlabIO, so mirror the GW driver's eager MPI bring-up
    # before the first collective HDF5 open.
    phdf5_init_mpi()
    mesh = runtime.mesh
    wfn = WfnLoader(config.paths.wfn_file, mesh=mesh)
    sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(
        wfn,
        sym,
        config.nval,
        config.ncond,
        config.nband,
        n_rmu=1,
        bispinor=config.bispinor,
    )
    meta.sys_dim = int(config.sys_dim)

    pt_file = resolve_input_path(
        config.input_dir, config.paths.parallel_transport_file
    )
    if args.dft_velocity_only:
        pt = _load_dft_velocity_stage(
            pt_file, mesh=mesh, wfn=wfn, meta=meta
        )
    else:
        pt = load_parallel_transport_head(
            pt_file, mesh=mesh, wfn=wfn, meta=meta
        )
    nb_available = int(pt.nb_logical)
    nb_logical = (
        nb_available if args.nbands is None else int(args.nbands)
    )
    if nb_logical > nb_available:
        raise ValueError(
            f"--nbands={nb_logical} exceeds the {nb_available}-band "
            "parallel-transport manifold."
        )
    nb_storage = int(pt.velocity_dft_cart.shape[-1])
    energies_kn, _ = get_enk_bandrange(
        wfn,
        sym,
        (0, nb_storage),
        (0, nb_storage),
        nspinor=meta.nspinor,
    )
    if tuple(energies_kn.shape) != (int(meta.nk_tot), nb_storage):
        raise ValueError(
            "expanded WFN energies do not match the parallel-transport "
            f"storage: E={energies_kn.shape}, expected "
            f"({meta.nk_tot},{nb_storage})."
        )

    capacity = float(spin_degeneracy_factor(wfn))
    kweights = np.full(int(meta.nk_tot), 1.0 / float(meta.nk_tot))
    mu_ry_device, occupations_logical = solve_mp1_occupations(
        energies_kn[:, :nb_logical],
        kweights,
        (float(wfn.num_electrons)
         if args.electron_count is None else float(args.electron_count)),
        width_ev / float(RYD_TO_EV),
        state_capacity=capacity,
    )
    mu_ry = float(np.asarray(jax.block_until_ready(mu_ry_device)))
    occupations_kn = jnp.pad(
        occupations_logical,
        ((0, 0), (0, nb_storage - nb_logical)),
        mode="constant",
        constant_values=0.0,
    )
    if args.drude_integration == "none":
        surface_weight_logical = None
    elif args.drude_integration == "mp1":
        surface_weight_logical = mp1_negative_derivative(
            energies_kn[:, :nb_logical],
            mu_ry_device,
            width_ev / float(RYD_TO_EV),
        )
    else:
        # ``head_drude_tensor_sharded`` carries the uniform 1/Nk factor.
        # Tetrahedron weights already integrate over the normalized BZ, so
        # multiply by Nk here to present the same contraction interface.
        tetra_weights = tetrahedron_delta_weights(
            np.asarray(energies_kn[:, :nb_logical], dtype=np.float64),
            np.asarray(sym.unfolded_kpts, dtype=np.float64),
            tuple(int(x) for x in wfn.kgrid),
            mu_ry,
        )
        surface_weight_logical = jnp.asarray(
            tetra_weights * float(meta.nk_tot), dtype=jnp.float64)
    surface_weight_kn = None
    if surface_weight_logical is not None:
        surface_weight_kn = jnp.pad(
            surface_weight_logical,
            ((0, 0), (0, nb_storage - nb_logical)),
            mode="constant",
            constant_values=0.0,
        )

    omega_ev = np.linspace(
        float(args.omega_min_ev),
        float(args.omega_max_ev),
        int(args.omega_points),
        dtype=np.float64,
    )
    omegas_ry = jnp.asarray(omega_ev / float(RYD_TO_EV), dtype=jnp.complex128)
    eta_ry = (
        float(config.head.wcoul0_eta)
        if args.eta_ev is None
        else float(args.eta_ev) / float(RYD_TO_EV)
    )
    s_cart = head_s_tensor_sharded(
        pt.velocity_dft_cart,
        energies_kn,
        occupations_kn,
        omegas_ry,
        mesh=mesh,
        nb_logical=nb_logical,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
        nspin=int(wfn.nspin),
        nspinor=int(meta.nspinor),
        eta_ry=eta_ry,
        surface_weight_kn=surface_weight_kn,
    )
    drude_tensor = None
    if surface_weight_kn is not None:
        drude_tensor = head_drude_tensor_sharded(
            pt.velocity_dft_cart,
            surface_weight_kn,
            mesh=mesh,
            nb_logical=nb_logical,
            cell_volume=float(meta.cell_volume),
            nk_tot=int(meta.nk_tot),
            nspin=int(wfn.nspin),
            nspinor=int(meta.nspinor),
        )
    s_host = np.asarray(jax.block_until_ready(s_cart), dtype=np.complex128)
    drude_host = (
        np.zeros((3, 3), dtype=np.complex128)
        if drude_tensor is None
        else np.asarray(jax.block_until_ready(drude_tensor), dtype=np.complex128)
    )
    static_sample = None
    static_kappa2 = None
    if surface_weight_kn is not None:
        surface_host = np.asarray(
            jax.block_until_ready(surface_weight_kn), dtype=np.float64)
        dos_ry_per_cell = (
            capacity * float(np.sum(surface_host)) / float(meta.nk_tot))
        static_kappa2 = (
            8.0 * np.pi * dos_ry_per_cell / float(meta.cell_volume))
        static_sample = head_samples_from_s(
            np.zeros((1, 3, 3), dtype=np.complex128),
            np.asarray([0.0 + 0.0j]),
            wfn=wfn,
            meta=meta,
            config=config,
            static_kappa2_bohr2=static_kappa2,
        )[0]
    occupations_host = np.asarray(
        jax.block_until_ready(occupations_logical), dtype=np.float64
    )

    q0_crystal = np.asarray(args.q0_crystal, dtype=np.float64)
    q0_cart = q0_crystal @ np.asarray(
        pt.reciprocal_lattice_cart, dtype=np.float64
    )
    q2 = float(np.dot(q0_cart, q0_cart))
    if q2 <= 1.0e-24:
        raise ValueError(
            "--q0-crystal maps to a numerically zero Cartesian vector."
        )
    chi00 = np.einsum("i,wij,j->w", q0_cart, s_host, q0_cart, optimize=True)
    qhat = q0_cart / np.sqrt(q2)
    plasma_ry2 = 8.0 * np.pi * float(
        np.real(np.einsum("i,ij,j->", qhat, drude_host, qhat)))
    plasma_ev = np.sqrt(max(plasma_ry2, 0.0)) * float(RYD_TO_EV)
    vcoul = 8.0 * np.pi / q2
    epsinv00 = 1.0 / (1.0 - vcoul * chi00)
    rows = np.column_stack(
        (omega_ev, chi00.real, chi00.imag, epsinv00.real, epsinv00.imag)
    )
    electron_count = capacity * float(
        np.einsum("k,kn->", kweights, occupations_host)
    )

    if rank == 0:
        output = Path(args.output).expanduser().resolve()
        _write_spectrum(
            output,
            rows=rows,
            input_file=input_file,
            wfn_file=str(config.paths.wfn_file),
            pt_file=str(pt_file),
            q0_crystal=q0_crystal,
            q0_cart=q0_cart,
            mu_ry=mu_ry,
            electron_count=electron_count,
            occ_broadening_ev=width_ev,
            eta_ry=eta_ry,
            nbands=nb_logical,
            nk=int(meta.nk_tot),
            plasma_ev=plasma_ev,
            drude_integration=args.drude_integration,
        )
        print0(
            f"Wrote {output}: {len(omega_ev)} frequencies, "
            f"mu={mu_ry * float(RYD_TO_EV):.9f} eV, "
            f"N={electron_count:.12f}."
        )
        print0(
            "Ab-initio Fermi-surface Drude tensor: directional "
            f"omega_p={plasma_ev:.9f} eV."
        )
        if static_sample is not None:
            print0(
                "Static tetrahedron DOS head: "
                f"kappa_TF^2={static_kappa2:.9g} bohr^-2, "
                f"vc0={static_sample.vc0:.9g}, "
                f"wcoul0={static_sample.wcoul0:.9g}."
            )
    return 0


if __name__ == "__main__":
    # Parse first so ``--help`` does not initialize the distributed runtime.
    _args = _parse_args()
    try:
        _rc = _run(_args)
    except BaseException:
        traceback.print_exc()
        _rc = 1
    from runtime import finalize_process

    finalize_process(_rc)
