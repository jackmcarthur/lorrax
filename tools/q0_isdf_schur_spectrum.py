#!/usr/bin/env python3
"""Direct q->0 body oracle plus the production ISDF Y-W-Z head fold."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--nbands", type=int, default=23)
    parser.add_argument("--electron-count", type=float, default=9.0)
    parser.add_argument("--eta-ev", type=float, default=0.1)
    parser.add_argument("--omega-max-ev", type=float, default=20.0)
    parser.add_argument("--omega-points", type=int, default=201)
    parser.add_argument(
        "--q0-crystal", type=float, nargs=3, default=(0.0, 0.0, 0.125))
    return parser.parse_args()


def _run(args):
    sys.path.insert(0, os.environ["LORRAX_CHECKOUT"] + "/src")
    from runtime import initialize_communicator_stack
    runtime = initialize_communicator_stack()

    import h5py
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    jax.config.update("jax_enable_x64", True)

    from common import Meta, RYD_TO_EV
    from common.shard_map import shard_map
    from file_io import load_centroids
    from file_io.paths import resolve_input_path
    from file_io.slab_io import SlabIO
    from ffi.common.ffi_loader import phdf5_init_mpi
    from gw.efermi import solve_mp1_occupations
    from gw.fermi_surface import tetrahedron_delta_weights
    from gw.gw_config import LorraxConfig
    from gw.gw_init import prepare_isdf_and_wavefunctions
    from gw.head_correction import fold_cartesian_head_wings_sharded
    from gw.qsgw_head import head_s_tensor_sharded, head_wings_sharded
    from gw.wavefunction_bundle import BandSlices
    from gw.w_isdf import solve_w
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    from wfn_loader import WfnLoader
    import symmetry_maps

    rank = int(jax.process_index())

    def print0(*values):
        if rank == 0:
            print(*values, flush=True)

    input_file = Path(args.input).resolve()
    config = LorraxConfig.from_input_file(str(input_file), print_fn=print0)
    phdf5_init_mpi()
    mesh = runtime.mesh
    wfn = WfnLoader(config.paths.wfn_file, mesh=mesh)
    sym = symmetry_maps.SymMaps(wfn)
    _, centroids, n_mu = load_centroids(
        config.paths.centroids_file, wfn.fft_grid)
    meta = Meta.from_system(
        wfn, sym, config.nval, config.ncond, config.nband, n_mu,
        config.bispinor)
    meta.rank = rank
    meta.n_proc = int(jax.process_count())
    meta.sys_dim = int(config.sys_dim)
    slices = BandSlices.from_band_edges(*meta.band_edges)
    tmp_dir = str(input_file.parent / "tmp")
    print0("SCHUR_Q0 stage=load_isdf_restart")
    isdf = prepare_isdf_and_wavefunctions(
        cfg=config, wfn=wfn, sym=sym, meta=meta,
        centroid_indices=centroids, band_slices=slices, mesh_xy=mesh,
        tmp_dir=tmp_dir,
        tensors_filename=os.path.join(tmp_dir, f"isdf_tensors_{n_mu}.h5"),
        print0=print0, bgw_v_grid_fn=None)
    wfns = isdf.wf_bundle

    pt_path = resolve_input_path(
        config.input_dir, config.paths.parallel_transport_file)
    with h5py.File(pt_path, "r") as raw:
        nb_stored = int(raw["band_stop"][()])
        reciprocal = np.asarray(
            raw["reciprocal_lattice_cart"][()], dtype=np.float64)
        print0(
            "SCHUR_Q0 velocity_validation_complete="
            f"{int(raw['velocity_validation_complete'][()])} "
            "(DFT velocity stage only)")
    with SlabIO(pt_path, mode="r", mesh=mesh) as io:
        velocity = io.read_slab(
            "velocity_dft_cart",
            shape=(3, int(meta.nk_tot), nb_stored, nb_stored),
            partition_spec=P(None, None, "x", "y"))

    nb = int(args.nbands)
    if not 1 <= nb <= nb_stored:
        raise ValueError(f"nbands={nb} outside saved [1,{nb_stored}]")
    energies = wfns.enk[:, :nb_stored]
    capacity = float(spin_degeneracy_factor(wfn))
    kweights = np.full(int(meta.nk_tot), 1.0 / float(meta.nk_tot))
    mu, occupations_logical = solve_mp1_occupations(
        energies[:, :nb], kweights, float(args.electron_count),
        float(config.screening.occ_broadening_ev) / float(RYD_TO_EV),
        state_capacity=capacity)
    mu.block_until_ready()
    occupations = jnp.pad(
        occupations_logical, ((0, 0), (0, nb_stored - nb)))
    tetra = tetrahedron_delta_weights(
        np.asarray(energies[:, :nb]), np.asarray(sym.unfolded_kpts),
        tuple(int(x) for x in wfn.kgrid), float(np.asarray(mu)))
    surface = jnp.pad(
        jnp.asarray(tetra * float(meta.nk_tot), dtype=jnp.float64),
        ((0, 0), (0, nb_stored - nb)))

    omega_ev = np.linspace(
        0.0, float(args.omega_max_ev), int(args.omega_points))
    eta_ry = float(args.eta_ev) / float(RYD_TO_EV)
    z = jnp.asarray(
        omega_ev / float(RYD_TO_EV) + 1j * eta_ry,
        dtype=jnp.complex128)
    print0("SCHUR_Q0 stage=head_and_wings")
    S_direct = head_s_tensor_sharded(
        velocity, energies, occupations, z, mesh=mesh, nb_logical=nb,
        cell_volume=float(meta.cell_volume), nk_tot=int(meta.nk_tot),
        nspin=int(wfn.nspin), nspinor=int(meta.nspinor), eta_ry=0.0,
        surface_weight_kn=surface)
    Y_x, Z_y = head_wings_sharded(
        velocity, wfns, energies, occupations, z, mesh=mesh,
        nb_logical=nb, nk_tot=int(meta.nk_tot), nspin=int(wfn.nspin),
        nspinor=int(meta.nspinor), eta_ry=0.0,
        surface_weight_kn=surface)

    def direct_body_q0(psi_x, psi_y, e, f, z_values):
        """Exact all-ordered-pair q=0 Kubo body, sharded on (mu_x,mu_y)."""
        block = 4
        nz = int(z_values.shape[0])
        nz_pad = ((nz + block - 1) // block) * block

        def local(px, py, en, occ, omega, pref):
            # b_nm(mu)=sum_s psi_n(mu)^* psi_m(mu).  Every rank owns a
            # distinct (mu_x,mu_y) output tile, so Px*Py share work evenly.
            bx = jnp.einsum(
                "ksmi,ksmj->kijm", jnp.conj(px), px, optimize=True)
            by = jnp.einsum(
                "ksni,ksnj->kijn", jnp.conj(py), py, optimize=True)
            de = en[:, :, None] - en[:, None, :]
            fd = occ[:, :, None] - occ[:, None, :]
            padded = jnp.pad(
                omega, (0, nz_pad - nz),
                constant_values=jnp.asarray(1j, dtype=jnp.complex128))
            blocks = padded.reshape(-1, block)
            out = jnp.zeros(
                (nz_pad, px.shape[2], py.shape[2]), dtype=jnp.complex128)
            indices = jnp.arange(blocks.shape[0], dtype=jnp.int32)

            def step(acc, node):
                ib, zb = node
                coeff = pref * fd[None, :, :, :] / (
                    zb[:, None, None, None] + de[None, :, :, :])
                value = jnp.einsum(
                    "wkij,kijm,kijn->wmn", coeff, bx,
                    jnp.conj(by), optimize=True)
                return jax.lax.dynamic_update_slice(
                    acc, value,
                    (ib * block, jnp.int32(0), jnp.int32(0))), None

            out, _ = jax.lax.scan(
                step, out, (indices, blocks), unroll=1)
            return out[:nz]

        fn = jax.jit(shard_map(
            local, mesh=mesh,
            in_specs=(
                P(None, None, "x", None), P(None, None, "y", None),
                P(None, None), P(None, None), P(None), P()),
            out_specs=P(None, "x", "y"), check_vma=False))
        pref = 2.0 / (
            float(meta.nk_tot) * float(max(int(wfn.nspin), 1))
            * float(max(int(meta.nspinor), 1)))
        return fn(
            psi_x[:, :, :, :nb], psi_y[:, :, :, :nb],
            e[:, :nb], f[:, :nb], z_values,
            jnp.asarray(pref, dtype=jnp.complex128))

    print0("SCHUR_Q0 stage=direct_kubo_body")
    chi_body = direct_body_q0(
        wfns.psi_xn, wfns.psi_yn, energies, occupations, z)
    chi_body.block_until_ready()
    v0_body = isdf.V_qmunu[0]
    v_scale = jnp.max(jnp.abs(v0_body))
    v_herm = jnp.max(
        jnp.abs(v0_body - jnp.conj(v0_body.T))) / v_scale
    v_herm_value = float(np.asarray(v_herm))
    print0(f"SCHUR_Q0 gamma_V_hermiticity_rel={v_herm_value:.6e}")
    if not np.isfinite(v_herm_value) or v_herm_value > 1.0e-10:
        raise ValueError("Gamma V is not a finite Hermitian ISDF body")

    # Reuse the production Dyson solver.  The direct oracle returned the
    # physical response, so divide out solve_w's matching internal prefactor.
    nz = int(z.shape[0])
    body_sharding = NamedSharding(mesh, P(None, "x", "y"))
    V_stack = jax.lax.with_sharding_constraint(
        jnp.broadcast_to(v0_body[None, :, :], (nz, n_mu, n_mu)),
        body_sharding)
    solve_pref = 2.0 / (
        np.sqrt(float(meta.nk_tot)) * float(max(int(meta.nspin), 1))
        * float(max(int(meta.nspinor), 1)))
    chi_solver = jax.lax.with_sharding_constraint(
        chi_body / solve_pref, body_sharding)
    print0("SCHUR_Q0 stage=dyson")
    W_body = solve_w(
        V_stack, chi_solver, meta, mesh, dyson_solver="local")
    W_body.block_until_ready()
    nonfinite = int(np.asarray(jnp.sum(~jnp.isfinite(W_body))))
    if nonfinite:
        raise FloatingPointError(
            f"W_body has {nonfinite} non-finite entries")

    print0("SCHUR_Q0 stage=production_schur_fold")
    S_folded = fold_cartesian_head_wings_sharded(
        S_direct, Y_x, W_body, Z_y, float(meta.cell_volume), mesh_xy=mesh)
    S_direct.block_until_ready()
    S_folded.block_until_ready()
    direct = np.asarray(S_direct)
    folded = np.asarray(S_folded)
    qfrac = np.asarray(args.q0_crystal, dtype=np.float64)
    qcart = qfrac @ reciprocal
    q2 = float(qcart @ qcart)
    vcoul = 8.0 * np.pi / q2
    chi_d = np.einsum("a,wab,b->w", qcart, direct, qcart)
    chi_f = np.einsum("a,wab,b->w", qcart, folded, qcart)
    eps_d = 1.0 / (1.0 - vcoul * chi_d)
    eps_f = 1.0 / (1.0 - vcoul * chi_f)
    wc_d = vcoul * (eps_d - 1.0)
    wc_f = vcoul * (eps_f - 1.0)
    occ_count = capacity * float(np.mean(np.sum(
        np.asarray(occupations_logical), axis=1)))
    corr = np.max(np.abs(folded - direct))
    if rank == 0:
        rows = np.column_stack((
            omega_ev, chi_d.real, chi_d.imag, chi_f.real, chi_f.imag,
            eps_d.real, eps_d.imag, eps_f.real, eps_f.imag,
            wc_d.real, wc_d.imag, wc_f.real, wc_f.imag))
        header = "\n".join((
            "lorrax_q0_isdf_schur schema=1",
            "body=exact all-ordered-pair q=0 Kubo sum in 96-centroid ISDF basis",
            "fold=S_eff=S_direct+Y W_body Z/cell_volume (production kernel)",
            f"source_head={os.environ['LORRAX_CHECKOUT']}",
            f"input={input_file}", f"nbands={nb}",
            f"electron_count={occ_count:.16g}",
            f"chemical_potential_ev={float(np.asarray(mu))*float(RYD_TO_EV):.16g}",
            f"occ_broadening_ev={float(config.screening.occ_broadening_ev):.16g}",
            f"eta_ev={float(args.eta_ev):.16g}",
            "q0_crystal=" + " ".join(map(str, qfrac)),
            "q0_cart_bohr^-1=" + " ".join(map(str, qcart)),
            f"v0={vcoul:.16g}",
            f"gamma_V_hermiticity_rel={v_herm_value:.6e}",
            f"max_abs_S_schur_correction={corr:.16e}",
            "columns: omega_ev Re_chi_direct Im_chi_direct "
            "Re_chi_folded Im_chi_folded Re_epsinv_direct "
            "Im_epsinv_direct Re_epsinv_folded Im_epsinv_folded "
            "Re_Wc_direct Im_Wc_direct Re_Wc_folded Im_Wc_folded"))
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, rows, fmt="%.16e", header=header)
        win = (omega_ev >= 3.0) & (omega_ev <= 10.0)
        pd = omega_ev[win][np.argmax(-eps_d.imag[win])]
        pf = omega_ev[win][np.argmax(-eps_f.imag[win])]
        print0(
            f"SCHUR_Q0 direct_peak_ev={pd:.6f} folded_peak_ev={pf:.6f}")
        print0(f"SCHUR_Q0 wrote={out}")


if __name__ == "__main__":
    parsed = _parse_args()
    code = 0
    try:
        _run(parsed)
    except BaseException:
        traceback.print_exc()
        code = 1
    from runtime import finalize_process
    finalize_process(code)
PATCH
