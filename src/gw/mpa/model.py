"""Build one disk-bounded MPA screening model."""

from __future__ import annotations

import os
import re

import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from file_io import mpa_store
from gw.mpa import evaluator, fit_driver, sample_plan


_CHI = "chi_qmunu_z"
_WC = "Wc_qmunu_z"


def iteration_artifact_paths(run_dir, label):
    """Return the two disk artifacts owned by one MPA screening map."""
    root = os.path.abspath(os.fspath(run_dir))
    return (
        os.path.join(root, f"mpa_samples_{label}.h5"),
        os.path.join(root, f"mpa_fit_{label}.h5"),
    )


def retain_iteration_artifacts(run_dir, label, *, print_fn=print):
    """Collectively retain only one completed SC map's sample/fit stores.

    Only exact managed names ``mpa_{samples,fit}_sc_NNNN.h5`` are eligible;
    an external input fit is never touched.  The caller invokes this only
    after the current map has built and consumed its replacement, so a
    failure preserves the last usable generation.  Scanning rather than
    deleting only ``N-1`` also removes stale later maps from a shorter rerun.
    """
    from common.collectives import barrier, process_rank

    root = os.path.abspath(os.fspath(run_dir))
    keep = set(iteration_artifact_paths(root, label))
    managed = re.compile(r"mpa_(?:samples|fit)_sc_[0-9]{4}\.h5\Z")
    removed = []
    is_root = process_rank() == 0
    if is_root and os.path.isdir(root):
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if managed.fullmatch(name) and path not in keep:
                os.remove(path)
                removed.append(name)
    barrier(f"mpa.model.retain.{label}", print_fn=print_fn)
    if is_root and removed:
        print_fn(f"  MPA scratch: retained {label}; discarded: "
                 + ", ".join(removed))


def _q_wedge(sym, centroid_indices, meta):
    from gw.v_q_g_flat import _resolve_ibz_q_list
    from symmetry_maps import QirrTables

    (_, q_frac, irr, sym_idx, perm, wraps, use_ibz,
     resolution) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=tuple(meta.kgrid), fft_grid=tuple(meta.fft_grid),
        context="MPA chi/Wc q-wedge storage", return_resolution=True)
    if not use_ibz:
        raise ValueError("MPA disk sampling requires an orbit-closed q wedge")
    tables = QirrTables(
        irr_idx_q=irr, sym_idx_q=sym_idx, q_irr_frac=q_frac,
        sym_perm=perm, L_table=wraps,
        n_sym_spatial=int(np.asarray(sym.sym_matrices).shape[0]))
    return np.asarray(sym.q_irr_full_idx, np.int32), tables, resolution.verdict


def _to_wedge(value, q_idx, mesh_xy):
    from symmetry_maps import slice_q_full_to_ibz
    return slice_q_full_to_ibz(
        value, q_idx,
        out_sharding=NamedSharding(mesh_xy, P(None, "x", "y")))


def _write_sample(path, index, value, q_idx, meta, mesh_xy, n_z):
    value = _to_wedge(value, q_idx, mesh_xy)
    value.block_until_ready()
    mpa_store.write_w_slab_collective(
        path, _CHI, index, value, mesh_xy=mesh_xy,
        global_shape=(n_z, q_idx.size, meta.n_rmu, meta.n_rmu))
    del value


def _fit_fixed_head(fit_path, head_resolver, probe_omega):
    """Store the established two-point DFT head beside each body fit.

    This is deliberately not an arbitrary-frequency MPA head: it reuses the
    DFT-basis scalar PPM head while the body is rebuilt.  Local-field head/wing
    dynamics are omitted until their production producer lands.
    """
    from gw.head_correction import fit_head_ppm_from_samples

    z = np.asarray([0.0j, complex(probe_omega)], np.complex128)
    static, probe = (head_resolver.at(complex(point)) for point in z)
    wc = np.asarray([
        complex(static.wcoul0) - complex(static.vc0),
        complex(probe.wcoul0) - complex(probe.vc0),
    ])
    head = fit_head_ppm_from_samples(
        static, probe, probe_omega=complex(probe_omega))
    Omega = np.asarray([complex(head.omega_h, -1.0e-6)], np.complex128)
    B = np.asarray([complex(head.R_h)], np.complex128)
    reconstructed = B[0] / (z - Omega[0]) - B[0] / (z + Omega[0])
    residual = np.max(np.abs(reconstructed - wc))
    backward = residual / max(float(np.max(np.abs(wc))), 1.0)
    mpa_store.write_head_fit(
        fit_path, z, wc, Omega, B,
        energy_unit="Ry",
        fit_condition=1.0, fit_backward_error=backward,
        fit_max_abs_residual=residual, model="fixed_dft_gn")


def _solve_wc(sample_path, V, n_z, q_idx, meta, mesh_xy, dyson_solver=None):
    from gw.w_isdf import solve_w

    shape = (n_z, q_idx.size, meta.n_rmu, meta.n_rmu)
    for index in range(n_z):
        chi, _ = mpa_store.read_w_slab_collective(
            sample_path, _CHI, index, mesh_xy=mesh_xy)
        Wc = solve_w(
            V, chi, meta, mesh_xy, dyson_solver=dyson_solver) - V
        Wc.block_until_ready()
        mpa_store.write_w_slab_collective(
            sample_path, _WC, index, Wc, mesh_xy=mesh_xy,
            global_shape=shape)
        del chi, Wc


def _fit_body(sample_path, fit_path, z, n_p, tile_bytes, mesh_xy):
    return fit_driver.run_fit_driver(
        sample_path, _WC, fit_path, z, n_p, mesh_xy=mesh_xy,
        tile_bytes=tile_bytes)


def build_mpa_fit(
    run_dir, label, *, wfns, V_q, quad, sym, centroid_indices,
    head_resolver, config, meta, mesh_xy, energy_reference=0.0,
    tile_bytes=None, print_fn=print,
):
    """Write chi/Wc samples and fitted q-wedge poles; return the fit path."""
    if config.mpa.material_class != "insulator":
        raise NotImplementedError(
            "GATE mpa_metal_evaluator_unavailable: the configured metallic "
            "double-parallel sample plan is available through "
            "config.mpa.sample_plan(omega_m_ry), but build_mpa_fit cannot "
            "evaluate it until occupation-weighted interband/intraband chi "
            "and Sigma branches land. FALSE case: "
            "config.mpa.material_class == 'insulator'.")

    from common.collectives import barrier, process_rank
    from gw.minimax_config import MinimaxConfig
    from gw.minimax_screening import build_imag_quadrature
    from gw.w_isdf import compute_chi0, compute_chi0_contour

    root = os.path.abspath(os.fspath(run_dir))
    if process_rank() == 0:
        os.makedirs(root, exist_ok=True)
    barrier("mpa.model.mkdir", print_fn=print_fn)

    n_p = int(config.mpa.n_poles)
    omega_m = float(quad.x_max)
    plan = config.mpa.sample_plan(omega_m)
    z_all = sample_plan.plan_z(plan)
    sample_plan.refuse_unsupported(plan, delta_max=omega_m)
    q_idx, tables, closure_verdict = _q_wedge(sym, centroid_indices, meta)
    sample_path, fit_path = iteration_artifact_paths(root, label)
    varpi = np.unique(z_all.imag)
    line = np.searchsorted(varpi, z_all.imag).astype(np.int32)
    common = dict(
        mesh_xy=mesh_xy, n_omega=z_all.size, n_q_on_disk=q_idx.size,
        n_mu=meta.n_rmu, n_rmu_logical=meta.n_rmu, tables=tables,
        closure_verdict=closure_verdict,
        omega=z_all, omega_line=line, energy_unit="Ry",
        sampling={"protocol": "double_parallel", "varpi": varpi,
                  "n_p": n_p, "alpha": config.mpa.sampling_alpha,
                  "omega_max": omega_m})
    mpa_store.allocate_w_omega_collective(
        sample_path, _CHI, mode="w", **common)
    mpa_store.allocate_w_omega_collective(
        sample_path, _WC, mode="a", **common)

    routes = sample_plan.plan_routes(plan)
    for point in routes["existing"]:
        used = quad if point["character"] == "static" else \
            build_imag_quadrature(
                quad, point["varpi"],
                MinimaxConfig(
                    target_error=config.minimax_config.target_error,
                    max_nodes=config.minimax_config.max_nodes))
        chi = compute_chi0(
            wfns, used, meta, mesh_xy, energy_reference=energy_reference)
        _write_sample(
            sample_path, point["index"], chi, q_idx, meta, mesh_xy,
            z_all.size)
    for varpi_i, points in routes["lines"]:
        z = np.asarray([point["z"] for point in points])
        rule = evaluator.damped_line_rule(
            varpi_i, omega_m + float(np.max(np.abs(z.real))),
            rel_tol=config.minimax_config.target_error,
            max_order=config.minimax_config.max_nodes)
        t, h = rule["t"], rule["h"]
        tau = np.concatenate((1j * t, -1j * t))
        signs = np.concatenate((np.ones(t.size, np.int8),
                                -np.ones(t.size, np.int8)))
        weights = np.broadcast_to(
            np.concatenate((1j * h, -1j * h)), (z.size, 2 * t.size))
        values = compute_chi0_contour(
            wfns, tau, weights, signs, z, meta, mesh_xy,
            energy_reference=energy_reference)
        values = (values,) if z.size == 1 else values
        for point, chi in zip(points, values):
            _write_sample(
                sample_path, point["index"], chi, q_idx, meta, mesh_xy,
                z_all.size)

    V = _to_wedge(V_q, q_idx, mesh_xy)
    _solve_wc(
        sample_path, V, z_all.size, q_idx, meta, mesh_xy,
        config.backend.w_dyson_solver)
    _, report = _fit_body(
        sample_path, fit_path, z_all, n_p, tile_bytes, mesh_xy)
    # ``write_head_fit`` is an intentionally small serial-h5py metadata
    # writer.  The body fit above is collective SlabIO; letting every rank
    # enter this final delete/create/write sequence races one HDF5 group.
    if process_rank() == 0:
        _fit_fixed_head(
            fit_path, head_resolver, 1j * float(config.ppm.omega_p))
    barrier("mpa.model.head_fit", print_fn=print_fn)
    print_fn(fit_driver.format_cost_report(report))
    return fit_path


__all__ = [
    "build_mpa_fit",
    "retain_iteration_artifacts",
    "iteration_artifact_paths",
]
