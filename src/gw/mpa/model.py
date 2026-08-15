"""Build one disk-bounded MPA screening model."""

from __future__ import annotations

import os

import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from file_io import mpa_store
from gw.mpa import evaluator, fit_driver, sample_plan


_CHI = "chi_qmunu_z"
_WC = "Wc_qmunu_z"


def make_mpa_plan(config, quad):
    """Build and validate the one frequency plan shared by body and head."""
    n_p = int(config.mpa.n_poles)
    omega_m = float(quad.x_max)
    plan = sample_plan.mpa_plan(
        n_p, omega_m, material_class=config.mpa.material_class,
        alpha=config.mpa.sampling_alpha,
        varpi_near=config.mpa.varpi_near_ry,
        varpi_far=config.mpa.varpi_far_ry,
        origin_shift=config.mpa.metal_origin_shift_ry, energy_unit="Ry")
    sample_plan.refuse_unsupported(plan, delta_max=omega_m)
    return plan


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
    from gw.qsgw_utils import remove_managed

    removed = remove_managed(
        run_dir, r"mpa_(?:samples|fit)_sc_[0-9]{4}\.h5\Z",
        keep=iteration_artifact_paths(run_dir, label),
        barrier_tag=f"mpa.model.retain.{label}", print_fn=print_fn)
    if removed:
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


def _fit_head_samples(
    fit_path, head_samples, z, n_p, grid_hash, mesh_xy, *, model,
    occupation_state=None,
):
    """Fit scalar Wc_head on the body's exact complex-frequency grid."""
    wc = np.asarray([
        complex(sample.wcoul0) - complex(sample.vc0)
        for sample in head_samples
    ], dtype=np.complex128)
    fitted = fit_driver.fit_scalar_samples(wc, z, n_p)
    provenance = {
        "solve_mode": fitted["solve"],
        "solve_affine": fitted["affine"],
        "solve_rcond": fitted["rcond"],
        "eig_mode": fitted["eig"],
        "n_valid": fitted["n_valid"],
        "condition_max_allowed": 1.0 / fitted["rcond"],
        "backward_error_max_allowed": float(
            np.sqrt(np.finfo(np.float64).eps)),
    }
    mpa_store.write_head_fit_collective(
        fit_path, z, wc, fitted["Omega_p"], fitted["B_p"],
        mesh_xy=mesh_xy, energy_unit="Ry",
        fit_condition=fitted["condition"],
        fit_backward_error=fitted["backward_error"],
        fit_max_abs_residual=fitted["max_abs_residual"],
        grid_hash=grid_hash, fit_provenance=provenance, model=model,
        occupation_state=occupation_state)


def _solve_wc(
    sample_path,
    V,
    n_z,
    q_idx,
    meta,
    mesh_xy,
    dyson_solver=None,
    *,
    head_response=None,
    wfn=None,
    config=None,
):
    from gw.qsgw_head import (
        IterationHeadSamples,
        finalize_iteration_head_sample,
    )
    from gw.w_isdf import solve_w

    if head_response is not None and len(head_response.omegas) < int(n_z):
        raise ValueError("MPA head response does not cover the body sample grid")
    gamma = None
    if head_response is not None and head_response.Y_x is not None:
        matches = np.flatnonzero(np.asarray(q_idx, dtype=np.int64) == 0)
        if matches.size != 1:
            raise ValueError(
                "MPA Schur head requires exactly one Gamma row in the q wedge")
        gamma = int(matches[0])

    head_samples = []
    shape = (n_z, q_idx.size, meta.n_rmu, meta.n_rmu)
    for index in range(n_z):
        chi, _ = mpa_store.read_w_slab_collective(
            sample_path, _CHI, index, mesh_xy=mesh_xy)
        W = solve_w(V, chi, meta, mesh_xy, dyson_solver=dyson_solver)
        W.block_until_ready()
        if head_response is not None:
            head_samples.append(finalize_iteration_head_sample(
                head_response,
                index,
                None if gamma is None else W[gamma],
                wfn=wfn,
                meta=meta,
                config=config,
                mesh=mesh_xy,
            ))
        Wc = W - V
        Wc.block_until_ready()
        mpa_store.write_w_slab_collective(
            sample_path, _WC, index, Wc, mesh_xy=mesh_xy,
            global_shape=shape)
        del chi, W, Wc

    if head_response is None:
        return None
    for index in range(int(n_z), len(head_response.omegas)):
        head_samples.append(finalize_iteration_head_sample(
            head_response,
            index,
            None,
            wfn=wfn,
            meta=meta,
            config=config,
            mesh=mesh_xy,
        ))
    return IterationHeadSamples(
        omegas=head_response.omegas,
        samples=tuple(head_samples),
        sigma_energies_ry=head_response.sigma_energies_ry,
        sigma_occupations=head_response.sigma_occupations,
        efermi_ry=head_response.efermi_ry,
    )


def _fit_body(sample_path, fit_path, z, n_p, tile_bytes, mesh_xy):
    return fit_driver.run_fit_driver(
        sample_path, _WC, fit_path, z, n_p, mesh_xy=mesh_xy,
        tile_bytes=tile_bytes)


def _metal_kminq_rows(sym, q_idx):
    """Per-wedge-row flat ``k → k−q`` maps for the static origin sample.

    Row order follows the stored wedge (``sym.q_irr_full_idx`` order); the
    Gamma row's map must be the identity, which is asserted because it is
    the one cheap invariant that discriminates a wedge-row/kq-column
    ordering mismatch.
    """
    from common.kq_mapping import kminq_idx_for_iq

    q_idx = np.asarray(q_idx, dtype=np.int64)
    rows = np.stack([
        kminq_idx_for_iq(sym, j) for j in range(int(q_idx.size))
    ])
    for j in np.flatnonzero(q_idx == 0):
        if not np.array_equal(
                rows[j], np.arange(rows.shape[1], dtype=rows.dtype)):
            raise ValueError(
                "MPA metal q wedge: the Gamma row's k-q map is not the "
                "identity, so the stored wedge ordering does not match "
                "SymMaps.kq_map columns")
    return rows


def _require_metal_occupations(config, occupation_state):
    """One owner of the metal-needs-occupations refusal (two gates pin it:
    the build_mpa_fit entry, before any inode exists, and the
    _evaluate_samples seam for direct callers)."""
    if config.mpa.material_class != "insulator" and occupation_state is None:
        raise ValueError(
            "GATE mpa_metal_needs_occupations: a metal MPA plan requires "
            "occupation_state (gw.efermi.OccupationState); got None. "
            "FALSE case: material_class == 'insulator', or an "
            "OccupationState was passed.")


def _evaluate_samples(
    wfns, routes, quad, config, meta, mesh_xy, *,
    energy_reference, occupation_state, write_full, write_wedge,
    static_gamma_override, gamma_row, kminq_rows,
):
    """Evaluate every plan point through its route's kernel.

    Insulating plans keep the historical kernels on a byte-identical code
    path.  Metal plans (shifted-origin double-parallel protocol) route the
    near line's first sample through the exact static divided-difference
    kernel — the damped contour rule at varpi = 2e-5 Ry needs ~1.0e6 nodes
    (probe record runs/records/metal_mpa_wave1_20260815/I1_origin_probe.md)
    — and every other point through the fractional contour kernel with the
    rule bandwidth derived from the occupation supports, not ``quad.x_max``.
    ``write_full`` takes a full-grid chi (the writer wedges it);
    ``write_wedge`` takes already-wedge-shaped rows.
    """
    from gw.minimax_config import MinimaxConfig
    from gw.minimax_screening import build_imag_quadrature
    from gw.w_isdf import (
        compute_chi0,
        compute_chi0_contour,
        compute_chi0_contour_fractional,
        compute_chi0_static_fractional,
        occupation_support_bandwidth,
    )

    metal = config.mpa.material_class != "insulator"
    _require_metal_occupations(config, occupation_state)
    omega_m = float(quad.x_max)
    if metal:
        delta_max = occupation_support_bandwidth(
            wfns.enk, occupation_state.f_kn)

    for point in routes["existing"]:
        if not metal:
            used = quad if point["character"] == "static" else \
                build_imag_quadrature(
                    quad, point["varpi"],
                    MinimaxConfig(
                        target_error=config.minimax_config.target_error,
                        max_nodes=config.minimax_config.max_nodes))
            chi = compute_chi0(
                wfns, used, meta, mesh_xy, energy_reference=energy_reference)
            if (
                point["character"] == "static"
                and static_gamma_override is not None
            ):
                chi = chi.at[0].set(static_gamma_override[0])
            write_full(point, chi)
        elif point["role"].startswith("near"):
            # The shifted-origin slot stores the exact static value; the
            # fit reads sample_z = i*varpi_1, and the inconsistency is
            # bounded by (varpi_1/(q*v_F))^2 at finite q (W1.a-2).
            chi_w = compute_chi0_static_fractional(
                wfns, meta, mesh_xy, occupation_state=occupation_state,
                kminq_rows=kminq_rows)
            if static_gamma_override is not None and gamma_row is not None:
                chi_w = chi_w.at[gamma_row].set(static_gamma_override[0])
            write_wedge(point, chi_w)
        else:
            # Far pure-imaginary point: the fractional contour is cheap at
            # O(1) Ry line heights (31 nodes at varpi=1, tol 1e-6).
            rule = evaluator.damped_line_rule(
                point["varpi"], delta_max + abs(point["omega"]),
                rel_tol=config.minimax_config.target_error,
                max_order=config.minimax_config.max_nodes)
            chi = compute_chi0_contour_fractional(
                wfns, rule["t"], rule["h"],
                np.asarray([point["z"]], dtype=np.complex128),
                meta, mesh_xy,
                occupations=occupation_state.f_kn,
                energy_reference=float(occupation_state.mu_ry))
            write_full(point, chi)

    for varpi_i, points in routes["lines"]:
        z = np.asarray([point["z"] for point in points])
        bandwidth = (
            (delta_max if metal else omega_m)
            + float(np.max(np.abs(z.real))))
        rule = evaluator.damped_line_rule(
            varpi_i, bandwidth,
            rel_tol=config.minimax_config.target_error,
            max_order=config.minimax_config.max_nodes)
        t, h = rule["t"], rule["h"]
        if metal:
            # Positive nodes only: the fractional kernel supplies both
            # Keldysh terms itself (never the symmetric ±tau doubling).
            values = compute_chi0_contour_fractional(
                wfns, t, h, z, meta, mesh_xy,
                occupations=occupation_state.f_kn,
                energy_reference=float(occupation_state.mu_ry))
        else:
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
            write_full(point, chi)


def build_mpa_fit(
    run_dir, label, *, wfns, V_q, quad, sym, centroid_indices, wfn=None,
    head_resolver, config, meta, mesh_xy, energy_reference=0.0,
    tile_bytes=None, plan=None, iteration_head_response=None,
    occupation_state=None, print_fn=print,
):
    """Write body/head samples and fits; return path plus iteration head."""
    # The former blanket metal gate (mpa_metal_evaluator_unavailable) is
    # discharged: occupation-weighted chi (fractional contour + finite-q
    # divided difference) and the weighted Sigma branches landed in Wave 1.
    # A metal plan still refuses without an OccupationState — here, before
    # any inode exists, and again at the _evaluate_samples seam.  The
    # driver-level UNIMPLEMENTED_MODES row gates the deck path until the
    # E2E claim.
    _require_metal_occupations(config, occupation_state)
    from common.collectives import barrier, process_rank

    root = os.path.abspath(os.fspath(run_dir))
    if process_rank() == 0:
        os.makedirs(root, exist_ok=True)
    barrier("mpa.model.mkdir", print_fn=print_fn)

    n_p = int(config.mpa.n_poles)
    omega_m = float(quad.x_max)
    plan = make_mpa_plan(config, quad) if plan is None else plan
    z_all = sample_plan.plan_z(plan)
    sample_plan.refuse_unsupported(plan, delta_max=omega_m)
    if iteration_head_response is not None:
        response_z = np.asarray(
            iteration_head_response.omegas[:z_all.size],
            dtype=np.complex128)
        if response_z.shape != z_all.shape or not np.array_equal(
                response_z, z_all):
            raise ValueError(
                "QSGW head and MPA body must use the identical stamped z grid")
        overrides = (
            config.head.vhead,
            config.head.whead_0freq,
            config.head.whead_imfreq,
        )
        if any(value is not None for value in overrides):
            raise ValueError(
                "multipoint QSGW-MPA head samples cannot be combined with "
                "single-frequency vhead/whead overrides")

    q_idx, tables, closure_verdict = _q_wedge(sym, centroid_indices, meta)
    sample_path, fit_path = iteration_artifact_paths(root, label)
    varpi = np.unique(z_all.imag)
    line = np.searchsorted(varpi, z_all.imag).astype(np.int32)
    # The origin shift is stamped ONLY when the deck declared it: it enters
    # mpa_store's `extra` channel as the additive attr
    # ``mpa_prov_metal_origin_shift_ry``, outside _SAMPLING_ORDER and so
    # outside the ω-grid digest.  A deck that leaves the key unset writes
    # the byte-identical store it wrote before the key existed, which is
    # the whole reason this is `extra` and not a sixth sampling field.
    sampling_record = {"protocol": "double_parallel", "varpi": varpi,
                       "n_p": n_p, "alpha": config.mpa.sampling_alpha,
                       "omega_max": omega_m}
    if config.mpa.metal_origin_shift_ry is not None:
        sampling_record["metal_origin_shift_ry"] = float(
            config.mpa.metal_origin_shift_ry)
    common = dict(
        mesh_xy=mesh_xy, n_omega=z_all.size, n_q_on_disk=q_idx.size,
        n_mu=meta.n_rmu, n_rmu_logical=meta.n_rmu, tables=tables,
        closure_verdict=closure_verdict,
        omega=z_all, omega_line=line, energy_unit="Ry",
        sampling=sampling_record)
    mpa_store.allocate_w_omega_collective(
        sample_path, _CHI, mode="w", **common)
    mpa_store.allocate_w_omega_collective(
        sample_path, _WC, mode="a", **common)

    routes = sample_plan.plan_routes(plan)
    metal = config.mpa.material_class != "insulator"
    kminq_rows = _metal_kminq_rows(sym, q_idx) if metal else None
    static_gamma_override = (
        iteration_head_response.static_chi_body_gamma
        if iteration_head_response is not None else None)
    gamma_matches = np.flatnonzero(np.asarray(q_idx, np.int64) == 0)
    gamma_row = int(gamma_matches[0]) if gamma_matches.size == 1 else None

    def _write_full(point, chi):
        _write_sample(
            sample_path, point["index"], chi, q_idx, meta, mesh_xy,
            z_all.size)

    def _write_wedge(point, chi_wedge):
        chi_wedge.block_until_ready()
        mpa_store.write_w_slab_collective(
            sample_path, _CHI, point["index"], chi_wedge, mesh_xy=mesh_xy,
            global_shape=(z_all.size, q_idx.size, meta.n_rmu, meta.n_rmu))

    _evaluate_samples(
        wfns, routes, quad, config, meta, mesh_xy,
        energy_reference=energy_reference,
        occupation_state=occupation_state,
        write_full=_write_full, write_wedge=_write_wedge,
        static_gamma_override=static_gamma_override,
        gamma_row=gamma_row, kminq_rows=kminq_rows)

    V = _to_wedge(V_q, q_idx, mesh_xy)
    iteration_head = _solve_wc(
        sample_path, V, z_all.size, q_idx, meta, mesh_xy,
        config.backend.w_dyson_solver,
        head_response=iteration_head_response,
        wfn=wfn if iteration_head_response is not None else None,
        config=config if iteration_head_response is not None else None,
    )
    _, report = _fit_body(
        sample_path, fit_path, z_all, n_p, tile_bytes, mesh_xy)
    if iteration_head is None:
        head_fit_samples = tuple(
            head_resolver.at(complex(z)) for z in z_all)
        head_model = "dft_direct_loewner"
    else:
        head_fit_samples = iteration_head.samples[:z_all.size]
        head_model = (
            "qsgw_schur_loewner"
            if iteration_head_response.Y_x is not None
            else "qsgw_direct_loewner"
        )
    fit_ledger = mpa_store.fit_completion_ledger(fit_path)
    _fit_head_samples(
        fit_path,
        head_fit_samples,
        z_all,
        n_p,
        fit_ledger["w_grid_hash"],
        mesh_xy,
        model=head_model,
        occupation_state=occupation_state,
    )
    print_fn(fit_driver.format_cost_report(report))
    return fit_path, iteration_head


__all__ = [
    "build_mpa_fit",
    "make_mpa_plan",
    "retain_iteration_artifacts",
    "iteration_artifact_paths",
]
