"""Build one disk-bounded MPA screening model."""

from __future__ import annotations

import os

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from file_io import mpa_store
from gw.mpa import evaluator, fit_driver, sample_plan


_CHI = "chi_qmunu_z"
_CHI1 = "chi1_qmunu_z"
_WC = "Wc_qmunu_z"


def _intraband_enabled(config):
    """Return the declared crossing-block state; missing means legacy off."""
    mpa = None if config is None else getattr(config, "mpa", None)
    enabled = bool(getattr(mpa, "intraband_block", False))
    if enabled and getattr(mpa, "material_class", None) != "metal":
        raise ValueError(
            "GATE mpa_intraband_block_metal_only: mpa_intraband_block is "
            "defined only for mpa_material_class = metal")
    return enabled


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
    z,
    q_idx,
    meta,
    mesh_xy,
    dyson_solver=None,
    *,
    head_response=None,
    head_channel=None,
    sym=None,
    wfn=None,
    config=None,
    print_fn=print,
    distrib_la_batched_route: str = "auto",
):
    """THE DEFAULT ``wc_source``: full W for the head, remainder Wc to fit.

    Signature IS the seam (see :func:`build_mpa_fit`'s ``wc_source``): the
    sample store, the wedge V, the z-list, the wedge q index and the mesh.
    ``z`` is carried rather than a bare count because a source that does
    not solve a Dyson equation — the ``screening_diagrams = w_bse`` ladder
    — needs the frequencies themselves, and passing a length would have
    made it derive them from a second copy of the sample plan.
    The keyword-only state belongs to the expanded metallic/QSGW path:
    it resolves the dynamic head from these same W samples and preserves
    the one-occupation-state-per-iteration contract.  A ladder source is
    reachable only for a single-shot insulator, so it implements the common
    positional seam and never silently consumes this RPA-only head state.
    """
    from gw.qsgw_head import (
        IterationHeadSamples,
        finalize_iteration_head_sample,
    )
    from gw.w_isdf import solve_w

    intraband = _intraband_enabled(config)

    raw_z = np.asarray(z)
    if raw_z.ndim == 0 and np.issubdtype(raw_z.dtype, np.integer):
        # Backward-compatible direct-call shape used by the disk-pipeline
        # unit: RPA Wc itself needs only the count.  Production passes the
        # stamped complex grid because the ladder source and the metallic
        # finite-q0 head both need the actual abscissae.
        n_z = int(raw_z)
        z_values = None
    else:
        z_values = np.asarray(z, dtype=np.complex128).reshape(-1)
        n_z = int(z_values.size)
    bgw_q0 = None
    bgw_vhead = None
    bgw_epsinv = []
    if config is not None and bool(config.head.uses_bgw_metal_q0shift):
        from gw.head_correction import (
            bgw_q0shift_vhead,
            resolve_bgw_q0_channel,
        )
        bgw_q0 = resolve_bgw_q0_channel(
            config, sym, q_idx,
            head_channel, kgrid=meta.kgrid)
        bgw_vhead = bgw_q0shift_vhead(wfn, meta)
        print_fn(
            "  [bgw q0 provenance] finite epsilon q0="
            f"{bgw_q0.q0_reduced}, full row={bgw_q0.requested_full_index}, "
            f"wedge row={bgw_q0.wedge_row} (representative full row "
            f"{bgw_q0.representative_full_index}); analytic-sphere "
            f"v_head={bgw_vhead.real:.12e} raw, "
            f"{bgw_vhead.real / (float(meta.nk_tot) * float(meta.cell_volume)):.12e} "
            "after 1/(Nk*Omega).")
        if head_response is None and z_values is None:
            raise ValueError(
                "MPA bgw_q0shift requires the stamped complex z grid, not "
                "only its length")

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
        chi1 = None
        chi_remainder = None
        if intraband:
            chi1, _ = mpa_store.read_w_slab_collective(
                sample_path, _CHI1, index, mesh_xy=mesh_xy)
            gamma_matches = np.flatnonzero(
                np.asarray(q_idx, dtype=np.int64) == 0)
            if gamma_matches.size != 1:
                raise ValueError(
                    "GATE intraband_gamma_empty: an enabled intraband block "
                    "requires exactly one Gamma row in the stored q wedge")
            gamma_abs = float(jnp.max(jnp.abs(
                chi1[int(gamma_matches[0])])).block_until_ready())
            if gamma_abs != 0.0:
                raise ValueError(
                    "GATE intraband_gamma_empty: chi1(Gamma) must be "
                    "identically zero at every sample; "
                    f"index={index}, max_abs={gamma_abs:.16e}")
            chi_remainder = chi - chi1
        chi_q0 = None
        if bgw_q0 is not None:
            # ``solve_w`` donates the full chi buffer.  Retain only one
            # distributed (mu,nu) row for the finite-q epsilon scalar; no
            # process materialises an N_mu^2 object.
            q0row = int(bgw_q0.wedge_row)
            chi_q0 = chi[q0row:q0row + 1].copy()
        W = solve_w(
            V, chi, meta, mesh_xy, dyson_solver=dyson_solver,
            distrib_la_batched_route=distrib_la_batched_route)
        W.block_until_ready()
        W0 = None
        if intraband:
            W0 = solve_w(
                V, chi_remainder, meta, mesh_xy,
                dyson_solver=dyson_solver,
                distrib_la_batched_route=distrib_la_batched_route)
            W0.block_until_ready()
        if bgw_q0 is not None:
            from gw.head_correction import (
                bgw_q0shift_head_sample,
                finite_q0_epsinv_head,
            )
            from gw.w_isdf import _w_solve_pref_scalar
            epsinv = finite_q0_epsinv_head(
                chi_q0,
                W[q0row:q0row + 1],
                bgw_q0.g_head,
                bgw_q0.v_bare,
                _w_solve_pref_scalar(meta),
                mesh_xy=mesh_xy,
            )
            eps_value = complex(np.asarray(epsinv)[0])
            bgw_epsinv.append(eps_value)
            omega = (
                complex(head_response.omegas[index])
                if head_response is not None
                else complex(z_values[index]))
            head_samples.append(
                bgw_q0shift_head_sample(bgw_vhead, eps_value, omega))
            print_fn(
                f"  [bgw q0] sample {index:02d}: z={omega!r}, "
                f"epsinv_00(head+wings)={eps_value.real:.12e}"
                + (f"{eps_value.imag:+.12e}j" if eps_value.imag else ""))
            del chi_q0, epsinv
        elif head_response is not None:
            head_samples.append(finalize_iteration_head_sample(
                head_response,
                index,
                None if gamma is None else W[gamma],
                wfn=wfn,
                meta=meta,
                config=config,
                mesh=mesh_xy,
            ))
        # The full W above remains the sole head input.  Only the body fit
        # target changes: with the crossing block enabled it is the exact
        # double-Dyson remainder W0-V at every stamped sample.
        Wc = (W0 if intraband else W) - V
        Wc.block_until_ready()
        mpa_store.write_w_slab_collective(
            sample_path, _WC, index, Wc, mesh_xy=mesh_xy,
            global_shape=shape)
        del chi, W, Wc
        if intraband:
            del chi1, chi_remainder, W0

    if head_response is None:
        return tuple(head_samples) if bgw_q0 is not None else None
    for index in range(int(n_z), len(head_response.omegas)):
        if bgw_q0 is not None:
            # Metallic MPA appends only the exact-static do_G0 sample after
            # the fit grid.  The near-line origin row was evaluated with the
            # exact divided-difference chi, so its finite-q epsinv is the
            # identical static response and is reused here.
            from gw.head_correction import bgw_q0shift_head_sample
            head_samples.append(bgw_q0shift_head_sample(
                bgw_vhead, bgw_epsinv[0], head_response.omegas[index]))
        else:
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


def _fit_body(sample_path, fit_path, z, n_p, tile_bytes, mesh_xy,
              provenance=None, occupation_state=None, n_extra_poles=0,
              finalize=True):
    return fit_driver.run_fit_driver(
        sample_path, _WC, fit_path, z, n_p, mesh_xy=mesh_xy,
        tile_bytes=tile_bytes, provenance=provenance,
        occupation_state=occupation_state,
        n_extra_poles=n_extra_poles,
        extra_pole_model=(
            "intraband_eigenmode_v1" if n_extra_poles else None),
        finalize=finalize)


def _relative_frobenius(model, exact):
    numerator = jnp.real(jnp.vdot(model - exact, model - exact))
    denominator = jnp.real(jnp.vdot(exact, exact))
    value = jnp.sqrt(numerator / jnp.maximum(
        denominator, np.finfo(np.float64).tiny))
    return float(np.asarray(value.block_until_ready()))


def _build_intraband_rows(sample_path, V, z, intraband_blocks, mesh_xy):
    """Build one frozen-static resolvent-moment block at a time."""
    from gw.mpa import intraband_block

    Wc0, _ = mpa_store.read_w_slab_collective(
        sample_path, _WC, 0, mesh_xy=mesh_xy)
    W0bar = Wc0 + V
    z_block = np.asarray(z, dtype=np.complex128).copy()
    z_block[0] = 0.0j
    rows = []
    for iq, pair_block in enumerate(intraband_blocks):
        if int(pair_block[0].shape[0]) == 0:
            rows.append(None)
            continue
        row = intraband_block.build_row(W0bar[iq], pair_block, z_block)
        rows.append(row)
    del Wc0, W0bar
    n_poles = max(
        (row.n_poles for row in rows if row is not None), default=0)
    if n_poles < 1:
        raise ValueError(
            "GATE intraband_finite_q_empty: every crossing-block row is "
            "empty; Gamma must be empty but finite q must contribute")
    return rows, int(n_poles)


def _padded_intraband_arrays(rows, n_poles, n_mu, mesh_xy):
    """Return row slabs plus the stacked wedge field used by certificates."""
    from gw.mpa import intraband_block

    pole_shard = NamedSharding(mesh_xy, P(None, "x", "y"))
    slabs = []
    for row in rows:
        if row is None:
            Om = jnp.ones((n_poles, n_mu, n_mu), dtype=jnp.complex128)
            Bp = jnp.zeros((n_poles, n_mu, n_mu), dtype=jnp.complex128)
            Om = jax.lax.with_sharding_constraint(Om, pole_shard)
            Bp = jax.lax.with_sharding_constraint(Bp, pole_shard)
        else:
            Om, Bp = intraband_block.pad_row(row, n_poles)
        slabs.append((Om, Bp))
    field_shard = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    Omega = jax.lax.with_sharding_constraint(
        jnp.stack([value[0] for value in slabs], axis=1), field_shard)
    B = jax.lax.with_sharding_constraint(
        jnp.stack([value[1] for value in slabs], axis=1), field_shard)
    return slabs, Omega, B


def _append_intraband_rows(fit_path, rows, slabs, mesh_xy):
    """Append every q row and leave the store unfinalized for certification."""
    for iq, (row, (Omega, Bp)) in enumerate(zip(rows, slabs)):
        live = jnp.abs(Bp) != 0.0
        nonfinite = jnp.count_nonzero(
            ~jnp.isfinite(Omega) | ~jnp.isfinite(Bp))
        bad = jnp.count_nonzero(
            live & ((jnp.real(Omega) <= 0.0) | (jnp.imag(Omega) > 0.0)))
        finite = int(np.asarray(nonfinite.block_until_ready())) == 0
        causal = int(np.asarray(bad.block_until_ready())) == 0
        anomaly = {} if row is None else {
            "folded_modes": row.folded_modes,
            "dropped_modes": row.dropped_modes,
            "folded_elements": row.folded_elements,
            "dropped_elements": row.dropped_elements,
        }
        scalars = {} if row is None else {
            "zero_mode_weight": row.zero_mode_weight,
            "zero_mode_pole_shift": row.zero_mode_pole_shift,
            "zero_mode_cluster": float(row.zero_mode_cluster),
        }
        mpa_store.write_intraband_row_collective(
            fit_path, iq, Omega[:, None, :, :], Bp[:, None, :, :],
            mesh_xy=mesh_xy, poles_finite=finite, poles_causal=causal,
            anomaly_counts=anomaly, row_scalars=scalars)


def _evaluate_fit_prefix(fit_path, n_p_fit, z_values, mesh_xy):
    """Evaluate a staged Loewner prefix without ever reading its suffix."""
    values = None
    with mpa_store.open_pole_reader(
            fit_path, mesh_xy=mesh_xy, allow_partial=True) as reader:
        for lo in range(0, int(n_p_fit), 4):
            hi = min(lo + 4, int(n_p_fit))
            Omega, Bp = reader.read(
                slice(lo, hi), unfold=False, return_sharded=True,
                to_unit="Ry")
            batch = [jnp.sum(
                2.0 * Omega * Bp
                / (complex(z) ** 2 - Omega * Omega), axis=0)
                for z in z_values]
            if values is None:
                values = batch
            else:
                values = [old + add for old, add in zip(values, batch)]
            del Omega, Bp
    return tuple(values)


def _two_shortest_finite_q_rows(q_idx, kgrid):
    """Indices of the two shortest non-Gamma irreducible star rows."""
    grid = np.asarray(kgrid, dtype=np.int64)
    rows = []
    for row, flat in enumerate(np.asarray(q_idx, dtype=np.int64)):
        coord = np.asarray(np.unravel_index(int(flat), tuple(grid)))
        wrapped = np.minimum(coord, grid - coord) / grid
        norm = float(np.linalg.norm(wrapped))
        if norm > 0.0:
            rows.append((norm, int(flat), row))
    if len(rows) < 2:
        raise ValueError(
            "GATE intraband_gap_stars: fewer than two finite-q wedge rows")
    return np.asarray([item[2] for item in sorted(rows)[:2]], np.int32)


def _combined_relative(models, exacts):
    numerator = 0.0
    denominator = 0.0
    for model, exact in zip(models, exacts):
        numerator = numerator + jnp.real(jnp.vdot(model - exact,
                                                   model - exact))
        denominator = denominator + jnp.real(jnp.vdot(exact, exact))
    value = jnp.sqrt(numerator / jnp.maximum(
        denominator, np.finfo(np.float64).tiny))
    return float(np.asarray(value.block_until_ready()))


def _maximum_relative(models, exacts):
    """Worst Frobenius-relative member of a stamped observable family."""
    return max(
        (_relative_frobenius(model, exact)
         for model, exact in zip(models, exacts)),
        default=0.0,
    )


def _certify_intraband_support(
    sample_path, fit_path, V, z, Omega_intra, B_intra, meta, mesh_xy,
    *, dyson_solver, distrib_la_batched_route,
):
    """Measure exact-DeltaW and total-model errors on all stamped samples."""
    from gw.mpa.intraband_block import evaluate_pole_sum
    from gw.w_isdf import solve_w

    prefix = _evaluate_fit_prefix(
        fit_path, mpa_store.fit_completion_ledger(fit_path)["n_p_fit"],
        z, mesh_xy)
    exact_delta, model_delta = [], []
    exact_total, model_total, prefix_exact = [], [], []
    for index, z_value in enumerate(z):
        chi, _ = mpa_store.read_w_slab_collective(
            sample_path, _CHI, index, mesh_xy=mesh_xy)
        remainder, _ = mpa_store.read_w_slab_collective(
            sample_path, _WC, index, mesh_xy=mesh_xy)
        W = solve_w(
            V, chi, meta, mesh_xy, dyson_solver=dyson_solver,
            distrib_la_batched_route=distrib_la_batched_route)
        full_wc = W - V
        block_z = 0.0j if index == 0 else z_value
        block = evaluate_pole_sum(Omega_intra, B_intra, block_z)
        exact_delta.append(full_wc - remainder)
        model_delta.append(block)
        exact_total.append(full_wc)
        model_total.append(prefix[index] + block)
        prefix_exact.append(remainder)
        del chi, remainder, W
    block_error = _maximum_relative(model_delta, exact_delta)
    total_error = _maximum_relative(model_total, exact_total)
    prefix_error = _maximum_relative(prefix, prefix_exact)
    static_error = _relative_frobenius(model_delta[0], exact_delta[0])
    return {
        "block_sample_frobenius_relative": block_error,
        "total_sample_frobenius_relative": total_error,
        "remainder_fit_frobenius_relative": prefix_error,
        "static_frobenius_relative": static_error,
        "block_sample_combined_frobenius_relative": _combined_relative(
            model_delta, exact_delta),
        "total_sample_combined_frobenius_relative": _combined_relative(
            model_total, exact_total),
        "remainder_fit_combined_frobenius_relative": _combined_relative(
            prefix, prefix_exact),
    }


def _evaluate_gap_chi(
    wfns, occupation_state, config, meta, mesh_xy,
):
    """The design's five held-out points on the 0.2-Ry line."""
    from gw.mpa import evaluator
    from gw.w_isdf import (
        compute_chi0_contour_fractional,
        occupation_support_bandwidth,
    )

    z_gap = np.asarray(
        [0.04 + 0.2j, 0.08 + 0.2j, 0.15 + 0.2j,
         0.30 + 0.2j, 0.60 + 0.2j], dtype=np.complex128)
    occ_window = float(config.mpa.occupation_window_threshold)
    delta_max = occupation_support_bandwidth(
        wfns.enk, occupation_state.f_kn,
        occupation_window_threshold=occ_window)
    rule = evaluator.damped_line_rule(
        0.2, delta_max + float(np.max(z_gap.real)),
        rel_tol=config.minimax_config.target_error,
        max_order=config.minimax_config.max_nodes)
    values = compute_chi0_contour_fractional(
        wfns, rule["t"], rule["h"], z_gap, meta, mesh_xy,
        occupations=occupation_state.f_kn,
        energy_reference=float(occupation_state.mu_ry),
        occupation_window_threshold=occ_window)
    return z_gap, tuple(values)


def _certify_intraband_gap(
    fit_path, V, z_gap, chi_gap, q_idx, Omega_intra, B_intra,
    meta, mesh_xy, *, dyson_solver, distrib_la_batched_route,
):
    """Held-out total-W observable on the two shortest finite-q stars."""
    from gw.mpa.intraband_block import evaluate_pole_sum
    from gw.w_isdf import solve_w

    rows = _two_shortest_finite_q_rows(q_idx, meta.kgrid)
    n_fit = mpa_store.fit_completion_ledger(fit_path)["n_p_fit"]
    prefix = _evaluate_fit_prefix(fit_path, n_fit, z_gap, mesh_xy)
    exacts, fit_only, with_block = [], [], []
    for index, (z_value, chi_full) in enumerate(zip(z_gap, chi_gap)):
        chi = _to_wedge(chi_full, q_idx, mesh_xy)
        W = solve_w(
            V, chi, meta, mesh_xy, dyson_solver=dyson_solver,
            distrib_la_batched_route=distrib_la_batched_route)
        exact = (W - V)[rows]
        block = evaluate_pole_sum(Omega_intra, B_intra, z_value)
        exacts.append(exact)
        fit_only.append(prefix[index][rows])
        with_block.append((prefix[index] + block)[rows])
        del chi, W
    return {
        "q_rows": rows,
        "fit_only_frobenius_relative": _maximum_relative(
            fit_only, exacts),
        "with_block_frobenius_relative": _maximum_relative(
            with_block, exacts),
        "fit_only_combined_frobenius_relative": _combined_relative(
            fit_only, exacts),
        "with_block_combined_frobenius_relative": _combined_relative(
            with_block, exacts),
    }


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
    intraband_blocks=None, write_intraband=None,
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
        intraband_chi1,
        occupation_support_bandwidth,
    )

    metal = config.mpa.material_class != "insulator"
    _require_metal_occupations(config, occupation_state)
    omega_m = float(quad.x_max)
    # ONE occupancy window for the whole fit: the rule bandwidth below and
    # every fractional-chi call in this function read the same deck value, so
    # the damped-line rule can never be sized for transitions the band slices
    # no longer contain.  Same key, same default and same predicate as the
    # Sigma planner's band window (gw.efermi.occupation_weight_floor).
    occ_window = float(config.mpa.occupation_window_threshold)
    if metal:
        delta_max = occupation_support_bandwidth(
            wfns.enk, occupation_state.f_kn,
            occupation_window_threshold=occ_window)

    if (intraband_blocks is None) != (write_intraband is None):
        raise ValueError(
            "intraband sample plumbing requires both intraband_blocks and "
            "write_intraband, or neither")

    def _emit(point, chi, *, wedge):
        (write_wedge if wedge else write_full)(point, chi)
        if intraband_blocks is None:
            return
        # The shifted near_00 grid coordinate stores the exact static
        # divided-difference value.  Subtract the matching exact chi1(0),
        # not chi1(i*shift), so the split is algebraically exact.
        z_block = 0.0j if point["role"] == "near_00" else point["z"]
        chi1_rows = [
            intraband_chi1(block, z_block) for block in intraband_blocks
        ]
        write_intraband(point, jnp.stack(chi1_rows, axis=0))

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
            _emit(point, chi, wedge=False)
        elif point["role"].startswith("near"):
            # The shifted-origin slot stores the exact static value; the
            # fit reads sample_z = i*varpi_1, and the inconsistency is
            # bounded by (varpi_1/(q*v_F))^2 at finite q (W1.a-2).
            chi_w = compute_chi0_static_fractional(
                wfns, meta, mesh_xy, occupation_state=occupation_state,
                kminq_rows=kminq_rows)
            if static_gamma_override is not None and gamma_row is not None:
                chi_w = chi_w.at[gamma_row].set(static_gamma_override[0])
            _emit(point, chi_w, wedge=True)
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
                energy_reference=float(occupation_state.mu_ry),
                occupation_window_threshold=occ_window)
            _emit(point, chi, wedge=False)

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
                energy_reference=float(occupation_state.mu_ry),
                occupation_window_threshold=occ_window)
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
            _emit(point, chi, wedge=False)


def build_mpa_fit(
    run_dir, label, *, wfns, V_q, quad, sym, centroid_indices, wfn=None,
    head_resolver, config, meta, mesh_xy, energy_reference=0.0,
    tile_bytes=None, plan=None, iteration_head_response=None,
    occupation_state=None, head_channel=None, wc_source=None, print_fn=print,
):
    """Write body/head samples and fits; return path plus iteration head.

    ``wc_source`` is the ONE seam ``screening_diagrams`` moves.  ``None``
    (the default) is :func:`_solve_wc` — Wc(z) from the Dyson solve of the
    sampled chi, the RPA path, byte-for-byte what this function has always
    done.  Under ``w_bse`` the caller passes
    ``gw.screening_bse.make_ladder_wc_source(...)``, which writes the same
    per-z, per-wedge-q ``Wc`` slabs into the same store from the ladder
    resolvent instead.  Everything downstream of the seam — the sample
    plan, the Pade fit, the SlabIO store lifecycle, the head fit and the
    Sigma consumer — is the same code on both branches; the ONLY
    difference the store records is the provenance stamp below.
    """
    # The former blanket metal gate (mpa_metal_evaluator_unavailable) is
    # discharged: occupation-weighted chi (fractional contour + finite-q
    # divided difference) and the weighted Sigma branches landed in Wave 1.
    # A metal plan still refuses without an OccupationState — here, before
    # any inode exists, and again at the _evaluate_samples seam — and that
    # refusal is now the only gate on the deck path: the driver-level
    # UNIMPLEMENTED_MODES row was deleted when the metal pipeline ran E2E.
    _require_metal_occupations(config, occupation_state)
    if wc_source is not None and config.mpa.material_class != "insulator":
        raise ValueError(
            "GATE w_bse_insulators_only: an alternate ladder wc_source "
            "requires mpa_material_class = insulator; keep the default "
            "RPA source for a metal.")
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
    metal = config.mpa.material_class != "insulator"
    kminq_rows = _metal_kminq_rows(sym, q_idx) if metal else None
    gamma_matches = np.flatnonzero(np.asarray(q_idx, np.int64) == 0)
    gamma_row = int(gamma_matches[0]) if gamma_matches.size == 1 else None
    intraband_blocks = None
    if _intraband_enabled(config):
        from gw.w_isdf import intraband_pair_block

        intraband_blocks = [
            intraband_pair_block(
                wfns, meta, occupation_state, kminq_rows, q_row)
            for q_row in range(int(q_idx.size))
        ]
        if gamma_row is None or int(intraband_blocks[gamma_row][0].size) != 0:
            count = -1 if gamma_row is None else int(
                intraband_blocks[gamma_row][0].size)
            raise ValueError(
                "GATE intraband_gamma_empty: the exact crossing selection "
                "must be empty at Gamma; selected pair count=" + str(count))
    sample_path, fit_path = iteration_artifact_paths(root, label)
    varpi = np.unique(z_all.imag)
    line = np.searchsorted(varpi, z_all.imag).astype(np.int32)
    # PROVENANCE (QUALITY_PATTERNS #10).  RPA poles and ladder poles have
    # identical shapes, identical certification and identical
    # plausibility; only the stamp distinguishes them, and the Sigma
    # consumer asserts it at load (mpa_store.validate_fit_store).
    diagrams = str(getattr(
        getattr(config.screening, "diagrams", "w_rpa"), "value",
        getattr(config.screening, "diagrams", "w_rpa")))
    provenance = {"screening_diagrams": diagrams}
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
        sampling=sampling_record, provenance=provenance)
    mpa_store.allocate_w_omega_collective(
        sample_path, _CHI, mode="w", **common)
    mpa_store.allocate_w_omega_collective(
        sample_path, _WC, mode="a", **common)
    if intraband_blocks is not None:
        mpa_store.allocate_w_omega_collective(
            sample_path, _CHI1, mode="a", **common)

    routes = sample_plan.plan_routes(plan)
    static_gamma_override = (
        iteration_head_response.static_chi_body_gamma
        if iteration_head_response is not None else None)

    def _write_full(point, chi):
        _write_sample(
            sample_path, point["index"], chi, q_idx, meta, mesh_xy,
            z_all.size)

    def _write_wedge(point, chi_wedge):
        chi_wedge.block_until_ready()
        mpa_store.write_w_slab_collective(
            sample_path, _CHI, point["index"], chi_wedge, mesh_xy=mesh_xy,
            global_shape=(z_all.size, q_idx.size, meta.n_rmu, meta.n_rmu))

    def _write_intraband(point, chi1_wedge):
        chi1_wedge.block_until_ready()
        mpa_store.write_w_slab_collective(
            sample_path, _CHI1, point["index"], chi1_wedge,
            mesh_xy=mesh_xy,
            global_shape=(z_all.size, q_idx.size,
                          meta.n_rmu, meta.n_rmu))

    _evaluate_samples(
        wfns, routes, quad, config, meta, mesh_xy,
        energy_reference=energy_reference,
        occupation_state=occupation_state,
        write_full=_write_full, write_wedge=_write_wedge,
        static_gamma_override=static_gamma_override,
        gamma_row=gamma_row, kminq_rows=kminq_rows,
        intraband_blocks=intraband_blocks,
        write_intraband=(
            _write_intraband if intraband_blocks is not None else None))

    V = _to_wedge(V_q, q_idx, mesh_xy)
    if wc_source is None:
        iteration_head = _solve_wc(
            sample_path, V, z_all, q_idx, meta, mesh_xy,
            config.backend.w_dyson_solver,
            head_response=iteration_head_response,
            head_channel=head_channel,
            sym=sym,
            wfn=wfn,
            config=config,
            print_fn=print_fn,
            distrib_la_batched_route=getattr(
                config.backend, "distrib_la_batched_route", "auto"),
        )
    else:
        if iteration_head_response is not None:
            raise ValueError(
                "GATE w_bse_self_consistency_unimplemented: a ladder "
                "wc_source cannot consume per-iteration QSGW head state; "
                "use screening_diagrams = w_rpa for self-consistency.")
        iteration_head = wc_source(
            sample_path, V, z_all, q_idx, meta, mesh_xy,
            config.backend.w_dyson_solver)
    if intraband_blocks is None:
        _, report = _fit_body(
            sample_path, fit_path, z_all, n_p, tile_bytes, mesh_xy,
            provenance=provenance, occupation_state=occupation_state)
    else:
        from common.collectives import barrier, process_rank
        from gw.mpa import intraband_block

        rows, n_p_intra = _build_intraband_rows(
            sample_path, V, z_all, intraband_blocks, mesh_xy)
        slabs, Omega_intra, B_intra = _padded_intraband_arrays(
            rows, n_p_intra, meta.n_rmu, mesh_xy)
        _, report = _fit_body(
            sample_path, fit_path, z_all, n_p, tile_bytes, mesh_xy,
            provenance=provenance, occupation_state=occupation_state,
            n_extra_poles=n_p_intra, finalize=False)
        _append_intraband_rows(fit_path, rows, slabs, mesh_xy)

        dyson = config.backend.w_dyson_solver
        distrib_route = getattr(
            config.backend, "distrib_la_batched_route", "auto")
        support_cert = _certify_intraband_support(
            sample_path, fit_path, V, z_all, Omega_intra, B_intra,
            meta, mesh_xy, dyson_solver=dyson,
            distrib_la_batched_route=distrib_route)
        z_gap, chi_gap = _evaluate_gap_chi(
            wfns, occupation_state, config, meta, mesh_xy)
        gap_cert = _certify_intraband_gap(
            fit_path, V, z_gap, chi_gap, q_idx, Omega_intra, B_intra,
            meta, mesh_xy, dyson_solver=dyson,
            distrib_la_batched_route=distrib_route)

        gap_allowed = max(
            np.finfo(np.float64).eps,
            min(4.0e-3,
                3.0 * support_cert["total_sample_frobenius_relative"]))
        certification = dict(report["certification"])
        certification.update({
            "intraband_sample_max_rel_error":
                support_cert["block_sample_frobenius_relative"],
            "intraband_sample_max_rel_error_max_allowed":
                intraband_block.SAMPLE_REL_TOL,
            "intraband_static_max_rel_error":
                support_cert["static_frobenius_relative"],
            "intraband_static_max_rel_error_max_allowed":
                intraband_block.STATIC_REL_TOL,
            "intraband_gap_max_rel_error":
                gap_cert["with_block_frobenius_relative"],
            "intraband_gap_max_rel_error_max_allowed": gap_allowed,
            "intraband_support_total_frobenius_relative":
                support_cert["total_sample_frobenius_relative"],
            "intraband_support_remainder_frobenius_relative":
                support_cert["remainder_fit_frobenius_relative"],
            "intraband_gap_fit_only_frobenius_relative":
                gap_cert["fit_only_frobenius_relative"],
            "intraband_gap_q_rows": ",".join(
                map(str, np.asarray(gap_cert["q_rows"]).tolist())),
            "intraband_builder_max_compression_rel_error": max(
                (row.sample_max_rel_error for row in rows
                 if row is not None), default=0.0),
            "intraband_builder_max_cluster_width_ry": max(
                (row.cluster_width_max_ry for row in rows
                 if row is not None), default=0.0),
            "intraband_zero_mode_weight_max": max(
                (row.zero_mode_weight for row in rows
                 if row is not None), default=0.0),
            "intraband_zero_mode_pole_shift_max": max(
                (row.zero_mode_pole_shift for row in rows
                 if row is not None), default=0.0),
        })
        if process_rank() == 0:
            mpa_store.finalize_fit_store(
                fit_path, certification=certification)
        barrier("mpa_fit_with_intraband_finalized")
        ledger = mpa_store.fit_completion_ledger_collective(
            fit_path, key="intraband-finalized")
        report["certification"] = certification
        report["n_p_intraband"] = n_p_intra
        report["n_p_total"] = ledger["n_p"]
        report["intraband_support"] = support_cert
        report["intraband_gap"] = gap_cert
        print_fn(
            "  MPA intraband block: "
            f"M_p={n_p_intra}, support="
            f"{support_cert['block_sample_frobenius_relative']:.3e}, "
            f"static={support_cert['static_frobenius_relative']:.3e}, "
            f"gap={gap_cert['with_block_frobenius_relative']:.3e} "
            f"(allowed {gap_allowed:.3e})")
    # ``_fit_body`` publishes through parallel HDF5 while the scalar-head
    # writer opens the same file through serial h5py.  A context-manager
    # return is rank-local: without this process barrier rank 0 can enter
    # h5py while a slower peer still owns the FFI writer, leaving HDF5's
    # superblock write-consistency flag set and refusing the ledger open.
    from common.collectives import barrier
    barrier("mpa_body_fit_closed_before_head")
    if isinstance(iteration_head, tuple):
        head_fit_samples = iteration_head
        iteration_head = None
        head_model = "bgw_q0shift_loewner"
    elif iteration_head is None:
        head_fit_samples = tuple(
            head_resolver.at(complex(z)) for z in z_all)
        head_model = "dft_direct_loewner"
    else:
        head_fit_samples = iteration_head.samples[:z_all.size]
        if bool(getattr(
                config.head, "uses_bgw_metal_q0shift", False)):
            head_model = "bgw_q0shift_loewner"
        else:
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
