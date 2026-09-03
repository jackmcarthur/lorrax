"""Build one disk-bounded MPA screening model."""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from file_io import mpa_store
from gw.mpa import evaluator, fit_driver, sample_plan


_CHI = "chi_qmunu_z"
_CHI_REFLECTED = "chi_qmunu_minus_conj_z"
_WC = "Wc_qmunu_z"
_WC_NEGATIVE = "Wc_qmunu_minus_z"


def make_mpa_plan(config, quad, *, material_class):
    """Build and validate the one frequency plan shared by body and head."""
    n_p = int(config.mpa.n_poles)
    omega_m = float(quad.x_max)
    plan = sample_plan.mpa_plan(
        n_p, omega_m, material_class=material_class,
        alpha=config.mpa.sampling_alpha,
        schedule=config.mpa.sampling_schedule,
        varpi_near=config.mpa.varpi_near_ry,
        varpi_far=config.mpa.varpi_far_ry,
        origin_shift=config.mpa.metal_origin_shift_ry, energy_unit="Ry")
    sample_plan.refuse_unsupported(plan, delta_max=omega_m)
    return plan


def make_mpa_plan_from_fit(config, fit_path, *, mesh_xy, material_class):
    """Rebuild and verify the frequency plan stored by a certified fit.

    A reused fit has no live screening quadrature.  Its scalar-head sample
    vector is the frequency provenance instead: frequencies are read in Ry,
    its largest real part supplies the original ``omega_m``, and the normal
    plan constructor supplies every remaining deck-owned knob.  Exact grid
    equality is required before the plan can be used by an SC head or body.

    Parameters
    ----------
    config : LorraxConfig
        Runtime configuration whose MPA sampling knobs must match the fit.
    fit_path : path-like
        Complete certified MPA body and scalar-head fit store.
    mesh_xy : jax.sharding.Mesh
        Process mesh used by the collective fit reader.
    """
    head = mpa_store.read_head_fit_collective(
        fit_path, mesh_xy=mesh_xy, to_unit="Ry")
    stored_z = np.asarray(head["sample_z"], dtype=np.complex128).reshape(-1)
    if stored_z.size == 0 or not np.all(np.isfinite(stored_z)):
        raise ValueError(
            "MPA certified-fit reuse requires a finite, nonempty stored "
            f"head sample_z vector; got shape={stored_z.shape} in {fit_path!s}")

    class _StoredFrequencyCeiling:
        x_max = float(np.max(stored_z.real))

    plan = make_mpa_plan(
        config, _StoredFrequencyCeiling(), material_class=material_class)
    planned_z = np.asarray(sample_plan.plan_z(plan), dtype=np.complex128)
    if planned_z.shape != stored_z.shape or not np.array_equal(
            planned_z, stored_z):
        max_delta = (
            float(np.max(np.abs(planned_z - stored_z)))
            if planned_z.shape == stored_z.shape else float("inf"))
        raise ValueError(
            "MPA certified-fit reuse provenance mismatch: rebuilding the "
            "deck's frequency plan does not exactly reproduce stored "
            f"sample_z in {fit_path!s}; planned_shape={planned_z.shape}, "
            f"stored_shape={stored_z.shape}, max_abs_delta={max_delta:.17g}. "
            "Use the sampling knobs that created this fit or rebuild it.")
    return plan


def validate_reused_mpa_fit(
    fit_path, *, config, live_plan, sym, centroid_indices, meta, mesh_xy,
    occupation_state, material_class, print_fn=print,
):
    """Certify an explicit read-only one-shot MPA fit against this run.

    This is the production reuse seam.  It validates the finalized store,
    diagram provenance, current q wedge and centroid identity, exact sampling
    grid, pole count, ordered-residue convention, and metallic occupations.
    No sample or fit inode is opened writable on this path.
    """
    path = os.path.abspath(os.fspath(fit_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"mpa_fit_reuse_file does not name a file: {path}")

    q_idx, tables, closure_verdict = _q_wedge(
        sym, centroid_indices, meta)
    # ``allocate_w_omega_collective`` writes the device-independent logical
    # q-table contract: its writer strips the product padding after asserting
    # that the permutation tail is identity and the wrap tail is zero.  Hash
    # that same logical object here.  Hashing the live table directly makes a
    # P=36 run with 2070 centroids compare its 2088-row carrier digest against
    # the fit's 2070-row on-disk digest, refusing a byte-identical physical
    # table solely because the reader's device count added a pad.
    logical_tables = tables.logical(int(meta.n_rmu))
    ledger = mpa_store.validate_fit_store(
        path,
        expected_identity={
            "w_table_hash": logical_tables.canonical().digest(),
            "w_centroid_hash": closure_verdict.centroid_hash,
        },
        expected_screening_diagrams=config.screening.diagrams,
    )
    expected_ordered = bool(
        material_class == "insulator" and not bool(sym.trs_allowed))
    extents = {
        "n_p": (int(ledger["n_p"]), int(config.mpa.n_poles)),
        "n_q": (int(ledger["n_q"]), int(q_idx.size)),
        "n_mu": (int(ledger["n_mu"]), int(meta.n_rmu)),
        "ordered_residues": (
            bool(ledger["ordered_residues"]), expected_ordered),
    }
    mismatched = {
        key: values for key, values in extents.items()
        if values[0] != values[1]
    }
    if mismatched:
        detail = ", ".join(
            f"{key}: store={got!r}, run={want!r}"
            for key, (got, want) in mismatched.items())
        raise ValueError(
            "MPA certified-fit reuse extent mismatch: " + detail)

    stored_plan = make_mpa_plan_from_fit(
        config, path, mesh_xy=mesh_xy, material_class=material_class)
    stored_z = np.asarray(sample_plan.plan_z(stored_plan), np.complex128)
    live_z = np.asarray(sample_plan.plan_z(live_plan), np.complex128)
    if stored_z.shape != live_z.shape or not np.array_equal(stored_z, live_z):
        delta = (float(np.max(np.abs(stored_z - live_z)))
                 if stored_z.shape == live_z.shape else float("inf"))
        raise ValueError(
            "MPA certified-fit reuse frequency mismatch against the live "
            f"band/quadrature span: stored_shape={stored_z.shape}, "
            f"live_shape={live_z.shape}, max_abs_delta={delta:.17g}")
    if material_class == "metal":
        mpa_store.assert_occupation_stamps(
            path, occupation_state, where="mpa_fit_reuse_file")
    print_fn(
        "  MPA screening reuse: certified finalized fit opened read-only: "
        f"{path} (n_q={ledger['n_q']}, n_mu={ledger['n_mu']}, "
        f"n_p={ledger['n_p']})")
    return path


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


def _write_sample(path, index, value, q_idx, meta, mesh_xy, n_z, *, name=_CHI):
    value = _to_wedge(value, q_idx, mesh_xy)
    value.block_until_ready()
    mpa_store.write_w_slab_collective(
        path, name, index, value, mesh_xy=mesh_xy,
        global_shape=(n_z, q_idx.size, meta.n_rmu, meta.n_rmu))
    del value


def _fit_head_samples(
    fit_path, head_samples, z, n_p, grid_hash, mesh_xy, *, model, solve,
    occupation_state=None,
):
    """Publish scalar Wc_head on the body's exact complex-frequency grid.

    ``head_correction=off`` is the exact zero model, not an ill-conditioned
    pole-fitting problem.  It still gets a complete scalar-head record because
    the Sigma consumer reads one uniformly; zero residues make that record a
    structural no-op for every real evaluation frequency.
    """
    wc = np.asarray([
        complex(sample.wcoul0) - complex(sample.vc0)
        for sample in head_samples
    ], dtype=np.complex128)
    if model == "head_off_zero":
        if np.any(wc != 0.0):
            raise ValueError(
                "head_off_zero received a nonzero scalar-head sample")
        # A non-real dummy pole avoids a zero denominator in the generic
        # evaluator.  Its value is immaterial because every residue is zero.
        fitted = {
            "Omega_p": np.full(n_p, -1.0j, dtype=np.complex128),
            "B_p": np.zeros(n_p, dtype=np.complex128),
            "condition": 0.0,
            "backward_error": 0.0,
            "max_abs_residual": 0.0,
        }
        provenance = {"solve_mode": "exact_zero", "n_valid": int(n_p)}
    else:
        fitted = fit_driver.fit_scalar_samples(wc, z, n_p, solve=solve)
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
    distrib_la_batched_route: str = "batch_reshard",
    reflected_chi_name=None,
    negative_wc_name=None,
):
    """THE DEFAULT ``wc_source``: Wc(z) = W(z) - V from the sampled chi.

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

    if (reflected_chi_name is None) != (negative_wc_name is None):
        raise ValueError(
            "_solve_wc requires reflected_chi_name and negative_wc_name "
            "together")

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
        Wc = W - V
        Wc.block_until_ready()
        mpa_store.write_w_slab_collective(
            sample_path, _WC, index, Wc, mesh_xy=mesh_xy,
            global_shape=shape)
        del chi, W, Wc

        if reflected_chi_name is not None:
            import jax

            chi_reflected, _ = mpa_store.read_w_slab_collective(
                sample_path, reflected_chi_name, index, mesh_xy=mesh_xy)
            W_reflected = solve_w(
                V, chi_reflected, meta, mesh_xy,
                dyson_solver=dyson_solver,
                distrib_la_batched_route=distrib_la_batched_route)
            # The ordered sweep supplies W(-conj(z)) in the upper half
            # plane.  Causality gives W(-z)=W(-conj(z))^dagger.  This is an
            # independently sampled partner, not a Hermitisation of W(z).
            Wc_negative = jnp.conj(jnp.swapaxes(W_reflected - V, -1, -2))
            Wc_negative = jax.lax.with_sharding_constraint(
                Wc_negative,
                NamedSharding(mesh_xy, P(None, "x", "y")))
            Wc_negative.block_until_ready()
            mpa_store.write_w_slab_collective(
                sample_path, negative_wc_name, index, Wc_negative,
                mesh_xy=mesh_xy, global_shape=shape)
            del chi_reflected, W_reflected, Wc_negative

    if head_response is None:
        return tuple(head_samples) if bgw_q0 is not None else None
    for index in range(int(n_z), len(head_response.omegas)):
        if bgw_q0 is not None:
            # Metallic MPA appends only the exact-static do_G0 sample after
            # the fit grid.  The Gamma row of the near-line origin sample was
            # replaced by the same static order-of-limits response above, so
            # that q=0 head value -- not the literal finite-q response -- is
            # the one reused here.
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
              provenance=None, occupation_state=None, solve="loewner",
              w_negative_name=None):
    return fit_driver.run_fit_driver(
        sample_path, _WC, fit_path, z, n_p, mesh_xy=mesh_xy,
        w_negative_name=w_negative_name,
        tile_bytes=tile_bytes, provenance=provenance,
        occupation_state=occupation_state, solve=solve)


def _metal_kminq_rows(sym, q_idx):
    """Per-wedge-row flat ``k → k−q`` maps for the direct origin sample.

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


def _require_metal_occupations(material_class, occupation_state):
    """One owner of the metal-needs-occupations refusal (two gates pin it:
    the build_mpa_fit entry, before any inode exists, and the
    _evaluate_samples seam for direct callers)."""
    if material_class == "metal" and occupation_state is None:
        raise ValueError(
            "GATE mpa_metal_needs_occupations: a metal MPA plan requires "
            "occupation_state (gw.efermi.OccupationState); got None. "
            "FALSE case: material_class == 'insulator', or an "
            "OccupationState was passed.")


def chi0_orientation_route(material_class: str, *, trs_allowed: bool) -> str:
    """Return the single run-record description of the MPA chi0 route."""
    if str(material_class) != "insulator":
        return (
            "fractional-occupation ordered pairs (independent of global "
            "TRS completion)")
    if not bool(trs_allowed):
        return (
            "global time reversal MEASURED BROKEN; ordered orientations use "
            "one contour sweep plus the q-negated conjugate partner, "
            "retaining the TR-odd channel")
    return (
        "global time reversal MEASURED to hold; incumbent symmetric "
        "completion retained bit-for-bit")


def _evaluate_samples(
    wfns, routes, quad, config, meta, mesh_xy, *,
    material_class, sym,
    energy_reference, occupation_state, write_full, write_wedge,
    static_gamma_override, gamma_row, kminq_rows, write_reflected=None,
    print_fn=print,
):
    """Evaluate every plan point through its route's kernel.

    Insulating plans whose measured verdict permits time reversal keep the
    historical kernels on a byte-identical code path.  Broken-TR insulating
    plans use one ordered-orientation contour sweep and obtain its partner by
    q-negated conjugation.  Metal plans (shifted-origin double-parallel
    protocol) route the
    near line's first sample through the exact direct ordered-pair kernel —
    the damped contour rule at varpi = 2e-5 Ry needs ~1.0e6 nodes
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
        compute_chi0_contour_ordered,
        compute_chi0_contour_fractional,
        compute_chi0_direct_fractional,
        occupation_support_bandwidth,
    )

    metal = material_class == "metal"
    _require_metal_occupations(material_class, occupation_state)
    if not hasattr(sym, "trs_allowed"):
        raise ValueError(
            "GATE mpa_needs_measured_trs: the MPA response route requires "
            "SymMaps.trs_allowed; the supplied symmetry object carries no "
            "measured verdict. Construct SymMaps from WfnLoader instead of "
            "asserting time reversal in the MPA consumer.")
    trs_allowed = bool(sym.trs_allowed)
    ordered = bool(not metal and not trs_allowed)
    if ordered and write_reflected is None:
        raise ValueError(
            "measured-broken-TR MPA sampling requires a reflected writer")
    q_neg = None
    if ordered:
        from symmetry_maps import q_negation_index

        q_neg = q_negation_index(
            (int(meta.nkx), int(meta.nky), int(meta.nkz)))
    print_fn(
        "  MPA chi0 orientation route: "
        + chi0_orientation_route(
            material_class, trs_allowed=trs_allowed)
        + ".")

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

    for point in routes["existing"]:
        if not metal:
            if point["character"] == "static":
                chi = compute_chi0(
                    wfns, quad, meta, mesh_xy,
                    energy_reference=energy_reference)
            elif ordered and point["character"] == "imag":
                rule = evaluator.damped_line_rule(
                    point["varpi"], omega_m,
                    rel_tol=config.minimax_config.target_error,
                    max_order=config.minimax_config.max_nodes)
                chi, chi_reflected = compute_chi0_contour_ordered(
                    wfns, rule["t"], rule["h"],
                    np.asarray([point["z"]], dtype=np.complex128),
                    meta, mesh_xy, q_neg_index=q_neg,
                    energy_reference=energy_reference,
                    return_reflected=True)
            elif ordered:
                raise ValueError(
                    "GATE mpa_broken_tr_existing_sample: a measured-broken-"
                    "TR insulating MPA plan contains an incumbent-only "
                    f"{point['character']!r} sample at z={point['z']!r}. "
                    "Only static and upper-imaginary existing-kernel cells "
                    "have an ordered-orientation completion; use the "
                    "double-parallel MPA plan.")
            else:
                used = build_imag_quadrature(
                    quad, point["varpi"],
                    MinimaxConfig(
                        target_error=config.minimax_config.target_error,
                        max_nodes=config.minimax_config.max_nodes))
                chi = compute_chi0(
                    wfns, used, meta, mesh_xy,
                    energy_reference=energy_reference)
            if (
                point["character"] == "static"
                and static_gamma_override is not None
            ):
                chi = chi.at[0].set(static_gamma_override[0])
            if ordered and point["character"] == "static":
                chi_reflected = chi
            write_full(point, chi)
            if ordered:
                write_reflected(point, chi_reflected)
        elif point["role"].startswith("near"):
            # Evaluate the literal shifted coordinate.  The shift avoids an
            # interpolation point at the metal's singular origin; writing a
            # static divided difference into this nonzero slot instead moved
            # the real Na response by up to 0.78% at finite q (claim 0385).
            chi_w = compute_chi0_direct_fractional(
                wfns, np.asarray([point["z"]], dtype=np.complex128),
                meta, mesh_xy, occupation_state=occupation_state,
                kminq_rows=kminq_rows,
                nb_logical=(
                    int(meta.b_id_4_chi_user) - int(wfns.slices.b0)),
                progress_fn=lambda q_done, q_total, elapsed: print_fn(
                    "  MPA direct chi0 shifted-origin q row "
                    f"{q_done}/{q_total} complete in {elapsed:.3f} s"))
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
                energy_reference=float(occupation_state.mu_ry),
                occupation_window_threshold=occ_window)
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
                energy_reference=float(occupation_state.mu_ry),
                occupation_window_threshold=occ_window)
        elif ordered:
            values, reflected_values = compute_chi0_contour_ordered(
                wfns, t, h, z, meta, mesh_xy, q_neg_index=q_neg,
                energy_reference=energy_reference,
                return_reflected=True)
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
        if ordered:
            reflected_values = (
                (reflected_values,) if z.size == 1 else reflected_values)
        else:
            reflected_values = (None,) * len(points)
        for point, chi, chi_reflected in zip(
                points, values, reflected_values):
            write_full(point, chi)
            if ordered:
                write_reflected(point, chi_reflected)


def build_mpa_fit(
    run_dir, label, *, wfns, V_q, quad, sym, centroid_indices, wfn=None,
    head_resolver, config, meta, mesh_xy, energy_reference=0.0,
    tile_bytes=None, plan=None, iteration_head_response=None,
    occupation_state=None, material_class, head_channel=None, wc_source=None,
    print_fn=print,
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
    _require_metal_occupations(material_class, occupation_state)
    if wc_source is not None and material_class == "metal":
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
    plan = (make_mpa_plan(config, quad, material_class=material_class)
            if plan is None else plan)
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
    ordered = bool(
        material_class == "insulator" and not bool(sym.trs_allowed))
    if ordered and wc_source is not None:
        raise ValueError(
            "GATE mpa_ordered_ladder_unimplemented: measured-broken-TR "
            "MPA currently requires the RPA Dyson source so both ordered "
            "frequency partners can be solved; use screening_diagrams = "
            "w_rpa or implement the reflected ladder source at the one "
            "wc_source seam.")
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
    if config.mpa.sampling_schedule != "nested":
        sampling_record["sampling_schedule"] = config.mpa.sampling_schedule
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
    if ordered:
        reflected_common = dict(common)
        reflected_common.update(
            omega=-np.conj(z_all),
            omega_line=line)
        mpa_store.allocate_w_omega_collective(
            sample_path, _CHI_REFLECTED, mode="a", **reflected_common)
        negative_common = dict(common)
        negative_common.update(
            omega=-z_all,
            omega_line=line)
        mpa_store.allocate_w_omega_collective(
            sample_path, _WC_NEGATIVE, mode="a", **negative_common)

    routes = sample_plan.plan_routes(plan)
    metal = material_class == "metal"
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

    def _write_reflected(point, chi):
        _write_sample(
            sample_path, point["index"], chi, q_idx, meta, mesh_xy,
            z_all.size, name=_CHI_REFLECTED)

    _evaluate_samples(
        wfns, routes, quad, config, meta, mesh_xy,
        material_class=material_class,
        sym=sym,
        energy_reference=energy_reference,
        occupation_state=occupation_state,
        write_full=_write_full, write_wedge=_write_wedge,
        static_gamma_override=static_gamma_override,
        gamma_row=gamma_row, kminq_rows=kminq_rows,
        write_reflected=_write_reflected if ordered else None,
        print_fn=print_fn)

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
                config.backend, "distrib_la_batched_route", "batch_reshard"),
            reflected_chi_name=_CHI_REFLECTED if ordered else None,
            negative_wc_name=_WC_NEGATIVE if ordered else None,
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
    _, report = _fit_body(
        sample_path, fit_path, z_all, n_p, tile_bytes, mesh_xy,
        provenance=provenance, occupation_state=occupation_state,
        solve=config.mpa.pole_solver,
        w_negative_name=_WC_NEGATIVE if ordered else None)
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
        head_model = f"bgw_q0shift_{config.mpa.pole_solver}"
    elif iteration_head is None:
        head_fit_samples = tuple(
            head_resolver.at(complex(z)) for z in z_all)
        correction = getattr(
            config.head.correction, "value", config.head.correction)
        head_model = (
            "head_off_zero" if correction == "off"
            else f"dft_direct_{config.mpa.pole_solver}")
    else:
        head_fit_samples = iteration_head.samples[:z_all.size]
        if bool(getattr(
                config.head, "uses_bgw_metal_q0shift", False)):
            head_model = f"bgw_q0shift_{config.mpa.pole_solver}"
        elif (head_fit_samples
              and getattr(getattr(head_fit_samples[0], "response_kind", None),
                          "value", None) == "micro_reducible"):
            head_model = (
                f"bse_resolvent_micro_{config.mpa.pole_solver}")
        else:
            head_model = (
                f"qsgw_schur_{config.mpa.pole_solver}"
                if (iteration_head_response is not None
                    and iteration_head_response.Y_x is not None)
                else f"qsgw_direct_{config.mpa.pole_solver}"
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
        solve=config.mpa.pole_solver,
        occupation_state=occupation_state,
    )
    print_fn(fit_driver.format_cost_report(report))
    return fit_path, iteration_head


__all__ = [
    "build_mpa_fit",
    "make_mpa_plan",
    "make_mpa_plan_from_fit",
    "validate_reused_mpa_fit",
    "retain_iteration_artifacts",
    "iteration_artifact_paths",
]
