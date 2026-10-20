"""Time a compact active-destination native Sigma sweep BEFORE BSE/full SC.

Use native_frozen_map_probe --pre-sigma-helper THIS_FILE. The accepted
reference and full microscopic W stay unchanged. This is a raw750 equilibrium
body benchmark, not driven MGKBA, a tail-extrapolated result, or a new SC map.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time
from types import SimpleNamespace


def validate_config(config):
    """Resolve real native typed fields BEFORE an expensive W construction."""
    from common.units import RYD_TO_EV
    settings = dict(regularization_width_ry=float(config.sigma.regularization_ev)/RYD_TO_EV,
        edge_factor=float(config.sigma.window_edge_factor), quadrature_eps=float(config.sigma.quadrature_eps),
        omega_grid_step_ry=float(config.sigma.omega_step_ev)/RYD_TO_EV,
        pole_batch_size=int(config.mpa.pole_batch_size),
        expected_screening_diagrams=config.screening.diagrams,
        sigma_w_model="shared_pole", material_class="insulator",
        linalg=config.backend.linalg, keep_branches=True, projection_bands=(100, 144))
    if settings["linalg"] not in ("local", "distributed") or not config.sigma.w_model == "shared_pole":
        raise ValueError("native shared-pole backend layout required")
    return settings


def active_request(energy, occupations, config):
    """Request ONLY [100,144) external energies, without changing virtuals."""
    import numpy as np
    from common.units import RYD_TO_EV
    from gw.efermi import resolve_sigma_efermi_ry
    from gw.ppm_sigma import on_shell_request
    energy, occupations = np.asarray(energy), np.asarray(occupations)
    if (energy.shape != (64, 750) or occupations.shape != energy.shape
            or not np.isfinite(energy).all() or not np.isfinite(occupations).all()
            or not np.isin(occupations, [0., 1.]).all()
            or not np.all(occupations.sum(axis=1) == 130)):
        raise ValueError("exact full64/750 accepted insulating energy/occupation table required")
    vbm, cbm = float(energy[occupations == 1].max()), float(energy[occupations == 0].min())
    if cbm <= vbm:
        raise ValueError("accepted insulating gap required")
    # Resolve the existing policy on CURRENT endpoints, not the original DFT WFN.
    efermi, policy = resolve_sigma_efermi_ry(config.sigma.fermi_reference,
        occupation_state=None, wfn=SimpleNamespace(vbm=vbm, cbm=cbm, efermi=(vbm+cbm)/2))
    external = (energy[:, 100:144]-efermi)*RYD_TO_EV
    request = on_shell_request(e_eval_rel_ev=external, e_dft_rel_ev=external,
        omega_range_ev=config.omega_grid_ev, out_of_grid=config.sigma.out_of_grid,
        irr_idx=np.arange(64), n_parent=64)
    return request, float(efermi), policy


def validate_reference(source):
    """Use the same authenticated own-grid contract as zero/driven callbacks."""
    path = Path(__file__).with_name("sc750_td_callback.py")
    spec = importlib.util.spec_from_file_location("active_sigma_reference_contract", path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner.source_response_policy(source)


def run_pre_sigma_probe(source, *, out):
    """Cold and held-session SAME-W sweeps; native gates are unchanged.

    The held session may serve a different certified, padded rule. These
    are identical physical inputs, not necessarily identical executables.
    """
    import numpy as np
    import jax
    from common.collectives import gather_to_host, rank0_transaction
    from common.units import RYD_TO_EV
    from gw import quadrature_log
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.mpa.sigma import compute_sigma_c_mpa_omega_grid
    meta, config, mesh = source["meta"], source["config"], source["mesh"]
    response_policy = validate_reference(source)
    energy = np.asarray(gather_to_host(source["energy_ry"]))
    occ = np.asarray(gather_to_host(source["occupations"]))
    request, efermi, frame_policy = active_request(energy, occ, config)
    model = source["W_by_role"]["shared_pole"]
    if not all(k in model for k in ("path", "identity", "digest")):
        raise ValueError("actual authenticated native W handle required")
    out = Path(out).resolve()
    rank0_transaction(out, stage="active_sigma/new_output", write=lambda: out.mkdir(parents=True, exist_ok=False))
    spec = importlib.util.spec_from_file_location("active_sigma_io", Path(__file__).with_name("run_sc_reference_frames.py"))
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)

    def save(name, *, record=None, arrays=None):
        rank0_transaction(out/name, stage="active_sigma/"+name,
            write=lambda: io.exclusive_write(out/name, record=record, arrays=arrays))

    common = dict(validate_config(config), request=request, efermi_ry=efermi,
        fit_identity=model["identity"], fit_digest=model["digest"])
    session = {}
    save("started.json", record=dict(status="STARTED_NOT_COMPLETED", frame=source["frame"],
        execution_revision=os.environ.get("LEG_HEAD"),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        response_policy=response_policy,
        active_destination_zero_based=[100, 144], virtual_bands=750,
        energy_frame_ry=efermi, energy_frame_policy=frame_policy,
        external_energy_range_ev=[float(energy[:, 100:144].min()*RYD_TO_EV),
                                  float(energy[:, 100:144].max()*RYD_TO_EV)],
        scope="raw750 equilibrium correlation BODY, no head/tail extrapolation, no driven full-coherent W refresh"))
    records, first, difference = [], None, None
    for repeat in range(2):
        started = time.perf_counter()
        current_meta = fresh_shared_pole_capacity(meta, mesh)
        # Price this sweep beside the ACTUAL live750 carrier, V and W.
        # A fresh ledger alone has no allocator baseline/room witness.
        current_meta.shared_pole_capacity.at_phase()
        result = compute_sigma_c_mpa_omega_grid(source["wfns"], model["path"],
            current_meta, mesh,
            fixed_quadrature_session=session, **common)
        device_pair = result.sigma_c_kij.pair
        jax.block_until_ready(device_pair)
        wall = time.perf_counter()-started
        pair = np.asarray(gather_to_host(device_pair))
        if pair.shape != (2, 64, 44, 44) or not np.isfinite(pair).all():
            raise ValueError("compact real-destination Sigma output contract failed")
        arrays = dict(pair_ry=pair)
        for space in ("cond", "val"):
            arrays[space+"_pair_ry"] = np.asarray(gather_to_host(result.branches[space].pair))
        save(f"sweep_{repeat}.npz", arrays=arrays)
        geometry, _ = quadrature_log.sigma_plan()
        if geometry is None:
            raise ValueError("missing actual native Sigma quadrature receipt")
        nodes = int(geometry["window_tau_pairs"])
        records.append(dict(repeat=repeat, seconds=wall, window_tau_pairs=nodes,
            branch_distinct_tau=int(geometry["distinct_tau_count"]),
            logical_windows=int(geometry["n_windows"]),
            amortized_wall_seconds_per_window_tau_pair=None if not nodes else wall/nodes,
            amortization_scope="whole sweep includes planning/compile/contraction; NOT isolated kernel latency",
            rules_built=int(geometry["rules_built"]), real_destinations=44,
            output_shape=list(pair.shape), no_external_prefix_outputs=True))
        save(f"sweep_{repeat}.json", record=records[-1])
        if first is None:
            first = pair.copy()
        else:
            # Different certified rules need not agree at machine precision.
            # Report the observed difference; this timing test does not derive
            # a full matrix error bound from the scalar quadrature tolerance.
            difference = dict(max_abs_ry=float(np.max(abs(pair-first))),
                relative_frobenius=None if not np.linalg.norm(first) else
                    float(np.linalg.norm(pair-first)/np.linalg.norm(first)),
                scope="same physical W/endpoints, different certified rules possible; NOT a convergence verdict")
    save("complete.json", record=dict(status="ACTIVE44_RAW750_BODY_TIMING_COMPLETE", sweeps=records,
        active_destination_zero_based=[100, 144], virtual_bands=750, model_digest=model["digest"],
        response_policy=response_policy,
        frame=source["frame"]["frame"], no_SC_update=True, no_TD_propagation=True,
        head_evaluated=False, band_extrapolation_evaluated=False,
        cold_held_difference=difference,
        scope="native equilibrium W and diagonal internal occupations; not the full coherent refreshed-W TD cost"))
