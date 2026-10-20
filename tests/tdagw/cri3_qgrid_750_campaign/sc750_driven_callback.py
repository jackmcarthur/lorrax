"""Prepared three-arm campaign on a live accepted SC750 input, NOT admitted TD.

Compatible with native_frozen_map_probe's post-Sigma callback seam. Here its
``--td-deck`` argument names an explicit JSON campaign plan, not a TOML deck.
Launch through that existing collective probe with this file as --td-helper;
create the reviewed JSON from example_plan() and fill actual prerequisite
paths. execute_mgkba is an internal launch control, not new user authorization.
No material launch, target energy, amplitude or convergence claim is encoded.
The strict four-active zero-field callback is deliberately unchanged.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import os
import time

ARMS = ("A_STATIC_TDAGW_NO_COLLISION", "B_FROZEN_W_MGKBA", "C_UPDATED_W_MGKBA")
MATCHING_ZERO_FS = .6
FS_PER_AU = .024188843265857


def sibling(name):
    path = Path(__file__).with_name(name+".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_plan():
    """Explicit finite candidate list, not an assumption that it reaches a target."""
    return dict(status="PREPARED_PENDING_MATERIAL_BSE_AND_ZERO_DATA", schema="sc750_driven_v1",
        bse_report="REQUIRED_ACTUAL_FINE_BSE/bright_subgap.json",
        zero_directory="REQUIRED_COMPLETED_ZERO_PROBE", window=[100, 144],
        degeneracy_mode="strict", duration_fs=10., dt_as=25., measure_as=50.,
        pulse=dict(start_fs=1., end_fs=5., phase_rad=0., polarization=[1., 0., 0.]),
        amplitude_candidates_v_m=[1.e8, 4.e8, 1.6e9, 6.4e9, 2.56e10],
        carrier_targets=[.01, .1], target_factor=2.,
        execute_mgkba=False, run_zero_controls=True,
        refresh_fs=.5, response_gamma_ev=.05, eta_ev=.25,
        snapshot_steps=32, snapshot_tol_ev=.0001, fit_tolerance=.02,
        wall_budget_s=5400., storage_gib=64.,
        gates=dict(rho_tolerance=1.e-6, trace_drift=1.e-5,
            fresh_trace_relative=1.e-7, fresh_trace_absolute_ry=1.e-12,
            tangent_tolerance_ry=1.e-8, zero_carrier_drift=.001))


def validate_plan(plan):
    """Reject implicit policy changes; no adaptive GW fits, fields or windows."""
    if plan.get("schema") != "sc750_driven_v1" or plan.get("window") != [100, 144]:
        raise ValueError("GATE driven_plan: explicit SC750 active [100,144) request required")
    if plan.get("degeneracy_mode") not in ("strict", "snap"):
        raise ValueError("GATE driven_plan: use existing strict/snap multiplet owner")
    for key in ("duration_fs", "dt_as", "measure_as", "refresh_fs", "response_gamma_ev",
                "eta_ev", "wall_budget_s", "storage_gib", "target_factor", "snapshot_tol_ev",
                "fit_tolerance"):
        if not math.isfinite(float(plan[key])) or float(plan[key]) <= 0:
            raise ValueError("GATE driven_plan: finite positive "+key+" required")
    if plan["duration_fs"] != 10. or plan["dt_as"] != 25. or plan["target_factor"] <= 1:
        raise ValueError("GATE driven_plan: first prepared comparison is 10fs/25as with finite target bracket")
    amp = list(map(float, plan["amplitude_candidates_v_m"]))
    if (not 1 <= len(amp) <= 6 or any(not math.isfinite(x) or x <= 0 for x in amp)
            or amp != sorted(set(amp)) or plan["carrier_targets"] != [.01, .1]):
        raise ValueError("GATE driven_plan: at most six distinct increasing amplitudes and .01/.1 targets")
    pulse = plan["pulse"]
    if not (0 <= pulse["start_fs"] < pulse["end_fs"] <= plan["duration_fs"]):
        raise ValueError("GATE driven_pulse: finite pulse must lie within trajectory")
    if not math.isfinite(float(pulse["phase_rad"])):
        raise ValueError("GATE driven_pulse: finite phase required")
    pol = list(map(float, pulse["polarization"]))
    if len(pol) != 3 or not all(math.isfinite(x) for x in pol) or abs(sum(x*x for x in pol)-1) > 1.e-12:
        raise ValueError("GATE driven_pulse: literal unit polarization required")
    if sum(x != 0. for x in pol) != 1 or max(map(abs, pol)) != 1.:
        raise ValueError("GATE driven_pulse: first campaign needs one signed Cartesian axis, not an unverified projection")
    if not isinstance(plan["execute_mgkba"], bool) or not isinstance(plan["run_zero_controls"], bool):
        raise ValueError("GATE driven_plan: internal execution flags must be boolean")
    if plan["execute_mgkba"] and plan["refresh_fs"] > .5:
        raise ValueError("GATE driven_plan: matching .6fs zero gate needs refresh_fs <= .5fs")
    if not isinstance(plan["snapshot_steps"], int) or not 1 <= plan["snapshot_steps"] <= 32:
        raise ValueError("GATE driven_plan: bounded native fixed-point steps required")
    for key, val in plan["gates"].items():
        if not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError("GATE driven_plan: explicit positive diagnostic gate "+key)
    return plan


def choose_calibrations(rows, targets=(.01, .1), factor=2.):
    """Choose actual complete A trajectories, never synthesize an amplitude."""
    eligible = [r for r in rows if r.get("completed") and r["carrier_peak"] > 0]
    selected = []
    for target in targets:
        if not eligible:
            raise ValueError("GATE driven_calibration: no complete physical A trajectory")
        row = min(eligible, key=lambda r: abs(math.log(r["carrier_peak"]/target)))
        if not target/factor <= row["carrier_peak"] <= target*factor:
            raise ValueError(f"GATE driven_calibration: finite bracket did not reach carrier target {target}")
        selected.append(dict(target_carriers=target, amplitude_v_m=row["amplitude_v_m"],
                             actual_A_carrier_peak=row["carrier_peak"], calibration_tag=row["tag"]))
    if len({r["amplitude_v_m"] for r in selected}) != len(selected):
        raise ValueError("GATE driven_calibration: nonlinear targets need distinct actual amplitudes")
    return selected


def pulse_trials(plan, energy_ev, amplitude):
    """Finite sin² pulse, Efunc t in attoseconds, frequency from actual BSE."""
    if not math.isfinite(energy_ev) or energy_ev <= 0 or amplitude < 0 or not math.isfinite(amplitude):
        raise ValueError("GATE driven_pulse: actual positive BSE energy and finite amplitude required")
    p = plan["pulse"]
    start, stop = 1000*p["start_fs"], 1000*p["end_fs"]
    carrier = (f"(({amplitude!r})*np.sin(np.pi*(t-({start!r}))/({stop-start!r}))**2"
               f"*np.cos(({energy_ev!r})*(t-({start!r}))/HBAR_IN_EV_AS+({p['phase_rad']!r}))"
               f"* ((t>=({start!r})) & (t<=({stop!r}))))")
    return {"E_"+a: (0. if c == 0 or amplitude == 0 else f"({c!r})*{carrier}")
            for a, c in zip("xyz", p["polarization"])}


def make_deck(plan, source, out, active, energy_ev, amplitude, arm, *, zero_gate=False):
    """Resolve native parser afresh; B/C differ ONLY by screening flag."""
    import tomllib
    from tdagw.td_config import parse_td_deck
    if arm not in ARMS:
        raise ValueError("GATE driven_arm: unknown arm")
    raw = tomllib.loads((Path(__file__).with_name("inputs")/"sc750_zero_joint_0p6fs.toml").read_text())
    if zero_gate and amplitude != 0.:
        raise ValueError("GATE driven_zero: matching-window gate must have exactly zero field")
    duration = MATCHING_ZERO_FS if zero_gate else plan["duration_fs"]
    raw["control"]["timesteps"].update(tf=1000*duration, dt_measure=plan["measure_as"])
    raw["control"]["files"]["outdir"] = str(out)
    raw["system"].update(ibndmin=active.start+1, ibndmax=active.stop, ivbm=130)
    raw["system"]["Efunc"] = pulse_trials(plan, energy_ev, amplitude)
    raw["measurements"]["3"].update(brange1=[active.start+1, active.stop+1],
                                    brange2=[active.start+1, active.stop+1])
    lor = raw["lorrax"]
    lor.update(dt=plan["dt_as"], chunk=1, bundle_input=source["input_deck"],
        isdf_input=source["input_deck"], collision_source=source["input_deck"],
        dipole_file=str(Path(source["input_dir"])/"dipole.h5"),
        collision="off" if arm == ARMS[0] else "raw",
        collision_screening="joint_frozen" if arm == ARMS[1] else "joint",
        collision_refresh_fs=plan["refresh_fs"], collision_adiabatic_ev=plan["response_gamma_ev"],
        collision_snapshot_steps=plan["snapshot_steps"],
        collision_snapshot_tol_ev=plan["snapshot_tol_ev"], collision_fit_tolerance=plan["fit_tolerance"])
    return parse_td_deck(raw, source="explicit SC750 driven campaign JSON plan")


def require_matching_zero_result(result, active, response_policy):
    """No four-band or wrong-policy result can admit a larger driven window."""
    if (not result.get("completed") or result.get("zero_gate") is not True
            or result.get("amplitude_v_m") != 0.
            or result.get("duration_fs") != MATCHING_ZERO_FS
            or result.get("actual_active") != [active.start, active.stop]
            or result.get("response_policy") != response_policy
            or result.get("accepted_refreshes", 0) < 1
            or len(result.get("fresh_snapshot_times_fs", ())) < 2
            or not any(t > 0 for t in result["fresh_snapshot_times_fs"])):
        raise ValueError("GATE driven_zero: completed matching-active .6fs/one-refresh raw gate required")
    return result


def validate_material_prerequisites(plan, frame, *, response_policy=None):
    """Read only actual tiny BSE/zero artifacts; no scalar gap surrogate."""
    import numpy as np
    import h5py
    bpath, zero = Path(plan["bse_report"]), Path(plan["zero_directory"])
    bse = json.loads(bpath.read_text())
    complete = json.loads((zero/"complete.json").read_text())
    ref = json.loads((zero/"reference.json").read_text())
    for record in (bse["frame"], ref):
        if any(record.get(k) != frame.get(k) for k in ("frame", "input_h_sha256", "wfn_fingerprint")):
            raise ValueError("GATE driven_reference: BSE/zero and live accepted input identities differ")
    if response_policy is not None and response_policy["response_qgrid"] != [8, 8, 1]:
        for record in (bse, ref):
            if (record.get("response_qgrid") != response_policy["response_qgrid"]
                    or record.get("response_window") != response_policy["response_window"]):
                raise ValueError("GATE driven_reference: BSE/zero response grid/window must match own coarse reference")
    if (ref.get("sigma_eta_ev") != plan["eta_ev"]
            or ref.get("response_gamma_ev") != plan["response_gamma_ev"]):
        raise ValueError("GATE driven_zero: zero-field eta/gamma differ from planned B/C closure")
    if (complete.get("status") != "SHORT_ZERO_FIELD_PROBE_COMPLETE_NOT_DRIVEN_ADMISSION"
            or complete.get("accepted_refreshes", 0) < 1 or complete.get("frame") != frame["frame"]):
        raise ValueError("GATE driven_zero: completed short probe with actual refreshed W required")
    target = bse.get("target_root_index")
    if target is None or target not in bse["bright_subgap_root_indices"]:
        raise ValueError("GATE driven_BSE: no actual resolved bright subgap root")
    energy = float(bse["energies_ev"][target])
    if not 0 < energy+float(bse["residual_norm_ev"][target]) < float(bse["gap_ev"]):
        raise ValueError("GATE driven_BSE: bright target is not resolved below current SC gap")
    if float(bse["residual_norm_ev"][target]) > .001:
        raise ValueError("GATE driven_BSE: unresolved native exciton residual")
    axis = next(i for i, x in enumerate(plan["pulse"]["polarization"]) if x != 0.)
    strength = float(bse["strength_xyz_bohr2_cell"][target][axis])
    if not math.isfinite(strength) or strength <= 1.e-8:
        raise ValueError("GATE driven_BSE_polarization: actual target is not bright on the selected Cartesian axis")
    with h5py.File(zero/"tdgw.h5", "r") as f:
        n = int(f["tdgw_data/checkpoint/it_next"][()])
        if n <= 1:
            raise ValueError("GATE driven_zero: no measured zero-field trajectory")
        carriers = np.asarray(f["tdgw_data/time_domain/carrier_density/data"][:n]).real
        if not np.isfinite(carriers).all() or abs(carriers).max() > plan["gates"]["zero_carrier_drift"]:
            raise ValueError("GATE driven_zero: measured zero-field carrier drift exceeds explicit exploratory limit")
    boundary_paths = sorted(zero.glob("snapshot_*_boundary.json"))
    if len(boundary_paths) < 2:
        raise ValueError("GATE driven_zero: initial plus refreshed boundary diagnostics required")
    for path in boundary_paths:
        r = json.loads(path.read_text())
        if abs(r["weighted_trace_rate_ry"]) > plan["gates"]["fresh_trace_absolute_ry"]:
            raise ValueError("GATE driven_zero: fresh number flux does not cancel; not an eta-tail excuse")
        for name, counts in (("empty_min_injection", "empty_count"), ("filled_min_blocking", "filled_count")):
            for value, count in zip(r["per_k"][name], r["per_k"][counts]):
                if count and (value is None or value < -plan["gates"]["tangent_tolerance_ry"]):
                    raise ValueError("GATE driven_zero: actual density-boundary tangent points outward")
    return dict(target_energy_ev=energy, target_root=target, target_axis="xyz"[axis],
        target_axis_strength_bohr2_cell=strength, bse=bse,
        bse_sha256=hashlib.sha256(bpath.read_bytes()).hexdigest(),
        zero_complete_sha256=hashlib.sha256((zero/"complete.json").read_bytes()).hexdigest(),
        zero_carrier_max=float(abs(carriers).max()), zero_boundary_records=len(boundary_paths),
        zero_active_bands=ref.get("active_bands"),
        zero_evidence_scope="PREFLIGHT_ONLY_NOT_DRIVEN_WINDOW_ADMISSION")


def completed_observables(path, *, duration_fs=None, measure_interval_fs=None):
    """Small native plot payload, cropped by actual checkpoint; no interpolation."""
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        n = int(f["tdgw_data/checkpoint/it_next"][()])
        if not 1 < n <= 2000:
            raise ValueError("GATE driven_samples: bounded actual checkpoint rows required")
        td = f["tdgw_data/time_domain"]
        if any(n > len(td[k+"/"+field]) for k in ("polarization", "Efield", "carrier_density")
               for field in ("times", "data")):
            raise ValueError("GATE driven_samples: checkpoint exceeds actual dataset length")
        times = np.asarray(td["polarization/times"][:n])
        if not all(np.array_equal(times, np.asarray(td[k+"/times"][:n]))
                   for k in ("carrier_density", "Efield")):
            raise ValueError("GATE driven_samples: native observable times differ; no interpolation")
        pol, field, carrier = (np.asarray(td[k+"/data"][:n])
                               for k in ("polarization", "Efield", "carrier_density"))
        checkpoint = float(f["tdgw_data/checkpoint/tchkp"][()])*FS_PER_AU
    if not all(np.isfinite(x).all() for x in (times, pol, field, carrier)):
        raise ValueError("GATE driven_samples: nonfinite native observables")
    if (times.ndim != 1 or pol.shape != (n, 3) or field.shape != (n, 3)
            or carrier.shape != (n, 2) or np.any(np.diff(times) <= 0)
            or not math.isfinite(checkpoint) or times[-1]*FS_PER_AU > checkpoint+1.e-8):
        raise ValueError("GATE driven_samples: invalid shapes, nonmonotonic times or samples beyond checkpoint")
    if (duration_fs is None) != (measure_interval_fs is None):
        raise ValueError("GATE driven_samples: duration and measurement interval must be bound together")
    if duration_fs is not None:
        if (not math.isfinite(duration_fs) or not math.isfinite(measure_interval_fs)
                or min(duration_fs, measure_interval_fs) <= 0):
            raise ValueError("GATE driven_samples: finite positive declared duration/measurement interval required")
        expected = np.arange(0., duration_fs-1.e-8*measure_interval_fs, measure_interval_fs)
        if (n != len(expected) or abs(checkpoint-duration_fs) > 1.e-8
                or not np.allclose(times*FS_PER_AU, expected, rtol=0, atol=1.e-8)):
            raise ValueError("GATE driven_samples: actual checkpoint/sample coverage is not the declared complete trajectory")
    carrier = carrier.real
    data = dict(time_fs=times*FS_PER_AU, polarization_au=pol, field_au=field,
        conduction_electrons_per_cell=carrier[:, 0], valence_holes_per_cell=carrier[:, 1])
    return data, dict(checkpoint_fs=checkpoint, measured_first_fs=float(times[0]*FS_PER_AU),
        measured_last_fs=float(times[-1]*FS_PER_AU), completed_samples=n,
        carrier_peak=float(carrier[:, 0].max()), hole_peak=float(carrier[:, 1].max()),
        polarization_unit="native atomic units e/bohr^2; complex diagnostic retained",
        field_unit="native atomic units", carrier_unit="electrons or holes per cell",
        time_scope="native checkpoint-cropped samples; no interpolation or fabricated terminal sample")


def gap_mixing_summary(U, energy_ry, reference_energy_ry, nval):
    """Window-only nominal-filling gaps and mixing outside reference multiplets."""
    import numpy as np
    from common.units import RYD_TO_EV
    e, u, ref = map(np.asarray, (energy_ry, U, reference_energy_ry))
    es = np.sort(e, axis=-1)
    outside = abs(ref[:, :, None]-ref[:, None, :]) > (.001/RYD_TO_EV)
    mixing = (abs(u)**2*outside).sum(axis=-2)
    return dict(minimum_direct_gap_ry=float(np.min(es[:, nval]-es[:, nval-1])),
        fundamental_gap_ry=float(es[:, nval].min()-es[:, nval-1].max()),
        filling_index=nval, scope="sorted active-window H at nominal filling, NOT occupation/full750 gap",
        reference_degeneracy_tolerance_ev=.001,
        mixing_outside_reference_degenerate_blocks_mean=float(mixing.mean()),
        mixing_outside_reference_degenerate_blocks_max=float(mixing.max()))


def run_td_probe(source, *, out, deck_path):
    """Run bounded A calibration, then explicitly enabled matched B/C arms.

    All native collectives/device operations execute on every rank. Exceptions
    propagate without adding collective I/O on a possibly divergent path.
    This must be called on the authentic live post-Sigma full64 rerotation.
    """
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, all_gather_processes, rank0_transaction
    from file_io.commit_state import agree_io_refusal
    from tdagw import stages, snapshot_runtime as sr
    from tdagw import propagate as tp
    from tdagw.dynsig import slow_basis_step, collision_basis_state
    from tdagw.td_system import _structure
    from tdagw.tdgw_io import efield_function
    from tdagw.tdgw_io import HA_IN_EV
    from common.units import RYD_TO_EV
    from common.band_degeneracy import resolve_band_window
    plan_path, out = Path(deck_path).resolve(), Path(out).resolve()
    plan = validate_plan(json.loads(plan_path.read_text()))
    for key in ("bse_report", "zero_directory"):
        plan[key] = str((plan_path.parent/plan[key]).resolve())
    mesh, meta, arrays = source["mesh"], source["meta"], source["frame_arrays"]
    zero = sibling("sc750_td_callback")
    response_policy = zero.source_response_policy(source)
    prereq = validate_material_prerequisites(plan, source["frame"], response_policy=response_policy)
    if (arrays["energy_ry"].shape != (64, 750)
            or arrays["U_active"].shape != (64, 160, 160)
            or source.get("spectral_tail_reference") is None
            or source.get("head") is None or source.get("driver_complete", {}).get("reference_ready") is not True):
        raise ValueError("GATE driven_source: own-grid accepted full64/750 post-Sigma source and tail required")
    if float(source["config"].sigma.regularization_ev) != plan["eta_ev"]:
        raise ValueError("GATE driven_eta: campaign and native source eta differ")
    nv, nc = resolve_band_window(arrays["energy_ry"][:, :160], 130, 30, 14,
        mode=plan["degeneracy_mode"], where="SC750 three-arm driven window")
    active = slice(130-nv, 130+nc)
    bse = prereq["bse"]
    if active.start > 130-int(bse["nval"]) or active.stop < 130+int(bse["ncond"]):
        raise ValueError("GATE driven_BSE_window: TD must contain the actual finite BSE window")
    io, static_owner = sibling("run_sc_reference_frames"), sibling("sc750_static_control")
    started = time.perf_counter()
    rank0_transaction(out, stage="driven/new_output", write=lambda: out.mkdir(parents=True, exist_ok=False))

    def host(x):
        if isinstance(x, (tuple, list)):
            return np.stack([host(v) for v in x])
        return np.asarray(gather_to_host(x)) if isinstance(x, jax.Array) else np.asarray(x)

    def scalar(x):
        if isinstance(x, dict):
            return {str(k): scalar(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [scalar(v) for v in x]
        if isinstance(x, (np.ndarray, jax.Array)):
            if x.size > 65536:
                raise ValueError("GATE driven_record: large data require explicit NPZ capture")
            return scalar(host(x).tolist())
        if isinstance(x, np.generic):
            return scalar(x.item())
        if isinstance(x, complex):
            return dict(real=x.real, imaginary=x.imag)
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if x is None or isinstance(x, (str, bool, float, int)):
            return x
        raise TypeError("unsupported driven record type "+type(x).__name__)

    def save(name, record=None, arrays=None):
        record = None if record is None else scalar(record)
        rank0_transaction(out/name, stage="driven/"+name,
            write=lambda: io.exclusive_write(out/name, record=record, arrays=arrays))

    def budget(stage):
        size, err = 0, None
        if jax.process_index() == 0:
            try:
                size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            except OSError as exc:
                err = exc
        agree_io_refusal(err, path=out, stage="driven/storage")
        live = np.asarray(all_gather_processes(np.array([time.perf_counter()-started, size]))).reshape(-1, 2).max(0)
        if live[0] > plan["wall_budget_s"] or live[1] > plan["storage_gib"]*2**30:
            raise RuntimeError("GATE driven_budget: "+stage)

    budget("before_reference")
    velocity = jax.jit(lambda v: v[:, :, :160, :160], out_shardings=NamedSharding(mesh, P()))(
        source["velocity_dft_cart"])
    inputs = dict(frame_receipt=source["frame"], driver_complete=source["driver_complete"],
        original_psi=source["original_psi"], velocity_dft_ry=host(velocity),
        wfn_fingerprint=source["frame"]["wfn_fingerprint"],
        velocity_wfn_fingerprint=source["frame"]["wfn_fingerprint"],
        structure=_structure(source["input_deck"]),
        bare_head_coefficient_ry=zero.fine_bare_head_coefficient(source))
    reference = sr.build_sc_td_reference(source, arrays, inputs, mesh=mesh, active=active)
    system = stages.TdSystem(reference["header"], reference["ops"], 64,
        reference["receipts"]["nk_pad"], active.stop-active.start, float(meta.cell_volume),
        int(reference["header"]["nspinor"]), reference["header"]["bands"], receipts=reference["receipts"])
    save("manifest.json", record=dict(status="STARTED_NOT_COMPLETED", plan=plan,
        actual_active=[active.start, active.stop], requested_active=[100, 144], reference=reference["receipts"],
        prerequisites={k: v for k, v in prereq.items() if k != "bse"},
        execution_revision=os.environ.get("LEG_HEAD"), reference_producer_revision=source["frame"]["source_revision"],
        execution_source_archive_sha256=os.environ.get("LEG_SOURCE_ARCHIVE_SHA256"),
        input_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        callback_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        snapshot_runtime_sha256=hashlib.sha256(Path(sr.__file__).read_bytes()).hexdigest(),
        static_control_sha256=hashlib.sha256(Path(static_owner.__file__).read_bytes()).hexdigest(),
        microscopic_W=dict(centroids=3000, response_q_count=math.prod(meta.qgrid),
            response_qgrid=list(meta.qgrid), response_window=response_policy["response_window"],
            electronic_k_count=64, virtual_bands=750, spatial_rank_truncation=False),
        response_policy=response_policy,
        scope="exploratory matched discretization, not optical/TD/positivity convergence"))
    w = host(system.ops.w_k)[:64]
    req = host(system.ops.rho_eq)[:64]
    fdiag = np.diagonal(req, axis1=-2, axis2=-1).real
    trace0 = float(np.einsum("k,kii->", w, req).real)
    rows = []
    completed_runs = []
    shared_equilibrium = {}

    def run_arm(tag, arm, amplitude, model, *, zero_gate=False):
        budget("before_"+tag)
        deck = make_deck(plan, source, out/tag, active, prereq["target_energy_ev"], amplitude, arm,
                         zero_gate=zero_gate)
        duration = MATCHING_ZERO_FS if zero_gate else plan["duration_fs"]
        health, evaluated, handles = [], [], []
        fresh_snapshot_times = []
        held_rates = {}
        A_diagnostic = dict(n=0, next_fs=0.)
        if arm == ARMS[0]:
            A_step = slow_basis_step(model.apply, system.ops.rho_eq, system.ops.e_qp,
                np.diagonal(req, axis1=-2, axis2=-1).real)

        def capture_A(rho_device, time_au):
            # Existing mean-field owner, no optical drive, no GW collision.
            # Diagnostic only at the common .5fs cadence, not an extra fit.
            U, e, _, _, _ = collision_basis_state(A_step, model.params, rho_device, time_au)
            U, e = host(U)[0, :64], host(e)[0, :64]*(HA_IN_EV/RYD_TO_EV)
            hm = (U*e[:, None, :])@U.conj().swapaxes(-1, -2)
            n = A_diagnostic["n"]
            save(f"{tag}/snapshot_{n:04d}.npz", arrays=dict(U=U, E_ry=e,
                rho_source=host(rho_device)[0, :64], H_material_ry=hm,
                time_fs=np.array(float(time_au)*FS_PER_AU)))
            save(f"{tag}/snapshot_{n:04d}.json", record=dict(
                history=dict(t_fs=float(time_au)*FS_PER_AU, screening="staticW0", collision="off"),
                gap_and_mixing=gap_mixing_summary(U, e, arrays["energy_ry"][:, active], nv),
                scope="actual staticA mean-field H without drive, existing slow_basis_step; fixed acceptedSC-active gauge"))
            A_diagnostic.update(n=n+1, next_fs=float(time_au)*FS_PER_AU+plan["refresh_fs"])
        oldwrite, oldfactory, oldsnapshot = stages._write_chunk, sr.make_native_snapshot_evaluator, sr.snapshot_collision
        save(tag+"/started.json", record=dict(arm=arm, amplitude_v_m=amplitude,
            zero_gate=zero_gate, duration_fs=duration, actual_active=[active.start, active.stop],
            response_policy=response_policy,
            nominal_vacuum_cycle_averaged_peak_intensity_w_cm2=.5*8.8541878128e-12*299792458*amplitude**2/1.e4,
            target_energy_ev=prereq["target_energy_ev"], resolved_values=deck.values,
            anchor="SC+deltaH-SEX(W0)" if arm == ARMS[0] else "SC+bareHFdelta+MH-MHeq",
            off_is_not_frozen_W=True))

        def observe_write(outs, current, i0, i1, samples, state_fn, t_state, fingerprint):
            state = state_fn()
            rho = host(state[0])[0, :64]
            adj = rho.conj().swapaxes(-1, -2)
            if not np.isfinite(rho).all():
                raise ValueError("GATE driven_density: nonfinite")
            eig = np.linalg.eigvalsh((rho+adj)*.5)
            occ = np.diagonal(rho, axis1=-2, axis2=-1).real
            spin = 2. if system.nspinor == 1 else 1.
            nc_now = float(spin*np.einsum("k,ka,ka->", w, 1-fdiag, occ))
            nh_now = float(spin*np.einsum("k,ka,ka->", w, fdiag, 1-occ))
            tr = float(np.einsum("k,kii->", w, rho).real)
            record = dict(time_fs=float(t_state)*FS_PER_AU, carrier=nc_now, holes=nh_now,
                trace=tr, purity=float(np.einsum("k,kab,kba->", w, rho, rho).real),
                rho_min=float(eig.min()), rho_max=float(eig.max()), hermiticity=float(abs(rho-adj).max()))
            if held_rates:
                g, d = held_rates["gamma_TD"], held_rates["injection_TD"]
                held_c = d-.5*(g@rho+rho@g)
                record.update(held_trace_rate_ry=float(np.einsum("k,kii->", w, held_c).real),
                    held_coefficients_from_fs=held_rates["t_fs"],
                    held_trace_scope="actual moving rho with held Gamma,D; not a fresh nonlinear flux test")
            health.append(record)
            save(f"{tag}/health_{len(health):04d}.json", record=record)
            if (eig.min() < -plan["gates"]["rho_tolerance"] or eig.max() > 1+plan["gates"]["rho_tolerance"]
                    or abs(tr-trace0) > plan["gates"]["trace_drift"] or record["hermiticity"] > 1.e-9):
                save(tag+"/refused_density.npz", arrays=dict(rho=rho, time_au=np.array(t_state)))
                raise ValueError("GATE driven_density: unmodified Pauli/trace/Hermiticity guard")
            if zero_gate and max(abs(nc_now), abs(nh_now)) > plan["gates"]["zero_carrier_drift"]:
                raise ValueError("GATE driven_zero: actual matching-window carrier drift exceeds explicit limit")
            oldwrite(outs, current, i0, i1, samples, lambda: state, t_state, fingerprint)
            if arm == ARMS[0] and record["time_fs"] >= A_diagnostic["next_fs"]-1.e-10:
                capture_A(tp.place_window_blocks(np.asarray(state[0]), mesh, system.nk_pad), t_state)
            budget(tag+"/chunk")

        def observe_factory(*a, **kw):
            evaluate, cache = oldfactory(*a, **kw)
            def observe(basis, **kwargs):
                budget(tag+"/before_snapshot")
                if kwargs.get("equilibrium") and shared_equilibrium:
                    if not all(np.array_equal(basis[k], shared_equilibrium["basis"][k])
                               for k in ("U", "e", "rho")):
                        raise ValueError("GATE driven_equilibrium: shared W initialization basis changed")
                    cache.update(equilibrium=shared_equilibrium["handle"],
                                 equilibrium_receipt=shared_equilibrium["receipt"])
                    value = shared_equilibrium["row"]
                else:
                    value = evaluate(basis, **kwargs)
                    if kwargs.get("equilibrium"):
                        shared_equilibrium.update(basis={k: np.array(basis[k], copy=True)
                            for k in ("U", "e", "rho")}, row=value,
                            handle=cache["equilibrium"], receipt=cache["equilibrium_receipt"])
                evaluated[:] = [(basis, value)]
                budget(tag+"/after_snapshot")
                return value
            return observe, cache

        def observe_snapshot(*a, **kw):
            collision = oldsnapshot(*a, **kw)
            handles.append(collision)
            index = 0
            def capture():
                nonlocal index
                state = collision["state"]
                basis = state["basis_last"][0]
                b, row = evaluated[-1]
                if b is not basis:
                    raise ValueError("GATE driven_snapshot: rate and density basis identities differ")
                gamma, inj = host(row["gamma"]), host(row["injection"])
                U, e, rb = (np.asarray(basis[k]) for k in ("U", "e", "rho"))
                rho, loss, raw, bd = zero.snapshot_boundary_report(gamma, inj, U, rb)
                trace = float(np.einsum("k,kii->", w, raw).real)
                flux = float(np.einsum("k,k->", w, np.linalg.norm(inj, axis=(-2, -1))+np.linalg.norm(loss, axis=(-2, -1))))
                hm = (U*e[:, None, :])@U.conj().swapaxes(-1, -2)
                data = dict(U=U, E_ry=e, rho_in_basis=rb, rho_source=rho,
                    Gamma_ry=gamma, D_ry=inj, MH_ry=host(row["m_h"]),
                    MH_eq_ry=host(collision["eq"]["m_h"]), H_material_ry=hm,
                    C_raw_ry=raw, C_gain_ry=inj, C_loss_ry=loss,
                    C_coherent_ry=-1j*(hm@rho-rho@hm), held_MH_ha=host(state["snapshot_held"]),
                    bundle_of=collision["bundle_of"], time_fs=np.array(state["history"][-1]["t_fs"]))
                gap = gap_mixing_summary(U, e, arrays["energy_ry"][:, active], nv)
                held_rates.update(gamma_TD=gamma[collision["bundle_of"]],
                    injection_TD=inj[collision["bundle_of"]], t_fs=state["history"][-1]["t_fs"])
                for branch in ("cond", "val"):
                    data["retarded_"+branch] = host(row["raw"]["retarded_pair"][branch])
                    data["active_"+branch] = host(row["raw"]["active_branch_pair"][branch])
                save(f"{tag}/snapshot_{index:04d}.npz", arrays=data)
                save(f"{tag}/snapshot_{index:04d}.json", record=dict(
                    history=state["history"][-1], boundary=bd, gap_and_mixing=gap, fresh_trace_rate_ry=trace,
                    fresh_trace_relative=abs(trace)/max(flux, 1.e-300),
                    scope="fresh coefficients in fixed SC-active basis/BUNDLE order; U is relative, not originalDFT",
                    held_trace="rho moves under held Gamma,D; not the fresh nonlinear cancellation identity",
                    gap_scope="active-window material H, not complete750 fundamental gap"))
                index += 1
                if abs(trace) > max(plan["gates"]["fresh_trace_absolute_ry"], plan["gates"]["fresh_trace_relative"]*flux):
                    raise ValueError("GATE driven_number: fresh raw trace flux; finite eta is not an excuse")
                for name, counts in (("empty_min_injection", "empty_count"), ("filled_min_blocking", "filled_count")):
                    vals = np.asarray(bd[name])[np.asarray(bd[counts]) > 0]
                    if vals.size and (not np.isfinite(vals).all() or vals.min() < -plan["gates"]["tangent_tolerance_ry"]):
                        raise ValueError("GATE driven_boundary: actual tangent outward; no PSD clipping")
                fresh_snapshot_times.append(float(state["history"][-1]["t_fs"]))
            capture()
            refresh = collision["refresh"]
            def observe_refresh(*a, **kw):
                value = refresh(*a, **kw)
                if value is not None:
                    capture()
                return value
            collision["refresh"] = observe_refresh
            return collision

        stages._write_chunk, sr.make_native_snapshot_evaluator, sr.snapshot_collision = observe_write, observe_factory, observe_snapshot
        try:
            outputs = stages.open_outputs(deck, system)
            if arm == ARMS[0]:
                capture_A(tp.place_window_blocks(req[None], mesh, system.nk_pad), 0.)
            stats = stages.propagate(deck, system, model, efield_function(deck.trials), outputs,
                mesh, collision=None if arm == ARMS[0] else source)
            refreshes = 0 if not handles else int(handles[0]["state"]["n"])
            if arm != ARMS[0] and refreshes < 1:
                raise ValueError("GATE driven_refresh: no current-electronic refresh reached")
            payload, measured = completed_observables(outputs.paths[0],
                duration_fs=duration, measure_interval_fs=plan["measure_as"]/1000.)
            save(tag+"/plot_data.npz", arrays=payload)
            result = dict(tag=tag, arm=arm, amplitude_v_m=amplitude, completed=True,
                zero_gate=zero_gate, duration_fs=duration, actual_active=[active.start, active.stop],
                response_policy=response_policy, fresh_snapshot_times_fs=fresh_snapshot_times,
                **measured, carrier_peak_scope="actual completed native measurement grid",
                checkpoint_carrier_peak=max(r["carrier"] for r in health),
                native_h5=outputs.paths[0], plot_data=str(out/tag/"plot_data.npz"),
                gap_mixing_scope="refresh snapshot JSON; window-only gaps and fixed-reference-degeneracy mixing",
                shared_equilibrium_W=True if arm != ARMS[0] else None,
                accepted_refreshes=refreshes, stats=stats)
            save(tag+"/complete.json", record=result)
            completed_runs.append(result)
            save(f"runs_{len(completed_runs):03d}.json", record=dict(runs=completed_runs,
                comparison_scope="A differs in coherent kernel; B/C differ only WR update, no automatic damping attribution"))
            return result
        finally:
            stages._write_chunk, sr.make_native_snapshot_evaluator, sr.snapshot_collision = oldwrite, oldfactory, oldsnapshot

    static = static_owner.build_static_control(source, reference)
    save("static_control.json", record=static["receipt"])
    if plan["run_zero_controls"]:
        run_arm("A_zero", ARMS[0], 0., static["model"], zero_gate=True)
    for index, amp in enumerate(plan["amplitude_candidates_v_m"]):
        row = run_arm(f"A_calibration_{index:02d}", ARMS[0], amp, static["model"])
        rows.append(row)
        save(f"calibration_{index:02d}.json", record=row)
        try:
            choose_calibrations(rows, plan["carrier_targets"], plan["target_factor"])
            break
        except ValueError:
            pass
    selected = choose_calibrations(rows, plan["carrier_targets"], plan["target_factor"])
    save("calibration_selected.json", record=dict(selected=selected, arm=ARMS[0], raw_observables=True))
    del static
    if plan["execute_mgkba"]:
        matching_gates = []
        # Mandatory even when the optional A-zero control is disabled. The
        # smaller supplied zero artifact remains a source/closure preflight.
        # Both arms must pass BEFORE any driven MGKBA arm starts.
        for arm in ARMS[1:]:
            result = run_arm(arm[0]+"_zero", arm, 0., reference["model"], zero_gate=True)
            matching_gates.append(require_matching_zero_result(result, active, response_policy))
        save("matching_active_zero_gates.json", record=dict(status="MATCHING_ACTIVE_BC_ZERO_GATES_COMPLETE",
            actual_active=[active.start, active.stop], response_policy=response_policy,
            duration_fs=MATCHING_ZERO_FS, supplied_zero_scope="PREFLIGHT_ONLY",
            gates=matching_gates, shared_equilibrium_W_and_Sigma=True))
        for arm in ARMS[1:]:
            for level, pick in enumerate(selected):
                run_arm(f"{arm[0]}_level_{level}", arm, pick["amplitude_v_m"], reference["model"])
    save("complete.json", record=dict(
        status="THREE_ARM_EXPLORATORY_COMPLETE" if plan["execute_mgkba"] else "A_CALIBRATION_ONLY_COMPLETE_BC_NOT_ATTEMPTED",
        selected=selected, wall_s=time.perf_counter()-started,
        no_precision_or_universal_positivity_claim=True,
        A_B_difference="different coherent functionals; not isolated damping",
        B_C_difference="W update including real MH/coherent and collision effects"))
