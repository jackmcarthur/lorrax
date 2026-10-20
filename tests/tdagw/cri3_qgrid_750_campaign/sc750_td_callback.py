"""Explicit accepted-SC750 -> native snapshot -> short zero-field TD probe.

Campaign adapter, not another GW/TD equation owner. Requires a live full64
post-Sigma rerotation and its actual spectral-shell fit. No Ceq subtraction,
population/PSD clipping, trace repair, or old 160-band source is used.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import time


def snapshot_boundary_report(gamma, injection, U, rho_in_basis):
    """Fresh boundary test in ONE gauge; all inputs are small host arrays, Ry."""
    import numpy as np
    from tdagw.collision import boundary_diagnostics
    rho = U@rho_in_basis@U.conj().swapaxes(-1, -2)
    loss = -.5*(gamma@rho+rho@gamma)
    raw = injection+loss
    boundary = boundary_diagnostics(gamma, injection, rho, occupation_tolerance=1e-8)
    return rho, loss, raw, boundary


def source_response_policy(source):
    """Require an own-grid accepted reference, including the coarse window."""
    meta, frame = source["meta"], source["frame"]
    electronic, grid = tuple(meta.kgrid), tuple(meta.qgrid)
    window = getattr(source["config"], "qgrid_response_window", "sharp")
    if getattr(meta, "qgrid_response_window", window) != window:
        raise ValueError("GATE sc750_response_grid: config/meta response windows differ")
    if electronic != (8, 8, 1) or grid not in ((8, 8, 1), (4, 4, 1)):
        raise ValueError("GATE sc750_response_grid: electronic8 and response8 or4 required")
    if grid != tuple(frame.get("qgrid", ())):
        raise ValueError("GATE sc750_response_reference: own converged qgrid frame required, not relabeled q8")
    policy = source.get("reference_policy")
    if grid != electronic and (not policy or tuple(policy.get("qgrid", ())) != grid
            or policy.get("response_window") != window
            or policy.get("accepted_input_frame") != frame["frame"]
            or policy.get("input_h_sha256") != frame["input_h_sha256"]
            or policy.get("accepted_deck_sha256") != frame.get("deck_sha256")):
        raise ValueError("GATE sc750_response_reference: authenticated coarse accepted-deck/window join required")
    return dict(electronic_kgrid=list(electronic), response_qgrid=list(grid),
        response_q_count=math.prod(grid), response_window=window,
        effective_response_window="identity" if grid == electronic else window,
        accepted_response_policy=policy)


def fine_bare_head_coefficient(source):
    """Bare HF uses the fine electronic cell, never the coarse correlation head."""
    import numpy as np
    sample = source["head_resolver"].direct_at(0j)
    coefficient = complex(sample.vc0)
    volume = float(source["meta"].cell_volume)
    if not (np.isfinite(coefficient) and abs(coefficient.imag) <= 1.e-10
            and math.isfinite(volume) and volume > 0):
        raise ValueError("GATE sc750_td_bare_head: verified real fine electronic-cell head required")
    return float(coefficient.real)/(volume*math.prod(source["meta"].kgrid))


def probe_active_window(deck):
    """Resolve the two explicit campaign windows, never an inferred resize."""
    v=deck.values
    request=(v['system.ibndmin'],v['system.ibndmax'],v['system.ivbm'])
    if request not in ((129,132,130),(101,144,130)):
        raise ValueError('GATE sc750_td_window: explicit [128,132) or [100,144) window required')
    active=slice(request[0]-1,request[1])
    expected=[active.start+1,active.stop+1]
    if not any(typ=='density_matrix' and name=='rho_full'
               and kwargs.get('brange1')==expected and kwargs.get('brange2')==expected
               for name,typ,kwargs in deck.measurements):
        raise ValueError('GATE sc750_td_window: matching full active rho_full measurement required')
    return active


def latest_snapshot_row(evaluated, basis):
    """Single-trial capture must join the latest evaluated basis by identity."""
    if len(evaluated)!=1 or evaluated[0][0] is not basis:
        raise ValueError('GATE sc750_td_capture: latest single-trial evaluated basis does not match snapshot state')
    return evaluated[0][1]


def validate_probe_deck(deck, *, tf_fs_limit=1.0):
    """Keep this first gate zero-field, short, raw, and reference anchored."""
    v = deck.values
    required = {"lorrax.collision": "raw", "lorrax.collision_statistics": "retarded_reference",
        "lorrax.collision_hamiltonian": "reference", "lorrax.collision_hermitian": "dynamic",
        "lorrax.collision_basis": "instantaneous", "lorrax.collision_internal_pairing": "vertex",
        "lorrax.collision_rate_intermediates": "active", "lorrax.collision_in_scattering": "full"}
    for key, value in required.items():
        if v[key] != value:
            raise ValueError(f"GATE sc750_td_deck: {key} must be {value!r}")
    if v["lorrax.collision_screening"] not in ("joint", "joint_frozen"):
        raise ValueError("GATE sc750_td_deck: native joint/joint_frozen screening required")
    probe_active_window(deck)
    if v["lorrax.intraband"] or v["system.depopulation"] or v["system.dephasing"]:
        raise ValueError("GATE sc750_td_deck: no added phenomenological damping or intraband model")
    # Parser times are Ha atomic units; this is a duration, not an output index.
    fs_per_au = 24.188843265857 / 1000.
    span = (v["control.timesteps.tf"]-v["control.timesteps.t0"])*fs_per_au
    if not 0 < span <= tf_fs_limit:
        raise ValueError("GATE sc750_td_duration: this gate is at most one fs")
    for axis in "xyz":
        value = v["system.Efunc.E_"+axis]
        if not isinstance(value, (int, float)) or value != 0:
            raise ValueError("GATE sc750_td_zero_field: a later driven deck needs a separate target gate")
    if len(deck.trials) != 1:
        raise ValueError("GATE sc750_td_trials: one explicit zero-field trial required")
    return span


def run_td_probe(source, *, out, deck_path, budget_s=900., storage_gib=64.):
    """Collective native execution on source['mesh']; no allocation or source rerun."""
    import importlib.util
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, rank0_transaction, all_gather_processes
    from file_io.commit_state import agree_io_refusal
    from tdagw import snapshot_runtime as sr, stages
    from tdagw.td_config import read_td_deck
    from tdagw.td_system import _structure
    from tdagw.tdgw_io import efield_function, HA_IN_EV
    from common.units import RYD_TO_EV
    started, out = time.perf_counter(), Path(out).resolve()
    if (not math.isfinite(budget_s) or not math.isfinite(storage_gib)
            or budget_s <= 0 or storage_gib <= 0):
        raise ValueError("positive finite explicit probe budgets required")
    mesh, meta = source["mesh"], source["meta"]
    frame, arrays, complete = source["frame"], source["frame_arrays"], source["driver_complete"]
    response_policy = source_response_policy(source)
    if (arrays["energy_ry"].shape != (64, 750)
            or arrays["U_active"].shape != (64, 160, 160)
            or not complete or not complete.get("reference_ready")
            or source.get("spectral_tail_reference") is None
            or source.get("head") is None or source.get("efermi_ry") is None):
        raise ValueError("GATE sc750_td_source: own-grid accepted full750 post-Sigma source/tail/head required")
    deck_path = Path(deck_path).resolve()
    deck = read_td_deck(str(deck_path))
    duration = validate_probe_deck(deck)
    active = probe_active_window(deck)
    active_bands = [active.start,active.stop]
    n_active = active.stop-active.start
    values = dict(deck.values)
    values["control.files.outdir"] = str(out)
    values["lorrax.bundle_input"] = source["input_deck"]
    values["lorrax.collision_source"] = source["input_deck"]
    values["lorrax.isdf_input"] = source["input_deck"]
    values["lorrax.dipole_file"] = str(Path(source["input_dir"])/"dipole.h5")
    tdgw = copy.deepcopy(deck.tdgw)
    tdgw.setdefault("control", {}).setdefault("files", {})["outdir"] = str(out)
    if "lorrax" in tdgw:
        for key in ("bundle_input", "collision_source", "isdf_input", "dipole_file"):
            tdgw["lorrax"][key] = values["lorrax."+key]
    deck = deck._replace(values=values, tdgw=tdgw)

    def host(value):
        if isinstance(value, (tuple, list)):
            return np.stack([host(x) for x in value])
        return np.asarray(gather_to_host(value)) if isinstance(value, jax.Array) else np.asarray(value)

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        if isinstance(value, (jax.Array, np.ndarray)):
            if value.size > 65536:
                raise ValueError("GATE sc750_td_record: large arrays belong in explicitly priced NPZ fields")
            return plain(host(value).tolist())
        if isinstance(value, np.generic):
            return plain(value.item())
        if isinstance(value, complex):
            return dict(real=value.real, imaginary=value.imag)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError("unsupported scalar report type: "+type(value).__name__)

    spec = importlib.util.spec_from_file_location(
        "sc750_frame_io", Path(__file__).with_name("run_sc_reference_frames.py"))
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    rank0_transaction(out, stage="sc750_td/new_output",
        write=lambda: out.mkdir(parents=True, exist_ok=False))

    def save(name, *, record=None, arrays=None):
        path = out/name
        record = None if record is None else plain(record)
        rank0_transaction(path, stage="sc750_td/"+name,
            write=lambda: io.exclusive_write(path, record=record, arrays=arrays))

    def resources(stage):
        size, error = 0, None
        if jax.process_index() == 0:
            try:
                size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            except OSError as exc:
                error = exc
        agree_io_refusal(error, path=out, stage="sc750_td/storage")
        totals = np.asarray(all_gather_processes(np.array(
            [time.perf_counter()-started, size], dtype=np.float64))).reshape(-1, 2).max(axis=0)
        if totals[0] > budget_s or totals[1] > storage_gib*2**30:
            raise RuntimeError(f"GATE sc750_td_resources: {stage}, elapsed={totals[0]:.1f}s bytes={totals[1]:.0f}")

    nb = arrays["U_active"].shape[-1]
    velocity = source["velocity_dft_cart"]
    if velocity is None:
        raise ValueError("GATE sc750_td_velocity: actual authenticated native DFT velocity required")
    velocity = jax.jit(lambda v: v[:, :, :nb, :nb],
        out_shardings=NamedSharding(mesh, P()))(velocity)
    # Same bare-head owner/normalization as ppm_pipeline.publish_window_hf_reference.
    head_coefficient = fine_bare_head_coefficient(source)
    inputs = dict(frame_receipt=frame, driver_complete=complete,
        original_psi=source["original_psi"], velocity_dft_ry=host(velocity),
        wfn_fingerprint=frame["wfn_fingerprint"], velocity_wfn_fingerprint=frame["wfn_fingerprint"],
        structure=_structure(source["input_deck"]), bare_head_coefficient_ry=head_coefficient)
    built = sr.build_sc_td_reference(source, arrays, inputs, mesh=mesh, active=active)
    hdr, ops = built["header"], built["ops"]
    system = stages.TdSystem(hdr, ops, 64, built["receipts"]["nk_pad"], n_active,
        float(inputs["structure"]["cell_volume"]), int(hdr["nspinor"]), hdr["bands"],
        receipts=built["receipts"])
    save("reference.json", record=dict(**built["receipts"],
        status="AUTHENTICATED_REFERENCE_NOT_COLLISION_STATIONARITY",
        deck_sha256=hashlib.sha256(deck_path.read_bytes()).hexdigest(),
        reference_producer_revision=frame["source_revision"], duration_fs=duration,
        **response_policy,
        execution_revision=os.environ.get("LEG_HEAD"),
        execution_source_archive_sha256=os.environ.get("LEG_SOURCE_ARCHIVE_SHA256"),
        execution_snapshot_runtime_sha256=hashlib.sha256(Path(sr.__file__).read_bytes()).hexdigest(),
        callback_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_source_paths={key: values[key] for key in ("lorrax.bundle_input",
            "lorrax.collision_source", "lorrax.dipole_file", "control.files.outdir")},
        runtime_path_scope="authenticated callback source; no standalone bundle reload",
        actual_active=active_bands, active_size=n_active,
        reference_occupied_active=130-active.start, reference_empty_active=active.stop-130,
        snapshot_U_scope=f"relative instantaneous rotation within fixed accepted-SC active [{active.start},{active.stop})",
        DFT_gauge_reconstruction="requires external accepted full U160 and active embedding; not snapshot U alone",
        snapshot_gap_scope=f"{n_active}-active-band material H, not complete750 fundamental gap after external reordering",
        sigma_eta_ev=float(source["config"].sigma.regularization_ev),
        response_gamma_ev=float(values["lorrax.collision_adiabatic_ev"]),
        spectral_tail="actual native accepted-state producer fit; fullvirtual MH only",
        active_rate_intermediates=active_bands, virtual_MH_bands=750,
        absolute_HF_not_published=True, H_anchor="accepted SC H once + bareHF delta + MH-MH_eq",
        Vcorr_scope="implicit accepted SC anchor; no separately published absolute HF/Vcorr decomposition",
        raw_collision_unsubtracted=True, clipping=False, trace_repair=False))
    evaluated, handles = [], []
    original_factory, original_snapshot, original_write = (
        sr.make_native_snapshot_evaluator, sr.snapshot_collision, stages._write_chunk)

    def factory(*a, **kw):
        evaluate, cache = original_factory(*a, **kw)
        def observed(basis, **kwargs):
            resources("before_native_snapshot")
            row = evaluate(basis, **kwargs)
            # One trial only. Init and refresh publish the basis from their
            # latest evaluate; never retain64 full44 operator rows just for
            # an identity lookup. The actual evaluator caches are untouched.
            evaluated[:] = [(basis,row)]
            resources("after_native_snapshot")
            return row
        return observed, cache

    def snapshot(*a, **kw):
        collision = original_snapshot(*a, **kw)
        handles.append(collision)
        n = 0
        def capture():
            nonlocal n
            state = collision["state"]
            basis = state["basis_last"][0]
            row = latest_snapshot_row(evaluated,basis)
            data = dict(U=basis["U"], E_ry=basis["e"], rho_in_basis=basis["rho"],
                Gamma_ry=host(row["gamma"]), D_ry=host(row["injection"]), MH_ry=host(row["m_h"]),
                MH_eq_ry=host(collision["eq"]["m_h"]), bundle_of=collision["bundle_of"],
                held_MH_ha=host(state["snapshot_held"]))
            rho_source, loss, raw, boundary = snapshot_boundary_report(
                data["Gamma_ry"], data["D_ry"], data["U"], data["rho_in_basis"])
            material = (data["U"]*data["E_ry"][:, None, :])@data["U"].conj().swapaxes(-1, -2)
            data.update(rho_source=rho_source, C_loss_ry=loss, C_gain_ry=data["D_ry"],
                C_raw_ry=raw, H_material_ry=material,
                C_coherent_ry=-1j*(material@rho_source-rho_source@material))
            if n == 0:
                data["C_raw_eq_ha"] = host(collision["c_eq"])
                order = collision["bundle_of"]
                req = host(ops.rho_eq)[:64]
                gam = data["Gamma_ry"][order]*(RYD_TO_EV/HA_IN_EV)
                inj = data["D_ry"][order]*(RYD_TO_EV/HA_IN_EV)
                data["C_gain_eq_ha"] = inj
                data["C_loss_eq_ha"] = -.5*(gam@req+req@gam)
                data["H_SC_anchor_ha"] = np.array([np.diag(x) for x in host(ops.e_qp)[:64]])
                data["C_coherent_initial_ha"] = -1j*(data["H_SC_anchor_ha"]@req-req@data["H_SC_anchor_ha"])
            for branch in ("cond", "val"):
                data["retarded_"+branch] = host(row["raw"]["retarded_pair"][branch])
                data["active_"+branch] = host(row["raw"]["active_branch_pair"][branch])
            save(f"snapshot_{n:04d}.npz", arrays=data)
            save(f"snapshot_{n:04d}.json", record=state["history"][-1])
            save(f"snapshot_{n:04d}_boundary.json", record=dict(
                occupation_tolerance=1e-8, rate_unit="Ry", per_k=boundary,
                weighted_trace_rate_ry=float(np.mean(np.trace(raw, axis1=-2, axis2=-1).real)),
                basis="Gamma,D,MH,rho_source in fixed accepted-SC active basis; BUNDLE k order",
                actual_active=active_bands, active_size=n_active,
                H_material_scope=f"{n_active}x{n_active} accepted-SC-active gauge; U relative within this window, not original DFT gauge",
                gap_scope="active-window eigenvalue separation only, not a full750 gap certificate",
                raw_branch_basis="instantaneous basis; BUNDLE k order; no arithmetic shell averaging",
                guarantee="state-local neighborhood diagnostic, not nonlinear Pauli invariance"))
            n += 1
        capture()
        refresh = collision["refresh"]
        def observed_refresh(*args, **kwargs):
            update = refresh(*args, **kwargs)
            if update is not None:
                capture()
            return update
        collision["refresh"] = observed_refresh
        return collision

    def write(outs, current, i0, i1, samples, state_fn, t_state, fingerprint):
        state = state_fn()
        rho = host(state[0])[:, :64]
        eig = np.linalg.eigvalsh((rho+rho.conj().swapaxes(-1, -2))/2)
        if (not np.isfinite(rho).all() or eig.min() < -1e-6 or eig.max() > 1+1e-6
                or np.max(abs(rho-rho.conj().swapaxes(-1, -2))) > 1e-9):
            save("refused_density.npz", arrays=dict(rho=rho, time_au=np.array(t_state)))
            raise ValueError("GATE sc750_td_density: exploratory numerical/Pauli guard, no clipping")
        original_write(outs, current, i0, i1, samples, lambda: state, t_state, fingerprint)
        resources("accepted_chunk")

    sr.make_native_snapshot_evaluator, sr.snapshot_collision, stages._write_chunk = factory, snapshot, write
    try:
        outputs = stages.open_outputs(deck, system)
        stats = stages.propagate(deck, system, built["model"], efield_function(deck.trials),
            outputs, mesh, collision=source)
        refreshes = 0 if not handles else int(handles[0]["state"]["n"])
        if values["lorrax.collision_screening"] == "joint" and refreshes < 1:
            raise ValueError("GATE sc750_td_refresh: no actual updated-W refresh was reached")
        save("complete.json", record=dict(status="SHORT_ZERO_FIELD_PROBE_COMPLETE_NOT_DRIVEN_ADMISSION",
            frame=frame["frame"], stats=stats, accepted_refreshes=refreshes,
            completed=True, zero_gate=True, amplitude_v_m=0., duration_fs=duration,
            actual_active=active_bands, active_size=n_active,
            response_qgrid=list(meta.qgrid), response_window=response_policy["response_window"],
            wall_s=time.perf_counter()-started,
            active_GPUh=jax.process_count()*(time.perf_counter()-started)/3600,
            stationarity="measure raw drift; not inferred from static SC convergence"))
    finally:
        # Do not enter collective I/O from an arbitrary rank-local exception.
        # The outer STARTED record and absence of CALLBACK_RETURNED preserve
        # incomplete status; normal rank logs retain the original traceback.
        sr.make_native_snapshot_evaluator, sr.snapshot_collision, stages._write_chunk = (
            original_factory, original_snapshot, original_write)
