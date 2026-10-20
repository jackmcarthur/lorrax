"""External same-input q8/q4 native GW evaluations, not another SC solver.

PREPARED capability, not hardware verified. Initialize through a native q4 SC
deck (which authenticates full electronic parents), intercept its solver entry,
and replay the EXISTING native map twice on one accepted saved input frame.
Both calls hold the exact input E/f/U and full750 tail; each computes its own
native W, head, Hartree, Sigma and spectral-shell tail. Optional post-Sigma
TD uses an explicitly reconstructed current source and the actual tail fit.
"""
import argparse
import copy
import dataclasses
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import zipfile


def helpers():
    spec = importlib.util.spec_from_file_location(
        "native_sc_frame_helpers", Path(__file__).with_name("run_sc_reference_frames.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_accepted_frame(directory):
    """Authenticate an actual native-converged INPUT frame, never output EQP."""
    import numpy as np
    h = helpers()
    directory = Path(directory)
    complete = json.loads((directory / "driver_complete.json").read_text())
    solver = json.loads((directory / "native_solver_return.json").read_text())
    if (not complete.get("reference_ready") or complete.get("driver_returncode") != 0
            or complete.get("solver") != solver
            or solver.get("status") != "CONVERGED_ACCEPTED_INPUT"
            or not solver.get("accepted_input_join")
            or not solver.get("proper_sc_requested")
            or not solver.get("native_converged")):
        raise ValueError("accepted native solver/driver INPUT join is required")
    stem = solver["frame"]
    if Path(stem).name != stem:
        raise ValueError("frame name must be a local artifact stem")
    record = json.loads((directory / (stem + ".json")).read_text())
    with zipfile.ZipFile(directory / (stem + ".npz")) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024**2:
            raise MemoryError("accepted small-band frame exceeds the 256 MiB metadata limit")
    with np.load(directory / (stem + ".npz"), allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if record.get("schema") != "native_sc_input_frame_v1" or record["frame"] != stem:
        raise ValueError("unsupported or mismatched input frame")
    for key, stamp in (("U_active", "U_sha256"), ("energy_ry", "energy_sha256"),
                       ("occupations", "occupation_sha256"),
                       ("input_H_loop_ry", "input_h_sha256")):
        if h.array_digest(arrays[key]) != record[stamp]:
            raise ValueError(f"frame array digest mismatch: {key}")
    a, b = record["active_slice"]
    E, U = arrays["energy_ry"], arrays["U_active"]
    if a != 0 or b != 160 or E.shape != (64, 750) or U.shape != (64, 160, 160):
        raise ValueError("this scoped probe requires full64 / logical750 / protected160")
    H = (U * E[:, None, a:b]) @ U.conj().swapaxes(-1, -2)
    rows = arrays["input_loop_full_rows"]
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0)
            or np.any(rows >= 64)):
        raise ValueError("invalid authenticated loop row witness")
    if not np.allclose(H[rows], arrays["input_H_loop_ry"], rtol=0, atol=1.e-9):
        raise ValueError("full frozen U/E do not reconstruct the accepted input H")
    if not (np.isfinite(E).all() and np.isfinite(U).all()
            and np.isfinite(arrays["occupations"]).all()):
        raise ValueError("nonfinite accepted input metadata")
    if not (np.array_equal(E, arrays["energy_padded_ry"][:, :750])
            and np.array_equal(arrays["occupations"], arrays["occupation_padded"][:, :750])):
        raise ValueError("logical frame and padded carrier disagree")
    record = dict(record, accepted_solver=solver, reference_ready=True,
                  accepted_driver_complete=complete,
                  frame_npz_sha256=hashlib.sha256(
                      (directory / (stem + ".npz")).read_bytes()).hexdigest())
    return arrays, record


def band_pair_comparison(fine, coarse):
    """Full complex band-operator differences, not a microscopic-W certificate."""
    import numpy as np
    from common.units import RYD_TO_EV
    if fine.shape != coarse.shape or fine.ndim != 4 or fine.shape[0] != 2:
        raise ValueError("want matching row/column on-shell pairs (2,k,M,M)")
    delta = coarse - fine
    n = fine.shape[-1]
    off = ~np.eye(n, dtype=bool)
    result = dict(shape=list(fine.shape), unit="rydberg",
                  max_abs_mev=float(np.max(np.abs(delta))) * RYD_TO_EV * 1000,
                  per_k_max_abs_mev=(np.max(np.abs(delta), axis=(0, 2, 3))
                                    * RYD_TO_EV * 1000).tolist())
    for label, x, d in (("complex", fine, delta), ("real", fine.real, delta.real),
                        ("imaginary", fine.imag, delta.imag),
                        ("offdiagonal", fine[..., off], delta[..., off])):
        numerator, denominator = float(np.linalg.norm(d)), float(np.linalg.norm(x))
        result[label] = dict(difference_frobenius_ry=numerator,
                             fine_frobenius_ry=denominator,
                             relative=None if denominator == 0 else numerator / denominator)
    return result


def selected_qgrids(only):
    """Select full-BZ response legs, never a fine-q patch."""
    return ((8, 8, 1), (4, 4, 1)) if only is None else ((int(only), int(only), 1),)


def accepted_response_policy(frame, deck_path):
    """Bind the actual converged producer deck, not the probe's new q label."""
    from gw.gw_config import read_lorrax_input, parse_qgrid
    path = Path(deck_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != frame.get("deck_sha256"):
        raise ValueError("accepted response policy needs the exact converged producer deck SHA")
    values = read_lorrax_input(str(path))
    grid = tuple(parse_qgrid(values.get("qgrid")) or frame["kgrid"])
    if grid != tuple(frame.get("qgrid", ())):
        raise ValueError("accepted producer deck and frame qgrid disagree")
    return dict(accepted_deck_sha256=digest, qgrid=list(grid),
        response_window=values.get("qgrid_response_window", "sharp"),
        accepted_input_frame=frame["frame"], input_h_sha256=frame["input_h_sha256"])


def validate_callback_reference(frame, grid, policy, window):
    """A material callback needs its OWN converged response-policy reference."""
    grid = tuple(grid)
    if grid != tuple(frame.get("qgrid", ())):
        raise ValueError("callback requires its own converged qgrid reference, not a relabeled state")
    if grid == (4, 4, 1) and (not policy or tuple(policy.get("qgrid", ())) != grid
            or policy.get("accepted_input_frame") != frame["frame"]
            or policy.get("input_h_sha256") != frame["input_h_sha256"]
            or policy.get("accepted_deck_sha256") != frame.get("deck_sha256")
            or policy.get("response_window") != window):
        raise ValueError("q4 callback requires authenticated accepted deck/window/frame policy")


def run_pre_sigma_only(callback, current, *, out, publish, finished):
    """Stop only AFTER a successful collective callback, before broad Sigma.

    This explicit external-driver boundary is not an SC convergence or
    fine/coarse comparison verdict. Callback exceptions propagate unchanged.
    """
    callback(current, out=out)
    grid = tuple(getattr(current.get("meta"), "qgrid", (8, 8, 1)))
    publish("pre_sigma_complete.json", record=dict(status="PRE_SIGMA_ONLY_COMPLETE",
        accepted_input_frame=current["frame"]["frame"],
        response_qgrid=list(grid), broad_protected_Sigma_performed=False,
        coarse_leg_performed=grid == (4, 4, 1), second_leg_performed=False,
        new_SC_convergence_claim=False, cutoff_accuracy_verdict=False))
    finished.append("PRE_SIGMA_ONLY")
    raise StopIteration("EXTERNAL_PRE_SIGMA_PROBE_COMPLETE")


def evaluate_frozen_leg(owner, initial_state, inputs, arrays, frame, *, qgrid,
                        directory, publish, gather, G0_canonical=None,
                        source_callback=None, post_sigma_callback=None, input_deck=None,
                        G0_receipt=None, reference_policy=None):
    """Evaluate one existing native map; source callback is synchronous only."""
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.parallel_transport import fingerprint_from_binding
    from gw.gw_init import screening_coulomb_view
    from gw.head_correction import HeadResolver
    from gw import sigma_dispatch, ppm_pipeline
    h = helpers()
    leg_started = time.perf_counter()
    carrier = inputs.wfns_dft.green_parent
    if (carrier is None or carrier.plan.n_parent != 64 or carrier.plan.n_full != 64
            or tuple(inputs.meta.kgrid) != (8, 8, 1)):
        raise ValueError("native q4 setup must supply complete full64 electronic parents")
    if (inputs.config.bispinor or inputs.material_class != "insulator"
            or inputs.config.sigma.w_model != "shared_pole"
            or inputs.config.sc.head_update != "dft_velocity"):
        raise ValueError("probe is scoped to scalar insulating native-SPM dft_velocity")
    if fingerprint_from_binding(inputs.wfn_fingerprint_binding, inputs.wfn) != frame["wfn_fingerprint"]:
        raise ValueError("frozen frame and freshly loaded native WFN identities differ")
    if not np.array_equal(inputs.meta.mu_basis.canonical_indices,
                          arrays["centroid_canonical_indices"]):
        raise ValueError("centroid canonical locations differ from accepted frame")
    if not np.allclose(np.asarray(inputs.sym.unfolded_kpts),
                       arrays["kpoints_full_frac"], rtol=0, atol=1.e-12):
        raise ValueError("electronic k-row/representative coordinates differ; no guessed reorder")
    if tuple(inputs.wfns_dft.enk.shape) != tuple(arrays["energy_padded_ry"].shape):
        raise ValueError("native and captured padded band carriers differ")
    meta = copy.copy(inputs.meta)
    meta.qgrid = tuple(qgrid)
    meta.__post_init__()
    meta.q_sym = inputs.sym.qgrid_view(meta.qgrid, kgrid=meta.kgrid)
    config = dataclasses.replace(inputs.config, qgrid=tuple(qgrid),
        debug=dataclasses.replace(inputs.config.debug, sigma_freq_debug_output=True))
    if source_callback is not None or post_sigma_callback is not None:
        validate_callback_reference(frame, qgrid, reference_policy, config.qgrid_response_window)
    paths = {key: str(directory / Path(getattr(config.paths, key)).name)
             for key in ("sigma_diag_file", "eqp0_file", "eqp1_file", "report_file",
                         "sigma_omega_h5_file")}
    config = dataclasses.replace(config, paths=dataclasses.replace(config.paths, **paths))
    q0_certificates = []
    resolver = HeadResolver(config, inputs.input_dir, inputs.wfn, inputs.sym,
                            meta, inputs.print_fn, q0_certificate_fn=q0_certificates.append,
                            mesh=inputs.mesh_xy)
    Vscreen = screening_coulomb_view(inputs.V_q, meta=meta, mesh_xy=inputs.mesh_xy)
    rep = NamedSharding(inputs.mesh_xy, P(None, None))
    matrix = NamedSharding(inputs.mesh_xy, P(None, "x", "y"))
    U = device_put_process_local(arrays["U_active"], matrix)
    E = device_put_process_local(arrays["energy_padded_ry"], rep)
    f = device_put_process_local(arrays["occupation_padded"], rep)
    Hhost = ((arrays["U_active"] * arrays["energy_ry"][:, None, :160])
             @ arrays["U_active"].conj().swapaxes(-1, -2))
    H = device_put_process_local(Hhost, NamedSharding(inputs.mesh_xy, P()))
    state = dataclasses.replace(initial_state, H_qp_dft=H,
        iteration=max(1, int(frame["map_call"])), outputs=None,
        partition=None, occupation_state=None, tail_z_kn=None,
        convergence_verdict=None)
    captured = {}
    original_eigh, original_rotate = owner._sc_eigh_bands, owner.rotate_wavefunctions
    original_finalize = sigma_dispatch.finalize_dynamic_sigma
    original_fit = ppm_pipeline.fit_band_extrapolation_spectral
    original_screen = inputs.screening_model_fn

    def eigh(current_H, **kwargs):
        if not np.allclose(gather(current_H), Hhost, rtol=0, atol=1.e-9):
            raise ValueError("unexpected non-input eigensystem requested by fixed-map replay")
        return E[:, :160], U

    def rotate(*args, **kwargs):
        if args[0] is not inputs.wfns_dft:
            raise ValueError("unexpected second/current-transverse rotation")
        # Frozen-state comparison holds the already ACTUAL full750 input tail,
        # not a freshly guessed/re-fitted warm-start ladder. Sigma tail fits
        # below remain native and independent for each q leg.
        kwargs.update(enk_active_new=E[:, :160], enk_base=E,
                      occupations=f, efermi=None)
        current = original_rotate(args[0], U, **kwargs)
        if not (np.array_equal(gather(current.enk), arrays["energy_padded_ry"])
                and np.array_equal(gather(current.occ), arrays["occupation_padded"])):
            raise ValueError("native rotation did not preserve the exact frozen E/f")
        captured["rotation_authenticated"] = True
        # Exercise the ACTUAL TD source boundary before the expensive W bank.
        # The electronic sym table and meta.q_sym are independently owned;
        # do not replace either with the other's physical coordinate domain.
        from tdagw.snapshot_runtime import prepare_native_source
        prepared = prepare_native_source(source_record(current, {}), inputs.mesh_xy)
        publish(directory.name+"_source_preflight.json", record=dict(
            status="NATIVE_SOURCE_METADATA_PREFLIGHT_PASS", input_frame=frame["frame"],
            electronic_kgrid=list(prepared["kgrid"]), response_qgrid=list(prepared["qgrid"]),
            electronic_k_count=prepared["nk"], response_q_count=prepared["nq"],
            logical_bands=prepared["nb"], padded_bands=prepared["nb_padded"],
            flat_grid=prepared["flat_grid"], no_W_or_Sigma_admission_claim=True))
        return current

    def source_record(current, roles):
        return dict(wfns=current, original_wfns=inputs.wfns_dft,
                V_q=inputs.V_q, V_screen=Vscreen, meta=meta, config=config,
                sym=inputs.sym, W_by_role=roles, U_active=U,
                energy_ry=E[:, :750], occupations=f[:, :750], frame=frame,
                mesh=inputs.mesh_xy, head_resolver=resolver,
                iteration_head=roles.get("iteration_head"), input_dir=inputs.input_dir,
                G0_canonical=G0_canonical, G0_receipt=G0_receipt,
                reference_policy=reference_policy,
                spectral_tail_reference=captured.get("spectral_tail_reference"),
                velocity_dft_cart=(None if inputs.parallel_transport is None else
                                   inputs.parallel_transport.velocity_dft_cart),
                original_psi=inputs.wfns_dft.green_parent.projection_faces()[0],
                frame_arrays=arrays, input_deck=input_deck,
                driver_complete=frame.get("accepted_driver_complete"),
                head=captured.get("sigma_head"), member="shared_pole",
                efermi_ry=captured.get("efermi_ry"), wall_s=time.perf_counter()-leg_started,
                source_cost_scope="this frozen map/callback setup; outer native source initialization excluded",
                screening_inputs=dict(wfns=current, V_q=Vscreen, meta=meta,
                    config=config, kwargs=dict(sym=inputs.sym)),
                source_lifetime="explicit post-Sigma rerotation; original DFT carrier retained")

    def screen(mode, current, V_screen, **kwargs):
        result = original_screen(mode, current, V_screen, **kwargs)
        model = result.get("shared_pole", {})
        captured["model"] = {key: str(model[key]) for key in
                             ("path", "identity", "digest") if key in model}
        if post_sigma_callback is not None:
            captured["roles"] = result  # model/head handles, NOT current psi faces
        if source_callback is not None:
            source_callback(source_record(current, result))
        return result

    def fit(*args, **kwargs):
        actual = original_fit(*args, **kwargs)
        if "spectral_tail_reference" in captured:
            raise ValueError("unexpected multiple spectral-shell fits in one native map")
        captured["spectral_tail_reference"] = dict(fit=actual,
            input_frame=frame["frame"], input_h_sha256=frame["input_h_sha256"],
            wfn_fingerprint=frame["wfn_fingerprint"], ladder_shift_ev=0.0)
        return actual

    def finalize(body, head, **kwargs):
        from gw.dynamic_sigma import add_head_sigma_diag
        captured["sigma_head"] = head
        captured["efermi_ry"] = kwargs.get("efermi_ry")
        raw = kwargs.get("sigma_c_body_omega_unextrap")
        if raw is None or getattr(raw, "pair", None) is None:
            raise ValueError("native raw750 complex pair was not retained by debug twin policy")
        if tuple(raw.pair.shape) != (2, 64, 160, 160):
            raise ValueError("raw-band gather is bounded to the declared full64/M160 pair")
        raw_head = add_head_sigma_diag(raw, head, band_axis=kwargs["sigma_band_axis"])
        captured["raw750_pair_ry"] = np.array(gather(raw_head.pair), copy=True)
        result = original_finalize(body, head, **kwargs)
        pair = result.sigma_c_on_shell
        if pair is None or pair.pair is None:
            raise ValueError("native extrapolated full complex pair is unavailable")
        if tuple(pair.pair.shape) != (2, 64, 160, 160):
            raise ValueError("extrapolated gather exceeds the declared full64/M160 pair")
        captured["extrapolated_pair_ry"] = np.array(gather(pair.pair), copy=True)
        captured["sigma_x_ry"] = np.array(gather(result.sigma_x_kij_ry), copy=True)
        captured["e_eval_ev"] = np.array(result.e_eval_ev, copy=True)
        captured["actual_shell_counts"] = list(result.band_extrapolation_counts or ())
        return result

    leg = dataclasses.replace(inputs, meta=meta, config=config, V_q_screen=Vscreen,
        q_sym=meta.q_sym, head_resolver=resolver, screening_model_fn=screen,
        kstar=None, input_dir=str(directory), screening_seed_cache={},
        fixed_quadrature_session={})
    owner._sc_eigh_bands, owner.rotate_wavefunctions = eigh, rotate
    sigma_dispatch.finalize_dynamic_sigma = finalize
    ppm_pipeline.fit_band_extrapolation_spectral = fit
    h.mark("frozen_native_map_begin", qgrid=list(qgrid), frame=frame["frame"])
    try:
        mapped = owner.gw_iteration_map(state, leg)
        if not captured.get("rotation_authenticated"):
            raise ValueError("frozen input rotation was not authenticated")
        reference = captured.get("spectral_tail_reference")
        if reference is None:
            raise ValueError("native producer did not return the actual spectral-shell fit")
        # Save native fit fields, including DFT/Weyl ladder, without pickle or
        # reconstructing beta/Omega. All are small band-only host quantities.
        fit_arrays, fit_scalars = {}, {}
        def flatten(value, prefix):
            for field in dataclasses.fields(value):
                item = getattr(value, field.name)
                key = prefix + field.name
                if dataclasses.is_dataclass(item):
                    flatten(item, key + "_")
                elif isinstance(item, np.ndarray) or isinstance(item, tuple):
                    fit_arrays[key] = np.asarray(item)
                else:
                    fit_scalars[key] = h.scalar_record(item)
        flatten(reference["fit"], "fit_")
        if sum(value.nbytes for value in fit_arrays.values()) > 64 * 1024**2:
            raise MemoryError("native small-band spectral fit exceeded its 64 MiB capture bound")
        publish(directory.name + "_spectral_fit.npz", arrays=fit_arrays)
        publish(directory.name + "_spectral_fit.json", record=dict(
            input_frame=reference["input_frame"],
            input_h_sha256=reference["input_h_sha256"],
            wfn_fingerprint=reference["wfn_fingerprint"], ladder_shift_ev=0.0,
            native_fields=fit_scalars,
            scope="ACTUAL native producer fit at this accepted frozen input; amplitude not frozen"))
        fields = {key: captured[key] for key in
                  ("raw750_pair_ry", "extrapolated_pair_ry", "sigma_x_ry", "e_eval_ev")}
        publish(directory.name + "_band_operators.npz", arrays=fields)
        publish(directory.name + "_receipt.json", record=dict(
            status="FROZEN_MAP_EVALUATED_NOT_SC", qgrid=list(qgrid),
            electronic_grid=[8, 8, 1], frame=frame["frame"],
            current_input_energy_sha256=frame["energy_sha256"],
            current_input_U_sha256=frame["U_sha256"], model=captured["model"],
            actual_shell_counts=captured["actual_shell_counts"],
            pair_slots=["Sigma_mn(E_m)", "Sigma_mn(E_n)"],
            pair_scope="correlation with current head; bare exchange stored separately",
            input_sum_tail="held exact current750 from accepted frame",
            sigma_sum_tail="native spectral_shell recomputed independently for this q leg",
            q0_certificate_count=len(q0_certificates),
            q0_certificate_records="owned by native report/head callback; not reserialized here",
            full_W_comparison="NOT_MEASURED",
            source_callback_executed=source_callback is not None,
            post_sigma_callback_requested=post_sigma_callback is not None,
            newly_converged=False, metadata_only_host_arrays=True))
        if post_sigma_callback is not None:
            # Commit the completed native GW result BEFORE optional TD can
            # refuse its distinct finite-width/raw-collision gate.
            publish(directory.name+"_post_sigma_callback_started.json", record=dict(
                status="STARTED_NOT_COMPLETED", input_frame=frame["frame"], qgrid=list(qgrid)))
            h.mark("post_sigma_explicit_rerotation_begin", qgrid=list(qgrid))
            current = original_rotate(inputs.wfns_dft, U, enk_active_new=E[:, :160],
                enk_base=E, efermi=None, occupations=f, active_slice=slice(0, 160),
                mesh_xy=inputs.mesh_xy)
            post_sigma_callback(source_record(current, captured["roles"]))
            del current
            publish(directory.name+"_post_sigma_callback_complete.json", record=dict(
                status="CALLBACK_RETURNED", input_frame=frame["frame"], qgrid=list(qgrid),
                scope="callback return, not an automatic driven/material acceptance"))
            h.mark("post_sigma_explicit_rerotation_end", qgrid=list(qgrid))
        del mapped
        return fields
    finally:
        owner._sc_eigh_bands, owner.rotate_wavefunctions = original_eigh, original_rotate
        sigma_dispatch.finalize_dynamic_sigma = original_finalize
        ppm_pipeline.fit_band_extrapolation_spectral = original_fit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--setup-deck", required=True,
                        help="native q4/full64 proper-SC setup deck, not a max_iter1 wrapper")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--only-qgrid", choices=("4", "8"),
                        help="evaluate ONLY this full-BZ response grid; skip the other W rebuild")
    parser.add_argument("--accepted-deck", help="exact accepted producer deck; required for own-q4 callbacks")
    parser.add_argument("--out", required=True)
    pre_sigma = parser.add_mutually_exclusive_group()
    pre_sigma.add_argument("--bse-helper")
    pre_sigma.add_argument("--pre-sigma-helper", help="run collective fine-W callback and STOP before broad Sigma")
    parser.add_argument("--td-helper", help="reviewed post-Sigma SC750 TD callback module")
    parser.add_argument("--td-deck", help="explicit short zero-field TD deck for that callback")
    parser.add_argument("--wall-cap", type=float, default=900.)
    parser.add_argument("--stack-after", type=float, default=0.)
    args = parser.parse_args(argv)
    h = helpers()
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if (out.exists() or not math.isfinite(args.wall_cap) or args.wall_cap <= 0
            or not math.isfinite(args.stack_after) or args.stack_after < 0):
        parser.error("fresh --out, positive finite wall cap and nonnegative stack interval required")
    if bool(args.td_helper) != bool(args.td_deck):
        parser.error("--td-helper and --td-deck are required together")
    if args.pre_sigma_helper and (args.td_helper or args.td_deck):
        parser.error("--pre-sigma-helper stops before Sigma and cannot run TD")
    if os.environ.get("LEG_HEAD") != args.source_revision:
        parser.error("--source-revision must match launcher's immutable LEG_HEAD")
    sys.path[:0] = [str(source), str(source / "src")]
    faulthandler.enable(all_threads=True)
    if args.stack_after > 0:
        faulthandler.dump_traceback_later(args.stack_after, repeat=True)
    done = threading.Event()
    def watchdog():
        if not done.wait(args.wall_cap):
            h.mark("frozen_probe_wall_stop", seconds=args.wall_cap)
            os._exit(124)
    threading.Thread(target=watchdog, daemon=True).start()
    import runtime
    from gw import gw_jax, gw_init, sc_iteration
    import numpy as np
    from common.collectives import gather_to_host, rank0_transaction
    for module in (runtime, gw_jax, gw_init, sc_iteration):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError("unexpected native source origin")
    arrays, frame = load_accepted_frame(args.frames)
    grids = selected_qgrids(args.only_qgrid)
    policy = (None if not args.accepted_deck else accepted_response_policy(frame, args.accepted_deck))
    if (args.bse_helper or args.pre_sigma_helper or args.td_helper) and grids[0] == (4, 4, 1):
        if policy is None:
            policy = accepted_response_policy(frame, args.setup_deck)
    finished, G0, G0_receipts = [], [], []
    original_solver = sc_iteration.run_self_consistency
    original_vq = gw_init._finalize_vq_views
    def publish(name, **kwargs):
        path = out / name
        rank0_transaction(path, stage="frozen_probe/" + name,
                          write=lambda: h.exclusive_write(path, **kwargs))
    def capture_vq(*a, **kw):
        result = original_vq(*a, **kw)
        # The original owner already gathers/selects the exact q=0 vector.
        # It may carry canonical SUFFIX padding, never orbit packing.
        vector = np.array(result[1], copy=True)
        meta = a[3] if len(a) > 3 else kw["meta"]
        basis = meta.mu_basis
        G0[:] = [vector]
        G0_receipts[:] = [dict(producer="gw.gw_init._finalize_vq_views",
            q_fractional=[0., 0., 0.], order="canonical_suffix_padded",
            shape=list(vector.shape), dtype=str(vector.dtype),
            logical_extent=int(basis.n_logical), canonical_extent=int(basis.n_canonical),
            packed_extent=int(basis.n_packed),
            canonical_value_sha256=hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest(),
            canonical_centroids_sha256=hashlib.sha256(
                np.ascontiguousarray(basis.canonical_indices).tobytes()).hexdigest())]
        return result
    def intercepted_solver(initial, inputs, **kwargs):
        if tuple(inputs.meta.qgrid) != (4, 4, 1):
            raise ValueError("setup deck must use q4 to force native full64 parent preparation")
        pair_bytes = 2 * 64 * 160 * 160 * 16
        exchange_bytes = 64 * 160 * 160 * 16
        small_preflight = 4 * (sum(value.nbytes for value in arrays.values())
            + 2 * (2 * pair_bytes + exchange_bytes + 64 * 160 * 8)
            + 64 * 1024**2)
        if small_preflight > 2 * 1024**3:
            raise MemoryError("fixed-state band/fit capture exceeds2GiB host bound before maps")
        directories = [out / ("fine_q8" if grid == (8, 8, 1) else "coarse_q4") for grid in grids]
        def stage():
            out.mkdir(parents=True, exist_ok=True)
            for directory in directories:
                directory.mkdir(exist_ok=False)
                # Only immutable input-file links, never scratch/model trees.
                for name in ("dipole.h5", "data-file-schema.xml"):
                    target = Path(inputs.input_dir) / name
                    if target.is_file():
                        (directory / name).symlink_to(target.resolve())
        rank0_transaction(out, stage="frozen_probe/staging", write=stage)
        code_names = ("src/gw/sc_iteration.py", "src/gw/gw_init.py",
                      "src/gw/sigma_dispatch.py", "src/gw/ppm_pipeline.py",
                      "src/gw/wavefunction_bundle.py")
        publish("probe_manifest.json", record=dict(
            source=str(source), source_revision=args.source_revision,
            source_code_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in code_names},
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            accepted_frame=frame["frame"], accepted_frame_npz_sha256=frame["frame_npz_sha256"],
            accepted_frame_source_revision=frame["source_revision"],
            logical_bands=750, protected_bands=160, electronic_grid=[8, 8, 1],
            evaluated_response_grids=[list(grid) for grid in grids], reference_policy=policy,
            source_lifetime="immutable DFT source held; current faces not retained across Sigma",
            host_band_metadata_bound_bytes=2 * 1024**3,
            host_band_metadata_price_bytes=small_preflight,
            pre_sigma_only=(None if not args.pre_sigma_helper else dict(
                helper=str(Path(args.pre_sigma_helper).resolve()),
                helper_sha256=hashlib.sha256(Path(args.pre_sigma_helper).read_bytes()).hexdigest(),
                scope="selected current W and explicit callback only; no broad160 Sigma/other leg/TD")),
            no_material_capacity_admission_claim=True))
        if G0_receipts:
            publish("source_fit_G0.json", record=G0_receipts[0])
            publish("source_fit_G0.npz", arrays=dict(G0_canonical=G0[0]))
        callback = None
        td_callback = None
        if args.pre_sigma_helper:
            spec = importlib.util.spec_from_file_location("native_pre_sigma_probe", args.pre_sigma_helper)
            pre = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pre)
            if hasattr(pre, "validate_config"):
                pre.validate_config(inputs.config)
            callback = lambda current: run_pre_sigma_only(pre.run_pre_sigma_probe, current,
                out=out/("fine_pre_sigma" if grids[0] == (8, 8, 1) else "coarse_pre_sigma"),
                publish=publish, finished=finished)
        if args.bse_helper:
            spec = importlib.util.spec_from_file_location("native_bse_probe", args.bse_helper)
            bse = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bse)
            if hasattr(bse, "source_fit_g0"):
                bse.source_fit_g0(dict(meta=inputs.meta,
                    G0_canonical=None if not G0 else G0[0],
                    G0_receipt=None if not G0_receipts else G0_receipts[0]))
            callback = lambda current: bse.run_bse_probe(current,
                out=out / ("fine_bse" if grids[0] == (8, 8, 1) else "coarse_bse"),
                nval=4, ncond=4, nroots=24, maxiter=200)
        if args.td_helper:
            spec = importlib.util.spec_from_file_location("sc750_td_callback", args.td_helper)
            td = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(td)
            td_callback = lambda current: td.run_td_probe(current,
                out=out / ("fine_td" if grids[0] == (8, 8, 1) else "coarse_td"), deck_path=args.td_deck)
        results = []
        for index, (grid, directory) in enumerate(zip(grids, directories)):
            results.append(evaluate_frozen_leg(sc_iteration, initial, inputs, arrays,
                frame, qgrid=grid, directory=directory, publish=publish,
                gather=gather_to_host, G0_canonical=None if not G0 else G0[0],
                G0_receipt=None if not G0_receipts else G0_receipts[0],
                source_callback=callback if index == 0 else None,
                post_sigma_callback=td_callback if index == 0 else None, reference_policy=policy,
                input_deck=str(Path(args.setup_deck).resolve())))
        # Conservative four copies of the two band-only payloads and accepted
        # frame cover retained arrays, NPZ staging, norms/differences and BLAS
        # scratch. This does NOT price native source/device/model workspaces.
        small_host_price = 4 * (sum(value.nbytes for result in results for value in result.values())
                                + sum(value.nbytes for value in arrays.values()))
        if small_host_price > 2 * 1024**3:
            raise MemoryError("fixed-state small-band host working price exceeds2GiB/process")
        if len(results) == 2:
            publish("comparison.json", record=dict(status="FROZEN_BAND_OPERATOR_COMPARISON",
                raw750=band_pair_comparison(results[0]["raw750_pair_ry"], results[1]["raw750_pair_ry"]),
                extrapolated=band_pair_comparison(results[0]["extrapolated_pair_ry"], results[1]["extrapolated_pair_ry"]),
                full_W_comparison="NOT_MEASURED", accepted_fine_reproduction="NOT_MEASURED",
                no_cutoff_accuracy_verdict=True, no_new_convergence_claim=True))
        else:
            publish("selected_leg_complete.json", record=dict(status="SINGLE_FROZEN_LEG_COMPLETE",
                qgrid=list(grids[0]), comparison_performed=False, no_new_convergence_claim=True))
        finished.append(True)
        # Intentional external harness boundary: do not continue native SC or
        # run its terminal writers using an unevaluated/output carry.
        raise StopIteration("EXTERNAL_FROZEN_PROBE_COMPLETE")
    sc_iteration.run_self_consistency, gw_init._finalize_vq_views = intercepted_solver, capture_vq
    try:
        try:
            gw_jax.main(["-i", str(Path(args.setup_deck).resolve())])
        except StopIteration as exc:
            expected = ("EXTERNAL_PRE_SIGMA_PROBE_COMPLETE" if finished == ["PRE_SIGMA_ONLY"]
                        else "EXTERNAL_FROZEN_PROBE_COMPLETE")
            if not finished or str(exc) != expected:
                raise
        if not finished:
            raise RuntimeError("native driver did not reach the frozen-state evaluation seam")
        h.mark("external_frozen_probe_complete", cutoff_accuracy_verdict=False,
               pre_sigma_only=finished == ["PRE_SIGMA_ONLY"])
    finally:
        sc_iteration.run_self_consistency, gw_init._finalize_vq_views = original_solver, original_vq
        done.set()
        if args.stack_after > 0:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
