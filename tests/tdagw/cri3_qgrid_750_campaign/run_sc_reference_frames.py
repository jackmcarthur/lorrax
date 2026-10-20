"""Run the native GW CLI and preserve small, actual SC input frames.

External measurement helper, not an SC implementation or physics verification.
No wavefunction/ISDF/Coulomb faces are retained. A launcher-enforced hard step
deadline remains mandatory; the Python watchdog is only a redundant safeguard.
"""
import argparse
import dataclasses
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time


def mark(phase, **fields):
    print(json.dumps(dict(phase=phase, rank=os.environ.get("SLURM_PROCID", "local"),
                          monotonic=time.monotonic(), **fields)), flush=True)


def scalar_record(value):
    """Copy only small scalar/dataclass provenance, never arbitrary objects."""
    if dataclasses.is_dataclass(value):
        return {field.name: scalar_record(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)) and len(value) <= 256:
        return [scalar_record(item) for item in value]
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        return scalar_record(value.item())
    raise TypeError(f"not scalar provenance: {type(value).__name__}")


def array_digest(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def exclusive_write(path, *, arrays=None, record=None):
    """Root-only host I/O; exclusive hard-link commit never overwrites evidence."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if arrays is not None:
                np.savez(stream, **arrays)
            else:
                stream.write((json.dumps(record, sort_keys=True, indent=2,
                                         allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def final_classification(final_h, latest, verdict, max_iter):
    """Authenticate the solver's accepted INPUT, independently of output EQPs."""
    joined = bool(latest and array_digest(final_h) == latest["input_h_sha256"])
    converged = bool(getattr(verdict, "converged", False))
    proper = int(max_iter) > 1
    status = ("CONVERGED_ACCEPTED_INPUT" if proper and joined and converged
              else "ONE_MAP_DIAGNOSTIC" if not proper else "NOT_CONVERGED")
    return dict(status=status, accepted_input_join=joined,
                native_converged=converged, proper_sc_requested=proper,
                join_scope="bitwise original-basis protected input-H digest, not output eigenvalues",
                frame=None if latest is None else latest["frame"],
                reference_ready=False,
                scope="native solver return; driver completion is a separate record")


def describe_native_frame(inputs, bundle, *, source, deck, revision, code_hashes):
    """Read the native SC/config/centroid metadata contract, not any mu faces."""
    import numpy as np
    from common.parallel_transport import fingerprint_from_binding
    binding = inputs.wfn_fingerprint_binding
    if binding is None:
        raise ValueError("native bound WFN identity required; no new payload fingerprint scan")
    basis = inputs.meta.mu_basis
    extra = dict(centroid_canonical_indices=np.asarray(basis.canonical_indices).copy(),
                 centroid_packed_to_canonical=np.asarray(
                     basis.layout.axis.packed_to_canonical).copy(),
                 kpoints_full_frac=np.asarray(inputs.sym.unfolded_kpts).copy())
    full_rows = np.arange(extra["kpoints_full_frac"].shape[0], dtype=np.int64)
    extra["input_loop_full_rows"] = (full_rows if inputs.kstar is None else
                                     np.asarray(inputs.kstar.select(full_rows)).copy())
    config = inputs.config
    record = dict(wfn_fingerprint=fingerprint_from_binding(binding, inputs.wfn),
                  source_path=str(source), deck_sha256=hashlib.sha256(deck.read_bytes()).hexdigest(),
                  source_revision=revision, source_code_sha256=code_hashes,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kgrid=list(inputs.meta.kgrid), qgrid=list(inputs.meta.qgrid),
                  centroid_logical=int(basis.n_logical), centroid_packed=int(basis.n_packed),
                  centroid_indices_sha256=array_digest(extra["centroid_canonical_indices"]),
                  packing_sha256=array_digest(extra["centroid_packed_to_canonical"]),
                  charge_zeta_identity=dict(inputs.charge_zeta_identity or {}),
                  head_correction=str(config.head.correction.value),
                  sc_head_update=str(config.sc.head_update),
                  initial_state_role=str(inputs.initial_state_role),
                  initial_seed_path=str(inputs.initial_seed_path),
                  input_tail_weights="input_tail_z_kn if present; absent means native unit weights",
                  energy_unit="rydberg", occupation_source="actual rotated bundle",
                  large_face_retention=False)
    return record, extra


def install_capture(owner, *, gather, publish, describe, expected_bands=750,
                    metadata_bound_bytes=512 * 1024**2):
    """Hook existing scalar owners; all gathers occur before root-only publish.

    ``publish(name, arrays=...|record=...)`` must be a collective transaction.
    ``describe(inputs, bundle)`` returns (JSON scalar metadata, small host arrays).
    Only digests and filenames survive a map; no returned bundle is captured.
    """
    import numpy as np
    original_rotate = owner.rotate_wavefunctions
    original_map = owner.gw_iteration_map
    original_solve = owner.run_self_consistency
    context, status = {}, {"maps": 0, "latest": None, "solver": None}

    def rotate(*args, **kwargs):
        bundle = original_rotate(*args, **kwargs)
        if not context or args[0] is not context["inputs"].wfns_dft:
            return bundle
        if context.get("captured"):
            raise RuntimeError("more than one scalar rotation in one native SC map")
        mark("sc_input_capture_begin", map_call=context["call"])
        # Materialize on ALL ranks, never inside the serial write callback.
        operands = dict(energy_padded_ry=bundle.enk,
                        occupation_padded=bundle.occ, U_active=args[1])
        advertised = sum(int(np.prod(value.shape)) * int(value.dtype.itemsize)
                         for value in operands.values()) + context["H"].nbytes
        if advertised * 3 > metadata_bound_bytes:
            raise MemoryError("SC frame metadata preflight exceeds declared host bound")
        arrays = {key: np.array(gather(value), copy=True)
                  for key, value in operands.items()}
        arrays["input_H_loop_ry"] = context["H"]
        if context.get("tail_z") is not None:
            arrays["input_tail_z_kn"] = context["tail_z"]
        arrays.update(context["partition_arrays"])
        slices = bundle.slices
        lo = int(slices.b0)
        logical = int(slices.b4_logical) - lo
        if logical != expected_bands:
            raise ValueError(f"SC frame wants {expected_bands} logical bands, got {logical}")
        active = kwargs.get("active_slice", slices.sigma)
        a, b = int(active.start or 0), int(active.stop)
        energy, U = arrays["energy_padded_ry"], arrays["U_active"]
        arrays["energy_ry"] = energy[:, :logical].copy()
        arrays["occupations"] = arrays["occupation_padded"][:, :logical].copy()
        h_full = (U * energy[:, None, a:b]) @ U.conj().swapaxes(-1, -2)
        selector = getattr(context["inputs"], "kstar", None)
        h_loop = h_full if selector is None else np.asarray(gather(selector.select(h_full)))
        mismatch = float(np.max(np.abs(h_loop - context["H"])))
        scale = max(1., float(np.max(np.abs(context["H"]))))
        if mismatch > 1.e-9 * scale:
            raise ValueError(f"captured rotation is not the current input H: {mismatch} Ry")
        metadata, extra = describe(context["inputs"], bundle)
        arrays.update(extra)
        byte_count = sum(value.nbytes for value in arrays.values())
        # Three copies price gather/copy, NPZ staging and dense H authentication.
        if 3 * byte_count > metadata_bound_bytes:
            raise MemoryError("SC frame exceeds declared metadata-only working bound")
        stem = f"frame_{status['maps']:04d}_call_{context['call']:04d}"
        record = dict(schema="native_sc_input_frame_v1", frame=stem,
                      map_call=context["call"], sequence=status["maps"],
                      logical_bands=logical, padded_bands=int(energy.shape[1]),
                      active_slice=[a, b], input_h_sha256=array_digest(context["H"]),
                      energy_sha256=array_digest(arrays["energy_ry"]),
                      occupation_sha256=array_digest(arrays["occupations"]),
                      U_sha256=array_digest(U), reconstructed_h_max_error_ry=mismatch,
                      arrays_bytes=byte_count, host_working_bound_bytes=metadata_bound_bytes,
                      capture_point="native rotate_wavefunctions return, before map consumers",
                      physics_verification=False, **metadata)
        publish(stem + ".npz", arrays=arrays)
        publish(stem + ".json", record=record)
        status["latest"] = {key: record[key] for key in
                            ("frame", "input_h_sha256", "map_call")}
        context["captured"] = True
        mark("sc_input_capture_end", frame=stem, arrays_bytes=byte_count)
        return bundle

    def iteration(state, inputs, *args, **kwargs):
        if context:
            raise RuntimeError("nested SC map capture unsupported")
        if getattr(inputs, "wfns_transverse", None) is not None:
            raise ValueError("reference frame runner supports the scalar native SC map only")
        context.update(inputs=inputs, call=int(state.iteration),
                       H=np.array(gather(state.H_qp_dft), copy=True))
        tail_z = getattr(state, "tail_z_kn", None)
        context["tail_z"] = (None if tail_z is None else
                             np.array(gather(tail_z), copy=True))
        partition = getattr(state, "partition", None)
        context["partition_arrays"] = ({} if partition is None else {
            "input_partition_" + key: np.array(gather(getattr(partition, key)), copy=True)
            for key in ("protected_mask", "in_range_mask")})
        mark("native_sc_map_begin", map_call=context["call"])
        try:
            result = original_map(state, inputs, *args, **kwargs)
            if not context.get("captured"):
                raise RuntimeError("native SC map did not expose its scalar input rotation")
            outputs = result.outputs
            screening = getattr(outputs, "screening", None)
            handle = getattr(screening, "shared_pole", None)
            handle_record = {}
            if isinstance(handle, dict):
                for key in ("path", "identity", "digest"):
                    if key in handle:
                        handle_record[key] = str(handle[key])
            publish(status["latest"]["frame"] + "_map_complete.json", record=dict(
                map_call=context["call"], input_frame=status["latest"]["frame"],
                tail_scissor_fit=scalar_record(getattr(outputs, "tail_scissor_fit", None)),
                screening_model=handle_record,
                iteration_head_present=getattr(screening, "iteration_head", None) is not None,
                output_h_sha256=array_digest(gather(result.H_qp_dft)),
                scope="map output provenance; NOT the accepted input-state label"))
            status["maps"] += 1
            mark("native_sc_map_end", map_call=context["call"])
            return result
        finally:
            context.clear()

    def solve(state, inputs, *args, **kwargs):
        max_iter = int(kwargs.get("max_iter", 1))
        mark("native_sc_solver_begin", max_iter=max_iter)
        final, history = original_solve(state, inputs, *args, **kwargs)
        record = final_classification(gather(final.H_qp_dft), status["latest"],
                                      final.convergence_verdict, max_iter)
        record.update(native_verdict=scalar_record(final.convergence_verdict),
                      map_calls=status["maps"], returned_iteration=int(final.iteration),
                      rms_history_ev=[float(value) for value in history])
        publish("native_solver_return.json", record=record)
        status["solver"] = record
        mark("native_sc_solver_end", status=record["status"])
        return final, history

    owner.rotate_wavefunctions, owner.gw_iteration_map = rotate, iteration
    owner.run_self_consistency = solve

    def restore():
        owner.rotate_wavefunctions, owner.gw_iteration_map = original_rotate, original_map
        owner.run_self_consistency = original_solve

    return status, restore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wall-cap", type=float, default=1140.)
    parser.add_argument("--stack-after", type=float, default=120.)
    parser.add_argument("--expected-bands", type=int, default=750)
    parser.add_argument("--source-revision", required=True,
                        help="immutable source commit authenticated by the external launcher")
    args = parser.parse_args(argv)
    if args.wall_cap <= 0 or args.stack_after <= 0:
        parser.error("positive wall/stack intervals required")
    source, deck, out = (Path(value).resolve() for value in
                         (args.source, args.deck, args.out))
    if out.exists():
        parser.error("--out must be a fresh, nonexistent evidence directory")
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source))
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(args.stack_after, repeat=True)
    done = threading.Event()

    def watchdog():
        if not done.wait(args.wall_cap):
            mark("reference_wall_stop", seconds=args.wall_cap)
            faulthandler.dump_traceback(all_threads=True)
            os._exit(124)

    threading.Thread(target=watchdog, daemon=True, name="SC-reference-wall-cap").start()
    mark("native_driver_import_begin", source=str(source))
    # Import the native entry point FIRST: it initializes its communicator stack
    # before its own JAX import. No control collectives or JAX imports precede it.
    import runtime
    from gw import gw_jax
    import numpy as np
    from common.collectives import gather_to_host, rank0_transaction
    from gw import sc_iteration
    import jax
    for module in (runtime, gw_jax, sc_iteration):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"wrong source closure: {module.__file__}")
    mark("native_driver_import_end", processes=jax.process_count())
    code_hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                   for name in ("src/gw/gw_jax.py", "src/gw/sc_iteration.py",
                                "src/gw/wavefunction_bundle.py")}

    def publish(name, **kwargs):
        path = out / name
        rank0_transaction(path, stage=f"sc_reference_frame/{name}",
                          write=lambda: exclusive_write(path, **kwargs))

    def describe(inputs, bundle):
        return describe_native_frame(inputs, bundle, source=source, deck=deck,
                                     revision=args.source_revision, code_hashes=code_hashes)

    status, restore = install_capture(sc_iteration, gather=gather_to_host,
                                      publish=publish, describe=describe,
                                      expected_bands=args.expected_bands)
    mark("native_gw_cli_begin", source=str(source), deck=str(deck), out=str(out))
    try:
        # No altered driver branch, solver, damping, tolerance or physical inputs.
        result = gw_jax.main(["-i", str(deck)])
        if result not in (0, None):
            raise RuntimeError(f"native GW CLI returned {result!r}")
        record = dict(driver_returncode=0, solver=status["solver"],
                      reference_ready=bool(status["solver"] and
                          status["solver"]["status"] == "CONVERGED_ACCEPTED_INPUT"),
                      scope="native driver completed; no q8/q4 comparison or TD verification")
        publish("driver_complete.json", record=record)
        mark("native_gw_cli_end", reference_ready=record["reference_ready"])
    except BaseException as exc:
        # No new collective on a possibly rank-divergent failure path.
        if jax.process_index() == 0:
            exclusive_write(out / "driver_failed.json", record=dict(
                reference_ready=False, error=repr(exc), scope="native CLI failure"))
        mark("native_gw_cli_failed", error=repr(exc))
        raise
    finally:
        restore()
        done.set()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
