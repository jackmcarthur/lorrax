"""Host-only diagnostic hooks around the original native SC frame runner.

No solver, linalg, cache, array, tolerance, physical input, or retry-policy
changes. Successful maps continue the original proper SC. All-rank GDB and
the launcher deadline remain mandatory; Python cannot catch native faults.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

RUNNER_SHA256 = "31f545b6d5a58d7c8c5d053249f8967dc607191cc8d2684a8ff094117b4a3e3b"
REVISIONS = {
    "1134957c6e63d3d5c8ac141799e0eb17e9423233",
    "a0c1882d1cc579b1670694615bb00f4b53b15b0d",
}
DECK_SHA256 = "8e6df25ebfce57a3dbbc7055cdd4f8c37def5a5e76a1c8b15d42f457bb1c00bf"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_signature(value):
    """Shape metadata only: never read, transfer, or stringify an array."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        sharding = getattr(value, "sharding", None)
        return {"shape": [int(v) for v in value.shape],
                "dtype": str(value.dtype),
                "spec": str(getattr(sharding, "spec", None))}
    if isinstance(value, (list, tuple)):
        return [static_signature(v) for v in value]
    if isinstance(value, dict):
        return {str(k): static_signature(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    return {"type": type(value).__name__}


def install_hooks(plan, execution, log):
    """Retain every original callable/cache; add no device calls/collectives."""
    restore, programs = [], {}

    def wrap(owner, name, describe, returned=None):
        original = getattr(owner, name)
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            metadata = describe(bound.arguments)
            log(name, "BEGIN", **metadata)
            before = time.monotonic()
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                log(name, "PYTHON_EXCEPTION", error=repr(error), **metadata)
                raise
            if returned is not None:
                returned(result, metadata)
            log(name, "END", host_seconds=time.monotonic() - before, **metadata)
            return result

        setattr(owner, name, wrapped)
        restore.append((owner, name, original))

    def stack(a):
        fields = {k: static_signature(a[k]) for k in
                  ("op", "nb", "n", "rounds", "phase", "gathered") if k in a}
        if "dtype" in a:
            fields["dtype"] = str(a["dtype"])
        return fields

    def program(a):
        return {k: static_signature(a[k]) for k in
                ("op", "shape", "dtype", "rounds", "phase", "cut", "gathered")}

    wrap(plan, "_reshard_stack_program", program,
         lambda result, metadata: programs.__setitem__(id(result), metadata.copy()))
    wrap(plan, "_stack_bytes", stack)
    wrap(plan, "_executable", lambda a: {
        "program": programs.get(id(a["program"]), {
            "name": getattr(a["program"], "__name__", type(a["program"]).__name__)}),
        "operands": static_signature(a["args"])})
    wrap(plan, "compiled_bytes", lambda a: {
        "program_name": getattr(a["program"], "__name__", type(a["program"]).__name__),
        "shape": list(a["shape"]), "dtype": str(a["dtype"])})
    wrap(plan.Plan, "_decide_stack", lambda a: dict(stack(a),
        provider=str(a["self"].batched_route), backend=str(a["self"].backend),
        budget_bytes=a["self"].budget_bytes))
    wrap(execution, "face_eigh", lambda a: {
        "n": int(a["n"]), "backend": str(a["backend"]),
        "room": static_signature(a["room"])})
    wrap(execution, "face_reduce_round", lambda a: {
        "states": static_signature(a["states"]),
        "tables": static_signature(a["tables"]),
        "ordered": bool(a["ordered"]), "keep_budget": static_signature(a["keep_budget"]),
        "carrier": static_signature(a["carrier"]), "room": static_signature(a["room"])})
    wrap(execution, "face_compiled", lambda a: {
        "operands": static_signature(a["args"]), "room": static_signature(a["room"])})

    def uninstall():
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)
    return uninstall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-revision", required=True, choices=sorted(REVISIONS))
    parser.add_argument("--runner", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--out", required=True, help="fresh original-runner output directory")
    parser.add_argument("--log-prefix", required=True, help="fresh shared parent/rank JSONL prefix")
    parser.add_argument("--wall-cap", type=float, default=480.)
    parser.add_argument("--stack-after", type=float, default=86400.,
                        help="archived runner requires >0; default exceeds the <=2h wrapper cap")
    args = parser.parse_args()
    if not 0 < args.wall_cap <= 7200 or args.stack_after <= 0:
        parser.error("positive wall cap <=7200 seconds and stack interval required")
    if os.environ.get("LEG_HEAD") != args.source_revision:
        raise ValueError("launcher LEG_HEAD/source revision mismatch")
    if digest(args.runner) != RUNNER_SHA256 or digest(args.deck) != DECK_SHA256:
        raise ValueError("only the original frame runner and exact q8-v2 deck are approved")
    source = Path(args.source).resolve()
    rank = int(os.environ["SLURM_PROCID"])
    prefix = Path(args.log_prefix).resolve()
    if prefix.is_relative_to(Path(args.out).resolve()):
        raise ValueError("--log-prefix must be outside the original runner's fresh --out directory")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive, per-rank files: no collective I/O or same-file rank race.
    stream = open(str(prefix) + f".rank{rank:02d}.jsonl", "x", buffering=1)
    started, counter = time.monotonic(), 0

    def log(stage, state, **metadata):
        nonlocal counter
        counter += 1
        record = dict(stage=stage, state=state, rank=rank, sequence=counter,
                      elapsed_seconds=time.monotonic() - started, **metadata)
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        print("SC_NATIVE_DEBUG " + line, flush=True)

    def expired():
        log("outer_wall_guard", "EXPIRED", wall_cap=args.wall_cap)
        os._exit(124)

    timer = threading.Timer(args.wall_cap, expired)
    timer.daemon = True
    timer.start()
    restore = None
    try:
        log("bootstrap", "BEGIN", source=str(source), revision=args.source_revision,
            harness_sha256=digest(__file__), runner_sha256=digest(args.runner),
            deck_sha256=digest(args.deck), archive_sha256=os.environ.get("LEG_SOURCE_ARCHIVE_SHA256"),
            scope="unchanged native proper SC plus static host logging; no acceptance claim")
        sys.path.insert(0, str(source / "src"))
        sys.path.insert(0, str(source))
        # Native bootstrap FIRST, before NumPy/JAX/distrib_la imports.
        from gw import gw_jax
        import jax
        plan = importlib.import_module("distrib_la.plan")
        execution = importlib.import_module("gw.shared_pole_execution")
        for module in (gw_jax, plan, execution):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError("wrong source origin: " + module.__file__)
        if (jax.process_count() != 16 or jax.local_device_count() != 1
                or tuple(gw_jax.RUNTIME.mesh.devices.shape) != (4, 4)
                or any(d.platform != "gpu" for d in jax.devices())):
            raise ValueError("actual P16 GPU/4x4 mesh required")
        log("bootstrap", "END", processes=16, mesh=[4, 4],
            plan_sha256=digest(plan.__file__), execution_sha256=digest(execution.__file__))
        restore = install_hooks(plan, execution, log)
        spec = importlib.util.spec_from_file_location("original_sc_frame_runner", args.runner)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        log("original_frame_runner", "BEGIN", successful_maps_continue=True)
        runner.main(["--source", str(source), "--source-revision", args.source_revision,
                     "--deck", args.deck, "--out", args.out,
                     "--wall-cap", str(max(.001, args.wall_cap - (time.monotonic() - started))),
                     "--stack-after", str(args.stack_after), "--expected-bands", "750"])
        log("original_frame_runner", "END",
            reference_status="use original native solver-return/driver-complete receipts, not this marker")
    except BaseException as error:
        print(traceback.format_exc(), flush=True)
        log("wrapper", "PYTHON_EXCEPTION", error=repr(error))
        raise
    finally:
        if restore is not None:
            restore()
        timer.cancel()
        stream.close()


if __name__ == "__main__":
    main()
