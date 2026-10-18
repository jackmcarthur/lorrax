"""Coordinate child-driver fixtures without initializing MPI/JAX in pytest.

Plain srun starts one pytest per rank; xdist starts independent workers. Only
srun ranks share these rendezvous. Child drivers retain their launch environment
and own the MPI/JAX runtime. Work directories live outside pytest's basetemp,
which pytest is allowed to delete independently in each process, and outside
the source tree, which is read-only in a module install: :func:`run_root` is
the per-user cache tree ``$SCRATCH/.cache/lorrax/hsuite``, which every rank
resolves alike.  ``SCRATCH`` is resolved by the runtime's own site rule
(:func:`runtime.apply_site_scratch`: a caller's export wins, else the site
profile's ``[paths] scratch``, e.g. Frontier's ``$MEMBERWORK/<account>``)
before the directory is chosen; with neither, the suite refuses rather than
fall back to HOME, where the drivers' HDF5 files fail (Frontier NFS:
``[Errno 524] Unable to synchronously open file``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from ffi import _services
from runtime import _resolve_proc_count, _resolve_proc_id, apply_site_scratch, site_scratch

_services.ensure_on_path()
from lxkit import user_cache_dir  # noqa: E402

_ROOT = None
_SEQUENCE = 0


def run_root() -> Path:
    """The suite's run and rendezvous root, ``$SCRATCH/.cache/lorrax/hsuite``.

    ``SCRATCH`` comes from the same resolution the runtime applies at start-up
    (:func:`runtime.apply_site_scratch`; idempotent, a caller's export wins), run
    here because the harness picks its directories before any driver starts.
    Refuses when there is still no ``SCRATCH``: the run directories hold the
    drivers' HDF5 files, and HOME is not a filesystem they work on.
    """
    global _ROOT
    if _ROOT is None:
        apply_site_scratch()
        if not os.environ.get("SCRATCH", "").strip():
            _, why = site_scratch()
            raise RuntimeError(
                "GATE hsuite_run_root: got: no scratch root for the driver suite's run "
                f"directories (SCRATCH is unset and {why}); want: SCRATCH, or a site "
                "profile [paths] scratch rule that expands to a writable directory; why: "
                "the run directories hold the drivers' HDF5 files, which fail on a home "
                "filesystem (Frontier NFS: [Errno 524] Unable to synchronously open file). "
                "Export SCRATCH=<a writable directory on the site's scratch filesystem>, "
                "or declare [paths] scratch in config/<site>/site.toml.")
        _ROOT = user_cache_dir("hsuite")
    return _ROOT


def __getattr__(name):
    # ``rank_session.ROOT``: the same root, resolved on first use (PEP 562).
    if name == "ROOT":
        return run_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def exchange(value, *, timeout=240):
    """Gather JSON values and return all ranks' values to every participant."""
    global _SEQUENCE
    count, rank = _resolve_proc_count(), _resolve_proc_id()
    if count == 1:
        return [value]
    assert not os.environ.get("PYTEST_XDIST_WORKER"), "do not nest xdist in MPI pytest"
    job = os.environ["SLURM_JOB_ID"]
    step = os.environ["SLURM_STEP_ID"]
    _SEQUENCE += 1
    key = hashlib.sha256(
        f"{job}:{step}:{_SEQUENCE}".encode()).hexdigest()
    root = run_root()
    root.mkdir(parents=True, exist_ok=True)
    address_file = root / f"{key}.json"
    deadline = time.monotonic() + timeout
    if rank == 0:
        with socket.socket() as listener:
            listener.bind(("0.0.0.0", 0))
            listener.listen(count)
            listener.settimeout(timeout)
            address = [socket.gethostname(), listener.getsockname()[1]]
            temporary = address_file.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(address))
            temporary.replace(address_file)
            streams = []
            sockets = []
            values = {0: value}
            try:
                while len(values) < count:
                    conn, _ = listener.accept()
                    sockets.append(conn)
                    conn.settimeout(timeout)
                    stream = conn.makefile("rw")
                    streams.append(stream)
                    peer, payload = json.loads(stream.readline())
                    assert peer not in values and 0 < peer < count
                    values[peer] = payload
                result = [values[i] for i in range(count)]
                for stream in streams:
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                return result
            finally:
                for stream in streams:
                    stream.close()
                for conn in sockets:
                    conn.close()
                address_file.unlink(missing_ok=True)
    while time.monotonic() < deadline:
        try:
            address = json.loads(address_file.read_text())
            conn = socket.create_connection(tuple(address), timeout=1)
        except (OSError, ValueError):
            time.sleep(0.05)
            continue
        with conn:
            conn.settimeout(timeout)
            with conn.makefile("rw") as stream:
                stream.write(json.dumps([rank, value]) + "\n")
                stream.flush()
                return json.loads(stream.readline())
    raise TimeoutError(f"core rank {rank} timed out waiting for {address_file}")


def stage(source, prepare):
    """Copy and prepare once, then publish the same absolute path to all ranks."""
    result = None
    if _resolve_proc_id() == 0:
        try:
            root = run_root()
            root.mkdir(parents=True, exist_ok=True)
            target = Path(tempfile.mkdtemp(prefix=source.name + "-", dir=root)) / source.name
            result = {"path": str(prepare(source, target))}
        except Exception as exc:
            result = {"error": repr(exc)}
    result = exchange(result)[0]
    assert "error" not in result, result
    target = Path(result["path"])
    assert target.is_dir(), f"staged fixture not visible: {target}"
    return target


def completed(result):
    """Wait for every child, propagate any failure, and share rank-zero stdout."""
    results = exchange({"returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
    for rank, row in enumerate(results):
        assert row["returncode"] == 0, (
            f"driver rank {rank} failed\n{row['stdout'][-6000:]}\n{row['stderr'][-6000:]}"
        )
    result.stdout = results[0]["stdout"]
    result.stderr = results[0]["stderr"]
    return result


def run_child(call, log_path):
    """Retain per-rank child evidence, including timeout output, before joining."""
    try:
        result = call()
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        result = subprocess.CompletedProcess(
            exc.cmd, 124, text(exc.stdout), text(exc.stderr) + f"\n{exc}")
    except OSError as exc:
        result = subprocess.CompletedProcess([], 127, "", repr(exc))
    path = Path(f"{log_path}.rank{_resolve_proc_id()}.log")
    path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
    return completed(result)
