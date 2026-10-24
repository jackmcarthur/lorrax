"""The seconds-long preflight of a driver run, before the mesh, its communicators and any compile.

``runtime.initialize_communicator_stack`` step 5f, after the backend is up (so the device
model and memory are known) and before step 6 builds the mesh.  Every rank evaluates the same
deterministic checks and refuses together (exit 2); rank 0 prints one line per finding:

1. The input file: present, readable, and -- for ``gw.gw_jax`` -- parsed by the driver's own
   strict reader with every cross-key validator (``LorraxConfig.from_input_file``, as
   ``lx doctor --deck`` does), with the wavefunction and centroid files it names present.
2. Memory: the deck's dominant ISDF objects, ``N_k N_mu^2 N_spinor^2 x 16 B`` summed over
   the run (the execute_workflow cost rule), against the device memory of the whole run.
   Above a quarter of it the run is warned (expect OOM); above all of it, refused.
3. The install self-check: ``lorrax-postinstall`` writes the stamp
   ``runtime.install_verification_stamp`` names.  Missing on a runtime image the user
   installed: refused, with the line to run.  On a shared install (another user's prefix) or
   a development checkout: warned (the startup block also says INSTALL NOT VERIFIED).

``LORRAX_PREFLIGHT=warn`` turns refusals into warnings; ``LORRAX_RANK_SETUP=0`` skips this too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .rank_setup import is_driver_process

__all__ = ["run", "deck_findings", "memory_findings", "stamp_findings"]


def _driver_module(argv=None) -> str | None:
    argv = list(getattr(sys, "orig_argv", None) or sys.argv) if argv is None else list(argv)
    for i, arg in enumerate(argv):
        if arg == "-m" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("-m") and len(arg) > 2:
            return arg[2:]
    return None


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"


def deck_findings(deck: str | None, module: str | None):
    """``(refusals, warnings, geometry)``; geometry = (N_k, N_mu, N_spinor) when known."""
    if not deck:
        return [], [], None
    path = Path(deck)
    if not path.is_file() or not os.access(path, os.R_OK):
        return [f"input file {deck} is not a readable file (run from the directory that holds it, "
                f"or pass its path to -i)"], [], None
    if module != "gw.gw_jax":
        return [], [], None
    try:
        from gw.gw_config import LorraxConfig
        config = LorraxConfig.from_input_file(str(path), runtime_platform="gpu",
                                              resolve_hardware=False)
    except SystemExit as exc:
        return [f"input file {deck}: {exc}"], [], None
    except Exception as exc:                                  # noqa: BLE001
        return [f"input file {deck}: {_first_line(exc)}"], [], None
    missing = [f"{role} {p}" for role, p in (("wfn_file", config.paths.wfn_file),
                                              ("centroids_file", config.paths.centroids_file))
               if p and not Path(p).is_file()]
    if missing:
        return [f"input file {deck} names files that do not exist: {'; '.join(missing)}"], [], None
    try:
        from file_io.mf_header import read_mf_header
        from file_io.centroids import load_centroids
        header = read_mf_header(config.paths.wfn_file)
        _, _, n_mu = load_centroids(config.paths.centroids_file, header.fft_grid)
        n_k = int(max(1, int(header.kgrid[0]) * int(header.kgrid[1]) * int(header.kgrid[2])))
        return [], [], (max(n_k, int(header.nkpts)), int(n_mu), int(header.nspinor))
    except Exception as exc:                                  # noqa: BLE001
        return [], [f"memory estimate skipped: {_first_line(exc)}"], None


def memory_findings(geometry, n_devices: int, bytes_per_device: float | None):
    """``(refusals, warnings)`` of the cost rule against the run's device memory."""
    if not geometry or not bytes_per_device or n_devices < 1:
        return [], []
    n_k, n_mu, n_spinor = geometry
    need = n_k * n_mu ** 2 * n_spinor ** 2 * 16.0
    have = n_devices * float(bytes_per_device)
    line = (f"N_k N_mu^2 N_spinor^2 x 16 B = {n_k} x {n_mu}^2 x {n_spinor}^2 x 16 B = "
            f"{need / 1e9:.1f} GB against {n_devices} x {bytes_per_device / 1e9:.1f} GB = "
            f"{have / 1e9:.1f} GB of device memory")
    more = max(1, -(-int(4 * need) // int(bytes_per_device)))   # GPUs that hold it at a quarter
    if need > have:
        return [f"{line}: this run cannot fit; run on at least {more} GPUs "
                f"(srun -n {more}, more nodes)"], []
    if need > have / 4:
        return [], [f"{line}: above a quarter of it, expect OOM; about {more} GPUs fit comfortably"]
    return [], []


def stamp_findings(stamp: Path | None, env=os.environ):
    """``(refusals, warnings)``: has ``lorrax-postinstall`` verified this install on this GPU?"""
    if stamp is None or stamp.is_file():
        return [], []
    prefix = env.get("LORRAX_PREFIX", "")
    line = (f"this install has not passed its self-check on this GPU model (no {stamp}): "
            "cold kernels compile per rank and a slow filesystem goes unnoticed; run "
            "`salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A <project>_g lorrax-postinstall`")
    own = False
    if env.get("LORRAX_RUNTIME") == "image" and prefix:
        try:
            own = Path(prefix).stat().st_uid == os.getuid()
        except OSError:
            own = False
    return ([line], []) if own else ([], [line])


def run(*, backend: str, device_kind: str, n_devices: int, bytes_per_device: float | None,
        process_index: int, stamp_of, say=print) -> None:
    """Step 5f.  ``stamp_of(device_kind)``: ``runtime.install_verification_stamp``."""
    if os.environ.get("LORRAX_RANK_SETUP", "").strip().lower() in ("0", "off", "false", "no"):
        return
    if not is_driver_process():
        return
    from . import _deck_of
    module = _driver_module()
    refuse, warn, geometry = deck_findings(_deck_of(sys.argv[1:]), module)
    gpu = str(backend).lower() in ("gpu", "cuda", "rocm")
    if gpu:
        r, w = memory_findings(geometry, n_devices, bytes_per_device)
        refuse += r
        warn += w
        try:
            stamp = stamp_of(device_kind)
        except Exception:                                     # noqa: BLE001
            stamp = None
        r, w = stamp_findings(stamp)
        refuse += r
        warn += w
    soft = os.environ.get("LORRAX_PREFLIGHT", "").strip().lower() == "warn"
    if process_index == 0:
        for line in warn:
            say(f"[lorrax preflight] WARNING: {line}")
        for line in refuse:
            say(f"[lorrax preflight] {'WARNING' if soft else 'REFUSED'}: {line}")
        if refuse and not soft:
            say("[lorrax preflight] (LORRAX_PREFLIGHT=warn runs anyway)")
    if refuse and not soft:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
