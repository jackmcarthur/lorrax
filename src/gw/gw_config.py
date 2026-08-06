"""Unified configuration for LORRAX GW calculations.

``LorraxConfig`` is built once via :meth:`LorraxConfig.from_input_file`
from the ``[cohsex]`` section of ``cohsex.in`` and threaded through the
entire driver.  Its ~80 input keys are grouped into sub-dataclasses
along the same axes the input file's section comments already use:

    config.head        — q→0 Coulomb-head sources & overrides
    config.minimax     — screening-minimax target error / max nodes / table mode
    config.ppm         — PPM model + sigma quadrature + on-shell σ_c options
    config.sigma_grid  — ω-grid for Σ_c(ω) output
    config.sc          — self-consistency loop knobs (qp_solver = self_consistent)
    config.memory      — chunk sizing
    config.backend     — FFI/IO backend selection (slab_io / gspace_io / w_dyson_solver)
    config.debug       — debug-only flags & file paths
    config.bse         — BSE interpolation setup (htransform-driven)
    config.paths       — output filenames

The top-level ``LorraxConfig`` retains only system geometry
(``nval`` / ``ncond`` / ``nband`` / ``sys_dim``) and the orthogonal
mode flags (``compute_mode`` / ``qp_solver`` / etc.) that the
driver reads on the fast path.

Derived sub-objects (the math-internal ``MinimaxConfig`` from
``minimax_config.py``, one instance per quadrature consumer) and derived
data (the Σ_c(ω) grid) are constructed on demand via ``LorraxConfig``
properties.
"""

from __future__ import annotations

import configparser
import enum
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.units import RYD_TO_EV


# ---------------------------------------------------------------------------
#  Environment grammar — ONE boolean vocabulary for the GW init/config layer
# ---------------------------------------------------------------------------
#
# THIS IS THE CANONICAL COPY.  The tree still carries the same recognised
# token set in three other places, each with a reason:
#
#   * ``ffi/common/gate.py::MODE_SPELLINGS`` — three-valued (auto/off/on).
#     Its ``auto`` is load-bearing (a gate may DEMOTE and say so), so it is
#     deliberately NOT folded into a two-valued test; only its on/off halves
#     are the same vocabulary as ours.
#   * ``runtime.__init__._FALSY_TOKENS`` — the sanctioned two-valued
#     falsy-test resolver (the two-resolver doctrine,
#     docs/architecture/layers.md): ``runtime`` must resolve knobs BEFORE
#     jax/config imports are safe, so it keeps its own tiny parser.
#   * ``file_io/_slab_io_mpi_host.py::_env_flag`` — mirrors the C++
#     writer's ``env_flag`` (``ffi/cpp/phdf5/context.cc``) so the TWO
#     phdf5 writers stay one grammar; jax-free file, kept local.
#
# ``isdf/core.py``'s ``_env_bool`` — historically the fourth copy — was
# retired by the P1.3 unification (2026-07-31): ``isdf.core`` now imports
# :func:`env_bool` from here (L1→L1; ``gw/__init__`` pulls only this
# jax-free module, so the import adds nothing and cannot cycle).  The
# remaining copies are pinned set-equal by
# ``tests/test_env_grammar.py::test_defect3_vocabulary_has_not_drifted``,
# which reads them straight out of the source text, without importing jax,
# and fails on any drift.
#
# SEMANTICS:
#   unset or blank      -> the caller's default
#   a truthy spelling   -> True        (case- and whitespace-insensitive)
#   a falsy spelling    -> False
#   anything else       -> False, AND announced once (see ``env_bool``)
#
# Resolving an unrecognised token to something other than False would
# split the grammar between converted and unconverted readers of the same
# knob; adding telemetry does not.

_ENV_TRUE = ("1", "on", "true", "yes")
_ENV_FALSE = ("0", "off", "false", "no")

#: (name, raw value) pairs already announced, so a knob read once per
#: r-chunk cannot spam a multi-hour log.
_ENV_ANNOUNCED: set = set()


def reset_env_announce_state() -> None:
    """Forget which grammar errors have been announced (tests only)."""
    _ENV_ANNOUNCED.clear()


def env_bool(name: str, default: bool, *, print_fn=print) -> bool:
    """Canonical boolean env parse for the GW init/config layer.

    Parameters
    ----------
    name : str
        Environment variable, e.g. ``"LORRAX_MEM_DEBUG"``.
    default : bool
        Value when the variable is unset or blank.  This is the knob's
        DOCUMENTED default, not a guess — ``docs/dev/env_vars.md`` is the
        table it has to agree with.

    Notes
    -----
    The three bugs this replaces, all in the four files this layer owns:

    * ``if os.environ.get("LORRAX_EXIT_AFTER_ZETA"):`` — a bare presence
      test, so ``=0`` ended a production run with ``SystemExit(0)``;
    * ``not in ("0", "off", "false")`` — case-SENSITIVE, so
      ``LORRAX_MALLOC_TRIM=OFF`` left the trim hook on while the
      documented sibling ``LORRAX_MALLOC_TUNE=OFF`` correctly turned off;
    * ``bool(os.environ.get(...))`` — same presence-test class.

    An unrecognised token resolves False (see the module comment) but is
    ANNOUNCED, once per (name, value), with the project's grep-able
    ``*** LORRAX SANITY`` marker.  Silently resolving a typo in either
    direction is the failure mode this whole helper exists to remove.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    tok = raw.strip().lower()
    if tok in _ENV_TRUE:
        return True
    if tok in _ENV_FALSE:
        return False
    key = (name, raw)
    if key not in _ENV_ANNOUNCED:
        _ENV_ANNOUNCED.add(key)
        print_fn(
            f"  *** LORRAX SANITY: {name}={raw!r} is not a recognised "
            f"boolean.  Accepted: {'/'.join(_ENV_TRUE)} (on), "
            f"{'/'.join(_ENV_FALSE)} (off), unset/blank (default="
            f"{'on' if default else 'off'}).  Treating it as OFF. ***")
    return False


def env_float(name: str, default: float, *, print_fn=print,
              refuse: bool = False) -> float:
    """Canonical numeric env parse: unset/blank → default, bad → ANNOUNCE
    (or, with ``refuse=True``, RAISE).

    The same defect class as :func:`env_bool`, one type along.  A
    ``try: float(...) except: default`` leaves the user believing a knob is
    in force when it is not — the exact failure this file's
    ``ISDF_ZCT_STAGE_CAP_GB`` handler already carries a comment about
    ("an OOM later, with no clue"), and which its sibling
    ``ISDF_CHUNK_TARGET_UTILIZATION`` was still committing.

    ``refuse=True`` is for knobs that GATE correctness rather than tune
    performance (``LORRAX_FH_ORTHO_TOL``): running with the default while
    the user believes a gate threshold is in force is itself the silent
    failure, so garbage refuses loudly, naming the variable — the
    announce-or-refuse doctrine's refuse half.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        if refuse:
            raise ValueError(
                f"{name}={raw!r} is not a number.  Accepted: a float "
                f"(e.g. '1e-6'), or unset/blank for the default "
                f"({default!r}).  Refusing rather than running with a "
                f"value the caller did not choose.") from None
        key = (name, raw)
        if key not in _ENV_ANNOUNCED:
            _ENV_ANNOUNCED.add(key)
            print_fn(f"  *** LORRAX SANITY: {name}={raw!r} is not a number; "
                     f"falling back to {default}.  The knob is NOT in "
                     f"force. ***")
        return default


def resolve_zct_stage_cap(cap_raw, frac_raw, *, per_device_gb: float,
                          total_gb: float, print_fn=print):
    """Resolve the ζCᵀ stage cap in GB, or ``None`` when no cap applies.

    Pure — the caller supplies ``total_gb`` (the physical card, ``0.0``
    when there is no device to take a fraction of, i.e. the CPU backend).

    Every path that ends in "no cap" says why.  The CPU path used to be
    silent: the fraction branch sat behind
    ``jax.default_backend() in ("gpu", "cuda")``, so a user who exported
    ``ISDF_ZCT_STAGE_CAP_FRAC`` on a CPU run got no cap, no clamp and no
    message — indistinguishable from the knob working.
    """
    if cap_raw and str(cap_raw).strip():
        try:
            return min(float(per_device_gb),
                       max(0.0, float(cap_raw)))
        except ValueError:
            # Deliberately does NOT fall through to the _FRAC branch: an
            # explicit absolute cap that cannot be parsed is a user error,
            # and quietly substituting a different knob's answer for it
            # would hide the typo behind a plausible number.
            print_fn(f"  *** LORRAX SANITY: ISDF_ZCT_STAGE_CAP_GB="
                     f"{cap_raw!r} is not a number; NO stage cap is in "
                     f"force (ISDF_ZCT_STAGE_CAP_FRAC is not consulted as "
                     f"a fallback for a malformed explicit cap). ***")
            return None
    if not (frac_raw and str(frac_raw).strip()):
        return None
    if float(total_gb) <= 0:
        print_fn(f"  *** LORRAX SANITY: ISDF_ZCT_STAGE_CAP_FRAC="
                 f"{frac_raw!r} is a fraction of the DEVICE's physical "
                 f"memory, and this backend reports none (total_gb=0) — "
                 f"so NO stage cap is in force.  Use "
                 f"ISDF_ZCT_STAGE_CAP_GB for an absolute cap. ***")
        return None
    try:
        frac = max(0.10, min(0.95, float(frac_raw)))
    except ValueError:
        print_fn(f"  *** LORRAX SANITY: ISDF_ZCT_STAGE_CAP_FRAC="
                 f"{frac_raw!r} is not a number; the stage cap is "
                 f"NOT set. ***")
        return None
    return min(float(per_device_gb), frac * float(total_gb))


# ---------------------------------------------------------------------------
#  ζ-truncating env knobs
# ---------------------------------------------------------------------------
#: Env knobs that make the ζ fit stop EARLY and still write a file.
#:
#: ``LORRAX_MAX_RCHUNKS=N`` breaks the r-chunk loop after N chunks
#: (``gw/isdf_fitting.py``), and the writer downstream of the loop still
#: calls ``mark_zeta_done`` — so the truncated ζ is stamped complete.  If
#: ``gw_init`` then stamps ``fit_provenance`` on it, ``_zeta_reuse_ok``
#: will REUSE that ζ in a later production run from the same directory,
#: because provenance records the *configuration*, which is identical.
#: The result is silently wrong physics from a profiling knob.
ZETA_TRUNCATING_ENV_KNOBS = ("LORRAX_MAX_RCHUNKS",)


def active_zeta_truncating_knobs() -> list[tuple[str, str]]:
    """``[(name, raw), ...]`` for every truncating knob currently in force.

    Blank counts as unset (the r-chunk loop's own guard is
    ``if _max_rchunks and ...``, so ``""`` does not truncate).
    """
    out = []
    for name in ZETA_TRUNCATING_ENV_KNOBS:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            out.append((name, raw))
    return out


# ---------------------------------------------------------------------------
#  XLA GPU memory environment — RE-EXPORTED, not defined here
# ---------------------------------------------------------------------------
#
# These four names used to be defined in this file, ~280 lines of them, and
# ``runtime.collect_startup_facts`` imported them from here — an L3 module
# reaching up into the GW driver package for something that knows nothing
# about GW.  They now live in :mod:`runtime.xla_memory`, next to
# ``runtime.set_default_env``, which is the code that decides which of these
# variables LORRAX ships.  See that module's docstring for the four traps it
# encodes and the measurements behind them (jobs 7882443 / 7882447).
#
# The re-export is not a compatibility shim to be swept away: ``gw_init``
# captions the ζ-fit peak and ``gw_output`` prints the startup banner from
# these, and reading them as ``gw_config.<name>`` is how those call sites and
# ``tests/test_env_grammar.py`` are written.  Keeping the alias here costs one
# import and keeps the deck-level vocabulary in one place.
#
# ``runtime`` imports jax only inside function bodies, so this import does NOT
# cost gw_config its jax-free property (the login-node config tests and
# ``gw_output``'s banner both depend on that).
from runtime.xla_memory import (       # noqa: F401
    XlaGpuMemoryEnv,
    XlaPoolReading,
    classify_xla_pool,
    resolve_xla_gpu_memory_env,
)


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class ComputeMode(str, enum.Enum):
    """The single axis describing what self-energy is computed.

    Orthogonal to ``qp_solver`` (how QP energies are extracted from Σ):
    any mode can be wrapped in the ``self_consistent`` QSGW loop — the
    loop dispatches through the mode-agnostic
    ``sigma_dispatch.compute_sigma_xc`` (COHSEX and GN-PPM verified
    end-to-end; see reports/gw_refactor_map_2026-07-01/
    G0W0_SC_TOGGLE_DESIGN.md §4).

    - ``X_ONLY`` — bare exchange Σ_X = -G·V (no screening, no correlation).
    - ``COHSEX`` — static screened-exchange + Coulomb-hole.
    - ``GN_PPM`` — dynamic Σ_c(ω) via GN plasmon-pole (probe at iω_p).
    - ``HL_PPM`` — dynamic Σ_c(ω) via HL plasmon-pole (probe at real Ω).
    """

    X_ONLY = "x_only"
    COHSEX = "cohsex"
    GN_PPM = "gn_ppm"
    HL_PPM = "hl_ppm"

    @property
    def needs_screening(self) -> bool:
        """True for COHSEX / GN-PPM / HL-PPM; False for bare X."""
        return self is not ComputeMode.X_ONLY

    @property
    def is_dynamic(self) -> bool:
        """True for GN-PPM / HL-PPM; False for static modes."""
        return self in (ComputeMode.GN_PPM, ComputeMode.HL_PPM)

    @property
    def ppm_model(self) -> str | None:
        """``'gn'`` for GN-PPM, ``'hl'`` for HL-PPM, else None."""
        return {
            ComputeMode.GN_PPM: "gn",
            ComputeMode.HL_PPM: "hl",
        }.get(self)


class QPSolver(str, enum.Enum):
    """How QP energies are extracted from Σ — orthogonal to ``compute_mode``.

    The three states are mutually exclusive answers to the same physics
    question, each naming a standard method:

    - ``ONE_SHOT_DFT`` — textbook G0W0 (THE DEFAULT).  Σ is built once
      from the DFT inputs and *everything* is evaluated at E_DFT: the
      eqp0/eqp1 text outputs (at-DFT Newton + Z-linearization, as always)
      AND the QSGW-symmetrised Σ_xc whose eigh produces ``E_qp_ry`` /
      ``qp_wfn_rotations.h5`` / ``WFN_qp.h5``.  No iteration of any kind.
    - ``FIXED_POINT`` — one-shot Σ + diagonal on-shell solve
      E = h0 + ReΣ(E) for the QSGW-build evaluation energies
      (eigenvalue-only; Σ is never rebuilt).  Dynamic modes only — static
      Σ has no ω-grid to solve on.  ``ppm.sigma_at_dft_extrapolate`` is a
      sub-knob of this state (scissor for out-of-grid bands).
    - ``SELF_CONSISTENT`` — full QSGW loop (:mod:`gw.sc_iteration`):
      Σ rebuilt each iteration from rotated ψ + the previous iteration's
      E.  Loop knobs live in :class:`SCConfig` (``config.sc``).

    eqp0.dat / eqp1.dat keep the same formula in all three states; only
    the provenance of Σ changes under ``SELF_CONSISTENT`` (converged Σ,
    still evaluated at E_DFT — one more at-DFT Newton step from the SC
    fixed point).
    """

    ONE_SHOT_DFT = "one_shot_dft"
    FIXED_POINT = "fixed_point"
    SELF_CONSISTENT = "self_consistent"


class SlabIOBackend(str, enum.Enum):
    """How big sigma/zeta/restart HDF5 files are written.

    - ``PHDF5_FFI`` — every rank writes its hyperslab via the parallel-HDF5
      FFI (collective MPI-IO).  Default on BOTH backends now: the C++ core
      (``ffi/cpp/phdf5/write_ffi.cc``) compiles into the CUDA lib and, since
      workstream AE, into the CUDA-free host lib under ``LORRAX_FFI_NO_CUDA``
      — where the D2H staging collapses to "H5Dwrite reads the XLA buffer in
      place".  ~5× faster than the rank-0 path once Lustre striping is
      applied.  Needs the host lib to export ``PhdfWriteHostFfi``
      (``ffi_loader.has_phdf5_write('cpu')``); the auto-router probes.
    - ``PHDF5_HOST`` — host-side equivalent driven by mpi4py +
      h5py(parallel) instead of the FFI.  Same per-rank collective MPI-IO
      write semantics.  Now the SECOND tier on CPU: it needs an extra
      environment (the ``lorrax_env_mpi_overlay`` two-package overlay,
      workstream AB) that the FFI path does not, so it is kept only as the
      fallback for a host lib without the write handler.
    - ``H5PY_ALLGATHER`` — gather to rank 0 and write via serial h5py.
      Last-resort fallback for systems without either parallel HDF5 or
      the FFI.  Slow at scale (rank-0 disk bandwidth bottleneck), and the
      gather itself is the single biggest collective in a large run
      (12.05 GB at 606 centroids / P=16 — scorecard AB.2).
    """
    PHDF5_FFI = "phdf5_ffi"
    PHDF5_HOST = "phdf5_host"
    H5PY_ALLGATHER = "h5py_allgather"


#: PMI/PMIx variable spellings a launcher (srun --mpi=pmi2/pmix,
#: mpiexec.hydra, mpirun) leaves in the environment.  Prefix-matched so the
#: versioned PMIx names (PMIX_SERVER_URI21, ...) are covered.  Plain SLURM
#: batch variables (SLURM_JOB_ID etc.) are deliberately NOT in this list: a
#: bare ``python`` inside an sbatch allocation has all of those and still no
#: PMI server to register with — that is exactly the failing launch shape.
_MPI_LAUNCHER_ENV_PREFIXES = ("PMI_", "PMIX_")
_MPI_LAUNCHER_ENV_VARS = ("HYDI_CONTROL_FD",)


def _mpi_launcher_env() -> "str | None":
    """The first launcher PMI/PMIx variable present, else None."""
    for name in _MPI_LAUNCHER_ENV_VARS:
        if os.environ.get(name):
            return name
    for name in sorted(os.environ):
        if name.startswith(_MPI_LAUNCHER_ENV_PREFIXES):
            return name
    return None


def _mpi_singleton_probe(child_code: str, what: str,
                         argv_extra=(), timeout_s: float = 60.0
                         ) -> tuple[bool, str]:
    """``(ok, reason)`` — run ``child_code`` in a THROWAWAY subprocess.

    Why a subprocess: on a bare launch (no PMI environment) whether
    ``MPI_Init_thread`` works as a singleton is a property of the MPI
    stack, and on the production stack it does not fail catchably — Intel
    MPI 2020 calls ``abort()`` inside ``MPIR_pmi_init`` (job 7884926), so
    an in-process probe kills the run it was meant to protect.  The child
    runs exactly the init the candidate tier would run; the router
    survives the child's death.  Only reached on the bare-launch path, so
    the ~1 s child never costs a production srun/mpirun start anything.

    Never raises.  A hung child (the init blocking rather than aborting)
    is killed at ``timeout_s`` and reported as not bootstrappable — the
    demotion direction is the safe one.
    """
    import subprocess
    import sys
    try:
        res = subprocess.run(
            [sys.executable, "-c", child_code, *argv_extra],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, (f"the singleton {what} probe hung for {timeout_s:.0f} s "
                       f"and was killed")
    except Exception as exc:                          # pragma: no cover
        return False, (f"the singleton {what} probe could not run "
                       f"({type(exc).__name__}: {exc})")
    if res.returncode == 0:
        return True, f"singleton {what} succeeded in a probe subprocess"
    return False, (f"singleton {what} exited rc={res.returncode} in a probe "
                   f"subprocess (Intel MPI aborts in MPIR_pmi_init on a "
                   f"PMI-less launch)")


def _slab_io_geometry() -> str:
    """The run geometry, as a one-line fragment for the routing banner.

    Printed WITH every routing decision, on purpose.  No archived
    multi-node log records a node count and the routing banner postdates
    every historical run, so settling "does phdf5 work across nodes"
    needed an archaeology pass rather than a grep.  The geometry belongs
    next to the decision it qualifies.

    SLURM's own node count is reported when present but is NOT the
    primary source, and the spelling matters.  Measured inside the
    Shifter container on Perlmutter, 2026-08-05, ``srun -N 4 -n 16``:
    ``SLURM_NNODES=4``, ``SLURM_NTASKS=16`` and ``SLURM_JOB_NODELIST``
    are all present, while ``SLURM_JOB_NUM_NODES`` is ABSENT — it is a
    batch/allocation-level variable, not a step-level one, so anything
    reading only that spelling sees nothing on a plain ``srun`` step.
    JAX's own process/device counts do not depend on the launcher's
    vocabulary at all, so they come first.
    """
    import jax as _jax
    parts = []
    try:
        parts.append(f"processes={_jax.process_count()}")
        parts.append(f"devices={_jax.device_count()}")
        parts.append(f"local_devices={_jax.local_device_count()}")
    except Exception:                                 # pragma: no cover
        parts.append("jax process/device counts unavailable")
    for _k in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES", "SLURM_NTASKS"):
        _v = os.environ.get(_k)
        if _v:
            parts.append(f"{_k}={_v}")
    return ", ".join(parts)


def _probe_mpi_bootstrap_ffi(platform: str = "cpu") -> tuple[bool, str]:
    """``(bootstrappable, how)`` for the PHDF5_FFI tier's own MPI init.

    ``platform`` is an :mod:`ffi.common.ffi_loader` platform key —
    ``"cpu"`` (the host lib) or ``"CUDA"``.  Both routers use this; the
    GPU one did not until 2026-08-05, because it declined the tier on
    node count instead of probing it.

    A launcher PMI environment settles it — that is the environment every
    green multi-rank run had (CLAIMS rows 3, 17).  Without one, probe the
    exact call the tier would make (``lrx_phdf5_init_mpi`` in the loaded
    .so) in a throwaway subprocess; see :func:`_mpi_singleton_probe`.

    NOTE what this does and does not prove.  A PMI environment proves a
    launcher registered this process; it does NOT prove the PMI flavour
    matches the MPI library.  A mismatch (``srun --mpi=pmi2`` against
    Shifter's Cray MPICH) yields singleton MPI — every rank sees
    ``MPI_Comm_size()==1`` — which no probe here detects and which
    independent-I/O writes survive silently.  ``--mpi=cray_shasta`` is
    the launcher's responsibility; see docs/architecture/slab_io.md.
    """
    var = _mpi_launcher_env()
    if var is not None:
        return True, f"launcher PMI environment present ({var})"
    try:
        from ffi.common.ffi_loader import loaded_lib_path
        so = loaded_lib_path(platform)
    except Exception as exc:                          # pragma: no cover
        return False, f"ffi_loader unavailable ({type(exc).__name__}: {exc})"
    if not so:
        return False, (f"no loaded {platform} FFI library to probe MPI init "
                       f"with")
    child = ("import ctypes, sys\n"
             "lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)\n"
             "lib.lrx_phdf5_init_mpi()\n")
    return _mpi_singleton_probe(child, f"MPI_Init_thread ({platform} FFI)",
                                argv_extra=(so,))


def _probe_phdf5_host_tier() -> tuple[bool, str]:
    """``(usable, reason)`` for the ``PHDF5_HOST`` (mpi4py + h5py-parallel)
    tier, testing exactly what a ``_MpiHostBackend`` will execute.

    ``import mpi4py`` alone proves nothing: the package import succeeds
    on a broken PMI stack and the run then dies at the first SlabIO open
    with ``MPI_Init_thread() failed [error code: 16]`` (a PMI flavour
    mismatch between the launcher and the MPI library).  So the probe
    does the real thing — ``from mpi4py import MPI`` — which runs
    ``MPI_Init_thread``.  The init cost is not wasted: the selected
    backend needed it anyway (same rationale as ``phdf5_init_mpi`` in
    ``gw_jax.configure_run``).

    Never raises: an init failure is a *probe* result, not a run killer
    — the router demotes and says why.  An explicit
    ``slab_io=phdf5_host`` request bypasses this probe and keeps
    fails-loudly semantics at the first SlabIO open.

    BEFORE the in-process ``MPI_Init_thread``: on a bare launch (no PMI
    environment) that init does not fail catchably — Intel MPI aborts the
    whole process inside ``MPIR_pmi_init`` (job 7884926), which would turn
    this probe into the crash it exists to prevent.  So a PMI-less launch
    first proves singleton init in a throwaway subprocess and demotes when
    the child dies.
    """
    if _mpi_launcher_env() is None:
        ok, how = _mpi_singleton_probe(
            "from mpi4py import MPI\n", "MPI_Init_thread (mpi4py)")
        if not ok:
            return False, (
                f"bare launch (no PMI/PMIx variables in the environment) "
                f"and {how} — an in-process mpi4py init would abort, not "
                f"raise")
    try:
        from mpi4py import MPI  # noqa: F401 — runs MPI_Init_thread
    except (Exception, SystemExit) as exc:
        # MPI init failures can be raw SystemExit — but NOT BaseException:
        # that also swallowed KeyboardInterrupt, turning a Ctrl-C during
        # config parse into a silent tier demotion with a misleading "PMI
        # mismatch" diagnostic (audit fix/zq 2026-07-28).
        return False, (
            f"mpi4py MPI init failed ({type(exc).__name__}: {exc}).  "
            "This usually means a PMI mismatch between the launcher and "
            "the MPI library (classic signature: 'MPI_Init_thread() "
            "failed ... error code: 16').  On SLURM launch with "
            "`srun --mpi=pmi2`; some MPI builds also need "
            "I_MPI_PMI_LIBRARY (or the site equivalent) pointing at the "
            "system's libpmi2 — ask your site docs for the path.")
    try:
        import h5py
        if not bool(h5py.get_config().mpi):
            return False, ("h5py is built without MPI support "
                           "(h5py.get_config().mpi is False) — an "
                           "HDF5_MPI=ON h5py build is required")
    except Exception as exc:
        return False, f"h5py import failed ({type(exc).__name__}: {exc})"
    return True, "available"


def _route_cpu_slab_io(print_fn) -> "SlabIOBackend":
    """Pick the ``slab_io=auto`` backend on the JAX **CPU** backend.

    Runs UNCONDITIONALLY for ``slab_io=auto`` — routing depends on no
    other input key (the legacy ``use_ffi_io`` boolean is honored only
    as an explicit override, see ``from_input_file``).  Three tiers,
    best first — each one loud about why it was or was not taken,
    because "which writer ran" is the single most consequential thing
    about a large run's memory profile:

    1. ``PHDF5_FFI`` — the host FFI lib exports the collective write
       handler.  Zero extra Python environment; the ζ tile never leaves
       the rank that owns it.
    2. ``PHDF5_HOST`` — no write handler in the lib, but the interpreter
       has ``mpi4py`` + an ``HDF5_MPI=ON`` h5py (the AB overlay) and
       MPI actually initializes (the probe runs ``MPI_Init_thread``, so
       a PMI-mismatched harness demotes here instead of dying later).
       Same MPI-IO semantics, extra env.
    3. ``H5PY_ALLGATHER`` — neither; rank-0 serial write behind a full
       ``process_allgather`` of the tensor.

    The tier-1 probe is :func:`ffi_loader.probe_target`, not a bare
    ``has_target``: its three-state reason distinguishes "lib not built
    with the handler" (rebuild) from "lib could not be loaded"
    (``LD_LIBRARY_PATH``) — a distinction that cost workstream P a day.
    Never raises: any probe failure demotes to the next tier.

    Handler presence alone is NOT capability: both MPI tiers call
    ``MPI_Init_thread``, and on a bare launch with no PMI environment
    Intel MPI aborts the process inside ``MPIR_pmi_init`` instead of
    returning an error (job 7884926 — the fastloop's bare P=1 gw stage
    died at the first collective H5Fcreate).  So tier 1 additionally
    requires :func:`_probe_mpi_bootstrap_ffi` (launcher PMI environment,
    else a subprocess singleton-init probe) and demotes with an
    announcement when MPI cannot bootstrap — the same degrade-and-say-why
    contract the GPU router uses.  An explicit
    ``slab_io=phdf5_ffi`` still bypasses this and fails loudly.

    Node count is not a tier condition here and never was; as of
    2026-08-05 it is not one in the GPU router either (see
    :func:`_route_gpu_slab_io`).
    """
    reason = "ffi.common.ffi_loader import failed"
    try:
        from ffi.common.ffi_loader import probe_target
        _ffi_write_ok, reason = probe_target("lorrax_phdf5_write", "cpu")
    except Exception as exc:                      # pragma: no cover
        _ffi_write_ok = False
        reason = f"{type(exc).__name__}: {exc}"
    if _ffi_write_ok:
        _mpi_ok, _mpi_how = _probe_mpi_bootstrap_ffi("cpu")
        if not _mpi_ok:
            _ffi_write_ok = False
            reason = (
                f"the host FFI exports the write handler but MPI cannot "
                f"bootstrap in this process: {_mpi_how}")
    if _ffi_write_ok:
        print_fn(
            "  [config] slab_io=auto on CPU backend: host FFI exports "
            "the collective phdf5 write handler and MPI can bootstrap "
            f"({_mpi_how}).  Routing SlabIO through "
            "PHDF5_FFI (bare-MPI C++ collective MPI-IO, no mpi4py needed) "
            f"[{_slab_io_geometry()}]."
        )
        return SlabIOBackend.PHDF5_FFI

    _host_ok, _host_reason = _probe_phdf5_host_tier()
    if _host_ok:
        print_fn(
            "  [config] slab_io=auto on CPU backend: the host FFI write "
            f"handler is unavailable ({reason}).  Routing SlabIO through "
            "PHDF5_HOST (mpi4py + h5py-parallel) — same per-rank "
            "collective MPI-IO write semantics."
        )
        return SlabIOBackend.PHDF5_HOST

    print_fn(
        "  [config] slab_io=auto on CPU backend but neither parallel "
        f"writer is available (host FFI: {reason}; PHDF5_HOST tier: "
        f"{_host_reason}).  Falling back to H5PY_ALLGATHER (rank-0 "
        "serial write behind a full all-gather — slow, and the "
        "all-gather is the memory wall at scale).  Building the host "
        "FFI lib with the write handler enables PHDF5_FFI (see e.g. "
        "config/frontera/build_ffi_host.sh for a worked example)."
    )
    return SlabIOBackend.H5PY_ALLGATHER


def _route_gpu_slab_io(print_fn) -> "SlabIOBackend":
    """Pick the ``slab_io=auto`` backend on a non-CPU (GPU) JAX backend.

    Same capability-probed, never-env-guessed contract as the CPU
    router.  Tiers:

    1. ``PHDF5_FFI`` — the CUDA FFI lib exports the collective write
       handler AND MPI can bootstrap in this process.  **Node count is
       not a condition.**

       It used to be.  Until 2026-08-05 this router declined PHDF5_FFI
       outright whenever ``SLURM_JOB_NUM_NODES > 1``, *without running
       the probe*, citing a "known cross-node failure on multi-node GPU
       stacks".  The failure that justified it was Intel MPI refusing a
       launch configured with ``I_MPI_FABRICS=shm`` on Frontera — a
       launcher misconfiguration, on an MPI stack and a machine this
       router does not run on.  The Perlmutter GPU path is Cray MPICH
       through Shifter.

       Deleted after measuring the cell it forbade (job 56389339,
       ``srun --mpi=cray_shasta -N 4 -n 16``): ``MPI_Comm_size()``
       asserted == 16 on every rank, PHDF5_FFI write and read of a
       0.5 GiB C128 array bit-exact on round-trip, and the payload md5
       byte-identical to the same logical array written by 4 ranks on 1
       node through a 2x2 mesh.  The branch was LIVE, not merely wrong: it read the max of
       ``SLURM_JOB_NUM_NODES`` and ``SLURM_NNODES``, and while the former
       is absent inside the Shifter container (it is a batch-level
       variable), ``SLURM_NNODES`` is exported by every ``srun`` step —
       measured 4 on the 4-node run.  So every multi-node GPU run really
       did get silently demoted off the FFI writer, which is what the
       archived logs show.
    2. ``PHDF5_HOST`` — mpi4py + h5py-parallel present, MPI inits, AND
       exactly one addressable device per process (the backend's
       one-shard-per-process contract; a single process driving N GPUs
       would fail at write time, so the probe declines it up front).
    3. ``H5PY_ALLGATHER`` — always works: gather via JAX collectives,
       rank-0 serial h5py write of host arrays.  Note this tier is a
       last resort and NOT a supported production path at scale: it
       materialises the whole array on rank 0, which is the memory wall
       the per-rank-tile contract exists to avoid.
    """
    import jax as _jax

    reason = "ffi.common.ffi_loader import failed"
    _ffi_write_ok = False
    _mpi_how = "not probed"
    try:
        from ffi.common.ffi_loader import probe_target
        _ffi_write_ok, reason = probe_target("lorrax_phdf5_write", "CUDA")
    except Exception as exc:                      # pragma: no cover
        reason = f"{type(exc).__name__}: {exc}"
    if _ffi_write_ok:
        # Handler presence is not capability: the tier calls
        # MPI_Init_thread.  Same gate the CPU router applies.
        _mpi_ok, _mpi_how = _probe_mpi_bootstrap_ffi("CUDA")
        if not _mpi_ok:
            _ffi_write_ok = False
            reason = ("the CUDA FFI exports the write handler but MPI "
                      f"cannot bootstrap in this process: {_mpi_how}")
    if _ffi_write_ok:
        print_fn(
            "  [config] slab_io=auto on GPU backend: CUDA FFI exports "
            "the collective phdf5 write handler and MPI can bootstrap "
            f"({_mpi_how}).  Routing SlabIO through PHDF5_FFI "
            f"[{_slab_io_geometry()}]."
        )
        return SlabIOBackend.PHDF5_FFI

    _host_ok, _host_reason = _probe_phdf5_host_tier()
    if _host_ok and _jax.local_device_count() != 1:
        _host_ok = False
        _host_reason = (
            f"{_jax.local_device_count()} addressable devices in this "
            "process; the PHDF5_HOST backend requires exactly one "
            "shard per process (one GPU per process)")
    if _host_ok:
        print_fn(
            "  [config] slab_io=auto on GPU backend: CUDA FFI write "
            f"handler unavailable ({reason}).  Routing SlabIO through "
            "PHDF5_HOST (mpi4py + h5py-parallel, per-rank collective "
            f"MPI-IO of the device shard via host staging) "
            f"[{_slab_io_geometry()}]."
        )
        return SlabIOBackend.PHDF5_HOST

    print_fn(
        "  [config] slab_io=auto on GPU backend: no parallel writer "
        f"available (CUDA FFI: {reason}; PHDF5_HOST tier: "
        f"{_host_reason}).  Falling back to H5PY_ALLGATHER (rank-0 "
        "serial write behind a full gather — always works, slow at "
        "scale)."
    )
    return SlabIOBackend.H5PY_ALLGATHER


class GspaceIO(str, enum.Enum):
    """How ψ(G) is moved into the ISDF r-chunk loop.

    Both modes keep ψ(G) on host in per-rank band-sharded layout and
    pull one band-chunk at a time into the jit via io_callback — never
    more than one bc on device at a time.

    - ``HOST_CACHE`` — read ψ(G) once at startup, keep resident in host
      RAM for the full run.  Default; fastest.
    - ``FILE_REREAD`` — rebuild the host buffer at each r-chunk via
      phdf5 collective read; drop between r-chunks.  Zero persistent
      host residency (needed for huge systems where host RAM can't
      hold ψ(G)).
    """
    HOST_CACHE = "host_cache"
    FILE_REREAD = "file_reread"


#: The two W Dyson plans (``gw/w_isdf.py``) — the ONLY legal resolved
#: values of the ``w_dyson_solver`` input key.
_W_DYSON_PLANS = ("local", "distributed")


def normalize_w_dyson_solver(value) -> str:
    """Normalise a ``w_dyson_solver`` spelling to one of the TWO plans.

    Single source of the vocabulary — the parser and
    ``w_isdf._resolve_w_solve_fn`` both call this, so a spelling cannot
    mean different things at parse time and solve time.

    - ``local`` / ``auto`` / None → ``"local"`` (the q-parallel per-q
      dense LU; ``auto`` is a permanent back-compat alias).
    - ``distributed`` → ``"distributed"`` (the 2-D-sharded stacked-GEMM
      backsolve through the ffi.linalg plan facade).
    - ``lu`` → ``"local"`` with a DeprecationWarning (it was the same
      route under its old name).
    - ``lstsq`` → ``ValueError``: the SVD min-norm inner solve was
      REMOVED in the two-plan cleanup (2026-07-27) — old decks fail
      informatively instead of silently rerouting.
    """
    s = ("auto" if value is None else str(value)).strip().lower()
    if s == "lu":
        import warnings
        warnings.warn(
            "w_dyson_solver = lu is deprecated: the per-q pivoted LU is "
            "now spelled 'local' (and is the default).  Update the deck "
            "to w_dyson_solver = local.",
            DeprecationWarning, stacklevel=2)
        s = "local"
    if s == "lstsq":
        raise ValueError(
            "w_dyson_solver = lstsq was REMOVED (two-plan W cleanup, "
            "2026-07-27).  The two plans are 'local' (per-q pivoted LU, "
            "default) and 'distributed' (2-D-sharded ScaLAPACK/cuSOLVERMp "
            "backsolve).  lstsq existed as a rank-deficiency fallback; a "
            "rank-deficient A = 1 - V·chi0 means the centroid basis has "
            "over-completed the pair-density rank — reduce n_mu (fewer "
            "centroids) or raise zeta_rcond instead of masking it with a "
            "min-norm solve.")
    if s == "auto":
        return "local"
    if s not in _W_DYSON_PLANS:
        raise ValueError(
            f"w_dyson_solver={value!r} invalid; expected "
            f"local (default; auto is an alias) or distributed.")
    return s


def eigh_backend_choices() -> tuple:
    """The legal ``eigh_backend`` spellings — the RESOLVER's own list.

    Read from :data:`ffi.linalg.resolve.BACKEND_CHOICES` so the parser and
    the thing that actually dispatches cannot drift.  They HAD drifted:
    this parser accepted only ``auto|off|cusolvermp|slate`` while the
    resolver had grown ``distributed`` (the portable "spread ONE tile over
    the mesh" spelling, and the ONLY eigh backend that exists on a host
    mesh, where it means ScaLAPACK ``pzheevd``) and ``scalapack``.  The
    effect was that the low-memory eigh could not be requested at all
    through a GW input file on CPU — the very platform it is needed on.

    Falls back to the literal tuple if ``ffi`` cannot be imported (a
    parser must not need the FFI package to read a deck); the fallback is
    pinned equal to the resolver's list by
    ``tests/test_bse_setup_qchunk.py``.
    """
    try:
        from ffi.linalg.resolve import BACKEND_CHOICES
        return tuple(BACKEND_CHOICES["eigh"])
    except Exception:
        return ("auto", "off", "distributed", "cusolvermp", "slate",
                "scalapack")


def resolve_eigh_backend(params) -> str:
    """``(eigh_backend, use_low_mem_eigh)`` → ONE backend string.

    THE single place the two spellings of one axis are combined, so a
    driver that reads the raw params dict (``bandstructure.htransform``,
    ``bse.exciton_bands``) gets the same answer as ``LorraxConfig``.

    * ``use_low_mem_eigh`` unset/false → ``eigh_backend`` verbatim.
    * true + ``auto`` → ``"distributed"`` (the platform's distributed
      library; ScaLAPACK on a host mesh, cuSOLVERMp on CUDA).
    * true + an explicit library name → that name (it already IS the
      distributed path).
    * true + ``off`` → ``ValueError``.  ``off`` pins the q-batched native
      eigh, which needs a WHOLE ``(rank, rank)`` matrix per device — the
      one thing the flag says is unaffordable.  Refusing at parse time is
      the doctrine: an explicit request that cannot be honoured never
      silently becomes its opposite.

    Vocabulary is checked here too, so an unknown spelling fails at parse
    time rather than at the first eigh.
    """
    raw = params.get("eigh_backend", "auto") if hasattr(params, "get") else "auto"
    backend = str("auto" if raw is None else raw).strip().lower()
    choices = eigh_backend_choices()
    if backend not in choices:
        raise ValueError(
            f"eigh_backend={backend!r} invalid; expected "
            f"{' / '.join(choices)}.")
    low_mem = bool(params.get("use_low_mem_eigh", False)
                   if hasattr(params, "get") else False)
    if not low_mem:
        return backend
    if backend in ("auto", "native"):
        return "distributed"
    if backend == "off":
        raise ValueError(
            "use_low_mem_eigh = true with eigh_backend = off is a "
            "contradiction: 'off' pins the q-batched NATIVE eigh, which is "
            "the path that needs one whole (rank, rank) matrix per device — "
            "exactly what the low-memory flag says will not fit.  Either "
            "drop use_low_mem_eigh, or set eigh_backend = auto (resolves to "
            "'distributed') or name a library "
            "(distributed|cusolvermp|slate|scalapack).")
    return backend


# ---------------------------------------------------------------------------
#  Defaults — single source of truth for every input key
# ---------------------------------------------------------------------------

_DEFAULTS = {
    # System geometry
    "nval": 5,
    "ncond": 5,
    "nband": 100,
    "sys_dim": 2,
    # Rebuild V_H from the CURRENT orbitals each self-consistent iteration
    # instead of rotating the fixed DFT one into the QP basis.  Off keeps
    # QSGW fixed-density, which is what every result before 2026-08-04 was.
    "density_self_consistent": False,
    # Run the SC loop's H / E / U on the IBZ, broadcasting back at the
    # boundary.  Sigma stays on the full BZ -- it is an FFT over the
    # k-grid.  Off keeps the loop entirely full-BZ.
    "sc_on_ibz": False,
    # Density-grid cutoff (Ry) for the psp matrix-element tools (kin_ion /
    # dipole).  None → the consumer defaults it to the WFN's own ``ecutwfc``.
    "ecutrho": None,
    # File paths
    "wfn_file": "WFN.h5",
    "centroids_file": "centroids_frac.txt",
    # Optional second centroid file used by the bispinor pipeline:
    # μ_L=1,2,3 (transverse) ζ-fits use Gordon-current-density centroids
    # rather than the charge-density centroids in ``centroids_file``.
    # Empty string == "not set" (cfg.centroids_file_current is None then).
    "centroids_file_current": "",
    "kin_ion_file": "kin_ion.h5",
    # Where H0's mean-field Hartree term comes from.  H0 = kin_ion + V_H is
    # a ~500 eV cancellation, so this is an explicit, validated choice
    # rather than something inferred from what happens to be on disk.
    #   auto   — stored 'v_hartree' array in kin_ion.h5 if present, else
    #            the legacy folded file if that is what it is, else isdf
    #   stored — require the exact array in kin_ion.h5 (raises if absent)
    #   isdf   — the ISDF V_q[0] tile (cohsex_sigma's Hartree kernel);
    #            distributed and in-loop capable, centroid-count dependent
    #   gspace — rebuild the exact FFT-grid matrix on the fly this run
    # See file_io/kin_ion.py's module docstring for the full contract and
    # the scorecard's S.5 table for the accuracy each buys.
    "hartree_source": "auto",
    # Three human-readable text outputs (always written):
    #   sigma_diag.dat — LORRAX-native per-(k,n) Σ-decomposition dump.
    #   eqp0.dat       — BGW-format zeroth-order QP energies.
    #   eqp1.dat       — BGW-format Z-linearized QP energies (Z=1 in
    #                    static COHSEX, central-difference Z in PPM).
    # The legacy ``output_file`` key (LORRAX-native eqp0.dat) and
    # ``eqp_output_file`` (unused) were dropped 2026-05-04; setting
    # them in cohsex.in now logs a deprecation warning and is ignored.
    "sigma_diag_file": "sigma_diag.dat",
    "eqp0_file": "eqp0.dat",
    "eqp1_file": "eqp1.dat",
    "sigma_omega_h5_file": "sigma_mnk.h5",
    # Core flags
    "restart": True,
    # ``compute_mode`` is the single axis describing the self-energy ansatz.
    # ``"auto"`` infers from the legacy ``do_screened`` / ``use_ppm_sigma`` /
    # ``ppm_model`` flags so existing input files keep working unchanged.
    # New input files should set ``compute_mode`` explicitly:
    #   "x_only" | "cohsex" | "gn_ppm" | "hl_ppm".
    "compute_mode": "auto",
    # ``qp_solver`` is the orthogonal axis describing how QP energies are
    # extracted from Σ (see the ``QPSolver`` enum).  ``"auto"`` resolves
    # from the deprecated ``self_consistent`` key (true → self_consistent)
    # and otherwise defaults to "one_shot_dft" (standard G0W0).  New input
    # files should set it explicitly:
    #   "one_shot_dft" | "fixed_point" | "self_consistent".
    "qp_solver": "auto",
    "do_screened": True,
    "bispinor": False,
    "do_G0": True,
    # Deprecated (2026-07-08): ``self_consistent = true`` is honored as an
    # alias for ``qp_solver = self_consistent`` via auto-resolution.  SC is
    # wired for ALL modes (mode-agnostic sigma_dispatch), not just COHSEX.
    "self_consistent": False,
    # Self-consistency loop knobs (read only when qp_solver=self_consistent).
    # Promoted from the LORRAX_SC_* env vars (2026-07-08); the envs are
    # still honored as deprecated overrides.
    "sc_max_iter": 20,
    "sc_tol_ev": 1.0e-4,
    "sc_accelerator": "rcrop",   # rcrop | linear
    "sc_history_depth": 5,       # rCROP history depth
    "sc_mixing": 1.0,            # linear-mixing α (accelerator=linear only)
    "sc_dump_dir": "",           # E-history npy dump dir ("" = off)
    "sc_eigh": "auto",           # auto | native | distributed (per-iteration
                                 # eigh of the (nk, nb, nb) carry; a LAYOUT
                                 # choice, independent of the physics knobs)
    "use_ppm_sigma": False,
    # BGW-style averaging of diagonal Σ within degenerate sets (mirrors
    # ``Sigma/shiftenergy.f90`` band-averaging).  ``no_degen_averaging =
    # true`` disables it and emits the raw QE-basis-dependent diagonals.
    # ``degen_avg_tol_ry`` matches BGW's ``TOL_Degeneracy = 1e-6 Ry``.
    "no_degen_averaging": False,
    "degen_avg_tol_ry": 1.0e-6,
    # DEPRECATED (2026-07-27) tri-state boolean, superseded by
    # ``slab_io``.  None (default) = unset: ``slab_io=auto`` routes by
    # capability alone.  Explicit ``false`` forces the historical
    # ``process_allgather`` + rank-0 ``h5py`` writer (same as
    # ``slab_io = h5py_allgather``).  Explicit ``true`` is redundant —
    # the router already prefers the parallel writers.
    "use_ffi_io": None,
    # SlabIO backend: "auto" (default — the capability-probed router for
    # the active JAX platform, see ``_route_cpu_slab_io`` /
    # ``_route_gpu_slab_io``) | "phdf5_ffi" | "phdf5_host" |
    # "h5py_allgather".  Every enum value is reachable from the input
    # file; an explicit value is honoured verbatim and fails loudly if
    # unavailable.
    "slab_io": "auto",
    # ψ(G) source for the ISDF r-chunk loop.  Both modes keep ψ(G) on
    # the HOST in per-rank band-sharded layout and pull one band-chunk
    # at a time into the jit via io_callback — never more than one bc
    # on device at a time.  Modes differ in host-side lifecycle:
    #   "host_cache"  – read once at startup, keep resident in host
    #                   RAM for the full run (default; fastest).
    #   "file_reread" – rebuild the host buffer at each r-chunk via
    #                   phdf5 collective read; drop between r-chunks.
    #                   Zero persistent host residency (needed for
    #                   huge systems where host RAM can't hold ψ(G)).
    "gspace_mode": "host_cache",
    # ``accumulate_rchunk_to_gflat`` flat-axis chunker.  Bounds the
    # per-scan-iter FFT box ``chunk_size · n_rtot``.
    # 0 (default) = one-shot; the gflat memory model overrides this
    # at runtime when its planner picks a smaller value, but cohsex.in
    # > 0 wins over the planner.
    "gflat_chunk_size": 0,
    # V_q inner G-axis GEMM chunk size.  Bounds the per-q ``lax.scan``
    # working set inside the per-q V_q kernel.
    # 0 (default) = auto (``_pick_g_chunk(ngkmax)`` → largest divisor
    # of ngkmax ≤ 4096).
    "vq_g_chunk_size": 0,
    # ζ-fit solver path overrides (3-state).  Default ``auto`` picks
    # cuSolverMp on true 2D meshes (p_x ≥ 2 AND p_y ≥ 2) and the
    # JAX/CUDA fallback otherwise.  Force a path with ``on`` / ``off``.
    # Distributed dense-linalg backends (block-cyclic).  Portable axes —
    # the values name LIBRARIES, not vendors' key names:
    #   distributed_cholesky = auto | off | cusolvermp | slate
    #       charge-channel ζ-fit Cholesky.  auto → cusolvermp on true-2D
    #       GPU meshes, in-tree sharded_cholesky otherwise.  slate is the
    #       portability path (Frontier/Aurora); explicit request fails
    #       loudly if the FFI/library is absent (optional dependency).
    #   distributed_lu = auto | off | cusolvermp | scalapack
    #       transverse-channel LU.  scalapack = the host/CPU-backend
    #       backend (Cray LibSci pXgetrf+pXgetrs via liblorrax_ffi_host);
    #       explicit, never auto-picked.  (SLATE getrf not yet written.)
    "distributed_cholesky": "auto",
    "distributed_lu":       "auto",
    #   eigh_backend = auto | off | distributed | cusolvermp | slate
    #                | scalapack
    #       Hermitian eigensolver for the BSE/htransform distributed-eigh
    #       sites (bse_setup fH_q, vq_interp coarse C_q tiles).  auto|off =
    #       the q-BATCHED native jnp.linalg.eigh (the measured default at
    #       every production tile size); the rest spread ONE tile over the
    #       whole mesh via the distributed-linalg FFI — the wide-band-window
    #       regime where a single matrix no longer fits on one device (square
    #       mesh + one process per device required; all guards fire at
    #       resolve time — see ffi/linalg + docs/dev/linalg_ffi.md).
    #       ``distributed`` = the PLATFORM's distributed library (ScaLAPACK
    #       pzheevd on a host mesh, cuSOLVERMp on CUDA) and is the spelling
    #       that ports; the vocabulary is ffi.linalg.resolve's own, checked
    #       against it at parse time so the two can never drift.  The
    #       --eigh-backend CLI flag of htransform / exciton_bands OVERRIDES
    #       this key.
    "eigh_backend":         "auto",
    #   use_low_mem_eigh = true | false   (default false)
    #       The SAME axis named by INTENT instead of by library: "one whole
    #       (rank, rank) matrix does not fit on a rank, keep it spread over
    #       the mu x nu face".  true + eigh_backend=auto  =>  'distributed'.
    #       true + an explicit library name keeps that name.  true +
    #       eigh_backend=off is a CONTRADICTION and is refused at parse time,
    #       as is a true that cannot be honoured on this mesh — never a
    #       silent fall back to the whole-matrix path the flag exists to
    #       avoid.  See bandstructure.bse_setup.compute_wfns_fi.
    "use_low_mem_eigh":     False,
    # Charge ζ-fit OPT-IN Tikhonov ridge ε (added ON TOP of the fixed
    # 1e-14·|tr| non-singularity floor, as a fraction of the mean CCT
    # diagonal tr(C)/n): C_q ← C_q + [1e-14·|tr| + ε·|tr|/n]·I before the
    # replicated Cholesky.  A per-q SCALAR, so mesh-invariant.  Default 0.0
    # ⇒ bit-identical to the historical factor (frozen-golden contract).  A
    # POSITIVE ε conditions a NEAR-SINGULAR CCT (n_μ over-complete for the
    # system's pair-density rank) so ζ = (C+εI)⁻¹Z stops amplifying the
    # ULP-level, mesh-dependent pair-density (cuBLAS-GEMM-per-shard-dim)
    # roundoff into a grid-dependent V_q.  MoS2 6×6 (n_μ=1600) needs ε≈1e-4
    # to bring cross-grid Re Σ_c agreement from O(10 eV) to ~10 meV.  It
    # PERTURBS the physical result (the regularised answer is ε-dependent on
    # this ill-posed fit) — hence opt-in, a physics call.  Env override
    # LORRAX_ZETA_RIDGE.  See reports/gw_zeta_mesh_invariance_2026-07-20.
    "zeta_ridge":           0.0,
    # Charge ζ-solve conditioner (μ_L=0 channel only).  "rank_truncate"
    # (DEFAULT) = rank-revealing eigh pseudo-inverse: drop eigenvalues
    # < zeta_rcond·λ_max before inverting, so the near-null directions of
    # the over-complete charge CCT (n_μ > pair-density rank, κ~1e13) are
    # removed at the source instead of amplified by plain Cholesky into
    # O(1) V_q errors that GN-PPM magnifies to tens of eV (the conduction
    # Σc blow-up / device-count / nband instability).  "cholesky" = the
    # historical replicated/cuSolverMp Cholesky path (bit-identical to the
    # pre-feature behavior; the selectable alternative).
    "charge_zeta_solve":    "rank_truncate",   # rank_truncate | cholesky
    # ζ BACK-SOLVE TIER — how much of the (nq, μ, μ) charge factor is ever
    # replicated.  The first three tiers below are numerically free (same
    # per-q arithmetic, only the gathered extent differs); `distributed`
    # replaces the factorization as well and is the only one that scales.
    #   replicated  = today's path: gather the whole (q_batch, μ, μ) stack
    #                 onto every rank, nq·μ²·16 B, re-gathered per r-chunk
    #                 (18.9 GB/rank at MoS2 12×12 / μ=1998).
    #   per_q       = gather ONE (μ, μ) tile at a time, loop q — the slice
    #                 is taken inside a shard_map so the partitioner cannot
    #                 turn it back into the full-stack gather (it did until
    #                 workstream AA; scorecard Y.2).  μ²·(1+1/p_y)·16 per
    #                 execution and nq executions per r-chunk, so its TOTAL
    #                 per-r-chunk traffic is ≈ the replicated tier's while
    #                 its LIVE gather is nq× smaller: use it when memory,
    #                 not bandwidth, is the binder and the mesh is not
    #                 square enough for `distributed`.
    #   distributed = NOTHING O(μ²) is replicated.  Distributed eigh
    #                 (ScaLAPACK pzheevd), truncation on the replicated
    #                 spectrum, 2D-sharded C⁺, and a stacked GEMM C⁺@Z with
    #                 both operands 2D-sharded.  The ONLY tier whose
    #                 FACTORIZATION divides by P — the other two run one
    #                 dense eigh per q redundantly on every rank, O(nq·μ³)
    #                 with no P-scaling (~86 h at μ=10k).  EXPLICIT opt-in:
    #                 a block-cyclic eigh picks a different (equally valid)
    #                 gauge, so ζ matches the other tiers to ~κ·ε, not
    #                 bit-exactly.  Needs charge_zeta_solve='rank_truncate'
    #                 and a SQUARE or 1-D mesh (pXheevd descriptor rule);
    #                 refuses at resolve time otherwise.  On the transverse
    #                 channels it resolves to per_q (indefinite CCT — its
    #                 distributed route is distributed_lu=scalapack).
    #   auto (DEFAULT) = replicated while the gather fits under
    #                 LORRAX_ZETA_GATHER_CAP_GIB (4 GiB), per_q above it.
    #                 Never `distributed`.  Fixture-scale stacks stay on
    #                 replicated, so the default path is bit-identical to
    #                 the pre-feature one.
    # A SEPARATE env bound governs the `distributed` tier's TRANSPORT, not
    # its memory: LORRAX_COLLECTIVE_CHUNK_MB (128 MB) caps ONE emitted
    # collective's payload.  The 4 GiB gather cap was satisfied when job
    # 7876062 died at P=144 inside a single 1.15 GB Gloo AllGather; see
    # isdf/core.py's "COLLECTIVE PAYLOAD CHUNKING" note and scorecard AF.
    "distributed_zeta_solve": "auto",  # auto | replicated | per_q | distributed
    # Rank-truncation cutoff (relative to λ_max, per q).  DEFAULT 1e-8 —
    # the LOW end of the over-complete recovery plateau.  An over-complete
    # basis needs it: at MoS2 4×4/1204c, 1e-10 only partially recovers (MAE
    # 1.4 eV vs BGW) while the whole 1e-8…1e-4 plateau collapses to ~0.04 eV
    # — so pick the plateau's low end, because truncation is NOT free on a
    # well-conditioned basis: bulk Si 4×4×4/960c (the BGW-anchored
    # si_cohsex_3d gate) does have eigenvalues below the cut, and 1e-6 drifts
    # its sigTOT by 1.021 meV where 1e-8 costs only 0.054 meV — the identical
    # over-complete cure at ~20× less drift (sweep table in
    # docs/docs_gwjax/COHSEX_INPUT.md).  Env override LORRAX_ZETA_RCOND.
    # Mirrored by the isdf/core.py + gw/isdf_fitting.py signature defaults.
    # reports/gw_rank_truncation_2026-07-20 + gw_bandrange_centroids_2026-07-21.
    "zeta_rcond":           1e-8,
    # Transverse ζ-solve family (bispinor μ_L=1,2,3 channels only; inert
    # otherwise).  "ridge" (DEFAULT) = the historical hoisted pivoted-LU
    # with the 1e-12·|tr|/n diagonal ridge — byte-identical to the
    # pre-feature behavior.  "rank_truncate" = per-q eigh pseudo-inverse
    # of the Hermitian INDEFINITE transverse CCT with an |λ| cut (drop
    # |λ| < transverse_zeta_rcond·|λ|_max): the charge channel's
    # conditioning cure ported to the transverse channel — TRS-paired
    # near-null current modes are REMOVED instead of inverted through at
    # the ridge floor (κ~1e12), and the per-q n_keep log doubles as the
    # transverse basis-adequacy instrument.  Grammar mirrors
    # charge_zeta_solve.  Its LOCAL plan (replicated whole-tile eigh,
    # q-parallel at P>1) runs at ANY centroid count on ANY mesh; its
    # DISTRIBUTED plan is selected by distributed_zeta_solve=distributed
    # (pzheevd at the padded extent — the mesh-divisibility constraint of
    # distributed_lu=scalapack does not apply).  distributed_lu is an LU
    # backend key and conflicts with this family: explicit
    # scalapack/cusolvermp + rank_truncate refuses at parse time.
    "transverse_zeta_solve": "ridge",   # ridge | rank_truncate
    # Transverse rank-truncation cutoff τ (relative to |λ|_max, per q).
    # Only read by transverse_zeta_solve=rank_truncate.  Default from the
    # 2026-08 MoS2 4×4 bispinor calibration ladder (eqp drift vs the
    # ridge control monotone in τ and within the 1e-4 eV gauge tolerance
    # across transverse set sizes).  No env twin (scorecard AV: policy
    # knobs live in the deck).
    "transverse_zeta_rcond": 1e-10,
    # γ̃-double-contract kernel variant inside the monolithic pair
    # pipeline (see ``common.gamma_matrices.gamma_double_contract``).
    # Math identical across all three; differ in HLO structure.
    #   "take"   – jnp.take + element-wise phase mul (default).
    #   "einsum" – materialise the sparse γ̃ and contract via einsum.
    #   "scan"   – lax.scan over the (a, b) spin axis pairs.
    "gamma_contract_mode": "take",
    # Memory / chunking
    "memory_per_device_gb": 0.0,  # 0 = auto-detect
    "band_chunk_size": 16,
    "r_chunk_size": 0,
    # ISDF
    # Which of the TWO W Dyson plans solves A·W = V, A = (1 - Vχ₀):
    #   local (default; auto is an alias)
    #                per-q pivoted LU inside the q-parallel shard_map.
    #   distributed  2-D-sharded stacked-GEMM backsolve through the
    #                ffi.linalg plan facade (ScaLAPACK on CPU meshes,
    #                cuSOLVERMp on CUDA); no rank ever materialises a
    #                full (μ, μ) tile.  Refuses loudly at resolve time
    #                on an unsupported mesh/build — never silently
    #                downgrades to local.
    # (lu → local with a DeprecationWarning; lstsq was removed.)
    "w_dyson_solver": "auto",
    "mc_average_vcoul_body": True,
    # Per-Q mini-BZ Coulomb head cell-averaging (BGW minibzaverage_3d/2d).
    # False (default) = current behavior, BIT-IDENTICAL: the q→0 head is the
    # pure-Sobol mini-BZ mean and every finite-Q exchange head is the analytic
    # POINT value v(Q+G*).  True routes the head through
    # ``gw.coulomb.base.minibz_average``: the q→0 3D head gains the analytic
    # Baldereschi-Tosatti sphere term (seed-independent), the Voronoi fold
    # widens (nmax 1→3), and the BSE arbitrary-Q ``eval_vq`` head becomes the
    # mini-BZ CELL AVERAGE ``<v_LR(Q+G*)>_mBZ`` (fixes the 4-13% near-Γ /
    # zone-boundary point-vs-cell-average error, arbitrary_q_bse.md §16.4).
    # The winding (2D e^{-i2θ}) is unaffected — only the head magnitude is
    # averaged; the phase-factored ζ̃ rank-1 structure carries the direction.
    "head_minibz_average": False,
    # BSE fine-grid densification.  When set to "NX NY NZ" (or "NX,NY,NZ") and
    # DIFFERENT from the coarse restart/WFN grid, the GENERAL BSE init
    # (``bse_io.load_bse_data_from_restart_sharded``) interpolates the ENTIRE
    # BSE problem — ψ, QP ε (htransform fH), V_Q exchange (vq_interp), and the
    # W direct term (zero-pad in R) — from the coarse grid onto this fine grid
    # BEFORE any solve, so EVERY BSE solver (exciton_bands / feast / nontda /
    # kpm / resolvent) transparently runs on the fine grid.  Each fine length
    # must be a positive multiple of the matching coarse length (coarse BZ ⊂
    # fine BZ).  Empty (default) or == the coarse grid → the coarse ``data``
    # bundle is returned byte-identically (fast path untouched).  Subsumes the
    # exciton_bands ``--w-coarse-grid`` W-only flag for the direct term.
    "bse_k_grid": "",
    "bare_coulomb_cutoff": None,
    # ζ-sphere cutoff (Ry).  When the writer emits zeta_q_G with per-q
    # WFN.h5-style spheres, this is the cutoff used to define the per-q
    # G-list.  Defaults to ecutwfc (mirrors the bare-Coulomb default);
    # max value is ecutrho.  Must be ≥ bare_coulomb_cutoff (V_q can't
    # use ζ̃(q+G) at G's the writer didn't store).
    "zeta_cutoff": None,
    # BGW vcoul override (for diagnostic BGW-vs-LORRAX comparison)
    "use_bgw_vcoul": False,
    "bgw_vcoul_file": "",
    # Aux WFN for pulling the 48-op crystal symmetry group when the main
    # WFN is nosym (its mf_header/symmetry/mtrx is truncated to identity).
    # Used only to fold LORRAX full-BZ q's onto BGW's IBZ q-list.
    "bgw_vcoul_sym_wfn": "",
    # Coulomb head
    "wcoul0_source": "s_tensor",
    "wcoul0_eta": 0.0,
    "vhead": None,
    "whead_0freq": None,
    "whead_imfreq": None,
    # Screening / minimax
    "screening_method": "minimax",
    "minimax_target_error": 1.0e-6,
    "minimax_max_nodes": 64,
    "regenerate_minimax_tables": False,
    "minimax_energy_reference": "midgap",
    # PPM
    # ppm_model picks the two-point pole-fit ansatz:
    #   "gn" — Godby-Needs: second probe at ω = i·ppm_omega_p (imaginary,
    #          ppm_omega_p ≈ 2 Ry by default).
    #   "hl" — Hybertsen-Louie: second probe at ω = ppm_omega_p (real,
    #          chosen above all transition energies; default 200 Ry).
    "ppm_model": "gn",
    "ppm_omega_p": 2.0,
    "ppm_fallback_omega": 2.0,
    # Override the head pole frequency Ω_h directly (Ry).  Useful for
    # testing against BGW's analytic head — set to BGW's
    # √(ω_p²/(1−ε_head⁻¹)) value to remove the LORRAX-vs-BGW
    # ε_head averaging convention as a source of disagreement.
    # None = compute Ω_h normally (analytic for HL, 2-pt fit for GN).
    "ppm_head_omega_h_ry": None,
    # Probe-χ₀ reuse (GN model only).  The probe-ω screening pass rebuilds
    # χ₀ with its own imaginary-axis minimax nodes — a second full τ sweep
    # (Gv/Gc build + FFTs + contraction per node) costing nearly as much
    # as the static pass (scorecard BC: 9.6 s vs 9.1 s at b300/P=16).
    #   "off"  (default) — dedicated probe quadrature, today's exact path.
    #   "auto" — represent the probe integrand x/(x²+ωp²) on the STATIC
    #        pass's τ nodes plus the MINIMAL augmentation from the
    #        dedicated quadrature's node set (Lawson-weighted fits;
    #        minimax_screening.refit_imag_alpha_augmented) at an error no
    #        worse than max(dedicated err, target_error); the probe χ₀
    #        then accumulates as a second weighted sum inside ONE fused τ
    #        sweep — shared nodes' tensors are computed once and only the
    #        k extras cost new compute.  Guaranteed fallback: with every
    #        extra node in, the exact dedicated weights are installed.
    # Numerics: same quadrature-error contract, different bits — NOT
    # bit-identical to "off" (pinned-baseline decks must keep "off" until
    # their references are re-pinned).  HL probes (real axis) always take
    # the dedicated path.
    "ppm_probe_chi_reuse": "off",
    "ppm_sigma_target_error": 1.0e-6,
    "ppm_sigma_max_nodes": 64,
    # Sigma frequency grid
    "sigma_omega_min_ev": -5.0,
    "sigma_omega_max_ev": 5.0,
    "sigma_omega_step_ev": 0.25,
    "sigma_regularization_ev": 0.25,
    "sigma_window_edge_factor": 1.5,
    "sigma_omega_batch_size": 4,
    "sigma_omega_accumulation": "auto",
    # Σ_c(ω,k,m,n) end-of-stage layout (wk_REL ω-cube sharding workstream):
    #   "replicated" (default) — today's path: the per-rank (m_X, n_Y) host
    #       tiles are gathered into the FULL cube on EVERY rank
    #       (n_ω·nk·nb²·16 B replicated; 2751 MB/rank at nb=512).
    #   "sharded" — the tiles stay where the stacked psum_scatter left them
    #       on the existing 2-D mesh; consumers (head injection, diag/eqp
    #       interpolation, QSGW build, sigma_mnk.h5 SlabIO write) read the
    #       P(None,None,'x','y')-sharded cube directly.  Outputs are
    #       bit-identical to "replicated" (movement-only; A/B gated by
    #       tests/multi_device/sigma_omega_layout_ab.py under BOTH
    #       one_shot_dft and self_consistent).
    #       Refusals (at driver resolve time, never mid-run): indivisible
    #       σ band window, and slab_io=h5py_allgather at P>1.  There is no
    #       qp_solver refusal: the SC loop never rotates the cube (it is
    #       absent from the finalize `replace` at sc_iteration.py:1321),
    #       so there is no rotation seam to port, and the two layouts
    #       measure bit-identical under SC (jobs 7889782/7889789).
    "sigma_omega_layout": "replicated",
    # PPM sigma options
    # PPM invalid-pole treatment (BGW invalid_gpp_mode). 'zero' drops Omega^2<0
    # poles (BGW mode 0); '2ry' keeps the fit's fallback pole (BGW mode 2);
    # 'static_limit' (default, matching BGW's default mode 3) drops the
    # dynamical pole and adds the analytic static-COHSEX term for those
    # modes — see ppm_sigma._compute_invalid_static_sigma.
    "ppm_invalid_mode": "static_limit",
    "fermi_reference": "midgap",
    "sigma_at_dft_extrapolate": False,
    # Deprecated (2026-07-08): ``sigma_at_dft_energies = true`` is honored
    # as an alias for ``qp_solver = one_shot_dft`` — which is now the
    # default — via auto-resolution.  (The key was parsed-but-unread for
    # its whole life; its intended meaning, authoritative at-DFT QP
    # evaluation, is exactly QPSolver.ONE_SHOT_DFT.)
    "sigma_at_dft_energies": False,
    # Debug
    "sigma_freq_debug_output": False,
    "sigma_freq_debug_file": "sigma_freq_debug.dat",
    # QP wavefunction file dump.  Default True: end-of-run write of
    # ``WFN_qp.h5`` (BGW format, ψ rotated by the final U, energies
    # replaced by E_QP).  Fires for both one-shot and SC; set False to
    # skip the ~10s of MB write when only eqp.dat is wanted.
    "write_wfn_h5": True,
    # BSE interpolation setup (htransform-driven fine-k wfn recovery; see
    # ``bandstructure.bse_setup.compute_wfns_fi``).
    "get_centroids_fi": False,   # Gate; if True, compute fine-grid wfns at coarse centroids.
    "wfn_fi_min": 0,             # Sub-window of htransform band axis (0-based).
    "wfn_fi_max": 0,             # Exclusive upper end. wfn_fi_max==0 → use full window.
    "kgrid_fi": "",              # "nx ny nz" or "nx,ny,nz". Empty → no fine grid.
    # Fine-grid q CHUNK: how many q-points of the fine set have their
    # f(H(q)) built and decomposed at once.  0 (DEFAULT) = N_q_co, the
    # COARSE k-point count prod(kgrid_co) — not a bare constant.  fH_R is
    # (nk_co, rank, rank) face-sharded, so one chunk of N_q_co q-points is
    # byte-for-byte the same per-rank residency as fH_R itself: a deck that
    # could build fH_R at all can afford exactly one such chunk, and the
    # fine-grid pass then completes for ANY N_q_fi.  Raise it to trade
    # memory for fewer collectives, lower it on a memory-tight rank.  It is
    # a FLOOR: rounded up to a multiple of the device count so the q axis
    # stays shardable (bse_setup pads with sharding_fit.padded_extent).
    # Ignored by the distributed-eigh path, whose chunk is 1 by
    # construction.  See bandstructure.bse_setup.compute_wfns_fi.
    "wfn_fi_q_chunk": 0,

    # Deck hygiene.  False (DEFAULT): a key that is not in _DEFAULTS and
    # not covered by a legacy/deprecation branch is reported in one
    # aggregated rank-0 warning and ignored.  True: the same condition
    # raises ValueError naming every unknown key — for CI decks and fresh
    # runs where a typo must not silently drop a knob.
    "strict_keys": False,
}

# Deck keys REMOVED from _DEFAULTS but still handled by an explicit
# legacy/deprecation branch in ``read_lorrax_input`` (raise or dedicated
# DeprecationWarning).  The unknown-key check skips these so one deck key
# never draws two messages.  Keys that are deprecated but still in
# _DEFAULTS (self_consistent, sigma_at_dft_energies, use_ffi_io) need no
# entry here.
_LEGACY_DECK_KEYS = frozenset({
    "use_shipped_minimax_tables",   # refused with replacement named
    "chunk_size",                   # warn-and-ignore (planner-owned)
    "output_file",                  # warn-and-ignore (sigma_diag_file)
    "eqp_output_file",              # warn-and-ignore (auto eqp0/eqp1)
})

# Keys whose string values should be lowercased and stripped
_NORMALIZE_STR = {
    "compute_mode",
    "qp_solver",
    "sc_accelerator",
    "sc_eigh",
    "wcoul0_source", "screening_method", "minimax_energy_reference",
    "sigma_omega_accumulation", "sigma_omega_layout", "fermi_reference",
    "w_dyson_solver",
    "ppm_invalid_mode",
    "ppm_model",
    "ppm_probe_chi_reuse",
    # distributed-linalg backend axes (consumed both via LorraxConfig and
    # directly from the params dict by htransform / exciton_bands).
    "eigh_backend",
}

# Tri-state booleans: _DEFAULTS value is None (= unset), an explicit
# input-file value parses as bool.  The parse loop needs the set because
# ``default is None`` otherwise means "nullable float".
_NULLABLE_BOOL = frozenset({"use_ffi_io"})


# ---------------------------------------------------------------------------
#  Input file parser
# ---------------------------------------------------------------------------

def read_lorrax_input(filename: str) -> dict:
    """Parse a LORRAX input file ([cohsex] section) into a params dict.

    Handles the QE-style K_POINTS block and strips it before INI parsing.
    All keys use ``_DEFAULTS`` for fallback values — no duplicate definitions.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Locate [cohsex] section
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('[cohsex]'):
            start = i
            break
    if start is None:
        for i, line in enumerate(lines):
            if re.match(r"\s*\[.*\]", line):
                start = i
                break
    end = len(lines)

    # Locate optional K_POINTS block
    kp_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("k_points"):
            kp_idx = i
            break
    kp_end = None
    if kp_idx is not None and kp_idx + 1 < len(lines):
        try:
            seg_count = int(lines[kp_idx + 1].strip().split()[0])
        except Exception:
            seg_count = 0
        kp_end = min(len(lines), kp_idx + 2 + max(seg_count, 0))

    if start is not None:
        for j in range(start + 1, len(lines)):
            if re.match(r"\s*\[.*\]", lines[j]):
                end = j
                break
        # Strip K_POINTS from INI text
        if kp_idx is not None and start <= kp_idx < end:
            section_lines = lines[start:kp_idx] + lines[(kp_end or kp_idx + 1):end]
        else:
            section_lines = lines[start:end]

        # inline_comment_prefixes so 'key = off  # note' parses to 'off', not
        # 'off  # note' (the latter silently voided flags — a real footgun).
        parser = configparser.ConfigParser(inline_comment_prefixes=('#',))
        parser.read_string(''.join(section_lines))
        section = parser["cohsex"] if "cohsex" in parser else parser[parser.sections()[0]]

        # Legacy key check
        if section.get("use_shipped_minimax_tables", fallback=None) is not None:
            raise ValueError(
                "Input key 'use_shipped_minimax_tables' is no longer supported. "
                "Use 'regenerate_minimax_tables = true/false' instead.")
        # ``chunk_size`` (legacy band-chunk knob) was a no-op: its only
        # consumer wrote ``meta.chunk_size``, which nothing ever read —
        # chunk sizing is owned by the gflat planner.  Dropped 2026-07-09.
        if section.get("chunk_size", fallback=None) is not None:
            import warnings
            warnings.warn(
                "Input key 'chunk_size' is no longer supported and will be "
                "ignored (it was a no-op; chunk sizing is planner-owned — "
                "see 'gflat_chunk_size' / 'band_chunk_size').",
                DeprecationWarning, stacklevel=2,
            )
        for legacy_key in ("output_file", "eqp_output_file"):
            if section.get(legacy_key, fallback=None) is not None:
                import warnings
                warnings.warn(
                    f"Input key '{legacy_key}' is no longer supported and "
                    f"will be ignored.  ``output_file`` (LORRAX-native eqp0) "
                    f"is now ``sigma_diag_file`` (defaults to "
                    f"``sigma_diag.dat``); BGW-format ``eqp0.dat`` and "
                    f"``eqp1.dat`` (with Z-linearization) are written "
                    f"automatically.  Remove '{legacy_key}' from your "
                    f"input file.",
                    DeprecationWarning, stacklevel=2,
                )
        # Deprecated qp_solver aliases (still honored via auto-resolution;
        # see ``LorraxConfig.qp_solver``).
        for legacy_key, replacement in (
            ("self_consistent", "qp_solver = self_consistent"),
            ("sigma_at_dft_energies", "qp_solver = one_shot_dft (the default)"),
        ):
            if section.get(legacy_key, fallback=None) is not None:
                import warnings
                warnings.warn(
                    f"Input key '{legacy_key}' is deprecated; it is honored "
                    f"via ``qp_solver = auto`` resolution.  Set "
                    f"'{replacement}' instead.",
                    DeprecationWarning, stacklevel=2,
                )

        # 'use_ffi_io' deprecation warns ONCE, at the resolution site in
        # ``from_input_file`` (which distinguishes the true/false/overridden
        # cases).  A second generic warning here made one deck emit two
        # overlapping DeprecationWarnings for one key (audit fix/zq
        # 2026-07-28).

        # REMOVED keys (owner-approved deletions, 2026-07-31; these behave
        # like any other unknown deck key — reported by the unknown-key
        # check below, never steering anything): ``isdf_memory_mode``
        # (two-plan W cleanup — the W Dyson solve is selected by
        # w_dyson_solver=local|distributed) and the legacy aliases
        # ``cusolvermp_charge``/``cusolvermp_lu`` (use distributed_cholesky
        # / distributed_lu).

        # --- Unknown-key check -----------------------------------------
        # Every key in the deck that is neither in ``_DEFAULTS`` nor
        # handled by one of the explicit legacy branches above is reported
        # in ONE aggregated rank-0 warning (key, line number, "ignored").
        # Silently dropping unknown keys turned every typo and every stale
        # doc into silent wrong physics.  Warn, don't refuse — archived
        # decks carry dead keys — unless the deck opts in via
        # ``strict_keys = true``, which upgrades the warning to a
        # ValueError naming all unknown keys at once.
        unknown = [k for k in section
                   if k not in _DEFAULTS and k not in _LEGACY_DECK_KEYS]
        if unknown:
            located = []
            for key in unknown:
                lineno = next(
                    (i + 1 for i in range(start, end)
                     if re.match(rf"\s*{re.escape(key)}\s*[=:]",
                                 lines[i], re.IGNORECASE)),
                    None)
                where = f"line {lineno}" if lineno is not None else "line ?"
                located.append(f"{key} ({where})")
            if section.getboolean(
                    "strict_keys",
                    fallback=bool(_DEFAULTS["strict_keys"])):
                raise ValueError(
                    f"strict_keys = true: {len(unknown)} unknown deck "
                    f"key(s) in {filename}:\n"
                    + "\n".join(f"    {loc}: not a recognized deck key"
                                for loc in located))
            msg = (f"read_lorrax_input: {len(unknown)} unrecognized deck "
                   f"key(s) in {filename}:\n"
                   + "\n".join(
                       f"    {loc}: ignored — not a recognized deck key"
                       for loc in located))
            # Rank-0-equivalent stdout.  ``process_rank`` is jax-free-safe
            # (lazy jax import inside, falls back to 0 when jax is absent
            # or uninitialised) — a downhill L1→L3 import, function-scoped
            # so this parser stays importable without the common package
            # fully initialised.
            try:
                from common.collectives import process_rank
                _rank = process_rank()
            except Exception:                              # noqa: BLE001
                _rank = 0
            if _rank == 0:
                print(msg)

        # Build params from _DEFAULTS, overriding with parsed values
        params = {}
        for key, default in _DEFAULTS.items():
            raw = section.get(key, fallback=None)
            if raw is None:
                params[key] = default
            elif key in _NULLABLE_BOOL:
                # Tri-state boolean (default None = unset); an explicit
                # value parses as bool.  Currently only the deprecated
                # ``use_ffi_io``.
                params[key] = section.getboolean(key)
            elif isinstance(default, bool):
                params[key] = section.getboolean(key)
            elif isinstance(default, int):
                params[key] = section.getint(key)
            elif isinstance(default, float):
                params[key] = section.getfloat(key)
            elif default is None:
                # Nullable float (vhead, whead_0freq, etc.)
                params[key] = section.getfloat(key, fallback=None)
            else:
                params[key] = str(raw)
            if key in _NORMALIZE_STR and isinstance(params[key], str):
                params[key] = params[key].strip().lower()
    else:
        params = dict(_DEFAULTS)

    # Parse optional QE K_POINTS block
    if kp_idx is not None:
        j = kp_idx + 1
        try:
            nseg = int(lines[j].strip().split()[0])
        except Exception:
            nseg = 0
        segments = []
        for k in range(nseg):
            row_idx = j + 1 + k
            if row_idx >= len(lines):
                break
            row_full = lines[row_idx].rstrip('\n')
            label = None
            for marker in ('#', '!', ';'):
                if marker in row_full:
                    label = row_full.split(marker, 1)[1].strip() or None
                    row_full = row_full.split(marker, 1)[0]
                    break
            row = row_full.strip()
            if not row:
                continue
            parts = row.split()
            if len(parts) < 3:
                continue
            segments.append({
                "k": [float(parts[0]), float(parts[1]), float(parts[2])],
                "n": int(parts[3]) if len(parts) >= 4 else 1,
                "label": label,
            })
        if segments:
            params["kpoints_crystal_b"] = {"segments": segments}

    return params


# Backward-compatible alias
read_cohsex_input = read_lorrax_input


# ---------------------------------------------------------------------------
#  LorraxConfig
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Sub-dataclasses (each frozen, attribute-accessed via ``config.<group>.X``)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilePaths:
    """Output filenames + non-WFN inputs.  Resolved to absolute paths."""
    wfn_file: str
    centroids_file: str
    # Bispinor: optional Gordon-current-density centroid file for μ_L=1,2,3.
    # ``None`` falls back to the scalar charge-only path (CC tile only).
    centroids_file_current: str | None
    kin_ion_file: str
    sigma_diag_file: str
    eqp0_file: str
    eqp1_file: str
    sigma_omega_h5_file: str


@dataclass(frozen=True)
class HeadConfig:
    """q→0 Coulomb-head sources, BGW vcoul override, bare-cutoff knobs.

    All Coulomb-at-small-q tweaks live here.  Σ head plumbing
    (``wcoul0_*``, ``vhead``/``whead_*``) is consumed by
    :class:`gw.head_correction.HeadResolver`; the BGW vcoul override is
    purely diagnostic (matches BGW's per-G mini-BZ averaging exactly for
    bit-reproducible comparisons).
    """
    wcoul0_source: str            # "s_tensor" | "epshead"
    wcoul0_eta: float
    vhead: float | None           # explicit override v_h[ω=0]
    whead_0freq: float | None     # explicit override W_h[ω=0]
    whead_imfreq: float | None    # explicit override W_h[iω_p]
    mc_average_vcoul_body: bool
    head_minibz_average: bool      # per-Q mini-BZ head cell-average (default off)
    bare_coulomb_cutoff: float | None
    zeta_cutoff: float | None
    use_bgw_vcoul: bool
    bgw_vcoul_file: str | None
    bgw_vcoul_sym_wfn: str | None


@dataclass(frozen=True)
class ScreeningConfig:
    """χ₀ / W screening: method choice + minimax-quadrature knobs."""
    method: str                   # "minimax" (only one currently)
    minimax_target_error: float
    minimax_max_nodes: int
    regenerate_minimax_tables: bool
    minimax_energy_reference: str  # "midgap" | "vbm"


@dataclass(frozen=True)
class PPMConfig:
    """Plasmon-pole model + Σ_c(ω) output grid + on-shell options.

    Single grouped home for everything PPM/Σ_c-related: the pole-fit
    ansatz, the probe-ω choice, the analytic head-pole override, the
    σ-quadrature minimax tolerances, the ω-grid for the output, and
    the post-hoc on-shell evaluation knobs.
    """
    # --- Model selection ---
    model: str                    # "gn" | "hl" — picked by ComputeMode usually
    omega_p: float                # probe ω (Ry); imag for GN, real for HL
    fallback_omega: float
    head_omega_h_ry: float | None # override Ω_h directly (BGW comparisons)
    #: Probe-χ₀ reuse: "off" (dedicated probe quadrature, exact historical
    #: path) | "auto" (weights-only refit on the static τ nodes, probe χ₀
    #: folded into the static sweep when the error gate passes — see
    #: _DEFAULTS["ppm_probe_chi_reuse"]).
    probe_chi_reuse: str

    # --- σ-quadrature minimax ---
    sigma_target_error: float
    sigma_max_nodes: int

    # --- ω-grid for Σ_c(ω) output (eV) ---
    omega_min_ev: float
    omega_max_ev: float
    omega_step_ev: float
    regularization_ev: float
    window_edge_factor: float
    omega_batch_size: int
    omega_accumulation: str       # "auto" | "kij"
    #: Σ_c(ω) end-of-stage layout: "replicated" gathers the full cube onto
    #: every rank (historical path); "sharded" keeps it (m_X, n_Y)-tiled on
    #: the existing mesh and consumers read the tiles directly (movement-only,
    #: bit-identical outputs; see _DEFAULTS["sigma_omega_layout"]).
    omega_layout: str             # "replicated" | "sharded"

    # --- on-shell evaluation knobs ---
    invalid_mode: str             # "zero" | "2ry" | "static_limit" | "infinity"(alias)
    fermi_reference: str          # "midgap" | "vbm"
    sigma_at_dft_extrapolate: bool
    sigma_at_dft_energies: bool

    def __post_init__(self):
        # Validate scalar knobs once, at the parse site (values are already
        # normalized in ``from_input_file``).  Capability gating for
        # invalid_mode ('imaginary' → NotImplementedError, needs a
        # complex-Ω path) stays in the Σ^c kernel — this checks only
        # that the *value* is recognized.
        if self.omega_step_ev <= 0.0:
            raise ValueError("ppm.omega_step_ev must be > 0.")
        if self.omega_max_ev < self.omega_min_ev:
            raise ValueError("ppm.omega_max_ev must be >= ppm.omega_min_ev.")
        if self.fermi_reference not in ("vbm", "midgap"):
            raise ValueError("ppm.fermi_reference must be 'vbm' or 'midgap'.")
        if self.omega_accumulation == "kij_stream":
            # REMOVED mode (2026-07-31): the single-process streamed-h5
            # accumulator is gone.  Refuse the removed VALUE of a known
            # key rather than silently rerouting an old deck.
            raise ValueError(
                "sigma_omega_accumulation = kij_stream was REMOVED: the "
                "single-process streamed-h5 accumulator is gone "
                "(superseded by host-tile accumulation and "
                "sigma_omega_layout=sharded for cubes that do not fit).  "
                "Use 'kij' or 'auto'.")
        if self.omega_accumulation not in ("auto", "kij"):
            raise ValueError(
                "ppm.omega_accumulation must be auto/kij.")
        if self.omega_layout not in ("replicated", "sharded"):
            raise ValueError(
                "sigma_omega_layout must be 'replicated' or 'sharded'; "
                f"got {self.omega_layout!r}.")
        if self.invalid_mode not in (
            "zero", "skip", "2ry", "static_limit", "infinity", "imaginary"
        ):
            raise ValueError(
                f"ppm.invalid_mode: unknown value {self.invalid_mode!r}")
        if self.omega_batch_size < 1:
            raise ValueError("ppm.omega_batch_size must be >= 1.")
        if self.probe_chi_reuse not in ("off", "auto"):
            raise ValueError(
                "ppm_probe_chi_reuse must be 'off' or 'auto'; "
                f"got {self.probe_chi_reuse!r}.")


@dataclass(frozen=True)
class SCConfig:
    """Self-consistency loop knobs (read only when qp_solver=self_consistent).

    Promoted from the ``LORRAX_SC_*`` env vars (NEXT_TARGETS #11); the
    envs are still honored as deprecated overrides at config construction
    (``from_input_file`` prints a note when one is active).

    - ``max_iter`` / ``tol_ev``: loop length and RMS-ΔE convergence (eV).
    - ``accelerator``: ``"rcrop"`` (Anderson-style restart-CROP, default —
      required for QSGW's typical 2-cycle Jacobian) or ``"linear"``
      (plain α-mixing, diagnostic).  rCROP makes TWO ``gw_iteration_map``
      calls per accelerator iteration (trial + residual).
    - ``history_depth``: rCROP history (m=5 is BGW's QSGW default).
    - ``mixing``: linear-mixing α (``accelerator="linear"`` only).
    - ``dump_dir``: per-iteration E-history .npy dump dir (None = off).
    - ``eigh``: which eigh diagonalises the ``(nk, nb, nb)`` carry each
      iteration — ``"native"`` (k-sharded batch: one WHOLE ``(nb, nb)``
      tile per device), ``"distributed"`` (one tile spread over the mesh),
      or ``"auto"``.  A LAYOUT choice: it does not change the physics and
      it is deliberately not a side effect of ``density_self_consistent``,
      which is what used to select it.  Resolution lives in
      ``sc_iteration._resolve_sc_eigh``.
    """
    max_iter: int
    tol_ev: float
    accelerator: str      # "rcrop" | "linear"
    history_depth: int
    mixing: float
    dump_dir: str | None
    eigh: str = "auto"    # "auto" | "native" | "distributed"

    def __post_init__(self):
        if self.max_iter < 1:
            raise ValueError("sc_max_iter must be >= 1.")
        if self.tol_ev <= 0.0:
            raise ValueError("sc_tol_ev must be > 0.")
        if self.accelerator not in ("rcrop", "linear"):
            raise ValueError(
                f"sc_accelerator must be 'rcrop' or 'linear'; "
                f"got {self.accelerator!r}.")
        if self.history_depth < 1:
            raise ValueError("sc_history_depth must be >= 1.")
        if not (0.0 < self.mixing <= 1.0):
            raise ValueError("sc_mixing must be in (0, 1].")
        if self.eigh not in ("auto", "native", "distributed"):
            raise ValueError(
                f"sc_eigh must be 'auto', 'native' or 'distributed'; "
                f"got {self.eigh!r}.")


@dataclass(frozen=True)
class MemoryConfig:
    """Per-device memory budget + chunk sizing + AOT chunk-chooser flag.

    ``memory_per_device_gb=0`` triggers GPU auto-detection at config
    construction time.  ``chunk_target_utilization`` is sourced from the
    ``ISDF_CHUNK_TARGET_UTILIZATION`` env var (default 0.97).
    ``zct_stage_cap_gb`` similarly from
    ``ISDF_ZCT_STAGE_CAP_GB`` / ``ISDF_ZCT_STAGE_CAP_FRAC``.
    """
    per_device_gb: float
    chunk_target_utilization: float
    band_chunk_size: int
    r_chunk_override: int         # 0 = auto
    zct_stage_cap_gb: float | None
    gflat_chunk_size: int         # 0 = one-shot (or planner-picked)
    vq_g_chunk_size: int          # 0 = auto _pick_g_chunk(ngkmax)


@dataclass(frozen=True)
class BackendConfig:
    """Three-axis backend selection: I/O + ψ(G) lifecycle + W Dyson plan.

    All three knobs were previously orthogonal-sounding boolean/string
    flags in different namespaces (``use_ffi_io`` / ``gspace_mode`` /
    ``isdf_memory_mode``) that secretly toggled FFI paths.  Grouped here
    so :meth:`summary` can print one line at startup describing what's
    actually active per channel.
    """
    slab_io: SlabIOBackend
    gspace_io: GspaceIO
    w_dyson_solver: str  # "local" | "distributed" (normalized; W Dyson plan)
    distributed_cholesky: str  # "auto" | "off" | "cusolvermp" | "slate"
    distributed_lu: str        # "auto" | "off" | "cusolvermp" | "scalapack"
    eigh_backend: str          # resolved: auto|off|distributed|cusolvermp|
                               #           slate|scalapack (use_low_mem_eigh
                               #           already folded in)
    use_low_mem_eigh: bool     # what the deck ASKED for, kept for the banner
    zeta_ridge: float          # charge-CCT Tikhonov ridge ε (rel. to tr/n)
    charge_zeta_solve: str     # "rank_truncate" | "cholesky"
    distributed_zeta_solve: str  # "auto"|"replicated"|"per_q"|"distributed"
    zeta_rcond: float          # rank-truncation cutoff (·λ_max)
    transverse_zeta_solve: str  # "ridge" | "rank_truncate" (bispinor ζ_T)
    transverse_zeta_rcond: float  # transverse cut τ (·|λ|_max)
    gamma_contract_mode: str  # "take" | "einsum" | "scan"

    def summary(self) -> str:
        """One-line "what's active" for the run banner."""
        return (
            f"backend: slab_io={self.slab_io.value}, "
            f"gspace_io={self.gspace_io.value}, "
            + (f"w_dyson_solver={self.w_dyson_solver}, "
               if self.w_dyson_solver != "local" else "")
            + f"distributed_cholesky={self.distributed_cholesky}, "
            f"distributed_lu={self.distributed_lu}, "
            + (f"eigh_backend={self.eigh_backend}"
               + (" (use_low_mem_eigh)" if self.use_low_mem_eigh else "")
               + ", " if self.eigh_backend != "auto" else "")
            + f"charge_zeta_solve={self.charge_zeta_solve}"
            + (f"(rcond={self.zeta_rcond:g})"
               if self.charge_zeta_solve == 'rank_truncate' else '')
            + (f", zeta_ridge={self.zeta_ridge:g}"
               if self.zeta_ridge else '')
            + (f", distributed_zeta_solve={self.distributed_zeta_solve}"
               if self.distributed_zeta_solve != 'auto' else '')
            + (f", transverse_zeta_solve={self.transverse_zeta_solve}"
               f"(rcond={self.transverse_zeta_rcond:g})"
               if self.transverse_zeta_solve != 'ridge' else '')
            + f", gamma_contract={self.gamma_contract_mode}"
        )


@dataclass(frozen=True)
class DebugConfig:
    """Debug-only flags + auxiliary output filenames."""
    sigma_freq_debug_output: bool
    sigma_freq_debug_file: str
    write_wfn_h5: bool


@dataclass(frozen=True)
class BSEConfig:
    """BSE interpolation setup (htransform-driven fine-k wfn recovery).

    See ``bandstructure.bse_setup.compute_wfns_fi``.  ``get_centroids_fi``
    is the master gate; if False the rest is unused.
    """
    get_centroids_fi: bool
    wfn_fi_min: int
    wfn_fi_max: int
    kgrid_fi: str
    wfn_fi_q_chunk: int   # 0 = N_q_co (prod(kgrid_co)); see compute_wfns_fi


@dataclass(frozen=True)
class LorraxConfig:
    """Unified, immutable configuration for a LORRAX GW calculation.

    Created once via :meth:`from_input_file` and threaded through the
    entire driver.  Top-level fields are ``hot-path`` reads (system
    geometry + the orthogonal mode flags); group sub-dataclasses
    organise the remaining ~70 input keys along the same axes the
    input file's section comments already use.

    Access pattern::

        config.compute_mode           # -> ComputeMode enum
        config.head.wcoul0_source     # head plumbing
        config.ppm.omega_p            # PPM probe ω
        config.backend.slab_io        # which writer backend
        config.debug.sigma_freq_debug_output

    See module docstring for the full grouping.  ``cohsex.in`` keys
    are unchanged — input files written for prior versions still parse
    (the factory unflattens the dict into sub-dataclasses).
    """

    # --- System geometry (top-level; hot path) ---
    nval: int
    ncond: int
    nband: int
    sys_dim: int
    density_self_consistent: bool
    sc_on_ibz: bool
    #: auto | stored | isdf | gspace — see HARTREE_SOURCES.
    hartree_source: str

    # --- Core mode flags (top-level; hot path) ---
    restart: bool
    compute_mode_raw: str         # "auto" | one of ComputeMode.value strings
    qp_solver_raw: str            # "auto" | one of QPSolver.value strings
    do_screened: bool
    bispinor: bool
    do_G0: bool
    self_consistent: bool         # deprecated alias; ``qp_solver`` is canonical
    use_ppm_sigma: bool           # legacy mirror; ``compute_mode`` is canonical
    no_degen_averaging: bool
    degen_avg_tol_ry: float

    # --- Sub-dataclass groups (everything else) ---
    paths: FilePaths
    head: HeadConfig
    screening: ScreeningConfig
    ppm: PPMConfig
    sc: SCConfig
    memory: MemoryConfig
    backend: BackendConfig
    debug: DebugConfig
    bse: BSEConfig

    # --- Optional parsed blocks ---
    kpoints_crystal_b: dict | None = None

    # --- Input directory (for resolving relative paths at runtime) ---
    input_dir: str = ""

    # ------------------------------------------------------------------
    #  Derived config objects
    # ------------------------------------------------------------------

    @property
    def compute_mode(self) -> ComputeMode:
        """Resolve ``compute_mode`` from explicit input or legacy flags.

        ``compute_mode = auto`` (the default) infers from
        ``do_screened`` / ``use_ppm_sigma`` / ``ppm.model``.  An explicit
        setting overrides them; the legacy fields are still parsed for
        back-compat but the enum is the load-bearing axis the driver
        pivots on.
        """
        raw = (self.compute_mode_raw or "auto").strip().lower()
        if raw == "auto":
            if self.use_ppm_sigma:
                if not self.do_screened:
                    raise ValueError(
                        "use_ppm_sigma=true requires do_screened=true."
                    )
                return (
                    ComputeMode.HL_PPM
                    if str(self.ppm.model).strip().lower() == "hl"
                    else ComputeMode.GN_PPM
                )
            return ComputeMode.COHSEX if self.do_screened else ComputeMode.X_ONLY
        try:
            explicit = ComputeMode(raw)
        except ValueError as exc:
            raise ValueError(
                f"compute_mode={raw!r} invalid; expected one of: "
                f"{', '.join(m.value for m in ComputeMode)}, or 'auto'."
            ) from exc
        # The enum is load-bearing: an explicit screened mode contradicts
        # the legacy ``do_screened = false``.  (Explicit ``x_only`` simply
        # wins over the do_screened default — the driver derives its
        # screening entirely from the mode.)
        if explicit is not ComputeMode.X_ONLY and not self.do_screened:
            raise ValueError(
                f"compute_mode={raw!r} requires screening, but the legacy "
                f"flag do_screened=false was also set. Remove one of the two."
            )
        return explicit

    @property
    def qp_solver(self) -> QPSolver:
        """Resolve ``qp_solver`` from explicit input or legacy flags.

        ``qp_solver = auto`` (the default) resolves:

        1. ``self_consistent = true`` → ``SELF_CONSISTENT`` (deprecated
           key, still honored);
        2. else → ``ONE_SHOT_DFT`` — standard G0W0 is the default.
           (The deprecated ``sigma_at_dft_energies = true`` alias also
           lands here: its intended meaning — authoritative at-DFT QP
           evaluation — IS the default.)

        An explicit setting overrides the legacy flags, mirroring how
        ``compute_mode`` absorbs ``do_screened`` / ``use_ppm_sigma``.

        Validation (mutually inconsistent axis combinations):

        - ``fixed_point`` × static mode → error (no ω-grid to solve on;
          a silent no-op would blur the axis).
        """
        raw = (self.qp_solver_raw or "auto").strip().lower()
        if raw == "auto":
            solver = (QPSolver.SELF_CONSISTENT if self.self_consistent
                      else QPSolver.ONE_SHOT_DFT)
        else:
            try:
                solver = QPSolver(raw)
            except ValueError as exc:
                raise ValueError(
                    f"qp_solver={raw!r} invalid; expected one of: "
                    f"{', '.join(s.value for s in QPSolver)}, or 'auto'."
                ) from exc
        mode = self.compute_mode
        if solver is QPSolver.FIXED_POINT and not mode.is_dynamic:
            raise ValueError(
                f"qp_solver=fixed_point requires a dynamic compute_mode "
                f"(gn_ppm / hl_ppm); static Σ ({mode.value}) has no ω-grid "
                f"to solve E = h0 + ReΣ(E) on.  Use one_shot_dft (identical "
                f"physics for static Σ) or self_consistent.")
        return solver

    @property
    def minimax_config(self):
        """Math-internal :class:`gw.minimax_config.MinimaxConfig` for χ₀."""
        from .minimax_config import MinimaxConfig
        return MinimaxConfig(
            target_error=self.screening.minimax_target_error,
            max_nodes=self.screening.minimax_max_nodes,
            regenerate_tables=self.screening.regenerate_minimax_tables,
            energy_reference=self.screening.minimax_energy_reference,
        )

    @property
    def sigma_quadrature_config(self):
        """Math-internal :class:`gw.minimax_config.MinimaxConfig` for Σ^c."""
        from .minimax_config import MinimaxConfig
        return MinimaxConfig(
            target_error=self.ppm.sigma_target_error,
            max_nodes=self.ppm.sigma_max_nodes,
            crossing_max_nodes=max(500, self.ppm.sigma_max_nodes),
            crossing_eps_q=1.0e-3,
            regenerate_tables=self.screening.regenerate_minimax_tables,
        )

    @property
    def omega_grid_ev(self):
        """Σ_c(ω) frequency grid in eV (length-stable single formula).

        ``n = floor((max−min)/step + 0.5) + 1`` — the Ry grid is derived
        from this one by division so the two can never disagree in length
        or accumulate independent float-step rounding.
        """
        p = self.ppm
        n = int(np.floor(
            (p.omega_max_ev - p.omega_min_ev) / p.omega_step_ev + 0.5)) + 1
        return p.omega_min_ev + p.omega_step_ev * np.arange(n, dtype=np.float64)

    @property
    def omega_grid_ry(self):
        """Σ_c(ω) frequency grid in Rydberg (derived from the eV grid)."""
        return self.omega_grid_ev / RYD_TO_EV

    # ------------------------------------------------------------------
    #  Back-compat aliases — the FFI/IO group changed semantics (bool /
    #  string → enum), so callers that still want the old names get
    #  coerced views.  New code should use ``config.backend.<field>`` /
    #  ``config.memory.<field>`` etc. directly.
    # ------------------------------------------------------------------

    @property
    def use_ffi_io(self) -> bool:
        """Legacy ``use_ffi_io: bool`` semantic — True for either of the
        per-rank-parallel-write PHDF5 backends (``PHDF5_FFI`` on GPU,
        ``PHDF5_HOST`` on CPU), False for the allgather fallback.
        Callers use this to branch between rank-0-gather and per-rank
        local-shard code paths; both PHDF5 variants share the latter.
        """
        return self.backend.slab_io in (
            SlabIOBackend.PHDF5_FFI, SlabIOBackend.PHDF5_HOST)

    @property
    def gspace_mode(self) -> str:
        """Legacy ``gspace_mode: str`` view of ``backend.gspace_io``."""
        return self.backend.gspace_io.value

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_input_file(cls, filename: str, *, print_fn=print) -> LorraxConfig:
        """Parse input file and resolve runtime settings (memory, env vars).

        Replaces ``read_cohsex_input`` + ``resolve_runtime_config`` +
        path resolution in one call.  Returns a ``LorraxConfig`` with
        sub-dataclasses fully populated.
        """
        from file_io import resolve_input_paths

        params = read_lorrax_input(filename)
        input_dir = os.path.dirname(os.path.abspath(filename))
        resolve_input_paths(params, input_dir)

        # --- Memory auto-detection ---
        memory_per_device_gb = float(params.get("memory_per_device_gb", 0.0))
        if memory_per_device_gb <= 0:
            from common.gpu_utils import get_device_memory_gb
            memory_per_device_gb = get_device_memory_gb()
            print_fn(
                f"  Auto-detected memory budget: {memory_per_device_gb:.2f} GB/device"
            )

        # --- Chunk utilization from env ---
        # 0.0 (default) = auto: the planner uses its ns²-aware default
        # (higher for scalar, lower for bispinor's 4× pair density).  A
        # positive env value overrides it, clamped to [0.85, 1.0].
        # ``env_float`` announces a non-numeric value instead of swallowing
        # it — the bare ``except Exception`` here left the user believing a
        # utilization was in force when it was not.
        chunk_utilization = env_float("ISDF_CHUNK_TARGET_UTILIZATION", 0.0,
                                      print_fn=print_fn)
        if chunk_utilization > 0:
            chunk_utilization = max(0.85, min(1.0, chunk_utilization))

        # --- ZCT stage cap from env ---
        # ``total_gb`` is the PHYSICAL card; 0.0 when the backend has no
        # device memory (CPU).  Passing it in keeps the decision — and
        # every "no cap, and here is why" announcement — in one pure,
        # testable place instead of behind a backend guard that used to
        # skip the fraction branch in silence.
        import jax
        _zct_total_gb = 0.0
        if jax.default_backend() in ("gpu", "cuda"):
            from common.gpu_utils import get_device_memory_info
            _zct_total_gb = float(
                get_device_memory_info().get("total_gb", 0.0))
        zct_stage_cap_gb = resolve_zct_stage_cap(
            os.environ.get("ISDF_ZCT_STAGE_CAP_GB"),
            os.environ.get("ISDF_ZCT_STAGE_CAP_FRAC"),
            per_device_gb=memory_per_device_gb,
            total_gb=_zct_total_gb,
            print_fn=print_fn)

        def _g(key):
            return params.get(key, _DEFAULTS.get(key))

        # --- Build sub-dataclasses ---
        cents_curr = _g("centroids_file_current")
        cents_curr_resolved = str(cents_curr) if cents_curr else None
        paths = FilePaths(
            wfn_file=str(_g("wfn_file")),
            centroids_file=str(_g("centroids_file")),
            centroids_file_current=cents_curr_resolved,
            kin_ion_file=str(_g("kin_ion_file")),
            sigma_diag_file=str(_g("sigma_diag_file")),
            eqp0_file=str(_g("eqp0_file")),
            eqp1_file=str(_g("eqp1_file")),
            sigma_omega_h5_file=str(_g("sigma_omega_h5_file")),
        )
        head = HeadConfig(
            wcoul0_source=str(_g("wcoul0_source")).strip().lower(),
            wcoul0_eta=float(_g("wcoul0_eta") or 0.0),
            vhead=_g("vhead"),
            whead_0freq=_g("whead_0freq"),
            whead_imfreq=_g("whead_imfreq"),
            mc_average_vcoul_body=bool(_g("mc_average_vcoul_body")),
            head_minibz_average=bool(_g("head_minibz_average")),
            bare_coulomb_cutoff=_g("bare_coulomb_cutoff"),
            zeta_cutoff=_g("zeta_cutoff"),
            use_bgw_vcoul=bool(_g("use_bgw_vcoul")),
            bgw_vcoul_file=(str(_g("bgw_vcoul_file")) or None),
            bgw_vcoul_sym_wfn=(str(_g("bgw_vcoul_sym_wfn")) or None),
        )
        screening = ScreeningConfig(
            method=str(_g("screening_method")).strip().lower(),
            minimax_target_error=float(_g("minimax_target_error")),
            minimax_max_nodes=int(_g("minimax_max_nodes")),
            regenerate_minimax_tables=bool(_g("regenerate_minimax_tables")),
            minimax_energy_reference=str(_g("minimax_energy_reference")).strip().lower(),
        )
        ppm = PPMConfig(
            model=str(_g("ppm_model")).strip().lower(),
            omega_p=float(_g("ppm_omega_p")),
            fallback_omega=float(_g("ppm_fallback_omega")),
            head_omega_h_ry=(
                float(_g("ppm_head_omega_h_ry"))
                if _g("ppm_head_omega_h_ry") is not None else None),
            probe_chi_reuse=str(_g("ppm_probe_chi_reuse")).strip().lower(),
            sigma_target_error=float(_g("ppm_sigma_target_error")),
            sigma_max_nodes=int(_g("ppm_sigma_max_nodes")),
            omega_min_ev=float(_g("sigma_omega_min_ev")),
            omega_max_ev=float(_g("sigma_omega_max_ev")),
            omega_step_ev=float(_g("sigma_omega_step_ev")),
            regularization_ev=float(_g("sigma_regularization_ev")),
            window_edge_factor=float(_g("sigma_window_edge_factor")),
            omega_batch_size=int(_g("sigma_omega_batch_size")),
            omega_accumulation=str(_g("sigma_omega_accumulation")).strip().lower(),
            omega_layout=str(_g("sigma_omega_layout")).strip().lower(),
            invalid_mode=str(_g("ppm_invalid_mode") or "static_limit").strip().lower(),
            fermi_reference=str(_g("fermi_reference")).strip().lower(),
            sigma_at_dft_extrapolate=bool(_g("sigma_at_dft_extrapolate")),
            sigma_at_dft_energies=bool(_g("sigma_at_dft_energies")),
        )
        # SC loop knobs.  The LORRAX_SC_* env vars are deprecated overrides
        # of the sc_* input keys (kept so existing sweep scripts run
        # unchanged); a note is printed whenever one is active.
        def _sc_env(env_key: str, cast, file_val, input_key: str):
            raw_env = os.environ.get(env_key)
            if raw_env is None or raw_env == "":
                return file_val
            val = cast(raw_env)
            print_fn(
                f"  [config] {env_key}={raw_env} (deprecated env override; "
                f"set '{input_key} = {raw_env}' in cohsex.in instead)")
            return val

        sc = SCConfig(
            max_iter=_sc_env(
                "LORRAX_SC_MAX_ITER", int, int(_g("sc_max_iter")),
                "sc_max_iter"),
            tol_ev=_sc_env(
                "LORRAX_SC_TOL_EV", float, float(_g("sc_tol_ev")),
                "sc_tol_ev"),
            accelerator=_sc_env(
                "LORRAX_SC_ACCEL", lambda s: str(s).strip().lower(),
                str(_g("sc_accelerator")).strip().lower(), "sc_accelerator"),
            history_depth=_sc_env(
                "LORRAX_SC_DEPTH", int, int(_g("sc_history_depth")),
                "sc_history_depth"),
            mixing=_sc_env(
                "LORRAX_SC_MIXING", float, float(_g("sc_mixing")),
                "sc_mixing"),
            dump_dir=_sc_env(
                "LORRAX_SC_DUMP_DIR", str, str(_g("sc_dump_dir") or ""),
                "sc_dump_dir") or None,
            # No env override: the LORRAX_SC_* envs are deprecated and a
            # new knob must not add one.
            eigh=str(_g("sc_eigh")).strip().lower(),
        )
        memory = MemoryConfig(
            per_device_gb=memory_per_device_gb,
            chunk_target_utilization=chunk_utilization,
            band_chunk_size=int(_g("band_chunk_size")),
            r_chunk_override=int(_g("r_chunk_size")),
            zct_stage_cap_gb=zct_stage_cap_gb,
            gflat_chunk_size=int(_g("gflat_chunk_size")),
            vq_g_chunk_size=int(_g("vq_g_chunk_size")),
        )
        # SlabIO routing + auto-route GPU FFIs off on the CPU backend.
        # cuSOLVERMp / cuBLASMp are GPU-only.  The phdf5 FFI is NOT: both
        # its read and its write core compile CUDA-free into
        # liblorrax_ffi_host.so (``LORRAX_FFI_NO_CUDA``; see
        # ``ffi/cpp/phdf5/platform_seam.h``), so on CPU it is preferred
        # whenever the deployed lib exports the handler.
        #
        #   * ``slab_io=auto`` (the default) ALWAYS runs the capability-
        #     probed router for the active JAX backend — CPU:
        #     ``_route_cpu_slab_io`` (PHDF5_FFI → PHDF5_HOST →
        #     H5PY_ALLGATHER); GPU: ``_route_gpu_slab_io`` (PHDF5_FFI if
        #     the CUDA probe passes and single-node, else the safest
        #     working fallback).  Capability-probed, never
        #     env-presence-guessed, loud about which tier it took, and
        #     gated on NO other input key (an "auto" silently inert
        #     behind a second key is quality-pattern #8; see
        #     docs/dev/QUALITY_PATTERNS.md).
        #   * ``use_ffi_io`` is a deprecated boolean override:
        #     ``false`` forces H5PY_ALLGATHER (the pre-FFI writer),
        #     ``true`` is the routed default anyway (no-op), unset means
        #     "route".  Superseded by ``slab_io=<backend>``.
        #   * on CPU, explicit ``cusolvermp`` is
        #     REFUSED at parse time (CUDA-only backend; doctrine 3);
        #     ``distributed_lu = auto`` demotes to ``"off"`` (in-tree
        #     per-q ``jnp.linalg.solve``) with an announcement.
        #
        # User-facing: same ``cohsex.in`` works on both backends.
        _use_ffi_io_in = _g("use_ffi_io")   # None (unset) | True | False
        if _use_ffi_io_in is not None:
            _use_ffi_io_in = bool(_use_ffi_io_in)
        _slab_io_in = str(_g("slab_io")).strip().lower()
        if _slab_io_in not in ("auto", "phdf5_ffi", "phdf5_host", "h5py_allgather"):
            raise ValueError(
                f"slab_io={_slab_io_in!r} invalid; expected auto / phdf5_ffi "
                f"/ phdf5_host / h5py_allgather.")
        # Distributed-linalg axes.
        _dist_chol = str(_g("distributed_cholesky")).strip().lower()
        _dist_lu = str(_g("distributed_lu")).strip().lower()
        if _dist_chol not in ("auto", "off", "cusolvermp", "slate"):
            raise ValueError(
                f"distributed_cholesky={_dist_chol!r} invalid; expected "
                f"auto / off / cusolvermp / slate.")
        if _dist_lu not in ("auto", "off", "cusolvermp", "scalapack"):
            raise ValueError(
                f"distributed_lu={_dist_lu!r} invalid; expected auto / off "
                f"/ cusolvermp / scalapack (a SLATE getrf wrapper does not "
                f"exist yet; scalapack is the host/CPU-backend option).")
        # eigh_backend + use_low_mem_eigh are ONE axis; ``resolve_eigh_backend``
        # is the single place they combine (the raw-params drivers call the
        # same function).  It also owns the vocabulary check, read from
        # ffi.linalg.resolve so parser and dispatcher cannot drift.
        _use_low_mem_eigh = bool(_g("use_low_mem_eigh"))
        _eigh_backend = resolve_eigh_backend({
            "eigh_backend": _g("eigh_backend"),
            "use_low_mem_eigh": _use_low_mem_eigh,
        })
        # No CPU rewriting for eigh_backend: an explicit FFI request keeps
        # the fails-loudly semantics — ffi.linalg.resolve_backend rejects
        # cusolvermp on a CPU mesh (and a slate-less build) at resolve time.
        _charge_zeta_solve = str(_g("charge_zeta_solve")).strip().lower()
        if _charge_zeta_solve not in ("rank_truncate", "cholesky"):
            raise ValueError(
                f"charge_zeta_solve={_charge_zeta_solve!r} invalid; expected "
                f"rank_truncate / cholesky.")
        # Normalised to one of the TWO plans at PARSE time (fails loudly
        # here on removed spellings, not 20 minutes into the run).
        _w_dyson_solver = normalize_w_dyson_solver(_g("w_dyson_solver"))
        _dist_zeta_solve = str(_g("distributed_zeta_solve")).strip().lower()
        if _dist_zeta_solve not in (
                "auto", "replicated", "per_q", "distributed"):
            raise ValueError(
                f"distributed_zeta_solve={_dist_zeta_solve!r} invalid; "
                f"expected auto / replicated / per_q / distributed.")
        _transverse_zeta_solve = str(
            _g("transverse_zeta_solve")).strip().lower()
        if _transverse_zeta_solve not in ("ridge", "rank_truncate"):
            raise ValueError(
                f"transverse_zeta_solve={_transverse_zeta_solve!r} invalid; "
                f"expected ridge / rank_truncate.")
        if (_transverse_zeta_solve == "rank_truncate"
                and _dist_lu in ("scalapack", "cusolvermp", "on")):
            # Same refusal isdf/core._resolve_solver_kind_transverse makes
            # at resolve time, surfaced at PARSE time so a doomed bispinor
            # deck refuses in seconds, not after the charge fit.
            raise ValueError(
                f"transverse_zeta_solve=rank_truncate selects the eigh "
                f"pseudo-inverse family (distributed plan via "
                f"distributed_zeta_solve=distributed), but "
                f"distributed_lu={_dist_lu!r} explicitly requests an LU "
                f"backend that family does not run.  Leave distributed_lu "
                f"at auto/off, or keep transverse_zeta_solve=ridge.")
        _transverse_zeta_rcond = float(_g("transverse_zeta_rcond"))
        if not (0.0 < _transverse_zeta_rcond < 1.0):
            raise ValueError(
                f"transverse_zeta_rcond={_transverse_zeta_rcond!r} must be "
                f"a relative cutoff in (0, 1).")
        try:
            import jax as _jax
            _is_cpu_backend = _jax.default_backend() == "cpu"
        except Exception:
            _is_cpu_backend = False
        # --- SlabIO backend resolution.  Precedence:
        #   1. explicit ``slab_io=<backend>`` — honoured verbatim (a wrong
        #      choice fails loudly at SlabIO open, which beats silently
        #      running a different backend than the input file says).
        #      The capability probes are skipped: a deck that already
        #      named its writer should not pay a host-lib dlopen.
        #   2. ``use_ffi_io=false`` (deprecated) — forces H5PY_ALLGATHER.
        #   3. ``slab_io=auto`` (default) — the platform router, always.
        # 'use_ffi_io' deprecation warns exactly ONCE per deck, here —
        # each deck hits exactly one of the three branches below, and the
        # warning text names what actually happened to the key (audit
        # fix/zq 2026-07-28: read_lorrax_input used to emit a second,
        # differently-worded warning for mere presence of the key).
        if _slab_io_in != "auto":
            if _use_ffi_io_in is not None:
                import warnings
                warnings.warn(
                    f"Input key 'use_ffi_io' is deprecated and IGNORED "
                    f"here: slab_io={_slab_io_in} is explicit and takes "
                    f"precedence.  Remove use_ffi_io from the deck.",
                    DeprecationWarning, stacklevel=2)
                print_fn(
                    f"  [config] use_ffi_io={str(_use_ffi_io_in).lower()} "
                    f"is ignored: slab_io={_slab_io_in} is explicit and "
                    "takes precedence (use_ffi_io is deprecated).")
            _slab_io_choice = SlabIOBackend(_slab_io_in)
        elif _use_ffi_io_in is False:
            import warnings
            warnings.warn(
                "Input key 'use_ffi_io' is deprecated; 'use_ffi_io = "
                "false' is honored as 'slab_io = h5py_allgather' (the "
                "pre-FFI rank-0 writer).  Set 'slab_io = h5py_allgather' "
                "explicitly instead.",
                DeprecationWarning, stacklevel=2)
            print_fn(
                "  [config] use_ffi_io=false (deprecated): forcing SlabIO "
                "through H5PY_ALLGATHER (rank-0 serial write).  This "
                "overrides the slab_io=auto router; prefer "
                "slab_io = h5py_allgather.")
            _slab_io_choice = SlabIOBackend.H5PY_ALLGATHER
        else:
            if _use_ffi_io_in is True:
                import warnings
                warnings.warn(
                    "Input key 'use_ffi_io' is deprecated; 'use_ffi_io = "
                    "true' is a no-op — the slab_io=auto router already "
                    "picks the best available parallel writer.  Remove "
                    "the key.",
                    DeprecationWarning, stacklevel=2)
                print_fn(
                    "  [config] use_ffi_io=true is deprecated and "
                    "redundant: slab_io=auto already routes to the best "
                    "available parallel writer.  Remove the key.")
            _slab_io_choice = (_route_cpu_slab_io(print_fn)
                               if _is_cpu_backend
                               else _route_gpu_slab_io(print_fn))
        if _is_cpu_backend:
            # Doctrine 3 (audit fix/zq 2026-07-28): an EXPLICIT
            # ``cusolvermp`` on a CPU JAX backend REFUSES at parse time —
            # matching the scalapack-on-GPU refusal below and
            # eigh_backend's fails-loudly contract — instead of being
            # rewritten to 'off' (which silently ran a different solver
            # than the input file names).  Only 'auto' may demote, with an
            # announcement.
            #
            # slate / scalapack pass through: host-platform FFIs
            # (liblorrax_ffi_host.so) with explicit-request-fails-loudly
            # semantics at their own resolve/call sites.
            #
            # ``distributed_cholesky = auto`` ALSO passes through.  It
            # used to be forced to 'off' here, but 'off' is an *override*
            # that short-circuits the whole route policy in isdf/core.py
            # straight to ``sharded_cholesky`` -- and the replicated route
            # it thereby skips is the ONLY one that carries the charge
            # ζ-solve rank-truncation cure
            # (charge_zeta_solve='rank_truncate').  That route is
            # replicated dense JAX with no FFI, so it is perfectly valid
            # on CPU; only the ABOVE-cap cuSOLVERMp branch is CUDA-only,
            # and isdf/core.py declines that on a CPU mesh.  Forcing 'off'
            # here silently produced a non-rank-conditioned ζ on CPU
            # (garbage V_q -> inverted QP gap) with no warning.
            if _dist_chol == "cusolvermp":
                raise ValueError(
                    "distributed_cholesky = cusolvermp is CUDA-only but "
                    "the JAX backend is CPU; use distributed_cholesky = "
                    "auto|off|slate on CPU runs (auto keeps the replicated "
                    "rank-truncation route; slate is the host FFI).")
            if _dist_lu == "cusolvermp":
                raise ValueError(
                    "distributed_lu = cusolvermp is CUDA-only but the JAX "
                    "backend is CPU; use distributed_lu = "
                    "auto|off|scalapack on CPU runs (scalapack is the "
                    "ScaLAPACK host FFI).")
            if _dist_lu == "auto":
                # 'auto' demote, announced: auto never picks an FFI LU on
                # a CPU backend (cuSOLVERMp is CUDA-only and auto never
                # selects ScaLAPACK), so 'off' (in-tree per-q
                # jnp.linalg.solve) is the same route auto would resolve
                # to — made explicit here, and said out loud.
                print_fn(
                    "  [config] distributed_lu=auto on CPU backend: auto "
                    "never picks an FFI LU here (cuSOLVERMp is CUDA-only; "
                    "ScaLAPACK is explicit-only).  Demoting to 'off' "
                    "(in-tree per-q jnp.linalg.solve).  The ScaLAPACK "
                    "host FFI is available via explicit "
                    "distributed_lu = scalapack."
                )
                _dist_lu = "off"
        elif _dist_lu == "scalapack":
            # Host-only backend on a non-CPU JAX backend: reject at parse
            # time — the alternative is a ValueError hours later at the
            # first transverse solve, after the C_q build.
            raise ValueError(
                "distributed_lu=scalapack is host-only (Cray LibSci) but "
                "the JAX backend is not CPU; use distributed_lu = "
                "auto|off|cusolvermp on GPU runs.")
        backend = BackendConfig(
            slab_io=_slab_io_choice,
            gspace_io=GspaceIO(str(_g("gspace_mode")).strip().lower()),
            w_dyson_solver=_w_dyson_solver,
            distributed_cholesky=_dist_chol,
            distributed_lu=_dist_lu,
            eigh_backend=_eigh_backend,
            use_low_mem_eigh=_use_low_mem_eigh,
            zeta_ridge=float(_g("zeta_ridge")),
            charge_zeta_solve=_charge_zeta_solve,
            distributed_zeta_solve=_dist_zeta_solve,
            zeta_rcond=float(_g("zeta_rcond")),
            transverse_zeta_solve=_transverse_zeta_solve,
            transverse_zeta_rcond=_transverse_zeta_rcond,
            gamma_contract_mode=str(_g("gamma_contract_mode")).strip().lower(),
        )
        # Validate the V_H source at PARSE time, not at the read that
        # would otherwise fail 20 minutes into a 40-node run.
        from file_io.kin_ion import HARTREE_SOURCES
        _hartree_source = str(_g("hartree_source") or "auto").strip().lower()
        if _hartree_source not in HARTREE_SOURCES:
            raise ValueError(
                f"hartree_source={_hartree_source!r} is not one of "
                f"{HARTREE_SOURCES}.  H0 = kin_ion + V_H is a ~500 eV "
                "cancellation; this key is not guessed.")

        debug = DebugConfig(
            sigma_freq_debug_output=bool(_g("sigma_freq_debug_output")),
            sigma_freq_debug_file=str(_g("sigma_freq_debug_file")),
            write_wfn_h5=bool(_g("write_wfn_h5")),
        )
        bse = BSEConfig(
            get_centroids_fi=bool(_g("get_centroids_fi")),
            wfn_fi_min=int(_g("wfn_fi_min")),
            wfn_fi_max=int(_g("wfn_fi_max")),
            kgrid_fi=str(_g("kgrid_fi") or ""),
            wfn_fi_q_chunk=int(_g("wfn_fi_q_chunk")),
        )
        if bse.wfn_fi_q_chunk < 0:
            raise ValueError(
                f"wfn_fi_q_chunk={bse.wfn_fi_q_chunk} invalid; expected >= 1, "
                f"or 0 for the default (= N_q_co, the coarse k-point count).")

        return cls(
            # Top-level: system + mode flags
            nval=int(_g("nval")),
            ncond=int(_g("ncond")),
            nband=int(_g("nband")),
            sys_dim=int(_g("sys_dim")),
            density_self_consistent=bool(_g("density_self_consistent")),
            sc_on_ibz=bool(_g("sc_on_ibz")),
            hartree_source=_hartree_source,
            restart=bool(_g("restart")),
            compute_mode_raw=str(_g("compute_mode") or "auto").strip().lower(),
            qp_solver_raw=str(_g("qp_solver") or "auto").strip().lower(),
            do_screened=bool(_g("do_screened")),
            bispinor=bool(_g("bispinor")),
            do_G0=bool(_g("do_G0")),
            self_consistent=bool(_g("self_consistent")),
            use_ppm_sigma=bool(_g("use_ppm_sigma")),
            no_degen_averaging=bool(_g("no_degen_averaging")),
            degen_avg_tol_ry=float(_g("degen_avg_tol_ry")),
            # Sub-dataclass groups
            paths=paths,
            head=head,
            screening=screening,
            ppm=ppm,
            sc=sc,
            memory=memory,
            backend=backend,
            debug=debug,
            bse=bse,
            # Parsed blocks
            kpoints_crystal_b=params.get("kpoints_crystal_b"),
            input_dir=input_dir,
        )
