"""Small driver-side helpers to keep ``gw_jax.py`` focused on orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Optional

import h5py
import numpy as np

from common.units import RYD_TO_EV
from .gw_config import LorraxConfig, SlabIOBackend


@dataclass(frozen=True)
class PPMSigmaRuntimeOptions:
    """Resolved PPM sigma options parsed once in the GW driver."""

    omega_p_ry: float
    ppm_fallback: float
    omega_grid_ev: np.ndarray
    omega_grid_ry: np.ndarray
    sigma_regularization_ry: float
    sigma_edge_factor: float
    sigma_omega_batch_size: int
    sigma_omega_accumulation: str
    ppm_invalid_mode: str
    sigma_freq_debug_output: bool
    fermi_reference: str
    sigma_at_dft_extrapolate: bool
    sigma_at_dft_energies: bool
    sigma_kij_h5_path: str
    sigma_freq_debug_file: str


def _resolve_input_path(input_dir: str, path: str) -> str:
    if path and (not os.path.isabs(path)):
        return os.path.join(input_dir, path)
    return path


class profile_section:
    """Context manager wrapping the optional ``pf`` profiling hooks.

    The sandbox keeps a small ``pf`` helper at
    ``scripts/profiling/pf.py`` for memory-snapshot + xprof-region
    annotation.  When importable, this CM:

    1. Adds ``scripts/profiling`` to ``sys.path`` and imports ``pf``.
    2. Takes a memory snapshot at ``{artifacts_dir}/memprof/{name}_pre.prof``.
    3. Enters ``pf.region(name)``.
    4. On exit takes ``{name}_post.prof``.

    Failures in any of (1)–(4) degrade gracefully — the caller's body
    still runs; the only visible effect is one ``print_fn`` line.
    Useful so call sites read as one ``with`` statement instead of
    six lines of ``try/except`` noise.

    Example::

        with profile_section("sigma_ppm", artifacts_dir, print_fn=print0):
            sigma_omega = compute_sigma_c_ppm_omega_grid(...)
    """

    __slots__ = ("_name", "_artifacts_dir", "_print_fn", "_pf", "_region_cm")

    def __init__(self, name: str, artifacts_dir: str | None = None,
                 *, print_fn=print):
        self._name = name
        self._artifacts_dir = (
            artifacts_dir if artifacts_dir is not None
            else os.environ.get("PF_ARTIFACTS_DIR", "profile")
        )
        self._print_fn = print_fn
        self._pf = None
        self._region_cm = None

    def _try_import_pf(self):
        try:
            import sys
            sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
            import pf  # type: ignore
            return pf
        except Exception as exc:
            self._print_fn(f"  [pf] profiling hooks unavailable: {exc}")
            return None

    def _snapshot(self, suffix: str):
        if self._pf is None:
            return
        try:
            path = f"{self._artifacts_dir}/memprof/{self._name}_{suffix}.prof"
            self._pf.snapshot_memory(path, label=f"{self._name}_{suffix}")
        except Exception:
            pass

    def __enter__(self):
        self._pf = self._try_import_pf()
        self._snapshot("pre")
        if self._pf is not None:
            try:
                self._region_cm = self._pf.region(self._name)
                self._region_cm.__enter__()
            except Exception:
                self._region_cm = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._region_cm is not None:
            try:
                self._region_cm.__exit__(exc_type, exc, tb)
            except Exception:
                pass
            self._region_cm = None
        self._snapshot("post")
        return False


def setup_runtime(config: LorraxConfig, mesh_xy, *, print_fn=print) -> None:
    """Pre-init NCCL + phdf5 MPI + JAX persistent compile cache.

    All three are best-effort startup optimizations whose absence is not
    fatal:

    - **NCCL warmup**: pre-allocates communicators for full-mesh and
      per-axis psums so the first real collective (sigma's all-reduce-
      start) doesn't eat a timed section.  No-op in single-process.
    - **phdf5 ``MPI_Init_thread``**: when the slab-IO backend is the
      phdf5 FFI, eagerly enter ``MPI_THREAD_MULTIPLE`` so the first
      collective ``H5Fcreate`` (in ``zeta_fit_chunked``) doesn't pay
      the ~400 ms MPI_Init cost on the critical path; failures are
      logged and swallowed.
    - **JAX persistent compile cache**: enable XDG-style on-disk cache
      so warm-cache cold starts skip ~3 s of XLA compile.  Opt out via
      ``ISDF_JAX_CACHE_DIR=""``.

    All three were inlined in ``main()``; this helper unifies them.
    """
    from runtime import nccl_warmup
    from common import timing as _timing

    with _timing.section("nccl_warmup"):
        nccl_warmup(mesh_xy)

    if config.backend.slab_io is SlabIOBackend.PHDF5_FFI:
        try:
            from ffi.common.ffi_loader import phdf5_init_mpi
            phdf5_init_mpi()
        except Exception as exc:
            print_fn(f"  [phdf5 init_mpi] skipped: {exc}")
    elif config.backend.slab_io is SlabIOBackend.PHDF5_HOST:
        # mpi4py initialises MPI on first import; do it here so the
        # ~400 ms MPI_Init cost is amortised before the first SlabIO
        # open (same rationale as the FFI's phdf5_init_mpi).
        try:
            from mpi4py import MPI  # noqa: F401
        except Exception as exc:
            print_fn(f"  [phdf5_host mpi4py init] skipped: {exc}")

    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as exc:
        print_fn(f"  [jax compile cache] skipped: {exc}")


def build_bgw_v_grid_fn(
    config: LorraxConfig, *,
    wfn,
    sym,
    input_dir: str,
    print_fn=print,
) -> Optional[Callable[[tuple], np.ndarray]]:
    """Build the optional BGW vcoul override closure (or None).

    When ``config.head.use_bgw_vcoul`` is set the GW driver substitutes
    BGW's MC-averaged ``v(q+G)`` for LORRAX's internal head-only
    mini-BZ average — this enables bit-reproducible BGW comparisons.
    The returned callable maps a fractional q-tuple to a dense G-grid
    of Coulomb values; pass it through to ``compute_V_q``.

    Returns ``None`` when the override is disabled, so callers can do
    ``bgw_v_grid_fn = build_bgw_v_grid_fn(...)`` unconditionally.
    """
    head = config.head
    if not head.use_bgw_vcoul:
        return None
    if head.bgw_vcoul_file is None:
        raise ValueError(
            "head.use_bgw_vcoul=true requires head.bgw_vcoul_file to be set"
        )

    from file_io import read_bgw_vcoul, fill_v_grid_for_q

    bgw_path = _resolve_input_path(input_dir, head.bgw_vcoul_file)
    print_fn(f"  BGW vcoul override: loading {bgw_path}")
    bgw_table = read_bgw_vcoul(bgw_path)
    print_fn(
        f"    {bgw_table.q_fracs.shape[0]} unique q-points, "
        f"G counts per q: {[len(g) for g in bgw_table.G_miller_per_q]}"
    )

    cell_volume = float(wfn.cell_volume)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    # BGW's vcoul file only stores unique IBZ q's; mapping LORRAX's
    # full-BZ q to those needs the full crystal sym group.  A nosym WFN
    # stores only identity in mf_header/symmetry/mtrx, so allow pulling
    # the 48 ops from an aux sym-reduced WFN when provided.
    if head.bgw_vcoul_sym_wfn:
        aux_path = _resolve_input_path(input_dir, head.bgw_vcoul_sym_wfn)
        with h5py.File(aux_path, "r") as fsym:
            sym_real = np.asarray(
                fsym["mf_header/symmetry/mtrx"][:], dtype=np.int32)
        print_fn(f"    crystal sym ops loaded from {aux_path}: {sym_real.shape[0]}")
        sym_mats_k = sym_real.transpose(0, 2, 1).copy()
    else:
        sym_mats_k = np.asarray(sym.sym_mats_k, dtype=np.int32)

    def bgw_v_grid_fn(q_frac_tuple):
        return fill_v_grid_for_q(
            bgw_table, q_frac_tuple, fft_grid, cell_volume,
            sym_mats_k=sym_mats_k,
        )

    return bgw_v_grid_fn


def build_ppm_sigma_runtime_options(
    config: LorraxConfig, *, input_dir: str
) -> PPMSigmaRuntimeOptions:
    """Build PPM sigma runtime options from a ``LorraxConfig``."""
    ppm = config.ppm
    debug = config.debug

    if ppm.omega_step_ev <= 0.0:
        raise ValueError("ppm.omega_step_ev must be > 0.")
    if ppm.omega_max_ev < ppm.omega_min_ev:
        raise ValueError("ppm.omega_max_ev must be >= ppm.omega_min_ev.")
    if ppm.fermi_reference not in ("vbm", "midgap"):
        raise ValueError("ppm.fermi_reference must be 'vbm' or 'midgap'.")

    n_omega = int(np.floor(
        (ppm.omega_max_ev - ppm.omega_min_ev) / ppm.omega_step_ev + 0.5
    )) + 1
    omega_grid_ev = (
        ppm.omega_min_ev + ppm.omega_step_ev * np.arange(n_omega, dtype=np.float64)
    )
    omega_grid_ry = omega_grid_ev / RYD_TO_EV
    sigma_regularization_ry = ppm.regularization_ev / RYD_TO_EV

    return PPMSigmaRuntimeOptions(
        omega_p_ry=float(ppm.omega_p),
        ppm_fallback=float(ppm.fallback_omega),
        omega_grid_ev=omega_grid_ev,
        omega_grid_ry=omega_grid_ry,
        sigma_regularization_ry=sigma_regularization_ry,
        sigma_edge_factor=float(ppm.window_edge_factor),
        sigma_omega_batch_size=int(max(1, ppm.omega_batch_size)),
        sigma_omega_accumulation=str(ppm.omega_accumulation).strip().lower(),
        ppm_invalid_mode=str(ppm.invalid_mode).strip().lower(),
        sigma_freq_debug_output=bool(debug.sigma_freq_debug_output),
        fermi_reference=str(ppm.fermi_reference).strip().lower(),
        sigma_at_dft_extrapolate=bool(ppm.sigma_at_dft_extrapolate),
        sigma_at_dft_energies=bool(ppm.sigma_at_dft_energies),
        sigma_kij_h5_path=_resolve_input_path(input_dir, str(config.paths.sigma_kij_h5_file or "").strip()),
        sigma_freq_debug_file=_resolve_input_path(input_dir, str(debug.sigma_freq_debug_file or "").strip()),
    )
