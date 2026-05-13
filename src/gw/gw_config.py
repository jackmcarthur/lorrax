"""Unified configuration for LORRAX GW calculations.

``LorraxConfig`` is built once via :meth:`LorraxConfig.from_input_file`
from the ``[cohsex]`` section of ``cohsex.in`` and threaded through the
entire driver.  Its ~80 input keys are grouped into sub-dataclasses
along the same axes the input file's section comments already use:

    config.head        — q→0 Coulomb-head sources & overrides
    config.minimax     — screening-minimax target error / max nodes / table mode
    config.ppm         — PPM model + sigma quadrature + on-shell σ_c options
    config.sigma_grid  — ω-grid for Σ_c(ω) output
    config.memory      — chunk sizing + AOT chunk-chooser
    config.backend     — FFI/IO backend selection (slab_io / gspace_io / screening_solver)
    config.debug       — debug-only flags & file paths
    config.bse         — BSE interpolation setup (htransform-driven)
    config.paths       — output filenames

The top-level ``LorraxConfig`` retains only system geometry
(``nval`` / ``ncond`` / ``nband`` / ``sys_dim``) and the orthogonal
mode flags (``compute_mode`` / ``self_consistent`` / etc.) that the
driver reads on the fast path.

Derived sub-objects (the math-internal ``MinimaxConfig`` and
``SigmaQuadratureConfig`` from ``minimax_config.py``, plus
``PPMSigmaRuntimeOptions`` from ``gw_driver_helpers.py``) are still
constructed on demand via ``LorraxConfig`` properties.
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
#  Enums
# ---------------------------------------------------------------------------

class ComputeMode(str, enum.Enum):
    """The single axis describing what self-energy is computed.

    Orthogonal to ``self_consistent``: any mode can in principle be wrapped
    in a fixed-point loop (currently only ``COHSEX`` is wired).

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


class SlabIOBackend(str, enum.Enum):
    """How big sigma/zeta/restart HDF5 files are written.

    - ``PHDF5_FFI`` — every rank writes its hyperslab via the parallel-HDF5
      FFI (collective MPI-IO).  Default.  ~5× faster than the rank-0 path
      once Lustre striping is applied.
    - ``H5PY_ALLGATHER`` — gather to rank 0 and write via h5py.  Fallback
      for systems without the FFI ``.so`` built or for non-Lustre
      filesystems.
    """
    PHDF5_FFI = "phdf5_ffi"
    H5PY_ALLGATHER = "h5py_allgather"


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


class ScreeningSolver(str, enum.Enum):
    """Which solver runs the W = (1 - V·χ₀)⁻¹·V Dyson equation.

    - ``JAX_NATIVE`` — q-parallel reshard + ``jax.scipy.linalg.lu_factor``
      / ``lu_solve`` per q.  Default; uses one all-gather + all-scatter.
    - ``CUBLASMP_FFI`` — fused symmetric Cholesky W = X·H⁻¹·X† via
      cuBLASMp + cuSOLVERMp FFI.  No JAX-level intermediates between
      the matmuls; needed when ``nq · n_rmu²`` exceeds VRAM.
    """
    JAX_NATIVE = "jax_native"
    CUBLASMP_FFI = "cublasmp_ffi"


# Legacy ``isdf_memory_mode`` strings → ``ScreeningSolver`` enum.  Used
# by both the input-file parser and the back-compat property aliases.
_LEGACY_ISDF_MEMORY_MODE = {
    "auto":     ScreeningSolver.JAX_NATIVE,    # back-compat default
    "high_mem": ScreeningSolver.JAX_NATIVE,
    "low_mem":  ScreeningSolver.CUBLASMP_FFI,
}
_SCREENING_SOLVER_TO_LEGACY = {
    ScreeningSolver.JAX_NATIVE:   "high_mem",
    ScreeningSolver.CUBLASMP_FFI: "low_mem",
}


# ---------------------------------------------------------------------------
#  Defaults — single source of truth for every input key
# ---------------------------------------------------------------------------

_DEFAULTS = {
    # System geometry
    "nval": 5,
    "ncond": 5,
    "nband": 100,
    "sys_dim": 2,
    # File paths
    "wfn_file": "WFN.h5",
    "centroids_file": "centroids_frac.txt",
    # Optional second centroid file used by the bispinor pipeline:
    # μ_L=1,2,3 (transverse) ζ-fits use Gordon-current-density centroids
    # rather than the charge-density centroids in ``centroids_file``.
    # Empty string == "not set" (cfg.centroids_file_current is None then).
    "centroids_file_current": "",
    "kin_ion_file": "kin_ion.h5",
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
    "sigma_kij_h5_file": "",
    # Core flags
    "restart": True,
    # ``compute_mode`` is the single axis describing the self-energy ansatz.
    # ``"auto"`` infers from the legacy ``do_screened`` / ``use_ppm_sigma`` /
    # ``ppm_model`` flags so existing input files keep working unchanged.
    # New input files should set ``compute_mode`` explicitly:
    #   "x_only" | "cohsex" | "gn_ppm" | "hl_ppm".
    # ``self_consistent`` remains an orthogonal flag (currently only
    # implemented on top of COHSEX).
    "compute_mode": "auto",
    "do_screened": True,
    "bispinor": False,
    "do_G0": True,
    "self_consistent": False,
    "use_ppm_sigma": False,
    # BGW-style averaging of diagonal Σ within degenerate sets (mirrors
    # ``Sigma/shiftenergy.f90`` band-averaging).  ``no_degen_averaging =
    # true`` disables it and emits the raw QE-basis-dependent diagonals.
    # ``degen_avg_tol_ry`` matches BGW's ``TOL_Degeneracy = 1e-6 Ry``.
    "no_degen_averaging": False,
    "degen_avg_tol_ry": 1.0e-6,
    # I/O backend: True routes big sigma/zeta writes through the
    # parallel-HDF5 FFI (collective MPI-IO, ~5× faster than the
    # rank-0 h5py path once Lustre striping is applied — see
    # ``_slab_io_ffi._lustre_prestripe``).  False keeps the historical
    # ``process_allgather`` + rank-0 ``h5py`` path as a fallback for
    # non-Lustre filesystems or systems without the FFI ``.so`` built.
    "use_ffi_io": True,
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
    # AOT-fit chunk chooser: replaces the per-stage byte heuristic in
    # compute_optimal_chunks with the driver-level
    # ``aot_memory_model.choose_chunks_aot`` — minimises total FLOPs
    # subject to the predicted peak fitting under the memory budget.
    # Requires fit artifacts at
    # src/gw/aot_memory_model/artifacts/fit_one_rchunk__current__*.json.
    "use_aot_chunk_chooser": False,
    # When ``use_aot_chunk_chooser`` is True, ``chunk_chooser_mode``
    # picks between the AOT chooser variants:
    #   "heuristic" – default; per-stage byte heuristic over the AOT
    #                 fit artifacts.
    #   "analytic"  – regressed-fit analytic chooser (alternate model
    #                 over the same artifacts).
    "chunk_chooser_mode": "heuristic",
    # Inner k-axis chunker in ``psi_G_store.fetch_psi_rchunk``.  Bounds
    # the per-fetch FFT box ``(k_chunk, bc, ns, n_rtot)`` to avoid XLA
    # materialising it unsharded at CrI3 6×6 80 Ry scale.
    # 0 (default) = no inner k-chunking (one shot over all k).
    "psig_k_chunk_size": 0,
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
    "cusolvermp_charge": "auto",   # auto | on | off  → cusolvermp_cholesky | sharded_cholesky
    "cusolvermp_lu":     "auto",   # auto | on | off  → cusolvermp_lu | lu
    # γ̃-double-contract kernel variant inside the monolithic pair
    # pipeline (see ``common.gamma_matrices.gamma_double_contract``).
    # Math identical across all three; differ in HLO structure.
    #   "take"   – jnp.take + element-wise phase mul (default).
    #   "einsum" – materialise the sparse γ̃ and contract via einsum.
    #   "scan"   – lax.scan over the (a, b) spin axis pairs.
    "gamma_contract_mode": "take",
    # Memory / chunking
    "memory_per_device_gb": 0.0,  # 0 = auto-detect
    "chunk_size": -1,
    "band_chunk_size": 16,
    "r_chunk_size": 0,
    # ISDF
    "isdf_memory_mode": "auto",   # auto | high_mem | low_mem
                                   # high_mem (default): 2D-blocked JAX Cholesky +
                                   #   replicate-L vmap trsm.  Fast for small n_rmu.
                                   # low_mem: batched cuSOLVERMp potrf + potrs on
                                   #   per-X-row sub-comm (L stays distributed).
                                   #   Needed when nq * n_rmu^2 exceeds VRAM.
                                   # auto → high_mem for back-compat.
    "mc_average_vcoul_body": True,
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
    # PPM sigma options
    "ppm_sigma_scale": 1.0,
    "ppm_sigma_flip_neg": False,
    "ppm_invalid_mode": "static_limit",
    "fermi_reference": "midgap",
    "sigma_at_dft_extrapolate": False,
    "sigma_at_dft_energies": False,
    # Debug
    "debug_hartree": False,
    "debug_omega": None,
    "sigma_freq_debug_output": False,
    "ppm_sigma_debug_static_norm": False,
    "ppm_static_cohsex_check": False,
    "sigma_debug_quadrature": False,
    "sigma_debug_quadrature_samples": 200,
    "write_w_copies_debug": False,
    "w_copies_debug_file": "",
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
}

# Keys whose string values should be lowercased and stripped
_NORMALIZE_STR = {
    "compute_mode",
    "wcoul0_source", "screening_method", "minimax_energy_reference",
    "sigma_omega_accumulation", "fermi_reference",
    "isdf_memory_mode",
    "ppm_invalid_mode",
    "ppm_model",
}


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

        parser = configparser.ConfigParser()
        parser.read_string(''.join(section_lines))
        section = parser["cohsex"] if "cohsex" in parser else parser[parser.sections()[0]]

        # Legacy key check
        if section.get("use_shipped_minimax_tables", fallback=None) is not None:
            raise ValueError(
                "Input key 'use_shipped_minimax_tables' is no longer supported. "
                "Use 'regenerate_minimax_tables = true/false' instead.")
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

        # Build params from _DEFAULTS, overriding with parsed values
        params = {}
        for key, default in _DEFAULTS.items():
            raw = section.get(key, fallback=None)
            if raw is None:
                params[key] = default
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
    sigma_kij_h5_file: str


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
    omega_accumulation: str       # "auto" | "kij" | "kij_stream"

    # --- on-shell evaluation knobs ---
    sigma_scale: float
    sigma_flip_neg: bool
    invalid_mode: str             # "static_limit"
    fermi_reference: str          # "midgap" | "vbm"
    sigma_at_dft_extrapolate: bool
    sigma_at_dft_energies: bool


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
    chunk_size: int               # legacy band-chunk knob; -1 = no chunking
    band_chunk_size: int
    r_chunk_override: int         # 0 = auto
    zct_stage_cap_gb: float | None
    use_aot_chunk_chooser: bool
    chunk_chooser_mode: str       # "heuristic" | "analytic"
    psig_k_chunk_size: int        # 0 = no inner k-chunking
    gflat_chunk_size: int         # 0 = one-shot (or planner-picked)
    vq_g_chunk_size: int          # 0 = auto _pick_g_chunk(ngkmax)


@dataclass(frozen=True)
class BackendConfig:
    """Three-axis backend selection: I/O + ψ(G) lifecycle + screening solver.

    All three knobs were previously orthogonal-sounding boolean/string
    flags in different namespaces (``use_ffi_io`` / ``gspace_mode`` /
    ``isdf_memory_mode``) that secretly toggled FFI paths.  Grouped here
    so :meth:`summary` can print one line at startup describing what's
    actually active per channel.
    """
    slab_io: SlabIOBackend
    gspace_io: GspaceIO
    screening_solver: ScreeningSolver
    cusolvermp_charge: str    # "auto" | "on" | "off"
    cusolvermp_lu: str        # "auto" | "on" | "off"
    gamma_contract_mode: str  # "take" | "einsum" | "scan"

    def summary(self) -> str:
        """One-line "what's active" for the run banner."""
        return (
            f"backend: slab_io={self.slab_io.value}, "
            f"gspace_io={self.gspace_io.value}, "
            f"screening_solver={self.screening_solver.value}, "
            f"cusolvermp_charge={self.cusolvermp_charge}, "
            f"cusolvermp_lu={self.cusolvermp_lu}, "
            f"gamma_contract={self.gamma_contract_mode}"
        )


@dataclass(frozen=True)
class DebugConfig:
    """Debug-only flags + auxiliary output filenames."""
    debug_hartree: bool
    debug_omega: float | None
    sigma_freq_debug_output: bool
    ppm_sigma_debug_static_norm: bool
    ppm_static_cohsex_check: bool
    sigma_debug_quadrature: bool
    sigma_debug_quadrature_samples: int
    write_w_copies_debug: bool
    w_copies_debug_file: str
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

    # --- Core mode flags (top-level; hot path) ---
    restart: bool
    compute_mode_raw: str         # "auto" | one of ComputeMode.value strings
    do_screened: bool
    bispinor: bool
    do_G0: bool
    self_consistent: bool
    use_ppm_sigma: bool           # legacy mirror; ``compute_mode`` is canonical
    no_degen_averaging: bool
    degen_avg_tol_ry: float

    # --- Sub-dataclass groups (everything else) ---
    paths: FilePaths
    head: HeadConfig
    screening: ScreeningConfig
    ppm: PPMConfig
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
            return ComputeMode(raw)
        except ValueError as exc:
            raise ValueError(
                f"compute_mode={raw!r} invalid; expected one of: "
                f"{', '.join(m.value for m in ComputeMode)}, or 'auto'."
            ) from exc

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
        """Math-internal :class:`gw.minimax_config.SigmaQuadratureConfig`."""
        from .minimax_config import SigmaQuadratureConfig
        return SigmaQuadratureConfig(
            target_error=self.ppm.sigma_target_error,
            max_nodes=self.ppm.sigma_max_nodes,
            crossing_max_nodes=max(500, self.ppm.sigma_max_nodes),
            crossing_eps_q=1.0e-3,
            regenerate_tables=self.screening.regenerate_minimax_tables,
        )

    @property
    def omega_grid_ry(self):
        """Σ_c(ω) frequency grid in Rydberg."""
        return np.arange(
            self.ppm.omega_min_ev / RYD_TO_EV,
            (self.ppm.omega_max_ev + 0.5 * self.ppm.omega_step_ev) / RYD_TO_EV,
            self.ppm.omega_step_ev / RYD_TO_EV,
        )

    @property
    def omega_grid_ev(self):
        """Σ_c(ω) frequency grid in eV."""
        return np.arange(
            self.ppm.omega_min_ev,
            self.ppm.omega_max_ev + 0.5 * self.ppm.omega_step_ev,
            self.ppm.omega_step_ev,
        )

    # ------------------------------------------------------------------
    #  Back-compat aliases — the FFI/IO group changed semantics (bool /
    #  string → enum), so callers that still want the old names get
    #  coerced views.  New code should use ``config.backend.<field>`` /
    #  ``config.memory.<field>`` etc. directly.
    # ------------------------------------------------------------------

    @property
    def use_ffi_io(self) -> bool:
        """Legacy ``use_ffi_io: bool`` → ``backend.slab_io is PHDF5_FFI``."""
        return self.backend.slab_io is SlabIOBackend.PHDF5_FFI

    @property
    def gspace_mode(self) -> str:
        """Legacy ``gspace_mode: str`` view of ``backend.gspace_io``."""
        return self.backend.gspace_io.value

    @property
    def isdf_memory_mode(self) -> str:
        """Legacy ``isdf_memory_mode`` view of ``backend.screening_solver``."""
        return _SCREENING_SOLVER_TO_LEGACY[self.backend.screening_solver]

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

        # --- Validate isdf_memory_mode (legacy string → ScreeningSolver) ---
        isdf_memory_mode = str(
            params.get("isdf_memory_mode", "auto")).strip().lower()
        if isdf_memory_mode not in _LEGACY_ISDF_MEMORY_MODE:
            raise ValueError(
                f"isdf_memory_mode={isdf_memory_mode!r} invalid; "
                f"expected one of {sorted(_LEGACY_ISDF_MEMORY_MODE)}"
            )

        # --- Memory auto-detection ---
        memory_per_device_gb = float(params.get("memory_per_device_gb", 0.0))
        if memory_per_device_gb <= 0:
            from common.gpu_utils import get_device_memory_gb
            memory_per_device_gb = get_device_memory_gb()
            print_fn(
                f"  Auto-detected memory budget: {memory_per_device_gb:.2f} GB/device"
            )

        # --- Chunk utilization from env ---
        try:
            chunk_utilization = float(
                os.environ.get("ISDF_CHUNK_TARGET_UTILIZATION", "0.97"))
        except Exception:
            chunk_utilization = 0.97
        chunk_utilization = max(0.85, min(1.0, chunk_utilization))

        # --- ZCT stage cap from env ---
        import jax
        zct_stage_cap_gb = None
        zct_cap_env = os.environ.get("ISDF_ZCT_STAGE_CAP_GB")
        zct_frac_env = os.environ.get("ISDF_ZCT_STAGE_CAP_FRAC")
        if zct_cap_env:
            try:
                zct_stage_cap_gb = min(
                    memory_per_device_gb, max(0.0, float(zct_cap_env)))
            except Exception:
                pass
        if (zct_stage_cap_gb is None and zct_frac_env
                and jax.default_backend() in ("gpu", "cuda")):
            from common.gpu_utils import get_device_memory_info
            total_gb = float(get_device_memory_info().get("total_gb", 0.0))
            if total_gb > 0:
                try:
                    frac = max(0.10, min(0.95, float(zct_frac_env)))
                    zct_stage_cap_gb = min(memory_per_device_gb, frac * total_gb)
                except Exception:
                    pass

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
            sigma_kij_h5_file=str(_g("sigma_kij_h5_file") or ""),
        )
        head = HeadConfig(
            wcoul0_source=str(_g("wcoul0_source")).strip().lower(),
            wcoul0_eta=float(_g("wcoul0_eta") or 0.0),
            vhead=_g("vhead"),
            whead_0freq=_g("whead_0freq"),
            whead_imfreq=_g("whead_imfreq"),
            mc_average_vcoul_body=bool(_g("mc_average_vcoul_body")),
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
            sigma_target_error=float(_g("ppm_sigma_target_error")),
            sigma_max_nodes=int(_g("ppm_sigma_max_nodes")),
            omega_min_ev=float(_g("sigma_omega_min_ev")),
            omega_max_ev=float(_g("sigma_omega_max_ev")),
            omega_step_ev=float(_g("sigma_omega_step_ev")),
            regularization_ev=float(_g("sigma_regularization_ev")),
            window_edge_factor=float(_g("sigma_window_edge_factor")),
            omega_batch_size=int(_g("sigma_omega_batch_size")),
            omega_accumulation=str(_g("sigma_omega_accumulation")).strip().lower(),
            sigma_scale=float(_g("ppm_sigma_scale")),
            sigma_flip_neg=bool(_g("ppm_sigma_flip_neg")),
            invalid_mode=str(_g("ppm_invalid_mode") or "static_limit"),
            fermi_reference=str(_g("fermi_reference")).strip().lower(),
            sigma_at_dft_extrapolate=bool(_g("sigma_at_dft_extrapolate")),
            sigma_at_dft_energies=bool(_g("sigma_at_dft_energies")),
        )
        memory = MemoryConfig(
            per_device_gb=memory_per_device_gb,
            chunk_target_utilization=chunk_utilization,
            chunk_size=int(_g("chunk_size")),
            band_chunk_size=int(_g("band_chunk_size")),
            r_chunk_override=int(_g("r_chunk_size")),
            zct_stage_cap_gb=zct_stage_cap_gb,
            use_aot_chunk_chooser=bool(_g("use_aot_chunk_chooser")),
            chunk_chooser_mode=str(_g("chunk_chooser_mode")).strip().lower(),
            psig_k_chunk_size=int(_g("psig_k_chunk_size")),
            gflat_chunk_size=int(_g("gflat_chunk_size")),
            vq_g_chunk_size=int(_g("vq_g_chunk_size")),
        )
        backend = BackendConfig(
            slab_io=(
                SlabIOBackend.PHDF5_FFI if bool(_g("use_ffi_io"))
                else SlabIOBackend.H5PY_ALLGATHER
            ),
            gspace_io=GspaceIO(str(_g("gspace_mode")).strip().lower()),
            screening_solver=_LEGACY_ISDF_MEMORY_MODE[isdf_memory_mode],
            cusolvermp_charge=str(_g("cusolvermp_charge")).strip().lower(),
            cusolvermp_lu=str(_g("cusolvermp_lu")).strip().lower(),
            gamma_contract_mode=str(_g("gamma_contract_mode")).strip().lower(),
        )
        debug = DebugConfig(
            debug_hartree=bool(_g("debug_hartree")),
            debug_omega=_g("debug_omega"),
            sigma_freq_debug_output=bool(_g("sigma_freq_debug_output")),
            ppm_sigma_debug_static_norm=bool(_g("ppm_sigma_debug_static_norm")),
            ppm_static_cohsex_check=bool(_g("ppm_static_cohsex_check")),
            sigma_debug_quadrature=bool(_g("sigma_debug_quadrature")),
            sigma_debug_quadrature_samples=int(_g("sigma_debug_quadrature_samples")),
            write_w_copies_debug=bool(_g("write_w_copies_debug")),
            w_copies_debug_file=str(_g("w_copies_debug_file") or ""),
            sigma_freq_debug_file=str(_g("sigma_freq_debug_file")),
            write_wfn_h5=bool(_g("write_wfn_h5")),
        )
        bse = BSEConfig(
            get_centroids_fi=bool(_g("get_centroids_fi")),
            wfn_fi_min=int(_g("wfn_fi_min")),
            wfn_fi_max=int(_g("wfn_fi_max")),
            kgrid_fi=str(_g("kgrid_fi") or ""),
        )

        return cls(
            # Top-level: system + mode flags
            nval=int(_g("nval")),
            ncond=int(_g("ncond")),
            nband=int(_g("nband")),
            sys_dim=int(_g("sys_dim")),
            restart=bool(_g("restart")),
            compute_mode_raw=str(_g("compute_mode") or "auto").strip().lower(),
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
            memory=memory,
            backend=backend,
            debug=debug,
            bse=bse,
            # Parsed blocks
            kpoints_crystal_b=params.get("kpoints_crystal_b"),
            input_dir=input_dir,
        )
