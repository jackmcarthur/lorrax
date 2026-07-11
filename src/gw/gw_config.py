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
    config.backend     — FFI/IO backend selection (slab_io / gspace_io / screening_solver)
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
      FFI (collective MPI-IO).  GPU backend default.  ~5× faster than the
      rank-0 path once Lustre striping is applied.  C++ side requires
      CUDA (cudaMemcpyAsync D2H, cudaEvent sync); GPU-only.
    - ``PHDF5_HOST`` — host-side equivalent of ``PHDF5_FFI``: each rank
      writes its own hyperslab via parallel HDF5 driven by mpi4py +
      h5py(parallel).  Spiritually identical to the FFI path minus the
      cudaMemcpy.  CPU backend default when the venv has mpi4py +
      h5py-parallel installed.
    - ``H5PY_ALLGATHER`` — gather to rank 0 and write via serial h5py.
      Last-resort fallback for systems without either parallel HDF5 or
      the FFI.  Slow at scale (rank-0 disk bandwidth bottleneck).
    """
    PHDF5_FFI = "phdf5_ffi"
    PHDF5_HOST = "phdf5_host"
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
    # Explicit SlabIO backend override: "auto" (default — route by
    # use_ffi_io + platform, see from_input_file) | "phdf5_ffi" |
    # "phdf5_host" | "h5py_allgather".  Every enum value is reachable
    # from the input file; auto keeps the legacy use_ffi_io semantics.
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
    # Legacy aliases (deprecation-warned): cusolvermp_charge /
    # cusolvermp_lu with values auto|on|off (on → cusolvermp).
    "distributed_cholesky": "auto",
    "distributed_lu":       "auto",
    # Deprecated aliases (still parsed; warned at load; honored only when
    # the portable key above is left at "auto"):
    "cusolvermp_charge": "auto",   # auto | on | off
    "cusolvermp_lu":     "auto",   # auto | on | off
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
}

# Keys whose string values should be lowercased and stripped
_NORMALIZE_STR = {
    "compute_mode",
    "qp_solver",
    "sc_accelerator",
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

        for _legacy_key, _new_key in (
            ("cusolvermp_charge", "distributed_cholesky"),
            ("cusolvermp_lu", "distributed_lu"),
        ):
            if section.get(_legacy_key, fallback=None) is not None:
                import warnings
                warnings.warn(
                    f"Input key '{_legacy_key}' is deprecated; use "
                    f"'{_new_key} = auto|off|cusolvermp' (legacy 'on' → "
                    f"'cusolvermp').",
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
        if self.omega_accumulation not in ("auto", "kij", "kij_stream"):
            raise ValueError(
                "ppm.omega_accumulation must be auto/kij/kij_stream.")
        if self.invalid_mode not in (
            "zero", "skip", "2ry", "static_limit", "infinity", "imaginary"
        ):
            raise ValueError(
                f"ppm.invalid_mode: unknown value {self.invalid_mode!r}")
        if self.omega_batch_size < 1:
            raise ValueError("ppm.omega_batch_size must be >= 1.")


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
    """
    max_iter: int
    tol_ev: float
    accelerator: str      # "rcrop" | "linear"
    history_depth: int
    mixing: float
    dump_dir: str | None

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
    distributed_cholesky: str  # "auto" | "off" | "cusolvermp" | "slate"
    distributed_lu: str        # "auto" | "off" | "cusolvermp"
    gamma_contract_mode: str  # "take" | "einsum" | "scan"

    def summary(self) -> str:
        """One-line "what's active" for the run banner."""
        return (
            f"backend: slab_io={self.slab_io.value}, "
            f"gspace_io={self.gspace_io.value}, "
            f"screening_solver={self.screening_solver.value}, "
            f"distributed_cholesky={self.distributed_cholesky}, "
            f"distributed_lu={self.distributed_lu}, "
            f"gamma_contract={self.gamma_contract_mode}"
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
        - ``fixed_point`` / ``self_consistent`` × dynamic mode with
          ``sigma_omega_accumulation = kij_stream`` → error (streamed
          Σ_c(ω) leaves no in-memory tensor for the on-shell solve /
          QSGW rebuild; previously this pair silently degraded the eigh
          outputs to static COHSEX).
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
        if (solver in (QPSolver.FIXED_POINT, QPSolver.SELF_CONSISTENT)
                and mode.is_dynamic
                and self.ppm.omega_accumulation == "kij_stream"):
            raise ValueError(
                f"qp_solver={solver.value} is incompatible with "
                f"sigma_omega_accumulation=kij_stream: streamed Σ_c(ω) "
                f"leaves no in-memory ω-tensor for the on-shell solve / "
                f"QSGW build (the eigh-family outputs would silently "
                f"degrade to static Σ).  Use 'kij' or 'auto'.")
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
        # 0.0 (default) = auto: the planner uses its ns²-aware default
        # (higher for scalar, lower for bispinor's 4× pair density).  A
        # positive env value overrides it, clamped to [0.85, 1.0].
        try:
            chunk_utilization = float(
                os.environ.get("ISDF_CHUNK_TARGET_UTILIZATION", "0.0"))
        except Exception:
            chunk_utilization = 0.0
        if chunk_utilization > 0:
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
        # Auto-route GPU FFIs off on the CPU backend.  The phdf5 FFI is
        # CUDA-only at the C++ level (cudaMemcpyAsync D2H, cudaEvent sync
        # — see ``src/ffi/phdf5/cpp/write_ffi.cc``).  cuSOLVERMp / cuBLASMp
        # are similarly GPU-only.  On the CPU backend:
        #
        #   * ``use_ffi_io=true`` → ``SlabIOBackend.PHDF5_HOST`` (parallel
        #     HDF5 via mpi4py + h5py-parallel — same per-rank collective
        #     MPI-IO write as the FFI, no cudaMemcpy needed); falls back
        #     to ``H5PY_ALLGATHER`` if the venv lacks mpi4py / h5py-parallel.
        #   * ``cusolvermp_charge`` / ``cusolvermp_lu`` → ``"off"`` (in-tree
        #     ``cholesky_2d`` and per-q ``jnp.linalg.solve`` paths).
        #
        # User-facing: same ``cohsex.in`` works on both backends.
        _use_ffi_io_in = bool(_g("use_ffi_io"))
        _slab_io_in = str(_g("slab_io")).strip().lower()
        if _slab_io_in not in ("auto", "phdf5_ffi", "phdf5_host", "h5py_allgather"):
            raise ValueError(
                f"slab_io={_slab_io_in!r} invalid; expected auto / phdf5_ffi "
                f"/ phdf5_host / h5py_allgather.")
        # Distributed-linalg axes.  Legacy ``cusolvermp_charge`` /
        # ``cusolvermp_lu`` (auto|on|off; deprecation-warned at parse)
        # are honored only when the portable key is left at "auto".
        _LEGACY_LINALG = {"auto": "auto", "on": "cusolvermp", "off": "off"}
        _dist_chol = str(_g("distributed_cholesky")).strip().lower()
        _dist_lu = str(_g("distributed_lu")).strip().lower()
        for _legacy_key, _cur in (
            ("cusolvermp_charge", _dist_chol),
            ("cusolvermp_lu", _dist_lu),
        ):
            _legacy_val = str(_g(_legacy_key)).strip().lower()
            _mapped = _LEGACY_LINALG.get(_legacy_val)
            if _mapped is None:
                raise ValueError(
                    f"{_legacy_key}={_legacy_val!r} invalid; expected auto/on/off.")
            if _cur == "auto" and _mapped != "auto":
                if _legacy_key == "cusolvermp_charge":
                    _dist_chol = _mapped
                else:
                    _dist_lu = _mapped
        if _dist_chol not in ("auto", "off", "cusolvermp", "slate"):
            raise ValueError(
                f"distributed_cholesky={_dist_chol!r} invalid; expected "
                f"auto / off / cusolvermp / slate.")
        if _dist_lu not in ("auto", "off", "cusolvermp", "scalapack"):
            raise ValueError(
                f"distributed_lu={_dist_lu!r} invalid; expected auto / off "
                f"/ cusolvermp / scalapack (a SLATE getrf wrapper does not "
                f"exist yet; scalapack is the host/CPU-backend option).")
        try:
            import jax as _jax
            _is_cpu_backend = _jax.default_backend() == "cpu"
        except Exception:
            _is_cpu_backend = False
        _slab_io_choice = (SlabIOBackend.PHDF5_FFI if _use_ffi_io_in
                           else SlabIOBackend.H5PY_ALLGATHER)
        if _is_cpu_backend:
            if _use_ffi_io_in:
                try:
                    import mpi4py  # noqa: F401
                    import h5py
                    _have_parallel_h5 = bool(h5py.get_config().mpi)
                except Exception:
                    _have_parallel_h5 = False
                if _have_parallel_h5:
                    print_fn(
                        "  [config] use_ffi_io=true on CPU backend; "
                        "phdf5 FFI is CUDA-only.  Routing SlabIO through "
                        "PHDF5_HOST (mpi4py + h5py-parallel) — same "
                        "per-rank collective MPI-IO write semantics."
                    )
                    _slab_io_choice = SlabIOBackend.PHDF5_HOST
                else:
                    print_fn(
                        "  [config] use_ffi_io=true on CPU backend but "
                        "venv lacks mpi4py / h5py-parallel; falling back "
                        "to H5PY_ALLGATHER (rank-0 serial write — slow "
                        "at scale).  Build h5py with HDF5_MPI=ON against "
                        "the system's parallel HDF5 to get PHDF5_HOST."
                    )
                    _slab_io_choice = SlabIOBackend.H5PY_ALLGATHER
            if _dist_chol not in ("off", "slate"):
                # slate passes through: it has a host-platform FFI
                # (liblorrax_ffi_host.so, Target::HostTask) and keeps the
                # explicit-request-fails-loudly semantics on CPU too.
                print_fn(
                    f"  [config] distributed_cholesky={_dist_chol} "
                    "requested but JAX backend is CPU; cuSOLVERMp is "
                    "CUDA-only and auto never picks SLATE.  Forcing 'off' "
                    "(in-tree sharded_cholesky).  SLATE's host FFI is "
                    "available via explicit distributed_cholesky = slate."
                )
                _dist_chol = "off"
            if _dist_lu not in ("off", "scalapack"):
                # scalapack passes through: it is the host-platform LU
                # backend and keeps explicit-request-fails-loudly
                # semantics on CPU too.
                print_fn(
                    f"  [config] distributed_lu={_dist_lu} requested "
                    "but JAX backend is CPU; cuSOLVERMp is CUDA-only and "
                    "auto never picks ScaLAPACK.  Forcing 'off' (in-tree "
                    "per-q jnp.linalg.solve).  The ScaLAPACK host FFI is "
                    "available via explicit distributed_lu = scalapack."
                )
                _dist_lu = "off"
        if _slab_io_in != "auto":
            # Explicit backend: no platform second-guessing — a wrong
            # choice fails loudly at SlabIO open (e.g. phdf5_ffi on CPU),
            # which beats silently running a different backend than the
            # input file says.
            _slab_io_choice = SlabIOBackend(_slab_io_in)
        backend = BackendConfig(
            slab_io=_slab_io_choice,
            gspace_io=GspaceIO(str(_g("gspace_mode")).strip().lower()),
            screening_solver=_LEGACY_ISDF_MEMORY_MODE[isdf_memory_mode],
            distributed_cholesky=_dist_chol,
            distributed_lu=_dist_lu,
            gamma_contract_mode=str(_g("gamma_contract_mode")).strip().lower(),
        )
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
        )

        return cls(
            # Top-level: system + mode flags
            nval=int(_g("nval")),
            ncond=int(_g("ncond")),
            nband=int(_g("nband")),
            sys_dim=int(_g("sys_dim")),
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
