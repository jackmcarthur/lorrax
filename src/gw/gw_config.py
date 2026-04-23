"""Unified configuration for LORRAX GW calculations.

LorraxConfig replaces the old params dict + cfg SimpleNamespace + derivative
config objects. It is created once from the input file and passed through
the entire driver pipeline.

Derived config sub-objects (MinimaxConfig, SigmaQuadratureConfig,
PPMSigmaRuntimeOptions) are constructed on demand via properties.
"""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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
    "output_file": "eqp0_noqsym.dat",
    "kin_ion_file": "kin_ion.h5",
    "eqp_output_file": "eqp.dat",
    "sigma_omega_h5_file": "sigma_mnk.h5",
    "sigma_kij_h5_file": "",
    # Core flags
    "restart": True,
    "x_only": False,
    "do_screened": True,
    "bispinor": False,
    "do_G0": True,
    "self_consistent": False,
    "use_ppm_sigma": False,
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
    "ppm_omega_p": 2.0,
    "ppm_fallback_omega": 2.0,
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
    "sigma_debug_split_contrib": False,
    "sigma_freq_debug_output": False,
    "ppm_sigma_debug_static_norm": False,
    "ppm_static_cohsex_check": False,
    "sigma_debug_quadrature": False,
    "sigma_debug_quadrature_samples": 200,
    "write_w_copies_debug": False,
    "w_copies_debug_file": "",
    "sigma_freq_debug_file": "sigma_freq_debug.dat",
}

# Keys whose string values should be lowercased and stripped
_NORMALIZE_STR = {
    "wcoul0_source", "screening_method", "minimax_energy_reference",
    "sigma_omega_accumulation", "fermi_reference",
    "isdf_memory_mode",
    "ppm_invalid_mode",
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

@dataclass(frozen=True)
class LorraxConfig:
    """Unified, immutable configuration for a LORRAX GW calculation.

    Created once via ``LorraxConfig.from_input_file()`` and threaded through
    the entire driver. Replaces the old params dict + cfg SimpleNamespace.
    """

    # --- System geometry ---
    nval: int
    ncond: int
    nband: int
    sys_dim: int

    # --- File paths (resolved to absolute) ---
    wfn_file: str
    centroids_file: str
    output_file: str
    kin_ion_file: str
    eqp_output_file: str
    sigma_omega_h5_file: str
    sigma_kij_h5_file: str

    # --- Core flags ---
    restart: bool
    x_only: bool
    do_screened: bool
    bispinor: bool
    do_G0: bool
    self_consistent: bool
    use_ppm_sigma: bool
    use_ffi_io: bool
    gspace_mode: str
    use_aot_chunk_chooser: bool

    # --- Memory / chunking (resolved at construction) ---
    memory_per_device_gb: float
    chunk_target_utilization: float
    chunk_size: int
    band_chunk_size: int
    r_chunk_override: int
    zct_stage_cap_gb: float | None

    # --- ISDF ---
    isdf_memory_mode: str
    mc_average_vcoul_body: bool
    bare_coulomb_cutoff: float | None
    use_bgw_vcoul: bool
    bgw_vcoul_file: str | None
    bgw_vcoul_sym_wfn: str | None

    # --- Coulomb head ---
    wcoul0_source: str
    wcoul0_eta: float
    vhead: float | None
    whead_0freq: float | None
    whead_imfreq: float | None

    # --- Screening / minimax ---
    screening_method: str
    minimax_target_error: float
    minimax_max_nodes: int
    regenerate_minimax_tables: bool
    minimax_energy_reference: str

    # --- PPM ---
    ppm_omega_p: float
    ppm_fallback_omega: float
    ppm_sigma_target_error: float
    ppm_sigma_max_nodes: int

    # --- Sigma frequency grid ---
    sigma_omega_min_ev: float
    sigma_omega_max_ev: float
    sigma_omega_step_ev: float
    sigma_regularization_ev: float
    sigma_window_edge_factor: float
    sigma_omega_batch_size: int
    sigma_omega_accumulation: str

    # --- PPM sigma options ---
    ppm_sigma_scale: float
    ppm_sigma_flip_neg: bool
    ppm_invalid_mode: str
    fermi_reference: str
    sigma_at_dft_extrapolate: bool
    sigma_at_dft_energies: bool

    # --- Debug ---
    debug_hartree: bool
    debug_omega: float | None
    sigma_debug_split_contrib: bool
    sigma_freq_debug_output: bool
    ppm_sigma_debug_static_norm: bool
    ppm_static_cohsex_check: bool
    sigma_debug_quadrature: bool
    sigma_debug_quadrature_samples: int
    write_w_copies_debug: bool
    w_copies_debug_file: str
    sigma_freq_debug_file: str

    # --- Optional parsed blocks ---
    kpoints_crystal_b: dict | None = None

    # --- Input directory (for resolving relative paths at runtime) ---
    input_dir: str = ""

    # ------------------------------------------------------------------
    #  Derived config objects
    # ------------------------------------------------------------------

    @property
    def minimax_config(self):
        from .minimax_config import MinimaxConfig
        return MinimaxConfig(
            target_error=self.minimax_target_error,
            max_nodes=self.minimax_max_nodes,
            regenerate_tables=self.regenerate_minimax_tables,
            energy_reference=self.minimax_energy_reference,
        )

    @property
    def sigma_quadrature_config(self):
        from .minimax_config import SigmaQuadratureConfig
        return SigmaQuadratureConfig(
            target_error=self.ppm_sigma_target_error,
            max_nodes=self.ppm_sigma_max_nodes,
            crossing_max_nodes=max(500, self.ppm_sigma_max_nodes),
            crossing_eps_q=1.0e-3,
            regenerate_tables=self.regenerate_minimax_tables,
        )

    @property
    def omega_grid_ry(self):
        """Sigma frequency grid in Rydberg."""
        ryd2ev = 13.6056980659
        return np.arange(
            self.sigma_omega_min_ev / ryd2ev,
            (self.sigma_omega_max_ev + 0.5 * self.sigma_omega_step_ev) / ryd2ev,
            self.sigma_omega_step_ev / ryd2ev,
        )

    @property
    def omega_grid_ev(self):
        """Sigma frequency grid in eV."""
        return np.arange(
            self.sigma_omega_min_ev,
            self.sigma_omega_max_ev + 0.5 * self.sigma_omega_step_ev,
            self.sigma_omega_step_ev,
        )

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_input_file(cls, filename: str, *, print_fn=print) -> LorraxConfig:
        """Parse input file and resolve runtime settings (memory, env vars).

        This replaces read_cohsex_input + resolve_runtime_config + path
        resolution in a single call.
        """
        from file_io import resolve_input_paths

        params = read_lorrax_input(filename)
        input_dir = os.path.dirname(os.path.abspath(filename))
        resolve_input_paths(params, input_dir)

        # --- Validate ---
        isdf_memory_mode = str(params.get("isdf_memory_mode", "auto")).strip().lower()
        if isdf_memory_mode not in ("auto", "high_mem", "low_mem"):
            raise ValueError(f"isdf_memory_mode={isdf_memory_mode!r} invalid; "
                             f"expected auto|high_mem|low_mem")
        if params["x_only"] and params["do_screened"]:
            raise ValueError("x_only and do_screened cannot both be True.")

        # --- Memory auto-detection ---
        memory_per_device_gb = float(params.get("memory_per_device_gb", 0.0))
        if memory_per_device_gb <= 0:
            from common.gpu_utils import get_device_memory_gb
            memory_per_device_gb = get_device_memory_gb()
            print_fn(f"  Auto-detected memory budget: {memory_per_device_gb:.2f} GB/device")

        # --- Chunk utilization from env ---
        try:
            chunk_utilization = float(os.environ.get("ISDF_CHUNK_TARGET_UTILIZATION", "0.97"))
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
                zct_stage_cap_gb = min(memory_per_device_gb, max(0.0, float(zct_cap_env)))
            except Exception:
                pass
        if zct_stage_cap_gb is None and zct_frac_env and jax.default_backend() in ("gpu", "cuda"):
            from common.gpu_utils import get_device_memory_info
            total_gb = float(get_device_memory_info().get("total_gb", 0.0))
            if total_gb > 0:
                try:
                    frac = max(0.10, min(0.95, float(zct_frac_env)))
                    zct_stage_cap_gb = min(memory_per_device_gb, frac * total_gb)
                except Exception:
                    pass

        def _get(key):
            return params.get(key, _DEFAULTS.get(key))

        return cls(
            # System
            nval=int(_get("nval")),
            ncond=int(_get("ncond")),
            nband=int(_get("nband")),
            sys_dim=int(_get("sys_dim")),
            # Paths
            wfn_file=str(_get("wfn_file")),
            centroids_file=str(_get("centroids_file")),
            output_file=str(_get("output_file")),
            kin_ion_file=str(_get("kin_ion_file")),
            eqp_output_file=str(_get("eqp_output_file")),
            sigma_omega_h5_file=str(_get("sigma_omega_h5_file")),
            sigma_kij_h5_file=str(_get("sigma_kij_h5_file") or ""),
            # Core flags
            restart=bool(_get("restart")),
            x_only=bool(_get("x_only")),
            do_screened=bool(_get("do_screened")),
            bispinor=bool(_get("bispinor")),
            do_G0=bool(_get("do_G0")),
            self_consistent=bool(_get("self_consistent")),
            use_ppm_sigma=bool(_get("use_ppm_sigma")),
            use_ffi_io=bool(_get("use_ffi_io")),
            gspace_mode=str(_get("gspace_mode")),
            use_aot_chunk_chooser=bool(_get("use_aot_chunk_chooser")),
            # Memory / chunking
            memory_per_device_gb=memory_per_device_gb,
            chunk_target_utilization=chunk_utilization,
            chunk_size=int(_get("chunk_size")),
            band_chunk_size=int(_get("band_chunk_size")),
            r_chunk_override=int(_get("r_chunk_size")),
            zct_stage_cap_gb=zct_stage_cap_gb,
            # ISDF
            isdf_memory_mode=isdf_memory_mode,
            mc_average_vcoul_body=bool(_get("mc_average_vcoul_body")),
            bare_coulomb_cutoff=_get("bare_coulomb_cutoff"),
            use_bgw_vcoul=bool(_get("use_bgw_vcoul")),
            bgw_vcoul_file=(str(_get("bgw_vcoul_file")) or None),
            bgw_vcoul_sym_wfn=(str(_get("bgw_vcoul_sym_wfn")) or None),
            # Coulomb head
            wcoul0_source=str(_get("wcoul0_source")).strip().lower(),
            wcoul0_eta=float(_get("wcoul0_eta") or 0.0),
            vhead=_get("vhead"),
            whead_0freq=_get("whead_0freq"),
            whead_imfreq=_get("whead_imfreq"),
            # Screening
            screening_method=str(_get("screening_method")).strip().lower(),
            minimax_target_error=float(_get("minimax_target_error")),
            minimax_max_nodes=int(_get("minimax_max_nodes")),
            regenerate_minimax_tables=bool(_get("regenerate_minimax_tables")),
            minimax_energy_reference=str(_get("minimax_energy_reference")).strip().lower(),
            # PPM
            ppm_omega_p=float(_get("ppm_omega_p")),
            ppm_fallback_omega=float(_get("ppm_fallback_omega")),
            ppm_sigma_target_error=float(_get("ppm_sigma_target_error")),
            ppm_sigma_max_nodes=int(_get("ppm_sigma_max_nodes")),
            # Sigma grid
            sigma_omega_min_ev=float(_get("sigma_omega_min_ev")),
            sigma_omega_max_ev=float(_get("sigma_omega_max_ev")),
            sigma_omega_step_ev=float(_get("sigma_omega_step_ev")),
            sigma_regularization_ev=float(_get("sigma_regularization_ev")),
            sigma_window_edge_factor=float(_get("sigma_window_edge_factor")),
            sigma_omega_batch_size=int(_get("sigma_omega_batch_size")),
            sigma_omega_accumulation=str(_get("sigma_omega_accumulation")).strip().lower(),
            # PPM sigma
            ppm_sigma_scale=float(_get("ppm_sigma_scale")),
            ppm_sigma_flip_neg=bool(_get("ppm_sigma_flip_neg")),
            ppm_invalid_mode=str(_get("ppm_invalid_mode") or "static_limit"),
            fermi_reference=str(_get("fermi_reference")).strip().lower(),
            sigma_at_dft_extrapolate=bool(_get("sigma_at_dft_extrapolate")),
            sigma_at_dft_energies=bool(_get("sigma_at_dft_energies")),
            # Debug
            debug_hartree=bool(_get("debug_hartree")),
            debug_omega=_get("debug_omega"),
            sigma_debug_split_contrib=bool(_get("sigma_debug_split_contrib")),
            sigma_freq_debug_output=bool(_get("sigma_freq_debug_output")),
            ppm_sigma_debug_static_norm=bool(_get("ppm_sigma_debug_static_norm")),
            ppm_static_cohsex_check=bool(_get("ppm_static_cohsex_check")),
            sigma_debug_quadrature=bool(_get("sigma_debug_quadrature")),
            sigma_debug_quadrature_samples=int(_get("sigma_debug_quadrature_samples")),
            write_w_copies_debug=bool(_get("write_w_copies_debug")),
            w_copies_debug_file=str(_get("w_copies_debug_file") or ""),
            sigma_freq_debug_file=str(_get("sigma_freq_debug_file")),
            # Parsed blocks
            kpoints_crystal_b=params.get("kpoints_crystal_b"),
            input_dir=input_dir,
        )
