"""GW driver output: banner, summary, and result serialization.

Analogous to QE's ``punch()`` / ``pw_restart_new`` — all format-specific
I/O lives here so the driver reads like a Methods section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from common.units import RYD_TO_EV
# The channel-availability table (L1, jax-free) — the writers below ask it
# which Σ channels a run built instead of hand-checking mode strings.
from .gw_config import (
    ComputeMode, SigmaChannel, UNIMPLEMENTED_MODES, explain_missing_channels,
    mode_builds_channels, sigma_channels_for)

# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class GWResults:
    """All quantities produced by a GW calculation, ready for serialization.

    Self-energy arrays are in **Rydberg** (the internal unit).  The writer
    converts to eV when producing human-readable files.

    Static-COHSEX components (``sig_sx``, ``sig_coh``) and bare exchange
    (``sig_x``) are always populated.  When ``use_ppm=True`` the dynamic
    correlation diagonal is in ``sigma_c_diag_at_dft_ry`` and the writer
    emits sigX/sigC columns instead of sigSX/sigCOH.

    Attributes
    ----------
    sig_sx : np.ndarray, (nk, nb, nb)
        Static screened-exchange Σ_SX (Ry).  In PPM mode this is still
        the static COHSEX value, retained for diagnostics/restart.
    sig_coh : np.ndarray, (nk, nb, nb)
        Static Coulomb-hole Σ_COH (Ry).
    sig_h : np.ndarray, (nk, nb, nb)
        Hartree self-energy (Ry).
    sig_x : np.ndarray, (nk, nb, nb)
        Bare exchange Σ_X (Ry).  Used as the "sigX" column in PPM mode
        and as a quality-of-fit check in COHSEX mode.
    E_qp_ry : np.ndarray, (nk, nb)
        Quasiparticle eigenvalues from diagonalisation (Rydberg).
    U_qp : np.ndarray, (nk, nb, nb)
        Quasiparticle eigenvectors  U[k,m,n] = ⟨m_DFT|n_QP⟩.
    E_dft_ry : np.ndarray, (nk, nb)
        DFT reference eigenvalues (Rydberg).
    kin_ion_ry : np.ndarray, (nk, nb, nb)
        H₀ = T + V_ion (+ V_H when ``kin_ion_has_hartree``) matrix (Ry).
    kin_ion_has_hartree : bool
        True when ``kin_ion.h5`` carried ``has_hartree=True``, i.e. the
        exact FFT-grid mean-field V_H is already inside ``kin_ion_ry``.
        The writer then adds NO Hartree term of its own; ``sig_h`` is
        already zeroed upstream in ``sigma_dispatch``, and this flag
        makes the writer's contract explicit rather than implicit.
    band_start, band_stop : int
        0-based band window [band_start, band_stop).
    use_ppm : bool
        If True, labels switch from SX/COH to X/C in output files and
        the writer pulls the correlated diagonal from
        ``sigma_c_diag_at_dft_ry``.
    self_consistent : bool
        Whether the self-energy was obtained self-consistently.
    sigma_c_diag_at_dft_ry : np.ndarray or None, (nk, nb)
        Diagonal of Σ_c interpolated at DFT energies (Ry).  Present only
        for PPM non-SC runs; the writer expands it to a band-diagonal
        matrix for the eqp0.dat ``sigC`` column.
    sigma_xc_at_dft_ev : np.ndarray or None, (nk, nb)
        Diagonal Σ_xc interpolated at DFT energies (eV).  Present only
        for G₀W₀-PPM non-self-consistent runs.
    sigma_omega_h5_path : str or None
        Path to the frequency-dependent σ(ω) HDF5 file, if written.
    tensors_filename : str or None
        Path to the ISDF restart file, for the closing status line.
    """

    sig_sx: np.ndarray
    sig_coh: np.ndarray
    sig_h: np.ndarray
    sig_x: np.ndarray
    E_qp_ry: np.ndarray
    U_qp: np.ndarray
    E_dft_ry: np.ndarray
    kin_ion_ry: np.ndarray
    band_start: int
    band_stop: int
    use_ppm: bool = False
    self_consistent: bool = False
    sigma_c_diag_at_dft_ry: np.ndarray | None = None
    sigma_xc_at_dft_ev: np.ndarray | None = None
    # Full ω-grid Σ_c diagonal (PPM modes only) — drives the Z-factor in
    # the BGW eqp1.dat writer.  Shape (n_omega, nk, nb_sigma), eV; ω is
    # relative to the DFT mid-gap E_F.  None in static modes ⇒ Z=1.
    sigma_c_omega_diag_ev: np.ndarray | None = None
    omega_rel_ev: np.ndarray | None = None
    # Mid-gap Fermi level computed once in ``ppm_pipeline`` from
    # ``meta.nelec`` (NOT from the sigma-window band count).  Required
    # by the eqp.dat writer in PPM modes; ``None`` in static modes.
    efermi_ev: float | None = None
    sigma_omega_h5_path: str | None = None
    tensors_filename: str | None = None
    kin_ion_has_hartree: bool = False
    #: 'stored' | 'folded' | 'isdf' | 'gspace' — see file_io.kin_ion.
    hartree_source: str | None = None


# ---------------------------------------------------------------------------
# Banner / summary  (QE ``summary()`` pattern)
# ---------------------------------------------------------------------------

def print_banner(
    backend: str,
    n_devices: int,
    grid_x: int,
    grid_y: int,
    n_procs: int,
    device_kind: str,
    print_fn=print,
):
    """Print the calculation header — device mesh, XLA pool, etc.

    Corresponds to the "announce what we are about to do" phase that QE's
    ``summary()`` / ``hp_summary()`` routines handle before any computation.
    """
    print_fn("")
    print_fn("=" * 72)
    print_fn("  COHSEX-JAX: Self-Energy Calculation")
    print_fn("=" * 72)
    print_fn(
        f"  Backend: {backend.upper():<8}  Devices: {n_devices}"
        f"  Mesh: {grid_x}×{grid_y}  Processes: {n_procs}"
    )
    print_fn(f"  Device type: {device_kind}")

    # ---- XLA GPU pool env -------------------------------------------------
    # These four variables are read ONLY by the CUDA PJRT plugin's option
    # builder (jaxlib ``generate_pjrt_gpu_plugin_options``); no CPU code
    # path consults any of them.  The old banner echoed the RAW strings
    # unconditionally, which stated a GPU fact on a CPU run and, worse,
    # printed "XLA preallocate: FALSE" for a value jax resolves as TRUE
    # (its test is case-sensitive) and "mem_fraction: unset" for a run
    # using the current variable name.  Report the RESOLVED values, and
    # qualify them by the live backend.
    from .gw_config import classify_xla_pool, resolve_xla_gpu_memory_env
    _xm = resolve_xla_gpu_memory_env()
    _is_gpu = str(backend).strip().lower() in ("gpu", "cuda", "rocm")
    if _is_gpu:
        _alloc = _xm.allocator
        if _xm.allocator_raw is not None and _xm.allocator_raw != _alloc:
            _alloc += f" (from {_xm.allocator_raw!r})"
        _frac = (f"{_xm.mem_fraction} [{_xm.mem_fraction_var}]"
                 if _xm.mem_fraction else "unset")
        print_fn(
            f"  XLA allocator: {_alloc}  preallocate: "
            f"{'on' if _xm.preallocate else 'off'}  mem_fraction: {_frac}")
        if not _xm.allocator_is_valid:
            print_fn(
                f"  *** LORRAX SANITY: XLA_PYTHON_CLIENT_ALLOCATOR="
                f"{_xm.allocator_raw!r} is not one of default/platform/bfc/"
                f"cuda_async — jax raises on this at backend init. ***")
        if _xm.preallocate_looks_like_a_typo:
            print_fn(
                f"  *** LORRAX SANITY: XLA_PYTHON_CLIENT_PREALLOCATE="
                f"{_xm.preallocate_raw!r} reads as 'off' but jax's test is "
                f"case-sensitive (only 'false'/'False'/'0' disable), so "
                f"preallocation is ON. ***")
        if _xm.mem_fraction_conflict:
            print_fn(
                "  *** LORRAX SANITY: XLA_CLIENT_MEM_FRACTION and the "
                "deprecated XLA_PYTHON_CLIENT_MEM_FRACTION are BOTH set — "
                "jax raises on this at backend init. ***")
        elif _xm.mem_fraction_deprecated:
            print_fn(
                "  [note] XLA_PYTHON_CLIENT_MEM_FRACTION is deprecated in "
                "jax 0.9; the current name is XLA_CLIENT_MEM_FRACTION.")
        if _xm.peak_note:
            print_fn(f"  [note] {_xm.peak_note}")
        if _xm.tf_gpu_allocator_raw:
            print_fn(
                f"  [note] TF_GPU_ALLOCATOR={_xm.tf_gpu_allocator_raw!r} is "
                f"set but INERT for jax (a TensorFlow variable); it does not "
                f"select an allocator here.")
    else:
        _gpu_knobs = {n: os.environ.get(n) for n in (
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_MEM_FRACTION")}
        _set = [n for n, v in _gpu_knobs.items() if v]
        _tail = (f" (set but ignored: {', '.join(_set)})" if _set else "")
        print_fn(
            f"  XLA GPU pool knobs: not applicable on the {backend} "
            f"backend{_tail}")

    # ---- the CLIENT ------------------------------------------------------
    # Everything above is what the ENVIRONMENT says.  os.environ is a false
    # witness for allocator state — the allocator is fixed at backend init,
    # and bootstrap() reaches backend init via jax.devices() inside
    # fallback_to_cpu_if_no_gpu_backend(), so any variable set afterwards
    # changes a string and not the client (measured, job 7882443:
    # identical os.environ, bytes_limit 11.805 GB vs 0.000 GB).  Read the
    # device and say so when the two disagree.
    try:
        import jax
        # local_devices(), not devices(): jax.devices() is the GLOBAL list, so
        # jax.devices()[0] is process 0's device on every rank and the banner
        # would report another process's pool (or throw) at every P > 1.
        _stats = jax.local_devices()[0].memory_stats()
        _pool = classify_xla_pool(_stats, backend=backend, env=_xm)
        if _pool.accounting_present:
            _bl = (_stats or {}).get("bytes_limit", 0) / 1e9
            _bu = (_stats or {}).get("bytes_in_use", 0) / 1e9
            _pk = (_stats or {}).get("peak_bytes_in_use", 0) / 1e9
            if _bl > 0:
                print_fn(
                    f"  XLA pool: limit={_bl:.2f} GB, in_use={_bu:.2f} GB,"
                    f" avail={_bl - _bu:.2f} GB, peak={_pk:.2f} GB"
                )
            else:
                # MEASURED (job 7882478): cuda_async populates in_use and
                # peak_bytes_in_use but reports bytes_limit=0.  The old
                # ``avail = limit - in_use`` then printed a NEGATIVE
                # available pool — a number that cannot exist.
                print_fn(
                    f"  XLA pool: in_use={_bu:.2f} GB, peak={_pk:.2f} GB"
                    f"  (this allocator reports no bytes_limit, so there is"
                    f" no 'available' figure)"
                )
        else:
            # ``None``/``{}``/all-zero is the documented answer on backends
            # with no arena accounting (the CPU client, and the ``platform``
            # allocator).  Say so — a missing line with no reason reads as
            # "nothing to report", which is a different claim.
            print_fn("  XLA pool: the client reports no arena accounting "
                     "(bytes_limit=0) — no pool figures available")
        if _pool.disagreement:
            print_fn(f"  *** LORRAX SANITY: {_pool.disagreement} ***")
    except Exception as exc:
        # Previously ``except Exception: pass``.  A swallowed failure here
        # deleted the pool line from the banner with no trace, so an
        # operator could not tell "no pool" from "the probe broke".
        print_fn(f"  XLA pool: unavailable ({type(exc).__name__}: {exc})")

    print_fn("=" * 72)
    print_fn("")


def print_section(title: str, print_fn=print):
    """Print a section divider: ---- TITLE ----"""
    print_fn("")
    print_fn("-" * 72)
    print_fn(f"  {title}")
    print_fn("-" * 72)


def print_system_summary(
    n_rmu: int,
    fft_grid: tuple[int, int, int],
    cell_volume: float,
    print_fn=print,
):
    """Print ISDF basis and grid metadata before the computation begins."""
    print_fn(f"  ISDF basis: {n_rmu} centroids")
    print_fn(
        f"  FFT grid: {fft_grid[0]}×{fft_grid[1]}×{fft_grid[2]}"
        f"   Cell volume: {cell_volume:.2f} a.u.³"
    )
    print_fn("")


# ---------------------------------------------------------------------------
# Restart persistence — the deck's write/skip decision, W0 + q→0 head scalars
# ---------------------------------------------------------------------------

def restart_tensor_writes_enabled(config, tensors_filename: str) -> bool:
    """``write_restart_tensors``: may this run persist the ISDF tensors?

    ONE decision point for a key that gates FOUR writes — the V_q / G0 /
    E_nk flush with its W0 placeholder, the ψ append, the centroid-hash
    stamp (all in ``gw_init.prepare_isdf_and_wavefunctions``) and the W0 +
    head-scalar write here.  Asking ``config.write_restart_tensors`` at
    four sites would be four chances for one of them to keep writing, and
    a restart file containing three of its five datasets is worse than no
    file at all: the ``W0_ready`` and window guards would pass on it.

    ANNOUNCED, ONCE, ON RANK 0.  Skipping the artifact is a policy the deck
    chose and it is invisible in every other way — the run simply gets
    faster and ``tmp/`` stays empty — so it says so.  Deduped on the
    filename through ``ffi.gate.announce_once``, which is also what keeps
    the four call sites to one line.

    Returns ``True`` when the writes should happen (the default, and
    today's behaviour unchanged).
    """
    if getattr(config, "write_restart_tensors", True):
        return True
    from ffi.gate import announce_once
    announce_once(
        ("write_restart_tensors", "off", tensors_filename),
        f"  [restart_write] write_restart_tensors = false — SKIPPING "
        f"{os.path.basename(tensors_filename)} entirely (V_qmunu, G0_mu_nu, "
        f"enk_full, psi_full_y, W0_qmunu, head scalars).  Nothing in the GW "
        f"driver reads these back; a BSE run against this directory will "
        f"refuse on the missing file.",
        scope="rank0")
    return False


def persist_w0_and_head(
    W_q,
    *,
    tensors_filename: str,
    head_resolver,
    config,
    meta,
    mesh_xy,
    sym=None,
    centroid_indices=None,
    print_fn=print,
):
    """Persist W0_qmunu + q=0 head scalars to the ISDF restart file.

    ``sym`` / ``centroid_indices`` are the run's symmetry tables and
    centroid set, and they are here for ONE reason: the q-storage decision.
    V's writer resolves it in ``gw_init``; this writer must reach the SAME
    answer, and the only honest way to reach it is to ask the same
    resolution point about the same centroid set.  Omitting them resolves
    ``full`` — today's bytes — which is the right answer for a caller that
    cannot say which centroid set its W was computed against.

    Downstream consumers (BSE, future Σ-builders) reload these and apply
    the rank-1 head update via ``head_correction.apply_q0_head_rank1``.
    The ``whead`` axis is length 1 for COHSEX (just static) and length 2
    for GN-PPM (static + iω_p).  ``vhead``/``whead_*`` cohsex.in overrides
    flow through automatically because ``HeadResolver`` consults the
    config's override fields first before falling back to s_tensor/epshead.

    No-op unless ``config.do_screened``, the deck asked for restart tensor
    writes (``write_restart_tensors``), and the restart file exists.  The
    file-exists arm is not redundant with the key: a ``restart = true`` run
    reuses tensors it did not write, and would otherwise re-stamp a W0 into
    a file the deck said to leave alone.
    """
    if not config.do_screened:
        return
    if not restart_tensor_writes_enabled(config, tensors_filename):
        return
    if not os.path.exists(tensors_filename):
        return
    from file_io import write_w0_qmunu_to_h5, write_head_scalars_to_h5
    # W_q is already flat-q (nq, μ, μ).  The W0_qmunu placeholder
    # created by ``write_restart_state_to_h5(init_W0=True)`` is
    # rank-3 (sized from V_qmunu), so we write W_q at flat-q.
    # Previously this reshaped to legacy 8-D and tripped
    # ``phdf5 async write: dataset rank mismatch ds=/W0_qmunu
    # file_rank=3 write_rank=8``.  Downstream (BSE) consumers
    # of W0_qmunu were already updated to flat-q in commit a052a1c.
    # THE SAME DECISION V TOOK, taken the same way, on W's OWN capture.
    # ``sym``/``centroid_indices`` reach here through the resolver rather
    # than through this function's signature because the head/W0 persist
    # path has never carried them; passing the RUN's config and the
    # producer's tables is what keeps the two writers on one answer.
    # ``take_pre_unfold`` REMOVES, so a second persist in a
    # self-consistency loop cannot re-store the first iteration's W.
    from .restart_q_storage import (resolve_restart_q_storage_for_run,
                                    take_pre_unfold)
    from common import timing as _tmg
    with _tmg.section("persist_w0.resolve"):
        _qirr = resolve_restart_q_storage_for_run(
        config, sym=sym, centroid_indices=centroid_indices,
        # ``getattr``, not ``meta.fft_grid``: the grid is read ONLY when
        # ``sym``/``centroid_indices`` are both present (the resolver
        # short-circuits to ``full`` otherwise), and this function is
        # reachable with a minimal meta that carries neither.  Evaluating
        # it unconditionally would make a caller that asked for nothing
        # new fail on an attribute it never needed.
            fft_grid=getattr(meta, "fft_grid", None), print_fn=print_fn,
            context="W0 restart tensor")
    with _tmg.section("persist_w0.write_w0"):
        write_w0_qmunu_to_h5(tensors_filename, W_q,
                             n_rmu_logical=int(meta.n_rmu),
                             mesh=mesh_xy,
                             qirr=_qirr.with_capture(
                                 take_pre_unfold("W0_qmunu")))
    with _tmg.section("persist_w0.head_static"):
        head_static = head_resolver.at(0.0 + 0.0j)
    # THE STORED ω GRID IS THE PROBE SET, so it is the POLE MODEL that
    # decides it, not "is this run dynamic".  The two questions agreed for
    # as long as every dynamic mode was a two-point plasmon-pole fit; MPA
    # is dynamic and samples W on the double-parallel grid instead, so a
    # dynamic-else-GN read here would stamp {0, iω_p} onto a file whose W
    # was never evaluated there — a restart artifact that lies quietly.
    ppm_model = config.compute_mode.ppm_model
    if ppm_model is not None:
        # GN-PPM: probe at iωp on the imaginary axis.
        # HL-PPM: probe at Ω on the real axis (above all transitions).
        if ppm_model == "hl":
            omega_imp = complex(float(config.ppm.omega_p), 0.0)
            _omega_grid_entry = float(omega_imp.real)
        else:
            omega_imp = 1j * float(config.ppm.omega_p)
            _omega_grid_entry = float(omega_imp.imag)
        with _tmg.section("persist_w0.head_imag"):
            head_imag = head_resolver.at(omega_imp)
        whead_arr = np.array(
            [head_static.wcoul0, head_imag.wcoul0], dtype=np.complex128)
        omega_grid = np.array([0.0, _omega_grid_entry], dtype=np.float64)
    elif config.compute_mode.is_dynamic:
        # Dynamic, but not a two-point pole fit: MPA today.  There is no
        # probe frequency to record, and inventing one would be worse than
        # a refusal — this is the file the next run reads W's head back
        # from.  Unreachable from the driver (the mode is refused at
        # entry); it stands so that landing the MPA Σ stage has to answer
        # this question rather than inherit an answer.
        raise NotImplementedError(
            f"persist_w0_and_head: compute_mode = "
            f"{config.compute_mode.value} builds Σ_c(ω) without a two-point "
            f"plasmon-pole probe, so the {{0, probe}} head grid this writer "
            f"stores is not its sample set.  Give the mode its own head "
            f"persistence when its Σ stage lands.")
    else:
        whead_arr = np.array([head_static.wcoul0], dtype=np.complex128)
        omega_grid = np.array([0.0], dtype=np.float64)
    with _tmg.section("persist_w0.write_head"):
        write_head_scalars_to_h5(
            tensors_filename,
            vhead=complex(head_static.vc0),
            whead=whead_arr,
            omega_grid=omega_grid,
        )
    print_fn(
        f"  Persisted W0_qmunu + q=0 head scalars: "
        f"vhead={head_static.vc0.real:.3f} a.u.,  "
        f"whead[ω=0]={whead_arr[0].real:.3f} a.u."
        + (f",  whead[iωp]={whead_arr[1].real:.3f} a.u." if len(whead_arr) > 1 else "")
    )


# ---------------------------------------------------------------------------
# One-shot QP-WFN dump (SC writes its own via dump_qp_wfn_artifacts)
# ---------------------------------------------------------------------------

def write_qp_wfn_oneshot(
    U_full: np.ndarray,
    E_full: np.ndarray,
    *,
    wfn,
    band_slices,
    input_dir: str,
    print_fn=print,
):
    """One-shot WFN_qp.h5 dump (drop-in BSE / restart input).

    The one-shot dump is an IBZ-only writer: it rotates the DFT bands the
    WFN carries coefficients for, so it needs Σ (hence U_full/E_full) on the
    same k-set as the WFN.  When Σ is unfolded to the full BZ the eigh runs
    on nk_full > wfn.nkpts and this path cannot build the artifact — skip
    with a warning rather than crash the whole run (the writer would raise
    ValueError on the k-count mismatch).
    """
    import jax

    if int(U_full.shape[0]) != int(wfn.nkpts):
        print_fn(
            f"  QP WFN (one-shot): skipped — Σ on {int(U_full.shape[0])} "
            f"k-points but WFN carries {int(wfn.nkpts)} (IBZ); the one-shot "
            f"dump only supports IBZ-Σ. Set debug.write_wfn_h5=false to silence.")
        return
    from file_io.qp_wfn import write_qp_wfn_h5
    _qp_wfn_path = os.path.join(input_dir, "WFN_qp.h5")
    if jax.process_index() == 0:
        write_qp_wfn_h5(
            _qp_wfn_path, wfn=wfn,
            U_kmn=np.asarray(U_full, dtype=np.complex128),
            enk_active_qp_ry=np.asarray(E_full, dtype=np.float64),
            band_start=band_slices.b0, band_stop=band_slices.b3,
        )
    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("oneshot_qp_wfn_h5_write")
    except Exception:
        pass
    print_fn(f"  QP WFN (one-shot): {_qp_wfn_path}")


# ---------------------------------------------------------------------------
# Σ-decomposition frequency-debug table
# ---------------------------------------------------------------------------

def write_freq_debug(
    results: "GWResults",
    *,
    config,
    static_head_terms,
    omega_dft_rel_ev,
    head_sigma_diag_w_kn_ry,
    omega_grid_ry,
    print_fn=print,
):
    """Optional Σ-decomposition debug table (rank-0 caller, all in eV).

    Single seam at the H-build output: dumps the diagonal pieces that
    feed the BGW-format ``eqp0`` (Σ_tot at E_DFT) and ``eqp1`` (the same
    Σ_tot − E_DFT extrapolated by the Z-factor central-difference slope
    of dRe[Σ_c]/dω at E=E_DFT).  By construction
    ``eqp0 ≟ kin_ion + V_H + x_bare + sig_c(Edft).Re`` exactly, and
    ``eqp1 ≟ E_DFT + Z·(eqp0 − E_DFT)`` with Z=1 in static modes
    (degenerate eqp1==eqp0).  Head corrections are also exposed as
    their own columns: ``x_head`` (always when the head is computed)
    and either ``sig_c_head(Edft)`` (PPM) or ``sex_head/coh_head``
    (static, screened mode).

    No-op unless ``config.debug.sigma_freq_debug_output`` is set.
    """
    if not config.debug.sigma_freq_debug_output:
        return
    from file_io import write_sigma_freq_debug_table
    from .eqp_bgw import (
        compute_eqp_diag, compute_z_factor_from_omega_grid)

    use_ppm_c = (
        results.use_ppm and results.sigma_c_diag_at_dft_ry is not None)
    _e_dft_ev_full = np.asarray(results.E_dft_ry, dtype=np.float64) * RYD_TO_EV
    _kin_diag_ev = np.real(
        np.diagonal(np.asarray(results.kin_ion_ry), axis1=1, axis2=2)) * RYD_TO_EV
    _v_h_diag_ev = np.real(
        np.diagonal(np.asarray(results.sig_h), axis1=1, axis2=2)) * RYD_TO_EV
    _sig_x_diag_ev = np.real(
        np.diagonal(np.asarray(results.sig_x), axis1=1, axis2=2)) * RYD_TO_EV
    _nk, _nb = _e_dft_ev_full.shape
    # Static-COHSEX q→0 head: band-diagonal ``(nb,)`` shifts applied
    # in-place to Σ_x / Σ_SX / Σ_COH inside ``cohsex_sigma``.  The
    # bare-X piece (``sigma_x_diag``) is added in PPM mode too (since
    # Σ_x is static there as well), so ``x_head`` is emitted whenever
    # the head was computed.  ``sex_head`` / ``coh_head`` are
    # screened-channel pieces that only apply when ``do_screened``.
    def _broadcast_head_diag_to_kij(diag_n_ry: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            np.real(np.asarray(diag_n_ry)) * RYD_TO_EV, (_nk, _nb)
        ).astype(np.float64)

    _cols = [
        ("E_dft", _e_dft_ev_full),
        ("Edft-Ef", _e_dft_ev_full - float(results.efermi_ev or 0.0)),
        ("kin_ion", _kin_diag_ev),
        ("V_H", _v_h_diag_ev),
        ("x_bare", _sig_x_diag_ev),
    ]
    if static_head_terms is not None:
        _cols.append((
            "x_head",
            _broadcast_head_diag_to_kij(static_head_terms.sigma_x_diag),
        ))
    if use_ppm_c:
        # Compute Σ_c(E_DFT) + Z via the SAME recipe the eqp{0,1}.dat
        # writer uses, so the freq_debug sig_c(Edft) column and the
        # eqp0/eqp1 columns are bit-consistent.  PPM pipeline's own
        # interp produces a numerically slightly different value (~10
        # meV) due to a different vectorisation path.
        _e_dft_rel_ev = np.asarray(omega_dft_rel_ev, dtype=np.float64)
        _sigma_c_at_dft_for_eqp, _z_factor = (
            compute_z_factor_from_omega_grid(
                sigma_c_omega_diag_ev=np.asarray(
                    results.sigma_c_omega_diag_ev, dtype=np.complex128),
                omega_rel_ev=np.asarray(results.omega_rel_ev, dtype=np.float64),
                e_dft_rel_ev=_e_dft_rel_ev,
            ))
        _cols.append(("sig_c(Edft)", _sigma_c_at_dft_for_eqp))
        # PPM analytic head interpolated at the same E_DFT − E_F used
        # for ``sig_c(Edft)`` (same ω-grid, same linear-interp recipe
        # → cancellation analyses work column-by-column).
        if head_sigma_diag_w_kn_ry is not None:
            from .qsgw_utils import interp_along_omega
            _eval_ry = (np.asarray(omega_dft_rel_ev, np.float64)
                        / RYD_TO_EV)
            _cols.append((
                "sig_c_head(Edft)",
                interp_along_omega(
                    head_sigma_diag_w_kn_ry,
                    omega_grid_ry,
                    _eval_ry,
                ) * RYD_TO_EV,
            ))
    else:
        _cols.append(
            ("sex_0", np.real(np.diagonal(
                np.asarray(results.sig_sx), axis1=1, axis2=2)) * RYD_TO_EV))
        _cols.append(
            ("coh_0", np.real(np.diagonal(
                np.asarray(results.sig_coh), axis1=1, axis2=2)) * RYD_TO_EV))
        if static_head_terms is not None and config.do_screened:
            _cols.append((
                "sex_head",
                _broadcast_head_diag_to_kij(static_head_terms.sigma_sx_diag),
            ))
            _cols.append((
                "coh_head",
                _broadcast_head_diag_to_kij(static_head_terms.sigma_coh_diag),
            ))
    # eqp0 / eqp1 — same math as the eqp{0,1}.dat writer.  In PPM
    # mode (Σ_c, Z) were already computed above for the
    # ``sig_c(Edft)`` column; in static modes the combined Σ_SX +
    # Σ_COH plays the role of Σ_c(E_DFT) and Z = 1 so eqp1 == eqp0.
    # MODE-CORRECT split, and it must match compute_eqp_diag's contract:
    # that helper forms sigma_x_diag_ev + sigma_c_at_dft_diag_ev.  Under
    # static COHSEX, Sigma = Sigma_SX + Sigma_COH and there is NO separate
    # bare-exchange term -- so handing it x_bare as sigma_x while sigma_c
    # ALREADY contains sig_sx counts exchange TWICE.  MEASURED at Gamma
    # n=1: this column read -25.316578 eV against the correct -8.184804 eV,
    # a ~17 eV error.  Same defect class as the eqp{0,1}.dat writer (fixed
    # in 2987003); this is the second site, and it is why the two disagreed.
    _sig_x_for_eqp = _sig_x_diag_ev          # PPM: bare X is correct
    if not use_ppm_c:
        _sigma_c_at_dft_for_eqp = np.real(np.diagonal(
            np.asarray(results.sig_coh), axis1=1, axis2=2)) * RYD_TO_EV
        _sig_x_for_eqp = np.real(np.diagonal(
            np.asarray(results.sig_sx), axis1=1, axis2=2)) * RYD_TO_EV
        _z_factor = None
    _eqp0_ev, _eqp1_ev = compute_eqp_diag(
        kin_ion_diag_ev=_kin_diag_ev,
        hartree_diag_ev=_v_h_diag_ev,
        sigma_x_diag_ev=_sig_x_for_eqp,
        sigma_c_at_dft_diag_ev=np.asarray(
            _sigma_c_at_dft_for_eqp, dtype=np.complex128),
        e_dft_ev=_e_dft_ev_full,
        z_factor=_z_factor,
    )
    _cols.append(("eqp0", _eqp0_ev.astype(np.float64)))
    _cols.append(("eqp1", _eqp1_ev.astype(np.float64)))
    write_sigma_freq_debug_table(
        config.debug.sigma_freq_debug_file, _cols)
    print_fn(f"  Sigma freq debug: {config.debug.sigma_freq_debug_file}")


def _runnable_modes_building(*channels: SigmaChannel) -> str:
    """The deck spellings an operator could switch to, read off the table.

    "Runnable" excludes the modes that are declared but refuse to run
    (``UNIMPLEMENTED_MODES``), because this string ends up in advice: a
    message that tells an operator to try ``mpa`` today would send them
    into the entry refusal.  When a mode's Σ stage lands and its row
    leaves that dict, this advice picks it up with no edit here.
    """
    names = [m.value for m in ComputeMode
             if m not in UNIMPLEMENTED_MODES
             and all(c in sigma_channels_for(m) for c in channels)]
    return " / ".join(names)


# ---------------------------------------------------------------------------
# The QP ladders of sigma_mnk.h5's plotting appendix
# ---------------------------------------------------------------------------
# THREE LADDERS OF ONE H₀, which is the whole reason they are worth
# plotting together: ``kin_ion + V_H`` is common to all three and only
# the Σ_xc added to it changes, so the vertical distance between two
# curves IS the difference between two approximations at that (k, n) and
# nothing else.  They are eigenvalues, so unlike the Σ cubes beside them
# they are BASIS-FREE — ``eigvalsh(U†HU) == eigvalsh(H)`` — which is why
# one seam serves the one-shot and the self-consistent paths alike even
# though those two hand this function matrices in different bases.
#
# WHAT IS NOT HERE, and why it is not manufactured.  ``qp_static_cohsex_ev``
# is H₀ + Σ_SX + Σ_COH, and those two channels are built only by
# ``compute_mode = cohsex`` (``sigma_dispatch``: the dynamic branch calls
# ``compute_v_h_sigma_x``, which touches W not at all).  A PPM run could
# be made to produce SOMETHING for that name — its Σ_x + Σ_c(ω=0) is also
# a static self-energy — but that is a different operator, it already has
# a name here (``qp_omega0_ev``), and a plot that plots one quantity twice
# under two labels is worse than a plot with one curve missing.  So a run
# that did not build the channels omits the dataset and says so.
#
# WHICH RUNS THOSE ARE IS NOT DECIDED HERE ANY MORE.  This writer used to
# ask ``results.use_ppm`` — a proxy that was correct for the four modes
# that existed and would have gone on being asked as more arrived.  The
# question it was really asking, "did this run build Σ_SX and Σ_COH", now
# has a table: ``gw_config.MODE_SIGMA_CHANNELS``.  Asking the table means
# a mode that builds those channels by a route nobody has written yet gets
# the dataset, and a mode that does not gets the same named omission this
# writer has always printed, without anyone editing this file.

def write_qsgw_qp_ladders(
    results: "GWResults",
    *,
    config,
    e_qp_ry,
    sigma_c_omega_diag_ev,
    omega_grid_ev,
    sigma_c_omega=None,
    print_fn=print,
):
    """Append the QP energy ladders to ``sigma_mnk.h5`` (rank-0 caller).

    No-op unless the deck sets ``write_qsgw_datasets`` and this run wrote
    a ``sigma_mnk.h5`` at all.  Returns the dataset names written.

    ``e_qp_ry`` is the driver's own QP ladder (the eigh of
    ``½(H + H†)``, ``H = kin_ion + Σ_xc + V_H``) — passed rather than
    recomputed so the file cannot disagree with ``eqp0.dat`` about what
    this run's QP energies were.  It is not itself one of the three; it
    is the reference the three are read against.
    """
    if not bool(getattr(config, "write_qsgw_datasets", False)):
        return []
    path = results.sigma_omega_h5_path
    if not path:
        print_fn(
            "  write_qsgw_datasets = true, but compute_mode "
            f"{config.compute_mode.value} writes no sigma_mnk.h5 — the QP "
            "ladders have no file to go in.  Use one of the modes that "
            f"builds Σ_c(ω) ({_runnable_modes_building(SigmaChannel.C_OMEGA)})"
            " for the appendix.")
        return []
    from file_io import append_qsgw_datasets_h5
    from .qsgw_utils import (
        is_band_sharded_sigma_omega, solve_diagonal_sigma_fixed_point)

    kin_ion = np.asarray(results.kin_ion_ry)
    sig_h = np.asarray(results.sig_h)
    sig_x = np.asarray(results.sig_x)
    efermi_ev = float(results.efermi_ev or 0.0)
    payload: dict[str, np.ndarray] = {}
    omitted: list[str] = []

    # ---- qp_static_cohsex_ev: H₀ + Σ_SX + Σ_COH -----------------------
    # THE TABLE ANSWERS IT.  In a run that does not build these two
    # channels ``sig_sx`` / ``sig_coh`` are the zeros ``gw_jax``
    # substitutes for the Nones, and an eigh of kin_ion + V_H alone is not
    # the static COHSEX ladder by any reading — so the question is which
    # runs build them, and that is one row of MODE_SIGMA_CHANNELS rather
    # than this writer's own reading of ``results.use_ppm``.
    if not mode_builds_channels(config.compute_mode,
                                SigmaChannel.SX, SigmaChannel.COH):
        omitted.append(
            "qp_static_cohsex_ev ("
            + explain_missing_channels(config.compute_mode,
                                       SigmaChannel.SX, SigmaChannel.COH)
            + ")")
    else:
        payload["qp_static_cohsex_ev"] = _eigen_ladder_ev(
            kin_ion + sig_h + np.asarray(results.sig_sx)
            + np.asarray(results.sig_coh))

    # ---- qp_omega0_ev: H₀ + Σ_x + Σ_c(ω ≈ 0) --------------------------
    # ω₀ is the grid point nearest zero, i.e. nearest E_F, since the Σ_c
    # grid is Fermi-relative by construction (``ppm_pipeline``).  The ONE
    # ω slice is taken before the host transfer for the same reason
    # ``sigma_star_spread_stats`` takes one: the cube is (nω, nk, nb, nb)
    # and pulling all of it back to read 1/nω of it is 1.3 GB on the
    # mos2_4x4 deck for a 33 MB answer.
    #
    # AND IT IS SKIPPED ON THE BAND-SHARDED LAYOUT, because THIS caller is
    # rank-0-only (it sits in ``gw_jax``'s rank-0 output block, beside
    # ``write_freq_debug``).  A ``sigma_omega_layout = sharded`` cube is
    # tiled across every process, so the host transfer of one ω slice from
    # one rank is the "spans non-addressable devices" error, not a slow
    # success.  Gathering it here would mean a collective inside a
    # single-rank block; naming the layout costs the operator one line and
    # one rerun.
    if sigma_c_omega is None or omega_grid_ev is None:
        omitted.append("qp_omega0_ev (no Σ_c(ω) cube in this run)")
    elif is_band_sharded_sigma_omega(sigma_c_omega):
        omitted.append(
            "qp_omega0_ev (sigma_omega_layout = sharded; the ω slice cannot "
            "be read from one rank)")
    else:
        i0 = int(np.argmin(np.abs(np.asarray(omega_grid_ev))))
        sigma_c_0 = np.asarray(sigma_c_omega[i0])
        payload["qp_omega0_ev"] = _eigen_ladder_ev(
            kin_ion + sig_h + sig_x + sigma_c_0)

    # ---- qp_diag_self_consistent_ev: E = h₀ + ReΣ_xc(E) ---------------
    # The SAME solver and the same operands ``qsgw_utils.solve_qp``'s
    # fixed_point branch uses, run here in eV on the diagonals ``gw_jax``
    # already extracted — so this ladder is what ``qp_solver =
    # fixed_point`` would have solved, whatever solver this run used.
    # Bands whose E_DFT leaves the ω grid are CLAMPED TO E_DFT rather
    # than to the grid edge: the solver's Σ_c is pinned at the boundary
    # for those, and a pinned Σ makes a QP energy that says more about
    # the grid than about the band.
    if sigma_c_omega_diag_ev is not None and omega_grid_ev is not None:
        omega_ev = np.asarray(omega_grid_ev, dtype=np.float64)
        h0_diag_ev = np.real(np.diagonal(
            kin_ion + sig_h, axis1=1, axis2=2)) * RYD_TO_EV
        sig_x_diag_ev = np.real(np.diagonal(
            sig_x, axis1=1, axis2=2)) * RYD_TO_EV
        sigma_xc_diag_w_kn_ev = (np.real(np.asarray(sigma_c_omega_diag_ev))
                                 + sig_x_diag_ev[None, :, :])
        e_dft_rel_ev = (np.asarray(results.E_dft_ry, dtype=np.float64)
                        * RYD_TO_EV - efermi_ev)
        e_sc_rel_ev, _, n_iter = solve_diagonal_sigma_fixed_point(
            h0_diag_ev - efermi_ev, sigma_xc_diag_w_kn_ev, omega_ev,
            max_iter=120, tol_ev=1.0e-7, mixing=0.6,
        )
        in_grid = ((e_dft_rel_ev >= omega_ev[0])
                   & (e_dft_rel_ev <= omega_ev[-1]))
        e_sc_rel_ev = np.where(in_grid, e_sc_rel_ev, e_dft_rel_ev)
        payload["qp_diag_self_consistent_ev"] = (
            e_sc_rel_ev + efermi_ev).astype(np.float64)
        print_fn(
            f"  QP ladders: diagonal Σ(E) fixed point converged in "
            f"{n_iter} iterations, {int(in_grid.sum())}/{in_grid.size} "
            f"(k, n) inside the ω grid")
    else:
        omitted.append(
            "qp_diag_self_consistent_ev (no Σ_c(ω) diagonal in this run)")

    if omitted:
        print_fn(
            "  QP ladders: omitting " + "; ".join(sorted(omitted))
            + " — those operands are not in this run, and putting a "
            "different operator under one of those names would be worse "
            "than its absence")
    if not payload:
        return []
    # AGAINST THE RUN'S OWN LADDER, which is what makes the log line
    # readable without opening the file: each approximation is reported by
    # how far it sits from the E_qp this run actually used (the eigh of
    # kin_ion + Σ_xc + V_H, the same array eqp0.dat is built from).  Both
    # sides are per-k ascending eigenvalues, so the elementwise difference
    # is band-for-band.  Taken BEFORE the append, on the full-BZ arrays,
    # because after it the ladders are one row per star.
    ref_ev = np.asarray(e_qp_ry, dtype=np.float64) * RYD_TO_EV
    deltas = {
        name: float(np.abs(arr - ref_ev).max())
        for name, arr in payload.items() if arr.shape == ref_ev.shape
    }
    written = append_qsgw_datasets_h5(path, payload, print_fn=print_fn)
    if deltas:
        print_fn(
            "  QP ladders vs this run's E_qp (max |Δ|, eV): "
            + ", ".join(f"{n.removeprefix('qp_').removesuffix('_ev')} "
                        f"{d:.3f}" for n, d in sorted(deltas.items())))
    return written


def _eigen_ladder_ev(h_kij_ry) -> np.ndarray:
    """``eigvalsh`` of the Hermitian part, in eV.  ``(nk, nb)`` float64.

    The Hermitisation is the one ``gw_jax``'s QP eigh does, spelled the
    same way: these ladders exist to be compared against that one, so a
    different symmetrisation here would be a difference nobody asked for.
    """
    h = np.asarray(h_kij_ry)
    h = 0.5 * (h + np.conj(np.swapaxes(h, -1, -2)))
    return (np.linalg.eigvalsh(h) * RYD_TO_EV).astype(np.float64)


# ---------------------------------------------------------------------------
# H₀ sanity gate  (mean-field side of eqp0/eqp1/eqp_g0w0)
# ---------------------------------------------------------------------------

# ⟨nk|V_xc|nk⟩ for LDA/PBE pseudo-systems lives in a narrow, strictly
# negative band (measured: −25 … −5 eV for MoS₂ across the semicore,
# valence and low-conduction manifolds; QE ``vxc.dat``).  Anything outside
# this generous window means H₀ = ⟨T+V_ion+V_NL⟩ + ⟨V_H⟩ has lost its
# cancellation, NOT that the exchange-correlation physics changed.
_VXC_IMPLIED_MIN_EV = -50.0
_VXC_IMPLIED_MAX_EV = 2.0


def _warn_on_unphysical_h0(
    *,
    e_dft_ev: np.ndarray,
    kin_ion_diag_ev: np.ndarray,
    hartree_diag_ev: np.ndarray,
    kin_ion_has_hartree: bool = False,
    hartree_source: str | None = None,
    print_fn=print,
) -> np.ndarray:
    """Flag a corrupted mean-field H₀ before it reaches eqp{0,1}.dat.

    ``H₀ = kin_ion + V_H`` is a *catastrophic cancellation*: for a
    pseudopotential system with semicore states both terms run to several
    hundred eV of opposite sign and their sum is only tens of eV (MoS₂:
    ⟨T+V_ion+V_NL⟩ = −502 eV, ⟨V_H⟩ = +461 eV, H₀ = −42 eV).  ``kin_ion``
    comes from an exact plane-wave evaluation (``gw.kin_ion_io``) while
    ``V_H`` is an ISDF centroid quadrature (``cohsex_sigma``'s ``hartree``
    kernel), so *any* relative error in the ISDF pair-product
    representation lands on H₀ multiplied by ~500 eV.  A 10 % ISDF error
    — which an under-resolved centroid set will happily produce while
    every stage still reports "successful" — is a 50 eV error in every QP
    energy.

    The cheap, assumption-free detector is the implied exchange-correlation
    potential ``V_xc = E_DFT − H₀``: the DFT eigenvalue identity
    ``E_DFT = ⟨T+V_ion+V_NL⟩ + ⟨V_H⟩ + ⟨V_xc⟩`` is exact, so a converged
    run must reproduce a physical ``V_xc`` band-by-band.  Returns the
    implied ``V_xc`` (nk, nb) in eV so callers can log it.

    The identity is mode-agnostic on purpose: when ``kin_ion_has_hartree``
    the V_H term already lives inside ``kin_ion_diag_ev`` and
    ``hartree_diag_ev`` is zero, so the same sum and the same window
    still apply.  Only the diagnosis printed on failure differs — an
    exact-V_H run that trips this gate is NOT an ISDF convergence
    problem, and saying so would send the reader down the wrong path.

    **Called from exactly one place:** ``eqp_bgw.assemble_eqp``, on the
    arrays the V_H seam just resolved.  Both eqp entry points (this
    module's ``write_results`` and the post-hoc ``make_eqp_bgw`` CLI) go
    through that assembly, so this gate now covers both and fires once.
    """
    implied_vxc_ev = np.asarray(e_dft_ev, dtype=np.float64) - (
        np.asarray(kin_ion_diag_ev, dtype=np.float64)
        + np.asarray(hartree_diag_ev, dtype=np.float64)
    )
    if implied_vxc_ev.size == 0:
        return implied_vxc_ev
    lo, hi = float(implied_vxc_ev.min()), float(implied_vxc_ev.max())
    n_bad = int(
        np.count_nonzero(
            (implied_vxc_ev < _VXC_IMPLIED_MIN_EV)
            | (implied_vxc_ev > _VXC_IMPLIED_MAX_EV)
        )
    )
    _exact = bool(kin_ion_has_hartree) or (
        hartree_source in ("stored", "gspace", "folded"))
    _src = {
        "folded": "kin_ion[exact V_H folded in]",
        "stored": "kin_ion + V_H[exact, stored]",
        "gspace": "kin_ion + V_H[exact, on-the-fly]",
        "isdf":   "kin_ion + V_H[ISDF]",
    }.get(hartree_source,
          "kin_ion[exact V_H folded in]" if kin_ion_has_hartree
          else "kin_ion + V_H[ISDF]")
    print_fn(
        f"  H0 check: implied Vxc = E_DFT - ({_src}) in "
        f"[{lo:.3f}, {hi:.3f}] eV over {implied_vxc_ev.size} (k,n)"
    )
    if n_bad == 0:
        return implied_vxc_ev
    k_bad, n_band_bad = np.unravel_index(
        int(np.argmax(np.abs(implied_vxc_ev))), implied_vxc_ev.shape)
    if _exact:
        _diagnosis = (
            "H0 is fully exact here (T + V_ion + V_NL + V_H, all plane-wave / "
            "FFT-grid), so this is NOT an ISDF convergence problem and raising "
            "the centroid count will not help.  Look instead at whether "
            "kin_ion.h5 was generated from THIS run's input file: a Coulomb "
            "truncation mismatch (sys_dim), a wrong occupied-band count in the "
            "density, or a WFN/deck mismatch are the ways this branch fails."
        )
    else:
        _diagnosis = (
            "Most likely cause: the ISDF centroid basis is too small to resolve "
            "<nk|V_H|nk> (V_H is a centroid quadrature; kin_ion is exact), so "
            "the ~500 eV cancellation in H0 does not close.  The durable fix is "
            "to regenerate kin_ion.h5 with the exact V_H folded in "
            "(gw.kin_ion_io, default); the stopgap is more centroids."
        )
    # Emitted through ``common.sanity.warn`` so this failure carries the
    # ``*** LORRAX SANITY FAILURE`` grep token every other gate in the
    # tree uses, and so ``LORRAX_SANITY=strict`` stops the run here
    # rather than writing an untrustworthy eqp.  Before AD the token was
    # attached to the post-hoc CLI's own copy of this check; folding the
    # two gates into one (``eqp_bgw.assemble_eqp``) would otherwise have
    # dropped it from the CLI and never given it to the live driver.
    from common import sanity
    sanity.warn(
        f"H0 = {_src} is UNPHYSICAL — "
        f"{n_bad} of {implied_vxc_ev.size} (k,n) have an implied Vxc outside "
        f"[{_VXC_IMPLIED_MIN_EV:.0f}, {_VXC_IMPLIED_MAX_EV:.0f}] eV "
        f"(worst: k={int(k_bad)} n={int(n_band_bad)}, "
        f"Vxc={float(implied_vxc_ev[k_bad, n_band_bad]):.3f} eV).  "
        "eqp0/eqp1/eqp_g0w0 are NOT trustworthy.  Sigma may still be fine — "
        f"the mean-field side is what failed.  {_diagnosis}  "
        "Cross-check H0 against pw2bgw's kih.dat.",
        print_fn=print_fn,
    )
    return implied_vxc_ev


# ---------------------------------------------------------------------------
# Result writer  (QE ``punch('all')`` pattern)
# ---------------------------------------------------------------------------

def write_results(
    results: GWResults,
    sigma_diag_file: str,
    eqp0_file: str,
    eqp1_file: str,
    input_dir: str,
    kpoints_crys: np.ndarray,
    kgrid: tuple[int, int, int],
    kpoints_irr_frac: np.ndarray,
    kpoints_reduced: np.ndarray | None = None,
    kirr_to_kfull: np.ndarray | None = None,
    print_fn=print,
    *,
    eqp_dE_ev: float = 0.5,
):
    """Serialize all GW outputs — the unified ``punch('all')`` gateway.

    Writes (always):

    1. ``sigma_diag.dat``  — LORRAX per-(k,n) Σ-decomposition diagnostic.
    2. ``eqp0.dat``        — BGW-format zeroth-order QP energies.
    3. ``eqp1.dat``        — BGW-format Z-linearized QP energies (Z=1 in
       static COHSEX, BGW central-difference Z in PPM modes).
    4. ``qp_wfn_rotations.h5`` — QP eigenvectors for band-structure interp.

    Conditional (PPM, non-SC):

    5. ``eqp_g0w0.dat``    — explicit Re/Im of (H₀ + Σ_xc(E_DFT)) for
       hand-debugging convergence.

    BGW-style degenerate-set averaging is applied **upstream** at the
    H-build seam in :mod:`gw.gw_jax`, so the writer just serializes
    whatever ``GWResults`` carries — no re-averaging here.

    Parameters
    ----------
    results : GWResults
        Populated results container (self-energy arrays in Rydberg).
    sigma_diag_file, eqp0_file, eqp1_file : str
        Output paths for the three text files.
    input_dir : str
        Base directory for ancillary output files (eqp_g0w0.dat,
        qp_wfn_rotations.h5).
    kpoints_crys : np.ndarray, (nk_full, 3)
        Full-zone k-points in crystal coordinates (for qp_wfn_rotations).
    kgrid : (nkx, nky, nkz)
        k-mesh dimensions.
    kpoints_irr_frac : np.ndarray, (nk_irr, 3)
        IBZ-wedge k-points in fractional coords (used for BGW eqp{0,1}.dat).
    kpoints_reduced, kirr_to_kfull : optional
        Reduced-zone k-point mapping for restart metadata.
    eqp_dE_ev : float
        Central-difference spacing for the Z-factor in eqp1.dat.
    """
    from file_io import (
        write_sigma_to_file,
        write_eqp_g0w0,
        write_qp_rotations_h5,
    )
    from .eqp_bgw import assemble_eqp

    r2e = RYD_TO_EV

    # ── Per-k Σ-decomposition diagnostic (LORRAX-native) ──────────────────
    # ``sigma_diag.dat`` is a human-eyeball dump of the diagonal Σ pieces
    # per (k, n).  Column labels switch on mode: COHSEX prints
    # sigSX/sigCOH/sigTOT/VH; PPM prints sigX/sigC/sigXC/VH (same array
    # slots, relabelled).  The driver passes the right arrays for each mode.
    #
    # NOTE on the VH column when ``kin_ion_has_hartree``: it reads 0.000
    # by design.  V_H is no longer a self-energy channel there — it was
    # folded into kin_ion at generation time — so the ISDF quadrature is
    # suppressed upstream and this column truthfully reports "no Hartree
    # added here".  The mean-field V_H is not separately recoverable from
    # ``kin_ion.h5``; regenerate with ``--no-hartree`` if you need it split.
    if results.use_ppm:
        sx_arr = results.sig_x
        diag_ry = results.sigma_c_diag_at_dft_ry
        corr_arr = np.zeros_like(results.sig_coh)
        if diag_ry is not None:
            nb = diag_ry.shape[1]
            idx = np.arange(nb)
            corr_arr[:, idx, idx] = np.asarray(diag_ry)
    else:
        sx_arr = results.sig_sx
        corr_arr = results.sig_coh

    sx_out    = r2e * sx_arr
    corr_out  = r2e * corr_arr
    sig_h_out = r2e * results.sig_h
    sig_x_out = r2e * results.sig_x  # always populated; needed for eqp{0,1}

    write_sigma_to_file(
        sx_out,
        sigma_diag_file,
        sigma_coh_kij_eV=corr_out,
        hartree_kij_eV=sig_h_out,
        energies_dft_ev=r2e * np.asarray(results.E_dft_ry, dtype=np.float64),
        sx_label="sigX" if results.use_ppm else "sigSX",
        corr_label="sigC" if results.use_ppm else "sigCOH",
        total_label="sigXC" if results.use_ppm else "sigTOT",
    )

    # ── BGW-format eqp0.dat / eqp1.dat ────────────────────────────────────
    # Single source of truth for the linearization math: ``eqp_bgw``
    # provides the central-difference Z-factor and the Newton update.
    # Static modes (COHSEX) hand in ``sigma_c_omega_diag_ev=None`` ⇒ Z=1
    # ⇒ eqp1 == eqp0, matching BGW's behavior for static runs.
    #
    # All gw_jax internal arrays live on the unfolded full BZ (nk_full);
    # BGW's eqp{0,1}.dat lists only the IBZ wedge.  ``kirr_to_kfull[i]``
    # is the full-BZ index of IBZ point ``i``; index with it to subset.
    if kirr_to_kfull is None:
        raise ValueError(
            "write_results requires kirr_to_kfull for the BGW eqp{0,1}.dat "
            "writer — pass sym.kirr_fullids."
        )
    irr_idx = np.asarray(kirr_to_kfull, dtype=np.int64)

    e_dft_ev_full = np.asarray(results.E_dft_ry, dtype=np.float64) * r2e
    e_dft_ev_irr = e_dft_ev_full[irr_idx]
    kin_ion_diag_ev = (
        np.real(np.diagonal(results.kin_ion_ry[irr_idx], axis1=1, axis2=2)) * r2e
    )
    # ── H₀'s Hartree term ─────────────────────────────────────────────────
    # Handed to the assembly PRE-seam: ``eqp_bgw.resolve_hartree_diag_ev``
    # is the one place that decides whether this column is suppressed
    # (legacy folded kin_ion), substituted (a stored exact V_H the caller
    # supplies) or used as given (ISDF quadrature, or the exact matrix
    # ``sigma_dispatch`` already substituted into ``sig_h``).  The rule
    # used to be restated here AND in ``eqp_bgw.make_eqp_bgw``; it is now
    # written once, so the live driver and the post-hoc CLI cannot drift.
    # The mean-field (implied-V_xc) gate moved with it — ``assemble_eqp``
    # runs ``_warn_on_unphysical_h0`` on the resolved arrays, so a broken
    # H₀ still reports itself exactly once, with the same wording.
    hartree_diag_ev = np.real(
        np.diagonal(sig_h_out[irr_idx], axis1=1, axis2=2))
    # MODE-CORRECT exchange, not the bare one.  ``sx_out`` is already
    # resolved per mode ~50 lines up: results.sig_x under PPM (where Sigma =
    # Sigma_x + Sigma_c and bare X is right), results.sig_sx under static
    # COHSEX (where Sigma = Sigma_SX + Sigma_COH and there is no separate
    # bare term).  ``assemble_eqp`` forms sigma_x_diag_ev +
    # sigma_c_at_dft_diag_ev (eqp_bgw.py:378), so feeding it sig_x_out here
    # substituted BARE exchange for SCREENED in every static run and dropped
    # the SX-X piece: measured 6725.6 meV MAE against BGW Eqp0' on the Si
    # 4x4x4 anchor, where the correct assembly gives 142.2 meV.  Invisible to
    # the regression gate, which compares only the sigma_diag file -- and that
    # writer already uses sx_out, which is why the two disagreed by exactly
    # Sigma_SX (4.474 eV at Gamma band 1).
    sigma_x_diag_ev = np.real(np.diagonal(sx_out[irr_idx], axis1=1, axis2=2))
    # Σ_c at E_DFT diagonal: in PPM mode this is the interpolated value
    # the driver already computed; in static modes it is the static Σ_COH
    # diagonal (post-degen-averaging if enabled).
    if results.use_ppm and results.sigma_c_diag_at_dft_ry is not None:
        sigma_c_at_dft_diag_ev = (
            np.asarray(results.sigma_c_diag_at_dft_ry, dtype=np.complex128)[irr_idx] * r2e
        )
    else:
        sigma_c_at_dft_diag_ev = np.diagonal(corr_out[irr_idx], axis1=1, axis2=2)

    # E_DFT relative to mid-gap E_F (matches gw_jax convention).  Only
    # needed when there is a finite ω-grid to interpolate against.
    # ``results.efermi_ev`` is the canonical value computed once upstream
    # in ``ppm_pipeline._eval_sigma_c_at_dft_energies`` from the actual
    # number of occupied bands (``meta.nelec``); the writer used to
    # recompute it from ``band_stop - band_start`` which silently treats
    # every sigma-window band as occupied → efermi = top-of-window-band.
    sigma_c_omega_diag_ev_irr = None
    e_dft_rel_ev_irr = None
    if results.sigma_c_omega_diag_ev is not None and results.omega_rel_ev is not None:
        sigma_c_omega_diag_ev_irr = np.asarray(
            results.sigma_c_omega_diag_ev, dtype=np.complex128
        )[:, irr_idx, :]
        if results.efermi_ev is None:
            raise ValueError(
                "write_results: results.efermi_ev required for PPM Z-factor; "
                "fill GWResults.efermi_ev from ppm_outputs.efermi_dft_ev."
            )
        e_dft_rel_ev_irr = e_dft_ev_irr - float(results.efermi_ev)

    assemble_eqp(
        kpoints_irr_frac=np.asarray(kpoints_irr_frac, dtype=np.float64),
        band_offset=results.band_start,
        e_dft_ev=e_dft_ev_irr,
        kin_ion_diag_ev=kin_ion_diag_ev,
        hartree_diag_ev=hartree_diag_ev,
        sigma_x_diag_ev=sigma_x_diag_ev,
        sigma_c_at_dft_diag_ev=sigma_c_at_dft_diag_ev,
        sigma_c_omega_diag_ev=sigma_c_omega_diag_ev_irr,
        omega_rel_ev=results.omega_rel_ev,
        e_dft_rel_ev=e_dft_rel_ev_irr,
        dE_ev=eqp_dE_ev,
        nspin=1,
        hartree_source=results.hartree_source,
        kin_ion_has_hartree=results.kin_ion_has_hartree,
        print_fn=print_fn,
    ).write(eqp0_path=eqp0_file, eqp1_path=eqp1_file)

    # ── eqp_g0w0.dat (PPM non-SC only) — explicit Re/Im of H₀+Σ_xc(E_DFT) ──
    if (
        not results.self_consistent
        and results.sigma_xc_at_dft_ev is not None
    ):
        # ``sig_h`` is identically zero in exact-V_H mode (suppressed in
        # ``sigma_dispatch``), so this sum is the same H₀ the eqp writer
        # used in both modes.
        h0_diag = (
            np.real(
                np.diagonal(results.kin_ion_ry + results.sig_h, axis1=1, axis2=2)
            )
            * r2e
        )
        g0w0_path = os.path.join(input_dir, "eqp_g0w0.dat")
        write_eqp_g0w0(
            g0w0_path,
            results.E_dft_ry * r2e,
            h0_diag + results.sigma_xc_at_dft_ev,
        )
        print_fn(f"  G0W0 diag (E_DFT):     {g0w0_path}")

    # ── qp_wfn_rotations.h5 — QP eigenvectors ─────────────────────────────
    nkx, nky, nkz = kgrid
    write_qp_rotations_h5(
        os.path.join(input_dir, "qp_wfn_rotations.h5"),
        U_mnk=results.U_qp,
        E_qp_nk=results.E_qp_ry / 2.0,  # Ry → Hartree
        band_start=results.band_start,
        band_stop=results.band_stop,
        kpoints_crys=kpoints_crys,
        nkx=nkx, nky=nky, nkz=nkz,
        kpoints_reduced=kpoints_reduced,
        kirr_to_kfull=kirr_to_kfull,
    )

    # ── Status summary ────────────────────────────────────────────────────
    print_fn(f"\n  Sigma diag:   {sigma_diag_file}")
    print_fn(f"  BGW eqp0:     {eqp0_file}")
    print_fn(f"  BGW eqp1:     {eqp1_file}")
    if results.sigma_omega_h5_path:
        print_fn(f"  Sigma(ω):     {results.sigma_omega_h5_path}")
    if results.tensors_filename:
        print_fn(f"  Restart:      {results.tensors_filename}")
    print_fn("")
