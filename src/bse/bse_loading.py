"""BSE window, padding and head assembly on the shared bundle reader."""
from __future__ import annotations

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from runtime.padding import pad_axis, padded_mu_extent
from common.band_degeneracy import (DEFAULT_MODE, DEGENERACY_TOL_RY,
                                    resolve_band_window)

from .bse_densify import (_interpolate_bse_data_to_grid,
                          _read_lorrax_input_quietly, _resolve_kgrid_fine,
                          _resolve_kgrid_fine_shift,
                          resolve_w_head_densify)
from .bse_head import _inject_q0_head, _resolve_head_params
from .bse_preconditioner import compute_pair_amplitude
from .bse_window import (PAD_EPS_GUARD_RY, _log0, _parse_wfn_path, resolve_n_occ,
                         assert_bse_window_in_zeta_training)
from file_io.restart_bundle import (apply_eqp_corrections)
from common.log import get_logger

log = get_logger(__name__)


def no_w_operand(mesh_xy: Mesh) -> jax.Array:
    """The W slot of a kernel without the screened direct term: a (p_x, p_y, 1, 1, 1)
    zero tile at ``P('x','y',None,None,None)``.  No matvec reads it (``kernel='rpa'``
    traces no W term); it only fills the signature, so no W tile is resident."""
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    return jax.jit(lambda: jnp.zeros((px, py, 1, 1, 1), jnp.complex128),
                   out_shardings=NamedSharding(mesh_xy, P("x", "y", None, None, None)))()


def _pad_last_axis(x: jax.Array, target: int) -> jax.Array:
    pad = target - x.shape[-1]
    if pad <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, pad)
    return jnp.pad(x, pad_width, mode="constant")


def _pad_last_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[-2]
    pad1 = target - x.shape[-1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-2] = (0, max(0, pad0))
    pad_width[-1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")


def _pad_first_two_axes(x: jax.Array, target: int) -> jax.Array:
    pad0 = target - x.shape[0]
    pad1 = target - x.shape[1]
    if pad0 <= 0 and pad1 <= 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[0] = (0, max(0, pad0))
    pad_width[1] = (0, max(0, pad1))
    return jnp.pad(x, pad_width, mode="constant")
















# ---------------------------------------------------------------------------
# SlabIO transport for the sharded BSE loader
# ---------------------------------------------------------------------------
# The tile geometry below is DELIBERATELY the tile geometry the serial readers
# compute.  The memory contract is unchanged and is the point: per-rank tiles,
# nothing larger than one rank's tile materialised anywhere, no allgather (an
# allgather is a refusal, not a fallback — owner ruling 2026-08-05).  Only who
# moves the bytes changes.  The serial readers are memory-correct and ~17x
# slower (0.17 GiB/s at P=4, CLAIMS 76, against 2.919 GiB/s for the phdf5 tile
# path at 16 ranks, CLAIMS 69): serial POSIX h5py issues one rank's W_q tile as
# nq × μ/px short row-runs, one at a time, with no collective buffering.
#
# SlabIO needs the phdf5 FFI and refuses outright where it is unavailable —
# there is no router and no allgather tier left to hand back — so the serial
# readers stay reachable as the only fallback.














def load_bse_data_from_restart_sharded(
    restart_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    fermi_energy: float = 0.0,
    mesh_xy: Optional[Mesh] = None,
    pad_bands: bool = True,
    use_nohead: bool = False,
    *,
    input_file: Optional[str] = None,
    cell_volume: Optional[float] = None,
    n_occ: Optional[int] = None,
    inject_head: bool = True,
    inject_w_head: bool | None = None,
    load_v_full: bool = False,
    kgrid_fine=None,
    w_head_densify=None,
    w_head_gamma_cell: str = "fine",
    degeneracy_mode: str = DEFAULT_MODE,
    degeneracy_tol_ry: float = DEGENERACY_TOL_RY,
    distrib_la_batched_route: str | None = None,
    htransform_a_band: int | None = None,
    htransform_rank_record_fn=None,
    htransform_quality_record_fn=None,
    eqp_file: Optional[str] = None,
    qp_rotations_file: Optional[str] = None,
    velocity_file: Optional[str] = None,
    velocity_full_window: bool = False,
    velocity_pair_rule: str = "mask",
    velocity_pair_gap_ev: float = 0.05,
    velocity_position_fix: bool = False,
    q0_head: str = "rank1",
    need_W: bool = True,
    coupling_legs: bool = True,
) -> dict:
    """Load BSE tensors from canonical gw_jax restart state (psi_parent_y/enk_full).

    ``need_W=False`` (a kernel without the screened direct term: the RPA,
    ``local_fields``) never reads, rebuilds or densifies W: ``data["W_q"]`` is
    :func:`no_w_operand`, a (p_x, p_y, 1, 1, 1) zero tile in W's sharding that
    fills the matvec signature's slot, ``data["w_loaded"]`` is False, and the
    screened q = 0 head is not taken (``whead`` belongs to W).  A bundle with no
    W0 then loads too.

    ``coupling_legs=False`` (a TDA consumer) returns ``psi_c_Y`` and
    ``psi_v_X`` as ``None``: the TDA matvec reads ψ_c on 'x' and ψ_v on 'y'
    alone, so the two other layouts (each N_k·n·n_s·N_μ/√P per rank) are
    dropped once ``M`` is built.  The coupling block (full BSE) needs them.

    ``qp_rotations_file`` (``qp_rotations_file``) puts the window in the GW
    run's QP basis (``bse.bse_qp_rotation``): the window is resolved on the
    QP ladder, ψ is read for the whole rotation block and rotated, eq. (2),
    and ε are the matched QP energies.  ``data["qp_rotation"]`` carries the
    window's U columns for the dipoles; it is ``None`` on the DFT basis.

    ``kgrid_fine`` (``(nx,ny,nz)`` / ``"nx ny nz"`` / ``None``) turns on
    fine-grid densification: when set and DIFFERENT from the coarse restart
    grid, the ENTIRE bundle (ψ, QP ε, V_Q exchange, W direct) is interpolated
    onto that grid via :func:`_interpolate_bse_data_to_grid` BEFORE returning,
    so every downstream solver runs on the fine grid transparently.  When
    ``None`` the value is read from the cohsex.in ``kgrid_fine`` key (via
    ``input_file``); unset or == the coarse grid → the coarse bundle is
    returned byte-identically (fast path untouched).

    ``eqp_file`` (``eqp_corrections``) enters the htransform on the densified path, so
    the fine ε are quasiparticle; ``data["densified"]`` tells the caller
    whether it still owes the coarse QP re-slice
    (``bse_window.apply_eqp_and_reslice_bands``).

    ``velocity_file`` (``dipole_file``) is read only on the densified path: the
    coarse WFN's ``dipole.h5``, carried through the same htransform as ψ, and
    ``data["dipoles"]`` holds the window dipoles in the fine ψ gauge
    (``bse_densify.densified_dipoles``).  Off it, the driver slices the file
    itself (``absorption_common.window_dipoles``).  ``velocity_full_window``
    also keeps the velocity over every window pair and its own band
    energies (``data["dipoles"]["v_window"]``, ``["e_dft_window"]``), which
    the TD-aGW system builder (``tdagw.td_system``) needs; off by default.
    With it the window position ``["r_window"]`` comes too (in the fine
    states of a QP f(H) with ``eqp_file`` only under
    ``velocity_position_fix``), its same-occupation pairs by
    ``velocity_pair_rule`` / ``velocity_pair_gap_ev``
    (``bse_densify.densified_dipoles``' ``pair_rule`` / ``pair_gap_ev``).
    ``velocity_position_fix`` (``dipole_position_fix``, opt-in) applies
    ``densified_dipoles``' exact-position fix under ``eqp_file``; off, the
    dipoles are ``v/ΔE_RQ`` and a QP/DFT mixing receipt is logged.

    ``inject_head=False`` returns the head-LESS V_q0 / W_q bodies exactly as
    stored on disk (the rank-1 q=0 head from vhead/whead is NOT added). Used by
    body-vs-body diagnostics such as the W(0) resolvent cross-check, where both
    sides must be head-less (bse_w_exact ``--compare-w0``).
    ``inject_w_head`` independently selects the screened direct head; None
    inherits ``inject_head``. A density ladder uses False/True to omit the
    external macroscopic Hartree head while retaining its internal static
    direct head. Explicit True requires a resolved screened head and volume.

    ``load_v_full=True`` additionally reads the FULL exchange tensor
    ``V_qmunu`` at every q as ``data['V_q_full']`` (μ, ν, nkx, nky, nkz),
    P('x','y',None,None,None) — same layout as ``W_q``.  The finite-q W_q
    resolvent (``bse_w_exact --compare-wq``) picks the tile
    ``V_q_full[:, :, qx, qy, qz]`` (NO head at q≠0) as the screening V and as
    the comparison target ``W_q[...,q] - V_q_full[...,q]``.  Default False keeps
    the q=0 path byte-identical.
    """
    if mesh_xy is None:
        raise ValueError("mesh_xy is required for sharded load")

    from common.collectives import _require_addressable
    _require_addressable(mesh_xy, origin="BSE restart reader")
    from file_io.restart_bundle import (read_metadata, read_bse_payload,
                                        refuse_mpa_static_w)
    if need_W:
        refuse_mpa_static_w(restart_file)            # GATE bse_mpa_w0
    header = read_metadata(restart_file)
    enk_full = header["energies"]
    nkx, nky, nkz = (int(n) for n in header["grid"])
    n_rmu = header["centroid_count"]
    n_rmu_pad = padded_mu_extent(n_rmu, mesh_xy)
    # Without the direct term no W is read, and no screened head is taken.
    w0_ready = header["screened_ready"] and need_W
    screened_w = None
    built_head = None
    if need_W and not w0_ready:
        if header["head_vector"] is None:
            raise ValueError("GATE bse_static_w_head_vector: missing ISDF G=0 vector; regenerate the restart")
        from gw.static_screening import build_static_w_from_restart
        screened_w, built_head, cell_volume = build_static_w_from_restart(
            restart_file, input_file, mesh_xy, print_fn=_log0)
        w0_ready = True
    n_occ = resolve_n_occ(enk_full, n_occ=n_occ, input_file=input_file,
        fermi_energy=fermi_energy if fermi_energy != 0.0 else None)
    nb_total = header["logical_band_count"]
    n_val, n_cond = min(n_val, n_occ), min(n_cond, nb_total-n_occ)
    if n_val <= 0 or n_cond <= 0:
        raise ValueError("BSE window contains no valence or conduction states")
    # ── Is a coarse→fine densification pending?  Resolved before the window
    # (the QP-rotated basis has no densified route yet) and before the head
    # injection, because C1 (the default) hands the densifier the
    # head-EXCLUDED body and re-attaches the head per fine q afterwards, so on
    # a densifying run the rank-1 whead must NOT go on now.
    fine_grid = _resolve_kgrid_fine(kgrid_fine, input_file)
    fine_shift = _resolve_kgrid_fine_shift(input_file)
    if fine_grid is None and any(fine_shift):
        raise ValueError("kgrid_fine_shift shifts kgrid_fine, which the deck does not set")
    densify_pending = (fine_grid is not None
                       and (fine_grid != (nkx, nky, nkz) or any(fine_shift)))
    qp_rotation = None
    if qp_rotations_file is None:
        n_val, n_cond = resolve_band_window(enk_full, n_occ, n_val, n_cond,
            tol_ry=degeneracy_tol_ry, mode=degeneracy_mode,
            where="load_bse_data_from_restart_sharded", log=_log0)
        assert_bse_window_in_zeta_training(header, n_occ-n_val, n_occ+n_cond)
        read_val = np.arange(n_occ-n_val, n_occ)
        read_cond = np.arange(n_occ, n_occ+n_cond)
        eps_v = jnp.asarray(enk_full[:, read_val])
        eps_c = jnp.asarray(enk_full[:, read_cond])
    else:
        if densify_pending:
            raise ValueError(
                f"GATE bse_qp_rotations_k_grid: got qp_rotations_file with "
                f"kgrid_fine = {fine_grid} on a {(nkx, nky, nkz)} restart; "
                f"want the native grid.  why: the densifying htransform takes "
                f"QP energies only (eqp_corrections), so the fine ψ and velocity would "
                f"stay in the DFT basis.  fix: run the native grid, or "
                f"densify a restart built from WFN_qp.h5.")
        from .bse_qp_rotation import load_qp_rotation
        qp_rotation = load_qp_rotation(
            restart_file, input_file, qp_rotations_file, header=header,
            n_occ=n_occ, n_val=n_val, n_cond=n_cond,
            degeneracy_mode=degeneracy_mode,
            degeneracy_tol_ry=degeneracy_tol_ry, log=_log0)
        n_val, n_cond = qp_rotation["n_val"], qp_rotation["n_cond"]
        q0, q1 = qp_rotation["band_range"]
        read_val, read_cond = np.arange(q0, n_occ), np.arange(n_occ, q1)
        eps_v = jnp.asarray(qp_rotation["eps_v"])
        eps_c = jnp.asarray(qp_rotation["eps_c"])
    grid_x, grid_y = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    g0_X = g0_Y = None
    vhead_restart = header["bare_head"]
    whead_restart = header["screened_head"]
    if built_head is not None:
        vhead_restart = built_head.vc0
        whead_restart = np.asarray([built_head.wcoul0])
    if header["head_vector"] is not None:
        from common.collectives import device_put_process_local
        from runtime.padding import pad_to_axis, padded_mu_axis
        g0 = pad_to_axis(header["head_vector"], padded_mu_axis(n_rmu, mesh_xy), axis=-1)
        g0_X = device_put_process_local(g0, NamedSharding(mesh_xy, P("x")))
        g0_Y = device_put_process_local(g0, NamedSharding(mesh_xy, P("y")))
    # A pending densification replaces ψ and M whole (the fine htransform
    # builds both), so the coarse ψ is not read at all.
    psi_v_X, psi_c_X, V_q0, W_q, V_q_full = read_bse_payload(
        restart_file, input_file, mesh_xy, read_val, read_cond,
        nohead=use_nohead, full_exchange=load_v_full, screened_w=screened_w,
        screened=need_W, read_psi=not densify_pending)
    screened_w = None                # relaid out as W_q: one W layout survives
    if not need_W:
        W_q = no_w_operand(mesh_xy)
    if qp_rotation is not None:
        from .bse_qp_rotation import rotate_window_psi
        psi_v_X, psi_c_X = rotate_window_psi(
            psi_v_X, psi_c_X, qp_rotation, mesh_xy, log=_log0)

    if pad_bands:
        # ψ pad = 0 (bilinear ⇒ inert, and it is what decouples the pad
        # block); ε pad = signed sentinel (diagonal of a diagonalisation).
        # ``.padded`` by name: n_val_pad / n_cond_pad are the MESH-ROUNDED
        # extents (bse_ring_comm asserts ``n_cond_pad % px == 0``), never
        # the logical band counts; ε and ψ pad the band axis alike.
        _ev = pad_axis(eps_v, grid_y, axis=1, fill=-PAD_EPS_GUARD_RY)
        _ec = pad_axis(eps_c, grid_x, axis=1, fill=PAD_EPS_GUARD_RY)
        eps_v, n_val_pad = _ev.array, _ev.padded
        eps_c, n_cond_pad = _ec.array, _ec.padded
        if not densify_pending:
            psi_v_X = pad_axis(psi_v_X, grid_y, axis=1).array
            psi_c_X = pad_axis(psi_c_X, grid_x, axis=1).array
    else:
        n_val_pad = int(eps_v.shape[1])
        n_cond_pad = int(eps_c.shape[1])
    # Replicated on the mesh: a single-device ε meeting the sharded solver
    # operands would compile a reshard program per shape.
    eps_v, eps_c = (device_put_process_local(e, NamedSharding(mesh_xy, P()))
                    for e in (eps_v, eps_c))
    psi_v_Y = psi_c_Y = None
    if not densify_pending:
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v_X, NamedSharding(mesh_xy, P(None, None, None, "y")))
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c_X, NamedSharding(mesh_xy, P(None, None, None, "y")))

    resolve_w_head_densify(                  # refuses a retired mode by name
        w_head_densify, _read_lorrax_input_quietly(input_file))
    defer_whead = densify_pending
    head_channel = None
    q0_head_ry = 0.0
    apply_w_head = inject_head if inject_w_head is None else bool(inject_w_head)

    if inject_w_head is True and (not need_W or not w0_ready or g0_X is None):
        raise ValueError("GATE bse_screened_head: an explicitly requested screened head "
                         "requires screened W0 and its G=0 vector")
    if g0_X is not None and (inject_head or apply_w_head):
        vhead, whead, cell_volume, head_src = _resolve_head_params(
            input_file, vhead_restart, whead_restart, cell_volume)
        vhead = vhead if inject_head else None
        whead = whead if apply_w_head else None
        if inject_w_head is True and (whead is None or cell_volume is None):
            raise ValueError("GATE bse_screened_head: an explicitly requested screened head "
                             "requires whead and cell_volume (Bohr^3)")

        if cell_volume is not None and (vhead is not None or whead is not None):
            # whead goes on a SCREENED W or nowhere — same gate, same helper,
            # as the single-device loader.
            V_q0, W_q, q0_head_ry, head_str = _inject_q0_head(
                V_q0, W_q, g0_X, g0_Y, vhead, whead, cell_volume,
                w0_ready=w0_ready, nk=nkx * nky * nkz, q0_head=q0_head,
                defer_whead=defer_whead)
            log.info(f"BSE-sharded: q=0 head injected (rank-1, dual-sharded G0, "
                     f"V_cell={cell_volume:.2f}): {head_str} "
                     f"[source: {head_src}]")
            if defer_whead and w0_ready and whead is not None:
                head_channel = {
                    "whead": float(complex(whead[0]).real),
                    "cell_volume": float(cell_volume),
                    "gamma_cell": w_head_gamma_cell,
                    "static_sample": built_head,
                    "q0_head": q0_head,
                }
        else:
            # G0_mu_nu is present and inject_head is True, but the head cannot
            # be built.  The loader has no wfn/meta/sym/S_cart with which to
            # recompute <v>_mBZ, so the only honest move is to warn loudly and
            # name the fix; silence here leaves a head-LESS q=0 tile no trace.
            import warnings
            reasons = []
            if cell_volume is None:
                reasons.append("cell_volume unknown (WFN not passed)")
            if vhead is None and whead is None:
                reasons.append("vhead/whead both unresolved "
                               "(no cohsex.in override, no restart scalars)")
            msg = (
                "BSE q=0 head NOT injected though G0_mu_nu is present and "
                f"inject_head=True: {', '.join(reasons)}.  The q=0 exchange "
                "tile is HEAD-LESS (missing the rank-1 (vhead/V_cell)·conj(g0)g0 "
                "term) — exciton binding energies will be under-bound at the "
                "zone centre.  FIX: add ``vhead``/``whead_0freq`` to cohsex.in, "
                "or write ``vhead``/``whead`` datasets into the restart.  "
                "(Recompute-from-WFN is not available at the loader.)")
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            log.debug(f"BSE-sharded: [WARN] {msg}")   # warnings.warn above carries the warning

    # Exchange pair amplitude M(k,c,v,μ) = Σ_s conj(ψ_c) ψ_v, hoisted so the
    # per-iteration matvec receives it instead of rebuilding it from ψ.  ONE
    # copy in the transition layout (c on x, v on y, μ whole): 1/P per rank,
    # and both V-term legs (encode, decode) read it locally.  Was two copies
    # sharded on μ only (M_X μ on x, M_Y ν on y), 2·M/√P per rank.
    M = None if densify_pending else jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, psi_v_X),
        NamedSharding(mesh_xy, P(None, "x", "y", None)))

    data = {
        "psi_c_X": psi_c_X,
        "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X,
        "psi_v_Y": psi_v_Y,
        "M": M,
        "eps_c": eps_c,
        "eps_v": eps_v,
        "W_q": W_q,
        "V_q0": V_q0,
        "V_q_full": V_q_full,
        "g0_X": g0_X,
        "g0_Y": g0_Y,
        "nkx": nkx,
        "nky": nky,
        "nkz": nkz,
        "n_rmu": n_rmu,
        "n_rmu_pad": n_rmu_pad,
        "n_val": n_val,
        "n_cond": n_cond,
        "n_val_pad": n_val_pad,
        "n_cond_pad": n_cond_pad,
        "fermi_energy": fermi_energy,
        "qp_rotation": qp_rotation,
        # Where W's screened q=0 head lives (bse_head module docstring):
        # "rank1" on the W tile (default), or "diagonal" as q0_head_ry (Ry),
        # which operators read through bse_head.head_shifted_eps_c.  The
        # densifier replaces q0_head_ry on a "diagonal" bundle.
        "q0_head": q0_head,
        "q0_head_ry": q0_head_ry,
        "w_loaded": bool(need_W),
    }
    # The bundle holds the only references: the densifier pops the coarse W
    # and frees it once the fine W exists.
    W_q = V_q0 = V_q_full = None

    # ── kgrid_fine coarse→fine densification ─────────────────────────────
    # Unset or == the coarse grid → the coarse bundle above is returned
    # UNTOUCHED.  That is the on-grid byte-identity guarantee, and it is
    # structural rather than measured.
    if densify_pending:
        if input_file is None:
            raise ValueError(
                "kgrid_fine densification needs input_file (cohsex.in) to run "
                "the htransform ψ/ε and vq_interp V_Q interpolation.")
        data = _interpolate_bse_data_to_grid(
            data, fine_grid, restart_file, input_file, mesh_xy,
            head_channel=head_channel,
            distrib_la_batched_route=distrib_la_batched_route,
            htransform_a_band=htransform_a_band,
            htransform_rank_record_fn=htransform_rank_record_fn,
            htransform_quality_record_fn=htransform_quality_record_fn,
            eqp_file=None if eqp_file is None else os.path.abspath(eqp_file),
            velocity_file=(None if velocity_file is None
                           else os.path.abspath(velocity_file)),
            velocity_full_window=bool(velocity_full_window),
            velocity_pair_rule=str(velocity_pair_rule),
            velocity_position_fix=bool(velocity_position_fix),
            velocity_pair_gap_ev=float(velocity_pair_gap_ev),
            fine_shift=fine_shift)
    # A densified bundle's ε already carry ``eqp_file``; the drivers re-slice
    # the coarse restart's QP energies only when this is False.
    data["densified"] = bool(densify_pending)
    data.setdefault("k_shift", (0.0, 0.0, 0.0))
    if not coupling_legs:
        # The TDA matvec reads ψ_c on 'x' and ψ_v on 'y' only; the other two
        # layouts feed the coupling block alone, and M is already built.
        data["psi_c_Y"] = data["psi_v_X"] = None
    return data






def _load_ring_subset(restart_file, n_val, n_cond, px, py, eqp_file=None,
                      n_occ=None, input_file=None, degeneracy_mode=DEFAULT_MODE,
                      degeneracy_tol_ry=DEGENERACY_TOL_RY):
    """Adapt the shared BSE payload to the serial matvec's array names."""
    from common.collectives import resolve_mesh
    from .bse_window import apply_eqp_and_reslice_bands
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val, n_cond, mesh_xy=resolve_mesh(), input_file=input_file,
        n_occ=n_occ, degeneracy_mode=degeneracy_mode,
        degeneracy_tol_ry=degeneracy_tol_ry, eqp_file=eqp_file)
    if eqp_file is not None and not data["densified"]:
        data["eps_v"], data["eps_c"], _ = apply_eqp_and_reslice_bands(
            restart_file, eqp_file, input_file, data["n_val"], data["n_cond"],
            n_occ, px, py, degeneracy_mode=degeneracy_mode,
            degeneracy_tol_ry=degeneracy_tol_ry)
    data["psi_c"], data["psi_v"] = data["psi_c_X"], data["psi_v_X"]
    data["nk"] = data["nkx"] * data["nky"] * data["nkz"]
    shape = (1, data["n_cond_pad"], data["n_val_pad"], data["nk"])
    key = jax.random.PRNGKey(0)
    data["X"] = jax.random.normal(key, shape) + 1j*jax.random.normal(key, shape)
    return data
