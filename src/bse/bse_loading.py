"""BSE window, padding and head assembly on the shared bundle reader."""
from __future__ import annotations

import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime.padding import pad_axis, padded_mu_extent
from common.band_degeneracy import (DEFAULT_MODE, DEGENERACY_TOL_RY,
                                    resolve_band_window)

from .bse_densify import (_interpolate_bse_data_to_grid,
                          _read_lorrax_input_quietly, _resolve_bse_k_grid,
                          resolve_w_head_densify)
from .bse_head import _inject_q0_head, _resolve_head_params
from .bse_serial import compute_pair_amplitude
from .bse_window import (PAD_EPS_GUARD_RY, _log0, _parse_wfn_path, resolve_n_occ)
from file_io.restart_bundle import (apply_eqp_corrections)


def finite_occupation_rpa_data(orbitals, energies_ry, occupations, coulomb_q,
                               kminq_index, mesh_xy, *, kgrid,
                               trs_receipt, envelope_threshold=1e-5):
    """Bind live full-band centroid orbitals to the TRS-folded RPA operator.

    ``orbitals`` holds ``psi_X``/``psi_Y`` with shape ``(nk,nb,spin,mu)``
    on centroid-X/Y faces. The caller must authenticate physical TRS and
    complete retained band projectors. Y is defined as Theta applied to X;
    no stored-state gauge rewrite or full-wavefunction host gather is used.
    ``energies_ry`` and exact ``occupations`` are small host tables; ``kminq``
    is the caller's canonical periodic k−q map, with no additional phase.
    No restart, W construction, symmetry reconstruction or head injection is
    performed. ``coulomb_q`` is the existing head-policy-compatible body.

    For retained positive transitions, D=ec(k−q)−ev(k), F=fv(k)−fc(k−q).
    The TRS-folded negative sector has −D and the same sqrt(F). Padding and
    D<=0 pairs have exactly zero coupling. Envelope truncation is explicit;
    the surviving Fermi differences are not clipped to binary occupations.
    """
    from runtime.padding import padded_axis, pad_to_axis
    from common.collectives import device_put_process_local
    from .occupation_pairs import occupation_band_envelopes

    e = np.asarray(energies_ry, dtype=np.float64)
    f = np.asarray(occupations, dtype=np.float64)
    if (trs_receipt.get("status") != "PASS"
            or trs_receipt.get("scope") != "physical_trs_complete_pair_subspaces"
            or trs_receipt.get("band_count") != e.shape[1]):
        raise ValueError("GATE fd_pair_trs: require authenticated physical TRS and complete retained pair subspaces; energy symmetry alone is insufficient")
    selection = occupation_band_envelopes(e, f, threshold=envelope_threshold)
    hi, lo = selection["hole_stop"], selection["particle_start"]
    nk, nb = e.shape
    if int(np.prod(kgrid)) != nk:
        raise ValueError("GATE fd_pair_grid: kgrid and energy table disagree")
    idx_host = np.asarray(kminq_index, dtype=np.int32)
    if idx_host.shape != (nk,) or not np.array_equal(np.sort(idx_host), np.arange(nk)):
        raise ValueError("GATE fd_pair_kmap: canonical k-minus-q map must be a permutation")
    pair_spec = P("x", "y", None)
    c_axis = padded_axis(nb-lo, mesh_xy, name="FD particle bands", spec=pair_spec, axis=0)
    v_axis = padded_axis(hi, mesh_xy, name="FD hole bands", spec=pair_spec, axis=1)
    rep = NamedSharding(mesh_xy, P())
    idx = device_put_process_local(idx_host, rep)
    data = dict(nkx=int(kgrid[0]), nky=int(kgrid[1]), nkz=int(kgrid[2]),
                n_cond=nb-lo, n_val=hi, n_cond_pad=c_axis.carrier,
                n_val_pad=v_axis.carrier, V_q0=coulomb_q,
                fd_trs_authenticated=True, fd_trs_receipt=dict(trs_receipt),
                fd_band_selection=selection, fd_particle_axis=c_axis,
                fd_hole_axis=v_axis)
    for face, mesh_axis in (("X", "x"), ("Y", "y")):
        psi = orbitals[f"psi_{face}"]
        if psi.shape[:2] != (nk, nb):
            raise ValueError("GATE fd_pair_orbitals: require uncut common retained-band array")
        target = NamedSharding(mesh_xy, P(None, None, None, mesh_axis))
        # Existing BSE generator/snapshot encodes conj(M); both legs must
        # therefore be conjugated to reproduce the GW density at +q.
        pc = jnp.conj(jnp.take(psi[:, lo:], idx, axis=0))
        pv = jnp.conj(psi[:, :hi])
        data[f"psi_c_{face}"] = jax.lax.with_sharding_constraint(pad_to_axis(pc, c_axis, axis=1), target)
        data[f"psi_v_{face}"] = jax.lax.with_sharding_constraint(pad_to_axis(pv, v_axis, axis=1), target)
    data["eps_c"] = pad_to_axis(device_put_process_local(e[idx_host, lo:], rep), c_axis, axis=1, fill=PAD_EPS_GUARD_RY)
    data["eps_v"] = pad_to_axis(device_put_process_local(e[:, :hi], rep), v_axis, axis=1, fill=-PAD_EPS_GUARD_RY)
    fc = pad_to_axis(device_put_process_local(f[idx_host, lo:], rep), c_axis, axis=1)
    fv = pad_to_axis(device_put_process_local(f[:, :hi], rep), v_axis, axis=1)

    # Only one-particle tables are replicated. The pair products are created
    # inside the final sharded program, never as replicated host/device cubes.
    from functools import partial

    @partial(jax.jit, out_shardings=(NamedSharding(mesh_xy, pair_spec),
             NamedSharding(mesh_xy, P(None, None, "x", "y", None)), rep))
    def pair_tables(ec, ev, occupation_c, occupation_v):
        delta = (ec[:, :, None]-ev[:, None, :]).transpose(1, 2, 0)
        difference = (occupation_v[:, None, :]-occupation_c[:, :, None]).transpose(1, 2, 0)
        logical = ((jnp.arange(c_axis.carrier)[:, None] < c_axis.logical)
                   & (jnp.arange(v_axis.carrier)[None, :] < v_axis.logical))[:, :, None]
        active = logical & (delta > 0) & (difference > 0)
        weight = jnp.sqrt(jnp.where(active, difference, 0.))
        diagonal = jnp.where(active, delta, PAD_EPS_GUARD_RY)
        bad = jnp.any(logical & (delta > 0) & (difference < -1e-14))
        return weight, jnp.stack((diagonal, -diagonal))[:, None], bad

    weight, diagonal, bad = pair_tables(data["eps_c"], data["eps_v"], fc, fv)
    if bool(jax.device_get(bad)):
        raise ValueError("GATE fd_pair_passivity: occupations must decrease with positive transition energy")
    data["fd_sqrt_weight"] = weight
    data["fd_signed_diagonal"] = diagonal
    data["n_rmu"] = int(orbitals["psi_X"].shape[-1])
    data["n_rmu_pad"] = data["n_rmu"]
    return data


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
    load_v_full: bool = False,
    bse_k_grid=None,
    w_head_densify=None,
    w_head_gamma_cell: str = "fine",
    degeneracy_mode: str = DEFAULT_MODE,
    degeneracy_tol_ry: float = DEGENERACY_TOL_RY,
    distrib_la_batched_route: str | None = None,
    htransform_a_band: int | None = None,
    htransform_rank_record_fn=None,
    htransform_quality_record_fn=None,
) -> dict:
    """Load BSE tensors from canonical gw_jax restart state (psi_parent_y/enk_full).

    ``bse_k_grid`` (``(nx,ny,nz)`` / ``"nx ny nz"`` / ``None``) turns on
    fine-grid densification: when set and DIFFERENT from the coarse restart
    grid, the ENTIRE bundle (ψ, QP ε, V_Q exchange, W direct) is interpolated
    onto that grid via :func:`_interpolate_bse_data_to_grid` BEFORE returning,
    so every downstream solver runs on the fine grid transparently.  When
    ``None`` the value is read from the cohsex.in ``bse_k_grid`` key (via
    ``input_file``); unset or == the coarse grid → the coarse bundle is
    returned byte-identically (fast path untouched).

    ``inject_head=False`` returns the head-LESS V_q0 / W_q bodies exactly as
    stored on disk (the rank-1 q=0 head from vhead/whead is NOT added). Used by
    body-vs-body diagnostics such as the W(0) resolvent cross-check, where both
    sides must be head-less (bse_w_exact ``--compare-w0``).

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
    from file_io.restart_bundle import read_metadata, read_bse_payload
    header = read_metadata(restart_file)
    enk_full = header["energies"]
    nkx, nky, nkz = (int(n) for n in header["grid"])
    n_rmu = header["centroid_count"]
    n_rmu_pad = padded_mu_extent(n_rmu, mesh_xy)
    w0_ready = header["screened_ready"]
    n_occ = resolve_n_occ(enk_full, n_occ=n_occ, input_file=input_file,
        fermi_energy=fermi_energy if fermi_energy != 0.0 else None)
    nb_total = header["logical_band_count"]
    n_val, n_cond = min(n_val, n_occ), min(n_cond, nb_total-n_occ)
    if n_val <= 0 or n_cond <= 0:
        raise ValueError("BSE window contains no valence or conduction states")
    n_val, n_cond = resolve_band_window(enk_full, n_occ, n_val, n_cond,
        tol_ry=degeneracy_tol_ry, mode=degeneracy_mode,
        where="load_bse_data_from_restart_sharded", log=_log0)
    val_indices = np.arange(n_occ-n_val, n_occ)
    cond_indices = np.arange(n_occ, n_occ+n_cond)
    eps_v = jnp.asarray(enk_full[:, val_indices])
    eps_c = jnp.asarray(enk_full[:, cond_indices])
    grid_x, grid_y = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    g0_X = g0_Y = None
    vhead_restart = header["bare_head"]
    whead_restart = header["screened_head"]
    if header["head_vector"] is not None:
        from common.collectives import device_put_process_local
        from runtime.padding import pad_to_axis, padded_mu_axis
        g0 = pad_to_axis(header["head_vector"], padded_mu_axis(n_rmu, mesh_xy), axis=-1)
        g0_X = device_put_process_local(g0, NamedSharding(mesh_xy, P("x")))
        g0_Y = device_put_process_local(g0, NamedSharding(mesh_xy, P("y")))
    psi_v_X, psi_c_X, V_q0, W_q, V_q_full = read_bse_payload(
        restart_file, input_file, mesh_xy, val_indices, cond_indices,
        nohead=use_nohead, full_exchange=load_v_full)

    if pad_bands:
        # ψ pad = 0 (bilinear ⇒ inert, and it is what decouples the pad
        # block); ε pad = signed sentinel (diagonal of a diagonalisation).
        _v = pad_axis(psi_v_X, grid_y, axis=1)
        _c = pad_axis(psi_c_X, grid_x, axis=1)
        # ``.padded`` by name: n_val_pad / n_cond_pad are the MESH-ROUNDED
        # extents (bse_ring_comm asserts ``n_cond_pad % px == 0``), never
        # the logical band counts.
        psi_v_X, n_val_pad = _v.array, _v.padded
        psi_c_X, n_cond_pad = _c.array, _c.padded
        eps_v = pad_axis(eps_v, grid_y, axis=1, fill=-PAD_EPS_GUARD_RY).array
        eps_c = pad_axis(eps_c, grid_x, axis=1, fill=PAD_EPS_GUARD_RY).array
    else:
        n_val_pad = int(psi_v_X.shape[1])
        n_cond_pad = int(psi_c_X.shape[1])
    psi_v_Y = jax.lax.with_sharding_constraint(psi_v_X, NamedSharding(mesh_xy, P(None, None, None, "y")))
    psi_c_Y = jax.lax.with_sharding_constraint(psi_c_X, NamedSharding(mesh_xy, P(None, None, None, "y")))

    # ── Is a coarse→fine densification pending?  Resolved HERE, before the
    # head injection, because C1 (the default) hands the densifier the
    # head-EXCLUDED body and re-attaches the head per fine q afterwards, so on
    # a densifying run the rank-1 whead must NOT go on now.
    fine_grid = _resolve_bse_k_grid(bse_k_grid, input_file)
    densify_pending = (fine_grid is not None
                       and fine_grid != (nkx, nky, nkz))
    w_head_mode = resolve_w_head_densify(
        w_head_densify, _read_lorrax_input_quietly(input_file))
    defer_whead = densify_pending and w_head_mode == "c1"
    head_channel = None

    if g0_X is not None and inject_head:
        vhead, whead, cell_volume, head_src = _resolve_head_params(
            input_file, vhead_restart, whead_restart, cell_volume)

        if cell_volume is not None and (vhead is not None or whead is not None):
            # whead goes on a SCREENED W or nowhere — same gate, same helper,
            # as the single-device loader.
            V_q0, W_q, head_str = _inject_q0_head(
                V_q0, W_q, g0_X, g0_Y, vhead, whead, cell_volume,
                w0_ready=w0_ready, defer_whead=defer_whead)
            print(f"BSE-sharded: q=0 head injected (rank-1, dual-sharded G0, "
                  f"V_cell={cell_volume:.2f}): {head_str} "
                  f"[source: {head_src}]")
            if defer_whead and w0_ready and whead is not None:
                head_channel = {
                    "whead": float(complex(whead[0]).real),
                    "cell_volume": float(cell_volume),
                    "gamma_cell": w_head_gamma_cell,
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
            print(f"BSE-sharded: [WARN] {msg}")

    # Exchange pair amplitudes M(k,c,v,μ) = Σ_s conj(ψ_c) ψ_v, hoisted so the
    # per-iteration matvec receives them instead of rebuilding them from ψ:
    # the V-term decode (M_X, μ on x) and encode (M_Y, ν on y) vertices.
    # Peak-neutral; the between-matvec floor rises by ~2·M/p.
    M_X = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, psi_v_X),
        NamedSharding(mesh_xy, P(None, None, None, "x")))
    M_Y = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_Y, psi_v_Y),
        NamedSharding(mesh_xy, P(None, None, None, "y")))

    data = {
        "psi_c_X": psi_c_X,
        "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X,
        "psi_v_Y": psi_v_Y,
        "M_X": M_X,
        "M_Y": M_Y,
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
    }

    # ── bse_k_grid coarse→fine densification ─────────────────────────────
    # Unset or == the coarse grid → the coarse bundle above is returned
    # UNTOUCHED.  That is the on-grid byte-identity guarantee, and it is
    # structural rather than measured.
    if densify_pending:
        if input_file is None:
            raise ValueError(
                "bse_k_grid densification needs input_file (cohsex.in) to run "
                "the htransform ψ/ε and vq_interp V_Q interpolation.")
        if w_head_mode == "legacy":
            print("BSE-sharded: [WARN] w_head_densify = legacy — W's Γ head "
                  "rides through the trigonometric interpolant as a Kronecker "
                  "delta.  That is the documented defect (gw.head_densify): "
                  "the interpolant of a delta is a Dirichlet kernel, so a "
                  "fraction of the head's ~10^3 meV prefactor is deposited at "
                  "fine q that should carry none of it, and the 1/q² rise "
                  "inside the coarse Γ cell is missing entirely.  This arm "
                  "exists to price the repair, not to be run for physics.")
        data = _interpolate_bse_data_to_grid(
            data, fine_grid, restart_file, input_file, mesh_xy,
            head_channel=head_channel,
            distrib_la_batched_route=distrib_la_batched_route,
            htransform_a_band=htransform_a_band,
            htransform_rank_record_fn=htransform_rank_record_fn,
            htransform_quality_record_fn=htransform_quality_record_fn)
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
        degeneracy_tol_ry=degeneracy_tol_ry)
    if eqp_file is not None:
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
