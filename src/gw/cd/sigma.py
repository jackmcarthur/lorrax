"""Sigma_c(omega) by contour deformation on the exact ISDF W(z): the production CD route.

Where this sits (cd_production_design.md section 2): ``gw.sigma_dispatch._compute_mpa_sigma``
calls this instead of ``gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`` when the deck says
``sigma_frequency_integration = contour_deformation``; the result is the same
:class:`gw.ppm_sigma.SigmaOmegaResult` the one finalizer consumes.  Nothing of the physics is
rebuilt here: chi0 is ``gw.w_isdf.compute_chi0_direct_fractional(ordered=True)`` (exact ordered
Lehmann sum at every node and knot), the Dyson pair is the response bank's equation set
(``gw.response_bank._response_programs``: W_c = H X (1-X)^-1 H, dW_c/ds = W (pref dchi) W, with
the Coulomb roots H of ``_coulomb_algebra``), the q-wedge unfold is ``symmetry_maps``
(``QirrOperator.unfold`` with the general-operator ``pair_transpose`` rule), the k unfold of the
cube is ``symmetry_maps.unfold_file_wedge_band_operator``, and the contour laws are
``gw.cd.laws`` (ordered_cd.py of the Fe harness, one layout).

Streaming (section 3): W(z) is never persisted.  Nodes and knots are streamed in chunks:

    chi(z) [nz, nq_irr, n, n]  --Dyson (local batched LU over (z, q))-->  W_c, dW_c/ds
      --unfold per z--> full zone [nz, nq, n, n], q-sharded over every rank
      --shard_map over q (scan over the rank's q, scan over z)--> accumulators [E, kw, m, n]

The pair densities rho_q[kw, m, l, mu] are recomputed per q (never stored over q), the node
kernels are closed forms of x = omega_E - eps_l(k - q) evaluated in-kernel, and the chunk
pipeline donates chi -> W -> contraction.  Sharding is by mesh-axis name: the q stack is
``P(None, ('x', 'y'))``; chi comes off the Lehmann producer on the face ``P(None, None, 'x', 'y')``
and is moved to the q-parallel layout one mesh axis at a time (w_isdf.py:1788-1800 pitfall).

Units: Ry throughout (omega, eta, energies, Sigma).  Occupations: the fixed-N
``OccupationState`` of a metal, or the loaded step table of an insulator.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from gw.cd import laws
from gw.cd.rule import tan_mapped_gauss_rule

#: Ghost knots per chunk side of the 9-point central d/dx (dchi/ds = (dchi/dx) / (2z)).
GH = 4
FD9 = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
#: Occupation floor (fe_cd): f below it is 0, 1 - f below it is 1 (moves f by <= 1e-14).
OCCUPATION_FLOOR = 1e-14
CD_RECIPE = dict(version="cd_production_v1", laws="ordered_cd TRREF / Run211 jets",
                 beta_ev=[0.5, 4.0], slope="fd9", imaginary_rule="tan_mapped_gauss two-panel",
                 chi0="w_isdf.compute_chi0_direct_fractional(ordered=True)",
                 dyson="response_bank pair equations, local batched jnp.linalg.solve")


def _say(print_fn, started, msg):
    print_fn(f"  CD +{time.monotonic() - started:7.1f}s {msg}")


def _ledger_row(stage, chunk):
    """One device-ledger row: bytes_in_use and peak on this process's first device."""
    try:
        st = jax.local_devices()[0].memory_stats() or {}
        return dict(stage=stage, chunk=int(chunk), bytes_in_use=int(st.get("bytes_in_use", -1)),
                    peak_bytes_in_use=int(st.get("peak_bytes_in_use", -1)))
    except Exception:  # pragma: no cover - CPU backends have no stats
        return dict(stage=stage, chunk=int(chunk), bytes_in_use=-1, peak_bytes_in_use=-1)


def compute_sigma_c_cd_omega_grid(
    wfns, cd_handle, meta, mesh_xy, *, sym, config, omega_grid_ry, efermi_ry,
    regularization_width_ry, occupation_state=None, print_fn=print, run_dir=None,
    sig_x=None, band_brackets=None, band_counts=None, omega_eta_ry=None, **_body_options):
    """Sigma_c(omega) [n_omega, nk_full, nb_c, nb_c] (Ry) on the deck's omega grid by CD.

    Parameters (the ones the dispatch seam passes; the rest of ``body_options`` is the pole
    executor's and is ignored by name): ``cd_handle`` is ``W_by_role["cd_stream"]`` from
    ``gw.screening.compute_screening_model`` (``V_q`` wedge operator, deck sub-keys);
    ``omega_grid_ry`` is relative to ``efermi_ry``; ``regularization_width_ry`` is eta.
    Returns :class:`gw.ppm_sigma.SigmaOmegaResult` with the cube at ``P(None, None, 'x', 'y')``
    on the padded Sigma band carrier and ``efermi_ry`` set.  Writes ``cd_sigma_receipt.json``,
    ``cd_sigma_own_energy.npz`` (diagonal at the DFT energies, the verification observable) and
    ``cd_leak_ledger.csv`` beside the deck on rank 0.
    """
    from common.collectives import barrier, gather_to_host, replicate_to_mesh
    from gw import response_bank as rb
    from gw.ppm_sigma import SigmaOmegaResult, sigma_band_axis
    from gw.w_isdf import compute_chi0_direct_fractional, _w_solve_pref_scalar
    from symmetry_maps import kgrid_shift_map, q_negation_index, star_tables_of
    from symmetry_maps.maps import unfold_wavefunction_local, unfold_file_wedge_band_operator

    started = time.monotonic()
    EV = float(RYD_TO_EV)
    if band_brackets is not None or band_counts is not None:
        raise NotImplementedError(
            "GATE cd_band_extrapolation_unimplemented: use_band_extrapolation got: true; "
            "want: false with sigma_frequency_integration = contour_deformation; why: the CD "
            "accumulator does not yet partition its l-sum into the bracket counts (IMPLEMENTATION LIMIT)")
    if omega_eta_ry is not None:
        raise ValueError("GATE cd_single_eta: sigma_omega_patches_ev names a per-window eta; want: one eta "
                         "(sigma_regularization_ev); why: a second eta is a second W knot set")
    eta = float(regularization_width_ry)
    h_ry = float(cd_handle["knot_step_ev"]) / EV
    n_imag = int(cd_handle["imaginary_nodes"])
    rank0 = jax.process_index() == 0
    rep = NamedSharding(mesh_xy, P())
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    face4 = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    qpar3 = NamedSharding(mesh_xy, P(("x", "y"), None, None))
    qpar4 = NamedSharding(mesh_xy, P(None, ("x", "y"), None, None))
    mid3 = NamedSharding(mesh_xy, P("x", None, "y"))
    ndev = jax.device_count()
    ledger = []

    def host(a):
        return np.asarray(gather_to_host(a)) if isinstance(a, jax.Array) else np.asarray(a)

    # ---- geometry: k grid, k-q maps, q wedge, file k wedge --------------------------------
    grid = tuple(int(v) for v in meta.kgrid)
    nk = int(np.prod(grid))
    n = int(meta.mu_basis.n_packed)
    q_irr = np.asarray(sym.q_irr_full_idx, dtype=np.int64)
    nq_irr = int(q_irr.size)
    qneg = np.asarray(q_negation_index(grid), dtype=np.int64)

    def kmq(q):
        off = -np.asarray(np.unravel_index(int(q), grid), np.int32)
        return np.asarray(kgrid_shift_map(*grid, off)[0], np.int64)
    KMQ = np.stack([kmq(q) for q in range(nk)])                 # [nq_full, nk]: index of k - q
    kq_sym = np.asarray(sym.kq_map, dtype=np.int64)             # [nk_full, nq_red]
    if kq_sym.shape == (nk, nk) and int(np.sum(KMQ != kq_sym.T)):
        raise ValueError("GATE cd_kq_map: kgrid_shift_map vs SymMaps.kq_map differ")
    irr_k, sidx_k, _nsp = star_tables_of(sym)
    nk_red = int(sym.nk_red)
    kw_rows = np.asarray([int(np.flatnonzero((irr_k == i) & (sidx_k == 0))[0]) for i in range(nk_red)])
    full_bz = nk_red == nk and np.array_equal(kw_rows, np.arange(nk))
    if nk % ndev:
        raise ValueError(f"GATE cd_q_sharding: nq_full={nk} is not divisible by the device count {ndev}")

    # ---- occupations and energies ---------------------------------------------------------
    energy = np.asarray(host(wfns.enk), dtype=np.float64)       # [nk, nb_full]
    nb_full = int(energy.shape[1])
    if occupation_state is not None:
        f_state = np.asarray(occupation_state.f_kn, dtype=np.float64)
        mu = float(occupation_state.mu_ry)
        occ_for_chi = occupation_state
        if f_state.shape != tuple(wfns.occ.shape):
            raise ValueError(f"GATE cd_occupations_shape: OccupationState {f_state.shape} vs carrier {tuple(wfns.occ.shape)}")
        wfns = dataclasses.replace(wfns, occ=replicate_to_mesh(f_state, mesh_xy))
    else:
        f_state = np.asarray(host(wfns.occ), dtype=np.float64)
        mu = float(efermi_ry)
        occ_for_chi = SimpleNamespace(f_kn=f_state, mu_ry=mu, smearing_family="step")
    if f_state.shape[1] < nb_full:
        f_state = np.pad(f_state, ((0, 0), (0, nb_full - f_state.shape[1])))
    f_cd = np.where(f_state < OCCUPATION_FLOOR, 0.0, np.where(1.0 - f_state < OCCUPATION_FLOOR, 1.0, f_state))
    slices = wfns.slices
    sig = slices.sigma
    lsum = slices.sigma_sum if slices.sigma_sum.stop else slices.full
    m0, m1 = int(sig.start or 0), int(sig.stop)
    l0, l1 = int(lsum.start or 0), int(lsum.stop)
    nb_s, L = m1 - m0, l1 - l0
    nb_log = int(meta.b_id_4_chi_user) - int(slices.b0)
    pref = float(_w_solve_pref_scalar(meta))
    omega = np.asarray(omega_grid_ry, dtype=np.float64) + float(efermi_ry)   # absolute Ry
    nE = int(omega.size)
    enk_L = energy[:, l0:l1]
    f_L = f_cd[:, l0:l1]
    eps_own = energy[kw_rows, m0:m1]                               # [kw, nb_s] own energies

    # residue coverage (GATE cd_residue_coverage): the largest |x| any live residue reads
    E_all = np.concatenate([omega, eps_own.ravel()])
    need_neg = np.where(f_L > 0, np.maximum(0.0, enk_L - E_all.min()), 0.0).max()
    need_pos = np.where(1.0 - f_L > 0, np.maximum(0.0, E_all.max() - enk_L), 0.0).max()
    need = float(max(need_neg, need_pos))
    reach = cd_handle.get("residue_reach_ev", "auto")
    if reach != "auto" and float(reach) / EV < need:
        raise ValueError(f"GATE cd_residue_coverage: cd_residue_reach_ev got: {reach}; want: >= {need * EV:.3f} eV "
                         "(largest live |omega - eps_l|); why: no clipping, no omitted tail")
    reach_ry = need if reach == "auto" else float(reach) / EV
    J = int(np.ceil(reach_ry / h_ry)) + 2
    real_ry = h_ry * np.arange(J)
    u_ev, w_ev = tan_mapped_gauss_rule(n_imag, eta * EV)
    u_ry, w_ry = u_ev / EV, w_ev / EV
    _say(print_fn, started, f"geometry nk={nk} nq_irr={nq_irr} nk_red={nk_red} n_packed={n} window m=[{m0},{m1}) l=[{l0},{l1}) "
         f"nE={nE} eta={eta * EV:.4f} eV h={h_ry * EV:.4f} eV J={J} knots to {real_ry[-1] * EV:.3f} eV "
         f"(need {need * EV:.3f} eV) imag nodes={n_imag} mu={mu * EV:.6f} eV full_bz={full_bz}")

    # ---- wavefunctions at every full k (parents unfolded by the typed centroid action) -----
    carrier = wfns.green_parent
    if wfns.psi_nmu is not None and (carrier is None or int(wfns.psi_nmu.shape[0]) == nk):
        psi_full = jax.jit(lambda a: a * 1.0, out_shardings=rep)(wfns.psi_nmu)      # [nk, nb, s, n]
    else:
        plan = carrier.plan
        psi_par = jax.jit(lambda a: a * 1.0, out_shardings=rep)(carrier.psi_nmu)   # [n_parent, nb, s, n]
        if int(plan.n_parent) == nk:
            psi_full = psi_par
        else:
            psi_full = jax.jit(lambda a: unfold_wavefunction_local(
                a, irr_idx=np.asarray(plan.irr_idx), sym_idx=np.asarray(plan.sym_idx),
                k_irr_frac=np.asarray(plan.k_parent_frac), local_perm=np.asarray(plan.sym_perm),
                L_table=np.asarray(plan.L_table), spin_action_full=np.asarray(plan.spin_action_full),
                n_sym_spatial=int(plan.n_sym_spatial), spin_axis=2, mu_axis=3), out_shardings=rep)(psi_par)
    if tuple(int(v) for v in psi_full.shape[:2]) != (nk, nb_full) or int(psi_full.shape[-1]) != n:
        raise ValueError(f"GATE cd_psi_shape: {psi_full.shape} vs (nk={nk}, nb_full={nb_full}, s, n={n})")
    psi_L = jax.jit(lambda a: a[:, l0:l1] * 1.0, out_shardings=rep)(psi_full)               # [nk, L, s, n]
    psi_M = jax.jit(lambda a, r: a[r, m0:m1] * 1.0, out_shardings=rep)(psi_full, jnp.asarray(kw_rows))  # [kw, nb_s, s, n]
    del psi_full
    def _rep(a):
        return jax.device_put(np.asarray(a), rep)
    tabs = dict(psi_L=psi_L, psi_M=psi_M, KMQ=_rep(KMQ.astype(np.int32)), kw=_rep(kw_rows.astype(np.int32)),
                enk=_rep(enk_L), f=_rep(f_L), omega=_rep(omega), eps_own=_rep(eps_own),
                u=_rep(u_ry), w=_rep(w_ry))
    u_dev, w_dev = tabs["u"], tabs["w"]
    beta = tuple(float(b) for b in laws.BETA_RY)

    # ---- Coulomb: wedge operator, full zone, roots -----------------------------------------
    V_q = cd_handle["V_q"]
    if hasattr(V_q, "unfold"):
        v_full = V_q.unfold(mesh_xy)
        v_irr = V_q.values
        if not np.array_equal(np.asarray(V_q.full_rows), q_irr):
            raise ValueError(f"GATE cd_vq_wedge: V_q.full_rows {np.asarray(V_q.full_rows).tolist()} differ from "
                             f"sym.q_irr_full_idx {q_irr.tolist()}")
        unfold_w = None if V_q.is_whole_zone() else (
            lambda w_irr: dataclasses.replace(V_q, values=w_irr, trs_rule="pair_transpose").unfold(mesh_xy))
    else:
        v_full, v_irr, unfold_w = V_q, V_q, None
        if nq_irr != nk:
            raise ValueError("GATE cd_vq_wedge: V_q carries no unfold but the q wedge is reduced")
    pack = jax.jit(lambda v: meta.mu_basis.pack_operator(v, spec=P(None, "x", "y")), out_shardings=face)
    if int(v_full.shape[-1]) != n:
        v_full = pack(v_full)
    if int(v_irr.shape[-1]) != n:
        v_irr = pack(v_irr)
    sqrt_v = rb._coulomb_algebra(mesh_xy, n, int(meta.mu_basis.n_logical), "local")
    h_irr, _, negative, support = sqrt_v(v_irr)
    if bool(np.any(host(negative))):
        raise ValueError("GATE cd_coulomb_psd: negative eigenvalue in V_q")
    h_rep = jax.jit(lambda a: a * 1.0, out_shardings=rep)(h_irr)                      # [nq_irr, n, n]
    eye = jnp.eye(n, dtype=jnp.complex128)

    def _pair_one(h, chi_raw, dchi_raw):
        """response_bank._response_programs: wc and derivative, one (z, q)."""
        x = pref * (h @ chi_raw @ h)
        e = jnp.linalg.solve(eye - x, eye)
        wc = h @ (x @ e) @ h
        w = wc + h @ h
        return wc, w @ (pref * dchi_raw) @ w

    def _value_one(h, chi_raw):
        x = pref * (h @ chi_raw @ h)
        e = jnp.linalg.solve(eye - x, eye)
        return h @ (x @ e) @ h

    zq_par = NamedSharding(mesh_xy, P(("x", "y"), None, None))

    @jax.jit
    def _to_zq(chi4):
        """[nz, nq_irr, n, n] face -> [nz*nq_irr, n, n] on the (z, q)-parallel layout, one axis at a time."""
        nz_ = chi4.shape[0]
        flat = chi4.reshape((nz_ * nq_irr, n, n))
        flat = jax.lax.with_sharding_constraint(flat, mid3)
        return jax.lax.with_sharding_constraint(flat, zq_par)

    roots_zq = jax.jit(lambda h, qidx: jnp.take(h, qidx, axis=0), out_shardings=zq_par)
    _dyson_pair = jax.jit(lambda hs, chi_flat, dchi_flat: jax.vmap(_pair_one)(hs, chi_flat, dchi_flat),
                          out_shardings=(zq_par, zq_par))
    _dyson_value = jax.jit(lambda hs, chi_flat: jax.vmap(_value_one)(hs, chi_flat), out_shardings=zq_par)

    to_face_z = jax.jit(lambda a: a * 1.0, out_shardings=face)
    to_qpar = jax.jit(lambda a: a * 1.0, out_shardings=qpar3)
    stack_q = jax.jit(lambda *a: jnp.stack(a), out_shardings=qpar4)
    partner = jax.jit(lambda w: jnp.take(w, jnp.asarray(qneg), axis=1), out_shardings=qpar4)

    def _full_zone(flat, nz_):
        """[nz*nq_irr, n, n] (z,q)-parallel -> [nz, nq_full, n, n] q-parallel (unfold per z)."""
        per_z = flat.reshape((nz_, nq_irr, n, n))
        outs = []
        for iz in range(nz_):
            w_irr = to_face_z(per_z[iz])
            w_full = w_irr if unfold_w is None else unfold_w(w_irr)
            outs.append(to_qpar(w_full))
        return stack_q(*outs)

    def direct_chi(z_ry):
        chi = compute_chi0_direct_fractional(
            wfns, np.asarray(z_ry, np.complex128), meta, mesh_xy, occupation_state=occ_for_chi,
            kminq_rows=KMQ[q_irr], nb_logical=nb_log, ordered=True)
        if len(z_ry) == 1:
            chi = chi[None]
        return jax.jit(lambda a: a * 1.0, out_shardings=face4)(chi)

    def _slope(chi_e, z_core):
        """FD-9 d/dx on the knot grid -> dchi/ds at the core knots: [nz, nq_irr, n, n]."""
        nz_ = chi_e.shape[0] - 2 * GH
        dz = sum(FD9[m + 4] * chi_e[GH + m: GH + m + nz_] for m in (-4, -3, -2, -1, 1, 2, 3, 4)) / h_ry
        return dz / (2 * z_core)[:, None, None, None]
    slope_fn = jax.jit(_slope, out_shardings=face4)
    core_fn = jax.jit(lambda a: a[GH:a.shape[0] - GH] * 1.0, out_shardings=face4)

    # ---- the contraction kernels: shard_map over the q axis, scan over this rank's q --------
    nq_loc = nk // ndev
    acc_shape = (nE, nk_red, nb_s, nb_s)
    own_shape = (nk_red, nb_s)

    def _rho(T, q):
        rows = T["KMQ"][q][T["kw"]]                                           # [kw]
        psi_l = T["psi_L"][rows]                                              # [kw, L, s, n]
        rho = jnp.einsum("klsu,kmsu->kmlu", jnp.conj(psi_l), T["psi_M"])      # [kw, nb_s, L, n]
        x = T["omega"][None, :, None] - T["enk"][rows][:, None, :]            # [kw, E, L]
        x_own = T["eps_own"][:, :, None] - T["enk"][rows][:, None, :]         # [kw, nb_s, L]
        return rho, x, x_own, T["f"][rows]

    def _contract(rho, op):
        t = jnp.einsum("kmlu,uv->kmlv", jnp.conj(rho), op)
        return -jnp.einsum("kmlv,knlv->kmnl", t, rho) / nk

    def _own(a, w_own):
        """Diagonal at the own energy: out[k, m] = sum_l a[k, m, m, l] w_own[k, m, l]."""
        diag = jnp.einsum("kmml->kml", a)
        return jnp.einsum("kml,kml->km", diag, w_own)

    def _rank_q0():
        return jax.lax.axis_index("x") * mesh_xy.shape["y"] + jax.lax.axis_index("y")

    def _imag_body(W, u_c, w_c, acc, acc_own, T):
        """W [nz, nq_loc, n, n] local; u_c, w_c [nz]; T the replicated tables."""
        q0 = _rank_q0() * nq_loc

        def over_q(carry, iq):
            acc_, own_ = carry
            rho, x, x_own, f = _rho(T, q0 + iq)

            def over_z(c2, iz):
                acc2, own2 = c2
                a = _contract(rho, W[iz, iq])
                acc2 = acc2 + laws.imag_raw_node(a, u_c[iz], w_c[iz], x, f, eta)
                mn, pl = laws.imag_node_kernels(u_c[iz], w_c[iz], x_own, f, eta)
                own2 = own2 + _own(a, mn) + _own(laws.dagger(a), pl)
                return (acc2, own2), None
            (acc_, own_), _ = jax.lax.scan(over_z, (acc_, own_), jnp.arange(W.shape[0]))
            return (acc_, own_), None
        (acc, acc_own), _ = jax.lax.scan(over_q, (acc, acc_own), jnp.arange(nq_loc))
        return acc, acc_own

    def _real_body(W, dW, Wt, dWt, z_c, node0, acc, acc_own, anc, anc_own, T, with_anchor):
        q0 = _rank_q0() * nq_loc
        u_dev, w_dev = T["u"], T["w"]

        def over_q(carry, iq):
            acc_, own_, anc_, anco_ = carry
            rho, x, x_own, f = _rho(T, q0 + iq)

            def over_z(c2, iz):
                acc2, own2 = c2
                a = _contract(rho, W[iz, iq])
                d = _contract(rho, dW[iz, iq])
                at = _contract(rho, jnp.swapaxes(Wt[iz, iq], -1, -2))
                dt = _contract(rho, jnp.swapaxes(dWt[iz, iq], -1, -2))
                acc2 = acc2 + laws.real_knot(node0 + iz, z_c[iz], a, d, at, dt, x, f, h_ry)
                # own-energy diagonal: the same law with the row's own x
                idx, vl, vr, dl, dr = laws.hermite_coefficients(x_own, f, h_ry)
                node = node0 + iz
                sv = jnp.where(idx == node, vl, 0.0) + jnp.where(idx + 1 == node, vr, 0.0)
                sd = jnp.where(idx == node, dl, 0.0) + jnp.where(idx + 1 == node, dr, 0.0)
                neg = x_own < 0.0
                z2 = 2 * z_c[iz]
                own2 = (own2 + _own(a, jnp.where(neg, 0.0, sv)) + _own(d * z2, jnp.where(neg, 0.0, sd))
                        + _own(at, jnp.where(neg, sv, 0.0)) + _own(dt * z2, jnp.where(neg, sd, 0.0)))
                return (acc2, own2), (a, d)
            (acc_, own_), (a_all, d_all) = jax.lax.scan(over_z, (acc_, own_), jnp.arange(W.shape[0]))
            if with_anchor:
                # knot 0 is z = i eta: the two-pole anchor and the imaginary rule's anchor subtraction
                a0, d0 = a_all[0], d_all[0]
                total, cp, cm = laws.anchor_part(a0, d0, x, f, eta, beta)
                sums = laws.imag_anchor_sums(u_dev, w_dev, x, f, eta, beta)
                anc_ = anc_ + total - laws.anchor_subtraction(cp, cm, sums)
                # own-energy twin of the anchor, diagonal only
                fb = jnp.broadcast_to(f[:, None, :], x_own.shape)
                dv, dc = x_own - 1j * eta, x_own + 1j * eta
                own_total = 0
                for p, m, b in zip(cp, cm, beta):
                    odd = (fb * laws.odd_anchor(dv, x_own, b, -np.pi) + (1.0 - fb) * laws.odd_anchor(dc, x_own, b, np.pi)) * (1j / (2 * np.pi))
                    own_total = own_total + _own(0.5 * (p + m), laws.anchor_selector(x_own, f, eta, b)) + _own(p - m, odd)
                sums_own = laws.imag_anchor_sums(u_dev, w_dev, x_own, f, eta, beta)
                for j in range(len(cp)):
                    own_total = own_total - _own(cp[j], sums_own[j][0]) - _own(cm[j], sums_own[j][1])
                anco_ = anco_ + own_total
            return (acc_, own_, anc_, anco_), None
        (acc, acc_own, anc, anc_own), _ = jax.lax.scan(over_q, (acc, acc_own, anc, anc_own), jnp.arange(nq_loc))
        return acc, acc_own, anc, anc_own

    qspec4 = P(None, ("x", "y"))

    def _psum(*xs):
        return tuple(jax.lax.psum(x, ("x", "y")) for x in xs)

    imag_kernel = jax.jit(shard_map(
        lambda W, u_c, w_c, acc, own, T: _psum(*_imag_body(W, u_c, w_c, acc, own, T)),
        mesh=mesh_xy, in_specs=(qspec4, P(), P(), P(), P(), P()), out_specs=(P(), P()), check_vma=False))

    def _make_real(with_anchor):
        return jax.jit(shard_map(
            lambda W, dW, Wt, dWt, z_c, node0, acc, own, anc, anco, T: _psum(
                *_real_body(W, dW, Wt, dWt, z_c, node0, acc, own, anc, anco, T, with_anchor)),
            mesh=mesh_xy, in_specs=(qspec4, qspec4, qspec4, qspec4, P(), P(), P(), P(), P(), P(), P()),
            out_specs=(P(), P(), P(), P()), check_vma=False))
    real_kernel = _make_real(False)
    real_kernel_anchor = _make_real(True)

    def _sigma_x_body(V, acc, T):
        q0 = _rank_q0() * nq_loc

        def over_q(acc_, iq):
            rho, x, x_own, f = _rho(T, q0 + iq)
            a = _contract(rho, V[iq])                                   # -(1/nk) <m l|V_q|n l>
            return acc_ + jnp.einsum("kmnl,kl->kmn", a, f), None
        acc, _ = jax.lax.scan(over_q, acc, jnp.arange(nq_loc))
        return acc
    sigma_x_kernel = jax.jit(shard_map(lambda V, acc, T: jax.lax.psum(_sigma_x_body(V, acc, T), ("x", "y")),
                                       mesh=mesh_xy, in_specs=(P(("x", "y")), P(), P()), out_specs=P(), check_vma=False))

    zeros = lambda *s: jax.device_put(np.zeros(s, np.complex128), rep)
    acc_imag, own_imag = zeros(*acc_shape), zeros(*own_shape)
    acc_real, own_real = zeros(*acc_shape), zeros(*own_shape)
    acc_anc, own_anc = zeros(*acc_shape), zeros(*own_shape)
    timings = {}

    def clock(name, t0):
        timings[name] = timings.get(name, 0.0) + time.monotonic() - t0

    # ---- Sigma_x identity (INVARIANT: the CD pair densities reproduce the static exchange) ---
    t0 = time.monotonic()
    v_qpar = to_qpar(v_full)
    sx_cd = sigma_x_kernel(v_qpar, zeros(nk_red, nb_s, nb_s), tabs)
    jax.block_until_ready(sx_cd)
    clock("sigma_x", t0)
    sx_cd_h = host(sx_cd)
    sx_note = "sig_x not comparable (shape)"
    if sig_x is not None:
        try:
            sx_ref = np.asarray(host(sig_x))
            if sx_ref.shape[0] == nk and sx_ref.shape[1] >= m1:
                ref = sx_ref[kw_rows][:, m0:m1, m0:m1]
            elif sx_ref.shape[0] == nk_red and sx_ref.shape[1] >= m1:
                ref = sx_ref[:, m0:m1, m0:m1]
            else:
                ref = None
            if ref is not None:
                dmax = float(np.max(np.abs(np.real(np.diagonal(sx_cd_h, axis1=1, axis2=2))
                                            - np.real(np.diagonal(ref, axis1=1, axis2=2)))) * EV * 1e3)
                sx_note = f"max|diag Sigma_x(CD) - diag Sigma_x(static)| = {dmax:.3e} meV"
                if dmax > 0.05:
                    raise ValueError(f"GATE cd_sigma_x_identity: {sx_note}; want: <= 0.05 meV; why: the CD pair "
                                     "densities and V_q must reproduce the static exchange (head off)")
        except ValueError:
            raise
        except Exception as exc:  # shape/layout surprises are reported, never silently passed
            sx_note = f"sig_x comparison not evaluated ({type(exc).__name__}: {str(exc)[:120]})"
    _say(print_fn, started, f"Sigma_x identity: {sx_note}")
    ledger.append(_ledger_row("sigma_x", 0))

    # ---- imaginary axis ---------------------------------------------------------------------
    chunk_i = int(cd_handle.get("knot_chunk") or 0) or max(1, min(24, (8 * 2 ** 30) // max(1, 2 * nk * n * n * 16)))
    for ic, c0 in enumerate(range(0, n_imag, chunk_i)):
        c1 = min(c0 + chunk_i, n_imag)
        nz_ = c1 - c0
        t0 = time.monotonic()
        chi = direct_chi(1j * u_ry[c0:c1])
        jax.block_until_ready(chi)
        clock("chi0", t0)
        t0 = time.monotonic()
        qidx = jnp.asarray(np.tile(np.arange(nq_irr), nz_), dtype=jnp.int32)
        w_flat = _dyson_value(roots_zq(h_rep, qidx), _to_zq(chi))
        del chi
        W = _full_zone(w_flat, nz_)
        del w_flat
        jax.block_until_ready(W)
        clock("dyson_unfold", t0)
        t0 = time.monotonic()
        acc_imag, own_imag = imag_kernel(W, u_dev[c0:c1], w_dev[c0:c1], acc_imag, own_imag, tabs)
        jax.block_until_ready(acc_imag)
        clock("contract_imag", t0)
        del W
        ledger.append(_ledger_row("imag", ic))
        _say(print_fn, started, f"imaginary nodes {c1}/{n_imag} " + json.dumps({k: round(v, 1) for k, v in timings.items()}))

    # ---- real axis: knot chunks with FD-9 ghosts; knot 0 carries the anchor -----------------
    chunk_r = int(cd_handle.get("knot_chunk") or 0) or max(1, min(16, (6 * 2 ** 30) // max(1, 4 * nk * n * n * 16)))
    for ic, j0 in enumerate(range(0, J, chunk_r)):
        j1 = min(j0 + chunk_r, J)
        nz_ = j1 - j0
        t0 = time.monotonic()
        xs = h_ry * np.arange(j0 - GH, j1 + GH)
        chi_e = direct_chi(xs + 1j * eta)
        jax.block_until_ready(chi_e)
        clock("chi0", t0)
        t0 = time.monotonic()
        z_core = jnp.asarray(real_ry[j0:j1] + 1j * eta)
        dchi = slope_fn(chi_e, z_core)
        chi_c = core_fn(chi_e)
        del chi_e
        qidx = jnp.asarray(np.tile(np.arange(nq_irr), nz_), dtype=jnp.int32)
        wc_flat, dw_flat = _dyson_pair(roots_zq(h_rep, qidx), _to_zq(chi_c), _to_zq(dchi))
        del chi_c, dchi
        W = _full_zone(wc_flat, nz_)
        dW = _full_zone(dw_flat, nz_)
        del wc_flat, dw_flat
        Wt, dWt = partner(W), partner(dW)
        jax.block_until_ready(dWt)
        clock("dyson_unfold", t0)
        t0 = time.monotonic()
        kern = real_kernel_anchor if j0 == 0 else real_kernel
        acc_real, own_real, acc_anc, own_anc = kern(W, dW, Wt, dWt, z_core, jnp.int32(j0),
                                                     acc_real, own_real, acc_anc, own_anc, tabs)
        jax.block_until_ready(acc_real)
        clock("contract_real", t0)
        del W, dW, Wt, dWt
        ledger.append(_ledger_row("real", ic))
        if ic % 10 == 0 or j1 == J:
            _say(print_fn, started, f"real knots {j1}/{J} " + json.dumps({k: round(v, 1) for k, v in timings.items()}))

    # ---- assemble: cube on the file wedge -> padded band carrier -> full BZ ------------------
    t0 = time.monotonic()
    sigma_w = acc_anc + acc_imag + acc_real                                  # [E, kw, nb_s, nb_s] replicated
    own = host(own_anc + own_imag + own_real)                                # [kw, nb_s]
    axis = sigma_band_axis(nb_s, mesh_xy, ansatz="cd")
    nbc = int(axis.carrier)
    padw = nbc - nb_s
    out_sh = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    cube = jax.jit(lambda s: jnp.pad(s, ((0, 0), (0, 0), (0, padw), (0, padw))), out_shardings=out_sh)(sigma_w)
    del sigma_w
    if not full_bz:
        cube = jax.jit(lambda v: jnp.moveaxis(unfold_file_wedge_band_operator(
            sym, jnp.moveaxis(v, 1, 0), trs_rule="transpose"), 0, 1), out_shardings=out_sh)(cube)
    jax.block_until_ready(cube)
    clock("assemble", t0)
    finite = bool(np.isfinite(own).all())
    barrier("cd_sigma_done")
    receipt = dict(recipe=CD_RECIPE, eta_ev=eta * EV, knot_step_ev=h_ry * EV, J=J, reach_ev=real_ry[-1] * EV,
                   need_ev=need * EV, imaginary_nodes=n_imag, nk=nk, nq_irr=nq_irr, nk_red=nk_red, n_packed=n,
                   window=[m0, m1], l_sum=[l0, l1], n_omega=nE, mu_ev=mu * EV, pref=pref,
                   chunk_imag=chunk_i, chunk_real=chunk_r, seconds=timings, elapsed_s=time.monotonic() - started,
                   sigma_x_identity=sx_note, coulomb_support_ranks=[int(host(support).min()), int(host(support).max())],
                   full_bz=full_bz, kw_rows=kw_rows.tolist(), finite=finite)
    if rank0 and run_dir is not None:
        np.savez(os.path.join(run_dir, "cd_sigma_own_energy.npz"), kw_rows=kw_rows, bands0=np.arange(m0, m1),
                 eps_own_ry=eps_own, sigma_c_own_ry=own, sigma_x_diag_ry=np.real(np.diagonal(sx_cd_h, axis1=1, axis2=2)),
                 sigma_x_kmn_ry=sx_cd_h, mu_ry=mu, omega_ry=omega, u_ry=u_ry, w_ry=w_ry, real_ry=real_ry)
        with open(os.path.join(run_dir, "cd_sigma_receipt.json"), "w") as fh:
            json.dump(receipt, fh, indent=1, default=str)
        with open(os.path.join(run_dir, "cd_leak_ledger.csv"), "w") as fh:
            fh.write("stage,chunk,bytes_in_use,peak_bytes_in_use\n")
            for row in ledger:
                fh.write(f"{row['stage']},{row['chunk']},{row['bytes_in_use']},{row['peak_bytes_in_use']}\n")
    if not finite:
        raise ValueError("GATE cd_finite: non-finite Sigma_c")
    _say(print_fn, started, "done " + json.dumps({k: round(v, 1) for k, v in timings.items()}))
    return SigmaOmegaResult(omega_ry=np.asarray(omega_grid_ry, np.float64),
                            omega_ev=np.asarray(omega_grid_ry, np.float64) * EV,
                            sigma_c_kij=cube, band_axis=axis, band_counts=(), efermi_ry=float(efermi_ry))
