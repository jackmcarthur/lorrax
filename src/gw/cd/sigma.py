"""Sigma_c(omega) by contour deformation on the exact ISDF W(z): the production CD route.

Where this sits (docs/dev/cd_frequency_integration.md): ``gw.sigma_dispatch._compute_mpa_sigma``
calls :func:`compute_sigma_c_cd_omega_grid` instead of the pole executor when the deck says
``sigma_frequency_integration = contour_deformation``; the result is the same
:class:`gw.ppm_sigma.SigmaOmegaResult` the one finalizer consumes.  No physics is rebuilt here:
chi0 is ``gw.w_isdf.compute_chi0_direct_fractional(ordered=True)`` (exact ordered Lehmann sum at
every node and knot), the Dyson pair is the response bank's equation set (W_c = H X (1-X)^-1 H,
dW_c/ds = W (pref dchi/ds) W, roots H of ``response_bank._coulomb_algebra``), the q-wedge unfold is
``symmetry_maps`` (``QirrOperator.unfold``, general-operator ``pair_transpose`` rule), the k unfold
of the cube is ``symmetry_maps.unfold_file_wedge_band_operator``, the contour laws are
``gw.cd.laws`` (ordered_cd.py of the Fe harness, one layout).

    Sigma_c(E) = anchor(E) + sum_i imag_i(E) - anchor_subtraction(E) + sum_j real_j(E)

Streaming.  W(z) is never persisted.  Nodes (z = i u) and knots (z = j h + i eta) are streamed in
chunks of ``runtime.tiles.tile_units`` (TASTE 96: the tile is sized from the chunk's own shapes,
never from free memory or the deck budget; the last chunk is padded with zero-weight nodes or
unreferenced knots so every chunk has one shape):

    chi(z) [nz, nq_irr, n, n] face --(q,z)-major flat, local batched LU--> W_c, dW_c/ds
      --unfold per z (wedge decks)--> [nz, nq_pad, n, n] q-parallel P(None, ('x','y'))
      --shard_map over q: scan over the rank's q, scan over z--> accumulators [E, kw, m, n]

The real axis needs no W_{-q} gather: the rank holding W_q' adds pair q''s x >= 0 residues
(contractions with W_q') and pair (-q')'s x < 0 residues (contractions with W_q'^T), recomputing
the pair densities rho_{-q'} from the replicated tables.  FD-9 ghost knots roll from chunk to
chunk (chi0 runs once per knot).  The q axis is padded to a multiple of P with zero W rows
(``qvalid`` masks them).  Accumulators are replicated; each kernel returns acc + psum(increment).

Units: Ry throughout (omega, eta, energies, Sigma).  Occupations: the fixed-N ``OccupationState``
of a metal, or the loaded step table of an insulator.
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
CD_RECIPE = dict(version="cd_production_v2", laws="ordered_cd TRREF / Run211 jets",
                 beta_ev=[0.5, 4.0], slope="fd9 rolling ghosts", imaginary_rule="tan_mapped_gauss two-panel",
                 chi0="w_isdf.compute_chi0_direct_fractional(ordered=True)",
                 dyson="response_bank pair equations, local batched jnp.linalg.solve over (q, z)",
                 tile="runtime.tiles.tile_units", real_axis="owner-rank branches, no W_-q gather")


def _say(print_fn, started, msg):
    print_fn(f"  CD +{time.monotonic() - started:7.1f}s {msg}")


def ledger_sample(stage, chunk, seconds):
    """One device-ledger row (this process's first device): bytes in use, the pool high-water since
    the previous row (``runtime.xla_memory.pool_high_water``, reset here), live array count."""
    from runtime.xla_memory import pool_high_water
    try:
        st = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # pragma: no cover - CPU backends have no stats
        st = {}
    high = pool_high_water(reset=True)
    return dict(stage=stage, chunk=int(chunk), seconds=round(float(seconds), 3),
                bytes_in_use=int(st.get("bytes_in_use", -1)), pool_high_water=-1 if high is None else int(high),
                peak_bytes_in_use=int(st.get("peak_bytes_in_use", -1)), live_arrays=len(jax.live_arrays()))


def chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, *, axis):
    """Per-rank bytes one node (imag) or knot (real) adds to a chunk's scaling set (complex128).

    imag: chi [face] + (q,z) flat + W full zone        = 16 n^2 (2 nq_irr + nq_pad) / P
    real: chi window + chi/dchi flat + W_c/dW flat + W, dW full zone = 16 n^2 (5 nq_irr + 2 nq_pad) / P
    """
    per = (2 * nq_irr + nq_pad) if axis == "imag" else (5 * nq_irr + 2 * nq_pad)
    return 16.0 * n * n * per / ndev


# ------------------------------------------------------------------------------- pair-basis kernels
def pair_rows(T, q):
    """Pair densities and energy tables of pair q (k -> k - q) at the file-wedge rows.

    T (replicated): psi_L [nk, L, s, n], psi_M [kw, m, s, n], KMQ [nq_pad, nk], kw [kw],
    enk [nk, L], f [nk, L], omega [E], eps_own [kw, m].
    Returns rho [kw, m, L, n] = conj(psi_{k-q,l}) psi_{k,m} summed over spinor, x [kw, E, L],
    x_own [kw, m, L] (Ry, omega - eps_l(k-q)), f [kw, L].
    """
    rows = T["KMQ"][q][T["kw"]]
    psi_l = T["psi_L"][rows]
    rho = jnp.einsum("klsu,kmsu->kmlu", jnp.conj(psi_l), T["psi_M"])
    x = T["omega"][None, :, None] - T["enk"][rows][:, None, :]
    x_own = T["eps_own"][:, :, None] - T["enk"][rows][:, None, :]
    return rho, x, x_own, T["f"][rows]


def contract(rho, op, nk):
    """a[k, m, n, l] = -(1/nk) sum_uv conj(rho[k,m,l,u]) op[u,v] rho[k,n,l,v]   (class C then D)."""
    t = jnp.einsum("kmlu,uv->kmlv", jnp.conj(rho), op)
    return -jnp.einsum("kmlv,knlv->kmnl", t, rho) / nk


def build_contraction_kernels(mesh_xy, *, nk, nq_loc, eta, h_ry, beta):
    """The four q-parallel CD kernels (jitted shard_maps; tables T and accumulators replicated).

    sigma_x(V [nq_pad, n, n] q-par, acc [kw, m, m], T)                     -> acc + Sigma_x
    imag(W [nz, nq_pad, n, n] q-par, u [nz], w [nz], acc [E,kw,m,m], own [kw,m], T) -> (acc', own')
    real(W, dW, z [nz], node0, acc, own, T)                                 -> (acc', own')
    anchor(W0, dW0 [nq_pad, n, n] at z = i eta, acc, own, T)                -> (acc', own')
    Every kernel returns its input accumulator plus the psum over ranks of its own increment.
    """
    axes = ("x", "y")

    def q0():
        return (jax.lax.axis_index("x") * mesh_xy.shape["y"] + jax.lax.axis_index("y")) * nq_loc

    def psum_add(acc, inc):
        return acc + jax.lax.psum(inc, axes)

    def cd_sigma_x(V, acc, T):
        def over_q(s, iq):
            g = q0() + iq
            rho, _, _, f = pair_rows(T, g)
            a = contract(rho, V[iq], nk)
            return s + T["qvalid"][g] * jnp.einsum("kmnl,kl->kmn", a, f), None
        inc, _ = jax.lax.scan(over_q, jnp.zeros_like(acc), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc)

    def cd_imag_contract(W, u_c, w_c, acc, own, T):
        def over_q(carry, iq):
            g = q0() + iq
            rho, x, x_own, f = pair_rows(T, g)

            def over_z(c2, iz):
                a = contract(rho, W[iz, iq], nk)
                ops, wts = laws.imag_node_terms(a, u_c[iz], w_c[iz], x, f, eta)
                _, wts_own = laws.imag_node_terms(a, u_c[iz], w_c[iz], x_own, f, eta)
                return (c2[0] + laws.apply(ops, wts), c2[1] + laws.own_apply(ops, wts_own)), None
            (s_acc, s_own), _ = jax.lax.scan(over_z, (jnp.zeros_like(acc), jnp.zeros_like(own)), jnp.arange(W.shape[0]), unroll=1)
            v = T["qvalid"][g]
            return (carry[0] + v * s_acc, carry[1] + v * s_own), None
        (inc, inc_own), _ = jax.lax.scan(over_q, (jnp.zeros_like(acc), jnp.zeros_like(own)), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc), psum_add(own, inc_own)

    def cd_real_contract(W, dW, z_c, node0, acc, own, T):
        def over_q(carry, iq):
            g = q0() + iq
            gn = T["qneg"][g]
            rho_p, x_p, xo_p, f_p = pair_rows(T, g)       # pair g, residues x >= 0, operator W_g
            rho_n, x_n, xo_n, f_n = pair_rows(T, gn)      # pair -g, residues x < 0, operator W_g^T

            def over_z(c2, iz):
                node, z = node0 + iz, z_c[iz]
                Wz, dWz = W[iz, iq], dW[iz, iq]
                a, d = contract(rho_p, Wz, nk), contract(rho_p, dWz, nk)
                at, dt = contract(rho_n, Wz.T, nk), contract(rho_n, dWz.T, nk)
                op_p, w_p = laws.real_knot_terms(node, z, a, d, x_p, f_p, h_ry, negative=False)
                op_n, w_n = laws.real_knot_terms(node, z, at, dt, x_n, f_n, h_ry, negative=True)
                _, wo_p = laws.real_knot_terms(node, z, a, d, xo_p, f_p, h_ry, negative=False)
                _, wo_n = laws.real_knot_terms(node, z, at, dt, xo_n, f_n, h_ry, negative=True)
                inc = laws.apply(jnp.concatenate([op_p, op_n], -1), jnp.concatenate([w_p, w_n], -1))
                inc_own = laws.own_apply(op_p, wo_p) + laws.own_apply(op_n, wo_n)
                return (c2[0] + inc, c2[1] + inc_own), None
            (s_acc, s_own), _ = jax.lax.scan(over_z, (jnp.zeros_like(acc), jnp.zeros_like(own)), jnp.arange(W.shape[0]), unroll=1)
            v = T["qvalid"][g]
            return (carry[0] + v * s_acc, carry[1] + v * s_own), None
        (inc, inc_own), _ = jax.lax.scan(over_q, (jnp.zeros_like(acc), jnp.zeros_like(own)), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc), psum_add(own, inc_own)

    def cd_anchor(W0, dW0, acc, own, T):
        def over_q(carry, iq):
            g = q0() + iq
            rho, x, x_own, f = pair_rows(T, g)
            a0, d0 = contract(rho, W0[iq], nk), contract(rho, dW0[iq], nk)
            total, cp, cm = laws.anchor_part(a0, d0, x, f, eta, beta)
            sub = laws.anchor_subtraction(cp, cm, laws.imag_anchor_sums(T["u_all"], T["w_all"], x, f, eta, beta))
            total_o, _, _ = laws.anchor_part(a0, d0, x_own, f, eta, beta, apply_fn=laws.own_apply)
            sub_o = laws.anchor_subtraction(cp, cm, laws.imag_anchor_sums(T["u_all"], T["w_all"], x_own, f, eta, beta),
                                            apply_fn=laws.own_apply)
            v = T["qvalid"][g]
            return (carry[0] + v * (total - sub), carry[1] + v * (total_o - sub_o)), None
        (inc, inc_own), _ = jax.lax.scan(over_q, (jnp.zeros_like(acc), jnp.zeros_like(own)), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc), psum_add(own, inc_own)

    q3, q4 = P(axes), P(None, axes)

    def wrap(fn, in_specs, out_specs):
        sm = shard_map(fn, mesh=mesh_xy, in_specs=in_specs, out_specs=out_specs, check_vma=False)

        def named(*args):
            return sm(*args)
        named.__name__ = named.__qualname__ = fn.__name__
        return jax.jit(named)

    return SimpleNamespace(
        sigma_x=wrap(cd_sigma_x, (q3, P(), P()), P()),
        imag=wrap(cd_imag_contract, (q4, P(), P(), P(), P(), P()), (P(), P())),
        real=wrap(cd_real_contract, (q4, q4, P(), P(), P(), P(), P()), (P(), P())),
        anchor=wrap(cd_anchor, (q3, q3, P(), P(), P()), (P(), P())))


def build_dyson_kernels(mesh_xy, *, n, pref):
    """Dyson pair on the (q, z)-parallel flat layout P(('x','y'), None, None): local batched LU (class B).

    pair(H [b, n, n], chi [b, n, n], dchi/ds [b, n, n]) -> (W_c, dW_c/ds); value(H, chi) -> W_c, with
    X = pref H chi H, W_c = H X (1 - X)^-1 H, W = W_c + H H, dW_c/ds = W (pref dchi/ds) W
    (the response bank's pair equations, ``gw.response_bank._response_programs``).
    """
    eye = jnp.eye(n, dtype=jnp.complex128)
    flat = NamedSharding(mesh_xy, P(("x", "y"), None, None))

    def one_value(h, chi_raw):
        x = pref * (h @ chi_raw @ h)
        return h @ (x @ jnp.linalg.solve(eye - x, eye)) @ h

    def one_pair(h, chi_raw, dchi_raw):
        wc = one_value(h, chi_raw)
        w = wc + h @ h
        return wc, w @ (pref * dchi_raw) @ w

    def cd_dyson_pair(hs, chi_flat, dchi_flat):
        return jax.vmap(one_pair)(hs, chi_flat, dchi_flat)

    def cd_dyson_value(hs, chi_flat):
        return jax.vmap(one_value)(hs, chi_flat)
    return (jax.jit(cd_dyson_pair, out_shardings=(flat, flat)), jax.jit(cd_dyson_value, out_shardings=flat))


# ------------------------------------------------------------------------------- the Sigma consumer
def compute_sigma_c_cd_omega_grid(
    wfns, cd_handle, meta, mesh_xy, *, sym, config, omega_grid_ry, efermi_ry,
    regularization_width_ry, occupation_state=None, print_fn=print, run_dir=None,
    sig_x=None, band_brackets=None, band_counts=None, omega_eta_ry=None, **_body_options):
    """Sigma_c(omega) [n_omega, nk_full, nb_c, nb_c] (Ry) on the deck's omega grid by CD.

    ``cd_handle`` is ``W_by_role["cd_stream"]`` from ``gw.screening.compute_screening_model``
    (``V_q`` wedge operator, deck sub-keys); ``omega_grid_ry`` is relative to ``efermi_ry``;
    ``regularization_width_ry`` is eta.  The rest of ``body_options`` is the pole executor's.
    Returns :class:`gw.ppm_sigma.SigmaOmegaResult`, cube at ``P(None, None, 'x', 'y')`` on the padded
    Sigma band carrier.  Rank 0 writes beside the deck: ``cd_sigma_receipt.json``,
    ``cd_sigma_own_energy.npz`` (diagonal at the DFT energies and its three contour parts),
    ``cd_leak_ledger.csv`` and the partial dumps ``cd_partial_{imag,anchor}.npz``.
    """
    from common.collectives import barrier, gather_to_host, replicate_to_mesh
    from gw import response_bank as rb
    from gw.ppm_sigma import SigmaOmegaResult, sigma_band_axis
    from gw.w_isdf import compute_chi0_direct_fractional, _w_solve_pref_scalar
    from runtime.tiles import tile_units, TILE_BYTES
    from symmetry_maps import (kgrid_shift_map, q_negation_index, star_tables_of,
                               unfold_file_wedge_band_operator, unfold_wavefunction_local)

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

    # ---- geometry: k grid, k-q maps, q wedge, file k wedge, q padding --------------------------------
    grid = tuple(int(v) for v in meta.kgrid)
    nk = int(np.prod(grid))
    n = int(meta.mu_basis.n_packed)
    q_irr = np.asarray(sym.q_irr_full_idx, dtype=np.int64)
    nq_irr = int(q_irr.size)
    qneg = np.asarray(q_negation_index(grid), dtype=np.int64)

    ijk = np.stack(np.unravel_index(np.arange(nk), grid), axis=-1)                    # [nk, 3]
    dif = (ijk[None, :, :] - ijk[:, None, :]) % np.asarray(grid)                        # [nq, nk, 3]: k - q
    KMQ = np.ravel_multi_index((dif[..., 0], dif[..., 1], dif[..., 2]), grid).astype(np.int64)
    probe_q = min(1, nk - 1)
    if not np.array_equal(KMQ[probe_q], np.asarray(kgrid_shift_map(*grid, -ijk[probe_q].astype(np.int32))[0])):
        raise ValueError("GATE cd_kq_map: the vectorized k - q table disagrees with symmetry_maps.kgrid_shift_map")
    kq_sym = np.asarray(sym.kq_map, dtype=np.int64)             # [nk_full, nq_red]
    if kq_sym.shape == (nk, nk) and int(np.sum(KMQ != kq_sym.T)):
        raise ValueError("GATE cd_kq_map: kgrid_shift_map vs SymMaps.kq_map differ")
    irr_k, sidx_k, _nsp = star_tables_of(sym)
    nk_red = int(sym.nk_red)
    idx0 = np.flatnonzero(np.asarray(sidx_k) == 0)
    stars, first = np.unique(np.asarray(irr_k)[idx0], return_index=True)
    if not np.array_equal(stars, np.arange(nk_red)):
        raise ValueError("GATE cd_file_wedge: a star has no identity member in star_tables_of(sym)")
    kw_rows = idx0[first]                                                               # [nk_red] file-wedge rows
    full_bz = nk_red == nk and np.array_equal(kw_rows, np.arange(nk))
    # Symmetry certificate (instrument, off by default; docs/dev/env_vars.md): append one non-identity member
    # of each of the first N multi-member stars to the computed rows; the wedge-unfolded cube is compared
    # with these direct rows after assembly.  Production rows stay the file wedge.
    n_cert = 0 if full_bz else int(os.environ.get("LORRAX_CD_STAR_CERTIFICATE", "0") or 0)
    cand = np.flatnonzero(np.asarray(sidx_k) != 0)
    c_stars, c_first = np.unique(np.asarray(irr_k)[cand], return_index=True)
    cert = list(zip(c_stars[:n_cert].tolist(), cand[c_first][:n_cert].tolist()))
    rows_all = np.concatenate([kw_rows, np.asarray([k for _, k in cert], dtype=kw_rows.dtype)])
    n_rows = int(rows_all.size)
    nq_pad = -(-nk // ndev) * ndev
    nq_loc = nq_pad // ndev
    pad_q = np.arange(nq_pad) % nk if nq_pad > nk else np.arange(nk)
    KMQ_pad = KMQ[np.where(np.arange(nq_pad) < nk, pad_q, 0)]
    qneg_pad = np.where(np.arange(nq_pad) < nk, qneg[pad_q], 0)
    qvalid = (np.arange(nq_pad) < nk).astype(np.float64)

    # ---- occupations and energies -------------------------------------------------------------------
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
    eps_own = energy[rows_all, m0:m1]                              # [rows, nb_s] own energies

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
    u_ry_all, w_ry_all = (np.asarray(v) / EV for v in tan_mapped_gauss_rule(n_imag, eta * EV))

    # ---- tiles (TASTE 96): from the chunk's shapes; (q, z) flat axis divisible by P -------------------
    zq_mult = ndev // int(np.gcd(nq_irr, ndev))
    override = int(cd_handle.get("knot_chunk") or 0)

    def tile(axis, count):
        units = override or tile_units(chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis=axis), count)
        return int(-(-max(1, min(units, count)) // zq_mult) * zq_mult)
    chunk_i, chunk_r = tile("imag", n_imag), tile("real", J)
    n_imag_pad = -(-n_imag // chunk_i) * chunk_i
    J_pad = -(-J // chunk_r) * chunk_r
    u_ry = np.concatenate([u_ry_all, np.full(n_imag_pad - n_imag, u_ry_all[-1])])
    w_ry = np.concatenate([w_ry_all, np.zeros(n_imag_pad - n_imag)])         # padded nodes weigh zero
    real_ry = h_ry * np.arange(J_pad)                                         # knots >= J are never referenced
    _say(print_fn, started, f"geometry nk={nk} nq_irr={nq_irr} nq_pad={nq_pad} nk_red={nk_red} n_packed={n} "
         f"window m=[{m0},{m1}) l=[{l0},{l1}) nE={nE} eta={eta * EV:.4f} eV h={h_ry * EV:.4f} eV J={J} "
         f"(padded {J_pad}) knots to {real_ry[J - 1] * EV:.3f} eV (need {need * EV:.3f} eV) imag nodes={n_imag} "
         f"(padded {n_imag_pad}) mu={mu * EV:.6f} eV full_bz={full_bz}; tile {TILE_BYTES / 2**30:.0f} GiB/rank: "
         f"chunk imag {chunk_i} ({chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis='imag') / 2**20:.1f} MiB/node), "
         f"real {chunk_r} ({chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis='real') / 2**20:.1f} MiB/knot)"
         + (f" [cd_knot_chunk override {override}]" if override else ""))

    # ---- wavefunctions at every full k (parents unfolded by the typed centroid action) ---------------
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
    psi_M = jax.jit(lambda a, r: a[r, m0:m1] * 1.0, out_shardings=rep)(psi_full, jnp.asarray(rows_all))
    del psi_full

    def _rep(a):
        return replicate_to_mesh(np.asarray(a), mesh_xy)
    tabs = dict(psi_L=psi_L, psi_M=psi_M, KMQ=_rep(KMQ_pad.astype(np.int32)), kw=_rep(rows_all.astype(np.int32)),
                enk=_rep(enk_L), f=_rep(f_L), omega=_rep(omega), eps_own=_rep(eps_own),
                qneg=_rep(qneg_pad.astype(np.int32)), qvalid=_rep(qvalid), u_all=_rep(u_ry_all), w_all=_rep(w_ry_all))
    u_dev, w_dev = _rep(u_ry), _rep(w_ry)
    beta = tuple(float(b) for b in laws.BETA_RY)
    K = build_contraction_kernels(mesh_xy, nk=nk, nq_loc=nq_loc, eta=eta, h_ry=h_ry, beta=beta)

    # ---- Coulomb: wedge operator, full zone, roots ------------------------------------------------------
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
    if unfold_w is None and not np.array_equal(q_irr, np.arange(nk)):
        raise ValueError("GATE cd_vq_wedge: whole-zone V_q but sym.q_irr_full_idx is not 0..nk-1")
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
    zq_par = NamedSharding(mesh_xy, P(("x", "y"), None, None))
    dyson_pair, dyson_value = build_dyson_kernels(mesh_xy, n=n, pref=pref)

    def cd_to_qz(chi4):
        """[nz, nq_irr, n, n] face -> [(q, z), n, n] q-major flat on the (q,z)-parallel layout."""
        nz_ = chi4.shape[0]
        flat = jnp.swapaxes(chi4, 0, 1).reshape((nq_irr * nz_, n, n))
        flat = jax.lax.with_sharding_constraint(flat, mid3)
        return jax.lax.with_sharding_constraint(flat, zq_par)
    to_qz = jax.jit(cd_to_qz)
    roots_qz = jax.jit(lambda h, nz_: jnp.repeat(h, nz_, axis=0, total_repeat_length=nq_irr * nz_),
                       static_argnums=1, out_shardings=zq_par)

    pad_rows = nq_pad - nk

    def cd_full_zone_direct(flat):
        nz_ = flat.shape[0] // nq_irr
        w = jnp.swapaxes(flat.reshape((nq_irr, nz_, n, n)), 0, 1)
        return jnp.pad(w, ((0, 0), (0, pad_rows), (0, 0), (0, 0))) if pad_rows else w
    full_zone_direct = jax.jit(cd_full_zone_direct, out_shardings=qpar4)
    take_z = jax.jit(lambda flat, iz: jax.lax.dynamic_index_in_dim(
        flat.reshape((nq_irr, flat.shape[0] // nq_irr, n, n)), iz, axis=1, keepdims=False), out_shardings=face)
    to_qpar = jax.jit(lambda a: (jnp.pad(a, ((0, pad_rows), (0, 0), (0, 0))) if pad_rows else a * 1.0),
                      out_shardings=qpar3)
    stack_q = jax.jit(lambda *a: jnp.stack(a), out_shardings=qpar4)

    def full_zone(flat):
        """[(q, z) flat] -> [nz, nq_pad, n, n] q-parallel (unfold per z on wedge decks)."""
        if unfold_w is None:
            return full_zone_direct(flat)
        nz_ = flat.shape[0] // nq_irr
        return stack_q(*[to_qpar(unfold_w(take_z(flat, iz))) for iz in range(nz_)])

    def direct_chi(z_ry):
        chi = compute_chi0_direct_fractional(
            wfns, np.asarray(z_ry, np.complex128), meta, mesh_xy, occupation_state=occ_for_chi,
            kminq_rows=KMQ[q_irr], nb_logical=nb_log, ordered=True)
        if len(z_ry) == 1:
            chi = chi[None]
        return jax.jit(lambda a: a * 1.0, out_shardings=face4)(chi)

    def cd_fd9_slope(chi_e, z_core):
        """FD-9 d/dx on the knot grid -> dchi/ds at the core knots: [nz, nq_irr, n, n]."""
        nz_ = chi_e.shape[0] - 2 * GH
        dz = sum(FD9[m + 4] * chi_e[GH + m: GH + m + nz_] for m in (-4, -3, -2, -1, 1, 2, 3, 4)) / h_ry
        return dz / (2 * z_core)[:, None, None, None]
    slope_fn = jax.jit(cd_fd9_slope, out_shardings=face4)
    core_fn = jax.jit(lambda a: a[GH:a.shape[0] - GH] * 1.0, out_shardings=face4)
    tail_fn = jax.jit(lambda a: a[a.shape[0] - 2 * GH:] * 1.0, out_shardings=face4)
    join_fn = jax.jit(lambda a, b: jnp.concatenate([a, b], axis=0), out_shardings=face4)

    def zeros(*shape):
        return jax.jit(lambda: jnp.zeros(shape, jnp.complex128), out_shardings=rep)()
    acc_shape, own_shape = (nE, n_rows, nb_s, nb_s), (n_rows, nb_s)
    acc_imag, own_imag = zeros(*acc_shape), zeros(*own_shape)
    acc_real, own_real = zeros(*acc_shape), zeros(*own_shape)
    acc_anc, own_anc = zeros(*acc_shape), zeros(*own_shape)
    timings = {}

    def clock(name, t0):
        timings[name] = timings.get(name, 0.0) + time.monotonic() - t0

    def partial_dump(name, **arrays):
        out = {k: host(v)[:nk_red] for k, v in arrays.items()}       # collective on every rank; wedge rows
        if rank0 and run_dir is not None:
            np.savez(os.path.join(run_dir, f"cd_partial_{name}.npz"), kw_rows=kw_rows, bands0=np.arange(m0, m1),
                     elapsed_s=time.monotonic() - started, **out)

    # ---- Sigma_x identity (INVARIANT: the CD pair densities reproduce the static exchange) ------------
    t0 = time.monotonic()
    sx_cd = K.sigma_x(to_qpar(v_full), zeros(n_rows, nb_s, nb_s), tabs)
    sx_cd = sx_cd[:nk_red] if cert else sx_cd
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
    ledger.append(ledger_sample("sigma_x", 0, time.monotonic() - t0))

    # ---- imaginary axis -----------------------------------------------------------------------------
    for ic, c0 in enumerate(range(0, n_imag_pad, chunk_i)):
        c1 = c0 + chunk_i
        t_chunk = t0 = time.monotonic()
        chi = direct_chi(1j * u_ry[c0:c1])
        jax.block_until_ready(chi)
        clock("chi0", t0)
        t0 = time.monotonic()
        w_flat = dyson_value(roots_qz(h_rep, chunk_i), to_qz(chi))
        del chi
        jax.block_until_ready(w_flat)
        clock("dyson", t0)
        t0 = time.monotonic()
        W = full_zone(w_flat)
        del w_flat
        jax.block_until_ready(W)
        clock("unfold", t0)
        t0 = time.monotonic()
        acc_imag, own_imag = K.imag(W, u_dev[c0:c1], w_dev[c0:c1], acc_imag, own_imag, tabs)
        jax.block_until_ready(acc_imag)
        clock("contract_imag", t0)
        del W
        ledger.append(ledger_sample("imag", ic, time.monotonic() - t_chunk))
        _say(print_fn, started, f"imaginary nodes {min(c1, n_imag)}/{n_imag} " + json.dumps({k: round(v, 1) for k, v in timings.items()}))
    partial_dump("imag", sigma_c_imag_own_ry=own_imag)

    # ---- real axis: knot chunks with rolling FD-9 ghosts; knot 0 carries the anchor -----------------
    tail = None
    for ic, j0 in enumerate(range(0, J_pad, chunk_r)):
        j1 = j0 + chunk_r
        t_chunk = t0 = time.monotonic()
        if tail is None:
            chi_e = direct_chi(h_ry * np.arange(j0 - GH, j1 + GH) + 1j * eta)
        else:
            chi_e = join_fn(tail, direct_chi(h_ry * np.arange(j0 + GH, j1 + GH) + 1j * eta))
        jax.block_until_ready(chi_e)
        clock("chi0", t0)
        t0 = time.monotonic()
        tail = tail_fn(chi_e)
        z_core = jnp.asarray(real_ry[j0:j1] + 1j * eta)
        dchi = slope_fn(chi_e, z_core)
        chi_c = core_fn(chi_e)
        del chi_e
        wc_flat, dw_flat = dyson_pair(roots_qz(h_rep, chunk_r), to_qz(chi_c), to_qz(dchi))
        del chi_c, dchi
        jax.block_until_ready(dw_flat)
        clock("dyson", t0)
        t0 = time.monotonic()
        W, dW = full_zone(wc_flat), full_zone(dw_flat)
        del wc_flat, dw_flat
        jax.block_until_ready(dW)
        clock("unfold", t0)
        if j0 == 0:
            t0 = time.monotonic()
            acc_anc, own_anc = K.anchor(W[0], dW[0], acc_anc, own_anc, tabs)
            jax.block_until_ready(acc_anc)
            clock("anchor", t0)
            partial_dump("anchor", sigma_c_anchor_own_ry=own_anc, sigma_c_imag_own_ry=own_imag)
        t0 = time.monotonic()
        acc_real, own_real = K.real(W, dW, z_core, jnp.int32(j0), acc_real, own_real, tabs)
        jax.block_until_ready(acc_real)
        clock("contract_real", t0)
        del W, dW
        ledger.append(ledger_sample("real", ic, time.monotonic() - t_chunk))
        if ic % 10 == 0 or j1 >= J_pad:
            _say(print_fn, started, f"real knots {min(j1, J)}/{J} " + json.dumps({k: round(v, 1) for k, v in timings.items()}))
    del tail

    # ---- assemble: cube on the file wedge -> padded band carrier -> full BZ ---------------------------
    t0 = time.monotonic()
    sigma_w = acc_anc + acc_imag + acc_real                                  # [E, rows, nb_s, nb_s] replicated
    sigma_cert = jax.jit(lambda a: a[:, nk_red:] * 1.0, out_shardings=rep)(sigma_w) if cert else None
    if cert:
        sigma_w = jax.jit(lambda a: a[:, :nk_red] * 1.0, out_shardings=rep)(sigma_w)
    parts = {k: host(v)[:nk_red] for k, v in (("anchor", own_anc), ("imag", own_imag), ("real", own_real))}
    own = parts["anchor"] + parts["imag"] + parts["real"]
    axis = sigma_band_axis(nb_s, mesh_xy, ansatz="cd")
    nbc = int(axis.carrier)
    padw = nbc - nb_s
    out_sh = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    cube = jax.jit(lambda s: jnp.pad(s, ((0, 0), (0, 0), (0, padw), (0, padw))), out_shardings=out_sh)(sigma_w)
    del sigma_w, acc_anc, acc_imag, acc_real
    if not full_bz:
        cube = jax.jit(lambda v: jnp.moveaxis(unfold_file_wedge_band_operator(
            sym, jnp.moveaxis(v, 1, 0), trs_rule="transpose"), 0, 1), out_shardings=out_sh)(cube)
    jax.block_until_ready(cube)
    clock("assemble", t0)
    cert_note = None
    if cert:
        k_cert = jnp.asarray([k for _, k in cert], dtype=jnp.int32)
        unf = host(jax.jit(lambda v, kk: jnp.take(v, kk, axis=1)[:, :, :nb_s, :nb_s], out_shardings=rep)(cube, k_cert))
        direct = host(sigma_cert)
        dev = float(np.abs(unf - direct).max() * EV * 1e3)
        scale = float(np.abs(direct).max() * EV * 1e3)
        cert_note = dict(stars=[[int(i), int(k)] for i, k in cert], max_abs_meV=dev, max_ref_meV=scale)
        _say(print_fn, started, f"symmetry certificate (direct star rows vs wedge-unfolded cube): {json.dumps(cert_note)}")
    ledger.append(ledger_sample("assemble", 0, time.monotonic() - t0))
    finite = bool(np.isfinite(own).all())
    barrier("cd_sigma_done")
    receipt = dict(recipe=CD_RECIPE, eta_ev=eta * EV, knot_step_ev=h_ry * EV, J=J, J_pad=J_pad,
                   reach_ev=real_ry[J - 1] * EV, need_ev=need * EV, imaginary_nodes=n_imag, n_imag_pad=n_imag_pad,
                   nk=nk, nq_irr=nq_irr, nq_pad=nq_pad, nk_red=nk_red, n_packed=n, window=[m0, m1], l_sum=[l0, l1],
                   n_omega=nE, mu_ev=mu * EV, pref=pref, chunk_imag=chunk_i, chunk_real=chunk_r,
                   tile_bytes=TILE_BYTES, bytes_per_node_imag=chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis="imag"),
                   bytes_per_knot_real=chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis="real"),
                   seconds=timings, elapsed_s=time.monotonic() - started, sigma_x_identity=sx_note,
                   coulomb_support_ranks=[int(host(support).min()), int(host(support).max())],
                   full_bz=full_bz, kw_rows=kw_rows.tolist(), finite=finite, ndev=ndev, symmetry_certificate=cert_note)
    if rank0 and run_dir is not None:
        np.savez(os.path.join(run_dir, "cd_sigma_own_energy.npz"), kw_rows=kw_rows, bands0=np.arange(m0, m1),
                 eps_own_ry=eps_own[:nk_red], sigma_c_own_ry=own, sigma_c_anchor_own_ry=parts["anchor"],
                 sigma_c_imag_own_ry=parts["imag"], sigma_c_real_own_ry=parts["real"],
                 sigma_x_diag_ry=np.real(np.diagonal(sx_cd_h, axis1=1, axis2=2)),
                 sigma_x_kmn_ry=sx_cd_h, mu_ry=mu, omega_ry=omega, u_ry=u_ry_all, w_ry=w_ry_all, real_ry=real_ry[:J])
        with open(os.path.join(run_dir, "cd_sigma_receipt.json"), "w") as fh:
            json.dump(receipt, fh, indent=1, default=str)
        with open(os.path.join(run_dir, "cd_leak_ledger.csv"), "w") as fh:
            cols = list(ledger[0])
            fh.write(",".join(cols) + "\n")
            for row in ledger:
                fh.write(",".join(str(row[c]) for c in cols) + "\n")
    if not finite:
        raise ValueError("GATE cd_finite: non-finite Sigma_c")
    _say(print_fn, started, "done " + json.dumps({k: round(v, 1) for k, v in timings.items()}))
    return SigmaOmegaResult(omega_ry=np.asarray(omega_grid_ry, np.float64),
                            omega_ev=np.asarray(omega_grid_ry, np.float64) * EV,
                            sigma_c_kij=cube, band_axis=axis, band_counts=(), efermi_ry=float(efermi_ry))
