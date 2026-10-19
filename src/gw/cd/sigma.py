"""Sigma_c(omega) by contour deformation on the exact ISDF W(z): the production CD route.

Where this sits (docs/dev/cd_frequency_integration.md): ``gw.sigma_dispatch._compute_mpa_sigma``
calls :func:`compute_sigma_c_cd_omega_grid` instead of the pole executor when the deck says
``sigma_frequency_integration = contour_deformation``; the result is the same
:class:`gw.ppm_sigma.SigmaOmegaResult` the one finalizer consumes.  No physics is rebuilt here:
chi0 at every node and knot is the shared-pole bank's own response quadrature (:func:`bank_chi0`:
``response_bank.response_support`` / ``response_quadrature`` shared-node rules at
``cd_chi0_rule_tolerance``, value rows of ``response_bank.integrate_response_group``,
``ordered=True``; no Lehmann sum), the Dyson pair is the response bank's equation set (W_c = H X (1-X)^-1 H,
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
the pair densities rho_{-q'} from the replicated tables.  A residue at (E, l) reads only the two knots
bracketing |x|, so each knot chunk contracts, per branch, only the (k, l) rows some residue of that
branch reads in the chunk (:func:`residue_band_tables`, host-built once; widths padded to BAND_RUNG
so a handful of shapes compile); every other row would add an exact zero.  FD-9 ghost knots roll from chunk to
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
from lxkit import ordered_psum

from common.units import RYD_TO_EV
from gw.cd import laws
from gw.cd.rule import tan_mapped_gauss_rule

#: Ghost knots per chunk side of the 9-point central d/dx (dchi/ds = (dchi/dx) / (2z)).
GH = 4
FD9 = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
#: Occupation floor (fe_cd): f below it is 0, 1 - f below it is 1 (moves f by <= 1e-14).
OCCUPATION_FLOOR = 1e-14
#: Default relative tolerance of the bank response rule serving chi0 at every CD point
#: (``cd_chi0_rule_tolerance``; the shared-pole production ``bank_rule_tolerance``).
CD_CHI0_RULE_TOLERANCE = 1e-8
CD_RECIPE = dict(version="cd_production_v3", laws="ordered_cd TRREF / Run211 jets",
                 beta_ev=[0.5, 4.0], slope="fd9 rolling ghosts", imaginary_rule="tan_mapped_gauss two-panel",
                 chi0="response_bank shared-node rules (response_support, response_quadrature) + "
                      "integrate_response_group value rows, ordered=True",
                 dyson="response_bank pair equations, local batched jnp.linalg.solve over (q, z)",
                 tile="runtime.tiles.tile_units", real_axis="owner-rank branches, no W_-q gather",
                 residue_rows="banded per knot chunk and branch (live bands, BAND_RUNG 8; exact)")


def _say(print_fn, started, msg):
    print_fn(f"  CD +{time.monotonic() - started:7.1f}s {msg}")


#: Sigma_x identity tolerance (``GATE cd_sigma_x_identity``), meV.  The CD pair densities and V_q reproduce
#: the static exchange to round-off: measured 3.3e-10 meV (Fe 64 k x 26 bands) and 2.1e-11 meV (Si 4x4x4
#: SOC, 8 k x 40 bands), so the design target 1e-3 meV (docs/dev/cd_frequency_integration.md) has six
#: orders of margin and still catches a wrong pair orientation, occupation table or q weight.
SIGMA_X_IDENTITY_TOL_MEV = 1e-3


def sigma_x_identity_check(sx_cd_diag_ry, sx_ref_diag_ry):
    """``(max |Re diag Sigma_x(CD) - Re diag Sigma_x(static)| in meV, note)``; refuses above the tolerance.

    Parameters: two arrays of one shape ``[k, m]`` (Ry, real or complex) -- the CD route's and the
    static route's diagonal exchange on the same rows.  Raises ``ValueError`` (``GATE
    cd_sigma_x_identity``) when the defect exceeds :data:`SIGMA_X_IDENTITY_TOL_MEV` or is not finite.
    """
    dmax = float(np.max(np.abs(np.real(sx_cd_diag_ry) - np.real(sx_ref_diag_ry)))) * float(RYD_TO_EV) * 1e3
    note = f"max|diag Sigma_x(CD) - diag Sigma_x(static)| = {dmax:.3e} meV"
    if not dmax <= SIGMA_X_IDENTITY_TOL_MEV:
        raise ValueError(f"GATE cd_sigma_x_identity: {note}; want: <= {SIGMA_X_IDENTITY_TOL_MEV:g} meV; why: the CD "
                         "pair densities and V_q must reproduce the static exchange (head off)")
    return dmax, note


def ledger_sample(stage, chunk, seconds):
    """One device-ledger row (this process's first device): bytes in use, pool high-water, live arrays.

    The high-water is read WITHOUT a reset.  ``common.timing`` owns the reset (one per section boundary)
    and charges each interval's peak to the section on top of the stack; a reset here, once per chunk,
    left the stage table only the interval after the last row (Fe: CD stage 0.36 GB reported against
    4.30 GB in this ledger).  The value is therefore cumulative since the enclosing section boundary,
    and the leak criterion is that it does not rise after the first chunk of each axis (with
    ``bytes_in_use`` equal between equal-shaped chunks).
    """
    from runtime.xla_memory import pool_high_water
    try:
        st = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # pragma: no cover - CPU backends have no stats
        st = {}
    high = pool_high_water(reset=False)
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


# ------------------------------------------------------------------------------- residue band tables
#: Width rung of the banded residue tables.  A knot chunk's per-branch band count (the most live bands of
#: any (q, k) row) is padded up to a multiple of BAND_RUNG, capped at L, so the real-axis kernel compiles
#: once per distinct (L+, L-) pair instead of once per chunk.  Fe 64 k x 36 bands, 276 chunks: 6 pairs,
#: 35 % of the dense rows contracted (29 % at exact widths, 38 pairs; rung 4: 32 %, 12 pairs).
BAND_RUNG = 8


def band_rung(count, L):
    """Static table width for ``count`` live bands: 0 (branch skipped) or min(L, BAND_RUNG * ceil(count / BAND_RUNG))."""
    count = int(count)
    return 0 if count == 0 else min(int(L), BAND_RUNG * -(-count // BAND_RUNG))


def residue_band_tables(enk_L, f_L, KMQ_pad, qneg_pad, rows, omega, eps_own, h_ry, chunk, n_chunks):
    """Which intermediate bands l each knot chunk's residues read, per occupation branch (host, numpy).

    The Hermite residue of (k, E, l) at x = E - eps_l(k - q) reads only the knots floor(|x|/h) + (0, 1)
    (``laws.HERMITE_STENCIL``), on the x >= 0 branch (weight -(1 - f), pair q, W_q) or the x < 0 branch
    (weight f, pair -q, W_{-q}^T).  A (k, l) row none of whose residues reads a knot of a chunk contributes
    an exact zero to that chunk, so the chunk need contract only its live rows (the integration rule is
    unchanged).  E runs over the omega grid AND the row's own energies (the own-energy diagonal).  The
    stencil is widened by one knot on each side: numpy's |x|/h and XLA's (which may fold the division into
    a reciprocal multiply) can differ in the last ulp, and a superset of the live rows is exact where a
    missed row is not.  x and f are formed by the same arithmetic as :func:`pair_rows`.

    Parameters
    ----------
    enk_L, f_L : [nk, L] float64 -- energies (Ry) and floored occupations of the kernel tables.
    KMQ_pad : [nq_pad, nk] int (k - q rows); qneg_pad : [nq_pad] int; rows : [kw] int (file wedge + certificate).
    omega : [E] (Ry, absolute); eps_own : [kw, m] (Ry); h_ry : knot step; chunk, n_chunks : the knot tiling.

    Returns
    -------
    ``(tables, summary)``.  ``tables(c)`` -> ``(bands, negative)``: a tuple of int32 tables
    [nq_pad, kw, Lb] (one per branch that chunk c reads; Lb = :func:`band_rung` of the chunk's largest
    live count) listing per (q, k) the live bands in ascending order, then dead bands as padding, and the
    matching tuple of branch flags.  Padding needs no mask: a dead band's weights at the chunk's knots are
    exact zeros.  ``summary`` holds the widths and live/contracted fractions for the receipt.  Storage is
    one bit per (branch, q, k, l, chunk).
    """
    enk_L, f_L = np.asarray(enk_L, np.float64), np.asarray(f_L, np.float64)
    rows, omega, eps_own = np.asarray(rows), np.asarray(omega, np.float64), np.asarray(eps_own, np.float64)
    nq_pad, kw, L = int(KMQ_pad.shape[0]), int(rows.size), int(enk_L.shape[1])
    n_knots = int(chunk) * int(n_chunks)
    nbytes = -(-int(n_chunks) // 8)
    packed = np.zeros((2, nq_pad, nbytes, kw, L), np.uint8)
    width = np.zeros((2, int(n_chunks)), np.int64)
    live_rows = np.zeros(2)
    ne = omega.size + eps_own.shape[1]
    qb = max(1, int((64 << 20) // (2 * kw * ne * L * 8)))                 # q block: x below 64 MiB
    pair_g = np.arange(nq_pad)
    pair_all = np.stack([pair_g, np.asarray(qneg_pad)[pair_g]])          # [branch, q]: the pair each branch reads
    for g0 in range(0, nq_pad, qb):
        g1 = min(g0 + qb, nq_pad)
        r = np.asarray(KMQ_pad)[pair_all[:, g0:g1]][..., rows]           # [2, qb, kw]
        e, f = enk_L[r], f_L[r]                                          # [2, qb, kw, L]
        x = np.concatenate([omega[None, None, None, :, None] - e[..., None, :],
                            np.broadcast_to(eps_own[None, None, :, :, None], e.shape[:3] + eps_own.shape[1:] + (1,))
                            - e[..., None, :]], axis=3)                   # [2, qb, kw, E + m, L]
        side = np.stack([x[0] >= 0.0, x[1] < 0.0])
        weight = np.stack([(1.0 - f[0]) != 0.0, f[1] != 0.0])[..., None, :]
        ok = side & weight                                               # live residues of each branch
        b_, q_, k_, _, l_ = np.nonzero(ok)
        lower = laws.hermite_lower_knot(x[ok], h_ry, xp=np).astype(np.int64)
        hit = np.zeros((2, g1 - g0, int(n_chunks), kw, L), bool)
        base = (b_ * (g1 - g0) + q_) * (int(n_chunks) * kw * L) + k_ * L + l_      # flat index at chunk 0
        flat = hit.reshape(-1)
        for d in (-1,) + tuple(laws.HERMITE_STENCIL) + (laws.HERMITE_STENCIL[-1] + 1,):
            knot = lower + d
            sel = (knot >= 0) & (knot < n_knots)
            flat[base[sel] + (knot[sel] // int(chunk)) * (kw * L)] = True
        packed[:, g0:g1] = np.packbits(hit, axis=2)
        counts = hit.sum(axis=-1)                                        # [2, qb, n_chunks, kw]
        width = np.maximum(width, counts.max(axis=(1, 3)))
        live_rows += counts.sum(axis=(1, 2, 3))

    widths = np.array([[band_rung(width[b, c], L) for c in range(int(n_chunks))] for b in (0, 1)])

    def tables(c):
        bits = (packed[:, :, c >> 3] >> (7 - (c & 7))) & 1               # [2, nq_pad, kw, L]
        bands, negative = [], []
        for b in (0, 1):
            if widths[b, c]:
                order = np.argsort(1 - bits[b], axis=-1, kind="stable")[..., :widths[b, c]]
                bands.append(order.astype(np.int32))
                negative.append(b == 1)
        return tuple(bands), tuple(negative)

    dense = float(nq_pad * kw * L * n_chunks)
    pairs = {}
    for c in range(int(n_chunks)):
        key = f"{widths[0, c]}/{widths[1, c]}"
        pairs[key] = pairs.get(key, 0) + 1
    summary = dict(rung=BAND_RUNG, L=L, n_chunks=int(n_chunks), live_fraction=[float(v / dense) for v in live_rows],
                   contracted_fraction=[float(widths[b].sum() / (L * n_chunks)) for b in (0, 1)],
                   empty_chunks=[int((widths[b] == 0).sum()) for b in (0, 1)], widths_pos_neg=pairs)
    return tables, summary


# ------------------------------------------------------------------------------- pair-basis kernels
def pair_rows(T, q, bands=None):
    """Pair densities and energy tables of pair q (k -> k - q) at the file-wedge rows.

    T (replicated): psi_L [nk, L, s, n], psi_M [kw, m, s, n], KMQ [nq_pad, nk], kw [kw],
    enk [nk, L], f [nk, L], omega [E], eps_own [kw, m].  ``bands`` [kw, Lb] int32 (optional): the
    intermediate bands to keep per row (a :func:`residue_band_tables` table at this q); all L if None.
    Returns rho [kw, m, Lb, n] = conj(psi_{k-q,l}) psi_{k,m} summed over spinor, x [kw, E, Lb],
    x_own [kw, m, Lb] (Ry, omega - eps_l(k-q)), f [kw, Lb].
    """
    rows = T["KMQ"][q][T["kw"]]
    if bands is None:
        psi_l, e, f = T["psi_L"][rows], T["enk"][rows], T["f"][rows]
    else:
        psi_l, e, f = (T[key][rows[:, None], bands] for key in ("psi_L", "enk", "f"))
    rho = jnp.einsum("klsu,kmsu->kmlu", jnp.conj(psi_l), T["psi_M"])
    x = T["omega"][None, :, None] - e[:, None, :]
    x_own = T["eps_own"][:, :, None] - e[:, None, :]
    return rho, x, x_own, f


def contract(rho, op, nk):
    """a[k, m, n, l] = -(1/nk) sum_uv conj(rho[k,m,l,u]) op[u,v] rho[k,n,l,v]   (class C then D)."""
    t = jnp.einsum("kmlu,uv->kmlv", jnp.conj(rho), op)
    return -jnp.einsum("kmlv,knlv->kmnl", t, rho) / nk


def build_contraction_kernels(mesh_xy, *, nk, nq_loc, eta, h_ry, beta):
    """The four q-parallel CD kernels (jitted shard_maps; tables T and accumulators replicated).

    sigma_x(V [nq_pad, n, n] q-par, acc [kw, m, m], T)                     -> acc + Sigma_x
    imag(W [nz, nq_pad, n, n] q-par, u [nz], w [nz], acc [E,kw,m,m], own [kw,m], T) -> (acc', own')
    real(W, dW, z [nz], node0, acc, own, T, bands, negative)                -> (acc', own')
    anchor(W0, dW0 [nq_pad, n, n] at z = i eta, acc, own, T)                -> (acc', own')
    Every kernel returns its input accumulator plus the psum over ranks of its own increment.
    ``real`` contracts only the banded rows of the chunk: ``bands`` are the replicated int32 tables
    [nq_pad, kw, Lb] of :func:`residue_band_tables` for the branches that read the chunk, ``negative``
    their branch flags (static).  Full tables (every l, both branches) give the dense contraction.
    """
    axes = ("x", "y")

    def q0():
        return (jax.lax.axis_index("x") * mesh_xy.shape["y"] + jax.lax.axis_index("y")) * nq_loc

    def psum_add(acc, inc):
        # rank-ordered adds (lxkit.ordered_psum; tests/test_reduction_lint.py): same sum, bits
        # independent of the GPU placement
        return acc + ordered_psum(inc, axes)

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

    def cd_real_contract(W, dW, z_c, node0, acc, own, T, bands, negative):
        """One knot chunk's Hermite residues on the banded rows of ``bands`` (one table per live branch).

        ``negative`` (static, one flag per table): False = x >= 0 residues of pair g against W_g,
        True = x < 0 residues of pair -g against W_g^T.  Per (chunk, q) the pair densities are gathered
        on the table's bands only and the Hermite coefficients are built once; per knot only the
        ``index == node`` selection, the contractions and the apply remain.
        """
        def over_q(carry, iq):
            g = q0() + iq
            sides = []
            for table, neg in zip(bands, negative):
                rho, x, x_own, f = pair_rows(T, T["qneg"][g] if neg else g, table[g])
                sides.append((rho, laws.hermite_branch_coefficients(x, f, h_ry, neg),
                              laws.hermite_branch_coefficients(x_own, f, h_ry, neg), neg))

            def over_z(c2, iz):
                node, z = node0 + iz, z_c[iz]
                Wz, dWz = W[iz, iq], dW[iz, iq]
                ops, wts, inc_own = [], [], 0
                for rho, coeffs, coeffs_own, neg in sides:
                    op = laws.real_knot_operators(z, contract(rho, Wz.T if neg else Wz, nk),
                                                  contract(rho, dWz.T if neg else dWz, nk))
                    ops.append(op)
                    wts.append(laws.real_knot_select(node, coeffs))
                    inc_own = inc_own + laws.own_apply(op, laws.real_knot_select(node, coeffs_own))
                inc = laws.apply(jnp.concatenate(ops, -1), jnp.concatenate(wts, -1))
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

    def wrap(fn, in_specs, out_specs, name=None):
        sm = shard_map(fn, mesh=mesh_xy, in_specs=in_specs, out_specs=out_specs, check_vma=False)

        def named(*args):
            return sm(*args)
        named.__name__ = named.__qualname__ = name or fn.__name__
        return jax.jit(named)

    real_programs = {}

    def real(W, dW, z_c, node0, acc, own, T, bands, negative):
        """Banded real-axis kernel; one program per branch-flag tuple, jit-cached per table width."""
        negative = tuple(bool(v) for v in negative)
        if len(negative) != len(bands) or not negative:
            raise ValueError(f"GATE cd_band_tables: {len(bands)} tables vs branch flags {negative}; want one flag per "
                             "table and at least one table (a chunk no residue reads is skipped by the caller)")
        if negative not in real_programs:
            def body(W, dW, z_c, node0, acc, own, T, bands):
                return cd_real_contract(W, dW, z_c, node0, acc, own, T, bands, negative)
            real_programs[negative] = wrap(body, (q4, q4, P(), P(), P(), P(), P(), P()), (P(), P()),
                                           name="cd_real_contract")
        return real_programs[negative](W, dW, z_c, node0, acc, own, T, tuple(bands))

    return SimpleNamespace(
        sigma_x=wrap(cd_sigma_x, (q3, P(), P()), P()),
        imag=wrap(cd_imag_contract, (q4, P(), P(), P(), P(), P()), (P(), P())),
        real=real,
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


# ------------------------------------------------------------------------------- chi0 provider
def bank_chi0(wfns, meta, mesh_xy, z_ry, *, q_ids, mu_ry, rule_tolerance, slots, print_fn=print):
    """chi0 at the CD points from the shared-pole bank's own response quadrature.

    ``z_ry`` lists every point the CD streams evaluate (imaginary nodes ``i u`` and real knots
    ``j h + i eta`` with their FD-9 ghosts; Im z > 0; a repeat is evaluated once).  One rule set serves
    all of them, built
    exactly as the bank builds its sample rules: ``response_bank.response_support`` (occupation
    weights, transition interval, envelope, band support) and ``response_bank.response_quadrature``
    (``minimax.ResponseGroupTree`` over every point, ``rel_tol = rule_tolerance / amplitude``, groups
    halved until each meets it).  chi0(q, z) is the value row of
    ``response_bank.integrate_response_group`` (``ordered=True``: the physical ``FT_q[chi]`` at the
    full-grid rows ``q_ids``), one Green-pair scan per rule group a chunk touches.  A chunk's members
    are taken out of their rule groups by index; every other rule key is the bank's.  No Lehmann sum.

    ``mu_ry`` is the current chemical potential; ``partial_at_mu`` follows the census rule of
    ``shared_pole_recipe.bind_shared_pole_census`` (a partially occupied band crossing mu).  ``slots``
    is the largest chunk.  Returns ``(chi, note)``: ``chi(points)`` -> ``[nz, nq, n, n]`` at
    ``P(None, None, 'x', 'y')`` for any sequence of points of ``z_ry`` (repeats allowed; a repeat is a
    copy); ``note`` is the rule receipt, updated as chunks stream.
    """
    from common.collectives import device_put_process_local_cached, gather_to_host
    from gw import response_bank as rb

    EV = float(RYD_TO_EV)
    index = {}
    for z in np.asarray(z_ry, dtype=np.complex128).ravel().tolist():     # one evaluation per distinct point
        index.setdefault(complex(z), len(index))
    z_ry = np.asarray(list(index), dtype=np.complex128)
    if not z_ry.size or not np.all(np.isfinite(z_ry)) or not np.all(z_ry.imag > 0):
        raise ValueError("GATE cd_chi0_points: got: an empty, nonfinite or non-causal CD point set; want: "
                         "finite points with Im z > 0; why: the bank rule fits upper-half-plane evaluations")
    if getattr(meta, "shared_pole_response_rules", None) is not None:
        raise ValueError("GATE cd_chi0_session: got: a held shared-pole rule session on meta; want: none; "
                         "why: the CD rule must neither reuse nor overwrite the bank's held-map rules")
    stop = int(wfns.slices.b4_logical) - int(wfns.slices.b0)
    energy = np.asarray(gather_to_host(wfns.enk), dtype=np.float64)[:, :stop]
    occupied = np.asarray(gather_to_host(wfns.occ), dtype=np.float64)[:, :stop]
    crossing = (energy.min(axis=0) <= mu_ry) & (energy.max(axis=0) >= mu_ry)
    partial = np.any((occupied > 0.0) & (occupied < 1.0), axis=0)
    plan = dict(z_ry=z_ry, distinct_id=np.arange(z_ry.size), held=np.zeros(z_ry.size, dtype=bool),
                role=np.full(z_ry.size, "cd_point"), bank_rule_tolerance=float(rule_tolerance),
                census=dict(mu_ry=float(mu_ry), partial_at_mu=bool(np.any(partial & crossing))))
    receipt = {"correlation_count": 0}
    support = rb.response_support(wfns, meta, plan, receipt, print_fn=print_fn)
    try:
        rules = rb.response_quadrature(meta, plan, receipt, support, group_size=int(slots), print_fn=print_fn)
    except ValueError as exc:
        raise ValueError(
            f"GATE cd_chi0_rule: got: the bank response rule refused the CD points ({exc}); smallest Im z "
            f"{z_ry.imag.min() * EV:.4g} eV over {z_ry.size} points, transition interval "
            f"[{support['lo'] * EV:.4g}, {support['hi'] * EV:.4g}] eV, tolerance {rule_tolerance:g}; want: a rule "
            "meeting cd_chi0_rule_tolerance at every point; why: chi0 has one provider, the bank quadrature "
            "(its pencil grows like (hi - lo) / min Im z)") from exc
    groups = rules["plan"]["rule_groups"]
    sliced = ("value", "derivative", "sampled_error", "coefficient_mass")
    note = dict(provider="gw.response_bank shared-node quadrature: response_support + response_quadrature "
                         "rules, integrate_response_group value rows (ordered=True)",
                rule_tolerance=float(rule_tolerance), points=int(z_ry.size),
                interval_ev=[float(v) * EV for v in receipt["rule"]["interval_ry"]],
                occupation_amplitude=float(receipt["rule"]["occupation_amplitude"]),
                decay_rate_ry_inv=float(receipt["rule"]["decay_rate_ry_inv"]),
                partial_at_mu=plan["census"]["partial_at_mu"], band_ranges=list(support["band_ranges"]),
                rule_nodes=int(sum(int(g["count"]) for g in groups)),
                rule_groups=[dict(points=len(g["members"]), nodes=int(g["count"]),
                                  sampled_error=float(np.max(g["sampled_error"])),
                                  re_ev=[float(z_ry[list(g["members"])].real.min() * EV),
                                         float(z_ry[list(g["members"])].real.max() * EV)],
                                  im_ev=[float(z_ry[list(g["members"])].imag.min() * EV),
                                         float(z_ry[list(g["members"])].imag.max() * EV)]) for g in groups],
                passes=0, green_pair_evaluations=0)
    q_ids = tuple(int(q) for q in np.asarray(q_ids).ravel())
    n = int(meta.mu_basis.n_packed)
    face4 = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    rep = NamedSharding(mesh_xy, P())

    def execute(kernel, args, stage, runtime_bytes=0):
        """The bank's compile cache (``response_bank._compiled``); the CD tiles own the memory plan."""
        result = rb._compiled(kernel, args)(*args)
        return kernel.finish(args, result) if hasattr(kernel, "finish") else result

    def _place(out, carry, src, dst):
        return out.at[dst].set(carry[2 * src], mode="drop")              # carry rows 2r: member r's value
    place = jax.jit(_place, out_shardings=face4, donate_argnums=(0,))
    blank = {}

    def chi(points):
        try:
            ids = [index[complex(z)] for z in np.asarray(points, dtype=np.complex128).ravel()]
        except KeyError as exc:
            raise ValueError(f"GATE cd_chi0_point_unplanned: got: z = {exc} Ry outside the planned CD points; "
                             "want: a point of the rule set; why: one rule set serves every chunk") from None
        want = set(ids)
        nz = len(ids)
        if nz not in blank:
            blank[nz] = jax.jit(lambda: jnp.zeros((nz, len(q_ids), n, n), jnp.complex128), out_shardings=face4)
        out = blank[nz]()
        for rule in groups:
            hit = [p for p, m in enumerate(rule["members"]) if m in want]
            for a in range(0, len(hit), int(slots)):
                part = hit[a:a + int(slots)]
                sub = dict(rule, members=[rule["members"][p] for p in part],
                           **{key: np.asarray(rule[key])[part] for key in sliced})
                carry = rb.integrate_response_group(wfns, meta, mesh_xy, rules, sub, q_ids=q_ids,
                                                    execute=execute, receipt=receipt, ordered=True)
                row = {m: r for r, m in enumerate(sub["members"])}
                pairs = [(row[i], d) for d, i in enumerate(ids) if i in row]
                src = np.zeros(nz, np.int32)
                dst = np.full(nz, nz, np.int32)                                  # out of range: dropped
                src[:len(pairs)], dst[:len(pairs)] = zip(*pairs)
                out = place(out, carry, device_put_process_local_cached(src, rep),
                            device_put_process_local_cached(dst, rep))
                del carry
                note["passes"] += 1
        note["green_pair_evaluations"] = int(receipt["correlation_count"])
        return out
    chi.rules, chi.points = rules, z_ry                # the bank rule set and its distinct points (tests)
    return chi, note


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
    from gw.w_isdf import _w_solve_pref_scalar
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
        if f_state.shape != tuple(wfns.occ.shape):
            raise ValueError(f"GATE cd_occupations_shape: OccupationState {f_state.shape} vs carrier {tuple(wfns.occ.shape)}")
        wfns = dataclasses.replace(wfns, occ=replicate_to_mesh(f_state, mesh_xy))
    else:
        f_state = np.asarray(host(wfns.occ), dtype=np.float64)
        mu = float(efermi_ry)
    if f_state.shape[1] < nb_full:
        f_state = np.pad(f_state, ((0, 0), (0, nb_full - f_state.shape[1])))
    f_cd = np.where(f_state < OCCUPATION_FLOOR, 0.0, np.where(1.0 - f_state < OCCUPATION_FLOOR, 1.0, f_state))
    slices = wfns.slices
    sig = slices.sigma
    lsum = slices.sigma_sum if slices.sigma_sum.stop else slices.full
    m0, m1 = int(sig.start or 0), int(sig.stop)
    l0, l1 = int(lsum.start or 0), int(lsum.stop)
    nb_s, L = m1 - m0, l1 - l0
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
    t_bands = time.monotonic()
    band_tables, band_summary = residue_band_tables(enk_L, f_L, KMQ_pad, qneg_pad, rows_all, omega, eps_own,
                                                    h_ry, chunk_r, J_pad // chunk_r)
    band_summary["build_s"] = round(time.monotonic() - t_bands, 3)
    _say(print_fn, started, "residue band tables (the rows each knot chunk contracts): " + json.dumps(band_summary))

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

    # chi0 at every CD point (imaginary nodes; real knots -GH .. J_pad + GH - 1 with the FD-9 ghosts)
    # from one bank response rule set (bank_chi0); the chunks below take their points from it.
    t0 = time.monotonic()
    z_cd = np.concatenate([1j * u_ry_all, h_ry * np.arange(-GH, J_pad + GH) + 1j * eta])
    bank_chi, chi_rule = bank_chi0(
        wfns, meta, mesh_xy, z_cd, q_ids=q_irr, mu_ry=mu,
        rule_tolerance=float(cd_handle.get("chi0_rule_tolerance", CD_CHI0_RULE_TOLERANCE)),
        slots=max(chunk_i, chunk_r + 2 * GH), print_fn=print_fn)
    clock("chi0_rule", t0)
    _say(print_fn, started, f"chi0 rule: {chi_rule['points']} points in {len(chi_rule['rule_groups'])} groups, "
         f"{chi_rule['rule_nodes']} nodes, tolerance {chi_rule['rule_tolerance']:.1e}, max sampled error "
         f"{max(g['sampled_error'] for g in chi_rule['rule_groups']):.2e}")

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
                _, sx_note = sigma_x_identity_check(np.diagonal(sx_cd_h, axis1=1, axis2=2),
                                                    np.diagonal(ref, axis1=1, axis2=2))
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
        chi = bank_chi(1j * u_ry[c0:c1])
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
            chi_e = bank_chi(h_ry * np.arange(j0 - GH, j1 + GH) + 1j * eta)
        else:
            chi_e = join_fn(tail, bank_chi(h_ry * np.arange(j0 + GH, j1 + GH) + 1j * eta))
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
        bands, negative = band_tables(ic)
        if bands:                                       # a chunk no live residue reads adds an exact zero
            acc_real, own_real = K.real(W, dW, z_core, jnp.int32(j0), acc_real, own_real, tabs,
                                        tuple(_rep(b) for b in bands), negative)
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
                   full_bz=full_bz, kw_rows=kw_rows.tolist(), finite=finite, ndev=ndev, symmetry_certificate=cert_note,
                   residue_bands=band_summary, chi0_rule=chi_rule)
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
