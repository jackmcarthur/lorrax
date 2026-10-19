"""Sigma_c(omega) by contour deformation on the ISDF W(z) (bank-quadrature chi0): the production CD route.

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

    chi(z) [nz, nq_irr, n, n] face --(q,z)-major flat, zero rows to a multiple of P, local batched LU-->
      W_c, dW_c/ds --unfold per z (wedge decks)--> [nz, nq_pad, n, n] q-parallel P(None, ('x','y'))
      --shard_map over q: scan over the rank's q, scan over band tiles, scan over z--> accumulators [E, kw, m, n]

A chunk may hold any number of nodes or knots: the (q, z) flat axis is padded with zero rows (their
W_c is exactly zero and is dropped) to a multiple of P, so the chunk is not rounded up to
P / gcd(nq_irr, P).  The pair densities rho[kw, m, l, n] exist one band tile at a time
(:func:`tile_sum`; ``tile_units`` of :func:`band_bytes_per_unit`), and a tile that covers every band
runs the untiled program bit for bit.  psi enters as per-rank pair slabs, q-sharded like W
(:func:`pair_psi_slabs`: the rows k - q of the rank's own pairs, and of the pairs -q up to the last
band an x < 0 residue reads), streamed from the band carrier one band block at a time; the Coulomb
roots stay on the face.  Nothing of psi or of the roots is replicated whole (:func:`cd_memory_plan`).

The real axis needs no W_{-q} gather: the rank holding W_q' adds pair q''s x >= 0 residues
(contractions with W_q') and pair (-q')'s x < 0 residues (contractions with W_q'^T), recomputing
the pair densities rho_{-q'} from its pair slab.  A residue at (E, l) reads only the two knots
bracketing |x|, so each knot chunk contracts, per branch, only the (k, l) rows some residue of that
branch reads in the chunk (:func:`residue_band_tables`, host-built once; widths padded to BAND_RUNG
so a handful of shapes compile); every other row would add an exact zero.  FD-9 ghost knots roll from chunk to
chunk (chi0 runs once per knot).  The q axis is padded to a multiple of P with zero W rows
(``qvalid`` masks them).  Accumulators are replicated; each kernel returns acc + psum(increment).
The two chunk loops are :func:`stream_cd_chunks`; ``cd_checkpoint`` keeps the replicated accumulators
after every chunk (:class:`CdCheckpoint`, ``file_io.sigma_checkpoint``), and a resume re-runs only the
chi0 calls that rebuild the ghost tail (:func:`real_replay_start`), so it reproduces the uninterrupted
accumulators.

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
from gw.cd import chi0_rules as cd_rules
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
                 residue_rows="banded per knot chunk and branch (live bands, BAND_RUNG 8; exact)",
                 band_tiles="tile_sum over l (runtime.tiles.tile_units; exact up to summation order)",
                 psi_tables="q-sharded pair slabs streamed from the band carrier (pair_psi_slabs)",
                 qz_flat="zero rows to a multiple of P (chunk free of P / gcd(nq_irr, P))")


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


def band_bytes_per_unit(kw, m, n, ns, n_omega):
    """Per-rank bytes one intermediate band l adds to a contraction kernel's band tile (complex128).

    16 kw [n (3 m + s) + 8 m^2 + 12 E]: the pair density rho and two contraction intermediates
    [kw, m, n] (the anchor's value and slope, the real axis' W and dW/ds), the psi rows [kw, s, n], the
    m x m contractions with their daggers and concatenations, and the per-(k, E) weights (imaginary
    selectors, anchor and Hermite coefficients).  :func:`build_contraction_kernels` takes the tile as
    ``tile_units(band_bytes_per_unit(...), L)``.
    """
    return 16.0 * kw * (n * (3 * m + ns) + 8 * m * m + 12 * n_omega)


def knot_chunk(units, count, *, ndev, nq_irr):
    """Nodes or knots per chunk from ``units`` (the deck's cd_knot_chunk, else the planner's ``tile_units``).

    At least one, at most ``count``.  A chunk of at least zq_mult = P / gcd(nq_irr, P) units is rounded up
    to a multiple of zq_mult, as before (its (q, z) flat axis divides P, no zero rows); a smaller chunk is
    kept, and :func:`qz_flat_programs` pads its flat axis with zero rows.  Before, every chunk was rounded
    up to zq_mult: 8 knots at P = 64 with 8 irreducible q, whatever the per-knot bytes.
    """
    zq_mult = int(ndev) // int(np.gcd(int(nq_irr), int(ndev)))
    units = int(max(1, min(int(units), int(count))))
    return units if units < zq_mult else -(-units // zq_mult) * zq_mult


def cd_memory_plan(*, n, nq_irr, nk, n_carrier_k, nb_carrier, n_rows, nb_s, L, L_neg, ns, n_omega, ndev,
                   mesh_y, chunk_imag, chunk_real, band_tile, slab_block, wedge):
    """Per-rank bytes of the CD stage by object and phase, from the shapes alone (complex128 = 16 B).

    The planner of record for the memory a CD run needs on one rank; it reads no device state.  Phases:
    ``setup`` (pair-slab build: the carrier on the ``axis`` layout and one replicated band block),
    ``sigma_x``, ``imag`` (one node chunk: chi face, the padded (q, z) flats of chi, roots and W_c, the
    LU workspace, then W on the full zone), ``real`` (one knot chunk: the rolling FD-9 ghost tail, chi,
    dchi/ds, the Dyson pair, W and dW/ds on the full zone).  Every phase adds the resident objects (pair
    slabs, psi_M, accumulators, Coulomb roots on the face, energy tables) and one band tile.  The
    batched-LU workspace is an estimate: 4 (value) / 5 (pair) n x n matrices per local flat row.  Objects
    the CD stage does not own (the psi carrier, V_q) are not counted.

    Returns ``dict(resident=..., phases=..., peak_bytes=..., peak_phase=..., layout=...)``, bytes per rank.
    """
    from runtime.padding import padded_axis
    P = int(ndev)
    u = 16.0 * n * n
    nq_pad = -(-nk // P) * P
    nq_loc = nq_pad // P

    def flat_rows(nz):           # local rows of the zero-padded (q, z) flat axis
        return padded_axis(nq_irr * nz, P, name="cd (q, z) flat").carrier // P

    resident = dict(
        psi_pair_slab=16.0 * nq_loc * n_rows * L * ns * n,
        psi_neg_pair_slab=16.0 * nq_loc * n_rows * L_neg * ns * n,
        psi_M=16.0 * n_rows * nb_s * ns * n,
        accumulators=3 * 16.0 * (n_omega * n_rows * nb_s * nb_s + n_rows * nb_s),
        coulomb_roots_face=u * nq_irr / P,
        energy_tables=16.0 * nk * L,
        band_tile=band_tile * band_bytes_per_unit(n_rows, nb_s, n, ns, n_omega))
    base = sum(resident.values())
    wedge_tr = (u * nk / P) if wedge else 0.0              # one z's full-zone face from QirrOperator.unfold
    zi, zr = int(chunk_imag), int(chunk_real)
    fi, fr = flat_rows(zi), flat_rows(zr)
    tail = 2 * GH * nq_irr * u / P
    phases = dict(
        setup=base - resident["band_tile"] + 16.0 * n_carrier_k * nb_carrier * ns * n / mesh_y
        + 16.0 * slab_block * ns * n * (n_carrier_k + nk + 2 * nq_loc * n_rows),
        sigma_x=base + u * nk / P + u * nq_loc,
        imag_dyson=base + zi * nq_irr * u / P + fi * u * (3 + 4),
        imag_unfold=base + fi * u + zi * nq_loc * u * (2 if wedge else 1) + wedge_tr,
        imag_contract=base + zi * nq_loc * u,
        real_chi=base + tail + (2 * zr + 2 * GH) * nq_irr * u / P,
        real_slope=base + tail + (zr + 2 * GH + 2 * zr) * nq_irr * u / P,
        real_dyson=base + tail + 2 * zr * nq_irr * u / P + fr * u * (5 + 5),
        real_unfold=base + tail + 2 * fr * u + 2 * zr * nq_loc * u * (1.5 if wedge else 1) + wedge_tr,
        real_contract=base + tail + 2 * zr * nq_loc * u)
    peak_phase = max(phases, key=phases.get)
    layout = dict(P=P, n=n, nq_pad=nq_pad, nq_loc=nq_loc, chunk_imag=zi, chunk_real=zr,
                  flat_rows_local_imag=fi, flat_rows_local_real=fr, band_tile=int(band_tile), slab_block=int(slab_block),
                  L=int(L), L_neg=int(L_neg))
    return dict(resident=resident, phases=phases, peak_bytes=phases[peak_phase], peak_phase=peak_phase,
                layout=layout)


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
    exact zeros.  ``summary`` holds the widths and live/contracted fractions for the receipt, and
    ``extent_pos_neg``: per branch, every table entry is below it (the pair slab of the x < 0 branch
    holds only those bands).  Storage is one bit per (branch, q, k, l, chunk).
    """
    enk_L, f_L = np.asarray(enk_L, np.float64), np.asarray(f_L, np.float64)
    rows, omega, eps_own = np.asarray(rows), np.asarray(omega, np.float64), np.asarray(eps_own, np.float64)
    nq_pad, kw, L = int(KMQ_pad.shape[0]), int(rows.size), int(enk_L.shape[1])
    n_knots = int(chunk) * int(n_chunks)
    nbytes = -(-int(n_chunks) // 8)
    packed = np.zeros((2, nq_pad, nbytes, kw, L), np.uint8)
    width = np.zeros((2, int(n_chunks)), np.int64)
    live_rows = np.zeros(2)
    top_live = np.full(2, -1, np.int64)                                  # highest live band per branch
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
        any_l = hit.reshape(2, -1, L).any(axis=1)                        # [2, L]
        top_live = np.maximum(top_live, np.where(any_l.any(axis=1), L - 1 - np.argmax(any_l[:, ::-1], axis=1), -1))
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
    # Every entry of a branch's tables lies below extent = max(top live band + 1, widest table): a row
    # lists its live bands, then the lowest dead ones, and the j-th lowest dead band is <= j + #live.
    extent = [int(max(top_live[b] + 1, widths[b].max(initial=0))) for b in (0, 1)]
    summary = dict(rung=BAND_RUNG, L=L, n_chunks=int(n_chunks), live_fraction=[float(v / dense) for v in live_rows],
                   contracted_fraction=[float(widths[b].sum() / (L * n_chunks)) for b in (0, 1)],
                   empty_chunks=[int((widths[b] == 0).sum()) for b in (0, 1)], widths_pos_neg=pairs,
                   extent_pos_neg=extent)
    return tables, summary


# ------------------------------------------------------------------------------- pair-basis kernels
#: Keys of the per-rank pair slabs in the table dict T (:func:`pair_psi_slabs`): q-sharded like W.
#: Every other table is replicated.
PAIR_SLAB_KEYS = ("psi_q", "psi_nq")


def table_specs(T, axes=("x", "y")):
    """shard_map specs of the table dict T: the pair slabs on the q axis, every other table replicated."""
    return {key: (P(axes) if key in PAIR_SLAB_KEYS else P()) for key in T}


def pair_rows(T, q, psi_rows, lo=0, width=None, bands=None):
    """Pair densities and energy tables of pair q (k -> k - q) at the file-wedge rows, on one band tile.

    T (replicated): psi_M [kw, m, s, n], KMQ [nq_pad, nk], kw [kw], enk [nk, L], f [nk, L], omega [E],
    eps_own [kw, m].  ``psi_rows`` [kw, Ls, s, n] is this rank's pair slab of pair q (psi at the rows
    KMQ[q][kw], bands from 0).  The bands kept are [lo, lo + width) (``lo`` may be traced; every band
    of the slab when ``width`` is None), or ``bands`` [kw, Lb] int32 when given: per row the bands to
    keep (a :func:`residue_band_tables` table at this q, or a column tile of one).
    Returns rho [kw, m, Lb, n] = conj(psi_{k-q,l}) psi_{k,m} summed over spinor, x [kw, E, Lb],
    x_own [kw, m, Lb] (Ry, omega - eps_l(k - q)), f [kw, Lb].
    """
    rows = T["KMQ"][q][T["kw"]]
    if bands is None:
        psi_l, e, f = psi_rows, T["enk"][rows], T["f"][rows]
        if width is not None:
            psi_l, e, f = (jax.lax.dynamic_slice_in_dim(a, lo, width, axis=1) for a in (psi_l, e, f))
    else:
        psi_l = psi_rows[jnp.arange(bands.shape[0])[:, None], bands]
        e, f = (T[key][rows[:, None], bands] for key in ("enk", "f"))
    rho = jnp.einsum("klsu,kmsu->kmlu", jnp.conj(psi_l), T["psi_M"])
    x = T["omega"][None, :, None] - e[:, None, :]
    x_own = T["eps_own"][:, :, None] - e[:, None, :]
    return rho, x, x_own, f


def contract(rho, op, nk):
    """a[k, m, n, l] = -(1/nk) sum_uv conj(rho[k,m,l,u]) op[u,v] rho[k,n,l,v]   (class C then D)."""
    t = jnp.einsum("kmlu,uv->kmlv", jnp.conj(rho), op)
    return -jnp.einsum("kmlv,knlv->kmnl", t, rho) / nk


def tile_sum(body, count, tile, zero):
    """sum over the band tiles [lo, lo + width) of [0, count) of ``body(lo, width)``, in tile order.

    ``tile`` None or >= count: the one call ``body(0, None)`` (the untiled program, bit for bit).
    Otherwise a ``lax.scan`` over the count // tile full tiles (``lo`` traced, ``width = tile``) from the
    pytree ``zero``, then, when tile does not divide count, one tail call (static ``lo``, ``width =
    count % tile``): two tile shapes whatever the count.  ``body`` returns a pytree shaped like ``zero``.
    The l sum is the untiled one regrouped by tile (exact up to summation order).
    """
    count = int(count)
    if tile is None or int(tile) >= count:
        return body(0, None)
    tile = int(tile)
    n_full, tail = divmod(count, tile)

    def add(a, b):
        return jax.tree_util.tree_map(jnp.add, a, b)

    def step(carry, t):
        return add(carry, body(t * tile, tile)), None
    out, _ = jax.lax.scan(step, zero, jnp.arange(n_full, dtype=jnp.int32), unroll=1)
    return add(out, body(n_full * tile, tail)) if tail else out


def build_contraction_kernels(mesh_xy, *, nk, nq_loc, eta, h_ry, beta, band_tile=None, own0=0):
    """The four q-parallel CD kernels (jitted shard_maps; accumulators and tables replicated, pair slabs q-sharded).

    sigma_x(V [nq_pad, n, n] q-par, acc [kw, m, m], T)                     -> acc + Sigma_x
    imag(W [nz, nq_pad, n, n] q-par, u [nz], w [nz], acc [E,kw,m,m], own [kw,m], T) -> (acc', own')
    real(W, dW, z [nz], node0, acc, own, T, bands, negative)                -> (acc', own')
    anchor(W0, dW0 [nq_pad, n, n] at z = i eta, acc, own, T)                -> (acc', own')
    Every kernel returns its input accumulator plus the psum over ranks of its own increment.
    T["psi_q"] and T["psi_nq"] are the pair slabs of :func:`pair_psi_slabs` (q on ('x', 'y')): pair q's
    rows over every band of the l sum, and pair -q's rows over the bands the x < 0 tables read.
    The l sum runs ``band_tile`` bands at a time (:func:`tile_sum`; on the real axis, ``band_tile``
    columns of each branch's table): pair densities exist one tile at a time.  None, or a tile no
    narrower than the band count (the widest table), is the untiled program bit for bit.
    ``real`` contracts only the banded rows of the chunk: ``bands`` are the replicated int32 tables
    [nq_pad, kw, Lb] of :func:`residue_band_tables` for the branches that read the chunk, ``negative``
    their branch flags (static).  Full tables (every l, both branches) give the dense contraction.
    ``own0`` (static, ``cd_own_energy_band_min``): the own-energy diagonal covers window rows m >= own0
    (``own`` [kw, m - own0]; T["eps_own"] holds those rows only); the omega-grid cube keeps every row.
    0 is the whole window, the same program as before the key existed.
    """
    axes = ("x", "y")
    tile = int(band_tile) if band_tile else None
    # The own-energy terms read the diagonal of operators built on the whole window; the kept block
    # [own0:, own0:] is the diagonal's tail, so slicing the operator is exact (laws.own_apply unchanged).
    own_apply = laws.own_apply if not own0 else (lambda a, w: laws.own_apply(a[:, own0:, own0:], w))

    def q0():
        return (jax.lax.axis_index("x") * mesh_xy.shape["y"] + jax.lax.axis_index("y")) * nq_loc

    def psum_add(acc, inc):
        # rank-ordered adds (lxkit.ordered_psum; tests/test_reduction_lint.py): same sum, bits
        # independent of the GPU placement
        return acc + ordered_psum(inc, axes)

    def zeros2(acc, own):
        return jnp.zeros_like(acc), jnp.zeros_like(own)

    def cd_sigma_x(V, acc, T):
        def over_q(s, iq):
            g = q0() + iq

            def band_tile_(lo, width):
                rho, _, _, f = pair_rows(T, g, T["psi_q"][iq], lo, width)
                return jnp.einsum("kmnl,kl->kmn", contract(rho, V[iq], nk), f)
            part = tile_sum(band_tile_, T["psi_q"].shape[2], tile, jnp.zeros_like(acc))
            return s + T["qvalid"][g] * part, None
        inc, _ = jax.lax.scan(over_q, jnp.zeros_like(acc), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc)

    def cd_imag_contract(W, u_c, w_c, acc, own, T):
        def over_q(carry, iq):
            g = q0() + iq

            def band_tile_(lo, width):
                rho, x, x_own, f = pair_rows(T, g, T["psi_q"][iq], lo, width)

                def over_z(c2, iz):
                    a = contract(rho, W[iz, iq], nk)
                    ops, wts = laws.imag_node_terms(a, u_c[iz], w_c[iz], x, f, eta)
                    _, wts_own = laws.imag_node_terms(a, u_c[iz], w_c[iz], x_own, f, eta)
                    return (c2[0] + laws.apply(ops, wts), c2[1] + own_apply(ops, wts_own)), None
                out, _ = jax.lax.scan(over_z, zeros2(acc, own), jnp.arange(W.shape[0]), unroll=1)
                return out
            s_acc, s_own = tile_sum(band_tile_, T["psi_q"].shape[2], tile, zeros2(acc, own))
            v = T["qvalid"][g]
            return (carry[0] + v * s_acc, carry[1] + v * s_own), None
        (inc, inc_own), _ = jax.lax.scan(over_q, zeros2(acc, own), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc), psum_add(own, inc_own)

    def cd_real_contract(W, dW, z_c, node0, acc, own, T, bands, negative):
        """One knot chunk's Hermite residues on the banded rows of ``bands`` (one table per live branch).

        ``negative`` (static, one flag per table): False = x >= 0 residues of pair g against W_g,
        True = x < 0 residues of pair -g against W_g^T.  Per (chunk, q) the pair densities are gathered
        on the table's bands only and the Hermite coefficients are built once; per knot only the
        ``index == node`` selection, the contractions and the apply remain.  When a table is wider than
        the band tile, each branch runs its own column tiles (:func:`tile_sum`) and applies per branch.
        """
        single = tile is None or all(int(t.shape[-1]) <= tile for t in bands)

        def over_q(carry, iq):
            g = q0() + iq
            if single:
                sides = []
                for table, neg in zip(bands, negative):
                    rho, x, x_own, f = pair_rows(T, T["qneg"][g] if neg else g, T["psi_nq" if neg else "psi_q"][iq],
                                                 bands=table[g])
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
                        inc_own = inc_own + own_apply(op, laws.real_knot_select(node, coeffs_own))
                    inc = laws.apply(jnp.concatenate(ops, -1), jnp.concatenate(wts, -1))
                    return (c2[0] + inc, c2[1] + inc_own), None
                (s_acc, s_own), _ = jax.lax.scan(over_z, zeros2(acc, own), jnp.arange(W.shape[0]), unroll=1)
            else:
                s_acc, s_own = zeros2(acc, own)
                for table, neg in zip(bands, negative):
                    tab, psi_rows, pair = table[g], T["psi_nq" if neg else "psi_q"][iq], T["qneg"][g] if neg else g

                    def column_tile(lo, width, tab=tab, psi_rows=psi_rows, pair=pair, neg=neg):
                        cols = tab if width is None else jax.lax.dynamic_slice_in_dim(tab, lo, width, axis=1)
                        rho, x, x_own, f = pair_rows(T, pair, psi_rows, bands=cols)
                        coeffs = laws.hermite_branch_coefficients(x, f, h_ry, neg)
                        coeffs_own = laws.hermite_branch_coefficients(x_own, f, h_ry, neg)

                        def over_z(c2, iz):
                            node, z = node0 + iz, z_c[iz]
                            Wz, dWz = W[iz, iq], dW[iz, iq]
                            op = laws.real_knot_operators(z, contract(rho, Wz.T if neg else Wz, nk),
                                                          contract(rho, dWz.T if neg else dWz, nk))
                            return (c2[0] + laws.apply(op, laws.real_knot_select(node, coeffs)),
                                    c2[1] + own_apply(op, laws.real_knot_select(node, coeffs_own))), None
                        out, _ = jax.lax.scan(over_z, zeros2(acc, own), jnp.arange(W.shape[0]), unroll=1)
                        return out
                    b_acc, b_own = tile_sum(column_tile, tab.shape[-1], tile, zeros2(acc, own))
                    s_acc, s_own = s_acc + b_acc, s_own + b_own
            v = T["qvalid"][g]
            return (carry[0] + v * s_acc, carry[1] + v * s_own), None
        (inc, inc_own), _ = jax.lax.scan(over_q, zeros2(acc, own), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc), psum_add(own, inc_own)

    def cd_anchor(W0, dW0, acc, own, T):
        def over_q(carry, iq):
            g = q0() + iq

            def band_tile_(lo, width):
                rho, x, x_own, f = pair_rows(T, g, T["psi_q"][iq], lo, width)
                a0, d0 = contract(rho, W0[iq], nk), contract(rho, dW0[iq], nk)
                total, cp, cm = laws.anchor_part(a0, d0, x, f, eta, beta)
                sub = laws.anchor_subtraction(cp, cm, laws.imag_anchor_sums(T["u_all"], T["w_all"], x, f, eta, beta))
                total_o, _, _ = laws.anchor_part(a0, d0, x_own, f, eta, beta, apply_fn=own_apply)
                sub_o = laws.anchor_subtraction(cp, cm, laws.imag_anchor_sums(T["u_all"], T["w_all"], x_own, f, eta, beta),
                                                apply_fn=own_apply)
                return total - sub, total_o - sub_o
            s_acc, s_own = tile_sum(band_tile_, T["psi_q"].shape[2], tile, zeros2(acc, own))
            v = T["qvalid"][g]
            return (carry[0] + v * s_acc, carry[1] + v * s_own), None
        (inc, inc_own), _ = jax.lax.scan(over_q, zeros2(acc, own), jnp.arange(nq_loc), unroll=1)
        return psum_add(acc, inc), psum_add(own, inc_own)

    q3, q4 = P(axes), P(None, axes)

    def wrap(fn, in_specs, out_specs, name=None):
        """jit(shard_map(fn)); the entry "T" of ``in_specs`` takes the table dict's :func:`table_specs`."""
        def named(*args):
            specs = tuple(table_specs(a, axes) if isinstance(s, str) else s for s, a in zip(in_specs, args))
            return shard_map(fn, mesh=mesh_xy, in_specs=specs, out_specs=out_specs, check_vma=False)(*args)
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
            real_programs[negative] = wrap(body, (q4, q4, P(), P(), P(), P(), "T", P()), (P(), P()),
                                           name="cd_real_contract")
        return real_programs[negative](W, dW, z_c, node0, acc, own, T, tuple(bands))

    return SimpleNamespace(
        sigma_x=wrap(cd_sigma_x, (q3, P(), "T"), P()),
        imag=wrap(cd_imag_contract, (q4, P(), P(), P(), P(), "T"), (P(), P())),
        real=real,
        anchor=wrap(cd_anchor, (q3, q3, P(), P(), "T"), (P(), P())),
        band_tile=tile)


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


def qz_flat_programs(mesh_xy, *, nq_irr, n):
    """The (q, z) flat layout of the Dyson step: row q * nz + z on P(('x','y'), None, None), zero rows to a multiple of P.

    to_qz(chi4 [nz, nq_irr, n, n] face)  -> [F, n, n], F = ``padded_axis(nq_irr nz, mesh)`` (the chunk need
                                            not make nq_irr nz divisible by P)
    roots_qz(h [nq_irr, n, n] face, nz)  -> h repeated over z on the same rows: the Coulomb roots stay on
                                            the face (no replicated copy); the values are copies
    rows(flat, nz)                       -> the nq_irr nz live rows (static slice; identity when F = nq_irr nz)
    A zero row has chi = 0 and H = 0, so its W_c and dW_c/ds are exactly zero; no consumer reads it.
    """
    from runtime.padding import padded_axis
    mid3 = NamedSharding(mesh_xy, P("x", None, "y"))
    zq_par = NamedSharding(mesh_xy, P(("x", "y"), None, None))

    def flat_rows(nz_):
        return padded_axis(nq_irr * int(nz_), mesh_xy, name="cd (q, z) flat").carrier

    def cd_to_qz(chi4):
        """[nz, nq_irr, n, n] face -> [(q, z) padded, n, n] q-major flat on the (q,z)-parallel layout."""
        nz_ = chi4.shape[0]
        flat = jnp.swapaxes(chi4, 0, 1).reshape((nq_irr * nz_, n, n))
        pad = flat_rows(nz_) - nq_irr * nz_
        if pad:
            flat = jnp.pad(flat, ((0, pad), (0, 0), (0, 0)))
        flat = jax.lax.with_sharding_constraint(flat, mid3)
        return jax.lax.with_sharding_constraint(flat, zq_par)

    def cd_roots_qz(h, nz_):
        return cd_to_qz(jnp.broadcast_to(h[None], (nz_,) + tuple(h.shape)))

    def rows(flat, nz_):
        return flat if int(flat.shape[0]) == nq_irr * nz_ else flat[:nq_irr * nz_]
    return SimpleNamespace(to_qz=jax.jit(cd_to_qz), roots_qz=jax.jit(cd_roots_qz, static_argnums=1, out_shardings=zq_par),
                           rows=rows, flat_rows=flat_rows)


def carrier_band_blocks(src, mesh_xy, *, unfold=None):
    """``block(lo, width)`` -> psi[:, lo:lo + width] on the full k grid, replicated: one band block of the carrier.

    ``src`` [nk_src, nb, s, n] is the psi carrier in any placement (the face ``PSI_NMU_SPEC`` in production);
    ``unfold`` None (src spans the full k grid) or a function [nk_src, b, s, n] -> [nk, b, s, n] (the typed
    parent action, replicated in and out).  The carrier is placed once on the ``axis`` layout
    (``common.wfn_layout.psi_specs``: bands local, mu on y; 1/P_y of the carrier per rank), so a block is a
    local band slice gathered over y and then unfolded: nothing wider than one block is replicated.
    ``width`` is static (one compile per width), ``lo`` is not; ``lo + width`` must not exceed nb.
    """
    from common.wfn_layout import psi_specs
    target = NamedSharding(mesh_xy, psi_specs("axis")[0])
    held = src if getattr(src, "sharding", None) == target else jax.jit(lambda a: a * 1.0, out_shardings=target)(src)
    nb = int(held.shape[1])
    rep = NamedSharding(mesh_xy, P())
    take = jax.jit(lambda a, lo, w: jax.lax.dynamic_slice_in_dim(a, lo, w, axis=1) * 1.0, static_argnums=2,
                   out_shardings=rep)

    def block(lo, width):
        lo, width = int(lo), int(width)
        if lo < 0 or lo + width > nb:
            raise ValueError(f"GATE cd_band_block: bands [{lo}, {lo + width}) outside the carrier's {nb}")
        out = take(held, jnp.int32(lo), width)
        return out if unfold is None else unfold(out)
    return block


def pair_psi_slabs(block, mesh_xy, *, rows_q, rows_nq, l0, L, L_neg, block_bands):
    """Per-rank pair slabs (q-sharded like W): psi at the rows k - q of each rank's own pairs.

    ``block`` is :func:`carrier_band_blocks`; ``rows_q`` / ``rows_nq`` [nq_pad, kw] int: the k - q rows of
    pair q (dense kernels and x >= 0 residues) and of pair -q (x < 0 residues).  Returns
    ``(psi_q [nq_pad, kw, L, s, n], psi_nq [nq_pad, kw, L_neg, s, n])`` on P(('x','y')): the bands l0 + [0, L)
    and l0 + [0, L_neg).  Bands stream ``block_bands`` at a time: each block is gathered once and every
    rank writes its own q rows into its (donated) slab.  The values are copies of the carrier's.
    """
    axes = ("x", "y")
    qsh = NamedSharding(mesh_xy, P(axes))
    rows_q, rows_nq = np.asarray(rows_q, np.int32), np.asarray(rows_nq, np.int32)
    nq_pad, kw = rows_q.shape
    rq, rn = (jax.make_array_from_callback(r.shape, qsh, lambda idx, r=r: r[idx]) for r in (rows_q, rows_nq))
    first = block(l0, min(int(block_bands), int(L)))
    ns, n, dtype = int(first.shape[2]), int(first.shape[3]), first.dtype

    def zeros(width):
        return jax.jit(lambda: jnp.zeros((nq_pad, kw, width, ns, n), dtype), out_shardings=qsh)()

    def write_local(slab, blk, rows, b0):
        return jax.lax.dynamic_update_slice_in_dim(slab, blk[rows], b0, axis=2)
    put = jax.jit(shard_map(write_local, mesh=mesh_xy, in_specs=(P(axes), P(), P(axes), P()), out_specs=P(axes),
                            check_vma=False), donate_argnums=0)
    psi_q, psi_nq = zeros(int(L)), zeros(int(L_neg))
    for b0 in range(0, int(L), int(block_bands)):
        w = min(int(block_bands), int(L) - b0)
        blk = first if b0 == 0 else block(l0 + b0, w)
        psi_q = put(psi_q, blk, rq, jnp.int32(b0))
        if b0 < L_neg:
            wn = min(w, int(L_neg) - b0)
            psi_nq = put(psi_nq, blk if wn == w else blk[:, :wn], rn, jnp.int32(b0))
        del blk
    del first
    return psi_q, psi_nq


# ------------------------------------------------------------------------------- chi0 provider
def cd_chunk_plan(n, nq_irr, nq_pad, ndev, n_imag, J, override=0):
    """``(chunk_imag, chunk_real, n_imag_pad, J_pad)`` of the CD streams (TASTE 96).

    The tile is ``runtime.tiles.tile_units`` of the chunk's own per-unit bytes (or the
    ``cd_knot_chunk`` override), sized by :func:`knot_chunk`: a chunk below P / gcd(nq_irr, P) units is
    kept and the (q, z) flat axis takes zero rows (:func:`qz_flat_programs`), a larger one is rounded up
    so the flat axis divides by P.  The counts are padded to whole chunks.  The precompute CLI
    (``gw.cd.chi0_rules``) plans the same windows.
    """
    from runtime.tiles import tile_units

    def tile(axis, count):
        return knot_chunk(override or tile_units(chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis=axis), count),
                          count, ndev=ndev, nq_irr=nq_irr)
    chunk_i, chunk_r = tile("imag", n_imag), tile("real", J)
    return chunk_i, chunk_r, -(-n_imag // chunk_i) * chunk_i, -(-J // chunk_r) * chunk_r


def bank_chi0(wfns, meta, mesh_xy, z_ry, *, q_ids, mu_ry, rule_tolerance, slots, windows=None,
              rule_capacity=None, rules_file=None, time_reversal=False, save_path=None, allow_partial=False,
              print_fn=print):
    """chi0 at the CD points from the shared-pole bank's own response quadrature.

    ``z_ry`` lists every point the CD streams evaluate (imaginary nodes ``i u`` and real knots
    ``j h + i eta`` with their FD-9 ghosts; Im z > 0; a repeat is evaluated once).  The rules are
    :mod:`gw.cd.chi0_rules`': one request per CD memory window (``windows``, from
    ``chi0_rules.window_members``: the Laplace rule on the imaginary windows, the bank tree at the
    CD capacity and mass gate on the knot windows), loaded from ``rules_file`` after its identity
    check when given, else built here (rank 0 writes them to ``save_path``).  ``windows=None`` is the
    bank's own single request over every point.  chi0(q, z) is the value row of
    ``response_bank.integrate_response_group`` (``ordered=True``: the physical ``FT_q[chi]`` at the
    full-grid rows ``q_ids``), one Green-pair scan per rule group a chunk touches; a chunk's members
    are taken out of their rule groups by index, every other rule key is the bank's.  No Lehmann sum.

    ``mu_ry`` is the current chemical potential.  ``slots`` is the largest chunk.  Returns
    ``(chi, note)``: ``chi(points)`` -> ``[nz, nq, n, n]`` at ``P(None, None, 'x', 'y')`` for any
    sequence of the points (repeats allowed; a repeat is a copy); ``note`` is the rule receipt
    (census rows, source), updated as chunks stream.  ``allow_partial`` (instrument runs only) loads a
    partial ``rules_file``; a point no built rule group holds then refuses (GATE cd_chi0_rule_unbuilt).
    """
    from common.collectives import device_put_process_local_cached
    from gw import response_bank as rb
    from gw.cd import chi0_rules as cr

    EV = float(RYD_TO_EV)
    index = {}
    for z in np.asarray(z_ry, dtype=np.complex128).ravel().tolist():       # one evaluation per distinct point
        index.setdefault(complex(z), len(index))
    points = np.asarray(list(index), dtype=np.complex128)
    route = "bank" if windows is None else "cd"
    if windows is None:
        windows = [dict(kind="all", chunk=0, members=list(range(points.size)))]
    sha = None
    if rules_file:
        plan, energy, occupied = cr.sample_plan(wfns, meta, points, mu_ry=mu_ry, tolerance=rule_tolerance)
        support = rb.response_support(wfns, meta, plan, {}, print_fn=print_fn)
        options = {} if route == "bank" else cr.cd_rule_options(rule_tolerance, rule_capacity)
        payload = cr.load_rules(rules_file, allow_partial=allow_partial)
        cr.check_identity(payload["identity"], cr.rule_identity(
            energy, occupied, points, windows, cr.plan_identity(support, plan["census"]["partial_at_mu"]),
            rule_tolerance, options, route))
        source = f"loaded {rules_file}"
    else:
        payload, support = cr.build_rules(wfns, meta, points, windows, mu_ry=mu_ry, tolerance=rule_tolerance,
                                          slots=slots, capacity=rule_capacity, time_reversal=time_reversal,
                                          route=route, print_fn=print_fn)
        source = "built in job"
        if save_path and jax.process_index() == 0:
            sha = cr.save_rules(save_path, payload)
            source += f", saved {save_path}"
    rules = cr.stream_rules(payload, support, slots)
    groups = payload["rule_groups"]
    plan = payload["plan"]
    sliced = ("value", "derivative", "sampled_error", "coefficient_mass")
    note = dict(provider="gw.response_bank shared-node quadrature on the CD memory windows (gw.cd.chi0_rules): "
                         "Laplace rule (imaginary windows) / ResponseGroupTree (knot windows), "
                         "integrate_response_group value rows (ordered=True)",
                source=source, rules_sha256=sha, route=route, partial_rules=bool(payload.get("partial")), rule_tolerance=float(rule_tolerance),
                options=payload["identity"]["options"], time_reversal=bool(time_reversal),
                points=int(points.size), windows=len(windows), interval_ev=[plan["lo"] * EV, plan["hi"] * EV],
                occupation_amplitude=plan["amplitude"], decay_rate_ry_inv=plan["decay_rate"],
                partial_at_mu=plan["metallic"], band_ranges=list(support["band_ranges"]),
                rule_nodes=int(sum(int(g["count"]) for g in groups)), rule_groups=payload["census"],
                passes=0, green_pair_evaluations=0)
    receipt = {"correlation_count": 0}
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

    def chi(points_):
        try:
            ids = [index[complex(z)] for z in np.asarray(points_, dtype=np.complex128).ravel()]
        except KeyError as exc:
            raise ValueError(f"GATE cd_chi0_point_unplanned: got: z = {exc} Ry outside the planned CD points; "
                             "want: a point of the rule set; why: one rule set serves every chunk") from None
        want = set(ids)
        nz = len(ids)
        if nz not in blank:
            blank[nz] = jax.jit(lambda: jnp.zeros((nz, len(q_ids), n, n), jnp.complex128), out_shardings=face4)
        out = blank[nz]()
        held = set()
        for rule in groups:
            hit = [p for p, m in enumerate(rule["members"]) if m in want]
            held.update(rule["members"][p] for p in hit)
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
        if want - held:
            miss = sorted(want - held)
            raise ValueError(f"GATE cd_chi0_rule_unbuilt: got: {len(miss)} point(s) (first z = {points[miss[0]]} Ry) "
                             "that no built rule group holds; want: every evaluated window built; why: a partial "
                             "(census) rule file serves only the windows it built, never a silent zero")
        note["green_pair_evaluations"] = int(receipt["correlation_count"])
        return out
    chi.rules, chi.groups, chi.points, chi.payload = rules, groups, points, payload   # (tests)
    return chi, note


# ------------------------------------------------------------------- own-energy rows, reach, checkpoints
#: ``cd_checkpoint`` (docs/input_reference.md): ``off`` writes nothing; ``write`` rewrites the accumulator
#: checkpoint after every completed chunk; ``resume`` first continues from an authenticated checkpoint,
#: then writes as ``write``.  A resume is never inferred from a file's presence.
CD_CHECKPOINT_MODES = ("off", "write", "resume")
#: Beside the deck, in ``tmp/`` like the one-shot sweep checkpoint of ``gw.sigma_dispatch``.
CD_CHECKPOINT_FILE = os.path.join("tmp", "cd_checkpoint.h5")
#: Deck keys a continuation leg rewrites without changing the result; :func:`cd_deck_digest` drops them.
CD_RESUME_FREE_KEYS = ("cd_checkpoint", "restart")


def residue_reach_need(omega, eps_own, enk_L, f_L):
    """The largest live |E - eps_l| (Ry) over E = the omega grid and the own energies (GATE cd_residue_coverage).

    Occupied l (f > 0) are read from the lowest E (x < 0 residues reach eps_l - E_min), unoccupied l
    (1 - f > 0) from the highest (x >= 0 residues reach E_max - eps_l).  ``omega`` [E] and ``eps_own``
    [rows, m] absolute Ry; ``enk_L``, ``f_L`` [nk, L] the band-sum tables.
    """
    E_all = np.concatenate([np.asarray(omega, np.float64).ravel(), np.asarray(eps_own, np.float64).ravel()])
    need_neg = np.where(f_L > 0, np.maximum(0.0, enk_L - E_all.min()), 0.0).max()
    need_pos = np.where(1.0 - f_L > 0, np.maximum(0.0, E_all.max() - enk_L), 0.0).max()
    return float(max(need_neg, need_pos))


def own_energy_window(m0, m1, band_min):
    """Offset ``o0`` of the own-energy rows in the Sigma window [m0, m1) under ``cd_own_energy_band_min``.

    ``band_min`` is a 1-based band index.  The own-energy rows (the DFT-energy diagonal: ``eps_own``, the
    own accumulators and the reach rule) are the window bands >= band_min; the omega-grid cube and the
    finalizer keep the whole window, so a band below it is read out of grid exactly as without CD.
    1 (the default) is the whole window.
    """
    o0 = max(0, int(band_min) - 1 - int(m0))
    if o0 >= int(m1) - int(m0):
        raise ValueError(f"GATE cd_own_energy_band_min: cd_own_energy_band_min got: {band_min}; want: <= {m1} "
                         f"(1-based, inside the Sigma window [{m0 + 1}, {m1}]); why: the own-energy rows are a "
                         "non-empty subset of the window")
    return o0


class CdStopAfter(RuntimeError):
    """``LORRAX_CD_STOP_AFTER`` reached: the run stops after its last chunk's checkpoint (a timing probe)."""


def parse_stop_after(text):
    """``LORRAX_CD_STOP_AFTER`` -> ``{'imag': N, 'real': M, 'real_from': K}`` (each None if absent).

    ``imag:N`` runs N imaginary chunks and then leaves the imaginary axis; ``real:M`` stops after M
    real-axis chunks; ``real_from:K`` starts the real axis at chunk K (negative: from the end; the ghost
    tail is rebuilt as on a resume) with zero real accumulators.  Counts are chunks executed by this
    process, after any resume.  ``imag`` and ``real_from`` change the result (no longer a Sigma), so the
    checkpoint identity records them; ``real`` alone only stops early (docs/reference/env_vars.md).
    """
    out = dict(imag=None, real=None, real_from=None)
    text = (text or "").strip()
    for item in filter(None, (t.strip() for t in text.split(","))):
        axis, _, count = item.partition(":")
        axis, count = axis.strip().lower(), count.strip()
        signed = count[1:] if (axis == "real_from" and count.startswith("-")) else count
        if axis not in out or not signed.isdigit():
            raise ValueError(f"GATE cd_stop_after: LORRAX_CD_STOP_AFTER got: {text!r}; want: "
                             "'imag:N,real:M,real_from:K' (each part optional; N, M >= 0; K may be negative); "
                             "why: a mistyped instrument must not run a production-length job")
        out[axis] = int(count)
    return out


def cd_deck_digest(path):
    """sha256 of the deck as the parser reads it, without :data:`CD_RESUME_FREE_KEYS`.

    The SC checkpoint's rule (``gw.sc_iteration._sc_checkpoint_identity``): comments and blank lines are
    dropped, each line is read as ``key = value`` or ``key: value`` with the key lower-cased, so spelling
    does not count.  An empty path digests the empty deck.
    """
    import hashlib
    import re
    lines = []
    if path:
        with open(path) as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                pair = re.match(r"([^=:]*)[=:](.*)", line)
                key = (pair.group(1) if pair else line).strip().lower()
                if line and key not in CD_RESUME_FREE_KEYS:
                    lines.append(f"{key}={pair.group(2).strip()}" if pair else line)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def real_replay_start(start, chunk, gh=GH):
    """First knot chunk whose chi0 window a resume at real chunk ``start`` re-runs (no Dyson, no contraction).

    The FD-9 ghost tail entering chunk ``start`` holds knots [start*chunk - gh, start*chunk + gh).  Each knot
    came from exactly one planned window call (``gw.cd.chi0_rules.cd_windows``: chunk 0 covers
    [-gh, chunk + gh), chunk c > 0 its new knots [c*chunk + gh, (c+1)*chunk + gh)).  The calls of chunks
    r = start - ceil(2 gh / chunk) .. start - 1 (r >= 0) cover the tail, each with its uninterrupted z
    batch and only planned points, so the rebuilt tail is bit-identical to the uninterrupted one.
    """
    start = int(start)
    return 0 if start <= 0 else max(0, start - (-(-2 * int(gh) // int(chunk))))


def stream_cd_chunks(*, imag_chunks, real_chunks, chi_real, join, tail_of, imag_step, real_step, accs,
                     resume=None, on_chunk=None, stop_after=None, gh=GH):
    """The CD chunk loops: imaginary nodes, then real knots with rolling FD-9 ghosts; resumable.

    imag_chunks, real_chunks : lists of (start, stop) unit ranges (the padded tiling).
    chi_real(ic)             : chi0 at real chunk ic's planned window (one call = one z batch): chunk 0
                               with both ghost sides, chunk c > 0 its new knots (gw.cd.chi0_rules.cd_windows).
    join(tail, new), tail_of(chi_e) : the ghost roll (concatenate; the last 2 gh knots).
    imag_step(ic, c0, c1, accs) -> accs : one node chunk (chi0 -> Dyson -> contraction).
    real_step(ic, j0, j1, box, accs) -> accs : one knot chunk on the ghost-extended chi0 ``box.pop()``;
        ``box`` is a one-element list so this loop holds no reference while the step runs.
    accs : dict of the replicated accumulators.  resume : None or ``CdCheckpoint.read()``'s dict (axis,
    chunk = the last completed chunk, accs); the imaginary chunks up to it are skipped, and on the real
    axis the chi0 calls from :func:`real_replay_start` on are re-run to rebuild the ghost tail.
    on_chunk(axis, ic, accs) : after every completed chunk (the checkpoint writer).
    stop_after : :func:`parse_stop_after`'s dict (``real_from`` starts the real axis at that chunk).
    Returns ``(accs, info)``; info: resumed_from, imag_run, real_run, replayed, imag_truncated, stopped.
    """
    stop_after = stop_after or {}
    lim_i, lim_r, real_from = stop_after.get("imag"), stop_after.get("real"), stop_after.get("real_from")
    n_i, n_r = len(imag_chunks), len(real_chunks)
    start_i = start_r = 0
    if real_from is not None and n_r:
        start_r = int(real_from) % n_r
    if resume is not None:
        accs = dict(resume["accs"])
        if resume["axis"] == "imag":
            start_i = int(resume["chunk"]) + 1
        elif resume["axis"] == "real":
            start_i, start_r = n_i, max(start_r, int(resume["chunk"]) + 1)
        else:
            raise ValueError(f"GATE cd_resume_axis: checkpoint axis {resume['axis']!r}; want imag | real")
    info = dict(resumed_from=None if resume is None else [str(resume["axis"]), int(resume["chunk"])],
                imag_run=0, real_run=0, replayed=0, imag_truncated=False, stopped=False)
    for ic in range(start_i, n_i):
        if lim_i is not None and info["imag_run"] >= lim_i:
            info["imag_truncated"] = True
            break
        c0, c1 = imag_chunks[ic]
        accs = imag_step(ic, c0, c1, accs)
        info["imag_run"] += 1
        if on_chunk is not None:
            on_chunk("imag", ic, accs)
    chunk = (real_chunks[0][1] - real_chunks[0][0]) if n_r else 1
    tail = rebuilt = None
    for ic in range(real_replay_start(start_r, chunk, gh), n_r):
        if ic >= start_r and lim_r is not None and info["real_run"] >= lim_r:
            info["stopped"] = True
            break
        j0, j1 = real_chunks[ic]
        new = chi_real(ic)
        if ic < start_r:                              # replay: the planned calls that cover the ghost tail
            rebuilt = new if rebuilt is None else join(rebuilt, new)
            info["replayed"] += 1
            if ic == start_r - 1:
                tail, rebuilt = tail_of(rebuilt), None
            del new
            continue
        chi_e = new if tail is None else join(tail, new)
        del new
        tail = tail_of(chi_e)
        box = [chi_e]
        del chi_e
        accs = real_step(ic, j0, j1, box, accs)
        info["real_run"] += 1
        if on_chunk is not None:
            on_chunk("real", ic, accs)
    del tail
    return accs, info


class CdCheckpoint:
    """Per-chunk checkpoint of the replicated CD accumulators (``file_io.sigma_checkpoint``, ``CD_SCHEMA``).

    One file, ``<run_dir>/tmp/cd_checkpoint.h5``, rewritten after every completed chunk through
    ``write_sigma_checkpoint`` (published by ``os.replace``: a kill mid-write leaves the previous chunk's
    file).  The accumulators travel as the host ``state`` blob (plain numpy; they are replicated, so
    rank 0 holds them whole) with the axis, chunk index and stage seconds; the record's ``identity`` is
    the driver's dict (deck digest, tree, chi0 provider and rule tolerance, geometry and tiling), and
    ``host`` tags the axis and chunk.  :meth:`read` returns the ``resume`` dict of
    :func:`stream_cd_chunks`, or None for an absent or foreign file (left in place and named).
    """

    def __init__(self, path, identity, mesh, *, print_fn=print):
        self.path, self.identity, self.mesh, self.print_fn = str(path), identity, mesh, print_fn
        self.writes, self.seconds, self.nbytes = 0, 0.0, 0

    def read(self):
        from common.collectives import replicate_to_mesh
        from file_io.sigma_checkpoint import CD_SCHEMA, read_sigma_checkpoint
        found = read_sigma_checkpoint(self.path, identity=self.identity, mesh=self.mesh, print_fn=self.print_fn,
                                      schema=CD_SCHEMA, discard=False, cubes=False)
        if found[0] is None:
            if found[1] != "absent":
                self.print_fn(f"  WARNING CD resume: {self.path} does not continue this run ({found[1]}); "
                              "starting at chunk 0 (this run's first checkpoint replaces the file)")
            return None
        _, host, state, commit = found
        accs = {k: replicate_to_mesh(np.asarray(v), self.mesh) for k, v in state["accs"].items()}
        return dict(axis=str(state["axis"]), chunk=int(state["chunk"]), accs=accs,
                    timings=dict(state.get("timings") or {}), commit=str(commit), host=host)

    def write(self, axis, chunk, accs, *, timings=None, host=None):
        """Collective: every rank calls it after the chunk; returns the write's seconds."""
        from common.collectives import gather_to_host
        from file_io.sigma_checkpoint import CD_SCHEMA, write_sigma_checkpoint
        t0 = time.monotonic()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        arrays = {k: np.asarray(gather_to_host(v)) if isinstance(v, jax.Array) else np.asarray(v)
                  for k, v in accs.items()}
        state = dict(axis=str(axis), chunk=int(chunk), accs=arrays,
                     timings={k: float(v) for k, v in (timings or {}).items()})
        nbytes, _ = write_sigma_checkpoint(self.path, identity=self.identity, cubes={}, mesh=self.mesh,
                                           host=dict(host or {}, axis=str(axis), chunk=int(chunk)),
                                           schema=CD_SCHEMA, state=state)
        dt = time.monotonic() - t0
        self.writes, self.seconds, self.nbytes = self.writes + 1, self.seconds + dt, int(nbytes)
        return dt


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
    ``cd_sigma_own_energy.npz`` (diagonal at the DFT energies on the own-energy rows ``bands0`` and its
    three contour parts; ``sigma_x_diag_ry`` on the whole window ``window_bands0``),
    ``cd_leak_ledger.csv`` and the partial dumps ``cd_partial_{imag,anchor}.npz``.  ``cd_checkpoint``
    (write | resume) keeps ``tmp/cd_checkpoint.h5`` after every chunk (:class:`CdCheckpoint`);
    ``LORRAX_CD_STOP_AFTER`` ends an instrument run with ``cd_stop_receipt.json`` and :class:`CdStopAfter`.
    """
    from common.collectives import barrier, gather_to_host, replicate_to_mesh
    from gw import response_bank as rb
    from gw.ppm_sigma import SigmaOmegaResult, sigma_band_axis
    from gw.w_isdf import _w_solve_pref_scalar
    from runtime.tiles import TILE_BYTES, tile_units
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
    # Own-energy rows (cd_own_energy_band_min): window bands >= the key.  The omega-grid cube, psi_M and the
    # finalizer keep the whole window [m0, m1), so a band below the key is read out of grid as without CD.
    o0 = own_energy_window(m0, m1, cd_handle.get("own_energy_band_min", 1))
    nb_o = nb_s - o0
    pref = float(_w_solve_pref_scalar(meta))
    omega = np.asarray(omega_grid_ry, dtype=np.float64) + float(efermi_ry)   # absolute Ry
    nE = int(omega.size)
    enk_L = energy[:, l0:l1]
    f_L = f_cd[:, l0:l1]
    eps_own = energy[rows_all, m0 + o0:m1]                         # [rows, nb_o] own energies

    # residue coverage (GATE cd_residue_coverage): the largest |x| any live residue reads
    need = residue_reach_need(omega, eps_own, enk_L, f_L)
    need_window = need if not o0 else residue_reach_need(omega, energy[rows_all, m0:m1], enk_L, f_L)
    reach = cd_handle.get("residue_reach_ev", "auto")
    if reach != "auto" and float(reach) / EV < need:
        raise ValueError(f"GATE cd_residue_coverage: cd_residue_reach_ev got: {reach}; want: >= {need * EV:.3f} eV "
                         "(largest live |omega - eps_l|); why: no clipping, no omitted tail")
    reach_ry = need if reach == "auto" else float(reach) / EV
    J = int(np.ceil(reach_ry / h_ry)) + 2
    u_ry_all, w_ry_all = (np.asarray(v) / EV for v in tan_mapped_gauss_rule(n_imag, eta * EV))

    # ---- tiles (TASTE 96): from the chunk's shapes; any chunk (the (q, z) flat axis takes zero rows) -----
    override = int(cd_handle.get("knot_chunk") or 0)
    chunk_i, chunk_r, n_imag_pad, J_pad = cd_chunk_plan(n, nq_irr, nq_pad, ndev, n_imag, J, override)
    u_ry = np.concatenate([u_ry_all, np.full(n_imag_pad - n_imag, u_ry_all[-1])])
    w_ry = np.concatenate([w_ry_all, np.zeros(n_imag_pad - n_imag)])         # padded nodes weigh zero
    real_ry = h_ry * np.arange(J_pad)                                         # knots >= J are never referenced
    _say(print_fn, started, f"geometry nk={nk} nq_irr={nq_irr} nq_pad={nq_pad} nk_red={nk_red} n_packed={n} "
         f"window m=[{m0},{m1}) l=[{l0},{l1}) nE={nE} eta={eta * EV:.4f} eV h={h_ry * EV:.4f} eV J={J} "
         f"(padded {J_pad}) knots to {real_ry[J - 1] * EV:.3f} eV (need {need * EV:.3f} eV) imag nodes={n_imag} "
         f"(padded {n_imag_pad}) own rows [{m0 + o0},{m1}) (whole-window need {need_window * EV:.3f} eV) "
         f"mu={mu * EV:.6f} eV full_bz={full_bz}; tile {TILE_BYTES / 2**30:.0f} GiB/rank: "
         f"chunk imag {chunk_i} ({chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis='imag') / 2**20:.1f} MiB/node), "
         f"real {chunk_r} ({chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis='real') / 2**20:.1f} MiB/knot)"
         + (f" [cd_knot_chunk override {override}]" if override else ""))
    t_bands = time.monotonic()
    band_tables, band_summary = residue_band_tables(enk_L, f_L, KMQ_pad, qneg_pad, rows_all, omega, eps_own,
                                                    h_ry, chunk_r, J_pad // chunk_r)
    band_summary["build_s"] = round(time.monotonic() - t_bands, 3)
    _say(print_fn, started, "residue band tables (the rows each knot chunk contracts): " + json.dumps(band_summary))

    # ---- wavefunctions: per-rank pair slabs, streamed from the carrier one band block at a time ---------
    carrier = wfns.green_parent
    unfold = None
    if wfns.psi_nmu is not None and (carrier is None or int(wfns.psi_nmu.shape[0]) == nk):
        src = wfns.psi_nmu                                                           # [nk, nb, s, n]
    else:
        plan = carrier.plan
        src = carrier.psi_nmu                                                        # [n_parent, nb, s, n]
        if int(plan.n_parent) != nk:                 # parents unfolded per block by the typed centroid action
            unfold = jax.jit(lambda a: unfold_wavefunction_local(
                a, irr_idx=np.asarray(plan.irr_idx), sym_idx=np.asarray(plan.sym_idx),
                k_irr_frac=np.asarray(plan.k_parent_frac), local_perm=np.asarray(plan.sym_perm),
                L_table=np.asarray(plan.L_table), spin_action_full=np.asarray(plan.spin_action_full),
                n_sym_spatial=int(plan.n_sym_spatial), spin_axis=2, mu_axis=3), out_shardings=rep)
    if int(src.shape[1]) != nb_full or int(src.shape[-1]) != n:
        raise ValueError(f"GATE cd_psi_shape: {tuple(src.shape)} vs (k, nb_full={nb_full}, s, n={n})")
    ns = int(src.shape[2])
    L_neg = max(1, int(band_summary["extent_pos_neg"][1]))          # the x < 0 tables read bands [0, L_neg) only
    band_tile = int(cd_handle.get("band_tile") or 0) or tile_units(band_bytes_per_unit(n_rows, nb_s, n, ns, nE), L)
    slab_block = tile_units(16.0 * ns * n * (int(src.shape[0]) + nk + 2 * nq_loc * n_rows), L)
    mem_plan = cd_memory_plan(n=n, nq_irr=nq_irr, nk=nk, n_carrier_k=int(src.shape[0]), nb_carrier=nb_full,
                              n_rows=n_rows, nb_s=nb_s, L=L, L_neg=L_neg, ns=ns, n_omega=nE, ndev=ndev,
                              mesh_y=int(mesh_xy.shape["y"]), chunk_imag=chunk_i, chunk_real=chunk_r,
                              band_tile=min(band_tile, L), slab_block=slab_block, wedge=nq_irr != nk)
    _say(print_fn, started, f"band tile {min(band_tile, L)} of L={L} (x < 0 slab {L_neg} bands), slab block "
         f"{slab_block}; memory plan per rank (GB, cd_memory_plan): "
         + json.dumps({k: round(v / 1e9, 2) for k, v in mem_plan["phases"].items()})
         + f" peak {mem_plan['peak_bytes'] / 1e9:.2f} ({mem_plan['peak_phase']})")
    block = carrier_band_blocks(src, mesh_xy, unfold=unfold)
    psi_m_full = block(m0, nb_s)                                                     # [nk, nb_s, s, n] replicated
    if int(psi_m_full.shape[0]) != nk:
        raise ValueError(f"GATE cd_psi_shape: unfolded carrier has {int(psi_m_full.shape[0])} k rows vs nk={nk}")
    psi_M = jax.jit(lambda a, r: a[r] * 1.0, out_shardings=rep)(psi_m_full, jnp.asarray(rows_all))
    del psi_m_full
    psi_q, psi_nq = pair_psi_slabs(block, mesh_xy, rows_q=KMQ_pad[:, rows_all], rows_nq=KMQ_pad[qneg_pad][:, rows_all],
                                   l0=l0, L=L, L_neg=L_neg, block_bands=slab_block)
    del block

    def _rep(a):
        return replicate_to_mesh(np.asarray(a), mesh_xy)
    tabs = dict(psi_q=psi_q, psi_nq=psi_nq, psi_M=psi_M, KMQ=_rep(KMQ_pad.astype(np.int32)),
                kw=_rep(rows_all.astype(np.int32)), enk=_rep(enk_L), f=_rep(f_L), omega=_rep(omega),
                eps_own=_rep(eps_own), qneg=_rep(qneg_pad.astype(np.int32)), qvalid=_rep(qvalid), u_all=_rep(u_ry_all),
                w_all=_rep(w_ry_all))
    u_dev, w_dev = _rep(u_ry), _rep(w_ry)
    beta = tuple(float(b) for b in laws.BETA_RY)
    K = build_contraction_kernels(mesh_xy, nk=nk, nq_loc=nq_loc, eta=eta, h_ry=h_ry, beta=beta, band_tile=band_tile,
                                  own0=o0)

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
    # The roots stay on the face; roots_qz lays them on the (q, z) flat rows as chi is laid (no replicated copy).
    dyson_pair, dyson_value = build_dyson_kernels(mesh_xy, n=n, pref=pref)
    QZ = qz_flat_programs(mesh_xy, nq_irr=nq_irr, n=n)
    to_qz, roots_qz = QZ.to_qz, QZ.roots_qz

    pad_rows = nq_pad - nk

    def cd_full_zone_direct(flat, nz_):
        w = jnp.swapaxes(QZ.rows(flat, nz_).reshape((nq_irr, nz_, n, n)), 0, 1)
        return jnp.pad(w, ((0, 0), (0, pad_rows), (0, 0), (0, 0))) if pad_rows else w
    full_zone_direct = jax.jit(cd_full_zone_direct, static_argnums=1, out_shardings=qpar4)
    take_z = jax.jit(lambda flat, nz_, iz: jax.lax.dynamic_index_in_dim(
        QZ.rows(flat, nz_).reshape((nq_irr, nz_, n, n)), iz, axis=1, keepdims=False), static_argnums=1,
        out_shardings=face)
    to_qpar = jax.jit(lambda a: (jnp.pad(a, ((0, pad_rows), (0, 0), (0, 0))) if pad_rows else a * 1.0),
                      out_shardings=qpar3)
    stack_q = jax.jit(lambda *a: jnp.stack(a), out_shardings=qpar4)

    def full_zone(flat, nz_):
        """[(q, z) flat] -> [nz, nq_pad, n, n] q-parallel (zero rows dropped; unfold per z on wedge decks)."""
        if unfold_w is None:
            return full_zone_direct(flat, nz_)
        return stack_q(*[to_qpar(unfold_w(take_z(flat, nz_, iz))) for iz in range(nz_)])


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
    acc_shape, own_shape = (nE, n_rows, nb_s, nb_s), (n_rows, nb_o)
    acc_imag, own_imag = zeros(*acc_shape), zeros(*own_shape)
    acc_real, own_real = zeros(*acc_shape), zeros(*own_shape)
    acc_anc, own_anc = zeros(*acc_shape), zeros(*own_shape)
    timings = {}

    def clock(name, t0):
        timings[name] = timings.get(name, 0.0) + time.monotonic() - t0

    def partial_dump(name, **arrays):
        out = {k: host(v)[:nk_red] for k, v in arrays.items()}       # collective on every rank; wedge rows
        if rank0 and run_dir is not None:
            np.savez(os.path.join(run_dir, f"cd_partial_{name}.npz"), kw_rows=kw_rows, bands0=np.arange(m0 + o0, m1),
                     elapsed_s=time.monotonic() - started, **out)

    # LORRAX_CD_STOP_AFTER (instrument): parsed before the rules, which it may load partially.
    stop_after = parse_stop_after(os.environ.get("LORRAX_CD_STOP_AFTER", ""))
    instrument = any(v is not None for v in stop_after.values())

    # chi0 at every CD point from the bank response rules, one request per CD memory window
    # (gw.cd.chi0_rules); each chunk below evaluates exactly its window's points.
    t0 = time.monotonic()
    imag_pts, real_pts = cd_rules.cd_windows(u_ry, h_ry, eta, J_pad, chunk_i, chunk_r, GH)
    points_cd, windows_cd = cd_rules.window_members(imag_pts, real_pts)
    rules_file = str(cd_handle.get("chi0_rules_file") or "")
    if rules_file and run_dir is not None and not os.path.isabs(rules_file):
        rules_file = os.path.join(run_dir, rules_file)
    trs = bool(getattr(sym, "trs_allowed", False))
    _say(print_fn, started, "chi0 rule geometry (precompute: python -m gw.cd.chi0_rules ... --out FILE; deck "
         f"cd_chi0_rules_file = FILE): --knots {J} --ndev {ndev} --nq-irr {nq_irr} --nk {nk} --n-packed {n} "
         f"--knot-chunk {override} --imaginary-nodes {n_imag} --eta-ev {eta * EV:.10g} --knot-step-ev "
         f"{h_ry * EV:.10g} --nb-chi {int(meta.b_id_4_chi_user)} --b4-logical {int(slices.b4_logical)} --mu-ev "
         f"{mu * EV:.12g}" + (" --time-reversal" if trs else "")
         + f"; {points_cd.size} points in {len(windows_cd)} windows")
    bank_chi, chi_rule = bank_chi0(
        wfns, meta, mesh_xy, points_cd, q_ids=q_irr, mu_ry=mu,
        rule_tolerance=float(cd_handle.get("chi0_rule_tolerance", CD_CHI0_RULE_TOLERANCE)),
        slots=max(chunk_i, chunk_r + 2 * GH), windows=windows_cd,
        rule_capacity=int(cd_handle.get("chi0_rule_capacity") or cd_rules.CD_RULE_CAPACITY),
        rules_file=rules_file or None, time_reversal=trs,
        save_path=None if (rules_file or run_dir is None) else os.path.join(run_dir, "cd_chi0_rules.pkl"),
        allow_partial=instrument, print_fn=print_fn)
    clock("chi0_rule", t0)
    _say(print_fn, started, f"chi0 rule ({chi_rule['source']}): {chi_rule['points']} points, "
         f"{len(chi_rule['rule_groups'])} groups in {chi_rule['windows']} windows, {chi_rule['rule_nodes']} nodes, "
         f"tolerance {chi_rule['rule_tolerance']:.1e}, max sampled error "
         f"{max(g['sampled_error'] for g in chi_rule['rule_groups']):.2e}")

    # ---- Sigma_x identity (INVARIANT: the CD pair densities reproduce the static exchange) ------------
    t0 = time.monotonic()
    sx_cd = K.sigma_x(to_qpar(v_full), zeros(n_rows, nb_s, nb_s), tabs)
    sx_cd = sx_cd[:nk_red] if cert else sx_cd
    jax.block_until_ready(sx_cd)
    del v_full                                                     # the full-zone V_q is read by Sigma_x only
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

    # ---- the chunk loops (stream_cd_chunks): resumable from cd_checkpoint, stoppable by LORRAX_CD_STOP_AFTER ----
    ckpt_mode = str(cd_handle.get("checkpoint", "off") or "off").strip().lower()
    if ckpt_mode not in CD_CHECKPOINT_MODES:
        raise ValueError(f"GATE cd_checkpoint: cd_checkpoint got: {ckpt_mode!r}; want: {' | '.join(CD_CHECKPOINT_MODES)}")
    imag_chunks = [(c0, c0 + chunk_i) for c0 in range(0, n_imag_pad, chunk_i)]
    real_chunks = [(j0, j0 + chunk_r) for j0 in range(0, J_pad, chunk_r)]
    ckpt = resume = None
    if ckpt_mode != "off":
        import hashlib
        from gw.kin_ion_io import _generator_commit
        if run_dir is None:
            raise ValueError("GATE cd_checkpoint: cd_checkpoint got: a mode with no run directory; want: a deck run")
        # Everything the accumulators depend on; json round trip so the stored record compares equal.
        identity = json.loads(json.dumps(dict(
            deck_sha256=cd_deck_digest(getattr(config, "input_file", "") or ""), tree=_generator_commit(),
            recipe=CD_RECIPE, chi0=CD_RECIPE.get("chi0"),
            chi0_rule_tolerance=float(cd_handle.get("chi0_rule_tolerance", CD_CHI0_RULE_TOLERANCE)),
            chi0_rule_capacity=cd_handle.get("chi0_rule_capacity"), chi0_rule_nodes=chi_rule.get("rule_nodes"),
            chi0_rule_windows=chi_rule.get("windows"),
            n_packed=n, nk=nk, nq_irr=nq_irr, nk_red=nk_red, rows=rows_all.tolist(), window=[m0, m1],
            own_window=[m0 + o0, m1], l_sum=[l0, l1], n_omega=nE,
            omega_sha256=hashlib.sha256(np.ascontiguousarray(omega).tobytes()).hexdigest(), eta_ry=eta,
            knot_step_ry=h_ry, J=J, J_pad=J_pad, imaginary_nodes=n_imag, n_imag_pad=n_imag_pad,
            chunk_imag=chunk_i, chunk_real=chunk_r, stop_after_imag=stop_after["imag"],
            stop_after_real_from=stop_after["real_from"]), default=str))
        ckpt = CdCheckpoint(os.path.join(run_dir, CD_CHECKPOINT_FILE), identity, mesh_xy, print_fn=print_fn)
        if ckpt_mode == "resume":
            resume = ckpt.read()
        _say(print_fn, started, f"checkpoint {ckpt_mode}: {ckpt.path} " + (
            f"resumes after {resume['axis']} chunk {resume['chunk']} (commit {resume['commit'][:16]})"
            if resume is not None else "(fresh)"))
    if instrument:
        _say(print_fn, started, f"LORRAX_CD_STOP_AFTER {json.dumps(stop_after)}: instrument run, no Sigma is finalized")
    chi_started, chi0_calls = [None], []

    def imag_step(ic, c0, c1, accs):
        t_chunk = t0 = time.monotonic()
        chi = bank_chi(imag_pts[ic])
        jax.block_until_ready(chi)
        clock("chi0", t0)
        chi0_calls.append(dict(axis="imag", chunk=int(ic), points=int(len(imag_pts[ic])),
                               seconds=round(time.monotonic() - t0, 3)))
        t0 = time.monotonic()
        w_flat = dyson_value(roots_qz(h_irr, chunk_i), to_qz(chi))
        del chi
        jax.block_until_ready(w_flat)
        clock("dyson", t0)
        t0 = time.monotonic()
        W = full_zone(w_flat, chunk_i)
        del w_flat
        jax.block_until_ready(W)
        clock("unfold", t0)
        t0 = time.monotonic()
        acc, own = K.imag(W, u_dev[c0:c1], w_dev[c0:c1], accs["acc_imag"], accs["own_imag"], tabs)
        jax.block_until_ready(acc)
        clock("contract_imag", t0)
        del W
        ledger.append(ledger_sample("imag", ic, time.monotonic() - t_chunk))
        _say(print_fn, started, f"imaginary nodes {min(c1, n_imag)}/{n_imag} " + json.dumps({k: round(v, 1) for k, v in timings.items()}))
        return dict(accs, acc_imag=acc, own_imag=own)

    def chi_real(ic):
        """chi0 at real chunk ``ic``'s planned window (gw.cd.chi0_rules.cd_windows: chunk 0 with both ghost
        sides, later chunks their new knots); the ghost roll is stream_cd_chunks'."""
        chi_started[0] = t0 = time.monotonic()
        chi_e = bank_chi(real_pts[ic])
        jax.block_until_ready(chi_e)
        clock("chi0", t0)
        chi0_calls.append(dict(axis="real", chunk=int(ic), points=int(len(real_pts[ic])),
                               seconds=round(time.monotonic() - t0, 3)))
        return chi_e

    def real_step(ic, j0, j1, box, accs):
        t_chunk, t0 = chi_started[0] or time.monotonic(), time.monotonic()
        chi_e = box.pop()
        z_core = jnp.asarray(real_ry[j0:j1] + 1j * eta)
        dchi = slope_fn(chi_e, z_core)
        chi_c = core_fn(chi_e)
        del chi_e
        wc_flat, dw_flat = dyson_pair(roots_qz(h_irr, chunk_r), to_qz(chi_c), to_qz(dchi))
        del chi_c, dchi
        jax.block_until_ready(dw_flat)
        clock("dyson", t0)
        t0 = time.monotonic()
        W, dW = full_zone(wc_flat, chunk_r), full_zone(dw_flat, chunk_r)
        del wc_flat, dw_flat
        jax.block_until_ready(dW)
        clock("unfold", t0)
        acc_a, own_a = accs["acc_anc"], accs["own_anc"]
        if j0 == 0:
            t0 = time.monotonic()
            acc_a, own_a = K.anchor(W[0], dW[0], acc_a, own_a, tabs)
            jax.block_until_ready(acc_a)
            clock("anchor", t0)
            partial_dump("anchor", sigma_c_anchor_own_ry=own_a, sigma_c_imag_own_ry=accs["own_imag"])
        t0 = time.monotonic()
        acc_r, own_r = accs["acc_real"], accs["own_real"]
        bands, negative = band_tables(ic)
        if bands:                                       # a chunk no live residue reads adds an exact zero
            acc_r, own_r = K.real(W, dW, z_core, jnp.int32(j0), acc_r, own_r, tabs,
                                  tuple(_rep(b) for b in bands), negative)
            jax.block_until_ready(acc_r)
        clock("contract_real", t0)
        del W, dW
        ledger.append(ledger_sample("real", ic, time.monotonic() - t_chunk))
        if ic % 10 == 0 or j1 >= J_pad:
            _say(print_fn, started, f"real knots {min(j1, J)}/{J} " + json.dumps({k: round(v, 1) for k, v in timings.items()}))
        return dict(accs, acc_anc=acc_a, own_anc=own_a, acc_real=acc_r, own_real=own_r)

    def on_chunk(axis, ic, accs):
        if axis == "imag" and ic == len(imag_chunks) - 1:
            partial_dump("imag", sigma_c_imag_own_ry=accs["own_imag"])
        if ckpt is not None:
            t0 = time.monotonic()
            ckpt.write(axis, ic, accs, timings=timings,
                       host=dict(n_imag_chunks=len(imag_chunks), n_real_chunks=len(real_chunks)))
            clock("checkpoint", t0)

    accs, loop_info = stream_cd_chunks(
        imag_chunks=imag_chunks, real_chunks=real_chunks, chi_real=chi_real, join=join_fn, tail_of=tail_fn,
        imag_step=imag_step, real_step=real_step, resume=resume, on_chunk=on_chunk, stop_after=stop_after,
        accs=dict(acc_imag=acc_imag, own_imag=own_imag, acc_real=acc_real, own_real=own_real,
                  acc_anc=acc_anc, own_anc=own_anc))
    acc_imag, own_imag, acc_real, own_real, acc_anc, own_anc = (
        accs[k] for k in ("acc_imag", "own_imag", "acc_real", "own_real", "acc_anc", "own_anc"))
    del accs
    ckpt_note = dict(mode=ckpt_mode, path=None if ckpt is None else ckpt.path,
                     writes=0 if ckpt is None else ckpt.writes, seconds=0.0 if ckpt is None else ckpt.seconds,
                     state_bytes=0 if ckpt is None else ckpt.nbytes,
                     seconds_before_resume=None if resume is None else resume["timings"], **loop_info)
    if instrument and (
            loop_info["stopped"] or loop_info["imag_truncated"] or stop_after["real_from"] is not None):
        # LORRAX_CD_STOP_AFTER: the run is a timing probe.  Its record, then the same refusal on every rank.
        stop = dict(recipe=CD_RECIPE, stop_after=stop_after, loop=ckpt_note, seconds=timings,
                    elapsed_s=time.monotonic() - started, J=J, J_pad=J_pad, reach_ev=real_ry[J - 1] * EV,
                    need_ev=need * EV, need_window_ev=need_window * EV, imaginary_nodes=n_imag,
                    n_imag_pad=n_imag_pad, chunk_imag=chunk_i, chunk_real=chunk_r,
                    n_imag_chunks=len(imag_chunks), n_real_chunks=len(real_chunks), nk=nk, nq_irr=nq_irr,
                    nq_pad=nq_pad, n_packed=n, window=[m0, m1], own_window=[m0 + o0, m1], ndev=ndev,
                    bytes_per_node_imag=chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis="imag"),
                    bytes_per_knot_real=chunk_bytes_per_unit(n, nq_irr, nq_pad, ndev, axis="real"),
                    sigma_x_identity=sx_note, residue_bands=band_summary, chi0_rule=chi_rule,
                    chi0_calls=chi0_calls, band_tile=min(band_tile, L), slab_block=slab_block,
                    qz_flat_rows=[QZ.flat_rows(chunk_i), QZ.flat_rows(chunk_r)],
                    memory_plan_gb=dict({k: v / 1e9 for k, v in mem_plan["phases"].items()},
                                        peak=mem_plan["peak_bytes"] / 1e9, peak_phase=mem_plan["peak_phase"]))
        if rank0 and run_dir is not None:
            with open(os.path.join(run_dir, "cd_stop_receipt.json"), "w") as fh:
                json.dump(stop, fh, indent=1, default=str)
            with open(os.path.join(run_dir, "cd_leak_ledger.csv"), "w") as fh:
                cols = list(ledger[0])
                fh.write(",".join(cols) + "\n")
                for row in ledger:
                    fh.write(",".join(str(row[c]) for c in cols) + "\n")
        barrier("cd_stop_after")
        _say(print_fn, started, "stop-after reached " + json.dumps({k: round(v, 1) for k, v in timings.items()}))
        raise CdStopAfter(f"LORRAX_CD_STOP_AFTER {json.dumps(stop_after)} reached after {json.dumps(loop_info)}; "
                          "cd_stop_receipt.json written; no Sigma is finalized (instrument run)")

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
                   residue_bands=band_summary, chi0_rule=chi_rule, band_tile=min(band_tile, L),
                   psi_slab_bands=[L, L_neg], slab_block=slab_block, qz_flat_rows=[QZ.flat_rows(chunk_i), QZ.flat_rows(chunk_r)],
                   memory_plan_gb=dict({k: v / 1e9 for k, v in mem_plan["phases"].items()},
                                       peak=mem_plan["peak_bytes"] / 1e9, peak_phase=mem_plan["peak_phase"]),
                   own_energy_band_min=m0 + o0 + 1, own_window=[m0 + o0, m1], need_window_ev=need_window * EV,
                   checkpoint=ckpt_note)
    if rank0 and run_dir is not None:
        np.savez(os.path.join(run_dir, "cd_sigma_own_energy.npz"), kw_rows=kw_rows, bands0=np.arange(m0 + o0, m1),
                 window_bands0=np.arange(m0, m1),
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
