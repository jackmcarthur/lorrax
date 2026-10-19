"""Where the mini-BZ Coulomb average is placed at q != 0 — the head channel.

WHAT THIS MODULE IS FOR
-----------------------
At q = 0 LORRAX averages the whole screened quantity over the mini-BZ
Voronoi cell and adds the resulting scalar back as a rank-1 update AFTER
the Dyson solve (``vcoul/bulk_3d.py:81-90`` -> ``head_correction.
apply_q0_head_rank1``).  **That machinery is not touched here, by
construction: every path in this module is a no-op at Γ.**

At q != 0 there is a second, separate machinery.
``build_v_head_miniBZ_fn_3d`` averages the BARE Coulomb potential over the
cell and ``vcoul.base.v_qG_table`` substitutes that average into the
``argmin |q+G|`` slots of the one production ``V`` tile — which is then
used as *both* the Dyson operator and the Dyson right-hand side
(``w_isdf.py:382-384``).  So in the head channel LORRAX evaluates

    W_LX(q) = <v>(q) / (1 - <v>(q) chi_c(q))

where BerkeleyGW evaluates (``Sigma/mtxel_cor.f90:1659-1662``,
``BSE/intkernel.f90:887``, both a ``vcoul(igp) * eps`` product on an
``eps^-1`` built from the unaveraged ``v``)

    W_BGW(q) = <v>(q) * eps_c^-1(q) ,   eps_c^-1 = 1 / (1 - v_c chi_c) .

The two agree at eta = 0 and differ at first order in
``eta = <v>/v_c - 1``: the ratio of the two mini-BZ corrections is
``eps(q) + (eps(q)-1) eta(q)``, a suppression of LORRAX's correction by
very nearly the dielectric constant.  The intended object is
``<W>_C = <v eps^-1>_C``; under the f-sum-rule scaling ``chi ~ q^2`` the
product ``v chi`` is constant across the cell, so ``eps_c^-1 <v>`` is that
average EXACTLY and LORRAX's form is a quadrature of nothing.  Full
derivation and the falsification of the competing ``<v/(1-v chi_c)>``
candidate: ``lorrax_bse_perf_2026-08-08/COULOMB_AVG_ARCHITECTURE.md``
sections 2.2-2.4.

THE MECHANISM: A SCALAR PER q-CELL, APPLIED AFTER THE SOLVE
-----------------------------------------------------------
The fix shape is the one ``BGW_ATTRIBUTION.md:398-407`` asked for and
``gw.experimental.head_wing_schur`` already implements algebraically:
make the q != 0 head an explicit channel with a SCALAR coefficient and put
the average on that scalar.  Writing ``V = V_body + v_head * P`` with
``P = sum_j conj(g_j) (x) g_j`` the head-slot projector in the centroid
basis, the Schur/Woodbury reduction of ``(I - V chi)^-1 V`` is exactly

    W = W_body0 + L . S . R ,
    W_body0 = (I - V_body chi)^-1 V_body                    (the body solve)
    L       = conj(G) + W_body0 chi conj(G)                 (left  wing)
    R       = G^T     + G^T chi W_body0                     (right wing)
    S       = v_head (I - v_head chi_eff)^-1                (the head scalar)
    chi_eff = G^T chi conj(G) + G^T chi W_body0 chi conj(G)

— which is ``head_wing_schur``'s ``extract_V_body`` /
``solve_W_body0`` / ``schur_reductions`` / ``W_head_scalar_per_q`` /
``assemble_W`` chain term for term (that module's ``chi_body`` argument is
the FULL chi; only ``V`` is decomposed).  ``S`` is the whole head channel
and it is the ONLY place ``v_head`` enters.  Therefore replacing the
numerator ``v_head -> <v>`` while leaving the denominator at the bare
``v_c`` gives BerkeleyGW's ``<v> eps_c^-1`` and nothing else moves; and
because ``S`` scales by the pure ratio ``r = <v>/v_c`` uniformly over the
tied set, the whole correction is **one real scalar per q-cell applied to
W's head channel after the solve**.

Hermiticity is structural, not restored by hand.  ``S`` is real, and
``conj(R_nu) = L_nu`` whenever ``chi`` and ``W_body0`` are Hermitian, so
``L . S . R`` is Hermitian; equivalently the identity below writes W as a
REAL combination of two single-V Dyson solves, each Hermitian by
congruence (``(I-V chi)^-1 V = V^1/2 (I - V^1/2 chi V^1/2)^-1 V^1/2``).
This is the property a two-tile ``(I - V_bare chi)^-1 V_avg`` form breaks
and ``test_alpha_herm_head_slot_link``'s ``ALPHA_HERM_RTOL`` asserts.

HOW IT IS COMPUTED HERE
-----------------------
``L . S . R = W(v_head) - W_body0`` for any ``v_head``, since W_body0 is
the same object in both.  So instead of forming the wings explicitly this
module evaluates the head channel by DIFFERENCE of two solves:

    W_body0 = solve(V - v_in_V * P,  chi)          head channel removed
    W_bare  = solve(V - (v_in_V - v_c) * P,  chi)  head channel at bare v_c
    W       = W_body0 + r * (W_bare - W_body0),    r = <v> / v_c

``v_in_V`` is whatever ``v_qG_table`` actually left in the head slots of
the production ``V`` (``<v>`` with ``mc_average_vcoul_body = true``, ``v_c``
without it), so ``V`` itself — the tile Sigma_X, Sigma_H, Sigma_SX,
Sigma_COH, ``Wc = W - V`` and the BSE exchange tile all read — is never
modified.  The by-difference route costs one extra LU per q and buys
exactness on the tied q: the argmin ties with multiplicity 2 and 4 on 13
of Si's 64 q, and a rank-1 Schur channel would silently correct only one
slot of each tied set.  Both routes are the same algebra; only the
arithmetic path differs.  The rank-1 subtractions themselves are
``head_wing_schur.extract_V_body_sharded``, applied once per tied slot.

At Γ ``mask`` is identically zero, so ``P = 0``, ``W_body0 = W_bare = W``,
``r = 1``, and every array above is the untouched production object.

MODES (deck key ``mc_average_placement``)
-----------------------------------------
``off``       — default.  This module is not reached; the q != 0 head keeps
                today's placement, byte for byte.
``bgw``       — BerkeleyGW parity: ``r = <v>/v_c`` per q != 0 cell, giving
                ``W_head = eps_c^-1 <v>``.  ``<v>`` comes from LORRAX's own
                ``build_v_head_miniBZ_fn_3d`` unless a BGW ``vcoul`` dump is
                supplied, in which case the ratio is taken byte-sourced from
                that file (the ``vhead``/``whead_0freq`` override pattern,
                one q-shell at a time).
``schur_avg`` — the derivable target ``<W>_C = <v eps^-1>_C`` with eps
                varying inside the cell.  REFUSED: the estimator is not
                built, and it cannot be until the falsification probe says
                which way chi runs across a Γ-adjacent cell (that question
                is open — on a 4x4x4 grid eps runs 21.97 -> 5.73 across one
                cell, so neither frozen model holds; see
                COULOMB_AVG_ARCHITECTURE.md section 2.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PLACEMENT_OFF = "off"
PLACEMENT_BGW = "bgw"
PLACEMENT_SCHUR_AVG = "schur_avg"
PLACEMENT_MODES = (PLACEMENT_OFF, PLACEMENT_BGW, PLACEMENT_SCHUR_AVG)

#: Spellings accepted for the default.  ``dyson`` is the name the
#: architecture audit used for today's placement before the deck key was
#: written; it is accepted so a reader of that document can type it.
_OFF_ALIASES = ("off", "false", "none", "dyson")


def normalize_placement(value) -> str:
    """Canonicalise a ``mc_average_placement`` deck value.

    Refuses an unknown spelling rather than falling back to ``off`` — a
    Coulomb-placement knob that silently does nothing is the failure mode
    the ``vhead`` parser fix (``W_HEAD_ISDF_FLOOR.md`` section 2.3) was
    written to remove, and this key has exactly the same shape.
    """
    if value is None:
        return PLACEMENT_OFF
    s = str(value).strip().lower()
    if s in _OFF_ALIASES:
        return PLACEMENT_OFF
    if s in PLACEMENT_MODES:
        return s
    raise ValueError(
        f"mc_average_placement = {value!r} is not a recognised mode. "
        f"Valid: {PLACEMENT_OFF!r} (default, today's placement), "
        f"{PLACEMENT_BGW!r} (BerkeleyGW parity: <v> on W's head channel "
        f"after the Dyson solve), {PLACEMENT_SCHUR_AVG!r} (not implemented).")


def refuse_if_unimplemented(mode: str) -> None:
    """Raise for modes that are wired but deliberately not built."""
    if mode == PLACEMENT_SCHUR_AVG:
        raise NotImplementedError(
            "mc_average_placement = 'schur_avg' is wiring only. It names the "
            "derivable target <W>_C = <v eps^-1>_C with eps varying INSIDE "
            "the mini-BZ cell, which needs chi(q+u) inside the cell — a "
            "quantity neither LORRAX nor BerkeleyGW computes on a grid. "
            "The two frozen models bracket it (COULOMB_AVG_ARCHITECTURE.md "
            "section 2.4: on the 4x4x4 anchor the true <W> at the first "
            "shell lies in (62.40, 66.46) and today's value 63.07 is inside "
            "that window), so building an estimator before the k-grid "
            "convergence study would be fitting a number, not deriving one. "
            "Use mc_average_placement = 'bgw' for the BerkeleyGW-parity "
            "placement, or 'off' for today's.")


@dataclass(frozen=True)
class HeadChannel:
    """Everything the q != 0 head-channel rescale needs, per full-BZ q.

    ``g_head``  ``(nq, k, mu)`` complex128 — the centroid-basis columns
                ``zeta(q, mu, G_j)`` of the head slots, already
                mask-weighted (padding columns are exactly zero) and
                already unfolded to the full BZ.  The head-slot projector
                is ``P_q = sum_j conj(g_head[q,j]) (x) g_head[q,j]``; the
                columns have passed through the symmetry service's exact
                one-leg action (source G relabel, centroid/L/tau action and
                the measured antiunitary row policy).
    ``v_bare``  ``(nq,)`` float64 — unaveraged ``v`` at the head slot.
    ``v_avg``   ``(nq,)`` float64 — the mini-BZ cell average there.
    ``v_in_V``  ``(nq,)`` float64 — the value the production ``V`` tile
                actually carries in those slots.
    ``mult``    ``(nq,)`` int32 — tie multiplicity, for the log line.
    ``len2``    ``(nq,)`` float64 — ``|q+G|^2`` at the head slot, the shell
                label the BGW ``vcoul`` dump is matched on.
    ``mode``    the canonicalised placement mode.
    ``ratio_source`` provenance string for ``v_avg`` ("lorrax minibz" or
                the BGW dump path), printed at build time so a run's log
                answers where the enhancement came from.
    """
    g_head: object
    v_bare: np.ndarray
    v_avg: np.ndarray
    v_in_V: np.ndarray
    mult: np.ndarray
    len2: np.ndarray
    mode: str
    ratio_source: str = "lorrax minibz"

    @property
    def n_q(self) -> int:
        return int(np.asarray(self.v_bare).shape[0])

    @property
    def k_head(self) -> int:
        return int(self.g_head.shape[1])

    def head_ratio(self) -> np.ndarray:
        """``r(q) = <v>/v_c`` — the mini-BZ enhancement, 1 where inert.

        1.0 at Γ and at any q with no head slot, so the returned array can
        be applied unconditionally.
        """
        vb = np.asarray(self.v_bare, dtype=np.float64)
        va = np.asarray(self.v_avg, dtype=np.float64)
        return np.where(vb > 0.0, va / np.where(vb > 0.0, vb, 1.0), 1.0)

    def eta(self) -> np.ndarray:
        """``eta(q) = <v>/v_c - 1``, the per-shell enhancement."""
        return self.head_ratio() - 1.0

    def body_subtract(self) -> np.ndarray:
        """Scalar to remove from the head slots to reach ``V_body``."""
        return np.asarray(self.v_in_V, dtype=np.float64)

    def bare_subtract(self) -> np.ndarray:
        """Scalar to remove from the head slots to reach ``V`` at bare ``v_c``.

        Zero when ``mc_average_vcoul_body`` is off (``V`` already carries
        ``v_c``) and zero at Γ, so that arm is a structural no-op there.
        """
        return (np.asarray(self.v_in_V, dtype=np.float64)
                - np.asarray(self.v_bare, dtype=np.float64))

    def summary(self) -> str:
        r = self.head_ratio()
        vb = np.asarray(self.v_bare)
        live = vb > 0.0
        n_live = int(live.sum())
        if n_live == 0:
            return (f"  [head placement] mode={self.mode}: no q carries a "
                    f"head slot; the rescale is a structural no-op.")
        eta = r[live] - 1.0
        return (
            f"  [head placement] mode={self.mode}, source={self.ratio_source}: "
            f"{n_live} of {self.n_q} q carry a head channel "
            f"(tie mult max {int(np.asarray(self.mult).max())}, "
            f"k={self.k_head}); eta = <v>/v_c - 1 spans "
            f"[{eta.min():.5f}, {eta.max():.5f}], mean {eta.mean():.5f}. "
            f"Gamma is untouched by construction.")


# ---------------------------------------------------------------------------
# The two V arms and the post-solve combine
# ---------------------------------------------------------------------------

def _rank1_subtract(V_q, g_head, scale_q, mesh_xy):
    """``V_q - scale_q[q] * sum_j conj(g_j) (x) g_j``, one tied slot at a time.

    Delegates the per-column arithmetic to
    ``head_wing_schur.extract_V_body_sharded``, which is the tested,
    zero-communication form of exactly this outer product (its HLO is
    asserted collective-free in ``tests/test_head_wing_schur.py``).  Padding
    columns are exact zeros, so the loop trip count is a compile-time
    constant and short-circuiting on the per-q multiplicity would only
    cost a recompile.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from .experimental.head_wing_schur import (G0_X_SPEC, G0_Y_SPEC,
                                               extract_V_body_sharded)

    scale = jnp.asarray(np.asarray(scale_q, dtype=np.float64),
                        dtype=jnp.complex128)
    k = int(g_head.shape[1])
    out = V_q
    for j in range(k):
        col = g_head[:, j, :]
        g_x = jax.lax.with_sharding_constraint(
            col, NamedSharding(mesh_xy, G0_X_SPEC))
        g_y = jax.lax.with_sharding_constraint(
            col, NamedSharding(mesh_xy, G0_Y_SPEC))
        out = extract_V_body_sharded(out, g_x, g_y, scale, mesh_xy=mesh_xy)
    del P
    return out


def build_v_arms(V_q, hc: HeadChannel, mesh_xy, *, q_index=None):
    """Return ``(V_body, V_bare)`` for the two Dyson solves.

    ``q_index`` optionally selects a q-subset (the IBZ wedge, when the
    solve runs on it); it indexes the leading axis of ``V_q``,
    ``hc.g_head`` and every per-q scalar together, so the three can never
    fall out of register.
    """
    g_head = hc.g_head
    v_body = hc.body_subtract()
    v_bare = hc.bare_subtract()
    if q_index is not None:
        idx = np.asarray(q_index, dtype=np.int64)
        g_head = g_head[idx]
        v_body = v_body[idx]
        v_bare = v_bare[idx]
    V_body = _rank1_subtract(V_q, g_head, v_body, mesh_xy)
    if np.all(v_bare == 0.0):
        # ``mc_average_vcoul_body = false``: the production tile ALREADY
        # carries the bare head, so this arm is V itself.  Taking the
        # identity rather than subtracting zero keeps that arm bitwise
        # equal to the un-gated solve.
        V_bare = V_q
    else:
        V_bare = _rank1_subtract(V_q, g_head, v_bare, mesh_xy)
    return V_body, V_bare


def combine_head_channel(W_body0, W_bare, hc: HeadChannel, *, q_index=None):
    """``W = W_body0 + r * (W_bare - W_body0)``, the head-channel rescale.

    ``r`` is real and per-q, so this is a real combination of two
    Hermitian-by-congruence Dyson solutions: W's Hermiticity and its
    ``W_q = conj(W_{-q})`` reciprocity both survive by construction
    (``r(q) = r(-q)`` because ``|q+G|`` is, and the tied-set mean makes
    ``<v>`` a class function of the shell).
    """
    import jax.numpy as jnp

    r = hc.head_ratio()
    if q_index is not None:
        r = r[np.asarray(q_index, dtype=np.int64)]
    if np.all(r == 1.0):
        return W_bare
    r_dev = jnp.asarray(r, dtype=W_bare.dtype)[:, None, None]
    return W_body0 + r_dev * (W_bare - W_body0)


# ---------------------------------------------------------------------------
# BGW vcoul dump -> byte-sourced <v>/v_c
# ---------------------------------------------------------------------------

def head_ratio_from_bgw_dump(path, hc: HeadChannel, *, bvec, rtol=1e-6):
    """Replace ``v_avg`` with BerkeleyGW's own averaged ``vcoul``, by shell.

    A BGW ``write_vcoul`` dump lists, per (q, G) slot the run asked about,
    the fractional q, the Miller index, and ``vcoul`` under that run's
    ``cell_average_cutoff`` — the raw ``<8 pi/|q+G+dq|^2>`` before ``fact``
    (``vcoul/bgw_parity.py`` docstring).  Only the RATIO to the unaveraged
    value is taken, and for untruncated 3D the unaveraged value is exactly
    ``8 pi / |q+G|^2``, recoverable from the dump's own (q, G).  So ONE
    dump suffices and no volume convention has to be reconciled: the
    quantity crossing the code boundary is dimensionless.

    Matching is on the ``|q+G|^2`` SHELL, not on index order.  BGW's
    absorption path computes ``vcoul`` once per unique ``|q|^2`` and caches
    it (``BSE/intkernel.f90:404-416``), so its dump has one row per shell
    while LORRAX has one head slot per q; the shell is the only label the
    two files share.

    This is the ``vhead`` / ``whead_0freq`` override pattern one level out:
    pin the cross-code comparison to BerkeleyGW's byte values so what is
    left over cannot be a difference of Monte-Carlo estimators.  LORRAX's
    own ``<v>`` already reproduces BGW's to 7e-4 - 2e-3 relative on every
    shell (``TRACE_vavg_consumers.md`` section 4), so this is a tightening,
    not a correction — and the size of the change it makes is itself a
    measurement worth printing.
    """
    from vcoul import read_bgw_vcoul

    tab = read_bgw_vcoul(str(path))
    bvec_f = np.asarray(bvec, dtype=np.float64)

    ref_len2, ref_ratio = [], []
    for qf, gm, vc in zip(np.asarray(tab.q_fracs, dtype=np.float64),
                          tab.G_miller_per_q, tab.vcoul_per_q):
        gm = np.asarray(gm, dtype=np.float64).reshape(-1, 3)
        vc = np.asarray(vc, dtype=np.float64).reshape(-1)
        K = (qf[None, :] + gm) @ bvec_f          # (m, 3) Cartesian q+G
        l2 = np.sum(K * K, axis=1)
        good = l2 > 1e-12
        if not np.any(good):
            continue
        ref_len2.append(l2[good])
        # v_bare(K) = 8 pi / |K|^2 in the dump's own (pre-``fact``) units.
        ref_ratio.append(vc[good] * l2[good] / (8.0 * np.pi))
    if not ref_len2:
        raise ValueError(
            f"{path}: BGW vcoul dump carried no q+G != 0 rows; nothing to "
            f"source a mini-BZ enhancement from.")
    ref_len2 = np.concatenate(ref_len2)
    ref_ratio = np.concatenate(ref_ratio)

    vb = np.asarray(hc.v_bare, dtype=np.float64)
    l2 = np.asarray(hc.len2, dtype=np.float64)
    va = np.array(hc.v_avg, dtype=np.float64, copy=True)
    matched, worst = 0, 0.0
    for qi in range(vb.shape[0]):
        if vb[qi] <= 0.0:
            continue
        j = int(np.argmin(np.abs(ref_len2 - l2[qi])))
        if abs(ref_len2[j] - l2[qi]) > rtol * max(l2[qi], 1.0):
            raise ValueError(
                f"{path}: LORRAX head slot at |q+G|^2 = {l2[qi]:.8g} has no "
                f"shell in the BGW vcoul dump within rtol={rtol:g} (nearest "
                f"{ref_len2[j]:.8g}). Refusing rather than silently keeping "
                f"LORRAX's own <v> under a deck key that asked for BGW's.")
        r_bgw = float(ref_ratio[j])
        worst = max(worst, abs(r_bgw * vb[qi] - va[qi]) / max(va[qi], 1e-30))
        va[qi] = r_bgw * vb[qi]
        matched += 1
    if matched == 0:
        raise ValueError(
            f"{path}: no q in this run carries a head slot, so the BGW "
            f"vcoul override has nothing to bind to.")
    return HeadChannel(
        g_head=hc.g_head, v_bare=hc.v_bare, v_avg=va, v_in_V=hc.v_in_V,
        mult=hc.mult, len2=hc.len2, mode=hc.mode,
        ratio_source=(f"BGW vcoul dump {path} ({matched} shells matched; "
                      f"max shift vs LORRAX's own <v> = {worst:.3e} rel)"))


__all__ = [
    "PLACEMENT_OFF", "PLACEMENT_BGW", "PLACEMENT_SCHUR_AVG", "PLACEMENT_MODES",
    "normalize_placement", "refuse_if_unimplemented",
    "HeadChannel", "build_v_arms", "combine_head_channel",
    "head_ratio_from_bgw_dump",
]
