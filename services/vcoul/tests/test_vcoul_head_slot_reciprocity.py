"""THE gate for the 2026-08-08 head-slot rule: v(+q) vs v(−q), at v level.

``V_q = conj(V_{−q})`` is a symmetry of the exact theory, and V_q is
bilinear in ζ over ``v(q+G)``, so if ``v`` itself is antisymmetric-free —
``v(+q, i) == v(−q, pair(i))`` for the slot pairing that sends
``K ↦ −K`` — then no downstream arm (direct full-BZ, IBZ + unfold, BSE)
can inherit a violation from the Coulomb head.  That makes this the
cheapest possible place to hold the rule, and the only one that is
arm-independent.

Everything here is the SERVICE's own arithmetic on the SERVICE's own
geometry: the per-q G-lists are built with ``bare_coulomb_sphere_mask``,
which is the predicate the ζ writer uses, so these are the on-disk lists
(verified against the real si_bse_debug fixture in the 2026-08-08 port —
all 64 q reproduced exactly).  No HDF5, no lorrax, no deck.

Four arms:

1. the pairing exists and is exactly ``K ↦ −K``;
2. the argmin is DEGENERATE on 13 of 64 q, stably across nine decades of
   tie tolerance, and 7 q are self-paired with the pairing SWAPPING their
   tied slots — which is the proof that no single-slot rule can work;
3. the acceptance: with the argmin + tied-mean rule the v-level mismatch
   is EXACTLY 0.0;
4. the RED TWIN: the retired Miller-(0,0,0) label rule, reinstated here
   locally, fails the same gate by ~1e-2.  Arm 3 without arm 4 would be a
   gate that cannot tell whether it is measuring anything.
"""
import numpy as np
import pytest

import vcoul as V

# The si_bse_debug / si_cohsex_debug fixture's blat * bvec (rows, 1/bohr).
# Pinned to the on-disk WFN header by tests/test_vcoul_minibz_head_draw.py
# on the lorrax side; repeated here as a literal because the service may
# not read a lorrax fixture.
_SI_C = 0.612323844648
SI_BVEC = _SI_C * np.array([[1.0, 1.0, -1.0],
                            [-1.0, 1.0, 1.0],
                            [1.0, -1.0, 1.0]])
SI_KGRID = (4, 4, 4)
SI_CELVOL = abs(8.0 * np.pi ** 3 / np.linalg.det(SI_BVEC))
#: Small enough to keep the cell fast, large enough that every q's sphere
#: is many shells deep.  The argmin structure is a property of the SMALLEST
#: |q+G| and is identical at the production cutoff (25 Ry / ngkmax 588).
_FFT_GRID = (12, 12, 12)
_CUTOFF_RY = 6.0


def _bgw_wrap_q(q_int, kgrid):
    kg = np.asarray(kgrid, dtype=np.float64)
    q = np.asarray(q_int, dtype=np.float64)
    return np.where(q > kg / 2, q - kg, q)


def _full_bz(kgrid):
    """(q_int, q_frac) over the full BZ, in the writer's BGW-wrap
    convention: wrap to (−k/2, k/2], THEN divide by kgrid."""
    nkx, nky, nkz = kgrid
    q_int = np.array([(x, y, z) for x in range(nkx)
                      for y in range(nky) for z in range(nkz)],
                     dtype=np.int64)
    return q_int, _bgw_wrap_q(q_int, kgrid) / np.asarray(kgrid, np.float64)


def _per_q_glists(bvec, kgrid, fft_grid, cutoff):
    """Per-q ``{G : |q+G|² ≤ cutoff}``, padded to a uniform ngkmax with the
    FFT-box corner — the WFN.h5 layout ``v_qG_table`` consumes."""
    q_int, q_frac = _full_bz(kgrid)
    mask, G_int = V.bare_coulomb_sphere_mask(fft_grid, bvec, q_frac, cutoff)
    ngk = mask.sum(axis=1).astype(np.int64)
    ngkmax = int(ngk.max())
    corner = np.array([int(s) // 2 for s in fft_grid], dtype=np.float64)
    gvec = np.empty((q_frac.shape[0], 3, ngkmax), dtype=np.float64)
    gvec[...] = corner[None, :, None]
    for qi in range(q_frac.shape[0]):
        sel = np.nonzero(mask[qi])[0]
        gvec[qi, :, :sel.size] = G_int[sel].T
    return q_int, q_frac, gvec, ngk


def _pairing(q_int, q_frac, gvec, ngk, bvec, kgrid):
    """``(neg_q, pair, max_K_sum)``.

    ``neg_q[qi]`` is the row of −q; ``pair[qi, i]`` is the slot at −q whose
    Cartesian K is exactly ``−K_i(+q)``, or −1 if the match failed."""
    kg = np.asarray(kgrid, dtype=np.int64)
    where = {tuple(int(v) for v in q_int[i]): i for i in range(len(q_int))}
    neg_q = np.array([where[tuple((-q_int[i]) % kg)]
                      for i in range(len(q_int))])
    K = np.einsum("qdn,de->qne", gvec, bvec) + (q_frac @ bvec)[:, None, :]
    pair = np.full(gvec.shape[::2], -1, dtype=np.int64)
    worst = 0.0
    for qi in range(len(q_int)):
        qj = int(neg_q[qi])
        A, B = K[qi, :int(ngk[qi])], K[qj, :int(ngk[qj])]
        lut = {tuple(np.round(-B[j], 9)): j for j in range(B.shape[0])}
        for i in range(A.shape[0]):
            j = lut.get(tuple(np.round(A[i], 9)))
            if j is None:
                continue
            pair[qi, i] = j
            worst = max(worst, float(np.max(np.abs(A[i] + B[j]))))
    return neg_q, pair, worst, K


@pytest.fixture(scope="module")
def si():
    q_int, q_frac, gvec, ngk = _per_q_glists(
        SI_BVEC, SI_KGRID, _FFT_GRID, _CUTOFF_RY)
    neg_q, pair, worst, K = _pairing(
        q_int, q_frac, gvec, ngk, SI_BVEC, SI_KGRID)
    return dict(q_int=q_int, q_frac=q_frac, gvec=gvec, ngk=ngk,
                neg_q=neg_q, pair=pair, worst=worst, K=K)


# ---------------------------------------------------------------------------
# Arm 1 — the pairing is exactly K -> -K.
# ---------------------------------------------------------------------------

def test_the_slot_pairing_is_exactly_K_to_minus_K(si):
    """Every real slot at +q has a partner at −q with ``K_j = −K_i``, to
    machine zero.  This is WHY argmin works at all: |K| is invariant under
    the pairing, so the argmin SET maps onto the argmin SET."""
    real = np.arange(si["gvec"].shape[2])[None, :] < si["ngk"][:, None]
    assert not np.any((si["pair"] < 0) & real), (
        "some real slot at +q has no K ↦ −K partner at −q — the per-q "
        "sphere is not closed under the pairing and every claim below is "
        "about a different geometry than production's")
    assert si["worst"] < 1e-12, si["worst"]


# ---------------------------------------------------------------------------
# Arm 2 — the degeneracy, and why one slot is impossible.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rtol", [1e-14, 1e-12, 1e-10, 1e-9, 1e-8, 1e-6])
def test_argmin_multiplicity_histogram_is_tolerance_stable(si, rtol):
    """1:51, 2:7, 4:6 — identical across nine decades of tie tolerance
    (Γ counted as multiplicity 1).  These are exact symmetry degeneracies,
    not knife-edges, which is what makes ``head_tie_rtol``'s value a
    non-decision rather than a tuned constant."""
    hist = {}
    for qi in range(len(si["q_int"])):
        d = np.sum(si["K"][qi, :int(si["ngk"][qi])] ** 2, axis=1)
        dmin = float(d.min())
        m = (1 if dmin <= 1e-12                      # Γ, skipped by the rule
             else int(np.count_nonzero(d <= dmin * (1.0 + rtol))))
        hist[m] = hist.get(m, 0) + 1
    assert hist == {1: 51, 2: 7, 4: 6}, hist


def test_the_degeneracies_are_not_knife_edges(si):
    """The smallest relative gap to the next distinct |q+G|² among the
    non-degenerate q is 7.3e−1 — six decades above ``head_tie_rtol``, so
    no q can drift across the tie window."""
    gaps = []
    for qi in range(len(si["q_int"])):
        d = np.sort(np.sum(si["K"][qi, :int(si["ngk"][qi])] ** 2, axis=1))
        if d[0] > 1e-12 and d[1] > d[0] * (1.0 + 1e-9):
            gaps.append((d[1] - d[0]) / d[0])
    assert min(gaps) > 0.7, min(gaps)


def test_seven_q_are_self_paired_with_their_tied_slots_SWAPPED(si):
    """THE IMPOSSIBILITY PROOF, measured.

    At these q, ``−q ≡ q`` as a grid index and the pairing exchanges the
    two tied slots.  Injecting at one and not its partner makes
    ``V_q ≠ conj(V_q)`` by construction — so no tie-break, however clever,
    can rescue a one-slot rule.  All of the argmin, or nothing.
    """
    found = []
    for qi in range(len(si["q_int"])):
        if int(si["neg_q"][qi]) != qi:
            continue
        d = np.sum(si["K"][qi, :int(si["ngk"][qi])] ** 2, axis=1)
        dmin = float(d.min())
        if dmin <= 1e-12:
            continue
        sel = np.nonzero(d <= dmin * (1.0 + 1e-9))[0]
        if sel.size > 1 and any(si["pair"][qi, s] != s for s in sel):
            found.append(qi)
    assert len(found) == 7, [tuple(si["q_frac"][i]) for i in found]


@pytest.mark.parametrize("kgrid,expect", [((4, 4, 4), 7), ((2, 2, 2), 7),
                                          ((3, 3, 3), 0), ((5, 5, 5), 0)])
def test_even_grids_have_the_swap_and_odd_grids_do_not(kgrid, expect):
    """The self-paired-with-swap q are a parity property of the grid, not
    an accident of Si 4×4×4: every EVEN grid has exactly 7, every ODD grid
    none.  The odd rows are the case where the impossibility does NOT
    hold — without them "7" would look like a number about silicon."""
    n = max(kgrid) * 3
    q_int, q_frac, gvec, ngk = _per_q_glists(
        SI_BVEC, kgrid, (n, n, n), _CUTOFF_RY)
    neg_q, pair, _w, K = _pairing(q_int, q_frac, gvec, ngk, SI_BVEC, kgrid)
    n_swap = 0
    for qi in range(len(q_int)):
        if int(neg_q[qi]) != qi:
            continue
        d = np.sum(K[qi, :int(ngk[qi])] ** 2, axis=1)
        dmin = float(d.min())
        if dmin <= 1e-12:
            continue
        sel = np.nonzero(d <= dmin * (1.0 + 1e-9))[0]
        if sel.size > 1 and any(pair[qi, s] != s for s in sel):
            n_swap += 1
    assert n_swap == expect


# ---------------------------------------------------------------------------
# Arms 3 and 4 — the acceptance, and the red twin that must fail it.
# ---------------------------------------------------------------------------

_NMC = 512          # the acceptance is a SYMMETRY: exact at any sample count


def _worst_v_mismatch(v, si):
    worst = 0.0
    for qi in range(len(si["q_int"])):
        qj = int(si["neg_q"][qi])
        for i in range(int(si["ngk"][qi])):
            j = int(si["pair"][qi, i])
            if j >= 0:
                worst = max(worst, abs(float(v[qi, i]) - float(v[qj, j])))
    return worst


def _label_rule_v(si, head_fn):
    """THE RETIRED RULE, reimplemented locally: inject at the slot whose
    Miller index is literally (0,0,0), valued at ``q_frac @ bvec``.

    This is the red twin's whole content.  It is written out here rather
    than reached for in production because production no longer has it —
    and a gate whose failing case has been deleted is a gate nobody can
    check.
    """
    v = V.v_qG_table(V.get_kernel(3), si["q_frac"], si["gvec"],
                     geometry=V.CoulombGeometry(bvec=SI_BVEC,
                                                cell_volume=SI_CELVOL))
    v = np.array(v, dtype=np.float64, copy=True)
    for qi in range(len(si["q_int"])):
        q_cart = si["q_frac"][qi] @ SI_BVEC
        if float(q_cart @ q_cart) < 1e-12:
            continue                              # Γ: head handled elsewhere
        g0 = np.nonzero(np.all(si["gvec"][qi] == 0.0, axis=0))[0]
        v[qi, g0] = float(head_fn(q_cart.reshape(1, 3))[0])
    return v


def test_the_argmin_tied_mean_rule_makes_v_exactly_reciprocal(si):
    """THE ACCEPTANCE.  ``max |v(+q, i) − v(−q, pair(i))| == 0.0``, exactly
    — not "small", zero, because both sides are the same float64 mean of
    the same sample set."""
    head_fn = V.build_v_head_miniBZ_fn_3d(
        SI_KGRID, SI_BVEC, SI_CELVOL, nmc=_NMC)
    v = V.v_qG_table(V.get_kernel(3), si["q_frac"], si["gvec"],
                     geometry=V.CoulombGeometry(bvec=SI_BVEC,
                                                cell_volume=SI_CELVOL),
                     v_head_fn=head_fn)
    assert _worst_v_mismatch(v, si) == 0.0


def test_the_bare_kernel_was_already_reciprocal(si):
    """Anti-attribution: ``v`` with NO head is reciprocal too, so anything
    the twin below measures is the head slot and not the bare formula."""
    v = V.v_qG_table(V.get_kernel(3), si["q_frac"], si["gvec"],
                     geometry=V.CoulombGeometry(bvec=SI_BVEC,
                                                cell_volume=SI_CELVOL))
    assert _worst_v_mismatch(v, si) == 0.0


def test_RED_TWIN_the_miller_zero_label_rule_fails_this_gate(si):
    """The case where the check comes out FALSE.

    Reinstating the Miller-(0,0,0) label rule must break reciprocity by
    ~1e−2 (MEASURED at the production cutoff and nmc=2**18: 1.293e−2).
    If this ever passes, the gate above has stopped discriminating and the
    label rule could walk back in unnoticed.
    """
    head_fn = V.build_v_head_miniBZ_fn_3d(
        SI_KGRID, SI_BVEC, SI_CELVOL, nmc=_NMC)
    worst = _worst_v_mismatch(_label_rule_v(si, head_fn), si)
    assert worst > 1e-3, (
        f"the retired Miller-(0,0,0) rule scored {worst:.3e} on the "
        f"reciprocity gate — it is supposed to FAIL it by ~1e-2; the gate "
        f"has lost its discriminating power")


def test_RED_TWIN_a_non_centrosymmetric_head_fn_also_fails(si):
    """The OTHER half of the rule, falsified separately.

    Right slots, wrong sample set: a head built on a one-sided δq draw is
    even in K only in the ``nmc → ∞`` limit, so it leaves an MC-sized
    residual even with the argmin selection.  This is why
    ``build_miniBZ_dq_cart`` closes the draw under negation instead of
    leaning on sample size.
    """
    dq = V.build_miniBZ_dq_cart(SI_KGRID, SI_BVEC, nmc=_NMC)[:_NMC]
    fact = 1.0 / SI_CELVOL

    def one_sided(K_cart):
        K = np.asarray(K_cart, dtype=np.float64).reshape(-1, 3)
        out = np.empty(K.shape[0], dtype=np.float64)
        for i in range(K.shape[0]):
            v, _ = V._minibz_kernel_bare(K[i], dq, kind="bulk_3d")
            out[i] = np.mean(v)
        return out * fact

    v = V.v_qG_table(V.get_kernel(3), si["q_frac"], si["gvec"],
                     geometry=V.CoulombGeometry(bvec=SI_BVEC,
                                                cell_volume=SI_CELVOL),
                     v_head_fn=one_sided)
    assert _worst_v_mismatch(v, si) > 0.0, (
        "a one-sided mini-BZ draw came out exactly reciprocal — either the "
        "draw is symmetric by accident at this seed, or the injection "
        "stopped evaluating slots at their own K")
