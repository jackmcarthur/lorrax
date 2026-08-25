"""Layer L-a: the pure-array surface — k-grid algebra, orbits, the ψ rule.

These are the service's numpy functions of integer tables: ``kgrid_shift_map``,
``find_irreducible_bz_points``, ``unfold_psi`` /
``spinor_rotation_for_sym_row`` / ``tau_phase_row``,
``centroid_source_map_and_wrap``, ``fft_grid_pullback_perm`` and
``recover_symmorphic_density_point_group``.  No deck, no mesh, no HDF5 except
in the one metric-convention cell, which reads a header and nothing else.

DELIBERATE NON-OVERLAP with ``tests/test_symmetry_unfold.py``.  That file
(750 lines, 17 cells) already covers ``find_irreducible_bz_points``'s five
shapes, ``centroid_source_map_and_wrap``'s ``extend_trs`` shape+content, the
non-symmorphic τ path, the open-orbit refusal, ``unfold_isdf_operator``'s TRS
rows,
and ``unfold_psi``'s hand reference / identity no-op / Θ² = −1.  It stays
in ``tests/`` until the replumb.  What is here is the GAPS the survey
ranked (§7.2) and the contracts the service owes its callers:

* G7 ``validate_kgrid_unfolding`` — kept rather than deleted (design
  decision 3), so it gets a passing case AND a constructible-FALSE case;
* G8 the ``ntran <= 1`` trivial branch's TRS augmentation against the
  general branch (``tests/test_symmetry_unfold.py:706`` covers the SHAPE
  only, and §Q's blast radius was V_NL collapse plus O(100 eV) in V_loc);
* G9 ``recover_symmorphic_density_point_group`` — 19683 candidate matrices,
  no test anywhere;
* S6 ``centroid_source_map_and_wrap``'s orbit-closure refusal, BOTH halves,
  driven by a SYNTHETIC dropped centroid (survey §8.2: no
  ``centroids_frac_*.txt`` is ever regenerated, and no deck's production
  set is ever re-run through kmeans);
* the Cartesian metric-convention discriminator (design decision 7), which
  is the cleanest executable statement of the ``R_cart`` transpose trap and
  needs no CrI3 fixture.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from _deck_stub import deck_available, read_deck
from symmetry_maps import (SymMaps, centroid_source_map_and_wrap,
                           common_uniform_grid_indices,
                           fft_grid_pullback_perm, find_irreducible_bz_points,
                           kgrid_shift_map, q_negation_index,
                           recover_symmorphic_density_point_group,
                           spinor_rotation_for_sym_row, tau_phase_row,
                           unfold_psi)
from symmetry_maps.maps import _I_SIGMA_Y


# ---------------------------------------------------------------------------
# common_uniform_grid_indices — exact native-grid intersection
# ---------------------------------------------------------------------------

def test_q_negation_index_is_the_c_order_involution():
    grid = (3, 4, 2)
    neg = q_negation_index(grid)
    coords = np.stack(np.unravel_index(np.arange(np.prod(grid)), grid), axis=1)
    expected = np.ravel_multi_index(((-coords) % np.asarray(grid)).T, grid)
    np.testing.assert_array_equal(neg, expected)
    np.testing.assert_array_equal(neg[neg], np.arange(neg.size))


def test_nonnested_8x8_to_12x12_has_the_exact_4x4_intersection():
    """RED TWIN: 8→12 shares 4 points/axis, not all 8 and not a prefix."""
    coarse_grid = np.array((8, 8, 1), dtype=np.int64)
    fine_grid = np.array((12, 12, 1), dtype=np.int64)
    coarse, fine = common_uniform_grid_indices(coarse_grid, fine_grid)

    assert coarse.dtype == np.int32 and fine.dtype == np.int32
    assert coarse.shape == fine.shape == (16,)
    coarse_xyz = np.stack(np.unravel_index(coarse, coarse_grid), axis=1)
    fine_xyz = np.stack(np.unravel_index(fine, fine_grid), axis=1)
    assert np.all(coarse_xyz[:, :2] % 2 == 0)
    assert np.all(fine_xyz[:, :2] % 3 == 0)
    np.testing.assert_allclose(
        coarse_xyz / coarse_grid[None, :],
        fine_xyz / fine_grid[None, :], rtol=0.0, atol=0.0)

    # A min(n_coarse,n_fine) prefix/trim is the defect's tempting bypass.
    # It compares different physical k points immediately after Gamma.
    prefix_n = min(int(np.prod(coarse_grid)), int(np.prod(fine_grid)))
    prefix_coarse = np.stack(
        np.unravel_index(np.arange(prefix_n), coarse_grid), axis=1)
    prefix_fine = np.stack(
        np.unravel_index(np.arange(prefix_n), fine_grid), axis=1)
    assert not np.array_equal(
        prefix_coarse / coarse_grid[None, :],
        prefix_fine / fine_grid[None, :])


def test_common_grid_map_keeps_every_point_when_grids_nest_or_match():
    for coarse_grid, fine_grid in [((3, 2, 1), (6, 4, 1)),
                                   ((5, 4, 2), (5, 4, 2))]:
        coarse, fine = common_uniform_grid_indices(coarse_grid, fine_grid)
        assert coarse.size == int(np.prod(coarse_grid))
        np.testing.assert_array_equal(coarse, np.arange(coarse.size))
        coarse_xyz = np.stack(
            np.unravel_index(coarse, coarse_grid), axis=1)
        fine_xyz = np.stack(np.unravel_index(fine, fine_grid), axis=1)
        np.testing.assert_allclose(
            coarse_xyz / np.asarray(coarse_grid)[None, :],
            fine_xyz / np.asarray(fine_grid)[None, :], rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# kgrid_shift_map — the C-order fold and its umklapp G
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_off", [(0, 0, 0), (1, 0, 0), (0, -1, 0),
                                   (2, 3, 1), (-3, 5, -2)])
def test_the_fold_equals_a_roll_and_the_wrap_counts_it(q_off):
    """``arr[kpq_index] == roll(arr, -q)`` and ``G_umk`` is the wrap count.

    The function's own docstring states the roll identity and says it is
    "verified by the identity ``kpq_index == roll`` in the finite-q gate" —
    a gate, not a collected test.  Here it is collected, on five offsets
    including negative ones (where ``//`` is floor division and a C-style
    truncation would give the wrong G) and one that wraps more than a full
    period on an axis.
    """
    nk = (3, 4, 2)
    idx, g = kgrid_shift_map(*nk, q_off)
    arr = np.arange(int(np.prod(nk))).reshape(nk)
    rolled = np.roll(arr, shift=tuple(-int(q) for q in q_off), axis=(0, 1, 2))
    np.testing.assert_array_equal(arr.reshape(-1)[idx], rolled.reshape(-1))
    # G is the wrap count: k_int + q == (k+q folded) + G * nk, per axis.
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in nk), indexing="ij")
    src = np.stack([ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)], axis=1)
    dst = np.stack(np.unravel_index(idx, nk), axis=1)
    np.testing.assert_array_equal(src + np.asarray(q_off)[None, :],
                                  dst + g * np.asarray(nk)[None, :])


def test_the_fold_identity_can_fail():
    """RED TWIN.  The roll identity is a real constraint, not a shape check.

    Compare against the roll for a DIFFERENT offset and it must disagree —
    otherwise the cell above would pass on any permutation whatever.
    """
    nk = (3, 4, 2)
    idx, _ = kgrid_shift_map(*nk, (1, 0, 0))
    arr = np.arange(int(np.prod(nk))).reshape(nk)
    wrong = np.roll(arr, shift=(0, -1, 0), axis=(0, 1, 2)).reshape(-1)
    assert not np.array_equal(arr.reshape(-1)[idx], wrong)


# ---------------------------------------------------------------------------
# find_irreducible_bz_points — both branches, and the anchored one's policy
# ---------------------------------------------------------------------------

def test_the_anchored_branch_reproduces_the_shipping_op_policy():
    """The anchored branch shadows ``find_symmetry_ops_simple`` bit-for-bit.

    Two functions in this module encode the same op-selection policy
    independently — outer loop over ``ikbar`` WITHOUT a break, so the
    HIGHEST matching irreducible k wins and then the lowest sym index for
    that k — and ``find_irreducible_bz_points``' comment at :118-120 says
    outright that it exists to preserve bit-equality with the other one.
    Nothing asserted that they still agree.

    Here they do, on the deck where it matters: ``cohsex_debug``'s tables
    have a star whose first row is a pure time reversal, which is exactly
    the row a different tie-break would move.  Register-don't-touch
    (survey §8.1): if this cell ever fails, the 15.9 eV fork moved and the
    correct response is to find out why, not to update the expectation.
    """
    if not deck_available("cohsex_debug"):
        pytest.skip("no cohsex_debug WFN in this checkout")
    pytest.importorskip("h5py", reason="h5py is not importable")
    wfn = read_deck("cohsex_debug")
    sym = SymMaps(wfn)
    kg = np.asarray(wfn.kgrid, dtype=np.int64)
    full = np.asarray(sym.kvecs_asints, dtype=np.int64)
    shift = np.asarray(wfn.shift, dtype=float) / kg.astype(float)
    irr_int = np.mod(np.rint(
        (np.mod(np.asarray(wfn.kpoints, float) - shift[None, :], 1.0))
        * kg[None, :]).astype(np.int64), kg[None, :])
    irr, sidx, _ = find_irreducible_bz_points(
        full, np.asarray(sym.sym_mats_k, dtype=np.int64),
        irr_kgrid_int=irr_int)
    np.testing.assert_array_equal(irr, np.asarray(sym.irr_idx_k))
    np.testing.assert_array_equal(sidx, np.asarray(sym.sym_idx_k))


def test_the_derive_branch_returns_lex_min_representatives():
    """The ``irr_kgrid_int=None`` branch, and WHY it cannot carry I5/I6.

    It picks lex-smallest orbit representatives.  A lex-min representative
    is reached from itself by the identity, so it is always SPATIAL — which
    is the mechanism behind ``test_a_derived_grid_cannot_carry_these_tests``
    in the star-contract file.  Asserted here at the source rather than
    inferred there.
    """
    grid = np.stack(np.meshgrid(np.arange(4), np.arange(4), np.arange(1),
                                indexing="ij"), -1).reshape(-1, 3)
    syms = np.stack([np.eye(3, dtype=int), -np.eye(3, dtype=int)])
    irr, sidx, irr_out = find_irreducible_bz_points(grid, syms)
    # Every returned representative is itself a row of the full set, mapped
    # by sym 0 (the identity).
    for j, rep in enumerate(irr_out):
        row = int(np.where((grid == rep[None, :]).all(axis=1))[0][0])
        assert int(irr[row]) == j and int(sidx[row]) == 0, (
            f"representative {rep.tolist()} is not reached by the identity")
    # And it is the lex-min of its orbit.
    key = (irr_out[:, 0] * 4 + irr_out[:, 1]) * 1 + irr_out[:, 2]
    for j in range(irr_out.shape[0]):
        members = grid[irr == j]
        mk = (members[:, 0] * 4 + members[:, 1]) * 1 + members[:, 2]
        assert int(key[j]) == int(mk.min())


def test_a_full_point_with_no_preimage_refuses():
    """RED TWIN for the anchored branch: a k the IBZ list cannot reach.

    The refusal names the offending row and its coordinates, which is the
    part a person acts on — an "unfolding failed" would not say which k.
    """
    full = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
    syms = np.stack([np.eye(3, dtype=int)])
    with pytest.raises(RuntimeError, match="no preimage"):
        find_irreducible_bz_points(full, syms,
                                   irr_kgrid_int=np.array([[0, 0, 0]]))


# ---------------------------------------------------------------------------
# G8 — the ntran <= 1 trivial branch against the general branch
# ---------------------------------------------------------------------------

class _TrivialDeck:
    """A symmetry-free (``nosym``) deck: ntran = 1, identity only."""

    def __init__(self, kgrid=(2, 2, 1)):
        kg = np.asarray(kgrid, dtype=np.int32)
        ax = [np.arange(n) / n for n in kg]
        self.kpoints = np.stack(np.meshgrid(*ax, indexing="ij"),
                                -1).reshape(-1, 3)
        self.kgrid = kg
        self.shift = np.zeros(3)
        self.nkpts = int(self.kpoints.shape[0])
        self.ntran = 1
        self.sym_matrices = np.eye(3, dtype=np.int32)[None]
        self.translations = np.zeros((1, 3))
        self.avec = np.eye(3)
        self.atom_types = np.array([1])
        self.atom_crys = np.zeros((1, 3))
        self.trs_holds = True


def test_the_trivial_branch_augments_trs_exactly_like_the_general_one():
    """G8.  ``len(sym_mats_k) == 2 * ntran`` in BOTH branches, with content.

    §Q: a trivial branch that left ``sym_mats_k`` at length 1 made
    ``n_sym_spatial = 1 // 2 = 0``, so ``unfold_psi`` classified the
    IDENTITY row as time-reversed and returned ``iσ_y · conj(ψ)`` on an
    un-negated G list — i.e. ψ(r) → ψ*(−r) for every k of every ``nosym``
    WFN.  Norms, ⟨ψ_m|ψ_n⟩, ⟨T⟩ and (because ρ inverts too) V_H all survive
    that; V_NL collapses and V_loc shifts by O(100 eV).

    The existing shape test (``tests/test_symmetry_unfold.py:706``) checks
    the LENGTH.  This one checks the length AND that the second half is
    ``−S`` AND that the derived ``n_sym_spatial`` classifies row 0 as
    spatial — which is the assertion the bug would have failed.
    """
    triv = SymMaps(_TrivialDeck())
    smk = np.asarray(triv.sym_mats_k)
    assert smk.shape == (2, 3, 3)
    np.testing.assert_array_equal(smk[1], -smk[0])
    nss = smk.shape[0] // 2
    assert nss == 1
    assert not (int(triv.sym_idx_k[0]) >= nss), (
        "the identity row must classify as SPATIAL; classifying it as "
        "time-reversed is scorecard §Q")
    # The general branch, same statement.
    if deck_available("gnppm_debug"):
        pytest.importorskip("h5py", reason="h5py is not importable")
        gen = SymMaps(read_deck("gnppm_debug"))
        gsm = np.asarray(gen.sym_mats_k)
        assert gsm.shape[0] == 2 * int(gen.sym_matrices.shape[0])
        np.testing.assert_array_equal(gsm[gsm.shape[0] // 2:],
                                      -gsm[:gsm.shape[0] // 2])


def test_the_trivial_branch_publishes_every_attribute_the_general_one_does():
    """Same surface from both branches, or a consumer breaks on ``nosym``.

    The two branches are 90 lines apart and share no code; the attribute
    list is the contract between them.  Compared as a SET so a new
    attribute on one side is a failure that names itself.
    """
    triv = SymMaps(_TrivialDeck())
    if not deck_available("gnppm_debug"):
        pytest.skip("no gnppm_debug WFN in this checkout")
    pytest.importorskip("h5py", reason="h5py is not importable")
    gen = SymMaps(read_deck("gnppm_debug"))
    missing = {k for k in vars(gen) if not k.startswith("__")} - set(vars(triv))
    assert missing == set(), (
        f"the ntran<=1 branch does not publish {sorted(missing)}; a caller "
        f"that reads them works on a symmetric deck and breaks on nosym")


# ---------------------------------------------------------------------------
# unfold_psi / spinor_rotation_for_sym_row / tau_phase_row
# ---------------------------------------------------------------------------

def _su2(rng):
    """A genuine SU(2) element — unitary AND det = +1.

    ``np.linalg.qr`` gives U(2), whose det is an arbitrary phase, and the
    intertwining identity below is an SU(2) statement: with det ≠ 1 it
    fails by that phase and the cell would be measuring the draw.
    """
    a = rng.standard_normal(4)
    a /= np.linalg.norm(a)
    w, x, y, z = a
    return np.array([[w + 1j * z, y + 1j * x],
                     [-y + 1j * x, w - 1j * z]], dtype=np.complex128)


def test_the_trs_spinor_factor_squares_to_minus_one():
    """Θ² = −1 on the spinor factor, and ``_I_SIGMA_Y`` is the source.

    Θ = iσ_y K is ANTIunitary, so squaring it is ``M conj(M)`` and not
    ``M @ M``: the conjugation sits BETWEEN the two applications.  Writing
    it the matrix way would give a different, meaningless number that
    happens to be easy to assert, which is why the composition is spelled
    out.

    Two statements, and only the first is Kramers:

    * for the PURE time reversal (spatial op = identity) the augmented
      factor squares to −1 exactly;
    * for a general SU(2) spatial op the augmented row squares to −U², via
      the intertwining identity ``iσ_y conj(U) = U iσ_y``.  That identity
      is what makes ``iσ_y · conj(U_spinor[s])`` the right factor at all,
      and it is checked directly rather than through its consequence.
    """
    rng = np.random.default_rng(2)
    ident = np.eye(2, dtype=np.complex128)[None]
    theta = spinor_rotation_for_sym_row(ident, 1, 1)
    np.testing.assert_allclose(theta, _I_SIGMA_Y, atol=1e-15)
    v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    np.testing.assert_allclose(theta @ np.conj(theta @ np.conj(v)), -v,
                               atol=1e-14)

    u = _su2(rng)
    np.testing.assert_allclose(np.linalg.det(u), 1.0, atol=1e-14)
    theta_s = spinor_rotation_for_sym_row(u[None], 1, 1)
    np.testing.assert_allclose(theta_s, _I_SIGMA_Y @ np.conj(u), atol=1e-14)
    np.testing.assert_allclose(_I_SIGMA_Y @ np.conj(u), u @ _I_SIGMA_Y,
                               atol=1e-14)
    np.testing.assert_allclose(theta_s @ np.conj(theta_s @ np.conj(v)),
                               -(u @ u) @ v, atol=1e-13)


def test_the_su2_intertwining_identity_can_fail():
    """RED TWIN.  ``iσ_y conj(U) = U iσ_y`` is an SU(2) statement.

    A merely UNITARY U (det an arbitrary phase) breaks it by that phase.
    Without this cell the one above would pass on any 2x2 whatever if the
    augmentation quietly stopped conjugating.
    """
    rng = np.random.default_rng(5)
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    u, _ = np.linalg.qr(a)
    assert abs(np.linalg.det(u) - 1.0) > 1e-3, (
        "PRECONDITION: this draw must NOT already be in SU(2)")
    assert not np.allclose(_I_SIGMA_Y @ np.conj(u), u @ _I_SIGMA_Y, atol=1e-6)


def test_a_spatial_row_is_not_augmented():
    """The other half: below ``n_tran`` the spinor op is passed through."""
    rng = np.random.default_rng(3)
    U = (rng.standard_normal((3, 2, 2)) + 1j * rng.standard_normal((3, 2, 2)))
    for s in range(3):
        np.testing.assert_array_equal(
            spinor_rotation_for_sym_row(U, s, 3), U[s])
    np.testing.assert_allclose(spinor_rotation_for_sym_row(U, 4, 3),
                               _I_SIGMA_Y @ np.conj(U[1]), atol=1e-15)


def test_the_augmentation_vectorizes_over_a_row_of_sym_indices():
    """Scalar and array ``sym_idx`` must agree elementwise.

    ``unfold_psi`` takes the scalar path per k; ``WfnLoader``'s phdf5 static
    table build takes the array path over the whole full BZ.  Two spellings
    of one rule, and only one of them is exercised by the existing tests.
    """
    rng = np.random.default_rng(4)
    U = (rng.standard_normal((3, 2, 2)) + 1j * rng.standard_normal((3, 2, 2)))
    idx = np.array([0, 1, 2, 3, 4, 5])
    batch = spinor_rotation_for_sym_row(U, idx, 3)
    assert batch.shape == (6, 2, 2)
    for i, s in enumerate(idx):
        np.testing.assert_array_equal(
            batch[i], spinor_rotation_for_sym_row(U, int(s), 3))


def test_tau_phase_row_is_none_at_tau_zero():
    """A symmorphic op has NO phase, and ``None`` is how that is said.

    Returning ``exp(0) = 1`` everywhere would be numerically identical and
    would cost an ``(ngk,)`` complex array plus a multiply at every k of
    every symmorphic deck — which is all three MoS2 decks in the tree.  The
    ``None`` is load-bearing, so it is pinned.
    """
    g = np.array([[0, 0, 0], [1, 0, 0], [0, 2, -1]], dtype=np.int32)
    S = np.eye(3, dtype=np.int32)
    assert tau_phase_row(S, np.zeros(3), g) is None
    assert tau_phase_row(S, np.array([1e-13, 0.0, 0.0]), g) is None
    tau = np.array([np.pi, 0.0, 0.0])
    ph = tau_phase_row(S, tau, g)
    assert ph is not None and ph.shape == (3,)
    np.testing.assert_allclose(np.abs(ph), 1.0, atol=1e-14)
    np.testing.assert_allclose(ph, np.exp(-1j * (g @ tau)), atol=1e-14)
    # A TRS row carries ``-S`` already, so the SAME formula returns the
    # CONJUGATE phase — which is what (★) demands, and there is no separate
    # τ for TRS rows.  Verified end-to-end on the genuinely non-symmorphic
    # si_cohsex_debug deck (tnp = π ⇒ τ_frac = 1/2).
    np.testing.assert_allclose(tau_phase_row(-S, tau, g), np.conj(ph),
                               atol=1e-14)


def test_unfold_psi_refuses_a_sym_table_that_is_not_twice_ntran():
    """S8.  The §Q guard: the ONE thing keeping the two halves of Θ in step.

    ``unfold_psi`` derives ``n_sym_spatial = len(sym_mats_k) // 2``.  A
    table of any other length makes that derivation silently wrong, and the
    consequence is ψ(r) → ψ*(−r) — invisible to every cheap check.  A hard
    raise, not a warning.
    """
    cnk = np.zeros((2, 1, 3), dtype=np.complex128)
    g = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        unfold_psi(cnk, sym_idx=0, g_kbar=g,
                   sym_mats_k=np.eye(3, dtype=np.int32)[None],   # length 1
                   translations=np.zeros((1, 3)),
                   U_spinor_spatial=np.eye(2, dtype=complex)[None])


# ---------------------------------------------------------------------------
# nspinor = 1 — the scalar (non-SOC) unfold, and the defect it replaced
# ---------------------------------------------------------------------------
#
# THE DEFECT, registered 2026-08-08 and fixed 2026-08-09.
# ``spinor_rotation_for_sym_row`` returned a 2x2 UNCONDITIONALLY, while a
# comment above its consumer's einsum claimed "for ns=1 U_eff is the 1x1
# identity and this einsum is a no-op".  It was not: numpy and JAX both
# BROADCAST a size-1 labelled einsum axis rather than raising, so a scalar
# WFN came back 2-component holding ``(U[j,0]+U[j,1])·psi``.  The loud end
# of it was a slab-write ``ValueError: could not broadcast (8,2,1457) into
# (8,1,1457)`` in ``WfnLoader._eager_build``, with nothing in the message
# about spinors.
#
# EVERY FIXTURE IN THIS TREE IS nspinor=2, so the deck suite structurally
# cannot see this path.  These cells are the whole in-tree gate on it and
# are therefore SYNTHETIC BY NECESSITY, not by preference — see the
# fixed-at-unit-level row in ``tests/KNOWN_FAILURES.md`` for the QE
# scalar-deck leg that is still owed.


def _ns1_deck(rng):
    """A 2-op synthetic deck with a scalar psi and every knob non-trivial.

    Non-identity spatial rotation, a genuine SU(2) spinor matrix for it
    (so a leaked spinor factor cannot hide behind an identity), non-zero
    tau (so the phase is live), and a TRS-augmented table of length
    ``2*n_tran`` so all four row kinds are reachable.
    """
    S1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32)
    sym_mats_k = np.stack([np.eye(3, dtype=np.int32), S1,
                           -np.eye(3, dtype=np.int32), -S1])
    U_sp = np.stack([np.eye(2, dtype=np.complex128), _su2(rng)])
    translations = np.array([[0.0, 0.0, 0.0],
                             [np.pi / 3, -np.pi / 5, np.pi / 7]])
    g = np.array([[0, 0, 0], [1, 0, 0], [0, 2, -1], [-1, 1, 3]],
                 dtype=np.int32)
    cnk = (rng.standard_normal((3, 1, 4))
           + 1j * rng.standard_normal((3, 1, 4))).astype(np.complex128)
    return sym_mats_k, U_sp, translations, g, cnk


def _ns1_reference(sym_mats_k, translations, g, cnk, sym_idx):
    """Hand reference for the scalar rule — no spinor factor anywhere.

    Spatial:  psi_full = exp(-i (S·G)·tau) · psi_kbar
    TRS:      psi_full = exp(+i (S·G)·tau) · conj(psi_kbar)

    Written out with ``np.exp`` rather than through ``tau_phase_row`` so
    this is an independent statement and not a restatement of the code.
    """
    n_tran = len(sym_mats_k) // 2
    is_trs = sym_idx >= n_tran
    s_sp = sym_idx - n_tran if is_trs else sym_idx
    S_spatial = np.asarray(sym_mats_k[s_sp], dtype=np.float64)
    tau = translations[s_sp]
    ph = np.exp(-1j * ((S_spatial @ g.T).T @ tau))              # (ngk,)
    if is_trs:
        return np.conj(cnk) * np.conj(ph)[None, None, :]
    return cnk * ph[None, None, :]


def test_the_scalar_spinor_factor_is_the_1x1_identity_on_every_row_kind():
    """ns=1: one factor, it is 1, and it does not look at ``U_spinor``.

    Both row kinds and both call spellings (scalar ``sym_idx`` for
    ``unfold_psi``'s per-k path, array for ``WfnLoader``'s device table).
    The independence from ``U_spinor_spatial`` is asserted rather than
    assumed: those matrices are built from the CARTESIAN rotations and
    exist whatever ``nspinor`` is, so a scalar deck still has a full
    ``(ntran, 2, 2)`` table sitting there ready to be used by mistake.
    """
    rng = np.random.default_rng(11)
    U = (rng.standard_normal((3, 2, 2)) + 1j * rng.standard_normal((3, 2, 2)))
    one = np.array([[1.0 + 0.0j]])
    for s in range(6):                       # 0..2 spatial, 3..5 TRS
        got = spinor_rotation_for_sym_row(U, s, 3, nspinor=1)
        assert got.shape == (1, 1), f"row {s} returned {got.shape}, not (1,1)"
        np.testing.assert_array_equal(got, one)
    batch = spinor_rotation_for_sym_row(U, np.arange(6), 3, nspinor=1)
    assert batch.shape == (6, 1, 1)
    np.testing.assert_array_equal(batch, np.ones((6, 1, 1), dtype=complex))

    # Same answer from a completely different U — the factor is not a
    # function of the spatial spinor table at all.
    other = (rng.standard_normal((3, 2, 2))
             + 1j * rng.standard_normal((3, 2, 2)))
    np.testing.assert_array_equal(
        spinor_rotation_for_sym_row(other, np.arange(6), 3, nspinor=1), batch)

    # And the 2-spinor answer is unchanged by the parameter existing.
    for s in range(6):
        np.testing.assert_array_equal(
            spinor_rotation_for_sym_row(U, s, 3, nspinor=2),
            spinor_rotation_for_sym_row(U, s, 3))


def test_spinless_time_reversal_squares_to_PLUS_one():
    """The ns=1 TRS row is a DIFFERENT operator, not a truncated iso_y.

    ``test_the_trs_spinor_factor_squares_to_minus_one`` above pins
    Theta^2 = -1 for the 2-component rep (Kramers).  For a scalar field
    Theta = K, so Theta^2 = +1 and there is no Kramers pair to protect.
    Squaring is again the ANTIunitary composition ``M conj(M conj(v))``,
    not ``M @ M``.  If someone ever "fixes" ns=1 by slicing the 2x2 down
    to its [0,0] element, this cell is what says that is not the same
    thing: iso_y's [0,0] is 0, not 1.
    """
    rng = np.random.default_rng(12)
    U = np.stack([np.eye(2, dtype=np.complex128), _su2(rng)])
    v = rng.standard_normal(1) + 1j * rng.standard_normal(1)
    for s in (0, 1, 2, 3):
        theta = spinor_rotation_for_sym_row(U, s, 2, nspinor=1)
        np.testing.assert_allclose(
            theta @ np.conj(theta @ np.conj(v)), v, atol=1e-15)
    assert _I_SIGMA_Y[0, 0] == 0.0, (
        "PRECONDITION: iso_y's [0,0] is 0 — truncating the 2x2 would "
        "ANNIHILATE a scalar wavefunction on a TRS row, which is why the "
        "ns=1 factor is built rather than sliced")


@pytest.mark.parametrize("sym_idx", [0, 1, 2, 3])
def test_unfold_psi_transforms_a_scalar_psi_with_no_spinor_factor(sym_idx):
    """The fix, stated as the thing the loader actually needs.

    A scalar psi through ``unfold_psi`` must come back scalar, equal to
    the hand rule (phase, and conjugation on TRS rows) with NOTHING from
    ``U_spinor``.  Row 1 carries both a non-identity SU(2) and a non-zero
    tau; row 3 is that same op time-reversed.
    """
    rng = np.random.default_rng(13)
    sym_mats_k, U_sp, translations, g, cnk = _ns1_deck(rng)
    out = unfold_psi(cnk, sym_idx=sym_idx, g_kbar=g, sym_mats_k=sym_mats_k,
                     translations=translations, U_spinor_spatial=U_sp)
    assert out.shape == cnk.shape == (3, 1, 4), (
        f"scalar psi came back {out.shape}; the spinor axis must survive "
        f"the unfold at width 1")
    np.testing.assert_allclose(
        out, _ns1_reference(sym_mats_k, translations, g, cnk, sym_idx),
        atol=1e-14, rtol=0.0)
    # Norm is preserved per band — a scalar unfold is a phase (and maybe a
    # conjugation), so it cannot move any |c|.
    np.testing.assert_allclose(np.abs(out), np.abs(cnk), atol=1e-14)


@pytest.mark.parametrize("sym_idx", [1, 3])
def test_RED_TWIN_the_unconditional_2x2_silently_broadcasts_over_a_scalar_psi(
        sym_idx):
    """RED TWIN.  Reproduce the defect and show it is silent, then caught.

    Three statements, in the order that makes the bug's shape clear:

    1. numpy does NOT protect you.  The exact einsum the consumer runs,
       handed a 2x2 and a size-1 spinor axis, returns without raising.
    2. What it returns is wrong twice over: the spinor axis has WIDENED
       to 2, and each output component is ``sum_k U[j,k]·psi[n,0,l]`` —
       the two spinor columns summed — which is not the scalar answer by
       a wide margin.  On row 3 (TRS) the summing is done by
       ``iso_y·conj(U)``'s off-diagonals.
    3. Post-fix that state is unreachable through ``unfold_psi`` (it asks
       for the factor at the psi's own ns), so the guard is exercised by
       putting the PRE-FIX helper back with a monkeypatch.  Without this
       third part the guard would be untested code.
    """
    rng = np.random.default_rng(14)
    sym_mats_k, U_sp, translations, g, cnk = _ns1_deck(rng)
    n_tran = len(sym_mats_k) // 2
    good = _ns1_reference(sym_mats_k, translations, g, cnk, sym_idx)

    # --- 1 + 2: the raw contraction, exactly as the consumer spells it.
    U_bad = spinor_rotation_for_sym_row(U_sp, sym_idx, n_tran)   # 2x2
    assert U_bad.shape == (2, 2), (
        "PRECONDITION: the un-told helper must still hand back a 2x2 — "
        "that is the pre-fix behaviour this twin reproduces")
    pre = np.conj(cnk) if sym_idx >= n_tran else cnk
    S_full = np.asarray(sym_mats_k[sym_idx], dtype=np.float64)
    s_sp = sym_idx % n_tran
    pre = pre * np.exp(
        -1j * ((S_full @ g.T).T @ translations[s_sp]))[None, None, :]
    broadcast = np.einsum("jk,nkl->njl", U_bad, pre)     # does NOT raise
    assert broadcast.shape == (3, 2, 4), (
        f"the broadcast produced {broadcast.shape}; the whole point is "
        f"that the spinor axis widens 1 -> 2 without a word")
    np.testing.assert_allclose(
        broadcast,
        np.stack([(U_bad[j, 0] + U_bad[j, 1]) * pre[:, 0, :]
                  for j in range(2)], axis=1), atol=1e-14)
    assert not np.allclose(broadcast[:, :1, :], good, atol=1e-6), (
        "the leading component of the broadcast happens to equal the "
        "scalar answer — pick a deck where it does not, or this twin is "
        "not red")

    # --- 3: through the real entry point, with the pre-fix helper restored.
    import symmetry_maps.maps as _maps

    def _prefix_helper(U_spatial, idx, ntran, *, nspinor=2):
        return spinor_rotation_for_sym_row(U_spatial, idx, ntran)  # always 2x2

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(_maps, "spinor_rotation_for_sym_row", _prefix_helper)
        with pytest.raises(ValueError, match="BROADCAST"):
            _maps.unfold_psi(cnk, sym_idx=sym_idx, g_kbar=g,
                             sym_mats_k=sym_mats_k,
                             translations=translations,
                             U_spinor_spatial=U_sp)
    finally:
        mp.undo()

    # Unpatched, the same call is correct — the monkeypatch was the defect,
    # not the deck.
    np.testing.assert_allclose(
        unfold_psi(cnk, sym_idx=sym_idx, g_kbar=g, sym_mats_k=sym_mats_k,
                   translations=translations, U_spinor_spatial=U_sp),
        good, atol=1e-14)


def test_the_helper_refuses_an_nspinor_it_has_no_representation_for():
    """4-spinor never comes through here — say so rather than guess.

    ``WfnLoader.load(bispinor=True)`` lifts 2 -> 4 components DOWNSTREAM
    of the unfold, so an ``nspinor=4`` arriving here means a caller has
    confused the two stages.  Silently returning the 2x2 would put a
    4-component psi through the broadcast trap this whole section is about.
    """
    U = np.eye(2, dtype=np.complex128)[None]
    for bad in (0, 3, 4):
        with pytest.raises(ValueError, match="nspinor must be 1"):
            spinor_rotation_for_sym_row(U, 0, 1, nspinor=bad)


# ---------------------------------------------------------------------------
# S6 — the centroid orbit-closure refusal, both halves, synthetically
# ---------------------------------------------------------------------------

def _closed_centroids(fft=(8, 8, 1)):
    """An orbit-closed centroid set under {I, σ_y}, built by construction.

    Never a production ``centroids_frac_*.txt``: survey §8.2 makes the
    production sets register-don't-touch and forbids regenerating any of
    them.  A test that needs a NON-closed set drops one point from this.
    """
    fft = np.asarray(fft, dtype=np.int64)
    syms = np.stack([np.eye(3, dtype=np.int64),
                     np.diag([1, -1, 1]).astype(np.int64)])
    rinv = np.rint(np.linalg.inv(syms)).astype(np.int64)
    seeds = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0],
                      [3, 2, 0], [2, 3, 0], [5, 1, 0]], dtype=np.int64)
    imgs = {tuple(((rinv[s] @ r) % fft).tolist())
            for r in seeds for s in range(syms.shape[0])}
    return fft, syms, np.zeros((2, 3)), np.array(sorted(imgs), dtype=np.int32)


def test_an_orbit_closed_set_passes_validation():
    """The positive case: every row is a permutation and no image escapes."""
    fft, syms, tau, cent = _closed_centroids()
    perm, L = centroid_source_map_and_wrap(cent, syms, 2 * np.pi * tau, fft,
                                           validate=True, extend_trs=True)
    n = cent.shape[0]
    assert perm.shape == (4, n) and L.shape == (4, n, 3)
    np.testing.assert_array_equal(perm[2:], perm[:2])
    np.testing.assert_array_equal(L[2:], L[:2])
    for s in range(perm.shape[0]):
        assert np.unique(perm[s]).size == n


def test_a_dropped_centroid_refuses_with_the_fix_named():
    """S6 half one — an image that is not in the table.

    Constructed SYNTHETICALLY: one point removed from the closed set above,
    never by rerunning ``kmeans_cli`` on a production deck (survey §8.2).
    The message must name the offending sym and centroid AND the fix; the
    refusal is already SERVICE_FORM-shaped and this pins that it stays so.
    """
    fft, syms, tau, cent = _closed_centroids()
    victim = None
    for drop in range(cent.shape[0]):
        trial = np.delete(cent, drop, axis=0)
        try:
            centroid_source_map_and_wrap(trial, syms, 2 * np.pi * tau, fft,
                                         validate=True)
        except RuntimeError as exc:
            victim, msg = drop, str(exc)
            break
    assert victim is not None, (
        "PRECONDITION: dropping SOME centroid must break closure, or this "
        "set was not orbit-closed to begin with and the positive case above "
        "is meaningless")
    assert "orbit closure" in msg
    assert "Regenerate centroids with orbit-aware kmeans" in msg
    assert "fall back to identity-only sym" in msg


def test_validation_off_returns_the_broken_table_instead_of_raising():
    """RED TWIN for the refusal: ``validate=False`` is the other branch.

    Register-don't-touch (design decision 14) is about the DEFAULT and the
    tolerance; this cell pins that the default is the checking one, by
    showing what the unchecked one hands back — a table with a −1 in it,
    which is exactly what would feed a ``promise_in_bounds`` gather and
    clip silently.
    """
    fft, syms, tau, cent = _closed_centroids()
    for drop in range(cent.shape[0]):
        trial = np.delete(cent, drop, axis=0)
        perm, _ = centroid_source_map_and_wrap(
            trial, syms, 2 * np.pi * tau, fft, validate=False)
        if (perm < 0).any():
            return
    pytest.fail("no dropped centroid produced an unmapped image; the set "
                "under test is not what this cell thinks it is")


def test_the_rgrid_permutation_covers_the_whole_fft_grid():
    """``fft_grid_pullback_perm`` — the ζ-side twin of the centroid table.

    Every row must be a permutation of the FULL grid (nothing escapes a
    grid that is closed by construction), and the identity row must be the
    identity.  Cheap, and it is the table ``zeta_loader`` gathers with.
    """
    fft = np.array([4, 4, 2], dtype=np.int64)
    syms = np.stack([np.eye(3, dtype=np.int64),
                     np.diag([1, -1, 1]).astype(np.int64)])
    perm = fft_grid_pullback_perm(syms, np.zeros((2, 3)), fft, validate=True)
    n = int(np.prod(fft))
    assert perm.shape == (2, n)
    np.testing.assert_array_equal(perm[0], np.arange(n))
    assert np.unique(perm[1]).size == n


# ---------------------------------------------------------------------------
# G9 — recover_symmorphic_density_point_group
# ---------------------------------------------------------------------------

def test_a_cubic_density_recovers_the_full_48_op_holohedry():
    """G9.  19683 candidate matrices, filtered against ρ on the FFT grid.

    A ρ built from ``cos 2πx + cos 2πy + cos 2πz`` on a simple-cubic cell is
    invariant under the full m-3m point group, so the recovery must return
    all 48 ops.  Timed, because the enumeration is
    ``itertools.product((-1,0,1), repeat=9)`` and a regression that made it
    minutes would be a real one — measured 0.10 s at 12³ on the dev box.
    """
    n = 12
    ax = np.arange(n) / n
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rho = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)
    t0 = time.perf_counter()
    ops = recover_symmorphic_density_point_group(np.eye(3), rho)
    dt = time.perf_counter() - t0
    assert ops.shape == (48, 3, 3), f"got {ops.shape[0]} ops, expected 48"
    assert dt < 5.0, f"the holohedry enumeration took {dt:.2f}s"
    # Every returned op is a genuine integer rotation of the lattice.
    dets = np.rint(np.linalg.det(ops.astype(float))).astype(int)
    assert set(dets.tolist()) == {1, -1}
    keys = {op.tobytes() for op in ops}
    assert np.eye(3, dtype=ops.dtype).tobytes() in keys


def test_a_broken_density_recovers_a_strict_subgroup():
    """RED TWIN.  Break the density and the recovered group must SHRINK.

    Without this, the cell above would pass on a routine that returned the
    holohedry unconditionally — which is exactly the failure a
    19683-candidate filter can have while looking like it filtered.  The
    result must also still be a GROUP containing the identity, or the
    recovery has broken in a different way.
    """
    n = 12
    ax = np.arange(n) / n
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    rho = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)
    broken = rho + 0.4 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    ops = recover_symmorphic_density_point_group(np.eye(3), broken)
    assert 0 < ops.shape[0] < 48, (
        f"a symmetry-broken density must recover a strict subgroup; "
        f"got {ops.shape[0]} of 48")
    assert 48 % ops.shape[0] == 0, (
        f"{ops.shape[0]} does not divide 48, so the result is not a "
        f"subgroup of the holohedry")
    assert np.eye(3, dtype=ops.dtype).tobytes() in {op.tobytes() for op in ops}


# ---------------------------------------------------------------------------
# The Cartesian metric convention (design decision 7 / survey §5.3)
# ---------------------------------------------------------------------------

def test_the_cartesian_metric_convention_is_discriminated_by_bdot():
    """The transpose contract, stated with h5py and no lorrax code at all.

    The prep swarm's ``mos2_5x5`` "defect" was RETRACTED — it was the
    diagnostic's own ``bvec @ v`` vs ``bvec.T @ v`` metric, and fixing the
    metric took the round trip 1.284e-01 → 1.230e-14.  What survives is the
    FALSIFICATION METHOD, and it is worth more than the defect was: BGW's
    own ``bdot`` is invariant under ``S = mtrx.T`` (so the stored ops really
    are symmetries of the reciprocal metric), ``bdot == blat² (bvec
    bvecᵀ)``, and the Cartesian image ``M S M⁻¹`` is ORTHOGONAL for
    ``M = bvec.T`` and is not for ``M = bvec``.

    MEASURED on ``cohsex_debug/WFNsmall.h5`` on the dev box, 2026-08-07:

        max_s |Sᵀ bdot S − bdot| = 4.053e-09   (2.753e-09 relative to
                                                max|bdot| — the deck's own
                                                header precision, not a
                                                tolerance choice)
        bdot − blat² (bvec bvecᵀ)                     = 0.000e+00
        orthogonality of M S M⁻¹, M = bvec.T          = 2.753e-09
        orthogonality of M S M⁻¹, M = bvec            = 6.667e-01

    Eight orders between the two conventions, with the FALSE case built in.
    (The survey quotes 6.66e-16 / 4.8e-16 / 5.1e-01 for the same three
    numbers on a different deck; the discriminating RATIO is the claim, and
    it reproduces.)

    cohsex_debug specifically: the assertion below refuses to run on a deck
    whose ops cannot tell the two apart, and three of the four in-tree decks
    cannot — Si and both MoS2 3x3 decks give ~1e-17 for BOTH spellings, so
    a suite that picked one of those would be green and vacuous.
    """
    if not deck_available("cohsex_debug"):
        pytest.skip("no cohsex_debug WFN in this checkout")
    pytest.importorskip("h5py", reason="h5py is not importable")
    d = read_deck("cohsex_debug")
    ops = np.asarray(d.sym_matrices[:d.ntran]).transpose(0, 2, 1).astype(float)
    scale = float(np.abs(d.bdot).max())
    inv = max(float(np.abs(S.T @ d.bdot @ S - d.bdot).max()) for S in ops)
    assert inv / scale < 1e-6, (
        f"the stored ops do not leave bdot invariant ({inv / scale:.3e} "
        f"relative); the header's symmetry list and its metric disagree")
    assert float(np.abs(d.bdot - d.blat ** 2
                        * (d.bvec @ d.bvec.T)).max()) < 1e-12 * scale

    def worst(M):
        Mi = np.linalg.inv(M)
        return max(float(np.abs((M @ S @ Mi) @ (M @ S @ Mi).T
                                - np.eye(3)).max()) for S in ops)

    right, wrong = worst(d.bvec.T), worst(d.bvec)
    assert wrong > 1e-2, (
        f"PRECONDITION: this deck must DISCRIMINATE the two conventions or "
        f"the cell proves nothing; the wrong one measured {wrong:.3e}")
    assert right < 1e-6, f"M = bvec.T must be the orthogonal one: {right:.3e}"
    assert wrong > 1e5 * max(right, 1e-18), (
        f"the two conventions must be orders apart: {right:.3e} vs "
        f"{wrong:.3e}")


def test_three_of_the_four_decks_cannot_tell_the_conventions_apart():
    """RED TWIN for the deck choice above — the reason it is not a loop.

    On Si and both MoS2 3x3 decks BOTH spellings come back orthogonal to
    1e-17, so the discriminator is vacuous there.  Writing that down is
    what stops a later "why not parametrize over all four?" from turning a
    real check into a green no-op.
    """
    pytest.importorskip("h5py", reason="h5py is not importable")
    vacuous = []
    for deck in ("si_cohsex_debug", "gnppm_debug", "bispinor_debug"):
        if not deck_available(deck):
            continue
        d = read_deck(deck)
        ops = np.asarray(d.sym_matrices[:d.ntran]).transpose(
            0, 2, 1).astype(float)

        def worst(M):
            Mi = np.linalg.inv(M)
            return max(float(np.abs((M @ S @ Mi) @ (M @ S @ Mi).T
                                    - np.eye(3)).max()) for S in ops)

        if worst(d.bvec) < 1e-6:
            vacuous.append(deck)
    if not vacuous:
        pytest.skip("no non-cohsex deck in this checkout")
    assert set(vacuous) <= {"si_cohsex_debug", "gnppm_debug",
                            "bispinor_debug"}
    assert vacuous, (
        "at least one deck must be vacuous for this cell to be the warning "
        "it is meant to be")


# ---------------------------------------------------------------------------
# G7 — validate_kgrid_unfolding (design decision 3: KEEP and TEST)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", ["si_cohsex_debug", "cohsex_debug",
                                  "gnppm_debug", "bispinor_debug"])
def test_the_kgrid_unfolding_validator_passes_on_every_in_tree_deck(deck):
    """G7 positive.  Zero consumers, zero tests — so it gets both here.

    ``S·k_irr + G_umklapp`` must reproduce every full-BZ k, on all four
    decks including the genuinely non-symmorphic Si one.  SERVICE_FORM says
    a test that cannot fail dies; the next cell is the one that can.
    """
    if not deck_available(deck):
        pytest.skip(f"no {deck} WFN in this checkout")
    pytest.importorskip("h5py", reason="h5py is not importable")
    wfn = read_deck(deck)
    sym = SymMaps(wfn)
    assert sym.validate_kgrid_unfolding(wfn) == []


def test_the_kgrid_unfolding_validator_can_fail():
    """G7 constructible-FALSE, BOTH of the validator's two failure surfaces.

    The fixture on disk is never touched (``cohsex_debug/README.md:44-52``
    — ``tests/regression/*`` is read-only on purpose and
    ``harness.protect_fixtures()`` re-applies it every session); the
    corruption happens on the in-memory tables the validator reads.

    (a) A full-BZ k the uniform grid cannot find comes back in the RETURNED
        list, naming the point.
    (b) A wrong ``sym_idx_k`` row does NOT come back in the list — it
        reaches ``get_umklapp_vector``, which RAISES with ``k_full``,
        ``S*kbar`` and ``kg0`` in the message.  Two different surfaces with
        two different fixes; a cell that only knew about the list would
        report "the validator missed it" for a case the code catches
        harder.  Both are pinned so neither can quietly become the other.
    """
    if not deck_available("gnppm_debug"):
        pytest.skip("no gnppm_debug WFN in this checkout")
    pytest.importorskip("h5py", reason="h5py is not importable")
    wfn = read_deck("gnppm_debug")

    sym = SymMaps(wfn)
    assert sym.validate_kgrid_unfolding(wfn) == []
    sym.unfolded_kpts = np.asarray(sym.unfolded_kpts).copy()
    sym.unfolded_kpts[3] += 0.123
    failures = sym.validate_kgrid_unfolding(wfn)
    assert failures, "a k-point the table no longer carries must be caught"
    assert any("missing from SymMaps.unfolded_kpts" in f for f in failures), (
        f"the failure must say WHICH point went missing; got {failures[:3]}")

    sym2 = SymMaps(wfn)
    victim = int(np.where(np.asarray(sym2.sym_idx_k) != 0)[0][0])
    sym2.sym_idx_k = np.asarray(sym2.sym_idx_k).copy()
    sym2.sym_idx_k[victim] = 0
    with pytest.raises(ValueError, match="Failed to determine symmetry "
                                         "umklapp"):
        sym2.validate_kgrid_unfolding(wfn)


def test_the_validator_reports_a_grid_size_mismatch_by_name():
    """The other refusal branch: the uniform grid and the table disagree.

    A different message with a different fix, so it is checked separately —
    "validation failed" collapsing the two is what makes a diagnostic
    useless at 2am.
    """
    triv = SymMaps(_TrivialDeck(kgrid=(2, 2, 1)))
    deck = _TrivialDeck(kgrid=(2, 2, 1))
    deck.kgrid = np.asarray([3, 3, 1], dtype=np.int32)
    out = triv.validate_kgrid_unfolding(deck)
    assert len(out) == 1 and "full-grid size mismatch" in out[0]
