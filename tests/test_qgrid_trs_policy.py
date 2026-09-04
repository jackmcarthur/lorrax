"""The q-axis time-reversal policy — explicit, never inferred here.

WHAT IS BEING PINNED.  ``symmetry_maps.qgrid_trs`` owns every decision the
IBZ→full-BZ q unfold takes about time reversal, and
``gw.qgrid_symmetry.qgrid_trs_policy_for`` is the one door that supplies
it with the verdict (``SymMaps.trs_allowed``; automatically checked for a
2c DFT reference) plus rank 0 and the
once-per-run announcement.  Bare ``V_q``, RPA ``W_q`` and ladder ``W_q``
all consume the same object.

THE THREE RED TWINS the register asked for are the three classes a q-grid
policy can meet, and each is a separate cell below:

* **magnetic (TRS-broken scalar)** — a ferromagnet whose q and −q are
  independent irreducible parents.  This is the CrI3 production failure
  (Perlmutter JID 57271494): the old unconditional composition refused
  *after* a 685.96-GB ζ fit, and where the parents had coincided it would
  instead have silently overwritten one independently solved row with the
  conjugate of the other.
* **nonmagnetic scalar** — the historical arm, whose behaviour must be
  bit-unchanged.
* **spinor / Kramers** — an SOC deck, where the antiunitary carries
  ``iσ_y`` at ψ level.  The point of this cell is that the q axis is
  *not* where that lives: the policy sees only ``sym_idx`` rows, so a
  magnetic SOC deck takes the TRS-broken arm on exactly the same
  evidence as a magnetic scalar one, and a Kramers-degenerate
  nonmagnetic SOC deck takes the composed arm.

Plus the DISCRIMINATOR cell: at a self-negative q the q↔−q reciprocity
gate degenerates to "``V_q`` is real" and reads ~1e-17 while the
point-group covariance the unfold assumes is violated by 1e-2.  Both
numbers are produced from ONE fixture here, because the register row this
closes is precisely that the two were confused for each other.
"""

from __future__ import annotations

import ast
import os
import pathlib

import numpy as np
import pytest

from symmetry_maps import (                                    # noqa: E402
    build_qgrid_trs_policy,
    little_group_covariance_residual,
    self_negative_q_mask,
    trs_pair_coherent_unfold_sym_idx,
    trs_project_self_negative_q_rows,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"


# ---------------------------------------------------------------------------
# 1. The row map — nonmagnetic scalar arm (moved from
#    tests/test_screening_diagrams_config.py §8, unchanged in behaviour)
# ---------------------------------------------------------------------------

def test_unfold_uses_one_spatial_gauge_for_each_q_pair():
    """Representatives stay fixed; every partner is the same row plus TRS.

    Pair 1/3 contains an irreducible representative.  Pair 4/12 belongs to
    the same four-fold rotation orbit, whose representative is row 1, so
    neither member is solved; its already-spatial member 12 is the
    deterministic source.  The four TRIM rows on this even grid stay fixed.
    """
    grid = (4, 4, 1)
    # Orbit 0 is {(0,+1), (0,-1), (+1,0), (-1,0)}.  The remaining
    # non-TRIM pairs each have their lower flat index as representative.
    irr = np.asarray(
        [5, 0, 6, 0, 0, 1, 2, 3, 7, 4, 8, 4, 0, 3, 2, 1], dtype=np.int32)
    sym = np.asarray(
        [0, 0, 0, 2, 5, 0, 1, 2, 0, 1, 0, 4, 1, 5, 4, 3], dtype=np.int32)
    reps = np.asarray([0, 1, 2, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    got = trs_pair_coherent_unfold_sym_idx(
        irr, sym, kgrid=grid, q_irr_full_idx=reps, n_sym_spatial=3)

    # Solved representatives and every self-negative q are value-untouched.
    assert np.array_equal(got[reps], sym[reps])
    assert np.array_equal(got[[0, 2, 8, 10]], sym[[0, 2, 8, 10]])
    # q=1 is the representative, so q=3 becomes TRS composed with row 0.
    assert got[1] == 0
    assert got[3] == 3
    # Neither q=4 nor q=12 is the representative; preserve spatial row 1 at 12.
    assert got[12] == 1
    assert got[4] == 4
    # The helper never mutates the generic symmetry service's table.
    assert np.array_equal(
        sym, [0, 0, 0, 2, 5, 0, 1, 2, 0, 1, 0, 4, 1, 5, 4, 3])


def test_unfold_keeps_a_fully_solved_q_table_unchanged():
    """A full-BZ solve has no missing partner to reconstruct through TRS."""
    selected = np.asarray([0, 0, 0])
    got = trs_pair_coherent_unfold_sym_idx(
        np.asarray([0, 1, 2]), selected,
        kgrid=(3, 1, 1), q_irr_full_idx=np.asarray([0, 1, 2]),
        n_sym_spatial=1)
    assert np.array_equal(got, selected)


def test_unfold_refuses_distinct_parents_in_a_partial_wedge():
    """A partial wedge cannot fabricate a q/-q relation across parents."""
    with pytest.raises(ValueError, match="do not fabricate reciprocity"):
        trs_pair_coherent_unfold_sym_idx(
            np.asarray([0, 1, 2, 2, 4]), np.asarray([0, 0, 0, 0, 0]),
            kgrid=(5, 1, 1), q_irr_full_idx=np.asarray([0, 1, 2, 4]),
            n_sym_spatial=1)


def test_pair_coherent_rows_restore_reciprocity_without_spatial_covariance():
    """The row policy must not assume a finite fitted parent is PG-covariant."""
    grid = (4, 4, 1)
    irr = np.asarray(
        [5, 0, 6, 0, 0, 1, 2, 3, 7, 4, 8, 4, 0, 3, 2, 1], dtype=np.int32)
    selected = np.asarray(
        [0, 0, 0, 2, 5, 0, 1, 2, 0, 1, 0, 4, 1, 5, 4, 3], dtype=np.int32)
    reps = np.asarray([0, 1, 2, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    coherent = trs_pair_coherent_unfold_sym_idx(
        irr, selected, kgrid=grid, q_irr_full_idx=reps, n_sym_spatial=3)

    # Rows 0..2 are spatial, 3..5 their TRS-composed copies.  The parent
    # deliberately does not commute with row 2's centroid permutation.
    perms = np.asarray([[0, 1], [0, 1], [1, 0],
                        [0, 1], [0, 1], [1, 0]], dtype=np.int32)
    parent = np.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])

    def image(row):
        p = perms[row]
        out = parent[np.ix_(p, p)]
        return np.conj(out) if row >= 3 else out

    # q rows 1 and 3 are negatives with one wedge parent.  The generic
    # first-match choices (0,2) ask parent to be spatially covariant and fail.
    assert not np.allclose(image(selected[1]), np.conj(image(selected[3])))
    # The coherent choices (0,3) use one spatial realization plus TRS.
    assert np.array_equal(image(coherent[1]), np.conj(image(coherent[3])))


def test_fixed_q_projector_changes_only_self_negative_rows():
    import jax.numpy as jnp

    grid = (4, 1, 1)
    rows = jnp.asarray([[[1 + 2j]], [[3 + 4j]], [[5 + 6j]], [[7 + 8j]]])
    got = np.asarray(trs_project_self_negative_q_rows(
        rows, np.arange(4), kgrid=grid))
    # q=0 and q=2 are their own negatives; q=1 and q=3 are a pair and are
    # handled by the row-map policy, not by this fixed-point projector.
    assert np.all(got[[0, 2]].imag == 0.0)
    assert np.array_equal(got[[0, 2]].real, np.asarray(rows)[[0, 2]].real)
    assert np.array_equal(got[[1, 3]], np.asarray(rows)[[1, 3]])


def test_self_negative_mask_is_every_trim_not_just_gamma():
    """The gate's blindness is at every ``2q == 0``, not only at q=0."""
    mask = self_negative_q_mask(np.arange(4 * 4 * 4), kgrid=(4, 4, 4))
    assert int(np.count_nonzero(mask)) == 8       # the eight TRIM
    assert bool(mask[0])                          # Gamma
    odd = self_negative_q_mask(np.arange(27), kgrid=(3, 3, 3))
    assert int(np.count_nonzero(odd)) == 1        # Gamma alone on odd meshes


# ---------------------------------------------------------------------------
# 2. RED TWIN — magnetic (TRS-broken scalar)
# ---------------------------------------------------------------------------

def _cri3_shaped_tables():
    """A wedge whose q and -q are INDEPENDENT irreducible parents.

    The shape of the ferromagnetic CrI3 charge-channel table: the spatial
    point group still reduces the q mesh, but no operation maps q to -q,
    so every +q/-q pair is two separately solved rows with two separately
    named parents.  ``(1->1, 8->8)`` is the pair the production log names.
    """
    grid = (9, 1, 1)
    # Only the identity is a symmetry here, so every q is its own parent.
    irr = np.arange(9, dtype=np.int32)
    sym = np.zeros(9, dtype=np.int32)
    reps = np.arange(9, dtype=np.int32)
    return grid, irr, sym, reps


def test_magnetic_deck_keeps_every_independently_solved_row():
    """TRS-BROKEN: no composition, no projector, no refusal — it just runs."""
    grid, irr, sym, reps = _cri3_shaped_tables()
    policy = build_qgrid_trs_policy(
        trs_measured=False, irr_idx_q=irr, sym_idx_q=sym,
        q_irr_full_idx=reps, kgrid=grid, n_sym_spatial=1,
        context="CrI3 FM V_q")
    assert policy.trs_measured is False
    assert np.array_equal(policy.unfold_sym_idx, sym)
    assert policy.n_pair_rewired == 0
    assert "TIME REVERSAL IS BROKEN" in policy.announcement()


def test_magnetic_deck_applies_no_fixed_q_projector():
    """The Theta projector has no warrant on a magnet and does not fire.

    ``(A + conj A)/2`` at a TRIM is the one-element group average of a
    group that is NOT a symmetry here.  Applying it would repair the
    downstream object so a gate passes — and the value it repairs is the
    only independent evidence of what the fit produced at that q.
    """
    import jax.numpy as jnp

    grid = (4, 1, 1)
    rows = jnp.asarray([[[1 + 2j]], [[3 + 4j]], [[5 + 6j]], [[7 + 8j]]])
    policy = build_qgrid_trs_policy(
        trs_measured=False, irr_idx_q=np.arange(4, dtype=np.int32),
        sym_idx_q=np.zeros(4, dtype=np.int32),
        q_irr_full_idx=np.arange(4, dtype=np.int32),
        kgrid=grid, n_sym_spatial=1)
    out, removed = policy.project_fixed_q(rows, np.arange(4))
    assert removed is None
    assert np.array_equal(np.asarray(out), np.asarray(rows))


def test_the_old_unconditional_composition_is_what_killed_cri3():
    """The SAME tables, on the measured-TRS arm, are the production failure.

    This is the negative control for the cell above: it fixes what the
    branch is worth.  Composing q with -q on a table whose parents are
    independent is exactly the refusal Perlmutter JID 57271494 hit after
    its 685.96-GB zeta fit had already closed.
    """
    grid, irr, sym, reps = _cri3_shaped_tables()
    # Widen the wedge so the "every q is its own parent" short-circuit does
    # not fire: two q now share a parent, the rest do not.
    irr = irr.copy()
    irr[8] = 7
    reps = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    with pytest.raises(ValueError, match="do not fabricate reciprocity"):
        build_qgrid_trs_policy(
            trs_measured=True, irr_idx_q=irr, sym_idx_q=sym,
            q_irr_full_idx=reps, kgrid=grid, n_sym_spatial=1)
    # ...and the measured-magnetic arm takes the same tables without a word.
    policy = build_qgrid_trs_policy(
        trs_measured=False, irr_idx_q=irr, sym_idx_q=sym,
        q_irr_full_idx=reps, kgrid=grid, n_sym_spatial=1)
    assert np.array_equal(policy.unfold_sym_idx, sym)


def test_a_theta_row_beside_a_magnetic_verdict_refuses_loudly():
    """Tables and verdict must come from one object, and say so if not.

    A ``SymMaps`` built from a magnetic WFN can never select a Theta row
    (its search set is the spatial half).  A table that names one anyway
    means the verdict and the tables were produced by different objects,
    and the honest outcome is a refusal rather than an unfold that
    conjugates a magnetic wavefunction.
    """
    grid = (4, 1, 1)
    with pytest.raises(ValueError, match="trs_measured_vs_tables"):
        build_qgrid_trs_policy(
            trs_measured=False,
            irr_idx_q=np.asarray([0, 1, 1, 0], dtype=np.int32),
            sym_idx_q=np.asarray([0, 0, 2, 2], dtype=np.int32),
            q_irr_full_idx=np.asarray([0, 1], dtype=np.int32),
            kgrid=grid, n_sym_spatial=2)


def test_trs_measured_has_no_default():
    """A caller that has not consulted the density gets a TypeError.

    The tree's house style for a predicate whose wrong branch is invisible
    (``trs_reference``: 13 explicit call sites, no default).  A permissive
    default here is exactly the defect being closed.
    """
    with pytest.raises(TypeError):
        build_qgrid_trs_policy(                       # type: ignore[call-arg]
            irr_idx_q=np.asarray([0]), sym_idx_q=np.asarray([0]),
            q_irr_full_idx=np.asarray([0]), kgrid=(1, 1, 1),
            n_sym_spatial=1)


# ---------------------------------------------------------------------------
# 3. RED TWIN — nonmagnetic scalar
# ---------------------------------------------------------------------------

def test_nonmagnetic_scalar_deck_composes_and_projects():
    """The historical arm, unchanged, and its projector reports its removal."""
    import jax.numpy as jnp

    grid = (4, 4, 1)
    irr = np.asarray(
        [5, 0, 6, 0, 0, 1, 2, 3, 7, 4, 8, 4, 0, 3, 2, 1], dtype=np.int32)
    sym = np.asarray(
        [0, 0, 0, 2, 5, 0, 1, 2, 0, 1, 0, 4, 1, 5, 4, 3], dtype=np.int32)
    reps = np.asarray([0, 1, 2, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    policy = build_qgrid_trs_policy(
        trs_measured=True, irr_idx_q=irr, sym_idx_q=sym,
        q_irr_full_idx=reps, kgrid=grid, n_sym_spatial=3)
    assert policy.trs_measured is True
    assert policy.n_pair_rewired > 0
    assert np.array_equal(
        policy.unfold_sym_idx,
        trs_pair_coherent_unfold_sym_idx(
            irr, sym, kgrid=grid, q_irr_full_idx=reps, n_sym_spatial=3))

    # The projector fires only at the wedge rows whose q is self-negative,
    # and it hands back the anti-Theta component it deleted rather than
    # swallowing it.  On this 4x4 grid the TRIM are flat q 0, 2, 8, 10, and
    # every one of them is a wedge representative here.
    fixed = self_negative_q_mask(reps, kgrid=grid)
    assert int(np.count_nonzero(fixed)) == 4
    wedge = jnp.asarray(np.arange(1, 10, dtype=np.complex128) * (1 + 1j)
                        ).reshape(9, 1, 1)
    out, removed = policy.project_fixed_q(wedge, reps)
    assert removed is not None and removed > 0.0
    got = np.asarray(out).reshape(-1)
    want = np.asarray(wedge).reshape(-1)
    assert np.all(got[fixed].imag == 0.0)
    assert np.array_equal(got[fixed].real, want[fixed].real)
    assert np.array_equal(got[~fixed], want[~fixed])


# ---------------------------------------------------------------------------
# 4. RED TWIN — spinor / Kramers
# ---------------------------------------------------------------------------

def test_spinor_kramers_nonmagnetic_takes_the_composed_arm():
    """SOC does not change the q-axis policy; the VERDICT does.

    The Kramers structure of the antiunitary (``Theta = i sigma_y K``,
    ``Theta^2 = -1``) lives entirely at psi level, in
    ``symmetry_maps.unfold_psi`` / ``spinor_rotation_for_sym_row``.  The q
    axis sees only ``sym_idx`` rows, so a nonmagnetic SOC deck composes
    exactly as a nonmagnetic scalar one does.  Pinned so a future spinor
    special case here has to argue with a cell.
    """
    grid = (4, 1, 1)
    irr = np.asarray([0, 1, 2, 1], dtype=np.int32)
    sym = np.asarray([0, 0, 0, 1], dtype=np.int32)
    reps = np.asarray([0, 1, 2], dtype=np.int32)
    scalar = build_qgrid_trs_policy(
        trs_measured=True, irr_idx_q=irr, sym_idx_q=sym,
        q_irr_full_idx=reps, kgrid=grid, n_sym_spatial=1,
        context="scalar")
    spinor = build_qgrid_trs_policy(
        trs_measured=True, irr_idx_q=irr, sym_idx_q=sym,
        q_irr_full_idx=reps, kgrid=grid, n_sym_spatial=1,
        context="nspinor=2")
    assert np.array_equal(scalar.unfold_sym_idx, spinor.unfold_sym_idx)
    assert spinor.unfold_sym_idx[3] == 1          # TRS-composed partner


def test_magnetic_soc_deck_runs_on_the_same_evidence_as_a_magnetic_scalar():
    """A ferromagnet WITH spin-orbit is a TRS-broken deck like any other.

    The historical trap is to reason "nspinor=2, therefore Kramers,
    therefore time reversal". The 2c reference check returns BROKEN for a
    magnetic SOC deck; the policy must consume that verdict and not infer
    one from the spinor count. Scalar cases below exercise policy algebra
    with an explicit verdict; they are not automatic reference checks.
    """
    grid, irr, sym, reps = _cri3_shaped_tables()
    policy = build_qgrid_trs_policy(
        trs_measured=False, irr_idx_q=irr, sym_idx_q=sym,
        q_irr_full_idx=reps, kgrid=grid, n_sym_spatial=1,
        context="CrI3 FM SOC")
    assert policy.trs_measured is False
    assert np.array_equal(policy.unfold_sym_idx, sym)


# ---------------------------------------------------------------------------
# 5. THE DISCRIMINATOR — reciprocity is blind where covariance is largest
# ---------------------------------------------------------------------------

def _covariance_fixture(epsilon: float):
    """A 2-centroid Gamma-only wedge whose parent breaks its little group.

    ``S`` swaps the two centroids and fixes q=Gamma, so it IS in the
    little group.  The stored parent is real symmetric plus ``epsilon`` of
    a swap-antisymmetric part: real (so q<->-q reciprocity is exact) but
    not swap-invariant (so the unfold's covariance assumption is violated
    by ``epsilon``).  That is the Na 8x8x8 c464 signature in miniature.
    """
    parent = np.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=np.complex128)
    parent = parent + epsilon * np.asarray([[1.0, 0.0], [0.0, -1.0]])
    return parent.reshape(1, 2, 2)


def test_reciprocity_reads_machine_epsilon_where_covariance_reads_one_percent():
    """The register row, reproduced from one fixture at both gates.

    ``check_q_conjugate_reciprocity`` at a self-negative q asks only
    whether the tile is REAL; the covariance gate asks whether it survives
    its own little group.  A tile can answer 1e-16 to the first and 1e-2
    to the second, and on the Na 8x8x8 SOC c464 deck it does (3.915e-17
    against 1.240e-02 at Gamma).
    """
    import jax.numpy as jnp
    from common import sanity

    V = jnp.asarray(_covariance_fixture(1.0e-2))
    # Gamma-only grid: the whole BZ is one self-negative q.
    assert sanity.check_q_conjugate_reciprocity(
        "fixture", V, (1, 1, 1), rtol=1e-8) is True

    cov = little_group_covariance_residual(
        V,
        q_irr_frac=np.zeros((1, 3)),
        q_irr_full_idx=np.zeros(1, dtype=np.int32),
        sym_mats_k=np.stack([np.eye(3, dtype=np.int64),
                             np.eye(3, dtype=np.int64)]),
        sym_perm=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        L_table=np.zeros((2, 2, 3), dtype=np.int64),
        kgrid=(1, 1, 1), n_sym_spatial=2)
    assert cov["n_ops_sampled"] == 1
    assert cov["max_rel"] > 1.0e-3, cov
    assert cov["worst_sym"] == 1


def test_covariance_reads_zero_on_a_covariant_parent():
    """The negative control, aimed at the real statistic and not a stub."""
    V = _covariance_fixture(0.0)
    cov = little_group_covariance_residual(
        V,
        q_irr_frac=np.zeros((1, 3)),
        q_irr_full_idx=np.zeros(1, dtype=np.int32),
        sym_mats_k=np.stack([np.eye(3, dtype=np.int64),
                             np.eye(3, dtype=np.int64)]),
        sym_perm=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        L_table=np.zeros((2, 2, 3), dtype=np.int64),
        kgrid=(1, 1, 1), n_sym_spatial=2)
    assert cov["max_rel"] < 1.0e-14, cov


def test_covariance_reports_nan_when_the_question_is_unanswerable():
    """No non-identity little-group op is UNANSWERABLE, not a pass.

    Same rule ``boundary_min_gaps`` follows: an ambiguous question returns
    ``nan``, which is neither ``> tol`` nor ``<= tol``, so a caller cannot
    read silence as safety.
    """
    V = _covariance_fixture(1.0e-2)
    cov = little_group_covariance_residual(
        V,
        q_irr_frac=np.zeros((1, 3)),
        q_irr_full_idx=np.zeros(1, dtype=np.int32),
        sym_mats_k=np.eye(3, dtype=np.int64)[None],
        sym_perm=np.asarray([[0, 1]], dtype=np.int32),
        L_table=np.zeros((1, 2, 3), dtype=np.int64),
        kgrid=(1, 1, 1), n_sym_spatial=1)
    assert np.isnan(cov["max_rel"])
    assert cov["n_ops_available"] == 0


def test_the_covariance_reporter_fails_on_the_defect_and_passes_on_zero():
    """A gate that cannot fail is not evidence — construct both cases."""
    from common import sanity

    bad = {"max_rel": 1.24e-2, "max_abs": 1.0, "scale": 1.0,
           "worst_parent": 0, "worst_sym": 7, "n_ops_sampled": 47,
           "n_ops_available": 47, "n_parents": 1}
    good = dict(bad, max_rel=1.6e-7)
    assert sanity.report_parent_covariance("t", bad, print_fn=lambda *a: None) \
        is False
    assert sanity.report_parent_covariance("t", good, print_fn=lambda *a: None) \
        is True


# ---------------------------------------------------------------------------
# 6. The ratchet — one door, and no driver takes a TRS decision of its own
# ---------------------------------------------------------------------------

_POLICY_NAMES = {
    "trs_pair_coherent_unfold_sym_idx",
    "trs_project_self_negative_q_rows",
    "build_qgrid_trs_policy",
}
# ``gw/qgrid_symmetry.py`` is the announcing adapter and is allowed to name
# the constructor.  Nothing under ``src/`` may name the two arms.
_ADAPTER = os.path.join("gw", "qgrid_symmetry.py")


def _src_modules():
    for root, _dirs, files in os.walk(_SRC):
        if "__pycache__" in root:
            continue
        for fn in sorted(files):
            if fn.endswith(".py"):
                path = os.path.join(root, fn)
                yield os.path.relpath(path, _SRC), path


def _names_used(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[-1])
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Name):
            out.add(n.id)
    return out


def test_no_driver_composes_q_with_minus_q_on_its_own():
    """The ratchet.  One door, and the verdict is read exactly once.

    Both halves matter.  Pinning only the two arms would let a driver
    build a policy with a hand-written ``trs_measured=True``; pinning only
    the constructor would let it call the arms directly and be back to an
    unconditional composition.  The site this was written for is the three
    charge-channel unfolds, which each spelled the same two calls.
    """
    offenders, scanned, matched = [], 0, []
    for rel, path in _src_modules():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        scanned += 1
        if not any(n in src for n in _POLICY_NAMES):
            continue
        matched.append(rel)
        if rel == _ADAPTER:
            continue
        names = _names_used(ast.parse(src, filename=path))
        if names & _POLICY_NAMES:
            offenders.append(rel)
    # COVERAGE, ASSERTED.  Without these two lines a ratchet that walked an
    # empty tree, or whose ``_SRC`` had moved, reads EXACTLY like one that
    # examined every module and found nothing -- ``TASTE.md`` rule 18's
    # corollary, and the shape that let ``test_env_registry`` scan 121 read
    # sites and report nothing at 9/9 green.  So the scan must have seen a
    # real tree, and it must have MATCHED the one module that does name the
    # constructor: that is the detector working on production source rather
    # than on a fixture.
    assert scanned > 100, (
        f"the ratchet examined only {scanned} modules under {_SRC} -- it is "
        f"not looking at the tree it claims to gate")
    assert _ADAPTER in matched, (
        f"the ratchet matched {matched} and NOT {_ADAPTER}, which is the one "
        f"module that must name ``build_qgrid_trs_policy``.  Either the "
        f"adapter stopped being the door or the name matching is broken; "
        f"either way a zero-offender result here means nothing.")
    assert not offenders, (
        f"{offenders} take a q-axis time-reversal decision without going "
        f"through ``gw.qgrid_symmetry.qgrid_trs_policy_for``, which is the "
        f"only place the MEASURED verdict (``SymMaps.trs_allowed``) is "
        f"read.  A site that composes q with -q itself is a site that will "
        f"do it on a ferromagnet again.")


def test_the_ratchet_fires_on_the_real_adapter_when_the_exemption_is_lifted():
    """The negative control, AIMED AT PRODUCTION SOURCE.

    ``TASTE.md`` rule 18: a control that only ever sees a synthetic fixture
    tests the function, not the gate.  The 2026-08-06 env-registry failure
    had exactly one negative control and it was pointed at a made-up
    string, so nothing ever watched the real page while ``covered()``
    returned True for every name.

    So this runs the ratchet's own detector -- ``_names_used`` over the
    real ``gw/qgrid_symmetry.py`` -- with the exemption removed, and
    requires it to flag the file.  If that fails, the detector cannot see
    a policy call in a real module and every green run of the cell above
    is vacuous.  It also pins WHICH name is found: the adapter is allowed
    the constructor and must not name the two arms.
    """
    path = os.path.join(_SRC, _ADAPTER)
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    found = _names_used(ast.parse(src, filename=path)) & _POLICY_NAMES
    assert found == {"build_qgrid_trs_policy"}, (
        f"the detector found {sorted(found)} in the adapter; it must find "
        f"exactly the constructor.  Finding NOTHING means the ratchet "
        f"cannot fire on any real module; finding an ARM means the adapter "
        f"has started calling the arms directly, which is the door leaking.")


def test_the_adapter_reads_the_measured_verdict_and_nothing_else():
    """``trs_allowed`` is read at the door, and no default is supplied."""
    path = os.path.join(_SRC, _ADAPTER)
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "sym.trs_allowed" in src, (
        "the adapter must take the verdict off SymMaps, not assume it")
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "qgrid_trs_policy_for":
            names = {a.arg for a in node.args.kwonlyargs}
            assert "trs_measured" not in names, (
                "the adapter must MEASURE the verdict, not accept one from "
                "a caller — an argument here is a way to assume TRS again")
            return
    raise AssertionError("qgrid_trs_policy_for is missing from the adapter")
