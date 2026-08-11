"""Gates for ``common.spectral_closure`` — where a rank cut may land.

THE SHAPE OF THIS FILE.  Every guard here is a discrimination, so every gate
comes in a pair: a TRUE arm whose spectrum has a degenerate block straddling
the cut, which must snap or refuse LOUDLY, and a FALSE arm with a clean gap
at the cut, which must be SILENT.  A guard that fires on both is not a guard,
and the FALSE arms are the half that catches that.

The FALSE arms are not synthetic conveniences.  The one that matters is
:func:`test_the_armF_cut_falls_in_a_gap`, built on the measured
``lam_min_kept/lam_drop_hi >= 1.46`` of the fixed Si 6x6x6 deck
(``tests/known_failures/2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md``):
that arm's Sigma_x k-star identity is satisfied to 0.0000 meV with the
truncation live on all 216 q, so the guard MUST NOT fire there.  If it did,
the guard would be refusing the only ISDF deck in the project that is known
to be exactly star-covariant.
"""
import math

import numpy as np
import pytest

from common import spectral_closure as sc


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def _smooth(n=64, decades=12.0):
    """A featureless power-law spectrum — no gap, knee or plateau anywhere.

    The shape ``rank_criterion`` argues every ISDF/Galerkin overlap spectrum
    has.  Used for the FALSE arms: on it, no cut anywhere is inside a block.
    """
    return np.array([10.0 ** (-decades * i / (n - 1)) for i in range(n)])


def _with_block(n=64, decades=12.0, at=20, size=4, rel=1e-9):
    """``_smooth`` with a degenerate block of ``size`` planted at index ``at``.

    The block's members agree to ``rel`` relative to each other, so a cut
    anywhere strictly inside it slices an eigenspace.
    """
    s = _smooth(n, decades)
    base = s[at]
    for j in range(size):
        s[at + j] = base * (1.0 + rel * j)
    return s


# ---------------------------------------------------------------------------
# 1. The criterion itself — TRUE and FALSE arms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cut_offset", [1, 2, 3])
def test_TRUE_a_cut_inside_a_block_fires_and_snaps_outward(cut_offset):
    """A cut strictly inside a planted block snaps past its bottom."""
    s = _with_block(at=20, size=4, rel=1e-9)
    info = sc.cluster_at_cut(s, 20 + cut_offset)
    assert info["fired"], "a cut inside a 4-fold block must fire"
    assert info["n_keep_snapped"] == 24, (
        f"the block occupies sorted positions 20..23, so any cut inside it "
        f"must snap OUTWARD to 24 (keep-more); got {info['n_keep_snapped']}")
    assert info["n_keep_snapped"] > info["n_keep"], "outward, never inward"
    assert len(info["members"]) == 4


@pytest.mark.parametrize("cut", [1, 5, 20, 24, 40, 63])
def test_FALSE_a_smooth_spectrum_never_fires_anywhere(cut):
    """The FALSE arm.  A featureless spectrum has no block to cut."""
    info = sc.cluster_at_cut(_smooth(), cut)
    assert not info["fired"], (
        f"the guard fired at cut={cut} on a spectrum with a clean 1.5-decade "
        f"ratio between every neighbour — it is finding blocks that are not "
        f"there, and every site would snap to full rank")
    assert info["n_keep_snapped"] == cut


def test_FALSE_a_cut_at_the_block_boundary_is_silent():
    """Cutting BETWEEN whole blocks is the legal case and must stay silent."""
    s = _with_block(at=20, size=4, rel=1e-9)
    for cut in (20, 24):
        info = sc.cluster_at_cut(s, cut)
        assert not info["fired"], (
            f"cut={cut} falls between whole blocks — the legal case — and the "
            f"guard fired on it")


def test_the_cut_the_guard_moves_to_is_itself_clean():
    """The snap must land in a gap, or it has moved the problem, not fixed it."""
    s = _with_block(at=20, size=4, rel=1e-9)
    info = sc.cluster_at_cut(s, 22)
    assert info["gap_rel"] <= info["rtol"], "precondition: the old cut is dirty"
    assert info["gap_rel_snapped"] > info["rtol"], (
        "the guard snapped to a cut that is ITSELF inside a block — the walk "
        "stopped early and the retained span is still not invariant")


def test_a_cut_at_either_end_slices_nothing():
    """The outer boundaries cut nothing, exactly as ``band_degeneracy`` has it."""
    s = _with_block()
    for cut in (0, len(s)):
        info = sc.cluster_at_cut(s, cut)
        assert not info["fired"] and info["gap_rel"] == math.inf


# ---------------------------------------------------------------------------
# 2. The three modes
# ---------------------------------------------------------------------------

def test_strict_refuses_and_names_the_block_and_the_rank_that_works():
    s = _with_block(at=20, size=4, rel=1e-9)
    with pytest.raises(sc.SpectralClusterError) as e:
        sc.resolve_spectral_cut(s, 22, mode="strict", where="gate")
    msg = str(e.value)
    assert "gate" in msg, "the message must locate itself"
    assert "24" in msg, "strict must name the rank that would work"
    assert "block holds 4 values" in msg, "strict must name the block"


def test_snap_repairs_loudly_and_names_the_move():
    s = _with_block(at=20, size=4, rel=1e-9)
    lines = []
    k, info = sc.resolve_spectral_cut(s, 22, mode="snap", where="gate",
                                      log=lines.append)
    assert k == 24 and info["fired"]
    body = "\n".join(lines)
    assert "SNAPPED OUTWARD" in body, "a silent repair is the failure mode"
    assert "22 -> 24" in body


def test_off_is_silent_and_changes_nothing():
    s = _with_block(at=20, size=4, rel=1e-9)
    lines = []
    k, info = sc.resolve_spectral_cut(s, 22, mode="off", where="gate",
                                      log=lines.append)
    assert k == 22 and not info["fired"] and lines == []


def test_the_false_arm_is_silent_in_every_mode():
    """The whole discrimination, in one gate: clean cut, nothing said, no raise."""
    s = _smooth()
    for mode in sc.MODES:
        lines = []
        k, info = sc.resolve_spectral_cut(s, 20, mode=mode, where="gate",
                                          log=lines.append)
        assert k == 20 and not info["fired"], mode
        assert lines == [], f"mode={mode} spoke on a clean cut: {lines}"


def test_a_misspelled_mode_raises_rather_than_meaning_off():
    with pytest.raises(ValueError):
        sc.resolve_mode("snapp")


# ---------------------------------------------------------------------------
# 3. THE armF GATE — the deck this whole saga produced must stay silent
# ---------------------------------------------------------------------------

def test_the_armF_cut_falls_in_a_gap():
    """The measured Si 6x6x6 ``nband=68`` cut must not fire.  MEASURED INPUT.

    ``tests/known_failures/2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md``:
    in the degeneracy-closed arm the zeta truncation is live on all 216 q
    (1104 -> 1095/1098 modes) with ``lam_min_kept/lam_drop_hi`` **as low as
    1.46**, and that arm's Sigma_x k-star spread is EXACTLY 0.0000 meV.  So
    the tightest cut on the only ISDF deck known to be exactly star-covariant
    has a relative gap of ``1 - 1/1.46 = 0.3151``, and the guard must be five
    decades away from firing on it.

    This is the arithmetic half of the armF proof and it runs on CPU.  The
    other half — re-running the armF regeneration path with the guard armed
    and reading a silent log — is a GPU leg and is DEFERRED BY NAME in the
    lane's ledger row.
    """
    worst_ratio = 1.46
    gap_rel = 1.0 - 1.0 / worst_ratio
    assert gap_rel == pytest.approx(0.31507, abs=1e-4)
    assert gap_rel > sc.DEFAULT_RTOL * 1e4, (
        f"the armF cut's relative gap {gap_rel:.4f} is not comfortably above "
        f"rtol {sc.DEFAULT_RTOL:.1e} — the default tolerance would put the "
        f"only known star-covariant ISDF deck at risk of a spurious snap")

    # And end to end, on a spectrum carrying that exact ratio at the cut.
    s = np.concatenate([_smooth(40, 8.0),
                        np.array([1e-9, 1e-9 / worst_ratio, 1e-13])])
    info = sc.cluster_at_cut(s, 41)
    assert not info["fired"], (
        "the guard fired on the armF ratio — it would refuse or re-rank the "
        "deck whose Sigma_x star identity is exactly 0.0000 meV")
    assert info["gap_rel"] == pytest.approx(gap_rel, rel=1e-6)


def test_the_armC_style_exact_degeneracy_does_fire():
    """The TRUE twin of the gate above, so it is a discrimination.

    The broken arm's signature is an EXACT degeneracy at the cut (the band
    window's ``E[60]-E[59] = 5e-14 eV`` is the same thing one index over).  A
    spectrum whose cut sits inside such a block must fire, or the armF gate
    above is passing for the trivial reason that nothing ever fires.
    """
    s = np.concatenate([_smooth(40, 8.0),
                        np.array([1e-9, 1e-9, 1e-9, 1e-13])])
    info = sc.cluster_at_cut(s, 41)
    assert info["fired"] and info["n_keep_snapped"] == 43


# ---------------------------------------------------------------------------
# 4. The bound that makes SNAP the default rather than STRICT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [2, 4, 8, 48])
def test_a_snap_moves_kappa_by_at_most_the_block_span(size):
    """The whole argument for snapping by default, as an assertion.

    ``rank_criterion``'s R19 anchor is that +41 % of retained rank cost
    5000 eV.  A closure snap is not in that class: every direction it admits
    is within ``rtol`` of one already retained, so ``kappa_eff`` moves by at
    most ``(1+rtol)**m`` over an m-member block.  At 1e-6 that is under one
    part in 10^4 for any block a crystal can produce.
    """
    s = _with_block(n=128, at=40, size=size, rel=sc.DEFAULT_RTOL / 4)
    info = sc.cluster_at_cut(s, 41)
    assert info["fired"]
    bound = (1.0 + sc.DEFAULT_RTOL) ** (info["n_keep_snapped"] - info["n_keep"])
    assert info["kappa_snapped"] / info["kappa"] <= bound * (1 + 1e-12), (
        "a closure snap moved kappa_eff by more than the block's own span — "
        "the bound that justifies snapping by default does not hold")
    assert info["kappa_snapped"] / info["kappa"] < 1.0001


def test_the_noise_floor_is_reported_and_is_rcond_dependent():
    """``eps/rcond`` is the finest relative agreement the arithmetic delivers."""
    assert sc.degeneracy_noise_rtol(1e-8) == pytest.approx(2.22e-8, rel=1e-2)
    assert sc.degeneracy_noise_rtol(1e-10) == pytest.approx(2.22e-6, rel=1e-2)
    # The 6x6x6 deck's own rcond puts the floor ABOVE the default tolerance,
    # and a report at that rcond has to SAY so rather than imply a clean bill.
    assert sc.degeneracy_noise_rtol(1e-10) > sc.DEFAULT_RTOL
    s = _with_block(at=20, size=4, rel=1e-9)
    with pytest.raises(sc.SpectralClusterError) as e:
        sc.resolve_spectral_cut(s, 22, mode="strict", where="g", rcond=1e-10)
    assert "BELOW THE FLOOR" in str(e.value)
    with pytest.raises(sc.SpectralClusterError) as e2:
        sc.resolve_spectral_cut(s, 22, mode="strict", where="g", rcond=1e-8)
    assert "BELOW THE FLOOR" not in str(e2.value)


# ---------------------------------------------------------------------------
# 5. The device face must agree with the host face, on every input ordering
# ---------------------------------------------------------------------------

def _keep_top(mag, k):
    """Boolean mask selecting the ``k`` largest magnitudes."""
    idx = np.argsort(-np.abs(mag))[:k]
    out = np.zeros(len(mag), dtype=bool)
    out[idx] = True
    return out


@pytest.mark.parametrize("order", ["descending", "ascending", "shuffled"])
@pytest.mark.parametrize("cut", [21, 22, 23])
def test_the_jit_face_matches_the_host_face(order, cut):
    """One criterion, two execution surfaces.  They must not drift.

    The ordering sweep is the point: the charge route hands the device face
    ASCENDING ``eigh`` output and the transverse route hands it an indefinite
    spectrum whose magnitudes are neither.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    s = _with_block(at=20, size=4, rel=1e-9)
    if order == "ascending":
        arr = s[::-1].copy()
    elif order == "shuffled":
        arr = s.copy()
        np.random.default_rng(3).shuffle(arr)
    else:
        arr = s.copy()

    keep = _keep_top(arr, cut)
    dev = jax.jit(sc.snap_keep_outward)(jnp.asarray(arr)[None, :],
                                        jnp.asarray(keep)[None, :])
    keep_out, n_pre, n_post = (np.asarray(x) for x in dev)
    host = sc.cluster_at_cut(arr, cut)
    assert int(n_pre[0]) == cut
    assert int(n_post[0]) == host["n_keep_snapped"], (
        f"jit face returned {int(n_post[0])}, host face {host['n_keep_snapped']} "
        f"on the same spectrum ({order}) — the two surfaces have drifted")
    assert int(keep_out[0].sum()) == host["n_keep_snapped"]
    # The snapped set must be exactly "the largest n_post by magnitude".
    assert set(np.flatnonzero(keep_out[0])) == set(
        np.argsort(-np.abs(arr))[:int(n_post[0])])


def test_the_jit_face_handles_an_indefinite_spectrum():
    """The transverse CCT is Hermitian INDEFINITE and is cut on ``|lambda|``."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    s = _with_block(at=20, size=4, rel=1e-9)
    signed = s * np.where(np.arange(len(s)) % 2, 1.0, -1.0)
    keep = _keep_top(signed, 22)
    _, n_pre, n_post = (np.asarray(x) for x in jax.jit(sc.snap_keep_outward)(
        jnp.asarray(signed)[None, :], jnp.asarray(keep)[None, :]))
    assert int(n_pre[0]) == 22 and int(n_post[0]) == 24, (
        "the guard must cut on |lambda| for the indefinite transverse channel")


def test_the_jit_face_is_batched_and_independent_per_q():
    """Per-q verdicts must not bleed: one dirty q must not move a clean one."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    dirty = _with_block(at=20, size=4, rel=1e-9)
    clean = _smooth()
    batch = np.stack([dirty, clean, dirty, clean])
    keep = np.stack([_keep_top(r, 22) for r in batch])
    _, n_pre, n_post = (np.asarray(x) for x in jax.jit(sc.snap_keep_outward)(
        jnp.asarray(batch), jnp.asarray(keep)))
    assert list(n_pre) == [22, 22, 22, 22]
    assert list(n_post) == [24, 22, 24, 22], (
        "a per-q guard leaked across the batch axis — on a real run that is "
        "a retained rank that depends on which q share a chunk")


def test_the_jit_face_never_snaps_through_an_exactly_null_tail():
    """Zero pad columns are mutually 'degenerate' and must never be swept in.

    A mesh pad contributes exactly-zero directions.  They are inert by
    construction, but they are also all equal, so a naive relative test would
    fuse them into one enormous block and snap the retained rank to full.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    s = np.concatenate([_smooth(20, 6.0), np.zeros(12)])
    keep = _keep_top(s, 20)
    _, n_pre, n_post = (np.asarray(x) for x in jax.jit(sc.snap_keep_outward)(
        jnp.asarray(s)[None, :], jnp.asarray(keep)[None, :]))
    assert int(n_post[0]) == 20, (
        f"the guard swept {int(n_post[0]) - 20} exactly-null pad directions "
        f"into the retained set; the retained rank would then depend on the "
        f"device count, which is the defect this whole family exists to stop")


def test_the_padded_distributed_helper_ignores_the_identity_pad():
    """The distributed tier's ``[C_log 0; 0 I]`` pad must not join a block.

    Its pad eigenvalues are exactly 1.0 and exactly degenerate with each
    other, so without the withdrawal in ``_close_the_cut_padded`` a cut near
    1.0 would swallow all of them and the retained rank would become a
    function of the device count.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from isdf.core import _close_the_cut_padded

    n_log, n_extra = 12, 8
    # Physical spectrum with a genuine block sitting right at 1.0, plus the
    # identity pad at exactly 1.0.  Ascending, as ``eigh`` returns.
    phys = np.array([1e-9, 1e-6, 1.0, 1.0 * (1 + 1e-9), 2.0, 5.0,
                     9.0, 20.0, 50.0, 100.0, 500.0, 1000.0])
    lam = np.sort(np.concatenate([phys, np.ones(n_extra)]))
    keep = lam > 0.5    # cuts inside the 1.0 cluster
    out = jax.jit(lambda a, b: _close_the_cut_padded(
        a, b, n_log=n_log, n_pad=n_log + n_extra, where="gate"))(
            jnp.asarray(lam)[None, :], jnp.asarray(keep)[None, :])
    n_kept = int(np.asarray(out)[0].sum())
    # The 8 identity-pad modes are withdrawn from the walk, so the snap can
    # only reach the two physical values at 1.0 — never a device-count-
    # dependent number.
    assert n_kept <= int(keep.sum()) + 2, (
        f"the closure walk pulled in identity-pad directions: kept {n_kept} "
        f"from a cut of {int(keep.sum())} with only 2 physical members "
        f"available.  The retained rank now depends on n_pad.")


# ---------------------------------------------------------------------------
# 6. Deferred refusal, for the seams that fire inside a jit
# ---------------------------------------------------------------------------

def test_a_device_finding_is_refused_at_the_next_host_seam_under_strict():
    sc.raise_if_pending(mode="off")            # clear any residue
    sc.note_device_snap("zeta rank_truncate", 1095, 1098)
    assert sc.pending(), "the finding was not recorded"
    with pytest.raises(sc.SpectralClusterError) as e:
        sc.raise_if_pending("zeta fit", mode="strict")
    assert "1095" in str(e.value) and "1098" in str(e.value)
    assert not sc.pending(), (
        "raise_if_pending must always clear — a later stage inheriting an "
        "earlier stage's finding would refuse the wrong thing")


def test_a_device_finding_only_warns_under_snap():
    sc.raise_if_pending(mode="off")
    sc.note_device_snap("zeta rank_truncate", 10, 12)
    lines = []
    sc.raise_if_pending("zeta fit", mode="snap", log=lines.append)
    assert len(lines) == 1 and "spectral-closure" in lines[0]
    assert not sc.pending()


def test_no_finding_means_no_output_and_no_raise():
    sc.raise_if_pending(mode="off")
    lines = []
    sc.raise_if_pending("zeta fit", mode="strict", log=lines.append)
    assert lines == []


# ---------------------------------------------------------------------------
# 7. The constants are declared ONCE — the ratchet its sibling earned
# ---------------------------------------------------------------------------

def test_the_default_mode_and_tolerance_have_exactly_one_literal():
    """``band_degeneracy`` learned this the hard way; the sibling inherits it.

    ``snap`` survived a day as an unwanted band-window default because "the
    default" was spelled six times in three files.  Every consumer here must
    read the constant by NAME.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for rel in ("isdf/core.py", "common/zeta_projection.py", "gw/downfold.py",
                "bandstructure/htransform.py", "centroid/pivoted_cholesky.py"):
        text = (root / rel).read_text()
        if "spectral_closure" not in text:
            offenders.append(f"{rel}: no longer wired to the guard at all")
            continue
        for literal in ('"snap"', "'snap'", '"strict"', "'strict'"):
            # A mode literal is legal only as a COMPARISON against the
            # resolved mode, never as a default.  ``mode=`` with a literal is
            # the shape that made this a ratchet.
            if f"mode={literal}" in text:
                offenders.append(f"{rel}: passes {literal} as a mode default")
    assert not offenders, "\n".join(offenders)


# ---------------------------------------------------------------------------
# 8. THE SELECTION SITE — snap-outward applied to centroid ORBITS
# ---------------------------------------------------------------------------
#
# The coordination row with the q_irr lane.  At q = 0 the selection Gram
# commutes with the whole point group, so every member of a centroid orbit
# carries the IDENTICAL Schur diagonal: an orbit is a degenerate block, and
# pivoted Cholesky's index-order tie-break between its members is exactly the
# round-off-chosen slice this module's spectral guard refuses.  The repair is
# the same repair — complete OUTWARD to whole orbits — and it forces the rank
# certificate and the ``auto`` ceiling to be re-taken on the completed set,
# which is this lane's row.

def _sel_setup(mu=12, orb=4, seed=5):
    """A parent Gram that genuinely commutes with a cyclic centroid group."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import resolve_mesh                 # noqa: F401

    n_orb = mu // orb
    perm = np.empty((orb, mu), dtype=np.int64)
    for g in range(orb):
        for o in range(n_orb):
            base = o * orb
            for j in range(orb):
                perm[g, base + j] = base + (j + g) % orb
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(mu, 3 * mu)) + 1j * rng.normal(size=(mu, 3 * mu))
    M = A @ A.conj().T
    G = sum(M[np.ix_(p, p)] for p in perm)
    G = 0.5 * (G + G.conj().T)
    for p in perm:
        assert np.max(np.abs(G[np.ix_(p, p)] - G)) < 1e-9, "G is not invariant"
    return G, perm


def test_TRUE_a_mid_orbit_selection_is_completed_outward():
    """The TRUE arm: a cut mid-orbit must be completed, never dropped."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw import downfold
    from common.collectives import resolve_mesh

    G, perm = _sel_setup()
    mesh = resolve_mesh()
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G), NamedSharding(mesh, P("x", "y")))

    fired = 0
    for mu_S in range(2, 12):
        base, _ = downfold.select_cur_centroids(
            G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
            mu_large_logical=12, print_fn=lambda *a, **k: None)
        if downfold.star_stability(base, perm).closed:
            continue                       # nothing to complete at this mu_S
        fired += 1
        got, rep = downfold.select_cur_centroids(
            G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
            mu_large_logical=12, print_fn=lambda *a, **k: None,
            sym_perm=perm)
        assert downfold.star_stability(got, perm).closed, (
            f"mu_S={mu_S}: completion did not close the orbit")
        assert set(base.tolist()) <= set(got.tolist()), (
            "completion DROPPED a selected centroid; the repair must be "
            "keep-more, or the retained subspace falls below what the rank "
            "refusal certified")
        assert got.size > base.size, "outward means larger"
        assert rep.mu_small == got.size, (
            "the SelectionReport still carries the requested mu_S — the "
            "delivered length is the authority and the report must say so")
        assert rep.star is not None and rep.star.closed
        # The re-taken certificate is on the COMPLETED set.
        assert rep.eigen_rank_kept <= got.size
    assert fired >= 1, (
        "no mu_S needed completion on this construction — the TRUE arm never "
        "ran and this gate proves nothing")


def test_FALSE_an_already_closed_selection_is_untouched():
    """The FALSE arm: whole-orbit selections must pass through bit-identical."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw import downfold
    from common.collectives import resolve_mesh

    G, perm = _sel_setup()
    mesh = resolve_mesh()
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G), NamedSharding(mesh, P("x", "y")))
    checked = 0
    for mu_S in (4, 8, 12):
        base, _ = downfold.select_cur_centroids(
            G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
            mu_large_logical=12, print_fn=lambda *a, **k: None)
        if not downfold.star_stability(base, perm).closed:
            continue
        checked += 1
        got, rep = downfold.select_cur_centroids(
            G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
            mu_large_logical=12, print_fn=lambda *a, **k: None,
            sym_perm=perm)
        assert np.array_equal(base, got), (
            f"mu_S={mu_S} was already orbit-closed and the guard moved it "
            f"anyway — a guard that fires on the clean arm is not a guard")
        assert rep.mu_small == mu_S
    assert checked >= 1, "the FALSE arm never ran"


def test_completion_that_would_exceed_the_rank_ceiling_REFUSES():
    """The knob-trap, re-taken.  Completion is not allowed to buy fiction.

    Orbit completion moves mu_S outward, and outward can cross the eigenvalue
    rank ceiling the window actually holds.  The refusal must fire on the
    COMPLETED count — the whole discipline of this function is that mu_S is
    validated against the eigenvalue rank, never against the selection
    certificate, and completion does not get an exemption.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw import downfold
    from common.collectives import resolve_mesh

    G, perm = _sel_setup()
    # Squeeze the ceiling to just under an orbit boundary by rank-deficiency:
    # project out the smallest directions so the window holds fewer than 12.
    w, V = np.linalg.eigh(G)
    w[:5] = w[5] * 1e-14                       # 7 directions above the cut
    G2 = (V * w) @ V.conj().T
    G2 = 0.5 * (G2 + G2.conj().T)
    mesh = resolve_mesh()
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G2), NamedSharding(mesh, P("x", "y")))
    lines = []
    raised = None
    for mu_S in range(2, 12):
        try:
            downfold.select_cur_centroids(
                G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
                mu_large_logical=12, print_fn=lines.append, sym_perm=perm)
        except ValueError as exc:
            if "Orbit completion needs" in str(exc):
                raised = str(exc)
                break
    assert raised is not None, (
        "no mu_S was refused: completion never crossed the ceiling on this "
        "construction, so the re-certification is untested here")
    assert "SYMMETRY-LEGAL" in raised and "lower mu_small" in raised, (
        "the refusal must say WHICH number is over the ceiling and name the "
        "fix; a bare refusal sends the user to loosen rcond")


def test_the_absence_of_sym_perm_is_reported_as_an_absence():
    """No table means UNMEASURED, and the driver must not let that read clean."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "downfold_run.py").read_text()
    assert "orbit closure NOT CHECKED" in src
    assert "ABSENCE" in src and "NOT A PASS" in src


def test_every_wired_site_imports_the_shared_guard():
    """The sweep's needs-guard list, asserted as a wiring manifest.

    If a site is dropped from the wiring, this fails by name rather than by a
    silently unguarded cut.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    manifest = {
        "isdf/core.py": ["_close_the_cut", "_close_the_cut_padded"],
        "common/zeta_projection.py": ["snap_keep_outward"],
        "gw/downfold.py": ["resolve_spectral_cut", "cluster_at_cut",
                           "orbit_complete_keep", "star_stability"],
        "bandstructure/htransform.py": ["resolve_spectral_cut"],
        "centroid/pivoted_cholesky.py": ["point_rank_closure_note"],
    }
    missing = []
    for rel, needles in manifest.items():
        text = (root / rel).read_text()
        for needle in needles:
            if needle not in text:
                missing.append(f"{rel} lost its call to {needle}")
    assert not missing, "\n".join(missing)


# ---------------------------------------------------------------------------
# THE CONSISTENCY CELL — the service's COPY of this kernel may not drift
# ---------------------------------------------------------------------------
# ``services/distrib_la`` wires the same guard onto its rank-revealing
# operations (pivoted-Cholesky pivot cuts, LU ``|diag(U)|`` cuts), and it
# may NOT import this module: services are import-isolated from ``src/`` by
# charter and by gate, and the worth of that isolation is that it has no
# exceptions.  So ``distrib_la.closure`` carries a COPY of the criterion —
# ~90 lines of stdlib arithmetic, cheaper than an import edge.
#
# A copy with nothing holding it is a fork waiting to happen, and this side
# is the only one that can see both.  These cells are the hold.  They run
# HERE rather than in the service suite for exactly that reason: the
# service cannot import ``common`` even to test itself against it, so a
# consistency check written over there would have to be written against a
# transcription of this file, which is the drift it is supposed to catch.


def _distrib_la_closure():
    """The service's copy, or a skip that names why it is unreachable."""
    try:
        from ffi._services import ensure_on_path
    except ImportError:                                    # pragma: no cover
        pytest.skip("ffi._services is the documented service bootstrap and "
                    "it is not importable here")
    ensure_on_path()
    return pytest.importorskip(
        "distrib_la.closure",
        reason="services/distrib_la is not on the path (standalone lorrax "
               "checkout); the consistency claim is unmeasured, not passed")


def _shared_spectra():
    """The synthetic spectra BOTH implementations are run over.

    Built here, once, and handed to both — a cell that built one spectrum
    per side would compare two constructions as well as two kernels, and a
    disagreement would not say which had moved.

    Deliberately hostile, not representative: exact ties, a null tail, an
    indefinite spectrum, a block open at the bottom, a single value, an
    empty spectrum, and the armF gap the whole saga turns on.
    """
    out = {}
    out["featureless"] = [0.5 ** i for i in range(24)]
    for pos in (0, 4, 8, 20):
        v = [0.5 ** i for i in range(24)]
        for j in range(4):
            v[pos + j] = v[pos] * (1.0 - 1e-9 * j)
        out[f"block@{pos}"] = v
    out["exact_ties"] = [1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-9]
    out["null_tail"] = [0.5 ** i for i in range(8)] + [0.0] * 6
    out["indefinite"] = [3.0, -3.0 * (1 - 1e-9), 1e-2, -1e-5, 1e-8]
    out["open_at_bottom"] = ([0.5 ** i for i in range(8)]
                             + [1e-9 * (1 - 1e-9 * j) for j in range(6)])
    out["armF"] = [1.0, 0.5, 1.46e-8, 1.0e-8, 1e-12]   # rel gap 0.315 at k=3
    out["single"] = [7.0]
    out["empty"] = []
    out["descending_dupes"] = [1e-4] * 8
    return out


@pytest.mark.parametrize("name", sorted(_shared_spectra()))
def test_the_service_copy_of_the_criterion_agrees_field_for_field(name):
    """``cluster_at_cut`` must return the SAME DICT on both sides.

    Every field, at every cut, on every shared spectrum — not just
    ``n_keep_snapped``.  The messages, the reports and the owner-facing
    numbers are all built out of ``gap_rel``, ``members``, ``span_rel`` and
    the two ``kappa``s, so a copy that agreed only on the rank would still
    print different evidence for the same event.
    """
    dl = _distrib_la_closure()
    values = _shared_spectra()[name]
    assert sc.DEFAULT_RTOL == dl.DEFAULT_RTOL, (
        "the two copies disagree on the tolerance itself; nothing below "
        "would be meaningful")
    assert sc.MODES == dl.MODES
    for k in range(len(values) + 2):
        a = sc.cluster_at_cut(values, k)
        b = dl.cluster_at_cut(values, k)
        assert set(a) == set(b), (name, k, set(a) ^ set(b))
        for field in sorted(a):
            av, bv = a[field], b[field]
            if isinstance(av, float) and math.isnan(av):
                assert math.isnan(bv), (name, k, field)
            else:
                assert av == bv, (
                    f"{name} at cut {k}: field {field!r} differs — "
                    f"common={av!r} distrib_la={bv!r}")


@pytest.mark.parametrize("mode", ["snap", "strict", "off"])
def test_the_service_copy_resolves_an_explicit_mode_identically(mode):
    """``resolve_spectral_cut`` under an EXPLICIT mode: same rank, same
    raise-or-not, on every shared spectrum and every cut.

    Explicit, because the DEFAULT is the one place the two deliberately
    differ and it has its own cell below.
    """
    dl = _distrib_la_closure()
    for name, values in sorted(_shared_spectra().items()):
        for k in range(len(values) + 1):
            def _run(mod):
                try:
                    n, _ = mod.resolve_spectral_cut(
                        values, k, mode=mode, log=lambda *_: None)
                    return ("ok", n)
                except Exception as exc:                   # noqa: BLE001
                    return ("raised", type(exc).__name__)
            got_common, got_svc = _run(sc), _run(dl)
            # The exception CLASSES are different types by construction —
            # each package owns its own — so compare the class NAME, which
            # is the same word on both sides on purpose.
            assert got_common == got_svc, (name, k, mode, got_common, got_svc)


def test_the_one_difference_is_the_default_and_it_is_deliberate():
    """**The divergence, pinned so it stays a decision rather than drift.**

    ``common.spectral_closure`` defaults to ``snap``: the arithmetic
    argument (a snapped spectral cut admits directions within ``rtol`` of
    ones already kept, so kappa_eff moves by under one part in 10^4) says
    refusing by default would be refusing the repair.

    ``distrib_la.closure`` defaults to ``off``, and the reason is not about
    the criterion at all — it is that the service's route semantics are
    CERTIFIED SURFACE, and a guard that arrived switched on would change
    the rank a shipped operation returns without its caller asking.  That
    is the same shape as the silent route change this tree measured at a
    QP gap of -161 eV.

    BOTH owner rows live in one place, here: flipping ``common``'s default
    to ``strict``, and flipping ``distrib_la``'s to ``snap``.  Each is a
    single constant, and this cell is what fails when one of them moves.
    """
    dl = _distrib_la_closure()
    assert sc.DEFAULT_MODE == "snap"
    assert dl.DEFAULT_MODE == "off"
    assert sc.MODE_ENV == "LORRAX_SPECTRAL_CLOSURE"
    assert dl.MODE_ENV == "LORRAX_DISTRIB_LA_CLOSURE"
    assert sc.MODE_ENV != dl.MODE_ENV, (
        "two guards with different defaults must not share one dial: a run "
        "that armed one would silently arm the other")


def test_the_service_copy_does_not_import_the_monorepo_one():
    """The copy has to be a COPY.  If ``distrib_la.closure`` ever grew a
    ``from common import ...`` the cells above would pass vacuously — and
    the service's import-isolation gate would fail, but only in the leg
    that runs it.  Cheap to assert here, next to the reason."""
    import ast
    import pathlib
    dl = _distrib_la_closure()
    tree = ast.parse(pathlib.Path(dl.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add((node.module or "").split(".")[0])
    assert "common" not in roots and "spectral_closure" not in roots, roots
