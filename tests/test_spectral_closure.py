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
def test_TRUE_a_cut_inside_a_block_fires_and_DROPS_the_block(cut_offset):
    """A cut strictly inside a planted block moves to the block's TOP.

    THE DIRECTION IS THE OWNER'S RULING OF 2026-08-10 and this is the cell
    that pins it: "if we're truncating in the middle of a block of degenerate
    singular values we should truncate the whole block."  The guard landed
    (1e0d9e23) doing the opposite, so a keep-more result here is not a
    near-miss — it is the defect this gate exists to catch, and the
    ``!= 24`` assertion below says so by name.
    """
    s = _with_block(at=20, size=4, rel=1e-9)
    info = sc.cluster_at_cut(s, 20 + cut_offset)
    assert info["fired"], "a cut inside a 4-fold block must fire"
    assert info["n_keep_closed"] == 20, (
        f"the block occupies sorted positions 20..23, so any cut inside it "
        f"must move to 20 — the whole block DROPPED; got "
        f"{info['n_keep_closed']}")
    assert info["n_keep_closed"] != 24, (
        "the cut kept the straddled block whole (the pre-ruling direction, "
        "landed at 1e0d9e23).  The ruling is to DROP it: keep fewer, floor "
        "semantics, because a block sitting at the rcond boundary is "
        "noise-adjacent and keeping it adds ill-conditioned directions")
    assert info["n_keep_closed"] < info["n_keep"], "inward, never outward"
    assert len(info["members"]) == 4
    # Both legal cuts are always reported, whichever one is taken.
    assert info["n_keep_dropped"] == 20 and info["n_keep_kept"] == 24
    assert info["direction"] == "drop_block" == sc.DEFAULT_DIRECTION


@pytest.mark.parametrize("cut_offset", [1, 2, 3])
def test_the_keep_block_direction_is_reachable_but_is_NOT_the_default(
        cut_offset):
    """The opt-out exists, is correct, and is not what any site gets.

    ``keep_block`` is retained for a call site with a measured reason to
    differ.  This cell proves it still works — so that a future lane with
    such a reason is not reaching for dead code — and the wiring ratchet
    below proves no site is using it.
    """
    s = _with_block(at=20, size=4, rel=1e-9)
    info = sc.cluster_at_cut(s, 20 + cut_offset, direction="keep_block")
    assert info["fired"] and info["n_keep_closed"] == 24
    assert info["direction"] == "keep_block"
    assert sc.DEFAULT_DIRECTION == "drop_block", (
        "the default direction moved; the owner's ruling of 2026-08-10 is "
        "drop_block and this constant is the one place it is decided")


def test_a_misspelled_direction_raises_rather_than_meaning_the_default():
    with pytest.raises(ValueError):
        sc.resolve_direction("outward")
    with pytest.raises(ValueError):
        sc.cluster_at_cut(_smooth(), 20, direction="drop")


def test_the_direction_is_not_reachable_from_the_environment(monkeypatch):
    """A ruling an env var can reverse is not a ruling.

    The MODE is a dial (a user may audit a deck under ``strict``); the
    DIRECTION is not.  Nothing in the module reads an environment variable
    for it, and this asserts the absence rather than trusting it.
    """
    import pathlib
    src = pathlib.Path(sc.__file__).read_text()
    body = src.split("def resolve_direction", 1)[1].split("\ndef ", 1)[0]
    assert "os.environ" not in body and "getenv" not in body
    monkeypatch.setenv("LORRAX_SPECTRAL_CLOSURE_DIRECTION", "keep_block")
    monkeypatch.setenv(sc.MODE_ENV, "snap")
    assert sc.resolve_direction() == "drop_block"
    s = _with_block(at=20, size=4, rel=1e-9)
    assert sc.cluster_at_cut(s, 22)["n_keep_closed"] == 20


@pytest.mark.parametrize("cut", [1, 5, 20, 24, 40, 63])
def test_FALSE_a_smooth_spectrum_never_fires_anywhere(cut):
    """The FALSE arm.  A featureless spectrum has no block to cut."""
    info = sc.cluster_at_cut(_smooth(), cut)
    assert not info["fired"], (
        f"the guard fired at cut={cut} on a spectrum with a clean 1.5-decade "
        f"ratio between every neighbour — it is finding blocks that are not "
        f"there, and every site would move its cut")
    assert info["n_keep_closed"] == cut


def test_FALSE_a_cut_at_the_block_boundary_is_silent():
    """Cutting BETWEEN whole blocks is the legal case and must stay silent."""
    s = _with_block(at=20, size=4, rel=1e-9)
    for cut in (20, 24):
        info = sc.cluster_at_cut(s, cut)
        assert not info["fired"], (
            f"cut={cut} falls between whole blocks — the legal case — and the "
            f"guard fired on it")


@pytest.mark.parametrize("direction", sc.DIRECTIONS)
def test_the_cut_the_guard_moves_to_is_itself_clean(direction):
    """The move must land in a gap, or it has relocated the problem.

    Parametrized over BOTH directions: the walk that finds the block's top
    edge is a different walk from the one that finds its bottom, and an
    off-by-one in either would leave the new cut inside the block.
    """
    s = _with_block(at=20, size=4, rel=1e-9)
    info = sc.cluster_at_cut(s, 22, direction=direction)
    assert info["gap_rel"] <= info["rtol"], "precondition: the old cut is dirty"
    assert info["gap_rel_closed"] > info["rtol"], (
        "the guard moved to a cut that is ITSELF inside a block — the walk "
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
    assert "keep 20 instead of 22" in msg, (
        "strict must name the rank that would work, and it is the DROPPED "
        "one — a strict refusal that sends the user to keep-more contradicts "
        "the ruling the snap path follows")
    assert "block holds 4 values" in msg, "strict must name the block"


def test_snap_repairs_loudly_and_names_the_move():
    s = _with_block(at=20, size=4, rel=1e-9)
    lines = []
    k, info = sc.resolve_spectral_cut(s, 22, mode="snap", where="gate",
                                      log=lines.append)
    assert k == 20 and info["fired"]
    body = "\n".join(lines)
    assert "DROPPED THE BLOCK" in body, "a silent repair is the failure mode"
    assert "22 -> 20" in body
    assert "SNAPPED OUTWARD" not in body, (
        "the pre-ruling banner survived the flip; a log line that says "
        "OUTWARD while the rank goes down is worse than no log line")


def test_a_block_that_reaches_sigma_max_REFUSES_rather_than_returning_zero():
    """The one failure the drop direction has and the keep direction does not.

    If every value from the top of the spectrum down to the cut is one
    degenerate block, dropping it leaves rank zero.  That is not a repair,
    and no mode but ``off`` may return it.
    """
    s = np.array([1.0, 1.0, 1.0, 1e-9])
    for mode in ("snap", "strict"):
        with pytest.raises(sc.SpectralBlockEmptiesCut) as e:
            sc.resolve_spectral_cut(s, 2, mode=mode, where="gate")
        assert "rank 0" in str(e.value) or "ZERO" in str(e.value)
        assert "raise rcond" in str(e.value), (
            "the refusal must name a fix; a bare refusal on a flat spectrum "
            "sends the user to loosen the guard")
    # ``off`` still returns the proposal, because ``off`` looks at nothing.
    k, _ = sc.resolve_spectral_cut(s, 2, mode="off", where="gate")
    assert k == 2
    # And the info dict says so without raising, for a caller that wants to
    # decide for itself.
    info = sc.cluster_at_cut(s, 2)
    assert info["empties"] and info["n_keep_dropped"] == 0
    # ``empties`` describes the BLOCK (it reaches sigma_max), so it is
    # reported in both directions; only ``drop_block`` is refused on it,
    # because only ``drop_block`` would act on it.
    keep_info = sc.cluster_at_cut(s, 2, direction="keep_block")
    assert keep_info["empties"] and keep_info["n_keep_closed"] == 3
    k2, _ = sc.resolve_spectral_cut(s, 2, mode="snap", where="gate",
                                    direction="keep_block",
                                    log=lambda *_: None)
    assert k2 == 3, "keep_block has no empty case here and must not refuse"


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
    assert info["fired"] and info["n_keep_closed"] == 40, (
        "the exact-degeneracy twin must fire and DROP the block back to 40")
    assert info["n_keep_kept"] == 43, "both legal cuts are still reported"


# ---------------------------------------------------------------------------
# 4. WHAT THE FLIP DID TO kappa_eff — the two directions are not symmetric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [2, 4, 8, 48])
def test_dropping_the_block_can_only_LOWER_kappa(size):
    """The direction's own argument, as an assertion.

    The old note recorded that a KEEP-snap moved ``kappa_eff`` by under 1e-4
    — bounded by ``(1+rtol)**m`` over an m-member block, which is true and is
    re-asserted below on the same construction.  This is that measurement's
    equivalent for the DROP direction, and the point is that it is not merely
    smaller: it has the opposite SIGN.  Dropping the block removes the
    smallest retained values, so ``lam_min(kept)`` rises and ``kappa_eff``
    falls.  The amplification cap ``rank_criterion`` sized the cut by is
    therefore satisfied by construction afterwards, with no slack term — the
    call sites assert it bare.
    """
    s = _with_block(n=128, at=40, size=size, rel=sc.DEFAULT_RTOL / 4)
    drop = sc.cluster_at_cut(s, 41)
    keep = sc.cluster_at_cut(s, 41, direction="keep_block")
    assert drop["fired"] and keep["fired"]

    # DROP: kappa can only improve, and it is bounded by the same block span.
    ratio_drop = drop["kappa_closed"] / drop["kappa"]
    assert ratio_drop <= 1.0, (
        f"dropping the block RAISED kappa_eff (ratio {ratio_drop:.6f}) — the "
        f"whole reason the ruling's direction is the safe one for a kept-set "
        f"quantity is that it cannot")
    # And the improvement is exactly the gap the cut moved ACROSS: the new
    # lam_min(kept) is the value just above the block, so the ratio is
    # (1 - gap_at_the_new_cut) to within the block's own span.  That is the
    # drop direction's equivalent of the old note's "<1e-4" number, and it is
    # a different KIND of number — a real gap rather than an rtol-scale
    # perturbation, because the cut crosses the block instead of sliding
    # inside it.
    assert ratio_drop == pytest.approx(1.0 - drop["gap_rel_closed"],
                                       abs=drop["span_rel"] + 1e-12)

    # KEEP: the landed direction's own bound, restated on the same case so
    # the two numbers are comparable rather than merely both true.
    bound = (1.0 + sc.DEFAULT_RTOL) ** (keep["n_keep_closed"] - keep["n_keep"])
    ratio_keep = keep["kappa_closed"] / keep["kappa"]
    assert 1.0 <= ratio_keep <= bound * (1 + 1e-12)
    assert ratio_keep < 1.0001, "the old note's number, unchanged"

    # The sign is the finding: one direction is >= 1 and the other <= 1, and
    # they bracket the unmoved cut.
    assert ratio_drop <= 1.0 <= ratio_keep


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
    dev = jax.jit(sc.close_keep_mask)(jnp.asarray(arr)[None, :],
                                      jnp.asarray(keep)[None, :])
    keep_out, n_pre, n_post = (np.asarray(x) for x in dev)
    host = sc.cluster_at_cut(arr, cut)
    assert int(n_pre[0]) == cut
    assert int(n_post[0]) == host["n_keep_closed"], (
        f"jit face returned {int(n_post[0])}, host face {host['n_keep_closed']} "
        f"on the same spectrum ({order}) — the two surfaces have drifted")
    assert int(keep_out[0].sum()) == host["n_keep_closed"]
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
    _, n_pre, n_post = (np.asarray(x) for x in jax.jit(sc.close_keep_mask)(
        jnp.asarray(signed)[None, :], jnp.asarray(keep)[None, :]))
    assert int(n_pre[0]) == 22 and int(n_post[0]) == 20, (
        "the guard must cut on |lambda| for the indefinite transverse channel")


def test_the_jit_face_is_batched_and_independent_per_q():
    """Per-q verdicts must not bleed: one dirty q must not move a clean one."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    dirty = _with_block(at=20, size=4, rel=1e-9)
    clean = _smooth()
    batch = np.stack([dirty, clean, dirty, clean])
    keep = np.stack([_keep_top(r, 22) for r in batch])
    _, n_pre, n_post = (np.asarray(x) for x in jax.jit(sc.close_keep_mask)(
        jnp.asarray(batch), jnp.asarray(keep)))
    assert list(n_pre) == [22, 22, 22, 22]
    assert list(n_post) == [20, 22, 20, 22], (
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
    _, n_pre, n_post = (np.asarray(x) for x in jax.jit(sc.close_keep_mask)(
        jnp.asarray(s)[None, :], jnp.asarray(keep)[None, :]))
    assert int(n_post[0]) == 20, (
        f"the guard swept {int(n_post[0]) - 20} exactly-null pad directions "
        f"into the retained set; the retained rank would then depend on the "
        f"device count, which is the defect this whole family exists to stop")


def test_the_padded_distributed_helper_ignores_the_identity_pad():
    """The distributed tier's ``[C_log 0; 0 I]`` pad must not join a block.

    Its pad eigenvalues are exactly 1.0 and exactly degenerate with each
    other, so a walk that reached them would move all ``n_pad - n_log`` of
    them at once and the retained rank would become a function of the DEVICE
    COUNT.  That is true in BOTH directions — ``keep_block`` would admit them
    and ``drop_block`` would discard them — so the withdrawal in
    ``_close_the_cut_padded`` is direction-independent and so is this gate.

    SWEPT OVER FOUR PAD SIZES, and the sweep is the assertion: the physical
    retained count must be the SAME number at every one.  A single-``n_pad``
    check cannot see a device-count dependence at all.

    THE FIRED CHECK IS NOT DECORATION.  Under ``drop_block`` the retained
    count can only fall, so a bound of the form ``n_kept <= keep.sum() + 2``
    — which is what this cell asserted before the direction flip — is
    satisfied by arithmetic rather than by the guard, and would pass on a
    completely broken withdrawal.  The gate therefore requires the guard to
    have FIRED on every pad size before it believes the invariance.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from isdf.core import _close_the_cut_padded

    n_log = 7
    # GEOMETRY MATTERS, and it is the opposite of what the keep direction
    # needed.  ``drop_block`` walks UP from the cut, so the pad is only in
    # danger when it lies ABOVE the cut and is degenerate with the values
    # straddling it.  So: a 4-member block just below 1.0 (within rtol of it,
    # hence linked to the pad), the pad's exact 1.0s sitting above the cut and
    # RETAINED, and one large value above everything to stop the walk — this
    # last so the case is a pad question rather than an ``empties`` one.
    #
    # A block planted anywhere else makes this gate pass with the withdrawal
    # DELETED (verified by mutation), which is how this construction was
    # arrived at rather than guessed.
    block = np.array([1.0 * (1 - k * 1e-9) for k in (1, 2, 3, 4)])
    phys = np.concatenate([np.array([50.0]), block, np.array([1e-2, 3e-4])])
    assert len(phys) == n_log
    rcond = (1 - 2.5e-9) / 50.0        # lam_max is 50.0; cut inside the block

    kept_phys, kept_pad, fired = [], [], []
    for n_pad in (12, 16, 24, 36):
        n_extra = n_pad - n_log
        lam = np.sort(np.concatenate([phys, np.ones(n_extra)]))
        keep = lam > (rcond * lam.max())
        pad_mask = (lam == 1.0)       # the pad exactly; no block member is
        assert pad_mask.sum() == n_extra
        out = jax.jit(lambda a, b, np_=n_pad: _close_the_cut_padded(
            a, b, n_log=n_log, n_pad=np_, where="gate"))(
                jnp.asarray(lam)[None, :], jnp.asarray(keep)[None, :])
        o = np.asarray(out)[0]
        fired.append(int((keep & ~pad_mask).sum()) != int((o & ~pad_mask).sum()))
        kept_phys.append(int((o & ~pad_mask).sum()))
        kept_pad.append(int((o & pad_mask).sum()) - int((keep & pad_mask).sum()))

    assert all(fired), (
        "the guard never fired on any pad size, so everything below is "
        "vacuous — this construction no longer straddles a block")
    assert kept_pad == [0, 0, 0, 0], (
        f"the block walk reached the identity pad and changed its retained "
        f"set by {kept_pad} at n_pad = 12/16/24/36.  Those directions are "
        f"exactly 1.0 and exactly degenerate with each other, so whatever the "
        f"walk does to one it does to all of them — and how many there are is "
        f"the DEVICE COUNT.")
    assert len(set(kept_phys)) == 1, (
        f"the physical retained rank depends on the pad size: {kept_phys} "
        f"for n_pad = 12/16/24/36.")


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
# round-off-chosen slice this module's spectral guard refuses.
#
# THE SELECTION FLOORS, AND IT FLOORS AGAINST THE CEILING AS RESOLVED.  Owner
# ruling, 2026-08-10: "everything the user has input on they should be
# specifying in units of points, and we should be choosing the quantity of
# orbits that comes closest to that number of points without exceeding it."
# ``mu_small`` is a BUDGET the user stated in points and overrunning it is the
# failure, so the realized set is the largest whole-orbit union at or below the
# request.  These cells were written against the previous design, in which the
# selection completed OUTWARD to whole orbits; that design is retired because
# on ``si_bse_debug`` it took mu_S from 185 to 480 (the whole parent basis) and
# then refused.  The cells are kept, pointed the other way, so the retirement
# is gated rather than merely described.
#
# The assertions below are on ``rep.eigen_rank_pool`` — the ceiling THIS RUN
# resolved — and never on how the straddled-block repair chose it, so they
# stand whichever way ``spectral_closure``'s own quantisation settles.

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


def test_TRUE_a_mid_orbit_request_is_FLOORED_INWARD():
    """The TRUE arm: a request between rungs lands on the rung BELOW it.

    And the realized set is orbit-closed, so the child has unfold tables — the
    composition the q_irr lane could not have because a point-granular
    selection cuts orbits.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw import downfold
    from common.collectives import resolve_mesh

    G, perm = _sel_setup()
    mesh = resolve_mesh()
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G), NamedSharding(mesh, P("x", "y")))
    sizes = np.bincount(downfold.centroid_orbit_id(perm))
    fired = 0
    for mu_S in range(4, 13):
        got, rep = downfold.select_cur_centroids(
            G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
            mu_large_logical=12, print_fn=lambda *a, **k: None,
            sym_perm=perm)
        assert downfold.star_stability(got, perm).closed, (
            f"mu_S={mu_S}: the orbit-mode selection is not orbit-closed")
        assert got.size <= mu_S, (
            f"mu_S={mu_S}: the floor realized {got.size} points and OVERRAN "
            f"the user's budget.  A budget that can be exceeded is not a "
            f"budget; this is the direction the ruling fixed")
        assert rep.mu_small == got.size, (
            "the SelectionReport must carry the REALIZED count — the "
            "delivered length is the authority")
        assert rep.mu_small_requested == mu_S, (
            "the report must also carry what was ASKED FOR, or a log reading "
            "168 in a deck that says 185 is unexplainable")
        # the realized count is the LARGEST legal rung at or below the request
        rungs = [n for n in range(1, 13)
                 if n % int(sizes[0]) == 0]          # equal orbits here
        assert got.size == max(r for r in rungs if r <= mu_S), (
            f"mu_S={mu_S}: floored to {got.size}, but "
            f"{max(r for r in rungs if r <= mu_S)} also fits the budget — "
            f"'closest without exceeding' means the LARGEST such value")
        if got.size < mu_S:
            fired += 1
    assert fired >= 1, (
        "every request landed exactly on a rung — the TRUE arm never ran and "
        "this gate proves nothing")


def test_RED_TWIN_a_selection_that_would_exceed_the_budget_floors_LOUDLY():
    """The red twin the ruling names: overrun is impossible, and it is SAID.

    A point-granular selection at the same mu_S takes a partial orbit — that
    is the measured default (0 of 185 admissible mu_S closed on
    ``si_bse_debug``).  Orbit mode cannot: it must come back smaller, closed,
    and it must PRINT both numbers, because a quantisation nobody is told
    about is a number that will be read as a typo.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw import downfold
    from common.collectives import resolve_mesh

    G, perm = _sel_setup()
    mesh = resolve_mesh()
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G), NamedSharding(mesh, P("x", "y")))

    mu_S = 11                                  # between the rungs 8 and 12
    point, _ = downfold.select_cur_centroids(
        G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
        mu_large_logical=12, print_fn=lambda *a, **k: None)
    assert point.size == mu_S
    assert not downfold.star_stability(point, perm).closed, (
        "the point-granular selection at mu_S=11 came back orbit-closed, so "
        "this cell is not exercising the case the floor exists for")

    lines = []
    got, rep = downfold.select_cur_centroids(
        G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
        mu_large_logical=12, print_fn=lines.append, sym_perm=perm)
    assert got.size < mu_S and downfold.star_stability(got, perm).closed
    blob = "\n".join(lines)
    assert "ORBIT-FLOORED" in blob, "the floor fired silently"
    assert f"requested {mu_S} points" in blob and f"REALIZED {got.size}" in blob, (
        f"the loud line must name BOTH numbers; got:\n{blob}")
    assert "were not spent" in blob, (
        "the line must say how much of the budget went unspent")
    # and the report says the same thing, for a consumer that reads objects
    assert rep.floored_by == mu_S - got.size
    assert rep.orbit_mode and rep.n_orbits_kept == got.size // 4


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


def test_the_floor_can_NEVER_exceed_the_rank_ceiling():
    """The knob-trap, re-taken — and now it is structural rather than checked.

    The previous design moved mu_S OUTWARD and could cross the eigenvalue
    ceiling the window holds, so it had to re-certify and refuse; on
    ``si_bse_debug`` that refusal was the production path.  Flooring cannot
    reach that state: realized <= requested <= ceiling, in POINTS, so the
    ceiling refusal only ever fires on the number the user typed.  This gate
    is the measurement of that claim over a rank-deficient window, and it also
    pins the OTHER half of the discipline — the selection certificate counts
    ORBITS in orbit mode and must not be compared to a point count.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw import downfold
    from common.collectives import resolve_mesh

    G, perm = _sel_setup()
    # Squeeze the ceiling by rank-deficiency: project out the smallest
    # directions so the window holds fewer than 12 independent ones.
    w, V = np.linalg.eigh(G)
    w[:5] = w[5] * 1e-14                       # 7 directions above the cut
    G2 = (V * w) @ V.conj().T
    G2 = 0.5 * (G2 + G2.conj().T)
    mesh = resolve_mesh()
    G_j = jax.lax.with_sharding_constraint(
        jnp.asarray(G2), NamedSharding(mesh, P("x", "y")))

    seen, over_ceiling = 0, []
    for mu_S in range(4, 13):
        lines = []
        try:
            got, rep = downfold.select_cur_centroids(
                G_j, mu_S, rcond=1e-10, select_tol=1e-12, mesh_xy=mesh,
                mu_large_logical=12, print_fn=lines.append, sym_perm=perm)
        except ValueError as exc:
            # The only admissible refusal is on the REQUEST, never on a
            # number the floor invented.
            assert "REFUSING mu_small" in str(exc), str(exc)
            continue
        seen += 1
        assert got.size <= rep.eigen_rank_pool, (
            f"mu_S={mu_S}: realized {got.size} points against a ceiling of "
            f"{rep.eigen_rank_pool} — the floor overran the rank criterion")
        assert got.size <= mu_S
        assert downfold.star_stability(got, perm).closed
        if got.size > rep.eigen_rank_pool:
            over_ceiling.append(mu_S)
        # THE KNOB TRAP: select_rank counts ORBITS here and the report must
        # say so rather than letting it be read as points.
        assert rep.orbit_rank == rep.select_rank
        assert "ORBITS" in rep.describe(), (
            "the selection certificate is in orbits and the report does not "
            "say so — this is the 42-of-42-directions-on-1908-points "
            "confusion the point_granularity_rank instrument exists for")
        assert rep.eigen_rank_kept <= got.size
    assert seen >= 1, "no mu_S was admitted; the gate measured nothing"
    assert not over_ceiling


def test_the_absence_of_sym_perm_is_reported_as_an_absence():
    """No table means UNMEASURED, and the driver must not let that read clean."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "downfold_run.py").read_text()
    assert "closure is UNMEASURED" in src
    assert "an absence and not a pass" in src


def test_every_wired_site_imports_the_shared_guard():
    """The sweep's needs-guard list, asserted as a wiring manifest.

    If a site is dropped from the wiring, this fails by name rather than by a
    silently unguarded cut.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    manifest = {
        "isdf/core.py": ["_close_the_cut", "_close_the_cut_padded"],
        "common/zeta_projection.py": ["close_keep_mask"],
        "gw/downfold.py": ["resolve_spectral_cut", "cluster_at_cut",
                           "star_stability"],
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


def test_NO_wired_site_opts_out_of_the_ruling_direction():
    """The ratchet that makes ``keep_block`` an exception rather than a habit.

    The owner's ruling is the default and every site takes it.  A site with a
    measured reason to differ may pass ``direction="keep_block"`` — and when
    one does, this gate fails and forces the reason to be written down here
    rather than discovered later in a log.  **Today the correct list is
    empty.**
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "spectral_closure.py":
            continue           # the guard documents its own opt-out
        text = path.read_text()
        if ('direction="keep_block"' in text
                or "direction='keep_block'" in text):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, (
        "these sites opt out of the ruling's direction (drop the straddled "
        "block) and none is recorded as having a measured reason:\n  "
        + "\n  ".join(offenders)
        + "\nIf the reason is real, add the site here with the measurement. "
          "If a site NEEDS keep-more to stay correct, that is a finding about "
          "what consumes its retained span, and it is reported rather than "
          "flagged away.")


def test_the_default_direction_is_spelled_exactly_once():
    """Same ratchet the mode has: the way a default survives unwanted is that
    it is spelled six times in three files."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    literals = []
    for path in root.rglob("*.py"):
        if path.name == "spectral_closure.py":
            continue
        text = path.read_text()
        if 'direction="drop_block"' in text or "direction='drop_block'" in text:
            literals.append(str(path.relative_to(root)))
    assert not literals, (
        "these sites re-spell the default direction as a literal instead of "
        "letting spectral_closure.DEFAULT_DIRECTION decide:\n  "
        + "\n  ".join(literals))
