"""Layer L-a: the degeneracy-closure guard on this service's rank cuts.

No mesh, no devices, no ``.so``.  The criterion is stdlib arithmetic and
the plan-level wiring resolves to ``native`` on a fake mesh, so this whole
file is a laptop and milliseconds — which is the point, because a guard
that can only be checked on a machine is a guard nobody checks.

WHAT IS BEING GUARDED, in one paragraph, because a test file that only
says "it snaps" is not evidence that snapping is the right thing.  A
rank-revealing factorization cuts a spectrum, and the retained set is a
SUBSPACE.  A symmetry that commutes with the factored operator maps each
of its eigenspaces onto itself and mixes the members of a degenerate block
freely.  Cut between whole blocks and the retained span is invariant; cut
THROUGH a block and the retained span is a symmetry-arbitrary slice of an
eigenspace, different at ``Sq`` than at ``q``, chosen by round-off.  The
guard moves such a cut OUTWARD past the block, or refuses.

EVERY CHECK SHIPS WITH THE CASE WHERE IT RETURNS FALSE (charter, no
exceptions).  A guard that fires on both arms is not a guard, so each
TRUE arm here is paired with a spectrum the guard must be silent on.

THE DEFAULT IS ``off`` AND THAT IS ASSERTED, not assumed.  This service's
route semantics are certified surface; the guard is opt-in this round, and
"opt-in" is only true while something fails when it stops being.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import distrib_la as D
from distrib_la import closure as C


# ---------------------------------------------------------------------------
# Fixtures: the two spectra every arm is built on
# ---------------------------------------------------------------------------

def _clean(n=24, decay=0.5):
    """A featureless geometric spectrum.  NO cut anywhere is inside a block.

    The FALSE arm's raw material.  Successive values differ by a relative
    ``1 - decay = 0.5``, which is five decades above the default 1e-6, so
    the guard has to be silent at every one of the ``n - 1`` interior
    boundaries — and the cells below check every one of them, not a
    convenient sample.
    """
    return [decay ** i for i in range(n)]


def _with_block(pos, size=4, n=24, decay=0.5, rel=1e-9):
    """``_clean`` with a degenerate block of ``size`` planted at ``pos``.

    The block members agree to ``rel`` relative to each other, so at the
    default ``rtol = 1e-6`` they are one block, and a cut anywhere strictly
    inside ``[pos, pos + size)`` slices it.  ``rel`` is spread across the
    members rather than repeated, so the block is a genuine cluster and not
    a run of bit-identical floats (which would pass a sloppier criterion
    that only tested equality).
    """
    v = _clean(n, decay)
    base = v[pos]
    for j in range(size):
        v[pos + j] = base * (1.0 - rel * j)
    return v


# ---------------------------------------------------------------------------
# THE CRITERION — TRUE against FALSE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [1, 2, 3])
def test_a_cut_inside_a_planted_block_snaps_out_to_the_block_edge(offset):
    """TRUE arm.  Cut at ``pos + offset`` — strictly inside a 4-member
    block — and the guard must deliver exactly the block's far edge.

    ``offset`` sweeps all three interior positions rather than one: a
    walk that stopped after a single step would pass at ``offset=3`` and
    fail at ``offset=1``, and one position cannot tell those apart.
    """
    pos, size = 8, 4
    v = _with_block(pos, size)
    rc = D.rank_cut("cholesky", v, n_keep=pos + offset, closure="snap",
                    log=lambda *_: None)
    assert rc.fired
    assert rc.n_keep_requested == pos + offset
    assert rc.n_keep == pos + size, (
        f"cut at {pos + offset} should snap to the block edge {pos + size}, "
        f"got {rc.n_keep}")
    assert rc.moved == size - offset
    assert len(rc.info["members"]) == size


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_the_cut_the_guard_moves_to_is_itself_clean(offset):
    """The difference between FIXING the problem and RELOCATING it.

    Snapping to a boundary that is itself mid-block would satisfy every
    assertion above and leave the calculation exactly as broken.  So the
    delivered cut is re-checked with a fresh call: it must not fire.
    """
    v = _with_block(8, 4)
    rc = D.rank_cut("cholesky", v, n_keep=8 + offset, closure="snap",
                    log=lambda *_: None)
    again = D.rank_cut("cholesky", v, n_keep=rc.n_keep, closure="snap",
                       log=lambda *_: None)
    assert not again.fired
    assert again.n_keep == rc.n_keep


@pytest.mark.parametrize("k", list(range(1, 24)))
def test_a_featureless_spectrum_fires_at_no_cut_anywhere(k):
    """FALSE arm, and it is EVERY interior cut rather than a sample.

    If this fired anywhere the TRUE arm above would be measuring "the
    guard fires", not "the guard discriminates".
    """
    rc = D.rank_cut("cholesky", _clean(24), n_keep=k, closure="strict")
    assert not rc.fired
    assert rc.n_keep == k
    assert rc.info["gap_rel"] > C.DEFAULT_RTOL


def test_a_cut_at_either_end_slices_nothing_and_is_exempt():
    """Boundary condition, stated rather than left to the arithmetic: a cut
    that keeps everything or nothing cannot be inside a block, and the
    report says ``exempt`` rather than ``clean`` so the two are not
    confused in a log."""
    v = _with_block(0, 24, n=24)          # the WHOLE spectrum is one block
    for k in (0, 24):
        rc = D.rank_cut("eigh", v, n_keep=k, closure="strict")
        assert not rc.fired
        assert rc.info["gap_rel"] == math.inf
        assert "exempt" in rc.describe()


# ---------------------------------------------------------------------------
# THE THREE MODES
# ---------------------------------------------------------------------------

def test_snap_is_loud_and_names_what_it_moved():
    """``snap`` must never be a silent repair.  The message names the
    block, the tolerance it was measured against, the move, and the
    kappa_eff either side of it — a rank that changed with no line in the
    log is indistinguishable from a rank that did not."""
    said: list[str] = []
    rc = D.rank_cut("cholesky", _with_block(8, 4), n_keep=10,
                    closure="snap", log=said.append)
    text = "\n".join(said)
    assert rc.n_keep == 12
    assert "SNAPPED OUTWARD" in text
    assert "10 -> 12" in text
    assert "kappa_eff" in text
    assert "degenerate block" in text


def test_strict_refuses_and_names_the_rank_that_would_work():
    """A refusal that does not say what to do instead is a stack trace.
    The message must carry the working rank AND both escape hatches."""
    with pytest.raises(C.SpectralClusterError) as excinfo:
        D.rank_cut("solve_lu", _with_block(8, 4), n_keep=10, closure="strict")
    msg = str(excinfo.value)
    assert "keep 12 instead of 10" in msg
    assert f"{C.MODE_ENV}=snap" in msg
    assert f"{C.MODE_ENV}=off" in msg
    # And it says which numbers were cut, so a reader knows whether to
    # believe the spectrum in the first place.
    assert C.RANK_SPECTRA["solve_lu"] in msg


def test_off_is_silent_and_changes_nothing():
    """``off`` returns the proposal untouched and does not even look.  It
    exists for the same reason its sibling keeps one: a guard with no off
    switch gets deleted rather than configured."""
    said: list[str] = []
    rc = D.rank_cut("cholesky", _with_block(8, 4), n_keep=10, closure="off",
                    log=said.append)
    assert said == []
    assert rc.n_keep == 10 and not rc.fired
    assert rc.mode == "off"


def test_off_still_reports_a_full_info_dict():
    """``off`` must not hand back a short dict.  A caller that logs the
    numbers has to be able to log them whether or not the guard was armed,
    or every log line grows a branch."""
    armed = D.rank_cut("cholesky", _clean(24), n_keep=6, closure="snap")
    dis = D.rank_cut("cholesky", _clean(24), n_keep=6, closure="off")
    assert set(armed.info) == set(dis.info)


def test_the_false_arm_is_silent_in_all_three_modes():
    """The pairing that makes the three cells above evidence: on a clean
    spectrum, ``snap`` says nothing, ``strict`` does not raise, ``off`` is
    ``off``, and all three deliver the same rank."""
    said: list[str] = []
    ranks = set()
    for mode in C.MODES:
        rc = D.rank_cut("cholesky", _clean(24), n_keep=6, closure=mode,
                        log=said.append)
        ranks.add(rc.n_keep)
    assert said == []
    assert ranks == {6}


def test_disarmed_and_clean_do_not_look_alike():
    """Measurement rule 10.  "No news" and "a good number" must be
    distinguishable in a log, or a reader cannot tell a checked cut from an
    unchecked one — which for an opt-in guard is the single most likely way
    to be misled by it."""
    clean = D.rank_cut("cholesky", _clean(24), n_keep=6, closure="strict")
    disarmed = D.rank_cut("cholesky", _clean(24), n_keep=6, closure="off")
    assert "DISARMED" in disarmed.describe()
    assert "DISARMED" not in clean.describe()
    assert "falls in a gap" in clean.describe()
    assert clean.describe() != disarmed.describe()


# ---------------------------------------------------------------------------
# THE DEFAULT, AND THE DIAL
# ---------------------------------------------------------------------------

def test_the_default_mode_is_off_and_that_is_the_opt_in_claim():
    """THE certified-surface promise, as one assertion.

    Flipping this constant is an owner decision (see
    ``distrib_la.closure``); flipping it by accident is what this cell
    exists to prevent.  If it ever legitimately becomes ``snap``, this
    cell is the place the change is registered.
    """
    assert C.DEFAULT_MODE == "off"
    assert D.CLOSURE_DEFAULT_MODE == "off"
    rc = D.rank_cut("cholesky", _with_block(8, 4), n_keep=10)
    assert rc.mode == "off" and rc.n_keep == 10 and not rc.fired


def test_the_environment_arms_it(monkeypatch):
    """The env default.  A driver that cannot reach the kwarg still has a
    way in, which is what makes the guard reachable from a deck-driven
    run without a new deck key this round."""
    monkeypatch.setenv(C.MODE_ENV, "snap")
    rc = D.rank_cut("cholesky", _with_block(8, 4), n_keep=10,
                    log=lambda *_: None)
    assert rc.mode == "snap" and rc.n_keep == 12


def test_an_empty_environment_variable_is_not_a_mode(monkeypatch):
    """``FOO=`` is how a shell says "unset".  Reading it as a mode would
    raise on a blank string, which turns an inert variable into a crash."""
    monkeypatch.setenv(C.MODE_ENV, "")
    assert C.mode_from_env() is None
    assert D.rank_cut("cholesky", _clean(24), n_keep=6).mode == "off"


def test_a_misspelled_mode_raises_rather_than_disarming(monkeypatch):
    """A guard silently disarmed by a typo is worse than no guard, because
    the log then shows a clean run.  Both routes in: the kwarg and the
    environment."""
    with pytest.raises(ValueError, match="not one of"):
        D.rank_cut("cholesky", _clean(24), n_keep=6, closure="snapp")
    monkeypatch.setenv(C.MODE_ENV, "SNAP-ish")
    with pytest.raises(ValueError, match="not one of"):
        D.rank_cut("cholesky", _clean(24), n_keep=6)


def test_the_mode_spelling_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv(C.MODE_ENV, "  SNAP ")
    assert D.rank_cut("cholesky", _clean(24), n_keep=6).mode == "snap"


def test_the_monorepo_guards_dial_does_not_arm_this_one(monkeypatch):
    """THE reason there are two dial names.

    ``LORRAX_SPECTRAL_CLOSURE`` arms the monorepo's zeta-fit guard and
    DEFAULTS TO SNAP there.  If this service read the same variable, a run
    that armed that seam would silently change the rank ``cholesky``
    hands back here — a route-affecting change nobody asked for, which is
    this package's named worst failure mode.
    """
    monkeypatch.setenv("LORRAX_SPECTRAL_CLOSURE", "strict")
    monkeypatch.delenv(C.MODE_ENV, raising=False)
    rc = D.rank_cut("cholesky", _with_block(8, 4), n_keep=10)
    assert rc.mode == "off" and rc.n_keep == 10


# ---------------------------------------------------------------------------
# THE PROPOSAL: exactly one, and the guard only ever grows it
# ---------------------------------------------------------------------------

def test_exactly_one_proposal_is_required():
    """Two proposals is two criteria, and a guard whose promise is "at
    least what you asked for" has to know what was asked."""
    for kw in ({}, dict(n_keep=4, rcond=1e-8)):
        with pytest.raises(ValueError, match="exactly one"):
            D.rank_cut("cholesky", _clean(24), **kw)


@pytest.mark.parametrize("rcond", [1e-2, 1e-4, 1e-6, 1e-8])
def test_the_rcond_route_is_the_relative_threshold_and_nothing_else(rcond):
    """``rcond`` must mean ``|v| > |v|_max * rcond`` and not a variant of
    it.  Re-derived here rather than re-run, so a change to the rule fails
    against the rule instead of against itself."""
    v = _clean(24)
    expect = sum(1 for x in v if abs(x) > max(abs(y) for y in v) * rcond)
    rc = D.rank_cut("cholesky", v, rcond=rcond, closure="strict")
    assert rc.n_keep_requested == expect
    assert rc.n_keep == expect


@pytest.mark.parametrize("pos", [0, 4, 8, 16, 20])
@pytest.mark.parametrize("k", [3, 6, 9, 12, 18, 22])
def test_the_guard_never_returns_a_smaller_rank(pos, k):
    """THE property, swept.  Outward is the only direction: every answer is
    at least the proposal, wherever the block sits.  A guard that could
    shrink a rank would be a second rank criterion, and ``rcond``'s
    amplification cap is the only one there is.

    ``snap`` and ``off`` only.  ``strict`` has no rank to be monotone
    ABOUT — it refuses instead of returning one — and folding it in here
    would make the sweep assert that strict never fires, which is the
    opposite of what strict is for.  Its refusal is its own cell.
    """
    v = _with_block(pos, 4)
    for mode in ("snap", "off"):
        rc = D.rank_cut("eigh", v, n_keep=k, closure=mode,
                        log=lambda *_: None)
        assert rc.n_keep >= k
        assert rc.moved >= 0


@pytest.mark.parametrize("pos", [0, 4, 8, 16, 20])
@pytest.mark.parametrize("k", [3, 6, 9, 12, 18, 22])
def test_strict_refuses_exactly_where_snap_moves(pos, k):
    """The discrimination that pairs with the sweep above, over the same
    grid: ``strict`` raises for exactly the (pos, k) where ``snap`` moved
    the cut, and returns quietly for exactly the ones where it did not.
    Neither mode may be a constant function of its input."""
    v = _with_block(pos, 4)
    snapped = D.rank_cut("eigh", v, n_keep=k, closure="snap",
                         log=lambda *_: None)
    if snapped.fired:
        with pytest.raises(C.SpectralClusterError):
            D.rank_cut("eigh", v, n_keep=k, closure="strict")
    else:
        assert D.rank_cut("eigh", v, n_keep=k, closure="strict").n_keep == k


def test_an_exactly_null_tail_is_never_swallowed():
    """The pad, and the reason it matters HERE rather than in the monorepo.

    A distributed factorization pads a non-dividing extent to the mesh, and
    the pad's entries are exactly 0 (or exactly 1, in an identity fill).
    Exactly-equal values are trivially "degenerate", so a walk that did not
    stop at a null pair would swallow the whole pad and make the retained
    rank a FUNCTION OF THE DEVICE COUNT — a different answer on a 2x2 than
    on a 4x4, for the same matrix.
    """
    v = _clean(8) + [0.0] * 6
    rc = D.rank_cut("cholesky", v, n_keep=8, closure="strict")
    assert not rc.fired and rc.n_keep == 8
    # ... and a cut INSIDE the null tail is exempt for the same reason,
    # rather than snapping to the bottom of the spectrum.
    rc = D.rank_cut("cholesky", v, n_keep=11, closure="strict")
    assert not rc.fired and rc.n_keep == 11


def test_a_block_still_open_at_the_bottom_says_so():
    """When the snap runs off the end of the spectrum there is no closed
    cut to snap to, and the retained span becomes everything.  That is a
    legal answer and a loud one — silently returning ``n`` would hide the
    fact that the criterion selected nothing."""
    said: list[str] = []
    v = _clean(8) + [1e-9 * (1.0 - 1e-9 * j) for j in range(6)]
    rc = D.rank_cut("cholesky", v, n_keep=10, closure="snap", log=said.append)
    assert rc.fired and rc.n_keep == len(v)
    assert "STILL OPEN AT THE BOTTOM" in "\n".join(said)


# ---------------------------------------------------------------------------
# WHAT GETS CUT: the two rank-revealing spectra
# ---------------------------------------------------------------------------

def test_the_cholesky_pivot_spectrum_is_the_squared_diagonal():
    """``|diag(L)|^2``, on the last two axes, single tile and stack alike."""
    jnp = pytest.importorskip("jax.numpy")
    import numpy as np
    rng = np.random.default_rng(11)
    A = rng.standard_normal((3, 6, 6)) + 1j * rng.standard_normal((3, 6, 6))
    A = A @ np.conj(np.swapaxes(A, -1, -2)) + 6 * np.eye(6)
    L = np.linalg.cholesky(A)
    got = np.asarray(D.cholesky_pivot_spectrum(jnp.asarray(L)))
    assert got.shape == (3, 6)
    assert np.allclose(got, np.abs(np.diagonal(L, axis1=-2, axis2=-1)) ** 2)
    assert np.asarray(D.cholesky_pivot_spectrum(jnp.asarray(L[0]))).shape == (6,)


def test_the_pivot_spectrum_is_SQUARED_and_it_matters():
    """RED TWIN for the square.  A relative-gap criterion is not invariant
    under a square root: two pivots agreeing to 1e-6 have diagonals
    agreeing to ~5e-7, so cutting ``|L_ii|`` instead of ``|L_ii|^2`` moves
    every block boundary by a factor of two in tolerance.

    Built at exactly the tolerance so the two answers DIFFER: on the
    squared spectrum the pair is one block and the guard fires; on the
    unsquared one it is two neighbours and the guard is silent.  If the
    helper stopped squaring, this cell fails.
    """
    jnp = pytest.importorskip("jax.numpy")
    import numpy as np
    # Pivots differing by 1.6e-6 relative -> NOT one block at rtol 1e-6.
    # Their square roots differ by 8e-7 relative -> one block.  So reading
    # the diagonal instead of the pivot invents a degeneracy.
    piv = [1.0, 1.0 - 1.6e-6, 1e-4]
    L = np.diag(np.sqrt(np.asarray(piv)))
    got = np.asarray(D.cholesky_pivot_spectrum(jnp.asarray(L)))
    assert not D.rank_cut("cholesky", got, n_keep=1, closure="strict").fired
    diag = np.abs(np.diagonal(L))
    assert D.rank_cut("cholesky", diag, n_keep=1, closure="snap",
                      log=lambda *_: None).fired, (
        "the unsquared diagonal must invent a block here, or this cell is "
        "not measuring the square")


def test_the_lu_rank_spectrum_is_the_unsquared_packed_diagonal():
    """``getrf`` packs ``U`` on and above the diagonal, so the diagonal of
    the packed factor IS ``diag(U)``; and it is NOT squared, because
    ``A = P L U`` already lives on the operator's scale."""
    jnp = pytest.importorskip("jax.numpy")
    import numpy as np
    rng = np.random.default_rng(13)
    LU = rng.standard_normal((2, 5, 5)) + 1j * rng.standard_normal((2, 5, 5))
    got = np.asarray(D.lu_rank_spectrum(jnp.asarray(LU)))
    assert got.shape == (2, 5)
    assert np.allclose(got, np.abs(np.diagonal(LU, axis1=-2, axis2=-1)))


def test_an_unknown_op_refuses_by_name():
    with pytest.raises(ValueError, match="unknown op"):
        D.rank_cut("qr", _clean(24), n_keep=4)


# ---------------------------------------------------------------------------
# THE WIRING: plan / factor carry the mode, and carry NOTHING ELSE new
# ---------------------------------------------------------------------------

class _FakeMesh:
    """``mesh.shape``/``devices`` — everything a NATIVE plan touches.

    A native plan builds no ``NamedSharding``, so it needs no real
    devices; using a fake keeps these cells in the laptop tier where the
    wiring they check actually lives.
    """

    def __init__(self, Px=2, Py=2, platform="cpu"):
        self.shape = {"x": Px, "y": Py}
        self.devices = SimpleNamespace(
            size=Px * Py, flat=[SimpleNamespace(platform=platform)])


def test_the_plan_carries_the_mode_and_the_kwarg_beats_the_env(monkeypatch):
    monkeypatch.setenv(C.MODE_ENV, "strict")
    assert D.plan("cholesky", _FakeMesh(), backend="off").closure == "strict"
    assert D.plan("cholesky", _FakeMesh(), backend="off",
                  closure="snap").closure == "snap"
    monkeypatch.delenv(C.MODE_ENV, raising=False)
    assert D.plan("cholesky", _FakeMesh(), backend="off").closure == "off"


def test_the_mode_is_resolved_once_at_plan_time_not_per_call(monkeypatch):
    """A guard whose mode can change between two calls of ONE plan is a
    guard whose verdict nobody can reproduce.  The plan is the resolve-once
    object; the mode resolves with everything else."""
    monkeypatch.setenv(C.MODE_ENV, "snap")
    p = D.plan("cholesky", _FakeMesh(), backend="off")
    monkeypatch.setenv(C.MODE_ENV, "strict")
    assert p.closure == "snap"
    assert p.rank_cut(_with_block(8, 4), n_keep=10,
                      log=lambda *_: None).n_keep == 12   # snapped, not raised


def test_the_plan_refuses_a_per_call_override():
    """The same rule as ``backend=``: a caller that overrides here has
    taken back the decision the plan exists to make once."""
    p = D.plan("cholesky", _FakeMesh(), backend="off", closure="snap")
    for kw in (dict(op="eigh"), dict(closure="off")):
        with pytest.raises(ValueError, match="cannot be overridden"):
            p.rank_cut(_clean(24), n_keep=6, **kw)


def test_the_plan_binds_its_own_op():
    """``Plan.rank_cut`` must name the plan's op in its report, so a log
    line from a cholesky plan cannot read as an eigh one."""
    p = D.plan("cholesky", _FakeMesh(), backend="off", closure="strict")
    with pytest.raises(C.SpectralClusterError, match="Cholesky pivot"):
        p.rank_cut(_with_block(8, 4), n_keep=10)


def _real_1x1():
    """A real 1x1 CPU mesh — one device, always available.

    Needed by the two cells below and by nothing else here: ``native2d``
    is not the ``native`` backend, so a plan on it builds real
    ``NamedSharding``s and a fake mesh will not do.  One device keeps it
    in the laptop tier all the same.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


@pytest.mark.parametrize("backend", ["off", "native2d"])
def test_arming_the_closure_changes_no_resolution_fact(backend):
    """**THE certified-surface guarantee, as a check rather than a claim.**

    Resolution, shardings, donation and the divisibility guard must be
    identical across all three modes.  The guard is a NEW decision on a
    NEW surface; if any of these moved it would be a silent route change,
    which is this package's named worst failure mode (a full G0W0 that
    finished with rc=0 and a QP gap of -161 eV).

    ``native2d`` is the arm that also compares ``batched_route`` and the
    two real shardings -- a ``native`` plan has neither, so the ``off``
    arm alone would leave the route half of the claim unmeasured.
    """
    pytest.importorskip("jax")
    mesh = _FakeMesh() if backend == "off" else _real_1x1()
    fields = ["op", "requested", "backend", "n", "in_sharding",
              "batch_in_sharding", "donates"]
    if backend != "off":
        fields.append("batched_route")
    seen = {}
    for mode in C.MODES:
        p = D.plan("cholesky", mesh, backend=backend, n=64, closure=mode)
        seen[mode] = tuple(str(getattr(p, f)) for f in fields)
        assert p.closure == mode
    assert len(set(seen.values())) == 1, seen


def test_the_closure_is_not_in_the_scan_cache_key():
    """It cannot be, and this is where that is written down: a compiled
    scan is the same executable whatever the guard says, because the guard
    runs on HOST after the call returns.  Keying on it would multiply every
    compile by three for nothing; the risk in the other direction — a cache
    too COARSE — does not exist here for the same reason.

    Checked two ways, because either alone is weak: the signature function
    does not take a closure at all (structure), and two plans differing
    ONLY in closure produce the same signature for the same operands
    (behaviour).
    """
    import inspect

    import jax.numpy as jnp

    from distrib_la.plan import scan_signature
    assert "closure" not in inspect.signature(scan_signature).parameters
    mesh = _real_1x1()
    A = jnp.zeros((2, 4, 4))
    sigs = {D.plan("cholesky", mesh, backend="native2d", n=4,
                   closure=m).closure:
            scan_signature("cholesky", "native2d", mesh, (A,), {})
            for m in C.MODES}
    assert len(set(sigs.values())) == 1, sigs


def test_the_describe_line_mentions_the_guard_only_when_it_is_armed():
    off = D.plan("cholesky", _FakeMesh(), backend="off").describe()
    on = D.plan("cholesky", _FakeMesh(), backend="off",
                closure="strict").describe()
    assert "rank-closure" not in off
    assert "rank-closure=strict" in on


# ---------------------------------------------------------------------------
# RATCHETS — the wiring cannot be dropped silently
# ---------------------------------------------------------------------------

def test_no_entry_point_spells_a_mode_literal_as_its_default():
    """The way ``snap`` survived a day as an unwanted default in the
    band-window guard was that "the default" was spelled six times in
    three files.  Here it is spelled once, and this is the ratchet: the
    only occurrence of a bare mode literal as a default anywhere in the
    package is :data:`closure.DEFAULT_MODE` itself.

    AST, not a regex over lines: this module's prose is full of the words
    ``snap`` and ``off``, and a line-based ratchet would either fire on
    every docstring that explains the modes or be loosened until it fired
    on nothing.  Parsing asks the precise question — is a mode STRING the
    default value of a ``closure`` parameter or field, anywhere but the
    one constant.
    """
    import ast
    import pathlib
    pkg = pathlib.Path(D.__file__).parent
    bad = []

    def _is_mode_literal(node):
        return isinstance(node, ast.Constant) and node.value in C.MODES

    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # (a) a keyword default: ``def f(*, closure="snap")``
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg, dflt in list(zip(a.kwonlyargs, a.kw_defaults)) + list(
                        zip(a.args[len(a.args) - len(a.defaults):],
                            a.defaults)):
                    if (arg.arg == "closure" and dflt is not None
                            and _is_mode_literal(dflt)):
                        bad.append(f"{path.name}:{node.lineno}: "
                                   f"def {node.name}(closure={dflt.value!r})")
            # (b) an annotated field: ``closure: str = "snap"``
            if (isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.value is not None
                    and _is_mode_literal(node.value)):
                if not (path.name == "closure.py"
                        and node.target.id == "DEFAULT_MODE"):
                    bad.append(f"{path.name}:{node.lineno}: "
                               f"{node.target.id} = {node.value.value!r}")
    assert not bad, ("a mode literal is being used as a default; the one "
                     "place the default is decided is closure.DEFAULT_MODE\n"
                     + "\n".join(bad))


def test_that_ratchet_can_fail(tmp_path):
    """RED TWIN for the ratchet.  A ratchet nobody has seen fail is a
    ratchet that may be scanning the wrong thing — this one was, in its
    first draft, matching prose in a docstring."""
    import ast
    src = 'def f(*, closure="snap"):\n    """closure: off"""\n    return 1\n'
    tree = ast.parse(src)
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and any(a.arg == "closure" for a in n.args.kwonlyargs)
            and any(isinstance(d, ast.Constant) and d.value in C.MODES
                    for d in n.args.kw_defaults if d is not None)]
    assert len(hits) == 1


def test_every_wired_entry_point_still_calls_the_guard_by_name():
    """Dropping the wiring must fail BY NAME rather than as a silently
    unguarded cut.  Both surfaces the feature was asked for -- the plan and
    the factor token -- plus the free function they delegate to."""
    assert callable(D.rank_cut)
    assert callable(D.Plan.rank_cut)
    assert callable(D.FactorToken.rank_cut)
    for name in ("rank_cut", "RankCut", "cholesky_pivot_spectrum",
                 "lu_rank_spectrum", "SpectralClusterError", "CLOSURE_MODES",
                 "CLOSURE_DEFAULT_MODE", "CLOSURE_RTOL", "CLOSURE_ENV"):
        assert name in D.__all__, f"{name} left the package surface"
        assert hasattr(D, name)


def test_the_closure_module_needs_no_jax_to_state_its_vocabulary():
    """The criterion and the vocabulary must be readable with no jax and no
    ``.so``, exactly as ``BACKEND_CHOICES`` is -- a deck parser or a log
    formatter must not need the FFI layer to name a mode.  The two spectrum
    helpers import jax inside their bodies, which is what buys this."""
    import ast
    import pathlib
    src = pathlib.Path(C.__file__).read_text()
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import)
             for a in n.names}
    names |= {(n.module or "").split(".")[0] for n in top
              if isinstance(n, ast.ImportFrom)}
    assert not (names - {"__future__", "dataclasses", "typing"}), names
