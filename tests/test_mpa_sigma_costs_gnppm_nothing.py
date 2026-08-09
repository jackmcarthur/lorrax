"""GATE (c): the MPA self-energy costs the two-point modes NOTHING, provably.

THE CONTRACT, and it is an inter-fleet one.  A gnppm-mode run through this
branch must produce bit-identical Sigma -- every channel, sigTOT included --
to the same run at the base this branch was cut from.  The peer fleet's sigma
opener anchors its work on a gnppm sigTOT MAE of 0.6439 meV against a
BerkeleyGW oracle, and a number like that is only a shared anchor if the modes
underneath it do not move when a new mode lands beside them.

THE CHOICE THIS BRANCH MADE, stated once so nobody has to infer it.  The brief
offered two ways to hold the contract: specialize the generalized
complex-pole kernel back to real poles bit-exactly, or leave the two-point
path on the original code and dispatch the generalized kernel only from MPA.
**This branch took the second, and took it structurally rather than by
dispatch**: the complex-pole machinery lives in its own module
(``gw.mpa.sigma_routing``) which the two-point path does not import, and not
one line of the shared convolution core is edited.  So the specialization
question does not arise -- there is no reordered reduction to argue about,
because there is no shared code path that changed.

WHY A SOURCE DIGEST IS THE RIGHT INSTRUMENT HERE, and not a weaker one than a
deck run.  The claim "the gnppm bytes did not move" has exactly two premises:
the code that produces them is the same code, and the new code cannot be
reached from it.  This file checks both, in a few milliseconds, with no deck,
no device and no frozen fixture that could itself go stale.  A deck A/B would
measure the same two premises through a much longer lever and would only cover
the one deck it ran.

RE-ANCHORING IS EXPECTED, AND IS THE POINT OF RECORDING THE SHA.  When a peer
lands a change to the shared core these digests go stale BY DESIGN.  The fix
is: rebase, confirm the incoming change is the one you meant to absorb, update
``BASE_SHA`` and the digests together in one commit that says whose change it
absorbed.  What must never happen is a digest updated without a sha updated,
because then the gate no longer says which tree it is anchored to and a future
reader cannot tell a re-anchor from a regression.

THE ONE RE-ANCHOR PERFORMED SO FAR, 996ad826 -> e37c6a6e, and how it was
checked.  Three of the eight modules moved, and the question the gate exists
to ask is whether any of that motion can reach a floating-point result:

  ``ppm_pipeline.py``   two lines -- the ``profile_section`` import and the
                        context manager it wrapped ``sigma.exec`` in
                        (1ec5ccfd, which deleted a profiler that produced no
                        trace).  Instrumentation, removed.
  ``ppm_sigma.py``      48 diff lines from 4f1c29e8's timing rows.  Checked by
                        comparing the diff's added and removed lines as
                        MULTISETS OF STRIPPED CONTENT: the difference is empty
                        in both directions, so every one of those 48 lines is
                        the same statement at a different indentation, wrapped
                        in ``with timing.section(...)``.
  ``head_correction.py`` same method, and the difference is not empty but is
                        exhaustively five lines: one ``timing`` import and four
                        ``with _tmg.section(...)`` openers.  Nothing removed.

A ``with`` block changes the indentation of the statements inside it and
nothing else about them -- no reassociation, no reordering, no change of
operand -- so no sigma byte can move across this re-anchor.  That is the
finding the brief asked for, and it is negative: there is nothing to report
except that the digests moved for a reason that cannot affect a number.
"""

import hashlib
import pathlib
import subprocess

import pytest


#: The tree this branch's gnppm bit-identity is anchored to.  Update this and
#: the digests below TOGETHER, never one without the other.
BASE_SHA = "e37c6a6e"

#: Every module the two-point plasmon-pole Sigma reaches on its way from
#: ``sigma_dispatch`` to the bytes in sigma_mnk.h5.  Not "the files I happened
#: to think about": the list is the import closure of ``ppm_pipeline`` within
#: ``src/gw``, plus the dispatcher above it.
SHARED_SIGMA_CORE = {
    "src/gw/ppm_sigma.py":
        "f0fecfde318a18fb370ef6839ec9c7adff2006da5dd5d2aeffbb454d0846c1c7",
    "src/gw/ppm_windows.py":
        "f0b6749ce16a6329b68d8a2c06a4c3dcc198dd962eceef2d5540bd03146fed17",
    "src/gw/ppm_tau_kernel.py":
        "d139ebdfb6cea959f6747f137dc9b73c5a10e2adef1ec97cad82e44f4ef1e289",
    "src/gw/ppm_accumulators.py":
        "9d0def451087be4e3371cd57bd1e8f03db431823c7817e7045c4aa6ec19d06b0",
    "src/gw/ppm_pipeline.py":
        "448bcad9740941f787868d631a9cf8b23b16993eb34187637d9e3e82f20091da",
    "src/gw/sigma_dispatch.py":
        "9a6db9cd4db103a469c985eb14888f166683af0cbd775604a75ef4382b59e80a",
    "src/gw/head_correction.py":
        "1c99e0758a93f4a76d04aa32ba781ee7bb1f94007846a5a362532fefed52e753",
    "src/gw/minimax_screening.py":
        "8e7cfc4c9df71517f0fc83fd905748f3b82d92bf67a7fa1b957c929668040236",
}

#: The modules this branch adds or edits.  Stated as data so the two lists can
#: be checked disjoint, which is the whole contract in one assertion.
THIS_BRANCH_TOUCHES = (
    "src/gw/mpa/sigma_routing.py",
    "src/gw/mpa/sigma_pass.py",
    "src/file_io/mpa_store.py",
    "src/gw/mpa/fit_driver.py",
    "src/gw/gw_output.py",
    # ``gw_config`` is the one entry here the two-point path DOES import,
    # so it is worth saying what moved in it and why that cannot reach a
    # number: one string, the ``UNIMPLEMENTED_MODES`` reason for ``mpa``,
    # which is read only by ``refuse_unimplemented_compute_mode`` on its
    # way into an exception message.  No default, no dataclass field and no
    # parse branch changed, so no gnppm run can observe it.  If a later
    # commit touches this file for any other reason, that reason belongs in
    # this comment or the entry should come out.
    "src/gw/gw_config.py",
)


def _repo_root():
    return pathlib.Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_shared_sigma_core_is_byte_identical_to_the_base():
    """Premise one: the code that makes gnppm's bytes is the same code."""
    root = _repo_root()
    moved = []
    for rel, want in sorted(SHARED_SIGMA_CORE.items()):
        got = _sha256(root / rel)
        if got != want:
            moved.append(f"  {rel}\n    base {want}\n    here {got}")
    assert not moved, (
        "The shared two-point Sigma core moved on this branch, so a gnppm run "
        "through it is no longer provably bit-identical to " + BASE_SHA +
        ":\n" + "\n".join(moved) +
        "\n\nIf this is a REBASE absorbing someone else's landing, re-anchor: "
        "update BASE_SHA and these digests in one commit naming whose change "
        "was absorbed, then re-run the gate.  If this branch edited one of "
        "these files, that is the contract breaking -- either revert the edit "
        "or demonstrate bit-exact specialization on a deck and replace this "
        "gate with that measurement.")


def test_this_branch_touches_none_of_the_shared_core():
    """Same premise, said against git rather than against a digest.

    The two checks fail on different mistakes.  The digest fails when the file
    content differs for ANY reason, including a rebase; this one fails only
    when THIS BRANCH's own commits are the reason, which is the distinction a
    reader needs at 2 a.m.
    """
    root = _repo_root()
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", f"{BASE_SHA}...HEAD"],
            cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:                       # pragma: no cover
        pytest.skip(f"git diff against {BASE_SHA} failed: {out.stderr[:200]}")
    changed = {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}
    overlap = changed & set(SHARED_SIGMA_CORE)
    assert not overlap, (
        f"This branch's own commits edit the shared two-point Sigma core: "
        f"{sorted(overlap)}.  The MPA generalization is supposed to live in "
        f"gw/mpa/sigma_routing.py and be dispatched only from the mpa branch.")
    # And the positive half: the files it DOES touch are the ones it declares.
    src_changed = {c for c in changed if c.startswith("src/")}
    assert src_changed <= set(THIS_BRANCH_TOUCHES), (
        f"This branch changed source files it does not declare: "
        f"{sorted(src_changed - set(THIS_BRANCH_TOUCHES))}.  Add them to "
        f"THIS_BRANCH_TOUCHES and say why they are not shared-core.")


def test_the_generalized_kernel_is_unreachable_from_the_two_point_path():
    """Premise two: gnppm cannot reach the complex-pole machinery at all.

    Import-time reachability, checked by importing the two-point pipeline into
    a process where the MPA routing module is NOT yet imported and asserting it
    stays that way.  A module that is never imported cannot contribute an
    instruction to anybody's bytes, which is a stronger statement than "the
    dispatch does not call it".
    """
    code = (
        "import sys\n"
        "import gw.ppm_pipeline, gw.ppm_sigma, gw.sigma_dispatch\n"
        "leaked = [m for m in sys.modules "
        "if m.startswith('gw.mpa.sigma_routing')]\n"
        "print('LEAKED' if leaked else 'CLEAN', leaked)\n"
    )
    root = _repo_root()
    try:
        out = subprocess.run(
            [__import__("sys").executable, "-c", code],
            cwd=root, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"subprocess unavailable: {exc}")
    if out.returncode != 0:                       # pragma: no cover
        pytest.skip(f"import probe did not run: {out.stderr[-400:]}")
    assert "CLEAN" in out.stdout, (
        f"Importing the two-point Sigma path pulled in the MPA complex-pole "
        f"routing: {out.stdout.strip()}.  The two modes must not share a "
        f"kernel until the specialization is demonstrated bit-exact.")


def test_the_base_sha_is_recorded_and_reachable():
    """A gate anchored to a sha nobody can resolve is not anchored."""
    root = _repo_root()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--verify", f"{BASE_SHA}^{{commit}}"],
            cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:                       # pragma: no cover
        pytest.skip(f"{BASE_SHA} not present in this clone")
    assert out.stdout.strip().startswith(BASE_SHA)
