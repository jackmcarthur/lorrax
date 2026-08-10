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

THE SECOND RE-ANCHOR, 2026-08-09, AND IT IS A DIFFERENT KIND -- read this
before trusting the shape of the gate.  The MPA DRIVER SEAM
(``feat/mpa-driver-seam-2026-08-09``) had to edit two of the eight modules
below, deliberately and as part of its brief, which is exactly the case the
paragraph above called "the contract breaking".  It is registered here
rather than papered over, and the gate is RESHAPED rather than relaxed:

* the six KERNELS -- ``ppm_sigma``, ``ppm_windows``, ``ppm_tau_kernel``,
  ``ppm_accumulators``, ``head_correction``, ``minimax_screening`` -- are
  where every floating-point operation of a gnppm run happens, and NOTHING
  may touch them.  They keep both checks, the digest and the git one, and
  :data:`SHARED_SIGMA_KERNELS` is that list.
* the two SEAMS -- ``ppm_pipeline`` and ``sigma_dispatch`` -- are
  orchestration: they sequence the kernels and assemble the result.  A
  second Sigma scheme cannot be dispatched without editing the dispatcher.
  They move to :data:`SHARED_SIGMA_SEAMS`, whose digests are pinned to THIS
  branch's content, so a further edit to either still fails this gate; what
  is no longer asserted is that the branch never touched them, because it
  did.

WHY NO GNPPM BYTE MOVES ACROSS IT, checked the same way the first
re-anchor's three files were:

  ``sigma_dispatch.py``  Two changes.  (1) A branch ``if mode is
                         ComputeMode.MPA:`` above the pole-model guard --
                         an identity test on an enum member, which a gnppm
                         run fails, so not one statement inside it is
                         evaluated on that path.  (2) The QSGW tail (the
                         Hermitisation, the optional cube write, the
                         ``SigmaResult`` assembly) moved verbatim into
                         ``_finish_dynamic_sigma`` and is called with the
                         same operands in the same order.  Relocating a
                         statement into a function changes the frame it
                         runs in and nothing about the arithmetic.
  ``ppm_pipeline.py``    Three changes.  (1) The band-diagonal head
                         injection -- ``_embed_dense`` plus the
                         sharded/dense branch -- moved verbatim into
                         ``qsgw_utils.add_band_diag``, which both heads now
                         call; the dense arm builds the same zero-filled
                         cube from the same three lines and performs the
                         same ``+``, and the sharded arm forwards the same
                         array to the same ``add_band_diag_sharded`` (the
                         added ``np.asarray`` is a no-op on an array that
                         is already one, returning the same object).
                         (2) The now-unused ``jax.numpy`` import went.
                         (3) ``PPMOutputs``' docstring widened.

``qsgw_utils`` is where that injection landed and it was NEVER in this
list, which is itself worth naming: the list claims to be "the import
closure of ``ppm_pipeline`` within ``src/gw``" and ``qsgw_utils`` is in
that closure (through function-local imports in three places).  So the
gate's coverage had a hole before this branch and still has it; closing it
means adding ``qsgw_utils`` here, which is a decision for whoever owns this
contract, not one to take while moving code into the file it does not
cover.  Registered, not fixed.

THE THIRD RE-ANCHOR, 2026-08-09 (later the same day), AND THIS ONE MOVES A
NUMBER ON PURPOSE.  ``fix/ppm-crossing-operator-im-2026-08-09`` @ 2fd22005,
merged as 2115b65a, edits ``ppm_accumulators.py`` -- a KERNEL, the exact
case the gate exists to make loud -- because the crossing window's tau-grid
completion was the ELEMENTWISE imaginary part where the sine sum needs the
operator one, ``(c*X - (c*X)^dagger)/2i``.  That is a physics fix,
adjudicated in numbers before this re-anchor absorbed it
(tests/KNOWN_FAILURES.md, 2026-08-09 amendment): Sigma_c star spread
43.85 -> 0.0000 eV, the five non-TRIM k's non-Hermiticity 8.8-30.3 eV
-> 0, |eqp0 - BGW GN twin| mean 5.03 -> 0.44 eV and max 20.25 -> 1.05 eV,
the three TRIM rows unchanged to 2.6e-7 eV, sigX bit-identical.  THE
INTER-FLEET CONSEQUENCE IS REAL: the peer fleet's gnppm sigTOT anchor
(0.6439 meV MAE vs the BGW oracle) was anchored to the ELEMENTWISE bytes
and is therefore SUPERSEDED, not drifted -- whoever quotes it next
re-measures on a tree containing 2fd22005, expecting the non-TRIM k to
move by the adjudicated amounts and TRIM to stay.  The gnppm/bispinor
frozen references are red for the same reason, and their re-freeze is an
OWNER row (adjudicated, registered in KNOWN_FAILURES.md), deliberately
not taken with this re-anchor.

THE FOURTH RE-ANCHOR, 2026-08-10, AND IT WITHDRAWS THE THIRD'S PLACEMENT.
This branch merged ``origin/main`` @ 965d7beb, which carries ``c80601b8``
-- the same crossing completion, derived independently, with the PLACEMENT
corrected.  The third re-anchor's ``2fd22005`` took the adjoint per-tau
INSIDE ``_project_tau_onto_omega_np``, i.e. inside the per-shard
projector; ``(Z - Z^dagger)`` pairs band element (i,j) with (j,i) and
Sigma_c tiles are sharded over the band axes, so that placement is the
band adjoint only when the mesh does not cut the band window -- true on
the -G=1 leg every measurement above was taken on, and silently wrong at
P>1 (0.132-0.257 eV Hermiticity violation at ALL EIGHT k, breaking the
three TRIM k the pre-fix code got exactly right).  ``c80601b8`` makes the
completion a WINDOW-level operator instead: the consumer returns the
one-sided half and ``_TauAccumulator._finish_window`` closes the window.
The ALGEBRA of the third re-anchor is unchanged and so are all of its
adjudicated numbers -- they were measured single-shard, where the two
placements agree -- so nothing quoted above is withdrawn except the
placement itself.  Four kernels move here (``ppm_accumulators``,
``ppm_sigma``, ``ppm_tau_kernel``, ``ppm_windows``) and ALL SIX kernels
are now byte-identical to ``origin/main`` @ 965d7beb, which is what this
BASE_SHA now names; the two SEAMS keep their branch-pinned digests.  The
gnppm/bispinor frozen references named above are no longer red: main
re-froze them at ``1e64d83a`` against the corrected form.
"""

import hashlib
import pathlib
import subprocess

import pytest


#: The tree this branch's gnppm bit-identity is anchored to.  Update this and
#: the digests below TOGETHER, never one without the other.
BASE_SHA = "965d7beb"

#: THE KERNELS: every module in which a floating-point operation of a gnppm
#: Sigma actually happens.  Nothing on this branch may touch them, and both
#: checks below apply -- the digest and the git one.
SHARED_SIGMA_KERNELS = {
    "src/gw/ppm_sigma.py":
        "f255ef2944808d114574101a8c45aff44e14ae9bc1583efe376715b34181b87e",
    "src/gw/ppm_windows.py":
        "0290ba57190e3c617b7d3f568fc5024aaf8a6bfb152d3adc807af4668800e7be",
    "src/gw/ppm_tau_kernel.py":
        "583425f1f5f06690ee050d56dc494c1fbed2cd07ef3cf9d134047bce8fa0c700",
    "src/gw/ppm_accumulators.py":
        "f75a91503834fe1a466d9441ee58ffa92f3a7f34bf47fef7863efe32ef9835fd",
    "src/gw/head_correction.py":
        "1c99e0758a93f4a76d04aa32ba781ee7bb1f94007846a5a362532fefed52e753",
    "src/gw/minimax_screening.py":
        "8e7cfc4c9df71517f0fc83fd905748f3b82d92bf67a7fa1b957c929668040236",
}

#: THE SEAMS: the two orchestration modules the MPA driver seam had to edit,
#: because a second Sigma scheme cannot be dispatched without editing the
#: dispatcher.  Digests pinned to THIS branch, so a further edit still fails
#: the gate; see the module docstring for the line-by-line account of why no
#: gnppm byte moves across the edits that are already in them.
SHARED_SIGMA_SEAMS = {
    "src/gw/ppm_pipeline.py":
        "d70aca0ba75095ce4418c64caa4f53882c2572d25b09ebc353da308a44a2292a",
    "src/gw/sigma_dispatch.py":
        "d212a1138b9f6408dd9aee918df4dc93d82f81fb65ae8de521b9e44b218ea4e2",
}

#: Both, for the callers that want the whole set.
SHARED_SIGMA_CORE = {**SHARED_SIGMA_KERNELS, **SHARED_SIGMA_SEAMS}

#: The modules this branch adds or edits.  Stated as data so the kernel list
#: and this one can be checked disjoint, which is what is left of the
#: original one-assertion contract.
THIS_BRANCH_TOUCHES = (
    "src/gw/mpa/sigma_routing.py",
    "src/gw/mpa/sigma_pass.py",
    "src/gw/mpa/sigma_head.py",
    "src/gw/mpa/head_dipole.py",
    "src/gw/mpa_pipeline.py",
    "src/file_io/mpa_store.py",
    "src/file_io/paths.py",
    "src/gw/mpa/fit_driver.py",
    "src/gw/gw_output.py",
    "src/common/mtxel_sweep.py",
    "src/psp/get_dipole_mtxels.py",
    # The three the two-point path DOES import.  Each gets its reason here,
    # and a later commit touching one for any other reason owes an update
    # to this comment or an entry removed.
    #   ``gw_config``     -- the compute-mode axis, the channel table and
    #                        three deck keys (mpa_fit_file,
    #                        mpa_pole_energy_unit, mpa_head_label).  No
    #                        existing default, dataclass field or parse
    #                        branch changed value, so no gnppm run observes
    #                        any of it.
    #   ``screening``     -- one branch, for a mode gnppm is not.
    #   ``qsgw_utils``    -- gained ``add_band_diag``, into which the head
    #                        injection moved verbatim from ppm_pipeline.
    #                        See the docstring: this module is in the
    #                        two-point path's import closure and was never
    #                        in the digest list, which is a coverage hole
    #                        this branch registers and does not close.
    #   ``gw_jax``        -- the driver: one entry refusal and a guard on
    #                        the W0 restart flush for a mode that requests
    #                        no static W.
    "src/gw/gw_config.py",
    "src/gw/screening.py",
    "src/gw/qsgw_utils.py",
    "src/gw/gw_jax.py",
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


def test_this_branch_touches_none_of_the_sigma_kernels():
    """Same premise, said against git rather than against a digest.

    The two checks fail on different mistakes.  The digest fails when the file
    content differs for ANY reason, including a rebase; this one fails only
    when THIS BRANCH's own commits are the reason, which is the distinction a
    reader needs at 2 a.m.

    NARROWED TO THE KERNELS on 2026-08-09, and the module docstring carries
    the account.  The two SEAMS are edited by this branch on purpose -- a
    second Sigma scheme cannot be dispatched without editing the dispatcher
    -- so asserting they were not would be asserting something false.  What
    is still asserted, and is the part that carries the contract, is that no
    module in which a gnppm floating-point operation happens was touched.
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
    overlap = changed & set(SHARED_SIGMA_KERNELS)
    assert not overlap, (
        f"This branch's own commits edit a two-point Sigma KERNEL: "
        f"{sorted(overlap)}.  The MPA generalization is supposed to live in "
        f"gw/mpa/ and be dispatched only from the mpa branch; the seams "
        f"({sorted(SHARED_SIGMA_SEAMS)}) are the only shared modules it may "
        f"edit, and each edit owes an entry in this file's docstring.")
    # And the positive half: the files it DOES touch are the ones it
    # declares, plus the two seams, whose edits are accounted for one by one
    # in this file's docstring and whose content is pinned by the digest
    # check above.
    src_changed = {c for c in changed
                   if c.startswith("src/") and c not in SHARED_SIGMA_SEAMS}
    assert src_changed <= set(THIS_BRANCH_TOUCHES), (
        f"This branch changed source files it does not declare: "
        f"{sorted(src_changed - set(THIS_BRANCH_TOUCHES))}.  Add them to "
        f"THIS_BRANCH_TOUCHES and say why they are not a Sigma kernel.")
    assert not (set(THIS_BRANCH_TOUCHES) & set(SHARED_SIGMA_KERNELS)), (
        "a module cannot be both a declared edit of this branch and a "
        "two-point Sigma kernel")


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
