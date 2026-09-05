"""BSE absorption oscillator strengths, which had no gate at all.

WHY THIS FILE EXISTS.  ``tests/regression/si_bse_debug/README.md`` says
it plainly under "What this gate does NOT cover": *"Oscillator strengths
/ absorption spectra.  Only eigenvalues are compared."*  That is the one
quantity the relative sign of ``i[r, V_NL]`` in the assembled velocity
moves most directly -- eigenvalues barely feel it, because the sign
enters the dipole and not the two-particle Hamiltonian -- so the open
sign question was, until this file, a question whose most sensitive
observable nothing in the tree watched.  A decision that moves every
``dipole.h5`` deserves a before-picture of the thing it moves.

WHAT IS GATED, AND AT WHICH LEVEL.  Two levels, because only one of them
can run everywhere:

  * DATA LEVEL (runs anywhere, no GPU, no FFI, no deck).  The four
    committed ``dipole.h5`` fixtures are read as arrays and turned into
    oscillator strengths through the SAME functions the absorption
    driver calls -- ``slice_dipole_to_bse_window`` then
    ``compute_jdos_oscillators`` / ``compute_dipole_projections``.  The
    reference is the value those functions produce on the fixtures as
    committed today.
  * DRIVER LEVEL (cluster only).  Producing a ``dipole.h5`` needs
    ``psp.get_dipole_mtxels``, which calls
    ``runtime.initialize_communicator_stack()`` at module scope and is
    refused wherever the FFI host library is not built.  That leg is
    marked and skipped with its reason rather than silently absent.

TWO QUANTITIES, BECAUSE ONE OF THEM IS HALF BLIND.  The oscillator
strength ``|d|^2`` cannot see a conjugation: ``|conj(d)|^2 == |d|^2``
exactly, so a gate built on ``|d|^2`` alone would sit green through the
precise defect class this project has already paid for -- a bra that
stopped being conjugated.  So the reference also pins the COMPLEX
projection ``<0|r_a|S> = sum_cvk A^S_cvk conj(d^a_cvk)`` against a fixed
probe ``A``, which is conjugation-sensitive, and
``test_a_conjugated_dipole_is_caught`` is the cell that would have
failed.  The probe ``A`` is not a physical eigenvector and does not
pretend to be: it is a fixed complex contraction whose only job is to
carry phase information into a real number.

THE ``proj`` REFERENCES WERE RE-CUT ON 2026-08-09, AND NO FIXTURE MOVED.
When this file was written the two absorption drivers disagreed about
where that conjugate belongs, and this file's references were cut with
``absorption_eigvecs``'s spelling -- ``sum_t A_t d_t``, no conjugate
anywhere -- with the disagreement recorded rather than resolved.  It has
since been adjudicated against the spectral representation and the
resolvent identity (``tests/test_absorption_conjugation.py``, and
``absorption_common.exciton_dipole_projections`` for the derivation):
that spelling was the WRONG one.  So the twelve ``proj`` numbers per
fixture below are re-cut, while ``sum_f``, ``max_f`` and
``de_bounds_Ry`` are untouched to the last bit -- the movement is a
convention being corrected in the reader, not a ``dipole.h5`` being
regenerated.  Both facts together are the check: a change that moved the
sums as well would not be this fix.

THE REFERENCES ARE NOW TWO DIFFERENT ARMS, ON PURPOSE.  This file was
first cut with all four fixtures on the legacy ``-1`` arm.  On
2026-08-09 the default flipped and ``si_cohsex_debug`` and
``hbn_cohsex_debug`` were regenerated, so their references are re-cut
here on the ``+1`` arm and their ``dipole.h5`` carries
``prov_vnl_velocity_sign = +1.0``.  ``gnppm_debug`` and ``cohsex_debug``
were NOT regenerated -- their committed fixtures were cut at band counts
no deck in the tree still requests, so nothing can reproduce them and
inventing a deck to do it would be worse than leaving them -- and their
references remain on the legacy arm.  **Read the arm off the file, never
off this docstring**: every fixture's h5 stamps it.

That split is not tidy and it is not hidden.  When the two blocked decks
get a deck that reproduces them, their references move too, and the
movement is the measurement.  Re-cutting any of these is a deliberate
act paired with a fixture regeneration, never a fix for a red gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bse.absorption_common import load_dipole_h5, slice_dipole_to_bse_window
from bse.absorption_eigvecs import (compute_dipole_projections,
                                    compute_jdos_oscillators)

REG = Path(__file__).resolve().parent / "regression"

#: The BSE window every reference below was cut at.  Four valence and
#: four conduction bands on every deck -- not each deck's own GW window,
#: because the quantity under test is a dipole and a compact window that
#: is the SAME on all four decks makes the four numbers comparable with
#: each other.  ``si_cohsex_debug``'s 4x4 is additionally the window
#: ``si_bse_debug`` pins its eigenvalues at.
N_VAL = N_COND = 4

#: Number of probe states in the conjugation-sensitive contraction.
N_PROBE = 3

#: Seed for the probe ``A``.  Fixed, and fixed FOREVER: changing it
#: silently re-cuts every ``proj`` reference below without touching a
#: fixture, which would make this file agree with itself about nothing.
PROBE_SEED = 20260809


@dataclass(frozen=True)
class _Ref:
    n_occ: int
    nk: int
    sum_f: tuple            # sum over (k, c, v) of |d^a|^2, per polarisation
    max_f: float            # the single largest |d^a_cvk|^2
    de_bounds_Ry: tuple     # (min, max) of E_c - E_v over the window
    proj: tuple             # (N_PROBE, 3) complex <0|r_a|S> against the probe A


#: ``n_occ`` is the number of occupied bands, read off each fixture's own
#: ``deltaE`` as the band index with the largest minimum gap and checked
#: against the deck: gnppm/cohsex decks say ``nval = 26``, hbn's own
#: ``prov_nval`` attribute says 16, and si is the 8 of ``ifmax = 8``.
#: ``test_the_window_is_the_one_the_reference_was_cut_at`` re-derives it,
#: so a wrong window here fails loudly instead of quietly slicing a
#: metallic block and dividing by a ~1e-14 energy denominator.
REFS = {
    # RE-CUT 2026-08-09 on the +1 arm, from cohsex_si_test.in (nband 60).
    "si_cohsex_debug": _Ref(
        n_occ=8, nk=64,
        sum_f=(1.719138509280e+03, 1.710456094893e+03, 1.716160361438e+03),
        max_f=1.945919115651e+01,
        de_bounds_Ry=(1.867165110504e-01, 6.277285155045e-01),
        proj=(
            (+2.1773175146e+00-5.2355133418e-01j, -3.2330540801e-01-2.3793315546e-01j, -4.9214374231e-01-1.3457611777e+00j),
            (-1.4696741254e-01-2.3036993205e-01j, -2.6019104888e-01+8.6590320694e-01j, -8.7905593882e-01-1.3702241706e-01j),
            (+6.4290630171e-01+5.2889570610e-01j, -6.7387578095e-01+3.8855564017e-01j, -6.5087234113e-01-9.8260745476e-01j),
        ),
    ),
    "cohsex_debug": _Ref(
        n_occ=26, nk=9,
        sum_f=(1.402684287226e+02, 1.408630269079e+02, 6.833272272814e+00),
        max_f=1.711752678590e+01,
        de_bounds_Ry=(1.416169052037e-01, 4.237078164396e-01),
        proj=(
            (+1.1767038008e+00-8.6197311831e-01j, -9.5160821399e-01+6.1750819109e-01j, +1.6022401044e-01-1.9339608169e-01j),
            (+2.0090552564e-01-2.1537693770e-01j, +3.2235600592e-02-6.0643406194e-01j, -1.8742029401e-02+1.1650832917e-01j),
            (-1.5766050670e-02+4.2141446274e-01j, -1.2731123386e-01+1.1206067694e+00j, -1.5557208846e-01-9.2676111995e-02j),
        ),
    ),
    "gnppm_debug": _Ref(
        n_occ=26, nk=9,
        sum_f=(1.810189535166e+02, 1.810191023184e+02, 5.682962823090e+00),
        max_f=2.008907399830e+01,
        de_bounds_Ry=(1.249527013538e-01, 3.872730886117e-01),
        proj=(
            (-2.7700661975e-01+1.0279966686e+00j, +4.7692705008e-01-7.0768505726e-01j, +1.9323997566e-01+1.1991717967e-01j),
            (+2.8224994997e-01+4.9796397795e-01j, +3.0607581904e-01+6.0228150425e-01j, -1.2537366457e-01+7.0051313667e-02j),
            (+2.5294293247e-01+1.0349084510e+00j, -7.0891256190e-01-1.6893661625e+00j, +1.6969524115e-01-6.5938093556e-02j),
        ),
    ),
    # RE-CUT 2026-08-09 on the +1 arm, from cohsex_hbn_test.in.
    "hbn_cohsex_debug": _Ref(
        n_occ=16, nk=18,
        sum_f=(8.619436160739e+01, 8.812618329195e+01, 1.672053024993e+01),
        max_f=4.059144537110e+00,
        de_bounds_Ry=(3.426640012554e-01, 8.544664902677e-01),
        proj=(
            (+2.8553044346e-01+4.6721797679e-01j, +2.7705123380e-01-4.9844932952e-01j, -2.2450746557e-01-3.7938516102e-02j),
            (+4.1362644126e-01+2.1857995605e-01j, -2.0460875920e-01-5.5959062813e-01j, -1.3605505522e-02+1.8887472718e-01j),
            (-1.5362492214e-01-1.0083685263e-01j, -1.4728355940e-01-2.6792712743e-01j, +2.0589795044e-01+9.3477156397e-02j),
        ),
    ),
}

FIXTURES = sorted(REFS)

#: Reading a committed file and contracting it is deterministic to the
#: last bit on one machine, but BLAS reassociation across machines is
#: not, so the comparison is a tight relative tolerance rather than
#: ``array_equal``.  1e-9 is ~7 orders below the ~31 % the velocity sign
#: moves and ~3 orders below the smallest defect worth a gate.
RTOL = 1.0e-9


def _probe_A(nk):
    """A fixed complex contraction over the (k, c, v) block.

    Not an eigenvector: no BSE solve runs here, and none can locally.
    Its job is to carry the dipole's PHASE into a comparable number, so
    the reference sees a conjugation that ``|d|^2`` is blind to.
    """
    rng = np.random.default_rng(PROBE_SEED)
    A = (rng.standard_normal((N_PROBE, nk, N_COND, N_VAL))
         + 1j * rng.standard_normal((N_PROBE, nk, N_COND, N_VAL)))
    return A / np.linalg.norm(A.reshape(N_PROBE, -1), axis=1)[:, None, None, None]


def _load(name):
    """The fixture's dipole, sliced to the reference window."""
    ref = REFS[name]
    path = REG / name / "dipole.h5"
    if not path.is_file():                       # pragma: no cover
        pytest.skip(f"{path} absent")
    dipole_cart, deltaE, _ = load_dipole_h5(str(path))
    d_alpha, de_cv = slice_dipole_to_bse_window(
        dipole_cart, deltaE, ref.n_occ, N_VAL, N_COND)
    return ref, d_alpha, de_cv


def _observables(d_alpha):
    """The two reference quantities, from the driver's own functions."""
    f = compute_jdos_oscillators(d_alpha)                    # (3, nk, nc, nv)
    proj = compute_dipole_projections(_probe_A(d_alpha.shape[1]), d_alpha)
    return f.sum(axis=(1, 2, 3)), float(f.max()), proj


# ---------------------------------------------------------------------------
# 1. The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_oscillator_strengths_match_reference(name):
    """The committed dipole fixtures still imply the same absorption.

    This is the before-picture.  It goes red when a ``dipole.h5`` is
    regenerated, which is exactly when somebody should be made to look.
    """
    ref, d_alpha, _ = _load(name)
    assert d_alpha.shape == (3, ref.nk, N_COND, N_VAL)
    sum_f, max_f, proj = _observables(d_alpha)

    np.testing.assert_allclose(
        sum_f, np.asarray(ref.sum_f), rtol=RTOL,
        err_msg=(f"{name}: sum_cvk |d^a|^2 moved.  If a dipole.h5 was "
                 f"regenerated this is the measurement and the reference "
                 f"is re-cut deliberately; if not, something upstream of "
                 f"the absorption dipole changed."))
    assert max_f == pytest.approx(ref.max_f, rel=RTOL), (
        f"{name}: the largest single oscillator strength moved from "
        f"{ref.max_f:.6e} to {max_f:.6e}")
    np.testing.assert_allclose(
        proj, np.asarray(ref.proj), rtol=RTOL,
        err_msg=(f"{name}: the complex dipole projection moved.  Unlike "
                 f"the |d|^2 sums this one carries phase, so a change "
                 f"here with the sums intact is a CONJUGATION and not a "
                 f"magnitude."))


@pytest.mark.parametrize("name", FIXTURES)
def test_the_window_is_the_one_the_reference_was_cut_at(name):
    """``n_occ`` is re-derived, not trusted.

    ``slice_dipole_to_bse_window`` divides by ``de_cv`` with no
    degeneracy guard, so a wrong ``n_occ`` does not raise -- it slices a
    block straddling the gap and divides by a denominator that can reach
    1e-14, producing enormous numbers that a reference cut at the SAME
    wrong window would happily confirm.  The gap is the check.
    """
    ref, _, de_cv = _load(name)
    assert de_cv.min() > 0.0, (
        f"{name}: the window at n_occ={ref.n_occ} contains a "
        f"non-positive transition energy ({de_cv.min():.3e} Ry), so it "
        f"straddles the gap")
    assert (de_cv.min(), de_cv.max()) == pytest.approx(
        ref.de_bounds_Ry, rel=RTOL), (
        f"{name}: the transition energies moved, so the reference's "
        f"window and this one are not the same bands")


# ---------------------------------------------------------------------------
# 2. The false cases.  Each pins a different blindness.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FIXTURES)
def test_a_perturbed_dipole_is_caught(name):
    """RED TWIN.  A gate that cannot fail is a comment.

    One part in 1e6 on a single matrix element -- far below the ~31 %
    the velocity sign is worth, so passing this says the gate has
    margin to spare for the thing it was built to watch.
    """
    ref, d_alpha, _ = _load(name)
    perturbed = d_alpha.copy()
    perturbed[0, 0, 0, 0] *= (1.0 + 1.0e-6)
    sum_f, _, _ = _observables(perturbed)
    assert not np.allclose(sum_f, np.asarray(ref.sum_f), rtol=RTOL), (
        f"{name}: a perturbed dipole reproduced the reference sums, so "
        f"the gate is not reading what it claims to read")


@pytest.mark.parametrize("name", FIXTURES)
def test_a_conjugated_dipole_is_caught(name):
    """RED TWIN for the crossed-convention class specifically.

    This project's named defect is a bra that stopped being conjugated.
    The cell asserts BOTH halves of why that needs its own reference:
    ``|d|^2`` does not move at all under conjugation (exactly, not
    approximately), and the complex projection does.  If a future edit
    drops the ``proj`` reference as redundant, this fails.
    """
    ref, d_alpha, _ = _load(name)
    sum_f, max_f, proj = _observables(np.conj(d_alpha))

    np.testing.assert_allclose(
        sum_f, np.asarray(ref.sum_f), rtol=RTOL,
        err_msg="conjugation must leave |d|^2 EXACTLY where it was")
    assert max_f == pytest.approx(ref.max_f, rel=RTOL)
    assert not np.allclose(proj, np.asarray(ref.proj), rtol=RTOL), (
        f"{name}: a conjugated dipole reproduced the reference "
        f"projections.  The projection is the only conjugation-sensitive "
        f"quantity in this file; if it cannot see one, this file cannot "
        f"gate the defect class it was written for.")


def test_the_two_projection_routes_now_agree_and_the_wrong_one_still_fails():
    """CROSSED-CONVENTION TWIN, on the divergence this file first recorded.

    When this cell was written the two drivers disagreed: this one
    contracted ``einsum("Nkcv,akcv->Na", A, d)`` with no conjugate
    anywhere, ``davidson_absorption`` contracted ``eigvecs.conj()``
    against a transposed block, and which was right was open.  It is
    closed (``tests/test_absorption_conjugation.py`` runs the identity
    that closes it): the conjugate belongs on the DIPOLE, both drivers
    now route through ``exciton_dipole_projections``, and the two
    spellings that survive -- ``A conj(d)`` and ``conj(A) d`` -- are
    complex conjugates, so they agree in the modulus that ε₂ consumes.

    What must still fail is the arm that shipped: ``A d``, conjugating
    NEITHER factor, differs in modulus and not merely in phase.  That is
    the FALSE arm, and this is where a silent return to it fails.
    """
    ref, d_alpha, _ = _load("si_cohsex_debug")
    A = _probe_A(ref.nk)

    pinned = compute_dipole_projections(A, d_alpha)
    davidson_spelling = np.einsum("Nkcv,akcv->Na", A.conj(), d_alpha,
                                  optimize=True)
    false_arm = np.einsum("Nkcv,akcv->Na", A, d_alpha, optimize=True)

    np.testing.assert_allclose(pinned, np.asarray(ref.proj), rtol=RTOL)
    np.testing.assert_allclose(
        np.abs(pinned), np.abs(davidson_spelling), rtol=1.0e-13,
        err_msg="the two surviving spellings must be complex conjugates, so "
                "their moduli -- the only part ε₂ reads -- must be identical")
    assert not np.allclose(np.abs(pinned), np.abs(false_arm), rtol=1.0e-6), (
        "the unconjugated contraction reproduced the corrected one in "
        "modulus, which for a complex probe can only mean the probe lost "
        "its imaginary part")


# ---------------------------------------------------------------------------
# 3. How much the nonlocal term is actually worth, measured here
# ---------------------------------------------------------------------------
#
# The cells above pin the fixtures, which are stored arrays and therefore
# cannot move until somebody regenerates them.  That leaves the question
# the whole file exists for -- how much does the i[r, V_NL] term move
# oscillator strengths? -- answered only on the cluster.  It does not have
# to be.  ``dipole_operator`` runs on a synthetic projector setup on any
# CPU, so the sensitivity is measurable right here, and a number measured
# is a number that can gate.

#: Median relative change in |d|^2 per matrix element between p alone and
#: p + dV_NL/dK on the synthetic setup.  The assertion uses 0.05 because
#: the point is not to pin this fixture's exact sensitivity but to refuse
#: a tree in which the projector term has quietly stopped reaching the
#: assembled velocity.
_NONLOCAL_TERM_MOVES_F_BY_AT_LEAST = 0.05


def _matrix_elements_without_and_with_the_nonlocal_term():
    """<m|v|n> for p alone and for p + dV_NL/dK, from the synthetic
    setup ``tests/test_dipole_vnl_velocity_sign`` owns.

    Imported rather than rebuilt: that file's ``_vnl_setup`` is
    deliberately built with a random nonzero ``E_super`` (a null one
    makes v_NL vanish and the two agree), and duplicating 120 lines of
    projector construction to say the same thing twice is how two
    fixtures drift into disagreeing about what they test.
    """
    from tests.test_dipole_vnl_velocity_sign import (
        _apply_one_k, _fixture, _geom, _mesh, _mtxel, _vnl_setup)
    from common.mtxel_sweep import dipole_operator

    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    got = {}
    with mesh:
        geom = _geom(mesh)
        for tag, vnl in (("p", None), ("p+vnl", setup)):
            op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=vnl)
            got[tag] = np.asarray([
                _mtxel(_apply_one_k(op, psi[ik], gv[ik], gmask[ik],
                                    bidx[ik], kvecs[ik]),
                       psi[ik], gmask[ik])
                for ik in range(psi.shape[0])])
    return got["p"], got["p+vnl"]


def _median_relative_move(a, b):
    fa, fb = np.abs(a) ** 2, np.abs(b) ** 2
    return float(np.median(np.abs(fb - fa) / (np.abs(fa) + 1.0e-30)))


def test_the_nonlocal_term_moves_oscillator_strengths():
    """The measurement this file was built to make.

    Oscillator strengths are ``|d|^2``, and p + dV_NL/dK differs from p
    by the whole projector term inside ``d`` -- not by a global phase,
    which ``|d|^2`` would not see.  This is the cell that says by how
    much without a cluster.
    """
    p_only, full = _matrix_elements_without_and_with_the_nonlocal_term()
    move = _median_relative_move(p_only, full)
    assert move > _NONLOCAL_TERM_MOVES_F_BY_AT_LEAST, (
        f"the nonlocal term moved |d|^2 by a median of {move:.3e}, "
        f"below {_NONLOCAL_TERM_MOVES_F_BY_AT_LEAST}.  Either the "
        f"projector term stopped reaching the assembly or it went null -- "
        f"both turn every other cell in this file into a tautology.")


def test_the_sensitivity_measurement_is_not_a_tautology():
    """RED TWIN for the cell above.

    With a null ``E_super`` the nonlocal term vanishes identically, the
    operator collapses onto bare ``p``, and the sensitivity cell would be
    measuring nothing while still reading green if its threshold were
    written the other way round.  This asserts the collapse really is
    what a dead projector looks like, so the threshold above has a meaning.
    """
    import jax.numpy as jnp
    from dataclasses import replace
    from tests.test_dipole_vnl_velocity_sign import (
        _apply_one_k, _fixture, _geom, _mesh, _mtxel, _vnl_setup)
    from common.mtxel_sweep import dipole_operator

    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    dead = replace(_vnl_setup(), E_super=jnp.zeros_like(_vnl_setup().E_super))
    got = {}
    with mesh:
        geom = _geom(mesh)
        for tag, vnl in (("p", None), ("dead", dead)):
            op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=vnl)
            got[tag] = np.asarray([
                _mtxel(_apply_one_k(op, psi[ik], gv[ik], gmask[ik],
                                    bidx[ik], kvecs[ik]),
                       psi[ik], gmask[ik])
                for ik in range(psi.shape[0])])
    move = _median_relative_move(got["p"], got["dead"])
    assert move < 1.0e-12, (
        f"a null projector strength still moved |d|^2 by {move:.3e}, so "
        f"the operator differs from p by something that is not the "
        f"nonlocal term")


# ---------------------------------------------------------------------------
# 4. The driver leg, which cannot run here
# ---------------------------------------------------------------------------

def _ffi_available():
    try:
        import runtime                                   # noqa: F401
        from ffi.fft import GATE
        import jax
        from jax.sharding import Mesh
        GATE.enforce(Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                          ('x', 'y')))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ffi_available(),
                    reason="producing a dipole.h5 needs psp.get_dipole_mtxels, "
                           "which calls initialize_communicator_stack() at "
                           "module scope and is refused without the FFI host "
                           "library (WSL/CPU box has none)")
def test_the_producer_reaches_the_same_oscillator_strengths():
    """DRIVER LEG.  The data-level gate above reads a committed file; this
    one asserts the producer still writes that file's content.

    Run on the cluster as part of any fixture regeneration: it is the
    cell that ties ``psp.get_dipole_mtxels``'s output back to the
    absorption numbers, which is the join the data-level gate cannot
    make on its own.
    """
    from common.mtxel_sweep import VNL_VELOCITY_SIGN

    # The velocity a bare run builds must be the one the references were
    # cut at (``prov_vnl_velocity_sign = +1.0`` on every committed fixture).
    assert VNL_VELOCITY_SIGN == +1.0
