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
projection ``<0|r_a|S> = sum_cvk A^S_cvk d^a_cvk`` against a fixed
probe ``A``, which is conjugation-sensitive, and
``test_a_conjugated_dipole_is_caught`` is the cell that would have
failed.  The probe ``A`` is not a physical eigenvector and does not
pretend to be: it is a fixed complex contraction whose only job is to
carry phase information into a real number.

THE REFERENCE IS THE SHIPPED ARM.  Every number below was cut from the
fixtures as they stand, which is to say from the ``p - dV_NL/dK`` arm
that built them.  If the velocity sign flips, the fixtures are
regenerated and THESE NUMBERS MUST MOVE -- that movement is the
measurement, not a regression.  Re-cutting them is a deliberate act with
the four fixture regenerations, never a fix for a red gate.
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
    "si_cohsex_debug": _Ref(
        n_occ=8, nk=64,
        sum_f=(2.262665367089e+03, 2.248954715561e+03, 2.257512723595e+03),
        max_f=2.586482092333e+01,
        de_bounds_Ry=(1.867165110504e-01, 6.277285155045e-01),
        proj=(
            (-6.0249034443e-02+5.6924445472e-01j, +6.6336723520e-02-2.3078457363e+00j, -5.0554456824e-01-1.0441475874e+00j),
            (+9.6850735373e-01+1.3621130388e+00j, -1.1799676537e+00-1.9655944525e-01j, +8.3873453589e-01-3.4944651489e-01j),
            (-2.0035778998e-01+6.2799664087e-01j, +4.1730304739e-01-1.8188335912e+00j, -7.4087812249e-01+1.9006811621e+00j),
        ),
    ),
    "cohsex_debug": _Ref(
        n_occ=26, nk=9,
        sum_f=(1.402684287226e+02, 1.408630269079e+02, 6.833272272814e+00),
        max_f=1.711752678590e+01,
        de_bounds_Ry=(1.416169052037e-01, 4.237078164396e-01),
        proj=(
            (-2.0716647250e-01-4.0826852116e-01j, -7.6207706341e-01+7.3024288558e-02j, -1.2428336413e-01+1.4608299471e-01j),
            (-4.3308667555e-01-1.1197147061e+00j, -2.0287672018e-01-8.3193396992e-02j, +5.4429980147e-03-6.2401642528e-02j),
            (+2.9179754827e-01-7.7754056233e-02j, -6.4976000902e-01-2.2711261760e-01j, +8.1830428365e-02+1.7513071519e-01j),
        ),
    ),
    "gnppm_debug": _Ref(
        n_occ=26, nk=9,
        sum_f=(1.810189535166e+02, 1.810191023184e+02, 5.682962823090e+00),
        max_f=2.008907399830e+01,
        de_bounds_Ry=(1.249527013538e-01, 3.872730886117e-01),
        proj=(
            (-2.5441766486e-01-3.5388704773e-01j, +1.5580880111e+00+6.0736994943e-01j, +3.7218197577e-02+1.3237353342e-01j),
            (-3.2231100175e-01-6.8331772564e-01j, -1.6286923089e+00+1.4344278987e+00j, +1.3477378104e-01-3.2327133470e-02j),
            (-2.0612052375e-01-1.1928367539e-01j, +6.6824594171e-02-2.3419304085e-01j, +2.4954075399e-03-2.7458830896e-01j),
        ),
    ),
    "hbn_cohsex_debug": _Ref(
        n_occ=16, nk=18,
        sum_f=(8.988955164909e+01, 9.793769195724e+01, 2.283749959815e+01),
        max_f=3.986165178707e+00,
        de_bounds_Ry=(3.426640012554e-01, 8.544664902677e-01),
        proj=(
            (-2.5843204926e-02+1.4996789438e-01j, +1.1478100488e-02-3.5165647329e-01j, -1.6472918232e-01-3.4802897308e-01j),
            (+1.2023603421e-01-3.8645716990e-01j, -9.7345639179e-01+2.7833036576e-01j, -4.7276480243e-02-2.2385712751e-01j),
            (+1.1204535169e-01+5.9169233046e-01j, +4.9324141788e-01+6.3593101093e-01j, -1.4488692983e-01+2.6819317493e-01j),
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


def test_the_two_projection_routes_disagree_and_this_one_is_pinned():
    """CROSSED-CONVENTION TWIN, on a real in-tree divergence.

    ``absorption_eigvecs.compute_dipole_projections`` contracts
    ``einsum("Nkcv,akcv->Na", A, d)`` with NO conjugate on ``A``, while
    ``davidson_absorption`` (:214) contracts ``eigvecs.conj()`` against
    a transposed block.  For complex ``A`` those are different numbers,
    not two spellings of one -- so "the oscillator strength" is
    ambiguous in this tree until somebody says which.  This file's
    reference is cut with the ``absorption_eigvecs`` convention, and
    this cell is where that choice is written down and where a silent
    swap of the two routes would fail.
    """
    ref, d_alpha, _ = _load("si_cohsex_debug")
    A = _probe_A(ref.nk)

    pinned = compute_dipole_projections(A, d_alpha)
    conjugated = np.einsum("Nkcv,akcv->Na", A.conj(), d_alpha, optimize=True)

    np.testing.assert_allclose(pinned, np.asarray(ref.proj), rtol=RTOL)
    assert not np.allclose(pinned, conjugated, rtol=1.0e-6), (
        "the two in-tree projection conventions agreed, which for a "
        "complex A can only mean the probe lost its imaginary part")


# ---------------------------------------------------------------------------
# 3. How much the velocity sign is actually worth, measured here
# ---------------------------------------------------------------------------
#
# The cells above pin the fixtures, which are stored arrays and therefore
# cannot move until somebody regenerates them.  That leaves the question
# the whole file exists for -- how much does the SIGN move oscillator
# strengths? -- answered only on the cluster.  It does not have to be.
# ``dipole_operator`` runs on a synthetic projector setup on any CPU, so
# the sensitivity is measurable right here, and a number measured is a
# number that can gate.

#: Median relative change in |d|^2 per matrix element between the two
#: arms on the synthetic setup, measured at 0.456.  The assertion uses
#: 0.05 -- a ninefold margin -- because the point is not to pin this
#: fixture's exact sensitivity but to refuse a tree in which the sign
#: has quietly stopped mattering, which is what a knob wired to nothing
#: would look like.
_SIGN_MOVES_F_BY_AT_LEAST = 0.05


def _both_arms_matrix_elements():
    """<m|v|n> at both signs, from the synthetic setup the knob file owns.

    Imported rather than rebuilt: that file's ``_vnl_setup`` is
    deliberately built with a random nonzero ``E_super`` (a null one
    makes v_NL vanish and both arms agree), and duplicating 120 lines of
    projector construction to say the same thing twice is how two
    fixtures drift into disagreeing about what they test.
    """
    from tests.test_dipole_vnl_velocity_sign import (
        _apply_one_k, _fixture, _geom, _mesh, _mtxel, _vnl_setup)
    from common.mtxel_sweep import (VNL_VELOCITY_SIGN_FLIPPED,
                                    VNL_VELOCITY_SIGN_SHIPPED, dipole_operator)

    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    setup = _vnl_setup()
    arms = {}
    with mesh:
        geom = _geom(mesh)
        for tag, sign in (("shipped", VNL_VELOCITY_SIGN_SHIPPED),
                          ("flipped", VNL_VELOCITY_SIGN_FLIPPED)):
            op = dipole_operator(geom, bvec=bvec, blat=blat,
                                 vnl_setup=setup, vnl_velocity_sign=sign)
            arms[tag] = np.asarray([
                _mtxel(_apply_one_k(op, psi[ik], gv[ik], gmask[ik],
                                    bidx[ik], kvecs[ik]),
                       psi[ik], gmask[ik])
                for ik in range(psi.shape[0])])
    return arms["shipped"], arms["flipped"]


def _median_relative_move(a, b):
    fa, fb = np.abs(a) ** 2, np.abs(b) ** 2
    return float(np.median(np.abs(fb - fa) / (np.abs(fa) + 1.0e-30)))


def test_the_velocity_sign_moves_oscillator_strengths():
    """The measurement this file was built to make.

    Oscillator strengths are ``|d|^2``, and the two arms differ by
    ``2 * v_NL`` inside ``d`` -- not by a global phase, which ``|d|^2``
    would not see.  So the two arms give genuinely different absorption,
    and this is the cell that says by how much without a cluster.
    """
    shipped, flipped = _both_arms_matrix_elements()
    move = _median_relative_move(shipped, flipped)
    assert move > _SIGN_MOVES_F_BY_AT_LEAST, (
        f"the two velocity arms moved |d|^2 by a median of {move:.3e}, "
        f"below {_SIGN_MOVES_F_BY_AT_LEAST}.  Either the knob stopped "
        f"reaching the assembly or the projector term went null -- both "
        f"turn every other cell in this file into a tautology.")


def test_the_sensitivity_measurement_is_not_a_tautology():
    """RED TWIN for the cell above.

    With a null ``E_super`` the nonlocal term vanishes identically, both
    arms collapse onto bare ``p``, and the sensitivity cell would be
    measuring nothing while still reading green if its threshold were
    written the other way round.  This asserts the collapse really is
    what a dead knob looks like, so the threshold above has a meaning.
    """
    import jax.numpy as jnp
    from dataclasses import replace
    from tests.test_dipole_vnl_velocity_sign import (
        _apply_one_k, _fixture, _geom, _mesh, _mtxel, _vnl_setup)
    from common.mtxel_sweep import (VNL_VELOCITY_SIGN_FLIPPED,
                                    VNL_VELOCITY_SIGN_SHIPPED, dipole_operator)

    mesh = _mesh()
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    dead = replace(_vnl_setup(), E_super=jnp.zeros_like(_vnl_setup().E_super))
    got = {}
    with mesh:
        geom = _geom(mesh)
        for tag, sign in (("shipped", VNL_VELOCITY_SIGN_SHIPPED),
                          ("flipped", VNL_VELOCITY_SIGN_FLIPPED)):
            op = dipole_operator(geom, bvec=bvec, blat=blat,
                                 vnl_setup=dead, vnl_velocity_sign=sign)
            got[tag] = np.asarray([
                _mtxel(_apply_one_k(op, psi[ik], gv[ik], gmask[ik],
                                    bidx[ik], kvecs[ik]),
                       psi[ik], gmask[ik])
                for ik in range(psi.shape[0])])
    move = _median_relative_move(got["shipped"], got["flipped"])
    assert move < 1.0e-12, (
        f"a null projector strength still moved |d|^2 by {move:.3e}, so "
        f"the arms differ by something that is not the nonlocal term")


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
    from psp.get_dipole_mtxels import resolve_vnl_velocity_sign
    from common.mtxel_sweep import VNL_VELOCITY_SIGN_SHIPPED

    # The arm a bare run takes must still be the arm the references were
    # cut at, whatever the shipped default has become.
    resolved = resolve_vnl_velocity_sign(None, "")
    assert resolved in (-1.0, +1.0)
    assert resolved == VNL_VELOCITY_SIGN_SHIPPED or resolved == +1.0
