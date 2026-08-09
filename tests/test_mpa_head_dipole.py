"""Gates for the q -> 0 head channel built from the dipole matrix
elements (``gw.mpa.head_dipole``).

Every cell here runs on the COMMITTED fixtures and on WSL CPU: the head
is a vertical-transition sum over ``dipole.h5``, so it needs no ISDF
basis, no device and no cluster leg.  That is a property of the head
worth stating, because it makes the most physically interpretable
channel of the whole multipole stack the cheapest one to test.

The ladder, in the order the brief fixes:

a. THE STATIC BRIDGE -- ``head_tensor`` at z = 0 against the shipped
   ``common.chi_from_dipole.compute_S_omega``, which is the q -> 0 head
   the plasmon-pole path has been using all along.
b. THE GN PROBE -- the same, at z = 2i Ry, so that a bug living only in
   the z-dependence has somewhere to show.
c. THE CUBIC-SYMMETRY TWIN -- silicon's head must be isotropic, and a
   deliberately broken dipole set must make it anisotropic.  The red
   twin is what turns "it came out isotropic" into evidence.
d. THE CROSSED-CONVENTION RED TWINS -- on the dipole's complex
   conjugate, in both the places it can hide.
e. THE PHYSICS GATE -- the fitted head pole against silicon's plasmon.

Cells (a)-(d) are cheap and unconditional.  Cell (e) runs the mini-BZ
averager and the Pade fit and is marked slow.
"""

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import h5py  # noqa: E402

from gw.mpa import head_dipole  # noqa: E402
from gw.mpa import pade_fit, sampling  # noqa: E402

RY_EV = 13.605693122994
_HERE = os.path.dirname(os.path.abspath(__file__))


def _deck(name):
    """Cell geometry + dipole block for one committed regression deck."""

    root = os.path.join(_HERE, "regression", name)
    wfn = os.path.join(root, "WFN.h5")
    dip = os.path.join(root, "dipole.h5")
    if not (os.path.exists(wfn) and os.path.exists(dip)):
        pytest.skip(f"{name} fixture is not present")
    with h5py.File(wfn, "r") as f:
        blat = float(np.asarray(f["mf_header/crystal/blat"]))
        out = {
            "cell_volume": float(np.asarray(f["mf_header/crystal/celvol"])),
            "bvec": blat * np.asarray(f["mf_header/crystal/bvec"],
                                      dtype=np.float64),
            "kgrid": tuple(int(v) for v in
                           np.asarray(f["mf_header/kpoints/kgrid"])),
            "nspin": int(np.asarray(f["mf_header/kpoints/nspin"])),
            "nspinor": int(np.asarray(f["mf_header/kpoints/nspinor"])),
            "nelec": int(np.asarray(
                f["mf_header/kpoints/ifmax"]).ravel()[0]),
        }
    with h5py.File(dip, "r") as f:
        out["dipole_cart"] = np.asarray(f["dipole_cart"])
        out["deltaE"] = np.asarray(f["deltaE"])
    out["nk_tot"] = int(out["dipole_cart"].shape[1])
    out["n_b"] = int(out["dipole_cart"].shape[2])
    n_occ = out["nelec"]
    out["delta"] = out["deltaE"][:, n_occ:out["n_b"], :n_occ]
    return out


def _s_tensor(deck, z_values):
    """The shipped q -> 0 head, at the same samples."""

    from common.chi_from_dipole import compute_S_omega

    occ = np.zeros((deck["nk_tot"], deck["n_b"]), dtype=float)
    occ[:, :deck["nelec"]] = 1.0
    return np.asarray(compute_S_omega(
        jnp.asarray(deck["dipole_cart"]), jnp.asarray(deck["deltaE"]),
        jnp.asarray(occ), deck["cell_volume"], deck["nk_tot"],
        deck["nspin"], deck["nspinor"], jnp.asarray(z_values)))


def _head(deck, z_values, **kw):
    return np.asarray(head_dipole.head_tensor(
        deck["dipole_cart"], deck["delta"], z_values,
        nelec=deck["nelec"], cell_volume=deck["cell_volume"],
        nk_tot=deck["nk_tot"], nspin=deck["nspin"],
        nspinor=deck["nspinor"], **kw))


# ---------------------------------------------------------------------
# (a) and (b): the bridges to the shipped head.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("deck_name", ["si_cohsex_debug", "hbn_cohsex_debug"])
def test_static_bridge_against_s_tensor(deck_name):
    """z = 0: the K_z route reproduces ``compute_S_omega`` exactly.

    THE RESIDUAL MECHANISM IS FLOATING-POINT SUMMATION ORDER AND NOTHING
    ELSE, and that is the honest reading of this gate rather than a
    weakness of it.  The two routes are the same rational function of
    ``Delta``: ``compute_S_omega`` evaluates
    ``1 / (Delta (z**2 - Delta**2))`` in one expression, and
    ``head_tensor`` forms ``d = v/Delta`` and multiplies by
    ``K_z = 2 Delta/(z**2 - Delta**2)``.  So this is not an independent
    check of the physics -- it is a check that the NORMALISATION, the
    band window, the occupied/empty split, the spin-degeneracy factor
    and the sign of the kernel are the production ones, which is the
    class of bug that would show up here as orders of magnitude rather
    than as ulps.
    """

    deck = _deck(deck_name)
    z = np.array([0.0 + 0.0j])
    a = _head(deck, z)[0]
    b = _s_tensor(deck, z)[0]
    scale = float(np.max(np.abs(b)))
    rel = float(np.max(np.abs(a - b))) / scale
    assert rel < 1.0e-12, (
        f"static bridge: max rel diff {rel:.3e} between head_tensor and "
        f"compute_S_omega on {deck_name}.  At this tolerance the only "
        f"admissible mechanism is summation order; anything larger is a "
        f"convention that moved.")


@pytest.mark.parametrize("deck_name", ["si_cohsex_debug", "hbn_cohsex_debug"])
def test_gn_probe_bridge_against_s_tensor(deck_name):
    """z = 2i Ry: the same identity, off the static cell.

    The static bridge cannot see a bug in the z-dependence, because at
    z = 0 the kernel is ``-2/Delta`` and every route that gets the
    normalisation right gets it.  The Godby-Needs probe is a genuine
    imaginary-axis sample -- the cell ``sample_plan`` labels ``imag`` --
    and it is the second of the two frequencies the shipped plasmon-pole
    driver actually asks the head resolver for, so agreeing here means
    the multipole head can stand in for the PPM head at both of its
    points.
    """

    deck = _deck(deck_name)
    z = np.array([2.0j])
    a = _head(deck, z)[0]
    b = _s_tensor(deck, z)[0]
    scale = float(np.max(np.abs(b)))
    rel = float(np.max(np.abs(a - b))) / scale
    assert rel < 1.0e-12, (
        f"GN probe bridge: max rel diff {rel:.3e} on {deck_name}")


def test_strip_sample_is_not_reachable_by_the_static_route():
    """The control: the two bridges above are not trivially true.

    If ``head_tensor`` ignored ``z`` the first two cells would still
    pass at z = 0 and fail at 2i, so the pair already discriminates.
    This cell adds the statement the strip needs: a sample with BOTH
    parts nonzero gives a head with a nonzero imaginary part, which the
    static and imaginary cells never do.
    """

    deck = _deck("si_cohsex_debug")
    A = _head(deck, np.array([0.0 + 0.0j, 2.0j, 1.0 + 0.5j]))
    assert abs(A[0][0, 0].imag) < 1.0e-14
    assert abs(A[1][0, 0].imag) < 1.0e-14
    assert abs(A[2][0, 0].imag) > 1.0e-3


# ---------------------------------------------------------------------
# (c) the cubic-symmetry twin, with its red arm.
# ---------------------------------------------------------------------

def test_cubic_head_is_isotropic_and_the_red_twin_is_not():
    """Silicon's head tensor is a multiple of the identity; a broken
    dipole set's is not.

    The GREEN arm is a physics statement about silicon that the code
    has no way to impose: nothing in ``head_tensor`` symmetrises
    anything, so isotropy to 1e-6 is the 48-operation cubic point group
    showing up in a sum over 26 624 unsymmetrised transitions.

    The RED arm is what makes that evidence.  Scaling the z component of
    every dipole by 1.5 is a change no norm-preserving convention error
    could produce, and it must land in the tensor's diagonal spread and
    nowhere else -- if the machinery had silently scalarised the head,
    the red arm would come back isotropic too and the green arm would
    have been proving nothing.
    """

    deck = _deck("si_cohsex_debug")
    z = np.array([0.0 + 0.0j, 2.0j])
    rep = head_dipole.isotropy_report(_head(deck, z))
    for r in rep:
        assert r["diag_spread_rel"] < 1.0e-4, r
        assert r["max_offdiag_rel"] < 1.0e-6, r

    broken = deck["dipole_cart"].copy()
    broken[2] *= 1.5
    red = np.asarray(head_dipole.head_tensor(
        broken, deck["delta"], z, nelec=deck["nelec"],
        cell_volume=deck["cell_volume"], nk_tot=deck["nk_tot"],
        nspin=deck["nspin"], nspinor=deck["nspinor"]))
    red_rep = head_dipole.isotropy_report(red)
    for r in red_rep:
        assert r["diag_spread_rel"] > 0.5, (
            "the deliberately anisotropic dipole set came back isotropic, "
            "which means the tensor machinery is not live", r)


def test_uniaxial_deck_is_measurably_anisotropic():
    """hBN, through the same call, is not isotropic -- and that is the
    argument for keeping the head a tensor.

    A scalar head is not a simplification on a uniaxial crystal; it is
    an error the size of the anisotropy.  Measured here on the committed
    fixture, so the number is not a claim about hBN in general but about
    what this code would get wrong.
    """

    deck = _deck("hbn_cohsex_debug")
    A = _head(deck, np.array([0.0 + 0.0j]))[0]
    eps = 1.0 - 8.0 * np.pi * np.diag(A)
    in_plane = 0.5 * (eps[0] + eps[1]).real
    c_axis = eps[2].real
    assert abs(eps[0] - eps[1]) / abs(eps[0]) < 1.0e-4, (
        "hBN's two in-plane directions must be equivalent", eps)
    assert in_plane / c_axis > 1.2, (
        "hBN's head came out nearly isotropic, which it is not", eps)


# ---------------------------------------------------------------------
# (d) the crossed-convention red twins on the dipole conjugation.
# ---------------------------------------------------------------------

def test_head_conjugation_left_and_right_are_the_same_physics():
    """The two spellings differ by a transpose the quadratic form kills.

    Stated in :data:`head_dipole.CONJ_MODES` and MEASURED here rather
    than asserted, because the tempting version of this gate -- "the
    conjugation convention is pinned by the head" -- is false, and a
    gate that passes for a false reason is worse than no gate.
    """

    deck = _deck("si_cohsex_debug")
    z = np.array([1.0 + 0.5j])
    left = _head(deck, z, conj_mode="left")[0]
    right = _head(deck, z, conj_mode="right")[0]
    floor = 1.0e-15 * float(np.max(np.abs(left)))
    assert np.max(np.abs(left - right.T)) < floor
    rng = np.random.RandomState(0)
    for _ in range(8):
        q = rng.normal(size=3)
        q /= np.linalg.norm(q)
        fl = q @ left @ q
        fr = q @ right @ q
        assert abs(fl - fr) <= 1.0e-12 * abs(fl)


def test_head_with_the_conjugate_dropped_is_measurably_wrong():
    """``conj_mode='none'`` -- the real conjugation bug -- IS visible.

    Dropping the conjugate turns ``sum conj(d_a) d_b`` into
    ``sum d_a d_b``: Hermitian becomes complex-symmetric.  On the static
    cell ``K_0`` is real, so the correct head is Hermitian EXACTLY, and
    that is the signature a reader can recognise without knowing any
    magnitude.  The diagonal is unchanged by the bug -- ``|d_a|**2``
    against ``d_a**2`` differ only by a phase that is 1 for a real
    component -- which is precisely why a diagonal-only head test would
    miss it and why this cell looks at the whole tensor.
    """

    deck = _deck("si_cohsex_debug")
    z = np.array([0.0 + 0.0j])
    good = _head(deck, z, conj_mode="left")[0]
    bad = _head(deck, z, conj_mode="none")[0]
    scale = float(np.max(np.abs(good)))
    assert np.max(np.abs(good - good.conj().T)) < 1.0e-15 * scale
    broken = float(np.max(np.abs(bad - bad.conj().T)))
    assert broken > 1.0e3 * np.max(np.abs(good - good.conj().T)), (
        "dropping the dipole's conjugate left the static head Hermitian, "
        "so the conjugate is not where the code thinks it is")


def test_crossed_wing_conjugation_moves_the_local_field_correction():
    """The wing twin, on a synthetic pair-density block.

    The wings are the only place the dipole's conjugation is
    observable, so this is the cell that actually pins the convention.
    A synthetic complex ``M`` is used rather than a deck's ISDF block
    because the statement is about the contraction, not about silicon:
    a real ``M`` would make the crossed arm agree by accident.
    """

    deck = _deck("si_cohsex_debug")
    n_trans = int(np.prod(deck["delta"].shape))
    n_mu = 5
    rng = np.random.RandomState(7)
    M = (rng.normal(size=(n_trans, n_mu))
         + 1j * rng.normal(size=(n_trans, n_mu)))
    M = M.reshape(deck["delta"].shape + (n_mu,))
    z = np.array([1.0 + 0.5j])
    kw = dict(nelec=deck["nelec"], cell_volume=deck["cell_volume"],
              nk_tot=deck["nk_tot"], nspin=deck["nspin"],
              nspinor=deck["nspinor"])
    Y, Z = head_dipole.wing_tensors(
        deck["dipole_cart"], deck["delta"], M, z, **kw)
    Yc, Zc = head_dipole.wing_tensors(
        deck["dipole_cart"], deck["delta"], M, z, crossed=True, **kw)
    assert Y.shape == (1, 3, n_mu) and Z.shape == (1, n_mu, 3)
    rel = (np.max(np.abs(np.asarray(Y) - np.asarray(Yc)))
           / np.max(np.abs(np.asarray(Y))))
    assert rel > 1.0e-3, (
        f"the crossed wing convention changed Y by only {rel:.2e}; the "
        f"wing is then not carrying the dipole phase at all")


def test_block_dyson_reduces_to_the_bare_head_with_no_wings():
    """``macroscopic_head_tensor`` with zero wings returns ``A``.

    The limit that says the local-field correction is a correction: it
    is the only statement about the block solve that holds independently
    of the body, and it is where an index transposition in the Schur
    complement would show.
    """

    deck = _deck("si_cohsex_debug")
    z = np.array([0.0 + 0.0j, 1.0 + 0.5j])
    A = _head(deck, z)
    n_mu = 4
    rng = np.random.RandomState(3)
    X = (rng.normal(size=(2, n_mu, n_mu))
         + 1j * rng.normal(size=(2, n_mu, n_mu))) * 0.01
    V = np.eye(n_mu) * 2.0
    Y = np.zeros((2, 3, n_mu), dtype=complex)
    Z = np.zeros((2, n_mu, 3), dtype=complex)
    out = np.asarray(head_dipole.macroscopic_head_tensor(A, Y, Z, X, V))
    assert np.allclose(out, np.asarray(A), rtol=0, atol=1.0e-18)

    # And with wings it moves, in the direction that reduces screening:
    Y1 = rng.normal(size=(2, 3, n_mu)) * 0.01
    Z1 = np.transpose(np.conj(Y1), (0, 2, 1))
    out1 = np.asarray(head_dipole.macroscopic_head_tensor(A, Y1, Z1, X, V))
    assert np.max(np.abs(out1 - np.asarray(A))) > 0.0


# ---------------------------------------------------------------------
# (e) the physics gate.
# ---------------------------------------------------------------------

def test_fsum_diagnostic_is_reported_not_assumed():
    """The deck's own plasma frequency against the classical one.

    This is a property of ``dipole.h5`` and it caps what the physics
    gate below can claim, so it is measured first and separately.  It is
    asserted only loosely -- the point is that the number exists and is
    reported, not that any particular deck saturates the sum rule.
    """

    deck = _deck("si_cohsex_debug")
    rep = head_dipole.head_fsum_from_transitions(
        deck["dipole_cart"], deck["delta"], nelec=deck["nelec"],
        cell_volume=deck["cell_volume"], nk_tot=deck["nk_tot"],
        nspin=deck["nspin"], nspinor=deck["nspinor"])
    assert rep["omega_p_classical_ry"] * RY_EV == pytest.approx(16.6, abs=0.1)
    assert 0.5 < rep["saturation_ratio"] < 3.0, rep


def test_head_channel_pole_is_the_plasmon():
    """The head channel, fitted, must put its dominant pole at the
    plasmon of the manifold it was fitted to.

    THIS GATE IS TWO CLAIMS AND THEY ARE KEPT APART.  The method claim
    is that the fitted pole reproduces the deck's OWN f-sum plasmon --
    that is a statement about this module and it is asserted.  The
    physics claim is that the deck's f-sum plasmon is silicon's
    (16.6 eV) -- that is a statement about ``dipole.h5`` and it is
    reported, not asserted, because the committed fixture over-saturates
    the sum rule and a gate that asserted 16.7 eV here would be
    asserting a property of a 4x4x4 debug deck.
    """

    deck = _deck("si_cohsex_debug")
    n_p = 6
    omega_m = float(np.max(deck["delta"])) * 1.05
    z = np.asarray(sampling.double_parallel_grid(
        n_p, omega_m, material_class="insulator", energy_unit="Ry"),
        dtype=np.complex128)
    A = _head(deck, z)
    from vcoul.geometry import CoulombGeometry
    geom = CoulombGeometry(bvec=deck["bvec"],
                           cell_volume=deck["cell_volume"])
    v_head, w_head = head_dipole.head_channel_samples(
        A, geom, deck["kgrid"], nsamples=2 ** 14, qmc_reps=2)
    wc = w_head - v_head
    Om, B, _diag = pade_fit.fit_mpa_poles(jnp.asarray(wc), jnp.asarray(z),
                                          n_p)
    Om = np.asarray(Om)
    B = np.asarray(B)
    order = np.argsort(-np.abs(B))
    a_p = float(np.real(Om[order[0]])) * RY_EV
    fsum = head_dipole.head_fsum_from_transitions(
        deck["dipole_cart"], deck["delta"], nelec=deck["nelec"],
        cell_volume=deck["cell_volume"], nk_tot=deck["nk_tot"],
        nspin=deck["nspin"], nspinor=deck["nspinor"])
    deck_wp = fsum["omega_p_ry"] * RY_EV
    assert a_p == pytest.approx(deck_wp, rel=0.15), (
        f"dominant head pole {a_p:.3f} eV against the deck's own f-sum "
        f"plasmon {deck_wp:.3f} eV (silicon's measured value is 16.7 eV; "
        f"this deck's saturation ratio is "
        f"{fsum['saturation_ratio']:.3f})")
