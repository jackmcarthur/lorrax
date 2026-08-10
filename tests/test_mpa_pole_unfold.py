"""THE GATE that retires ``refuse_wedge_pole_slab``, and its red twins.

The refusal that stood here until 2026-08-10 turned away a fit store on
the symmetry wedge and named the gate that would retire it: *rebuild*
``W_c(q, z_j)`` *from the UNFOLDED poles at the store's own* ``2*n_p``
*sample points and compare against the same store's* ``W`` *unfolded by*
``unfold_isdf_operator``.  That is arm A below, run over every full-BZ q
and every sample, element by element.

WHAT THE GATE IS ACTUALLY MEASURING.  Two objects reach the full zone by
two different routes and must agree:

* ``W`` goes through ``unfold_isdf_operator`` — a centroid double
  permute, an umklapp phase, and a completion on the time-reversed rows.
* the POLE FIELD goes through the same map applied to ``(Omega_p, B_p)``,
  and the claim being certified is that the map COMMUTES with the
  multipole model: the unfold multiplies every element by a
  FREQUENCY-INDEPENDENT scalar, so it can only permute pole positions and
  scale residues, and can never move a pole.

THE FLOOR IS MEASURED AND NOT ASSUMED.  A rebuilt ``W`` is a MODEL and
the reference is the SAMPLES, so the comparison can never be tighter
than the fit's own forward residual — and that residual is not the
store's backward error.  On this deck the store reports a backward error
of ~2e-15 while the model misses its own samples by ~1e-13, which is
what a conditioned least-squares solve does and is not a defect.  So arm
A measures the WEDGE's residual first, against the wedge's own stored
samples, and then requires the full-BZ residual not to EXCEED it: the
claim being certified is "the unfold adds nothing", and that is a
comparison between two residuals rather than a number typed into an
assertion.  Arm A2 then compares the pole field itself against the map
written out by hand, where the bar really is floating point.

THE CONJUGATION, WHICH IS THE HAZARD THE REFUSAL NAMED.  Five defects in
this tree have come from an elementwise conjugate standing in for an
operation on the ``(mu, nu)`` PAIR — most recently the crossing
operator-Im fix, where ``Im`` elementwise stood in for
``(cX - cX^dagger)/2i`` and cost 43.8 eV of star spread.  This is the
same class and the same answer: time reversal acts on a ``(mu, nu)``
operator as the pair transpose, the conjugate is its Hermitian shorthand,
and under the correct rule NOTHING in a pole field is conjugated.  So the
red twins in arm B are not decoration.  They exhibit, in order: the
conjugation applied to the time-reversed members (the rule the refusal
feared), the conjugation applied to the members that are NOT time
reversed (the predicate inverted, which is how the 183.61 eV kin_ion
defect actually happened), the pair transpose skipped, and the umklapp
phase dropped.  Each must fail the gate loudly, and the first two must
fail on DISJOINT sets of q -- the localization the refusal promised as
its diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("symmetry_maps.qirr_store")
jax = pytest.importorskip("jax")

from symmetry_maps import (centroid_source_map_and_wrap,          # noqa: E402
                           verify_centroid_orbit_closure)
from symmetry_maps import qirr_store as QS                        # noqa: E402

from file_io import mpa_store as MS                               # noqa: E402
from gw.mpa import fit_driver, sampling                           # noqa: E402
from gw.mpa import sigma_pass as SP                               # noqa: E402

# ---------------------------------------------------------------------------
# Geometry: a zone with TWO time-reversed rows and THREE self-parent rows
# ---------------------------------------------------------------------------
# The glide {sigma_z | (1/2, 0, 0)} on a 12-grid, as in the sibling MPA
# suites -- the one in-memory geometry in this tree whose L rows are
# non-trivial, which is what makes the dropped-phase twin able to fail.
_FFT = np.array([12, 12, 12], dtype=np.int64)
_SYMS = np.stack([np.eye(3, dtype=np.int64),
                  np.diag([1, 1, -1]).astype(np.int64)])
_TNP = np.array([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])

#: 6 full-BZ q onto 3 IBZ parents.  Rows 3 and 5 fold only through TIME
#: REVERSAL (sym >= n_sym_spatial = 2); rows 0, 2 and 4 are their
#: parents' own representatives (sym = 0, identity permutation, zero
#: wrap) and are the TRIM-shaped rows arm D holds to bit equality.
_IRR = np.array([0, 1, 1, 2, 2, 0], dtype=np.int32)
_SYM = np.array([0, 1, 0, 3, 0, 2], dtype=np.int32)
_N_SYM_SPATIAL = 2
_N_Q_IBZ = 3
_N_Q_FULL = 6

_Q_IRR = np.array([[0.0, 0.0, 0.0],
                   [1 / 3, 0.0, 1 / 4],
                   [0.0, 1 / 3, 1 / 3]])

_SEEDS = ((0, 0, 0), (1, 2, 3), (2, 5, 7), (3, 1, 9),
          (4, 4, 4), (5, 9, 2), (6, 3, 8), (7, 7, 1))

_N_MU = 16
_N_P = 3
_OMEGA_M = 4.0
_W_NAME = "W_qmunu_omega"
_SAMPLING = {"varpi": [0.1, 1.0], "n_p": _N_P, "alpha": 1,
             "omega_max": _OMEGA_M, "protocol": "double_parallel"}

_TRS_ROWS = np.flatnonzero(_SYM >= _N_SYM_SPATIAL)
#: The rows the map leaves ALONE: sym 0 is the identity operation, so
#: identity permutation, zero lattice wrap, no time reversal.
_SELF_ROWS = np.flatnonzero(_SYM == 0)


def _closed_centroid_set():
    S = np.asarray(_SYMS, dtype=np.float64)
    rinv = np.rint(np.linalg.inv(S)).astype(np.int64)
    tint = np.rint(np.asarray(_TNP, dtype=np.float64) / (2.0 * np.pi)
                   * _FFT).astype(np.int64)
    imgs = set()
    for r in np.asarray(_SEEDS, dtype=np.int64):
        for s in range(S.shape[0]):
            imgs.add(tuple(((rinv[s] @ r + tint[s]) % _FFT).tolist()))
    return np.array(sorted(imgs), dtype=np.int32)


def _geometry():
    cent = _closed_centroid_set()
    verdict = verify_centroid_orbit_closure(
        cent.astype(np.float64) / _FFT, _SYMS, tnp=_TNP, fft_grid=_FFT)
    assert verdict.closed, verdict.describe()
    perm, L = centroid_source_map_and_wrap(
        cent, _SYMS, _TNP, _FFT, validate=True, extend_trs=True)
    tables = QS.QirrTables(_IRR, _SYM, _Q_IRR, perm, L, _N_SYM_SPATIAL)
    return tables, verdict, int(cent.shape[0])


def _mesh():
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


# ---------------------------------------------------------------------------
# A planted pole field, and the W it is exactly the fit of
# ---------------------------------------------------------------------------

def _planted_field(n_p, n_q, n_mu):
    """A fourth-quadrant field with structure in p, q, mu AND nu.

    Structure in BOTH centroid indices matters here and does not in the
    fit suites: a field symmetric in (mu, nu) would make the pair
    transpose a no-op and the red twin that inverts the TRS predicate
    unable to fail.
    """
    rng = np.random.default_rng(9871)
    a = 0.30 + 0.55 * np.arange(n_p)[:, None, None, None]
    a = a * (1.0 + 0.25 * rng.random((1, n_q, n_mu, n_mu)))
    gamma = (0.06 + 0.03 * np.arange(n_p)[:, None, None, None]) * a
    Omega = (a - 1j * gamma).astype(np.complex128)
    mag = 0.4 + rng.random((n_p, n_q, n_mu, n_mu))
    phase = 2 * np.pi * rng.random((n_p, n_q, n_mu, n_mu))
    B = (mag * np.exp(1j * phase)).astype(np.complex128)
    return Omega, B


def _model(Omega, B, z):
    """``sum_p 2 Omega_p B_p / (z**2 - Omega_p**2)`` over a whole field.

    ``Omega``/``B`` are ``(n_p, ...)``; returns ``(n_z, ...)``.
    """
    zz = np.asarray(z, dtype=np.complex128)
    idx = (slice(None),) + (None,) * (Omega.ndim - 1)
    denom = zz[(slice(None), None) + (None,) * (Omega.ndim - 1)] ** 2 \
        - Omega[None] ** 2
    assert np.all(np.abs(denom) > 0.0), "a sample sits on a planted pole"
    return np.sum(2.0 * Omega[None] * B[None] / denom, axis=1)


@pytest.fixture(scope="module")
def wedge(tmp_path_factory):
    """A wedge W(omega) store, its planted field, and the fit of it."""
    root = tmp_path_factory.mktemp("mpa_pole_unfold")
    tables, verdict, n_mu = _geometry()
    assert n_mu == _N_MU, f"geometry drifted: {n_mu} centroids"
    z = sampling.double_parallel_grid(
        _N_P, _OMEGA_M, material_class="insulator", alpha=1,
        energy_unit="Ha")
    Omega, B = _planted_field(_N_P, _N_Q_IBZ, n_mu)
    W = _model(Omega, B, z)
    w_path = root / "W_omega.h5"
    MS.allocate_w_omega(
        str(w_path), _W_NAME, n_omega=2 * _N_P, n_q_on_disk=_N_Q_IBZ,
        n_mu=n_mu, tables=tables, omega=z, sampling=_SAMPLING,
        omega_line=np.array([0] * _N_P + [1] * _N_P, dtype=np.int32),
        closure_verdict=verdict, screening_content="W_c",
        provenance={"deck": "synthetic-glide-pole-unfold"})
    for i in range(2 * _N_P):
        MS.write_w_slab(str(w_path), _W_NAME, i, W[i], ready=True)

    fit_path = root / "mpa_fit.h5"
    ledger, _ = fit_driver.run_fit_driver(
        str(w_path), _W_NAME, str(fit_path), z, _N_P)
    return {"root": root, "w_path": w_path, "fit_path": fit_path,
            "tables": tables, "z": z, "n_mu": n_mu, "ledger": ledger}


def _read_unfolded_poles(fit_path, n_p, mesh):
    """The production read, stacked over the pole axis."""
    Om, Bp = [], []
    for p in range(n_p):
        o, b = MS.read_pole_slice(str(fit_path), p, unfold=True, mesh_xy=mesh)
        Om.append(np.asarray(o))
        Bp.append(np.asarray(b))
    return np.stack(Om), np.stack(Bp)


def _read_wedge_poles(fit_path, n_p):
    Om, Bp = [], []
    for p in range(n_p):
        o, b = MS.read_pole_slice(str(fit_path), p)
        Om.append(np.asarray(o))
        Bp.append(np.asarray(b))
    return np.stack(Om), np.stack(Bp)


def _hand_unfold_poles(Omega, B, tables, *, phase=True, conj_extra=None,
                       swap=True):
    """The certified map in plain numpy, and its crossed twins.

    With the defaults this is the map ``mpa_store.unfold_pole_field``
    implements, written out independently so arm A2 compares two
    computations rather than one against itself::

        Omega'[q, mu, nu] = Omega[i(q), a(mu), a(nu)]     (swapped on TRS)
        B'[q, mu, nu]     = c(mu, nu) * B[i(q), a(mu), a(nu)]

    with ``c = exp(2i*pi*q_irr.(L_mu - L_nu))``, conjugated on the
    time-reversed rows, and NOTHING conjugated in the data.

    The knobs are the twins.  ``conj_extra='trs'`` conjugates
    ``(Omega, B)`` on the time-reversed rows — the rule the retired
    refusal feared, and the one that sends a fourth-quadrant pole to
    ``a + i*Gamma``.  ``conj_extra='spatial'`` conjugates the OTHER half,
    which is the same mistake with the predicate inverted.  ``swap=False``
    drops the pair transpose.  ``phase=False`` drops the umklapp factor.
    """
    irr = np.asarray(tables.irr_idx_q)
    sym = np.asarray(tables.sym_idx_q)
    ntran = int(tables.n_sym_spatial)
    n_p, _, n_mu, _ = Omega.shape
    Om = np.zeros((n_p, len(irr), n_mu, n_mu), dtype=np.complex128)
    Bp = np.zeros_like(Om)
    for iq in range(len(irr)):
        i, s = int(irr[iq]), int(sym[iq])
        a = np.asarray(tables.sym_perm[s])
        qL = (np.asarray(tables.L_table[s], dtype=float)
              @ np.asarray(tables.q_irr_frac)[i])
        ph = np.exp(2j * np.pi * qL) if phase else np.ones(n_mu, complex)
        pf = ph[:, None] * np.conj(ph)[None, :]
        is_trs = s >= ntran
        src = np.ix_(a, a)
        o = Omega[:, i][:, src[0], src[1]]
        b = B[:, i][:, src[0], src[1]]
        if is_trs and swap:
            o = np.swapaxes(o, -1, -2)
            b = np.swapaxes(b, -1, -2)
        if conj_extra is not None and is_trs == (conj_extra == "trs"):
            o, b = np.conj(o), np.conj(b)
        Om[:, iq] = o
        Bp[:, iq] = (np.conj(pf) if is_trs else pf)[None] * b
    return Om, Bp


def _rebuild_residual(Omega_full, B_full, W_ref, z):
    """Per-q max relative deviation of the rebuilt W from the reference."""
    got = _model(Omega_full, B_full, z)                 # (n_z, n_q, mu, mu)
    ref = np.asarray(W_ref)
    scale = np.abs(ref).max()
    return np.abs(got - ref).max(axis=(0, 2, 3)) / scale


def _reference_W(wedge, mesh):
    """``unfold_isdf_operator``'s W at every stored sample, full BZ."""
    slabs = []
    for i in range(2 * _N_P):
        arr, _ = MS.read_w_slab(str(wedge["w_path"]), _W_NAME, i,
                                unfold=True, mesh_xy=mesh)
        slabs.append(np.asarray(arr))
    return np.stack(slabs)


def _z_in_pole_units(wedge):
    """The store's grid in the unit ``read_pole_slice`` returns poles in.

    ``W_c`` is invariant under scaling ``(Omega, B, z)`` together -- the
    model is homogeneous of degree zero in them -- so this is a
    bookkeeping conversion and not a physical one, and it is done here
    rather than by reading the poles ``raw`` so the arms exercise the
    production read.
    """
    unit = wedge["ledger"]["energy_unit"]
    return np.asarray(wedge["z"]) * (2.0 if unit == "Ha" else 1.0)


# ---------------------------------------------------------------------------
# (A) THE GATE
# ---------------------------------------------------------------------------

def test_the_unfolded_poles_rebuild_the_unfolded_W(wedge):
    """The retirement gate for ``refuse_wedge_pole_slab``, run in full.

    Every full-BZ q, every one of the store's ``2*n_p`` samples, every
    element -- against the FLOOR the same fit sets on its own wedge.
    """
    mesh = _mesh()
    assert wedge["ledger"]["q_storage"] == "ibz"
    assert wedge["ledger"]["n_q_full"] == _N_Q_FULL
    z = _z_in_pole_units(wedge)

    # THE FLOOR: the fit against its OWN samples, on the wedge, with no
    # unfold anywhere in it.
    Om_w, B_w = _read_wedge_poles(wedge["fit_path"], _N_P)
    W_wedge = np.stack([np.asarray(MS.read_w_slab(
        str(wedge["w_path"]), _W_NAME, i)[0]) for i in range(2 * _N_P)])
    floor_q = _rebuild_residual(Om_w, B_w, W_wedge, z)
    floor = float(floor_q.max())

    Om_f, B_f = _read_unfolded_poles(wedge["fit_path"], _N_P, mesh)
    assert Om_f.shape == (_N_P, _N_Q_FULL, _N_MU, _N_MU)
    per_q = _rebuild_residual(Om_f, B_f, _reference_W(wedge, mesh), z)
    worst = float(per_q.max())
    print(f"[pole-unfold gate] wedge floor {floor:.3e}; unfolded worst "
          f"{worst:.3e} (ratio {worst / floor:.3f}); per q "
          + " ".join(f"{v:.1e}" for v in per_q)
          + f"; store backward error "
            f"{wedge['ledger']['backward_error_max']:.3e}")
    assert worst <= floor * 1.05 + 1e-16, (
        f"the unfold ADDED error: the same fit reproduces its own wedge "
        f"samples to {floor:.3e} and the unfolded field reproduces the "
        f"unfolded W only to {worst:.3e}, worst at q="
        f"{int(np.argmax(per_q))}.  Rows {_TRS_ROWS.tolist()} are the "
        f"TIME-REVERSED members; a deviation confined to them localizes "
        f"the defect to the TRS completion, and one spread over the "
        f"spatial rows localizes it to the permutation or the phase.")


def test_the_shipped_unfold_is_the_pair_transpose_map(wedge):
    """The production read against the hand map, element by element.

    Arm A compares two ROUTES to W; this compares the pole field itself
    against the map written out in numpy, so a defect that happened to
    cancel in the rebuild would still be caught.
    """
    mesh = _mesh()
    Om_f, B_f = _read_unfolded_poles(wedge["fit_path"], _N_P, mesh)
    Om_w, B_w = _read_wedge_poles(wedge["fit_path"], _N_P)
    Om_h, B_h = _hand_unfold_poles(Om_w, B_w, wedge["tables"])
    assert np.abs(Om_f - Om_h).max() / np.abs(Om_h).max() < 1e-15
    assert np.abs(B_f - B_h).max() / np.abs(B_h).max() < 1e-15


def test_the_unfold_cannot_leave_the_fourth_quadrant(wedge):
    """``Re Omega > 0``, ``Im Omega < 0`` -- and NOT as a repair.

    The refusal this file retires was written against one failure in
    particular: *"a fourth-quadrant pole conjugated the wrong way becomes
    exp(+Gamma*tau), which grows."*  Under the certified map nothing
    conjugates a pole, so the full-BZ widths are a PERMUTATION of the
    wedge's -- asserted here as multiset equality, which is a much
    stronger statement than a sign check and is the one that says no
    repair happened.
    """
    mesh = _mesh()
    Om_f, _ = _read_unfolded_poles(wedge["fit_path"], _N_P, mesh)
    Om_w, _ = _read_wedge_poles(wedge["fit_path"], _N_P)
    assert (Om_w.imag < 0).all(), "PRECONDITION: the wedge is time-ordered"
    assert (Om_f.real > 0).all() and (Om_f.imag < 0).all()
    for p in range(_N_P):
        got = np.sort_complex(Om_f[p].ravel())
        # every full-BZ element's pole comes from SOME wedge element's
        # pole, so the full multiset is the wedge multiset with the
        # per-q multiplicities the map assigns.
        assert np.isin(np.round(got, 12),
                       np.round(Om_w[p].ravel(), 12)).all()


# ---------------------------------------------------------------------------
# (B) RED TWINS -- crossed conventions, each with its own teeth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,kw,expect_rows",
    [("conjugate the time-reversed members",
      dict(conj_extra="trs"), "trs"),
     ("conjugate everything BUT the time-reversed members",
      dict(conj_extra="spatial"), "spatial"),
     ("skip the pair transpose on the time-reversed members",
      dict(swap=False), "trs"),
     ("drop the umklapp phase",
      dict(phase=False), "any")])
def test_a_crossed_convention_fails_the_gate_loudly(wedge, label, kw,
                                                    expect_rows):
    """Three wrong maps, and the gate must reject all three.

    A gate that only ever sees the right answer is measuring nothing.
    Each twin here is a convention this tree has actually shipped
    somewhere: the conjugation on the antiunitary rows (the W unfold's
    own default, correct for a Hermitian operator and not for this one),
    the same conjugation with the predicate INVERTED (the kin_ion defect,
    183.61 eV with the diagonal exactly zero), and the missing umklapp
    phase (~unity relative error on CrI3 V_q before the 2026-05 fix).
    """
    mesh = _mesh()
    Om_w, B_w = _read_wedge_poles(wedge["fit_path"], _N_P)
    Om_t, B_t = _hand_unfold_poles(Om_w, B_w, wedge["tables"], **kw)
    per_q = _rebuild_residual(Om_t, B_t, _reference_W(wedge, mesh),
                              _z_in_pole_units(wedge))
    worst = float(per_q.max())
    assert worst > 1e-3, (
        f"'{label}' changed the answer by only {worst:.3e}; this "
        f"geometry cannot see that convention and the gate has no teeth "
        f"against it")

    bad = np.flatnonzero(per_q > 1e-9)
    spatial = np.setdiff1d(np.arange(_N_Q_FULL), _TRS_ROWS)
    if expect_rows == "trs":
        # THE DIAGNOSTIC THE REFUSAL PROMISED: a wrong conjugation on the
        # time-reversed members moves those rows and no others.
        assert set(bad.tolist()) == set(_TRS_ROWS.tolist()), (
            f"'{label}' moved rows {bad.tolist()}, not the time-reversed "
            f"rows {_TRS_ROWS.tolist()}; the localization the gate's "
            f"failure message offers would point at the wrong half")
    elif expect_rows == "spatial":
        assert not set(bad.tolist()) & set(_TRS_ROWS.tolist()), (
            f"'{label}' moved a TIME-REVERSED row ({bad.tolist()}); the "
            f"inverted predicate must show up on the other half")
        assert set(bad.tolist()) <= set(spatial.tolist())
        assert bad.size >= 1


def test_the_two_conjugation_twins_fail_on_disjoint_rows(wedge):
    """The localization, stated as one fact rather than two.

    "Disagreement confined to the time-reversed members localizes to the
    conjugation" is only a usable diagnostic if the OTHER mistake --
    conjugating the wrong half -- lands somewhere else.  It does, and the
    two sets are disjoint and together cover every q that the map does
    not leave alone.
    """
    mesh = _mesh()
    ref = _reference_W(wedge, mesh)
    z = _z_in_pole_units(wedge)
    Om_w, B_w = _read_wedge_poles(wedge["fit_path"], _N_P)

    def rows(**kw):
        o, b = _hand_unfold_poles(Om_w, B_w, wedge["tables"], **kw)
        return set(np.flatnonzero(
            _rebuild_residual(o, b, ref, z) > 1e-9).tolist())

    a = rows(conj_extra="trs")
    b = rows(conj_extra="spatial")
    assert a and b and not (a & b)
    assert a == set(_TRS_ROWS.tolist())


def test_the_conjugation_twin_is_the_growing_exponential(wedge):
    """WHY the wrong conjugation was worth a refusal, in one number.

    ``Omega_p = a - i*Gamma`` conjugated is ``a + i*Gamma``, and the tau
    stage reads ``W(tau) = sum_p B_p exp(-i*Omega_p*tau)`` -- so the twin
    does not merely disagree, it puts a GROWING exponential into the
    integrand.  Counted here rather than described: the twin's field has
    poles above the real axis and the certified map's has none.
    """
    Om_w, B_w = _read_wedge_poles(wedge["fit_path"], _N_P)
    good, _ = _hand_unfold_poles(Om_w, B_w, wedge["tables"])
    bad, _ = _hand_unfold_poles(Om_w, B_w, wedge["tables"], conj_extra="trs")
    assert int(np.count_nonzero(good.imag > 0)) == 0
    n_bad = int(np.count_nonzero(bad.imag > 0))
    assert n_bad == _N_P * _TRS_ROWS.size * _N_MU * _N_MU, (
        f"{n_bad} poles crossed the real axis; the twin should put every "
        f"element of every time-reversed row up there")


# ---------------------------------------------------------------------------
# (C) The rows the map leaves alone
# ---------------------------------------------------------------------------

def test_the_self_parent_rows_are_bit_identical_to_the_wedge(wedge):
    """Zero differing elements -- the property the W unfold already proved.

    A q that is its own IBZ representative folds through the identity:
    identity permutation, zero lattice wrap, no time reversal.  The
    unfold must therefore return those rows UNCHANGED, and the bar is
    bit equality and not a tolerance, because a copy that is only
    approximately a copy is a copy that went through arithmetic it had no
    business doing.  Counted, not normed: two conjugation defects in this
    tree were diagonal-preserving and off-diagonal-destroying.
    """
    mesh = _mesh()
    Om_f, B_f = _read_unfolded_poles(wedge["fit_path"], _N_P, mesh)
    Om_w, B_w = _read_wedge_poles(wedge["fit_path"], _N_P)
    assert _SELF_ROWS.size >= 2, "PRECONDITION: the zone must have such rows"
    for q in _SELF_ROWS:
        i = int(_IRR[q])
        for got, want, name in ((Om_f[:, q], Om_w[:, i], "Omega_p"),
                                (B_f[:, q], B_w[:, i], "B_p")):
            n_diff = int(np.count_nonzero(got != want))
            assert n_diff == 0, (
                f"{name} at self-parent q={q} (IBZ {i}): {n_diff} of "
                f"{got.size} elements differ, worst "
                f"{np.abs(got - want).max():.3e}")


# ---------------------------------------------------------------------------
# (D) The refusals that are LEFT
# ---------------------------------------------------------------------------

def test_a_wedge_store_with_no_tables_refuses_by_name(wedge, tmp_path):
    """The tables are the map; without them there is nothing to guess with."""
    import shutil
    stripped = tmp_path / "no_tables.h5"
    shutil.copy(wedge["fit_path"], stripped)
    # The realistic corruption: the store still SAYS it is a wedge of a
    # six-point zone, and the group that says how is gone.
    with h5py.File(stripped, "a") as f:
        del f[MS.FIT_TABLE_OWNER + QS.QIRR_TABLE_SUFFIX]
    with pytest.raises(ValueError) as exc:
        MS.read_pole_slice(str(stripped), 0, unfold=True, mesh_xy=_mesh())
    assert "stamp_fit_unfold_tables" in str(exc.value)


def test_the_unfold_needs_a_mesh_and_says_so(wedge):
    with pytest.raises(ValueError, match="needs a mesh"):
        MS.read_pole_slice(str(wedge["fit_path"]), 0, unfold=True)


def test_restamping_different_tables_refuses(wedge):
    """The poles did not change, so at most one table group describes them."""
    tables, _, _ = _geometry()
    other = QS.QirrTables(_IRR, _SYM, _Q_IRR + 0.05, tables.sym_perm,
                          tables.L_table, _N_SYM_SPATIAL)
    with pytest.raises(ValueError, match="already carries unfold tables"):
        MS.stamp_fit_unfold_tables(str(wedge["fit_path"]), other)
    # ...and re-stamping the SAME tables is a no-op, so a driver may
    # leave the call in.
    MS.stamp_fit_unfold_tables(str(wedge["fit_path"]), tables)


def test_the_sigma_pass_reads_this_store_as_a_wedge(wedge):
    """The verdict the pass loop takes, on the real ledger."""
    assert SP.resolve_pole_q_axis(wedge["ledger"], _N_Q_FULL) is True
    assert SP.resolve_pole_q_axis(
        dict(wedge["ledger"], n_q=_N_Q_FULL), _N_Q_FULL) is False
