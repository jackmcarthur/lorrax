"""Gates for Σ_c band-convergence extrapolation (``gw.band_extrapolation``).

THE LOAD-BEARING ONE IS :func:`test_brackets_partition_the_band_sum`.  Every
other property of the feature is arithmetic on top of the claim that the
three band brackets PARTITION the Σ_c band sum — that no band is summed
twice and none is dropped.  If that claim breaks (an off-by-one on a bracket
bound, a slice taken on the wrong axis, ψ and its energies sliced
inconsistently, a shared operand consumed by the first bracket's donation)
every number downstream is wrong and nothing else in the suite notices: the
cumulative sums still look smooth, the fit still returns a value, and the
diagnostics still look plausible.  So it is gated directly, against the
un-bracketed kernel, on a mesh, at the level of the τ kernel itself.

Tiny by design: seconds, no deck, no preprocessing.  The two kernel cells run
the PRODUCTION τ kernel on a 1x1 mesh, in process.  They are deliberately NOT
``@pytest.mark.mesh(4)``: an in-process multi-device mesh that reaches the
flat-k FFT aborts uncatchably (``tests/harness`` says so by name, and it does
— rc=-6, no junit), so the P=4 form of exactly these two assertions is the
multi-process script ``tests/multi_device/band_bracket_partition_p4.py``.
One device is enough to catch a bracketing error, which is what this file is
for; the P=4 script is what certifies the sharded tile bookkeeping.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.band_degeneracy import (
    DEGENERACY_TOL_RY,
    boundary_min_gaps,
    snap_cut_to_clean_boundary,
)
from common.units import RYD_TO_EV
from gw.band_extrapolation import (
    BandExtrapolationRefused,
    build_band_ladder,
    fit_band_extrapolation_spectral,
    format_spectral_report,
    plan_band_brackets,
    trivial_plan,
)


def _spectral_fit(counts, S):
    """The shipped estimator on a free-electron ladder of 200 DFT bands.

    For the tests of the rulings the fit feeds (SC tolerance, static tail,
    report notes).  ``-1 + 30/N`` data gives β ≈ 3 and a tail of about
    0.5 eV at N₃ ≈ 50.
    """
    n = np.arange(1, 201, dtype=np.float64)
    enk_ry = ((-6.0 + 3.0 * n ** (2.0 / 3.0)) / RYD_TO_EV)[None, :]
    ladder = build_band_ladder(enk_ry=enk_ry, kweights=None, n_target=400)
    return fit_band_extrapolation_spectral(counts, S, ladder)


# ---------------------------------------------------------------------------
#  (2)  THE BRACKETS PARTITION THE BAND SUM  — the gate that matters
# ---------------------------------------------------------------------------

def _mesh_1x1():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))


def _skip_if_no_fft_handler(exc: Exception):
    """Skip on ABSENCE.  FAIL on a library that is built and will not load.

    The τ kernel is FFI-required (decisions.md 2026-08-01), so a platform
    whose ``liblorrax_ffi*.so`` has no flat-k / gw_conv handler cannot run
    this gate at all.  That is an ABSENCE, not a measurement — and it must be
    reported as the named absence rather than swallowed into a green run.

    ABSENT AND BROKEN ARE NOT THE SAME SKIP.  ``FfiLibraryUnusable`` used to
    be in the skip list beside ``FfiLibraryNotBuilt``, which collapsed "no
    build on this machine" (a legitimate gate) and "the build is present and
    the dynamic linker refused it" (a defect) into one green-looking result.
    That is the shape that hid 19 real failures in the linalg contract suite
    on 2026-08-06; ``ffi_loader`` has raised the two types separately since
    ``8352bcb`` and ``services/distrib_la/tests`` already branches on them.
    A skip here reads as "not applicable on this machine", which for a
    present-but-unloadable library is false, and a false not-applicable is
    worse than a failure because it stops anyone looking.
    """
    txt = f"{type(exc).__name__}: {exc}"
    if "FfiLibraryUnusable" in txt or "FfiAbiMismatch" in txt:
        raise AssertionError(
            f"the FFI library IS BUILT and will not load: {txt}\n"
            f"This is a DEFECT, not an unavailable platform, and it is "
            f"reported as a failure on purpose.  Most likely a DT_NEEDED "
            f"that cannot be resolved in this environment — check "
            f"`readelf -d <so> | grep NEEDED` against `ldd <so> | grep "
            f"'not found'` IN THE ENVIRONMENT THAT FAILED.") from exc
    for sig in ("FfiLibraryNotBuilt", "LORRAX_FFT_FFI", "handler"):
        if sig in txt:
            pytest.skip(f"no FFT FFI handler on this platform — {txt}")
    raise exc


def _operands(mesh, *, kgrid=(2, 2, 2), nb=12, n_mu=8, rng=None):
    """Use one logical wavefunction and a typed identity-group parent plan."""
    from multi_device.full_photon_head_sigma_gate import _bundle
    from gw.wavefunction_bundle import BandSlices, parent_sigma_operands, sigma_face_kernel_kwargs
    rng = rng or np.random.default_rng(20260815)
    nk = int(np.prod(kgrid))

    def c(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    psi = c(nk, nb, 1, n_mu)
    energy = np.abs(rng.standard_normal((nk, nb)))
    wfns = _bundle(mesh, psi, energy, np.zeros((nk, nb)),
        BandSlices.from_band_edges(0, 0, 0, 4, nb), kgrid=kgrid)
    xn, yr, xr, yn, energy, _ = parent_sigma_operands(wfns)
    q_spec = NamedSharding(mesh, P(None, 'x', 'y'))
    return dict(psi_xn=xn, psi_yr=yr, psi_xr=xr, psi_yn=yn, E_A=energy,
        mask_A=jnp.asarray(rng.random((nk, nb)) > 0.3),
        B_q=jax.device_put(c(nk, n_mu, n_mu), q_spec),
        Omega_q=jax.device_put(np.abs(rng.standard_normal((nk, n_mu, n_mu))) + 0.1, q_spec),
        mask_B=jax.device_put(np.ones((nk, n_mu, n_mu), dtype=bool), q_spec),
        face_kwargs=sigma_face_kernel_kwargs(wfns))


def _run_tau(mesh, op, brackets, *, kgrid=(2, 2, 2)):
    from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
    try:
        kern = get_shared_sigma_tau_kernel(
            mesh_xy=mesh, kgrid=kgrid, brackets=brackets, **op["face_kwargs"])
    except Exception as exc:                       # noqa: BLE001 — see helper
        _skip_if_no_fft_handler(exc)
    with mesh:
        out = kern(
            op['psi_xn'], op['psi_yr'], op['psi_xr'], op['psi_yn'],
            op['E_A'], op['mask_A'],
            jnp.where(op['mask_B'], op['B_q'], 0.0)[None, ...],
            op['Omega_q'].astype(jnp.complex128)[None, ...],
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([[0.0, np.inf, -np.inf, -np.inf,
                          np.inf, np.inf]], dtype=jnp.float64),
            jnp.asarray([False]),
            jnp.asarray(0.25, dtype=jnp.float64),
            jnp.asarray(0.10, dtype=jnp.float64),
            jnp.asarray(0.3 - 0.7j, dtype=jnp.complex128),
        )
    return jax.tree.map(lambda a: np.asarray(jax.device_get(a)), out)


def test_brackets_partition_the_band_sum():
    """cumsum(brackets)[-1] == the un-bracketed full-band σ^τ, to roundoff.

    THE gate for the feature, through the sole shared complex-carrier kernel.
    """
    mesh = _mesh_1x1()
    nb = 12
    op = _operands(mesh, nb=nb)

    one = _run_tau(mesh, op, ((0, nb),))
    three = _run_tau(mesh, op, ((0, 5), (5, 9), (9, nb)))

    assert one.shape[0] == 1, "the ordinary path must carry a length-1 axis"
    assert three.shape[0] == 3
    assert one.shape[1:] == three.shape[1:]
    total = np.cumsum(three, axis=0)[-1]
    ref = one[0]
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    err = float(np.max(np.abs(total - ref))) / scale
    assert err < 1e-12, (
        f"bracket cumulative sum != full-band sum: rel {err:.3e}.  "
        f"The brackets do not partition the band sum.")


def test_single_bracket_is_bit_identical_to_the_full_band_kernel():
    """The length-1 leading axis must not perturb a single bit.

    Same kernel, brackets=((0, nb),), compared against the identical
    computation reached through the MPA-shaped ``brackets=None`` entry.
    ``max|Δ| == 0`` exactly — not a tolerance.
    """
    from gw.ppm_tau_kernel import _get_sigma_kij_kernel
    mesh = _mesh_1x1()
    nb = 12
    op = _operands(mesh, nb=nb)
    args = (op['psi_xn'], op['psi_yr'], op['psi_xr'], op['psi_yn'],
            op['E_A'], op['mask_A'],
            jnp.asarray(0.25, dtype=jnp.float64),
            jnp.asarray(0.3 - 0.7j, dtype=jnp.complex128),
            op['B_q'])
    try:
        k_none = _get_sigma_kij_kernel(
            mesh_xy=mesh, kgrid=(2, 2, 2), merged_x=True, brackets=None, **op["face_kwargs"])
        k_one = _get_sigma_kij_kernel(
            mesh_xy=mesh, kgrid=(2, 2, 2), merged_x=True, brackets=((0, nb),), **op["face_kwargs"])
    except Exception as exc:                       # noqa: BLE001 — see helper
        _skip_if_no_fft_handler(exc)
    with mesh:
        a = np.asarray(jax.device_get(k_none(*args)))
        b = np.asarray(jax.device_get(k_one(*args)))
    assert b.shape == (1,) + a.shape
    assert np.array_equal(a, b[0]), (
        f"length-1 bracket axis changed the result: "
        f"max|d| = {np.max(np.abs(a - b[0])):.3e}")


# ---------------------------------------------------------------------------
#  (3)  degeneracy-clean snapping, against actual eigenvalues
# ---------------------------------------------------------------------------

def _si_like_spectrum():
    """A spectrum with EXACT multiplets at known places.

    Bands 0..1, 2..4 and 8..11 are exactly degenerate at every k, so the
    boundaries strictly inside those blocks are the SPLIT ones and every
    other boundary is CLEAN.  Built rather than read so the expected answer
    is a fact about the array, not about a fixture.
    """
    nk, nb = 6, 16
    base = np.cumsum(np.full(nb, 0.5)) + 1.0
    e = np.tile(base, (nk, 1))
    for blk in ((0, 2), (2, 5), (8, 12)):
        e[:, blk[0]:blk[1]] = e[:, blk[0]][:, None]
    # A little k dispersion on the non-degenerate bands only.
    free = [b for b in range(nb) if b not in (1, 3, 4, 9, 10, 11)]
    e[:, free] += 0.01 * np.arange(nk)[:, None]
    return np.sort(e, axis=1)


def test_snapped_cuts_are_clean_against_the_actual_eigenvalues():
    e = _si_like_spectrum()
    gaps = boundary_min_gaps(e, is_full_spectrum=True)  # fixture IS the whole spectrum
    clean = {b for b in range(1, e.shape[1]) if gaps[b] > DEGENERACY_TOL_RY}
    split = {b for b in range(1, e.shape[1])} - clean
    assert split, "the fixture must contain SPLIT boundaries to be a test"
    for req in sorted(split):
        got = snap_cut_to_clean_boundary(e, req, lo=1, hi=e.shape[1] - 1)
        assert got in clean, f"snap({req}) -> {got}, which splits a multiplet"
        assert abs(got - req) <= 3
    for req in sorted(clean - {e.shape[1]}):
        if 1 <= req <= e.shape[1] - 1:
            assert snap_cut_to_clean_boundary(
                e, req, lo=1, hi=e.shape[1] - 1) == req, \
                "an already-clean cut must not move"


def test_plan_cuts_are_clean_and_cover_every_band():
    e = _si_like_spectrum()
    nb = e.shape[1]
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=2, nb_logical=nb, nb_padded=nb)
    gaps = boundary_min_gaps(e, is_full_spectrum=True)  # fixture IS the whole spectrum
    for c in plan.counts[:-1]:
        assert gaps[c] > DEGENERACY_TOL_RY, f"cut {c} splits a multiplet"
    assert not plan.notes, "a clean boundary was available; nothing to note"
    assert list(plan.counts) == sorted(set(plan.counts))
    # Contiguous, and covering [0, nb_padded) exactly — the partition claim
    # at the level of the plan, before any kernel is involved.
    assert plan.bounds[0][0] == 0
    assert plan.bounds[-1][1] == nb
    for (a, b), (c, d) in zip(plan.bounds, plan.bounds[1:]):
        assert b == c, f"bracket gap/overlap between {(a, b)} and {(c, d)}"


def test_fractions_are_of_the_total_band_count_not_the_conduction_count():
    """The regression guard for the 2026-08-15 parametrisation change.

    ``N_i = f * N_max``, NOT ``n_occ + f * n_cond``.  The two agree to ~2 %
    when the occupied manifold is small — which is exactly why this needs a
    fixture where it is LARGE, or the test would pass either way.  The
    counting law that makes 1/N the right variable is written in the total
    (module docstring: p = 1.481 measured for the total against 1.212 for
    the conduction-only form), and the fit's lever arm is 1/N1 - 1/N3, which
    is a ratio of totals.
    """
    nk, nb = 4, 200
    n_occ = 80                                  # 40 % occupied: the two rules
    e = np.tile(np.linspace(1.0, 20.0, nb), (nk, 1))   # differ by 24 bands
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=n_occ, nb_logical=nb, nb_padded=nb,
        fractions=(0.80, 0.90))
    assert plan.counts == (160, 180, 200), plan.counts
    conduction_rule = tuple(
        int(round(n_occ + f * (nb - n_occ))) for f in (0.80, 0.90))
    assert conduction_rule == (176, 188)
    assert plan.counts[:2] != conduction_rule, \
        "fractions must be of N_max, not of n_cond"


def test_the_conduction_coordinate_is_a_named_spelling_of_the_same_fractions():
    """The owner's 2026-08-18 coordinate, reachable ONLY by name.

    Same fixture as the cell above (40 % occupied, so the two rules differ by
    24 bands): ``conduction_fractions`` must produce exactly the counts that
    cell asserts the DEFAULT must not produce.  Two cells, one fixture,
    opposite assertions — which is what makes either of them evidence.
    """
    nk, nb = 4, 200
    n_occ = 80
    e = np.tile(np.linspace(1.0, 20.0, nb), (nk, 1))
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=n_occ, nb_logical=nb, nb_padded=nb,
        fractions=(0.80, 0.90), bracket_scheme="conduction_fractions")
    assert plan.counts == (176, 188, 200), plan.counts
    assert plan.bracket_scheme == "conduction_fractions"


def test_both_coordinates_come_from_one_conversion_with_no_default():
    """``bracket_counts_from_fractions`` is the only place either rule lives.

    ``coordinate`` is keyword-only with NO default: at a call site "0.80" in
    the two coordinates is the same three characters and a different band.
    """
    import inspect

    from gw.band_extrapolation import bracket_counts_from_fractions as conv

    sig = inspect.signature(conv)
    assert sig.parameters["coordinate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["coordinate"].default is inspect.Parameter.empty, (
        "a defaulted coordinate is an inferred coordinate")
    assert conv((0.80, 0.90), 80, 200, coordinate="total_fractions") == (
        160, 180)
    assert conv((0.80, 0.90), 80, 200,
                coordinate="conduction_fractions") == (176, 188)
    # A scheme that consumes no fractions must not be answerable here.
    with pytest.raises(ValueError, match="consumes no fractions"):
        conv((0.80,), 80, 200, coordinate="conduction_energy_midpoint")


def test_an_unknown_bracket_scheme_refuses_by_name():
    """No fallback: this key selects which three band sums are COMPUTED."""
    e = _si_like_spectrum()
    with pytest.raises(ValueError, match="not known"):
        plan_band_brackets(
            enabled=True, enk_ry=e, n_occ=2, nb_logical=e.shape[1],
            nb_padded=e.shape[1], bracket_scheme="conduction_maybe")


def test_conduction_energy_midpoint_is_conduction_relative_and_rectangular():
    """The opt-in geometry is exactly the measured proposal.

    N1 is half of the INCLUDED conduction bands, not half of all bands.  N2
    is the nearest rectangular count in the k-mean DFT boundary-energy
    ladder, not an E_ck mask with a different count at every k.
    """
    nb, n_occ = 200, 80
    band = np.arange(nb, dtype=float)
    # Non-linear spacing makes the energy midpoint visibly different from
    # the equal-band-count midpoint; k offsets cancel only after the mean.
    e = np.stack((band ** 2, band ** 2 + 0.2 * band,
                  band ** 2 + 10.0), axis=0)
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=n_occ,
        nb_logical=nb, nb_padded=nb,
        bracket_scheme="conduction_energy_midpoint")

    assert plan.bracket_scheme == "conduction_energy_midpoint"
    assert plan.counts[0] == n_occ + round(0.5 * (nb - n_occ)) == 140
    ebar = np.mean(e, axis=0)
    target = 0.5 * (ebar[plan.counts[0] - 1] + ebar[nb - 1])
    expected_n2 = min(
        range(plan.counts[0] + 1, nb),
        key=lambda n: (abs(ebar[n - 1] - target), n))
    assert plan.counts[1] == expected_n2
    assert plan.counts[1] != round(0.5 * (plan.counts[0] + nb)), \
        "the middle is in ENERGY, not in band index"
    assert len(plan.bounds) == 3, "one rectangular bracket plan serves all k"
    assert np.allclose(
        plan.boundary_mean_energy_ev,
        [ebar[n - 1] * RYD_TO_EV for n in plan.counts])


def test_conduction_energy_midpoint_chooses_nearest_clean_energy_boundary():
    """Energy proximity is optimized over clean boundaries, after N1 snaps."""
    nk, nb, n_occ = 4, 40, 10
    band = np.arange(nb, dtype=float)
    e = np.tile(band ** 2, (nk, 1))
    # Raw N1 is 25.  Put it inside a four-band multiplet, so the actual
    # midpoint must be recomputed from the snapped N1 rather than the raw one.
    e[:, 23:27] = e[:, 23][:, None]
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=n_occ,
        nb_logical=nb, nb_padded=nb,
        bracket_scheme="conduction_energy_midpoint")
    gaps = boundary_min_gaps(e, is_full_spectrum=True)
    assert plan.counts[0] != 25, "the fixture must move the raw N1"
    assert all(gaps[n] > DEGENERACY_TOL_RY for n in plan.counts[:-1])
    ebar = np.mean(e, axis=0)
    target = 0.5 * (ebar[plan.counts[0] - 1] + ebar[nb - 1])
    candidates = [n for n in range(plan.counts[0] + 1, nb)
                  if gaps[n] > DEGENERACY_TOL_RY]
    expected = min(candidates,
                   key=lambda n: (abs(ebar[n - 1] - target), n))
    assert plan.counts[1] == expected


def test_default_bracket_scheme_is_the_legacy_total_fraction_plan():
    """Adding a named geometry must not reinterpret an existing deck."""
    e = _si_like_spectrum()
    kw = dict(enabled=True, enk_ry=e, n_occ=2,
              nb_logical=e.shape[1], nb_padded=e.shape[1])
    implicit = plan_band_brackets(**kw)
    explicit = plan_band_brackets(**kw, bracket_scheme="total_fractions")
    assert implicit == explicit
    assert implicit.bracket_scheme == "total_fractions"


def test_pad_bands_land_in_the_last_bracket():
    """A mesh-padded band axis must still be covered end to end."""
    e = _si_like_spectrum()
    nb = e.shape[1]
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=2, nb_logical=nb, nb_padded=nb + 3)
    assert plan.bounds[-1][1] == nb + 3
    assert plan.counts[-1] == nb, "counts must stay LOGICAL (pads sum to zero)"


# ---------------------------------------------------------------------------
#  (4)  the refusal
# ---------------------------------------------------------------------------

def test_unsnappable_interior_cut_falls_back_instead_of_refusing():
    """A spectrum with no clean boundary near the cut must NOT stop the run.

    The interior cuts are sampling points on a partial-sum curve, not Σ
    windows: only N3 is a band sum anything downstream reports.  MEASURED
    cost of a cut that splits a multiplet: <= 6.4 meV on the Si 4x4x4 SOC
    deck.  Under SOC every Kramers pair is exactly degenerate, so clean
    boundaries are sparse by construction and the old refusal fired on decks
    whose unsnapped points work fine.  Trading <= 6.4 meV for a dead run is
    the wrong trade, so this asserts the trade is not made.
    """
    # One exactly-degenerate block running to the TOP of the range, so the
    # second interior cut has no clean boundary to reach at all.  (It has to
    # reach the top: leave any clean boundary above the cut and the snapper
    # finds it, and this stops testing the fallback.)
    nk, nb = 4, 40
    e = np.tile(np.linspace(1.0, 10.0, nb), (nk, 1))
    e[:, 32:] = e[:, 32][:, None]
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=4, nb_logical=nb, nb_padded=nb,
        fractions=(0.80, 0.90))
    assert plan.counts == (32, 36, 40), plan.counts
    assert list(plan.counts) == sorted(set(plan.counts))
    assert plan.notes, "the fallback must be reported, not silent"
    joined = " ".join(plan.notes)
    assert "UNSNAPPED" in joined
    assert "36" in joined, "the note must name the cut it kept"
    # And the report block carries it, so it reaches the log either way.
    N = np.asarray(plan.counts, dtype=float)
    fit = _spectral_fit(plan.counts, (-1.0 + 30.0 / N)[:, None, None])
    assert "NOTE:" in format_spectral_report(plan, fit)


def test_snapping_never_starves_a_later_cut():
    """Snapping cut i upward must not consume the room cut i+1 needs.

    Found by an exhaustive n_occ x nband sweep against the real Si deck
    eigenvalues: at nband = 17 the first cut snapped 14 -> 16 = nb_logical-1,
    the second had nowhere legal left, and the plan came back NON-ASCENDING
    (16, 15, 17) and refused — on a spectrum with three perfectly good
    sampling points in it.  The planner now reserves one band per remaining
    interior cut.
    """
    nk, nb = 4, 17
    e = np.tile(np.linspace(1.0, 10.0, nb), (nk, 1))
    e[:, 13:16] = e[:, 13][:, None]        # multiplet straddling the top cut
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=5, nb_logical=nb, nb_padded=nb,
        fractions=(0.80, 0.90))
    assert list(plan.counts) == sorted(set(plan.counts)), plan.counts
    assert plan.counts[-1] == nb
    assert all(plan.n_occ < c < nb for c in plan.counts[:-1])


def _kramers_ladder(nk, nb):
    """Every band doubled at every k, as under SOC with inversion: each odd
    count splits a Kramers pair everywhere."""
    base = np.linspace(1.0, 10.0, nb // 2)
    return np.repeat(base, 2)[None, :].repeat(nk, axis=0).copy()


def test_snapping_never_crosses_the_next_request():
    """A snapped cut must not pass the next cut's request (lane BX2).

    The Si 6x6x6 SOC deck at 64 bands: the only 1-meV-clean boundaries in
    39..63 were 40 and 60.  Cut 1 snapped 51 -> 60, past cut 2's request of
    58, and cut 2 was left at 61 -- shells of one and three bands at the top
    of the window, a Kramers split at every k, and Gamma valence states moved
    by -4.6 to -8.3 eV.  This fixture has the same clean set.
    """
    nk, nb = 4, 64
    e = _kramers_ladder(nk, nb)
    e[0, 40:60] = e[0, 40]            # one k: pairs 20..29 degenerate
    e[0, 60:64] = e[0, 60]            # and pairs 30, 31
    gaps = boundary_min_gaps(e, is_full_spectrum=True)
    clean = [n for n in range(39, 64) if gaps[n] > DEGENERACY_TOL_RY]
    assert clean == [40, 60], clean
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=8, nb_logical=nb, nb_padded=nb)
    assert plan.requested == (51, 58, 64)
    assert plan.counts == (40, 60, 64), plan.counts
    assert not plan.notes


def test_the_fallback_takes_the_least_degenerate_boundary():
    """With no clean boundary in range, a near miss beats a Kramers split.

    The old fallback kept the raw request.  At an odd request under SOC that
    is a split pair at every k, which makes S(N) gauge-dependent: Gamma
    quartet partners differed by 1e-4 eV in S(61) and by 3.6 eV after the
    extrapolation (lane BX2).
    """
    nk, nb = 4, 40
    e = _kramers_ladder(nk, nb)
    half_mev = 0.5e-3 / 13.605693122994
    e[0, 32:40] = e[0, 32] + half_mev * np.repeat(np.arange(4), 2)
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=4, nb_logical=nb, nb_padded=nb,
        fractions=(0.80, 0.875))
    assert plan.requested == (32, 35, 40)
    assert plan.counts == (32, 34, 40), plan.counts
    joined = " ".join(plan.notes)
    assert "UNSNAPPED" in joined and "35 -> 34" in joined


def test_refuses_only_when_the_fractions_themselves_collapse():
    """The surviving refusal names the SIGMA count, because that is what
    fixes it.  Since the chi/Sigma split (2026-08-16) "raise nband" is not
    actionable advice on a deck whose two counts differ, and raising the CHI
    count would not move a single bracket."""
    nk, nb = 4, 3
    e = np.tile(np.linspace(1.0, 2.0, nb), (nk, 1))
    with pytest.raises(BandExtrapolationRefused) as exc:
        plan_band_brackets(enabled=True, enk_ry=e, n_occ=1,
                           nb_logical=nb, nb_padded=nb,
                           fractions=(0.80, 0.90))
    msg = str(exc.value)
    assert "DISTINCT" in msg and "number_bands_sigma" in msg
    assert "number_bands_chi" in msg, (
        "the refusal must say which count does NOT fix it")
    assert "0.8" in msg, "the refusal must show the fractions it used"


def test_refuses_when_ncond_below_nocc():
    """The hard gate: nband >= 2*n_occ, i.e. n_cond >= n_occ.

    Owner ruling 2026-08-16, whose words were "kill the calculation if the
    number of bands requested is not >= 2*N_electrons".  ``n_occ`` is the
    spin-convention-independent spelling of ``N_electrons`` (see the
    companion assertion on the message below), and the refusal has to say so
    or an operator on a non-SOC deck will double the wrong number.
    """
    e = _si_like_spectrum()
    nb = e.shape[1]                       # 16
    n_occ = nb // 2 + 1                   # 9 -> n_cond = 7 < 9
    with pytest.raises(BandExtrapolationRefused) as exc:
        plan_band_brackets(enabled=True, enk_ry=e, n_occ=n_occ,
                           nb_logical=nb, nb_padded=nb)
    msg = str(exc.value)
    assert "use_band_extrapolation" in msg
    assert "n_cond" in msg and "n_occ" in msg
    # ── WHICH KNOB (split branch) ───────────────────────────────────────
    # After the chi/sigma split "nband" alone is not an actionable name, so
    # the refusal must say WHICH count it is talking about -- and must say
    # that the other one will not help.  Merge ruling 2026-08-16.
    assert "number_bands_sigma" in msg, \
        "the refusal must name the knob that fixes it"
    assert "number_bands_chi" in msg, \
        "...and the one that does not"
    # ── WHICH THRESHOLD (SC branch) ─────────────────────────────────────
    assert str(2 * n_occ) in msg, "the refusal must state the band count needed"
    # THE MAPPING IS PART OF THE REFUSAL, not just of the docs: the owner's
    # rule is in N_electrons and the gate is in n_occ, and those differ by a
    # factor of two depending on whether the deck has SOC.
    assert "N_electrons" in msg
    assert "SOC" in msg


def test_ncond_equal_nocc_is_allowed_and_is_the_boundary():
    """``n_cond == n_occ`` RUNS.  This is the 2026-08-16 relaxation.

    The gate that shipped before refused on ``n_cond <= n_occ`` (strictly
    greater), which is one band tighter than the owner asked for.  The two
    thresholds are now one threshold, and this test pins which side of it the
    equality case falls on -- the exact band count a reader of
    ``nband >= 2*n_occ`` would expect to be legal.
    """
    e = _si_like_spectrum()
    nb = e.shape[1]                       # 16
    n_occ = nb // 2                       # 8 -> n_cond = 8 == n_occ
    plan = plan_band_brackets(enabled=True, enk_ry=e, n_occ=n_occ,
                              nb_logical=nb, nb_padded=nb)
    assert plan.enabled is True
    assert plan.n_cond == plan.n_occ == n_occ
    assert plan.counts[-1] == nb
    assert len(set(plan.counts)) == 3, "three distinct ascending counts"


def test_the_two_refusals_quote_the_same_band_floor():
    """One gate, one threshold -- not two.

    ``plan_band_brackets`` can refuse twice: on the band-count gate, and on
    the three fractions failing to resolve into distinct counts.  Before
    2026-08-16 the second quoted ``2*n_occ + 1`` while the first quoted
    ``2*n_occ + 1`` too -- both one band tighter than the rule.  A deck that
    satisfies the message it was given must not then hit the other refusal
    with a different number.
    """
    nk, nb = 4, 3
    e = np.tile(np.linspace(1.0, 2.0, nb), (nk, 1))
    n_occ = 1
    with pytest.raises(BandExtrapolationRefused) as exc:
        plan_band_brackets(enabled=True, enk_ry=e, n_occ=n_occ,
                           nb_logical=nb, nb_padded=nb,
                           fractions=(0.80, 0.90))
    # The collapse refusal's floor must never be BELOW the activation gate's,
    # or raising nband to satisfy it would land on the other refusal.
    assert f"at least {max(10, 2 * n_occ)}" in str(exc.value) or \
        "at least" in str(exc.value)
    # The activation gate's own floor, on the same n_occ, is 2*n_occ.
    e2 = _si_like_spectrum()
    with pytest.raises(BandExtrapolationRefused) as exc2:
        plan_band_brackets(enabled=True, enk_ry=e2, n_occ=9,
                           nb_logical=16, nb_padded=16)
    assert "18" in str(exc2.value), "2*n_occ = 18, not 2*n_occ + 1"


def test_disabled_returns_the_trivial_single_bracket_plan():
    e = _si_like_spectrum()
    nb = e.shape[1]
    plan = plan_band_brackets(
        enabled=False, enk_ry=e, n_occ=nb // 2, nb_logical=nb, nb_padded=nb)
    assert plan.enabled is False
    assert plan.n_brackets == 1
    assert plan.bounds == ((0, nb),)
    assert plan.counts == (nb,)
    ref = trivial_plan(nb, nb // 2, nb)
    assert (plan.bounds, plan.counts, plan.requested,
            plan.n_occ, plan.n_cond, plan.enabled) == (
        ref.bounds, ref.counts, ref.requested,
        ref.n_occ, ref.n_cond, ref.enabled)
    # mean_energy_ev is NaN on the trivial plan (nothing was cut, so there is
    # no cutoff-flavoured number to report) and NaN != NaN, so it is compared
    # by predicate rather than by equality.
    assert all(np.isnan(v) for v in plan.mean_energy_ev)
    assert plan.notes == (), "the trivial plan decides nothing worth noting"


def _cohsex_dispatch(explicit, print_fn=None, *, scheme_explicit=False):
    import types
    from gw.gw_config import ComputeMode
    from gw.sigma_dispatch import compute_sigma_xc

    cfg = types.SimpleNamespace(
        sigma=types.SimpleNamespace(
            band_extrapolation=True,
            band_extrapolation_explicit=explicit,
            band_extrapolation_bracket_scheme=(
                "conduction_energy_midpoint" if scheme_explicit
                else "total_fractions"),
            band_extrapolation_bracket_scheme_explicit=scheme_explicit))
    kw = {} if print_fn is None else {"print_fn": print_fn}
    return compute_sigma_xc(
        ComputeMode.COHSEX, wfns=None, V_q=None, W_by_role={},
        e_qp_ev=None, static_head_terms=None, head_resolver=None,
        quad=None, config=cfg, meta=None, mesh_xy=None, sym=None,
        wfn=None, band_slices=None, input_dir=".",
        material_class="insulator", **kw)


def test_dispatch_refuses_an_EXPLICIT_key_on_a_non_ppm_mode():
    """A key the DECK NAMED, that no kernel reads, must refuse."""
    with pytest.raises(NotImplementedError) as exc:
        _cohsex_dispatch(explicit=True)
    msg = str(exc.value)
    assert "use_band_extrapolation" in msg
    # The refusal must say the default would have behaved differently, or an
    # operator cannot tell why their neighbour's COHSEX run did not refuse.
    assert "default" in msg


def test_dispatch_also_refuses_an_explicit_bracket_scheme_with_no_ppm_stage():
    """A named compute geometry is no less explicit than the ON switch."""
    with pytest.raises(NotImplementedError) as exc:
        _cohsex_dispatch(explicit=False, scheme_explicit=True)
    msg = str(exc.value)
    assert "conduction_energy_midpoint" in msg
    assert "NO stage of this run consumes it" in msg


def test_dispatch_AUTO_DISABLES_a_defaulted_key_on_a_non_ppm_mode():
    """A DEFAULTED key on a static mode disables itself and says so.

    This is what makes a default-on key coherent with the pre-existing
    non-PPM refusal instead of fighting it.  The physics guard is the same in
    both branches -- no static-mode Sigma is ever extrapolated -- but
    refusing a run over a default the operator never chose would make every
    COHSEX / MPA / X_ONLY deck in the tree unrunnable.

    It must NOT be silent: the log line is the only record that this stage
    did not extrapolate.
    """
    said = []
    # It gets past the extrapolation guard and dies later on the None
    # operands, which is exactly the point -- the guard did not stop it.
    with pytest.raises(Exception) as exc:
        _cohsex_dispatch(explicit=False, print_fn=said.append)
    assert not isinstance(exc.value, NotImplementedError) or \
        "use_band_extrapolation" not in str(exc.value), (
            "a defaulted key must not refuse on a non-PPM mode")
    blob = "\n".join(said)
    assert "AUTO-DISABLED" in blob, "the auto-disable must be recorded"
    assert "288.2" in blob, (
        "the note must carry the measurement that justifies it -- the static "
        "COHSEX arm ANTI-converging 94.9 -> 288.2 meV")


# ---------------------------------------------------------------------------
#  the SC coupling: extrapolate Sigma, THEN diagonalize
# ---------------------------------------------------------------------------


def test_weights_are_real_and_affine():
    """REAL (so Hermiticity survives) and summing to 1 (so a converged Sigma
    passes through unchanged rather than being rescaled)."""
    from gw.band_extrapolation import extrapolation_weights

    for counts in ([100, 112, 124], [40, 45, 50], [12, 14, 16]):
        w = extrapolation_weights(counts)
        assert w.dtype == np.float64, "complex weights would break Hermiticity"
        assert abs(float(w.sum()) - 1.0) < 1e-12, \
            "an affine combination: sum(c) == 1"


def test_eigenvalue_extrapolation_is_not_the_same_operation():
    """Why the order is not a matter of taste.

    Extrapolating the SPECTRUM of each bracket gives a different answer from
    extrapolating Sigma and diagonalizing once, because eigenvalues are not
    linear in the matrix.  The rejected order also corresponds to no
    Hamiltonian at all.  This test does not assert which is "better" -- it
    asserts they DIFFER, so that the choice recorded in
    ``extrapolation_weights`` is a real one and cannot be quietly inverted.
    """
    from gw.band_extrapolation import extrapolation_weights

    rng = np.random.default_rng(99)
    nb = 5
    pts = []
    for scale in (1.0, 1.3, 1.55):
        A = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        H = 0.5 * (A + np.conj(A.T)) * scale
        pts.append(H)
    cube = np.stack(pts)
    w = extrapolation_weights([100, 112, 124])

    e_then_x = np.tensordot(
        w, np.stack([np.linalg.eigvalsh(H) for H in cube]), axes=(0, 0))
    x_then_e = np.linalg.eigvalsh(np.tensordot(w, cube, axes=(0, 0)))
    assert not np.allclose(e_then_x, x_then_e, atol=1e-8), \
        "if these agreed, the ordering ruling would be vacuous"


# ---------------------------------------------------------------------------
#  the deck key, its deprecated alias, and the tolerance ruling
# ---------------------------------------------------------------------------

def test_use_band_extrapolation_defaults_true():
    from gw.gw_config import (
        USE_BAND_EXTRAPOLATION_DEFAULT, resolve_band_extrapolation)
    assert USE_BAND_EXTRAPOLATION_DEFAULT is True
    enabled, explicit = resolve_band_extrapolation(None, None)
    assert enabled is True, "the feature runs by default"
    assert explicit is False, "nobody named it -- that is what lets a static "\
        "mode auto-disable instead of refusing"


def test_deprecated_alias_still_drives_the_feature():
    """Committed decks written before the rename must keep working."""
    from gw.gw_config import resolve_band_extrapolation
    for alias_val in (True, False):
        enabled, explicit = resolve_band_extrapolation(None, alias_val)
        assert enabled is alias_val
        assert explicit is True, "an alias the deck NAMED is still explicit"


def test_both_keys_disagreeing_refuses_by_name():
    """No winner is picked -- both names appear in the refusal."""
    from gw.gw_config import resolve_band_extrapolation
    for a, b in ((True, False), (False, True)):
        with pytest.raises(ValueError) as exc:
            resolve_band_extrapolation(a, b)
        msg = str(exc.value)
        assert "use_band_extrapolation" in msg
        assert "sigma_band_extrapolation" in msg
        assert "DISAGREE" in msg


def test_both_keys_agreeing_is_accepted():
    from gw.gw_config import resolve_band_extrapolation
    for v in (True, False):
        enabled, explicit = resolve_band_extrapolation(v, v)
        assert enabled is v and explicit is True


def test_sc_tolerance_ruling_warns_when_the_tolerance_is_inside_the_bar():
    """The single most important robustness point: it must not be SILENT."""
    from gw.band_extrapolation import sc_tolerance_ruling, tolerance_bar_ev

    N = np.array([100, 112, 124])
    # A correction of ~0.17 eV -> a p90 bar of ~25 meV at 15 %.
    fit = _spectral_fit(N, (-1.0 + 30.0 / N)[:, None])
    med, mx = tolerance_bar_ev(fit)
    assert med > 0.0 and mx >= med

    inside, text = sc_tolerance_ruling(fit, 1.0e-4)     # the SHIPPED default
    assert inside is True, "0.1 meV is inside a tens-of-meV bar"
    assert "***" in text, "an unmissable marker, not a footnote"
    assert "sc_tol_ev" in text and "meV" in text
    # It must report BOTH numbers, per iteration, and say which is which.
    assert "median" in text and "max" in text
    # And it must tell the reader what to quote instead.
    assert "do NOT quote" in text

    outside, text2 = sc_tolerance_ruling(fit, 10.0)     # 10 eV, absurdly loose
    assert outside is False
    assert "***" not in text2


def test_sc_tolerance_ruling_does_not_refuse():
    """It WARNS.  A refusal would fire on the shipped default configuration.

    ``sc_tol_ev`` defaults to 1e-4 eV and ``use_band_extrapolation`` now
    defaults to True, so "tolerance inside the bar" is the DEFAULT state of
    the code.  A gate that refuses its own default is not a safety property.
    """
    from gw.band_extrapolation import sc_tolerance_ruling
    N = np.array([100, 112, 124])
    fit = _spectral_fit(N, (-1.0 + 30.0 / N)[:, None])
    inside, text = sc_tolerance_ruling(fit, 1.0e-9)
    assert inside is True
    assert isinstance(text, str) and text            # returned, not raised


# ---------------------------------------------------------------------------
#  A LEADING nspin AXIS ON THE OCCUPANCY MASK  — the 1x1-mesh slice defect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("brackets", [((0, 12),), ((0, 5), (5, 9), (9, 12))])
def test_mask_with_a_leading_nspin_axis_gives_the_same_bracketed_sum(brackets):
    """A ``(1, nk, nb)`` mask must slice on its BAND axis, not on ``nk``.

    THE DEFECT THIS PINS, and why it was in the DEFAULT path.  ``_bracketed``
    restricted the band range with ``mask_A[:, lo:hi]``.  That is the band axis
    only while the mask is 2-D ``(nk, nb)``, which is what a 2x2 processor mesh
    delivers because it squeezes the leading nspin axis.  A 1x1 mesh does NOT
    squeeze it, so the mask arrives ``(1, nk, nb)`` and ``[:, lo:hi]`` cut
    ``nk`` instead -- and the G build then died inside ``build_G_tau``'s shape
    normalisation with ``cannot reshape (1, 9, 52) into (9, 42)``.

    ``brackets=((0, nb),)`` is parametrised deliberately: the trivial
    SINGLE-bracket plan is the ordinary non-extrapolated Sigma_c, so this was
    never a band-extrapolation bug.  It was reproduced on clean ``97f6f544``
    and it broke any GN-PPM run that reached ``_bracketed`` on one GPU.

    The assertion is EXACT equality, not a tolerance: the two masks carry the
    same bits, so a correct slice makes the two runs the same program.
    """
    mesh = _mesh_1x1()
    nb = 12
    # nk > nb ON PURPOSE.  The bad slice was ``mask_A[:, lo:hi]`` on a
    # ``(1, nk, nb)`` array, which cuts nk to (hi-lo).  With nk <= nb that is
    # a no-op for the trivial bracket ``(0, nb)`` and the SINGLE-bracket arm
    # silently passes even against the defect -- verified: at 9444c724 with
    # nk=8, nb=12 the 3-bracket arm failed and the 1-bracket arm did not.
    # The reported single-bracket reproduction was ``(1, 20, 20) -> (64, 20)``,
    # i.e. nk=64 against nb=20, so nk > nb is the condition that makes the
    # DEFAULT path observable.  Both arms must fail at base and pass here.
    # kgrid must multiply out to nk -- the gw_conv FFI checks it.
    op = _operands(mesh, kgrid=(4, 2, 2), nb=nb)
    two_d = _run_tau(mesh, op, brackets, kgrid=(4, 2, 2))

    op3 = dict(op)
    op3['mask_A'] = jnp.reshape(op['mask_A'], (1,) + op['mask_A'].shape)
    assert op3['mask_A'].ndim == 3
    three_d = _run_tau(mesh, op3, brackets, kgrid=(4, 2, 2))

    assert two_d.shape == three_d.shape, (
        f"a leading nspin axis on the mask changed the output SHAPE: "
        f"{two_d.shape} vs {three_d.shape} -- the slice cut the wrong axis.")
    assert np.array_equal(two_d, three_d), (
        f"a leading nspin axis on the mask changed the VALUES: "
        f"max|delta| = {float(np.max(np.abs(three_d - two_d))):.3e}")


# ---------------------------------------------------------------------------
#  THE STATIC-LIMIT CONTAMINANT  — the check the per-compute_mode guard cannot
#  make, because the compute_mode genuinely IS gn_ppm
# ---------------------------------------------------------------------------

def _fit_for_static_tests():
    N = np.array([42, 46, 52])
    return N, _spectral_fit(N, (-1.0 + 30.0 / N)[:, None])


def test_a_constant_offset_moves_S_inf_and_nothing_else():
    """Why folding the static-limit term into bracket 0 is the RIGHT fold.

    ``ppm_sigma`` adds the invalid-pole static-COHSEX term to bracket 0 only,
    so after the cumulative sum it is a CONSTANT on the three band-count
    points.  This estimator is affine, so a constant reaches ``S_inf`` 1:1 and
    perturbs no diagnostic -- which makes the fold exactly equivalent to
    "extrapolate the dynamical part, then add the static part back".  That
    equivalence is the whole reason a static Coulomb hole inside a GN-PPM
    Sigma does not get run through the 1/N law, and it is pinned here so a
    future editor cannot "fix" the fold into a per-bracket one.
    """
    N, fit = _fit_for_static_tests()
    C = 0.37
    fit2 = _spectral_fit(N, fit.s_at_counts + C)
    assert np.max(np.abs((fit2.s_inf - fit.s_inf) - C)) < 1e-12
    for nm, a, b in (("beta", fit.beta, fit2.beta),
                     ("delta_tail", fit.delta_tail, fit2.delta_tail)):
        assert np.max(np.abs(np.asarray(b) - np.asarray(a))) < 1e-12, nm


def test_static_limit_ruling_is_silent_on_a_band_independent_term():
    """No band dependence, nothing omitted -- the ruling must not cry wolf."""
    from gw.band_extrapolation import static_limit_tail_ruling
    N, fit = _fit_for_static_tests()
    flat = np.stack([np.full_like(np.asarray(fit.s_inf), -0.30)
                     for _ in N])
    exceeds, text, stats = static_limit_tail_ruling(fit, flat)
    assert exceeds is False
    assert stats["span_max_ev"] == 0.0
    # Not exactly zero: it is C * (sum(weights) - 1), and that residual is
    # ~8e-15 rather than 0 -- the module docstring's "sum(c) == 1 identically"
    # is true analytically and not in IEEE arithmetic.  Pinned so the claim
    # cannot quietly become load-bearing.
    assert stats["delta_static_max_ev"] < 1e-12


def test_static_limit_ruling_measures_the_tail_it_omits():
    """A 1/N static term: the omitted tail is exactly the term's own residual.

    THE NUMBER THE PER-compute_mode GUARD CANNOT PRODUCE.  ``sigma_dispatch``
    refuses the extrapolation on a static ``compute_mode``, but
    ``ppm_invalid_mode = "static_limit"`` -- the shipping default -- puts a
    static Coulomb hole inside a Sigma whose mode IS ``gn_ppm``, one logical
    ISDF mode at a time.  This is that check, moved below the mode.
    """
    from gw.band_extrapolation import static_limit_tail_ruling
    N, fit = _fit_for_static_tests()
    base = np.asarray(fit.s_inf)
    # C(N) = -0.30 - 0.9/N: an exact two-parameter tail, so the estimator
    # recovers the limit -0.30 and the omitted tail must be |C(N3) - (-0.30)|.
    C = np.stack([np.full_like(base, -0.30 - 0.9 / n) for n in N])
    exceeds, text, stats = static_limit_tail_ruling(fit, C)
    assert abs(stats["delta_static_median_ev"] - 0.9 / N[-1]) < 1e-12
    assert stats["span_median_ev"] > 0.0, (
        "a band-DEPENDENT static term must report a nonzero span; the span is "
        "the direct refutation of the 'band-count independent' claim that "
        "used to justify the bracket-0 fold in ppm_sigma.")
    # ESCALATION IS A COMPARISON, NOT A CONSTANT.  This fit's own correction is
    # ~0.5 eV, so its p90 bar is ~75 meV against a 17.3 meV
    # static tail -- the omission is real, measured, and SMALLER than the bar
    # the run already quotes, which is exactly the case that must NOT escalate.
    # The escalating case is covered separately below.
    assert abs(stats["ratio_median"]
               - stats["delta_static_median_ev"] / stats["bar_median_ev"]) < 1e-12
    assert exceeds is (stats["ratio_median"] > 1.0)
    assert exceeds is False
    assert "static-limit" in text and "ppm_invalid_mode = zero" not in text


def test_static_limit_ruling_warns_and_does_not_refuse():
    """It must not raise.  BOTH sides of this are shipping defaults.

    ``ppm_invalid_mode`` defaults to ``static_limit`` and
    ``use_band_extrapolation`` now defaults to True, so a refusal on their
    conjunction would refuse the configuration the code ships with -- the same
    reasoning that keeps ``sc_tolerance_ruling`` a warning.
    """
    from gw.band_extrapolation import static_limit_tail_ruling
    N, fit = _fit_for_static_tests()
    base = np.asarray(fit.s_inf)
    C = np.stack([np.full_like(base, -3.0 - 500.0 / n) for n in N])
    exceeds, text, stats = static_limit_tail_ruling(fit, C)   # must not raise
    assert exceeds is True
    assert isinstance(text, str) and text


def test_static_limit_ruling_refuses_a_mismatched_bracket_plan():
    """Comparing a static term built on other band counts is meaningless."""
    from gw.band_extrapolation import static_limit_tail_ruling
    N, fit = _fit_for_static_tests()
    base = np.asarray(fit.s_inf)
    with pytest.raises(ValueError, match="SAME cumulative band counts"):
        static_limit_tail_ruling(fit, np.stack([base, base]))
    with pytest.raises(ValueError, match="state axes"):
        static_limit_tail_ruling(
            fit, np.stack([np.zeros(base.shape + (2,))] * 3))


def test_static_limit_ruling_does_not_escalate_on_a_ratio_of_two_zeros():
    """A near-zero tail must not escalate however it compares to a ~0 bar.

    MEASURED, not anticipated.  On a Si arm whose three bracket points came
    out bit-identical -- the brackets above the deck's real band window
    contributed exactly nothing, so ``Delta_tail`` was 0 -- this ruling
    divided a ~1e-9 meV tail by a ~1e-14 meV bar, reported ``ratio = 122880``
    and escalated on a run whose true omission was zero to every digit it
    printed.  A ratio of two numbers that are both ~0 is not a signal, and a
    warning that fires there would train its reader to ignore it.
    """
    from gw.band_extrapolation import (
        static_limit_tail_ruling, STATIC_LIMIT_TAIL_FLOOR_EV)
    N = np.array([62, 90, 100])
    # Three IDENTICAL points: the exact degenerate case observed.
    fit = _spectral_fit(N, np.full((3, 4), 1.088910))
    assert float(np.max(np.abs(fit.delta_tail))) == 0.0
    C = np.stack([np.full((4,), -1e-9 * (1.0 + 1e-3 * i)) for i in range(3)])
    exceeds, text, stats = static_limit_tail_ruling(fit, C)
    assert stats["ratio_median"] > 1.0, (
        "the guarded case is precisely the one where the RATIO is large")
    assert stats["delta_static_median_ev"] < STATIC_LIMIT_TAIL_FLOOR_EV
    assert exceeds is False, (
        f"escalated on a {stats['delta_static_median_ev'] * 1e3:.3e} meV tail "
        f"because the bar was {stats['bar_median_ev'] * 1e3:.3e} meV")
    assert "Nothing to act on" in text
