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
from gw.band_extrapolation import (
    BandExtrapolationRefused,
    fit_band_extrapolation,
    format_extrapolation_report,
    plan_band_brackets,
    trivial_plan,
    trust_verdict,
)


# ---------------------------------------------------------------------------
#  (2)  THE BRACKETS PARTITION THE BAND SUM  — the gate that matters
# ---------------------------------------------------------------------------

def _mesh_1x1():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))


def _skip_if_no_fft_handler(exc: Exception):
    """Skip ONLY on the named FFI-absence signature; re-raise anything else.

    The τ kernel is FFI-required (decisions.md 2026-08-01), so a platform
    whose ``liblorrax_ffi*.so`` has no flat-k / gw_conv handler cannot run
    this gate at all.  That is an ABSENCE, not a measurement — and it must be
    reported as the named absence rather than swallowed into a green run.
    """
    txt = f"{type(exc).__name__}: {exc}"
    for sig in ("FfiLibraryNotBuilt", "FfiLibraryUnusable",
                "LORRAX_FFT_FFI", "handler"):
        if sig in txt:
            pytest.skip(f"no FFT FFI handler on this platform — {txt}")
    raise exc


def _operands(mesh, *, nk=8, nb=12, n_mu=8, rng=None):
    """Randomised τ-kernel operands at the production shapes/shardings."""
    rng = rng or np.random.default_rng(20260815)

    def c(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    # ψ carriers: (nk, s, μ_X, n) / (nk, n, s, μ_Y) / (nk, m, s, μ_X) /
    # (nk, s, μ_Y, n) with nspin = 1, exactly as wavefunction_bundle emits.
    psi_xn = jnp.asarray(c(nk, 1, n_mu, nb))
    psi_yr = jnp.asarray(c(nk, nb, 1, n_mu))
    m = 4                                    # QP window; divides both mesh axes
    psi_xr = jnp.asarray(c(nk, m, 1, n_mu))
    psi_yn = jnp.asarray(c(nk, 1, n_mu, m))
    E_A = jnp.asarray(np.abs(rng.standard_normal((nk, nb))))
    mask_A = jnp.asarray(rng.random((nk, nb)) > 0.3)
    B_q = jnp.asarray(c(nk, n_mu, n_mu))
    Omega_q = jnp.asarray(np.abs(rng.standard_normal((nk, n_mu, n_mu))) + 0.1)
    mask_B = jnp.asarray(np.ones((nk, n_mu, n_mu), dtype=bool))
    return dict(
        psi_xn=jax.device_put(psi_xn, NamedSharding(mesh, P(None, None, 'x', None))),
        psi_yr=jax.device_put(psi_yr, NamedSharding(mesh, P(None, None, None, 'y'))),
        psi_xr=jax.device_put(psi_xr, NamedSharding(mesh, P(None, None, None, 'x'))),
        psi_yn=jax.device_put(psi_yn, NamedSharding(mesh, P(None, None, 'y', None))),
        E_A=jax.device_put(E_A, NamedSharding(mesh, P(None, None))),
        mask_A=mask_A,
        B_q=jax.device_put(B_q, NamedSharding(mesh, P(None, 'x', 'y'))),
        Omega_q=jax.device_put(Omega_q, NamedSharding(mesh, P(None, 'x', 'y'))),
        mask_B=jax.device_put(mask_B, NamedSharding(mesh, P(None, 'x', 'y'))),
    )


def _run_tau(mesh, op, brackets, *, merged_x, kgrid=(2, 2, 2)):
    from gw.ppm_tau_kernel import _get_sigma_tau_kernel
    try:
        kern = _get_sigma_tau_kernel(
            mesh_xy=mesh, kgrid=kgrid, merged_x=merged_x, brackets=brackets)
    except Exception as exc:                       # noqa: BLE001 — see helper
        _skip_if_no_fft_handler(exc)
    with mesh:
        out = kern(
            op['psi_xn'], op['psi_yr'], op['psi_xr'], op['psi_yn'],
            op['E_A'], op['mask_A'], op['B_q'], op['Omega_q'], op['mask_B'],
            jnp.asarray(-np.inf, dtype=jnp.float64),
            jnp.asarray(np.inf, dtype=jnp.float64),
            jnp.asarray(0.25, dtype=jnp.float64),
            jnp.asarray(0.10, dtype=jnp.float64),
            jnp.asarray(0.3 - 0.7j, dtype=jnp.complex128),
        )
    return jax.tree.map(lambda a: np.asarray(jax.device_get(a)), out)


@pytest.mark.parametrize("merged_x", [True, False])
def test_brackets_partition_the_band_sum(merged_x):
    """cumsum(brackets)[-1] == the un-bracketed full-band σ^τ, to roundoff.

    THE gate for the feature.  Both channel plans are exercised because they
    are two different kernels with two different output carriers, and the
    stack that adds the bracket axis has to be channel-wise in one of them.
    """
    mesh = _mesh_1x1()
    nb = 12
    op = _operands(mesh, nb=nb)

    one = _run_tau(mesh, op, ((0, nb),), merged_x=merged_x)
    three = _run_tau(mesh, op, ((0, 5), (5, 9), (9, nb)), merged_x=merged_x)

    chans_one = (one,) if merged_x else one
    chans_three = (three,) if merged_x else three
    for c1, c3 in zip(chans_one, chans_three):
        assert c1.shape[0] == 1, "the ordinary path must still carry a length-1 axis"
        assert c3.shape[0] == 3
        assert c1.shape[1:] == c3.shape[1:]
        total = np.cumsum(c3, axis=0)[-1]
        ref = c1[0]
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
            mesh_xy=mesh, kgrid=(2, 2, 2), merged_x=True, brackets=None)
        k_one = _get_sigma_kij_kernel(
            mesh_xy=mesh, kgrid=(2, 2, 2), merged_x=True, brackets=((0, nb),))
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
    gaps = boundary_min_gaps(e)
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
    gaps = boundary_min_gaps(e)
    for c in plan.counts[:-1]:
        assert gaps[c] > DEGENERACY_TOL_RY, f"cut {c} splits a multiplet"
    assert list(plan.counts) == sorted(set(plan.counts))
    # Contiguous, and covering [0, nb_padded) exactly — the partition claim
    # at the level of the plan, before any kernel is involved.
    assert plan.bounds[0][0] == 0
    assert plan.bounds[-1][1] == nb
    for (a, b), (c, d) in zip(plan.bounds, plan.bounds[1:]):
        assert b == c, f"bracket gap/overlap between {(a, b)} and {(c, d)}"


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

def test_refuses_when_ncond_le_nval():
    e = _si_like_spectrum()
    nb = e.shape[1]
    with pytest.raises(BandExtrapolationRefused) as exc:
        plan_band_brackets(enabled=True, enk_ry=e, n_occ=nb // 2,
                           nb_logical=nb, nb_padded=nb)
    msg = str(exc.value)
    assert "sigma_band_extrapolation" in msg
    assert "n_cond" in msg and "n_occ" in msg
    assert "nband" in msg, "the refusal must name the knob that fixes it"


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


def test_dispatch_refuses_the_key_on_a_non_ppm_mode():
    """A key no kernel reads must refuse, not be ignored."""
    import types
    from gw.gw_config import ComputeMode
    from gw.sigma_dispatch import compute_sigma_xc

    cfg = types.SimpleNamespace(
        sigma=types.SimpleNamespace(band_extrapolation=True))
    with pytest.raises(NotImplementedError) as exc:
        compute_sigma_xc(
            ComputeMode.COHSEX, wfns=None, V_q=None, W_by_role={},
            e_qp_ev=None, static_head_terms=None, head_resolver=None,
            quad=None, config=cfg, meta=None, mesh_xy=None, sym=None,
            wfn=None, band_slices=None, input_dir=".")
    assert "sigma_band_extrapolation" in str(exc.value)


# ---------------------------------------------------------------------------
#  the fit
# ---------------------------------------------------------------------------

def test_fit_recovers_an_exact_two_parameter_tail():
    N = np.array([52, 76, 100])
    s_inf, A = -1.25 + 0.3j, 40.0 - 2.0j
    S = s_inf + A / N[:, None, None]
    fit = fit_band_extrapolation(N, S)
    assert np.allclose(fit.s_inf, s_inf, atol=1e-12)
    assert np.allclose(fit.amplitude, A, atol=1e-10)
    # On an EXACT 1/N series every pairwise intercept is the same number, so
    # the curvature diagnostic must be zero and the tail must be the real
    # remaining correction.
    assert float(np.max(np.abs(fit.delta_model))) < 1e-11
    assert np.allclose(fit.delta_tail, np.abs(A / N[-1]), atol=1e-11)
    assert float(np.max(np.abs(fit.residual))) < 1e-11
    assert "consistent" in trust_verdict(fit)


def test_pairwise_intercepts_are_the_closed_form_two_point_solution():
    N = np.array([52.0, 76.0, 100.0])
    S = np.array([-1.0, -0.8, -0.7])[:, None]
    fit = fit_band_extrapolation(N, S)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        expect = (N[j] * S[j] - N[i] * S[i]) / (N[j] - N[i])
        assert np.allclose(fit.pair_s_inf[(i, j)], expect)


def test_sign_reversal_is_called_out():
    N = np.array([52.0, 76.0, 100.0])
    S = np.array([-1.0, -0.5, -0.9])[:, None]      # non-monotone in 1/N
    fit = fit_band_extrapolation(N, S)
    assert "NOT TRUSTWORTHY" in trust_verdict(fit)


def test_report_carries_full_band_and_extrapolated_side_by_side():
    e = _si_like_spectrum()
    nb = e.shape[1]
    plan = plan_band_brackets(
        enabled=True, enk_ry=e, n_occ=2, nb_logical=nb, nb_padded=nb)
    N = np.asarray(plan.counts, dtype=float)
    # (3, nk, nb) — the shape the driver actually fits, so a per-state index
    # is a genuine 2-axis address.  A (3, 1, 1) stand-in hid a real defect:
    # `ExtrapolationFit.at((k, n))` spliced the tuple wrong and returned a
    # (3, 2) array, which only fails when nb > 1.
    nk_t, nb_t = 4, 6
    rng = np.random.default_rng(7)
    S = ((-1.0 + 30.0 / N)[:, None, None]
         + 0.01 * rng.standard_normal((1, nk_t, nb_t)))
    fit = fit_band_extrapolation(N, S)
    assert np.shape(fit.at((2, 3)).s_at_counts) == (3,), \
        "at((k, n)) must reduce the state axes to a scalar per point"
    text = format_extrapolation_report(
        plan, fit, states=[("VBM k=2 n=3", (2, 3))])
    for needle in ("N1 =", "N2 =", "N3 =", "S_inf", "S(N3)", "A =",
                   "S_inf^(12)", "S_inf^(23)", "S_inf^(13)",
                   "Delta_tail", "Delta_model", "verdict",
                   "VBM k=2 n=3", "envelope"):
        assert needle in text, f"report is missing {needle!r}"
    for c in plan.counts:
        assert str(c) in text
    # The named-state row must carry BOTH the full-band value and the
    # extrapolated one, side by side, on one line — that is the output
    # requirement, not a formatting preference.
    side_by_side = [ln for ln in text.splitlines()
                    if "full" in ln and "S_inf =" in ln]
    assert side_by_side, "no line carries S(N3) and S_inf side by side"
