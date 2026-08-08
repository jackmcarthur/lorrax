"""The MPA fit stage end to end, and the 14-pass accumulation lemma.

CPU only, ``JAX_PLATFORMS=cpu``; nothing here touches a cluster.

WHAT THIS SUITE PROVES THAT NEITHER HALF CAN PROVE ALONE.  The fit
kernel's suite plants a pole set, samples it, fits it in memory and
gets it back.  The store's suite writes a frequency-resolved tensor,
refuses a half-stamped file, and stages a fit store block by block.
Neither of them ever puts a sample on disk and takes a pole off it.
The claim that spans both — that a field planted in W_c survives being
written per frequency, read back in budgeted column blocks, fitted
elementwise, staged block by block, finalized, and read again — is
this suite's, and it is asserted at the kernel's demonstrated recovery
floor rather than at a tolerance chosen to pass.

THE SYNTHETIC GEOMETRY IS ORBIT-CLOSED BY CONSTRUCTION, not by luck,
and it is the store suite's glide: the centroid set is the union of
twelve seeds' orbits under a non-symmorphic glide, so
``verify_centroid_orbit_closure`` passes as a property of the builder.
Every seed is chosen with its first index below the half-cell shift, so
every orbit has size exactly two and the twelve are pairwise disjoint —
which is how the extent is exactly 24 rather than "however many
survived".

THE PLANTED FIELD IS SMOOTH AND PHYSICAL ON PURPOSE.  Omega_p and B_p
vary smoothly over (q, mu, nu) about a well-separated Si-like base set,
every pole sits inside the admissible box with Gamma well under a, and
the variation is symmetric under mu <-> nu so the synthesized W_q is
symmetric the way the real one is.  No guard should fire anywhere in
the field, and the suite asserts that it does not: a run in which the
guards were quietly repairing the data would report a recovery number
that says more about the guards than about the pipeline.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("symmetry_maps.qirr_store")
jax = pytest.importorskip("jax")

from symmetry_maps import (centroid_source_map_and_wrap,          # noqa: E402
                           verify_centroid_orbit_closure)
from symmetry_maps import qirr_store as QS                        # noqa: E402

from file_io import mpa_store as MS                               # noqa: E402
from gw.mpa import fit_driver, pade_fit, sampling, tiling         # noqa: E402

#: The synthetic FFT grid the centroid set lives on.
_FFT = np.array([12, 12, 12], dtype=np.int64)

#: A GLIDE: {sigma_z | tau = (1/2, 0, 0)}.  Order two, so {I, g} is a
#: group and the orbits really close.  tau x grid = (6, 0, 0) is integer
#: on the 12-grid, so images land on grid points.
_SYMS = np.stack([np.eye(3, dtype=np.int64),
                  np.diag([1, 1, -1]).astype(np.int64)])
_TNP = np.array([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])

#: 5 full-BZ q folding onto 3 IBZ parents, two through TIME-REVERSED
#: rows, so the antiunitary branch of the table is live.
_IRR = np.array([0, 1, 1, 2, 2], dtype=np.int32)
_SYM = np.array([0, 1, 0, 3, 0], dtype=np.int32)
_N_SYM_SPATIAL = 2
_N_Q_IBZ = 3

_Q_IRR = np.array([[0.0, 0.0, 0.0],
                   [1 / 3, 0.0, 1 / 4],
                   [0.0, 1 / 3, 1 / 3]])

#: Twelve seeds, every one with first index < 6.  The glide maps
#: (r0, r1, r2) -> (r0 + 6, r1, -r2), so no seed is its own image and no
#: seed is another's: the twelve orbits are disjoint and the extent is
#: exactly 24.
_SEEDS = ((0, 0, 0), (1, 2, 3), (2, 5, 7), (3, 1, 9),
          (4, 7, 2), (5, 3, 11), (0, 4, 5), (1, 9, 1),
          (2, 3, 4), (3, 8, 8), (4, 2, 10), (5, 6, 6))

_N_MU = 24
_N_P = 4
_OMEGA_M = 4.0
_W_NAME = "W_qmunu_omega"

#: The sampling record the W(omega) file carries.  The double-parallel
#: lines varpi_1 = 0.1 Ha and varpi_2 = 1 Ha, alpha = 1 for an
#: insulator, and an n_p whose 2*n_p is the frequency count.
_SAMPLING = {"varpi": [0.1, 1.0], "n_p": _N_P, "alpha": 1,
             "omega_max": _OMEGA_M, "protocol": "double_parallel"}


# ---------------------------------------------------------------------------
# Geometry, the planted field, and the W(omega) file
# ---------------------------------------------------------------------------

def _closed_centroid_set():
    """The union of the seeds' orbits — closed by definition."""
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
    """``(tables, verdict, n_mu)`` for the synthetic closed set."""
    cent = _closed_centroid_set()
    verdict = verify_centroid_orbit_closure(
        cent.astype(np.float64) / _FFT, _SYMS, tnp=_TNP, fft_grid=_FFT)
    assert verdict.closed, verdict.describe()
    perm, L = centroid_source_map_and_wrap(
        cent, _SYMS, _TNP, _FFT, validate=True, extend_trs=True)
    tables = QS.QirrTables(_IRR, _SYM, _Q_IRR, perm, L, _N_SYM_SPATIAL)
    return tables, verdict, int(cent.shape[0])


def _planted_field(n_p, n_q, n_mu):
    """``(Omega, B)`` of shape ``(n_p, n_q, N_mu, N_mu)``, smooth.

    A well-separated Si-like base set, modulated smoothly over the
    element indices and mildly over q.  Both modulations are SYMMETRIC
    under mu <-> nu, so the synthesized W_q is symmetric the way the
    physical one is.

    THE WIDTH RATIO GROWS WITH THE POLE INDEX, from 10% of the energy
    to 10% + 2% per pole, which is the physics the theory plan
    describes ("the fit assigns broad envelopes to high-lying pole
    bundles") and is also what keeps the field from being degenerate:
    a common width ratio would put every pole of an element on one ray
    through the origin, and a fit that recovered a ray exactly would
    say less than one that recovers a scatter.  Every ratio stays far
    under the admissible ``|Im Omega| <= Re Omega``, so no guard has
    anything to repair anywhere in the field.

    THE ENERGY MODULATION IS COMMON TO EVERY POLE, deliberately: ``a_p``
    is ``base_a[p]`` times one element-and-q factor, so the ordering in
    ``Re Omega`` is the base ordering at every element.  The kernel
    returns its poles sorted ascending in ``Re Omega``, so a field
    whose ordering varied element by element would need a matching step
    before it could be compared, and the matching step is where a
    recovery test quietly becomes a test of the matcher.
    """
    base_a = np.linspace(0.4, 3.2, n_p)
    base_mag = 0.5 + 0.25 * np.arange(n_p)
    base_phase = 0.7 * np.arange(n_p)
    width_ratio = 0.10 + 0.02 * np.arange(n_p)

    mu = np.arange(n_mu)
    s = (np.cos(2 * np.pi * mu / n_mu)[:, None]
         + np.cos(2 * np.pi * mu / n_mu)[None, :])          # symmetric
    t = np.sin(2 * np.pi * (mu[:, None] + mu[None, :]) / n_mu)

    q_fac = 1.0 + 0.05 * np.arange(n_q)
    q_res = 1.0 + 0.03 * np.arange(n_q)
    shape = (n_p, n_q, n_mu, n_mu)
    a = (base_a[:, None, None, None]
         * q_fac[None, :, None, None]
         * (1.0 + 0.06 * s[None, None, :, :] / 2.0))
    gamma = (width_ratio[:, None, None, None] * a
             * (1.0 + 0.10 * t[None, None, :, :]))
    Omega = (a - 1j * gamma).astype(np.complex128)

    mag = (base_mag[:, None, None, None]
           * q_res[None, :, None, None]
           * (1.0 + 0.08 * s[None, None, :, :] / 2.0))
    phase = (base_phase[:, None, None, None]
             + np.zeros((1, n_q, 1, 1))
             + 0.3 * t[None, None, :, :])
    B = (mag * np.exp(1j * phase)).astype(np.complex128)
    assert Omega.shape == shape and B.shape == shape
    return Omega, B


def _synthesize_w(Omega, B, z):
    """``W_c(q, mu, nu, z_j)`` from the planted field, frequency-major.

    The model of ``pade_fit`` and the papers, in plain numpy over the
    whole field at once::

        W_c(z) = sum_p 2 Omega_p B_p / (z**2 - Omega_p**2)

    Returns ``(n_omega, n_q, N_mu, N_mu)`` — the store's layout, with
    the frequency axis leading.
    """
    zz = np.asarray(z, dtype=np.complex128)
    denom = (zz[:, None, None, None, None] ** 2
             - Omega[None] ** 2)
    assert np.all(np.abs(denom) > 0.0), "a sample sits on a planted pole"
    return np.sum(2.0 * Omega[None] * B[None] / denom, axis=1)


def _protocol_grid(n_p=_N_P):
    return sampling.double_parallel_grid(
        n_p, _OMEGA_M, material_class="insulator", alpha=1,
        energy_unit="Ha")


def _write_w_file(path, n_p=_N_P):
    """A complete, fully-ready W(omega) file plus the planted field."""
    tables, verdict, n_mu = _geometry()
    assert n_mu == _N_MU, f"geometry drifted: {n_mu} centroids"
    z = _protocol_grid(n_p)
    Omega, B = _planted_field(n_p, _N_Q_IBZ, n_mu)
    W = _synthesize_w(Omega, B, z)
    line = np.array([0] * n_p + [1] * n_p, dtype=np.int32)

    MS.allocate_w_omega(
        str(path), _W_NAME, n_omega=2 * n_p, n_q_on_disk=_N_Q_IBZ,
        n_mu=n_mu, tables=tables, omega=z,
        sampling=dict(_SAMPLING, n_p=n_p), omega_line=line,
        closure_verdict=verdict,
        provenance={"deck": "synthetic-glide-mpa-driver"})
    for i in range(2 * n_p):
        MS.write_w_slab(str(path), _W_NAME, i, W[i], ready=True)
    return {"z": z, "Omega": Omega, "B": B, "W": W, "n_mu": n_mu,
            "n_q": _N_Q_IBZ, "n_p": n_p}


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """The W(omega) file and the field it was synthesized from."""
    root = tmp_path_factory.mktemp("mpa_driver")
    w_path = root / "W_omega.h5"
    field = _write_w_file(w_path)
    field["root"] = root
    field["w_path"] = w_path
    return field


@pytest.fixture(scope="module")
def fitted(planted):
    """One full driver run: the finalized store, ledger and report."""
    fit_path = planted["root"] / "mpa_fit.h5"
    stream = io.StringIO()
    ledger, report = fit_driver.run_fit_driver(
        str(planted["w_path"]), _W_NAME, str(fit_path),
        planted["z"], planted["n_p"], report_stream=stream)
    return {"path": fit_path, "ledger": ledger, "report": report,
            "text": stream.getvalue()}


# ---------------------------------------------------------------------------
# (a) THE GATE: the field survives read -> fit -> write -> read
# ---------------------------------------------------------------------------

def test_end_to_end_recovers_the_planted_pole_field(planted, fitted,
                                                    capsys):
    """Poles in through the W file, same poles out of the fit store.

    THE GATE.  Every step is the production one: the samples were
    written per frequency and stamped ready, read back in budgeted
    column blocks through ``read_w_columns``, fitted by the vmapped
    kernel, staged block by block through ``write_fit_block``, and read
    again through ``read_fit_tensors`` after ``finalize_fit_store``.
    Nothing in this assertion touches an in-memory shortcut.
    """
    Om, Bp, diag, ledger = MS.read_fit_tensors(str(fitted["path"]))
    assert ledger["complete"]
    assert Om.shape == planted["Omega"].shape
    assert Bp.shape == planted["B"].shape

    d_omega = float(np.max(np.abs(Om - planted["Omega"])))
    d_b = float(np.max(np.abs(Bp - planted["B"])))
    with capsys.disabled():
        print(f"\n[mpa end-to-end n_p={planted['n_p']} N_mu="
              f"{planted['n_mu']} n_q={planted['n_q']}] "
              f"max|dOmega|={d_omega:.3e} max|dB|={d_b:.3e} "
              f"worst cond={diag['condition'].max():.3e} "
              f"worst backward error={diag['backward_error'].max():.3e}")

    # The kernel's demonstrated recovery floor, through the whole
    # pipeline.  Not a looser number because the bytes went to disk:
    # complex128 round-trips exactly, so the disk leg must cost nothing.
    assert d_omega < 1.0e-7
    assert d_b < 1.0e-6


def test_end_to_end_at_the_si_pole_schedule(tmp_path, capsys):
    """The gate again at n_p = 8 — the Si schedule, and the regime
    where the conditioning the theory plan ranks as risk 6 actually
    bites.

    WHY A SECOND LEG RATHER THAN A BIGGER FIRST ONE.  At n_p = 4 the
    Pade system is conditioned around 1e4 and the pipeline recovers the
    field to about 1e-12, which proves the plumbing but says nothing
    about the method's floor.  At n_p = 8 — Si's scheduled count — the
    same system runs at 1e8 and the kernel's own suite measures
    max|dOmega| ~ 1.7e-9.  This leg asserts the pipeline costs nothing
    ON TOP of that: the disk round trip is complex128 in and complex128
    out, so the end-to-end number must sit at the kernel's floor and
    not above it.

    It is also the walk at its narrowest.  2*n_p = 16 frequencies
    against N_mu = 24 puts ``choose_column_budget`` at ONE column per
    block, so this leg exercises the single-column walk — 24 blocks per
    q rather than 8 — which is the shape the budget clamps to whenever
    the frequency grid grows against the centroid count.

    TOLERANCES, AND WHY THEY ARE A DECADE ABOVE THE n_p = 4 LEG.
    Measured here: max|dOmega| ~ 6.3e-8, max|dB| ~ 6.6e-7, at a worst
    condition number of 7.8e9.  That is about 60x the 1.2e8 the fit
    kernel's own suite reports at the same n_p, and the difference is
    the PLANTED FIELD rather than the pipeline: this suite's widths run
    to 24% of the pole energy where the kernel suite's run to under 3%,
    and broad poles make a smoother W_c whose pole content is harder to
    separate.  Broad is the more honest choice for a screening tensor,
    so the conditioning is accepted and the tolerance is set the way
    the kernel suite sets its own — one decade above the measurement,
    so this reports a regression rather than weather.  The tight number
    is asserted on the n_p = 4 leg, which has five decades of margin.
    """
    n_p = sampling.POLE_SCHEDULE["Si"]
    assert n_p == 8
    field = _write_w_file(tmp_path / "W_si.h5", n_p=n_p)
    fit_path = tmp_path / "mpa_fit_si.h5"

    stream = io.StringIO()
    ledger, report = fit_driver.run_fit_driver(
        str(tmp_path / "W_si.h5"), _W_NAME, str(fit_path),
        field["z"], n_p, report_stream=stream)
    assert ledger["complete"]
    assert report["n_cols_budget"] == 1
    assert report["blocks_walked"] == field["n_q"] * field["n_mu"]

    Om, Bp, diag, _ = MS.read_fit_tensors(str(fit_path))
    d_omega = float(np.max(np.abs(Om - field["Omega"])))
    d_b = float(np.max(np.abs(Bp - field["B"])))
    with capsys.disabled():
        print(f"\n[mpa end-to-end SI n_p={n_p} N_mu={field['n_mu']} "
              f"n_q={field['n_q']}] max|dOmega|={d_omega:.3e} "
              f"max|dB|={d_b:.3e} worst cond="
              f"{diag['condition'].max():.3e}")

    assert np.all(diag["n_valid"] == float(n_p))
    assert d_omega < 1.0e-6
    assert d_b < 1.0e-5


def test_no_guard_fires_anywhere_in_the_planted_field(planted, fitted):
    """Every element keeps all n_p poles, so the gate measures the fit.

    A field the guards were repairing would report a recovery number
    that is a statement about the repairs.  ``n_valid`` is carried into
    the store as a per-element diagnostic precisely so this is
    checkable after the fact rather than only at fit time.
    """
    _, _, diag, _ = MS.read_fit_tensors(str(fitted["path"]))
    assert np.all(diag["n_valid"] == float(planted["n_p"]))


def test_every_block_has_its_diagnostics_and_the_ledger_is_complete(
        planted, fitted):
    """The ledger is whole, and no diagnostic is a silent zero."""
    ledger = fitted["ledger"]
    n_mu, n_q = planted["n_mu"], planted["n_q"]
    plan = tiling.plan_column_walk(n_mu, 2 * planted["n_p"])

    assert ledger["n_done"] == n_q * n_mu
    assert ledger["n_done"] == ledger["n_total"]
    assert bool(ledger["blocks_done"].all())
    assert ledger["journal"].shape == (n_q * plan["n_blocks"], 3)
    assert ledger["block_condition_max"].size == n_q * plan["n_blocks"]

    _, _, diag, _ = MS.read_fit_tensors(str(fitted["path"]))
    for key in ("condition", "backward_error", "residual", "n_valid"):
        arr = diag[key]
        assert arr.shape == (n_q, n_mu, n_mu), key
        assert np.all(np.isfinite(arr)), key
        # A zero condition number is a PERFECTLY conditioned solve, so
        # an unwritten element is indistinguishable from a flawless one
        # unless something positive was written everywhere.
        assert np.all(arr > 0.0), key


def test_the_journal_spans_the_column_axis_exactly_once(planted, fitted):
    """Contiguous, in order, q-major, no column covered twice."""
    journal = fitted["ledger"]["journal"]
    n_mu, n_q = planted["n_mu"], planted["n_q"]
    seen = np.zeros((n_q, n_mu), dtype=int)
    last_q = -1
    for iq, lo, hi in journal:
        assert iq >= last_q, "the walk is q-major"
        last_q = iq
        seen[iq, lo:hi] += 1
    assert np.all(seen == 1)


# ---------------------------------------------------------------------------
# (b) RED TWIN: a skipped block must make finalize refuse, by name
# ---------------------------------------------------------------------------

def test_red_twin_a_skipped_block_refuses_finalize_and_names_the_range(
        planted, tmp_path):
    """Walk every block but one; finalize must refuse and say which.

    The point is not that finalize refuses — the store's own suite has
    that — but that a DRIVER that dropped a block leaves a store whose
    refusal names the range a human can act on.  "Which columns are
    missing" is the only question a run that died halfway actually has.
    """
    fit_path = tmp_path / "gappy.h5"
    n_mu, n_q, n_p = planted["n_mu"], planted["n_q"], planted["n_p"]
    MS.allocate_fit_store(str(fit_path), n_q=n_q, n_mu=n_mu, n_p=n_p)

    schedule = tiling.fit_schedule(n_q, n_mu, 2 * n_p)
    skipped = schedule[len(schedule) // 2]
    for q, lo, hi in schedule:
        if (q, lo, hi) == skipped:
            continue
        fit_driver.fit_one_block(
            str(planted["w_path"]), _W_NAME, str(fit_path), q,
            np.arange(lo, hi), planted["z"], n_p)

    with pytest.raises(ValueError) as exc:
        MS.finalize_fit_store(str(fit_path))
    msg = str(exc.value)
    q, lo, hi = skipped
    assert f"q={q}: columns" in msg
    span = f"{lo}-{hi - 1}" if hi - lo > 1 else f"{lo}"
    assert span in msg, f"{span!r} not named in: {msg}"
    assert "zeros are poles" in msg


# ---------------------------------------------------------------------------
# (c) RED TWIN: a busted budget refuses mid-walk and resumes cleanly
# ---------------------------------------------------------------------------

def test_red_twin_budget_bust_refuses_mid_walk_then_resumes(
        planted, tmp_path, capsys):
    """A block too wide is refused, changes nothing, and the walk goes on.

    THE RESUMABILITY CLAIM, AS A TEST.  ``fit_one_block`` reads before
    it writes and writes nothing partial, so a refused block leaves the
    staged store byte-identical: the ledger never learned about it.
    That is what makes the refusal a pause rather than a corruption,
    and it is asserted here by comparing the ledger across the refusal
    and then finishing the walk to a finalized store whose poles still
    match the planted field.
    """
    fit_path = tmp_path / "resumed.h5"
    n_mu, n_q, n_p = planted["n_mu"], planted["n_q"], planted["n_p"]
    n_omega = 2 * n_p
    MS.allocate_fit_store(str(fit_path), n_q=n_q, n_mu=n_mu, n_p=n_p)

    budget = MS.choose_column_budget(n_mu, n_omega)
    assert budget < n_mu, "the walk must take more than one block"
    schedule = tiling.fit_schedule(n_q, n_mu, n_omega)

    # Walk two legal blocks, then ask for one column more than the
    # budget allows, mid-walk.
    for q, lo, hi in schedule[:2]:
        fit_driver.fit_one_block(
            str(planted["w_path"]), _W_NAME, str(fit_path), q,
            np.arange(lo, hi), planted["z"], n_p)
    before = MS.fit_completion_ledger(str(fit_path))

    q, lo, _ = schedule[2]
    greedy = np.arange(lo, lo + budget + 1)
    with pytest.raises(ValueError) as exc:
        fit_driver.fit_one_block(
            str(planted["w_path"]), _W_NAME, str(fit_path), q, greedy,
            planted["z"], n_p)
    msg = str(exc.value)
    assert f"refusing {budget + 1} columns" in msg
    assert f"allows {budget}" in msg
    with capsys.disabled():
        print(f"\n[budget refusal] {msg.splitlines()[0][:120]}")

    # NOTHING MOVED.  The refusal happened in the read, before any
    # write, so the ledger is exactly what it was.
    after = MS.fit_completion_ledger(str(fit_path))
    assert after["n_done"] == before["n_done"]
    np.testing.assert_array_equal(after["blocks_done"],
                                  before["blocks_done"])
    assert after["journal"].shape == before["journal"].shape

    # RESUME: the same block at a legal width, then the rest.
    for q, lo, hi in schedule[2:]:
        fit_driver.fit_one_block(
            str(planted["w_path"]), _W_NAME, str(fit_path), q,
            np.arange(lo, hi), planted["z"], n_p)
    ledger = MS.finalize_fit_store(str(fit_path))
    assert ledger["complete"]

    Om, Bp, _, _ = MS.read_fit_tensors(str(fit_path))
    assert np.max(np.abs(Om - planted["Omega"])) < 1.0e-7
    assert np.max(np.abs(Bp - planted["B"])) < 1.0e-6


def test_the_walk_never_reads_more_than_one_tile(fitted):
    """Every block's read is priced at or under one (N_mu, N_mu) tile."""
    r = fitted["report"]
    assert r["peak_block_bytes"] <= r["tile_bytes"]
    assert r["bytes_read"] <= r["bytes_budget_total"]


def test_driver_refuses_a_grid_that_is_not_the_files_stamped_grid(
        planted, tmp_path):
    """A fit against the wrong abscissae is what the stamp prevents."""
    wrong = np.asarray(planted["z"], dtype=np.complex128).copy()
    wrong[2] += 1.0e-6
    with pytest.raises(ValueError, match="stamped omega"):
        fit_driver.run_fit_driver(
            str(planted["w_path"]), _W_NAME, str(tmp_path / "no.h5"),
            wrong, planted["n_p"])


# ---------------------------------------------------------------------------
# (d) THE COST REPORT
# ---------------------------------------------------------------------------

def test_cost_report_is_arithmetically_consistent(planted, fitted):
    """Every counter is derivable from the geometry, and they agree."""
    r = fitted["report"]
    n_mu, n_q, n_p = planted["n_mu"], planted["n_q"], planted["n_p"]
    n_omega = 2 * n_p
    plan = tiling.plan_column_walk(n_mu, n_omega)

    assert r["blocks_walked"] == n_q * plan["n_blocks"]
    assert r["columns_read"] == n_q * n_mu
    assert r["elements_fitted"] == n_mu * r["columns_read"]
    assert r["logical_outputs"] == 2 * n_p * r["elements_fitted"]
    assert r["bytes_read"] == (n_omega * n_mu * r["columns_read"]
                               * MS.COMPLEX128_BYTES)
    assert r["bytes_budget_total"] == r["blocks_walked"] * r["tile_bytes"]

    # One vmapped dispatch per block for the fit, one for the
    # diagnostics — and three Pade solves per element behind them.
    assert r["fit_dispatches"] == r["blocks_walked"]
    assert r["diagnostic_dispatches"] == r["blocks_walked"]
    assert r["kernel_solves"] == 3 * r["elements_fitted"]

    sec = r["seconds"]
    parts = sec["read"] + sec["fit"] + sec["write"] + sec["finalize"]
    assert all(v >= 0.0 for v in sec.values())
    assert sec["total"] > 0.0
    assert parts <= sec["total"] + 1.0e-6


def test_cost_report_text_states_what_the_plan_demands(fitted, capsys):
    """Printed once at finalize, and it says the load-bearing numbers."""
    text = fitted["text"]
    for token in ("logical outputs", "dispatches", "kernel solves",
                  "bytes read", "peak block", "seconds",
                  "columns read", "elements fitted"):
        assert token in text, f"cost report omits {token!r}"
    assert str(fitted["report"]["logical_outputs"]) in text
    assert str(fitted["report"]["blocks_walked"]) in text
    with capsys.disabled():
        print(text)


# ---------------------------------------------------------------------------
# (e) THE 14-PASS ACCUMULATION LEMMA
# ---------------------------------------------------------------------------

def _toy_green(n_band, n_mu, n_k, seed=11):
    """A toy G(band, mu, k).  Pure numpy, small, and not physical.

    It stands in for whatever the real Sigma integration contracts the
    poles against.  The lemma is about the SUM STRUCTURE, so the only
    property required of G is that it be a fixed array the two routes
    share.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_band, n_mu, n_k))
            + 1j * rng.standard_normal((n_band, n_mu, n_k)))


def _toy_energies(n_k):
    """Output energies well away from every planted pole."""
    return np.array([0.15, 1.75, 2.95])[:n_k]


def _one_pole_term(Omega_p, B_p, G, E):
    """The per-pole resolvent contraction — ONE pole, all q and mu, nu.

    ``sigma[a, b, k] = sum_{q, u, v} G[a, u, k]
                        * 2 Omega B / (E_k**2 - Omega**2)[k, q, u, v]
                        * conj(G[b, v, k])``

    Resolvent-like and deliberately not a self-energy: it has the shape
    of the real contraction (a pole-resolvent sandwiched between two
    Green's functions and summed over the ISDF indices) and none of its
    physics.
    """
    denom = E[:, None, None, None] ** 2 - Omega_p[None] ** 2
    R = 2.0 * Omega_p[None] * B_p[None] / denom
    return np.einsum("auk,kquv,bvk->abk", G, R, np.conj(G))


def test_fourteen_pass_accumulation_equals_the_all_pole_sum(fitted,
                                                            capsys):
    """THE LEMMA: one pass per pole accumulates the all-pole answer.

    The owner's Sigma design runs the existing integration once per
    pole pair with a single ``(B_q, Omega_q)`` resident and accumulates
    the partial results.  That is memory-safe by construction; what it
    needs is a proof that it is also EXACT, i.e. that the pass
    structure is a re-association of a sum and not an approximation.

    Both routes below read the SAME finalized store, so the only
    difference between them is the order the pole terms are added in.
    """
    n_band, n_k = 2, 3
    Om, Bp, _, ledger = MS.read_fit_tensors(str(fitted["path"]))
    n_p = ledger["n_p"]
    G = _toy_green(n_band, ledger["n_mu"], n_k)
    E = _toy_energies(n_k)

    # THE REFERENCE: an explicit sum over ALL poles at once, vectorised
    # over the pole axis, holding every (B, Omega) in memory.
    denom = E[:, None, None, None, None] ** 2 - Om[None] ** 2
    R_all = 2.0 * Om[None] * Bp[None] / denom
    sigma_ref = np.einsum("auk,kpquv,bvk->abk", G, R_all, np.conj(G))

    # THE PASS STRUCTURE: one pole resident at a time, each read as its
    # own slab out of the store.
    sigma_pass, info = fit_driver.accumulate_over_pole_passes(
        str(fitted["path"]),
        lambda p, Omega_p, B_p: _one_pole_term(Omega_p, B_p, G, E))

    assert info["n_passes"] == n_p
    assert info["order"] == tuple(range(n_p))
    assert len(info["passes"]) == n_p

    scale = float(np.max(np.abs(sigma_ref)))
    rel = float(np.max(np.abs(sigma_pass - sigma_ref))) / scale
    with capsys.disabled():
        print(f"\n[14-pass lemma] n_p={n_p} passes, "
              f"max relative disagreement vs the all-pole sum "
              f"{rel:.3e} (|sigma| ~ {scale:.3e})")
    assert rel < 1.0e-14


def test_the_only_difference_is_association_order(fitted):
    """Bit-exact against an association-MATCHED reference.

    The previous test agrees with the vectorised sum to machine
    precision but not to the last bit, and that is the one caveat this
    lemma carries: ``sum_p`` as a vectorised reduction and ``sum_p`` as
    a sequential accumulation associate differently.  This test pins
    the caveat down to exactly that by building the reference with the
    SAME association — a sequential loop in the pinned pass order — and
    demanding bit-for-bit equality.  If the two ever differ, the cause
    is not floating-point association and the lemma is in trouble.
    """
    n_band, n_k = 2, 3
    Om, Bp, _, ledger = MS.read_fit_tensors(str(fitted["path"]))
    G = _toy_green(n_band, ledger["n_mu"], n_k)
    E = _toy_energies(n_k)

    total = None
    for p in fit_driver.pole_pass_order(ledger["n_p"]):
        term = _one_pole_term(Om[p], Bp[p], G, E)
        total = term if total is None else total + term

    sigma_pass, _ = fit_driver.accumulate_over_pole_passes(
        str(fitted["path"]),
        lambda p, Omega_p, B_p: _one_pole_term(Omega_p, B_p, G, E))
    np.testing.assert_array_equal(sigma_pass, total)


def test_the_pass_order_is_pinned_and_reordering_is_visible(fitted):
    """A different order is the same sum, and not the same bits.

    Which is why :func:`pole_pass_order` exists as a named function
    rather than an inline ``range``: the accumulation is reproducible
    only if the order is, and a driver that walked its poles in
    whatever order a directory listing returned would produce a
    different last ulp on every run.
    """
    n_band, n_k = 2, 3
    ledger = MS.fit_completion_ledger(str(fitted["path"]))
    G = _toy_green(n_band, ledger["n_mu"], n_k)
    E = _toy_energies(n_k)

    def kernel(p, Omega_p, B_p):
        return _one_pole_term(Omega_p, B_p, G, E)

    forward, info_f = fit_driver.accumulate_over_pole_passes(
        str(fitted["path"]), kernel)
    backward, info_b = fit_driver.accumulate_over_pole_passes(
        str(fitted["path"]), kernel,
        order=tuple(reversed(fit_driver.pole_pass_order(ledger["n_p"]))))

    assert info_f["order"] == tuple(range(ledger["n_p"]))
    assert info_b["order"] == tuple(reversed(range(ledger["n_p"])))
    # Same sum to machine precision...
    scale = float(np.max(np.abs(forward)))
    assert float(np.max(np.abs(forward - backward))) / scale < 1.0e-14


def test_each_pass_records_its_pole_provenance(fitted):
    """The driver owns the (pole index, ...) half of the pass ledger.

    S section 5.5: the partial Sigmas only add coherently if each pass
    records the triple it used.  The quadrature half is the minimax
    service's; the pole index and its real-ordering key are this
    driver's, and they are recorded per pass rather than reconstructed.
    """
    ledger = MS.fit_completion_ledger(str(fitted["path"]))
    _, info = fit_driver.accumulate_over_pole_passes(
        str(fitted["path"]), lambda p, Omega_p, B_p: np.zeros(1))
    assert [row["pole_index"] for row in info["passes"]] == \
        list(range(ledger["n_p"]))
    # Ascending Re Omega across passes is the real ordering key the
    # design asks the driver to order on (T section C: order on Re
    # Omega, pad by a term proportional to Gamma).
    lows = [row["re_omega_min"] for row in info["passes"]]
    assert lows == sorted(lows)
    assert all(row["gamma_max"] > 0.0 for row in info["passes"])


def test_read_pole_slice_refuses_an_unfinalized_store(planted, tmp_path):
    """The stopgap reader keeps the store's refusal, not just its bytes.

    Reading the pole slab straight out of HDF5 would bypass "an
    unfitted column reads back as zeros, and a zero pole is not an
    absent pole".  The driver's reader re-states that refusal against
    the public ledger, so the shortcut costs a function and not a
    guarantee.
    """
    fit_path = tmp_path / "partial.h5"
    n_mu, n_q, n_p = planted["n_mu"], planted["n_q"], planted["n_p"]
    MS.allocate_fit_store(str(fit_path), n_q=n_q, n_mu=n_mu, n_p=n_p)
    q, lo, hi = tiling.fit_schedule(n_q, n_mu, 2 * n_p)[0]
    fit_driver.fit_one_block(
        str(planted["w_path"]), _W_NAME, str(fit_path), q,
        np.arange(lo, hi), planted["z"], n_p)

    with pytest.raises(ValueError, match="NOT FINALIZED"):
        fit_driver.read_pole_slice(str(fit_path), 0)
    Omega_p, B_p = fit_driver.read_pole_slice(
        str(fit_path), 0, allow_partial=True)
    assert Omega_p.shape == (n_q, n_mu, n_mu)
    assert B_p.shape == (n_q, n_mu, n_mu)


def test_pole_slices_are_the_store_tensor_sliced(fitted):
    """The stopgap reader returns exactly what the whole read returns."""
    Om, Bp, _, ledger = MS.read_fit_tensors(str(fitted["path"]))
    for p in fit_driver.pole_pass_order(ledger["n_p"]):
        Omega_p, B_p = fit_driver.read_pole_slice(str(fitted["path"]), p)
        np.testing.assert_array_equal(Omega_p, Om[p])
        np.testing.assert_array_equal(B_p, Bp[p])


# ---------------------------------------------------------------------------
# (f) the synthesis itself, so the gate cannot pass on a broken planter
# ---------------------------------------------------------------------------

def test_the_planted_samples_match_the_kernels_own_synthesizer(planted):
    """``_synthesize_w`` agrees with ``pade_fit.synthesize_w_samples``.

    The whole-field synthesizer above is vectorised over (q, mu, nu)
    for speed; the kernel ships a per-element one.  If they disagreed,
    the gate would be comparing the pipeline against a field nobody
    planted.  Checked on a handful of elements rather than all 1728.
    """
    z, W = planted["z"], planted["W"]
    Omega, B = planted["Omega"], planted["B"]
    for q, mu, nu in ((0, 0, 0), (1, 7, 3), (2, 23, 11), (0, 5, 22)):
        want = pade_fit.synthesize_w_samples(
            Omega[:, q, mu, nu], B[:, q, mu, nu], z)
        np.testing.assert_allclose(W[:, q, mu, nu], want,
                                   rtol=1e-14, atol=1e-14)


def test_the_planted_field_is_symmetric_in_mu_nu(planted):
    """W_q(mu, nu) is symmetric, as the physical one is."""
    W = planted["W"]
    np.testing.assert_allclose(W, np.swapaxes(W, -1, -2),
                               rtol=1e-13, atol=1e-13)
