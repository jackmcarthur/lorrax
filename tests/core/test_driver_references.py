"""One cached run of each GW route on only the two tiny systems."""
from __future__ import annotations

import re
import shutil

import h5py
import numpy as np
import pytest

import harness
from core import rank_session


EQP_ATOL_EV = 2.0e-5
# The GN deck fits Sigma to 1e-3 and permits the explicitly uncertified
# imaginary-axis quadrature. P4 tightening to 1e-4 changes Eqp1 by 0.155 meV;
# the historical pin differs by 0.323 meV at 1e-3, 0.168 meV at 1e-4.
# A 0.5 meV budget covers this approximation; static pins stay at 0.02 meV.
GN_EQP_ATOL_EV = 5.0e-4
# One-shot MPA at 1e-4 differs from 1e-3 by 0.101 meV (Eqp1),
# and the 1e-3 reference differs across P by 0.076 meV (Eqp0).
MPA_EQP_ATOL_EV = 2.0e-4
ZETA_ATOL = 2.0e-10


def _stage(source, target):
    shutil.copytree(source, target)
    rules = target / "tmp" / "sigma_quadrature_rules"
    saved_rules = target.parent / f"{target.name}_rules"
    if rules.is_dir():
        shutil.copytree(rules, saved_rules)
    shutil.rmtree(target / "tmp", ignore_errors=True)
    if saved_rules.is_dir():
        shutil.copytree(saved_rules, rules)
    harness.make_writable(target)
    return target


def _run(run_dir, deck, *, allow_runtime_solve=False):
    return rank_session.run_child(lambda: harness.run_gw_jax(
        run_dir, deck, platform="gpu",
        extra_env={
            "LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE": (
                "1" if allow_runtime_solve else "0"
            ),
        },
        timeout=180,
    ), run_dir / deck)


def _assert_eqp(got, reference, *, atol=EQP_ATOL_EV):
    np.testing.assert_allclose(
        harness.eqp_column(got), harness.eqp_column(reference),
        rtol=0.0, atol=atol,
    )


@pytest.mark.gpu
def test_a_zeta_cohsex_gnppm_match_references(
        core_fixtures):
    if not harness.gpu_available():
        pytest.skip("tiny driver reference chain is the GPU core cell")

    source_a = core_fixtures / "A"
    run_a = rank_session.stage(source_a, _stage)
    _run(run_a, "cohsex.in")
    with h5py.File(run_a / "tmp" / "zeta_q.h5") as got_h5, h5py.File(
            source_a / "tmp" / "zeta_q.h5") as ref_h5:
        got = np.asarray(got_h5["zeta_q_G"])
        ref = np.asarray(ref_h5["zeta_q_G"])
    assert got.shape == ref.shape == (5, 21, 210)
    np.testing.assert_allclose(got, ref, rtol=ZETA_ATOL, atol=ZETA_ATOL)
    _assert_eqp(run_a / "cohsex_eqp0.dat", source_a / "cohsex_eqp0.dat")
    _assert_eqp(run_a / "cohsex_eqp1.dat", source_a / "cohsex_eqp1.dat")

    # The noncrossing-imag family intentionally has no shipped certified table;
    # GN therefore uses the service's announced offline-development escape hatch.
    _run(run_a, "gnppm.in", allow_runtime_solve=True)
    _assert_eqp(run_a / "gnppm_eqp0.dat", source_a / "gnppm_eqp0.dat",
                atol=GN_EQP_ATOL_EV)
    _assert_eqp(run_a / "gnppm_eqp1.dat", source_a / "gnppm_eqp1.dat",
                atol=GN_EQP_ATOL_EV)


@pytest.mark.gpu
def test_b_mpa_one_update_matches_references(core_fixtures):
    if not harness.gpu_available():
        pytest.skip("tiny driver reference chain is the GPU core cell")
    source_b = core_fixtures / "B"
    run_b = rank_session.stage(source_b, _stage)
    _run(run_b, "mpa.in", allow_runtime_solve=True)
    _assert_eqp(run_b / "mpa_eqp0.dat", source_b / "mpa_eqp0.dat",
                atol=MPA_EQP_ATOL_EV)
    _assert_eqp(run_b / "mpa_eqp1.dat", source_b / "mpa_eqp1.dat",
                atol=MPA_EQP_ATOL_EV)

    _run(run_b, "mpa_sc1.in", allow_runtime_solve=True)
    _assert_eqp(run_b / "mpa_sc1_eqp0.dat", source_b / "mpa_sc1_eqp0.dat")
    _assert_eqp(run_b / "mpa_sc1_eqp1.dat", source_b / "mpa_sc1_eqp1.dat")
    report = (run_b / "mpa_sc1.out").read_text(encoding="utf-8")
    residuals = [float(value) for value in re.findall(
        r"SC iteration: call=\d+ role=linear .*?max\|dE\|=([0-9.e+-]+)",
        report,
    )]
    assert residuals == pytest.approx([3.640626335, 0.3476364921], abs=2e-5)
    partitions = re.findall(
        r"SC iteration: call=\d+ role=linear .*?"
        r"active=(\S+) protected=(\S+) in_range=(\S+)", report,
    )
    assert partitions == [("1-3", "1-2", "1-2")] * 2
    box_line = next(line for line in report.splitlines()
                    if "SC fixed window: ω≥E_F cond:pole_tail" in line)
    assert "box=(" in box_line and "padded_box=(" in box_line
    sup, target = (float(value) for value in re.search(
        r"sup=([0-9.e+-]+)/([0-9.e+-]+)", box_line).groups())
    assert sup <= target == pytest.approx(1.0e-3)
    assert "rebuilds_this_iteration=6, rebuilds_total=6" in report
    gain = float(re.search(r"SC map gain:.*? = ([0-9.e+-]+)", report)[1])
    assert gain == pytest.approx(0.185133, abs=1e-5)
    assert "SC done: 2 GW map calls" in report
