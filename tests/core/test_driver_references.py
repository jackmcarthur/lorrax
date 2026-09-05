"""One cached run of each GW route on only the two tiny systems."""
from __future__ import annotations

import re
import shutil

import h5py
import numpy as np
import pytest

import harness


EQP_ATOL_EV = 2.0e-5
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
    result = harness.run_gw_jax(
        run_dir, deck, platform="gpu",
        extra_env={
            "LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE": (
                "1" if allow_runtime_solve else "0"
            ),
        },
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{deck} failed\n--- stdout ---\n{result.stdout[-5000:]}\n"
        f"--- stderr ---\n{result.stderr[-5000:]}"
    )
    return result


def _assert_eqp(got, reference):
    np.testing.assert_allclose(
        harness.eqp_column(got), harness.eqp_column(reference),
        rtol=0.0, atol=EQP_ATOL_EV,
    )


@pytest.mark.gpu
def test_a_zeta_cohsex_gnppm_and_b_mpa_one_update_match_references(
        core_fixtures, tmp_path):
    if not harness.gpu_available():
        pytest.skip("tiny driver reference chain is the GPU core cell")

    source_a = core_fixtures / "A"
    run_a = _stage(source_a, tmp_path / "A")
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
    _assert_eqp(run_a / "gnppm_eqp0.dat", source_a / "gnppm_eqp0.dat")
    _assert_eqp(run_a / "gnppm_eqp1.dat", source_a / "gnppm_eqp1.dat")

    source_b = core_fixtures / "B"
    run_b = _stage(source_b, tmp_path / "B")
    _run(run_b, "mpa.in", allow_runtime_solve=True)
    _assert_eqp(run_b / "mpa_eqp0.dat", source_b / "mpa_eqp0.dat")
    _assert_eqp(run_b / "mpa_eqp1.dat", source_b / "mpa_eqp1.dat")

    _run(run_b, "mpa_sc1.in", allow_runtime_solve=True)
    _assert_eqp(run_b / "mpa_sc1_eqp0.dat", source_b / "mpa_sc1_eqp0.dat")
    _assert_eqp(run_b / "mpa_sc1_eqp1.dat", source_b / "mpa_sc1_eqp1.dat")
    report = (run_b / "mpa_sc1.out").read_text(encoding="utf-8")
    residuals = [float(value) for value in re.findall(
        r"SC iteration: call=\d+ role=linear .*?max\|dE\|=([0-9.e+-]+)",
        report,
    )]
    assert residuals == pytest.approx([3.640565708, 0.3476255047], abs=2e-5)
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
    assert "SC map gain:" in report and "= 0.185134" in report
    assert "SC done: 2 GW map calls" in report
