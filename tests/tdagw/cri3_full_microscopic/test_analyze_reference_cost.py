"""Tiny standard-library parser proofs, not a GW/timing or physical validation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reference_cost", Path(__file__).with_name("analyze_reference_cost.py"))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def row(index, w=2., tau=3., wall=7.):
    return (f"SC map {index} stages (s): W response {w} | Sigma rule refit 0.1 | "
            f"Sigma rule plan 0.2 | Sigma tau sweep {tau} | Sigma exchange 0.1 | "
            f"Sigma Hartree 0 | other 1.6 | wall {wall}")


def inputs(tmp_path):
    log = tmp_path/"gwjax.out"
    receipt = dict(window_tau_pairs=23, distinct_tau_count=19, n_windows=2,
                   branches=[dict(windows=[dict(name="x", node_count=17)])])
    log.write_text("Accelerators : 16 GPU devices (test)\nchi0/W sum : 1-898\nSigma sum : 1-898\n"
        +"Sigma quadrature receipt: "+json.dumps(receipt)+"\n"+row(0)+"\n"
        +row(1, w=1., tau=4.)+"\nSC verdict: CONVERGED after 2 GW map calls\n")
    deck = tmp_path/"sc.in"
    deck.write_text("[cohsex]\nnumber_bands=898\ncompute_mode=mpa\nsigma_w_model=shared_pole\n"
                    "qp_solver=self_consistent\nhead_correction=full\nuse_band_extrapolation=true\n")
    return log, deck


def test_individual_rows_literal_counts_and_unproven_attribution(tmp_path):
    log, deck = inputs(tmp_path)
    report = helper.analyze(log, deck=deck, source_code_prefix="declared-only")
    assert [r["sigma_tau_stage_wall_seconds"] for r in report["maps"]] == [3., 4.]
    assert report["maps"][0]["tau_receipts"][0]["window_tau_pairs"] == 23
    assert report["maps"][0]["tau_receipts"][0]["distinct_tau_count"] == 19
    assert report["maps"][0]["tau_receipts"][0]["windows"][0]["node_count"] == 17
    assert report["totals_seconds"]["W response"] == 3.
    assert report["stage_sync"]["status"] == "UNPROVEN"
    assert all(r["device_kernel_seconds"] is None for r in report["maps"])
    assert report["physical_scope"]["full_frequency_shared_pole_declared"]
    assert report["physical_scope"]["both_band_sums_at_least_800"]
    assert not report["physical_scope"]["reference_state_authenticated"]
    assert not report["provenance"]["source_code_prefix_authenticated"]
    assert report["sc_status"] == "CONVERGED_LOG_DECLARATION"


def test_gpu_wall_and_reservation_are_separate_explicit_assumptions(tmp_path):
    log, deck = inputs(tmp_path)
    report = helper.analyze(log, deck=deck, gpu_count=16,
                            active_step_seconds=20, reservation_seconds=30)
    costs = report["gpu_accounting"]
    assert costs["sum_map_wall_gpu_hours"] == pytest.approx(14*16/3600)
    assert costs["active_step_gpu_hours"] == pytest.approx(20*16/3600)
    assert costs["reservation_gpu_hours"] == pytest.approx(30*16/3600)
    assert helper.analyze(log)["gpu_accounting"]["active_step_gpu_hours"] is None
    with pytest.raises(ValueError, match="conflicts"):
        helper.analyze(log, gpu_count=4)
    with pytest.raises(ValueError, match="explicit gpu_count"):
        helper.analyze(log, active_step_seconds=20)
    with pytest.raises(ValueError, match="shorter"):
        helper.analyze(log, gpu_count=16, active_step_seconds=20, reservation_seconds=10)


@pytest.mark.parametrize("status,expected", [
    ("ONE-MAP DIAGNOSTIC (not converged)", "ONE_MAP_DIAGNOSTIC"),
    ("NOT CONVERGED after 30 calls", "NOT_CONVERGED"),
    ("STALLED", "NOT_CONVERGED")])
def test_status_never_confuses_negative_or_diagnostic_with_convergence(status, expected):
    assert helper.parse_log(row(0)+"\nSC verdict: "+status)["sc_status"] == expected


def test_malformed_rows_receipts_and_duplicate_sessions_refuse():
    with pytest.raises(ValueError, match="no literal"):
        helper.parse_log("aggregate sigma.tau_sweep 50.0")
    with pytest.raises(ValueError, match="duplicate map"):
        helper.parse_log(row(0)+"\n"+row(0))
    with pytest.raises(ValueError, match="incomplete"):
        helper.parse_log("SC map 0 stages (s): W response 2 | wall 2")
    with pytest.raises(ValueError):
        helper.parse_log('Sigma quadrature receipt: {"window_tau_pairs": NaN}\n'+row(0))
    with pytest.raises(ValueError, match="literal window_tau_pairs"):
        helper.parse_log('Sigma quadrature receipt: {"window_tau_pairs": 1.5}\n'+row(0))


def test_literal_sync_is_preserved_but_does_not_become_a_kernel_measurement():
    log = 'Stage synchronization receipt: {"map":0,"synchronized":true}\n'+row(0)
    report = helper.parse_log(log)
    assert report["stage_sync"]["receipts"][0]["receipt"]["synchronized"]
    assert report["stage_sync"]["device_kernel_attribution"] == "UNPROVEN"
    assert report["maps"][0]["executable_only_seconds"] is None


def test_hashes_missing_deck_and_literal_band_authentication(tmp_path):
    log, deck = inputs(tmp_path)
    expected = helper.file_record(log)["sha256"]
    report = helper.analyze(log, expected_log_sha256=expected, source_files=[deck])
    assert report["physical_scope"]["number_bands_chi"] == 898
    assert report["physical_scope"]["full_frequency_shared_pole_declared"] is None
    assert report["provenance"]["additional_source_files"][0]["sha256"]
    with pytest.raises(ValueError, match="SHA256"):
        helper.analyze(log, expected_log_sha256="0"*64)
    deck.write_text("number_bands=750")
    with pytest.raises(ValueError, match="band count conflicts"):
        helper.analyze(log, deck=deck)


def test_cli_exclusive_atomic_publication(tmp_path, capsys):
    log, deck = inputs(tmp_path)
    output = tmp_path/"report.json"
    args = ["--log", str(log), "--deck", str(deck), "--output", str(output), "--gpu-count", "16"]
    helper.main(args)
    capsys.readouterr()
    original = output.read_bytes()
    assert json.loads(original)["status"] == "COMPLETE_REPORT_NOT_MODEL_ADMISSION"
    with pytest.raises(FileExistsError):
        helper.main(args)
    with pytest.raises(FileExistsError):
        helper.write_exclusive_atomic(output, {"wrong": True})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.pending"))
    assert not list(tmp_path.glob(".*.pending"))
