"""Host lifecycle only: no native map, screening, Sigma or convergence claim."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("pre_sigma_protocol", Path(__file__).with_name("native_frozen_map_probe.py"))
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def test_pre_sigma_success_marks_after_return_and_stops_before_broad_work(tmp_path):
    events, finished = [], []
    current = dict(frame=dict(frame="actual_accepted_input"))
    def run(value, *, out):
        assert value is current and out == tmp_path
        assert not finished
        events.append("callback-return")
    def publish(path, *, record):
        assert events == ["callback-return"]
        assert record["accepted_input_frame"] == "actual_accepted_input"
        assert record["broad_protected_Sigma_performed"] is False
        assert record["coarse_leg_performed"] is False
        assert record["new_SC_convergence_claim"] is False
        events.append(path)
    with pytest.raises(StopIteration, match="^EXTERNAL_PRE_SIGMA_PROBE_COMPLETE$"):
        owner.run_pre_sigma_only(run, current, out=tmp_path, publish=publish, finished=finished)
        pytest.fail("must not continue broad Sigma")
    assert finished == ["PRE_SIGMA_ONLY"]


def test_pre_sigma_failure_never_enters_collective_completion_publish(tmp_path):
    finished, calls = [], []
    error = ValueError("actual callback refusal")
    def run(*a, **kw):
        raise error
    with pytest.raises(ValueError) as actual:
        owner.run_pre_sigma_only(run, {}, out=tmp_path,
            publish=lambda *a, **kw: calls.append(a), finished=finished)
    assert actual.value is error and not finished and not calls


@pytest.mark.parametrize("extra", [["--bse-helper", "bse.py", "--pre-sigma-helper", "pre.py"],
    ["--pre-sigma-helper", "pre.py", "--td-helper", "td.py", "--td-deck", "td.toml"]])
def test_incompatible_modes_refuse_before_native_bootstrap(tmp_path, extra):
    base = ["--source", "missing", "--source-revision", "fake", "--setup-deck", "native.in",
            "--frames", "frames", "--out", str(tmp_path/"new")]
    with pytest.raises(SystemExit) as error:
        owner.main(base+extra)
    assert error.value.code == 2
