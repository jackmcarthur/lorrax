"""Contract for ``runtime.run_session.RunSession``, the one driver session."""

import sys
from types import SimpleNamespace

import pytest

from runtime.run_session import RUNTIME_ROW, RunSession


class _Report:
    """The report protocol the session needs, recording every call."""

    def __init__(self, path, *, runtime, debug, stdout, **labels):
        self.path = path
        self.labels = labels
        self.stdout = stdout
        self.calls = []

    def legacy_print(self, *args, **kwargs):
        self.calls.append(("legacy", " ".join(str(a) for a in args)))

    def timings(self, stages, *, wall):
        self.calls.append(("timings", tuple(stages), wall))

    def warnings(self):
        self.calls.append(("warnings",))

    def files(self, rows):
        self.calls.append(("files", tuple(rows)))

    def finish(self, *, status="completed"):
        self.calls.append(("finish", status))


def _runtime(count=1):
    return SimpleNamespace(process_index=0, process_count=count, facts={
        "elapsed": {"distributed": 1.0, "mesh": 2.0, "total": 3.0}})


@pytest.fixture()
def barrier_calls(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "common.collectives",
                        SimpleNamespace(barrier=calls.append))
    for name in ("JAX_PROCESS_COUNT", "JAX_NUM_PROCESSES", "SLURM_NTASKS"):
        monkeypatch.delenv(name, raising=False)
    return calls


def test_completed_run_prints_stages_files_and_leaves_together(
        tmp_path, barrier_calls):
    stdout_before = sys.stdout
    written = tmp_path / "out.h5"
    written.write_bytes(b"x")
    with RunSession(_runtime(), "drv", _Report, str(tmp_path / "drv.out"),
                    stages=(("solve", "drv.solve"),), driver_name="d") as run:
        assert run.report.labels == {"driver_name": "d"}
        run.complete(files=[("result", "written", str(written)),
                            ("missing", "written", str(tmp_path / "no.h5"))])
    assert sys.stdout is stdout_before
    names = [call[0] for call in run.report.calls]
    assert names == ["timings", "warnings", "files", "finish"]
    stages = run.report.calls[0][1]
    assert stages[0][0] == RUNTIME_ROW and stages[1][0] == "solve"
    files = dict((label, state) for label, state, _ in run.report.calls[2][1])
    assert files == {"result": "written", "missing": "absent"}
    assert run.report.calls[3] == ("finish", "completed")
    assert barrier_calls == ["drv.report_written"]


def test_a_refusal_closes_the_report_and_propagates(tmp_path, barrier_calls):
    stdout_before = sys.stdout
    with pytest.raises(ValueError, match="GATE thing_refused"):
        with RunSession(_runtime(), "drv", _Report,
                        str(tmp_path / "drv.out"), stages=()) as run:
            raise ValueError("GATE thing_refused: got x; want y")
    assert sys.stdout is stdout_before
    assert run.report.calls[-1] == (
        "finish", "REFUSED (ValueError: GATE thing_refused)")
    assert barrier_calls == []          # a refusal never enters a collective


def test_a_zero_exit_is_not_a_refusal(tmp_path, barrier_calls):
    with pytest.raises(SystemExit):
        with RunSession(_runtime(), "drv", _Report,
                        str(tmp_path / "drv.out")) as run:
            raise SystemExit(0)
    assert run.report.calls == []


def test_an_unpublished_artifact_refuses(tmp_path, barrier_calls):
    with pytest.raises(RuntimeError, match="GATE artifacts_unpublished"):
        with RunSession(_runtime(), "drv", _Report,
                        str(tmp_path / "drv.out"), stages=()) as run:
            run.complete(published=[(
                "result", str(tmp_path / "absent.h5"), True)])
    finishes = [c for c in run.report.calls if c[0] == "finish"]
    assert finishes == [("finish",
                         "REFUSED (RuntimeError: GATE artifacts_unpublished)")]
    assert [c[0] for c in run.report.calls].count("timings") == 1


def test_a_split_launch_refuses_before_the_report_opens(
        tmp_path, barrier_calls, monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "4")
    opened = []

    class _Opened(_Report):
        def __init__(self, *args, **kwargs):
            opened.append(True)
            super().__init__(*args, **kwargs)

    with pytest.raises(SystemExit, match="world of 1"):
        with RunSession(_runtime(count=1), "drv", _Opened,
                        str(tmp_path / "drv.out")):
            pass
    assert opened == []


def test_the_session_without_a_report_keeps_warnings(tmp_path, barrier_calls):
    with RunSession(_runtime(), "drv", None) as run:
        run._stdout.warning_fn("UserWarning: kept")
        run.complete()
    assert run.warnings == ["UserWarning: kept"]
    assert barrier_calls == ["drv.report_written"]
