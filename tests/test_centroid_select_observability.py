"""Refusal and phase-receipt contracts for centroid pruning."""
from __future__ import annotations



def test_distributed_gram_diagnostic_excludes_padding_and_always_executes():
    import jax.numpy as jnp
    from centroid import pivoted_cholesky as pc

    gram = jnp.diag(jnp.asarray([7.0, 3.0, 5.0, 0.0]))
    quiet = pc._report_logical_gram_diagonal(
        gram, 3, verbose=False,
        print_fn=lambda _text: (_ for _ in ()).throw(AssertionError()))
    lines = []
    loud = pc._report_logical_gram_diagonal(
        gram, 3, verbose=True, print_fn=lines.append)

    assert tuple(map(float, quiet)) == (3.0, 7.0)
    assert tuple(map(float, loud)) == (3.0, 7.0)
    assert lines == [
        "[pivoted_cholesky] G built, shape=(3, 3), "
        "diag range [3.000e+00, 7.000e+00]"]


def test_select_phase_receipts_name_lower_compile_and_execute(monkeypatch):
    from centroid import pivoted_cholesky as pc

    class _Compiled:
        def __call__(self, *operands):
            return operands[0]

    class _Lowered:
        def compile(self):
            return _Compiled()

    class _Step:
        def lower(self, *operands):
            return _Lowered()

    lines = []
    monkeypatch.setattr(pc.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(pc, "process_rank", lambda: 0)
    got = pc._run_select_with_progress(
        _Step(), (17,),
        n_candidates=1028, n_groups=25, point_budget=800,
        print_fn=lines.append, start_progress=lambda: None)
    assert got == 17
    for phase in ("lowering", "compilation", "execution"):
        assert any(
            f"phase={phase} state=start" in line for line in lines), lines
        assert any(
            f"phase={phase} state=done" in line for line in lines), lines
    assert "candidates=1028 groups=25 point_budget=800" in lines[0]


def test_select_heartbeat_survives_skew_on_nonroot(monkeypatch):
    import io
    import time
    from centroid import pivoted_cholesky as pc

    stream = io.StringIO()
    monkeypatch.setattr(pc, "_SELECT_HEARTBEAT_S", 0.01)
    monkeypatch.setattr(pc, "process_rank", lambda: 3)
    monkeypatch.setattr(pc, "failure_output_streams", lambda: (stream,))
    monkeypatch.setattr(pc.jax, "block_until_ready", lambda value: value)

    class Step:
        def lower(self, *operands):
            time.sleep(0.08)
            return self

        def compile(self):
            return lambda value: value

    got = pc._run_select_with_progress(
        Step(), (17,), n_candidates=1028, n_groups=25, point_budget=800,
        print_fn=lambda text: None, start_progress=lambda: None)
    assert got == 17
    text = stream.getvalue()
    assert "rank=3 still working phase=lowering" in text
    assert "elapsed_s=" in text
    assert "candidates=1028 groups=25 point_budget=800" in text
    time.sleep(0.03)
    assert stream.getvalue() == text  # the reporter stopped with the selector


def test_kmeans_cli_has_no_process_local_budget():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "centroid.kmeans_cli", "--help"],
        capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    assert "--prune-time-budget-seconds" not in out.stdout
