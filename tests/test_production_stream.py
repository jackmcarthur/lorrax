"""The production driver owns stdout even when a component calls print."""

import os
import sys
import warnings

import pytest

from runtime.production_stream import ProductionStdout


def test_production_discards_incidental_stdout_but_emits_report(capsys):
    stream = ProductionStdout(debug=False, rank=0)
    stream.install()
    print("legacy HDF5 chatter")
    stream.emit("scientific report")
    stream.close()
    print("restored")

    out = capsys.readouterr().out
    assert "legacy HDF5 chatter" not in out
    assert out == "scientific report\nrestored\n"


def test_nonzero_rank_cannot_emit_and_debug_does_not_route(capsys):
    peer = ProductionStdout(debug=False, rank=2)
    peer.install()
    peer.emit("peer report")
    peer.close()

    debug = ProductionStdout(debug=True, rank=0)
    debug.install()
    print("forensic detail")
    debug.close()

    out = capsys.readouterr().out
    assert "peer report" not in out
    assert out == "forensic detail\n"


def test_production_consolidates_rank_zero_warnings_and_silences_peers(capsys):
    retained = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        owner = ProductionStdout(
            debug=False, rank=0, warning_fn=retained.append)
        owner.install()
        warnings.warn("uncertified minimax table", RuntimeWarning)
        owner.close()

        peer = ProductionStdout(
            debug=False, rank=3, warning_fn=retained.append)
        peer.install()
        warnings.warn("uncertified minimax table", RuntimeWarning)
        peer.close()

    captured = capsys.readouterr()
    assert captured.err == ""
    assert retained == ["RuntimeWarning: uncertified minimax table"]


def test_production_hides_jax_donation_hint(capsys):
    retained = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        owner = ProductionStdout(
            debug=False, rank=0, warning_fn=retained.append)
        owner.install()
        warnings.warn(
            "Some donated buffers were not usable: complex128[64,20].",
            UserWarning)
        owner.close()

    captured = capsys.readouterr()
    assert captured.err == ""
    assert retained == []


def test_failfast_survives_production_stdout_redirection(monkeypatch, capsys):
    """The fatal traceback reaches launcher stdout while normal prints do not.

    This is the exact ordering in ``gw_jax``: fail-fast is installed during
    bootstrap, then :class:`ProductionStdout` redirects ``sys.stdout``.  Run
    51's missing-restart leg exited every rank nonzero yet retained no Python
    exception because stderr vanished under srun and the redundant stdout
    copy went to ``/dev/null``.
    """
    import runtime

    class _ExitCalled(Exception):
        pass

    previous_hook = sys.excepthook
    had_sentinel = getattr(sys, "_lorrax_failfast_installed", False)
    stream = ProductionStdout(debug=False, rank=0)
    stream.install()
    monkeypatch.setattr(runtime, "_resolve_proc_count", lambda: 4)
    monkeypatch.setattr(runtime, "_resolve_proc_id", lambda: 2)
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(
        _ExitCalled(code)))
    if had_sentinel:
        del sys._lorrax_failfast_installed
    try:
        runtime.install_failfast_excepthook()
        try:
            raise RuntimeError("restart tensor is missing")
        except RuntimeError:
            exc_info = sys.exc_info()
        with pytest.raises(_ExitCalled, match="1"):
            sys.excepthook(*exc_info)
        print("ordinary component chatter")
    finally:
        sys.excepthook = previous_hook
        if had_sentinel:
            sys._lorrax_failfast_installed = True
        elif hasattr(sys, "_lorrax_failfast_installed"):
            del sys._lorrax_failfast_installed
        stream.close()

    captured = capsys.readouterr()
    assert "RuntimeError: restart tensor is missing" in captured.out
    assert "LORRAX FAIL-FAST: rank 2/4" in captured.out
    assert "ordinary component chatter" not in captured.out
    # Keep stderr redundancy too; either capture may disappear at launch.
    assert "LORRAX FAIL-FAST: rank 2/4" in captured.err
