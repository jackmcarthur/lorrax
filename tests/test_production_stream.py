"""The production driver owns stdout even when a component calls print."""

import warnings

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
