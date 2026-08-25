"""The production driver owns stdout even when a component calls print."""

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
