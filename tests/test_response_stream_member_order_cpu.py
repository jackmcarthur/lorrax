"""A streamed response pair must reach its ORIGINAL frequency and q span."""
from types import SimpleNamespace

import numpy as np
import pytest

from gw.response_bank import _sample_processing_order, _stream_member_reader


MEMBERS = [16, 20, 17, 0, 1, 2, 3, 4, 5, 6, 18, 7, 8, 9, 10, 11, 12, 19, 13, 14, 15]


def _requests(committed=(), *, split_lines=False):
    ordered = _sample_processing_order(MEMBERS, (0, 16))
    requests = []
    for row, sample in ordered:
        if sample in committed:
            continue
        spans = (None,) if sample < 16 and not split_lines else ((0, 16), (16, 32), (32, 48), (48, 64))
        requests.extend((row, span) for span in spans)
    return ordered, requests


@pytest.mark.parametrize("committed,split_lines", [((), False), ((16, 1, 18), False),
                                                   ((20, 7), True)])
def test_actual_fine_member_order_and_partial_commits_preserve_frequency_pairs(committed, split_lines):
    ordered, requests = _requests(committed, split_lines=split_lines)
    assert [sample for _, sample in ordered] == [16, 20, 17, 18, 19]+list(range(16))
    seen = []
    def reader(spans):
        for start, stop, span in spans:
            seen.append((start//2, span))
            assert stop == start+2
            nq = 64 if span is None else span[1]-span[0]
            yield np.stack([np.full((nq, 1, 1), start), np.full((nq, 1, 1), start+1)])
    _, take = _stream_member_reader(SimpleNamespace(q=64, reader=reader), requests)
    for row, span in requests:
        result = take(row, span)
        np.testing.assert_array_equal(result[:, 0, 0, 0], [2*row, 2*row+1])
    assert seen == requests


def test_reader_refuses_wrong_row_or_same_row_wrong_span_before_silent_misassociation():
    bank = SimpleNamespace(q=64, reader=lambda spans: iter([np.zeros((2, 16, 1, 1))]))
    for request in ((4, (0, 16)), (3, (16, 32))):
        _, take = _stream_member_reader(bank, [(3, (0, 16))])
        with pytest.raises(ValueError, match="response_stream_member"):
            take(*request)
    bank.reader = lambda spans: iter([np.zeros((2, 64, 1, 1))])
    _, take = _stream_member_reader(bank, [(3, (0, 16))])
    with pytest.raises(ValueError, match="shape mismatch"):
        take(3, (0, 16))


def test_real_tiny_file_bank_retains_value_derivative_and_q_span_association(tmp_path):
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from file_io.slab_io import StreamedBank
    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    nq, side = 64, 2
    values = (np.arange(42)[:, None, None, None]*10000
              +np.arange(nq)[None, :, None, None]*100
              +np.arange(side*side).reshape(1, 1, side, side)).astype(complex)
    values += 1j*(values+3)
    bank = StreamedBank(mesh, root=tmp_path, label="frequency-pairs", kind="file",
        n_out=42, q=nq, segments=((side, side, ((0, 0, 0, 0, side, side),)),), tile=(side, side))
    try:
        assert bank.fits
        device = jax.device_put(values, NamedSharding(mesh, P(None, None, "x", "y")))
        bank.put(0, device, [(o, o) for o in range(42)])
        bank.commit()
        _, requests = _requests((1, 18))
        _, take = _stream_member_reader(bank, requests)
        for row, span in requests:
            selected = slice(None) if span is None else slice(*span)
            np.testing.assert_array_equal(np.asarray(take(row, span)), values[2*row:2*row+2, selected])
    finally:
        bank.release()
