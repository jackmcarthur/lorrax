"""gw.comm_model: the planner's three comm rules (pure Python, no devices)."""
import math

from gw import comm_model as cm


def test_min_efficient_payload_is_the_20_percent_latency_point():
    for peers in (3, 15, 99, 999):
        v = cm.min_efficient_payload(peers)
        latency = cm.comm_time(0, peers)
        assert math.isclose(latency / cm.comm_time(v, peers), 0.2)


def test_split_calls_and_overlap():
    gb, mb = 2**30, 2**20
    assert cm.split_calls(gb, 16 * mb, 15) == (64, True)     # 16 MB >= 5.7 MB
    assert cm.split_calls(gb, 1 * mb, 15) == (1024, False)   # latency-bound
    assert cm.split_calls(gb, 4 * gb, 15) == (1, True)
    assert cm.split_calls(0, mb, 15) == (0, True)
    # Double buffering: the longer stream plus one step of the shorter.
    assert cm.overlapped_time(3.0, 4.0, 100) == 4.0 + 3.0 / 100
    assert cm.overlapped_time(3.0, 4.0, 1) == 7.0             # no overlap
