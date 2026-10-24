"""Allocated orbit-batch slots determine the existing host/disk choice."""
from gw.gflat_memory_model import zstore_layout


def test_resolved_orbit_slots_reprice_the_store_at_the_same_quota():
    # Captured Fe P64: the proposal had 1152 slots; two executed batches
    # allocate 2*1024. Q=59 and the padded G carrier=832 are unchanged.
    quota = 20_000_000
    kwargs = dict(n_vertex=1, n_ranks=64, host_budget_bytes=quota)
    assert zstore_layout(59, 1152, 832, **kwargs) == (14_137_344.0, 'host')
    actual, placement = zstore_layout(59, 2048, 832, **kwargs)
    assert (actual, placement) == (25_133_056.0, 'disk')
    for available, expected in ((actual, 'host'), (actual - 1, 'disk')):
        assert zstore_layout(
            59, 2048, 832, n_vertex=1, n_ranks=64,
            host_budget_bytes=available) == (actual, expected)
    # Each current channel owns a separate store at the same padded shape.
    assert zstore_layout(
        59, 2048, 832, n_vertex=3, n_ranks=64,
        host_budget_bytes=3 * actual) == (3 * actual, 'host')
