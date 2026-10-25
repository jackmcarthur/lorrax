"""Rank-skewed proposal prices agree before the first compiled group shape.

This is a pure policy/compile-cache witness with synthetic byte quotes, not
an actual MPI, allocator, physical-response or performance measurement.
"""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('minimum', [False, True])
def test_skewed_prices_choose_common_cold_warm_and_minimum_groups(monkeypatch, minimum):
    from gw import response_bank as owner
    from runtime import aot_memory as aot
    from common import gpu_utils
    from lxkit import memory

    # Local previews differ, including one rank with the minimum room.
    rooms = [1000, 900, 950, 920]
    streams = [160, 260, 220, 230] if not minimum else [910, 920, 915, 912]
    residents, workspaces = [10, 30, 20, 15], [50, 100, 90, 80]
    common_stream, common_room = max(streams), min(rooms)
    fixed = max(common_stream, max(residents)+max(workspaces))
    common_size = 1 if minimum else 6
    expected_builds = [1] if minimum else [6, 4]
    monkeypatch.setattr(gpu_utils, 'device_budget_bytes', lambda: 10000)
    monkeypatch.setattr(aot, 'compiled_new_bytes', lambda ex, *, extra=0, **_kw: ex.new+extra)
    warnings = []
    monkeypatch.setattr(memory, 'admit', lambda *a, **k: warnings.append((a, k)))
    all_builds = []

    class Ledger:
        live_stages = ()
        def __init__(self, room): self.room = room
        def room_bytes_per_rank(self, _live): return self.room
        def preview(self, *, resident_bytes_per_rank, workspace_bytes_per_rank, **_kw):
            return dict(device_budget_status=('PASS' if
                resident_bytes_per_rank+workspace_bytes_per_rank <= self.room else 'FAIL'))

    for rank in range(4):
        cache = {};new_compiles = []
        for _warm in range(2):
            proposals = [];builds = []
            # Model one all-rank reduction of each proposal input. Subsequent
            # compiled figures are common because every rank builds the same
            # signature; their existing agreed_chunk calls keep identity.
            reductions = [-common_stream, -max(residents), -max(workspaces), common_room, common_size]
            def agree(value):
                proposals.append(int(value))
                return reductions[len(proposals)-1] if len(proposals)<=len(reductions) else int(value)
            monkeypatch.setattr(aot, 'agreed_chunk', agree)
            meta = SimpleNamespace(tangential_capacity=Ledger(rooms[rank]))
            proposed, actual_fixed, room, _ = owner.response_group_size(meta, None,
                n_samples=8, carry_per_sample=100, stream_workspace=streams[rank],
                selection=(residents[rank], workspaces[rank]))
            assert (proposed, actual_fixed, room)==(common_size, fixed, common_room)

            def build(size):
                # No executable can be built before the common proposal.
                assert len(proposals)>=5
                builds.append(size)
                if size not in cache:
                    new_compiles.append(size)
                    cache[size] = SimpleNamespace(new=fixed+size*150)
                return cache[size]
            check = aot.check_chunk(proposed, build=build, fixed=actual_fixed,
                per_unit=100, room=room, stage='test response proposal')
            assert builds==expected_builds
            assert check.chunk==(1 if minimum else 4)
            assert check.recompiled is (not minimum)
            all_builds.append(builds)
        # Warm controls reuse the same two (or minimum-one) fake compiled
        # signatures, rather than manufacturing another rank-dependent shape.
        assert new_compiles==expected_builds
    assert len({tuple(b) for b in all_builds})==1
    assert bool(warnings) is minimum
