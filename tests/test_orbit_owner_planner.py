"""Metadata-only owner search agrees with exhaustive materialized orbit tables."""
from types import SimpleNamespace

import numpy as np

from isdf.zeta_mubatch import best_owner_orbit_batches, owner_orbit_batches


def test_owner_search_preserves_exhaustive_tables_and_closed_orbits():
    for sizes in ((1, 1, 1, 2, 3, 5, 8), (6, 6, 6, 1, 2, 3), (12, 12, 4, 3, 2, 1)):
        logical, mu = sum(sizes), sum(sizes) + 7
        perm = np.broadcast_to(np.arange(mu), (3, mu)).copy()
        start = 0
        for size in sizes:
            group = np.arange(start, start + size)
            perm[1, group], perm[2, group] = np.roll(group, 1), np.roll(group, -1)
            start += size
        plan = SimpleNamespace(
            n_centroid_packed=mu, sym_idx=np.array([0, 1, 2]), sym_perm=perm,
            L_table=np.arange(9 * mu).reshape(3, mu, 3),
            layout=SimpleNamespace(axis=SimpleNamespace(active_mask=np.arange(mu) < logical)))
        for ranks in (4, 64, 144):
            for cap in (1, 7, 13, 29):
                best = None
                for c in range(cap, 0, -1):
                    candidate = owner_orbit_batches(plan, mu, ranks, c_target=c)
                    if best is not None and candidate.c > cap:
                        continue
                    cost = candidate.n_batch * (candidate.c + 1)
                    if best is None or cost < best[0]:
                        best = cost, candidate
                got = best_owner_orbit_batches(plan, mu, ranks, c_max=cap)
                expected = best[1]
                assert (got.c, got.n_batch) == (expected.c, expected.n_batch)
                for name in ('mu', 'left_perm', 'left_L', 'slot_of_packed'):
                    a, b = getattr(got, name), getattr(expected, name)
                    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                # An independent transport invariant: every active centroid
                # appears once, and every mapped source stays in its owner bin.
                assert sorted(got.mu[got.mu >= 0]) == list(range(logical))
                for batch in range(got.n_batch):
                    for owner in range(ranks):
                        local = got.mu[batch, owner * got.c:(owner + 1) * got.c]
                        live = local >= 0
                        for row in range(3):
                            source = local[got.left_perm[batch, owner, row, live]]
                            np.testing.assert_array_equal(source, perm[row, local[live]])
                            np.testing.assert_array_equal(
                                got.left_L[batch, owner, row, live], plan.L_table[row, local[live]])
