"""The kmeans candidate Gram and feature metric never read the budget (CPU).

The Gram's k batches and square tiles, and the metric's band and k chunks,
come from the shapes alone (TASTE 96), so a 40 GB and an 80 GB card sum in
the same order and select the same centroids.  Toy faces stand in for the
WFN loader.
"""
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def _gram_schedule(monkeypatch, budget_gb):
    import common.meta
    import common.wfn_transforms
    import isdf
    from centroid import pivoted_cholesky as pc

    nk, nb, ns, m = 6, 5, 1, 8
    rng = np.random.default_rng(7)
    psi = rng.standard_normal((nk, nb, ns, m)) + 1j * rng.standard_normal((nk, nb, ns, m))
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    loads, tiles = [], []

    def load(wfn, sym, meta, cand_idx, bispinor, mesh_xy, band_range, *,
             band_chunk_size, k_chunk_size, full_k_rows):
        rows = list(range(nk)) if full_k_rows is None else list(full_k_rows)
        loads.append(tuple(rows))
        y = psi[rows][:, band_range[0]:band_range[1]]
        x = np.ascontiguousarray(y.transpose(0, 3, 1, 2))
        return (jax.device_put(y, NamedSharding(mesh_xy, P(None, None, None, "y"))),
                jax.device_put(x, NamedSharding(mesh_xy, P(None, "x", None, None))))

    from isdf.core import gram_q0_tiled_from_psi_sm as tiled

    def record(*args, tile_width, **kwargs):
        tiles.append(int(tile_width))
        return tiled(*args, tile_width=tile_width, **kwargs)

    monkeypatch.setattr(common.wfn_transforms, "load_centroids_band_chunked", load)
    monkeypatch.setattr(isdf, "gram_q0_tiled_from_psi_sm", record)
    monkeypatch.setattr(common.meta.Meta, "from_system",
                        classmethod(lambda cls, *a, **k: SimpleNamespace()))
    stars = [[k] for k in range(nk)]
    wfn = SimpleNamespace(nelec=2, nspinor=ns, full_k_parent_groups=lambda rows: [
        (s[0], np.asarray(s)) for s in stars if s[0] in set(int(r) for r in rows)])
    G = pc.build_gram_q0_via_loadwfns(
        wfn, SimpleNamespace(nk_tot=nk), np.zeros((m, 3), np.int32),
        n_val=2, n_cond=3, mesh_xy=mesh, verbose=False,
        memory_per_device_gb=budget_gb)
    return np.asarray(G), loads, tiles


def test_candidate_gram_schedule_ignores_budget(monkeypatch):
    # 1e-6 GB is below one k's faces; 1e3 GB holds the whole Gram.  Before
    # the fixed tile these took different k batches and Gram routes.
    small = _gram_schedule(monkeypatch, 1e-6)
    large = _gram_schedule(monkeypatch, 1e3)
    assert small[1] == large[1] and small[2] == large[2]
    np.testing.assert_array_equal(small[0], large[0])
    assert np.abs(large[0]).max() > 0


def test_feature_metric_plan_ignores_budget(monkeypatch):
    import common.gpu_utils
    from centroid.sampling_metric import _metric_chunk_plan

    shapes = dict(n_parents=40, n_bands=600, n_band_shards=4, ns=2,
                  ngkmax=40000, n_grid=60 * 60 * 300, n_windows=2)
    plans = []
    for card in (40e9, 80e9):
        monkeypatch.setattr(common.gpu_utils, "device_budget_bytes", lambda c=card: c)
        plans.append(_metric_chunk_plan(**shapes))
    assert plans[0] == plans[1]
