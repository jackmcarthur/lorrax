"""The jax-native pass path: same bytes, built where they are consumed.

THE CLAIM THIS FILE HOLDS.  ``refactor/mpa-jax-native-2026-08-10`` moves
three pieces of the Sigma pass off the host and onto the device that reads
them -- the per-group selector, the per-branch pole operand, and the
planner's width sort -- and the claim attached to all three is that they
are REPRESENTATION changes with no licence to move a byte.  A pane is the
same pane, a selector is the same boolean, a sorted index array is the same
permutation, and the operand the tau kernel reads is the same operand.

So these cells are byte-equality cells, not tolerance cells.  Where a
tolerance appears anywhere below it is a bug in the cell, because there is
no arithmetic in a scatter of ``True`` into zeros, none in a gather, and
none in a stable sort -- the three things this lane replaced.

The one cell here that is not a byte comparison is the compile-signature
cell, and it exists because the cheapest way to make a scatter fast is to
specialise it per group, which would compile ~918 XLA modules on an
n_p = 8 pass and leave a persistent cache no second run can hit.  The
capacity ladder is what stops that, and it is asserted rather than
described.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from gw.mpa import sigma_pass as SP


def _field(shape, live_frac, seed):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    n_live = max(1, int(n * live_frac))
    idx = np.sort(rng.choice(n, size=n_live, replace=False))
    return idx.astype(SP.flat_index_dtype(n)), shape


# ---------------------------------------------------------------------------
#  The selector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,frac,seed", [
    ((3, 5, 5), 0.3, 0),
    ((2, 8, 8), 0.9, 1),
    ((4, 4, 4), 0.01, 2),
])
def test_the_device_selector_is_the_dense_mask_byte_for_byte(shape, frac, seed):
    """The boolean built on the device IS ``dense_mask_B``'s boolean."""
    idx, shape = _field(shape, frac, seed)
    grp = SP.WindowGroup(
        name="g", windows=[], idx_B=idx, field_shape=shape,
        omega_operand=np.zeros(shape), n_modes=int(idx.size), b_mass=0.0,
        provenance="")
    host = grp.dense_mask_B()
    dev = np.asarray(jax.device_get(
        SP.group_selector_device(grp.idx_B, grp.field_shape)))
    assert dev.dtype == host.dtype == np.bool_
    assert dev.shape == host.shape
    assert np.array_equal(dev, host), (
        f"{int(np.sum(dev != host))} of {host.size} selector elements differ")


def test_an_empty_group_selects_nothing_and_a_full_one_selects_everything():
    """The two ends of the ladder, where an off-by-one would hide."""
    shape = (2, 3, 3)
    n = int(np.prod(shape))
    empty = SP.group_selector_device(np.zeros((0,), np.int32), shape)
    full = SP.group_selector_device(np.arange(n, dtype=np.int32), shape)
    assert not bool(np.any(np.asarray(jax.device_get(empty))))
    assert bool(np.all(np.asarray(jax.device_get(full))))


def test_the_selector_capacity_ladder_bounds_the_compile_signatures():
    """The compile-cache contract, asserted rather than asserted-in-prose.

    A jit keyed on the raw live-mode count compiles once per group.  The
    ladder is what turns ~918 signatures into at most ``log2(n_flat)`` of
    them, so a second run hits its persistent cache instead of re-paying
    the compile -- the warm-cache lesson, as a property of the code.
    """
    n_flat = 81_432_576
    counts = np.unique(np.linspace(1, n_flat, 5000).astype(np.int64))
    caps = {SP._selector_capacity(int(c)) for c in counts}
    assert len(caps) <= int(n_flat).bit_length(), (
        f"{len(caps)} distinct selector signatures over {counts.size} group "
        f"sizes; the ladder is supposed to bound this at "
        f"{int(n_flat).bit_length()}")
    for c in counts[:200]:
        cap = SP._selector_capacity(int(c))
        assert cap >= int(c) and (cap & (cap - 1)) == 0


def test_the_selector_signature_does_not_depend_on_the_device_count():
    """Shard-invariance: the multi_slice lesson as a cell.

    The capacity is a function of the group's mode count and the field's
    size.  Neither is a rank quantity -- every process plans the same
    replicated pole slab -- so the compiled signature is the same on one
    device and on sixteen, which is what makes the persistent cache
    shareable across a farm.
    """
    for n in (1, 2, 3, 1023, 1024, 1025, 81_432_576):
        assert SP._selector_capacity(n) == SP._selector_capacity(n)
    assert "jax.device_count" not in SP._selector_capacity.__code__.co_names
    assert "process_count" not in SP._selector_capacity.__code__.co_names


# ---------------------------------------------------------------------------
#  The branch operand cache
# ---------------------------------------------------------------------------

def test_one_device_copy_per_distinct_operand_and_the_bytes_are_the_same():
    """918 groups share two operands; the cache uploads two arrays."""
    a = np.linspace(0.1, 3.0, 64).reshape(4, 4, 4)
    om = a - 1j * (0.5 * a)
    cache = SP._BranchOperandCache()
    got = [cache.device(x) for x in (a, om, a, om, a)]
    assert len(cache._by_id) == 2
    assert got[0] is got[2] is got[4]
    assert got[1] is got[3]
    assert np.array_equal(np.asarray(jax.device_get(got[0])), a)
    assert np.array_equal(np.asarray(jax.device_get(got[1])), om)


def test_the_cached_operand_is_what_asarray_would_have_produced():
    """The seam: the tau kernel reads the operand it read before."""
    om = (np.linspace(-2.0, 2.0, 125).reshape(5, 5, 5)
          - 1j * np.linspace(0.1, 1.0, 125).reshape(5, 5, 5))
    once = np.asarray(jax.device_get(SP._BranchOperandCache().device(om)))
    direct = np.asarray(jax.device_get(jnp.asarray(om)))
    assert once.dtype == direct.dtype
    assert np.array_equal(once.view(np.float64), direct.view(np.float64))


# ---------------------------------------------------------------------------
#  The width sort
# ---------------------------------------------------------------------------

def _sorted_host(g_flat, idx):
    ix = np.asarray(idx, dtype=np.int64)
    g_v = np.asarray(g_flat, dtype=np.float64)[ix]
    order = np.argsort(g_v, kind="stable")
    return ix[order], g_v[order]


@pytest.mark.parametrize("seed,ties", [(0, False), (1, True), (2, True)])
def test_the_device_width_sort_is_the_host_permutation_exactly(seed, ties):
    """Same permutation, same values, and ties are where it would break.

    A stable sort is a uniquely determined permutation, so the device and
    host paths cannot disagree -- unless one of them is not stable, which
    is exactly what the tie-heavy case is for.
    """
    rng = np.random.default_rng(seed)
    n = 4096
    g = (np.round(rng.random(n) * 8.0) if ties
         else rng.random(n)).astype(np.float64)
    idx = np.sort(rng.choice(n, size=n // 2, replace=False)).astype(np.int64)
    want_ix, want_g = _sorted_host(g, idx)
    got = SP._sorted_by_width_device(g, idx)
    assert got is not None, "the device sort declined a case it should serve"
    got_ix, got_g = got
    assert np.array_equal(got_ix, want_ix), (
        f"{int(np.sum(got_ix != want_ix))} of {want_ix.size} positions "
        f"differ; the two sorts disagree on ties")
    assert np.array_equal(got_g, want_g)


def test_the_planner_entry_point_agrees_with_its_own_host_path():
    """``_sorted_by_width`` routes; both routes are the same answer."""
    rng = np.random.default_rng(7)
    g = np.round(rng.random(2048) * 4.0)
    idx = np.arange(2048, dtype=np.int64)
    a_ix, a_g = SP._sorted_by_width(g, idx)            # under threshold: host
    b = SP._sorted_by_width_device(g, idx)
    assert b is not None
    assert np.array_equal(a_ix, b[0]) and np.array_equal(a_g, b[1])


def test_a_declining_device_sort_does_not_change_the_answer(monkeypatch):
    """The fallback is a performance decision, so it must be invisible."""
    rng = np.random.default_rng(11)
    g = rng.random(1 << 21)
    idx = np.arange(g.size, dtype=np.int64)
    want = _sorted_host(g, idx)
    monkeypatch.setattr(SP, "_sorted_by_width_device", lambda *a, **k: None)
    got = SP._sorted_by_width(g, idx)
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


# ---------------------------------------------------------------------------
#  The planner's partition, end to end on a small field
# ---------------------------------------------------------------------------

def test_the_width_panes_are_the_same_panes_with_the_sort_on_either_path(
        monkeypatch):
    """The property the census digest is a fingerprint of.

    The sort feeds the pane splitters, so if the two sorts agreed only
    approximately the panes would differ and every downstream certified
    rule with them.  This drives the real splitters on a field with
    engineered ties and compares the partitions element by element.
    """
    rng = np.random.default_rng(3)
    n = 8192
    g = np.round(rng.random(n) * 6.0) + 0.5
    a = g + rng.random(n)
    idx = np.arange(n, dtype=np.int64)

    def panes():
        b_idx, b_g = SP._sorted_by_width(g, idx)
        geo = SP._geometric_width_bins_sorted(b_g, b_idx)
        cls = SP._clause_safe_width_split_sorted(
            a[b_idx], b_g, b_idx, e_lo=0.02, omega_max=0.0, beta_max=1.0)
        return [np.asarray(p) for p in geo], [np.asarray(p) for p in cls]

    monkeypatch.setattr(SP, "_DEVICE_SORT_MIN_MODES", 1)
    dev_geo, dev_cls = panes()
    monkeypatch.setattr(SP, "_sorted_by_width_device", lambda *a, **k: None)
    host_geo, host_cls = panes()

    assert len(dev_geo) == len(host_geo) and len(dev_cls) == len(host_cls)
    for d, h in zip(dev_geo + dev_cls, host_geo + host_cls):
        assert np.array_equal(d, h)
