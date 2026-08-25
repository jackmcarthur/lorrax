"""THE FOUR SIBLING CANONICALIZATIONS, gated where they can be gated: on CPU.

``tests/test_jax_cache_contract.py`` is the standing gate, and it needs four
processes.  These cells need none: every one of the four fixes turns on a
PURE partition function whose two properties — *the partition is still a
partition* and *every rank gets the same shape* — are properties of the
arithmetic, not of the device count.  Gating them here means a regression is
caught by ``pytest`` in seconds instead of by a P=4 leg in an hour, and it
means the fixes stay gated on a machine with no GPU at all.

WHAT EACH FIX HAD TO PRESERVE
-----------------------------
A canonicalization that changes the answer is not a canonicalization.  For
each site the invariant is the same shape and is asserted twice, once for
correctness and once for uniformity:

    COVERAGE   the global work items are covered EXACTLY ONCE across ranks,
               so no sample, band or slot is gained or lost;
    UNIFORMITY every rank's window has the SAME EXTENT, because the extent
               is the traced shape and a second extent is a second compiled
               program with a cache key exactly one rank holds.

The fifth site (``local_share``'s empty share) is red-listed, not fixed, and
has no cell here — it has a dated row in ``cache_key_lint.ALLOW`` and in
``tests/known_failures/2026-08-10-jax-cache-contract.md``.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_pure(rel: str, name: str):
    """Exec ONE top-level function out of a module, with no imports run.

    ``src/gw/kin_ion_io.py`` and ``src/centroid/charge_density.py`` both
    import the FFI subpackage at module scope, which REFUSES on a machine
    with no host ``.so`` — that is deliberate (the FFI layer is required)
    and it would make these cells un-runnable off-cluster for a reason that
    has nothing to do with what they check.  The functions under test are
    pure integer arithmetic over ``numpy``, so they are lifted out directly.
    """
    src = (REPO_ROOT / rel).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module([node], []), rel, "exec"), ns)
            return ns[name]
    raise AssertionError(f"{rel} has no top-level {name}()")


# ---------------------------------------------------------------------------
# SIBLING 2 — gw/kin_ion_io.py::rho_work_items
# ---------------------------------------------------------------------------
#: ``(nk, nocc, world)``.  The first ten are the cases the pinned contract
#: cells in ``test_sanity_gates_jax.py`` already use — quoted rather than
#: invented so the two gates cannot drift — plus prime and awkward band
#: counts, which are where a divisor rule earns its keep or fails.
_RHO_CASES = [
    (9, 26, 1), (9, 26, 4), (9, 26, 16), (9, 26, 64), (144, 26, 1),
    (144, 26, 16), (144, 26, 80), (144, 26, 144), (144, 26, 512), (4, 3, 64),
    (2, 101, 64), (16, 60, 256), (1, 1, 8), (7, 12, 29), (3, 97, 100),
]


@pytest.mark.parametrize("nk,nocc,world", _RHO_CASES)
def test_rho_work_items_are_uniform_width_and_cover_once(nk, nocc, world):
    """Sibling 2: no ragged band chunk, and still a partition."""
    f = _load_pure("src/gw/kin_ion_io.py", "rho_work_items")
    items = f(nk, nocc, world)
    widths = {hi - lo for _, lo, hi in items}
    assert len(widths) == 1, (
        f"nk={nk} nocc={nocc} P={world}: band chunks have widths {widths}.  "
        f"Each distinct width is a separate compiled to_box and a separate "
        f"density IFFT, held by whichever rank drew that chunk and by no "
        f"other — the class-B divergence this fix removed.")
    seen = set()
    for ik, lo, hi in items:
        for b in range(lo, hi):
            assert (ik, b) not in seen, f"(k={ik}, band={b}) counted twice"
            seen.add((ik, b))
    assert len(seen) == nk * nocc, (
        f"covered {len(seen)} of {nk * nocc} (k, band) pairs")


def test_rho_work_items_still_is_the_serial_sweep_at_p_le_nk():
    """THE BIT-PARITY PRECONDITION the fix must not disturb."""
    f = _load_pure("src/gw/kin_ion_io.py", "rho_work_items")
    for world in (1, 2, 9, 144):
        assert f(144, 26, world) == [(ik, 0, 26) for ik in range(144)]


def test_rho_work_items_applies_the_band_memory_bound_without_ragged_shapes():
    """The explicit deck bound is stronger than the P<=nk serial shape."""
    f = _load_pure("src/gw/kin_ion_io.py", "rho_work_items")
    items = f(36, 130, 16, max_bands_per_item=16)
    assert len(items) == 36 * 10
    assert {(lo, hi) for _, lo, hi in items} == {
        (i * 13, (i + 1) * 13) for i in range(10)}
    assert {hi - lo for _, lo, hi in items} == {13}

    # A looser memory bound must preserve the incumbent parallel divisor.
    incumbent = f(9, 26, 64)
    assert f(9, 26, 64, max_bands_per_item=26) == incumbent


def test_rho_work_items_balance_is_unchanged():
    f = _load_pure("src/gw/kin_ion_io.py", "rho_work_items")
    items = f(9, 26, 4)
    counts = [len(items[r::4]) for r in range(4)]
    assert max(counts) - min(counts) <= 1 and sum(counts) == len(items)


def test_the_divisor_rule_never_asks_for_more_chunks_than_the_budget():
    """The snap is DOWNWARD: chunks may only get bigger, never smaller.

    The memory budget above this function sized the chunk; a rule that
    rounded UP would quietly hand the sweep more, smaller FFT batches than
    the budget allowed for.
    """
    f = _load_pure("src/gw/kin_ion_io.py", "rho_work_items")
    for nk, nocc, world in _RHO_CASES:
        target = max(1, min(-(-world // max(nk, 1)), nocc))
        n_chunks = len({(lo, hi) for _, lo, hi in f(nk, nocc, world)})
        assert n_chunks <= target, (
            f"nk={nk} nocc={nocc} P={world}: {n_chunks} chunks against a "
            f"budget of {target}")
        assert nocc % n_chunks == 0, "the chunk count must divide nocc"


# ---------------------------------------------------------------------------
# SIBLING 3 — centroid/charge_density.py::_uniform_band_windows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("b_lo,b_hi,width", [
    (0, 10, 4), (0, 12, 4), (0, 3, 10), (0, 1, 1), (5, 17, 5), (0, 10, 3),
    (0, 600, 128), (0, 601, 128), (3, 4, 9), (0, 2, 1), (7, 100, 13),
])
def test_uniform_band_windows_are_uniform_and_cover_once(b_lo, b_hi, width):
    """Sibling 3: equal-width windows that OVERLAP rather than shorten."""
    f = _load_pure("src/centroid/charge_density.py", "_uniform_band_windows")
    windows = f(b_lo, b_hi, width)
    assert windows, "a non-empty band range produced no window"
    widths = {int(m.shape[0]) for _, m in windows}
    assert len(widths) == 1, (
        f"windows have widths {widths}: a short window is a second compiled "
        f"to_rbox IFFT that exactly one rank owns")
    w = widths.pop()
    assert w == min(width, b_hi - b_lo)
    for lo, m in windows:
        assert b_lo <= lo and lo + w <= b_hi, f"window {lo}..{lo + w} is out of range"
        assert set(np.unique(m)) <= {0.0, 1.0}, "the mask must be 0/1"
    cover = np.zeros(b_hi - b_lo, dtype=int)
    for lo, m in windows:
        for j, v in enumerate(m):
            if v:
                cover[lo + j - b_lo] += 1
    assert np.all(cover == 1), (
        f"bands covered {sorted(set(cover.tolist()))} times, not exactly once")


def test_uniform_band_windows_is_empty_only_for_an_empty_range():
    f = _load_pure("src/centroid/charge_density.py", "_uniform_band_windows")
    assert f(4, 4, 8) == []
    assert f(9, 3, 8) == []
    assert len(f(0, 1, 8)) == 1


def test_a_masked_row_contributes_exactly_zero():
    """The exactness the fix rests on, stated as arithmetic.

    The summand is ``|psi|^2 * w_k``: finite and non-negative.  A masked row
    is multiplied by ``0.0``, so it contributes exactly zero rather than
    'approximately nothing' — which is what lets the overlap be removed
    without touching the values that remain.
    """
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((3, 4, 2)) + 1j * rng.standard_normal((3, 4, 2))
    w = np.abs(psi) ** 2
    mask = np.array([1.0, 1.0, 0.0, 0.0]).reshape(1, -1, 1)
    assert np.array_equal((w * mask)[:, 2:, :], np.zeros_like(w[:, 2:, :]))
    assert np.array_equal((w * mask)[:, :2, :], w[:, :2, :]), (
        "multiplying by 1.0 must be the identity, or the unmasked rows moved")


# ---------------------------------------------------------------------------
# SIBLING 1 — bse/vq_interp.py, the mini-BZ slab
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_kept,nranks", [
    (1, 1), (7, 1), (10, 4), (26, 4), (40, 8), (2621440, 4), (3, 8), (10, 16),
    (97, 5), (1, 64),
])
def test_the_minibz_chunking_is_uniform_and_covers_every_slot_once(
        n_kept, nranks):
    """Sibling 1: one extent on every rank, the same global slots.

    The arithmetic is lifted rather than imported — ``vq_interp`` pulls in
    the BSE stack — and it is three lines, so it is quoted here in the form
    the source uses.  ``test_the_minibz_chunking_matches_the_source`` below
    is what stops this copy drifting away from it.
    """
    chunk = max(1, -(-n_kept // max(nranks, 1)))
    cover = np.zeros(n_kept, dtype=int)
    extents = set()
    for rank in range(nranks):
        lo = rank * chunk
        keep = (lo + np.arange(chunk)) < n_kept
        extents.add(chunk)                       # the TRACED shape
        for s in (lo + np.arange(chunk))[keep]:
            cover[s] += 1
    assert len(extents) == 1, f"ranks traced extents {extents}"
    assert np.all(cover == 1), (
        f"n_kept={n_kept} nranks={nranks}: slots covered "
        f"{sorted(set(cover.tolist()))} times, not exactly once")


def test_the_minibz_chunking_matches_the_source():
    """The source still spells the uniform chunk, and the guard is gone.

    A copy of an algorithm in a test is only a gate while it is still the
    same algorithm; this cell is what makes the copy above honest.
    """
    src = (REPO_ROOT / "src" / "bse" / "vq_interp.py").read_text()
    assert "chunk = max(1, -(-n_kept // max(nranks, 1)))" in src, (
        "vq_interp no longer computes a uniform ceil-width chunk")
    assert "lo = rank * chunk" in src
    assert "keep = (lo + np.arange(chunk)) < n_kept" in src, (
        "the tail mask is gone; without it the uniform chunk over-counts")
    assert "lo = rank * n_kept // nranks" not in src, (
        "the RAGGED rank slab is back in vq_interp")
    assert "if hi > lo:" not in src, (
        "the guard that made surplus ranks compile NOTHING is back")


def test_the_memo_key_no_longer_carries_the_rank_slab():
    """``lo`` is out of the key; ``chunk`` is in it.

    ``lo`` is ``rank * chunk`` and both are constant within a process, so
    within the only scope a host memo has it is a function of ``chunk`` —
    and keeping it would have put the rank slab back into a cache key for a
    lookup that could never benefit.
    """
    src = (REPO_ROOT / "src" / "bse" / "vq_interp.py").read_text()
    i = src.index("_MBZ_DQ_CACHE.get(key)")
    key_block = src[src.rindex("key = (", 0, i):i]
    assert "int(chunk)" in key_block, f"chunk is not in the key:\n{key_block}"
    assert "int(lo)" not in key_block, (
        f"the rank slab is back in the memo key:\n{key_block}")


# ---------------------------------------------------------------------------
# SIBLING 5 — the per-process FFI dials and the cross-rank fingerprint
# ---------------------------------------------------------------------------
def test_every_ffi_dial_is_in_the_cross_rank_fingerprint():
    """The two lists are one declaration in two places; they must agree.

    A dial that reaches ``ffi_dial_key`` and not the fingerprint is a rank
    that can emit a different HLO BODY from its peers with nothing noticing
    — the divergence arriving through the environment instead of through a
    shard offset.
    """
    import cache_key_lint as lint                             # noqa: E402
    dials = lint._string_tuple(                               # noqa: SLF001
        REPO_ROOT / "src" / "ffi" / "__init__.py", "FFI_DIAL_ENV")
    covered = lint._string_tuple(                             # noqa: SLF001
        REPO_ROOT / "src" / "common" / "jax_compile_cache.py",
        "RANK_FINGERPRINT_ENV")
    assert dials, "src/ffi/__init__.py declares no FFI_DIAL_ENV"
    assert dials <= covered, (
        f"dial(s) {sorted(dials - covered)} are read at kernel-factory time "
        f"but are not fingerprinted across ranks")


def test_the_declared_dials_are_the_ones_ffi_dial_key_actually_reads():
    """The declaration must not drift from the function it describes."""
    src = (REPO_ROOT / "src" / "ffi" / "__init__.py").read_text()
    body = src[src.index("def ffi_dial_key"):]
    for dial, gate in (("LORRAX_FFT_FFI_FUSED", "fused_fft_ffi_enabled"),
                       ("LORRAX_FFT_FFI", "fft_ffi_enabled"),
                       ("LORRAX_BANDS_GEMM_FFI", "gemm_ffi_enabled")):
        assert gate in body, f"{gate} is no longer read by ffi_dial_key"
        assert dial in src, f"{dial} is not declared in FFI_DIAL_ENV"


def test_the_fingerprint_folds_unset_and_empty_together():
    """Two ranks that differ only in whether the variable EXISTS are not
    divergent: ``Gate.mode`` treats "" as "take the default"."""
    src = (REPO_ROOT / "src" / "common" / "jax_compile_cache.py").read_text()
    block = src[src.index("for name in RANK_FINGERPRINT_ENV"):][:400]
    assert 'os.environ.get(name, "").strip().lower()' in block, (
        "the fingerprint no longer normalises unset/empty/case, so an "
        "unset dial and an empty one would read as a divergence")
