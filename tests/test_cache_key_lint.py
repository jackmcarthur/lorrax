"""THE LINT'S OWN GATE — both directions, on real code.

``tests/cache_key_lint.py`` is the cheap half of the JAX cache contract: it
finds class-B key divergence (a rank-dependent value reaching a compile-time
surface) by reading the source, before the four-GPU leg that would otherwise
be the only thing able to see it.

A lint is only worth its false-positive budget if BOTH directions are gated,
so this file asserts:

    NEGATIVE   ``src`` + ``services`` are clean, except for the sites the
               red list names — and every red-list row is DATED, carries a
               reason, and still matches something.
    POSITIVE   the five siblings of FIX_multislice_cachekey.md §6.1 are
               flagged, taken against their FOUND state at ``aca88841``
               (``tests/cache_lint_corpus/``), because the live files are
               fixed on this branch and the positive direction has nowhere
               else to live.
    NEGATIVE   the CANONICALIZED ``_multi_slice`` — the fix all five were
               measured against — is not flagged.

The corpus is text, not fixtures generated at test time.  A lint tested only
against snippets someone wrote to make it pass tests the snippets, not the
lint; these are the actual functions, extracted whole, with their provenance
in the header.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_key_lint as lint                               # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(__file__).resolve().parent / "cache_lint_corpus"

#: sibling -> the rules its FOUND state must trigger.  Named per sibling so
#: a change that keeps the count but moves which rule fires is still a
#: failure: the rule IS the diagnosis, and the wrong diagnosis sends the
#: next reader to the wrong fix.
EXPECTED = {
    "sibling1_vq_interp.txt": {"rank-shape", "rank-cache-key", "rank-branch"},
    "sibling2_kin_ion_io.txt": {"rank-branch"},
    "sibling3_charge_density.txt": {"rank-branch"},
    "sibling4_collectives.txt": {"rank-branch"},
}


@pytest.fixture(scope="module")
def seed():
    """The whole-tree index of functions that reach XLA."""
    return lint.tree_compiling_index(REPO_ROOT)


# ---------------------------------------------------------------------------
# POSITIVE — it flags the five siblings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_lint_flags_the_found_state_of_each_sibling(name, seed):
    path = CORPUS / name
    assert path.is_file(), f"corpus snapshot {name} is missing"
    found = lint.scan_source(path.read_text(), name, seed)
    rules = {f.rule for f in found}
    assert rules >= EXPECTED[name], (
        f"{name}: the lint found {sorted(rules) or 'NOTHING'} but this "
        f"sibling's found state must trigger {sorted(EXPECTED[name])}.\n"
        + lint.format_report(found))


def test_the_env_dial_rule_flags_an_undeclared_dial(tmp_path):
    """SIBLING 5, both directions, on a synthetic two-file tree.

    The ``env-dial`` rule is a relation BETWEEN two files — the dial
    declaration and the cross-rank fingerprint — so it cannot be expressed
    as a corpus snippet the way the other four can.
    """
    (tmp_path / "src" / "ffi").mkdir(parents=True)
    (tmp_path / "src" / "common").mkdir(parents=True)
    ffi = tmp_path / "src" / "ffi" / "__init__.py"
    fp = tmp_path / "src" / "common" / "jax_compile_cache.py"

    # (a) THE FOUND STATE: no declaration at all.
    ffi.write_text((CORPUS / "sibling5_ffi_dial_key.txt").read_text())
    fp.write_text("RANK_FINGERPRINT_ENV = ()\n")
    found = lint.env_dial_findings(tmp_path)
    assert found and found[0].rule == "env-dial", (
        "a tree whose FFI dials are not declared anywhere came back clean")
    assert "FFI_DIAL_ENV" in str(found[0])

    # (b) DECLARED BUT NOT FINGERPRINTED — the real defect.
    ffi.write_text('FFI_DIAL_ENV = ("LORRAX_FFT_FFI", "LORRAX_FFT_FFI_FUSED")\n')
    found = lint.env_dial_findings(tmp_path)
    assert len(found) == 1 and "LORRAX_FFT_FFI_FUSED" in found[0].symbol, (
        f"a declared-but-unfingerprinted dial was not flagged: {found}")
    assert "HLO BODY" in str(found[0]), (
        "the message must say WHY this is worse than a key divergence — it "
        "changes the emitted module, not merely its key")

    # (c) FIXED.
    fp.write_text('RANK_FINGERPRINT_ENV = ("LORRAX_FFT_FFI", '
                  '"LORRAX_FFT_FFI_FUSED")\n')
    assert lint.env_dial_findings(tmp_path) == [], (
        "a dial present in BOTH lists is still being flagged")


def test_every_finding_carries_its_recipe():
    """A lint that says only NO gets switched off."""
    for rule, recipe in lint.RECIPE.items():
        assert len(recipe) > 80, f"{rule} has no usable recipe"
    f = lint.Finding("a.py", 1, "rank-shape", "lo", "x")
    assert lint.RECIPE["rank-shape"] in str(f)
    assert "a.py:1" in str(f)


# ---------------------------------------------------------------------------
# NEGATIVE — the fixed tree is clean, and the red list is honest
# ---------------------------------------------------------------------------
def test_the_canonicalized_multi_slice_is_not_flagged(seed):
    """THE FIX ITSELF MUST PASS.

    ``common/jax_compile_cache.py`` holds the canonicalization every one of
    the five siblings was measured against (sizes static, offsets dynamic).
    A lint that flags the sanctioned fix is a lint that tells people to undo
    it.
    """
    src = REPO_ROOT / "src" / "common" / "jax_compile_cache.py"
    found = lint.scan_source(src.read_text(), "src/common/jax_compile_cache.py",
                             seed)
    assert found == [], (
        "the canonical fix is being flagged by its own lint:\n"
        + lint.format_report(found))


def test_the_tree_is_clean_apart_from_the_red_list():
    """THE GATE.  Every unallowed finding fails, by file:line, with a recipe."""
    findings = lint.scan_tree(REPO_ROOT)
    unallowed, allowed = lint.partition(findings)
    assert not unallowed, (
        f"{len(unallowed)} class-B site(s) with no red-list row.  Either "
        f"canonicalize them or add a dated AllowRow saying why the fix is "
        f"not mechanical:\n\n" + lint.format_report(unallowed))


def test_no_red_list_row_is_stale():
    """A row that matches nothing is a debt already paid, still on the books.

    Left alone, it is where the next real regression hides.
    """
    stale = lint.stale_allow_rows(lint.scan_tree(REPO_ROOT))
    assert not stale, (
        "red-list row(s) no longer match anything — delete them:\n"
        + "\n".join(f"  {r.path} [{r.rule}] added {r.dated}" for r in stale))


def test_every_red_list_row_is_dated_and_has_a_live_ledger():
    """A red-listed site is a DEBT WITH A NAME ON IT, not a silence."""
    import re
    for row in lint.ALLOW:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.dated), row
        assert len(row.reason) > 120, (
            f"{row.path} [{row.rule}]: a red-list reason has to say why "
            f"forcing the fix would be worse than the defect")
        assert (REPO_ROOT / row.ledger).is_file(), (
            f"{row.path} [{row.rule}] points at ledger {row.ledger}, which "
            f"does not exist")
        assert (REPO_ROOT / row.ledger).read_text().find(row.path) >= 0, (
            f"the ledger {row.ledger} never mentions {row.path}")


def test_the_lint_covers_the_source_roots_it_claims_to():
    """A scan that silently stopped seeing a root is a green run of nothing."""
    seen = {rel for rel, _ in lint.iter_sources(REPO_ROOT)}
    assert len(seen) > 200, f"only {len(seen)} files scanned — the walk broke"
    assert any(r.startswith("src/gw/") for r in seen)
    assert any(r.startswith("src/bse/") for r in seen)
    assert any(r.startswith("services/") and "/src/" in r for r in seen)
    # A service's OWN tests are where red twins live deliberately.
    assert not any("/tests/" in f"/{r}" for r in seen)


def test_the_taint_is_two_level(seed):
    """A bare rank id is not a work partition, and the distinction is load
    bearing: firing on it gave 84 findings on this tree against 8."""
    rank0_only = (
        "import jax\n"
        "def f(write):\n"
        "    if jax.process_index() == 0:\n"
        "        write(jnp.zeros((4, 4)))\n")
    assert lint.scan_source(rank0_only, "t.py", seed) == [], (
        "the rank-0-only writer idiom is being flagged")

    partition = (
        "import jax\n"
        "def f(n):\n"
        "    rank = jax.process_index()\n"
        "    world = jax.process_count()\n"
        "    lo = rank * n // world\n"
        "    hi = (rank + 1) * n // world\n"
        "    return jnp.arange(lo, hi)\n")
    found = lint.scan_source(partition, "t.py", seed)
    assert any(f.rule == "rank-shape" for f in found), (
        f"a textbook rank slab reaching jnp.arange was not flagged: {found}")


def test_a_rank_static_jit_argument_is_flagged(seed):
    """The canonical class-B shape: ``jit__multi_slice`` in six lines."""
    src = (
        "import jax\n"
        "from functools import partial\n"
        "@partial(jax.jit, static_argnums=(1,))\n"
        "def _slice_it(x, offset):\n"
        "    return x[offset:]\n"
        "def go(x, n):\n"
        "    rank = jax.process_index()\n"
        "    world = jax.process_count()\n"
        "    lo = rank * n // world\n"
        "    return _slice_it(x, lo)\n")
    found = lint.scan_source(src, "t.py", seed)
    assert any(f.rule == "rank-static-arg" for f in found), (
        f"a rank-dependent STATIC jit argument was not flagged: {found}")
