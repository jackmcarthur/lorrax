"""Which ``shard_map`` this tree calls, and with which kwarg.

``shard_map`` moved and changed shape.  ``jax.shard_map`` does not exist on
jax 0.5.3 (the production container); ``jax.experimental.shard_map`` is
deprecated from jax 0.8.0 and does not support nesting.  Both exist on 0.7.0,
the migration target, and they are DIFFERENT OBJECTS with different kwargs.

``common.shard_map`` makes that decision once.  This file is the ratchet that
keeps it made in one place, in three parts:

1. **The decision table, by construction.**  ``select_mode`` is pure, so every
   row --- including the refusal, which no container can exhibit --- is
   checkable on a login node with no GPU and no jax of the right version.
2. **No module says ``check_rep=``.**  The legacy kwarg, on the legacy symbol.
3. **No module imports a jax ``shard_map`` directly.**  The whole point.

The equivalence that licenses part 2 is not an inference.  Read out of the
running 0.7.0 container, the experimental wrapper's entire body is::

    axis_names = frozenset(mesh.axis_names) - auto
    return jshmap._shard_map(
        f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
        check_vma=check_rep, axis_names=axis_names, _skip_mesh_check=True)

``check_vma=check_rep``: the same value, handed to the same internal
``_shard_map`` that ``jax.shard_map`` calls.  There is no code path on which
the two can differ.  Confirmed behaviourally as well (JID 56405696): both
``=True`` arms raise the same replication ``ValueError``, both ``=False`` arms
return bit-identical bytes (sha256 ``32064687fdcfa677``, the same on 0.5.3),
and for an unmarked ``fori_loop`` carry both defaults raise the same VMA
``TypeError`` while both ``=False`` pass.

That same line is also upstream's own statement of the ``auto`` ->
``axis_names`` polarity: ``auto`` named the AUTOMATIC axes, ``axis_names``
names the MANUAL ones.  It is an inversion --- and an AST census of all 94
construction sites found ZERO passing ``auto=``, so every site inherits
defaults that coincide.  ``common.shard_map`` therefore exposes neither kwarg.
"""
from __future__ import annotations

import ast
import os

import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


def _iter_src():
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "build", "build_host")]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                try:
                    yield os.path.relpath(p, SRC), open(p, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue


# ===========================================================================
# 1.  the version decision table, by construction
# ===========================================================================

def test_select_mode_table():
    """Both real rows plus the refusal, which no container can exhibit.

        jax     jax.shard_map   jax.experimental.shard_map
        0.5.3   ABSENT          present
        0.7.0   present         present (deprecated at 0.8.0)
    """
    from common.shard_map import select_mode

    # 0.5.3 -- only the legacy symbol exists, so it is the only choice.
    assert select_mode(False, True, (0, 5, 3)) == "jax.experimental.shard_map"
    # 0.7.0 onward -- both exist; the modern one wins, which is the whole
    # point of the migration.
    assert select_mode(True, True, (0, 7, 0)) == "jax.shard_map"
    assert select_mode(True, True, (0, 8, 1)) == "jax.shard_map"
    # ...and once the deprecated one is finally deleted, nothing changes.
    assert select_mode(True, False, (0, 10, 0)) == "jax.shard_map"


def test_select_mode_refuses_rather_than_degrading():
    """A jax with NEITHER symbol must refuse, and name the file to fix.

    There is no meaningful fallback: every distributed kernel in ``src/`` is
    written in ``shard_map``.  Silently degrading is what the previous
    per-file ``try/except ImportError`` shims did in spirit --- they picked a
    symbol without translating its kwarg, which is a no-op that fails later
    and elsewhere.
    """
    from common.shard_map import select_mode, ShardMapSupportError

    with pytest.raises(ShardMapSupportError) as e:
        select_mode(False, False, (0, 99, 0))
    msg = str(e.value)
    assert "src/common/shard_map.py" in msg, msg
    assert "0.99.0" in msg, msg


def test_resolved_mode_matches_the_running_jax():
    """The module body agrees with its own pure function on THIS jax."""
    import jax
    from common.shard_map import select_mode, shard_map_mode

    expected = select_mode(hasattr(jax, "shard_map"), True,
                           tuple(jax.version.__version_info__))
    assert shard_map_mode().startswith(expected)


# ===========================================================================
# 2 and 3.  the ratchets
# ===========================================================================

def test_no_module_says_check_rep_any_more():
    """``check_rep=`` is the LEGACY kwarg and no longer belongs anywhere in src/.

    Measured on jax 0.7.0, in-container, reading the experimental wrapper's own
    body::

        axis_names = frozenset(mesh.axis_names) - auto
        return jshmap._shard_map(
            f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
            check_vma=check_rep, axis_names=axis_names, _skip_mesh_check=True)

    ``check_vma=check_rep`` --- the same value, handed to the same internal
    ``_shard_map`` that ``jax.shard_map`` calls.  So the rename is exact, and
    since 2026-08-06 every ``shard_map`` in this tree comes from
    ``common.shard_map``, whose kwarg is the modern ``check_vma``.
    ``common.shard_map`` translates once, for the 0.5.3 container that has no
    ``jax.shard_map`` at all.

    This is the ratchet that keeps the translation in ONE file.
    ``common/shard_map.py`` itself is exempt: its legacy branch IS the
    translation, and converting it is a break that is invisible on 0.7.0 (the
    branch is dead there) and total on 0.5.3.  That is not a hypothetical --- a
    mechanical sweep did exactly that on its first run.
    """
    offenders = []
    for rel, src in _iter_src():
        if rel.replace(os.sep, "/") == "common/shard_map.py":
            continue
        if "check_rep" not in src:
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        # AST, not grep: several modules discuss the kwarg in prose.
        for n in ast.walk(tree):
            if isinstance(n, ast.keyword) and n.arg == "check_rep":
                offenders.append("%s:%d" % (rel, n.lineno))
    assert not offenders, (
        "these sites pass the legacy check_rep= kwarg; the tree's shard_map "
        "comes from common.shard_map and takes check_vma=:\n  "
        + "\n  ".join(offenders))


def test_every_shard_map_comes_from_common_shard_map():
    """One module decides which ``shard_map`` symbol this jax wants.

    ``jax.shard_map`` does not exist on 0.5.3 and ``jax.experimental.shard_map``
    is deprecated from 0.8.0 (upstream's own words), so neither spelling can be
    written directly in a tree that has to run on both.  The failure this
    prevents is not hypothetical: ``bse/bse_ring_comm.py`` and
    ``bse/bse_stack_matvec.py`` each carried a private
    ``try: from jax import shard_map / except ImportError`` shim, in two
    copies, and NEITHER translated the check kwarg --- so the first site to add
    ``check_vma=False`` under them would have raised ``TypeError`` on 0.5.3
    with nothing to say why.
    """
    offenders = []
    for rel, src in _iter_src():
        if rel.replace(os.sep, "/") == "common/shard_map.py":
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.module and n.module.startswith("jax.experimental.shard_map"):
                    offenders.append("%s:%d  from %s" % (rel, n.lineno, n.module))
                elif n.module in ("jax", "jax.experimental"):
                    for a in n.names:
                        if a.name == "shard_map":
                            offenders.append("%s:%d  from %s import shard_map"
                                             % (rel, n.lineno, n.module))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("jax.experimental.shard_map"):
                        offenders.append("%s:%d  import %s"
                                         % (rel, n.lineno, a.name))
    assert not offenders, (
        "these modules reach for a jax shard_map symbol directly instead of "
        "importing it from common.shard_map:\n  " + "\n  ".join(offenders))
