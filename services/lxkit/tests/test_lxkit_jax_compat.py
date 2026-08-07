"""``lxkit.jax_compat`` — the jax-version boundary.

Two tiers, on purpose:

* :func:`select_mode` is PURE, so the whole decision table is testable on any
  jax and on none — including the rows the running container cannot exhibit
  (the other spelling, and the refusal).  The compatibility path nobody can
  see is what produced the defect this module replaces.
* :func:`mark_varying` is exercised against whatever jax is actually here,
  WITH its red twin: the same loop without the mark must be rejected, or the
  mark is decoration.

Local jax is 0.9.1 (``lax.pcast`` row); the Perlmutter container is 0.7.0
(``lax.pvary`` row).  Only one row can execute per interpreter — which is
exactly why the table above it is pure.
"""

from __future__ import annotations

import pytest

from lxkit.jax_compat import (
    VMA_TRACKING_SINCE, VmaSupportError, mark_varying, select_mode, vma_mode,
)


# ---------------------------------------------------------------------------
# The decision table — pure, no jax
# ---------------------------------------------------------------------------

def test_the_measured_table_from_the_module_docstring():
    """jax   VMA tracked?  lax.pvary  lax.pcast   (Perlmutter A100,
    2026-08-06, JID 56405158 / 56405696)"""
    assert select_mode(False, False, (0, 5, 3)) == "identity"
    assert select_mode(False, True, (0, 7, 0)) == "lax.pvary"
    assert select_mode(False, True, (0, 7, 2)) == "lax.pvary"
    assert select_mode(True, True, (0, 9, 0)) == "lax.pcast"


def test_the_newest_spelling_wins_when_both_exist():
    """0.9 has both; pvary is deprecated there."""
    assert select_mode(True, True, (0, 9, 1)) == "lax.pcast"


def test_tracking_starts_at_0_7_0_not_0_9():
    """A guard that assumes 'only 0.9 needs this' is wrong over the whole
    0.7-0.8 range."""
    assert VMA_TRACKING_SINCE == (0, 7, 0)
    assert select_mode(False, False, (0, 6, 9)) == "identity"
    with pytest.raises(VmaSupportError):
        select_mode(False, False, (0, 7, 0))


def test_a_tracking_jax_with_neither_spelling_REFUSES():
    """`try: lax.pcast / except AttributeError: identity` installs a NO-OP on
    exactly the versions that enforce the rule — the defect that was live in
    common/cholesky_2d.py.  Refusing is the whole point of this module."""
    with pytest.raises(VmaSupportError) as ei:
        select_mode(False, False, (0, 8, 0))
    msg = str(ei.value)
    assert "0.8.0" in msg and "0.7.0" in msg
    assert "lax.pcast" in msg and "lax.pvary" in msg
    assert "jax_compat.py" in msg, "the refusal must name the fix"


def test_the_identity_is_reachable_only_below_the_tracking_window():
    """Where the identity is not merely safe but the only behaviour that
    exists."""
    assert select_mode(False, False, (0, 4, 0)) == "identity"
    for ver in [(0, 7, 0), (0, 9, 0), (1, 0, 0)]:
        with pytest.raises(VmaSupportError):
            select_mode(False, False, ver)


# ---------------------------------------------------------------------------
# The live jax
# ---------------------------------------------------------------------------

def test_this_interpreter_resolved_to_the_spelling_its_jax_has():
    jax = pytest.importorskip("jax")
    from jax import lax
    expected = select_mode(hasattr(lax, "pcast"), hasattr(lax, "pvary"),
                           tuple(jax.version.__version_info__))
    assert vma_mode().startswith(expected)


def _scan_over_a_sharded_carry(mark: bool):
    """A shard_map whose scan carry starts as a plain ``jnp.zeros``.

    This is the shape the module docstring's TypeError comes from: the
    accumulator has an EMPTY varying-manual-axes set and the first
    ``c + a`` makes the output varying.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((1, 1), ("x", "y"))
    spec = P("x", "y")

    def body(a):
        init = jnp.zeros(a.shape, a.dtype)
        if mark:
            init = mark_varying(init, ("x", "y"))
        out, _ = jax.lax.scan(lambda c, _: (c + a, None), init, jnp.arange(3))
        return out

    a = jax.device_put(jnp.ones((4, 4)), NamedSharding(mesh, spec))
    fn = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(spec,),
                               out_specs=spec, check_vma=True))
    return float(fn(a).sum())


def test_mark_varying_lets_a_zeros_carry_through_a_shard_map_scan():
    pytest.importorskip("jax")
    assert _scan_over_a_sharded_carry(mark=True) == pytest.approx(3 * 16)


def test_the_same_loop_WITHOUT_the_mark_is_rejected():
    """The red twin.  On a jax that tracks VMA the unmarked carry must fail
    at trace time with 'the varying manual axes do not match' — if it did
    not, mark_varying would be decoration and this suite would be asserting
    nothing."""
    pytest.importorskip("jax")
    if vma_mode().startswith("identity"):
        pytest.skip("this jax does not track VMA, so nothing can reject an "
                    "unmarked carry — the identity is the only behaviour")
    with pytest.raises(TypeError) as ei:
        _scan_over_a_sharded_carry(mark=False)
    assert "varying manual axes" in str(ei.value)


def test_mark_varying_announces_its_spelling_once(capsys):
    pytest.importorskip("jax")
    import lxkit.jax_compat as jc
    jc._ANNOUNCED = False
    _scan_over_a_sharded_carry(mark=True)
    first = capsys.readouterr().out
    _scan_over_a_sharded_carry(mark=True)
    second = capsys.readouterr().out
    assert "[vma] marking loop carries via" in first and vma_mode() in first
    assert "[vma]" not in second
