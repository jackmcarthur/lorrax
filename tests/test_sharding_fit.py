"""``common.sharding_fit`` — and proof that the thing it fixes is real.

The defect: ``bandstructure.bse_setup`` and ``bse.vq_interp`` shard the RAW
ISDF centroid count while ``bse.bse_io`` shards its own PADDED copy.  n_mu is a
k-means output, so on the MoS2 4x4 deck it is 785 = 5 x 157 — divisible by
neither 4 nor 8, and ``exciton_bands`` died at EVERY P>1 (job 7882476, cells
exb16s at P=16 and exb64s at P=64).

Two properties have to hold and each is worthless without its opposite:

  * when every axis divides, the fitter must return the SAME spec — otherwise
    it silently moves the HLO of every currently-working run;
  * when one does not, it must drop THAT axis and only that one.

and one fact has to be established rather than assumed: that the raw spec
really is refused by jax.  A fitter that "fixes" a non-problem would pass
every logic test above and tell you nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import sharding_fit as SF


class _StubMesh:
    """Just the ``.shape`` mapping ``legal_spec`` reads — so the spec algebra
    is testable at any device count, including one."""

    def __init__(self, **axes):
        self.shape = dict(axes)


# ---------------------------------------------------------------------------
# spec algebra
# ---------------------------------------------------------------------------
def test_a_spec_that_divides_is_returned_unchanged_and_identical():
    """Object identity, not equality: this is what guarantees a working run's
    sharding — and therefore its HLO — does not move."""
    m = _StubMesh(x=4, y=4)
    spec = P(None, "x", "y")
    out = SF.legal_spec(m, spec, (16, 640, 640), "unit")
    assert out is spec


def test_the_real_mos2_case_drops_exactly_the_mu_axis(capsys):
    """(nq, nb_fi, ns, n_mu) = (208, 4, 2, 785) under P(None,None,None,'y') on
    a 4x4 mesh — verbatim the shape and spec from the exb16s failure."""
    m = _StubMesh(x=4, y=4)
    out = SF.legal_spec(m, P(None, None, None, "y"), (208, 4, 2, 785),
                        "unit.psi_rmu_Y")
    assert tuple(out) == (None, None, None, None)
    msg = capsys.readouterr().out
    assert "785" in msg and "sharding_fit" in msg, \
        "the degradation must be announced, never silent"


def test_a_tuple_entry_uses_the_PRODUCT_of_its_mesh_axes():
    """``P(('x','y'), ...)`` flattens both axes onto one array axis, so a
    q-batch of 32 is legal at P=16 and illegal at P=64.  Getting this wrong in
    either direction is how the q-batched path breaks."""
    m16, m64 = _StubMesh(x=4, y=4), _StubMesh(x=8, y=8)
    spec = P(("x", "y"), None, None)
    keep = SF.legal_spec(m16, spec, (32, 608, 608), "unit.qbatch16")
    assert keep is spec, "32 % 16 == 0 must be left alone"
    drop = SF.legal_spec(m64, spec, (32, 608, 608), "unit.qbatch64")
    # PARTIAL retention: 64 does not divide 32 but 8 does, so the entry keeps
    # one mesh axis instead of replicating over both.  Full replication here
    # is an 8x memory regression on an axis that was fine — the same argument
    # ``test_only_the_offending_axis_is_dropped`` makes between entries.
    assert tuple(drop) == ("x", None, None), \
        "32 % 64 != 0 must degrade to the largest sub-product that divides"
    assert SF.shard_factor(m64, tuple(drop)[0]) == 8


def test_a_tuple_entry_replicates_fully_when_NO_sub_product_divides():
    """The sibling that keeps the test above honest: partial retention must be
    driven by divisibility, not applied unconditionally.  785 = 5 x 157 is
    divisible by neither 8 nor 64, so nothing can be kept."""
    m64 = _StubMesh(x=8, y=8)
    out = SF.legal_spec(m64, P(("x", "y"), None), (785, 9), "unit.nosub")
    assert tuple(out) == (None, None)


def test_partial_retention_keeps_one_axis_of_a_square_mesh():
    """Square-mesh ruling (decisions.md 2026-08-01): both axes have the same
    extent, so equal-length subsets have equal products and the fitter keeps
    the earliest order-preserving one that divides — deterministically."""
    m = _StubMesh(x=8, y=8)            # product 64
    out = SF.legal_spec(m, P(("x", "y"), None), (24, 3), "unit.sq")
    assert tuple(out) == ("x", None), "24 % 64 != 0 but 24 % 8 == 0 — keep 8"


# ---------------------------------------------------------------------------
# padded_extent — the durable half of the module
# ---------------------------------------------------------------------------
def test_padded_extent_rounds_up_to_the_same_divisor_the_fitter_judges_by():
    """A call site that pads against one divisor and is judged against another
    is worse than not padding at all, so both must read ``shard_factor``."""
    m64 = _StubMesh(x=8, y=8)
    assert SF.padded_extent(m64, ("x", "y"), 32) == 64
    assert SF.padded_extent(m64, "y", 785) == 792
    assert SF.padded_extent(m64, None, 785) == 785


def test_padded_extent_is_a_no_op_when_the_extent_already_divides():
    m16 = _StubMesh(x=4, y=4)
    assert SF.padded_extent(m16, ("x", "y"), 32) == 32


def test_the_padded_q_batch_is_exactly_what_the_fitter_stops_complaining_about():
    """THE point of the pair: pad with ``padded_extent`` and ``legal_spec``
    returns the spec untouched — the q-batch stops being degraded at all.

    The negative leg (raw 32) is what the exb wall measured; without it this
    test would pass for a fitter that never degrades anything."""
    m64 = _StubMesh(x=8, y=8)
    spec = P(("x", "y"), None, None)
    raw = SF.legal_spec(m64, spec, (32, 608, 608), "unit.pairA")
    assert raw is not spec, "the un-padded batch MUST still degrade"
    bs = SF.padded_extent(m64, ("x", "y"), 32)
    padded = SF.legal_spec(m64, spec, (bs, 608, 608), "unit.pairB")
    assert padded is spec, "the padded batch must need no fitting at all"


def test_the_announcement_quantifies_the_cost_in_GiB(capsys):
    """A generic 'memory rises by the dropped factor' reads the same at 150 MB
    and at 13.4 GB.  The converged 12x12 / n_mu=2412 C_q is the second, and
    the log has to say so."""
    m64 = _StubMesh(x=8, y=8)
    SF.legal_spec(m64, P(None, "x", "y"), (144, 2412, 2412), "unit.C_q")
    msg = capsys.readouterr().out
    # 144 * 2412^2 * 16 B = 13.404 GB = 12.484 GiB replicated; /64 = 0.195 GiB.
    assert "12.484" in msg, f"per-device GiB not reported: {msg!r}"
    assert "0.195" in msg, \
        f"the intended (sharded) size not reported: {msg!r}"
    assert "x64" in msg, f"the replication factor not reported: {msg!r}"


def test_only_the_offending_axis_is_dropped():
    """A mixed case: 'x' divides, 'y' does not.  Dropping both would be a
    silent 8x memory regression on an axis that was fine."""
    m = _StubMesh(x=4, y=4)
    out = SF.legal_spec(m, P("x", "y"), (640, 785), "unit.mixed")
    assert tuple(out) == ("x", None)


def test_shard_factor_covers_none_str_and_tuple():
    m = _StubMesh(x=8, y=8)
    assert SF.shard_factor(m, None) == 1
    assert SF.shard_factor(m, "x") == 8
    assert SF.shard_factor(m, ("x", "y")) == 64


def test_the_announcement_fires_once_per_site_not_once_per_call(capsys):
    """A per-call print inside a q-chunk loop would bury the log; a per-site
    one is the useful granularity."""
    m = _StubMesh(x=4, y=4)
    SF.legal_spec(m, P("y", None), (785, 9), "unit.once")
    first = capsys.readouterr().out
    SF.legal_spec(m, P("y", None), (785, 9), "unit.once")
    second = capsys.readouterr().out
    assert "sharding_fit" in first
    assert second == "", "repeat announcements for the same site"


# ---------------------------------------------------------------------------
# the fact the whole module rests on: jax really does refuse
# ---------------------------------------------------------------------------
# Was a ``skipif(jax.device_count() < 4)``, which skipped in every suite run
# regardless of the node — tests/conftest.py pins each test process to one
# GPU, so ``device_count()`` is 1 by construction.  The marker states the
# requirement and the conftest supplies it; see the mesh section there.
needs4 = pytest.mark.mesh(4)


@needs4
def test_jax_refuses_the_raw_spec_and_accepts_the_fitted_one():
    """THE CONTROL.  Without this leg every test above could be describing a
    problem that does not exist.

    Uses the smallest mesh that reproduces it: a 2x2 mesh and an extent of
    785, the MoS2 4x4 deck's centroid count.
    """
    devs = jax.devices()[:4]
    mesh = Mesh(np.asarray(devs).reshape(2, 2), axis_names=("x", "y"))
    shape = (3, 785)

    raw = NamedSharding(mesh, P(None, "y"))
    with pytest.raises(Exception) as ei:
        raw.shard_shape(shape)
    assert "785" in str(ei.value) or "divis" in str(ei.value).lower(), \
        f"refused for an unexpected reason: {ei.value}"

    fitted = SF.fit_sharding(mesh, P(None, "y"), shape, "unit.control")
    assert fitted.shard_shape(shape) == shape, \
        "the fitted sharding must be usable for the same shape"


@needs4
def test_a_divisible_extent_needs_no_fitting_on_a_real_mesh():
    """Sibling control: the same call on an extent that DOES divide must keep
    the sharding — so the test above is measuring divisibility, not just
    'fitting always replicates'."""
    devs = jax.devices()[:4]
    mesh = Mesh(np.asarray(devs).reshape(2, 2), axis_names=("x", "y"))
    shape = (3, 784)                       # 784 = 2*392
    spec = P(None, "y")
    fitted = SF.fit_sharding(mesh, spec, shape, "unit.control_ok")
    assert fitted.spec is spec
    assert fitted.shard_shape(shape) == (3, 392)


@needs4
def test_jax_refuses_a_TUPLE_spec_whose_product_does_not_divide():
    """THE CONTROL for the q-batch half of the module.

    ``P(('x','y'), None)`` over a 2x2 mesh needs the extent divisible by 4.
    A 6-row batch is not — and 6 IS divisible by 2, so this also establishes
    the premise of partial retention: the FULL product is refused while the
    sub-product the fitter keeps is accepted.

    Without this leg, ``test_a_tuple_entry_uses_the_PRODUCT_of_its_mesh_axes``
    is only asserting my own arithmetic against itself.  It is also the
    evidence behind the filed request against ``htransform.h_transform``,
    whose ``batch_mat_shard`` / ``batch_eig_shard`` are RAW NamedShardings on
    a fixed 32-wide batch: at a device count that does not divide 32 that path
    does not degrade, it raises.
    """
    devs = jax.devices()[:4]
    mesh = Mesh(np.asarray(devs).reshape(2, 2), axis_names=("x", "y"))
    shape = (6, 5)

    raw = NamedSharding(mesh, P(("x", "y"), None))
    with pytest.raises(Exception) as ei:
        raw.shard_shape(shape)
    assert "6" in str(ei.value) or "divis" in str(ei.value).lower(), \
        f"refused for an unexpected reason: {ei.value}"

    fitted = SF.fit_sharding(mesh, P(("x", "y"), None), shape, "unit.tupctl")
    assert tuple(fitted.spec) == ("x", None), \
        f"expected partial retention of one axis, got {fitted.spec}"
    assert fitted.shard_shape(shape) == (3, 5), \
        "the kept axis must actually split the extent 2 ways"

    # And the padded route: pad to 8 and the FULL product is legal again.
    bs = SF.padded_extent(mesh, ("x", "y"), 6)
    assert bs == 8
    ok = NamedSharding(mesh, P(("x", "y"), None))
    assert ok.shard_shape((bs, 5)) == (2, 5)


@needs4
def test_a_real_array_actually_lands_under_the_fitted_sharding():
    """End to end: device_put a 785-long axis that cannot be split, and read
    the value back.  Placement changed; the data did not."""
    devs = jax.devices()[:4]
    mesh = Mesh(np.asarray(devs).reshape(2, 2), axis_names=("x", "y"))
    x = np.arange(3 * 785, dtype=np.float64).reshape(3, 785)
    sh = SF.fit_sharding(mesh, P(None, "y"), x.shape, "unit.e2e")
    y = jax.device_put(x, sh)
    assert np.array_equal(np.asarray(jax.device_get(y)), x)
