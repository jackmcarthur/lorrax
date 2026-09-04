"""Gate for ``common.staged_reshard.face_to_batch_reshard``.

Gate order follows the staged-reshard doctrine §4: unit gates first, and
EVERY instrument is shown failing before it is trusted (wk_REL README §5.1
— six checks in the 2026-07-30 session were void, four of them cheerfully
green).

The instruments here, each with its red twin:

1. **value parity** — the staged chain must return the input, reassembled.
   Red twin: ``test_value_parity_detects_a_wrong_block_map`` stages the two
   exchanges in the WRONG order and requires the same comparison to go red.
   Without it, "the arrays are equal" would only prove that ``device_get``
   and the ``out_specs`` annotation agree with each other.
2. **the compiled module** — the staged module issues exactly two
   ``all-to-all`` and no other collective, no buffer in it exceeds ONE
   shard, and its compile-time temp allocation is below the size of the
   whole array.  Red twin *and* FALSE arm:
   ``test_hlo_red_twin_the_unstaged_chain_replicates`` puts the unstaged
   single-constraint chain through the SAME assertion helper and requires
   it to raise.
3. **the production instrument** — the same three facts taken in a FRESH
   PROCESS at the deck geometry the module docstring's own diagnostic was
   written from (``c128[64,84,84]``), because a cold compile in a job is
   the shape production runs in.  Red twin: the unstaged chain in the same
   subprocess must show the replication.
4. **refusals** — divisibility and inverted-mesh, each raising before any
   collective with the caller's fix named.

**THE GPU FORM IS THE PRIMARY ONE, AND THAT IS A REPAIR** (2026-08-10).
Instruments 2 and 3 were written on XLA:CPU and were never true on CUDA:

* Instrument 2 counted the SYNCHRONOUS spelling ``all-to-all(``.  XLA:GPU
  rewrites every collective into an async ``-start``/``-done`` pair, so the
  counter read **0 all-to-all at num_partitions=4 on CUDA** while both
  exchanges were present and correct — ``all-to-all-start`` on
  ``replica_groups={{0,2},{1,3}}`` then ``{{0,1},{2,3}}``, one shard of
  payload each.  The pin was blind to its own subject on the platform
  production runs on.
* Instrument 3 grepped a subprocess's stderr for the compiler's
  ``Involuntary full rematerialization`` warning.  **That line is dead on
  this stack.**  It appears zero times on CUDA *and* on the emulated CPU
  mesh, under Shardy *and* under the legacy GSPMD partitioner
  (``JAX_USE_SHARDY_PARTITIONER=false``), at 30 KB, 7.2 MB and 67 MB of
  replicated payload — and it is not a rewording, since ``strings`` still
  finds the literal in the shipped ``libjax_common.so`` and
  ``xla_cuda_plugin.so``.  Meanwhile the HAZARD it announced is fully
  present and fully measurable: the unstaged chain at the deck geometry
  names a ``c128[64,84,84]`` full-batch buffer on one device and its temp
  allocation is **6.0x one shard** (10,838,532 B against 1,806,336 B).  So
  the grep is retired as an instrument and the accounting that the grep was
  a proxy FOR is asserted directly.  A gate that cannot go red is worse
  than no gate; the evidence is
  ``/pscratch/sd/j/jackm/reshard_instr_0810/`` (``_truth3/``, leg3.log).

Both instruments are now written on the collective's ISSUE SITE and its
PAYLOAD rather than on a backend's spelling, so the same cells are true on
a real GPU mesh and on an emulated CPU one — the CPU run is the secondary
parameterization, not a substitute (AGENT_PREAMBLE, the four-GPU rule).

Run on the real 4-GPU mesh (primary), bypassing the one-GPU pin in
``tests/conftest.py`` until the ``mesh(n)`` marker's conftest lands::

    JAX_ENABLE_X64=1 python -m pytest tests/test_staged_reshard.py -q \
        --noconftest -p no:cacheprovider

Run on 4 emulated devices (secondary)::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_ENABLE_X64=1 \
        python -m pytest tests/test_staged_reshard.py -q --noconftest
"""
import os
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.staged_reshard import (
    band_to_product_r_reshard,
    face_to_batch_reshard,
    face_to_batch_reshard_supported,
    shard_local_slice_pad,
    shard_local_update,
)

# FOUR DEVICES.  Was a ``skipif(jax.device_count() < 4)``, which skipped in
# every suite run regardless of the node — tests/conftest.py pins each test
# process to one GPU, so ``device_count()`` is 1 by construction.  The marker
# states the requirement and the conftest supplies it; see the mesh section
# there.
pytestmark = pytest.mark.mesh(4)


# (B, M, N) chosen so every extent divides but none is equal to another —
# a transposed axis or a swapped stage would change the shape, not merely
# the values, and be caught twice.
B, M, N = 8, 12, 20
FACE = P(None, 'x', 'y')
BATCH = P(('x', 'y'), None, None)

# Rank-4 wavefunction carrier for the product-band -> product-r route.
# Every dimension is index-visible, both distributed extents divide the 2x2
# mesh, and no two nontrivial axes have the same size.
RK, RB, RS, RR = 3, 8, 2, 20
PRODUCT_BAND = P(None, ('x', 'y'), None, None)
PRODUCT_R = P(None, None, None, ('y', 'x'))


def _mesh(px=2, py=2, names=('x', 'y')):
    devs = np.array(jax.devices()[:px * py]).reshape(px, py)
    return Mesh(devs, names)


def _probe(b=B, m=M, n=N):
    """Values that encode their own (batch, row, col) index."""
    idx = np.arange(b * m * n, dtype=np.float64).reshape(b, m, n)
    return jnp.asarray(idx + 1j * (idx * 0.5 + 1.0))


def _product_probe():
    """Rank-4 values encoding their exact global ``(k,b,s,r)`` order."""
    idx = np.arange(RK * RB * RS * RR, dtype=np.float64).reshape(
        RK, RB, RS, RR)
    return jnp.asarray(idx + 1j * (idx * 0.25 + 3.0))


def test_shard_local_slice_pad_never_repartitions_a_smaller_global_face():
    from common.collectives import device_put_process_local
    mesh = _mesh()
    spec = P(None, 'x', None)
    sh = NamedSharding(mesh, spec)
    rep = NamedSharding(mesh, P())
    src_np = np.arange(2 * 12 * 3, dtype=np.float64).reshape(2, 12, 3)
    src = jax.device_put(src_np, sh)
    take = shard_local_slice_pad(
        mesh, spec=spec, axis=1, mesh_axis='x', local_size=2)

    start = device_put_process_local(np.int32(5), rep)
    got = np.asarray(take(src, start))
    expected = np.zeros((2, 4, 3), dtype=np.float64)
    expected[:, 0, :] = src_np[:, 5, :]
    expected[:, 2, :] = src_np[:, 11, :]
    np.testing.assert_array_equal(got, expected)

    compiled = take.lower(
        jax.ShapeDtypeStruct(src.shape, src.dtype, sharding=sh),
        jax.ShapeDtypeStruct((), jnp.int32, sharding=rep),
    ).compile()
    hlo = compiled.as_text().lower()
    assert 'all-gather' not in hlo and 'all-to-all' not in hlo


def test_shard_local_update_writes_tail_in_each_owned_face_without_clamp():
    from common.collectives import device_put_process_local
    mesh = _mesh()
    spec = P('x', 'y')
    sh = NamedSharding(mesh, spec)
    rep = NamedSharding(mesh, P())
    dst = jax.device_put(np.zeros((6, 6), dtype=np.float64), sh)
    tile = jax.device_put(np.full((4, 4), 7.0, dtype=np.float64), sh)
    starts = device_put_process_local(
        np.asarray((2, 2), dtype=np.int32), rep)
    update = shard_local_update(mesh, spec=spec)
    got = np.asarray(update(dst, tile, starts))
    expected = np.zeros((6, 6), dtype=np.float64)
    expected[np.ix_((2, 5), (2, 5))] = 7.0
    np.testing.assert_array_equal(got, expected)


def _unstaged(mesh):
    """The historical chain: face constraint, then ask XLA for the batch."""
    face = NamedSharding(mesh, FACE)
    batch = NamedSharding(mesh, BATCH)

    @jax.jit
    def f(a):
        a = jax.lax.with_sharding_constraint(a, face)
        return jax.lax.with_sharding_constraint(a, batch)
    return f


# ---------------------------------------------------------------------------
# 1. value parity, and the red twin that proves the comparison can fail
# ---------------------------------------------------------------------------

def test_staged_reshard_returns_the_input_reassembled():
    mesh = _mesh()
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    out = jax.jit(face_to_batch_reshard(mesh))(a)
    # jax NORMALISES a PartitionSpec by dropping trailing ``None`` entries, so
    # ``P(('x','y'), None, None) != P(('x','y'),)`` as objects while being the
    # same sharding (the same trap ``wfn_transforms._sharding_key`` documents
    # at :319).  Compare the padded tuples, not the objects.
    got = tuple(out.sharding.spec) + (None,) * (3 - len(out.sharding.spec))
    assert got == (('x', 'y'), None, None), out.sharding.spec
    # Byte equality: the body issues all_to_all only — no arithmetic — so
    # this is the bit-exact parity class, not the 1e-12 reassociation class.
    np.testing.assert_array_equal(np.asarray(out), np.asarray(_probe()))


def test_value_parity_detects_a_wrong_block_map():
    """RED TWIN for test 1.

    Stage over the MINOR axis first.  That is still a volume-preserving
    pair of all_to_alls and still lands one whole (M, N) matrix per rank —
    it only numbers the B-blocks 'y'-major, contradicting the
    ``P(('x','y'), ...)`` out_specs.  If the comparison above could not see
    a wrong block map it would be worthless, so this must fail.
    """
    from common.shard_map import shard_map

    mesh = _mesh()

    def _body_swapped(t):
        t = jax.lax.all_to_all(t, 'y', split_axis=0, concat_axis=2, tiled=True)
        t = jax.lax.all_to_all(t, 'x', split_axis=0, concat_axis=1, tiled=True)
        return t

    bad = jax.jit(shard_map(_body_swapped, mesh=mesh,
                            in_specs=(FACE,), out_specs=BATCH,
                            check_vma=False))
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    got = np.asarray(bad(a))
    assert got.shape == (B, M, N)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(got, np.asarray(_probe()))


def test_staged_matches_the_unstaged_chain_value_for_value():
    """The unstaged chain is SLOW, not wrong — so it is the value oracle."""
    mesh = _mesh()
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    ref = np.asarray(_unstaged(mesh)(a))
    got = np.asarray(jax.jit(face_to_batch_reshard(mesh))(a))
    np.testing.assert_array_equal(got, ref)


def test_every_rank_holds_exactly_one_shard_of_the_batch():
    """Residency, stated as a shape rather than as a promise.

    The whole point is that no rank ever holds more than B·M·N/ndev
    elements.  Assert it on the addressable shards of the OUTPUT (the
    unstaged form's per-device buffer is the same size — it is the
    intermediate that blows up, which the HLO pin below covers).
    """
    mesh = _mesh()
    ndev = mesh.devices.size
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    out = jax.jit(face_to_batch_reshard(mesh))(a)
    for shard in out.addressable_shards:
        assert shard.data.shape == (B // ndev, M, N), shard.data.shape


# ---------------------------------------------------------------------------
# 2. what the COMPILED MODULE contains, and the FALSE arm that fires on it
# ---------------------------------------------------------------------------
#
# Read the optimized HLO, not the jaxpr and not the lowering: the optimized
# module is the only ground truth for what will actually run
# (QUALITY_PATTERNS §4).  Everything below is expressed in ISSUE SITES and
# BYTES, because those are the two things that mean the same on every
# backend — see the module docstring for what happened when this gate was
# written in one backend's spelling instead.
# ---------------------------------------------------------------------------

#: Opcodes that move bytes BETWEEN devices.  A collective that is not on this
#: list cannot appear in a movement-only reshard, so the pin below asserts the
#: whole dict rather than a chosen few counts.
_COLLECTIVE_OPCODES = (
    "all-to-all", "all-gather", "all-reduce", "reduce-scatter",
    "collective-permute", "collective-broadcast", "ragged-all-to-all",
)

#: ``%name = <shape> <opcode>(operands), attrs`` — the instruction form.
_INSTRUCTION = re.compile(
    r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*(.+?)\s+([\w-]+)\(")
#: ``c128[64,42,42]`` anywhere on a line (result, operand or tuple member).
_SHAPE_TOKEN = re.compile(r"\b([a-z]+[0-9]*)\[([0-9,]*)\]")
_ELEM_BYTES = {"pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "bf16": 2,
               "f16": 2, "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8,
               "f64": 8, "c64": 8, "c128": 16}


def _issued_collectives(txt):
    """``{opcode: count}`` over the collectives the module ISSUES.

    Three ways of counting this have now been wrong in this file's history,
    so the reasoning is written down rather than re-derived:

    * ``txt.count('all-to-all')`` also matches the instruction NAMES XLA
      derives from the opcode (``%all-to-all.3``) and the ``op_name=``
      metadata XLA propagates onto ordinary LOCAL ops — the staged module
      on CUDA carries twelve ``op_name="jit(_reshard)/all_to_all"``
      mentions, most of them on transposes inside ``%fused_transpose`` that
      move no bytes off the device at all.
    * matching the call form ``all-to-all(`` fixes that and introduces a
      worse bug: it is XLA:CPU's spelling.  XLA:GPU rewrites every
      collective into an async ``all-to-all-start(`` / ``all-to-all-done(``
      pair, so the counter reads **0** on CUDA with both exchanges present.
      That is how this gate was vacuous on the production platform from the
      day it was written until 2026-08-10.
    * counting ``-start`` and ``-done`` separately double-counts one
      collective as two.

    So: take the opcode from the ``= <shape> <opcode>(`` position, fold
    ``-start``/``-done`` back onto the collective they implement, and count
    the ISSUE (the synchronous form, or the ``-start``) — never the
    completion.
    """
    counts = {}
    for line in txt.splitlines():
        m = _INSTRUCTION.match(line)
        if not m:
            continue
        opcode = m.group(2)
        if opcode.endswith("-done"):
            continue          # the completion of one already counted
        base = opcode[:-len("-start")] if opcode.endswith("-start") else opcode
        if base in _COLLECTIVE_OPCODES:
            counts[base] = counts.get(base, 0) + 1
    return counts


def _largest_buffer_bytes(txt):
    """Bytes in the biggest per-device buffer any instruction names.

    After SPMD partitioning every shape in the module is a PER-DEVICE shape,
    so this is a residency statement and not a bookkeeping one: a
    movement-only reshard whose largest buffer is one shard has, by
    construction, not replicated anything anywhere.  Operands and tuple
    members are scanned as well as results — an async collective's buffers
    live inside its tuple, and a replicated intermediate hiding there would
    otherwise be invisible.
    """
    biggest = 0
    for line in txt.splitlines():
        if not _INSTRUCTION.match(line):
            continue
        for dtype, dims in _SHAPE_TOKEN.findall(line):
            if dtype not in _ELEM_BYTES:
                continue
            n = 1
            for d in dims.split(","):
                if d.strip():
                    n *= int(d)
            biggest = max(biggest, n * _ELEM_BYTES[dtype])
    return biggest


def _module_facts(fn, a):
    """Compile once and return the three facts this gate is about."""
    compiled = fn.lower(a).compile()
    txt = compiled.as_text()
    return {
        "text": txt,
        "collectives": _issued_collectives(txt),
        "largest_buffer": _largest_buffer_bytes(txt),
        "temp": int(compiled.memory_analysis().temp_size_in_bytes),
    }


def _assert_movement_only(facts, shard_bytes, full_bytes, where=""):
    """The staged reshard's whole claim about its compiled module.

    ONE helper, used by the TRUE arm and by the FALSE arm, so the two
    cannot drift apart and the twin proves this exact predicate can fail
    rather than a paraphrase of it.

    The three clauses, and the measured margins each carries (real 2x2
    A100 mesh and 4 emulated CPU devices, 2026-08-10,
    ``/pscratch/sd/j/jackm/reshard_instr_0810/_truth3/``):

    1. **schedule** — exactly two ``all-to-all`` issue sites and nothing
       else.  Staged: ``{'all-to-all': 2}`` on both platforms.  Unstaged:
       ``{'all-gather': 2}``.
    2. **residency** — no buffer larger than one shard.  Staged: exactly
       one shard on both platforms (1,806,336 B at the deck geometry).
       Unstaged: a full ``c128[64,84,84]``, four shards.
    3. **accounting** — the compile-time temp allocation stays below the
       size of the WHOLE array, which is what full replication costs.
       Staged: 1.00x one shard on GPU, 2.02x on the emulated CPU mesh, so
       the threshold at 4.0x (= the whole array at this device count) has
       a 2x margin on the tightest side.  Unstaged: 6.0x on both.
    """
    tag = f" [{where}]" if where else ""
    assert facts["collectives"] == {"all-to-all": 2}, (
        f"schedule{tag}: expected exactly two all-to-all issue sites and no "
        f"other collective, got {facts['collectives']}.  A movement-only "
        f"reshard that issues an all-gather or a reduce-scatter is the "
        f"unstaged behaviour this primitive exists to remove.\n"
        + facts["text"])
    assert facts["largest_buffer"] == shard_bytes, (
        f"residency{tag}: the biggest per-device buffer in the module is "
        f"{facts['largest_buffer']} B, not one shard ({shard_bytes} B).  "
        f"Every intermediate in the staged chain is exactly one shard by "
        f"construction (module docstring, PER-RANK RESIDENCY), so anything "
        f"else means the partitioner materialised something.\n"
        + facts["text"])
    assert facts["temp"] < full_bytes, (
        f"accounting{tag}: temp_size_in_bytes is {facts['temp']} B, at or "
        f"above the whole array ({full_bytes} B) — which is exactly what "
        f"replicate-then-partition costs.  Staged compiles measure 1.0-2.1x "
        f"one shard ({shard_bytes} B).\n" + facts["text"])


def test_hlo_pin_two_all_to_all_zero_all_gather():
    """The pin, GPU-first.  Was ``2 all-to-all, 0 all-gather`` counted in
    XLA:CPU's spelling and therefore 0-of-2 on CUDA; now the issue-site
    count, plus the residency and accounting clauses that say what the
    count was a proxy for."""
    mesh = _mesh()
    ndev = mesh.devices.size
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    facts = _module_facts(jax.jit(face_to_batch_reshard(mesh)), a)
    full = B * M * N * 16
    _assert_movement_only(facts, full // ndev, full, where="staged")


def test_band_to_product_r_preserves_order_layout_and_one_shard_hlo():
    """The rank-4 wavefunction move is exact and movement-only on 2x2."""
    mesh = _mesh()
    ndev = int(mesh.devices.size)
    a = jax.device_put(
        _product_probe(), NamedSharding(mesh, PRODUCT_BAND))
    fn = band_to_product_r_reshard(mesh)

    out = fn(a)
    # PRODUCT_R ends in a non-None entry, so JAX's normalized spec has all
    # four entries and can be pinned exactly rather than padded for comparison.
    assert tuple(out.sharding.spec) == tuple(PRODUCT_R), out.sharding.spec
    np.testing.assert_array_equal(np.asarray(out), np.asarray(_product_probe()))

    full = RK * RB * RS * RR * 16
    facts = _module_facts(fn, a)
    _assert_movement_only(
        facts, full // ndev, full, where="band-to-product-r")


def test_band_to_product_r_refuses_a_wrong_concrete_source_layout():
    """A shard_map boundary may not silently synthesize the pre-reshard."""
    mesh = _mesh()
    a = jax.device_put(_product_probe(), NamedSharding(mesh, PRODUCT_R))
    with pytest.raises(ValueError, match="implicit pre-reshard"):
        band_to_product_r_reshard(mesh)(a)


def test_hlo_red_twin_the_unstaged_chain_replicates():
    """RED TWIN for the pin above, and its FALSE ARM.

    The unstaged chain goes through the SAME assertion helper and must
    raise — if it did not, the pin would be measuring nothing.  Written as
    ``pytest.raises`` on the helper rather than as an independent
    re-statement of what "bad" looks like, because a twin that checks a
    paraphrase proves only that the paraphrase can fail.

    Then, positively: the replication has to be THERE and be nameable, so
    that a future XLA which stops replicating turns this cell red (meaning
    "the staging is no longer needed") instead of leaving it quietly true.
    """
    mesh = _mesh()
    ndev = mesh.devices.size
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    facts = _module_facts(_unstaged(mesh), a)
    full = B * M * N * 16
    with pytest.raises(AssertionError) as exc:
        _assert_movement_only(facts, full // ndev, full, where="unstaged")
    assert "schedule [unstaged]" in str(exc.value), str(exc.value)[:400]

    assert facts["collectives"].get("all-gather", 0) >= 1, (
        "the unstaged chain compiled with NO all-gather — then this XLA no "
        f"longer replicates to make this move ({facts['collectives']}) and "
        "the whole gate is vacuous; re-measure before trusting the staged "
        "path's numbers.\n" + facts["text"])
    assert facts["largest_buffer"] >= full, (
        "the unstaged chain materialised nothing bigger than "
        f"{facts['largest_buffer']} B, below the whole array ({full} B) — "
        "the full-batch replication this primitive exists to remove is not "
        "there any more.\n" + facts["text"])
    assert facts["temp"] >= full, (
        f"the unstaged chain's temp is {facts['temp']} B, below the whole "
        f"array ({full} B).\n" + facts["text"])


# ---------------------------------------------------------------------------
# 3. the production instrument: a COLD COMPILE at the deck geometry
# ---------------------------------------------------------------------------
#
# WHAT THIS REPLACED, AND THE PROOF THAT IT HAD TO BE REPLACED
# ------------------------------------------------------------
# Until 2026-08-10 this section grepped a subprocess's stderr for the
# compiler's own ``Involuntary full rematerialization`` warning — the line
# the campaign harness counts out of a job log, and the line the module
# docstring of ``common.staged_reshard`` quotes as its diagnostic. Its red
# twin required the UNSTAGED chain to emit it, on the principle that a grep
# which has never printed a hit is not evidence of absence.
#
# The twin can no longer print a hit, so the instrument is retired. DEAD
# PROOF, measured on a real 2x2 A100 mesh and on 4 emulated CPU devices on
# the same node (evidence ``/pscratch/sd/j/jackm/reshard_instr_0810/``,
# ``_truth3/*.stderr`` and ``leg3.log``, jax 0.7.0.dev20260810):
#
#   platform   partitioner   payload replicated   'Involuntary...' lines
#   CUDA       Shardy        30 KB / 7.2 MB / 67 MB          0 / 0 / 0
#   CUDA       legacy GSPMD  30 KB / 7.2 MB / 67 MB          0 / 0 / 0
#   CPU x4     Shardy        30 KB / 7.2 MB                    0 / 0
#   CPU x4     legacy GSPMD  30 KB                                 0
#
# — sixteen compiles of the chain that DOES replicate, at three sizes, on
# both platforms and under both partitioners, with the partitioner choice
# verified per-arm by the presence or absence of XLA's own "Using Shardy
# for XLA SPMD propagation" line. Not one hit. Raising the partitioner's
# verbosity (``TF_CPP_VMODULE=spmd_partitioner=3``) produces none either.
#
# And the message has NOT merely been reworded: ``strings`` finds it exactly
# once in ``libjax_common.so`` and once in ``xla_cuda_plugin.so`` in the
# shipped image. The literal the harness greps for is still compiled in;
# this pattern simply no longer reaches the branch that logs it. So the
# retirement is not "the string moved" — it is "the channel is silent for
# the thing we are gating".
#
# The HAZARD, however, is entirely intact and entirely measurable: at the
# deck geometry the unstaged chain still names a ``c128[64,84,84]``
# full-batch buffer on a single device and still asks for 10,838,532 B of
# temp against a 1,806,336 B shard — 6.0x. So what is retired is the
# CHANNEL (a compiler log line), not the claim. The claim is now read off
# the compiled module itself, which is where it was always true.
#
# Consequence outside this file, registered rather than fixed here: any
# A/B that quotes "involuntary-remat lines N -> 0" is quoting an instrument
# that reads zero on both arms on this stack. The 2026-07-31 measurement in
# ``common/staged_reshard``'s docstring was taken on Frontera's XLA, where
# the line was live; it is history, not a check that can be re-run.
# ---------------------------------------------------------------------------

#: The geometry the module docstring's own SPMD diagnostic was written from
#: (``c128[64,84,84]`` at ``rank`` 672 / ``bs`` 64, job 7882974), scaled to
#: this gate's 2x2 mesh. Deliberately NOT the (8, 12, 20) toy the in-process
#: cells use: an instrument is certified at the geometry it is consumed at,
#: and a 30 KB array is small enough that a backend could reasonably decide
#: to replicate it and be right.
DECK_B, DECK_M, DECK_N = 64, 84, 84

_SUBPROC = textwrap.dedent(
    """
    import sys
    import numpy as np, jax, jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.staged_reshard import face_to_batch_reshard
    B, M, N = {B}, {M}, {N}
    FACE, BATCH = P(None,'x','y'), P(('x','y'), None, None)
    # REFUSE rather than emulate: a cold compile that quietly ran on one
    # device, or on the host platform because the parent's pin hid the GPUs,
    # would report a clean module and mean nothing.
    print('NDEV', jax.device_count(), flush=True)
    print('PLATFORM', jax.devices()[0].platform, flush=True)
    devs = np.array(jax.devices()[:4]).reshape(2, 2)
    mesh = Mesh(devs, ('x','y'))
    idx = np.arange(B*M*N, dtype=np.float64).reshape(B, M, N)
    a = jax.device_put(jnp.asarray(idx + 1j*idx), NamedSharding(mesh, FACE))
    if sys.argv[1] == 'staged':
        f = jax.jit(face_to_batch_reshard(mesh))
    else:
        face, batch = NamedSharding(mesh, FACE), NamedSharding(mesh, BATCH)
        @jax.jit
        def f(x):
            x = jax.lax.with_sharding_constraint(x, face)
            return jax.lax.with_sharding_constraint(x, batch)
    c = f.lower(a).compile()
    open(sys.argv[2], 'w').write(c.as_text())
    print('TEMP', int(c.memory_analysis().temp_size_in_bytes), flush=True)
    print('COMPILED', flush=True)
    """
)


def _compile_in_subprocess(which, tmp_path):
    """A COLD compile in a fresh interpreter, and the facts it produced.

    The HLO comes back as a file rather than as parsed numbers so that the
    predicate has exactly one implementation (``_assert_movement_only``),
    shared with the in-process cells above.
    """
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Inherit PYTHONPATH — under the campaign harness this file is a COPY in
    # wk_REL and the pinned source lives in a snapshot, so a ``__file__``-
    # relative guess would import a DIFFERENT tree than the suite itself and
    # the subprocess would silently gate the wrong source.  Add the repo's
    # own src/ only when this file really is sitting in a checkout.
    repo_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if os.path.isdir(os.path.join(repo_src, "common")):
        env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
    hlo = os.path.join(str(tmp_path), f"{which}.optimized.hlo.txt")
    p = subprocess.run(
        [sys.executable, "-c",
         _SUBPROC.format(B=DECK_B, M=DECK_M, N=DECK_N), which, hlo],
        env=env, capture_output=True, text=True, timeout=900)
    assert "COMPILED" in p.stdout, (
        f"the {which} subprocess did not compile — anything it reported "
        f"would be vacuous.\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    # The child must have got the SAME platform and four real devices as the
    # process that launched it; four emulated CPU devices on a GPU node is
    # the failure the peer conftest lane measured and named.
    ndev = int(re.search(r"^NDEV (\d+)$", p.stdout, re.M).group(1))
    plat = re.search(r"^PLATFORM (\w+)$", p.stdout, re.M).group(1)
    assert ndev == 4, f"the {which} subprocess saw {ndev} devices, not 4"
    assert plat == jax.devices()[0].platform, (
        f"the {which} subprocess compiled on {plat!r} while this process is "
        f"on {jax.devices()[0].platform!r} — a cold compile on a different "
        f"platform is not this platform's evidence")
    with open(hlo) as fh:
        txt = fh.read()
    return {
        "text": txt,
        "collectives": _issued_collectives(txt),
        "largest_buffer": _largest_buffer_bytes(txt),
        "temp": int(re.search(r"^TEMP (\d+)$", p.stdout, re.M).group(1)),
        "stderr": p.stderr,
    }


def test_red_twin_the_unstaged_chain_replicates_at_the_deck_geometry(tmp_path):
    """RED TWIN for the production instrument, at the deck geometry.

    Replaces ``test_red_twin_the_unstaged_chain_emits_the_spmd_warning``,
    which required the compiler's ``Involuntary full rematerialization``
    line and cannot pass on this stack — see the dead proof at the head of
    this section.  Same role, same subject, a channel that still carries
    the signal: a cold compile of the UNSTAGED chain must SHOW the
    replication, both as a collective that gathers and as a buffer the size
    of the whole array.
    """
    facts = _compile_in_subprocess("unstaged", tmp_path)
    full = DECK_B * DECK_M * DECK_N * 16
    ndev = 4
    with pytest.raises(AssertionError):
        _assert_movement_only(facts, full // ndev, full, where="cold unstaged")
    assert facts["collectives"].get("all-gather", 0) >= 1, (
        "a cold compile of the unstaged chain issued no all-gather at the "
        f"deck geometry ({facts['collectives']}) — this XLA no longer "
        "replicates to make the move, so the staged path's numbers no "
        "longer measure a saving.  Re-measure before trusting them.")
    assert facts["largest_buffer"] >= full, (
        f"the unstaged chain's biggest buffer is {facts['largest_buffer']} "
        f"B, below the whole array ({full} B) — the full-batch replication "
        "this primitive removes is not there any more.")


def test_the_staged_chain_holds_one_shard_at_the_deck_geometry(tmp_path):
    """The instrument itself: a cold compile of the STAGED chain moves two
    shards and never holds more than one.

    Replaces ``test_the_staged_chain_emits_no_spmd_warning``.  The old cell
    asserted the ABSENCE of a compiler line, which on this stack is absent
    from every module including the ones that replicate — it was green for
    the wrong reason.  This asserts the presence of the property.
    """
    facts = _compile_in_subprocess("staged", tmp_path)
    full = DECK_B * DECK_M * DECK_N * 16
    _assert_movement_only(facts, full // 4, full, where="cold staged")


# ---------------------------------------------------------------------------
# 4. refusals — before any collective, with the fix named
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(B + 1, M, N), (B, M + 1, N), (B, M, N + 1)])
def test_indivisible_extent_is_refused_with_the_fix_named(shape):
    mesh = _mesh()
    assert not face_to_batch_reshard_supported(mesh, shape)
    reshard = face_to_batch_reshard(mesh)
    a = jnp.zeros(shape, dtype=jnp.complex128)
    with pytest.raises(ValueError) as exc:
        jax.jit(reshard).lower(a)
    msg = str(exc.value)
    assert "carrier extent" in msg and "expected" in msg, msg


def test_supported_accepts_the_divisible_case():
    mesh = _mesh()
    assert face_to_batch_reshard_supported(mesh, (B, M, N))


def test_inverted_mesh_is_refused():
    """§3.2 doctrine: the LAST mesh axis is the one with consecutive-rank
    replica groups, and it is also the one ``P(('x','y'), ...)`` numbers
    minor.  A mesh built the other way round must refuse, not silently
    renumber the batch."""
    inverted = _mesh(names=('y', 'x'))
    assert not face_to_batch_reshard_supported(inverted, (B, M, N))
    with pytest.raises(ValueError) as exc:
        face_to_batch_reshard(inverted)
    assert "minor axis" in str(exc.value)


def test_non_rank3_is_refused():
    mesh = _mesh()
    reshard = face_to_batch_reshard(mesh)
    with pytest.raises(ValueError) as exc:
        jax.jit(reshard).lower(jnp.zeros((B, M), dtype=jnp.complex128))
    assert "rank-3" in str(exc.value)
