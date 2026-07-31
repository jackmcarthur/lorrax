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
2. **HLO pin** — the staged module carries exactly two ``all-to-all`` and
   ZERO ``all-gather``.  Red twin: the unstaged single-constraint chain,
   compiled in the same test, must carry a full-batch replication.
3. **the production instrument** — the compiler's own
   ``Involuntary full rematerialization`` warning, grepped out of a
   subprocess's stderr exactly as the campaign harness greps it out of a job
   log.  Red twin: the unstaged chain in the same subprocess must EMIT it.
   A grep that has never printed a hit is not evidence of absence.
4. **refusals** — divisibility and inverted-mesh, each raising before any
   collective with the caller's fix named.

Run on 4 emulated devices::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_ENABLE_X64=1 \
        python -m pytest tests/test_staged_reshard.py -q
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.staged_reshard import (
    face_to_batch_reshard,
    face_to_batch_reshard_supported,
)

pytestmark = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="needs >=4 devices: XLA_FLAGS=--xla_force_host_platform_device_count=4")


# (B, M, N) chosen so every extent divides but none is equal to another —
# a transposed axis or a swapped stage would change the shape, not merely
# the values, and be caught twice.
B, M, N = 8, 12, 20
FACE = P(None, 'x', 'y')
BATCH = P(('x', 'y'), None, None)


def _mesh(px=2, py=2, names=('x', 'y')):
    devs = np.array(jax.devices()[:px * py]).reshape(px, py)
    return Mesh(devs, names)


def _probe(b=B, m=M, n=N):
    """Values that encode their own (batch, row, col) index."""
    idx = np.arange(b * m * n, dtype=np.float64).reshape(b, m, n)
    return jnp.asarray(idx + 1j * (idx * 0.5 + 1.0))


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
    from jax.experimental.shard_map import shard_map

    mesh = _mesh()

    def _body_swapped(t):
        t = jax.lax.all_to_all(t, 'y', split_axis=0, concat_axis=2, tiled=True)
        t = jax.lax.all_to_all(t, 'x', split_axis=0, concat_axis=1, tiled=True)
        return t

    bad = jax.jit(shard_map(_body_swapped, mesh=mesh,
                            in_specs=(FACE,), out_specs=BATCH,
                            check_rep=False))
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
# 2. HLO pin + its red twin
# ---------------------------------------------------------------------------

def _optimized_text(fn, a):
    return fn.lower(a).compile().as_text()


def _op_count(txt, op):
    """Count real op INVOCATIONS, not mentions.

    ``txt.count('all-to-all')`` also matches the instruction NAMES XLA
    derives from the opcode (``%all-to-all.3``) and any async
    start/done pair, so it over-counts by a backend-dependent factor.
    Match the call form instead.
    """
    import re
    return len(re.findall(r"(?<![-\w])" + re.escape(op) + r"\(", txt))


def test_hlo_pin_two_all_to_all_zero_all_gather():
    mesh = _mesh()
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    txt = _optimized_text(jax.jit(face_to_batch_reshard(mesh)), a)
    n_a2a = _op_count(txt, "all-to-all")
    n_ag = _op_count(txt, "all-gather")
    assert n_a2a == 2, f"expected 2 all-to-all, got {n_a2a}\n{txt}"
    assert n_ag == 0, f"expected 0 all-gather, got {n_ag}\n{txt}"


def test_hlo_red_twin_the_unstaged_chain_replicates():
    """RED TWIN for the HLO pin.

    The unstaged chain must show the replication (an all-gather, or a
    literal full-shape buffer) that the staged one does not.  If BOTH
    modules looked clean the pin above would be measuring nothing.
    """
    mesh = _mesh()
    a = jax.device_put(_probe(), NamedSharding(mesh, FACE))
    txt = _optimized_text(_unstaged(mesh), a)
    full = f"[{B},{M},{N}]"
    assert (_op_count(txt, "all-gather") > 0) or (full in txt), (
        "the unstaged chain compiled WITHOUT any full-batch replication — "
        "then this XLA no longer needs the staging and the whole gate is "
        "vacuous; re-measure before trusting the staged path's numbers.\n"
        + txt)


# ---------------------------------------------------------------------------
# 3. the production instrument: the compiler's own warning, and its red twin
# ---------------------------------------------------------------------------

_SUBPROC = textwrap.dedent(
    """
    import sys
    import numpy as np, jax, jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.staged_reshard import face_to_batch_reshard
    B, M, N = {B}, {M}, {N}
    FACE, BATCH = P(None,'x','y'), P(('x','y'), None, None)
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
    f.lower(a).compile()
    print('COMPILED', flush=True)
    """
)

_WARNING = "Involuntary full rematerialization"


def _compile_in_subprocess(which):
    env = dict(os.environ)
    env.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
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
    p = subprocess.run(
        [sys.executable, "-c", _SUBPROC.format(B=B, M=M, N=N), which],
        env=env, capture_output=True, text=True, timeout=900)
    assert "COMPILED" in p.stdout, (
        f"the {which} subprocess did not compile — the warning count it "
        f"reports would be vacuous.\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stderr


def test_red_twin_the_unstaged_chain_emits_the_spmd_warning():
    """RED TWIN for the campaign's primary instrument.

    ``grep -c 'Involuntary full rematerialization'`` over the job log is
    what the A/B gate reads.  A grep that has never printed a hit proves
    nothing (README §5.1: ``grep -c`` printed 0 while exiting 1 in this
    campaign and was believed).  So: the UNSTAGED chain must emit it.
    """
    err = _compile_in_subprocess("unstaged")
    assert _WARNING in err, (
        "the unstaged chain compiled with NO involuntary-remat warning on "
        "this XLA — the instrument cannot go red here, so a zero count from "
        "the staged path means nothing.\n" + err)


def test_the_staged_chain_emits_no_spmd_warning():
    err = _compile_in_subprocess("staged")
    assert _WARNING not in err, err


# ---------------------------------------------------------------------------
# 4. refusals — before any collective, with the fix named
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(B + 1, M, N), (B, M + 1, N), (B, M, N + 1)])
def test_indivisible_extent_is_refused_with_the_fix_named(shape):
    mesh = _mesh()
    assert not face_to_batch_reshard_supported(mesh, shape)
    reshard = face_to_batch_reshard(mesh, divisibility_hint="HINT-TOKEN")
    a = jnp.zeros(shape, dtype=jnp.complex128)
    with pytest.raises(ValueError) as exc:
        jax.jit(reshard).lower(a)
    msg = str(exc.value)
    assert "padded_extent" in msg and "HINT-TOKEN" in msg, msg


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
