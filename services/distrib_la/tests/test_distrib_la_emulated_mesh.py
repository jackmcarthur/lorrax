"""Layer L-b: four emulated devices, real arrays, real shard_map.

L-a is arithmetic about tiles; this tier actually builds them.  Four host
devices in ONE process, a 2x2 ('x','y') mesh, and the pure-JAX backends
running over it for real.

WHY ONLY THE PURE-JAX BACKENDS LIVE HERE.  GUARD 4 (``resolve.py``, the
coverage rung) refuses every FFI backend when ``jax.process_count()`` is
smaller than ``mesh.devices.size``: cuSOLVERMp, SLATE and ScaLAPACK carry
a per-PROCESS MPI/NCCL context, so a single process pretending to be four
devices cannot drive them.  That is not a limitation of this file, it is
the contract — and it means an emulated mesh can exercise ``native``,
``native2d`` and the shape algebra, and NOTHING else.  The FFI backends on
a 2x2 are layer L-c (``test_distrib_la_multiproc.py``, four real ranks).

WHY native2d IS THE SUBJECT.  Coverage of ``common/cholesky_2d.py`` at
96a6399 was ZERO — the only in-tree references were string assertions
about its name — while ``common/vma.py`` records that a real VMA defect
was live inside it.  A distributed kernel with a known-live defect class
and no executing test is the highest-value thing in this package to point
a mesh at, and until this file nothing had.

WHY THE CELLS SKIP RATHER THAN ASSERT below four devices: device count is
a property of how the leg was LAUNCHED, not of the code under test.
``tests/KNOWN_FAILURES.md`` lists eleven cells that failed a 1-device leg
purely for writing ``assert n_dev >= 4``.  ``conftest.py`` sets
``XLA_FLAGS`` when it can and explains exactly when it cannot; the skip
reason names the leg that does run these.
"""

from __future__ import annotations

import numpy as np
import pytest
from lxkit.testing import hostile_extents, require_devices

import distrib_la as D
from distrib_la._native2d import block_size_for, dense_to_tiles, tiles_to_dense

RTOL = 1e-12          # the engine-parity bar (C8): relative, never bit-exact


def _mesh(px, py):
    """A ``px x py`` mesh of HOST devices.

    ``"cpu"`` explicitly, not ``jax.devices()``.  The emulation knob is
    ``--xla_force_host_platform_device_count``, which creates HOST
    devices; on any box with a GPU the default backend is ``cuda`` and
    ``jax.device_count()`` answers 1, so asking the default backend makes
    these cells skip on the machine where the flag worked (measured, WSL,
    one CudaDevice).  Asking for CPU devices also keeps this tier off the
    SHARED GPUs on Perlmutter, which is where it belongs: nothing here
    needs a GPU, and the FFI backends that do cannot run on an emulated
    mesh at all (guard 4).
    """
    import jax
    from jax.sharding import Mesh
    require_devices(px * py, "cpu")
    return Mesh(np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py),
                ("x", "y"))


def _hpd(rng, nq, n, dtype):
    """A Hermitian positive-definite stack, identical on every device.

    ``(n + 4)`` on the diagonal, not the contract suite's ``n``.  The
    hostile-geometry families reach n = 1 and n = 2, where ``n * I`` is not
    enough to dominate a standard normal: at n = 1 the matrix is
    ``Re(z) + 1``, which is indefinite whenever ``Re(z) < -1`` — about one
    draw in six.  MEASURED, not defensive: the ``empty-tiles-on-axis0``
    row hit it while collecting this file's numbers, and it would have
    been an intermittent red that looked like a kernel defect.
    """
    z = rng.standard_normal((nq, n, n))
    if np.dtype(dtype).kind == "c":
        z = z + 1j * rng.standard_normal((nq, n, n))
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))
            + (n + 4) * np.eye(n)[None]).astype(dtype)


def _put(a, mesh, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(np.asarray(a), NamedSharding(mesh, P(*spec)))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)


# ---------------------------------------------------------------------------
# native2d, on a mesh, against the reference — its first real correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,dtype", [
    (8, "float64"), (8, "complex128"),
    (16, "float64"), (16, "complex128"),
    (24, "complex128"), (32, "complex128"),
])
def test_native2d_factorizes_like_numpy(n, dtype):
    """``L L^H == A`` and ``L == numpy.linalg.cholesky(A)``.

    BOTH, because they fail differently: a wrong tile assembly can satisfy
    the reconstruction with a differently-permuted factor, and comparing
    only against numpy would not say whether the factor reconstructs.
    """
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(3)
    A = _hpd(rng, 3, n, dtype)
    L = np.asarray(D.plan("cholesky", mesh, backend="native2d",
                          n=n).batched(_put(A, mesh, (None, "x", "y"))))
    assert L.shape == A.shape and L.dtype == A.dtype
    recon = _rel(L @ np.conj(np.swapaxes(L, -1, -2)), A)
    vs_np = _rel(np.tril(L), np.linalg.cholesky(A))
    assert recon < RTOL, f"n={n} {dtype}: ||L L^H - A|| rel {recon:.3e}"
    assert vs_np < RTOL, f"n={n} {dtype}: L vs numpy rel {vs_np:.3e}"


def test_native2d_is_bit_deterministic_on_a_rerun():
    """Same operands, same mesh, same answer.  A blocked algorithm whose
    reduction order drifted between calls would show up here and nowhere
    else in this file."""
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(5)
    A = _put(_hpd(rng, 2, 16, "complex128"), mesh, (None, "x", "y"))
    p = D.plan("cholesky", mesh, backend="native2d", n=16)
    assert np.array_equal(np.asarray(p.batched(A)), np.asarray(p.batched(A)))


def test_native2d_tiles_round_trip_on_a_real_sharded_array():
    """L-a asserts this arithmetic on host arrays; here the same round trip
    runs through ``with_sharding_constraint`` on a 2x2, which is where a
    reshape that was right on one device can stop being right."""
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(7)
    A = _put(rng.standard_normal((3, 12, 12)), mesh, (None, "x", "y"))
    for b, J in ((6, 2), (3, 4), (2, 6)):
        assert 12 // b == J
        tiles = dense_to_tiles(A, b)
        assert tiles.shape == (3, J, J, b, b)
        back = np.asarray(tiles_to_dense(tiles, b))
        assert np.array_equal(np.tril(back), np.tril(np.asarray(A)))
        # The zeroed half is BLOCK-triangular, not triangular: inside a
        # DIAGONAL tile the upper elements survive, because that is what
        # jnp.linalg.cholesky is handed and it reads the lower triangle
        # itself.  Asserting the elementwise version instead is a check
        # that fails at b=2 (measured, L-a).
        idx = np.arange(12)
        above = (idx[:, None] // b) < (idx[None, :] // b)
        assert not back[:, above].any()


# ---------------------------------------------------------------------------
# HOSTILE GEOMETRY — non-dividing extents through the padding contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", hostile_extents((2, 2)),
                         ids=lambda c: c.name)
def test_a_nondividing_extent_survives_identity_padding(case):
    """The band-pad bug class, on the kernel that has a tile decomposition.

    A logical extent the mesh cannot tile is REFUSED (L-a pins that).  The
    fix is to pad: zero rows and columns, IDENTITY on the pad-block
    diagonal, factor at the padded extent, read the logical block back.
    Four assertions, and each one fails differently:

      1. the logical block matches the reference factor of the logical
         matrix -- the answer is right;
      2. the pad rows BELOW the logical block are EXACT zeros -- no
         garbage leaked down from the padded region;
      3. the pad block factors to EXACTLY the identity -- the pad did not
         perturb itself, which is what makes step 2 meaningful;
      4. the pad was NOT VACUOUS.  Mirrors the ``pad divisor made the
         check a no-op`` self-assertion in the FFI padding contract: if
         ``n_pad == n_log`` every assertion above is about the unpadded
         problem and the cell tests nothing.  Here it is stronger --
         ``block_size_for`` must REFUSE the logical extent, so the padding
         is provably load-bearing rather than merely nonzero.
    """
    mesh = _mesh(2, 2)
    n_log, n_pad = case.logical[0], case.padded[0]
    if n_log == n_pad:
        pytest.skip(f"{case.name}: logical extent {n_log} already tiles a "
                    f"2x2 mesh, so there is no padding to test here — the "
                    f"non-dividing families are the subject")

    # 4. ANTI-TAUTOLOGY, first, so a vacuous case cannot reach the rest.
    assert n_pad > n_log
    with pytest.raises(ValueError, match="no valid block size"):
        block_size_for(n_log, 2, 2)
    block_size_for(n_pad, 2, 2)              # ...and the padded one tiles

    rng = np.random.default_rng(11)
    nq, dtype = 2, "complex128"
    A_log = _hpd(rng, nq, n_log, dtype)
    A_pad = np.zeros((nq, n_pad, n_pad), dtype)
    A_pad[:, :n_log, :n_log] = A_log
    for i in range(n_log, n_pad):
        A_pad[:, i, i] = 1.0

    L = np.asarray(D.plan("cholesky", mesh, backend="native2d", n=n_pad)
                   .batched(_put(A_pad, mesh, (None, "x", "y"))))
    assert L.shape == (nq, n_pad, n_pad)

    # 1. the logical block is the factor of the logical matrix
    rel = _rel(np.tril(L[:, :n_log, :n_log]), np.linalg.cholesky(A_log))
    assert rel < RTOL, (
        f"{case.name}: padded logical block differs from the unpadded "
        f"factor by rel {rel:.3e} (pad-extent leak)")
    # 2. exact zeros under the logical block
    assert not L[:, n_log:, :n_log].any(), (
        f"{case.name}: pad rows are not exact zeros")
    # 3. the pad block factored to exactly I
    pad = L[:, n_log:, n_log:]
    assert np.array_equal(pad, np.broadcast_to(
        np.eye(n_pad - n_log, dtype=pad.dtype), pad.shape)), (
        f"{case.name}: the identity pad block did not factor to the "
        f"identity, so the zero-check above is about a perturbed problem")


def test_the_padding_check_can_fail():
    """RED TWIN for the cell above.  A pad block that is NOT the identity
    (zeros on the diagonal, the naive "just pad with zeros") must break the
    contract — otherwise the identity requirement is decoration and the
    check would pass on operands nobody prepared correctly."""
    mesh = _mesh(2, 2)
    n_log, n_pad, nq = 7, 8, 2
    rng = np.random.default_rng(13)
    A_log = _hpd(rng, nq, n_log, "complex128")
    A_bad = np.zeros((nq, n_pad, n_pad), "complex128")
    A_bad[:, :n_log, :n_log] = A_log        # NO identity on the pad diagonal
    L = np.asarray(D.plan("cholesky", mesh, backend="native2d", n=n_pad)
                   .batched(_put(A_bad, mesh, (None, "x", "y"))))
    pad = L[:, n_log:, n_log:]
    assert not np.array_equal(pad, np.broadcast_to(
        np.eye(n_pad - n_log, dtype=pad.dtype), pad.shape)), (
        "a zero pad block factored to the identity, so the identity-pad "
        "requirement asserts nothing")


# ---------------------------------------------------------------------------
# MESH INVARIANCE — 1x1 against 2x2
# ---------------------------------------------------------------------------

def test_native2d_agrees_between_a_1x1_and_a_2x2_mesh():
    """The same matrix, two mesh shapes, the same factor to 1e-12.

    NOT bit-exact, and the difference is the point.  ``block_size_for``
    picks J = lcm(Px,Py), so a 1x1 mesh factors n=16 as ONE 16x16 tile
    (a single ``jnp.linalg.cholesky``) and a 2x2 mesh factors it as a 2x2
    grid of 8x8 tiles driven by the blocked POTRF/TRSM/SYRK loop.  Those
    are different reductions in a different order; demanding bit-equality
    would be demanding that the algorithm not be blocked.  The repo's
    engine-parity bar is relative 1e-12 and NEVER bit-exact (C8), and this
    is exactly the case it was written for.

    The measured delta is asserted AND reported, because "how close" is
    the finding and "under the bar" is only the verdict.
    """
    mesh1 = _mesh(1, 1)
    mesh4 = _mesh(2, 2)
    rng = np.random.default_rng(17)
    for n in (8, 16, 32):
        A = _hpd(rng, 2, n, "complex128")
        L1 = np.asarray(D.plan("cholesky", mesh1, backend="native2d", n=n)
                        .batched(_put(A, mesh1, (None, "x", "y"))))
        L4 = np.asarray(D.plan("cholesky", mesh4, backend="native2d", n=n)
                        .batched(_put(A, mesh4, (None, "x", "y"))))
        rel = _rel(L4, L1)
        assert rel < RTOL, f"n={n}: 1x1 vs 2x2 rel {rel:.3e}"
        # The DISTINCTNESS control: the two runs really did take different
        # decompositions, so the agreement above is not the same code
        # compared with itself.
        assert block_size_for(n, 1, 1) != block_size_for(n, 2, 2)


def test_native_eigh_agrees_between_a_1x1_and_a_2x2_mesh():
    """``native`` is replicated ``jnp.linalg.eigh``: the mesh changes
    where the array lives and not what is computed, so here bit-equality
    IS the promise and is asserted as such.  Stating the weaker tolerance
    for both backends would hide that they differ."""
    mesh1, mesh4 = _mesh(1, 1), _mesh(2, 2)
    rng = np.random.default_rng(19)
    A = _hpd(rng, 2, 16, "complex128")
    W1, _ = D.plan("eigh", mesh1, backend="off").batched(
        _put(A, mesh1, (None, "x", "y")))
    W4, _ = D.plan("eigh", mesh4, backend="off").batched(
        _put(A, mesh4, (None, "x", "y")))
    assert np.array_equal(np.asarray(W1), np.asarray(W4))
    assert _rel(np.asarray(W4), np.linalg.eigvalsh(A)) < 1e-10


def test_native_eigh_recovers_a_padded_spectrum_on_a_2x2():
    """The eigh half of the padding contract.  An identity-padded operand
    must give back the logical spectrum plus exactly ``n_pad - n_log``
    eigenvalues equal to 1 — the sentinel convention the call sites use to
    slice the pad away."""
    mesh = _mesh(2, 2)
    n_log, n_pad = 13, 16
    rng = np.random.default_rng(23)
    A_log = _hpd(rng, 1, n_log, "complex128")[0]
    A_pad = np.zeros((n_pad, n_pad), "complex128")
    A_pad[:n_log, :n_log] = A_log
    for i in range(n_log, n_pad):
        A_pad[i, i] = 1.0
    assert n_pad > n_log                      # the pad is not vacuous
    W, _ = D.plan("eigh", mesh, backend="off")(_put(A_pad, mesh, ("x", "y")))
    W = np.sort(np.asarray(W))
    want = np.sort(np.concatenate(
        [np.linalg.eigvalsh(A_log), np.ones(n_pad - n_log)]))
    assert _rel(W, want) < 1e-10, f"padded spectrum rel {_rel(W, want):.3e}"


# ---------------------------------------------------------------------------
# The trace-safe end of the two-phase API, on a real mesh
# ---------------------------------------------------------------------------

def test_native_fn_is_traceable_and_matches_the_eager_call():
    """``Plan.native_fn`` is the pure closure a fusion-critical site puts
    INSIDE its own jit.  Two claims: it traces at all (no
    ``process_count``, no ``device_put``, no ``dlopen`` in the body), and
    it computes what the eager call computes."""
    import jax
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(29)
    A = _put(_hpd(rng, 2, 16, "complex128"), mesh, (None, "x", "y"))
    p = D.plan("cholesky", mesh, backend="native2d", n=16)
    eager = np.asarray(p.batched(A))
    jitted = np.asarray(jax.jit(p.native_fn)(A))
    assert np.array_equal(eager, jitted), (
        f"native_fn under jit differs from the eager call by "
        f"{_rel(jitted, eager):.3e}")


def test_an_ffi_backend_refuses_an_emulated_multi_device_mesh():
    """GUARD 4, observed rather than assumed.

    This is WHY this file only tests the pure-JAX backends, so it is the
    one thing here that must be checked rather than asserted in a comment.
    Four devices, one process: every FFI backend refuses, and the refusal
    names the coverage rule.  (On a machine with no ``.so`` the capability
    rung refuses first — both are resolve-time refusals with a reason,
    which is the contract; the rung depends on the machine.)
    """
    import jax
    mesh = _mesh(2, 2)
    assert jax.process_count() == 1 < mesh.devices.size
    for op, backend in (("cholesky", "cusolvermp"), ("cholesky", "slate"),
                        ("solve_lu", "scalapack"), ("eigh", "scalapack")):
        with pytest.raises((ValueError, RuntimeError)) as ei:
            D.resolve_backend(op, backend, mesh)
        assert str(ei.value).strip(), f"{op}/{backend} refused with no reason"
    # ...and the pure-JAX backends are NOT refused on the same mesh, which
    # is what makes the loop above about guard 4 and not about the mesh.
    assert D.resolve_backend("cholesky", "native2d", mesh, n=16) == "native2d"
    assert D.resolve_backend("eigh", "off", mesh) == "native"


# ---------------------------------------------------------------------------
# The batched surface IS a scan over the single-matrix op
# ---------------------------------------------------------------------------
#
# Guard 4 keeps every real FFI backend off an emulated mesh, so these cells
# put a STAND-IN backend module behind the plan: a module with the right
# entry-point names whose maths is ``jnp``.  That is not a weaker test of
# the thing under test — the thing under test is the batching structure
# (one traced body, the per-backend normaliser applied per matrix, the
# stack that comes out, the refusals), and none of it is library-specific.
# The libraries themselves are L-c, on four real ranks, and the
# batched-vs-serial bit-identity gate is where the two routes are compared
# on the real ones.
#
# The plans here are built by CONSTRUCTING ``Plan`` rather than calling
# ``plan()``, which the class docstring says never to do — correctly, for
# production code, because the constructor runs no guards.  Here that is
# the point: there is no ``.so`` to resolve against and the guards are
# somebody else's cells.

def _planmod():
    """The ``distrib_la.plan`` MODULE.

    Not ``from distrib_la import plan``: the door re-exports the
    ``plan()`` function under that name and the binding shadows the
    submodule, so the obvious spelling hands back a function.  Going
    through ``sys.modules`` is the only spelling that cannot be quietly
    wrong (the batched-eigh gate learned this the same way).
    """
    import distrib_la                                          # noqa: F401
    import sys
    return sys.modules["distrib_la.plan"]


def _stand_in_plan(monkeypatch, op, backend, mesh, module):
    """A ``Plan`` for ``(op, backend)`` whose backend module is ``module``.

    Empties ``plan._SCAN_CACHE`` first.  That cache is keyed on ``(op,
    backend, mesh, signature)`` and NOT on the module object, which is
    right in production — ``backend_module(name)`` is a singleton, so the
    name determines the function — and wrong here, where every cell puts a
    different stand-in behind the same name.  Without the reset, a cell
    that counts traces would silently be reading the PREVIOUS cell's
    compiled scan and counting zero.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    pm = _planmod()
    pm._SCAN_CACHE.clear()
    monkeypatch.setattr(pm, "backend_module", lambda b: module)
    return pm.Plan(op=op, requested=backend, backend=backend, mesh=mesh,
                   n=None,
                   in_sharding=NamedSharding(mesh, P("x", "y")),
                   batch_in_sharding=NamedSharding(mesh, P(None, "x", "y")))


def _eigh_stand_in(calls, raw_layout=False):
    """A module exposing ``distributed_eigh`` on ONE tile.

    ``raw_layout`` mimics cuSOLVERMp, whose wrapper returns the conjugate
    transpose of the eigenvector matrix rather than the columns — the
    convention ``plan._eigh_columns`` exists to normalise.
    """
    import types
    import jax.numpy as jnp

    def distributed_eigh(A, *, mesh):
        assert A.ndim == 2 and A.shape[0] == A.shape[1], (
            f"the scan must hand the single-matrix entry ONE tile; "
            f"got {A.shape}")
        calls.append(tuple(A.shape))
        w, Q = jnp.linalg.eigh(A)
        return (w, jnp.conj(jnp.swapaxes(Q, -1, -2))) if raw_layout else (w, Q)

    return types.SimpleNamespace(distributed_eigh=distributed_eigh)


@pytest.mark.parametrize("nq", [1, 2, 8])
def test_the_scan_route_reproduces_the_single_matrix_calls(monkeypatch, nq):
    """``p.batched(A)`` == stacking ``p(A[q])``, to the bit.

    This is the definition of the batched surface written as a check: the
    scan is the single-matrix op under a loop the compiler owns, so it may
    not change an answer.  Bit-equality is the right bar because both sides
    run the SAME kernel on the SAME operands — a relative bar would pass on
    a scan that quietly reassociated something.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(4041)
    A = jnp.asarray(_hpd(rng, nq, 8, "complex128"))

    calls_loop = []
    p_loop = _stand_in_plan(monkeypatch, "eigh", "slate", mesh,
                            _eigh_stand_in(calls_loop))
    W_ref, Z_ref = zip(*(p_loop(A[q]) for q in range(nq)))
    W_ref, Z_ref = jnp.stack(W_ref), jnp.stack(Z_ref)

    calls_scan = []
    p_scan = _stand_in_plan(monkeypatch, "eigh", "slate", mesh,
                            _eigh_stand_in(calls_scan))
    W, Z = p_scan.batched(A)

    assert p_scan.batched_route == D.ROUTE_SCAN
    assert (W.shape, Z.shape) == ((nq, 8), (nq, 8, 8))
    assert np.array_equal(np.asarray(W), np.asarray(W_ref))
    assert np.array_equal(np.asarray(Z), np.asarray(Z_ref))


@pytest.mark.parametrize("nq", [2, 8])
def test_the_scan_body_is_traced_once_however_long_the_batch(monkeypatch, nq):
    """ONE trace of the single-matrix entry, whatever ``nq`` is.

    This is the whole reason the batched surface is a scan and not a Python
    loop, reduced to something a laptop can falsify: the Python loop called
    the wrapper ``nq`` times and handed the compiler ``nq`` separate calls,
    and on the wrappers that build their ``shard_map`` eagerly that was
    ``nq`` traces and ``nq`` compiles.  The scan traces the body once at
    ``unroll=1`` and the count must not move with ``nq``.

    It fails the moment the scan is replaced by a loop again, which is
    exactly the regression worth a cell.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(4042)
    A = jnp.asarray(_hpd(rng, nq, 8, "complex128"))
    calls = []
    p = _stand_in_plan(monkeypatch, "eigh", "slate", mesh,
                       _eigh_stand_in(calls))
    p.batched(A)
    assert len(calls) == D.BATCHED_SCAN_UNROLL, (
        f"the single-matrix entry was traced {len(calls)} times for a "
        f"{nq}-matrix batch; a scan at unroll={D.BATCHED_SCAN_UNROLL} "
        f"traces it exactly that many times regardless of nq")
    assert calls[0] == (8, 8)


def test_a_repeated_batched_call_reuses_the_compiled_scan(monkeypatch):
    """Calling ``batched()`` again with the same signature does NOT retrace.

    MEASURED, and the reason the cache exists rather than being tidiness:
    Perlmutter CPU 2x2, nq=8 n=64, an UNCACHED eager scan cost 0.080 s warm
    against 0.011 s for the same scan as a compiled executable — ~69 ms of
    pure per-call retrace-and-relower, which made the scan SLOWER than the
    Python loop it replaced (0.024 s warm, which paid nothing per call
    because every backend call hit the wrapper's own jit cache).

    The trace count is the cheap proxy for that wall clock, and it is the
    thing that actually regresses if somebody drops the cache.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(4047)
    A = jnp.asarray(_hpd(rng, 4, 8, "complex128"))
    calls = []
    p = _stand_in_plan(monkeypatch, "eigh", "slate", mesh,
                       _eigh_stand_in(calls))
    first = p.batched(A)
    for _ in range(3):
        again = p.batched(A)
    assert len(calls) == D.BATCHED_SCAN_UNROLL, (
        f"four batched calls of one signature traced the body "
        f"{len(calls)} times; the compiled scan is cached per signature, "
        f"so it must be traced once")
    # ...and reuse must not have changed the answer.
    assert np.array_equal(np.asarray(first[0]), np.asarray(again[0]))
    assert np.array_equal(np.asarray(first[1]), np.asarray(again[1]))


def test_the_scan_route_applies_the_per_backend_normaliser(monkeypatch):
    """Eigenvectors come back as COLUMNS on the raw-layout backend too.

    cuSOLVERMp's wrapper returns the conjugate transpose of the
    eigenvector matrix, and the serial path once returned it unnormalised
    — ROWS, on the only platform that took that path.  The scan applies
    ``_eigh_columns`` per matrix for the same reason the loop body did, and
    the transposed residual is the negative control: without the
    normaliser the column check passes anyway on a Hermitian operand only
    if you do not look, which is how the original defect survived.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(4043)
    A = jnp.asarray(_hpd(rng, 3, 8, "complex128"))
    p = _stand_in_plan(monkeypatch, "eigh", "cusolvermp", mesh,
                       _eigh_stand_in([], raw_layout=True))
    W, Z = p.batched(A)

    def _resid(Z_):
        lhs = jnp.einsum("qmn,qnp->qmp", A, Z_)
        rhs = Z_ * W[:, None, :].astype(Z_.dtype)
        return float(jnp.max(jnp.abs(lhs - rhs))) / float(jnp.max(jnp.abs(A)))

    assert _resid(Z) < 1e-12, "A Z != Z diag(W): the scan returned ROWS"
    assert _resid(jnp.conj(jnp.swapaxes(Z, -1, -2))) > 1e-6, (
        "the transposed control also passes, so the check above is "
        "vacuous on this operand and proves nothing about the layout")


def test_a_handle_returning_backend_refuses_the_scan_BEFORE_it_runs(
        monkeypatch):
    """SLATE cholesky refuses ``batched()`` without calling the library.

    Its single-matrix ``potrf`` returns a ``SlateLowerL``, so there is no
    array stack for a scan to build.  ``_IMPL`` DECLARES that, which is why
    the refusal can name the op, the backend and the surface to use instead
    — and why nothing runs first.  ``calls == []`` is the half of this that
    a discovered-by-calling check could not make.
    """
    import types
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    calls = []

    def distributed_cholesky(A, *, mesh):
        calls.append(tuple(A.shape))
        return object()                       # a handle, as SLATE's is

    p = _stand_in_plan(monkeypatch, "cholesky", "slate", mesh,
                       types.SimpleNamespace(
                           distributed_cholesky=distributed_cholesky))
    rng = np.random.default_rng(4044)
    A = jnp.asarray(_hpd(rng, 3, 8, "complex128"))
    with pytest.raises(NotImplementedError) as ei:
        p.batched(A)
    msg = str(ei.value)
    assert "slate" in msg and "HANDLE" in msg and "factor()" in msg
    assert calls == [], (
        f"the refusal ran the library first ({len(calls)} calls); it is a "
        f"declared property of the row, not something to discover by "
        f"calling")


def test_without_the_declaration_the_handle_case_fails_unreadably(
        monkeypatch):
    """RED TWIN for the cell above — why ``one_handle`` is DECLARED.

    Clear the flag and the same call still fails, but from inside
    ``lax.scan``, with a message that names neither the op nor the backend
    nor what to do instead.  That is the failure the declaration buys out,
    and this cell is the evidence that it is a real one rather than a
    stylistic preference.
    """
    import types
    import jax.numpy as jnp
    pm = _planmod()
    mesh = _mesh(2, 2)
    calls = []

    def distributed_cholesky(A, *, mesh):
        calls.append(tuple(A.shape))
        return object()

    row = dict(pm._IMPL[("cholesky", "slate")])
    row.pop("one_handle")
    monkeypatch.setitem(pm._IMPL, ("cholesky", "slate"), row)

    p = _stand_in_plan(monkeypatch, "cholesky", "slate", mesh,
                       types.SimpleNamespace(
                           distributed_cholesky=distributed_cholesky))
    rng = np.random.default_rng(4045)
    A = jnp.asarray(_hpd(rng, 3, 8, "complex128"))
    with pytest.raises(Exception) as ei:
        p.batched(A)
    assert not isinstance(ei.value, NotImplementedError), (
        "clearing one_handle still produced the readable refusal, so the "
        "declaration is not what produces it and this twin is not red")
    assert calls, "the library was never reached, so nothing was learned"


def test_the_reserved_batch_reshard_route_refuses_by_name(monkeypatch):
    """Route (c) is a NAME with no code, and asking for it says so.

    The reserved slot exists so that adding the batch-axis reshard is an
    edit in the one place that already decides, rather than a new surface.
    A reserved name that silently did something else — fell back to the
    scan, say — would defeat the whole point of reserving it.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    p = _stand_in_plan(monkeypatch, "eigh", "slate", mesh,
                       _eigh_stand_in([]))
    rng = np.random.default_rng(4046)
    A = jnp.asarray(_hpd(rng, 2, 8, "complex128"))
    with pytest.raises(NotImplementedError) as ei:
        p.batched(A, _route=D.ROUTE_BATCH_RESHARD)
    assert "face_to_batch_reshard" in str(ei.value)
    with pytest.raises(ValueError) as ei2:
        p.batched(A, _route="whatever")
    assert D.ROUTE_SCAN in str(ei2.value)
