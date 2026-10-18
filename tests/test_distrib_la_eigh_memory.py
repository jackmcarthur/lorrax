"""EIGHMEM: the distributed eigh route's buffers, and that none of the changes moves a bit (CPU).

* cyclic_to_block in rounds of ppermute pieces is the one-shot all_to_all's result, bitwise;
* the gather-free column permutation is take_along_axis's result, bitwise;
* the deferred reorder (deflate_zero_rows(defer=True) + sort_eigenpairs) is the in-program
  reorder's result, bitwise;
* the eager Plan route reads the reorder flag with its check status and reorders in a
  separate program; the eager, retried, traced, checked_program and donated calls agree
  bitwise (a numeric stand-in for the library);
* the cuSOLVERMp wrapper hands the solver its operand as an aliased result (ABI 10).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


@pytest.fixture(scope="module")
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _bits(x):
    return np.asarray(x).view(np.uint8)


@pytest.mark.parametrize("n,mb,axis", [(128, 8, "x"), (128, 8, "y"), (192, 4, "x"),
                                       (80, 5, "y"), (42, 3, "x"), (70, 5, "y"), (1024, 16, "y")])
def test_rounds_reshuffle_is_the_all_to_all_bitwise(mesh, n, mb, axis):
    from jax import shard_map
    from distrib_la._cyclic import cyclic_to_block, cyclic_to_block_all_to_all
    rng = np.random.default_rng(n + mb)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    outs = []
    for f in (cyclic_to_block, cyclic_to_block_all_to_all):
        body = ((lambda q, f=f: f(q, n=n, p=2, mb=mb, axis="x")) if axis == "x" else
                (lambda q, f=f: f(q.T, n=n, p=2, mb=mb, axis="y").T))
        outs.append(jax.jit(shard_map(body, mesh=mesh, in_specs=P("x", "y"),
                                      out_specs=P("x", "y"), check_vma=False))(a))
    assert np.array_equal(_bits(outs[0]), _bits(outs[1]))


@pytest.mark.parametrize("n", [64, 34])          # 34: n/p odd, the global take
def test_gather_free_column_permutation_is_the_take_bitwise(mesh, n):
    from distrib_la._result_check import _permute_columns
    rng = np.random.default_rng(n)
    v = jax.device_put(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)),
                       NamedSharding(mesh, P("x", "y")))
    order = jnp.asarray(rng.permutation(n))
    got = jax.jit(lambda v, o: _permute_columns(v, o, mesh))(v, order)
    want = jnp.take_along_axis(np.asarray(v), np.asarray(order)[None, :], axis=-1)
    assert np.array_equal(_bits(got), _bits(want))


def _padded_indefinite(n, live, seed=3):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((live // 2, live)) + 1j * rng.standard_normal((live // 2, live))
    h = b.conj().T @ b / live - 0.05 * np.eye(live)     # negative live eigenvalues
    rows = np.sort(np.random.default_rng(5).permutation(n)[:live])
    a = np.zeros((n, n), complex)
    a[np.ix_(rows, rows)] = h
    return a


def test_deferred_reorder_is_the_in_program_reorder_bitwise():
    from distrib_la._result_check import deflate_zero_rows, sort_eigenpairs
    a = jnp.asarray(_padded_indefinite(40, 30))
    now = jax.jit(deflate_zero_rows(jnp.linalg.eigh))(a)
    w, v, flag = jax.jit(deflate_zero_rows(jnp.linalg.eigh, defer=True))(a)
    assert bool(flag)                                    # this matrix needs the reorder
    later = jax.jit(sort_eigenpairs)(w, v)
    assert np.array_equal(_bits(now[0]), _bits(later[0]))
    assert np.array_equal(_bits(now[1]), _bits(later[1]))


@pytest.fixture
def standin(mesh, monkeypatch):
    """The cuSOLVERMp backend with jnp.linalg.eigh behind it (raw V^H buffer), on CPU."""
    import sys
    from distrib_la import resolve as R, _cusolvermp as C
    PL = sys.modules["distrib_la.plan"]
    state = dict(corrupt=False)
    monkeypatch.setattr(R, "mesh_platform", lambda m: "CUDA")
    monkeypatch.setattr(PL, "mesh_platform", lambda m: "CUDA")
    monkeypatch.setattr(C.loader, "get_lib", lambda *a, **k: None)
    monkeypatch.setattr(R.loader, "probe_target", lambda *a, **k: (True, ""))
    monkeypatch.setattr(C, "context_key", lambda *a, **k: 0)

    def eigh(A, *, mesh, compute_evecs=True, block=None, side=None):
        w, v = jnp.linalg.eigh(A)
        raw = jnp.conj(v.T) * (1.001 if state["corrupt"] and block is None else 1)
        return w, jax.lax.with_sharding_constraint(raw, NamedSharding(mesh, P("x", "y")))
    monkeypatch.setattr(C, "distributed_eigh", eigh)
    count = jax.process_count
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    import distrib_la as Dl
    plans = {n: Dl.plan("eigh", mesh, backend="cusolvermp", n=n) for n in (64, 34)}
    monkeypatch.setattr(jax, "process_count", count)

    def fresh(corrupt=False):                 # corruption is decided at trace time
        PL._SCAN_CACHE.clear(); C._JIT_CACHE.clear()
        state["corrupt"] = corrupt
    return plans, fresh


@pytest.mark.parametrize("n", [64, 34])
def test_eager_traced_retried_and_donated_routes_agree_bitwise(mesh, standin, n, capfd):
    import distrib_la as Dl
    plans, fresh = standin
    p = plans[n]
    face, rep = NamedSharding(mesh, P("x", "y")), NamedSharding(mesh, P())
    a_host = _padded_indefinite(n, n - n // 5)
    a = jax.device_put(a_host, face)
    fresh(); ref = p(a)
    w = np.asarray(ref[0])
    assert np.all(np.diff(w) >= 0) and np.sum(w == 0) == n // 5
    v = np.asarray(ref[1])
    assert np.linalg.norm(a_host @ v - v * w) < 1e-12 * np.linalg.norm(a_host)
    fresh(); traced = jax.jit(lambda x: p(x), out_shardings=(rep, face))(a)
    fresh(); program = Dl.checked_program(lambda x: p(x), mesh, (rep, face))(a)
    fresh(corrupt=True); retried = p(a)
    assert "solving again in a separate program" in capfd.readouterr().err
    fresh(); b = jax.device_put(a_host, face); donated = p(b, donate=True)
    assert b.is_deleted()
    from distrib_la._cusolvermp import retry_block
    # The retry is the same solve at a smaller block where one exists (the stand-in ignores
    # the block: bitwise the first attempt), else the shifted attempt (its own bits).
    same = [traced, program, donated] + ([retried] if retry_block(n, 2) is not None else [])
    for other in same:
        assert np.array_equal(_bits(ref[0]), _bits(other[0]))
        assert np.array_equal(_bits(ref[1]), _bits(other[1]))
    wr, vr = np.asarray(retried[0]), np.asarray(retried[1])
    assert np.all(np.diff(wr) >= 0)
    assert np.linalg.norm(a_host @ vr - vr * wr) < 1e-12 * np.linalg.norm(a_host)
    fresh(corrupt=True); b = jax.device_put(a_host, face)
    with pytest.raises(ValueError, match="no attempt passed"):
        p(b, donate=True)


def test_donate_is_refused_off_the_library_eigh(mesh):
    import distrib_la as Dl
    with pytest.raises(ValueError, match="donate=True"):
        Dl.plan("eigh", mesh, backend="native")(jnp.eye(8), donate=True)


def test_cusolvermp_wrapper_aliases_its_operand(mesh, monkeypatch):
    from distrib_la import _cusolvermp as C
    monkeypatch.setattr(C.loader, "get_lib", lambda *a, **k: None)
    monkeypatch.setattr(C, "context_key", lambda *a, **k: 0)
    monkeypatch.setattr(C, "_validate_mesh", lambda m: (2, 2))
    a = jax.ShapeDtypeStruct((512, 512), jnp.complex128, sharding=NamedSharding(mesh, P("x", "y")))
    try:
        C.distributed_eigh(a, mesh=mesh)            # caches the jitted wrapper, then fails (abstract)
    except Exception:
        pass
    text = list(C._JIT_CACHE.values())[-1].lower(a).as_text()
    assert "lorrax_cusolvermp_eigh" in text and "output_operand_aliases" in text
