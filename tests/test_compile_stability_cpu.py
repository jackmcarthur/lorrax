"""Compile-stability guards on toy inputs (CPU only, seconds).

A program JAX's persistent cache cannot store, or one that compiles again
for the same work, costs minutes per process at production scale and shows
in no result (compile audit 2026-10-05). Each test fails on the defect's
shape, not on a timing:

0. ``RANK_FINGERPRINT_ENV`` mirrors ``ffi.FFI_DIAL_ENV``.
1. Checked solves leave no host callback in the programs that hold them
   (``compiler._cache_write`` never stores a program with one), and the
   traced fallback outside ``checked_program`` warns once.
2. A second process reuses a checked program from the persistent cache.
3. One plan on one stack from two call sites compiles once.
4. Partner directions keep one carrier whatever their count (e2200b174).
5. Face constructor rounds share one width, the ragged tail included.
"""
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]


def _mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _host_callbacks(lowered):
    return len(lowered._lowering.compile_args.get("host_callbacks", ()))


def _backend_compiles():
    """A list that grows by one name per XLA backend compile from now on."""
    from jax._src import dispatch, monitoring
    names = []
    monitoring.register_event_duration_secs_listener(
        lambda event, secs, **kw: names.append(kw.get("fun_name", "?"))
        if event == dispatch.BACKEND_COMPILE_EVENT else None)
    return names


def _tuple_of(path, name):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in {path}")


def test_rank_fingerprint_env_mirrors_ffi_dials():
    assert (_tuple_of(ROOT / "src/common/jax_compile_cache.py", "RANK_FINGERPRINT_ENV")
            == _tuple_of(ROOT / "src/ffi/__init__.py", "FFI_DIAL_ENV"))


def _checked_eigh(a, site="test eigh"):
    from distrib_la import _result_check as rc
    (w, v), status = rc.checked_eigh((jnp.linalg.eigh,), a)
    rc.raise_if_failed(status, "eigh", a.shape[-1], a.dtype, site)
    return w, v


def _hermitian(n, b=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n, n) if b is None else (b, n, n)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return jnp.asarray(a + np.conj(np.swapaxes(a, -1, -2)) + 4 * n * np.eye(n))


def test_checked_program_has_no_host_callback():
    from distrib_la import checked_program
    mesh = _mesh()
    a = _hermitian(8)
    call = checked_program(lambda x: _checked_eigh(x)[0], mesh, NamedSharding(mesh, P()))
    assert _host_callbacks(call.lower(a)) == 0
    np.testing.assert_allclose(np.asarray(call(a)), np.linalg.eigvalsh(np.asarray(a)), atol=1e-10)
    # Outside checked_program the status can only reach the host by a callback:
    # that program is never stored (the fallback warns: next test).
    bare = jax.jit(lambda x: _checked_eigh(x)[0])
    assert _host_callbacks(bare.lower(a)) == 1


def test_traced_fallback_warns_once(capsys):
    a = _hermitian(8)
    for _ in range(2):
        jax.jit(lambda x: _checked_eigh(x, "fallback warning")[0]).lower(a)
    assert capsys.readouterr().err.count("UNCACHEABLE") == 1


class _CheckedPlan:
    """A distrib_la plan whose eigh and LU solve are checked as the FFI ones are."""

    def __init__(self, op, *args, **kwargs):
        self.op = op

    def describe(self):
        return "checked test plan"

    def batched(self, A, *rest, **kwargs):
        from distrib_la import _result_check as rc
        if self.op == "eigh":
            return _checked_eigh(A)
        B, = rest
        sketch = (*rc.matrix_sketch(A), *rc.rhs_sketch(B))
        X, status = rc.checked("solve_lu", (lambda x: x,), lambda x: rc.solve_errors(sketch, x),
                               (jnp.linalg.solve(A, B),), n=A.shape[-1],
                               dtype=A.dtype)
        rc.raise_if_failed(status, "solve_lu", A.shape[-1], A.dtype, "test solve")
        return X


def _matmul(a, b, *, transa="N", transb="N", **kwargs):
    op = {"N": lambda x: x, "T": lambda x: jnp.swapaxes(x, -1, -2),
          "C": lambda x: jnp.conj(jnp.swapaxes(x, -1, -2))}
    return op[transa](a) @ op[transb](b)


@pytest.fixture
def checked_service(monkeypatch):
    import distrib_la
    monkeypatch.setattr(distrib_la, "plan", _CheckedPlan)
    monkeypatch.setattr(distrib_la, "matmul", _matmul)


def test_response_programs_have_no_host_callback(checked_service):
    from gw import response_bank as rb
    mesh, n, nq = _mesh(), 8, 2
    face = jax.ShapeDtypeStruct((nq, n, n), jnp.complex128, sharding=NamedSharding(mesh, P(None, "x", "y")))
    dyson, slope, moments, _ = rb._response_programs.__wrapped__(
        mesh, n, "distributed", "auto", 0.5, False, None)
    sqrt_v = rb._coulomb_algebra.__wrapped__(mesh, n, n, "distributed")
    counts = {"dyson.value": _host_callbacks(dyson.value.lower(face, face)),
              "slope": _host_callbacks(slope.lower(face, face, face)),
              "moments": _host_callbacks(moments.lower(face, face, face)),
              "sqrt_v": _host_callbacks(sqrt_v.lower(face))}
    assert counts == dict.fromkeys(counts, 0)


_REUSE = textwrap.dedent("""
    import sys
    import jax, jax.numpy as jnp, numpy as np
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", sys.argv[1])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    from jax._src import compiler, monitoring
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    from distrib_la import checked_program
    from distrib_la import _result_check as rc
    real, hits = [], []
    original = compiler.backend_compile_and_load
    def counting(*args, **kwargs):
        real.append(1)
        return original(*args, **kwargs)
    compiler.backend_compile_and_load = counting
    monitoring.register_event_listener(
        lambda name, **kw: hits.append(1) if name == "/jax/compilation_cache/cache_hits" else None)
    def fn(a):
        (w, v), status = rc.checked_eigh((jnp.linalg.eigh,), a)
        rc.raise_if_failed(status, "eigh", a.shape[-1], a.dtype, "reuse")
        return w
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    a = jnp.asarray(np.eye(16) + 0.01)
    call = checked_program(fn, mesh, NamedSharding(mesh, P()))
    jax.block_until_ready(call(a))
    print("REUSE", len(real), len(hits))
""")


def _reuse(script, cache):
    out = subprocess.run([sys.executable, str(script), str(cache)], capture_output=True, text=True,
                         env=dict(os.environ, JAX_PLATFORMS="cpu"), timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("REUSE")][-1]
    return tuple(int(v) for v in line.split()[1:])


def test_checked_program_is_reused_by_a_second_process(tmp_path):
    script = tmp_path / "reuse.py"
    script.write_text(_REUSE)
    first = _reuse(script, tmp_path / "cache")
    second = _reuse(script, tmp_path / "cache")
    assert first[0] >= 1, first
    assert second == (0, second[1]) and second[1] >= 1, (first, second)


def test_one_stack_from_two_call_sites_compiles_once():
    from distrib_la import plan
    mesh = _mesh()
    p = plan("eigh", mesh, backend="auto", n=8, batched_route="batch_reshard")
    a = jax.device_put(_hermitian(8, b=2), NamedSharding(mesh, P(None, "x", "y")))

    def site_one(x):
        return p.batched(x)

    def site_two(x):
        return p.batched(x)
    jax.block_until_ready(site_one(a))
    names = _backend_compiles()
    jax.block_until_ready(site_two(a))
    assert names == [], names


def test_partner_directions_keep_one_carrier():
    """O's part outside span(Q) of rank 0, 2 and 6 comes back on Q's own carrier."""
    import distrib_la
    from gw.shared_pole_directions import _round_kernels, _round_partner_directions, port_extent
    mesh = _mesh()
    n, r = 16, 8
    eigh = distrib_la.plan("eigh", mesh, backend="off", n=n)
    kernels = _round_kernels(mesh, "face")
    basis = np.linalg.qr(np.random.default_rng(1).standard_normal((n, n)))[0]
    face = NamedSharding(mesh, P(None, "x", "y"))
    q = jax.device_put(jnp.asarray(basis[None, :, :r], dtype=jnp.complex128), face)
    shapes, counts = set(), []
    for extra in (0, 2, 6):
        o = basis[None, :, :r] @ np.diag(np.arange(1.0, r + 1))
        o[0, :, :extra] += basis[:, r:r + extra]
        directions, width = _round_partner_directions(
            q, jax.device_put(jnp.asarray(o, dtype=jnp.complex128), face), 1e-6, real=1, kernels=kernels,
            eigh_plan=eigh, column_extent=port_extent(mesh), tol=1e-6)
        shapes.add(tuple(directions.shape))
        counts.append(width[0])
    assert counts == [0, 2, 6]
    assert shapes == {tuple(q.shape)}


@pytest.mark.xfail(strict=True, reason="audit row 3: the face schedule's last round is ragged; "
                   "flip when row 3 lands")
def test_face_rounds_share_one_width():
    from types import SimpleNamespace
    from gw.shared_pole_execution import sector_round_schedule
    config = SimpleNamespace(backend=SimpleNamespace(linalg="distributed"))
    rounds = sector_round_schedule(None, {"n_q_irr": 7}, None, config, _mesh(),
                                   execution="face", batch_width=3)
    assert [real for _, real, _, _ in rounds] == [3, 3, 1]
    assert {len(ids) for ids, _, _, _ in rounds} == {3}
    assert all(np.array_equal(slots, np.arange(3)) for _, _, slots, _ in rounds)
    assert sorted({q for ids, real, _, _ in rounds for q in ids[:real]}) == list(range(7))
