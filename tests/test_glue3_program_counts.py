"""Programs keyed by shapes, not by context or branch (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import ast
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.monitoring
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
_EVENTS = []
jax.monitoring.register_event_duration_secs_listener(lambda event, *_, **__: _EVENTS.append(event))


def _lowerings():
    return sum(e == "/jax/core/compile/jaxpr_to_mlir_module_duration" for e in _EVENTS)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def mesh_contexts(source):
    """Line numbers of ``with <mesh>:`` items (a bare name or attribute ``mesh``/``mesh_xy``)."""
    def named(e):
        return (e.id if isinstance(e, ast.Name) else e.attr if isinstance(e, ast.Attribute)
                else None) in ("mesh", "mesh_xy")
    return [n.lineno for n in ast.walk(ast.parse(source))
            if isinstance(n, (ast.With, ast.AsyncWith)) and any(named(i.context_expr) for i in n.items)]


def test_src_enters_no_mesh_context():
    """Every sharding in src/ is explicit, so a legacy mesh context only adds the mesh
    stack to every jit's trace key: a nested one re-traces and re-lowers identical programs."""
    roots = [ROOT / "src", *sorted((ROOT / "services").glob("*/src"))]
    found = [f"{path.relative_to(ROOT)}:{line}" for root in roots for path in sorted(root.rglob("*.py"))
             for line in mesh_contexts(path.read_text())]
    assert found == []


def test_mesh_context_scanner_can_fail():
    assert mesh_contexts("with mesh_xy:\n    with self.mesh, open(p):\n        pass\n") == [1, 2]
    assert mesh_contexts("with SlabIO(p, mesh=mesh) as io:\n    pass\n") == []


def test_mesh_context_is_part_of_the_program_key():
    mesh = _mesh()
    f = jax.jit(lambda x: x + 1)
    x = jax.device_put(np.ones(4), NamedSharding(mesh, P("x")))
    f(x)
    before = _lowerings()
    f(x)
    assert _lowerings() == before
    with mesh:
        f(x)
    assert _lowerings() == before + 1


def test_one_lorentz_kconv_serves_both_w_branches():
    """The hole tables are the particle tables read at -q: the same shapes and structural
    zeros, so a convolution built from the particle tables reads the hole load bit for bit
    as one built from the hole tables (the sector route's one τ program)."""
    from tests.test_kconv_xla_gate import _tables
    from ffi import fft as F
    from symmetry_maps import device_load_tables
    from gw.mpa.sector_sigma import hole_tables
    mesh = _mesh()
    rng = np.random.default_rng(3)
    kg, n_parent, m, ns = (4, 2, 1), 3, 2, 2
    nk, mu = int(np.prod(kg)), 2 * m
    vl = (((0, 1), (1, 1j)), ((1, 0), (-1, 1j)))
    vr = (((1, 0), (1j, -1)), ((0, 1), (1, -1j)))
    tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (2, 2))
    particle = _tables(rng, nk, n_parent, m, m, len(vl), len(vl), len(vr), (2, 2))
    hole = hole_tables(particle, kg)
    face = NamedSharding(mesh, P(None, "x", None, "y", None))
    rnd = lambda *s: jax.device_put(rng.standard_normal(s) + 1j * rng.standard_normal(s), face)
    G, Gt = rnd(n_parent, mu, ns, mu, ns), rnd(n_parent, mu, ns, mu, ns)
    W, Wt = rnd(n_parent, mu, len(vl), mu, len(vr)), rnd(n_parent, mu, len(vl), mu, len(vr))
    load, w_load = device_load_tables(tg, mesh), device_load_tables(hole, mesh)
    built = [F.make_kconv_lorentz_unfold(mesh, kg, tg, w_tables=t, left_vertices=vl, right_vertices=vr,
                                         store_rows=[0, 3, 5], norm="ortho", mult=0.7)
             for t in (particle, hole)]
    a, b = (np.asarray(f(G, Gt, W, Wt, load=load, w_load=w_load)) for f in built)
    assert np.array_equal(a, b) and np.any(a)


def test_head_tables_from_the_host_need_no_reshard(monkeypatch):
    """Host (k, band) tables reach the S(ω) kernel as NumPy: no single-device copy is resliced
    onto the mesh (``ArrayImpl._multi_slice``), and the tensor is the device-table one bit for bit."""
    from jax._src import array as _array
    from gw.qsgw_head import _pad_head_band_manifold, head_s_tensor_sharded
    mesh = _mesh()
    rng = np.random.default_rng(5)
    nk, nb = 2, 6
    v = rng.standard_normal((3, nk, nb, nb)) + 1j * rng.standard_normal((3, nk, nb, nb))
    v = v + np.conj(np.swapaxes(v, -1, -2))
    e = np.sort(rng.standard_normal((nk, nb)), axis=1)
    f = (e < 0).astype(np.float64)
    run = lambda e, f: np.asarray(head_s_tensor_sharded(
        v, e, f, np.asarray([0.1j, 0.3 + 0.2j]), mesh=mesh, nb_logical=nb - 1, cell_volume=10.0,
        nk_tot=nk, nspin=1, nspinor=2))
    device = run(jnp.asarray(e), jnp.asarray(f))
    calls = []
    original = _array.ArrayImpl._multi_slice
    monkeypatch.setattr(_array.ArrayImpl, "_multi_slice",
                        lambda self, *a, **k: calls.append(self.shape) or original(self, *a, **k))
    assert np.array_equal(run(e, f), device)
    assert calls == []
    # A table narrower than v's band axis (the logical manifold) pads to the carrier.
    _, e2, _, _ = _pad_head_band_manifold(np.pad(v, ((0, 0), (0, 0), (0, 2), (0, 2))), e, e, e, mesh=mesh)
    assert isinstance(e2, np.ndarray) and np.array_equal(e2[:, :nb], e) and not np.any(e2[:, nb:])

