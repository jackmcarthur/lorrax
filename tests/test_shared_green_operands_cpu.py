"""Shared response/Σ placement: actual AOT admission, parity and payload lifetime."""
import gc
from pathlib import Path
import runpy
import weakref

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('side,projection', [(2, False), (3, True)])
def test_admission_execution_and_no_payload_cache(monkeypatch, side, projection):
    from gw import subtile_stream as S
    from gw.greens_function_kernel import build_G_parents
    from distrib_la import gemm_plan
    from ffi import fft as F
    from lxkit import memory
    from runtime.aot_memory import compiled_new_bytes
    fixture = runpy.run_path(str(Path(__file__).with_name(
        "test_minimax_response_rows_cpu.py")))["fixture"]

    if jax.device_count() < side * side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side * side]).reshape(side, side), ('x', 'y'))
    grid, plan, psi, e, _, _ = fixture(mesh, 2, True)
    # A substantial sum-band carrier makes the two retained placement sizes
    # distinct, while the ragged μ geometry and zero carrier tails stay real.
    psi = np.tile(psi, (1, 1, 1, 10))
    np_, ns, mu, nb = psi.shape
    put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))
    left = put(psi, P(None, None, 'x', 'y'))
    right = put(psi.transpose(0, 3, 1, 2), P(None, 'x', None, 'y'))
    faces = (left, right)
    if projection:
        faces += (put(psi[..., :6].transpose(0, 3, 1, 2), P(None, 'y', None, 'x')),
                  put(psi[..., :6], P(None, None, 'x', 'y')))
    phase = np.arange(np_ * nb).reshape(np_, nb)
    weight_host = np.cos(phase) + .1j * np.sin(phase)
    weights = put(weight_host, P())
    monkeypatch.setattr(memory, 'room', lambda: 1 << 40)
    compiled = {}
    needs = {}

    def build(layout, abstract):
        gm = gemm_plan(mesh, m=mu * ns, k=nb, n=mu * ns, nq=np_,
                       dtype=jnp.complex128, layout=layout, warmup=False)

        @jax.jit
        def product(prepared, weights):
            l, r, *projectors = prepared
            if layout == 'axis':
                l = S.window_green_rows(l, mesh)
            value = build_G_parents(l, None, phases=weights, layout=layout,
                gemm=gm, k_unfold_plan=plan, real_weights=False, right=r)
            # Both antiunitary operands matter. Optional projection faces are
            # returned too, so their exact placement is tested independently.
            return value.G, value.transpose, tuple(projectors)

        ex = product.lower(abstract, weights).compile()
        compiled[layout] = ex
        place = S.green_operand_program(mesh, layout).lower(*faces).compile()
        ma = place.memory_analysis()
        needs[layout] = max(compiled_new_bytes(place),
            max(0, ma.output_size_in_bytes - ma.alias_size_in_bytes)
            + compiled_new_bytes(ex))
        return ex, [ex], 0

    expected = np.einsum('pamb,pb,pcnb->pmanc', psi, weight_host, psi.conj())
    expected_partner = np.einsum('pamb,pb,pcnb->pmanc', psi.conj(), weight_host, psi)
    with F.xla_reference():
        for layout in ('axis', 'face'):
            chosen, prepared, ex = S.prepare_green_operands(
                faces, mesh, build, label='test Green', _force_layout=layout)
            assert chosen == layout
            got, partner, projectors = ex(prepared, weights)
            np.testing.assert_allclose(got, expected, rtol=3e-12, atol=2e-11)
            np.testing.assert_allclose(partner, expected_partner, rtol=3e-12, atol=2e-11)
            for actual, original in zip(projectors, faces[2:]):
                np.testing.assert_array_equal(actual, original)
            # A cached placement/executable must not retain this call's banks.
            refs = [weakref.ref(a) for a in prepared if not any(a is f for f in faces)]
            if projection:
                del actual, original
            del prepared, got, partner, projectors
            gc.collect()
            assert all(ref() is None for ref in refs)

        assert needs['axis'] > needs['face'], needs
        room = (needs['axis'] + needs['face']) // 2
        monkeypatch.setattr(memory, 'room', lambda: room)
        chosen, prepared, ex = S.prepare_green_operands(faces, mesh, build, label='test Green')
        assert chosen == 'face' and needs[chosen] <= room
        np.testing.assert_allclose(ex(prepared, weights)[0], expected, rtol=3e-12, atol=2e-11)
        del prepared

        # A compile-only call must not execute even the selected placement.
        original = S.green_operand_program
        executed = []

        class Program:
            def __init__(self, inner):
                self.inner = inner
            def lower(self, *args):
                lowered = self.inner.lower(*args)
                class Lowered:
                    def compile(self):
                        inner = lowered.compile()
                        class Executable:
                            def __getattr__(self, name):
                                return getattr(inner, name)
                            def __call__(self, *args):
                                executed.append(True)
                                return inner(*args)
                        return Executable()
                return Lowered()

        monkeypatch.setattr(S, 'green_operand_program', lambda mesh, layout: Program(original(mesh, layout)))
        chosen, prepared, ex = S.prepare_green_operands(
            faces, mesh, build, label='test Green', compile_only=True)
        assert chosen == 'face' and prepared is None and not executed


def test_response_wrapper_uses_prepared_operands(monkeypatch):
    from types import SimpleNamespace
    from ffi import fft as F
    from gw import subtile_stream as S, w_isdf as W
    from gw.minimax_screening import MinimaxNodes
    from lxkit import memory
    helpers = runpy.run_path(str(Path(__file__).with_name('test_minimax_response_rows_cpu.py')))
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    grid, plan, psi, e, v, c = helpers['fixture'](mesh, 2, True)
    np_, ns, mu, nb = psi.shape
    put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))
    pm = put(psi, P(None, None, 'x', 'y'))
    pn = put(psi.transpose(0, 3, 1, 2), P(None, 'x', None, 'y'))
    parent = SimpleNamespace(psi_mun=pm, psi_nmu=pn, layout='face', plan=plan)
    wfns = SimpleNamespace(layout='face', psi_mun=None, green_parent=parent,
                           slices=SimpleNamespace(nb_full=nb))
    meta = SimpleNamespace(nkx=grid[0], nky=grid[1], nkz=grid[2])
    t = np.array([.1, .4], complex)
    alpha = np.array([-.3+.2j, .12-.03j])
    nodes = MinimaxNodes(jnp.asarray(t), jnp.asarray(alpha))
    args = (nodes, pm, pn, put(v, P()), put(c, P()), put(e, P()),
            jnp.asarray(e[v].max()), jnp.asarray(e[c].min()))
    expected = helpers['numpy_response'](grid, plan, psi, e, v, c, t, alpha[None], False, True)[0]
    original = S.prepare_green_operands
    monkeypatch.setattr(memory, 'room', lambda: 1 << 40)
    with F.xla_reference():
        for layout in ('axis', 'face'):
            def prepare(*args, **kwargs):
                return original(*args, **kwargs, _force_layout=layout)
            monkeypatch.setattr(S, 'prepare_green_operands', prepare)
            assert W._run_minimax_chi(wfns, meta, mesh, args,
                complex_contour=True, compile_only=True) is None
            got = W._run_minimax_chi(wfns, meta, mesh, args, complex_contour=True)
            np.testing.assert_allclose(got, expected, rtol=5e-12, atol=3e-11)
