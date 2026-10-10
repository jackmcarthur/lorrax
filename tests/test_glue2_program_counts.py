"""Programs keyed by shapes, not by values (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
Each case counts JAX's own lowering event over repeated calls at one shape.
"""
import numpy as np
import pytest
import jax
import jax.monitoring
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

_EVENTS = []
jax.monitoring.register_event_duration_secs_listener(lambda event, *_, **__: _EVENTS.append(event))


def _lowerings():
    return sum(e == "/jax/core/compile/jaxpr_to_mlir_module_duration" for e in _EVENTS)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _dirac_plan(mesh, nk, mu):
    """An identity-symmetry four-spinor plan: every k is its own parent."""
    from common.grouped_layout import build_square_grouped_shard_layout
    from gw.centroid_k_unfold import CentroidKUnfoldPlan
    layout = build_square_grouped_shard_layout(np.arange(mu), (2, 2))
    n_pad = int(layout.axis.n_padded)
    return CentroidKUnfoldPlan(
        mesh_xy=mesh, layout=layout, irr_idx=np.arange(nk, dtype=np.int32),
        sym_idx=np.zeros(nk, np.int32),
        sym_perm=layout.axis.pack_permutations_host(np.arange(mu, dtype=np.int32)[None]),
        L_table=np.zeros((1, n_pad, 3), np.int64),
        k_parent_frac=np.stack([np.arange(nk) / nk, np.zeros(nk), np.zeros(nk)], axis=1),
        spin_action_full=np.broadcast_to(np.eye(4, dtype=np.complex128), (nk, 4, 4)).copy(),
        n_sym_spatial=1, nspinor=4)


def test_dirac_cct_gamma_pairs_share_one_program():
    from distrib_la import gemm_plan
    from isdf.core import c_q_from_psi_sm
    mesh = _mesh()
    nk, mu, nb = 2, 8, 4
    plan = _dirac_plan(mesh, nk, mu)
    mu = int(plan.n_centroid_packed)
    gemm = gemm_plan(mesh, m=2 * mu, k=nb, n=2 * mu, nq=nk, dtype=jnp.complex128,
                     layout="face", warmup=False)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(nk, 4, mu, nb)) + 1j * rng.normal(size=(nk, 4, mu, nb))
    pm = jax.device_put(psi, NamedSharding(mesh, P(None, None, "x", "y")))
    pn = jax.device_put(np.conj(np.transpose(psi, (0, 3, 1, 2))),
                        NamedSharding(mesh, P(None, "x", None, "y")))
    w = np.ones(nb)
    call = lambda i, j: c_q_from_psi_sm(
        pm, pn, w, w, k_unfold_plan=plan, kgrid=(nk, 1, 1), mesh_xy=mesh, gemm=gemm,
        gamma_L=i, gamma_R=j)
    first = call(0, 0)
    before = _lowerings()
    out = {(i, j): np.asarray(call(i, j)) for i, j in ((0, 0), (1, 1), (2, 2), (3, 3), (1, 3), (3, 1))}
    assert _lowerings() == before
    # gamma^0 is the identity: its Gram is Hermitian per k; every vertex pair differs.
    c00 = out[(0, 0)]
    assert np.array_equal(c00, np.asarray(first))
    assert np.allclose(c00, np.conj(np.swapaxes(c00, -1, -2)), atol=1e-12)
    assert len({v.tobytes() for v in out.values()}) == len(out)


@pytest.mark.parametrize("route", ["synthesis", "kconv_tables"])
def test_window_runner_branches_share_one_program(route):
    """Both W branches run one window program: the branch's tables arrive as operands, from
    the synthesis's window operands (scalar route) or from ``kconv_tables(space)`` (sectors)."""
    from gw.mpa.sigma import SynthesisTau, WSynthesis
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    mesh = _mesh()
    face = NamedSharding(mesh, P(None, "x", "y"))
    rep = NamedSharding(mesh, P())
    tables = {s: jax.device_put(np.full(3, v), rep) for s, v in (("cond", 1.0), ("val", 2.0))}
    by_synthesis = route == "synthesis"
    synthesis = WSynthesis(lambda load, _ref, _t: load,
                           lambda space, _i, _b: (tables[space] if by_synthesis else tables["cond"],),
                           lambda: (), lambda _r=None: None, 0, ("glue2-toy", route))
    spatial = lambda xn, yr, xr, yn, energies, weight, e_ref, t, w, *k: xn * ((k[0] if k else w)[0] * t)
    xn = jax.device_put(np.ones((2, 4, 4), np.complex128), face)
    tau = SynthesisTau(spatial, synthesis, None, None, 0, "glue2", None, ("glue2-toy", route), (),
                       kconv_tables=None if by_synthesis else tables.__getitem__)
    acc = DeviceOmegaAccumulator(np.arange(3.0), shape=(3, 2, 4, 4),
                                 sharding=NamedSharding(mesh, P(None, None, "x", "y")), omega_axis=0)
    count, zero = jax.device_put((np.int32(1), np.float64(0.0)), rep)

    def window(space, **kw):
        args = tau.window_arguments(xn, xn, None, None, zero, zero, space, None, None)
        return acc.integrate_window(tau.window_kernel(), args, np.ones(2), np.ones(2),
                                    n_active=2, active_count=count, capacity=2, omega_sign=1.0,
                                    prefactor=1.0, **kw)

    window("cond", compile_only=True)
    before = _lowerings()
    for space in ("cond", "val", "cond", "val"):
        window(space)
    assert _lowerings() == before
    # Two nodes per window, two windows per branch, each node's sigma(t) = the branch table.
    total = np.asarray(acc.finalize())
    coeff = np.exp(1j * np.arange(3.0))[:, None, None, None]
    expect = 4 * coeff * (1.0 + 2.0) * np.ones((3, 2, 4, 4))
    assert np.allclose(total, expect, rtol=1e-12)


def test_accumulator_zeros_are_native_and_lower_once():
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    mesh = _mesh()
    sh = NamedSharding(mesh, P(None, None, "x", "y"))
    make = lambda: DeviceOmegaAccumulator(np.arange(3.0), shape=(3, 2, 4, 4), sharding=sh,
                                          omega_axis=0).finalize()
    first = make()
    before = _lowerings()
    again = make()
    assert _lowerings() == before
    assert again.sharding == sh == first.sharding and not np.any(np.asarray(again))


def test_host_tables_meet_sharded_operands_without_a_reshard(monkeypatch):
    """Host-born masks and (k, band) tables stay NumPy, so no single-device copy is
    resliced onto the mesh (``ArrayImpl._multi_slice``) when they meet a sharded operand."""
    from jax._src import array as _array
    from file_io.parallel_transport import head_velocity_set
    from gw.qsgw_head import _pad_head_band_manifold
    mesh = _mesh()
    calls = []
    original = _array.ArrayImpl._multi_slice
    monkeypatch.setattr(_array.ArrayImpl, "_multi_slice",
                        lambda self, *a, **k: calls.append(self.shape) or original(self, *a, **k))
    occ = jax.device_put(np.linspace(0.0, 1.0, 2 * 6).reshape(2, 6), NamedSharding(mesh, P(None, "x")))
    mask = head_velocity_set(occ)
    v = np.ones((3, 2, 6, 6), np.complex128)
    e = np.zeros((2, 6))
    v, e2, f2, s2 = _pad_head_band_manifold(v, e, e, e, mesh=mesh)
    assert all(isinstance(a, np.ndarray) and a.shape == (2, int(v.shape[-1])) for a in (e2, f2, s2))
    assert calls == []
    assert mask.shape == (2, 6, 6) and bool(mask[0, 0, 0]) is False
