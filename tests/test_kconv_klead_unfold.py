"""The Σ unfold convolution door against the chain it replaced, on real symmetry plans.

``ffi.fft.make_kconv_klead_unfold(G, Gt, W_prep)`` reads the raw-parent Green
and must equal, bit for bit, the old Σ chain: the typed unfold
(``plan.unfold_operator``), the spin action, ``sigma_conv_operand`` and the
k-leading door ``make_kconv_klead``.  Plans: the order-two glide group with
spin mixing and an antiunitary row (ns = 2, 4) and A-cubic (48 operations,
ns = 1).  The table composition alone (``symmetry_maps.
apply_unfold_load_tables_local``) must equal the unfold too.  Red twin: the
right source table rolled by one slot must miss.  ``unfold_case`` is reused
on GPU by ``tests/multi_device/kconv_router_p4.py`` (mathdx mode 7).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import zeta_mubatch_fixtures as fixtures


def unfold_case(mesh, fx, seed=0):
    """(bitwise door == old chain, bitwise tables == unfold, rel error of the red twin, max|Δ|)."""
    from common.shard_map import shard_map
    from ffi import fft as F
    from gw.wavefunction_bundle import SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC, sigma_conv_operand
    from symmetry_maps import apply_unfold_load_tables_local, local_unfold_load_tables
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    ns, mu, n_par, nk = int(plan.nspinor), int(plan.n_centroid_packed), int(plan.n_parent), int(plan.n_full)
    rng = np.random.default_rng(seed + 17 * ns)
    crand = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    sw = NamedSharding(mesh, P(None, "x", "y"))
    G, Gt, W = crand(n_par, mu, ns, mu, ns), crand(n_par, mu, ns, mu, ns), crand(nk, mu, mu)
    Gd, Gtd = fixtures._put(G, s5), fixtures._put(Gt, s5)
    anti = bool(np.any(np.asarray(plan.sym_idx) >= plan.n_sym_spatial))
    mult = -1.0 / np.sqrt(float(nk))
    kconv = F.make_kconv_klead(mesh, kg, SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC, norm="ortho", mult=mult)
    Wp = kconv.prep(fixtures._put(W, sw))
    Gk = plan.unfold_operator(Gd, operator_transpose=Gtd if anti else None)
    ref = fixtures._host(kconv.apply(sigma_conv_operand(Gk), Wp))
    tables = plan.unfold_load_tables()
    door = F.make_kconv_klead_unfold(mesh, kg, tables, norm="ortho", mult=mult)
    got = fixtures._host(door(Gd, Gtd if anti else None, Wp))

    @partial(shard_map, mesh=mesh, in_specs=(s5.spec, s5.spec), out_specs=s5.spec,
             check_vma=False)
    def composed(g, gt):
        flat = lambda a: a.reshape(a.shape[0], a.shape[1] * ns, a.shape[3] * ns)
        return apply_unfold_load_tables_local(flat(g), flat(gt if anti else g),
                                              local_unfold_load_tables(tables), tables.spin)
    O = fixtures._host(jax.jit(composed)(Gd, Gtd))
    red_tables = tables._replace(rsrc=np.roll(tables.rsrc, 1, axis=1))
    red_door = F.make_kconv_klead_unfold(mesh, kg, red_tables, norm="ortho", mult=mult)
    red = fixtures._host(red_door(Gd, Gtd if anti else None, Wp))
    return dict(ns=ns, nk=nk, n_parent=n_par, mu=mu, antiunitary=anti,
                door_bitwise=bool(np.array_equal(got, ref)),
                max_abs=float(np.max(np.abs(got - ref))),
                rel=float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))),
                tables_bitwise=bool(np.array_equal(O, fixtures._host(Gk))),
                red_rel=float(np.max(np.abs(red - ref)) / np.max(np.abs(ref))))


def _mesh():
    from lxkit.testing import require_devices
    require_devices(4)
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_unfold_door_matches_old_sigma_chain_bitwise():
    """cpu leg on the glide plans (ns 2, 4) and A-cubic (ns 1): bitwise; red twin fires."""
    from ffi import fft as F
    mesh = _mesh()
    assert F.kconv_backend(mesh) == "plan"
    rng = np.random.default_rng(3)
    cases = [fixtures._glide_fixture(mesh, rng, ns) for ns in (2, 4)]
    cases.append(fixtures._acubic_fixture(mesh, rng))
    for fx in cases:
        r = unfold_case(mesh, fx)
        assert r["tables_bitwise"], r
        assert r["door_bitwise"], r
        assert r["red_rel"] > 1e-3, r
