"""The Σ unfold convolution door against the chain it replaced, on real symmetry plans.

``ffi.fft.make_kconv_klead_unfold(G, Gt, W_prep)`` reads the raw-parent Green
and must equal, bit for bit, the old Σ chain: the typed unfold
(``plan.unfold_operator``), the spin action, ``sigma_conv_operand`` and the
k-leading door ``make_kconv_klead``.  Plans: the order-two glide group with
spin mixing and an antiunitary row (ns = 2, 4), A-cubic (48 operations,
ns = 1), and a C3 group on a 3x3 k grid whose spin action has general complex
entries and whose umklapp phases are general (q = n/3), so no product has a
zero term and every FMA spelling is distinguishable (ns = 2, 4).  The table composition alone (``symmetry_maps.
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


def c3_fixture(mesh, ns, kgrid=(3, 3, 1)):
    """C3 on a hexagonal (6, 6, 2) grid, TR rows, general complex U; 3 parents, any k grid.

    The k assignment (parent k % 3, operation row k % 6 past the parents) is
    a valid table, not the physical star: both chains read the same plan.
    """
    from types import SimpleNamespace
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import centroid_source_map_and_wrap, spinor_rotation_for_sym_row
    fft_grid = (6, 6, 2)
    nk = int(np.prod(kgrid))
    R = np.asarray([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int64)
    ops = np.stack([np.eye(3, dtype=np.int64), R, R @ R])
    tnp = np.zeros((3, 3))
    kfrac = np.stack(np.unravel_index(np.arange(nk), kgrid), 1) / np.asarray(kgrid, float)
    irr = np.asarray([0, 1, 2] + [k % 3 for k in range(3, nk)], np.int32)
    sym = np.asarray([0, 0, 0] + [k % 6 for k in range(3, nk)], np.int32)  # rows 3..5: time reversal
    n = np.asarray([0.3, -0.5, 0.81]); n = n / np.linalg.norm(n)
    sig = np.asarray([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    ns_ = np.einsum("i,ijk->jk", n, sig)
    U1 = np.cos(np.pi / 3) * np.eye(2) - 1j * np.sin(np.pi / 3) * ns_
    U_spatial = np.stack([np.eye(2, dtype=np.complex128), U1, U1 @ U1])
    sym_ns = SimpleNamespace(
        sym_matrices=ops, translations=tnp, irr_idx_k=irr, sym_idx_k=sym,
        unfolded_kpts=kfrac, kirr_fullids=np.asarray([0, 1, 2]),
        spinor_action=lambda rows, *, nspinor: spinor_rotation_for_sym_row(
            U_spatial, np.asarray(rows), 2, nspinor=nspinor, R_cart=ops))
    ix, iy, iz = np.meshgrid(*(np.arange(v) for v in fft_grid), indexing="ij")
    grid = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], 1).astype(np.int32)
    perm_g, _ = centroid_source_map_and_wrap(grid, ops, tnp, fft_grid, extend_trs=True)
    cent = []
    for seed in (7, 20, 0, 30, 45, 61, 13, 50):
        orbit = sorted({int(perm_g[s, seed]) for s in range(3)})
        if not any(c in cent for c in orbit):
            cent.extend(orbit)
        if len(cent) >= 8:
            break
    plan = build_centroid_k_unfold_plan(sym_ns, grid[np.asarray(sorted(cent))], fft_grid, mesh,
                                        nspinor=ns, parent_k_frac=kfrac[[0, 1, 2]])
    return dict(plan=plan, kgrid=kgrid)


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
    cases += [c3_fixture(mesh, ns) for ns in (2, 4)]
    for fx in cases:
        r = unfold_case(mesh, fx)
        assert r["tables_bitwise"], r
        assert r["door_bitwise"], r
        assert r["red_rel"] > 1e-3, r


def test_unfold_door_refuses_operands_its_tables_were_not_built_for():
    """Red twins of the door guards: a G with one parent row too few, and a mesh of another shape."""
    from ffi import fft as F
    mesh = _mesh()
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(5), 2)
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    ns, mu, n_par, nk = 2, int(plan.n_centroid_packed), int(plan.n_parent), int(plan.n_full)
    tables = plan.unfold_load_tables()
    door = F.make_kconv_klead_unfold(mesh, kg, tables, norm="ortho", mult=1.0)
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    short = fixtures._put(np.zeros((n_par - 1, mu, ns, mu, ns), complex), s5)
    W = fixtures._put(np.zeros((nk, mu, mu), complex), NamedSharding(mesh, P(None, "x", "y")))
    try:
        door(short, short, W)
    except ValueError as exc:
        assert "does not match its tables" in str(exc)
    else:
        raise AssertionError("a G with too few parent rows was accepted")
    flat = Mesh(np.asarray(jax.devices()[:4]).reshape(4, 1), ("x", "y"))
    try:
        F.make_kconv_klead_unfold(flat, kg, tables, norm="ortho", mult=1.0)
    except ValueError as exc:
        assert "mesh" in str(exc)
    else:
        raise AssertionError("tables cut for a 2x2 mesh were accepted on a 4x1 mesh")
