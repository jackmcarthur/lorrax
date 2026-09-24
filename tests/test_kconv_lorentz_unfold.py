"""The four-current Σ door against the chain it replaced, on real symmetry plans.

``ffi.fft.make_kconv_lorentz_unfold(G, Gt, V)`` reads the raw-parent Green and
must equal the old Lorentz chain (deleted from ``gw.cohsex_sigma``, kept
verbatim below as ``old_chain``) on bispinor (ns = 4) plans: the typed unfold (``plan.unfold_operator``),
``sigma_conv_operand``, one ``norm='ortho'`` transform of the Green, a scan
over the Lorentz blocks (γ̃_A on the left spin axis, γ̃_B† on the right, times
the transformed block interaction), the forward transform, then
``prefactor * ... * mult``.  Cases: the glide plan (spin mixing, an
antiunitary row), C3 with a general complex U and q = n/3, and a
RECTANGULAR glide class (left and right centroid sets differ, as charge x
current).  Classes: CC (1 block), CT (1 x 3), TT (3 x 3).  Red twin: the right
source table rolled by one slot must miss.  ``lorentz_case`` is reused on GPU
by ``tests/multi_device/kconv_router_p4.py`` (mathdx mode 8, bitwise).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import zeta_mubatch_fixtures as fixtures

#: (left vertices, right vertices) of the three endpoint classes.
CLASSES = {"CC": ((0,), (0,)), "CT": ((0,), (1, 2, 3)), "TT": ((1, 2, 3), (1, 2, 3))}


def old_chain(mesh, kg, nk_tot):
    """The deleted ``_make_lorentz_convolution`` (non-q0 branch), verbatim."""
    from common.gamma_matrices import gamma_apply
    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn
    from gw.wavefunction_bundle import SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC, sigma_conv_operand
    scale = -1.0 / np.sqrt(float(nk_tot))
    inverse_g = make_flat_k_ifftn(mesh, kg, SIGMA_CONV_G7D_SPEC, norm='ortho')
    forward_g = make_flat_k_fftn(mesh, kg, SIGMA_CONV_G7D_SPEC, norm='ortho')
    inverse_v = make_flat_k_ifftn(mesh, kg, V_FFT5D_SPEC, norm='ortho')

    @jax.jit
    def convolve(G_k, interactions, prefactor, vertices):
        G_k = sigma_conv_operand(G_k)
        green = inverse_g(G_k)

        def add(total, block):
            interaction, (left, right) = block
            value = gamma_apply(green, *left, axis=1)
            value = gamma_apply(value, right[0], jnp.conj(right[1]), axis=3)
            weight = inverse_v(interaction)[:, None, :, None, :]
            return total + value * weight, None

        sigma, _ = jax.lax.scan(add, jnp.zeros_like(green), (interactions, vertices), unroll=1)
        return prefactor * forward_g(sigma) * scale
    return convolve


def right_glide_plan(mesh, fx, n_cent=4):
    """A second, smaller orbit-closed centroid set on the glide fixture's symmetry."""
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import centroid_source_map_and_wrap
    plan, fg = fx["plan"], fx["fft_grid"]
    grid = fixtures._grid_points(fg)
    perm_g, _ = centroid_source_map_and_wrap(grid, fx["ops"], fx["tnp"], fg, extend_trs=True)
    cent = []
    for seed in (9, 14, 33, 47, 2, 18, 38, 55):
        orbit = sorted({int(perm_g[s, seed]) for s in range(perm_g.shape[0])})
        if len(cent) + len(orbit) <= n_cent and not any(c in cent for c in orbit):
            cent.extend(orbit)
        if len(cent) == n_cent:
            break
    assert len(cent) == n_cent, cent
    parent_k = fx["kfull"][np.asarray(plan.sym.kirr_fullids)]
    return build_centroid_k_unfold_plan(plan.sym, grid[np.asarray(sorted(cent))], fg, mesh,
                                        nspinor=int(plan.nspinor), parent_k_frac=parent_k)


def lorentz_case(mesh, fx, cls, *, right_plan=None, prefactor=1.0, seed=0):
    """(bitwise door == old chain, max|Δ|, rel, red-twin rel) for one class on one plan."""
    from ffi import fft as F
    from common.gamma_matrices import gamma_perm_phase, gamma_perm_phase_host
    from gw.cohsex_sigma import lorentz_class_vertices
    lefts, rights = CLASSES[cls]
    keys = tuple((A, B) for A in lefts for B in rights)
    assert lorentz_class_vertices(keys) == (lefts, rights)
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    rplan = plan if right_plan is None else right_plan
    ns, n_par, nk = int(plan.nspinor), int(plan.n_parent), int(plan.n_full)
    mx, my = int(plan.n_centroid_packed), int(rplan.n_centroid_packed)
    rng = np.random.default_rng(seed + 17 * ns + 5 * len(keys))
    crand = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    G, Gt = crand(n_par, mx, ns, my, ns), crand(n_par, mx, ns, my, ns)
    blocks = crand(len(keys), nk, mx, my)
    anti = bool(np.any(np.asarray(plan.sym_idx) >= plan.n_sym_spatial))
    Gd, Gtd = fixtures._put(G, s5), fixtures._put(Gt, s5)
    Gk = plan.unfold_operator(Gd, operator_transpose=Gtd if anti else None,
                              right_plan=None if right_plan is None else rplan)
    vertices = jax.tree.map(lambda *v: jnp.stack(v),
                            *((gamma_perm_phase(A), gamma_perm_phase(B)) for A, B in keys))
    ref = fixtures._host(old_chain(mesh, kg, nk)(
        Gk, fixtures._put(blocks, NamedSharding(mesh, P(None, None, "x", "y"))),
        prefactor, vertices))
    V = blocks.reshape(len(lefts), len(rights), nk, mx, my).transpose(2, 3, 0, 4, 1)
    Vd = fixtures._put(V, s5)
    mult = -1.0 / np.sqrt(float(nk))

    def door_for(tables):
        door = F.make_kconv_lorentz_unfold(
            mesh, kg, tables, left_vertices=[gamma_perm_phase_host(A) for A in lefts],
            right_vertices=[gamma_perm_phase_host(B) for B in rights], norm="ortho", mult=mult)
        return fixtures._host(prefactor * door(Gd, Gtd if anti else None, Vd))
    tables = plan.unfold_load_tables(right_plan=right_plan)
    got = door_for(tables)
    red = door_for(tables._replace(rsrc=np.roll(tables.rsrc, 1, axis=1)))
    dmax = float(np.max(np.abs(got - ref)))
    return dict(cls=cls, ns=ns, nk=nk, mx=mx, my=my, antiunitary=anti, prefactor=prefactor,
                rectangular=right_plan is not None,
                door_bitwise=bool(np.array_equal(got, ref)), max_abs=dmax,
                rel=dmax / float(np.max(np.abs(ref))),
                red_rel=float(np.max(np.abs(red - ref)) / np.max(np.abs(ref))))


def cases(mesh, rng):
    """(fixture, class, right plan, prefactor) over the plans and classes the gates run."""
    from test_kconv_klead_unfold import c3_fixture
    # The γ̃ vertices are 4x4: the Lorentz sums exist only on bispinor (ns = 4) Greens.
    fx4 = fixtures._glide_fixture(mesh, rng, 4)
    out = [(fx4, cls, None, 1.0) for cls in CLASSES]
    out += [(fx4, "CT", right_glide_plan(mesh, fx4), -0.5),
            (fx4, "TT", right_glide_plan(mesh, fx4), 1.0)]
    c3 = c3_fixture(mesh, 4)
    out += [(c3, "TT", None, 1.0), (c3, "CT", None, -0.5)]
    return out


def _mesh():
    from lxkit.testing import require_devices
    require_devices(4)
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_lorentz_door_matches_old_chain():
    """cpu leg vs the old Lorentz chain within 2 ulp of max|ref|; red twin fires.
    The CUDA kernel (mode 8) is held to the same chain by the P4 gate."""
    from ffi import fft as F
    mesh = _mesh()
    assert F.kconv_backend(mesh) == "plan"
    eps = np.finfo(float).eps
    for fx, cls, rplan, pref in cases(mesh, np.random.default_rng(11)):
        r = lorentz_case(mesh, fx, cls, right_plan=rplan, prefactor=pref)
        assert r["rel"] <= 2 * eps, r
        assert r["red_rel"] > 1e-3, r


def test_lorentz_door_refuses_non_product_classes_and_bad_operands():
    """A class that is not one A x B product, and a V of the wrong block count, refuse by name."""
    from ffi import fft as F
    from common.gamma_matrices import gamma_perm_phase_host
    from gw.cohsex_sigma import lorentz_class_vertices
    try:
        lorentz_class_vertices(((1, 1), (2, 2)))
    except ValueError as exc:
        assert "product" in str(exc)
    else:
        raise AssertionError("a diagonal-only class was accepted as a product")
    mesh = _mesh()
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(5), 4)
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    mu, n_par, nk = int(plan.n_centroid_packed), int(plan.n_parent), int(plan.n_full)
    door = F.make_kconv_lorentz_unfold(
        mesh, kg, plan.unfold_load_tables(),
        left_vertices=[gamma_perm_phase_host(1)],
        right_vertices=[gamma_perm_phase_host(b) for b in (1, 2, 3)], norm="ortho", mult=1.0)
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    G = fixtures._put(np.zeros((n_par, mu, 4, mu, 4), complex), s5)
    V = fixtures._put(np.zeros((nk, mu, 1, mu, 2), complex), s5)
    try:
        door(G, G, V)
    except ValueError as exc:
        assert "vertices" in str(exc)
    else:
        raise AssertionError("a V with 2 right blocks was accepted for 3 right vertices")
