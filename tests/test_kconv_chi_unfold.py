"""The chi0 door read from the raw-parent Green pair, against the chain it replaces.

``ffi.fft.make_kconv_chi_unfold`` accumulates ``acc[o] += alpha[o] * chi_tau`` with
``chi_tau = sum_ab conj(Gc'_ab) Gv'_ab`` (+ c.c. on a real contour), ``G' = ifftn_k`` of the
typed unfold.  The chain it replaces (``gw.w_isdf._get_chi_minimax_kernel_face``) unfolds to
full k conjugated, transforms forward and contracts ``Gc_R conj(Gv_R)``; the two are the same
number because ``fftn(conj x) = conj(ifftn x)`` (ortho), so they agree to rounding, not bit
for bit.  Held within 8 ulp of ``max|chi|``.  Cases: the glide plans (ns = 2, 4; an
antiunitary row), A-cubic (ns = 1, 12^3: the k-box single pass on a 3-D grid) and the C3
group with general phases and spin action (ns = 2, 4), each with the partner read as
``conj(G)`` (real weights) and as its own tile (a complex contour), one and two weight rows.
Red twin: the right source table rolled by one slot must miss.  ``chi_case`` is reused on GPU
by ``tests/multi_device/kconv_router_p4.py`` (mathdx mode 11, both k-box arms).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import zeta_mubatch_fixtures as fixtures


def chi_case(mesh, fx, *, n_out=1, complete=True, conj_src=True, seed=0):
    """(rel of the door vs the incumbent chain, red twin rel) for one tau node."""
    from common.fft_helpers import make_flat_k_fftn
    from ffi import fft as F
    from gw.wavefunction_bundle import G_FFT7D_SPEC
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    ns, mu, n_par, nk = int(plan.nspinor), int(plan.n_centroid_packed), int(plan.n_parent), int(plan.n_full)
    rng = np.random.default_rng(seed + 31 * ns + 7 * n_out)
    crand = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    sacc = NamedSharding(mesh, P(None, None, "x", "y"))
    Gv, Gc = crand(n_par, mu, ns, mu, ns), crand(n_par, mu, ns, mu, ns)
    Gvt, Gct = (np.conj(Gv), np.conj(Gc)) if conj_src else (crand(n_par, mu, ns, mu, ns),
                                                           crand(n_par, mu, ns, mu, ns))
    acc0 = crand(n_out, nk, mu, mu)
    alpha = crand(n_out)
    put5 = lambda a: fixtures._put(a, s5)
    anti = bool(np.any(np.asarray(plan.sym_idx) >= plan.n_sym_spatial))
    # The incumbent chain: the conjugated full-k Greens, forward transform, Gc_R conj(Gv_R).
    fftn = make_flat_k_fftn(mesh, kg, G_FFT7D_SPEC, norm="ortho")
    R = lambda g, gt: fixtures._host(fftn(plan.unfold_operator(
        put5(g), operator_transpose=put5(gt) if anti else None, conjugate=True)))
    Gv_R, Gc_R = R(Gv, Gvt), R(Gc, Gct)
    chi = np.einsum("Rmanb,Rmanb->Rmn", Gc_R, np.conj(Gv_R))
    if complete:
        chi = chi + np.conj(chi)
    ref = acc0 + alpha[:, None, None, None] * chi[None]
    tables = plan.unfold_load_tables()

    def run(t):
        door = F.make_kconv_chi_unfold(mesh, kg, t, n_out=n_out, complete=complete, norm="ortho")
        parts = () if conj_src else (put5(Gvt), put5(Gct))
        return fixtures._host(door(fixtures._put(acc0, sacc), put5(Gv), put5(Gc),
                                   fixtures._put(alpha, NamedSharding(mesh, P(None))), *parts))
    got = run(tables)
    red = run(tables._replace(rsrc=np.roll(tables.rsrc, 1, axis=1)))
    scale = float(np.max(np.abs(chi)))
    return dict(ns=ns, nk=nk, n_parent=n_par, mu=mu, antiunitary=anti, n_out=n_out,
                complete=complete, conj_src=conj_src,
                rel=float(np.max(np.abs(got - ref))) / scale,
                red_rel=float(np.max(np.abs(red - ref))) / scale)


def chi_cases(mesh, rng):
    """The case list: (fixture, n_out, complete, conj_src)."""
    from test_kconv_klead_unfold import c3_fixture
    out = []
    for ns in (2, 4):
        fx = fixtures._glide_fixture(mesh, rng, ns)
        out += [(fx, 1, True, True), (fx, 2, False, False)]
    out.append((fixtures._acubic_fixture(mesh, rng), 1, True, True))
    for ns in (2, 4):
        fx = c3_fixture(mesh, ns)
        out += [(fx, 1, True, True), (fx, 2, False, False)]
    return out


def _mesh():
    from lxkit.testing import require_devices
    require_devices(4)
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_chi_door_matches_the_incumbent_chain():
    from ffi import fft as F
    mesh = _mesh()
    assert F.kconv_backend(mesh) == "plan"
    eps = float(np.finfo(float).eps)
    for fx, n_out, complete, conj_src in chi_cases(mesh, np.random.default_rng(5)):
        r = chi_case(mesh, fx, n_out=n_out, complete=complete, conj_src=conj_src)
        assert r["rel"] <= 8 * eps, r
        assert r["red_rel"] > 1e-3, r


def test_chi_door_refuses_one_partner():
    from ffi import fft as F
    mesh = _mesh()
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(6), 2)
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    ns, mu, n_par, nk = int(plan.nspinor), int(plan.n_centroid_packed), int(plan.n_parent), int(plan.n_full)
    door = F.make_kconv_chi_unfold(mesh, kg, plan.unfold_load_tables(), n_out=1, complete=True)
    z5 = fixtures._put(np.zeros((n_par, mu, ns, mu, ns), complex), NamedSharding(mesh, P(None, "x", None, "y", None)))
    acc = fixtures._put(np.zeros((1, nk, mu, mu), complex), NamedSharding(mesh, P(None, None, "x", "y")))
    alpha = jnp.ones((1,), jnp.complex128)
    try:
        door(acc, z5, z5, alpha, z5, None)
    except ValueError as exc:
        assert "both partners" in str(exc)
    else:
        raise AssertionError("a chi door with one partner tile was accepted")


def test_chi_unfold_refusal_matches_the_handler_residency_rule():
    """The route's predicate reproduces the arms the handler built on an A100 (opt-in 166912 B)
    and refuses a grid neither arm holds."""
    from ffi.fft import chi_unfold_refusal
    a100 = 166912
    assert chi_unfold_refusal((6, 6, 1), 4, a100) == ""       # single pass (75776 B at tr = 4)
    assert chi_unfold_refusal((12, 12, 9), 2, a100) == ""     # single pass at one pair, 166016 B
    assert chi_unfold_refusal((8, 8, 8), 4, a100) == ""       # split: plane 18688 B, pencil 33792 B
    assert "opt-in" in chi_unfold_refusal((40, 40, 40), 4, a100)
