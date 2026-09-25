"""The mixed-basis pair convolution against materialized references, on CPU meshes (P1, P4).

``gw.mixed_basis_pair_convolution.MixedBasisPairConvolution`` computes one τ node of
X_q(G,G') = FFT_{r→G} FFT_{r'→G'} Σ_R e^{..} Σ_αβ A(r+R,r') conj C(r+R,r').  The
references (``mixed_basis_pair_conv_cases``) materialize every a_k(r, r') and the
supercell A(r+R, r') with explicit Fourier matrices, or take the direct k-sum
convolution; tolerance 1e-11 relative to max|X|.

* random operands, n_s = 1 and 2, garbage in every pad slot, a q subset, both
  backends (``'xla'`` and the router's cpu leg), one chunk and forced r'/batch/k/q
  chunks with a tail;
* A-cubic (real diamond-H2 tables: 48 operations with glides, 2x2x2 k) and the
  order-two glide group with spin mixing and an antiunitary row (n_s = 2): the
  parent input with the typed G-space transport equals the full-grid input built
  from children realized by the r-space action;
* red twins: a rolled source table, conjugated transport phases (A-cubic), a dropped
  antiunitary flag (glide) each miss by > 1e-3; an aliasing box refuses.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import mixed_basis_pair_conv_cases as cases
import zeta_mubatch_fixtures as fixtures

TOL = 1e-11


def _mesh(n):
    if n == 1:
        return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    from lxkit.testing import require_devices
    require_devices(4)
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _conv(mesh, kgrid, fft_grid, sph, ngk, kfrac, out, transport, **kw):
    from gw.mixed_basis_pair_convolution import (MixedBasisPairConvolution, PairOperand,
                                                SphereSet)
    op = PairOperand(SphereSet(sph, ngk, kfrac), transport)
    return MixedBasisPairConvolution(mesh, kgrid=kgrid, fft_grid=fft_grid, left=op, right=op,
                                     out=SphereSet(*out), **kw)


def _run(conv, A, C, At=None, Ct=None):
    mesh = conv.mesh
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    put = lambda a: None if a is None else fixtures._put(
        cases.pad_tiles(a, conv.width_carrier[0]), s5)
    X = conv(put(A), put(C), A_partner=put(At), C_partner=put(Ct))
    return conv.strip(X)


def random_case(ns, *, seed=0):
    kgrid, fft_grid = (2, 2, 1), (6, 5, 5)
    kfrac = cases.kgrid_frac(kgrid)
    sph, ngk = cases.spheres(kfrac, np.eye(3), 1.3)
    rng = np.random.default_rng(seed + ns)
    A = cases.random_green(rng, len(kfrac), sph.shape[1], ngk, ns, garbage=1e3)
    C = cases.random_green(rng, len(kfrac), sph.shape[1], ngk, ns, garbage=1e3)
    qsel = np.asarray([0, 3, 1])
    osph, ongk = cases.spheres(kfrac[qsel], np.eye(3), 1.0)
    out = (osph, ongk, kfrac[qsel])
    return dict(kgrid=kgrid, fft_grid=fft_grid, kfrac=kfrac, sph=sph, ngk=ngk, A=A, C=C,
                out=out, ns=ns)


def _references(c, A=None, C=None):
    args = (c["sph"], c["ngk"], c["kfrac"], c["kgrid"], c["fft_grid"], *c["out"])
    A = c["A"] if A is None else A
    C = c["C"] if C is None else C
    return cases.dense_reference(A, C, *args), cases.ksum_reference(A, C, *args)


@pytest.mark.parametrize("ns", [1, 2])
def test_references_agree(ns):
    c = random_case(ns)
    d, k = _references(c)
    assert cases.rel(k, d) <= 1e-12


@pytest.mark.parametrize("n_mesh", [1, 4])
@pytest.mark.parametrize("ns", [1, 2])
@pytest.mark.parametrize("backend", ["xla", "router"])
def test_random_operands_match_the_dense_reference(n_mesh, ns, backend):
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    c = random_case(ns)
    mesh = _mesh(n_mesh)
    ref, _ = _references(c)
    tr = SphereTransport.identity(SphereSet(c["sph"], c["ngk"], c["kfrac"]), ns)
    base = dict(transport=tr, backend=backend, budget_bytes=int(1e10))
    conv = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"], **base)
    assert conv.chunks.n_c == 1 and conv.backend == backend
    got = _run(conv, c["A"], c["C"])
    assert cases.rel(got, ref) <= TOL, cases.rel(got, ref)
    # r' chunks with a batch tail, k and q chunks
    tight = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                  transport=tr, backend=backend, chunks=(3, 4, 2, 1), budget_bytes=int(1e10))
    assert (tight.chunks.n_c, tight.chunks.kc, tight.chunks.qc) == (3, 2, 1) \
        and tight.n_batch >= 2, tight.describe()
    got = _run(tight, c["A"], c["C"])
    assert cases.rel(got, ref) <= TOL, (cases.rel(got, ref), tight.describe())


def test_rolled_transport_misses():
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    c = random_case(1)
    mesh = _mesh(1)
    ref, _ = _references(c)
    tr = SphereTransport.identity(SphereSet(c["sph"], c["ngk"], c["kfrac"]), 1)
    red = SphereTransport(row=tr.row, anti=tr.anti, spin=tr.spin, src=np.roll(tr.src, 1, axis=1),
                          phase=tr.phase, n_parent=tr.n_parent)
    conv = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                 transport=red, backend="xla", budget_bytes=int(1e10))
    assert cases.rel(_run(conv, c["A"], c["C"]), ref) > 1e-3


def test_aliasing_box_refuses():
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    c = random_case(1)
    tr = SphereTransport.identity(SphereSet(c["sph"], c["ngk"], c["kfrac"]), 1)
    with pytest.raises(ValueError, match="GATE pairconv-alias"):
        _conv(_mesh(1), c["kgrid"], (4, 5, 5), c["sph"], c["ngk"], c["kfrac"], c["out"],
              transport=tr, backend="xla", budget_bytes=int(1e10))


# ---------------------------------------------------------------------------
# symmetry: typed parents against full-grid input
# ---------------------------------------------------------------------------

def symmetry_case(fx, *, ecut, metric, box, nb=5, seed=3):
    """Parent ψ on metric spheres; the full grid through the r-space action."""
    from common.gvec_fft_box import build_sphere_box_index
    plan = fx["plan"]
    ns = int(plan.nspinor)
    kpar, kful = np.asarray(plan.k_parent_frac), np.asarray(fx["kfull"])
    kgrid = tuple(int(v) for v in fx["kgrid"])
    order = np.argsort(np.ravel_multi_index(
        (np.rint(kful * np.asarray(kgrid)).astype(int) % np.asarray(kgrid)).T, kgrid))
    assert np.array_equal(order, np.arange(len(kful))), "fixture rows are not C-order"
    psph, pngk = cases.spheres(kpar, metric, ecut)
    csph, cngk = cases.spheres(kful, metric, ecut, width=psph.shape[1])
    rng = np.random.default_rng(seed)
    cpar = (rng.standard_normal((len(kpar), nb, ns, psph.shape[1]))
            + 1j * rng.standard_normal((len(kpar), nb, ns, psph.shape[1])))
    cpar *= (np.arange(psph.shape[1])[None, :] < pngk[:, None])[:, None, None, :]
    cchild, leak = cases.children_psi(fx, cpar, psph, pngk, csph, cngk)
    wA, wC = rng.standard_normal(nb), rng.standard_normal(nb)          # real weights
    sidx = build_sphere_box_index(psph, tuple(fx["fft_grid"]), psph.shape[1], ngk_valid=pngk)
    return dict(plan=plan, ns=ns, kgrid=kgrid, fft_grid=box, kfrac=kful, sph=csph, ngk=cngk,
                psph=psph, pngk=pngk, sidx=sidx, leak=leak, fx=fx,
                A_par=cases.green_from_psi(cpar, wA), C_par=cases.green_from_psi(cpar, wC),
                A=cases.green_from_psi(cchild, wA), C=cases.green_from_psi(cchild, wC),
                out=(*cases.spheres(kful[[0, 1, len(kful) - 1]], metric, 0.8 * ecut),
                     kful[[0, 1, len(kful) - 1]]))


def _symmetry_check(mesh, c, backend):
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    assert c["leak"] <= 1e-13, c["leak"]
    children = SphereSet(c["sph"], c["ngk"], c["kfrac"])
    typed = SphereTransport.typed(c["plan"], fft_grid=c["fx"]["fft_grid"],
                                  parent_sphere_index=c["sidx"], children=children)
    ref, _ = _references(c)
    kw = dict(backend=backend, budget_bytes=int(1e10))
    full = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                 transport=SphereTransport.identity(children, c["ns"]), **kw)
    par = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                transport=typed, **kw)
    got_full = _run(full, c["A"], c["C"])
    got_par = _run(par, c["A_par"], c["C_par"])                       # conj partner: real weights
    red = {}
    for name, bad in (("conj_phase", dict(phase=np.conj(typed.phase))),
                      ("no_anti", dict(anti=np.zeros_like(typed.anti)))):
        fields = dict(dict(row=typed.row, anti=typed.anti, spin=typed.spin, src=typed.src,
                           phase=typed.phase, n_parent=typed.n_parent), **bad)
        conv = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                     transport=SphereTransport(**fields), **kw)
        red[name] = cases.rel(_run(conv, c["A_par"], c["C_par"]), ref)
    return dict(full=cases.rel(got_full, ref), parent=cases.rel(got_par, ref),
                anti=bool(np.any(typed.anti)), red=red)


def acubic_case(mesh):
    fx = fixtures._acubic_fixture(mesh, np.random.default_rng(1))
    from file_io import WfnLoader
    root = fixtures._HERE / "core" / "fixtures" / "A-cubic"
    with WfnLoader(root / "WFN.h5", backend="eager", qe_schema=root / "data-file-schema.xml") as w:
        b = np.asarray(w.bvec, dtype=np.float64)
    return symmetry_case(fx, ecut=1.05 * float(np.min(np.einsum("ij,ij->i", b, b))),
                         metric=b @ b.T, box=(8, 8, 8))


def glide_case(mesh, ns):
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), ns)
    return symmetry_case(fx, ecut=1.3, metric=np.eye(3), box=(6, 6, 5))


@pytest.mark.parametrize("backend", ["xla", "router"])
def test_acubic_parents_equal_full_grid(backend):
    mesh = _mesh(4)
    r = _symmetry_check(mesh, acubic_case(mesh), backend)
    assert r["full"] <= TOL and r["parent"] <= TOL, r
    assert r["red"]["conj_phase"] > 1e-3, r         # the glide phases are live here


@pytest.mark.parametrize("ns", [2, 4])
@pytest.mark.parametrize("backend", ["xla", "router"])
def test_glide_parents_equal_full_grid(ns, backend):
    mesh = _mesh(4)
    r = _symmetry_check(mesh, glide_case(mesh, ns), backend)
    assert r["anti"], "the glide fixture must carry an antiunitary row"
    assert r["full"] <= TOL and r["parent"] <= TOL, r
    assert r["red"]["no_anti"] > 1e-3, r            # the antiunitary row is live here
