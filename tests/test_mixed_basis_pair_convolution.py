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


def random_case(ns, *, seed=0, kgrid=(2, 2, 1), fft_grid=(6, 5, 5)):
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


def test_collective_census():
    """The streamed middle compiles to no collective; each redistribution is one all-to-all
    (the census counts nonzero where collectives exist, so an empty middle is a measurement)."""
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    c = random_case(2)
    tr = SphereTransport.identity(SphereSet(c["sph"], c["ngk"], c["kfrac"]), 2)
    conv = _conv(_mesh(4), c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                 transport=tr, backend="xla", budget_bytes=int(1e10), chunks=(2, 3, 2, 1))
    census = conv.collective_census()
    assert census["middle"] == {}, census
    assert census["final"] == {"all-to-all": 2}, census
    for k in ("slab left", "slab right", "expand left", "expand right"):
        assert census[k] == {"all-to-all": 1}, census


def test_recentred_representatives():
    """k in [0, 1) on a 3x1x2 grid (2/3 recentres to -1/3 inside the plan): still the reference."""
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    c = random_case(2, kgrid=(3, 1, 2), fft_grid=(7, 6, 6))
    assert np.any(np.rint(c["kfrac"]) != 0)
    ref, _ = _references(c)
    tr = SphereTransport.identity(SphereSet(c["sph"], c["ngk"], c["kfrac"]), 2)
    for backend in ("xla", "router"):
        conv = _conv(_mesh(4), c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                     transport=tr, backend=backend, budget_bytes=int(1e10))
        assert cases.rel(_run(conv, c["A"], c["C"]), ref) <= TOL


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


# ---------------------------------------------------------------------------
# the r'-column wedge: covariant operands, against the dense reference and the dense-column plan
# ---------------------------------------------------------------------------

def covariant_case(fx, *, ecut, metric, box, nb=4, seed=7, complex_weights=False):
    """Parent ψ closed under each parent's little group (``close_under_little_groups``), so the
    Greens are covariant under the whole group; the full grid through the r-space action.
    ``out_full``: the output sphere at every full-grid q (the wedge's middle rows)."""
    from common.gvec_fft_box import build_sphere_box_index
    plan = fx["plan"]
    ns = int(plan.nspinor)
    kpar, kful = np.asarray(plan.k_parent_frac), np.asarray(fx["kfull"])
    kgrid = tuple(int(v) for v in fx["kgrid"])
    psph, pngk = cases.spheres(kpar, metric, ecut)
    csph, cngk = cases.spheres(kful, metric, ecut, width=psph.shape[1])
    rng = np.random.default_rng(seed)
    c0 = (rng.standard_normal((len(kpar), nb, ns, psph.shape[1]))
          + 1j * rng.standard_normal((len(kpar), nb, ns, psph.shape[1])))
    c0 *= (np.arange(psph.shape[1])[None, :] < pngk[:, None])[:, None, None, :]
    cpar, leak0 = cases.close_under_little_groups(fx, c0, psph, pngk, rows=fx["rows"],
                                                  spinor_action=fx["spinor_action"], ns=ns)
    cchild, leak = cases.children_psi(fx, cpar, psph, pngk, csph, cngk)
    n_img = cpar.shape[1] // nb
    if complex_weights:
        wA = np.tile(rng.standard_normal(nb) + 1j * rng.standard_normal(nb), n_img)
        wC = np.tile(rng.standard_normal(nb) + 1j * rng.standard_normal(nb), n_img)
    else:
        wA, wC = np.tile(rng.standard_normal(nb), n_img), np.tile(rng.standard_normal(nb), n_img)
    sidx = build_sphere_box_index(psph, tuple(fx["fft_grid"]), psph.shape[1], ngk_valid=pngk)
    qsel = [0, 1, len(kful) - 1]
    osph, ongk = cases.spheres(kful[qsel], metric, 0.8 * ecut)
    fsph, fngk = cases.spheres(kful, metric, 0.8 * ecut)
    A_par, C_par = cases.green_from_psi(cpar, wA), cases.green_from_psi(cpar, wC)
    return dict(plan=plan, ns=ns, kgrid=kgrid, fft_grid=box, kfrac=kful, sph=csph, ngk=cngk,
                psph=psph, pngk=pngk, sidx=sidx, leak=max(leak0, leak), fx=fx,
                A_par=A_par, C_par=C_par,
                At_par=np.transpose(A_par, (0, 3, 4, 1, 2)) if complex_weights else None,
                Ct_par=np.transpose(C_par, (0, 3, 4, 1, 2)) if complex_weights else None,
                A=cases.green_from_psi(cchild, wA), C=cases.green_from_psi(cchild, wC),
                out=(osph, ongk, kful[qsel]), out_full=(fsph, fngk, kful))


def _wedge(c, rows):
    from gw.mixed_basis_pair_convolution import ColumnWedge, SphereSet
    fx = c["fx"]
    return ColumnWedge(np.asarray(fx["ops"]), np.asarray(fx["tnp"]), np.asarray(rows),
                       SphereSet(*c["out_full"]))


def _wedge_check(mesh, c, backend, rows, *, twins=()):
    """Wedge on typed parents against the dense reference (full-grid operands) and against the
    dense-column plan on the same parents; ``twins`` names red twins to measure."""
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    assert c["leak"] <= 1e-13, c["leak"]
    children = SphereSet(c["sph"], c["ngk"], c["kfrac"])
    typed = SphereTransport.typed(c["plan"], fft_grid=c["fx"]["fft_grid"],
                                  parent_sphere_index=c["sidx"], children=children)
    ref = cases.dense_reference(c["A"], c["C"], c["sph"], c["ngk"], c["kfrac"], c["kgrid"],
                                c["fft_grid"], *c["out"])
    kw = dict(backend=backend, budget_bytes=int(1e10), transport=typed)
    args = (c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"])
    pa = (c["A_par"], c["C_par"], c["At_par"], c["Ct_par"])
    dense = _run(_conv(mesh, *args, **kw), *pa)
    conv = _conv(mesh, *args, wedge=_wedge(c, rows), **kw)
    got = _run(conv, *pa)
    red = {}
    for name in twins:
        tw = _conv(mesh, *args, wedge=_wedge(c, rows), **kw)
        par, pslot, gph, anti, sel, phl = tw._dev_rebuild
        if name == "no_conj":             # antiunitary rows read without the conjugation
            anti = tw._put(np.zeros_like(tw._wt["anti"]))
        elif name == "no_wrap":           # the lattice-wrap phase e^{-2πi q·S⁻¹L} dropped
            phl = tw._put(np.where(np.abs(tw._wt["phl"]) > 0, 1.0 + 0j, 0j), P(None, ("x", "y")))
        tw._dev_rebuild = (par, pslot, gph, anti, sel, phl)
        red[name] = cases.rel(_run(tw, *pa), ref)
    return dict(ref=cases.rel(got, ref), dense=cases.rel(got, dense), dense_ref=cases.rel(dense, ref),
                orbits=conv._wt["n_orbits"], nr=conv.nr, red=red, conv=conv)


@pytest.mark.parametrize("backend", ["xla", "router"])
@pytest.mark.parametrize("rows,twins", [((0, 1, 2, 3), ("no_wrap",)), ((0, 3), ("no_conj",))])
def test_wedge_glide_equals_dense(backend, rows, twins):
    """Glide group, n_s = 2, spin mixing: the full group (unitary rows kept per spatial part) and
    {E, Θ·glide} (the antiunitary branch carries every non-identity column)."""
    mesh = _mesh(4)
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True)
    c = covariant_case(fx, ecut=1.3, metric=np.eye(3), box=(6, 6, 5))
    r = _wedge_check(mesh, c, backend, rows, twins=twins)
    assert r["ref"] <= TOL and r["dense"] <= TOL and r["dense_ref"] <= TOL, r
    assert r["orbits"] < r["nr"], r
    for name in twins:
        assert r["red"][name] > 1e-3, r


@pytest.mark.parametrize("backend", ["xla", "router"])
def test_wedge_acubic_equals_dense(backend):
    """A-cubic: 48 spatial operations with glides, n_s = 1."""
    mesh = _mesh(4)
    fx = fixtures._acubic_fixture(mesh, np.random.default_rng(1))
    from file_io import WfnLoader
    root = fixtures._HERE / "core" / "fixtures" / "A-cubic"
    with WfnLoader(root / "WFN.h5", backend="eager", qe_schema=root / "data-file-schema.xml") as w:
        b = np.asarray(w.bvec, dtype=np.float64)
    c = covariant_case(fx, ecut=1.05 * float(np.min(np.einsum("ij,ij->i", b, b))), metric=b @ b.T,
                       box=(8, 8, 8), nb=2)
    r = _wedge_check(mesh, c, backend, fx["rows"], twins=("no_wrap",))
    assert r["ref"] <= TOL and r["dense"] <= TOL, r
    assert r["orbits"] * 8 < r["nr"], r
    assert r["red"]["no_wrap"] > 1e-3, r


@pytest.mark.parametrize("n_mesh", [1, 4])
def test_wedge_partners_unitary_only(n_mesh):
    """Complex weights with transposed partners: the unitary rows are exact, antiunitary rows refuse."""
    mesh = _mesh(n_mesh)
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True)
    c = covariant_case(fx, ecut=1.3, metric=np.eye(3), box=(6, 6, 5), complex_weights=True)
    r = _wedge_check(mesh, c, "xla", (0, 1))
    assert r["ref"] <= TOL and r["dense"] <= TOL, r
    with pytest.raises(ValueError, match="GATE pairconv-wedge-partner"):
        _wedge_check(mesh, c, "xla", (0, 3))


def test_wedge_census_and_chunks():
    """The rebuild compiles to no collective; forced r'/batch/k/q chunks on the wedge agree."""
    mesh = _mesh(4)
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True)
    c = covariant_case(fx, ecut=1.3, metric=np.eye(3), box=(6, 6, 5))
    r = _wedge_check(mesh, c, "xla", (0, 1, 2, 3))
    census = r["conv"].collective_census()
    assert census["rebuild"] == {} and census["middle"] == {}, census
    assert census["final"] == {"all-to-all": 2}, census
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    typed = SphereTransport.typed(c["plan"], fft_grid=fx["fft_grid"], parent_sphere_index=c["sidx"],
                                  children=SphereSet(c["sph"], c["ngk"], c["kfrac"]))
    tight = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                  transport=typed, backend="xla", budget_bytes=int(1e10), chunks=(2, 3, 2, 1),
                  wedge=_wedge(c, (0, 1, 2, 3)))
    ref = cases.dense_reference(c["A"], c["C"], c["sph"], c["ngk"], c["kfrac"], c["kgrid"],
                                c["fft_grid"], *c["out"])
    assert tight.chunks.n_c == 2 and tight.n_batch >= 2, tight.describe()
    assert cases.rel(_run(tight, c["A_par"], c["C_par"]), ref) <= TOL
