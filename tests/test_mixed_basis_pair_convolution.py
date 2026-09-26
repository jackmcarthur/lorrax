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
import dataclasses

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
    red = dataclasses.replace(tr, src=np.roll(tr.src, 1, axis=1))
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

EXPAND_TWINS = ("par_no_wrap", "par_identity_map")


def expand_twin(conv, name):
    """A red twin of the expand at the k-parents, in place on ``conv``'s device tables: the column
    map's lattice wrap dropped (``par_no_wrap``: y = x_α, not x_α + L), or the map itself
    (``par_identity_map``: each child reads its parent's column at r', not at mtrx·(r' − τ))."""
    for i, de in enumerate(conv._dev_exp):
        pt = conv._ptables[i]
        for v, tabs in de.items():
            t = list(tabs)
            if name == "par_no_wrap":
                t[9] = conv._put(np.zeros_like(pt["L"]))
            elif name == "par_identity_map":
                t[8] = conv._put(np.broadcast_to(np.arange(conv.nr, dtype=np.int32),
                                                 pt["alpha"].shape).copy())
            else:
                raise ValueError(name)
            de[v] = tuple(t)
    return conv


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
        conv = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                     transport=dataclasses.replace(typed, **bad), **kw)
        red[name] = cases.rel(_run(conv, c["A_par"], c["C_par"]), ref)
    for name in EXPAND_TWINS:
        conv = _conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"], c["out"],
                     transport=typed, **kw)
        red[name] = cases.rel(_run(expand_twin(conv, name), c["A_par"], c["C_par"]), ref)
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
    The spin actions must form a representation (the glide fixture at θ = π/2).
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
        if name in EXPAND_TWINS:
            red[name] = cases.rel(_run(expand_twin(tw, name), *pa), ref)
            continue
        par, pslot, gph, anti, U, sel, phl = tw._dev_rebuild
        if name == "no_conj":             # antiunitary rows read without the conjugation
            anti = tw._put(np.zeros_like(tw._wt["anti"]))
        elif name == "no_wrap":           # the lattice-wrap phase e^{-2πi q·S⁻¹L} dropped
            phl = tw._put(np.where(np.abs(tw._wt["phl"]) > 0, 1.0 + 0j, 0j), P(None, ("x", "y")))
        tw._dev_rebuild = (par, pslot, gph, anti, U, sel, phl)
        red[name] = cases.rel(_run(tw, *pa), ref)
    return dict(ref=cases.rel(got, ref), dense=cases.rel(got, dense), dense_ref=cases.rel(dense, ref),
                orbits=conv._wt["n_orbits"], nr=conv.nr, red=red, conv=conv)


@pytest.mark.parametrize("backend", ["xla", "router"])
@pytest.mark.parametrize("rows,twins", [((0, 1, 2, 3), ("no_wrap",)), ((0, 3), ("no_conj",))])
def test_wedge_glide_equals_dense(backend, rows, twins):
    """Glide group, n_s = 2, spin mixing: the full group (unitary rows kept per spatial part) and
    {E, Θ·glide} (the antiunitary branch carries every non-identity column)."""
    mesh = _mesh(4)
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True,
                                   theta=np.pi / 2)
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
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True,
                                   theta=np.pi / 2)
    c = covariant_case(fx, ecut=1.3, metric=np.eye(3), box=(6, 6, 5), complex_weights=True)
    r = _wedge_check(mesh, c, "xla", (0, 1))
    assert r["ref"] <= TOL and r["dense"] <= TOL, r
    with pytest.raises(ValueError, match="GATE pairconv-wedge-partner"):
        _wedge_check(mesh, c, "xla", (0, 3))


def test_wedge_census_and_chunks():
    """The rebuild compiles to no collective; forced r'/batch/k/q chunks on the wedge agree."""
    mesh = _mesh(4)
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True,
                                   theta=np.pi / 2)
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


def test_screened_sphere_set_and_alias_cap():
    """``screened_coulomb_cutoff``: the default is ecutwfc, the sphere is |q+G|² ≤ cutoff on whole
    shells, and a cutoff at or above the box's measured alias cap refuses by name."""
    from gw.mixed_basis_pair_convolution import (SphereSet, alias_free_margin,
                                                screened_coulomb_cutoff_cap, screened_sphere_set)
    c = random_case(1, kgrid=(2, 2, 2), fft_grid=(8, 8, 8))
    psi = SphereSet(c["sph"], c["ngk"], c["kfrac"])
    ecut = 1.3
    cap = screened_coulomb_cutoff_cap((8, 8, 8), psi, bvec=np.eye(3), q_frac=c["kfrac"])
    s = screened_sphere_set(fft_grid=(8, 8, 8), psi=psi, bvec=np.eye(3), q_frac=c["kfrac"], ecutwfc=ecut)
    want, wngk = cases.spheres(c["kfrac"], np.eye(3), ecut, span=4)
    for i in range(len(c["kfrac"])):
        assert {tuple(g) for g in s.gvecs[i, :s.ngk[i]]} == {tuple(g) for g in want[i, :wngk[i]]}
    below = screened_sphere_set(fft_grid=(8, 8, 8), psi=psi, bvec=np.eye(3), q_frac=c["kfrac"],
                                ecutwfc=ecut, screened_coulomb_cutoff=0.999 * cap)
    m = alias_free_margin((8, 8, 8), *(psi.recentred().union_support(),) * 2,
                          below.recentred().union_support())
    assert np.all(m >= 1), m
    with pytest.raises(ValueError, match="GATE screened-coulomb-cutoff"):
        screened_sphere_set(fft_grid=(8, 8, 8), psi=psi, bvec=np.eye(3), q_frac=c["kfrac"],
                            ecutwfc=ecut, screened_coulomb_cutoff=1.0001 * cap)


# ---------------------------------------------------------------------------
# Σ, the second caller (product='scalar'): A = G on the ψ sphere, B = W on the χ sphere,
# the output on the ψ sphere; B's rows at the other representative of every ±½ plane
# ---------------------------------------------------------------------------

def _sigma_conv(mesh, kgrid, fft_grid, g_op, w_op, out, **kw):
    from gw.mixed_basis_pair_convolution import MixedBasisPairConvolution, SphereSet
    return MixedBasisPairConvolution(mesh, kgrid=kgrid, fft_grid=fft_grid, left=g_op, right=w_op,
                                     out=SphereSet(*out), product="scalar", **kw)


def _run_sigma(conv, G, W, Gt=None):
    """G ``(n, w, ns, w, ns)`` and W ``(n, w, 1, w, 1)`` padded to the carriers; W goes in as the
    3-D ``(n, M, M)`` at ``P(None, 'x', 'y')`` (the χ plan's output layout)."""
    mesh = conv.mesh
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    s3 = NamedSharding(mesh, P(None, "x", "y"))
    g = fixtures._put(cases.pad_tiles(G, conv.width_carrier[0]), s5)
    gt = None if Gt is None else fixtures._put(cases.pad_tiles(Gt, conv.width_carrier[0]), s5)
    w = fixtures._put(cases.pad_tiles(W, conv.width_carrier[1])[:, :, 0, :, 0], s3)
    return conv.strip(conv(g, w, A_partner=gt))


def sigma_random_case(ns, *, seed=0, kgrid=(2, 2, 1), fft_grid=(8, 7, 7)):
    kfrac = cases.kgrid_frac(kgrid)
    wfrac = kfrac - (kfrac >= 0.5)                     # W rows at the other representative of ±½
    sph, ngk = cases.spheres(kfrac, np.eye(3), 1.3)
    wsp, wngk = cases.spheres(wfrac, np.eye(3), 1.0)
    rng = np.random.default_rng(seed + 10 * ns)
    G = cases.random_green(rng, len(kfrac), sph.shape[1], ngk, ns, garbage=1e3)
    W = cases.random_green(rng, len(kfrac), wsp.shape[1], wngk, 1, garbage=1e3)
    ksel = np.asarray([0, 3, 1])
    osph, ongk = cases.spheres(kfrac[ksel], np.eye(3), 1.3)
    return dict(kgrid=kgrid, fft_grid=fft_grid, kfrac=kfrac, wfrac=wfrac, sph=sph, ngk=ngk,
                wsp=wsp, wngk=wngk, G=G, W=W, out=(osph, ongk, kfrac[ksel]), ns=ns)


def _sigma_ref(c, G=None, W=None):
    return cases.dense_scalar_reference(
        c["G"] if G is None else G, c["sph"], c["ngk"], c["kfrac"],
        c["W"] if W is None else W, c["wsp"], c["wngk"], c["wfrac"],
        c["kgrid"], c["fft_grid"], *c["out"])


def _identity_ops(c):
    from gw.mixed_basis_pair_convolution import PairOperand, SphereSet, SphereTransport
    gs, ws = SphereSet(c["sph"], c["ngk"], c["kfrac"]), SphereSet(c["wsp"], c["wngk"], c["wfrac"])
    return (PairOperand(gs, SphereTransport.identity(gs, c["ns"])),
            PairOperand(ws, SphereTransport.identity(ws, 1)))


@pytest.mark.parametrize("n_mesh", [1, 4])
@pytest.mark.parametrize("ns", [1, 2])
@pytest.mark.parametrize("backend", ["xla", "router"])
def test_sigma_random_operands_match_the_dense_reference(n_mesh, ns, backend):
    c = sigma_random_case(ns)
    mesh = _mesh(n_mesh)
    ref = _sigma_ref(c)
    g_op, w_op = _identity_ops(c)
    conv = _sigma_conv(mesh, c["kgrid"], c["fft_grid"], g_op, w_op, c["out"], backend=backend,
                       budget_bytes=int(1e10))
    assert conv.chunks.n_c == 1 and conv.spins == (ns, 1, ns)
    got = _run_sigma(conv, c["G"], c["W"])
    assert got.shape == ref.shape and cases.rel(got, ref) <= TOL, cases.rel(got, ref)
    tight = _sigma_conv(mesh, c["kgrid"], c["fft_grid"], g_op, w_op, c["out"], backend=backend,
                        budget_bytes=int(1e10), chunks=(3, 4, 2, 1))
    assert (tight.chunks.n_c, tight.chunks.kc, tight.chunks.qc) == (3, 2, 1) \
        and tight.n_batch >= 2, tight.describe()
    got = _run_sigma(tight, c["G"], c["W"])
    assert cases.rel(got, ref) <= TOL, (cases.rel(got, ref), tight.describe())


def test_sigma_time_reversal_is_the_plain_product(monkeypatch):
    """Red twin: without the time-reversed transport the plan forms A ⊙ conj B, not A ⊙ B."""
    from gw.mixed_basis_pair_convolution import SphereTransport
    c = sigma_random_case(2)
    ref = _sigma_ref(c)
    g_op, w_op = _identity_ops(c)
    monkeypatch.setattr(SphereTransport, "time_reversed", lambda self, sphere: self)
    conv = _sigma_conv(_mesh(1), c["kgrid"], c["fft_grid"], g_op, w_op, c["out"], backend="xla",
                       budget_bytes=int(1e10))
    assert cases.rel(_run_sigma(conv, c["G"], c["W"]), ref) > 1e-3


def test_sigma_census_and_refusals():
    from gw.mixed_basis_pair_convolution import SphereTransport
    c = sigma_random_case(2)
    g_op, w_op = _identity_ops(c)
    conv = _sigma_conv(_mesh(4), c["kgrid"], c["fft_grid"], g_op, w_op, c["out"], backend="xla",
                       budget_bytes=int(1e10), chunks=(2, 3, 2, 1))
    census = conv.collective_census()
    assert census["middle"] == {}, census
    assert census["final"] == {"all-to-all": 2}, census
    for k in ("slab left", "slab right", "expand left", "expand right"):
        assert census[k] == {"all-to-all": 1}, census
    with pytest.raises(ValueError, match="one-channel right operand"):
        _sigma_conv(_mesh(1), c["kgrid"], c["fft_grid"], g_op, g_op, c["out"], backend="xla")
    with pytest.raises(ValueError, match="pairs equal spin widths"):
        from gw.mixed_basis_pair_convolution import MixedBasisPairConvolution, SphereSet
        MixedBasisPairConvolution(_mesh(1), kgrid=c["kgrid"], fft_grid=c["fft_grid"], left=g_op,
                                  right=w_op, out=SphereSet(*c["out"]), backend="xla")
    from gw.mixed_basis_pair_convolution import SphereSet
    ws = w_op.sphere                   # row 1 loses its last shell slot: no longer inversion-closed
    skew = SphereSet(ws.gvecs, ws.ngk - (np.arange(ws.n) == 1), ws.frac)
    with pytest.raises(ValueError, match="time_reversed"):
        SphereTransport.identity(skew, 1).time_reversed(skew)


@pytest.mark.parametrize("ns", [1, 2])
def test_sigma_and_chi_normalization_against_band_sums(ns):
    """The physical factors: Σ_k = −X/(Ω·N_r²) and χ_q = X/(Ω·N_r²) (Ω = 1 here) against BGW's
    band sums over explicit ⟨nk|e^{i(q+G)·r}|m, k−q⟩ (no box, no FFT)."""
    from gw.mixed_basis_pair_convolution import (MixedBasisPairConvolution, PairOperand, SphereSet,
                                                SphereTransport)
    kgrid, fft_grid = (2, 2, 1), (8, 7, 7)
    kfrac = cases.kgrid_frac(kgrid)
    wfrac = kfrac - (kfrac >= 0.5)
    nr = int(np.prod(fft_grid))
    sph, ngk = cases.spheres(kfrac, np.eye(3), 1.3)
    wsp, wngk = cases.spheres(wfrac, np.eye(3), 1.0)
    rng = np.random.default_rng(40 + ns)
    nb = 3
    cpsi = (rng.standard_normal((len(kfrac), nb, ns, sph.shape[1]))
            + 1j * rng.standard_normal((len(kfrac), nb, ns, sph.shape[1])))
    cpsi *= (np.arange(sph.shape[1])[None, :] < ngk[:, None])[:, None, None, :]
    g = rng.standard_normal(nb)
    W = cases.random_green(rng, len(kfrac), wsp.shape[1], wngk, 1)
    mesh = _mesh(1)
    gs, ws = SphereSet(sph, ngk, kfrac), SphereSet(wsp, wngk, wfrac)
    g_op = PairOperand(gs, SphereTransport.identity(gs, ns))
    w_op = PairOperand(ws, SphereTransport.identity(ws, 1))
    rows = [0, 3]
    conv = _sigma_conv(mesh, kgrid, fft_grid, g_op, w_op, (sph[rows], ngk[rows], kfrac[rows]),
                       backend="xla", budget_bytes=int(1e10))
    X = _run_sigma(conv, cases.green_from_psi(cpsi, g), W)
    sig = -X / nr ** 2
    got = np.einsum("knap,kpaqb,kmbq->knm", np.conj(cpsi[rows]), sig, cpsi[rows])
    want = cases.band_sum_sigma(cpsi, g, sph, ngk, kfrac, W, wsp, wngk, wfrac, kgrid, rows)
    assert cases.rel(got, want) <= TOL, cases.rel(got, want)
    # χ₀ = X/(Ω·N_r²) on the χ sphere
    wc, wv = rng.standard_normal(nb), rng.standard_normal(nb)
    chi = MixedBasisPairConvolution(mesh, kgrid=kgrid, fft_grid=fft_grid, left=g_op, right=g_op,
                                    out=SphereSet(wsp, wngk, wfrac), backend="xla",
                                    budget_bytes=int(1e10))
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    put = lambda a: fixtures._put(cases.pad_tiles(a, chi.width_carrier[0]), s5)
    Xc = chi.strip(chi(put(cases.green_from_psi(cpsi, wc)), put(cases.green_from_psi(cpsi, wv))))
    want = cases.band_sum_chi(cpsi, wc, wv, sph, ngk, kfrac, kgrid, wsp, wngk, wfrac)
    assert cases.rel(Xc / nr ** 2, want) <= TOL, cases.rel(Xc / nr ** 2, want)


# ---- symmetry: G at the k-parents, W at the q-IBZ, both unfolded on load --------------------

def _scalar_plan(plan):
    """The plan's tables with the trivial spin representation (a spin-scalar operand's)."""
    from types import SimpleNamespace
    n = len(np.asarray(plan.irr_idx))
    return SimpleNamespace(irr_idx=np.asarray(plan.irr_idx), sym_idx=np.asarray(plan.sym_idx),
                           k_parent_frac=np.asarray(plan.k_parent_frac),
                           n_sym_spatial=int(plan.n_sym_spatial), n_full=n,
                           spatial_ops=np.asarray(plan.spatial_ops),
                           translations=np.asarray(plan.translations),
                           spin_action_full=np.ones((n, 1, 1), np.complex128))


def w_parents_and_children(c, *, ecut_w, metric, nphi=3, seed=17, covariant=False):
    """W = Σ_m φ_m λ_m φ_m† on the χ sphere (λ real: the conj rule), parents random (or closed
    under their little groups, ``covariant``), children by the r-space action (trivial spin)."""
    from common.gvec_fft_box import build_sphere_box_index
    fx = c["fx"]
    sfx = dict(fx, plan=_scalar_plan(fx["plan"]))
    kpar, kful = np.asarray(fx["plan"].k_parent_frac), np.asarray(fx["kfull"])
    wps, wpn = cases.spheres(kpar, metric, ecut_w)
    wcs, wcn = cases.spheres(kful, metric, ecut_w, width=wps.shape[1])
    rng = np.random.default_rng(seed)
    phi = (rng.standard_normal((len(kpar), nphi, 1, wps.shape[1]))
           + 1j * rng.standard_normal((len(kpar), nphi, 1, wps.shape[1])))
    phi *= (np.arange(wps.shape[1])[None, :] < wpn[:, None])[:, None, None, :]
    leak0 = 0.0
    if covariant:
        phi, leak0 = cases.close_under_little_groups(sfx, phi, wps, wpn, rows=fx["rows"],
                                                     spinor_action=None, ns=1)
    lam = np.tile(rng.standard_normal(nphi), phi.shape[1] // nphi)
    phic, leak = cases.children_psi(sfx, phi, wps, wpn, wcs, wcn)
    widx = build_sphere_box_index(wps, tuple(fx["fft_grid"]), wps.shape[1], ngk_valid=wpn)
    return dict(W_par=cases.green_from_psi(phi, lam), W=cases.green_from_psi(phic, lam),
                wps=wps, wpn=wpn, wsp=wcs, wngk=wcn, wfrac=kful, widx=widx,
                wleak=max(leak0, leak), splan=sfx["plan"])


def _sigma_typed_ops(c, w):
    from gw.mixed_basis_pair_convolution import PairOperand, SphereSet, SphereTransport
    fx = c["fx"]
    gs = SphereSet(c["sph"], c["ngk"], c["kfrac"])
    ws = SphereSet(w["wsp"], w["wngk"], w["wfrac"])
    g_tr = SphereTransport.typed(fx["plan"], fft_grid=fx["fft_grid"], parent_sphere_index=c["sidx"],
                                 children=gs)
    w_tr = SphereTransport.typed(fx["plan"], fft_grid=fx["fft_grid"], parent_sphere_index=w["widx"],
                                 children=ws, ns=1)
    return PairOperand(gs, g_tr), PairOperand(ws, w_tr)


def _sigma_symmetry_check(mesh, c, w, backend, *, wedge_rows=None, twins=(), ref=None):
    """Σ from typed parents (G at the k-parents, W at the q-IBZ) against the dense reference on
    full-grid children (``ref``: that reference, computed here when None; returned as ``r['ref_arr']``);
    with ``wedge_rows`` also the r'-wedge against the dense-column plan."""
    from gw.mixed_basis_pair_convolution import (ColumnWedge, PairOperand, SphereSet,
                                                SphereTransport)
    assert c["leak"] <= 1e-13 and w["wleak"] <= 1e-13, (c["leak"], w["wleak"])
    if ref is None:
        ref = _sigma_ref(dict(c, G=c["A"], W=w["W"], wsp=w["wsp"], wngk=w["wngk"], wfrac=w["wfrac"]))
    g_op, w_op = _sigma_typed_ops(c, w)
    kw = dict(backend=backend, budget_bytes=int(1e10))
    args = (mesh, c["kgrid"], c["fft_grid"])
    dense = _run_sigma(_sigma_conv(*args, g_op, w_op, c["out"], **kw), c["A_par"], w["W_par"])
    r = dict(parent=cases.rel(dense, ref), anti=bool(np.any(w_op.transport.anti)), red={}, ref_arr=ref)
    # W's own unfold (the conj rule) against the r-space children, on the tiles
    r["w_tile"] = cases.rel(cases.unfold_tile(w_op.transport, w["W_par"][:, :, 0, :, 0]),
                            w["W"][:, :, 0, :, 0])
    for name in twins:
        if name == "no_anti_W":
            t = w_op.transport
            bad = PairOperand(w_op.sphere, dataclasses.replace(t, anti=np.zeros_like(t.anti)))
            r["red"][name] = cases.rel(_run_sigma(_sigma_conv(*args, g_op, bad, c["out"], **kw),
                                                  c["A_par"], w["W_par"]), ref)
    if wedge_rows is not None:
        fx = c["fx"]
        ns = c["ns"]
        rows = np.asarray(wedge_rows)
        spin = None if ns == 1 else np.asarray(fx["spinor_action"](rows, nspinor=ns))
        wedge = ColumnWedge(np.asarray(fx["ops"]), np.asarray(fx["tnp"]), rows,
                            SphereSet(*c["out_full"]), spin)
        conv = _sigma_conv(*args, g_op, w_op, c["out"], wedge=wedge, **kw)
        got = _run_sigma(conv, c["A_par"], w["W_par"])
        r.update(wedge_ref=cases.rel(got, ref), wedge_dense=cases.rel(got, dense),
                 orbits=conv._wt["n_orbits"], nr=conv.nr)
        for name in set(twins) & set(EXPAND_TWINS):
            tw = expand_twin(_sigma_conv(*args, g_op, w_op, c["out"], wedge=wedge, **kw), name)
            r["red"][name] = cases.rel(_run_sigma(tw, c["A_par"], w["W_par"]), ref)
        if "no_spin" in twins and ns > 1:
            tw = _sigma_conv(*args, g_op, w_op, c["out"], wedge=wedge, **kw)
            d = list(tw._dev_rebuild)
            d[4] = tw._put(np.broadcast_to(np.eye(ns, dtype=np.complex128), (len(tw.wedge.rows), ns, ns)))
            tw._dev_rebuild = tuple(d)
            r["red"]["no_spin"] = cases.rel(_run_sigma(tw, c["A_par"], w["W_par"]), ref)
    return r


def sigma_glide_case(mesh, *, covariant):
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True,
                                   theta=np.pi / 2)
    if covariant:
        c = covariant_case(fx, ecut=1.3, metric=np.eye(3), box=(8, 8, 7))
        c["out"] = (*cases.spheres(c["kfrac"][[0, 1, 3]], np.eye(3), 1.3), c["kfrac"][[0, 1, 3]])
        c["out_full"] = (*cases.spheres(c["kfrac"], np.eye(3), 1.3), c["kfrac"])
    else:
        c = symmetry_case(fx, ecut=1.3, metric=np.eye(3), box=(8, 8, 7))
        c["out"] = (*cases.spheres(c["kfrac"][[0, 1, 3]], np.eye(3), 1.3), c["kfrac"][[0, 1, 3]])
    w = w_parents_and_children(c, ecut_w=1.0, metric=np.eye(3), covariant=covariant)
    return c, w


@pytest.mark.parametrize("backend", ["xla", "router"])
def test_sigma_glide_parents_equal_full_grid(backend):
    """Glide group, n_s = 2, spin mixing, an antiunitary row: G from its parents and W from its
    q-IBZ (the conj rule, trivial spin) against the dense supercell reference on full-grid input."""
    mesh = _mesh(4)
    c, w = sigma_glide_case(mesh, covariant=False)
    r = _sigma_symmetry_check(mesh, c, w, backend, twins=("no_anti_W",))
    assert r["anti"] and r["parent"] <= TOL and r["w_tile"] <= 1e-13, r
    assert r["red"]["no_anti_W"] > 1e-3, r


@pytest.mark.parametrize("backend", ["xla", "router"])
@pytest.mark.parametrize("rows", [(0, 1, 2, 3), (0, 3)])
def test_sigma_wedge_glide_equals_dense(backend, rows):
    """The r'-wedge on Σ: covariant G and W, the output's spin sandwich live (θ = π/2 mixes spin);
    dropping it (U = 1) misses."""
    mesh = _mesh(4)
    c, w = sigma_glide_case(mesh, covariant=True)
    r = _sigma_symmetry_check(mesh, c, w, backend, wedge_rows=rows, twins=("no_spin",))
    assert r["parent"] <= TOL and r["wedge_ref"] <= TOL and r["wedge_dense"] <= TOL, r
    assert r["orbits"] < r["nr"] and r["red"]["no_spin"] > 1e-3, r


def test_sigma_wedge_needs_spin():
    from gw.mixed_basis_pair_convolution import ColumnWedge, SphereSet
    mesh = _mesh(1)
    c, w = sigma_glide_case(mesh, covariant=True)
    g_op, w_op = _sigma_typed_ops(c, w)
    fx = c["fx"]
    wedge = ColumnWedge(np.asarray(fx["ops"]), np.asarray(fx["tnp"]), (0, 1), SphereSet(*c["out_full"]))
    with pytest.raises(ValueError, match="GATE pairconv-wedge-spin"):
        _sigma_conv(mesh, c["kgrid"], c["fft_grid"], g_op, w_op, c["out"], wedge=wedge,
                    backend="xla", budget_bytes=int(1e10))


def conj_rule_check(mesh, c, *, ecut_w, metric, backend="xla"):
    """W's antiunitary rule at fixed τ, measured: χ₀(τ) of covariant Greens at every full-grid q
    against the typed one-channel unfold (conj rule, ``cases.unfold_tile``) of χ₀ at the parents;
    the red twin reads the antiunitary rows without the conjugation."""
    from common.gvec_fft_box import build_sphere_box_index
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    fx = c["fx"]
    kpar, kful = np.asarray(fx["plan"].k_parent_frac), np.asarray(fx["kfull"])
    wps, wpn = cases.spheres(kpar, metric, ecut_w)
    wcs, wcn = cases.spheres(kful, metric, ecut_w, width=wps.shape[1])
    tr = SphereTransport.identity(SphereSet(c["sph"], c["ngk"], c["kfrac"]), c["ns"])
    kw = dict(transport=tr, backend=backend, budget_bytes=int(1e10))
    x_full = _run(_conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"],
                        (wcs, wcn, kful), **kw), c["A"], c["C"])
    x_par = _run(_conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"],
                       (wps, wpn, kpar), **kw), c["A"], c["C"])
    widx = build_sphere_box_index(wps, tuple(fx["fft_grid"]), wps.shape[1], ngk_valid=wpn)
    w_tr = SphereTransport.typed(fx["plan"], fft_grid=fx["fft_grid"], parent_sphere_index=widx,
                                 children=SphereSet(wcs, wcn, kful), ns=1)
    return dict(rule=cases.rel(cases.unfold_tile(w_tr, x_par), x_full),
                red=cases.rel(cases.unfold_tile(w_tr, x_par, anti_conj=False), x_full),
                n_anti=int(np.sum(w_tr.anti)))


def test_conj_rule_on_chi_glide():
    """χ₀(τ) of covariant Greens (glide, n_s = 2, spin mixing, antiunitary rows) unfolds from the
    parents by the conj rule with the trivial spin action; without the conjugation it misses."""
    mesh = _mesh(1)
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True,
                                   theta=np.pi / 2)
    c = covariant_case(fx, ecut=1.3, metric=np.eye(3), box=(6, 6, 5))
    r = conj_rule_check(mesh, c, ecut_w=1.0, metric=np.eye(3))
    assert r["n_anti"] > 0 and r["rule"] <= TOL and r["red"] > 1e-3, r
