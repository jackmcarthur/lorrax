"""The Sigma term of the QSGW head velocity against exact commutators.

``gw.qsgw_head.sigma_commutator`` (the route of every
``sc_head_update = parallel_transport`` map) and ``scissor_capture`` (the
self-test each SC run prints) on a spinful tight-binding model whose
commutators are known in closed form: orbitals at the cell origin, so
``i[O, r] = d_k O(k)`` exactly for any ``O(k) = sum_R e^{ik.R} o_R``, on a
coarse 4x4x4 mesh with second-neighbour hops, where the finite links recover
only part of a nonlocal term (``docs/theory/qp-velocity.md`` section 4a).

1. Scissor ``DeltaH = Delta P_c``: exact ``v_vc Delta/(E_c - E_v)``.  The
   links alone (the route before the band-diagonal split) recover well under
   all of it; the route recovers all of it.
2. A scissor on slightly rotated states (small cross-gap mixing): the
   route is much closer to the exact commutator than the links alone; a
   purely off-diagonal ``DeltaH`` takes the links unchanged.
3. Exact doublets (spin): the route is covariant under a random frame
   rotation inside every doublet at every k.

CPU, 2x2 host mesh: ``JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import numpy as np
import pytest

GRID = (4, 4, 4)
NSP = 4                      # spatial orbitals; x2 spin -> 8 bands
NOCC = 4


def _mesh():
    import jax
    from jax.sharding import Mesh
    jax.config.update("jax_enable_x64", True)
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host CPU devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _hops(rng, scale_1, scale_2):
    """Half set of hops ``R -> t_R`` (the h.c. supplies ``-R``)."""
    hops = {}
    for a in range(3):
        for n, s in ((1, scale_1), (2, scale_2)):
            R = np.zeros(3, dtype=int)
            R[a] = n
            t = rng.normal(size=(NSP, NSP)) + 1j * rng.normal(size=(NSP, NSP))
            hops[tuple(R)] = s * t
    return hops


def _bloch(onsite, hops, kc):
    """``O(k)`` and ``d_k O(k)`` (3, ...) of ``onsite + sum_R t_R e^{ikR} + h.c.``."""
    O = onsite.astype(complex).copy()
    dO = np.zeros((3,) + onsite.shape, dtype=complex)
    for R, t in hops.items():
        ph = np.exp(1j * float(np.dot(kc, R)))
        O += t * ph + t.conj().T * np.conj(ph)
        for a in range(3):
            dO[a] += 1j * R[a] * (t * ph - t.conj().T * np.conj(ph))
    return O, dO


def _rotated_scissor(onsite, hops, loc, kc, shift=0.6):
    """``shift * P_c[H(k) + loc]``: a scissor on states rotated by ``loc``."""
    H, _ = _bloch(onsite, hops, kc)
    _, c = np.linalg.eigh(H + loc)
    return shift * (c[:, NOCC // 2:] @ c[:, NOCC // 2:].conj().T)


def _model(seed=7, gauge_seed=None, mixing=0.06):
    rng = np.random.default_rng(seed)
    onsite = np.diag([-2.0, -1.6, 1.6, 2.1]).astype(complex)
    m = 0.15 * (rng.normal(size=(NSP, NSP)) + 1j * rng.normal(size=(NSP, NSP)))
    onsite = onsite + m + m.conj().T
    hops = _hops(rng, 0.30, 0.12)
    # DeltaH: a scissor on the conduction states of H + loc, i.e. a rigid
    # shift that also rotates the states (cross-gap mixing ~ ``mixing``).
    loc = mixing * (rng.normal(size=(NSP, NSP)) + 1j * rng.normal(size=(NSP, NSP)))
    loc = loc + loc.conj().T
    B = 2.0 * np.pi * np.eye(3)                    # simple cubic, a = 1
    kap = np.array([[i, j, l] for i in range(GRID[0]) for j in range(GRID[1])
                    for l in range(GRID[2])], dtype=float) / np.asarray(GRID)
    nk, nb = len(kap), 2 * NSP
    spin = np.eye(2)
    grng = None if gauge_seed is None else np.random.default_rng(gauge_seed)
    C = np.zeros((nk, nb, nb), complex)
    E = np.zeros((nk, nb))
    v = np.zeros((3, nk, nb, nb), complex)
    dH = np.zeros((nk, nb, nb), complex)
    dH_exact = np.zeros((3, nk, nb, nb), complex)
    G = np.zeros((nk, nb, nb), complex)
    for k, kk in enumerate(kap):
        kc = kk @ B
        H, dHk = _bloch(onsite, hops, kc)
        D = _rotated_scissor(onsite, hops, loc, kc)
        dD = np.zeros((3, NSP, NSP), complex)
        for a in range(3):                         # analytic in k: a fine difference
            h = np.zeros(3)
            h[a] = 1.0e-5
            dD[a] = (_rotated_scissor(onsite, hops, loc, kc + h)
                     - _rotated_scissor(onsite, hops, loc, kc - h)) / 2.0e-5
        e, c = np.linalg.eigh(np.kron(H, spin))
        g = np.eye(nb, dtype=complex)
        if grng is not None:                       # random frame inside each doublet
            for p in range(0, nb, 2):
                z = grng.normal(size=(2, 2)) + 1j * grng.normal(size=(2, 2))
                q, _ = np.linalg.qr(z)
                g[p:p + 2, p:p + 2] = q
        c = c @ g
        C[k], E[k], G[k] = c, e, g
        for a in range(3):
            v[a, k] = c.conj().T @ np.kron(dHk[a], spin) @ c
            dH_exact[a, k] = c.conj().T @ np.kron(dD[a], spin) @ c
        dH[k] = c.conj().T @ np.kron(D, spin) @ c
    assert np.min(E[:, NOCC] - E[:, NOCC - 1]) > 0.3       # an insulator
    assert np.max(E[:, 1::2] - E[:, 0::2]) < 1e-10          # exact doublets
    from common.parallel_transport import link_stencil
    st = link_stencil(GRID, B)
    idx = lambda t: (t[0] % GRID[0]) * GRID[1] * GRID[2] + (t[1] % GRID[1]) * GRID[2] + t[2] % GRID[2]
    ints = np.rint(kap * np.asarray(GRID)).astype(int)
    plus = np.array([[idx(ints[k] + s) for s in st.steps] for k in range(nk)],
                    dtype=np.int64)
    links = np.zeros((st.ndir, nk, nb, nb), complex)
    for d in range(st.ndir):
        for k in range(nk):
            u, _, wh = np.linalg.svd(C[k].conj().T @ C[plus[k, d]])
            links[d, k] = u @ wh
    occ = np.zeros((nk, nb))
    occ[:, :NOCC] = 1.0
    return dict(B=B, E=E, v=v, dH=dH, dH_exact=dH_exact, G=G, links=links,
                plus=plus, occ=occ, nk=nk, nb=nb)


def _put(mesh, x, spec):
    import jax
    from jax.sharding import NamedSharding
    return jax.device_put(x, NamedSharding(mesh, spec))


def _source(mesh, m):
    from jax.sharding import PartitionSpec as P
    from gw.qsgw_head import ParallelTransportHeadData
    return ParallelTransportHeadData(
        forward_links=_put(mesh, m["links"], P(None, None, "x", "y")),
        forward_neighbors=m["plus"],
        velocity_dft_cart=_put(mesh, m["v"], P(None, None, "x", "y")),
        nb_logical=m["nb"], reciprocal_lattice_cart=m["B"], validation={},
        singular_values=None, nb_links=m["nb"],
        energies_dft_kn_ry=m["E"], occupations_dft_kn=m["occ"])


def _routes(mesh, m, delta):
    """(route, links alone) on DeltaH = ``delta`` (nk, nb, nb)."""
    import jax
    from jax.sharding import PartitionSpec as P
    from gw.qsgw_head import covariant_link_derivative, sigma_commutator
    src = _source(mesh, m)
    d = _put(mesh, delta, P(None, "x", "y"))
    kw = dict(mesh=mesh, kgrid=GRID, bvec_cart=m["B"])
    route, link_part, frac = sigma_commutator(
        d, src.velocity_dft_cart, m["E"], src.forward_links, m["plus"],
        nb_logical=m["nb"], nb_links=m["nb"], **kw)
    links = covariant_link_derivative(d, src.forward_links, m["plus"], **kw)
    return (np.asarray(jax.device_get(route)), np.asarray(jax.device_get(links)),
            np.asarray(jax.device_get(link_part)), frac)


def _head_error(m, x, exact):
    """Relative weighted error on valence-conduction pairs, w = 1/dE^3."""
    E = m["E"]
    de = E[:, None, NOCC:] - E[:, :NOCC, None]
    w = (1.0 / de ** 3)[None]
    a, b = x[:, :, :NOCC, NOCC:], exact[:, :, :NOCC, NOCC:]
    return float(np.sqrt(np.sum(w * np.abs(a - b) ** 2) / np.sum(w * np.abs(b) ** 2)))


def test_scissor_links_fall_short_and_the_route_is_exact():
    from gw.qsgw_head import scissor_capture
    mesh = _mesh()
    m = _model()
    # The links alone keep the sign and miss most of a scissor's commutator
    # on this mesh (Si 4^3: 0.32); the route recovers all of it.
    f_links, f_route = scissor_capture(_source(mesh, m), mesh=mesh, kgrid=GRID)
    assert 0.0 < f_links < 0.5, f_links
    assert abs(f_route - 1.0) < 1e-10, f_route


def test_rotating_scissor_closer_to_exact_and_off_diagonal_unchanged():
    mesh = _mesh()
    m = _model()
    # A scissor on rotated states, first-order cross-gap mixing theta = 0.12:
    # the remainder R rides the links and carries an error ~ 2 theta (the
    # split stops paying near theta ~ 0.5 on this mesh,
    # docs/theory/qp-velocity.md section 4a); the links alone miss ~ 90 %.
    route, links, _, frac = _routes(mesh, m, m["dH"])
    assert 0.0 < frac < 1.0
    err_route = _head_error(m, route, m["dH_exact"])
    err_links = _head_error(m, links, m["dH_exact"])
    assert err_links > 0.8 and err_route < 0.3, (err_route, err_links)
    # No band diagonal: D = 0 and the route is the links, bit for bit up to
    # round-off.
    off = m["dH"] - np.einsum("kn,nm->knm", np.einsum("knn->kn", m["dH"]),
                              np.eye(m["nb"]))
    route0, links0, link_part0, frac0 = _routes(mesh, m, off)
    np.testing.assert_allclose(route0, links0, atol=1e-12)
    np.testing.assert_allclose(link_part0, links0, atol=1e-12)
    assert abs(frac0 - 1.0) < 1e-12


def test_route_is_covariant_inside_exact_doublets():
    mesh = _mesh()
    m0, m1 = _model(), _model(gauge_seed=3)
    r0, _, _, f0 = _routes(mesh, m0, m0["dH"])
    r1, _, _, f1 = _routes(mesh, m1, m1["dH"])
    # m1's frame is m0's times G (block-diagonal in the doublets), the model
    # being the same; so r1 = G^H r0 G at every k.
    G = m1["G"]
    back = np.einsum("kmi,akmn,knj->akij", G.conj(), r0, G)
    np.testing.assert_allclose(r1, back, atol=1e-10)
    assert abs(f0 - f1) < 1e-12
