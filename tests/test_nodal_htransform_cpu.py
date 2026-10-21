"""Nodal htransform: covariance, accuracy, the WS code form, the metric, the back-map (CPU).

A face-centred-cubic toy with general orbits of 48 sites (orbit-closed node
sets that the home cell folds NON-uniformly) and a distance-only hopping
model.  The node-pair kernel ``fH`` of the lowest bands, continued off the
4^3 grid with per-pair Wigner-Seitz images and diagonalized directly, gives
star-invariant energies to rounding; a lattice-box R set (the former
htransform R grid) does not.  On the coarse grid it reproduces f(eps)
exactly; off the grid it is closer to the exact bands than its control
variants (sign-flipped pair displacement, one lattice image set per class,
the box).  The orbit-tied quadrature metric and the Loewdin step make
sampled nodal values exactly orthonormal, and the weights are one per orbit.
The top-block subspace solve returns the full eigensolver's window off the
grid with a block far below the rank, and stopping it at the power steps
does not.  The back-map recovers the states at targets that are not the
nodes, through a non-uniform metric (G != 1): exactly on the grid, and off
the grid better than with the wrong target geometry.

Covariance is exact only when the window top lies in a gap at every coarse k
(the precondition ``common.band_degeneracy.check_band_window`` reports): a
window that cuts a degenerate multiplet makes the coarse kernels
gauge-dependent.  Here 15 bands end in a 0.039 Ry gap; 14 would cut one.
"""
import functools
import itertools

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from bandstructure.fh_interp import (TOP_BLOCK_PAD, TOP_BLOCK_POWER,  # noqa: E402
                                     TOP_BLOCK_REFUSE, coarse_start_block,
                                     f_transform_eigs, log_top_block,
                                     newton_inv, top_block_eigh)
from common.ws_images import (pair_ws_code, pair_ws_code_fourier,  # noqa: E402
                              pair_ws_code_table, pair_ws_fourier,
                              pair_ws_images, pair_ws_weights)
from isdf.galerkin import _loewdin, _quadrature_metric  # noqa: E402

A = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]) * 10.0
KGRID = (4, 4, 4)
N_WINDOW, N_GUARD = 10, 5


def _oh():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            S = np.zeros((3, 3))
            S[np.arange(3), perm] = signs
            ops.append(S)
    return ops


def _orbit(x0_cart):
    """Distinct images of one Cartesian point, folded into [0,1) crystal."""
    inv = np.linalg.inv(A)
    pts = []
    for S in _oh():
        f = np.mod((S @ x0_cart) @ inv, 1.0)
        f[np.isclose(f, 1.0)] = 0.0
        if not any(np.allclose(f, p) for p in pts):
            pts.append(f)
    return np.array(pts)


ORBIT_POINTS = (np.array([0.61, 0.23, 1.37]), np.array([1.83, 0.52, 0.29]),
                np.array([0.27, 2.41, 0.94]))
SITES = _orbit(ORBIT_POINTS[0])
RY_TO_MEV = 13605.693


def _sites(n_orbits):
    """``n_orbits`` general orbits, concatenated, and each site's orbit."""
    orbits = [_orbit(p) for p in ORBIT_POINTS[:n_orbits]]
    return (np.concatenate(orbits),
            np.repeat(np.arange(n_orbits), [len(o) for o in orbits]))


def _bloch(points, k_frac, fn, sites=None):
    """``M(k)_ij = sum_R e^{-2 pi i k.R} fn(|p_i + R - r_j|)`` for points
    ``p`` against the sites (the toy's one Bloch-sum convention)."""
    sites = SITES if sites is None else sites
    R = np.array(list(itertools.product(range(-3, 4), repeat=3)), float)
    d = points[:, None, None, :] + R[None, :, None, :] - sites[None, None, :, :]
    ph = np.exp(-2j * np.pi * (np.atleast_2d(k_frac) @ R.T))
    return np.einsum('qr,irj->qij', ph, fn(np.linalg.norm(d @ A, axis=-1)))


def _hamiltonian(k_frac, sites=None, onsite=None):
    """``H(k)_ij = sum_R e^{-2 pi i k.R} t(|r_i + R - r_j|)`` (Ry), plus an
    optional on-site term."""
    sites = SITES if sites is None else sites
    H = _bloch(sites, k_frac, lambda r: -0.4 * np.exp(-(r / 2.2) ** 2)
               * (r < 9.0), sites)
    if onsite is not None:
        H = H + np.diag(onsite)[None]
    return 0.5 * (H + np.conj(np.swapaxes(H, -1, -2)))


def _coarse_k():
    return np.stack(np.meshgrid(*[np.arange(n) / n for n in KGRID],
                                indexing='ij'), -1).reshape(-1, 3)


def _kernel(sites=None, onsite=None, nb=N_WINDOW + N_GUARD):
    """Coarse fH kernel of orthonormal nodal coefficients, class-major R."""
    E, C = np.linalg.eigh(_hamiltonian(_coarse_k(), sites, onsite))
    E, C = E[:, :nb], C[:, :, :nb]                  # C_k[nu, n]
    f_eps, a, n, shift = f_transform_eigs(E.T)
    f = np.asarray(f_eps).T
    X_k = np.einsum('kin,kn,kjn->kij', C, f, np.conj(C))
    X = np.fft.ifftn(X_k.reshape(KGRID + X_k.shape[1:]), axes=(0, 1, 2))
    return X.reshape((-1,) + X_k.shape[1:]), (a, n, shift), E, nb


@functools.lru_cache(maxsize=None)
def _ws_tables():
    images = pair_ws_images(KGRID, A, SITES, 3)
    return images, pair_ws_code_table(images)


@functools.lru_cache(maxsize=None)
def _default_kernel():
    """``_kernel()`` at its defaults, built once per process (read-only)."""
    return _kernel()


def _box(X_R, k):
    axes = [np.where(np.arange(m) >= (m + 1) // 2, np.arange(m) - m,
                     np.arange(m)) for m in KGRID]
    R = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    return np.einsum('qc,cij->qij', np.exp(-2j * np.pi * k @ R.T), X_R)


def _energies(fH_R, k, params, nb, fourier):
    X = np.asarray(fourier(fH_R, k))
    X = 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))
    values = np.linalg.eigvalsh(X)[:, :nb]          # f <= 0 ascending
    E, _res = newton_inv(*params, values)
    return np.asarray(E)[:, :N_WINDOW]


def _star_spread(fourier):
    fH_R, params, _, nb = _default_kernel()
    rng = np.random.default_rng(3)
    k0 = rng.random((6, 3))
    inv = np.linalg.inv(A)
    E0 = _energies(fH_R, k0, params, nb, fourier)
    worst = 0.0
    for S in _oh():
        M = A @ S.T @ inv                            # R_frac -> R_frac @ M
        kS = k0 @ np.linalg.inv(M).T
        ES = _energies(fH_R, kS, params, nb, fourier)
        worst = max(worst, float(np.abs(ES - E0).max()))
    return worst


def _check_node_set_folds_non_uniformly():
    inv = np.linalg.inv(A)
    wraps = {tuple(np.floor((S @ (SITES[0] @ A)) @ inv + 1e-9).astype(int))
             for S in _oh()}
    assert SITES.shape == (48, 3) and len(wraps) > 1


def _check_direct_eigh_is_exact_on_the_coarse_grid():
    fH_R, params, E, nb = _default_kernel()
    _, table = _ws_tables()
    got = _energies(fH_R, _coarse_k(), params, nb,
                    lambda X, k: pair_ws_code_fourier(X, table, k))
    np.testing.assert_allclose(got, E[:, :N_WINDOW], atol=1e-10)


def _check_ws_kernel_is_star_covariant_and_the_box_is_not():
    _, table = _ws_tables()
    ws = _star_spread(lambda X, k: pair_ws_code_fourier(X, table, k))
    box = _star_spread(_box)
    assert ws < 1e-10, ws                            # Ry: rounding
    assert box > 1e-4, box                           # the control fails


def _off_grid_error(fourier):
    """max |E - E_exact| (meV) of the window at random off-grid q."""
    fH_R, params, _, nb = _default_kernel()
    q = np.random.default_rng(13).random((8, 3)) - 0.5
    got = _energies(fH_R, q, params, nb, fourier)
    exact = np.linalg.eigvalsh(_hamiltonian(q))[:, :N_WINDOW]
    return RY_TO_MEV * float(np.abs(got - exact).max())


def _check_per_pair_ws_is_closer_off_the_grid_than_its_control_variants():
    """At 4^3 the per-pair images beat the sign-flipped displacement
    (d -> -d), one lattice image set per class and the box by >= 3x; every
    variant but the box is star-covariant, so only this test tells them
    apart (the audit's table: 2.16, 37.9, 10.1, 15.8 meV)."""
    images, table = _ws_tables()
    n = SITES.shape[0]

    def per_pair(img):
        return lambda X, k: pair_ws_fourier(
            np.moveaxis(X, 0, -1).reshape((n, n) + KGRID), img, k)

    err = {
        "per-pair": _off_grid_error(
            lambda X, k: pair_ws_code_fourier(X, table, k)),
        "sign-flipped": _off_grid_error(
            per_pair(dict(images, pos=-images["pos"]))),
        "lattice-only": _off_grid_error(
            per_pair(dict(images, pos=np.zeros_like(images["pos"])))),
        "box": _off_grid_error(_box),
    }
    assert err["per-pair"] < 4.0, err
    for variant in ("sign-flipped", "lattice-only", "box"):
        assert err[variant] > 3.0 * err["per-pair"], (variant, err)


def _check_quadrature_metric_and_loewdin_orthonormalize_sampled_values():
    """Phi = U / sqrt(v) with v constant on each orbit: NNLS returns w = v,
    the Gram is the identity, and C = W^{1/2} Phi G^{-1/2} is orthonormal."""
    rng = np.random.default_rng(11)
    nk, nb, n_mu = 9, 5, 12
    labels = np.repeat(np.arange(3), 4)
    v = np.array([0.5, 2.0, 1.3])[labels]
    U = np.linalg.qr(rng.standard_normal((nk, 2 * n_mu, nb))
                     + 1j * rng.standard_normal((nk, 2 * n_mu, nb)))[0]
    phi = np.moveaxis(U.reshape(nk, 2, n_mu, nb), -1, 1) / np.sqrt(v)
    w, gram = _quadrature_metric(phi, labels)        # phi (nk, nb, ns, n_mu)
    np.testing.assert_allclose(w, v, rtol=1e-10)
    inv_half, eps = _loewdin(gram)
    assert eps < 1e-10
    C = np.einsum('kmn,kmsv,v->knsv', inv_half, phi, np.sqrt(w)).reshape(
        nk, nb, -1)
    np.testing.assert_allclose(np.einsum('kna,kma->knm', np.conj(C), C),
                               np.broadcast_to(np.eye(nb), (nk, nb, nb)),
                               atol=1e-12)


def _check_quadrature_metric_weights_are_one_per_orbit():
    """On values no metric makes exactly orthonormal, ``w`` is constant on
    every orbit and is the NNLS of the orbit-folded least-squares system
    ``sum_nu w_nu Phi^H Phi = 1``; the per-node NNLS (the control: a metric
    that breaks the star) is not orbit-constant."""
    from scipy.optimize import nnls
    rng = np.random.default_rng(17)
    nk, nb, n_mu = 7, 4, 12
    labels = np.repeat(np.arange(3), 4)
    phi = (rng.standard_normal((nk, nb, 1, n_mu))
           + 1j * rng.standard_normal((nk, nb, 1, n_mu))) / np.sqrt(n_mu)
    w, gram = _quadrature_metric(phi, labels)
    rows = np.einsum('kmsv,knsv->kmnv', np.conj(phi), phi).reshape(-1, n_mu)
    lhs = np.concatenate([rows.real, rows.imag])
    rhs = np.concatenate([np.broadcast_to(np.eye(nb), (nk, nb, nb)).ravel(),
                          np.zeros(rows.shape[0])])
    fold = np.eye(3)[labels]                         # node -> orbit
    np.testing.assert_allclose(w, fold @ nnls(lhs @ fold, rhs)[0], rtol=1e-8)
    assert all(np.ptp(w[labels == o]) == 0.0 for o in range(3)), w
    free = nnls(lhs, rhs)[0]
    assert max(np.ptp(free[labels == o]) for o in range(3)) > 1e-2, free
    np.testing.assert_allclose(
        gram, np.einsum('kmsv,v,knsv->kmn', np.conj(phi), w, phi), atol=1e-12)


def _back_map_states(q, sites, labels, targets, onsite, nb, scale):
    """Exact values of the toy's lowest ``nb`` states: smeared, per-orbit
    scaled at the nodes (``Phi``, a non-unitary sampling) and smeared at the
    targets, ``(nq, nb, n)`` each, in one eigenvector gauge."""
    _, U = np.linalg.eigh(_hamiltonian(q, sites, onsite))
    U = U[:, :, :nb]
    smear = lambda r: np.exp(-(r / 1.1) ** 2) * (r < 6.0)
    nodes = scale[labels][None, :, None] * (_bloch(sites, q, smear, sites) @ U)
    at_t = _bloch(targets, q, smear, sites) @ U
    return np.swapaxes(nodes, 1, 2), np.swapaxes(at_t, 1, 2)


def _back_map_toy(scale=(1.0, 2.5)):
    """The back-map toy of the test above: two orbits (96 nodes) with a
    smeared, per-orbit-scaled sampling, 16 targets, 16 fitted / 9 returned
    bands, 4 coarse and 6 off-grid q.  One orbit per entry of ``scale``."""
    sites, labels = _sites(len(scale))
    onsite, nb = 0.05 * labels, 16
    targets = np.random.default_rng(23).random((16, 3))
    k = _coarse_k()
    E = np.linalg.eigvalsh(_hamiltonian(k, sites, onsite))[:, :nb]
    phi, at_t = _back_map_states(k, sites, labels, targets, onsite, nb,
                                 np.asarray(scale))
    q = np.concatenate([k[[0, 7, 22, 41]],
                        np.random.default_rng(31).random((6, 3)) - 0.5])
    return sites, labels, targets, E, phi, at_t, q


def _fine(ctilde, interp, E, q, n_win, **kw):
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from bandstructure.bse_setup import compute_wfns_fi
    from ffi.fft import xla_reference
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    with xla_reference():
        fine = compute_wfns_fi(
            ctilde=jnp.asarray(ctilde), B_at_mu=interp,
            enk_sigma=jnp.asarray(E.T), kgrid_co=KGRID, q_list=q,
            band_window_fi=(0, n_win), mesh_xy=mesh, **kw)
    psi = np.asarray(fine.psi_rmu_Y)[:, :, 0, :]
    return (np.asarray(fine.enk_full),
            np.einsum('qnt,qns->qts', psi, np.conj(psi)))


def _check_dropping_zero_weight_nodes_is_exact():
    """A node orbit with w = 0 has exactly zero rows of C = W^{1/2} Phi
    G^{-1/2}; dropping it from the rank axis, with every node kept as a
    back-map target, leaves the energies and the window projector at the
    targets unchanged up to roundoff: 4 ulp of max |E| per node on the
    rank axis (144 nodes, max |E| 18.4 Ry: 2.4e-12 Ry; measured 3.0 ulp,
    1.2e-14 Ry).  Three orbits, the middle one at w = 0: the two kept
    orbits resolve the 16 bands (min/max eigenvalue of G >= 0.046); one
    orbit alone does not (the red twin below)."""
    import jax.numpy as jnp
    import pytest
    from isdf.galerkin import NodalInterp
    if jax.device_count() < 4:
        pytest.skip("needs 4 host devices")
    sites, labels, _, E, phi, _, q = _back_map_toy((1.0, 2.5, 1.6))
    w, _ = _quadrature_metric(phi[:, :, None, :], labels)
    w = np.where(labels == 1, 0.0, w)                # orbit 2: zero weight
    inv_half, _ = _loewdin(np.einsum('kmv,v,knv->kmn', np.conj(phi), w, phi))
    ctilde = np.einsum('kmn,kmv,v->knv', inv_half, phi, np.sqrt(w))
    keep = w > 0.0
    assert keep.sum() == 96 and not np.any(ctilde[:, :, ~keep])
    e_all, p_all = _fine(ctilde, NodalInterp(jnp.asarray(phi), sites, A, 3, 1),
                         E, q, 9)
    e_kept, p_kept = _fine(
        ctilde[:, :, keep],
        NodalInterp(jnp.asarray(phi[:, :, keep]), sites[keep], A, 3, 1,
                    target=jnp.asarray(phi), target_frac=sites), E, q, 9)
    d_e = float(np.abs(e_kept - e_all).max())
    tol_e = 4 * phi.shape[-1] * np.finfo(np.float64).eps * float(np.abs(E).max())
    assert d_e <= tol_e, (RY_TO_MEV * d_e, RY_TO_MEV * tol_e)   # meV
    assert np.abs(p_kept - p_all).max() <= 1e-12 * np.abs(p_all).max()


def _check_loewdin_refuses_a_gram_singular_to_roundoff():
    """Red twin of the drop test: on the two-orbit toy one orbit alone does
    not resolve the 16 bands (rank 15 at Gamma, min/max eigenvalue ~1e-16,
    either sign run to run), so the drop leaves a singular Gram and
    ``_loewdin`` refuses it whichever side of zero roundoff puts it."""
    import pytest
    _, labels, _, _, phi, _, _ = _back_map_toy()
    w, _ = _quadrature_metric(phi[:, :, None, :], labels)
    w = np.where(labels == 1, 0.0, w)
    with pytest.raises(ValueError, match="not positive definite"):
        _loewdin(np.einsum('kmv,v,knv->kmn', np.conj(phi), w, phi))


def _check_face_route_matches_the_q_batched_route():
    """``compute_wfns_fi`` on the face route (``use_low_mem_eigh``, the
    nmu2 mesh backup) and on the q-batched route agree to 1e-12 on the
    back-map toy (2x2 host mesh): energies and window projectors."""
    import jax.numpy as jnp
    import pytest
    from isdf.galerkin import NodalInterp
    if jax.device_count() < 4:
        pytest.skip("needs 4 host devices")
    sites, labels, targets, E, phi, at_t, q = _back_map_toy()
    w, gram = _quadrature_metric(phi[:, :, None, :], labels)
    inv_half, _ = _loewdin(gram)
    ctilde = np.einsum('kmn,kmv,v->knv', inv_half, phi, np.sqrt(w))
    interp = lambda: NodalInterp(jnp.asarray(phi), sites, A, 3, 1,
                                 target=jnp.asarray(at_t), target_frac=targets)
    e_q, p_q = _fine(ctilde, interp(), E, q, 9)
    e_f, p_f = _fine(ctilde, interp(), E, q, 9, use_low_mem_eigh=True)
    assert np.abs(e_f - e_q).max() <= 1e-12 * np.abs(e_q).max()
    assert np.abs(p_f - p_q).max() <= 1e-12 * np.abs(p_q).max()


# ---------------------------------------------------------------------------
# The tests.  The ``_check_*`` helpers above are the individual properties,
# grouped by what they protect; each group is one test so a failure names the
# invariant and the first violated property.
# ---------------------------------------------------------------------------

def test_ws_kernel_is_exact_on_grid_star_covariant_and_beats_its_controls():
    """The direct eigh of the WS-continued node-pair kernel reproduces f(eps)
    exactly on the coarse grid, is star-invariant to rounding where the box
    is not, and off the grid is closer to the exact bands than the
    sign-flipped, lattice-only and box controls.  The hsuite path is on the
    grid at its corners and cannot see the last two."""
    _check_node_set_folds_non_uniformly()      # the premise: the box must fail
    _check_direct_eigh_is_exact_on_the_coarse_grid()
    _check_ws_kernel_is_star_covariant_and_the_box_is_not()
    _check_per_pair_ws_is_closer_off_the_grid_than_its_control_variants()


def test_quadrature_metric_loewdin_and_its_refusal():
    """The orbit-tied NNLS metric and the Loewdin step make sampled nodal
    values exactly orthonormal, the weights are one per orbit (and the
    per-node control is not), and a Gram singular to roundoff is refused
    either side of zero."""
    _check_quadrature_metric_and_loewdin_orthonormalize_sampled_values()
    _check_quadrature_metric_weights_are_one_per_orbit()
    _check_loewdin_refuses_a_gram_singular_to_roundoff()


def test_back_map_routes_and_zero_weight_drop():
    """On the back-map toy: dropping a zero-weight orbit from the rank axis
    leaves the energies and window projectors at the targets unchanged to
    roundoff, and the face route (``use_low_mem_eigh``) agrees with the
    q-batched route to 1e-12.  The hsuite takes the q-batched route only
    (``linalg = local``)."""
    _check_dropping_zero_weight_nodes_is_exact()
    _check_face_route_matches_the_q_batched_route()


def test_fit_refuses_non_orbit_closed_nodes():
    """A node (or ISDF centroid) set that a spatial operation maps off itself
    is refused before anything is sampled, naming the remedy; the set with its
    image is one orbit.  The hsuite's k-means always produces closed sets, so
    this refusal is reached nowhere else."""
    import types
    import pytest
    from isdf import galerkin
    sym = types.SimpleNamespace(
        sym_matrices=np.array([np.eye(3, dtype=int), np.diag([-1, 1, 1])]),
        translations=np.zeros((2, 3)))
    meta = types.SimpleNamespace(nspinor=1, mu_basis=None, fft_grid=(6, 6, 6))
    closed = np.array([[1, 2, 3], [5, 2, 3]])          # x -> -x mod 6
    labels, n_ops = galerkin._orbit_labels(closed, sym, meta, "nodes")
    assert n_ops == 2 and np.unique(labels).size == 1
    with pytest.raises(ValueError, match=r"not orbit-closed.*--orbit"):
        galerkin._orbit_labels(closed[:1], sym, meta, "nodes")
    fit = lambda **kw: galerkin.fit_galerkin_basis(
        None, sym, meta, kw.pop("centroids"), None, (0, 2), **kw)
    with pytest.raises(ValueError, match="the nodes are not orbit-closed"):
        fit(centroids=closed[:1])
    with pytest.raises(ValueError, match="ISDF centroids .targets. are not"):
        fit(centroids=closed[:1], node_indices=closed)
