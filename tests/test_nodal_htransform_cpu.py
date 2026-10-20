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


def _ws_tables():
    images = pair_ws_images(KGRID, A, SITES, 3)
    return images, pair_ws_code_table(images)


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
    fH_R, params, _, nb = _kernel()
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


def test_node_set_folds_non_uniformly():
    inv = np.linalg.inv(A)
    wraps = {tuple(np.floor((S @ (SITES[0] @ A)) @ inv + 1e-9).astype(int))
             for S in _oh()}
    assert SITES.shape == (48, 3) and len(wraps) > 1


def test_code_fourier_equals_pair_ws_fourier():
    images, table = _ws_tables()
    rng = np.random.default_rng(0)
    n = SITES.shape[0]
    X = rng.standard_normal((64, n, n)) + 1j * rng.standard_normal((64, n, n))
    k = rng.random((5, 3))
    ref = pair_ws_fourier(np.moveaxis(X, 0, -1).reshape((n, n) + KGRID),
                          images, k)
    got = pair_ws_code_fourier(X, table, k)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


def _per_pair_code(images):
    """The rule pair by pair (the former table): every (class, i, j) mask."""
    n_t = int(images["img"].shape[1])
    bits = np.left_shift(np.int64(1), np.arange(n_t, dtype=np.int64))
    masks = np.stack([
        (np.asarray(pair_ws_weights(images["pos"], a, b, v, images["tol"])) > 0)
        .astype(np.int64) @ bits
        for a, b, v in zip(images["a"], images["b"], images["valid"])])
    patterns, code = np.unique(masks, return_inverse=True)
    member = (patterns[:, None] & bits[None, :]) != 0
    return code.reshape(masks.shape), member / member.sum(axis=1, keepdims=True)


def test_code_table_is_the_per_pair_rule():
    """One decision per distinct displacement gives the per-pair table: the
    grid-like orbit (repeated displacements) and random positions."""
    random = np.random.default_rng(7).random((40, 3))
    for pos in (SITES, random):
        images = pair_ws_images(KGRID, A, pos, 3)
        table = pair_ws_code_table(images)
        code, weights = _per_pair_code(images)
        np.testing.assert_array_equal(pair_ws_code(table), code)
        np.testing.assert_array_equal(table["weights"], weights)
    rows, cols = np.array([3, -1, 0]), np.array([-1, 5])
    block = pair_ws_code(table, rows, cols)
    assert block.shape == (code.shape[0], 3, 2)
    assert not block[:, 1].any() and not block[:, :, 0].any()
    np.testing.assert_array_equal(block[:, [0, 2], 1], code[:, [3, 0], 5])


def test_node_pair_table_places_the_tiled_code_on_the_face():
    """Each rank's face tile of the (n_c, rank, rank) code is its block of
    the spin-tiled scalar code, pad rows and columns pattern 0."""
    import types
    if jax.device_count() < 4:
        import pytest
        pytest.skip("needs 4 host devices")
    from jax.sharding import Mesh
    from bandstructure.fh_interp import node_pair_table
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    n, ns, rank = SITES.shape[0], 2, 2 * SITES.shape[0] + 4
    nodes = types.SimpleNamespace(node_frac=SITES, lattice_bohr=A, sys_dim=3,
                                  shape=(rank, ns, n), targets_are_nodes=True)
    table = node_pair_table(nodes, KGRID, mesh)
    scalar = pair_ws_code(pair_ws_code_table(pair_ws_images(KGRID, A, SITES, 3)))
    want = np.zeros((scalar.shape[0], rank, rank), dtype=np.int16)
    want[:, :ns * n, :ns * n] = np.tile(scalar, (1, ns, ns))
    np.testing.assert_array_equal(np.asarray(table["code"]), want)
    assert all(s.data.shape == (scalar.shape[0], rank // 2, rank // 2)
               for s in table["code"].addressable_shards)
    # Back-map targets: one image table over nodes and targets; the target
    # rows (s, t) take the target x node code of that table.
    targets = np.random.default_rng(2).random((6, 3))
    t_carrier = ns * targets.shape[0] + 4
    mapped = types.SimpleNamespace(
        node_frac=SITES, lattice_bohr=A, sys_dim=3, targets_are_nodes=False,
        target_frac=targets, target=np.zeros((1, 1, t_carrier)),
        shape=(rank, ns, targets.shape[0]))
    table = node_pair_table(mapped, KGRID, mesh)
    union = pair_ws_code(pair_ws_code_table(pair_ws_images(
        KGRID, A, np.concatenate([SITES, targets]), 3)))
    want_t = np.zeros((union.shape[0], t_carrier, rank), dtype=np.int16)
    want_t[:, :ns * 6, :ns * n] = np.tile(union[:, n:, :n], (1, ns, ns))
    np.testing.assert_array_equal(np.asarray(table["target_code"]), want_t)
    want[:, :ns * n, :ns * n] = np.tile(union[:, :n, :n], (1, ns, ns))
    np.testing.assert_array_equal(np.asarray(table["code"]), want)


def test_direct_eigh_is_exact_on_the_coarse_grid():
    fH_R, params, E, nb = _kernel()
    _, table = _ws_tables()
    got = _energies(fH_R, _coarse_k(), params, nb,
                    lambda X, k: pair_ws_code_fourier(X, table, k))
    np.testing.assert_allclose(got, E[:, :N_WINDOW], atol=1e-10)


def test_ws_kernel_is_star_covariant_and_the_box_is_not():
    _, table = _ws_tables()
    ws = _star_spread(lambda X, k: pair_ws_code_fourier(X, table, k))
    box = _star_spread(_box)
    assert ws < 1e-10, ws                            # Ry: rounding
    assert box > 1e-4, box                           # the control fails


def _off_grid_error(fourier):
    """max |E - E_exact| (meV) of the window at random off-grid q."""
    fH_R, params, _, nb = _kernel()
    q = np.random.default_rng(13).random((8, 3)) - 0.5
    got = _energies(fH_R, q, params, nb, fourier)
    exact = np.linalg.eigvalsh(_hamiltonian(q))[:, :N_WINDOW]
    return RY_TO_MEV * float(np.abs(got - exact).max())


def test_per_pair_ws_is_closer_off_the_grid_than_its_control_variants():
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


def test_quadrature_metric_and_loewdin_orthonormalize_sampled_values():
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


def test_quadrature_metric_weights_are_one_per_orbit():
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


def test_top_block_matches_full_eigh_off_the_grid():
    """Top block (15 fitted + 32 columns of 48) from the nearest coarse C:
    window eigenpairs equal ``eigh`` to 1e-10 at random off-grid q."""
    fH_R, _, _, nb = _kernel()
    _, table = _ws_tables()
    E, C = np.linalg.eigh(_hamiltonian(_coarse_k()))
    ctilde = np.swapaxes(C[:, :, :nb], 1, 2)         # (nk, nb, rank)
    q = np.random.default_rng(5).random((6, 3)) - 0.5
    K = np.asarray(pair_ws_code_fourier(fH_R, table, q))
    K = 0.5 * (K + np.conj(np.swapaxes(K, -1, -2)))
    x0 = coarse_start_block(ctilde, KGRID, q)
    lam, V, steps, res = top_block_eigh(K, x0, (0, N_WINDOW))
    ref, U = np.linalg.eigh(K)
    lam, V = np.asarray(lam)[:, :N_WINDOW], np.asarray(V)[:, :, :N_WINDOW]
    scale = np.abs(ref[:, :N_WINDOW])
    assert float(res) <= 1e-10 and int(steps) <= 8, (float(res), int(steps))
    assert np.max(np.abs(lam - ref[:, :N_WINDOW]) / scale) < 1e-10
    proj = lambda X: np.einsum('qan,qbn->qab', X, np.conj(X))
    assert np.max(np.abs(proj(V) - proj(U[:, :, :N_WINDOW]))) < 1e-8


def test_top_block_iterates_with_a_block_far_below_the_rank():
    """Three orbits (rank 144) and a block of 16 + 32 columns (16 bands end
    in a 0.009 Ry gap): the solve iterates past its power steps to the full
    eigensolver's window; stopped at the power steps (``max_iter = power``,
    the control) it is not converged, and the stage receipt refuses a
    residual above the bound."""
    sites, labels = _sites(3)
    fH_R, _, _, nb = _kernel(sites, onsite=0.05 * labels, nb=16)
    table = pair_ws_code_table(pair_ws_images(KGRID, A, sites, 3))
    _, C = np.linalg.eigh(_hamiltonian(_coarse_k(), sites, 0.05 * labels))
    ctilde = np.swapaxes(C[:, :, :nb], 1, 2)
    q = np.random.default_rng(29).random((6, 3)) - 0.5
    K = np.asarray(pair_ws_code_fourier(fH_R, table, q))
    K = 0.5 * (K + np.conj(np.swapaxes(K, -1, -2)))
    x0 = coarse_start_block(ctilde, KGRID, q)
    assert x0.shape[2] == nb + TOP_BLOCK_PAD and 3 * x0.shape[2] <= K.shape[1]
    ref = np.linalg.eigvalsh(K)[:, :nb]
    lam, _, steps, res = top_block_eigh(K, x0, (0, nb))
    err = np.max(np.abs(np.asarray(lam)[:, :nb] - ref) / np.abs(ref))
    assert float(res) <= 1e-10 and int(steps) > TOP_BLOCK_POWER, (
        float(res), int(steps))
    assert err < 1e-9, err
    lam_c, _, steps_c, res_c = top_block_eigh(K, x0, (0, nb),
                                              max_iter=TOP_BLOCK_POWER)
    err_c = np.max(np.abs(np.asarray(lam_c)[:, :nb] - ref) / np.abs(ref))
    assert int(steps_c) == TOP_BLOCK_POWER and float(res_c) > 1e-10, (
        int(steps_c), float(res_c))
    assert err_c > 10.0 * err, (err_c, err)
    log_top_block(lambda *a: None, "toy", nb, [steps], [res])
    import pytest
    with pytest.raises(RuntimeError, match="top_block_not_converged"):
        log_top_block(lambda *a: None, "toy", nb, [steps_c],
                      [max(float(res_c), 10.0 * TOP_BLOCK_REFUSE)])


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


def test_back_map_recovers_states_at_targets_through_a_nonuniform_metric():
    """Nodes = two orbits whose values carry different scales and a smearing
    (W non-uniform, G != 1); targets = 16 other points; 16 fitted bands
    (0.08 Ry gap above), 9 returned (0.40 Ry gap).  ``compute_wfns_fi``
    returns the window's states at the targets exactly at coarse k (the
    projector, gauge-free) and off the grid closer to the exact ones than
    with the targets' geometry withheld from the WS images (the control)."""
    import jax.numpy as jnp
    import pytest
    from jax.sharding import Mesh
    from bandstructure.bse_setup import compute_wfns_fi
    from ffi.fft import xla_reference
    from isdf.galerkin import NodalInterp
    if jax.device_count() < 4:
        pytest.skip("needs 4 host devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    sites, labels = _sites(2)
    onsite = 0.05 * labels
    nb, n_win = 16, 9
    targets = np.random.default_rng(23).random((16, 3))
    scale = np.array([1.0, 2.5])
    k = _coarse_k()
    E = np.linalg.eigvalsh(_hamiltonian(k, sites, onsite))[:, :nb]
    phi, at_t = _back_map_states(k, sites, labels, targets, onsite, nb, scale)
    w, gram = _quadrature_metric(phi[:, :, None, :], labels)
    inv_half, eps = _loewdin(gram)
    assert np.ptp(w) > 0.1 * w.max() and eps > 1e-2, (w, eps)
    ctilde = np.einsum('kmn,kmv,v->knv', inv_half, phi, np.sqrt(w))
    q_on = k[[0, 7, 22, 41]]
    q_off = np.random.default_rng(31).random((6, 3)) - 0.5
    q = np.concatenate([q_on, q_off])
    _, exact = _back_map_states(q, sites, labels, targets, onsite, nb, scale)
    proj = lambda X: np.einsum('qnt,qns->qts', X, np.conj(X))
    want = proj(exact[:, :n_win])

    def errors(target_frac):
        interp = NodalInterp(jnp.asarray(phi), sites, A, 3, 1,
                             target=jnp.asarray(at_t), target_frac=target_frac)
        with xla_reference():
            fine = compute_wfns_fi(
                ctilde=jnp.asarray(ctilde), B_at_mu=interp,
                enk_sigma=jnp.asarray(E.T), kgrid_co=KGRID, q_list=q,
                band_window_fi=(0, n_win), mesh_xy=mesh)
        got = proj(np.asarray(fine.psi_rmu_Y)[:, :, 0, :])
        rel = (np.abs(got - want).max(axis=(1, 2))
               / np.abs(want).max(axis=(1, 2)))
        return rel[:len(q_on)].max(), rel[len(q_on):].max()

    on, off = errors(targets)
    assert on < 1e-9, on
    on_c, off_c = errors(np.zeros_like(targets))
    assert on_c < 1e-9, on_c                         # on-grid: geometry-free
    assert off < 0.05 and off_c > 3.0 * off, (off, off_c)


def _back_map_toy():
    """The back-map toy of the test above: two orbits (96 nodes) with a
    smeared, per-orbit-scaled sampling, 16 targets, 16 fitted / 9 returned
    bands, 4 coarse and 6 off-grid q."""
    sites, labels = _sites(2)
    onsite, nb = 0.05 * labels, 16
    targets = np.random.default_rng(23).random((16, 3))
    k = _coarse_k()
    E = np.linalg.eigvalsh(_hamiltonian(k, sites, onsite))[:, :nb]
    phi, at_t = _back_map_states(k, sites, labels, targets, onsite, nb,
                                 np.array([1.0, 2.5]))
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


def test_dropping_zero_weight_nodes_is_exact():
    """A node orbit with w = 0 has exactly zero rows of C = W^{1/2} Phi
    G^{-1/2}; dropping it from the rank axis, with every node kept as a
    back-map target, leaves the energies (<= 1e-10 meV) and the window
    projector at the targets unchanged."""
    import jax.numpy as jnp
    import pytest
    from isdf.galerkin import NodalInterp
    if jax.device_count() < 4:
        pytest.skip("needs 4 host devices")
    sites, labels, _, E, phi, _, q = _back_map_toy()
    w, _ = _quadrature_metric(phi[:, :, None, :], labels)
    w = np.where(labels == 1, 0.0, w)                # orbit 2: zero weight
    inv_half, _ = _loewdin(np.einsum('kmv,v,knv->kmn', np.conj(phi), w, phi))
    ctilde = np.einsum('kmn,kmv,v->knv', inv_half, phi, np.sqrt(w))
    keep = w > 0.0
    assert keep.sum() == 48 and not np.any(ctilde[:, :, ~keep])
    e_all, p_all = _fine(ctilde, NodalInterp(jnp.asarray(phi), sites, A, 3, 1),
                         E, q, 9)
    e_kept, p_kept = _fine(
        ctilde[:, :, keep],
        NodalInterp(jnp.asarray(phi[:, :, keep]), sites[keep], A, 3, 1,
                    target=jnp.asarray(phi), target_frac=sites), E, q, 9)
    assert RY_TO_MEV * np.abs(e_kept - e_all).max() <= 1e-10
    assert np.abs(p_kept - p_all).max() <= 1e-12 * np.abs(p_all).max()


def test_face_route_matches_the_q_batched_route():
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


def test_fit_drops_zero_weight_nodes_and_the_mask_round_trips(tmp_path,
                                                              monkeypatch):
    """``fit_galerkin_basis`` on sampled values Phi = U / sqrt(v) (three
    orbits of 4 nodes, the nodes are the 12 centroids) with orbit 3 at
    w = 0: its 4 nodes leave the rank axis, C is the kept columns of the
    full fit, every centroid stays a back-map target with its pre-drop
    values, and ``galerkin_dft.h5`` returns the kept-node mask, the
    requested nodes and every array unchanged."""
    import types
    import pytest
    from jax.sharding import Mesh
    from common import parallel_transport
    from isdf import galerkin
    if jax.device_count() < 4:
        pytest.skip("needs 4 host devices")
    rng = np.random.default_rng(13)
    nk, nb, grid = 8, 5, (12, 12, 12)
    labels = np.repeat(np.arange(3), 4)
    U = np.linalg.qr(rng.standard_normal((nk, labels.size, nb))
                     + 1j * rng.standard_normal((nk, labels.size, nb)))[0]
    phi = np.moveaxis(U, -1, 1) / np.sqrt(np.array([0.5, 2.0, 1.3])[labels])
    flat = rng.permutation(np.prod(grid))[:labels.size]
    idx = np.stack(np.unravel_index(flat, grid), axis=1)
    metric = galerkin._quadrature_metric

    def zero_orbit_3(values, orbit, **kw):
        w = np.where(orbit == 2, 0.0, metric(values, orbit, **kw)[0])
        return w, np.einsum('kmsv,v,knsv->kmn', np.conj(values), w, values)

    monkeypatch.setattr(galerkin, "_quadrature_metric", zero_orbit_3)
    monkeypatch.setattr(galerkin, "_orbit_labels", lambda *a: (labels, 4))
    monkeypatch.setattr(galerkin, "_sample", lambda *a: phi[:, :, None, :])
    monkeypatch.setattr(parallel_transport, "wfn_fingerprint", lambda w: "toy")
    wfn = types.SimpleNamespace(avec=A / 10.0, alat=10.0)
    meta = types.SimpleNamespace(nspinor=1, nk_tot=nk, fft_grid=grid,
                                 kgrid=(2, 2, 2), sys_dim=3, mu_basis=None)
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    logs = []
    fit = galerkin.fit_galerkin_basis(wfn, None, meta, idx, mesh, (0, nb),
                                      log_fn=logs.append)
    keep = labels < 2
    assert fit.node_indices == galerkin._index_rows(idx[keep])
    assert fit.requested_nodes == galerkin._index_rows(idx)
    assert fit.rank_physical == 8 and "4 zero-weight node(s) dropped of 12" \
        in logs[0]
    w, gram = zero_orbit_3(phi[:, :, None, :], labels)
    ctilde = np.einsum('kmn,kmsv,v->knsv', _loewdin(gram)[0],
                       phi[:, :, None, :], np.sqrt(w), optimize=True)
    np.testing.assert_allclose(np.asarray(fit.ctilde)[:, :, :8],
                               ctilde[:, :, 0, keep], rtol=0, atol=1e-13)
    np.testing.assert_array_equal(
        np.asarray(fit.basis_at_nodes.target)[:, :, :12], phi)
    path = tmp_path / "galerkin_dft.h5"
    galerkin.write_galerkin_basis(path, fit, wfn=wfn, meta=meta,
                                  centroid_indices=idx, bispinor=False,
                                  refit_rcond=1e-3, mesh_xy=mesh)
    back = galerkin.read_galerkin_basis(
        path, wfn=wfn, meta=meta, centroid_indices=idx, node_indices=idx,
        band_range=(0, nb), bispinor=False, refit_rcond=1e-3, mesh_xy=mesh)
    assert back.node_indices == fit.node_indices
    assert back.requested_nodes == fit.requested_nodes
    assert back.node_weights == fit.node_weights
    for name in ("ctilde", "loewdin"):
        np.testing.assert_array_equal(np.asarray(getattr(back, name)),
                                      np.asarray(getattr(fit, name)))
    for name in ("nodal", "target", "node_frac", "target_frac"):
        np.testing.assert_array_equal(
            np.asarray(getattr(back.basis_at_nodes, name)),
            np.asarray(getattr(fit.basis_at_nodes, name)))


def test_node_cache_is_published_once_and_keyed_on_the_kmeans_version(
        tmp_path, monkeypatch):
    """``htransform_nodes`` above ``KMEANS_ABOVE x want``: a miss runs the
    k-means once and publishes ``galerkin_nodes_*.txt``; the next call
    reuses it (rank 0's read) without the k-means; a new ``KMEANS_VERSION``
    misses and publishes a second cache."""
    import sys
    import types
    from bandstructure import fh_interp
    from centroid import kmeans_isdf
    from common import parallel_transport
    calls = []

    def kmeans(wfn_path, n_points, *, fit_window, seed=0):
        calls.append(n_points)
        return np.arange(3 * n_points).reshape(-1, 3) % 7

    fake = types.ModuleType("centroid.kmeans_cli")
    fake.orbit_kmeans_points = kmeans
    monkeypatch.setitem(sys.modules, "centroid.kmeans_cli", fake)
    monkeypatch.setattr(parallel_transport, "wfn_fingerprint",
                        lambda w: "f" * 64)
    run = lambda: fh_interp.htransform_nodes(
        None, np.zeros((200, 3), dtype=np.int64), (0, 2), wfn_path="WFN.h5",
        basis_path=str(tmp_path / "galerkin_dft.h5"))
    first, again = run(), run()
    assert calls == [44] and first.shape == (44, 3)
    np.testing.assert_array_equal(again, first)
    assert len(list(tmp_path.glob("galerkin_nodes_*.txt"))) == 1
    monkeypatch.setattr(kmeans_isdf, "KMEANS_VERSION", "kmeans-isdf-bumped")
    run()
    assert calls == [44, 44]
    assert len(list(tmp_path.glob("galerkin_nodes_*.txt"))) == 2
