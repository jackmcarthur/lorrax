"""The c1 head channel carries the full-Bloch gauge phase at q != 0 (``gw.head_densify.attach_head_channel``), CPU.

The W tiles are in the full-Bloch gauge, ``W_μν(q) = Σ_R e^{−2πi q·R} W(r_μ + R, r_ν)`` with ``r_μ = idx/N ∈ [0,1)``,
so their G = 0 channel carries ``e^{2πi q·(r_μ − r_ν)}`` and the head projector at a fine q is
``g₀(q)_μ = g₀_μ e^{−2πi q·r_μ}`` (q the shortest representative).  Toy: simple cubic (a = 1 bohr), Oh about a centre
``c`` off the lattice points (``g r = S r + c − S c``, 48 operations), two centroid orbits about ``c`` folded into the home
cell, so an operation's lattice offsets ``g r_μ = r_{Pμ} + L_μ`` differ between centroids (as in diamond Si); Gaussian interpolation vectors ``ζ̃_{q,μ}(G) = e^{−σ²|K|²/2} e^{−iK·r_μ}``, ``K = 2π(q + G)``, the convention of
``tests/test_srlr_densify_cpu.py``; coarse 2³ → fine 4³, the head channel on the 27 fine q of the coarse Γ cell.

1. star covariance of the c1 tile (the head-less toy body plus the attached head) at every head q and operation,
   ``W_{Pμ,Pν}(Sq) = e^{−2πi Sq·(L_μ − L_ν)} W_μν(q)``, to 1e-12; the frozen projector (``g₀(q) = g₀``) breaks it;
2. the attached head against the toy's exact G = 0 term: the error is the frozen periodic factor, ``e^{σ²|K|²} − 1``;
3. a grid nonzero only at Γ reproduces the frozen-projector update bit for bit (srlr's head, the fine == coarse identity).
"""
import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from gw.head_densify import attach_head_channel, fine_q_cart  # noqa: E402

FINE, COARSE = (4, 4, 4), (2, 2, 2)
BVEC = 2 * np.pi * np.eye(3)                                   # rows; a = 1 bohr
SIGMA, KAPPA = 0.15, 0.5
G_SET = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=np.float64)   # closed under Oh


def _ops():
    """The 48 signed permutation matrices (Oh of the simple-cubic lattice; τ = 0)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (j, sg) in enumerate(zip(perm, signs)):
                M[i, j] = sg
            out.append(M)
    return out


OPS = _ops()
CENTRE = np.array([0.08, 0.05, 0.11])
TAUS = [CENTRE - S @ CENTRE for S in OPS]


def _orbit(d):
    pts = []
    for S in OPS:
        x = np.mod(CENTRE + S @ d, 1.0)
        if not any(np.allclose(x, y) for y in pts):
            pts.append(x)
    return pts


R_MU = np.array(_orbit(np.array([0.13, 0.04, 0.07])) + _orbit(np.array([0.10, 0.10, 0.02])))   # 48 + 24 rows
N_MU = len(R_MU)
G0 = np.r_[np.full(48, 0.7 + 0.2j), np.full(N_MU - 48, -0.4 + 0.5j)]                          # invariant per orbit
Q_FRAC = fine_q_cart(BVEC, FINE) @ np.linalg.inv(BVEC)          # (4,4,4,3) shortest representatives
HEAD_Q = np.argwhere(np.all(np.abs(Q_FRAC) <= 0.25 + 1e-12, axis=-1))   # the coarse Γ cell: 27 fine q


def _perm_and_offsets(S, tau):
    """``S r_μ + τ = r_{P(μ)} + L_μ`` with ``r ∈ [0,1)`` and integer ``L``."""
    img = R_MU @ S.T + tau
    P = np.array([int(np.argmin(np.abs(np.mod(x - R_MU + 0.5, 1.0) - 0.5).sum(1))) for x in img])
    L = img - R_MU[P]
    assert np.allclose(L, np.rint(L)) and len(set(P)) == N_MU
    return P, np.rint(L)


def _toy_tiles():
    """Exact toy W(q) at every fine q and its G = 0 term (Ω = 1, bare kernel 4π/(K² + κ))."""
    W = np.zeros((N_MU, N_MU) + FINE, complex)
    H = np.zeros_like(W)
    for idx in itertools.product(*[range(n) for n in FINE]):
        q = Q_FRAC[idx]
        K = 2 * np.pi * (q + G_SET)
        Z = np.exp(-0.5 * SIGMA ** 2 * np.sum(K * K, 1))[:, None] * np.exp(-1j * K @ R_MU.T)
        v = 4 * np.pi / (np.sum(K * K, 1) + KAPPA)
        W[(slice(None), slice(None)) + idx] = (Z.conj().T * v) @ Z
        g = np.flatnonzero(np.all(G_SET == 0, axis=1))[0]
        H[(slice(None), slice(None)) + idx] = v[g] * np.outer(Z[g].conj(), Z[g])
    return W, H


def _head_scalars(gamma_only=False):
    S = np.zeros(FINE)
    for idx in HEAD_Q:
        q = Q_FRAC[tuple(idx)] @ BVEC
        if gamma_only and np.any(q):
            continue
        S[tuple(idx)] = 4 * np.pi / (q @ q + KAPPA)
    return S


def _attach(W, g0, S, **gauge):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    put = lambda a, spec: jax.device_put(a, NamedSharding(mesh, spec))   # noqa: E731
    out = attach_head_channel(put(W.copy(), P("x", "y", None, None, None)), put(g0, P("x")), put(g0, P("y")),
                              S, 1.0, **gauge)
    return np.asarray(out)


def _frozen(W, g0, S):
    """The pre-2026-10-20 update: ONE g₀ broadcast over every fine q."""
    return W + (np.conj(g0)[:, None] * g0[None, :])[:, :, None, None, None] * S[None, None]


def covariance_residual(Wt):
    """max over operations and head q of ``|W_{Pμ,Pν}(Sq) − e^{−2πi Sq·(L_μ−L_ν)} W_μν(q)| / max|W(q)|``."""
    worst = 0.0
    for S, tau in zip(OPS, TAUS):
        P, L = _perm_and_offsets(S, tau)
        for idx in HEAD_Q:
            q = Q_FRAC[tuple(idx)]
            sq = S @ q
            j = tuple(np.rint(np.mod(sq, 1.0) * np.array(FINE)).astype(int) % np.array(FINE))
            ph = np.exp(-2j * np.pi * (L @ sq))
            A = Wt[(slice(None), slice(None)) + tuple(idx)]
            B = Wt[(slice(None), slice(None)) + j][np.ix_(P, P)]
            worst = max(worst, float(np.abs(B - ph[:, None] * np.conj(ph)[None, :] * A).max() / np.abs(A).max()))
    return worst


def test_toy_body_and_offsets_are_covariant():
    """The relation itself, on the exact toy W (head included): it must hold before anything is attached."""
    W, _ = _toy_tiles()
    assert sum(len(set(map(tuple, _perm_and_offsets(S, t)[1]))) > 1 for S, t in zip(OPS, TAUS)) > 40   # non-uniform folding
    assert covariance_residual(W) <= 1e-12


def test_c1_tile_is_star_covariant_at_the_first_shell():
    W, H = _toy_tiles()
    S = _head_scalars()
    c1 = _attach(W - H, G0, S, centroid_frac=R_MU, bvec=BVEC)
    frozen = _frozen(W - H, G0, S)
    assert covariance_residual(c1) <= 1e-12
    assert covariance_residual(frozen) > 1e-2          # the defect this test guards


def test_head_matches_the_exact_g0_term_up_to_the_periodic_factor():
    W, H = _toy_tiles()
    S = np.zeros(FINE)
    for idx in HEAD_Q:                                  # S(q) = v(q): the toy's own G = 0 kernel
        q = Q_FRAC[tuple(idx)] @ BVEC
        S[tuple(idx)] = 4 * np.pi / (q @ q + KAPPA)
    zero = np.zeros_like(W)
    new = _attach(zero, np.ones(N_MU, complex), S, centroid_frac=R_MU, bvec=BVEC)
    old = _frozen(zero, np.ones(N_MU, complex), S)
    for idx in HEAD_Q:
        sl = (slice(None), slice(None)) + tuple(idx)
        K = 2 * np.pi * Q_FRAC[tuple(idx)]
        bound = np.exp(SIGMA ** 2 * (K @ K)) - 1.0      # ζ̄̃_q(0) / ζ̄̃_0(0) = e^{−σ²K²/2}, frozen at 1
        ref = np.abs(H[sl]).max()
        assert np.abs(new[sl] - H[sl]).max() / ref <= bound * (1 + 1e-9) + 1e-13
        if np.any(K):
            assert np.abs(old[sl] - H[sl]).max() / ref > 0.2


def test_gamma_only_grid_is_the_frozen_update_bit_for_bit():
    """Against the previous jitted program itself (``W + (conj(g0_X) ⊗ g0_Y) ⊗ s``), not a NumPy spelling of it."""
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    rng = np.random.default_rng(3)
    W = rng.standard_normal((N_MU, N_MU) + FINE) + 1j * rng.standard_normal((N_MU, N_MU) + FINE)
    S = _head_scalars(gamma_only=True)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    put = lambda a, spec: jax.device_put(a, NamedSharding(mesh, spec))   # noqa: E731
    previous = jax.jit(lambda W_q, gx, gy, s: W_q + (jnp.conj(gx)[:, None] * gy[None, :])[:, :, None, None, None]
                       * s[None, None, :, :, :])
    ref = np.asarray(previous(put(W, P("x", "y", None, None, None)), put(G0, P("x")), put(G0, P("y")),
                              jnp.asarray(S, dtype=W.dtype)))
    assert np.array_equal(_attach(W, G0, S, centroid_frac=R_MU, bvec=BVEC), ref)
