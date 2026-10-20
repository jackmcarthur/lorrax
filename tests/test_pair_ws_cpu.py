"""Per-node-pair Wigner–Seitz images of the htransform off-grid sum (CPU).

``fh_interp.pair_ws_lattice_images`` maps an α-space lattice image X(R) to
images X(R, T) at R + L T chosen per centroid pair by
|R + L T + r_mu - r_nu| (Wannier90 ``use_ws_distance``), through the α basis
at the nodes.  Checked on a random basis and random node positions:

* the image sum reproduces X(k) = sum_R e^{-2 pi i k.R} X(R) at every coarse k
  (on-grid results unchanged);
* with every node at one point, the pair images reduce to one shared
  Wigner–Seitz set with 1/N_deg weights (the shared set is the d = 0 case);
* an X(k) Hermitian on the coarse grid stays Hermitian off the grid;
* the candidate filter never drops an image that is minimal for some
  displacement inside the node-displacement box.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from bandstructure.fh_interp import (  # noqa: E402
    build_R_grid_np, lattice_fourier_phase, pair_ws_lattice_images,
    ws_candidate_images)

FCC = 10.26 * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
HEX = 5.97 * np.array([[1, 0, 0], [-.5, np.sqrt(3) / 2, 0], [0, 0, 6.0]])
MESH = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


def _case(kgrid, lat, n_mu=10, ns=2, rank=12, same_point=False, seed=0):
    rng = np.random.default_rng(seed)
    nk = int(np.prod(kgrid))
    B = rng.standard_normal((rank, ns, n_mu)) + 1j * rng.standard_normal((rank, ns, n_mu))
    r = np.zeros((n_mu, 3)) if same_point else rng.random((n_mu, 3))
    if kgrid[2] == 1:
        r[:, 2] = 0.3 if same_point else rng.random(n_mu) * 0.2
    # X(k) Hermitian on the coarse grid -> X(R) via the make_flat_k_ifftn sign
    A = rng.standard_normal((nk, rank, rank)) + 1j * rng.standard_normal((nk, rank, rank))
    Xk = A + np.conj(np.swapaxes(A, 1, 2))
    box = build_R_grid_np(kgrid)
    kc = box / np.asarray(kgrid)
    XR = np.einsum('kr,kab->rab', np.exp(2j * np.pi * kc @ box.T), Xk) / nk
    R_img, (Xi,) = pair_ws_lattice_images(
        [jnp.asarray(XR)], kgrid, lat, r, jnp.asarray(B), MESH)
    return kc, Xk, XR, R_img, np.asarray(Xi), box


@pytest.mark.parametrize("kgrid, lat", [((3, 3, 3), FCC), ((4, 4, 1), HEX)])
def test_on_grid_unchanged_and_offgrid_hermitian(kgrid, lat):
    kc, Xk, XR, R_img, Xi, _ = _case(kgrid, lat)
    on = np.einsum('kj,jab->kab', np.asarray(lattice_fourier_phase(jnp.asarray(kc), R_img)), Xi)
    np.testing.assert_allclose(on, Xk, atol=1e-9 * np.abs(Xk).max())
    k = np.random.default_rng(3).random((6, 3))
    if kgrid[2] == 1:
        k[:, 2] = 0
    off = np.einsum('kj,jab->kab', np.asarray(lattice_fourier_phase(jnp.asarray(k), R_img)), Xi)
    np.testing.assert_allclose(off, np.conj(np.swapaxes(off, 1, 2)),
                               atol=1e-9 * np.abs(off).max())
    assert R_img.shape[0] > XR.shape[0]          # some classes split across pairs


@pytest.mark.parametrize("kgrid, lat", [((4, 4, 4), FCC), ((6, 6, 1), HEX)])
def test_coincident_nodes_reduce_to_one_shared_ws_set(kgrid, lat):
    kc, Xk, XR, R_img, Xi, box = _case(kgrid, lat, same_point=True)
    img, valid = ws_candidate_images(kgrid, lat, np.zeros(3), np.zeros(3))
    G = lat @ lat.T
    k = np.random.default_rng(5).random((5, 3))
    if kgrid[2] == 1:
        k[:, 2] = 0
    phase = np.zeros((len(k), len(box)), complex)
    for c in range(len(box)):
        v = img[c, valid[c]]
        d2 = np.einsum('ti,ij,tj->t', v, G, v)
        sel = v[d2 <= d2.min() * (1 + 1e-6) + 1e-9]
        phase[:, c] = np.exp(-2j * np.pi * k @ sel.T).mean(axis=1)
    ref = np.einsum('kr,rab->kab', phase, XR)
    got = np.einsum('kj,jab->kab', np.asarray(lattice_fourier_phase(jnp.asarray(k), R_img)), Xi)
    np.testing.assert_allclose(got, ref, atol=1e-9 * np.abs(ref).max())


@pytest.mark.parametrize("kgrid, lat", [((4, 4, 4), FCC), ((9, 9, 1), HEX), ((3, 3, 3), FCC)])
def test_candidate_filter_keeps_every_minimal_image(kgrid, lat):
    """Brute force over T in [-4, 4]: the minimal image of R + L T + d, for
    random d in the node-displacement box, is always among the candidates."""
    lo, hi = -np.ones(3), np.ones(3)
    if kgrid[2] == 1:
        lo[2], hi[2] = -0.2, 0.2
    img, valid = ws_candidate_images(kgrid, lat, lo, hi)
    G = lat @ lat.T
    rng = np.random.default_rng(11)
    T = np.stack(np.meshgrid(*[np.arange(-4, 5) if n > 1 else np.zeros(1) for n in kgrid],
                             indexing="ij"), -1).reshape(-1, 3)
    box = build_R_grid_np(kgrid)
    for c in range(len(box)):
        cand = {tuple(np.round(v).astype(int)) for v in img[c, valid[c]]}
        allv = box[c] + T * np.asarray(kgrid)
        for d in lo + (hi - lo) * rng.random((40, 3)):
            u = allv + d
            best = allv[np.argmin(np.einsum("ti,ij,tj->t", u, G, u))]
            assert tuple(np.round(best).astype(int)) in cand
