"""P=4 gate for the μ-batch ζ-fit kernels (docs/architecture/zeta_fit_mubatch.md).

Four processes, one GPU each, 2x2 mesh.  A synthetic two-component Bloch
system on a box whose three extents differ (6, 5, 8), per-k ψ spheres, a
3x2x1 k grid (a nontrivial -q map), asymmetric L/R band windows (so the LR+RL completion runs),
three of four q rows selected, a μ carrier that the batch does not divide,
and band chunks with pad slots.  Every ψ route (flat-block cache; planes with
ψ(G) resident or read from host) times both row owners (q-owned chunks for
the R4 tier, μ-owned rows otherwise) times every Z-store placement (device,
host, slab_io disk) times each read layout must reproduce an
independent NumPy evaluation of

    Z_q(μ, G) = FFT_r[e^{-iq·r} Σ_k Σ_ab D^L_{k,ab}(μ,r) conj(D^R_{k+q,ab}(μ,r))]

at 1e-12 relative.  Red twin (TASTE 21): the centroid samples of ONE k row
with their two spinor components swapped must miss by far more than the
tolerance.  (Swapping them on every k is an exact invariance of the charge
Z, which traces the spin pair: that twin cannot fire and is not a check.)
Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/zeta_mubatch_p4.py``.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

TAG = "[zeta-mubatch-p4]"
FFT = (6, 5, 8)
KGRID = (3, 2, 1)
NS, NB, MU, TOL = 2, 8, 10, 1.0e-12


def _fail(msg):
    print(f"{TAG} FAIL: {msg}", flush=True)
    raise SystemExit(1)


def _kfrac():
    ks = np.array([(i, j, l) for i in range(KGRID[0]) for j in range(KGRID[1])
                   for l in range(KGRID[2])], float) / np.asarray(KGRID, float)
    return np.where(ks > 0.5, ks - 1.0, ks)


def _rfrac():
    nx, ny, nz = FFT
    i = np.arange(nx * ny * nz)
    return np.stack([i // (ny * nz) / nx, (i // nz) % ny / ny, i % nz / nz], 1)


def _system(rng):
    nk = int(np.prod(KGRID))
    n_r = int(np.prod(FFT))
    g = np.stack(np.meshgrid(*[np.fft.fftfreq(n) * n for n in FFT],
                             indexing="ij"), -1).reshape(-1, 3)
    kf = _kfrac()
    spheres = [np.flatnonzero(np.sum((g + kf[k]) ** 2, 1) < 4.2) for k in range(nk)]
    ngk = max(s.size for s in spheres)
    g_index = np.full((nk, n_r), ngk, np.int32)
    psi_G = np.zeros((nk, NB, NS, ngk), complex)
    for k, s in enumerate(spheres):
        g_index[k, s] = np.arange(s.size)
        psi_G[k, :, :, :s.size] = (rng.standard_normal((NB, NS, s.size))
                                   + 1j * rng.standard_normal((NB, NS, s.size)))
    box = np.zeros((nk, NB, NS, n_r), complex)
    for k in range(nk):
        ok = g_index[k] < ngk
        box[k][:, :, ok] = psi_G[k][:, :, g_index[k][ok]]
    psi_r = np.fft.ifftn(box.reshape(nk, NB, NS, *FFT), axes=(-3, -2, -1),
                         norm="ortho").reshape(nk, NB, NS, n_r)
    psi_r *= np.exp(2j * np.pi * kf @ _rfrac().T)[:, None, None, :]
    return psi_G, g_index.reshape(nk, *FFT), psi_r


def _reference(psi_r, cen, w_l, w_r, q_neg, q_sel, sphere):
    nk = psi_r.shape[0]
    x = psi_r[:, :, :, cen]                              # (k, n, a, mu)
    y = np.conj(psi_r)                                   # (k, n, b, r)
    D_l = np.einsum('n,knam,knbr->abkmr', w_l, x, y)
    D_r = np.einsum('n,knam,knbr->abkmr', w_r, x, y)
    sh = KGRID + (len(cen), psi_r.shape[-1])
    acc = np.zeros(sh, complex)
    for a in range(NS):
        for b in range(NS):
            Pl = np.fft.ifftn(np.conj(D_l[a, b]).reshape(sh), axes=(0, 1, 2),
                              norm="forward")
            Pr = np.fft.ifftn(np.conj(D_r[a, b]).reshape(sh), axes=(0, 1, 2),
                              norm="forward")
            acc += np.conj(Pl) * Pr
    Z = np.fft.fftn(acc, axes=(0, 1, 2), norm="forward").reshape(nk, *sh[3:])
    Z = (Z + np.conj(Z[q_neg]))[q_sel]
    qf = _kfrac()[q_sel]
    Z = Z * np.exp(-2j * np.pi * qf @ _rfrac().T)[:, None, :]
    G = np.fft.fftn(Z.reshape(Z.shape[:2] + FFT), axes=(-3, -2, -1)).reshape(
        Z.shape[:2] + (-1,))
    return np.take_along_axis(G, sphere[:, None, :], axis=2)


class _Store:
    """The PsiGStore surface the kernels read, over an in-memory ψ(G)."""

    def __init__(self, psi_G, g_index, kvecs, bcr, P_, mesh):
        self.psi_G = psi_G
        self.band_chunk_ranges = tuple(bcr)
        self._bpd_per_bc = tuple((hi - lo) // P_ for lo, hi in bcr)
        self.b_p = max(self._bpd_per_bc)
        self.P = P_
        self.py = int(mesh.shape['y'])
        nk, _, ns, ngk = psi_G.shape
        self.local_band_chunk_shape = (nk, self.b_p, ns, ngk)
        self.band_chunk_carrier = self.b_p * P_
        rep = NamedSharding(mesh, P())
        from common.collectives import device_put_process_local
        self.g_index = device_put_process_local(g_index, rep)
        self.kvecs_frac = device_put_process_local(kvecs, rep)

    def read_local_band_chunk(self, x, y, bc):
        p = int(x) * self.py + int(y)
        lo, _ = self.band_chunk_ranges[int(bc)]
        bpd = self._bpd_per_bc[int(bc)]
        out = np.zeros(self.local_band_chunk_shape, complex)
        out[:, :bpd] = self.psi_G[:, lo + p * bpd: lo + (p + 1) * bpd]
        return out

    def release_host_tiles(self):
        pass


def _gather(x):
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def main():
    from isdf import zeta_mubatch as zmb
    from isdf.core import build_psi_G_resident_sm
    from common.collectives import device_put_process_local

    if jax.process_count() != 4 or jax.device_count() != 4:
        _fail(f"needs 4 processes x 1 GPU; got {jax.process_count()} x "
              f"{jax.local_device_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(20260923)
    psi_G, g_index, psi_r = _system(rng)
    nk = psi_G.shape[0]
    n_r = int(np.prod(FFT))
    mu_pad = 12                                          # carrier; 10 real
    cen = rng.choice(n_r, MU, replace=False)
    face = np.zeros((nk, NS, mu_pad, NB), complex)
    face[:, :, :MU, :] = np.transpose(psi_r[:, :, :, cen], (0, 2, 3, 1))
    w_l = np.r_[np.ones(6), np.zeros(2)]
    w_r = np.r_[np.zeros(2), np.ones(6)]
    ks = np.array([(i, j, l) for i in range(KGRID[0]) for j in range(KGRID[1])
                   for l in range(KGRID[2])])
    kn = (-ks) % np.asarray(KGRID)
    q_neg = (kn[:, 0] * KGRID[1] * KGRID[2] + kn[:, 1] * KGRID[2]
             + kn[:, 2]).astype(np.int32)
    q_sel = np.array([0, 1, 3], np.int32)
    qf = _kfrac()[q_sel]
    g = np.stack(np.meshgrid(*[np.fft.fftfreq(n) * n for n in FFT],
                             indexing="ij"), -1).reshape(-1, 3)
    sph = [np.flatnonzero(np.sum((g + q) ** 2, 1) < 3.1) for q in qf]
    ngk_z = max(s.size for s in sph)
    sphere = np.stack([np.r_[s, np.repeat(s[:1], ngk_z - s.size)] for s in sph])
    cen_all = np.r_[cen, np.zeros(mu_pad - MU, int)]
    ref = _reference(psi_r, cen_all, w_l, w_r, q_neg, q_sel, sphere)
    ref[:, MU:, :] = 0.0

    bcr = ((0, 4), (4, 8))
    store = _Store(psi_G, g_index, _kfrac(), bcr, 4, mesh)
    band_rel, bl, br = zmb.band_slot_tables(store, band_start=0, nb_face=NB,
                                            weight_l=w_l, weight_r=w_r)
    rep = NamedSharding(mesh, P())
    tables = tuple(device_put_process_local(np.asarray(a), rep)
                   for a in (band_rel, bl, br))
    face_dev = jax.make_array_from_callback(
        face.shape, NamedSharding(mesh, P(None, None, 'x', 'y')),
        lambda idx: face[idx])
    b, g_tile = 8, 8
    n_Gt = -(-ngk_z // g_tile)
    sph_pad = np.concatenate(
        [sphere, np.repeat(sphere[:, -1:], n_Gt * g_tile - ngk_z, axis=1)], 1)

    def run(route, source, placement, *, rows='mu', swap_spin=False, tmpdir=None):
        rb = zmb.make_r_blocks(FFT, 4, route=route,
                               r_sub=(16 if route == 'cache' else 1))
        cyl = None
        if source == 'cache':
            src = zmb.build_psi_block_cache(store, mesh=mesh, rb=rb)
        else:
            cyl = zmb.plane_cylinder(store, rb)
            src = (build_psi_G_resident_sm(store, mesh_xy=mesh)
                   if source == 'resident' else jnp.zeros((1,), jnp.complex128))
        kern = zmb.make_batch_kernel(
            mesh=mesh, rb=rb, kgrid=KGRID, fft_grid=FFT, nk=nk, ns=NS, b=b,
            n_bc=len(bcr), bc_w=int(band_rel.shape[1]), nb_face=NB,
            q_sel=q_sel, q_neg=q_neg, sphere_idx=sph_pad, qvec_frac=qf,
            row_chunk=3, source=source, rows=rows, psi_G_store=store,
            cylinder=cyl)
        zs = zmb.ZStore(mesh=mesh, Q=len(q_sel), mu_pad=mu_pad,
                        n_G=n_Gt * g_tile, b=b, g_tile=g_tile,
                        placement=placement, rows=rows,
                        scratch_path=os.path.join(tmpdir or ".", "zstore.h5"))
        pts = device_put_process_local(zmb.block_points(rb), rep)
        cyl_ops = cyl if cyl is not None else tuple(
            jnp.zeros((1,), jnp.int32) for _ in range(3))
        for beta in range(zs.n_batch):
            X = zmb.gather_batch_centroids(
                face_dev, zmb.batch_slots(mu_pad, b, beta), mesh=mesh)
            if swap_spin:
                X = X.at[1].set(X[1, ::-1])
            zs.write_batch(beta, kern(src, X, *tables, pts, store.kvecs_frac,
                                      cyl_ops, zmb.dummy_extra(mesh)))
        out = {}
        for layout in (('q',) if rows == 'q' else ('q', 'g')):
            tiles = [_gather(zs.read_tile(t, layout=layout)) for t in range(zs.n_Gt)]
            full = np.concatenate(tiles, axis=2)[:len(q_sel), :, :ngk_z]
            out[layout] = full
        zs.close()
        return out

    rel = lambda a: float(np.linalg.norm(a - ref) / np.linalg.norm(ref))
    worst = 0.0
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCRATCH", ".")) as td:
        shared = multihost_utils.broadcast_one_to_all(
            np.frombuffer(td.encode().ljust(512, b"\0"), np.uint8))
        td0 = bytes(np.asarray(shared)).rstrip(b"\0").decode()
        for route, source in (('cache', 'cache'), ('planes', 'resident'),
                              ('planes', 'host')):
            for rows in ('q', 'mu'):
                for placement in ('device', 'host', 'disk'):
                    if placement == 'disk' and source != 'cache':
                        continue
                    res = run(route, source, placement, rows=rows, tmpdir=td0)
                    for layout, Z in res.items():
                        e = rel(Z)
                        worst = max(worst, e)
                        if jax.process_index() == 0:
                            print(f"{TAG} route={route:<6s} source={source:<8s} "
                                  f"rows={rows:<2s} store={placement:<6s} "
                                  f"read={layout}  rel={e:.2e}", flush=True)
                        if not e <= TOL:
                            _fail(f"{route}/{source}/{rows}/{placement}/{layout}: "
                                  f"{e:.3e}")
        red = rel(run('cache', 'cache', 'device', swap_spin=True)['q'])
    if jax.process_index() == 0:
        print(f"{TAG} red twin (spinor components swapped in X_B at k=1): rel={red:.2e}",
              flush=True)
    if not red > 1e-3:
        _fail(f"red twin did not fire: {red:.3e}")
    if jax.process_index() == 0:
        print(f"{TAG} PASS worst={worst:.2e} (tol {TOL:.0e})", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
