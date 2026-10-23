"""The distributed ladder rung against a literal two-particle ISDF matrix.

CPU reference FFT/BLAS routes avoid native FFI. Use four virtual CPU devices
to also exercise the 2 × 2 mesh. The hybrid density-response lower row is
checked independently; it is not the optical non-TDA row.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from bse.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings
from ffi.fft import xla_reference

jax.config.update('jax_enable_x64', True)


def _fixture(mesh, ns, batch, *, mu=8, nk=3):
    rng = np.random.default_rng(241 + ns)
    complex_array = lambda shape: rng.normal(size=shape) + 1j*rng.normal(size=shape)
    nc, nv = 4, 2
    c, v = complex_array((nk, nc, ns, mu)), complex_array((nk, nv, ns, mu))
    w = complex_array((nk, mu, mu)) / (mu*ns)
    wr = np.fft.ifft(w, axis=0, norm='ortho').transpose(1, 2, 0)[..., None, None]
    x = complex_array((2, batch, nc, nv, nk))
    sh = make_bse_shardings(mesh)
    put = lambda a, s: jax.device_put(jnp.asarray(a), s)
    # Different density and physical-rung operands detect using the wrong slots.
    args = (put(x, sh.X_full), put(c.conj(), sh.psi_x), put(c.conj(), sh.psi_y),
            put(v.conj(), sh.psi_x), put(v.conj(), sh.psi_y),
            put(np.zeros((nk, nc)), sh.eps), put(np.zeros((nk, nv)), sh.eps),
            put(wr, sh.W), put(np.zeros((mu, mu), complex), sh.V),
            put(np.zeros((nk, nc, nv, mu), complex), sh.M),
            put(c, sh.psi_x), put(c, sh.psi_y), put(v, sh.psi_x), put(v, sh.psi_y))
    return args, (x, c, v, w)


def _dense_rung(x, c, v, w):
    nk, nc, ns, mu = c.shape
    nv = v.shape[1]
    a = np.zeros((nc, nv, nk, nc, nv, nk), complex)
    b = np.zeros_like(a)
    for k in range(nk):
        for j in range(nk):
            a[:, :, k, :, :, j] = np.einsum('ctm,dtm,mn,vsn,wsn->cvdw',
                c[k].conj(), c[j], w[(k-j) % nk], v[k], v[j].conj())/nk
            b[:, :, k, :, :, j] = np.einsum('ctm,wtm,mn,vsn,dsn->cvdw',
                c[k].conj(), v[j], w[(k-j) % nk], v[k], c[j].conj())/nk
    a, b = a.reshape(nc*nv*nk, -1), b.reshape(nc*nv*nk, -1)
    xx, yy = x[0].reshape(x.shape[1], -1), x[1].reshape(x.shape[1], -1)
    # V=D=0: upper row=-WA X-WB Y; lower row=conj(WB conjX+WA conjY).
    return np.stack([-xx @ a.T - yy @ b.T,
                     xx @ b.conj().T + yy @ a.conj().T]).reshape(x.shape)


def test_distributed_ladder_trials_match_literal_dense_matrix(monkeypatch):
    import common.contract_bands as cb
    monkeypatch.setattr(cb, 'bands_gemm_ffi_enabled', lambda: False)
    devices = jax.devices('cpu')
    meshes = [Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))]
    if len(devices) >= 4:
        meshes.append(Mesh(np.array(devices[:4]).reshape(2, 2), ('x', 'y')))
    for mesh in meshes:
        for ns, batch in ((1, 1), (2, 3)):
            args, source = _fixture(mesh, ns, batch)
            expected = _dense_rung(*source)
            with xla_reference():
                for fuse in (False, True):
                    apply = build_bse_ring_matvec_full(mesh, 3, 1, 1,
                        ladder_rung_slots=True, fuse_ladder_rung=fuse)
                    actual = np.asarray(apply(*args))
                    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
