"""Bounded-trial physical R/A rungs against literal two-particle ISDF sums.

CPU FFT/BLAS references avoid native FFI. Four virtual CPU devices also
exercise the 2x2 mesh. Independent random W deliberately need not be real
or Hermitian: all four mixed branch contractions must agree separately.
"""
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from bse.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings
from bse.bse_w_exact import build_finite_q_data
from bse.bse_feast import ladder_matvec_operands
from ffi.fft import xla_reference

jax.config.update('jax_enable_x64', True)


def _fixture(mesh, ns, batch, q, *, mu=8, nk=3):
    rng = np.random.default_rng(241 + ns)
    complex_array = lambda shape: rng.normal(size=shape) + 1j*rng.normal(size=shape)
    nc, nv = 4, 2
    c, v = complex_array((nk, nc, ns, mu)), complex_array((nk, nv, ns, mu))
    w = complex_array((nk, mu, mu)) / (mu*ns)
    wr = np.fft.ifft(w, axis=0, norm='ortho').transpose(1, 2, 0)[..., None, None]
    x = complex_array((2, batch, nc, nv, nk))
    sh = make_bse_shardings(mesh)
    put = lambda a, s: jax.device_put(jnp.asarray(a), s)
    data = dict(nkx=nk, nky=1, nkz=1,
        psi_c_X=put(c, sh.psi_x), psi_c_Y=put(c, sh.psi_y),
        psi_v_X=put(v, sh.psi_x), psi_v_Y=put(v, sh.psi_y),
        eps_c=put(np.zeros((nk, nc)), sh.eps),
        eps_v=put(np.zeros((nk, nv)), sh.eps), W_R=put(wr, sh.W),
        V_q_full=put(np.zeros((mu, mu, nk, 1, 1), complex), sh.W))
    dq = build_finite_q_data(data, (q, 0, 0), mesh, physical_branches=True)
    return (put(x, sh.X_full), *ladder_matvec_operands(dq)), (x, c, v, w, q)


def _dense_rung(x, c, v, w, q):
    nk, nc, _ns, _mu = c.shape
    nv = v.shape[1]
    cq, vq = c[(np.arange(nk)+q) % nk], v[(np.arange(nk)+q) % nk]
    blocks = np.zeros((2, 2, nc, nv, nk, nc, nv, nk), complex)
    for k in range(nk):
        for j in range(nk):
            wk = w[(k-j) % nk]
            blocks[0, 0, :, :, k, :, :, j] = np.einsum('ctm,dtm,mn,vsn,wsn->cvdw',
                cq[k].conj(), cq[j], wk, v[k], v[j].conj())/nk
            blocks[0, 1, :, :, k, :, :, j] = np.einsum('ctm,wtm,mn,vsn,dsn->cvdw',
                cq[k].conj(), vq[j], wk, v[k], c[j].conj())/nk
            blocks[1, 0, :, :, k, :, :, j] = np.einsum('vtm,dtm,mn,csn,wsn->cvdw',
                vq[k].conj(), cq[j], wk, c[k], v[j].conj())/nk
            blocks[1, 1, :, :, k, :, :, j] = np.einsum('vtm,wtm,mn,csn,dsn->cvdw',
                vq[k].conj(), vq[j], wk, c[k], c[j].conj())/nk
    blocks = blocks.reshape(2, 2, nc*nv*nk, nc*nv*nk)
    xx = x.reshape(2, x.shape[1], -1)
    return np.stack([-xx[0] @ blocks[0, 0].T - xx[1] @ blocks[0, 1].T,
                      xx[0] @ blocks[1, 0].T + xx[1] @ blocks[1, 1].T]).reshape(x.shape)


def test_distributed_ladder_trials_match_literal_dense_matrix(monkeypatch):
    import common.contract_bands as cb
    monkeypatch.setattr(cb, 'bands_gemm_ffi_enabled', lambda: False)
    devices = jax.devices('cpu')
    meshes = [Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))]
    if len(devices) >= 4:
        meshes.append(Mesh(np.array(devices[:4]).reshape(2, 2), ('x', 'y')))
    for mesh in meshes:
        for ns, batch in ((1, 1), (2, 3)):
            with xla_reference():
                apply = build_bse_ring_matvec_full(mesh, 3, 1, 1, physical_branches=True)
                for q in (0, 1):
                    args, source = _fixture(mesh, ns, batch, q)
                    actual = np.asarray(apply(*args))
                    np.testing.assert_allclose(actual, _dense_rung(*source),
                                               rtol=2e-12, atol=2e-12)
