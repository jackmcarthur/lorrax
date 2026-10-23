"""Distributed nodal fitting/refit against physical design-matrix oracles.

Run as a compute-node P4 script after the machine launch prelude.
"""
import sys


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from types import SimpleNamespace
    from scipy.optimize import nnls
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import resolve_device_budget_gb
    from isdf.galerkin import (GalerkinBasis, NodalInterp, _compact_rows,
        _coefficient_matrix, _loewdin, _nodal_face, _quadrature_metric,
        _quadrature_normal, _rank_carrier, iter_galerkin_rchunks)
    resolve_device_budget_gb(linalg='distributed')
    from common.collectives import prepare_mesh
    mesh = prepare_mesh(stack.mesh)
    p = int(mesh.shape['x'])
    rng = np.random.default_rng(277)
    for ns in (1, 2):
        labels = np.repeat(np.arange(7), [1, 3, 2, 1, 2, 1, 3])
        n, nk, nb = len(labels), 3, 5
        nmu = -(-n // p) * p
        phi = (rng.normal(size=(nk, nb, ns, n))
               + 1j*rng.normal(size=(nk, nb, ns, n))) / np.sqrt(n)
        arr = device_put_process_local(np.pad(phi,
            ((0, 0), (0, 0), (0, 0), (0, nmu-n))),
            NamedSharding(mesh, P(None, None, None, 'y')))
        a, b, order = _quadrature_normal(arr, labels, mesh, 1024)
        # Independent physical design; no normal-assembly code is reused.
        rows = np.einsum('kmsv,knsv->kmnv', phi.conj(), phi).reshape(-1, n)
        design = np.concatenate([rows.real, rows.imag]) @ np.eye(7)[labels]
        rhs = np.concatenate([np.broadcast_to(np.eye(nb), (nk, nb, nb)).ravel(),
                              np.zeros(rows.shape[0])])
        ao, bo = design.T@design, design.T@rhs
        ridge = 1e-14*np.trace(ao)/7
        got = np.asarray(gather_to_host(a))[:7, :7]
        np.testing.assert_allclose(got, ao[np.ix_(order[:7], order[:7])]
                                   + ridge*np.eye(7), atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(np.asarray(gather_to_host(b))[:7],
                                   bo[order[:7]], atol=1e-13)
        w, g = _quadrature_metric(arr, labels, mesh_xy=mesh, block_bytes=1024)
        np.testing.assert_allclose(w, nnls(design, rhs)[0][labels],
                                   atol=1e-12, rtol=1e-10)
        gg = np.asarray(gather_to_host(g))[:, :nb, :nb]
        np.testing.assert_allclose(gg, np.einsum('kmsv,v,knsv->kmn',
                                   phi.conj(), w, phi), atol=1e-12)
        inverse, _ = _loewdin(g, mesh_xy=mesh, n_bands=nb)
        vals, z = np.linalg.eigh(gg)
        expected = np.einsum('kmi,ki,kni->kmn', z, vals**-.5, z.conj())
        np.testing.assert_allclose(np.asarray(gather_to_host(inverse))[:, :nb, :nb],
                                   expected, atol=1e-11)
        for keep in (np.arange(n), np.array([0, 3, 4, 9, 11])):
            face = _nodal_face(arr, keep, mesh, extra_rank_pad=3)
            compact = np.asarray(gather_to_host(_compact_rows(face, nb, mesh)))
            want = phi[..., keep].reshape(nk, nb, -1)
            np.testing.assert_array_equal(compact[:, :, :want.shape[2]], want)
            assert not np.any(compact[:, :, want.shape[2]:])
        # Full refit, not just its normal matrix: independent least-squares
        # rows recover a known grid continuation, with uneven band/r slabs.
        nodes, nr = 7, 11
        physical = ns*nodes
        carrier = _rank_carrier(physical, mesh, 3)
        orthogonal = np.linalg.qr(rng.normal(size=(nk, physical, nb))
                      + 1j*rng.normal(size=(nk, physical, nb)))[0]
        c = orthogonal.transpose(0, 2, 1)
        cp = np.pad(c, ((0, 0), (0, 0), (0, carrier-physical)))
        coeff = device_put_process_local(cp,
                         NamedSharding(mesh, P(None, None, ('x', 'y'))))
        node_idx = tuple((i, 0, 0) for i in range(nodes))
        interp = NodalInterp(coeff, np.zeros((nodes, 3)), np.eye(3), 3, ns)
        basis = GalerkinBasis(coeff, interp, physical, (0, nb), node_idx,
                              refit_rcond=1e-7)
        continuation = rng.normal(size=(nodes, nr)) + 1j*rng.normal(size=(nodes, nr))
        values = np.einsum('kbsn,nr->kbsr', c.reshape(nk, nb, ns, nodes),
                            continuation)
        class Source:
            band_chunk_ranges = ((0, 3), (3, nb))
            band_chunk_carrier = -(-3 // p)*p
            def iter_rchunk_bandwise(self, r0, r1, *, product_r_spec):
                extent = -(-(r1-r0) // (p*p)) * (p*p)
                for lo, hi in self.band_chunk_ranges:
                    chunk = np.pad(values[:, lo:hi, :, r0:r1],
                        ((0, 0), (0, self.band_chunk_carrier-(hi-lo)),
                         (0, 0), (0, extent-(r1-r0))))
                    yield (lo, hi), device_put_process_local(chunk,
                        NamedSharding(mesh, product_r_spec))
        receipt = {}
        for r0, r1, rows, retained in iter_galerkin_rchunks(
                Source(), basis, SimpleNamespace(nspinor=ns, n_rtot=nr), mesh,
                r_chunk_ranges=((0, 5), (5, nr)), retained_band_range=(1, nb),
                receipt=receipt):
            got = np.asarray(gather_to_host(rows))
            want = np.zeros_like(got)
            for spin in range(ns):
                want[spin*nodes:(spin+1)*nodes, spin, :r1-r0] = continuation[:, r0:r1]
            np.testing.assert_allclose(got, want, atol=3e-11)
            for (lo, hi), part in retained:
                np.testing.assert_array_equal(np.asarray(gather_to_host(part))[:, :, :, :r1-r0],
                                               values[:, lo:hi, :, r0:r1])
        assert receipt['zeta'] < 1e-11, receipt
        assert receipt['fh'] < 1e-7, receipt
        if jax.process_index() == 0:
            print('GALERKIN_ORACLE PASS spin', ns, 'receipt', receipt, flush=True)
    runtime.finalize_process(0)
    return 0


def test_distributed_galerkin_requires_compute_node():
    import pytest
    pytest.skip('run tests/test_galerkin_distributed.py as a P4 compute-node script')


if __name__ == '__main__':
    sys.exit(main())
