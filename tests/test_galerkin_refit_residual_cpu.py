"""Streamed Galerkin residuals: physical continuation, band clips and carriers."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('ns', (1, 2))
def test_refit_residual_reuses_full_blocks_but_preserves_clips_and_tail(monkeypatch, ns):
    import distrib_la
    from common.gpu_utils import resolve_device_budget_gb
    from isdf import galerkin as G
    if jax.device_count() < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    resolve_device_budget_gb(linalg='distributed')
    # Local units use the existing exact JAX eigensolver on the tiny Gram;
    # native P4 refit controls separately exercise the distributed provider.
    original_plan = distrib_la.plan

    def reference_plan(op, mesh, **kw):
        assert op == 'eigh'
        return original_plan(op, mesh, **dict(kw, backend='off'))

    monkeypatch.setattr(distrib_la, 'plan', reference_plan)
    p, nk, nb, nodes, nr = 2, 3, 5, 7, 13
    rng = np.random.default_rng(311 + ns)
    physical = ns * nodes
    carrier = G._rank_carrier(physical, mesh, 3)
    q = np.linalg.qr(rng.normal(size=(nk, physical, nb))
                     + 1j * rng.normal(size=(nk, physical, nb)))[0]
    c = q.transpose(0, 2, 1)
    cp = np.pad(c, ((0, 0), (0, 0), (0, carrier - physical)))
    coeff = jax.device_put(cp, NamedSharding(mesh, P(None, None, ('x', 'y'))))
    interp = G.NodalInterp(coeff, np.zeros((nodes, 3)), np.eye(3), 3, ns)
    basis = G.GalerkinBasis(coeff, interp, physical, (0, nb),
        tuple((i, 0, 0) for i in range(nodes)), refit_rcond=1e-7)
    continuation = rng.normal(size=(nodes, nr)) + 1j * rng.normal(size=(nodes, nr))
    values = np.einsum('kbsn,nr->kbsr', c.reshape(nk, nb, ns, nodes), continuation)

    class Source:
        band_chunk_ranges = ((0, 3), (3, nb))
        band_chunk_carrier = 4

        def iter_rchunk_bandwise(self, r0, r1, *, product_r_spec):
            extent = -(-(r1-r0) // (p*p)) * (p*p)
            for lo, hi in self.band_chunk_ranges:
                chunk = np.pad(values[:, lo:hi, :, r0:r1],
                    ((0, 0), (0, self.band_chunk_carrier-(hi-lo)),
                     (0, 0), (0, extent-(r1-r0))))
                yield (lo, hi), jax.device_put(chunk, NamedSharding(mesh, product_r_spec))

    calls = []
    original = G._coefficient_matrix

    def observe(*a, **kw):
        calls.append((kw.get('band_range'), kw.get('band_carrier')))
        return original(*a, **kw)

    monkeypatch.setattr(G, '_coefficient_matrix', observe)
    for keep in ((0, 3), (0, 5), (1, 4), (3, 5)):
        calls.clear()
        receipt = {}
        for r0, r1, rows, retained in G.iter_galerkin_rchunks(
                Source(), basis, SimpleNamespace(nspinor=ns, n_rtot=nr), mesh,
                r_chunk_ranges=((0, 3), (3, 8), (8, nr)),
                retained_band_range=keep, receipt=receipt):
            got = np.asarray(rows)
            want = np.zeros_like(got)
            for spin in range(ns):
                want[spin*nodes:(spin+1)*nodes, spin, :r1-r0] = continuation[:, r0:r1]
            np.testing.assert_allclose(got, want, atol=3e-11)
            for (lo, hi), part in retained:
                np.testing.assert_array_equal(np.asarray(part)[:, :, :, :r1-r0],
                                               values[:, lo:hi, :, r0:r1])
        assert receipt['zeta'] < 1e-11, receipt
        assert receipt['fh'] < 1e-7, receipt
        # The independent continuation/residual checks above are primary.
        # Count only redundant coefficient constructions: the already-live
        # full block must not be rebuilt once per spatial slab. A clipped
        # piece or shorter tail still takes the original construction.
        residual = [bands for bands, width in calls if width is None]
        if keep in ((0, 3), (0, 5)):
            assert (0, 3) not in residual
        if keep == (1, 4):
            assert (1, 3) in residual and (3, 4) in residual
        if keep in ((0, 5), (3, 5)):
            assert (3, 5) in residual  # default2 differs from the source's4
