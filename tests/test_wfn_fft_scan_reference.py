"""Run directly under MPI: sparse Fourier oracle for the bounded WFN FFT scan."""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.wfn_layout import band_sphere_spec
    from common.wfn_transforms import gflat_to_rmu

    grid = (7, 5, 4)
    total, nk, ns, ng, nmu = int(np.prod(grid)), 3, 2, 23, 17
    logical = int(stack.mesh.size)+1
    nb = -(-logical//int(stack.mesh.size))*int(stack.mesh.size)
    k, b, s, g = np.ix_(np.arange(nk), np.arange(nb), np.arange(ns), np.arange(ng))
    psi_host = np.sin(.21*k+.17*b+.13*s+.07*g) + 1j*np.cos(.19*k-.23*b+.29*s+.11*g)
    psi_host[:, logical:] = 0.
    gidx = (np.arange(ng)[None, :]*13+np.arange(nk)[:, None]*17) % total
    gidx[:, 20:] = total+np.arange(20, ng)
    all_rmu = np.column_stack(np.unravel_index(
        np.random.default_rng(504).choice(total, nmu, replace=False), grid)).astype(np.int32)
    kvec = np.array([[.137, -.23, .31], [.41, .17, -.11], [-.19, .37, .07]])
    psi = device_put_process_local(psi_host, NamedSharding(stack.mesh, band_sphere_spec()))
    indices = device_put_process_local(gidx.astype(np.int32), NamedSharding(stack.mesh, P()))
    maximum = 0.
    for nmu in (1, 17):
        rmu = all_rmu[:nmu]
        for phase in (False, True):
            for mapped in (False, True):
                row_map = np.array([2, 0, 1], dtype=np.int32) if mapped else np.arange(nk)
                expected = np.zeros((nk, nb, ns, nmu), dtype=np.complex128)
                for original_k, table_k in enumerate(row_map):
                    cells = np.column_stack(np.unravel_index(gidx[table_k, :20], grid))
                    exponent = np.exp(2j*np.pi*(cells/np.array(grid)) @ rmu.T)
                    expected[original_k] = np.einsum(
                        'bsg,gm->bsm', psi_host[original_k, :, :, :20], exponent)/np.sqrt(total)
                    if phase:
                        expected[original_k] *= np.exp(2j*np.pi*(rmu/np.array(grid)) @ kvec[table_k])
                for cs in (1, 2, 3, 5, nk*(nb//int(stack.mesh.size))):
                    actual = gflat_to_rmu(psi, indices, rmu, mesh=stack.mesh,
                        fft_grid=grid, norm='ortho', chunk_size=cs,
                        kvecs_frac=kvec if phase else None,
                        k_row_map=row_map if mapped else None)
                    host = np.asarray(gather_to_host(actual))
                    maximum = max(maximum, float(np.max(np.abs(host-expected))))
                    np.testing.assert_allclose(host, expected, rtol=2e-13, atol=2e-13)
                    np.testing.assert_array_equal(host[:, logical:], 0.)
                    actual.delete()
    if jax.process_index() == 0:
        print(f'FFT_SCAN_REFERENCE PASS: phase/map/chunks/padding, max_abs={maximum:.3g}', flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip("run this module directly with the configured MPI/GPU launcher")


def test_flat_box_indices_preserve_large_offsets():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from common.wfn_transforms import _box_kernel, _flat_box_indices

    with jax.enable_x64():
        huge = jnp.array([[0, 1_500_000_000], [1_499_999_999, 1_500_000_000]],
                         dtype=jnp.int32)
        wide = jax.jit(lambda x: _flat_box_indices(x, 1_500_000_000))(huge)
        np.testing.assert_array_equal(wide, [[0, 1_500_000_000], [3_000_000_000, 3_000_000_001]])
        assert wide.dtype == jnp.int64
        clipped = jax.jit(lambda i: jnp.take(jnp.array([11, 17]), i, mode='clip'))(
            jnp.array([3_000_000_001], dtype=jnp.int64))
        np.testing.assert_array_equal(clipped, [17])
        inverse = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        ordinary = _flat_box_indices(inverse, 3)
        np.testing.assert_array_equal(ordinary, [[0, 1], [6, 7]])
        assert ordinary.dtype == jnp.int32
        for ngk, dtype in ((1_073_741_823, jnp.int32), (1_073_741_824, jnp.int64)):
            boundary = _flat_box_indices(jnp.full((2, 2), ngk, jnp.int32), ngk)
            np.testing.assert_array_equal(boundary, [[ngk, ngk], [2 * ngk + 1, 2 * ngk + 1]])
            assert boundary.dtype == dtype
        # Independent sparse scatter checks the actual ordinary box gather.
        psi = np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(np.complex128)
        sphere = np.array([[0, 2, 5], [1, 3, 4]], dtype=np.int32)
        expected = np.zeros((2, 2, 2, 6), dtype=np.complex128)
        for k in range(2):
            for g, cell in enumerate(sphere[k]):
                expected[k, :, :, cell] = psi[k, :, :, g]
        actual = jax.jit(lambda x, i: _box_kernel(x, i, fft_grid=(2, 3, 1)))(
            jnp.asarray(psi), jnp.asarray(sphere))
        np.testing.assert_array_equal(actual, expected.reshape(2, 2, 2, 2, 3, 1))


if __name__ == "__main__":
    main()
