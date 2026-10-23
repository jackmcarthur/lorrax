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
    rmu = np.column_stack(np.unravel_index(
        np.random.default_rng(504).choice(total, nmu, replace=False), grid)).astype(np.int32)
    kvec = np.array([[.137, -.23, .31], [.41, .17, -.11], [-.19, .37, .07]])
    psi = device_put_process_local(psi_host, NamedSharding(stack.mesh, band_sphere_spec()))
    indices = device_put_process_local(gidx.astype(np.int32), NamedSharding(stack.mesh, P()))
    maximum = 0.
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


if __name__ == "__main__":
    main()
