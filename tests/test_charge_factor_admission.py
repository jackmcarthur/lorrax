"""P4 charge factor versus an independent complex PSD inverse, cut and pad oracle."""
import weakref
import os
import runtime
import numpy as np


def main():
    platform = 'cpu' if os.environ.get('JAX_PLATFORMS', '').split(',')[0] == 'cpu' else 'gpu'
    mesh = runtime.initialize_communicator_stack(platform=platform).mesh
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import resolve_device_budget_gb
    from distrib_la import matmul, workspace_bytes_per_rank
    from lxkit import memory
    from isdf import core
    resolve_device_budget_gb(28, 'local')
    assert mesh.size in (1, 4)
    if jax.local_devices()[0].platform == 'cpu':
        assert memory.contexts().at(mesh.size) == 0
    nq, n, logical = 7, 48, 41
    index = np.arange(logical, dtype=np.float64)
    u = np.exp(2j*np.pi*index[:, None]*index[None, :]/logical)/np.sqrt(logical)
    eigenvalues = np.linspace(.5, 2., logical)
    eigenvalues[0] = 0
    c, expected = np.zeros((2, nq, n, n), np.complex128)
    for q in range(nq):
        lam = eigenvalues*(1 + .1*q)
        c[q, :logical, :logical] = (u*lam) @ u.conj().T
        inv = np.zeros_like(lam)
        inv[1:] = 1/lam[1:]
        expected[q, :logical, :logical] = (u*inv) @ u.conj().T
        c[q, logical:, logical:] = 99*np.eye(n-logical)
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    value = device_put_process_local(c, face)
    admitted, admit, force = [], memory.admit, None

    def route(stage, routes, room, **kwargs):
        chosen = admit(stage, routes, room, **kwargs)
        if stage == 'charge Cplus factor':
            admitted.append(chosen)
            return force or chosen
        return chosen

    for force in (None, 'whole'):
        memory.admit = route
        try:
            factor = core._factor_c_q_face(value, mesh, logical)
        finally:
            memory.admit = admit
        inverse = matmul(factor, factor, mesh=mesh, transb='C', backend='distributed')
        np.testing.assert_allclose(np.asarray(gather_to_host(inverse)), expected,
                                   rtol=2e-11, atol=2e-11)
        np.testing.assert_array_equal(np.asarray(gather_to_host(factor))[:, logical:], 0)
        del factor, inverse
    assert admitted
    if jax.local_devices()[0].platform == 'cpu':
        from distrib_la import plan
        eig = plan('eigh', mesh, n=n)
        assert workspace_bytes_per_rank(eig, 'eigh', ((n, n),), np.complex128, checked=True) == 0
    ref = weakref.ref(value)
    del value
    assert ref() is None, 'compiled factor cache retained its input array'
    print('CHARGE_FACTOR_ADMISSION PASS', int(mesh.size),
          'independent inverse, rank cut, pads, first/backup and cache release', flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
