"""P1/P4 complex PSD roots, padded support and actual-price route admission."""
import weakref
from unittest.mock import patch
import numpy as np
import runtime


def main():
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import resolve_device_budget_gb, shared_device_room
    from lxkit import memory
    from runtime.aot_memory import agreed_chunk, compiled_new_bytes
    import gw.response_bank as owner
    assert mesh.size in (1, 4)
    resolve_device_budget_gb(28, 'local')
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    cases = 0
    for n, logical, nq, negative in ((48, 37, 3, False), (128, 103, 7, False),
                                    (48, 37, 3, True)):
        # Analytic complex Fourier eigenvectors, independent of any solver.
        index = np.arange(logical, dtype=np.float64)
        u = np.exp(2j*np.pi*index[:, None]*index[None, :]/logical)/np.sqrt(logical)
        eigenvalues = np.linspace(.2, 3., logical)
        eigenvalues[:logical//4] = 0.
        eigenvalues[-1] = 1e-5
        if negative:
            eigenvalues[0] = -.03
        def carrier(diagonal):
            value = np.zeros((n, n), np.complex128)
            value[:logical, :logical] = (u*diagonal) @ u.conj().T
            return value
        values, expected_h, expected_hi, expected_rank = [], [], [], []
        for q in range(nq):
            lam = eigenvalues*(1 + .1*q)
            supported = lam > logical*np.finfo(float).eps*np.max(np.abs(lam))
            values.append(carrier(lam))
            expected_h.append(carrier(np.sqrt(np.maximum(lam, 0))))
            expected_hi.append(carrier(np.where(supported,
                1/np.sqrt(np.where(supported, lam, 1)), 0)))
            expected_rank.append(int(supported.sum()))
        value = device_put_process_local(np.asarray(values), face)
        jax.block_until_ready(value)
        kernels = {
            'local': owner._coulomb_algebra(mesh, n, logical, 'local',
                shared_device_room('Coulomb root admission oracle')),
            'distributed': owner._coulomb_algebra(mesh, n, logical, 'distributed')}
        prices, native = {}, {}
        for layout, kernel in kernels.items():
            executable = owner._compiled(kernel, (value,))
            native[layout] = owner.response_dense_workspace(mesh, n, nq, layout, with_eigh=True)
            prices[layout] = -agreed_chunk(-compiled_new_bytes(executable,
                extra=native[layout]['total']))
            if mesh.size > 1:
                prices[layout] += memory.contexts().at(mesh.size)
            raw = executable(value)
            jax.block_until_ready(raw)
            h, hi, bad, ranks = kernel.finish((value,), raw)
            np.testing.assert_allclose(np.asarray(gather_to_host(h)), expected_h,
                                       rtol=3e-9, atol=3e-10)
            np.testing.assert_allclose(np.asarray(gather_to_host(hi)), expected_hi,
                                       rtol=3e-8, atol=3e-8)
            np.testing.assert_array_equal(np.asarray(ranks), expected_rank)
            assert bool(bad) == negative
            assert h.sharding.spec == hi.sharding.spec == face.spec
            del raw, h, hi
        # Complete checked programs use the same full-mesh EIGH, even when
        # the reconstruction GEMMs use the local layout. This also gates P1.
        assert native['local']['eigh'] == native['distributed']['eigh']
        assert native['local']['total'] >= native['local']['eigh'] >= 0
        count = len(owner._COMPILED)
        roomy = max(prices.values()) + 1048576
        with patch.object(memory, 'room', return_value=roomy):
            chosen = owner._select_coulomb_program(kernels['local'], value,
                mesh_xy=mesh, n_logical=logical, layout='local')
        assert chosen is kernels['local']
        if prices['local'] > prices['distributed']:
            # A controlled room between measured complete-program prices:
            # test selection without fabricating an expensive device hold.
            between = int((prices['local'] + prices['distributed'])//2)
            with patch.object(memory, 'room', return_value=between):
                chosen = owner._select_coulomb_program(kernels['local'], value,
                    mesh_xy=mesh, n_logical=logical, layout='local')
            assert chosen is kernels['distributed']
        assert len(owner._COMPILED) == count
        ref = weakref.ref(value)
        del value
        assert ref() is None, 'compiled root cache retained its input array'
        cases += 1
    print('COULOMB_ROOT_ADMISSION PASS', int(mesh.size), cases,
          'complex/padded/negative cases, native query, controlled room and cache release', flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
