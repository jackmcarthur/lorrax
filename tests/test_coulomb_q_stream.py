"""P4 independent all-q statistics and runtime transport from canonical parents.

The q grid is ragged with respect to both the panel and processor count.
Complex parents, antiunitary rows, nonlocal permutations and wraps test the
same owner as scalar and photon charge initialization; one active centroid
also exercises ranks whose physical block is empty.
"""
import runtime


def main():
    mesh = runtime.initialize_communicator_stack().mesh
    import gc
    import weakref
    import dataclasses
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import device_budget_bytes
    from common import sanity
    from symmetry_maps import QirrTables, QirrOperator
    from gw.gw_init import _canonical_coulomb_stats, _full_coulomb_for_restart

    assert mesh.size == 4
    device_budget_bytes()
    rng = np.random.default_rng(20261023)
    place = lambda a: device_put_process_local(a, NamedSharding(mesh, P(None, 'x', 'y')))
    query = lambda a: device_put_process_local(np.asarray(a, np.int32), NamedSharding(mesh, P()))
    errors = []
    for logical, nq in ((13, 67), (1, 9)):
        n = 16
        parents = np.zeros((4, n, n), complex)
        for parent in range(4):
            a = rng.normal(size=(logical, logical)) + 1j*rng.normal(size=(logical, logical))
            parents[parent, :logical, :logical] = a @ a.conj().T
        parents[0] = parents[0].real
        perm = np.tile(np.arange(n, dtype=np.int32), (4, 1))
        perm[1, :logical] = np.roll(np.arange(logical), 3)
        perm[3] = perm[1]
        wraps = np.zeros((4, n, 3), np.int64)
        wraps[1, :logical] = rng.integers(-2, 3, (logical, 3))
        wraps[3] = wraps[1]
        irr, sym = np.zeros(nq, np.int32), np.zeros(nq, np.int32)
        irr[1:4] = np.arange(1, 4)
        for row in range(1, nq//2+1):
            if row >= 4:
                irr[row], sym[row] = 1 + row % 3, 1
            irr[nq-row], sym[nq-row] = irr[row], sym[row]+2
        q = rng.uniform(-.5, .5, (4, 3)); q[0] = 0
        tables = QirrTables(irr, sym, q, perm, wraps, 2)
        source = place(parents)
        operator = QirrOperator.from_tables(source, tables)
        whole = QirrOperator.whole_zone(source)
        assert whole.unfold(mesh, prepare=True)[0] is whole.unfold(mesh, prepare=True)[0]
        np.testing.assert_array_equal(np.asarray(gather_to_host(
            _full_coulomb_for_restart(whole, mesh))), parents)
        # Literal NumPy permutation/phase/conjugation, independent of the service.
        literal = []
        for parent, action in zip(irr, sym):
            phase = np.exp(2j*np.pi*(wraps[action] @ q[parent]))
            a = parents[parent][perm[action, :, None], perm[action, None, :]]
            a = phase[:, None]*a*phase.conj()[None, :]
            literal.append(a.conj() if action >= 2 else a)
        literal = np.asarray(literal)
        full = np.asarray(gather_to_host(operator.unfold(mesh)))
        np.testing.assert_allclose(full, literal, rtol=2e-13, atol=2e-13)
        np.testing.assert_array_equal(np.asarray(gather_to_host(
            _full_coulomb_for_restart(operator, mesh))), full)
        program = jax.jit(lambda a, rows: operator.with_values(a).unfold(mesh, rows=rows))
        # Same shape, changed runtime q rows; includes repeats and noncontiguous order.
        before = program._cache_size()
        for rows in ([0, 8, 2, 5, 3], [8, 1, 8, 0, 7], [6, 5, 4, 3, 2]):
            actual = np.asarray(gather_to_host(program(source, query(rows))))
            np.testing.assert_allclose(actual, literal[rows], rtol=2e-13, atol=2e-13)
        assert program._cache_size() == before + 1
        totals, pair, gamma = _canonical_coulomb_stats(operator, (nq, 1, 1))
        negative = (-np.arange(nq)) % nq
        expected = (float(np.max(np.abs(full-np.conj(full[negative])))),
                    float(np.max(np.abs(full))))
        np.testing.assert_allclose(pair, expected, rtol=1e-13, atol=1e-13)
        assert totals[:2] == (0, 0) and totals[3] == full.size
        np.testing.assert_allclose(totals[2], np.max(np.abs(full)), rtol=1e-13)
        np.testing.assert_array_equal(np.asarray(gather_to_host(gamma)), full[0])
        with sanity.forced_level('strict'):
            assert sanity.report_finite_stats('oracle all q', totals)
            assert sanity.report_q_conjugate_residual('oracle all q', *pair, rtol=1e-12)
        for invalid in (np.asarray([-1], np.int32), np.asarray([nq], np.int32),
                        np.asarray([1.0]), np.empty(0, np.int32), np.zeros((1, 1), np.int32)):
            try:
                operator.unfold(mesh, rows=invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid full-zone row was not refused')
        damaged_query = np.asarray(gather_to_host(program(source, query([0, -1, 2, nq, 3]))))
        assert np.isnan(damaged_query[[1, 3]]).all()
        np.testing.assert_array_equal(damaged_query[[0, 2, 4]], full[[0, 2, 3]])
        for dtype in (np.int64, np.uint64):
            wide = device_put_process_local(np.asarray([0, 2**32, 2, 2**32+1, 3], dtype),
                                            NamedSharding(mesh, P()))
            result = np.asarray(gather_to_host(program(source, wide)))
            assert np.isnan(result[[1, 3]]).all()
            np.testing.assert_array_equal(result[[0, 2, 4]], full[[0, 2, 3]])
        corrupt = dataclasses.replace(operator, full_rows=np.asarray([0, 1, 2, 8]))
        try:
            _canonical_coulomb_stats(corrupt, (nq, 1, 1))
        except ValueError:
            pass
        else:
            raise AssertionError('untrusted selected rows were accepted')
        # Nonfinite counts must count every full-zone occurrence, not parent counts.
        bad = parents.copy(); bad[1, 0, 0] = np.nan + 1j*np.inf
        bad_op = operator.with_values(place(bad))
        stats, _, _ = _canonical_coulomb_stats(bad_op, (nq, 1, 1))
        bad_full = np.asarray(gather_to_host(bad_op.unfold(mesh)))
        finite = np.isfinite(bad_full.real) & np.isfinite(bad_full.imag)
        nan = np.isnan(bad_full.real) | np.isnan(bad_full.imag)
        assert stats[:2] == (int((~finite).sum()), int(nan.sum()))
        try:
            with sanity.forced_level('strict'):
                sanity.report_finite_stats('oracle damaged all q', stats, print_fn=lambda *_: None)
        except sanity.SanityError:
            pass
        else:
            raise AssertionError('nonfinite all-q gate did not refuse')
        errors.append(float(np.max(np.abs(full-literal))))
        finite_only, no_pair, g0 = _canonical_coulomb_stats(operator, (nq, 1, 1), reciprocity=False)
        assert finite_only == totals and no_pair == (0., 0.)
        with sanity.forced_level('off'):
            disabled, no_pair, only_gamma = _canonical_coulomb_stats(operator, (nq, 1, 1))
        assert disabled[:3] == (0, 0, 0.) and no_pair == (0., 0.)
        np.testing.assert_array_equal(np.asarray(gather_to_host(only_gamma)), full[0])
        # Caches may retain geometry/programs, never the input operator or arrays.
        ref = weakref.ref(bad_op.values)
        del bad_op, bad_full
        jax.effects_barrier(); gc.collect()
        assert ref() is None
    print('COULOMB_Q_STREAM PASS', jax.process_index(), errors, flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
