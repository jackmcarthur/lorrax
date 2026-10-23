"""P4 W-factor placement and causal synthesis against an independent pole sum.

Run as a module under the qualified MPI/GPU rank wrapper. This reaches the
same face GEMM as the scalar and component-sector consumers, with ragged
parent panels, padded endpoints/columns, complex time and empty intervals.
"""


def main():
    import argparse
    import json
    from pathlib import Path

    import runtime
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from jax.experimental.multihost_utils import process_allgather
    from distrib_la import gemm_plan
    from gw.mpa.sigma import (_face_factor_panel, _tangential_factor_specs,
                              synthesize_tangential_parents)
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    from common.jax_compile_cache import compile_cache_stats

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def put(value, spec):
        value = np.asarray(value)
        return device_put_process_local(
            np.ascontiguousarray(value) if value.ndim else value, NamedSharding(mesh, spec))
    nq, mu, logical_mu, columns, logical_columns = 5, 12, 10, 14, 13
    intervals = np.array([[0, 13], [2, 9], [5, 5], [1, 12], [3, 13]], np.int32)
    poles = (1 + np.arange(nq)[:, None] / 16 + np.arange(columns)[None, :] / 32) ** 2
    reference, time = .17, .31 + .09j
    rows = []
    for components in (1, 3):
        q, m, c, k = np.indices((nq, mu, components, columns))
        b = ((1 + q / 8 + m / 32 + c / 64 + k / 128)
             + 1j * (q / 16 - m / 64 + c / 32 - k / 256)) / 32
        b[:, logical_mu:] = 0
        b[..., logical_columns:] = 0
        faces = tuple(put(b, spec) for spec in _tangential_factor_specs('face'))
        results = []
        for span in ((0, nq), (0, 3), (3, nq), (2, nq)):
            lo, hi = span
            before = compile_cache_stats()['compiles']
            formatted = tuple(_face_factor_panel(mesh, spec, hi-lo)(a, np.int32(lo))
                              for spec, a in zip(_tangential_factor_specs('face'), faces))
            jax.block_until_ready(formatted)
            if span == (2, nq):
                assert compile_cache_stats()['compiles'] == before, 'same-width offset recompiled'
            for value, spec in zip(formatted, _tangential_factor_specs('face')):
                assert value.sharding == NamedSharding(mesh, spec)
                assert value.format.layout.major_to_minor == (3, 0, 1, 2)
                local_index = value.sharding.devices_indices_map(value.shape)[jax.local_devices()[0]]
                np.testing.assert_array_equal(np.asarray(value.addressable_data(0)), b[lo:hi][local_index])
            gemm = gemm_plan(mesh, m=mu * components, k=columns, n=mu * components,
                             nq=hi-lo, dtype=np.complex128, layout='face',
                             enable_active_range=True, warmup=False)

            @jax.jit
            def node(x, y, p, bounds):
                return synthesize_tangential_parents(x, y, p, bounds, reference, time,
                    mesh_xy=mesh, gemm=gemm, layout='face', active_range=True)

            actual = node(*formatted, put(poles[lo:hi], P()), put(intervals[lo:hi], P()))
            actual = [np.asarray(process_allgather(x, tiled=True)) for x in actual]
            expected = np.zeros((hi-lo, mu*components, mu*components), np.complex128)
            for iq, parent in enumerate(range(lo, hi)):
                # An explicit physical-column sum; no production weight/helper is used.
                for column in range(*intervals[parent]):
                    omega = np.sqrt(poles[parent, column])
                    weight = np.exp(-1j * (omega-reference) * time) / (2*omega)
                    vector = b[parent, :, :, column].reshape(-1)
                    expected[iq] += weight * np.outer(vector, vector.conj())
            np.testing.assert_allclose(actual[0], expected, rtol=3e-12, atol=3e-13)
            np.testing.assert_allclose(actual[1], expected.swapaxes(-1, -2), rtol=3e-12, atol=3e-13)
            assert np.all(actual[0].reshape(hi-lo, mu, components, mu, components)[:, logical_mu:] == 0)
            results.append(actual[0])
            rows.append(dict(components=components, span=list(span), max_error=float(np.max(np.abs(actual[0]-expected)))))
            if span == (0, nq):
                # The compiled-program cache must distinguish actual input
                # formats, even when logical shapes and shardings match.
                def tau(x, y, p, bounds, t, _active):
                    return synthesize_tangential_parents(x, y, p, bounds, reference, t,
                        mesh_xy=mesh, gemm=gemm, layout='face', active_range=True)[0]
                acc = DeviceOmegaAccumulator.grid(put(np.array([0.]), P()),
                    shape=(1, nq, mu*components, mu*components),
                    sharding=NamedSharding(mesh, P(None, None, 'x', 'y')), omega_axis=0)
                tail = (put(poles, P()), put(intervals, P()))
                times, coeff = put(np.array([time]), P()), put(np.ones((1, 1), complex), P())
                one, count = put(np.array(1, np.int32), P()), put(np.array(0, np.int32), P())
                c_program = acc.run_window(tau, (*faces, *tail), times, coeff, one, count, compile_only=True)
                f_program = acc.run_window(tau, (*formatted, *tail), times, coeff, one, count, compile_only=True)
                assert c_program is not f_program
                assert f_program is acc.run_window(tau, (*formatted, *tail), times, coeff, one, count, compile_only=True)
                abstract = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
                                 for x in faces)
                acc.run_window(tau, (*abstract, *tail), times, coeff, one, count, compile_only=True)
                got = acc.run_window(tau, (*formatted, *tail), times, coeff, one, count)
                got = np.asarray(process_allgather(got, tiled=True))[0]
                np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-13)
        np.testing.assert_allclose(np.concatenate(results[1:3]), results[0], rtol=3e-12, atol=3e-13)
    # A mixed scalar/current endpoint has a rectangular operator. Its partner
    # conjugates the factors, never the causal weight or the whole operator.
    q, m, c, k = np.indices((nq, mu, 3, columns))
    right = ((1 + q/8 + m/32 + c/16 + k/128)
             + 1j*(q/32 - m/64 + c/8 + k/256))/32
    left = (right[:, :, :1] * (.5+.25j)).copy()
    left[:, logical_mu:] = right[:, logical_mu:] = 0
    left[..., logical_columns:] = right[..., logical_columns:] = 0
    specs = _tangential_factor_specs('face')
    factors = tuple(_face_factor_panel(mesh, spec, nq)(put(value, spec), np.int32(0))
                    for spec, value in zip(specs, (left, right)))
    mixed_gemm = gemm_plan(mesh, m=mu, k=columns, n=3*mu, nq=nq,
                           dtype=np.complex128, layout='face',
                           enable_active_range=True, warmup=False)
    @jax.jit
    def mixed(x, y, p, bounds):
        return synthesize_tangential_parents(x, y, p, bounds, reference, time,
            mesh_xy=mesh, gemm=mixed_gemm, layout='face', active_range=True, same_factor=False)
    got = [np.asarray(process_allgather(x, tiled=True)) for x in
           mixed(*factors, put(poles, P()), put(intervals, P()))]
    expected = [np.zeros((nq, mu, 3*mu), complex) for _ in range(2)]
    for parent in range(nq):
        for column in range(*intervals[parent]):
            omega = np.sqrt(poles[parent, column])
            weight = np.exp(-1j*(omega-reference)*time)/(2*omega)
            a, b = left[parent, :, :, column].reshape(-1), right[parent, :, :, column].reshape(-1)
            expected[0][parent] += weight*np.outer(a, b.conj())
            expected[1][parent] += weight*np.outer(a.conj(), b)
    for actual, reference_value in zip(got, expected):
        np.testing.assert_allclose(actual, reference_value, rtol=3e-12, atol=3e-13)
    rows.append(dict(components=[1, 3], span=[0, nq],
                     max_error=max(float(np.max(np.abs(a-b)))for a,b in zip(got,expected))))
    result = dict(verdict='PASS', cases=rows, logical_padding_zero=True,
                  scope='Actual face-factor placement and common causal W synthesis; independent pole sum, not a whole-GW peak')
    (args.out / f'rank{jax.process_index()}.json').write_text(json.dumps(result, indent=2)+'\n')
    print('TANGENTIAL_FACTOR_LAYOUT PASS', jax.process_index(), flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
