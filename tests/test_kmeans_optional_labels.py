"""Skipping unused final labels preserves literal/orbit Lloyd results.

Compute-node script under the production MPI launcher. The odd 7³ grid
exercises mesh padding; explicit NumPy nearest-image distances validate the
default labels independently. A second default run separates ordinary GPU
reduction roundoff from any change caused by omitting the final labels.
"""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import itertools
    import jax
    import numpy as np
    from common.collectives import gather_to_host
    from centroid.kmeans_isdf import weighted_kmeans_jax
    side, nc = 7, 9
    points = np.asarray(np.meshgrid(*[np.arange(side)/side]*3, indexing='ij')).reshape(3, -1).T
    rho = (1+np.sin(np.arange(side**3)*.137)**2).reshape((side,)*3)
    avec = np.asarray([[1., 0, 0], [.25, 1.3, 0], [.1, .15, 1.7]])
    metric = avec @ avec.T
    offsets = np.asarray(list(itertools.product(range(-2, 3), repeat=3)))
    for orbit in (False, True):
        syms = np.stack((np.eye(3, dtype=np.int32), -np.eye(3, dtype=np.int32)))
        kwargs = dict(R=syms, Rinv=syms, tau=np.zeros((2, 3))) if orbit else {}
        results = []
        for enabled in (True, False, True):
            output = {} if enabled else dict(return_labels=False)
            labels, centers, steps, movement = weighted_kmeans_jax(
                avec, rho, nc, max_steps=2, tolerance=0., seed=504,
                mesh=stack.mesh, mesh_axis=('x', 'y'), init_method='random',
                **kwargs, **output)
            host = np.asarray(gather_to_host(centers))
            results.append((host, steps, movement))
            if enabled:
                labels = np.asarray(gather_to_host(labels))
                assert np.all(labels[side**3:] == -1)
                chosen = labels[:side**3]
                distances = np.full((len(points), nc), np.inf)
                for image in (host, (-host) % 1.) if orbit else (host,):
                    delta = points[:, None, :]-image[None, :, :]
                    delta -= np.round(delta)
                    delta = delta[:, :, None, :]+offsets[None, None, :, :]
                    distances = np.minimum(distances,
                        np.einsum('pcsi,ij,pcsj->pcs', delta, metric, delta).min(axis=2))
                assert np.all(distances[np.arange(len(points)), chosen]
                              <= distances.min(axis=1)+2e-14)
            else:
                assert labels is None
        for name, result in zip(('omitted', 'repeated'), results[1:]):
            difference = np.max(np.abs(results[0][0]-result[0]))
            if jax.process_index() == 0:
                print(f'OPTIONAL_LABELS orbit={orbit} {name} '
                      f'center_max_abs={difference:.3g}', flush=True)
            # GPU segmented reductions need not sum in the same order on two
            # identical calls. Require agreement far below the label-distance
            # tolerance, and independently check each default label array.
            np.testing.assert_allclose(result[0], results[0][0], rtol=0., atol=2e-14)
            assert result[1] == results[0][1]
            np.testing.assert_allclose(result[2], results[0][2], rtol=0., atol=2e-14)
    if jax.process_index() == 0:
        print('KMEANS_OPTIONAL_LABELS PASS: literal/orbit, ragged grid, independent labels', flush=True)
    runtime.finalize_process(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
