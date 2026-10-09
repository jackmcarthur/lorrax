"""Q-owner physical-provider parity for exact prepared spline coefficients."""
from pathlib import Path
from types import SimpleNamespace
import json
import os


def check_prepared_provider(runtime):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from isdf.atomic_coulomb import radial_coulomb_provider
    from isdf.coulomb_fourier_cache import load_coulomb_fourier_cache
    from isdf.zeta_mubatch import _local_coefficient_solve

    mesh = runtime.mesh
    assert mesh.size == 4
    base = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/780_augmented_isdf_20261006')
    manifest = json.loads((base/'atomic/manifest_cached_native_spd_w1e08_r8800_adaptive96f01_l6_packet48_fullband_unitdiagonal_onsitecross_servedmonopole/manifest.json').read_text())
    radial = manifest['radial']
    artifact = base/'local_coulomb_fourier_artifacts_v1/adaptive96_l6_d5_80ry.npz'
    sha = 'd70b5299499c767b0e7b7dde8e4c13c0b6e77e7300fd2685c6f689859b09c16d'
    prepared = load_coulomb_fourier_cache(artifact, expected_file_sha256=sha)
    lm = np.asarray([(l, m) for l in range(7) for m in range(-l, l+1)], int)
    Q, Qp, mu, na, ng, gt = 3, 4, 8, 2, 11, 4
    nr, nh = len(radial['radius']), len(lm)
    nf = na*nr*nh
    rng = np.random.default_rng(610209)
    random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    spec = NamedSharding(mesh, P(('x', 'y'), None, None))
    put = lambda value: device_put_process_local(np.asarray(value), spec)
    factor = np.stack([np.linalg.qr(random((mu, mu)))[0]/np.sqrt(np.arange(1, mu+1))[None]
                       for _ in range(Qp)])
    factor[Q:] = 0.
    rhs, smooth_rhs, moment_rhs = random((Qp, mu, nf))*.003, random((Qp, mu, nf))*.007, random((Qp, mu, na))*.005
    rhs[Q:] = 1e8*(1+3j)
    smooth_rhs[Q:] = 2e8*(2-1j)
    moment_rhs[Q:] = -3e8*(1+2j)
    vectors = rng.normal(size=(Q, ng, 3))*1.8
    vectors[0, 0] = 0.
    assert np.linalg.norm(vectors, axis=-1).max() < prepared['maximum_wavevector']
    # Match the incumbent factory's interval exactly for the bitwise control.
    vectors[0, 1] = (prepared['maximum_wavevector'], 0., 0.)
    ngk = np.asarray([11, 8, 10])
    stub = SimpleNamespace(mesh=mesh, factor=put(factor), solver_kind='rank_truncate', n_rmu_solve=mu,
        ngk_per_q=ngk, store=SimpleNamespace(Q=Q, Q_pad=Qp, mu_pad=mu, g_tile=gt,
        g_axis=SimpleNamespace(logical=ng)))
    volume, nfft = 463.12290632887095, 125000
    options = dict(radius=radial['radius'], weights_dr=radial['weights_dr'], lm=lm,
        centers_cart=np.asarray([[.2, -.1, .4], [5.6, .5, -.3]]), q_plus_G_cart=vectors,
        cell_volume=volume, fft_points=nfft, support_radius=radial['support_radius'],
        interpolation_degree=radial['interpolation_degree'], quadrature_order=radial['quadrature_order'],
        fourier_points=4097)
    incumbent = radial_coulomb_provider(stub, put(rhs), smooth_rhs=put(smooth_rhs),
        monopole_rhs=put(moment_rhs), **options)
    cached = radial_coulomb_provider(stub, put(rhs), smooth_rhs=put(smooth_rhs),
        monopole_rhs=put(moment_rhs), prepared_cache=prepared, **options)
    solve = _local_coefficient_solve(mesh, stub.solver_kind, mu, Q)
    coefficients = solve(stub.factor, incumbent['rhs'])
    cached_coefficients = solve(stub.factor, cached['rhs'])
    host = lambda value: np.asarray(gather_to_host(value))
    same_coefficients = np.array_equal(host(coefficients), host(cached_coefficients))
    onsite, onsite_cached = host(incumbent['onsite'](coefficients)), host(cached['onsite'](cached_coefficients))
    body, body_cached = np.zeros((Qp, mu, mu), complex), np.zeros((Qp, mu, mu), complex)
    max_delta = max_comp = 0.
    for tile in range((ng+gt-1)//gt):
        delta, comp = map(host, incumbent['fourier_tile'](tile, coefficients))
        delta_cached, comp_cached = map(host, cached['fourier_tile'](tile, cached_coefficients))
        max_delta = max(max_delta, float(abs(delta-delta_cached).max()))
        max_comp = max(max_comp, float(abs(comp-comp_cached).max()))
        smooth = random((Qp, mu, gt))*.04
        weights = np.zeros((Qp, gt))
        for q in range(Q):
            count = max(0, min(gt, int(ngk[q])-tile*gt))
            K = vectors[q, tile*gt:tile*gt+count]
            k2 = np.sum(K*K, axis=-1)
            weights[q, :count] = np.divide(8*np.pi/volume, k2,
                out=np.zeros_like(k2), where=k2 > 0)
        body += np.einsum('qmg,qg,qng->qmn', (smooth+comp).conj(), weights, smooth+comp)
        body_cached += np.einsum('qmg,qg,qng->qmn', (smooth+comp_cached).conj(), weights, smooth+comp_cached)
        assert np.count_nonzero(delta[Q:]) == np.count_nonzero(comp[Q:]) == 0
    result = dict(scope='P4 q-owner provider with actual adaptive96/quintic/L6 prepared table, complex nonidentity shared charge factor, delta/PS/served-M0 RHSs, finite-q phases, q/G padding, physical delta Fourier image and compensated Ry body plus full signed onsite.',
        artifact_sha256=sha, same_solved_coefficients=same_coefficients,
        delta_Fourier_max_error=max_delta, compensation_Fourier_max_error=max_comp,
        onsite_max_error=float(abs(onsite-onsite_cached).max()),
        full_complex_V_max_error=float(abs((body+onsite)-(body_cached+onsite_cached)).max()),
        full_complex_V_bitwise=np.array_equal(body+onsite, body_cached+onsite_cached))
    out = os.environ.get('LORRAX_TEST_RESULT')
    if jax.process_index() == 0:
        if out: Path(out).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    assert same_coefficients and max_delta == max_comp == result['onsite_max_error'] == 0.
    assert result['full_complex_V_bitwise']
    return 0


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    run_main_and_finalize(lambda: check_prepared_provider(runtime))
