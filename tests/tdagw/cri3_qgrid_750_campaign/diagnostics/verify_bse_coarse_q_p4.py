"""Prepared native coarse-q BSE matvec proof, NOT a material eigensolve."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import traceback


def run_oracle(mesh):
    """Actual BSE owner versus literal coarse1/Nq W + fine1/Nk exchange."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.fft_helpers import make_bosonic_q_comb, get_donated_kfft_kminor
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_head import _inject_q0_head
    from bse.bse_preconditioner import compute_pair_amplitude
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from gw.ppm_tau_kernel import sigma_lattice_tile

    rng = np.random.default_rng(177)
    grid, coarse, nk, nq, nc, nv, ns, mu = (8, 8, 1), (4, 4, 1), 64, 16, 2, 2, 2, 4
    def random(shape):
        return .2*(rng.normal(size=shape)+1j*rng.normal(size=shape))
    pc, pv = random((nk, nc, ns, mu)), random((nk, nv, ns, mu))
    X = random((2, nc, nv, nk))
    bare, screened = random((mu, mu)), random((nq, mu, mu))
    bare = bare+bare.conj().T
    ec, ev = np.full((nk, nc), .7), np.full((nk, nv), -.6)
    g0 = np.array([.1+.2j, .2-.1j, -.3+.1j, .1-.2j])
    vhead, whead, volume = .13, .37, 9.
    sh = make_bse_shardings(mesh)
    put = lambda a, spec: device_put_process_local(np.ascontiguousarray(a), NamedSharding(mesh, spec))
    host = lambda a: np.asarray(gather_to_host(a))
    v, w, _ = _inject_q0_head(put(bare, P('x', 'y')),
        put(np.moveaxis(screened, 0, -1).reshape(mu, mu, *coarse), sh.W.spec),
        put(g0, P('x')), put(g0, P('y')), vhead, jnp.asarray([whead]), volume, w0_ready=True)
    flat = jax.jit(lambda a: jnp.moveaxis(a.reshape(mu, mu, nq), -1, 0),
        out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))(w)
    comb = make_bosonic_q_comb(mesh, grid, coarse)
    fine = comb(flat)
    wc = screened.copy(); wc[0] += (whead/volume)*np.outer(g0.conj(), g0)
    vb = bare+(vhead/volume)*np.outer(g0.conj(), g0)
    fine_expected = np.zeros((nk, mu, mu), complex)
    qs = np.indices(coarse).reshape(3, -1).T*2
    slots = np.ravel_multi_index(qs.T, grid)
    fine_expected[slots] = 4*wc
    np.testing.assert_allclose(host(fine), fine_expected, rtol=2.e-14, atol=2.e-15)
    comb_error = float(np.max(abs(host(fine)-fine_expected)))
    # Independent Fourier dual: the same q-comb is periodic R tiling,
    # NOT an interpolating zero-embedding of the real-space tensor.
    coarse_r = np.fft.ifftn(wc.reshape(*coarse, mu, mu), axes=(0, 1, 2), norm='ortho').reshape(nq, mu, mu)
    tiled = host(sigma_lattice_tile(put(coarse_r, P(None, 'x', 'y')), grid, coarse))
    fine_r = np.fft.ifftn(fine_expected.reshape(*grid, mu, mu), axes=(0, 1, 2), norm='ortho').reshape(nk, mu, mu)
    np.testing.assert_allclose(tiled, fine_r, rtol=3.e-13, atol=3.e-14)
    W = jax.jit(lambda a: jnp.moveaxis(a, 0, -1).reshape(mu, mu, *grid), out_shardings=sh.W)(fine)
    WR = get_donated_kfft_kminor(mesh, grid, sh.W.spec, kind='ifftn', norm='ortho')(W)
    pcx, pcy, pvx, pvy = (put(a, spec) for a, spec in
        ((pc, sh.psi_x.spec), (pc, sh.psi_y.spec), (pv, sh.psi_x.spec), (pv, sh.psi_y.spec)))
    M = jax.jit(compute_pair_amplitude, out_shardings=sh.M)(pcx, pvx)
    actual = host(build_bse_stack_matvec(mesh, *grid, kernel='bse')(
        put(X, sh.X.spec), pcx, pcy, pvx, pvy, put(ec, sh.eps.spec), put(ev, sh.eps.spec), WR, v, M))
    m = np.einsum('kcsm,kvsm->kcvm', pc.conj(), pv)
    expected = (ec.T[None, :, None]-ev.T[None, None])*X
    expected += np.einsum('kcvm,mn,ldwn,bdwl->bcvk', m, vb, m.conj(), X)/nk
    coords = np.indices(grid).reshape(3, -1).T
    for iq, q in enumerate(qs):
        partner = np.ravel_multi_index(((coords-q) % grid).T, grid)
        cc = np.einsum('kcsM,kdsM->kcdM', pc.conj(), pc[partner])
        vv = np.einsum('kvsN,kwsN->kvwN', pv, pv[partner].conj())
        expected -= np.einsum('kcdM,MN,kvwN,bdwk->bcvk', cc, wc[iq], vv, X[..., partner])/nq
    np.testing.assert_allclose(actual, expected, rtol=3.e-10, atol=3.e-12)
    return dict(max_complex_matvec_error=float(np.max(abs(actual-expected))),
        max_q_comb_error=comb_error,
        electronic_grid=list(grid), response_grid=list(coarse), weight=4.,
        heads=dict(fine_bare=vhead, coarse_screened=whead, bare_applications=1, screened_applications=1),
        scope='native_complex_matvec_ONLY; no material_BSE_roots_or_SC/TD_claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=120.)
    args = parser.parse_args()
    if os.environ.get('LEG_HEAD') != args.source_revision or not 0 < args.max_wall_seconds <= 120:
        raise ValueError('exact launcher source revision and <=120s wall cap required')
    timer = threading.Timer(args.max_wall_seconds, lambda: os._exit(124)); timer.daemon = True; timer.start()
    import gw.gw_jax as driver  # Collective native bootstrap BEFORE arrays.
    import runtime
    import jax
    from common.collectives import rank0_transaction
    from ffi.fft import kconv_backend
    import common.fft_helpers, bse.bse_stack_matvec
    root = Path(args.source_root).resolve()
    origins = {m.__name__: str(Path(m.__file__).resolve()) for m in
               (driver, common.fft_helpers, bse.bse_stack_matvec)}
    if (any(root not in Path(p).parents for p in origins.values())
            or jax.process_count() != 4 or jax.local_device_count() != 1
            or driver.RUNTIME.mesh.devices.shape != (2, 2) or not jax.config.x64_enabled
            or any(d.platform != 'gpu' for d in jax.devices())
            or kconv_backend(driver.RUNTIME.mesh, (8, 8, 1)) != 'mathdx'):
        raise ValueError('native source-bound CUDA P4 FP64/mathdx required')
    out = Path(args.out).resolve()
    rank0_transaction(out, stage='coarse_bse/new_output', write=lambda: out.mkdir(parents=True, exist_ok=False))
    started = time.monotonic()
    try:
        result = run_oracle(driver.RUNTIME.mesh)
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise  # No new collective on a potentially rank-local failure.
    report = dict(status='PASS_SOFTWARE_ONLY', result=result, revision=args.source_revision,
        module_origins=origins, elapsed_seconds=time.monotonic()-started,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'), material_admitted=False)
    def save():
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    rank0_transaction(out, stage='coarse_bse/receipt', write=save)
    timer.cancel(); runtime.finalize_process(0)


if __name__ == '__main__':
    main()
