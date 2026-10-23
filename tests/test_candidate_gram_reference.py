"""Candidate Gram versus literal plane-wave transition features.

Compute-node script, P4/P16/P64. The bundled Na WFN supplies nontrivial
full-zone k rows; the independent reference evaluates the Fourier sum and
forms transition features directly. Equal, nested and disjoint band
windows agree for whole-k and forced k-batch schedules, including band and
centroid padding. No candidate-size Gram is gathered in production.
"""


def main():
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wfn', default=str(Path(__file__).parent / 'hsuite/fixture_na/WFN.h5'))
    args = p.parse_args()
    import runtime
    stack = runtime.initialize_communicator_stack()
    import json
    import jax
    import jax.numpy as jnp
    import numpy as np
    from common.collectives import gather_to_host
    from centroid.pivoted_cholesky import build_gram_q0_via_loadwfns
    from runtime import tiles
    from wfn_loader import WfnLoader
    errors = {}
    original_tile = tiles.TILE_BYTES
    mesh = stack.mesh
    with WfnLoader(args.wfn, mesh=mesh) as wfn:
        sym = wfn.symmetry()
        grid = np.asarray(wfn.fft_grid, dtype=np.int64)
        rng = np.random.default_rng(836)
        n = 17
        indices = np.asarray(np.unravel_index(rng.choice(np.prod(grid), n, replace=False), grid)).T
        rows = [0, 1, min(10, int(sym.nk_tot)-1)]
        weights = np.zeros(sym.nk_tot)
        weights[rows] = [.2, .3, .5]
        samples = {}
        for k in rows:
            coeff = np.asarray(gather_to_host(wfn.load(bands=(0, 8), k=[k])))[0, :8]
            ng = int(wfn.ngk_valid(k=[k])[0])
            g = wfn.gvecs(k=[k])[0, :ng]
            phase = np.exp(2j*np.pi*((g+wfn.kvecs(k=[k])[0]) @ (indices/grid).T)) / np.sqrt(np.prod(grid))
            samples[k] = np.einsum('bsg,gm->bsm', coeff[:, :, :ng], phase)
        for name, left, right in (('equal', (0, 5), (0, 5)),
                                  ('nested', (1, 4), (0, 7)),
                                  ('disjoint', (0, 3), (4, 8))):
            ref = np.zeros((n, n), dtype=np.complex128)
            for k, psi in samples.items():
                # The metric's documented feature is R-dagger/L.
                z = np.einsum('csm,vsm->mvc', np.conj(psi[right[0]:right[1]]),
                              psi[left[0]:left[1]]).reshape(n, -1)
                ref += weights[k] * (z @ z.conj().T)
            for label, cap in (('whole', original_tile), ('batched', 4096)):
                tiles.TILE_BYTES = cap
                value = build_gram_q0_via_loadwfns(
                    wfn, sym, jnp.asarray(indices), mesh_xy=mesh,
                    band_range_left=left, band_range_right=right,
                    k_weights=weights, verbose=False)
                actual = np.asarray(gather_to_host(value))
                rel = float(np.max(np.abs(actual[:n, :n]-ref)) / np.max(np.abs(ref)))
                errors[f'{name}/{label}'] = rel
                assert rel < 2e-11, (name, label, rel)
                assert np.all(actual[n:, :] == 0) and np.all(actual[:, n:] == 0)
    tiles.TILE_BYTES = original_tile
    if jax.process_index() == 0:
        print('CANDIDATE_GRAM_REFERENCE ' + json.dumps(dict(
            errors=errors, max_relative=max(errors.values()), mesh=list(mesh.devices.shape))), flush=True)
        print('CANDIDATE_GRAM_REFERENCE PASS', flush=True)
    runtime.finalize_process(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
