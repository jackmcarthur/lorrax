"""P4 oracle: raw lifted parent faces versus the full-k G-sphere loader."""
from pathlib import Path
import argparse
import json
import sys

from runtime import initialize_communicator_stack


def main():
    """Compare both families in the Seitz gauge, removing the loader’s typed reciprocal phase."""
    args = argparse.ArgumentParser()
    args.add_argument('--wfn', required=True)
    args.add_argument('--charge', required=True)
    args.add_argument('--current', required=True)
    args.add_argument('--antiunitary', action='store_true')
    args = args.parse_args()
    runtime = initialize_communicator_stack(platform='gpu')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    from common.centroid_basis import PackedCentroidBasis
    from common.meta import Meta
    from common.wfn_transforms import load_centroids_band_chunked
    from file_io.centroids import load_centroids
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.wavefunction_bundle import parent_faces
    from wfn_loader import WfnLoader

    mesh = runtime.mesh
    assert mesh.size == 4 and jax.process_count() == 4
    wfn = WfnLoader(args.wfn, mesh=mesh)
    sym = wfn.symmetry()
    if args.antiunitary:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from test_parent_projector_unfold_oracle import antiunitary_row_assignment
        sym.sym_idx_k, n_anti = antiunitary_row_assignment(sym, wfn)
        assert n_anti == 64
    results = {}
    for family, path in [('charge', args.charge), ('current', args.current)]:
        _, points, count = load_centroids(path, wfn.fft_grid)
        basis = PackedCentroidBasis.build(points, sym, wfn.fft_grid, mesh)
        meta = Meta.from_system(wfn, sym, 8, 0, 8, count, True, mesh_xy=mesh, mu_basis=basis)
        plan = build_centroid_k_unfold_plan(sym, points, wfn.fft_grid, mesh,
                    nspinor=4, parent_k_frac=wfn.kvecs(k='ibz'), layout=basis.layout)
        parent = load_centroids_band_chunked(wfn, sym, meta, points, True, mesh,
                    band_range=(0, 8), band_chunk_size=4, k_domain='ibz', bispinor_lift='raw')
        full = load_centroids_band_chunked(wfn, sym, meta, points, True, mesh,
                    band_range=(0, 8), band_chunk_size=4, k_domain='full_bz', bispinor_lift='raw')
        parent, full = parent_faces(*parent, mesh_xy=mesh), parent_faces(*full, mesh_xy=mesh)
        errors = []
        for par, ref, spin, mu, axis, spec in zip(parent, full, (2, 1), (3, 2), ('y', 'x'),
                    (P(None, 'x', None, 'y'), P(None, None, 'x', 'y'))):
            fn = jax.jit(shard_map(lambda x: plan.unfold_face(x, spin_axis=spin,
                    mu_axis=mu, mesh_axis=axis), mesh=mesh, in_specs=spec, out_specs=spec,
                    check_vma=False))
            child = fn(par)
            error = float(jnp.max(jnp.abs(child-ref)) / jnp.max(jnp.abs(ref)))
            phases = [sym.reciprocal_phase(int(row), plan.k_parent_frac[int(parent)][None])
                      for parent, row in zip(plan.irr_idx, plan.sym_idx)]
            phase = jnp.asarray([1.0 if value is None else value[0] for value in phases])
            reference = ref * phase[:, None, None, None]
            gauge_error = float(jnp.max(jnp.abs(child-reference)) / jnp.max(jnp.abs(ref)))
            assert gauge_error < 1e-10, (family, gauge_error)
            errors.append(dict(raw_loader_gauge=error, typed_gauge=gauge_error))
        results[family] = dict(n_mu=count, packed=basis.n_packed, n_parent=plan.n_parent,
                               n_full=plan.n_full, relative_errors=errors)
    if jax.process_index() == 0:
        Path('face_parity.json').write_text(json.dumps(results, indent=2)+'\n')
        print('BISPINOR_PARENT_FACES_PASS', results, flush=True)


if __name__ == '__main__':
    main()
