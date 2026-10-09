"""P4 local angular buckets versus the independent canonical tensor map."""
from pathlib import Path
import json
import os


def check_angular_compression(runtime):
    from types import SimpleNamespace
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from gw.isdf_augmentation import (_angular_bucket_tables,
        _angular_compression_workspace_bytes, _compress_rhs_kernel)

    mesh = runtime.mesh
    assert tuple(mesh.shape[k] for k in ('x', 'y')) == (2, 2)
    rng = np.random.default_rng(61782)
    results = []
    for na, nh, rp, nang, padded, all_ghost_shard in ((2, 2, 3, 3, 24, False),
                                                    (1, 3, 3, 3, 48, True)):
        logical = na*rp*nang
        packed = np.full(padded, -1, np.int32)
        packed[:logical] = rng.permutation(logical)
        if not all_ghost_shard:
            packed = packed[rng.permutation(padded)]
        active = packed >= 0
        plan = SimpleNamespace(layout=SimpleNamespace(axis=SimpleNamespace(
            packed_to_canonical=packed, active_mask=active, n_logical=logical)))
        weights = rng.normal(size=(nh, nang))+1j*rng.normal(size=(nh, nang))
        z = rng.normal(size=(3, 4, padded))+1j*rng.normal(size=(3, 4, padded))
        z[..., ~active] = 1e80*(1+2j)
        z[2] = 0.
        canonical = np.zeros((3, 4, logical), complex)
        canonical[..., packed[active]] = z[..., active]
        reference = np.einsum('qmarn,hn->qmahr',
            canonical.reshape(3, 4, na, rp, nang), weights).reshape(3, 4, -1)
        kernel, nf, nfp = _compress_rhs_kernel(mesh, plan, na, nh, rp, weights)
        source = device_put_process_local(z, NamedSharding(mesh, P(None, 'x', 'y')))
        got = np.asarray(gather_to_host(kernel(source)))
        error = float(np.linalg.norm(got[..., :nf]-reference)/np.linalg.norm(reference))
        assert error < 4e-15
        assert np.max(abs(got[2])) == 0.
        if nfp > nf:
            assert np.max(abs(got[..., nf:])) == 0.
        indices, angular = _angular_bucket_tables(plan, na, rp, weights, 2)
        if all_ghost_shard:
            assert not np.any(active[padded//2:])
            assert np.max(abs(angular[1])) == 0.
        workspace = _angular_compression_workspace_bytes(3, 4, na, nh, rp, nang, 2)
        live_gather = 16*3*2*na*rp*indices.shape[-1]
        assert workspace['bucket_gather'] >= live_gather
        assert workspace['total'] == sum(workspace[k] for k in
            ('bucket_gather', 'pre_scatter_result', 'index_and_angular_tables'))
        results.append(dict(relative_error=error, wholly_ghost_point_shard=all_ghost_shard,
                            feature_pad=nfp-nf, bucket_width=indices.shape[-1]))
        bad = packed.copy()
        bad[np.flatnonzero(active)[0]] = logical
        bad_plan = SimpleNamespace(layout=SimpleNamespace(axis=SimpleNamespace(
            packed_to_canonical=bad, active_mask=active, n_logical=logical)))
        try:
            _angular_bucket_tables(bad_plan, na, rp, weights, 2)
        except ValueError:
            pass
        else:
            raise AssertionError('noncanonical logical point domain was accepted')
        bad_weights = weights.copy();bad_weights[0, 0] = np.nan
        try:
            _angular_bucket_tables(plan, na, rp, bad_weights, 2)
        except ValueError:
            pass
        else:
            raise AssertionError('nonfinite angular weights were accepted')
    result = dict(status='PASS', cases=results,
                  scope='Independent canonical tensor map, shuffled point packing, huge poisoned ghost columns, wholly ghost point-Y shard, q/feature pads and conservative workspace bound.')
    if int(runtime.process_index) == 0:
        print(json.dumps(result), flush=True)
        if os.environ.get('ANGULAR_COMPRESSION_REPORT'):
            Path(os.environ['ANGULAR_COMPRESSION_REPORT']).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    def main():
        check_angular_compression(runtime)
        return 0
    run_main_and_finalize(main)
