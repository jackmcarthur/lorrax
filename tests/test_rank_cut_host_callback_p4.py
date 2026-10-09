"""P4 rank-cut refusal when only one q owner emits a host policy finding.

Run as a module on four MPI ranks. Host callback reductions must not invoke
JAX compilation on the finding's owner while peers proceed to the host seam.
The conditioning and discarded-weight criteria are unchanged.
"""
from pathlib import Path
import argparse
import json
import os


def check_host_callback(runtime):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.shard_map import shard_map
    from common import rank_criterion
    from isdf.core import _certify_the_cut
    from tests.hsuite import rank_session

    if runtime.process_count != 4:
        raise ValueError('The owner-specific callback regression requires P4')
    if rank_criterion.resolve_policy_mode(os.environ.get(rank_criterion.POLICY_MODE_ENV)) != 'refuse':
        raise ValueError('The policy regression requires the normal refusal policy')
    rank_criterion.raise_if_pending('start of callback regression', mode='refuse')
    spec = P(('x', 'y'), None)
    layout = NamedSharding(runtime.mesh, spec)
    keep = np.zeros((4, 8), bool)
    keep[0, :2] = True
    exclude = np.ones_like(keep)
    exclude[0, :3] = False
    put = lambda a: device_put_process_local(a, layout)

    def certify(spectrum, selected, padding):
        _certify_the_cut(spectrum, selected,
            where='single-q-owner callback regression', kappa_certified=1.e8,
            rcond=1.e-9, exclude=padding)
        return spectrum
    kernel = jax.jit(shard_map(certify, mesh=runtime.mesh,
        in_specs=(spec, spec, spec), out_specs=spec, check_vma=False))
    records = {}
    for label, second, should_refuse in [('accepted', 1.e-7, False),
                                         ('uncertified', 1.e-9, True)]:
        spectrum = np.zeros((4, 8), np.float64)
        spectrum[0, :3] = (1., second, 1.e-12)
        result = kernel(put(spectrum), put(keep), put(exclude))
        result.block_until_ready()
        jax.effects_barrier()
        findings = rank_criterion.pending()
        refused = False
        message = ''
        try:
            rank_criterion.raise_if_pending('callback regression host seam', mode='refuse')
        except rank_criterion.RankPolicyError as error:
            refused, message = True, str(error)
        rows = rank_session.exchange(dict(rank=runtime.process_index,
            findings=len(findings), refused=refused, message=message))
        if should_refuse:
            assert sum(row['findings'] for row in rows) == 1, rows
            assert sum(row['refused'] for row in rows) == 1, rows
            assert any('1.000e+09' in row['message'] for row in rows), rows
        else:
            assert not any(row['findings'] or row['refused'] for row in rows), rows
        assert not rank_criterion.pending()
        records[label] = rows
    return dict(P=4, q_owned=True, physical_q=1, carrier_q=4,
        single_owner_refusal=True, accepted_cut_unchanged=True,
        conditioning_ceiling=1.e8, unadmitted_test_conditioning=1.e9,
        host_callback_avoids_JAX_reduction=True, records=records,
        scope='Policy-flow regression only; no changed numerical cut, factor or physics admission.')


def check_spectral_callback(runtime):
    """Refuse a split multiplet on one q owner without changing its mask."""
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.shard_map import shard_map
    from common import spectral_closure
    from isdf.core import _close_the_cut
    from tests.hsuite import rank_session

    spec = P(('x', 'y'), None)
    layout = NamedSharding(runtime.mesh, spec)
    spectrum = np.zeros((4, 8), np.float64)
    spectrum[0, :4] = (1., 1.e-4, (1.-1.e-11)*1.e-4, 1.e-8)
    keep = np.zeros_like(spectrum, dtype=bool)
    keep[0, :2] = True
    prior_mode = os.environ.get(spectral_closure.MODE_ENV)
    spectral_closure.raise_if_pending('start of strict callback regression', mode='strict')
    try:
        os.environ[spectral_closure.MODE_ENV] = 'strict'
        kernel = jax.jit(shard_map(
            lambda a, k: _close_the_cut(a, k, where='single-q-owner strict callback regression'),
            mesh=runtime.mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False))
        result = kernel(device_put_process_local(spectrum, layout),
                        device_put_process_local(keep, layout))
        result.block_until_ready()
        jax.effects_barrier()
        # Strict mode records and refuses; only snap mode changes the mask.
        for shard in result.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), keep[shard.index])
        findings = spectral_closure.pending()
        refused = False
        message = ''
        try:
            spectral_closure.raise_if_pending('strict callback regression host seam', mode='strict')
        except spectral_closure.SpectralClusterError as error:
            refused, message = True, str(error)
        rows = rank_session.exchange(dict(rank=runtime.process_index,
            findings=len(findings), refused=refused, message=message))
        assert sum(row['findings'] for row in rows) == 1, rows
        assert sum(row['refused'] for row in rows) == 1, rows
        assert any('retained rank 2 would have become 1' in row['message'] for row in rows), rows
        assert not spectral_closure.pending()
        return dict(single_owner_refusal=True, strict_mask_unchanged=True, records=rows)
    finally:
        if prior_mode is None:
            os.environ.pop(spectral_closure.MODE_ENV, None)
        else:
            os.environ[spectral_closure.MODE_ENV] = prior_mode


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    runtime = initialize_communicator_stack()
    def main():
        import jax
        result = check_host_callback(runtime)
        result['strict_spectral_closure'] = check_spectral_callback(runtime)
        if jax.process_index() == 0:
            path = Path(args.report)
            if path.exists():raise FileExistsError(path)
            path.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
        return 0
    run_main_and_finalize(main)
