"""P4 native/JAX stream byte parity, including ragged output tiles."""
from pathlib import Path
import json
import os
import sys
import numpy as np


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.contour_accumulator import contour_accumulator
    root = Path(sys.argv[1])
    assert jax.device_count() == jax.process_count() == 4
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ('x', 'y'))
    native = contour_accumulator(mesh)
    contour_accumulator(mesh)  # registration is idempotent
    rng = np.random.default_rng(3992203)
    def put(a, spec):
        return jax.make_array_from_callback(a.shape, NamedSharding(mesh, spec), lambda i: a[i])
    def complex_random(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)
    rows = []
    for outputs in (1, 5, 12):
        initial = put(complex_random((outputs, 3, 12, 16)), P(None, None, 'x', 'y'))
        cs = put(complex_random((17, 3, 12, 16)), P(None, None, 'x', 'y'))
        ps = put(complex_random((17, outputs)), P())
        def stream(op):
            def run(a, cs, ps):
                return jax.lax.scan(lambda a, cp: (op(a, *cp), None), a, (cs, ps))[0]
            return jax.jit(run)
        pure = stream(lambda a, c, p: a+p[:, None, None, None]*c[None])
        ffi = stream(native)
        want = pure(initial, cs, ps)
        got = ffi(initial, cs, ps)
        jax.block_until_ready((want, got))
        pairs = zip(want.addressable_shards, got.addressable_shards)
        mismatches = sum(np.count_nonzero(np.asarray(a.data).view(np.uint64) != np.asarray(b.data).view(np.uint64)) for a, b in pairs)
        error = float(jnp.max(jnp.abs(got-want)))
        rows.append(dict(outputs=outputs, steps=17, byte_mismatches=int(mismatches), max_error=error))
        if jax.process_index() == 0:
            (root/f'jax_{outputs}.hlo').write_text(pure.lower(initial, cs, ps).compile().as_text())
        assert mismatches == 0, rows
    receipt = dict(status='PASS', job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], rank=jax.process_index(), cases=rows)
    (root/f'receipt_rank{jax.process_index()}.json').write_text(json.dumps(receipt, indent=2))
    if jax.process_index() == 0:
        (root/'receipt.json').write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    run_main_and_finalize(main)
