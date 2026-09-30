"""Compile-only owner diagnosis of Coulomb packing and stage gates."""
from runtime import initialize_communicator_stack, run_main_and_finalize, rank0_print
R = initialize_communicator_stack()
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common import sanity
from common.centroid_basis import PackedCentroidBasis
from common.grouped_layout import build_square_grouped_shard_layout
from common.collectives import device_put_process_local, gather_to_host

def main():
    sh = NamedSharding(R.mesh, P(None, 'x', 'y'))
    rng = np.random.default_rng(437)
    # Interleaved packed pads, changed extent, both directions, partial tile.
    for n,group_width,canonical in ((53,8,56),(58,3,64)):
        layout = build_square_grouped_shard_layout(np.arange(n)//group_width, (2,2))
        b = PackedCentroidBasis(R.mesh, layout, np.zeros((n,3),np.int32),
                                canonical)
        source = rng.normal(size=(137,b.n_canonical,b.n_canonical)) + 1j*rng.normal(size=(137,b.n_canonical,b.n_canonical))
        source[:,n:,:] = 0; source[:,:,n:] = 0
        expected = b.pack_host(b.pack_host(source,axis=-2),axis=-1)
        value = device_put_process_local(source,sh)
        got = b._operator_kernel(sh.spec,False,1024)(value)
        packed = np.asarray(gather_to_host(got))
        assert np.array_equal(packed,expected)
        back = np.asarray(gather_to_host(b._operator_kernel(sh.spec,True,1024)(got)))
        assert np.array_equal(back,source)
        direct = np.asarray(gather_to_host(b._operator_kernel(sh.spec,False,1<<30)(value)))
        assert np.array_equal(direct,packed)
        rank0_print(json.dumps(dict(kind='parity',logical=n,canonical=b.n_canonical,
            packed=b.n_packed,max_error=float(np.max(np.abs(packed-expected))),
            roundtrip_error=float(np.max(np.abs(back-source))))),flush=True)
    # The gate remains active and counts NaN versus Inf independently.
    bad = source.copy(); bad[0,0,0]=np.nan; bad[-1,0,1]=np.inf
    stats = sanity._finite_stats(device_put_process_local(bad,sh))
    assert stats[:2] == (2,1), stats
    try:
        sanity.refuse_nonfinite('deliberate negative control',device_put_process_local(bad,sh),print_fn=lambda *args:None)
    except sanity.NonFiniteResultError:
        pass
    else:
        raise AssertionError('nonfinite gate failed to refuse')
    rank0_print(json.dumps(dict(kind='nonfinite_control',stats=stats)),flush=True)
    x = jax.ShapeDtypeStruct((8000, 904, 904), jnp.complex128, sharding=sh)
    fn = sanity._finite_stats_fn(x.shape, x.dtype, sh)
    for name, f in [('finite', fn)]:
        m = f.lower(x).compile().memory_analysis()
        rank0_print(json.dumps(dict(name=name, argument=m.argument_size_in_bytes,
            output=m.output_size_in_bytes, temp=m.temp_size_in_bytes)), flush=True)
    groups = np.arange(898, dtype=np.int32) // 4
    layout = build_square_grouped_shard_layout(groups, (2, 2))
    b = PackedCentroidBasis(R.mesh, layout, np.zeros((898,3), np.int32), 900)
    rank0_print(b.describe(), flush=True)
    y = jax.ShapeDtypeStruct((8000,900,900), jnp.complex128, sharding=sh)
    exe = b._operator_kernel(sh.spec,False).lower(y).compile()
    m = exe.memory_analysis()
    rank0_print(json.dumps(dict(name='pack', argument=m.argument_size_in_bytes,
        output=m.output_size_in_bytes,temp=m.temp_size_in_bytes)),flush=True)
    assert m.temp_size_in_bytes < 2e9, m
    old = jax.jit(lambda op:b.pack_axis(b.pack_axis(op,-2,spec=sh.spec),-1,spec=sh.spec))
    old_m = old.lower(y).compile().memory_analysis()
    assert old_m.temp_size_in_bytes > 20e9, old_m
    rank0_print(json.dumps(dict(name='unbounded_negative',temp=old_m.temp_size_in_bytes)),flush=True)
    z = jax.ShapeDtypeStruct((8000,b.n_packed,b.n_packed),jnp.complex128,sharding=sh)
    un = b._operator_kernel(sh.spec,True).lower(z).compile().memory_analysis()
    assert un.temp_size_in_bytes < 2e9, un
    rank0_print(json.dumps(dict(name='unpack',argument=un.argument_size_in_bytes,
        output=un.output_size_in_bytes,temp=un.temp_size_in_bytes)),flush=True)
    return 0
run_main_and_finalize(main)
