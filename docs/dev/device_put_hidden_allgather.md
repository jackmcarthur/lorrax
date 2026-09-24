# `device_put` onto a multi-process sharding is an all-gather

On a multi-process mesh, `jax.device_put(host_array, NamedSharding(mesh, spec))`
costs `O(P × array)` in wall time and resident memory on every rank before any
work starts: JAX all-gathers the whole array to assert that every process
passed the same value. Place process-replicated host data with
`common.collectives.device_put_process_local` instead.

## Mechanism (jax 0.9.1)

`jax._src.dispatch._device_put_sharding_impl` enters its multi-process branch
when the target sharding is not fully addressable, the operand is numpy, a
Python scalar or an **uncommitted** `jax.Array`, and every process owns a
device in the sharding. It then calls `multihost_utils.assert_equal`, which
`process_allgather`s the operand (`tiled=True`) and builds the expected value
host-side with `np.concat([x] * process_count())`: two `P × x.nbytes` buffers
per rank, silently, charged to whichever stage made the call.

| the branch is skipped when | because |
|---|---|
| `process_count() == 1` | nothing to assert |
| the target is a `Device`, or there is no target | not a `Sharding` |
| the target sharding is fully addressable | the multi-process branch is not entered |
| the operand is a **committed** `jax.Array` | this is a reshard, and reshards do not assert |
| some process owns no device in the sharding | JAX cannot tell a subset mesh from divergent input |

`jnp.zeros`, `jnp.ones` and `jnp.asarray(numpy)` produce uncommitted arrays and
take the branch. The output of a `jax.jit` with explicit `out_shardings`, or of
any operation on committed operands, is committed and does not.

Scale: staging a 0.336 GiB ψ at P = 16 this way raised per-rank peak from
1.2 GiB to 13.0 GiB and turned a 1 s kernel into a timeout; at P = 64 it
raised inside `assert_equal`. The cost is reported as whatever stage is
running, so in a benchmark it reads as a regression in unrelated code. Stage
every benchmark input with a collective-free idiom, or the measurement is of
the staging.

## The idiom

| the process holds | use |
|---|---|
| the whole global array, identical on every rank, inside `src/` | `common.collectives.device_put_process_local(host, sharding)` |
| the whole global array, in a harness | `jax.make_array_from_callback(shape, sharding, lambda idx: host[idx])` |
| only its own slice | `jax.make_array_from_process_local_data(sharding, local, global_shape)` |

`device_put_process_local` (owned by `lxkit.placement`, re-exported from
`common.collectives`) slices only the shards of this process's addressable
devices and assembles them with `make_array_from_single_device_arrays`. It
falls back to plain `device_put` at P = 1, on a fully addressable target, and
when the process owns no device in the sharding; a tracer or an already
globally sharded array is resharded, never pulled to host.
`LORRAX_CHECK_REPLICA=1` (or `check=True`) re-arms JAX's assertion for a
debugging run. Precondition, the one `device_put` spends `2P × |x|` to check:
**the host array must be bit-identical on every process.**

## Where `device_put` is correct

* single-process runs;
* a `Device` target or no target;
* resharding a committed, already-sharded `jax.Array` (its intended use);
* small replicated scalars and index tables, where `2P × nbytes` is bytes.

The threshold is size × P, and nothing at the call site says which side of it
a call is on. A `device_put(x, sharding)` of a μ²-class array is safe only
while `x` is committed; before editing such a line, confirm the producer is a
jit with `out_shardings` or another committed operation, or switch it to
`device_put_process_local` so the safety is local.
