"""Tiny exact-value transport checks on the mini suite's real P4 mesh."""
from pathlib import Path


def check_slab_io(mesh, path: Path):
    import h5py
    import jax
    import numpy as np
    import pytest
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.slab_io import SlabIO

    # These independent reference arrays are at most a few KiB. Device
    # operands are constructed directly as this rank's addressable tiles.
    logical = (5, 7)
    padded = (8, 8)
    reference = np.arange(35).reshape(logical).astype(np.complex128)
    reference += 1j * (100 + reference.real)
    carrier = np.full(padded, -999 + 77j, dtype=np.complex128)
    carrier[:5, :7] = reference
    specs = {"xy": P("x", "y"), "rows": P(("x", "y"), None)}

    def placed(host, spec):
        return jax.make_array_from_callback(
            host.shape, NamedSharding(mesh, spec), lambda index: host[index])

    def exact_local(got, expected, spec):
        assert got.shape == expected.shape
        assert got.sharding.is_equivalent_to(NamedSharding(mesh, spec), got.ndim), (
            got.sharding, spec)
        assert len(got.addressable_shards) == 1, "mini needs one GPU per process"
        for shard in got.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])

    expected = np.zeros(padded, dtype=np.complex128)
    expected[:5, :7] = reference
    with SlabIO(path, mode="w", mesh=mesh) as io:
        backend = type(io._backend).__name__
        assert backend == "_FfiBackend", "real P4 must use parallel HDF5"
        for name, spec in specs.items():
            io.create_dataset(name, shape=logical, dtype=np.complex128,
                              attrs={"layout": name})
            io.write_slab(name, placed(carrier, spec))
        io.write_attr("steps", np.array([1, 3, 5], dtype=np.int64))
        io.sync_writes()
        exact_local(io.read_slab("xy", shape=padded, partition_spec=specs["xy"]),
                    expected, specs["xy"])
        # These are replicated Python preconditions: every rank must refuse
        # before entering any mismatched collective. A silent accept fails.
        with pytest.raises(ValueError, match="valid slab exceeds dataset extent"):
            io.write_slab("xy", placed(carrier, specs["xy"]),
                          offset=(4, 0), valid_shape=logical)
        with pytest.raises(ValueError, match="extent|divis"):
            io.read_slab("xy", shape=logical, partition_spec=specs["xy"])

    # Append updates one logical corner, clipping the rest of a padded tile.
    patch = np.full((4, 4), 123 + 456j, dtype=np.complex128)
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.create_dataset("xy", shape=logical, dtype=np.complex128)
        io.write_slab("xy", placed(patch, specs["xy"]), offset=(4, 4))
    modified = reference.copy()
    modified[4:, 4:] = patch[:1, :3]

    with SlabIO(path, mode="r", mesh=mesh) as io:
        shapes = {}
        for name, spec in specs.items():
            got = io.read_slab(name, partition_spec=spec)
            want = np.zeros(got.shape, dtype=np.complex128)
            want[:5, :7] = modified if name == "xy" else reference
            exact_local(got, want, spec)
            shapes[name] = list(got.shape)
        # Three disjoint row windows, with a ragged final window. The last
        # process in the product-sharded rows case above has only padding.
        offsets = [(0, 0), (2, 0), (4, 0)]
        valid = [(2, 7), (2, 7), (1, 7)]
        windows = io.read_slabs("xy", shape=(4, 8), offsets=offsets,
                                valid_shapes=valid, partition_spec=specs["xy"],
                                window_axis=0)
        want_windows = np.zeros((3, 4, 8), dtype=np.complex128)
        for i, ((start, _), (count, _)) in enumerate(zip(offsets, valid)):
            want_windows[i, :count, :7] = modified[start:start + count]
        exact_local(windows, want_windows, P(None, "x", "y"))
        np.testing.assert_array_equal(io.read_small("steps"), [1, 3, 5])

    # An independent, bounded host read after collective close catches two
    # transport mistakes cancelling in a write/read round trip.
    with h5py.File(path, "r") as h5:
        for name, want in (("xy", modified), ("rows", reference)):
            assert h5[name].shape == logical, "padding leaked onto disk"
            np.testing.assert_array_equal(h5[name][...], want)
            assert h5[name].attrs["layout"] == name
        np.testing.assert_array_equal(h5["steps"][...], [1, 3, 5])
    return {"backend": backend, "logical_shape": list(logical),
            "read_shapes": shapes, "window_count": 3, "refusals": 2,
            "checks": ["xy_tiles", "product_axis_and_empty_rank", "zero_padding",
                       "same_handle_read", "append_offset", "packed_windows",
                       "metadata", "independent_disk_values"]}
