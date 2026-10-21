"""``SlabIO`` and its neighbours on the serial (emulated-mesh) tier, CPU only.

Each test holds one contract that the hsuite chain cannot see:

1. a padded block clips to the dataset, reads back through every reader, and a
   dataset that exists is reused or refused, never rewritten;
2. ``read_multi`` and ``read_slabs`` return what ``read_slab`` returns;
3. ``write_slab(require_finite=True)`` refuses a non-finite block and publishes
   nothing (the file tier and the resident-bank twin);
4. the shared-directory nonce passes on one directory and refuses per-rank ones;
5. the platform key follows the vendor and a missing leg is refused by name;
6. the per-rank streamed tier's capacity arithmetic and its host fallback;
7. ``open_h5_settled`` waits out only a transiently short size view.

The serial tier refuses a non-``cpu`` mesh platform by design, so the tests that
drive ``SlabIO`` skip on a GPU node.  What needs the collective tier (MPI, GPU)
is not here: see ``tests/phdf5_read_multi_gate.py``.
"""

import h5py
import numpy as np
import pytest

import runtime  # noqa: F401  (runtime before jax: the repo's bootstrap)
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def _cpu_mesh():
    """An emulated 2x2 HOST mesh.  The serial tier refuses any other platform."""
    cpu = jax.devices("cpu")
    if len(cpu) < 4:
        pytest.skip("needs 4 host CPU devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    mesh = Mesh(np.array(cpu[:4]).reshape(2, 2), ("x", "y"))
    platform = mesh.devices.flat[0].platform
    if platform != "cpu":
        pytest.skip(f"CPU only: SlabIO's serial tier refuses a {platform!r} mesh")
    return mesh


def _put(mesh, host, spec):
    return jax.device_put(host, NamedSharding(mesh, spec))


# ---- 1. round trip, clip, reuse-or-refuse --------------------------------------

def test_a_padded_block_clips_reads_back_and_a_dataset_that_exists_is_reused_or_refused(tmp_path):
    from file_io.slab_io import SlabIO
    mesh = _cpu_mesh()
    spec = P(None, "x", "y")
    path = str(tmp_path / "round_trip.h5")
    rng = np.random.default_rng(3)
    logical = rng.standard_normal((4, 5, 7)) + 1j * rng.standard_normal((4, 5, 7))
    # The dataset is (6, 5, 7); the block is padded to the mesh-divisible
    # (6, 6, 8) and filled with a sentinel, and starts at row 2: three dims
    # clip (4 rows fit, 5 of 6 columns, 7 of 8 planes).
    block = np.full((6, 6, 8), 77.0 + 77.0j)
    block[:4, :5, :7] = logical
    want = np.zeros((6, 5, 7), np.complex128)
    want[2:6] = logical

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(6, 5, 7), dtype=np.complex128,
                          attrs={"k_storage": "wedge"})
        io.write_slab("A", _put(mesh, block, spec), offset=(2, 0, 0))
        io.write_attr("omega_ev", np.float64(13.6))
        io.write_attr("grid", np.array([6, 6, 1], np.int32))
        io.stamp_dataset_attrs("omega_ev", {"units": "eV"})
    # Raw bytes, by h5py after the handle closed: the pad never reached the
    # file, rows 0-1 are untouched, and the deferred metadata landed.
    with h5py.File(path, "r") as f:
        assert f["A"].shape == (6, 5, 7)
        assert np.array_equal(f["A"][()], want)
        assert f["A"].attrs["k_storage"] == "wedge"
        assert f["omega_ev"][()] == 13.6 and f["omega_ev"].attrs["units"] == "eV"
        assert f["grid"][()].tolist() == [6, 6, 1]

    with SlabIO(path, mode="r", mesh=mesh) as io:
        # No shape: the dataset rounded UP to the mesh, zero past the dataset.
        full = io.read_slab("A", partition_spec=spec)
        assert full.shape == (6, 6, 8) and full.sharding.spec == spec
        pad = np.zeros((6, 6, 8), np.complex128)
        pad[:, :5, :7] = want
        assert np.array_equal(np.asarray(full), pad)
        # A window at an offset; the ragged-chunk override zeroes past valid_shape.
        win = io.read_slab("A", shape=(4, 6, 8), offset=(2, 0, 0), partition_spec=spec,
                           valid_shape=(4, 3, 7), as_numpy=True)
        assert isinstance(win, np.ndarray)
        assert np.array_equal(win[:, :3, :7], logical[:, :3, :7])
        assert not win[:, 3:].any() and not win[:, :, 7:].any()
        # An override that overruns the dataset refuses; it is never shrunk.
        with pytest.raises(ValueError, match="valid slab exceeds dataset extent"):
            io.read_slab("A", shape=(6, 6, 8), offset=(2, 0, 0), partition_spec=spec,
                         valid_shape=(6, 5, 7))
        # The scalar entry point: the dataset's own shape and value, dtype on request.
        scalar = io.read_small("omega_ev")
        assert scalar.shape == () and float(scalar) == 13.6
        grid = io.read_small("grid", dtype=np.int64)
        assert grid.dtype == np.int64 and grid.tolist() == [6, 6, 1]

    # Reuse or refuse.  The same shape and dtype reuse the dataset (and its
    # bytes); a different shape or dtype refuses, naming the clash.
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.create_dataset("A", shape=(6, 5, 7), dtype=np.complex128)
        with pytest.raises(ValueError, match="the dataset exists"):
            io.create_dataset("A", shape=(6, 5, 8), dtype=np.complex128)
        with pytest.raises(ValueError, match="the dataset exists"):
            io.create_dataset("A", shape=(6, 5, 7), dtype=np.float64)
        io.write_attr("omega_ev", np.float64(27.2))        # replaces the queued name
    # A padded write on a handle that never registered the existing dataset
    # would have to re-create it at the padded extent: it refuses instead.
    with pytest.raises(ValueError, match="the dataset exists"):
        with SlabIO(path, mode="a", mesh=mesh) as io:
            io.write_slab("A", _put(mesh, block, spec), offset=(2, 0, 0))
    with h5py.File(path, "r") as f:
        assert f["A"].shape == (6, 5, 7) and np.array_equal(f["A"][()], want)
        assert f["omega_ev"][()] == 27.2


# ---- 2. read_multi and read_slabs return what read_slab returns ----------------

NQ, FIELDS, ROWS = 5, 3, 6
BATCH = P(("x", "y"), None, None, None)
FACE = P(None, None, "x", "y")


def _ref(k, shape, dtype):
    n = int(np.prod(shape))
    x = (np.arange(n, dtype=np.float64) + 1000.0 * k).reshape(shape)
    if np.issubdtype(dtype, np.complexfloating):
        return (x + 1j * (0.5 - x)).astype(dtype)
    return x.astype(dtype)


_READ_SET = {                      # name -> (k, shape, dtype)
    "line_a": (0, (NQ, FIELDS, ROWS, 5), np.complex128),
    "line_b": (1, (NQ, FIELDS, ROWS, 7), np.complex128),
    "real_f64": (2, (NQ, FIELDS, ROWS, 4), np.float64),
    "index_i64": (3, (NQ, 2, 3, 4), np.int64),
    "empty": (4, (NQ, FIELDS, ROWS, 0), np.complex128),
}


def _expected(name, offset, shape, valid_shape, dtype=None):
    """The numpy oracle: the file window, clipped to the dataset and to
    ``valid_shape``, in a zero buffer of ``shape``."""
    k, dshape, fdtype = _READ_SET[name]
    full = _ref(k, dshape, fdtype)
    out = np.zeros(shape, np.dtype(dtype or fdtype))
    take = tuple(max(0, min(v, d - o)) for v, o, d in zip(valid_shape, offset, dshape))
    src = tuple(slice(o, o + t) for o, t in zip(offset, take))
    dst = tuple(slice(0, t) for t in take)
    out[dst] = full[src]
    return out


def _same(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and a.sharding == b.sharding
            and np.array_equal(np.asarray(a), np.asarray(b)))


def test_read_multi_and_read_slabs_return_what_read_slab_returns(tmp_path):
    from file_io.slab_io import SlabIO
    mesh = _cpu_mesh()
    path = str(tmp_path / "reads.h5")
    with h5py.File(path, "w") as f:
        for name, (k, shape, dtype) in _READ_SET.items():
            f[name] = _ref(k, shape, dtype)

    def requests(spec, names, q0, count):
        out = []
        for name in names:
            fields, rows, width = _READ_SET[name][1][1:]
            if spec == BATCH:
                shape = (count, fields, rows, width)
            else:      # face layout: rows and width padded to the mesh
                shape = (count, fields, -(-rows // 2) * 2, -(-width // 2) * 2)
            out.append(dict(name=name, shape=shape, offset=(q0, 0, 0, 0),
                            valid_shape=(count, fields, rows, width)))
        return out

    with SlabIO(path, mode="r", mesh=mesh) as io:
        cases = [("batch", BATCH, ["line_a", "line_b"], 0, 4),
                 ("batch_last_rows", BATCH, ["line_a", "line_b"], 1, 4),
                 ("face_ragged", FACE, ["line_a", "line_b", "line_a"], 1, 3),
                 ("mixed_dtype", FACE, ["line_b", "real_f64", "index_i64"], 2, 3)]
        for case, spec, names, q0, count in cases:
            reqs = requests(spec, names, q0, count)
            multi = io.read_multi(reqs, partition_spec=spec)
            single = [io.read_slab(r["name"], shape=r["shape"], offset=r["offset"],
                                   valid_shape=r["valid_shape"], partition_spec=spec)
                      for r in reqs]
            assert len(multi) == len(reqs), case
            for m, s, r in zip(multi, single, reqs):
                assert _same(m, s), case
                assert np.array_equal(np.asarray(m), _expected(
                    r["name"], r["offset"], r["shape"], r["valid_shape"])), case
        # With no valid_shape the window clips to the dataset: rows 3..6 of 5.
        r = dict(name="line_a", shape=(4, FIELDS, ROWS, 5), offset=(3, 0, 0, 0))
        clipped, = io.read_multi([r], partition_spec=BATCH)
        assert _same(clipped, io.read_slab(r["name"], shape=r["shape"], offset=r["offset"],
                                           partition_spec=BATCH))
        assert not np.asarray(clipped)[2:].any() and np.asarray(clipped)[:2].any()
        # A requested dtype is honoured by both routes alike.
        r = dict(name="real_f64", shape=(4, FIELDS, ROWS, 4), offset=(0, 0, 0, 0),
                 dtype=np.float32)
        conv, = io.read_multi([r], partition_spec=BATCH)
        assert conv.dtype == np.float32 and _same(conv, io.read_slab(
            "real_f64", shape=r["shape"], offset=r["offset"], dtype=np.float32,
            partition_spec=BATCH))

        # A zero-size slab is an empty array beside a real one, not an error.
        real = requests(BATCH, ["line_a"], 0, 4)[0]
        zero = dict(name="empty", shape=(4, FIELDS, ROWS, 0), offset=(0, 0, 0, 0),
                    valid_shape=(4, FIELDS, ROWS, 0))
        a, z = io.read_multi([real, zero], partition_spec=BATCH)
        assert z.shape == (4, FIELDS, ROWS, 0) and z.dtype == np.complex128
        assert _same(z, io.read_slab("empty", shape=zero["shape"], offset=zero["offset"],
                                     valid_shape=zero["valid_shape"], partition_spec=BATCH))
        assert np.array_equal(np.asarray(a), _expected(
            "line_a", real["offset"], real["shape"], real["valid_shape"]))

        # read_slabs: n windows of one slab shape, the window axis before the
        # fields.  The second window is ragged (one field of two is real).
        windows = [((0, 0, 0, 0), (4, 2, ROWS, 4)), ((0, 2, 0, 0), (4, 1, ROWS, 4))]
        shape = (4, 2, ROWS, 4)
        packed = io.read_slabs("real_f64", shape=shape, offsets=[w[0] for w in windows],
                               valid_shapes=[w[1] for w in windows], partition_spec=BATCH,
                               window_axis=1)
        assert packed.shape == (4, 2, 2, ROWS, 4)
        assert packed.sharding.spec == P(("x", "y"), None, None, None, None)
        per_window = [io.read_slab("real_f64", shape=shape, offset=o, valid_shape=v,
                                   partition_spec=BATCH) for o, v in windows]
        got = np.asarray(packed)
        for w, (o, v) in enumerate(windows):
            assert np.array_equal(got[:, w], np.asarray(per_window[w]))
            assert np.array_equal(got[:, w], _expected("real_f64", o, shape, v))
        assert not got[:, 1, 1].any() and got[:, 1, 0].any()
        # The two tables describe one list of windows: a length mismatch refuses.
        with pytest.raises(ValueError, match="different window counts"):
            io.read_slabs("real_f64", shape=shape, offsets=[windows[0][0]],
                          valid_shapes=[w[1] for w in windows], partition_spec=BATCH,
                          window_axis=1)


# ---- 3. write_slab(require_finite=True) ----------------------------------------

# ---- 4. the shared-directory nonce, two real processes -------------------------

# ---- 5. the platform key and the library legs ----------------------------------

# ---- 6. the per-rank streamed tier ---------------------------------------------

# ---- 7. open_h5_settled --------------------------------------------------------
