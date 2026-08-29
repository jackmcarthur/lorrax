"""SlabIO on an emulated mesh: four devices, one process, no MPI.

Layer L-b for the sharded-HDF5 transport, in the sense
``services/distrib_la/tests/test_distrib_la_emulated_mesh.py`` and
``services/wfn_loader/tests/test_wfn_loader_emulated_mesh.py`` already use:
four host devices in one process, a real 2x2 ``('x','y')`` mesh, real
sharded arrays, real HDF5 files.

Under test is ``file_io._slab_io_serial._SerialBackend``, the tier
``file_io.slab_io.SlabIO`` selects when ``common.collectives.
mesh_is_emulated(mesh)`` holds.  The tier exists because the phdf5
transport cannot serve this geometry: its C++ handler derives every
hyperslab from ``ctx->rank``, a per-process scalar, so four devices in one
process would all read and write shard (0, 0).  ``ffi.io.open_file``
refuses ``p*q != jax.process_count()`` for exactly that reason.

The first cell is the red twin and it is the point of the file.  A serial
tier added beside a refusal is indistinguishable, from a green suite
alone, from a refusal that was quietly weakened —
``test_the_phdf5_transport_still_refuses_an_emulated_mesh`` is the cell
that tells them apart, and it fails the moment someone "fixes"
``ffi/io.py`` instead.  The two doors are then shown to be different doors
by ``test_slab_io_selects_the_serial_tier_on_an_emulated_mesh``.

Every round trip is judged against plain h5py, not against another SlabIO
call: a transport compared only with itself agrees with its own bugs.  The
h5py handle is always opened after SlabIO has closed, which is the
one-owner ordering ``file_io.hdf5_owner`` enforces
(``docs/architecture/slab_io.md#one-owner``).

What this file cannot see.  It is single-process by construction, so it
says nothing about the collective transport's own correctness, nothing
about P>1, and nothing about performance — an emulated run serialises the
process-parallel k sweep (``common.collectives.local_share``) onto one
process.  Shapes and numbers only.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

h5py = pytest.importorskip("h5py")

from common.collectives import mesh_is_emulated          # noqa: E402
from file_io.slab_io import SlabIO                       # noqa: E402
from file_io._slab_io_ffi import _FfiBackend             # noqa: E402
from file_io._slab_io_serial import _SerialBackend       # noqa: E402


def _mesh(px: int = 2, py: int = 2) -> Mesh:
    """A ``px x py`` mesh of host devices, or a skip.

    ``jax.devices("cpu")`` explicitly and not ``jax.devices()``: the
    emulation knob is ``--xla_force_host_platform_device_count``, which
    creates host devices, so on a box whose default backend is CUDA the
    default device list answers 1 and these cells would skip on the very
    machine where the flag worked (the reasoning
    ``test_distrib_la_emulated_mesh._mesh`` records).
    """
    try:
        devs = jax.devices("cpu")
    except RuntimeError:                                   # no CPU backend
        devs = []
    n = px * py
    if len(devs) < n:
        pytest.skip(
            f"needs {n} cpu devices, have {len(devs)}; run this file with "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={n}")
    return Mesh(np.asarray(devs[:n]).reshape(px, py), ("x", "y"))


def _sharded(host: np.ndarray, mesh: Mesh, spec: P):
    return jax.device_put(jnp.asarray(host), NamedSharding(mesh, spec))


# ---------------------------------------------------------------------------
# The red twin: the door this tier is beside must still be shut
# ---------------------------------------------------------------------------

@pytest.mark.mesh(4)
def test_the_phdf5_transport_still_refuses_an_emulated_mesh(monkeypatch, tmp_path):
    """``ffi.io.open_file`` refuses ``p*q != process_count()``, observed.

    Without this cell the serial tier is unfalsifiable: a suite that only
    shows SlabIO working on an emulated mesh looks identical whether the
    tier was added beside the refusal or the refusal was deleted.

    ``ffi_loader.get_lib`` is stubbed because it runs first
    (``ffi/io.py``, the line above the check), so on a machine with no
    ``.so`` the library error would arrive instead and this cell would pass
    for the wrong reason — which is a cell that cannot fail.  The stub is
    the library-presence probe only; the refusal under test is untouched.
    """
    from ffi import io as ffi_io
    from ffi.common import ffi_loader

    mesh = _mesh(2, 2)
    assert jax.process_count() == 1 < mesh.devices.size
    monkeypatch.setattr(ffi_loader, "get_lib", lambda *_a, **_k: None)
    with pytest.raises(ValueError) as ei:
        ffi_io.open_file(str(tmp_path / "never_created.h5"), mesh=mesh,
                         mode="w")
    msg = str(ei.value)
    assert "process_count" in msg, msg
    assert "2×2" in msg or "2x2" in msg, msg
    # ...and the file was never created, i.e. the refusal came before any
    # byte moved.  A refusal that has already made an inode is a different
    # (and worse) thing than a refusal.
    assert not (tmp_path / "never_created.h5").exists()


@pytest.mark.mesh(4)
def test_slab_io_selects_the_serial_tier_on_an_emulated_mesh(tmp_path):
    """The door SlabIO walks through is the other one — checked by type.

    Not by "it worked": the FFI tier could also work on some machine at
    some geometry, and then a silent relaxation of ``ffi/io.py`` would look
    like this cell passing.  The backend's identity is the fact.
    """
    mesh = _mesh(2, 2)
    assert mesh_is_emulated(mesh)
    with SlabIO(tmp_path / "tier.h5", mode="w", mesh=mesh) as io:
        assert isinstance(io._backend, _SerialBackend)
        assert not isinstance(io._backend, _FfiBackend)


@pytest.mark.mesh(4)
def test_the_tier_announces_itself_in_the_log(tmp_path, capsys):
    """The transport line, observed in stdout rather than assumed.

    A parity claim made from an emulated arm has to quote the transport
    that moved its bytes (TASTE rule 15), and a reader must not have to
    infer it from the device count.  The line is unconditional — this
    tier prints on exactly the runs that did not go through collective
    MPI-IO, so it cannot become noise — and once per (devices, processes)
    geometry, not once per file.
    """
    import file_io._slab_io_serial as serial

    mesh = _mesh(2, 2)
    serial._ANNOUNCED.clear()
    with SlabIO(tmp_path / "a.h5", mode="w", mesh=mesh):
        pass
    first = capsys.readouterr().out
    assert "transport = serial" in first, first
    assert "1 process" in first and "4 devices" in first, first
    assert "process_count" in first, first
    with SlabIO(tmp_path / "b.h5", mode="w", mesh=mesh):
        pass
    assert "transport = serial" not in capsys.readouterr().out


@pytest.mark.mesh(4)
def test_the_receipt_is_discarded_by_the_production_stdout_sink(tmp_path):
    """…and a production driver run does not show it.  Measured, not assumed.

    ``runtime.production_stream.ProductionStdout`` — installed by
    ``gw.gw_jax`` and every other driver — points ``sys.stdout`` at
    ``/dev/null`` so that "ordinary component chatter is discarded".  The
    announcement above is ordinary component chatter by that definition,
    so it does not reach a production log; it reappears under
    ``LORRAX_DEBUG_PRINT=1`` (``debug=True`` here).

    This cell exists because the tier's first docstring claimed the line
    printed "on exactly the runs whose bytes did not go through collective
    MPI-IO", and the first end-to-end emulated gnppm_debug run's log named
    the transport nowhere.  A claim about where output lands is a claim
    that has to be executed.  The consequence — the driver's scientific
    report does not yet carry the slab transport the way it carries
    ``Wavefunctions  : <backend> reader`` — is a gap, recorded here so it
    is a known fact rather than a rediscovery.
    """
    import io as _io
    import sys

    import file_io._slab_io_serial as serial
    from runtime.production_stream import ProductionStdout

    mesh = _mesh(2, 2)

    def visible(debug: bool) -> bool:
        serial._ANNOUNCED.clear()
        cap = _io.StringIO()
        real, sys.stdout = sys.stdout, cap
        ps = ProductionStdout(debug=debug, rank=0)
        ps.install()
        try:
            with SlabIO(tmp_path / f"p{int(debug)}.h5", mode="w", mesh=mesh):
                pass
        finally:
            ps.close()
            sys.stdout = real
        return "transport = serial" in cap.getvalue()

    assert visible(True), "LORRAX_DEBUG_PRINT=1 must still show the receipt"
    assert not visible(False), (
        "the production sink no longer swallows the receipt — if that is "
        "intentional, this cell and the tier's docstring table are the "
        "record that has to change with it")


@pytest.mark.mesh(4)
def test_the_serial_tier_refuses_a_non_emulated_mesh(tmp_path):
    """A 1x1 at one process is not emulated and must not reach this tier.

    The tier is selected from a predicate, and the predicate is false here;
    a tier that accepted the geometry anyway would be a second transport
    for the production single-device case, which is what the 2026-08-06
    deletion removed.
    """
    one = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    assert not mesh_is_emulated(one)
    with pytest.raises(RuntimeError) as ei:
        _SerialBackend(str(tmp_path / "no.h5"), mesh=one, mode="w")
    assert "mesh_is_emulated" in str(ei.value)


@pytest.mark.mesh(4)
def test_the_serial_tier_refuses_above_one_process(tmp_path, monkeypatch):
    """P > 1 refuses rather than demoting — reached by stubbing the count.

    This branch cannot be entered by a single-process suite, and a refusal
    no test can reach is a refusal nobody has read.  ``process_count`` is
    stubbed at the module that consults it, which is the only input the
    branch has; the message must name the mechanism and both ways out.
    """
    import file_io._slab_io_serial as serial

    mesh = _mesh(2, 2)
    monkeypatch.setattr(serial, "process_count", lambda: 4)
    with pytest.raises(RuntimeError) as ei:
        serial._SerialBackend(str(tmp_path / "no.h5"), mesh=mesh, mode="w")
    msg = str(ei.value)
    assert "process_count() == 1" in msg, msg
    assert "phdf5" in msg, msg


class _FakeGpuDevice:
    """A device that says ``gpu``, and nothing else.

    ``ffi.gate.mesh_ffi_platform`` reads exactly one attribute
    (``mesh.devices.flat[0].platform``), so this is the whole of what a GPU
    looks like to the guard under test — and the guard is exercised through
    the real helper and the real predicate rather than a stub of either.
    """

    platform = "gpu"

    def __repr__(self) -> str:                          # pragma: no cover
        return "<FakeGpuDevice>"


class _FakeGpuMesh:
    """A 2x2 mesh of them: single-process, multi-GPU — the deleted arm."""

    def __init__(self) -> None:
        self.devices = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                self.devices[i, j] = _FakeGpuDevice()
        self.axis_names = ("x", "y")
        self.shape = {"x": 2, "y": 2}


@pytest.mark.mesh(4)
def test_the_serial_tier_refuses_a_single_process_multi_gpu_mesh(tmp_path):
    """The geometry TASTE rule 10 calls deleted must not come back here.

    ``mesh_is_emulated`` reads only ``devices.size`` against
    ``process_count()`` and is platform-blind — asserted below on the very
    object, so this is not a claim about the predicate but an observation of
    it.  A single-process multi-GPU mesh therefore satisfies it: that is
    ``resolve_mesh()`` on a 4-GPU box, and the mesh harness child that hands
    one process all four GPUs.

    On base that geometry hit ``ffi.io.open_file``'s ``p*q !=
    process_count()`` and refused.  Adding a serial tier keyed on the
    predicate alone would have re-opened it through a route nobody chose —
    a deleted arm downgrading instead of refusing, which is the one thing
    the rule forbids.  The guard is what keeps the tier CPU-emulation-only,
    and this cell is its false case: without it, the construction below
    succeeds and writes an HDF5 file.
    """
    from ffi.gate import mesh_ffi_platform

    fake = _FakeGpuMesh()
    # The two premises, observed rather than asserted in prose.
    assert mesh_is_emulated(fake), "the predicate is platform-blind"
    assert mesh_ffi_platform(fake) == "CUDA"

    path = tmp_path / "gpu_emulated.h5"
    with pytest.raises(RuntimeError) as ei:
        _SerialBackend(str(path), mesh=fake, mode="w")
    msg = str(ei.value)
    assert "multi-GPU" in msg, msg
    assert "src/ffi/io.py" in msg, msg
    # ...and it refused before making an inode, like the phdf5 door above.
    assert not path.exists()


@pytest.mark.mesh(4)
@pytest.mark.parametrize("mode", ["r+", "x", "w-", "bogus"])
def test_the_serial_tier_refuses_a_mode_outside_w_a_r(tmp_path, mode):
    """The mode vocabulary is the transport's, not h5py's.

    ``ffi.io.open_file`` refuses anything outside ``w``/``a``/``r``
    (``src/ffi/io.py``), and h5py accepts six modes.  Left to h5py, an
    emulated run would accept ``r+``, and would accept ``x`` and ``w-``
    while creating a file that skipped ``_replace_inode_for_write`` — so
    ``mode`` would mean different things depending on the device geometry,
    and only the emulated tier would say so in h5py's words, naming modes
    SlabIO does not have.

    ``bogus`` is in the list on purpose: it used to refuse too, but with
    h5py's text ("Invalid mode; must be one of r, r+, w, w-, x, a"), which
    advertises the wrong vocabulary and no LORRAX rule.  All four must now
    refuse the same way, from the same door.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / f"mode_{mode.replace('+', 'p').replace('-', 'm')}.h5"
    with pytest.raises(ValueError) as ei:
        _SerialBackend(str(path), mesh=mesh, mode=mode)
    msg = str(ei.value)
    assert "mode must be w/a/r" in msg, msg
    assert repr(mode) in msg, msg
    assert not path.exists(), "a refused mode must not have made an inode"


@pytest.mark.mesh(4)
def test_a_close_that_raises_still_releases_the_owner_claim(tmp_path):
    """``note_close`` in a ``finally``, not after a close that returned.

    A leaked claim is worse than the failed close that caused it: the path
    stays owned in ``file_io.hdf5_owner`` for the life of the process, and
    the next open of it — by either library — is refused by a claim whose
    owner is gone.  The FFI tier releases unconditionally and documents
    why; a tier whose thesis is contract-parity with it must not diverge.

    Reached by breaking the handle, which is the only input the branch has.
    The false case: with the release outside the ``finally`` the assertion
    below fails, because ``_owner_token`` is still an int.
    """
    from file_io import hdf5_owner

    mesh = _mesh(2, 2)
    path = tmp_path / "raises.h5"
    be = _SerialBackend(str(path), mesh=mesh, mode="w")
    assert be._owner_token is not None

    class _Boom:
        def close(self):
            raise OSError("H5Fclose blew up")

    be._h5 = _Boom()
    with pytest.raises(OSError):
        be.close()
    assert be._owner_token is None, "the hdf5_owner claim leaked"
    # The registry agrees, which is the fact that matters: a fresh open of
    # the same path must not be refused by the dead handle's claim.
    tok = hdf5_owner.note_open(str(path), hdf5_owner.STACK_H5PY, "r",
                               where="test: the path must be re-openable")
    hdf5_owner.note_close(str(path), tok)


# ---------------------------------------------------------------------------
# Round trips, judged against plain h5py
# ---------------------------------------------------------------------------

@pytest.mark.mesh(4)
def test_a_two_axis_sharded_write_lands_where_h5py_reads_it(tmp_path):
    """Every shard's own bytes, at its own global offset.

    The failure this excludes is the one that made the refusal necessary:
    four devices writing shard (0, 0), i.e. one quarter of the array
    replicated four times.  Comparing against the whole host array makes
    that a mismatch on three quarters of the file rather than an equality
    that holds on the piece everyone wrote.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "two_axis.h5"
    rng = np.random.default_rng(11)
    host = (rng.standard_normal((8, 6, 5))
            + 1j * rng.standard_normal((8, 6, 5))).astype(np.complex128)
    A = _sharded(host, mesh, P("x", "y", None))
    assert len(A.addressable_shards) == 4

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=host.shape, dtype=np.complex128)
        io.write_slab("A", A)

    with h5py.File(path, "r") as f:
        assert np.array_equal(np.asarray(f["A"][:]), host)


@pytest.mark.mesh(4)
def test_a_replicated_write_lands_once_and_correctly(tmp_path):
    """The kin_ion / sigma_mnk shape: ``P(None, ...)`` on every dim.

    Four devices hold identical bytes; the file must hold them once.  This
    is the request ``shard_index.h``'s world-size backstop refuses even
    though it has no per-rank hyperslab to protect — on this tier there is
    no rank arithmetic to protect either, and the de-duplication is by
    shard index rather than by rank.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "replicated.h5"
    host = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    A = _sharded(host, mesh, P(None, None))

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("R", shape=host.shape, dtype=np.float64)
        io.write_slab("R", A)

    with h5py.File(path, "r") as f:
        assert np.array_equal(np.asarray(f["R"][:]), host)


def _counting_dataset_io(monkeypatch):
    """Count real ``h5py.Dataset`` element writes and reads.

    Counted at h5py's own door rather than at ``_SerialBackend``'s ``seen``
    set: the ``seen`` set is the mechanism under test, and a counter built
    on it would agree with it by construction.
    """
    counts = {"write": 0, "read": 0}
    real_set = h5py.Dataset.__setitem__
    real_get = h5py.Dataset.__getitem__

    def counting_set(self, key, value):
        counts["write"] += 1
        return real_set(self, key, value)

    def counting_get(self, key):
        counts["read"] += 1
        return real_get(self, key)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", counting_set)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", counting_get)
    return counts


@pytest.mark.mesh(4)
def test_a_replicated_write_costs_exactly_ONE_h5py_write(tmp_path,
                                                         monkeypatch):
    """The de-duplication, counted — the cell above cannot see it.

    ``test_a_replicated_write_lands_once_and_correctly`` asserts only that
    the bytes are right, and four identical writes of identical bytes leave
    a file indistinguishable from one.  Falsified: disabling the ``seen``
    check in ``_SerialBackend.write_slab`` left that cell — and the whole
    file — green.  This one counts, so it goes red at 4.

    Four devices, a fully replicated spec, one dataset: one ``__setitem__``.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "dedup_w.h5"
    host = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    A = _sharded(host, mesh, P(None, None))

    counts = _counting_dataset_io(monkeypatch)
    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("R", shape=host.shape, dtype=np.float64)
        io.write_slab("R", A)
    assert counts["write"] == 1, (
        f"a replicated 4-device write issued {counts['write']} h5py writes; "
        f"the replica de-duplication by shard index is not firing")

    with h5py.File(path, "r") as f:                     # still correct
        assert np.array_equal(np.asarray(f["R"][:]), host)


@pytest.mark.mesh(4)
@pytest.mark.parametrize("spec, want_reads", [
    (P(None, None), 1),        # jax itself calls the callback once here
    (P("x", None), 2),         # two distinct row blocks, two devices each
    (P(None, "y"), 2),         # ...and the alternating order, which a
                               #    one-entry memo would miss entirely
    (P("x", "y"), 4),          # four distinct shards — the floor, no cache
])
def test_a_sharded_read_issues_one_h5py_read_per_DISTINCT_shard(
        tmp_path, monkeypatch, spec, want_reads):
    """``read_slab``'s replica memo, counted.  It had no cell at all.

    The memo exists so a 4-device replicated read is one ``H5Dread`` and
    not four.  That is invisible to every other cell in this file, all of
    which judge values — and identical bytes read four times are the same
    bytes.  The three rows are the whole trade: the floor is one read per
    distinct shard, and the memo reaches it.

    ``P(None,'y')`` is the row that earns its place: its indices arrive
    alternating (col0, col1, col0, col1), so a cache holding only the last
    block hits zero times and reads 4.  A one-entry memo was written here
    first and this row is what caught it; without it the suite would have
    accepted a 2x regression in read count on half the partially-replicated
    specs.

    The false case for the replicated rows is a cache that never hits (4
    reads); for the fully sharded row it is a cache that hits when it must
    not (a wrong-shard read, which the value assertion below then catches).
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "dedup_r.h5"
    host = np.arange(4 * 4, dtype=np.float64).reshape(4, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("Z", data=host)

    counts = _counting_dataset_io(monkeypatch)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        got = io.read_slab("Z", partition_spec=spec)
    assert counts["read"] == want_reads, (
        f"{spec} issued {counts['read']} h5py reads, expected {want_reads}")
    assert np.array_equal(np.asarray(jax.device_get(got)), host)


@pytest.mark.mesh(4)
def test_a_host_numpy_operand_and_an_off_mesh_array_both_write(tmp_path):
    """The two operands that carry no mesh sharding at all.

    ``file_io.sigma_output``'s ``sigma_mnk.h5`` writer hands SlabIO host
    arrays assembled by ``extract_and_stamp_k_irr``, and a committed
    single-device ``jax.Array`` reaches the same door from other callers.
    Neither has a ``NamedSharding`` on this mesh, so neither can be read
    through the mesh's axis indices — the FFI tier re-places them as
    replicated, this one takes their single shard.  Both must land the
    same bytes as the host array they are.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "hostops.h5"
    host = np.arange(4 * 4, dtype=np.float64).reshape(4, 4)
    committed = jax.device_put(jnp.asarray(host), jax.devices("cpu")[0])

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("H", shape=host.shape, dtype=np.float64)
        io.write_slab("H", host)
        io.create_dataset("D", shape=host.shape, dtype=np.float64)
        io.write_slab("D", committed)

    with h5py.File(path, "r") as f:
        assert np.array_equal(np.asarray(f["H"][:]), host)
        assert np.array_equal(np.asarray(f["D"][:]), host)


@pytest.mark.mesh(4)
def test_pad_rows_past_the_dataset_are_dropped_with_no_argument(tmp_path):
    """``write_slab`` clips to ``min(A.shape, dataset - offset)``.

    The μ-pad case: the buffer is mesh-divisible, the dataset is the
    physics extent, and the caller states neither.  ``valid_shape`` is
    SlabIO's derivation on both tiers, so a tier that wrote its pad rows
    would grow the file past the extent it advertises.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "padded_write.h5"
    logical = (5, 3)                       # 5 is not divisible by 2
    padded = np.arange(6 * 3, dtype=np.float64).reshape(6, 3)
    A = _sharded(padded, mesh, P("x", None))

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("Z", shape=logical, dtype=np.float64)
        io.write_slab("Z", A)

    with h5py.File(path, "r") as f:
        assert f["Z"].shape == logical
        assert np.array_equal(np.asarray(f["Z"][:]), padded[:5])


@pytest.mark.mesh(4)
def test_the_easy_read_returns_the_padded_buffer_zero_filled(tmp_path):
    """``read_slab`` with no ``shape``: mesh-divisible, zeros past the file.

    The contract ``SlabIO.read_slab`` publishes ("the easy call is the
    correct call"), which the caller depends on to size its own buffers.
    A tier that returned the dataset's own extent would refuse to shard at
    all on this geometry.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "padded_read.h5"
    ds = np.arange(5 * 4, dtype=np.float64).reshape(5, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("Z", data=ds)

    with SlabIO(path, mode="r", mesh=mesh) as io:
        out = io.read_slab("Z", partition_spec=P("x", "y"))
    got = np.asarray(jax.device_get(out))

    assert got.shape == (6, 4)                    # ceil(5/2)*2
    assert np.array_equal(got[:5], ds)
    assert np.array_equal(got[5], np.zeros(4))


@pytest.mark.mesh(4)
def test_an_offset_read_of_a_sub_block_matches_h5py(tmp_path):
    """``offset`` + explicit ``shape``, sharded, against the same slice."""
    mesh = _mesh(2, 2)
    path = tmp_path / "offset_read.h5"
    ds = np.arange(10 * 8, dtype=np.float64).reshape(10, 8)
    with h5py.File(path, "w") as f:
        f.create_dataset("Z", data=ds)

    with SlabIO(path, mode="r", mesh=mesh) as io:
        out = io.read_slab("Z", shape=(4, 4), offset=(3, 2),
                           partition_spec=P("x", "y"))
    assert np.array_equal(np.asarray(jax.device_get(out)), ds[3:7, 2:6])


@pytest.mark.mesh(4)
def test_write_then_read_round_trips_through_the_same_handle(tmp_path):
    """A read after a write on one handle sees the written bytes.

    On the collective tier this is the read-after-write hazard the
    unconditional drain exists for; here the writes are synchronous, so the
    cell measures that the stronger guarantee actually holds rather than
    assuming it from the absence of a queue.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "rw.h5"
    rng = np.random.default_rng(3)
    host = rng.standard_normal((4, 4)).astype(np.float64)
    A = _sharded(host, mesh, P("x", "y"))

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=host.shape, dtype=np.float64)
        io.write_slab("A", A)
        back = np.asarray(jax.device_get(
            io.read_slab("A", partition_spec=P("x", "y"))))
    assert np.array_equal(back, host)


@pytest.mark.mesh(4)
def test_write_attr_and_dataset_attrs_land_at_close(tmp_path):
    """The deferred metadata contract, in the file, after ``close``.

    ``k_storage`` is the stamp whose loss made a cluster-written wedge Σ
    cube read back as full-BZ, so the attrs path is not decorative.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "attrs.h5"
    io = SlabIO(path, mode="w", mesh=mesh)
    io.create_dataset("A", shape=(4, 4), dtype=np.float64,
                      attrs={"k_storage": "ibz"})
    io.write_attr("omega_ev", np.linspace(0.0, 1.0, 5))
    io.write_slab("A", _sharded(np.zeros((4, 4)), mesh, P("x", "y")))
    with h5py.File(path, "r") as f:
        assert "omega_ev" not in f          # not before close: the contract
    io.close()

    with h5py.File(path, "r") as f:
        assert np.allclose(np.asarray(f["omega_ev"][:]),
                           np.linspace(0.0, 1.0, 5))
        assert str(f["A"].attrs["k_storage"]) == "ibz"


@pytest.mark.mesh(4)
def test_read_small_serves_a_scalar_dataset(tmp_path):
    """The scalar door: an H5 rank-0 dataspace has no hyperslab."""
    mesh = _mesh(2, 2)
    path = tmp_path / "scalar.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("nband", data=np.int64(46))

    with SlabIO(path, mode="r", mesh=mesh) as io:
        got = io.read_small("nband")
    assert int(got) == 46


@pytest.mark.mesh(4)
def test_create_dataset_reuses_an_identical_one_and_refuses_a_clash(tmp_path):
    """decisions.md 2026-08-04, restated on this tier because it must be.

    The collective tier gets reuse-or-refuse from
    ``lrx_phdf5_ensure_dataset``; this one has to implement it, so it has
    to be checked here or the two tiers disagree about what a second
    ``create_dataset`` means — and the disagreement is data loss on one of
    them.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "clash.h5"
    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(4, 4), dtype=np.float64)
        io.create_dataset("A", shape=(4, 4), dtype=np.float64)   # idempotent
        with pytest.raises(ValueError) as ei:
            io.create_dataset("A", shape=(8, 4), dtype=np.float64)
    assert "(4, 4)" in str(ei.value) and "(8, 4)" in str(ei.value)


@pytest.mark.mesh(4)
def test_read_slabs_packs_n_windows_exactly_as_n_read_slab_calls_would(tmp_path):
    """The multi-window read, against the fold-down it replaces.

    ``read_slabs`` exists on the collective tier because n per-window
    collectives cost 1.4 s of rendezvous at the production deck; the
    serial tier owes the same answer, not the same mechanism.  The
    fold-down is the reference precisely because it shares no code with
    the window loop under test.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "windows.h5"
    ds = np.arange(12 * 4, dtype=np.float64).reshape(12, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("W", data=ds)

    # Three disjoint ascending windows of a common (4, 4) slab; the last is
    # ragged (only 2 of its 4 rows are inside the dataset).
    offsets = [(0, 0), (4, 0), (8, 0)]
    valid = [(4, 4), (4, 4), (2, 4)]

    with SlabIO(path, mode="r", mesh=mesh) as io:
        packed = np.asarray(jax.device_get(io.read_slabs(
            "W", shape=(4, 4), offsets=offsets, valid_shapes=valid,
            partition_spec=P("x", "y"), window_axis=0)))
        folded = np.stack([
            np.asarray(jax.device_get(io.read_slab(
                "W", shape=(4, 4), offset=o, valid_shape=v,
                partition_spec=P("x", "y"))))
            for o, v in zip(offsets, valid)])

    assert packed.shape == (3, 4, 4)
    assert np.array_equal(packed, folded)
    assert np.array_equal(packed[0], ds[0:4])
    assert np.array_equal(packed[2][:2], ds[8:10])
    assert np.array_equal(packed[2][2:], np.zeros((2, 4)))


@pytest.mark.mesh(4)
def test_mode_w_replaces_the_file_rather_than_appending_to_it(tmp_path):
    """``mode='w'`` is a replace contract on both tiers."""
    mesh = _mesh(2, 2)
    path = tmp_path / "replace.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("old", data=np.zeros(3))

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("new", shape=(4, 4), dtype=np.float64)
        io.write_slab("new", _sharded(np.ones((4, 4)), mesh, P("x", "y")))

    with h5py.File(path, "r") as f:
        assert "old" not in f
        assert "new" in f


@pytest.mark.mesh(4)
def test_read_slabs_at_the_production_geometry(tmp_path):
    """The shape ``wfn_loader`` actually asks for, not a convenient one.

    ``services/wfn_loader/src/wfn_loader/loader.py`` reads ``wfns/coeffs``
    as a 4-D slab with ``window_axis=2`` and
    ``partition_spec=P(('x','y'), None, None, None)`` — a window axis that
    is not zero, and one dim sharded by both mesh axes.  The cell above
    exercises neither: at ``window_axis=0`` the ``PartitionSpec`` split
    around the window is trivial, and a single-axis spec never reaches
    ``_sharding_to_axis_info``'s tuple branch.  A tier tested only on the
    easy geometry is a tier tested on a geometry no caller uses.

    Judged against the per-window fold-down, which shares no code with the
    window loop, and against plain h5py for window 0.
    """
    mesh = _mesh(2, 2)
    path = tmp_path / "prod_windows.h5"
    # (bands, ns, ngkmax, 2) with three k windows down the ngkmax axis.
    nb, ns, ngkmax = 8, 2, 5
    ds = np.arange(nb * ns * 15 * 2, dtype=np.float64).reshape(nb, ns, 15, 2)
    with h5py.File(path, "w") as f:
        f.create_dataset("coeffs", data=ds)

    offsets = [(0, 0, 0, 0), (0, 0, 5, 0), (0, 0, 10, 0)]
    valid = [(nb, ns, 5, 2), (nb, ns, 4, 2), (nb, ns, 5, 2)]  # middle ragged
    spec = P(("x", "y"), None, None, None)

    with SlabIO(path, mode="r", mesh=mesh) as io:
        packed = np.asarray(jax.device_get(io.read_slabs(
            "coeffs", shape=(nb, ns, ngkmax, 2), offsets=offsets,
            valid_shapes=valid, partition_spec=spec, window_axis=2)))
        folded = np.stack([
            np.asarray(jax.device_get(io.read_slab(
                "coeffs", shape=(nb, ns, ngkmax, 2), offset=o,
                valid_shape=v, partition_spec=spec)))
            for o, v in zip(offsets, valid)], axis=2)

    assert packed.shape == (nb, ns, 3, ngkmax, 2)
    assert np.array_equal(packed, folded)
    assert np.array_equal(packed[:, :, 0], ds[:, :, 0:5])
    # the ragged window: 4 real rows, then zero
    assert np.array_equal(packed[:, :, 1, :4], ds[:, :, 5:9])
    assert np.array_equal(packed[:, :, 1, 4], np.zeros((nb, ns, 2)))


@pytest.mark.mesh(4)
def test_the_journal_names_the_library_the_tier_actually_used(tmp_path,
                                                              monkeypatch,
                                                              request):
    """``stack=h5py``, never ``stack=ffi``, on an emulated open.

    The journal's whole subject is which HDF5 library instance touched a
    file — it is read beside ``file_io.hdf5_owner``'s verdict, which is
    keyed on the same two names.  ``slab_io`` used to stamp a module
    constant ``_STACK = "ffi"``, true while there was one transport; the
    first emulated gnppm_debug run's ``h5_journal.rank0.log`` then showed
    ``stack=h5py op=open`` from the serial backend and ``stack=ffi
    op=open`` from the door, on the same file in the same millisecond.

    Read out of the journal file rather than off the class attribute: the
    attribute is what the fix changed, so asserting on it would test the
    fix against itself.
    """
    from file_io import h5_journal as J

    monkeypatch.setenv("LORRAX_H5_JOURNAL", "1")
    monkeypatch.chdir(tmp_path)          # the journal lands in the cwd
    J.reset_for_test()                   # ...and only if no stream is cached
    request.addfinalizer(J.reset_for_test)
    mesh = _mesh(2, 2)
    with SlabIO(tmp_path / "j.h5", mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(4, 4), dtype=np.float64)
        io.write_slab("A", _sharded(np.ones((4, 4)), mesh, P("x", "y")))

    logs = sorted(tmp_path.glob("h5_journal.rank*.log"))
    assert logs, f"no journal written into {tmp_path}"
    lines = [ln for log in logs for ln in log.read_text().splitlines()
             if "/j.h5" in ln]
    assert lines, "journal has no lines for the file under test"
    assert all("stack=h5py" in ln for ln in lines), (
        "an emulated-tier operation was journaled under the FFI library:\n"
        + "\n".join(ln for ln in lines if "stack=h5py" not in ln))


@pytest.mark.mesh(4)
def test_one_slab_write_is_ONE_journal_line_on_this_tier_too(tmp_path,
                                                             monkeypatch,
                                                             request):
    """``docs/architecture/slab_io.md``: "one slab read or write is **one**
    line".  The serial tier emitted two.

    The seam (``SlabIO.write_slab``) opens an ``op_scope`` for every op on
    both tiers; ``_SerialBackend`` opened a second one of its own for
    ``write``, ``read_slab`` and ``read_slabs``, so the same op appeared
    twice — once with ``off=-`` from the door and once with ``off=(0,0)``
    from the backend.  The FFI backend journals no write or read of its
    own, which is what makes the two tiers' logs comparable line for line.

    Counted from the file, and counted for both directions.  The false case
    is the backend re-opening its own scope: the counts go to 2.
    """
    from file_io import h5_journal as J

    monkeypatch.setenv("LORRAX_H5_JOURNAL", "1")
    monkeypatch.chdir(tmp_path)
    # The journal opens its stream once per process and caches the path, so a
    # cell that runs after another journal cell would append to that one's
    # tmp_path and read an empty log here.  Reset both ways round.
    J.reset_for_test()
    request.addfinalizer(J.reset_for_test)
    mesh = _mesh(2, 2)
    with SlabIO(tmp_path / "lines.h5", mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(4, 4), dtype=np.float64)
        io.write_slab("A", _sharded(np.ones((4, 4)), mesh, P("x", "y")))
    with SlabIO(tmp_path / "lines.h5", mode="r", mesh=mesh) as io:
        io.read_slab("A", partition_spec=P("x", "y"))

    lines = [ln for log in sorted(tmp_path.glob("h5_journal.rank*.log"))
             for ln in log.read_text().splitlines() if "/lines.h5" in ln]
    assert lines, "journal has no lines for the file under test"
    n_write = sum(1 for ln in lines if "op=write" in ln)
    n_read = sum(1 for ln in lines if "op=read" in ln)
    assert n_write == 1, (
        f"one write_slab produced {n_write} op=write lines:\n"
        + "\n".join(ln for ln in lines if "op=write" in ln))
    assert n_read == 1, (
        f"one read_slab produced {n_read} op=read lines:\n"
        + "\n".join(ln for ln in lines if "op=read" in ln))


@pytest.mark.mesh(4)
def test_the_journal_receipt_survives_the_production_stdout_sink(tmp_path,
                                                                 monkeypatch,
                                                                 request):
    """The journal line a production parity claim can actually quote.

    ``test_the_receipt_is_discarded_by_the_production_stdout_sink`` pins the
    bad news: the printed ``transport = serial`` line does not reach a
    production log.  This is the good news, and it has to be executed for
    the same reason that one did — the tier's docstring now points a reader
    at this receipt, so "the journal survives the sink" is a claim about
    where output lands, and those are claims that get run.

    The journal writes to a file through a stream ``file_io.h5_journal``
    owns; ``ProductionStdout`` replaces ``sys.stdout`` only.  So with
    ``LORRAX_H5_JOURNAL=1`` the ``stack=h5py op=open`` line is there even
    on a run whose printed receipt was swallowed — both facts asserted in
    the same block, so they cannot drift apart.
    """
    import io as _io
    import sys

    import file_io._slab_io_serial as serial
    from file_io import h5_journal as J
    from runtime.production_stream import ProductionStdout

    monkeypatch.setenv("LORRAX_H5_JOURNAL", "1")
    monkeypatch.chdir(tmp_path)
    J.reset_for_test()                       # see the cell above for why
    request.addfinalizer(J.reset_for_test)
    mesh = _mesh(2, 2)
    serial._ANNOUNCED.clear()

    cap = _io.StringIO()
    real, sys.stdout = sys.stdout, cap
    ps = ProductionStdout(debug=False, rank=0)
    ps.install()
    try:
        with SlabIO(tmp_path / "prod.h5", mode="w", mesh=mesh) as io:
            io.create_dataset("A", shape=(4, 4), dtype=np.float64)
    finally:
        ps.close()
        sys.stdout = real

    assert "transport = serial" not in cap.getvalue(), (
        "the printed receipt reached production stdout; if that is now "
        "intended, the tier's docstring table is the record to change")
    lines = [ln for log in sorted(tmp_path.glob("h5_journal.rank*.log"))
             for ln in log.read_text().splitlines() if "/prod.h5" in ln]
    opens = [ln for ln in lines if "op=open" in ln and "stack=h5py" in ln]
    assert opens, (
        "no `stack=h5py op=open` line survived the production sink — the "
        "tier's docstring names this as THE production-visible receipt, so "
        "either the journal or that paragraph is now wrong:\n"
        + "\n".join(lines))
