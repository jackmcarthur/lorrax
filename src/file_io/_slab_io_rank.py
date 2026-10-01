"""SlabIO's per-rank streamed tier: a sharded operator written segment by segment and read back output by output.

The direct response stream (``gw.response_bank``, ``gw.subtile_stream``) finishes
the rows of one row pass for every output at once (the value and the slope of
every sample).  When the whole bank of outputs does not fit the devices, it
comes here instead of being streamed again for each sample group: every pass
is written once, and each consumer reads one sample's value and slope back.

**One rank, one private store.**  Each device keeps its own (μ_X, ν_Y) tile
in one file (``kind="file"``) or one host buffer (``kind="host"``).  No rank
reads another's, so the tier issues no collective of its own; its commit,
reads and release end in :func:`common.collectives.agree_io_error`, which
every rank calls at the same point, so a failure on one rank is raised on all
of them (INVARIANTS 21).  Nothing that depends on a rank's filesystem decides
whether a collective runs.

**Layout (per device).**  A record is segment ``s``'s local tile ``[q, rows_s,
cols_s]`` of output ``o`` (a row pass of the charge stream, or one family
pair's row pass of the four-current stream with its channel blocks packed;
``gw.subtile_stream.segment_blocks``), ``16·q·rows_s·cols_s`` bytes padded to
4 KiB.  Records are output-major, at ``o·S + start_s`` with ``S = Σ_s
padded_s``, so one consumer's read of outputs ``[o0, o1)`` is the one
contiguous run ``[o0·S, o1·S)``; each segment's rectangles go back to their
places in the local carry tile on the device.

**Write.**  A drain thread moves each finished segment's records off the
device in 64 MiB pieces (one program per piece shape lands a piece, padded
to 4 KiB, in pinned host memory: 4 KiB aligned, so its numpy view is the
O_DIRECT source with no copy) and hands them to a pool of I/O threads.  At
most :data:`PIECES_IN_FLIGHT` pieces are staged per rank (256 MiB pinned):
a piece is moved only after an earlier one is written (backpressure), and
the devices compute the next segment meanwhile; at most :data:`IN_FLIGHT`
segment carries are alive.  One program per segment digests its records
(position-weighted, per record).

**Read.**  One run is read one ahead of the consumer by the I/O threads into
an aligned staging buffer (a file) or taken in place (host), copied to the
device, cut back from records to rows and digested by one program; the
digests must equal the write digests.

Files are created on a 4-stripe, 4 MiB Lustre layout through liblustreapi
when it is present (a one-stripe file reads at a quarter of the rate:
measured 1.2 vs 6.1 GB/s per rank, one thread); O_DIRECT is required, and a
filesystem without it refuses by name.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import mmap
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import agree_io_error

#: O_DIRECT alignment of every buffer address, file offset and length.
ALIGN = 4096
#: Bytes per pwrite/pread call.
PIECE = 64 << 20
#: I/O threads per process (four reach 13-14 GB/s per rank on a 4-stripe file).
IO_THREADS = 4
#: Segment carries alive at once (one computing, one draining).
IN_FLIGHT = 2
#: Pinned pieces staged per rank (backpressure: a piece moves only after an earlier one is written).
PIECES_IN_FLIGHT = 4
#: Lustre layout of every file: stripe count and stripe size.
STRIPES, STRIPE_BYTES = 4, 4 << 20


def padded(nbytes: int) -> int:
    """``nbytes`` rounded up to :data:`ALIGN`."""
    return -(-int(nbytes) // ALIGN) * ALIGN


def _digest(flat):
    """Position-weighted digest of each row of a complex128 ``[..., m]``: Σ_i bits_i·(2i+1) mod 2⁶⁴."""
    bits = jax.lax.bitcast_convert_type(jnp.stack([flat.real, flat.imag], axis=-1), jnp.uint64)
    bits = bits.reshape(flat.shape[:-1] + (-1,))
    weight = 2 * jnp.arange(bits.shape[-1], dtype=jnp.uint64) + 1
    return jnp.sum(bits * weight, axis=-1, dtype=jnp.uint64)


@lru_cache(maxsize=None)
def _record_digests(mesh, shape):
    """Segment carry ``[n_out, q, rows, cols]`` at ``P(None, None, 'x', 'y')`` → each local
    record's digest ``(px, py, n_out)``."""
    def local(carry):
        return _digest(carry.reshape(carry.shape[0], -1))[None, None]
    return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=P(None, None, "x", "y"),
                                 out_specs=P("x", "y", None), check_vma=False))


@lru_cache(maxsize=None)
def _piece(mesh, shape, length16, padded16):
    """``(carry, o, a)`` → elements ``[a, a + length16)`` of every rank's local record ``o``,
    zero-padded to ``padded16``, as ``(px, py, padded16)`` in pinned host memory."""
    def local(carry, o, a):
        flat = carry.reshape(carry.shape[0], -1)
        piece = jax.lax.dynamic_slice(flat, (o, a), (1, length16))[0]
        return jnp.pad(piece, (0, padded16 - length16))[None, None]
    body = jax.shard_map(local, mesh=mesh, in_specs=(P(None, None, "x", "y"), P(), P()),
                         out_specs=P("x", "y", None), check_vma=False)
    return jax.jit(body, out_shardings=NamedSharding(mesh, P("x", "y", None), memory_kind="pinned_host"))


@lru_cache(maxsize=None)
def _unpack(mesh, n_out, q, shapes, rects, tile, records16):
    """Records ``(px, py, n_out·S16)`` → the tile ``[n_out, q, px·tile_r, py·tile_c]`` at
    ``P(None, None, 'x', 'y')`` (zero outside every rectangle) and the record digests
    ``(px, py, n_out, n_segment)``."""
    starts = np.concatenate([[0], np.cumsum(records16)[:-1]]).astype(int)
    total = int(sum(records16))

    def local(flat):
        flat = flat.reshape(n_out, total)
        out = jnp.zeros((n_out, q) + tuple(tile), flat.dtype)
        digests = []
        for start, (rows, cols), places in zip(starts, shapes, rects):
            record = flat[:, start:start + q * rows * cols]
            digests.append(_digest(record))
            record = record.reshape(n_out, q, rows, cols)
            for r0, c0, R0, C0, nr, nc in places:
                out = out.at[:, :, R0:R0 + nr, C0:C0 + nc].set(record[:, :, r0:r0 + nr, c0:c0 + nc])
        return out, jnp.stack(digests, axis=-1)[None, None]
    body = jax.shard_map(local, mesh=mesh, in_specs=P("x", "y", None),
                         out_specs=(P(None, None, "x", "y"), P("x", "y", None, None)),
                         check_vma=False)
    return jax.jit(body)


@lru_cache(maxsize=1)
def _lustre_create():
    """``llapi_file_create`` from liblustreapi, or ``None`` when it is absent."""
    name = ctypes.util.find_library("lustreapi")
    if not name:
        return None
    try:
        create = ctypes.CDLL(name).llapi_file_create
    except (OSError, AttributeError):
        return None
    create.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    create.restype = ctypes.c_int
    return create


def _fallocate(fd, nbytes):
    """``fallocate(fd, 0, 0, nbytes)``: 0, or ``errno.EOPNOTSUPP`` (never emulated by writing), or -1."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
    if libc.fallocate(int(fd), 0, 0, int(nbytes)) == 0:
        return 0
    return errno.EOPNOTSUPP if ctypes.get_errno() in (errno.EOPNOTSUPP, errno.ENOSYS) else -1


def _aligned(nbytes):
    """An anonymous, pre-faulted, page-aligned host buffer."""
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_POPULATE", 0)
    return mmap.mmap(-1, max(int(nbytes), ALIGN), flags=flags)


class _Store:
    """One device's records: an O_DIRECT file or an aligned host buffer."""

    def __init__(self, path, nbytes, kind):
        self.path, self.kind, self.fd, self.host = Path(path), kind, None, None
        if kind == "host":
            self.host = _aligned(nbytes)
            self.view = np.frombuffer(self.host, dtype=np.uint8)
            return
        if os.path.lexists(self.path):
            os.unlink(self.path)
        create = _lustre_create()
        self.layout = "default"
        if create is not None:
            rc = create(str(self.path).encode(), STRIPE_BYTES, -1, STRIPES, 0)
            if rc == 0:
                self.layout = f"{STRIPES} x {STRIPE_BYTES >> 20} MiB"
            elif os.path.lexists(self.path):
                os.unlink(self.path)
        if not hasattr(os, "O_DIRECT"):
            raise OSError("GATE streamed_bank: this platform has no O_DIRECT")
        try:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600)
        except OSError as exc:
            raise OSError(f"GATE streamed_bank: cannot open {self.path} with O_DIRECT ({exc}); "
                          "the streamed bank needs a filesystem with direct I/O") from exc
        # Reserve every byte now, so a bank the disk (or the quota) cannot hold
        # refuses before any compute; a filesystem without fallocate reserves nothing.
        if _fallocate(self.fd, int(nbytes)) not in (0, errno.EOPNOTSUPP):
            err = ctypes.get_errno()
            os.close(self.fd)
            self.fd = None
            os.unlink(self.path)
            raise OSError(err, f"GATE streamed_bank_capacity: cannot reserve {int(nbytes)} bytes "
                          f"for {self.path}: {os.strerror(err)}")

    def write(self, source, offset):
        """``source`` an aligned memoryview whose length is a multiple of :data:`ALIGN`."""
        if self.kind == "host":
            np.copyto(self.view[offset:offset + len(source)], np.frombuffer(source, np.uint8))
            return
        done = 0
        while done < len(source):
            done += os.pwrite(self.fd, source[done:], offset + done)

    def read(self, target, offset):
        done = 0
        while done < len(target):
            got = os.preadv(self.fd, [target[done:]], offset + done)
            if got <= 0:
                raise OSError(f"GATE streamed_bank: short read of {self.path} at {offset + done}")
            done += got

    def sync(self):
        if self.fd is not None:
            os.fsync(self.fd)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            os.unlink(self.path)
        if self.host is not None:
            self.view = None
            self.host.close()
            self.host = None


class StreamedBank:
    """``n_out`` outputs of a ``[q, tile_r, tile_c]``-per-device operator, stored per device.

    ``segments`` are ``((rows, cols, rects), ...)``: each segment's local shape
    and the rectangles ``(r0, c0, R0, C0, nr, nc)`` that place it in the local
    ``tile`` (``gw.subtile_stream.segment_blocks``).  :meth:`put` takes one segment's carry for some outputs,
    :meth:`commit` makes every put durable, and :meth:`reader` returns the
    outputs back as ``[o1-o0, q, rows_X, cols_Y]`` arrays.  Every rank calls
    every method in the same order.
    """

    #: Segment carries a caller keeps alive on the devices (one computing, one draining).
    in_flight = IN_FLIGHT

    def __init__(self, mesh, *, root, label, kind, n_out, q, segments, tile):
        if kind not in ("host", "file"):
            raise ValueError(f"GATE streamed_bank: kind {kind!r}; want 'host' or 'file'")
        self.mesh, self.kind, self.label = mesh, kind, str(label)
        self.n_out, self.q = int(n_out), int(q)
        self.shapes = tuple((int(r), int(c)) for r, c, _ in segments)
        self.rects = tuple(tuple(tuple(int(v) for v in rect) for rect in places) for _, _, places in segments)
        self.tile = tuple(int(v) for v in tile)
        self.records = tuple(padded(16 * self.q * r * c) for r, c in self.shapes)
        self.starts = tuple(int(s) for s in np.concatenate([[0], np.cumsum(self.records)[:-1]]))
        self.S = int(sum(self.records))
        self.nbytes = self.n_out * self.S
        self.dir = Path(root) / "streamed_bank"
        cells = NamedSharding(mesh, P("x", "y")).addressable_devices_indices_map(
            (int(mesh.shape["x"]), int(mesh.shape["y"])))
        self.devices = tuple(cells)
        self._cell = {(index[0].start or 0, index[1].start or 0): d for d, index in cells.items()}
        self.seconds = dict(write=0., read=0., wait=0.)
        self.bounced = 0
        self.digests = {d: np.zeros((self.n_out, len(self.shapes)), np.uint64) for d in self.devices}
        self.written = np.zeros((self.n_out, len(self.shapes)), bool)
        self.stores, self._inflight, self._error = {}, deque(), None
        self._pool = ThreadPoolExecutor(IO_THREADS, thread_name_prefix="bank-io")
        self._drain = ThreadPoolExecutor(1, thread_name_prefix="bank-drain")
        error = None
        try:
            if kind == "file":
                os.makedirs(self.dir, exist_ok=True)
            for d in self.devices:
                self.stores[d] = _Store(self.dir / f"{self.label}.{d.id:05d}", self.nbytes, kind)
        except BaseException as exc:
            error = exc
        self.fits = True
        try:
            agree_io_error(error, path=self.dir, stage="streamed_bank.create")
        except RuntimeError:
            # Every rank leaves the same way: its own stores closed and deleted.
            # The caller decides (``fits``); the device sample groups need no disk.
            self._close_stores()
            self.fits = False

    def receipt(self):
        """Description of the tier: shapes (rank-identical) and this rank's file layout."""
        first = next(iter(self.stores.values()), None)
        return dict(tier=self.kind, bytes_per_rank=self.nbytes, outputs=self.n_out,
                    segments=len(self.shapes), record_bytes=list(self.records),
                    layout="output-major [o][segment] records of [q, rows, cols], 4 KiB padded",
                    lustre=None if first is None or self.kind == "host" else first.layout)

    # -- write -------------------------------------------------------------------------
    def put(self, p, carry, outputs):
        """Store segment ``p`` of ``carry`` ``[n, q, px·rows_p, py·cols_p]``: carry row ``r``
        becomes bank output ``o`` for every ``(r, o)`` in ``outputs``.  Returns at once;
        an I/O failure is held for :meth:`commit`, never raised here."""
        outputs = tuple((int(r), int(o)) for r, o in outputs)
        digest = _record_digests(self.mesh, tuple(carry.shape))(carry)
        while len(self._inflight) >= IN_FLIGHT:
            self._retire()
        self._inflight.append(self._drain.submit(self._write_pass, p, carry, digest, outputs))

    def _write_pass(self, p, carry, digest, outputs):
        """Drain one segment: its records' pieces, at most PIECES_IN_FLIGHT staged (drain thread)."""
        started = time.monotonic()
        record = self.records[p]
        length16 = self.q * self.shapes[p][0] * self.shapes[p][1]
        step16 = PIECE // 16
        cuts = [(a, min(step16, length16 - a)) for a in range(0, length16, step16)]
        shape = tuple(carry.shape)
        staged = deque()

        def land(staging):
            host, o, a = staging
            for shard in host.addressable_shards:
                source = np.asarray(shard.data).reshape(-1).view(np.uint8)
                if source.ctypes.data % ALIGN:
                    # The pinned pool hands out 4 KiB-aligned buffers (measured); a
                    # sub-allocated one is copied once into an aligned buffer.
                    bounce = _aligned(len(source))
                    np.copyto(np.frombuffer(bounce, np.uint8, count=len(source)), source)
                    source = np.frombuffer(bounce, np.uint8, count=len(source))
                    self.bounced += 1
                self.stores[shard.device].write(memoryview(source),
                                                o * self.S + self.starts[p] + 16 * a)

        def retire_one():
            staged.popleft().result()

        for row, o in outputs:
            for a, n16 in cuts:
                padded16 = n16 if n16 == step16 else (min(record, 16 * a + padded(16 * n16)) - 16 * a) // 16
                host = _piece(self.mesh, shape, n16, padded16)(carry, np.int64(row), np.int64(a))
                while len(staged) >= PIECES_IN_FLIGHT:
                    retire_one()
                staged.append(self._pool.submit(land, (host, o, a)))
                del host
        del carry
        while staged:
            retire_one()
        digests = {s.device: np.asarray(s.data)[0, 0] for s in digest.addressable_shards}
        for d in self.devices:
            for row, o in outputs:
                self.digests[d][o, p] = digests[d][row]
        for _, o in outputs:
            self.written[o, p] = True
        return time.monotonic() - started

    def _retire(self):
        started = time.monotonic()
        try:
            self.seconds["write"] += self._inflight.popleft().result()
        except BaseException as exc:
            self._error = self._error or exc
        self.seconds["wait"] += time.monotonic() - started

    def commit(self):
        """Every put durable and every pass of every put output written, agreed by all ranks."""
        while self._inflight:
            self._retire()
        error = self._error
        if error is None:
            try:
                for store in self.stores.values():
                    store.sync()
            except BaseException as exc:
                error = exc
        agree_io_error(error, path=self.dir, stage="streamed_bank.commit")

    # -- read --------------------------------------------------------------------------
    def reader(self, spans):
        """Iterator over ``spans`` ``[(o0, o1), ...]``: each output run as one device array
        ``[o1-o0, q, px·tile_r, py·tile_c]`` at ``P(None, None, 'x', 'y')``, read one ahead."""
        return _Reader(self, [(int(a), int(b)) for a, b in spans])

    def _load(self, o0, o1, staging):
        """The run ``[o0·S, o1·S)`` of every device as one ``(px, py, n/16)`` array at
        ``P('x', 'y', None)``, each device's row from its own store (the reader's thread)."""
        started = time.monotonic()
        n = (o1 - o0) * self.S
        local = {}
        for d in self.devices:
            store = self.stores[d]
            if self.kind == "host":
                local[d] = store.view[o0 * self.S:o1 * self.S]
            else:
                if staging.get(d) is None or len(staging[d]) < n:
                    staging[d] = _aligned(n)
                target = memoryview(staging[d])[:n]
                pieces = [self._pool.submit(store.read, target[a:min(n, a + PIECE)], o0 * self.S + a)
                          for a in range(0, n, PIECE)]
                for piece in pieces:
                    piece.result()
                local[d] = np.frombuffer(staging[d], dtype=np.uint8, count=n)
        shape = (int(self.mesh.shape["x"]), int(self.mesh.shape["y"]), n // 16)
        flat = jax.make_array_from_callback(
            shape, NamedSharding(self.mesh, P("x", "y", None)),
            lambda index: local[self._cell[(index[0].start or 0, index[1].start or 0)]]
            .view(np.complex128).reshape(1, 1, -1))
        flat.block_until_ready()
        return flat, time.monotonic() - started

    def release(self):
        """Close and delete every store; agreed by all ranks."""
        error = None
        try:
            while self._inflight:
                self._retire()
            self._pool.shutdown(wait=True)
            self._drain.shutdown(wait=True)
            self._close_stores()
        except BaseException as exc:
            error = exc
        agree_io_error(error, path=self.dir, stage="streamed_bank.release")

    def _close_stores(self):
        stores, self.stores = self.stores, {}
        for store in stores.values():
            store.close()


class _Reader:
    """One-ahead reader of a :class:`StreamedBank`'s output runs (two staging slots)."""

    def __init__(self, bank, spans):
        self.bank, self.spans, self.at = bank, list(spans), 0
        self._staging = ({}, {})
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="bank-read")
        self._next = self._submit(0)

    def _submit(self, i):
        if i >= len(self.spans):
            return None
        return self._thread.submit(self.bank._load, *self.spans[i], self._staging[i % 2])

    def __iter__(self):
        return self

    def __next__(self):
        bank = self.bank
        if self.at >= len(self.spans):
            self._thread.shutdown(wait=True)
            raise StopIteration
        o0, o1 = self.spans[self.at]
        error, value = None, None
        started = time.monotonic()
        try:
            flat, seconds = self._next.result()
            bank.seconds["read"] += seconds
        except BaseException as exc:
            error, flat = exc, None
        self.at += 1
        self._next = self._submit(self.at)
        bank.seconds["wait"] += time.monotonic() - started
        agree_io_error(error, path=bank.dir, stage="streamed_bank.read")
        value, digest = _unpack(bank.mesh, o1 - o0, bank.q, bank.shapes, bank.rects, bank.tile,
                                tuple(r // 16 for r in bank.records))(flat)
        del flat
        try:
            for shard in digest.addressable_shards:
                if not bank.written[o0:o1].all():
                    raise OSError(f"GATE streamed_bank: outputs [{o0}, {o1}) were not all written")
                if not np.array_equal(np.asarray(shard.data)[0, 0], bank.digests[shard.device][o0:o1]):
                    raise OSError(f"GATE streamed_bank: digest mismatch reading outputs [{o0}, {o1}) "
                                  f"of {bank.stores[shard.device].path}")
        except BaseException as exc:
            error = exc
        agree_io_error(error, path=bank.dir, stage="streamed_bank.digest")
        return value
