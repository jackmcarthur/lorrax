"""The streamed bank tier: a sharded operator written row pass by row pass and read back output by output.

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

**Layout (per device).**  A record is the pass ``p`` tile ``[q, rows_p,
cols]`` of output ``o``, ``16·q·rows_p·cols`` bytes padded to 4 KiB.  Records
are output-major, at ``o·S + start_p`` with ``S = Σ_p padded_p``, so one
consumer's read of outputs ``[o0, o1)`` is the one contiguous run
``[o0·S, o1·S)``.

**Write.**  One program per pass shape packs the pass carry into padded
records and lands them in pinned host memory (4 KiB aligned, so the numpy
view is the O_DIRECT source with no copy), with one position-weighted digest
per record.  A drain thread hands 64 MiB O_DIRECT pieces to a pool of I/O
threads while the devices compute the next pass; at most :data:`IN_FLIGHT`
passes are in flight.

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
#: Passes whose records may be in flight to the store at once.
IN_FLIGHT = 2
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
def _pack(mesh, shape, record16):
    """Pass carry ``[n_out, q, rows, cols]`` at ``P(None, None, 'x', 'y')`` → its padded
    records ``(px, py, n_out, record16)`` in pinned host memory and their digests."""
    def local(carry):
        flat = carry.reshape(carry.shape[0], -1)
        digest = _digest(flat)
        flat = jnp.pad(flat, ((0, 0), (0, record16 - flat.shape[1])))
        return flat[None, None], digest[None, None]
    body = jax.shard_map(local, mesh=mesh, in_specs=P(None, None, "x", "y"),
                         out_specs=(P("x", "y", None, None), P("x", "y", None)), check_vma=False)
    return jax.jit(body, out_shardings=(
        NamedSharding(mesh, P("x", "y", None, None), memory_kind="pinned_host"),
        NamedSharding(mesh, P("x", "y", None))))


@lru_cache(maxsize=None)
def _unpack(mesh, n_out, q, rows, cols, records16):
    """Records ``(px, py, n_out·S16)`` → ``[n_out, q, px·Σrows, py·cols]`` at
    ``P(None, None, 'x', 'y')`` and the record digests ``(px, py, n_out, n_pass)``."""
    starts = np.concatenate([[0], np.cumsum(records16)[:-1]]).astype(int)
    total = int(sum(records16))

    def local(flat):
        flat = flat.reshape(n_out, total)
        parts, digests = [], []
        for start, xr in zip(starts, rows):
            record = flat[:, start:start + q * xr * cols]
            digests.append(_digest(record))
            parts.append(record.reshape(n_out, q, xr, cols))
        return jnp.concatenate(parts, axis=2), jnp.stack(digests, axis=-1)[None, None]
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
    """``n_out`` outputs of a ``[q, rows, cols]``-per-device operator, stored per device.

    ``passes`` are the row passes ``((x0, xr), ...)`` of every X shard's local
    rows (``gw.subtile_stream.PassPlan.passes``) and ``cols`` the local
    columns.  :meth:`put` takes one pass's carry for some outputs,
    :meth:`commit` makes every put durable, and :meth:`reader` returns the
    outputs back as ``[o1-o0, q, rows_X, cols_Y]`` arrays.  Every rank calls
    every method in the same order.
    """

    def __init__(self, mesh, *, root, label, kind, n_out, q, passes, cols):
        if kind not in ("host", "file"):
            raise ValueError(f"GATE streamed_bank: kind {kind!r}; want 'host' or 'file'")
        self.mesh, self.kind, self.label = mesh, kind, str(label)
        self.n_out, self.q, self.cols = int(n_out), int(q), int(cols)
        self.passes = tuple((int(x0), int(xr)) for x0, xr in passes)
        self.rows = tuple(xr for _, xr in self.passes)
        self.records = tuple(padded(16 * self.q * xr * self.cols) for xr in self.rows)
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
        self.digests = {d: np.zeros((self.n_out, len(self.passes)), np.uint64) for d in self.devices}
        self.written = np.zeros((self.n_out, len(self.passes)), bool)
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
        agree_io_error(error, path=self.dir, stage="streamed_bank.create")

    def receipt(self):
        """Rank-identical description of the tier (a function of the shapes only)."""
        first = next(iter(self.stores.values()), None)
        return dict(tier=self.kind, bytes_per_rank=self.nbytes, outputs=self.n_out,
                    passes=len(self.passes), record_bytes=list(self.records),
                    layout="output-major [o][pass] records of [q, rows, cols], 4 KiB padded",
                    lustre=None if first is None or self.kind == "host" else first.layout)

    # -- write -------------------------------------------------------------------------
    def put(self, p, carry, outputs):
        """Store pass ``p`` of ``carry`` ``[n, q, px·rows_p, py·cols]``: carry row ``r``
        becomes bank output ``o`` for every ``(r, o)`` in ``outputs``.  Returns at once;
        an I/O failure is held for :meth:`commit`, never raised here."""
        outputs = tuple((int(r), int(o)) for r, o in outputs)
        host, digest = _pack(self.mesh, tuple(carry.shape), self.records[p] // 16)(carry)
        while len(self._inflight) >= IN_FLIGHT:
            self._retire()
        self._inflight.append(self._drain.submit(self._write_pass, p, host, digest, outputs))

    def _write_pass(self, p, host, digest, outputs):
        started = time.monotonic()
        record = self.records[p]
        digests = {s.device: np.asarray(s.data)[0, 0] for s in digest.addressable_shards}
        pieces, bounce = [], []
        for shard in host.addressable_shards:
            store = self.stores[shard.device]
            source = np.asarray(shard.data).reshape(-1).view(np.uint8)
            for row, o in outputs:
                self.digests[shard.device][o, p] = digests[shard.device][row]
                part = source[row * record:(row + 1) * record]
                if part.ctypes.data % ALIGN:
                    # The pinned pool hands out 4 KiB-aligned buffers (measured); a
                    # sub-allocated one is copied once into an aligned buffer.
                    bounce.append(_aligned(record))
                    np.copyto(np.frombuffer(bounce[-1], np.uint8, count=record), part)
                    part = np.frombuffer(bounce[-1], np.uint8, count=record)
                    self.bounced += 1
                part = memoryview(part)
                for a in range(0, record, PIECE):
                    pieces.append(self._pool.submit(store.write, part[a:min(record, a + PIECE)],
                                                    o * self.S + self.starts[p] + a))
        for piece in pieces:
            piece.result()
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
        ``[o1-o0, q, px·Σrows, py·cols]`` at ``P(None, None, 'x', 'y')``, read one ahead."""
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
            for store in self.stores.values():
                store.close()
        except BaseException as exc:
            error = exc
        self.stores = {}
        agree_io_error(error, path=self.dir, stage="streamed_bank.release")


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
        value, digest = _unpack(bank.mesh, o1 - o0, bank.q, bank.rows, bank.cols,
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
