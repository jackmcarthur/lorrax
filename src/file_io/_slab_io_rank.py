"""SlabIO's per-rank streamed tier: a sharded operator written segment by segment, read back by output.

Each device keeps its own (μ_X, ν_Y) tile in one private store: a file
(``kind="file"``) or a host buffer (``kind="host"``).  No rank reads
another's, so the tier issues no collective; create, commit, read and
release end in :func:`common.collectives.agree_io_error`, which every rank
calls at the same point (INVARIANTS 21).

Layout per device: a record is segment ``s``'s ``[q, rows_s, cols_s]`` tile
of output ``o``, padded to 4 KiB, at ``o·S + start_s`` (``S = Σ_s
padded_s``), so outputs ``[o0, o1)`` are one contiguous run.  On read, each
segment's rectangles go back to their places in the local tile.

Write: a drain thread moves a finished segment's records off the device in
64 MiB pieces through JAX host arrays (pinned host memory where the platform
has it), at most :data:`PIECES_IN_FLIGHT` staged per rank, and I/O threads
write them while the devices compute the next segment.  Read: one run ahead
of the consumer.  Every record carries a digest taken on the device at write
and checked on the device at read.

The service chooses, once per file and with no setting: O_DIRECT where the
platform and filesystem accept it (aligned pieces; else plain buffered I/O),
a 4 x 4 MiB stripe layout where ``lfs`` exists, and a reservation of every
byte where ``fallocate`` exists, so a bank the disk or quota cannot hold is
refused before any compute (:attr:`StreamedBank.fits`).
"""
from __future__ import annotations

import ctypes
import errno
import mmap
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import agree_io_error, all_gather_processes

# PIECE and IO_THREADS were measured on Perlmutter Lustre (4 x 4 MiB stripes):
# 64 MiB calls from four threads reach 13-14 GB/s per rank; unpacked ~53 MB calls
# from one thread drop to about 3 GB/s.
PIECE = 64 << 20        # bytes per write/read call
IO_THREADS = 4
IN_FLIGHT = 2           # segment carries alive (one computing, one draining)
PIECES_IN_FLIGHT = 4    # pieces staged per rank before the next is moved


def padded(nbytes: int, align: int) -> int:
    return -(-int(nbytes) // int(align)) * int(align)


def _alignment(directory):
    """Direct-I/O alignment here: the larger of the page size and the filesystem block."""
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    try:
        return max(int(page), int(os.statvfs(directory).f_bsize))
    except OSError:
        return int(page)


def _digest(flat):
    """Σ_i bits_i·(2i+1) mod 2⁶⁴ over each row of a complex128 ``[..., m]``."""
    bits = jax.lax.bitcast_convert_type(jnp.stack([flat.real, flat.imag], axis=-1), jnp.uint64)
    bits = bits.reshape(flat.shape[:-1] + (-1,))
    return jnp.sum(bits * (2 * jnp.arange(bits.shape[-1], dtype=jnp.uint64) + 1), axis=-1,
                   dtype=jnp.uint64)


def _host_kind(mesh):
    """Pinned host memory for an accelerator's pieces; a CPU device's own memory is host memory."""
    device = mesh.devices.flat[0]
    kinds = {m.kind for m in device.addressable_memories()}
    return "pinned_host" if device.platform != "cpu" and "pinned_host" in kinds else None


@lru_cache(maxsize=None)
def _record_digests(mesh, shape):
    """Carry ``[n_out, q, rows, cols]`` at ``P(None, None, 'x', 'y')`` → digests ``(px, py, n_out)``."""
    return jax.jit(jax.shard_map(lambda c: _digest(c.reshape(c.shape[0], -1))[None, None],
                                 mesh=mesh, in_specs=P(None, None, "x", "y"),
                                 out_specs=P("x", "y", None), check_vma=False))


@lru_cache(maxsize=None)
def _piece(mesh, shape, length16, padded16):
    """``(carry, o, a)`` → each rank's local record ``o`` elements ``[a, a + length16)``,
    zero-padded to ``padded16``, as ``(px, py, padded16)`` in host memory."""
    def local(carry, o, a):
        piece = jax.lax.dynamic_slice(carry.reshape(carry.shape[0], -1), (o, a), (1, length16))[0]
        return jnp.pad(piece, (0, padded16 - length16))[None, None]
    body = jax.shard_map(local, mesh=mesh, in_specs=(P(None, None, "x", "y"), P(), P()),
                         out_specs=P("x", "y", None), check_vma=False)
    return jax.jit(body, out_shardings=NamedSharding(mesh, P("x", "y", None), memory_kind=_host_kind(mesh)))


@lru_cache(maxsize=None)
def _record_run(mesh, shape, rows, padded16):
    """``(carry, r)`` → carry rows ``[r, r + rows)``, each zero-padded to ``padded16``, as one
    contiguous ``(px, py, rows·padded16)`` run in host memory (consecutive small records)."""
    def local(carry, r):
        flat = carry.reshape(carry.shape[0], -1)
        run = jax.lax.dynamic_slice_in_dim(flat, r, rows, axis=0)
        return jnp.pad(run, ((0, 0), (0, padded16 - run.shape[1]))).reshape(-1)[None, None]
    body = jax.shard_map(local, mesh=mesh, in_specs=(P(None, None, "x", "y"), P()),
                         out_specs=P("x", "y", None), check_vma=False)
    return jax.jit(body, out_shardings=NamedSharding(mesh, P("x", "y", None), memory_kind=_host_kind(mesh)))


@lru_cache(maxsize=None)
def _unpack(mesh, n_out, q, shapes, rects, tile, records16):
    """Records ``(px, py, n_out·S/16)`` → the tile ``[n_out, q, px·tile_r, py·tile_c]``
    (zeros outside every rectangle) and the digests ``(px, py, n_out, n_segment)``."""
    starts = np.concatenate([[0], np.cumsum(records16)[:-1]]).astype(int)

    def local(flat):
        flat = flat.reshape(n_out, int(sum(records16)))
        out, digests = jnp.zeros((n_out, q) + tuple(tile), flat.dtype), []
        for start, (rows, cols), places in zip(starts, shapes, rects):
            record = flat[:, start:start + q * rows * cols]
            digests.append(_digest(record))
            record = record.reshape(n_out, q, rows, cols)
            for r0, c0, R0, C0, nr, nc in places:
                out = out.at[:, :, R0:R0 + nr, C0:C0 + nc].set(record[:, :, r0:r0 + nr, c0:c0 + nc])
        return out, jnp.stack(digests, axis=-1)[None, None]
    return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=P("x", "y", None),
                                 out_specs=(P(None, None, "x", "y"), P("x", "y", None, None)),
                                 check_vma=False))


def _aligned(nbytes):
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_POPULATE", 0)
    return mmap.mmap(-1, max(int(nbytes), mmap.PAGESIZE), flags=flags)


def _reserve(fd, nbytes):
    """Allocate every byte where ``fallocate`` exists (never emulated by writing)."""
    try:
        fallocate = ctypes.CDLL(None, use_errno=True).fallocate
    except (OSError, AttributeError):
        return
    fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
    if fallocate(int(fd), 0, 0, int(nbytes)) != 0:
        err = ctypes.get_errno()
        if err not in (errno.EOPNOTSUPP, errno.ENOSYS):
            raise OSError(err, f"GATE streamed_bank_capacity: cannot reserve {nbytes} bytes: "
                               f"{os.strerror(err)}")


class _Store:
    """One device's records: a file (direct I/O where accepted) or a host buffer.

    A direct-I/O call the filesystem rejects (EINVAL) moves this file to plain
    buffered I/O for every later call; the bytes and offsets are unchanged.
    """

    def __init__(self, path, nbytes, kind):
        self.path, self.fd, self.host, self.direct = Path(path), None, None, False
        self._lock = threading.Lock()
        if kind == "host":
            self.host = _aligned(nbytes)
            self.view = np.frombuffer(self.host, dtype=np.uint8)
            return
        if os.path.lexists(self.path):
            os.unlink(self.path)
        if shutil.which("lfs"):
            done = subprocess.run(["lfs", "setstripe", "-c", "4", "-S", "4M", str(self.path)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
            if done.returncode:
                print(f"[slab_io] lfs setstripe failed for {self.path} (rc {done.returncode}): "
                      f"{done.stderr.decode(errors='replace').strip()[:200]}; default layout", flush=True)
        flags = os.O_RDWR | os.O_CREAT
        try:
            self.fd = os.open(self.path, flags | os.O_DIRECT, 0o600)
            self.direct = True
        except (AttributeError, OSError):
            self.fd = os.open(self.path, flags, 0o600)
        try:
            _reserve(self.fd, nbytes)
        except OSError:
            self.close()
            raise

    def _buffered(self, fd):
        """After a rejected direct call on ``fd``: the buffered descriptor every later call uses."""
        with self._lock:
            if self.direct and fd == self.fd:
                os.fsync(self.fd)
                buffered = os.open(self.path, os.O_RDWR)
                os.close(self.fd)
                self.fd, self.direct = buffered, False
            return self.fd

    def _call(self, op):
        fd = self.fd
        try:
            return op(fd)
        except OSError as exc:
            if exc.errno != errno.EINVAL or not self.direct:
                raise
            return op(self._buffered(fd))

    def write(self, source, offset):
        if self.host is not None:
            np.copyto(self.view[offset:offset + len(source)], np.frombuffer(source, np.uint8))
            return
        done = 0
        while done < len(source):
            done += self._call(lambda fd: os.pwrite(fd, source[done:], offset + done))

    def _read_once(self, fd, target, offset):
        if hasattr(os, "preadv"):
            return os.preadv(fd, [target], offset)
        data = os.pread(fd, len(target), offset)
        target[:len(data)] = data
        return len(data)

    def read(self, target, offset):
        done = 0
        while done < len(target):
            got = self._call(lambda fd: self._read_once(fd, target[done:], offset + done))
            if got <= 0:
                raise OSError(f"GATE streamed_bank: short read of {self.path} at {offset + done}")
            done += got

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
    """``n_out`` outputs of a ``[q, tile_r, tile_c]``-per-device operator, one store per device.

    ``segments`` ``((rows, cols, rects), ...)``: each segment's local shape and the
    rectangles ``(r0, c0, R0, C0, nr, nc)`` placing it in the local ``tile``
    (``gw.subtile_stream.segment_blocks``).  :meth:`put` takes one segment's carry,
    :meth:`commit` makes the puts durable, :meth:`reader` gives outputs back as
    ``[o1-o0, q, px·tile_r, py·tile_c]`` arrays.  Every rank calls every method in
    the same order.  ``fits`` is False (on every rank) when a store could not be
    created or reserved; nothing is then held.
    """

    in_flight = IN_FLIGHT

    def __init__(self, mesh, *, root, label, kind, n_out, q, segments, tile, unwritten_zero=False):
        self.mesh, self.kind, self.n_out, self.q = mesh, kind, int(n_out), int(q)
        self.shapes = tuple((int(r), int(c)) for r, c, _ in segments)
        self.rects = tuple(tuple(tuple(int(v) for v in r) for r in places) for _, _, places in segments)
        self.tile = tuple(int(v) for v in tile)
        self.dir = Path(root or ".") / "streamed_bank"
        # A store that is read before every record is written (the W bank) reads
        # unwritten records as the zeros the reservation holds (digest 0).
        self.unwritten_zero = bool(unwritten_zero)
        # One alignment on every rank (the largest page or filesystem block), so the
        # record offsets are the same everywhere.
        local = _alignment(root) if kind == "file" else mmap.PAGESIZE
        self.align = int(np.max(np.asarray(all_gather_processes(np.asarray(local, np.int64)))))
        self.piece = padded(PIECE, self.align)
        self.records = tuple(padded(16 * self.q * r * c, self.align) for r, c in self.shapes)
        self.starts = tuple(int(s) for s in np.cumsum((0,) + self.records[:-1]))
        self.S = int(sum(self.records))
        self.nbytes = self.n_out * self.S
        cells = NamedSharding(mesh, P("x", "y")).addressable_devices_indices_map(
            (int(mesh.shape["x"]), int(mesh.shape["y"])))
        self.devices = tuple(cells)
        self._cell = {(i[0].start or 0, i[1].start or 0): d for d, i in cells.items()}
        self.seconds, self.bounced = dict(write=0., read=0., wait=0.), 0
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
                self.stores[d] = _Store(self.dir / f"{label}.{d.id:05d}", self.nbytes, kind)
        except BaseException as exc:
            error = exc
        try:
            agree_io_error(error, path=self.dir, stage="streamed_bank.create")
            self.fits = True
        except RuntimeError:
            self._close_stores()
            self.fits = False

    def receipt(self):
        return dict(tier=self.kind, bytes_per_rank=self.nbytes, outputs=self.n_out,
                    segments=len(self.shapes), record_bytes=list(self.records),
                    direct_io=all(s.direct for s in self.stores.values()))

    def put(self, p, carry, outputs):
        """Store segment ``p``: carry row ``r`` as output ``o`` for each ``(r, o)``.  Returns at
        once; a failure is held for :meth:`commit`."""
        outputs = tuple((int(r), int(o)) for r, o in outputs)
        digest = _record_digests(self.mesh, tuple(carry.shape))(carry)
        while len(self._inflight) >= IN_FLIGHT:
            self._retire()
        self._inflight.append(self._drain.submit(self._drain_segment, p, carry, digest, outputs))

    def _drain_segment(self, p, carry, digest, outputs):
        started = time.monotonic()
        length16 = self.q * self.shapes[p][0] * self.shapes[p][1]
        step16, staged = self.piece // 16, deque()

        def stage(host, o, a):
            while len(staged) >= PIECES_IN_FLIGHT:
                staged.popleft().result()
            staged.append(self._pool.submit(land, host, o, a))

        def land(host, o, a):
            for shard in host.addressable_shards:
                store = self.stores[shard.device]
                source = np.asarray(shard.data).reshape(-1).view(np.uint8)
                if store.direct and source.ctypes.data % self.align:
                    aligned = _aligned(len(source))       # a sub-allocated host buffer
                    np.copyto(np.frombuffer(aligned, np.uint8, count=len(source)), source)
                    source = np.frombuffer(aligned, np.uint8, count=len(source))
                    self.bounced += 1
                store.write(memoryview(source), o * self.S + self.starts[p] + 16 * a)

        record16 = self.records[p] // 16
        if len(self.shapes) == 1 and 16 * record16 <= self.piece // 2:
            # Small records of a one-segment store: consecutive outputs are
            # consecutive in the file, so runs of them move as one piece.
            per, runs, i = max(1, self.piece // (16 * record16)), [], 0
            while i < len(outputs):
                j = i + 1
                while (j < len(outputs) and j - i < per and outputs[j][0] == outputs[j - 1][0] + 1
                       and outputs[j][1] == outputs[j - 1][1] + 1):
                    j += 1
                runs.append((outputs[i][0], outputs[i][1], j - i))
                i = j
            for row, o, m in runs:
                stage(_record_run(self.mesh, tuple(carry.shape), m, record16)(carry, np.int64(row)), o, 0)
        else:
            for row, o in outputs:
                for a in range(0, length16, step16):
                    n16 = min(step16, length16 - a)
                    program = _piece(self.mesh, tuple(carry.shape), n16, padded(16 * n16, self.align) // 16)
                    stage(program(carry, np.int64(row), np.int64(a)), o, a)
        del carry
        while staged:
            staged.popleft().result()
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
        while self._inflight:
            self._retire()
        error = self._error
        try:
            for store in self.stores.values():
                if store.fd is not None:
                    os.fsync(store.fd)
        except BaseException as exc:
            error = error or exc
        agree_io_error(error, path=self.dir, stage="streamed_bank.commit")

    def reader(self, spans):
        """Each ``(o0, o1)`` of ``spans`` in turn as one device array, read one ahead."""
        return _Reader(self, [(int(a), int(b)) for a, b in spans])

    def _load(self, o0, o1, staging):
        """Run ``[o0·S, o1·S)`` of every local store as ``(px, py, n/16)`` at ``P('x','y',None)``."""
        started = time.monotonic()
        n, local = (o1 - o0) * self.S, {}
        for d in self.devices:
            store = self.stores[d]
            if store.host is not None:
                local[d] = store.view[o0 * self.S:o1 * self.S]
                continue
            if staging.get(d) is None or len(staging[d]) < n:
                staging[d] = _aligned(n)
            target = memoryview(staging[d])[:n]
            for piece in [self._pool.submit(store.read, target[a:a + self.piece], o0 * self.S + a)
                          for a in range(0, n, self.piece)]:
                piece.result()
            local[d] = np.frombuffer(staging[d], dtype=np.uint8, count=n)
        flat = jax.make_array_from_callback(
            (int(self.mesh.shape["x"]), int(self.mesh.shape["y"]), n // 16),
            NamedSharding(self.mesh, P("x", "y", None)),
            lambda i: local[self._cell[(i[0].start or 0, i[1].start or 0)]].view(np.complex128)[None, None])
        flat.block_until_ready()
        return flat, time.monotonic() - started

    def release(self):
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
        self.bank, self.spans, self.at, self._staging = bank, spans, 0, ({}, {})
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="bank-read")
        self._next = self._submit(0)

    def _submit(self, i):
        if i < len(self.spans):
            return self._thread.submit(self.bank._load, *self.spans[i], self._staging[i % 2])

    def __iter__(self):
        return self

    def __next__(self):
        bank = self.bank
        if self.at >= len(self.spans):
            self._thread.shutdown(wait=True)
            raise StopIteration
        (o0, o1), error, started = self.spans[self.at], None, time.monotonic()
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
        try:
            if not bank.unwritten_zero and not bank.written[o0:o1].all():
                raise OSError(f"GATE streamed_bank: outputs [{o0}, {o1}) were not all written")
            for shard in digest.addressable_shards:
                if not np.array_equal(np.asarray(shard.data)[0, 0], bank.digests[shard.device][o0:o1]):
                    raise OSError(f"GATE streamed_bank: digest mismatch reading outputs [{o0}, {o1})")
        except BaseException as exc:
            error = exc
        agree_io_error(error, path=bank.dir, stage="streamed_bank.digest")
        return value
