"""Checkpoints of sharded cubes plus a small host record, read back by authenticated identity.

Two owners use it. ``gw.sigma_dispatch`` keeps a one-shot shared-pole Σ sweep
(over an hour on large decks) between the τ sweep and finalize: a rerun with
``restart = true`` whose state authenticates goes straight to finalize. The SC
loop (``gw.sc_iteration``) keeps its Anderson window and carry beside each
warm seed, ``sc_seed/sc_checkpoint.h5`` (``SC_SCHEMA``), so a fresh process
continues the trajectory (docs/self_consistency.md §8).

The cubes go through SlabIO from their own shards: no gather, no copy. The
host record (JSON) and, for the SC loop, one ``state`` blob are written by
rank 0 after the cubes, and the ``commit`` attribute, the record's digest, is
written last; the file is built as a private sibling and published by
``os.replace``, so a kill mid-write leaves the previous checkpoint. The blob
is a pickle of plain Python and numpy containers only (dict, list, tuple,
str, numbers, None, ndarray); :func:`_plain_loads` refuses any other class,
and the writer round-trips it through that loader before publishing.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import time
from pathlib import Path

import numpy as np

SCHEMA = "sigma-sweep-checkpoint-v1"
SC_SCHEMA = "sc-anderson-checkpoint-v1"
#: Staging files older than this process's first checkpoint use are orphans.
_STARTED = time.time()
CUBES = ("body", "unextrap", "odd")
#: Measured SlabIO write rate of a swept cube, one node at P4 (Na 8^3, 61.4 GB
#: in 47.7 s); more ranks write faster, so this predicts long.
WRITE_BYTES_PER_S = 1.3e9
#: The checkpoint is written only when the sweep it protects took at least
#: this many times its predicted write.
PAYOFF = 5.0


def checkpoint_pays(cubes, sweep_seconds):
    """Return (write?, predicted write seconds) for these cubes."""
    nbytes = sum(int(np.prod(c.shape)) * np.dtype(c.dtype).itemsize
                 for c in cubes.values() if c is not None)
    predicted = nbytes / WRITE_BYTES_PER_S
    return sweep_seconds >= PAYOFF * predicted, predicted


def _encode(value):
    # Floats go through repr, which round-trips float64 exactly.
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"__nd__": value.real.tolist(), "__im__": value.imag.tolist(),
                    "dtype": str(value.dtype), "shape": list(value.shape)}
        return {"__nd__": value.tolist(), "dtype": str(value.dtype),
                "shape": list(value.shape)}
    if isinstance(value, (complex, np.complexfloating)):
        return {"__c__": [float(value.real), float(value.imag)]}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _decode(value):
    if isinstance(value, dict):
        if "__nd__" in value:
            real = np.asarray(value["__nd__"])
            if "__im__" in value:
                real = real + 1j * np.asarray(value["__im__"])
            return real.astype(value["dtype"]).reshape(value["shape"])
        if "__c__" in value:
            return complex(*value["__c__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _json(value):
    return json.dumps(value, default=_encode, sort_keys=True, allow_nan=False)


def sweep_identity(**parts):
    """One digest of everything the swept state depends on.

    Arrays enter by dtype, shape and bytes; everything else by canonical JSON.
    """
    digest = hashlib.sha256()
    for key in sorted(parts):
        value = parts[key]
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            digest.update(f"{value.dtype}{value.shape}".encode())
            digest.update(value.tobytes())
        else:
            # Identity only: an option without a JSON form (an enum) enters by repr.
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False, default=lambda v: (
                _encode(v) if isinstance(v, (np.ndarray, np.generic, complex)) else repr(v))
            ).encode())
    return digest.hexdigest()


def _spec(cube):
    from jax.sharding import NamedSharding
    sharding = getattr(cube, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return None
    return [list(a) if isinstance(a, tuple) else a for a in sharding.spec]


def checkpointable(cubes):
    """True when every present cube is a NamedSharding jax array."""
    return all(cube is None or _spec(cube) is not None for cube in cubes.values())


#: The classes a pickle of plain containers and numpy arrays names (numpy 1
#: and 2 module paths); :class:`_PlainLoader` refuses every other one.
_PLAIN_CLASSES = frozenset({
    ("builtins", "complex"), ("numpy", "dtype"), ("numpy", "ndarray"),
    *((f"numpy.{core}.{mod}", name) for core in ("core", "_core")
      for mod, name in (("multiarray", "_reconstruct"), ("multiarray", "scalar"),
                        ("numeric", "_frombuffer")))})


class _PlainLoader(pickle.Unpickler):
    """Unpickle plain containers and numpy arrays; refuse every other class."""

    def find_class(self, module, name):
        if (module, name) in _PLAIN_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"checkpoint state holds {module}.{name}; only plain containers "
            "and numpy arrays are stored")


def _plain_loads(data):
    return _PlainLoader(io.BytesIO(data)).load()


def _cubes_sha256(f, names):
    """One digest of the stored cube bytes, read slice by slice (mesh-free); each
    slice is hashed through its buffer, with no second host copy."""
    digest = hashlib.sha256()
    for name in sorted(names):
        ds = f[name]
        digest.update(f"{name}{ds.dtype}{ds.shape}".encode())
        for i in range(ds.shape[0]):
            digest.update(np.ascontiguousarray(ds[i]))
    return digest.hexdigest()


def write_sigma_checkpoint(path, *, identity, cubes, mesh, host=None,
                           schema=SCHEMA, state=None):
    """Write the cubes, host record and optional ``state`` collectively.

    Every rank writes its cube shards; rank 0 then adds the record (cube and
    state digests) and the commit, and the file is published atomically
    (``collective_atomic_file_transaction``). ``state`` is pickled on rank 0
    (plain containers only, round-tripped through :func:`_plain_loads`).
    The cube digest is taken from the closed staging file, not from memory:
    no rank holds a whole cube, and it binds the bytes that landed on disk
    (INVARIANTS 26). Staging files a dead process left beside ``path`` are
    removed. Returns (bytes, seconds).
    """
    import h5py
    from common.collectives import collective_atomic_file_transaction, process_rank
    from .slab_io import SlabIO

    started = time.monotonic()
    present = {name: cube for name, cube in cubes.items() if cube is not None}
    sizes = [0]

    def write(staging):
        with SlabIO(str(staging), mode="w", mesh=mesh) as io:
            for name, cube in present.items():
                io.create_dataset(name, shape=tuple(cube.shape), dtype=cube.dtype)
                io.write_slab(name, cube)
        if process_rank() != 0:
            return
        for orphan in Path(staging).parent.glob(f".{Path(path).name}.*.tmp"):
            if orphan != Path(staging) and orphan.stat().st_mtime < _STARTED:
                orphan.unlink(missing_ok=True)
        blob = None
        if state is not None:
            blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            _plain_loads(blob)
            sizes[0] = len(blob)
        with h5py.File(staging, "a") as f:
            record = dict(
                schema=schema, identity=identity,
                cubes={name: dict(shape=list(cube.shape), dtype=str(cube.dtype),
                                  spec=_spec(cube)) for name, cube in present.items()},
                host=host, cubes_sha256=_cubes_sha256(f, present),
                state_sha256=None if blob is None else hashlib.sha256(blob).hexdigest())
            text = _json(record)
            if blob is not None:
                f.create_dataset("state", data=np.frombuffer(blob, np.uint8))
            f.create_dataset("record", data=np.bytes_(text))
            f.attrs["commit"] = hashlib.sha256(text.encode()).hexdigest()

    def validate(staging):
        with h5py.File(staging, "r") as f:
            text = f["record"][()]
            if hashlib.sha256(bytes(text)).hexdigest() != str(f.attrs["commit"]):
                raise ValueError("checkpoint record does not match its commit")

    collective_atomic_file_transaction(path, stage="checkpoint.write", write=write,
                                       validate_file=validate)
    nbytes = sum(int(np.prod(c.shape)) * np.dtype(c.dtype).itemsize for c in present.values())
    return nbytes + sizes[0], time.monotonic() - started


def discard_sigma_checkpoint(path):
    """Remove a checkpoint (collective; rank 0 unlinks)."""
    from common.collectives import rank0_transaction
    path = Path(path)
    rank0_transaction(path, stage="sigma_checkpoint.discard",
                      write=lambda: path.unlink(missing_ok=True))


def read_sigma_checkpoint(path, *, identity, mesh, print_fn=print,
                          schema=SCHEMA, discard=True, cubes=True):
    """Return ``(cubes, host, state, commit)`` for an authenticated
    checkpoint, else ``(None, reason)``.

    Rank 0 checks the commit, the cube digest and the identity, and returns
    the commit; every rank then checks that the record it reads carries that
    commit. A partial or foreign file is removed (``discard``) so the caller
    recomputes; with ``discard=False`` it is left in place and the reason
    names the identity fields that differ. ``cubes=False`` defers the cubes
    to :func:`read_checkpoint_cubes`.
    """
    import h5py
    from common.collectives import agree_io_error, rank0_transaction

    path = Path(path)

    def check():
        if not path.exists():
            return ["absent", None]
        try:
            with h5py.File(path, "r") as f:
                text = bytes(f["record"][()])
                commit = str(f.attrs["commit"])
                record = json.loads(text)
                got = record.get("identity")
                if (hashlib.sha256(text).hexdigest() != commit
                        or record.get("schema") != schema):
                    reason = "partial or other schema"
                elif got != identity:
                    reason = "identity differs" + (": " + ", ".join(sorted(
                        k for k in set(got) | set(identity) if got.get(k) != identity.get(k)))
                        if isinstance(got, dict) and isinstance(identity, dict) else "")
                elif record.get("cubes_sha256") != _cubes_sha256(f, record["cubes"]):
                    reason = "cube bytes differ from their digest"
                else:
                    return ["match", commit]
        except (OSError, KeyError, ValueError):
            reason = "unreadable or partial"
        if discard:
            path.unlink()
            return [f"removed ({reason})", None]
        return [reason, None]
    verdict, commit = rank0_transaction(path, stage="checkpoint.check", write=check,
                                        return_value=True)
    if verdict.startswith("removed"):
        print_fn(f"WARNING checkpoint: {verdict} {path}; recomputing")
    if verdict != "match":
        return None, verdict
    record, state, error = None, None, None
    try:
        with h5py.File(path, "r") as f:
            text = bytes(f["record"][()])
            if hashlib.sha256(text).hexdigest() != commit:
                raise ValueError("checkpoint changed between rank 0's check and this read")
            record = json.loads(text)
            if record.get("state_sha256") is not None:
                blob = f["state"][()].tobytes()
                if hashlib.sha256(blob).hexdigest() != record["state_sha256"]:
                    raise ValueError("checkpoint state digest differs from its record")
                state = _plain_loads(blob)
    except (OSError, KeyError, ValueError) as exc:
        error = exc
    agree_io_error(error, path=path, stage="checkpoint.record")
    host = _decode(record["host"])
    if not cubes:
        return {}, host, state, commit
    print_fn(f"checkpoint: authenticated {schema} state read from {path}")
    return read_checkpoint_cubes(path, commit=commit, mesh=mesh), host, state, commit


def read_checkpoint_cubes(path, *, commit, mesh):
    """Every cube of an authenticated checkpoint, at its recorded spec.

    ``shape=None`` lets SlabIO round the stored extent up to this mesh, so a
    reader on another mesh gets its own padded carrier (zero past the data).
    """
    import h5py
    from jax.sharding import PartitionSpec as P
    from common.collectives import agree_io_error
    from .slab_io import SlabIO

    error = None
    try:
        with h5py.File(path, "r") as f:
            text = bytes(f["record"][()])
        if hashlib.sha256(text).hexdigest() != commit:
            raise ValueError("checkpoint changed after it was authenticated")
        record = json.loads(text)
    except (OSError, KeyError, ValueError) as exc:
        error = exc
    agree_io_error(error, path=path, stage="checkpoint.cubes")
    cubes = dict.fromkeys(CUBES) if record["schema"] == SCHEMA else {}
    with SlabIO(str(path), mode="r", mesh=mesh) as io:
        for name, meta in record["cubes"].items():
            spec = P(*[tuple(a) if isinstance(a, list) else a for a in meta["spec"]])
            cubes[name] = io.read_slab(name, dtype=np.dtype(meta["dtype"]), mesh=mesh,
                                       partition_spec=spec)
    return cubes
