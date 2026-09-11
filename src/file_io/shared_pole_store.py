"""Canonical shared real-pole model and construction scratch I/O.

The physical convention is Wc(z) = b (z_Ry**2 - Lambda)**-1 b†.
The on-disk dataset remains ``factor`` for compatibility; it stores b.
b is never divided by sqrt(2 Omega) here. Bulk payloads cross SlabIO only;
centroid packing belongs exclusively to these I/O boundaries. All entry points
are collective over the supplied mesh, including validation and publication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common import timing
from common.collectives import agree_io_error, rank0_transaction, psum_replicate
from file_io.slab_io import SlabIO, mesh_divisible_shape
from file_io.commit_state import assert_committed, set_commit_state
from symmetry_maps import QirrTables, validate_qirr_tables

SCHEMA = "lorrax.shared-real-pole.v1"
BANK_SCHEMA = "lorrax.shared-real-pole-bank.v1"
_TABLE_KEYS = ("irr_idx_q", "sym_idx_q", "q_irr_frac", "sym_perm", "L_table")
_IDENTITY_KEYS = ("iteration_id", "hamiltonian", "energies", "occupations",
                  "wavefunctions", "centroids")


def _refuse(message):
    raise ValueError(f"GATE shared_pole_store: {message}; want: authenticated "
                     "current-map canonical scalar model; fix: rebuild the artifact")


def _capacity(meta):
    """Require the map ledger; unknown caller lifetimes never mean zero."""
    ledger = getattr(meta, "shared_pole_capacity", None)
    if ledger is None:
        _refuse("missing map CapacityLedger before tensor I/O")
    ledger.live_stages
    expected = dict(nq=int(meta.nk_tot),nspinor=int(meta.nspinor),
                    nmu=int(meta.mu_basis.n_logical),
                    px=int(meta.mu_basis.mesh_xy.shape["x"]),
                    py=int(meta.mu_basis.mesh_xy.shape["y"]))
    if ledger.geometry != expected:
        _refuse("capacity ledger geometry differs from current map")
    return ledger


def _check_io_capacity(ledger, mesh, header):
    if ledger is None:
        _refuse("tensor reader requires capacity")
    expected = dict(nq=len(header["qirr"]["irr_idx_q"]),
                    nspinor=header["nspinor"],nmu=header["n_mu_logical"],
                    px=int(mesh.shape["x"]),py=int(mesh.shape["y"]))
    if ledger.geometry != expected:
        _refuse("capacity ledger geometry differs from stored model/current mesh")
    ledger.live_stages


def _local_bytes(shape, dtype, mesh, spec):
    """Maximum local payload bytes for an already divisible named layout."""
    count = int(np.prod(shape))
    for axes in spec:
        if axes is not None:
            for axis in ((axes,) if isinstance(axes, str) else axes):
                count //= int(mesh.shape[axis])
    return count * np.dtype(dtype).itemsize


def _conversion_bytes(basis, shape, spec, *, unpack, operator=False):
    """Compile the existing basis conversion without allocating its operand.

    Return argument, output and compiler temporary bytes per rank. These are
    the actual pack/unpack kernel's split-padded collective buffers, not a
    second permutation implementation. Inputs supplied by a writer belong to
    its caller's live reservation; a reader owns its canonical input too.
    """
    argument = _local_bytes(shape, np.complex128, basis.mesh_xy, spec)
    if basis.is_identity:
        return argument, argument, 0
    kernel = (basis._operator_kernel(spec, unpack) if operator
              else basis._axis_kernel(1, spec, unpack))
    operand = jax.ShapeDtypeStruct(shape, jnp.complex128,
                                  sharding=NamedSharding(basis.mesh_xy, spec))
    stats = kernel.lower(operand).compile().memory_analysis()
    if stats is None:
        _refuse("compiler did not supply centroid conversion memory statistics")
    if int(stats.argument_size_in_bytes) < argument:
        _refuse("compiler memory statistics are smaller than the rank-local argument")
    return (int(stats.argument_size_in_bytes), int(stats.output_size_in_bytes),
            max(0, int(stats.temp_size_in_bytes)-int(stats.alias_size_in_bytes)))


def _admit(ledger, stage, resident, workspace=0, *, host_payload=0,
           device_panel=0, host_metadata=0, native_host=False, io=None):
    """Reserve device bytes and emit a separate host-staging receipt.

    Caller live_stages must cover ALL supplied device arguments and other
    arrays retained across this call; store reservations own only additional
    buffers. Host payload copies never exceed the local device panel.
    Native phdf5 staging rounds each read/write high-water buffer to 2 MiB.
    """
    if ledger is None:
        _refuse("payload authentication requires the map capacity ledger")
    high_water = int(device_panel)
    if io is not None:
        high_water = max(high_water,getattr(io,"_shared_pole_host_high_water",0))
        io._shared_pole_host_high_water = high_water
    host = dict(status="PASS" if host_payload <= device_panel else "FAIL",
                payload_copy_bytes_per_rank=int(host_payload),
                device_panel_bytes_per_rank=int(device_panel),
                metadata_bytes_per_rank=int(host_metadata),
                native_staging_bytes_per_rank=(2*((high_water+2097151)//2097152)*2097152
                                               if native_host and high_water else 0),
                scope="host bytes; excluded from device 3U admission")
    if host["status"] != "PASS":
        _refuse("host payload copy exceeds its bounded device panel")
    row = ledger.reserve(f"store.{stage}.{len(ledger.entries)}",
                         resident_bytes_per_rank=int(resident),
                         workspace_bytes_per_rank=int(workspace),
                         concurrent_with=ledger.live_stages)
    if jax.process_index() == 0:
        print("[shared_pole_store capacity] " + _json(dict(device=row, host=host)), flush=True)
    return row


def _json(value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, complex):
            return {"real": x.real, "imag": x.imag}
        raise TypeError(f"not JSON metadata: {type(x).__name__}")
    return json.dumps(value, default=encode, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _span(span, n, name):
    if isinstance(span, slice):
        if span.step not in (None, 1):
            _refuse(f"{name} must be contiguous, got {span}")
        lo, hi = 0 if span.start is None else span.start, n if span.stop is None else span.stop
    else:
        if len(span) != 2:
            _refuse(f"{name} needs (start, stop), got {span}")
        lo, hi = span
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           for v in (lo, hi)) or not 0 <= lo < hi <= n:
        _refuse(f"{name} out of bounds: {span}, extent {n}")
    return int(lo), int(hi)


def _check_identity(actual, expected):
    for name in _IDENTITY_KEYS:
        if not isinstance(actual.get(name), str) or not actual[name]:
            _refuse(f"identity missing nonempty {name}")
    if _json(actual) != _json(expected):
        mismatches = [f"{key}: got {actual.get(key)!r}, want {expected.get(key)!r}"
                      for key in sorted(set(actual) | set(expected))
                      if key not in actual or key not in expected
                      or _json(actual[key]) != _json(expected[key])]
        _refuse("stale input/SC identity; " + "; ".join(mismatches))


def _read_header(path):
    with h5py.File(path, "r") as f:
        assert_committed(f, path=path)
        if "header_json" not in f:
            _refuse("missing header_json")
        raw = f["header_json"][()]
        return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def _write_header(io, header):
    io.write_attr("header_json", np.bytes_(_json(header)))


def _stamp_header(path, header, stage):
    # Only metadata, after every collective handle has closed.
    def publish():
        with h5py.File(path, "a") as f:
            if "header_json" in f:
                del f["header_json"]
            f.create_dataset("header_json", data=np.bytes_(_json(header)))
            f.flush()
    rank0_transaction(path, stage=stage, write=publish)


def _check_basis(meta, header):
    """Authenticate scientific centroid order independently of mesh padding."""
    basis = meta.mu_basis
    digest = hashlib.sha256(np.asarray(
        basis.canonical_indices, dtype="<i4").tobytes()).hexdigest()
    if (basis.n_logical != header["n_mu_logical"]
            or int(meta.nspinor) != header["nspinor"]
            or digest != header["centroid_digest"]):
        _refuse("reader/writer logical centroid or spin identity changed")
    return basis


def _metadata(meta, tables, recipe, identity):
    """Authenticate small scientific identities; no tensor data is gathered.

    ``tables`` is a plain mapping with canonical ``qirr`` (QirrTables),
    ``q_irr_full_idx`` and the generating ``sym`` (SymMaps). ``meta.mu_basis``
    owns logical centroid identity and the I/O packing; ``meta.nspinor`` is 1.
    """
    _check_identity(identity, identity)
    basis = meta.mu_basis
    if int(meta.nspinor) != 1:
        _refuse(f"unsupported Nspinor={meta.nspinor}")
    sym = tables["sym"]
    if not bool(sym.trs_allowed):
        _refuse("TRS-broken representation is unsupported")
    qt = tables["qirr"].logical(basis.n_logical).canonical()
    validate_qirr_tables(qt, qt.n_q_ibz, basis.n_logical)
    qids = np.asarray(tables["q_irr_full_idx"])
    if (qids.dtype.kind not in "iu" or qids.shape != (qt.n_q_ibz,)
            or np.any(qids < 0) or np.any(qids >= qt.n_q_full)
            or len(np.unique(qids)) != len(qids)
            or not np.array_equal(qt.irr_idx_q[qids], np.arange(qt.n_q_ibz))):
        _refuse("q_irr_full_idx does not identify canonical raw parents")
    rows = np.arange(qt.sym_perm.shape[0], dtype=np.int32)
    rotation, translation, antiunitary = sym.operation_rows(rows)
    spin = sym.spinor_action(rows, nspinor=1)
    active = np.asarray(sym.active_symmetry_rows, dtype=np.int32)
    if not np.all(np.isin(qt.sym_idx_q, active)):
        _refuse("QirrTables uses unauthorized operation rows")
    if not isinstance(recipe, dict) or not recipe:
        _refuse("missing resolved recipe and gate versions")
    centroid_hash = hashlib.sha256(np.asarray(
        basis.canonical_indices, dtype="<i4").tobytes()).hexdigest()
    return {
        "schema": SCHEMA, "identity": identity, "recipe": recipe,
        "recipe_hash": hashlib.sha256(_json(recipe).encode()).hexdigest(),
        # Keep the v1 disk spelling for existing models; its C denotes b.
        "normalization": "Wc=C/(z_Ry^2-Lambda_Ry2)*C_dagger",
        "units": {"factor": "Ry^(3/2)", "poles2_ry2": "Ry^2"},
        "representation": "scalar-trs-even-s", "parent_convention": "raw-parent",
        "n_q_irr": qt.n_q_ibz, "n_q_full": qt.n_q_full,
        "n_mu_logical": basis.n_logical, "nspinor": 1,
        "centroid_digest": centroid_hash,
        "grid": [int(meta.nkx), int(meta.nky), int(meta.nkz)],
        "fft_grid": np.asarray(meta.fft_grid).tolist(),
        "q_order": "canonical-full-flat", "q_shift": [0.0, 0.0, 0.0],
        "q_irr_full_idx": qids.tolist(),
        "qirr": {**{k: getattr(qt, k).tolist() for k in _TABLE_KEYS},
                 "n_sym_spatial": qt.n_sym_spatial, "digest": qt.digest()},
        "operations": {"rows": rows.tolist(), "rotation": rotation.tolist(),
                       "translation": translation.tolist(),
                       "antiunitary": antiunitary.tolist(),
                       "spin_real": np.real(spin).tolist(),
                       "spin_imag": np.imag(spin).tolist(),
                       "authorized_rows": active.tolist(),
                       "typing_source": str(sym.operation_typing_source)},
        "finalized": False,
    }


def _write_metadata(io, header):
    """Persist typed small tables, with the JSON header authenticating them."""
    io.write_attr("q_irr_full_idx", np.asarray(header["q_irr_full_idx"], np.int64))
    qt = header["qirr"]
    for name in _TABLE_KEYS:
        dtype = np.float64 if name == "q_irr_frac" else np.int32
        io.write_attr("qirr/" + name, np.asarray(qt[name], dtype))
    io.write_attr("qirr/n_sym_spatial", np.int64(qt["n_sym_spatial"]))
    for name, value in header["operations"].items():
        if name == "typing_source":
            value = np.bytes_(value)
        elif name in ("rows", "rotation", "authorized_rows"):
            value = np.asarray(value, np.int32)
        elif name == "antiunitary":
            value = np.asarray(value, np.int8)
        else:
            value = np.asarray(value, np.float64)
        io.write_attr("operations/" + name, value)


@jax.jit
def _factor_valid(b, poles2, K):
    """Reduce the factor contract to one scalar without eager array temporaries.

    b keeps its caller's named row/column sharding; poles2 and K carry
    squared Ry poles and active counts. The result is a replicated boolean.
    """
    active = jnp.arange(b.shape[3])[None, :] < K[:, None]
    return (jnp.all(jnp.isfinite(b)) & jnp.all(jnp.isfinite(poles2))
          & jnp.all(jnp.where(active, poles2 > 0, poles2 == 1))
          & jnp.all(jnp.where(active[:, None, None, :], True, b == 0))
          & jnp.all(jnp.where(active[:, 1:], poles2[:, 1:] >= poles2[:, :-1], True)))


@timing.timed("factor_validation")
def _check_factor(b, poles2, K):
    """Gate the physical active prefix and exact inactive sentinel."""
    if b.dtype != np.complex128 or poles2.dtype != np.float64 or K.dtype != np.int64:
        _refuse(f"dtypes {(b.dtype, poles2.dtype, K.dtype)}")
    if b.ndim != 4 or poles2.shape != (b.shape[0], b.shape[3]) or K.shape != (b.shape[0],):
        _refuse("factor/poles/count shape mismatch")
    if np.any(K < 0) or np.any(K > b.shape[3]):
        _refuse("K outside factor column capacity")
    ok = _factor_valid(b, poles2, K)
    if not bool(ok):
        _refuse("nonfinite, unsorted/nonpositive active poles or invalid inactive sentinel")


@timing.timed("shared_pole_store.write_model")
def write_shared_pole_model(path, b, poles2, K, *, q_span, meta, tables,
                            recipe, receipts):
    """Stage a bounded q batch and finalize automatically at complete K census.

    Parameters
    ----------
    b : jax.Array, complex128, (parent, mu_p, spin, Kp)
        Physical Ry^(3/2) factor; NamedSharding(mesh_xy,P(None,'x',None,'y')).
    poles2 : jax.Array, float64, (b, Kp)
        Sorted squared poles in Ry², same padded column capacity as b.
    K : array, int64, (b,)
        Physical active counts; inactive b=0 and poles2=1 exactly.
    q_span : slice or pair of int
        Contiguous canonical raw-parent range [start, stop).
    meta, tables, recipe, receipts : bundles / dict
        Packed basis, canonical tables, resolved recipe, and construction
        receipts including ``identity`` with current SC/input content identities.

    Returns
    -------
    dict
        Header; finalized and digest are published only after all parents close.
        Staging plus final datasets use at most twice the compact payload bytes.
    """
    with timing.section("staging"):
        header = _metadata(meta, tables, recipe, receipts["identity"])
        mesh = meta.mu_basis.mesh_xy
        want = NamedSharding(mesh, P(None, "x", None, "y"))
        if not isinstance(b, jax.Array) or not b.sharding.is_equivalent_to(want, 4):
            _refuse("constructor factor is not the declared XY handoff")
        if b.shape[1:3] != (meta.mu_basis.n_packed, 1):
            _refuse("factor does not use the current packed centroid basis")
        ledger = _capacity(meta)
        arg, output, temporary = _conversion_bytes(meta.mu_basis, b.shape, want.spec, unpack=True)
        # One factor-sized envelope covers eager finite/sentinel check scratch.
        _admit(ledger, "write_model", output, temporary+arg+3*int(poles2.size)*8,
               device_panel=max(arg,output,8*int(poles2.size)), native_host=True)
        K = np.asarray(K)
        _check_factor(b, poles2, K)
        lo, hi = _span(q_span, header["n_q_irr"], "q_span")
        if hi - lo != b.shape[0]:
            _refuse("q_span does not match factor batch")
        if Path(path).exists():
            previous = _read_header(path)
            if previous["finalized"]:
                _refuse("finalized models are immutable")
            for key in header:
                if key != "finalized" and _json(previous[key]) != _json(header[key]):
                    _refuse(f"staging identity changed: {key}")
            header = previous
        else:
            header.update(written_q=[False] * header["n_q_irr"],
                          K=[0] * header["n_q_irr"], batches=[], construction_receipts=[])
        if any(header["written_q"][lo:hi]):
            _refuse("q_span already committed; never overwrite staged parents")
        width = int(K.max(initial=0))
        name = f"staging/q{lo}_{hi}"
    with timing.section("canonical_basis_conversion_and_packing"):
        canonical = meta.mu_basis.unpack_axis(b, 1)
        canonical.block_until_ready()
    if not Path(path).exists():
        with SlabIO(path, mode="w", mesh=mesh) as io:
            _write_metadata(io, header)
            _write_header(io, header)
    # The native contiguous-dataset API requires its parent group to exist.
    # Metadata setup is ordered before opening the collective payload handle.
    def prepare_group():
        with h5py.File(path, "a") as f:
            f.require_group(name)
    rank0_transaction(path, stage="shared_pole.stage_group", write=prepare_group)
    with SlabIO(path, mode="a", mesh=mesh) as io:
        if width:
            io.create_dataset(name + "/factor", shape=(hi-lo, header["n_mu_logical"], 1, width), dtype=np.complex128)
            io.create_dataset(name + "/poles2", shape=(hi-lo, width), dtype=np.float64)
            with timing.section("write_slab"):
                io.write_slab(name + "/factor", canonical)
                io.write_slab(name + "/poles2", poles2)
        header["written_q"][lo:hi] = [True] * (hi-lo)
        header["K"][lo:hi] = K.tolist()
        header["batches"].append({"lo": lo, "hi": hi, "width": width, "name": name})
        header["construction_receipts"].append({"q_span": [lo, hi], "receipt": receipts})
        _write_metadata(io, header)
        io.write_attr("written_q", np.asarray(header["written_q"], np.int8))
        _write_header(io, header)
        with timing.section("sync_writes"):
            io.sync_writes()
    del canonical
    if all(header["written_q"]):
        return _finalize_model(path, meta=meta, header=header)
    return header


def finalize_shared_pole_model(path, *, meta, expected_identity):
    """Resume finalization from a successfully closed, complete staging census.

    A native write failure retains the global incomplete marker and refuses;
    this retries the recoverable boundary after all staged batches closed.
    """
    header = _read_header(path)
    _check_identity(header["identity"], expected_identity)
    _check_basis(meta, header)
    if header["schema"] != SCHEMA or not all(header["written_q"]):
        _refuse("finalization requires every staged parent")
    if header["finalized"]:
        return validate_shared_pole_model(
            path, expected_identity=expected_identity, mesh_xy=meta.mu_basis.mesh_xy,
            capacity=_capacity(meta))
    return _finalize_model(path, meta=meta, header=header)


@timing.timed("shared_pole_store.finalize")
def _finalize_model(path, *, meta, header):
    mesh, basis = meta.mu_basis.mesh_xy, meta.mu_basis
    nq, nmu = header["n_q_irr"], header["n_mu_logical"]
    kmax = max(header["K"])
    panel = 16*basis.n_canonical*((kmax+int(mesh.shape["y"])-1)//int(mesh.shape["y"]))/int(mesh.shape["x"])
    batch_width = max(v["hi"] - v["lo"] for v in header["batches"])
    _admit(_capacity(meta), "finalize", batch_width*(int(panel)+24*kmax),
           device_panel=batch_width*max(int(panel),8*kmax), native_host=True)
    header["Kmax"] = kmax
    header["compact_payload_bytes"] = nq * (16*nmu*kmax + 8*kmax + 8)
    header["staging_payload_bytes"] = sum((v["hi"]-v["lo"]) * v["width"] * (16*nmu+8) for v in header["batches"])
    header["peak_payload_bytes"] = header["compact_payload_bytes"] + header["staging_payload_bytes"]
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.create_dataset("factor", shape=(nq, nmu, 1, kmax), dtype=np.complex128)
        io.create_dataset("poles2_ry2", shape=(nq, kmax), dtype=np.float64)
        for batch in header["batches"]:
            # Preserve the constructor's admitted q batch through finalization.
            # Kmax is already known from the committed census; no all-q carrier.
            lo, hi = batch["lo"], batch["hi"]
            spec = P(None, "x", None, "y")
            read_shape = mesh_divisible_shape(
                (hi-lo, basis.n_canonical, 1, kmax), mesh, spec)
            if kmax == 0:
                continue
            with timing.section("staging_read_and_padding"):
                if batch["width"]:
                    factor = io.read_slab(batch["name"] + "/factor",
                        shape=read_shape, offset=(0, 0, 0, 0), partition_spec=spec)
                    poles = io.read_slab(batch["name"] + "/poles2",
                        shape=(hi-lo, kmax), offset=(0, 0), partition_spec=P())
                else:
                    factor = jax.jit(lambda: jnp.zeros(read_shape, jnp.complex128),
                                     out_shardings=NamedSharding(mesh, spec))()
                    poles = jnp.ones((hi-lo, kmax), jnp.float64)
                active = jnp.arange(kmax)[None, :] < jnp.asarray(header["K"][lo:hi])[:, None]
                poles = jnp.where(active, poles, 1.0)
            with timing.section("write_slab"):
                io.write_slab("factor", factor, offset=(lo, 0, 0, 0))
                io.write_slab("poles2_ry2", poles, offset=(lo, 0))
            with timing.section("sync_writes"):
                io.sync_writes()
            del factor, poles
        io.write_attr("K", np.asarray(header["K"], np.int64))
        _write_header(io, header)
    header["digest"] = _model_digest(path, header, mesh, capacity=_capacity(meta))
    def finish():
        with h5py.File(path, "a") as f:
            set_commit_state(f, False)
            del f["staging"]
            header["finalized"] = True
            del f["header_json"]
            f.create_dataset("header_json", data=np.bytes_(_json(header)))
            f.create_dataset("final_commit", data=np.bytes_(header["digest"]))
            set_commit_state(f, True)
    with timing.section("finalisation"):
        rank0_transaction(path, stage="shared_pole.finalize", write=finish)
        return _read_header(path)


@timing.timed("shared_pole_store.digest")
def _model_digest(path, header, mesh, *, capacity):
    """Grid-independent SHA256 of metadata and canonical row digests.

    Read a bounded q batch and column panel through SlabIO. Each canonical row
    hash is updated in column order, so neither column partition nor mesh
    changes the digest. Only row hashes (32 bytes per centroid) are exchanged.
    """
    _check_io_capacity(capacity,mesh,header)
    identity = {k:v for k,v in header.items() if k not in
                ("digest", "finalized", "batches", "staging_payload_bytes", "peak_payload_bytes")}
    digest = hashlib.sha256(_json(identity).encode())
    nmu, kmax = header["n_mu_logical"], header["Kmax"]
    ncan = ((nmu + int(mesh.size)-1)//int(mesh.size))*int(mesh.size)
    column_cap = max(1, (kmax + int(mesh.shape["y"])-1)//int(mesh.shape["y"]))
    panel = 16*ncan*min(kmax,column_cap)//int(mesh.shape["x"])
    batch_limit = min(int(mesh.size), header["n_q_irr"])
    _admit(capacity, "digest", batch_limit*(panel+8*kmax+256*nmu),
           batch_limit*(panel+24*kmax), host_payload=batch_limit*panel,
           device_panel=batch_limit*panel,
           host_metadata=batch_limit*(256*nmu+8*kmax), native_host=True)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        for q0 in range(0, header["n_q_irr"], batch_limit):
            q1 = min(q0+batch_limit, header["n_q_irr"])
            batch = q1-q0
            active_counts = np.asarray(header["K"][q0:q1], np.int64)
            if kmax == 0:
                if np.any(active_counts != 0):
                    _refuse("nonzero K in empty model")
                for _ in range(batch):
                    digest.update(hashlib.sha256(b"").digest() * nmu)
                continue
            with timing.section('pole_read'):
                poles = io.read_slab("poles2_ry2", shape=(batch,kmax), offset=(q0,0), partition_spec=P())
                host_poles = np.asarray(poles)
            for row, count in zip(host_poles, active_counts):
                if np.any(np.diff(row[:count]) < 0):
                    _refuse("active poles are unsorted across column panels")
            hashers = [{} for _ in range(batch)]
            for c0 in range(0, kmax, column_cap):
                c1 = min(kmax, c0+column_cap)
                with timing.section('factor_read_and_validation'):
                    b = io.read_slab("factor", shape=(batch,ncan,1,c1-c0), offset=(q0,0,0,c0),
                                     partition_spec=P(None,"x",None,None))
                    counts = np.clip(active_counts-c0, 0, c1-c0)
                    _check_factor(b, poles[:,c0:c1], counts)
                with timing.section('host_digest_hashing'):
                    for shard in b.addressable_shards:
                        if shard.replica_id != 0:
                            continue
                        start = shard.index[1].start or 0
                        local = np.asarray(shard.data)[:,:,0,:]
                        for q in range(batch):
                            for i in range(min(local.shape[1], nmu-start)):
                                hasher = hashers[q].setdefault(start+i, hashlib.sha256())
                                hasher.update(np.asarray(local[q,i], dtype="<c16").tobytes())
                    shard = local = None
                    del b
            row_hash = np.zeros((batch,nmu,32), np.uint32)
            for q in range(batch):
                for row, hasher in hashers[q].items():
                    row_hash[q,row] = np.frombuffer(hasher.digest(), np.uint8)
            with timing.section('digest_reduction'):
                row_hash = psum_replicate(row_hash, mesh)
                # Preserve the original q-major row-hashes then poles byte stream.
                for q in range(batch):
                    digest.update(row_hash[q].astype(np.uint8).tobytes())
                    digest.update(np.asarray(host_poles[q:q+1], dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_shared_pole_model(path, *, expected_identity, mesh_xy, capacity=None):
    """Refuse partial, stale, malformed or changed payload; return header.

    Validation is collective and bounded to one irreducible parent face. It
    verifies storage integrity, not the constructor's physical gate claims.
    With capacity=None only metadata is checked and the returned ephemeral
    validation_receipt explicitly says NOT_MEASURED; tensor readers and
    restart membership refuse that receipt. Production callers supply the
    map ledger with bound caller lifetimes. No receipt is appended to disk.
    """
    error = None
    try:
        header = _read_header(path)
        _check_identity(header["identity"], expected_identity)
        if header["schema"] != SCHEMA or not header["finalized"] or not all(header["written_q"]):
            _refuse("missing final shared-pole commit or incomplete q census")
        qt = shared_pole_qirr_tables(header)
        validate_qirr_tables(qt, header["n_q_irr"], header["n_mu_logical"])
        with h5py.File(path, "r") as f:
            for name, shape, dtype in (
                ("factor", (header["n_q_irr"],header["n_mu_logical"],1,header["Kmax"]), np.complex128),
                ("poles2_ry2", (header["n_q_irr"],header["Kmax"]), np.float64),
                ("K", (header["n_q_irr"],), np.int64)):
                if name not in f or f[name].shape != shape or f[name].dtype != dtype or f[name].chunks is not None:
                    _refuse(f"dataset {name} shape/dtype/contiguity mismatch")
            if "final_commit" not in f or f["final_commit"][()].decode() != header["digest"]:
                _refuse("missing or inconsistent final commit")
            if not np.array_equal(f["K"][:], header["K"]) or not np.array_equal(f["written_q"][:],header["written_q"]):
                _refuse("count/completion metadata mismatch")
            for key in _TABLE_KEYS:
                if not np.array_equal(f["qirr/"+key][:], np.asarray(header["qirr"][key])):
                    _refuse(f"typed qirr metadata changed: {key}")
            if (not np.array_equal(f["q_irr_full_idx"][:], header["q_irr_full_idx"])
                    or int(f["qirr/n_sym_spatial"][()]) != header["qirr"]["n_sym_spatial"]):
                _refuse("typed parent or operation-count metadata changed")
            for key, expected in header["operations"].items():
                actual = f["operations/"+key][()]
                if key == "typing_source":
                    actual = actual.decode()
                if not np.array_equal(actual, expected):
                    _refuse(f"typed operation metadata changed: {key}")
    except Exception as exc:
        error = exc
    try:
        agree_io_error(error, path=path, stage="shared_pole_model/metadata")
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if capacity is None:
        # No tensor allocation is permitted without admission. This receipt
        # must not be mistaken for payload authentication by a restart caller.
        return dict(header, validation_receipt={"status":"NOT_MEASURED",
                    "scope":"metadata only; payload digest not authenticated"})
    if _model_digest(path, header, mesh_xy, capacity=capacity) != header["digest"]:
        _refuse("model payload/identity digest mismatch")
    return header


def shared_pole_qirr_tables(header):
    """Return the existing symmetry-service table in canonical logical order.

    Runtime endpoint packing/locality certification stays with mu_basis and
    symmetry_maps; the store never invents a second star-action table.
    """
    return QirrTables(
        **{k:np.asarray(header["qirr"][k]) for k in _TABLE_KEYS},
        n_sym_spatial=header["qirr"]["n_sym_spatial"]).canonical()


def read_shared_pole_census(io, *, header, capacity=None):
    """Return replicated float64 poles² [Nq,Kmax] and int64 counts [Nq].

    This O(Nq*Kmax) metadata is the window planner's census, not factor data.
    Counts mask every padded prefix; the sentinel is never a physical pole.
    capacity is the current map ledger with explicitly bound live_stages.
    """
    if header.get("validation_receipt", {}).get("status") == "NOT_MEASURED":
        _refuse("metadata-only validation cannot authorize tensor reads")
    if header["schema"] != SCHEMA or not header["finalized"]:
        _refuse("pole census requires a validated finalized model")
    _check_io_capacity(capacity,io.mesh,header)
    amount = 8*header["n_q_irr"]*header["Kmax"]
    _admit(capacity,"read_census",amount+8*header["n_q_irr"],2*amount,
           device_panel=amount,native_host=True,io=io)
    poles = (io.read_slab("poles2_ry2", partition_spec=P()) if header["Kmax"]
             else jnp.ones((header["n_q_irr"],0),jnp.float64))
    counts = jnp.asarray(header["K"], dtype=jnp.int64)
    active = jnp.arange(header["Kmax"])[None,:] < counts[:,None]
    return jnp.where(active, poles, 1.0), counts


def read_shared_pole_faces(io, q_span, *, meta, header, column_span=None):
    """Read canonical row faces and pack once at the I/O boundary.

    Returns b_X, b_Y, poles2, K with shapes (b,mu_p,spin,Kcap),
    (b,mu_p,spin,Kcap), (b,Kcap), (b,). Faces use P(None,'x',None,None)
    and P(None,'y',None,None); poles and int64 counts are replicated. K is
    the active count *within the returned column slice*, so every consumer
    can mask with arange(Kcap)<K even when column_span starts above zero.
    """
    if header.get("validation_receipt", {}).get("status") == "NOT_MEASURED":
        _refuse("metadata-only validation cannot authorize tensor reads")
    if header["schema"] != SCHEMA or not header["finalized"]:
        _refuse("face reader requires a validated finalized model")
    basis = _check_basis(meta, header)
    ledger = _capacity(meta)
    if io.mesh is not basis.mesh_xy:
        _refuse("reader mesh differs from packed basis mesh")
    _check_io_capacity(ledger,io.mesh,header)
    lo, hi = _span(q_span, header["n_q_irr"], "q_span")
    if header["Kmax"] == 0 and column_span is None:
        _admit(ledger,"empty_faces",8*(hi-lo))
        shape = (hi-lo,basis.n_packed,1,0)
        faces = [jax.jit(lambda: jnp.zeros(shape,jnp.complex128),
                         out_shardings=NamedSharding(io.mesh,P(None,axis,None,None)))()
                 for axis in ("x","y")]
        return (*faces,jnp.ones((hi-lo,0),jnp.float64),jnp.zeros(hi-lo,jnp.int64))
    c0, c1 = _span(column_span or (0,header["Kmax"]), header["Kmax"], "column_span")
    totals = []
    for axis in ("x","y"):
        shape = (hi-lo,basis.n_canonical,1,c1-c0)
        totals.append(_conversion_bytes(basis,shape,P(None,axis,None,None),unpack=False))
    ax,fx,tx = totals[0]; ay,fy,ty = totals[1]
    metadata = 32*(hi-lo)*(c1-c0)+8*(hi-lo)
    peak = max(ax+fx+tx, 2*fx, fx+ay+fy+ty, fx+2*fy)
    _admit(ledger,"read_faces",fx+fy+metadata,max(0,peak-fx-fy),
           device_panel=max(ax,ay,8*(hi-lo)*(c1-c0)),native_host=True,io=io)
    counts = jnp.asarray(np.clip(np.asarray(header["K"][lo:hi])-c0,0,c1-c0), dtype=jnp.int64)
    active = jnp.arange(c1-c0)[None,:] < counts[:,None]
    faces = []
    for axis in ("x", "y"):
        spec = P(None,axis,None,None)
        b = io.read_slab("factor", shape=(hi-lo,basis.n_canonical,1,c1-c0),
                         offset=(lo,0,0,c0), partition_spec=spec)
        b = basis.pack_axis(b, 1, spec=spec)
        faces.append(jnp.where(active[:,None,None,:] & jnp.asarray(
            basis.active_mask)[None,:,None,None], b, 0.0))
        # Complete masking before allocating the other face: Python reference
        # release alone does not end an asynchronously dispatched input lifetime.
        faces[-1].block_until_ready()
        del b
    poles = io.read_slab("poles2_ry2", shape=(hi-lo,c1-c0), offset=(lo,c0), partition_spec=P())
    return (*faces, jnp.where(active,poles,1.0), counts)


# Scratch bank uses the same identity and metadata transactions.
_BANK_SAMPLE_FIELDS = ("Wc", "dWc_ds")
_BANK_MOMENT_FIELDS = ("M1", "M3")


def _bank_plan(recipe):
    """Validate the resolver's flat native typed point/role arrays.

    The JSON header uses the common complex encoder; decoding that header
    here restores complex128 metadata without defining another record view.
    """
    codes = recipe.get("role_codes")
    names = {"line", "imaginary", "infinity", "held_line", "held_imaginary"}
    if (not isinstance(codes, dict) or set(codes) != names
            or len(set(codes.values())) != len(names)):
        _refuse("scratch plan requires the IINPUTS named role_codes table")
    plan = {}
    for name, dtype in (("z_ry", np.complex128), ("role", np.int8),
                        ("distinct_id", np.int64), ("held", np.bool_)):
        if name not in recipe:
            _refuse(f"scratch plan lacks flat {name} array")
        value = recipe[name]
        if name == "z_ry" and isinstance(value, list):
            value = [complex(v["real"], v["imag"]) if isinstance(v, dict) else v for v in value]
        raw = np.asarray(value)
        # Native producer arrays carry their declared ABI dtype; JSON lists
        # from an authenticated header are decoded to those fixed dtypes.
        if isinstance(value, np.ndarray) and raw.dtype != np.dtype(dtype):
            _refuse(f"scratch plan {name} must have dtype {np.dtype(dtype)}")
        plan[name] = np.asarray(value, dtype=dtype)
    n = plan["z_ry"].size
    if not n or any(value.shape != (n,) for value in plan.values()):
        _refuse("scratch plan arrays must have equal nonempty flat lengths")
    if not np.isin(plan["role"], list(codes.values())).all():
        _refuse("scratch plan has unnamed role codes")
    if np.any(plan["role"] == codes["infinity"]):
        _refuse("infinity is a reserved moment role, not a bank sample")
    finite = plan["role"] != codes["infinity"]
    if (not np.isfinite(plan["z_ry"][finite]).all()
            or np.any(plan["distinct_id"][finite] < 0)):
        _refuse("scratch plan finite evaluation coordinates/IDs are invalid")
    # Infinity directions consume the moment fields, never a Wc sample.
    ids = np.unique(plan["distinct_id"][finite])
    if not len(ids) or not np.array_equal(ids, np.arange(len(ids))):
        _refuse("scratch finite evaluation IDs must be contiguous from zero")
    for idx in ids:
        rows = finite & (plan["distinct_id"] == idx)
        if len(np.unique(plan["z_ry"][rows])) != 1:
            _refuse("scratch evaluation ID aliases different physical points")
        if len(np.unique(plan["held"][rows])) != 1:
            _refuse("scratch evaluation is both held and fitted")
    for name in ("support_pair", "fit_ids", "held_ids"):
        if name not in recipe:
            _refuse(f"scratch plan lacks canonical {name}")
        raw = np.asarray(recipe[name])
        if isinstance(recipe[name], np.ndarray) and raw.dtype != np.int64:
            _refuse(f"scratch plan {name} must have dtype int64")
        plan[name] = np.asarray(raw, dtype=np.int64)
    if plan["support_pair"].shape != (n, 2):
        _refuse("scratch support_pair must have shape (N,2)")
    for name, held in (("fit_ids", False), ("held_ids", True)):
        expected = np.unique(plan["distinct_id"][plan["held"] == held])
        if not np.array_equal(np.sort(plan[name]), expected):
            _refuse(f"scratch {name} contradicts typed role rows")
    plan["role_codes"] = codes
    return plan


def _bank_nsample(plan):
    finite = plan["role"] != plan["role_codes"]["infinity"]
    return len(np.unique(plan["distinct_id"][finite]))


def initialize_shared_pole_bank(path, *, meta, tables, recipe, identity,
                                mesh_xy):
    """Create optional construction-resume scratch with no payload marked ready.

    Parameters
    ----------
    path : path-like
        New HDF5 path. Existing files are never truncated.
    meta : object
        Carries the canonical/packed centroid basis and scalar representation.
    tables, recipe, identity : dict or canonical table bundle
        Same authenticated metadata as the compact model. Flat ``z_ry``,
        ``role``, ``distinct_id`` and ``held`` arrays preserve the exact plan.
    mesh_xy : jax.sharding.Mesh
        Named ``x``/``y`` mesh used by the collective SlabIO transport.

    Returns
    -------
    dict
        Header for the unfinalized scratch file. Wc is in Ry, dWc/ds in
        Ry**-1, M1 in Ry**3 and M3 in Ry**5, with s=z_Ry**2.
    """
    if Path(path).exists():
        _refuse("scratch bank already exists; validate it before resuming")
    plan = _bank_plan(recipe)
    header = _metadata(meta, tables, recipe, identity)
    parents = (tables["q_irr_full_idx"] if isinstance(tables, dict)
               else tables.q_irr_full_idx)
    nq, nsample, d = len(parents), _bank_nsample(plan), int(meta.mu_basis.n_logical)
    if nq <= 0:
        _refuse("scratch bank requires at least one irreducible q")
    header.update(
        schema=BANK_SCHEMA, bank_shape={"nq": nq, "nsample": nsample, "d": d},
        bank_sample_plan=plan,
        bank_plan_digest=hashlib.sha256(_json(plan).encode()).hexdigest(),
        sample_written=np.zeros((nq, nsample, 2), dtype=bool).tolist(),
        moment_written=np.zeros((nq, 2), dtype=bool).tolist(),
        complete=False, final_commit=None,
        units={"Wc": "Ry", "dWc_ds": "Ry^-1", "M1": "Ry^3", "M3": "Ry^5"},
        derivative_variable="s=z_Ry^2",
        moment_convention="S_m = 2 M_(2m+1); physical M1 and M3 only",
        payload_bytes=16 * nq * (2 * nsample + 2) * d * d)
    with SlabIO(path, mode="w", mesh=mesh_xy) as io:
        for field in _BANK_SAMPLE_FIELDS:
            io.create_dataset(field, shape=(nq, nsample, d, d), dtype=np.complex128)
        for field in _BANK_MOMENT_FIELDS:
            io.create_dataset(field, shape=(nq, d, d), dtype=np.complex128)
        io.write_attr("sample_written", np.asarray(header["sample_written"], dtype=np.bool_))
        io.write_attr("moment_written", np.asarray(header["moment_written"], dtype=np.bool_))
        for name in ("z_ry", "role", "distinct_id", "held", "support_pair", "fit_ids", "held_ids"):
            io.write_attr(name, plan[name])
        io.write_attr("role_codes_json", np.bytes_(_json(plan["role_codes"])))
        _write_metadata(io, header)
        _write_header(io, header)
    return header


def validate_shared_pole_bank(path, *, expected_identity, mesh_xy,
                              require_complete=False):
    """Authenticate the scratch plan and per-field commit masks before replay.

    Only small metadata is read. Partial files are resumable, but a consumer
    cannot read a field whose q/sample mask is absent. Finalization follows
    the collective close; a completed bank is immutable.
    """
    header = _read_header(path)
    if header.get("schema") != BANK_SCHEMA:
        _refuse("scratch bank schema mismatch")
    _check_identity(header["identity"], expected_identity)
    if hashlib.sha256(_json(header["recipe"]).encode()).hexdigest() != header["recipe_hash"]:
        _refuse("scratch bank recipe digest mismatch")
    plan = _bank_plan(header["bank_sample_plan"])
    digest = hashlib.sha256(_json(plan).encode()).hexdigest()
    if digest != header.get("bank_plan_digest"):
        _refuse("scratch bank sample plan digest mismatch")
    if _json(plan) != _json(_bank_plan(header["recipe"])):
        _refuse("scratch bank stale recipe/roles/held map")
    shape = header["bank_shape"]
    nq, nsample = int(shape["nq"]), int(shape["nsample"])
    samples = np.asarray(header["sample_written"], dtype=bool)
    moments = np.asarray(header["moment_written"], dtype=bool)
    if samples.shape != (nq, nsample, 2) or moments.shape != (nq, 2):
        _refuse("scratch bank malformed written masks")
    if (nq != header["n_q_irr"] or nsample != _bank_nsample(plan)
            or shape["d"] != header["n_mu_logical"] or header["nspinor"] != 1):
        _refuse("scratch bank geometry/representation mismatch")
    # Geometry only: never load a matrix through the metadata handle.
    with h5py.File(path, "r") as file:
        # Boolean HDF5 enums are metadata, outside phdf5's numeric ABI.
        if not np.array_equal(file["sample_written"][()], samples):
            _refuse("scratch bank sample transaction mismatch")
        if not np.array_equal(file["moment_written"][()], moments):
            _refuse("scratch bank moment transaction mismatch")
        for name in ("z_ry", "role", "distinct_id", "held", "support_pair", "fit_ids", "held_ids"):
            if (name not in file or file[name].shape != plan[name].shape
                    or file[name].dtype != plan[name].dtype
                    or not np.array_equal(file[name][()], plan[name], equal_nan=True)):
                _refuse(f"scratch bank typed plan {name} digest mismatch")
        if file["role_codes_json"][()].decode() != _json(plan["role_codes"]):
            _refuse("scratch bank role code table mismatch")
        for name in _BANK_SAMPLE_FIELDS + _BANK_MOMENT_FIELDS:
            expected = ((nq, nsample) if name in _BANK_SAMPLE_FIELDS else (nq,)) + (shape["d"],) * 2
            if (name not in file or file[name].shape != expected
                    or file[name].dtype != np.dtype(np.complex128)
                    or file[name].chunks is not None):
                _refuse(f"scratch bank {name} schema geometry/dtype/layout mismatch")
    complete = bool(samples.all() and moments.all())
    if header.get("complete") and (not complete or not header.get("final_commit")):
        _refuse("scratch bank invalid completion transaction")
    if header.get("complete"):
        precommit = dict(header, final_commit=None)
        if hashlib.sha256(_json(precommit).encode()).hexdigest() != header["final_commit"]:
            _refuse("scratch bank completion digest mismatch")
    if require_complete and not header.get("complete"):
        _refuse("scratch bank is incomplete")
    return header


def write_shared_pole_bank(path, *, q_span, sample_span=None, Wc=None,
                           dWc_ds=None, M1=None, M3=None, meta,
                           expected_identity, mesh_xy):
    """Write one bounded q/sample batch and commit masks after collective close.

    Sample arrays are complex128 ``[b,a,mu_p,mu_p]`` at
    ``P(None,None,'x','y')``; physical moments are complex128
    ``[b,mu_p,mu_p]`` at ``P(None,'x','y')``. Both endpoint conversions
    go through ``meta.mu_basis``; no off-axis Hermitization or rescaling.
    Already committed fields refuse overwrite, including in partial files.
    """
    header = validate_shared_pole_bank(
        path, expected_identity=expected_identity, mesh_xy=mesh_xy)
    _check_basis(meta, header)
    if mesh_xy is not meta.mu_basis.mesh_xy:
        _refuse("writer mesh differs from packed basis mesh")
    if header.get("complete"):
        _refuse("completed scratch bank is immutable")
    if not (np.asarray(header["sample_written"], dtype=bool).all()
            and np.asarray(header["moment_written"], dtype=bool).all()
            and all(value is None for value in (Wc, dWc_ds, M1, M3))):
        prepared = _prepare_bank_write(header, q_span=q_span,
            sample_span=sample_span, meta=meta, mesh_xy=mesh_xy,
            Wc=Wc, dWc_ds=dWc_ds, M1=M1, M3=M3)
        with SlabIO(path, mode="a", mesh=mesh_xy) as io:
            _write_bank_payload(io, header, meta, prepared)
            _write_bank_masks(io, header)
    _complete_bank(path, header)
    return header


def _write_bank_masks(io, header):
    """Publish drained payload masks within the open collective handle."""
    io.write_attr("sample_written", np.asarray(header["sample_written"], dtype=bool))
    io.write_attr("moment_written", np.asarray(header["moment_written"], dtype=bool))
    _write_header(io, header)


def _complete_bank(path, header):
    """Stamp a fully written bank only after its collective handle closes."""
    if (np.asarray(header["sample_written"], dtype=bool).all()
            and np.asarray(header["moment_written"], dtype=bool).all()):
        header["complete"] = True
        header["final_commit"] = hashlib.sha256(_json(header).encode()).hexdigest()
        _stamp_header(path, header, "shared_pole_bank.complete")


def _prepare_bank_write(header, *, q_span, meta, mesh_xy,
                        sample_span=None, Wc=None, dWc_ds=None,
                        M1=None, M3=None):
    """Validate/admit a packed span before opening or mutating a bank."""
    _check_basis(meta, header)
    if mesh_xy is not meta.mu_basis.mesh_xy:
        _refuse("writer mesh differs from packed basis mesh")
    if header.get("complete"):
        _refuse("completed scratch bank is immutable")
    shape = header["bank_shape"]
    q0, q1 = _span(q_span, shape["nq"], "q_span")
    has_samples = Wc is not None or dWc_ds is not None
    if has_samples and sample_span is None:
        _refuse("scratch sample write requires explicit sample_span")
    if not has_samples and sample_span is not None:
        _refuse("sample_span supplied without sample payload")
    a0, a1 = (_span(sample_span, shape["nsample"], "sample_span")
              if has_samples else (0, 0))
    pending = [(name, value) for name, value in
               (("Wc", Wc), ("dWc_ds", dWc_ds), ("M1", M1), ("M3", M3))
               if value is not None]
    if not pending:
        _refuse("scratch write has no payload")
    basis = meta.mu_basis
    ledger = _capacity(meta)
    _check_io_capacity(ledger,basis.mesh_xy,header)
    if int(basis.n_logical) != int(shape["d"]):
        _refuse("scratch centroid extent mismatch")
    sample_mask = np.asarray(header["sample_written"], dtype=bool)
    moment_mask = np.asarray(header["moment_written"], dtype=bool)
    # Validate every argument before opening the writer: a bad second field
    # must not leave an otherwise legal first field queued in the same call.
    for name, array in pending:
        sample = name in _BANK_SAMPLE_FIELDS
        fields = _BANK_SAMPLE_FIELDS if sample else _BANK_MOMENT_FIELDS
        marked = (sample_mask[q0:q1, a0:a1, fields.index(name)] if sample
                  else moment_mask[q0:q1, fields.index(name)])
        if marked.any():
            _refuse(f"scratch {name} span already committed")
        expected = ((q1-q0, a1-a0) if sample else (q1-q0,)) + (
            basis.n_packed, basis.n_packed)
        spec = P(None, None, 'x', 'y') if sample else P(None, 'x', 'y')
        if tuple(array.shape) != expected or np.dtype(array.dtype) != np.dtype(np.complex128):
            _refuse(f"scratch {name} requires complex128 packed shape {expected}")
        if not isinstance(array, jax.Array) or not array.sharding.is_equivalent_to(NamedSharding(mesh_xy, spec), array.ndim):
            _refuse(f"scratch {name} requires NamedSharding(mesh_xy, {spec})")
        arg, output, temporary = _conversion_bytes(basis,array.shape,spec,unpack=True,operator=True)
        # Factor-sized finite-check envelope, separate from caller input.
        _admit(ledger,"write_bank_"+name,output,temporary+arg,
               device_panel=max(arg,output),native_host=True)
        if not bool(jnp.all(jnp.isfinite(array))):
            _refuse(f"scratch {name} contains nonfinite values")
    return q0, q1, a0, a1, pending, sample_mask, moment_mask


def _write_bank_payload(io, header, meta, prepared):
    """Write admitted spans; caller publishes masks after the queue drains.

    The prepared arrays carry the public writer's units and face shardings.
    Authentication/admission stays in the common preparation owner.
    """
    q0, q1, a0, a1, pending, sample_mask, moment_mask = prepared
    shape, basis = header["bank_shape"], meta.mu_basis
    for name, array in pending:
        sample = name in _BANK_SAMPLE_FIELDS
        spec = P(None, None, 'x', 'y') if sample else P(None, 'x', 'y')
        disk_shape = ((shape["nq"], shape["nsample"]) if sample
                      else (shape["nq"],)) + (shape["d"], shape["d"])
        io.create_dataset(name, shape=disk_shape, dtype=np.complex128)
        canonical = basis.unpack_operator(array, spec=spec)
        offset = (q0, a0, 0, 0) if sample else (q0, 0, 0)
        io.write_slab(name, canonical, offset=offset)
        # SlabIO's write queue owns canonical until drained. Drain each
        # field so endpoint staging cannot accumulate across fields.
        io.sync_writes()
        del canonical
        if sample:
            sample_mask[q0:q1, a0:a1, _BANK_SAMPLE_FIELDS.index(name)] = True
        else:
            moment_mask[q0:q1, _BANK_MOMENT_FIELDS.index(name)] = True
    header["sample_written"] = sample_mask.tolist()
    header["moment_written"] = moment_mask.tolist()


def read_shared_pole_bank(io, q_span, *, meta, header, sample_span=None,
                          fields=("Wc", "dWc_ds")):
    """Read committed bounded scratch fields into packed distributed operators.

    Returns a plain dict of complex128 arrays in the writer layouts. Samples
    require explicit contiguous ``sample_span``; no implicit all-bank read is
    offered. The caller authenticates ``header`` before opening ``io``.
    """
    if header.get("schema") != BANK_SCHEMA:
        _refuse("scratch bank reader schema mismatch")
    _check_basis(meta, header)
    if io.mesh is not meta.mu_basis.mesh_xy:
        _refuse("reader mesh differs from packed basis mesh")
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields) or any(
            f not in _BANK_SAMPLE_FIELDS + _BANK_MOMENT_FIELDS for f in fields):
        _refuse("scratch bank fields must be distinct Wc/dWc_ds/M1/M3 names")
    shape = header["bank_shape"]
    q0, q1 = _span(q_span, shape["nq"], "q_span")
    need_samples = any(name in _BANK_SAMPLE_FIELDS for name in fields)
    if need_samples and sample_span is None:
        _refuse("scratch sample read requires explicit sample_span")
    a0, a1 = (_span(sample_span, shape["nsample"], "sample_span")
              if need_samples else (0, 0))
    basis = meta.mu_basis
    ledger = _capacity(meta)
    _check_io_capacity(ledger,basis.mesh_xy,header)
    if int(basis.n_logical) != int(shape["d"]):
        _refuse("scratch centroid extent mismatch")
    sample_mask = np.asarray(header["sample_written"], dtype=bool)
    moment_mask = np.asarray(header["moment_written"], dtype=bool)
    for name in fields:
        marked = (sample_mask[q0:q1, a0:a1, _BANK_SAMPLE_FIELDS.index(name)]
                  if name in _BANK_SAMPLE_FIELDS
                  else moment_mask[q0:q1, _BANK_MOMENT_FIELDS.index(name)])
        if not marked.all():
            _refuse(f"scratch bank {name} requested span is incomplete")
    out = {}
    retained = 0
    for name in fields:
        sample = name in _BANK_SAMPLE_FIELDS
        spec = P(None, None, 'x', 'y') if sample else P(None, 'x', 'y')
        prefix = (q1-q0, a1-a0) if sample else (q1-q0,)
        offset = (q0, a0, 0, 0) if sample else (q0, 0, 0)
        arg, output, temporary = _conversion_bytes(basis,
            prefix+(basis.n_canonical,basis.n_canonical),spec,unpack=False,operator=True)
        _admit(ledger,"read_bank_"+name,retained+arg+output,temporary,
               device_panel=arg,native_host=True,io=io)
        canonical = io.read_slab(
            name, shape=prefix + (basis.n_canonical, basis.n_canonical),
            valid_shape=prefix + (basis.n_logical, basis.n_logical),
            offset=offset, dtype=np.complex128, partition_spec=spec)
        out[name] = basis.pack_operator(canonical, spec=spec)
        out[name].block_until_ready()
        retained += output
        del canonical
    return out


def _export_spatial_header(path, source_wfn, meta, *, kind, source):
    """Append small BGW/centroid metadata to a collectively created export.

    As for zeta_q.h5, SlabIO has already created the inode with the MPI-IO
    striping policy. Serial metadata appends never create or replace it.
    """
    from file_io.mf_header import copy_mf_header
    from file_io.isdf_header import write_centroid_coordinates

    def append():
        copy_mf_header(source_wfn, path, dst_mode="a")
        with h5py.File(path, "a") as f:
            group = f.create_group(kind + "_header")
            indices = np.asarray(meta.mu_basis.canonical_indices, np.int32)
            write_centroid_coordinates(
                group, indices, indices / np.asarray(meta.fft_grid, np.float64))
            group.create_dataset("source_store", data=np.bytes_(str(source)))
            group.create_dataset("q_order", data=np.bytes_("canonical raw irreducible parents"))
            if kind == "poles":
                # b is the factorised plasmon-pole residue, B=b b†;
                # Lambda is the Omega² analogue. No causal tau weight enters b.
                f["b"] = f["factor"]  # HDF5 hard link: one payload, v1 readers unchanged.
                group.create_dataset("representation", data=np.bytes_(
                    "Wc(z)=b (z_Ry^2-Lambda_Ry2)^-1 b_dagger; "
                    "b[q,mu,spin,j], Lambda=poles2_ry2[q,j], active j<K[q]; "
                    "no tau weight, q weight or additional Coulomb factor"))
            else:
                group.create_dataset("representation", data=np.bytes_(
                    "physical Wc(q,z_i), s=z_Ry^2; fixed bank samples, not W+(tau); "
                    "Wc and dWc_ds[q,sample,mu,nu], M1 and M3[q,mu,nu]; "
                    "z_ry[role] and distinct_id[role] identify each sample"))
    rank0_transaction(path, stage="shared_pole.export_headers", write=append)


def export_shared_pole_outputs(handle, *, meta, config, mesh_xy, source_wfn,
                               run_dir, label, tables, print_fn):
    """Export current-map poles and/or fixed W samples through their writers.

    Large arrays stay XY tiled. The copy loop holds one parent (and one
    frequency for W) at a time; authentication uses the existing bounded
    parent/column panels and exchanges only row hashes. Existing packing, admission, commit and digest owners
    are reused; the source stores are immutable, including on restart.
    The bank export retains its derivative and moment companions so it is
    readable by the existing bank reader. No screening or pole fit is rerun.
    """
    source = Path(handle["path"])
    if source_wfn is None:
        _refuse("export requires the source WFN path for verbatim mf_header")
    ledger = _capacity(meta)
    model = validate_shared_pole_model(source, expected_identity=handle["identity"],
        mesh_xy=mesh_xy, capacity=ledger)
    if model["digest"] != handle["digest"]:
        _refuse("export handle differs from current model")
    basis = meta.mu_basis
    outputs = {}
    targets = {kind: Path(run_dir) / f"{label}_{kind}.h5" for kind, enabled in
               (("poles", config.write_poles), ("w", config.write_w)) if enabled}
    for path in targets.values():
        if path.exists():
            _refuse(f"export already exists: {path}; use a fresh output directory")
    bank_source = source.parent / "bank.h5"
    if config.write_w:
        if not bank_source.is_file():
            _refuse(f"write_w needs the current-map bank {bank_source}; "
                    "a model-only restart cannot supply frequency samples")
        bank_header = validate_shared_pole_bank(bank_source,
            expected_identity=handle["identity"], mesh_xy=mesh_xy, require_complete=True)
    if config.write_poles:
        path = targets["poles"]
        spec = P(None, "x", None, "y")
        shape = mesh_divisible_shape((1, basis.n_canonical, 1, model["Kmax"]), mesh_xy, spec)
        arg, output, temp = _conversion_bytes(basis, shape, spec, unpack=False)
        previous = ledger.live_stages
        with SlabIO(source, mode="r", mesh=mesh_xy) as io:
            for q, count in enumerate(model["K"]):
                row = _admit(ledger, "export_poles", arg + output + 24*shape[-1], temp,
                             device_panel=max(arg, 8*shape[-1]), native_host=True)
                ledger.live_stages = (*previous, row["stage"])
                try:
                    canonical = (io.read_slab("factor", shape=shape,
                        offset=(q, 0, 0, 0), partition_spec=spec) if model["Kmax"] else
                        jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                            out_shardings=NamedSharding(mesh_xy, spec))())
                    b = basis.pack_axis(canonical, 1, spec=spec)
                    poles = (io.read_slab("poles2_ry2", shape=(1, shape[-1]),
                        offset=(q, 0), partition_spec=P()) if model["Kmax"] else
                        jnp.ones((1, 0), jnp.float64))
                    poles = jnp.where(jnp.arange(shape[-1])[None, :] < count, poles, 1.0)
                    write_shared_pole_model(path, b, poles, np.asarray([count], np.int64),
                        q_span=(q, q+1), meta=meta, tables=tables,
                        recipe=model["recipe"], receipts=dict(identity=handle["identity"],
                            source_model_digest=model["digest"], source_store=str(source)))
                    del canonical, b, poles
                finally:
                    ledger.live_stages = previous
        _export_spatial_header(path, source_wfn, meta, kind="poles", source=source)
        outputs["poles"] = dict(path=str(path), payload_bytes=model["compact_payload_bytes"])
    if config.write_w:
        path = targets["w"]
        output_header = initialize_shared_pole_bank(path, meta=meta, tables=tables,
            recipe=bank_header["recipe"], identity=handle["identity"], mesh_xy=mesh_xy)
        previous = ledger.live_stages
        with SlabIO(bank_source, mode="r", mesh=mesh_xy) as io, \
                SlabIO(path, mode="a", mesh=mesh_xy) as output_io:
            for q in range(bank_header["bank_shape"]["nq"]):
                for field in _BANK_SAMPLE_FIELDS + _BANK_MOMENT_FIELDS:
                    sample = field in _BANK_SAMPLE_FIELDS
                    for i in range(bank_header["bank_shape"]["nsample"] if sample else 1):
                        span = (i, i+1) if sample else None
                        values = read_shared_pole_bank(io, (q, q+1), meta=meta,
                            header=bank_header, sample_span=span, fields=(field,))
                        value = values[field]
                        spec = P(None, None, "x", "y") if sample else P(None, "x", "y")
                        row = _admit(ledger, "export_w_live",
                            _local_bytes(value.shape, value.dtype, mesh_xy, spec))
                        ledger.live_stages = (*previous, row["stage"])
                        try:
                            prepared = _prepare_bank_write(output_header,
                                q_span=(q, q+1), sample_span=span, meta=meta,
                                mesh_xy=mesh_xy, **values)
                            _write_bank_payload(output_io, output_header, meta, prepared)
                            del prepared
                            del values, value
                        finally:
                            ledger.live_stages = previous
            _write_bank_masks(output_io, output_header)
        _complete_bank(path, output_header)
        _export_spatial_header(path, source_wfn, meta, kind="w", source=bank_source)
        outputs["w"] = dict(path=str(path), payload_bytes=bank_header["payload_bytes"])
    for kind, receipt in outputs.items():
        print_fn(f"write_{kind}: {receipt['path']}; payload={receipt['payload_bytes']} bytes")
    return outputs
