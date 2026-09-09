"""Canonical shared real-pole model and construction scratch I/O.

The physical convention is Wc(z) = C (z_Ry**2 - Lambda)**-1 C†.
C is never divided by sqrt(2 Omega) here. Bulk payloads cross SlabIO only;
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

from common.collectives import rank0_transaction, psum_replicate
from file_io.slab_io import SlabIO
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
        _refuse("stale input/SC identity")


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
        "normalization": "Wc=C/(z_Ry^2-Lambda_Ry2)*C_dagger",
        "units": {"factor": "Ry^(3/2)", "poles2_ry2": "Ry^2"},
        "representation": "scalar-trs-even-s", "parent_convention": "raw-parent",
        "n_q_irr": qt.n_q_ibz, "n_q_full": qt.n_q_full,
        "n_mu_logical": basis.n_logical, "nspinor": 1,
        "centroid_digest": centroid_hash,
        "grid": np.asarray(meta.kgrid).tolist(),
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


def _check_factor(C, poles2, K):
    """Gate the physical active prefix and exact inactive sentinel."""
    if C.dtype != np.complex128 or poles2.dtype != np.float64 or K.dtype != np.int64:
        _refuse(f"dtypes {(C.dtype, poles2.dtype, K.dtype)}")
    if C.ndim != 4 or poles2.shape != (C.shape[0], C.shape[3]) or K.shape != (C.shape[0],):
        _refuse("factor/poles/count shape mismatch")
    if np.any(K < 0) or np.any(K > C.shape[3]):
        _refuse("K outside factor column capacity")
    active = jnp.arange(C.shape[3])[None, :] < jnp.asarray(K)[:, None]
    ok = (jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(poles2))
          & jnp.all(jnp.where(active, poles2 > 0, poles2 == 1))
          & jnp.all(jnp.where(active[:, None, None, :], True, C == 0))
          & jnp.all(jnp.where(active[:, 1:], poles2[:, 1:] >= poles2[:, :-1], True)))
    if not bool(ok):
        _refuse("nonfinite, unsorted/nonpositive active poles or invalid inactive sentinel")


def write_shared_pole_model(path, C, poles2, K, *, q_span, meta, tables,
                            recipe, receipts):
    """Stage a bounded q batch and finalize automatically at complete K census.

    Parameters
    ----------
    C : jax.Array, complex128, (b, mu_p, spin, Kp)
        Physical Ry^(3/2) factor; NamedSharding(mesh_xy,P(None,'x',None,'y')).
    poles2 : jax.Array, float64, (b, Kp)
        Sorted squared poles in Ry², same padded column capacity as C.
    K : array, int64, (b,)
        Physical active counts; inactive C=0 and poles2=1 exactly.
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
    header = _metadata(meta, tables, recipe, receipts["identity"])
    mesh = meta.mu_basis.mesh_xy
    want = NamedSharding(mesh, P(None, "x", None, "y"))
    if not isinstance(C, jax.Array) or not C.sharding.is_equivalent_to(want, 4):
        _refuse("constructor factor is not the declared XY handoff")
    if C.shape[1:3] != (meta.mu_basis.n_packed, 1):
        _refuse("factor does not use the current packed centroid basis")
    K = np.asarray(K)
    _check_factor(C, poles2, K)
    lo, hi = _span(q_span, header["n_q_irr"], "q_span")
    if hi - lo != C.shape[0]:
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
    if width == 0:
        _refuse("empty batch has no supported physical pole")
    name = f"staging/q{lo}_{hi}"
    canonical = meta.mu_basis.unpack_axis(C, 1)
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
        io.create_dataset(name + "/factor", shape=(hi-lo, header["n_mu_logical"], 1, width), dtype=np.complex128)
        io.create_dataset(name + "/poles2", shape=(hi-lo, width), dtype=np.float64)
        io.write_slab(name + "/factor", canonical)
        io.write_slab(name + "/poles2", poles2)
        header["written_q"][lo:hi] = [True] * (hi-lo)
        header["K"][lo:hi] = K.tolist()
        header["batches"].append({"lo": lo, "hi": hi, "width": width, "name": name})
        header["construction_receipts"].append({"q_span": [lo, hi], "receipt": receipts})
        _write_metadata(io, header)
        io.write_attr("written_q", np.asarray(header["written_q"], np.int8))
        _write_header(io, header)
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
            path, expected_identity=expected_identity, mesh_xy=meta.mu_basis.mesh_xy)
    return _finalize_model(path, meta=meta, header=header)


def _finalize_model(path, *, meta, header):
    mesh, basis = meta.mu_basis.mesh_xy, meta.mu_basis
    nq, nmu = header["n_q_irr"], header["n_mu_logical"]
    kmax = max(header["K"])
    header["Kmax"] = kmax
    header["compact_payload_bytes"] = nq * (16*nmu*kmax + 8*kmax + 8)
    header["staging_payload_bytes"] = sum((v["hi"]-v["lo"]) * v["width"] * (16*nmu+8) for v in header["batches"])
    header["peak_payload_bytes"] = header["compact_payload_bytes"] + header["staging_payload_bytes"]
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.create_dataset("factor", shape=(nq, nmu, 1, kmax), dtype=np.complex128)
        io.create_dataset("poles2_ry2", shape=(nq, kmax), dtype=np.float64)
        for batch in header["batches"]:
            # One parent at a time; no all-parent carrier to discover Kmax.
            for q in range(batch["lo"], batch["hi"]):
                factor = io.read_slab(batch["name"] + "/factor",
                    shape=(1, basis.n_canonical, 1, kmax),
                    offset=(q-batch["lo"], 0, 0, 0), partition_spec=P(None,"x",None,None))
                poles = io.read_slab(batch["name"] + "/poles2",
                    shape=(1, kmax), offset=(q-batch["lo"],0), partition_spec=P())
                active = jnp.arange(kmax)[None,:] < header["K"][q]
                poles = jnp.where(active, poles, 1.0)
                io.write_slab("factor", factor, offset=(q,0,0,0))
                io.write_slab("poles2_ry2", poles, offset=(q,0))
                io.sync_writes()
        io.write_attr("K", np.asarray(header["K"], np.int64))
        _write_header(io, header)
    header["digest"] = _model_digest(path, header, mesh)
    def finish():
        with h5py.File(path, "a") as f:
            set_commit_state(f, False)
            del f["staging"]
            header["finalized"] = True
            del f["header_json"]
            f.create_dataset("header_json", data=np.bytes_(_json(header)))
            f.create_dataset("final_commit", data=np.bytes_(header["digest"]))
            set_commit_state(f, True)
    rank0_transaction(path, stage="shared_pole.finalize", write=finish)
    return _read_header(path)


def _model_digest(path, header, mesh):
    """Grid-independent SHA256 of metadata and canonical row digests.

    Read one q factor face at a time through SlabIO. Only row hashes (32 bytes
    per centroid) are exchanged; a full factor is never gathered onto a rank.
    """
    identity = {k:v for k,v in header.items() if k not in
                ("digest", "finalized", "batches", "staging_payload_bytes", "peak_payload_bytes")}
    digest = hashlib.sha256(_json(identity).encode())
    nmu, kmax = header["n_mu_logical"], header["Kmax"]
    ncan = ((nmu + int(mesh.size)-1)//int(mesh.size))*int(mesh.size)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        for q in range(header["n_q_irr"]):
            C = io.read_slab("factor", shape=(1,ncan,1,kmax), offset=(q,0,0,0),
                             partition_spec=P(None,"x",None,None))
            poles = io.read_slab("poles2_ry2", shape=(1,kmax), offset=(q,0), partition_spec=P())
            _check_factor(C, poles, np.asarray([header["K"][q]], np.int64))
            row_hash = np.zeros((nmu,32), np.uint32)
            for shard in C.addressable_shards:
                if shard.replica_id != 0:
                    continue
                start = shard.index[1].start or 0
                local = np.asarray(shard.data)[0,:,0,:]
                for i in range(min(local.shape[0], nmu-start)):
                    row_hash[start+i] = np.frombuffer(hashlib.sha256(
                        np.asarray(local[i], dtype="<c16").tobytes()).digest(), np.uint8)
            row_hash = psum_replicate(row_hash, mesh)
            digest.update(row_hash.astype(np.uint8).tobytes())
            digest.update(np.asarray(poles, dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_shared_pole_model(path, *, expected_identity, mesh_xy):
    """Refuse partial, stale, malformed or changed payload; return header.

    Validation is collective and bounded to one irreducible parent face. It
    verifies storage integrity, not the constructor's physical gate claims.
    """
    header = _read_header(path)
    _check_identity(header["identity"], expected_identity)
    if header["schema"] != SCHEMA or not header["finalized"] or not all(header["written_q"]):
        _refuse("missing final shared-pole commit or incomplete q census")
    qt = QirrTables(**{k:np.asarray(header["qirr"][k]) for k in _TABLE_KEYS},
                    n_sym_spatial=header["qirr"]["n_sym_spatial"])
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
    if _model_digest(path, header, mesh_xy) != header["digest"]:
        _refuse("model payload/identity digest mismatch")
    return header


def read_shared_pole_faces(io, q_span, *, meta, header, column_span=None):
    """Read canonical row faces and pack once at the I/O boundary.

    Returns C_X, C_Y, poles2, K with shapes (b,mu_p,spin,Kcap),
    (b,mu_p,spin,Kcap), (b,Kcap), (b,). Faces use P(None,'x',None,None)
    and P(None,'y',None,None); poles and int64 counts are replicated. K is
    the active count *within the returned column slice*, so every consumer
    can mask with arange(Kcap)<K even when column_span starts above zero.
    """
    if header["schema"] != SCHEMA or not header["finalized"]:
        _refuse("face reader requires a validated finalized model")
    basis = _check_basis(meta, header)
    lo, hi = _span(q_span, header["n_q_irr"], "q_span")
    c0, c1 = _span(column_span or (0,header["Kmax"]), header["Kmax"], "column_span")
    counts = jnp.asarray(np.clip(np.asarray(header["K"][lo:hi])-c0,0,c1-c0), dtype=jnp.int64)
    active = jnp.arange(c1-c0)[None,:] < counts[:,None]
    faces = []
    for axis in ("x", "y"):
        spec = P(None,axis,None,None)
        C = io.read_slab("factor", shape=(hi-lo,basis.n_canonical,1,c1-c0),
                         offset=(lo,0,0,c0), partition_spec=spec)
        C = basis.pack_axis(C, 1, spec=spec)
        faces.append(jnp.where(active[:,None,None,:] & jnp.asarray(
            basis.active_mask)[None,:,None,None], C, 0.0))
    poles = io.read_slab("poles2_ry2", shape=(hi-lo,c1-c0), offset=(lo,c0), partition_spec=P())
    return (*faces, jnp.where(active,poles,1.0), counts)


# Append to file_io/shared_pole_store.py. Uses its common metadata helpers.
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
        for name in ("z_ry", "role", "distinct_id", "held"):
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
        for name in ("z_ry", "role", "distinct_id", "held"):
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
    if header.get("complete"):
        _refuse("completed scratch bank is immutable")
    # A process may stop after payload/masks close but before the final stamp.
    # A payload-free call retries only that metadata transaction.
    if (np.asarray(header["sample_written"], dtype=bool).all()
            and np.asarray(header["moment_written"], dtype=bool).all()
            and all(value is None for value in (Wc, dWc_ds, M1, M3))):
        header["complete"] = True
        header["final_commit"] = hashlib.sha256(_json(header).encode()).hexdigest()
        _stamp_header(path, header, "shared_pole_bank.complete")
        return header
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
        if not bool(jnp.all(jnp.isfinite(array))):
            _refuse(f"scratch {name} contains nonfinite values")
    with SlabIO(path, mode="a", mesh=mesh_xy) as io:
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
            if sample:
                sample_mask[q0:q1, a0:a1, _BANK_SAMPLE_FIELDS.index(name)] = True
            else:
                moment_mask[q0:q1, _BANK_MOMENT_FIELDS.index(name)] = True
        header["sample_written"] = sample_mask.tolist()
        header["moment_written"] = moment_mask.tolist()
        io.write_attr("sample_written", sample_mask)
        io.write_attr("moment_written", moment_mask)
        _write_header(io, header)
    if sample_mask.all() and moment_mask.all():
        header["complete"] = True
        header["final_commit"] = hashlib.sha256(_json(header).encode()).hexdigest()
        _stamp_header(path, header, "shared_pole_bank.complete")
    return header


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
    for name in fields:
        sample = name in _BANK_SAMPLE_FIELDS
        spec = P(None, None, 'x', 'y') if sample else P(None, 'x', 'y')
        prefix = (q1-q0, a1-a0) if sample else (q1-q0,)
        offset = (q0, a0, 0, 0) if sample else (q0, 0, 0)
        canonical = io.read_slab(
            name, shape=prefix + (basis.n_canonical, basis.n_canonical),
            valid_shape=prefix + (basis.n_logical, basis.n_logical),
            offset=offset, dtype=np.complex128, partition_spec=spec)
        out[name] = basis.pack_operator(canonical, spec=spec)
    return out
