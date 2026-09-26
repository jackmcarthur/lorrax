"""Canonical shared real-pole model and construction scratch I/O.

The physical convention is Wc(z) = b (z_Ry**2 - Lambda)**-1 b†.
The on-disk dataset remains ``factor`` for compatibility; it stores b.
b is never divided by sqrt(2 Omega) here. Bulk payloads cross SlabIO only;
centroid packing belongs exclusively to these I/O boundaries. All entry points
are collective over the supplied mesh, including validation and publication.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from contextlib import contextmanager
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from runtime.padding import combined_divisor, padded_axis
from common import timing
from common.collectives import (device_put_process_local, process_rank, psum_replicate,
                                rank0_transaction)
from file_io.slab_io import SlabIO, mesh_divisible_shape
from file_io.commit_state import agree_io_refusal, assert_committed, set_commit_state
from symmetry_maps import QirrTables, validate_qirr_tables

SCHEMA = "lorrax.shared-real-pole.v1"
BANK_SCHEMA = "lorrax.shared-real-pole-bank.v3"
SECTOR_SCHEMA = "lorrax.shared-real-pole-sectors.v1"
_TABLE_KEYS = ("irr_idx_q", "sym_idx_q", "q_irr_frac", "sym_perm", "L_table")
_IDENTITY_KEYS = ("iteration_id", "hamiltonian", "energies", "occupations",
                  "wavefunctions", "centroids")


def _refuse(message):
    raise ValueError(f"GATE shared_pole_store: {message}; want: authenticated "
                     "current-map canonical scalar model; fix: rebuild the artifact")


def charge_representation(meta):
    """True when the bank operator is the spin-traced charge response.

    The response stream traces both spinor endpoints into one ``[q, mu, mu]``
    charge operator. A four-component kinetic-balance carrier retains the
    two-component source WFN identity; it does not add current vertices here.
    """
    nspinor = int(meta.nspinor)
    source_spinor = int(getattr(meta, "nspinor_wfnfile", nspinor))
    return (nspinor == 1 or (nspinor == 2 and source_spinor == 2)
            or (nspinor == 4 and source_spinor == 2))


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
    if "capacity_geometry" in header:
        expected = dict(header["capacity_geometry"], px=int(mesh.shape["x"]),
                        py=int(mesh.shape["y"]))
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


def _conversion_bytes(basis, shape, spec, *, unpack, operator=False, axis=1):
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
              else basis._axis_kernel(axis, spec, unpack))
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
        if isinstance(x, ResidentSectorModel):
            return str(x)
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
    if isinstance(path, (ResidentBankPayload, ResidentSectorModel)):
        if path.header_json is None:
            _refuse(f"{path} has no committed header")
        return json.loads(path.header_json)
    with h5py.File(path, "r") as f:
        assert_committed(f, path=path)
        if "header_json" not in f:
            _refuse("missing header_json")
        raw = f["header_json"][()]
        return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def _write_header(io, header):
    io.write_attr("header_json", np.bytes_(_json(header)))


def _read_staging_header(path):
    """Close every serial reader before any rank can reopen the staged writer."""
    header = error = None
    try:
        if (path.header_json is not None if isinstance(path, ResidentSectorModel)
                else Path(path).exists()):
            header = _read_header(path)
    except BaseException as exc:
        error = exc
    agree_io_refusal(error, path=path, stage="shared_pole.staging_header")
    return header


def _stamp_header(path, header, stage):
    # Only metadata, after every collective handle has closed.
    if isinstance(path, ResidentBankPayload):
        path.header_json = _json(header)
        return
    def publish():
        with h5py.File(path, "a") as f:
            if "header_json" in f:
                del f["header_json"]
            f.create_dataset("header_json", data=np.bytes_(_json(header)))
            f.flush()
    rank0_transaction(path, stage=stage, write=publish)


def _check_basis(meta, header, basis=None):
    """Authenticate scientific centroid order independently of mesh padding."""
    basis = meta.mu_basis if basis is None else basis
    digest = hashlib.sha256(np.asarray(
        basis.canonical_indices, dtype="<i4").tobytes()).hexdigest()
    logical = (header["photon_layout"]["logical_extents"][0]
               if "photon_layout" in header else header["n_mu_logical"])
    if "photon_layout" in header:
        side = header["photon_layout"]["mesh_side"]
        if any(int(basis.mesh_xy.shape[a]) != side for a in ("x", "y")):
            _refuse("photon scratch mesh differs from its recorded packed ordering")
    if (basis.n_logical != logical
            or int(meta.nspinor) != header["nspinor"]
            or digest != header["centroid_digest"]):
        _refuse("reader/writer logical centroid or spin identity changed")
    return basis


def _metadata(meta, tables, recipe, identity, ordered=None, *, basis=None, sector=None, photon=False):
    """Authenticate small scientific identities; no tensor data is gathered.

    ``tables`` is a plain mapping with canonical ``qirr`` (QirrTables),
    ``q_irr_full_idx`` and the generating ``sym`` (SymMaps). The endpoint
    basis owns logical centroid identity and I/O packing. Sector factors
    use the complete bispinor map's capacity ledger.
    """
    _check_identity(identity, identity)
    basis = meta.mu_basis if basis is None else basis
    if photon:
        if int(meta.nspinor) != 4 or sector is not None or ordered is not None:
            _refuse("photon metadata requires a four-component scratch bank")
    elif sector is None:
        if not charge_representation(meta):
            _refuse(f"unsupported Nspinor={meta.nspinor}")
    elif sector not in ("CC", "TT", "CT_C", "CT_T") or int(meta.nspinor) != 4:
        _refuse("sector factors require Nspinor=4 and CC/TT/CT_C/CT_T")
    sym = tables["sym"]
    # A bank (ordered=None) follows the measured TRS state; a model store
    # states its representation and must agree with it.
    bank = ordered is None
    if bank:
        ordered = photon or not bool(sym.trs_allowed)
    elif (bool(sym.trs_allowed) == bool(ordered)
          and not (ordered and sector in ("CT_C", "CT_T"))):
        _refuse("ordered representation requires authenticated broken TRS" if ordered
                else "TRS-broken representation is unsupported")
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
    if sector is not None:
        recipe = dict(recipe, operator_realization="raw-sector-endpoint-v1")
    header = {
        "schema": SCHEMA, "identity": identity, "recipe": recipe,
        "recipe_hash": hashlib.sha256(_json(recipe).encode()).hexdigest(),
        # Keep the v1 disk spelling for existing models; its C denotes b.
        "normalization": ("Wc_q=sum_q C C_dagger/(2W(z_Ry-W))-sum_(-q) conj(C) C^T/(2W(z_Ry+W)), W=sqrt(Lambda_Ry2)"
                          if ordered and not bank else "Wc=C/(z_Ry^2-Lambda_Ry2)*C_dagger"),
        "units": {"factor": "Ry^(3/2)", "poles2_ry2": "Ry^2"},
        # An ordered bank is not even in s; consumers that need the TRS form
        # (constructor, operator realizer) refuse this representation by name.
        # An ordered model store keeps positive poles per parent.
        "representation": (("charge-ordered-z" if bank else "scalar-ordered-ph")
                           if ordered else "scalar-trs-even-s"),
        "parent_convention": "raw-parent",
        "n_q_irr": qt.n_q_ibz, "n_q_full": qt.n_q_full,
        "n_mu_logical": basis.n_logical, "nspinor": int(meta.nspinor),
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
    if sector is not None:
        header.update(sector=sector, factor_components=3 if sector in ("TT", "CT_T") else 1,
                      capacity_geometry=dict(_capacity(meta).geometry))
        header["representation"] = "sector-ordered-ph" if ordered else "sector-trs-even-s"
    if ordered:
        header["ordered"] = True
    return header


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
                            recipe, receipts, ordered=False, basis=None, sector=None):
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
    basis : PackedCentroidBasis, optional
        Scientific endpoint basis; defaults to meta.mu_basis. Current
        endpoints supply the existing transverse basis. Capacity remains
        charged to the complete map's ledger, not a fictitious sector map.
    sector : str, optional
        CC, TT, CT_C or CT_T on Nspinor=4 maps. TT and CT_T have three
        component rows. CT_C and CT_T store the two factors of one CT
        model; the construction owner must bind their common pole census.

    Returns
    -------
    dict
        Header; finalized and digest are published only after all parents close.
        Staging plus final datasets use at most twice the compact payload bytes.
    """
    with timing.section("staging"):
        basis = meta.mu_basis if basis is None else basis
        header = _metadata(meta, tables, recipe, receipts["identity"], ordered,
                           basis=basis, sector=sector)
        components = header.get("factor_components", 1)
        mesh = basis.mesh_xy
        want = NamedSharding(mesh, P(None, "x", None, "y"))
        if not isinstance(b, jax.Array) or not b.sharding.is_equivalent_to(want, 4):
            _refuse("constructor factor is not the declared XY handoff")
        if b.shape[1:3] != (basis.n_packed, components):
            _refuse("factor does not use the current packed centroid basis")
        ledger = _capacity(meta)
        arg, output, temporary = _conversion_bytes(basis, b.shape, want.spec, unpack=True)
        # One factor-sized envelope covers eager finite/sentinel check scratch.
        _admit(ledger, "write_model", output, temporary+arg+3*int(poles2.size)*8,
               device_panel=max(arg,output,8*int(poles2.size)), native_host=True)
        K = np.asarray(K)
        _check_factor(b, poles2, K)
        lo, hi = _span(q_span, header["n_q_irr"], "q_span")
        if hi - lo != b.shape[0]:
            _refuse("q_span does not match factor batch")
        previous = _read_staging_header(path)
        if previous is not None:
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
        width = _k_extent(meta, header, int(K.max(initial=0)), record=False)
        name = f"staging/q{lo}_{hi}"
    with timing.section("canonical_basis_conversion_and_packing"):
        canonical = basis.unpack_axis(b, 1)
    if isinstance(path, ResidentSectorModel):
        # The staged batch stays on the devices; its dataset extents are the
        # file's, so finalization trims exactly what the file would store.
        path.staging[name] = (canonical, poles2, width)
        header["written_q"][lo:hi] = [True] * (hi-lo)
        header["K"][lo:hi] = K.tolist()
        header["batches"].append({"lo": lo, "hi": hi, "width": width, "name": name})
        header["construction_receipts"].append({"q_span": [lo, hi], "receipt": receipts})
        path.header_json = _json(header)
        del canonical
        if all(header["written_q"]):
            return _finalize_model(path, meta=meta, header=header, basis=basis)
        return header
    if previous is None:
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
            io.create_dataset(name + "/factor", shape=(hi-lo, header["n_mu_logical"], components, width), dtype=np.complex128)
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
        return _finalize_model(path, meta=meta, header=header, basis=basis)
    return header


def finalize_shared_pole_model(path, *, meta, expected_identity, basis=None):
    """Resume finalization from a successfully closed, complete staging census.

    A native write failure retains the global incomplete marker and refuses;
    this retries the recoverable boundary after all staged batches closed.
    """
    header = _read_staging_header(path)
    if header is None:
        _refuse("missing staged model")
    _check_identity(header["identity"], expected_identity)
    basis = _check_basis(meta, header, basis)
    if header["schema"] != SCHEMA or not all(header["written_q"]):
        _refuse("finalization requires every staged parent")
    if header["finalized"]:
        return validate_shared_pole_model(
            path, expected_identity=expected_identity, mesh_xy=meta.mu_basis.mesh_xy,
            capacity=_capacity(meta))
    return _finalize_model(path, meta=meta, header=header, basis=basis)


#: Headroom of the SC run's held pole-column extent over the live Kmax it is
#: set from. After map 1 the sector models' Kmax grows by about 1% per map (Fe
#: 4^3 bispinor CC/TT 1328 -> 1335 -> 1338, CT 723 -> 729), and an exact extent
#: recompiled every store, read and Sigma-window program each map; 3% holds
#: that drift for several maps and pads less than the eighth-octave ladder it
#: replaces (up to 12.5%).
_K_HEADROOM = 0.03


def _k_extent(meta, header, live, *, record):
    """The model's pole-column extent: the live Kmax, or the SC run's held one.

    An SC map past map 0 binds ``meta.shared_pole_k_capacity`` (a dict the
    quadrature session keeps). Map 1 sets the extent to its live Kmax plus
    :data:`_K_HEADROOM`; later maps keep it while the live Kmax fits and grow
    it (again with headroom) only when it does not, noting the growth in
    ``held["_events"]`` for the SC log. So a drifting Kmax keeps one dataset
    shape and every store, read and Sigma program is reused (Fe 4^3 charge:
    746 -> 736 -> 734 recompiled finalize, SlabIO and census programs at map
    2). Columns past each parent's K are zero factors and unit poles, as
    before. No binding (a one-shot, map 0): the live Kmax.
    """
    held = getattr(meta, "shared_pole_k_capacity", None)
    if held is None:
        return int(live)
    sector = header.get("sector")
    key = "CT" if sector in ("CT_C", "CT_T") else str(sector)
    before = int(held.get(key, 0))
    if int(live) <= before:
        return before
    if not record:
        return int(live)
    extent = int(np.ceil(int(live) * (1.0 + _K_HEADROOM)))
    if before:
        held.setdefault("_events", []).append(
            f"shared-pole K extent ({key}): live Kmax {int(live)} exceeds the held "
            f"{before}; grown to {extent}")
    held[key] = extent
    return extent


@timing.timed("shared_pole_store.finalize")
def _finalize_model(path, *, meta, header, basis=None):
    basis = meta.mu_basis if basis is None else basis
    mesh = basis.mesh_xy
    components = header.get("factor_components", 1)
    nq, nmu = header["n_q_irr"], header["n_mu_logical"]
    kmax = _k_extent(meta, header, max(header["K"]), record=True)
    panel = 16*components*basis.n_canonical*((kmax+int(mesh.shape["y"])-1)//int(mesh.shape["y"]))/int(mesh.shape["x"])
    batch_width = max(v["hi"] - v["lo"] for v in header["batches"])
    _admit(_capacity(meta), "finalize", batch_width*(int(panel)+24*kmax),
           device_panel=batch_width*max(int(panel),8*kmax), native_host=True)
    header["Kmax"] = kmax
    header["compact_payload_bytes"] = nq * (16*nmu*components*kmax + 8*kmax + 8)
    header["staging_payload_bytes"] = sum((v["hi"]-v["lo"]) * v["width"] * (16*nmu*components+8) for v in header["batches"])
    header["peak_payload_bytes"] = header["compact_payload_bytes"] + header["staging_payload_bytes"]
    if isinstance(path, ResidentSectorModel):
        path.finalize_payload(header, n_canonical=basis.n_canonical)
        # In-process payload: nothing can drift from the header it was built
        # with, so the digest binds the metadata only (no payload rehash; the
        # writer already checked every factor). Files keep the payload digest.
        identity = {k: v for k, v in header.items() if k not in
                    ("digest", "finalized", "batches", "staging_payload_bytes", "peak_payload_bytes")}
        header["digest"] = "resident:" + hashlib.sha256(_json(identity).encode()).hexdigest()
        header["finalized"] = True
        path.header_json = _json(header)
        path.final_commit = header["digest"]
        return _read_header(path)
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.create_dataset("factor", shape=(nq, nmu, components, kmax), dtype=np.complex128)
        io.create_dataset("poles2_ry2", shape=(nq, kmax), dtype=np.float64)
        for batch in header["batches"]:
            # Preserve the constructor's admitted q batch through finalization.
            # Kmax is already known from the committed census; no all-q carrier.
            lo, hi = batch["lo"], batch["hi"]
            spec = P(None, "x", None, "y")
            read_shape = mesh_divisible_shape(
                (hi-lo, basis.n_canonical, components, kmax), mesh, spec)
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


def _covering_windows(total, span):
    """``(start, first_new)`` of ``span``-wide windows covering ``range(total)``.

    The last window is shifted back to end at ``total``, so every window has
    one shape; ``first_new`` is its first index no earlier window covered.
    """
    starts = list(range(0, total, span))
    if total > span:
        starts[-1] = total - span
    return [(start, 0 if i == 0 else starts[i - 1] + span - start)
            for i, start in enumerate(starts)]


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
    components = header.get("factor_components", 1)
    ncan = ((nmu + int(mesh.size)-1)//int(mesh.size))*int(mesh.size)
    column_cap = max(1, (kmax + int(mesh.shape["y"])-1)//int(mesh.shape["y"]))
    # One panel shape per model: an SC run holds Kmax (``_k_extent``), and
    # the last parent batch and column panel are shifted back to overlap
    # their predecessors. An overlapped parent or column is validated twice
    # and hashed once.
    width = min(kmax, column_cap)
    panel = 16*components*ncan*width//int(mesh.shape["x"])
    batch_limit = min(int(mesh.size), header["n_q_irr"])
    _admit(capacity, "digest", batch_limit*(panel+8*kmax+256*nmu*components),
           batch_limit*(panel+24*kmax), host_payload=batch_limit*panel,
           device_panel=batch_limit*panel,
           host_metadata=batch_limit*(256*nmu*components+8*kmax), native_host=True)
    with open_shared_pole_model(path, mesh_xy=mesh) as io:
        for q0, q_new in _covering_windows(header["n_q_irr"], batch_limit):
            q1 = q0+batch_limit
            batch = q1-q0
            active_counts = np.asarray(header["K"][q0:q1], np.int64)
            if kmax == 0:
                if np.any(active_counts != 0):
                    _refuse("nonzero K in empty model")
                for _ in range(q_new, batch):
                    digest.update(hashlib.sha256(b"").digest() * (nmu*components))
                continue
            with timing.section('pole_read'):
                poles = io.read_slab("poles2_ry2", shape=(batch,kmax), offset=(q0,0), partition_spec=P())
                host_poles = np.asarray(poles)
            for row, count in zip(host_poles, active_counts):
                if np.any(np.diff(row[:count]) < 0):
                    _refuse("active poles are unsorted across column panels")
            hashers = [{} for _ in range(batch)]
            for c0, c_new in _covering_windows(kmax, width):
                c1 = c0+width
                with timing.section('factor_read_and_validation'):
                    b = io.read_slab("factor", shape=(batch,ncan,components,width), offset=(q0,0,0,c0),
                                     partition_spec=P(None,"x",None,None))
                    counts = np.clip(active_counts-c0, 0, width)
                    _check_factor(b, host_poles[:,c0:c1], counts)
                with timing.section('host_digest_hashing'):
                    for shard in b.addressable_shards:
                        if shard.replica_id != 0:
                            continue
                        start = shard.index[1].start or 0
                        local = np.asarray(shard.data)
                        for q in range(q_new, batch):
                            for i in range(min(local.shape[1], nmu-start)):
                                for component in range(components):
                                    row = (start+i)*components+component
                                    hasher = hashers[q].setdefault(row, hashlib.sha256())
                                    hasher.update(np.asarray(local[q,i,component,c_new:], dtype="<c16").tobytes())
                    shard = local = None
                    del b
            row_hash = np.zeros((batch,nmu*components,32), np.uint32)
            for q in range(batch):
                for row, hasher in hashers[q].items():
                    row_hash[q,row] = np.frombuffer(hasher.digest(), np.uint8)
            with timing.section('digest_reduction'):
                row_hash = psum_replicate(row_hash, mesh)
                # Preserve the original q-major row-hashes then poles byte stream.
                for q in range(q_new, batch):
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
        sector = header.get("sector")
        components = header.get("factor_components", 1)
        if sector is not None:
            if (sector not in ("CC", "TT", "CT_C", "CT_T") or header["nspinor"] != 4
                    or components != (3 if sector in ("TT", "CT_T") else 1)
                    or "capacity_geometry" not in header):
                _refuse("invalid sector endpoint metadata")
        elif components != 1:
            _refuse("component factors require an explicit sector identity")
        qt = shared_pole_qirr_tables(header)
        validate_qirr_tables(qt, header["n_q_irr"], header["n_mu_logical"])
        if isinstance(path, ResidentSectorModel):
            # One in-process copy: the typed tables have no second record to
            # drift from; the payload extents and commit must still agree.
            if (path.logical("factor") != (header["n_q_irr"], header["n_mu_logical"],
                                           components, header["Kmax"])
                    or path.logical("poles2_ry2") != (header["n_q_irr"], header["Kmax"])
                    or path.final_commit != header["digest"]):
                _refuse("resident model payload/commit mismatch")
        else:
            with h5py.File(path, "r") as f:
                for name, shape, dtype in (
                    ("factor", (header["n_q_irr"],header["n_mu_logical"],header.get("factor_components",1),header["Kmax"]), np.complex128),
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
    agree_io_refusal(error, path=path, stage="shared_pole_model/metadata")
    if capacity is None:
        # No tensor allocation is permitted without admission. This receipt
        # must not be mistaken for payload authentication by a restart caller.
        return dict(header, validation_receipt={"status":"NOT_MEASURED",
                    "scope":"metadata only; payload digest not authenticated"})
    if isinstance(path, ResidentSectorModel):
        return header  # bound by object identity and final_commit above
    if _model_digest(path, header, mesh_xy, capacity=capacity) != header["digest"]:
        _refuse("model payload/identity digest mismatch")
    return header


def write_shared_pole_sector_manifest(path, *, models, bank, identity, receipts, mesh_xy):
    """Publish one immutable handle only after all four endpoint stores close.

    ``models`` maps CC/TT/CT_C/CT_T to (path, finalized header). The bank
    supplies the independently retained W_infinity-V constant in physical Ry.
    This small manifest binds resources; bulk reads keep their SlabIO owner.
    """
    if set(models) != {'CC','TT','CT_C','CT_T'}:
        _refuse('sector publication requires CC, TT and both CT endpoints')
    handles={}
    for sector,(filename,header) in models.items():
        _check_identity(header['identity'],identity)
        if (not header.get('finalized') or header.get('sector')!=sector or not header.get('digest')
                or not header.get('ordered')
                or header['recipe'].get('operator_realization')!='raw-sector-endpoint-v1'):
            _refuse(f'unfinalized or mistyped sector {sector}')
        handles[sector]=dict(path=(filename if isinstance(filename,ResidentSectorModel)
                                   else str(Path(filename).resolve())),identity=identity,
                             digest=header['digest'],K=header['K'])
    left,right=models['CT_C'][1],models['CT_T'][1]
    for key in ('K','Kmax','q_irr_full_idx','ordered'):
        if _json(left.get(key))!=_json(right.get(key)):
            _refuse(f'CT endpoint publication disagrees on {key}')
    bank_header=validate_shared_pole_bank(bank['path'],expected_identity=identity,
                              mesh_xy=mesh_xy,require_complete=True)
    if isinstance(bank['path'],ResidentBankPayload):
        # The resident payload is released after construction; the Sigma
        # consumer's W_infinity-V constant is published to its own file.
        constant=write_bank_constant(bank['path'],Path(path).with_name('constant.h5'),
                                     header=bank_header,mesh_xy=mesh_xy)
    else:
        constant=dict(path=str(Path(bank['path']).resolve()),identity=identity,field='constant')
    content=dict(schema=SECTOR_SCHEMA,representation='sector-ordered-ph',identity=identity,
        operator_realization='raw-sector-endpoint-v1',
        sectors=handles,constant=constant,construction=receipts)
    digest=hashlib.sha256(_json(content).encode()).hexdigest()
    header=dict(content,digest=digest)
    path=Path(path)
    def publish():
        if path.exists():
            _refuse('finalized sector manifests are immutable')
        path.write_text(_json(header)+'\n')
    rank0_transaction(path,stage='shared_pole.sector_manifest',write=publish)
    return dict(path=str(path.resolve()),identity=identity,digest=digest,
                representation=header['representation'],sectors=handles,constant=content['constant'])


def validate_shared_pole_sector_manifest(path, *, expected_identity, mesh_xy, capacity=None,
                                         resident=None):
    """Authenticate the manifest and each bound current-map model resource.

    ``resident`` maps sectors to this process's ResidentSectorModel objects;
    each must be the one the manifest names, and it replaces that name in the
    returned sector handles.
    """
    header=None
    error=None
    try:
        header=json.loads(Path(path).read_text())
    except Exception as exc:
        error=exc
    agree_io_refusal(error,path=path,stage='shared_pole.sector_manifest.read')
    _check_identity(header.get('identity'),expected_identity)
    if (header.get('schema')!=SECTOR_SCHEMA or header.get('representation')!='sector-ordered-ph'
            or header.get('operator_realization')!='raw-sector-endpoint-v1'):
        _refuse('unsupported sector manifest')
    content={k:v for k,v in header.items() if k!='digest'}
    if hashlib.sha256(_json(content).encode()).hexdigest()!=header.get('digest'):
        _refuse('sector manifest digest mismatch')
    if set(header.get('sectors',{}))!={'CC','TT','CT_C','CT_T'}:
        _refuse('incomplete sector manifest')
    model_headers={}
    for sector,handle in header['sectors'].items():
        _check_identity(handle['identity'],expected_identity)
        if sector in (resident or {}):
            if str(resident[sector])!=handle['path']:
                _refuse(f'resident sector model is not the published {sector}')
            handle['path']=resident[sector]
        model=validate_shared_pole_model(handle['path'],expected_identity=expected_identity,
                                        mesh_xy=mesh_xy,capacity=capacity)
        if (model.get('sector')!=sector or not model.get('ordered')
                or model['recipe'].get('operator_realization')!='raw-sector-endpoint-v1'
                or model['digest']!=handle['digest']
                or model['K']!=handle['K']):
            _refuse(f'sector manifest binding mismatch: {sector}')
        model_headers[sector]=model
    constant=header['constant']
    _check_identity(constant['identity'],expected_identity)
    if constant['field']!='constant':
        _refuse('sector constant must name W_infinity-V')
    read_bank_constant_header(constant,mesh_xy=mesh_xy)
    return dict(header,model_headers=model_headers)


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


def read_shared_pole_matrix(io, q_span, *, meta, header):
    """Read b[mu_X,K_Y] for bounded matrix evaluation, in packed order.

    Returns complex128 [q,mu,Kp], float64 poles² [q,Kp] and int64 K [q].
    Unlike the two endpoint faces used by Sigma, this matrix remains on all
    processors even for one Gamma parent. Only scalar pole tables replicate.
    """
    from runtime.padding import padded_axis

    if header.get("sector") is not None:
        _refuse("scalar matrix reader cannot consume a sector endpoint; use the face reader")
    if header.get("validation_receipt", {}).get("status") == "NOT_MEASURED":
        _refuse("metadata-only validation cannot authorize tensor reads")
    if header["schema"] != SCHEMA or not header["finalized"]:
        _refuse("matrix reader requires a finalized model")
    basis, ledger = _check_basis(meta, header), _capacity(meta)
    # Both guards belong to the reader this one was extracted from
    # (:func:`read_shared_pole_faces`) and are not optional here: a packed
    # basis is bound to the mesh it was packed on, and a dataset validated
    # as empty has no slab to read.
    if io.mesh is not basis.mesh_xy:
        _refuse("reader mesh differs from packed basis mesh")
    _check_io_capacity(ledger, io.mesh, header)
    lo, hi = _span(q_span, header["n_q_irr"], "q_span")
    spec = P(None, "x", None, "y")
    if header["Kmax"] == 0:
        # A model with zero retained poles is a state every other consumer
        # supports (gw/mpa/sigma.py, the census, the exporter). Asking
        # read_slab for a >= 1 wide slab of a dataset just validated as
        # empty is not a read this reader may make.
        _admit(ledger, "empty_matrix", 8*(hi-lo))
        zeros = jax.jit(
            lambda: jnp.zeros((hi-lo, basis.n_packed, 0), jnp.complex128),
            out_shardings=NamedSharding(io.mesh, P(None, "x", "y")))
        return (zeros(), jnp.ones((hi-lo, 0), jnp.float64),
                jnp.zeros(hi-lo, jnp.int64))
    width = padded_axis(header["Kmax"], io.mesh, name="shared_pole_K",
                        specs=((spec, 3),)).carrier
    shape = (hi-lo, basis.n_canonical, 1, width)
    arg, output, temporary = _conversion_bytes(basis, shape, spec, unpack=False)
    scalar = 32*(hi-lo)*width
    _admit(ledger, "read_matrix", output+scalar, arg+temporary+output,
           device_panel=max(arg, scalar), native_host=True, io=io)
    b = io.read_slab("factor", shape=shape, offset=(lo, 0, 0, 0), partition_spec=spec)
    b = basis.pack_axis(b, 1, spec=spec)
    poles = io.read_slab("poles2_ry2", shape=(hi-lo, width), offset=(lo, 0),
                         partition_spec=P())
    counts = jnp.asarray(header["K"][lo:hi], jnp.int64)
    active = jnp.arange(width)[None, :] < counts[:, None]
    return (jnp.where(active[:, None, :], b[:, :, 0, :], 0),
            jnp.where(active, poles, 1.0), counts)


def face_width(mesh, kmax, column_span=None):
    """The pole-column carrier ``read_shared_pole_faces`` returns for one read.

    A whole-K read is the model's Kmax, which an SC run holds fixed
    (``_k_extent``), so every face consumer keyed by this width is compiled
    once.  A column panel keeps the width its schedule admitted.  Both face
    orientations tile it.
    """
    k = int(kmax) if column_span is None else int(column_span[1]) - int(column_span[0])
    return padded_axis(k, mesh, name="shared_pole_face_K",
                       specs=((P(None,"x",None,"y"),3),(P(None,"y",None,"x"),3))).carrier


def read_shared_pole_faces(io, q_span, *, meta, header, column_span=None, basis=None,
                           orientations=("x", "y")):
    """Read canonical row faces and pack once at the I/O boundary.

    Returns b_X, b_Y, poles2, K with shapes (b,mu_p,spin,Kcap),
    (b,mu_p,spin,Kcap), (b,Kcap), (b,). Faces use P(None,'x',None,'y')
    and P(None,'y',None,'x'); poles and int64 counts are replicated. K is
    the active count *within the returned column slice*, so every consumer
    can mask with arange(Kcap)<K even when column_span starts above zero.
    ``orientations`` may select one face when an ordered endpoint consumer
    needs only that orientation. The unrequested tuple slot is ``None``;
    the default preserves the original two-face contract.
    """
    orientations = tuple(orientations)
    if not orientations or len(set(orientations)) != len(orientations) or any(
            axis not in ("x", "y") for axis in orientations):
        _refuse(f"face orientations must be a nonempty subset of ('x','y'); got {orientations}")
    if header.get("validation_receipt", {}).get("status") == "NOT_MEASURED":
        _refuse("metadata-only validation cannot authorize tensor reads")
    if header["schema"] != SCHEMA or not header["finalized"]:
        _refuse("face reader requires a validated finalized model")
    basis = _check_basis(meta, header, basis)
    components = header.get("factor_components", 1)
    ledger = _capacity(meta)
    if io.mesh is not basis.mesh_xy:
        _refuse("reader mesh differs from packed basis mesh")
    _check_io_capacity(ledger,io.mesh,header)
    lo, hi = _span(q_span, header["n_q_irr"], "q_span")
    if header["Kmax"] == 0 and column_span is None:
        _admit(ledger,"empty_faces",8*(hi-lo))
        shape = (hi-lo,basis.n_packed,components,0)
        faces = {axis: jax.jit(lambda: jnp.zeros(shape,jnp.complex128),
                              out_shardings=NamedSharding(io.mesh,P(None,axis,None,"y" if axis == "x" else "x")))()
                 for axis in orientations}
        return (faces.get("x"), faces.get("y"), jnp.ones((hi-lo,0),jnp.float64),
                jnp.zeros(hi-lo,jnp.int64))
    c0, c1 = _span(column_span or (0,header["Kmax"]), header["Kmax"], "column_span")
    # Selected orientations share one padded pole extent; counts exclude padding.
    width = face_width(io.mesh, header["Kmax"], None if column_span is None else (c0, c1))
    totals = {}
    for axis in orientations:
        shape = (hi-lo,basis.n_canonical,components,width)
        totals[axis] = _conversion_bytes(
            basis,shape,P(None,axis,None,"y" if axis == "x" else "x"),unpack=False)
    metadata = 32*(hi-lo)*width+8*(hi-lo)
    resident = sum(totals[axis][1] for axis in orientations)
    held = 0
    peak = 0
    for axis in orientations:
        raw, packed, temporary = totals[axis]
        peak = max(peak, held+raw+packed+temporary, held+2*packed)
        held += packed
    _admit(ledger,"read_faces",resident+metadata,max(0,peak-resident),
           device_panel=max(*(totals[axis][0] for axis in orientations),
                            8*(hi-lo)*width),native_host=True,io=io)
    counts = jnp.asarray(np.clip(np.asarray(header["K"][lo:hi])-c0,0,c1-c0), dtype=jnp.int64)
    active = jnp.arange(width)[None,:] < counts[:,None]
    faces = {}
    for axis in orientations:
        spec = P(None,axis,None,"y" if axis == "x" else "x")
        b = io.read_slab("factor", shape=(hi-lo,basis.n_canonical,components,width),
                         offset=(lo,0,0,c0), valid_shape=(hi-lo,basis.n_logical,components,c1-c0),
                         partition_spec=spec)
        b = basis.pack_axis(b, 1, spec=spec)
        faces[axis] = jnp.where(active[:,None,None,:] & jnp.asarray(
            basis.active_mask)[None,:,None,None], b, 0.0)
        del b
    poles = io.read_slab("poles2_ry2", shape=(hi-lo,width), offset=(lo,c0), valid_shape=(hi-lo,c1-c0), partition_spec=P())
    return (faces.get("x"), faces.get("y"), jnp.where(active,poles,1.0), counts)


def read_shared_pole_cross_faces(readers, q_span, *, meta, headers, bases,
                                 column_span=None):
    """Read both CT endpoint factors with one authenticated pole census.

    ``readers``, ``headers`` and ``bases`` are charge/current pairs. The
    returned (b_C_X,b_T_Y,poles2,K) has the ordinary public face layouts.
    TC is obtained by exchanging endpoint roles, never stored separately.
    The two files carry the same iteration, parent order and exact poles.
    All factors remain XY sharded and only pole metadata is compared.
    """
    left, right = headers
    if left.get("sector") != "CT_C" or right.get("sector") != "CT_T":
        _refuse("CT requires CT_C and CT_T endpoint stores")
    for key in ("identity", "K", "Kmax", "q_irr_full_idx", "ordered"):
        if _json(left.get(key)) != _json(right.get(key)):
            _refuse(f"CT endpoint stores disagree on {key}")
    c = read_shared_pole_faces(readers[0], q_span, meta=meta, header=left,
                               column_span=column_span, basis=bases[0])
    ledger = _capacity(meta)
    previous = ledger.live_stages
    amount = (sum(_local_bytes(a.shape, a.dtype, readers[0].mesh, a.sharding.spec)
                  for a in c[:2]) + sum(a.size*a.dtype.itemsize for a in c[2:]))
    row = _admit(ledger, "cross_charge_faces", amount)
    ledger.live_stages = (*previous, row["stage"])
    try:
        t = read_shared_pole_faces(readers[1], q_span, meta=meta, header=right,
                                   column_span=column_span, basis=bases[1])
    finally:
        ledger.live_stages = previous
    if not bool(jnp.all(jax.lax.bitcast_convert_type(c[2], jnp.uint64) ==
                        jax.lax.bitcast_convert_type(t[2], jnp.uint64))):
        _refuse("CT endpoint stores disagree on exact pole bits")
    return c[0], t[1], c[2], c[3]


# Scratch bank uses the same identity and metadata transactions.
_BANK_SAMPLE_FIELDS = ("Wc", "dWc_ds")
_BANK_MOMENT_FIELDS = ("M1", "M3")
# An ordered bank also carries the time-reversal-odd z-moments the constructor
# reads: M0 at 1/z and M2 at 1/z^3, with M_k = C_(k+1)/2 for the coefficient
# C_(k+1) of Wc at 1/z^(k+1) (the convention that already gives S_m = 2 M_(2m+1)).
_BANK_ODD_MOMENT_FIELDS = ("M0", "M2")

# Retired bank schemas, refused by name: v1 stored W_q(-conj z) at every
# sample, v2 at the fitted line samples, both dense beside a dense W and dW/ds.
_RETIRED_BANK_SCHEMAS = ("lorrax.shared-real-pole-bank.v1", "lorrax.shared-real-pole-bank.v2")


def line_panel_span(plan):
    """Sample ids [p0, p1) stored as direction panels: the fitted samples with Re z != 0.

    At such a line support the constructor reads only the directions Q selected
    from W(z) itself and the actions of W, dW and (ordered) the minus-q partner
    W_q(-conj z) on Q and on O = W Q (``gw.shared_pole_directions``), so the
    producer stores those and no dense matrix. A fitted sample on the imaginary
    axis (-conj z = z; its ordered conjugate direction needs O outside span(Q))
    and every held sample stay dense. The recipe lists the fitted line supports
    first, so the span is contiguous (refused otherwise).
    """
    z = {int(d): complex(v) for d, v in zip(plan["distinct_id"], plan["z_ry"])}
    ids = sorted(int(s) for s in plan["fit_ids"] if z[int(s)].real != 0)
    if not ids:
        return (0, 0)
    if ids != list(range(ids[0], ids[-1] + 1)):
        _refuse("line-panel samples are not one contiguous sample span")
    return (ids[0], ids[-1] + 1)


def _panel_span(header):
    return tuple(int(v) for v in header["line_panels"]["sample_span"])


def _has_line_panels(header):
    p0, p1 = _panel_span(header)
    return p1 > p0


def dense_sample_rows(header, ids):
    """Rows of the dense Wc/dWc_ds sample axis for sample ids outside the panel span.

    The dense axis holds every sample except the line-panel span [p0, p1):
    row = id below p0, id - (p1 - p0) above it. A line-panel id refuses by name.
    """
    p0, p1 = _panel_span(header)
    nsample = int(header["bank_shape"]["nsample"])
    rows = []
    for sid in ids:
        sid = int(sid)
        if not 0 <= sid < nsample:
            _refuse(f"sample id {sid} outside the bank's {nsample} samples")
        if p0 <= sid < p1:
            _refuse(f"sample {sid} is a line-panel sample: its dense W is not stored, "
                    "only its directions and their actions (read_line_panels)")
        rows.append(sid if sid < p0 else sid - (p1 - p0))
    return rows


def _dense_sample_count(header):
    p0, p1 = _panel_span(header)
    return int(header["bank_shape"]["nsample"]) - (p1 - p0)


def _bank_sample_fields(header):
    """Every dense dataset with a sample axis."""
    return _BANK_SAMPLE_FIELDS


def _sample_field(header, name):
    """(mask key, mask column, dense sample rows) of a dense sample-axis field, else None."""
    if name in _BANK_SAMPLE_FIELDS:
        return "sample_written", _BANK_SAMPLE_FIELDS.index(name), _dense_sample_count(header)
    return None


def line_panel_name(family, sample, cross=False):
    """Dataset of one line sample's panels for one endpoint family."""
    return f"line_{family}_{int(sample):03d}" + ("_cross" if cross else "")


def _bank_masks(header):
    """Mask key -> bool array of every committed-field mask the bank carries."""
    keys = ("sample_written", "moment_written") + (
        ("line_written",) if _has_line_panels(header) else ())
    return {key: np.asarray(header[key], dtype=bool) for key in keys}


def _bank_moment_fields(header):
    """Moment datasets of one bank: M1/M3, plus M0/M2 when it carries odd moments."""
    return (_BANK_MOMENT_FIELDS + (_BANK_ODD_MOMENT_FIELDS if header.get("odd_moments") else ())
            + (("constant",) if "photon_layout" in header else ()))


class ResidentBankPayload:
    """One map's construction scratch held on the devices instead of a file.

    The bank is produced frequency-major (one Green/FFT stream per sample for
    every parent) and consumed parent-major by the constructor, so the file
    route writes and then re-reads the whole payload once. When that payload
    fits (``gw.shared_pole_screening._bank_residence``) it stays here: each field is the
    canonical-order array the file would hold, ``[nq, nsample, d_c, d_c]`` or
    ``[nq, d_c, d_c]`` complex128 at ``P(None,[None,],'x','y')`` with exact
    zeros past the logical extent, i.e. ``16 * payload / P`` bytes per rank.
    A line-panel dataset (``line_panel_name``) is the rectangular
    ``[nq, fields, rows, width]`` array the file holds, stored at its own shape.

    It implements only the dataset subset of the ``SlabIO`` handle that the
    bank writer and reader use (``create_dataset``, ``write_slab``,
    ``read_slab``, ``sync_writes``, ``write_attr``), so the header, masks,
    identity checks, admissions and canonical packing are the file route's
    own code. The header is kept as the JSON text the file would store.
    Reads return exactly the file read's values: one sliced face copy, moved
    to batch layout by the same staged exchange as a permuted file read.

    ``memory_kind="pinned_host"`` is the tier for a payload the devices cannot
    hold: each [d_c, d_c] face tile (one q, one sample) is a pinned-host array
    with the face sharding, written by the frequency-major producer and moved
    back whole by the parent-major reader, so every move is one contiguous
    per-rank DMA and the devices hold only the span being read.
    """

    def __init__(self, mesh, *, carrier, label, memory_kind="device"):
        if memory_kind not in ("device", "pinned_host"):
            _refuse(f"resident bank memory kind {memory_kind!r}")
        self.mesh = mesh
        self.carrier = int(carrier)
        self.label = str(label)
        self.memory_kind = memory_kind
        self.header_json = None
        self._fields = {}
        self._logical = {}
        self._stored = {}

    def __str__(self):
        tier = "device-resident" if self.memory_kind == "device" else "pinned-host"
        return f"{tier} shared-pole bank ({self.label})"

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def payload_bytes_per_rank(self):
        return sum(_local_bytes(shape, np.complex128, self.mesh, self._spec(len(shape)))
                   for shape in self._stored.values())

    def release(self):
        """Drop every device payload; the handle cannot be read afterwards."""
        self._fields.clear()
        self.header_json = None

    def _spec(self, ndim):
        return P(*((None,) * (ndim - 2)), "x", "y")

    def _lead(self, offset):
        # Rank-identical host offsets, placed without a cross-process assertion.
        return device_put_process_local(np.asarray(offset[:-2], np.int32),
                                        NamedSharding(self.mesh, P()))

    def create_dataset(self, name, *, shape, dtype):
        shape = tuple(int(s) for s in shape)
        if np.dtype(dtype) != np.dtype(np.complex128) or len(shape) not in (3, 4):
            _refuse(f"resident bank {name} must be complex128 [nq,(nsample,),d,d]")
        if name in self._fields:
            if self._logical[name] != shape:
                _refuse(f"resident bank {name} extent changed")
            return
        if name.startswith("line_"):
            # Direction panels are stored at their own rectangular shape.
            stored = shape
        elif max(shape[-2:]) > self.carrier:
            _refuse(f"resident bank {name} logical extent exceeds canonical carrier")
        else:
            stored = shape[:-2] + (self.carrier, self.carrier)
        # Host tiles are keyed by their lead index; an unwritten one reads as
        # the file's zero fill.
        self._fields[name] = (_resident_zeros(self.mesh, stored)()
                              if self.memory_kind == "device" else {})
        self._logical[name] = shape
        self._stored[name] = stored

    def write_attr(self, name, value):
        # Masks and typed tables are authenticated by the JSON header alone.
        if name == "header_json":
            self.header_json = bytes(value).decode()

    def sync_writes(self):
        return None

    def write_slab(self, name, A, *, offset):
        if name not in self._fields:
            _refuse(f"resident bank {name} was not created")
        store, stored = self._fields[name], self._stored[name]
        offset = tuple(int(v) for v in offset)
        if (A.ndim != len(stored) or tuple(A.shape[-2:]) != stored[-2:]
                or any(o != 0 for o in offset[-2:])
                or any(o + s > n for o, s, n in zip(offset, A.shape, stored))
                or not A.sharding.is_equivalent_to(NamedSharding(self.mesh, self._spec(A.ndim)), A.ndim)):
            _refuse(f"resident bank {name} write must be a face-tiled full-carrier span")
        # A square operator is zeroed past its logical extent, as the file
        # stores it; a direction panel is stored whole.
        logical = None if name.startswith("line_") else self._logical[name][-1]
        if self.memory_kind == "device":
            self._fields[name] = _resident_update(self.mesh, len(stored), logical)(
                store, A, self._lead(offset))
            return
        if logical is not None:
            A = _resident_mask(self.mesh, A.ndim, logical)(A)
        host = NamedSharding(self.mesh, P("x", "y"), memory_kind=self.memory_kind)
        for index in np.ndindex(A.shape[:-2]):
            store[tuple(o + i for o, i in zip(offset, index))] = jax.device_put(A[index], host)

    def read_slab(self, name, *, shape, offset, dtype, partition_spec, valid_shape=None):
        if name not in self._fields:
            _refuse(f"resident bank {name} has no payload")
        store, stored = self._fields[name], self._stored[name]
        shape = tuple(int(s) for s in shape)
        offset = tuple(int(v) for v in offset)
        valid = shape if valid_shape is None else tuple(int(v) for v in valid_shape)
        logical = self._logical[name]
        if (np.dtype(dtype) != np.dtype(np.complex128) or len(shape) != len(stored)
                or shape[-2:] != stored[-2:] or any(o != 0 for o in offset[-2:])
                or valid[:-2] != shape[:-2] or valid[-2:] != logical[-2:]
                or any(o + s > n for o, s, n in zip(offset, shape, stored))):
            _refuse(f"resident bank {name} read must request full-carrier spans")
        face = self._spec(len(stored))
        if self.memory_kind == "device":
            value = _resident_slice(self.mesh, len(stored), shape[:-2])(store, self._lead(offset))
        else:
            device = NamedSharding(self.mesh, P("x", "y"))
            tiles = [store.get(tuple(o + i for o, i in zip(offset, index)))
                     for index in np.ndindex(shape[:-2])]
            tiles = [_resident_zeros(self.mesh, shape[-2:])() if t is None
                     else jax.device_put(t, device) for t in tiles]
            value = _resident_stack(self.mesh, shape[:-2])(*tiles)
        layout = _bank_layout(None if tuple(partition_spec) == tuple(face) else partition_spec)
        return value if layout == "face" else _bank_face_to_batch(self.mesh, value.ndim)(value)


@lru_cache(maxsize=None)
def _resident_zeros(mesh, shape):
    spec = P(*((None,) * (len(shape) - 2)), "x", "y")
    return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                   out_shardings=NamedSharding(mesh, spec))


def _logical_mask(value, logical):
    rows = jnp.arange(value.shape[-2])[:, None] < logical
    cols = jnp.arange(value.shape[-1])[None, :] < logical
    return jnp.where(rows & cols, value, jnp.zeros((), value.dtype))


@lru_cache(maxsize=None)
def _resident_mask(mesh, ndim, logical):
    """Zero past the logical extent as the file stores it (host-tier writes)."""
    spec = NamedSharding(mesh, P(*((None,) * (ndim - 2)), "x", "y"))
    return jax.jit(lambda value: _logical_mask(value, logical), out_shardings=spec)


@lru_cache(maxsize=None)
def _resident_stack(mesh, lead_shape):
    """Host-tier read: device face tiles stacked into one face-tiled span."""
    spec = NamedSharding(mesh, P(*((None,) * len(lead_shape)), "x", "y"))
    return jax.jit(lambda *tiles: jnp.stack(tiles).reshape(tuple(lead_shape) + tiles[0].shape),
                   out_shardings=spec)


@lru_cache(maxsize=None)
def _resident_update(mesh, ndim, logical):
    """In-place span update; zero past the logical extent as the file stores it."""
    spec = NamedSharding(mesh, P(*((None,) * (ndim - 2)), "x", "y"))

    def update(store, value, lead):
        if logical is not None:
            value = _logical_mask(value, logical)
        zero = jnp.zeros((), lead.dtype)
        start = tuple(lead[i] for i in range(ndim - 2)) + (zero, zero)
        return jax.lax.dynamic_update_slice(store, value, start)
    return jax.jit(update, donate_argnums=(0,), out_shardings=spec)


@lru_cache(maxsize=None)
def _resident_slice(mesh, ndim, lead_shape):
    spec = NamedSharding(mesh, P(*((None,) * (ndim - 2)), "x", "y"))

    def take(store, lead):
        zero = jnp.zeros((), lead.dtype)
        start = tuple(lead[i] for i in range(ndim - 2)) + (zero, zero)
        return jax.lax.dynamic_slice(store, start, tuple(lead_shape) + tuple(store.shape[-2:]))
    return jax.jit(take, out_shardings=spec)


class ResidentSectorModel:
    """One sector model held on the devices for Sigma instead of a file.

    The constructor writes each sector round by round and Sigma reads it once
    (census, then endpoint faces) in the same map. When the four models fit
    (``gw.shared_pole_sectors._sector_model_residence``) they stay here:
    ``write_shared_pole_model`` stages each batch, finalization assembles the
    file's own datasets (``factor`` [nq, nmu, components, Kmax], exact zeros
    past each K; ``poles2_ry2`` [nq, Kmax], 1 past each K) on a mesh-divisible
    carrier, and every reader (census, faces, digest) goes through
    ``read_slab`` with SlabIO's semantics, so Sigma reads the file route's
    values bit for bit. Only reads are implemented; writes are the staging.
    """

    def __init__(self, mesh, *, label):
        self.mesh = mesh
        self.label = str(label)
        self.header_json = None
        self.final_commit = None
        self.staging = {}
        self._fields = {}
        self._logical = {}

    def __str__(self):
        return f"device-resident shared-pole model ({self.label})"

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def logical(self, name):
        return self._logical.get(name)

    def release(self):
        """Drop every device payload; the model cannot be read afterwards."""
        self.staging.clear()
        self._fields.clear()
        self.header_json = None

    def finalize_payload(self, header, *, n_canonical):
        """Assemble the file's final datasets from the staged batches."""
        nq, nmu, kmax = header["n_q_irr"], header["n_mu_logical"], header["Kmax"]
        components = header.get("factor_components", 1)
        carrier = padded_axis(max(kmax, 1), combined_divisor(self.mesh.shape["x"], self.mesh.shape["y"]),
                              name="shared_pole_model_K").carrier
        counts = np.asarray(header["K"], np.int64)
        factors, poles = [], []
        for batch in sorted(header["batches"], key=lambda row: row["lo"]):
            canonical, staged, width = self.staging.pop(batch["name"])
            # Batch widths are runtime operands: one executable per staged shape.
            factors.append(_model_factor_block(self.mesh, canonical.shape, carrier, nmu)(
                canonical, np.int32(width)))
            poles.append(_model_pole_block(self.mesh, staged.shape, kmax)(
                staged, counts[batch["lo"]:batch["hi"]]))
        self._fields["factor"] = _model_concat(self.mesh, 4)(*factors)
        self._fields["poles2_ry2"] = _model_concat(self.mesh, 2)(*poles)
        self._logical["factor"] = (nq, nmu, components, kmax)
        self._logical["poles2_ry2"] = (nq, kmax)

    def read_slab(self, name, *, shape=None, offset=None, partition_spec=P(),
                  valid_shape=None, dtype=None):
        if name not in self._fields:
            _refuse(f"{self} has no dataset {name}")
        logical = self._logical[name]
        shape = logical if shape is None else tuple(int(v) for v in shape)
        offset = (0,) * len(shape) if offset is None else tuple(int(v) for v in offset)
        valid = shape if valid_shape is None else tuple(int(v) for v in valid_shape)
        return _model_read(self.mesh, logical, shape, offset, valid, tuple(partition_spec))(
            self._fields[name])


@lru_cache(maxsize=None)
def _model_factor_block(mesh, shape, carrier, nmu):
    """A staged canonical batch as its file dataset holds it: rows < nmu, columns < width."""
    def block(b, width):
        b = b[..., :min(shape[-1], carrier)]
        b = jnp.pad(b, ((0, 0),) * 3 + ((0, carrier - b.shape[-1]),))
        keep = ((jnp.arange(shape[1]) < nmu)[None, :, None, None]
                & (jnp.arange(carrier) < width)[None, None, None, :])
        return jnp.where(keep, b, jnp.zeros((), b.dtype))
    return jax.jit(block, out_shardings=NamedSharding(mesh, P(None, "x", None, "y")))


@lru_cache(maxsize=None)
def _model_pole_block(mesh, shape, kmax):
    """Staged poles as finalization writes them: 1 past K (every column past the batch width is past K)."""
    def block(poles, counts):
        poles = poles[:, :min(kmax, shape[-1])]
        poles = jnp.pad(poles, ((0, 0), (0, kmax - poles.shape[-1])))
        return jnp.where(jnp.arange(kmax)[None, :] < counts[:, None], poles, 1.0)
    return jax.jit(block, out_shardings=NamedSharding(mesh, P()))


@lru_cache(maxsize=None)
def _model_concat(mesh, ndim):
    spec = P(None, "x", None, "y") if ndim == 4 else P()
    return jax.jit(lambda *blocks: jnp.concatenate(blocks, axis=0),
                   out_shardings=NamedSharding(mesh, spec))


@lru_cache(maxsize=None)
def _model_read(mesh, logical, shape, offset, valid, spec):
    """SlabIO read semantics: dataset values inside its extent and the valid prefix, zero elsewhere."""
    def take(store):
        stop = tuple(min(o + s, n) for o, s, n in zip(offset, shape, logical))
        value = store[tuple(slice(o, max(o, e)) for o, e in zip(offset, stop))]
        value = jnp.pad(value, tuple((0, s - v) for s, v in zip(shape, value.shape)))
        for axis, v in enumerate(valid):
            if v < shape[axis]:
                keep = jax.lax.broadcasted_iota(jnp.int32, shape, axis) < v
                value = jnp.where(keep, value, jnp.zeros((), value.dtype))
        return value
    return jax.jit(take, out_shardings=NamedSharding(mesh, P(*spec)))


def open_shared_pole_model(path, *, mesh_xy):
    """Read handle for a finalized model, file or device resident."""
    return path if isinstance(path, ResidentSectorModel) else SlabIO(path, mode="r", mesh=mesh_xy)


def _bank_io(path, mode, mesh):
    """The bank's payload handle: the resident arrays or one SlabIO transaction."""
    return path if isinstance(path, ResidentBankPayload) else SlabIO(path, mode=mode, mesh=mesh)


def open_shared_pole_bank(path, *, mesh_xy):
    """Read handle for an authenticated bank, file or device resident."""
    return _bank_io(path, "r", mesh_xy)


def line_panel_geometry(meta, *, ordered, photon_bases=None):
    """Families of a bank's line panels: {family: (rows, cross rows or None)}, and
    the per-family state count S (TRS 2: z, conj z; ordered 4: also -z, -conj z).

    A charge bank has one family whose rows are the canonical centroid carrier.
    A photon bank has C (packed charge rows) and T (packed current rows, three
    Cartesian components per centroid), each with a cross panel on the other
    family's rows: the CT/TC actions on that family's directions.
    """
    states = 4 if ordered else 2
    if photon_bases is None:
        return {"charge": (int(meta.mu_basis.n_canonical), None)}, states
    charge, current = (int(b.n_packed) for b in photon_bases)
    return {"C": (charge, 3 * current), "T": (3 * current, charge)}, states


def shared_pole_bank_payload_bytes(meta, *, recipe, ordered, nq, mesh_xy, line_widths=None,
                                   photon_extent=None, photon_bases=None):
    """Per-rank bytes of the complete bank payload, an admission bound.

    Dense Wc, dWc/ds at every sample outside the line-panel span
    (``line_panel_span``), the moments M1, M3 (plus M0, M2 ordered), and per
    line sample and family the panel [1+2S, rows, r] (plus [2S, cross rows, r]
    on a photon bank) at the caller's width bound ``line_widths[family]``:
    16 N_q [(2 N_dense + N_m) d^2 + N_line sum_f ((1+2S) n_f + 2S n_f') r_f] / P.
    The caller bounds r by the carrier of the line direction cap, the
    constructor's own admission width for a line support; a multiplet closed
    past the cap stores its actual width. A photon bank (``photon_extent`` =
    its packed extent, ``photon_bases`` its two centroid bases) is always
    ordered and also holds the per-parent ``constant`` and the three [1,d,d]
    static-contact diagnostics (Pi_grid, Drude, TT_contact).
    """
    plan = _bank_plan(recipe)
    p0, p1 = line_panel_span(plan)
    dense = _bank_nsample(plan) - (p1 - p0)
    families, states = line_panel_geometry(meta, ordered=bool(ordered) or photon_extent is not None,
                                           photon_bases=photon_bases)
    panel = 0
    for family, (rows, cross) in (families.items() if p1 > p0 else ()):
        width = int(line_widths[family])
        panel += ((1 + 2 * states) * rows + (2 * states * cross if cross else 0)) * width
    panel_bytes = 16 * int(nq) * (p1 - p0) * panel
    if photon_extent is not None:
        tiles = int(nq) * (2 * dense + 5) + 3
        return int((16 * tiles * int(photon_extent)**2 + panel_bytes) // int(mesh_xy.size))
    moments = 4 if ordered else 2
    carrier = int(meta.mu_basis.n_canonical)
    return int((16 * int(nq) * (2 * dense + moments) * carrier**2 + panel_bytes) // int(mesh_xy.size))


def write_bank_contact(path, fields, *, mesh_xy):
    """Write the photon bank's [1,d,d] static-contact diagnostics, file or resident.

    ``fields`` maps Pi_grid/Drude/TT_contact to face-tiled complex128 arrays.
    """
    with _bank_io(path, "a", mesh_xy) as io:
        for name, value in fields.items():
            if isinstance(io, ResidentBankPayload):
                io.create_dataset(name, shape=value.shape, dtype=np.complex128)
                io.write_slab(name, value, offset=(0,) * value.ndim)
            else:
                io.write_slab(name, value, offset=(0,) * value.ndim, global_shape=value.shape)
                io.sync_writes()


_CONSTANT_KIND = "photon_bank_constant_v1"


def write_bank_constant(payload, dest, *, header, mesh_xy):
    """Publish a resident photon bank's per-parent W_infinity-V constant to ``dest``.

    Returns the manifest resource ``dict(path, identity, field, kind, commit)``.
    The file holds dataset ``constant`` [nq,d,d] (Ry, canonical photon order)
    and ``constant_json``: the authenticated bank header plus a commit digest,
    so consumers read the same geometry and symmetry tables as from the bank.
    """
    nq, d = int(header["bank_shape"]["nq"]), int(header["bank_shape"]["d"])
    record = dict(header, constant_kind=_CONSTANT_KIND)
    commit = hashlib.sha256(_json(record).encode()).hexdigest()
    value = payload.read_slab("constant", shape=(nq, d, d), valid_shape=(nq, d, d),
                              offset=(0, 0, 0), dtype=np.complex128,
                              partition_spec=P(None, "x", "y"))
    with SlabIO(str(dest), mode="w", mesh=mesh_xy) as io:
        io.write_slab("constant", value, offset=(0, 0, 0), global_shape=(nq, d, d))
        io.sync_writes()
        io.write_attr("constant_json", np.bytes_(_json(dict(record, commit=commit))))
    return dict(path=str(Path(dest).resolve()), identity=header["identity"], field="constant",
                kind=_CONSTANT_KIND, commit=commit)


def read_bank_constant_header(resource, *, mesh_xy):
    """Authenticated bank header of a manifest ``constant`` resource (bank or published file)."""
    if resource.get("kind") != _CONSTANT_KIND:
        return validate_shared_pole_bank(resource["path"], expected_identity=resource["identity"],
                                         mesh_xy=mesh_xy, require_complete=True)
    with h5py.File(resource["path"], "r") as f:
        raw = f["constant_json"][()]
    header = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    _check_identity(header["identity"], resource["identity"])
    if header.pop("commit", None) != resource.get("commit"):
        _refuse("photon bank constant commit mismatch")
    return header


def read_bank_constant(resource, header, *, meta, mesh_xy):
    """The [nq,d,d] face-tiled W_infinity-V constant named by a manifest resource."""
    nq, d = int(header["bank_shape"]["nq"]), int(header["bank_shape"]["d"])
    with SlabIO(resource["path"], mode="r", mesh=mesh_xy) as io:
        if resource.get("kind") != _CONSTANT_KIND:
            return read_shared_pole_bank(io, (0, nq), meta=meta, header=header,
                                         fields=("constant",))["constant"]
        return io.read_slab("constant", shape=(nq, d, d), valid_shape=(nq, d, d),
                            offset=(0, 0, 0), dtype=np.complex128,
                            partition_spec=P(None, "x", "y"))


_STATIC_REFERENCE_KIND = "photon_static_contact_v1"


def write_static_reference(path, fields, *, header, mesh_xy):
    """Publish map 0's static contact, the run-wide reference later SC maps freeze.

    Both bank tiers write the three small [1,d,d] contact arrays (48 d^2 bytes)
    to this file with the bank identity, photon layout and centroid digests, at
    the run level beside the per-map scratch generations, so retention of those
    generations never keeps a bank for it. Returns the JSON-safe reference
    ``dict(path, identity, kind, commit)``.
    """
    record = dict(identity=header["identity"], photon_layout=header["photon_layout"],
                  photon_centroid_digests=header["photon_centroid_digests"],
                  kind=_STATIC_REFERENCE_KIND)
    commit = hashlib.sha256(_json(record).encode()).hexdigest()
    record["commit"] = commit
    with SlabIO(str(path), mode="w", mesh=mesh_xy) as io:
        for name, value in fields.items():
            io.write_slab(name, value, offset=(0,) * value.ndim, global_shape=value.shape)
            io.sync_writes()
        io.write_attr("static_reference_json", np.bytes_(_json(record)))
    return dict(path=str(path), identity=header["identity"], kind=_STATIC_REFERENCE_KIND,
                commit=commit)


def static_reference_record(reference):
    """Authenticate a static reference by its committed record; host h5py only.

    Refuses anything but a :func:`write_static_reference` record whose stored
    identity and commit match ``reference``. Safe on one rank (no collective).
    """
    if reference.get("kind") != _STATIC_REFERENCE_KIND:
        _refuse("photon static reference is not a static-contact record")
    with h5py.File(reference["path"], "r") as f:
        raw = f["static_reference_json"][()]
    record = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    _check_identity(record["identity"], reference["identity"])
    if record.get("commit") != reference.get("commit"):
        _refuse("photon static reference commit mismatch")
    return record


def read_static_reference(reference, *, n, mesh_xy):
    """Authenticated static-contact record and (Pi_grid, Drude, TT_contact) of a reference.

    COLLECTIVE over ``mesh_xy``. Returns ``(record, arrays)``; the record
    carries photon_layout, photon_centroid_digests and ``commit``.
    """
    names = ("Pi_grid", "Drude", "TT_contact")
    record = static_reference_record(reference)
    with SlabIO(reference["path"], mode="r", mesh=mesh_xy) as io:
        arrays = tuple(io.read_slab(key, shape=(1, n, n), partition_spec=P(None, "x", "y"),
                                    dtype=np.complex128) for key in names)
    return record, arrays


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
                                mesh_xy, photon_layout=None, mu_bases=None):
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
    if (path.header_json is not None if isinstance(path, ResidentBankPayload)
            else Path(path).exists()):
        _refuse("scratch bank already exists; validate it before resuming")
    plan = _bank_plan(recipe)
    header = _metadata(meta, tables, recipe, identity, photon=photon_layout is not None)
    if photon_layout is not None:
        photon_layout.assert_mesh(mesh_xy)
        if (mu_bases is None or len(mu_bases) != 2
                or tuple(b.n_logical for b in mu_bases) != photon_layout.logical_extents[:2]):
            _refuse("photon scratch needs both authenticated centroid bases")
        header.update(photon_layout=dict(logical_extents=list(photon_layout.logical_extents),
            carrier_extents=list(photon_layout.carrier_extents), mesh_side=photon_layout.mesh_side,
            ordering=photon_layout.ordering, packed_extent=photon_layout.packed_extent),
            photon_centroid_digests=[hashlib.sha256(np.asarray(b.canonical_indices, dtype="<i4").tobytes()).hexdigest()
                                     for b in mu_bases],
            n_mu_logical=photon_layout.packed_extent,
            capacity_geometry=dict(_capacity(meta).geometry), representation="photon-ordered-z",
            normalization="Wc=W-W_infinity; constant=W_infinity-V; both current endpoints retained")
    odd = bool(header.get("ordered"))
    if odd:
        # One source of truth for the infinity block: the ordered bank itself.
        header["odd_moments"] = True
    fields = _bank_moment_fields(header)
    parents = (tables["q_irr_full_idx"] if isinstance(tables, dict)
               else tables.q_irr_full_idx)
    nq, nsample, d = len(parents), _bank_nsample(plan), int(header["n_mu_logical"])
    if nq <= 0:
        _refuse("scratch bank requires at least one irreducible q")
    p0, p1 = line_panel_span(plan)
    families, states = line_panel_geometry(meta, ordered=odd, photon_bases=(
        mu_bases if photon_layout is not None else None))
    nodes = ("z", "conj(z)", "-z", "-conj(z)")[:states]
    header["line_panels"] = dict(
        sample_span=[p0, p1],
        samples="fitted supports with Re z != 0; every other sample is stored dense",
        rows={family: rows for family, (rows, _) in families.items()},
        cross_rows={family: cross for family, (_, cross) in families.items() if cross},
        row_order=("canonical centroid carrier" if photon_layout is None else
                   "packed endpoint basis, Cartesian component minor (the sector read's order)"),
        states=list(nodes),
        fields=["Q"] + [f"{kind}@{node}" for node in nodes for kind in ("output", "action")],
        cross_fields=([f"{kind}@{node}" for node in nodes for kind in ("output", "action")]
                      if photon_layout is not None else []),
        directions=("Q: right singular vectors of W(z) above the relative cutoff, at most the line cap, "
                    "whole multiplets; the conj(z) state's direction is O = W(z) Q"),
        output="W at the state's node applied to its direction: W Q, W^H O"
               + (", W_q(-conj z)^H Q, W_q(-conj z) O" if odd else ""),
        action=("dW/dz at the state's node applied to its direction" if odd
                else "dW/ds at the state's node applied to its direction"),
        minus_q_partner=("W_q(-conj z) = conj(W_{-q}(z)) from the exact -q response rows through "
                         "parent q's own V" + (" and frozen contact" if photon_layout is not None else "")
                         if odd else None),
        width={family: [None] * (p1 - p0) for family in families},
        counts={family: np.zeros((nq, p1 - p0), np.int64).tolist() for family in families})
    header.update(
        schema=BANK_SCHEMA, bank_shape={"nq": nq, "nsample": nsample, "d": d},
        bank_sample_plan=plan,
        bank_plan_digest=hashlib.sha256(_json(plan).encode()).hexdigest(),
        sample_written=np.zeros((nq, nsample - (p1 - p0), len(_BANK_SAMPLE_FIELDS)), dtype=bool).tolist(),
        moment_written=np.zeros((nq, len(fields)), dtype=bool).tolist(),
        complete=False, final_commit=None,
        units={"Wc": "Ry", "dWc_ds": "Ry^-1", "M1": "Ry^3", "M3": "Ry^5",
               **({"M0": "Ry^2", "M2": "Ry^4"} if odd else {}),
               **({"constant": "Ry"} if photon_layout is not None else {}),
               "line_output": "Ry", "line_action": "Ry^0 (dW/dz)" if odd else "Ry^-1 (dW/ds)"},
        derivative_variable="s=z_Ry^2",
        moment_convention=("S_m = 2 M_(2m+1); physical M1 and M3; odd M0 (1/z) and M2 (1/z^3), M_k = C_(k+1)/2"
                           if odd else "S_m = 2 M_(2m+1); physical M1 and M3 only"))
    if p1 > p0:
        header["line_written"] = np.zeros((nq, p1 - p0), dtype=bool).tolist()
    header["payload_bytes"] = 16 * nq * (2 * _dense_sample_count(header) + len(fields)) * d * d
    with _bank_io(path, "w", mesh_xy) as io:
        for field in _bank_sample_fields(header):
            io.create_dataset(field, shape=(nq, _sample_field(header, field)[2], d, d), dtype=np.complex128)
        for field in fields:
            io.create_dataset(field, shape=(nq, d, d), dtype=np.complex128)
        for key, mask in _bank_masks(header).items():
            io.write_attr(key, mask)
        for name in ("z_ry", "role", "distinct_id", "held", "support_pair", "fit_ids", "held_ids"):
            io.write_attr(name, plan[name])
        io.write_attr("role_codes_json", np.bytes_(_json(plan["role_codes"])))
        _write_metadata(io, header)
        _write_header(io, header)
    return header


def validate_shared_pole_bank(path, *, expected_identity, mesh_xy,
                              require_complete=False, expected_recipe=None):
    """Authenticate the scratch plan and per-field commit masks before replay.

    Only small metadata is read. Partial files are resumable, but a consumer
    cannot read a field whose q/sample mask is absent. Finalization follows
    the collective close; a completed bank is immutable. ``expected_recipe``
    additionally binds a preserved producer to the current resolved supports,
    eta, widths and conventions before constructor-only resume.
    """
    header = _read_header(path)
    if header.get("schema") in _RETIRED_BANK_SCHEMAS or "mirror_mode" in header:
        _refuse(f"scratch bank schema {header.get('schema')} is retired: it stores dense W at every "
                f"line sample and the minus-q partner fields; {BANK_SCHEMA} stores only the "
                "line-sample direction panels")
    if header.get("schema") != BANK_SCHEMA:
        _refuse("scratch bank schema mismatch")
    _check_identity(header["identity"], expected_identity)
    if hashlib.sha256(_json(header["recipe"]).encode()).hexdigest() != header["recipe_hash"]:
        _refuse("scratch bank recipe digest mismatch")
    if (expected_recipe is not None
            and _json(header["recipe"]) != _json(expected_recipe)):
        _refuse("scratch bank does not match the current resolved recipe")
    plan = _bank_plan(header["bank_sample_plan"])
    digest = hashlib.sha256(_json(plan).encode()).hexdigest()
    if digest != header.get("bank_plan_digest"):
        _refuse("scratch bank sample plan digest mismatch")
    if _json(plan) != _json(_bank_plan(header["recipe"])):
        _refuse("scratch bank stale recipe/roles/held map")
    panels = header.get("line_panels")
    if (not isinstance(panels, dict) or list(panels.get("sample_span", ())) != list(line_panel_span(plan))
            or set(panels.get("rows", {})) != set(panels.get("width", {}))
            or len(panels.get("states", ())) != (4 if header.get("ordered") else 2)
            or bool(panels.get("minus_q_partner")) != bool(header.get("ordered"))):
        _refuse("scratch bank line-panel span/family/state map mismatch")
    shape = header["bank_shape"]
    nq, nsample = int(shape["nq"]), int(shape["nsample"])
    p0, p1 = _panel_span(header)
    masks = _bank_masks(header)
    samples, moments = masks["sample_written"], masks["moment_written"]
    moment_fields = _bank_moment_fields(header)
    units = {"Wc": "Ry", "dWc_ds": "Ry^-1", "M1": "Ry^3", "M3": "Ry^5",
             "M0": "Ry^2", "M2": "Ry^4", "constant": "Ry"}
    fields = _bank_sample_fields(header) + moment_fields
    if (header.get("derivative_variable") != "s=z_Ry^2"
            or any(header.get("units", {}).get(name) != units[name]
                   for name in fields)):
        _refuse("scratch bank response/derivative convention mismatch")
    if (samples.shape != (nq, nsample - (p1 - p0), len(_BANK_SAMPLE_FIELDS))
            or moments.shape != (nq, len(moment_fields))
            or ("line_written" in masks and masks["line_written"].shape != (nq, p1 - p0))):
        _refuse("scratch bank malformed written masks")
    counts = {family: np.asarray(value, np.int64) for family, value in panels["counts"].items()}
    written = masks.get("line_written", np.zeros((nq, 0), bool))
    for family, width in panels["width"].items():
        if (len(width) != p1 - p0 or counts[family].shape != (nq, p1 - p0)
                or any((w is None) != (not written[:, i].any()) for i, w in enumerate(width))
                or np.any(counts[family] < 0)
                or any(w is not None and np.any(counts[family][:, i] > w) for i, w in enumerate(width))):
            _refuse(f"scratch bank line-panel {family} widths/counts disagree with its commit mask")
    if (nq != header["n_q_irr"] or nsample != _bank_nsample(plan)
            or shape["d"] != header["n_mu_logical"]
            or header["nspinor"] not in ((4,) if "photon_layout" in header else (1, 2, 4))):
        _refuse("scratch bank geometry/representation mismatch")
    # Geometry only: never load a matrix through the metadata handle. A
    # device-resident bank has no file; its masks live in the header above.
    if not isinstance(path, ResidentBankPayload):
        with h5py.File(path, "r") as file:
            # Boolean HDF5 enums are metadata, outside phdf5's numeric ABI.
            for key, mask in masks.items():
                if key not in file or not np.array_equal(file[key][()], mask):
                    _refuse(f"scratch bank {key} transaction mismatch")
            for name in ("z_ry", "role", "distinct_id", "held", "support_pair", "fit_ids", "held_ids"):
                if (name not in file or file[name].shape != plan[name].shape
                        or file[name].dtype != plan[name].dtype
                        or not np.array_equal(file[name][()], plan[name], equal_nan=True)):
                    _refuse(f"scratch bank typed plan {name} digest mismatch")
            if file["role_codes_json"][()].decode() != _json(plan["role_codes"]):
                _refuse("scratch bank role code table mismatch")
            for name in fields:
                geometry = _sample_field(header, name)
                expected = ((nq, geometry[2]) if geometry else (nq,)) + (shape["d"],) * 2
                if (name not in file or file[name].shape != expected
                        or file[name].dtype != np.dtype(np.complex128)
                        or file[name].chunks is not None):
                    _refuse(f"scratch bank {name} schema geometry/dtype/layout mismatch")
            nfields, ncross = len(panels["fields"]), len(panels["cross_fields"])
            for family, widths in panels["width"].items():
                for i, width in enumerate(widths):
                    if width is None:
                        continue
                    expected = [(line_panel_name(family, p0 + i), (nq, nfields, panels["rows"][family], width))]
                    if family in panels["cross_rows"]:
                        expected.append((line_panel_name(family, p0 + i, cross=True),
                                         (nq, ncross, panels["cross_rows"][family], width)))
                    for name, dims in expected:
                        if (name not in file or file[name].shape != dims
                                or file[name].dtype != np.dtype(np.complex128)):
                            _refuse(f"scratch bank {name} line-panel geometry/dtype mismatch")
    complete = all(bool(mask.all()) for mask in masks.values())
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
                           dWc_ds=None, M1=None, M3=None, M0=None, M2=None, constant=None,
                           line=None, meta, expected_identity, mesh_xy):
    """Write one bounded q/sample batch and commit masks after collective close.

    Dense sample arrays are complex128 ``[b,a,mu_p,mu_p]`` at
    ``P(None,None,'x','y')`` for a span of dense sample ids; physical moments
    are complex128 ``[b,mu_p,mu_p]`` at ``P(None,'x','y')``. ``line`` is one
    line sample's panels (``_prepare_line_write``). Endpoint conversions go
    through ``meta.mu_basis``; no off-axis Hermitization or rescaling.
    Already committed fields refuse overwrite, including in partial files.
    """
    with shared_pole_bank_writer(path, meta=meta, expected_identity=expected_identity,
                                 mesh_xy=mesh_xy) as (_, header, write):
        if not (all(mask.all() for mask in _bank_masks(header).values())
                and line is None
                and all(value is None for value in (Wc, dWc_ds, M1, M3, M0, M2, constant))):
            write(q_span=q_span, sample_span=sample_span, Wc=Wc, dWc_ds=dWc_ds,
                  M1=M1, M3=M3, M0=M0, M2=M2, constant=constant, line=line)
    return header


@contextmanager
def shared_pole_bank_writer(path, *, meta, expected_identity, mesh_xy):
    """One collective transaction for bounded slices of one frequency.

    Yield the existing read handle, authenticated header and slice writer.
    Every slice drains before releasing its staging array. Masks publish only
    after a successful transaction; final completion still follows close.
    """
    error = None
    try:
        header = validate_shared_pole_bank(path, expected_identity=expected_identity, mesh_xy=mesh_xy)
        _check_basis(meta, header)
        if mesh_xy is not meta.mu_basis.mesh_xy:
            _refuse("writer mesh differs from packed basis mesh")
        if header.get("complete"):
            _refuse("completed scratch bank is immutable")
    except Exception as exc:
        error = exc
    # Every serial metadata reader must close before any collective writer opens.
    agree_io_refusal(error, path=str(path), stage="shared_pole.bank_writer")
    with _bank_io(path, "a", mesh_xy) as io:
        def write(**fields):
            prepared = _prepare_bank_write(header, meta=meta, mesh_xy=mesh_xy, **fields)
            _write_bank_payload(io, header, meta, prepared)
        yield io, header, write
        _write_bank_masks(io, header)
    _complete_bank(path, header)


def _write_bank_masks(io, header):
    """Publish drained payload masks within the open collective handle."""
    for key, mask in _bank_masks(header).items():
        io.write_attr(key, mask)
    _write_header(io, header)


def _complete_bank(path, header):
    """Stamp a fully written bank only after its collective handle closes."""
    if all(mask.all() for mask in _bank_masks(header).values()):
        header["complete"] = True
        header["final_commit"] = hashlib.sha256(_json(header).encode()).hexdigest()
        _stamp_header(path, header, "shared_pole_bank.complete")


def _prepare_line_write(header, line, *, q0, q1, meta, mesh_xy, ledger):
    """Validate/admit one line sample's panels for parents [q0, q1).

    ``line`` is ``dict(sample, panels, counts[, cross])``: ``panels[family]``
    complex128 ``[b, 1+2S, rows, r]`` and ``cross[family]`` ``[b, 2S, cross
    rows, r]`` (photon) at ``P(None,None,'x','y')``, rows in the packed basis
    (the writer moves charge rows to the canonical carrier), ``counts[family]``
    int ``[b]`` the retained direction count of each parent (``<= r``). Every
    family of the sample is written in one call; its width is fixed by the
    first write of the sample.
    """
    panels_meta = header["line_panels"]
    p0, p1 = _panel_span(header)
    sample = int(line["sample"])
    if not p0 <= sample < p1:
        _refuse(f"line sample {sample} outside the line-panel span [{p0}, {p1})")
    if set(line["panels"]) != set(panels_meta["rows"]) or set(line.get("cross") or {}) != set(panels_meta["cross_rows"]):
        _refuse("line write must carry every family's panels (and cross panels on a photon bank)")
    if np.asarray(header["line_written"], bool)[q0:q1, sample - p0].any():
        _refuse(f"line sample {sample} span already committed")
    charge = "photon_layout" not in header
    basis = meta.mu_basis
    spec = P(None, None, "x", "y")
    pending = []
    for family, array in line["panels"].items():
        width = int(array.shape[-1])
        stored = panels_meta["width"][family][sample - p0]
        if stored is not None and int(stored) != width:
            _refuse(f"line sample {sample} family {family} width {width} differs from its first write {stored}")
        counts = np.asarray(line["counts"][family], np.int64)
        if counts.shape != (q1 - q0,) or np.any(counts < 0) or np.any(counts > width):
            _refuse(f"line sample {sample} family {family} counts must be [b] within [0, {width}]")
        rows = basis.n_packed if charge else panels_meta["rows"][family]
        arrays = [(line_panel_name(family, sample), array, (q1 - q0, len(panels_meta["fields"]), rows, width))]
        if family in panels_meta["cross_rows"]:
            arrays.append((line_panel_name(family, sample, cross=True), line["cross"][family],
                           (q1 - q0, len(panels_meta["cross_fields"]), panels_meta["cross_rows"][family], width)))
        for name, value, expected in arrays:
            if tuple(value.shape) != expected or np.dtype(value.dtype) != np.dtype(np.complex128):
                _refuse(f"scratch {name} requires complex128 shape {expected}, got {tuple(value.shape)}")
            if not isinstance(value, jax.Array) or not value.sharding.is_equivalent_to(
                    NamedSharding(mesh_xy, spec), value.ndim):
                _refuse(f"scratch {name} requires NamedSharding(mesh_xy, {spec})")
            if charge and not name.endswith("_cross"):
                arg, output, temporary = _conversion_bytes(basis, value.shape, spec, unpack=True, axis=2)
            else:
                arg = output = _local_bytes(value.shape, value.dtype, mesh_xy, spec)
                temporary = 0
            _admit(ledger, "write_bank_" + name, output, temporary + arg,
                   device_panel=max(arg, output), native_host=True)
            if not bool(jnp.all(jnp.isfinite(value))):
                _refuse(f"scratch {name} contains nonfinite values")
            pending.append((family, name, value, counts))
    return sample, pending


def _prepare_bank_write(header, *, q_span, meta, mesh_xy,
                        sample_span=None, Wc=None, dWc_ds=None,
                        M1=None, M3=None, M0=None, M2=None, constant=None, line=None):
    """Validate/admit a packed span before mutating a bank."""
    _check_basis(meta, header)
    if mesh_xy is not meta.mu_basis.mesh_xy:
        _refuse("writer mesh differs from packed basis mesh")
    if header.get("complete"):
        _refuse("completed scratch bank is immutable")
    shape = header["bank_shape"]
    q0, q1 = _span(q_span, shape["nq"], "q_span")
    has_samples = any(v is not None for v in (Wc, dWc_ds))
    if has_samples and sample_span is None:
        _refuse("scratch sample write requires explicit sample_span")
    if not has_samples and sample_span is not None:
        _refuse("sample_span supplied without sample payload")
    a0, a1 = (_span(sample_span, shape["nsample"], "sample_span")
              if has_samples else (0, 0))
    rows = dense_sample_rows(header, range(a0, a1)) if has_samples else []
    if rows and rows != list(range(rows[0], rows[0] + len(rows))):
        _refuse(f"dense sample_span {(a0, a1)} straddles the line-panel span")
    pending = [(name, value) for name, value in
               (("Wc", Wc), ("dWc_ds", dWc_ds), ("M1", M1), ("M3", M3),
                ("M0", M0), ("M2", M2), ("constant", constant))
               if value is not None]
    if not pending and line is None:
        _refuse("scratch write has no payload")
    basis = meta.mu_basis
    ledger = _capacity(meta)
    _check_io_capacity(ledger,basis.mesh_xy,header)
    if "photon_layout" not in header and int(basis.n_logical) != int(shape["d"]):
        _refuse("scratch centroid extent mismatch")
    masks = _bank_masks(header)
    # Validate every argument before opening the writer: a bad second field
    # must not leave an otherwise legal first field queued in the same call.
    for name, array in pending:
        geometry = _sample_field(header, name)
        sample = geometry is not None
        if not sample and name not in _bank_moment_fields(header):
            _refuse(f"scratch bank has no {name} field (odd moments belong to an ordered bank)")
        if sample:
            key, column, _ = geometry
            marked = masks[key][q0:q1, rows[0]:rows[-1] + 1, column]
        else:
            marked = masks["moment_written"][q0:q1, _bank_moment_fields(header).index(name)]
        if marked.any():
            _refuse(f"scratch {name} span already committed")
        expected = ((q1-q0, a1-a0) if sample else (q1-q0,)) + (
            (shape["d"],) * 2 if "photon_layout" in header else (basis.n_packed,) * 2)
        spec = P(None, None, 'x', 'y') if sample else P(None, 'x', 'y')
        if tuple(array.shape) != expected or np.dtype(array.dtype) != np.dtype(np.complex128):
            _refuse(f"scratch {name} requires complex128 packed shape {expected}")
        if not isinstance(array, jax.Array) or not array.sharding.is_equivalent_to(NamedSharding(mesh_xy, spec), array.ndim):
            _refuse(f"scratch {name} requires NamedSharding(mesh_xy, {spec})")
        if "photon_layout" in header:
            arg = output = _local_bytes(array.shape, array.dtype, mesh_xy, spec)
            temporary = 0
        else:
            arg, output, temporary = _conversion_bytes(basis,array.shape,spec,unpack=True,operator=True)
        # Factor-sized finite-check envelope, separate from caller input.
        _admit(ledger,"write_bank_"+name,output,temporary+arg,
               device_panel=max(arg,output),native_host=True)
        if not bool(jnp.all(jnp.isfinite(array))):
            _refuse(f"scratch {name} contains nonfinite values")
    line = (None if line is None else
            _prepare_line_write(header, line, q0=q0, q1=q1, meta=meta, mesh_xy=mesh_xy, ledger=ledger))
    return q0, q1, rows, pending, line, masks


def _write_bank_payload(io, header, meta, prepared):
    """Write admitted spans; caller publishes masks after the queue drains.

    The prepared arrays carry the public writer's units and face shardings.
    Authentication/admission stays in the common preparation owner.
    """
    q0, q1, rows, pending, line, masks = prepared
    shape, basis = header["bank_shape"], meta.mu_basis
    for name, array in pending:
        geometry = _sample_field(header, name)
        spec = P(None, None, 'x', 'y') if geometry else P(None, 'x', 'y')
        disk_shape = ((shape["nq"], geometry[2]) if geometry
                      else (shape["nq"],)) + (shape["d"], shape["d"])
        io.create_dataset(name, shape=disk_shape, dtype=np.complex128)
        canonical = array if "photon_layout" in header else basis.unpack_operator(array, spec=spec)
        offset = (q0, rows[0], 0, 0) if geometry else (q0, 0, 0)
        io.write_slab(name, canonical, offset=offset)
        # SlabIO's write queue owns canonical until drained. Drain each
        # field so endpoint staging cannot accumulate across fields.
        io.sync_writes()
        del canonical
        if geometry:
            key, column, _ = geometry
            masks[key][q0:q1, rows[0]:rows[-1] + 1, column] = True
        else:
            masks["moment_written"][q0:q1, _bank_moment_fields(header).index(name)] = True
    if line is not None:
        sample, arrays = line
        panels = header["line_panels"]
        p0 = _panel_span(header)[0]
        spec = P(None, None, "x", "y")
        for family, name, array, counts in arrays:
            charge = "photon_layout" not in header and not name.endswith("_cross")
            value = basis.unpack_axis(array, 2, spec=spec) if charge else array
            io.create_dataset(name, shape=(shape["nq"],) + tuple(value.shape[1:]), dtype=np.complex128)
            io.write_slab(name, value, offset=(q0, 0, 0, 0))
            io.sync_writes()
            del value
            if not name.endswith("_cross"):
                panels["width"][family][sample - p0] = int(array.shape[-1])
                table = np.asarray(panels["counts"][family], np.int64)
                table[q0:q1, sample - p0] = counts
                panels["counts"][family] = table.tolist()
        masks["line_written"][q0:q1, sample - p0] = True
        header["payload_bytes"] = int(header["payload_bytes"]) + sum(
            16 * int(np.prod(array.shape)) for _, _, array, _ in arrays)
    for key, mask in masks.items():
        header[key] = mask.tolist()


_BATCH_LAYOUT = ("x", "y")


def _bank_layout(partition_spec):
    """Reader layout from the caller's spec: None is the face; P(('x','y'), None, ...) is batch layout."""
    if partition_spec is None:
        return "face"
    spec = tuple(partition_spec)
    lead = spec[0] if spec else None
    if (isinstance(lead, (tuple, list)) and tuple(lead) == _BATCH_LAYOUT
            and all(entry is None for entry in spec[1:])):
        return "batch"
    _refuse(f"scratch bank partition_spec must be None (face tiles) or P(('x','y'), None, ...) "
            f"(batch layout); got {partition_spec}")


@lru_cache(maxsize=None)
def _bank_face_to_batch(mesh, ndim):
    """Staged face -> batch layout for a [B, ..., mu, nu] bank stack (x then y all_to_all).

    The literal schedule of ``common.staged_reshard.face_to_batch_reshard`` with the
    face axes last: no arithmetic, bit-exact, no rank ever holds another's whole row.
    """
    from common.shard_map import shard_map
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])

    def body(a):
        if px > 1:
            a = jax.lax.all_to_all(a, "x", split_axis=0, concat_axis=ndim - 2, tiled=True)
        if py > 1:
            a = jax.lax.all_to_all(a, "y", split_axis=0, concat_axis=ndim - 1, tiled=True)
        return a
    return jax.jit(shard_map(body, mesh=mesh, in_specs=P(*((None,) * (ndim - 2)), "x", "y"),
                             out_specs=P(_BATCH_LAYOUT, *((None,) * (ndim - 1))), check_vma=False))


@lru_cache(maxsize=None)
def _bank_stack_rows(mesh, spec):
    """Concatenate per-parent reads on the replicated leading axis, pinned to their layout."""
    return jax.jit(lambda rows: jnp.concatenate(rows, axis=0),
                   out_shardings=NamedSharding(mesh, spec))


@lru_cache(maxsize=None)
def _bank_concat_columns(mesh, spec):
    """Retain the bank column join for each mesh and output layout."""
    return jax.jit(lambda *values: jnp.concatenate(values, axis=-1),
                   out_shardings=NamedSharding(mesh, spec))


@lru_cache(maxsize=None)
def _resident_sector_select(mesh, ndim, starts, widths):
    """Local C/T rectangle of each rank's photon face tile, still face tiled."""
    from common.shard_map import shard_map
    spec = P(*((None,) * (ndim - 2)), "x", "y")

    def local(tile):
        return tile[..., starts[0]:starts[0] + widths[0], starts[1]:starts[1] + widths[1]]
    return jax.jit(shard_map(local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))


def _read_photon_bank_sector(io, name, prefix, offset, spec, header, sector, ledger, retained):
    """Read only C/T rectangles; never load the full photon panel to slice it.

    Native union reads select the row windows. Each column window is one
    union read; JAX joins those columns while retaining the requested face
    or parent layout. The resulting axis order is the stored mesh-major
    channel order restricted to the named family (including zero padding).
    """
    layout = header["photon_layout"]
    side = int(layout["mesh_side"])
    c, t = (int(v)//side for v in layout["carrier_extents"][:2])
    stride = c+3*t
    widths = tuple(c if family == "C" else 3*t for family in sector)
    starts = tuple(0 if family == "C" else c for family in sector)
    output_shape = prefix+(side*widths[0],side*widths[1])
    output_bytes = _local_bytes(output_shape,np.complex128,io.mesh,spec)
    if isinstance(io, ResidentBankPayload):
        # The mesh-interleaved carrier puts each rank's C and T rows in its
        # own face tile, so a sector rectangle is a local slice of that tile.
        d = side*stride
        face = P(*((None,)*len(prefix)), 'x', 'y')
        full = _local_bytes(prefix+(d,d),np.complex128,io.mesh,face)
        _admit(ledger,"read_bank_sector_"+name,retained+output_bytes,output_bytes+full,
               device_panel=output_bytes)
        value = io.read_slab(name, shape=prefix+(d,d), valid_shape=prefix+(d,d),
                             offset=offset, dtype=np.complex128, partition_spec=face)
        value = _resident_sector_select(io.mesh, len(prefix)+2, starts, widths)(value)
        if spec is None or tuple(spec) == tuple(face):
            return value
        return _bank_face_to_batch(io.mesh, value.ndim)(value)
    _admit(ledger,"read_bank_sector_"+name,retained+output_bytes,output_bytes,
           device_panel=output_bytes,native_host=True,io=io)
    columns = []
    for column in range(side):
        shape = prefix+widths
        offsets = [offset[:-2]+(row*stride+starts[0],column*stride+starts[1])
                   for row in range(side)]
        array = io.read_slabs(name,shape=shape,offsets=np.asarray(offsets,np.int64),
            valid_shapes=np.asarray([shape]*side,np.int64),partition_spec=spec,
            window_axis=len(prefix),dtype=np.complex128)
        columns.append(array.reshape(prefix+(side*widths[0],widths[1])))
    return _bank_concat_columns(io.mesh, spec)(*columns)


def _parent_runs(ids, layout, mesh):
    """Read runs for a parent id list: (runs, contiguous, interval); see read_shared_pole_bank."""
    contiguous = ids == list(range(ids[0], ids[0] + len(ids)))
    # Ordered parent rounds can permute a complete interval and pad it with
    # repeated parents. Read that interval once, then restore the round order
    # on the already sharded face before the existing face-to-batch exchange.
    # The interval has no more rows than the old per-parent read stack.
    interval = (layout == "batch" and not contiguous
                and len(set(ids)) == max(ids) - min(ids) + 1)
    ranks = int(mesh.shape["x"]) * int(mesh.shape["y"])
    if layout == "batch" and len(ids) % ranks:
        _refuse(f"batch-layout bank read needs a multiple of {ranks} parents, got {len(ids)}")
    if contiguous:
        runs = [(ids[0], len(ids))]
    elif interval:
        runs = [(min(ids), max(ids)-min(ids)+1)]
    else:
        runs = [(q, 1) for q in ids]
    return runs, contiguous, interval


def _parent_ids(header, q_span, q_ids):
    shape = header["bank_shape"]
    if (q_span is None) == (q_ids is None):
        _refuse("scratch bank read takes exactly one of q_span or q_ids")
    if q_ids is None:
        q0, q1 = _span(q_span, shape["nq"], "q_span")
        return list(range(q0, q1))
    ids = [int(v) for v in q_ids]
    if not ids or any(isinstance(v, (bool, np.bool_)) for v in q_ids) or any(
            not 0 <= v < shape["nq"] for v in ids):
        _refuse(f"q_ids out of bounds or empty: {list(q_ids)}, extent {shape['nq']}")
    return ids


def _restore_parent_order(value, ids, layout, contiguous, interval, mesh):
    if layout == "batch" and not contiguous:
        if interval:
            value = jnp.take(value, np.asarray(ids)-min(ids), axis=0)
        value = _bank_face_to_batch(mesh, value.ndim)(value)
    return value


def read_shared_pole_bank(io, q_span=None, *, meta, header, sample_span=None, sample_ids=None,
                          fields=("Wc", "dWc_ds"), q_ids=None, partition_spec=None, sector=None):
    """Read committed bounded scratch fields into packed distributed operators.

    Returns a plain dict of complex128 arrays. Dense samples require explicit
    ``sample_span`` (a contiguous id span) or ``sample_ids`` (ids whose dense
    rows are contiguous, e.g. the fitted samples on either side of the
    line-panel span); no implicit all-bank read is offered, and a line-panel
    sample is read with ``read_line_panels``. The caller authenticates
    ``header`` before opening ``io``.

    Parents are a contiguous ``q_span`` or a list ``q_ids`` (repeats allowed:
    a round's synthetic slots repeat its last parent); the leading axis of
    every field follows that order. ``partition_spec`` chooses the layout:
    ``None`` returns face tiles (``[q, s, mu_X, nu_Y]`` samples,
    ``[q, mu_X, nu_Y]`` moments); ``P(('x','y'), None, ...)`` returns batch
    layout, whole matrices on the rank that owns each parent (rank
    ``x*Py + y`` owns rows ``[r*q/P, (r+1)*q/P)``), for which the number of
    parents must be a multiple of P. A contiguous ascending run is one
    collective read in the requested layout: in batch layout each rank reads
    only its own whole rows. A permuted complete interval is read once in face
    layout, reordered, then moved to batch layout by the staged x-then-y
    exchange. Sparse lists use one face read per parent. Values are identical
    either way.
    """
    if header.get("schema") != BANK_SCHEMA:
        _refuse("scratch bank reader schema mismatch")
    _check_basis(meta, header)
    if io.mesh is not meta.mu_basis.mesh_xy:
        _refuse("reader mesh differs from packed basis mesh")
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields) or any(
            f not in _bank_sample_fields(header) + _bank_moment_fields(header) for f in fields):
        _refuse("scratch bank fields must be distinct Wc/dWc_ds/M1/M3 names "
                "(M0/M2 on an ordered bank)")
    shape = header["bank_shape"]
    layout = _bank_layout(partition_spec)
    if sector is not None and ("photon_layout" not in header or len(sector) != 2
                              or any(v not in ("C", "T") for v in sector)):
        _refuse("sector bank read requires photon metadata and endpoint labels C/T")
    ids = _parent_ids(header, q_span, q_ids)
    mesh = meta.mu_basis.mesh_xy
    runs, contiguous, interval = _parent_runs(ids, layout, mesh)
    need_samples = any(_sample_field(header, name) for name in fields)
    if need_samples and (sample_span is None) == (sample_ids is None):
        _refuse("scratch sample read requires exactly one of sample_span or sample_ids")
    rows = []
    if need_samples:
        if sample_ids is None:
            a0, a1 = _span(sample_span, shape["nsample"], "sample_span")
            sample_ids = range(a0, a1)
        rows = dense_sample_rows(header, sample_ids)
        if not rows or rows != list(range(rows[0], rows[0] + len(rows))):
            _refuse(f"dense samples {list(sample_ids)} are not one contiguous row span")
    basis = meta.mu_basis
    ledger = _capacity(meta)
    _check_io_capacity(ledger,basis.mesh_xy,header)
    if "photon_layout" not in header and int(basis.n_logical) != int(shape["d"]):
        _refuse("scratch centroid extent mismatch")
    masks = _bank_masks(header)
    unique = sorted(set(ids))
    for name in fields:
        geometry = _sample_field(header, name)
        if geometry:
            key, column, _ = geometry
            marked = masks[key][unique, rows[0]:rows[-1] + 1, column]
        elif name in _bank_moment_fields(header):
            marked = masks["moment_written"][unique, _bank_moment_fields(header).index(name)]
        else:
            _refuse(f"scratch bank has no {name} field")
        if not marked.all():
            _refuse(f"scratch bank {name} requested span is incomplete")
    out = {}
    retained = 0
    for name in fields:
        geometry = _sample_field(header, name)
        sample = geometry is not None
        face = P(None, None, 'x', 'y') if sample else P(None, 'x', 'y')
        spec = face if layout == "face" or not contiguous else (
            P(_BATCH_LAYOUT, None, None, None) if sample else P(_BATCH_LAYOUT, None, None))
        parts = []
        for q, count in runs:
            prefix = (count, len(rows)) if sample else (count,)
            offset = (q, rows[0], 0, 0) if sample else (q, 0, 0)
            if sector is not None:
                row = _read_photon_bank_sector(io, name, prefix, offset, spec, header, sector, ledger, retained)
                retained += _local_bytes(row.shape, row.dtype, mesh, spec)
                parts.append(row)
                continue
            d = shape["d"] if "photon_layout" in header else basis.n_canonical
            logical = shape["d"] if "photon_layout" in header else basis.n_logical
            if "photon_layout" in header:
                arg = output = _local_bytes(prefix+(d,d), np.complex128, mesh, spec)
                temporary = 0
            else:
                arg, output, temporary = _conversion_bytes(basis,
                    prefix+(d,d),spec,unpack=False,operator=True)
            _admit(ledger,"read_bank_"+name,retained+arg+output,temporary,
                   device_panel=arg,native_host=True,io=io)
            canonical = io.read_slab(
                name, shape=prefix + (d,d),
                valid_shape=prefix + (logical,logical),
                offset=offset, dtype=np.complex128, partition_spec=spec)
            row = canonical if "photon_layout" in header else basis.pack_operator(canonical, spec=spec)
            retained += output
            del canonical
            parts.append(row)
        value = parts[0] if len(parts) == 1 else _bank_stack_rows(mesh, spec)(tuple(parts))
        del parts
        out[name] = _restore_parent_order(value, ids, layout, contiguous, interval, mesh)
    return out


def read_line_panels(io, q_span=None, *, meta, header, family, sample, cross=False,
                     q_ids=None, partition_spec=None):
    """One line sample's panels for one family and their retained counts.

    Returns ``(panels, counts)``: complex128 ``[b, 1+2S, rows, r]`` (or the
    ``[b, 2S, cross rows, r]`` cross panel) in the packed row order the
    constructor uses, face tiles ``P(None,None,'x','y')`` or batch layout as
    in :func:`read_shared_pole_bank`, and the host int ``[b]`` retained count
    of each requested parent. The field order is the header's
    ``line_panels.fields``: Q, then output and action per state.
    """
    if header.get("schema") != BANK_SCHEMA:
        _refuse("scratch bank reader schema mismatch")
    _check_basis(meta, header)
    if io.mesh is not meta.mu_basis.mesh_xy:
        _refuse("reader mesh differs from packed basis mesh")
    panels = header["line_panels"]
    p0, p1 = _panel_span(header)
    sample = int(sample)
    if not p0 <= sample < p1 or family not in panels["rows"]:
        _refuse(f"no line panels for family {family} at sample {sample}")
    if cross and family not in panels["cross_rows"]:
        _refuse(f"family {family} has no cross panel")
    ids = _parent_ids(header, q_span, q_ids)
    if not np.asarray(header["line_written"], bool)[sorted(set(ids)), sample - p0].all():
        _refuse(f"line sample {sample} requested parents are incomplete")
    layout = _bank_layout(partition_spec)
    mesh = meta.mu_basis.mesh_xy
    runs, contiguous, interval = _parent_runs(ids, layout, mesh)
    width = int(panels["width"][family][sample - p0])
    charge = "photon_layout" not in header and not cross
    fields = len(panels["cross_fields" if cross else "fields"])
    rows = panels["cross_rows"][family] if cross else panels["rows"][family]
    name = line_panel_name(family, sample, cross=cross)
    face = P(None, None, "x", "y")
    spec = face if layout == "face" or not contiguous else P(_BATCH_LAYOUT, None, None, None)
    basis = meta.mu_basis
    ledger = _capacity(meta)
    _check_io_capacity(ledger, basis.mesh_xy, header)
    parts, retained = [], 0
    for q, count in runs:
        shape = (count, fields, rows, width)
        if charge:
            arg, output, temporary = _conversion_bytes(basis, shape, spec, unpack=False, axis=2)
        else:
            arg = output = _local_bytes(shape, np.complex128, mesh, spec)
            temporary = 0
        _admit(ledger, "read_" + name, retained + arg + output, temporary,
               device_panel=arg, native_host=True, io=io)
        value = io.read_slab(name, shape=shape, valid_shape=shape, offset=(q, 0, 0, 0),
                             dtype=np.complex128, partition_spec=spec)
        parts.append(basis.pack_axis(value, 2, spec=spec) if charge else value)
        retained += output
        del value
    value = parts[0] if len(parts) == 1 else _bank_stack_rows(mesh, spec)(tuple(parts))
    del parts
    counts = np.asarray(panels["counts"][family], np.int64)[ids, sample - p0]
    return _restore_parent_order(value, ids, layout, contiguous, interval, mesh), counts


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


#: The exact export names one self-consistent map owns. Used twice and only
#: here: as the eligibility test for the export just written, and as the scan
#: pattern that releases its predecessors. A one-shot export ("oneshot_w.h5")
#: does not match, so it neither scans nor is scanned for.
_MANAGED_EXPORT = r"sc_[0-9]{4}_(?:poles|w)\.h5"


def _retain_current_map_exports(targets, *, run_dir, label, identity, mesh_xy,
                                print_fn):
    """Release earlier maps' exports once this map's are proven complete.

    An export is named for its map, so under self-consistency an N-map run
    would otherwise leave N full banks on disk -- a cost linear in maps. Only
    the exact managed names ``sc_NNNN_{poles,w}.h5`` are eligible, and only
    when the export just written is itself one of them, so a one-shot run
    scans nothing, unlinks nothing and takes no extra collective.

    Order is write, authenticate, then release: the current export is read
    back through the store's own commit and digest owners before anything is
    unlinked, so a predecessor is discarded only after its replacement exists
    and is complete. Scanning rather than unlinking only ``N-1`` also clears
    managed exports a longer earlier run left in the same scratch directory.
    This is ``gw.mpa.model.retain_iteration_artifacts``'s rule for the
    shared-pole exports, through the same removal owner.
    """
    import os
    import re
    from gw.qsgw_utils import remove_managed

    if not targets or not all(re.fullmatch(_MANAGED_EXPORT, path.name)
                              for path in targets.values()):
        return ()
    for kind, path in targets.items():
        if kind == "poles":
            validate_shared_pole_model(path, expected_identity=identity,
                                       mesh_xy=mesh_xy)
        else:
            validate_shared_pole_bank(path, expected_identity=identity,
                                      mesh_xy=mesh_xy, require_complete=True)
    root = os.path.abspath(os.fspath(run_dir))
    removed = remove_managed(
        root, _MANAGED_EXPORT,
        keep=[os.path.join(root, path.name) for path in targets.values()],
        barrier_tag=f"shared_pole_store.retain.{label}", print_fn=print_fn)
    if removed:
        print_fn(f"shared-pole exports: retained {label}; released "
                 + ", ".join(sorted(removed)))
    return tuple(sorted(removed))


def _export_is_current(path, *, identity, digest):
    """Whether an existing export was written from THIS model.

    Metadata only, and bounded: both writers stamp the model identity into
    the export's header, and the pole writer stamps the source model digest
    into every construction receipt beside it.  An export that is not
    committed, or carries another identity or digest, is not this model's
    and the caller refuses it rather than deciding on its behalf.
    """
    try:
        header = _read_header(path)
    except (OSError, ValueError, KeyError):
        return False
    if _json(header.get("identity")) != _json(dict(identity)):
        return False
    stamped = {row.get("receipt", {}).get("source_model_digest")
               for row in header.get("construction_receipts", ())}
    return stamped <= {digest, None}


def export_shared_pole_outputs(handle, *, meta, config, mesh_xy, source_wfn,
                               run_dir, label, tables, print_fn):
    """Export current-map poles and/or fixed W samples through their writers.

    Large arrays stay XY tiled. The copy loop holds one parent (and one
    frequency for W) at a time; authentication uses the existing bounded
    parent/column panels and exchanges only row hashes. Existing packing, admission, commit and digest owners
    are reused; the source stores are immutable, including on restart.
    The bank export retains its derivative and moment companions so it is
    readable by the existing bank reader. No screening or pole fit is rerun.
    A managed self-consistent export releases its predecessors once it is
    itself authenticated (:func:`_retain_current_map_exports`), so the
    retained set is one map, not one per map; a one-shot export is untouched.

    ``write_poles`` is the production export. ``config.debug.write_w`` is a
    DEBUG dump of the whole frequency sample bank and is not needed for BSE:
    the exported model evaluates as ``Wc(s) = b (s - Lambda)^-1 b^dagger``
    with ``s = z^2``, so ``Wc(omega=0) = -b Lambda^-1 b^dagger`` exactly from
    ``(b, Lambda)``, and the bank itself carries no ``omega = 0`` sample.
    Both kinds go through the same managed retention above, so a debug bank
    under self-consistency is also bounded to one map.
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
               (("poles", config.write_poles), ("w", config.debug.write_w)) if enabled}
    # A RESTART MUST NOT RE-EXPORT OVER ITS OWN EXPORT.  The export is
    # written only once a map is complete, so a committed file on disk is a
    # finished export; a restart that rebuilds nothing has nothing new to
    # write, and rewriting it would cost a full bank (20.9 GB on the Na
    # reference) to reproduce bytes that are already there.  Skipping is a
    # receipt line, not silence, and the question is answered from the
    # export's own stamped provenance rather than by re-reading tensors --
    # a full re-validation is exactly the work a restart exists to avoid.
    # Anything else keeping that name still refuses: the overwrite guard is
    # what protects a one-shot run from a second map writing over it.
    # ``pending`` is what still has to be WRITTEN; ``targets`` stays whole,
    # because it is also this map's keep-set for retention below and a
    # retained export must not be released as if it belonged to an earlier map.
    pending = dict(targets)
    for kind, path in targets.items():
        if not path.exists():
            continue
        if not _export_is_current(path, identity=handle["identity"],
                                  digest=handle["digest"]):
            _refuse(f"export already exists: {path}; use a fresh output directory")
        del pending[kind]
        outputs[kind] = dict(path=str(path), status="retained", reason=(
            "export already written from this model identity and digest; "
            "restart rewrote nothing"))
        print_fn(f"shared-pole export: {kind} at {path} was written from this "
                 "model; retained, not rewritten")
    bank_source = source.parent / "bank.h5"
    if "w" in pending:
        if not bank_source.is_file():
            _refuse(f"write_w needs the current-map bank {bank_source}; "
                    "a model-only restart cannot supply frequency samples")
        bank_header = validate_shared_pole_bank(bank_source,
            expected_identity=handle["identity"], mesh_xy=mesh_xy, require_complete=True)
    if "poles" in pending:
        path = pending["poles"]
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
                            source_model_digest=model["digest"], source_store=str(source)),
                        ordered=bool(model.get("ordered", False)))
                    del canonical, b, poles
                finally:
                    ledger.live_stages = previous
        _export_spatial_header(path, source_wfn, meta, kind="poles", source=source)
        outputs["poles"] = dict(path=str(path), payload_bytes=model["compact_payload_bytes"])
    if "w" in pending:
        path = pending["w"]
        output_header = initialize_shared_pole_bank(path, meta=meta, tables=tables,
            recipe=bank_header["recipe"], identity=handle["identity"], mesh_xy=mesh_xy)
        previous = ledger.live_stages
        with SlabIO(bank_source, mode="r", mesh=mesh_xy) as io, \
                SlabIO(path, mode="a", mesh=mesh_xy) as output_io:
            p0, p1 = _panel_span(bank_header)
            dense = [i for i in range(bank_header["bank_shape"]["nsample"]) if not p0 <= i < p1]
            for q in range(bank_header["bank_shape"]["nq"]):
                for field in _bank_sample_fields(bank_header) + _bank_moment_fields(bank_header):
                    sample = _sample_field(bank_header, field) is not None
                    for i in (dense if sample else (None,)):
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
                # A line sample is exported as stored: its directions and actions.
                for i in range(p0, p1):
                    panels = {family: read_line_panels(io, (q, q+1), meta=meta, header=bank_header,
                                                       family=family, sample=i)
                              for family in bank_header["line_panels"]["rows"]}
                    row = _admit(ledger, "export_w_live", sum(
                        _local_bytes(v.shape, v.dtype, mesh_xy, P(None, None, "x", "y"))
                        for v, _ in panels.values()))
                    ledger.live_stages = (*previous, row["stage"])
                    try:
                        prepared = _prepare_bank_write(output_header, q_span=(q, q+1), meta=meta,
                            mesh_xy=mesh_xy, line=dict(sample=i,
                                panels={f: v for f, (v, _) in panels.items()},
                                counts={f: c for f, (_, c) in panels.items()}))
                        _write_bank_payload(output_io, output_header, meta, prepared)
                        del prepared, panels
                    finally:
                        ledger.live_stages = previous
            _write_bank_masks(output_io, output_header)
        _complete_bank(path, output_header)
        _export_spatial_header(path, source_wfn, meta, kind="w", source=bank_source)
        outputs["w"] = dict(path=str(path), payload_bytes=bank_header["payload_bytes"])
    for kind, receipt in outputs.items():
        size = receipt.get("payload_bytes")
        print_fn(f"write_{kind}: {receipt['path']}; "
                 + (f"payload={size} bytes" if size is not None
                    else f"status={receipt['status']}"))
    _retain_current_map_exports(targets, run_dir=run_dir, label=label,
        identity=handle["identity"], mesh_xy=mesh_xy, print_fn=print_fn)
    return outputs
