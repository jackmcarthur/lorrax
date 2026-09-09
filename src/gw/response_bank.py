"""Physical response-bank algebra (DESIGN §3.1, DBANK D5–D6).

Operators have packed centroid-major endpoints ``mu * nspinor + spin``.
The public bank supports scalar representations; disk conversion belongs to
the scratch writer. Dense products and solves enter through ``distrib_la``.
"""
from functools import partial
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def response_algebra(meta, config, *, mesh_xy, n):
    """Plan physical Dyson samples and exact high-frequency moments.

    Parameters
    ----------
    meta, config
        Current physical-state metadata and resolved runtime configuration.
        The prefactor always uses the full physical q count in ``meta``.
    mesh_xy
        Named ``x,y`` mesh for all face operators.
    n : int
        Packed endpoint extent; padding is supplied by the centroid owner.

    Returns
    -------
    samples, moments, receipt
        Jitted functions accepting complex128 ``[b,n,n]`` operators at
        ``P(None,'x','y')``, and the resolved backend/prefactor description.
        ``samples(H, chi_raw, dchi_raw)`` returns physical Wc (Ry) and its
        s derivative (Ry^-1). ``moments(H, A0, A1)`` takes already scaled
        bare-response expansion coefficients and returns M1/M3 (Ry^3/Ry^5).
        Neither routine Hermitizes its inputs or outputs.
    """
    from distrib_la import matmul, plan
    from .gw_config import linalg_resolution
    from .w_isdf import _w_solve_pref_scalar

    resolution = linalg_resolution(
        config if hasattr(config, "get") else {"linalg": config.backend.linalg})
    route = resolution.batched_route
    backend = "off" if resolution.layout == "local" else "distributed"
    lu = plan("solve_lu", mesh_xy, backend=backend, n=n,
              batched_route=route)
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    pref = _w_solve_pref_scalar(meta)

    def mm(a, b):
        return matmul(a, b, mesh=mesh_xy, backend=backend,
                      batched_route=route)

    def congruence(h, a):
        return mm(mm(h, a), h)

    @partial(jax.jit, in_shardings=(face, face, face),
             out_shardings=(face, face))
    def samples(h, chi_raw, dchi_raw):
        # X=p0 H chi_raw H; Wc=H X (I-X)^-1 H.  The derivative
        # uses E on BOTH sides, never E† (off-axis E is not Hermitian).
        x = pref * congruence(h, chi_raw)
        xd = pref * congruence(h, dchi_raw)
        identity = jnp.broadcast_to(jnp.eye(n, dtype=h.dtype), x.shape)
        e = lu.batched(identity - x, identity.copy())
        return congruence(h, mm(x, e)), congruence(h, mm(mm(e, xd), e))

    @partial(jax.jit, in_shardings=(face, face, face),
             out_shardings=(face, face))
    def moments(h, a0, a1):
        # chi_scaled=A0/s+A1/s²; whitened Dyson coefficients are
        # B0=H A0 H, B1=H A1 H, S0=B0, S1=B1+B0².
        b0 = congruence(h, a0)
        b1 = congruence(h, a1)
        return (0.5 * congruence(h, b0),
                0.5 * congruence(h, b1 + mm(b0, b0)))

    return samples, moments, {
        "linalg": resolution.layout,
        "solve": lu.describe(),
        "batched_route": route,
        "prefactor": pref,
        "prefactor_q_count": int(meta.nk_tot),
        "moment_convention": "S_m = 2 M_(2m+1) in physical coordinates",
        "units": {"Wc": "Ry", "dWc_ds": "Ry^-1",
                  "M1": "Ry^3", "M3": "Ry^5"},
    }


def response_weights(wfns, meta):
    """Return exact current-state screening-band weights, with carrier bands masked.

    All returned tables are replicated ``[full_k, band]`` arrays. Energy
    powers use Ry and are evaluated about the mean physical-band energy;
    the binomial expansion is independent of that reference. No occupation
    activity floor enters the exact moments.
    """
    from common.collectives import gather_to_host

    energy = np.asarray(gather_to_host(wfns.enk), dtype=np.float64)
    occupied = np.asarray(gather_to_host(wfns.occ), dtype=np.float64)
    stop = min(int(meta.b_id_4_chi_user), int(wfns.slices.b4_logical))
    first = int(wfns.slices.b0)
    count = stop - first
    logical_count = int(wfns.slices.b4_logical) - first
    if (energy.shape != occupied.shape or count <= 0
            or count > energy.shape[1] or not np.isfinite(energy).all()
            or not np.isfinite(occupied).all()):
        raise ValueError("GATE response_occupations: got invalid band table; "
                         "want current finite screening bands; "
                         "why: exact response uses both occupation sectors")
    physical = np.arange(energy.shape[1])[None, :] < count
    f = np.where(physical, occupied, 0.0)
    u = np.where(physical, 1.0 - occupied, 0.0)
    reference = float(np.mean(energy[:, :count]))
    return energy, f, u, reference, {
        "band_start": first, "band_stop": stop,
        "band_carrier": energy.shape[1],
        "energy_sha256": hashlib.sha256(energy[:, :logical_count].tobytes()).hexdigest(),
        "occupation_sha256": hashlib.sha256(occupied[:, :logical_count].tobytes()).hexdigest(),
        "energy_min_ry": float(energy[:, :count].min()),
        "energy_max_ry": float(energy[:, :count].max()),
        "occupation_activity_floor": 0.0,
        "discarded_occupation_mass": 0.0,
    }


def response_stream(wfns, meta, *, mesh_xy, q_ids, n_outputs,
                    pair_mode="retarded"):
    """Bind the existing one-particle Green/FFT primitive to a q batch.

    Returns a jitted kernel and its fixed ψ/energy arguments. Caller supplies
    time, projections, final weights and energy reference. The output is
    ``[len(q_ids), n_outputs, mu_p, mu_p]`` with both endpoints sharded.
    """
    from .w_isdf import _get_chi_fractional_contour_kernel_face

    if wfns.layout != "face" or int(meta.nspinor) != 1:
        raise ValueError("GATE response_representation: got non-scalar or legacy "
                         "wavefunctions; want scalar face carrier; why: bank "
                         "requires explicit endpoint shardings")
    carrier = wfns.green_parent
    source = wfns if carrier is None else carrier
    parent = None if carrier is None else carrier.plan
    nk = int(meta.nk_tot) if parent is None else int(parent.n_parent)
    n = int(meta.mu_basis.n_packed)
    kernel = _get_chi_fractional_contour_kernel_face(
        mesh_xy, (meta.nkx, meta.nky, meta.nkz), n_outputs,
        (nk, int(wfns.slices.nb_full), n, int(meta.nspinor)),
        k_unfold_plan=parent, selected_q=tuple(q_ids), pair_mode=pair_mode)
    return kernel, (source.psi_mun, source.psi_nmu, source.enk)


def stream_weights(wfns, weights, mesh_xy):
    """Place small band weights and restrict to existing raw parents."""
    from common.collectives import replicate_to_mesh

    result = replicate_to_mesh(np.asarray(weights), mesh_xy)
    if wfns.green_parent is not None:
        result = wfns.green_parent.plan.parent_rows(result)
    return result


def exact_bare_moments(wfns, meta, *, mesh_xy, q_ids, execute):
    """Compute A0/A1 of scaled chi=A0/s+A1/s² by six correlations.

    The binomial coefficients expand ``(E_u-E_f)`` and its cube. Imaginary
    particle weights turn the retarded primitive's difference into the sum
    of both orientations at t=0 (Run183 energy-power owner). ``execute``
    admits compiled aggregate memory before calling each kernel and records
    its timing. Returned moments are face-sharded ``[b,mu_p,mu_p]``.
    """
    from .w_isdf import _w_solve_pref_scalar

    energy, f, u, reference, census = response_weights(wfns, meta)
    erel = energy - reference
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
                                    q_ids=q_ids, n_outputs=1)
    terms = (((-1., 1, 0), (1., 0, 1)),
             ((-1., 3, 0), (3., 2, 1), (-3., 1, 2), (1., 0, 3)))
    totals = []
    for moment_terms in terms:
        total = None
        for coefficient, a, b in moment_terms:
            weight_f = stream_weights(wfns, f * erel**a, mesh_xy)
            weight_u = stream_weights(wfns, -1j * u * erel**b, mesh_xy)
            args = (jnp.asarray([0.]), jnp.asarray([[1. + 0j]]), *fixed,
                    weight_f.astype(jnp.complex128), weight_u,
                    jnp.asarray(reference))
            raw = execute(kernel, args, "moment_correlation")[:, 0]
            term = (_w_solve_pref_scalar(meta) * coefficient) * raw
            total = term if total is None else total + term
            total.block_until_ready()
        totals.append(total)
    return (*totals, census)


def _bank_context(wfns, meta, sym, bank_io, mesh_xy):
    """Authenticate A/B's existing scratch transaction and physical state."""
    from file_io.shared_pole_store import validate_shared_pole_bank

    if int(meta.nspinor) != 1 or not bool(sym.trs_allowed):
        raise ValueError("GATE response_representation: want scalar and "
                         "authenticated TRS; odd/open-spin bank is unsupported")
    header = validate_shared_pole_bank(bank_io["path"],
        expected_identity=bank_io["identity"], mesh_xy=mesh_xy)
    qids = np.asarray(sym.q_irr_full_idx, dtype=np.int64)
    if not np.array_equal(qids, header["q_irr_full_idx"]):
        raise ValueError("GATE response_q_identity: scratch parent order differs")
    authenticate_coulomb(bank_io, qids)
    _, _, _, _, census = response_weights(wfns, meta)
    for name, field in (("energies", "energy_sha256"),
                        ("occupations", "occupation_sha256")):
        if bank_io["identity"][name] != census[field]:
            raise ValueError(f"GATE response_state_identity: stale {name}")
    return header, qids, census


@lru_cache(maxsize=16)
def _resource_hash(path, size, mtime_ns):
    """Hash one immutable resource generation on the designated root."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def authenticate_coulomb(bank_io, qids):
    """Authenticate bounded-read Coulomb resource against its fixed identity."""
    from jax.experimental import multihost_utils

    resource = bank_io["coulomb"]
    if resource["basis"] != "canonical" or not np.array_equal(
            resource["q_irr_full_idx"], qids):
        raise ValueError("GATE response_coulomb_identity: wrong basis/q order")
    path = Path(resource["path"])
    stat = path.stat()
    digest = np.zeros(32, dtype=np.uint8)
    if jax.process_index() == 0:
        digest[:] = np.frombuffer(bytes.fromhex(_resource_hash(
            str(path), stat.st_size, stat.st_mtime_ns)), dtype=np.uint8)
    digest = multihost_utils.broadcast_one_to_all(digest)
    if bytes(np.asarray(digest)).hex() != resource["sha256"]:
        raise ValueError("GATE response_coulomb_identity: content hash differs")


def _reserve(meta, stage, resident, workspace=0):
    """Reserve a uniquely named actual-batch footprint in the shared ledger."""
    ledger = meta.shared_pole_capacity
    name = f"{stage}:{len(ledger.entries)}"
    row = ledger.reserve(name, resident_bytes_per_rank=int(resident),
                        workspace_bytes_per_rank=int(workspace),
                        concurrent_with=ledger.live_stages)
    return name, row


def _bank_execution(meta, mesh_xy, bank_io, receipt):
    """Compile and admit new dense work; stream outputs are reserved by batch."""
    def execute(kernel, args, stage):
        started = time.monotonic()
        executable = kernel.lower(*args).compile()
        receipt["seconds"]["compilation"] = (receipt["seconds"].get("compilation", 0.)
            + time.monotonic() - started)
        memory = executable.memory_analysis()
        if memory is None:
            raise ValueError("GATE response_capacity: compiled memory unavailable")
        stream = stage in ("real_time", "laplace", "moment_correlation")
        if not stream:
            _, row = _reserve(meta, stage, memory.argument_size_in_bytes,
                memory.output_size_in_bytes + memory.temp_size_in_bytes)
            receipt["memory"].append(row)
        receipt["compiled"].append(dict(stage=stage,
            arguments=memory.argument_size_in_bytes, outputs=memory.output_size_in_bytes,
            temporaries=memory.temp_size_in_bytes, inherited_stream=stream))
        started = time.monotonic()
        result = executable(*args)
        jax.block_until_ready(result)
        receipt["seconds"][stage] = receipt["seconds"].get(stage, 0.) + time.monotonic()-started
        return result
    return execute


@lru_cache(maxsize=8)
def _coulomb_algebra(mesh_xy, n_packed, n_logical, layout):
    """One cached service plan for H=V^(1/2) and its supported inverse."""
    from distrib_la import matmul, plan
    from .gw_config import linalg_resolution
    resolution = linalg_resolution({"linalg": layout})
    backend = "off" if resolution.layout == "local" else "distributed"
    eig = plan("eigh", mesh_xy, backend=backend, n=n_packed,
               batched_route=resolution.batched_route)
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    rep = NamedSharding(mesh_xy, P())

    @partial(jax.jit, in_shardings=(face,), out_shardings=(face, face, rep, rep))
    def sqrt_v(value):
        lam, vectors = eig.batched(value)
        tolerance = n_logical * np.finfo(np.float64).eps
        scale = jnp.max(jnp.abs(lam), axis=-1, keepdims=True)
        supported = lam > tolerance * scale
        root = jnp.sqrt(jnp.where(supported, lam, 0.0))
        h = matmul(vectors * root[:, None, :], vectors, transb="C",
                   mesh=mesh_xy, backend=backend, batched_route=resolution.batched_route)
        inverse = jnp.where(supported, 1.0 / jnp.where(supported, root, 1.0), 0.0)
        hi = matmul(vectors * inverse[:, None, :], vectors, transb="C",
                    mesh=mesh_xy, backend=backend, batched_route=resolution.batched_route)
        return h, hi, jnp.any(lam < -tolerance * scale), jnp.sum(supported, axis=-1)
    return sqrt_v


def _coulomb_batch(meta, config, bank_io, mesh_xy, q_span, execute):
    """Read one authenticated canonical V batch; convert through its owner."""
    from file_io.slab_io import SlabIO
    basis = meta.mu_basis
    shape = (q_span[1]-q_span[0], basis.n_canonical, basis.n_canonical)
    spec = P(None, "x", "y")
    abstract = jax.ShapeDtypeStruct(shape, jnp.complex128,
                                   sharding=NamedSharding(mesh_xy, spec))
    packed = jax.jit(lambda v: basis.pack_operator(v, spec=spec))
    compiled = packed.lower(abstract).compile()
    memory = compiled.memory_analysis()
    _reserve(meta, "coulomb_read_pack", memory.argument_size_in_bytes,
             memory.output_size_in_bytes + memory.temp_size_in_bytes)
    resource = bank_io["coulomb"]
    with SlabIO(resource["path"], mode="r", mesh=mesh_xy) as io:
        canonical = io.read_slab(resource["dataset"], shape=shape,
            offset=(q_span[0], 0, 0), partition_spec=spec)
        v = compiled(canonical)
        v.block_until_ready()
    del canonical
    layout = config.get("linalg", "local") if hasattr(config, "get") else config.backend.linalg
    kernel = _coulomb_algebra(mesh_xy, basis.n_packed, basis.n_logical, layout)
    h, hi, negative, ranks = execute(kernel, (v,), "coulomb_sqrt")
    if bool(negative):
        raise ValueError("GATE response_coulomb_psd: resolved negative eigenvalue")
    return h, hi, np.asarray(ranks).tolist()


def bank_points(sample_plan):
    """Deduplicate physical evaluations while preserving the role map.

    IINPUTS owns the flat role vocabulary. Infinity has distinct_id=-1 and
    is not a frequency evaluation. Every finite ID must describe exactly
    one upper-half-plane point and cannot mix held and fitted roles.
    """
    z = np.asarray(sample_plan["z_ry"], dtype=np.complex128)
    ids = np.asarray(sample_plan["distinct_id"], dtype=np.int64)
    held = np.asarray(sample_plan["held"], dtype=bool)
    role = np.asarray(sample_plan["role"])
    if z.ndim != 1 or not (z.shape == ids.shape == held.shape == role.shape):
        raise ValueError("GATE response_sample_plan: mismatched role arrays")
    finite = sorted(set(ids[ids >= 0].tolist()))
    if not finite or finite != list(range(len(finite))):
        raise ValueError("GATE response_sample_plan: noncontiguous evaluation IDs")
    points = []
    for sample_id in finite:
        rows = ids == sample_id
        point = z[rows][0]
        if (not np.isfinite(point) or point.imag <= 0
                or not np.all(z[rows] == point)
                or not np.all(held[rows] == held[rows][0])):
            raise ValueError("GATE response_sample_plan: mixed point/held roles "
                             "or noncausal evaluation")
        points.append(point)
    if len(set(points)) != len(points):
        raise ValueError("GATE response_sample_plan: duplicate physical evaluations")
    return np.asarray(points, dtype=np.complex128)


def response_windows(energy, f, u, *, chemical_potential_ry):
    """Partition the Run183/188 windowed response into one stream and cells.

    The campaign window is [-35,40] eV relative to the current chemical
    potential and its sample-only activity floor is 1e-14. Exact moments do
    not call this routine. Bounds are extrema over all k/band pairings, so
    they cover every q without constructing a transition table.
    """
    from common.units import RYD_TO_EV

    ft = np.where(np.abs(f) >= 1e-14, f, 0.0)
    ut = np.where(np.abs(u) >= 1e-14, u, 0.0)
    physical = (f != 0) | (u != 0)
    ev = (energy - chemical_potential_ry) * RYD_TO_EV
    masks = [physical & (ev < -35.0),
             physical & (ev >= -35.0) & (ev <= 40.0),
             physical & (ev > 40.0)]
    # A remote diagonal with both sectors occupied cannot be discarded.
    # Merge it into the resonant stream before constructing any cells.
    for idx in (0, 2):
        if np.any(ft * masks[idx]) and np.any(ut * masks[idx]):
            masks[1] |= masks[idx]
            masks[idx] = np.zeros_like(masks[idx])
    cells = []
    for lower, upper in ((0, 1), (0, 2), (1, 2)):
        bounds, active = [], 0
        for lw, uw in ((ft, ut), (ut, ft)):
            low = energy[masks[lower] & (lw != 0)]
            high = energy[masks[upper] & (uw != 0)]
            if low.size and high.size:
                bounds.append((float(high.min()-low.max()),
                               float(high.max()-low.min())))
                active += int(low.size * high.size)
        if not bounds:
            continue
        lo = energy[masks[lower] & ((ft != 0) | (ut != 0))]
        hi = energy[masks[upper] & ((ft != 0) | (ut != 0))]
        refs = (float(lo.max()), float(hi.min()))
        if refs[1] <= refs[0]:
            raise ValueError("GATE response_remote_cell: unordered energy windows")
        cells.append(dict(lower=lower, upper=upper,
            delta_min_ry=min(v[0] for v in bounds),
            delta_max_ry=max(v[1] for v in bounds),
            references_ry=refs, active_global_pairs=active))
    receipt = dict(window_ev_relative_mu=[-35.0, 40.0],
        occupation_activity_floor=1e-14,
        discarded_f_mass=float(np.sum(np.abs(f-ft))),
        discarded_u_mass=float(np.sum(np.abs(u-ut))),
        bound_scope="all active k/band extrema, safe for every q")
    return masks, ft, ut, cells, receipt


def authenticate_sample_plan(sample_plan, header):
    """Compare the caller's actual role plan with the authenticated scratch."""
    stored = header["bank_sample_plan"]
    if sample_plan["role_codes"] != stored["role_codes"]:
        raise ValueError("GATE response_sample_identity: role vocabulary differs")
    for key in ("z_ry", "role", "distinct_id", "held", "support_pair", "fit_ids", "held_ids"):
        values = stored[key]
        if key == "z_ry":
            values = [complex(v["real"], v["imag"]) if isinstance(v, dict)
                      else v for v in values]
        if not np.array_equal(np.asarray(sample_plan[key]),
                              np.asarray(values), equal_nan=True):
            raise ValueError(f"GATE response_sample_identity: stale {key}")
    return header["bank_plan_digest"]


def _receipt(stage, census, bank_io):
    """Start an incomplete, state-bound A/B execution receipt."""
    return dict(schema="lorrax.response-bank.v1", stage=stage,
        identity=dict(bank_io["identity"]), census=census,
        job=os.getenv("SLURM_JOB_ID"), step=os.getenv("SLURM_STEP_ID"),
        coulomb_identity=dict(bank_io["coulomb"]), seconds={}, memory=[],
        completion=False, correlation_count=0, batches=[], compiled=[],
        native_workspace_status="NOT_MEASURED",
        native_workspace_reason="ISERV provider bounds pending; compiled admission only",
        peak_bytes=None, peak_reason="execution has not completed",
        moment_convention="S_m = 2 M_(2m+1) in physical coordinates",
        units={"Wc": "Ry", "dWc_ds": "Ry^-1", "M1": "Ry^3", "M3": "Ry^5"})


def _stream_comparison(wfns, meta, mesh_xy, qids, receipt):
    """Record ruling9's matched two-output compile bound on this geometry."""
    ledger = meta.shared_pole_capacity
    if ledger.stream_peak["status"] != "NOT_MEASURED":
        return
    from .w_isdf import _get_chi_fractional_contour_kernel_face
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
                                    q_ids=(int(qids[0]),), n_outputs=2)
    _, f, u, ref, _ = response_weights(wfns, meta)
    args = (jnp.zeros(1000), jnp.ones((2,1000), dtype=jnp.complex128), *fixed,
            stream_weights(wfns, f, mesh_xy), stream_weights(wfns, u, mesh_xy), jnp.asarray(ref))
    parent = wfns.green_parent
    nk = meta.nk_tot if parent is None else parent.plan.n_parent
    old = _get_chi_fractional_contour_kernel_face(mesh_xy,
        (meta.nkx,meta.nky,meta.nkz), 2,
        (nk,wfns.slices.nb_full,meta.mu_basis.n_packed,meta.nspinor),
        k_unfold_plan=None if parent is None else parent.plan)
    sizes = []
    for name, item in (("bank", kernel),("incumbent",old)):
        executable = item.lower(*args).compile()
        m = executable.memory_analysis()
        sizes.append(m.argument_size_in_bytes+m.output_size_in_bytes+m.temp_size_in_bytes-m.alias_size_in_bytes)
        receipt.setdefault("stream_comparison", {})[name] = str(m)
    ledger.record_stream_peak(*sizes,
        reason=f"matched two-output/1000-node raw-parent compiled lower bounds; job.step {receipt['job']}.{receipt['step']}; same method as58108302.15; native workspace omitted equally")


def _finish_receipt(receipt, meta, header, started):
    receipt["seconds"]["total"] = time.monotonic()-started
    receipt["bank_complete"] = bool(header["complete"])
    receipt["capacity"] = meta.shared_pole_capacity.receipt()
    receipt["peak_reason"] = "compiled admission; native measured peak pending ISERV"
    receipt["plan_hash"] = header["bank_plan_digest"]
    return receipt


def compute_moment_bank(wfns, meta, config, *, mesh_xy, sym, bank_io):
    """Stage B: six exact correlations, physical recurrence, scratch write."""
    from file_io.shared_pole_store import write_shared_pole_bank
    header, qids, census = _bank_context(wfns, meta, sym, bank_io, mesh_xy)
    receipt = _receipt("moments", census, bank_io)
    execute = _bank_execution(meta, mesh_xy, bank_io, receipt)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    started = time.monotonic()
    _stream_comparison(wfns, meta, mesh_xy, qids, receipt)
    face_bytes = 16*meta.mu_basis.n_packed**2 // mesh_xy.size
    # Two totals, next correlation, arithmetic temporaries and bounded H/solve.
    name, _ = _reserve(meta, "bank_outputs_moments", (8*len(qids)+16)*face_bytes)
    ledger.live_stages = ambient+(name,)
    _, moments, receipt["algebra"] = response_algebra(meta, config,
        mesh_xy=mesh_xy, n=meta.mu_basis.n_packed)
    if not np.asarray(header["moment_written"]).all():
        a0, a1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh_xy,
                                      q_ids=tuple(qids), execute=execute)
        for iq in range(len(qids)):
            marked = header["moment_written"][iq]
            if all(marked):
                continue
            span = (iq,iq+1)
            h, hi, ranks = _coulomb_batch(meta, config, bank_io, mesh_xy, span, execute)
            del hi
            m1, m3 = execute(moments, (h,a0[iq:iq+1],a1[iq:iq+1]), "moment_dyson")
            io_started = time.monotonic()
            header = write_shared_pole_bank(bank_io["path"], q_span=span,
                M1=None if marked[0] else m1, M3=None if marked[1] else m3,
                meta=meta, expected_identity=bank_io["identity"], mesh_xy=mesh_xy)
            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
            receipt["batches"].append(dict(q_span=span,support_ranks=ranks))
            del h,m1,m3
        del a0,a1
        receipt["correlation_count"] = 6
    ledger.live_stages = ambient
    receipt["completion"] = bool(np.asarray(header["moment_written"]).all())
    return _finish_receipt(receipt,meta,header,started)


def produce_sample_bank(wfns, meta, config, *, mesh_xy, sym, sample_plan, bank_io):
    """Stage A: one windowed stream per admitted sample batch, all parent faces."""
    from file_io.shared_pole_store import write_shared_pole_bank
    header,qids,census = _bank_context(wfns,meta,sym,bank_io,mesh_xy)
    authenticate_sample_plan(sample_plan,header)
    z = bank_points(sample_plan)
    receipt = _receipt("samples",census,bank_io)
    started = time.monotonic()
    execute = _bank_execution(meta,mesh_xy,bank_io,receipt)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    _stream_comparison(wfns,meta,mesh_xy,qids,receipt)
    samples,_,receipt["algebra"] = response_algebra(meta,config,
        mesh_xy=mesh_xy,n=meta.mu_basis.n_packed)
    energy,f,u,reference,_ = response_weights(wfns,meta)
    masks,ft,ut,cells,receipt["windows"] = response_windows(energy,f,u,
        chemical_potential_ry=sample_plan["census"]["mu_ry"])
    adapter = bank_io.get("rule_adapter")
    if adapter is None:
        import minimax
        bank_rule,laplace_rule = minimax.response_bank_rule,minimax.response_laplace_rule
        receipt["rule_provider"] = "minimax"
    else:
        bank_rule,laplace_rule = adapter.response_bank_rule,adapter.response_laplace_rule
        receipt["rule_provider"] = "adapter"
    middle = energy[masks[1]]
    delta = float(middle.max()-middle.min())
    rule = bank_rule(z,delta,rel_tol=sample_plan["bank_rule_tolerance"])
    t,weights = np.asarray(rule["t"]),np.asarray(rule["h"])
    phase = weights[None,:]*np.exp(1j*z[:,None]*t[None,:])
    derivative = phase*(1j*t[None,:]/(2*z[:,None]))
    receipt["rule"] = {k:v for k,v in rule.items() if k not in ("t","h")}
    receipt["nodes"] = len(t)
    remote = []
    for cell in cells:
        rr = laplace_rule(cell["delta_min_ry"],cell["delta_max_ry"],z,
                         rel_tol=sample_plan["bank_rule_tolerance"])
        remote.append((cell,rr))
    receipt["laplace_cells"] = [{**cell,**{k:v for k,v in rr.items()
        if k not in ("t","projection_value","projection_derivative")}}
        for cell,rr in remote]
    face_bytes = 16*meta.mu_basis.n_packed**2//mesh_xy.size
    # At most three carries coexist during remote addition. Leave one quarter
    # of U for dense service/transport work; the common ledger is authoritative.
    width = max(1,min(len(z),int(.75*ledger.U_bytes_per_rank/(2*len(qids)*face_bytes))))
    for lo in range(0,len(z),width):
        hi = min(lo+width,len(z));a = hi-lo
        ledger.live_stages = ambient
        name,_ = _reserve(meta,"bank_outputs",(6*a*len(qids)+32)*face_bytes
            + int(phase.nbytes+derivative.nbytes))
        ledger.live_stages = ambient+(name,)
        kernel,fixed = response_stream(wfns,meta,mesh_xy=mesh_xy,
            q_ids=tuple(qids),n_outputs=2*a)
        raw = execute(kernel,(jnp.asarray(t),jnp.asarray(np.vstack((phase[lo:hi],derivative[lo:hi]))),
            *fixed,stream_weights(wfns,ft*masks[1],mesh_xy),
            stream_weights(wfns,ut*masks[1],mesh_xy),jnp.asarray(reference)),"real_time")
        receipt["correlation_count"] += len(t)
        for cell,rr in remote:
            lower,upper = cell["lower"],cell["upper"]
            refs = np.asarray(cell["references_ry"])
            tau = np.asarray(rr["t"])
            projections = -np.vstack((rr["projection_value"][lo:hi],rr["projection_derivative"][lo:hi]))*np.exp(-(refs[1]-refs[0])*tau)[None,:]
            lk,lfixed = response_stream(wfns,meta,mesh_xy=mesh_xy,
                q_ids=tuple(qids),n_outputs=2*a,pair_mode="laplace")
            lw = np.stack([ft*masks[lower],ut*masks[lower]])
            uw = np.stack([ut*masks[upper],ft*masks[upper]])
            # Parent selection applies to the k axis, separately for each role.
            lw = jnp.stack([stream_weights(wfns,x,mesh_xy) for x in lw])
            uw = jnp.stack([stream_weights(wfns,x,mesh_xy) for x in uw])
            contribution = execute(lk,(jnp.asarray(tau),jnp.asarray(projections),
                *lfixed,lw,uw,jnp.asarray(refs)),"laplace")
            raw = raw+contribution
            raw.block_until_ready();del contribution,lw,uw
            receipt["correlation_count"] += 2*len(tau)
        for iq in range(len(qids)):
            span = (iq,iq+1)
            if np.asarray(header["sample_written"])[iq,lo:hi].all():
                continue
            h,hinv,ranks = _coulomb_batch(meta,config,bank_io,mesh_xy,span,execute)
            del hinv
            for ia in range(lo,hi):
                marked = header["sample_written"][iq][ia]
                if all(marked):continue
                value,ds = execute(samples,(h,raw[iq:iq+1,ia-lo],raw[iq:iq+1,a+ia-lo]),"sample_dyson")
                io_started = time.monotonic()
                header = write_shared_pole_bank(bank_io["path"],q_span=span,sample_span=(ia,ia+1),
                    Wc=None if marked[0] else value[:,None],
                    dWc_ds=None if marked[1] else ds[:,None],meta=meta,
                    expected_identity=bank_io["identity"],mesh_xy=mesh_xy)
                receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                del value,ds
            del h
        receipt["batches"].append(dict(q_span=(0,len(qids)),sample_span=(lo,hi)))
        del raw
    ledger.live_stages = ambient
    receipt["completion"] = bool(np.asarray(header["sample_written"]).all())
    return _finish_receipt(receipt,meta,header,started)
