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
from common import timing
from common.units import RYD_TO_EV
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def response_algebra(meta, config, *, mesh_xy, n, ordered=False):
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
    from .gw_config import dense_layout, linalg_resolution
    from .w_isdf import _w_solve_pref_scalar

    dial = linalg_resolution(
        config if hasattr(config, "get") else {"linalg": config.backend.linalg})
    # Sample stacks fill the mesh: whole matrices per device at or below the measured
    # extent. The moment Dyson is one q at a time and keeps the dial.
    resolution = linalg_resolution({"linalg": dense_layout(dial, "solve", n)})
    moment_resolution = linalg_resolution({"linalg": dense_layout(
        dial, "gemm", n, batch=1, mesh_size=int(mesh_xy.size))})
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

    def mm_m(a, b):
        return matmul(a, b, mesh=mesh_xy,
                      backend="off" if moment_resolution.layout == "local" else "distributed",
                      batched_route=moment_resolution.batched_route)

    def congruence_m(h, a):
        return mm_m(mm_m(h, a), h)

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
        b0 = congruence_m(h, a0)
        b1 = congruence_m(h, a1)
        return (0.5 * congruence_m(h, b0),
                0.5 * congruence_m(h, b1 + mm_m(b0, b0)))

    if ordered:
        @partial(jax.jit, in_shardings=(face, face, face, face, face),
                 out_shardings=(face, face, face, face))
        def moments(h, a0, a1, o0, o1):
            # chi_scaled = o0/z + a0/z^2 + o1/z^3 + a1/z^4 with no time-
            # reversal or reality assumption; X_k = H chi_k H and
            # Wc = H X (I-X)^-1 H. Coefficients C1..C4 of 1/z..1/z^4 keep
            # every X1 cross term (o0 vanishes only in a complete basis).
            # Returns M0=C1/2, M1=C2/2, M2=C3/2, M3=C4/2 (M_k = C_(k+1)/2, the
            # constructor's convention); with o0=o1=0 the even pair reduces
            # to the incumbent M1/M3 exactly.
            x1, x2, x3, x4 = (congruence_m(h, v) for v in (o0, a0, o1, a1))
            x11 = mm_m(x1, x1)
            c2 = x2 + x11
            c3 = x3 + mm_m(x1, x2) + mm_m(x2, x1) + mm_m(x11, x1)
            c4 = (x4 + mm_m(x1, x3) + mm_m(x3, x1) + mm_m(x2, x2) + mm_m(x11, x2)
                  + mm_m(mm_m(x1, x2), x1) + mm_m(x2, x11) + mm_m(x11, x11))
            return (0.5 * congruence_m(h, x1), 0.5 * congruence_m(h, c2),
                    0.5 * congruence_m(h, c3), 0.5 * congruence_m(h, c4))

    algebra = {
        "linalg": resolution.layout,
        "moment_linalg": moment_resolution.layout,
        "solve": lu.describe(),
        "batched_route": route,
        "prefactor": pref,
        "prefactor_q_count": int(meta.nk_tot),
        "moment_convention": "S_m = 2 M_(2m+1) in physical coordinates",
        "units": {"Wc": "Ry", "dWc_ds": "Ry^-1",
                  "M1": "Ry^3", "M3": "Ry^5"},
    }
    if ordered:
        algebra["moment_convention"] += "; odd M0 (1/z) and M2 (1/z^3), M_k = C_(k+1)/2"
        algebra["units"].update(M0="Ry^2", M2="Ry^4")
    return samples, moments, algebra


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
                    pair_mode="retarded", bank_carry=False, node_weights=False):
    """Bind the existing one-particle Green/FFT primitive to a q batch.

    Returns a jitted kernel and its fixed ψ/energy arguments. Caller supplies
    time, projections, final weights and energy reference. The output is
    ``[len(q_ids), n_outputs, mu_p, mu_p]`` with both endpoints sharded.
    """
    from .w_isdf import _get_chi_fractional_contour_kernel_face

    from file_io.shared_pole_store import charge_representation

    if wfns.layout != "face" or not charge_representation(meta):
        raise ValueError("GATE response_representation: got bispinor or legacy "
                         "wavefunctions; want scalar or two-component charge face "
                         "carrier; why: bank requires explicit endpoint shardings")
    carrier = wfns.green_parent
    source = wfns if carrier is None else carrier
    parent = None if carrier is None else carrier.plan
    nk = int(meta.nk_tot) if parent is None else int(parent.n_parent)
    n = int(meta.mu_basis.n_packed)
    kernel = _get_chi_fractional_contour_kernel_face(
        mesh_xy, (meta.nkx, meta.nky, meta.nkz), n_outputs,
        (nk, int(wfns.slices.nb_full), n, int(meta.nspinor)),
        k_unfold_plan=parent, selected_q=tuple(q_ids), pair_mode=pair_mode,
        bank_carry=bank_carry, node_weights=node_weights)
    return kernel, (source.psi_mun, source.psi_nmu, source.enk)


def stream_weights(wfns, weights, mesh_xy):
    """Place small band weights and restrict to existing raw parents."""
    from common.collectives import replicate_to_mesh

    result = replicate_to_mesh(np.asarray(weights), mesh_xy)
    if wfns.green_parent is not None:
        result = wfns.green_parent.plan.parent_rows(result)
    return result


def exact_bare_moments(wfns, meta, *, mesh_xy, q_ids, execute, ordered=False):
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
    terms = (((-1., 1, 0), (1., 0, 1)),
             ((-1., 3, 0), (3., 2, 1), (-3., 1, 2), (1., 0, 3)))
    # (scale, terms, particle phase): imaginary particle weights for the even pair.
    groups = [(1., moment_terms, -1j) for moment_terms in terms]
    if ordered:
        # Odd coefficients of 1/z and 1/z^3: sum (P - conj P_{-q}) Delta^m, m=0,2.
        # Real particle weights keep the retarded difference -i(X - conj X), so
        # the chi coefficient is i*raw.
        groups += [(1j, moment_terms, 1.) for moment_terms in
                   (((1., 0, 0),), ((1., 2, 0), (-2., 1, 1), (1., 0, 2)))]
    rows = [(a, b, phase) for _, moment_terms, phase in groups for _, a, b in moment_terms]
    # All correlations in one stream call: node i carries correlation i's band
    # weights at t=0 and projection row i selects it (other rows add exact zeros).
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy, q_ids=q_ids,
                                    n_outputs=len(rows), node_weights=True)
    weight_f = jnp.stack([stream_weights(wfns, f * erel**a, mesh_xy).astype(jnp.complex128)
                          for a, _, _ in rows])
    weight_u = jnp.stack([stream_weights(wfns, phase * u * erel**b, mesh_xy).astype(jnp.complex128)
                          for _, b, phase in rows])
    args = (jnp.zeros(len(rows)), jnp.eye(len(rows), dtype=jnp.complex128), *fixed,
            weight_f, weight_u, jnp.asarray(reference))
    raw = execute(kernel, args, "moment_correlation")
    del weight_f, weight_u
    totals, index = [], 0
    for scale, moment_terms, _ in groups:
        total = None
        for coefficient, _, _ in moment_terms:
            term = (scale * _w_solve_pref_scalar(meta) * coefficient) * raw[:, index]
            total = term if total is None else total + term
            index += 1
        total.block_until_ready()
        totals.append(total)
    del raw
    return (*totals, census)


@jax.jit
def _odd_moment_ratios(M0, M1, M2, M3):
    # Ratios of the 1/z and 1/z^3 coefficients, m0 = 2 M0 and m2 = 2 M2.
    return jnp.stack([2 * jnp.linalg.norm(M0) / jnp.linalg.norm(M1),
                      2 * jnp.linalg.norm(M2) / jnp.linalg.norm(M3)])


def _record_odd_moments(iq, M0, M1, M2, M3, receipt):
    """Record one parent's band-truncation diagnostic ||m0||/||M1||, ||m2||/||M3||."""
    ratios = np.asarray(_odd_moment_ratios(M0, M1, M2, M3), dtype=np.float64)
    row = dict(q_parent=int(iq), m0_over_M1_fro=float(ratios[0]),
               m2_over_M3_fro=float(ratios[1]))
    receipt.setdefault("odd_moments", []).append(row)
    if jax.process_index() == 0:
        print("TRBANK odd_moments " + " ".join(f"{k}={row[k]}" for k in row), flush=True)


def _bank_context(wfns, meta, sym, bank_io, mesh_xy):
    """Authenticate A/B's existing scratch transaction and physical state."""
    from file_io.shared_pole_store import (charge_representation,
                                           validate_shared_pole_bank)

    # The measured time-reversal verdict selects the orientation (callers
    # read sym.trs_allowed); only the operator representation refuses here.
    if not charge_representation(meta):
        raise ValueError("GATE response_representation: want scalar or "
                         "two-component charge operator; bispinor bank is unsupported")
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


@lru_cache(maxsize=32)
def response_dense_workspace(mesh_xy, n, batch, layout, *, with_eigh):
    """Query the dense provider on actual bank shapes, before allocation."""
    from distrib_la import plan, workspace_bytes_per_rank
    from .gw_config import linalg_resolution
    resolution = linalg_resolution({"linalg": layout})
    policy = plan("eigh", mesh_xy, n=n,
        backend="off" if layout == "local" else "distributed",
        batched_route=resolution.batched_route)
    gemm = workspace_bytes_per_rank(policy,"gemm",((batch,n,n),(batch,n,n)),np.complex128)
    eig = workspace_bytes_per_rank(policy,"eigh",((1,n,n),),np.complex128) if with_eigh else 0
    return dict(gemm=gemm,eigh=eig,total=gemm+eig,scope="actual-shape ISERV query; GEMM persistent plus concurrent eigh scratch")


def _bank_execution(meta, mesh_xy, bank_io, receipt, config):
    """Compile and admit new dense work; stream outputs are reserved by batch."""
    def execute(kernel, args, stage):
        timing.fence('bank.compile.' + stage, sync_ranks=True)
        with timing.section('bank.compile.' + stage):
            started = time.monotonic()
            executable = kernel.lower(*args).compile()
            receipt["seconds"]["compilation"] = (receipt["seconds"].get("compilation", 0.)
                + time.monotonic() - started)
        timing.fence('bank.admission.' + stage, sync_ranks=True)
        with timing.section('bank.admission.' + stage):
            memory = executable.memory_analysis()
            if memory is None:
                raise ValueError("GATE response_capacity: compiled memory unavailable")
            stream = stage in ("real_time", "laplace", "moment_correlation")
            if not stream:
                layout = config.get("linalg", "local") if hasattr(config,"get") else config.backend.linalg
                if stage in ("sample_dyson", "moment_dyson", "coulomb_sqrt"):
                    from .gw_config import dense_layout, linalg_resolution
                    layout = dense_layout(linalg_resolution({"linalg": layout}),
                        {"coulomb_sqrt": "eigh", "sample_dyson": "solve", "moment_dyson": "gemm"}[stage],
                        args[0].shape[-1], **({"batch": args[0].shape[0], "mesh_size": mesh_xy.size}
                                             if stage == "moment_dyson" else {}))
                native = response_dense_workspace(mesh_xy,args[0].shape[-1],args[0].shape[0],layout,
                    with_eigh=stage=="coulomb_sqrt")
                receipt.setdefault("native_queries",[]).append(dict(stage=stage,**native))
                _, row = _reserve(meta, stage, memory.argument_size_in_bytes,
                    memory.output_size_in_bytes + memory.temp_size_in_bytes + native["total"])
                receipt["memory"].append(row)
            receipt["compiled"].append(dict(stage=stage,
                arguments=memory.argument_size_in_bytes, outputs=memory.output_size_in_bytes,
                temporaries=memory.temp_size_in_bytes, inherited_stream=stream))
        timing.fence('bank.execute.' + stage, sync_ranks=True)
        with timing.section('bank.execute.' + stage):
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
    from .gw_config import dense_layout, linalg_resolution
    resolution = linalg_resolution({"linalg": dense_layout(
        linalg_resolution({"linalg": layout}), "eigh", n_packed)})
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


@lru_cache(maxsize=8)
def _coulomb_pack(basis,mesh_xy):
    return jax.jit(lambda v: basis.pack_operator(v,spec=P(None,"x","y")))


@lru_cache(maxsize=8)
def _face_row(mesh_xy):
    """One leading row [1,n,n] of a [b,n,n] panel, kept on the x/y face."""
    return jax.jit(lambda a, i: jax.lax.dynamic_slice_in_dim(a, i, 1, axis=0),
                   out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))


def _coulomb_batch(meta, config, bank_io, mesh_xy, q_span, execute):
    """Read one authenticated canonical V batch; convert through its owner."""
    from file_io.slab_io import SlabIO
    basis = meta.mu_basis
    shape = (q_span[1]-q_span[0], basis.n_canonical, basis.n_canonical)
    spec = P(None, "x", "y")
    abstract = jax.ShapeDtypeStruct(shape, jnp.complex128,
                                   sharding=NamedSharding(mesh_xy, spec))
    packed = _coulomb_pack(basis,mesh_xy)
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
        native_accumulator_workspace_bytes=0,
        native_workspace_status="NOT_MEASURED",
        native_workspace_reason="other external native allocations excluded from compiler/JAX allocator counts",
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
    stats = [d.memory_stats() for d in jax.local_devices()]
    local_peak = max((v.get("peak_bytes_in_use",0) for v in stats if v),default=0)
    from jax.experimental import multihost_utils
    peak = int(np.max(multihost_utils.process_allgather(np.asarray(local_peak,dtype=np.int64))))
    receipt["peak_bytes"] = peak or None
    receipt["peak_reason"] = "maximum JAX allocator high-water bytes across ranks; inherited arrays included, external native allocations excluded"
    receipt["plan_hash"] = header["bank_plan_digest"]
    return receipt


def compute_moment_bank(wfns, meta, config, *, mesh_xy, sym, bank_io):
    """Stage B: six exact correlations, physical recurrence, scratch write."""
    from file_io.shared_pole_store import write_shared_pole_bank
    header, qids, census = _bank_context(wfns, meta, sym, bank_io, mesh_xy)
    receipt = _receipt("moments", census, bank_io)
    if not bool(sym.trs_allowed):
        # M1/M3 are the 1/s and 1/s^2 coefficients; the odd channel starts at
        # 1/z^3, so the same six correlations stay exact on an ordered bank.
        receipt["ordered"] = True
    execute = _bank_execution(meta, mesh_xy, bank_io, receipt, config)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    started = time.monotonic()
    _stream_comparison(wfns, meta, mesh_xy, qids, receipt)
    face_bytes = 16*meta.mu_basis.n_packed**2 // mesh_xy.size
    # Two totals, next correlation, arithmetic temporaries and bounded H/solve.
    ordered = not bool(sym.trs_allowed)
    _, moments, receipt["algebra"] = response_algebra(meta, config,
        mesh_xy=mesh_xy, n=meta.mu_basis.n_packed, **({"ordered": True} if ordered else {}))
    # Per q: every correlation of the one stream call, the totals, arithmetic
    # temporaries and one resident Coulomb root.
    per_q = 18 if ordered else 12
    qwidth = max(1,min(len(qids),int((.75*ledger.U_bytes_per_rank/face_bytes-16)/per_q)))
    for q0 in range(0,len(qids),qwidth):
        q1 = min(q0+qwidth,len(qids))
        ledger.live_stages = ambient
        name,_ = _reserve(meta,"bank_outputs_moments",(per_q*(q1-q0)+16)*face_bytes)
        ledger.live_stages = ambient+(name,)
        if not np.asarray(header["moment_written"])[q0:q1].all():
            if ordered:
                a0, a1, o0, o1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh_xy,
                    q_ids=tuple(qids[q0:q1]), execute=execute, ordered=True)
            else:
                a0, a1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh_xy,
                                          q_ids=tuple(qids[q0:q1]), execute=execute)
            # One Coulomb read and root for the q batch, not one per q.
            roots, _, panel_ranks = _coulomb_batch(meta, config, bank_io, mesh_xy, (q0, q1), execute)
            for iq in range(q0,q1):
                marked = header["moment_written"][iq]
                if all(marked):
                    continue
                span = (iq,iq+1)
                h = _face_row(mesh_xy)(roots, np.int32(iq-q0))
                ranks = panel_ranks[iq-q0:iq-q0+1]
                odd = {}
                if ordered:
                    part = slice(iq-q0, iq-q0+1)
                    M0, m1, M2, m3 = execute(moments, (h,a0[part],a1[part],o0[part],o1[part]), "moment_dyson")
                    _record_odd_moments(iq, M0, m1, M2, m3, receipt)
                    # The ordered bank's moment masks are (M1, M3, M0, M2).
                    odd = dict(M0=None if marked[2] else M0, M2=None if marked[3] else M2)
                    del M0, M2
                else:
                    m1, m3 = execute(moments, (h,a0[iq-q0:iq-q0+1],a1[iq-q0:iq-q0+1]), "moment_dyson")
                io_started = time.monotonic()
                header = write_shared_pole_bank(bank_io["path"], q_span=span,
                    M1=None if marked[0] else m1, M3=None if marked[1] else m3, **odd,
                    meta=meta, expected_identity=bank_io["identity"], mesh_xy=mesh_xy)
                receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                receipt["batches"].append(dict(q_span=span,support_ranks=ranks))
                del h,m1,m3,odd
            del a0,a1,roots
            receipt["correlation_count"] += 10 if ordered else 6
    ledger.live_stages = ambient
    receipt["completion"] = bool(np.asarray(header["moment_written"]).all())
    return _finish_receipt(receipt,meta,header,started)


def _self_negative(q_full, meta):
    """True where q = -q on the canonical C-ordered full grid (a TRIM parent)."""
    grid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    return all((2*int(i)) % n == 0 for i, n in zip(np.unravel_index(q_full, grid), grid))


@jax.jit
def _census_scalars(chi, w, w_even):
    def mx(a):
        return jnp.max(jnp.abs(a))
    sym = 0.5*(chi + chi.T)
    odd = 0.5*(chi - chi.T)
    return jnp.stack([mx(odd)/mx(sym),
                      jnp.linalg.norm(odd)/jnp.linalg.norm(sym),
                      mx(chi - chi.conj().T)/mx(chi),
                      mx(w - w.conj().T)/mx(w),
                      mx(w_even - w_even.conj().T)/mx(w_even),
                      mx(w - w.T)/mx(w)])


def _tr_odd_census(receipt, samples, h, chi, dchi, value, z, q_full):
    """Measure the time-reversal-odd channel of an ordered bank at q = -q.

    At a self-negative q an imaginary-axis chi0 is real, and its transpose-
    antisymmetric part is the odd channel. Records that part against the
    symmetric part (max and Frobenius), the Hermiticity of the ordered
    W = V + Wc, of the even-route W from the symmetric part through the same
    Coulomb root and Dyson algebra, and max|W - W^T|/max|W|. Scalars only;
    every rank computes, rank 0 prints one line per sample.
    """
    imaginary = np.flatnonzero(np.real(np.asarray(z)) == 0.0)
    if not imaginary.size:
        return
    v = (h @ h)[0]
    names = ("chi_odd_max_rel", "chi_odd_fro_rel", "chi_hermiticity_rel",
             "w_hermiticity_rel", "w_even_route_hermiticity_rel", "w_transpose_rel")
    # The Dyson owner requires face-sharded [1,n,n] operands.
    symmetric = jax.jit(lambda c, dc: (0.5*(c + jnp.swapaxes(c, -1, -2)),
                                       0.5*(dc + jnp.swapaxes(dc, -1, -2))),
                        out_shardings=(h.sharding, h.sharding))
    for s in imaginary.tolist():
        sym, dsym = symmetric(chi[s:s+1], dchi[s:s+1])
        w_even = v + samples(h, sym, dsym)[0][0]
        values = np.asarray(_census_scalars(chi[s], v + value[s], w_even), dtype=np.float64)
        row = dict(q_full=q_full, z_ry=[float(z[s].real), float(z[s].imag)],
                   **{k: float(x) for k, x in zip(names, values)})
        receipt.setdefault("tr_odd_census", []).append(row)
        if jax.process_index() == 0:
            print("TRBANK tr_odd_census " + " ".join(f"{k}={row[k]}" for k in row), flush=True)


def _ordered_cost_probe(wfns, meta, mesh_xy, qid, tau, projections, lw, uw, refs, n_out):
    """Warm seconds per remote Laplace node, even versus ordered kernel.

    Same deck, geometry, q parent, band weights and first nodes; each kernel
    runs once to compile and once timed. The difference is the ordered
    per-node work (one more chi FFT, q gather and accumulation); node counts
    come from the rule certificate. Receipt-only; nothing is written.
    """
    nodes = min(4, len(tau))
    seconds = {}
    for mode, rows in (("laplace", projections[:n_out]), ("laplace_ordered", projections)):
        kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy, q_ids=(qid,),
                                        n_outputs=n_out, pair_mode=mode)
        args = (jnp.asarray(tau[:nodes]), jnp.asarray(rows[:, :nodes]), *fixed,
                lw, uw, jnp.asarray(refs))
        jax.block_until_ready(kernel(*args))
        timing.fence('bank.ordered_cost_probe', sync_ranks=True)
        started = time.monotonic()
        jax.block_until_ready(kernel(*args))
        seconds[mode] = (time.monotonic() - started)/nodes
    return dict(q_full=qid, nodes=nodes, seconds_per_node_even=seconds["laplace"],
                seconds_per_node_ordered=seconds["laplace_ordered"],
                ratio=seconds["laplace_ordered"]/seconds["laplace"],
                scope="warm rank-0 wall seconds per remote Laplace node, one q parent, after a synchronized fence")


def produce_sample_bank(wfns, meta, config, *, mesh_xy, sym, sample_plan, bank_io):
    """Stage A: one windowed stream per admitted sample batch, all parent faces."""
    timing.fence('bank.setup', sync_ranks=True)
    with timing.section('bank.setup'):
        from file_io.shared_pole_store import write_shared_pole_bank
        header,qids,census = _bank_context(wfns,meta,sym,bank_io,mesh_xy)
        authenticate_sample_plan(sample_plan,header)
        z = bank_points(sample_plan)
        receipt = _receipt("samples",census,bank_io)
        # Time reversal measured broken: both particle-hole orientations keep
        # independent weights. The retarded stream already forms the partner as
        # conj in R space (the -q orientation); remote cells add the odd kernel.
        ordered = not bool(sym.trs_allowed)
        if ordered:
            receipt["ordered"] = True
        # Odd-channel census and ordered per-node cost probe are diagnostics: they run
        # only under the debug output switch (owner ruling: no debug work by default).
        debug = bool(getattr(getattr(config, "debug", None), "sigma_freq_debug_output", False))
        started = time.monotonic()
        execute = _bank_execution(meta,mesh_xy,bank_io,receipt,config)
        ledger = meta.shared_pole_capacity
        ambient = ledger.live_stages
    timing.fence('bank.stream_reference_compile', sync_ranks=True)
    with timing.section('bank.stream_reference_compile'):
        _stream_comparison(wfns,meta,mesh_xy,qids,receipt)
    timing.fence('bank.window_geometry', sync_ranks=True)
    with timing.section('bank.window_geometry'):
        samples,_,receipt["algebra"] = response_algebra(meta,config,
            mesh_xy=mesh_xy,n=meta.mu_basis.n_packed)
        energy,f,u,reference,_ = response_weights(wfns,meta)
        masks,ft,ut,cells,receipt["windows"] = response_windows(energy,f,u,
            chemical_potential_ry=sample_plan["census"]["mu_ry"])
    timing.fence('bank.quadrature', sync_ranks=True)
    with timing.section('bank.quadrature'):
        import minimax
        bank_rule,laplace_rule = minimax.response_bank_rule,minimax.response_laplace_rule
        receipt["rule_provider"] = "minimax"
        middle = energy[masks[1]]
        delta = float(middle.max()-middle.min())
        session = getattr(meta, "shared_pole_response_rules", None)
        # Each one-particle endpoint gets 2 eV: a transition edge gets 4 eV.
        pad = 4.0/RYD_TO_EV if session is not None else 0.0
        rule = bank_rule(z,delta,rel_tol=sample_plan["bank_rule_tolerance"],
            previous=None if session is None else session.get("stream"),
            domain_pad_ry=pad)
        if session is not None:
            session["stream"] = rule
        t,weights = np.asarray(rule["t"]),np.asarray(rule["h"])
        phase = np.asarray(rule["projection_value"])
        derivative = np.asarray(rule["projection_derivative"])
        receipt["rule"] = {k:v for k,v in rule.items() if k not in ("t","h","projection_value","projection_derivative")}
        receipt["nodes"] = len(t)
        remote = []
        for cell in cells:
            key = (cell["lower"], cell["upper"])
            rr = laplace_rule(cell["delta_min_ry"],cell["delta_max_ry"],z,
                rel_tol=sample_plan["bank_rule_tolerance"],
                previous=None if session is None else session.get(key),
                domain_pad_ry=pad,**({"ordered": True} if ordered else {}))
            if session is not None:
                session[key] = rr
            remote.append((cell,rr))
        receipt["laplace_cells"] = [{**cell,**{k:v for k,v in rr.items()
            if k not in ("t","projection_value","projection_derivative","coefficient_rows",
                         "odd_projection_value","odd_projection_derivative")}}
            for cell,rr in remote]
    timing.fence('bank.capacity_planning_compile', sync_ranks=True)
    with timing.section('bank.capacity_planning_compile'):
        face_bytes = 16*meta.mu_basis.n_packed**2//mesh_xy.size
        # One donated internal [output,q,x,y] carry spans every window. Public
        # writer slices are [q,output,x,y]; only that bounded slice is transposed.
        # Reserve output plus a dense/transport headroom, and batch only when the
        # common ledger's admitted panel budget cannot hold the full point plan.
        layout = config.get("linalg","local") if hasattr(config,"get") else config.backend.linalg
        from .gw_config import dense_layout, linalg_resolution
        # Dyson GEMMs follow the routed layout; the Coulomb eigh keeps its own.
        dyson_layout = dense_layout(linalg_resolution({"linalg": layout}), "solve", meta.mu_basis.n_packed)
        native = dict(response_dense_workspace(mesh_xy,meta.mu_basis.n_packed,len(z),dyson_layout,with_eigh=False))
        eigh_layout = dense_layout(linalg_resolution({"linalg": layout}), "eigh", meta.mu_basis.n_packed)
        native["eigh"] = response_dense_workspace(mesh_xy,meta.mu_basis.n_packed,len(z),eigh_layout,with_eigh=True)["eigh"]
        native["total"] = native["gemm"] + native["eigh"]
        headroom = 16*face_bytes + native["gemm"] + int(phase.nbytes+derivative.nbytes)
        # Ask the ledger owner for R24's remaining device budget. The zero-byte
        # planning row includes ambient live reservations but allocates nothing.
        _, budget = _reserve(meta,"bank_planning",0)
        live_bytes = budget["aggregate_bytes_per_rank"]
        device_available = budget["available_device_bytes_per_rank"]
        scaling_target = budget["limit_bytes_per_rank"]

        @lru_cache(maxsize=None)
        def dense_bytes(width):
            abstract = jax.ShapeDtypeStruct((width,meta.mu_basis.n_packed,meta.mu_basis.n_packed),
                jnp.complex128,sharding=NamedSharding(mesh_xy,P(None,"x","y")))
            stats = samples.lower(abstract,abstract,abstract).compile().memory_analysis()
            if stats is None:
                raise ValueError("GATE response_capacity: sample solve planning memory unavailable")
            dense = stats.argument_size_in_bytes+stats.output_size_in_bytes+stats.temp_size_in_bytes
            # Preserve the existing dense/native and writer conversion envelopes.
            return max(dense+native["total"],4*width*face_bytes+native["total"])

        minimum = headroom+live_bytes+2*face_bytes+dense_bytes(1)
        # R24 makes 3U a reported scaling target, not the device admission limit.
        # Replaying a Green/FFT stream to meet that preference repeats every time
        # node even when the complete output panel fits. Use the ledger's remaining
        # device budget, including the inherited stream and ambient/native costs;
        # larger systems still split q/sample panels before any allocation.
        planning_limit = device_available
        available = planning_limit-headroom-live_bytes
        # Per q: two sample outputs at width one and one resident Coulomb root.
        qwidth = min(len(qids),int((available-dense_bytes(1))//(3*face_bytes)))
        receipt["panel_budget"] = dict(
            scaling_target_bytes_per_rank=scaling_target,
            device_budget_bytes_per_rank=budget["device_budget_bytes_per_rank"],
            inherited_peak_bytes_per_rank=budget["inherited_peak_bytes_per_rank"],
            available_device_bytes_per_rank=device_available,
            ambient_live_bytes_per_rank=live_bytes,headroom_bytes_per_rank=headroom,
            native_workspace=native,planning_limit_bytes_per_rank=planning_limit,
            minimum_panel_bytes_per_rank=minimum,
            policy="minimize stream replays within remaining device budget; report 3U scaling target (ruling24)")
        if qwidth < 1:
            raise ValueError(f"GATE response_capacity: one q/sample panel needs {minimum} B/rank "
                             f"including live/native costs; remaining device budget is {device_available} B/rank "
                             f"(3U scaling target {scaling_target} B/rank)")
    for q0 in range(0,len(qids),qwidth):
        timing.fence('bank.panel_admission', sync_ranks=True)
        with timing.section('bank.panel_admission'):
            q1 = min(q0+qwidth,len(qids))
            width = len(z)
            while (2*width+1)*(q1-q0)*face_bytes+dense_bytes(width) > available:
                width -= 1
            planned_bytes = headroom+live_bytes+(2*width+1)*(q1-q0)*face_bytes+dense_bytes(width)
            receipt.setdefault("panel_plans",[]).append(dict(q_span=(q0,q1),sample_width=width,
                aggregate_bytes_per_rank=planned_bytes,
                scaling_status="PASS" if planned_bytes <= scaling_target else "WARN",
                device_budget_status="PASS",scaling_target_bytes_per_rank=scaling_target,
                available_device_bytes_per_rank=device_available))
        for lo in range(0,len(z),width):
            timing.fence('bank.stream_arguments', sync_ranks=True)
            with timing.section('bank.stream_arguments'):
                hi = min(lo+width,len(z));a = hi-lo
                ledger.live_stages = ambient
                name,_ = _reserve(meta,"bank_outputs",(2*a+1)*(q1-q0)*face_bytes + headroom)
                ledger.live_stages = ambient+(name,)
                kernel,fixed = response_stream(wfns,meta,mesh_xy=mesh_xy,
                    q_ids=tuple(qids[q0:q1]),n_outputs=2*a,bank_carry=True)
                raw = jax.jit(lambda: jnp.zeros((2*a,q1-q0,meta.mu_basis.n_packed,meta.mu_basis.n_packed),jnp.complex128),
                    out_shardings=NamedSharding(mesh_xy,P(None,None,"x","y")))()
            raw = execute(kernel,(jnp.asarray(t),jnp.asarray(np.vstack((phase[lo:hi],derivative[lo:hi]))),
                *fixed,stream_weights(wfns,ft*masks[1],mesh_xy),
                stream_weights(wfns,ut*masks[1],mesh_xy),jnp.asarray(reference),raw),"real_time")
            receipt["correlation_count"] += len(t)
            # Cell data are dynamic arguments; reuse one compiled owner for
            # equal-shaped Laplace cells instead of retracing each closure.
            if remote:
                lk,lfixed = response_stream(wfns,meta,mesh_xy=mesh_xy,
                    q_ids=tuple(qids[q0:q1]),n_outputs=2*a,
                    pair_mode="laplace_ordered" if ordered else "laplace",bank_carry=True)
            for cell,rr in remote:
                timing.fence('bank.laplace_arguments', sync_ranks=True)
                with timing.section('bank.laplace_arguments'):
                    lower,upper = cell["lower"],cell["upper"]
                    refs = np.asarray(cell["references_ry"])
                    tau = np.asarray(rr["t"])
                    projections = -np.vstack((rr["projection_value"][lo:hi],rr["projection_derivative"][lo:hi]))*np.exp(-(refs[1]-refs[0])*tau)[None,:]
                    if ordered:
                        # Rows [even value, even ds, odd value, odd ds].
                        projections = np.vstack((projections,
                            -np.vstack((rr["odd_projection_value"][lo:hi],rr["odd_projection_derivative"][lo:hi]))*np.exp(-(refs[1]-refs[0])*tau)[None,:]))
                    lw = np.stack([ft*masks[lower],ut*masks[lower]])
                    uw = np.stack([ut*masks[upper],ft*masks[upper]])
                    # Parent selection applies to the k axis, separately for each role.
                    lw = jnp.stack([stream_weights(wfns,x,mesh_xy) for x in lw])
                    uw = jnp.stack([stream_weights(wfns,x,mesh_xy) for x in uw])
                if ordered and debug and "ordered_cost_probe" not in receipt:
                    receipt["ordered_cost_probe"] = _ordered_cost_probe(
                        wfns,meta,mesh_xy,int(qids[q0]),tau,projections,lw,uw,refs,2*a)
                raw = execute(lk,(jnp.asarray(tau),jnp.asarray(projections),
                    *lfixed,lw,uw,jnp.asarray(refs),raw),"laplace")
                del lw,uw
                receipt["correlation_count"] += 2*len(tau)
            pending = [iq for iq in range(q0,q1)
                       if not np.asarray(header["sample_written"])[iq,lo:hi].all()]
            if pending:
                # One Coulomb read and root for the q panel, not one per q.
                roots,_,_ = _coulomb_batch(meta,config,bank_io,mesh_xy,(q0,q1),execute)
            for iq in pending:
                span = (iq,iq+1)
                h = _face_row(mesh_xy)(roots,np.int32(iq-q0))
                ia = lo
                while ia < hi:
                    marked = tuple(header["sample_written"][iq][ia])
                    stop = ia+1
                    while stop < hi and tuple(header["sample_written"][iq][stop]) == marked:
                        stop += 1
                    if all(marked):
                        ia = stop
                        continue
                    chi = raw[ia-lo:stop-lo,iq-q0]
                    dchi = raw[a+ia-lo:a+stop-lo,iq-q0]
                    hbatch = jnp.broadcast_to(h,chi.shape)
                    value,ds = execute(samples,(hbatch,chi,dchi),"sample_dyson")
                    if ordered and debug and _self_negative(int(qids[iq]),meta):
                        _tr_odd_census(receipt,samples,h,chi,dchi,value,z[ia:stop],int(qids[iq]))
                    value = None if marked[0] else value[None]
                    ds = None if marked[1] else ds[None]
                    timing.fence('bank.write', sync_ranks=True)
                    with timing.section('bank.write'):
                        io_started = time.monotonic()
                        header = write_shared_pole_bank(bank_io["path"],q_span=span,
                            sample_span=(ia,stop),Wc=value,dWc_ds=ds,meta=meta,
                            expected_identity=bank_io["identity"],mesh_xy=mesh_xy)
                        receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                    del value,ds,chi,dchi,hbatch
                    ia = stop
                del h
            if pending:
                del roots
            receipt["batches"].append(dict(q_span=(q0,q1),sample_span=(lo,hi)))
            del raw
    timing.fence('bank.finalize', sync_ranks=True)
    with timing.section('bank.finalize'):
        ledger.live_stages = ambient
        receipt["stream_passes"] = len(receipt["batches"])
        receipt["batch_reason"] = "full plan admitted" if len(receipt["batches"]) == 1 else "remaining device-budget panels require bounded replays; see panel_budget and panel_plans"
        receipt["completion"] = bool(np.asarray(header["sample_written"]).all())
        return _finish_receipt(receipt,meta,header,started)
