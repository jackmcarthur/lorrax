"""Physical response-bank algebra (DESIGN §3.1, DBANK D5–D6).

Charge operators have packed centroid-major endpoints ``mu * nspinor + spin``.
Photon operators use ``PhotonBasisLayout`` for charge and current endpoints.
Disk conversion belongs to the scratch writer. Dense products and solves
enter through ``distrib_la``.
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


def response_algebra(meta, config, *, mesh_xy, n, ordered=False, photon=False):
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
            x1, x2, x3, x4 = (congruence(h, v) for v in (o0, a0, o1, a1))
            x11 = mm(x1, x1)
            c2 = x2 + x11
            c3 = x3 + mm(x1, x2) + mm(x2, x1) + mm(x11, x1)
            c4 = (x4 + mm(x1, x3) + mm(x3, x1) + mm(x2, x2) + mm(x11, x2)
                  + mm(mm(x1, x2), x1) + mm(x2, x11) + mm(x11, x11))
            return (0.5 * congruence(h, x1), 0.5 * congruence(h, c2),
                    0.5 * congruence(h, c3), 0.5 * congruence(h, c4))

    algebra = {
        "linalg": resolution.layout,
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
    if photon:
        volume = float(meta.cell_volume)
        @partial(jax.jit, in_shardings=(face, face), out_shardings=face)
        def infinity(v, contact):
            # chi(z)=chi_param(z)-contact, so W_inf=(I+V contact)^-1 V.
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), v.shape)
            return lu.batched(identity + mm(v, volume * contact), v.copy())

        @partial(jax.jit, in_shardings=(face, face, face, face),
                 out_shardings=(face, face))
        def samples(v, chi_raw, dchi_raw, contact):
            # Signed photon Dyson: W=(I-V chi)^-1 V. No square root of V_TT.
            # dW/ds=W (dchi/ds) W, with no adjoint at complex frequency.
            chi = pref * chi_raw - volume * contact
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), chi.shape)
            w = lu.batched(identity - mm(v, chi), v.copy())
            return w - v, mm(mm(w, pref * dchi_raw), w)

        @partial(jax.jit, in_shardings=(face,) * 6,
                 out_shardings=(face,) * 5)
        def moments(v, a0, a1, o0, o1, contact):
            # W=W_inf + sum C_k/z^k. Dyson recurrence
            # C_k=W_inf [chi_k W_inf + sum_{i=1}^{k-1} chi_i C_(k-i)].
            # Bare coefficients are already physically scaled.
            winf = infinity(v, contact)
            coefficients = (o0, a0, o1, a1)
            result = []
            for k, coefficient in enumerate(coefficients):
                rhs = mm(coefficient, winf)
                for i in range(k):
                    rhs = rhs + mm(coefficients[i], result[k-i-1])
                result.append(mm(winf, rhs))
            return (winf - v, *(0.5 * c for c in result))

        algebra["representation"] = "signed packed photon"
        algebra["constant"] = "W_infinity-V, retained separately from M0..M3"
        algebra["moment_convention"] = "M_k=C_(k+1)/2 about W_infinity"
        algebra["units"].update(M0="Ry^2", M2="Ry^4", constant="Ry")
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


def prepare_photon_carriers(wfns, wfns_transverse, mu_bases, *,
                            mesh_xy, layout):
    """Prepare bare/J-applied photon endpoints for the one response stream.

    Each family is unfolded by its authenticated parent plan before applying
    ``J=(I,alpha_x,alpha_y,alpha_z)``. Canonical centroid conversion precedes
    the photon pack. Returned arguments are ``((bare_mun,J_mun),
    (bare_nmu,J_nmu), energy)``; endpoints have four spin components and
    ``layout.packed_extent`` centroids, with both face axes distributed.
    Only linear-size wavefunction carriers are materialized here.
    """
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    from common.shard_map import shard_map
    from common.wfn_layout import psi_specs
    from .photon_layout import pack_photon_faces
    from .w_isdf import _require_current_chi_endpoints

    left, right = _require_current_chi_endpoints(wfns, wfns_transverse)
    for name in ("irr_idx", "sym_idx", "k_parent_frac", "spin_action_full"):
        if not np.array_equal(getattr(left.plan, name), getattr(right.plan, name)):
            raise ValueError("GATE response_vertex: endpoint parent actions disagree")
    if not np.array_equal(np.asarray(wfns.enk), np.asarray(wfns_transverse.enk)):
        raise ValueError("GATE response_vertex: endpoint energies disagree")
    if not np.array_equal(np.asarray(wfns.occ), np.asarray(wfns_transverse.occ)):
        raise ValueError("GATE response_vertex: endpoint occupations disagree")
    nmu_spec, mun_spec = psi_specs(wfns.layout)
    families = []
    for carrier, basis in zip((left, right), mu_bases):
        plan = carrier.plan
        @partial(shard_map, mesh=mesh_xy, in_specs=(mun_spec, nmu_spec),
                 out_specs=(mun_spec, nmu_spec), check_vma=False)
        def unfold(mun, nmu):
            return (plan.unfold_face(mun, spin_axis=1, mu_axis=2, mesh_axis="x"),
                    plan.unfold_face(nmu, spin_axis=2, mu_axis=3, mesh_axis="y"))
        mun, nmu = jax.jit(unfold)(carrier.psi_mun, carrier.psi_nmu)
        families.append((basis.unpack_axis(mun, 2, spec=mun_spec),
                         basis.unpack_axis(nmu, 3, spec=nmu_spec)))
    endpoints = []
    for index, orientation, spin_axis in ((0, "mun", 1), (1, "nmu", 2)):
        bare = (families[0][index],) + (families[1][index],) * 3
        current = tuple(gamma_apply(face, *gamma_perm_phase(A), axis=spin_axis,
                                    is_identity=A == 0)
                        for A, face in enumerate(bare))
        endpoints.append(tuple(pack_photon_faces(faces, layout, mesh_xy,
            orientation=orientation, wfn_layout=wfns.layout) for faces in (bare, current)))
    return (*endpoints, wfns.enk)


def photon_static_contact(wfns, meta, *, mesh_xy, layout, vertex,
                          occupation_state, sample_plan, execute, receipt):
    r"""Build Pi_grid(0,0), centroid D and their TT sum once per bank.

    The FD zero-Matsubara stream contains ``-D`` on diagonal transitions.
    Therefore ``Pi_grid=Pi_FD+D`` removes that contribution; the prescribed
    contact is then ``Pi_grid+D``. Both quantities here are physical density
    responses (one factor ``1/Omega``). ``response_algebra`` converts the
    contact to the convention of the stored ``V/ Omega`` at insertion.
    For a step-occupation insulator the same stream uses its ordinary
    Laplace weights and D is exactly zero. It never assigns a gap to a metal.
    All returned packed operators are ``[1,N,N]`` at ``P(None,'x','y')``.
    """
    from common.shard_map import shard_map
    from .static_gauge_response import (fermi_dirac_current_drude,
                                         photon_diagonal_current_faces)
    from .w_isdf import _w_solve_pref_scalar, matsubara_rule

    energy, f, u, _, census = response_weights(wfns, meta)
    live = (np.arange(energy.shape[1])[None, :]
            < census["band_stop"]-census["band_start"])
    live = np.broadcast_to(live, energy.shape)
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    if occupation_state is not None:
        if not np.array_equal(np.asarray(occupation_state.f_kn), np.asarray(wfns.occ)):
            raise ValueError("GATE photon_contact_state: bank and FD occupations differ")
        beta, mu, _, rule = matsubara_rule(wfns, occupation_state, (0,),
            rel_tol=sample_plan["bank_rule_tolerance"])
        kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
            q_ids=(0,), n_outputs=1, pair_mode="kms_static", vertex=vertex)
        raw = execute(kernel, (jnp.asarray(rule["t"]), jnp.asarray(rule["weights"]),
            *fixed, jnp.asarray(live), jnp.asarray(live), jnp.asarray([beta, mu])), "static_reference")
        currents = photon_diagonal_current_faces(vertex, mesh_xy=mesh_xy,
            layout=layout, wfn_layout=wfns.layout)
        currents = (currents[0] * jnp.asarray(live)[:, None, :],
                    currents[1] * jnp.asarray(live)[:, :, None])
        census_state = sample_plan["census"]
        drude = fermi_dirac_current_drude(currents, occupation_state,
            state_capacity=census_state["state_capacity"], cell_volume=meta.cell_volume,
            kweights=census_state["k_weights"], mesh_xy=mesh_xy)[None]
        receipt["static_rule"] = rule["certificate"]
        count = len(rule["t"])
    else:
        if np.any((f != 0) & (f != 1)):
            raise ValueError("GATE photon_contact_state: fractional bank needs its FD state")
        from .minimax_screening import solve_laplace_minimax_interval
        lo, hi = float(energy[f > 0].max()), float(energy[u > 0].min())
        gap = hi-lo
        if gap <= 0:
            raise ValueError("GATE photon_contact_state: gapless state needs FD occupations")
        quad = solve_laplace_minimax_interval(gap,
            float(energy[u > 0].max()-energy[f > 0].min()),
            target_error=sample_plan["bank_rule_tolerance"])
        kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
            q_ids=(0,), n_outputs=1, pair_mode="laplace_ordered", vertex=vertex)
        projections = np.stack((-quad.alpha*np.exp(-gap*quad.tau), np.zeros_like(quad.tau)))
        raw = execute(kernel, (jnp.asarray(quad.tau), jnp.asarray(projections, dtype=jnp.complex128), *fixed,
            jnp.asarray(np.stack((f, np.zeros_like(f)))),
            jnp.asarray(np.stack((u, np.zeros_like(u)))), jnp.asarray([lo, hi])), "static_reference")
        drude = jax.jit(lambda: jnp.zeros((1, layout.packed_extent, layout.packed_extent), complex),
                        out_shardings=face)()
        receipt["static_rule"] = dict(provenance=quad.provenance, max_error=quad.max_error)
        count = len(quad.tau)
    # Remove charge rows/columns locally; only TT has a body contact.
    width = layout.carrier_extent(0)//layout.mesh_side
    tt_only = shard_map(lambda x: x.at[:, :width, :].set(0).at[:, :, :width].set(0),
        mesh=mesh_xy, in_specs=P(None, "x", "y"), out_specs=P(None, "x", "y"), check_vma=False)
    pi_fd = jax.jit(tt_only)(raw[:, 0] * (_w_solve_pref_scalar(meta)/float(meta.cell_volume)))
    pi_grid = pi_fd + drude
    contact = pi_grid + drude
    jax.block_until_ready((pi_grid, drude, contact))
    receipt["correlation_count"] += count
    receipt["contact"] = dict(equation="Pi_grid(0,0)+D", diagonal_reference="Pi_FD=Pi_grid-D",
        units="physical response density, 1/Omega", scope="built once for this bank state")
    return pi_grid, drude, contact


def response_stream(wfns, meta, *, mesh_xy, q_ids, n_outputs,
                    pair_mode="retarded", bank_carry=False, ordered=False,
                    vertex=None):
    """Bind the existing one-particle Green/FFT primitive to a q batch.

    Returns a jitted kernel and its fixed ψ/energy arguments. Caller supplies
    time, projections, final weights and energy reference. The output is
    ``[len(q_ids), n_outputs, mu_p, mu_p]`` with both endpoints sharded.
    ``ordered`` (time reversal measured broken) returns the physical
    orientation ``chi_q = FT_q[chi]`` that Sigma's contraction assumes.
    """
    from .w_isdf import _get_chi_fractional_contour_kernel_face

    from file_io.shared_pole_store import charge_representation

    if vertex is not None:
        if pair_mode == "laplace":
            raise ValueError("GATE response_vertex: photon Laplace cells must retain odd rows")
        n = int(vertex[0][0].shape[2])
        kernel = _get_chi_fractional_contour_kernel_face(
            mesh_xy, (meta.nkx, meta.nky, meta.nkz), n_outputs,
            (int(meta.nk_tot), int(wfns.slices.nb_full), n, 4),
            layout=wfns.layout, selected_q=tuple(q_ids), pair_mode=pair_mode,
            bank_carry=bank_carry, ordered=True, vertex=True)
        return kernel, vertex
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
        bank_carry=bank_carry, ordered=ordered)
    return kernel, (source.psi_mun, source.psi_nmu, source.enk)


def stream_weights(wfns, weights, mesh_xy, *, parents=True):
    """Place small band weights and restrict to existing raw parents."""
    from common.collectives import replicate_to_mesh

    result = replicate_to_mesh(np.asarray(weights), mesh_xy)
    if parents and wfns.green_parent is not None:
        result = wfns.green_parent.plan.parent_rows(result)
    return result


def exact_bare_moments(wfns, meta, *, mesh_xy, q_ids, execute, ordered=False,
                       vertex=None):
    """Compute A0/A1 of scaled chi=A0/s+A1/s² by six correlations.

    The binomial coefficients expand ``(E_u-E_f)`` and its cube. Imaginary
    particle weights turn the retarded primitive's difference into the sum
    of both orientations at t=0 (Run183 energy-power owner). ``execute``
    admits compiled aggregate memory before calling each kernel and records
    its timing. Returned moments are face-sharded ``[b,mu_p,mu_p]``.
    """
    from .w_isdf import _w_solve_pref_scalar

    energy, f, u, reference, census = response_weights(wfns, meta)
    ordered = ordered or vertex is not None
    weights = partial(stream_weights, parents=vertex is None)
    erel = energy - reference
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
                                    q_ids=q_ids, n_outputs=1, ordered=ordered,
                                    vertex=vertex)
    terms = (((-1., 1, 0), (1., 0, 1)),
             ((-1., 3, 0), (3., 2, 1), (-3., 1, 2), (1., 0, 3)))
    totals = []
    for moment_terms in terms:
        total = None
        for coefficient, a, b in moment_terms:
            weight_f = weights(wfns, f * erel**a, mesh_xy)
            weight_u = weights(wfns, -1j * u * erel**b, mesh_xy)
            args = (jnp.asarray([0.]), jnp.asarray([[1. + 0j]]), *fixed,
                    weight_f.astype(jnp.complex128), weight_u,
                    jnp.asarray(reference))
            raw = execute(kernel, args, "moment_correlation")[:, 0]
            term = (_w_solve_pref_scalar(meta) * coefficient) * raw
            total = term if total is None else total + term
            total.block_until_ready()
        totals.append(total)
    if not ordered:
        return (*totals, census)
    # Odd coefficients of 1/z and 1/z^3: sum (P - conj P_{-q}) Delta^m, m=0,2.
    # Real particle weights keep the retarded difference -i(X - conj X), so
    # the chi coefficient is i*raw. Four more correlations, same kernel.
    for moment_terms in (((1., 0, 0),), ((1., 2, 0), (-2., 1, 1), (1., 0, 2))):
        total = None
        for coefficient, a, b in moment_terms:
            weight_f = weights(wfns, f * erel**a, mesh_xy)
            weight_u = weights(wfns, u * erel**b, mesh_xy)
            args = (jnp.asarray([0.]), jnp.asarray([[1. + 0j]]), *fixed,
                    weight_f.astype(jnp.complex128), weight_u.astype(jnp.complex128),
                    jnp.asarray(reference))
            raw = execute(kernel, args, "moment_correlation")[:, 0]
            term = (1j * _w_solve_pref_scalar(meta) * coefficient) * raw
            total = term if total is None else total + term
            total.block_until_ready()
        totals.append(total)
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
    if not charge_representation(meta) and "photon_v" not in bank_io:
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


def resource_digest(path):
    """SHA256 of one immutable resource: read on rank 0, broadcast to all.

    The producer stamps a resource with this and every consumer checks it with
    the same call, so the two cannot drift. Every rank leaves it with the same
    string, which is what keeps a refusal from being rank-conditional
    (INVARIANTS 21).
    """
    from jax.experimental import multihost_utils

    path = Path(path)
    stat = path.stat()
    digest = np.zeros(32, dtype=np.uint8)
    if jax.process_index() == 0:
        digest[:] = np.frombuffer(bytes.fromhex(_resource_hash(
            str(path), stat.st_size, stat.st_mtime_ns)), dtype=np.uint8)
    return bytes(np.asarray(multihost_utils.broadcast_one_to_all(digest))).hex()


def authenticate_coulomb(bank_io, qids):
    """Authenticate bounded-read Coulomb resource against its fixed identity."""
    resource = bank_io["coulomb"]
    if resource["basis"] not in ("canonical", "photon") or not np.array_equal(
            resource["q_irr_full_idx"], qids):
        raise ValueError("GATE response_coulomb_identity: wrong basis/q order")
    if resource_digest(resource["path"]) != resource["sha256"]:
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


def _bank_execution(meta, mesh_xy, receipt, config):
    """Compile and admit new dense work; stream outputs are reserved by batch."""
    def execute(kernel, args, stage):
        with timing.fenced_section('bank.compile.' + stage):
            started = time.monotonic()
            executable = kernel.lower(*args).compile()
            receipt["seconds"]["compilation"] = (receipt["seconds"].get("compilation", 0.)
                + time.monotonic() - started)
        with timing.fenced_section('bank.admission.' + stage):
            memory = executable.memory_analysis()
            if memory is None:
                raise ValueError("GATE response_capacity: compiled memory unavailable")
            stream = stage in ("real_time", "laplace", "moment_correlation", "static_reference")
            if not stream:
                layout = config.get("linalg", "local") if hasattr(config,"get") else config.backend.linalg
                native = response_dense_workspace(mesh_xy,args[0].shape[-1],args[0].shape[0],layout,
                    with_eigh=stage=="coulomb_sqrt")
                receipt.setdefault("native_queries",[]).append(dict(stage=stage,**native))
                _, row = _reserve(meta, stage, memory.argument_size_in_bytes,
                    memory.output_size_in_bytes + memory.temp_size_in_bytes + native["total"])
                receipt["memory"].append(row)
            receipt["compiled"].append(dict(stage=stage,
                arguments=memory.argument_size_in_bytes, outputs=memory.output_size_in_bytes,
                temporaries=memory.temp_size_in_bytes, inherited_stream=stream))
        with timing.fenced_section('bank.execute.' + stage):
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


@lru_cache(maxsize=8)
def _coulomb_pack(basis,mesh_xy):
    return jax.jit(lambda v: basis.pack_operator(v,spec=P(None,"x","y")))


def _coulomb_batch(meta, config, bank_io, mesh_xy, q_span, execute):
    """Read one authenticated canonical V batch; convert through its owner."""
    if "photon_v" in bank_io:
        value = bank_io["photon_v"][q_span[0]:q_span[1]]
        return value, None, [value.shape[-1]] * (q_span[1]-q_span[0])
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


#: Largest Taylor ratio a remote Laplace cell may carry (response_windows).
#: At rho <= 0.3 ``minimax.response_laplace_rule`` stops at order <= 20 for the
#: production tolerance 1e-8 and amplifies row errors by (1+rho)/(1-rho) <= 1.86
#: (value), <= 3.45 (s derivative), so its rows need only ~tol/14; Fe's semicore
#: cell at rho = 0.72 needs order 76 (budget 64) and rows at ~tol/151.
REMOTE_RHO_MAX = 0.3


def _window_cells(energy, ft, ut, masks):
    """Laplace cells (lower, upper) between the partitioned windows."""
    cells = []
    for lower, upper in ((0, 1), (0, 2), (1, 2)):
        bounds, active, partner = [], 0, []
        for lw, uw in ((ft, ut), (ut, ft)):
            low = energy[masks[lower] & (lw != 0)]
            high = energy[masks[upper] & (uw != 0)]
            if low.size and high.size:
                bounds.append((float(high.min()-low.max()),
                               float(high.max()-low.min())))
                partner.append((float(low.max()), float(high.min())))
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
            references_ry=refs, active_global_pairs=active,
            nearest_partner_ry=(max(v[0] for v in partner),
                                min(v[1] for v in partner))))
    return cells


def response_windows(energy, f, u, *, chemical_potential_ry, z_ry):
    """Partition the Run183/188 windowed response into one stream and cells.

    States split by energy relative to the current chemical potential into
    lower remote | real-time stream | upper remote windows. The sample-only
    activity floor is 1e-14. Exact moments do not call this routine. Bounds
    are extrema over all k/band pairings, so they cover every q without
    constructing a transition table.

    Edges derive from the bank's own samples ``z_ry`` [sample] (Ry). A cell
    with transition floor delta_lo is remote only if the Taylor ratio of
    ``minimax.response_laplace_rule`` at these samples and their smallest
    eta, rho = max|z**2+eta**2|/(delta_lo**2+eta**2), is at most
    ``REMOTE_RHO_MAX``; otherwise the remote states nearest the cell's
    partner window join the real-time stream (the lower remote side for
    cells (0,1) and (0,2), the upper for (1,2)) and the cells are rebuilt,
    worst cell first, until every cell passes. This is the repartition the
    Laplace rule's refusal names. The campaign edges [-35,40] eV are a floor:
    the stream never narrows below them, so a deck whose cells already pass
    is partitioned exactly as before.
    """
    ft = np.where(np.abs(f) >= 1e-14, f, 0.0)
    ut = np.where(np.abs(u) >= 1e-14, u, 0.0)
    physical = (f != 0) | (u != 0)
    ev = (energy - chemical_potential_ry) * RYD_TO_EV
    z = np.asarray(z_ry, dtype=np.complex128)
    eta = float(z.imag.min())
    reach = float(np.abs(z*z + eta*eta).max())
    floor = (-35.0, 40.0)
    edges = list(floor)
    while True:
        masks = [physical & (ev < edges[0]),
                 physical & (ev >= edges[0]) & (ev <= edges[1]),
                 physical & (ev > edges[1])]
        # A remote diagonal with both sectors occupied cannot be discarded.
        # Merge it into the resonant stream before constructing any cells.
        for idx in (0, 2):
            if np.any(ft * masks[idx]) and np.any(ut * masks[idx]):
                masks[1] |= masks[idx]
                masks[idx] = np.zeros_like(masks[idx])
        cells = _window_cells(energy, ft, ut, masks)
        for cell in cells:
            cell["taylor_rho"] = reach/(cell["delta_min_ry"]**2 + eta*eta)
        failing = [c for c in cells if c["taylor_rho"] > REMOTE_RHO_MAX]
        if not failing:
            break
        cell = min(failing, key=lambda c: c["delta_min_ry"])
        delta = np.sqrt(reach/REMOTE_RHO_MAX - eta*eta)
        side = 0 if cell["lower"] == 0 else 2
        remote = masks[side] & ((ft != 0) | (ut != 0))
        if side == 0:
            move = remote & (energy > cell["nearest_partner_ry"][1] - delta)
            move |= remote & (energy == energy[remote].max())
            edges[0] = float(ev[move].min())
        else:
            move = remote & (energy < cell["nearest_partner_ry"][0] + delta)
            move |= remote & (energy == energy[remote].min())
            edges[1] = float(ev[move].max())
    receipt = dict(window_ev_relative_mu=[float(v) for v in edges],
        occupation_activity_floor=1e-14,
        discarded_f_mass=float(np.sum(np.abs(f-ft))),
        discarded_u_mass=float(np.sum(np.abs(u-ut))),
        bound_scope="all active k/band extrema, safe for every q",
        edge_rule=dict(floor_ev_relative_mu=list(floor), rho_max=REMOTE_RHO_MAX,
            eta_ry=eta, sample_reach_ry2=reach,
            derived_edge_binds=[float(v) for v in edges] != list(floor),
            cell_taylor_rho=[c["taylor_rho"] for c in cells]))
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


def compute_moment_bank(wfns, meta, config, *, mesh_xy, sym, bank_io,
                        vertex=None, contact=None):
    """Stage B: six exact correlations, physical recurrence, scratch write."""
    from file_io.shared_pole_store import write_shared_pole_bank
    header, qids, census = _bank_context(wfns, meta, sym, bank_io, mesh_xy)
    receipt = _receipt("moments", census, bank_io)
    if not bool(sym.trs_allowed):
        # M1/M3 are the 1/s and 1/s^2 coefficients; the odd channel starts at
        # 1/z^3, so the same six correlations stay exact on an ordered bank.
        receipt["ordered"] = True
    execute = _bank_execution(meta, mesh_xy, receipt, config)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    started = time.monotonic()
    if vertex is None:
        _stream_comparison(wfns, meta, mesh_xy, qids, receipt)
    n = meta.mu_basis.n_packed if vertex is None else vertex[0][0].shape[2]
    face_bytes = 16*n**2 // mesh_xy.size
    # Two totals, next correlation, arithmetic temporaries and bounded H/solve.
    ordered = vertex is not None or not bool(sym.trs_allowed)
    _, moments, receipt["algebra"] = response_algebra(meta, config,
        mesh_xy=mesh_xy, n=n, ordered=ordered, photon=vertex is not None)
    per_q = 12 if ordered else 8
    qwidth = max(1,min(len(qids),int((.75*ledger.U_bytes_per_rank/face_bytes-16)/per_q)))
    for q0 in range(0,len(qids),qwidth):
        q1 = min(q0+qwidth,len(qids))
        ledger.live_stages = ambient
        name,_ = _reserve(meta,"bank_outputs_moments",(per_q*(q1-q0)+16)*face_bytes)
        ledger.live_stages = ambient+(name,)
        if not np.asarray(header["moment_written"])[q0:q1].all():
            if ordered:
                a0, a1, o0, o1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh_xy,
                    q_ids=tuple(qids[q0:q1]), execute=execute, ordered=True, vertex=vertex)
            else:
                a0, a1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh_xy,
                                          q_ids=tuple(qids[q0:q1]), execute=execute)
            for iq in range(q0,q1):
                marked = header["moment_written"][iq]
                if all(marked):
                    continue
                span = (iq,iq+1)
                h, hi, ranks = _coulomb_batch(meta, config, bank_io, mesh_xy, span, execute)
                del hi
                odd = {}
                if ordered:
                    part = slice(iq-q0, iq-q0+1)
                    operands = (h,a0[part],a1[part],o0[part],o1[part])
                    result = execute(moments, operands + (() if vertex is None else (contact,)), "moment_dyson")
                    if vertex is not None:
                        constant, *result = result
                    M0, m1, M2, m3 = result
                    _record_odd_moments(iq, M0, m1, M2, m3, receipt)
                    # The ordered bank's moment masks are (M1, M3, M0, M2).
                    odd = dict(M0=None if marked[2] else M0, M2=None if marked[3] else M2)
                    if vertex is not None:
                        odd["constant"] = None if marked[4] else constant
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
            del a0,a1
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


@jax.jit
def _reciprocity_scalars(value):
    """``max|W - W^T| / max|W|`` for each sample of one parent's batch."""
    defect = jnp.max(jnp.abs(value - jnp.swapaxes(value, -1, -2)), axis=(-2, -1))
    scale = jnp.max(jnp.abs(value), axis=(-2, -1))
    return defect / jnp.where(scale > 0, scale, 1)


def _reciprocity_census(receipt, value, z_batch, q_full, parent, meta):
    """Record the reciprocity defect ``W_q(z) = W_q(z)^T`` at REAL supports.

    ONLY AT A SELF-NEGATIVE (TRIM) PARENT, where ``W_q = W_q^T`` is an exact
    property of one stored tile. At a generic q the exact relation is
    ``W_q^T = W_{-q}``, ACROSS parents, and a tile's own transpose defect is
    O(1) by correct physics (1.3-1.8 on this deck) — recording it there would
    label correct physics as a defect.

    Where it applies, the transpose-antisymmetric part is pure error with a
    known-zero target. At a real support (``z = iu``, so ``s = z**2`` is real)
    such a tile is additionally real, so the same number is then its
    Hermiticity defect. Off the real axis W must NOT be Hermitian, so the
    transpose form is the one that is meaningful at every support.

    This row REFUSES NOTHING. It is recorded because it is the quantity that
    fails first when a self-consistent loop amplifies an off-manifold
    reciprocity-violating mode, and nothing else in the construction measures
    it: the shared-pole Gram gate sees it only after the Loewner pencil has
    amplified it by ~1e2 (KNOWN_LORRAX_ISSUES 2026-09-16 SCGRAM-A/SCGRAM-B).
    Every rank evaluates it; no rank-conditional device work (INVARIANTS 21).
    """
    if not _self_negative(int(q_full), meta):
        return
    points = np.asarray(z_batch)
    rows = [i for i, z in enumerate(points) if z.real == 0.0]
    if not rows:
        return
    scalars = np.asarray(_reciprocity_scalars(value))
    receipt.setdefault("reciprocity_real_supports", []).extend(
        dict(q_parent=int(parent), q_full=int(q_full),
             u_ry=float(points[i].imag), defect=float(scalars[i]))
        for i in rows)


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


def response_quadrature(wfns, meta, sample_plan, receipt, *, ordered=False):
    """Plan the same windowed real-time/Laplace rules for charge or photon panels.

    ``sample_plan`` is the existing authenticated support plan. The returned
    plain dictionary holds only band tables and quadrature rows, no operators.
    """
    z = bank_points(sample_plan)
    energy,f,u,reference,_ = response_weights(wfns,meta)
    masks,ft,ut,cells,receipt["windows"] = response_windows(energy,f,u,
        chemical_potential_ry=sample_plan["census"]["mu_ry"],z_ry=z)
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
    return dict(t=t, phase=phase, derivative=derivative, remote=remote,
                masks=masks, ft=ft, ut=ut, reference=reference)


def integrate_response_panel(wfns, meta, mesh_xy, rules, *, q_ids, sample_span,
                             execute, receipt, ordered=False, vertex=None):
    """Integrate an admitted panel through the single Green/FFT stream.

    Returns ``[2*n_sample,n_q,n_mu,n_mu]`` at P(None,None,x,y), value rows
    followed by d/ds rows. Photon endpoints are prepared once by
    ``prepare_photon_carriers``. The caller's ``execute`` admits the compiled
    memory before execution, as in the charge bank.
    """
    qids = tuple(q_ids)
    lo, hi = sample_span
    a = hi-lo
    ordered = ordered or vertex is not None
    n = meta.mu_basis.n_packed if vertex is None else vertex[0][0].shape[2]
    weights = partial(stream_weights, parents=vertex is None)
    t, phase, derivative, remote, masks, ft, ut, reference = (
        rules[k] for k in ("t", "phase", "derivative", "remote", "masks", "ft", "ut", "reference"))
    kernel,fixed = response_stream(wfns,meta,mesh_xy=mesh_xy,
        q_ids=tuple(qids),n_outputs=2*a,bank_carry=True,ordered=ordered,vertex=vertex)
    raw = jax.jit(lambda: jnp.zeros((2*a,len(qids),n,n),jnp.complex128),
        out_shardings=NamedSharding(mesh_xy,P(None,None,"x","y")))()
    raw = execute(kernel,(jnp.asarray(t),jnp.asarray(np.vstack((phase[lo:hi],derivative[lo:hi]))),
        *fixed,weights(wfns,ft*masks[1],mesh_xy),
        weights(wfns,ut*masks[1],mesh_xy),jnp.asarray(reference),raw),"real_time")
    receipt["correlation_count"] += len(t)
    # Cell data are dynamic arguments; reuse one compiled owner for
    # equal-shaped Laplace cells instead of retracing each closure.
    if remote:
        lk,lfixed = response_stream(wfns,meta,mesh_xy=mesh_xy,
            q_ids=tuple(qids),n_outputs=2*a,
            pair_mode="laplace_ordered" if ordered else "laplace",bank_carry=True,vertex=vertex)
    for cell,rr in remote:
        with timing.fenced_section('bank.laplace_arguments'):
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
            lw = jnp.stack([weights(wfns,x,mesh_xy) for x in lw])
            uw = jnp.stack([weights(wfns,x,mesh_xy) for x in uw])
        if ordered and vertex is None and "ordered_cost_probe" not in receipt:
            receipt["ordered_cost_probe"] = _ordered_cost_probe(
                wfns,meta,mesh_xy,int(qids[0]),tau,projections,lw,uw,refs,2*a)
        raw = execute(lk,(jnp.asarray(tau),jnp.asarray(projections),
            *lfixed,lw,uw,jnp.asarray(refs),raw),"laplace")
        del lw,uw
        receipt["correlation_count"] += 2*len(tau)
    return raw


def produce_sample_bank(wfns, meta, config, *, mesh_xy, sym, sample_plan, bank_io,
                        vertex=None, contact=None):
    """Stage A: one windowed stream per admitted sample batch, all parent faces."""
    with timing.fenced_section('bank.setup'):
        from file_io.shared_pole_store import write_shared_pole_bank
        header,qids,census = _bank_context(wfns,meta,sym,bank_io,mesh_xy)
        authenticate_sample_plan(sample_plan,header)
        z = bank_points(sample_plan)
        receipt = _receipt("samples",census,bank_io)
        # Time reversal measured broken: both particle-hole orientations keep
        # independent weights. The retarded stream already forms the partner as
        # conj in R space (the -q orientation); remote cells add the odd kernel.
        # ordered=True stores the physical orientation W_q = FT_q[W].
        ordered = vertex is not None or not bool(sym.trs_allowed)
        literal_mirrors = vertex is not None
        if literal_mirrors and header.get("mirror_mode") != "literal_same_operator_v1":
            raise ValueError("GATE response_mirror_contract: photon production requires a new literal-mirror bank")
        if literal_mirrors:
            from symmetry_maps.maps import q_negation_index
            negative = np.asarray(q_negation_index((int(meta.nkx), int(meta.nky), int(meta.nkz))), dtype=np.int64)
            mirror_qids = negative[qids]
            receipt["mirror_contract"] = dict(header["mirror_contract"],
                mode=header["mirror_mode"], support_count=len(z),
                operator_provenance=bank_io["mirror_operator_provenance"],
                original_parent_count=len(qids),
                full_q_rows=len(set(qids.tolist()+mirror_qids.tolist())),
                green_stream="union of exact q and minus-q output rows in the same response panel",
                dyson="original parent V/contact for both orientations; same moment operator")

        def panel_rows(first, last):
            rows = qids[first:last].tolist()
            if literal_mirrors:
                rows = list(dict.fromkeys(rows+mirror_qids[first:last].tolist()))
            return tuple(rows)

        n = meta.mu_basis.n_packed if vertex is None else vertex[0][0].shape[2]
        if ordered:
            receipt["ordered"] = True
        started = time.monotonic()
        execute = _bank_execution(meta, mesh_xy, receipt, config)
        ledger = meta.shared_pole_capacity
        ambient = ledger.live_stages
    with timing.fenced_section('bank.stream_reference_compile'):
        if vertex is None:
            _stream_comparison(wfns,meta,mesh_xy,qids,receipt)
    with timing.fenced_section('bank.window_geometry'):
        samples,_,receipt["algebra"] = response_algebra(meta,config,
            mesh_xy=mesh_xy,n=n,photon=vertex is not None)
        rules = response_quadrature(wfns, meta, sample_plan, receipt, ordered=ordered)
        phase, derivative = rules["phase"], rules["derivative"]
    with timing.fenced_section('bank.capacity_planning_compile'):
        face_bytes = 16*n**2//mesh_xy.size
        # One donated internal [output,q,x,y] carry spans every window. Public
        # writer slices are [q,output,x,y]; only that bounded slice is transposed.
        # Reserve output plus a dense/transport headroom, and batch only when the
        # common ledger's admitted panel budget cannot hold the full point plan.
        layout = config.get("linalg","local") if hasattr(config,"get") else config.backend.linalg
        native = response_dense_workspace(mesh_xy,n,len(z),layout,with_eigh=vertex is None)
        headroom = 16*face_bytes + native["gemm"] + int(phase.nbytes+derivative.nbytes)
        # Ask the ledger owner for R24's remaining device budget. The zero-byte
        # planning row includes ambient live reservations but allocates nothing.
        _, budget = _reserve(meta,"bank_planning",0)
        live_bytes = budget["aggregate_bytes_per_rank"]
        device_available = budget["available_device_bytes_per_rank"]
        scaling_target = budget["limit_bytes_per_rank"]

        @lru_cache(maxsize=None)
        def dense_bytes(width):
            abstract = jax.ShapeDtypeStruct((width,n,n),
                jnp.complex128,sharding=NamedSharding(mesh_xy,P(None,"x","y")))
            stats = samples.lower(*((abstract,)* (3 if vertex is None else 4))).compile().memory_analysis()
            if stats is None:
                raise ValueError("GATE response_capacity: sample solve planning memory unavailable")
            dense = stats.argument_size_in_bytes+stats.output_size_in_bytes+stats.temp_size_in_bytes
            # Preserve the existing dense/native and writer conversion envelopes.
            return max(dense+native["total"],4*width*face_bytes+native["total"])

        minimum = headroom+live_bytes+2*max(len(panel_rows(i,i+1)) for i in range(len(qids)))*face_bytes+dense_bytes(1)
        # R24 makes 3U a reported scaling target, not the device admission limit.
        # Replaying a Green/FFT stream to meet that preference repeats every time
        # node even when the complete output panel fits. Use the ledger's remaining
        # device budget, including the inherited stream and ambient/native costs;
        # larger systems still split q/sample panels before any allocation.
        planning_limit = device_available
        available = planning_limit-headroom-live_bytes
        qwidth = min(len(qids),int((available-dense_bytes(1))//(2*face_bytes)))
        while qwidth > 0 and any(2*len(panel_rows(i,min(i+qwidth,len(qids))))*face_bytes+dense_bytes(1) > available
                                  for i in range(0,len(qids),qwidth)):
            qwidth -= 1
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
        with timing.fenced_section('bank.panel_admission'):
            q1 = min(q0+qwidth,len(qids))
            response_rows = panel_rows(q0,q1)
            row_index = {q: i for i,q in enumerate(response_rows)}
            width = len(z)
            while 2*width*len(response_rows)*face_bytes+dense_bytes(width) > available:
                width -= 1
            planned_bytes = headroom+live_bytes+2*width*len(response_rows)*face_bytes+dense_bytes(width)
            receipt.setdefault("panel_plans",[]).append(dict(q_span=(q0,q1),sample_width=width, response_q_full_idx=response_rows,
                aggregate_bytes_per_rank=planned_bytes,
                scaling_status="PASS" if planned_bytes <= scaling_target else "WARN",
                device_budget_status="PASS",scaling_target_bytes_per_rank=scaling_target,
                available_device_bytes_per_rank=device_available))
        for lo in range(0,len(z),width):
            with timing.fenced_section('bank.stream_arguments'):
                hi = min(lo+width,len(z));a = hi-lo
                ledger.live_stages = ambient
                name,_ = _reserve(meta,"bank_outputs",2*a*len(response_rows)*face_bytes + headroom)
                ledger.live_stages = ambient+(name,)
            raw = integrate_response_panel(wfns, meta, mesh_xy, rules,
                q_ids=response_rows, sample_span=(lo, hi), execute=execute,
                receipt=receipt, ordered=ordered, vertex=vertex)
            for iq in range(q0,q1):
                span = (iq,iq+1)
                if np.asarray(header["sample_written"])[iq,lo:hi].all():
                    continue
                h,hinv,ranks = _coulomb_batch(meta,config,bank_io,mesh_xy,span,execute)
                del hinv
                if vertex is not None:
                    from file_io.shared_pole_store import read_shared_pole_bank
                    from file_io.slab_io import SlabIO
                    with SlabIO(bank_io["path"], mode="r", mesh=mesh_xy) as io:
                        constant = read_shared_pole_bank(io, span, meta=meta, header=header,
                                                       fields=("constant",))["constant"]
                ia = lo
                while ia < hi:
                    marked = tuple(header["sample_written"][iq][ia])
                    stop = ia+1
                    while stop < hi and tuple(header["sample_written"][iq][stop]) == marked:
                        stop += 1
                    if all(marked):
                        ia = stop
                        continue
                    # Exact PH partners are formed at the bare-response level.
                    # Dyson always uses this original parent's V and contact;
                    # no spatial reconstruction of a screened operator enters.
                    for mirror in range(2 if literal_mirrors else 1):
                        field0 = 2*mirror
                        if marked[field0] and marked[field0+1]:
                            continue
                        row = row_index[int(mirror_qids[iq] if mirror else qids[iq])]
                        chi = raw[ia-lo:stop-lo,row]
                        dchi = raw[a+ia-lo:a+stop-lo,row]
                        if mirror:
                            chi, dchi = jnp.conj(chi), jnp.conj(dchi)
                        hbatch = jnp.broadcast_to(h,chi.shape)
                        operands = (hbatch,chi,dchi) + (() if vertex is None else
                            (jnp.broadcast_to(contact,chi.shape),))
                        value,ds = execute(samples,operands,"mirror_dyson" if mirror else "sample_dyson")
                        if vertex is not None:
                            value = value - constant
                            if not mirror:
                                _photon_sample_norms(receipt, value, iq, ia, bank_io["photon_layout"], mesh_xy)
                        else:
                            _reciprocity_census(receipt,value,z[ia:stop],int(qids[iq]),iq,meta)
                        if vertex is None and ordered and _self_negative(int(qids[iq]),meta):
                            _tr_odd_census(receipt,samples,h,chi,dchi,value,z[ia:stop],int(qids[iq]))
                        value = None if marked[field0] else value[None]
                        ds = None if marked[field0+1] else ds[None]
                        payload = (dict(Wc_mirror=value,dWc_mirror_ds=ds) if mirror
                                   else dict(Wc=value,dWc_ds=ds))
                        with timing.fenced_section('bank.write'):
                            io_started = time.monotonic()
                            header = write_shared_pole_bank(bank_io["path"],q_span=span,
                                sample_span=(ia,stop),**payload,meta=meta,
                                expected_identity=bank_io["identity"],mesh_xy=mesh_xy)
                            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                        del value,ds,chi,dchi,hbatch,operands,payload
                    ia = stop
                del h
            receipt["batches"].append(dict(q_span=(q0,q1),sample_span=(lo,hi)))
            del raw
    with timing.fenced_section('bank.finalize'):
        ledger.live_stages = ambient
        receipt["stream_passes"] = len(receipt["batches"])
        receipt["batch_reason"] = "full plan admitted" if len(receipt["batches"]) == 1 else "remaining device-budget panels require bounded replays; see panel_budget and panel_plans"
        receipt["completion"] = bool(np.asarray(header["sample_written"]).all())
        return _finish_receipt(receipt,meta,header,started)


def _photon_sample_norms(receipt, value, parent, first, layout, mesh_xy):
    """Record Frobenius norms of CC/CT/TC/TT without gathering operators."""
    from .photon_layout import photon_block_view
    norms = {}
    for sector, pairs in (("CC", ((0, 0),)),
                          ("CT", tuple((0, b) for b in range(1, 4))),
                          ("TC", tuple((a, 0) for a in range(1, 4))),
                          ("TT", tuple((a, b) for a in range(1, 4) for b in range(1, 4)))):
        squared = sum(jnp.sum(jnp.abs(photon_block_view(value, layout, a, b, mesh_xy))**2,
                              axis=(-2, -1)) for a, b in pairs)
        norms[sector] = np.asarray(jnp.sqrt(squared)).tolist()
    receipt.setdefault("sector_sample_norms", []).append(
        dict(parent=int(parent), first_sample=int(first), **norms))


def photon_bare_operator(wfns, wfns_transverse, meta, *, path, mu_bases, layout, mesh_xy):
    """Read authenticated raw-parent photon V through its sole packing owner.

    The reader returns MuBasis-packed family tiles. Undo that family packing
    before the photon owner inserts canonical channel chunks, exactly as for
    the endpoint carriers. All operators stay at P(None,x,y).
    """
    from file_io.restart_bundle import BispinorVqReader
    from .photon_layout import pack_photon_operator
    from .v_q_bispinor import ZERO_TILES
    plans = (wfns.green_parent.plan, wfns_transverse.green_parent.plan)
    nq = len(plans[0].sym.q_irr_full_idx)
    with BispinorVqReader(path, mesh_xy, mu_bases=mu_bases, family_plans=plans) as reader:
        if reader.n_q_total != nq:
            raise ValueError("GATE photon_bank_coulomb: parent census differs")
        def block(a, b):
            if (a, b) in ZERO_TILES:
                return None
            value = reader.get_tile(a, b)
            left, right = mu_bases[int(a != 0)], mu_bases[int(b != 0)]
            if left is right:
                return left.unpack_operator(value, spec=P(None, "x", "y"))
            value = left.unpack_axis(value, 1, spec=P(None, "x", "y"))
            return right.unpack_axis(value, 2, spec=P(None, "x", "y"))
        return pack_photon_operator(block, nq, layout, mesh_xy)


def compute_photon_bank(wfns, wfns_transverse, meta, config, *, mesh_xy, sym,
                        mu_bases, layout, occupation_state, sample_plan, bank_io):
    """Build a full photon bank through the existing sample/moment stages.

    ``bank_io`` names the initialized scratch path, current identity and
    ``bispinor_v_q_path``. The stored samples are W-W_infinity; the committed
    ``constant`` field is W_infinity-V. M0..M3 use the same convention as the
    ordered charge bank. Both CT and TC are retained. The scalar producer,
    memory planner, quadrature, transaction masks and reader are shared.
    Later SC maps supply ``bank_io['static_reference']`` with the initial
    bank's ``path`` and ``identity`` to keep its contact fixed.
    """
    from file_io.shared_pole_store import validate_shared_pole_bank
    from file_io.slab_io import SlabIO

    started = time.monotonic()
    header = validate_shared_pole_bank(bank_io["path"],
        expected_identity=bank_io["identity"], mesh_xy=mesh_xy)
    if header.get("photon_layout", {}).get("packed_extent") != layout.packed_extent:
        raise ValueError("GATE photon_bank_layout: scratch has a different packed photon layout")
    expected = [hashlib.sha256(np.asarray(b.canonical_indices, dtype="<i4").tobytes()).hexdigest()
                for b in mu_bases]
    if expected != header["photon_centroid_digests"]:
        raise ValueError("GATE photon_bank_centroids: scratch/current endpoint identity differs")
    bank = dict(bank_io, photon_layout=layout)
    bank["coulomb"] = dict(path=str(bank["bispinor_v_q_path"]), basis="photon",
        q_irr_full_idx=np.asarray(sym.q_irr_full_idx).tolist(),
        sha256=resource_digest(bank["bispinor_v_q_path"]))
    census = response_weights(wfns, meta)[-1]
    receipt = _receipt("photon", census, bank)
    execute = _bank_execution(meta, mesh_xy, receipt, config)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    nq, n = len(sym.q_irr_full_idx), layout.packed_extent
    # Packed V, contact/reference/D and one family-read envelope, all XY tiled.
    vbytes = 16*(nq+4)*n*n//mesh_xy.size
    name, row = _reserve(meta, "photon_endpoints_and_V", vbytes)
    ledger.live_stages = ambient+(name,)
    receipt["memory"].append(row)
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: preparing shared vertex endpoints and bare V", flush=True)
    vertex = prepare_photon_carriers(wfns, wfns_transverse, mu_bases,
                                     mesh_xy=mesh_xy, layout=layout)
    bank["photon_v"] = photon_bare_operator(wfns, wfns_transverse, meta,
        path=bank["bispinor_v_q_path"], mu_bases=mu_bases, layout=layout, mesh_xy=mesh_xy)
    jax.block_until_ready((vertex, bank["photon_v"]))
    receipt["seconds"]["endpoints_and_V"] = time.monotonic()-before
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: centroid D and static grid reference", flush=True)
    reference = bank.get("static_reference")
    if reference is None:
        grid, drude, contact = photon_static_contact(wfns, meta, mesh_xy=mesh_xy,
            layout=layout, vertex=vertex, occupation_state=occupation_state,
            sample_plan=sample_plan, execute=execute, receipt=receipt)
        reference = {key: bank[key] for key in ("path", "identity")}
    else:
        initial = validate_shared_pole_bank(reference["path"],
            expected_identity=reference["identity"], mesh_xy=mesh_xy, require_complete=True)
        for key in ("photon_layout", "photon_centroid_digests"):
            if initial.get(key) != header[key]:
                raise ValueError(f"GATE photon_static_reference: initial/current {key} differs")
        with SlabIO(reference["path"], mode="r", mesh=mesh_xy) as io:
            grid, drude, contact = (io.read_slab(key, shape=(1,n,n),
                partition_spec=P(None,"x","y"), dtype=np.complex128)
                for key in ("Pi_grid", "Drude", "TT_contact"))
    receipt["static_reference"] = reference
    bank["mirror_operator_provenance"] = dict(coulomb=bank["coulomb"],
        static_reference=reference,
        static_reference_commit=(initial["final_commit"] if bank.get("static_reference") is not None else None),
        state_identity=bank["identity"],
        moments="M0,M1,M2,M3 and constant computed with the identical photon_v and contact arrays")
    # Persist the contact's two physically defined pieces as bank diagnostics;
    # the constructor consumes the separately committed constant, not these.
    with SlabIO(bank["path"], mode="a", mesh=mesh_xy) as io:
        for key, value in (("Pi_grid",grid),("Drude",drude),("TT_contact",contact)):
            io.write_slab(key, value, offset=(0,0,0), global_shape=(1,n,n))
            io.sync_writes()
    receipt["seconds"]["static_contact"] = time.monotonic()-before
    del grid, drude
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: exact moments and W_infinity", flush=True)
    receipt["moments"] = compute_moment_bank(wfns, meta, config, mesh_xy=mesh_xy,
        sym=sym, bank_io=bank, vertex=vertex, contact=contact)
    receipt["seconds"]["moments"] = time.monotonic()-before
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: ordered samples and derivatives", flush=True)
    receipt["samples"] = produce_sample_bank(wfns, meta, config, mesh_xy=mesh_xy,
        sym=sym, sample_plan=sample_plan, bank_io=bank, vertex=vertex, contact=contact)
    receipt["seconds"]["samples"] = time.monotonic()-before
    header = validate_shared_pole_bank(bank["path"], expected_identity=bank["identity"],
                                       mesh_xy=mesh_xy, require_complete=True)
    ledger.live_stages = ambient
    receipt["completion"] = True
    receipt["bank_header"] = header
    return _finish_receipt(receipt, meta, header, started)
