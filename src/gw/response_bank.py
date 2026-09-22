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
from common.progress import LoopProgress
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
    value, slope, moments, receipt
        Jitted functions accepting complex128 ``[b,n,n]`` operators at
        ``P(None,'x','y')``, and the resolved backend/prefactor description.
        ``value(H, chi_raw)`` returns Wc (Ry); ``slope(H, Wc, dchi_raw)``
        returns its s derivative (Ry^-1), restoring the bare operator internally (photon input is W-V). ``moments(H, A0, A1)`` takes already scaled
        bare-response expansion coefficients and returns M1/M3 (Ry^3/Ry^5).
        Neither routine Hermitizes its inputs or outputs.
    """
    from .gw_config import linalg_resolution
    from .w_isdf import _w_solve_pref_scalar

    resolution = linalg_resolution(
        config if hasattr(config, "get") else {"linalg": config.backend.linalg})
    route = resolution.batched_route
    backend = "off" if resolution.layout == "local" else "distributed"
    pref = _w_solve_pref_scalar(meta)
    value, slope, moments, lu = _response_programs(
        mesh_xy, n, backend, route, pref, ordered,
        float(meta.cell_volume) if photon else None)
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
        algebra["representation"] = "signed packed photon"
        algebra["constant"] = "W_infinity-V, retained separately from M0..M3"
        algebra["moment_convention"] = "M_k=C_(k+1)/2 about W_infinity"
        algebra["units"].update(M0="Ry^2", M2="Ry^4", constant="Ry")
    return value, slope, moments, algebra


@lru_cache(maxsize=None)
def _response_programs(mesh_xy, n, backend, route, pref, ordered, volume):
    """Reuse compiled algebra across SC maps without retaining state arrays."""
    from distrib_la import matmul, plan

    lu = plan("solve_lu", mesh_xy, backend=backend, n=n,
              batched_route=route)
    face = NamedSharding(mesh_xy, P(None, "x", "y"))

    def mm(a, b):
        return matmul(a, b, mesh=mesh_xy, backend=backend,
                      batched_route=route)

    def congruence(h, a):
        return mm(mm(h, a), h)

    @partial(jax.jit, in_shardings=(face, face), out_shardings=face)
    def value(h, chi_raw):
        x = pref * congruence(h, chi_raw)
        identity = jnp.broadcast_to(jnp.eye(n, dtype=h.dtype), x.shape)
        e = lu.batched(identity - x, identity.copy())
        return congruence(h, mm(x, e))

    def derivative(w, dchi_raw):
        # Full W on BOTH sides; no adjoint at complex frequency.
        return mm(mm(w, pref * dchi_raw), w)

    @partial(jax.jit, in_shardings=(face, face, face), out_shardings=face)
    def slope(h, wc, dchi_raw):
        return derivative(wc + mm(h, h), dchi_raw)

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

    if volume is not None:
        @partial(jax.jit, in_shardings=(face, face), out_shardings=face)
        def infinity(v, contact):
            # chi(z)=chi_param(z)-contact, so W_inf=(I+V contact)^-1 V.
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), v.shape)
            return lu.batched(identity + mm(v, jnp.broadcast_to(volume * contact, v.shape)), v.copy())

        @partial(jax.jit, in_shardings=(face, face, face), out_shardings=face)
        def value(v, chi_raw, contact):
            chi = pref * chi_raw - volume * contact
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), chi.shape)
            return lu.batched(identity - mm(v, chi), v.copy()) - v

        @partial(jax.jit, in_shardings=(face, face, face), out_shardings=face)
        def slope(v, wc, dchi_raw):
            return derivative(wc + v, dchi_raw)

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

    return value, slope, moments, lu


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
        lo, hi = float(energy[f != 0].max()), float(energy[u != 0].min())
        gap = hi-lo
        if gap <= 0:
            raise ValueError("GATE photon_contact_state: gapless state needs FD occupations")
        quad = solve_laplace_minimax_interval(gap,
            float(energy[u != 0].max()-energy[f != 0].min()),
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


@lru_cache(maxsize=16)
def _response_stream_kernel(mesh_xy, kgrid, n_outputs, shape, *, _ffi_key, **options):
    """Cache programs, never state arrays; window data remain dynamic inputs."""
    from .w_isdf import _get_chi_fractional_contour_kernel_face
    return _get_chi_fractional_contour_kernel_face(
        mesh_xy, kgrid, n_outputs, shape, **options)


def response_stream(wfns, meta, *, mesh_xy, q_ids, n_outputs,
                    pair_mode="retarded", bank_carry=False, ordered=False,
                    vertex=None, band_ranges=None):
    """Bind the existing one-particle Green/FFT primitive to a q batch.

    Returns a jitted kernel and its fixed ψ/energy arguments. Caller supplies
    time, projections, final weights and energy reference. The output is
    ``[len(q_ids), n_outputs, mu_p, mu_p]`` with both endpoints sharded.
    ``ordered`` (time reversal measured broken) returns the physical
    orientation ``chi_q = FT_q[chi]`` that Sigma's contraction assumes.
    """
    from ffi import ffi_dial_key

    from file_io.shared_pole_store import charge_representation

    if vertex is not None:
        if pair_mode == "laplace":
            raise ValueError("GATE response_vertex: photon Laplace cells must retain odd rows")
        n = int(vertex[0][0].shape[2])
        kernel = _response_stream_kernel(
            mesh_xy, (meta.nkx, meta.nky, meta.nkz), n_outputs,
            (int(meta.nk_tot), int(wfns.slices.nb_full), n, 4),
            _ffi_key=ffi_dial_key(), layout=wfns.layout, selected_q=tuple(q_ids), pair_mode=pair_mode,
            bank_carry=bank_carry, ordered=True, vertex=True, band_ranges=band_ranges)
        return kernel, vertex
    if not charge_representation(meta):
        raise ValueError("GATE response_representation: want scalar or "
                         "two-component charge endpoints; use the photon vertex "
                         "for bispinor response")
    carrier = wfns.green_parent
    source = wfns if carrier is None else carrier
    parent = None if carrier is None else carrier.plan
    nk = int(meta.nk_tot) if parent is None else int(parent.n_parent)
    n = int(meta.mu_basis.n_packed)
    kernel = _response_stream_kernel(
        mesh_xy, (meta.nkx, meta.nky, meta.nkz), n_outputs,
        (nk, int(wfns.slices.nb_full), n, int(meta.nspinor)),
        k_unfold_plan=parent, _ffi_key=ffi_dial_key(), layout=wfns.layout, selected_q=tuple(q_ids),
        pair_mode=pair_mode, bank_carry=bank_carry, ordered=ordered, band_ranges=band_ranges)
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


def _bank_execution(meta, mesh_xy, receipt, config, *, photon=False):
    """Compile and admit new dense work; stream outputs are reserved by batch."""
    def execute(kernel, args, stage):
        with timing.fenced_section('bank.compile.' + stage, announce=True):
            started = time.monotonic()
            executable = kernel.lower(*args).compile()
            receipt["seconds"]["compilation"] = (receipt["seconds"].get("compilation", 0.)
                + time.monotonic() - started)
        with timing.fenced_section('bank.admission.' + stage, announce=True):
            memory = executable.memory_analysis()
            if memory is None:
                raise ValueError("GATE response_capacity: compiled memory unavailable")
            stream = stage in ("real_time", "laplace", "windowed", "direct", "moment_correlation", "static_reference")
            if stream:
                # The caller reserves the sole carry; admit the actual
                # Green/FFT workspace, without compiling a legacy comparator.
                _, row = _reserve(meta, stage + "_temporaries", 0,
                                  memory.temp_size_in_bytes)
                receipt["memory"].append(row)
            elif not stream:
                layout = config.get("linalg", "local") if hasattr(config,"get") else config.backend.linalg
                native = response_dense_workspace(mesh_xy,args[0].shape[-1],args[0].shape[0],layout,
                    with_eigh=stage=="coulomb_sqrt")
                receipt.setdefault("native_queries",[]).append(dict(stage=stage,**native))
                _, row = _reserve(meta, stage, memory.argument_size_in_bytes,
                    memory.output_size_in_bytes + memory.temp_size_in_bytes + native["total"])
                receipt["memory"].append(row)
            receipt["compiled"].append(dict(stage=stage,
                arguments=memory.argument_size_in_bytes, outputs=memory.output_size_in_bytes,
                temporaries=memory.temp_size_in_bytes,
                aliases=memory.alias_size_in_bytes,
                inherited_stream=False,
                stream_temporaries_admitted=stream))
        with timing.fenced_section('bank.execute.' + stage, announce=stream,
                                  label=f"shared-pole bank {stage} execute"):
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
        refs = (max(v[0] for v in partner), min(v[1] for v in partner))
        if refs[1] <= refs[0]:
            raise ValueError("GATE response_remote_cell: unordered energy windows")
        cells.append(dict(lower=lower, upper=upper,
            delta_min_ry=min(v[0] for v in bounds),
            delta_max_ry=max(v[1] for v in bounds),
            references_ry=refs, active_global_pairs=active))
    return cells


def response_sample_weights(f, u):
    """Existing sample-only activity floor; exact moments retain every weight."""
    ft = np.where(np.abs(f) >= 1e-14, f, 0.0)
    ut = np.where(np.abs(u) >= 1e-14, u, 0.0)
    return ft, ut, dict(occupation_activity_floor=1e-14,
        discarded_f_mass=float(np.sum(np.abs(f-ft))),
        discarded_u_mass=float(np.sum(np.abs(u-ut))))


def response_windows(energy, f, u, *, chemical_potential_ry, z_ry):
    """One crossing state window and three noncrossing product cells.

    Include all occupation tails and transitions that can resonate at the
    actual Re(z), with the same 1.5*eta separation used by Sigma's geometry.
    Imaginary-axis supports do not widen the crossing window. Bounds use
    all k/band extrema, so no transition table or q-specific partition is needed.
    The sample-only 1e-14 activity floor does not affect exact moments.
    """
    ft, ut, activity = response_sample_weights(f, u)
    physical = (f != 0) | (u != 0)
    z = np.asarray(z_ry, dtype=np.complex128)
    reach = float(np.abs(z.real).max())
    margin = 1.5*float(z.imag.min())
    occupied, empty = energy[ft != 0], energy[ut != 0]
    if not occupied.size or not empty.size:
        raise ValueError("GATE response_windows: no active occupied or empty states")
    # Include both occupation extrema even for gaps larger than the sample range.
    lower = min(float(empty.min())-reach-margin, float(occupied.max()))
    upper = max(float(occupied.max())+reach+margin, float(empty.min()))
    masks = [physical & (energy < lower),
             physical & (energy >= lower) & (energy <= upper),
             physical & (energy > upper)]
    cells = _window_cells(energy, ft, ut, masks)
    receipt = dict(window_ev_relative_mu=[
        (v-chemical_potential_ry)*RYD_TO_EV for v in (lower, upper)],
        **activity,
        bound_scope="all active k/band extrema, safe for every q",
        edge_rule=dict(max_real_z_ry=reach, noncrossing_margin_ry=margin))
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
    execute = _bank_execution(meta, mesh_xy, receipt, config, photon=vertex is not None)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    started = time.monotonic()
    n = meta.mu_basis.n_packed if vertex is None else vertex[0][0].shape[2]
    face_bytes = 16*n**2 // mesh_xy.size
    # Two totals, next correlation, arithmetic temporaries and bounded H/solve.
    ordered = vertex is not None or not bool(sym.trs_allowed)
    _, _, moments, receipt["algebra"] = response_algebra(meta, config,
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
            h, hi, ranks = _coulomb_batch(meta, config, bank_io, mesh_xy, (q0,q1), execute)
            del hi
            operands = (h,a0,a1,o0,o1) if ordered else (h,a0,a1)
            result = execute(moments, operands + (() if vertex is None else (contact,)),
                             "moment_dyson")
            names = (("constant", "M0", "M1", "M2", "M3") if vertex is not None else
                     (("M0", "M1", "M2", "M3") if ordered else ("M1", "M3")))
            values = dict(zip(names,result))
            if ordered:
                for iq in range(q0,q1):
                    part = slice(iq-q0,iq-q0+1)
                    _record_odd_moments(iq, *(values[name][part] for name in
                        ("M0", "M1", "M2", "M3")), receipt)
            # One write for a fresh batch; partial restarts group identical
            # commit masks so no already committed field is overwritten.
            marked = np.asarray(header["moment_written"], bool)[q0:q1]
            edges = np.r_[0, 1+np.flatnonzero(np.any(marked[1:] != marked[:-1],axis=1)), q1-q0]
            fields = ("M1", "M3", "M0", "M2", "constant")[:marked.shape[1]]
            for lo,hi in zip(edges[:-1],edges[1:]):
                if marked[lo].all():
                    continue
                span = (int(q0+lo),int(q0+hi))
                io_started = time.monotonic()
                header = write_shared_pole_bank(bank_io["path"], q_span=span,
                    **{name: values[name][lo:hi] for i,name in enumerate(fields) if not marked[lo,i]},
                    meta=meta, expected_identity=bank_io["identity"], mesh_xy=mesh_xy)
                receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                receipt["batches"].append(dict(q_span=span,support_ranks=ranks[lo:hi]))
            del h,a0,a1,operands,result,values
            if ordered:
                del o0,o1
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


def _tr_odd_census(receipt, solve_value, h, chi, value, z, q_full):
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
    symmetric = jax.jit(lambda c: 0.5*(c + jnp.swapaxes(c, -1, -2)),
                        out_shardings=h.sharding)
    for s in imaginary.tolist():
        sym = symmetric(chi[s:s+1])
        w_even = v + solve_value(h, sym)[0]
        values = np.asarray(_census_scalars(chi[s], v + value[s], w_even), dtype=np.float64)
        row = dict(q_full=q_full, z_ry=[float(z[s].real), float(z[s].imag)],
                   **{k: float(x) for k, x in zip(names, values)})
        receipt.setdefault("tr_odd_census", []).append(row)
        if jax.process_index() == 0:
            print("TRBANK tr_odd_census " + " ".join(f"{k}={row[k]}" for k in row), flush=True)


def response_quadrature(wfns, meta, sample_plan, receipt, *, ordered=False, print_fn=print):
    """Plan independent frequency sums across hosts; broadcast only scalars."""
    import minimax
    from jax.experimental import multihost_utils
    z = bank_points(sample_plan)
    energy, f, u, _, _ = response_weights(wfns, meta)
    f, u, receipt["sample_activity"] = response_sample_weights(f, u)
    refs = np.array([energy[f != 0].max(), energy[u != 0].min()])
    lo, hi = refs[1]-refs[0], float(energy[u != 0].max()-energy[f != 0].min())
    session = getattr(meta, "shared_pole_response_rules", None)
    old = None if session is None else session.get("frequency")
    metallic = sample_plan["census"]["partial_at_mu"]
    reuse = (old is not None and old["lo"] <= lo and hi <= old["hi"]
             and old["metallic"] == metallic and np.array_equal(old["z"], z))
    if reuse:
        plan = old
    else:
        pad = 4./RYD_TO_EV if session is not None else 0.
        plan = dict(lo=lo-pad, hi=hi+pad, z=z, metallic=metallic,
            t=np.zeros((len(z), 2, minimax.RESPONSE_RULE_CAPACITY), complex),
            value=np.zeros((len(z), 2, minimax.RESPONSE_RULE_CAPACITY), complex),
            derivative=np.zeros((len(z), 2, minimax.RESPONSE_RULE_CAPACITY), complex),
            sampled_error=np.zeros((len(z), 2, 2)), counts=np.zeros((len(z), 2), np.int64))
        progress = LoopProgress(len(z), print_fn, title="response rule construction",
                                item_name="frequency", max_updates=len(z)).start()
        rank, workers = jax.process_index(), jax.process_count()
        for start in range(0, len(z), workers):
            assigned = start + rank
            status = np.zeros(1024, np.uint8)
            rule = None
            if assigned < len(z):
                try:
                    rule = minimax.response_frequency_rule(
                        plan["lo"], plan["hi"], z[assigned],
                        rel_tol=sample_plan["bank_rule_tolerance"],
                        previous=None if old is None else old["t"][assigned])
                except Exception as error:
                    message = str(error).encode()[:1023]
                    status[:len(message)] = np.frombuffer(message, dtype=np.uint8)
            for i in range(start, min(start + workers, len(z))):
                owner = i == assigned
                error = np.asarray(multihost_utils.broadcast_one_to_all(
                    status, is_source=owner))
                if error.any():
                    raise ValueError(bytes(error).rstrip(b"\0").decode(errors="replace"))
                for key in ("t", "value", "derivative", "sampled_error", "counts"):
                    payload = rule[key] if owner else plan[key][i]
                    plan[key][i] = np.asarray(multihost_utils.broadcast_one_to_all(
                        payload, is_source=owner))
                progress.step()
        progress.finish()
        if session is not None:
            session["frequency"] = plan
    receipt["rule_provider"] = "minimax frequency-specific complex times; sampled accuracy"
    receipt["rule"] = dict(interval_ry=[plan["lo"], plan["hi"]],
        counts=plan["counts"].tolist(), sampled_error=plan["sampled_error"].tolist(),
        reused=reuse)
    receipt["nodes"] = int(plan["counts"].sum())
    if jax.process_index() == 0:
        print_fn("Response quadrature: z(eV)   forward backward  sampled value/ds error; "
              + ("reused" if reuse else "constructed"), flush=True)
        for point, counts, errors in zip(z, plan["counts"], plan["sampled_error"]):
            print_fn(f"Response quadrature: {point*RYD_TO_EV:20.8g} {counts[0]:4d} {counts[1]:4d}  "
                  f"{errors[:,0].max():.2e} {errors[:,1].max():.2e}", flush=True)
    band_ranges = None
    if wfns.layout == "axis":
        from .greens_function_kernel import _phase_band_interval
        lo_band, hi_band = jax.device_get(_phase_band_interval(jnp.asarray(np.stack((f, u)))))
        # Enclose every parent's exact weight support. Fixed bounds share one
        # batched GEMM and remain safe when a complex-time phase underflows.
        band_ranges = tuple((int(lo.min()), int(hi.max())) for lo, hi in zip(lo_band, hi_band))
        if jax.process_index() == 0:
            print_fn(f"Response occupied/empty band intervals: {band_ranges} of {f.shape[-1]}")
    return dict(plan=plan, f=f, u=u, refs=refs, band_ranges=band_ranges)


def integrate_response_frequency(wfns, meta, mesh_xy, rules, *, q_ids, sample,
                                 execute, receipt, ordered=False, vertex=None):
    """Donated [value/ds,q,mu_X,nu_Y]; one Green/FFT scan per frequency."""
    plan, refs = rules["plan"], rules["refs"]
    times = plan["t"][sample]
    # Move the padded scalar-domain origin to the physical endpoint references.
    coefficients = -np.stack((plan["value"][sample], plan["derivative"][sample])) * np.exp(
        -(refs[1]-refs[0]-plan["lo"])*times)
    n = meta.mu_basis.n_packed if vertex is None else vertex[0][0].shape[2]
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
        q_ids=q_ids, n_outputs=2, pair_mode="direct", bank_carry=True,
        ordered=ordered, vertex=vertex, band_ranges=rules["band_ranges"])
    raw = jax.jit(lambda: jnp.zeros((2,len(q_ids),n,n),jnp.complex128),
        out_shardings=NamedSharding(mesh_xy,P(None,None,"x","y")))()
    weights = partial(stream_weights, parents=vertex is None)
    args = ((jnp.asarray(times.reshape(-1)), jnp.repeat(jnp.array([False, True]), times.shape[1])),
        jnp.asarray(coefficients.reshape(2,-1)), *fixed,
        weights(wfns, rules["f"], mesh_xy), weights(wfns, rules["u"], mesh_xy),
        jnp.asarray(refs), raw)
    raw = execute(kernel, args, "direct")
    receipt["correlation_count"] += int(plan["counts"][sample].sum())
    return raw


def produce_sample_bank(wfns, meta, config, *, mesh_xy, sym, sample_plan, bank_io,
                        vertex=None, contact=None, print_fn=print):
    """Stage A: integrate value and derivative together, one frequency at a time."""
    with timing.fenced_section('bank.setup', announce=True):
        header,qids,census = _bank_context(wfns,meta,sym,bank_io,mesh_xy)
        authenticate_sample_plan(sample_plan,header)
        z = bank_points(sample_plan)
        receipt = _receipt("samples",census,bank_io)
        # Time reversal measured broken: both particle-hole orientations keep
        # independent weights. The retarded stream already forms the partner as
        # conj in R space (the -q orientation); remote cells add the odd kernel.
        # ordered=True stores the physical orientation W_q = FT_q[W].
        ordered = vertex is not None or not bool(sym.trs_allowed)
        literal_mirrors = ordered
        if literal_mirrors and header.get("mirror_mode") != "literal_same_operator_v1":
            raise ValueError("GATE response_mirror_contract: ordered production requires a literal-mirror bank")
        if literal_mirrors:
            from symmetry_maps import q_negation_index
            negative = np.asarray(q_negation_index((int(meta.nkx), int(meta.nky), int(meta.nkz))), dtype=np.int64)
            mirror_qids = negative[qids]
            mirror_provenance = bank_io.get("mirror_operator_provenance")
            if mirror_provenance is None:
                mirror_provenance = dict(
                    coulomb=bank_io["coulomb"],
                    state_identity=bank_io["identity"],
                    operator="same original parent V and moment operator as Wc and M0..M3")
            receipt["mirror_contract"] = dict(header["mirror_contract"],
                mode=header["mirror_mode"], support_count=len(z),
                operator_provenance=mirror_provenance,
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
        execute = _bank_execution(meta, mesh_xy, receipt, config, photon=vertex is not None)
        ledger = meta.shared_pole_capacity
        ambient = ledger.live_stages
    from file_io.shared_pole_store import read_shared_pole_bank, shared_pole_bank_writer
    with timing.fenced_section('bank.window_geometry', announce=True,
                              label="shared-pole frequency rule construction"):
        solve_value, solve_slope, _, receipt["algebra"] = response_algebra(meta,config,
            mesh_xy=mesh_xy,n=n,photon=vertex is not None)
        rules = response_quadrature(wfns, meta, sample_plan, receipt, ordered=ordered, print_fn=print_fn)
    response_rows = panel_rows(0, len(qids))
    row_index = {q: i for i, q in enumerate(response_rows)}
    face_bytes = 16*n*n//mesh_xy.size
    caller_live = ambient
    if vertex is None:
        # V is frequency independent. Its all-P root bank is small compared
        # with either full-zone Green and reuses the existing batched solver.
        roots, inverse, _ = _coulomb_batch(meta, config, bank_io, mesh_xy,
                                         (0, len(qids)), execute)
        del inverse
        root_stage, _ = _reserve(meta, "coulomb_roots", len(qids)*face_bytes)
        ambient += (root_stage,)
    else:
        roots = bank_io["photon_v"]
    # The paired response accumulator is all-P sharded. Dense work and slab I/O
    # batch the irreducible parents of one frequency, with their own admission.
    fields = (("Wc", "dWc_ds"), ("Wc_mirror", "dWc_mirror_ds"))
    progress = LoopProgress(len(z), print_fn, title="response frequency integration",
                            item_name="frequency", max_updates=len(z)).start()
    for sample, point in enumerate(z):
        if np.asarray(header["sample_written"])[:,sample].all():
            progress.step()
            continue
        io_started = time.monotonic()
        with shared_pole_bank_writer(bank_io["path"], meta=meta,
                expected_identity=bank_io["identity"], mesh_xy=mesh_xy) as (bank_handle, header, write):
            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
            ledger.live_stages = ambient
            name, _ = _reserve(meta, "bank_outputs", 2*len(response_rows)*face_bytes)
            ledger.live_stages = ambient+(name,)
            raw = integrate_response_frequency(wfns, meta, mesh_xy, rules,
                q_ids=response_rows, sample=sample, execute=execute,
                receipt=receipt, ordered=ordered, vertex=vertex)
            for derivative in (False, True):
                field_indices = [2*m+int(derivative) for m in range(2 if literal_mirrors else 1)]
                if np.asarray(header["sample_written"])[:,sample,field_indices].all():
                    continue
                for mirror in range(2 if literal_mirrors else 1):
                    field = fields[mirror][int(derivative)]
                    pending = ~np.asarray(header["sample_written"], bool)[:,sample,2*mirror+int(derivative)]
                    # A fresh frequency is one q_irr slab. Preserve legacy partial
                    # masks by batching each contiguous unfinished span.
                    edges = np.flatnonzero(np.diff(np.r_[False,pending,False].astype(np.int8)))
                    for q0, q1 in edges.reshape(-1,2):
                        span = (int(q0), int(q1))
                        selected = (mirror_qids if mirror else qids)[q0:q1]
                        rows = np.asarray([row_index[int(q)] for q in selected])
                        chi = raw[int(derivative),rows]
                        if mirror:
                            chi = jnp.conj(chi)
                        h = roots[q0:q1]
                        constant = 0.
                        if vertex is not None:
                            constant = read_shared_pole_bank(bank_handle, span, meta=meta, header=header,
                                                            fields=("constant",))["constant"]
                        if derivative:
                            saved = read_shared_pole_bank(bank_handle, span, meta=meta, header=header,
                                sample_span=(sample,sample+1), fields=(fields[mirror][0],))
                            w = saved[fields[mirror][0]][:,0] + constant
                            value = execute(solve_slope, (h, w, chi), "sample_slope")
                            del saved, w
                        else:
                            value = execute(solve_value, (h,chi)+(() if vertex is None else (contact,)),
                                            "sample_dyson") - constant
                            for iq in range(q0,q1):
                                part = slice(iq-q0,iq-q0+1)
                                if vertex is not None and not mirror:
                                    _photon_sample_norms(receipt,value[part],iq,sample,bank_io["photon_layout"],mesh_xy)
                                elif vertex is None:
                                    _reciprocity_census(receipt,value[part],z[sample:sample+1],int(qids[iq]),iq,meta)
                                    if ordered and _self_negative(int(qids[iq]),meta):
                                        _tr_odd_census(receipt,solve_value,h[part],chi[part],value[part],z[sample:sample+1],int(qids[iq]))
                        io_started = time.monotonic()
                        write(q_span=span, sample_span=(sample,sample+1), **{field: value[:,None]})
                        receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                        del chi, value, h, constant
            receipt["batches"].append(dict(sample=sample))
            del raw
            progress.step()
            io_started = time.monotonic()
        receipt["seconds"]["io"] += time.monotonic()-io_started
    progress.finish()
    if jax.process_index() == 0:
        print_fn("Response quadrature: seconds " + " ".join(
            f"{key}={value:.3f}" for key, value in receipt["seconds"].items()), flush=True)
    del roots
    ledger.live_stages = caller_live
    receipt["stream_passes"] = len(receipt["batches"])
    receipt["batch_reason"] = "one frequency per stream; value and derivative share both Green/FFT products"
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
                        mu_bases, layout, occupation_state, sample_plan, bank_io, print_fn=print):
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
    execute = _bank_execution(meta, mesh_xy, receipt, config, photon=True)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    nq, n = len(sym.q_irr_full_idx), layout.packed_extent
    # Packed V, contact/reference/D and one family-read envelope, all XY tiled.
    vbytes = 16*(nq+4)*n*n//mesh_xy.size
    from common.wfn_layout import psi_specs
    nmu_spec, mun_spec = psi_specs(wfns.layout)
    nk, nb = wfns.enk.shape
    ns = wfns.green_parent.psi_mun.shape[1]
    carrier_shapes = ((nk, ns, n, nb), (nk, nb, ns, n))
    endpoint_bytes = 2 * sum(16 * int(np.prod(
        NamedSharding(mesh_xy, spec).shard_shape(shape)))
        for shape, spec in zip(carrier_shapes, (mun_spec, nmu_spec)))
    name, row = _reserve(meta, "photon_endpoints_and_V", vbytes + endpoint_bytes)
    ledger.live_stages = ambient+(name,)
    receipt["memory"].append(row)
    receipt["endpoint_memory"] = dict(
        retained_bytes_per_rank=endpoint_bytes,
        carrier_shapes=[list(shape) for shape in carrier_shapes],
        copies_per_orientation=2, layout=wfns.layout,
        scope="four prepared carriers; preparation intermediates and external native workspace not included")
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
        sym=sym, sample_plan=sample_plan, bank_io=bank, vertex=vertex, contact=contact, print_fn=print_fn)
    receipt["seconds"]["samples"] = time.monotonic()-before
    header = validate_shared_pole_bank(bank["path"], expected_identity=bank["identity"],
                                       mesh_xy=mesh_xy, require_complete=True)
    ledger.live_stages = ambient
    receipt["completion"] = True
    receipt["bank_header"] = header
    return _finish_receipt(receipt, meta, header, started)
