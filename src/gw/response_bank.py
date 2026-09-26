"""Physical response-bank algebra (DESIGN §3.1, DBANK D5–D6).

Charge operators have packed centroid-major endpoints ``mu * nspinor + spin``.
Photon operators use ``PhotonBasisLayout`` for charge and current endpoints.
Disk conversion belongs to the scratch writer. Dense products and solves
enter through ``distrib_la``.
"""
from dataclasses import dataclass
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
    from distrib_la import local_batch, matmul, plan

    lu = plan("solve_lu", mesh_xy, backend=backend, n=n,
              batched_route=route)
    face = NamedSharding(mesh_xy, P(None, "x", "y"))

    def program(inputs, outputs=1):
        def wrap(fn):
            if backend == "off":
                return local_batch(fn, mesh_xy)
            return jax.jit(fn, in_shardings=(face,) * inputs,
                out_shardings=face if outputs == 1 else (face,) * outputs)
        return wrap

    solve = jnp.linalg.solve if backend == "off" else lu.batched

    def mm(a, b):
        if backend == "off":
            return a @ b
        return matmul(a, b, mesh=mesh_xy, backend=backend,
                      batched_route=route)

    def congruence(h, a):
        return mm(mm(h, a), h)

    @program(2)
    def value(h, chi_raw):
        x = pref * congruence(h, chi_raw)
        identity = jnp.broadcast_to(jnp.eye(n, dtype=h.dtype), x.shape)
        e = solve(identity - x, identity.copy())
        return congruence(h, mm(x, e))

    def derivative(w, dchi_raw):
        # Full W on BOTH sides; no adjoint at complex frequency.
        return mm(mm(w, pref * dchi_raw), w)

    @program(3)
    def slope(h, wc, dchi_raw):
        return derivative(wc + mm(h, h), dchi_raw)

    @program(3, 2)
    def moments(h, a0, a1):
        # chi_scaled=A0/s+A1/s²; whitened Dyson coefficients are
        # B0=H A0 H, B1=H A1 H, S0=B0, S1=B1+B0².
        b0 = congruence(h, a0)
        b1 = congruence(h, a1)
        return (0.5 * congruence(h, b0),
                0.5 * congruence(h, b1 + mm(b0, b0)))

    if ordered:
        @program(5, 4)
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
        def infinity(v, contact):
            # chi(z)=chi_param(z)-contact, so W_inf=(I+V contact)^-1 V.
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), v.shape)
            return solve(identity + mm(v, jnp.broadcast_to(volume * contact, v.shape)), v.copy())

        @program(3)
        def value(v, chi_raw, contact):
            chi = pref * chi_raw - volume * contact
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), chi.shape)
            return solve(identity - mm(v, chi), v.copy()) - v

        @program(3)
        def slope(v, wc, dchi_raw):
            return derivative(wc + v, dchi_raw)

        @program(6, 5)
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


@dataclass(frozen=True, eq=False)
class PhotonEndpoints:
    """The four-current stream's endpoints: two families' raw-parent faces.

    ``families`` is the stream's static description (plans, layouts, order
    bases); ``mun``/``nmu`` the ``(charge, current)`` parent faces in each
    family's packed centroid order; ``enk`` the parents' energies.  No psi
    face is unfolded or vertex-applied: the stream contracts each family
    pair's Green on the parents and applies ``(1, alpha)`` on its spin
    indices (``w_isdf._get_chi_fractional_contour_kernel_face``).
    """
    families: object
    mun: tuple
    nmu: tuple
    enk: jax.Array

    @property
    def n(self) -> int:
        """The canonical photon extent of the stream's output rows."""
        return int(self.families.layout.packed_extent)

    @property
    def fixed(self) -> tuple:
        return (self.mun, self.nmu, self.enk)


def prepare_photon_carriers(wfns, wfns_transverse, mu_bases, *,
                            mesh_xy, layout):
    """Bind the charge and current families' raw-parent faces for the one response stream.

    Returns :class:`PhotonEndpoints`.  ``layout`` is the bank's canonical
    photon layout; the stream accumulates in the families' packed layout and
    converts its rows once per call (``photon_layout.PhotonFamilies``).
    Only the linear-size parent carriers are referenced; nothing is copied.
    """
    from .photon_layout import PhotonBasisLayout, PhotonFamilies
    from .w_isdf import _require_current_chi_endpoints

    left, right = _require_current_chi_endpoints(wfns, wfns_transverse)
    for name in ("irr_idx", "sym_idx", "k_parent_frac", "spin_action_full"):
        if not np.array_equal(getattr(left.plan, name), getattr(right.plan, name)):
            raise ValueError("GATE response_vertex: endpoint parent actions disagree")
    if not np.array_equal(np.asarray(wfns.enk), np.asarray(wfns_transverse.enk)):
        raise ValueError("GATE response_vertex: endpoint energies disagree")
    if not np.array_equal(np.asarray(wfns.occ), np.asarray(wfns_transverse.occ)):
        raise ValueError("GATE response_vertex: endpoint occupations disagree")
    for carrier, basis in zip((left, right), mu_bases):
        if int(carrier.plan.n_centroid_packed) != int(basis.n_packed):
            raise ValueError(
                "GATE response_vertex: a family's parent plan and centroid basis "
                f"disagree on the packed extent ({carrier.plan.n_centroid_packed} "
                f"vs {basis.n_packed})")
    packed = PhotonBasisLayout.from_centroid_extents(
        mu_bases[0].n_packed, mu_bases[1].n_packed, mesh_xy, packed=True)
    same_order = (all(basis.is_identity for basis in mu_bases)
                  and packed.carrier_extents == layout.carrier_extents)
    families = PhotonFamilies(
        plans=(left.plan, right.plan),
        packed_layout=layout if same_order else packed, layout=layout,
        bases=None if same_order else tuple(mu_bases))
    return PhotonEndpoints(families, (left.psi_mun, right.psi_mun),
                           (left.psi_nmu, right.psi_nmu), left.enk)


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
        live_rows = stream_weights(wfns, live, mesh_xy)
        raw = execute(kernel, (jnp.asarray(rule["t"]), jnp.asarray(rule["weights"]),
            *fixed, live_rows, live_rows, jnp.asarray([beta, mu])), "static_reference")
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
            stream_weights(wfns, np.stack((f, np.zeros_like(f))), mesh_xy),
            stream_weights(wfns, np.stack((u, np.zeros_like(u))), mesh_xy),
            jnp.asarray([lo, hi])), "static_reference")
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
        n_input = (int(meta.nk_tot) if vertex.families.n_parent is None
                   else vertex.families.n_parent)
        kernel = _response_stream_kernel(
            mesh_xy, (meta.nkx, meta.nky, meta.nkz), n_outputs,
            (n_input, int(wfns.slices.nb_full), vertex.n, 4),
            _ffi_key=ffi_dial_key(), layout=wfns.layout, selected_q=tuple(q_ids), pair_mode=pair_mode,
            bank_carry=bank_carry, ordered=True, vertex=vertex.families, band_ranges=band_ranges)
        return kernel, vertex.fixed
    if not charge_representation(meta):
        raise ValueError("GATE response_representation: want an authenticated "
                         "scalar, two-component, or four-component charge carrier")
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


def stream_weights(wfns, weights, mesh_xy):
    """Place small band weights and restrict to existing raw parents."""
    from common.collectives import replicate_to_mesh

    result = replicate_to_mesh(np.asarray(weights), mesh_xy)
    if wfns.green_parent is not None:
        result = wfns.green_parent.plan.parent_rows(result, axis=result.ndim-2)
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
            weight_f = stream_weights(wfns, f * erel**a, mesh_xy)
            weight_u = stream_weights(wfns, -1j * u * erel**b, mesh_xy)
            args = (jnp.asarray([0.]), jnp.asarray([[1. + 0j]]), *fixed,
                    weight_f.astype(jnp.complex128), weight_u,
                    jnp.asarray(reference))
            raw = execute(kernel, args, "moment_correlation")[:, 0]
            term = (_w_solve_pref_scalar(meta) * coefficient) * raw
            total = term if total is None else total + term
        totals.append(total)
    if not ordered:
        return (*totals, census)
    # Odd coefficients of 1/z and 1/z^3: sum (P - conj P_{-q}) Delta^m, m=0,2.
    # Real particle weights keep the retarded difference -i(X - conj X), so
    # the chi coefficient is i*raw. Four more correlations, same kernel.
    for moment_terms in (((1., 0, 0),), ((1., 2, 0), (-2., 1, 1), (1., 0, 2))):
        total = None
        for coefficient, a, b in moment_terms:
            weight_f = stream_weights(wfns, f * erel**a, mesh_xy)
            weight_u = stream_weights(wfns, u * erel**b, mesh_xy)
            args = (jnp.asarray([0.]), jnp.asarray([[1. + 0j]]), *fixed,
                    weight_f.astype(jnp.complex128), weight_u.astype(jnp.complex128),
                    jnp.asarray(reference))
            raw = execute(kernel, args, "moment_correlation")[:, 0]
            term = (1j * _w_solve_pref_scalar(meta) * coefficient) * raw
            total = term if total is None else total + term
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
        raise ValueError("GATE response_representation: want an authenticated "
                         "scalar, two-component, or four-component charge carrier")
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


_COMPILED = {}


def _compiled(kernel, args):
    """``kernel.lower(*args).compile()`` once per kernel and argument signature, per process.

    AOT lowering bypasses jit's executable cache, so an admission repeated at
    every sample of every SC map would recompile an unchanged program.  The
    kernels are module-cached builders, so their identity is stable; the
    caller still admits the executable's memory on every call.
    """
    key = (kernel, jax.tree.structure(args), tuple(
        (tuple(x.shape), str(x.dtype), getattr(x, "sharding", None))
        if hasattr(x, "shape") else x for x in jax.tree.leaves(args)))
    executable = _COMPILED.get(key)
    if executable is None:
        executable = _COMPILED[key] = kernel.lower(*args).compile()
    return executable


def _bank_execution(meta, mesh_xy, receipt, config, *, photon=False):
    """Compile and admit new dense work; stream outputs are reserved by batch."""
    def execute(kernel, args, stage):
        with timing.section('bank.compile.' + stage, announce=True):
            started = time.monotonic()
            executable = _compiled(kernel, args)
            receipt["seconds"]["compilation"] = (receipt["seconds"].get("compilation", 0.)
                + time.monotonic() - started)
        with timing.section('bank.admission.' + stage, announce=True):
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
        with timing.section('bank.dispatch.' + stage, announce=stream,
                            label=f"shared-pole bank {stage} dispatch"):
            started = time.monotonic()
            result = executable(*args)
            receipt["seconds"][stage+"_dispatch"] = receipt["seconds"].get(stage+"_dispatch", 0.) + time.monotonic()-started
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
    compiled = _compiled(_coulomb_pack(basis,mesh_xy), (abstract,))
    memory = compiled.memory_analysis()
    _reserve(meta, "coulomb_read_pack", memory.argument_size_in_bytes,
             memory.output_size_in_bytes + memory.temp_size_in_bytes)
    resource = bank_io["coulomb"]
    with SlabIO(resource["path"], mode="r", mesh=mesh_xy) as io:
        canonical = io.read_slab(resource["dataset"], shape=shape,
            offset=(q_span[0], 0, 0), partition_spec=spec)
        v = compiled(canonical)
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


def response_sample_weights(f, u):
    """Existing sample-only activity floor; exact moments retain every weight."""
    ft = np.where(np.abs(f) >= 1e-14, f, 0.0)
    ut = np.where(np.abs(u) >= 1e-14, u, 0.0)
    return ft, ut, dict(occupation_activity_floor=1e-14,
        discarded_f_mass=float(np.sum(np.abs(f-ft))),
        discarded_u_mass=float(np.sum(np.abs(u-ut))))


def response_occupation_envelope(energy, f, u, mu):
    """Bound |f_n u_m| by amplitude*min(1, exp(beta*(E_m-E_n)))."""
    bounds, amplitude = [], 1.
    for weight, offset in ((np.abs(f), energy-mu), (np.abs(u), mu-energy)):
        maximum = max(1., float(weight.max()))
        amplitude *= maximum
        tails = (offset > 0) & (weight != 0)
        if np.any(tails):
            bounds.append(float(np.min(np.log(maximum/weight[tails])/offset[tails])))
    return min(bounds) if bounds else 0., amplitude


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
                        vertex=None, contact=None, direct_head=None):
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
    n = meta.mu_basis.n_packed if vertex is None else vertex.n
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
            if direct_head is not None and q0 == 0:
                if int(qids[0]) != 0:
                    raise ValueError("GATE photon_direct_gamma_parent: Γ is not the first q parent")
                from .photon_direct_head import add_direct_gamma_field
                for name, coefficient in (("constant", direct_head["constant"]),
                        *((f"M{i}", direct_head["moments"][i]) for i in range(4))):
                    values[name] = add_direct_gamma_field(values[name], coefficient,
                        gamma_vectors=direct_head["gamma_vectors"],
                        layout=bank_io["photon_layout"], mesh=mesh_xy)
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


def _reciprocity_scalars(value):
    """``max|W - W^T| / max|W|`` for each sample of one parent's batch."""
    from common.collectives import xy_tile_mesh
    return _reciprocity_scalars_kernel(xy_tile_mesh(value))(value)


@lru_cache(maxsize=None)
def _reciprocity_scalars_kernel(mesh):
    from common.collectives import transpose_xy

    @jax.jit
    def kernel(value):
        defect = jnp.max(jnp.abs(value - transpose_xy(value, mesh)), axis=(-2, -1))
        scale = jnp.max(jnp.abs(value), axis=(-2, -1))
        return defect / jnp.where(scale > 0, scale, 1)
    return kernel


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


@lru_cache(maxsize=8)
def _symmetric_part(mesh, sharding):
    """(c + c^T)/2 of an all-mesh tile, once per mesh and output placement."""
    from common.collectives import transpose_xy
    return jax.jit(lambda c: 0.5*(c + transpose_xy(c, mesh)), out_shardings=sharding)


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
    from common.collectives import xy_tile_mesh
    symmetric = _symmetric_part(xy_tile_mesh(chi), h.sharding)
    for s in imaginary.tolist():
        sym = symmetric(chi[s:s+1])
        w_even = v + solve_value(h, sym)[0]
        values = np.asarray(_census_scalars(chi[s], v + value[s], w_even), dtype=np.float64)
        row = dict(q_full=q_full, z_ry=[float(z[s].real), float(z[s].imag)],
                   **{k: float(x) for k, x in zip(names, values)})
        receipt.setdefault("tr_odd_census", []).append(row)
        if jax.process_index() == 0:
            print("TRBANK tr_odd_census " + " ".join(f"{k}={row[k]}" for k in row), flush=True)


def response_groups(z, group_size):
    """Consecutive sample groups: imaginary axis by Im z, then the rest by Re z.

    Neighbouring samples need nearly the same exponentials, so a group costs
    about as many Green pairs as its hardest member.
    """
    order = sorted(range(len(z)), key=lambda i: (z[i].real != 0.,
                   z[i].real if z[i].real != 0. else z[i].imag))
    return [order[i:i+group_size] for i in range(0, len(order), group_size)]


def _gather_group_rules(requests, build):
    """Build requested groups round-robin across hosts; replicate the small rules."""
    import pickle
    from common.collectives import all_gather_processes, process_count, process_rank
    rank, world = int(process_rank()), int(process_count())
    local = []
    for index in range(rank, len(requests), world):
        try:
            local.append((index, build(requests[index]), None))
        except Exception as error:  # refusals cross hosts as data, then raise
            local.append((index, None, f"{type(error).__name__}: {error}"))
    if world == 1:
        shards = [local]
    else:
        payload = np.frombuffer(pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL), np.uint8)
        lengths = np.asarray(all_gather_processes(np.asarray(payload.size, np.int32)), np.int64).reshape(-1)
        padded = np.zeros(int(lengths.max()), np.uint8)
        padded[:payload.size] = payload
        gathered = np.asarray(all_gather_processes(padded), np.uint8)
        shards = [pickle.loads(np.ascontiguousarray(gathered[r, :int(n)]).tobytes())
                  for r, n in enumerate(lengths)]
    rows = sorted((row for shard in shards for row in shard), key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(requests))):
        raise RuntimeError("response rule construction did not gather every group")
    for _, _, error in rows:
        if error is not None:
            raise ValueError(error)
    return [rule for _, rules, _ in rows for rule in rules]


def response_support(wfns, meta, sample_plan, receipt, *, print_fn=print):
    """Occupation weights, transition interval, envelope and band support."""
    energy, f, u, _, _ = response_weights(wfns, meta)
    f, u, receipt["sample_activity"] = response_sample_weights(f, u)
    refs = np.array([energy[f != 0].max(), energy[u != 0].min()])
    lo, hi = refs[1]-refs[0], float(energy[u != 0].max()-energy[f != 0].min())
    mu = sample_plan["census"]["mu_ry"]
    decay_rate, amplitude = response_occupation_envelope(energy, f, u, mu)
    band_ranges = None
    if wfns.layout == "axis":
        from .greens_function_kernel import _phase_band_interval
        lo_band, hi_band = jax.device_get(_phase_band_interval(jnp.asarray(np.stack((f, u)))))
        # Enclose every parent's exact weight support. Fixed bounds share one
        # batched GEMM and remain safe when a complex-time phase underflows.
        band_ranges = tuple((int(lo.min()), int(hi.max())) for lo, hi in zip(lo_band, hi_band))
        if jax.process_index() == 0:
            print_fn(f"Response occupied/empty band intervals: {band_ranges} of {f.shape[-1]}")
    return dict(f=f, u=u, refs=refs, lo=lo, hi=hi, mu=mu, decay_rate=decay_rate,
                amplitude=amplitude, band_ranges=band_ranges)


def response_quadrature(meta, sample_plan, receipt, support, *, group_size, print_fn=print):
    """Plan shared complex-time rules for sample groups; replicate small rules.

    Each node of a group is ONE Green-pair evaluation that serves every
    member's forward and reverse orientation (see ``minimax.response_group_rules``).
    """
    import minimax
    from .sigma_box_plan import snap_outward
    z = bank_points(sample_plan)
    f, u, refs = support["f"], support["u"], support["refs"].copy()
    mu = support["mu"]
    # Rule and reuse decision are functions of grid cells, not of the exact
    # support: a round-off change otherwise flips reuse and moves SC eqp.
    lo, hi = snap_outward(support["lo"], 1., -1), snap_outward(support["hi"], 1., +1)
    decay_rate = snap_outward(support["decay_rate"], 1., -1)
    amplitude = snap_outward(support["amplitude"], 1., +1)
    session = getattr(meta, "shared_pole_response_rules", None)
    old = None if session is None else session.get("frequency")
    metallic = sample_plan["census"]["partial_at_mu"]
    reuse = (old is not None and old["lo"] <= lo and hi <= old["hi"]
             and old["decay_rate"] <= decay_rate and old["amplitude"] >= amplitude
             and old["metallic"] == metallic and np.array_equal(old["z"], z)
             and old["group_size"] == group_size)
    if reuse:
        plan = old
    else:
        pad = 4./RYD_TO_EV if session is not None else 0.
        plan = dict(lo=snap_outward(support["lo"]-pad, 1., -1),
                    hi=snap_outward(support["hi"]+pad, 1., +1), z=z, metallic=metallic,
                    group_size=group_size, decay_rate=decay_rate, amplitude=amplitude)
        plan["reference"] = 0. if decay_rate else plan["lo"]
        requests = response_groups(z, group_size)
        previous = [] if old is None else old["groups"]

        def build(members):
            local = {m: i for i, m in enumerate(members)}
            warm = [dict(rule, members=[local[m] for m in rule["members"]])
                    for rule in previous if set(rule["members"]) <= set(members)]
            rules = minimax.response_group_rules(
                plan["lo"], plan["hi"], z[members],
                rel_tol=sample_plan["bank_rule_tolerance"]/amplitude,
                decay_rate=decay_rate, previous=warm)
            return [dict(rule, members=[members[m] for m in rule["members"]]) for rule in rules]

        with timing.section("bank.rule_construction", announce=True):
            plan["groups"] = _gather_group_rules(requests, build)
        if session is not None:
            session["frequency"] = plan
    if plan["decay_rate"]:
        refs[:] = mu
    groups = plan["groups"]
    receipt["rule_provider"] = ("minimax shared-node group fit (forward t, reverse conj t from "
                                "one Green pair); sampled scalar accuracy")
    receipt["rule"] = dict(interval_ry=[plan["lo"], plan["hi"]], group_size=group_size,
        groups=[dict(members=[int(m) for m in g["members"]], count=int(g["count"]),
                     sampled_error=g["sampled_error"].tolist(),
                     coefficient_mass=g["coefficient_mass"].tolist()) for g in groups],
        decay_rate_ry_inv=plan["decay_rate"], occupation_amplitude=plan["amplitude"], reused=reuse)
    receipt["nodes"] = int(sum(g["count"] for g in groups))
    if jax.process_index() == 0:
        print_fn(f"Response interval (eV): [{plan['lo']*RYD_TO_EV:.8g}, {plan['hi']*RYD_TO_EV:.8g}]", flush=True)
        print_fn(f"Response quadrature: {len(z)} samples in {len(groups)} shared-node groups "
                 f"(planned size {group_size}); each node is one Green pair serving value, "
                 "ds and both orientations; " + ("reused" if reuse else "constructed"), flush=True)
        for g in groups:
            points = " ".join(f"{complex(z[m])*RYD_TO_EV:.4g}" for m in g["members"])
            print_fn(f"Response quadrature: nodes {g['count']:4d}  error {g['sampled_error'].max():.2e}  "
                     f"kappa {g['coefficient_mass'].max():.2e}  z(eV) {points}", flush=True)
        print_fn(f"Response quadrature: {receipt['nodes']} total Green-pair evaluations "
                 f"(value + derivative, forward + reverse)", flush=True)
    return dict(plan=plan, f=f, u=u, refs=refs, band_ranges=support["band_ranges"])


def _group_stream_arguments(rules, group):
    """Shared times and [forward/reverse, value/ds per member, node] weights."""
    plan, refs = rules["plan"], rules["refs"]
    times = group["t"]
    # Translate the scalar gauge to the physical endpoint references; the
    # reverse orientation is evaluated at conj(t).
    shift = -(refs[1]-refs[0]-plan["reference"])
    gauge = (np.exp(shift*times), np.exp(shift*np.conj(times)))
    members = len(group["members"])
    weights = np.zeros((2, 2*members, times.size), np.complex128)
    for side in (0, 1):
        weights[side, 0::2] = -group["value"][:, side]*gauge[side]
        weights[side, 1::2] = -group["derivative"][:, side]*gauge[side]
    return times, weights


@lru_cache(maxsize=16)
def _group_zeros(mesh_xy, shape):
    """The donated group carry [member, q, mu_X, nu_Y], one program per shape."""
    return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                   out_shardings=NamedSharding(mesh_xy, P(None, None, "x", "y")))


def integrate_response_group(wfns, meta, mesh_xy, rules, group, *, q_ids,
                             execute, receipt, ordered=False, vertex=None):
    """Donated [value/ds per member, q, mu_X, nu_Y]; one Green/FFT scan per group."""
    times, weights = _group_stream_arguments(rules, group)
    n = meta.mu_basis.n_packed if vertex is None else vertex.n
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
        q_ids=q_ids, n_outputs=weights.shape[1], pair_mode="direct", bank_carry=True,
        ordered=ordered, vertex=vertex, band_ranges=rules["band_ranges"])
    raw = _group_zeros(mesh_xy, (weights.shape[1],len(q_ids),n,n))()
    args = (jnp.asarray(times), jnp.asarray(weights), *fixed,
        stream_weights(wfns, rules["f"], mesh_xy), stream_weights(wfns, rules["u"], mesh_xy),
        jnp.asarray(rules["refs"]), raw)
    raw = execute(kernel, args, "direct")
    receipt["correlation_count"] += int(group["count"])
    return raw


def _stream_workspace(wfns, meta, mesh_xy, support, *, q_ids, n_outputs, ordered, vertex):
    """Compiled temporaries of the group stream; nothing is allocated.

    Lowered with the production shapes, so the first group's dispatch reuses
    this compilation when every sample fits in one group.
    """
    import minimax
    n = meta.mu_basis.n_packed if vertex is None else vertex.n
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy, q_ids=q_ids,
        n_outputs=n_outputs, pair_mode="direct", bank_carry=True, ordered=ordered,
        vertex=vertex, band_ranges=support["band_ranges"])
    capacity = minimax.RESPONSE_NODE_CAPACITY
    abstract = (jax.ShapeDtypeStruct((capacity,), jnp.complex128),
                jax.ShapeDtypeStruct((2, n_outputs, capacity), jnp.complex128),
                *fixed, stream_weights(wfns, support["f"], mesh_xy),
                stream_weights(wfns, support["u"], mesh_xy),
                jax.ShapeDtypeStruct((2,), jnp.float64),
                jax.ShapeDtypeStruct((n_outputs, len(q_ids), n, n), jnp.complex128,
                    sharding=NamedSharding(mesh_xy, P(None, None, "x", "y"))))
    with timing.section('bank.compile.direct', announce=True):
        memory = kernel.lower(*abstract).compile().memory_analysis()
    if memory is None:
        raise ValueError("GATE response_capacity: compiled memory unavailable")
    return int(memory.temp_size_in_bytes)


def response_group_size(meta, mesh_xy, *, n_samples, carry_per_sample, stream_workspace):
    """Largest sample group whose carry and stream workspace fit the ledger.

    One route and no dial: every sample in one group when it fits (symmetric
    decks), otherwise the largest group that does (about four on a
    two-component deck without q symmetry, where the carry is G/2 Green tiles).
    """
    ledger = meta.shared_pole_capacity
    fits = lambda g: ledger.preview(resident_bytes_per_rank=g*carry_per_sample,
        workspace_bytes_per_rank=stream_workspace,
        concurrent_with=ledger.live_stages)["device_budget_status"] == "PASS"
    if not fits(1):
        return 1   # admission refuses with the actual compiled bytes
    size = 1
    while size < n_samples and fits(size+1):
        size += 1
    return size


def produce_sample_bank(wfns, meta, config, *, mesh_xy, sym, sample_plan, bank_io,
                        vertex=None, contact=None, direct_head=None, print_fn=print):
    """Stage A: integrate value and derivative together, one frequency at a time."""
    with timing.section('bank.setup', announce=True):
        header,qids,census = _bank_context(wfns,meta,sym,bank_io,mesh_xy)
        authenticate_sample_plan(sample_plan,header)
        z = bank_points(sample_plan)
        receipt = _receipt("samples",census,bank_io)
        # Time reversal measured broken: both particle-hole orientations keep
        # independent weights. The retarded stream already forms the partner as
        # conj in R space (the -q orientation); remote cells add the odd kernel.
        # ordered=True stores the physical orientation W_q = FT_q[W].
        ordered = vertex is not None or not bool(sym.trs_allowed)
        if ordered != ("minus_q_partner" in header):
            raise ValueError("GATE response_minus_q_partner: got: a bank whose minus-q partner fields "
                             "disagree with the ordered route; want: W_q(-conj z) stored exactly when "
                             "time reversal is broken; fix: rebuild the bank")
        # W_q(-conj z) = conj(W_{-q}(z)) at each fitted line sample [p0, p1):
        # the exact -q rows of the same stream, conjugated, through parent q's
        # own V (and contact). An imaginary node is its own partner.
        p0, p1 = (int(v) for v in header["minus_q_partner"]["sample_span"]) if ordered else (0, 0)
        if p1 > p0:
            from symmetry_maps import q_negation_index
            negative = np.asarray(q_negation_index((int(meta.nkx), int(meta.nky), int(meta.nkz))), dtype=np.int64)
            partner_qids = negative[qids]
            partner_provenance = bank_io.get("minus_q_operator_provenance")
            if partner_provenance is None:
                partner_provenance = dict(
                    coulomb=bank_io["coulomb"],
                    state_identity=bank_io["identity"],
                    operator="same original parent V and moment operator as Wc and M0..M3")
            receipt["minus_q_partner"] = dict(header["minus_q_partner"],
                operator_provenance=partner_provenance,
                original_parent_count=len(qids),
                full_q_rows=len(set(qids.tolist()+partner_qids.tolist())),
                green_stream="union of exact q and minus-q output rows in the same response panel",
                dyson="original parent V/contact for both; same moment operator")

        def panel_rows(first, last):
            rows = qids[first:last].tolist()
            if p1 > p0:
                rows = list(dict.fromkeys(rows+partner_qids[first:last].tolist()))
            return tuple(rows)

        def committed(sample):
            done = np.asarray(header["sample_written"], bool)[:, sample].all()
            if p0 <= sample < p1:
                done = done and np.asarray(header["minus_q_written"], bool)[:, sample-p0].all()
            return bool(done)

        n = meta.mu_basis.n_packed if vertex is None else vertex.n
        if ordered:
            receipt["ordered"] = True
        started = time.monotonic()
        execute = _bank_execution(meta, mesh_xy, receipt, config, photon=vertex is not None)
        ledger = meta.shared_pole_capacity
        ambient = ledger.live_stages
    from file_io.shared_pole_store import read_shared_pole_bank, shared_pole_bank_writer
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
    carry_per_sample = 2*len(response_rows)*face_bytes
    with timing.section('bank.window_geometry', announce=True,
                              label="shared-pole frequency rule construction"):
        solve_value, solve_slope, _, receipt["algebra"] = response_algebra(meta,config,
            mesh_xy=mesh_xy,n=n,photon=vertex is not None)
        support = response_support(wfns, meta, sample_plan, receipt, print_fn=print_fn)
        ledger.live_stages = ambient
        # Price the stream once at "every sample in one group"; the compiled
        # temporaries do not grow with the group, only the donated carry does.
        workspace = _stream_workspace(wfns, meta, mesh_xy, support, q_ids=response_rows,
            n_outputs=2*len(z), ordered=ordered, vertex=vertex)
        group_size = response_group_size(meta, mesh_xy, n_samples=len(z),
            carry_per_sample=carry_per_sample, stream_workspace=workspace)
        rules = response_quadrature(meta, sample_plan, receipt, support,
                                    group_size=group_size, print_fn=print_fn)
    # The group accumulator is all-P sharded. Dense work and slab I/O batch
    # the irreducible parents of one frequency, with their own admission.
    fields = (("Wc", "dWc_ds"), ("Wc_minus_q", "dWc_minus_q_ds"))
    progress = LoopProgress(len(z), print_fn, title="response frequency integration",
                            item_name="frequency", max_updates=len(z)).start()
    for group in rules["plan"]["groups"]:
        members = [int(m) for m in group["members"]]
        if all(committed(m) for m in members):
            for _ in members:
                progress.step()
            continue
        ledger.live_stages = ambient
        name, _ = _reserve(meta, "bank_outputs", len(members)*carry_per_sample)
        ledger.live_stages = ambient+(name,)
        raw_group = integrate_response_group(wfns, meta, mesh_xy, rules, group,
            q_ids=response_rows, execute=execute, receipt=receipt,
            ordered=ordered, vertex=vertex)
        io_started = time.monotonic()
        # One collective writer transaction per group, not per sample.
        with shared_pole_bank_writer(bank_io["path"], meta=meta,
                expected_identity=bank_io["identity"], mesh_xy=mesh_xy) as (bank_handle, header, write):
            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
            for row, sample in enumerate(members):
                if committed(sample):
                    progress.step()
                    continue
                raw = raw_group[2*row:2*row+2]
                for partner in ((0, 1) if p0 <= sample < p1 else (0,)):
                    marked = (np.asarray(header["minus_q_written"], bool)[:,sample-p0] if partner
                              else np.asarray(header["sample_written"], bool)[:,sample])
                    # A fresh frequency is one q_irr slab. Partial restarts keep
                    # contiguous rows with identical value/slope masks together.
                    edges = np.r_[0, 1+np.flatnonzero(np.any(marked[1:] != marked[:-1], axis=1)), len(qids)]
                    for q0, q1 in zip(edges[:-1], edges[1:]):
                        need_value, need_slope = ~marked[q0]
                        if not (need_value or need_slope):
                            continue
                        span = (int(q0), int(q1))
                        selected = (partner_qids if partner else qids)[q0:q1]
                        rows = np.asarray([row_index[int(q)] for q in selected])
                        h = roots[q0:q1]
                        constant = 0.
                        if vertex is not None:
                            io_started = time.monotonic()
                            constant = read_shared_pole_bank(bank_handle, span, meta=meta, header=header,
                                                            fields=("constant",))["constant"]
                            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                        head_update = None
                        if direct_head is not None and q0 == 0:
                            from .photon_direct_head import add_direct_gamma_field
                            head_update = direct_head["constant"] + (
                                direct_head["Wc_minus_q"][sample] if partner
                                else direct_head["Wc"][sample])
                            def gamma_add(packed, coefficient):
                                return add_direct_gamma_field(packed, coefficient,
                                    gamma_vectors=direct_head["gamma_vectors"],
                                    layout=bank_io["photon_layout"], mesh=mesh_xy)
                        if need_value:
                            chi_value = raw[0,rows]
                            if partner:
                                chi_value = jnp.conj(chi_value)
                            value = execute(solve_value, (h,chi_value)+(() if vertex is None else (contact,)),
                                            "sample_dyson") - constant
                        else:
                            io_started = time.monotonic()
                            saved = read_shared_pole_bank(bank_handle, span, meta=meta, header=header,
                                sample_span=(sample,sample+1), fields=(fields[partner][0],))
                            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                            value = saved[fields[partner][0]][:,0]
                            del saved
                            if head_update is not None:
                                value = gamma_add(value, -head_update)
                        if need_slope:
                            chi = raw[1,rows]
                            if partner:
                                chi = jnp.conj(chi)
                            w = value if vertex is None else value+constant
                            slope = execute(solve_slope, (h, w, chi), "sample_slope")
                            if head_update is not None:
                                coefficient = (direct_head["dWc_minus_q_ds"][sample]
                                               if partner else direct_head["dWc_ds"][sample])
                                slope = gamma_add(slope, coefficient)
                            io_started = time.monotonic()
                            write(q_span=span, sample_span=(sample,sample+1), **{fields[partner][1]: slope[:,None]})
                            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                            del chi, w, slope
                        if need_value:
                            if head_update is not None:
                                value = gamma_add(value, head_update)
                            if vertex is not None and not partner:
                                _photon_sample_norms(receipt,value,q0,sample,bank_io["photon_layout"],mesh_xy)
                            for iq in range(q0,q1):
                                part = slice(iq-q0,iq-q0+1)
                                if vertex is None:
                                    _reciprocity_census(receipt,value[part],z[sample:sample+1],int(qids[iq]),iq,meta)
                                    if ordered and _self_negative(int(qids[iq]),meta):
                                        _tr_odd_census(receipt,solve_value,h[part],chi_value[part],value[part],z[sample:sample+1],int(qids[iq]))
                            io_started = time.monotonic()
                            write(q_span=span, sample_span=(sample,sample+1), **{fields[partner][0]: value[:,None]})
                            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                            del chi_value
                        del value, h, constant
                receipt["batches"].append(dict(sample=sample, group=members))
                del raw
                progress.step()
            io_started = time.monotonic()
        receipt["seconds"]["io"] += time.monotonic()-io_started
        del raw_group
    progress.finish()
    if jax.process_index() == 0:
        print_fn("Response quadrature: seconds " + " ".join(
            f"{key}={value:.3f}" for key, value in receipt["seconds"].items()), flush=True)
    del roots
    ledger.live_stages = caller_live
    receipt["stream_passes"] = len(rules["plan"]["groups"])
    receipt["batch_reason"] = ("one stream per sample group; each Green pair serves every member's value, "
                               "derivative and both orientations")
    receipt["io_scope"] = "I/O envelope includes device readiness, packing, and finite checks; not pure storage time"
    receipt["completion"] = all(committed(sample) for sample in range(len(z)))
    return _finish_receipt(receipt,meta,header,started)


def _photon_sample_norms(receipt, value, parent, first, layout, mesh_xy):
    """Record Frobenius norms of CC/CT/TC/TT without gathering operators."""
    from .photon_layout import photon_block_view
    norms = []
    for sector, pairs in (("CC", ((0, 0),)),
                          ("CT", tuple((0, b) for b in range(1, 4))),
                          ("TC", tuple((a, 0) for a in range(1, 4))),
                          ("TT", tuple((a, b) for a in range(1, 4) for b in range(1, 4)))):
        squared = sum(jnp.sum(jnp.abs(photon_block_view(value, layout, a, b, mesh_xy))**2,
                              axis=(-2, -1)) for a, b in pairs)
        norms.append(jnp.sqrt(squared))
    values = np.asarray(jnp.stack(norms, axis=-1))
    receipt.setdefault("sector_sample_norms", []).extend(
        dict(parent=int(parent)+i, first_sample=int(first),
             **{name: [float(v)] for name, v in zip(("CC","CT","TC","TT"), row)})
        for i, row in enumerate(values))


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
                        mu_bases, layout, occupation_state, sample_plan, bank_io,
                        wfn=None, photon_g0_vectors=None,
                        wfn_fingerprint_binding=None, photon_head_cache=None,
                        photon_head_rotation=None,
                        print_fn=print):
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

    started = time.monotonic()
    # The photon bank's head, fenced so it is no longer an unnamed band of
    # spole.bank (P2-S): scratch validation, the V-file digest, the census.
    with timing.section("bank.photon_setup"):
        header = validate_shared_pole_bank(bank_io["path"],
            expected_identity=bank_io["identity"], mesh_xy=mesh_xy)
        if header.get("photon_layout", {}).get("packed_extent") != layout.packed_extent:
            raise ValueError("GATE photon_bank_layout: scratch has a different packed photon layout")
        expected = [hashlib.sha256(np.asarray(b.canonical_indices, dtype="<i4").tobytes()).hexdigest()
                    for b in mu_bases]
        if expected != header["photon_centroid_digests"]:
            raise ValueError("GATE photon_bank_centroids: scratch/current endpoint identity differs")
        bank = dict(bank_io, photon_layout=layout)
        with timing.section("bank.coulomb_digest"):
            v_digest = resource_digest(bank["bispinor_v_q_path"])
        bank["coulomb"] = dict(path=str(bank["bispinor_v_q_path"]), basis="photon",
            q_irr_full_idx=np.asarray(sym.q_irr_full_idx).tolist(), sha256=v_digest)
        census = response_weights(wfns, meta)[-1]
        receipt = _receipt("photon", census, bank)
        execute = _bank_execution(meta, mesh_xy, receipt, config, photon=True)
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    nq, n = len(sym.q_irr_full_idx), layout.packed_extent
    # Packed V, contact/reference/D, all XY tiled.  The stream reads the two
    # families' resident raw-parent carriers; it prepares no endpoint copy.
    vbytes = 16*(nq+4)*n*n//mesh_xy.size
    name, row = _reserve(meta, "photon_endpoints_and_V", vbytes)
    ledger.live_stages = ambient+(name,)
    receipt["memory"].append(row)
    receipt["endpoint_memory"] = dict(
        retained_bytes_per_rank=0, layout=wfns.layout,
        scope="the stream reads the resident raw-parent carriers of both families")
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: preparing shared vertex endpoints and bare V", flush=True)
    with timing.section("bank.photon_endpoints"):
        vertex = prepare_photon_carriers(wfns, wfns_transverse, mu_bases,
                                         mesh_xy=mesh_xy, layout=layout)
        bank["photon_v"] = photon_bare_operator(wfns, wfns_transverse, meta,
            path=bank["bispinor_v_q_path"], mu_bases=mu_bases, layout=layout, mesh_xy=mesh_xy)
        from .gw_config import uses_direct_bispinor_shared_pole_head
        direct_gamma = None
        if uses_direct_bispinor_shared_pole_head(config):
            from .photon_direct_head import subtract_bare_tt_from_bank
            if photon_g0_vectors is None or len(photon_g0_vectors) != 4:
                raise ValueError("GATE photon_direct_gamma_vectors: expected four authenticated G=0 vectors")
            # V tiles cross from orbit-packed centroids into PhotonBasisLayout's
            # canonical family carriers above. Make the same basis conversion for
            # their G=0 vectors at this bank boundary, using its shared owner.
            direct_gamma = tuple(
                basis.unpack_axis(vector, -1)
                for basis, vector in zip((mu_bases[0],) + (mu_bases[1],) * 3,
                                         photon_g0_vectors))
            bank["photon_v"] = subtract_bare_tt_from_bank(
                bank["photon_v"], direct_gamma,
                layout=layout, mesh=mesh_xy, wfn=wfn, meta=meta)
    receipt["seconds"]["endpoints_and_V"] = time.monotonic()-before
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: centroid D and static grid reference", flush=True)
    with timing.section("bank.static_contact"):
        from file_io.shared_pole_store import (ResidentBankPayload, read_static_reference,
                                               write_bank_contact, write_static_reference)
        reference = bank.get("static_reference")
        initial = None
        if reference is None:
            grid, drude, contact = photon_static_contact(wfns, meta, mesh_xy=mesh_xy,
                layout=layout, vertex=vertex, occupation_state=occupation_state,
                sample_plan=sample_plan, execute=execute, receipt=receipt)
            # Later maps freeze this contact from its own small file, written on
            # both tiers beside (not inside) this map's scratch generation.
            generation = Path(bank["path"].label if isinstance(bank["path"], ResidentBankPayload)
                              else bank["path"]).parent
            reference = write_static_reference(
                generation.with_name(generation.name + "_photon_static_reference.h5"),
                dict(Pi_grid=grid, Drude=drude, TT_contact=contact),
                header=header, mesh_xy=mesh_xy)
        else:
            initial, (grid, drude, contact) = read_static_reference(reference, n=n, mesh_xy=mesh_xy)
            for key in ("photon_layout", "photon_centroid_digests"):
                if initial.get(key) != header[key]:
                    raise ValueError(f"GATE photon_static_reference: initial/current {key} differs")
        receipt["static_reference"] = reference
        bank["minus_q_operator_provenance"] = dict(coulomb=bank["coulomb"],
            static_reference=reference,
            static_reference_commit=(initial["commit"] if initial is not None else None),
            state_identity=bank["identity"],
            moments="M0,M1,M2,M3 and constant computed with the identical photon_v and contact arrays")
        # Persist the contact's two physically defined pieces as bank diagnostics;
        # the constructor consumes the separately committed constant, not these.
        write_bank_contact(bank["path"], dict(Pi_grid=grid, Drude=drude, TT_contact=contact),
                           mesh_xy=mesh_xy)
    receipt["seconds"]["static_contact"] = time.monotonic()-before
    del grid, drude
    direct_head = None
    with timing.section("bank.direct_head"):
        if uses_direct_bispinor_shared_pole_head(config):
            from .qsgw_head import read_authenticated_dipole_velocity, _pad_head_band_manifold
            from .photon_direct_head import build_direct_photon_head, packed_gamma_vectors
            cache = photon_head_cache if photon_head_cache is not None else {}
            velocity = cache.get("direct_photon_velocity")
            if velocity is None:
                host = read_authenticated_dipole_velocity(
                    os.path.join(config.input_dir, "dipole.h5"), wfn=wfn,
                    meta=meta, config=config,
                    wfn_fingerprint_binding=wfn_fingerprint_binding)
                nk, nb = int(host.shape[1]), int(host.shape[-1])
                empty = np.zeros((nk, nb), np.float64)
                velocity, _, _, _ = _pad_head_band_manifold(
                    host, empty, empty, empty, mesh=mesh_xy)
                cache["direct_photon_velocity"] = velocity
                del host
            if photon_head_rotation is not None:
                from .qsgw_head import rotate_velocity_active_to_qp
                velocity = rotate_velocity_active_to_qp(
                    velocity, photon_head_rotation, mesh=mesh_xy)
            direct_head = build_direct_photon_head(
                velocity, wfns, occupation_state, contact_packed=contact,
                photon_g0_vectors=direct_gamma, layout=layout,
                mesh=mesh_xy, meta=meta, wfn=wfn,
                frequencies_ry=bank_points(sample_plan), print_fn=print_fn)
            direct_head["gamma_vectors"] = packed_gamma_vectors(
                direct_gamma, layout, mesh_xy)
            receipt["direct_gamma"] = dict(
                approximation="first_order_dipole_current_fd",
                sectors="CC_CT_TC_TT", local_fields=False,
                samples="4x131072 Sobol exterior plus screened sphere",
                static_limit="Thomas-Fermi at z=0; dynamic Drude for Im(z)>0")
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: exact moments and W_infinity", flush=True)
    receipt["moments"] = compute_moment_bank(wfns, meta, config, mesh_xy=mesh_xy,
        sym=sym, bank_io=bank, vertex=vertex, contact=contact,
        direct_head=direct_head)
    receipt["seconds"]["moments"] = time.monotonic()-before
    before = time.monotonic()
    if jax.process_index() == 0:
        print("photon bank: ordered samples and derivatives", flush=True)
    receipt["samples"] = produce_sample_bank(wfns, meta, config, mesh_xy=mesh_xy,
        sym=sym, sample_plan=sample_plan, bank_io=bank, vertex=vertex,
        contact=contact, direct_head=direct_head, print_fn=print_fn)
    receipt["seconds"]["samples"] = time.monotonic()-before
    header = validate_shared_pole_bank(bank["path"], expected_identity=bank["identity"],
                                       mesh_xy=mesh_xy, require_complete=True)
    ledger.live_stages = ambient
    receipt["completion"] = True
    receipt["bank_header"] = header
    return _finish_receipt(receipt, meta, header, started)
