"""Physical response-bank algebra (DESIGN §3.1, DBANK D5–D6).

Operators have packed centroid-major endpoints ``mu * nspinor + spin``.
The public bank supports scalar representations; disk conversion belongs to
the scratch writer. Dense products and solves enter through ``distrib_la``.
"""
from functools import partial
import hashlib

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
        config if hasattr(config, "get") else {"linalg": config.linalg})
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
