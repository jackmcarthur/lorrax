"""Per-state error budgets for MPA denominator-box rules.

For one external matrix element, the spatial MPA executor expands a selected
state/pole tuple as

``M_knp / d_knp`` with
``d_knp = omega - s_pole * (E_kn + Omega_p) + i eta``.

The Green-function selector contributes ``|u_kn|`` (one for the insulating
boolean path, or ``|f_kn|``/``|1-f_kn|`` for fractional occupations), the W
factor contributes the fitted residue ``B_p(q, mu, nu)``, and the spatial
kernel's explicit ``1/sqrt(N_k)`` combines with the orthonormal flat-k
convolution's ``1/sqrt(N_k)`` to give ``1/N_k``.  The final band projection
is bounded by Cauchy--Schwarz using
``wavefunction_bundle.projected_state_amplitude_envelope``.  Consequently a
cheap scalar bound on every tuple is

``|M_knp| <= |u_kn| A_kn |B_p(q,mu,nu)| / N_k``.

No phase or cancellation is claimed in this inequality.  For state row
``n`` let ``m[n,b]`` be the sum of these nonnegative tuple bounds in profile
bin ``b``, ``m[n] = sum_b m[n,b]``, and let ``B[n]`` count the bins in which
that state has mass.  If the rule error in bin ``b`` obeys

``tau[b] = min_n m[n] / (B[n] * m[n,b]) * eps / eta``,

then, for every state independently,

``sum_b m[n,b] tau[b] <= sum_b m[n] eps / (B[n] eta)``
``                         = m[n] eps / eta``.

The right side is exactly the bound supplied by the crossing box's uniform
rule, whose pointwise tolerance is ``eps / eta``.  Empty bins have zero
currency (unbounded tolerance).  Equivalently the minimax builder receives

``rho[b] = eta * max_n B[n] * m[n,b] / m[n]``

and keeps its dimensionless acceptance threshold ``eps``.  This is the
normalization missing from the first attempt: normalizing by one global mass
peak omitted both the per-state total ``m[n]`` and the ``1/B[n]`` division of
the state's error budget.  In particular, an arbitrarily light state is not
diluted by heavy states because its common occupation/projection factor
cancels between ``m[n,b]`` and ``m[n]``.

Residue magnitude is reduced on device into a bounded
``(a, log(1+gamma/eta))`` lattice for each exact pole selector.  The real
spacing is no coarser than ``2*pi*eta/log(1/eps)``, the shortest scale on
which the rule error varies; imaginary broadening is log-spaced.  A
state/frequency shift translates this pole histogram without changing its
total mass or occupied-bin count.  Production profiles are restricted to
crossing boxes, where the uniform comparator is the constant ``eps/eta``;
sign-definite windows retain their incumbent relative-error rule unchanged.
"""

from __future__ import annotations

import hashlib
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from common.collectives import all_gather_processes
from minimax import box_samples


_PROFILE_VERSION = 2
_PROFILE_REAL_BINS_MIN = 32
_PROFILE_IMAG_BINS_MIN = 24
_PROFILE_SHIFT_CHUNK = 32


def profile_grid(max_a, max_gamma, eta, eps):
    """Return pole-coordinate nodes at the rule-error resolution."""
    a_hi, gamma_hi, width = float(max_a), float(max_gamma), float(eta)
    if not (np.isfinite([a_hi, gamma_hi, width]).all()
            and a_hi > 0.0 and gamma_hi >= 0.0 and width > 0.0):
        raise ValueError("invalid Sigma tolerance-profile extent")
    # The shortest real-axis error scale occurs at Im(d)=eta.  The first
    # attempt used log(a), whose far-tail cells became much wider than this
    # scale even though Im(d) had not changed.  Keep a uniform physical-a
    # grid; gamma remains logarithmic because the scale grows with Im(d).
    real_step = 2.0 * np.pi * width / np.log(1.0 / float(eps))
    v_hi = np.log1p(gamma_hi / width)
    n_real = max(_PROFILE_REAL_BINS_MIN,
                 int(np.ceil(a_hi / max(real_step, 1.0e-12))) + 1)
    n_imag = max(_PROFILE_IMAG_BINS_MIN,
                 int(np.ceil(v_hi / max(
                     np.log1p(real_step / width), 1.0e-12))) + 1)
    return (np.linspace(0.0, a_hi, n_real, dtype=np.float64),
            np.linspace(0.0, max(v_hi, 1.0e-12), n_imag,
                        dtype=np.float64))


def _selected_by_regions(a, gamma, bounds):
    regions = jnp.asarray(bounds)

    def add_region(index, selected):
        row = jax.lax.dynamic_index_in_dim(
            regions, index, axis=0, keepdims=False)
        return selected | (
            (a > row[0]) & (a <= row[1])
            & (gamma >= row[2]) & (gamma > row[3])
            & (gamma < row[4]) & (gamma <= row[5]))

    return jax.lax.fori_loop(
        0, regions.shape[0], add_region,
        jnp.zeros(a.shape, dtype=bool))


def _axis_deposition(coordinate, nodes):
    scaled = coordinate / nodes[-1] * (nodes.size - 1)
    lower = jnp.clip(jnp.floor(scaled), 0, nodes.size - 2).astype(jnp.int32)
    fraction = jnp.clip(scaled - lower, 0.0, 1.0)
    return lower, fraction


@partial(jax.jit, static_argnames=("n_real", "n_imag"))
def _local_profile_histograms(Omega, B, selectors, u_nodes, v_nodes, eta,
                              *, n_real, n_imag):
    """Reduce one resident pole shard for every exact selector."""
    a = jnp.real(Omega).reshape(-1)
    gamma = (-jnp.imag(Omega)).reshape(-1)
    residue = jnp.abs(B).reshape(-1)
    finite = (jnp.isfinite(a) & jnp.isfinite(gamma)
              & jnp.isfinite(residue) & (residue > 0.0)
              & (a > 0.0) & (gamma >= 0.0))
    u = jnp.where(finite, a, 0.0)
    v = jnp.log1p(jnp.where(finite, gamma, 0.0) / eta)
    iu, fu = _axis_deposition(u, u_nodes)
    iv, fv = _axis_deposition(v, v_nodes)
    size = int(n_real) * int(n_imag)

    def one_selector(bounds):
        live = finite & _selected_by_regions(a, gamma, bounds)
        mass = jnp.where(live, residue, 0.0)

        def corner(row):
            upper_u, upper_v = row[0], row[1]
            wu = jnp.where(upper_u, fu, 1.0 - fu)
            wv = jnp.where(upper_v, fv, 1.0 - fv)
            index = (iu + upper_u) * n_imag + iv + upper_v
            return jnp.bincount(
                index, weights=mass * wu * wv, length=size)

        corners = jnp.asarray(((0, 0), (0, 1), (1, 0), (1, 1)),
                              dtype=jnp.int32)
        return jnp.sum(jax.vmap(corner)(corners), axis=0).reshape(
            n_real, n_imag)

    # Sequential mapping bounds the live spatial temporary independently of
    # selector count; only the small histograms are stacked.
    return jax.lax.map(one_selector, selectors)


def profile_histogram_batch(Omega, B, selectors, u_nodes, v_nodes, eta):
    """Return replicated selector histograms without gathering pole fields."""
    omega_shards = list(Omega.addressable_shards)
    residue_shards = list(B.addressable_shards)
    if len(omega_shards) != 1 or len(residue_shards) != 1:
        raise RuntimeError(
            "Sigma tolerance profile expects one pole shard per process")
    names = tuple(selectors)
    bounds = np.stack([np.asarray(selectors[name], np.float64)
                       for name in names])
    local = _local_profile_histograms(
        omega_shards[0].data, residue_shards[0].data,
        jnp.asarray(bounds), jnp.asarray(u_nodes), jnp.asarray(v_nodes),
        float(eta), n_real=len(u_nodes), n_imag=len(v_nodes))
    gathered = np.asarray(all_gather_processes(
        np.asarray(jax.device_get(local), np.float64)), np.float64)
    total = np.sum(gathered, axis=0)
    return {name: total[index] for index, name in enumerate(names)}


def _interpolate_histogram(histogram, a, gamma, u_nodes, v_nodes, eta):
    shape = np.broadcast_shapes(np.shape(a), np.shape(gamma))
    a = np.broadcast_to(np.asarray(a, np.float64), shape)
    gamma = np.broadcast_to(np.asarray(gamma, np.float64), shape)
    live = (np.isfinite(a) & np.isfinite(gamma)
            & (a > 0.0) & (gamma >= 0.0))
    u = np.maximum(a, 0.0)
    v = np.log1p(np.maximum(gamma, 0.0) / float(eta))

    def axis(values, nodes):
        scaled = values / nodes[-1] * (nodes.size - 1)
        lower = np.clip(np.floor(scaled).astype(np.int64),
                        0, nodes.size - 2)
        fraction = np.clip(scaled - lower, 0.0, 1.0)
        return lower, fraction

    iu, fu = axis(u, np.asarray(u_nodes))
    iv, fv = axis(v, np.asarray(v_nodes))
    hist = np.asarray(histogram, np.float64)
    value = ((1.0 - fu) * (1.0 - fv) * hist[iu, iv]
             + (1.0 - fu) * fv * hist[iu, iv + 1]
             + fu * (1.0 - fv) * hist[iu + 1, iv]
             + fu * fv * hist[iu + 1, iv + 1])
    in_grid = (u <= u_nodes[-1]) & (v <= v_nodes[-1])
    return np.where(live & in_grid, value, 0.0)


def _state_shift_envelope(states, state_masses, frequencies, pole_sign,
                          eta, eps):
    """Max-deposit state/frequency rows at the rule's real resolution."""
    energies = np.asarray(states, np.float64).reshape(-1)
    masses = np.asarray(state_masses, np.float64).reshape(-1)
    if energies.shape != masses.shape or not energies.size:
        raise ValueError("states and state_masses must be nonempty peers")
    if (not np.all(np.isfinite(energies))
            or not np.all(np.isfinite(masses)) or np.any(masses < 0.0)):
        raise ValueError(
            "state energies and masses must be finite and nonnegative")
    omega = np.asarray(frequencies, np.float64).reshape(-1)
    shifts = (float(pole_sign) * omega[:, None]
              - energies[None, :]).reshape(-1)
    shift_masses = np.broadcast_to(
        masses, (omega.size, masses.size)).reshape(-1)
    spacing = 2.0 * np.pi * float(eta) / np.log(1.0 / float(eps))
    origin = np.floor(float(np.min(shifts)) / spacing) * spacing
    indices = np.floor((shifts - origin) / spacing).astype(np.int64)
    envelope = np.zeros(int(np.max(indices)) + 1, np.float64)
    np.maximum.at(envelope, indices, shift_masses)
    occupied = envelope > 0.0
    centers = origin + (np.nonzero(occupied)[0] + 0.5) * spacing
    return centers, envelope[occupied], spacing


def per_state_bin_currency(mass_by_state_bin, eta):
    """Return ``rho_b`` implementing the per-state uniform-bound budget.

    Parameters
    ----------
    mass_by_state_bin : array_like, shape (n_state, n_bin)
        Nonnegative ``m[n,b]`` values made from the executor tuple bounds.
    eta : float
        Positive crossing-box broadening in the same energy units as ``d``.

    Returns
    -------
    rho : ndarray, shape (n_bin,)
        ``eta * max_n B[n] m[n,b] / m[n]``.  It is zero in bins with no
        state mass, representing an unbounded pointwise tolerance there.

    Notes
    -----
    Multiplying ``eps / rho[b]`` by each state's bin masses gives a bound no
    larger than ``m[n] * eps / eta``.  This function is the small algebraic
    owner used by the synthetic low-mass-state regression; the production
    translated-histogram path below evaluates the same expression without
    materializing a state-by-bin table.
    """
    mass = np.asarray(mass_by_state_bin, np.float64)
    width = float(eta)
    if mass.ndim != 2 or not mass.shape[0] or not mass.shape[1]:
        raise ValueError("mass_by_state_bin must be a nonempty matrix")
    if (not np.all(np.isfinite(mass)) or np.any(mass < 0.0)
            or not np.isfinite(width) or width <= 0.0):
        raise ValueError("profile masses must be finite and nonnegative; eta > 0")
    totals = np.sum(mass, axis=1)
    live = totals > 0.0
    if not np.any(live):
        raise ValueError("Sigma tolerance profile has no state mass")
    bins = np.count_nonzero(mass[live] > 0.0, axis=1)
    normalized = (bins[:, None] * mass[live]
                  / totals[live, None])
    return width * np.max(normalized, axis=0)


def _max_mass_from_shifts(denominators, shift_nodes, shift_masses,
                          pole_sign, pole_histogram, u_nodes, v_nodes, eta):
    """Evaluate the maximum binned state row without summing states."""
    d = np.asarray(denominators, np.complex128)
    flat = d.reshape(-1)
    shifts = np.asarray(shift_nodes, np.float64).reshape(-1)
    masses = np.asarray(shift_masses, np.float64).reshape(-1)
    maximum = np.zeros(flat.size, np.float64)
    gamma = flat.imag - float(eta)
    for lo in range(0, shifts.size, _PROFILE_SHIFT_CHUNK):
        hi = min(lo + _PROFILE_SHIFT_CHUNK, shifts.size)
        # d = omega - pole_sign*(E + a) + i*(gamma + eta), hence
        # a = (pole_sign*omega - E) - pole_sign*Re(d).
        a = (shifts[lo:hi, None]
             - float(pole_sign) * flat.real[None, :])
        pole_mass = _interpolate_histogram(
            pole_histogram, a, gamma[None, :], u_nodes, v_nodes, eta)
        maximum = np.maximum(
            maximum, np.max(masses[lo:hi, None] * pole_mass, axis=0))
    return maximum.reshape(d.shape)


def state_max_mass(denominators, states, state_masses, frequencies,
                   pole_sign, pole_histogram, u_nodes, v_nodes, eta,
                   eps=1.0e-4):
    """Evaluate ``max_state m_state(d)`` on the resolution-binned cloud."""
    shifts, masses, _spacing = _state_shift_envelope(
        states, state_masses, frequencies, pole_sign, eta, eps)
    return _max_mass_from_shifts(
        denominators, shifts, masses, pole_sign, pole_histogram,
        u_nodes, v_nodes, eta)


def build_tolerance_profile(box, kind, pole_sign, states, state_masses,
                            frequencies, pole_histogram, u_nodes, v_nodes,
                            eta, eps=1.0e-4):
    """Build the per-state-budget ``rho(d)`` and its cache identity."""
    if kind != "crossing":
        raise ValueError(
            "per-state tolerance profiles are defined only on crossing boxes")
    shifts, shift_masses, shift_spacing = _state_shift_envelope(
        states, state_masses, frequencies, pole_sign, eta, eps)
    pole_histogram = np.asarray(pole_histogram, np.float64)
    pole_total = float(np.sum(pole_histogram))
    occupied_bins = int(np.count_nonzero(pole_histogram > 0.0))
    if (not np.isfinite(pole_total) or pole_total <= 0.0
            or occupied_bins <= 0):
        raise ValueError("Sigma tolerance profile has no selected pole mass")
    canonical = box_samples(*box, per_unit=8.0, n_im=48)
    raw = _max_mass_from_shifts(
        canonical, shifts, shift_masses, pole_sign, pole_histogram,
        u_nodes, v_nodes, eta)
    # Each state/frequency row differs only by a real translation and a
    # constant |u_n| A_n/N_k factor.  Translation preserves pole_total and
    # occupied_bins; the state factor cancels in m[n,b]/m[n].  Divide raw by
    # the matching shift mass before taking the maximum.  _state_shift_
    # envelope max-deposits coincident shifts, so its retained mass is the
    # correct common factor at each represented row.
    nonzero_shift = shift_masses > 0.0
    if not np.any(nonzero_shift):
        raise ValueError("Sigma tolerance profile has no mass on rule cloud")

    unit_shift_masses = np.ones_like(shift_masses)

    def budget_ratio(d):
        local = _max_mass_from_shifts(
            d, shifts, unit_shift_masses, pole_sign, pole_histogram,
            u_nodes, v_nodes, eta)
        return np.maximum(
            occupied_bins * local / pole_total, 0.0)

    def rho(d):
        values = np.asarray(d, np.complex128)
        return float(eta) * budget_ratio(values)

    ratio = budget_ratio(canonical)
    # Quantize relative to the represented maximum while storing that scale
    # explicitly.  Unlike the first attempt, this is cache identity only; it
    # does not renormalize the physical currency.
    ratio_peak = float(np.max(ratio, initial=0.0))
    if not np.isfinite(ratio_peak) or ratio_peak <= 0.0:
        raise ValueError("Sigma tolerance profile has no mass on rule cloud")
    quantized = np.rint(
        np.minimum(ratio / ratio_peak, 1.0) * 65535.0).astype("<u2")
    digest = hashlib.sha256()
    digest.update(f"sigma-state-profile-v{_PROFILE_VERSION}".encode())
    digest.update(json.dumps({
        "box": [float(value) for value in box],
        "kind": str(kind), "pole_sign": float(pole_sign),
        "shape": list(quantized.shape),
        "ratio_peak": ratio_peak,
        "occupied_bins": occupied_bins,
        "pole_total": pole_total,
    }, sort_keys=True, separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(quantized).view(np.uint8))
    return rho, digest.hexdigest(), {
        "version": _PROFILE_VERSION,
        "mass_peak": float(np.max(raw, initial=0.0)),
        "pole_mass_total": pole_total,
        "bins_per_state": occupied_bins,
        "budget_ratio_peak": ratio_peak,
        "nonzero_fraction": float(np.count_nonzero(ratio) / ratio.size),
        "mean_budget_ratio": float(np.mean(ratio)),
        "quantization_levels": 65535,
        "state_aggregation": "max_Bn_mnb_over_massn",
        "state_count": int(np.asarray(states).size),
        "frequency_count": int(np.asarray(frequencies).size),
        "occupied_shift_bins": int(shifts.size),
        "shift_spacing_ry": float(shift_spacing),
    }


__all__ = [
    "build_tolerance_profile", "per_state_bin_currency", "profile_grid",
    "profile_histogram_batch", "state_max_mass",
]
