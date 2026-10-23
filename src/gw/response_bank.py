"""Physical response-bank algebra (DESIGN §3.1, DBANK D5–D6).

Charge operators have packed centroid-major endpoints ``mu * nspinor + spin``.
Photon operators use ``PhotonBasisLayout`` for charge and current endpoints.
Disk conversion belongs to the scratch writer. Dense products and solves
enter through ``distrib_la``.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import time

import jax
from lxkit import ordered_norm, ordered_sum
from common import timing
from common.log import get_logger, message, record

log = get_logger(__name__)

from common.progress import LoopProgress
from common.units import RYD_TO_EV
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from .efermi import OCCUPATION_WEIGHT_FLOOR, band_in_occupation_window


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
    dyson, slope, moments, receipt
        Compiled-program callables (``__call__``, ``lower``, ``finish``) accepting
        complex128 ``[b,n,n]`` operators at ``P(None,'x','y')``, and the resolved
        backend/prefactor description.  The programs that hold the Dyson solve are
        keyed on its LU plan's decision, never on the room (:class:`_Decided`).
        ``dyson.value(H, chi_raw)`` returns Wc (Ry); ``slope(H, Wc, dchi_raw)``
        returns its s derivative (Ry^-1), restoring the bare operator internally (photon input is W-V). ``moments(H, A0, A1)`` takes already scaled
        bare-response expansion coefficients and returns M1/M3 (Ry^3/Ry^5).
        ``dyson.pair(layout)(dyson.place(H), chi_raw, dchi_raw)`` returns
        (Wc, dWc/ds) of one sample in one program, the same equations and bits
        as value then slope; ``place`` lays the charge roots out once for every
        sample, and ``layout='batch'`` (charge) leaves both outputs in the batch
        layout (local linalg) for a consumer of whole matrices per rank.  The
        photon pair takes ``(V, chi_raw, dchi_raw, contact, W_inf - V)`` on the
        face and returns (W - W_inf, its slope), the contact and the constant in
        the split routines' order; V is not held a second time (its batch copy
        would double the largest resident of the photon bank).
        Neither routine Hermitizes its inputs or outputs.
    """
    from .gw_config import linalg_resolution
    from .w_isdf import _w_solve_pref_scalar

    resolution = linalg_resolution({"linalg": dense_layout(meta, config, mesh_xy, n)})
    route = resolution.batched_route
    backend = "off" if resolution.layout == "local" else "distributed"
    pref = _w_solve_pref_scalar(meta)
    built = _response_programs(mesh_xy, n, backend, route, pref, ordered,
                               float(meta.cell_volume) if photon else None)
    room = _dyson_room(meta, mesh_xy, n) if backend == "distributed" else None
    dyson, moments, lu = built.bind(room)
    slope = built.slope
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
    return dyson, slope, moments, algebra


def _dyson_room(meta, mesh_xy, n):
    """Per-rank room of the Dyson solve, for distrib_la's size route (resolve guard 1b).

    The shared ledger's room beside its bound live reservations
    (``CapacityLedger.room_bytes_per_rank``) less the solve's own face operands (H, chi
    and W: 3 nq n^2 complex128 over the P ranks, the pricing of ``w_isdf._dyson_route``). A shape-and-ledger rule every rank shares: no
    exchange, no sync. None (the leg's library, never one_device) on one rank, with no
    ledger, or with the ledger's caller lifetimes unbound (unknown live bytes are not zero).
    """
    ledger = getattr(meta, "tangential_capacity", None)
    ranks = int(mesh_xy.size)
    if ledger is None or ranks <= 1:
        return None
    try:
        live = ledger.live_stages
    except ValueError:
        return None
    faces = 16 * int(n) * int(n) * 3 * int(meta.nk_tot) // ranks
    return max(0, ledger.room_bytes_per_rank(concurrent_with=live) - faces)


#: The solve-bearing programs per (builder, program, call signature, LU decision): never the room.
_VARIANTS: dict = {}


@dataclass(frozen=True)
class _Decided:
    """A solve-bearing response program keyed on the Dyson LU plan's decision.

    The room enters the HLO only through ``Plan.decision`` (backend, route, rounds).  Every
    solve in these bodies takes ``(b, n, n)`` operands at the face, ``b`` the leading
    extent of operand 1, so each call signature takes the variant cached on ``(builder,
    name, signature, decision)``, built with this room's plan on a miss and only ever traced
    on that signature: a room that moves without moving the decision lowers and compiles
    nothing again (COMPILEAUDIT 2026-10-05 rule 4).  A value (``builder`` is
    :func:`_response_programs`' key; ``lu``, that room's plan, is not compared), so equal
    programs compare equal.  The check-then-set needs no lock: ``compile_ahead`` lowers on
    the calling thread (INVARIANTS 21/25).
    """
    builder: tuple
    name: str
    room: object
    lu: object = field(compare=False, repr=False)

    def resolve(self, *args):
        """The variant program for this call signature."""
        x, n = args[1], self.builder[1]
        face = jax.ShapeDtypeStruct((int(x.shape[0]), n, n), x.dtype)
        key = (self.builder, self.name, tuple((tuple(a.shape), str(a.dtype)) for a in args),
               self.lu.decision(face, face))
        if key not in _VARIANTS:
            builder = _response_programs(*self.builder)
            _VARIANTS[key] = builder.program_for(self.name, self.lu.batched)
        return _VARIANTS[key]

    def __call__(self, *args):
        return self.resolve(*args)(*args)

    def lower(self, *args):
        return self.resolve(*args).lower(*args)

    def finish(self, args, result):
        return self.resolve(*args).finish(args, result)


@lru_cache(maxsize=None)
def _response_programs(mesh_xy, n, backend, route, pref, ordered, volume):
    """Reuse compiled algebra across SC maps without retaining state arrays.

    Returns the builder: ``slope``, the charge ``moments`` (no solve), and
    ``bind(room) -> (dyson, moments, lu)`` with the Dyson LU plan ``lu`` at ``room``
    (:func:`_dyson_room`: on every leg distrib_la takes one_device only where one whole
    matrix fits it, else the distributed library).  On the distributed backend the
    programs that hold the solve are :class:`_Decided`, keyed on that plan's decision and
    never on the room that decides it; on ``off`` they solve locally and are built once.
    """
    from types import SimpleNamespace
    from distrib_la import batch_layout, checked_program, local_batch, lu_workspace_bytes, matmul, plan

    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    key = (mesh_xy, n, backend, route, pref, ordered, volume)

    def program(inputs, outputs=1, resident=(), layout="face", donate=()):
        # ``donate``: operands the bank hands over (fresh χ rows and moment totals, read by no
        # one after the call), so their faces hold the temporaries and outputs. The bank runs
        # the whole-chain program (``_compiled``: ``call.lower``), the arguments' last reader.
        def wrap(fn):
            if backend == "off":
                return local_batch(fn, mesh_xy, resident=resident, out_layout=layout)
            # The checked LU solves return their status: no host callback.
            return checked_program(fn, mesh_xy, face if outputs == 1 else (face,) * outputs,
                                   in_shardings=(face,) * inputs, donate_argnums=donate)
        return wrap

    def mm(a, b):
        if backend == "off":
            return a @ b
        return matmul(a, b, mesh=mesh_xy, backend=backend,
                      batched_route=route)

    def congruence(h, a):
        return mm(mm(h, a), h)

    def derivative(w, dchi_raw):
        # Full W on BOTH sides; no adjoint at complex frequency.
        return mm(mm(w, pref * dchi_raw), w)

    def bodies(solve):
        """The solve-bearing bodies over ``solve``: name -> (body, inputs, outputs,
        resident, layout)."""
        def wc(h, chi_raw):
            # Wc = H X (I-X)^-1 H = H (I-X)^-1 X H: X is the right-hand side the LU
            # overwrites, so no identity right-hand side is held and no product follows it
            # (not H ((I-X)^-1 - I) H: that cancels to eps/||X|| as X -> 0).
            x = pref * congruence(h, chi_raw)
            identity = jnp.broadcast_to(jnp.eye(n, dtype=h.dtype), x.shape)
            return congruence(h, solve(identity - x, x))

        def pair(h, chi_raw, dchi_raw):
            value = wc(h, chi_raw)
            return value, derivative(value + mm(h, h), dchi_raw)

        if volume is None:
            # Local: one sample's value and slope in one program, the roots H laid out
            # once for every sample (no exchange), and a 'batch' consumer takes both outputs
            # without the exchange back. On the face, value then slope (two programs, the
            # same products): dχ/ds is not live beside the value's LU and first GEMMs.
            # The χ rows are donated (fresh rows of the carry).
            resident = (0,) if backend == "off" else ()
            out = {"value": (wc, 2, 1, (), "face", (1,))}
            for layout in (("face", "batch") if backend == "off" else ()):
                out["pair:" + layout] = (pair, 3, 2, resident, layout, (1, 2))
            return out

        def infinity(v, contact):
            # chi(z)=chi_param(z)-contact, so W_inf=(I+V contact)^-1 V.
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), v.shape)
            return solve(identity + mm(v, jnp.broadcast_to(volume * contact, v.shape)), v.copy())

        def w_minus_v(v, chi_raw, contact):
            chi = pref * chi_raw - volume * contact
            identity = jnp.broadcast_to(jnp.eye(n, dtype=v.dtype), chi.shape)
            return solve(identity - mm(v, chi), v.copy()) - v

        def photon_pair(v, chi_raw, dchi_raw, contact, constant):
            # The split routines' order: Wc = (W - V) - (W_inf - V), and the
            # slope reads W = (Wc + (W_inf - V)) + V.
            value = w_minus_v(v, chi_raw, contact) - constant
            return value, derivative((value + constant) + v, dchi_raw)

        def photon_moments(v, a0, a1, o0, o1, contact):
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

        return {"value": (w_minus_v, 3, 1, (), "face", ()),
                "pair:face": (photon_pair, 5, 2, (), "face", ()),
                "moments": (photon_moments, 6, 5, (), "face", ())}

    def program_for(name, solve):
        body, inputs, outputs, resident, layout, donate = bodies(solve)[name]
        return program(inputs, outputs, resident, layout, donate)(body)

    local = ({name: program_for(name, jnp.linalg.solve) for name in bodies(jnp.linalg.solve)}
             if backend == "off" else None)
    names = tuple(bodies(None))

    def bind(room):
        lu = plan("solve_lu", mesh_xy, backend=backend, n=n, batched_route=route, budget_bytes=room)
        # The provider beside the room: one factor face and its getrf/getrs workspace and pivots
        # (cuSOLVERMp: distrib_la.lu_workspace_bytes, measured panels, nrhs = n, getrs at the
        # default cap -- no cap is posted at bind; a dispatch books the cap it posts beside this
        # (_bank_execution); the Dyson bodies solve against I or V; no vendor query), else the getrs workspace cap
        # (batched_solve_lu_ffi: 2 GB, at most a face); over it, no route fits.
        if room is not None and lu.backend != "one_device":
            faces = 16 * int(n) * int(n) // int(mesh_xy.size)
            need = faces + (lu_workspace_bytes(mesh_xy, n, np.complex128) if lu.backend == "cusolvermp"
                            else min(2_000_000_000, faces))
            if need > room:
                raise RuntimeError(f"GATE response_dyson_capacity: the Dyson LU n={n} on the {lu.backend} "
                                   f"mesh needs {need / 1e9:.2f} GB/rank beside its operands (factor face "
                                   f"{faces / 1e9:.2f} + workspace), over the room {room / 1e9:.2f} GB/rank.")
        progs = local or {name: _Decided(key, name, room, lu) for name in names}
        pairs = {k.split(":", 1)[1]: v for k, v in progs.items() if k.startswith("pair:")}
        resident = backend == "off" and volume is None
        dyson = SimpleNamespace(value=progs["value"], pair=pairs.get,
                                place=(lambda h: batch_layout(h, mesh_xy)) if resident else (lambda h: h))
        return dyson, progs.get("moments", moments), lu

    if volume is None:
        @program(3, donate=(2,))
        def slope(h, wc, dchi_raw):
            return derivative(wc + mm(h, h), dchi_raw)

        @program(3, 2, donate=(1, 2))
        def moments(h, a0, a1):
            # chi_scaled=A0/s+A1/s²; whitened Dyson coefficients are
            # B0=H A0 H, B1=H A1 H, S0=B0, S1=B1+B0².
            b0 = congruence(h, a0)
            b1 = congruence(h, a1)
            return (0.5 * congruence(h, b0),
                    0.5 * congruence(h, b1 + mm(b0, b0)))
        if ordered:
            # chi_scaled = o0/z + a0/z^2 + o1/z^3 + a1/z^4 with no time-reversal or reality
            # assumption; X_k = H chi_k H and Wc = H X (I-X)^-1 H. Coefficients C1..C4 of
            # 1/z..1/z^4 keep every X1 cross term (o0 vanishes only in a complete basis), by
            # the Dyson recurrence C_k = X_k + sum_(i<k) X_i C_(k-i) (the photon moments'
            # form): 6 products, not 12 of the expanded series. M_k = C_(k+1)/2 (the
            # constructor's convention); with o0=o1=0 the even pair reduces to the incumbent
            # M1/M3 exactly. One program per step (:func:`ordered_moments`), each result in a
            # consumed operand's face: in one program XLA held the independent products side
            # by side (17 faces of temporaries at n 30000, P16, compiled on XLA:CPU).
            moments = SimpleNamespace(
                term=program(2, donate=(1,))(lambda h, chi: congruence(h, chi)),
                add_product=program(3)(lambda c, x, y: c + mm(x, y)),
                accumulate=program(3, donate=(0,))(lambda c, x, y: c + mm(x, y)),
                half_term=program(2, donate=(1,))(lambda h, c: 0.5 * congruence(h, c)))

    else:
        @program(3)
        def slope(v, wc, dchi_raw):
            return derivative(wc + v, dchi_raw)
        moments = None

    return SimpleNamespace(program_for=program_for, bind=bind, slope=slope, moments=moments)


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


@lru_cache(maxsize=8)
def _tt_only(mesh_xy, width):
    """Zero the charge rows/columns [:width] of each rank's face tile: one program per mesh and width."""
    from jax import shard_map
    return jax.jit(shard_map(lambda x: x.at[:, :width, :].set(0).at[:, :, :width].set(0),
        mesh=mesh_xy, in_specs=P(None, "x", "y"), out_specs=P(None, "x", "y"), check_vma=False))


def photon_static_contact(wfns, meta, *, mesh_xy, layout, vertex,
                          occupation_state, sample_plan, execute, receipt):
    r"""Build the TT Ward contact ``Pi_FD(0,0)``, with ``Pi_grid`` and centroid D, once per bank.

    The FD zero-Matsubara stream at q = 0 is the static limit
    ``lim_{q->0} Pi(q, 0)``: its diagonal transitions carry the Fermi-surface
    term ``-D`` (the same sign as the head's static ``-(alpha/2)^2 D``).
    The Ward contact is that limit, so a normal metal's static transverse
    response ``Pi(q, 0) - contact`` vanishes as q -> 0 and its dynamic limit
    ``Pi(0, z) - contact`` is ``+D``, the Drude weight.  ``Pi_grid = Pi_FD+D``
    (the interband part) and D are returned as diagnostics only.  Both are
    physical density responses (one factor ``1/Omega``). ``response_algebra``
    converts the contact to the convention of the stored ``V/ Omega`` at
    insertion.  For a step-occupation insulator the same stream uses its
    ordinary Laplace weights and D is exactly zero. It never assigns a gap to
    a metal.  All returned packed operators are ``[1,N,N]`` at
    ``P(None,'x','y')``.
    """
    from .static_gauge_response import (fermi_dirac_current_drude,
                                         photon_diagonal_current_faces)
    from .w_isdf import _w_solve_pref_scalar, matsubara_rule

    energy, f, u, _, census = response_weights(wfns, meta)
    live = (np.arange(energy.shape[1])[None, :]
            < census["band_stop"]-census["band_start"])
    live = np.broadcast_to(live, energy.shape)
    if occupation_state is not None:
        if not np.array_equal(np.asarray(occupation_state.f_kn), np.asarray(wfns.occ)):
            raise ValueError("GATE photon_contact_state: bank and FD occupations differ")
        beta, mu, _, rule = matsubara_rule(wfns, occupation_state, (0,),
            rel_tol=sample_plan["bank_rule_tolerance"])
        kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
            q_ids=(0,), n_outputs=1, pair_mode="kms_static", vertex=vertex)
        live_rows = stream_weights(wfns, live, mesh_xy)
        raw = execute(kernel, (jnp.asarray(rule["t"]), jnp.asarray(rule["weights"]),
            *fixed, live_rows, live_rows, jnp.asarray([beta, mu])), "static_reference",
            runtime_bytes=_stream_scratch(wfns, meta, mesh_xy, vertex))
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
            jnp.asarray([lo, hi])), "static_reference",
            runtime_bytes=_stream_scratch(wfns, meta, mesh_xy, vertex))
        drude = jnp.zeros((1, layout.packed_extent, layout.packed_extent), complex,
                          device=NamedSharding(mesh_xy, P(None, "x", "y")))
        receipt["static_rule"] = dict(provenance=quad.provenance, max_error=quad.max_error)
        count = len(quad.tau)
    # Remove charge rows/columns locally; only TT has a body contact.
    width = layout.carrier_extent(0)//layout.mesh_side
    pi_fd = _tt_only(mesh_xy, width)(raw[:, 0] * (_w_solve_pref_scalar(meta)/float(meta.cell_volume)))
    pi_grid = pi_fd + drude
    contact = pi_fd
    receipt["correlation_count"] += count
    receipt["contact"] = dict(equation="Pi_FD(0,0)=Pi_grid(0,0)-D", diagonal_reference="Pi_grid=Pi_FD+D",
        units="physical response density, 1/Omega", scope="built once for this bank state")
    return pi_grid, drude, contact


#: The run's stream programs by factory arguments (:func:`_response_stream_kernel`).
_STREAM_KERNELS: dict = {}


# Placement policy beside the existing program cache; no state arrays are held.
_STREAM_PROGRAMS: dict = {}


def _stream_variant(state, layout):
    variants = state["variants"]
    if layout not in variants:
        variants[layout] = state["build"](layout)
    return variants[layout]


def _resolve_stream_program(state, args, *, ahead=False):
    """Admit the actual stream before bank dispatch; axis gathers stay inline."""
    force = state["force"]
    if ahead:
        return _program_compiled(_stream_variant(state, force or "axis"), args, ahead=True)
    from lxkit import memory
    from common.gpu_utils import device_budget_bytes
    from runtime.aot_memory import agreed_chunk, compiled_new_bytes
    device_budget_bytes()
    for layout in ((force,) if force else ("axis", "face")):
        kernel = _stream_variant(state, layout)
        executable = _program_compiled(kernel, args)
        need = -agreed_chunk(-compiled_new_bytes(executable, extra=state["native"]))
        room = agreed_chunk(int(memory.room()))
        if layout == "axis" and need > room:
            if force:
                raise ValueError("GATE response_axis_room: forced axis stream exceeds compiled room")
            continue
        price = memory.priced(0, f"response {layout} summed bands, compiled stream + native scratch",
                              spread=need*state["mesh"].size, ranks=state["mesh"].size)
        memory.admit("Response sum-band operands", [(layout, price)], room)
        if state["last_layout"] != layout:
            log.info(f"Response summed bands: {layout} layout")
            state["last_layout"] = layout
        return kernel
    raise RuntimeError("Response sum-band layout selection produced no route")


def _response_stream_kernel(mesh_xy, kgrid, n_outputs, shape, *, _ffi_key,
                            _native_scratch=0, _force_sum_layout=None, **options):
    """Cache programs, never state arrays; window data remain dynamic inputs.

    A map's stream programs are one fixed set (the streamed χ bank's and the
    moment stream's segment programs, one per stream or four-current family
    pair, and the group programs) met again at every SC map, so every program
    is kept: a cache smaller than the set rebuilt every factory and re-lowered
    every program on every map.  A program binds its plan's placed kconv tables, which the kconv call
    caches (``w_isdf._CHARGE_KCONV``, ``_PHOTON_KCONV``) hold and the ledger
    reserves; when a cache evicts a plan, its programs and their executables are
    dropped with it (:func:`_release_evicted_programs`), so no table outlives
    its reservation.
    """
    from .w_isdf import _get_chi_fractional_contour_kernel_face
    key = (mesh_xy, kgrid, n_outputs, shape, _ffi_key, int(_native_scratch),
           _force_sum_layout, tuple(sorted(options.items())))
    kernel = _STREAM_KERNELS.get(key)
    if kernel is None:
        def build(layout):
            return _get_chi_fractional_contour_kernel_face(
                mesh_xy, kgrid, n_outputs, shape, sum_layout=layout, **options)
        row_stream = (options.get("vertex") is not None and bool(options.get("vertex"))) or (
            options.get("pair_mode") == "direct" and options.get("k_unfold_plan") is not None)
        initial = _force_sum_layout or "axis"
        kernel = _STREAM_KERNELS[key] = build(initial if row_stream else "axis")
        if row_stream:
            _STREAM_PROGRAMS[kernel] = dict(mesh=mesh_xy, build=build, native=int(_native_scratch),
                force=_force_sum_layout, variants={initial: kernel}, last_layout=None)
        _release_evicted_programs()
    return kernel


def _release_evicted_programs():
    """Drop the stream programs, and their executables, that bind an evicted plan's kconv tables."""
    from .w_isdf import placed_kconv_tables_live
    for key in [k for k, kernel in _STREAM_KERNELS.items()
                if not placed_kconv_tables_live(getattr(kernel, "tail", ()))]:
        dead = _STREAM_KERNELS.pop(key)
        state = _STREAM_PROGRAMS.pop(dead, None)
        programs = tuple(state["variants"].values()) if state else (dead,)
        for entry in [entry for entry in _COMPILED if entry[0] in programs]:
            del _COMPILED[entry]


def response_stream(wfns, meta, *, mesh_xy, q_ids, n_outputs,
                    pair_mode="retarded", bank_carry=False, ordered=False,
                    vertex=None, band_ranges=None, stream_pass=None, _force_sum_layout=None):
    """Bind the existing one-particle Green/FFT primitive to a q batch.

    Returns a jitted kernel and its fixed ψ/energy arguments. Caller supplies
    time, projections, final weights and energy reference. The output is
    ``[len(q_ids), n_outputs, mu_p, mu_p]`` with both endpoints sharded.
    ``ordered`` (time reversal measured broken) returns the physical
    orientation ``chi_q = FT_q[chi]`` that Sigma's contraction assumes.
    ``stream_pass`` runs one segment of the direct stream's row-pass engine
    into that segment's carry (the streamed bank, :func:`stream_segments`):
    the program serves every segment of its stream (the four-current stream:
    of segment ``stream_pass``'s family pair), the segment index its last
    argument.
    """
    from ffi import ffi_dial_key

    from file_io.tangential_store import charge_representation

    if vertex is not None:
        if pair_mode == "laplace":
            raise ValueError("GATE response_vertex: photon Laplace cells must retain odd rows")
        n_input = vertex.families.n_parent
        # The direct stream's row passes per family pair come from runtime.tiles
        # and the shapes (w_isdf._photon_pass_plans); its Greens read the active bands.
        if stream_pass is not None:
            from .w_isdf import photon_segments
            stream_pass = photon_segments(
                _photon_plans(wfns, meta, mesh_xy, vertex.families, len(q_ids)))[int(stream_pass)][0]
        kernel = _response_stream_kernel(
            mesh_xy, (meta.nkx, meta.nky, meta.nkz), n_outputs,
            (n_input, int(wfns.slices.nb_full), vertex.n, 4),
            _ffi_key=ffi_dial_key(), layout=wfns.layout, selected_q=tuple(q_ids), pair_mode=pair_mode,
            _native_scratch=_stream_scratch(wfns, meta, mesh_xy, vertex),
            _force_sum_layout=_force_sum_layout,
            bank_carry=bank_carry, ordered=True, vertex=vertex.families,
            band_ranges=band_ranges if pair_mode == "direct" else None,
            **({} if stream_pass is None else dict(stream_pass=int(stream_pass))))
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
        _native_scratch=_stream_scratch(wfns, meta, mesh_xy, None),
        _force_sum_layout=_force_sum_layout,
        pair_mode=pair_mode, bank_carry=bank_carry, ordered=ordered, band_ranges=band_ranges,
        # One segment program for every pass; the segment index is a runtime argument.
        **({} if stream_pass is None else dict(stream_pass=0)))
    return kernel, (source.psi_mun, source.psi_nmu, source.enk)


def stream_weights(wfns, weights, mesh_xy):
    """Place small band weights and restrict to existing raw parents."""
    from common.collectives import replicate_to_mesh

    result = replicate_to_mesh(np.asarray(weights), mesh_xy)
    if wfns.green_parent is not None:
        result = wfns.green_parent.plan.parent_rows(result, axis=result.ndim-2)
    return result


#: The exact moments' correlations ``(c, a, b)``: sum c * Corr(f E^a, u E^b) with
#: E = energy - reference, the binomial expansions of (E_u - E_f)^1 and ^3 (the
#: even totals A0, A1; imaginary particle weights) and of ^0 and ^2 (the odd
#: totals O0, O1 of an ordered bank; real particle weights).
EVEN_MOMENT_TERMS = (((-1., 1, 0), (1., 0, 1)),
                     ((-1., 3, 0), (3., 2, 1), (-3., 1, 2), (1., 0, 3)))
ODD_MOMENT_TERMS = (((1., 0, 0),), ((1., 2, 0), (-2., 1, 1), (1., 0, 2)))


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
    totals = []
    for moment_terms in EVEN_MOMENT_TERMS:
        total = None
        for coefficient, a, b in moment_terms:
            weight_f = stream_weights(wfns, f * erel**a, mesh_xy)
            weight_u = stream_weights(wfns, -1j * u * erel**b, mesh_xy)
            args = (jnp.asarray([0.]), jnp.asarray([[1. + 0j]]), *fixed,
                    weight_f.astype(jnp.complex128), weight_u,
                    jnp.asarray(reference))
            raw = execute(kernel, args, "moment_correlation",
                          runtime_bytes=_stream_scratch(wfns, meta, mesh_xy, vertex))[:, 0]
            term = (_w_solve_pref_scalar(meta) * coefficient) * raw
            total = term if total is None else total + term
        totals.append(total)
    if not ordered:
        return (*totals, census)
    # Odd coefficients of 1/z and 1/z^3: sum (P - conj P_{-q}) Delta^m, m=0,2.
    # Real particle weights keep the retarded difference -i(X - conj X), so
    # the chi coefficient is i*raw. Four more correlations, same kernel.
    for moment_terms in ODD_MOMENT_TERMS:
        total = None
        for coefficient, a, b in moment_terms:
            weight_f = stream_weights(wfns, f * erel**a, mesh_xy)
            weight_u = stream_weights(wfns, u * erel**b, mesh_xy)
            args = (jnp.asarray([0.]), jnp.asarray([[1. + 0j]]), *fixed,
                    weight_f.astype(jnp.complex128), weight_u.astype(jnp.complex128),
                    jnp.asarray(reference))
            raw = execute(kernel, args, "moment_correlation",
                          runtime_bytes=_stream_scratch(wfns, meta, mesh_xy, vertex))[:, 0]
            term = (1j * _w_solve_pref_scalar(meta) * coefficient) * raw
            total = term if total is None else total + term
        totals.append(total)
    return (*totals, census)


def _odd_moment_ratios(M0, M1, M2, M3):
    # Per parent of a [b, n, n] batch: ratios of the 1/z and 1/z^3 coefficients,
    # m0 = 2 M0 and m2 = 2 M2.  Norms in rank order on each stack's own layout
    # (lxkit.ordered_norm reads the concrete sharding).
    norm = lambda a: ordered_norm(a, axis=(-2, -1))
    return jnp.stack([2 * norm(M0) / norm(M1), 2 * norm(M2) / norm(M3)], axis=-1)


@lru_cache(maxsize=8)
def _anti_hermitian_ratios(mesh_xy):
    """Per-parent max|A - A^H| / max|A| of face-tiled [b, n, n] stacks, replicated.

    A^H through the transpose partner (``common.collectives.transpose_xy``):
    a swapaxes under this replicated output all-gathered every stack (2x2
    host mesh: 8 all-gathers, 18x one face stack of temp; 0 and 3.5x).
    """
    from common.collectives import transpose_xy

    def ratio(a):
        defect = jnp.max(jnp.abs(a - jnp.conj(transpose_xy(a, mesh_xy))), axis=(-2, -1))
        scale = jnp.max(jnp.abs(a), axis=(-2, -1))
        return jnp.where(scale > 0, defect / jnp.where(scale > 0, scale, 1), 0)
    return jax.jit(lambda *stacks: jnp.stack([ratio(a) for a in stacks]),
                   out_shardings=NamedSharding(mesh_xy, P()))


def _check_bare_hermitian(mesh_xy, q0, receipt, **moments):
    """GATE response_moment_hermiticity on one q batch's even bare moments.

    The constructor selects its infinity directions from the Hermitian part of
    M1 (``tangential_directions.infinity_directions``), so the Hermiticity of
    what M1 is formed from is checked here, at the checked eigh's round-off
    tolerance ``distrib_la.roundoff_tol(n)``. GPU streams give <= 4e-16 (CrI3
    24x24, n = 3328); the replicated ratios make every rank refuse alike.
    """
    from distrib_la import roundoff_tol
    ratios = np.asarray(_run_priced(_anti_hermitian_ratios(mesh_xy), *moments.values(),
                                    label="bare moment Hermiticity check"), dtype=np.float64)
    receipt["bare_anti_hermitian_max"] = max(float(np.nanmax(ratios, initial=0.0)),
                                             receipt.get("bare_anti_hermitian_max", 0.0))
    tol = roundoff_tol(next(iter(moments.values())).shape[-1])
    bad = ~(ratios <= tol)
    if bad.any():
        f, i = (int(v) for v in np.argwhere(bad)[0])
        raise ValueError(
            f"GATE response_moment_hermiticity: got: bare moment {list(moments)[f]} of q parent "
            f"{q0 + i} has max|A-A^H|/max|A| = {ratios[f, i]:.3e} ({int(bad.sum())} of {bad.size} "
            f"(moment, parent) rows above {tol:.1e}); want: <= {tol:.1e}, distrib_la.roundoff_tol "
            "(GPU streams give <= 4e-16); why: the exact moments are Hermitian and the constructor "
            "reads M1's Hermitian part, so an anti-Hermitian input would be dropped unseen")


def _record_odd_moments(q0, M0, M1, M2, M3, receipt):
    """Record a q batch's band-truncation diagnostic ||m0||/||M1||, ||m2||/||M3||.

    One program and one host read per batch (it was one of each per parent:
    1062 synchronous reads per SC map at Ni 20^3).
    """
    ratios = np.asarray(_odd_moment_ratios(M0, M1, M2, M3), dtype=np.float64)
    rows = [dict(q_parent=int(q0 + i), m0_over_M1_fro=float(r[0]), m2_over_M3_fro=float(r[1]))
            for i, r in enumerate(ratios)]
    receipt.setdefault("odd_moments", []).extend(rows)


def _bank_context(wfns, meta, sym, bank_io, mesh_xy):
    """Authenticate A/B's existing scratch transaction and physical state."""
    from file_io.tangential_store import (charge_representation,
                                           validate_tangential_bank)

    # The measured time-reversal verdict selects the orientation (callers
    # read sym.trs_allowed); only the operator representation refuses here.
    if not charge_representation(meta) and "photon_v" not in bank_io:
        raise ValueError("GATE response_representation: want an authenticated "
                         "scalar, two-component, or four-component charge carrier")
    header = validate_tangential_bank(bank_io["path"],
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


#: Bytes per digest chunk.  Each chunk is read and hashed by one process (round
#: robin), so a production resource (the Fe 20^3 V wedge, 54.6 GB) costs size / P
#: of reading per rank, not one rank reading all of it (~90 s at P64).
DIGEST_CHUNK_BYTES = 256 << 20


def _chunk_digests(path, size, rank, world):
    """This process's rows of the per-chunk SHA256 table (other rows zero)."""
    rows = np.zeros((max(1, -(-int(size) // DIGEST_CHUNK_BYTES)), 32), dtype=np.uint8)
    fd = os.open(path, os.O_RDONLY)
    try:
        for c in range(rank, rows.shape[0], world):
            chunk = os.pread(fd, DIGEST_CHUNK_BYTES, c * DIGEST_CHUNK_BYTES)
            rows[c] = np.frombuffer(hashlib.sha256(chunk).digest(), dtype=np.uint8)
    finally:
        os.close(fd)
    return rows


def resource_digest(path):
    """Content digest of one immutable resource, the same string on every rank.

    SHA256 of the file's per-chunk SHA256s (:data:`DIGEST_CHUNK_BYTES`, read
    round robin by the processes) and its size, so the value does not depend
    on the process count.  The producer stamps a resource with this and every
    consumer checks it with the same call, so the two cannot drift.  Rank 0
    decides whether this file generation was already hashed (a hard link of an
    SC map's re-staged V is not read again) and broadcasts the decision and
    the size, so every rank takes the same branch and leaves with the same
    string, which keeps a refusal from being rank-conditional (INVARIANTS 21).
    """
    from jax.experimental import multihost_utils
    from common.collectives import all_gather_processes

    path = Path(path)
    head = np.zeros(41, dtype=np.uint8)
    if jax.process_index() == 0:
        stat = path.stat()
        generation = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        head[1:9] = np.frombuffer(int(stat.st_size).to_bytes(8, "little"), dtype=np.uint8)
        hexdigest = _HASH_BY_GENERATION.get(generation)
        if hexdigest is not None:
            head[0] = 1
            head[9:] = np.frombuffer(bytes.fromhex(hexdigest), dtype=np.uint8)
    head = np.asarray(multihost_utils.broadcast_one_to_all(head), dtype=np.uint8)
    if head[0]:
        return bytes(head[9:]).hex()
    size = int.from_bytes(bytes(head[1:9]), "little")
    rows = _chunk_digests(str(path), size, int(jax.process_index()), int(jax.process_count()))
    rows = np.asarray(all_gather_processes(rows), dtype=np.uint8).max(axis=0)
    hexdigest = hashlib.sha256(rows.tobytes() + bytes(head[1:9])).hexdigest()
    if jax.process_index() == 0:
        stat = path.stat()
        _HASH_BY_GENERATION[(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)] = hexdigest
    return hexdigest


_HASH_BY_GENERATION: dict = {}


def authenticate_coulomb(bank_io, qids):
    """Authenticate the Coulomb resource against its fixed identity.

    A file resource (the photon V) by its content hash; the scalar on-device
    operator (``path`` None) by its live token and device digest.
    """
    resource = bank_io["coulomb"]
    if resource["basis"] not in ("canonical", "photon") or not np.array_equal(
            resource["q_irr_full_idx"], qids):
        raise ValueError("GATE response_coulomb_identity: wrong basis/q order")
    if resource.get("path") is None:
        values = _COULOMB_OPERATORS.get(resource.get("operator"))
        if values is None or operator_digest(values, None) != resource["sha256"]:
            raise ValueError("GATE response_coulomb_identity: the bare V operator is not "
                             "held or its values differ")
    elif resource_digest(resource["path"]) != resource["sha256"]:
        raise ValueError("GATE response_coulomb_identity: content hash differs")


#: The one bare-V operator of this process, by token.  The strong reference
#: keeps the token (an object id) unique for the operator's lifetime, which is
#: the run's: every SC map screens with the same V_q.
_COULOMB_OPERATORS: dict = {}
_OPERATOR_DIGESTS: dict = {}


def _operator_token(values):
    """Process-local identity of the bare V operator, stable across SC maps.

    :func:`_coulomb_batch` keys its held Coulomb roots on it, so a map that
    names the same V again reuses the first map's roots.
    """
    token = f"V@{id(values):x}"
    if _COULOMB_OPERATORS.get(token) is not values:
        _COULOMB_OPERATORS.clear()
        _OPERATOR_DIGESTS.clear()
        _COULOMB_OPERATORS[token] = values
    return token


@lru_cache(maxsize=4)
def _operator_digest_program(mesh_xy, shape):
    """Σ bits·(2g+1) mod 2^64 over the real and imaginary words at global flat index g."""
    from jax import shard_map
    nq, n, m = shape
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])

    def local(v):
        r0 = jax.lax.axis_index("x") * (n // px)
        c0 = jax.lax.axis_index("y") * (m // py)
        q = jnp.arange(v.shape[0], dtype=jnp.uint64)[:, None, None]
        r = (r0 + jnp.arange(v.shape[1])).astype(jnp.uint64)[None, :, None]
        c = (c0 + jnp.arange(v.shape[2])).astype(jnp.uint64)[None, None, :]
        g = 2 * ((q * jnp.uint64(n) + r) * jnp.uint64(m) + c)
        bits = jax.lax.bitcast_convert_type(jnp.stack([v.real, v.imag], -1), jnp.uint64)
        total = jnp.sum(bits[..., 0] * (2 * g + 1) + bits[..., 1] * (2 * g + 3), dtype=jnp.uint64)
        return jax.lax.psum(total, ("x", "y"))  # exact-reduce: uint64 checksum
    return jax.jit(shard_map(local, mesh=mesh_xy, in_specs=P(None, "x", "y"), out_specs=P(),
                             check_vma=False))


def operator_digest(values, mesh_xy):
    """A device digest of a face-tiled operator ``[nq, n, m]``: one 64-bit word per
    value position, summed with its global index, the same at every P.  Kept per
    operator object (the bare V is one object for the run)."""
    hit = _OPERATOR_DIGESTS.get(id(values))
    if hit is not None and hit[0] is values:
        return hit[1]
    mesh = values.sharding.mesh if mesh_xy is None else mesh_xy
    word = int(np.asarray(_operator_digest_program(mesh, tuple(values.shape))(values)))
    digest = hashlib.sha256(f"{tuple(values.shape)}:{word:016x}".encode()).hexdigest()
    _OPERATOR_DIGESTS[id(values)] = (values, digest)
    return digest


#: Whole n x n matrices a rank holds when it owns one parent of the dense
#: Dyson algebra: the operand rows (H, chi, dchi, Wc, dWc/ds) and the kernel's
#: working set (products, the LU factor and its inverse, W).
_WHOLE_PARENT_MATRICES = 13


def dense_layout(meta, config, mesh_xy, n):
    """The bank's dense algebra layout: the deck's ``linalg``, except that on a
    local deck a parent whose whole matrices do not fit takes the full mesh,
    one parent after another (``whole_parent_execution``, the line selection's
    rule). Every rank decides from the shapes and the shared ledger."""
    from .gw_config import linalg_resolution
    from .tangential_execution import whole_parent_execution
    layout = linalg_resolution(
        config if hasattr(config, "get") else {"linalg": config.backend.linalg}).layout
    if layout != "local":
        return layout
    whole = _WHOLE_PARENT_MATRICES * 16 * int(n) * int(n)
    execution, _, _ = whole_parent_execution(
        lambda e: (0, whole if e == "local" else -(-whole // int(mesh_xy.size))),
        ledger=meta.tangential_capacity, whole=int(n))
    return "local" if execution == "local" else "distributed"


def _reserve(meta, stage, resident, workspace=0, runs_in=None, book=True):
    """Reserve a uniquely named actual-batch footprint in the shared ledger, priced in the
    section that allocates it (``runs_in``; ``book`` False: a live set later phases' rooms
    hold, no stage's own price)."""
    ledger = meta.tangential_capacity
    name = f"{stage}:{len(ledger.entries)}"
    row = ledger.reserve(name, resident_bytes_per_rank=int(resident),
                        workspace_bytes_per_rank=int(workspace),
                        concurrent_with=ledger.live_stages, runs_in=runs_in, book=book)
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
    eig = workspace_bytes_per_rank(policy,"eigh",((batch,n,n),),np.complex128) if with_eigh else 0
    return dict(gemm=gemm,eigh=eig,total=gemm+eig,scope="actual-shape ISERV query; GEMM persistent plus concurrent eigh scratch")


_COMPILED = {}


def _compiled(kernel, args, *, ahead=False):
    """``kernel.lower(*args).compile()`` once per kernel and argument signature, per process.

    AOT lowering bypasses jit's executable cache, so an admission repeated at
    every sample of every SC map would recompile an unchanged program.  The
    kernels are module-cached builders, so their identity is stable; the
    caller still admits the executable's memory on every call.  An abstract
    leaf keys by its shape, dtype and sharding, as the array it stands for.
    The compile runs on a helper thread (``jax_compile_cache.compile_ahead``):
    ``ahead=True`` returns at once, from abstract ``args`` at the point the
    shapes become known, and the later call with the live arrays finds the
    executable or waits for it.
    """
    state = _STREAM_PROGRAMS.get(kernel)
    if state is not None:
        if ahead:
            return _resolve_stream_program(state, args, ahead=True)
        kernel = _resolve_stream_program(state, args)
    if isinstance(kernel, _Decided):
        kernel = kernel.resolve(*args)          # the variant its LU decision names
    return _program_compiled(kernel, args, ahead=ahead)


def _program_compiled(kernel, args, *, ahead=False):
    """The one executable cache, after any program placement decision."""
    from common.jax_compile_cache import compile_ahead
    key = (kernel, jax.tree.structure(args), tuple(
        (tuple(x.shape), str(x.dtype), getattr(x, "sharding", None))
        if hasattr(x, "shape") else x for x in jax.tree.leaves(args)))
    future = _COMPILED.get(key)
    if future is None:
        future = _COMPILED[key] = compile_ahead(kernel, *args)
    return future if ahead else future.result()


def _bank_execution(meta, mesh_xy, receipt, config, *, photon=False):
    """Compile and admit new dense work; stream outputs are reserved by batch."""
    def execute(kernel, args, stage, runtime_bytes=0):
        """``runtime_bytes``: what the executable draws at run time outside its
        buffer assignment (mathdx mode 11's split arm), admitted with it."""
        meta.tangential_capacity.at_phase()
        with timing.section('bank.compile.' + stage, announce=True):
            started = time.monotonic()
            executable = _compiled(kernel, args)
            receipt["seconds"]["compilation"] = (receipt["seconds"].get("compilation", 0.)
                + time.monotonic() - started)
        with timing.section('bank.admission.' + stage, announce=True):
            from lxkit import memory as allocator
            from runtime.aot_memory import compiled_new_bytes
            memory = executable.memory_analysis()
            if memory is None:
                raise ValueError("GATE response_capacity: compiled memory unavailable")
            stream = stage in ("real_time", "laplace", "windowed", "direct", "moment_correlation", "static_reference")
            # The price over what is live at the dispatch's entry: the program's temporaries +
            # outputs - donated aliases (+ cuFFT plans), what it draws outside its buffers
            # (``runtime_bytes``), a dense stage's provider workspace, and the cuSOLVERMp Dyson
            # LU's native size query at the getrs cap posted for this dispatch (the room the
            # ledger leaves; the query sizes the chunk the solve then takes).
            new = compiled_new_bytes(executable) + int(runtime_bytes)
            lu = None
            if stream:
                # The caller reserves the sole carry; the ledger admits the Green/FFT workspace.
                _, row = _reserve(meta, stage + "_temporaries", 0,
                                  memory.temp_size_in_bytes + int(runtime_bytes),
                                  runs_in="bank.dispatch." + stage, book=False)
            else:
                layout = dense_layout(meta, config, mesh_xy, args[0].shape[-1])
                native = response_dense_workspace(mesh_xy,args[0].shape[-1],args[0].shape[0],layout,
                    with_eigh=stage=="coulomb_sqrt")
                receipt.setdefault("native_queries",[]).append(dict(stage=stage,**native))
                new += native["total"]
                lu = getattr(kernel, "lu", None)
                lu = lu if lu is not None and lu.backend == "cusolvermp" else None
                if lu is not None:
                    from distrib_la import lu_workspace_query
                    ledger, n_lu = meta.tangential_capacity, int(args[0].shape[-1])
                    room = max(0, ledger.room_bytes_per_rank(ledger.live_stages) - memory.argument_size_in_bytes - new)
                    try:
                        # getrf, its pivots and one block column (cap 0: the native's narrowest
                        # chunk) sit beside the chunks: the cap is what the room leaves past them.
                        lu.chunk_room(n_lu, n_lu, 0)
                        cap = allocator.bucket(room - lu_workspace_query(mesh_xy, n_lu, n_lu, np.complex128, 0))
                        lu.chunk_room(n_lu, n_lu, cap)
                        new += lu_workspace_query(mesh_xy, n_lu, n_lu, np.complex128, cap)
                    except BaseException:
                        lu.chunk_room(n_lu, n_lu, None)
                        raise
                _, row = _reserve(meta, stage, memory.argument_size_in_bytes, new,
                                  runs_in="bank.dispatch." + stage, book=False)
            receipt["memory"].append(row)
            allocator.book(allocator.priced(new, f"bank {stage} program (temporaries + outputs - aliases, "
                "run-time, dense and LU workspace)", increment=True), stage="bank.dispatch." + stage)
            receipt["compiled"].append(dict(stage=stage,
                arguments=memory.argument_size_in_bytes, outputs=memory.output_size_in_bytes,
                temporaries=memory.temp_size_in_bytes,
                aliases=memory.alias_size_in_bytes,
                runtime=int(runtime_bytes),
                inherited_stream=False,
                stream_temporaries_admitted=stream))
        with timing.section('bank.dispatch.' + stage, announce=stream,
                            label=f"tangential bank {stage} dispatch"):
            started = time.monotonic()
            try:
                result = executable(*args)
                # A checked program's executable returns its status too (checked_program).
                if hasattr(kernel, "finish"):
                    result = kernel.finish(args, result)
            finally:
                if lu is not None:
                    lu.chunk_room(n_lu, n_lu, None)
            receipt["seconds"][stage+"_dispatch"] = receipt["seconds"].get(stage+"_dispatch", 0.) + time.monotonic()-started
        return result
    return execute


@lru_cache(maxsize=8)
def _coulomb_algebra(mesh_xy, n_packed, n_logical, layout, room=None):
    """One cached service plan for H=V^(1/2) and its supported inverse."""
    from distrib_la import checked_program, matmul, plan
    from .gw_config import linalg_resolution
    resolution = linalg_resolution({"linalg": layout})
    backend = "off" if resolution.layout == "local" else "distributed"     # the root's GEMMs
    if resolution.layout == "local":
        # place() decides the root's eigh in the room beside the bank's live set (the caller's
        # lxkit.memory.room, bucketed): whole matrices per rank only
        # where they fit (a whole n^2 per rank unpriced was 115 GB/rank at n 30000).
        if room is None:
            from common.gpu_utils import shared_device_room
            room = shared_device_room("tangential bank Coulomb root eigh")
        eig = plan("eigh", mesh_xy, n=n_packed, budget_bytes=room)
    else:
        eig = plan("eigh", mesh_xy, backend="distributed", n=n_packed,
                   batched_route=resolution.batched_route)
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    rep = NamedSharding(mesh_xy, P())

    # The checked eigh returns its status: no host callback.
    @partial(checked_program, mesh=mesh_xy, out_shardings=(face, face, rep, rep), in_shardings=(face,))
    def sqrt_v(value):
        tolerance = n_logical * np.finfo(np.float64).eps
        lam, vectors = eig.batched(value, cut=tolerance)     # checked on the supported columns
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


#: V^(1/2), V^(-1/2) and support ranks of the run's bare Coulomb operator, per
#: requested q span.  V is the same operator at every SC map (fixed ISDF basis
#: and q set), so the bank, moments and constructor of every map after the
#: first reuse the first map's roots instead of re-reading V and re-solving its
#: eigenproblems.  Keyed by the operator token of the resource
#: (``tangential_screening._coulomb_resource``), the span and the solve layout;
#: a new operator drops the old roots.  Held bytes per rank: 32 N_mu_packed^2 per
#: held q row / P; the bank, moments and constructor rounds share (0, N_q)
#: (a round slices its rows, one transient round slice), so
#: 32 N_q N_mu_packed^2 / P in all.
_COULOMB_ROOTS: dict = {}


def _coulomb_batch(meta, config, bank_io, mesh_xy, q_span, execute):
    """Read one authenticated canonical V batch; convert through its owner."""
    if "photon_v" in bank_io:
        value = bank_io["photon_v"][q_span[0]:q_span[1]]
        return value, None, [value.shape[-1]] * (q_span[1]-q_span[0])
    basis = meta.mu_basis
    resource = bank_io["coulomb"]
    layout = dense_layout(meta, config, mesh_xy, basis.n_packed)
    token = resource.get("operator")
    key = None if token is None else (token, (int(q_span[0]), int(q_span[1])),
                                      basis.n_packed, basis.n_logical, layout, mesh_xy)
    held = _COULOMB_ROOTS.get(key)
    if held is not None and not any(x.is_deleted() for x in held[:2]):
        return held
    if key is not None:
        # A constructor round asks for a sub-span of the parents the producer
        # already rooted (the held all-q entry): slice those rows instead of a
        # second read and eigensolve of the same V.
        for (k_token, (k_lo, k_hi), *k_rest), cover in _COULOMB_ROOTS.items():
            if (k_token == token and tuple(k_rest) == key[2:] and k_lo <= key[1][0]
                    and key[1][1] <= k_hi and not any(x.is_deleted() for x in cover[:2])):
                a, b = key[1][0] - k_lo, key[1][1] - k_lo
                rows, start = _span_rows(mesh_xy, b - a), _span_cursor(mesh_xy, (k_token, k_lo), a, b)
                return rows(cover[0], start), rows(cover[1], start), list(cover[2][a:b])
    h, hi, ranks = _coulomb_roots(meta, basis, resource, layout, mesh_xy, q_span, execute)
    if key is not None:
        for stale in [k for k in _COULOMB_ROOTS if k[0] != token]:
            del _COULOMB_ROOTS[stale]
        _COULOMB_ROOTS[key] = (h, hi, ranks)
    return h, hi, ranks


@lru_cache(maxsize=32)
def _span_rows(mesh_xy, width):
    """``width`` rows of a face-tiled parent stack from a placed start row, kept on the face.

    One program per width: the start is an operand, so a constructor round's span
    (a new start each round) runs the same program (a static slice compiled one per round).
    """
    return jax.jit(lambda x, start: jax.lax.dynamic_slice_in_dim(x, start, width, axis=0),
                   out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))


_SPAN_CURSOR = {}


def _span_cursor(mesh_xy, cover, a, b):
    """The start row ``a`` of a sub-span ``[a, b)`` of ``cover``, carried on device.

    Constructor rounds take a cover's parents in order: span 0 starts at a
    compiled zero and each next span at the previous start plus its width (one
    device add), so no start crosses from the host.  An out-of-order span
    takes :func:`_span_start` (placed once per value).
    """
    key = (mesh_xy, cover)
    held = _SPAN_CURSOR.get(key)
    if a == 0:
        start = _span_zero(mesh_xy)()
    elif held is not None and held[0] == a:
        start = held[1]
    else:
        start = _span_start(mesh_xy, a)
    while len(_SPAN_CURSOR) >= 4 and key not in _SPAN_CURSOR:
        _SPAN_CURSOR.pop(next(iter(_SPAN_CURSOR)))
    _SPAN_CURSOR[key] = (b, _span_advance(mesh_xy, b - a)(start))
    return start


@lru_cache(maxsize=8)
def _span_zero(mesh_xy):
    return jax.jit(lambda: jnp.zeros((), jnp.int32), out_shardings=NamedSharding(mesh_xy, P()))


@lru_cache(maxsize=32)
def _span_advance(mesh_xy, width):
    return jax.jit(lambda start: start + jnp.asarray(width, start.dtype),
                   out_shardings=NamedSharding(mesh_xy, P()))


@lru_cache(maxsize=None)
def _span_start(mesh_xy, start):
    """A span's start row, placed replicated once per value."""
    from common.collectives import device_put_process_local
    return device_put_process_local(np.int32(start), NamedSharding(mesh_xy, P()))


@lru_cache(maxsize=8)
def _coulomb_canonical_pack(basis, mesh_xy, q0, q1):
    """Parents [q0, q1) of the packed V through the canonical order and back (the
    carrier's padding zeroed exactly as a canonical copy would hold it)."""
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    return jax.jit(lambda v: basis.pack_operator(basis.unpack_operator(v[q0:q1], spec=P(None, "x", "y")),
                                                 spec=P(None, "x", "y")), out_shardings=face)


def _packed_v(mesh_xy, n_q, n_packed):
    """The abstract packed V of ``n_q`` parents, as ``_coulomb_roots`` hands it to H = V^(1/2)."""
    return jax.ShapeDtypeStruct((int(n_q), int(n_packed), int(n_packed)), jnp.complex128,
                                sharding=NamedSharding(mesh_xy, P(None, "x", "y")))


def compile_coulomb_roots_ahead(meta, config, mesh_xy, n_q):
    """Submit H = V^(1/2) of the parents (0, n_q) to a helper thread's compile.

    Its shape is the centroid basis's and the q wedge's, known before the
    zeta fit, and it is the bank's largest compile; the bank's
    ``_coulomb_roots`` then finds it compiled. A deck whose ``linalg`` is
    local is skipped: there the bank's ledger picks the layout.
    """
    from .gw_config import linalg_resolution
    basis = meta.mu_basis
    if linalg_resolution({"linalg": config.backend.linalg}).layout == "local":
        return
    layout = dense_layout(meta, config, mesh_xy, basis.n_packed)
    _compiled(_coulomb_algebra(mesh_xy, basis.n_packed, basis.n_logical, layout),
              (_packed_v(mesh_xy, n_q, basis.n_packed),), ahead=True)


def _coulomb_roots(meta, basis, resource, layout, mesh_xy, q_span, execute):
    """The PSD roots of the q span of V through the service plan.

    The scalar operator is held on the devices (``path`` None) and its parents
    are taken from it; a file resource is read in its canonical order.
    """
    from file_io.slab_io import SlabIO
    shape = (q_span[1]-q_span[0], basis.n_canonical, basis.n_canonical)
    spec = P(None, "x", "y")
    # A span the W-model setup did not submit (compile_coulomb_roots_ahead)
    # compiles beside the read and the pack below instead of after them.
    # Under linalg = local the root's eigh is placed in the room beside what the bank holds now
    # (bucketed: the program is cached on it); every rank enters the agreement.
    from .gw_config import linalg_resolution
    from common.gpu_utils import shared_device_room
    room = (shared_device_room("tangential bank Coulomb root eigh")
            if linalg_resolution({"linalg": layout}).layout == "local" else None)
    kernel = _coulomb_algebra(mesh_xy, basis.n_packed, basis.n_logical, layout, room)
    _compiled(kernel, (_packed_v(mesh_xy, shape[0], basis.n_packed),), ahead=True)
    if resource.get("path") is None:
        values = _COULOMB_OPERATORS[resource["operator"]]
        program = _coulomb_canonical_pack(basis, mesh_xy, int(q_span[0]), int(q_span[1]))
        memory = _compiled(program, (values,)).memory_analysis()
        _reserve(meta, "coulomb_read_pack", memory.argument_size_in_bytes,
                 memory.output_size_in_bytes + memory.temp_size_in_bytes)
        v = _compiled(program, (values,))(values)
    else:
        abstract = jax.ShapeDtypeStruct(shape, jnp.complex128,
                                       sharding=NamedSharding(mesh_xy, spec))
        compiled = _compiled(_coulomb_pack(basis,mesh_xy), (abstract,))
        memory = compiled.memory_analysis()
        _reserve(meta, "coulomb_read_pack", memory.argument_size_in_bytes,
                 memory.output_size_in_bytes + memory.temp_size_in_bytes)
        with SlabIO(resource["path"], mode="r", mesh=mesh_xy) as io:
            canonical = io.read_slab(resource["dataset"], shape=shape,
                offset=(q_span[0], 0, 0), partition_spec=spec)
            v = compiled(canonical)
        del canonical
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
    """The sample weights on the one branch support; exact moments retain every weight.

    ``gw.efermi.band_in_occupation_window`` is the support Sigma and the
    fractional chi0 use, so the bank samples the same states.
    """
    ft = np.where(band_in_occupation_window(f), f, 0.0)
    ut = np.where(band_in_occupation_window(u), u, 0.0)
    return ft, ut, dict(occupation_activity_floor=OCCUPATION_WEIGHT_FLOOR,
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
    _resolve_reciprocity(receipt)
    receipt["seconds"]["total"] = time.monotonic()-started
    receipt["bank_complete"] = bool(header["complete"])
    receipt["capacity"] = meta.tangential_capacity.receipt()
    stats = [d.memory_stats() for d in jax.local_devices()]
    local_peak = max((v.get("peak_bytes_in_use",0) for v in stats if v),default=0)
    from jax.experimental import multihost_utils
    peak = int(np.max(multihost_utils.process_allgather(np.asarray(local_peak,dtype=np.int64))))
    receipt["peak_bytes"] = peak or None
    receipt["peak_reason"] = "maximum JAX allocator high-water bytes across ranks; inherited arrays included, external native allocations excluded"
    receipt["plan_hash"] = header["bank_plan_digest"]
    return receipt


def parent_span(n_q, *, layer, parent_bytes, fixed_bytes=0):
    """Parents per round of the bank's per-parent dense stages, sized in the layout they run in.

    A round holds ``fixed_bytes`` plus ``parent_bytes`` per parent per rank (its
    face stacks).  ``layer`` is how many parents fill the mesh once: the mesh
    size P under ``linalg = local``, whose Dyson solves, line selection and
    moment Dyson run one whole matrix per rank (``distrib_la.local_batch``), so
    a round narrower than P leaves ranks idle; 1 on the full mesh.  The round
    takes the most layers whose bytes fit the fixed tile (``runtime.tiles``),
    at least one: P parents, ``P·parent_bytes`` per rank (about 6·16·n² for a
    sample's value and slope with its unpack and prefetch, 6.6 GB at CrI3
    24×24).  The rounds are balanced, ``ceil(n_q / rounds)`` parents each.  The
    rounds are independent, so the width moves no number.
    """
    from runtime.tiles import TILE_BYTES
    n_q, layer = int(n_q), max(1, int(layer))
    layers = max(1, (int(TILE_BYTES) - int(fixed_bytes)) // max(1, layer * int(parent_bytes)))
    width = max(1, min(n_q, layers * layer))
    return -(-n_q // -(-n_q // width))


def _parent_layer(receipt, mesh_xy):
    """Parents per mesh layer of the bank's dense algebra: P when it runs whole matrices per rank."""
    return int(mesh_xy.size) if receipt["algebra"]["linalg"] == "local" else 1


@contextmanager
def _moment_phase(receipt, name):
    """One stage of a moment batch: a timing section, its wall summed in ``receipt["seconds"]``."""
    started = time.monotonic()
    with timing.section("bank.moment_" + name):
        yield
    receipt["seconds"][name] = receipt["seconds"].get(name, 0.0) + time.monotonic() - started


@lru_cache(maxsize=8)
def _moment_batch_major(mesh_xy, n_out, width, n_batch):
    """Pass carry ``[n_out, q, m, n]`` → ``[n_batch * n_out, width, m, n]``, batch-major
    (q padded with zeros to ``n_batch * width``), so a q batch's totals are one bank run."""
    spec = NamedSharding(mesh_xy, P(None, None, "x", "y"))

    def order(carry):
        pad = n_batch * width - carry.shape[1]
        carry = jnp.pad(carry, ((0, 0), (0, pad), (0, 0), (0, 0)))
        carry = carry.reshape((n_out, n_batch, width) + carry.shape[2:])
        return jnp.swapaxes(carry, 0, 1).reshape((n_batch * n_out, width) + carry.shape[3:])
    return jax.jit(order, out_shardings=spec)


def _run_priced(program, *args, label):
    """Run ``program`` as the executable it books in the open section: its own temp and outputs
    (``runtime.aot_memory.compiled_new_bytes``) beside the bytes in use, compiled once per signature."""
    from lxkit import memory
    from runtime.aot_memory import compiled_new_bytes
    from gw.tangential_execution import compiled_for
    executable = compiled_for(program, args)
    memory.book(memory.priced(compiled_new_bytes(executable), f"{label} (compiled)", increment=True))
    return executable(*args)


@lru_cache(maxsize=8)
def _moment_batch_split(mesh_xy, n_out, rows):
    """One q batch's bank run ``[n_out, width, m, n]`` → its ``n_out`` totals ``[rows, m, n]``."""
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    return jax.jit(lambda run: tuple(run[o, :rows] for o in range(n_out)),
                   out_shardings=(face,) * n_out)


def _batch_totals(run, finish, split):
    """A q batch's totals from its read run. The photon path's canonical copy (``finish``) is
    live beside the run and the totals: a part of its own. The split shares the read-back's
    part (they run in turn beside the same residents).
    """
    canonical = run if finish is None else _run_priced(finish, run, label="moment canonical order")
    return _run_priced(split, canonical, label="moment correlations")


def ordered_moments(steps, execute, h, a0, a1, o0, o1):
    """M0..M3 of an ordered charge bank from its scaled totals, one step per dispatch
    (``_response_programs``' ordered ``moments``). Every total is consumed: X_k is formed in
    its total's face, C_k accumulates in a face of its own (X2 and X3 are read again; C4 in
    X4's), and M_k in C_k's. At most X1..X3, C2, C3, the running C4 and one product's work
    are live."""
    xs = [execute(steps.term, (h, v), "moment_dyson") for v in (o0, a0, o1, a1)]
    cs = [xs[0]]
    for k in range(1, 4):
        c = xs[k]
        for i in range(k):
            step = steps.add_product if i == 0 and k < 3 else steps.accumulate
            c = execute(step, (c, xs[i], cs[k - i - 1]), "moment_dyson")
        cs.append(c)
    del xs, c
    return tuple(execute(steps.half_term, (h, cs.pop(0)), "moment_dyson") for _ in range(4))


def streamed_moment_totals(wfns, meta, *, mesh_xy, qids, width, execute, ordered, receipt, root,
                           vertex=None):
    """Every parent's exact-moment totals from one pass of the row-pass engine, or ``None``.

    Each correlation of :data:`EVEN_MOMENT_TERMS` (and :data:`ODD_MOMENT_TERMS`
    on an ordered bank) is one direct node at t = 0 with its own band weights
    ``f E^a`` and ``u E^b``; the node-to-output weights add ``-1j (ahead -
    behind) pref c`` (even) or ``(ahead - behind) pref c`` (odd) to its total,
    the bare correlation :func:`exact_bare_moments` forms one q batch at a
    time.  Row passes run outer and every parent inner, so each correlation's
    Greens, kconv calls and transforms run once per map instead of once per q
    batch; the four-current stream (``vertex``) runs its family pairs' passes
    in turn.  Each finished pass goes to a :class:`file_io.slab_io.StreamedBank`
    whose outputs are the totals of one q batch of ``width`` parents, so a
    batch's totals are read back as one run.  Returns ``(bank, finish)``
    (``finish`` takes a read run to the canonical photon order; ``None`` for
    charge), or ``None`` without the engine (mathdx mode 11 from raw parents)
    or when the store cannot be reserved.
    """
    from common.gpu_utils import host_bytes_per_process
    from file_io.slab_io import StreamedBank
    from .w_isdf import _w_solve_pref_scalar
    segments = stream_segments(wfns, meta, mesh_xy, len(qids), vertex)
    if segments is None:
        return None
    energy, f, u, reference, _ = response_weights(wfns, meta)
    erel, pref = energy - reference, _w_solve_pref_scalar(meta)
    totals = ([(terms, -1j) for terms in EVEN_MOMENT_TERMS]
              + ([(terms, 1.) for terms in ODD_MOMENT_TERMS] if ordered else []))
    lower, upper, columns = [], [], []
    for o, (terms, scale) in enumerate(totals):
        for c, a, b in terms:
            lower.append(f * erel ** a)
            upper.append((-1j if scale == -1j else 1.) * u * erel ** b)
            column = np.zeros((2, len(totals)), np.complex128)
            column[:, o] = scale * pref * c * np.asarray([1., -1.])
            columns.append(column)
    n_out, nq, n_nodes = len(totals), len(qids), len(columns)
    n_batch = -(-nq // int(width))
    nbytes = n_batch * n_out * 16 * int(width) * sum(r * c for r, c, _ in segments[0])
    bank = StreamedBank(mesh_xy, root=root, label="moments",
                        kind="host" if nbytes <= host_bytes_per_process() // 2 else "file",
                        n_out=n_batch * n_out, q=int(width), segments=segments[0], tile=segments[1])
    if not bank.fits:
        return None
    common = (jnp.zeros(n_nodes, jnp.complex128), jnp.asarray(np.stack(columns, axis=-1)))
    tail = (stream_weights(wfns, np.stack(lower), mesh_xy).astype(jnp.complex128),
            stream_weights(wfns, np.stack(upper), mesh_xy).astype(jnp.complex128),
            jnp.asarray([reference, reference]))
    scratch = _stream_scratch(wfns, meta, mesh_xy, vertex)
    order = _moment_batch_major(mesh_xy, n_out, int(width), n_batch)
    outputs = [(r, r) for r in range(n_batch * n_out)]
    # The passes being drained stay live beside every correlation's admission.
    ledger = meta.tangential_capacity
    live = ledger.live_stages
    in_flight = bank.in_flight * 2 * n_batch * int(width) * n_out * 16 * max(r * c for r, c, _ in segments[0])
    carries, _ = _reserve(meta, "moment_pass_carry", 0, in_flight, book=False)
    ledger.live_stages = live + (carries,)
    # The stream's own price: what it holds now (bank, weights, scratch: measured) and the
    # passes in flight; each pass's dispatch is priced by its program (bank.dispatch).
    from lxkit import memory
    memory.book(memory.priced(in_flight, "moment passes in flight (carry and its batch-major copy)",
                              increment=True))
    passes = _passes_ahead(bank.shapes, (n_out, nq), mesh_xy, common, tail, lambda p: response_stream(
        wfns, meta, mesh_xy=mesh_xy, q_ids=tuple(qids), n_outputs=n_out, pair_mode="direct",
        bank_carry=True, ordered=ordered, vertex=vertex, stream_pass=p))
    for p, (kernel, fixed, shape) in enumerate(passes):
        carry = _group_zeros(mesh_xy, shape)
        carry = execute(kernel, common + tuple(fixed) + tail + (carry, jnp.int32(p)),
                        "moment_correlation", runtime_bytes=scratch)
        bank.put(p, order(carry), outputs)
        del carry
    bank.commit()
    ledger.live_stages = live
    receipt["correlation_count"] += n_nodes
    receipt["moment_stream"] = dict(bank.receipt(), passes=len(bank.shapes), nodes=n_nodes,
                                    q_batches=n_batch, width=int(width))
    return bank, segments[2]


def compute_moment_bank(wfns, meta, config, *, mesh_xy, sym, bank_io,
                        vertex=None, contact=None, direct_head=None):
    """Stage B: six exact correlations, physical recurrence, scratch write."""
    from file_io.tangential_store import write_tangential_bank
    header, qids, census = _bank_context(wfns, meta, sym, bank_io, mesh_xy)
    receipt = _receipt("moments", census, bank_io)
    if not bool(sym.trs_allowed):
        # M1/M3 are the 1/s and 1/s^2 coefficients; the odd channel starts at
        # 1/z^3, so the same six correlations stay exact on an ordered bank.
        receipt["ordered"] = True
    execute = _bank_execution(meta, mesh_xy, receipt, config, photon=vertex is not None)
    ledger = meta.tangential_capacity
    ambient = ledger.live_stages
    started = time.monotonic()
    n = meta.mu_basis.n_packed if vertex is None else vertex.n
    face_bytes = 16*n**2 // mesh_xy.size
    # Two totals, next correlation, arithmetic temporaries and bounded H/solve.
    ordered = vertex is not None or not bool(sym.trs_allowed)
    _, _, moments, receipt["algebra"] = response_algebra(meta, config,
        mesh_xy=mesh_xy, n=n, ordered=ordered, photon=vertex is not None)
    # Every parent's totals in one stream (correlations outer would repeat each
    # correlation's Greens and kconv calls per q batch); a bank partly written
    # by an earlier attempt and a backend without the row-pass engine take the
    # per-batch correlations.  A streamed batch holds its totals and the next
    # batch's (read ahead), H, the Dyson outputs and its temporaries (3 faces
    # per total plus 4 per parent; the four-current read's canonical copy one
    # more per total); a correlated batch two totals, the next correlation,
    # arithmetic temporaries and bounded H/solve.
    n_total = 4 if ordered else 2
    streamed = finish = None
    streamed_per_q = (3 if vertex is None else 4) * n_total + 4
    # A batch's outputs are (per_q·w + 16) faces; its Dyson runs in the dense layout.
    moment_width = lambda per_q: parent_span(len(qids), layer=_parent_layer(receipt, mesh_xy),
                                             parent_bytes=per_q * face_bytes, fixed_bytes=16 * face_bytes)
    if not np.asarray(header["moment_written"]).any():
        qwidth = moment_width(streamed_per_q)
        with timing.section("bank.moment_stream"):
            stream_started = time.monotonic()
            streamed = streamed_moment_totals(wfns, meta, mesh_xy=mesh_xy, qids=qids,
                width=qwidth, execute=execute, ordered=ordered, receipt=receipt,
                root=bank_io["root"], vertex=vertex)
            if streamed is not None:
                streamed, finish = streamed
            receipt["seconds"]["moment_stream"] = time.monotonic() - stream_started
    per_q = streamed_per_q if streamed is not None else (12 if ordered else 8)
    if streamed is None:
        qwidth = moment_width(per_q)
    receipt["q_width"] = int(qwidth)
    receipt["q_batches"] = [[q0, min(q0 + qwidth, len(qids))] for q0 in range(0, len(qids), qwidth)]
    # A batch's read-back and split run in turn beside the same residents: one part, the larger.
    runs = None if streamed is None else streamed.reader(
        [(j * n_total, (j + 1) * n_total) for j in range(len(receipt["q_batches"]))],
        run=partial(_run_priced, label="moment correlations"))
    # The q batches in their own section. Each program a batch runs is priced by its own
    # executable in its dispatch section (bank.dispatch.coulomb_sqrt, .moment_dyson); the
    # batch's ledger row admits its outputs and is no price of this section.
    with timing.section("bank.moment_batches"):
        for q0 in range(0,len(qids),qwidth):
            q1 = min(q0+qwidth,len(qids))
            ledger.live_stages = ambient
            name,_ = _reserve(meta,"bank_outputs_moments",(per_q*(q1-q0)+16)*face_bytes, book=False)
            ledger.live_stages = ambient+(name,)
            if not np.asarray(header["moment_written"])[q0:q1].all():
                # The roots first: their factorization runs beside no total.
                with _moment_phase(receipt, "coulomb"):
                    h, hi, ranks = _coulomb_batch(meta, config, bank_io, mesh_xy, (q0,q1), execute)
                    del hi
                with _moment_phase(receipt, "correlations"):
                    if runs is not None:
                        run = next(runs)
                        totals = _batch_totals(run, finish, _moment_batch_split(mesh_xy, n_total, q1 - q0))
                        del run
                        a0, a1, o0, o1 = totals if ordered else (*totals, None, None)
                        del totals
                    elif ordered:
                        a0, a1, o0, o1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh_xy,
                            q_ids=tuple(qids[q0:q1]), execute=execute, ordered=True, vertex=vertex)
                    else:
                        a0, a1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh_xy,
                                                  q_ids=tuple(qids[q0:q1]), execute=execute)
                    # The batch's totals finish here, so their device time is theirs.
                    timing.stage_wait((a0, a1) + ((o0, o1) if ordered else ()))
                with _moment_phase(receipt, "diagnostics"):
                    _check_bare_hermitian(mesh_xy, q0, receipt, A0=a0, A1=a1)
                operands = (h,a0,a1,o0,o1) if ordered else (h,a0,a1)
                with _moment_phase(receipt, "dyson"):
                    result = (ordered_moments(moments, execute, *operands) if ordered and vertex is None
                              else execute(moments, operands + (() if vertex is None else (contact,)),
                                           "moment_dyson"))
                    timing.stage_wait(result)
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
                    with _moment_phase(receipt, "diagnostics"):
                        _record_odd_moments(q0, *(values[name] for name in ("M0", "M1", "M2", "M3")),
                                            receipt)
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
                    header = write_tangential_bank(bank_io["path"], q_span=span,
                        **{name: values[name][lo:hi] for i,name in enumerate(fields) if not marked[lo,i]},
                        meta=meta, expected_identity=bank_io["identity"], mesh_xy=mesh_xy)
                    receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                    receipt["batches"].append(dict(q_span=span,support_ranks=ranks[lo:hi]))
                del h,a0,a1,operands,result,values
                if ordered:
                    del o0,o1
                if runs is None:
                    receipt["correlation_count"] += 10 if ordered else 6
    if streamed is not None:
        for key, value in streamed.seconds.items():
            receipt["seconds"]["moment_stream_" + key] = value
        streamed.release()
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
    it: the tangential Gram gate sees it only after the Loewner pencil has
    amplified it by ~1e2.
    Every rank evaluates it; no rank-conditional device work (INVARIANTS 21).
    """
    if not _self_negative(int(q_full), meta):
        return
    points = np.asarray(z_batch)
    rows = [i for i, z in enumerate(points) if z.real == 0.0]
    if not rows:
        return
    # The defect stays on the device until the receipt is finished: nothing on the
    # host decides on it, so one stacked read serves every recorded support
    # (_finish_receipt), not one read per sample and parent.
    receipt.setdefault("_reciprocity_pending", []).append(
        (_reciprocity_scalars(value), [(dict(q_parent=int(parent), q_full=int(q_full),
                                             u_ry=float(points[i].imag)), i) for i in rows]))


def _resolve_reciprocity(receipt):
    """The pending reciprocity defects as receipt rows, read back in one transfer."""
    pending = receipt.pop("_reciprocity_pending", None)
    if not pending:
        return
    import jax.numpy as jnp
    flat = np.asarray(jnp.concatenate([jnp.ravel(values) for values, _ in pending]))
    rows, offset = receipt.setdefault("reciprocity_real_supports", []), 0
    for values, entries in pending:
        for entry, i in entries:
            rows.append(dict(entry, defect=float(flat[offset + i])))
        offset += int(np.prod(values.shape))


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
    every rank computes, rank 0 prints one line per sample. Runs only under
    ``sigma_freq_debug_output``: no gate reads it, and its extra Dyson solve
    per imaginary sample is debug work (owner rule: none in the prefactor).
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
        log.debug("TRBANK tr_odd_census " + " ".join(f"{k}={row[k]}" for k in row))


def response_groups(z, group_size):
    """Consecutive sample groups: imaginary axis by Im z, then the rest by Re z.

    Neighbouring samples need nearly the same exponentials, so a group costs
    about as many Green pairs as its hardest member.
    """
    order = sorted(range(len(z)), key=lambda i: (z[i].real != 0.,
                   z[i].real if z[i].real != 0. else z[i].imag))
    return [order[i:i+group_size] for i in range(0, len(order), group_size)]


#: Stands in a request's attempt table for a node whose attempt raised: the
#: walk stops there, and the refusal is raised only if the serial walk reaches it.
_REFUSED = object()


#: An idle process's speculative attempt may run this many times the round's
#: longest needed load (Fe V2 replay simulation: P=8 28.9 s against 41.5 s
#: without it; P=2 and P=4 unchanged).
_IDLE_SPECULATION_SPAN = 1.5


def _attempt_cost(tree, node):
    """Relative wall of one whole-group attempt, per pencil route.

    Host LAPACK pencils: (size/1000)^3, +0.85 for one sample. The shift
    pencil's QR/SVD is ~size^3 per geometry; a one-sample attempt scans without
    the stagnation patience. Fitted on 21 Frontier attempts of the Fe V2 call
    (7 BLAS threads; 3.8 s*(size/1000)^3 + 3.3 s per one-sample node, rms 1.3 s
    against 1.9 s for size^3 alone).

    Device pencils (``tree.pencil`` set): the pencil runs ahead of the host
    scan, so the scan dominates: (size/1000)^3 + 1.9, +3.0 for one sample.
    Fitted on the 14 attempts of the Fe V2 walk on one MI250X GCD (1.05 s*
    (size/1000)^3 + 1.96 s + 3.17 s per one-sample node, rms 1.1 s against 1.8
    s for the host form; pool log rcgpu_ab1). Only the planner's balance reads
    it, never a rule.

    A warm node (an earlier map's rule on the same member set) is tried first
    on that rule's nodes: one projection pass and no pencil when it fits, so
    it costs 0 here (Fe 8^3 SC maps 1-3: 0.2-0.8 s, against 2.4 s modelled,
    whose speculation then outlasted the round's needed work by ~3 s).
    """
    if tree.warm(node):
        return 0.
    one = len(node) == 1
    if getattr(tree, "pencil", None) is not None:
        return (tree.size(node)/1000.)**3 + 1.9 + (3.0 if one else 0.)
    return (tree.size(node)/1000.)**3 + (0.85 if one else 0.)


def _rule_round(trees, known, world):
    """One round's attempts per process: the serial walks' next nodes, then speculation.

    Every process computes the same plan from the same ``known`` tables. The
    nodes the walks need now are placed longest-first on the least-loaded
    process; the round then lasts about the longest load ``T``. Untried
    descendants of those nodes, most likely needed first (the product of
    their undecided tried ancestors' pencil sizes over the largest, since a
    wider shared pencil fails more often), fill each process up to ``T``. A
    speculative attempt that the walk never reaches is discarded; it never
    changes a rule. Costs are :func:`_attempt_cost`. A process the round
    left idle may take one candidate up to :data:`_IDLE_SPECULATION_SPAN`
    times ``T``: on many processes the next level of the chain then starts a
    round early instead of waiting for its own.
    """
    need = [(i, node) for i, tree in enumerate(trees) for node in tree.pending(known[i])]
    if not need:
        return None
    cost = {}

    def estimate(i, node):
        if (i, node) not in cost:
            cost[i, node] = _attempt_cost(trees[i], node)
        return cost[i, node]
    load, plan = [0.]*world, [[] for _ in range(world)]
    for task in sorted(need, key=lambda t: -estimate(*t)):
        r = min(range(world), key=lambda r: (load[r], r))
        plan[r].append(task)
        load[r] += estimate(*task)
    if world > 1:
        span = max(load)
        wide = max(float(t.size(t.root)) for t in trees)
        candidates = []

        def descend(i, node, odds, depth):
            for half in trees[i].children(node):
                if trees[i].tried(half):
                    if half not in known[i]:
                        candidates.append((-odds, depth, len(candidates), i, half))
                    elif known[i][half] is not None:
                        continue
                    descend(i, half, odds*min(1., trees[i].size(half)/wide), depth+1)
                else:
                    descend(i, half, odds, depth+1)
        for i, node in need:
            descend(i, node, min(1., trees[i].size(node)/wide), 1)
        for *_, i, node in sorted(candidates):
            r = min(range(world), key=lambda r: (load[r], r))
            if load[r] + estimate(i, node) <= span or (
                    not load[r] and estimate(i, node) <= _IDLE_SPECULATION_SPAN*span):
                plan[r].append((i, node))
                load[r] += estimate(i, node)
    return plan


def _gather_group_rules(requests, tree):
    """Build every request's rule tree across hosts; replicate the small rules.

    ``tree(members)`` is the request's ``minimax.ResponseGroupTree``. Its
    serial walk is a chain of whole-group attempts, each a pure function of
    its node (one machine class, pinned BLAS threads; with device pencils, one
    device class), so any process may
    evaluate any attempt: rounds of attempts (:func:`_rule_round`) run on all
    processes, cross hosts as data, and every process walks the same tables
    to the serial walk's rules, byte for byte (one list per request). Refusals
    cross as data and are raised only where the serial walk would raise them.
    """
    import pickle
    from common.collectives import all_gather_processes, process_count, process_rank
    rank, world = int(process_rank()), int(process_count())
    trees = [tree(members) for members in requests]
    known = [dict() for _ in trees]
    errors = [dict() for _ in trees]
    while (plan := _rule_round(trees, known, world)) is not None:
        # A device pencil source compiles the round's shapes here, on every
        # rank in the same order (cross-rank compile agreement), each rank's
        # first attempt first; the attempts below, which differ per rank, then
        # only execute. Only the round's shapes: compiles share the rank's
        # cores with the host scans.
        for i, t in enumerate(trees):
            prepare = getattr(getattr(t, "pencil", None), "prepare", None)
            if prepare is not None:
                prepare([shape for position in range(max(map(len, plan)))
                         for tasks in plan if position < len(tasks) and tasks[position][0] == i
                         for shape in t.pencil_shapes(tasks[position][1])])
        local = []
        for i, node in plan[rank]:
            try:
                local.append((i, node, trees[i].attempt(node), None))
            except Exception as error:  # refusals cross hosts as data, then raise
                local.append((i, node, None, f"{type(error).__name__}: {error}"))
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
        for shard in shards:
            for i, node, rules, error in shard:
                known[i][node] = _REFUSED if error is not None else rules
                if error is not None:
                    errors[i][node] = error
    out = []
    for i, t in enumerate(trees):
        attempts = {node: got for node, got in known[i].items() if got is not _REFUSED}
        refused = [node for node in t.pending(attempts) if node in errors[i]]
        if refused:
            raise ValueError(errors[i][refused[0]])
        out.append(t.rules(attempts))
    return out


#: The energy pad (eV) of a held SC response plan: the rule interval and the
#: compiled band bounds of the direct stream are rebuilt only when the support
#: moves past it.
RESPONSE_HOLD_PAD_EV = 4.0


def response_support(wfns, meta, sample_plan, receipt, *, print_fn=print):
    """Occupation weights, transition interval, envelope and band support."""
    energy, f, u, _, _ = response_weights(wfns, meta)
    f, u, receipt["sample_activity"] = response_sample_weights(f, u)
    refs = np.array([energy[f != 0].max(), energy[u != 0].min()])
    lo, hi = refs[1]-refs[0], float(energy[u != 0].max()-energy[f != 0].min())
    mu = sample_plan["census"]["mu_ry"]
    decay_rate, amplitude = response_occupation_envelope(energy, f, u, mu)
    from .greens_function_kernel import _phase_band_interval
    lo_band, hi_band = _phase_band_interval(np.stack((np.asarray(f), np.asarray(u))))
    # Enclose every parent's exact weight support. Fixed bounds share one
    # batched GEMM on the direct stream's band-complete ψ rows and remain safe
    # when a complex-time phase underflows.
    band_ranges = tuple((int(lo.min()), int(hi.max())) for lo, hi in zip(lo_band, hi_band))
    # SC: the bounds are compiled into the stream, so they are held across maps
    # with the response rule's own energy pad and widened only when the support
    # leaves them. Bands outside the support carry exact-zero weight, so a held
    # wider interval is the same product.
    session = getattr(meta, "tangential_response_rules", None)
    if session is not None:
        held = session.get("band_ranges")
        if held is None or any(a < c or b > d for (a, b), (c, d) in zip(band_ranges, held)):
            pad = RESPONSE_HOLD_PAD_EV / RYD_TO_EV
            hi_f = int(np.flatnonzero(energy.min(axis=0) <= energy[f != 0].max() + pad).max()) + 1
            lo_u = int(np.flatnonzero(energy.max(axis=0) >= energy[u != 0].min() - pad).min())
            padded = ((band_ranges[0][0], max(band_ranges[0][1], hi_f)),
                      (min(band_ranges[1][0], lo_u), band_ranges[1][1]))
            if held is not None:
                padded = tuple((min(a, c), max(b, d)) for (a, b), (c, d) in zip(padded, held))
            session["band_ranges"] = padded
        band_ranges = session["band_ranges"]
    log.debug(f"Response occupied/empty band intervals: {band_ranges} of {f.shape[-1]}")
    support = dict(f=f, u=u, refs=refs, lo=lo, hi=hi, mu=mu, decay_rate=decay_rate,
                   amplitude=amplitude, band_ranges=band_ranges)
    spec = getattr(meta, "tangential_response_blocks", None)
    if spec is not None:
        # The TD refit only (tdagw.dynsig.spm_screening): chi0 as four blocks on the
        # same stream; the band bounds above enclose every block's weights.
        support["blocks"] = response_blocks(energy, f, u, spec["active"])
    return support


#: The TD refit's chi0 pair blocks (hole space x electron space): F the frozen
#: bands (occupations 0/1 throughout), A the active range.  Only AA crosses zero.
RESPONSE_BLOCKS = ("FF", "FA", "AF", "AA")


def response_blocks(energy, f, u, active):
    """chi0 as four (hole x electron) blocks, each with its own transition interval.

    ``active`` = (a0, a1): the active bands A (0-based, half-open, on the band
    axis of ``energy``); F is every other live band and must hold 0/1
    occupations.  Hole weights are f (G^<), electron weights u = 1 - f (G^>),
    as on the one-block path; a block's weights are f and u masked to its
    hole and electron bands, so sum_b f_b u_b^T = f u^T exactly.  Each block's
    interval and references come from band energies only (F's occupied bands
    as holes and empty bands as electrons, every A band on both sides), never
    from the populations, so its rule is built once and held for the run
    (:func:`response_quadrature`).  FF, FA and AF must be strictly positive;
    AA, on [-span_A, +span_A], is the one crossing block (no occupation
    envelope: decay rate 0).  A block without hole or electron bands is dropped.
    """
    a0, a1 = (int(x) for x in active)
    band = np.arange(energy.shape[1])[None, :]
    live = (f != 0) | (u != 0)
    inside = (band >= a0) & (band < a1)
    A, F = inside & live, ~inside & live
    if not (0 <= a0 < a1 <= energy.shape[1]) or not np.any(A):
        raise ValueError(f"GATE response_blocks_active: got active bands [{a0}, {a1}) of "
                         f"{energy.shape[1]}; want a nonempty range of live chi0 bands")
    bad = F & (((f != 0) & (f != 1)) | ((u != 0) & (u != 1)))
    if np.any(bad):
        k, b = (int(x[0]) for x in np.nonzero(bad))
        raise ValueError("GATE response_blocks_frozen_fractional: got a fractional occupation "
                         f"f = {f[k, b]:.6g} on frozen band {b} (k row {k}) outside the active "
                         f"bands [{a0}, {a1}); want 0/1 on every frozen band; why: the frozen "
                         "blocks' rules are positive-interval rules built once for the run")
    holes, electrons = dict(F=F & (f == 1), A=A), dict(F=F & (u == 1), A=A)
    out = []
    for name in RESPONSE_BLOCKS:
        h, e = holes[name[0]], electrons[name[1]]
        if not np.any(h) or not np.any(e):
            continue
        refs = np.array([energy[h].max(), energy[e].min()])
        lo, hi = float(refs[1] - refs[0]), float(energy[e].max() - energy[h].min())
        if name != "AA" and not lo > 0:
            raise ValueError(f"GATE response_blocks_interval: got the {name} block's interval "
                             f"[{lo*RYD_TO_EV:.6g}, {hi*RYD_TO_EV:.6g}] eV with active bands "
                             f"[{a0}, {a1}); want lo > 0 (a band gap between A and F on that "
                             "side); fix: widen the active range to the next gap")
        out.append(dict(name=name, f=np.where(h, f, 0.), u=np.where(e, u, 0.), refs=refs,
                        lo=lo, hi=hi))
    return out


def _block_quadrature(meta, sample_plan, receipt, support, *, group_size, print_fn=print):
    """Every block's rule (held for the run) and the evaluation groups that add the blocks.

    A block's rule is built once on every sample (decay rate 0, its own
    interval and gauge reference; minimax ``panels`` for a sample whose pencil
    fails) and held in the block spec (``meta.tangential_response_blocks
    ["held"]``) while its interval and the samples are unchanged.  An
    evaluation group is at most ``group_size`` samples in the one-block order
    (:func:`response_groups`); its passes are every block rule group with a
    member in it, that rule's weights placed at the group's slots (zeros
    elsewhere), so every pass runs the one stream program into the group's
    one carry (:func:`integrate_response_group`).
    """
    import minimax
    from .sigma_box_plan import snap_outward
    from .response_pencil import response_pencil_source
    spec = meta.tangential_response_blocks
    held = spec.setdefault("held", {})
    z = bank_points(sample_plan)
    rel_tol = float(sample_plan["bank_rule_tolerance"])
    pencil = response_pencil_source()
    blocks, plans, reused = support["blocks"], [], []
    for blk in blocks:
        lo, hi = snap_outward(blk["lo"], 1., -1), snap_outward(blk["hi"], 1., +1)
        old = held.get(blk["name"])
        keep = (old is not None and old["lo"] == lo and old["hi"] == hi
                and old["rel_tol"] == rel_tol and np.array_equal(old["z"], z))
        if not keep:
            def tree(members, lo=lo, hi=hi):
                return minimax.ResponseGroupTree(lo, hi, z[members], rel_tol=rel_tol,
                                                 decay_rate=0., pencil=pencil, panels=True)
            requests = response_groups(z, len(z))
            with timing.section("bank.rule_construction", announce=True):
                rule_groups = [dict(rule, members=[members[m] for m in rule["members"]])
                               for members, rules in zip(requests, _gather_group_rules(requests, tree))
                               for rule in rules]
            held[blk["name"]] = old = dict(name=blk["name"], lo=lo, hi=hi, z=z, rel_tol=rel_tol,
                                           reference=lo, rule_groups=rule_groups)
        plans.append(old)
        reused.append(keep)
    order = response_groups(z, len(z))[0]
    groups = []
    for a in range(0, len(order), group_size):
        members = order[a:a+group_size]
        slot = {m: i for i, m in enumerate(members)}
        passes = []
        for b, plan in enumerate(plans):
            for rule in plan["rule_groups"]:
                rows = [(slot[m], j) for j, m in enumerate(rule["members"]) if m in slot]
                if rows:
                    passes.append(dict(block=b, rule=rule, rows=rows))
        error = np.zeros((len(members), 2, 2))
        mass = np.zeros((len(members), 2, 2))
        for p in passes:
            for i, j in p["rows"]:
                error[i] += p["rule"]["sampled_error"][j]
                mass[i] = np.maximum(mass[i], p["rule"]["coefficient_mass"][j])
        groups.append(dict(members=members, passes=passes,
                           count=int(sum(p["rule"]["count"] for p in passes)),
                           sampled_error=error, coefficient_mass=mass))
    receipt["rule_provider"] = ("minimax shared-node group fits per chi0 block (FF, FA, AF positive; "
                                "AA crossing, decay rate 0; closed-form panels where a pencil fails)")
    receipt["rule"] = dict(
        interval_ry=[min(p["lo"] for p in plans), max(p["hi"] for p in plans)],
        group_size=group_size, decay_rate_ry_inv=0., occupation_amplitude=1., reused=all(reused),
        blocks=[dict(name=p["name"], interval_ry=[p["lo"], p["hi"]], reused=r,
                     nodes=int(sum(g["count"] for g in p["rule_groups"])),
                     groups=[dict(members=[int(m) for m in g["members"]], count=int(g["count"]),
                                  panel=g.get("panel"), sampled_error=g["sampled_error"].tolist())
                             for g in p["rule_groups"]])
                for p, r in zip(plans, reused)],
        groups=[dict(members=[int(m) for m in g["members"]], count=g["count"], passes=len(g["passes"]),
                     sampled_error=g["sampled_error"].tolist()) for g in groups])
    receipt["nodes"] = int(sum(g["count"] for g in groups))
    for p, r in zip(plans, reused):
        log.debug(f"Response block {p['name']}: interval (eV) [{p['lo']*RYD_TO_EV:.8g}, "
                 f"{p['hi']*RYD_TO_EV:.8g}], {len(p['rule_groups'])} rule group(s), "
                 f"{sum(g['count'] for g in p['rule_groups'])} nodes; "
                 + ("held" if r else "constructed"))
    log.debug(f"Response quadrature: {len(z)} samples in {len(groups)} evaluation group(s) "
             f"(planned size {group_size}), {sum(len(g['passes']) for g in groups)} block passes, "
             f"{receipt['nodes']} total Green-pair evaluations")
    return dict(plan=None, groups=groups, slots=group_size, f=support["f"], u=support["u"],
                refs=support["refs"], band_ranges=support["band_ranges"],
                blocks=[dict(name=blk["name"], f=blk["f"], u=blk["u"], refs=blk["refs"],
                             reference=plan["reference"]) for blk, plan in zip(blocks, plans)])


def response_quadrature(meta, sample_plan, receipt, support, *, group_size, print_fn=print,
                        requests=None, rule_options=None):
    """Plan the shared complex-time rule on every sample; replicate small rules.

    Each node is ONE Green-pair evaluation that serves every member's forward
    and reverse orientation (see ``minimax.response_group_rules``).  The rule
    is fitted on all samples at once, so its nodes and eqp do not depend on
    ``group_size``; the evaluation streams the rule's nodes for at most
    ``group_size`` members per pass (``groups``).  A support with ``blocks``
    (the TD refit) takes :func:`_block_quadrature`.

    ``requests`` (default: every sample in one request, the bank's own) lists
    the member sets the rule trees are built on, one tree each (the CD memory
    windows, ``gw.cd.chi0_rules``); samples in no request get no rule.
    ``rule_options`` (``capacity``, ``max_kappa``) go to
    ``minimax.ResponseGroupTree``; a non-default capacity builds on the host
    pencils. The bank passes neither.
    """
    if "blocks" in support:
        if requests is not None or rule_options is not None:
            raise ValueError('the TD refit block quadrature takes no requests or rule_options')
        return _block_quadrature(meta, sample_plan, receipt, support,
                                 group_size=group_size, print_fn=print_fn)
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
    session = getattr(meta, "tangential_response_rules", None)
    old = None if session is None else session.get("frequency")
    metallic = sample_plan["census"]["partial_at_mu"]
    failed = [] if old is None else [
        text for text, held in (
            (f"lo {old['lo']:.6g} -> {lo:.6g} Ry", old["lo"] <= lo),
            (f"hi {old['hi']:.6g} -> {hi:.6g} Ry", hi <= old["hi"]),
            (f"decay_rate {old['decay_rate']:.6g} -> {decay_rate:.6g} /Ry",
             old["decay_rate"] <= decay_rate),
            (f"amplitude {old['amplitude']:.6g} -> {amplitude:.6g}", old["amplitude"] >= amplitude),
            (f"metallic {old['metallic']} -> {metallic}", old["metallic"] == metallic),
            (f"z moved (max {np.max(np.abs(old['z']-z))*RYD_TO_EV:.3g} eV)"
             if np.shape(old["z"]) == np.shape(z) else "z sample count changed",
             np.array_equal(old["z"], z)))
        if not held]
    reuse = old is not None and not failed
    if reuse:
        plan = old
    else:
        if failed and jax.process_index() == 0:
            log.warning("Response rule rebuilt at a held map; failed reuse test: "
                     + "; ".join(failed))
        pad = RESPONSE_HOLD_PAD_EV/RYD_TO_EV if session is not None else 0.
        plan = dict(lo=snap_outward(support["lo"]-pad, 1., -1),
                    hi=snap_outward(support["hi"]+pad, 1., +1), z=z, metallic=metallic,
                    decay_rate=decay_rate, amplitude=amplitude)
        plan["reference"] = 0. if decay_rate else plan["lo"]
        # One rule on every sample, whatever the evaluation group: the node set
        # (and so eqp) never depends on the memory budget; the group only
        # batches the evaluation below.
        requests = (response_groups(z, len(z)) if requests is None
                    else [[int(m) for m in r] for r in requests])
        previous = [] if old is None else old["rule_groups"]
        from .response_pencil import response_pencil_source
        options = dict(rule_options or {})
        pencil = response_pencil_source()
        if options.get("capacity") not in (None, minimax.RESPONSE_RULE_CAPACITY):
            pencil = None                     # the device pencils serve the bank's rank only
        log.debug("Response rule pencils: " + ("device QR + dilation eigh, decisions on host"
                 if pencil is not None else "host LAPACK"))

        def tree(members):
            local = {m: i for i, m in enumerate(members)}
            warm = [dict(rule, members=[local[m] for m in rule["members"]])
                    for rule in previous if set(rule["members"]) <= set(members)]
            return minimax.ResponseGroupTree(
                plan["lo"], plan["hi"], z[members],
                rel_tol=sample_plan["bank_rule_tolerance"]/amplitude,
                decay_rate=decay_rate, previous=warm, pencil=pencil, **options)

        with timing.section("bank.rule_construction", announce=True):
            plan["rule_groups"] = [
                dict(rule, members=[members[m] for m in rule["members"]])
                for members, rules in zip(requests, _gather_group_rules(requests, tree))
                for rule in rules]
        if session is not None:
            session["frequency"] = plan
    if plan["decay_rate"]:
        refs[:] = mu
    # Each evaluation pass streams its rule's whole node set for at most
    # group_size members (their own value and derivative rows).
    groups = [dict(rule, **{key: rule[key][a:a+group_size] for key in
                            ("members", "value", "derivative",
                             "sampled_error", "coefficient_mass")})
              for rule in plan["rule_groups"]
              for a in range(0, len(rule["members"]), group_size)]
    receipt["rule_provider"] = ("minimax shared-node group fit (forward t, reverse conj t from "
                                "one Green pair); sampled scalar accuracy")
    receipt["rule"] = dict(interval_ry=[plan["lo"], plan["hi"]], group_size=group_size,
        groups=[dict(members=[int(m) for m in g["members"]], count=int(g["count"]),
                     sampled_error=g["sampled_error"].tolist(),
                     coefficient_mass=g["coefficient_mass"].tolist()) for g in groups],
        decay_rate_ry_inv=plan["decay_rate"], occupation_amplitude=plan["amplitude"], reused=reuse)
    receipt["nodes"] = int(sum(g["count"] for g in groups))
    if jax.process_index() == 0:
        log.debug(f"Response interval (eV): [{plan['lo']*RYD_TO_EV:.8g}, {plan['hi']*RYD_TO_EV:.8g}]")
        log.debug(f"Response quadrature: {len(z)} samples in {len(groups)} shared-node groups "
                 f"(planned size {group_size}); each node is one Green pair serving value, "
                 "ds and both orientations; " + ("reused" if reuse else "constructed"))
        for g in groups:
            points = " ".join(f"{complex(z[m])*RYD_TO_EV:.4g}" for m in g["members"])
            log.debug(f"Response quadrature: nodes {g['count']:4d}  error {g['sampled_error'].max():.2e}  "
                     f"kappa {g['coefficient_mass'].max():.2e}  z(eV) {points}")
        log.debug(f"Response quadrature: {receipt['nodes']} total Green-pair evaluations "
                 f"(value + derivative, forward + reverse)",
                  extra=record("response.quadrature", samples=len(z), nodes=receipt["nodes"],
                               interval_ev=[plan["lo"] * RYD_TO_EV, plan["hi"] * RYD_TO_EV],
                               z_ev=[[complex(v).real * RYD_TO_EV, complex(v).imag * RYD_TO_EV]
                                     for v in z],
                               reused=bool(reuse), rule=receipt["rule"]))
    return dict(plan=plan, groups=groups, slots=group_size, f=f, u=u, refs=refs,
                band_ranges=support["band_ranges"])


def _group_stream_arguments(rules, group):
    """Shared times and [forward/reverse, value/ds per slot, node] weights.

    Every group has ``rules["slots"]`` member slots, the planned group size: a
    short last group's empty slots carry zero weights, so every group runs the
    one executable the planner checked (a ragged last group compiled a third
    stream program at Fe/Ni 20^3).  An empty slot adds exact zeros to its own
    carry rows, which nothing reads; the members' rows are unchanged.
    """
    plan, refs = rules["plan"], rules["refs"]
    times = group["t"]
    # Translate the scalar gauge to the physical endpoint references; the
    # reverse orientation is evaluated at conj(t).
    shift = -(refs[1]-refs[0]-plan["reference"])
    gauge = (np.exp(shift*times), np.exp(shift*np.conj(times)))
    members = len(group["members"])
    weights = np.zeros((2, 2*max(members, int(rules["slots"])), times.size), np.complex128)
    for side in (0, 1):
        weights[side, 0:2*members:2] = -group["value"][:, side]*gauge[side]
        weights[side, 1:2*members:2] = -group["derivative"][:, side]*gauge[side]
    return times, weights


def _block_pass_arguments(rules, group, p):
    """Times and [forward/reverse, value/ds per slot, node] weights of one block pass.

    The rule's member rows go to the evaluation group's slots, every other
    slot is zero; the gauge is the block's (its references and its rule's
    reference), as :func:`_group_stream_arguments` does for the one block.
    """
    blk, rule = rules["blocks"][p["block"]], p["rule"]
    times = rule["t"]
    shift = -(blk["refs"][1]-blk["refs"][0]-blk["reference"])
    gauge = (np.exp(shift*times), np.exp(shift*np.conj(times)))
    weights = np.zeros((2, 2*max(len(group["members"]), int(rules["slots"])), times.size), np.complex128)
    for side in (0, 1):
        for slot, j in p["rows"]:
            weights[side, 2*slot] = -rule["value"][j, side]*gauge[side]
            weights[side, 2*slot+1] = -rule["derivative"][j, side]*gauge[side]
    return times, weights


def _integrate_block_passes(wfns, meta, mesh_xy, rules, group, *, q_ids, execute, receipt,
                            ordered, vertex, bank, outputs):
    """:func:`integrate_response_group` for an evaluation group of chi0 blocks.

    Every pass runs the one stream program (union band bounds, the block's
    masked weights and references as operands) and adds into the group's one
    donated carry, the previous pass's output.
    """
    n = meta.mu_basis.n_packed if vertex is None else vertex.n
    tails = [(stream_weights(wfns, blk["f"], mesh_xy), stream_weights(wfns, blk["u"], mesh_xy),
              jnp.asarray(blk["refs"])) for blk in rules["blocks"]]
    args = [(p["block"], *_block_pass_arguments(rules, group, p)) for p in group["passes"]]
    width = args[0][2].shape[1]
    scratch = _stream_scratch(wfns, meta, mesh_xy, vertex)
    receipt["correlation_count"] += int(group["count"])
    if bank is not None:
        passes = _passes_ahead(bank.shapes, (width, len(q_ids)), mesh_xy,
            (jnp.asarray(args[0][1]), jnp.asarray(args[0][2])), tails[args[0][0]],
            lambda p: response_stream(wfns, meta, mesh_xy=mesh_xy, q_ids=q_ids,
                n_outputs=width, pair_mode="direct", bank_carry=True, ordered=ordered,
                vertex=vertex, band_ranges=rules["band_ranges"], stream_pass=p))
        for p, (kernel, fixed, shape) in enumerate(passes):
            raw = _group_zeros(mesh_xy, shape)
            for b, times, weights in args:
                raw = execute(kernel, (jnp.asarray(times), jnp.asarray(weights)) + tuple(fixed)
                              + tails[b] + (raw, jnp.int32(p)), "direct", runtime_bytes=scratch)
            bank.put(p, raw, outputs)
            del raw
        return None
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
        q_ids=q_ids, n_outputs=width, pair_mode="direct", bank_carry=True,
        ordered=ordered, vertex=vertex, band_ranges=rules["band_ranges"])
    raw = _group_zeros(mesh_xy, (width, len(q_ids), n, n))
    for b, times, weights in args:
        raw = execute(kernel, (jnp.asarray(times), jnp.asarray(weights)) + tuple(fixed) + tails[b]
                      + (raw,), "direct", runtime_bytes=scratch)
    return raw


def _group_zeros(mesh_xy, shape):
    """The donated group carry [member, q, mu_X, nu_Y], made on device (no fill-value transfer)."""
    return _group_zeros_program(mesh_xy, tuple(int(n) for n in shape))()


@lru_cache(maxsize=16)
def _group_zeros_program(mesh_xy, shape):
    return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                   out_shardings=NamedSharding(mesh_xy, P(None, None, "x", "y")))


def _group_carry(mesh_xy, shape):
    """:func:`_group_zeros`' carry as its abstract value: what a compile needs, no allocation."""
    return jax.ShapeDtypeStruct(shape, jnp.complex128,
                                sharding=NamedSharding(mesh_xy, P(None, None, "x", "y")))


def _passes_ahead(shapes, lead, mesh_xy, common, tail, stream):
    """``(kernel, fixed, carry shape)`` of every row pass of a streamed bank, each
    pass's program already handed to a helper thread's compile.

    The passes' programs differ by segment shape and run one after another, so
    they compile side by side here instead of one at each pass's first run.
    ``stream(p)`` is pass ``p``'s ``response_stream``; its fixed operands are
    the stream's resident tables, so building every pass up front allocates
    nothing.
    """
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    passes = [(*stream(p), (*lead, px * rows, py * cols)) for p, (rows, cols) in enumerate(shapes)]
    for p, (kernel, fixed, shape) in enumerate(passes):
        _compiled(kernel, common + tuple(fixed) + tail + (_group_carry(mesh_xy, shape), jnp.int32(p)),
                  ahead=True)
    return passes


def integrate_response_group(wfns, meta, mesh_xy, rules, group, *, q_ids,
                             execute, receipt, ordered=False, vertex=None, bank=None, outputs=()):
    """Donated [value/ds per member, q, mu_X, nu_Y]; one Green/FFT scan per group.

    With a streamed ``bank`` (``file_io.slab_io.StreamedBank``) the scan runs one row
    pass at a time into that pass's carry, and each finished pass goes to the
    bank (carry row ``r`` as bank output ``o`` for ``(r, o)`` in ``outputs``)
    while the next pass computes; nothing is returned.  An evaluation group of
    chi0 blocks (``passes``) takes :func:`_integrate_block_passes`.
    """
    if "passes" in group:
        return _integrate_block_passes(wfns, meta, mesh_xy, rules, group, q_ids=q_ids,
            execute=execute, receipt=receipt, ordered=ordered, vertex=vertex, bank=bank,
            outputs=outputs)
    times, weights = _group_stream_arguments(rules, group)
    n = meta.mu_basis.n_packed if vertex is None else vertex.n
    common = (jnp.asarray(times), jnp.asarray(weights))
    tail = (stream_weights(wfns, rules["f"], mesh_xy), stream_weights(wfns, rules["u"], mesh_xy),
            jnp.asarray(rules["refs"]))
    scratch = _stream_scratch(wfns, meta, mesh_xy, vertex)
    receipt["correlation_count"] += int(group["count"])
    if bank is not None:
        passes = _passes_ahead(bank.shapes, (weights.shape[1], len(q_ids)), mesh_xy, common, tail,
            lambda p: response_stream(wfns, meta, mesh_xy=mesh_xy, q_ids=q_ids,
                n_outputs=weights.shape[1], pair_mode="direct", bank_carry=True, ordered=ordered,
                vertex=vertex, band_ranges=rules["band_ranges"], stream_pass=p))
        for p, (kernel, fixed, shape) in enumerate(passes):
            raw = _group_zeros(mesh_xy, shape)
            raw = execute(kernel, common + tuple(fixed) + tail + (raw, jnp.int32(p)),
                          "direct", runtime_bytes=scratch)
            bank.put(p, raw, outputs)
            del raw
        return None
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy,
        q_ids=q_ids, n_outputs=weights.shape[1], pair_mode="direct", bank_carry=True,
        ordered=ordered, vertex=vertex, band_ranges=rules["band_ranges"])
    raw = _group_zeros(mesh_xy, (weights.shape[1],len(q_ids),n,n))
    return execute(kernel, common + tuple(fixed) + tail + (raw,), "direct", runtime_bytes=scratch)


def _stream_executable(wfns, meta, mesh_xy, support, *, q_ids, n_outputs, ordered, vertex):
    """The group stream at ``n_outputs``, compiled once under the key its dispatch
    looks up (:func:`_compiled`), so the executable the planner checks is the
    one that runs.  The small operands are zero placeholders built as the
    dispatch builds them; the carry is abstract, so nothing large is allocated.
    """
    import minimax
    n = meta.mu_basis.n_packed if vertex is None else vertex.n
    kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh_xy, q_ids=q_ids,
        n_outputs=n_outputs, pair_mode="direct", bank_carry=True, ordered=ordered,
        vertex=vertex, band_ranges=support["band_ranges"])
    capacity = minimax.RESPONSE_NODE_CAPACITY
    args = (jnp.asarray(np.zeros(capacity, np.complex128)),
            jnp.asarray(np.zeros((2, n_outputs, capacity), np.complex128)),
            *fixed, stream_weights(wfns, support["f"], mesh_xy),
            stream_weights(wfns, support["u"], mesh_xy),
            jnp.asarray(np.zeros_like(support["refs"])),
            _group_carry(mesh_xy, (n_outputs, len(q_ids), n, n)))
    with timing.section('bank.compile.direct', announce=True):
        compiled = _compiled(kernel, args)
    if compiled.memory_analysis() is None:
        raise ValueError("GATE response_capacity: compiled memory unavailable")
    return compiled


def _direct_passes(wfns, meta, mesh_xy, q_count):
    """The charge direct stream's row passes (``w_isdf._direct_pass_plan``), or ``None``
    without a raw-parent plan."""
    from .w_isdf import _direct_pass_plan
    import minimax
    parent = wfns.green_parent
    kgrid, ns = (meta.nkx, meta.nky, meta.nkz), int(meta.nspinor)
    if parent is None:
        return None
    return _direct_pass_plan(mesh_xy, kgrid, parent.plan, n_rmu=meta.mu_basis.n_packed, ns=ns,
                             n_band=int(wfns.slices.nb_full), q_count=q_count,
                             n_nodes=minimax.RESPONSE_NODE_CAPACITY)


def stream_segments(wfns, meta, mesh_xy, q_count, vertex):
    """The direct stream's segments on the row-pass engine, or ``None`` without the engine.

    Returns ``(segments, tile, finish)``: per segment its local ``(rows, cols,
    rects)`` (``gw.subtile_stream.segment_blocks``: a charge row pass, or one
    family pair's row pass of the four-current stream), the local carry tile,
    and the program taking an assembled carry to the bank's layout (the
    four-current packed → canonical order; ``None`` for charge).
    """
    from .subtile_stream import segment_blocks
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    if vertex is None:
        plan = _direct_passes(wfns, meta, mesh_xy, q_count)
        if plan is None:
            return None
        n = int(meta.mu_basis.n_packed)
        segments = tuple(segment_blocks(plan, p, n // py)[1:] for p in range(len(plan.passes)))
        return segments, (n // px, n // py), None
    from .w_isdf import photon_segments
    families = vertex.families
    plans = _photon_plans(wfns, meta, mesh_xy, families, q_count)
    segments = tuple(segment_blocks(plans[i], p, 0)[1:] for i, p in photon_segments(plans))
    packed = int(families.packed_layout.packed_extent) // int(families.layout.mesh_side)
    return segments, (packed, packed), _photon_canonical(families, mesh_xy)


def _photon_plans(wfns, meta, mesh_xy, families, q_count):
    """The four-current direct stream's window plans per family pair (``w_isdf._photon_pass_plans``)."""
    import minimax
    from .w_isdf import _photon_pass_plans
    half_plans = tuple(plan.dirac_halves()[0] for plan in families.plans)
    return _photon_pass_plans(mesh_xy, (meta.nkx, meta.nky, meta.nkz), families, half_plans,
                              n_band=int(wfns.slices.nb_full), q_count=int(q_count),
                              n_nodes=minimax.RESPONSE_NODE_CAPACITY)


@lru_cache(maxsize=4)
def _photon_canonical(families, mesh_xy):
    """The packed → canonical photon order of a ``[n, q, N, N]`` carry (one program)."""
    from .photon_layout import photon_carry_order
    return jax.jit(lambda value: photon_carry_order(value, families, mesh_xy, to_packed=False),
                   out_shardings=NamedSharding(mesh_xy, P(None, None, "x", "y")))


def _stream_scratch(wfns, meta, mesh_xy, vertex):
    """Run-time scratch of the stream outside its compiled temporaries:
    mathdx mode 11's split arm on a raw-parent plan (w_isdf direct stream); for
    the four-current stream, the largest family pair's kconv call on its whole tile
    (a bound on every row pass's)."""
    from .greens_function_kernel import chi0_kconv_scratch
    kgrid = (meta.nkx, meta.nky, meta.nkz)
    if vertex is not None:
        from .photon_layout import FAMILY_PAIRS, family_channels
        carrier = vertex.families.packed_layout.carrier_extent
        return max(int(chi0_kconv_scratch(
            kgrid=kgrid, n_parent=int(vertex.families.n_parent),
            n_rmu=carrier(family_channels(L)[0]), ns=2, mesh=mesh_xy,
            n_right=carrier(family_channels(R)[0]))) for L, R in FAMILY_PAIRS)
    parent = wfns.green_parent
    if parent is None:
        return 0
    ns = int(meta.nspinor)
    return int(chi0_kconv_scratch(kgrid=kgrid, n_parent=int(parent.plan.n_parent),
                                 n_rmu=meta.mu_basis.n_packed, ns=ns, mesh=mesh_xy))


def _stream_workspace(wfns, meta, mesh_xy, support, *, q_ids, n_outputs, ordered, vertex):
    """(compiled temporaries + run-time scratch of the group stream, its executable).

    Lowered with the production shapes, so the first group's dispatch reuses
    this compilation when every sample fits in one group.
    """
    compiled = _stream_executable(wfns, meta, mesh_xy, support, q_ids=q_ids,
                                  n_outputs=n_outputs, ordered=ordered, vertex=vertex)
    return (int(compiled.memory_analysis().temp_size_in_bytes)
            + _stream_scratch(wfns, meta, mesh_xy, vertex)), compiled


def _unitary_inversion(plan):
    """The plan's spatial row equal to -1 with a complete centroid map, or ``None``.

    The row must be one the symmetry layer authorizes as unitary
    (``SymMaps.active_symmetry_rows``, INVARIANTS 16): where QE types inversion
    with time reversal (a PT-symmetric antiferromagnet) inversion alone is no
    symmetry, so it supplies no chi_{-q} and the -q rows are streamed.
    """
    ops, perm = np.asarray(plan.spatial_ops), np.asarray(plan.sym_perm)
    permitted = set() if plan.sym is None else set(np.asarray(plan.sym.active_symmetry_rows).tolist())
    rows = [r for r in range(int(plan.n_sym_spatial)) if r in permitted
            and np.array_equal(ops[r], -np.eye(3, dtype=ops.dtype)) and np.all(perm[r] >= 0)]
    return rows[0] if rows else None


@lru_cache(maxsize=8)
def _photon_rows_order(families, mesh_xy, to_packed):
    """The −q mirror's crossing of ``[q, N, N]`` rows between the canonical and packed
    photon orders (:func:`gw.photon_layout.photon_carry_order`), one cached program
    each way: to packed, the given fields joined along q; to canonical, the joined
    rows split back into ``n_fields`` stacks."""
    from .photon_layout import photon_carry_order
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    if to_packed:
        return jax.jit(lambda *fields: photon_carry_order(
            jnp.concatenate(fields)[None], families, mesh_xy, to_packed=True)[0], out_shardings=face)
    return jax.jit(lambda rows, n_fields: tuple(jnp.split(photon_carry_order(
        rows[None], families, mesh_xy, to_packed=False)[0], n_fields)),
        static_argnums=1, out_shardings=face)


@lru_cache(maxsize=64)
def _photon_mirror_mix(families, mesh_xy, inversion, n, keys, block, add):
    """One unfolded source ``block``'s Lorentz mix into its class ``keys``
    (``symmetry_maps.mix_lorentz_blocks``) as one cached program: the class's
    first block forms the totals, each later one is added onto them (``add``,
    donated), in the class's order."""
    from symmetry_maps import mix_lorentz_blocks
    sym, ops = families.plans[0].sym, np.full(int(n), int(inversion), dtype=np.int32)

    def mix(source, *total):
        mixed = mix_lorentz_blocks({block: source}, sym=sym, sym_idx=ops, mesh_xy=mesh_xy, keys=keys)
        return tuple(t + m for t, m in zip(total, mixed.values())) if add else tuple(mixed.values())
    return jax.jit(mix, donate_argnums=tuple(range(1, 1 + len(keys))) if add else ())


def _minus_q_mirror_photon(families, meta, mesh_xy):
    """The four-current ``chi_{-q}`` rows from ``chi_q`` rows by a unitary inversion, or ``None``.

    As :func:`_minus_q_mirror` on each Lorentz block, with the two families' plans on its endpoints and the inversion's
    Lorentz action (``symmetry_maps.mix_lorentz_blocks``), one source block
    at a time (``w_isdf.photon_blocks_full_q``'s restore), on the families'
    packed layout (the plans' centroid order).  Every field (χ and dχ/ds)
    given runs as one stack of rows.
    """
    from symmetry_maps import bgw_integer_q_to_fractional, unfold_isdf_operator
    from .photon_layout import _empty, _insert, photon_block_view
    plans, layout = tuple(families.plans), families.packed_layout
    rows = {_unitary_inversion(plan) for plan in plans}
    if len(rows) != 1 or None in rows:
        return None
    inversion = rows.pop()
    sym = plans[0].sym
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    q_frac = np.asarray(bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, kgrid))
    classes = tuple(tuple((C, D) for C in ((1, 2, 3) if a else (0,))
                          for D in ((1, 2, 3) if b else (0,))) for a in (0, 1) for b in (0, 1))

    def mirror(fields, q0, q1):
        n = (int(q1) - int(q0)) * len(fields)
        irr, ops = np.arange(n, dtype=np.int32), np.full(n, inversion, dtype=np.int32)
        frac = np.tile(q_frac[q0:q1], (len(fields), 1))
        # The plans act on each family's packed centroid order; the bank rows
        # are canonical, so the rows cross to the packed layout and back.
        rows = _photon_rows_order(families, mesh_xy, True)(*fields)
        out = _empty(n, layout, mesh_xy, rows.dtype)
        for keys in classes:
            left, right = plans[int(keys[0][0] != 0)], plans[int(keys[0][1] != 0)]
            total = ()
            for C, D in keys:
                source = unfold_isdf_operator(
                    photon_block_view(rows, layout, C, D, mesh_xy), irr_idx=irr, sym_idx=ops,
                    sym_perm=left.sym_perm, L_table=left.L_table,
                    right_sym_perm=right.sym_perm, right_L_table=right.L_table,
                    q_irr_frac=frac, mesh_xy=mesh_xy,
                    n_sym_spatial=int(left.n_sym_spatial),
                    axis_local_sym_perm=left.centroid_local_perm,
                    right_axis_local_sym_perm=right.centroid_local_perm)
                total = _photon_mirror_mix(families, mesh_xy, inversion, n, keys, (C, D),
                                           bool(total))(source, *total)
                del source
            for (C, D), block in zip(keys, total):
                out = _insert(out, block, layout, C, D, mesh_xy)
            del total
        del rows
        return _photon_rows_order(families, mesh_xy, False)(out, len(fields))
    return mirror


def _minus_q_mirror(plan, sym, meta, mesh_xy):
    """``chi_{-q}`` rows from ``chi_q`` rows by a unitary inversion, or ``None``.

    Inversion maps every q to -q, so ``chi_{-q} = U_I chi_q U_I^dagger`` is the
    parent row unfolded by that operation (``symmetry_maps.unfold_isdf_operator``
    with the plan's packed centroid tables, as V and W are restored).  Taken
    when the plan holds a unitary (spatial) operation equal to -1 whose centroid
    map is complete.  Returns ``mirror(fields, q0, q1)``: each field's rows of the
    parents ``[q0, q1)`` mirrored.
    """
    from symmetry_maps import bgw_integer_q_to_fractional, unfold_isdf_operator
    n_spatial = int(plan.n_sym_spatial)
    inversion = _unitary_inversion(plan)
    if inversion is None:
        return None
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    q_frac = np.asarray(bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, kgrid))

    def mirror(fields, q0, q1):
        n = int(q1) - int(q0)
        return tuple(unfold_isdf_operator(
            rows, irr_idx=np.arange(n, dtype=np.int32),
            sym_idx=np.full(n, inversion, dtype=np.int32),
            sym_perm=plan.sym_perm, L_table=plan.L_table, q_irr_frac=q_frac[q0:q1],
            mesh_xy=mesh_xy, n_sym_spatial=n_spatial,
            axis_local_sym_perm=plan.centroid_local_perm) for rows in fields)
    return mirror


def sample_q_width(face_bytes, nq, layer):
    """Parents per q span of a streamed sample's read (:func:`parent_span`): the span's value
    and slope rows, its unpack and the next prefetch; ``None`` for one span."""
    width = parent_span(nq, layer=layer, parent_bytes=3 * 2 * int(face_bytes))
    return None if width >= int(nq) else width


def _widest(mesh, held):
    """``[(span, panels)]`` of one line sample with every family's panels (and cross
    panels) zero-padded to the sample's widest span: the store holds one width."""
    widths = {(key, f): max(p[key][f].shape[-1] for _, p in held)
              for key in ("panels", "cross") for f in held[0][1].get(key, {})}
    for span, panels in held:
        out = dict(panels, **{key: dict(panels[key]) for key in ("panels", "cross") if key in panels})
        for (key, f), width in widths.items():
            if out[key][f].shape[-1] < width:
                out[key][f] = _pad_columns(mesh, out[key][f].ndim, width)(out[key][f])
        yield span, out


@lru_cache(maxsize=None)
def _pad_columns(mesh, ndim, width):
    spec = NamedSharding(mesh, P(*((None,) * (ndim - 2)), "x", "y"))
    return jax.jit(lambda a: jnp.pad(a, [(0, 0)] * (ndim - 1) + [(0, width - a.shape[-1])]),
                   out_shardings=spec)


class _RowWindow:
    """``raw[i, rows]`` of a q-span read: absolute response rows mapped to the span's own."""

    def __init__(self, value, first):
        self.value, self.first = value, int(first)

    def __getitem__(self, key):
        i, rows = key
        # The cached gather program, not eager indexing (several dispatches per take).
        return _MemberRows(self.value, 0)[i, np.asarray(rows) - self.first]


class _MemberRows:
    """``raw[i, rows]`` of one group member, read from the group carry ``[2m, q, μ, ν]``."""

    def __init__(self, carry, first):
        self.carry, self.first = carry, int(first)

    def __getitem__(self, key):
        i, rows = key
        sharding = getattr(self.carry, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            return _member_rows(self.carry, self.first + int(i), np.asarray(rows))
        # The member index and its rows repeat across groups: placed once per
        # distinct value instead of two host transfers on every take.
        from common.collectives import device_put_process_local_cached
        rep = NamedSharding(sharding.mesh, P())
        return _member_rows(self.carry,
                            device_put_process_local_cached(np.int64(self.first + int(i)), rep),
                            device_put_process_local_cached(np.asarray(rows), rep))


def _parent_row(value, i, mesh_xy):
    """``value[i:i+1]`` on device: the row index is a placed constant, not an eager slice's host start."""
    from common.collectives import device_put_process_local_cached
    return _parent_row_program(value, device_put_process_local_cached(
        np.int32(i), NamedSharding(mesh_xy, P())))


@jax.jit
def _parent_row_program(value, i):
    return jax.lax.dynamic_slice_in_dim(value, i, 1, axis=0)


@jax.jit
def _member_rows(carry, i, rows):
    """One program per carry shape and row count; eager indexing is several."""
    return carry[i, rows]


def _member_block(carry, first, mesh_xy):
    """``carry[first:first+2]``, one member's value and slope rows out of the group carry, so
    the carry can be freed while that member is solved (the start a placed constant)."""
    from common.collectives import device_put_process_local_cached
    return _member_block_program(carry, device_put_process_local_cached(
        np.int32(first), NamedSharding(mesh_xy, P())))


@jax.jit
def _member_block_program(carry, first):
    return jax.lax.dynamic_slice_in_dim(carry, first, 2, axis=0)


#: Face-sized arrays one parent's partner unfold holds at once per field (the
#: tile's gathered rows, their packed copy, the unfolded block, the mixed block
#: and the output on the four-current route), the bound the partner tile is
#: sized from.
MIRROR_TILE_STACKS = 5


def _dyson_phase(dyson, solve_slope, roots, held_roots, mesh_xy, layout, *, nq, n, extra=(),
                 mirrored=False):
    """(resident, workspace) bytes per rank of the sample Dyson phase beside a group's carry.

    Counted as :func:`_bank_execution` admits them: the compiled arguments
    resident, the outputs, temporaries and native solver workspace on top.
    One ``sample_dyson`` pair ``(H held, chi, dchi) -> (Wc, dWc/ds)``; the
    photon pair also takes ``extra`` (the contact) and W_inf - V on the face.
    A deck without a pair (every face deck): ``sample_dyson`` takes ``(H, chi)``, donating
    chi, while the slope's dchi rows are already held (one more face stack); ``sample_slope``
    then takes ``(H, Wc, dchi)``.  Beside the operands: temporaries + outputs - donated
    aliases, the dense workspace and the Dyson LU's measured workspace on cuSOLVERMp.  Both at the full parent span ``nq``, the largest a sample solves.  ``mirrored``:
    a partner's χ and dχ/ds rows are collected from unfolded tiles before they
    are concatenated (two more stacks, plus one tile's unfold).
    """
    from runtime.aot_memory import compiled_new_bytes
    sharding = getattr(roots, "sharding", None)
    if sharding is None or not hasattr(roots, "shape"):
        return 0, 0
    face = jax.ShapeDtypeStruct((int(nq), int(n), int(n)), jnp.complex128, sharding=sharding)
    h = jax.ShapeDtypeStruct((int(nq),) + tuple(roots.shape[1:]), roots.dtype, sharding=sharding)
    native = response_dense_workspace(mesh_xy, int(n), int(nq), layout, with_eigh=False)["total"]
    held = 16 * int(nq) * int(n) * int(n) // int(mesh_xy.size)
    if mirrored:
        from runtime.tiles import TILE_BYTES
        extra_bytes = 2 * held + min(int(TILE_BYTES), 2 * MIRROR_TILE_STACKS * held)
    else:
        extra_bytes = 0
    phases = []
    pair = dyson.pair("face")
    stages = (((pair, (jax.ShapeDtypeStruct(held_roots.shape, held_roots.dtype, sharding=held_roots.sharding),
                       face, face) + tuple(extra) + ((face,) if extra else ()), extra_bytes),)
              if pair is not None else
              ((dyson.value, (h, face) + tuple(extra), held + extra_bytes),
               (solve_slope, (h, face, face), extra_bytes)))
    lu = getattr(dyson.value, "lu", None)
    if lu is not None and lu.backend == "cusolvermp":
        from distrib_la import lu_workspace_bytes
        native += lu_workspace_bytes(mesh_xy, int(n), np.complex128)
    for kernel, args, kept in stages:
        executable = _compiled(kernel, args)
        memory = executable.memory_analysis()
        if memory is None:
            raise ValueError("GATE response_capacity: compiled memory unavailable")
        phases.append((int(memory.argument_size_in_bytes) + kept,
                       compiled_new_bytes(executable) + int(native)))
    return tuple(max(p[i] for p in phases) for i in (0, 1))


def response_group_size(meta, mesh_xy, *, n_samples, carry_per_sample, stream_workspace,
                        selection=(0, 0)):
    """Largest sample group whose carry and stream workspace fit, and its room.

    One route and no dial: every sample in one group when it fits (symmetric
    decks), otherwise the largest group that does (about four on a
    two-component deck without q symmetry, where the carry is G/2 Green tiles).
    ``selection`` is the (resident, workspace) bytes that run beside the
    group's carry after its stream: the line selection's reservation plus the
    sample Dyson value and slope solves admitted inside it (:func:`_dyson_phase`).  The group must fit the
    capacity ledger (the deck budget ``memory_per_device_gb`` less the inherited
    peak, the reserved live stages and the runtime reserve): the one size that
    follows the budget, because a larger group buys more than 10 % per map and
    the group moves no number (the rule is fitted on every sample at once).  On
    the device the stream and the selection are two phases, so a group's new
    bytes are its carry plus the larger phase.  Returns ``(size, fixed, room,
    live)``: the group's bytes beside its carry, the ledger's room, and the
    budget less that room.  Every rank computes the same group.
    """
    from common.gpu_utils import device_budget_bytes
    ledger = meta.tangential_capacity
    device_room = ledger.room_bytes_per_rank(ledger.live_stages)
    fixed = max(int(stream_workspace), int(selection[0]) + int(selection[1]))
    fits = lambda g: (ledger.preview(resident_bytes_per_rank=g*carry_per_sample+int(selection[0]),
        workspace_bytes_per_rank=max(stream_workspace, int(selection[1])),
        concurrent_with=ledger.live_stages)["device_budget_status"] == "PASS"
        and g*carry_per_sample + fixed <= device_room)
    size = 1   # at one sample the admission refuses with the actual compiled bytes
    while size < n_samples and fits(size+1):
        size += 1
    return size, fixed, device_room, int(device_budget_bytes()) - device_room


def response_bank_residence(meta, *, segments, n_samples, carry_per_sample, group_size, host_reserved=0):
    """Where this map's χ bank (value and slope of every sample) lives.

    The rule of the tangential bank (``tangential_screening._bank_residence``):
    on the devices when every sample fits one group (``group_size``, from the
    capacity ledger and the compiled stream); otherwise streamed, the stream
    running once with every sample and each row pass written out
    (``file_io.slab_io.StreamedBank``): to host memory when the bank takes at most
    half of this process's host budget beside ``host_reserved`` (a host-tier
    W bank), else to per-rank files.  A stream without the row-pass engine
    (``segments`` ``None``: the full-k Green route) keeps sample groups on
    the devices.  Every rank computes the same answer.
    Returns ``(residence, receipt)``.
    """
    from common.gpu_utils import host_bytes_per_process
    total = int(n_samples) * int(carry_per_sample)
    receipt = dict(bytes_per_rank=total, group_size=int(group_size))
    if int(group_size) >= int(n_samples):
        return "device", dict(receipt, reason="every sample fits one group on the devices")
    if segments is None:
        return "device", dict(receipt, reason="no row-pass engine on this stream; sample groups")
    receipt["half_host_budget_bytes_per_rank"] = int(host_bytes_per_process()) // 2 - int(host_reserved)
    if total <= receipt["half_host_budget_bytes_per_rank"]:
        return "host", dict(receipt, reason="streamed; fits half the host budget")
    return "file", dict(receipt, reason="streamed; exceeds half the host budget")


def produce_sample_bank(wfns, meta, config, *, mesh_xy, sym, sample_plan, bank_io,
                        vertex=None, contact=None, direct_head=None, print_fn=print):
    """Stage A: integrate value and derivative together, one frequency at a time.

    A fitted line sample off the imaginary axis stores only its direction
    panels: its directions are selected from W(z) itself and the constructor
    reads nothing else from it (``gw.tangential_directions.LineSelection``).
    Samples on the imaginary axis and held samples are stored dense.
    """
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
        tr_odd_census = bool(config.debug.sigma_freq_debug_output)
        if ordered != bool(header.get("ordered")):
            raise ValueError("GATE response_representation: got: a bank whose ordered layout disagrees "
                             "with the measured time reversal; fix: rebuild the bank")
        # Line-panel samples [p0, p1). On the ordered route each one also needs
        # its minus-q partner W_q(-conj z) = conj(W_{-q}(z)): the exact -q rows of
        # the same stream, conjugated, through parent q's own V (and contact).
        # An imaginary node is its own partner.
        p0, p1 = (int(v) for v in header["line_panels"]["sample_span"])
        partnered = ordered and p1 > p0
        if partnered:
            from symmetry_maps import q_negation_index
            negative = np.asarray(q_negation_index((int(meta.nkx), int(meta.nky), int(meta.nkz))), dtype=np.int64)
            partner_qids = negative[qids]
            partner_provenance = bank_io.get("minus_q_operator_provenance")
            if partner_provenance is None:
                partner_provenance = dict(
                    coulomb=bank_io["coulomb"],
                    state_identity=bank_io["identity"],
                    operator="same original parent V and moment operator as Wc and M0..M3")
            receipt["minus_q_partner"] = dict(
                sample_span=[p0, p1], value="W_q(-conj z) = conj(W_{-q}(z))",
                operator_provenance=partner_provenance,
                original_parent_count=len(qids),
                full_q_rows=len(set(qids.tolist()+partner_qids.tolist())),
                green_stream="union of exact q and minus-q output rows in the same response panel",
                dyson="original parent V/contact for both; consumed by the line selection, never stored")

        # A unitary operation that maps every q to -q (inversion) gives the
        # partner rows from the parent rows on load (TASTE 97), so the stream
        # carries no -q rows; otherwise they are streamed.
        mirror = None
        if partnered and vertex is not None:
            mirror = _minus_q_mirror_photon(vertex.families, meta, mesh_xy)
        elif partnered and wfns.green_parent is not None:
            mirror = _minus_q_mirror(wfns.green_parent.plan, sym, meta, mesh_xy)
        if mirror is not None:
            receipt["minus_q_partner"]["green_stream"] = (
                "parent q rows only; -q rows by the unitary inversion's unfold of the parent rows")

        def panel_rows(first, last):
            rows = qids[first:last].tolist()
            if partnered and mirror is None:
                rows = list(dict.fromkeys(rows+partner_qids[first:last].tolist()))
            return tuple(rows)

        def committed(sample):
            if p0 <= sample < p1:
                return bool(np.asarray(header["line_written"], bool)[:, sample-p0].all())
            row = dense_sample_rows(header, (sample,))[0]
            return bool(np.asarray(header["sample_written"], bool)[:, row].all())

        n = meta.mu_basis.n_packed if vertex is None else vertex.n
        if ordered:
            receipt["ordered"] = True
        started = time.monotonic()
        execute = _bank_execution(meta, mesh_xy, receipt, config, photon=vertex is not None)
        ledger = meta.tangential_capacity
        ambient = ledger.live_stages
    from file_io.tangential_store import (dense_sample_rows, read_tangential_bank,
                                           tangential_bank_writer)
    from .tangential_execution import line_selection_execution, line_selection_room
    response_rows = panel_rows(0, len(qids))
    row_index = {q: i for i, q in enumerate(response_rows)}
    face_bytes = 16*n*n//mesh_xy.size
    caller_live = ambient
    dyson, solve_slope, _, receipt["algebra"] = response_algebra(meta, config,
        mesh_xy=mesh_xy, n=n, photon=vertex is not None)
    if vertex is None:
        # V is frequency independent. Its all-P root bank is small compared
        # with either full-zone Green and reuses the existing batched solver.
        roots, inverse, _ = _coulomb_batch(meta, config, bank_io, mesh_xy,
                                         (0, len(qids)), execute)
        del inverse
        root_stage, _ = _reserve(meta, "coulomb_roots", len(qids)*face_bytes)
        ambient += (root_stage,)
        # The roots in the layout of every sample's Dyson pair, laid out once.
        held = dyson.place(roots)
        if held is not roots:
            root_stage, _ = _reserve(meta, "coulomb_roots_batch",
                16*n*n*int(np.prod(held.sharding.shard_shape(held.shape)[:-2])))
            ambient += (root_stage,)
    else:
        roots = held = bank_io["photon_v"]
    carry_per_sample = 2*len(response_rows)*face_bytes
    selection = None
    # The deck's eigh request for a face selection plan (resolve_linalg).
    from .gw_config import linalg_resolution
    face_eigh_backend = linalg_resolution({"linalg": config.backend.linalg}).eigh_backend

    def line_route(width, carry):
        """The line selection of ``width`` parents beside ``carry`` bytes: (selection,
        resident, workspace), its route admitted at ``width`` (local or face)."""
        ledger.live_stages = ambient
        rows = ([int(meta.mu_basis.n_packed)] if vertex is None else
                [int(b.n_packed) * (3 if f else 1) for f, b in enumerate(bank_io["mu_bases"])])
        execution, resident, workspace = line_selection_execution(
            rows, mesh=mesh_xy, ledger=ledger, nq=width, carry=carry,
            eigh_backend=face_eigh_backend)
        # A face selection's eigh stacks are decided in the room its admission leaves.
        room = line_selection_room(ledger, resident, carry=carry) if execution == "face" else None
        return line_selector(execution, width, room), resident, workspace

    def line_selector(execution, width, room):
        if vertex is None:
            from .tangential_directions import charge_line_selection
            return charge_line_selection(meta, mesh_xy=mesh_xy, ordered=ordered,
                                         execution=execution, nq=width,
                                         eigh_backend=face_eigh_backend, room=room)
        from .tangential_sectors import sector_line_selection
        return sector_line_selection(bank_io, meta, mesh_xy=mesh_xy, execution=execution, nq=width,
                                     eigh_backend=face_eigh_backend, room=room)

    if p1 > p0:
        # The line selection runs beside one sample's carry; its route is
        # admitted here so the group size below leaves room for it.
        selection, selection_resident, selection_workspace = line_route(len(qids), carry_per_sample)
        receipt["line_selection"] = dict(execution=selection.execution, samples=[p0, p1],
            resident_bytes_per_rank=selection_resident, workspace_bytes_per_rank=selection_workspace)
    # A q-local selection reads whole matrices per rank: its line samples leave
    # the Dyson pair in the batch layout, with no exchange to the face and back.
    line_layout = ("batch" if selection is not None and selection.execution == "local"
                   and dyson.pair("batch") is not None else "face")
    with timing.section('bank.window_geometry', announce=True,
                              label="tangential frequency rule construction"):
        support = response_support(wfns, meta, sample_plan, receipt, print_fn=print_fn)
        ledger.live_stages = ambient
        segments = stream_segments(wfns, meta, mesh_xy, len(response_rows), vertex)
        # After its stream, a group's carry holds while each sample's Dyson
        # value and slope solve run at every parent at once; a line sample's
        # solves run inside its line-selection reservation. Count both as the
        # admissions will (their compiled executables), summed.
        chosen = (0, 0) if selection is None else (selection_resident, selection_workspace)
        chosen = tuple(a + b for a, b in zip(chosen, _dyson_phase(
            dyson, solve_slope, roots, held, mesh_xy, receipt["algebra"]["linalg"], nq=len(qids), n=n,
            extra=() if vertex is None else (contact,), mirrored=mirror is not None)))
        tables_reserved = []

        def reserve_kconv_tables():
            """The kconv tables a stream placed stay on the devices beside every later
            phase: a live stage, counted once (after the first stream program is built)."""
            nonlocal ambient
            if vertex is None and not tables_reserved:
                from .w_isdf import charge_kconv_table_bytes
                tables, _ = _reserve(meta, "kconv_tables", charge_kconv_table_bytes(), book=False)
                tables_reserved.append(tables)
                ambient += (tables,)
                ledger.live_stages = ambient
        from runtime.aot_memory import agreed_chunk, check_chunk
        from lxkit import memory
        scratch = _stream_scratch(wfns, meta, mesh_xy, vertex)

        def device_groups():
            """The largest sample group on the devices: the ledger's, then the compiled check
            (runtime.aot_memory.check_chunk: the carry a donated argument the caller
            allocates, the mode-11 scratch a run-time draw)."""
            # Price the stream once at "every sample in one group"; the compiled
            # temporaries do not grow with the group, only the donated carry does.
            workspace, whole = _stream_workspace(wfns, meta, mesh_xy, support, q_ids=response_rows,
                n_outputs=2*len(z), ordered=ordered, vertex=vertex)
            reserve_kconv_tables()
            with timing.section('bank.plan.direct'):
                size, fixed, room, live = response_group_size(meta, mesh_xy, n_samples=len(z),
                    carry_per_sample=carry_per_sample, stream_workspace=workspace,
                    selection=chosen)
            with timing.section('bank.memcheck.direct'):
                check = check_chunk(
                    size, stage="response direct stream",
                    build=lambda g: _stream_executable(wfns, meta, mesh_xy, support,
                        q_ids=response_rows, n_outputs=2*g, ordered=ordered, vertex=vertex),
                    compiled=whole if size == len(z) else None,
                    fixed=fixed, per_unit=carry_per_sample, room=room,
                    extra=lambda g, _: g*carry_per_sample + scratch)
            # The compiled figure is read on each rank; the group (and so the
            # residence below) is the smallest of them, agreed by one all-gather.
            return check, agreed_chunk(check.chunk), live, room

        # One rule, never a refusal (TASTE 96 and the owner's "never refuse"):
        # every sample in one group on the devices when it fits; else one group
        # streamed through SlabIO's tier; device sample groups remain only as
        # the fallback when the disk or quota refuses the bank on any rank, and
        # for a stream without the row-pass engine (the full-k route). Every
        # input to the choice is the same on every rank: the ledger room, the
        # agreed compiled check, the agreed host budget, the agreed store creation.
        check = None
        if segments is not None and len(z)*carry_per_sample > ledger.room_bytes_per_rank(ambient):
            # Every sample's carry alone exceeds the room: no group of all of
            # them can fit, so the stream is not compiled whole to learn it.
            group_size = 0
        else:
            check, group_size, live, room = device_groups()
        host_reserved = (bank_io["path"].payload_bytes_per_rank()
                         if getattr(bank_io["path"], "memory_kind", None) == "host" else 0)
        residence, residence_receipt = response_bank_residence(
            meta, segments=segments, n_samples=len(z), carry_per_sample=carry_per_sample,
            group_size=group_size, host_reserved=host_reserved)
        stream_bank = None
        if residence != "device":
            from file_io.slab_io import StreamedBank
            # The store reserves every byte at creation, so a bank the disk or
            # the quota cannot hold is refused on every rank before any compute.
            # Then the samples stream in disk groups, each try half the last,
            # one bank reused group after group; once a disk group would be no
            # larger than the device group, the samples run in groups on the
            # devices (a smaller group, never a refusal).
            disk_group = len(z)
            while True:
                stream_bank = StreamedBank(mesh_xy, root=bank_io["root"], label="chi",
                    kind=residence, n_out=2*disk_group, q=len(response_rows),
                    segments=segments[0], tile=segments[1])
                if stream_bank.fits:
                    break
                stream_bank = None
                if check is None:
                    check, group_size, live, room = device_groups()
                if -(-disk_group // 2) <= group_size:
                    residence = "device"
                    residence_receipt = dict(residence_receipt, reason="streamed bank refused by "
                        "the filesystem (capacity); sample groups on the devices")
                    break
                disk_group = -(-disk_group // 2)
        if residence == "device":
            receipt["group_check"] = dict(chunk=check.chunk, agreed=group_size, analytic=check.analytic,
                compiled=check.compiled_bytes, price=check.price, live=live, room=room,
                recompiled=check.recompiled, seconds=check.seconds)
        else:
            # Streamed: one group of every sample the disk holds; the devices
            # hold one segment carry being computed and one being drained, then
            # one sample's value and slope being read, unpacked and prefetched.
            group_size = disk_group
            if disk_group < len(z):
                residence_receipt = dict(residence_receipt, reason=residence_receipt["reason"]
                                         + f"; disk groups of {disk_group} samples")
        del check
        receipt["bank_residence"] = dict(residence_receipt, residence=residence)
        rules = response_quadrature(meta, sample_plan, receipt, support,
                                    group_size=group_size, print_fn=print_fn)
        if stream_bank is not None:
            # Slots: the largest rule group's members (empty slots add zeros nobody reads).
            rules["slots"] = max(len(g["members"]) for g in rules["groups"])
            pass_carry = 2*rules["slots"]*len(response_rows)*16*max(r*c for r, c, _ in segments[0])
            # The first segment's program places the kconv tables the rest share.
            response_stream(wfns, meta, mesh_xy=mesh_xy, q_ids=response_rows,
                n_outputs=2*rules["slots"], pair_mode="direct", bank_carry=True, ordered=ordered,
                vertex=vertex, band_ranges=rules["band_ranges"], stream_pass=0)
            reserve_kconv_tables()
            finish = segments[2]
            receipt["bank_residence"].update(stream_bank.receipt(),
                device_bytes_per_rank=max(stream_bank.in_flight*pass_carry, 3*carry_per_sample))
    # The group accumulator is all-P sharded. Dense work and slab I/O batch
    # the irreducible parents of one frequency, with their own admission.
    progress = LoopProgress(len(z), title="response frequency integration",
                            item_name="frequency", max_updates=len(z)).start()

    def read_constant(span, bank_handle):
        """The photon W_inf - V of parents ``span`` (0 for charge), read from the bank."""
        if vertex is None:
            return 0.
        io_started = time.monotonic()
        constant = read_tangential_bank(bank_handle, span, meta=meta, header=header,
                                         fields=("constant",))["constant"]
        receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
        return constant

    def solve(raw, partner, q0, q1, bank_handle, sample, need_value=True, layout="face",
              constant=None):
        """W and dW/ds of parents [q0, q1) at one sample, as the bank stores them.

        ``layout='batch'`` (a charge line sample beside a q-local selection)
        returns both in the batch layout.  ``constant``: the span's W_inf - V
        when the caller holds it (a line span's two solves), else read here.
        """
        mirrored = partner and mirror is not None
        selected = (partner_qids if partner and not mirrored else qids)[q0:q1]
        rows = np.asarray([row_index[int(q)] for q in selected])

        def chi_rows(fields):
            """The rows of each field (0 χ, 1 dχ/ds), every field through one mirror call."""
            if not mirrored:
                return tuple(raw[i, rows] for i in fields)
            # The partner rows are formed in parent tiles (runtime.tiles): one
            # tile's unfold transients beside the collected stacks.
            from runtime.tiles import tile_units
            step = tile_units(MIRROR_TILE_STACKS * len(fields) * face_bytes, len(rows))
            parts = [mirror(tuple(raw[i, rows[a:a+step]] for i in fields), q0+a,
                            q0+min(a+step, len(rows))) for a in range(0, len(rows), step)]
            return parts[0] if len(parts) == 1 else tuple(
                jnp.concatenate(column, axis=0) for column in zip(*parts))
        span = (int(q0), int(q1))
        # The whole span solves on the held roots; a slice would be one more face per parent.
        h = roots if span == (0, len(qids)) else roots[q0:q1]
        if constant is None:
            constant = read_constant(span, bank_handle)
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
        # The rows this sample gathers from the carry (and a partner's conjugate beside them)
        # are live until its Dyson programs consume them: a named reservation every room sees.
        fields = 2 if need_value else 1
        live_rows = ledger.live_stages
        rows_stage, _ = _reserve(meta, "sample_rows", 0, (fields + bool(partner)) * len(rows) * face_bytes,
                                 book=False)
        ledger.live_stages = live_rows + (rows_stage,)
        chi, *chi_value = chi_rows((1, 0) if need_value else (1,))
        if partner:
            chi = jnp.conj(chi)
        if need_value:
            chi_value, = chi_value
            if partner:
                chi_value = jnp.conj(chi_value)
        # The Dyson programs consume their χ rows (donated); the debug census reads them after.
        census_chi = chi_value + 0 if need_value and ordered and tr_odd_census and not partner else None
        if need_value and dyson.pair(layout) is not None:
            # Wc and dWc/ds in one program: on the held roots (charge), or
            # with the contact and W_inf - V (photon).
            value, slope = execute(dyson.pair(layout), (held if span == (0, len(qids))
                                   else dyson.place(h), chi_value, chi)
                                   + (() if vertex is None else (contact, constant)), "sample_dyson")
        else:
            if need_value:
                value = execute(dyson.value, (h,chi_value)+(() if vertex is None else (contact,)),
                                "sample_dyson")
                if vertex is not None:          # charge: constant 0 (no copy of Wc)
                    value = value - constant
            else:
                io_started = time.monotonic()
                saved = read_tangential_bank(bank_handle, span, meta=meta, header=header,
                    sample_span=(sample,sample+1), fields=("Wc",))
                receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                value = saved["Wc"][:,0]
                del saved
                if head_update is not None:
                    value = gamma_add(value, -head_update)
            w = value if vertex is None else value+constant
            slope = execute(solve_slope, (h, w, chi), "sample_slope")
            del w
        del chi
        ledger.live_stages = live_rows
        if head_update is not None:
            coefficient = (direct_head["dWc_minus_q_ds"][sample]
                           if partner else direct_head["dWc_ds"][sample])
            slope = gamma_add(slope, coefficient)
            value = gamma_add(value, head_update)
        if need_value and not partner:
            if vertex is not None:
                _photon_sample_norms(receipt,value,q0,sample,bank_io["photon_layout"],mesh_xy)
            for iq in range(q0,q1):
                # Both censuses read only self-negative (TRIM) parents at real
                # supports (Re z = 0); test that on the host first, so no other
                # parent or sample dispatches a device slice (0.2 s per map on
                # Fe 4^3 charge), and a batch-layout line pair is never sliced.
                if vertex is None and z[sample].real == 0 and _self_negative(int(qids[iq]),meta):
                    part = slice(iq-q0,iq-q0+1)
                    _reciprocity_census(receipt,_parent_row(value,iq-q0,mesh_xy),z[sample:sample+1],
                                        int(qids[iq]),iq,meta)
                    # Diagnostic only (nothing reads it): an extra Dyson solve per
                    # imaginary sample at every TRIM parent, so it runs under the
                    # debug-output dial, never in the production prefactor.
                    if ordered and tr_odd_census:
                        _tr_odd_census(receipt,dyson.value,h[part],census_chi[part],value[part],z[sample:sample+1],int(qids[iq]))
        return value, slope

    def dense_spans(sample):
        """``[(q0, q1, need_value, need_slope), ...]`` of a dense sample still to write: a fresh
        frequency is one q_irr slab; partial restarts keep contiguous rows with identical
        value/slope masks together."""
        marked = np.asarray(header["sample_written"], bool)[:, dense_sample_rows(header, (sample,))[0]]
        edges = np.r_[0, 1+np.flatnonzero(np.any(marked[1:] != marked[:-1], axis=1)), len(qids)]
        return [(int(q0), int(q1), bool(~marked[q0][0]), bool(~marked[q0][1]))
                for q0, q1 in zip(edges[:-1], edges[1:]) if not marked[q0].all()]

    # On the streamed tier a sample's value and slope rows come back in q spans
    # (the read, its unpack and the next prefetch live at once), so a sample
    # larger than a card is solved span by span; one span when it fits.  Under
    # local linalg a span is at least P parents, one whole matrix per rank.
    q_width = None
    if stream_bank is not None:
        q_width = sample_q_width(face_bytes, len(qids), _parent_layer(receipt, mesh_xy))
        receipt["bank_residence"]["q_width"] = q_width or len(qids)
    # A line sample selects span by span too when each parent's minus-q partner
    # comes from its own rows (no partner, or the inversion mirror); each span's
    # route is admitted at its width and the panels written per span.
    line_width, line_selections = None, {}

    def line_spans():
        width = line_width or len(qids)
        return [(a, min(len(qids), a + width)) for a in range(0, len(qids), width)]
    if q_width is not None and p1 > p0 and (not partnered or mirror is not None):
        line_width = q_width
        selection, selection_resident, selection_workspace = line_route(line_width, 3*2*line_width*face_bytes)
        line_selections = {line_width: selection}
        if len(qids) % line_width:
            # The remainder span runs under the full span's admitted route: its execution and room.
            rest = len(qids) % line_width
            line_selections[rest] = line_selector(selection.execution, rest, selection.room)
        receipt["line_selection"].update(execution=selection.execution, q_width=line_width,
            resident_bytes_per_rank=selection_resident, workspace_bytes_per_rank=selection_workspace)
        line_layout = ("batch" if selection.execution == "local" and dyson.pair("batch") is not None
                       else "face")
    if q_width is not None and (line_width is not None or p1 == p0):
        # Every read is one q span: its value and slope rows, the unpack and the prefetch.
        receipt["bank_residence"]["device_bytes_per_rank"] = max(stream_bank.in_flight*pass_carry,
                                                                 3*2*q_width*face_bytes)

    for group in rules["groups"]:
        members = [int(m) for m in group["members"]]
        if all(committed(m) for m in members):
            for _ in members:
                progress.step()
            continue
        ledger.live_stages = ambient
        name, _ = _reserve(meta, "bank_outputs", rules["slots"]*carry_per_sample if stream_bank is None
                           else receipt["bank_residence"]["device_bytes_per_rank"])
        ledger.live_stages = ambient+(name,)
        if stream_bank is None:
            raw_group = integrate_response_group(wfns, meta, mesh_xy, rules, group,
                q_ids=response_rows, execute=execute, receipt=receipt,
                ordered=ordered, vertex=vertex)
            # Read the member's value and slope rows from the group carry at
            # each solve; no copy of its whole carry is made.
            take = lambda row: _MemberRows(raw_group, 2*row)
        else:
            # Each pass's rows of every member go to the bank as they finish;
            # each member's value and slope come back one ahead of its solve.
            fresh = [(row, sample) for row, sample in enumerate(members) if not committed(sample)]
            integrate_response_group(wfns, meta, mesh_xy, rules, group, q_ids=response_rows,
                execute=execute, receipt=receipt, ordered=ordered, vertex=vertex, bank=stream_bank,
                outputs=[(2*row+k, 2*row+k) for row, _ in fresh for k in (0, 1)])
            with timing.section('bank.stream_commit'):
                stream_bank.commit()
            # A sample larger than its q width reads one q span per solve (the
            # loop's own order); a line sample without line spans reads every parent.
            raw_group = stream_bank.reader([
                (2*row, 2*row+2, span) for row, sample in fresh
                for span in ([None] if q_width is None or (p0 <= sample < p1 and line_width is None) else
                             line_spans() if p0 <= sample < p1 else
                             [(a, min(q1, a+q_width)) for q0, q1, _, _ in dense_spans(sample)
                              for a in range(q0, q1, q_width)])])
            take = ((lambda row: next(raw_group)) if finish is None
                    else (lambda row: finish(next(raw_group))))
        io_started = time.monotonic()
        # One collective writer transaction per group, not per sample.
        with tangential_bank_writer(bank_io["path"], meta=meta,
                expected_identity=bank_io["identity"], mesh_xy=mesh_xy) as (bank_handle, header, write):
            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
            # Dense members first, line members (the selection, the bank's largest phase) last;
            # on the devices the last member's rows leave the group carry, which is freed
            # before that member's solves.
            order = sorted(enumerate(members), key=lambda m: p0 <= m[1] < p1)
            last = ([row for row, sample in order if not committed(sample)] or [None])[-1]
            for row, sample in order:
                if committed(sample):
                    progress.step()
                    continue
                if stream_bank is None and row == last:
                    raw = _MemberRows(_member_block(raw_group, 2 * row, mesh_xy), 0)
                    raw_group = take = None
                else:
                    raw = take(row) if q_width is None or (p0 <= sample < p1 and line_width is None) else None
                if p0 <= sample < p1:
                    # Select from W(z) itself, then act with the minus-q partner on
                    # the same directions; only the panels reach the bank.
                    if np.asarray(header["line_written"], bool)[:, sample-p0].any():
                        raise ValueError(f"GATE response_line_panel: sample {sample} is partly committed; "
                                         "a line sample commits every parent at once; fix: rebuild the bank")
                    # The selection's phases are priced by their own programs (below).
                    stage, _ = _reserve(meta, "line_selection", selection_resident,
                                                    selection_workspace, book=False)
                    live = ledger.live_stages
                    ledger.live_stages = live + (stage,)
                    # line_selection times the selection alone: this sample's Dyson
                    # solves stay under their own dispatch keys, so the keys partition.
                    solved = lambda: sum(receipt["seconds"].get(k + "_dispatch", 0.)
                                         for k in ("sample_dyson", "sample_slope"))
                    started_selection = time.monotonic() - solved()
                    # A span's panels are as wide as its own widest parent; they are
                    # held (narrow, [q, F, rows, r]) and written at the sample's widest.
                    span_panels = []
                    for span in line_spans():
                        sel = line_selections.get(span[1] - span[0], selection)
                        rows = raw if line_width is None else _RowWindow(take(row), span[0])
                        # The span's W_inf - V serves both orientations' solves.
                        constant = read_constant(span, bank_handle)
                        # The pair goes to the selection as a list it empties: W and dW/ds are
                        # freed once their selection blocks are cut.
                        pair = list(solve(rows, 0, *span, bank_handle, sample, layout=line_layout,
                                          constant=constant))
                        if not ordered and line_width is None:
                            rows = raw = None          # every row of this member is read
                        # Each phase over what is live at its entry: the cut blocks and normal
                        # matrices (the admitted resident) and the eigh stack's own program
                        # (place()); the mirror cuts the partner's blocks; the panels are to_face
                        # compiled at their fields.
                        memory.book(memory.priced(selection_resident + sel.eigh_bytes(), "line selection: "
                            "cut blocks (admitted) + the eigh stack's program", increment=True), stage='bank.line_select')
                        with timing.section('bank.line_select'):
                            lines = sel.select(sample, pair)
                        if ordered:
                            pair = list(solve(rows, 1, *span, bank_handle, sample, layout=line_layout,
                                              constant=constant))
                            if line_width is None:
                                rows = raw = None      # every row of this member is read
                            memory.book(memory.priced(selection_resident, "line mirror: the partner's cut "
                                "blocks (admitted)", increment=True), stage='bank.line_mirror')
                            with timing.section('bank.line_mirror'):
                                sel.mirror(sample, lines, pair)
                        del pair
                        del constant
                        from runtime.aot_memory import compiled_new_bytes
                        memory.book(memory.priced(sum(compiled_new_bytes(_compiled(sel.to_face, fields))
                                                      for _, _, fields in sel.panel_fields(lines)),
                            "line panels: to_face compiled at its fields", increment=True), stage='bank.line_panels')
                        with timing.section('bank.line_panels'):
                            span_panels.append((span, sel.panels(sample, lines)))
                        del lines, rows
                    receipt["seconds"]["line_selection"] = (receipt["seconds"].get("line_selection", 0.)
                        + time.monotonic() - solved() - started_selection)
                    io_started = time.monotonic()
                    with timing.section('bank.line_write'):
                        for span, panels in _widest(mesh_xy, span_panels):
                            write(q_span=span, line=panels)
                    del span_panels, panels
                    receipt["seconds"]["io"] = receipt["seconds"].get("io", 0.) + time.monotonic() - io_started
                    ledger.live_stages = live
                else:
                    for q0, q1, need_value, need_slope in dense_spans(sample):
                        width = q_width or (q1 - q0)
                        for a in range(q0, q1, width):
                            span = (int(a), int(min(q1, a + width)))
                            rows = raw if q_width is None else _RowWindow(take(row), a)
                            value, slope = solve(rows, 0, *span, bank_handle, sample,
                                                 need_value=need_value)
                            del rows
                            io_started = time.monotonic()
                            if need_slope:
                                write(q_span=span, sample_span=(sample,sample+1), dWc_ds=slope[:,None])
                            if need_value:
                                write(q_span=span, sample_span=(sample,sample+1), Wc=value[:,None])
                            receipt["seconds"]["io"] = receipt["seconds"].get("io",0.)+time.monotonic()-io_started
                            del value, slope
                receipt["batches"].append(dict(sample=sample, group=members))
                del raw
                progress.step()
            io_started = time.monotonic()
        receipt["seconds"]["io"] += time.monotonic()-io_started
        del raw_group, take
    progress.finish()
    if stream_bank is not None:
        for key, value in stream_bank.seconds.items():
            receipt["seconds"]["bank_" + key] = value
        receipt["bank_residence"]["bounced_records"] = stream_bank.bounced
        receipt["bank_residence"]["rereads"] = stream_bank.rereads
        stream_bank.release()
    if jax.process_index() == 0:
        passes = _direct_passes(wfns, meta, mesh_xy, len(response_rows)) if vertex is None else None
        layout = ("" if passes is None else
                  f", {len(passes.passes)} row pass(es) of <= {max(r for _, r in passes.passes)} "
                  f"local rows, {passes.chunk} node(s) per accumulate")
        log.debug(f"Response quadrature: chi build {receipt['seconds'].get('direct_dispatch', 0.):.2f} s "
                 f"({len(rules['groups'])} group(s), {receipt['correlation_count']} node evaluations"
                 f"{layout}); bank write {receipt['seconds'].get('io', 0.):.2f} s")
        residence = receipt["bank_residence"]
        log.debug(f"Response quadrature: chi bank {residence['residence']}, "
                  f"{residence['bytes_per_rank'] / 2**30:.2f} GiB/rank; {residence['reason']}"
                  + ("" if residence["residence"] == "device" else
                     f"; q spans of {residence.get('q_width')} parents"
                     f"; write {receipt['seconds']['bank_write']:.2f} s, read "
                     f"{receipt['seconds']['bank_read']:.2f} s, waited {receipt['seconds']['bank_wait']:.2f} s"
                     + (f"; {residence['rereads']} span(s) re-read after a failed check"
                        if residence.get("rereads") else "")))
        if residence["residence"] != "device" and residence.get("rereads"):
            log.warning(f"Response quadrature: {residence['rereads']} chi bank span(s) "
                        "re-read after a failed check",
                        extra=message("response_bank_span_reread"))
        log.debug("Response quadrature: seconds " + " ".join(
            f"{key}={value:.3f}" for key, value in receipt["seconds"].items()),
                  extra=record("response.bank", samples=len(z),
                               node_evaluations=receipt["correlation_count"],
                               groups=len(rules["groups"]), seconds=dict(receipt["seconds"]),
                               residence=dict(residence)))
    del roots, held
    ledger.live_stages = caller_live
    receipt["stream_passes"] = len(rules["groups"])
    receipt["batch_reason"] = ("one stream per sample group; each Green pair serves every member's value, "
                               "derivative and both orientations")
    receipt["io_scope"] = "I/O envelope includes device readiness, packing, and finite checks; not pure storage time"
    receipt["completion"] = all(committed(sample) for sample in range(len(z)))
    return _finish_receipt(receipt,meta,header,started)


_PHOTON_SECTOR_BLOCKS = (("CC", ((0, 0),)),
                         ("CT", tuple((0, b) for b in range(1, 4))),
                         ("TC", tuple((a, 0) for a in range(1, 4))),
                         ("TT", tuple((a, b) for a in range(1, 4) for b in range(1, 4))))


@lru_cache(maxsize=8)
def _photon_norm_program(layout, mesh_xy):
    """Frobenius norms of the CC/CT/TC/TT blocks of every parent in one program, not ~70 eager passes."""
    from .photon_layout import photon_block_view

    @jax.jit
    def norms(value):
        face = P(None, "x", "y")
        return jnp.stack([jnp.sqrt(sum(ordered_sum(jnp.abs(photon_block_view(value, layout, a, b, mesh_xy))**2,
                                                   axis=(-2, -1), mesh=mesh_xy, spec=face) for a, b in pairs))
                          for _, pairs in _PHOTON_SECTOR_BLOCKS], axis=-1)
    return norms


def _photon_sample_norms(receipt, value, parent, first, layout, mesh_xy):
    """Record Frobenius norms of CC/CT/TC/TT without gathering operators."""
    values = np.asarray(_photon_norm_program(layout, mesh_xy)(value))
    receipt.setdefault("sector_sample_norms", []).extend(
        dict(parent=int(parent)+i, first_sample=int(first),
             **{name: [float(v)] for (name, _), v in zip(_PHOTON_SECTOR_BLOCKS, row)})
        for i, row in enumerate(values))


#: The packed photon V of the run's V file, held on the host as each process's
#: device shards (one entry, a ``common.collectives.HostSpill``): V does not
#: change across SC maps, so every map places it on the devices from this copy
#: instead of re-reading the file, and no device memory is held between banks.
_PHOTON_V_HOST: dict = {}


def photon_bare_operator(wfns, wfns_transverse, meta, *, path, mu_bases, layout, mesh_xy):
    """Read authenticated raw-parent photon V through its sole packing owner.

    The reader returns MuBasis-packed family tiles. Undo that family packing
    before the photon owner inserts canonical channel chunks, exactly as for
    the endpoint carriers. All operators stay at P(None,x,y).  The packed V
    is read once per V file (path, size, modification time), endpoint bases
    and q parents, and moved to its host copy (:data:`_PHOTON_V_HOST`); every
    call places that copy (``common.collectives.restore_from_host``: each
    process its own shards, no collective).
    """
    from common.collectives import restore_from_host, spill_to_host
    plans = (wfns.green_parent.plan, wfns_transverse.green_parent.plan)
    stat = os.stat(path)
    # By content, so a later map's endpoint objects of the same bases hit.
    bases = hashlib.sha256(b"".join(
        np.asarray(b.canonical_indices, dtype="<i4").tobytes() + str(int(b.n_packed)).encode()
        for b in mu_bases)).hexdigest()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns), layout, mesh_xy, bases,
           tuple(int(q) for q in plans[0].sym.q_irr_full_idx))
    held = _PHOTON_V_HOST.get(key)
    if held is None:
        from common.gpu_utils import record_host_hold
        value = _read_photon_bare_operator(path, plans, mu_bases, layout, mesh_xy)
        _PHOTON_V_HOST.clear()
        record_host_hold("packed photon V (photon_bare_operator)",
                         sum(s.data.nbytes for s in value.addressable_shards))
        held = _PHOTON_V_HOST[key] = spill_to_host(value)
    return restore_from_host(held)


def _read_photon_bare_operator(path, plans, mu_bases, layout, mesh_xy):
    """The packed photon V read from the V file (:func:`photon_bare_operator`)."""
    from file_io.restart_bundle import BispinorVqReader
    from .photon_layout import pack_photon_operator
    from .v_q_bispinor import ZERO_TILES
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
                        photon_head_state=None,
                        print_fn=print):
    """Build a full photon bank through the existing sample/moment stages.

    ``bank_io`` names the initialized scratch path, current identity and
    ``bispinor_v_q_path``. The stored samples are W-W_infinity; the committed
    ``constant`` field is W_infinity-V. M0..M3 use the same convention as the
    ordered charge bank. Both CT and TC are retained. The scalar producer,
    memory planner, quadrature, transaction masks and reader are shared.
    Every call, every SC map included, builds its own static contact: the
    Ward proxy subtracts the static limit of this map's response.
    ``photon_head_state = (rotation, wfns, occupation_state, velocity)``
    sets the direct head's state (``sc_head_update``): None entries are the
    bank's own state, no rotation and the dipole velocity (one-shot); SC
    ``dft_velocity`` passes the map's QP rotation, ``parallel_transport``
    also this map's ``qsgw_head.qp_velocity`` (DFT basis, head storage), and
    ``off`` the DFT bundle and its fixed-N state.
    """
    from file_io.tangential_store import validate_tangential_bank

    started = time.monotonic()
    # The photon bank's head, fenced so it is no longer an unnamed band of
    # spole.bank (P2-S): scratch validation, the V-file digest, the census.
    with timing.section("bank.photon_setup"):
        header = validate_tangential_bank(bank_io["path"],
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
    ledger = meta.tangential_capacity
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
    log.debug("photon bank: preparing shared vertex endpoints and bare V")
    with timing.section("bank.photon_endpoints"):
        vertex = prepare_photon_carriers(wfns, wfns_transverse, mu_bases,
                                         mesh_xy=mesh_xy, layout=layout)
        bank["photon_v"] = photon_bare_operator(wfns, wfns_transverse, meta,
            path=bank["bispinor_v_q_path"], mu_bases=mu_bases, layout=layout, mesh_xy=mesh_xy)
        from .gw_config import uses_direct_bispinor_tangential_head
        direct_gamma = None
        if uses_direct_bispinor_tangential_head(config):
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
    log.debug("photon bank: centroid D and static grid reference")
    with timing.section("bank.static_contact"):
        from file_io.tangential_store import write_bank_contact
        grid, drude, contact = photon_static_contact(wfns, meta, mesh_xy=mesh_xy,
            layout=layout, vertex=vertex, occupation_state=occupation_state,
            sample_plan=sample_plan, execute=execute, receipt=receipt)
        bank["minus_q_operator_provenance"] = dict(coulomb=bank["coulomb"],
            static_contact="this map's Pi_FD(0,0)",
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
        if uses_direct_bispinor_tangential_head(config):
            from .qsgw_head import read_authenticated_dipole_velocity
            from .photon_direct_head import build_direct_photon_head, packed_gamma_vectors
            cache = photon_head_cache if photon_head_cache is not None else {}
            rotation, head_wfns, head_occupation, velocity = (
                photon_head_state or (None, None, None, None))
            if velocity is None:
                velocity = cache.get("direct_photon_velocity")
            if velocity is None:
                velocity = read_authenticated_dipole_velocity(
                    os.path.join(config.input_dir, "dipole.h5"), wfn=wfn,
                    meta=meta, config=config, mesh=mesh_xy,
                    wfn_fingerprint_binding=wfn_fingerprint_binding)
                cache["direct_photon_velocity"] = velocity
            if rotation is not None:
                from .qsgw_head import rotate_velocity_active_to_qp
                velocity = rotate_velocity_active_to_qp(
                    velocity, rotation, mesh=mesh_xy)
            direct_head = build_direct_photon_head(
                velocity, wfns if head_wfns is None else head_wfns,
                occupation_state if head_occupation is None else head_occupation,
                photon_g0_vectors=direct_gamma, layout=layout,
                mesh=mesh_xy, meta=meta, wfn=wfn,
                frequencies_ry=bank_points(sample_plan), print_fn=print_fn)
            direct_head["gamma_vectors"] = packed_gamma_vectors(
                direct_gamma, layout, mesh_xy,
                current_basis_rows=meta.current_basis_rows)
            receipt["direct_gamma"] = dict(
                approximation="first_order_dipole_current_fd",
                sectors="CC_CT_TC_TT", local_fields=False,
                samples=direct_head["rule"],
                rule_spread_ry=direct_head["rule_spread_ry"],
                dyson_residual=direct_head["dyson_residual"],
                static_limit="Thomas-Fermi at z=0; dynamic Drude for Im(z)>0")
    before = time.monotonic()
    log.debug("photon bank: exact moments and W_infinity")
    receipt["moments"] = compute_moment_bank(wfns, meta, config, mesh_xy=mesh_xy,
        sym=sym, bank_io=bank, vertex=vertex, contact=contact,
        direct_head=direct_head)
    receipt["seconds"]["moments"] = time.monotonic()-before
    before = time.monotonic()
    log.debug("photon bank: ordered samples and derivatives")
    receipt["samples"] = produce_sample_bank(wfns, meta, config, mesh_xy=mesh_xy,
        sym=sym, sample_plan=sample_plan, bank_io=bank, vertex=vertex,
        contact=contact, direct_head=direct_head, print_fn=print_fn)
    receipt["seconds"]["samples"] = time.monotonic()-before
    header = validate_tangential_bank(bank["path"], expected_identity=bank["identity"],
                                       mesh_xy=mesh_xy, require_complete=True)
    ledger.live_stages = ambient
    receipt["completion"] = True
    receipt["bank_header"] = header
    return _finish_receipt(receipt, meta, header, started)
