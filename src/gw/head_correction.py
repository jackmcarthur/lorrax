"""The q=0, G=G'=0 (Gamma-cell) head of the Coulomb / photon interaction; see docs/architecture/four_current_wiring.md."""

from __future__ import annotations

import functools
from dataclasses import dataclass
import enum
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.shard_map import shard_map
from common.units import RYD_TO_EV


def _analytic_q0_sphere(params) -> bool:
    """One compatibility-preserving resolver for the q=0 v estimator."""
    return (
        bool(params.get("head_minibz_average", False))
        or str(params.get("bgw_metal_q0_treatment", "exact")).strip().lower()
        == "bgw_q0shift"
    )


class HeadResponseKind(str, enum.Enum):
    """Reduction state of the response that produced a scalar head; see docs/architecture/four_current_wiring.md."""

    DIRECT_IRREDUCIBLE = "direct_irreducible"
    FULL_LOCAL_FIELDS = "full_local_fields"
    MICRO_REDUCIBLE = "micro_reducible"
    OVERRIDE = "override"
    OFF = "off"


@dataclass(frozen=True)
class HeadSample:
    """Resolved q=0 Coulomb head sample at one frequency."""

    vc0: complex
    wcoul0: complex
    source: str
    omega: complex
    #: The Cartesian q²-coefficient tensor ``S(ω)`` that PRODUCED ``wcoul0``,
    #: when there was one — ``(3, 3)`` complex, the convention of
    #: ``docs/theory/s-tensor-convention.md``.  ``None`` on the ``epshead``
    #: branch, which has no tensor (it fits an isotropic γ instead).
    #:
    #: Carried because ``wcoul0`` alone is the cell average on ONE grid, and
    #: the coarse→fine W densifier (``gw.head_densify``) needs the INTEGRAND
    #: to re-attach the head per fine q.  Persisted to the restart by
    #: ``file_io.write_head_scalars_to_h5`` so the BSE need not rebuild it.
    S_cart: np.ndarray | None = None
    response_kind: HeadResponseKind = HeadResponseKind.DIRECT_IRREDUCIBLE


@dataclass(frozen=True)
class HeadGNParams:
    """Fitted GN parameters for the scalar Coulomb head."""

    omega_h_sq: float
    omega_h: float
    B_h: float
    R_h: float
    wc_head_0: float
    wc_head_iwp: float
    vc0: float
    omega_p: float


@dataclass(frozen=True)
class StaticHeadTerms:
    """Exact static q=0 head terms for bare X / SX / COHSEX; see docs/architecture/four_current_wiring.md."""

    sigma_x_diag: jnp.ndarray
    sigma_sx_diag: jnp.ndarray
    sigma_sx_minus_x_diag: jnp.ndarray
    sigma_coh_diag: jnp.ndarray
    vc0: complex
    wcoul0: complex
    wc_head_0: complex
    source: str


_STATIC_GAUGE_WARD_RESIDUAL_MAX = 1.0e-8
_STATIC_GAUGE_HERMITICITY_RESIDUAL_MAX = 1.0e-10


def static_hall_linear_response(sigma_H) -> jax.Array:
    """Return the unique static Hall-only linear CT/TC tensor; see docs/architecture/four_current_wiring.md."""
    sigma_raw = np.asarray(sigma_H)
    if sigma_raw.shape != (3,):
        raise ValueError(
            f"sigma_H must be a three-component Hall pseudovector; got "
            f"{sigma_raw.shape}")
    if not np.all(np.isfinite(sigma_raw)):
        raise ValueError("sigma_H contains non-finite values")
    if np.any(np.imag(sigma_raw) != 0.0):
        raise ValueError(
            "static sigma_H must be explicitly real; refusing to discard an "
            "imaginary component")
    sigma = np.asarray(np.real(sigma_raw), dtype=np.float64)
    axes = np.eye(3, dtype=np.float64)[:2]
    # epsilon[b,a,i] sigma[b] = (sigma x e_a)[i].  The minus is fixed by the
    # live band orientation P=-Delta*D: the direct Adler--Wiser response
    # energy-orders the bra and conjugates the row, so its linear CT
    # imaginary part is the NEGATIVE of the persisted occupied-bra Berry
    # tensor that sigma_H stores (register row "Hall CT/TC block is inserted
    # with the wrong sign", taken from 7d7df280; the oracle is
    # tests/test_qsgw_parallel_transport_head.py::
    # test_raw_hall_matches_orbital_cB_owner_and_documented_sign).
    ct = -1j * np.stack([np.cross(sigma, axis) for axis in axes], axis=0)
    linear = np.zeros((2, 4, 4), dtype=np.complex128)
    linear[:, 0, 1:] = ct
    linear[:, 1:, 0] = np.conj(ct)
    return jnp.asarray(linear)


def canonicalize_static_gauge_q2_tensor(S_direct) -> jax.Array:
    """Return the unique coordinate-symmetric representative of ``q S q``; see docs/architecture/four_current_wiring.md."""
    S = jnp.asarray(S_direct)
    if tuple(S.shape) != (2, 2, 4, 4):
        raise ValueError(f"S_direct must be (2,2,4,4); got {S.shape}")
    return 0.5 * (S + jnp.swapaxes(S, 0, 1))


def static_gauge_tensor_residuals(S_direct) -> tuple[float, float]:
    """Return algebraic in-plane Ward and Hermiticity residuals of ``S``; see docs/architecture/four_current_wiring.md."""
    S = np.asarray(jax.device_get(S_direct), dtype=np.complex128)
    if S.shape != (2, 2, 4, 4):
        raise ValueError(f"S_direct must be (2,2,4,4); got {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S_direct contains non-finite values")
    scale = float(np.max(np.abs(S)))
    if scale == 0.0:
        return 0.0, 0.0

    coordinate_error = float(np.max(np.abs(S - np.swapaxes(S, 0, 1))))
    lorentz_error = float(np.max(np.abs(
        S - np.conj(np.swapaxes(S, 2, 3)))))

    # Coefficients of (qx^3, qx^2*qy, qx*qy^2, qy^3), retaining the open
    # opposite Lorentz leg.  Current x/y are Lorentz indices 1/2.
    left = np.stack((
        S[0, 0, 1, :],
        S[0, 1, 1, :] + S[1, 0, 1, :] + S[0, 0, 2, :],
        S[1, 1, 1, :] + S[0, 1, 2, :] + S[1, 0, 2, :],
        S[1, 1, 2, :],
    ), axis=0)
    right = np.stack((
        S[0, 0, :, 1],
        S[0, 1, :, 1] + S[1, 0, :, 1] + S[0, 0, :, 2],
        S[1, 1, :, 1] + S[0, 1, :, 2] + S[1, 0, :, 2],
        S[1, 1, :, 2],
    ), axis=0)
    ward_error = float(max(np.max(np.abs(left)), np.max(np.abs(right))))
    return ward_error / scale, max(coordinate_error, lorentz_error) / scale


@dataclass(frozen=True)
class BGWQ0Channel:
    """One finite grid-q head channel used as BGW's epsilon q0 sample."""

    requested_full_index: int
    representative_full_index: int
    wedge_row: int
    q0_reduced: tuple[float, float, float]
    g_head: object                 # (1, n_mu), sharded on the centroid axis
    v_bare: float


def resolve_bgw_q0_channel(
    config, sym, q_wedge_full_indices, head_channel, *, kgrid,
):
    """Bind the deck's reduced q0 vector to one stored W-wedge row; see docs/architecture/four_current_wiring.md."""
    if not bool(config.head.uses_bgw_metal_q0shift):
        return None
    if head_channel is None:
        raise ValueError(
            "bgw_metal_q0_treatment=bgw_q0shift requires the finite-q "
            "Coulomb head channel, but GW initialization did not build it.")
    kgrid = np.asarray(kgrid, dtype=np.int64)
    if kgrid.shape != (3,) or np.any(kgrid <= 0):
        raise ValueError(
            "BGW q0 resolution requires a positive 3-D kgrid; "
            f"got {kgrid}.")
    q0 = np.asarray(config.head.bgw_metal_q0_vector, dtype=np.float64)
    steps_float = q0 * kgrid
    steps = np.rint(steps_float).astype(np.int64)
    if not np.allclose(steps_float, steps, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "bgw_metal_q0_vector must lie on the deck's reciprocal grid: "
            f"q0={tuple(q0)}, kgrid={tuple(int(n) for n in kgrid)}, "
            f"q0*kgrid={tuple(float(x) for x in steps_float)}.")
    full_steps = np.asarray(sym.kvecs_asints, dtype=np.int64)
    target_mod = np.mod(steps, kgrid)
    matches = np.flatnonzero(np.all(
        np.mod(full_steps, kgrid[None, :]) == target_mod[None, :], axis=1))
    if matches.size != 1:
        raise ValueError(
            "bgw_metal_q0_vector did not identify exactly one full-grid q: "
            f"q0={tuple(q0)}, matches={matches.tolist()}.")
    requested = int(matches[0])
    wedge_row = int(np.asarray(sym.irr_idx_q, dtype=np.int64)[requested])
    q_idx = np.asarray(q_wedge_full_indices, dtype=np.int64)
    if not 0 <= wedge_row < q_idx.size:
        raise ValueError(
            "BGW q0 irreducible row is outside the stored W wedge: "
            f"row={wedge_row}, wedge size={q_idx.size}.")
    representative = int(q_idx[wedge_row])
    multiplicity = int(np.asarray(head_channel.mult)[representative])
    if multiplicity != 1:
        raise ValueError(
            "bgw_metal_q0_vector must select a unique G=0-like head slot; "
            f"its irreducible representative has multiplicity {multiplicity}.")
    v_bare = float(np.asarray(head_channel.v_bare)[representative])
    if not np.isfinite(v_bare) or v_bare <= 0.0:
        raise ValueError(
            "bgw_metal_q0_vector selected no finite bare Coulomb head "
            f"(representative full-q row {representative}, v={v_bare}).")
    return BGWQ0Channel(
        requested_full_index=requested,
        representative_full_index=representative,
        wedge_row=wedge_row,
        q0_reduced=tuple(float(x) for x in q0),
        g_head=head_channel.g_head[representative, 0:1, :],
        v_bare=v_bare,
    )


@functools.partial(jax.jit, static_argnames=("mesh_xy",))
def finite_q0_epsinv_head(
    chi_q0,
    W_q0,
    g_head_q0,
    v_bare,
    chi_prefactor,
    *,
    mesh_xy: Mesh,
):
    """Return the full finite-q ``epsilon^{-1}_{00}``, including wings; see docs/architecture/four_current_wiring.md."""
    chi = jnp.asarray(chi_q0) * jnp.asarray(
        chi_prefactor, dtype=jnp.asarray(chi_q0).dtype)
    W = jnp.asarray(W_q0)
    g = jnp.asarray(g_head_q0)
    sh_x = NamedSharding(mesh_xy, P(None, "x"))
    sh_y = NamedSharding(mesh_xy, P(None, "y"))
    sh_q = NamedSharding(mesh_xy, P(None))
    u_x = jax.lax.with_sharding_constraint(jnp.conj(g), sh_x)
    u_y = jax.lax.with_sharding_constraint(jnp.conj(g), sh_y)
    r_x = jax.lax.with_sharding_constraint(g, sh_x)
    chi_u_x = jnp.einsum("qmn,qn->qm", chi, u_y, optimize=True)
    chi_u_y = jax.lax.with_sharding_constraint(chi_u_x, sh_y)
    W_chi_u_x = jnp.einsum("qmn,qn->qm", W, chi_u_y, optimize=True)
    l_x = u_x + W_chi_u_x
    l_y = jax.lax.with_sharding_constraint(l_x, sh_y)
    chi_l_x = jnp.einsum("qmn,qn->qm", chi, l_y, optimize=True)
    response = jnp.einsum("qm,qm->q", r_x, chi_l_x, optimize=True)
    epsinv = 1.0 + jnp.asarray(v_bare, dtype=response.dtype) * response
    return jax.lax.with_sharding_constraint(epsinv, sh_q)


def bgw_q0shift_vhead(wfn, meta):
    """BGW analytic-sphere mini-BZ bare head, in LORRAX's raw units."""
    from gw.vcoul import compute_q0_averages

    vc0, _ = compute_q0_averages(
        wfn,
        jnp.asarray(1.0, dtype=jnp.float64),
        meta,
        S_cart=None,
        analytic_sphere=True,
    )
    return complex(vc0)


def bgw_q0shift_head_sample(vc0, epsinv, omega) -> HeadSample:
    """Compose BGW's finite-q epsilon head with its q=0 v-cell average."""
    eps = complex(epsinv)
    v = complex(vc0)
    return HeadSample(
        vc0=v,
        wcoul0=eps * v,
        source="bgw_q0shift(analytic-sphere v; finite-q0 epsinv head+wings)",
        omega=complex(omega),
        S_cart=None,
        response_kind=HeadResponseKind.FULL_LOCAL_FIELDS,
    )


def _representative_entry(diag: jnp.ndarray) -> complex:
    """Return a representative diagonal value for diagnostics."""

    arr = np.asarray(diag).reshape(-1)
    if arr.size == 0:
        return 0.0 + 0.0j
    nz = np.flatnonzero(np.abs(arr) > 0.0)
    idx = int(nz[0]) if nz.size else 0
    return complex(arr[idx])


def resolve_head_override(params, omega) -> HeadSample | None:
    """Return explicit head overrides when both v and W are provided."""

    omega_val = complex(omega)
    vhead_override = params.get("vhead")
    w_key = "whead_0freq" if abs(omega_val) <= 1.0e-14 else "whead_imfreq"
    whead_override = params.get(w_key)
    if vhead_override is None or whead_override is None:
        return None
    source = "override" if abs(omega_val) <= 1.0e-14 else f"override(omega={omega_val} Ry)"
    return HeadSample(
        vc0=complex(vhead_override),
        wcoul0=complex(whead_override),
        source=source,
        omega=omega_val,
        response_kind=HeadResponseKind.OVERRIDE,
    )


def _check_dipole_coverage(
    dipole_path, *, nb_file, nk_file, nk_run, nb_run, nelec, print_fn,
):
    """Loud coverage check on ``dipole.h5`` at the point of use; see docs/architecture/four_current_wiring.md."""
    from common import sanity

    if not sanity.sanity_enabled():
        return
    attrs_nb, attrs_nk = None, None
    try:
        import h5py
        with h5py.File(dipole_path, "r") as h5:
            if "nbands" in h5.attrs:
                attrs_nb = int(np.asarray(h5.attrs["nbands"]))
            if "nk" in h5.attrs:
                attrs_nk = int(np.asarray(h5.attrs["nk"]))
    except (OSError, KeyError, ValueError) as exc:
        print_fn(f"  [dipole guard] could not read attrs from {dipole_path} "
                 f"({type(exc).__name__}: {exc})")
    if attrs_nk is not None and attrs_nk != int(nk_run):
        sanity.warn(
            f"{dipole_path} was generated on nk={attrs_nk} but this run has "
            f"nk_tot={int(nk_run)}.  The head S(ω) would be assembled from a "
            f"different k-sampling than Σ — refusing to trust it is the only "
            f"safe reading of this file.",
            print_fn=print_fn)
    n_cond_file = max(0, int(nb_file) - int(nelec))
    print_fn(
        f"  dipole.h5 coverage: {int(nb_file)} bands on disk "
        f"({int(nelec)} occ + {n_cond_file} cond)"
        + (f", run Σ window = {int(nb_run)} bands" if nb_run else ""))
    if nb_run and int(nb_file) < int(nb_run):
        sanity.warn(
            f"{dipole_path} carries only {int(nb_file)} bands but this run's "
            f"Σ window spans {int(nb_run)}.  The q→0 head S(ω) sums "
            f"conduction states arange({int(nelec)}, {int(nb_file)}) — "
            f"{n_cond_file} of them — so wcoul0, and every Σ_SX/Σ_COH head "
            f"correction built from it, is converged to a SMALLER transition "
            f"space than Σ itself.  This does not crash and does not change "
            f"the exit code; it makes the head systematically wrong.  "
            f"Regenerate dipole.h5 with nbands >= {int(nb_run)} "
            f"(psp.get_dipole_mtxels) to close it.",
            print_fn=print_fn)


def _dipole_window_from_params(params, wfn) -> tuple[int, int, int]:
    """``(nval, ncond, nband)`` — the RUN's resolved band window, or a refusal; see docs/architecture/four_current_wiring.md."""
    missing = [k for k in ("nval", "ncond", "nband")
               if params.get(k) is None]
    if missing:
        raise ValueError(
            f"dipole provenance: the run's band window is missing "
            f"{', '.join(missing)} — the checker was handed a params dict "
            f"that carries no band window (got keys "
            f"{sorted(str(k) for k in params)}).  These three numbers are "
            f"the reference the file's prov_nval/prov_ncond/prov_nband "
            f"stamps are compared against; defaulting them to 5/5/"
            f"max(wfn.nbands={int(getattr(wfn, 'nbands', 0) or 0)}, ...) "
            f"made the comparison invent its own reference and accuse every "
            f"correctly-stamped file whose deck is not 5/5.  Pass the "
            f"resolved config window (gw.head_correction.HeadResolver does "
            f"this from config.nval/config.ncond/config.nband).")
    return (int(params["nval"]), int(params["ncond"]), int(params["nband"]))


def _check_dipole_provenance(dipole_path, *, params, wfn, print_fn) -> None:
    """Was ``dipole.h5`` built from THIS DFT solution and THIS band window?; see docs/architecture/four_current_wiring.md."""
    from common import sanity
    from common.four_current_model import resolve_four_current_representation

    representation = resolve_four_current_representation(
        bool(params.get("_four_current_bispinor", False)),
        params.get("bispinor_gw"))
    explicit_comparison = bool(params.get("_four_current_bispinor", False)) and (
        not representation.scalar_head_bispinor)
    if not sanity.sanity_enabled() and not explicit_comparison:
        return
    nval, ncond, nband = _dipole_window_from_params(params, wfn)
    try:
        from psp.get_dipole_mtxels import check_dipole_provenance
    except Exception as exc:            # psp stack unavailable (h5py-less env)
        if explicit_comparison:
            raise ValueError(
                "GATE comparison_charge_dipole_provenance: the explicit "
                "Pauli comparison cannot authenticate its dipole artifact.\n"
                f"  got:  provenance checker import failed with "
                f"{type(exc).__name__}: {exc}\n"
                "  want: an available check_dipole_provenance owner\n"
                "  why:  without the checker the run cannot establish "
                "that dipole.h5 was built for the scalar charge carrier") from exc
        print_fn(f"  [dipole provenance] check unavailable "
                 f"({type(exc).__name__}: {exc})")
        return
    expected_bispinor = params.get("_charge_bispinor")
    authenticated = check_dipole_provenance(
        dipole_path, wfn=wfn, nval=nval, ncond=ncond, nband=nband,
        bispinor=expected_bispinor, print_fn=print_fn)
    if explicit_comparison and not authenticated:
        raise ValueError(
            "GATE comparison_charge_dipole_provenance: the explicit "
            "four-current comparison received an unauthenticated dipole.\n"
            f"  got:  dipole_file = {dipole_path!r}, "
            f"prov_bispinor != {expected_bispinor!r} or another provenance "
            "field mismatched\n"
            "  want: dipole.h5 regenerated from this exact deck with "
            "prov_bispinor = false\n"
            "  why:  the comparison's charge head uses the Pauli carrier; "
            "a four-spinor or stale artifact measures a different operator")


def resolve_head_sample(params, input_dir, wfn, sym, meta, print_fn, omega) -> HeadSample:
    """Resolve a q=0 head sample using overrides and configured source order."""

    override = resolve_head_override(params, omega)
    if override is not None:
        return override

    # MIXED-HEAD ANNOUNCEMENT.  ``resolve_head_override`` keys on ONE deck
    # scalar per frequency: ``whead_0freq`` at ω=0 and ``whead_imfreq`` at
    # the PPM probe (the only non-zero ω the driver ever asks for).  A deck
    # that sets ``vhead``/``whead_0freq`` and omits ``whead_imfreq``
    # therefore overrides one end of the plasmon-pole fit and computes the
    # other, which is a legitimate thing to want — it pins the static head
    # to a reference code and lets the dispersion come from here — but it
    # is not what "head override" reads like, and the per-sample
    # diagnostics say "Head source: override" at ω=0 without ever naming
    # the other end.  Printed once per run: the resolver memoizes per
    # frequency.
    if (abs(complex(omega)) > 1.0e-14
            and params.get("vhead") is not None
            and params.get("whead_0freq") is not None
            and params.get("whead_imfreq") is None):
        print_fn(
            f"  [head] MIXED HEAD: 'vhead'/'whead_0freq' override W(q→0) at "
            f"omega=0, but 'whead_imfreq' is unset, so W(q→0) at the PPM "
            f"probe omega={complex(omega)} Ry is COMPUTED from "
            f"wcoul0_source="
            f"{str(params.get('wcoul0_source', 's_tensor')).strip().lower()}. "
            f"The two-point pole fit therefore mixes an overridden head "
            f"with a computed one; set 'whead_imfreq' to override both."
        )

    want_source = str(params.get("wcoul0_source", "s_tensor")).strip().lower()
    if want_source not in ("epshead", "s_tensor"):
        raise ValueError(
            f"wcoul0_source={want_source!r} is invalid; expected 'epshead' "
            "or 's_tensor'. The head source never falls back silently.")

    omega_val = complex(omega)
    eta = float(params.get("wcoul0_eta", 0.0) or 0.0)
    eps0_path = os.path.join(input_dir, "eps0mat.h5")
    dipole_path = os.path.join(input_dir, "dipole.h5")

    def from_epshead() -> HeadSample:
        if not os.path.exists(eps0_path):
            raise FileNotFoundError(
                f"wcoul0_source=epshead requested {eps0_path}, but it does "
                "not exist; refusing to substitute s_tensor.")
        if abs(omega_val) > 1.0e-14:
            print_fn(
                f"wcoul0_source=epshead is static-only; using epshead(0) for omega={omega_val} Ry"
            )
        from file_io.epsreader import EPSReader
        from gw.vcoul import compute_q0_averages

        # ``with``: EPSReader's matrix attributes are h5py dataset HANDLES
        # since 2026-08-22 (it used to slurp the whole ``mats/matrix`` into
        # host memory on every rank, for the six numbers this branch wants),
        # so the file's lifetime is now the caller's to state.  ``epshead``
        # is read in the constructor, so nothing here outlives the block.
        with EPSReader(eps0_path) as eps0:
            head = jnp.asarray(eps0.epshead, dtype=jnp.complex128)
        vc0_mean, wcoul0 = compute_q0_averages(
            wfn,
            head,
            meta,
            S_cart=None,
            analytic_sphere=_analytic_q0_sphere(params),
            certificate_fn=params.get("_q0_certificate_fn"),
        )
        source = "epshead(0)" if abs(omega_val) > 1.0e-14 else "epshead"
        return HeadSample(
            vc0=complex(vc0_mean),
            wcoul0=complex(wcoul0),
            source=source,
            omega=omega_val,
        )

    def from_s_tensor() -> HeadSample:
        if not os.path.exists(dipole_path):
            raise FileNotFoundError(
                f"wcoul0_source=s_tensor requested {dipole_path}, but it "
                "does not exist; refusing to substitute epshead.")
        from gw.vcoul import compute_q0_averages

        from common import timing as _tmg
        S_cart_omega = build_S_cart_omega(
            wfn, sym, meta, params, dipole_path, omega_val, eta=eta,
            print_fn=print_fn)
        with _tmg.section("head.q0_avg"):
            vc0_mean, wcoul0 = compute_q0_averages(
                wfn,
                jnp.asarray(0.0, dtype=jnp.float64),
                meta,
                S_cart=S_cart_omega,
                analytic_sphere=_analytic_q0_sphere(params),
                certificate_fn=params.get("_q0_certificate_fn"),
            )
        source = "s_tensor" if abs(omega_val) <= 1.0e-14 else f"s_tensor(omega={omega_val} Ry)"
        return HeadSample(
            vc0=complex(vc0_mean),
            wcoul0=complex(wcoul0),
            source=source,
            omega=omega_val,
            S_cart=np.asarray(S_cart_omega, dtype=np.complex128),
        )

    return from_epshead() if want_source == "epshead" else from_s_tensor()


def build_S_cart_omega(wfn, sym, meta, params, dipole_path, omega,
                       *, eta: float = 0.0, print_fn=print) -> np.ndarray:
    """``S(ω)``, the Cartesian q²-coefficient tensor, from ``dipole.h5``; see docs/architecture/four_current_wiring.md."""
    from common.chi_from_dipole import read_dipole_h5, compute_S_omega
    from common import timing as _tmg

    with _tmg.section("head.read_dipole"):
        dipole_cart, deltaE = read_dipole_h5(dipole_path)
    nk_tot = int(sym.nk_tot)
    nb = int(dipole_cart.shape[2])
    nelec = int(wfn.nelec)
    with _tmg.section("head.checks"):
        _check_dipole_coverage(
            dipole_path, nb_file=nb, nk_file=int(dipole_cart.shape[1]),
            nk_run=nk_tot, nb_run=int(getattr(meta, "nb_sigma", 0) or 0),
            nelec=nelec, print_fn=print_fn)
        _check_dipole_provenance(dipole_path, params=params or {}, wfn=wfn,
                                 print_fn=print_fn)
    # TODO(metal-head): this legacy one-shot dipole path retains the ifmax
    # step occupation; metallic QSGW uses the explicit weighted PT head.
    occ = np.zeros((nk_tot, nb), dtype=float)
    occ[:, :max(0, min(nelec, nb))] = 1.0
    f_nk = jnp.asarray(occ, dtype=jnp.float64)
    omega_grid = jnp.asarray([complex(omega)], dtype=jnp.complex128)
    with _tmg.section("head.S_omega"):
        S_cart_omega = compute_S_omega(
            dipole_cart, deltaE, f_nk, float(wfn.cell_volume), int(sym.nk_tot),
            int(wfn.nspin), int(wfn.nspinor), omega_grid, eta=float(eta),
        )[0]
    return np.asarray(S_cart_omega, dtype=np.complex128)


@functools.partial(jax.jit, static_argnames=("mesh_xy",))
def fold_small_head_wings_sharded(
    R_direct: jax.Array,
    Y_x: jax.Array,
    W_body_xy: jax.Array,
    Z_y: jax.Array,
    Vcell: float,
    *,
    mesh_xy: Mesh,
) -> jax.Array:
    """Fold a bounded small-field response through the screened body; see docs/architecture/four_current_wiring.md."""
    n_lead = W_body_xy.ndim - 2
    arrays = (R_direct, Y_x, W_body_xy, Z_y)
    if n_lead < 0 or any(a.ndim != n_lead + 2 for a in arrays):
        raise ValueError(
            "R_direct, Y, W_body, and Z must have two non-leading axes")
    leading = tuple(W_body_xy.shape[:-2])
    if any(tuple(a.shape[:-2]) != leading for a in arrays):
        raise ValueError(
            "R_direct, Y, W_body, and Z must share their leading axes")
    if (
        int(R_direct.shape[-2]) != int(Y_x.shape[-2])
        or int(Y_x.shape[-1]) != int(W_body_xy.shape[-2])
        or int(W_body_xy.shape[-1]) != int(Z_y.shape[-2])
        or int(Z_y.shape[-1]) != int(R_direct.shape[-1])
    ):
        raise ValueError(
            "small-field/body extents do not compose: "
            f"R={R_direct.shape}, Y={Y_x.shape}, "
            f"W={W_body_xy.shape}, Z={Z_y.shape}")
    lead = [None] * n_lead

    def _local_fold(s_direct, y_local, w_local, z_local, volume):
        # Each rank owns exactly one (mu_X, nu_Y) body tile.  Contract that
        # tile before communicating, then reduce only the small-field result.
        # Leaving this as a global einsum lets XLA select a full-matrix
        # temporary at large body extent even though the public field axes
        # are bounded.
        local = jnp.einsum(
            "...am,...mn,...nb->...ab",
            y_local, w_local, z_local, optimize=True)
        correction = jax.lax.psum(local, ("x", "y"))
        return s_direct + correction / volume

    folded = shard_map(
        _local_fold,
        mesh=mesh_xy,
        in_specs=(
            P(*lead, None, None),
            P(*lead, None, "x"),
            P(*lead, "x", "y"),
            P(*lead, "y", None),
            P(),
        ),
        out_specs=P(*lead, None, None),
    )(
        R_direct,
        Y_x,
        W_body_xy,
        Z_y,
        jnp.asarray(Vcell),
    )
    return folded


def fold_cartesian_head_wings_sharded(
    S_direct: jax.Array,
    Y_x: jax.Array,
    W_body_xy: jax.Array,
    Z_y: jax.Array,
    cell_volume: float,
    *,
    mesh_xy: Mesh,
) -> jax.Array:
    """Charge-head adapter to :func:`fold_small_head_wings_sharded`; see docs/architecture/four_current_wiring.md."""
    return fold_small_head_wings_sharded(
        S_direct, Y_x, W_body_xy, Z_y, cell_volume, mesh_xy=mesh_xy)


@functools.partial(jax.jit, static_argnames=("mesh_xy",))
def _small_head_wing_halves_sharded(Y_x, W_body_xy, Z_y, *, mesh_xy):
    """Compiled body of :func:`small_head_wing_halves_sharded`."""
    def _local(y_local, w_local, z_local):
        yw = jax.lax.psum(
            jnp.einsum("aAi,ij->aAj", y_local, w_local, optimize=True),
            "x")
        wz = jax.lax.psum(
            jnp.einsum("ij,bjB->biB", w_local, z_local, optimize=True),
            "y")
        return yw, wz

    return shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=(P(None, None, "x"), P("x", "y"),
                  P(None, "y", None)),
        out_specs=(P(None, None, "y"), P(None, "x", None)),
        check_vma=False,
    )(Y_x, W_body_xy, Z_y)


def small_head_wing_halves_sharded(
    Y_x: jax.Array,
    W_body_xy: jax.Array,
    Z_y: jax.Array,
    *,
    mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """Contract each small photon wing through one resident body ``W``; see docs/architecture/four_current_wiring.md."""
    if Y_x.ndim != 3 or W_body_xy.ndim != 2 or Z_y.ndim != 3:
        raise ValueError(
            "small photon-head halves require Y=(2,4,N), W=(N,N), "
            f"Z=(2,N,4); got {Y_x.shape}, {W_body_xy.shape}, {Z_y.shape}")
    n_body = int(W_body_xy.shape[0])
    expected = ((2, 4, n_body), (n_body, n_body), (2, n_body, 4))
    if (tuple(Y_x.shape), tuple(W_body_xy.shape), tuple(Z_y.shape)) != expected:
        raise ValueError(
            "small photon-head half-contraction extents do not compose: "
            f"Y={Y_x.shape}, W={W_body_xy.shape}, Z={Z_y.shape}")

    required = (
        (Y_x, "Y_x", P(None, None, "x")),
        (W_body_xy, "W_body_xy", P("x", "y")),
        (Z_y, "Z_y", P(None, "y", None)),
    )
    for array, name, spec in required:
        wanted = NamedSharding(mesh_xy, spec)
        sharding = getattr(array, "sharding", None)
        if (sharding is None
                or not sharding.is_equivalent_to(wanted, array.ndim)):
            raise ValueError(
                f"{name} must already have sharding {spec}; got {sharding}. "
                "Refusing an implicit photon-body reshard.")
    return _small_head_wing_halves_sharded(
        Y_x, W_body_xy, Z_y, mesh_xy=mesh_xy)


@jax.jit
def _static_slab_photon_head_moment_chunk(
    q_cart: jax.Array,
    D_raw: jax.Array,
    H_hall: jax.Array,
    S_quadratic: jax.Array,
    valid_count: jax.Array,
    sample_weight: jax.Array,
):
    """Accumulate one fixed-size chunk of the coupled small-head solve; see docs/architecture/four_current_wiring.md."""
    q = jnp.asarray(q_cart, dtype=jnp.float64)
    D = jnp.asarray(D_raw, dtype=jnp.complex128)
    H = jnp.asarray(H_hall, dtype=jnp.complex128)
    S = jnp.asarray(S_quadratic, dtype=jnp.complex128)
    qxy = q[:, :2]
    R = (
        jnp.einsum("sa,aij->sij", qxy, H, optimize=True)
        + jnp.einsum("sa,sb,abij->sij", qxy, qxy, S, optimize=True)
    )
    identity = jnp.eye(4, dtype=jnp.complex128)[None, :, :]
    lhs = identity - jnp.einsum("sik,skj->sij", D, R, optimize=True)
    W_head = jnp.linalg.solve(lhs, D)

    valid = jnp.arange(q.shape[0], dtype=jnp.int32) < valid_count
    weight = jnp.where(
        valid, jnp.asarray(sample_weight, dtype=jnp.float64), 0.0)
    basis = jnp.concatenate(
        (jnp.ones((q.shape[0], 1), dtype=jnp.float64), qxy), axis=1)
    moments = jnp.einsum(
        "s,su,sij,sv->uvij", weight, basis, W_head, basis,
        optimize=True)
    D_sum = jnp.einsum("s,sij->ij", weight, D, optimize=True)

    residual = jnp.einsum(
        "sik,skj->sij", lhs, W_head, optimize=True) - D
    residual_norm = jnp.linalg.norm(residual, axis=(-2, -1))
    lhs_norm = jnp.linalg.norm(lhs, axis=(-2, -1))
    W_norm = jnp.linalg.norm(W_head, axis=(-2, -1))
    D_norm = jnp.linalg.norm(D, axis=(-2, -1))
    backward = residual_norm / jnp.maximum(
        lhs_norm * W_norm + D_norm,
        jnp.asarray(1.0e-300, dtype=jnp.float64))
    singular_values = jnp.linalg.svd(lhs, compute_uv=False)
    sigma_min = singular_values[:, -1]
    # Frobenius throughout: the backward error above and this kappa use the
    # same submultiplicative norm.  Their product theta is the conditioned
    # backward error entering the rigorous 2*theta/(1-theta) forward bound;
    # theta itself is not that bound.
    inverse_lhs_norm = jnp.linalg.norm(
        1.0 / jnp.maximum(
            singular_values, jnp.asarray(1.0e-300, dtype=jnp.float64)),
        axis=-1)
    condition = lhs_norm * inverse_lhs_norm
    conditioned_backward = condition * jnp.maximum(
        backward, jnp.asarray(np.finfo(np.float64).eps, dtype=jnp.float64))
    max_backward = jnp.max(jnp.where(valid, backward, 0.0))
    min_sigma = jnp.min(jnp.where(valid, sigma_min, jnp.inf))
    max_condition = jnp.max(jnp.where(valid, condition, 0.0))
    max_conditioned_backward = jnp.max(
        jnp.where(valid, conditioned_backward, 0.0))
    return (moments, D_sum, valid_count, max_backward, min_sigma,
            max_condition, max_conditioned_backward)


def static_slab_photon_head_moment_chunk(
    q_cart,
    D_raw,
    sigma_H,
    S_quadratic,
    valid_count,
    sample_weight,
):
    """Validated entry to the fixed-size static slab photon-head graph; see docs/architecture/four_current_wiring.md."""
    q_shape = tuple(np.shape(q_cart))
    d_shape = tuple(np.shape(D_raw))
    sigma_shape = tuple(np.shape(sigma_H))
    S_shape = tuple(np.shape(S_quadratic))
    if len(q_shape) != 2 or q_shape[1] != 3:
        raise ValueError(f"q_cart must be (chunk,3), got {q_shape}")
    expected_D = (q_shape[0], 4, 4)
    if d_shape != expected_D:
        raise ValueError(f"D_raw must be {expected_D}, got {d_shape}")
    if sigma_shape != (3,) or S_shape != (2, 2, 4, 4):
        raise ValueError(
            "static slab response requires sigma_H=(3,) and "
            f"S_quadratic=(2,2,4,4); got {sigma_shape}/{S_shape}")
    n_valid = int(valid_count)
    if not 0 <= n_valid <= q_shape[0]:
        raise ValueError(
            f"valid_count must lie in [0,{q_shape[0]}], got {n_valid}")
    weight = np.asarray(sample_weight, dtype=np.float64)
    if weight.shape != (q_shape[0],):
        raise ValueError(
            f"sample_weight must be {(q_shape[0],)}, got {weight.shape}")
    if (not np.all(np.isfinite(weight)) or np.any(weight[:n_valid] < 0.0)
            or np.any(weight[n_valid:] != 0.0)):
        raise ValueError(
            "sample_weight must be finite and nonnegative on valid rows, "
            "with exact zeros on padded rows")
    H_hall = static_hall_linear_response(sigma_H)
    S_sharding = getattr(S_quadratic, "sharding", None)
    if isinstance(S_sharding, NamedSharding):
        H_hall = device_put_process_local(
            H_hall, NamedSharding(S_sharding.mesh, P()))
    result = _static_slab_photon_head_moment_chunk(
        q_cart, D_raw, H_hall, S_quadratic,
        jnp.asarray(n_valid, dtype=jnp.int32), jnp.asarray(weight))
    return result


@dataclass(frozen=True)
class StaticPhotonQ0FactorCarrier:
    """Bounded factors for the exact q=0 updates inserted into V and W; see docs/architecture/four_current_wiring.md."""

    bare_pair: tuple[jax.Array, jax.Array]
    screened_pairs: tuple[tuple[jax.Array, jax.Array], ...]
    family_plans: tuple = ()

    def __post_init__(self) -> None:
        if len(self.bare_pair) != 2:
            raise ValueError("static photon bare q=0 carrier needs one pair")
        if len(self.screened_pairs) != 9:
            raise ValueError(
                "static photon screened q=0 carrier needs the complete "
                f"3x3 moment grid (9 pairs); got {len(self.screened_pairs)}")


@dataclass(frozen=True)
class StaticSlabPhotonHeadCompletion:
    """Evidence and bounded runtime carrier for one slab q=0 completion."""

    bare_D_mean: np.ndarray
    screened_moments: np.ndarray
    cubature_receipt: object
    observed_physical_counts: tuple[int, int, int]
    observed_padded_solve_counts: tuple[int, int, int]
    max_backward_residual: float
    min_dyson_singular_value: float
    max_dyson_condition_number: float
    max_dyson_forward_error_bound: float
    mixed_scale_qstar: float
    mixed_convergence_error_ratios: tuple[float, float]
    ward_residual: float
    hermiticity_residual: float
    #: The Hall pseudovector that entered ``R(q)`` (bohr^-1) and where it
    #: came from (the authenticated artifact producer, or the announced
    #: ``sigma_H = 0`` default when no ``static_gauge_hall_file`` exists).
    sigma_H: np.ndarray
    hall_source: str
    q0_factors: StaticPhotonQ0FactorCarrier


_STATIC_PHOTON_DYSON_NUMERICAL_BUDGET = 1.0e-9
_STATIC_PHOTON_POLYGON_CONVERGENCE_RTOL = 1.0e-8
_STATIC_PHOTON_POLYGON_CONVERGENCE_ATOL = 1.0e-12


def _finite_static_photon_values(values, *, gate: str, label: str):
    """Return one finite nonempty float64 vector before any host reduction."""
    array = np.asarray(values, dtype=np.float64)
    if (array.ndim != 1 or array.size == 0
            or not np.all(np.isfinite(array))):
        raise ValueError(
            f"GATE {gate}: a photon-head certificate vector is invalid.\n"
            f"  got:  {label}: shape = {array.shape}, size = {array.size}, "
            f"all_finite = {bool(np.all(np.isfinite(array)))}\n"
            f"  want: {label} to be a nonempty finite 1-D vector\n"
            "  why:  reducing an empty or nonfinite diagnostic would let "
            "an unmeasured cubature/Dyson result pass its certificate")
    return array


def _static_photon_mixed_error_ratio(previous_blocks, current_blocks) -> float:
    """Return one finite mixed-error maximum without Python NaN swallowing."""
    if len(previous_blocks) != len(current_blocks) or not previous_blocks:
        raise ValueError(
            "GATE static_photon_polygon_nonfinite: cubature block lists "
            "cannot be compared.\n"
            f"  got:  previous_blocks = {len(previous_blocks)}, "
            f"current_blocks = {len(current_blocks)}\n"
            "  want: equal nonzero block counts at adjacent orders\n"
            "  why:  convergence is a pairwise norm ratio; unequal or "
            "empty lists do not define that ratio")
    ratios = []
    for block_index, (previous, current) in enumerate(
            zip(previous_blocks, current_blocks)):
        delta = float(np.linalg.norm(current - previous))
        norms = _finite_static_photon_values(
            (delta, float(np.linalg.norm(previous)),
             float(np.linalg.norm(current))),
            gate="static_photon_polygon_nonfinite",
            label=f"cubature block {block_index} norms")
        scale = float(np.max(norms[1:]))
        limit = (
            _STATIC_PHOTON_POLYGON_CONVERGENCE_ATOL
            + _STATIC_PHOTON_POLYGON_CONVERGENCE_RTOL * scale)
        ratio = delta / limit
        ratios.append(ratio)
    ratio_array = _finite_static_photon_values(
        ratios, gate="static_photon_polygon_nonfinite",
        label="cubature mixed-error ratios")
    return float(np.max(ratio_array))


def _reduce_static_photon_order_diagnostics(
    residuals,
    min_sigmas,
    max_conditions,
    conditioned_backward_errors,
) -> tuple[float, float, float, float]:
    """Check every order diagnostic finite, then reduce with NumPy."""
    vectors = tuple(
        _finite_static_photon_values(
            values, gate="static_photon_dyson_nonfinite", label=label)
        for values, label in (
            (residuals, "per-order backward errors"),
            (min_sigmas, "per-order minimum singular values"),
            (max_conditions, "per-order condition numbers"),
            (conditioned_backward_errors,
             "per-order conditioned backward errors"),
        )
    )
    if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
        raise ValueError(
            "GATE static_photon_dyson_nonfinite: per-order Dyson "
            "diagnostics have inconsistent shapes.\n"
            f"  got:  diagnostic shapes = {[v.shape for v in vectors]}\n"
            "  want: one equal-length vector per diagnostic\n"
            "  why:  maxima/minima from different cubature orders cannot "
            "form one numerical certificate")
    return (
        float(np.max(vectors[0])),
        float(np.min(vectors[1])),
        float(np.max(vectors[2])),
        float(np.max(vectors[3])),
    )


def _static_photon_dyson_forward_error_bound(
    max_conditioned_backward: float,
) -> float:
    r"""Return ``2 theta/(1-theta)`` for finite ``0 <= theta < 1``."""
    theta = float(max_conditioned_backward)
    if not np.isfinite(theta) or theta < 0.0:
        raise ValueError(
            "GATE static_photon_dyson_nonfinite: conditioned backward "
            "error is invalid.\n"
            f"  got:  theta = {theta!r}\n"
            "  want: finite theta >= 0\n"
            "  why:  a negative or nonfinite error cannot bound the "
            "forward error of the Dyson solve")
    if theta >= 1.0:
        raise ValueError(
            "GATE static_photon_dyson_forward_bound_denominator: the rigorous "
            "forward-error inequality requires theta < 1, where "
            "theta=kappa_F*max(backward_error,eps); "
            f"got theta={theta:.6e}")
    bound = float(2.0 * theta / (1.0 - theta))
    if not np.isfinite(bound):
        raise ValueError(
            "GATE static_photon_dyson_nonfinite: transformed forward-error "
            "bound is nonfinite.\n"
            f"  got:  forward_error_bound = {bound!r}, theta = {theta!r}\n"
            "  want: a finite forward-error bound\n"
            "  why:  a nonfinite bound cannot certify the inserted head")
    return bound


def _require_static_photon_numerical_certificate(
    D_mean,
    moments_mean,
    *,
    max_backward: float,
    min_sigma: float,
    max_condition: float,
    max_conditioned_backward: float,
    mixed_error_ratios,
) -> float:
    """Refuse an unconditioned solve or non-finite cubature convergence."""
    diagnostics = (
        max_backward, min_sigma, max_condition, max_conditioned_backward)
    if (not np.all(np.isfinite(D_mean))
            or not np.all(np.isfinite(moments_mean))
            or not np.all(np.isfinite(diagnostics))
            or min_sigma <= 0.0):
        raise ValueError(
            "GATE static_photon_dyson_nonfinite: coupled photon mini-BZ "
            "data fail the numerical certificate.\n"
            f"  got:  D_finite = {bool(np.all(np.isfinite(D_mean)))}, "
            f"moments_finite = {bool(np.all(np.isfinite(moments_mean)))}, "
            f"diagnostics = {diagnostics}, min_sigma = {min_sigma!r}\n"
            "  want: finite averages/diagnostics and min_sigma > 0\n"
            "  why:  a singular or nonfinite 4x4 Dyson solve has no "
            "trustworthy Gamma-cell limit")
    max_forward_error_bound = _static_photon_dyson_forward_error_bound(
        max_conditioned_backward)
    if max_forward_error_bound > _STATIC_PHOTON_DYSON_NUMERICAL_BUDGET:
        raise ValueError(
            "GATE static_photon_dyson_conditioning: the 4x4 Dyson solve "
            "missed its forward-error budget.\n"
            f"  got:  forward_error_bound = 2*theta/(1-theta) = "
            f"{max_forward_error_bound:.3e}, "
            f"theta = {max_conditioned_backward:.3e}, backward_error = "
            f"{max_backward:.3e}, min_sigma = {min_sigma:.3e}, "
            f"kappa = {max_condition:.3e}\n"
            f"  want: forward_error_bound <= "
            f"{_STATIC_PHOTON_DYSON_NUMERICAL_BUDGET:.1e}\n"
            "  why:  inserting a head whose rigorous error exceeds the "
            "budget would publish uncertified q=0 matrix elements")
    convergence = np.asarray(mixed_error_ratios, dtype=np.float64)
    if (convergence.shape != (2,) or not np.all(np.isfinite(convergence))):
        raise ValueError(
            "GATE static_photon_polygon_nonfinite: the fixed cubature "
            "ladder produced invalid convergence diagnostics.\n"
            f"  got:  convergence shape = {convergence.shape}, values = "
            f"{convergence.tolist()}\n"
            "  want: two finite error ratios for 16/24/32 orders\n"
            "  why:  without both adjacent-order ratios the fixed ladder "
            "cannot certify convergence")
    if convergence[-1] > 1.0:
        raise ValueError(
            "GATE static_photon_polygon_not_converged: the final polygon "
            "Duffy--Gauss order pair did not converge every dimensionless "
            "bare/screened moment under the mixed absolute+relative budget: "
            f"error_ratio={convergence[-1]:.3e} > 1.  The provider ladder "
            "is fixed; refusing insertion rather than accepting a caller "
            "dial.")
    return max_forward_error_bound


@functools.partial(jax.jit, static_argnames=('layout', 'plans', 'mesh_xy'))
def _photon_q0_factor_orbit(left, right, *, layout, plans, mesh_xy):
    """Transport both rank-four Γ factors through every typed little-group row before averaging their products."""
    from common.shard_map import shard_map
    from symmetry_maps import apply_band_matrix_symmetry
    from .photon_layout import _q0_local_factor_piece
    if not plans:
        return (jax.lax.with_sharding_constraint(left[None], NamedSharding(mesh_xy, P(None, None, 'x'))),
                jax.lax.with_sharding_constraint(right[None], NamedSharding(mesh_xy, P(None, None, 'y'))))
    sym = plans[0].sym
    rows = np.asarray(sym.active_symmetry_rows, dtype=np.int32)
    anti = sym.operation_rows(rows)[2]
    action = np.asarray(sym.lorentz_action(rows))

    def transform(factor, axis):
        spec = P(None, axis)
        @functools.partial(shard_map, mesh=mesh_xy, in_specs=spec,
                 out_specs=P(None, None, axis), check_vma=False)
        def local(value):
            pieces = []
            for family, channels in enumerate(((0,), (1, 2, 3))):
                plan = plans[family]
                extent = layout.carrier_extent(channels[0]) // layout.mesh_side
                table = jnp.asarray(plan.centroid_local_perm[rows])
                perm = jax.lax.dynamic_slice_in_dim(
                    table, jax.lax.axis_index(axis) * extent, extent, axis=1)
                data = jnp.stack([_q0_local_factor_piece(value,
                    axis_name=axis, local_extent=extent,
                    local_offset=layout.local_offset(channel),
                    logical_extent=layout.logical_extent(channel))
                    for channel in channels])
                gathered = jnp.take_along_axis(data[None],
                    perm[:, None, None, :], axis=-1)
                mixed = apply_band_matrix_symmetry(gathered,
                    antiunitary=jnp.asarray(anti),
                    component_mix=jnp.asarray(action[:, channels, :][:, :, channels]),
                    component_axis=1)
                pieces.extend(mixed[:, i] for i in range(len(channels)))
            return jnp.concatenate(pieces, axis=-1)
        return local(factor)
    return transform(left, 'x'), transform(right, 'y') / len(rows)


@functools.partial(jax.jit, static_argnames=("mesh_xy",))
def _fold_photon_q0_response(S_direct, Y_x, W_gamma, Z_y, volume, *, mesh_xy):
    """Fold four coordinate pairs in order through one resident distributed body."""
    def pair(S_effective, index):
        a, b = index // 2, index % 2
        folded = fold_small_head_wings_sharded(
            S_direct[a, b], Y_x[a], W_gamma, Z_y[b], volume, mesh_xy=mesh_xy)
        return S_effective.at[a, b].set(folded), None
    S_effective, _ = jax.lax.scan(pair, S_direct, jnp.arange(4), unroll=1)
    return canonicalize_static_gauge_q2_tensor(S_effective)


def complete_static_slab_photon_q0(
    V_packed: jax.Array,
    W_packed: jax.Array,
    response,
    g0_X: jax.Array,
    g0_Y: jax.Array,
    cubature_receipt,
    *,
    mesh_xy: Mesh,
    family_plans: tuple = (),
) -> tuple[jax.Array, jax.Array, StaticSlabPhotonHeadCompletion]:
    """Complete bare and screened packed photon operators in the Γ cell; see docs/architecture/four_current_wiring.md."""
    from .photon_layout import (
        MAX_Q0_UPDATE_RANK, add_photon_q0_low_rank)
    from .static_gauge_response import require_static_photon_head_response

    response = require_static_photon_head_response(response, mesh_xy)
    layout = response.layout
    packed_shape = (int(V_packed.shape[0]), layout.packed_extent,
                    layout.packed_extent)
    if (tuple(V_packed.shape) != packed_shape
            or tuple(W_packed.shape) != packed_shape):
        raise ValueError(
            "coupled photon head requires equal packed V/W bodies; got "
            f"V={V_packed.shape}, W={W_packed.shape}, expected={packed_shape}")
    factor_shape = (MAX_Q0_UPDATE_RANK, layout.packed_extent)
    if tuple(g0_X.shape) != factor_shape or tuple(g0_Y.shape) != factor_shape:
        raise ValueError(
            f"packed Γ vectors must both be {factor_shape}; got "
            f"{g0_X.shape}/{g0_Y.shape}")
    from vcoul import validate_slab_minibz_photon_receipt
    cubature_receipt = validate_slab_minibz_photon_receipt(
        cubature_receipt)
    cell_volume = float(cubature_receipt.cell_volume)

    W_gamma = W_packed[0]
    S_effective = _fold_photon_q0_response(
        response.S_direct, response.Y_x, W_gamma, response.Z_y,
        float(cell_volume), mesh_xy=mesh_xy)
    YW_y, WZ_x = small_head_wing_halves_sharded(
        response.Y_x, W_gamma, response.Z_y, mesh_xy=mesh_xy)
    jax.block_until_ready((S_effective, YW_y, WZ_x))
    effective_ward, effective_hermiticity = static_gauge_tensor_residuals(
        S_effective)
    if effective_ward > _STATIC_GAUGE_WARD_RESIDUAL_MAX:
        raise ValueError(
            "GATE static_gauge_head_fold_ward: the folded response violates "
            "the Ward identity.\n"
            f"  got:  ward_residual = {effective_ward:.6e}\n"
            f"  want: ward_residual <= "
            f"{_STATIC_GAUGE_WARD_RESIDUAL_MAX:.1e}\n"
            "  why:  a nonconserving folded response cannot define the "
            "charge q->0 head")
    if effective_hermiticity > _STATIC_GAUGE_HERMITICITY_RESIDUAL_MAX:
        raise ValueError(
            "GATE static_gauge_head_fold_hermiticity: the folded response "
            "violates Hermiticity.\n"
            f"  got:  hermiticity_residual = {effective_hermiticity:.6e}\n"
            f"  want: hermiticity_residual <= "
            f"{_STATIC_GAUGE_HERMITICITY_RESIDUAL_MAX:.1e}\n"
            "  why:  a non-Hermitian static response would insert a "
            "nonphysical q->0 screened interaction")

    # Exactly three provider-issued rules are solved sequentially.  All three
    # have the same padded carrier, so the JAX graph compiles once; the
    # physical and executed row counts remain separate evidence.
    per_order_moments = []
    per_order_D = []
    residuals = []
    min_sigmas = []
    max_conditions = []
    conditioned_backward_errors = []
    observed_physical = []
    observed_padded = []
    for chunk in cubature_receipt.chunks:
        q_host = np.asarray(chunk.q_cart, dtype=np.float64)
        weight = np.asarray(chunk.sample_weight, dtype=np.float64)
        n_valid = int(chunk.physical_count)
        (moments, bare_sum, returned_count, backward, chunk_sigma_min,
         chunk_condition_max,
         chunk_conditioned_backward) = static_slab_photon_head_moment_chunk(
            q_host, chunk.D_raw, response.sigma_H, S_effective,
            n_valid, weight)
        (moment_host, D_host, count_host, residual_host, sigma_host,
         condition_host, conditioned_backward_host) = jax.device_get((
             moments, bare_sum, returned_count, backward,
             chunk_sigma_min, chunk_condition_max,
             chunk_conditioned_backward))
        returned = int(np.asarray(count_host))
        if returned != n_valid:
            raise RuntimeError(
                "static photon head moment kernel changed physical_count")
        measure = float(np.sum(weight[:n_valid]))
        if not np.isfinite(measure) or measure <= 0.0:
            raise ValueError(
                "provider-issued photon cubature has nonpositive measure")
        per_order_moments.append(
            np.asarray(moment_host, dtype=np.complex128) / measure)
        per_order_D.append(
            np.asarray(D_host, dtype=np.complex128) / measure)
        residuals.append(float(np.asarray(residual_host)))
        min_sigmas.append(float(np.asarray(sigma_host)))
        max_conditions.append(float(np.asarray(condition_host)))
        conditioned_backward_errors.append(
            float(np.asarray(conditioned_backward_host)))
        observed_physical.append(returned)
        observed_padded.append(int(q_host.shape[0]))

    if (tuple(observed_physical) != cubature_receipt.physical_counts
            or tuple(observed_padded) != cubature_receipt.padded_counts):
        raise RuntimeError(
            "executed photon solve counts differ from the vcoul receipt: "
            f"physical={tuple(observed_physical)}/"
            f"{cubature_receipt.physical_counts}, padded="
            f"{tuple(observed_padded)}/{cubature_receipt.padded_counts}")
    D_mean = per_order_D[-1]
    moments_mean = per_order_moments[-1]
    qstar = np.sqrt(float(cubature_receipt.polygon_area))

    def _dimensionless_moments(moment):
        scaled = np.empty_like(moment)
        degrees = (0, 1, 1)
        for u in range(3):
            for v in range(3):
                scaled[u, v] = moment[u, v] / (
                    qstar ** (degrees[u] + degrees[v]))
        return scaled

    def _mixed_error_ratio(index):
        previous_blocks = [per_order_D[index - 1]]
        current_blocks = [per_order_D[index]]
        previous_scaled = _dimensionless_moments(
            per_order_moments[index - 1])
        current_scaled = _dimensionless_moments(per_order_moments[index])
        for u in range(3):
            for v in range(3):
                previous_blocks.append(previous_scaled[u, v])
                current_blocks.append(current_scaled[u, v])
        return _static_photon_mixed_error_ratio(
            previous_blocks, current_blocks)

    mixed_error_ratios = tuple(
        _mixed_error_ratio(index)
        for index in range(1, len(per_order_D)))
    (max_residual, min_sigma, max_condition,
     max_conditioned_backward) = _reduce_static_photon_order_diagnostics(
         residuals, min_sigmas, max_conditions,
         conditioned_backward_errors)
    max_forward_error_bound = _require_static_photon_numerical_certificate(
        D_mean, moments_mean,
        max_backward=max_residual,
        min_sigma=min_sigma,
        max_condition=max_condition,
        max_conditioned_backward=max_conditioned_backward,
        mixed_error_ratios=mixed_error_ratios)

    dtype = V_packed.dtype
    sh_x = NamedSharding(mesh_xy, P(None, "x"))
    sh_y = NamedSharding(mesh_xy, P(None, "y"))
    volume = jnp.asarray(float(cell_volume), dtype=jnp.float64)
    with mesh_xy:
        left_bare = jax.lax.with_sharding_constraint(
            jnp.conj(g0_X).astype(dtype), sh_x)
        right_bare = jax.lax.with_sharding_constraint(
            jnp.einsum(
                "AB,Bj->Aj", jnp.asarray(D_mean, dtype=dtype), g0_Y,
                optimize=True) / volume,
            sh_y)
    images = _photon_q0_factor_orbit(
        left_bare, right_bare, layout=layout, plans=family_plans, mesh_xy=mesh_xy)
    V_packed = add_photon_q0_low_rank(
        V_packed, layout, mesh_xy,
        left_rows_X=images[0], right_rows_Y=images[1])

    left_basis = (
        left_bare,
        jax.lax.with_sharding_constraint(jnp.swapaxes(WZ_x[0], 0, 1), sh_x),
        jax.lax.with_sharding_constraint(jnp.swapaxes(WZ_x[1], 0, 1), sh_x),
    )
    right_basis = (g0_Y, YW_y[0], YW_y[1])
    screened_pairs = []
    for u in range(3):
        for v in range(3):
            with mesh_xy:
                right_rows = jax.lax.with_sharding_constraint(
                    jnp.einsum(
                        "AB,Bj->Aj",
                        jnp.asarray(moments_mean[u, v], dtype=dtype),
                        right_basis[v], optimize=True) / volume,
                    sh_y)
            images = _photon_q0_factor_orbit(left_basis[u], right_rows,
                layout=layout, plans=family_plans, mesh_xy=mesh_xy)
            W_packed = add_photon_q0_low_rank(
                W_packed, layout, mesh_xy,
                left_rows_X=images[0], right_rows_Y=images[1])
            screened_pairs.append((left_basis[u], right_rows))

    evidence = StaticSlabPhotonHeadCompletion(
        bare_D_mean=D_mean,
        screened_moments=moments_mean,
        cubature_receipt=cubature_receipt,
        observed_physical_counts=tuple(observed_physical),
        observed_padded_solve_counts=tuple(observed_padded),
        max_backward_residual=max_residual,
        min_dyson_singular_value=min_sigma,
        max_dyson_condition_number=max_condition,
        max_dyson_forward_error_bound=max_forward_error_bound,
        mixed_scale_qstar=qstar,
        mixed_convergence_error_ratios=mixed_error_ratios,
        ward_residual=max(float(response.ward_residual), effective_ward),
        hermiticity_residual=max(
            float(response.hermiticity_residual), effective_hermiticity),
        sigma_H=np.asarray(
            jax.device_get(response.sigma_H), dtype=np.float64),
        hall_source=str(response.hall_source),
        q0_factors=StaticPhotonQ0FactorCarrier(
            bare_pair=(left_bare, right_bare),
            screened_pairs=tuple(screened_pairs), family_plans=family_plans),
    )
    return V_packed, W_packed, evidence


def resolve_head_S_cart(restart_file=None, *, input_file=None, wfn=None,
                        sym=None, meta=None, params=None, print_fn=print):
    """The ``S`` tensor behind the restart's ``whead`` — read it, or rebuild it; see docs/architecture/four_current_wiring.md."""
    if restart_file is not None:
        try:
            import h5py
            with h5py.File(restart_file, "r") as f:
                if "S_cart_head" in f:
                    S = np.asarray(f["S_cart_head"][:], dtype=np.complex128)
                    if S.shape == (3, 3):
                        return S, f"restart S_cart_head ({os.path.basename(restart_file)})"
                    print_fn(f"BSE head: restart S_cart_head has shape "
                             f"{S.shape}, expected (3,3); ignoring it")
        except Exception as exc:                    # never crash a load on this
            print_fn(f"BSE head: could not read S_cart_head ({exc})")

    if wfn is None or sym is None or meta is None or input_file is None:
        return None, ("no S_cart: the restart carries none and the caller "
                      "passed no wfn/sym/meta to rebuild one from dipole.h5")
    dipole_path = os.path.join(os.path.dirname(os.path.abspath(input_file)),
                               "dipole.h5")
    if not os.path.exists(dipole_path):
        return None, f"no S_cart: restart carries none and {dipole_path} is absent"
    try:
        S = build_S_cart_omega(wfn, sym, meta, params or {}, dipole_path, 0.0,
                               eta=float((params or {}).get("wcoul0_eta", 0.0) or 0.0),
                               print_fn=print_fn)
    except Exception as exc:
        return None, f"no S_cart: rebuild from dipole.h5 failed ({exc})"
    return S, "rebuilt from dipole.h5 (S(ω=0), the GW run's own route)"


class HeadResolver:
    """Memoized q=0 head-sample resolver for a single GW run; see docs/architecture/four_current_wiring.md."""

    __slots__ = ("_params", "_input_dir", "_wfn", "_sym", "_meta",
                 "_print_fn", "_cache", "_direct_cache", "_policy",
                 "_screened")

    def __init__(self, config, input_dir, wfn, sym, meta, print_fn,
                 q0_certificate_fn=None):
        head = config.head
        from common.four_current_model import (
            resolve_four_current_representation)
        representation = resolve_four_current_representation(
            bool(config.bispinor), config.bispinor_gw)
        self._params = {
            # These are GW-run controls, not head sub-config fields.  The
            # dipole provenance reader must compare against the consumer's
            # requested window rather than reconstructing the writer defaults.
            "nval": config.nval,
            "ncond": config.ncond,
            "nband": config.nband,
            "bispinor_gw": getattr(
                getattr(config, "bispinor_gw", "bare_transverse"),
                "value", getattr(config, "bispinor_gw", "bare_transverse")),
            "_four_current_bispinor": bool(config.bispinor),
            "_charge_bispinor": representation.scalar_head_bispinor,
            "wcoul0_source": head.wcoul0_source,
            "wcoul0_eta": head.wcoul0_eta,
            "vhead": head.vhead,
            "whead_0freq": head.whead_0freq,
            "whead_imfreq": head.whead_imfreq,
            "head_minibz_average": head.head_minibz_average,
            "bgw_metal_q0_treatment": head.bgw_metal_q0_treatment,
            "_q0_certificate_fn": q0_certificate_fn,
        }
        from gw.gw_config import coerce_head_correction
        self._policy = coerce_head_correction(
            getattr(head, "correction", "full"))
        self._screened = bool(getattr(config, "do_screened", True))
        self._input_dir = input_dir
        self._wfn = wfn
        self._sym = sym
        self._meta = meta
        self._print_fn = print_fn
        self._cache: dict[tuple[float, float], HeadSample] = {}
        self._direct_cache: dict[tuple[float, float], HeadSample] = {}

    def _cache_key(self, omega) -> tuple[float, float]:
        z = complex(omega)
        return (round(z.real, 12), round(z.imag, 12))

    @property
    def wfn(self):
        """The WFN whose transition manifold this provider validates."""
        return self._wfn

    def direct_at(self, omega) -> HeadSample:
        """Return the configured no-local-field/direct diagnostic sample."""
        key = self._cache_key(omega)
        cached = self._direct_cache.get(key)
        if cached is not None:
            return cached
        sample = resolve_head_sample(
            self._params, self._input_dir, self._wfn, self._sym,
            self._meta, self._print_fn, omega=omega,
        )
        self._direct_cache[key] = sample
        return sample

    def install_samples(self, samples) -> None:
        """Install finalized samples, rejecting incomplete or double folds."""
        from gw.gw_config import HeadCorrection

        for sample in tuple(samples):
            if not isinstance(sample, HeadSample):
                raise TypeError("head provider accepts HeadSample objects only")
            if self._policy is HeadCorrection.OFF:
                raise ValueError(
                    "head_correction=off cannot install a screened head")
            if (self._policy is HeadCorrection.FULL
                    and sample.response_kind
                    not in (HeadResponseKind.FULL_LOCAL_FIELDS,
                            HeadResponseKind.MICRO_REDUCIBLE,
                            HeadResponseKind.OVERRIDE)):
                raise ValueError(
                    "head_correction=full requires a once-folded "
                    "full_local_fields response or an already "
                    "micro_reducible response; got "
                    f"{sample.response_kind.value}. Refusing the unfolded "
                    "epsilon head.")
            self._cache[self._cache_key(sample.omega)] = sample

    def at(self, omega) -> HeadSample:
        """Resolve the policy-selected finalized head at ``omega`` in Ry."""
        from gw.gw_config import HeadCorrection

        key = self._cache_key(omega)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._policy is HeadCorrection.OFF:
            sample = HeadSample(
                vc0=0.0j, wcoul0=0.0j, source="head_correction=off",
                omega=complex(omega), S_cart=None,
                response_kind=HeadResponseKind.OFF)
            self._cache[key] = sample
            return sample
        direct = self.direct_at(omega)
        if direct.response_kind is HeadResponseKind.OVERRIDE:
            self._cache[key] = direct
            return direct
        if (self._policy is HeadCorrection.NO_LOCAL_FIELDS
                or not self._screened):
            self._cache[key] = direct
            return direct
        raise RuntimeError(
            "head_correction=full was requested, but no finalized head was "
            f"installed at omega={complex(omega)} Ry. Build compatible "
            "wings/body (direct RPA) or a micro-reducible BSE resolvent; "
            "the direct epsilon head is not a silent fallback.")


def format_head_sample_diagnostics(head: HeadSample, *, include_screened: bool = True) -> str:
    """Return a compact diagnostic summary for one resolved head sample."""

    lines = [
        "",
        "-" * 72,
        "  FINITE-SIZE CORRECTIONS",
        "-" * 72,
        f"  Head source: {head.source}",
        f"  v(q→0)  = {head.vc0.real:12.3f} a.u.  (bare Coulomb head)",
    ]
    if include_screened:
        if abs(head.omega) > 1.0e-14:
            lines.append(f"  Head frequency ω = {head.omega} Ry")
        lines.append(f"  W(q→0)  = {head.wcoul0.real:12.3f} a.u.  (screened Coulomb head)")
        lines.append(
            f"  ΔW      = {(head.wcoul0.real - head.vc0.real):12.3f} a.u.  (screening correction)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic GN-PPM scalar head
# ---------------------------------------------------------------------------

def fit_head_ppm(
    vc0: float,
    wcoul0_static: float,
    wcoul0_probe: float,
    probe_omega: complex,
) -> HeadGNParams:
    """Fit a scalar PPM pole from two W^c head samples; see docs/architecture/four_current_wiring.md."""

    z = complex(probe_omega)
    omega_2_sq = float((z * z).real)
    # Real-axis log-magnitude for diagnostics ("ω_p" was historically the
    # imaginary-axis magnitude; for HL it's the real frequency itself).
    probe_mag = float(abs(z))

    w1 = wcoul0_static - vc0
    w2 = wcoul0_probe - vc0

    denom = w1 - w2
    if abs(denom) < 1.0e-30:
        return HeadGNParams(
            omega_h_sq=1.0,
            omega_h=1.0,
            B_h=0.0,
            R_h=0.0,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=probe_mag,
        )

    omega_h_sq = -w2 * omega_2_sq / denom
    B_h = -w1 * omega_h_sq

    if omega_h_sq <= 0.0:
        omega_h = abs(omega_h_sq) ** 0.5 if omega_h_sq != 0.0 else 1.0
        # Bug fix (2026-07-04): B_h at L318 used the SIGNED (negative) omega_h_sq
        # while omega_h = abs(...)^0.5 is positive, so R_h = B_h/(2 omega_h) came
        # out sign-flipped vs the positive branch — the whole q->0 head Sigma_c
        # (hundreds of meV) had the wrong sign whenever the GN head fit went
        # imaginary.  Use the magnitude so |R_h| is continuous across Omega^2=0.
        B_h = -w1 * abs(omega_h_sq)
        R_h = B_h / (2.0 * omega_h) if omega_h > 1.0e-30 else 0.0
        return HeadGNParams(
            omega_h_sq=omega_h_sq,
            omega_h=omega_h,
            B_h=B_h,
            R_h=R_h,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=probe_mag,
        )

    omega_h = omega_h_sq ** 0.5
    R_h = B_h / (2.0 * omega_h)
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=w2,
        vc0=vc0,
        omega_p=probe_mag,
    )


def fit_head_ppm_from_samples(
    head_static: HeadSample,
    head_probe: HeadSample,
    *,
    probe_omega: complex,
) -> HeadGNParams:
    """Fit the scalar PPM head from resolved static and probe-frequency samples; see docs/architecture/four_current_wiring.md."""
    return fit_head_ppm(
        vc0=float(head_static.vc0.real),
        wcoul0_static=float(head_static.wcoul0.real),
        wcoul0_probe=float(head_probe.wcoul0.real),
        probe_omega=probe_omega,
    )


def fit_head_hl_analytic(
    vc0: float,
    wcoul0_static: float,
    omega_p_sq_ry: float,
) -> HeadGNParams:
    """Set the HL-PPM head pole analytically from the bulk plasmon, BGW-style; see docs/architecture/four_current_wiring.md."""
    w1 = wcoul0_static - vc0  # W^c(0) head, in a.u.
    if abs(w1) < 1.0e-30 or abs(vc0) < 1.0e-30:
        return HeadGNParams(
            omega_h_sq=1.0, omega_h=1.0, B_h=0.0, R_h=0.0,
            wc_head_0=w1, wc_head_iwp=0.0, vc0=vc0,
            omega_p=float(omega_p_sq_ry) ** 0.5 if omega_p_sq_ry > 0 else 1.0,
        )

    # I_ε_head = 1 − ε⁻¹(0,0) = 1 − W(0)/v(0) = (v − W)/v = −W^c/v.
    I_eps_head = -w1 / float(vc0)
    if I_eps_head <= 0.0:
        I_eps_head = 1.0  # graceful fallback; prevents sqrt of negative

    omega_h_sq = float(omega_p_sq_ry) / I_eps_head
    omega_h = omega_h_sq ** 0.5
    B_h = -w1 * omega_h_sq
    R_h = B_h / (2.0 * omega_h)
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=0.0,  # not used in this analytic path
        vc0=vc0,
        omega_p=float(omega_p_sq_ry) ** 0.5,
    )


def fit_head_hl_analytic_from_sample(
    head_static: HeadSample,
    *,
    omega_p_sq_ry: float,
) -> HeadGNParams:
    """Wrapper for :func:`fit_head_hl_analytic` using a resolved head sample."""
    return fit_head_hl_analytic(
        vc0=float(head_static.vc0.real),
        wcoul0_static=float(head_static.wcoul0.real),
        omega_p_sq_ry=omega_p_sq_ry,
    )


def fit_head_with_fixed_omega(
    vc0: float,
    wcoul0_static: float,
    omega_h_ry: float,
) -> HeadGNParams:
    """Build head params with a user-supplied pole frequency Ω_h; see docs/architecture/four_current_wiring.md."""
    w1 = wcoul0_static - vc0
    omega_h = float(omega_h_ry)
    omega_h_sq = omega_h ** 2
    B_h = -w1 * omega_h_sq
    R_h = B_h / (2.0 * omega_h) if abs(omega_h) > 1.0e-30 else 0.0
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=0.0,
        vc0=vc0,
        omega_p=omega_h,
    )


def fit_head_with_fixed_omega_from_sample(
    head_static: HeadSample,
    *,
    omega_h_ry: float,
) -> HeadGNParams:
    """Wrapper for :func:`fit_head_with_fixed_omega` using a HeadSample."""
    return fit_head_with_fixed_omega(
        vc0=float(head_static.vc0.real),
        wcoul0_static=float(head_static.wcoul0.real),
        omega_h_ry=omega_h_ry,
    )


# ---------------------------------------------------------------------------
# Exact static COHSEX head
# ---------------------------------------------------------------------------

def compute_static_head_terms(
    *,
    vc0: complex,
    wcoul0_static: complex,
    occ: np.ndarray | jnp.ndarray,
    cell_volume: float,
    nk_tot: int,
    source: str = "unknown",
) -> StaticHeadTerms:
    """Build exact static COHSEX head terms (Σ^X, Σ^SX, Σ^{SX-X}, Σ^COH) in band space; see docs/architecture/four_current_wiring.md."""

    occ_arr = jnp.asarray(occ, dtype=jnp.complex128)
    ones = jnp.ones_like(occ_arr, dtype=jnp.complex128)
    pref = jnp.asarray(1.0 / (float(cell_volume) * float(nk_tot)), dtype=jnp.complex128)

    v_h = jnp.asarray(vc0, dtype=jnp.complex128)
    w_h = jnp.asarray(wcoul0_static, dtype=jnp.complex128)
    wc_h = w_h - v_h

    sigma_x_diag = -(v_h * pref) * occ_arr
    sigma_sx_diag = -(w_h * pref) * occ_arr
    sigma_sx_minus_x_diag = -(wc_h * pref) * occ_arr
    sigma_coh_diag = 0.5 * (wc_h * pref) * ones

    return StaticHeadTerms(
        sigma_x_diag=sigma_x_diag,
        sigma_sx_diag=sigma_sx_diag,
        sigma_sx_minus_x_diag=sigma_sx_minus_x_diag,
        sigma_coh_diag=sigma_coh_diag,
        vc0=complex(vc0),
        wcoul0=complex(wcoul0_static),
        wc_head_0=complex(wcoul0_static) - complex(vc0),
        source=source,
    )


def compute_static_head_terms_from_sample(head: HeadSample, *,
                                          occ, cell_volume: float,
                                          nk_tot: int) -> StaticHeadTerms:
    """Build exact static COHSEX head terms from a resolved head sample."""
    return compute_static_head_terms(vc0=head.vc0, wcoul0_static=head.wcoul0,
                                     occ=occ, cell_volume=cell_volume,
                                     nk_tot=nk_tot, source=head.source)


def format_static_head_diagnostics(head: StaticHeadTerms) -> str:
    """Return a concise summary of the exact static COHSEX head terms."""

    x_occ = _representative_entry(head.sigma_x_diag)
    sx_occ = _representative_entry(head.sigma_sx_diag)
    sxmx_occ = _representative_entry(head.sigma_sx_minus_x_diag)
    coh_all = _representative_entry(head.sigma_coh_diag)
    lines = [
        "",
        "-" * 72,
        "  STATIC HEAD TERMS (exact COHSEX / BGW-style)",
        "-" * 72,
        f"  Head source: {head.source}",
        f"  v_h(q→0)           = {head.vc0.real:12.6f} a.u.",
        f"  W_h(q→0, ω=0)      = {head.wcoul0.real:12.6f} a.u.",
        f"  W_h^c              = {head.wc_head_0.real:12.6f} a.u.",
        f"  Σ^X head (occ)     = {x_occ.real:12.6e} Ry",
        f"  Σ^SX head (occ)    = {sx_occ.real:12.6e} Ry",
        f"  Σ^(SX-X) head(occ) = {sxmx_occ.real:12.6e} Ry",
        f"  Σ^COH head (all)   = {coh_all.real:12.6e} Ry",
    ]
    return "\n".join(lines)


@functools.partial(jax.jit, static_argnames=('nk_tot', 'nb'))
def _expand_band_diagonal_to_kij_jit(diag, *, nk_tot: int, nb: int):
    """JIT'd body of :func:`expand_band_diagonal_to_kij`."""
    eye = jnp.eye(nb, dtype=jnp.complex128)
    if diag.ndim == 2:
        return eye[None, :, :] * diag[:, :, None]
    one_k = eye[None, :, :] * diag[None, :, None]
    return jnp.broadcast_to(one_k, (nk_tot, nb, nb))


def expand_band_diagonal_to_kij(diag: jnp.ndarray, nk_tot: int) -> jnp.ndarray:
    """Broadcast a band-diagonal shift to a dense ``(nk, nb, nb)`` matrix; see docs/architecture/four_current_wiring.md."""
    diag_arr = jnp.asarray(diag, dtype=jnp.complex128)
    if diag_arr.ndim == 1:
        nb = int(diag_arr.shape[0])
    elif diag_arr.ndim == 2:
        if int(diag_arr.shape[0]) != int(nk_tot):
            raise ValueError(
                f"k-dependent diagonal has {diag_arr.shape[0]} rows, "
                f"expected nk_tot={nk_tot}")
        nb = int(diag_arr.shape[1])
    else:
        raise ValueError("band diagonal must be (nb,) or (nk,nb)")
    return _expand_band_diagonal_to_kij_jit(diag_arr, nk_tot=int(nk_tot), nb=nb)


def static_head_terms_to_kij(
    head: StaticHeadTerms,
    *,
    nk_tot: int,
    do_screened: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Expand exact static head shifts to dense ``(k, i, j)`` matrices; see docs/architecture/four_current_wiring.md."""

    sx_diag = head.sigma_sx_diag if do_screened else head.sigma_x_diag
    return (
        expand_band_diagonal_to_kij(sx_diag, nk_tot),
        expand_band_diagonal_to_kij(head.sigma_coh_diag, nk_tot),
    )


def compute_ppm_head_sigma_kij(
    head: HeadGNParams,
    *,
    omega_grid_ry: np.ndarray,
    enk_ry: np.ndarray,
    efermi_ry: float,
    n_occ: int,
    cell_volume: float,
    nk_tot: int,
    eta: float = 1.0e-6,
) -> np.ndarray:
    """q→0, G=G'=0 head contribution to PPM ``Σ^c_kij(ω)``; see docs/architecture/four_current_wiring.md."""

    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    n_omega = int(omega.size)
    nk, nb = enk.shape
    sigma_diag = compute_ppm_head_sigma_diag(
        head, omega_grid_ry=omega, enk_ry=enk, efermi_ry=efermi_ry,
        n_occ=n_occ, cell_volume=cell_volume, nk_tot=nk_tot, eta=eta)
    out = np.zeros((n_omega, nk, nb, nb), dtype=np.complex128)
    idx = np.arange(nb)
    out[:, :, idx, idx] = sigma_diag
    return out


def compute_ppm_head_sigma_diag(
    head: HeadGNParams,
    *,
    omega_grid_ry: np.ndarray,
    enk_ry: np.ndarray,
    efermi_ry: float,
    n_occ: int,
    occupations: np.ndarray | None = None,
    cell_volume: float,
    nk_tot: int,
    eta: float = 1.0e-6,
) -> np.ndarray:
    """Band-DIAGONAL of :func:`compute_ppm_head_sigma_kij` — ``(nω, nk, nb)``; see docs/architecture/four_current_wiring.md."""
    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    n_omega = int(omega.size)
    nk, nb = enk.shape
    if abs(head.R_h) < 1.0e-30 or abs(head.omega_h) < 1.0e-30:
        return np.zeros((n_omega, nk, nb), dtype=np.complex128)

    if occupations is None:
        f = np.zeros((nb,), dtype=np.float64)
        f[: max(0, min(int(n_occ), nb))] = 1.0
    else:
        f = np.asarray(occupations, dtype=np.float64)
    return compute_complex_pole_head_sigma_diag(
        omega_grid_ry=omega,
        enk_ry=enk,
        efermi_ry=efermi_ry,
        occupations=f,
        poles_ry=np.asarray([head.omega_h - 1j * eta]),
        residues_ry=np.asarray([head.R_h]),
        cell_volume=cell_volume,
        nk_tot=nk_tot,
    )


def on_shell_occupied_head_sigma_ry(
    head: HeadGNParams,
    *,
    cell_volume: float,
    nk_tot: int,
    eta: float = 1.0e-6,
) -> float:
    """Re(Σ^head) for an OCCUPIED band evaluated ON SHELL (ω = ε_nk − E_F); see docs/architecture/four_current_wiring.md."""
    val = compute_ppm_head_sigma_diag(
        head,
        omega_grid_ry=np.zeros(1, dtype=np.float64),
        enk_ry=np.zeros((1, 1), dtype=np.float64),
        efermi_ry=0.0,
        n_occ=1,
        cell_volume=cell_volume,
        nk_tot=nk_tot,
        eta=eta,
    )
    return float(np.real(val[0, 0, 0]))


def compute_complex_pole_head_sigma_diag(
    *,
    omega_grid_ry: np.ndarray,
    enk_ry: np.ndarray,
    efermi_ry: float,
    occupations: np.ndarray,
    poles_ry: np.ndarray,
    residues_ry: np.ndarray,
    cell_volume: float,
    nk_tot: int,
) -> np.ndarray:
    """Band-diagonal head self-energy for generic retarded complex poles; see docs/architecture/four_current_wiring.md."""
    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    nk, nb = enk.shape
    occ = np.asarray(occupations, dtype=np.float64)
    if occ.shape == (nb,):
        occ = np.broadcast_to(occ[None, :], (nk, nb))
    elif occ.shape != (nk, nb):
        raise ValueError(
            f"occupations must have shape {(nb,)} or {(nk, nb)}, got "
            f"{occ.shape}")
    poles = np.asarray(poles_ry, dtype=np.complex128).reshape(-1)
    # Preserve a real residue dtype.  Besides avoiding a needless complex
    # multiply for real fits, this keeps the one-real-pole PPM wrapper
    # bit-identical to its historical arithmetic.
    residues = np.asarray(residues_ry).reshape(-1)
    if residues.shape != poles.shape:
        raise ValueError("poles_ry and residues_ry must have the same length")
    if poles.size == 0:
        return np.zeros((omega.size, nk, nb), dtype=np.complex128)

    eps_rel = enk - float(efermi_ry)
    delta = omega[:, None, None] - eps_rel[None, :, :]              # (nω, nk, nb)
    pole = poles[:, None, None, None]
    f = occ[None, None, :, :]
    occ_term = f / (delta[None, :, :, :] + pole)
    emp_term = (1.0 - f) / (delta[None, :, :, :] - pole)
    pref = residues / (float(cell_volume) * float(nk_tot))
    sigma_diag = np.sum(
        pref[:, None, None, None] * (occ_term + emp_term), axis=0)
    return np.asarray(sigma_diag, dtype=np.complex128)


def format_head_diagnostics(head: HeadGNParams, cell_volume: float) -> str:
    """Return a short multiline diagnostic summary for the scalar head fit."""

    lines = [
        "",
        "-" * 72,
        "  HEAD CORRECTION (scalar GN, separate from ISDF body)",
        "-" * 72,
        f"  v(q→0)             = {head.vc0:12.3f} a.u.",
        f"  W^c(q→0, ω=0)      = {head.wc_head_0:12.3f} a.u.",
        f"  W^c(q→0, ω=iωp)    = {head.wc_head_iwp:12.3f} a.u.  [ωp={head.omega_p:.4f} Ry]",
        f"  Ω_h²               = {head.omega_h_sq:12.6f} Ry²",
        f"  Ω_h                = {head.omega_h:12.6f} Ry  ({head.omega_h * RYD_TO_EV:.6f} eV)",
        f"  B_h                = {head.B_h:12.6f} Ry² · a.u.",
        f"  R_h                = {head.R_h:12.6f} Ry · a.u.",
    ]
    if abs(head.omega_h) > 1.0e-30:
        lines.append(
            f"  R_h / (Ω_h · vol)  = {head.R_h / (head.omega_h * cell_volume):12.6e} (Ry)"
        )
    else:
        lines.append("  R_h / (Ω_h · vol)  = 0.0 (degenerate)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Rank-1 head injection in the (μ, ν) ISDF basis at q=0
# ---------------------------------------------------------------------------
#
# ``compute_vcoul`` zeroes the ``G=G'=0`` element of ``v(q+G)`` at q=0 to
# avoid the divergence; the BGW-equivalent mini-BZ-averaged value is the
# scalar ``vhead = v_h``.  In the centroid basis the missing piece factors as
#
#     ΔV_{q=0,μν} = (v_h / V_cell) · ζ̄(0, μ, G=0) · ζ(0, ν, G=0)
#                 = (v_h / V_cell) · conj(G0[μ]) · G0[ν]            (rank 1)
#
# The ``1/V_cell`` factor matches the LORRAX storage convention for
# ``V_qmunu`` / ``W_qmunu`` (see ``scripts/checks/sigma_direct_check.py`` for the canonical
# reference).  Conjugation lands on ``μ`` because
# ``V_{qμν} = Σ_GG' ζ*(q,μ,G) v(G,G') ζ(q,ν,G')``.


def _head_rank1_scalars(vhead, whead, cell_volume, omega_index, dtype):
    """Resolve (v_scalar, w_scalar) = (head / V_cell) for a rank-1 update."""
    inv_V = 1.0 / float(cell_volume)
    v_scalar = (jnp.asarray(complex(vhead) * inv_V, dtype=dtype)
                if vhead is not None else None)
    if whead is None:
        return v_scalar, None
    whead_arr = jnp.asarray(whead, dtype=jnp.complex128)
    w_val = whead_arr if whead_arr.ndim == 0 else whead_arr[omega_index]
    return v_scalar, jnp.asarray(w_val * inv_V, dtype=dtype)


def apply_q0_head_rank1(
    V_qmunu: jnp.ndarray,
    W_qmunu: jnp.ndarray | None,
    G0_mu_nu: jnp.ndarray,
    vhead: complex | float | None,
    whead: jnp.ndarray | complex | float | None,
    cell_volume: float,
    *,
    omega_index: int = 0,
):
    """Inject the q=0 Coulomb head as a rank-1 update in the centroid basis; see docs/architecture/four_current_wiring.md."""
    g0g0 = jnp.einsum('m,n->mn', jnp.conj(G0_mu_nu), G0_mu_nu)
    v_scalar, w_scalar = _head_rank1_scalars(
        vhead, whead, cell_volume, omega_index,
        dtype=(W_qmunu.dtype if W_qmunu is not None else V_qmunu.dtype))

    if v_scalar is not None:
        # V layout: (..., nkx, nky, nkz, n_μ, n_ν); q=0 is index 0 on each k axis.
        V_qmunu = V_qmunu.at[..., 0, 0, 0, :, :].add(v_scalar * g0g0)
    if W_qmunu is not None and w_scalar is not None:
        W_qmunu = W_qmunu.at[..., 0, 0, 0, :, :].add(w_scalar * g0g0)
    return V_qmunu, W_qmunu


def apply_q0_head_rank1_sharded(
    V_q0: jnp.ndarray,
    W_q: jnp.ndarray | None,
    g0_X: jnp.ndarray,
    g0_Y: jnp.ndarray,
    vhead: complex | float | None,
    whead: jnp.ndarray | complex | float | None,
    cell_volume: float,
    *,
    omega_index: int = 0,
):
    """Sharded q=0 head injection — local on every proc; see docs/architecture/four_current_wiring.md."""
    g0g0 = jnp.conj(g0_X)[:, None] * g0_Y[None, :]
    v_scalar, w_scalar = _head_rank1_scalars(
        vhead, whead, cell_volume, omega_index,
        dtype=(W_q.dtype if W_q is not None else V_q0.dtype))

    if v_scalar is not None:
        V_q0 = V_q0 + v_scalar * g0g0
    if W_q is not None and w_scalar is not None:
        # W_q layout: (n_μ, n_ν, nkx, nky, nkz). Add to the q=0 slice only.
        W_q = W_q.at[:, :, 0, 0, 0].add(w_scalar * g0g0)
    return V_q0, W_q
