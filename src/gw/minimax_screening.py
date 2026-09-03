"""Minimax-window helpers for static chi0/W and GN-PPM extraction.

This module is intentionally scoped to the static path first:
- Build a single non-crossing minimax window pair compatible with ``w_isdf.compute_chi0``.
- Reuse existing sharded kernels (no duplicate FFT kernels here).
- Provide Godby-Needs PPM parameter extraction from precomputed chi matrices.

WHAT THIS MODULE IS AFTER THE 2026-08-08 EXTRACTION.  It is the RUNTIME
layer over the ``minimax`` service, and the boundary is the one
``docs/architecture/layers.md`` recorded years ago: the solvers, the
shipped catalog and the disk cache are ``services/minimax/`` now; what
stays here is everything that knows about physics.  Concretely:

* the three ``solve_*`` wrappers rescale the service's SCALED tables
  (``[1, R]`` for the Laplace families, ``[0, A]`` for the crossing one)
  into Rydberg and name *windows*.  For a Laplace family both the achieved
  error and the requested physical error scale with ``1/x_min``; keeping
  that units conversion here makes this the single physical-window owner;
* ``MinimaxNodes`` and the complex128 / ``time_axis`` convention stay,
  because they are a jax pytree in this package's sign convention and
  keeping them out of the service is what keeps the service jax-free;
* the χ₀/Σ quadrature builders, ``refit_imag_alpha_augmented``,
  ``_lawson_weights_fit``, ``resolve_minimax_energy_reference`` and the
  GN-PPM fit all stay.

Every quadrature the service serves now carries a ``provenance`` string
saying which artifact answered, and the driver prints below say it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils as _mh
import numpy as np

# Path bootstrap; dies with the owner's workspace fix -- see _services.py.
from ffi import _services      # noqa: F401
from runtime.env_flags import env_bool

_services.ensure_on_path()

import minimax as _mm                                        # noqa: E402
from .minimax_config import MinimaxConfig                     # noqa: E402

#: The ``complex_laplace`` family's selection rule, off the DOOR rather
#: than by `from minimax import beta_selector` -- which is a submodule
#: reach and which `test_layering` refuses, correctly.  The service
#: publishes this module as part of its public surface for exactly this
#: consumer; binding it here keeps the call sites short without spending
#: the door rule to do it.
_beta_selector = _mm.beta_selector


_TINY = 1.0e-12

#: User-ruled GN-PPM variant: coarsen at most the lowest and highest 0.2%
#: of successfully fitted logical matrix elements.  This is a fixed physics
#: policy, not an environment/runtime choice, so its one owner is beside the
#: GN fit.  It is deliberately not strict BGW finite-positive-pole parity;
#: exact scalar panes still plan the reduced support downstream.  The integer
#: form makes both tail budgets exact: ``floor(n_valid / 500)``.
GN_PPM_EXTREME_TAIL_DIVISOR = 500

#: DEBUG A/B switch for the measured-broken-TR GN fit.  This is deliberately
#: read at the fit owner: ``ordered_orientations`` is the measured TRS verdict
#: at this layer, and nowhere downstream can distinguish a TRS deck from a
#: broken-TR deck whose odd residue happens to be zero.
_DEBUG_GN_ODD_RESIDUE_OFF_ENV = "LORRAX_DEBUG_GN_ODD_RESIDUE_OFF"


def _scaled_laplace_error_bound(x_min: float, target_error: float) -> float:
    """Convert a physical Laplace-kernel tolerance to the ``[1, R]`` units.

    With ``y = x / x_min``, both supported targets have the form

    ``K(x) = f(y) / x_min``.

    Therefore a scaled rule with error ``eps_hat`` has physical error
    ``eps_hat / x_min``.  To honor a caller's physical absolute tolerance
    ``eps_phys``, the minimax service must be asked for
    ``eps_hat <= eps_phys * x_min``.  The reverse conversion is already used
    when the achieved error is stamped on ``LaplaceMinimaxQuadrature``.

    Reject invalid tolerances rather than clipping them: a floor here would
    silently loosen the physical contract for small gaps.
    """

    x_min = float(x_min)
    target_error = float(target_error)
    if not np.isfinite(x_min) or x_min <= 0.0:
        raise ValueError(
            f"x_min must be finite and positive before minimax rescaling; "
            f"got {x_min!r}.")
    if not np.isfinite(target_error) or target_error <= 0.0:
        raise ValueError(
            f"target_error must be a finite positive physical absolute "
            f"tolerance; got {target_error!r}.")
    scaled = target_error * x_min
    if not np.isfinite(scaled) or scaled <= 0.0:
        raise ValueError(
            f"Scaled minimax tolerance is not representable: "
            f"{target_error!r} * {x_min!r} = {scaled!r}.")
    return scaled


def _scalar_to_host_float(a) -> float:
    """Fetch a scalar JAX value in a multihost-safe way."""

    if jax.process_count() > 1:
        arr = jnp.asarray(a)
        # ``process_allgather`` rejects ``tiled=False`` for globally-sharded
        # (non-fully-addressable) inputs:
        #   ValueError: Gathering global non-fully-addressable arrays only
        #               supports tiled=True
        # which aborted every multi-host GN-PPM run in fit_gn_ppm_from_wc_pair.
        # Dispatch is CONDITION-TESTED (same explicit ``is_fully_addressable``
        # branch as ``ppm_windows._to_host_np``), not exception-swallowed: a
        # try/except around the collective converted ANY allgather failure
        # (Gloo/NCCL error, partial-rank abort) into a silent device_get
        # fallback that could return on some ranks while others raised —
        # a rank-desync hazard masking real collective failures (audit
        # fix/zq 2026-07-28).
        # Tiled gathering needs >= 1-D, so promote the scalar first.
        # ``reshape((-1,))[:1]`` rather than ``reshape((1,))``: the latter
        # would raise for any input with size > 1, whereas the historical
        # tiled=False path flattened after gathering and took element 0.
        # Keep that tolerance so this stays a strict bug-fix.
        if not getattr(arr, "is_fully_addressable", True):
            gathered = _mh.process_allgather(arr.reshape((-1,))[:1], tiled=True)
        else:
            gathered = jax.device_get(arr)
        return float(np.asarray(gathered, dtype=np.float64).reshape(-1)[0])
    return float(np.asarray(jax.device_get(a), dtype=np.float64))


# ---------------------------------------------------------------------------
#  THE CATALOG, THE DISK CACHE AND THE SOLVERS LEFT THIS MODULE (2026-08-08).
#
#  They are `services/minimax/` now, reached through the door as `_mm`.  What
#  used to sit here was ~190 lines carrying six `except Exception:` handlers
#  that turned every lookup failure -- a missing catalog, an unreadable
#  table, a malformed entry, a failed cache write -- into the same silent
#  `None`, which the code below then read as "no table, solve it yourself"
#  and spent minutes doing, uncertified, with nothing in the log to say so.
#  The service replaces each of those with a value the caller can see, and
#  every rule it serves now announces where it came from.
#
#  WHAT STAYED HERE, and why: the RESCALE.  The door serves tables in the
#  scaled units the catalog tabulates (`[1, R]` for the Laplace families,
#  `[0, A]` for the crossing one); the three wrappers below divide by
#  `x_min` and name windows in Rydberg.  Keeping that split is what makes
#  the extraction bit-identical by construction rather than by tolerance --
#  the bytes are `np.load`'s, and the arithmetic that follows them is the
#  same arithmetic in the same module.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinimaxNodes:
    """τ nodes + weights in complex form, passable to jit as a pytree.

    Both chi0's Laplace quad (real τ → Im(t)=0) and sigma's crossing /
    non-crossing quads (``-1j·τ`` or ``τ/ξ``) live in the same complex128
    storage so one sibling function shape (``minimax_tau_integrate_*``)
    handles both pipelines.
    """

    t: jax.Array       # complex128, shape (n,)
    alpha: jax.Array   # complex128, shape (n,)


jax.tree_util.register_dataclass(
    MinimaxNodes, data_fields=['t', 'alpha'], meta_fields=[])


def _laplace_to_minimax_nodes(
    tau: np.ndarray, alpha: np.ndarray, *, time_axis: str,
) -> MinimaxNodes:
    """Convert a (real τ, real α) Laplace quadrature into complex ``MinimaxNodes``.

    ``time_axis``:
      * ``'real'``      — chi0 Laplace: ``t = τ + 0j``, α cast to complex.
                          exp(-t·ΔE) stays real-valued for real ΔE.
      * ``'imag'``      — sigma Laplace windows (single/a_stripe/b_slab):
                          ``t = -1j·τ``, α cast to complex.
    """
    tau_j = jnp.asarray(np.asarray(tau, dtype=np.float64), dtype=jnp.float64)
    alpha_j = jnp.asarray(np.asarray(alpha, dtype=np.float64), dtype=jnp.float64)
    if time_axis == 'real':
        t = tau_j.astype(jnp.complex128)
    elif time_axis == 'imag':
        t = (-1j) * tau_j.astype(jnp.complex128)
    else:
        raise ValueError(
            f"Unknown time_axis={time_axis!r}; expected 'real' or 'imag'.")
    return MinimaxNodes(t=t, alpha=alpha_j.astype(jnp.complex128))


def _crossing_to_minimax_nodes(
    tau: np.ndarray, alpha: np.ndarray, *, time_axis: str,
) -> MinimaxNodes:
    """Convert a crossing quadrature into complex ``MinimaxNodes``.

    ``time_axis='crossing_hgl'`` keeps τ real (cast to complex) — the
    crossing window integrates ``Im[...]`` on the real-τ axis directly.
    Callers that need to rescale by 1/ξ apply that externally.
    """
    if time_axis != 'crossing_hgl':
        raise ValueError(
            f"Unknown time_axis={time_axis!r} for crossing quadrature; "
            f"expected 'crossing_hgl'.")
    tau_j = jnp.asarray(np.asarray(tau, dtype=np.float64), dtype=jnp.float64)
    alpha_j = jnp.asarray(np.asarray(alpha, dtype=np.float64), dtype=jnp.float64)
    return MinimaxNodes(
        t=tau_j.astype(jnp.complex128),
        alpha=alpha_j.astype(jnp.complex128),
    )


@dataclass(frozen=True)
class LaplaceMinimaxQuadrature:
    """Quadrature summary for ``1/x`` on ``[x_min, x_max]``."""

    x_min: float
    x_max: float
    tau: np.ndarray
    alpha: np.ndarray
    max_error: float
    #: WHERE THIS RULE CAME FROM, as ``minimax.Provenance.one_line()``
    #: rendered it: the shipped payload path and its hash, or the cache
    #: entry, or "runtime solve, no artifact".  Trailing and defaulted so
    #: every existing construction site is untouched; ``None`` means the
    #: object was built by hand (a test fixture) rather than served.
    provenance: str | None = None
    #: THE ODD KERNEL (ordered particle-hole orientations, magnets only).
    #: ``alpha_odd`` fits ``omega_p/(x^2+omega_p^2)`` on the SAME ``tau``
    #: nodes -- the imaginary part of ``-1/(x+i*omega_p)`` whose real part
    #: ``alpha`` fits.  Present only when the rule was built with
    #: ``with_odd_kernel=True``; the last ``n_odd_extra`` nodes of ``tau``
    #: then carry ``alpha == 0`` (they exist for the odd part alone, so the
    #: even accumulation is numerically the served even rule).  See
    #: ``docs/dev/notes/DERIVATION_gnppm_nonhermitian.md`` section 2.
    alpha_odd: np.ndarray | None = None
    max_error_odd: float = float("nan")
    n_odd_extra: int = 0

    @property
    def node_count(self) -> int:
        return int(self.tau.shape[0])

    def to_minimax_nodes(self, *, time_axis: str) -> MinimaxNodes:
        """Return ``MinimaxNodes`` in the caller's sign convention.

        See ``_laplace_to_minimax_nodes`` for the set of accepted
        ``time_axis`` values.  The returned pytree is safe to close over
        in a jit or pass as an argument.
        """
        return _laplace_to_minimax_nodes(
            self.tau, self.alpha, time_axis=time_axis)


@dataclass(frozen=True)
class CrossingMinimaxQuadrature:
    """Quadrature summary for crossing regularization target on ``[0, A_dim]``."""

    A_dim: float
    tau: np.ndarray
    alpha: np.ndarray
    max_error: float
    target_kind: str
    #: See :class:`LaplaceMinimaxQuadrature.provenance`.
    provenance: str | None = None

    @property
    def node_count(self) -> int:
        return int(self.tau.shape[0])

    def to_minimax_nodes(self, *, time_axis: str = 'crossing_hgl') -> MinimaxNodes:
        """Return ``MinimaxNodes`` for the crossing-window τ axis."""
        return _crossing_to_minimax_nodes(
            self.tau, self.alpha, time_axis=time_axis)


@dataclass(frozen=True)
class GNPPMFitResult:
    """Fitted pole tensors and their scalar conditioning/cost census."""

    omega_qmunu: jax.Array
    B_qmunu: jax.Array
    valid_qmunu: jax.Array
    unfulfilled_fraction: float
    n_valid: int
    omega_min_raw: float
    omega_max_raw: float
    pair_relative_separation_min: float
    n_tail_low: int = 0
    n_tail_high: int = 0
    omega_min_after: float = float("nan")
    omega_max_after: float = float("nan")
    tail_anchor_omega: float = float("nan")
    #: The ODD residue ``D = (R_+ - R_-)/2`` of the ordered-orientation fit
    #: (``docs/dev/notes/DERIVATION_gnppm_nonhermitian.md`` section 3),
    #: Hermitian elementwise, zero on dead modes.  ``None`` is the incumbent
    #: single-residue model (``R_+ = R_- = B_qmunu``), the only model a
    #: time-reversal-symmetric deck ever sees.
    B_odd_qmunu: jax.Array | None = None


def fit_gn_ppm_from_wc_pair(
    Wc0_qmunu: jax.Array,
    Wc_probe_qmunu: jax.Array,
    probe_omega: complex,
    *,
    fallback_omega: float,
    n_mu_logical: int,
    q_neg_index: np.ndarray | None = None,
    coarsen_extreme_tails: bool = False,
    ordered_orientations: bool = False,
    print_fn=print,
) -> GNPPMFitResult:
    """Fit GN-PPM pole data elementwise on an already-sharded ``(q,mu,nu)`` tensor pair.

    Parameters
    ----------
    Wc0_qmunu
        ``W^c(0)`` in shape ``(nq,n_rmu,n_rmu)``.
    Wc_probe_qmunu
        ``W^c(z_probe)`` in the same shape/sharding as ``Wc0_qmunu``.
    probe_omega
        Complex probe frequency ``z_probe`` in Ry. For the standard GN fit this is
        purely imaginary, e.g. ``2j``.
    ordered_orientations
        THE MAGNET MODEL.  ``False`` (default) is the incumbent single-residue
        fit, bit-for-bit: ``W^c(z_probe)`` is consumed as given and both pole
        branches share ``B``.  ``True`` splits ``W^c(i*omega_p)`` elementwise
        into its Hermitian half ``h`` (which fixes ``Omega`` and ``B`` through
        the incumbent algebra) and its anti-Hermitian half ``a``, and returns
        the odd residue ``D = i*a*(omega_p^2 + Omega^2)/(2*omega_p)`` in
        ``B_odd_qmunu`` so that the conduction branch of Sigma consumes
        ``R_+ = B + D`` and the valence branch ``R_- = B - D``
        (``docs/dev/notes/DERIVATION_gnppm_nonhermitian.md`` sections 3-4).
        Requires a purely imaginary probe: at a real probe ``W^c`` is Hermitian
        for any system and the odd residue is unobservable.  On a Hermitian
        input ``D`` is exactly zero and ``Omega``/``B`` agree with the incumbent
        to roundoff (a different XLA program: not bit-identical, which is why
        the caller routes only measured-broken-TR decks here).
    fallback_omega
        Positive real fallback pole in Ry for entries that do not produce a valid
        positive-real ``Omega^2`` estimate.
    n_mu_logical
        Logical centroid count (``meta.n_rmu``).  REQUIRED — the trailing
        (μ, ν) axes may carry the padded extent, and pad modes must be born
        DEAD here: ``Ω = 0``, ``B = 0``, ``valid = False``.  Handing them the
        live-looking fallback Ω instead used to inflate the mode census and
        the masked-Ω window statistics by a pad-extent- (= device-count-)
        dependent amount (ROOT_CAUSE.md 2026-07-08).  Zeroing Ω at birth
        makes every present and future ``Omega_q``/``B_q`` consumer
        structurally pad-safe: the ``Ω > 1e-14`` mode mask excludes pads with
        no mask argument anywhere downstream.  Pass the padded extent
        (all-true mask) when the inputs are unpadded.
    q_neg_index
        Canonical full-grid ``q -> -q`` row permutation from the public
        ``symmetry_maps.q_negation_index`` service.  Required only when
        ``coarsen_extreme_tails`` is true; the GN policy uses it together with
        ``mu <-> nu`` to keep the physical four-lane partner orbit atomic and
        never reconstructs the q convention locally.
    coarsen_extreme_tails
        Apply the user-ruled GN policy to successfully fitted logical modes:
        replace at most the lowest and highest 0.2% of pole frequencies by
        ``fallback_omega`` and recompute ``B = -Wc(0) Omega / 2``.  The
        initial order-statistic boundary is exact-key-group atomic; physical
        partner-orbit closure can further undershoot the budget and can
        distinguish unrelated equal-key lanes.  The fitted-valid mask is
        unchanged.
        This changes the affected finite poles and their ``1/z^2`` moments,
        so it is not strict BGW pole parity.  False is required for HL and
        for low-level parity gates.

    Returns
    -------
    GNPPMFitResult
        Elementwise GN-PPM parameters in the same ``(nq,n_rmu,n_rmu)``
        layout; ``unfulfilled_fraction`` and the census count LOGICAL modes
        only.  The extrema are over valid logical modes; the relative
        separation is ``|Wc(0)-Wc(z)| / max(|Wc(0)|, |Wc(z)|)``.  The fit is
        pure local algebra: no tensor host gathers and no communication
        beyond scalar reductions on the inputs' existing sharding.  When
        tail coarsening is enabled, its exact distributed order statistics
        likewise use scalar reductions only.
    """

    n_mu = int(jnp.asarray(Wc0_qmunu).shape[-1])
    n_log = int(n_mu_logical)
    if not (0 < n_log <= n_mu):
        raise ValueError(
            f"fit_gn_ppm_from_wc_pair: n_mu_logical={n_log} outside "
            f"(0, {n_mu}] for input extent {n_mu}.")

    fallback_host = float(fallback_omega)
    if not np.isfinite(fallback_host) or fallback_host <= 0.0:
        raise ValueError(
            "fit_gn_ppm_from_wc_pair: fallback_omega must be finite and "
            f"positive; got {fallback_omega!r}.")
    _z = jnp.asarray(probe_omega, dtype=jnp.complex128)
    _fb = jnp.asarray(fallback_host, dtype=jnp.float64)
    _W0 = jnp.asarray(Wc0_qmunu)
    ordered = bool(ordered_orientations)
    debug_odd_off = env_bool(
        _DEBUG_GN_ODD_RESIDUE_OFF_ENV, False, print_fn=print_fn)
    if debug_odd_off and not ordered:
        raise ValueError(
            "GATE debug_gn_odd_residue_off_scope:\n"
            f"  got:  {_DEBUG_GN_ODD_RESIDUE_OFF_ENV}=1 with "
            "ordered_orientations=False\n"
            "  want: this debug switch only on a measured-broken-TR "
            "GN-PPM fit (ordered_orientations=True)\n"
            "  why:  a TRS deck has no time-reversal-odd GN residue to "
            "discard\n"
            "  fix:   unset LORRAX_DEBUG_GN_ODD_RESIDUE_OFF")
    if debug_odd_off:
        print_fn(
            "WARNING -- DEBUG: LORRAX_DEBUG_GN_ODD_RESIDUE_OFF=1; "
            "measured-broken-TR GN-PPM fit is discarding the "
            "anti-Hermitian half of W(i omega_p): D=0 and R+=R-=B. "
            "This arm is for A/B diagnosis only, never production.")
    if ordered:
        _zh = complex(probe_omega)
        if (not np.isfinite(_zh.real) or not np.isfinite(_zh.imag)
                or abs(_zh.real) > 0.0 or abs(_zh.imag) == 0.0):
            raise ValueError(
                "GATE gn_ppm_ordered_probe_axis: the ordered probe is not "
                "on the nonzero imaginary axis.\n"
                f"  got:  probe_omega = {probe_omega!r}\n"
                "  want: Re(probe_omega) = 0 and Im(probe_omega) != 0\n"
                "  why:  the odd residue is read from the anti-Hermitian "
                "part of Wc(i*omega_p), which a real or zero probe cannot "
                "carry\n"
                "  doc:  docs/dev/notes/DERIVATION_gnppm_nonhermitian.md")

    # --- q-CHUNKED EVALUATION (movement-only; see the note above the kernel).
    # Leading axis is the q family; the trailing two are (mu, nu).  One q-slice
    # of the LOCAL (already-sharded) tile is what sizes the arena.
    _nq = int(_W0.shape[0])
    _qneg = None
    if coarsen_extreme_tails:
        if q_neg_index is None:
            raise ValueError(
                "fit_gn_ppm_from_wc_pair: q_neg_index is required when "
                "coarsen_extreme_tails=True.")
        _qneg_raw = np.asarray(q_neg_index)
        if (_qneg_raw.shape != (_nq,)
                or not np.all(np.isfinite(_qneg_raw))
                or not np.array_equal(_qneg_raw, np.rint(_qneg_raw))):
            raise ValueError(
                "fit_gn_ppm_from_wc_pair: q_neg_index must be one finite "
                f"integer row per q; got shape={_qneg_raw.shape} for nq={_nq}.")
        _qneg = _qneg_raw.astype(np.int32)
        if (np.any(_qneg < 0) or np.any(_qneg >= _nq)
                or not np.array_equal(_qneg[_qneg], np.arange(_nq))):
            raise ValueError(
                "fit_gn_ppm_from_wc_pair: q_neg_index must be an involution "
                f"over [0,{_nq}).")
    _per_q = 1
    for _d in _W0.shape[1:]:
        _per_q *= int(_d)
    _per_q *= _W0.dtype.itemsize
    _qb = _gn_ppm_fit_q_block(_nq, _per_q)

    # The anti-Hermitian half of the probe, kept only on the ordered path
    # (one extra (nq, mu, nu) c128 tile, needed again after the tail policy
    # has fixed the final Omega).
    a_odd = None
    if _qb >= _nq:
        if ordered:
            (omega_vals, B_vals, good, n_good, n_modes,
             omega_min, omega_max, pair_rel_min,
             a_odd) = _gn_ppm_fit_kernel_ordered(
                Wc0_qmunu, Wc_probe_qmunu, _z, _fb, n_log)
        else:
            # Whole thing fits: the historical single-shot call, untouched.
            (omega_vals, B_vals, good, n_good, n_modes,
             omega_min, omega_max, pair_rel_min) = _gn_ppm_fit_kernel(
                Wc0_qmunu, Wc_probe_qmunu, _z, _fb, n_log)
    else:
        _om, _bv, _gd, _aod = [], [], [], []
        n_good = jnp.asarray(0.0, dtype=jnp.float64)
        n_modes = jnp.asarray(0.0, dtype=jnp.float64)
        omega_min = jnp.asarray(jnp.inf, dtype=jnp.float64)
        omega_max = jnp.asarray(-jnp.inf, dtype=jnp.float64)
        pair_rel_min = jnp.asarray(jnp.inf, dtype=jnp.float64)
        for _q0 in range(0, _nq, _qb):
            _q1 = min(_q0 + _qb, _nq)
            if ordered:
                (_o, _b, _g, _ng, _nm,
                 _omin, _omax, _rmin, _a) = _gn_ppm_fit_kernel_ordered(
                     Wc0_qmunu[_q0:_q1], Wc_probe_qmunu[_q0:_q1],
                     _z, _fb, n_log)
                _aod.append(_a)
            else:
                (_o, _b, _g, _ng, _nm,
                 _omin, _omax, _rmin) = _gn_ppm_fit_kernel(
                     Wc0_qmunu[_q0:_q1], Wc_probe_qmunu[_q0:_q1],
                     _z, _fb, n_log)
            _om.append(_o); _bv.append(_b); _gd.append(_g)
            # Exact integer counts -> summation order is irrelevant.
            n_good = n_good + _ng
            n_modes = n_modes + _nm
            omega_min = jnp.minimum(omega_min, _omin)
            omega_max = jnp.maximum(omega_max, _omax)
            pair_rel_min = jnp.minimum(pair_rel_min, _rmin)
        omega_vals = jnp.concatenate(_om, axis=0)
        B_vals = jnp.concatenate(_bv, axis=0)
        good = jnp.concatenate(_gd, axis=0)
        if ordered:
            a_odd = jnp.concatenate(_aod, axis=0)
        del _om, _bv, _gd, _aod

    fulfilled = n_good / jnp.maximum(n_modes, 1.0)
    # Every host transfer in the fit is a scalar and deliberately outside
    # the kernel: ``_scalar_to_host_float`` gathers, which cannot happen
    # under ``jit``.  The fitted tensors never leave their input sharding.
    n_valid = int(_scalar_to_host_float(n_good))
    unfulfilled = 1.0 - _scalar_to_host_float(fulfilled)
    if n_valid:
        omega_min_host = _scalar_to_host_float(omega_min)
        omega_max_host = _scalar_to_host_float(omega_max)
        pair_rel_min_host = _scalar_to_host_float(pair_rel_min)
    else:
        omega_min_host = omega_max_host = pair_rel_min_host = float("nan")

    n_tail_low = n_tail_high = 0
    omega_min_after = omega_min_host
    omega_max_after = omega_max_host
    tail_anchor = float("nan")
    if coarsen_extreme_tails:
        (omega_vals, B_vals, n_low_j, n_high_j,
         omega_min_after_j, omega_max_after_j,
         tail_anchor_j) = _coarsen_gn_ppm_extreme_tails(
             omega_vals, B_vals, good, Wc0_qmunu, _qneg, _fb,
             tail_divisor=GN_PPM_EXTREME_TAIL_DIVISOR)
        n_tail_low = int(_scalar_to_host_float(n_low_j))
        n_tail_high = int(_scalar_to_host_float(n_high_j))
        if n_valid:
            omega_min_after = _scalar_to_host_float(omega_min_after_j)
            omega_max_after = _scalar_to_host_float(omega_max_after_j)
        tail_anchor = _scalar_to_host_float(tail_anchor_j)
        budget = n_valid // GN_PPM_EXTREME_TAIL_DIVISOR
        if n_tail_low > budget or n_tail_high > budget:
            raise RuntimeError(
                "GATE gn_ppm_extreme_tail_budget: "
                f"gn_ppm_extreme_tail_counts got: low={n_tail_low}, "
                f"high={n_tail_high}; want: each count <= budget={budget} "
                f"({1.0 / GN_PPM_EXTREME_TAIL_DIVISOR:.3%} of valid modes); "
                "why: tail coarsening may alter only its bounded extreme-mode "
                "allocation.")
        if n_valid and (
                omega_min_after < omega_min_host
                or omega_max_after > omega_max_host):
            raise RuntimeError(
                "GATE gn_ppm_extreme_tail_range: omega_support got: "
                f"before=[{omega_min_host}, {omega_max_host}], "
                f"after=[{omega_min_after}, {omega_max_after}]; want: "
                "after support contained in before support; why: tail "
                "coarsening may contract but must never enlarge fitted pole "
                "support.")

    # THE ODD RESIDUE, from the FINAL pole.  ``D = i a (omega_p^2 +
    # Omega^2) / (2 omega_p)`` is a statement about the pole the Sigma
    # kernel will actually evolve, so it is formed after the optional tail
    # policy has re-anchored Omega -- the same order in which ``B`` was
    # recomputed from the anchored pole.  Dead modes (Omega = 0) carry
    # D = 0 exactly, which keeps every downstream ``Omega > 1e-14`` mask
    # sufficient for the odd residue too.
    B_odd_vals = None
    if ordered:
        B_odd_vals = (
            jnp.zeros_like(B_vals) if debug_odd_off else
            _gn_ppm_odd_residue(a_odd, omega_vals, _z))
        del a_odd

    return GNPPMFitResult(
        omega_qmunu=omega_vals,
        B_qmunu=B_vals,
        valid_qmunu=good,
        unfulfilled_fraction=unfulfilled,
        n_valid=n_valid,
        omega_min_raw=omega_min_host,
        omega_max_raw=omega_max_host,
        pair_relative_separation_min=pair_rel_min_host,
        n_tail_low=n_tail_low,
        n_tail_high=n_tail_high,
        omega_min_after=omega_min_after,
        omega_max_after=omega_max_after,
        tail_anchor_omega=tail_anchor,
        B_odd_qmunu=B_odd_vals,
    )


# ---------------------------------------------------------------------------
# GN-PPM fit q-chunking (size campaign 2026-07-29, ladder notes R32/R33)
#
# MEASURED DEFECT.  At MoS2 4x4 / mu_pad = 24,960 / P = 64 the rank-0 HLO of
# the run that OOMed (job 7879469, module_0914.jit__gn_ppm_fit_kernel) reports
#     allocation 35: size 74.27GiB, preallocated-temp
# against parameters/outputs of only 8.27 GiB total, all correctly sharded
# c128[16,3120,3120] tiles (3120 = mu_pad/p_x).
#
# > CLAIM-DECAY (R37, 2026-07-29).  The original reading of that line -- "32
# > live temporaries because XLA fused only 6 of ~111 full-tile instructions"
# > -- is WRONG, and so is the conclusion that the guard chain does not fuse.
# > XLA:CPU fuses the chain completely: at the reference shape the entry
# > computation contains SEVEN full-tile instructions (2 parameters + 5 kLoop
# > fusions) and ZERO unfused full-tile elementwise ops; the "111" counted
# > instructions INSIDE %fused_computation bodies, which never materialise.
# > The 74.27 GiB was ONE buffer -- the replicated global mode-count mask, see
# > the n_modes note in the kernel -- and it is now gone.  The "32.0x one tile"
# > multiple was the mesh identity p_x^2/2, not a temporary count.
#
# THE FIX IS MOVEMENT-ONLY.  Every operation in the kernel is elementwise in
# the leading (q) axis and the two reductions are exact integer counts, so
# evaluating q in blocks changes evaluation ORDER and PLACEMENT only.  The
# arena falls as 1/n_blocks while the (already-sharded) inputs and outputs stay
# resident.  Nothing about the arithmetic, the guards, or the fitted values
# changes -- and the guards are deliberately NOT touched (see R33 note: a
# finiteness/branch guard that costs scratch is an owner question, not
# something to quietly restructure).
_GN_PPM_FIT_ARENA_BUDGET_BYTES = int(
    float(os.environ.get("LORRAX_PPM_FIT_ARENA_GIB", "8")) * 1024 ** 3)
#: Live-footprint multiple of one (q-block, mu, nu) c128 tile.
#: DELIBERATELY CONSERVATIVE.  Once the replicated mode-count mask is gone
#: (R37) the measured multiple is ~4 (params + outputs + a half-tile temp), so
#: 32 over-estimates by ~8x and therefore over-chunks.  Campaign doctrine (R5,
#: R30.3) is that a sizer which reads HIGH is safe and one that reads LOW is
#: not, so the value is left high on purpose; lowering it is a measured,
#: separately-gated change, not a comment edit.  It does not affect the
#: reference deck, which takes the single-shot path either way.
_GN_PPM_FIT_LIVE_TILES = 32


def _gn_ppm_fit_q_block(nq: int, tile_bytes_per_q: int) -> int:
    """Largest q-block whose temp arena fits the budget.  Floor 1, cap nq.

    ``tile_bytes_per_q`` is ONE q-slice of the local (already-sharded) tile.
    Returns ``nq`` (the historical single-shot path, bit-identical) whenever
    the whole thing already fits.
    """
    per_q = max(1, int(tile_bytes_per_q) * _GN_PPM_FIT_LIVE_TILES)
    return max(1, min(int(nq), _GN_PPM_FIT_ARENA_BUDGET_BYTES // per_q))


@partial(
    jax.jit,
    static_argnames=("tail_divisor",),
    donate_argnums=(0, 1),
)
def _coarsen_gn_ppm_extreme_tails(
    omega_qmunu,
    B_qmunu,
    valid_qmunu,
    Wc0_qmunu,
    q_neg_index,
    fallback_omega,
    *,
    tail_divisor: int,
):
    """Apply the lossy user-ruled GN tail policy without a tensor gather.

    Positive IEEE-754 float64 values have monotonically ordered signed-int64
    bit patterns.  Two simultaneous 63-step lower-bound searches therefore
    recover the exact lower/upper order-statistic boundaries using only
    scalar reductions on the input sharding.  No pole-sized host value or
    index array is made.  Each exact per-lane candidate mask is then reduced
    to its group-closure interior over the physical orbit
    ``(q,mu,nu)``, ``(q,nu,mu)``, ``(-q,mu,nu)``, ``(-q,nu,mu)`` (with the
    natural collapses on diagonals and self-negative q).  An orbit is changed
    only if every member was already inside the same tail candidate; a
    one-ulp boundary split therefore retains the whole orbit.  This can only
    undershoot the per-tail budget, never exceed it, and all selected lanes
    share the one replacement anchor below.

    The replacement frequency is the configured fallback clipped into the
    retained central support.  Thus the operation can never enlarge the
    fitted Ω range.  Its residue is recomputed from the GN static identity

        Wc(0) = -2 B / Ω,  hence  B' = -Wc(0) Ω' / 2.

    It cannot also preserve the leading high-frequency moment ``2 B Ω``:
    imposing both identities gives ``Ω'^2 = Ω^2`` for nonzero Wc(0), so any
    nontrivial real-positive coarsening must choose one.  Static W is the
    fitted observable and is preserved exactly to the arithmetic precision
    of the existing fit; the high-frequency moment and strict BGW finite-pole
    parity are not.  The canonical exact-pane planner remains downstream and
    partitions this reduced support; this policy does not replace it.
    """
    if tail_divisor < 2:
        raise ValueError("tail_divisor must be an integer >= 2")

    omega = jnp.asarray(omega_qmunu, dtype=jnp.float64)
    B = jnp.asarray(B_qmunu, dtype=jnp.complex128)
    valid = jnp.asarray(valid_qmunu, dtype=bool)
    Wc0 = jnp.asarray(Wc0_qmunu, dtype=jnp.complex128)
    q_neg = jnp.asarray(q_neg_index, dtype=jnp.int32)
    fallback = jnp.asarray(fallback_omega, dtype=jnp.float64)
    if omega.ndim != 3 or omega.shape[-2] != omega.shape[-1]:
        raise ValueError(
            "GN tail policy requires flat-q square matrix tiles; got "
            f"shape={omega.shape}.")
    if q_neg.shape != (omega.shape[0],):
        raise ValueError(
            "GN tail policy q_neg_index extent must equal flat q extent; "
            f"got {q_neg.shape} for shape={omega.shape}.")

    # The valid-fit predicate already proves these are finite and positive.
    # Re-state it here so this owner remains fail-closed if called directly.
    eligible = valid & jnp.isfinite(omega) & (omega > 0.0)
    keys = jax.lax.bitcast_convert_type(omega, jnp.int64)
    n_valid = jnp.sum(eligible, dtype=jnp.int64)
    budget = n_valid // jnp.asarray(tail_divisor, dtype=jnp.int64)

    min_key = jnp.min(jnp.where(eligible, keys, jnp.iinfo(jnp.int64).max))
    max_key = jnp.max(jnp.where(eligible, keys, jnp.asarray(0, jnp.int64)))
    min_key = jnp.where(n_valid > 0, min_key, jnp.asarray(0, jnp.int64))
    max_key = jnp.where(n_valid > 0, max_key, jnp.asarray(0, jnp.int64))

    # 1-indexed ascending ranks.  Clamp the zero-budget/empty cases to rank 1;
    # ``budget > 0`` below keeps their masks empty.
    targets = jnp.stack((
        jnp.maximum(budget, jnp.asarray(1, jnp.int64)),
        jnp.maximum(n_valid - budget + 1, jnp.asarray(1, jnp.int64)),
    ))
    lo0 = jnp.full((2,), min_key, dtype=jnp.int64)
    hi0 = jnp.full((2,), max_key, dtype=jnp.int64)

    def _bisect(_iteration, bounds):
        lo, hi = bounds
        mid = lo + (hi - lo) // 2
        count0 = jnp.sum(eligible & (keys <= mid[0]), dtype=jnp.int64)
        count1 = jnp.sum(eligible & (keys <= mid[1]), dtype=jnp.int64)
        counts = jnp.stack((count0, count1))
        go_left = counts >= targets
        return (
            jnp.where(go_left, lo, mid + 1),
            jnp.where(go_left, mid, hi),
        )

    boundaries, _ = jax.lax.fori_loop(0, 63, _bisect, (lo0, hi0))
    lower_key, upper_key = boundaries[0], boundaries[1]

    n_lower_le = jnp.sum(eligible & (keys <= lower_key), dtype=jnp.int64)
    n_upper_ge = jnp.sum(eligible & (keys >= upper_key), dtype=jnp.int64)
    lower = eligible & jnp.where(
        n_lower_le <= budget, keys <= lower_key, keys < lower_key)
    upper = eligible & jnp.where(
        n_upper_ge <= budget, keys >= upper_key, keys > upper_key)

    # Imaginary-axis W is Hermitian at fixed q and independently obeys
    # W[q] = conj(W[-q]).  Keep a candidate only when its entire four-lane
    # orbit is already present.  This is an interior, never an expansion:
    # counts cannot grow and no new per-orbit ordering/weight policy appears.
    def _orbit_interior(mask):
        mask_mq = jnp.take(mask, q_neg, axis=0)
        return (
            mask & jnp.swapaxes(mask, -1, -2) & mask_mq
            & jnp.swapaxes(mask_mq, -1, -2)
        )

    lower = _orbit_interior(lower)
    upper = _orbit_interior(upper)
    has_budget = budget > 0
    lower &= has_budget
    upper &= has_budget
    tail = lower | upper

    retained = eligible & (~tail)
    retained_min = jnp.min(jnp.where(retained, omega, jnp.inf))
    retained_max = jnp.max(jnp.where(retained, omega, -jnp.inf))
    anchor = jnp.where(
        jnp.any(retained),
        jnp.clip(fallback, retained_min, retained_max),
        fallback,
    )
    omega_out = jnp.where(tail, anchor, omega)
    B_out = jnp.where(
        tail,
        -0.5 * Wc0 * anchor.astype(jnp.complex128),
        B,
    )
    omega_min_after = jnp.min(jnp.where(eligible, omega_out, jnp.inf))
    omega_max_after = jnp.max(jnp.where(eligible, omega_out, -jnp.inf))
    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    anchor = jnp.where(has_budget & (n_valid > 0), anchor, nan)
    return (
        omega_out, B_out,
        jnp.sum(lower, dtype=jnp.int64),
        jnp.sum(upper, dtype=jnp.int64),
        omega_min_after, omega_max_after,
        anchor,
    )

@partial(jax.jit, static_argnums=(4,))
def _gn_ppm_fit_kernel(Wc0_qmunu, Wc_probe_qmunu, z_probe, fallback, n_log):
    """The GN-PPM fit as ONE XLA module.  Elementwise; layout-preserving.

    Why jitted (scorecard J.3 / AD).  Run eagerly this chain materialises
    ~15 concurrent ``(nq, μ, μ)`` complex128 temporaries — ``denom``,
    ``safe``, ``ratio``, ``omega_sq``, its real part, four boolean masks,
    two ``where`` results, ``B_vals`` and the two reduction operands —
    each a separate device allocation with **zero buffer reuse**, on top
    of a resident ``V_q`` and the W pair that feed it.  At MoS₂ 12×12,
    μ_pad = 2048 that is 15 × 4.8 GB of arena the ISDF memory model does
    not know about (it stops at Stage E).  Under one jit XLA fuses the
    whole chain into a handful of loops and reuses buffers; J estimated
    ~3 live slots.

    Bit-exactness: every operation here is elementwise, so fusion cannot
    reassociate anything.  The two reductions count booleans, i.e. exact
    integers in float64.  The fitted ``Ω``/``B`` are deterministic and
    are gated bit-identical before/after.

    ``n_log`` is STATIC (the mask is a shape-dependent constant, and the
    logical extent is a host-side property of the run), so this compiles
    once per (shape, logical extent) — the same key the eager path would
    have retraced on anyway.  Module-level, so no in-body-jit recompile
    hazard (scorecard Z.1 class (a)).

    Returns the fitted arrays, exact logical counts, and the valid-mode
    pole/two-point-conditioning extrema; the caller turns the counts and
    scalar reductions into one host census.
    """
    Wc0 = jnp.asarray(Wc0_qmunu, dtype=jnp.complex128)
    Wc_probe = jnp.asarray(Wc_probe_qmunu, dtype=jnp.complex128)
    n_mu = int(Wc0.shape[-1])

    mu_log = jnp.arange(n_mu) < n_log
    mode_mask = mu_log[:, None] & mu_log[None, :]   # (μ, ν) logical selector

    denom = Wc0 - Wc_probe
    safe = jnp.abs(denom) > 1.0e-14
    # INTERMEDIATE REDUCTION 1 (2026-07-29, owner directive; ladder notes R34).
    # The old form was ``ratio = where(safe, Wc_probe/denom, 0)`` — a full-tile
    # c128 SELECT (2.32 GiB at mu_pad=24960/P=64) purely as defensive masking.
    # It is REDUNDANT, provably: ``safe`` remains ANDed into ``good`` below, and
    # the ONLY consumer of ``ratio`` is omega_sq -> omega_sq_re -> sqrt, whose
    # value is discarded by ``where(good, ...)`` on exactly the lanes where
    # ``safe`` is false.  Case check on a lane with safe == False:
    #   old: ratio=0 -> omega_sq_re=0 -> isfinite(0)=T, (0>0)=F -> good=F
    #   new: ratio=inf/nan -> omega_sq_re=inf/nan -> isfinite=F     -> good=F
    # both give good=False, and omega_vals/B_vals then take the SAME branch.
    # Guard SEMANTICS are unchanged: ``safe`` still gates ``good``.  Only the
    # materialisation of a masked copy is removed.
    ratio = Wc_probe / denom
    omega_sq = -(z_probe * z_probe) * ratio
    omega_sq_re = jnp.real(omega_sq)
    good = (
        safe
        & jnp.isfinite(omega_sq_re)
        & (omega_sq_re > 0.0)
        & mode_mask
    )

    # Pad modes born DEAD: Ω = 0 (hence B = -Wc0·Ω/2 = 0) outside the
    # logical block — see ``n_mu_logical`` in the wrapper.
    #
    # INTERMEDIATE REDUCTION 2: the old form was a NESTED pair of full-tile
    # selects, ``where(mode_mask, where(good, sqrt, fallback), 0.0)``.  Because
    # ``good`` already contains ``& mode_mask``, the outer select can be folded
    # into the inner one's FALSE operand, and that operand then depends only on
    # ``mode_mask`` — a (mu, nu) 2-D array with NO q axis.  Equivalence, all
    # three reachable cases:
    #   good=T (=> mode_mask=T): old sqrt        ; new sqrt                 ✓
    #   good=F, mode_mask=T    : old fallback    ; new where(T,fallback,0)  ✓
    #   good=F, mode_mask=F    : old 0.0         ; new where(F,fallback,0)  ✓
    # Saves one full-tile f64 select (1.16 GiB) and one full-tile broadcast;
    # the surviving fallback operand is nq times smaller.
    _fallback_or_dead = jnp.where(mode_mask, fallback, 0.0)   # (mu, nu), 2-D
    omega_vals = jnp.where(good, jnp.sqrt(omega_sq_re), _fallback_or_dead)
    B_vals = -0.5 * Wc0 * omega_vals.astype(jnp.complex128)
    # ---------------------------------------------------------------------
    # THE ARENA (size campaign 2026-07-29, ladder notes R37).  ``n_modes`` used
    # to be computed as
    #     n_modes = jnp.sum(jnp.broadcast_to(mode_mask, good.shape)
    #                       .astype(jnp.float64))
    # and THAT SINGLE LINE was the whole 74.27 GiB allocation that killed
    # mu = 24,933 at P = 64.  ``mode_mask`` is built from ``jnp.arange(n_mu)``,
    # which carries NO sharding, so GSPMD kept this branch REPLICATED: every
    # rank materialised the FULL GLOBAL ``f64[nq, mu_pad, mu_pad]`` mask just to
    # add up its ones.  Read straight out of the failing run's own HLO:
    #     %fused_computation () -> f64[2,2496,2496]      <- global, not sharded
    #     allocation 33: size 95.06MiB, preallocated-temp
    #         95.06MiB; 3 values; f64[2,2496,2496], f64[2,312,312], f64[]
    # 2*2496*2496*8 = 99,680,256 B = 95.06 MiB  == the entire "arena", and at
    # production 16*24960*24960*8 = 79,744,204,800 B == the exact OOM byte
    # count.  The famous "32.0x one tile" multiple was never 32 temporaries: it
    # is the identity (mu_pad/mu_local)^2 * (8/16) = p_x^2/2 = 64/2, i.e. a
    # property of the 8x8 MESH, which is why it read exactly 32.0 at two very
    # different problem sizes (both were P=64).
    #
    # The value is a CONSTANT.  ``mode_mask`` has exactly ``n_log**2`` true
    # entries by construction (an outer AND of ``arange(n_mu) < n_log`` with
    # itself), broadcast over the leading axes, so the sum is exactly
    # ``prod(lead) * n_log**2`` -- a non-negative integer.  Summing 0.0/1.0 in
    # float64 is EXACT while every partial sum stays below 2**53 (production is
    # ~1e10), and the answer is that same integer, so emitting the integer
    # directly is BIT-IDENTICAL, not merely mathematically equal.  Guards are
    # untouched: ``mode_mask`` still gates ``good`` and still zeroes pad modes.
    _n_lead = 1
    for _d in good.shape[:-2]:
        _n_lead *= int(_d)
    _n_modes_exact = _n_lead * n_log * n_log
    if _n_modes_exact < (1 << 53):
        n_modes = jnp.asarray(float(_n_modes_exact), dtype=jnp.float64)
    else:
        # Unreachable on any hardware this runs on (needs mu ~ 2.4e7); kept so
        # the float64 exactness argument above is never silently violated.
        n_modes = jnp.sum(
            jnp.broadcast_to(mode_mask, good.shape).astype(jnp.float64))
    n_good = jnp.sum(good.astype(jnp.float64))
    omega_min = jnp.min(jnp.where(good, omega_vals, jnp.inf))
    omega_max = jnp.max(jnp.where(good, omega_vals, -jnp.inf))
    pair_scale = jnp.maximum(
        jnp.maximum(jnp.abs(Wc0), jnp.abs(Wc_probe)),
        jnp.finfo(jnp.float64).tiny,
    )
    pair_rel = jnp.abs(denom) / pair_scale
    pair_rel_min = jnp.min(jnp.where(good, pair_rel, jnp.inf))
    # RAW COUNTS, not the ratio (q-chunking 2026-07-29).  Both are sums of
    # booleans, i.e. EXACT integers in float64 (max here ~1e10 << 2^53), so
    # summing them across q-blocks is associativity-safe and the wrapper's
    # single division reproduces the one-shot value BIT-EXACTLY.
    return (
        omega_vals, B_vals, good, n_good, n_modes,
        omega_min, omega_max, pair_rel_min,
    )


@partial(jax.jit, static_argnums=(4,))
def _gn_ppm_fit_kernel_ordered(Wc0_qmunu, Wc_probe_qmunu, z_probe, fallback,
                               n_log):
    """The ordered-orientation twin of :func:`_gn_ppm_fit_kernel`.

    Same module, same eight outputs, plus the anti-Hermitian half of the
    probe.  The elementwise fit is handed the HERMITIAN half of
    ``W^c(i*omega_p)`` -- ``(W + W^H)/2`` over the trailing ``(mu, nu)`` pair
    -- so ``Omega`` stays real symmetric and ``B`` Hermitian whatever the
    deck (``docs/dev/notes/DERIVATION_gnppm_nonhermitian.md`` section 3).
    The adjoint is taken INSIDE the jit for the same reason
    ``common.sanity.check_hermitian`` takes its transpose inside one: on the
    ``P(None, 'x', 'y')`` layout the transpose is an X<->Y resharding, and
    fused under one module GSPMD moves the tile locally instead of
    all-gathering both operands.  This is a separate XLA program from the
    incumbent kernel by design: the incumbent stays bit-identical.
    """
    Wc_probe = jnp.asarray(Wc_probe_qmunu, dtype=jnp.complex128)
    Wc_probe_adj = jnp.conj(jnp.swapaxes(Wc_probe, -1, -2))
    herm = 0.5 * (Wc_probe + Wc_probe_adj)
    anti = 0.5 * (Wc_probe - Wc_probe_adj)
    (omega_vals, B_vals, good, n_good, n_modes,
     omega_min, omega_max, pair_rel_min) = _gn_ppm_fit_kernel(
        Wc0_qmunu, herm, z_probe, fallback, n_log)
    return (
        omega_vals, B_vals, good, n_good, n_modes,
        omega_min, omega_max, pair_rel_min, anti,
    )


@jax.jit
def _gn_ppm_odd_residue(anti_qmunu, omega_qmunu, z_probe):
    """``D = i a (omega_p^2 + Omega^2) / (2 omega_p)`` on live modes, else 0.

    ``a`` is the anti-Hermitian half of ``W^c(i*omega_p)`` and ``Omega`` the
    FINAL fitted pole (after any tail policy).  ``omega_p = |Im z|``.  ``D`` is
    Hermitian because ``i*a`` is and the multiplier is real symmetric.
    """
    anti = jnp.asarray(anti_qmunu, dtype=jnp.complex128)
    omega = jnp.asarray(omega_qmunu, dtype=jnp.float64)
    omega_p = jnp.abs(jnp.imag(jnp.asarray(z_probe, dtype=jnp.complex128)))
    scale = (omega_p * omega_p + omega * omega) / (2.0 * omega_p)
    D = (1j * anti) * scale.astype(jnp.complex128)
    return jnp.where(omega > 0.0, D,
                     jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))


def solve_laplace_minimax_interval(
    x_min: float,
    x_max: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    use_shipped_tables: bool = True,
) -> LaplaceMinimaxQuadrature:
    """Fit ``1/x ≈ sum alpha_l exp(-tau_l x)`` on ``[x_min, x_max]``.

    ``target_error`` is the requested physical L-infinity absolute error on
    ``[x_min, x_max]``.  The service works on ``[1, R]``; its request is
    therefore ``target_error * x_min``, while its achieved error is divided by
    ``x_min`` on return.  This is not a relative-at-endpoint tolerance.
    """

    x_min = max(float(x_min), _TINY)
    x_max = max(float(x_max), x_min * (1.0 + 1.0e-9))
    scaled_target_error = _scaled_laplace_error_bound(x_min, target_error)
    max_nodes = max(4, int(max_nodes))

    R = x_max / x_min

    served = _mm.serve(
        family="noncrossing", target="inverse",
        range_value=R, error_bound=scaled_target_error, n_max=max_nodes,
        use_shipped=bool(use_shipped_tables),
    )
    tau_hat, w_hat, err_hat = served.nodes, served.weights, served.max_error

    tau = tau_hat / x_min
    alpha = w_hat / x_min
    err_abs = err_hat / x_min

    return LaplaceMinimaxQuadrature(
        x_min=x_min,
        x_max=x_max,
        tau=np.asarray(tau, dtype=np.float64),
        alpha=np.asarray(alpha, dtype=np.float64),
        max_error=float(err_abs),
        provenance=served.provenance.one_line(),
    )


#: The most recent refusal from the ``complex_laplace`` selector, kept so
#: that a caller (or a test, or the census shim) can read WHY a request
#: fell through to the runtime solve without the door having to raise.
#: Under R1 stage 2 this refusal becomes the error; today it is only the
#: explanation, because arming the refusal is a separate, staged decision
#: and this commit changes nothing about what the default path computes
#: except where a certified table now answers.
LAST_IMAG_TABLE_REFUSAL: _beta_selector.TableRefusal | None = None


#: Ceiling on the nodes the odd-kernel augmentation may add to a served
#: imaginary-axis rule.  Measured 2026-09-01 on MoS2-, CrI3- and Si-like
#: (x_min, x_max, omega_p) triples: 5, 1 and 5 extras reach 1e-6 from the 7,
#: 11 and 11 even nodes; a weights-only refit on the even nodes alone stalls
#: at 1.8e-3 / 1.3e-5 / 5.7e-4.  Beyond this ceiling the rule refuses by
#: name rather than ship an odd channel it cannot represent.
ODD_KERNEL_MAX_EXTRA_NODES = 16


def _augment_odd_kernel_nodes(tau, x_min, x_max, omega_p, *,
                              gate_error: float,
                              max_extra: int = ODD_KERNEL_MAX_EXTRA_NODES,
                              n_grid: int = 4096, n_candidates: int = 48):
    """Nodes and weights for ``omega_p/(x^2+omega_p^2)`` on ``[x_min, x_max]``.

    The served even rule's nodes are kept as they are and the odd kernel is
    represented on them PLUS the fewest extra nodes, drawn greedily from a
    geometric candidate grid around the even nodes, that bring the measured
    sup-norm error under ``gate_error`` -- the same weights-only Lawson
    machinery and the same greedy pattern as
    :func:`refit_imag_alpha_augmented`, whose measured lesson applies here
    too: the even nodes do not resolve the odd kernel on their own.

    Returns ``(tau_full, beta, k_extra, max_err)``; the first ``len(tau)``
    entries of ``tau_full`` are the input nodes unchanged.  Raises when the
    ceiling is hit: an odd channel represented to 1e-3 is not the physics
    the caller asked for, and there is no even-rule fallback that is
    correct here.
    """
    tau_s = np.asarray(tau, dtype=np.float64)
    x = np.geomspace(float(x_min), float(x_max), int(n_grid))
    wp = float(omega_p)
    f_odd = wp / (x * x + wp * wp)
    cand = np.geomspace(float(np.min(tau_s)) / 8.0,
                        float(np.max(tau_s)) * 8.0, int(n_candidates))
    cur = tau_s.copy()
    beta, err = _lawson_weights_fit(cur, f_odd, x)
    k = 0
    while err > float(gate_error) and k < int(max_extra):
        best = None
        for c in cand:
            if np.any(np.abs(np.log(c / cur)) < 1.0e-9):
                continue
            b_try, e_try = _lawson_weights_fit(np.append(cur, c), f_odd, x)
            if best is None or e_try < best[1]:
                best = (c, e_try, b_try)
        if best is None:
            break
        cur = np.append(cur, best[0])
        beta, err = best[2], best[1]
        k += 1
    if err > float(gate_error):
        raise RuntimeError(
            "GATE odd_kernel_representation: the odd probe kernel missed "
            "the configured representation accuracy.\n"
            f"  got:  sup_error = {err:.3e}, target = "
            f"{float(gate_error):.3e}, extra_nodes = {k}, "
            f"max_extra = {int(max_extra)}, interval = "
            f"[{float(x_min):.6g}, {float(x_max):.6g}] Ry, "
            f"omega_p = {wp:.6g} Ry\n"
            "  want: sup_error <= minimax_target_error\n"
            "  why:  a less accurate odd rule would model broken-TR W at "
            "lower accuracy than the served even rule\n"
            "  doc:  docs/dev/notes/DERIVATION_gnppm_nonhermitian.md")
    return cur, np.asarray(beta, dtype=np.float64), int(k), float(err)


def solve_laplace_minimax_imag_interval(
    x_min: float,
    x_max: float,
    omega_p: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    use_shipped_tables: bool = True,
    beta_clause: str = _beta_selector.HEIGHT,
    print_fn=None,
    with_odd_kernel: bool = False,
) -> LaplaceMinimaxQuadrature:
    """Fit ``x/(x^2+omega_p^2) ≈ sum alpha_l exp(-tau_l x)`` on ``[x_min, x_max]``.

    Used for chi0(i*omega_p) where the resonant+antiresonant sum gives
    2*x/(x^2+omega_p^2) with x = E_c - E_v.

    ``with_odd_kernel=True`` (ordered orientations, magnets) additionally
    fits ``omega_p/(x^2+omega_p^2)`` -- the imaginary part of
    ``-1/(x+i*omega_p)`` whose real part this rule has always fitted -- and
    returns it in ``alpha_odd`` on the same nodes: the certified complex
    table's ``Im alpha`` when one answers, otherwise the even nodes plus a
    few greedily added ones (:func:`_augment_odd_kernel_nodes`), with the
    even weights zero on the extras so the even accumulation is unchanged.

    ``target_error`` is the requested physical L-infinity absolute error.
    As for :func:`solve_laplace_minimax_interval`, the scaled service request
    is ``target_error * x_min`` and the achieved scaled error is divided by
    ``x_min`` on return.

    THE BETA AXIS.  On the scaled interval this is ``u/(u^2 + beta^2)``
    with ``beta = omega_p / x_min``, which is the real part of the
    ``complex_laplace`` catalog's ``1/(u - i beta)`` -- one payload, two
    consumers, so ``alpha.real`` off a certified complex table is exactly
    what this function has always solved for.  ``beta_clause`` defaults to
    ``height`` because ``omega_p`` is a sampling line height over a band
    gap and not a pole width; see ``minimax.beta_selector`` for why
    that word has to be said rather than inferred.

    When no certified table covers the request the selector refuses in
    prose and this function does exactly what it did before: the same
    uncertified runtime solve, through the same cache, with the same key.
    """

    global LAST_IMAG_TABLE_REFUSAL

    x_min = max(float(x_min), _TINY)
    x_max = max(float(x_max), x_min * (1.0 + 1.0e-9))
    omega_p = float(omega_p)
    scaled_target_error = _scaled_laplace_error_bound(x_min, target_error)
    max_nodes = max(4, int(max_nodes))

    R = x_max / x_min
    omega_hat = omega_p / x_min

    # THE STRUCTURAL HOLE, AND THE HALF OF IT THIS BATCH CLOSES.
    # `catalog.json` ships 31 entries in two families, {'crossing': 5,
    # 'noncrossing': 26}; there is no `noncrossing_imag` family in it at
    # all, so this call never consulted a table and never could.  The
    # `complex_laplace` bundle is the campaign that answers it -- its
    # target's real part IS this function -- and the beta axis is what
    # makes those entries selectable.  So the path is two-branch now: ask
    # the axis for a certified table, and fall through to the door's
    # announced escape hatch when it refuses.  A refusal costs an
    # explanation and nothing else; under R1 stage 2 it becomes the error.
    picked = None
    if use_shipped_tables:
        picked = _beta_selector.select(
            range_value=R,
            beta=omega_hat,
            beta_clause=beta_clause,
            target_error=scaled_target_error,
            max_nodes=max_nodes,
        )
    if isinstance(picked, _beta_selector.TableSelection):
        LAST_IMAG_TABLE_REFUSAL = None
        _beta_selector.announce(picked, print_fn=print_fn)
        # The catalog's own alias: Re 1/(u - i beta) = u/(u^2 + beta^2) is
        # ``minimax._imag_target`` character for character, so this
        # consumer takes the real part of the same certified payload the
        # Sigma-side complex consumer takes whole.
        tau_hat = np.asarray(picked.tau, dtype=np.float64)
        w_hat = np.ascontiguousarray(np.real(picked.alpha), dtype=np.float64)
        err_hat = float(picked.certified_error)
        provenance = picked.one_line()
    else:
        LAST_IMAG_TABLE_REFUSAL = picked
        # `use_shipped=use_shipped_tables`, not the default: with the deck
        # key clear the caller asked for the uncertified path explicitly,
        # and the door should announce THAT rather than report a miss it
        # was never allowed to look for.
        served = _mm.serve(
            family="noncrossing_imag", target="inverse_imag",
            range_value=R, error_bound=scaled_target_error, n_max=max_nodes,
            omega_hat=omega_hat, use_shipped=use_shipped_tables,
        )
        tau_hat, w_hat, err_hat = (served.nodes, served.weights,
                                   served.max_error)
        provenance = served.provenance.one_line()

    tau = tau_hat / x_min
    alpha = w_hat / x_min
    err_abs = err_hat / x_min

    alpha_odd = None
    err_odd = float("nan")
    n_extra = 0
    if with_odd_kernel:
        tau = np.asarray(tau, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        if isinstance(picked, _beta_selector.TableSelection):
            # One certified payload, both parts: the complex table fits
            # 1/(u - i beta) in modulus, so its imaginary part on the same
            # nodes IS the odd kernel, at the certified modulus error.
            alpha_odd = np.ascontiguousarray(
                np.imag(picked.alpha), dtype=np.float64) / x_min
            err_odd = float(picked.modulus_error) / x_min
        else:
            tau_full, beta, n_extra, err_odd = _augment_odd_kernel_nodes(
                tau, x_min, x_max, omega_p,
                gate_error=max(float(target_error), float(err_abs)))
            alpha = np.concatenate([alpha, np.zeros(n_extra)])
            tau = tau_full
            alpha_odd = beta

    return LaplaceMinimaxQuadrature(
        x_min=x_min,
        x_max=x_max,
        tau=np.asarray(tau, dtype=np.float64),
        alpha=np.asarray(alpha, dtype=np.float64),
        max_error=float(err_abs),
        provenance=provenance,
        alpha_odd=alpha_odd,
        max_error_odd=err_odd,
        n_odd_extra=int(n_extra),
    )


def solve_phase_minimax_bandwidth(
    A_dim: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 500,
    eps_q: float = 1.0e-3,
    target_kind: str = "hgl",
    use_shipped_tables: bool = True,
) -> CrossingMinimaxQuadrature:
    """Fit crossing regularization target on ``[0, A_dim]`` as ``sum alpha_l sin(tau_l u)``.

    Error convention:
      ``target_error`` is the L-infinity absolute error on the target function itself,
      e.g. ``max_{u in [0, A_dim]} |G(u) - approx(u)|`` for the chosen regularization
      target. This is the same absolute convention used by the current solver and the
      shipped tables below.
    """

    A_dim = max(float(A_dim), 1.0e-12)
    target_error = max(float(target_error), 1.0e-14)
    eps_q = max(float(eps_q), 1.0e-12)
    max_nodes = max(8, int(max_nodes))
    kind = str(target_kind).strip().lower()

    served = _mm.serve(
        family="crossing", target=kind,
        range_value=A_dim, error_bound=target_error, n_max=max_nodes,
        eps_q=eps_q, use_shipped=bool(use_shipped_tables),
    )
    tau_hat, w_hat, err = served.nodes, served.weights, served.max_error
    return CrossingMinimaxQuadrature(
        A_dim=A_dim,
        tau=np.asarray(tau_hat, dtype=np.float64),
        alpha=np.asarray(w_hat, dtype=np.float64),
        max_error=float(err),
        target_kind=kind,
        provenance=served.provenance.one_line(),
    )




# ---------------------------------------------------------------------------
#  Quadrature builders — the χ₀/Σ frequency axes, solved on G's spectrum
#  (moved from gw/w_isdf.py 2026-07-09: B1 frequency code belongs with the
#  minimax engine, not one of its consumers).
# ---------------------------------------------------------------------------

def resolve_minimax_energy_reference(
    enk_v: jax.Array,
    enk_c: jax.Array,
    *,
    reference: str | float | int | None = "midgap",
    reference_fn: Callable[[jax.Array, jax.Array], float] | None = None,
) -> float:
    """Resolve the minimax energy reference used to shift band energies.

    This shift is algebraically neutral for χ0/W (only E_c-E_v enters), but
    exposing it at the top-level minimax pipeline keeps reference conventions
    explicit and synchronized with sigma paths.
    """
    if reference_fn is not None:
        return float(reference_fn(enk_v, enk_c))

    if reference is None:
        return 0.0
    if isinstance(reference, (int, float)):
        return float(reference)

    ref = str(reference).strip().lower()
    if ref in ("none", "raw", "zero"):
        return 0.0

    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64)
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64)
    vbm_ref = float(np.max(enk_v_host))
    cbm_ref = float(np.min(enk_c_host))

    if ref == "midgap":
        return 0.5 * (vbm_ref + cbm_ref)
    if ref == "vbm":
        return vbm_ref
    if ref == "cbm":
        return cbm_ref
    raise ValueError(f"Unknown minimax energy reference '{reference}'. Expected midgap/vbm/cbm/none or float.")


# ---------------------------------------------------------------------------
#  Top-level screening helpers (used directly by gw_jax.main)
# ---------------------------------------------------------------------------

def build_static_quadrature(wfns, minimax_config, *,
                            occupation_width_ry=None, print_fn=None):
    """Build static minimax quadrature and energy reference from wavefunction bundle.

    Returns (quad, e_ref) where quad is a LaplaceMinimaxQuadrature for 1/x
    on the band-energy interval, and e_ref is the global energy zero.
    """
    s = wfns.slices
    # NOTE(metal-screening): the transition window is still cut at the
    # ifmax-based b2 index rather than enumerated over pairs with nonzero
    # f_nk - f_mk.  What that cut used to do on a METAL is now handled by the
    # occupation-width floor on x_min below; see there.
    enk_v = wfns.enk[:, s.val]
    # ``cond_all_logical``, not ``cond``: this quadrature is built once and reused by
    # BOTH χ0 and Σ, so its interval must cover the union of the two band
    # sums.  ``cond`` is the χ0 leg only and would under-cover a deck whose
    # ``number_bands_sigma`` is the larger of the two.  Identical to ``cond``
    # on every unsplit deck (see BandSlices.cond_all_logical).
    # ``cond_all`` is the padded loaded carrier.  Its tail contains zero psi
    # but finite sentinel/reader energies, so using it here makes x_max (and
    # therefore every MPA sample frequency) depend on process count.  Spectral
    # extrema are physics and end at the logical union carried alongside it.
    enk_c = wfns.enk[:, s.cond_all_logical]
    e_ref = resolve_minimax_energy_reference(
        enk_v, enk_c, reference=minimax_config.energy_reference)

    # Interval derivation for 1/x on the band-energy span [x_min, x_max].
    # (Inlined from the former minimax_screening.build_static_minimax_window_pair;
    #  the window-pair object it returned was discarded here — only ``quad`` is used.)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64)
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64)
    if enk_v_host.size == 0 or enk_c_host.size == 0:
        raise ValueError(
            "Cannot build minimax window with empty valence/conduction energies.")
    vmin = float(np.min(enk_v_host))
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    cmax = float(np.max(enk_c_host))
    # ``cmin - vmax`` is the fundamental gap, which is the right x_min for an
    # insulator and MEANINGLESS on a metal: the bands cross E_F, the gap goes
    # <= 0, and x_min collapsed to _TINY = 1e-12.  That handed the solver the
    # degenerate interval [1e-12, ~7] Ry -- measured R = 6.9e12 with a scaled
    # tolerance of 1e-18, which no rule of any order meets, so the solver ran
    # its whole node ladder and surrendered the N_max rule after ~18 minutes.
    #
    # The floor is the occupation smearing width, and it is physics rather
    # than a guard.  chi0 needs (f_v - f_c)/x, not 1/x: for two states inside
    # the smearing window the occupation difference vanishes LINEARLY with x
    # (f_v - f_c ~ x/width), so the integrand is bounded as x -> 0 and 1/x
    # accuracy below the smearing width is accuracy nothing asks for.  A deck
    # with no smearing (an insulator) passes None and keeps the exact
    # incumbent interval.
    gap_ry = cmin - vmax
    floor_ry = _TINY
    if occupation_width_ry is not None:
        floor_ry = max(float(occupation_width_ry), _TINY)
    x_min = max(gap_ry, floor_ry)
    x_max = max(cmax - vmin, x_min * (1.0 + 1.0e-9))
    quad = solve_laplace_minimax_interval(
        x_min,
        x_max,
        target_error=float(minimax_config.target_error),
        max_nodes=int(minimax_config.max_nodes),
        use_shipped_tables=bool(minimax_config.use_shipped_tables),
    )
    if print_fn is not None:
        R = quad.x_max / quad.x_min
        # R2: this line used to say `nodes=` and `fit_err~` and never say
        # whether that came from a shipped table or four minutes of scipy.
        # It says now.
        print_fn(
            "  Minimax static window: "
            f"x=[{quad.x_min:.6e}, {quad.x_max:.6e}] Ry, "
            f"R={R:.2f}, nodes={quad.node_count}, fit_err~{quad.max_error:.3e}"
            f"  [{quad.provenance or 'provenance unrecorded'}]"
        )
    return quad, e_ref


def build_imag_quadrature(quad, omega_p, minimax_config, *, print_fn=None,
                          with_odd_kernel: bool = False):
    """Build imaginary-frequency minimax quadrature for x/(x²+ωp²).

    Uses the same energy interval as the static quadrature.
    ``with_odd_kernel`` adds the odd kernel ``ωp/(x²+ωp²)`` for the
    ordered-orientation χ₀ route (magnets) — see
    :func:`solve_laplace_minimax_imag_interval`.
    """
    quad_imag = solve_laplace_minimax_imag_interval(
        quad.x_min, quad.x_max, float(omega_p),
        target_error=float(minimax_config.target_error),
        max_nodes=int(minimax_config.max_nodes),
        use_shipped_tables=bool(minimax_config.use_shipped_tables),
        print_fn=print_fn,
        with_odd_kernel=bool(with_odd_kernel),
    )
    if print_fn is not None:
        R = quad_imag.x_max / quad_imag.x_min
        n_even = quad_imag.node_count - int(quad_imag.n_odd_extra)
        print_fn(
            f"  PPM imag-freq quadrature (ωp={float(omega_p):.4f} Ry): "
            f"R={R:.1f}, nodes={n_even}, err~{quad_imag.max_error:.1e}"
            f"  [{quad_imag.provenance or 'provenance unrecorded'}]")
        if quad_imag.alpha_odd is not None:
            print_fn(
                "  PPM imag-freq ODD kernel ωp/(x²+ωp²) (ordered "
                f"orientations, TR-odd channel): +{quad_imag.n_odd_extra} "
                f"nodes -> {quad_imag.node_count} total, "
                f"err~{quad_imag.max_error_odd:.1e} (gate "
                f"{max(float(minimax_config.target_error), float(quad_imag.max_error)):.1e})")
    return quad_imag


def _lawson_weights_fit(tau, f_x, x, n_iter: int = 60):
    """Weights-only sup-norm fit of ``f(x) ≈ Σ α_l exp(-τ_l x)``.

    Lawson's algorithm (iteratively reweighted least squares whose weights
    converge toward the L∞ solution) with column scaling for conditioning.
    Returns ``(alpha, max_err)`` — the best iterate by measured sup error.
    Host-side numpy/LAPACK, deterministic for identical inputs — the same
    per-rank replication contract the minimax solvers themselves rely on.
    """
    E = np.exp(-np.outer(x, tau))                  # (n_grid, L)
    s = np.linalg.norm(E, axis=0)
    s[s == 0.0] = 1.0
    Es = E / s
    w = np.ones(x.shape[0])
    best_a, best_e = None, np.inf
    for _ in range(int(n_iter)):
        sw = np.sqrt(w)
        a, *_ = np.linalg.lstsq(Es * sw[:, None], f_x * sw, rcond=None)
        r = Es @ a - f_x
        err = float(np.max(np.abs(r)))
        if err < best_e:
            best_a, best_e = a / s, err
        w *= np.abs(r) + 1.0e-30
        w /= w.sum()
    return np.asarray(best_a, dtype=np.float64), best_e


def refit_imag_alpha_augmented(quad, quad_dedicated, omega_p, *,
                               gate_error: float, n_grid: int = 4096):
    """Probe-χ₀ node plan for the reuse path (``ppm_probe_chi_reuse=auto``).

    Represent ``x/(x²+ωp²)`` on the STATIC quadrature's τ nodes plus the
    MINIMAL number of extra nodes, drawn greedily from the dedicated
    imag-axis quadrature's own node set, such that the measured sup-norm
    error meets ``gate_error``.  The probe χ₀ then reuses the static
    sweep's per-node G-build/FFT/contraction tensors on every shared node
    and only the ``k`` extras cost new compute.

    Weights-only refits are NOT enough on their own: the probe integrand
    is the Laplace transform of ``cos(ωp t)`` and the 1/x-minimax static
    grid is far too coarse to resolve that oscillation in the τ tail
    (measured 2.6e-4 sup error vs the dedicated solver's 1.3e-6 at the
    b300 window, job 7885097) — hence the augmentation.

    GUARANTEED to terminate acceptably: with ALL dedicated nodes appended,
    the exact dedicated solution (its α on its nodes, zeros on the static
    nodes) is in the feasible set and is installed verbatim whenever the
    fitted candidate is worse.

    Returns ``(tau_full, alpha_static_row, alpha_probe_row, k_extra,
    max_err)``: the union node vector, the static weights zero-padded onto
    it (row 0 of the fused sweep — zero-weight extras add exact zeros, so
    the static accumulation is numerically the static quadrature), the
    probe weights on it, the number of extra nodes, and the measured
    sup-norm error of the probe representation.
    """
    x_min = float(quad.x_min)
    x_max = float(quad.x_max)
    omega_p = float(omega_p)
    tau_s = np.asarray(quad.tau, dtype=np.float64)
    tau_d = np.asarray(quad_dedicated.tau, dtype=np.float64)
    alpha_d = np.asarray(quad_dedicated.alpha, dtype=np.float64)

    x = np.geomspace(x_min, x_max, int(n_grid))
    f_x = x / (x * x + omega_p * omega_p)

    def _pack(extras_idx, alpha_probe, err):
        k = len(extras_idx)
        tau_full = np.concatenate([tau_s, tau_d[extras_idx]])
        a_static = np.concatenate([
            np.asarray(quad.alpha, dtype=np.float64), np.zeros(k)])
        return tau_full, a_static, np.asarray(alpha_probe), int(k), float(err)

    chosen: list = []
    remaining = list(range(tau_d.shape[0]))
    a_cur, e_cur = _lawson_weights_fit(tau_s, f_x, x)
    while e_cur > float(gate_error) and remaining:
        best = None
        for c in remaining:
            tau_try = np.concatenate([tau_s, tau_d[chosen + [c]]])
            a_try, e_try = _lawson_weights_fit(tau_try, f_x, x)
            if best is None or e_try < best[1]:
                best = (c, e_try, a_try)
        chosen.append(best[0])
        remaining.remove(best[0])
        a_cur, e_cur = best[2], best[1]

    if e_cur > float(gate_error):
        # All extras in and still above gate: install the exact dedicated
        # embedding (zeros on static nodes, dedicated α on its own nodes)
        # — same math as the dedicated pass, by construction.
        chosen = list(range(tau_d.shape[0]))
        alpha_probe = np.concatenate([np.zeros(tau_s.shape[0]), alpha_d])
        return _pack(chosen, alpha_probe,
                     float(quad_dedicated.max_error))
    return _pack(chosen, a_cur, e_cur)


def build_real_quadrature(quad, Omega, minimax_config, *, print_fn=None):
    """Build real-frequency (HL-PPM) χ₀(Ω) quadrature without a new minimax kernel.

    Decomposes the real-axis target into two ``1/y`` pieces and reuses
    the existing static (noncrossing) Laplace minimax twice::

        x / (x² - Ω²) = (1/2) · [ 1/(x - Ω)  +  1/(x + Ω) ]
                      = -(1/2)/(Ω - x)  +  (1/2)/(Ω + x)

    For ``Ω > x_max`` both ``Ω-x`` and ``Ω+x`` are strictly positive on
    ``x ∈ [x_min, x_max]``, so each can be approximated by a standard
    ``1/y`` minimax on the shifted interval (no new solver needed).

    Combining via the substitutions ``y = Ω-x`` and ``y = Ω+x`` and
    folding the constant ``e^{-τ·Ω}`` shift into the weights gives the
    same ``Σ_l α_l e^{-τ_l x}`` representation that ``compute_chi0``
    already consumes — with mixed-sign ``τ_l``: positive on the
    ``(Ω+x)`` branch, negative on the ``(Ω-x)`` branch.

    The numerical-stability prefold inside ``compute_chi0`` works
    transparently because in the realistic HL regime (``Ω`` ≈ 200 Ry,
    ``x_max`` ≈ 5 Ry → ``R'`` of either shifted interval ≈ 1.03)
    each ``1/y`` minimax needs only 1-3 nodes and ``|τ_l|`` ≈ ``1/Ω``,
    so any residual exponent ``|τ_l|·x_range`` ≈ 0.025 is harmless.

    Requires ``Omega > quad.x_max``.
    """
    Omega = float(Omega)
    if Omega <= float(quad.x_max):
        raise ValueError(
            f"build_real_quadrature requires Omega > x_max "
            f"(got Omega={Omega}, x_max={quad.x_max}). "
            f"HL-PPM is only defined for probes above all transitions."
        )
    target_error = float(minimax_config.target_error)
    max_nodes = int(minimax_config.max_nodes)

    # (Ω + x) branch: y ∈ [Ω + x_min, Ω + x_max] (strictly positive).
    quad_plus = solve_laplace_minimax_interval(
        Omega + quad.x_min, Omega + quad.x_max,
        target_error=target_error, max_nodes=max_nodes,
    )
    tau_plus = np.asarray(quad_plus.tau, dtype=np.float64)
    alpha_plus = (
        +0.5 * np.asarray(quad_plus.alpha, dtype=np.float64)
        * np.exp(-tau_plus * Omega)
    )

    # (Ω - x) branch: y ∈ [Ω - x_max, Ω - x_min] (strictly positive for Ω > x_max).
    quad_minus = solve_laplace_minimax_interval(
        Omega - quad.x_max, Omega - quad.x_min,
        target_error=target_error, max_nodes=max_nodes,
    )
    tau_minus_raw = np.asarray(quad_minus.tau, dtype=np.float64)
    # 1/(Ω - x) ≈ Σ α e^{-τ(Ω-x)} = Σ [α e^{-τ·Ω}] e^{+τ·x}
    # Cast into the kernel's e^{-τ'·x} form by τ' = -τ.  Decomposition sign is -1/2.
    tau_minus = -tau_minus_raw
    alpha_minus = (
        -0.5 * np.asarray(quad_minus.alpha, dtype=np.float64)
        * np.exp(-tau_minus_raw * Omega)
    )

    tau = np.concatenate([tau_plus, tau_minus])
    alpha = np.concatenate([alpha_plus, alpha_minus])
    err_combined = float(0.5 * (quad_plus.max_error + quad_minus.max_error))

    # Two branches, two artifacts.  A fused rule whose halves came from
    # different sources (one shipped, one solved) is a real possibility on a
    # wide Ω, so both provenances ride rather than one standing in for the
    # pair.
    fused = LaplaceMinimaxQuadrature(
        x_min=float(quad.x_min),
        x_max=float(quad.x_max),
        tau=tau,
        alpha=alpha,
        max_error=err_combined,
        provenance=(f"+branch: {quad_plus.provenance or 'unrecorded'} ; "
                    f"-branch: {quad_minus.provenance or 'unrecorded'}"),
    )

    if print_fn is not None:
        print_fn(
            f"  PPM real-freq quadrature (Ω={Omega:.4f} Ry, "
            f"decomposed via 1/y minimax): "
            f"+branch nodes={quad_plus.node_count} (R'={Omega/quad.x_min + quad.x_max/quad.x_min:.3f}), "
            f"-branch nodes={quad_minus.node_count} "
            f"(R'={(Omega-quad.x_min)/(Omega-quad.x_max):.3f}), "
            f"err~{err_combined:.1e}  [{fused.provenance}]")
    return fused
