"""Inputs of the packed photon Gamma-cell completion.

``bispinor_gw = full_static_cohsex`` completes the ``q = Gamma, G = 0`` slot
of the packed sixteen-block V and W (``gw.head_correction.
complete_static_photon_q0``) from one bounded response

    ``R(q) = q_a H_a(sigma_H) + q_a q_b S_ab``

that this module assembles.  Its content, by declaration:

* present: the charge CC ``q^2`` head ``S^{00}`` from the incumbent scalar
  producer (:func:`gw.qsgw_head.build_dft_head_response` at the requested
  imaginary frequency),
  the charge one-leg wings ``Y^{0}``/``Z^{0}`` that fold it through the
  packed body, and the Hall CT/TC ``q^1`` term generated structurally from
  ``sigma_H`` (:func:`gw.head_correction.static_hall_linear_response`);
* omitted by model: the current ``q^2`` response (TT, CT/TC), the current
  wings, the uniform static current response ``tt_q0`` (zero by gauge
  invariance for an insulator), the diamagnetic/contact terms and the
  negative-energy (complement-space) closure.  They are never stored as
  accidental zeros of a larger schema; ``S_direct`` has charge support only.

The Hall term is optional.  ``sigma_H`` is the exact requested row of the
immutable frequency-sample artifact written by
``get_dipole_mtxels --static-gauge-hall-only`` when the deck's
``static_gauge_hall_file`` exists and authenticates against the run's WFN,
band manifold and k-count; when the file is absent ``sigma_H = 0`` and
``hall_source`` says so.  For a Chern-trivial insulator the static Hall
coefficient is exactly zero in the converged limit (it is the occupied
Berry-curvature sum, i.e. a Chern number). Dynamic broken-TR GN requests both
``z=0`` and ``z=i*omega_p`` through this one producer; an absent artifact is
then announced rather than treated as a measured finite-frequency zero.

This module owns no WFN load, symmetry unfold, current, FFT, body-response
or packing implementation.  It composes the existing routines and retains
only O(N_mu) wing carriers.  The record it issues is sealed: only the
producer below can construct it, so a fabricated response cannot reach the
completion.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from gw.photon_layout import PhotonBasisLayout, pack_photon_channel_vectors


_PRODUCER_TOKEN = object()
HALL_SOURCE_NONE = "none: sigma_H = 0 (no static_gauge_hall_file)"


def _canonical_wfn_sha256(value) -> str:
    value = str(value).strip()
    if (len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(
            "static photon head WFN identifier must be 64 lowercase hex")
    return value


@dataclass(frozen=True)
class StaticPhotonHeadResponse:
    r"""Inputs for ``R(q)=q_a H_a(sigma_H)+q_a q_b S_ab``.

    ``S_direct`` and the wings have charge support only.  The Hall tensor is
    intentionally not stored: :func:`gw.head_correction.
    static_hall_linear_response` constructs it from ``sigma_H`` at the
    numerical kernel that consumes it.  ``hall_source`` records where
    ``sigma_H`` came from (the authenticated artifact path, or
    :data:`HALL_SOURCE_NONE`).
    """

    layout: PhotonBasisLayout
    dimension: int
    S_direct: jax.Array               # (d,d,4,4), replicated charge CC
    sigma_H: jax.Array                # (3,), replicated real bohr^-1
    hall_source: str
    Y_x: jax.Array                    # (d,4,Npacked), P(None,None,'x')
    Z_y: jax.Array                    # (d,Npacked,4), P(None,'y',None)
    charge_Y_x: jax.Array             # (d,Ncharge), P(None,'x')
    charge_Z_y: jax.Array             # (d,Ncharge), P(None,'y')
    ward_residual: float
    hermiticity_residual: float
    wing_reciprocity_residual: float
    _producer_token: object
    frequency_ry: complex = 0.0 + 0.0j

    def __post_init__(self) -> None:
        if self._producer_token is not _PRODUCER_TOKEN:
            raise TypeError(
                "StaticPhotonHeadResponse is issued only by "
                "build_static_photon_head_response")


def _same_mesh_sharding(array, mesh: Mesh, spec: P) -> bool:
    sharding = getattr(array, "sharding", None)
    return (
        isinstance(sharding, NamedSharding)
        and tuple(sharding.mesh.axis_names) == tuple(mesh.axis_names)
        and np.array_equal(sharding.mesh.devices, mesh.devices)
        and sharding.is_equivalent_to(NamedSharding(mesh, spec), array.ndim)
    )


def require_static_photon_head_response(
    response: StaticPhotonHeadResponse, mesh_xy: Mesh,
) -> StaticPhotonHeadResponse:
    """Check the bounded model's support, dtype and sharding."""
    if not isinstance(response, StaticPhotonHeadResponse):
        raise TypeError(
            "the packed static photon completion requires "
            f"StaticPhotonHeadResponse; got {type(response).__name__}")
    response.layout.assert_mesh(mesh_xy)

    n_packed = int(response.layout.packed_extent)
    dimension = int(response.dimension)
    if dimension not in (2, 3):
        raise ValueError(
            f"static photon head dimension must be 2 or 3; got {dimension}")
    arrays = (
        (response.S_direct, "S_direct",
         (dimension, dimension, 4, 4), np.complex128, P()),
        (response.sigma_H, "sigma_H", (3,), np.float64, P()),
        (response.Y_x, "Y_x", (dimension, 4, n_packed), np.complex128,
         P(None, None, "x")),
        (response.Z_y, "Z_y", (dimension, n_packed, 4), np.complex128,
         P(None, "y", None)),
        (response.charge_Y_x, "charge_Y_x",
         (dimension, response.layout.padded_extent(0)), np.complex128,
         P(None, "x")),
        (response.charge_Z_y, "charge_Z_y",
         (dimension, response.layout.padded_extent(0)), np.complex128,
         P(None, "y")),
    )
    for array, name, shape, dtype, spec in arrays:
        if tuple(array.shape) != shape:
            raise ValueError(f"{name} shape {array.shape} != {shape}")
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise TypeError(f"{name} dtype {array.dtype} != {np.dtype(dtype)}")
        if not _same_mesh_sharding(array, mesh_xy, spec):
            raise ValueError(f"{name} must have production sharding {spec}")

    S = np.asarray(jax.device_get(response.S_direct))
    sigma = np.asarray(jax.device_get(response.sigma_H))
    if not np.all(np.isfinite(sigma)):
        raise ValueError("static photon head sigma_H is not finite")
    if np.any(S[:, :, 0, 1:] != 0.0) or np.any(S[:, :, 1:, :] != 0.0):
        raise ValueError("static photon head S has non-charge support")
    from gw.head_correction import static_gauge_tensor_residuals
    structural_ward, structural_hermiticity = (
        static_gauge_tensor_residuals(S))
    residuals = np.asarray((
        response.ward_residual, response.hermiticity_residual,
        response.wing_reciprocity_residual,
        structural_ward, structural_hermiticity), dtype=np.float64)
    if np.any(~np.isfinite(residuals)) or np.any(residuals < 0.0):
        raise ValueError("static photon head response has invalid residuals")
    if max(response.ward_residual, structural_ward) > 1.0e-8:
        raise ValueError(
            "static photon head response violates the static Ward gate")
    if max(response.hermiticity_residual,
           structural_hermiticity) > 1.0e-10:
        raise ValueError(
            "static photon head response violates the Hermiticity gate")
    if response.wing_reciprocity_residual > 1.0e-10:
        raise ValueError(
            "static photon head response violates wing reciprocity")
    frequency = complex(response.frequency_ry)
    if (not np.isfinite(frequency.real) or not np.isfinite(frequency.imag)
            or frequency.real != 0.0):
        raise ValueError(
            "photon head completion samples must lie on the imaginary axis; "
            f"got frequency_ry={frequency!r}")
    return response


def _replicated(value, mesh: Mesh, *, dtype):
    return device_put_process_local(
        np.asarray(value, dtype=dtype), NamedSharding(mesh, P()))


def _channel_zeros(nq: int, extent: int, mesh: Mesh, axis: str):
    return device_put_process_local(
        np.zeros((int(nq), int(extent)), dtype=np.complex128),
        NamedSharding(mesh, P(None, axis)))


def build_static_photon_head_response(
    wfns,
    *,
    input_dir: str,
    mesh: Mesh,
    wfn,
    meta,
    config,
    layout: PhotonBasisLayout,
    hall_transaction=None,
    wfn_fingerprint_binding=None,
    frequency_ry: complex = 0.0 + 0.0j,
    direct_response=None,
) -> StaticPhotonHeadResponse:
    r"""Compose one imaginary-frequency charge/Hall head sample.

    The scalar response is evaluated by
    :func:`gw.qsgw_head.build_dft_head_response`.  ``direct_response`` may
    carry a batched evaluation containing ``frequency_ry``; this is how the
    32-point Faraday plan pays the energy-ordered pair loop once.  Its
    periodic-direction velocity rows become the derivatives of the charge
    head and charge-only wings.  Three exact-zero transverse vectors are
    passed to the canonical packer; no current wing is inferred.

    ``hall_transaction`` is either ``None`` (``sigma_H = 0``) or the full-BZ
    result of :func:`gw.qsgw_head.static_gauge_hall_transaction`, which must
    name the same WFN identity and band manifold as the charge response.  This
    owner.  ``frequency_ry`` is exact and must be purely imaginary.  The
    incumbent default is ``z=0``; the Faraday completion requests its stored
    ``z=i*omega_p`` row through the same object and the same response kernel.
    """
    from common.parallel_transport import (
        fingerprint_from_binding, wfn_fingerprint)
    from gw.head_correction import (
        canonicalize_static_gauge_q2_tensor,
        static_gauge_tensor_residuals,
    )
    from gw.qsgw_head import (
        StaticGaugeHallTransaction, build_dft_head_response)

    if not isinstance(layout, PhotonBasisLayout):
        raise TypeError(
            "static photon head response requires PhotonBasisLayout")
    layout.assert_mesh(mesh)
    try:
        dimension = int(config.sys_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "static photon head response requires explicit config.sys_dim") \
            from exc
    if dimension not in (2, 3):
        raise ValueError(
            "static photon head response supports config.sys_dim in (2,3); "
            f"got {dimension}")
    frequency = complex(frequency_ry)
    if (not np.isfinite(frequency.real) or not np.isfinite(frequency.imag)
            or frequency.real != 0.0):
        raise ValueError(
            "photon head response frequency must lie on the imaginary axis; "
            f"got {frequency!r}")

    wfn_fp = _canonical_wfn_sha256(
        wfn_fingerprint(wfn)
        if wfn_fingerprint_binding is None
        else fingerprint_from_binding(wfn_fingerprint_binding, wfn))
    start, stop = int(meta.b_id_0), int(meta.b_id_4_chi_user)
    if hall_transaction is None:
        sigma_host = np.zeros(3, dtype=np.float64)
        hall_source = HALL_SOURCE_NONE
    else:
        if not isinstance(hall_transaction, StaticGaugeHallTransaction):
            raise TypeError(
                "static photon head response requires the sealed full-BZ "
                "Hall transaction or None")
        if (int(hall_transaction.band_start),
                int(hall_transaction.band_stop)) != (start, stop):
            raise ValueError(
                "charge and Hall responses use different band manifolds: "
                f"charge=[{start},{stop}), Hall=[{hall_transaction.band_start},"
                f"{hall_transaction.band_stop})")
        if _canonical_wfn_sha256(hall_transaction.wfn_fingerprint) != wfn_fp:
            raise ValueError(
                "charge and Hall responses use different WFN identities")
        sigma_sample = np.asarray(jax.device_get(
            hall_transaction.sigma_H_at(frequency)), dtype=np.complex128)
        if (sigma_sample.shape != (3,)
                or not np.all(np.isfinite(sigma_sample))
                or np.any(np.imag(sigma_sample) != 0.0)):
            raise ValueError(
                "Hall transaction has an invalid real imaginary-axis sample")
        sigma_host = np.asarray(np.real(sigma_sample), dtype=np.float64)
        hall_source = (
            f"{hall_transaction.producer_id} "
            f"(operator {hall_transaction.hamiltonian_config_operator_fingerprint}; "
            f"z={frequency!r} Ry)")

    direct = direct_response
    if direct is None:
        direct = build_dft_head_response(
            wfns, (frequency,), input_dir=input_dir, mesh=mesh,
            wfn=wfn, meta=meta, config=config,
            wfn_fingerprint_binding=wfn_fingerprint_binding)
    try:
        frequency_index = tuple(complex(value) for value in direct.omegas).index(
            frequency)
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "batched direct photon-head response does not contain exact "
            f"frequency {frequency!r} Ry") from exc
    if direct.Y_x is None or direct.Z_y is None:
        raise ValueError("incumbent charge response returned no body wings")
    charge_extent = int(layout.padded_extent(0))
    n_frequency = len(direct.omegas)
    if (tuple(direct.Y_x.shape) != (n_frequency, 3, charge_extent)
            or tuple(direct.Z_y.shape) != (n_frequency, charge_extent, 3)):
        raise ValueError(
            "charge response/layout mismatch: "
            f"Y={direct.Y_x.shape}, Z={direct.Z_y.shape}, "
            f"charge padded extent={charge_extent}")

    charge_y = direct.Y_x[frequency_index, :dimension, :]
    charge_z = jnp.transpose(
        direct.Z_y[frequency_index, :, :dimension], (1, 0))
    zeros_x = tuple(
        _channel_zeros(dimension, layout.padded_extent(A), mesh, "x")
        for A in range(1, 4))
    zeros_y = tuple(
        _channel_zeros(dimension, layout.padded_extent(A), mesh, "y")
        for A in range(1, 4))
    Y_x = pack_photon_channel_vectors(
        (charge_y, *zeros_x), layout, mesh, axis_name="x")
    Z_packed_y = pack_photon_channel_vectors(
        (charge_z, *zeros_y), layout, mesh, axis_name="y")
    Z_y = jnp.transpose(Z_packed_y, (0, 2, 1))

    S_host = np.zeros(
        (dimension, dimension, 4, 4), dtype=np.complex128)
    charge_S = np.asarray(
        jax.device_get(direct.S_direct[
            frequency_index, :dimension, :dimension]),
        dtype=np.complex128)
    S_host[:, :, 0, 0] = charge_S
    S_direct = canonicalize_static_gauge_q2_tensor(
        _replicated(S_host, mesh, dtype=np.complex128))
    sigma_H = _replicated(sigma_host, mesh, dtype=np.float64)
    # At static imaginary frequency the Adler--Wiser weight is real, hence
    # the two incumbent wing orientations obey Z[b,mu]=conj(Y[b,mu]).  Move
    # only this O(N_mu) vector to Y sharding for a scalar certificate.
    charge_z_x = device_put_process_local(
        charge_z, NamedSharding(mesh, P(None, "x")))
    wing_delta = jnp.max(jnp.abs(charge_y - jnp.conj(charge_z_x)))
    wing_scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(charge_y)),
                    jnp.max(jnp.abs(charge_z_x))), 1.0e-300)
    wing_reciprocity = float(
        np.asarray(jax.device_get(wing_delta / wing_scale)))
    ward, hermiticity = static_gauge_tensor_residuals(S_direct)

    response = StaticPhotonHeadResponse(
        layout=layout, dimension=dimension,
        S_direct=S_direct, sigma_H=sigma_H, hall_source=hall_source,
        Y_x=Y_x, Z_y=Z_y,
        charge_Y_x=charge_y, charge_Z_y=charge_z,
        ward_residual=float(ward), hermiticity_residual=float(hermiticity),
        wing_reciprocity_residual=wing_reciprocity,
        _producer_token=_PRODUCER_TOKEN,
        frequency_ry=frequency,
    )
    return require_static_photon_head_response(response, mesh)


__all__ = [
    "HALL_SOURCE_NONE",
    "StaticPhotonHeadResponse",
    "build_static_photon_head_response",
    "require_static_photon_head_response",
]
