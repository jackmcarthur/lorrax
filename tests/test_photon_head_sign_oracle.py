"""One-convention sign oracle for the packed Gamma-cell photon completion.

Two lanes fixed different halves of the ``q -> 0`` photon head on different
branches (the ``D -> P`` Adler--Wiser wing sign and the Hall CT/TC ``-i``),
each with its own oracle over its own half.  Nothing had checked the MERGED
4x4 completion with every block nonzero at once.  This module does, from ONE
written-down convention:

CONVENTION (the single place it is stated; see
``reports/bisp_i_head_sign_audit_2026-09-01/report.md`` for the derivation)

* Irreducible static response, Adler--Wiser, energy-ordered pairs
  (``dE = E_bra - E_ket > 0``, ``f_diff = f_ket - f_bra``)::

      chi_AB(q,z) = (4 / (V * Nk * nspin * nspinor))
                    * sum_{dE>0} f_diff * dE / (z^2 - dE^2)
                      * conj(M_A) * M_B

* Long-wavelength charge vertex: ``M_0(ij) = q_a D_a(ij)`` with the density
  jet ``D_a = -v_a / dE``.  Equivalently ``P_a = v_a = -dE * D_a`` -- the
  ``P = -Delta*D`` relation both lanes argued from.  A vertex carrying ONE
  charge leg therefore picks up exactly one minus relative to the naive
  ``v``-spelled kernel; a vertex carrying TWO charge legs picks up none.
* Centroid body vertex: ``M_mu(ij) = sum_s conj(psi_i(mu)) psi_j(mu)``.
* Coulomb-gauge bare photon propagator, raw vcoul units (no ``1/V``)::

      D00 = v(q),  D0i = Di0 = 0,  Dij = COULOMB_GAUGE_TT_SIGN * v * P^T_ij
                                        = -v (delta_ij - qhat_i qhat_j)

* Dyson: ``eps = 1 - D chi``, so ``W_h = (1 - D R)^{-1} D`` with ``R`` the
  irreducible head response.  Equivalently, on the subspace where ``D`` is
  invertible, ``W_h = (D^{-1} - R)^{-1}`` -- the spelling this oracle uses,
  because it is a different algebra from the production ``solve(I-DR, D)``.
* Head/body Schur fold: ``S_eff = S + Y W_Gamma Z / V`` is the irreducible
  head response dressed by local fields, with ``Y = V*chi_hb``,
  ``Z = V*chi_bh`` the SAME-sign irreducible wings.
* Rank-4 reattachment: with ``b_u(q) = (1, qx, qy)``, the completed body is
  ``W = W0 + sum_uv L_u <b_u W_h b_v> R_v / V`` where
  ``L = (conj(g0), (W0 Z_x)^T, (W0 Z_y)^T)`` and ``R = (g0, Y_x W0, Y_y W0)``.

SCOPE, stated up front (TASTE.md: state what a check could NOT have seen).

* Part A tests the DEFINITIONS: it builds ``S``, the wings and the Hall CT
  from ONE reference function and compares them against the three
  production kernels.  This is the only part that can see a sign error
  shared by both wings (a joint ``Y,Z`` flip is invisible to every
  downstream check, because the Schur fold is quadratic in the wings).
* Part B tests the ASSEMBLY: the fold, the coupled 4x4 Dyson, the cubature
  moments and the rank-4 insertion, against an independent NumPy path on
  the SAME provider-issued cubature.  It consumes the response record, so
  it is blind to a joint wing flip and to a sign error inside the shared
  ``vcoul`` D; negative controls N2/N3 measure that sensitivity explicitly.
* Part C tests the ABSOLUTE CENTROID MEASURE in an exact ISDF model:
  centroids are every real-space point and ``zeta_mu(r)=delta_mu,r``.  The
  physical body operator is ``Omega*W_code``, while the directly evaluated
  Adler--Wiser wings are ``chi_hb=Y/Omega`` and ``chi_bh=Z/Omega``.  It
  forms the wing attachment and all four reattachment families through
  those two independent representations.  Omitting the final physical-to-
  stored ``1/Omega`` is an explicit factor-``Omega`` negative control.
* Meshes are 1x1 (and 2x2 when four devices are visible).  Sharding,
  collectives and the padded-channel mask are NOT the subject here; a 2x2
  arm exercises the mask only incidentally.
"""
from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

import common.parallel_transport as parallel_transport  # noqa: E402
import gw.qsgw_head as qsgw_head  # noqa: E402
from gw.head_correction import (  # noqa: E402
    _adjoint_factor_family,
    STATIC_PHOTON_CHARGE_BLOCK_ABSENT,
    STATIC_PHOTON_CHARGE_BLOCK_PRESENT,
    _dynamic_photon_head_moments,
    _dynamic_photon_head_small_system,
    _faraday_cttc_factor_pairs,
    _fit_ordered_faraday_factor_samples,
    canonicalize_static_gauge_q2_tensor,
    complete_static_photon_q0,
    fit_dynamic_photon_cttc_q0,
    fold_small_head_wings_sharded,
    small_head_wing_halves_sharded,
    static_hall_linear_response,
)
from gw.photon_layout import (  # noqa: E402
    PhotonBasisLayout, pack_photon_channel_vectors, pack_photon_operator,
    photon_block_view, photon_q0_low_rank_block)
from gw.qsgw_head import (  # noqa: E402
    head_s_tensor_sharded,
    head_wings_sharded,
    raw_hall_pseudovector_sharded,
)

_WFN_SHA = "7" * 64


# --------------------------------------------------------------------------
# THE convention, written once, as executable code.
# --------------------------------------------------------------------------

def adler_wiser_chi(M_left, M_right, energies, occupations, *,
                    cell_volume, nk_tot, nspin, nspinor, z=0.0 + 0.0j):
    """``chi_AB`` from the convention in this module's docstring.

    ``M_left``/``M_right`` are ``(A, nk, nb, nb)`` vertex matrices in the
    band-pair (bra, ket) layout.  Only energy-ordered pairs contribute.
    """
    e = np.asarray(energies, dtype=np.float64)
    f = np.asarray(occupations, dtype=np.float64)
    dE = e[:, :, None] - e[:, None, :]              # bra - ket
    f_diff = f[:, None, :] - f[:, :, None]          # ket - bra
    ordered = dE > 0.0
    denom = z * z - dE * dE
    weight = np.where(ordered, f_diff * dE / np.where(ordered, denom, 1.0), 0.0)
    pref = 4.0 / (float(cell_volume) * float(nk_tot)
                  * float(nspin) * float(nspinor))
    return pref * np.einsum(
        "aknm,knm,bknm->ab",
        np.conj(np.asarray(M_left)), weight, np.asarray(M_right),
        optimize=True)


def density_jet(velocity, energies):
    """``D_a(ij) = -v_a(ij)/dE``, zero on non-energy-ordered pairs."""
    e = np.asarray(energies, dtype=np.float64)
    dE = e[:, :, None] - e[:, None, :]
    ordered = dE > 0.0
    safe = np.where(ordered, dE, 1.0)
    return np.where(ordered[None], -np.asarray(velocity) / safe[None], 0.0)


# --------------------------------------------------------------------------
# Part A -- the three q->0 response DEFINITIONS under one convention.
# --------------------------------------------------------------------------

def _mesh(side=1):
    devices = jax.devices()
    if side * side > len(devices):
        side = 1
    return Mesh(np.asarray(devices[:side * side]).reshape(side, side),
                ("x", "y"))


def _micro_model(seed=20260901):
    rng = np.random.default_rng(seed)
    nk, nb, ns, nmu, nocc = 3, 6, 2, 5, 3
    energies = np.sort(rng.normal(size=(nk, nb)), axis=1)
    raw = (rng.normal(size=(3, nk, nb, nb))
           + 1j * rng.normal(size=(3, nk, nb, nb)))
    velocity = 0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))
    occupations = np.zeros((nk, nb))
    occupations[:, :nocc] = 1.0
    # psi_xn / psi_yn legacy carrier layout is (nk, ns, nmu, nb).
    psi = (rng.normal(size=(nk, ns, nmu, nb))
           + 1j * rng.normal(size=(nk, ns, nmu, nb)))
    return SimpleNamespace(
        nk=nk, nb=nb, ns=ns, nmu=nmu, nocc=nocc, nspin=1, nspinor=2,
        cell_volume=37.25, energies=energies, occupations=occupations,
        velocity=velocity, psi=psi)


def _centroid_vertex(psi):
    """``M_mu(ij) = sum_s conj(psi_i(mu)) psi_j(mu)`` as ``(nmu,nk,nb,nb)``."""
    return np.einsum("ksmi,ksmj->mkij", np.conj(psi), psi, optimize=True)


def part_a_residuals(mesh):
    """Return the definition-level residual dictionary."""
    m = _micro_model()
    D_jet = density_jet(m.velocity, m.energies)
    out = {}

    # --- S00: two charge legs, both minuses cancel -------------------------
    S_ref = adler_wiser_chi(
        D_jet, D_jet, m.energies, m.occupations,
        cell_volume=m.cell_volume, nk_tot=m.nk, nspin=m.nspin,
        nspinor=m.nspinor)
    S_code = np.asarray(head_s_tensor_sharded(
        m.velocity, m.energies, m.occupations, np.asarray([0.0 + 0.0j]),
        mesh=mesh, nb_logical=m.nb, cell_volume=m.cell_volume,
        nk_tot=m.nk, nspin=m.nspin, nspinor=m.nspinor))[0]
    out["S00"] = (_rel(S_code, S_ref), S_ref)
    out["S00_flipped"] = (_rel(S_code, -S_ref), None)

    # --- wings: ONE charge leg, one explicit minus --------------------------
    b_mu = _centroid_vertex(m.psi)
    Y_ref = m.cell_volume * adler_wiser_chi(
        D_jet, b_mu, m.energies, m.occupations,
        cell_volume=m.cell_volume, nk_tot=m.nk, nspin=m.nspin,
        nspinor=m.nspinor)
    Z_ref = m.cell_volume * adler_wiser_chi(
        b_mu, D_jet, m.energies, m.occupations,
        cell_volume=m.cell_volume, nk_tot=m.nk, nspin=m.nspin,
        nspinor=m.nspinor)
    stub = SimpleNamespace(psi_xn=jnp.asarray(m.psi), psi_yn=jnp.asarray(m.psi))
    Y_code, Z_code = head_wings_sharded(
        m.velocity, stub, m.energies, m.occupations,
        np.asarray([0.0 + 0.0j]), mesh=mesh, nb_logical=m.nb, nk_tot=m.nk,
        nspin=m.nspin, nspinor=m.nspinor)
    Y_code = np.asarray(Y_code)[0]           # (3, nmu)
    Z_code = np.asarray(Z_code)[0]           # (nmu, 3)
    out["wing_Y"] = (_rel(Y_code, Y_ref), Y_ref)
    out["wing_Z"] = (_rel(Z_code, Z_ref), Z_ref)
    out["wing_Y_flipped"] = (_rel(Y_code, -Y_ref), None)
    out["wing_Z_flipped"] = (_rel(Z_code, -Z_ref), None)
    out["wing_reciprocity"] = (_rel(Z_code, np.conj(Y_code).T), None)

    # --- Hall CT: one charge leg (v), one current leg (Gamma) ---------------
    from common.bispinor_init import HALFALPHA
    gamma_raw = HALFALPHA * np.transpose(m.velocity, (1, 0, 2, 3))
    hall_frequencies = np.asarray(
        [0.0 + 0.0j, 0.0 + 0.7j, 0.0 - 0.7j, 0.0 + 1.3j])
    sigma_samples = np.asarray(raw_hall_pseudovector_sharded(
        gamma_raw, m.energies, m.occupations, hall_frequencies,
        mesh=mesh, nb_logical=m.nb,
        cell_volume=m.cell_volume, nk_tot=m.nk, nspin=m.nspin,
        nspinor_wfn=m.nspinor))
    sigma_H = sigma_samples[0].real
    gamma_dir = np.transpose(gamma_raw, (1, 0, 2, 3))     # (3,nk,nb,nb)
    chi_0i = adler_wiser_chi(
        D_jet[:2], gamma_dir, m.energies, m.occupations,
        cell_volume=m.cell_volume, nk_tot=m.nk, nspin=m.nspin,
        nspinor=m.nspinor)                                 # (2,3), a x i
    CT_code = np.asarray(static_hall_linear_response(
        sigma_H, dimension=2))[:, 0, 1:]
    out["hall_CT_imag"] = (_rel(CT_code.imag, chi_0i.imag), chi_0i.imag)
    out["hall_CT_imag_flipped"] = (_rel(CT_code.imag, -chi_0i.imag), None)
    out["hall_TC_is_CT_dagger"] = (
        _rel(np.asarray(static_hall_linear_response(
            sigma_H, dimension=2))[:, 1:, 0],
             np.conj(CT_code)), None)

    basis = np.eye(3)
    epsilon = np.asarray([
        [[np.dot(basis[b], np.cross(basis[a], basis[i]))
          for i in range(3)] for a in range(3)] for b in range(3)])
    sigma_reference = []
    for z in hall_frequencies:
        chi_dynamic = adler_wiser_chi(
            D_jet, gamma_dir, m.energies, m.occupations,
            cell_volume=m.cell_volume, nk_tot=m.nk, nspin=m.nspin,
            nspinor=m.nspinor, z=z)
        sigma_reference.append(
            0.5j * np.einsum("bai,ai->b", epsilon, chi_dynamic))
    sigma_reference = np.asarray(sigma_reference)
    out["hall_dynamic_kubo"] = (
        _rel(sigma_samples, sigma_reference), sigma_reference)
    out["hall_dynamic_kubo_flipped"] = (
        _rel(sigma_samples, -sigma_reference), None)
    out["hall_dynamic_even"] = (
        _rel(sigma_samples[1], sigma_samples[2]), None)
    out["hall_dynamic_imag_axis_reality"] = (
        float(np.max(np.abs(sigma_samples.imag))), None)

    sigma_reversed = np.asarray(raw_hall_pseudovector_sharded(
        np.conj(gamma_raw), m.energies, m.occupations, hall_frequencies,
        mesh=mesh, nb_logical=m.nb, cell_volume=m.cell_volume,
        nk_tot=m.nk, nspin=m.nspin, nspinor_wfn=m.nspinor))
    out["hall_dynamic_magnetisation_odd"] = (
        _rel(sigma_reversed, -sigma_samples), None)
    gamma_trs = np.concatenate((gamma_raw, -np.conj(gamma_raw)), axis=0)
    energies_trs = np.concatenate((m.energies, m.energies), axis=0)
    occupations_trs = np.concatenate(
        (m.occupations, m.occupations), axis=0)
    sigma_trs = np.asarray(raw_hall_pseudovector_sharded(
        gamma_trs, energies_trs, occupations_trs, hall_frequencies,
        mesh=mesh, nb_logical=m.nb, cell_volume=m.cell_volume,
        nk_tot=2 * m.nk, nspin=m.nspin, nspinor_wfn=m.nspinor))
    out["hall_dynamic_trs_twin"] = (
        float(np.max(np.abs(sigma_trs))), None)
    # Reduction last bits are backend-specific.  These are the pre-extension
    # z=0 bytes captured on the certified CPU and CUDA paths; selecting the
    # old occupied-Berry row above must preserve the appropriate literal.
    frozen_static = {
        "cpu": "358bf00049d5823ff158d907361e8f3fec79838d1a3296bf",
        "gpu": "358bf00049d5823ff158d907361e8f3ff379838d1a3296bf",
    }
    platform = jax.devices()[0].platform
    out["hall_static_frozen_bits"] = (
        0.0 if sigma_samples[0].real.tobytes().hex()
        == frozen_static.get(platform) else 1.0,
        None,
    )
    return out


def _rel(got, want):
    got = np.asarray(got)
    want = np.asarray(want)
    scale = max(float(np.max(np.abs(want))), float(np.max(np.abs(got))),
                1.0e-300)
    return float(np.max(np.abs(got - want))) / scale


# --------------------------------------------------------------------------
# Part B -- the merged completion, every block nonzero.
# --------------------------------------------------------------------------

_ALAT = 5.97
_CLAT = 30.0
_KGRID = (3, 3, 1)


def _geometry():
    import vcoul
    a1 = _ALAT * np.array([1.0, 0.0, 0.0])
    a2 = _ALAT * np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0])
    a3 = np.array([0.0, 0.0, _CLAT])
    A = np.stack((a1, a2, a3), axis=0)
    cell_volume = float(abs(np.linalg.det(A)))
    bvec = 2.0 * np.pi * np.linalg.inv(A).T           # ROWS are b1,b2,b3
    return vcoul.CoulombGeometry(bvec=bvec, cell_volume=cell_volume)


def _receipt():
    import vcoul
    return vcoul.slab_minibz_photon_cubature(
        vcoul.get_kernel(2), _geometry(), _KGRID)


def _bulk_receipt(*, hexagonal):
    """Production bulk receipt for cubic or elongated hexagonal mini-cell."""
    import vcoul
    if hexagonal:
        bvec = np.asarray((
            (1.0, 0.0, 0.0),
            (0.5, np.sqrt(3.0) / 2.0, 0.0),
            (0.0, 0.0, 1.0 / 3.0),
        ))
    else:
        bvec = np.eye(3)
    cell_volume = float((2.0 * np.pi) ** 3 / abs(np.linalg.det(bvec)))
    geometry = vcoul.CoulombGeometry(
        bvec=bvec, cell_volume=cell_volume)
    return vcoul.minibz_photon_cubature(
        vcoul.get_kernel(3), geometry, (3, 3, 3))


def _fixture(mesh, *, sigma_H, dimension=2, wings=True, seed=20260901):
    """Sealed response + packed body + Gamma vectors, everything nonzero."""
    rng = np.random.default_rng(seed)
    layout = PhotonBasisLayout.from_centroid_extents(4, 3, mesh)
    n_packed = layout.packed_extent
    charge_extent = layout.padded_extent(0)

    # Magnitudes are physical-scale, not decorative: the slab kernel has
    # v(q) -> 8*pi*zc/q, so the charge screening strength is 8*pi*f2d*S and
    # the Hall strength is 8*pi*zc*sigma_H.  Both are kept O(0.1) so the
    # provider's fixed 16/24/32 polygon ladder converges inside its own
    # GATE static_photon_polygon_not_converged budget.
    charge_S = np.diag(
        np.asarray((-1.2e-2, -8.0e-3, -5.0e-3)[:dimension]))
    if dimension >= 2:
        charge_S[0, 1] = charge_S[1, 0] = 3.0e-3
    S_host = np.zeros((1, 3, 3), dtype=np.complex128)
    S_host[0, :dimension, :dimension] = charge_S

    Y_host = np.zeros((1, 3, charge_extent), dtype=np.complex128)
    if wings:
        wing = (rng.normal(size=(dimension, charge_extent))
                + 1j * rng.normal(size=(dimension, charge_extent))) * 1.0
        Y_host[0, :dimension, :] = wing
    Z_host = np.conj(np.transpose(Y_host, (0, 2, 1))).copy()

    direct = SimpleNamespace(
        S_direct=_put(S_host, mesh, P()),
        Y_x=_put(Y_host, mesh, P(None, None, "x")),
        Z_y=_put(Z_host, mesh, P(None, "y", None)))

    class _Hall:
        pass

    saved_direct = qsgw_head.build_dft_head_response
    saved_hall = qsgw_head.StaticGaugeHallTransaction
    saved_fp = parallel_transport.wfn_fingerprint
    try:
        qsgw_head.build_dft_head_response = lambda *a, **k: direct
        qsgw_head.StaticGaugeHallTransaction = _Hall
        parallel_transport.wfn_fingerprint = lambda _w: _WFN_SHA
        hall = None
        if np.any(np.asarray(sigma_H) != 0.0):
            hall = _Hall()
            hall.sigma_H = _put(np.asarray(sigma_H, dtype=np.float64),
                                mesh, P())
            hall.sigma_H_at = lambda _frequency: hall.sigma_H
            hall.wfn_fingerprint = _WFN_SHA
            hall.band_start, hall.band_stop = 0, 4
            hall.producer_id = "oracle-fixture"
            hall.hamiltonian_config_operator_fingerprint = "sha256:" + "d" * 64
        from gw.static_gauge_response import build_static_photon_head_response
        response = build_static_photon_head_response(
            None,
            input_dir="/bounded/not-read", mesh=mesh,
            wfn=SimpleNamespace(nspin=1),
            meta=SimpleNamespace(b_id_0=0, b_id_4_chi_user=4),
            config=SimpleNamespace(sys_dim=dimension),
            layout=layout, hall_transaction=hall)
    finally:
        qsgw_head.build_dft_head_response = saved_direct
        qsgw_head.StaticGaugeHallTransaction = saved_hall
        parallel_transport.wfn_fingerprint = saved_fp

    # The packed body is padded-consistent: internal channel pad rows and
    # columns are EXACT ZERO, as every production packer leaves them.  The
    # q=0 update masks those rows structurally
    # (`photon_layout._q0_local_factor_piece`); with a physical body the mask
    # is a no-op, and a fixture that fills the pads would be comparing the
    # masked code against unmasked algebra rather than testing a sign.
    live = _logical_mask(layout, mesh)
    body = (rng.normal(size=(n_packed, n_packed))
            + 1j * rng.normal(size=(n_packed, n_packed))) * 0.05
    body = 0.5 * (body + np.conj(body.T)) + 0.4 * np.eye(n_packed)
    body = body * live[:, None] * live[None, :]
    W_host = np.zeros((1, n_packed, n_packed), dtype=np.complex128)
    W_host[0] = body
    V_host = np.zeros((1, n_packed, n_packed), dtype=np.complex128)

    # ONE set of literal-Gamma channel vectors, packed twice with the two
    # shardings -- exactly what `w_isdf.compute_static_photon_response` does
    # (`w_isdf.py:2272-2278`).  Hermiticity of the completed body DEPENDS on
    # g0_Y carrying the same values as g0_X; two independent draws break it.
    g0_hosts = []
    for row in range(4):
        width = layout.padded_extent(row)
        logical = layout.logical_extent(row)
        vec = np.zeros((1, width), dtype=np.complex128)
        vec[0, :logical] = (rng.normal(size=logical)
                            + 1j * rng.normal(size=logical)) * 0.5
        g0_hosts.append(vec)

    def _g0(axis):
        spec = P(None, "x") if axis == "x" else P(None, "y")
        return pack_photon_channel_vectors(
            tuple(_put(v, mesh, spec) for v in g0_hosts),
            layout, mesh, axis_name=axis)[0]

    return SimpleNamespace(
        layout=layout, response=response, V_host=V_host, W_host=W_host,
        g0_X=_g0("x"), g0_Y=_g0("y"))


def _logical_mask(layout, mesh):
    """1.0 on every LOGICAL packed slot, 0.0 on internal channel padding.

    Read off the canonical packer with per-channel one vectors rather than
    recomputing the interleaving, which is not a closed-form offset.
    """
    ones = []
    for row in range(4):
        vec = np.zeros((1, layout.padded_extent(row)), dtype=np.complex128)
        vec[0, :layout.logical_extent(row)] = 1.0
        ones.append(_put(vec, mesh, P(None, "x")))
    packed = _gather(pack_photon_channel_vectors(
        tuple(ones), layout, mesh, axis_name="x")[0])
    mask = np.sum(np.abs(packed), axis=0)
    if not np.array_equal(np.unique(mask), np.array([0.0, 1.0])) and \
            not np.array_equal(np.unique(mask), np.array([1.0])):
        raise AssertionError(f"packed logical mask is not 0/1: {mask}")
    return mask


def _put(value, mesh, spec):
    return jax.device_put(np.asarray(value), NamedSharding(mesh, spec))


def _gather(value):
    return np.asarray(jax.device_get(value))


def _packed_pair(fixture, mesh):
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    return (jax.device_put(fixture.V_host, sharding),
            jax.device_put(fixture.W_host, sharding))


def _block_diagonal_body(fixture, mesh):
    """Bare-transverse body: scalar W00 plus diagonal bare current blocks."""
    layout = fixture.layout
    rng = np.random.default_rng(72)
    blocks = {}
    for channel in range(4):
        width = layout.padded_extent(channel)
        if channel == 0:
            raw = (rng.normal(size=(width, width))
                   + 1j * rng.normal(size=(width, width)))
            block = 0.03 * (raw + np.conj(raw.T)) + 0.7 * np.eye(width)
        else:
            block = (0.15 + 0.01 * channel) * np.eye(
                width, dtype=np.complex128)
        blocks[channel, channel] = _put(
            np.asarray(block, dtype=np.complex128)[None], mesh,
            P(None, "x", "y"))
    return pack_photon_operator(
        lambda A, B: blocks.get((A, B)), 1, layout, mesh)


def _cttc_dense_vector(pairs, layout, mesh):
    """Dense CT/TC block vector for a test-sized factor carrier."""
    blocks = []
    for A, B in ((0, 1), (0, 2), (0, 3),
                 (1, 0), (2, 0), (3, 0)):
        block = np.asarray(photon_q0_low_rank_block(
            pairs, layout, A, B, mesh))[0]
        blocks.append(block.reshape(-1))
    return np.concatenate(blocks)


# ---- the independent NumPy reference ------------------------------------

def _levi_civita():
    eps = np.zeros((3, 3, 3))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[i, j, k] = 1.0
        eps[i, k, j] = -1.0
    return eps


def hall_response_reference(sigma_H, *, dimension=2, ct_sign=-1.0):
    """``chi_0i^{(a)} = ct_sign * i * eps[b,a,i] sigma_b``; TC = CT^dagger."""
    eps = _levi_civita()
    sigma = np.asarray(sigma_H, dtype=np.float64)
    H = np.zeros((dimension, 4, 4), dtype=np.complex128)
    for a in range(dimension):
        for i in range(3):
            value = ct_sign * 1j * float(
                sum(eps[b, a, i] * sigma[b] for b in range(3)))
            H[a, 0, i + 1] = value
            H[a, i + 1, 0] = np.conj(value)
    return H


def schur_folded_response(response, W_gamma, cell_volume, *, wing_sign=1.0):
    """``S_eff[a,b] = S[a,b] + wing_sign * Y_a W Z_b / V``, a<->b symmetric."""
    S = _gather(response.S_direct).copy()
    Y = _gather(response.Y_x)
    Z = _gather(response.Z_y)
    out = np.array(S, dtype=np.complex128)
    dimension = int(response.dimension)
    for a in range(dimension):
        for b in range(dimension):
            out[a, b] = S[a, b] + wing_sign * (
                Y[a] @ W_gamma @ Z[b]) / cell_volume
    return 0.5 * (out + np.swapaxes(out, 0, 1))


def head_propagator_bruteforce(q, D, R, *, tt_sign=-1.0):
    """``W_h = Pi (Dred^{-1} - Pi^dag R Pi)^{-1} Pi^dag`` on ``ker(D)^perp``.

    ``D`` is singular for a slab (the in-plane longitudinal current has no
    Coulomb-gauge propagator), so the literal ``D^{-1}`` of the convention
    exists only after projecting that direction out.  The projected inverse
    is exact: ``(I-DR)^{-1}D`` has that direction as an exact null row AND
    column.  ``tt_sign`` exists only for the negative control.
    """
    q = np.asarray(q, dtype=np.float64)
    qn = q / np.linalg.norm(q)
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ qn)))]
    t1 = np.cross(reference, qn)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(qn, t1)
    Pi = np.zeros((4, 3), dtype=np.complex128)
    Pi[0, 0] = 1.0
    Pi[1:, 1] = t1
    Pi[1:, 2] = t2
    v = float(np.real(D[0, 0]))
    D_red = np.diag(np.array([v, tt_sign * v, tt_sign * v],
                             dtype=np.complex128))
    R_red = np.conj(Pi.T) @ R @ Pi
    W_red = np.linalg.inv(np.linalg.inv(D_red) - R_red)
    return Pi @ W_red @ np.conj(Pi.T)


def reference_moments(receipt, S_eff, sigma_H, *, ct_sign=-1.0, tt_sign=-1.0):
    """``<b_u W_h b_v>`` and ``<D>`` on the receipt's final rule."""
    dimension = int(receipt.dimension)
    H = hall_response_reference(
        sigma_H, dimension=dimension, ct_sign=ct_sign)
    chunk = receipt.chunks[-1]
    n = int(chunk.physical_count)
    q = np.asarray(chunk.q_cart[:n], dtype=np.float64)
    D = np.asarray(chunk.D_raw[:n], dtype=np.complex128)
    w = np.asarray(chunk.sample_weight[:n], dtype=np.float64)
    measure = float(np.sum(w))
    basis_size = 1 + dimension
    moments = np.zeros(
        (basis_size, basis_size, 4, 4), dtype=np.complex128)
    D_mean = np.zeros((4, 4), dtype=np.complex128)
    for s in range(n):
        R = (np.einsum("a,aij->ij", q[s, :dimension], H)
             + np.einsum(
                 "a,b,abij->ij", q[s, :dimension], q[s, :dimension],
                 S_eff))
        W_h = head_propagator_bruteforce(q[s], D[s], R, tt_sign=tt_sign)
        basis = np.concatenate(([1.0], q[s, :dimension]))
        moments += w[s] * np.einsum("u,ij,v->uvij", basis, W_h, basis)
        D_mean += w[s] * D[s]
    return moments / measure, D_mean / measure


def reference_insertion(fixture, moments, D_mean, cell_volume):
    """The rank-4 reattachment written straight from the convention."""
    W0 = fixture.W_host[0]
    Y = _gather(fixture.response.Y_x)
    Z = _gather(fixture.response.Z_y)
    g0x = _gather(fixture.g0_X)
    g0y = _gather(fixture.g0_Y)
    dimension = int(fixture.response.dimension)
    left = [np.conj(g0x)] + [
        (W0 @ Z[direction]).T for direction in range(dimension)]
    right = [g0y] + [
        Y[direction] @ W0 for direction in range(dimension)]
    V_ref = fixture.V_host[0] + np.einsum(
        "Ai,AB,Bj->ij", np.conj(g0x), D_mean, g0y,
        optimize=True) / cell_volume
    W_ref = np.array(W0, dtype=np.complex128)
    for u in range(1 + dimension):
        for v in range(1 + dimension):
            W_ref = W_ref + np.einsum(
                "Ai,AB,Bj->ij", left[u], moments[u, v], right[v],
                optimize=True) / cell_volume
    return V_ref, W_ref


_BLOCKS = {
    "CC": (slice(0, 1), slice(0, 1)),
    "CT": (slice(0, 1), slice(1, 4)),
    "TC": (slice(1, 4), slice(0, 1)),
    "TT": (slice(1, 4), slice(1, 4)),
}


def run_case(mesh, sigma_H, *, wings=True, seed=20260901,
             ct_sign=-1.0, tt_sign=-1.0, wing_sign=1.0, receipt=None):
    """Run the production completion and the reference; return residuals."""
    receipt = _receipt() if receipt is None else receipt
    cell_volume = float(receipt.cell_volume)
    dimension = int(receipt.dimension)
    fixture = _fixture(
        mesh, sigma_H=sigma_H, dimension=dimension, wings=wings, seed=seed)
    V_in, W_in = _packed_pair(fixture, mesh)
    V_out, W_out, evidence = complete_static_photon_q0(
        V_in, W_in, fixture.response, fixture.g0_X, fixture.g0_Y,
        receipt, mesh_xy=mesh,
        charge_block_state=STATIC_PHOTON_CHARGE_BLOCK_PRESENT)

    S_eff = schur_folded_response(
        fixture.response, fixture.W_host[0], cell_volume, wing_sign=wing_sign)
    moments, D_mean = reference_moments(
        receipt, S_eff, np.asarray(sigma_H, dtype=np.float64),
        ct_sign=ct_sign, tt_sign=tt_sign)
    got_moments = np.asarray(evidence.screened_moments)
    got_D = np.asarray(evidence.bare_D_mean)
    # The inverse oracle owns the screened moments.  The bare V insertion
    # uses the completion's already-certified <D>: re-reducing up to 383328
    # rows in NumPy gives a different tree from XLA and can add ~1e-13 of
    # platform-only summation drift to an otherwise exact reattachment check.
    # ``bare_D`` below remains the independent provider-to-kernel comparison.
    V_ref, W_ref = reference_insertion(
        fixture, moments, got_D, cell_volume)
    residual = {
        "bare_D": _rel(got_D, D_mean),
        "bare_D_flipped": _rel(got_D, -D_mean),
        "bare_D_abs": float(np.max(np.abs(got_D - D_mean))),
        "moments_all": _rel(got_moments, moments),
        "V_packed": _rel(_gather(V_out)[0], V_ref),
        "W_packed": _rel(_gather(W_out)[0], W_ref),
    }
    # Per-block residuals are normalized by the moment tensor's GLOBAL scale.
    # A block-local norm reports ~1 for a block that is zero by symmetry in
    # BOTH the code and the reference (M_0a is odd in q and vanishes on a
    # centrosymmetric polygon when sigma_H = 0), which is a scale artifact,
    # not a disagreement.
    scale = max(float(np.max(np.abs(moments))),
                float(np.max(np.abs(got_moments))), 1.0e-300)
    for name, (rows, cols) in _BLOCKS.items():
        residual[f"moments_{name}"] = float(np.max(np.abs(
            got_moments[:, :, rows, cols]
            - moments[:, :, rows, cols]))) / scale
        residual[f"weight_{name}"] = float(np.max(np.abs(
            moments[:, :, rows, cols]))) / scale
    for u in range(1 + dimension):
        for v in range(1 + dimension):
            residual[f"moment_{u}{v}"] = float(np.max(np.abs(
                got_moments[u, v] - moments[u, v]))) / scale
    W_done = _gather(W_out)[0]
    scale = max(float(np.max(np.abs(W_done))), 1.0e-300)
    residual["W_hermiticity"] = float(
        np.max(np.abs(W_done - np.conj(W_done.T)))) / scale
    return residual, evidence, fixture, moments, D_mean, S_eff


# --------------------------------------------------------------------------
# pytest cells
# --------------------------------------------------------------------------

_SIGMA_PLUS = np.array([0.0, 0.0, 1.5e-4])
_SIGMA_FULL = np.array([4.0e-5, -7.0e-5, 1.5e-4])
_TOL = 2.0e-10


def test_part_a_definitions_share_one_convention():
    out = part_a_residuals(_mesh(1))
    for key in ("S00", "wing_Y", "wing_Z", "hall_CT_imag",
                "hall_TC_is_CT_dagger", "wing_reciprocity",
                "hall_dynamic_kubo", "hall_dynamic_even",
                "hall_dynamic_imag_axis_reality",
                "hall_dynamic_magnetisation_odd",
                "hall_dynamic_trs_twin",
                "hall_static_frozen_bits"):
        assert out[key][0] < 1.0e-12, (key, out[key][0])
    # Negative controls: each sign is observable, so none of these agreements
    # is a tautology of the reference's own spelling.
    for key in ("S00_flipped", "wing_Y_flipped", "wing_Z_flipped",
                "hall_CT_imag_flipped", "hall_dynamic_kubo_flipped"):
        assert out[key][0] > 1.0e-3, (key, out[key][0])


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("sigma", [_SIGMA_PLUS, -_SIGMA_PLUS, _SIGMA_FULL,
                                   -_SIGMA_FULL])
def test_completion_matches_the_independent_reference(sigma, side):
    """Both signs of ``sigma_H``, every block nonzero.

    The ``side=2`` cell SKIPS rather than silently degenerating to 1x1: a
    1x1 mesh canonicalizes ``P(None,'x','y')`` away and would report a
    two-axis pass it never ran (TASTE.md 2026-08-30).  It also adds the
    padded-channel mask, since ``from_centroid_extents(4,3)`` pads the three
    transverse channels from 3 to 4 at divisor 4.
    """
    if side * side > len(jax.devices()):
        pytest.skip(f"{side}x{side} mesh needs {side * side} devices; "
                    f"have {len(jax.devices())}")
    residual, _, _, _, _, _ = run_case(_mesh(side), sigma)
    skip = {"W_hermiticity", "bare_D_flipped", "bare_D_abs"}
    for key, value in residual.items():
        if key in skip or key.startswith("weight_"):
            continue
        assert value < _TOL, (key, value)
    # bare_D agreeing is not a tautology: the opposite sign is observable.
    assert residual["bare_D_flipped"] > 1.0, residual["bare_D_flipped"]


def test_completed_gamma_body_is_hermitian():
    """W(Gamma) stays Hermitian after all ten rank-4 updates.

    This holds only because the two literal-Gamma factors carry the SAME
    values (production packs one ``photon_g0_vectors`` twice,
    ``w_isdf.py:2272-2278``), the two wings obey ``Z_a = conj(Y_a)^T``, and
    every moment ``<b_u W_h b_v>`` is Hermitian.  It is a joint check on the
    (0,a)/(a,0) sign PAIR: flipping one of them alone breaks it.
    """
    residual, _, fixture, _, _, _ = run_case(_mesh(1), _SIGMA_FULL)
    assert residual["W_hermiticity"] < 1.0e-12, residual["W_hermiticity"]
    # ... and the completion actually moved the body, so the invariant is
    # not being read off an untouched Hermitian input.
    _, W_in = _packed_pair(fixture, _mesh(1))
    _, W_out, _ = complete_static_photon_q0(
        *_packed_pair(fixture, _mesh(1)), fixture.response,
        fixture.g0_X, fixture.g0_Y, _receipt(), mesh_xy=_mesh(1),
        charge_block_state=STATIC_PHOTON_CHARGE_BLOCK_PRESENT)
    delta = _gather(W_out)[0] - fixture.W_host[0]
    assert float(np.max(np.abs(delta))) > 1.0e-6


def test_zero_hall_decouples_into_charge_head_plus_bare_transverse():
    receipt = _receipt()
    residual, evidence, _, moments, D_mean, S_eff = run_case(
        _mesh(1), np.zeros(3), receipt=receipt)
    assert residual["moments_all"] < _TOL
    got = np.asarray(evidence.screened_moments)
    scale = float(np.max(np.abs(got)))
    # sigma_H = 0 leaves R charge-only, so D R has one nonzero column and the
    # 4x4 solve is exactly block diagonal.
    assert np.max(np.abs(got[:, :, 0, 1:])) / scale < 1.0e-14
    assert np.max(np.abs(got[:, :, 1:, 0])) / scale < 1.0e-14
    # ... and the TT moment is the BARE transverse average, unscreened.
    bare_tt = _reference_bare_tt_moments(receipt)
    assert _rel(got[:, :, 1:, 1:], bare_tt) < 1.0e-13
    # <D_TT> = -<v P^T>: trace is -2<v> exactly, for any polygon anisotropy.
    D_got = np.asarray(evidence.bare_D_mean)
    assert abs(np.trace(D_got[1:, 1:]) + 2.0 * D_got[0, 0]) <= (
        1.0e-13 * abs(D_got[0, 0]))


def test_absent_charge_completion_inserts_only_bare_D():
    """Dynamic bare routes must not evaluate a hidden charge-head fold."""
    mesh = _mesh(1)
    receipt = _receipt()
    fixture = _fixture(mesh, sigma_H=np.zeros(3))
    V_in, _ = _packed_pair(fixture, mesh)
    V_out, W_out, evidence = complete_static_photon_q0(
        V_in, None, None, fixture.g0_X, fixture.g0_Y, receipt,
        mesh_xy=mesh,
        charge_block_state=STATIC_PHOTON_CHARGE_BLOCK_ABSENT,
        layout=fixture.response.layout)
    _, D_mean = reference_moments(
        receipt, np.zeros((2, 2, 4, 4), dtype=np.complex128), np.zeros(3))
    g0x = _gather(fixture.g0_X)
    g0y = _gather(fixture.g0_Y)
    V_ref = fixture.V_host[0] + np.einsum(
        "Ai,AB,Bj->ij", np.conj(g0x), D_mean, g0y,
        optimize=True) / float(receipt.cell_volume)
    assert W_out is None
    assert evidence.screened_moments is None
    assert evidence.max_backward_residual is None
    assert evidence.charge_block_state == STATIC_PHOTON_CHARGE_BLOCK_ABSENT
    assert len(evidence.q0_factors.screened_pairs) == 1
    assert _rel(np.asarray(evidence.bare_D_mean), D_mean) < _TOL
    assert _rel(_gather(V_out)[0], V_ref) < _TOL


@pytest.mark.parametrize("hexagonal", (False, True))
def test_bulk_completion_matches_subspace_inverse_and_geometry_oracles(
        hexagonal):
    """The one completion owner consumes cubic and elongated-hex bulk rules.

    Hall physics is deliberately not inferred here: ``sigma_H=0`` is the
    bulk default.  That limit must decouple CT/TC exactly, zero wings must
    leave ``S_eff=S``, and the independent ``D^-1-R`` inverse on the
    non-null Coulomb-gauge subspace must reproduce both packed insertions.
    """
    receipt = _bulk_receipt(hexagonal=hexagonal)
    residual, evidence, fixture, _, _, S_eff = run_case(
        _mesh(1), np.zeros(3), wings=False, receipt=receipt)
    for key in ("V_packed", "W_packed"):
        assert residual[key] <= 1.0e-13, (key, residual[key])
    assert residual["moments_all"] <= 3.0e-13
    np.testing.assert_array_equal(
        np.asarray(evidence.screened_moments)[:, :, 0, 1:], 0.0)
    np.testing.assert_array_equal(
        np.asarray(evidence.screened_moments)[:, :, 1:, 0], 0.0)
    np.testing.assert_array_equal(S_eff, _gather(fixture.response.S_direct))
    assert len(evidence.q0_factors.screened_pairs) == 16

    chunk = receipt.chunks[-1]
    physical = int(chunk.physical_count)
    weight = np.asarray(chunk.sample_weight[:physical])
    D = np.asarray(chunk.D_raw[:physical])
    trace_identity = np.sum(weight * (
        np.trace(D[:, 1:, 1:], axis1=1, axis2=2) + 2.0 * D[:, 0, 0]))
    assert abs(trace_identity) / abs(np.sum(weight * D[:, 0, 0])) <= 1.0e-14
    D_mean = np.einsum("s,sij->ij", weight, D)
    if hexagonal:
        np.testing.assert_allclose(
            D_mean[1, 1], D_mean[2, 2], rtol=1.0e-12, atol=1.0e-12)
        assert not np.isclose(
            D_mean[1, 1], D_mean[3, 3], rtol=1.0e-3, atol=0.0)
    else:
        np.testing.assert_allclose(
            D_mean[1:, 1:],
            -(2.0 / 3.0) * D_mean[0, 0] * np.eye(3),
            rtol=1.0e-12, atol=1.0e-12)


def test_zero_wings_reduce_the_folded_response_to_S():
    receipt = _receipt()
    fixture = _fixture(_mesh(1), sigma_H=_SIGMA_FULL, wings=False)
    S_eff = schur_folded_response(
        fixture.response, fixture.W_host[0], float(receipt.cell_volume))
    S_direct = _gather(fixture.response.S_direct)
    assert _rel(S_eff, S_direct) == 0.0
    residual, _, _, _, _, _ = run_case(
        _mesh(1), _SIGMA_FULL, wings=False, receipt=receipt)
    assert residual["moments_all"] < _TOL
    assert residual["W_packed"] < _TOL


def test_dynamic_faraday_rank_four_fit_closes_dense_contour_and_twins():
    """The ordered Hall pole is the Hall-on/off CT/TC completion itself.

    This is the synthetic-magnet end-to-end cell: every contour point runs
    the incumbent coupled 4x4 completion kernel, then its CT/TC rank-four
    factors are materialized only at this tiny oracle boundary.  Production
    keeps the factors.  The ``sigma_H=0`` twin is exact zero, reversing the
    magnet reverses the inserted family, and the two-sample even pole closes
    the dense contour within the toy's deliberately finite Hall nonlinearity.
    """
    mesh = _mesh(2)
    receipt = _receipt()
    sigma_static = np.array([0.0, 0.0, 1.0e-4])
    fixture = _fixture(mesh, sigma_H=sigma_static, wings=True)
    W_packed = _block_diagonal_body(fixture, mesh)
    W_charge_gamma = photon_block_view(
        W_packed, fixture.layout, 0, 0, mesh)[0]

    probe_frequency = 2.0j
    omega_reference = 1.0
    probe_ratio = omega_reference ** 2 / (
        abs(probe_frequency) ** 2 + omega_reference ** 2)
    response_probe = replace(
        fixture.response,
        sigma_H=fixture.response.sigma_H * probe_ratio,
        frequency_ry=probe_frequency)
    fitted = fit_dynamic_photon_cttc_q0(
        fixture.response, response_probe,
        W_resident_body_gamma=W_packed[0],
        W_resident_charge_gamma=W_charge_gamma,
        W_static_charge_gamma=W_charge_gamma,
        W_probe_charge_gamma=W_charge_gamma,
        g0_X=fixture.g0_X, g0_Y=fixture.g0_Y,
        cubature_receipt=receipt, mesh_xy=mesh)

    # This toy's completed probe is Hermitian, so its ordered D cancels even
    # though production retains the four-factor representation.
    assert len(fitted.D_pairs) == 4
    assert fitted.odd_even_residue_ratio < 1.0e-12
    assert fitted.probe_fit_relative_error < 1.0e-4
    B_dense = _cttc_dense_vector(
        fitted.B_pairs, fixture.layout, mesh)
    D_dense = _cttc_dense_vector(
        fitted.D_pairs, fixture.layout, mesh)
    static_dense = _cttc_dense_vector(
        fitted.static_pairs, fixture.layout, mesh)
    probe_dense = _cttc_dense_vector(
        fitted.probe_pairs, fixture.layout, mesh)
    np.testing.assert_allclose(
        -2.0 * B_dense / fitted.omega_h_ry, static_dense,
        rtol=0.0, atol=2.0e-13 * np.max(np.abs(static_dense)))
    fitted_probe = (
        (2.0 * fitted.omega_h_ry * B_dense
         + 2.0 * probe_frequency * D_dense)
        / (probe_frequency ** 2 - fitted.omega_h_ry ** 2))
    assert _rel(fitted_probe, probe_dense) < 1.0e-4

    # A legitimate factor gauge changes L and R by inverse phases.  It leaves
    # both the packed operator and its complex overlap invariant; therefore it
    # cannot be used to rotate away a physical anti-Hermitian probe component.
    rng = np.random.default_rng(903)
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, len(fitted.probe_pairs)))
    gauged_probe = tuple(
        (phase * left, right / phase)
        for phase, (left, right) in zip(phases, fitted.probe_pairs))
    (gauge_omega, gauge_B, gauge_D, gauge_error, gauge_ratio,
     gauge_overlaps) = _fit_ordered_faraday_factor_samples(
         fitted.static_pairs, gauged_probe, probe_frequency,
         mesh_xy=mesh, print_fn=lambda _message: None)
    np.testing.assert_allclose(gauge_omega, fitted.omega_h_ry, rtol=2e-14)
    np.testing.assert_allclose(gauge_error, fitted.probe_fit_relative_error,
                               rtol=0.0, atol=5e-11)
    assert gauge_ratio < 64.0 * np.sqrt(np.finfo(np.float64).eps)
    np.testing.assert_allclose(gauge_overlaps, fitted.raw_pair_overlaps,
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        _cttc_dense_vector(gauge_B, fixture.layout, mesh), B_dense,
        rtol=2e-14, atol=2e-14 * np.max(np.abs(B_dense)))
    np.testing.assert_allclose(
        _cttc_dense_vector(gauge_D, fixture.layout, mesh), D_dense,
        rtol=0.0, atol=2e-13 * np.max(np.abs(B_dense)))

    # A physical one-sided phase is different from the inverse factor gauge:
    # it makes the completed probe non-Hermitian.  The ordered split retains
    # it as D_H and closes the complex sample without weakening the real-
    # overlap gate on the Hermitian half.
    ordered_ratio = 0.35
    ordered_anti = 0.07
    ordered_probe = tuple(
        ((ordered_ratio + 1j * ordered_anti) * left, right)
        for left, right in fitted.static_pairs)
    (ordered_omega, ordered_B, ordered_D, ordered_error,
     ordered_D_over_B, ordered_overlaps) = (
        _fit_ordered_faraday_factor_samples(
            fitted.static_pairs, ordered_probe, probe_frequency,
            mesh_xy=mesh, print_fn=lambda _message: None))
    assert abs(sum(ordered_overlaps).imag) > 1.0e-3 * abs(
        sum(ordered_overlaps).real)
    assert ordered_D_over_B > 1.0e-3
    # Gram-norm subtraction resolves an exactly collinear residual only to
    # O(sqrt(eps)); the material adequacy gate is two orders looser at 1e-4.
    assert ordered_error < 64.0 * np.sqrt(np.finfo(np.float64).eps)
    ordered_B_dense = _cttc_dense_vector(
        ordered_B, fixture.layout, mesh)
    ordered_D_dense = _cttc_dense_vector(
        ordered_D, fixture.layout, mesh)
    ordered_probe_dense = _cttc_dense_vector(
        ordered_probe, fixture.layout, mesh)
    ordered_model = (
        (2.0 * ordered_omega * ordered_B_dense
         + 2.0 * probe_frequency * ordered_D_dense)
        / (probe_frequency ** 2 - ordered_omega ** 2))
    np.testing.assert_allclose(
        ordered_model, ordered_probe_dense, rtol=2e-12,
        atol=2e-12 * np.max(np.abs(ordered_probe_dense)))

    # Negative control: two independently scaled Hermitian sectors change
    # the probe operator's direction.  The complex-overlap gate is green and
    # D is exactly zero, but a shared even pole cannot represent the sample;
    # two frequencies do not identify the required second even pole.
    static_ct = fitted.static_pairs[0]
    deformed_ct = (
        0.4 * static_ct[0] + 0.1 * jnp.roll(static_ct[0], 1, axis=0),
        static_ct[1],
    )
    unrepresented_even_probe = (
        deformed_ct,) + _adjoint_factor_family((deformed_ct,), mesh)
    with pytest.raises(
            NotImplementedError,
            match="dynamic_hall_head_unimplemented_second_even_pole"):
        _fit_ordered_faraday_factor_samples(
            fitted.static_pairs, unrepresented_even_probe, probe_frequency,
            mesh_xy=mesh, print_fn=lambda _message: None)

    static_S, static_YW, static_WZ = _dynamic_photon_head_small_system(
        fixture.response, W_packed[0],
        W_resident_charge_gamma=W_charge_gamma,
        W_target_charge_gamma=W_charge_gamma,
        cell_volume=float(receipt.cell_volume), mesh_xy=mesh)
    off_moments = _dynamic_photon_head_moments(
        np.zeros(3, dtype=np.float64), static_S, receipt)

    def direct_pairs(scale, frequency):
        response = replace(
            fixture.response,
            sigma_H=fixture.response.sigma_H * scale,
            frequency_ry=frequency)
        moments = _dynamic_photon_head_moments(
            response.sigma_H, static_S, receipt)
        return _faraday_cttc_factor_pairs(
            moments - off_moments,
            g0_X=fixture.g0_X, g0_Y=fixture.g0_Y,
            YW_y=static_YW, WZ_x=static_WZ,
            layout=fixture.layout, mesh_xy=mesh,
            cell_volume=float(receipt.cell_volume))

    contour_error = 0.0
    for y in (0.0, 0.5, 1.0, 2.0, 4.0):
        z = 1j * y
        ratio = omega_reference ** 2 / (y * y + omega_reference ** 2)
        direct = _cttc_dense_vector(
            direct_pairs(ratio, z), fixture.layout, mesh)
        model = ((2.0 * fitted.omega_h_ry * B_dense + 2.0 * z * D_dense)
                 / (z * z - fitted.omega_h_ry ** 2))
        contour_error = max(contour_error, _rel(model, direct))
    assert contour_error < 1.0e-3

    zero = _cttc_dense_vector(
        direct_pairs(0.0, 0.7j), fixture.layout, mesh)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))
    plus = _cttc_dense_vector(
        direct_pairs(0.4, 0.7j), fixture.layout, mesh)
    minus = _cttc_dense_vector(
        direct_pairs(-0.4, 0.7j), fixture.layout, mesh)
    np.testing.assert_allclose(
        minus, -plus, rtol=0.0, atol=2.0e-13 * np.max(np.abs(plus)))


def test_dynamic_faraday_probe_charge_delta_matches_explicit_packed_toy():
    """O(N) probe folding equals replacing CC in a tiny packed body."""
    mesh = _mesh(2)
    receipt = _receipt()
    fixture = _fixture(
        mesh, sigma_H=np.array([0.0, 0.0, 2.0e-4]), wings=True)
    _, W_packed = _packed_pair(fixture, mesh)
    W_static = W_packed[0]
    W_static_charge = photon_block_view(
        W_packed, fixture.layout, 0, 0, mesh)[0]
    target_charge = 0.73 * W_static_charge
    got_S, got_YW, got_WZ = _dynamic_photon_head_small_system(
        fixture.response, W_static,
        W_resident_charge_gamma=W_static_charge,
        W_target_charge_gamma=target_charge,
        cell_volume=float(receipt.cell_volume), mesh_xy=mesh)

    # Materializing the hybrid body is allowed only at this tiny oracle
    # boundary. Production proves the same identity with the O(N) delta fold.
    explicit_packed = pack_photon_operator(
        lambda A, B: (
            target_charge[None]
            if (A, B) == (0, 0) else
            photon_block_view(W_packed, fixture.layout, A, B, mesh)),
        1, fixture.layout, mesh)
    explicit = explicit_packed[0]
    want_S = fixture.response.S_direct
    for a in range(2):
        for b in range(2):
            folded = fold_small_head_wings_sharded(
                fixture.response.S_direct[a, b], fixture.response.Y_x[a],
                explicit, fixture.response.Z_y[b], float(receipt.cell_volume),
                mesh_xy=mesh)
            want_S = want_S.at[a, b].set(folded)
    want_S = canonicalize_static_gauge_q2_tensor(want_S)
    want_YW, want_WZ = small_head_wing_halves_sharded(
        fixture.response.Y_x, explicit, fixture.response.Z_y, mesh_xy=mesh)
    np.testing.assert_allclose(got_S, want_S, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(got_YW, want_YW, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(got_WZ, want_WZ, rtol=0.0, atol=2.0e-14)


def test_bulk_faraday_factor_family_consumes_the_third_direction():
    """The dimension-general adapter must not truncate the bulk z moments."""
    mesh = _mesh(1)
    fixture = _fixture(
        mesh, sigma_H=np.array([1.0e-4, -2.0e-4, 3.0e-4]),
        dimension=3, wings=True)
    W_packed = _block_diagonal_body(fixture, mesh)
    YW_y, WZ_x = small_head_wing_halves_sharded(
        fixture.response.Y_x, W_packed[0], fixture.response.Z_y,
        mesh_xy=mesh)
    delta = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    delta[3, 0, 0, 1] = 0.4j
    delta[0, 3, 1, 0] = -0.2j
    pairs = _faraday_cttc_factor_pairs(
        delta, g0_X=fixture.g0_X, g0_Y=fixture.g0_Y,
        YW_y=YW_y, WZ_x=WZ_x, layout=fixture.layout,
        mesh_xy=mesh, cell_volume=5.0)
    dense = _cttc_dense_vector(pairs, fixture.layout, mesh)
    assert np.max(np.abs(dense)) > 1.0e-8

    zero_pairs = _faraday_cttc_factor_pairs(
        np.zeros_like(delta), g0_X=fixture.g0_X, g0_Y=fixture.g0_Y,
        YW_y=YW_y, WZ_x=WZ_x, layout=fixture.layout,
        mesh_xy=mesh, cell_volume=5.0)
    np.testing.assert_array_equal(
        _cttc_dense_vector(zero_pairs, fixture.layout, mesh),
        np.zeros_like(dense))


def _reference_bare_tt_moments(receipt):
    chunk = receipt.chunks[-1]
    n = int(chunk.physical_count)
    q = np.asarray(chunk.q_cart[:n], dtype=np.float64)
    D = np.asarray(chunk.D_raw[:n], dtype=np.complex128)
    w = np.asarray(chunk.sample_weight[:n], dtype=np.float64)
    basis = np.column_stack((np.ones(n), q[:, :receipt.dimension]))
    return np.einsum("s,su,sij,sv->uvij", w, basis, D[:, 1:, 1:], basis,
                     optimize=True) / float(np.sum(w))


@pytest.mark.parametrize("control", ["ct_sign", "tt_sign", "wing_sign"])
def test_negative_controls_are_caught(control):
    """A check that cannot fail is not evidence (TASTE.md, 2026-08-06)."""
    kwargs = {"ct_sign": -1.0, "tt_sign": -1.0, "wing_sign": 1.0}
    kwargs[control] = -kwargs[control] if control != "wing_sign" else 0.0
    residual, _, _, _, _, _ = run_case(_mesh(1), _SIGMA_FULL, **kwargs)
    assert residual["moments_all"] > 1.0e-6, (control, residual)
    assert residual["W_packed"] > 1.0e-8, (control, residual)


def test_schur_fold_and_rank4_reattachment_are_one_block_inverse_identity():
    r"""Where the fold's PLUS and the four insertion families come from.

    Nothing here touches lorrax: it is the block inverse of
    ``eps = 1 - D chi`` in an explicit head(+)body basis, which is the only
    statement that fixes

      * ``S_eff = chi_hh + chi_hb W0_bb chi_bh``  -- PLUS, and dressed by the
        HEADLESS body ``W0_bb``, not by the bare ``v_b`` and not by the
        completed ``W``;
      * the four q=0 update families as the four cross terms of ONE rank-4
        factorization ``(L + W0 chi_bh) W_h (R + chi_hb W0)`` -- so the same
        screened head ``W_h`` sits in all four and their relative signs are
        not independent choices.

    A minus in the fold, or the bare ``v_b`` in place of ``W0_bb``, fails.
    """
    rng = np.random.default_rng(4242)
    nh, nb = 4, 7

    def _herm(n):
        m = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return 0.5 * (m + np.conj(m.T)) * 0.3

    chi_hh = _herm(nh)
    chi_bb = _herm(nb)
    chi_hb = (rng.normal(size=(nh, nb)) + 1j * rng.normal(size=(nh, nb))) * 0.2
    chi_bh = np.conj(chi_hb.T)
    D_h = np.diag(rng.uniform(0.5, 1.5, nh)).astype(np.complex128)
    v_b = np.diag(rng.uniform(0.5, 1.5, nb)).astype(np.complex128)

    D = np.zeros((nh + nb, nh + nb), dtype=np.complex128)
    D[:nh, :nh] = D_h
    D[nh:, nh:] = v_b
    chi = np.block([[chi_hh, chi_hb], [chi_bh, chi_bb]])
    W = np.linalg.solve(np.eye(nh + nb) - D @ chi, D)

    W0_bb = np.linalg.solve(np.eye(nb) - v_b @ chi_bb, v_b)
    S_eff = chi_hh + chi_hb @ W0_bb @ chi_bh
    W_h = np.linalg.inv(np.linalg.inv(D_h) - S_eff)

    np.testing.assert_allclose(W[:nh, :nh], W_h, rtol=0, atol=1.0e-12)
    np.testing.assert_allclose(
        W[:nh, nh:], W_h @ chi_hb @ W0_bb, rtol=0, atol=1.0e-12)
    np.testing.assert_allclose(
        W[nh:, :nh], W0_bb @ chi_bh @ W_h, rtol=0, atol=1.0e-12)
    np.testing.assert_allclose(
        W[nh:, nh:], W0_bb + W0_bb @ chi_bh @ W_h @ chi_hb @ W0_bb,
        rtol=0, atol=1.0e-12)

    # Negative controls: each alternative the audit had to rule out.
    minus = np.linalg.inv(
        np.linalg.inv(D_h) - (chi_hh - chi_hb @ W0_bb @ chi_bh))
    assert np.max(np.abs(minus - W[:nh, :nh])) > 1.0e-3
    bare = np.linalg.inv(
        np.linalg.inv(D_h) - (chi_hh + chi_hb @ v_b @ chi_bh))
    assert np.max(np.abs(bare - W[:nh, :nh])) > 1.0e-3
    unfolded = np.linalg.inv(np.linalg.inv(D_h) - chi_hh)
    assert np.max(np.abs(unfolded - W[:nh, :nh])) > 1.0e-3


# --------------------------------------------------------------------------
# Part C -- exact centroid measure and absolute wing normalisation.
# --------------------------------------------------------------------------

def part_c_exact_measure_residuals(seed=585):
    r"""Close the absolute wing relation in an exact discrete ISDF basis.

    With every grid point selected as a centroid,
    ``zeta_mu(r)=delta_mu,r`` and the forward Fourier convention gives
    ``g0_mu=1/N_r``.  Real-space wavefunctions use the matching FFT
    normalisation ``sum_r |psi_n(r)|^2=N_r``, hence the literal-Gamma
    insertion is exactly one.  A physical real-space body kernel is
    ``Omega`` times its stored LORRAX matrix, while Part A's independently
    evaluated Adler--Wiser wings obey ``Y=Omega*chi_hb`` and
    ``Z=Omega*chi_bh``.  Therefore

    ``(Omega W_code)(Z/Omega) = W_code Z``

    on the left (and analogously on the right), and converting the physical
    completed body back to storage supplies exactly one final ``1/Omega``.
    """
    rng = np.random.default_rng(seed)
    n_r, n_band, n_occ = 7, 4, 2
    cell_volume = 17.25
    energies = np.asarray([[-1.4, -0.5, 0.7, 1.9]], dtype=np.float64)
    occupations = np.zeros_like(energies)
    occupations[:, :n_occ] = 1.0

    psi = (rng.normal(size=(1, 1, n_r, n_band))
           + 1j * rng.normal(size=(1, 1, n_r, n_band)))
    psi *= np.sqrt(
        n_r / np.sum(np.abs(psi) ** 2, axis=(1, 2), keepdims=True))
    g0 = np.full(n_r, 1.0 / n_r, dtype=np.complex128)
    insertion = np.einsum(
        "m,ksmn,ksmn->kn", g0, np.conj(psi), psi, optimize=True)

    raw_v = (rng.normal(size=(2, 1, n_band, n_band))
             + 1j * rng.normal(size=(2, 1, n_band, n_band)))
    velocity = 0.5 * (raw_v + np.conj(np.swapaxes(raw_v, -1, -2)))
    D_jet = density_jet(velocity, energies)
    body_vertex = _centroid_vertex(psi)
    chi_hb = adler_wiser_chi(
        D_jet, body_vertex, energies, occupations,
        cell_volume=cell_volume, nk_tot=1, nspin=1, nspinor=1)
    chi_bh = adler_wiser_chi(
        body_vertex, D_jet, energies, occupations,
        cell_volume=cell_volume, nk_tot=1, nspin=1, nspinor=1)
    Y = cell_volume * chi_hb
    Z = cell_volume * chi_bh

    raw_W = (rng.normal(size=(n_r, n_r))
             + 1j * rng.normal(size=(n_r, n_r)))
    W_code = 0.04 * (raw_W + np.conj(raw_W.T))
    W_physical = cell_volume * W_code
    left_direct = W_physical @ chi_bh
    right_direct = chi_hb @ W_physical
    left_rank4 = W_code @ Z
    right_rank4 = Y @ W_code

    # Scalar charge-head moments for u,v in (0,x,y).  The values are
    # deliberately nonsymmetric so every family is independently nonzero.
    moments = np.asarray(
        [[1.3, -0.17, 0.09],
         [0.21, 0.031, -0.014],
         [-0.08, 0.019, 0.027]], dtype=np.complex128)
    direct_families = {
        "00": moments[0, 0] * np.outer(np.conj(g0), g0) / cell_volume,
        "0a": sum(
            moments[0, a + 1] * np.outer(np.conj(g0), right_direct[a])
            for a in range(2)) / cell_volume,
        "a0": sum(
            moments[a + 1, 0] * np.outer(left_direct[:, a], g0)
            for a in range(2)) / cell_volume,
        "ab": sum(
            moments[a + 1, b + 1]
            * np.outer(left_direct[:, a], right_direct[b])
            for a in range(2) for b in range(2)) / cell_volume,
    }
    rank4_families = {
        "00": moments[0, 0] * np.outer(np.conj(g0), g0) / cell_volume,
        "0a": sum(
            moments[0, a + 1] * np.outer(np.conj(g0), right_rank4[a])
            for a in range(2)) / cell_volume,
        "a0": sum(
            moments[a + 1, 0] * np.outer(left_rank4[:, a], g0)
            for a in range(2)) / cell_volume,
        "ab": sum(
            moments[a + 1, b + 1]
            * np.outer(left_rank4[:, a], right_rank4[b])
            for a in range(2) for b in range(2)) / cell_volume,
    }

    residuals = {
        "insertion": float(np.max(np.abs(insertion - 1.0))),
        "left_attachment": _rel(left_rank4, left_direct),
        "right_attachment": _rel(right_rank4, right_direct),
        # This is claim 585's relation written literally; g0 is nonzero at
        # every point in the exact delta basis.
        "quoted_relation": _rel(
            left_direct / np.conj(g0)[:, None],
            left_rank4 / np.conj(g0)[:, None]),
    }
    for family in ("00", "0a", "a0", "ab"):
        residuals[f"family_{family}"] = _rel(
            rank4_families[family], direct_families[family])

    direct_ab_physical = direct_families["ab"] * cell_volume
    direct_ab_extra_divide = direct_families["ab"] / cell_volume
    residuals["omega_omitted"] = _rel(
        rank4_families["ab"], direct_ab_physical)
    residuals["omega_doubled"] = _rel(
        rank4_families["ab"], direct_ab_extra_divide)
    residuals["omega_omitted_norm_ratio"] = float(
        np.linalg.norm(direct_ab_physical)
        / np.linalg.norm(rank4_families["ab"]))
    residuals["omega_doubled_norm_ratio"] = float(
        np.linalg.norm(direct_ab_extra_divide)
        / np.linalg.norm(rank4_families["ab"]))
    residuals["cell_volume"] = cell_volume
    return residuals


def test_part_c_exact_measure_closes_absolute_wing_normalisation():
    result = part_c_exact_measure_residuals()
    for key in ("insertion", "left_attachment", "right_attachment",
                "quoted_relation", "family_00", "family_0a", "family_a0",
                "family_ab"):
        assert result[key] < 1.0e-12, (key, result[key])
    np.testing.assert_allclose(
        result["omega_omitted_norm_ratio"], result["cell_volume"],
        rtol=1.0e-14, atol=0.0)
    np.testing.assert_allclose(
        result["omega_doubled_norm_ratio"], 1.0 / result["cell_volume"],
        rtol=1.0e-14, atol=0.0)
    assert result["omega_omitted"] > 0.9
    assert result["omega_doubled"] > 0.9


def test_receipt_samples_never_reach_the_singular_origin():
    receipt = _receipt()
    for chunk in receipt.chunks:
        n = int(chunk.physical_count)
        q = np.asarray(chunk.q_cart[:n])
        assert float(np.min(np.linalg.norm(q[:, :2], axis=1))) > 0.0


# --------------------------------------------------------------------------
# Report driver
# --------------------------------------------------------------------------

def main():  # pragma: no cover - reporting entry point
    import sys
    side = 2 if len(jax.devices()) >= 4 else 1
    mesh = _mesh(side)
    print(f"[oracle] devices={len(jax.devices())} mesh={side}x{side} "
          f"platform={jax.devices()[0].platform}")
    # Part A is a definition-level algebra check on a deliberately
    # NON-mesh-divisible centroid axis (nmu=5), so it runs on 1x1; the wing
    # kernel shards that axis and does not pad it.
    print("\n== PART A: definitions under one convention (mesh 1x1) ==")
    a = part_a_residuals(_mesh(1))
    for key in sorted(a):
        print(f"  {key:28s} rel = {a[key][0]:.3e}")

    receipt = _receipt()
    print(f"\n[oracle] cubature: orders={receipt.orders} "
          f"physical={receipt.physical_counts} "
          f"cell_volume={receipt.cell_volume:.6f} "
          f"minibz_measure={receipt.minibz_measure:.6e}")
    print("\n== PART B: merged completion, all blocks nonzero ==")
    worst = 0.0
    for label, sigma in (("sigma_H = +z", _SIGMA_PLUS),
                         ("sigma_H = -z", -_SIGMA_PLUS),
                         ("sigma_H = +xyz", _SIGMA_FULL),
                         ("sigma_H = -xyz", -_SIGMA_FULL),
                         ("sigma_H = 0", np.zeros(3))):
        residual, evidence, _, _, _, _ = run_case(mesh, sigma, receipt=receipt)
        print(f"\n  --- {label} ---")
        for key in ("bare_D", "moments_all", "moments_CC", "moments_CT",
                    "moments_TC", "moments_TT", "V_packed", "W_packed",
                    "W_hermiticity"):
            extra = ""
            if key.startswith("moments_") and key != "moments_all":
                extra = f"   (block weight {residual['weight_' + key[8:]]:.3e})"
            print(f"    {key:16s} rel = {residual[key]:.3e}{extra}")
        print(f"    bare_D abs diff = {residual['bare_D_abs']:.3e}, "
              f"flipped-sign control = {residual['bare_D_flipped']:.3e}")
        print("    per-moment (u,v):", " ".join(
            f"{u}{v}={residual[f'moment_{u}{v}']:.1e}"
            for u in range(3) for v in range(3)))
        print(f"    solve: kappa_max={evidence.max_dyson_condition_number:.3e}"
              f" sigma_min={evidence.min_dyson_singular_value:.3e}"
              f" fwd_bound={evidence.max_dyson_forward_error_bound:.3e}"
              f" cubature={evidence.mixed_convergence_error_ratios}")
        worst = max(worst, max(
            v for k, v in residual.items()
            if k not in ("W_hermiticity", "bare_D_flipped", "bare_D_abs")
            and not k.startswith("weight_")))

    print("\n== LIMITS ==")
    residual0, evidence0, _, _, _, _ = run_case(
        mesh, np.zeros(3), receipt=receipt)
    got = np.asarray(evidence0.screened_moments)
    scale = float(np.max(np.abs(got)))
    print(f"  sigma_H=0 CT leakage    = "
          f"{np.max(np.abs(got[:, :, 0, 1:])) / scale:.3e}")
    print(f"  sigma_H=0 TC leakage    = "
          f"{np.max(np.abs(got[:, :, 1:, 0])) / scale:.3e}")
    print(f"  TT == bare <D_TT>       = "
          f"{_rel(got[:, :, 1:, 1:], _reference_bare_tt_moments(receipt)):.3e}")
    D_got = np.asarray(evidence0.bare_D_mean)
    print(f"  <v>                     = {D_got[0, 0].real:.6f}")
    print(f"  <D_TT> diag             = "
          f"{np.round(np.real(np.diag(D_got[1:, 1:])) / D_got[0, 0].real, 8)}"
          f"  (isotropic slab: -[0.5,0.5,1])")
    print(f"  tr<D_TT> + 2<v>         = "
          f"{abs(np.trace(D_got[1:, 1:]) + 2 * D_got[0, 0]):.3e}")
    fixture_nw = _fixture(mesh, sigma_H=_SIGMA_FULL, wings=False)
    print("  wings=0 -> S_eff == S   = "
          f"{_rel(schur_folded_response(fixture_nw.response, fixture_nw.W_host[0], float(receipt.cell_volume)), _gather(fixture_nw.response.S_direct)):.3e}")
    rw, _, _, _, _, _ = run_case(
        mesh, _SIGMA_FULL, wings=False, receipt=receipt)
    print(f"  wings=0 completion      = moments {rw['moments_all']:.3e}, "
          f"W {rw['W_packed']:.3e}")

    print("\n== NEGATIVE CONTROLS (must be LARGE) ==")
    for label, kwargs in (
            ("CT sign +i instead of -i", {"ct_sign": +1.0}),
            ("D_TT metric sign +v P^T", {"tt_sign": +1.0}),
            ("Schur fold dropped", {"wing_sign": 0.0}),
            ("Schur fold sign flipped", {"wing_sign": -1.0})):
        residual, _, _, _, _, _ = run_case(
            mesh, _SIGMA_FULL, receipt=receipt, **kwargs)
        print(f"  {label:26s} all {residual['moments_all']:.3e}"
              f"  CC {residual['moments_CC']:.3e}"
              f"  CT {residual['moments_CT']:.3e}"
              f"  TT {residual['moments_TT']:.3e}"
              f"  W {residual['W_packed']:.3e}")

    print("\n== PART C: exact centroid measure / absolute normalisation ==")
    exact = part_c_exact_measure_residuals()
    for key in ("insertion", "left_attachment", "right_attachment",
                "quoted_relation", "family_00", "family_0a", "family_a0",
                "family_ab"):
        print(f"  {key:28s} rel = {exact[key]:.3e}")
    print(f"  omitted 1/Omega control ratio = "
          f"{exact['omega_omitted_norm_ratio']:.12f} "
          f"(Omega={exact['cell_volume']:.12f})")
    print(f"  doubled 1/Omega control ratio = "
          f"{exact['omega_doubled_norm_ratio']:.12f} "
          f"(1/Omega={1.0 / exact['cell_volume']:.12f})")

    worst = max(worst, max(
        exact[key] for key in (
            "insertion", "left_attachment", "right_attachment",
            "quoted_relation", "family_00", "family_0a", "family_a0",
            "family_ab")))
    print(f"\n[oracle] WORST residual over every positive case = {worst:.3e}")
    return 0 if worst < _TOL else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
