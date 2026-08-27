"""Face-layout q-linear head/body wings and static metallic wings.

Guide: reports/gwjax_low_mem_bands_audit_2026-08-22/report.md, census
rows 6/7 ("Full q->0 head/body wings", "Static metallic wings").

Emulated CPU mesh (unit-scope; the FOUR-GPU RULE exempts unit/CPU cells
per QUALITY_PATTERNS.md).  Every legacy/face pair here is built from the
SAME host ``psi`` array via ``build_wavefunctions``/``build_wavefunctions_
face`` (the carrier's own "same psi, two faces" guarantee), so a
legacy-vs-face mismatch cannot be explained by the two bundles secretly
holding different physics -- and every case is ALSO checked against a
third, independent NumPy oracle built straight from the documented
formula (``head_wings_sharded``'s own docstring), so a bug shared by both
kernels would still be caught.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices,
    build_wavefunctions,
    build_wavefunctions_face,
    with_lorentz_vertices,
)
from common.gamma_matrices import gammas  # noqa: E402
import gw.qsgw_head as qsgw_head  # noqa: E402
from gw.qsgw_head import (  # noqa: E402
    _pad_head_band_manifold_to,
    head_wings_sharded,
    static_head_wings_sharded,
)


def _mesh_xy():
    devices = np.asarray(jax.devices("cpu"), dtype=object)
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    elif devices.size >= 2:
        devices = devices[:2].reshape(1, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _put(a, mesh, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def _gather(a):
    return np.asarray(jax.device_get(a))


def _host_inputs(rng, nk, nb, ns, nmu):
    """(psi_rmu_Y, psi_rmuT_X) -- see wavefunction_bundle.build_wavefunctions'
    docstring for the exact convention each carries."""
    psi = (rng.standard_normal((nk, nb, ns, nmu))
           + 1j * rng.standard_normal((nk, nb, ns, nmu)))
    psi_rmuT_X = np.conj(psi).transpose(0, 3, 1, 2)
    return psi, psi_rmuT_X


def _build_pair(rng, mesh, *, nk, nb, ns, nmu):
    """Legacy and face ``Wavefunctions`` bundles built from the IDENTICAL
    host psi, plus the raw host psi (nk,nb,ns,nmu) for the numpy oracle."""
    psi, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    occ_cut = nb // 2
    occ = np.where(np.arange(nb)[None, :] < occ_cut, 1.0, 0.0)
    occ = np.broadcast_to(occ, (nk, nb)).copy()
    slices = BandSlices.from_band_edges(0, 0, occ_cut, nb, nb)

    y_in = _put(psi, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    legacy = build_wavefunctions(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)
    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)
    # occ is derived from enk/efermi inside the builders; read it back so
    # the numpy oracle uses the SAME occupations both kernels consume.
    occ_used = _gather(legacy.occ)
    return legacy, face, psi, enk, occ_used


def _numpy_wings(v, e, f, psi, *, nb_logical, nk_tot, nspin, nspinor,
                  omega, eta, psi_y_bra=None, psi_y_ket=None,
                  psi_z_bra=None, psi_z_ket=None):
    """Independent O(nk*nb^2*n_omega) oracle straight from
    ``head_wings_sharded``'s own docstring formula -- no ring, no gather,
    no einsum-string reuse with either kernel under test."""
    nk, nb = e.shape
    mu = psi.shape[-1]
    n_omega = len(omega)
    n_vertex = int(v.shape[0])
    psi_y_bra = psi if psi_y_bra is None else psi_y_bra
    psi_y_ket = psi if psi_y_ket is None else psi_y_ket
    psi_z_bra = psi if psi_z_bra is None else psi_z_bra
    psi_z_ket = psi if psi_z_ket is None else psi_z_ket
    Y = np.zeros((n_omega, n_vertex, mu), dtype=np.complex128)
    Z = np.zeros((n_omega, mu, n_vertex), dtype=np.complex128)
    spin_denom = max(int(nspin), 1) * max(int(nspinor), 1)
    pref = -4.0 / (float(nk_tot) * spin_denom)
    for k in range(nk):
        for i in range(nb_logical):
            for j in range(nb_logical):
                dE = e[k, i] - e[k, j]
                if dE <= 0.0:
                    continue
                fdiff = f[k, j] - f[k, i]
                bij_y = np.einsum(
                    "sm,sm->m", np.conj(psi_y_bra[k, i]),
                    psi_y_ket[k, j])
                bij_z_conj = np.einsum(
                    "sm,sm->m", psi_z_bra[k, i],
                    np.conj(psi_z_ket[k, j]))
                for iw, om in enumerate(omega):
                    z = om + 1j * eta
                    denom = z * z - dE * dE
                    if abs(denom) <= 1.0e-16:
                        continue
                    w = pref * fdiff / denom
                    Y[iw] += (
                        np.conj(v[:, k, i, j])[:, None] * w
                        * bij_y[None, :])
                    Z[iw] += (
                        bij_z_conj[:, None] * w
                        * v[:, k, i, j][None, :])
    return Y, Z


def _host_gamma_apply(mu_lorentz, psi):
    """Dense host oracle for the bundle-owned monomial gamma apply."""
    gamma = np.asarray(jax.device_get(gammas[int(mu_lorentz)]))
    return np.einsum("st,kntm->knsm", gamma, psi, optimize=True)


def _numpy_static_wings(psi, surface, *, nb_logical, nk_tot, nspin, nspinor):
    nk, nb, ns, mu = psi.shape
    prefactor = -2.0 / (
        float(nk_tot) * max(int(nspin), 1) * max(int(nspinor), 1))
    density = np.sum(np.square(np.abs(psi)), axis=2)  # (nk, nb, mu)
    weight = np.where(np.arange(nb)[None, :] < nb_logical, surface, 0.0)
    return prefactor * np.einsum("kn,knm->m", weight, density)


# ---------------------------------------------------------------------------
# Dynamic wings: legacy vs face vs numpy oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nb_logical,nb", [(6, 6), (4, 6)])
def test_head_wings_face_matches_legacy_and_oracle(nb_logical, nb):
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260822 + nb_logical)
    nk, ns, nmu = 2, 2, 10
    nk_tot, nspin, nspinor = nk, 1, ns
    legacy, face, psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)

    # Matches the real calling convention (gw_jax.build_dft_head_response):
    # velocity_cart arrives ALREADY sliced to the [b0,b4) == nb_logical
    # chi0 manifold, not the bundle's full nb.
    v = (rng.standard_normal((3, nk, nb_logical, nb_logical))
         + 1j * rng.standard_normal((3, nk, nb_logical, nb_logical)))
    omega = np.asarray([0.1 + 0.0j, 0.3 + 0.02j])
    eta = 0.01

    e_slice = jnp.asarray(enk[:, :nb_logical])
    f_slice = jnp.asarray(occ[:, :nb_logical])

    Y_legacy, Z_legacy = head_wings_sharded(
        v, legacy, e_slice, f_slice, omega, mesh=mesh,
        nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor, eta_ry=eta)
    Y_face, Z_face = head_wings_sharded(
        v, face, e_slice, f_slice, omega, mesh=mesh,
        nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor, eta_ry=eta)

    Y_ref, Z_ref = _numpy_wings(
        v, enk, occ, psi, nb_logical=nb_logical, nk_tot=nk_tot,
        nspin=nspin, nspinor=nspinor, omega=omega, eta=eta)

    Y_legacy, Z_legacy = _gather(Y_legacy), _gather(Z_legacy)
    Y_face, Z_face = _gather(Y_face), _gather(Z_face)

    for name, got in (("legacy", (Y_legacy, Z_legacy)),
                      ("face", (Y_face, Z_face))):
        y_err = np.max(np.abs(got[0] - Y_ref)) / max(1.0e-300, np.max(np.abs(Y_ref)))
        z_err = np.max(np.abs(got[1] - Z_ref)) / max(1.0e-300, np.max(np.abs(Z_ref)))
        assert y_err < 1.0e-9, f"{name} Y_x vs numpy oracle rel err {y_err}"
        assert z_err < 1.0e-9, f"{name} Z_y vs numpy oracle rel err {z_err}"

    y_parity = np.max(np.abs(Y_legacy - Y_face))
    z_parity = np.max(np.abs(Z_legacy - Z_face))
    assert y_parity < 1.0e-10, f"legacy vs face Y_x parity {y_parity}"
    assert z_parity < 1.0e-10, f"legacy vs face Z_y parity {z_parity}"


def test_packed_vertex_wings_preserve_three_axis_bits_legacy_and_face():
    """One n_vertex kernel serves Cartesian and packed (a,I) axes."""
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260825)
    nk, nb, ns, nmu = 2, 4, 2, 8
    legacy, face, psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
    v3 = (rng.standard_normal((3, nk, nb, nb))
          + 1j * rng.standard_normal((3, nk, nb, nb)))
    v8 = np.zeros((8, nk, nb, nb), dtype=np.complex128)
    v8[:3] = v3
    kwargs = dict(
        energies_kn_ry=enk, occupations_kn=occ,
        omegas_ry=np.asarray([0.19 + 0.03j]), mesh=mesh,
        nb_logical=nb, nk_tot=nk, nspin=1, nspinor=ns)

    for wfns in (legacy, face):
        y3, z3 = head_wings_sharded(v3, wfns, **kwargs)
        y3_from_packed, z3_from_packed = head_wings_sharded(
            v8[:3], wfns, **kwargs)
        y8, z8 = head_wings_sharded(v8, wfns, **kwargs)
        y3, z3, y3_from_packed, z3_from_packed, y8, z8 = map(
            _gather,
            (y3, z3, y3_from_packed, z3_from_packed, y8, z8),
        )
        np.testing.assert_array_equal(y3_from_packed, y3)
        np.testing.assert_array_equal(z3_from_packed, z3)

        y8_oracle, z8_oracle = _numpy_wings(
            v8, enk, occ, psi,
            nb_logical=nb, nk_tot=nk, nspin=1, nspinor=ns,
            omega=np.asarray([0.19 + 0.03j]), eta=0.0)
        scale = max(
            1.0, float(np.max(np.abs(y8_oracle))),
            float(np.max(np.abs(z8_oracle))))
        assert max(
            float(np.max(np.abs(y8 - y8_oracle))),
            float(np.max(np.abs(z8 - z8_oracle))),
        ) <= 64.0 * np.finfo(float).eps * scale
        np.testing.assert_array_equal(y8[:, 3:], 0.0)
        np.testing.assert_array_equal(z8[..., 3:], 0.0)
        with pytest.raises(ValueError, match="canonical n_vertex"):
            head_wings_sharded(v8[:4], wfns, **kwargs)


def test_head_wings_face_mu_blocking_exercised(monkeypatch):
    """Force multiple mu blocks with a tiny mu count by shrinking
    ``_HEAD_WING_MU_BLOCK`` -- exercises the pad/scan/truncate path a
    default block of 64 would hide at this test's scale."""
    monkeypatch.setattr(qsgw_head, "_HEAD_WING_MU_BLOCK", 3)
    qsgw_head._KERNEL_CACHE.clear()
    mesh = _mesh_xy()
    rng = np.random.default_rng(99)
    nk, nb, ns, nmu = 2, 4, 1, 8  # mu=8, block=3 -> 3 blocks (3,3,2 padded)
    nk_tot, nspin, nspinor = nk, 1, 1
    legacy, face, psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)

    v = (rng.standard_normal((3, nk, nb, nb))
         + 1j * rng.standard_normal((3, nk, nb, nb)))
    omega = np.asarray([0.2 + 0.0j])
    eta = 0.02
    e_slice = jnp.asarray(enk[:, :nb])
    f_slice = jnp.asarray(occ[:, :nb])

    Y_face, Z_face = head_wings_sharded(
        v, face, e_slice, f_slice, omega, mesh=mesh, nb_logical=nb,
        nk_tot=nk_tot, nspin=nspin, nspinor=nspinor, eta_ry=eta)
    Y_ref, Z_ref = _numpy_wings(
        v, enk, occ, psi, nb_logical=nb, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor, omega=omega, eta=eta)
    assert _gather(Y_face).shape == (1, 3, nmu)
    assert _gather(Z_face).shape == (1, nmu, 3)
    y_err = np.max(np.abs(_gather(Y_face) - Y_ref))
    z_err = np.max(np.abs(_gather(Z_face) - Z_ref))
    assert y_err < 1.0e-9, y_err
    assert z_err < 1.0e-9, z_err
    qsgw_head._KERNEL_CACHE.clear()


def test_head_wings_face_separate_lorentz_endpoints_preserve_order():
    """The one face kernel must distinguish CC/TC/CT/TT endpoint algebra.

    ``with_lorentz_vertices(A,0)`` owns the Y/bra face and
    ``with_lorentz_vertices(0,B)`` owns the Z/ket face.  Gamma-2 makes the
    CT reference explicitly complex.  The phase twins below are the sharper
    placement discriminator: bra-only and ket-only multiplication by ``i``
    have opposite signs in both Y and Z, so an implementation that aliases
    either endpoint (the old gamma^dagger gamma cancellation) cannot pass.
    """
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260827)
    nk, nb, ns, nmu = 2, 4, 4, 8
    _legacy, face, psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
    v = (rng.standard_normal((3, nk, nb, nb))
         + 1j * rng.standard_normal((3, nk, nb, nb)))
    omega = np.asarray([0.17 + 0.03j])
    common = dict(
        energies_kn_ry=enk, occupations_kn=occ, omegas_ry=omega,
        mesh=mesh, nb_logical=nb, nk_tot=nk, nspin=1, nspinor=ns,
        eta_ry=0.01)

    left_T = with_lorentz_vertices(face, 1, 0)
    right_T = with_lorentz_vertices(face, 0, 2)
    gamma1_psi = _host_gamma_apply(1, psi)
    gamma2_psi = _host_gamma_apply(2, psi)

    cases = (
        ("CC", {}, {}),
        ("TC", {"body_bra_wfns": left_T},
         {"psi_y_bra": gamma1_psi}),
        ("CT", {"body_ket_wfns": right_T},
         {"psi_z_ket": gamma2_psi}),
        ("TT", {"body_bra_wfns": left_T, "body_ket_wfns": right_T},
         {"psi_y_bra": gamma1_psi, "psi_z_ket": gamma2_psi}),
    )
    outputs = {}
    for name, endpoint_kwargs, oracle_kwargs in cases:
        Y, Z = head_wings_sharded(v, face, **common, **endpoint_kwargs)
        wanted_y = NamedSharding(mesh, P(None, None, "x"))
        wanted_z = NamedSharding(mesh, P(None, "y", None))
        assert Y.sharding.is_equivalent_to(wanted_y, Y.ndim)
        assert Z.sharding.is_equivalent_to(wanted_z, Z.ndim)
        Y_ref, Z_ref = _numpy_wings(
            v, enk, occ, psi, nb_logical=nb, nk_tot=nk, nspin=1,
            nspinor=ns, omega=omega, eta=0.01, **oracle_kwargs)
        Y, Z = _gather(Y), _gather(Z)
        scale = max(1.0, float(np.max(np.abs(Y_ref))),
                    float(np.max(np.abs(Z_ref))))
        np.testing.assert_allclose(Y, Y_ref, rtol=0.0,
                                   atol=64.0 * np.finfo(float).eps * scale)
        np.testing.assert_allclose(Z, Z_ref, rtol=0.0,
                                   atol=64.0 * np.finfo(float).eps * scale)
        outputs[name] = (Y, Z)

    # TC changes only Y; CT changes only Z.  These are direction/order
    # assertions, not a weak non-zero-magnitude check.
    np.testing.assert_array_equal(outputs["TC"][1], outputs["CC"][1])
    np.testing.assert_array_equal(outputs["CT"][0], outputs["CC"][0])
    assert np.max(np.abs(outputs["TC"][0] - outputs["CC"][0])) > 1.0e-6
    assert np.max(np.abs(outputs["CT"][1] - outputs["CC"][1])) > 1.0e-6

    phase_face = dataclasses.replace(
        face, psi_mun=1j * face.psi_mun, psi_nmu=1j * face.psi_nmu)
    Y_bra, Z_bra = map(_gather, head_wings_sharded(
        v, face, **common, body_bra_wfns=phase_face))
    Y_ket, Z_ket = map(_gather, head_wings_sharded(
        v, face, **common, body_ket_wfns=phase_face))
    Y_cc, Z_cc = outputs["CC"]
    scale = max(1.0, float(np.max(np.abs(Y_cc))),
                float(np.max(np.abs(Z_cc))))
    atol = 64.0 * np.finfo(float).eps * scale
    np.testing.assert_allclose(Y_bra, -1j * Y_cc, rtol=0.0, atol=atol)
    np.testing.assert_allclose(Z_bra, +1j * Z_cc, rtol=0.0, atol=atol)
    np.testing.assert_allclose(Y_ket, +1j * Y_cc, rtol=0.0, atol=atol)
    np.testing.assert_allclose(Z_ket, -1j * Z_cc, rtol=0.0, atol=atol)


# ---------------------------------------------------------------------------
# Static wings: legacy vs face vs numpy oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nb_logical,nb", [(6, 6), (4, 6)])
def test_static_head_wings_face_matches_legacy_and_oracle(nb_logical, nb):
    mesh = _mesh_xy()
    rng = np.random.default_rng(555 + nb_logical)
    nk, ns, nmu = 2, 2, 8
    nk_tot, nspin, nspinor = nk, 1, ns
    legacy, face, psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
    surface = np.abs(rng.standard_normal((nk, nb)))

    left_legacy, right_legacy = static_head_wings_sharded(
        legacy, surface, mesh=mesh, nb_logical=nb_logical, nk_tot=nk_tot,
        nspin=nspin, nspinor=nspinor)
    left_face, right_face = static_head_wings_sharded(
        face, surface, mesh=mesh, nb_logical=nb_logical, nk_tot=nk_tot,
        nspin=nspin, nspinor=nspinor)

    ref = _numpy_static_wings(
        psi, surface, nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor)

    left_legacy, right_legacy = _gather(left_legacy), _gather(right_legacy)
    left_face, right_face = _gather(left_face), _gather(right_face)

    for name, (left, right) in (("legacy", (left_legacy, right_legacy)),
                                 ("face", (left_face, right_face))):
        l_err = np.max(np.abs(left - ref))
        r_err = np.max(np.abs(right - ref))
        assert l_err < 1.0e-10, f"{name} left vs oracle {l_err}"
        assert r_err < 1.0e-10, f"{name} right vs oracle {r_err}"

    assert np.max(np.abs(left_legacy - left_face)) < 1.0e-10
    assert np.max(np.abs(right_legacy - right_face)) < 1.0e-10


# ---------------------------------------------------------------------------
# Refusals / helper correctness
# ---------------------------------------------------------------------------

def test_pad_head_band_manifold_to_refuses_narrower_width():
    mesh = _mesh_xy()
    v = jnp.zeros((3, 2, 4, 4), dtype=jnp.complex128)
    e = jnp.zeros((2, 4))
    f = jnp.zeros((2, 4))
    s = jnp.zeros((2, 4))
    with pytest.raises(ValueError):
        _pad_head_band_manifold_to(v, e, f, s, mesh=mesh, width=2)


def test_pad_head_band_manifold_to_commits_to_declared_sharding():
    mesh = _mesh_xy()
    v = jnp.zeros((3, 2, 4, 4), dtype=jnp.complex128)
    e = jnp.zeros((2, 4))
    f = jnp.zeros((2, 4))
    s = jnp.zeros((2, 4))
    v_out, e_out, f_out, s_out = _pad_head_band_manifold_to(
        v, e, f, s, mesh=mesh, width=8)
    assert v_out.shape == (3, 2, 8, 8)
    assert v_out.committed
    assert e_out.shape == f_out.shape == s_out.shape == (2, 8)


def test_dft_head_refuses_dipole_provenance_before_read_or_allocation(
        tmp_path, monkeypatch):
    """A four-spinor charge body must not consume a two-spinor dipole."""
    import common.chi_from_dipole as dipole_reader
    import psp.get_dipole_mtxels as dipole_owner

    (tmp_path / "dipole.h5").write_bytes(b"not read")
    monkeypatch.setattr(
        dipole_owner, "resolve_vnl_velocity_sign", lambda *_args: +1)
    monkeypatch.setattr(
        dipole_owner, "check_dipole_provenance", lambda *_args, **_kw: False)
    monkeypatch.setattr(
        dipole_reader, "read_dipole_h5",
        lambda *_args, **_kw: pytest.fail("dipole payload was read"))

    config = SimpleNamespace(
        nval=2, ncond=2, nband=4, vnl_velocity_sign="")
    meta = SimpleNamespace(nspinor=4)
    with pytest.raises(ValueError, match="dft_head_dipole_provenance"):
        qsgw_head.build_dft_head_response(
            object(), [0.0], input_dir=str(tmp_path), mesh=object(),
            wfn=object(), meta=meta, config=config)


def test_head_wings_sharded_face_requires_face_psi_fields():
    mesh = _mesh_xy()
    rng = np.random.default_rng(1)
    nk, nb, ns, nmu = 1, 2, 1, 2
    legacy, face, _psi, enk, occ = _build_pair(
        rng, mesh, nk=nk, nb=nb, ns=ns, nmu=nmu)
    broken = face.__class__(
        enk=face.enk, occ=face.occ, slices=face.slices, layout="face")
    v = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    omega = np.asarray([0.1 + 0.0j])
    with pytest.raises(ValueError):
        head_wings_sharded(
            v, broken, jnp.asarray(enk), jnp.asarray(occ), omega,
            mesh=mesh, nb_logical=nb, nk_tot=nk, nspin=1, nspinor=1)
    with pytest.raises(ValueError, match="face-layout only"):
        head_wings_sharded(
            v, legacy, jnp.asarray(enk), jnp.asarray(occ), omega,
            mesh=mesh, nb_logical=nb, nk_tot=nk, nspin=1, nspinor=1,
            body_bra_wfns=face)
