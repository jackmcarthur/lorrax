"""Velocity kinetic balance for the SPATIAL-CURRENT carrier (2026-09-04).

The charge carrier keeps ``psi_S = (alpha/2) sigma.p psi_L``; the
spatial-current carrier may be lifted with ``sigma.v``,
``v = p + dV_NL/dk`` (deck ``bispinor_current_balance = velocity``), so
the ``alpha^i`` vertex is the pseudo-Hamiltonian's velocity at first order
in q.  Three altitudes are gated here:

* the resolver / deck grammar (``common.four_current_model``, ``gw_config``);
* the lift algebra on synthetic data (``common.bispinor_init``);
* the loader hook on a real WFN with its own projectors
  (``tests/regression/cohsex_debug``: MoS2, nspinor=2, FR ONCV UPFs) —
  including the q=0 current identity

      (2/alpha) <m,k| alpha^i |n,k>
          = <m| v_i |n>  +  (i/2) eps_ijk <m| [sigma^k, v_j^NL] |n>     (Ry)

  which for the raw sigma.p lift collapses to the bare momentum
  ``<m| 2(k+G)_i |n>`` — the whole point of the change is the middle term.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from common.bispinor_init import (
    ALPHA_FS,
    HALFALPHA,
    KINETIC_BALANCE_LIFT_PROVENANCE,
    RAW_KINETIC_BALANCE_LIFT,
    VELOCITY_KINETIC_BALANCE_LIFT,
    VELOCITY_KINETIC_BALANCE_LIFT_PROVENANCE,
    kinetic_balance_lift_provenance,
    lift_to_4spinor,
    sigma_dot_cartesian_kets,
)
from common.four_current_model import (
    RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION,
    VELOCITY_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION,
    resolve_four_current_representation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "regression" / "cohsex_debug"
WFN_PATH = FIXTURE_DIR / "WFNsmall.h5"

_PAULI = np.asarray([
    [[0.0, 1.0], [1.0, 0.0]],
    [[0.0, -1.0j], [1.0j, 0.0]],
    [[1.0, 0.0], [0.0, -1.0]],
], dtype=np.complex128)
_ALPHA4 = np.zeros((3, 4, 4), dtype=np.complex128)
for _a in range(3):
    _ALPHA4[_a, :2, 2:] = _PAULI[_a]
    _ALPHA4[_a, 2:, :2] = _PAULI[_a]


# ---------------------------------------------------------------------------
# resolver + deck grammar
# ---------------------------------------------------------------------------

def test_velocity_moves_only_the_current_carrier():
    raw = resolve_four_current_representation(True, "bare_transverse")
    vel = resolve_four_current_representation(
        True, "bare_transverse", current_lift="velocity")
    assert raw.current_lift == RAW_KINETIC_BALANCE_LIFT
    assert vel.current_lift == VELOCITY_KINETIC_BALANCE_LIFT
    assert (raw.spatial_current_representation
            == RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION)
    assert (vel.spatial_current_representation
            == VELOCITY_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION)
    # Everything charge-side is untouched: same charge lift, same charge
    # stamp, same scalar-head producer.
    assert vel.charge_lift == raw.charge_lift == RAW_KINETIC_BALANCE_LIFT
    assert vel.charge_representation == raw.charge_representation
    assert vel.scalar_head_bispinor == raw.scalar_head_bispinor
    assert vel.charge_bispinor and vel.current_bispinor


def test_velocity_without_bispinor_refuses_by_name():
    with pytest.raises(ValueError,
                       match="bispinor_current_balance_requires_bispinor"):
        resolve_four_current_representation(
            False, "bare_transverse", current_lift="velocity")
    with pytest.raises(ValueError, match="not a spatial-current carrier"):
        resolve_four_current_representation(
            True, "bare_transverse", current_lift="isometric")


def test_provenance_names_the_velocity_lift():
    assert (kinetic_balance_lift_provenance("velocity")
            == VELOCITY_KINETIC_BALANCE_LIFT_PROVENANCE)
    assert (kinetic_balance_lift_provenance("raw")
            == KINETIC_BALANCE_LIFT_PROVENANCE)
    assert (VELOCITY_KINETIC_BALANCE_LIFT_PROVENANCE
            != KINETIC_BALANCE_LIFT_PROVENANCE)


def test_deck_key_resolves_to_the_lift_selector(tmp_path):
    from gw.gw_config import LorraxConfig, coerce_bispinor_current_lift
    assert coerce_bispinor_current_lift("kinetic") == "raw"
    assert coerce_bispinor_current_lift(None) == "raw"
    assert coerce_bispinor_current_lift("") == "raw"
    assert coerce_bispinor_current_lift(" Velocity ") == "velocity"
    with pytest.raises(ValueError, match="bispinor_current_balance"):
        coerce_bispinor_current_lift("sigma_v")

    base = ("[cohsex]\nnval = 2\nncond = 2\nnband = 10\n"
            "memory_per_device_gb = 4.0\n")
    quiet = dict(print_fn=lambda *a, **k: None)
    deck = tmp_path / "default.in"
    deck.write_text(base)
    cfg = LorraxConfig.from_input_file(str(deck), **quiet)
    assert cfg.bispinor_current_lift == "raw"
    assert cfg.paths.pseudo_dir is None

    deck = tmp_path / "vel.in"
    deck.write_text(base + "bispinor = true\n"
                    "bispinor_current_balance = velocity\n"
                    "pseudo_dir = pp\n")
    cfg = LorraxConfig.from_input_file(str(deck), **quiet)
    assert cfg.bispinor_current_lift == "velocity"
    # Resolved against the deck directory like every other input path.
    assert cfg.paths.pseudo_dir == str(tmp_path / "pp")

    deck = tmp_path / "scalar_vel.in"
    deck.write_text(base + "bispinor_current_balance = velocity\n")
    with pytest.raises(ValueError,
                       match="bispinor_current_balance_requires_bispinor"):
        LorraxConfig.from_input_file(str(deck), **quiet)


# ---------------------------------------------------------------------------
# lift algebra, synthetic
# ---------------------------------------------------------------------------

def _synthetic():
    rng = np.random.default_rng(7)
    n_k, nb, ng = 2, 3, 5
    psi = (rng.standard_normal((n_k, nb, 2, ng))
           + 1j * rng.standard_normal((n_k, nb, 2, ng)))
    gvecs = rng.integers(-3, 4, size=(n_k, ng, 3)).astype(np.float64)
    kvecs = rng.uniform(-0.5, 0.5, size=(n_k, 3))
    bvec = np.asarray([[1.1, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.3]])
    kets = (rng.standard_normal((n_k, 3, nb, 2, ng))
            + 1j * rng.standard_normal((n_k, 3, nb, 2, ng)))
    return psi, gvecs, kvecs, bvec, kets


def test_sigma_dot_cartesian_kets_matches_explicit_pauli_sum():
    _, _, _, _, kets = _synthetic()
    got = np.asarray(sigma_dot_cartesian_kets(jnp.asarray(kets)))
    want = np.einsum("aij,kabjg->kbig", _PAULI, kets)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-14)
    with pytest.raises(ValueError, match="\\(\\.\\.\\., 3, band, 2, G\\)"):
        sigma_dot_cartesian_kets(jnp.asarray(kets[:, :2]))


def test_velocity_lift_is_raw_lift_plus_alpha_over_four_sigma_vnl():
    psi, gvecs, kvecs, bvec, kets = _synthetic()
    sigma_v = np.einsum("aij,kabjg->kbig", _PAULI, kets)
    raw = np.asarray(lift_to_4spinor(
        jnp.asarray(psi), jnp.asarray(gvecs), jnp.asarray(kvecs),
        jnp.asarray(bvec)))
    vel = np.asarray(lift_to_4spinor(
        jnp.asarray(psi), jnp.asarray(gvecs), jnp.asarray(kvecs),
        jnp.asarray(bvec), representation="velocity",
        sigma_vnl_velocity_ry=jnp.asarray(sigma_v)))
    # large components untouched, small ones shifted by (alpha/4) sigma.v_NL
    np.testing.assert_array_equal(vel[:, :, :2], raw[:, :, :2])
    np.testing.assert_allclose(
        vel[:, :, 2:] - raw[:, :, 2:], 0.5 * HALFALPHA * sigma_v,
        rtol=0.0, atol=1e-15)
    assert abs(0.5 * HALFALPHA - ALPHA_FS / 4.0) < 1e-18


def test_velocity_lift_refuses_a_missing_or_stray_operand():
    psi, gvecs, kvecs, bvec, kets = _synthetic()
    args = (jnp.asarray(psi), jnp.asarray(gvecs), jnp.asarray(kvecs),
            jnp.asarray(bvec))
    with pytest.raises(ValueError, match="requires sigma_vnl_velocity_ry"):
        lift_to_4spinor(*args, representation="velocity")
    sigma_v = jnp.asarray(np.einsum("aij,kabjg->kbig", _PAULI, kets))
    with pytest.raises(ValueError, match="only meaningful"):
        lift_to_4spinor(*args, sigma_vnl_velocity_ry=sigma_v)
    with pytest.raises(ValueError, match="shape"):
        lift_to_4spinor(*args, representation="velocity",
                        sigma_vnl_velocity_ry=sigma_v[:, :2])


# ---------------------------------------------------------------------------
# loader hook + real projectors (MoS2 fixture)
# ---------------------------------------------------------------------------

def _fixture_or_skip():
    if not WFN_PATH.exists():
        pytest.skip(f"fixture WFN missing: {WFN_PATH}")
    if not list(FIXTURE_DIR.glob("*.upf")) and not list(
            FIXTURE_DIR.glob("*.UPF")):
        pytest.skip("cohsex_debug carries no pseudopotentials")


def _open_loader_with_setup():
    from wfn_loader import WfnLoader
    from psp import vnl_ops
    from psp.pseudos import load_pseudopotentials
    loader = WfnLoader(str(WFN_PATH))
    if int(loader.nspinor) != 2:
        loader.close()
        pytest.skip("test requires a 2-spinor WFN")
    sym = loader.symmetry()
    pseudos = load_pseudopotentials(str(FIXTURE_DIR))
    setup = vnl_ops.build_vnl_setup(
        loader, sym, None, pseudos, nspinor=2, print_fn=lambda *a, **k: None)
    return loader, sym, setup


def _valid_sphere(loader, ik, k):
    n = int(loader.ngk_valid(k=k)[ik])
    g = np.asarray(loader.gvecs(k=k))[ik, :n].astype(np.int32)
    kvec = np.asarray(loader.kvecs(k=k), dtype=np.float64)[ik]
    return n, g, kvec


def test_velocity_lift_needs_the_hook_and_then_matches_the_projector_velocity():
    _fixture_or_skip()
    from psp import vnl_ops
    loader, sym, setup = _open_loader_with_setup()
    try:
        nb = min(4, int(loader.nbands))
        with pytest.raises(ValueError,
                           match="bispinor_velocity_lift_needs_projectors"):
            loader.load(bands=(0, nb), k="full_bz", bispinor=True,
                        bispinor_lift="velocity")
        loader.nonlocal_velocity_lift = vnl_ops.nonlocal_velocity_lift(setup)
        raw = np.asarray(loader.load(bands=(0, nb), k="full_bz",
                                     bispinor=True))
        vel = np.asarray(loader.load(bands=(0, nb), k="full_bz",
                                     bispinor=True, bispinor_lift="velocity"))
        # Same-hook kinetic request is byte-identical to the shipped lift.
        raw_again = np.asarray(loader.load(bands=(0, nb), k="full_bz",
                                           bispinor=True, bispinor_lift="raw"))
        np.testing.assert_array_equal(raw_again, raw)
        np.testing.assert_array_equal(vel[:, :, :2], raw[:, :, :2])
        n_k = raw.shape[0]
        for ik in range(n_k):
            n, g, kvec = _valid_sphere(loader, ik, "full_bz")
            psi_L = raw[ik, :, :2, :n]
            kdata = vnl_ops.build_vnl_kdata_from_kvec(
                kvec, g, setup, compute_dZ=True)
            v_ket = np.asarray(vnl_ops.apply_vnl_velocity_to_ket(
                jnp.asarray(psi_L), kdata.Z, kdata.dZ, kdata.E_super))
            want = 0.5 * HALFALPHA * np.einsum(
                "aij,abjg->big", _PAULI, v_ket)
            got = vel[ik, :, 2:, :n] - raw[ik, :, 2:, :n]
            scale = max(float(np.max(np.abs(want))), 1e-300)
            np.testing.assert_allclose(got, want, rtol=0.0,
                                       atol=1e-12 * scale)
            assert scale > 0.0, "V_NL velocity vanished on a real deck"
            # pad columns are exactly zero on both lifts
            assert not np.any(vel[ik, :, :, n:])
            assert not np.any(raw[ik, :, :, n:])
    finally:
        loader.close()


def _current_matrix(psi4, ngk):
    """(2/alpha) <m| alpha^i |n> over the valid sphere; (3, nb, nb)."""
    p = psi4[:, :, :ngk]
    return (2.0 / ALPHA_FS) * np.einsum(
        "msg,ist,ntg->imn", np.conj(p), _ALPHA4, p)


def test_q0_current_identity_kinetic_gives_momentum_velocity_gives_dH_dk():
    """The load-bearing physics cell.

    raw lift:       (2/alpha)<alpha^i> = <2(k+G)_i>                        (Ry)
    velocity lift:  (2/alpha)<alpha^i> = <2(k+G)_i + dV_NL/dk_i>
                                         + (i/2) eps_ijk <[sigma^k, dV_NL/dk_j]>
    The commutator survives only through the j-resolved (spin-orbit) part
    of V_NL and is transverse-free; both arms are checked against
    independent NumPy contractions of the same projector kets.
    """
    _fixture_or_skip()
    from psp import vnl_ops
    from psp.dft_operators import momentum_matrix_k
    loader, sym, setup = _open_loader_with_setup()
    try:
        nb = min(6, int(loader.nbands))
        loader.nonlocal_velocity_lift = vnl_ops.nonlocal_velocity_lift(setup)
        raw = np.asarray(loader.load(bands=(0, nb), k="ibz", bispinor=True))
        vel = np.asarray(loader.load(bands=(0, nb), k="ibz", bispinor=True,
                                     bispinor_lift="velocity"))
        B = jnp.asarray(float(loader.blat) * np.asarray(loader.bvec, float))
        checked = 0
        for ik in range(raw.shape[0]):
            n, g, kvec = _valid_sphere(loader, ik, "ibz")
            psi_L = jnp.asarray(raw[ik, :, :2, :n])
            p_mn = np.asarray(momentum_matrix_k(
                psi_L, jnp.asarray(g), jnp.asarray(kvec), B))     # (3,nb,nb) Ry
            kdata = vnl_ops.build_vnl_kdata_from_kvec(
                kvec, g, setup, compute_dZ=True)
            vnl_mn = np.asarray(vnl_ops.vnl_velocity_matrix(
                psi_L, kdata.Z, kdata.dZ, kdata.E_super))
            v_ket = np.asarray(vnl_ops.apply_vnl_velocity_to_ket(
                psi_L, kdata.Z, kdata.dZ, kdata.E_super))      # (3,nb,2,n)
            psi_np = np.asarray(psi_L)
            # C[j,k,m,n] = <m| sigma^k v_j - v_j sigma^k |n>
            #   <m|sigma^k v_j|n> = psi_m^dag sigma^k (v_j psi_n)
            #   <m|v_j sigma^k|n> = (v_j psi_m)^dag sigma^k psi_n   (v Hermitian)
            sig_v = np.einsum("kst,jntg->jknsg", _PAULI, v_ket)
            sig_psi = np.einsum("kst,ntg->knsg", _PAULI, psi_np)
            C = (np.einsum("msg,jknsg->jkmn", np.conj(psi_np), sig_v)
                 - np.einsum("jmsg,knsg->jkmn", np.conj(v_ket), sig_psi))
            eps = np.zeros((3, 3, 3))
            eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
            eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
            comm = 0.5j * np.einsum("ijk,jkmn->imn", eps, C)

            lhs_raw = _current_matrix(raw[ik], n)
            lhs_vel = _current_matrix(vel[ik], n)
            scale = float(np.max(np.abs(p_mn)))
            np.testing.assert_allclose(lhs_raw, p_mn, rtol=0.0,
                                       atol=1e-11 * scale)
            np.testing.assert_allclose(lhs_vel, p_mn + vnl_mn + comm,
                                       rtol=0.0, atol=1e-11 * scale)
            # and the difference between the arms IS the nonlocal velocity
            # (plus its spin-orbit commutator) — not a rounding-level shift
            assert float(np.max(np.abs(vnl_mn))) > 1e-6 * scale
            checked += 1
        assert checked > 0
    finally:
        loader.close()
