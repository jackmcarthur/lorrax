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
  including the q=0 current identity, per channel a,

      (2/alpha) <m,k| alpha^a |n,k>_{carrier a} = <m| 2(k+G)_a + dV_NL/dk_a |n>   (Ry)

  EXACTLY, with j-RESOLVED (spin-orbit) projectors: the spin-scalar part of
  the velocity rides the sigma sandwich and the spin-orbit part sits behind
  sigma^a alone, so no commutator term survives.  The raw sigma.p lift gives
  the bare momentum ``<m| 2(k+G)_a |n>``.  A single sigma.v carrier would
  give the velocity plus ``(i/2) eps_abc <[sigma^c, dV_SO/dk_b]>``; that
  arm is gone from the tree and this file is what says so.
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
    VELOCITY_KINETIC_BALANCE_LIFTS,
    VELOCITY_KINETIC_BALANCE_LIFT_PROVENANCE,
    kinetic_balance_lift_provenance,
    lift_to_4spinor,
    sigma_dot_cartesian_kets,
    velocity_lift_channel,
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
    assert raw.one_current_carrier and not vel.one_current_carrier
    for a in (1, 2, 3):
        assert raw.current_lift_for(a) == RAW_KINETIC_BALANCE_LIFT
        assert vel.current_lift_for(a) == VELOCITY_KINETIC_BALANCE_LIFTS[a - 1]
        assert velocity_lift_channel(vel.current_lift_for(a)) == a
    with pytest.raises(ValueError, match="mu_L must be 1, 2 or 3"):
        vel.current_lift_for(0)
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
    for sel in VELOCITY_KINETIC_BALANCE_LIFTS:
        assert (kinetic_balance_lift_provenance(sel)
                == VELOCITY_KINETIC_BALANCE_LIFT_PROVENANCE)
    assert velocity_lift_channel("raw") is None
    assert velocity_lift_channel("velocity") is None
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
        jnp.asarray(bvec), representation="velocity_1",
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
        lift_to_4spinor(*args, representation="velocity_2")
    sigma_v = jnp.asarray(np.einsum("aij,kabjg->kbig", _PAULI, kets))
    with pytest.raises(ValueError, match="names the per-channel scheme"):
        lift_to_4spinor(*args, representation="velocity",
                        sigma_vnl_velocity_ry=sigma_v)
    with pytest.raises(ValueError, match="only meaningful"):
        lift_to_4spinor(*args, sigma_vnl_velocity_ry=sigma_v)
    with pytest.raises(ValueError, match="shape"):
        lift_to_4spinor(*args, representation="velocity_3",
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
                        bispinor_lift="velocity_1")
        loader.nonlocal_velocity_lift = vnl_ops.nonlocal_velocity_lift(setup)
        raw = np.asarray(loader.load(bands=(0, nb), k="full_bz",
                                     bispinor=True))
        # Same-hook kinetic request is byte-identical to the shipped lift.
        raw_again = np.asarray(loader.load(bands=(0, nb), k="full_bz",
                                           bispinor=True, bispinor_lift="raw"))
        np.testing.assert_array_equal(raw_again, raw)
        # The public lift of the same window reproduces load(bispinor=True).
        psi_2 = loader.load(bands=(0, nb), k="full_bz")
        np.testing.assert_array_equal(
            np.asarray(loader.lift(psi_2, k="full_bz")), raw)
        E_SR, E_SO = vnl_ops.spin_orbit_split_E(setup)
        assert bool(setup.soc), "fixture should resolve j-RESOLVED"
        assert float(jnp.max(jnp.abs(E_SO))) > 0.0
        n_k = raw.shape[0]
        for a in (1, 2, 3):
            vel = np.asarray(loader.load(
                bands=(0, nb), k="full_bz", bispinor=True,
                bispinor_lift=f"velocity_{a}"))
            np.testing.assert_array_equal(
                np.asarray(loader.lift(psi_2, k="full_bz",
                                       bispinor_lift=f"velocity_{a}")), vel)
            np.testing.assert_array_equal(vel[:, :, :2], raw[:, :, :2])
            for ik in range(n_k):
                n, g, kvec = _valid_sphere(loader, ik, "full_bz")
                psi_L = raw[ik, :, :2, :n]
                kdata = vnl_ops.build_vnl_kdata_from_kvec(
                    kvec, g, setup, compute_dZ=True)
                v_sr = np.asarray(vnl_ops.apply_vnl_velocity_to_ket(
                    jnp.asarray(psi_L), kdata.Z, kdata.dZ, E_SR))
                v_so = np.asarray(vnl_ops.apply_vnl_velocity_to_ket(
                    jnp.asarray(psi_L), kdata.Z, kdata.dZ, E_SO))
                ket = (np.einsum("bij,bnjg->nig", _PAULI, v_sr)
                       + np.einsum("ij,njg->nig", _PAULI[a - 1], v_so[a - 1]))
                want = 0.5 * HALFALPHA * ket
                got = vel[ik, :, 2:, :n] - raw[ik, :, 2:, :n]
                scale = max(float(np.max(np.abs(want))), 1e-300)
                np.testing.assert_allclose(got, want, rtol=0.0,
                                           atol=1e-12 * scale)
                assert not np.any(vel[ik, :, :, n:])
    finally:
        loader.close()


def _current_matrix(psi4, ngk):
    """(2/alpha) <m| alpha^i |n> over the valid sphere; (3, nb, nb)."""
    p = psi4[:, :, :ngk]
    return (2.0 / ALPHA_FS) * np.einsum(
        "msg,ist,ntg->imn", np.conj(p), _ALPHA4, p)


def test_q0_current_identity_kinetic_gives_momentum_velocity_gives_dH_dk():
    """The load-bearing physics cell.

    raw lift:         (2/alpha)<alpha^a> = <2(k+G)_a>                          (Ry)
    velocity_a lift:  (2/alpha)<alpha^a> = <2(k+G)_a + dV_NL/dk_a>  EXACTLY,
                      with j-RESOLVED projectors (dV_NL includes the
                      spin-orbit part), no commutator remainder.
    Both arms against independent contractions of the same projector kets.
    """
    _fixture_or_skip()
    from psp import vnl_ops
    from psp.dft_operators import momentum_matrix_k
    loader, sym, setup = _open_loader_with_setup()
    try:
        assert bool(setup.soc)
        nb = min(6, int(loader.nbands))
        loader.nonlocal_velocity_lift = vnl_ops.nonlocal_velocity_lift(setup)
        raw = np.asarray(loader.load(bands=(0, nb), k="ibz", bispinor=True))
        vel = [np.asarray(loader.load(bands=(0, nb), k="ibz", bispinor=True,
                                      bispinor_lift=f"velocity_{a}"))
               for a in (1, 2, 3)]
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
                psi_L, kdata.Z, kdata.dZ, kdata.E_super))  # full V_NL incl. SO
            scale = float(np.max(np.abs(p_mn)))
            lhs_raw = _current_matrix(raw[ik], n)
            np.testing.assert_allclose(lhs_raw, p_mn, rtol=0.0,
                                       atol=1e-11 * scale)
            for a in (1, 2, 3):
                lhs_a = _current_matrix(vel[a - 1][ik], n)[a - 1]
                np.testing.assert_allclose(
                    lhs_a, p_mn[a - 1] + vnl_mn[a - 1],
                    rtol=0.0, atol=1e-11 * scale)
            # and the nonlocal part is not a rounding-level shift
            assert float(np.max(np.abs(vnl_mn))) > 1e-6 * scale
            checked += 1
        assert checked > 0
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# One-pass lift: the three channel carriers from one projector build
# ---------------------------------------------------------------------------

def test_hook_channel_zero_stacks_the_three_channel_kets():
    """``channel=0`` builds Z/dZ once and returns the three kets the
    per-channel calls return (to rounding: one apply of dV_SO over all three
    components against three single-component applies)."""
    _fixture_or_skip()
    from psp import vnl_ops
    loader, sym, setup = _open_loader_with_setup()
    try:
        nb = min(4, int(loader.nbands))
        psi_2 = loader.load(bands=(0, nb), k="full_bz")
        gv = jnp.asarray(np.asarray(loader.gvecs(k="full_bz")), dtype=jnp.int32)
        kv = jnp.asarray(np.asarray(loader.kvecs(k="full_bz"), dtype=np.float64))
        ngk = jnp.asarray(np.asarray(loader.ngk_valid(k="full_bz")), dtype=jnp.int32)
        hook = vnl_ops.nonlocal_velocity_lift(setup)
        all3 = np.asarray(hook(psi_2, gv, kv, ngk, channel=0))
        assert all3.shape == (3,) + tuple(np.asarray(psi_2).shape)
        for a in (1, 2, 3):
            one = np.asarray(hook(psi_2, gv, kv, ngk, channel=a))
            scale = max(float(np.max(np.abs(one))), 1e-300)
            np.testing.assert_allclose(all3[a - 1], one, rtol=0.0,
                                       atol=1e-13 * scale)
        assert float(np.max(np.abs(all3[0] - all3[1]))) > 1e-6 * float(
            np.max(np.abs(all3[0]))), "channels must differ (SOC is on)"
    finally:
        loader.close()


def test_lift_many_matches_the_per_channel_lifts():
    _fixture_or_skip()
    from psp import vnl_ops
    loader, sym, setup = _open_loader_with_setup()
    try:
        loader.nonlocal_velocity_lift = vnl_ops.nonlocal_velocity_lift(setup)
        nb = min(4, int(loader.nbands))
        psi_2 = loader.load(bands=(0, nb), k="full_bz")
        lifts = ("velocity_1", "velocity_2", "velocity_3", "raw")
        many = loader.lift_many(psi_2, k="full_bz", bispinor_lifts=lifts)
        assert len(many) == 4
        for lift, got in zip(lifts, many):
            want = np.asarray(loader.lift(psi_2, k="full_bz", bispinor_lift=lift))
            got = np.asarray(got)
            scale = max(float(np.max(np.abs(want))), 1e-300)
            np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-13 * scale)
        # the raw member is exact: no projector term enters it
        np.testing.assert_array_equal(
            np.asarray(many[3]), np.asarray(loader.lift(psi_2, k="full_bz")))
    finally:
        loader.close()


@pytest.mark.parametrize("k_chunk_size", [1, None])
def test_tuple_lift_centroid_load_matches_three_single_loads(k_chunk_size):
    """The one-pass streamed centroid load (parent route with k_chunk_size=1,
    direct band tiles with None) returns, per lift, what three single-lift
    loads return."""
    _fixture_or_skip()
    from jax.sharding import Mesh
    from common import Meta
    from common.wfn_transforms import load_centroids_band_chunked
    from psp import vnl_ops
    from wfn_loader import WfnLoader
    from psp.pseudos import load_pseudopotentials
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    loader = WfnLoader(str(WFN_PATH), mesh=mesh)
    try:
        if int(loader.nspinor) != 2:
            pytest.skip("test requires a 2-spinor WFN")
        sym = loader.symmetry()
        setup = vnl_ops.build_vnl_setup(
            loader, sym, None, load_pseudopotentials(str(FIXTURE_DIR)),
            nspinor=2, print_fn=lambda *a, **k: None)
        loader.nonlocal_velocity_lift = vnl_ops.nonlocal_velocity_lift(setup)
        nb = min(4, int(loader.nbands))
        nx, ny, nz = (int(s) for s in loader.fft_grid)
        r_mu = jnp.asarray([[0, 0, 0], [1 % nx, 2 % ny, 3 % nz],
                            [2 % nx, 1 % ny, 4 % nz]], dtype=jnp.int32)
        meta = Meta.from_system(loader, sym, nval=2, ncond=nb - 2, nband=nb,
                                n_rmu=int(r_mu.shape[0]), bispinor=True)
        meta.memory_per_device_gb = 1000.0
        lifts = ("velocity_1", "velocity_2", "velocity_3")
        with mesh:
            many = load_centroids_band_chunked(
                loader, sym, meta, r_mu, True, mesh, (0, nb),
                band_chunk_size=2, k_chunk_size=k_chunk_size,
                bispinor_lift=lifts)
            assert len(many) == 3
            for lift, (got_y, got_x) in zip(lifts, many):
                ref_y, ref_x = load_centroids_band_chunked(
                    loader, sym, meta, r_mu, True, mesh, (0, nb),
                    band_chunk_size=2, k_chunk_size=k_chunk_size,
                    bispinor_lift=lift)
                for got, ref in ((got_y, ref_y), (got_x, ref_x)):
                    got = np.asarray(got); ref = np.asarray(ref)
                    scale = max(float(np.max(np.abs(ref))), 1e-300)
                    np.testing.assert_allclose(got, ref, rtol=0.0,
                                               atol=1e-12 * scale)
    finally:
        loader.close()
