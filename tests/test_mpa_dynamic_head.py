"""Two independent small oracles for the scalar dynamic MPA head."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.head_correction import (
    HeadGNParams,
    compute_complex_pole_head_sigma_diag,
    compute_ppm_head_sigma_diag,
    fold_cartesian_head_wings_sharded,
)
from gw.qsgw_head import (
    _HEAD_WING_FREQUENCY_BLOCK,
    _fold_static_kappa2,
    _pad_head_band_manifold,
    _s_tensor_kernel,
)


def _complex(rng, shape, scale=1.0):
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def test_cartesian_ywz_fold_matches_dense_bordered_dyson():
    """The reduced 3x3 head has the dense Dyson sign and one 1/V factor."""
    rng = np.random.default_rng(418)
    n_z, n_mu = 2, 5
    cell_volume = 97.0
    lam = 0.05
    qhat = np.asarray([0.31, -0.48, 0.82])
    qhat /= np.linalg.norm(qhat)
    vbare = 8.0 * np.pi / lam**2

    V_body = np.empty((n_z, n_mu, n_mu), dtype=np.complex128)
    chi_body = np.empty_like(V_body)
    W_body = np.empty_like(V_body)
    for iz in range(n_z):
        a = _complex(rng, (n_mu, n_mu), 0.15)
        V_body[iz] = a @ a.conj().T + 0.4 * np.eye(n_mu)
        b = _complex(rng, (n_mu, n_mu), 0.08)
        chi_body[iz] = -(b @ b.conj().T)
        W_body[iz] = np.linalg.solve(
            np.eye(n_mu) - V_body[iz] @ chi_body[iz], V_body[iz])

    S_direct = _complex(rng, (n_z, 3, 3), 2.0e-3)
    Y = _complex(rng, (n_z, 3, n_mu), 8.0e-3)
    Z = _complex(rng, (n_z, n_mu, 3), 8.0e-3)

    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, axis_names=("x", "y"))
    got_S = np.asarray(fold_cartesian_head_wings_sharded(
        jnp.asarray(S_direct), jnp.asarray(Y), jnp.asarray(W_body),
        jnp.asarray(Z), cell_volume, mesh_xy=mesh))

    for iz in range(n_z):
        chi_aug = np.zeros((n_mu + 1, n_mu + 1), dtype=np.complex128)
        # compute_S_omega's direct tensor already owns 1/V_cell; the
        # transition-space bordered matrix uses its unnormalised numerator.
        chi_aug[0, 0] = (
            lam**2 * cell_volume * (qhat @ S_direct[iz] @ qhat))
        chi_aug[0, 1:] = lam * (qhat @ Y[iz])
        chi_aug[1:, 0] = lam * (Z[iz] @ qhat)
        chi_aug[1:, 1:] = chi_body[iz]

        V_aug = np.zeros_like(chi_aug)
        V_aug[0, 0] = vbare / cell_volume
        V_aug[1:, 1:] = V_body[iz]
        W_aug = np.linalg.solve(
            np.eye(n_mu + 1) - V_aug @ chi_aug, V_aug)

        dense_head = cell_volume * W_aug[0, 0]
        reduced_head = vbare / (
            1.0 - vbare * lam**2 * (qhat @ got_S[iz] @ qhat))
        np.testing.assert_allclose(reduced_head, dense_head, rtol=2e-13,
                                   atol=2e-11)


def test_cartesian_ywz_fold_keeps_body_tiled_on_2d_mesh():
    """The production fold reduces local tiles, never a replicated body."""
    if len(jax.devices()) < 4:
        import pytest
        pytest.skip("requires four CPU test devices")

    rng = np.random.default_rng(207)
    n_z, n_mu = 2, 12
    volume = 83.0
    direct = _complex(rng, (n_z, 3, 3))
    left = _complex(rng, (n_z, 3, n_mu))
    body = _complex(rng, (n_z, n_mu, n_mu))
    right = _complex(rng, (n_z, n_mu, 3))
    expected = direct + np.einsum(
        "...am,...mn,...nb->...ab", left, body, right,
        optimize=True) / volume

    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    put = jax.device_put
    got = fold_cartesian_head_wings_sharded(
        put(direct, NamedSharding(mesh, P(None, None, None))),
        put(left, NamedSharding(mesh, P(None, None, "x"))),
        put(body, NamedSharding(mesh, P(None, "x", "y"))),
        put(right, NamedSharding(mesh, P(None, "y", None))),
        volume,
        mesh_xy=mesh,
    )

    assert got.sharding.spec == P(None, None, None)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=3e-13,
                               atol=3e-13)


def test_static_schur_fold_updates_thomas_fermi_kappa():
    """The scalar static wing fold feeds the screened mini-BZ parameter."""
    from types import SimpleNamespace

    kappa2 = 0.72
    volume = 91.0
    left = jnp.asarray([-0.08, 0.03, -0.04])
    right = jnp.asarray([-0.07, 0.02, -0.05])
    body = jnp.asarray([
        [1.4, 0.1, 0.0],
        [0.1, 1.1, 0.2],
        [0.0, 0.2, 0.9],
    ], dtype=jnp.complex128)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    response = SimpleNamespace(
        static_kappa2_bohr2=kappa2,
        static_Y_x=left,
        static_Z_y=right,
    )
    got = _fold_static_kappa2(response, body, volume, mesh)
    f_direct = -kappa2 / (8.0 * np.pi)
    f_effective = f_direct + np.asarray(left @ body @ right) / volume
    want = -8.0 * np.pi * f_effective.real
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)


def test_complex_pole_head_sum_matches_ppm_and_direct_denominators():
    """One real PPM pole preserves bytes; two damped poles pin width signs."""
    omega = np.asarray([-0.7, 0.0, 0.4, 1.1])
    enk = np.asarray([[-0.8, -0.1, 0.25], [-0.65, 0.05, 0.6]])
    efermi = -0.03
    cell_volume, nk_tot, eta = 83.0, 8, 2.0e-5
    head = HeadGNParams(
        omega_h_sq=1.44, omega_h=1.2, B_h=3.6, R_h=1.5,
        wc_head_0=-2.5, wc_head_iwp=-1.0, vc0=30.0, omega_p=0.5)

    got_ppm = compute_ppm_head_sigma_diag(
        head, omega_grid_ry=omega, enk_ry=enk, efermi_ry=efermi,
        n_occ=1, cell_volume=cell_volume, nk_tot=nk_tot, eta=eta)
    eps_rel = enk - float(efermi)
    f = np.asarray([1.0, 0.0, 0.0])
    delta = omega[:, None, None] - eps_rel[None, :, :]
    occ_term = f[None, None, :] / (delta + head.omega_h - 1j * eta)
    emp_term = ((1.0 - f[None, None, :])
                / (delta - head.omega_h + 1j * eta))
    legacy = ((head.R_h / (float(cell_volume) * float(nk_tot)))
              * (occ_term + emp_term))
    assert np.array_equal(got_ppm, np.asarray(legacy, dtype=np.complex128))

    occupations = np.asarray([[1.0, 0.7, 0.0], [1.0, 0.2, 0.0]])
    poles = np.asarray([0.65 - 0.04j, 1.7 - 0.22j])
    residues = np.asarray([0.8 + 0.15j, -0.25 + 0.4j])
    got = compute_complex_pole_head_sigma_diag(
        omega_grid_ry=omega, enk_ry=enk, efermi_ry=efermi,
        occupations=occupations, poles_ry=poles, residues_ry=residues,
        cell_volume=cell_volume, nk_tot=nk_tot)
    direct = np.zeros_like(got)
    for pole, residue in zip(poles, residues):
        direct += residue / (cell_volume * nk_tot) * (
            occupations[None, :, :] / (delta + pole)
            + (1.0 - occupations[None, :, :]) / (delta - pole))
    np.testing.assert_allclose(got, direct, rtol=2e-15, atol=2e-17)

    wrong_width_sign = compute_complex_pole_head_sigma_diag(
        omega_grid_ry=omega, enk_ry=enk, efermi_ry=efermi,
        occupations=occupations, poles_ry=poles.conj(),
        residues_ry=residues, cell_volume=cell_volume, nk_tot=nk_tot)
    assert np.max(np.abs(got - wrong_width_sign)) > 1.0e-5


def _s_tensor_reference(v, e, f, omegas, pref, eta, nb_logical):
    """Unblocked ``jax.vmap`` reference -- the PRE-FIX kernel body, kept
    here (not imported) so the A/B does not depend on the code under test.
    """
    vj = jnp.asarray(v, dtype=jnp.complex128)
    ej = jnp.asarray(e, dtype=jnp.float64)
    fj = jnp.asarray(f, dtype=jnp.float64)
    dE = ej[:, :, None] - ej[:, None, :]
    f_diff = fj[:, None, :] - fj[:, :, None]
    nb = v.shape[-1]
    ix = jnp.arange(nb)
    logical = ((ix[:, None] < nb_logical) & (ix[None, :] < nb_logical))[None]
    transition = logical & (dE > 0.0)

    def _one(omega):
        z = omega + 1j * eta
        denom = dE * (z * z - dE * dE)
        weight = jnp.where(
            transition & (jnp.abs(denom) > 1e-16),
            pref * f_diff / denom, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        return jnp.einsum(
            "akij,kij,bkij->ab", jnp.conj(vj), weight, vj, optimize=True)

    return jax.vmap(_one)(jnp.asarray(omegas, dtype=jnp.complex128))


def test_s_tensor_kernel_streamed_matches_unblocked_reference():
    """``_s_tensor_kernel`` streams frequency in blocks of
    ``_HEAD_WING_FREQUENCY_BLOCK`` (fix/head-fold-streamed-2026-08-22: it
    previously ran ``jax.vmap(_one)(omegas)`` over the FULL omega axis at
    once, the one frequency-by-band-pair temporary
    ``_head_wing_kernel``'s own comment names as the sibling kernel's
    bounded twin -- this one was the unbounded half).  Exercise n_omega
    below, at, and straddling the block boundary (1, block, block+1, and a
    multiple) and require value-level (not necessarily bit-exact, since
    blocking reassociates the frequency-batched reduction) agreement with
    the unblocked reference.
    """
    if len(jax.devices()) < 4:
        import pytest
        pytest.skip("requires four CPU test devices")

    rng = np.random.default_rng(2026_08_22)
    nk, nb, nb_logical = 5, 6, 6
    v = _complex(rng, (3, nk, nb, nb), 1.0)
    v = 0.5 * (v + np.conj(np.swapaxes(v, -1, -2)))
    e = np.sort(rng.standard_normal((nk, nb)), axis=-1)
    occ = np.where(np.arange(nb) < nb // 2, 1.0, 0.0)
    f = np.broadcast_to(occ, (nk, nb)).copy()
    pref, eta = 0.41, 0.015

    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    put = jax.device_put
    v_sh = put(jnp.asarray(v, dtype=jnp.complex128),
               NamedSharding(mesh, P(None, None, "x", "y")))
    e_x = put(jnp.asarray(e), NamedSharding(mesh, P(None, "x")))
    e_y = put(jnp.asarray(e), NamedSharding(mesh, P(None, "y")))
    f_x = put(jnp.asarray(f), NamedSharding(mesh, P(None, "x")))
    f_y = put(jnp.asarray(f), NamedSharding(mesh, P(None, "y")))
    kernel = _s_tensor_kernel(mesh, nb_logical=nb_logical)

    block = _HEAD_WING_FREQUENCY_BLOCK
    for n_omega in sorted({1, block, block + 1, 2 * block + 3}):
        rng2 = np.random.default_rng(100 + n_omega)
        omegas = rng2.uniform(0.0, 3.0, n_omega)
        ref = np.asarray(jax.device_get(
            _s_tensor_reference(v, e, f, omegas, pref, eta, nb_logical)))
        got = np.asarray(jax.device_get(kernel(
            v_sh, e_x, e_y, f_x, f_y,
            jnp.asarray(omegas, dtype=jnp.complex128),
            jnp.asarray(pref, dtype=jnp.complex128), jnp.asarray(eta))))
        assert got.shape == (n_omega, 3, 3)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_s_tensor_kernel_temp_size_is_bounded_past_one_frequency_block():
    """Red twin: compiled peak must NOT keep growing once ``n_omega``
    exceeds one frequency block -- this is the exact regression the
    pre-fix ``jax.vmap(_one)(omegas)`` body could not pass (its compiled
    temp scaled linearly in ``n_omega`` with no cap; measured directly at
    production MoS2 9x9x1 shape via ``.lower().compile().memory_analysis()``
    on 16 REAL A100 processes: unbounded growth is not reachable from a 4-
    fake-CPU-device unit test at this array size, so this asserts the
    SHAPE of the scaling law -- flat past one block -- which the
    production-scale probe in
    ``runs/MoS2/86_bgw_lorrax_scaling_20260819/points/k9_c600_integ/
    lorrax/attempts/headfold_hlo_probe_20260822/probe_bounded_check.py``
    confirmed numerically (5.84 GiB temp flat across n_omega in
    {16, 64, 256} after this fix, vs. linear growth before it).
    """
    if len(jax.devices()) < 4:
        import pytest
        pytest.skip("requires four CPU test devices")

    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    nk, nb_pad, nb_logical = 9, 8, 8
    block = _HEAD_WING_FREQUENCY_BLOCK

    def sds(shape, dtype, spec):
        return jax.ShapeDtypeStruct(
            shape, dtype, sharding=NamedSharding(mesh, spec))

    c128, f64 = jnp.complex128, jnp.float64
    v = sds((3, nk, nb_pad, nb_pad), c128, P(None, None, "x", "y"))
    e_x = sds((nk, nb_pad), f64, P(None, "x"))
    e_y = sds((nk, nb_pad), f64, P(None, "y"))
    f_x = sds((nk, nb_pad), f64, P(None, "x"))
    f_y = sds((nk, nb_pad), f64, P(None, "y"))
    pref = sds((), c128, P())
    eta = sds((), f64, P())
    kernel = _s_tensor_kernel(mesh, nb_logical=nb_logical)

    def compiled_peak(n_omega):
        omega = sds((n_omega,), c128, P(None))
        ma = kernel.lower(v, e_x, e_y, f_x, f_y, omega, pref, eta) \
                   .compile().memory_analysis()
        return (ma.temp_size_in_bytes + ma.argument_size_in_bytes
                + ma.output_size_in_bytes - ma.alias_size_in_bytes)

    peak_one_block = compiled_peak(block)
    peak_many_blocks = compiled_peak(8 * block)
    # A still-unbounded kernel would be ~8x bigger here; an actually
    # bounded one grows only by the tiny (n_omega, 3, 3) output/argument
    # tail. 2x is a generous ceiling that fails loudly if the block cap
    # regresses back to a single ``vmap`` over the whole axis.
    assert peak_many_blocks < 2.0 * peak_one_block, (
        f"_s_tensor_kernel compiled peak grew {peak_many_blocks/peak_one_block:.2f}x "
        f"from n_omega={block} to n_omega={8 * block} "
        f"({peak_one_block} -> {peak_many_blocks} bytes) -- the frequency "
        "block cap is not bounding memory; see "
        "KNOWN_LORRAX_ISSUES.md 'the bounded full-head fold still needs a "
        "fresh-fit lifetime boundary'."
    )


def test_pad_head_band_manifold_commits_v_to_the_declared_mesh_sharding():
    """Red twin: every ``_pad_head_band_manifold`` caller
    (``head_s_tensor_sharded``, ``head_wings_sharded``,
    ``head_drude_tensor_sharded``) builds ``v`` from ``velocity_cart`` via
    a bare ``jnp.asarray(...)`` on a freshly host-read dipole array --
    never sharded, never committed.  Before this fix that uncommitted,
    single-device array flowed straight into a ``shard_map``-wrapped
    ``jax.jit`` declaring ``P(None, None, 'x', 'y')`` for it, on every
    call, fresh fit or restart alike.  On the production MoS2
    9x9x1/626-band/mu=5288 shape (P=16) that dispatch-time reshard was
    measured requesting a single 81.74 GiB allocation at the
    ``block_until_ready`` in ``build_dft_head_response`` -- ~10x one full
    ``(nk,ns,mu,nb)`` psi copy, against a compile-only peak of 5.98 GiB
    for the same two kernels when every input already carries the
    declared sharding.  A P=4 fresh-vs-restart A/B on this branch (2026-
    08-22, ``runs/MoS2/86_bgw_lorrax_scaling_20260819/points/k6_c600/
    lorrax/attempts/head_restart_diag_20260822/``) showed the restart
    loader delivers ψ in the identical contract the fresh path does --
    restart is not at fault; it is simply the only path that survives far
    enough at production scale to reach this call, since the fresh zeta
    fit OOMs earlier on an unrelated binder.  This test would have failed
    before the fix (``v`` stayed a ``SingleDeviceSharding``,
    ``committed=False``) and passes after it (``v`` is
    ``jax.device_put`` onto the mesh before any kernel sees it).
    """
    if len(jax.devices()) < 4:
        import pytest
        pytest.skip("requires four CPU test devices")

    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(2026_08_22)
    nk, nb = 5, 6
    v_np = _complex(rng, (3, nk, nb, nb), 1.0)
    e_np = np.sort(rng.standard_normal((nk, nb)), axis=-1)
    f_np = np.broadcast_to(
        np.where(np.arange(nb) < nb // 2, 1.0, 0.0), (nk, nb)).copy()

    # Exactly how every caller builds these today: bare jnp.asarray on a
    # host numpy array, with no sharding placement at all.
    v = jnp.asarray(v_np, dtype=jnp.complex128)
    e = jnp.asarray(e_np, dtype=jnp.float64)
    f = jnp.asarray(f_np, dtype=jnp.float64)
    surface = jnp.zeros_like(e)
    assert not v.committed, (
        "test setup drifted: v must start life uncommitted/single-device "
        "to exercise the reshard-on-dispatch path this test guards.")

    v_out, _e_out, _f_out, _surf_out = _pad_head_band_manifold(
        v, e, f, surface, mesh=mesh)

    want = NamedSharding(mesh, P(None, None, "x", "y"))
    assert v_out.committed, (
        "_pad_head_band_manifold returned v uncommitted -- the kernels "
        "it feeds will dispatch a foreign single-device sharding again.")
    assert v_out.sharding == want, (
        f"_pad_head_band_manifold returned v sharded {v_out.sharding!r}, "
        f"want {want!r}.")
    np.testing.assert_allclose(
        np.asarray(jax.device_get(v_out))[:, :, :nb, :nb], v_np, rtol=0, atol=0)
