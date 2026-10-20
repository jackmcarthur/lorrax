"""The head's imaginary-axis moment M_1 and its pin (CPU, P1; unit cell, exempt from P4).

``M_1 = sum_p B_p/Omega_p^2 = (1/pi) int_0^inf dxi [W^c_h(i xi) - W^c_h(0+)]/xi^2``
(``gw.qsgw_head.head_moment_m1``) is the slope of the on-shell head,
``d Sigma_h/d omega = -M_1/(Omega N_k)``; the fit pins it beside ``W^c_h(0)``
(``gw.mpa.model._pin_static_head_sample``).

* (i) Insulator: a six-pole model with known ``B_p, Omega_p`` sampled on the
  MPA plan and the rule.  The quadrature reproduces ``M_1`` to 1e-8; N = 16
  and 32 agree to 1e-6; a three-pole fit that misses ``M_1`` by more than
  1e-5 carries it to 1e-8 after the pin, and the on-shell head (eqp0) is
  unchanged to 1e-12 eV.
* (ii) Metal: a two-band metal through the production head builders
  (``build_iteration_head_response``, shared-pole Gamma fold, cell average).
  ``W^c_h(i xi) + <v> = O(xi^2)`` (the dynamic model's own limit) while the
  Thomas-Fermi row is not ``-<v>``; ``M_1`` is finite, N = 16 and 32 agree to
  1e-6, it does not move with the Sobol count (2^14 vs 2^18), and
  ``gamma_0 -> 0`` is continuous with the undamped Drude head.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from ffi import _services  # noqa: E402

_services.ensure_on_path()

from common import RYD_TO_EV  # noqa: E402
from gw.gw_config import HeadCorrection  # noqa: E402
from gw.qsgw_head import head_moment_m1, head_quadrature_rule  # noqa: E402

NK, NB, MU, NS = 8, 2, 6, 1
A = 6.0
OMEGA = A ** 3
KGRID = (2, 2, 2)
KT = 5.0e-3


def _config(gamma0_ev=0.0):
    return SimpleNamespace(
        head=SimpleNamespace(correction=HeadCorrection.FULL, wcoul0_eta=0.0, vhead=None,
                             whead_0freq=None, whead_imfreq=None, analytic_q0_sphere=False,
                             head_minibz_average=True, uses_bgw_metal_q0shift=False,
                             drude_gamma0_ev=gamma0_ev, drude_gamma_width_ev=1.0),
        mpa=SimpleNamespace(metal_origin_shift_ry=None))


def _nodes(*orders):
    return tuple(1j * x for n in orders for x in head_quadrature_rule(n)[0])


# --------------------------------------------------------------------- (i)
OM = np.array([0.31, 0.47, 0.8, 1.25, 2.1, 6.0])
BP = np.array([0.9, 2.5, 6.0, 30.0, 8.0, 3.0])
VC0 = 1469.0


def _wc(z):
    z = np.asarray(z, np.complex128)
    return np.sum(2.0 * BP * OM / (z[:, None] ** 2 - OM ** 2), axis=1)


def _samples(z):
    return [SimpleNamespace(wcoul0=w + VC0, vc0=VC0) for w in _wc(z)]


def test_i_insulator_moment_quadrature_and_pin():
    from gw.head_correction import compute_complex_pole_head_sigma_diag
    from gw.mpa.model import fit_head_samples
    from gw.mpa.sampling import double_parallel_grid
    exact = float(np.sum(BP / OM ** 2))
    n_p = 3
    z = np.asarray(double_parallel_grid(n_p, 3.0, material_class="insulator",
                                        energy_unit="Ry"), np.complex128)
    assert np.any(z == 0)
    omegas = tuple(z) + _nodes(24, 16, 32)
    rows = _samples(omegas)
    m1 = {n: head_moment_m1(rows, omegas, metal=False, order=n) for n in (16, 24, 32)}
    assert abs(m1[24] / exact - 1.0) < 1.0e-8, m1
    assert abs(m1[16] / m1[32] - 1.0) < 1.0e-6, m1
    fits = {pin: fit_head_samples(rows[:z.size], z, n_p, model="qsgw_schur_loewner",
                                  solve="loewner", moment_m1=m1[24] if pin else None)
            for pin in (False, True)}
    unpinned = complex(fits[False]["provenance"]["moment_m1_unpinned"])
    assert abs(unpinned / exact - 1.0) > 1.0e-5, unpinned       # the fit alone misses M_1
    pinned = fits[True]
    assert abs(np.sum(pinned["B_p"] / pinned["Omega_p"] ** 2) / exact - 1.0) < 1.0e-8
    assert complex(pinned["provenance"]["moment_m1"]) == m1[24]
    # eqp0: the on-shell head is (1/2 - f) W^c_fit(0)/(Omega N_k), pinned in both fits.
    enk = np.array([[-0.3, -0.1, 0.25, 0.4]]); occ = np.array([[1.0, 1.0, 0.0, 0.0]])
    sig = [compute_complex_pole_head_sigma_diag(
        omega_grid_ry=enk[0], enk_ry=enk, efermi_ry=0.0, occupations=occ,
        poles_ry=f["Omega_p"], residues_ry=f["B_p"], cell_volume=OMEGA, nk_tot=NK)
        for f in fits.values()]
    on_shell = [np.einsum("iki->ki", s) for s in sig]
    assert float(np.max(np.abs(on_shell[1] - on_shell[0]))) * RYD_TO_EV < 1.0e-12
    assert float(np.max(np.abs(on_shell[0]))) > 1.0e-4                # a live head


# --------------------------------------------------------------------- (ii)
def _metal():
    rng = np.random.default_rng(3)
    e = np.stack([rng.uniform(-2.0e-3, 2.0e-3, NK), np.full(NK, 0.55)], axis=1)
    psi = (rng.normal(size=(NK, NS, MU, NB))
           + 1j * rng.normal(size=(NK, NS, MU, NB))) / np.sqrt(OMEGA)
    v = 0.3 * (rng.normal(size=(3, NK, NB, NB)) + 1j * rng.normal(size=(3, NK, NB, NB)))
    v = 0.5 * (v + np.conj(np.swapaxes(v, -1, -2)))
    h = NK // 2                                    # time reversal: k, -k
    e[h:], psi[h:], v[:, h:] = e[:h], np.conj(psi[:h]), -np.conj(v[:, :h])
    f = 1.0 / (1.0 + np.exp(e / KT))
    return e, f, f * (1.0 - f) / KT, psi, v.astype(np.complex128)


def _metal_head(mesh, config, omegas):
    import distrib_la
    from gw.qsgw_head import (QPVelocity, build_iteration_head_response,
                              finalize_iteration_head_rows, iteration_head_sample_terms,
                              metal_response_nodes)
    from gw.shared_pole_head import _gamma_body
    e, f, s, psi, v = _metal()
    wfn = SimpleNamespace(nspin=1, blat=2.0 * np.pi / A, bvec=np.eye(3), cell_volume=OMEGA)
    meta = SimpleNamespace(cell_volume=OMEGA, nk_tot=NK, nspinor_wfnfile=1,
                           nkx=KGRID[0], nky=KGRID[1], nkz=KGRID[2], sys_dim=3)
    wfns = SimpleNamespace(layout="face", psi_mun=jnp.asarray(psi),
                           psi_nmu=jnp.asarray(psi.transpose(0, 3, 1, 2)), enk=jnp.asarray(e),
                           occ=jnp.asarray(f), slices=None, green_parent=None)
    vel = jnp.asarray(v)
    response = build_iteration_head_response(
        QPVelocity(dft_cart=vel, base=vel, correction=None, bound=None, zeroed=None,
                   label="model"),
        jnp.broadcast_to(jnp.eye(NB, dtype=jnp.complex128), (NK, NB, NB)),
        e, f, np.asarray(omegas), surface_weight_qp_kn=s, mesh=mesh, kgrid=KGRID,
        bvec_cart=wfn.blat * wfn.bvec, nb_logical=NB, sigma_energies_ry=e, efermi_ry=0.0,
        wfn=wfn, meta=meta, config=config, wfns_qp=wfns)
    rng = np.random.default_rng(11)
    b = 0.2 * rng.normal(size=(1, MU, 4)).astype(np.complex128)
    m = rng.normal(size=(MU, MU))
    face, rep = NamedSharding(mesh, P(None, "x", "y")), NamedSharding(mesh, P())
    ev = _gamma_body(mesh, ("off", distrib_la.ROUTE_BATCH_RESHARD), False)
    args = (jax.device_put(jnp.asarray(b), face),
            jax.device_put(jnp.asarray(rng.uniform(0.3, 2.0, (1, 4))), rep),
            jax.device_put(jnp.asarray(np.array([4])), rep),
            jax.device_put(jnp.asarray(0.5 * (m @ m.T / MU + np.eye(MU))[None]
                                       .astype(np.complex128)), face))
    nodes = metal_response_nodes(omegas, config, metal=True)
    terms = [iteration_head_sample_terms(
        response, i, ev(jnp.asarray(zi, jnp.complex128), *args)[0], meta=meta,
        config=config, mesh=mesh) for i, zi in enumerate(nodes)]
    return finalize_iteration_head_rows(response, terms, wfn=wfn, meta=meta, config=config)


def test_ii_metal_moment_is_finite_converged_and_draw_independent(monkeypatch):
    import gw.vcoul as vcoul
    from gw.mpa.sampling import double_parallel_grid
    from gw.qsgw_head import drude_memory_damping
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    plan = tuple(np.asarray(double_parallel_grid(2, 1.0, material_class="metal",
                                                 energy_unit="Ry"), np.complex128))
    omegas = plan + _nodes(16, 32)

    screened = vcoul.compute_q0_averages_screened       # nsamples = 2^18 by default

    def m1(gamma0_ev=0.0, nsamples=None, orders=(16, 32)):
        if nsamples is not None:
            monkeypatch.setattr(vcoul, "compute_q0_averages_screened",
                                lambda *a, **k: screened(*a, **dict(k, nsamples=nsamples)))
        rows = _metal_head(mesh, _config(gamma0_ev), omegas)
        monkeypatch.undo()
        return rows, {n: head_moment_m1(rows, omegas, metal=True, order=n) for n in orders}

    rows, base = m1()
    wc = np.array([complex(r.wcoul0) - complex(r.vc0) for r in rows])
    vc0 = complex(rows[0].vc0).real
    xi16 = head_quadrature_rule(16)[0]
    # The dynamic model's own xi -> 0 limit is -<v>: W^c + <v> = O(xi^2) at the
    # two smallest nodes; the Thomas-Fermi origin row is a different limit.
    small = (wc[len(plan):len(plan) + 2] + vc0).real
    assert abs(small[0] / small[1] / (xi16[0] / xi16[1]) ** 2 - 1.0) < 0.05, small
    assert abs(wc[0] + vc0) / vc0 > 1.0e-4, (wc[0], vc0)
    assert np.isfinite(base[16]) and base[16].real > 0.0, base
    assert abs(base[16] / base[32] - 1.0) < 1.0e-6, base
    _, coarse = m1(nsamples=2 ** 14, orders=(32,))
    assert abs(coarse[32] / base[32] - 1.0) < 1.0e-3, (coarse, base)
    # gamma_0 = 0 is the undamped Drude head; small gamma_0 is continuous with it.
    assert drude_memory_damping(0.5j, _config(0.0)) is None
    _, tiny = m1(gamma0_ev=1.0e-6, orders=(32,))
    _, damped = m1(gamma0_ev=0.3, orders=(32,))
    assert abs(tiny[32] / base[32] - 1.0) < 1.0e-5, (tiny, base)
    assert abs(damped[32] / base[32] - 1.0) > 1.0e-3, (damped, base)
