"""The full Gamma head of an ordered (time-reversal-broken) METAL (CPU, P1 and P4).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
A four-band metal: one band below, one above, and a pair at E_F split by
``Delta`` (3e-5, 1e-5, 3e-6 Ry, all above the 1e-6 Ry degeneracy cut).  The
head is built by the production builder (``qsgw_head.build_iteration_head_response``)
on the metal MPA plan and folded with a shared-pole Gamma body
(``shared_pole_head._gamma_body``) on the shared-pole sample path
(``iteration_head_sample_terms`` + ``finalize_iteration_head_rows``).

* (a) The folded ``S_eff`` and ``W^c_h`` at every plan node do not depend on
  ``Delta`` as ``Delta -> 0``.  Without the pair split (cell moment ``Q = 0``)
  the static wing of the pair grows as ``1/Delta``; and for ``|z| >> Delta``
  a split pair's wing does not depend on its share ``phi`` (the intraband wing
  is the pair weight's own ``dE -> 0`` limit, with its sign).
* (b) A time-reversal-even model folded with the ordered body equals the TRS
  route at every node to 1e-12.
* (c) ``W^c_h`` at the shifted origin is the folded Thomas-Fermi cell value
  ``<8 pi/(q.eps_inf.q + kappa_eff^2)> - <v>``, built here in numpy, to 1e-10.
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

NK, NB, MU, NS = 8, 4, 6, 1
A = 6.0
OMEGA = A ** 3
KGRID = (2, 2, 2)
KT = 5.0e-3
DELTAS = (3.0e-5, 1.0e-5, 3.0e-6)
TOL_DEG = 1.0e-6


def _meshes():
    out = [Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))]
    if len(jax.devices()) >= 4:
        out.append(Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y")))
    return out


def _plan():
    from gw.mpa.sampling import double_parallel_grid
    return double_parallel_grid(2, 1.0, material_class="metal", energy_unit="Ry")


def _model(delta, *, trs, couple_outer=True, seed=7):
    """Energies, FD occupations, -df/dE, centroid psi and a Hermitian velocity."""
    rng = np.random.default_rng(seed)
    center = rng.uniform(-2.0e-3, 2.0e-3, NK)
    e = np.stack([np.full(NK, -0.45), center - 0.5 * delta, center + 0.5 * delta,
                  np.full(NK, 0.55)], axis=1)
    psi = (rng.normal(size=(NK, NS, MU, NB))
           + 1j * rng.normal(size=(NK, NS, MU, NB))) / np.sqrt(OMEGA)
    v = 0.3 * (rng.normal(size=(3, NK, NB, NB)) + 1j * rng.normal(size=(3, NK, NB, NB)))
    v = 0.5 * (v + np.conj(np.swapaxes(v, -1, -2)))
    if not couple_outer:                       # only the split pair carries a vertex
        mask = np.zeros((NB, NB), bool); mask[1:3, 1:3] = True
        v = np.where(mask, v, 0.0)
    if trs:                                    # k and kbar = -k: psi* , -v*, same E
        h = NK // 2
        e[h:], psi[h:], v[:, h:] = e[:h], np.conj(psi[:h]), -np.conj(v[:, :h])
    f = 1.0 / (1.0 + np.exp(e / KT))
    s = f * (1.0 - f) / KT
    return e, f, s, psi, v.astype(np.complex128)


def _setup():
    from gw.gw_config import HeadCorrection
    wfn = SimpleNamespace(nspin=1, blat=2.0 * np.pi / A, bvec=np.eye(3), cell_volume=OMEGA)
    meta = SimpleNamespace(cell_volume=OMEGA, nk_tot=NK, nspinor_wfnfile=1,
                           nkx=KGRID[0], nky=KGRID[1], nkz=KGRID[2], sys_dim=3)
    config = SimpleNamespace(
        head=SimpleNamespace(correction=HeadCorrection.FULL, wcoul0_eta=0.0, vhead=None,
                             whead_0freq=None, whead_imfreq=None,
                             analytic_q0_sphere=False, head_minibz_average=True),
        mpa=SimpleNamespace(metal_origin_shift_ry=None))
    return wfn, meta, config


def _wfns(e, f, psi):
    return SimpleNamespace(layout="face", psi_mun=jnp.asarray(psi),
                           psi_nmu=jnp.asarray(psi.transpose(0, 3, 1, 2)),
                           enk=jnp.asarray(e), occ=jnp.asarray(f), slices=None,
                           green_parent=None)


def _response(mesh, model, z):
    from gw.qsgw_head import QPVelocity, build_iteration_head_response
    e, f, s, psi, v = model
    wfn, meta, config = _setup()
    vel = jnp.asarray(v)
    velocity = QPVelocity(dft_cart=vel, base=vel, correction=None, bound=None,
                          zeroed=None, label="model")
    return build_iteration_head_response(
        velocity, jnp.broadcast_to(jnp.eye(NB, dtype=jnp.complex128), (NK, NB, NB)),
        e, f, z, surface_weight_qp_kn=s, mesh=mesh, kgrid=KGRID,
        bvec_cart=wfn.blat * wfn.bvec, nb_logical=NB, sigma_energies_ry=e,
        efermi_ry=0.0, wfn=wfn, meta=meta, config=config, wfns_qp=_wfns(e, f, psi))


def _store(*, real):
    rng = np.random.default_rng(11)
    b = 0.2 * (rng.normal(size=(1, MU, 4)) + (0 if real else 1j) * rng.normal(size=(1, MU, 4)))
    poles = rng.uniform(0.3, 2.0, (1, 4))
    m = rng.normal(size=(MU, MU))
    V = 0.5 * (m @ m.T / MU + np.eye(MU))
    return b.astype(np.complex128), poles, np.array([4]), V[None].astype(np.complex128)


def _body(mesh, ordered, store):
    import distrib_la
    from gw.shared_pole_head import _gamma_body
    face, rep = NamedSharding(mesh, P(None, "x", "y")), NamedSharding(mesh, P())
    b, poles, counts, V = store
    ev = _gamma_body(mesh, ("off", distrib_la.ROUTE_BATCH_RESHARD), ordered)
    args = (jax.device_put(jnp.asarray(b), face), jax.device_put(jnp.asarray(poles), rep),
            jax.device_put(jnp.asarray(counts), rep), jax.device_put(jnp.asarray(V), face))
    return lambda z: ev(jnp.asarray(z, jnp.complex128), *args)[0]


def _head(mesh, model, *, ordered, store):
    """Folded S_eff (n_z, 3, 3) and W^c_h (n_z,) on the shared-pole sample path."""
    from gw.qsgw_head import (finalize_iteration_head_rows, iteration_head_sample_terms,
                              metal_response_nodes)
    wfn, meta, config = _setup()
    z = _plan()
    response = _response(mesh, model, z)
    body = _body(mesh, ordered, store)
    # The shared-pole builder's body nodes (build_shared_pole_head).
    nodes = metal_response_nodes(z, config, metal=True)
    terms = [iteration_head_sample_terms(response, i, body(zi), meta=meta, config=config,
                                         mesh=mesh) for i, zi in enumerate(nodes)]
    samples = finalize_iteration_head_rows(response, terms, wfn=wfn, meta=meta, config=config)
    S = np.stack([t["S_effective"] for t in terms])
    Wc = np.array([complex(x.wcoul0) - complex(x.vc0) for x in samples])
    return S, Wc, response, terms, samples


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _split(model, moment):
    from gw.qsgw_head import MetalPairSplit
    diag = np.real(np.einsum("akii->aki", model[4]))
    return MetalPairSplit(diag=None, moment=np.asarray(moment, float), diag_host=diag)


def _wings(mesh, model, z, moment):
    from gw.qsgw_head import head_wings_sharded
    e, f, s, psi, v = model
    Y, _ = head_wings_sharded(jnp.asarray(v), _wfns(e, f, psi), jnp.asarray(e),
                              jnp.asarray(f), jnp.asarray(z), mesh=mesh, nb_logical=NB,
                              nk_tot=NK, nspin=1, nspinor=NS, surface_weight_kn=s,
                              pair_split=_split(model, moment))
    return np.asarray(Y)


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_a_folded_head_is_independent_of_the_pair_split(mesh):
    store = _store(real=False)
    heads = {d: _head(mesh, _model(d, trs=False), ordered=True, store=store)[:2]
             for d in DELTAS}
    S_ref, W_ref = heads[DELTAS[-1]]
    spread = {d: (_rel(heads[d][0], S_ref), _rel(heads[d][1], W_ref)) for d in DELTAS[:-1]}
    assert max(max(x) for x in spread.values()) < 1.0e-3, spread
    assert max(spread[DELTAS[1]]) < max(spread[DELTAS[0]]), spread   # converging
    from vcoul import minibz_coulomb_moment
    Q = minibz_coulomb_moment(2.0 * np.pi / A * np.eye(3), KGRID)
    # Sensitivity: without the split (Q = 0) the pair's static wing is ~1/Delta.
    y0 = [_wings(mesh, _model(d, trs=False), [0.0], np.zeros((3, 3)))[0] for d in DELTAS]
    assert _rel(y0[2], y0[0]) > 1.0, _rel(y0[2], y0[0])
    # For |z| >> Delta a pair's wing does not depend on phi: Q (phi ~ 1) and
    # Q = 0 (phi = 0) agree to O(Delta/|z|); a wrong-signed intraband wing
    # would differ by the factor 2 phi - 1.
    pair = _model(DELTAS[0], trs=False, couple_outer=False)
    zf = [2.0j, 0.3 + 0.2j]
    assert _rel(_wings(mesh, pair, zf, Q), _wings(mesh, pair, zf, np.zeros((3, 3)))) < 1.0e-4


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_b_trs_model_stored_ordered_reproduces_the_trs_route(mesh):
    store = _store(real=True)
    model = _model(DELTAS[1], trs=True)
    S_o, W_o = _head(mesh, model, ordered=True, store=store)[:2]
    S_t, W_t = _head(mesh, model, ordered=False, store=store)[:2]
    assert _rel(S_o, S_t) < 1.0e-12, _rel(S_o, S_t)
    assert _rel(W_o, W_t) < 1.0e-12, _rel(W_o, W_t)


def _numpy_static_row(model, W, moment):
    """S_eff(0) and kappa_eff^2 of the static row from the Lehmann sums (numpy)."""
    e, f, s, psi, v = model
    dE = e[:, :, None] - e[:, None, :]
    fd = f[:, None, :] - f[:, :, None]
    diag = np.real(np.einsum("akii->aki", v))
    dv = diag[:, :, None, :] - diag[:, :, :, None]
    ext = (np.einsum("ab,akxy,bkxy->kxy", moment, dv, dv)
           + 4.0 * np.real(np.einsum("ab,akxy,bkxy->kxy", moment, np.conj(v), v)))
    surf = (s[:, :, None] + s[:, None, :]) > 0
    phi = np.where(np.abs(dE) < TOL_DEG, 1.0, np.where(surf, ext / (dE ** 2 + ext), 0.0))
    keep = 1.0 - phi
    c = 4.0 / NK
    up = dE >= TOL_DEG
    S = (c / OMEGA) * np.einsum("akxy,kxy,bkxy->ab", np.conj(v),
                                np.where(up, keep * fd / np.where(up, -dE ** 3, 1.0), 0.0), v)
    rho = np.einsum("ksmx,ksmy->kxym", np.conj(psi), psi)
    off = dE != 0
    g = np.where(off, keep * 0.5 * c * fd / np.where(off, dE ** 2, 1.0), 0.0)
    Y = np.einsum("akxy,kxy,kxym->am", np.conj(v), g, rho)
    Z = np.einsum("akxy,kxy,kxym->ma", v, g, np.conj(rho))
    S_eff = S + Y @ W @ Z / OMEGA
    C = -(2.0 / NK) * np.einsum("kn,ksmn->m", s, np.abs(psi) ** 2)
    kappa2 = 8.0 * np.pi * 2.0 * s.sum() / NK / OMEGA
    f00 = -kappa2 / (8.0 * np.pi) + C @ W @ C / OMEGA
    return S_eff, -8.0 * np.pi * float(np.real(f00))


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_c_shifted_origin_is_the_folded_thomas_fermi_row(mesh):
    from gw.qsgw_head import _extra_chi_rows, metal_static_rows
    from gw.vcoul import compute_q0_averages_screened
    from vcoul import minibz_coulomb_moment
    wfn, meta, config = _setup()
    store = _store(real=False)
    model = _model(DELTAS[1], trs=False)
    S, Wc, response, terms, samples = _head(mesh, model, ordered=True, store=store)
    z = _plan()
    origin = int(np.flatnonzero(metal_static_rows(z, config))[0])
    assert abs(z[origin] - 2.0e-5j) < 1e-18 and terms[origin]["static_row"]
    W = np.asarray(_body(mesh, True, store)(0.0))
    Q = minibz_coulomb_moment(2.0 * np.pi / A * np.eye(3), KGRID)
    S_np, kappa2_np = _numpy_static_row(model, W, Q)
    assert _rel(S[origin], S_np) < 1.0e-12, _rel(S[origin], S_np)
    vc0, (wc0,) = compute_q0_averages_screened(
        wfn, meta, [S_np], extra_chi_rows=_extra_chi_rows(
            [("constant", -kappa2_np / (8.0 * np.pi))], None))
    want = complex(wc0) - complex(vc0)
    assert abs(Wc[origin] - want) <= 1.0e-10 * max(1.0, abs(want)), (Wc[origin], want)
    # The row is Thomas-Fermi, W_h = <8 pi/(q.eps.q + kappa^2)> <= 8 pi/kappa^2,
    # not the Drude-remainder limit W_h -> 0 (W^c_h -> -<v>).
    w_h = (Wc[origin] + complex(vc0)).real
    assert 0.3 < w_h * kappa2_np / (8.0 * np.pi) <= 1.0, (w_h, kappa2_np)
