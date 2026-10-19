"""The q -> 0 head of a non-thermal population (td_agw_dynamical.md §1.4 "Head", §9 item 7), CPU.

``gw.qsgw_head.population_head_response`` gives the direct head of a population
table f from a run's response at f_ref: S(z; f) = S(z; f_ref) + S(z; f - f_ref)
plus the k.p band-curvature Drude term D(f - f_ref)/z^2 at every z != 0
(``population_drude_tensor_sharded``).  Checked here:

* rho_eq: at f = f_ref the run's response comes back itself, and the general
  arithmetic at df = 0 adds exact zeros (bitwise the run's S);
* linearity: S(z; f_ref) + S(z; df) is the S of f;
* the Drude kernel equals a direct numpy k.p sum on a random time-reversal-broken
  system, and on a plane-wave model it is the band curvature
  sum df d^2E/dk^2 (finite differences of the bands);
* fractional f: on a 2D plane-wave model with a periodic vector potential
  (time reversal broken; d^2H/dk^2 = 2 delta_ab exactly and every band is in the
  k.p sum) qhat.[S(z; df) + D(df)/z^2].qhat is the q -> 0 limit of the
  finite-q response of the grid populations,
  chi(q, z) = C/(Omega N_k) sum_nk df_nk sum_m [|<m,k+q|n,k>|^2/(z - dE_+)
              - |<m,k-q|n,k>|^2/(z + dE_-)],
  to O(q); without D it is not.

Run on a login node or the CPU pool (``JAX_PLATFORMS=cpu``; four host
devices give the 2 x 2 mesh case).
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

jax.config.update("jax_enable_x64", True)


def _meshes():
    out = [Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))]
    if len(jax.devices()) >= 4:
        out.append(Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y")))
    return out


Z = np.array([0.0, 0.3 + 0.4j, 1.0j, 2.0 + 0.5j, 0.05 + 0.6j])     # Ry; z = 0 is the static slot


# --------------------------------------------------------------------------- the plane-wave model

A_LAT = 4.0                                     # bohr, square lattice in the plane
G_MAX = 2


def _gvectors():
    b = 2.0 * np.pi / A_LAT
    ij = [(i, j) for i in range(-G_MAX, G_MAX + 1) for j in range(-G_MAX, G_MAX + 1)]
    return np.asarray(ij, np.int64), b * np.asarray([(i, j, 0.0) for i, j in ij])


def _model(seed=7):
    """H_GG'(k) = |k+G|^2 delta + V(G-G') + (2k+G+G').A(G-G'): real V(r), real A(r) (TRS broken)."""
    rng = np.random.default_rng(seed)
    ij, G = _gvectors()
    n = len(ij)
    index = {tuple(x): i for i, x in enumerate(ij.tolist())}
    vft, aft = {}, {}
    for d in {tuple(x) for x in (ij[:, None, :] - ij[None, :, :]).reshape(-1, 2).tolist()}:
        if d == (0, 0) or d in vft:
            continue
        m = (-d[0], -d[1])
        s = np.exp(-0.25 * (d[0] ** 2 + d[1] ** 2))
        vft[d] = s * 0.6 * (rng.normal() + 1j * rng.normal())
        vft[m] = np.conj(vft[d])
        a = s * 0.25 * (rng.normal(size=3) + 1j * rng.normal(size=3))
        a[2] = 0.0
        aft[d], aft[m] = a, np.conj(a)
    V = np.zeros((n, n), complex)
    Avec = np.zeros((n, n, 3), complex)
    for i in range(n):
        for j in range(n):
            d = tuple(ij[i] - ij[j])
            if d != (0, 0):
                V[i, j] = vft[d]
                Avec[i, j] = aft[d]
    return dict(G=G, V=V, A=Avec, n=n, index=index)


def _hamiltonian(m, k):
    kG = k[None, :] + m["G"]
    H = np.diag(np.sum(kG ** 2, axis=1)).astype(complex) + m["V"]
    s = kG[:, None, :] + kG[None, :, :]                       # k+G + k+G'
    H = H + np.einsum("ija,ija->ij", s, m["A"])
    dH = np.zeros((3,) + H.shape, complex)
    for a in range(3):
        dH[a] = np.diag(2.0 * kG[:, a]) + 2.0 * m["A"][:, :, a]
    return H, dH


def _bands(m, k):
    H, dH = _hamiltonian(m, k)
    e, c = np.linalg.eigh(H)
    v = np.einsum("in,aij,jm->anm", np.conj(c), dH, c)
    return e, c, v


def _grid(nk1=3, shift=(0.11, 0.07)):
    b = 2.0 * np.pi / A_LAT
    return np.asarray([((i + shift[0]) / nk1 * b, (j + shift[1]) / nk1 * b, 0.0)
                       for i in range(nk1) for j in range(nk1)])


def _system(seed=7, nk1=3):
    m = _model(seed)
    ks = _grid(nk1)
    e, c, v = zip(*(_bands(m, k) for k in ks))
    e, c = np.asarray(e), np.asarray(c)
    v = np.moveaxis(np.asarray(v), 0, 1)                     # (3, nk, nb, nb)
    return m, ks, e, c, v


def _populations(e, seed=11, nocc=4, lo=2, hi=8, amp=0.3):
    rng = np.random.default_rng(seed)
    f_ref = np.zeros(e.shape)
    f_ref[:, :nocc] = 1.0
    df = np.zeros(e.shape)
    df[:, lo:hi] = rng.uniform(-amp, amp, (e.shape[0], hi - lo))
    df[:, lo:nocc] = -np.abs(df[:, lo:nocc])                   # holes below, electrons above
    df[:, nocc:hi] = np.abs(df[:, nocc:hi])
    return f_ref, df


CELL = A_LAT * A_LAT * 1.0                                      # Omega (unit height)
CAP = 2.0                                                        # C = 2/(nspin nspinor), scalar


def _head_tensor(mesh, v, e, df, z, *, drude=True):
    from gw.qsgw_head import head_s_tensor_sharded, population_drude_tensor_sharded
    nk, nb = e.shape
    S = np.asarray(head_s_tensor_sharded(jnp.asarray(v), e, df, z, mesh=mesh, nb_logical=nb,
                                         cell_volume=CELL, nk_tot=nk, nspin=1, nspinor=1))
    D = population_drude_tensor_sharded(jnp.asarray(v), e, df, mesh=mesh, nb_logical=nb,
                                        cell_volume=CELL, nk_tot=nk, nspin=1, nspinor=1)
    if drude:
        S = S + D[None] / np.asarray(z)[:, None, None] ** 2
    return S, D


def _finite_q(m, ks, e, c, df, qvec, z):
    """chi(q, z) of the grid populations df (f-only Lehmann form, every final state unblocked)."""
    out = np.zeros(len(z), complex)
    for ik, k in enumerate(ks):
        for sgn in (+1.0, -1.0):
            ep, cp, _ = _bands(m, k + sgn * qvec)
            M2 = np.abs(np.conj(cp).T @ c[ik]) ** 2                       # [m, n] |<m,k+-q|n,k>|^2
            dE = ep[:, None] - e[ik][None, :]                            # E_m(k+-q) - E_n(k)
            for iz, zz in enumerate(z):
                if sgn > 0:
                    out[iz] += np.sum(df[ik][None, :] * M2 / (zz - dE))
                else:
                    out[iz] -= np.sum(df[ik][None, :] * M2 / (zz + dE))
    return CAP / (CELL * len(ks)) * out


# --------------------------------------------------------------------------- tests

@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_rho_eq_returns_the_run_response_bitwise(mesh):
    from gw.qsgw_head import (IterationHeadResponse, head_s_tensor_sharded,
                              population_drude_tensor_sharded, population_head_response)
    _, _, e, _, v = _system()
    f_ref, df = _populations(e)
    nk, nb = e.shape
    S_run = head_s_tensor_sharded(jnp.asarray(v), e, f_ref, Z, mesh=mesh, nb_logical=nb,
                                  cell_volume=CELL, nk_tot=nk, nspin=1, nspinor=1)
    run = IterationHeadResponse(omegas=tuple(complex(x) for x in Z), S_direct=S_run, Y_x=None,
                                Z_y=None, static_kappa2_bohr2=None, static_Y_x=None,
                                static_Z_y=None, static_chi_body_gamma=None,
                                sigma_energies_ry=e[:, :6], sigma_occupations=f_ref[:, :6],
                                efermi_ry=0.0)
    kw = dict(mesh=mesh, nb_logical=nb, cell_volume=CELL, nk_tot=nk, nspin=1, nspinor=1)
    same, parts = population_head_response(run, jnp.asarray(v), e, f_ref, f_ref.copy(), **kw)
    assert same is run and not parts["changed"] and not np.any(parts["drude"])
    # The general arithmetic at df = 0 adds exact zeros.
    zero = np.zeros_like(f_ref)
    dS = np.asarray(head_s_tensor_sharded(jnp.asarray(v), e, zero, Z, **kw))
    D0 = population_drude_tensor_sharded(jnp.asarray(v), e, zero, **kw)
    assert not np.any(dS) and not np.any(D0)
    assert np.array_equal(np.asarray(S_run) + dS + D0[None], np.asarray(S_run))
    # Linearity: the run's S plus the change is the S of f; the static slot has no Drude.
    new, parts = population_head_response(run, jnp.asarray(v), e, f_ref, f_ref + df, **kw)
    S_f = np.asarray(head_s_tensor_sharded(jnp.asarray(v), e, f_ref + df, Z, **kw))
    inv = np.where(np.abs(Z) > 0, 1.0 / np.where(np.abs(Z) > 0, Z, 1.0) ** 2, 0.0)
    want = S_f + parts["drude"][None] * inv[:, None, None]
    got = np.asarray(new.S_direct)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    df_eff = (f_ref + df) - f_ref                                  # the change the function forms
    assert np.array_equal(got[0], np.asarray(S_run)[0] + np.asarray(
        head_s_tensor_sharded(jnp.asarray(v), e, df_eff, Z, **kw))[0])      # z = 0: interband only
    assert np.array_equal(new.sigma_occupations, (f_ref + df)[:, :6])


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_drude_kernel_matches_numpy_kp_sum(mesh):
    from gw.degen_average import TOL_DEGENERACY_RY
    from gw.qsgw_head import population_drude_tensor_sharded
    rng = np.random.default_rng(5)
    nk, nb = 3, 9
    e = np.sort(rng.uniform(-1.0, 1.0, (nk, nb)), axis=1)
    e[1, 4] = e[1, 3] + 0.3 * TOL_DEGENERACY_RY                    # a degenerate pair: cut
    v = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    v = 0.5 * (v + np.conj(np.swapaxes(v, -1, -2)))
    df = rng.uniform(-0.4, 0.4, (nk, nb))
    got = population_drude_tensor_sharded(jnp.asarray(v), e, df, mesh=mesh, nb_logical=nb,
                                          cell_volume=2.5, nk_tot=nk, nspin=1, nspinor=2)
    ref = np.zeros((3, 3))
    for k in range(nk):
        for n in range(nb):
            ref += 2.0 * df[k, n] * np.eye(3)
            for mm in range(nb):
                d = e[k, n] - e[k, mm]
                if mm != n and abs(d) >= TOL_DEGENERACY_RY:
                    ref += df[k, n] * 2.0 * np.real(np.outer(v[:, k, n, mm], v[:, k, mm, n])) / d
    ref *= 2.0 / (2.5 * nk * 1 * 2)
    assert np.allclose(got, got.T, atol=0) and np.isrealobj(got)
    assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref)), np.max(np.abs(got - ref))


def test_drude_is_the_band_curvature():
    from gw.qsgw_head import population_drude_tensor_sharded
    mesh = _meshes()[0]
    m, ks, e, c, v = _system()
    _, df = _populations(e)
    D = population_drude_tensor_sharded(jnp.asarray(v), e, df, mesh=mesh, nb_logical=e.shape[1],
                                        cell_volume=CELL, nk_tot=len(ks), nspin=1, nspinor=1)
    h = 1.0e-4
    curv = np.zeros((3, 3))
    for a, b in ((0, 0), (1, 1), (0, 1)):
        ea, eb = np.eye(3)[a] * h, np.eye(3)[b] * h
        for ik, k in enumerate(ks):
            E = lambda dk: np.linalg.eigvalsh(_hamiltonian(m, k + dk)[0])
            d2 = (E(ea + eb) - E(ea - eb) - E(eb - ea) + E(-ea - eb)) / (4.0 * h * h)
            curv[a, b] += np.sum(df[ik] * d2)
        curv[b, a] = curv[a, b]
    curv *= CAP / (CELL * len(ks))
    sel = (slice(0, 2), slice(0, 2))
    assert np.max(np.abs(D[sel] - curv[sel])) <= 1e-4 * np.max(np.abs(curv[sel])), (D[sel], curv[sel])


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_fractional_head_is_the_finite_q_limit(mesh):
    m, ks, e, c, v = _system()
    _, df = _populations(e)
    zd = Z[1:]                                                      # the dynamic samples
    S, D = _head_tensor(mesh, v, e, df, zd)
    S_noD, _ = _head_tensor(mesh, v, e, df, zd, drude=False)
    for qhat in (np.array([1.0, 0.0, 0.0]), np.array([0.6, 0.8, 0.0])):
        want = np.einsum("a,zab,b->z", qhat, S, qhat)
        bare = np.einsum("a,zab,b->z", qhat, S_noD, qhat)
        errs = []
        for q in (2.0e-3, 1.0e-3):
            # The head is the even part (q.Pi.q); the odd O(q^3) part of chi is the magnet's
            # nonreciprocity, removed by the +-q average, so the error is O(q^2).
            chi = 0.5 * (_finite_q(m, ks, e, c, df, q * qhat, zd)
                         + _finite_q(m, ks, e, c, df, -q * qhat, zd)) / q ** 2
            errs.append(np.max(np.abs(chi - want) / np.abs(want)))
        assert errs[1] < 2e-4 and errs[1] < 0.3 * errs[0], errs        # O(q^2) convergence
        assert np.max(np.abs(chi - bare) / np.abs(want)) > 0.05         # the Drude term matters


# --------------------------------------------------------------------------- the deck key and wiring

def _raw(**lorrax):
    raw = {"control": {"calc_type": "COHSEX",
                       "timesteps": {"t0": 0.0, "tf": 100.0, "dt_measure": 50.0},
                       "files": {"outdir": "/tmp/spm_head_cpu"}},
           "system": {"ibndmin": 129, "ibndmax": 132, "ivbm": 130, "geometry": "slab",
                      "esmear": 0.05, "Efunc": {"E_x": 0.0, "E_y": 0.0, "E_z": 0.0}},
           "measurements": {"types": ["Efield"]},
           "lorrax": {"method": "cfm4", "dt": 25.0, "n_corr": 2, "chunk": 5, "system": "bundle",
                      "bundle_input": "/a/07/oneshot.in", "dipole_file": "/a/07/dipole.h5",
                      "model": "isdf", "collision": "subtracted", "collision_refresh_fs": 0.5,
                      "collision_hermitian": "dynamic"}}
    raw["lorrax"].update(lorrax)
    return raw


_SPM = dict(collision_screening="shared_pole", collision_source="/a/04/oneshot.in",
            collision_refresh_sweep="active")


def test_deck_key_default_and_refit():
    from tdagw.td_config import parse_td_deck
    assert parse_td_deck(_raw(**_SPM), source="cpu").values["lorrax.collision_spm_head"] == "frozen"
    for mode in ("refit", "sites"):
        v = parse_td_deck(_raw(**_SPM, collision_spm_head=mode), source="cpu").values
        assert v["lorrax.collision_spm_head"] == mode


def test_deck_key_refused_and_inactive():
    from tdagw.td_config import inactive_keys, parse_td_deck
    with pytest.raises(Exception, match="GATE td_collision"):
        parse_td_deck(_raw(**_SPM, collision_spm_head="thermal"), source="cpu")
    d = parse_td_deck(_raw(collision_screening="refreshed", collision_refresh_sweep="active",
                           collision_spm_head="refit"), source="cpu")
    assert "lorrax.collision_spm_head" in inactive_keys(d)


@pytest.mark.parametrize("kw", [dict(screening="refreshed", spm_head="refit"),
                                dict(screening="shared_pole", spm_head="thermal",
                                     spm_scratch="/tmp/x")])
def test_window_collision_refuses_the_head_off_its_member(kw):
    from tdagw import dynsig as D
    member = "shared_pole" if kw["screening"] == "shared_pole" else "gn_ppm"
    with pytest.raises(Exception, match="GATE dynsig_collision_spm_head"):
        D.window_collision({"member": member}, None, mesh=None, k_frac=None, b_lo=0, ntr=1,
                           sweep="active", **kw)


# --------------------------------------------------------------------------- the sites head

def _pole_model(z, om, b):
    """The MPA scalar form W(z) = sum_p 2 Omega_p B_p / (z^2 - Omega_p^2); W(0) = -2 sum B/Omega."""
    z = np.asarray(z)[:, None]
    return np.sum(2.0 * om * b / (z * z - om * om), axis=1)


def _sites():
    line = np.array([0.11, 0.16, 0.22, 0.3]) + 0.0735j          # Ry: a line at height 1 eV
    imag = 1j * np.array([0.02, 0.2])
    return np.concatenate([line, imag])


def test_site_fit_static_value_is_the_model_s():
    from tdagw import dynsig as D
    z = _sites()
    om = np.array([0.12 - 0.005j, 0.2 - 0.01j, 0.28 - 0.02j])   # inside 2 max|z_s| (the fit's box)
    b = np.array([120.0 - 3.0j, 800.0 - 10.0j, 2500.0 - 40.0j])        # W^c(0) < 0: screening
    hctx = dict(sites_z=z, config=SimpleNamespace(mpa=SimpleNamespace(pole_solver="loewner")))
    fit = D._head_site_fit(hctx, _pole_model(z, om, b))
    w0 = -2.0 * np.sum(b / om)
    assert fit["n_valid"] == 3
    assert abs(fit["w0"] - w0) < 1e-7 * abs(w0), (fit["w0"], w0)


def test_sites_head_moves_only_the_static_value(monkeypatch):
    """spm_head_refit_sites: the run's poles, residues rescaled to W_run + [M_f(0) - M_eq(0)]."""
    import dataclasses
    from tdagw import dynsig as D

    @dataclasses.dataclass(frozen=True)
    class _Head:
        poles_ry: np.ndarray
        residues_ry: np.ndarray

    z = _sites()
    om = np.array([0.12 - 0.005j, 0.2 - 0.01j, 0.28 - 0.02j])   # inside 2 max|z_s| (the fit's box)
    b_eq = np.array([120.0 - 3.0j, 800.0 - 10.0j, 2500.0 - 40.0j])
    b_f = b_eq * np.array([0.7, 1.0, 1.02])
    cfg = SimpleNamespace(mpa=SimpleNamespace(pole_solver="loewner"))
    head0 = _Head(poles_ry=np.array([0.3 + 0.0j, 0.9 + 0.0j]),
                  residues_ry=np.array([300.0 + 0.0j, 1200.0 + 0.0j]))
    f_eq = np.zeros((2, 3)); f_eq[:, 0] = 1.0
    hctx = dict(sites_z=z, config=cfg, nb=3, f_eq=f_eq, head0=head0)
    fit_eq = D._head_site_fit(hctx, _pole_model(z, om, b_eq))
    hctx.update(sites_w0_eq=fit_eq["w0"], sites_fit_eq=fit_eq, sites_wc_eq=_pole_model(z, om, b_eq))
    monkeypatch.setattr(D, "_head_wc_at_sites", lambda h, f, drude=True: _pole_model(z, om, b_f))
    same, rec0 = D.spm_head_refit_sites(hctx, f_eq)
    assert same is head0 and rec0["delta"] == 0.0
    f = f_eq.copy(); f[0, 0], f[0, 1] = 0.9, 0.1
    head, rec = D.spm_head_refit_sites(hctx, f)
    static = lambda h: complex(-2.0 * np.sum(np.asarray(h.residues_ry) / np.asarray(h.poles_ry)))
    want = static(head0) + (-2.0 * np.sum(b_f / om) - (-2.0 * np.sum(b_eq / om))).real
    assert np.array_equal(head.poles_ry, head0.poles_ry)
    assert abs(static(head) - want) < 1e-7 * abs(want), (static(head), want)
