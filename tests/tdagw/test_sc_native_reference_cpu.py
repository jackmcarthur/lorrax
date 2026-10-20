"""SC-reference gauge and native full-centroid HF-delta algebra; not TD admission."""
import runtime  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace as NS
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tdagw import snapshot_runtime as S
from tdagw import propagate as tp

jax.config.update("jax_enable_x64", True)


def _fixture():
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices.reshape((2, 2) if len(devices) == 4 else (1, -1)), ("x", "y"))
    nk, nb, ns, mu = 4, 4, 2, 4
    rng = np.random.default_rng(1637)
    psi = (rng.normal(size=(nk, nb, ns, mu))
           + 1j*rng.normal(size=(nk, nb, ns, mu)))*0.1
    a = rng.normal(size=(mu, mu))
    v0 = a@a.T
    a = rng.normal(size=(mu, mu))+1j*rng.normal(size=(mu, mu))
    v1 = a@a.conj().T
    V = np.stack([v0, v1, 0.7*v0, v1.conj()])*0.03
    put = lambda x, spec: jax.device_put(x, NamedSharding(mesh, spec))
    prepared = dict(mesh=mesh, kgrid=(nk, 1, 1), nk=nk, nb=nb, spin_weight=1.0,
        psi=put(psi, P(None, None, None, ("x", "y"))),
        V_q=put(V, P(None, "x", "y")), nmu_logical=mu)
    return prepared, psi, V, rng


def test_whole_protected_velocity_rotation_not_window_only():
    _, _, _, rng = _fixture()
    U = np.stack([np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))[0]
                  for _ in range(4)])
    a = rng.normal(size=(3, 4, 4, 4))+1j*rng.normal(size=(3, 4, 4, 4))
    v = a+a.conj().swapaxes(-1, -2)
    active = slice(1, 3)
    full = U.conj().swapaxes(-1, -2)[None]@v@U[None]
    actual = S.sc_reference_velocity(U, v, active)
    np.testing.assert_allclose(actual, full[:, :, active, active], atol=2e-14)
    wrong = U[:, active, active].conj().swapaxes(-1, -2)[None]@v[:, :, active, active]@U[:, active, active][None]
    assert np.linalg.norm(actual-wrong) > 1.0
    with pytest.raises(ValueError, match="nonunitary"):
        S.sc_reference_velocity(2*U, v, active)


def _literal_hf(psi, V, delta, spin):
    nk, na = psi.shape[:2]
    charge = np.einsum("kasu,tkab,kbsu->tu", psi, delta, psi.conj())
    h = spin/nk*np.einsum("kasu,uv,tv,kbsu->tkab", psi.conj(), V[0], charge, psi)
    x = np.zeros_like(h)
    for k in range(nk):
        for kp in range(nk):
            x[:, k] += np.einsum("asu,csu,uv,tcd,drv,brv->tab",
                psi[k].conj(), psi[kp], V[(k-kp) % nk], delta[:, kp],
                psi[kp].conj(), psi[k], optimize=True)/nk
    return h-x


def test_native_hf_delta_literal_coherences_head_once_and_trace():
    from ffi import fft
    from common.units import RYD_TO_EV
    from tdagw.tdgw_io import HA_IN_EV
    prepared, psi, V, rng = _fixture()
    mesh, active = prepared["mesh"], slice(1, 3)
    req = np.broadcast_to(np.diag([1., 0.]), (4, 2, 2)).astype(complex).copy()
    reqd = jax.device_put(req, NamedSharding(mesh, P(("x", "y"), None, None)))
    a = rng.normal(size=(1, 4, 2, 2))+1j*rng.normal(size=(1, 4, 2, 2))
    delta = (a+a.conj().swapaxes(-1, -2))*0.03
    rho = tp.place_window_blocks(req[None]+delta, mesh, 4)
    with fft.xla_reference():
        model = S.native_hf_delta_model(prepared, active, reqd,
            bare_head_ha=0.007, hartree_q0=prepared["V_q"][0])
        run = jax.jit(lambda p, r: model.apply(p, r, 0.0))
        actual = np.asarray(run(model.params, rho))
        zero = np.asarray(run(model.params, tp.place_window_blocks(req[None], mesh, 4)))
    expected = _literal_hf(psi[:, active], V, delta, prepared["spin_weight"])
    expected = expected*(RYD_TO_EV/HA_IN_EV)-0.007*delta
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))
    np.testing.assert_allclose(actual, actual.conj().swapaxes(-1, -2), atol=3e-13)
    comm = actual@np.asarray(rho)-np.asarray(rho)@actual
    assert abs(np.trace(comm, axis1=-2, axis2=-1).sum()) < 1e-14


def test_native_hf_requires_explicit_hartree_policy():
    prepared, _, _, _ = _fixture()
    with pytest.raises(ValueError, match="G=0-excluded"):
        S.native_hf_delta_model(prepared, slice(1, 3), np.zeros((4, 2, 2)))


def _sc_case(monkeypatch):
    prepared, psi, V, rng = _fixture()
    mesh = prepared["mesh"]
    U = np.stack([np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))[0]
                  for _ in range(4)])
    E = np.tile([-1., -.3, .4, 1.], (4, 1))
    f = np.tile([1., 1., 0., 0.], (4, 1))
    H = (U*E[:, None])@U.conj().swapaxes(-1, -2)
    loop_rows = np.array([0, 2])
    frame = dict(U_active=U, energy_ry=E, occupations=f,
        input_H_loop_ry=H[loop_rows], input_loop_full_rows=loop_rows)
    record = dict(schema="native_sc_input_frame_v1", frame="frame_accepted", active_slice=[0, 4],
        wfn_fingerprint="authenticated-original",
        **{key: S._sc_array_digest(frame[name]) for key, name in
        (("U_sha256", "U_active"), ("energy_sha256", "energy_ry"),
         ("occupation_sha256", "occupations"), ("input_h_sha256", "input_H_loop_ry"))})
    complete = dict(reference_ready=True, solver=dict(status="CONVERGED_ACCEPTED_INPUT",
        frame=record["frame"], accepted_input_join=True, proper_sc_requested=True, native_converged=True))
    scpsi = np.einsum("kmsu,kma->kasu", psi, U)
    prepared.update(psi=jax.device_put(scpsi, prepared["psi"].sharding), energy=E,
        occupations=f, k_frac=np.c_[np.arange(4)/4, np.zeros((4, 2))])
    a = rng.normal(size=(3, 4, 4, 4))+1j*rng.normal(size=(3, 4, 4, 4))
    velocity = a+a.conj().swapaxes(-1, -2)
    inputs = dict(frame_receipt=record, driver_complete=complete,
        original_psi=jax.device_put(psi, prepared["psi"].sharding),
        velocity_dft_ry=velocity, wfn_fingerprint="authenticated-original",
        velocity_wfn_fingerprint="authenticated-original",
        bare_head_coefficient_ry=.013,
        structure=dict(rprim=np.eye(3)*5., atomic_numbers=[1],
            atomic_positions_red=np.zeros((1, 3)), fftgrid=[4, 4, 4]))
    source = dict(config=NS(sys_dim=2, head=NS(use_bgw_vcoul=False)))
    monkeypatch.setattr(S, "prepare_native_source", lambda src, msh: prepared)
    return source, frame, inputs, prepared, mesh


def test_accepted_sc_builder_keeps_tail_mixing_gauge_and_anchor_once(monkeypatch):
    from ffi import fft
    from common.units import RYD_TO_EV
    from tdagw.tdgw_io import HA_IN_EV
    source, frame, inputs, prepared, mesh = _sc_case(monkeypatch)
    with fft.xla_reference():
        built = S.build_sc_td_reference(source, frame, inputs, mesh=mesh, active=slice(1, 3))
        assert S.authenticate_orbital_gauge(prepared, built["model"], slice(1, 3))["checked"]
        req = built["ops"].rho_eq
        z = tp.place_window_blocks(np.zeros((1, 4, 2, 2), complex), mesh, 4)
        model = S.bare_hf_coherent_model(built["model"], prepared, req, z)
        h = jax.jit(lambda p, r: model.apply(p, r, 0.0))(model.params, req[None])
    np.testing.assert_array_equal(h, np.zeros_like(h))
    np.testing.assert_allclose(built["header"]["e_kn"],
        frame["energy_ry"][:, 1:3]*RYD_TO_EV/HA_IN_EV, rtol=1e-14)
    assert built["receipts"]["protected_bands"] == [0, 4]
    assert built["receipts"]["collision_equilibrium_subtraction"] is False
    # Joint decks call this mode 'dynamic'; the complete M_H is retained.
    assert built["snapshot_options"]["hermitian"] == "dynamic"
    assert built["receipts"]["zero_field_drift_gate"] == "required_not_run"
    with pytest.raises(ValueError, match="twice"):
        S.bare_hf_coherent_model(built["model"], prepared, req, z, bare_head_ha=.013)


def test_sc_builder_refuses_unconverged_or_changed_frame_and_wrong_gauge(monkeypatch):
    from ffi import fft
    source, frame, inputs, prepared, mesh = _sc_case(monkeypatch)
    inputs["driver_complete"]["reference_ready"] = False
    with pytest.raises(ValueError, match="converged evaluated INPUT"):
        S.build_sc_td_reference(source, frame, inputs, mesh=mesh, active=slice(1, 3))
    inputs["driver_complete"]["reference_ready"] = True
    frame["energy_ry"][0, 0] += .1
    with pytest.raises(ValueError, match="digest"):
        S.build_sc_td_reference(source, frame, inputs, mesh=mesh, active=slice(1, 3))
    frame["energy_ry"][0, 0] -= .1
    # A second rotation of only the active block is not the saved SC gauge.
    prepared["psi"] = prepared["psi"].at[:, 1:3].multiply(1j)
    with fft.xla_reference(), pytest.raises(ValueError, match="full protected SC rotation"):
        S.build_sc_td_reference(source, frame, inputs, mesh=mesh, active=slice(1, 3))


def test_sc_snapshot_anchor_needs_no_fabricated_absolute_hf_and_keeps_raw_C(monkeypatch):
    from ffi import fft
    from common.units import RYD_TO_EV
    from tdagw.tdgw_io import HA_IN_EV
    source, frame, inputs, prepared, mesh = _sc_case(monkeypatch)
    prepared.update(nb_chi=4, nb_sigma=4)
    M = np.broadcast_to(np.array([[.06, .01j], [-.01j, -.02]]), (4, 2, 2)).copy()
    gamma = np.broadcast_to(np.eye(2)*.02, M.shape).copy()
    injection = np.broadcast_to(np.eye(2)*.001, M.shape).copy()
    def factory(source, prepared, options):
        def evaluate(basis, equilibrium=False, fit_bases=None):
            return dict(gamma=gamma, injection=injection, m_h=M,
                static=np.zeros_like(M), screening_receipts=[], native_receipt={})
        return evaluate, {}
    monkeypatch.setattr(S, "make_native_snapshot_evaluator", factory)
    with fft.xla_reference():
        system = S.build_sc_td_reference(source, frame, inputs, mesh=mesh, active=slice(1, 3))
        snapshot = S.snapshot_collision(source, system["ops"], mesh=mesh,
            k_frac=prepared["k_frac"], b_lo=1, ntr=1, model=system["model"],
            options=system["snapshot_options"], say=lambda _: None)
    basis = snapshot["state"]["basis_last"][0]
    H0 = (basis["U"]*basis["e"][:, None])@basis["U"].conj().swapaxes(-1, -2)
    np.testing.assert_allclose(H0, np.eye(2)[None]*frame["energy_ry"][:, None, 1:3], atol=1e-12)
    assert snapshot["state"]["hf_reference_ry"] is None
    assert snapshot["state"]["one_body_counterterm_ry"] is None
    assert snapshot["t0"]["form"] == "raw"
    current = np.asarray(system["ops"].rho_eq)[None]
    expected = (injection[None]-.5*(gamma[None]@current+current@gamma[None]))*RYD_TO_EV/HA_IN_EV
    actual = S.C.rate(snapshot["diss"], jnp.zeros_like(system["ops"].rho_eq)[None])
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert np.linalg.norm(actual) > .001


def _tail_reference():
    from gw.band_extrapolation import BandLadder, fit_band_extrapolation_spectral
    ladder = BandLadder(e_dft_ev=np.arange(1., 21.)[:, None],
        e_weyl_ev=np.arange(21., 41.), w_k=np.ones(1), e0_ev=0., n0=0.,
        c_ev=1., r2=1., estar_ev=10., n_dft=20, n_target=40, fit_window=(10, 20))
    e = np.array([[-2., -1.]])
    def shell(lo, hi):
        nodes, weights = ladder._terms(lo, hi)
        return np.sum(weights/(nodes-e[..., None]+2.)**3, axis=-1)
    first = np.full(e.shape, .2+.4j)
    sums = np.stack([first, first+shell(6, 9), first+shell(6, 12)])
    fit = fit_band_extrapolation_spectral([6, 9, 12], sums, ladder, e_state_ev=e)
    assert fit.n_failed == 0
    return dict(fit=fit, input_frame="accepted", input_h_sha256="input-digest", ladder_shift_ev=0.)


def test_native_evaluator_binds_actual_producer_tail_or_refuses_missing(monkeypatch, tmp_path):
    import gw.ppm_sigma as sigma
    _, _, _, prepared, mesh = _sc_case(monkeypatch)
    config = NS(sigma=NS(band_extrapolation=True))
    source = dict(config=config, screening_inputs=dict(config=config),
        frame=dict(frame="accepted", input_h_sha256="input-digest"))
    plan = {}
    monkeypatch.setattr(sigma, "window_sigma_plan", lambda *a, **kw: plan)
    options = dict(b_lo=1, n_active=2, spm_scratch=str(tmp_path))
    with pytest.raises(ValueError, match="source used band extrapolation"):
        S.make_native_snapshot_evaluator(source, prepared, options)
    source["spectral_tail_reference"] = _tail_reference()
    S.make_native_snapshot_evaluator(source, prepared, options)
    assert plan["spectral_tail"] is source["spectral_tail_reference"]
    source["spectral_tail_reference"] = dict(source["spectral_tail_reference"], input_frame="stale")
    with pytest.raises(ValueError, match="different evaluated SC input frame"):
        S.make_native_snapshot_evaluator(source, prepared, options)
    source["spectral_tail_reference"]["input_frame"] = "accepted"
    source["spectral_tail_reference"]["input_h_sha256"] = "other-digest"
    with pytest.raises(ValueError, match="input-H digest"):
        S.make_native_snapshot_evaluator(source, prepared, options)
