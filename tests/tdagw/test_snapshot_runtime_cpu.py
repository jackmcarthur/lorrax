"""Four-CPU distributed Nystrom/vertex and full-rho adapter smoke gates."""
from types import SimpleNamespace as NS
from typing import NamedTuple

import runtime  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tdagw.snapshot_runtime import prepare_projected_source, rotate_pair_vertices

jax.config.update("jax_enable_x64", True)


def _source():
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices.reshape((2, 2) if devices.size == 4 else (1, -1)), ("x", "y"))
    nk, nb, ns, nmu = 4, 4, 2, max(8, devices.size)
    rng = np.random.default_rng(781)
    psi = rng.normal(size=(nk, nb, ns, nmu)) + 1j * rng.normal(size=(nk, nb, ns, nmu))
    A = rng.normal(size=(nmu, nmu)) + 1j * rng.normal(size=(nmu, nmu))
    v = A @ A.conj().T + np.eye(nmu)
    V = np.stack([v.real, v, v.real, v.conj()]) * 0.001
    k = np.stack((np.arange(nk) / nk, np.zeros(nk), np.zeros(nk)), axis=1)
    carrier = NS(psi_nmu=jax.device_put(psi, NamedSharding(mesh, P(None, None, None, "y"))),
                 plan=NS(k_parent_frac=k, irr_idx=np.arange(nk), parent_full_rows=np.arange(nk)))
    wfns = NS(green_parent=carrier, slices=NS(b0=0, b4_chi=nb, b4_logical=nb),
              enk=jnp.tile(jnp.array([-0.4, -0.2, 0.3, 0.7]), (nk, 1)),
              occ=jnp.tile(jnp.array([1.0, 1.0, 0.0, 0.0]), (nk, 1)))
    source = dict(wfns=wfns, meta=NS(kgrid=(nk, 1, 1), n_rmu=nmu),
                  V_q=jax.device_put(V, NamedSharding(mesh, P(None, "x", "y"))))
    h0 = np.einsum("ki,ij->kij", np.asarray(wfns.enk), np.eye(nb)).astype(complex)
    source["hf_reference"] = dict(kin_ion_ry=jnp.asarray(h0),
        hartree_ry=jnp.zeros_like(jnp.asarray(h0)), exchange_ry=jnp.zeros_like(jnp.asarray(h0)),
        band_start=0, band_stop=nb, basis="source_wfn", kset="full_bz", bare_head_coefficient_ry=0.0)
    return source, mesh, psi, V


def test_distributed_nystrom_vertices_match_dense_full_rank():
    source, mesh, psi, V = _source()
    prepared = prepare_projected_source(source, mesh, rank=V.shape[-1])
    omega = np.asarray(prepared["omega"])
    for iq in (0, 1, 3):
        T, receipt = prepared["vertices"](iq)
        gram = omega.conj().T @ V[iq] @ omega
        values, vectors = np.linalg.eigh(gram)
        F = V[iq] @ omega @ ((vectors / np.sqrt(values)[None]) @ vectors.conj().T)
        np.testing.assert_allclose(F @ F.conj().T, V[iq], rtol=2e-10, atol=2e-12)
        reference = np.einsum("kbsu,kdsu,ur->kbdr", psi,
            psi[prepared["kmq"][iq]].conj(), F.conj(), optimize=True)
        np.testing.assert_allclose(T, reference, rtol=2e-9, atol=2e-10)
        assert abs(receipt["omitted_coulomb_trace_fraction"]) < 1e-10
    assert prepared["psi"].sharding.spec[-1] == ("x", "y")


def test_active_pair_rotation_matches_rotated_orbitals():
    source, mesh, psi, V = _source()
    prepared = prepare_projected_source(source, mesh, rank=3)
    rng = np.random.default_rng(371)
    nk, nb = psi.shape[:2]
    U = np.stack([np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
                  for _ in range(nk)])
    active = slice(1, 3)
    iq = 1
    T, _ = prepared["vertices"](iq)
    rotated = rotate_pair_vertices(T, U, prepared["kmq"][iq], active)
    omega = np.asarray(prepared["omega"])
    gram = omega.conj().T @ V[iq] @ omega
    values, vectors = np.linalg.eigh(gram)
    F = V[iq] @ omega @ ((vectors / np.sqrt(values)[None]) @ vectors.conj().T)
    orbitals = psi.copy()
    orbitals[:, active] = np.einsum("kasu,kab->kbsu", psi[:, active], U)
    expected = np.einsum("kbsu,kdsu,ur->kbdr", orbitals,
                        orbitals[prepared["kmq"][iq]].conj(), F.conj(), optimize=True)
    np.testing.assert_allclose(rotated, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("real_axis_fit", [False, True])
def test_evaluator_forwards_explicit_screening_capacities(monkeypatch, real_axis_fit):
    from tdagw import snapshot_runtime as runtime
    _, mesh, _, _ = _source()
    nk, nb = 4, 4
    energy = np.tile([-0.4, -0.2, 0.3, 0.7], (nk, 1))
    prepared = dict(mesh=mesh, nk=nk, nb_chi=nb, nb_sigma=nb,
        energy=energy, occupations=np.tile([1.0, 1.0, 0.0, 0.0], (nk, 1)),
        spin_weight=1.0, kmq=np.tile(np.arange(nk), (nk, 1)), q_negative=np.arange(nk),
        vertices=lambda iq: (np.zeros((nk, nb, nb, 1), complex), {}))
    captured = {}

    def screen(*args, **options):
        captured.update(options)
        raise RuntimeError("screening capacities checked")

    monkeypatch.setattr(runtime.SS, "make_projected_sampler", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime.SS, "screen_snapshot", screen)
    evaluate, _ = runtime.make_snapshot_evaluator(prepared, dict(b_lo=1, n_active=2,
        max_projection_order=800, max_pencil_bytes=256 * 1024 ** 2,
        real_axis_fit=real_axis_fit))
    basis = dict(U=np.tile(np.eye(2), (nk, 1, 1)), e=energy[:, 1:3],
                 rho=np.tile(np.diag([1.0, 0.0]), (nk, 1, 1)))
    with pytest.raises(RuntimeError, match="screening capacities checked"):
        evaluate(basis, equilibrium=True)
    assert captured["max_projection_order"] == 800
    assert captured["max_pencil_bytes"] == 256 * 1024 ** 2
    if real_axis_fit:
        fit_sites, check_sites = runtime.snapshot_supports(energy, {})
        np.testing.assert_array_equal(captured["real_fit_sites"], fit_sites.real)
        assert np.intersect1d(captured["real_fit_sites"], check_sites.real).size == 0
    else:
        assert captured["real_fit_sites"] is None


@pytest.mark.parametrize("location", ["V_q", "screening_inputs"])
def test_typed_whole_zone_coulomb_source(location):
    from symmetry_maps import QirrOperator
    source, mesh, _, _ = _source()
    reference = prepare_projected_source(source, mesh, rank=2)["vertices"](1)[0]
    typed = QirrOperator.whole_zone(source.pop("V_q"))
    if location == "V_q":
        source["V_q"] = typed
    else:
        source["screening_inputs"] = dict(V_q=typed)
    actual = prepare_projected_source(source, mesh, rank=2)["vertices"](1)[0]
    np.testing.assert_array_equal(actual, reference)


def test_typed_coulomb_nonidentity_full_rows_refused():
    from dataclasses import replace
    from symmetry_maps import QirrOperator
    source, mesh, _, _ = _source()
    source["V_q"] = replace(QirrOperator.whole_zone(source["V_q"]), full_rows=np.arange(4)[::-1])
    with pytest.raises(ValueError, match="GATE td_snapshot_coulomb_full_grid"):
        prepare_projected_source(source, mesh, rank=2)


@pytest.mark.parametrize("nspin,nspinor_file,expected", [(1, 1, 2.0), (2, 1, 1.0),
                                                       (1, 2, 1.0), (2, 2, 0.5)])
def test_spin_weight_uses_file_multiplicities(nspin, nspinor_file, expected):
    source, mesh, _, _ = _source()
    source["meta"].nspin = nspin
    source["meta"].nspinor_wfnfile = nspinor_file
    assert prepare_projected_source(source, mesh, rank=2)["spin_weight"] == expected


def test_band_tables_use_collective_host_gather(monkeypatch):
    from common import collectives
    source, mesh, _, _ = _source()
    original, gathered = collectives.gather_to_host, []

    def gather(value):
        gathered.append(value)
        return original(value)

    monkeypatch.setattr(collectives, "gather_to_host", gather)
    prepared = prepare_projected_source(source, mesh, rank=2)
    assert any(value is source["wfns"].enk for value in gathered)
    assert any(value is source["wfns"].occ for value in gathered)
    np.testing.assert_array_equal(prepared["energy"], source["wfns"].enk)
    np.testing.assert_array_equal(prepared["occupations"], source["wfns"].occ)


def test_nystrom_q_pairing_preserves_conjugate_vertices():
    source, mesh, _, _ = _source()
    prepared = prepare_projected_source(source, mesh, rank=3)
    positive, _ = prepared["vertices"](1)
    negative, _ = prepared["vertices"](3)
    paired = negative[prepared["kmq"][1]].swapaxes(1, 2).conj()
    np.testing.assert_allclose(positive, paired, rtol=2e-11, atol=2e-11)


def test_axis_carrier_uses_its_projection_faces():
    from gw.wavefunction_bundle import ParentGreenCarrier
    source, mesh, psi, _ = _source()
    old = source["wfns"].green_parent
    mun = jax.device_put(psi.transpose(0, 2, 3, 1), NamedSharding(mesh, P(None, None, "x", None)))
    source["wfns"].green_parent = ParentGreenCarrier(
        psi_nmu=old.psi_nmu, psi_mun=mun, enk=source["wfns"].enk,
        occ=source["wfns"].occ, plan=old.plan, layout="axis")
    prepared = prepare_projected_source(source, mesh, rank=2)
    np.testing.assert_array_equal(prepared["psi"], psi)


def test_source_model_gauge_authenticates_actual_orbitals():
    import functools
    from tdagw.self_energy import SelfEnergyModel, _isdf_apply
    from tdagw.snapshot_runtime import authenticate_orbital_gauge
    source, mesh, psi, _ = _source()
    prepared = prepare_projected_source(source, mesh, rank=2)
    orbital_sh = NamedSharding(mesh, P(None, None, None, "y"))
    model_psi = jax.device_put(psi[:, 1:3], orbital_sh)
    model = SelfEnergyModel(apply=functools.partial(_isdf_apply, ()),
        params=((model_psi,), None), linear=True, name="ISDF synthetic gauge")
    assert authenticate_orbital_gauge(prepared, model, slice(1, 3))["relative_error"] == 0.0
    mismatch = model_psi.at[1, 0].multiply(-1.0)
    with pytest.raises(ValueError, match="GATE td_snapshot_orbital_gauge"):
        authenticate_orbital_gauge(prepared, model._replace(params=((mismatch,), None)), slice(1, 3))


def test_hf_source_capture_reuses_existing_authenticated_outputs():
    from gw.ppm_pipeline import observe_window_sigma_source, publish_window_hf_reference
    source, _, _, _ = _source()
    fields = source["hf_reference"]
    result = NS(v_h_kij_ry=fields["hartree_ry"], sigma_x_kij_ry=fields["exchange_ry"],
                hartree_omitted=False, kset="full_bz")
    meta = NS(cell_volume=12.0, nk_tot=4)
    bands = NS(b0=0, b3=4)
    # An unobserved ordinary GW call does not even inspect its operands.
    publish_window_hf_reference(None, None, None, None, None)
    with observe_window_sigma_source() as captured:
        publish_window_hf_reference(fields["kin_ion_ry"], result, NS(vc0=0.24), meta, bands)
    hf = captured["hf_reference"]
    assert hf["kin_ion_ry"] is not fields["kin_ion_ry"]
    np.testing.assert_array_equal(hf["kin_ion_ry"], fields["kin_ion_ry"])
    assert hf["bare_head_coefficient_ry"] == 0.24 / 48.0
    assert hf["basis"] == "source_wfn"


@pytest.mark.parametrize("statistics", [None, "unspecified", "physical_adiabatic"])
def test_snapshot_statistics_guard_precedes_source_setup(monkeypatch, statistics):
    from tdagw import snapshot_runtime as runtime

    def unexpected_setup(*args, **kwargs):
        pytest.fail("statistical guard must precede any source preparation")

    monkeypatch.setattr(runtime, "prepare_projected_source", unexpected_setup)
    options = {} if statistics is None else dict(statistics=statistics)
    with pytest.raises(ValueError, match="GATE td_collision_statistics"):
        runtime.snapshot_collision(None, None, mesh=None, k_frac=None,
                                   b_lo=0, ntr=1, model=None, options=options)


def test_joint_fit_refusal_does_not_install_a_snapshot(monkeypatch):
    from common.units import RYD_TO_EV
    from tdagw import collision as C
    from tdagw.self_energy import zero_model
    from tdagw import snapshot_runtime as runtime
    from tdagw.tdgw_io import HA_IN_EV
    source, mesh, _, _ = _source()

    def factory(prepared, options):
        def evaluate(basis, equilibrium=False):
            if not equilibrium:
                raise ValueError("GATE snapshot_fit_error: independently checked response refused")
            zero = np.zeros((4, 2, 2), complex)
            return dict(gamma=zero, injection=zero, m_h=zero, static=zero,
                        screening_receipts=[])
        return evaluate, {}

    monkeypatch.setattr(runtime, "make_snapshot_evaluator", factory)
    req = np.broadcast_to(np.diag([1.0, 0.0]), (4, 2, 2)).astype(complex)
    put = lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec))
    ops = NS(e_qp=put(np.asarray(source["wfns"].enk)[:, 1:3] * RYD_TO_EV / HA_IN_EV,
                     P(("x", "y"), None)),
             rho_eq=put(req, P(("x", "y"), None, None)),
             w_k=put(np.full(4, 0.25), P(("x", "y"))))
    model = zero_model()
    built = runtime.snapshot_collision(source, ops, mesh=mesh,
        k_frac=source["wfns"].green_parent.plan.k_parent_frac, b_lo=1, ntr=1, model=model,
        options=dict(statistics="finite_lag", hamiltonian="reference", hf_model=model, screening="joint",
                     policy=C.RefreshPolicy(every=1.0)), say=lambda _: None)
    held = np.asarray(built["held"]).copy()
    with pytest.raises(ValueError, match="GATE snapshot_fit_error"):
        built["refresh"](2.0, runtime.tp.place_window_blocks(req[None], mesh, 4), ops,
                           (built["model"].params, built["held"]))
    assert built["state"]["n"] == 0 and len(built["state"]["history"]) == 1
    np.testing.assert_array_equal(built["state"]["snapshot_held"], held)
    assert "snapshot_fit_error" in built["state"]["snapshot_failure"]["error"]


def test_bare_hf_kernel_replaces_full_screened_W_and_keeps_bare_head():
    from ffi import fft as FF
    from tdagw.dynsig import make_dynsig_kernels, _w_R
    from tdagw.model import assemble_td_model, ZETA_G_SPEC, V_G_SPEC
    from tdagw.self_energy import isdf_model
    from tdagw.snapshot_runtime import bare_hf_coherent_model, authenticate_orbital_gauge
    from tdagw import propagate as tp
    source, mesh, _, _ = _source()
    prepared = prepare_projected_source(source, mesh, rank=1)
    put = lambda a, spec: jax.device_put(np.asarray(a), NamedSharding(mesh, spec))
    with FF.xla_reference():
        kern = make_dynsig_kernels(mesh, prepared["kgrid"])
        bare_R = jax.jit(lambda v: _w_R(v, kern))(prepared["V_q"])
        model_data = assemble_td_model(prepared["psi"][:, 1:3], 5.0 * bare_R,
            put(np.zeros((8, 4), complex), ZETA_G_SPEC), put(np.zeros(4), V_G_SPEC),
            mesh_xy=mesh, kgrid=prepared["kgrid"], k_frac=prepared["k_frac"],
            band_lo=1, n_bands=2, n_rmu=8)
        req = put(np.broadcast_to(np.diag([1.0, 0.0]), (4, 2, 2)).astype(complex),
                  P(("x", "y"), None, None))
        model = isdf_model(model_data, mesh, k_frac=prepared["k_frac"], rho_eq=req)
        assert authenticate_orbital_gauge(prepared, model, slice(1, 3))["checked"]
        zero = tp.place_window_blocks(np.zeros((1, 4, 2, 2), complex), mesh, 4)
        bare = bare_hf_coherent_model(model, prepared, req, zero, bare_head_ha=0.004)
        rh = np.broadcast_to(np.array([[0.8, 0.03j], [-0.03j, 0.2]]), (1, 4, 2, 2)).copy()
        rho = tp.place_window_blocks(rh, mesh, 4)
        run = jax.jit(lambda p, r: model.apply(p, r, 0.0))
        run_bare = jax.jit(lambda p, r: bare.apply(p, r, 0.0))
        original, actual = run(model.params, rho), run_bare(bare.params, rho)
        np.testing.assert_allclose(actual, original / 5.0 - 0.004 * (rho - req[None]),
                                   rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("convention", ["hf", "reference"])
@pytest.mark.parametrize("statistics", ["finite_lag", "retarded_reference"])
def test_hf_baseline_and_reference_anchor_ignore_screened_static_mismatch(monkeypatch, convention, statistics):
    from common.units import RYD_TO_EV
    from tdagw import collision as C
    from tdagw.self_energy import SelfEnergyModel, zero_model
    from tdagw import snapshot_runtime as runtime
    from tdagw.tdgw_io import HA_IN_EV

    class Ops(NamedTuple):
        e_qp: object
        rho_eq: object
        w_k: object
        collision: object = None

    source, mesh, _, _ = _source()
    ntr, nk, na = 1, 4, 2
    rhoeq = np.broadcast_to(np.diag([1.0, 0.0]), (nk, na, na)).astype(complex).copy()
    hcore = source["hf_reference"]["kin_ion_ry"].at[:, 1, 2].set(0.015j).at[:, 2, 1].set(-0.015j)
    source["hf_reference"]["kin_ion_ry"] = hcore
    M = np.broadcast_to(np.array([[0.04, 0.006j], [-0.006j, -0.01]]), (nk, na, na)).copy()
    basis_calls = []

    def factory(prepared, options):
        def evaluate(basis, equilibrium=False, fit_bases=None):
            basis_calls.append((equilibrium, fit_bases))
            U = basis["U"]
            current = U @ basis["rho"] @ U.conj().swapaxes(-1, -2)
            delta = current - rhoeq
            z = np.zeros_like(current)
            receipts = [dict(fit_order=1, bubble_check_error=0.0,
                             screening_check_error=0.0, noise_check_error=0.0)] * nk
            # The grossly different static table must never be subtracted
            # from the bare-HF-plus-MH Hamiltonian, in either convention.
            return dict(gamma=z, injection=z, m_h=M + 0.1 * delta,
                        static=1000.0 * delta, screening_receipts=receipts)
        return evaluate, {}

    monkeypatch.setattr(runtime, "make_snapshot_evaluator", factory)
    sh3, sh2, sh1 = (NamedSharding(mesh, P(("x", "y"), None, None)),
                     NamedSharding(mesh, P(("x", "y"), None)), NamedSharding(mesh, P(("x", "y"))))
    energies = np.asarray(source["wfns"].enk)[:, 1:3]
    ops = Ops(jax.device_put(energies * RYD_TO_EV / HA_IN_EV, sh2),
              jax.device_put(rhoeq, sh3), jax.device_put(np.full(nk, 0.25), sh1))
    reference = ops.rho_eq
    # This supplied screened kernel is deliberately inconsistent with the
    # reduced static term. Only the explicitly supplied bare-HF model may
    # contribute its density increment.
    original = SelfEnergyModel(lambda p, r, t: 7.0 * (r - p[None]), reference, True, "screened red twin")
    options = dict(statistics=statistics, hamiltonian=convention, hf_model=zero_model(), policy=C.RefreshPolicy(every=1.0),
                   snapshot_tol_ev=1e-8, snapshot_steps=8)
    built = runtime.snapshot_collision(source, ops, mesh=mesh,
        k_frac=source["wfns"].green_parent.plan.k_parent_frac,
        b_lo=1, ntr=ntr, model=original, options=options, say=lambda _: None)
    expected0 = np.asarray(hcore)[:, 1:3, 1:3] + M if convention == "hf" else (
        np.einsum("ki,ij->kij", energies, np.eye(na)))
    basis = built["state"]["basis_last"][0]
    actual0 = (basis["U"] * basis["e"][..., None, :]) @ basis["U"].conj().swapaxes(-1, -2)
    np.testing.assert_allclose(actual0, expected0, atol=1e-10)
    if convention == "reference":
        np.testing.assert_allclose(np.asarray(hcore)[:, 1:3, 1:3] + M
                                   + built["state"]["one_body_counterterm_ry"], expected0, atol=1e-12)
    else:
        assert built["t0"]["prepared_density_commutator_ha"] > 0.0
    if statistics == "retarded_reference":
        cold_calls = [cache for equilibrium, cache in basis_calls if equilibrium]
        if convention == "reference":
            assert cold_calls == [None]
        else:
            assert all(cache is cold_calls[0] for cache in cold_calls)
            assert cold_calls[0] is not None
    excited = rhoeq.copy()
    excited[:, 0, 0], excited[:, 1, 1] = 0.8, 0.2
    excited[:, 0, 1], excited[:, 1, 0] = 0.03j, -0.03j
    rho = runtime.tp.place_window_blocks(excited[None], mesh, nk)
    result = built["refresh"](2.0, rho, ops, (built["model"].params, built["held"]))
    got = built["model"].apply(result["params"][0], rho, 2.0) + result["params"][1]
    full = np.einsum("ki,ij->kij", energies, np.eye(na))[None] + np.asarray(got) * HA_IN_EV / RYD_TO_EV
    np.testing.assert_allclose(full[0], expected0 + 0.1 * (excited - rhoeq), atol=1e-9)
    if statistics == "retarded_reference":
        first_calls = [cache for equilibrium, cache in basis_calls if not equilibrium]
        assert len(first_calls) > 1
        assert first_calls[0] is not None
        assert all(cache is first_calls[0] for cache in first_calls)
        previous_count = len(basis_calls)
        built["refresh"](4.0, rho, result["ops"], result["params"])
        next_calls = [cache for equilibrium, cache in basis_calls[previous_count:] if not equilibrium]
        assert next_calls and next_calls[0] is not first_calls[0]
        assert all(cache is next_calls[0] for cache in next_calls)


@pytest.mark.parametrize("acceleration", ["linear", "anderson"])
def test_snapshot_refusal_preserves_scalar_residual_history(acceleration):
    from tdagw.dynsig import collision_snapshot_state
    history = []
    _, mesh, _, _ = _source()
    held = jax.device_put(np.zeros((1, 4, 2, 2), complex),
        NamedSharding(mesh, P(None, ("x", "y"), None, None)))
    shift = .001 * jnp.eye(2)
    def evaluate(candidate):
        return candidate + shift, None
    with pytest.raises(ValueError, match="dynsig_collision_snapshot_convergence"):
        collision_snapshot_state(evaluate, held, steps=3, tol_ha=1e-8,
            acceleration=acceleration, residual_history=history)
    np.testing.assert_allclose(history, np.full(3, .001 * np.sqrt(2)), atol=1e-14)


@pytest.mark.parametrize("with_head", [False, True])
@pytest.mark.parametrize("statistics", ["finite_lag", "lossless_bath", "retarded_reference"])
def test_snapshot_adapter_equilibrium_refresh_and_timing(with_head, statistics):
    from common.units import RYD_TO_EV
    from tdagw import collision as C
    from tdagw import propagate as tp
    from tdagw.self_energy import zero_model
    from tdagw.snapshot_runtime import snapshot_collision
    from tdagw.tdgw_io import HA_IN_EV

    class Ops(NamedTuple):
        e_qp: object
        rho_eq: object
        w_k: object
        collision: object = None

    source, mesh, _, _ = _source()
    if with_head:
        from gw.head_correction import HeadSigmaDiag
        source["efermi_ry"] = 0.137
        source["head"] = HeadSigmaDiag.build(omega_grid_ry=np.array([0.0]),
            enk_ry=np.asarray(source["wfns"].enk), efermi_ry=source["efermi_ry"],
            occupations=np.asarray(source["wfns"].occ), poles_ry=np.array([0.8 - 0.01j]),
            residues_ry=np.array([0.004]), cell_volume=12.0, nk_tot=4)
    f = np.asarray(source["wfns"].occ)[:, 1:3]
    rhoeq = np.einsum("ki,ij->kij", f, np.eye(2)).astype(complex)
    spec3 = NamedSharding(mesh, P(("x", "y"), None, None))
    spec2 = NamedSharding(mesh, P(("x", "y"), None))
    spec1 = NamedSharding(mesh, P(("x", "y")))
    ops = Ops(jax.device_put(np.asarray(source["wfns"].enk)[:, 1:3] * RYD_TO_EV / HA_IN_EV, spec2),
              jax.device_put(rhoeq, spec3), jax.device_put(np.full(4, 0.25), spec1))
    model = zero_model()
    options = dict(statistics=statistics, density_rank=1, max_order=48, fit_samples=192, fit_tolerance=2e-3,
                   adiabatic_ev=0.15, fit_height_ev=0.3, policy=C.RefreshPolicy(every=1.0),
                   # Isolate the coupled HF/Sigma runtime from adaptive
                   # rational fit order changes: both consume the identical
                   # admitted equilibrium W, with full-rho electronic lines.
                   screening="joint_frozen", hamiltonian="reference",
                   hf_model=model)
    built = snapshot_collision(source, ops, mesh=mesh,
        k_frac=source["wfns"].green_parent.plan.k_parent_frac,
        b_lo=1, ntr=1, model=model, options=options, say=lambda _: None)
    rho = tp.place_window_blocks(rhoeq[None], mesh, 4)
    update = built["refresh"](2.0, rho, ops, (built["model"].params, built["held"]))
    np.testing.assert_allclose(update["params"][1], 0.0, atol=1e-13)
    assert built["state"]["history"][-1]["snapshot"]["iterations"] == 1
    assert len(built["state"]["basis_last"]) == 1
    assert all(row["screening_s"] >= 0.0 for row in built["t0"]["q_receipts"])
    assert built["t0"]["statistics"] == statistics
    if statistics == "finite_lag":
        assert built["t0"]["screened_equilibrium_fdt"] == "known_not_generally_satisfied"
        assert built["state"]["history"][-1]["physical_lossless_statistics"] == "not_implemented"
    elif statistics == "lossless_bath":
        assert built["t0"]["screened_equilibrium_fdt"] == "same_subspace_residue_KMS"
        assert built["state"]["history"][-1]["screening_density_approximation"] == (
            "spectral_pinching_in_instantaneous_H")
        assert all(not row["lag_decay_used"] for row in built["t0"]["q_receipts"])
    else:
        assert built["t0"]["screened_equilibrium_fdt"] == "not_a_nonequilibrium_statistics_claim"
        assert built["state"]["history"][-1]["screening_density_approximation"] == (
            "full_matrix_ordered_snapshot")
    if not with_head:
        excited = rhoeq[None].copy()
        excited[..., 0, 0], excited[..., 1, 1] = 0.9, 0.1
        excited[..., 0, 1], excited[..., 1, 0] = 0.025j, -0.025j
        changed = built["refresh"](4.0, tp.place_window_blocks(excited, mesh, 4),
                                   update["ops"], update["params"])
        basis = built["state"]["basis_last"][0]
        reconstructed = ((basis["U"] * basis["e"][..., None, :])
                         @ basis["U"].conj().swapaxes(-1, -2)) * RYD_TO_EV / HA_IN_EV
        reference_h = np.einsum("ki,ij->kij", np.asarray(ops.e_qp), np.eye(2))
        np.testing.assert_allclose(reconstructed, reference_h + np.asarray(changed["params"][1])[0],
                                   rtol=1e-12, atol=1e-12)
        last = built["state"]["history"][-1]
        assert last["snapshot"]["converged"]
        assert last["snapshot"]["iterations"] > 1
        assert max(last["number_residual_rel"]) < 1e-6
        print("SNAPSHOT_COHERENT", dict(iterations=last["snapshot"]["iterations"],
            residual_ha=last["snapshot"]["residual"],
            number_residual_rel=last["number_residual_rel"],
            number_residual_per_fs=last["number_residual_per_fs"]))
    print("SNAPSHOT_RUNTIME_SMOKE", dict(setup_s=built["t0"]["setup_s"],
        q_phase_s=[{key: row[key] for key in ("project_s", "screening_s", "sigma_s")}
                   for row in built["t0"]["q_receipts"]]))
