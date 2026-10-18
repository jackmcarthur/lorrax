"""The on-shell Sigma fold against its closed form, on a host mesh (CPU, seconds).

``gw.ppm_accumulators`` folds each time node into the on-shell pair
``Sigma_mn(E_m)``, ``Sigma_mn(E_n)`` and the diagonal slots (the value on each
table and dSigma_nn/domega) instead of an omega grid.  A synthetic
``sigma(t)`` and two windows (the omega >= 0 and omega < 0 halves, one of them
anti-Hermitian) are folded both ways and compared with the quadrature summed in
NumPy at the same energies:

* the pair and every value slot equal the closed form at the state's own
  energy, to round-off;
* the derivative slot equals the closed form's exact omega derivative, and a
  central difference of it;
* with band brackets, every cumulative count's slots and the kept matrices do;
* ``HeadSigmaDiag.on_shell`` is independent of its grid and its derivative
  is the central difference of its value;
* a single pole through the PRODUCTION box rule (``sigma_quadrature_eps``,
  ``sigma_regularization_ev`` defaults): the value slot against
  ``1/(omega - E + i eta)`` and the derivative slot against its exact
  derivative ``-1/(omega - E + i eta)^2``.
* the omega-grid fold (``sigma_omega_grid_debug``) is the same closed form, and
  its cube's diagonal read by the grid reader at the on-shell energies is the
  on-shell slots;
* the eqp2 deck keys refuse by name at parse time; ``sigma_omega_grid_debug``
  parses.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` for a 2x2 band
tiling; one host device gives the 1x1 mesh.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

NK, NB = 3, 4


def _mesh():
    devices = jax.devices("cpu")
    n = 2 if len(devices) >= 4 else 1
    return Mesh(np.array(devices[:n * n]).reshape(n, n), ("x", "y"))


def _tau_kernel(A, lam, t, active_count):
    """``sigma(t) = sum_j A_j exp(-i lam_j t)``, (..., k, nb, nb)."""
    return jnp.einsum("j...,j->...", A, jnp.exp(-1j * lam * t))


def _windows(rng):
    """Two windows: (half sign, omega_sign, prefactor, e_ref, t, alpha, antihermitian)."""
    def nodes(n):
        t = (0.4 + 0.07j) * np.arange(1, n + 1)
        return t, rng.normal(size=n) + 1j * rng.normal(size=n)
    return ((+1, -1, 0.7, 0.31, *nodes(5), False),
            (-1, +1, -1.3, -0.12, *nodes(4), True))


def _reference(A, lam, windows, omega_signed):
    """The quadrature in NumPy at each signed omega: ``(n_omega, ..., nb, nb)``."""
    S = lambda t: np.einsum("j...,j->...", A, np.exp(-1j * lam * t))
    out = np.zeros((omega_signed.size,) + A.shape[1:], np.complex128)
    for ext, sign, pref, e_ref, t, alpha, anti in windows:
        on = (omega_signed < 0) if ext < 0 else (omega_signed >= 0)
        w = ext * omega_signed[on]
        Z = sum((pref * a) * np.exp(-1j * (e_ref - sign * w) * ti)[(...,) + (None,) * (A.ndim - 1)]
                * S(ti)[None] for ti, a in zip(t, alpha))
        if anti:
            Z = (Z - np.conj(np.swapaxes(Z, -1, -2))) / 2j
        out[on] += Z
    return out


def _derivative(A, lam, windows, omega_signed):
    """The closed form's exact omega derivative, ``(n_omega, ..., nb, nb)``."""
    S = lambda t: np.einsum("j...,j->...", A, np.exp(-1j * lam * t))
    out = np.zeros((omega_signed.size,) + A.shape[1:], np.complex128)
    for ext, sign, pref, e_ref, t, alpha, anti in windows:
        on = (omega_signed < 0) if ext < 0 else (omega_signed >= 0)
        w = ext * omega_signed[on]
        Z = sum((1j * sign * ext * ti * pref * a)
                * np.exp(-1j * (e_ref - sign * w) * ti)[(...,) + (None,) * (A.ndim - 1)]
                * S(ti)[None] for ti, a in zip(t, alpha))
        if anti:
            Z = (Z - np.conj(np.swapaxes(Z, -1, -2))) / 2j
        out[on] += Z
    return out


def _request(rng):
    """Two tables (pair, dft0) of per-state energies straddling omega = 0."""
    from gw.ppm_sigma import on_shell_request
    from common.units import RYD_TO_EV
    e_eval = rng.uniform(-0.6, 0.6, size=(NK, NB)) * RYD_TO_EV
    e_dft = e_eval + rng.uniform(-0.05, 0.05, size=(NK, NB)) * RYD_TO_EV
    grid = np.linspace(-1.0, 1.0, 9) * RYD_TO_EV
    return on_shell_request(e_eval_rel_ev=e_eval, e_dft_rel_ev=e_dft,
                            omega_range_ev=grid, out_of_grid="cover",
                            irr_idx=np.arange(NK), n_parent=NK)


def _fold(acc, A, lam, windows, frequencies, mesh):
    from common.collectives import device_put_process_local
    rep = NamedSharding(mesh, P())
    A_d, lam_d = (device_put_process_local(np.asarray(x), rep) for x in (A, lam))
    for ext, sign, pref, e_ref, t, alpha, anti in windows:
        cols = np.nonzero(frequencies < 0)[0] if ext < 0 else np.nonzero(frequencies >= 0)[0]
        acc.integrate_window(
            _tau_kernel, (A_d, lam_d), t, alpha, active_count=None, capacity=max(6, len(t)), omega_sign=sign, prefactor=pref,
            e_ref_sum=e_ref, antihermitian=anti, omega_indices=cols,
            omega_values=ext * frequencies[cols])
    return acc.finalize()


def _parts(n_br, rng):
    lam = rng.uniform(-0.5, 0.5, size=3)
    shape = (3,) + ((n_br,) if n_br else ()) + (NK, NB, NB)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape), lam


@pytest.mark.parametrize("n_br", [None, 3])
def test_on_shell_fold_matches_closed_form(n_br):
    from gw.mpa.sigma import _on_shell_reduce
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    from gw.ppm_sigma import OnShellSigma

    rng = np.random.default_rng(7)
    mesh = _mesh()
    p_y = int(mesh.shape["y"])
    request = _request(rng)
    A, lam = _parts(n_br, rng)
    windows = _windows(rng)
    F = request.frequencies_ry
    spec = NamedSharding(mesh, P(None, None, "x", "y"))

    n_counts = n_br or 1
    shape = (((2, NK, NB, NB),) * (2 if n_counts > 1 else 1),
             ((request.n_slot, NK, NB, p_y),) * n_counts)
    acc = DeviceOmegaAccumulator(request, carrier=NB, shape=shape, sharding=spec,
                                 reduce=_on_shell_reduce(mesh, n_br))
    matrices, diagonals = _fold(acc, A, lam, windows, F, mesh)

    # The closed form at every list frequency, cumulative over brackets.
    ref = _reference(A, lam, windows, F)
    dref = _derivative(A, lam, windows, F)
    if n_br:
        ref, dref = np.cumsum(ref, axis=1), np.cumsum(dref, axis=1)
        kept = (0, n_br - 1)
    else:
        ref, dref = ref[:, None], dref[:, None]
        kept = (0,)
    k = np.arange(NK)[:, None, None]
    m = np.arange(NB)[None, :, None]
    n = np.arange(NB)[None, None, :]
    I = request.parent_index[0]
    for slot_m, b in zip(matrices, kept):
        got = np.asarray(slot_m)
        np.testing.assert_allclose(got[0], ref[I[:, :, None], b, k, m, n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got[1], ref[I[:, None, :], b, k, m, n], rtol=1e-12, atol=1e-12)
    kk, bb = np.arange(NK)[:, None], np.arange(NB)[None, :]
    for b, slots in enumerate(diagonals):
        point = OnShellSigma(pair=None, slots=slots, request=request, mesh_xy=mesh)
        for name in request.names:
            Ii = request.parent_index[request.slot(name)]
            np.testing.assert_allclose(point.diagonal(name), ref[Ii, b, kk, bb, bb],
                                       rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(point.diagonal("derivative"), dref[I, b, kk, bb, bb],
                                   rtol=1e-11, atol=1e-11)
        # Every slot column but the owner's is an exact zero.
        owner = (np.arange(NB)[:, None] // (NB // p_y)) == np.arange(p_y)[None, :]
        assert not np.any(np.asarray(slots)[..., ~owner])

    # The exact derivative is the central difference of the closed form.
    h = 1.0e-6
    E = request.tables_ry[0]
    fd = (_reference(A, lam, windows, (E + h).ravel()) - _reference(A, lam, windows, (E - h).ravel())) / (2 * h)
    fd = fd.reshape(E.shape + fd.shape[1:])              # (k, b, [bracket,] k, nb, nb)
    if n_br:
        fd = fd.sum(axis=2)                                  # the last cumulative count
    np.testing.assert_allclose(dref[I, -1, kk, bb, bb], fd[kk, bb, kk, bb, bb],
                               rtol=1e-6, atol=1e-6)


def test_grid_fold_matches_closed_form():
    """The grid fold (``sigma_omega_grid_debug``) is the same closed form."""
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    rng = np.random.default_rng(3)
    mesh = _mesh()
    A, lam = _parts(None, rng)
    windows = _windows(rng)
    F = np.linspace(-0.5, 0.5, 7)
    spec = NamedSharding(mesh, P(None, None, "x", "y"))
    acc = DeviceOmegaAccumulator.grid(F, shape=(F.size, NK, NB, NB), sharding=spec, omega_axis=0)
    np.testing.assert_allclose(np.asarray(_fold(acc, A, lam, windows, F, mesh)),
                               _reference(A, lam, windows, F), rtol=1e-12, atol=1e-12)


def test_grid_cube_diagonal_at_on_shell_energies_is_the_on_shell_slots():
    """The debug cube and the on-shell slots are one quadrature: folded on the grid
    of the request's own frequencies, the grid reader (``qsgw_utils.
    interp_sigma_diag_along_omega``, the eqp0 at-E_DFT read) at each table's
    energies returns the on-shell diagonal of that table, to round-off."""
    from common.units import RYD_TO_EV
    from gw.mpa.sigma import _on_shell_reduce
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    from gw.ppm_sigma import OnShellSigma
    from gw.qsgw_utils import interp_sigma_diag_along_omega
    rng = np.random.default_rng(11)
    mesh = _mesh()
    p_y = int(mesh.shape["y"])
    request = _request(rng)
    A, lam = _parts(None, rng)
    windows = _windows(rng)
    F = request.frequencies_ry
    spec = NamedSharding(mesh, P(None, None, "x", "y"))
    on_shell = DeviceOmegaAccumulator(
        request, carrier=NB, sharding=spec, reduce=_on_shell_reduce(mesh, None),
        shape=(((2, NK, NB, NB),), ((request.n_slot, NK, NB, p_y),)))
    (pair,), (slots,) = _fold(on_shell, A, lam, windows, F, mesh)
    point = OnShellSigma(pair=pair, slots=slots, request=request, mesh_xy=mesh)
    grid = DeviceOmegaAccumulator.grid(F, shape=(F.size, NK, NB, NB), sharding=spec, omega_axis=0)
    cube = _fold(grid, A, lam, windows, F, mesh)
    for name in request.names:
        e_ev = request.tables_ry[request.slot(name)] * RYD_TO_EV
        read = interp_sigma_diag_along_omega(cube, mesh, F * RYD_TO_EV, e_ev)
        np.testing.assert_allclose(read, point.diagonal(name), rtol=1e-12, atol=1e-12)
    # The pair: Sigma_mn at the row and the column state's energy.
    I = request.parent_index[0]
    k = np.arange(NK)[:, None, None]
    m, n = np.arange(NB)[None, :, None], np.arange(NB)[None, None, :]
    full = np.asarray(cube)
    np.testing.assert_allclose(np.asarray(pair)[0], full[I[:, :, None], k, m, n], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pair)[1], full[I[:, None, :], k, m, n], rtol=1e-12, atol=1e-12)


def test_head_on_shell_is_its_cells_and_derivative():
    from gw.head_correction import HeadSigmaDiag
    rng = np.random.default_rng(11)
    enk = rng.uniform(-0.5, 0.5, size=(NK, NB))
    occ = (enk < 0.0).astype(np.float64)
    head = HeadSigmaDiag.build(
        omega_grid_ry=np.linspace(-1, 1, 5), enk_ry=enk, efermi_ry=0.0, occupations=occ,
        poles_ry=np.asarray([0.8 - 1e-6j, 1.3 - 0.2j]), residues_ry=np.asarray([0.4, -0.1]),
        cell_volume=12.0, nk_tot=NK)
    omega = rng.uniform(-0.4, 0.4, size=(NK, NB))
    value, deriv = head.on_shell(omega)
    probe = HeadSigmaDiag.build(
        omega_grid_ry=np.sort(omega.ravel()), enk_ry=enk, efermi_ry=0.0, occupations=occ,
        poles_ry=head.poles_ry, residues_ry=head.residues_ry, cell_volume=12.0, nk_tot=NK)
    np.testing.assert_array_equal(value, probe.on_shell(omega)[0])
    h = 1e-6
    fd = (head.on_shell(omega + h)[0] - head.on_shell(omega - h)[0]) / (2 * h)
    np.testing.assert_allclose(deriv, fd, rtol=1e-6, atol=1e-8)


def test_request_tables_alias_equal_reads():
    from gw.ppm_sigma import on_shell_request
    e = np.asarray([[-2.0, 0.5, 30.0]])
    k_plan = dict(irr_idx=np.arange(1), n_parent=1)
    one_shot = on_shell_request(e_eval_rel_ev=e, e_dft_rel_ev=e,
                                omega_range_ev=np.asarray([-10.0, 10.0]), out_of_grid="cover",
                                **k_plan)
    assert one_shot.names == ("pair",) and one_shot.n_slot == 2
    assert one_shot.slot("dft0") == one_shot.slot("eval0") == 0
    assert one_shot.table_ev("pair")[0, 2] == 0.0           # uncovered: Sigma(omega = 0)
    assert list(one_shot.covered_kn[0]) == [True, True, False]
    clamp = on_shell_request(e_eval_rel_ev=e, e_dft_rel_ev=e + 0.1,
                             omega_range_ev=np.asarray([-10.0, 10.0]), out_of_grid="clamp",
                             **k_plan)
    assert clamp.names == ("pair", "eval0", "dft0")
    assert clamp.table_ev("pair")[0, 2] == 10.0
    census = on_shell_request(e_eval_rel_ev=e, e_dft_rel_ev=e,
                              omega_range_ev=np.asarray([-10.0, 10.0]), out_of_grid="static",
                              edge_window_ev=(-12.0, 12.0), **k_plan)
    assert census.names == ("pair", "zero", "edge")      # eval0/dft0 alias pair
    assert list(census.table_ev("edge")[0]) == [-10.0, 10.0, 10.0]


def _production_pole_nodes(frequencies_ry, pole_ry, eta_ry, eps):
    """The production Sigma box rule for one pole ``pole_ry``, ``(t, alpha)``.

    The planner's chain for one conduction window at one live state of energy
    zero: :func:`gw.sigma_box_plan.make_sigma_box_spec` -> the built rule
    (``analytic_box_rule`` on the outward-snapped build box) ->
    :func:`gw.sigma_box_plan._accept_rule` -> the executor nodes, so that
    ``sum_i alpha_i exp(i t_i (omega - E))`` approximates ``1/(omega - E + i eta)``.
    """
    from gw import sigma_box_plan as plan
    spec = plan.make_sigma_box_spec(
        name="single-pole", frequencies=frequencies_ry, states=np.zeros(1),
        pole_stats=[(pole_ry, pole_ry, 0.0, 0.0)], pole_sign=+1, eta_ry=eta_ry)
    rule = plan._BOX_RULE_BUILDER(plan._build_box(spec["box"], eta_ry, widen=True), eps,
                                  mass_cap=plan._noise_amplification_cap())
    fit = plan._accept_rule(spec, rule, eps, rule_source="built")
    nodes = plan.sigma_box_executor_nodes(fit, +1, eta_ry)
    return np.asarray(nodes.t), np.asarray(nodes.alpha), spec["kind"]


#: The derivative slot's error bound, in the derivative's box currency (eta^-2
#: crossing, |d|^-2 sign-definite: the square of the value's currency), as a
#: multiple of eps.  Measured 1.3 eps (crossing) and 0.18 eps (sign-definite).
DERIVATIVE_TOL_EPS = 10.0


@pytest.mark.parametrize("pole_ev", [5.0, -3.0])
def test_single_pole_derivative_at_production_eps_eta(pole_ev, capsys):
    """dQ/domega of one pole against -1/(omega - E + i eta)^2 (production eps, eta).

    Frequencies span [0, 10] eV: ``E = 5`` eV is a crossing box, ``E = -3`` eV
    a sign-definite one.  The value slot is certified by the rule; the
    derivative slot is not, and this measures it on the same nodes.
    """
    from common.units import RYD_TO_EV
    from gw.gw_config import _DEFAULTS
    from gw.mpa.sigma import _on_shell_reduce
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    from gw.ppm_sigma import OnShellSigma, on_shell_request

    eps = float(_DEFAULTS["sigma_quadrature_eps"])
    eta = float(_DEFAULTS["sigma_regularization_ev"]) / RYD_TO_EV
    mesh = _mesh()
    p_y = int(mesh.shape["y"])
    omega_ev = np.linspace(0.0, 10.0, NK * NB).reshape(NK, NB)
    request = on_shell_request(e_eval_rel_ev=omega_ev, e_dft_rel_ev=omega_ev,
                               omega_range_ev=np.asarray([-1.0, 11.0]), out_of_grid="cover",
                               irr_idx=np.arange(NK), n_parent=NK)
    F = request.frequencies_ry
    pole = pole_ev / RYD_TO_EV
    t, alpha, kind = _production_pole_nodes(F, pole, eta, eps)

    spec = NamedSharding(mesh, P(None, None, "x", "y"))
    acc = DeviceOmegaAccumulator(
        request, carrier=NB, shape=(((2, NK, NB, NB),), ((request.n_slot, NK, NB, p_y),)),
        sharding=spec, reduce=_on_shell_reduce(mesh, None))
    A = np.ones((1, NK, NB, NB), np.complex128)
    window = ((+1, +1, 1.0, 0.0, t, alpha, False),)
    _, (slots,) = _fold(acc, A, np.asarray([pole]), window, F, mesh)
    point = OnShellSigma(pair=None, slots=slots, request=request, mesh_xy=mesh)

    w = request.table_ev("pair") / RYD_TO_EV
    d = w - pole + 1j * eta
    value, deriv = point.diagonal("pair"), point.diagonal("derivative")
    rho = eta if kind == "crossing" else np.abs(d)          # the value's box currency
    value_err = float(np.max(np.abs(value - 1.0 / d) * rho))
    deriv_err = np.abs(deriv + 1.0 / d ** 2)
    deriv_box = float(np.max(deriv_err * rho ** 2))
    deriv_rel = float(np.max(deriv_err * np.abs(d) ** 2))
    with capsys.disabled():
        print(f"\n  single pole E={pole_ev:+.1f} eV ({kind}, {t.size} nodes, eps={eps:g}, "
              f"eta={eta * RYD_TO_EV:g} eV): value err {value_err:.2e}, dSigma/domega err "
              f"{deriv_box:.2e} (box currency); dSigma/domega max relative {deriv_rel:.2e}")
    assert value_err <= 1.05 * eps
    assert deriv_box <= DERIVATIVE_TOL_EPS * eps


@pytest.mark.parametrize("key, value", [("write_eqp2", "true"), ("eqp2_tol_ev", "1e-3"),
                                        ("eqp2_max_iter", "20"), ("eqp2_file", "eqp2.dat")])
def test_eqp2_keys_refuse_by_name(tmp_path, key, value):
    """The keys of the retired eqp2 ladder refuse at parse time, by name, any value."""
    from gw.gw_config import read_lorrax_input
    deck = tmp_path / "cohsex.in"
    deck.write_text(f"[cohsex]\nsys_dim = 3\n{key} = {value}\n")
    with pytest.raises(ValueError, match="GATE eqp2_retired") as refusal:
        read_lorrax_input(str(deck))
    assert key in str(refusal.value) and "UPGRADE_NOTES.md" in str(refusal.value)


def test_omega_grid_debug_key_is_accepted():
    """``sigma_omega_grid_debug`` is a live debug key again (default off): the early
    retired-key gate passes it and the debug group carries it."""
    import configparser
    import dataclasses
    from gw import gw_config
    parser = configparser.ConfigParser()
    parser.read_string("[cohsex]\nsys_dim = 3\nsigma_omega_grid_debug = true\n")
    gw_config._report_early_retired_keys(parser["cohsex"])
    assert gw_config._DEFAULTS["sigma_omega_grid_debug"] is False
    assert "sigma_omega_grid_debug" not in gw_config._LEGACY_DECK_KEYS
    assert "sigma_omega_grid_debug" in {f.name for f in dataclasses.fields(gw_config.DebugConfig)}


def _sweep_identity_of(tmp_path, request, e_qp_ev):
    """``sigma_dispatch._sweep_checkpoint`` for a one-shot on a toy deck."""
    from types import SimpleNamespace
    from gw.sigma_dispatch import _sweep_checkpoint
    config = SimpleNamespace(write_restart_tensors=True,
                             sigma=SimpleNamespace(band_extrapolation=False))
    options = dict(efermi_ry=0.1, edge_factor=2.0, print_fn=print)
    if request is not None:
        options["request"] = request
    return _sweep_checkpoint(
        config, tmp_path, {"iteration_id": "oneshot"}, "w-digest", None, False,
        e_qp_ev=e_qp_ev, body_options=options, efermi_ry=0.1,
        band_slices=SimpleNamespace(b0=0, b3=NB, sigma_range="0:4"), request=request)


class _OneProcessSlabs:
    """``SlabIO`` on one process through h5py.  The per-rank tile transport is
    not under test here (the Fe legs check it); the record, digest, commit,
    atomic publish and identity check of ``file_io.sigma_checkpoint`` are."""

    def __init__(self, path, mode, mesh):
        import h5py
        self.f = h5py.File(path, mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()

    def create_dataset(self, name, shape, dtype):
        self.f.create_dataset(name, shape=shape, dtype=dtype)

    def write_slab(self, name, cube):
        self.f[name][...] = np.asarray(cube)

    def read_slab(self, name, *, dtype, mesh, partition_spec):
        return jax.device_put(self.f[name][()].astype(dtype),
                              NamedSharding(mesh, partition_spec))


def test_on_shell_sweep_checkpoint_round_trip(tmp_path, capsys, monkeypatch):
    """The on-shell sweep checkpoint (``tmp/sigma_checkpoint_oneshot_on_shell.h5``):
    the pair and slots read back bit for bit onto the live request; moved
    on-shell energies (E_eval, or E_DFT alone under ``clamp``) and the grid
    evaluation have other identities, and a stale file is removed by name."""
    import file_io.slab_io
    monkeypatch.setattr(file_io.slab_io, "SlabIO", _OneProcessSlabs)
    from file_io.sigma_checkpoint import (
        read_sigma_checkpoint, sweep_identity, write_sigma_checkpoint)
    from gw.ppm_sigma import OnShellSigma, on_shell_request
    from gw.sigma_dispatch import _checkpoint_arrays, _checkpoint_values

    mesh = _mesh()
    rng = np.random.default_rng(11)
    spec = NamedSharding(mesh, P(None, None, "x", "y"))
    e = rng.uniform(-3.0, 3.0, (NK, NB))

    def request_at(e_eval, e_dft):
        return on_shell_request(e_eval_rel_ev=e_eval, e_dft_rel_ev=e_dft,
                                omega_range_ev=np.asarray([-10.0, 10.0]),
                                out_of_grid="clamp", irr_idx=np.arange(NK), n_parent=NK)

    def cplx(shape):
        return jax.device_put(rng.normal(size=shape) + 1j * rng.normal(size=shape), spec)
    request = request_at(e, e + 0.1)
    p_y = mesh.shape["y"]
    body = OnShellSigma(pair=cplx((2, NK, NB, NB)), slots=cplx((request.n_slot, NK, NB, p_y)),
                        request=request, mesh_xy=mesh)
    twin = OnShellSigma(pair=None, slots=cplx((request.n_slot, NK, NB, p_y)),
                        request=request, mesh_xy=mesh)
    cubes = dict(body=body, odd=None, unextrap=twin)
    path, identity = _sweep_identity_of(tmp_path, request, e)
    assert path.name == "sigma_checkpoint_oneshot_on_shell.h5"
    path.parent.mkdir()
    write_sigma_checkpoint(path, identity=identity, cubes=_checkpoint_arrays(cubes, request),
                           host=dict(band_axis=None, head_diag=None), mesh=mesh)
    arrays, host, _, _ = read_sigma_checkpoint(path, identity=identity, mesh=mesh)
    got = _checkpoint_values(arrays, request, mesh)
    assert got["odd"] is None and got["unextrap"].pair is None
    assert got["body"].request is request and got["unextrap"].request is request
    for name in ("body", "unextrap"):
        for part in ("pair", "slots"):
            want = getattr(cubes[name], part)
            if want is not None:
                have = getattr(got[name], part)
                assert have.sharding.spec == want.sharding.spec
                assert np.asarray(have).tobytes() == np.asarray(want).tobytes()
    # A value on another request is not stored under the live one.
    other = request_at(e, e + 0.1)
    assert _checkpoint_arrays(dict(body=OnShellSigma(
        pair=body.pair, slots=body.slots, request=other, mesh_xy=mesh)), request) is None

    # Other energies, or the grid evaluation: other identities and files.
    moved_eval = _sweep_identity_of(tmp_path, request_at(e + 1e-6, e + 0.1), e + 1e-6)[1]
    moved_dft = _sweep_identity_of(tmp_path, request_at(e, e + 0.1 + 1e-6), e)[1]
    grid_path, grid = _sweep_identity_of(tmp_path, None, e)
    assert len({identity, moved_eval, moved_dft, grid}) == 4
    assert grid_path.name == "sigma_checkpoint_oneshot.h5"
    # The grid identity is the one main wrote before the on-shell payload.
    assert grid == sweep_identity(
        schema="sigma-sweep-checkpoint-v1", w_digest="w-digest",
        w_identity={"iteration_id": "oneshot"}, head=None,
        e_qp_ev=np.asarray(e, np.float64), options=dict(efermi_ry=0.1, edge_factor=2.0),
        sigma_bands=[0, NB, "0:4"], band_extrapolation=False, efermi_ry=0.1)
    capsys.readouterr()
    assert read_sigma_checkpoint(path, identity=moved_dft, mesh=mesh) == (
        None, "removed (identity differs)")
    assert f"removed (identity differs) {path}; recomputing" in capsys.readouterr().out
    assert not path.exists()
