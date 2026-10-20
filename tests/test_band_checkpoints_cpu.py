"""``band_extrapolation_checkpoints`` on a host mesh (CPU, seconds).

One on-shell sweep over the UNION of several counts' bracket cuts serves
every checkpoint count.  A synthetic per-band ``sigma(t)`` (a decaying
band tail with state-dependent phases) is folded

* once over the union plan, with the run's N1/N3 matrices kept
  (``_on_shell_reduce(..., keep)``), and
* once per checkpoint ``N`` over that count's OWN three-bracket plan
  (``plan_band_brackets`` on ``[0, N)``: the plan of a run with
  ``number_bands_sigma = N``, the band sum restricted to ``N`` bands),

with the same windows.  Each checkpoint read off the union sweep
(``evaluate_band_checkpoints``) must equal, to round-off, the production
extrapolation (``extrapolate_sigma_body``) of its restricted fold: the raw
``S(N)``, the shell ``S_hat(N)`` and Z on every table.  The run's own
extrapolation through the union cube must equal the plain three-bracket run.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` for a 2x2 band
tiling; one host device gives the 1x1 mesh.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

NK, NB, NBAND, NOCC = 3, 4, 48, 2
RY = 13.605693122994


def _mesh():
    devices = jax.devices("cpu")
    n = 2 if len(devices) >= 4 else 1
    return Mesh(np.array(devices[:n * n]).reshape(n, n), ("x", "y"))


def _tau_kernel(A, lam, t, active_count):
    """``sigma_b(t) = sum_j A_jb exp(-i lam_j t)`` for every bracket b."""
    return jnp.einsum("j...,j->...", A, jnp.exp(-1j * lam * t))


def _system(rng):
    """Band energies (Ry), one per (k, band): a free-electron-like ladder."""
    m = np.arange(NBAND)
    base = -0.6 + 0.05 * (m + 1) ** (2.0 / 3.0) * 8.0
    enk = base[None, :] + 0.01 * rng.normal(size=(NK, NBAND))
    enk.sort(axis=1)
    return enk


def _per_band(rng, enk):
    """``A[j, m, k, a, b]``: band m's contribution, a beta = 3 tail with phases."""
    lam = rng.uniform(-0.5, 0.5, size=3)
    gap = np.abs(enk[:, :, None] - enk[:, None, :NB])                    # (k, m, a)
    amp = 1.0 / (gap + 0.6) ** 3
    A = np.zeros((3, NBAND, NK, NB, NB), np.complex128)
    for j in range(3):
        phase = rng.normal(size=(NBAND, NK, NB, NB)) + 1j * rng.normal(size=(NBAND, NK, NB, NB))
        herm = 0.5 * (phase + np.conj(np.swapaxes(phase, -1, -2)))
        a = np.transpose(amp, (1, 0, 2))                                  # (m, k, a)
        scale = np.sqrt(a[:, :, :, None] * a[:, :, None, :])
        A[j] = (0.2 + 0.1 * j) * scale * (1.0 + 0.05 * herm)
    return A, lam


def _windows(rng):
    def nodes(n):
        t = (0.4 + 0.07j) * np.arange(1, n + 1)
        return t, rng.normal(size=n) + 1j * rng.normal(size=n)
    return ((+1, -1, 0.7, 0.31, *nodes(5), False),
            (-1, +1, -1.3, -0.12, *nodes(4), True))


def _request(enk):
    from gw.ppm_sigma import on_shell_request
    e = enk[:, :NB] * RY
    return on_shell_request(e_eval_rel_ev=e, e_dft_rel_ev=e,
                            omega_range_ev=np.linspace(-1.0, 1.0, 9) * RY,
                            out_of_grid="cover", irr_idx=np.arange(NK), n_parent=NK)


def _fold(bounds, A, lam, windows, request, mesh, keep=None):
    """The executor's on-shell fold over ``bounds``: a BandCountCube on the full BZ."""
    from common.collectives import device_put_process_local
    from gw.mpa.sigma import _on_shell_reduce
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    from gw.ppm_sigma import BandCountCube
    A_br = np.stack([A[:, lo:hi].sum(axis=1) for lo, hi in bounds], axis=1)
    n_br, p_y = len(bounds), int(mesh.shape["y"])
    rep = NamedSharding(mesh, P())
    spec = NamedSharding(mesh, P(None, None, "x", "y"))
    n_matrix = len(keep) if keep is not None else (2 if n_br > 1 else 1)
    shape = (((2, NK, NB, NB),) * n_matrix, ((request.n_slot, NK, NB, p_y),) * n_br)
    F = request.frequencies_ry
    acc = DeviceOmegaAccumulator(request, carrier=NB, shape=shape, sharding=spec,
                                 reduce=_on_shell_reduce(mesh, n_br, keep))
    A_d, lam_d = (device_put_process_local(np.asarray(x), rep) for x in (A_br, lam))
    for ext, sign, pref, e_ref, t, alpha, anti in windows:
        cols = np.nonzero(F < 0)[0] if ext < 0 else np.nonzero(F >= 0)[0]
        acc.integrate_window(_tau_kernel, (A_d, lam_d), t, alpha,
                             active_count=None, capacity=max(6, len(t)), omega_sign=sign, prefactor=pref,
                             e_ref_sum=e_ref, antihermitian=anti, omega_indices=cols,
                             omega_values=ext * F[cols])
    matrices, diagonals = acc.finalize()
    return BandCountCube(matrices=list(matrices), diagonals=[], unfold=lambda v: v,
                         unfold_rows=lambda v: v, nk=NK, fresh=False, mesh_xy=mesh,
                         request=request, slot_diagonals=list(diagonals),
                         matrix_index=() if keep is None else tuple(keep))


def _context(enk):
    wfn = SimpleNamespace(
        energies=np.asarray(enk)[None], kweights=np.full(NK, 1.0 / NK),
        ngk=np.full(NK, NBAND + 12), nspinor=1, efermi=0.0,
        symmetry=lambda: SimpleNamespace(kirr_fullids=np.arange(NK)))
    config = SimpleNamespace(
        omega_grid_ev=np.linspace(-1.0, 1.0, 9) * RY,
        sigma=SimpleNamespace(band_extrapolation_estimator="spectral_shell"),
        debug=SimpleNamespace(sigma_freq_debug_output=False),
        paths=SimpleNamespace(sigma_diag_file="sigma_diag.dat"))
    slices = SimpleNamespace(b0=0, b2=NOCC)
    return wfn, config, slices


def _result(cube, counts):
    from gw.ppm_sigma import SigmaOmegaResult
    return SigmaOmegaResult(sigma_c_kij=cube, band_counts=tuple(counts), efermi_ry=0.0)


def _z(point):
    """``1/(1 - Re dSigma_nn/domega)`` at E_eval, 1 off the sampled range."""
    d = np.real(point.diagonal("derivative"))
    return np.where(point.request.covered_kn, 1.0 / (1.0 - d), 1.0)


def _plans(enk, checkpoints):
    from gw.band_extrapolation import plan_band_checkpoints
    return plan_band_checkpoints(checkpoints=checkpoints, enk_ry=enk, n_occ=NOCC,
                                 nb_logical=NBAND, nb_padded=NBAND)


def test_checkpoint_plan_members_are_their_own_runs():
    from gw.band_extrapolation import parse_band_checkpoints, plan_band_brackets
    enk = _system(np.random.default_rng(1))
    assert parse_band_checkpoints("") == ()
    assert parse_band_checkpoints("48, 20 30") == (20, 30, 48)
    for bad in ("12, 12", "0", "1.5", "x"):
        with pytest.raises(ValueError):
            parse_band_checkpoints(bad)
    plan = _plans(enk, (20, 30, 48))
    ck = plan.checkpoints
    plain = plan_band_brackets(enabled=True, enk_ry=enk, n_occ=NOCC, nb_logical=NBAND,
                               nb_padded=NBAND)
    assert plan.counts == plain.counts and plan.bounds == plain.bounds
    for n, member, idx in zip((20, 30, 48), ck.members, ck.member_index):
        own = plan_band_brackets(enabled=True, enk_ry=enk, n_occ=NOCC, nb_logical=n,
                                 nb_padded=n)
        assert member.counts == own.counts and member.counts[-1] == n
        assert tuple(ck.union.counts[i] for i in idx) == own.counts
    u = ck.union
    assert u.bounds[0][0] == 0 and u.bounds[-1][1] == NBAND
    assert all(a[1] == b[0] for a, b in zip(u.bounds, u.bounds[1:]))
    assert list(u.counts) == sorted(set(u.counts)) and u.counts[-1] == NBAND
    assert plan.sweep_matrix_counts == (ck.main_index[0], ck.main_index[2])
    # Unset: the plan and the sweep are the plain run's objects.
    assert plain.checkpoints is None and plain.sweep_bounds is plain.bounds
    assert plain.sweep_counts is plain.counts and plain.sweep_matrix_counts is None
    with pytest.raises(Exception):
        _plans(enk, (20, NBAND + 1))
    # A checkpoint that splits a degenerate pair refuses by name.
    from gw.band_extrapolation import BandExtrapolationRefused
    deg = enk.copy()
    deg[:, 24] = deg[:, 23]
    with pytest.raises(BandExtrapolationRefused, match="split a degenerate"):
        _plans(deg, (24, 30))
    _plans(deg, (23, 30))


def test_checkpoints_equal_their_restricted_plans_and_the_run_is_unchanged(tmp_path):
    from gw.ppm_pipeline import evaluate_band_checkpoints, extrapolate_sigma_body
    rng = np.random.default_rng(7)
    mesh = _mesh()
    enk = _system(rng)
    request = _request(enk)
    A, lam = _per_band(rng, enk)
    windows = _windows(rng)
    wfn, config, slices = _context(enk)
    e_state = enk[:, :NB] * RY
    checkpoints = (20, 30, NBAND)
    plan = _plans(enk, checkpoints)
    ck = plan.checkpoints

    union = _fold(ck.union.bounds, A, lam, windows, request, mesh,
                  keep=plan.sweep_matrix_counts)
    payload = evaluate_band_checkpoints(
        _result(union, ck.union.counts), None, e_state_ev=e_state, plan=plan,
        band_slices=slices, wfn=wfn)

    for member, got in zip(ck.members, payload["members"]):
        # The plan restricted to N: its own three brackets over [0, N).
        cube = _fold(member.bounds, A, lam, windows, request, mesh)
        ref, _, twin = extrapolate_sigma_body(
            _result(cube, member.counts), None, e_state_ev=e_state, plan=member,
            config=config, band_slices=slices, wfn=wfn, mesh_xy=mesh, print_fn=lambda *a: None)
        assert got["count"] == member.counts[-1] and got["counts"] == member.counts
        for name in ("pair", "dft0", "derivative"):
            np.testing.assert_allclose(got["shell"][name], ref.diagonal(name),
                                       rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(got["raw"][name], twin.diagonal(name),
                                       rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(got["shell"]["z"], _z(ref), rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(got["raw"]["z"], _z(twin), rtol=1e-11, atol=1e-13)
        assert got["fit_vs_combination_ev"] < 1e-10
        # Something to extrapolate: the shell moved the raw sum on most states.
        if member.counts[-1] < NBAND:
            moved = np.abs(got["shell"]["pair"] - got["raw"]["pair"]) > 1e-12
            assert moved.mean() > 0.5

    # The run's own count through the union cube equals the plain 3-bracket run.
    from gw.ppm_pipeline import write_band_checkpoints
    config.paths = SimpleNamespace(sigma_diag_file=str(tmp_path / "sigma_diag.dat"))
    union2 = _fold(ck.union.bounds, A, lam, windows, request, mesh,
                   keep=plan.sweep_matrix_counts)
    via_union, _, twin_u = extrapolate_sigma_body(
        _result(union2, ck.union.counts), None, e_state_ev=e_state, plan=plan,
        config=config, band_slices=slices, wfn=wfn, mesh_xy=mesh, print_fn=lambda *a: None)
    plain = _fold(plan.bounds, A, lam, windows, request, mesh)
    via_plain, _, twin_p = extrapolate_sigma_body(
        _result(plain, plan.counts), None, e_state_ev=e_state, plan=dataclass_plain(plan),
        config=config, band_slices=slices, wfn=wfn, mesh_xy=mesh, print_fn=lambda *a: None)
    np.testing.assert_allclose(np.asarray(via_union.pair), np.asarray(via_plain.pair),
                               rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(np.asarray(via_union.slots), np.asarray(via_plain.slots),
                               rtol=1e-11, atol=1e-13)
    assert (tmp_path / "sigma_band_checkpoints.h5").exists()
    rows = (tmp_path / "sigma_band_checkpoints.dat").read_text().splitlines()
    assert len([r for r in rows if not r.startswith("#")]) == len(checkpoints) * NK * NB


def dataclass_plain(plan):
    import dataclasses
    return dataclasses.replace(plan, checkpoints=None)


def test_gate_refuses_checkpoints_off_the_scalar_on_shell_route():
    from gw.gw_config import ComputeMode
    from gw.sigma_dispatch import _validate_band_checkpoints
    base = dict(band_extrapolation=True, band_extrapolation_checkpoints=(20, 48))
    cfg = lambda **kw: SimpleNamespace(
        sigma=SimpleNamespace(**{**base, **kw.get("sigma", {})}),
        debug=SimpleNamespace(sigma_omega_grid_debug=kw.get("grid", False)),
        bispinor=False, qp_solver=SimpleNamespace(value=kw.get("qp", "one_shot_dft")))
    _validate_band_checkpoints(cfg(), ComputeMode.MPA)
    _validate_band_checkpoints(cfg(sigma=dict(band_extrapolation_checkpoints=())),
                               ComputeMode.GN_PPM)
    for bad, mode in ((cfg(sigma=dict(band_extrapolation=False)), ComputeMode.MPA),
                      (cfg(grid=True), ComputeMode.MPA),
                      (cfg(), ComputeMode.GN_PPM),
                      (cfg(qp="self_consistent"), ComputeMode.MPA)):
        with pytest.raises(ValueError, match="GATE band_extrapolation_checkpoints"):
            _validate_band_checkpoints(bad, mode)
