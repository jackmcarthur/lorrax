"""The on-shell Sigma fold against its closed form, on a host mesh (CPU, seconds).

``gw.ppm_accumulators`` folds each time node into the on-shell pair
``Sigma_mn(E_m)``, ``Sigma_mn(E_n)`` and the diagonal slots (the value on each
table and dSigma_nn/domega) instead of an omega grid.  A synthetic
``sigma(t)`` and two windows (the omega >= 0 and omega < 0 halves, one of them
anti-Hermitian) are folded both ways and compared with the quadrature summed in
NumPy at the same energies:

* the pair and every value slot equal the closed form (and the grid fold) at
  the state's own energy, to round-off;
* the derivative slot equals the closed form's exact omega derivative, and a
  central difference of it;
* with band brackets, every cumulative count's slots and the kept matrices do;
* ``HeadSigmaDiag.on_shell`` is its own cells and their exact derivative.

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
                            omega_range_ev=grid, out_of_grid="cover")


def _fold(acc, A, lam, windows, frequencies, mesh):
    from common.collectives import device_put_process_local
    rep = NamedSharding(mesh, P())
    A_d, lam_d = (device_put_process_local(np.asarray(x), rep) for x in (A, lam))
    for ext, sign, pref, e_ref, t, alpha, anti in windows:
        cols = np.nonzero(frequencies < 0)[0] if ext < 0 else np.nonzero(frequencies >= 0)[0]
        acc.integrate_window(
            _tau_kernel, (A_d, lam_d), t, alpha, n_active=len(t),
            active_count=None, capacity=6, omega_sign=sign, prefactor=pref,
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
    from gw.ppm_accumulators import DeviceOmegaAccumulator, OnShellTables
    from gw.ppm_sigma import OnShellSigma, band_diagonal_slots
    from common.collectives import device_put_process_local

    rng = np.random.default_rng(7)
    mesh = _mesh()
    p_y = int(mesh.shape["y"])
    request = _request(rng)
    A, lam = _parts(n_br, rng)
    windows = _windows(rng)
    F = request.frequencies_ry
    spec = NamedSharding(mesh, P(None, None, "x", "y"))

    tables = OnShellTables(
        omega=device_put_process_local(request.tables_ry, NamedSharding(mesh, P())),
        index=request.index, omega_host=request.tables_ry)
    n_counts = n_br or 1
    shape = (((2, NK, NB, NB),) * (2 if n_counts > 1 else 1),
             ((request.n_slot, NK, NB, p_y),) * n_counts)
    acc = DeviceOmegaAccumulator(F, shape=shape, sharding=spec, omega_axis=0,
                                 reduce=_on_shell_reduce(mesh, n_br), on_shell=tables)
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
    I = request.index[0]
    for slot_m, b in zip(matrices, kept):
        got = np.asarray(slot_m)
        np.testing.assert_allclose(got[0], ref[I[:, :, None], b, k, m, n], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got[1], ref[I[:, None, :], b, k, m, n], rtol=1e-12, atol=1e-12)
    kk, bb = np.arange(NK)[:, None], np.arange(NB)[None, :]
    for b, slots in enumerate(diagonals):
        point = OnShellSigma(pair=None, slots=slots, request=request, mesh_xy=mesh)
        for name in request.names:
            Ii = request.index[request.slot(name)]
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
    acc = DeviceOmegaAccumulator(F, shape=(F.size, NK, NB, NB), sharding=spec, omega_axis=0)
    np.testing.assert_allclose(np.asarray(_fold(acc, A, lam, windows, F, mesh)),
                               _reference(A, lam, windows, F), rtol=1e-12, atol=1e-12)


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
    slots = np.searchsorted(probe.omega_grid_ry, omega)[None]
    np.testing.assert_array_equal(value, probe.at_slots(slots)[0])
    h = 1e-6
    fd = (head.on_shell(omega + h)[0] - head.on_shell(omega - h)[0]) / (2 * h)
    np.testing.assert_allclose(deriv, fd, rtol=1e-6, atol=1e-8)


def test_request_tables_alias_equal_reads():
    from gw.ppm_sigma import on_shell_request
    e = np.asarray([[-2.0, 0.5, 30.0]])
    one_shot = on_shell_request(e_eval_rel_ev=e, e_dft_rel_ev=e,
                                omega_range_ev=np.asarray([-10.0, 10.0]), out_of_grid="cover")
    assert one_shot.names == ("pair",) and one_shot.n_slot == 2
    assert one_shot.slot("dft0") == one_shot.slot("eval0") == 0
    assert one_shot.table_ev("pair")[0, 2] == 0.0           # uncovered: Sigma(omega = 0)
    assert list(one_shot.covered_kn[0]) == [True, True, False]
    clamp = on_shell_request(e_eval_rel_ev=e, e_dft_rel_ev=e + 0.1,
                             omega_range_ev=np.asarray([-10.0, 10.0]), out_of_grid="clamp")
    assert clamp.names == ("pair", "eval0", "dft0")
    assert clamp.table_ev("pair")[0, 2] == 10.0
