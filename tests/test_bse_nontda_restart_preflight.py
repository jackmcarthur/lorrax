"""Gate: the non-TDA path refuses a restart that cannot make A Hermitian.

The non-TDA resonant block ``A = D + K^x - K^d`` is Hermitian **if and only if**
the restart's screened-Coulomb tile obeys ``W_MN(q) = conj(W_MN(-q))`` --
equivalently, iff the real-space ``W_R`` is real.  That is a property of the
file the GW stage wrote, not of this solver: the BSE stage does not compute W,
it reads it.  A restart written before the mini-BZ Coulomb head-slot fix
therefore carries a pre-fix operator frozen into an artifact, and the
definite-pencil solver's ``A is not Hermitian`` refusal is a TRUE POSITIVE that
names the wrong thing -- it fires after the O(N^2) dense build, and it names the
matrix rather than the file.

These cells pin the preflight that moves the refusal to the input and names the
GW re-run that repairs it, and -- the half that matters just as much -- pin that
it does NOT stand between a good restart and its solver: the last cell is a
complete non-TDA solve on a small window of the fresh fixture, through the same
entry point, and it must stay green.

The first two cells are pure host numpy/CPU jax (no restart, no GPU) so the
measure itself is covered in the plain suite.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
from jax.sharding import Mesh  # noqa: E402

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# The measure, on synthetic tiles (CPU).
# ---------------------------------------------------------------------------
def _tile_from_real_space(rng, nmu, grid):
    """A tile that is conjugate-reciprocal BY CONSTRUCTION: FFT of a REAL
    real-space kernel.  ``W(q) = sum_R W_R e^{-iqR}`` with ``W_R`` real gives
    ``W(-q) = conj(W(q))`` exactly, to floating point."""
    W_R = rng.standard_normal((nmu, nmu) + grid)
    return np.fft.fftn(W_R, axes=(2, 3, 4), norm="ortho")


def test_the_minus_q_gather_is_the_reciprocity_partner():
    """``w_q_reciprocity``'s flip+roll IS the ``(-i) % n`` gather on all three k
    axes, and the measure is ~0 on a reciprocal tile and O(1) on one that is not.
    Pure host numpy + CPU jax."""
    from bse.bse_nontda import w_q_reciprocity
    import jax.numpy as jnp

    grid = (3, 4, 2)
    rng = np.random.default_rng(11)

    # 1. flip+roll == the (-i) % n index gather, on an index-labelled tile.
    lab = np.arange(np.prod(grid)).reshape((1, 1) + grid).astype(np.complex128)
    ax = (2, 3, 4)
    got = np.asarray(jnp.roll(jnp.flip(jnp.asarray(lab), axis=ax),
                              shift=(1, 1, 1), axis=ax))
    ix = [(-np.arange(n)) % n for n in grid]
    want = lab[:, :, ix[0]][:, :, :, ix[1]][:, :, :, :, ix[2]]
    assert np.array_equal(got, want), "flip+roll is not the -q gather"

    # 2. reciprocal by construction -> machine zero.
    W_ok = _tile_from_real_space(rng, 5, grid)
    rel_ok = w_q_reciprocity(jnp.asarray(W_ok))
    assert rel_ok < 1e-13, f"reciprocal tile measured {rel_ok:.3e}"

    # 3. an imaginary real-space part is exactly what breaks it.
    W_bad = W_ok + 0.25 * np.fft.fftn(
        1j * rng.standard_normal((5, 5) + grid), axes=ax, norm="ortho")
    rel_bad = w_q_reciprocity(jnp.asarray(W_bad))
    assert rel_bad > 1e-2, f"broken tile measured only {rel_bad:.3e}"


def test_the_preflight_refuses_and_names_the_regeneration():
    """The refusal carries the producer, not just the number -- the
    ``load_dipole_h5`` pattern.  ``tol=None`` measures without refusing."""
    from bse.bse_nontda import check_restart_reciprocity, _NONTDA_RECIP_TOL
    import jax.numpy as jnp

    grid = (2, 2, 2)
    rng = np.random.default_rng(3)
    W_ok = jnp.asarray(_tile_from_real_space(rng, 4, grid))
    W_bad = W_ok + 0.1j * jnp.asarray(rng.standard_normal((4, 4) + grid))

    # Good tile: returns, does not raise, and is far inside the threshold.
    rel = check_restart_reciprocity(W_ok)
    assert rel < _NONTDA_RECIP_TOL

    with pytest.raises(ValueError) as ei:
        check_restart_reciprocity(W_bad, input_file="/deck/bse_si_test.in")
    msg = str(ei.value)
    assert "gw.gw_jax -i bse_si_test.in" in msg, (
        f"refusal does not name the regeneration command:\n{msg}")
    assert "conj(W_MN(-q))" in msg and "STALE ARTIFACT" in msg

    # The escape hatch measures and returns rather than refusing.
    assert check_restart_reciprocity(W_bad, tol=None) > 1e-3


# ---------------------------------------------------------------------------
# The real fixture: the positive control, the refusal, and an end-to-end solve.
# ---------------------------------------------------------------------------
def _small_nontda_data(gnppm_session, *, n_val=2, n_cond=2):
    """Load the fresh gnppm restart into the ``solve_bse_nontda_sharded`` data
    contract on a 1x1 mesh, at the 2v2c window (MoS2 3x3x1, nk=9 => N = 36) --
    the same window ``test_bse_dense_reference``'s non-TDA cells use, so the
    definite-pencil positive-definiteness precondition is one this deck is
    already known to satisfy."""
    from bse import bse_io
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=n_val, n_cond=n_cond, mesh_xy=mesh, input_file=input_path)
    return data, mesh


@pytest.mark.gpu
def test_the_fresh_restart_passes_the_preflight(gnppm_session):
    """POSITIVE CONTROL, and the evidence the threshold is set on: a restart the
    GW stage wrote on THIS tree is conjugate-reciprocal well inside the
    preflight's tolerance, so the refusal cannot fire on a current artifact."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_nontda import w_q_reciprocity, _NONTDA_RECIP_TOL
    data, _ = _small_nontda_data(gnppm_session)
    rel = w_q_reciprocity(data["W_q"])
    print(f"[preflight] fresh gnppm restart: max|W(q)-conj(W(-q))|/max|W| "
          f"= {rel:.6e} (tol {_NONTDA_RECIP_TOL:.1e})")
    assert rel < _NONTDA_RECIP_TOL, (
        f"a FRESH restart measured {rel:.3e} against tol "
        f"{_NONTDA_RECIP_TOL:.1e} -- either the GW stage regressed or the "
        f"threshold is wrong; do not raise it without deciding which")


@pytest.mark.gpu
def test_the_solver_refuses_a_broken_restart_before_the_dense_build(gnppm_session):
    """RED TWIN carrier: perturbing the loaded W tile so it breaks q<->-q makes
    ``solve_bse_nontda_sharded`` refuse AT THE INPUT, with the regeneration
    command -- not 258 s later with ``A is not Hermitian``.  Deleting the
    preflight call in ``solve_bse_nontda_sharded`` turns this cell RED."""
    harness.skip_unless_gpu(pytest)
    import jax.numpy as jnp
    from bse.bse_nontda import solve_bse_nontda_sharded
    data, mesh = _small_nontda_data(gnppm_session)
    scale = float(jnp.max(jnp.abs(data["W_q"])))
    # A purely imaginary constant added to every q is an odd-in-q defect: it
    # survives the (q -> -q, conjugate) map with the wrong sign.
    data["W_q"] = data["W_q"] + (1e-3 * scale) * 1j
    with pytest.raises(ValueError) as ei:
        solve_bse_nontda_sharded(data, mesh, n_eig=1, include_W=True)
    msg = str(ei.value)
    assert "gw.gw_jax" in msg, f"refusal does not name the producer:\n{msg}"
    assert "not conjugate-reciprocal in q" in msg


@pytest.mark.gpu
def test_nontda_runs_end_to_end_on_the_fresh_fixture(gnppm_session):
    """THE FALSE CASE this ships with: the same entry point, the same fresh
    restart, unperturbed -- a complete non-TDA solve at the smallest window.
    Green on both sides of the red twin, which is what makes the cell above a
    measurement of the preflight rather than of the solver."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_nontda import solve_bse_nontda_sharded
    data, mesh = _small_nontda_data(gnppm_session)
    nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
    N = int(data["n_cond_pad"]) * int(data["n_val_pad"]) * nk
    omega, evecs, _ = solve_bse_nontda_sharded(data, mesh, n_eig=2, include_W=True)
    omega = np.asarray(jax.device_get(omega))
    evecs = np.asarray(jax.device_get(evecs))
    print(f"[nontda e2e] N={N}, lowest omega (Ry) = {omega}")
    assert omega.shape == (2,) and np.all(np.isfinite(omega))
    assert np.all(omega > 0.0), f"non-TDA returned non-positive omega {omega}"
    assert np.all(np.diff(omega) >= -1e-12), "omega not ascending"
    # The (X, Y) pair convention is the solver's contract; check it holds.
    for i in range(2):
        X = evecs[i, 0].reshape(-1); Y = evecs[i, 1].reshape(-1)
        snorm = float(np.real(np.conj(X) @ X - np.conj(Y) @ Y))
        assert abs(snorm - 1.0) < 1e-6, f"state {i}: X^HX-Y^HY = {snorm:.6f}"
