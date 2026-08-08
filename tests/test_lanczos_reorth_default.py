"""Gates for the SHIPPED DEFAULTS of the Lanczos reorthogonalisation window.

The module comment in ``solvers/lanczos.py`` promised this file; until
2026-08-08 it did not exist, and that is exactly how the defaults went
unexercised.  Every cell here calls the solvers **with no ``n_reorth=``
argument** (except where a window is the thing under test), because the failure
these gates exist to catch is "production passes the sentinel and every test
pins full reorth explicitly, so nothing ever runs what a new caller gets".

Cells and the failure each one catches:

* ``test_the_shipped_defaults_are_full_reorth`` -- the default silently
  reverting to a finite window.
* ``test_resolve_n_reorth_maps_the_sentinel_and_none`` -- the resolver itself.
* ``test_sentinel_is_resolved_before_both_consumers`` -- THE RED TWIN for the
  semantic trap: ``_reorth_window`` and ``_announce_reorth`` both read
  ``n_reorth`` RAW, so if the resolver stops running first the mask goes EMPTY
  and the announced collective count goes to ZERO, both silently.  The cell
  drives -1 through BOTH routes and checks the announced count and the mask
  width against the resolved value, then proves the two consumers really are
  unsafe against the raw sentinel.
* ``test_the_default_beats_a_short_window`` -- the measurement behind the flip.
* ``test_krylov_clamp_caps_at_the_vector_space`` -- running past Krylov
  exhaustion, which manufactures Ritz values below the true spectrum.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from solvers import lanczos as LZ


def _hermitian(n, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    lam = np.linspace(1.0, 12.0, n)
    H = (Q * lam) @ Q.conj().T
    return jnp.asarray(0.5 * (H + H.conj().T), dtype=jnp.complex128), lam


def _degenerate(n, seed=11):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    lam = np.linspace(1.0, 12.0, n)
    lam[:4] = 1.0
    lam[4:8] = 1.05
    lam = np.sort(lam)
    H = (Q * lam) @ Q.conj().T
    return jnp.asarray(0.5 * (H + H.conj().T), dtype=jnp.complex128), lam


# --------------------------------------------------------------------------
# the defaults themselves
# --------------------------------------------------------------------------

def test_the_shipped_defaults_are_full_reorth():
    """Every solver's DEFAULT window is the sentinel, not a finite width.

    A finite default is a footgun with a specific shape: it gets monotonically
    WORSE with more iterations, so a caller who asks for more work and gets a
    worse answer has no way to notice from the outside.
    """
    import inspect
    for fn in (LZ.lanczos_eig_jit, LZ.block_lanczos_eig_jit,
               LZ.block_lanczos_eig_jit_converged):
        default = inspect.signature(fn).parameters["n_reorth"].default
        assert default == LZ.FULL_REORTH, (
            f"{fn.__name__} defaults to n_reorth={default!r}, not FULL_REORTH")

    from bse import bse_lanczos as BL
    for fn in (BL.solve_bse, BL.solve_bse_sharded):
        default = inspect.signature(fn).parameters["n_reorth"].default
        assert default == LZ.FULL_REORTH, (
            f"{fn.__name__} defaults to n_reorth={default!r}, not FULL_REORTH")


def test_resolve_n_reorth_maps_the_sentinel_and_none():
    assert LZ.FULL_REORTH == -1
    for depth in (1, 7, 200):
        assert LZ.resolve_n_reorth(LZ.FULL_REORTH, depth) == depth
        assert LZ.resolve_n_reorth(None, depth) == depth
        assert LZ.resolve_n_reorth(-5, depth) == depth      # any negative
        assert LZ.resolve_n_reorth(0, depth) == 0           # 0 is a real width
        assert LZ.resolve_n_reorth(3, depth) == 3
        # idempotent, which is what lets bse_jax keep its own pre-resolution
        assert LZ.resolve_n_reorth(LZ.resolve_n_reorth(-1, depth), depth) == depth


# --------------------------------------------------------------------------
# THE RED TWIN for the semantic trap
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["cgs2", "mgs"])
def test_sentinel_is_resolved_before_both_consumers(capsys, kind):
    """-1 must never reach ``_reorth_window`` or ``_announce_reorth``.

    Both read ``n_reorth`` as a WIDTH.  Fed the raw sentinel, the first builds
    ``idx >= j + 1`` -- empty once intersected with ``idx <= j``, i.e. full
    reorth silently becomes NO reorth -- and the second announces ZERO
    collectives, so the log a reader would use to catch the first lies too.
    """
    n, max_iter = 48, 12
    H, _ = _hermitian(n)
    mv = lambda v: H @ v

    capsys.readouterr()
    jax.block_until_ready(LZ.lanczos_eig_jit(
        mv, n, n_eig=4, max_iter=max_iter, n_reorth=LZ.FULL_REORTH,
        reorth=kind, seed=3))
    line = capsys.readouterr().out
    assert f"n_reorth={max_iter}" in line, (
        f"the sentinel reached the announce unresolved: {line!r}")
    expect = LZ.reorth_collective_count(kind, max_iter, max_iter)
    assert f"-> {expect} reorth all-reduces" in line, (
        f"announced count is not the full-reorth count ({expect}): {line!r}")

    # the mask the batched route applies is built from the RESOLVED width
    resolved = LZ.resolve_n_reorth(LZ.FULL_REORTH, max_iter)
    assert resolved == max_iter
    for j in (0, 3, 7, 11):
        sel = np.asarray(LZ._reorth_window(j, max_iter + 1, resolved))
        assert int(sel.sum()) == min(j, resolved) + 1, (j, sel)
        assert sel[j], j

    # RED TWIN: the two consumers really are unsafe against the raw sentinel,
    # so removing the resolve call is caught by the assertions above rather
    # than sailing through as "well, -1 probably means something sensible".
    raw = np.asarray(LZ._reorth_window(5, 24, LZ.FULL_REORTH))
    assert not raw.any(), (
        "RED TWIN DID NOT GO RED: the raw sentinel no longer empties the "
        "window mask, so this cell has stopped proving the resolver matters")
    assert LZ.mgs_trip_count(max_iter, LZ.FULL_REORTH) == 0, (
        "RED TWIN DID NOT GO RED: the raw sentinel no longer zeroes the "
        "announced collective count")


def test_the_default_and_an_explicit_full_window_agree(capsys):
    """Passing nothing must be identical to passing the full width."""
    n, max_iter = 64, 24
    H, _ = _hermitian(n)
    mv = lambda v: H @ v
    ev_default, _ = LZ.lanczos_eig_jit(mv, n, n_eig=5, max_iter=max_iter, seed=3)
    ev_explicit, _ = LZ.lanczos_eig_jit(
        mv, n, n_eig=5, max_iter=max_iter, n_reorth=max_iter, seed=3)
    assert np.array_equal(np.asarray(ev_default), np.asarray(ev_explicit))


# --------------------------------------------------------------------------
# the measurement behind the flip
# --------------------------------------------------------------------------

def test_the_default_beats_a_short_window():
    """RED TWIN for the flip: a short window must be VISIBLY worse.

    If this cell ever goes green because the window got good rather than
    because the default got full, the flip has stopped being justified and
    somebody should say so out loud.

    Deliberately a NON-degenerate spectrum.  On a degenerate one, a
    single-vector Lanczos run deep into the space legitimately re-discovers a
    degenerate root as a second copy even under full reorth — the Krylov space
    meets each eigenspace once only in exact arithmetic — and that is a
    different phenomenon from the one the flip is about.  Measuring the flip on
    a degenerate synthetic would blame the window for something the window does
    not cause.
    """
    n, max_iter = 128, 96
    H, lam = _hermitian(n)
    mv = lambda v: H @ v
    ev_default, _ = LZ.lanczos_eig_jit(mv, n, n_eig=6, max_iter=max_iter, seed=3)
    ev_window, _ = LZ.lanczos_eig_jit(
        mv, n, n_eig=6, max_iter=max_iter, n_reorth=2, seed=3)

    err_d = float(np.max(np.abs(np.asarray(ev_default)[:3] - lam[:3])))
    err_w = float(np.max(np.abs(np.asarray(ev_window)[:3] - lam[:3])))
    assert err_d < 1e-10, f"the DEFAULT is not converging: {err_d:.3e}"
    assert err_w > 1e3 * max(err_d, 1e-14), (
        f"RED TWIN DID NOT GO RED: n_reorth=2 error {err_w:.3e} vs default "
        f"{err_d:.3e} -- the short window is no longer worse, so the flip's "
        f"justification needs re-measuring")


# --------------------------------------------------------------------------
# the Krylov-exhaustion clamp
# --------------------------------------------------------------------------

def test_krylov_clamp_caps_at_the_vector_space(capsys):
    """max_iter cannot exceed the space; past exhaustion Ritz values go wild.

    The announce line carries the effective ``max_iter``, which makes the clamp
    directly observable rather than something inferred from a shape.
    """
    n = 8
    H, lam = _hermitian(n)
    mv = lambda v: H @ v
    capsys.readouterr()
    ev, _ = jax.block_until_ready(
        LZ.lanczos_eig_jit(mv, n, n_eig=3, max_iter=64, seed=3))
    line = capsys.readouterr().out
    assert f"max_iter={n}" in line, (
        f"64 iterations were not clamped to n={n}: {line!r}")
    assert "max_iter=64" not in line
    ev = np.asarray(ev)
    assert np.all(np.isfinite(ev))
    # and nothing below the true spectrum -- the ghost the clamp exists to kill
    assert ev.min() > lam.min() - 1e-6, (ev.min(), lam.min())


def test_block_krylov_clamp_caps_at_floor_n_over_bs(capsys):
    n, bs = 24, 4
    H, lam = _hermitian(n)
    mvb = lambda Vb: (H @ Vb.T).T
    capsys.readouterr()
    ev, _ = jax.block_until_ready(LZ.block_lanczos_eig_jit(
        mvb, n, n_eig=3, block_size=bs, max_iter=99, seed=5))
    line = capsys.readouterr().out
    assert f"max_iter={n // bs}" in line, line
    ev = np.asarray(ev)
    assert np.all(np.isfinite(ev))
    assert ev.min() > lam.min() - 1e-6, (ev.min(), lam.min())
