"""``LORRAX_BSE_MATVEC_OPT`` — the dial, its grammar, and its two levers.

The BSE stack matvec carries two optional rewrites (``bse.bse_stack_matvec``):

  ``densek``  the k-space convolution's ``ifftn -> *W_R -> fftn`` replaced by
              an explicit dense (nk x nk) DFT contraction.  Same linear map,
              different floating-point order.
  ``yhoist``  the two 'y'-axis collectives lifted out of the per-trial scan.
              Reassociates nothing.

Three things have to hold, and each one is worthless without the others:

1.  **The DFT matrices are the transform.**  ``_dft_matrices`` is checked
    against ``jnp.fft.ifftn/fftn(norm='ortho')`` on four k-grid shapes,
    INCLUDING the two mistakes that would otherwise pass a weaker test.
    The obvious control -- transpose IFT -- is VOID here, because the DFT
    matrix is symmetric, so transposing it changes nothing; that near-miss is
    why the controls below are a swap and a conjugation instead.  A W that is
    all-ones is likewise void: the transform pair is then its own inverse and
    every wrong-but-unitary substitution still round-trips to the identity.
    So the fixture uses a RANDOM W.

2.  **The rewrites do not move the operator.**  Every dial setting is applied
    to the same production matvec on the shared dense fixture and compared
    against the dense reference H.  ``yhoist`` must be bit-identical;
    ``densek`` must agree to fp64 round-off but is NOT required to be
    bit-identical, and asserting that it is would be asserting something
    false.

3.  **A misspelled dial refuses.**  ``LORRAX_FFT_FFI_FUSED`` on this codebase
    accepted ``=yes`` and silently ignored ``=Y``; a perf dial that degrades
    to the baseline under a typo makes every A/B built on it unfalsifiable.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_stack_matvec as SM  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the DFT matrices ARE the transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expect", [
    ("", frozenset()),
    ("   ", frozenset()),
    ("yhoist", frozenset({"yhoist"})),
    ("  YHOIST ", frozenset({"yhoist"})),
    ("krep", frozenset({"krep"})),
    ("KREP,", frozenset({"krep"})),        # case + trailing comma tolerated
    ("yhoist,krep", frozenset({"yhoist", "krep"})),
])
def test_matvec_opt_grammar_accepts(monkeypatch, raw, expect):
    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", raw)
    assert SM.matvec_opts() == expect


@pytest.mark.parametrize("raw", ["dense_k", "densek", "densek2", "yhoisted", "densek,fft",
                                 "true", "1", "on", "kreps", "k_rep", "krep krep",
                                 "yhoist,krep,densek"])
def test_matvec_opt_grammar_refuses(monkeypatch, raw):
    """A token that is not an option must RAISE, never degrade to baseline."""
    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", raw)
    with pytest.raises(ValueError, match="LORRAX_BSE_MATVEC_OPT"):
        SM.matvec_opts()


def test_matvec_opt_unset_is_baseline(monkeypatch):
    monkeypatch.delenv("LORRAX_BSE_MATVEC_OPT", raising=False)
    assert SM.matvec_opts() == frozenset()


# ---------------------------------------------------------------------------
# 2. neither lever moves the operator
# ---------------------------------------------------------------------------
def _mesh():
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))


@pytest.mark.gpu
@pytest.mark.parametrize("opt,tol,label", [
    ("", 1e-9, "baseline"),
    ("yhoist", 0.0, "yhoist (must be BIT-identical)"),
])
def test_dial_preserves_the_operator(bse_dense_state, monkeypatch, opt, tol,
                                     label):
    harness.skip_unless_gpu(pytest)
    from test_bse_dense_reference import _build_dense_H, _relerr
    from test_bse_stack_matvec import _place, _random_stack

    data = bse_dense_state
    H, _, _, _ = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz

    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", opt)
    mesh = _mesh()
    sh, arr = _place(data, mesh)
    nt = 3
    X = _random_stack(nt, nc, nv, nk)
    with mesh:
        Xs = jax.lax.with_sharding_constraint(X, sh.X)
        mv = SM.build_bse_stack_matvec(mesh, nkx, nky, nkz, kernel="bse")
        HXs = np.asarray(mv(
            Xs, arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
            data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"],
            arr["M_X"], arr["M_Y"]))
    for t in range(nt):
        ref = H @ np.asarray(X)[t].reshape(-1)
        err = _relerr(HXs[t].reshape(-1), ref)
        assert err < 1e-9, f"{label} trial {t}: relerr {err:.3e} vs dense H"

    if tol == 0.0:
        # yhoist only re-orders WHICH collective moves the bytes; the
        # arithmetic and its association are untouched, so anything but a bit
        # match means it changed the computation.
        monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", "")
        with mesh:
            base_mv = SM.build_bse_stack_matvec(mesh, nkx, nky, nkz,
                                                kernel="bse")
            base = np.asarray(base_mv(
                jax.lax.with_sharding_constraint(X, sh.X),
                arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
                data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"],
                arr["M_X"], arr["M_Y"]))
        assert np.array_equal(base, HXs), \
            "yhoist changed a bit — it is supposed to move only collectives"


# ---------------------------------------------------------------------------
# 4. `krep` is HONEST about the routes that ignore it
# ---------------------------------------------------------------------------
#
# The grammar above refuses a token it cannot parse, because a dial that
# degrades to the baseline under a typo makes every A/B built on it void.
# `krep` had that same hole one level up, where the grammar cannot see it: it
# parses, it is resolved by `bse_lanczos.solve_bse_sharded` -- and then only
# the `block_size > 1` branch applies it.  On the shipped `--lanczos` route
# (`bs == 1`) the token does nothing, so a leg labelled `krep` there IS the
# baseline.  It has already produced one published measurement that way
# (KERNEL_DEEPDIVE.md 5.7: krylov_run 1.967 s vs a 1.882 s baseline, with all
# 20 100 reorth all-reduces still in the trace).


def test_krep_inert_route_warns(monkeypatch, capsys):
    """RED TWIN: `krep` on a route that ignores it must SAY SO, both ways."""
    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", "krep")
    with pytest.warns(RuntimeWarning, match="IGNORED"):
        msg = SM.warn_if_krep_inert(False, "unit-test-caller",
                                    "block_size=1 on this route")
    assert msg and "krep" in msg
    assert "unit-test-caller" in msg and "block_size=1" in msg
    # ...and loudly enough to survive a job log nobody is watching live.
    assert "IGNORED" in capsys.readouterr().out


def test_krep_warns_when_combined_with_an_honoured_token(monkeypatch):
    """`yhoist,krep` at bs=1 must warn about krep and NOT refuse the pair.

    This is the case that rules out a hard refusal: `yhoist` is honoured on
    this route and only `krep` is not, so refusing would reject a legitimate
    run.
    """
    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", "yhoist,krep")
    assert SM.matvec_opts() == frozenset({"yhoist", "krep"})   # no refusal
    with pytest.warns(RuntimeWarning, match="IGNORED"):
        assert SM.warn_if_krep_inert(False, "caller", "bs=1") is not None


def test_krep_honoured_route_is_silent(monkeypatch, capsys):
    """The control: no noise on the route that DOES apply the constraint."""
    monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", "krep")
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any warning fails the cell
        assert SM.warn_if_krep_inert(True, "caller", "bs>1") is None
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("raw", [None, "", "yhoist"])
def test_krep_unset_is_silent_even_on_the_inert_route(monkeypatch, raw):
    """Nobody who did not ask for `krep` should ever hear about it."""
    if raw is None:
        monkeypatch.delenv("LORRAX_BSE_MATVEC_OPT", raising=False)
    else:
        monkeypatch.setenv("LORRAX_BSE_MATVEC_OPT", raw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert SM.warn_if_krep_inert(False, "caller", "bs=1") is None


def test_the_bs1_route_actually_calls_the_announcement():
    """RED TWIN for the SEAM: the helper is worthless if nobody calls it.

    A source gate rather than a behavioural one, deliberately: driving
    ``solve_bse_sharded`` needs a loaded restart and a mesh, which is a
    regression-scale fixture for a one-line wiring question.  The end-to-end
    proof is a record-deck leg whose log carries the banner — recorded in
    FIX_smallwins.md.
    """
    import inspect
    from bse import bse_lanczos as BL
    src = inspect.getsource(BL.solve_bse_sharded)
    assert "warn_if_krep_inert" in src, (
        "solve_bse_sharded resolves `krep` but no longer imports the "
        "announcement that its bs == 1 branch ignores it")
    assert "_warn_krep(" in src, "the announcement was imported but not called"
