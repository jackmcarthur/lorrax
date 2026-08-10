"""The gate that was missing when the dipole producer stopped producing.

WHAT WENT WRONG (2026-08-09).  ``psp.get_dipole_mtxels`` — the default
producer, the path that wrote every committed ``dipole.h5`` — could not
write one at all on three of the four regression decks.  It died 30 s in,
six frames inside a jitted einsum::

    mtxel_sweep.py:758  vnl_ops.apply_vnl_velocity_to_ket(psi, kdata.Z, kdata.dZ, ...)
    TypeError: conjugate requires ndarray or scalar arguments, got <class 'NoneType'>

``kdata.dZ`` was ``None`` although the sweep asked for ``compute_dZ=True``,
because ``_build_vnl_kdata_core`` returned ``None`` whenever the setup had
no channels — and the setup had no channels because the deck directory had
no ``*.upf`` in it.  ``si_cohsex_debug``, ``gnppm_debug`` and
``hbn_cohsex_debug`` do not carry their pseudopotentials in the repo;
``cohsex_debug`` does, which is exactly the deck that kept working.

WHY NOTHING CAUGHT IT.  Two gaps, and this file closes both.

  1. No cell ever asked ``_build_vnl_kdata_core`` for ``dZ`` on a setup
     with no channels.  ``test_psp_padded_gvectors`` builds an
     empty-``channels`` setup, but only exercises ``Z``.
  2. No cell ever ran the producer end to end.  The regeneration path had
     no smoke test at any level, so it could stop working for four months
     without a red anywhere.

THE THREE ARMS, AND WHY ONLY ONE WAS LOUD.  With no pseudopotentials:
``--vnl-mode analytic`` (the default) crashes as above; ``--vnl-mode
numeric`` finite-differences an EMPTY projector set, so V_NL is
identically zero and it writes a p̂-only file stamped
``prov_skip_vnl=False``; ``--skip-vnl`` is correct and is the only arm
entitled to run without projectors.  Measured on si_cohsex_debug, the
numeric artifact agrees with the ``--skip-vnl`` artifact to 5.8e-15 — the
silent arm was the dangerous one.  Hence the refusal below is gated on
``--skip-vnl``, not on the mode.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from psp import vnl_ops

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_SRC = REPO_ROOT / "src" / "psp" / "get_dipole_mtxels.py"
REG_ROOT = REPO_ROOT / "tests" / "regression"


# ---------------------------------------------------------------------------
# Synthetic setups
# ---------------------------------------------------------------------------

def _empty_channel_setup(total_R: int = 3, n_q: int = 64):
    """The setup ``build_vnl_setup`` returns when no pseudopotential loaded.

    Deliberately the same shape as the fixture in
    ``test_psp_padded_gvectors`` — ``channels=[]`` with live ``row_*``
    metadata — because that is the one the tree already trusts for the
    ``Z`` half, and the whole point here is that the ``dZ`` half was never
    asked the same question.
    """
    return vnl_ops.VNLSetup(
        channels=[], dq=0.01, n_q=n_q, q_max=1.0,
        G_table=jnp.ones((1, n_q), dtype=jnp.float64),
        Gp_table=jnp.zeros((1, n_q), dtype=jnp.float64),
        prefactor=1.0, B=np.eye(3), cell_volume=1.0,
        total_R=total_R, nspinor=1,
        E_super=jnp.zeros((1, 1, total_R, total_R), dtype=jnp.complex128),
        l_max=0,
        row_beta_idx=jnp.zeros(total_R, dtype=jnp.int32),
        row_l=jnp.zeros(total_R, dtype=jnp.int32),
        row_m=jnp.zeros(total_R, dtype=jnp.int32),
        row_tau=jnp.zeros((total_R, 3), dtype=jnp.float64),
    )


def _one_channel_setup(n_q: int = 64, natoms: int = 2):
    """A LIVE setup: one ``l=1`` channel, one beta, two atoms.

    The control arm.  The empty-channel branch is a new code path and this
    cell is what says the old one still runs it.
    """
    l, nbeta = 1, 1
    msize = 2 * l + 1
    R = nbeta * msize
    q = np.linspace(0.0, 1.0, n_q)
    G = np.exp(-2.0 * q ** 2)
    Gp = -4.0 * q * G
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])[:natoms]
    E = np.zeros((2, 2, R, R), dtype=np.complex128)
    for s in range(2):
        E[s, s] = np.eye(R)
    ch = vnl_ops.ChannelMeta(l=l, nbeta=nbeta, msize=msize, R=R,
                             tau=tau, E=E, beta_table_start=0, natoms=natoms)
    total_R = R * natoms
    E_super = np.zeros((1, 1, total_R, total_R), dtype=np.complex128)
    for a in range(natoms):
        E_super[0, 0, a * R:(a + 1) * R, a * R:(a + 1) * R] = np.eye(R)
    return vnl_ops.VNLSetup(
        channels=[ch], dq=float(q[1] - q[0]), n_q=n_q, q_max=1.0,
        G_table=jnp.asarray(G[None, :], dtype=jnp.float64),
        Gp_table=jnp.asarray(Gp[None, :], dtype=jnp.float64),
        prefactor=1.0, B=np.eye(3) * 1.1, cell_volume=1.0,
        total_R=total_R, nspinor=1,
        E_super=jnp.asarray(E_super, dtype=jnp.complex128),
        l_max=l,
        row_beta_idx=jnp.zeros(total_R, dtype=jnp.int32),
        row_l=jnp.full(total_R, l, dtype=jnp.int32),
        row_m=jnp.asarray(np.tile(np.arange(msize), natoms), dtype=jnp.int32),
        row_tau=jnp.asarray(np.repeat(tau, msize, axis=0), dtype=jnp.float64),
    )


_KVEC = np.array([0.25, -0.125, 0.0])
_GK = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 1, 0], [0, 0, 2]],
               dtype=int)


# ---------------------------------------------------------------------------
# 1. The kernel contract: compute_dZ=True must hand back an ARRAY
# ---------------------------------------------------------------------------

def test_compute_dZ_true_never_returns_none_on_an_empty_projector_set():
    """FALSE ARM: before the fix this returns ``None`` and the assert fires."""
    kd = vnl_ops.build_vnl_kdata_from_kvec(_KVEC, _GK, _empty_channel_setup())
    assert kd.dZ is None, "compute_dZ defaults to False; dZ must stay None"

    kd = vnl_ops.build_vnl_kdata_from_kvec(_KVEC, _GK, _empty_channel_setup(),
                                            compute_dZ=True)
    assert kd.dZ is not None, (
        "compute_dZ=True returned dZ=None on a setup with no channels — "
        "this is the 2026-08-09 dipole producer defect")
    assert np.asarray(kd.dZ).shape == (3, kd.total_R, _GK.shape[0])
    assert np.all(np.asarray(kd.dZ) == 0.0), (
        "an empty projector set has zero velocity, not a nonzero one")


def test_compute_dZ_true_never_returns_none_under_a_trace():
    """The sweep reaches the core THROUGH ``lax.scan``; check that door too.

    The failing driver used the traced entry point, so this cell asks the
    same question the sweep asks.  It is NOT a traced-only defect — the
    eager cell above fails identically before the fix — and this pair is
    what says so.
    """
    setup = _empty_channel_setup()

    @jax.jit
    def _dz(kvec, gk):
        kd = vnl_ops.build_vnl_kdata_traced(kvec, gk, setup, compute_dZ=True)
        assert kd.dZ is not None, (
            "traced compute_dZ=True returned dZ=None on an empty channel set")
        return kd.dZ

    out = _dz(jnp.asarray(_KVEC), jnp.asarray(_GK, dtype=jnp.int32))
    assert np.asarray(out).shape == (3, setup.total_R, _GK.shape[0])


def test_apply_vnl_velocity_to_ket_survives_an_empty_projector_set():
    """The exact cluster failure, at unit scale.

    FALSE ARM: before the fix this raises
    ``TypeError: conjugate requires ndarray or scalar arguments, got
    <class 'NoneType'>`` — the traceback the eight regeneration legs died
    with on 2026-08-09.
    """
    setup = _empty_channel_setup()
    kd = vnl_ops.build_vnl_kdata_from_kvec(_KVEC, _GK, setup, compute_dZ=True)
    nb, ns, nG = 4, 1, _GK.shape[0]
    psi = jnp.asarray(np.random.default_rng(0).standard_normal((nb, ns, nG))
                      + 0j)
    v = vnl_ops.apply_vnl_velocity_to_ket(psi, kd.Z, kd.dZ, kd.E_super)
    assert np.asarray(v).shape == (3, nb, ns, nG)
    assert np.all(np.asarray(v) == 0.0)


def test_a_live_projector_set_still_builds_a_nonzero_dZ():
    """CONTROL: the fix adds a branch; this says the old branch still runs."""
    setup = _one_channel_setup()
    kd = vnl_ops.build_vnl_kdata_from_kvec(_KVEC, _GK, setup, compute_dZ=True)
    dZ = np.asarray(kd.dZ)
    assert dZ.shape == (3, setup.total_R, _GK.shape[0])
    assert np.all(np.isfinite(dZ))
    assert np.max(np.abs(dZ)) > 0.0, "live channels produced a null dZ"


# ---------------------------------------------------------------------------
# 2. The driver wiring, checked without importing the driver
# ---------------------------------------------------------------------------
# ``psp.get_dipole_mtxels`` calls ``initialize_communicator_stack()`` at
# module scope, which REQUIRES the FFI host library, so it cannot be
# imported on a box without one.  These two cells read the source instead:
# they are canaries for the wiring, not substitutes for the end-to-end
# cells below, which do run the real thing wherever the .so exists.

def _driver_ast():
    return ast.parse(DRIVER_SRC.read_text())


def test_the_driver_calls_the_shared_pseudopotential_preflight():
    """FALSE ARM: before the fix ``validate_operator_inputs`` is absent here.

    ``psp.operator_checks``'s own docstring names this caller — "before
    computing kin+ion, DIPOLE matrix elements, ..." — and ``gw.kin_ion_io``
    and ``psp.get_DFT_mtxels`` both call it.  The dipole driver was the one
    that never did.
    """
    called = {
        n.func.id for n in ast.walk(_driver_ast())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "validate_operator_inputs" in called, (
        "get_dipole_mtxels does not run the shared pseudopotential "
        "pre-flight; a deck with no *.upf will reach the sweep and die "
        "inside a jitted einsum instead of being refused up front")


def test_the_driver_can_be_pointed_at_a_pseudopotential_directory():
    """Three of four regression decks keep their UPFs outside the tree."""
    src = DRIVER_SRC.read_text()
    assert "--pseudo-dir" in src, (
        "no --pseudo-dir flag: a fixture re-cut from a clean checkout "
        "cannot find the deck's pseudopotentials (gw.kin_ion_io has had "
        "--pseudo_dir all along)")


# ---------------------------------------------------------------------------
# 3. End to end — the regeneration smoke gate that did not exist
# ---------------------------------------------------------------------------

def _ffi_available():
    """The driver imports the runtime, which REQUIRES the host FFI."""
    try:
        from ffi.common import ffi_loader
        return ffi_loader.probe_target("lorrax_mklfft_flat_k", "cpu")
    except Exception as exc:                       # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _run_driver(run_dir: Path, *args, timeout=1800):
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-u", "-m", "psp.get_dipole_mtxels", *args],
        cwd=run_dir, env=env, capture_output=True, text=True,
        timeout=timeout, check=False)


def _stage(tmp_path: Path, deck: str) -> Path:
    run_dir = tmp_path / deck
    shutil.copytree(REG_ROOT / deck, run_dir,
                    ignore=shutil.ignore_patterns("tmp", "*.dat", "*.log"))
    for p in run_dir.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o200)
    return run_dir


@pytest.mark.regression
def test_the_default_analytic_sweep_writes_a_valid_dipole_h5(tmp_path):
    """THE SMOKE GATE.  Default arm, real deck, end to end, artifact checked.

    ``cohsex_debug`` is the only regression deck that carries its own
    ``*.upf``, which is what makes this cell runnable from a clean
    checkout with no staging.  It is the deck the 2026-08-09 defect spared,
    so on its own it would not have caught that failure — the refusal cell
    below is the arm that would have.  This one closes the other half of
    the gap: that nothing at all ran the producer.
    """
    ok, reason = _ffi_available()
    if not ok:
        pytest.skip(f"driver needs the host FFI library: {reason}")

    run = _stage(tmp_path, "cohsex_debug")
    full = _run_driver(run, "-i", "cohsex_test.in", "--out", "dipole_regen.h5")
    assert full.returncode == 0, (
        f"default analytic sweep failed rc={full.returncode}\n"
        f"{full.stdout[-4000:]}\n{full.stderr[-4000:]}")

    import h5py
    out = run / "dipole_regen.h5"
    assert out.is_file() and out.stat().st_size > 0
    with h5py.File(out) as h:
        assert set(h.keys()) >= {"dipole_cart", "deltaE"}
        d = np.asarray(h["dipole_cart"])
        assert h.attrs["prov_vnl_mode"] == "analytic"
        assert not bool(h.attrs["prov_skip_vnl"])
    assert np.all(np.isfinite(d)), "dipole_cart carries non-finite entries"
    assert np.max(np.abs(d)) > 0.0

    # AND V_NL ACTUALLY ENTERED IT.  Without this arm the gate would pass
    # on the silent failure: a file whose V_NL is identically zero while
    # its provenance says otherwise is exactly what --vnl-mode numeric
    # produced on a pseudo-less deck.
    bare = _run_driver(run, "-i", "cohsex_test.in", "--skip-vnl",
                       "--out", "dipole_bare.h5")
    assert bare.returncode == 0, bare.stderr[-4000:]
    with h5py.File(run / "dipole_bare.h5") as h:
        p_only = np.asarray(h["dipole_cart"])
    delta = float(np.max(np.abs(d - p_only)))
    scale = float(np.max(np.abs(d)))
    assert delta > 1e-8 * scale, (
        f"the analytic arm is indistinguishable from --skip-vnl "
        f"(max|Δ| = {delta:.3e}, scale {scale:.3e}): V_NL contributed "
        f"NOTHING, which is what an empty projector set looks like")


@pytest.mark.regression
def test_the_driver_refuses_a_deck_with_no_pseudopotentials(tmp_path):
    """THE ARM THAT WOULD HAVE CAUGHT IT.

    FALSE ARM: before the fix this run reaches the sweep and dies with
    ``conjugate requires ndarray or scalar arguments, got <class
    'NoneType'>`` — asserted against explicitly below, so a regression
    that restores the old behaviour fails on the message, not just on the
    return code.
    """
    ok, reason = _ffi_available()
    if not ok:
        pytest.skip(f"driver needs the host FFI library: {reason}")

    run = _stage(tmp_path, "cohsex_debug")
    removed = list(run.glob("*.upf"))
    assert removed, "fixture no longer carries the UPFs this cell removes"
    for p in removed:
        p.unlink()

    res = _run_driver(run, "-i", "cohsex_test.in", "--out", "dipole_nopp.h5")
    blob = res.stdout + res.stderr
    assert res.returncode != 0, (
        "the driver produced a dipole.h5 with NO pseudopotentials loaded")
    assert "conjugate requires ndarray" not in blob, (
        "still failing deep inside the jitted sweep instead of refusing "
        "up front:\n" + blob[-4000:])
    assert "No pseudopotentials loaded" in blob, (
        "refusal did not name the missing pseudopotentials:\n"
        + blob[-4000:])
    assert not (run / "dipole_nopp.h5").exists()

    # --skip-vnl is the one arm entitled to run without them, and it must
    # still be allowed through: the refusal is about a LIE in the
    # provenance block, not about pseudopotentials as such.
    bare = _run_driver(run, "-i", "cohsex_test.in", "--skip-vnl",
                       "--out", "dipole_bare.h5")
    assert bare.returncode == 0, (
        "--skip-vnl needs no projectors and must not be refused:\n"
        + (bare.stdout + bare.stderr)[-4000:])
