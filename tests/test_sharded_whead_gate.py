"""A screened head must not land on an unscreened tile — at any mesh.

THE DEFECT (found by the Schur design pass, ``SCHUR_BSE_DESIGN.md`` §5(a),
fixed on ``fix/sharded-whead-gate-2026-08-10``).  Both BSE restart loaders
fall back to bare Coulomb ``V`` for ``W`` when the restart carries no ready
``W0_qmunu`` — the April all-zero-screening gate, ``W0_ready``, which
``tests/test_bse_w0_ready_gate.py`` pins on the READING side.  Both then
re-attach the q=0 Coulomb head that ``compute_vcoul`` removed before the
Dyson solve, as a rank-1 update ``(head/V_cell)·conj(g0)⊗g0``.

``vhead`` belongs on the exchange tile either way; ``whead`` is the head of
the SCREENED interaction, so it belongs on the W tile only when a screened W
was actually loaded.  ``_load_ring_subset`` (single device) asked that
question.  ``load_bse_data_from_restart_sharded`` — the P>1 path, and the one
production runs at scale — did not: it passed ``W_q`` into the injector
unconditionally, so a fallback run added a 2600-meV-class screened head to a
tile that is bare ``V``.  The fallback warns loudly; the head on top of it
was silent, and it is not a smaller error than the fallback it rides on but a
different one, on the q=0 tile the exciton binding is most sensitive to.

WHAT IS MEASURED HERE, and why it takes subprocesses.  The claim is about
composition of the returned tensor, so every cell below runs the REAL loader
on a REAL (synthetic) restart file and compares tiles — no source matching
decides whether the code is correct.  A mesh is fixed at process start
(``--xla_force_host_platform_device_count``), so each device count is one
CPU worker subprocess, JSON on stdout; this module is both the pytest file
and that worker, the convention ``tests/test_zeta_mesh_invariance.py``
established.  CPU-only and fixture-free: it runs on any box.

The four arms, per mesh:

``sharded/fallback``  W0 present but ``W0_ready=False`` → the bug's state.
``sharded/ready``     a real W0 → the normal path, which must not move.
``ring/*``            the same two through the single-device loader, whose
                      gate was already right, as the parity target.
``prefix/fallback``   THE RED TWIN: the same loader with the gate forced
                      open (the pre-fix spelling monkeypatched back in),
                      which must show the injected head at P>1.  A gate
                      whose red twin is not executed is a gate that can rot
                      into a tautology.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import pytest

# NO ``import jax`` AT MODULE SCOPE.  The device count is an XLA flag read at
# backend init, so the meshes below can only be built in a fresh process; a
# jax imported here would be the pytest session's backend, not the worker's.

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BSE_IO = _REPO_ROOT / "src" / "bse" / "bse_io.py"

# ---------------------------------------------------------------------------
# The fixture.  Small enough to be instant, big enough that the μ axis does
# NOT divide the 2x2 mesh: n_μ = 6 pads to 8 at P=4 and stays 6 at P=1, so
# every cross-mesh comparison below is over the LOGICAL block and the pad
# rows are exercised rather than assumed away.
# ---------------------------------------------------------------------------
N_MU = 6
NKX, NKY, NKZ = 2, 1, 1
NK, NB, NSPINOR = 2, 4, 1
N_OCC, N_VAL, N_COND = 2, 1, 1
CELL_VOLUME = 270.0            # Bohr³, Si-ish; only its reciprocal is used
VHEAD = 3.5                    # Ry
WHEAD = 1.25                   # Ry — deliberately NOT equal to vhead, so a
                               # tile carrying the wrong one is visible
_SEED = 20260810


def _write_restart(path, *, w0_ready: bool):
    """A canonical restart in the two states the gate distinguishes.

    ``W0_qmunu`` is PRESENT and correctly shaped in both; only the
    ``W0_ready`` attr differs — which is the whole point of that flag (see
    ``tests/test_bse_w0_ready_gate.py``: ``gw_init`` allocates a full-size
    zero W0 unconditionally, so presence is not persistence).  The V/W
    tensors are the 8-D legacy layout so both loaders resolve the k-grid
    from the shape and neither needs a WFN.
    """
    rng = np.random.default_rng(_SEED)

    def _cplx(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    V = _cplx((1, 1, 1, NKX, NKY, NKZ, N_MU, N_MU))
    W0 = _cplx((1, 1, 1, NKX, NKY, NKZ, N_MU, N_MU))
    psi = _cplx((NK, NB, NSPINOR, N_MU))
    # Well separated: a strict --band-degeneracy resolve must not snap the
    # window, or the two loaders would be reading different bands.
    enk = np.array([[-2.0, -1.0, 1.0, 2.0], [-2.2, -1.2, 1.2, 2.2]])
    g0 = _cplx((N_MU,))

    import h5py
    with h5py.File(path, "w") as f:
        dv = f.create_dataset("V_qmunu", data=V)
        dv.attrs["V_ready"] = True
        dw = f.create_dataset("W0_qmunu", data=W0)
        dw.attrs["W0_ready"] = bool(w0_ready)
        f.create_dataset("psi_full_y", data=psi)
        f.create_dataset("enk_full", data=enk)
        f.create_dataset("G0_mu_nu", data=g0)
        f.create_dataset("vhead", data=VHEAD)
        f.create_dataset("whead", data=np.array([WHEAD], dtype=np.complex128))
        f.create_dataset("kgrid", data=np.array([NKX, NKY, NKZ]))
    return V[0, 0, 0], W0[0, 0, 0], g0


def _sha(a: np.ndarray) -> str:
    """Bit identity, transportable through JSON."""
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.complex128).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _worker(px: int, py: int) -> dict:
    """Run every arm this device count supports; return the measurements."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from bse import bse_io

    mesh = Mesh(np.asarray(jax.devices()[:px * py]).reshape(px, py),
                axis_names=("x", "y"))
    out: dict[str, dict] = {}
    tmpdir = tempfile.mkdtemp(prefix="whead_gate_")

    for arm in ("fallback", "ready"):
        path = os.path.join(tmpdir, f"restart_{arm}.h5")
        V_q, W0_q, g0 = _write_restart(path, w0_ready=(arm == "ready"))
        # What the loader is supposed to have READ on this arm: the screened
        # W0 when it is ready, the bare exchange tensor when it is not.
        src = W0_q if arm == "ready" else V_q
        out[f"disk/{arm}"] = {
            "src_q0": _sha(src[0, 0, 0]),
            "src_q1": _sha(src[1, 0, 0]),
            "V_q0": _sha(V_q[0, 0, 0]),
            # The same tile PLUS the rank-1 screened head, in the loader's
            # own arithmetic — jnp, complex128, the same scalar and outer
            # product as ``head_correction._head_rank1_scalars`` — so
            # "differs by exactly the head" is a bit claim, not a tolerance.
            "src_q0_plus_head": _sha(np.asarray(
                jnp.asarray(src[0, 0, 0])
                + jnp.asarray(WHEAD / CELL_VOLUME, dtype=jnp.complex128)
                * (jnp.conj(jnp.asarray(g0))[:, None] * jnp.asarray(g0)[None, :]))),
        }

        def _record(tag, data):
            W_q = np.asarray(jax.device_get(data["W_q"]))
            V_q0 = np.asarray(jax.device_get(data["V_q0"]))
            out[tag] = {
                "q0_tile": _sha(W_q[:N_MU, :N_MU, 0, 0, 0]),
                "finite_q_tile": _sha(W_q[:N_MU, :N_MU, 1, 0, 0]),
                "V_q0": _sha(V_q0[:N_MU, :N_MU]),
                "padded_extent": int(W_q.shape[0]),
            }

        _record(f"sharded/{arm}", bse_io.load_bse_data_from_restart_sharded(
            path, n_val=N_VAL, n_cond=N_COND, mesh_xy=mesh, n_occ=N_OCC,
            cell_volume=CELL_VOLUME))

        if px * py == 1:
            # The single-device full-file loader, which refuses at P>1 by
            # construction.  It resolves ``cell_volume`` from the WFN named
            # by a deck, and this fixture has neither; supply THAT ONE
            # NUMBER and leave the rest of the resolution real, so the two
            # loaders are compared on identical head inputs.
            _real = bse_io._resolve_head_params

            def _with_cell_volume(input_file, vhead_restart, whead_restart,
                                  cell_volume=None):
                return _real(input_file, vhead_restart, whead_restart,
                             CELL_VOLUME if cell_volume is None else cell_volume)

            bse_io._resolve_head_params = _with_cell_volume
            try:
                _record(f"ring/{arm}", bse_io._load_ring_subset(
                    path, n_val=N_VAL, n_cond=N_COND, px=1, py=1, n_occ=N_OCC))
            finally:
                bse_io._resolve_head_params = _real

        if arm == "fallback":
            # THE RED TWIN, EXECUTED.  ``_inject_q0_head`` with its gate
            # forced open is the pre-fix line verbatim (``W_q`` passed
            # straight in, no ``w0_ready``), so this arm is the shipped
            # loader running the shipped defect on the shipped fallback
            # state.  If it stops showing the injected head, the fixture
            # has stopped constructing the hazard and every green cell
            # below is vacuous.
            _real_inject = bse_io._inject_q0_head

            def _ungated(*args, **kwargs):
                kwargs["w0_ready"] = True
                return _real_inject(*args, **kwargs)

            bse_io._inject_q0_head = _ungated
            try:
                _record("prefix/fallback",
                        bse_io.load_bse_data_from_restart_sharded(
                            path, n_val=N_VAL, n_cond=N_COND, mesh_xy=mesh,
                            n_occ=N_OCC, cell_volume=CELL_VOLUME))
            finally:
                bse_io._inject_q0_head = _real_inject

    return out


def _run(px: int, py: int) -> dict:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={px * py}").strip()
    res = subprocess.run([sys.executable, os.path.abspath(__file__),
                          f"{px}x{py}"],
                         env=env, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, (
        f"worker {px}x{py} failed rc={res.returncode}\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"no JSON from worker\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(lines[-1])


@pytest.fixture(scope="module")
def p1():
    """P=1: the sharded loader, the single-device loader, and the red twin."""
    return _run(1, 1)


@pytest.fixture(scope="module")
def p4():
    """P=4 (2x2): the regime the defect was found in, and pads μ 6 → 8."""
    return _run(2, 2)


# ---------------------------------------------------------------------------
# (a) The fallback state
# ---------------------------------------------------------------------------

def test_the_pre_fix_loader_puts_a_screened_head_on_the_bare_v_tile(p4):
    """THE DEFECT, REPRODUCED, at P>1 and through the real loader.

    Gate forced open, fallback file: the returned q=0 tile is NOT the bare
    ``V`` that was read from disk, and it is EXACTLY that ``V`` plus the
    rank-1 ``(whead/V_cell)·conj(g0)⊗g0``.  Both halves matter — the first
    says something was added, the second says it was the screened head and
    not a read error.
    """
    got = p4["prefix/fallback"]["q0_tile"]
    disk = p4["disk/fallback"]
    assert got != disk["src_q0"], (
        "the pre-fix spelling left the fallback tile alone; the fixture no "
        "longer constructs the hazard this file exists to gate")
    assert got == disk["src_q0_plus_head"], (
        "the pre-fix tile is neither bare V nor V+head — the arithmetic in "
        "this cell's reference has drifted from head_correction's")


@pytest.mark.parametrize("mesh", ["p1", "p4"])
def test_the_fallback_tile_is_returned_exactly_as_it_was_read(mesh, request):
    """THE FIX.  No screened head on an unscreened tile, at either mesh.

    BIT equality against the dataset on disk, not a tolerance: the gated
    path performs no arithmetic on ``W_q`` at all, so anything short of the
    identical bytes is a different bug.
    """
    res = request.getfixturevalue(mesh)
    assert res["sharded/fallback"]["q0_tile"] == res["disk/fallback"]["src_q0"], (
        "the sharded loader's bare-V fallback tile carries an injected "
        "screened head (SCHUR_BSE_DESIGN.md §5(a)); whead must be gated on "
        "w0_ready exactly as _load_ring_subset gates it")


def test_the_two_loaders_compose_the_fallback_identically(p1, p4):
    """GATE PARITY, stated as bit equality between the two readers.

    The single-device loader's gate was already right, so it is the target:
    sharded at 1x1 and at 2x2 must both produce the tile ``_load_ring_subset``
    produces, on the same file, for the same head.  The exchange tile is
    compared too — ``vhead`` is NOT gated (the q=0 exchange tile is bare
    Coulomb whether or not screening loaded), and a fix that over-reached
    into V would show up here rather than in a review.
    """
    ring = p1["ring/fallback"]
    for tag, res in (("1x1", p1), ("2x2", p4)):
        assert res["sharded/fallback"]["q0_tile"] == ring["q0_tile"], (
            f"sharded {tag} and single-device disagree on the fallback W tile")
        assert res["sharded/fallback"]["V_q0"] == ring["V_q0"], (
            f"sharded {tag} and single-device disagree on the fallback "
            f"exchange tile; vhead must still be injected")


# ---------------------------------------------------------------------------
# (b) The normal path does not move
# ---------------------------------------------------------------------------

def test_the_ready_path_is_untouched_at_every_mesh(p1, p4):
    """W0 loaded → the head is injected, identically everywhere.

    Three statements in one: the ready tile is the on-disk W0 PLUS the
    screened head (so the gate did not over-fire and skip a real
    injection); the sharded loader agrees bit-for-bit with the
    single-device one; and 1x1 agrees with 2x2, i.e. the μ padding that
    appears at P=4 changes nothing in the logical block.
    """
    for tag, res in (("1x1", p1), ("2x2", p4)):
        assert res["sharded/ready"]["q0_tile"] == \
            res["disk/ready"]["src_q0_plus_head"], (
            f"sharded {tag} ready tile is not W0 + the rank-1 head; the "
            f"gate has over-fired and skipped a legitimate injection")
    assert p1["sharded/ready"]["q0_tile"] == p1["ring/ready"]["q0_tile"]
    assert p4["sharded/ready"]["q0_tile"] == p1["ring/ready"]["q0_tile"]
    assert p1["sharded/ready"]["V_q0"] == p1["ring/ready"]["V_q0"]


@pytest.mark.parametrize("mesh", ["p1", "p4"])
def test_only_the_q0_slice_is_ever_touched(mesh, request):
    """The head is a q=0 object; a finite-q tile must be the disk bytes.

    Cheap, and it is the property that makes the cells above local: if the
    injection ever leaked along the q axis, "the q=0 tile is right" would
    stop being the whole claim.
    """
    res = request.getfixturevalue(mesh)
    for arm in ("fallback", "ready"):
        assert res[f"sharded/{arm}"]["finite_q_tile"] == \
            res[f"disk/{arm}"]["src_q1"], (
            f"the {arm} arm's q=(1,0,0) tile is not the bytes on disk")


@pytest.mark.parametrize("mesh", ["p1", "p4"])
def test_the_fallback_and_ready_arms_are_actually_different_files(mesh, request):
    """CONTROL.  Two arms that returned the same tile would gate nothing."""
    res = request.getfixturevalue(mesh)
    assert res["sharded/fallback"]["q0_tile"] != res["sharded/ready"]["q0_tile"]


def test_the_padding_really_differs_between_the_two_meshes(p1, p4):
    """CONTROL for the cross-mesh comparisons: P=4 is not P=1 in disguise."""
    assert p1["sharded/ready"]["padded_extent"] == N_MU
    assert p4["sharded/ready"]["padded_extent"] > N_MU


# ---------------------------------------------------------------------------
# One spelling, not two
# ---------------------------------------------------------------------------

def test_neither_loader_reaches_past_the_shared_injector():
    """The gate cannot drift apart again, because there is one of it.

    Both loaders call ``_inject_q0_head`` and NEITHER calls
    ``apply_q0_head_rank1_sharded`` directly — which is what the sharded
    path used to do, and how it came to carry a different gate from its
    twin.  A structural check, and it governs only WHERE the gate lives;
    every cell above decides whether it is right.
    """
    tree = ast.parse(_BSE_IO.read_text())
    for name in ("load_bse_data_from_restart_sharded", "_load_ring_subset"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        called = {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
                  for c in ast.walk(fn) if isinstance(c, ast.Call)}
        assert "_inject_q0_head" in called, (
            f"{name} no longer injects the q=0 head through the shared "
            f"helper; the two loaders' gates can drift again")
        assert "apply_q0_head_rank1_sharded" not in called, (
            f"{name} calls the rank-1 injector directly, past the "
            f"w0_ready gate that _inject_q0_head owns")


if __name__ == "__main__":
    # The worker.  Its own tree's ``src`` first: this file is run by path
    # from a worktree, where an installed/editable lorrax on the venv's
    # .pth would otherwise be a DIFFERENT checkout than the one under test.
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    _px, _py = (int(v) for v in sys.argv[1].split("x"))
    print(json.dumps(_worker(_px, _py)))
