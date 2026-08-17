"""Operator identities for the ladder screening resolvent (``w_bse``).

Each identity in DESIGN_2026-08-15 section 3 is its own cell with its full node
id spelled out (KNOWN_FAILURES:1292 lesson — a parametrized blob hides which
leg failed):

  * ``test_ladder_include_w_false_is_bit_identical_to_rpa`` — the ``include_w``
    switch's RPA limit is the RPA operator, bit for bit.  Agent B's
    wiring-closure test rides this flag, so "close" is not good enough.
  * ``test_ladder_delegates_to_rpa_builder_structurally`` — and it is the SAME
    object graph, not merely the same arguments (AST; no GPU, no fixture).
  * ``test_ladder_with_zero_w_kernel_reproduces_rpa_w`` — ``W_R -> 0`` with the
    direct term WIRED IN reproduces the RPA W to solver tolerance.  This is the
    one that exercises the new code path and still lands on a known answer.
  * ``test_ladder_w_tile_is_hermitian_at_q0`` — ``W_ladder(0)`` sub-tile
    Hermitian on the TRS fixture.
  * ``test_compare_wq_driver_arm_still_closes_after_the_loop_refactor`` — the
    driver arm that now CALLS the shared q loop still closes at the quadrature
    floor at every IBZ q, and is the only non-``extra`` coverage
    ``w_ladder.sweep_q_wedge`` has.
  * the ``refuse_chain_path`` / ``compute_wc_qwedge`` argument cells — the
    structural refusals, which need neither GPU nor restart.

The pre-existing ``tests/test_bse_w0_resolvent.py`` cells are NOT touched by
this feature and must stay green unchanged; nothing here imports from them.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
from jax.sharding import Mesh, PartitionSpec as P                # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_io                                           # noqa: E402
from bse.bse_w_exact import (                                    # noqa: E402
    _build_rpa_resolvent, apply_screening_resolvent_block,
    _symmetry_reduced_q_list,
)
from bse import w_ladder                                         # noqa: E402
from bse.w_ladder import (                                       # noqa: E402
    build_ladder_resolvent, compute_wc_qwedge, refuse_chain_path,
)


@pytest.fixture(scope="module")
def ladder_payload(gnppm_session):
    """Head-less 2v2c BSE payload + 1x1 mesh (same shape as the dense gate's)."""
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=2, n_cond=2, mesh_xy=mesh, input_file=input_path,
        inject_head=False, load_v_full=True)
    return data, mesh, input_path


def _cols(data, n=3):
    nlog = int(data["n_rmu"])
    T = (np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
         - np.asarray(jax.device_get(data["V_q0"])))
    order = np.argsort(-np.linalg.norm(T[:nlog, :nlog], axis=0))
    return order[:n].astype(int)


def _solve(data, mesh, stack, cols, z, *, max_iter=400, tol=1e-11,
           ladder=False):
    """``ladder=True`` for a ``build_ladder_resolvent(include_w=True)`` stack:
    its matvec carries the four rung operand slots and needs
    ``ladder_matvec_operands`` (on raw payloads the fallback aliases the
    density arrays, which are physical there)."""
    from bse.bse_feast import ladder_matvec_operands, matvec_operands
    matvec, diag_h, gen, snapshot, sh = stack
    n_rmu = int(data["V_q0"].shape[0])
    py = mesh.devices.shape[1]
    n_pad = int(np.ceil(len(cols) / py) * py)
    G = np.zeros((n_pad, n_rmu), dtype=np.float64)
    for i, nu0 in enumerate(cols):
        G[i, int(nu0)] = 1.0
    W_tile, resids = apply_screening_resolvent_block(
        G, z, data, matvec, diag_h, gen, snapshot, sh,
        max_iter=max_iter, tol=tol,
        operands_fn=(ladder_matvec_operands if ladder else matvec_operands))
    return (np.asarray(jax.device_get(W_tile))[:, :len(cols)],
            np.asarray(jax.device_get(resids))[:len(cols)], W_tile)


# ---------------------------------------------------------------------------
# Identity 1 — the include_w=False limit IS the RPA operator.
# ---------------------------------------------------------------------------
def test_ladder_delegates_to_rpa_builder_structurally():
    """``build_ladder_resolvent(include_w=False)`` delegates to
    ``_build_rpa_resolvent`` — one object graph, not a parallel one.

    Pure AST on the function source: no GPU, no restart, no jax execution.  A
    re-implemented RPA branch would satisfy any numeric comparison at first and
    then drift, which is exactly the shadow-accounting class
    (QUALITY_PATTERNS 3).
    """
    src = textwrap.dedent(inspect.getsource(build_ladder_resolvent))
    fn = ast.parse(src).body[0]
    guarded_returns = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "_build_rpa_resolvent"
    ]
    assert guarded_returns, (
        "build_ladder_resolvent(include_w=False) must RETURN "
        "_build_rpa_resolvent(...), not rebuild the RPA stack itself")


@pytest.mark.gpu
def test_ladder_include_w_false_is_bit_identical_to_rpa(ladder_payload):
    """``include_w=False`` reproduces the RPA W tile BIT for bit."""
    harness.skip_unless_gpu(pytest)
    data, mesh, _ = ladder_payload
    cols = _cols(data)
    rpa = _solve(data, mesh, _build_rpa_resolvent(mesh, data), cols, 0.0 + 0.0j)
    lad = _solve(data, mesh,
                 build_ladder_resolvent(mesh, data, include_w=False),
                 cols, 0.0 + 0.0j)
    assert np.array_equal(rpa[0], lad[0]), (
        "include_w=False is not bit-identical to the RPA resolvent: "
        f"max|delta| = {np.abs(rpa[0] - lad[0]).max():.3e}")
    assert lad[2].sharding.spec == P("x", "y")


# ---------------------------------------------------------------------------
# Identity 2 — W_R -> 0 with the ladder path WIRED IN reproduces the RPA W.
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_ladder_with_zero_w_kernel_reproduces_rpa_w(ladder_payload):
    """Zeroing the kernel W (not the flag) makes the ladder operator the RPA one.

    This is the identity that actually exercises the new code — the direct
    encode/convolve/decode chain, the swapped B-block encode and the
    anti-resonant row all RUN, contracted against a zero kernel — and still has
    a known answer.  Agreement is to solver tolerance, not bitwise: the ladder
    path adds `- 0` terms in a different fusion order.
    """
    harness.skip_unless_gpu(pytest)
    data, mesh, _ = ladder_payload
    cols = _cols(data)
    ref, ref_resid, _ = _solve(
        data, mesh, _build_rpa_resolvent(mesh, data), cols, 0.0 + 0.0j)

    import jax.numpy as jnp
    zeroed = dict(data)
    zeroed["W_q"] = jnp.zeros_like(data["W_q"])
    zeroed.pop("W_R", None)
    zeroed.pop("_W_q0_for_precond", None)
    got, resid, _ = _solve(
        zeroed, mesh, build_ladder_resolvent(mesh, zeroed, include_w=True),
        cols, 0.0 + 0.0j, ladder=True)

    assert resid.max() < 1e-8, f"GMRES not converged (resid {resid.max():.2e})"
    assert ref_resid.max() < 1e-8, f"RPA leg not converged ({ref_resid.max():.2e})"
    nlog = int(data["n_rmu"])
    rel = float(np.linalg.norm(got[:nlog] - ref[:nlog])
                / np.linalg.norm(ref[:nlog]))
    assert rel < 1e-8, f"W_R->0 ladder != RPA W: rel_err {rel:.3e}"


# ---------------------------------------------------------------------------
# Identity 3 — hermiticity of the static ladder tile on a TRS deck.
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_ladder_w_tile_is_hermitian_at_q0(ladder_payload):
    """``W_ladder(q=0, omega=0)`` sub-tile is Hermitian within solver tolerance.

    Probing the SAME index set on both axes gives a square sub-block of the
    (mu, nu) tile, so hermiticity is testable without ever forming the full
    mu^2 object.

    THIS CELL IS WHAT FOUND THE ANTI-RESONANT ROW.  With the naive symplectic
    row ``[-B, -A]`` it read 2.13e-05 and everything else in the suite —
    including the dense oracle — was green, because the oracle was built with
    the same row.  Hermiticity is the one observable that does not share the
    assumption.  With the derived row (ring un-conjugated, direct conjugated)
    it reads 6.9e-15.  Finite-q hermiticity is NOT covered here; it belongs to
    ``gw/screening._gate_w`` on the assembled tiles (w_ladder docstring, step 6).
    """
    harness.skip_unless_gpu(pytest)
    data, mesh, _ = ladder_payload
    cols = _cols(data, n=4)
    got, resid, _ = _solve(
        data, mesh, build_ladder_resolvent(mesh, data, include_w=True),
        cols, 0.0 + 0.0j, ladder=True)
    assert resid.max() < 1e-8, f"GMRES not converged (resid {resid.max():.2e})"
    sub = got[np.asarray(cols, dtype=int), :]           # (len(cols), len(cols))
    asym = float(np.abs(sub - sub.conj().T).max())
    scale = max(float(np.abs(sub).max()), 1e-300)
    assert asym / scale < 1e-9, (
        f"W_ladder(0) sub-tile is not Hermitian: max|W - W^dag|/|W| = "
        f"{asym / scale:.3e}. The derived row makes K^d a Hermitian kernel "
        f"matrix, so this is evidence about the operator (measured 6.9e-15), "
        f"not a tolerance to loosen — 2.1e-05 is the naive-row signature")


# ---------------------------------------------------------------------------
# Structural refusals and facade argument checks (no GPU, no restart).
# ---------------------------------------------------------------------------
def test_refuse_chain_path_refuses_the_ladder_and_names_the_reason():
    """The z^2 chain is refused for include_w=True, naming algebra + deferral."""
    with pytest.raises(NotImplementedError) as exc:
        refuse_chain_path(True)
    msg = str(exc.value)
    for token in ("A - B = D", "w_omega_chain", "SDY", "eta H", "full 2N"):
        assert token in msg, f"refusal text does not name {token!r}: {msg}"
    assert "make_ab_appliers" not in msg


def test_refuse_chain_path_permits_the_rpa_operator():
    """The RPA operator keeps A - B = D, so the chain stays legal for it."""
    assert refuse_chain_path(False) is None


def test_compute_wc_qwedge_refuses_a_missing_input_deck():
    """Omitting input_file names exactly what is missing (contract deviation)."""
    with pytest.raises(ValueError, match="input_file"):
        compute_wc_qwedge("/nonexistent/restart.h5", [0.0 + 0.0j], None,
                          gmres_tol=1e-10, gmres_max_iter=10)


def test_compute_wc_qwedge_refuses_a_probe_chunk_that_is_not_a_py_multiple():
    """probe_chunk must tile the reduce-scatter axis; the refusal names py."""
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    with pytest.raises(ValueError, match="probe_chunk"):
        compute_wc_qwedge("/nonexistent/restart.h5", [0.0 + 0.0j], mesh,
                          gmres_tol=1e-10, gmres_max_iter=10, probe_chunk=0,
                          input_file="/nonexistent/deck.in")


def test_compute_wc_qwedge_refuses_an_empty_z_list():
    """An empty frequency plan is a caller bug, not an empty result."""
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    with pytest.raises(ValueError, match="z_list_ry"):
        compute_wc_qwedge("/nonexistent/restart.h5", [], mesh,
                          gmres_tol=1e-10, gmres_max_iter=10,
                          input_file="/nonexistent/deck.in")


def test_w_ladder_wedge_spec_is_the_documented_sharding():
    """The wedge's PartitionSpec is P(None, None, 'x', 'y') — no replicated mu^2.

    Reads the module rather than a run: the spec is the contract Agent B's
    stage helper and ``symmetry_maps.unfold_isdf_operator`` are written against.
    """
    src = inspect.getsource(w_ladder.compute_wc_qwedge)
    assert 'P(None, None, "x", "y")' in src, (
        "compute_wc_qwedge no longer constrains the wedge to "
        "P(None, None, 'x', 'y'); that spec is the facade contract")


# ---------------------------------------------------------------------------
# Facade end-to-end (``extra``: the full-basis wedge is N_mu solves per (q, z),
# 399 x n_q on this fixture — real coverage, but not a per-commit cost).
# Run with: pytest tests/test_bse_w_ladder_identities.py -m extra -o addopts=''
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_compare_wq_driver_arm_still_closes_after_the_loop_refactor(gnppm_session,
                                                                    capsys):
    """``bse_w_exact.main --compare-wq`` now CALLS ``w_ladder.sweep_q_wedge``.

    The q loop moved out of the driver, so this cell is what keeps the move
    honest: it runs the arm in-process over the whole IBZ wedge and reads the
    closure numbers back out of its own table.  It is also the only coverage
    ``sweep_q_wedge`` has that is not deselected — the facade's full-basis
    end-to-end below is an ``extra``.

    Checks the FAILURE signature, not the return: the arm returns None whatever
    happens, so the assertion is on the printed per-q ``max_rel_err`` staying at
    the GW minimax-quadrature floor and on every q of the wedge appearing.
    """
    harness.skip_unless_gpu(pytest)
    from bse import bse_w_exact
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    bse_w_exact.main(["-i", input_path, "--compare-wq", "--n-cols", "4",
                      "--gmres-max-iter", "200", "--gmres-tol", "1e-10"])
    out = capsys.readouterr().out
    # ``iq  (qx, qy, qz)  q_flat  max_rel_err  median  max_resid  max_it ...``
    # — the q column carries spaces, so match it rather than splitting.
    row_re = re.compile(r"^\s*(\d+)\s+\(\s*\d+,\s*\d+,\s*\d+\)\s+(\d+)\s+(\S+)\s+")
    rows = [m for m in (row_re.match(ln) for ln in out.splitlines()) if m]
    n_q = len(_symmetry_reduced_q_list(input_path))
    assert len(rows) == n_q, (
        f"--compare-wq printed {len(rows)} q rows for a {n_q}-point IBZ wedge; "
        "the refactored sweep dropped or duplicated q points")
    assert [int(m.group(1)) for m in rows] == list(range(n_q)), (
        "q rows are out of order or renumbered")
    rel = np.array([float(m.group(3)) for m in rows])
    assert rel.max() < 1e-6, (
        f"finite-q RPA closure regressed after the loop refactor: "
        f"max per-q rel_err {rel.max():.3e}")


@pytest.mark.gpu
@pytest.mark.extra
def test_compute_wc_qwedge_end_to_end(gnppm_session):
    """The facade returns a padded, wedge-sharded W(z) - v with residuals.

    ``wc`` is on the PADDED mu extent with the logical count in ``n_rmu`` —
    the design's "pad stripped" wording cannot coexist with the documented
    ``P(None, None, 'x', 'y')`` sharding (399 does not divide a 2x2), which
    the wiring-closure gate measured on 2026-08-15.  Both facts are asserted
    here so the contract cannot drift back silently.
    """
    harness.skip_unless_gpu(pytest)
    import time

    from bse.bse_ring_comm import create_mesh_xy
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = create_mesh_xy(1, 1)
    # BOUNDED PROBE CHUNK.  ``probe_chunk=None`` solves the whole padded basis
    # (399 columns on this fixture) as one block, which is the shape that made
    # this cell an ``extra`` in the first place.  A chunked sweep is the thing
    # worth gating: it walks the SAME compiled engine once per chunk and must
    # still close, so it certifies the memory knob rather than only the default.
    chunk = 64
    t0 = time.time()
    wedge = compute_wc_qwedge(
        restart, [0.0 + 0.0j], mesh, include_w=True,
        gmres_tol=1e-9, gmres_max_iter=300, probe_chunk=chunk,
        input_file=input_path)
    wall = time.time() - t0
    npad = wedge.wc.shape[-1]
    print(f"[wc_qwedge] {len(wedge.q_irr_kgrid_int)} q x 1 z, probe_chunk="
          f"{chunk}, mu_pad={npad}, n_rmu={wedge.n_rmu}: {wall:.1f} s, "
          f"max resid {wedge.gmres_resid.max():.2e}, max iters "
          f"{int(wedge.gmres_iters.max())}")
    assert wedge.wc.shape == (1, len(wedge.q_irr_kgrid_int), npad, npad)
    # EQUIVALENCE, not spec equality.  On a 1x1 mesh JAX normalises the
    # constraint ``P(None, None, 'x', 'y')`` to ``PartitionSpec()`` — the two
    # describe the same placement when both axes are size 1 — so a string
    # compare reports a failure that is purely about how the spec is spelled.
    # ``is_equivalent_to`` asks the question the contract actually makes.
    assert wedge.wc.sharding.is_equivalent_to(
        jax.sharding.NamedSharding(mesh, P(None, None, "x", "y")),
        wedge.wc.ndim), (
        f"the wedge is not on the documented sharding: got "
        f"{wedge.wc.sharding.spec} on a "
        f"{mesh.devices.shape[0]}x{mesh.devices.shape[1]} mesh")
    # PADDED, with the logical count carried alongside — the contract the
    # sharding forces (see this module's note above).
    assert npad >= wedge.n_rmu
    assert wedge.gmres_resid.max() < 1e-6, (
        f"unconverged columns in the wedge: max resid "
        f"{wedge.gmres_resid.max():.2e}")
    assert wedge.include_w is True
    assert tuple(wedge.q_irr_kgrid_int[0]) == (0, 0, 0)
    # The pad columns must solve to ZERO, not to noise: they probe a centroid
    # whose psi is zero.  A non-zero pad would ride the unfold into every
    # downstream contraction as a centroid that does not exist.
    if npad > wedge.n_rmu:
        pad = np.asarray(jax.device_get(wedge.wc[:, :, wedge.n_rmu:, :]))
        assert np.all(pad == 0.0), (
            f"pad rows of the ladder wedge are not zero "
            f"(max |.| = {np.abs(pad).max():.3e})")
