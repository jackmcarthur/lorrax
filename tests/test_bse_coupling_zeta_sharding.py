"""Cross-mesh gate for the non-TDA COUPLING block under ζ sharding.

THE regression guard for the 2026-08-08 ``K^d_B`` defect.  The coupling
block's encode carries Henneke's ``j_c <-> j_v`` swap, so the ζ index it
builds by contracting the conduction axis (sharded on ``'x'``) is ``nu``
(sharded on ``'y'``), and the one it builds from the valence axis (on
``'y'``) is ``mu`` (on ``'x'``).  That pairing is CROSSED relative to the
resonant block's, and the shipped two-stage ring consequently ``ppermute``d
a partially-contracted intermediate along ``'y'`` -- the very axis its
``nu`` shard lived on.  A ppermute moves every axis of the buffer it is
handed, so each ``'y'`` rank accumulated its neighbours' ζ tiles against its
own ζ shard.

The failure was invisible at P=1 (one shard; ppermute is the identity) and
cost two thirds of the coupling correction at P=2x2 (-0.698 -> -0.223 meV),
with the block losing the complex symmetry ``B = B^T`` it is required to
have (2.9e-11 -> 6.9e-01).  Nothing tested the coupling block at P>1, which
is why it shipped.

These gates are PORTABLE: a synthetic payload on CPU host devices, no GPU, no
restart file and no deck.  They assemble the REAL half-appliers
(``build_bse_ring_matvec_full(..., return_half_appliers=True)``) on a 1x1 and
a 2x2 mesh and compare, on BOTH the ``low_mem`` ring route and the
``all_gather`` route, and they carry their FALSE case: the pre-fix chain,
monkeypatched back in, must turn every one of them red.

Evidence: ~/lorrax_bse_perf_2026-08-08/FIX_kdb_sharding.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_NDEV = 4
# The correct routes agree at the ULP floor (measured 2.2e-16); the defect is
# an O(1) effect (measured 5.5e-01 on the block, 9.9e-01 on the encode).  1e-12
# separates them by ten orders of magnitude on either side.
_TOL = 1.0e-12
_RED_MIN = 1.0e-3

NKX, NKY, NKZ = 2, 2, 3          # nkz=3 so R -> -R is not the identity
NK = NKX * NKY * NKZ
NC, NV, NS, NMU, NCOL = 4, 4, 1, 8, 4
N = NC * NV * NK


# ---------------------------------------------------------------------------
# the PRE-FIX coupling encode, kept here as the red twin
# ---------------------------------------------------------------------------
def _pre_fix_B_encode(X, psi_c_Y, psi_v_X, v_chunk, px, py, mu_local,
                      nu_local):
    """The 2026-08-08 defect, in the signature of ``_ring_sum_B_encode``.

    Ring 'x' to consume c -- correct, nu is produced into a stationary
    accumulator -- then ring 'y' to consume v, which drags the buffer's nu
    shard along with its v block.
    """
    import jax.numpy as jnp
    from jax import lax
    from common.vma import mark_varying
    from bse.bse_ring_comm import _ring_perm

    c_chunk = X.shape[1]
    nk, _, ns, _ = psi_c_Y.shape
    z = jnp.int32(0)
    xi = jnp.asarray(lax.axis_index("x"), jnp.int32)
    yj = jnp.asarray(lax.axis_index("y"), jnp.int32)

    R0 = mark_varying(
        jnp.zeros((X.shape[0], v_chunk, nk, ns, nu_local), X.dtype), ("x", "y"))

    def step_c(i, carry):
        buf, R = carry
        origin = (xi - jnp.asarray(i, jnp.int32)) % px
        pc = lax.dynamic_slice(psi_c_Y, (z, origin * c_chunk, z, z),
                               (nk, c_chunk, ns, nu_local))
        R = R + jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(pc), buf)
        return lax.ppermute(buf, "x", _ring_perm(px)), R

    R = lax.fori_loop(0, px, step_c, (X, R0))[1]

    T0 = mark_varying(
        jnp.zeros((X.shape[0], mu_local, nu_local, ns, ns, nk), X.dtype),
        ("x", "y"))

    def step_v(i, carry):
        buf, T = carry
        origin = (yj - jnp.asarray(i, jnp.int32)) % py
        pv = lax.dynamic_slice(psi_v_X, (z, origin * v_chunk, z, z),
                               (nk, v_chunk, ns, mu_local))
        T = T + jnp.einsum("kvtM,bvksN->bMNtsk", pv, buf)
        return lax.ppermute(buf, "y", _ring_perm(py)), T   # <- moves nu too

    return lax.fori_loop(0, py, step_v, (R, T0))[1]


def _worker() -> int:
    """Child process: assemble A, B and K^d_B on a 1x1 and a 2x2 CPU mesh,
    on both encode routes, with the shipped chain and with the pre-fix one."""
    import numpy as np
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import bse.bse_ring_comm as brc
    from bse.bse_serial import compute_pair_amplitude

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0

    rng = np.random.default_rng(20260808)
    cx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    psi_c, psi_v = cx(NK, NC, NS, NMU), cx(NK, NV, NS, NMU)
    eps_c = np.sort(rng.random((NK, NC)), axis=1) + 1.0
    eps_v = np.sort(rng.random((NK, NV)), axis=1) - 1.0
    V = cx(NMU, NMU)
    V_q0 = V + V.conj().T
    # W with the real-space reciprocity W_R[mu,nu] = W_{-R}[nu,mu], which is
    # exactly the condition the coupling block needs to come out complex
    # SYMMETRIC -- so ||K^d_B - (K^d_B)^T|| is a physical gate here, not just a
    # cross-mesh one.
    G = cx(NMU, NMU, NKX, NKY, NKZ)
    Gm = np.roll(G[:, :, ::-1, ::-1, ::-1], 1, axis=(2, 3, 4))     # G_{-R}
    W_R = 0.5 * (G + np.transpose(Gm, (1, 0, 2, 3, 4)))

    def blocks(px, py, low_mem):
        mesh = Mesh(np.asarray(devs[:px * py]).reshape(px, py), ("x", "y"))
        sh = brc.make_bse_shardings(mesh)
        pcx = jax.device_put(psi_c, sh.psi_x); pcy = jax.device_put(psi_c, sh.psi_y)
        pvx = jax.device_put(psi_v, sh.psi_x); pvy = jax.device_put(psi_v, sh.psi_y)
        ec = jax.device_put(eps_c, sh.eps); ev = jax.device_put(eps_v, sh.eps)
        Wd = jax.device_put(W_R, sh.W); Vd = jax.device_put(V_q0, sh.V)
        M_X = jax.jit(compute_pair_amplitude, out_shardings=sh.psi_x)(pcx, pvx)
        eye = np.eye(N, dtype=np.complex128)
        out = {}
        for tag, incW in (("W", True), ("x", False)):
            _mv, apA, apB = brc.build_bse_ring_matvec_full(
                mesh, NKX, NKY, NKZ, low_mem=low_mem, include_W=incW,
                screening=False, return_half_appliers=True)
            A = np.empty((N, N), np.complex128) if incW else None
            B = np.empty((N, N), np.complex128)
            for j0 in range(0, N, NCOL):
                col = jax.device_put(
                    eye[j0:j0 + NCOL].reshape(-1, NC, NV, NK), sh.X)
                if incW:
                    A[:, j0:j0 + NCOL] = np.asarray(jax.device_get(apA(
                        col, pcx, pcy, pvx, pvy, ec, ev, Wd, Vd, M_X))
                    ).reshape(NCOL, -1).T
                B[:, j0:j0 + NCOL] = np.asarray(jax.device_get(apB(
                    col, pcx, pcy, pvx, pvy, Wd, Vd, M_X))).reshape(NCOL, -1).T
            if incW:
                out["A"] = A
            out[f"B_{tag}"] = B
        out["Kd_B"] = out["B_x"] - out["B_W"]
        return out

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(a))

    def sym(m):
        return float(np.linalg.norm(m - m.T) / np.linalg.norm(m))

    def score(key, low_mem):
        one, four = blocks(1, 1, low_mem), blocks(2, 2, low_mem)
        res[key] = {
            "A": rel(one["A"], four["A"]),
            "B": rel(one["B_W"], four["B_W"]),
            "Bx": rel(one["B_x"], four["B_x"]),
            "Kd": rel(one["Kd_B"], four["Kd_B"]),
            "sym1": sym(one["Kd_B"]), "sym4": sym(four["Kd_B"]),
        }

    # The GREEN arms touch no module internals: whatever the coupling encode is
    # spelled as, the public half-appliers must agree across meshes.  A revert
    # of the fix therefore fails these on the NUMBER, not on a missing name.
    res = {}
    for low_mem in (True, False):
        score(f"fix_{'ring' if low_mem else 'gather'}", low_mem)

    # The RED arm needs a seam.  It patches the shipped coupling encode helper;
    # if that helper is gone the twin cannot be installed and says so rather
    # than passing silently.
    shipped = getattr(brc, "_ring_sum_B_encode", None)
    if shipped is None:
        res["red_unavailable"] = (
            "bse_ring_comm._ring_sum_B_encode is gone, so the pre-fix twin "
            "could not be installed")
    else:
        brc._ring_sum_B_encode = _pre_fix_B_encode
        try:
            score("red_ring", True)      # the gather route has its own body
        finally:
            brc._ring_sum_B_encode = shipped
    print(json.dumps(res))
    return 0


def _run_worker(tag: str, timeout: int = 900):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={_NDEV}").strip()
    # THIS GATE'S OWN OPT-OUT (P19, 2026-08-09) -- the same declaration
    # tests/test_bse_coupling_routes_mesh_invariance.py::_results makes, for
    # the same reason.  The worker's call chain reaches
    # contract_bands_block_reshard -> gate.require, which REFUSES on a box
    # with no mklblas host handler unless the dial announces the debug
    # opt-out.  This gate used to inherit that pin from a module-scope
    # os.environ.setdefault that leaked out of tests/test_contract_bands.py
    # at collection time, so it was green in the default census and red
    # standalone on an FFI-less box.  Declaring it here makes the child's
    # environment identical in every arrangement.  The subject of this gate
    # is zeta sharding, not the GEMM plan; gate.require is untouched.
    env["LORRAX_BANDS_GEMM_FFI"] = "0"
    res = subprocess.run([sys.executable, os.path.abspath(__file__), tag],
                         env=env, capture_output=True, text=True,
                         timeout=timeout)
    assert res.returncode == 0, (
        f"worker {tag} failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}")
    line = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert line, f"no JSON.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(line[-1])


@pytest.fixture(scope="module")
def cross_mesh():
    out = _run_worker("worker")
    if "skip" in out:
        pytest.skip(f"coupling cross-mesh gate: {out['skip']}")
    return out


@pytest.mark.parametrize("route", ["ring", "gather"])
def test_coupling_block_is_mesh_invariant(cross_mesh, route):
    """A, the coupling exchange and the coupling screened-direct term all
    agree between a 1x1 and a 2x2 mesh, on both encode routes."""
    r = cross_mesh[f"fix_{route}"]
    for name, key in (("A (resonant)", "A"), ("B (coupling)", "B"),
                      ("K^x_B (coupling exchange)", "Bx"),
                      ("K^d_B (coupling screened-direct)", "Kd")):
        assert r[key] <= _TOL, (
            f"{name} differs between a 1x1 and a 2x2 mesh on the {route} "
            f"encode: ||P1-P4||/||P1|| = {r[key]:.3e} > {_TOL:g}.  A ζ-sharded "
            f"contraction that depends on the mesh shape is wrong on one of "
            f"them; see FIX_kdb_sharding.md.")


@pytest.mark.parametrize("route", ["ring", "gather"])
def test_coupling_screened_direct_stays_complex_symmetric(cross_mesh, route):
    """On a payload whose W carries the real-space reciprocity
    ``W_R[mu,nu] = W_{-R}[nu,mu]``, ``K^d_B`` is complex symmetric -- and
    stays so when the ζ axes are sharded.  This is the invariant the non-TDA
    operator's whole spectrum rests on."""
    r = cross_mesh[f"fix_{route}"]
    assert r["sym1"] <= 1e-10, (
        f"the fixture is not delivering a symmetric K^d_B at P=1 "
        f"({r['sym1']:.3e}); the gate below would be meaningless")
    assert r["sym4"] <= 1e-10, (
        f"K^d_B loses its complex symmetry at 2x2 on the {route} encode: "
        f"||K-K^T||/||K|| = {r['sym4']:.3e} at 2x2 vs {r['sym1']:.3e} at 1x1")


def test_the_pre_fix_coupling_encode_is_caught(cross_mesh):
    """FALSE case.  The 2026-08-08 chain -- ring 'y' over a buffer carrying
    the 'y'-sharded nu index -- must trip both gates above, and must trip
    ONLY the coupling screened-direct term: A and the coupling exchange do
    not go through it and must stay clean."""
    if "red_unavailable" in cross_mesh:
        pytest.fail(
            f"the FALSE case could not be run: {cross_mesh['red_unavailable']}."
            f"  A gate whose red twin cannot be installed is not a gate — "
            f"re-point the twin at whatever the coupling encode is now called.")
    red = cross_mesh["red_ring"]
    assert red["Kd"] >= _RED_MIN, (
        f"the pre-fix coupling encode was re-introduced and the cross-mesh "
        f"gate did not see it: ||P1-P4||/||P1|| = {red['Kd']:.3e}")
    assert red["sym4"] >= _RED_MIN, (
        f"the pre-fix coupling encode was re-introduced and the symmetry gate "
        f"did not see it: ||K-K^T||/||K|| = {red['sym4']:.3e}")
    # ... and it is specific: the twin must NOT move the blocks it does not touch.
    assert red["A"] <= _TOL, (
        f"the red twin moved the RESONANT block ({red['A']:.3e}); it is "
        f"supposed to isolate the coupling screened-direct term")
    assert red["Bx"] <= _TOL, (
        f"the red twin moved the coupling EXCHANGE ({red['Bx']:.3e}); it is "
        f"supposed to isolate the coupling screened-direct term")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        raise SystemExit(_worker())
    raise SystemExit(2)
