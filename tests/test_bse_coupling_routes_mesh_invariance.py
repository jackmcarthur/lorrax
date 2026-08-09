"""THE CLASS GATE: every coupling-block route is mesh-invariant, or this is red.

The defect class, twice instantiated and twice cured on 2026-08-08:

    A coupling-block encode communicates a PARTIALLY CONTRACTED intermediate
    along a mesh axis on which that intermediate's own zeta shard also lives.

The coupling (B) block carries Henneke's ``j_c <-> j_v`` swap, so the zeta
index built by contracting the conduction axis (sharded on ``'x'``) is ``nu``
(sharded on ``'y'``), and the one built from the valence axis (on ``'y'``) is
``mu`` (on ``'x'``).  That pairing is CROSSED relative to the resonant block's.
A ``ppermute``/``all_gather`` moves EVERY axis of the buffer it is handed, so
transporting the partial ``R`` -- which carries ``nu`` on ``'y'`` as well as
``v`` on ``'y'`` -- makes each ``'y'`` rank accumulate its neighbours' zeta
tiles against its own zeta shard.

It is bit-exact at P=1 (a one-rank collective is the identity) and wrong at
every P>1.  That is what makes the class so quiet: nothing short of assembling
the block on two different meshes can see it.

The cure, both times: communicate the TRIAL VECTOR, which carries no zeta axis
at all.  Both zeta legs then stay in stationary accumulators.

    instance 1  bse_ring_comm._ring_sum_B_encode / _encode_T_B_gather
                fix/kdb-zeta-sharding-2026-08-08 @ 443a23fe, FIX_kdb_sharding.md
    instance 2  bse_stack_matvec._encode_T_B  (a PORT of instance 1, which
                inherited the defect and did not inherit the fix)
                @ 3a8223e4, CONSOLIDATION2_REPORT.md §3-4

Instance 2 was found only because the SDY lane declared its port and the
consolidation census re-measured it.  Instance 1's gate
(``test_bse_coupling_zeta_sharding.py``) covers ``bse_ring_comm`` ONLY, so the
ported encode -- the production non-TDA matrix-free route -- was cured but
UNGUARDED: every test touching ``build_bse_stack_pair_matvec`` builds a 1x1
mesh (``test_bse_sp_lanczos._pair_ctx``), the one regime where the defect is
invisible by construction.  A revert would have been silent.

THIS gate closes the class rather than the instance.  It is PARAMETRIZED over
a registry of every live multi-device coupling route, so:

  * every registered route is asserted mesh-invariant and complex-symmetric,
    through the REAL public builders -- no module internals are touched by the
    green arms, so a revert of either fix fails these on the NUMBER, not on a
    missing symbol;
  * every route is asserted to agree with every other route at both meshes,
    which is what caught instance 2;
  * a NEW coupling encode added later is DISCOVERED by
    ``test_route_registry_covers_every_coupling_encode`` and must either be
    exercised by a registered route or be explicitly waived with a reason.
    That test fails CLOSED: silence is not coverage.

Relationship to ``test_fft_shardmap_context`` (the ratchet noted in
CONSOLIDATION2_REPORT.md §5): that gate is a LEXICAL proxy -- it passes a call
when some function in the enclosing chain mentions a shard_map-constructing
name, which is why the SDY port's code motion forced a ratchet entry there.
This gate is BEHAVIOURAL: it runs the real appliers on two real meshes and
compares numbers, so code motion cannot fool it and it needs no ratchet.  The
one structural check here (the registry-completeness scan) governs only WHICH
routes get behaviourally gated; it never decides whether code is correct.  It
does not touch, extend, or depend on the fft gate's ratchet list.

PORTABLE: synthetic payload on CPU host devices, no GPU, no deck, no restart
file, no FFI requirement of its own.  Runs in the default census, and -- since
P19 (2026-08-09) -- gives the SAME verdict standalone, because ``_results``
now pins ``LORRAX_BANDS_GEMM_FFI=0`` in the worker environment it builds
rather than inheriting that pin from whatever else the session collected.
The gate makes no FFI requirement; what it CALLS does, and that is the
distinction the inherited pin used to blur.

  CAVEAT, MEASURED 2026-08-09 (landing-completeness audit) — "no FFI
  requirement OF ITS OWN" is exact, and it is not the same as "runs anywhere".
  On a box with NO built ``.so``, this file's result depends on COLLECTION
  SCOPE, because what it calls does have an FFI requirement:
  ``build_bse_ring_matvec_full`` -> ``contract_bands_block_reshard`` ->
  ``gate.require`` refuses with "MKL batched-GEMM host backend unavailable"
  unless ``LORRAX_BANDS_GEMM_FFI=0`` announces the debug opt-out.  Measured on
  WSL, no ``.so``, same tree, same commit:

      pytest tests/test_bse_coupling_routes_mesh_invariance.py   ->  13 failed, 2 passed
      LORRAX_BANDS_GEMM_FFI=0 pytest <same file>                 ->  15 passed

  It is green in the default census only because collecting
  ``tests/test_contract_bands.py`` executes a MODULE-SCOPE
  ``os.environ.setdefault("LORRAX_BANDS_GEMM_FFI", "0")`` (that file, top of
  module) which leaks to the whole session.  So the 13-red is an artefact of
  running this file alone on an FFI-less box, NOT a mesh-invariance failure and
  NOT evidence that either K^d_B fix regressed.  Set the variable, or run the
  full census, before reading a red here as a physics result.

  The durable fix belongs to the OTHER file: move that ``setdefault`` into a
  fixture so it cannot leak.  Left alone here because it changes another gate's
  setup, not this one's.

Evidence: ~/lorrax_bse_perf_2026-08-08/COUPLING_MESH_GATE.md
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys
from functools import lru_cache

import pytest

_NDEV = 4
# Measured spread on this payload: every correct route sits at 2e-16..7e-16,
# every defective one at 4e-01..1.2e+00.  1e-12 separates them by ten orders of
# magnitude on either side; _RED_MIN is the floor a twin must clear to count as
# caught.
_TOL = 1.0e-12
_RED_MIN = 1.0e-3

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


# ===========================================================================
#  THE ROUTE REGISTRY -- every live multi-device coupling-block route
# ===========================================================================
#: id -> human description.  Adding a route here is all it takes to put it
#: under every gate in this file; the worker builds it from the id.
COUPLING_ROUTES = {
    "ring": "bse_ring_comm.build_bse_ring_matvec_full(low_mem=True) "
            "half-applier apply_B -- the production non-TDA ring route",
    "gather": "bse_ring_comm.build_bse_ring_matvec_full(low_mem=False) "
              "half-applier apply_B -- the all_gather encode spelling",
    "stack_fused": "bse_stack_matvec.build_bse_stack_pair_matvec(fuse=True) "
                   "-- the SDY matrix-free pair applier (production)",
    "stack_unfused": "bse_stack_matvec.build_bse_stack_pair_matvec(fuse=False) "
                     "-- the unfused twin that prices the fusion",
}

#: The coupling encodes each route exercises, by qualified name.  This is the
#: bridge between the STRUCTURAL discovery below and the BEHAVIOURAL gates: an
#: encode is covered iff some registered route drives it.
_ENCODE_TO_ROUTES = {
    "bse/bse_ring_comm.py::_ring_sum_B_encode": ("ring",),
    "bse/bse_ring_comm.py::build_bse_ring_matvec_full._encode_T_B_gather": (
        "gather",),
    "bse/bse_stack_matvec.py::_encode_T_B": ("stack_fused", "stack_unfused"),
    "bse/bse_stack_matvec.py::build_bse_stack_pair_matvec._w_pair": (
        "stack_fused", "stack_unfused"),
}

#: Encodes deliberately NOT behaviourally gated, each with the reason it cannot
#: harbour the defect.  Empty today.  An entry here is a claim someone has to
#: defend, which is the point of making it explicit rather than implicit.
_WAIVED_ENCODES: dict[str, str] = {}

_COLLECTIVES = {"all_gather", "ppermute", "psum_scatter", "all_to_all"}


def _discover_coupling_encodes() -> dict[str, int]:
    """Every function in ``src/`` carrying the defect's SHAPE.

    The shape is two things at once, and neither alone is interesting:

      1. the CROSSED zeta pairing -- both ``psi_c_Y`` and ``psi_v_X`` in the
         signature.  That naming convention is what marks a coupling encode in
         this tree: the conduction wavefunction sharded on ``'y'`` against the
         valence one on ``'x'``, i.e. the ``j_c <-> j_v`` swap.  The resonant
         block's encodes take the uncrossed ``(psi_c_X, psi_v_Y)``.
      2. a COLLECTIVE in the same body -- ``all_gather`` / ``ppermute`` /
         ``psum_scatter`` / ``all_to_all``.  Without transport there is nothing
         to get wrong; a pure delegator (e.g. the thin ``_encode_T_B`` wrapper
         that only calls ``_ring_sum_B_encode``) is covered through its
         delegate, which this scan does find.

    Returns ``{qualified_name: lineno}``.  Pure AST: no jax, no imports of the
    tree under test, runs on a login node in milliseconds.
    """
    found: dict[str, int] = {}
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:              # not ours to police
            continue
        stack: list[str] = []

        class _V(ast.NodeVisitor):
            def visit_FunctionDef(self, node):          # noqa: N802
                stack.append(node.name)
                a = node.args
                names = {p.arg for p in (*a.args, *a.kwonlyargs, *a.posonlyargs)}
                if {"psi_c_Y", "psi_v_X"} <= names:
                    for sub in ast.walk(node):
                        if (isinstance(sub, ast.Call)
                                and isinstance(sub.func, ast.Attribute)
                                and sub.func.attr in _COLLECTIVES):
                            rel_path = path.relative_to(_SRC).as_posix()
                            key = f"{rel_path}::{'.'.join(stack)}"
                            found[key] = node.lineno
                            break
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

        _V().visit(tree)
    return found


def test_route_registry_covers_every_coupling_encode():
    """A new coupling encode is gated, waived, or this suite is red.

    THE DURABLE PART.  The behavioural gates below can only test routes they
    know about; this is what makes "knows about" a property of the tree rather
    than of whoever last edited this file.  It fails CLOSED in both directions:
    an undeclared encode is a failure, and a stale registry entry is too.
    """
    found = _discover_coupling_encodes()
    assert found, (
        "the coupling-encode scan found NOTHING, which means its marker has "
        "drifted from the tree (renamed psi_c_Y/psi_v_X, or the collectives "
        "moved).  A gate that cannot find its subject is not a passing gate.")

    known = set(_ENCODE_TO_ROUTES) | set(_WAIVED_ENCODES)
    unregistered = sorted(set(found) - known)
    assert not unregistered, (
        "NEW COUPLING ENCODE(S) WITH NO MESH GATE:\n  "
        + "\n  ".join(f"{k} (line {found[k]})" for k in unregistered)
        + "\n\nThis is the 2026-08-08 defect class: an encode carrying the "
          "crossed zeta pairing AND a collective. Either\n"
          "  (a) add a route to COUPLING_ROUTES that drives it and map it in "
          "_ENCODE_TO_ROUTES (preferred -- that gets it behaviourally gated), or\n"
          "  (b) add it to _WAIVED_ENCODES with the reason it cannot transport "
          "a zeta-carrying partial.\n"
          "Do not delete this assertion.")

    stale = sorted(known - set(found))
    assert not stale, (
        "registry lists coupling encodes that no longer exist: "
        + ", ".join(stale)
        + "\nIf a route was removed, drop it from _ENCODE_TO_ROUTES/"
          "_WAIVED_ENCODES and from COUPLING_ROUTES.")

    for enc, routes in _ENCODE_TO_ROUTES.items():
        for r in routes:
            assert r in COUPLING_ROUTES, (
                f"{enc} is mapped to unknown route {r!r}")


# ===========================================================================
#  the worker: assemble every route's blocks on a 1x1 and a 2x2 CPU mesh
# ===========================================================================
NKX, NKY, NKZ = 2, 2, 3          # nkz=3 so R -> -R is not the identity
NK = NKX * NKY * NKZ
NC, NV, NS, NMU, NCOL = 4, 4, 1, 8, 4
N = NC * NV * NK


def _pre_fix_stack_encode(Xb_b, psi_c_Y, psi_v_X):
    """RED TWIN for instance 2 -- the PRE-PORT bse_stack_matvec._encode_T_B.

    Gathers the partially contracted ``R`` along ``'y'``, the very axis ``R``'s
    own ``nu`` shard lives on.
    """
    import jax.numpy as jnp
    from jax import lax
    R = jnp.einsum("kcsN,cvk->vksN", jnp.conj(psi_c_Y), Xb_b)
    Rv = lax.all_gather(R, "y", axis=0, tiled=True)      # <- moves nu too
    return jnp.einsum("kvtM,vksN->MNtsk", psi_v_X, Rv)


def _pre_fix_ring_encode(X, psi_c_Y, psi_v_X, v_chunk, px, py, mu_local,
                         nu_local):
    """RED TWIN for instance 1 -- the PRE-FIX _ring_sum_B_encode.

    Ring ``'x'`` to consume c (correct: nu goes to a stationary accumulator),
    then ring ``'y'`` to consume v, which drags the buffer's nu shard along.
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
    """Child process: score every registered route on 1x1 vs 2x2, then score
    the two red twins THROUGH THE SAME scoring function."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    import bse.bse_ring_comm as brc
    import bse.bse_stack_matvec as bsm
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
    # W carrying the real-space reciprocity W_R[mu,nu] = W_{-R}[nu,mu] -- the
    # condition that makes K^d_B complex SYMMETRIC.  That turns the symmetry
    # column into a PHYSICAL detector, not merely a second cross-mesh one.
    G = cx(NMU, NMU, NKX, NKY, NKZ)
    Gm = np.roll(G[:, :, ::-1, ::-1, ::-1], 1, axis=(2, 3, 4))
    W_R = 0.5 * (G + np.transpose(Gm, (1, 0, 2, 3, 4)))
    eye = np.eye(N, dtype=np.complex128)

    def payload(px, py):
        mesh = Mesh(np.asarray(devs[:px * py]).reshape(px, py), ("x", "y"))
        sh = brc.make_bse_shardings(mesh)
        d = dict(mesh=mesh, sh=sh)
        d["pcx"] = jax.device_put(psi_c, sh.psi_x)
        d["pcy"] = jax.device_put(psi_c, sh.psi_y)
        d["pvx"] = jax.device_put(psi_v, sh.psi_x)
        d["pvy"] = jax.device_put(psi_v, sh.psi_y)
        d["ec"] = jax.device_put(eps_c, sh.eps)
        d["ev"] = jax.device_put(eps_v, sh.eps)
        d["Wd"] = jax.device_put(W_R, sh.W)
        d["Vd"] = jax.device_put(V_q0, sh.V)
        d["M_X"] = jax.jit(compute_pair_amplitude,
                           out_shardings=sh.psi_x)(d["pcx"], d["pvx"])
        d["M_Y"] = jax.jit(compute_pair_amplitude,
                           out_shardings=sh.psi_y)(d["pcy"], d["pvy"])
        return d

    def ring_blocks(px, py, *, low_mem):
        d = payload(px, py); sh = d["sh"]
        out = {}
        for tag, incW in (("W", True), ("x", False)):
            _mv, apA, apB = brc.build_bse_ring_matvec_full(
                d["mesh"], NKX, NKY, NKZ, low_mem=low_mem, include_W=incW,
                screening=False, return_half_appliers=True)
            A = np.empty((N, N), np.complex128) if incW else None
            B = np.empty((N, N), np.complex128)
            for j0 in range(0, N, NCOL):
                col = jax.device_put(
                    eye[j0:j0 + NCOL].reshape(-1, NC, NV, NK), sh.X)
                if incW:
                    A[:, j0:j0 + NCOL] = np.asarray(jax.device_get(apA(
                        col, d["pcx"], d["pcy"], d["pvx"], d["pvy"],
                        d["ec"], d["ev"], d["Wd"], d["Vd"], d["M_X"]))
                    ).reshape(NCOL, -1).T
                B[:, j0:j0 + NCOL] = np.asarray(jax.device_get(apB(
                    col, d["pcx"], d["pcy"], d["pvx"], d["pvy"],
                    d["Wd"], d["Vd"], d["M_X"]))).reshape(NCOL, -1).T
            if incW:
                out["A"] = A
            out[f"B_{tag}"] = B
        out["Kd_B"] = out["B_x"] - out["B_W"]
        return out

    def stack_blocks(px, py, *, fuse):
        d = payload(px, py); sh = d["sh"]
        out = {}
        for tag, kern in (("W", "bse"), ("x", "rpa")):
            pair = bsm.build_bse_stack_pair_matvec(
                d["mesh"], NKX, NKY, NKZ, kernel=kern, fuse=fuse)
            A = np.empty((N, N), np.complex128)
            AB = np.empty((N, N), np.complex128)
            with d["mesh"]:
                for j0 in range(0, N, NCOL):
                    col = jax.device_put(
                        eye[j0:j0 + NCOL].reshape(-1, NC, NV, NK), sh.X)
                    args = (d["pcx"], d["pcy"], d["pvx"], d["pvy"], d["ec"],
                            d["ev"], d["Wd"], d["Vd"], d["M_X"], d["M_Y"])
                    r0 = pair(col, jnp.asarray(0.0), *args)
                    r1 = pair(col, jnp.asarray(1.0), *args)
                    A[:, j0:j0 + NCOL] = np.asarray(
                        jax.device_get(r0)).reshape(NCOL, -1).T
                    AB[:, j0:j0 + NCOL] = np.asarray(
                        jax.device_get(r1)).reshape(NCOL, -1).T
            out[f"A_{tag}"] = A
            out[f"B_{tag}"] = AB - A
        out["A"] = out["A_W"]
        out["Kd_B"] = out["B_x"] - out["B_W"]
        return out

    #: id -> builder.  The ONLY place a route id becomes code.
    BUILDERS = {
        "ring":          lambda px, py: ring_blocks(px, py, low_mem=True),
        "gather":        lambda px, py: ring_blocks(px, py, low_mem=False),
        "stack_fused":   lambda px, py: stack_blocks(px, py, fuse=True),
        "stack_unfused": lambda px, py: stack_blocks(px, py, fuse=False),
    }

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(a))

    def sym(m):
        return float(np.linalg.norm(m - m.T) / np.linalg.norm(m))

    def score(ids):
        """THE PARAMETRIZED RUNNER.  Green arms and red arms both come through
        here, so a twin is caught by the same code that passes the shipped
        tree -- not by a bespoke assertion written to catch it."""
        out, blocks = {}, {}
        for rid in ids:
            one, four = BUILDERS[rid](1, 1), BUILDERS[rid](2, 2)
            blocks[rid] = (one, four)
            out[rid] = {
                "A_cross":    rel(one["A"], four["A"]),
                "Bx_cross":   rel(one["B_x"], four["B_x"]),
                "BW_cross":   rel(one["B_W"], four["B_W"]),
                "KdB_cross":  rel(one["Kd_B"], four["Kd_B"]),
                "KdB_sym1x1": sym(one["Kd_B"]),
                "KdB_sym2x2": sym(four["Kd_B"]),
            }
        base = ids[0]
        for rid in ids[1:]:
            for label, i in (("1x1", 0), ("2x2", 1)):
                out.setdefault("cross_route", {})[f"{base}|{rid}|{label}"] = rel(
                    blocks[base][i]["Kd_B"], blocks[rid][i]["Kd_B"])
        return out

    res = {"green": score(list(COUPLING_ROUTES))}

    # --- RED ARM 1: instance 2's defect, back in, caught through score() -----
    # Each twin ships with a CONTROL route it must leave clean: that is what
    # makes the red evidence rather than noise -- the gate is pointed at the
    # coupling screened-direct term, not at the mesh in general.
    shipped = getattr(bsm, "_encode_T_B", None)
    if shipped is None:
        res["red_stack_unavailable"] = (
            "bse_stack_matvec._encode_T_B is gone, so the pre-port twin could "
            "not be installed")
    else:
        bsm._encode_T_B = _pre_fix_stack_encode
        try:
            res["red_stack"] = score(["stack_fused", "ring"])
        finally:
            bsm._encode_T_B = shipped

    # --- RED ARM 2: instance 1's defect, back in ----------------------------
    shipped_ring = getattr(brc, "_ring_sum_B_encode", None)
    if shipped_ring is None:
        res["red_ring_unavailable"] = (
            "bse_ring_comm._ring_sum_B_encode is gone, so the pre-fix twin "
            "could not be installed")
    else:
        brc._ring_sum_B_encode = _pre_fix_ring_encode
        try:
            res["red_ring"] = score(["ring", "stack_fused"])
        finally:
            brc._ring_sum_B_encode = shipped_ring

    print("RESULT " + json.dumps(res))
    return 0


@lru_cache(maxsize=1)
def _results() -> dict:
    """Run the worker ONCE per session; every test below reads this."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={_NDEV}").strip()
    # THIS GATE'S OWN OPT-OUT (P19, 2026-08-09).  The gate makes no FFI
    # requirement, but what it calls does: build_bse_ring_matvec_full ->
    # contract_bands_block_reshard -> gate.require REFUSES on a box with no
    # mklblas host handler unless the dial announces the debug opt-out.
    # Until 2026-08-09 the child inherited a "0" that leaked out of
    # tests/test_contract_bands.py's module scope, so this gate was 15
    # passed in the default census and 13 failed / 2 passed standalone on
    # an FFI-less box -- a verdict that depended on what else pytest
    # collected, and that read exactly like a K^d_B regression.  Declaring
    # it here makes the child's environment the SAME in every arrangement:
    # the subject of this gate is mesh invariance, not the GEMM plan, and
    # pinning the XLA lowering is what makes the number comparable across
    # an FFI box and an FFI-less one.  gate.require itself is untouched.
    env["LORRAX_BANDS_GEMM_FFI"] = "0"
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "worker"],
                          env=env, capture_output=True, text=True, timeout=1800)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
    assert proc.returncode == 0, f"coupling-route worker failed:\n{tail}"
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, f"worker printed no RESULT:\n{tail}"
    return json.loads(line[-1][len("RESULT "):])


def _green(route: str) -> dict:
    res = _results()
    if "skip" in res:
        pytest.skip(res["skip"])
    return res["green"][route]


# ===========================================================================
#  THE GATES -- parametrized over the registry
# ===========================================================================
@pytest.mark.parametrize("route", sorted(COUPLING_ROUTES))
def test_coupling_route_is_mesh_invariant(route):
    """1x1 and 2x2 must produce the SAME coupling block, on every route.

    This is the assertion the class exists to defeat: the defect is bit-exact
    at P=1, so only a cross-mesh comparison can see it.  Touches no module
    internals -- a revert of either fix fails this on the NUMBER.
    """
    g = _green(route)
    for key in ("A_cross", "Bx_cross", "BW_cross", "KdB_cross"):
        assert g[key] < _TOL, (
            f"route {route!r}: {key} = {g[key]:.6e} exceeds {_TOL:.0e}.\n"
            f"{COUPLING_ROUTES[route]}\n"
            "The coupling block differs between a 1x1 and a 2x2 mesh. This is "
            "the 2026-08-08 zeta-transport class: find the collective in this "
            "route's encode that moves a PARTIALLY CONTRACTED tensor along an "
            "axis its own mu/nu shard lives on, and move the communication "
            "onto the trial vector instead.")


@pytest.mark.parametrize("route", sorted(COUPLING_ROUTES))
def test_coupling_route_stays_complex_symmetric(route):
    """K^d_B is complex SYMMETRIC (B = B^T) -- at 2x2 as much as at 1x1.

    Independent of the cross-mesh arm: this is a physical property of the
    screened-direct coupling term, and the fixture's W is built to carry the
    real-space reciprocity that makes it hold.  The defect breaks it outright
    (measured 1.19 against 6e-16).
    """
    g = _green(route)
    assert g["KdB_sym1x1"] < _TOL, (
        f"route {route!r}: K^d_B is not complex symmetric even at P=1 "
        f"({g['KdB_sym1x1']:.6e}) -- that is a kernel/convention defect, not a "
        "sharding one.")
    assert g["KdB_sym2x2"] < _TOL, (
        f"route {route!r}: K^d_B loses complex symmetry at 2x2 "
        f"({g['KdB_sym2x2']:.6e} vs {g['KdB_sym1x1']:.6e} at 1x1). "
        "A sharding defect, not a physics one.")


@pytest.mark.parametrize("route", sorted(set(COUPLING_ROUTES) - {"ring"}))
def test_coupling_routes_agree_with_each_other(route):
    """Every route must reproduce every other route's block, at BOTH meshes.

    This is the arm that caught instance 2.  Agreement at 1x1 with divergence
    at 2x2 is the class's fingerprint: the routes share a convention, so the
    divergence can only be the sharding.
    """
    res = _results()
    if "skip" in res:
        pytest.skip(res["skip"])
    cr = res["green"]["cross_route"]
    for label in ("1x1", "2x2"):
        key = f"ring|{route}|{label}"
        assert cr[key] < _TOL, (
            f"route {route!r} disagrees with 'ring' at {label}: "
            f"{cr[key]:.6e}.\n"
            f"  ring|{route}|1x1 = {cr[f'ring|{route}|1x1']:.6e}\n"
            f"  ring|{route}|2x2 = {cr[f'ring|{route}|2x2']:.6e}\n"
            "Clean at 1x1 and dirty at 2x2 means the two paths share a "
            "convention and differ only in how they shard -- i.e. one of them "
            "transports a zeta-carrying partial.")


# ===========================================================================
#  THE FALSE CASES -- both instances, caught through the parametrized runner
# ===========================================================================
def test_the_pre_port_stack_encode_is_caught():
    """FALSE CASE, instance 2: re-introduce the ported defect; the runner must
    catch it, and must leave the ring route and the non-coupling blocks clean.

    The specificity is the evidence.  A twin that reddened everything would
    only prove the harness notices a changed number."""
    res = _results()
    if "skip" in res:
        pytest.skip(res["skip"])
    assert "red_stack_unavailable" not in res, res["red_stack_unavailable"]
    red = res["red_stack"]

    bad = red["stack_fused"]
    assert bad["KdB_cross"] > _RED_MIN, (
        "the pre-port stack encode was re-introduced and the cross-mesh gate "
        f"did NOT catch it (K^d_B cross-mesh {bad['KdB_cross']:.6e}). The gate "
        "is not wired to the thing it claims to guard.")
    assert bad["KdB_sym2x2"] > _RED_MIN, (
        "the pre-port stack encode did not break complex symmetry at 2x2 "
        f"({bad['KdB_sym2x2']:.6e}) -- the symmetry arm is not wired.")

    # ... and it is SPECIFIC: P=1 untouched, resonant + exchange untouched,
    # and the control route untouched.
    assert bad["KdB_sym1x1"] < _TOL, (
        f"the twin moved P=1 ({bad['KdB_sym1x1']:.6e}); the defect is a P>1 "
        "transport error and must leave P=1 bit-exact.")
    assert bad["A_cross"] < _TOL, "the twin dirtied the RESONANT block"
    assert bad["Bx_cross"] < _TOL, "the twin dirtied the coupling EXCHANGE"
    ctrl = red["ring"]
    assert ctrl["KdB_cross"] < _TOL and ctrl["KdB_sym2x2"] < _TOL, (
        f"the stack twin dirtied the CONTROL ring route "
        f"({ctrl['KdB_cross']:.6e}) -- the two routes are supposed to be "
        "independent, so this twin is not localised.")
    assert red["cross_route"]["stack_fused|ring|1x1"] < _TOL, (
        "the twin moved the 1x1 cross-route agreement; it should only diverge "
        "at 2x2.")
    assert red["cross_route"]["stack_fused|ring|2x2"] > _RED_MIN, (
        "the twin did not make the routes disagree at 2x2 -- that divergence "
        "is exactly what caught this defect on 2026-08-08.")


def test_the_pre_fix_ring_encode_is_caught():
    """FALSE CASE, instance 1: the original kdb defect, caught through the same
    runner, leaving the stack route clean."""
    res = _results()
    if "skip" in res:
        pytest.skip(res["skip"])
    assert "red_ring_unavailable" not in res, res["red_ring_unavailable"]
    red = res["red_ring"]

    bad = red["ring"]
    assert bad["KdB_cross"] > _RED_MIN, (
        "the pre-fix ring encode was re-introduced and the cross-mesh gate did "
        f"NOT catch it ({bad['KdB_cross']:.6e}).")
    assert bad["KdB_sym2x2"] > _RED_MIN, (
        f"the pre-fix ring encode kept complex symmetry at 2x2 "
        f"({bad['KdB_sym2x2']:.6e}) -- the symmetry arm is not wired.")
    assert bad["KdB_sym1x1"] < _TOL, "the twin moved P=1"
    assert bad["A_cross"] < _TOL, "the twin dirtied the RESONANT block"
    assert bad["Bx_cross"] < _TOL, "the twin dirtied the coupling EXCHANGE"
    ctrl = red["stack_fused"]
    assert ctrl["KdB_cross"] < _TOL and ctrl["KdB_sym2x2"] < _TOL, (
        f"the ring twin dirtied the CONTROL stack route ({ctrl['KdB_cross']:.6e})")


def test_the_registry_scan_catches_an_unregistered_encode(tmp_path):
    """FALSE CASE for the DISCOVERY half: a new coupling encode with a
    collective, dropped into a scratch tree, must be found.

    Without this, ``test_route_registry_covers_every_coupling_encode`` could
    pass forever by simply never finding anything -- the failure mode that
    makes a coverage gate worthless."""
    global _SRC
    pkg = tmp_path / "bse"
    pkg.mkdir()
    (pkg / "brand_new_route.py").write_text(
        "from jax import lax\n"
        "import jax.numpy as jnp\n"
        "def _encode_T_B_v2(X, psi_c_Y, psi_v_X):\n"
        "    R = jnp.einsum('kcsN,cvk->vksN', jnp.conj(psi_c_Y), X)\n"
        "    Rv = lax.all_gather(R, 'y', axis=0, tiled=True)\n"
        "    return jnp.einsum('kvtM,vksN->MNtsk', psi_v_X, Rv)\n")
    saved = _SRC
    try:
        _SRC = tmp_path
        found = _discover_coupling_encodes()
        assert "bse/brand_new_route.py::_encode_T_B_v2" in found, (
            "the discovery scan MISSED a new coupling encode carrying the "
            "crossed pairing and an all_gather. The registry-completeness "
            f"gate is blind. found={sorted(found)}")
        with pytest.raises(AssertionError, match="NEW COUPLING ENCODE"):
            test_route_registry_covers_every_coupling_encode()
    finally:
        _SRC = saved

    # and the scan must NOT fire on the resonant (uncrossed) pairing
    (pkg / "brand_new_route.py").write_text(
        "from jax import lax\n"
        "import jax.numpy as jnp\n"
        "def _encode_T_A_v2(X, psi_c_X, psi_v_Y):\n"
        "    R = jnp.einsum('kvsN,cvk->cksN', jnp.conj(psi_v_Y), X)\n"
        "    Rc = lax.all_gather(R, 'x', axis=0, tiled=True)\n"
        "    return jnp.einsum('kctM,cksN->MNtsk', psi_c_X, Rc)\n")
    try:
        _SRC = tmp_path
        assert not _discover_coupling_encodes(), (
            "the scan fired on a RESONANT encode (uncrossed pairing); it would "
            "cry wolf on every A-block route added from now on.")
    finally:
        _SRC = saved


if __name__ == "__main__":
    sys.exit(_worker())
