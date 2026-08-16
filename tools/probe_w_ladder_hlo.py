"""HLO probe: prove the LADDER W path (``screening_diagrams = w_bse``) compiles
with NO gather-class collective on an N_mu^2-class operand.

WHAT THIS PROVES (that no AST gate and no unit test can).  ``w_ladder.py``'s
module docstring states a scaling envelope — "memory high-water per rank is the
pair-basis probe block ``X_full`` plus ONE ``P('x','y')`` output tile; there is
no whole-``mu^2`` per-rank object anywhere on the path".  That is a property of
the COMPILED program, not of the Python (QUALITY_PATTERNS #4; sandbox
``docs/HLO_HOWTO.md``).  This probe lowers the two pieces the ladder actually
adds or changes —

  1. the LADDER ring matvec ``build_bse_ring_matvec_full(screening=True,
     include_W=True)``: FOUR screened-direct rung applications per matvec
     (``W_d X``, ``W_d^B Y``, and the CONJUGATED pair in the anti-resonant row),
     each convolving the ``mu^2``-class ``W_R`` kernel against the
     ``mu^2``-class ``T`` tensor.  The RPA matvec is lowered beside it so the
     ladder's collective DELTA is measured, not asserted;
  2. the resolvent PROJECT stage ``build_density_snapshot_operator(
     scatter_nu_on_y=True)`` — the ``psum_scatter`` that must land the tile
     ``W(mu_X, nu_Y) = P('x','y')`` with no replicated ``(mu, nu)``;

— on a 2x2 ('x','y') mesh of 4 host devices with the production
``PartitionSpec``s, and scans the OPTIMIZED HLO of each.

GATE ON OPERAND CLASS, NOT ON BARE OPCODE (HLO_HOWTO's scoped ``--forbid``
rule).  Shard-sized gather traffic is legal, and a volume-preserving reshard
legitimately emits some; what the scaling doctrine forbids is a rank coming to
hold a full mu-square tile.  The synthetic payload therefore uses an ``n_mu``
that appears at NO other extent in the program (24, local 12, vs nc=nv=2, nk=2,
n_probe=4), so "this collective restored a FULL mu axis on some rank" is a
decidable property of the printed post-SPMD result shape.

RED TWIN (mandatory — ``ASSERTIONS.md`` records that the analyzer's ``--forbid``
path has never been observed to go red, so a green scan from an unexercised
detector proves nothing).  Arm ``red-twin`` lowers a deliberately
replicated-``W_q`` variant: the same ``mu^2``-class kernel tile, ``out_shardings``
replicated, which forces XLA to emit exactly the all-gather this probe exists to
forbid.  The probe FAILS if the scanner does not flag it.  Exit 0 therefore
means both "no mu^2-class gather in the ladder programs" AND "the detector
demonstrably fires when there is one".

Arms (all run by default; ``--arms`` selects a subset):

  hlo       lower + scan the ladder matvec, the RPA matvec, and the snapshot
  numerics  sharded ladder matvec, and the assembled W tile from the full
            seed/solve/project resolvent, vs a DENSE eager reference built from
            psi/eps/V/W by explicit numpy einsums (no production kernel)
  red-twin  the reachability proof above
  sweep     retrace / dispatch audit of the q x z sweep: drives
            ``w_ladder.sweep_q_wedge`` over 2 q-points x 2 z-values with
            ``jax_explain_cache_misses`` on, and asserts no tracing-cache miss
            after the first (q, z) compile (DESIGN_2026-08-15 section 7: a
            per-q or per-z retrace is a defect)

USAGE (compute node — Perlmutter login nodes are edit/inspect only; the probe
runs on CPU HOST devices, so one GPU-partition step at the default ``-G 1`` is
enough and ``JAX_PLATFORMS=cpu`` keeps it off the card):

    XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_dump_to=$PWD/xla_dump" \
        python3 tools/probe_w_ladder_hlo.py --out <evidence-dir>

Cache-cold is MANDATORY and is FORCED here rather than defaulted (INVARIANTS
row 4): an inherited warm ``ISDF_JAX_CACHE_DIR`` silently under-reports,
because a cache-hit module never re-dumps its HLO.  Exit 0 = verdict PASS.
"""
import argparse
import contextlib
import io
import json
import logging
import os
import re
import sys
import time

# Cache-cold is a precondition of every table below, so it is FORCED
# (INVARIANTS row 4), not setdefault'ed.
#
# ``ISDF_JAX_CACHE_DIR=""`` IS NOT SUFFICIENT ON PERLMUTTER, and this probe is
# the measurement that showed it.  That variable is read only by
# ``common.jax_compile_cache``, which arms the cache from
# ``runtime.initialize_communicator_stack``; the JAX-NATIVE
# ``JAX_COMPILATION_CACHE_DIR`` is independent, and the ``lx`` site environment
# exports it (``/pscratch/sd/j/jackm/.jax_cache``, 168 MB, shared between
# agents).  With only the ISDF opt-out set, a second run of this probe served
# the ladder ``jit(_block)`` SOLVE from that cache and the module NEVER
# RE-DUMPED — exactly the silent under-report INVARIANTS row 4 exists to
# prevent.  So kill both, before ``import jax`` and again on the config after.
os.environ["ISDF_JAX_CACHE_DIR"] = ""
os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
_DEVCOUNT = "--xla_force_host_platform_device_count=4"
if "force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _DEVCOUNT).strip()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np                                             # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)   # see the note above
jax.config.update("jax_compilation_cache_dir", None)

# Collective classes.  GATHER is the class by which a rank can come to hold data
# it did not own; PERMUTE (the ring rotations) and REDUCE (psum / psum_scatter)
# are volume-preserving or reducing and are counted, never forbidden
# (HLO_HOWTO's grep table).
_CLASS = {"all-gather": "gather", "all-to-all": "gather",
          "collective-broadcast": "gather", "collective-permute": "permute",
          "reduce-scatter": "reduce", "all-reduce": "reduce"}
_RE_OP = re.compile(r"\b(" + "|".join(_CLASS) + r")(?:-start|-done)?\(")
_RE_SHAPE = re.compile(r"\[([\d,]+)\]")

# Payload geometry.  n_mu is deliberately unique among the extents, so a full-mu
# axis in a post-SPMD shape is unambiguous evidence of a gathered mu axis.
N_MU, NC, NV, NKX, NKY, NKZ, N_PROBE = 24, 2, 2, 2, 1, 1, 4


# ---------------------------------------------------------------------------
# HLO scan
# ---------------------------------------------------------------------------
def scan(hlo: str, n_mu: int = N_MU) -> dict:
    """Classify every collective in an optimized-HLO text.

    A gather-class op is a FAILURE iff its result shape carries the FULL logical
    ``n_mu`` extent on some axis: that is the only way a rank starts holding
    mu^2-class data.  Everything else is shard-sized traffic — reported with its
    shapes, never forbidden.
    """
    rows = []
    for ln in hlo.splitlines():
        s = ln.strip()
        m = _RE_OP.search(s)
        if m is None or "=" not in s[:m.start()]:
            continue
        head = s[:m.start()]
        shapes = [tuple(int(d) for d in g.split(",")) for g in _RE_SHAPE.findall(head)]
        full_mu = max((sum(1 for d in sh if d == n_mu) for sh in shapes), default=0)
        cls = _CLASS[m.group(1)]
        rows.append({"op": m.group(1), "cls": cls, "shapes": shapes,
                     "full_mu_axes": full_mu,
                     "verdict": "MU2-GATHER" if (cls == "gather" and full_mu) else "ok",
                     "line": s[:400]})
    return {"rows": rows,
            "bad": [r for r in rows if r["verdict"] == "MU2-GATHER"],
            "counts": {c: sum(1 for r in rows if r["cls"] == c)
                       for c in ("gather", "permute", "reduce")}}


def report(name: str, res: dict, expect_red: bool = False) -> bool:
    c = res["counts"]
    print(f"[{name}] collectives: gather-class {c['gather']}, permute-class "
          f"{c['permute']}, reduce-class {c['reduce']}  |  gather-class on an "
          f"N_mu^2-class operand: {len(res['bad'])}"
          f"{'  (RED EXPECTED)' if expect_red else ''}")
    agg = {}
    for r in res["rows"]:
        k = (r["verdict"], r["op"], tuple(r["shapes"]))
        agg[k] = agg.get(k, 0) + 1
    for (verdict, op, shapes), n in sorted(agg.items()):
        print(f"    {verdict:<11} x{n:<3} {op:<20} {list(shapes)}")
    for r in res["bad"]:
        print(f"      -> {r['line']}")
    return len(res["bad"]) == 0


# ---------------------------------------------------------------------------
# Payload — restart-FREE, lifted from
# tests/test_bse_w_ladder_dense.py::_synthetic_payload (same key contract, same
# physical structure: V real symmetric, W Hermitian per q and W(-q)=conj(W(q)))
# and extended with V_q_full, which the sweep arm's build_finite_q_data needs.
# ---------------------------------------------------------------------------
def synthetic_payload(mesh, *, nkx=NKX, nky=NKY, nkz=NKZ, nc=NC, nv=NV,
                      nmu=N_MU, seed=7):
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude

    nk = nkx * nky * nkz
    rng = np.random.default_rng(seed)
    sh = make_bse_shardings(mesh)

    def _c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / 8.0

    def _herm_qgrid(scale):
        A = _c(nmu, nmu, nk) * scale
        A = 0.5 * (A + np.conj(np.transpose(A, (1, 0, 2))))          # Hermitian per q
        idx = np.ravel_multi_index(
            tuple((-np.array(np.unravel_index(np.arange(nk), (nkx, nky, nkz))))
                  % np.array([[nkx], [nky], [nkz]])), (nkx, nky, nkz))
        A = 0.5 * (A + np.conj(A[:, :, idx]))                        # X(-q)=conj(X(q))
        return A.reshape(nmu, nmu, nkx, nky, nkz)

    psi_c, psi_v = _c(nk, nc, 1, nmu), _c(nk, nv, 1, nmu)
    eps_c = 1.0 + 0.3 * rng.standard_normal((nk, nc))
    eps_v = -1.0 + 0.3 * rng.standard_normal((nk, nv))
    Vq0 = rng.standard_normal((nmu, nmu)) * 0.1
    Vq0 = Vq0 + Vq0.T                                                # real symmetric
    Wq, Vfull = _herm_qgrid(0.1), _herm_qgrid(0.1)
    Vfull[:, :, 0, 0, 0] = Vq0

    with mesh:
        d = {
            "psi_c_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_x),
            "psi_c_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_y),
            "psi_v_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_x),
            "psi_v_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_y),
            "eps_c": jnp.asarray(eps_c), "eps_v": jnp.asarray(eps_v),
            "W_q": jax.lax.with_sharding_constraint(jnp.asarray(Wq), sh.W),
            "V_q0": jax.lax.with_sharding_constraint(jnp.asarray(Vq0), sh.V),
            "V_q_full": jax.lax.with_sharding_constraint(jnp.asarray(Vfull), sh.W),
            "nkx": nkx, "nky": nky, "nkz": nkz,
            "n_cond_pad": nc, "n_val_pad": nv, "n_rmu": nmu,
        }
        d["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_X"], d["psi_v_X"]), sh.psi_x)
        d["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_Y"], d["psi_v_Y"]), sh.psi_y)
    return d


# ---------------------------------------------------------------------------
# Eager reference — explicit numpy einsums, no production kernel (the dense
# oracle of tests/test_bse_w_ladder_dense.py, trimmed to what this probe scores)
# ---------------------------------------------------------------------------
def _host(x):
    return np.asarray(jax.device_get(x))


def dense_H(p, ladder=True):
    """``H`` (2N x 2N) for the ladder (or, ``ladder=False``, RPA) operator.

    ``ladder=False`` drops both direct rungs and reproduces the symplectic RPA
    density-response operator ``[[D+Kx, Kx], [-Kx, -(D+Kx)]]`` — the twin that
    makes the ``include_w=False`` delegation in ``build_ladder_resolvent`` a
    measured claim rather than a code-reading one.
    """
    psi_c, psi_v = _host(p["psi_c_X"]), _host(p["psi_v_X"])
    eps_c, eps_v = _host(p["eps_c"]), _host(p["eps_v"])
    V, W = _host(p["V_q0"]), _host(p["W_q"])
    grid = (int(p["nkx"]), int(p["nky"]), int(p["nkz"]))
    nk = grid[0] * grid[1] * grid[2]
    nc, nv, nmu = psi_c.shape[1], psi_v.shape[1], psi_c.shape[3]
    N = nc * nv * nk

    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)
    D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    Kx = np.einsum("kcvN,KCVN->cvkCVK",
                   np.einsum("kcvM,MN->kcvN", M, V), np.conj(M)).reshape(N, N) / nk

    Wf = W.reshape(nmu, nmu, nk)
    Kd = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
    KdB = np.zeros_like(Kd)
    for k in range(nk if ladder else 0):
        for kp in range(nk):
            dq = (np.array(np.unravel_index(k, grid))
                  - np.array(np.unravel_index(kp, grid))) % np.array(grid)
            Wq = Wf[:, :, int(np.ravel_multi_index(tuple(dq), grid))]
            Pc = np.einsum("ctm,Ctm->cCm", np.conj(psi_c[k]), psi_c[kp])
            Pv = np.einsum("vsn,Vsn->vVn", psi_v[k], np.conj(psi_v[kp]))
            Kd[:, :, k, :, :, kp] = np.einsum("cCm,mn,vVn->cvCV", Pc, Wq, Pv) / nk
            PcB = np.einsum("ctm,Vtm->cVm", np.conj(psi_c[k]), psi_v[kp])
            PvB = np.einsum("vsn,Csn->vCn", psi_v[k], np.conj(psi_c[kp]))
            KdB[:, :, k, :, :, kp] = np.einsum("cVm,mn,vCn->cvCV", PcB, Wq, PvB) / nk

    Kd, KdB = Kd.reshape(N, N), KdB.reshape(N, N)
    Dm = np.diag(D.reshape(-1).astype(np.complex128))
    # A = D + Kx - Kd, B = Kx - Kd_B; anti-resonant row: ring un-conjugated,
    # direct conjugated (w_ladder module docstring, derivation step 4).
    H = np.block([[Dm + Kx - Kd, Kx - KdB],
                  [-(Kx - np.conj(KdB)), -(Dm + Kx - np.conj(Kd))]])
    return H, np.transpose(M, (1, 2, 0, 3)).reshape(N, nmu), N, V, nk


def dense_wc_columns(p, cols, z, ladder=True):
    """``W(z) - v`` columns from the dense 2N solve (seed [f;-f], readout X+Y)."""
    H, Mmat, N, V, nk = dense_H(p, ladder=ladder)
    snk = np.sqrt(float(nk))
    lhs = z * np.eye(2 * N, dtype=np.complex128) - H
    out = np.zeros((V.shape[0], len(cols)), dtype=np.complex128)
    for i, nu0 in enumerate(cols):
        g = np.zeros(V.shape[0])
        g[int(nu0)] = 1.0
        f = (Mmat @ (V @ g)) / snk
        x = np.linalg.solve(lhs, np.concatenate([f, -f]))
        out[:, i] = (V @ (np.conj(Mmat).T @ (x[:N] + x[N:]))) / snk
    return out


def relerr(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def arm_hlo(mesh, data, do_numerics, texts):
    from bse.bse_feast import matvec_operands, ladder_matvec_operands
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_w_exact import apply_screening_resolvent_block
    from bse.w_ladder import build_ladder_resolvent

    sh = make_bse_shardings(mesh)
    nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
    ok = True
    rng = np.random.default_rng(3)
    xf = (rng.standard_normal((2, 1, NC, NV, nk))
          + 1j * rng.standard_normal((2, 1, NC, NV, nk)))
    X_full = jax.device_put(jnp.asarray(xf), sh.X_full)

    for tag, include_w in (("matvec-RPA", False), ("matvec-LADDER", True)):
        mv, diag_h, gen, snap, _ = build_ladder_resolvent(mesh, data, include_w=include_w)
        # The LADDER engine is built with ``ladder_rung_slots=True`` (landed
        # 2026-08-16): its rung takes four EXTRA operands, the rolled UN-flipped
        # psi arrays, which ``ladder_matvec_operands`` supplies and which alias
        # the density arrays on a raw payload like this one.  Feeding the
        # 11-tuple to the 15-operand matvec is a pjit shape refusal, not a wrong
        # answer -- and it is what this probe hit on 2026-08-16.
        ops_fn = ladder_matvec_operands if include_w else matvec_operands
        ops = ops_fn(data)
        texts[tag] = mv.lower(X_full, *ops).compile().as_text()
        ok &= report(tag, scan(texts[tag]))

        if do_numerics:
            H, _, _, _, _ = dense_H(data, ladder=include_w)
            v = xf.reshape(2, -1)
            ref = (H @ np.concatenate([v[0], v[1]])).reshape(2, -1)
            err = relerr(_host(mv(X_full, *ops)).reshape(2, -1), ref)
            print(f"[{tag}] rel err vs DENSE eager H.x: {err:.3e}")
            ok &= err < 1e-11

        if include_w:
            # PROJECT / snapshot: the reduce-scatter that must land P('x','y').
            s_all = jax.device_put(
                jnp.zeros((N_PROBE, NC, NV, nk), dtype=jnp.complex128), sh.X)
            texts["snapshot-PROJECT"] = snap.lower(
                s_all, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]).compile().as_text()
            ok &= report("snapshot-PROJECT", scan(texts["snapshot-PROJECT"]))

        # Run the FULL seed/solve/project resolvent for BOTH operators.  It is
        # what puts the two shifted block-GMRES SOLVE programs (jit(_block)) in
        # the XLA dump, where the whole-dump scan below measures the ladder's
        # collective delta on the SOLVE and not only on the matvec.
        cols = np.array([0, 5, 11, 17], dtype=int)
        G = np.zeros((N_PROBE, N_MU))
        G[np.arange(len(cols)), cols] = 1.0
        tile, resids = apply_screening_resolvent_block(
            G, 0.0 + 0.0j, data, mv, diag_h, gen, snap, sh,
            max_iter=200, tol=1e-13, operands_fn=ops_fn)
        spec_ok = tile.sharding.spec == P("x", "y")
        line = (f"[resolvent {tag[7:]}] tile sharding {tile.sharding.spec} "
                f"(P('x','y') required: {spec_ok}); max GMRES resid "
                f"{float(np.max(_host(resids))):.2e}")
        if do_numerics:
            err = relerr(_host(tile)[:, :len(cols)],
                         dense_wc_columns(data, cols, 0.0 + 0.0j, ladder=include_w))
            line += f"; rel err vs dense (z-H)^-1: {err:.3e}"
            ok &= err < 1e-8
        print(line)
        ok &= spec_ok
    return ok


def arm_assembly(mesh, texts, dump_dir):
    """The wedge ASSEMBLY tail of ``compute_wc_qwedge``, on synthetic tiles.

    ``compute_wc_qwedge`` itself needs a restart AND the input deck (``SymMaps``
    is built from the WFN the deck names), so it cannot run restart-free.  Its
    tail can: :func:`w_ladder._accumulate_columns` (the ``dynamic_update_slice``
    that places a probe chunk into the mu^2 tile) and the stack + mu-pad strip +
    ``P(None,None,'x','y')`` constraint are pure array code over ``P('x','y')``
    tiles.  Both carry an explicit no-gather claim in their own comments, so both
    get lowered and scanned here.

    Each is measured against a CONTROL that differs only in shard alignment —
    one chunk vs two for the placement, ``nlog == n_pad`` vs ``nlog < n_pad`` for
    the strip.  A gather that appears only in the misaligned arm is attributable
    to the alignment and not to the operation.
    """
    from bse import w_ladder as wl
    n_pad, nz, nq = N_MU, 2, 2
    px, py = mesh.devices.shape
    tile_sh = NamedSharding(mesh, P("x", "y"))
    wedge_sh = NamedSharding(mesh, P(None, None, "x", "y"))
    rng = np.random.default_rng(11)
    ok = True

    def _tile(m, n):
        return jax.device_put(
            jnp.asarray(rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))),
            tile_sh if (m, n) == (n_pad, n_pad) else NamedSharding(mesh, P("x", "y")))

    # --- 1. the chunk placement, 1 chunk (aligned) vs 2 chunks -------------
    for nchunk in (1, 2):
        chunk = n_pad // nchunk
        tag = f"accumulate-{nchunk}chunk"
        f = jax.jit(
            lambda acc, upd, c0=0: wl._accumulate_columns(
                acc, upd, int(c0), chunk, n_pad, mesh),
            static_argnums=(2,),
            in_shardings=(tile_sh, tile_sh), out_shardings=tile_sh)
        acc0 = jax.device_put(jnp.zeros((n_pad, n_pad), dtype=jnp.complex128), tile_sh)
        upd = _tile(n_pad, chunk)
        texts[tag] = f.lower(acc0, upd, chunk).compile().as_text()
        ok &= report(tag, scan(texts[tag]))

    # --- 2. the mu-pad strip, aligned control vs the production shape ------
    def _strip(nlog):
        g = jax.jit(
            lambda s: jax.lax.with_sharding_constraint(
                s[:, :, :nlog, :nlog].astype(jnp.complex128), wedge_sh),
            in_shardings=(NamedSharding(mesh, P(None, None, "x", "y")),),
            out_shardings=wedge_sh)
        st = jax.device_put(
            jnp.zeros((nz, nq, n_pad, n_pad), dtype=jnp.complex128),
            NamedSharding(mesh, P(None, None, "x", "y")))
        return g.lower(st).compile().as_text()

    for nlog, why in ((n_pad, "aligned control, nlog == n_pad"),
                      (N_MU - 2, "production shape, nlog < n_pad")):
        tag = f"mu-pad-strip-nlog{nlog}"
        texts[tag] = _strip(nlog)
        print(f"  ({why}; n_pad={n_pad}, mesh {px}x{py})")
        ok &= report(tag, scan(texts[tag]))

    # --- 3. the EAGER call pattern -----------------------------------------
    # `compute_wc_qwedge` calls both of these EAGERLY (`_on_result` per chunk,
    # then the stack/strip at the end), and eager dispatch is a different
    # compilation unit from the jitted lowerings above: each op becomes its own
    # tiny program whose output sharding XLA picks without seeing the surrounding
    # constraint.  So the jitted scans above do NOT settle the production
    # question; this sub-arm runs the eager pattern and attributes every module
    # that appears in the dump while it runs.
    seen = _dump_modules(dump_dir)
    results = {}
    for nchunk in (1, 2):
        chunk = n_pad // nchunk
        acc = None
        for c0 in range(0, n_pad, chunk):
            acc = wl._accumulate_columns(acc, _tile(n_pad, chunk), c0, chunk,
                                         n_pad, mesh)
        jax.block_until_ready(acc)
        results[f"eager accumulate, {nchunk} chunk(s)"], seen = _scan_new(dump_dir, seen)
        if nchunk == 2:
            tiles = [acc] * (nz * nq)

    # --- the WEDGE RETURN, as production now spells it ---------------------
    # `compute_wc_qwedge` returns the stacked wedge on the PADDED mu extent
    # (`WLadderWedge.wc`, contract amended 2026-08-16): a pad-STRIPPED tile
    # cannot carry P(None,None,'x','y') for an n_rmu that does not divide the
    # mesh, and the eager strip that used to sit here all-gathered both mu
    # axes.  Required green.
    stacked = jnp.stack([jnp.stack(tiles[i * nq:(i + 1) * nq], axis=0)
                         for i in range(nz)], axis=0)
    wc = jax.lax.with_sharding_constraint(
        stacked.astype(jnp.complex128), wedge_sh)
    jax.block_until_ready(wc)
    results[f"eager stack, padded return (n_pad={n_pad})"], seen = \
        _scan_new(dump_dir, seen)

    # --- and the pattern it replaced, kept as a RED CONTROL ----------------
    # Not dead code and not a regression: it is the reachability proof for the
    # scanner on THIS operand class, the same job `arm_red_twin` does for a
    # replicated out_sharding.  A run in which this stays green means the
    # detector could not have caught the defect it was written for, so it is
    # asserted RED and excluded from the arm's verdict.
    red_control = {}
    for nlog in (n_pad, N_MU - 2):
        wc_strip = jax.lax.with_sharding_constraint(
            stacked[:, :, :nlog, :nlog].astype(jnp.complex128),
            NamedSharding(mesh, P(None, None, "x", "y")))
        jax.block_until_ready(wc_strip)
        red_control[f"RED CONTROL eager stack+strip, nlog={nlog} of "
                    f"n_pad={n_pad}"], seen = _scan_new(dump_dir, seen)

    # The pad-zero producer guarantee: two eager non-shard-aligned slices
    # (``wc[:, :, nlog:, :]`` / ``wc[..., nlog:]``) followed by a max-reduce.
    # Its docstring says "nothing mu^2-class comes back to the HOST", which is
    # about the host; this measures the DEVICE side of the same statement.
    wc_full = jax.lax.with_sharding_constraint(
        jnp.stack([jnp.stack(tiles[i * nq:(i + 1) * nq], axis=0)
                   for i in range(nz)], axis=0).astype(jnp.complex128), wedge_sh)
    jax.block_until_ready(wc_full)
    _, seen = _scan_new(dump_dir, seen)
    for nlog in (n_pad, N_MU - 2):
        try:
            wl._assert_pad_block_is_zero(wc_full, nlog)
        except Exception as exc:                       # a firing gate is fine
            print(f"    (_assert_pad_block_is_zero(nlog={nlog}) raised: "
                  f"{type(exc).__name__} — expected on random tiles)")
        results[f"eager _assert_pad_block_is_zero, nlog={nlog}"], seen = \
            _scan_new(dump_dir, seen)
    for what, (nmod, ngath, nbad, lines) in results.items():
        verdict = "MU2 GATHER" if nbad else "clean"
        print(f"[assembly/eager] {what:<44} {nmod:>2} new module(s), "
              f"{ngath} gather-class, {nbad} mu^2-class  -> {verdict}")
        for ln in lines:
            print(f"        {ln}")
        ok &= nbad == 0

    # The red control, reported and REQUIRED red.  Its module names go back to
    # the caller so the whole-dump arm can exclude them the way it already
    # excludes `arm_red_twin`'s.
    red_modules = set()
    for what, (nmod, ngath, nbad, lines) in red_control.items():
        print(f"[assembly/red ] {what:<44} {nmod:>2} new module(s), "
              f"{ngath} gather-class, {nbad} mu^2-class  -> "
              f"{'MU2 GATHER (expected)' if nbad else 'CLEAN — DETECTOR DEAD'}")
        for ln in lines:
            print(f"        {ln}")
            red_modules.add(ln.split(":", 1)[0])
    fired = any(nbad for (_n, _g, nbad, _l) in red_control.values())
    print(f"[assembly/red ] detector fired on the removed strip pattern: "
          f"{fired} — "
          f"{'the scan can see this operand class' if fired else 'THE GATE IS DEAD'}")
    ok &= fired

    spec_ok = (acc.sharding.spec == P("x", "y")
               and wc.sharding.spec == P(None, None, "x", "y"))
    print(f"[assembly] declared OUTPUT shardings {'hold' if spec_ok else 'DO NOT hold'} "
          f"(accumulator {acc.sharding.spec}, wedge {wc.sharding.spec}) — the "
          f"scans above are about what happens in between")
    return bool(ok and spec_ok), red_modules


def _dump_dir(args):
    """``--dump-dir``, else the ``--xla_dump_to`` XLA already has."""
    if args.dump_dir:
        return args.dump_dir
    m = re.search(r"--xla_dump_to=(\S+)", os.environ.get("XLA_FLAGS", ""))
    return m.group(1) if m else None


def _dump_modules(dump_dir):
    import glob
    return set(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt"))) if dump_dir else set()


def _scan_new(dump_dir, seen):
    """Scan the modules that appeared in ``dump_dir`` since ``seen`` was taken."""
    now = _dump_modules(dump_dir)
    new = now - seen
    ngath = nbad = 0
    lines = []
    for f in sorted(new):
        with open(f) as fh:
            res = scan(fh.read())
        ngath += res["counts"]["gather"]
        nbad += len(res["bad"])
        lines += [f"{os.path.basename(f).split('.cpu_after')[0]}: {r['line']}"
                  for r in res["bad"]]
    return (len(new), ngath, nbad, lines), now


def arm_dump_scan(dump_dir: str, red_twin_names=("jit__lambda",)) -> bool:
    """Scan EVERY optimized-HLO module in the XLA dump, not only the two lowered
    by hand — the whole-path statement.

    ``red_twin_names`` are the deliberately-red control module(s); they are
    reported separately and excluded from the verdict, since their whole job is
    to be a mu^2 gather.
    """
    import glob
    files = sorted(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")))
    if not files:
        print(f"[dump] no optimized-HLO files in {dump_dir} — was --xla_dump_to set?")
        return False
    tot = {"gather": 0, "permute": 0, "reduce": 0}
    offenders, rows = [], []
    for f in files:
        name = os.path.basename(f).split(".cpu_after")[0]
        with open(f) as fh:
            res = scan(fh.read())
        for k in tot:
            tot[k] += res["counts"][k]
        rows.append((name, res["counts"], len(res["bad"])))
        if res["bad"] and not any(t in name for t in red_twin_names):
            offenders.append((name, res["bad"]))
    print(f"[dump] {len(files)} optimized modules in {dump_dir}")
    print(f"[dump] totals: gather-class {tot['gather']}, permute-class "
          f"{tot['permute']}, reduce-class {tot['reduce']}")
    for name, c, nbad in rows:
        if sum(c.values()):
            flag = ("  <- RED TWIN (control)" if nbad and any(t in name for t in red_twin_names)
                    else ("  <- MU2 GATHER" if nbad else ""))
            print(f"    {name:<44} gather {c['gather']:>3}  permute "
                  f"{c['permute']:>3}  reduce {c['reduce']:>3}{flag}")
    for name, bad in offenders:
        for r in bad:
            print(f"    OFFENDER {name}: {r['line']}")
    return not offenders


def arm_red_twin(mesh, data, texts):
    """Reachability proof: the SAME mu^2-class kernel tile, replicated on exit.

    XLA has no choice but to all-gather it, so a scanner that stays green here
    is a scanner that cannot go red at all — the standing ASSERTIONS.md caveat
    about the analyzer's never-exercised ``--forbid`` path.
    """
    from bse.bse_ring_comm import make_bse_shardings
    sh = make_bse_shardings(mesh)
    f = jax.jit(lambda w: w * 2.0, in_shardings=(sh.W,),
                out_shardings=NamedSharding(mesh, P(None, None, None, None, None)))
    texts["red-twin"] = f.lower(data["W_q"]).compile().as_text()
    res = scan(texts["red-twin"])
    report("red-twin (W_q forced replicated)", res, expect_red=True)
    fired = len(res["bad"]) > 0
    print(f"[red-twin] detector fired: {fired} — "
          f"{'failure path is REACHABLE' if fired else 'THE GATE IS DEAD'}")
    return fired


def arm_sweep(mesh, data):
    """Retrace / dispatch audit of the q x z sweep (DESIGN_2026-08-15 section 7).

    Drives ``w_ladder.sweep_q_wedge`` — the production per-q loop that
    ``compute_wc_qwedge`` itself calls — over 2 q-points x 2 z-values on the
    synthetic payload, with ``jax_explain_cache_misses`` on.  Every tracing
    cache miss after the FIRST (q, z) point is a defect: all q/z-dependent
    tensors are supposed to flow as runtime args through ``matvec_operands``,
    with the solver cached on ``(id(matvec), max_iter, tol, dtype)``.
    """
    from bse import bse_w_exact as bwe
    from bse import w_ladder as wl

    marks, misses = [], []
    counts = {"gen": 0, "solve": 0, "snapshot": 0}

    class _Cap(logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            if "cache miss" in msg.lower() or "cache_miss" in msg.lower():
                misses.append((len(marks), msg))

    def _count(fn, key):
        def _f(*a, **k):
            if not any(isinstance(x, jax.core.Tracer) for x in a):
                counts[key] += 1
            return fn(*a, **k)
        return _f

    orig_build, orig_get = wl.build_ladder_resolvent, bwe._get_block_gmres_solver

    def _patched_build(*a, **k):
        mv, diag_h, gen, snap, sh = orig_build(*a, **k)
        return mv, diag_h, _count(gen, "gen"), _count(snap, "snapshot"), sh

    def _on_result(iq, q, iz, z, c0, n_real, tile, resids, its):
        jax.block_until_ready(tile)
        marks.append({"q": q, "z": z, "t": time.perf_counter(), "disp": dict(counts),
                      "spec": str(tile.sharding.spec),
                      "resid": float(np.max(_host(resids)))})

    q_list, z_list = [(0, 0, 0), (1, 0, 0)], [0.0 + 0.0j, 0.0 + 0.5j]
    G = np.zeros((N_PROBE, N_MU))
    G[np.arange(2), [0, 5]] = 1.0

    lg, cap, err_buf = logging.getLogger("jax"), _Cap(), io.StringIO()
    lg.setLevel(logging.DEBUG)
    lg.addHandler(cap)
    bwe._BLOCK_GMRES_CACHE.clear()
    jax.config.update("jax_explain_cache_misses", True)
    try:
        wl.build_ladder_resolvent = _patched_build
        bwe._get_block_gmres_solver = lambda *a, **k: _count(orig_get(*a, **k), "solve")
        t0 = time.perf_counter()
        with contextlib.redirect_stderr(err_buf):
            wl.sweep_q_wedge(data, mesh, q_list, z_list, include_w=True,
                             probe_blocks_for_q=lambda _i, _q: [(0, 2, G)],
                             gmres_tol=1e-12, gmres_max_iter=200,
                             on_result=_on_result)
    finally:
        wl.build_ladder_resolvent, bwe._get_block_gmres_solver = orig_build, orig_get
        jax.config.update("jax_explain_cache_misses", False)
        lg.removeHandler(cap)

    stderr_text = err_buf.getvalue()
    if stderr_text.strip():
        print("--- captured stderr during the sweep ---")
        print(stderr_text[:8000])
        print("--- end captured stderr ---")

    print(f"[sweep] {len(q_list)} q x {len(z_list)} z = {len(marks)} points "
          f"(one probe block each), driven through w_ladder.sweep_q_wedge")
    prev_t, prev = t0, {k: 0 for k in counts}
    for i, m in enumerate(marks):
        d = {k: m["disp"][k] - prev[k] for k in prev}
        print(f"    point {i} q={m['q']} z={m['z']}: wall {m['t'] - prev_t:8.3f} s  "
              f"dispatches {d}  tile {m['spec']}  max resid {m['resid']:.1e}")
        prev_t, prev = m["t"], m["disp"]

    late = [(i, msg) for i, msg in misses if i >= 1]
    n_cache = len(bwe._BLOCK_GMRES_CACHE)
    print(f"[sweep] block-GMRES engine cache entries after the sweep: {n_cache} "
          f"(1 required — the whole q x z sweep must share ONE executable)")
    print(f"[sweep] tracing-cache-miss explanations captured: {len(misses)} total, "
          f"{len(late)} AFTER the first (q, z) point")
    for i, msg in late:
        print(f"    RETRACE after point {i}: {msg[:800]}")
    return len(late) == 0 and n_cache == 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="ladder-W HLO / retrace probe")
    ap.add_argument("--arms", default="hlo,numerics,red-twin,sweep,assembly,dump")
    ap.add_argument("--out", default=None,
                    help="directory to deposit the compiled HLO texts + summary")
    ap.add_argument("--dump-dir", default=None,
                    help="XLA dump dir for the whole-path scan (default: read "
                         "--xla_dump_to out of XLA_FLAGS)")
    args = ap.parse_args()
    arms = {a.strip() for a in args.arms.split(",")}

    devs = jax.devices()
    if len(devs) < 4:
        print(f"REFUSING: need >= 4 devices for a 2x2 mesh, have {len(devs)}. "
              f"Set XLA_FLAGS={_DEVCOUNT}")
        return 1
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))
    print(f"[probe] {len(devs)} x {devs[0].platform} devices, mesh 2x2 ('x','y'); "
          f"n_mu={N_MU} (local {N_MU // 2}) nc={NC} nv={NV} "
          f"nk={NKX * NKY * NKZ} n_probe={N_PROBE}")
    print(f"[probe] cache-cold (INVARIANTS row 4): "
          f"ISDF_JAX_CACHE_DIR={os.environ['ISDF_JAX_CACHE_DIR']!r}, "
          f"JAX_COMPILATION_CACHE_DIR="
          f"{os.environ.get('JAX_COMPILATION_CACHE_DIR')!r}, "
          f"jax_enable_compilation_cache="
          f"{jax.config.jax_enable_compilation_cache}")
    print(f"[probe] XLA_FLAGS={os.environ['XLA_FLAGS']!r}")
    from common.contract_bands import bands_gemm_ffi_mode
    print(f"[probe] bands-gemm FFI gate: {bands_gemm_ffi_mode()}")

    data = synthetic_payload(mesh)
    texts, verdicts = {}, {}
    if arms & {"hlo", "numerics"}:
        verdicts["hlo/numerics"] = arm_hlo(mesh, data, "numerics" in arms, texts)
    if "red-twin" in arms:
        verdicts["red-twin"] = arm_red_twin(mesh, data, texts)
    if "sweep" in arms:
        verdicts["sweep"] = arm_sweep(mesh, data)
    if "assembly" in arms:
        verdicts["assembly"], _red_mods = arm_assembly(
            mesh, texts, _dump_dir(args))
    if "dump" in arms:
        dd = _dump_dir(args)
        if dd is None:
            print("[dump] skipped: no --xla_dump_to in XLA_FLAGS and no --dump-dir")
        else:
            # The deliberately-red controls are excluded from the whole-dump
            # verdict by NAME, the way arm_red_twin's already is: their job
            # is to be a mu^2 gather, and the arm would otherwise re-report
            # the proof that the detector works as a defect.
            verdicts["dump-whole-path"] = arm_dump_scan(
                dd, red_twin_names=("jit__lambda",) + tuple(sorted(_red_mods)))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for tag, txt in texts.items():
            with open(os.path.join(args.out, f"{tag}.after_optimizations.txt"), "w") as fh:
                fh.write(txt)
        with open(os.path.join(args.out, "probe_summary.json"), "w") as fh:
            json.dump({"geometry": {"n_mu": N_MU, "nc": NC, "nv": NV,
                                    "nk": NKX * NKY * NKZ, "n_probe": N_PROBE},
                       "verdicts": verdicts,
                       "scan": {t: scan(x)["counts"] for t, x in texts.items()},
                       "mu2_gathers": {t: len(scan(x)["bad"]) for t, x in texts.items()}},
                      fh, indent=2)
        print(f"[probe] HLO texts + probe_summary.json written to {args.out}")

    print("\n=== VERDICT ===")
    for k, v in verdicts.items():
        print(f"  {k:<14} {'PASS' if v else 'FAIL'}")
    ok = all(verdicts.values())
    print("OK: the ladder matvec and the PROJECT reduce-scatter carry no "
          "gather-class collective on an N_mu^2-class operand, the detector was "
          "shown to fire on one, and the q x z sweep is dispatch-only after the "
          "first compile."
          if ok else
          "FAIL: see the arm(s) marked FAIL above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
