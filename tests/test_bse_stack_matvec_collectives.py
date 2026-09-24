"""The stack matvec's W term runs NO collective per trial (survey_C §C1).

``bse.bse_stack_matvec`` gathers the trial block X once per block, builds
both encodes with the full c and v locally, decodes through this rank's
(μ_loc, ν_loc) into a (b, c, v, nk) partial, and completes both sums with ONE
reduce-scatter after the scan.  Two checks on a 2x2 mesh:

1. The compiled TDA matvec and the non-TDA pair applier carry ZERO
   collectives inside any while body (the trial scan) AND the same total
   collective count at NT and 2·NT trials — unroll-proof, since XLA:GPU
   may unroll or hoist a short scan and make a while-body census vacuous.
   Red twin: a carry-dependent per-trial psum must fail that criterion.
2. The TDA matvec (and the pair at s = ±1) on the 2x2 mesh equals the 1x1
   mesh result to 1e-12 relative — the sums are only reassociated.

The coupling block's mesh invariance and route agreement are gated by
``test_bse_coupling_routes_mesh_invariance.py``.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                        # noqa: E402
from jax.sharding import Mesh                                  # noqa: E402

jax.config.update("jax_enable_x64", True)

NKX, NKY, NKZ = 2, 2, 3
NK = NKX * NKY * NKZ
NC, NV, NS, NMU, NT = 4, 4, 1, 8, 4
RTOL = 1e-12
_COLL = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all",
         "collective-permute")


def _mesh(px, py):
    devs = jax.devices()
    if len(devs) < px * py:
        pytest.skip(f"needs {px * py} devices, have {len(devs)}")
    return Mesh(np.asarray(devs[:px * py]).reshape(px, py), ("x", "y"))


def _payload(px, py, ns=NS):
    import bse.bse_ring_comm as brc
    from bse.bse_serial import compute_pair_amplitude

    rng = np.random.default_rng(20260924)
    cx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)  # noqa: E731
    psi_c, psi_v = cx(NK, NC, ns, NMU), cx(NK, NV, ns, NMU)
    eps_c = np.sort(rng.random((NK, NC)), axis=1) + 1.0
    eps_v = np.sort(rng.random((NK, NV)), axis=1) - 1.0
    V = cx(NMU, NMU)
    W_R = cx(NMU, NMU, NKX, NKY, NKZ)
    X = cx(NT, NC, NV, NK)
    mesh = _mesh(px, py)
    sh = brc.make_bse_shardings(mesh)
    d = dict(mesh=mesh, sh=sh)
    put = jax.device_put
    d["args"] = [put(psi_c, sh.psi_x), put(psi_c, sh.psi_y),
                 put(psi_v, sh.psi_x), put(psi_v, sh.psi_y),
                 put(eps_c, sh.eps), put(eps_v, sh.eps),
                 put(W_R, sh.W), put(V + V.conj().T, sh.V)]
    d["args"] += [
        jax.jit(compute_pair_amplitude, out_shardings=sh.psi_x)(
            d["args"][0], d["args"][2]),
        jax.jit(compute_pair_amplitude, out_shardings=sh.psi_y)(
            d["args"][1], d["args"][3])]
    d["X"] = put(X, sh.X)
    return d


def _census(fn, *args):
    """(collectives inside while bodies, collectives outside) of the program."""
    text = jax.jit(fn).lower(*args).compile().as_text()
    comps, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s+\(.*\)\s*->.*\{\s*$", line)
        if m:
            cur = m.group(1)
            comps[cur] = []
        elif cur is not None:
            comps[cur].append(line)
    calls, bodies = {c: set() for c in comps}, set()
    for c, lines in comps.items():
        for ln in lines:
            for name in re.findall(r"(?:body|condition|calls|to_apply)=%?([\w.\-]+)", ln):
                if name in comps:
                    calls[c].add(name)
            mb = re.search(r"\bwhile\(.*body=%?([\w.\-]+)", ln)
            if mb:
                bodies.add(mb.group(1))
    seen, stack = set(), list(bodies)
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.add(c)
            stack.extend(calls.get(c, ()))
    inside = outside = 0
    for c, lines in comps.items():
        for ln in lines:
            if re.search(r"\s(" + "|".join(_COLL) + r")(-start)?\(", ln):
                if c in seen:
                    inside += 1
                else:
                    outside += 1
    return inside, outside


def _tda(d):
    from bse.bse_stack_matvec import build_bse_stack_matvec
    mv = build_bse_stack_matvec(d["mesh"], NKX, NKY, NKZ, kernel="bse")
    return mv, (d["X"], *d["args"])


def _pair(d):
    from bse.bse_stack_matvec import build_bse_stack_pair_matvec
    pair = build_bse_stack_pair_matvec(d["mesh"], NKX, NKY, NKZ, kernel="bse")
    return pair


def _per_trial_free(fn_of_nt):
    """(inside, total@NT, total@2NT) — unroll-proof per-trial census.

    XLA:GPU may fully unroll (or hoist out of) a short trial scan, which
    makes "no collective inside the while body" vacuous.  A per-trial
    collective shows either INSIDE a while body or as a total count that
    grows with the trial count, so both are checked.
    """
    i4, o4 = fn_of_nt(NT)
    i8, o8 = fn_of_nt(2 * NT)
    return i4 + i8, i4 + o4, i8 + o8


@pytest.mark.mesh(4)
def test_no_collective_per_trial():
    d = _payload(2, 2)
    with d["mesh"]:
        mv, args = _tda(d)
        pair = _pair(d)

        def tda_census(nt):
            X = jnp.concatenate([d["X"]] * (nt // NT), axis=0)
            return _census(mv, X, *args[1:])

        def pair_census(nt):
            X = jnp.concatenate([d["X"]] * (nt // NT), axis=0)
            return _census(pair, X, jnp.asarray(1.0), *d["args"])

        for name, fn in (("TDA", tda_census), ("pair", pair_census)):
            inside, t4, t8 = _per_trial_free(fn)
            assert inside == 0, f"{name}: {inside} collective(s) in the scan"
            assert t4 == t8 > 0, (
                f"{name}: collective count grows with the trial count "
                f"({t4} at {NT} trials, {t8} at {2 * NT})")

        # RED TWIN: a per-trial psum that depends on the carry (so it can be
        # neither hoisted nor batched) must fail the same criterion.
        from common.shard_map import shard_map
        from jax.sharding import PartitionSpec as P

        def twin_census(nt):
            X = jnp.concatenate([d["X"]] * (nt // NT), axis=0)

            def body(c, xb):
                y = jax.lax.psum(xb * c, "x")
                return c + jnp.sum(y).real * 1e-30, y
            f = shard_map(
                lambda a: jax.lax.scan(body, jnp.float64(1.0), a,
                                       unroll=1)[1],
                mesh=d["mesh"], in_specs=P(None, "x", "y", None),
                out_specs=P(None, "x", "y", None), check_vma=False)
            return _census(f, X)
        inside, t4, t8 = _per_trial_free(twin_census)
        assert inside > 0 or t8 > t4, (
            "the census cannot see a per-trial collective")


@pytest.mark.mesh(4)
@pytest.mark.parametrize("ns", [1, 2, 4])     # scalar, Pauli, bispinor
def test_tda_and_pair_are_mesh_invariant(ns):
    one, four = _payload(1, 1, ns), _payload(2, 2, ns)
    out = {}
    for tag, d in (("1x1", one), ("2x2", four)):
        with d["mesh"]:
            mv, args = _tda(d)
            pair = _pair(d)
            out[tag] = [np.asarray(mv(*args))] + [
                np.asarray(pair(d["X"], jnp.asarray(s), *d["args"]))
                for s in (1.0, -1.0)]
    for a, b, name in zip(out["1x1"], out["2x2"], ("tda", "pair+", "pair-")):
        rel = np.abs(a - b).max() / np.abs(a).max()
        assert rel <= RTOL, f"{name} ns={ns}: 1x1 vs 2x2 rel {rel:.3e}"
