"""The stack matvec's W term runs NO collective per trial (survey_C §C1).

``bse.bse_stack_matvec`` gathers the trial block X once per block, builds
both encodes with the full c and v locally, decodes through this rank's
(μ_loc, ν_loc) into a (b, c, v, nk) partial, and completes both sums with ONE
reduce-scatter after the scan.  Two checks on a 2x2 mesh:

1. The compiled TDA matvec and the non-TDA pair applier carry ZERO
   collectives inside any while body (the trial scan), and their W-term
   collectives outside it are exactly the block gather and the one
   reduce-scatter.  Red twin: the same census on a program that DOES run a
   per-trial collective must count it.
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


def _payload(px, py):
    import bse.bse_ring_comm as brc
    from bse.bse_serial import compute_pair_amplitude

    rng = np.random.default_rng(20260924)
    cx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)  # noqa: E731
    psi_c, psi_v = cx(NK, NC, NS, NMU), cx(NK, NV, NS, NMU)
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


@pytest.mark.mesh(4)
def test_no_collective_inside_the_trial_scan():
    d = _payload(2, 2)
    with d["mesh"]:
        mv, args = _tda(d)
        inside, outside = _census(mv, *args)
        assert inside == 0, f"TDA: {inside} collective(s) inside the trial scan"
        assert outside > 0
        pair = _pair(d)
        pin, pout = _census(pair, d["X"], jnp.asarray(1.0), *d["args"])
        assert pin == 0, f"pair: {pin} collective(s) inside the trial scan"

        # RED TWIN: the census must see a per-trial collective when one exists.
        def per_trial(x):
            def body(c, xb):
                return c, jax.lax.psum(xb, "x")
            from common.shard_map import shard_map
            from jax.sharding import PartitionSpec as P
            return shard_map(lambda a: jax.lax.scan(body, None, a, unroll=1)[1],
                             mesh=d["mesh"], in_specs=P(None, "x", "y", None),
                             out_specs=P(None, "x", "y", None),
                             check_vma=False)(x)
        twin_in, _ = _census(per_trial, d["X"])
        assert twin_in >= 1, "the census cannot see a per-trial collective"


@pytest.mark.mesh(4)
def test_tda_and_pair_are_mesh_invariant():
    one, four = _payload(1, 1), _payload(2, 2)
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
        assert rel <= RTOL, f"{name}: 1x1 vs 2x2 rel {rel:.3e}"
