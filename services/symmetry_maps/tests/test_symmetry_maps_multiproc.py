"""Layer L-c: the L-b bodies under FOUR REAL PROCESSES.

    lx run --cpu -N 1 -n 4 python3 \\
        services/symmetry_maps/tests/test_symmetry_maps_multiproc.py --mesh 2x2

ONE SET OF CHECK BODIES, TWO CALLERS — the ``_CLI_CELLS`` pattern, copied
from ``services/distrib_la/tests/test_distrib_la_multiproc.py`` rather than
reinvented.  Every ``check_*(mesh)`` below is called by a pytest cell (on
whatever mesh this process can build) AND by ``_cli_main`` under ``srun``.
Duplicating the logic across the two would mean the multi-rank leg tests
something slightly different from the thing the suite pins, which is how a
matrix leg drifts out of agreement with its own reference.

WHAT FOUR PROCESSES ADD OVER FOUR EMULATED DEVICES.  ``unfold_v_q``'s
``lax.all_to_all`` redistribution is a COLLECTIVE.  On an emulated mesh all
four "devices" are threads in one address space and the collective is a
local shuffle; on four ranks it is a real exchange, the operand is
genuinely partitioned, and the "never more than 1x single-tile per rank"
memory contract is a claim about four separate heaps rather than about one.
Nothing here is a different assertion — the numbers are the same numbers —
but a redistribution that silently relied on the whole array being
addressable would pass emulated and fail here.

SKIP, NEVER ASSERT, below four processes.  ``jax.process_count()`` is a
property of how the leg was LAUNCHED, not of the code under test;
``tests/KNOWN_FAILURES.md`` lists eleven cells that went red on a
single-process leg purely for writing ``assert n >= 4``.  The pytest cells
here run their bodies on whatever mesh exists (1x1 on the dev box, which is
still a real check of the same body) and the four-rank claim comes from the
CLI mode.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

# Path first: this file runs as a bare script under srun, where nothing has
# put services/*/src anywhere (`lx` rewrites the container PYTHONPATH to
# exactly <checkout>/src).
_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
for _svc in ("lxkit", "symmetry_maps"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

# CLI multi-rank mode: jax.distributed.initialize must run before ANY
# XLA-backend touch, so it happens at import time when this module is the
# entry point of a multi-task launch.
if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        import jax
        jax.distributed.initialize()

import test_symmetry_maps_emulated_mesh as L_b                # noqa: E402
from lxkit.testing import require_devices                     # noqa: E402
from symmetry_maps import (KStarMap, star_broadcast,          # noqa: E402
                           star_select, star_spread, unfold_v_q)

#: The engine-parity bar for a redistributed kernel: RELATIVE, never
#: bit-exact.  The sharded path computes the umklapp phase per rank from a
#: dynamic slice — same arithmetic, different order.
RTOL = 1e-13


# ---------------------------------------------------------------------------
# Check bodies — called by BOTH the pytest cells and _cli_main
# ---------------------------------------------------------------------------

def check_unfold_matches_the_hand_reference(mesh):
    """``unfold_v_q`` against the pure-numpy per-element reference.

    The reference lives in the L-b module and is imported, not copied: two
    references would be two chances to be wrong about the same formula.
    """
    import jax
    import jax.numpy as jnp
    perm, L, n_rmu = L_b._divisible_geometry()
    V = L_b._hermitian_ibz(n_rmu)
    ref = L_b._hand_unfold(V, perm=perm, L=L, n_rmu=n_rmu)
    out = unfold_v_q(jnp.asarray(V), irr_idx=L_b._IRR,
                     sym_idx=L_b._SYM, sym_perm=perm, L_table=L,
                     q_irr_frac=L_b._Q_IRR, mesh_xy=mesh,
                     n_sym_spatial=L_b._NTRAN)
    # The output is P(None,'x','y')-sharded: at P>1 no rank holds the
    # global array, and ``np.asarray`` on it raises "spans non-addressable
    # devices" (measured on the real 4-rank leg, 2026-08-07).  Compare each
    # ADDRESSABLE shard against the reference block its ``index`` names —
    # this also checks placement, which an allgather would erase.
    if jax.process_count() == 1:
        num = float(np.abs(np.asarray(out) - ref).max())
    else:
        num = 0.0
        for shard in out.addressable_shards:
            blk = ref[shard.index]
            num = max(num, float(np.abs(np.asarray(shard.data) - blk).max()))
    rel = num / float(np.abs(ref).max())
    assert rel < RTOL, f"unfold_v_q differs from the reference: {rel:.3e}"
    return f"rel {rel:.2e} n_rmu {n_rmu}"


def check_hostile_extent_refuses(mesh):
    """G10 on real ranks: a non-dividing μ extent must REFUSE, not clip.

    On a 1x1 there is nothing to divide and the guard is unreachable, so
    the body says so instead of asserting a refusal that cannot happen —
    the same distinction ``require_devices`` makes for device counts.
    """
    import jax.numpy as jnp
    px = int(mesh.shape["x"])
    py = int(mesh.shape["y"])
    perm, L, n_rmu = L_b._hostile_geometry()
    V = jnp.asarray(L_b._hermitian_ibz(n_rmu))
    if n_rmu % (px * py) == 0:
        return f"inapplicable: n_rmu={n_rmu} divides {px}x{py}"
    try:
        unfold_v_q(V, irr_idx=L_b._IRR, sym_idx=L_b._SYM, sym_perm=perm,
                   L_table=L, q_irr_frac=L_b._Q_IRR, mesh_xy=mesh,
                   n_sym_spatial=L_b._NTRAN)
    except ValueError as exc:
        assert f"n_rmu_padded={n_rmu}" in str(exc), str(exc)
        assert f"Px*Py={px * py}" in str(exc), str(exc)
        return f"refused n_rmu={n_rmu} on {px}x{py}"
    raise AssertionError(
        f"n_rmu={n_rmu} does not divide {px}x{py}={px * py} and unfold_v_q "
        f"accepted it; a promise_in_bounds gather clips SILENTLY")


def check_star_helpers_keep_the_sharding(mesh):
    """``star_select`` / ``star_broadcast`` on a genuinely sharded operand.

    Bit-identical to the host answer, and the output spec is the input
    spec: the SC loop hands these a ``P(None,'x','y')`` array four times an
    iteration and must get one back.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from symmetry_maps.maps import _star_conj_flags
    irr = np.array([0, 2, 2, 6, 8, 7, 6, 7, 8], dtype=np.int32)
    sidx = np.array([0, 2, 0, 2, 2, 2, 0, 0, 0], dtype=np.int32)
    rng = np.random.default_rng(41)
    base = {}
    for v in sorted({int(x) for x in irr}):
        m = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        base[v] = 0.5 * (m + m.conj().T)
    _, xor = _star_conj_flags(irr, sidx, 2)
    A = np.stack([np.conj(base[int(irr[k])]) if xor[k] else base[int(irr[k])]
                  for k in range(len(irr))])
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dev = jax.device_put(jnp.asarray(A), sh)

    sel_h, labels = star_select(A, irr)
    sel_d, _ = star_select(dev, irr)
    back_h = star_broadcast(sel_h, irr, sidx, 2, irr_labels=labels)
    back_d = star_broadcast(sel_d, irr, sidx, 2, irr_labels=labels)
    assert sel_d.sharding.spec == sh.spec, sel_d.sharding.spec
    assert back_d.sharding.spec == sh.spec, back_d.sharding.spec

    def host(a):
        rep = NamedSharding(mesh, P(*([None] * int(a.ndim))))
        return np.asarray(jax.device_put(a, rep))

    d_sel = float(np.abs(host(sel_d) - sel_h).max())
    d_bck = float(np.abs(host(back_d) - back_h).max())
    assert d_sel == 0.0 and d_bck == 0.0, (
        f"device/host differ: select {d_sel:.1e} broadcast {d_bck:.1e}")
    km = KStarMap(irr, sidx, 2)
    assert km.spread_rel(dev) == km.spread_rel(A)
    assert star_spread(dev, irr, sidx, 2) == 0.0
    return "select/broadcast/spread bit-identical, spec kept"


def _nan_operand(mesh, spec):
    """``(KStarMap, poisoned operand)`` — one all-NaN k row on ``spec``.

    THE POISON IS APPLIED ON DEVICE, NOT ON THE HOST, AND THAT IS
    LOAD-BEARING: ``device_put`` on a multi-process mesh verifies the value
    is identical on every process by ELEMENTWISE COMPARISON, and
    ``NaN != NaN``, so a NaN-carrying host array is rejected as "not the
    same on each process" even though every rank built it identically
    (measured on the real 4-rank leg, 2026-08-07).  The clean array goes
    through ``device_put``; a jitted ``.at[3].set(nan)`` with explicit
    ``out_shardings`` poisons it where it lives.
    """
    import functools

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    irr = np.array([0, 0, 1, 1], dtype=np.int32)
    sidx = np.zeros(4, dtype=np.int32)
    A = np.ones((4, 8, 8), dtype=np.complex128)
    A[1] *= 2.0
    km = KStarMap(irr, sidx, 1)
    sh = NamedSharding(mesh, spec)
    dev = jax.device_put(jnp.asarray(A), sh)

    @functools.partial(jax.jit, out_shardings=sh)
    def _poison(x):
        return x.at[3].set(jnp.asarray(np.nan + 0j, dtype=x.dtype))

    return km, _poison(dev), A


def check_spread_rel_is_one_replicated_scalar(mesh):
    """S3 at P>1: ``spread_rel`` returns a plain float on every rank.

    ``_star_stats`` pins the reduction's output to a replicated ``P()``
    precisely so the single ``np.asarray`` is legal at P>1 — a partially
    addressable array raises "spans non-addressable devices".  That refusal
    is invisible in one process, which is why this body is here.

    THE NaN HALF IS ASSERTED ON A REPLICATED OPERAND, AND THAT IS NOT AN
    ACCIDENT.  ``sc_iteration.py:780`` is spelled ``not (spread <= tol)``
    so a poisoned Σ REFUSES; that needs ``spread_rel`` to hand NaN back.
    It does — on the host, on a single device, and on a REPLICATED device
    array.  It does NOT on a MESH-SHARDED one, which is what the SC loop
    actually passes, and the measurement is in
    :func:`test_a_nan_survives_the_sharded_reduction` below.  Asserting the
    replicated case here keeps this body green on every leg while the
    sharded case stays visible as the defect it is instead of being
    softened into "NaN sometimes propagates".
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    irr = np.array([0, 0, 1, 1], dtype=np.int32)
    sidx = np.zeros(4, dtype=np.int32)
    A = np.ones((4, 8, 8), dtype=np.complex128)
    A[1] *= 2.0
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dev = jax.device_put(jnp.asarray(A), sh)
    km = KStarMap(irr, sidx, 1)
    got = km.spread_rel(dev)
    assert isinstance(got, float), type(got)
    assert got == km.spread_rel(A)

    _, rep_poisoned, _ = _nan_operand(mesh, P(None, None, None))
    rep = km.spread_rel(rep_poisoned)
    assert isinstance(rep, float) and np.isnan(rep), (
        f"a REPLICATED device operand must propagate NaN; got {rep!r}")
    assert not (rep <= 1e-6), "a NaN spread must REFUSE, not pass"

    # RECORDED so the four-rank artifact carries the number, and it is NOT
    # the same number the emulated mesh gives.  MEASURED 2026-08-07,
    # Perlmutter, jax 0.7.0, JAX_PLATFORMS=cpu, srun -n 4 at BOTH 2x2 and
    # 4x1: this returns ``nan`` — the real cross-process all-reduce
    # PROPAGATES.  On a single-process EMULATED four-device mesh it
    # returns -inf (the strict xfail cell below).  Two variables move
    # between those legs — process topology and jax version — and which
    # one causes the loss is unresolved; this line is the artifact that
    # keeps the distinction from being lost.
    _, shard_poisoned, _ = _nan_operand(mesh, P(None, "x", "y"))
    sharded = km.spread_rel(shard_poisoned)
    return (f"spread_rel {got:.3e}, replicated-NaN propagates, "
            f"sharded-NaN -> {sharded!r}")


# ---------------------------------------------------------------------------
# The pytest cells — the same bodies, on whatever mesh this process has
# ---------------------------------------------------------------------------

def _mesh_here(px=1, py=1):
    import jax
    from jax.sharding import Mesh
    require_devices(px * py, platform="cpu")
    return Mesh(np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py),
                ("x", "y"))


@pytest.mark.parametrize("body", [
    check_unfold_matches_the_hand_reference,
    check_hostile_extent_refuses,
    check_star_helpers_keep_the_sharding,
    check_spread_rel_is_one_replicated_scalar,
], ids=lambda f: f.__name__[len("check_"):])
def test_the_check_body_passes_on_this_processs_mesh(body):
    """Every L-c body, run at 1x1 (or whatever this process can build).

    Not a substitute for the four-rank leg — it is the thing that keeps the
    bodies from rotting between Perlmutter windows.  A body that stopped
    even compiling would otherwise be discovered by an ``srun`` job.
    """
    body(_mesh_here())


@pytest.mark.parametrize("body", [
    check_unfold_matches_the_hand_reference,
    check_hostile_extent_refuses,
    check_star_helpers_keep_the_sharding,
    check_spread_rel_is_one_replicated_scalar,
], ids=lambda f: f.__name__[len("check_"):])
def test_the_check_body_passes_on_an_emulated_2x2(body):
    """The same bodies on four emulated devices, when the flag took.

    SKIPS below four — via ``require_devices``, which skips and never
    asserts.  In the full monorepo run the XLA flag is inert by design and
    these four cells are the ones the skip-honesty allowlist's universal
    "needs >= 4 devices" row covers.
    """
    body(_mesh_here(2, 2))


@pytest.mark.xfail(strict=True, reason=(
    "MEASURED DEFECT on an EMULATED four-device mesh, jax 0.9.1 / XLA CPU: "
    "a PARTITIONED max reduction discards NaN.  A REAL srun -n 4 mesh "
    "propagates it (measured 2026-08-07) — see the docstring for both legs. "
    "Not this package's arithmetic, and out of scope for the symmetry_maps "
    "branch, so it is carried as a strict xfail rather than silently "
    "softened.  Delete the marker the day the emulated reduction "
    "propagates."))
def test_a_nan_survives_the_sharded_reduction():
    """S3's remaining half, and it does NOT hold.  Measured, not inferred.

    ``sc_iteration._check_kstar_spread`` refuses on ``not (spread <= tol)``
    — spelled that way DELIBERATELY (``tests/test_sc_kstar_spread.py`` pins
    it) so a poisoned Σ cannot pass by comparing False.  That only works if
    ``KStarMap.spread_rel`` hands NaN back.  On the operand the SC loop
    actually passes — a ``P(None,'x','y')`` sharded ``jax.Array`` — it does
    not.

    MEASURED 2026-08-07, jax 0.9.1, ``JAX_PLATFORMS=cpu``,
    ``--xla_force_host_platform_device_count=4``, on ``np.ones((4,8,8))``
    with row 3 set to NaN::

        sharding              jnp.max     jnp.min     jnp.sum
        P(None,'x','y')       -inf        +inf        nan
        P() / all-replicated  nan         nan         nan
        P('x',None,None)      1.0   <-- a WRONG FINITE VALUE
        single device         nan         nan         nan
        numpy host            nan         nan         nan

        KStarMap.spread_rel:  host nan | single-device nan | sharded -inf

    The per-shard local maxima ARE NaN (read back individually), so the
    loss is in the cross-shard reduction, not in ``jnp.abs`` and not in
    anything ``_star_stats`` does: it is the max/min IDENTITY (-inf/+inf)
    coming back from a collective that compared NaN and lost.  ``jnp.sum``
    is unaffected.

    WHAT THE REAL FOUR-RANK LEG SAYS, measured afterwards and NOT the same
    answer.  2026-08-07, Perlmutter, jax 0.7.0, ``JAX_PLATFORMS=cpu``,
    ``srun -n 4`` at 2x2 AND 4x1, the same
    ``check_spread_rel_is_one_replicated_scalar`` body: ``sharded-NaN ->
    nan``.  A real cross-process all-reduce PROPAGATES.  So the loss above
    is so far an EMULATED-mesh result only, and two things differ between
    the legs — process topology (one process holding four devices vs four
    processes) and jax version (0.9.1 vs 0.7.0).  Which one causes it is
    UNRESOLVED; a deconfound probe belongs in the land-readiness report,
    not in this docstring.

    CONSEQUENCE, and this is the owner-visible part: WHEREVER the loss
    holds, a NaN Σ makes ``spread_rel`` return −inf, ``-inf <= 1e-6`` is
    True, and the k-star spread gate PASSES a poisoned iteration.  The
    real-rank measurement above is the reason that is not yet a statement
    about production — it is a statement about every mesh that reduces the
    way this one does, and about any jax upgrade that makes a production
    mesh reduce that way.

    NOT FIXED HERE.  ``_star_stats`` is a diagnostic on a
    register-don't-touch module and the fix is a decision about which
    reduction to trust (a ``jnp.sum``-based NaN sentinel beside the max, or
    the refusal moving into ``sc_iteration``), which is not this branch's
    call.  ``strict=True`` so the day the reduction starts propagating this
    turns RED and somebody deletes the marker instead of the suite quietly
    keeping a stale claim.
    """
    from jax.sharding import PartitionSpec as P
    mesh = _mesh_here(2, 2)
    km, poisoned, _ = _nan_operand(mesh, P(None, "x", "y"))
    got = km.spread_rel(poisoned)
    assert isinstance(got, float)
    assert np.isnan(got), (
        f"spread_rel returned {got!r} for a NaN-poisoned sharded operand; "
        f"`not (spread <= tol)` cannot refuse on that")


def test_every_check_body_has_a_cli_row():
    """The failure the CLI mode cannot report about itself.

    A body with no ``_CLI_CELLS`` row is a body the four-rank leg never
    runs, and the leg prints ``done: N cells ran`` without noticing.  The
    referenced globals are read out of each lambda's code object rather
    than matched on its label, because matching on the label is the version
    of this check that silently rots: rename a body and the substring stops
    matching while both sides still exist.
    """
    names = {name for name, _ in _CLI_CELLS}
    assert len(names) == len(_CLI_CELLS), "duplicate _CLI_CELLS name"
    called = set()
    for name, fn in _CLI_CELLS:
        refs = {g for g in fn.__code__.co_names if g.startswith("check_")}
        assert refs, f"{name}: calls no check body"
        called |= refs
    bodies = {k for k in globals() if k.startswith("check_")}
    assert bodies - called == set(), (
        f"check bodies with no _CLI_CELLS row (the 2x2 leg would never run "
        f"them): {sorted(bodies - called)}")
    assert called - bodies == set(), (
        f"_CLI_CELLS names bodies that do not exist: {sorted(called - bodies)}")


def test_the_cli_row_check_can_fail():
    """RED TWIN.  A body missing from the table must be detected.

    The check above is a set difference and would read as passing if the
    table were empty; this constructs the omission and asserts the
    difference is non-empty.
    """
    bodies = {"check_a", "check_b"}
    rows = {"check_a"}
    assert bodies - rows == {"check_b"}


# ---------------------------------------------------------------------------
# CLI mode — the real multi-rank matrix
# ---------------------------------------------------------------------------

_CLI_CELLS = [
    ("unfold_reference", lambda mesh: check_unfold_matches_the_hand_reference(
        mesh)),
    ("hostile_extent", lambda mesh: check_hostile_extent_refuses(mesh)),
    ("star_sharding", lambda mesh: check_star_helpers_keep_the_sharding(mesh)),
    ("spread_rel_scalar",
     lambda mesh: check_spread_rel_is_one_replicated_scalar(mesh)),
]


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _cli_main():
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="PxQ process mesh")
    ap.add_argument("--only", default="", help="substring filter")
    args = ap.parse_args()
    mesh = _mesh_from_arg(args.mesh)
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}",
       flush=True)

    failures, ran = 0, 0
    for name, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        tag = f"{name}[{args.mesh}]"
        try:
            out = fn(mesh)
            ran += 1
            p0(f"PASS {tag} {out if out is not True else ''}", flush=True)
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {tag}: {exc}", flush=True)
        except Exception as exc:                               # noqa: BLE001
            failures += 1
            p0(f"ERROR {tag}: {type(exc).__name__}: "
               f"{' '.join(str(exc).split())[:400]}", flush=True)
    # RAN, not just failures.  "0 failures" out of 0 cells is the shape of
    # every artifact-free green in this tree's history.
    p0(f"done: {ran} cells ran, {failures} failures", flush=True)
    return 1 if (failures or ran == 0) else 0


if __name__ == "__main__":
    sys.exit(_cli_main())
