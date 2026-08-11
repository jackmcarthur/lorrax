"""``vq_interp.refit_vq`` at FOUR PROCESSES, on an EVEN n_μ — SMALL_ISSUES 39.

THE COVERAGE THIS REPLACES WAS BLIND BY CONSTRUCTION.  ``refit_vq`` fetched
ζ'(G) to host with a bare ``jax.device_get``.  Whether that works is decided
by the parity of n_μ, because ``common.sharding_fit`` drops a mesh axis the
extent cannot divide:

  * n_μ ODD  → the μ axis is REPLICATED → ``Array._value`` serves it from the
    local shard → the fetch works at any process count;
  * n_μ EVEN → the μ axis really shards → at P>1 the array spans processes →
    ``RuntimeError: Fetching value for jax.Array that spans non-addressable
    (non process local) devices``.

Every P=4 leg the refit path ever had was the μ=191 downfolded child, which is
odd.  The μ=960 parent is even and died on its first four-process leg
(``qsign_recut_0811/_logs/xb_ctl_parent.log``), which is why that lane's parent
control had to be taken at P=1.  So this file's requirement is not "run it at
P=4" but "run it at P=4 **on an even n_μ**" — the odd case cannot see the
defect and neither can a single-process four-DEVICE leg, where every shard is
addressable by definition.

The payload is ``tests/_refit_shard_twin.py``, launched through
``tests/mesh_launch.py``: four real GPUs in four processes under
``lx run -N 1 -G 4 -n 1 -- pytest tests/test_refit_vq_shard_p4.py``, four local
CPU processes off-cluster.  Both are honest for this defect — it is
device-count/process-count LOGIC, which the four-GPU rule's own unit/CPU clause
exempts — and the mode travels in every failure message so a CPU run can never
be quoted as the GPU one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_launch                                          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TWIN = REPO_ROOT / "tests" / "_refit_shard_twin.py"

#: A requirement declaration, not a selector — same convention as
#: ``tests/test_jax_cache_contract.py``.
pytestmark = pytest.mark.procs(4)

_TIMEOUT_S = 900

#: Placement must not move arithmetic.  The sharded and replicated arms run
#: the same jitted kernels on the same values in a different layout, so the
#: only difference available to them is floating-point reassociation inside
#: XLA; 1e-12 relative on a Frobenius norm is far above that and far below
#: anything a transport bug could hide in.
_PLACEMENT_REL_TOL = 1e-12


def _require_mesh4():
    mode, why = mesh_launch.choose_mode(dict(__import__("os").environ))
    if mode == mesh_launch.NONE:
        pytest.skip(f"no four-process launch available here: {why}")
    return mode, why


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    """One four-process run of the twin; every cell below reads its dumps."""
    mode, why = _require_mesh4()
    out = tmp_path_factory.mktemp("refit_shard_twin")
    env = dict(__import__("os").environ)
    env["LORRAX_TWIN_OUT"] = str(out)
    env.setdefault("JAX_ENABLE_X64", "1")
    res = mesh_launch.run_mesh4([sys.executable, str(TWIN)],
                                cwd=REPO_ROOT, env=env,
                                timeout=_TIMEOUT_S, mode=mode)
    dumps = sorted(
        (json.loads(p.read_text()) for p in out.glob("rank*_of*.json")),
        key=lambda d: d.get("proc_idx", 0))
    if not res.ok:
        pytest.fail(res.blame(
            f"refit_vq at P=4 on an EVEN n_mu did not complete (launch mode "
            f"{mode}: {why}).  This is the RED TWIN of SMALL_ISSUES row 39: "
            f"on the pre-fix tree it dies in vq_interp.refit_vq with "
            f"'Fetching value for `jax.Array` that spans non-addressable "
            f"(non process local) devices'.  {len(dumps)} of "
            f"{mesh_launch.NPROC} rank dumps were written."))
    return res, dumps


def test_every_rank_reported(twin):
    """ARM 0.  A missing dump is a rank that never reached exit."""
    res, dumps = twin
    assert len(dumps) == mesh_launch.NPROC, res.blame(
        f"{len(dumps)} rank dump(s) for {mesh_launch.NPROC} ranks "
        f"(proc_idx present = {[d.get('proc_idx') for d in dumps]})")
    assert {d["proc_count"] for d in dumps} == {mesh_launch.NPROC}
    assert {d["device_count"] for d in dumps} == {4}


def test_the_even_arm_really_spans_processes(twin):
    """ARM 1, AND WITHOUT IT THE TWIN MEANS NOTHING.

    If the even arm's ζ'(G) box came back replicated or fully addressable then
    the fetch was never on the defective path and a green here would be a
    green that measured nothing — the same shape as the odd-μ coverage this
    file exists to replace.  So the twin RECORDS what it sharded and this
    cell refuses anything but a genuinely process-spanning array.
    """
    res, dumps = twin
    for d in dumps:
        ev = d["even"]
        assert ev["addressable"] is False, res.blame(
            f"rank {d['proc_idx']}: the EVEN n_mu={ev['n_mu']} zeta'(G) box "
            f"is fully addressable (spec {ev['spec']}), so this run did not "
            f"exercise the cross-process fetch at all")
        assert ev["replicated"] is False, res.blame(
            f"rank {d['proc_idx']}: the EVEN n_mu={ev['n_mu']} zeta'(G) box "
            f"is fully REPLICATED (spec {ev['spec']}) — that is the odd-mu "
            f"arm's layout, not the sharded one under test")


def test_the_odd_arm_is_replicated_and_still_works(twin):
    """ARM 2 — the discrimination, and the arm that must not regress.

    n_μ odd is what every previous P=4 refit leg ran, and it works through
    ``gather_to_host``'s REPLICATED branch, which issues no collective at
    all.  A "fix" that pushed it through ``process_allgather`` instead would
    be correct and would quietly add a collective to the arm the production
    downfolded child takes; this cell is what would notice.
    """
    _res, dumps = twin
    for d in dumps:
        od = d["odd"]
        assert od["replicated"] is True, (
            f"rank {d['proc_idx']}: n_mu={od['n_mu']} is odd and should have "
            f"been replicated by sharding_fit, but the box reports spec "
            f"{od['spec']}")
        assert od["shape"] == [od["n_mu"], od["n_mu"]]
        assert od["herm"] < 1e-9


def test_the_even_arm_returns_the_tile(twin):
    """ARM 3 — THE FIX.  Pre-fix the fixture itself fails; this names why."""
    _res, dumps = twin
    for d in dumps:
        ev = d["even"]
        assert ev["shape"] == [ev["n_mu"], ev["n_mu"]]
        assert ev["fro"] > 0.0
        assert ev["herm"] < 1e-9, (
            f"rank {d['proc_idx']}: the sharded tile is not Hermitian "
            f"({ev['herm']:.3e}) — a partial gather would look exactly like "
            f"this")


def test_the_fix_is_placement_only(twin):
    """ARM 4.  Sharded tile == replicated tile.  No arithmetic moved."""
    _res, dumps = twin
    for d in dumps:
        rel = d["even_vs_replicated_rel"]
        assert rel < _PLACEMENT_REL_TOL, (
            f"rank {d['proc_idx']}: the mu-SHARDED refit tile differs from "
            f"the REPLICATED one by {rel:.3e} relative.  The fix is a "
            f"transport change and must not move a value")


def test_every_rank_got_the_same_global_tile(twin):
    """ARM 5 — the silent one.

    ``gather_to_host`` returns the whole global array on every process.  A
    gather that returned this process's SHARD instead would still be
    Hermitian, still be the right dtype, and would differ between ranks — so
    the byte checksums, compared across the four dumps, are the only thing
    that says the gather was global.
    """
    _res, dumps = twin
    for key in ("even_bytes_md5", "odd_bytes_md5"):
        got = {d["proc_idx"]: d[key] for d in dumps}
        assert len(set(got.values())) == 1, (
            f"{key} differs across ranks: {got}.  Each rank fetched "
            f"something different, which is what a non-global gather looks "
            f"like")
