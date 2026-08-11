"""THE RED TWIN for ``vq_interp.refit_vq`` at P>1, on BOTH parities of n_μ.

NEVER AUTO-COLLECTED — the leading underscore keeps it out of ``test_*.py``
globbing, like ``tests/_cache_contract_probe.py`` and ``tests/_env_leak_twin.py``.
It is driven by ``tests/test_refit_vq_shard_p4.py``, which launches it as FOUR
PROCESSES through ``tests/mesh_launch.py``.

WHY THIS NEEDS FOUR PROCESSES AND WHY IT NEEDS AN EVEN n_μ
----------------------------------------------------------
``refit_vq`` fetched its ζ'(G) rows to host with a bare ``jax.device_get``.
That is correct at one process and correct on a REPLICATED array at any
number of processes — ``Array._value`` serves a fully replicated array out of
the local shard before it ever reaches the addressability check — and it
raises on an array whose shards live on other processes.  Which of those three
a run gets is decided by the PARITY OF n_μ:

* ζ inherits its μ-axis sharding from ``bse_setup.psi_rmu_Y``,
  ``P(None, None, None, 'y')``;
* ``common.sharding_fit`` drops a mesh axis the extent cannot divide, so an
  ODD n_μ (the μ=191 downfolded child) is REPLICATED and the fetch works;
* an EVEN n_μ (the μ=960 parent) shards for real, spans processes, and the
  fetch dies with ``RuntimeError: Fetching value for jax.Array that spans
  non-addressable (non process local) devices``.

Every P=4 leg the refit path had ever had was the odd-μ child arm, so the
whole existing coverage was blind to it by construction
(``tests/known_failures/SMALL_ISSUES.md`` row 39).  A single-process
four-DEVICE leg is blind to it too — everything is addressable there — which
is why this is a four-PROCESS twin and not a ``mesh(4)`` cell.

WHAT IT ASSERTS, IN THE ORDER THAT MAKES THE TWIN MEAN ANYTHING
---------------------------------------------------------------
1. **The instrument**: for the even arm, the ζ'(G) box the fetch is applied to
   is recorded and must be genuinely NOT fully addressable and NOT fully
   replicated.  Without this the twin could go green on the pre-fix tree
   because nothing sharded — a green that measured nothing.
2. **The odd arm still works** — it is the arm that has always passed, and the
   replicated branch of ``gather_to_host`` is what keeps it collective-free.
3. **The even arm returns a tile at all** — the fix.
4. **Placement only**: the even arm's tile equals the tile computed with the
   same inputs REPLICATED, to 1e-12 relative.  The arithmetic did not move.
5. **Every rank got the same tile** — checked by the driving cell across the
   four per-rank JSON dumps, because a gather that returned one process's
   shard would be silent otherwise.

The inputs are synthetic and physics-free on purpose: what is under test is
the cross-process transport of ζ'(G), and a deck would put a GW run in front
of it without making the transport any more real.  The SHAPES and the
SHARDING are production's — the μ spec is taken from ``sharding_fit`` exactly
as ``bse_setup`` takes it, so the parity behaviour under test is the module's
own decision and not this file's imitation of it.

Usage (the launcher supplies the four processes)::

    LORRAX_TWIN_OUT=<dir> python tests/_refit_shard_twin.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The two bases.  8 divides a 2x2 mesh's 'y' axis and shards; 7 does not and
#: is replicated.  Same story as 960 vs 191, three orders of magnitude cheaper.
N_MU_EVEN = 8
N_MU_ODD = 7

_NK_GRID = (2, 2, 1)
_NB = 2
_NS = 1
_RANK = 4
_FFT = (4, 4, 4)
_R_CHUNK = 32
_ZETA_CUTOFF = 2.0
#: On the (2,2,1) coarse grid, so ``m_leg="stored"``'s ``kq_index_of_frac``
#: can resolve it.  Non-zero so the centroid winding phase is exercised.
_Q_TILE = np.array([0.5, 0.0, 0.0])


def _synthetic(n_mu):
    """A ``(zx, rst)`` pair with production's shapes and no physics."""
    kg = np.array(_NK_GRID, dtype=np.int64)
    k_int = np.array([[i, j, 0] for i in range(kg[0]) for j in range(kg[1])],
                     dtype=np.int64)
    nk = k_int.shape[0]
    nx, ny, nz = _FFT
    n_rtot = nx * ny * nz
    n_rp = ((n_rtot + _R_CHUNK - 1) // _R_CHUNK) * _R_CHUNK
    rng = np.random.default_rng(20260811 + n_mu)

    def cx(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    zx = {
        "nk": nk, "nb": _NB, "ns": _NS, "n_mu": n_mu, "nq": nk,
        "k_int": k_int, "kgrid": kg,
        "k_lookup": {tuple(v): i for i, v in enumerate(k_int)},
        "psi": cx(nk, _NB, _NS, n_mu),
        "nx": nx, "ny": ny, "nz": nz, "n_rtot": n_rtot,
        "rmu_frac": rng.random((n_mu, 3)),
        "zeta_cutoff": _ZETA_CUTOFF,
        "bvec": np.eye(3, dtype=np.float64),
    }
    rst = {
        "rank": _RANK, "r_chunk": _R_CHUNK, "n_rp": n_rp, "n_rtot": n_rtot,
        "psi_r": cx(nk * _NB, _NS * n_rp),
        "B_full": cx(_RANK, _NS * n_rp),
        # v(q+G) on the fit sphere: positive, q-dependent, and NOT the
        # Coulomb kernel — the tile is a contraction either way and the
        # kernel is not what this twin is about.
        "v_on_set": lambda qw, GS: 1.0 / (
            np.sum((np.asarray(qw)[:, None] + GS.astype(np.float64)) ** 2,
                   axis=0) + 0.1),
    }
    return zx, rst


def main() -> int:
    from runtime import init_jax_distributed
    init_jax_distributed()

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from bse import vq_interp as vqi
    from common.collectives import device_put_process_local
    from common.sharding_fit import fit_sharding

    devs = jax.devices()
    nproc, idx = jax.process_count(), jax.process_index()
    if len(devs) < 4:
        print(f"[twin] REFUSING: {len(devs)} global device(s), need 4",
              file=sys.stderr)
        return 95
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), axis_names=("x", "y"))

    # THE INSTRUMENT.  ``local_fftn3``'s output IS ``ztG_box`` — the array the
    # host fetch is applied to — and patching here (rather than the fetch)
    # records the same fact on the pre-fix and post-fix trees alike.
    seen: dict = {}
    _orig_fftn3 = vqi.local_fftn3

    def _record(x, **kw):
        out = _orig_fftn3(x, **kw)
        seen["addressable"] = bool(getattr(out, "is_fully_addressable", True))
        seen["replicated"] = bool(getattr(out, "is_fully_replicated", False))
        seen["spec"] = str(getattr(getattr(out, "sharding", None), "spec", None))
        return out

    vqi.local_fftn3 = _record

    def _tile(n_mu, *, replicate):
        """One ``refit_vq`` on a μ-sharded (or forcibly replicated) ψ."""
        zx, rst = _synthetic(n_mu)
        spec = P() if replicate else P(None, None, None, "y")
        sh = (NamedSharding(mesh, spec) if replicate else
              fit_sharding(mesh, spec, zx["psi"].shape,
                           "twin.psi_rmu_Y", print_fn=lambda *_: None))
        zx["psi"] = device_put_process_local(zx["psi"], sh)
        seen.clear()
        V = vqi.refit_vq(zx, rst, _Q_TILE, mesh, log_fn=lambda *_: None,
                         m_leg="stored")
        return np.asarray(V), dict(seen)

    out = {"proc_idx": idx, "proc_count": nproc,
           "device_count": len(devs), "local_devices": jax.local_device_count(),
           "platform": devs[0].platform}

    # ODD n_μ — the arm that has always passed, and the one that proves the
    # replicated branch is still collective-free rather than merely correct.
    V_odd, seen_odd = _tile(N_MU_ODD, replicate=False)
    out["odd"] = {"n_mu": N_MU_ODD, "shape": list(V_odd.shape),
                  **seen_odd,
                  "fro": float(np.linalg.norm(V_odd)),
                  "herm": float(np.max(np.abs(V_odd - V_odd.conj().T)))}

    # EVEN n_μ — the defect.  Pre-fix this raises before anything below runs.
    V_even, seen_even = _tile(N_MU_EVEN, replicate=False)
    out["even"] = {"n_mu": N_MU_EVEN, "shape": list(V_even.shape),
                   **seen_even,
                   "fro": float(np.linalg.norm(V_even)),
                   "herm": float(np.max(np.abs(V_even - V_even.conj().T)))}

    # PLACEMENT ONLY — the same tile with everything replicated.
    V_rep, seen_rep = _tile(N_MU_EVEN, replicate=True)
    denom = max(float(np.linalg.norm(V_rep)), 1e-300)
    out["even_vs_replicated_rel"] = float(
        np.linalg.norm(V_even - V_rep) / denom)
    out["replicated_arm"] = seen_rep

    # Rank agreement is checked by the driving cell across the four dumps;
    # the checksum is what makes that comparison exact rather than statistical.
    out["even_bytes_md5"] = __import__("hashlib").md5(
        np.ascontiguousarray(V_even).tobytes()).hexdigest()
    out["odd_bytes_md5"] = __import__("hashlib").md5(
        np.ascontiguousarray(V_odd).tobytes()).hexdigest()

    dest = os.environ.get("LORRAX_TWIN_OUT")
    if dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / f"rank{idx}_of{nproc}.json").write_text(
            json.dumps(out, indent=1))
    print("[twin] " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    from runtime import finalize_process
    finalize_process(main())
