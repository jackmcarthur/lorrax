"""The streamed Stage-C ζ transform runs, and is priced, over k tiles.

Refusal this gates (JID 58786388 step .0, CrI3 16x16 bispinor charge fit at
P64): the planner priced the cache-free ψ(G)->ψ(r) box at every k row at
once, a P-independent 105 GB (analytic) / 52 GB (measured) term, so
``P_min`` ran to the 2**20 sentinel and the only "fitting" r chunk was 8.
The executor transforms the ψ(G) store's k rows (the 30 raw parents there)
in ``gw.gflat_memory_model.zeta_fft_k_tile`` tiles, and the plan prices
that same tile.

The transform cells run in a CPU subprocess (x64 before jax import; the
pattern of ``tests/test_wfn_rpoints_and_indexed_gflat.py``).
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _case_transform() -> dict:
    """Tiled vs untiled ``to_rpoints_inner`` on a synthetic G sphere."""
    import jax.numpy as jnp

    from common.wfn_transforms import to_rpoints_inner

    rng = np.random.default_rng(20260923)
    fft_grid = (4, 4, 6)
    n_rtot = 96
    nk, nb, ns, ngkmax = 6, 2, 4, 11
    psi = jnp.asarray(
        rng.standard_normal((nk, nb, ns, ngkmax))
        + 1j * rng.standard_normal((nk, nb, ns, ngkmax)))
    g_index = np.full((nk,) + fft_grid, ngkmax, dtype=np.int32)
    for k in range(nk):
        flat = g_index[k].reshape(n_rtot)
        flat[rng.choice(n_rtot, size=ngkmax, replace=False)] = np.arange(
            ngkmax, dtype=np.int32)
    g_index = jnp.asarray(g_index)
    kvecs = jnp.asarray(rng.uniform(-0.5, 0.5, size=(nk, 3)))
    # Scattered tile cells, including two out-of-range pad slots.
    r_idx = jnp.asarray(np.concatenate([
        rng.permutation(n_rtot)[:13], [n_rtot, n_rtot + 5]]).astype(np.int32))

    import jax

    def _compiled(kt, kv):
        # Compiled, as in the executor: an eager reference would compare
        # op-by-op dispatch against a fused program, not tiled vs untiled.
        return np.asarray(jax.jit(lambda p, g, r, k: to_rpoints_inner(
            p, g, fft_grid, r, kvecs_frac=k, norm="ortho", k_tile=kt))(
                psi, g_index, r_idx, kv))

    out = {}
    for label, kv in (("phase", kvecs), ("nophase", None)):
        ref = _compiled(None, kv)
        for kt in (1, 2, 3, 6):
            tiled = _compiled(kt, kv)
            out[f"{label}_kt{kt}_equal"] = bool(np.array_equal(tiled, ref))
            out[f"{label}_kt{kt}_maxabs"] = float(np.max(np.abs(tiled - ref)))
    try:
        to_rpoints_inner(psi, g_index, fft_grid, r_idx, k_tile=4)
        out["non_divisor_refused"] = False
    except ValueError:
        out["non_divisor_refused"] = True
    return out


def _case_executor() -> dict:
    """The parent ζ executor, streamed route, tiled vs forced-untiled.

    Reuses the parent-parity fixture (3 raw parents, dense NumPy q/band
    reference) on a 1x1 mesh, where the rule's tile is 1 of 3 rows.  Every
    streamed ``z_q_from_psi_sm`` call is re-run with the rule forced to the
    full row count; the two Z_q must be bit-identical.
    """
    import importlib.util

    import jax
    import gw.gflat_memory_model as gmm
    import isdf.core as core

    spec = importlib.util.spec_from_file_location(
        "_zq_parent_parity",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_isdf_zq_parent_parity.py"))
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)

    real_z, real_rule = core.z_q_from_psi_sm, gmm.zeta_fft_k_tile
    record = {"streamed_calls": 0, "all_equal": True, "tiles": []}

    def _untiled(*, n_k_rows, band_chunk, p_band):
        return int(n_k_rows)

    def _compare(**kw):
        tiled = real_z(**kw)
        if kw.get("psi_r_cache") is None:
            store = kw["psi_G_store"]
            record["tiles"].append([real_rule(
                n_k_rows=store.local_band_chunk_shape[0],
                band_chunk=store.band_chunk_carrier,
                p_band=int(kw["mesh_xy"].devices.size)),
                int(store.local_band_chunk_shape[0])])
            gmm.zeta_fft_k_tile = _untiled
            try:
                untiled = real_z(**kw)
            finally:
                gmm.zeta_fft_k_tile = real_rule
            record["streamed_calls"] += 1
            record["all_equal"] &= bool(np.array_equal(
                np.asarray(jax.block_until_ready(tiled)),
                np.asarray(jax.block_until_ready(untiled))))
        return tiled

    core.z_q_from_psi_sm = _compare
    results = {}
    try:
        for case in ("ns4_charge", "ns2_multi"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                parity._worker(case, mesh_shape=(1, 1))
            line = [ln for ln in buf.getvalue().splitlines()
                    if ln.strip().startswith("{")][-1]
            results[case] = json.loads(line)["max_rel"]
    finally:
        core.z_q_from_psi_sm = real_z
    record["max_rel_vs_dense"] = results
    return record


_CASES = {"transform": _case_transform, "executor": _case_executor}


def _run_worker(case_name: str, ndev: int = 4, timeout: int = 900) -> dict:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={ndev}").strip()
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["PYTHONPATH"] = src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "worker", case_name],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker {case_name} failed rc={res.returncode}\nSTDOUT:\n"
        f"{res.stdout}\nSTDERR:\n{res.stderr}")
    lines = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(lines[-1])


def test_tiled_rpoints_transform_is_bit_identical():
    """Each k row is an independent box gather + 3-D IFFT + phase: exact."""
    out = _run_worker("transform")
    assert out.pop("non_divisor_refused"), "a non-divisor k tile must refuse"
    assert all(v for k, v in out.items() if k.endswith("_equal")), out


def test_streamed_parent_executor_tiles_k_bit_identically():
    out = _run_worker("executor")
    assert out["streamed_calls"] > 0, out
    # The fixture must actually tile: 1-row tiles of the 3 raw parents.
    assert all(tile < rows for tile, rows in out["tiles"]), out
    assert out["all_equal"], out
    assert all(v < 1.0e-10 for v in out["max_rel_vs_dense"].values()), out


def test_zeta_k_tile_rule_is_the_centroid_bound_snapped_to_a_divisor():
    from gw.gflat_memory_model import zeta_fft_k_tile
    rule = zeta_fft_k_tile
    assert rule(n_k_rows=256, band_chunk=64, p_band=64) == 64
    assert rule(n_k_rows=256, band_chunk=512, p_band=64) == 64
    assert rule(n_k_rows=30, band_chunk=64, p_band=64) == 30
    assert rule(n_k_rows=30, band_chunk=16, p_band=16) == 15
    assert rule(n_k_rows=36, band_chunk=16, p_band=16) == 12
    assert rule(n_k_rows=37, band_chunk=16, p_band=16) == 1


def _cri3_bispinor_p64_plan():
    """Production inputs of JID 58786388 step .0 (the charge fit).

    Calibrated to its refusal receipt: persistent 6.00, B 40.34, E 8.42,
    F 3.85 GB/dev (mu_pad 4032 = 3998 on the 8x8 mesh, ngkmax 96000,
    face carrier 488, fit window 481, 30 raw parents / selected q).
    """
    from gw.gw_init import _plan_gflat_chunks_for_channel
    from gw.wavefunction_bundle import BandSlices

    meta = SimpleNamespace(
        nk_tot=256, nspinor=4, n_rmu=3998, n_rmu_padded=4032,
        n_rtot=80 * 80 * 250, fft_grid=(80, 80, 250))
    cfg = SimpleNamespace(
        zeta_nband=481,
        memory=SimpleNamespace(
            per_device_gb=70.32, chunk_target_utilization=0.0,
            band_chunk_size=16, r_chunk_override=0, gflat_chunk_size=0,
            low_mem_bands=False),
        backend=SimpleNamespace(distributed_zeta_solve="auto"))
    mesh = SimpleNamespace(
        shape={'x': 8, 'y': 8}, devices=np.empty(64, dtype=object))
    _, plan = _plan_gflat_chunks_for_channel(
        meta=meta, cfg=cfg,
        band_slices=BandSlices.from_band_edges(0, 0, 130, 183, 488),
        mesh_xy=mesh, is_bispinor=True, n_q_selected=30,
        parent_route=dict(n_parent=30, parents_only=True),
        print_fn=lambda *_: None)
    return plan


def test_cri3_bispinor_charge_fit_plans_at_p64():
    """Analytic (4x, conservative) FFT pricing; no refusal is raised."""
    plan = _cri3_bispinor_p64_plan()
    assert not plan.cache_psi_r
    assert abs(plan.persistent_bytes - 6.00e9) < 0.005e9  # the receipt's
    assert plan.zeta_k_chunk == 30
    # 30 k x 1 band/rank x 4 spinor x 1.6e6 r x 16 B x 4.0 analytic.
    assert plan.zeta_transform_fft_bytes == 12_288_000_000
    assert plan.p_min <= 64
    assert plan.hwm_bytes <= plan.budget_bytes * plan.target_utilization
    # A usable r chunk: at least the mu-wide performance floor.
    assert plan.r_chunk >= 4032 and plan.n_r_chunks <= 400, plan.format()
    assert "zeta FFT      = k_tile 30 (streamed)" in plan.format()


def test_executor_and_planner_share_the_one_tile_rule():
    repo = Path(__file__).resolve().parents[1]
    core = ast.parse((repo / "src/isdf/core.py").read_text())
    fn = next(n for n in ast.walk(core)
              if isinstance(n, ast.FunctionDef) and n.name == "_z_q_face_parent")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    assert any(isinstance(c.func, ast.Name) and c.func.id == "zeta_fft_k_tile"
               for c in calls)
    streamed = [c for c in calls if isinstance(c.func, ast.Name)
                and c.func.id == "to_rpoints_inner"]
    assert len(streamed) == 1
    assert any(kw.arg == "k_tile" and ast.unparse(kw.value) == "zeta_k_tile"
               for kw in streamed[0].keywords)
    model = ast.parse((repo / "src/gw/gflat_memory_model.py").read_text())
    planner = next(n for n in model.body
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "plan_gflat_chunks")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "zeta_fft_k_tile" for n in ast.walk(planner))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "worker":
        print(json.dumps(_CASES[sys.argv[2]]()))
        sys.exit(0)
    raise SystemExit("usage: python test_zeta_fft_ktile.py worker <case>")
