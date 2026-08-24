"""The streamed-Galerkin Q accumulator: memory ledger, mesh refusal, and
the explicit band->r transition (2026-08-22).

Register rows closed here (sandbox KNOWN_LORRAX_ISSUES 2026-08-19):

* ``htransform.py streaming_galerkin_solve`` priced only the streamed psi
  band chunk while the full-r ``Q[rank, ns, n_rtot]`` accumulator stayed
  live across all band chunks — CrI3 81k/160b/6800c at P=4 advertised
  3.39 GB/chunk while XLA priced Q at 133.98 GiB/device (JID 57269074).
  ``galerkin_q_ledger`` now charges Q (x2 donated in/out overlap), both
  equal-volume transition shards of the psi chunk, replicated Gram state, Vh, G
  and the psi(G-flat) window, and ``_refuse_unfit_galerkin_mesh``
  refuses a non-fitting mesh BEFORE compilation, naming the smallest
  square mesh that fits.
* The generic product-band → r-on-y reshard fully rematerialised the carrier.
  The production route now uses ``common.staged_reshard``'s two explicit,
  volume-preserving all-to-alls and ``_make_accum_kernel`` pins its psi input
  to the resulting product-r layout.

The multi-device transition twin runs in a 4-CPU-device subprocess (the
house pattern of ``test_transverse_rank_truncate``).  Importing
``bandstructure.htransform`` needs the FFI host library, so every cell
skips cleanly where it is not built (login nodes); the P=4 GPU leg is the
compute-node ``lx test`` run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

_NDEV = 4


def _import_ht():
    pytest.importorskip("jax")
    try:
        from bandstructure import htransform as ht
    except RuntimeError as exc:  # FFI host library not built here
        if "FFI" in str(exc) or "liblorrax" in str(exc):
            pytest.skip(f"FFI host library unavailable: {exc}")
        raise
    return ht


# ---------------------------------------------------------------------------
# Ledger arithmetic (pure shape algebra, but lives behind the ht import)
# ---------------------------------------------------------------------------

def test_the_ledger_charges_q_not_just_the_chunk():
    """The CrI3 geometry that produced the 3.39 GB/chunk vs 133.98
    GiB/device escape: the ledger's Q term must dominate and reproduce
    XLA's own figure (rank*ns*n_rtot*16/q_shards)."""
    ht = _import_ht()
    rank, nk, ns, n_rtot = 12960, 81, 2, 1_406_256
    led = ht.galerkin_q_ledger(
        rank=rank, nk=nk, nspinor=ns, n_rtot=n_rtot, band_chunk=16,
        m_states=nk * 160, mu_pad=6800, psi_win_elems=nk * 160 * ns * 90_000,
        p_total=4, q_shards=4, y_shards=2)
    q_dev = rank * ns * n_rtot * 16 / 4          # XLA's 133.98 GiB class
    assert led["Q accumulator (x2 donated in/out overlap)"] == pytest.approx(
        2.0 * q_dev)
    # Q dominates every other row on this geometry.
    others = {k: v for k, v in led.items()
              if k not in ("TOTAL", "Q accumulator (x2 donated in/out "
                           "overlap)")}
    assert all(v < q_dev for v in others.values()), others
    # And the old banner's unit — one streamed band, the "3.39 GB/chunk"
    # line — under-advertised the live set by more than an order of
    # magnitude on this geometry.
    bytes_per_band = nk * ns * n_rtot * 16
    assert led["TOTAL"] > 50 * bytes_per_band


def test_refusal_names_a_fitting_square_mesh(monkeypatch):
    """A pool smaller than the ledger refuses BEFORE compilation and
    names the smallest square mesh that fits — the register's required
    message shape."""
    ht = _import_ht()
    import jax
    from jax.sharding import Mesh, PartitionSpec as P
    import common.gpu_utils as gpu_utils

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    rank, nk, ns, n_rtot = 512, 8, 2, 4096
    kw = dict(rank=rank, nk=nk, nspinor=ns, n_rtot=n_rtot, band_chunk=8,
              m_states=nk * 16, mu_pad=256, psi_win_elems=nk * 16 * ns * 512)
    led = ht.galerkin_q_ledger(p_total=1, q_shards=1, y_shards=1, **kw)
    # Pool sized so the 1x1 ledger overflows but a wider square mesh fits.
    limit = int(led["TOTAL"] * 0.5)
    monkeypatch.setattr(gpu_utils, "_get_jax_gpu_memory_bytes",
                        lambda: (float(limit), 0.0, float(limit)))
    lines = []
    with pytest.raises(ValueError) as err:
        ht._refuse_unfit_galerkin_mesh(
            led, mesh_xy=mesh, q_spec=P(None, None, ("x", "y")),
            log_fn=lines.append, **kw)
    msg = str(err.value)
    assert "does not fit" in msg and "GiB pool" in msg
    assert "smallest square mesh that fits" in msg, msg
    # The named mesh really does fit under the same arithmetic.
    import re
    m = re.search(r"pool size is (\d+)x(\d+)", msg)
    assert m and m.group(1) == m.group(2), msg
    named = int(m.group(1))
    from runtime.padding import round_up
    kw_s = {**kw, "n_rtot": round_up(n_rtot, named * named)}
    led_s = ht.galerkin_q_ledger(
        p_total=named * named, q_shards=named * named,
        y_shards=named, **kw_s)
    assert led_s["TOTAL"] <= limit


def test_refusal_gate_announces_when_it_cannot_run(monkeypatch):
    """A gate reporting nothing must be distinguishable from a gate that
    checked nothing (TASTE 2026-08-15): with no readable pool the ledger
    prints and the refusal announces itself OFF instead of passing
    silently."""
    ht = _import_ht()
    import jax
    from jax.sharding import Mesh, PartitionSpec as P
    import common.gpu_utils as gpu_utils

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    kw = dict(rank=64, nk=4, nspinor=1, n_rtot=64, band_chunk=4,
              m_states=16, mu_pad=32, psi_win_elems=1024)
    led = ht.galerkin_q_ledger(p_total=1, q_shards=1, y_shards=1, **kw)
    monkeypatch.setattr(gpu_utils, "_get_jax_gpu_memory_bytes",
                        lambda: (None, None, None))
    lines = []
    ht._refuse_unfit_galerkin_mesh(
        led, mesh_xy=mesh, q_spec=P(None, None, ("x", "y")),
        log_fn=lines.append, **kw)
    assert any("DID NOT RUN" in ln for ln in lines)


# ---------------------------------------------------------------------------
# The multi-device transition twin (4 forced CPU devices, subprocess)
# ---------------------------------------------------------------------------

def _twin_body() -> dict:
    """The transition twin proper — shared by the in-process (4 real
    devices, the lx-test GPU leg) and subprocess (4 forced host devices)
    arms, so both drive the SAME production kernels."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from bandstructure import htransform as ht
    from common.sharding_fit import shard_factor
    from common.staged_reshard import band_to_product_r_reshard

    devs = jax.devices()
    if len(devs) < _NDEV:
        return {"skip": f"only {len(devs)} devices"}
    mesh = Mesh(np.asarray(devs[:_NDEV]).reshape(2, 2), ("x", "y"))

    rank, nk, w, ns, n_rtot = 8, 3, 4, 2, 16     # n_rtot % 4 == 0
    rep = NamedSharding(mesh, P())
    sharding_q = NamedSharding(mesh, P(None, None, ("y", "x")))
    q_entry = sharding_q.spec[2]
    psi_layout = NamedSharding(mesh, P(None, None, None, q_entry))

    rng = np.random.default_rng(20260822)
    inv_s = rng.standard_normal((rank, 1))
    grid_xy = NamedSharding(mesh, P("x", "y"))

    # Two band chunks accumulated into one Q, then folded into G —
    # against a plain numpy reference of the same sum.
    Q = jax.jit(lambda: jnp.zeros((rank, ns, n_rtot), dtype=jnp.complex128),
                out_shardings=sharding_q)()
    G = jax.jit(lambda: jnp.zeros((rank, rank), dtype=jnp.complex128),
                out_shardings=grid_xy)()
    Q_ref = np.zeros((rank, ns * n_rtot), dtype=np.complex128)
    band_sh = NamedSharding(mesh, P(None, ("x", "y"), None, None))
    layout_ok = True
    for _chunk in range(2):
        UH = (rng.standard_normal((rank, nk * w))
              + 1j * rng.standard_normal((rank, nk * w)))
        psi = (rng.standard_normal((nk, w, ns, n_rtot))
               + 1j * rng.standard_normal((nk, w, ns, n_rtot)))
        # The loader hands the slab BAND-sharded over ('x','y').
        psi_dev = jax.device_put(jnp.asarray(psi), band_sh)
        to_r = band_to_product_r_reshard(mesh)
        psi_r = to_r(psi_dev)
        layout_ok &= (psi_r.sharding.spec == psi_layout.spec)
        accum = ht._make_accum_kernel(rank, w, ns, mesh, rep,
                                      psi_layout, sharding_q)
        Q = accum(jax.device_put(jnp.asarray(UH), rep),
                  jax.device_put(jnp.asarray(inv_s), rep), psi_r, Q)
        Q_ref += inv_s * (UH @ psi.reshape(nk * w, ns * n_rtot))
    q_layout_ok = (Q.sharding.spec == sharding_q.spec)

    # Force the production fold through several bounded r tiles, including a
    # one-column tail, on this tiny fixture.  Restore the process-global knob
    # so another in-process caller cannot inherit the test ceiling.
    old_fold_tile_bytes = ht._FOLD_Q_TILE_BYTES
    try:
        ht._FOLD_Q_TILE_BYTES = 768
        fold = ht._make_fold_G_kernel(rank, mesh, sharding_q, grid_xy)
        G = fold(Q, G)
    finally:
        ht._FOLD_Q_TILE_BYTES = old_fold_tile_bytes
    G_ref = Q_ref @ Q_ref.conj().T

    g = np.asarray(jax.device_get(G))
    g_err = float(np.max(np.abs(g - G_ref)) / np.max(np.abs(G_ref)))
    return {"g_rel_err": g_err,
            "psi_layout_ok": bool(layout_ok),
            "q_layout_ok": bool(q_layout_ok),
            "q_entry": str(q_entry),
            "q_shards": shard_factor(mesh, q_entry)}


def _worker_transition() -> int:
    print(json.dumps(_twin_body()))
    return 0


def _run_worker(tag: str, timeout: int = 900):
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    # The production process model pins ONE GPU per process (runtime's
    # one-process-per-GPU rule), so the pytest interpreter never sees 4
    # devices.  The subprocess drops the pin: on a GPU node it takes the
    # node's 4 cards (tiny arrays — no preallocation, so the parent's
    # pool is undisturbed); elsewhere it forces 4 host CPU devices,
    # which needs the host FFI build and skips cleanly without it.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("LORRAX_GPU_DEVICE", None)
    try:
        import jax
        _gpu = jax.default_backend() in ("gpu", "cuda")
    except Exception:
        _gpu = False
    if _gpu:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                            + f" --xla_force_host_platform_device_count="
                              f"{_NDEV}").strip()
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), tag],
        env=env, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0 and ("liblorrax" in res.stderr
                                or "FFI" in res.stderr):
        pytest.skip("FFI host library unavailable in the worker")
    assert res.returncode == 0, (
        f"worker {tag} failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}")
    line = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert line, f"no JSON from worker.\nSTDOUT:\n{res.stdout}"
    return json.loads(line[-1])


def test_band_to_r_transition_matches_reference_at_p4():
    """The two staged band->r all-to-alls + pinned-layout accumulation
    reproduce the flattened reference sum, land psi in Q's exact product-r
    layout, and keep Q in that layout.  Scope: one process over 4
    devices — REAL GPUs when this interpreter already has >= 4 (the lx
    test leg), else 4 forced host CPU devices in a subprocess.  The
    multi-PROCESS P=4 leg is a driver run, not this cell."""
    _import_ht()   # skip early if FFI is absent in THIS interpreter
    import jax
    if len(jax.devices()) >= _NDEV:
        out = _twin_body()
    else:
        out = _run_worker("worker_transition")
    if "skip" in out:
        pytest.skip(out["skip"])
    assert out["psi_layout_ok"], "transition did not land psi in Q's layout"
    assert out["q_layout_ok"], "accumulated Q left its fitted layout"
    assert out["q_shards"] == 4, out
    assert out["g_rel_err"] < 1e-12, out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    if tag == "worker_transition":
        sys.exit(_worker_transition())
    sys.exit(2)
