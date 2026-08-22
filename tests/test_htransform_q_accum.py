"""The streamed-Galerkin Q accumulator: memory ledger, mesh refusal, and
the explicit band->r transition (2026-08-22).

Register rows closed here (sandbox KNOWN_LORRAX_ISSUES 2026-08-19):

* ``htransform.py streaming_galerkin_solve`` priced only the streamed psi
  band chunk while the full-r ``Q[rank, ns, n_rtot]`` accumulator stayed
  live across all band chunks — CrI3 81k/160b/6800c at P=4 advertised
  3.39 GB/chunk while XLA priced Q at 133.98 GiB/device (JID 57269074).
  ``galerkin_q_ledger`` now charges Q (x2 donated in/out overlap), both
  transition layouts of the psi chunk, the replicated Gram state, Vh, G
  and the psi(G-flat) window, and ``_refuse_unfit_galerkin_mesh``
  refuses a non-fitting mesh BEFORE compilation, naming the smallest
  square mesh that fits.
* The band-sharded psi slab entered the accumulation contraction with no
  explicit reshard; the legacy SPMD partitioner fell back to fully
  replicating a 54.3-GiB slab per GPU at P16 (JID 57271407,
  "Involuntary full rematerialization").  ``_make_to_r_kernel`` is now
  the ONE explicit band->r all-to-all, and ``_make_accum_kernel`` pins
  its psi input sharding to Q's fitted r layout.

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
    led_s = ht.galerkin_q_ledger(
        p_total=named * named,
        q_shards=ht._square_q_shards(n_rtot, named),
        y_shards=(named if n_rtot % named == 0 else 1), **kw)
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


def test_square_q_shards_matches_the_fitter_arithmetic():
    ht = _import_ht()
    # 27000 = 2^3 3^3 5^3 — the cohsex fixture extent from the sharding_fit
    # history: indivisible by 64, divisible by 8.
    assert ht._square_q_shards(27000, 8) == 8
    assert ht._square_q_shards(27000, 4) == 4    # 16 does not divide; 4 does
    assert ht._square_q_shards(27000, 7) == 1
    assert ht._square_q_shards(4096, 4) == 16


# ---------------------------------------------------------------------------
# The multi-device transition twin (4 forced CPU devices, subprocess)
# ---------------------------------------------------------------------------

def _worker_transition() -> int:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from bandstructure import htransform as ht
    from common.sharding_fit import fit_sharding, shard_factor

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0
    mesh = Mesh(np.asarray(devs[:_NDEV]).reshape(2, 2), ("x", "y"))

    rank, nk, w, ns, n_rtot = 8, 3, 4, 2, 16     # n_rtot % 4 == 0
    rep = NamedSharding(mesh, P())
    sharding_q = fit_sharding(mesh, P(None, None, ("x", "y")),
                              (rank, ns, n_rtot), "twin.Q")
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
        to_r = ht._make_to_r_kernel(mesh, psi_layout)
        psi_r = to_r(psi_dev)
        layout_ok &= (psi_r.sharding.spec == psi_layout.spec)
        accum = ht._make_accum_kernel(rank, w, ns, mesh, rep,
                                      psi_layout, sharding_q)
        Q = accum(jax.device_put(jnp.asarray(UH), rep),
                  jax.device_put(jnp.asarray(inv_s), rep), psi_r, Q)
        Q_ref += inv_s * (UH @ psi.reshape(nk * w, ns * n_rtot))
    q_layout_ok = (Q.sharding.spec == sharding_q.spec)

    fold = jax.jit(
        lambda Q_, G_: G_ + jnp.einsum("asr,bsr->ab", Q_, jnp.conj(Q_),
                                       optimize=True),
        donate_argnums=(0, 1), out_shardings=grid_xy)
    G = fold(Q, G)
    G_ref = Q_ref @ Q_ref.conj().T

    q_err = 0.0  # Q was donated into the fold; compare G only.
    g = np.asarray(jax.device_get(G))
    g_err = float(np.max(np.abs(g - G_ref)) / np.max(np.abs(G_ref)))
    print(json.dumps({"g_rel_err": g_err, "q_rel_err": q_err,
                      "psi_layout_ok": bool(layout_ok),
                      "q_layout_ok": bool(q_layout_ok),
                      "q_entry": str(q_entry),
                      "q_shards": shard_factor(mesh, q_entry)}))
    return 0


def _run_worker(tag: str, timeout: int = 900):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={_NDEV}"
                        ).strip()
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
    """The explicit band->r all-to-all + pinned-layout accumulation
    reproduce the flattened reference sum, land psi in Q's exact fitted
    layout, and keep Q in that layout.  Scope: 4 forced CPU devices, one
    process; the production multi-process GPU leg is the lx-test/lx-run
    P=4 gate."""
    _import_ht()   # skip early if FFI is absent in THIS interpreter too
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
