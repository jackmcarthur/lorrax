"""P=4 gate for MPA's disk-bounded sample -> fit -> pole stream.

Run only through the Perlmutter multi-process launcher.  Five poles and
``N_mu=P+2`` on the required P=4 mesh make N_mu divisible by each mesh side
but not by the device product.  This is the hostile geometry that separates
SlabIO's per-axis minimum from the canonical in-memory μ carrier while also
exercising the production four-pole plus one-pole tail reads.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
for _service in ("symmetry_maps", "minimax"):
    sys.path.insert(0, os.path.join(ROOT, "services", _service, "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import barrier, process_count, process_rank, resolve_mesh  # noqa: E402
from file_io import mpa_store  # noqa: E402
from gw.mpa import fit_driver, sampling  # noqa: E402
from gw.ppm_tau_kernel import build_shared_w_tau  # noqa: E402
from runtime.padding import padded_mu_extent  # noqa: E402
from symmetry_maps import (centroid_source_map_and_wrap,  # noqa: E402
                           verify_centroid_orbit_closure)
from symmetry_maps import qirr_store as QS  # noqa: E402


N_POLES = 5
N_Q_FULL = 4
W_NAME = "Wc_qmunu_z"


def _fail(message):
    raise RuntimeError(f"[mpa_fit_stream rank={process_rank()}] {message}")


def _tables(n_mu):
    fft = np.asarray([n_mu, 1, 1], np.int64)
    cent = np.stack((np.arange(n_mu), np.zeros(n_mu), np.zeros(n_mu)), axis=1)
    sym = np.eye(3, dtype=np.int64)[None]
    tnp = np.zeros((1, 3), np.float64)
    verdict = verify_centroid_orbit_closure(
        cent.astype(np.float64) / fft, sym, tnp=tnp, fft_grid=fft)
    if not verdict.closed:
        _fail(verdict.describe())
    perm, wrap = centroid_source_map_and_wrap(
        cent, sym, tnp, fft, validate=True, extend_trs=True)
    tables = QS.QirrTables(
        irr_idx_q=np.zeros(N_Q_FULL, np.int32),
        sym_idx_q=np.zeros(N_Q_FULL, np.int32),
        q_irr_frac=np.zeros((1, 3), np.float64),
        sym_perm=perm, L_table=wrap, n_sym_spatial=1)
    return tables, verdict


def _planted(n_mu):
    a0 = np.asarray([0.38, 0.71, 1.19, 1.91, 3.08])
    gamma0 = np.asarray([0.045, 0.082, 0.14, 0.23, 0.39])
    mu, nu = np.meshgrid(np.arange(n_mu), np.arange(n_mu), indexing="ij")
    scale = 1.0 + 0.017 * (mu + 2 * nu) / max(n_mu - 1, 1)
    Omega = (a0[:, None, None] * scale
             - 1j * gamma0[:, None, None] * scale)
    amp = 0.7 + 0.11 * np.arange(N_POLES)[:, None, None]
    phase = 0.13 * mu[None] - 0.21 * nu[None] + 0.19 * np.arange(N_POLES)[:, None, None]
    B = amp * (1.0 + 0.03 * mu[None] + 0.02 * nu[None]) * np.exp(1j * phase)
    return Omega.astype(np.complex128), B.astype(np.complex128)


def _samples(Omega, B, z):
    denom = z[:, None, None, None] ** 2 - Omega[None] ** 2
    return np.sum(2.0 * Omega[None] * B[None] / denom, axis=1)


def _check_local(name, got, expected, *, rtol=0.0, atol=0.0):
    for shard in got.addressable_shards:
        local = np.asarray(shard.data)
        want = expected[shard.index]
        if not np.allclose(local, want, rtol=rtol, atol=atol):
            err = float(np.max(np.abs(local - want)))
            _fail(f"{name} shard {shard.index} max error {err:.3e}")


def main():
    world = process_count()
    if world != 4:
        _fail(f"requires exactly four processes, got {world}")
    mesh = resolve_mesh()
    n_mu = world + 2
    tables, verdict = _tables(n_mu)
    Omega_ref, B_ref = _planted(n_mu)
    z = sampling.double_parallel_grid(
        N_POLES, 4.0, material_class="insulator", alpha=1,
        energy_unit="Ha")
    W = _samples(Omega_ref, B_ref, z)
    line = np.asarray([0] * N_POLES + [1] * N_POLES, np.int32)
    root = os.environ.get("MPA_STREAM_GATE_DIR", os.getcwd())
    sample_path = os.path.join(root, "mpa_fit_stream_samples.h5")
    fit_path = os.path.join(root, "mpa_fit_stream_poles.h5")
    if process_rank() == 0:
        os.makedirs(root, exist_ok=True)
        for path in (sample_path, fit_path):
            if os.path.exists(path):
                os.remove(path)
    barrier("mpa_fit_stream_clean")

    mpa_store.allocate_w_omega_collective(
        sample_path, W_NAME, mesh_xy=mesh, n_omega=z.size,
        n_q_on_disk=1, n_mu=n_mu, tables=tables, omega=z,
        omega_line=line, closure_verdict=verdict,
        sampling={"protocol": "double_parallel", "varpi": [0.1, 1.0],
                  "n_p": N_POLES, "alpha": 1, "omega_max": 4.0},
        n_rmu_logical=n_mu, energy_unit="Ha", mode="w")
    spec = P(None, "x", "y")
    n_mu_padded = padded_mu_extent(n_mu, mesh)
    padded = (1, n_mu_padded, n_mu_padded)
    sharding = NamedSharding(mesh, spec)
    global_shape = (z.size, 1, n_mu, n_mu)
    for iz in range(z.size):
        host = np.zeros(padded, np.complex128)
        host[0, :n_mu, :n_mu] = W[iz][None]
        slab = jax.device_put(host, sharding)
        mpa_store.write_w_slab_collective(
            sample_path, W_NAME, iz, slab, mesh_xy=mesh,
            global_shape=global_shape)

    # This exact full-slab reload feeds the production Dyson solve.  At P=4,
    # n_mu=6 is already divisible by each 2-D mesh side, so the historical
    # per-axis reader returned 6 while every canonical μ carrier was padded
    # to 8.  Assert the product-padded carrier and its exact-zero pad before
    # the fit's independently tiled column reader can hide the mismatch.
    reloaded, _ = mpa_store.read_w_slab_collective(
        sample_path, W_NAME, 0, mesh_xy=mesh)
    if tuple(reloaded.shape) != padded:
        _fail(f"full-slab reload shape {tuple(reloaded.shape)} != "
              f"canonical carrier {padded}")
    expected_reload = np.zeros(padded, np.complex128)
    expected_reload[0, :n_mu, :n_mu] = W[0][None]
    _check_local("product-padded full-slab reload", reloaded,
                 expected_reload)
    del reloaded

    ledger, report = fit_driver.run_fit_driver(
        sample_path, W_NAME, fit_path, z, N_POLES, mesh_xy=mesh)
    if not ledger["complete"] or report["pade_solves"] != n_mu * n_mu:
        _fail(f"incomplete fit ledger/report: {ledger}, {report}")

    total = None
    widths = []
    t = np.float64(0.37)
    for lo, hi in ((0, 4), (4, 5)):
        Omega, B = mpa_store.read_poles(
            fit_path, pole_slice=slice(lo, hi), mesh_xy=mesh,
            unfold=True, return_sharded=True)
        widths.append(int(Omega.shape[0]))
        if Omega.is_fully_addressable or B.is_fully_addressable:
            _fail(f"pole range [{lo},{hi}) became fully addressable")
        width = hi - lo
        bounds = jnp.asarray(np.tile(
            [0.0, np.inf, 0.0, -np.inf, np.inf, np.inf], (width, 1)))
        Wt = jax.jit(build_shared_w_tau)(
            B, Omega, jnp.arange(width, dtype=jnp.int32), bounds,
            jnp.zeros(width, dtype=bool), jnp.asarray(0.0), jnp.asarray(t))
        total = Wt if total is None else total + Wt
    if widths != [4, 1]:
        _fail(f"pole stream widths are {widths}, expected [4, 1]")

    n_pad = int(total.shape[-1])
    expected = np.zeros((N_Q_FULL, n_pad, n_pad), np.complex128)
    temporal = np.sum(B_ref * np.exp(-1j * Omega_ref * t), axis=0)
    expected[:, :n_mu, :n_mu] = temporal
    scale = max(float(np.max(np.abs(expected))), 1.0)
    _check_local("temporal pole sum", total, expected,
                 rtol=2.0e-5, atol=2.0e-5 * scale)

    barrier("mpa_fit_stream_done")
    if process_rank() == 0:
        print(f"[mpa_fit_stream] PASS world=4 N_mu={n_mu} poles=4+1 "
              f"blocks={report['blocks_walked']} fit={report['seconds']['fit']:.3f}s",
              flush=True)
    return 0


if __name__ == "__main__":
    finalize_process(main())
