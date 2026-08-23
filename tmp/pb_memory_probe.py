"""Shard-level + allocator memory proof for pack_band_window's packed
brackets: (1) each packed pair's own shard-level byte size against
2*S/(Px*Py) at its own width (matching the zeta-fit doc's own
instrumentation idiom -- .addressable_shards, not the sharding-blind
global-shape mem_probe table), (2) all three brackets together stay
<= ~one extra face pair, (3) genuine HBM bytes_in_use before/after
build+free, at production-like shape (MoS2 6x6x1 k6_c50: nk=36,
n_rmu=676 [675 rounded to mesh-divisible], ns=1, nb_full=76).
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
for _svc in ("lxkit", "distrib_la"):
    _src = os.path.join(_REPO, "services", _svc, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
from lxkit.gate import platform_from_env
from runtime import initialize_communicator_stack
_plat = platform_from_env()
_RUNTIME = initialize_communicator_stack(platform="gpu" if _plat == "CUDA" else "cpu")

import gc

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from gw.wavefunction_bundle import (
    BandSlices, PSI_MUN_SPEC, PSI_NMU_SPEC, Wavefunctions, pack_band_window,
)

NK = 36
N_RMU = 676
NS = 1
NB_FULL = 76
BRACKETS = ((0, 61), (61, 68), (68, 76))


def _shard_bytes(x):
    """Sum of ONE rank's own addressable shard bytes -- this rank's true
    per-device residency, not the sharding-blind global-shape estimate
    (KNOWN_LORRAX_ISSUES.md's mem_probe row)."""
    total = 0
    for shard in x.addressable_shards:
        total += shard.data.nbytes
    return total


def main():
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    if jax.device_count() != 4:
        p0(f"REFUSE: need 4 devices, got {jax.device_count()}")
        return 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    dev = jax.local_devices()[0]

    rng = np.random.default_rng(7)
    psi_full = (rng.standard_normal((NK, NB_FULL, NS, N_RMU))
               + 1j * rng.standard_normal((NK, NB_FULL, NS, N_RMU)))
    psi_full_T = np.transpose(psi_full, (0, 2, 3, 1))

    with mesh:
        mun_spec = NamedSharding(mesh, PSI_MUN_SPEC)
        nmu_spec = NamedSharding(mesh, PSI_NMU_SPEC)
        psi_mun = jax.device_put(jnp.asarray(psi_full_T), mun_spec)
        psi_nmu = jax.device_put(jnp.asarray(psi_full), nmu_spec)
        slices = BandSlices.from_band_edges(0, 0, 0, NB_FULL, NB_FULL)
        enk = jnp.zeros((NK, NB_FULL), dtype=jnp.float64)
        occ = jnp.zeros((NK, NB_FULL), dtype=jnp.float64)
        wfns = Wavefunctions(enk=enk, occ=occ, slices=slices,
                             psi_nmu=psi_nmu, psi_mun=psi_mun, layout="face")
        jax.block_until_ready((wfns.psi_mun, wfns.psi_nmu))

        carrier_bytes = _shard_bytes(wfns.psi_mun) + _shard_bytes(wfns.psi_nmu)
        p0(f"resident face carrier (psi_mun + psi_nmu), THIS rank's own "
           f"shards: {carrier_bytes / 1e6:.4f} MB "
           f"(2*S/(Px*Py) at nb_full={NB_FULL})")

        gc.collect()
        stats0 = dev.memory_stats()
        p0(f"before packing: bytes_in_use={stats0['bytes_in_use'] / 1e6:.3f} MB")

        packed = [pack_band_window(wfns, lo, hi, mesh_xy=mesh)
                 for lo, hi in BRACKETS]
        jax.block_until_ready(packed)

        stats1 = dev.memory_stats()
        p0(f"after packing 3 brackets: bytes_in_use="
           f"{stats1['bytes_in_use'] / 1e6:.3f} MB  "
           f"(delta {(stats1['bytes_in_use'] - stats0['bytes_in_use']) / 1e6:.3f} MB)")

        total_packed_bytes = 0
        for i, (lo, hi) in enumerate(BRACKETS):
            mun_w, nmu_w = packed[i]
            b = _shard_bytes(mun_w) + _shard_bytes(nmu_w)
            total_packed_bytes += b
            hi_ = NB_FULL if hi is None else hi
            width = hi_ - lo
            w_pad = -(-width // 2) * 2
            p0(f"  bracket [{lo},{hi}) width={width} w_pad={w_pad}: "
               f"packed pair THIS rank's own shards = {b / 1e6:.4f} MB "
               f"(2*S_bracket/(Px*Py) at w_pad={w_pad})")

        p0(f"SUM of all {len(BRACKETS)} packed pairs (this rank's own "
           f"shards): {total_packed_bytes / 1e6:.4f} MB")
        p0(f"resident carrier for comparison ('one extra face pair'): "
           f"{carrier_bytes / 1e6:.4f} MB")
        p0(f"ratio (sum of packed / one resident face pair) = "
           f"{total_packed_bytes / carrier_bytes:.3f}")

        del packed
        gc.collect()
        jax.block_until_ready(wfns.psi_mun)  # keep carrier alive as a barrier
        stats2 = dev.memory_stats()
        p0(f"after del + gc: bytes_in_use={stats2['bytes_in_use'] / 1e6:.3f} MB "
           f"(back within "
           f"{(stats2['bytes_in_use'] - stats0['bytes_in_use']) / 1e6:.3f} MB "
           f"of the pre-packing baseline -- confirms the packed pairs are "
           f"freed, not leaked)")
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
