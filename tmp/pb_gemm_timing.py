"""Microbenchmark: does a bracket's G-build GEMM genuinely cost less at its
own (packed) width than at the shared full nb_full width the mask route
always pays?  Production-scale mu/nb/nk (MoS2 6x6x1 k6_c50 deck: nk=36,
n_rmu=675, nb_full=76), real 4-rank CUDA (2x2 mesh) -- the SAME shape the
task's production A/B leg runs at, isolated from fit/screening noise.
"""
import os
import sys
import time

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

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from distrib_la import gemm_plan

NK = 36
N_RMU = 676  # 675 rounded to the nearest mesh-divisible mu_s (n_rmu*ns)
NS = 1
NB_FULL = 76
BRACKETS = ((0, 61), (61, 68), (68, 76))
N_REPEAT = 20


def main():
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    if jax.device_count() != 4:
        p0(f"REFUSE: need 4 devices, got {jax.device_count()}")
        return 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    mu_s = N_RMU * NS

    with mesh:
        p0(f"shape: nk={NK} n_rmu={N_RMU} ns={NS} nb_full={NB_FULL} "
           f"mu_s={mu_s} mesh=2x2")

        def _time_plan(width, label):
            plan = gemm_plan(mesh, m=mu_s, k=width, n=mu_s, nq=NK,
                             dtype=jnp.complex128)
            p0(f"  {label}: {plan.describe()}")
            A = jnp.zeros((NK, mu_s, width), dtype=jnp.complex128)
            B = jnp.zeros((NK, width, mu_s), dtype=jnp.complex128)
            with mesh:
                A = jax.device_put(A, plan.in_sharding_a)
                B = jax.device_put(B, plan.in_sharding_b)
            # warm (already warmed by gemm_plan's own construction, but one
            # more call ensures dispatch-side python overhead is amortized
            # before timing starts)
            jax.block_until_ready(plan(A, B))
            t0 = time.perf_counter()
            for _ in range(N_REPEAT):
                out = plan(A, B)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            per_call = (t1 - t0) / N_REPEAT
            p0(f"  {label}: {per_call * 1e3:.3f} ms/call "
               f"({N_REPEAT} calls, {t1 - t0:.3f} s total)")
            return per_call

        t_full = _time_plan(NB_FULL, "full nb_full (mask route, every bracket)")
        per_bracket = []
        for lo, hi in BRACKETS:
            width = hi - lo
            w_pad = -(-width // 2) * 2
            t = _time_plan(w_pad, f"bracket [{lo},{hi}) padded to {w_pad}")
            per_bracket.append(t)
        sum_packed = sum(per_bracket)
        sum_mask = 3 * t_full
        p0("")
        p0(f"SUMMARY: mask route pays full-width 3x: "
           f"3 * {t_full * 1e3:.3f} ms = {sum_mask * 1e3:.3f} ms")
        p0(f"         packed route pays each bracket's own width: "
           f"sum = {sum_packed * 1e3:.3f} ms "
           f"({', '.join(f'{t * 1e3:.3f}' for t in per_bracket)} ms)")
        p0(f"         packed / mask ratio = {sum_packed / sum_mask:.3f} "
           f"(1.0 = no saving; <1.0 = packed route is cheaper)")
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
