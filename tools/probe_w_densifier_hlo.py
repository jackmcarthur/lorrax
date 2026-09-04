"""HLO probe: prove ``bse_io.make_w_densifier`` compiles with NO all-gather.

WHAT THIS PROVES (that the AST gate cannot).  The gate
``tests/test_fft_shardmap_context.py`` bans the eager
``local_ifftn3``/``local_fftn3`` call form syntactically; the SEMANTIC claim
of audit P0-4 — the densifier never materializes a replicated N_μ²-class W
tile per rank — is a property of the compiled program.  This probe lowers
``make_w_densifier`` (both ``output='k'`` and ``output='R'``) on a 2×2
('x','y') mesh with the production sharding ``P('x','y',None,None,None)``,
then scans the optimized HLO for gather-class collectives
(``all-gather`` / ``all-to-all``) touching the W operand.  It also checks the
numbers: the sharded densifier must match the plain eager
ifftn→pad→fftn reference to ~1e-13.

USAGE (compute node, or any host where jax imports — NOT a Frontera login
node; do not srun this there):

    XLA_FLAGS="--xla_force_host_platform_device_count=4" \
        python3 tools/probe_w_densifier_hlo.py

Exit 0 = no gather-class collective in either program AND numeric parity;
exit 1 = a collective was found or parity failed (the HLO lines are
printed).  Single process, 4 host devices, arrays of a few MB — no
distributed init, no MPI.
"""
import os
import sys

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        f"{_xla_flags} --xla_force_host_platform_device_count=4".strip())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                   # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Collectives that would mean the (μ,ν) sharding was dropped and the full W
# gathered per rank.  (Plain FFT + pad on replicated k-axes needs none.)
_GATHER_OPS = ("all-gather", "all-to-all")


def main() -> int:
    from bse.bse_io import pad_W_R_to_grid, make_w_densifier

    devs = jax.devices()
    if len(devs) < 4:
        print(f"REFUSING: need >= 4 devices for a 2x2 mesh, have {len(devs)}. "
              f"Set XLA_FLAGS=--xla_force_host_platform_device_count=4")
        return 1
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))
    w_spec = P("x", "y", None, None, None)
    w_sh = NamedSharding(mesh, w_spec)

    n_mu, coarse, fine = 8, (2, 2, 1), (4, 4, 1)
    rng = np.random.default_rng(0)
    W_np = (rng.standard_normal((n_mu, n_mu, *coarse))
            + 1j * rng.standard_normal((n_mu, n_mu, *coarse)))
    W_q = jax.device_put(jnp.asarray(W_np), w_sh)

    # Eager single-process reference (legal here: one process, replicated
    # semantics) — the sharded densifier must reproduce it.
    W_R_ref = jnp.fft.ifftn(jnp.asarray(W_np), axes=(2, 3, 4), norm="ortho")
    W_R_ref = pad_W_R_to_grid(W_R_ref, fine)
    W_k_ref = jnp.fft.fftn(W_R_ref, axes=(2, 3, 4), norm="ortho")

    failed = False
    for output, ref in (("k", W_k_ref), ("R", W_R_ref)):
        fn = make_w_densifier(mesh, w_spec, fine, output=output)
        hlo = fn.lower(W_q).compile().as_text()
        bad = [ln.strip() for ln in hlo.splitlines()
               if any(op in ln for op in _GATHER_OPS)]
        out = fn(W_q)
        err = float(jnp.max(jnp.abs(out - ref)))
        ok_sh = out.sharding.is_equivalent_to(w_sh, out.ndim)
        print(f"[output={output!r}] gather-class HLO ops: {len(bad)}; "
              f"max|Δ| vs eager reference: {err:.3e}; "
              f"out sharding pinned: {ok_sh}")
        for ln in bad:
            print("    " + ln)
        if bad or err > 1e-12 or not ok_sh:
            failed = True

    print("FAIL: densifier gathers or diverges — audit P0-4 NOT closed"
          if failed else
          "OK: no all-gather/all-to-all in either program; numeric parity "
          "and output sharding hold — P0-4 semantic claim verified on this "
          "host (re-run on a multi-process compute-node mesh for the "
          "distributed statement)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
