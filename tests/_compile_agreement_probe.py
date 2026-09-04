"""Real-process positive and negative controls for compile agreement.

The parent test launches four one-GPU processes.  ``identical`` presents the
same named jit and shape on every rank; ``divergent`` changes that shape with
``jax.process_index()`` and must be refused before backend compilation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def main(argv) -> int:
    kind = argv[1] if len(argv) > 1 else "identical"
    if kind not in ("identical", "divergent"):
        print(f"unknown probe kind {kind!r}", file=sys.stderr)
        return 2

    from runtime import init_jax_distributed, set_default_env

    set_default_env()
    init_jax_distributed()

    import jax
    import jax.numpy as jnp
    from common.jax_compile_cache import (
        compile_cache_stats, install_compile_agreement)

    install_compile_agreement()
    rank = int(jax.process_index())
    size = 32 + rank if kind == "divergent" else 32
    host_x = np.arange(size, dtype=np.float64)

    @jax.jit
    def compile_agreement_probe(x):
        return jnp.sin(x) + 2.0 * x

    value = compile_agreement_probe(host_x)
    value.block_until_ready()
    stats = compile_cache_stats()
    receipt = {
        "kind": kind,
        "rank": rank,
        "process_count": int(jax.process_count()),
        "checks": stats["compile_agreement_checks"],
        "fingerprint_s": stats["compile_fingerprint_secs"],
        "exchange_s": stats["compile_agreement_secs"],
        "per_module_overhead_s": (
            (stats["compile_fingerprint_secs"]
             + stats["compile_agreement_secs"])
            / max(1, stats["compile_agreement_checks"])),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "result": float(np.asarray(value).sum()),
    }
    print("COMPILE_AGREEMENT_PROBE " + json.dumps(receipt, sort_keys=True),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
