"""Native bounded SUMMA C alias against the previous XLA product/add owner."""
from runtime import initialize_communicator_stack, run_main_and_finalize, rank0_print
R = initialize_communicator_stack()
import argparse
import importlib
import json
from pathlib import Path
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local, gather_to_host, rank0_transaction

owner = importlib.import_module('distrib_la._panel_matmul')
ap = argparse.ArgumentParser()
ap.add_argument('--output', required=True)
args = ap.parse_args()

def main():
    assert tuple(R.mesh.shape.values()) == (2, 2)
    rng = np.random.default_rng(1709)
    face = NamedSharding(R.mesh, P(None, 'x', 'y'))
    rep = NamedSharding(R.mesh, P())
    rows = []
    for k in (18, 20):  # three full panels, and a final narrower panel
        for dtype in (np.float64, np.complex128):
            a = rng.normal(size=(3, 12, k)).astype(dtype)
            b = rng.normal(size=(3, k, 16)).astype(dtype)
            if dtype == np.complex128:
                a += 1j * rng.normal(size=a.shape)
                b += 1j * rng.normal(size=b.shape)
            aa = device_put_process_local(a, face)
            bb = device_put_process_local(b, face)
            for weighted in (False, True):
                w = rng.uniform(.1, 1., (3, k)).astype(dtype)
                if dtype == np.complex128:
                    w += .1j * rng.normal(size=w.shape)
                ww = device_put_process_local(w, rep) if weighted else None
                for partner in (False, True):
                    kw = dict(mesh=R.mesh, panel_bytes=8064, weights=ww, partner=partner)
                    owner._interleaved_kernel.cache_clear()
                    native = owner.panel_matmul(aa, bb, **kw)
                    jax.block_until_ready(native)
                    owner._interleaved_kernel.cache_clear()
                    # Preserve the same bounded gather/weights/partner logic;
                    # only restore its previous local XLA product/add.
                    with patch.object(owner, '_panel_contraction', return_value=None):
                        previous = owner.panel_matmul(aa, bb, **kw)
                        jax.block_until_ready(previous)
                    nv = native if partner else (native,)
                    pv = previous if partner else (previous,)
                    error = max(float(np.max(np.abs(gather_to_host(n) - gather_to_host(p))) /
                                max(np.max(np.abs(gather_to_host(p))), 1e-30))
                                for n, p in zip(nv, pv))
                    assert error < 3e-13, (k, dtype, weighted, partner, error)
                    rows.append(dict(k=k, dtype=np.dtype(dtype).name, weighted=weighted,
                                     partner=partner, relative_error=error))
    rank0_print(json.dumps(rows), flush=True)
    rank0_transaction(args.output, stage='native SUMMA accumulator parity',
                      write=lambda: Path(args.output).write_text(json.dumps(rows, indent=2)+'\n'))
    return 0

run_main_and_finalize(main)
