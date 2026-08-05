"""Does the two-step constraint remove the global all-reduce in rotate_bands?

Prints every collective in the lowered HLO with its byte size, so the
3.36 GiB c128[nk,nb,ns,ngkmax] all-reduce is visible if it is still there.
"""
import os, re, sys
from runtime import initialize_communicator_stack, finalize_process
RUNTIME = initialize_communicator_stack()
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import resolve_mesh, process_rank
from common.mtxel_sweep import band_sphere_spec
from gw.qsgw_density import rotate_bands, band_rotation_spec

NK = int(os.environ.get("RP_NK", "16")); NB = int(os.environ.get("RP_NB", "640"))
NS = int(os.environ.get("RP_NS", "2")); NGK = int(os.environ.get("RP_NGK", "11008"))

def main():
    p0 = print if process_rank() == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    rng = np.random.default_rng(0)
    psi = jax.make_array_from_callback(
        (NK, NB, NS, NGK), NamedSharding(mesh, band_sphere_spec()),
        lambda idx: np.zeros(tuple(s.stop - s.start if isinstance(s, slice) else 1
                                   for s in idx), dtype=np.complex128))
    U = jax.make_array_from_callback(
        (NK, NB, NB), NamedSharding(mesh, band_rotation_spec()),
        lambda idx: np.zeros(tuple(s.stop - s.start if isinstance(s, slice) else 1
                                   for s in idx), dtype=np.complex128))
    p0(f"[rp] mesh=({px},{py}) psi={psi.shape} U={U.shape}")
    lowered = jax.jit(lambda a, b: rotate_bands(a, b, mesh=mesh)).lower(psi, U)
    txt = lowered.compile().as_text()
    def nbytes(shape_str):
        d = [int(x) for x in re.findall(r'\d+', shape_str)]
        n = 1
        for v in d: n *= v
        return n * 16 / 2**30
    worst = 0.0
    for line in txt.splitlines():
        for kind in ("all-reduce", "all-gather", "collective-permute",
                     "reduce-scatter", "all-to-all"):
            if f" {kind}(" in line or f"= {kind}" in line:
                m = re.search(r'c128\[[\d,]+\]', line)
                gb = nbytes(m.group(0)) if m else 0.0
                worst = max(worst, gb)
                p0(f"[rp]   {kind:19s} {m.group(0) if m else '?':32s} "
                   f"{gb:8.3f} GiB")
                break
    p0(f"[rp] LARGEST COLLECTIVE OPERAND: {worst:.3f} GiB/rank")
    return 0

if __name__ == "__main__":
    import traceback
    rc = 1
    try: rc = main()
    except BaseException:
        traceback.print_exc(); sys.stderr.flush(); sys.stdout.flush(); rc = 1
    finalize_process(rc)
