"""TDGW reads ``tdagw.kernel_export.write_tdgw_kernel``'s file back as the kernel written.

A random kernel K (Ry) of a 2v x 2c window on four k points is written and read
with TDGW's own ``TDGW.td.io.read_kernel`` (extended, nblocks = 4).  TDGW's
convention is K^TDGW = −kernel/2 Ha, so the gate is ``read == K / 2``; the
red twins write a sign-flipped kernel, one with the valence pair left in
ascending order and one with k and k' swapped, and each must fail the same
check (or be refused by TDGW).  The bridge's parallel-h5py gate is checked
beside it: a serial h5py refuses under more than one TDGW rank and passes on
one, and a bridge with no check (the twin) must fail to refuse.

Runs as a script where TDGW is importable (one process)::

    python3 tests/test_tdagw_kernel_export.py

and refuses by name where it is not.  Under pytest it skips.
"""
import os
import sys
import tempfile

import numpy as np

TOL = 1e-15


def _read_with_tdgw(path, ibndmin, ibndmax, ivbm):
    from TDGW.common import basis
    from TDGW.td.io import read_kernel
    from tensor2.constants import TENSOR_DATA_NP
    k = read_kernel(path, ivbm=ivbm, bands_b=basis.SequentialBasis(ibndmin, ibndmax + 1, 1),
                    grid=None)
    if k.is_TDA:
        raise AssertionError("an nblocks = 4 file was read as TDA")
    return np.asarray(k.kernel_knn_knn.get_data(TENSOR_DATA_NP))


def run_checks(tmp):
    from tdagw.kernel_export import write_tdgw_kernel
    rng = np.random.default_rng(7)
    nk, nb, ibndmin, ivbm = 4, 4, 129, 130
    K = rng.standard_normal((nk, nb, nb) * 2) + 1j * rng.standard_normal((nk, nb, nb) * 2)
    kpts = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]], dtype=float)
    rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    out, twins = {}, {}
    p = os.path.join(tmp, "kernel.h5")
    write_tdgw_kernel(p, K, kpts=kpts, ibndmin=ibndmin, ivbm=ivbm, sys_dim=2, nspinor=2)
    out["TDGW read == K/2 (Ha)"] = rel(_read_with_tdgw(p, ibndmin, ibndmin + nb - 1, ivbm), K / 2)
    # The twins write a kernel carrying one convention error each; a TDGW refusal
    # to read it counts as caught (recorded as inf).
    v_unreversed = np.array([1, 0, 2, 3])       # the valence pair left in ascending order
    wrong = {
        "sign flipped": -K,
        "valence order not reversed": K[:, v_unreversed][:, :, v_unreversed][
            :, :, :, :, v_unreversed][..., v_unreversed],
        "k and k' axes swapped": np.transpose(K, (3, 1, 2, 0, 4, 5)),
    }
    for name, Kw in wrong.items():
        write_tdgw_kernel(p, Kw, kpts=kpts, ibndmin=ibndmin, ivbm=ivbm, sys_dim=2, nspinor=2)
        try:
            twins[name] = rel(_read_with_tdgw(p, ibndmin, ibndmin + nb - 1, ivbm), K / 2)
        except Exception as e:                  # noqa: BLE001 - a refusal is a catch
            print(f"     twin {name}: TDGW refused ({type(e).__name__})", flush=True)
            twins[name] = float("inf")
    # The bridge's parallel-h5py gate: TDGW writes collectively at P > 1, so a serial h5py
    # must refuse there (and only there); a missing check is the twin.
    from tdagw.tdgw_bridge import require_parallel_h5py

    def refuses(check, n_ranks, mpi):
        try:
            check(n_ranks, mpi)
        except RuntimeError as e:
            return "GATE tdgw_bridge_h5py_mpi" in str(e)
        return False
    out["bridge refuses serial h5py, 4 ranks"] = 0.0 if refuses(require_parallel_h5py, 4, False) else 1.0
    out["bridge admits serial h5py, 1 rank"] = 0.0 if not refuses(require_parallel_h5py, 1, False) else 1.0
    out["bridge admits MPI h5py, 4 ranks"] = 0.0 if not refuses(require_parallel_h5py, 4, True) else 1.0
    twins["no h5py check, 4 serial ranks"] = 0.0 if refuses(lambda n, m: None, 4, False) else 1.0
    return out, twins


def main() -> int:
    try:
        import TDGW  # noqa: F401
    except ImportError as e:
        print(f"REFUSED: TDGW is not importable here ({e}); this gate needs TDGW's own reader",
              flush=True)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        out, twins = run_checks(tmp)
    ok = all(v <= TOL for v in out.values()) and all(v > 1e-3 for v in twins.values())
    for k, v in out.items():
        print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:34s} {v:.3e}", flush=True)
    for k, v in twins.items():
        print(f"{'ok  ' if v > 1e-3 else 'FAIL'} twin {k:29s} {v:.3e} (must exceed 1e-3)", flush=True)
    print(f"TDAGW_KERNEL_EXPORT_GATE {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def test_tdagw_kernel_export_needs_tdgw():
    """Collected by pytest: TDGW is not a LORRAX dependency, so this runs as a script."""
    import pytest
    pytest.skip("run as a script where TDGW is installed: python3 tests/test_tdagw_kernel_export.py")


if __name__ == "__main__":
    sys.exit(main())
