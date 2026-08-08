"""WfnLoader: phdf5 backend bit-equal vs eager — the cluster's argv door.

GRADUATED 2026-08-07.  This file used to be the ONLY multi-rank parity
instrument in the tree and it carried its own copy of the comparison, its
own defaults, and no anti-tautology guard.  Its defaults were
``--bands 0,4`` on a 4-rank mesh (a window that DIVIDES the world, so the
per-rank band clamp never fired), ``--atol 1e-12`` (a tolerance where the
contract is bit-equality), and a ``--wfn`` pointing at
``/pscratch/.../MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5`` on a machine that
no longer exists.  That combination is why the ``22049c3`` band-bound
divergence lived for months behind a green suite.

Now it is a THIN ARGV WRAPPER over
``services/wfn_loader/tests/test_wfn_loader_multiproc.py``'s ``check_*``
bodies — the same functions the pytest cells call and the same functions
``_cli_main`` runs under ``srun``.  ONE implementation, three callers.
The defaults it ships are the ones that would have caught the defect:

* ``--wfn`` defaults to the in-repo hostile deck
  ``tests/regression/gnppm_debug/WFN.h5`` (nrk 9, mnband 82, nspinor 2,
  ngkmax 1963, ngk 1917..1963), which survey w1_wfn_loader §6.4
  established is byte-size identical to the dead ``/pscratch`` file.  That
  path still works as an explicit override; whether to restage it is
  registered to the owner (Q7).
* ``--bands`` defaults to ``0,10`` — NON-DIVISIBLE on four ranks, giving
  per-rank clamped counts [3, 3, 3, 1].  ``check_band_pad_clamp_parity``
  ASSERTS ``(b_hi - b_lo) % world != 0`` before it opens a loader, so a
  future edit of this default cannot silently return the instrument to
  the geometry that hid the bug.
* ``--atol`` defaults to ``0.0`` and NOTHING ELSE IS ACCEPTED.  Both
  backends read the same f64 bytes off the same file and assemble them
  with the same unfold; a difference of one ulp is a difference in WHICH
  BYTES, not in rounding.  The flag survives so an old command line keeps
  parsing, and refuses rather than quietly measuring something weaker.

WHERE THIS FILE LIVES AND WHY.  ``tests/bench`` is in ``norecursedirs``
(pyproject.toml), so pytest never collects this script — it is an
argv-driven driver with a jax preamble, like everything else in that
directory.  Its LOGIC, however, is now collected: the ``check_*`` bodies
it calls are pytest cells in the service suite.  Before the graduation the
logic here ran in NO CI at all.

Usage::

    lx run --cpu -N 1 -n 4 python3 -u \\
        tests/bench/wfn_loader_backend_parity_test.py --mesh 2x2
    lx run -N 1 -G 4 -n 4 python3 -u \\
        tests/bench/wfn_loader_backend_parity_test.py --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402

jax.config.update("jax_enable_x64", True)

# <repo>/tests/bench/<this file>  ->  three levels up is the checkout.
_REPO = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
_SVC_TESTS = os.path.join(_REPO, "services", "wfn_loader", "tests")
assert os.path.isdir(_SVC_TESTS), (
    f"the wfn_loader service tests are not at {_SVC_TESTS!r}; this harness "
    f"is a wrapper over their check_* bodies and has no second copy of the "
    f"comparison to fall back on")

_DIST_FLAG = "_LORRAX_JAX_DISTRIBUTED_DONE"


def _bootstrap_jax_distributed() -> None:
    if os.environ.get(_DIST_FLAG):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        one_gpu = cvd and "," not in cvd
        kwargs = {"local_device_ids": [0]} if one_gpu else {}
        try:
            jax.distributed.initialize(**kwargs)
        except Exception:
            pass
    os.environ[_DIST_FLAG] = "1"


# BEFORE the check module is imported: it imports ``wfn_loader``, which
# reaches the XLA backend, and ``jax.distributed.initialize`` has to run
# first.  The check module's own ``__main__`` guard means importing it
# here does NOT try to initialize a second time.
_bootstrap_jax_distributed()

if _SVC_TESTS not in sys.path:
    sys.path.insert(0, _SVC_TESTS)

from test_wfn_loader_multiproc import (                       # noqa: E402
    DEFAULT_BANDS, _CLI_CELLS, deck_path)


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _parse_mesh(spec: str) -> tuple[int, int]:
    parts = spec.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"--mesh expects PxQ form, got {spec!r}")
    return int(parts[0]), int(parts[1])


def _build_mesh(p: int, q: int):
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices()).reshape(p, q),
                axis_names=("x", "y"))


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False,
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--wfn", default="",
                    help="WFN.h5 (default: the in-repo gnppm hostile deck; "
                         "pass /pscratch/... explicitly to override)")
    ap.add_argument("--mesh", required=True, help="PxQ")
    ap.add_argument("--bands", default=",".join(map(str, DEFAULT_BANDS)),
                    help="b_lo,b_hi — the default is NON-DIVISIBLE on four "
                         "ranks on purpose (see the module docstring)")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="0.0 only.  Exact equality is the contract; a "
                         "tolerance here would hide the clamp class this "
                         "harness exists to catch.")
    ap.add_argument("--only", default="", help="substring filter on cells")
    args = ap.parse_args()

    if args.atol != 0.0:
        _log(f"REFUSED --atol {args.atol!r}: the two backends read the same "
             f"f64 bytes off the same file, so the contract is BIT "
             f"equality (np.array_equal) and a tolerance would hide the "
             f"22049c3 band-clamp class.  Drop the flag.")
        return 2

    deck = args.wfn or deck_path()
    if not deck or not os.path.exists(deck):
        _log(f"WFN not found: {deck!r}")
        return 1

    p, q = _parse_mesh(args.mesh)
    if jax.device_count() < p * q:
        _log(f"need {p*q} devices, found {jax.device_count()}")
        return 1
    mesh = _build_mesh(p, q)
    bands = tuple(int(v) for v in args.bands.split(","))

    _log(f"[WfnLoader parity] wfn={deck}  mesh={p}x{q}  bands={bands}  "
         f"atol=0.0 (exact)")
    _log(f"[WfnLoader parity] world={jax.process_count()}  "
         f"devices={jax.device_count()}  backend={jax.default_backend()}")

    ran, failures = 0, 0
    for name, platform, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        is_cpu = jax.default_backend() == "cpu"
        if (platform == "cpu" and not is_cpu) or (platform == "CUDA" and is_cpu):
            _log(f"SKIP {name} ({platform}-only)")
            continue
        try:
            out = fn(mesh, deck, bands)
            ran += 1
            _log(f"PASS {name} {out}")
        except AssertionError as exc:
            failures += 1
            _log(f"FAIL {name}: {exc}")
        except Exception as exc:                               # noqa: BLE001
            failures += 1
            _log(f"ERROR {name}: {type(exc).__name__}: "
                 f"{' '.join(str(exc).split())[:600]}")

    # RAN, not just failures: "ALL PASS" out of zero cells is the shape of
    # every artifact-free green in this tree's history.
    _log(f"[WfnLoader parity] {ran} cells ran, {failures} failures — "
         f"{'ALL PASS' if (ran and not failures) else 'FAILED'}")
    return 0 if (ran and not failures) else 1


if __name__ == "__main__":
    sys.exit(main())
