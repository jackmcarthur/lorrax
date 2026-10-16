"""The TD-aGW kernel as a TDGW ``kernel.h5`` (the dense algebraic twin of the model).

``write_tdgw_kernel`` stores K_{knm,k'n'm'} (Ry; ΔΣ_{k,nm} = Σ K δρ_{k',n'm'})
in exactly the layout TDGW's ``td/io.py:read_kernel`` reads with
``bse_header/params/nblocks = 4``:

    kernel[k', k, f(n), f(m), f(n'), f(m'), (re, im)] = −K[k, n, m, k', n', m']   (Ry)

TDGW then uses K^TDGW = −kernel/2 Ha.  ``f`` is the BerkeleyGW band order of
the file: valence from i_vbm downwards, then conduction from i_vbm + 1 upwards
(TDGW's ``bands_file_b``).  k and k' follow ``kpts``, which must be the k list
of TDGW's WFN.h5 in its own order (``BSEKernel.check_basis``).

``python -m tdagw.kernel_export`` builds the model from a GW restart and
TDGW's WFN (``tdagw.model.load_td_model``), samples K column by column from
the model's operator (``dense_td_kernel``) and writes the file on rank 0.
Small windows only: the file holds (N_k n_b²)² complex numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import h5py
import numpy as np

__all__ = ["tdgw_file_band_order", "write_tdgw_kernel", "main"]

#: BerkeleyGW's ``header`` codes (TDGW ``td/io.py``): screening, truncation, kernel type.
SCREENING_SEMICONDUCTOR = 0
TRUNCATION = {3: 0, 2: 6}             # sys_dim -> none / cell_slab
KERNEL_TYPE = {1: 1, 2: 3}            # nspinor -> singlet / spinor


def tdgw_file_band_order(ibndmin: int, nb: int, ivbm: int) -> np.ndarray:
    """File index of each window band ``ibndmin, ibndmin+1, …`` (1-based).

    Valence ``b ≤ ivbm`` sits at ``ivbm − b``, conduction at
    ``n_v + b − ivbm − 1`` with ``n_v = ivbm − ibndmin + 1`` — TDGW's
    ``IndexedBasis(concat(v_file_b, c_file_b))`` order.
    """
    bands = int(ibndmin) + np.arange(int(nb))
    n_v = int(ivbm) - int(ibndmin) + 1
    if not 0 < n_v < int(nb):
        raise ValueError(f"GATE tdgw_kernel_window: got: ibndmin={ibndmin}, nb={nb}, "
                         f"ivbm={ivbm}; want: ibndmin <= ivbm < ibndmin+nb-1; fix: give "
                         f"a window with valence and conduction bands")
    return np.where(bands <= ivbm, ivbm - bands, n_v + bands - ivbm - 1)


def write_tdgw_kernel(path: str, K: np.ndarray, *, kpts: np.ndarray, ibndmin: int,
                      ivbm: int, sys_dim: int, nspinor: int, provenance: dict | None = None):
    """Write ``K`` (Ry) as TDGW's extended (nblocks = 4) ``kernel.h5``.

    Parameters
    ----------
    path : str
    K : (nk, nb, nb, nk, nb, nb) complex
        K_{knm,k'n'm'} in Ry, k in the order of ``kpts``, bands ascending from
        ``ibndmin``: ΔΣ_{k,nm} = Σ_{k'n'm'} K[k,n,m,k',n',m'] δρ_{k',n'm'}.
    kpts : (nk, 3) float — TDGW's WFN k list, crystal coordinates.
    ibndmin, ivbm : int — 1-based first window band and valence-band maximum.
    sys_dim, nspinor : int — the header's truncation and kernel-type codes.
    provenance : dict, optional — stored as JSON in the root attribute
        ``lorrax_provenance``.
    """
    K = np.asarray(K)
    nk, nb = int(K.shape[0]), int(K.shape[1])
    if K.shape != (nk, nb, nb, nk, nb, nb):
        raise ValueError(f"GATE tdgw_kernel_shape: got: {K.shape}; want: (nk, nb, nb, nk, nb, nb); "
                         f"fix: pass the dense kernel of the window")
    if np.shape(kpts) != (nk, 3):
        raise ValueError(f"GATE tdgw_kernel_kpts: got: {np.shape(kpts)}; want: ({nk}, 3); "
                         f"fix: pass TDGW's WFN k list")
    f = tdgw_file_band_order(ibndmin, nb, ivbm)
    inv = np.argsort(f)                       # window band at each file slot
    Kt = np.transpose(K, (3, 0, 1, 2, 4, 5))  # (k', k, n, m, n', m')
    Kt = Kt[:, :, inv][:, :, :, inv][:, :, :, :, inv][:, :, :, :, :, inv]
    data = np.stack([-Kt.real, -Kt.imag], axis=-1).astype(np.float64)
    n_v = int(ivbm) - int(ibndmin) + 1
    with h5py.File(path, "w") as h:
        bh = h.create_group("bse_header")
        bh["flavor"] = np.int32(2)
        bh.create_group("params")["nblocks"] = np.int32(4)
        bh.create_group("kpoints")["kpts"] = np.asarray(kpts, dtype=np.float64)
        b = bh.create_group("bands")
        b["n1b"], b["n2b"] = np.int32(nb), np.int32(nb)
        b["nvb"], b["ncb"] = np.int32(n_v), np.int32(nb - n_v)
        kh = h.create_group("header")
        kh["screening"] = np.int32(SCREENING_SEMICONDUCTOR)
        kh["truncation"] = np.int32(TRUNCATION[int(sys_dim)])
        kh["kernel_type"] = np.int32(KERNEL_TYPE[int(nspinor)])
        h.create_dataset("kernel", data=data)
        h.attrs["lorrax_provenance"] = json.dumps(provenance or {}, default=str)


def _parse(argv):
    p = argparse.ArgumentParser(allow_abbrev=False, description=(
        "Write the TD-aGW ISDF kernel of a TDGW band window as TDGW's kernel.h5"))
    p.add_argument("--input", required=True, help="GW deck of the restart bundle")
    p.add_argument("--wfn", required=True, help="TDGW's full-grid WFN.h5 (its band gauge)")
    p.add_argument("--ibndmin", type=int, required=True, help="1-based first band (TDGW)")
    p.add_argument("--ibndmax", type=int, required=True, help="1-based last band (TDGW)")
    p.add_argument("--ivbm", type=int, required=True, help="1-based valence-band maximum")
    p.add_argument("--zeta", default=None, help="zeta_q.h5 (default: beside the restart)")
    p.add_argument("--terms", choices=("both", "sex", "hartree"), default="both")
    p.add_argument("--out", required=True, help="kernel.h5 to write")
    p.add_argument("--receipt", default=None, help="JSON receipt (default: <out>.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    from tdagw.model import build_td_agw_apply, dense_td_kernel, load_td_model
    mesh = stack.mesh
    model = load_td_model(args.input, mesh, bands=(args.ibndmin - 1, args.ibndmax),
                          wfn_file=args.wfn, zeta_file=args.zeta, n_occ=args.ivbm)
    apply = build_td_agw_apply(mesh, model, sex=args.terms in ("both", "sex"),
                               hartree=args.terms in ("both", "hartree"))
    K = dense_td_kernel(apply, model, mesh_xy=mesh)                # model (flat-grid) k order
    from tdagw.model import flat_k_rows
    from wfn_loader import WfnLoader
    w = WfnLoader(args.wfn)
    k_tdgw = np.asarray(w.kvecs(k="ibz"))
    w.close()
    rows = flat_k_rows(k_tdgw, model["kgrid"])
    K = K[rows][:, :, :, rows]
    rec = dict(model["receipts"], terms=args.terms, out=os.path.abspath(args.out),
               ibndmin=args.ibndmin, ibndmax=args.ibndmax, ivbm=args.ivbm,
               kernel_frobenius_ry=float(np.linalg.norm(K)),
               kernel_antihermitian_rel=float(
                   np.linalg.norm(K.reshape(K.shape[0] * K.shape[1] ** 2, -1)
                                  - K.reshape(K.shape[0] * K.shape[1] ** 2, -1).conj().T)
                   / max(np.linalg.norm(K), 1e-300)))
    if jax.process_index() == 0:
        write_tdgw_kernel(args.out, K, kpts=k_tdgw, ibndmin=args.ibndmin, ivbm=args.ivbm,
                          sys_dim=int(model["receipts"]["coulomb"]["sys_dim"]),
                          nspinor=int(model["nspinor"]), provenance=rec)
        with open(args.receipt or args.out + ".json", "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        print(f"[tdagw] kernel.h5 written: {args.out}  |K|_F = {rec['kernel_frobenius_ry']:.6e} Ry, "
              f"anti-Hermitian part {rec['kernel_antihermitian_rel']:.3e} (relative)", flush=True)
    runtime.finalize_process(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
