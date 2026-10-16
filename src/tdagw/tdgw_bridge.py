"""TDGW's TD-aGW propagator with its self-energy update taken from the LORRAX model.

TDGW (``td/driver/agw.py``) propagates G^<_k = iρ_k on the full k grid and
updates ΔΣ_k = K·δρ with a dense BerkeleyGW kernel.  The bridge changes
nothing in TDGW: :func:`lorrax_driver_class` subclasses ``aGWDriver`` and
overrides ``get_dSigma`` only, and :func:`tdgw_run_lorrax` is ``tdgw_run``
without ``_load_kernel``.  Each TDGW rank is one JAX process with one GPU;
TDGW already all-gathers G^<, so every rank passes the same full δρ, the
model applies on the run's 2x2 mesh, and every rank keeps its own k rows of
ΔΣ (:func:`make_tdgw_dsigma`).

Units and layout at the seam: TDGW holds δρ = −i(G^< − G^<_eq) as
(nk, nb, nb) in its WFN's k order and its window's ascending band order, and
ΔΣ in Ha; the model works in Ry on the flat C-order k grid, so the seam
permutes k by :func:`tdagw.model.flat_k_rows` and halves.

``python -m tdagw.tdgw_bridge tdgw.toml --input <GW deck>`` runs one TDGW
calculation; TDGW's ``[system]`` band window and ``ivbm`` select the model's
window, and ``indir/WFN.h5`` its ψ (``tdagw.model.load_td_model``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np

__all__ = ["lorrax_driver_class", "make_tdgw_dsigma", "tdgw_run_lorrax", "main"]


def make_tdgw_dsigma(apply, model: dict, mesh_xy, k_rows: np.ndarray):
    """``dsigma(drho) -> ΔΣ`` at TDGW's seam: host (nk, nb, nb), TDGW k order, Ha.

    Parameters
    ----------
    apply : callable — :func:`tdagw.model.build_td_agw_apply`'s operator.
    model : dict — its bundle (``n_bands``, ``nb_pad``).
    k_rows : (nk,) int — the model's flat k row of each TDGW k point.

    The δρ passed in is the same full array on every process; it is placed
    on the mesh from that host copy (each process supplies its own shards),
    and ΔΣ is replicated back so every process can keep its own k rows.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from tdagw.model import pad_td_state, strip_td_state, td_state_sharding
    st = td_state_sharding(mesh_xy)
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    k_rows = np.asarray(k_rows, dtype=np.int64)

    def dsigma(drho):
        x = np.empty_like(np.asarray(drho, dtype=np.complex128))
        x[k_rows] = drho
        xc = pad_td_state(x[None], model)
        xd = jax.make_array_from_callback(xc.shape, st, lambda i: xc[i])
        y = strip_td_state(np.asarray(rep(apply(xd)).addressable_data(0)), model)[0]
        return 0.5 * y[k_rows]
    return dsigma


def lorrax_driver_class():
    """``aGWDriver`` with ``get_dSigma`` from :func:`make_tdgw_dsigma` (TDGW imported here)."""
    import tensor2
    from tensor2.constants import TENSOR_DATA_NP
    from TDGW.td.driver.agw import aGWDriver
    from TDGW.utils.timer import tdgwtimerhandler as T

    class LorraxAGWDriver(aGWDriver):
        """ΔΣ_k = K^ISDF δρ (``docs/theory/td_agw.md``) instead of a dense kernel."""

        lorrax_dsigma = None

        def get_dSigma(self, lessG_knn):
            with T.lessGcomm.timeit():
                lessG_full_knn = tensor2.mpi_allgather(lessG_knn)
            dG = (np.asarray(lessG_full_knn.get_data(TENSOR_DATA_NP))
                  - np.asarray(self.lessGeq_full_knn.get_data(TENSOR_DATA_NP)))
            dsig = type(self).lorrax_dsigma(-1j * dG)
            local = np.ascontiguousarray(dsig[self.wfnhd.kpts_b.pax.pslice])
            return tensor2.astensor(local, distrib=self._distrib_knn), lessG_full_knn

    return LorraxAGWDriver


def tdgw_run_lorrax(config, dsigma, *, k_frac_expected=None, comm=None):
    """``TDGW.td.run.tdgw_run`` with the model's ΔΣ and no ``kernel.h5``.

    ``k_frac_expected`` (nk, 3), when given, is the k list ``dsigma`` was
    built for; TDGW's own list must equal it point by point (mod G) or the
    run refuses before propagating.
    """
    from os.path import join
    from TDGW.constants import TDGW_DATA_DEFAULT_FNAME
    from TDGW.td import run as R
    from TDGW.td.ivp.ivp import solve_ivp
    if config.control.calc_type != "COHSEX":
        raise ValueError(f"GATE tdgw_bridge_calc_type: got: {config.control.calc_type!r}; "
                         f"want: 'COHSEX'; fix: the bridge supplies the self-energy update, "
                         f"so set calc_type = \"COHSEX\" (an IP run needs no bridge)")
    grid = R._prep(config, comm)
    if grid is None:
        return
    stru, wfnhd, dipole = R._load_header(config, grid)
    if k_frac_expected is not None:
        k_tdgw = np.asarray(wfnhd.kpts_b.basis_full.kpts, dtype=np.float64)
        d = k_tdgw - np.asarray(k_frac_expected)
        if k_tdgw.shape != np.shape(k_frac_expected) or np.max(np.abs(d - np.rint(d))) > 1e-8:
            raise ValueError("GATE tdgw_bridge_kpts: got: TDGW's k list differs from the "
                             "model's TDGW-order list; want: the same points in the same "
                             "order; fix: build the model from this run's indir/WFN.h5")
    chkpdata = R._load_chkp(config, grid)
    Driver = lorrax_driver_class()
    Driver.lorrax_dsigma = staticmethod(dsigma)
    driver = Driver(config.control.timesteps.t0, config.control.timesteps.dt_measure,
                    config.control.timesteps.tf, config.control.timesteps.ntcheck,
                    stru, wfnhd, dipole, None, config.system.Efunc,
                    config.system.depopulation, config.system.dephasing,
                    config.measurements, join(config.control.files.outdir, TDGW_DATA_DEFAULT_FNAME),
                    grid, chkpdata=chkpdata)
    if chkpdata is not None:
        chkpdata.close()
    solve_ivp(driver, rtol=config.control.ivp.rtol, atol=config.control.ivp.atol,
              max_step=config.control.timesteps.max_step)
    R._finalize()


def _parse(argv):
    p = argparse.ArgumentParser(allow_abbrev=False, description=(
        "Run TDGW with its self-energy update from the LORRAX TD-aGW model"))
    p.add_argument("config", help="TDGW toml")
    p.add_argument("--input", required=True, help="GW deck of the restart bundle")
    p.add_argument("--zeta", default=None, help="zeta_q.h5 (default: beside the restart)")
    p.add_argument("--terms", choices=("both", "sex", "hartree"), default="both")
    p.add_argument("--receipt", default=None, help="JSON receipt (default: <outdir>/lorrax_model.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    from TDGW.td.cli import load_tdgw_config
    from tdagw.model import build_td_agw_apply, flat_k_rows, load_td_model
    from wfn_loader import WfnLoader
    mesh = stack.mesh
    config = load_tdgw_config(args.config)
    wfn_file = os.path.join(config.control.files.indir, "WFN.h5")
    s = config.system
    model = load_td_model(args.input, mesh, bands=(s.ibndmin - 1, s.ibndmax), wfn_file=wfn_file,
                          zeta_file=args.zeta, n_occ=s.ivbm)
    apply = build_td_agw_apply(mesh, model, sex=args.terms in ("both", "sex"),
                               hartree=args.terms in ("both", "hartree"))
    w = WfnLoader(wfn_file)
    k_tdgw = np.asarray(w.kvecs(k="ibz"))
    w.close()
    dsigma = make_tdgw_dsigma(apply, model, mesh, flat_k_rows(k_tdgw, model["kgrid"]))
    if jax.process_index() == 0:
        os.makedirs(config.control.files.outdir, exist_ok=True)
        with open(args.receipt or os.path.join(config.control.files.outdir,
                                               "lorrax_model.json"), "w") as fh:
            json.dump(dict(model["receipts"], terms=args.terms, config=os.path.abspath(args.config)),
                      fh, indent=1, default=str)
    tdgw_run_lorrax(config, dsigma, k_frac_expected=k_tdgw)
    runtime.finalize_process(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
