"""Time-reversal gate for the chi0 family at NONZERO tau (detector only).

Why it exists.  Every symmetry check that the chi0 family had ran at t = 0.
At t = 0 the assembly cancels a tau-dependent defect.  SCFIX (2026-09-16,
Si 4x4x4 scalar) measured the exact moment M1 transpose-clean at 8.9e-17 at
t = 0, and the same reconstruction at 2.0e-10 at t = -1.0i.  So a check at
t = 0 cannot see this class of defect.  This gate measures at nonzero t.

What it measures.  On a scalar deck with time-reversal symmetry,
psi_{-k} = conj(psi_k) and E(-k) = E(k).  The following are then known-zero
for any band weights that are equal at k and -k, including the complex phases
exp(-t (E - E_ref)):

    parent   ||G_p - G_p^T|| / ||G_p||           TRIM parents, read at their own
             ||Im G_p|| / ||G_p||                 identity rows of the unfolded tile
                                                  (Im at the real t only)
    unfold   max_k ||X_k^T - X_-k|| / max_k ||X_k||   after the k-unfold (build_G_tau)
    fft      max_R ||X_R^T - X_-R|| / max_R ||X_R||   after the flat-k FFT
    chi      ||chi_q - chi_q^T|| / ||chi_q||      every TRIM q, through the
                                                  production retarded stream

The ||X_k^T - X_-k|| metric is conjugation-invariant, so it does not depend on
the conj(build_G_tau) orientation used by the incumbent kernel.  Each operand
comes from the production owner: ``WfnLoader`` / ``SymMaps``,
``load_centroid_basis``, ``PackedCentroidBasis``, ``Meta.from_system``,
``build_centroid_k_unfold_plan``, ``load_centroids_band_chunked``,
``build_packed_parent_green_carrier``, ``response_weights`` /
``stream_weights`` / ``response_stream`` (the bank's own binding of
``_get_chi_fractional_contour_kernel_face``), ``build_G_tau`` and
``make_flat_k_fftn``.  The metrics are reductions only.  No tile is
symmetrised, projected or averaged (owner TRIM ruling).  This is a detector.

Threshold.  ``THRESHOLD = 1e-6`` on every row.  SCFIX's floors for this deck
were 4.334e-09 (unfold, the DFT input's own time-reversal-reality error;
the raw QE coefficients already carry it) and 2.5e-10 (chi).  Its planted
errors were 1.912e+00 (wrap-phase conjugation) and 1.45e-02 (a band window
that cuts a degenerate doublet).  The bar is about 230x above that input
floor and 14,500x below the smaller planted error.

Twins (``--plant``), both must be red:
    conj_wrap    the right-endpoint lattice-wrap phase of the k-unfold is
                 conjugated (the plan's L_table is negated on the right endpoint)
    doublet_cut  the unoccupied band weight stops at ``--cut``, inside a
                 degenerate doublet
Each twin refuses to run if its plant would change nothing.

Refusals.  On a deck where time reversal is broken (``WfnLoader.trs_holds``
is False, the only source of ``SymMaps.trs_allowed``, read before the tables
are built), these quantities contain physics and have no known-zero target.
The gate refuses there and measures nothing.  It also refuses spinor decks
and a centroid set that is not orbit-closed.

Cost.  A cold P1 run takes about 5 s of gate wall: 1.8 s runtime startup
and 3 s body.  About 2.3 s of the body is compile and provider setup that
belongs to the production stack itself (the centroid loader, the GEMM
provider, the kernel's GEMM warmup, the kernel).  The gate's own programs
compile in about 0.5 s.  ``phase_seconds`` in the receipt has the split
(TAUGATE FAST, 2026-09-17).  Any square P works.  P4 also covers the sharded
unfold, and costs about 11 s.

Exit codes: 0 PASS, 1 FAIL, 3 REFUSED.  Run on a compute node, one rank per
GPU (Si 4x4x4 scalar is certified at P1 and P4)::

    lx run --jid $JID -N 1 -G 1 -n 1 python3 -u \
      tests/multi_device/nonzero_tau_trs_gate.py \
      --wfn .../qe/WFN.h5 --centroids .../qe/centroids_frac_368.txt \
      --nval 4 --ncond 30 --nband 34 --out <evidence dir> [--plant ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

_T0 = time.monotonic()
from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

THRESHOLD = 1.0e-6
#: Nonzero t on the Green stages: the retarded node t = -1.0i, one generic
#: complex t, and one real (Laplace) t where the weights are real.
GREEN_NODES = (-1.0j, 0.5 - 1.0j, 1.0 + 0.0j)
#: Retarded-stream time nodes for chi (the kernel forms t = -i * time).
CHI_TIMES = (1.0,)
#: Two parent energies closer than this (Ry) are one multiplet.
DEGENERATE_RY = 1.0e-8
REFUSED = 3


class Refusal(Exception):
    """The gate declines to measure; the message names the reason."""


def _refuse(name, got, want, why):
    raise Refusal(f"GATE nonzero_tau_trs_{name}: got {got}; want {want}; "
                  f"why: {why}. REFUSED, nothing measured.")


def _conjugated_right_wrap(plan):
    """Twin plan: the unfold's right-endpoint wrap phase is conjugated."""
    from gw.centroid_k_unfold import CentroidKUnfoldPlan

    right = dataclasses.replace(plan, L_table=-plan.L_table)

    @dataclasses.dataclass(frozen=True, eq=False)
    class ConjugatedRightWrapPlan(CentroidKUnfoldPlan):
        def unfold_operator(self, operator_parent, *, operator_transpose=None,
                            right_plan=None):
            return CentroidKUnfoldPlan.unfold_operator(
                self, operator_parent, operator_transpose=operator_transpose,
                right_plan=right)

    fields = {f.name: getattr(plan, f.name) for f in dataclasses.fields(plan)}
    return ConjugatedRightWrapPlan(**fields)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--nval", type=int, required=True)
    ap.add_argument("--ncond", type=int, required=True)
    ap.add_argument("--nband", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plant", choices=("none", "conj_wrap", "doublet_cut"),
                    default="none")
    ap.add_argument("--cut", type=int, default=33,
                    help="doublet_cut: first band index removed from the u weight")
    args = ap.parse_args(argv)

    started = _T0
    phases, last = {}, [_T0]

    def stamp(name, *arrays):
        """Wall seconds since the previous stamp; ``arrays`` are synchronized first."""
        if arrays:
            jax.block_until_ready(arrays)
        now = time.monotonic()
        phases[name] = round(now - last[0], 3)
        last[0] = now

    runtime = initialize_communicator_stack(platform="gpu")
    stamp("runtime_startup")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common import Meta
    from common.centroid_basis import PackedCentroidBasis
    from common.collectives import gather_to_host
    from common.wfn_layout import psi_specs
    from common.fft_helpers import make_flat_k_fftn
    from common.wfn_transforms import get_enk_bandrange, load_centroids_band_chunked
    from distrib_la import gemm_plan
    from file_io.centroids import load_centroid_basis
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.greens_function_kernel import build_G_tau
    from gw.response_bank import response_stream, response_weights, stream_weights
    from gw.wavefunction_bundle import (
        BandSlices, G_FFT7D_SPEC, G_FLATK_SPEC, attach_packed_parent_green_carrier,
        parent_faces, wavefunctions_face_from_restart)
    from symmetry_maps import q_negation_index, self_negative_q_mask
    from wfn_loader import WfnLoader

    stamp("imports")
    mesh = runtime.mesh
    root = jax.process_index() == 0
    say = print if root else (lambda *a, **k: None)
    out = Path(args.out)
    arm = args.plant
    receipt = dict(
        gate="nonzero_tau_trs_v1", arm=arm, threshold=THRESHOLD,
        wfn=os.path.abspath(args.wfn), centroids=os.path.abspath(args.centroids),
        mesh=dict(x=int(mesh.shape["x"]), y=int(mesh.shape["y"])),
        processes=int(jax.process_count()),
        job=os.getenv("SLURM_JOB_ID"), step=os.getenv("SLURM_STEP_ID"),
        rows=[])

    def finish(verdict, rc, reason=""):
        stamp("report")
        receipt.update(verdict=verdict, reason=reason, phase_seconds=phases,
                       seconds=round(time.monotonic() - started, 2))
        if root:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{arm}.json").write_text(json.dumps(receipt, indent=1) + "\n")
            print("[taugate] phase seconds: "
                  + " ".join(f"{k}={v}" for k, v in phases.items()), flush=True)
            print(f"[taugate] arm={arm} VERDICT {verdict} rc={rc} "
                  f"seconds={receipt['seconds']} {reason}", flush=True)
        return rc

    try:
        # ---- scope: time reversal, representation, orbit closure ---------
        # The loader's measured verdict is the only source of
        # SymMaps.trs_allowed, and it exists before the symmetry tables do:
        # refuse on it first, so a deck whose tables cannot be built without
        # time reversal still gets the physics refusal.
        wfn = WfnLoader(args.wfn, mesh=mesh)
        receipt["trs_holds"] = wfn.trs_holds
        receipt["nspinor"] = int(wfn.nspinor)
        stamp("wfn_loader")
        if wfn.trs_holds is not True:
            _refuse("time_reversal",
                    f"WfnLoader.trs_holds = {wfn.trs_holds} (the SymMaps.trs_allowed verdict)",
                    "a time-reversal-symmetric deck",
                    "with time reversal broken, |X_k^T - X_-k| and the TRIM-parent "
                    "|Im G_p| contain physics and have no known-zero target, so "
                    "they are not a gate on this deck")
        if int(wfn.nspinor) != 1:
            _refuse("representation", f"nspinor = {int(wfn.nspinor)}", "nspinor = 1",
                    "the known-zero relation X_k^T = X_-k is the scalar one; the "
                    "spinor form needs the Kramers operator and is not built here")
        sym = wfn.symmetry()
        receipt["trs_allowed"] = bool(sym.trs_allowed)
        stamp("symmetry")
        centroids = load_centroid_basis(args.centroids, wfn.fft_grid, sym=sym)
        if not centroids.orbit_closed:
            _refuse("centroids", "a centroid set that is not orbit-closed",
                    "an orbit-closed set", "production then drops to the trivial "
                    "symmetry view, and the unfold under test does not run")

        # ---- production system setup (gw_jax._load_system_inputs order) --
        idx = centroids.centroid_indices
        basis = PackedCentroidBasis.build(idx, sym, wfn.fft_grid, mesh)
        meta = Meta.from_system(wfn, sym, args.nval, args.ncond, args.nband,
                                centroids.n_rmu, False, mesh_xy=mesh, mu_basis=basis)
        meta.bispinor = False
        slices = BandSlices.from_band_edges(
            *meta.band_edges, b4_chi=meta.b_id_4_chi, b4_sigma=meta.b_id_4_sigma,
            b4_logical=meta.b_id_4_user)
        plan = build_centroid_k_unfold_plan(
            sym, idx, meta.fft_grid, mesh, nspinor=1,
            parent_k_frac=wfn.kvecs(k=sym.parent_k_domain), layout=basis.layout)
        stamp("centroids_basis_meta_plan")
        parent_y, parent_x = load_centroids_band_chunked(
            wfn, sym, meta, idx, False, mesh, band_range=slices.full_range,
            k_domain=sym.parent_k_domain, bispinor_lift="raw")
        psi_nmu, psi_mun = parent_faces(parent_y, parent_x, mesh_xy=mesh, layout="face")
        del parent_y, parent_x
        stamp("load_centroid_faces", psi_nmu, psi_mun)
        enk_full, _ = get_enk_bandrange(wfn, sym, slices.full_range,
                                        (slices.b1, slices.b3), nspinor=1)
        wfns = wavefunctions_face_from_restart(
            None, None, layout="face", enk_full=enk_full,
            slices=slices, mesh_xy=mesh)

        grid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
        nk = int(np.prod(grid))
        neg = q_negation_index(grid)
        trim_q = tuple(int(q) for q in np.flatnonzero(
            self_negative_q_mask(np.arange(nk), kgrid=grid)))
        parent_rows = np.asarray(plan.parent_full_rows)
        trim_parents = [int(p) for p in np.flatnonzero(
            self_negative_q_mask(parent_rows, kgrid=grid))]
        energy, f, u, reference, census = response_weights(
            attach_packed_parent_green_carrier(wfns, psi_nmu, psi_mun, plan=plan,
                                               mesh_xy=mesh), meta)
        e_parent = energy[parent_rows]
        top = int(census["band_stop"]) - int(census["band_start"])
        edge_gap = (float(np.min(np.abs(e_parent[:, top] - e_parent[:, top - 1])))
                    if top < e_parent.shape[1] else None)
        antiunitary_rows = int(np.sum(np.asarray(plan.sym_idx) >= plan.n_sym_spatial))
        receipt.update(
            kgrid=list(grid), n_parent=int(plan.n_parent), parent_full_rows=parent_rows.tolist(),
            trim_q=list(trim_q), trim_parent_slots=trim_parents,
            n_centroid=int(centroids.n_rmu), n_packed=int(plan.n_centroid_packed),
            census=census, reference_ry=float(reference),
            window_top_edge_min_gap_ry=edge_gap, antiunitary_unfold_rows=antiunitary_rows,
            green_nodes=[[t.real, t.imag] for t in GREEN_NODES],
            chi_times=list(CHI_TIMES))
        say(f"[taugate] arm={arm} mesh={mesh.shape['x']}x{mesh.shape['y']} P={jax.process_count()} "
            f"kgrid={grid} n_parent={plan.n_parent} TRIM parents(slots)={trim_parents} "
            f"TRIM q={list(trim_q)} n_mu={centroids.n_rmu} packed={plan.n_centroid_packed} "
            f"bands={census['band_start']}..{census['band_stop']} antiunitary unfold rows="
            f"{antiunitary_rows}", flush=True)

        # ---- the plants ---------------------------------------------------
        if arm == "conj_wrap":
            phase = np.einsum("kmi,ki->km",
                              np.asarray(plan.L_table)[np.asarray(plan.sym_idx)],
                              np.asarray(plan.k_parent_frac)[np.asarray(plan.irr_idx)])
            moved = int(np.sum(np.abs(np.sin(2 * np.pi * phase)) > 1e-12))
            receipt["plant"] = dict(kind="conj_wrap", nontrivial_wrap_phases=moved)
            if moved == 0:
                _refuse("vacuous_twin", "no nontrivial wrap phase on this deck",
                        "a wrap phase the plant can conjugate",
                        "a twin whose plant changes nothing proves nothing")
            plan = _conjugated_right_wrap(plan)
        if arm == "doublet_cut":
            cut = int(args.cut)
            gaps = np.abs(e_parent[:, cut] - e_parent[:, cut - 1])
            split = [int(p) for p in np.flatnonzero(gaps < DEGENERATE_RY)]
            receipt["plant"] = dict(kind="doublet_cut", cut=cut,
                                    gaps_ry=gaps.tolist(), split_parent_slots=split)
            if not split or not (0 < cut < top):
                _refuse("vacuous_twin", f"cut={cut}, split parents {split}",
                        "a cut inside the window that splits a degenerate pair",
                        "a twin whose plant changes nothing proves nothing")
            u = u.copy()
            u[:, cut:] = 0.0
        wfns = attach_packed_parent_green_carrier(wfns, psi_nmu, psi_mun, plan=plan,
                                                  mesh_xy=mesh)
        carrier = wfns.green_parent
        w_f = stream_weights(wfns, f, mesh)
        w_u = stream_weights(wfns, u, mesh)
        stamp("weights_and_carrier", w_f, w_u)

        # ---- Green stages: parent, unfold, FFT ----------------------------
        n_mu = int(plan.n_centroid_packed)
        nb = int(slices.nb_full)
        # warmup=False: this plan is only called inside the one jit below, so
        # its standalone warm executables would be compiled for nothing.
        g_plan = gemm_plan(mesh, m=n_mu, k=nb, n=n_mu, nq=int(plan.n_parent),
                           dtype=jnp.complex128, layout="face", warmup=False)
        G_fftn = make_flat_k_fftn(mesh, grid, G_FFT7D_SPEC, norm="ortho")
        G_shard = NamedSharding(mesh, G_FLATK_SPEC)
        # Rows k of the unfolded tile followed by rows R of its FFT; both
        # negate on the same grid.
        neg_both = jnp.asarray(np.concatenate([neg, neg + nk]).astype(np.int32))
        # The TRIM parents' own full-k rows (SymMaps.kirr_fullids) carry the
        # identity operation, so the unfolded tile there IS the parent tile,
        # and at k = -k the pair defect there IS ||G_p - G_p^T||: one Green
        # build serves the parent and the unfold stages.
        trim_rows = parent_rows[trim_parents]
        receipt["trim_parent_rows_identity"] = bool(
            np.all(np.asarray(plan.sym_idx)[trim_rows] == 0))

        rep0, rep2 = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, None))
        nmu_spec, mun_spec = psi_specs("face")

        @partial(jax.jit, in_shardings=(NamedSharding(mesh, mun_spec),
                                        NamedSharding(mesh, nmu_spec), rep2, rep2, rep0, rep0))
        def green_stages(psi_mun_p, psi_nmu_p, enk_p, weight, t, ref):
            """Per-row squared norms of X^T - X_-, X and Im X, for the unfold then the FFT.

            One reduction body over both tiles, written as re^2 + im^2: the
            same numbers as ``abs(.)**2`` on each tile separately, at a
            third of the compile time (TAUGATE FAST profile).
            """
            full = jax.lax.with_sharding_constraint(
                build_G_tau(psi_mun_p, psi_nmu_p, enk_p, t, e_ref=ref,
                            band_weight=weight, layout="face", gemm=g_plan,
                            k_unfold_plan=plan), G_shard)
            both = jnp.concatenate([full, G_fftn(full)], axis=0)
            diff = jnp.transpose(both, (0, 3, 4, 1, 2)) - jnp.take(both, neg_both, axis=0)
            axes = (1, 2, 3, 4)
            return jnp.stack([jnp.sum(diff.real ** 2 + diff.imag ** 2, axis=axes),
                              jnp.sum(both.real ** 2 + both.imag ** 2, axis=axes),
                              jnp.sum(both.imag ** 2, axis=axes)])

        def host(x):
            return np.asarray(gather_to_host(x))

        def add(stage, node, leg, metric, value, gated=True):
            value = float(value)
            ok = bool(np.isfinite(value) and value <= THRESHOLD)
            receipt["rows"].append(dict(stage=stage, node=node, leg=leg, metric=metric,
                                        value=value, gated=gated,
                                        status=("PASS" if ok else "FAIL") if gated else "INFO"))
            say(f"[taugate]   {stage:7s} {node:>14s} {leg:2s} {metric:44s} {value:.6e} "
                f"{'PASS' if ok else 'FAIL' if gated else 'info'}", flush=True)

        for t in GREEN_NODES:
            node = f"t={t.real + 0.0:+.2f}{t.imag:+.2f}i"
            for leg, weight in (("f", w_f), ("u", w_u)):
                num, den, imag = np.sqrt(host(green_stages(
                    carrier.psi_mun, carrier.psi_nmu, carrier.enk, weight,
                    jnp.asarray(t, dtype=jnp.complex128), jnp.asarray(reference))))
                gp_t, gp_im = num[trim_rows] / den[trim_rows], imag[trim_rows] / den[trim_rows]
                unfold = num[:nk].max() / den[:nk].max()
                fft = num[nk:].max() / den[nk:].max()
                for slot, value in zip(trim_parents, gp_t):
                    add("parent", node, leg, f"||G_p - G_p^T||/||G_p|| slot {slot}", value)
                if t.imag == 0.0:
                    for slot, value in zip(trim_parents, gp_im):
                        add("parent", node, leg, f"||Im G_p||/||G_p|| slot {slot}", value)
                add("unfold", node, leg, "max_k||X_k^T - X_-k||/max_k||X_k||", unfold)
                add("fft", node, leg, "max_R||X_R^T - X_-R||/max_R||X_R||", fft)
                if "green_first_call" not in phases:
                    stamp("green_first_call")
        stamp("green_other_calls")

        # ---- chi at TRIM q through the bank's own retarded stream ---------
        kernel, fixed = response_stream(wfns, meta, mesh_xy=mesh, q_ids=trim_q, n_outputs=1)
        stamp("chi_kernel_build")

        for time_node in CHI_TIMES:
            # One program: the production kernel and the two reductions.
            @jax.jit
            def chi_stage(psi_mun_p, psi_nmu_p, enk_p, weight_f, weight_u):
                chi = kernel(jnp.asarray([time_node]), jnp.asarray([[1.0 + 0.0j]]),
                             psi_mun_p, psi_nmu_p, enk_p, weight_f, weight_u,
                             jnp.asarray(reference))[:, 0]
                diff = chi - jnp.swapaxes(chi, 1, 2)
                return jnp.stack([jnp.sum(diff.real ** 2 + diff.imag ** 2, axis=(1, 2)),
                                  jnp.sum(chi.real ** 2 + chi.imag ** 2, axis=(1, 2))])

            diff2, norm2 = host(chi_stage(*fixed, w_f, w_u))
            defect, norm = np.sqrt(diff2 / norm2), np.sqrt(norm2)
            node = f"t={0.0:+.2f}{-time_node:+.2f}i"
            for q, value, size in zip(trim_q, defect, norm):
                add("chi", node, "fu", f"||chi_q - chi_q^T||/||chi_q|| q {q}", value)
                add("chi", node, "fu", f"||chi_q|| q {q} (denominator)", size, gated=False)
                if not (np.isfinite(size) and size > 0.0):
                    receipt["rows"][-2]["status"] = "FAIL"
            stamp(f"chi_call_time_{time_node}")

        failed = [r for r in receipt["rows"] if r["status"] == "FAIL"]
        worst = {}
        for r in receipt["rows"]:
            if r["gated"]:
                worst[r["stage"]] = max(worst.get(r["stage"], 0.0), r["value"])
        receipt["worst_by_stage"] = worst
        say(f"[taugate] arm={arm} worst by stage: "
            + " ".join(f"{k}={v:.3e}" for k, v in worst.items()), flush=True)
        if failed:
            first = failed[0]
            return finish("FAIL", 1, f"{len(failed)} row(s) above {THRESHOLD:.0e}; first: "
                          f"{first['stage']} {first['node']} {first['leg']} "
                          f"{first['metric']} = {first['value']:.3e}")
        return finish("PASS", 0, f"{len(receipt['rows'])} rows, all gated rows "
                      f"<= {THRESHOLD:.0e}")
    except Refusal as refusal:
        return finish("REFUSED", REFUSED, str(refusal))


if __name__ == "__main__":
    import traceback
    rc = 1
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        rc = 1
    finalize_process(rc)
