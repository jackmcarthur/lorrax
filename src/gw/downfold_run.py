"""The downfold pipeline: read one restart bundle, write a smaller one.

Five stages, in order, each timed under a ``downfold.*`` section so it shows
up in the standard timing report beside ``zeta_fit`` and ``sigma.exec``:

    downfold.load        the parent bundle, sharded, wedge already unfolded
    downfold.grams       S_LL over the retained window (one kernel call)
    downfold.select      CUR selection + the rank refusal
    downfold.solve       T = S_SS^-1 S_cross, rank-truncated
    downfold.congruence  V, W, the _nohead twins and g0 through T
    downfold.residual    the per-q Pythagorean error bar
    downfold.write       the small bundle, in the SAME format at smaller mu

THE HEADLINE PROPERTY, and the thing to protect if anyone proposes changing
this file: the output is a restart bundle in the **unchanged format** at a
smaller mu.  Not a new bespoke layout, not a sidecar.  That is what makes
every existing BSE consumer a zero-change drop-in —
``bse_io.load_bse_data_from_restart_sharded`` reads it, ``_MunuSlabPlan``
plans it, the ring and stack matvecs eat it, and none of them learns that a
downfold happened.  If a reviewer proposes a bespoke ``.h5`` for downfolded
objects, the answer is no.

WHAT THE BAND AXIS DOES *NOT* DO, in stage 1.  The retained band window is
the FIT window — it decides which pair densities the compression is faithful
to.  It is NOT a truncation of the stored band axis: ``psi_full_y`` and
``enk_full`` are written with their band axis unchanged, and only mu is
compressed.  That is deliberate.  Truncating bands would renumber every band
index in the bundle, move the ``band_window`` stamp that
``assert_restart_window_matches`` refuses on, and break the drop-in property
outright — a consumer asking for ``--n-occ 8`` would silently get different
states.  Band-axis truncation is a separate, later stage with its own
renumbering contract.
"""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common import timing
from common.collectives import barrier, process_rank
from file_io import (
    load_restart_state_from_h5,
    parse_coulomb_policy,
    read_coulomb_policy_from_h5,
    read_munu_tensor_from_h5,
    write_head_scalars_to_h5,
    write_restart_state_to_h5,
)
from gw.downfold import (
    BandWindow,
    R19_ANCHOR,
    build_transfer,
    congruence,
    epsilon_w,
    pair_density_gram,
    select_cur_centroids,
    slice_psi_to_centroids,
    transform_head_vector,
)

__all__ = ["DownfoldResult", "resolve_restart_file", "run_downfold"]

#: Bumped when the provenance group's contents change meaning.
DOWNFOLD_PROVENANCE_VERSION = 1

#: The two-point tensors a downfold transports.  Each is a LINEAR object in
#: the centroid basis and therefore transforms by congruence.  A plasmon-pole
#: reduction does NOT belong on this list: ``B_q`` would transform but
#: ``Omega_q`` is a pole POSITION per matrix element and no congruence maps a
#: table of pole frequencies between bases.  Downfold the linear objects and
#: re-fit the PPM/MPA model in the small basis.
_MUNU_TENSORS = ("V_qmunu", "W0_qmunu", "V_qmunu_nohead", "W0_qmunu_nohead")


@dataclass
class DownfoldResult:
    """Everything the run learned, for the report and for the provenance."""

    source_file: str
    output_file: str
    mu_large: int
    mu_small: int
    n_q: int
    n_bands: int
    keep_idx: np.ndarray
    selection: object
    rank_per_q: np.ndarray
    eps_w: dict = field(default_factory=dict)
    norms: dict = field(default_factory=dict)
    wrote_nohead: tuple = ()
    centroid_file: str = ""


# ---------------------------------------------------------------------------
# locating the parent bundle
# ---------------------------------------------------------------------------

def resolve_restart_file(path: str) -> str:
    """A run directory or an ``.h5`` → THE parent restart file.

    ``isdf_tensors_*.h5`` is namespaced by CENTROID COUNT, not by run, so a
    directory can legitimately hold several and picking one silently is a
    wrong answer that passes every shape check downstream:
    ``isdf_tensors_1194.h5`` sorts before ``isdf_tensors_276.h5`` and neither
    order is meaningful.  ``bse_io._find_restart_file`` resolves the same
    ambiguity by newest mtime with a loud warning, which is the right call
    for a driver whose input file already named the run.  This one REFUSES
    instead, because the downfold's whole job is to produce a second bundle
    at a different mu in a nearby directory — it is the tool most likely to
    create the ambiguity, so it is the one that must not resolve it by luck.
    Name the file when there is more than one.
    """
    if os.path.isfile(path):
        return os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"downfold: source_restart={path!r} is neither a file nor a "
            f"directory.  Point it at the finished GW run's directory, or at "
            f"its tmp/isdf_tensors_<mu>.h5 directly.")
    cands = sorted(glob.glob(os.path.join(path, "tmp", "isdf_tensors_*.h5")))
    cands += sorted(glob.glob(os.path.join(path, "isdf_tensors_*.h5")))
    if not cands:
        raise FileNotFoundError(
            f"downfold: no isdf_tensors_*.h5 under {path} (looked in "
            f"'tmp/' and in the directory itself).  A downfold needs a "
            f"FINISHED GW calculation's restart bundle; if the GW run set "
            f"write_restart_tensors = false there is nothing on disk to "
            f"compress and the run has to be repeated with it true.")
    if len(cands) > 1:
        raise ValueError(
            f"downfold: {len(cands)} restart bundles under {path}: "
            f"{[os.path.basename(c) for c in cands]}.  They hold DIFFERENT "
            f"centroid counts and nothing about the file names says which "
            f"one this downfold means.  Set source_restart to the file "
            f"itself.")
    return os.path.abspath(cands[0])


def _read_geometry(filename: str) -> dict:
    """The small, replicated facts, read serially before any tensor moves."""
    with h5py.File(filename, "r") as f:
        if "psi_full_y" not in f:
            raise ValueError(
                f"downfold: {filename} has no psi_full_y.  Without "
                f"psi-at-centroids there are no pair densities to fit "
                f"against, so there is no downfold to do — this is not a "
                f"LORRAX restart bundle, or it is one from before the "
                f"canonical writer.")
        if "V_qmunu" not in f:
            raise ValueError(
                f"downfold: {filename} has no V_qmunu.")
        if "W0_qmunu" not in f:
            raise ValueError(
                f"downfold: {filename} has no W0_qmunu — the parent GW run "
                f"wrote its V and psi but never got as far as persisting the "
                f"screened interaction.  A downfold of V alone is a "
                f"legitimate thing to want and is not what this driver does; "
                f"finish the parent run first.")
        if not bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            raise ValueError(
                f"downfold: {filename} carries W0_qmunu but its W0_ready "
                f"flag is FALSE — the dataset is the all-zeros PLACEHOLDER "
                f"the writer pre-allocates, not screening.  Downfolding it "
                f"would produce a small bundle full of zeros that every "
                f"shape check passes; the flag exists because that happened "
                f"once already.  Re-run the parent GW to completion.")
        if not bool(f["V_qmunu"].attrs.get("V_ready", True)):
            raise ValueError(
                f"downfold: {filename} says V_ready = False.")
        geom = {
            "n_rmu_logical": (int(np.asarray(f["n_rmu_logical"])[()])
                              if "n_rmu_logical" in f
                              else int(f["V_qmunu"].shape[-1])),
            "kgrid": (tuple(int(v) for v in np.asarray(f["kgrid"])[:])
                      if "kgrid" in f else None),
            "band_window": (np.asarray(f["band_window"])[:].astype(np.int64)
                            if "band_window" in f else None),
            "nb": int(f["psi_full_y"].shape[1]),
            "nk": int(f["psi_full_y"].shape[0]),
            "nspinor": int(f["psi_full_y"].shape[2]),
            "vhead": (np.asarray(f["vhead"])[()] if "vhead" in f else None),
            "whead": (np.asarray(f["whead"][:]) if "whead" in f else None),
            "omega_grid": (np.asarray(f["whead"].attrs["omega_grid"])
                           if "whead" in f and "omega_grid" in f["whead"].attrs
                           else None),
            "centroids_charge_md5": f.attrs.get("centroids_charge_md5"),
            "present": tuple(n for n in _MUNU_TENSORS if n in f),
        }
    if geom["kgrid"] is None:
        raise ValueError(
            f"downfold: {filename} carries no kgrid, so the q axis of "
            f"V/W cannot be split into (nkx, nky, nkz).  The Gram build is a "
            f"convolution over k and needs that split; there is no way to "
            f"guess it from the flat q extent.  The bundle predates the "
            f"kgrid stamp — regenerate it with a current gw_jax, or read the "
            f"grid off the WFN the parent run used and note that this "
            f"driver deliberately does not take a WFN (its whole premise is "
            f"that a finished restart is self-describing).")
    return geom


class _BandSlices:
    """The five-integer band-window stamp, carried through verbatim.

    ``write_restart_state_to_h5`` wants an object with ``b0..b4`` and stamps
    those five numbers so a later run under a CHANGED window refuses instead
    of silently misindexing.  The downfold does not touch the band axis, so
    the right stamp is the PARENT's, unchanged — a small bundle that claimed
    a different window than the psi and enk it carries would be exactly the
    lie the stamp exists to catch.
    """

    def __init__(self, five):
        self.b0, self.b1, self.b2, self.b3, self.b4 = (int(v) for v in five)


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def run_downfold(cfg, mesh_xy, *, print_fn=print) -> DownfoldResult:
    """Read ``cfg.source_restart``, write ``cfg.output_restart``, report."""
    from runtime.padding import padded_mu_extent

    src = resolve_restart_file(cfg.source_restart)
    print_fn(f"  [downfold] parent bundle: {src}")

    with timing.section("downfold.load", announce=True,
                        label="parent restart bundle"):
        geom = _read_geometry(src)
        mu_L = int(geom["n_rmu_logical"])
        window = BandWindow(left=tuple(cfg.band_range_left),
                            right=tuple(cfg.band_range_right))
        window.validate(geom["nb"])
        rs = load_restart_state_from_h5(src, mesh_xy)
        tensors = {}
        for name in geom["present"]:
            if name == "V_qmunu":
                tensors[name] = rs.V_qmunu
            else:
                tensors[name] = read_munu_tensor_from_h5(src, name, mesh_xy)
        n_q = int(tensors["V_qmunu"].shape[0])
        mu_L_pad = int(tensors["V_qmunu"].shape[-1])

    print_fn(
        f"  [downfold] parent: mu_L={mu_L} (in memory {mu_L_pad}), "
        f"n_q={n_q}, nk={geom['nk']}, nb={geom['nb']}, "
        f"nspinor={geom['nspinor']}, kgrid={geom['kgrid']}")
    print_fn(f"  [downfold] retained window: {window.describe()}")
    if not window.symmetric:
        print_fn(
            "  [downfold] NOTE: the window is ASYMMETRIC.  That is the "
            "Sigma-serving shape — Sigma's internal band sum runs over the "
            "full window while its outer projection does not — and the rank "
            "probe measures it at about 2x the mu_S of the symmetric case, "
            "not the order of magnitude the design feared.  The algebra "
            "below is identical, but NO END-TO-END Sigma GATE HAS BEEN RUN "
            "on it: stage 1 was scoped BSE-first.  Treat the result as "
            "unvalidated.")
    if len(geom["present"]) > 2:
        print_fn(f"  [downfold] parent also carries "
                 f"{[n for n in geom['present'] if n not in ('V_qmunu', 'W0_qmunu')]}"
                 f" — they ride the same congruence.")

    # ---- D1: the parent Gram over the retained window -------------------
    with timing.section("downfold.grams", announce=True,
                        label="pair-density Grams"):
        S_LL = pair_density_gram(rs.psi_rmuT_X, rs.psi_rmu_Y, window,
                                 kgrid=geom["kgrid"], mesh_xy=mesh_xy)
        S_LL.block_until_ready()

    # ---- D2: CUR selection, with the rank refusal ------------------------
    with timing.section("downfold.select", announce=True,
                        label="CUR centroid selection"):
        # q = 0 is index 0 of the flat-q axis (the axis is the FFT's own
        # (kx,ky,kz) flattening), and the probe measured the retained rank to
        # be flat in q, so the q = 0 Gram is the right selection Gram.
        S_q0 = jax.lax.with_sharding_constraint(
            S_LL[0], NamedSharding(mesh_xy, P("x", "y")))
        keep_idx, sel = select_cur_centroids(
            S_q0, cfg.mu_small, rcond=cfg.downfold_rcond,
            select_tol=cfg.downfold_select_tol, mesh_xy=mesh_xy,
            mu_large_logical=mu_L, print_fn=print_fn)
        mu_S = int(sel.mu_small)
        mu_S_pad = int(padded_mu_extent(mu_S, int(jax.device_count())))
    print_fn(sel.describe())
    if sel.eigen_rank_kept < mu_S:
        print_fn(
            f"  [downfold/select] NOTE: {mu_S - sel.eigen_rank_kept} of the "
            f"{mu_S} kept centroids add no independent direction at "
            f"rcond={cfg.downfold_rcond:g}.  The transfer solve will "
            f"truncate about that many modes per q; the small basis is "
            f"smaller than it looks, and mu_small = auto would have said so "
            f"up front.")
    if mu_S_pad != mu_S:
        print_fn(
            f"  [downfold] mu_S {mu_S} -> {mu_S_pad} in memory (+"
            f"{mu_S_pad - mu_S} ZERO centroid rows so the extent divides "
            f"both mesh axes).  A zero psi row gives a zero pair density, "
            f"hence a zero row and column of every Gram and a zero "
            f"eigenvalue the truncation drops — the pad is inert, and disk "
            f"stores the logical {mu_S}.")

    # ---- the small basis's coefficients: a column slice ------------------
    psi_S_X, psi_S_Y = slice_psi_to_centroids(
        rs.psi_rmuT_X, rs.psi_rmu_Y, keep_idx, mu_S_pad, mesh_xy)

    with timing.section("downfold.grams_small", announce=False):
        S_cross = pair_density_gram(psi_S_X, rs.psi_rmu_Y, window,
                                    kgrid=geom["kgrid"], mesh_xy=mesh_xy)
        S_SS = pair_density_gram(psi_S_X, psi_S_Y, window,
                                 kgrid=geom["kgrid"], mesh_xy=mesh_xy)
        S_SS.block_until_ready()

    # ---- D3: the transfer solve ------------------------------------------
    with timing.section("downfold.solve", announce=True,
                        label="rank-truncated transfer solve"):
        T_x, T_y, reports = build_transfer(
            S_SS, S_cross, mesh_xy, rcond=cfg.downfold_rcond,
            print_fn=print_fn)
        T_x.block_until_ready()
    rank_per_q = np.array([r.rank_criterion for r in reports], dtype=np.int64)

    # ---- D4: the congruence ----------------------------------------------
    with timing.section("downfold.congruence", announce=True,
                        label="W_S = T W_L T-dagger"):
        project = congruence(mesh_xy, T_x, T_y)
        small = {name: project(arr) for name, arr in tensors.items()}
        g0_S = (transform_head_vector(rs.G0_mu_nu, T_x, 0, mesh_xy)
                if rs.G0_mu_nu is not None else None)
        small["V_qmunu"].block_until_ready()
    if g0_S is not None:
        print_fn(
            "  [downfold] g0_mu (zeta(G=0)) transported as g0_S = "
            "conj(T[q=0]) g0_L.  It is a one-index object, so it does not "
            "take the congruence — and the CONJUGATE is load-bearing: the "
            "q->0 head is the rank-1 s*conj(g0_mu)*g0_nu, and only "
            "conj(T) g0 makes the congruence of that matrix the same "
            "rank-1 form in the small basis.  This inherits the open "
            "'g0_mu placement' owner row on the zeta_loader ledger rather "
            "than answering it.")

    # ---- D5: the error bar ------------------------------------------------
    eps = {}
    norms = {}
    if cfg.report_residual:
        with timing.section("downfold.residual", announce=True,
                            label="Pythagorean error bar"):
            for name in ("W0_qmunu", "V_qmunu"):
                e, n_L, n_S = epsilon_w(tensors[name], S_LL, small[name],
                                        S_SS, mesh_xy)
                eps[name] = e
                norms[name] = (n_L, n_S)
        _announce_residual(eps, print_fn)
        worst = max(float(np.nanmax(v)) for v in eps.values())
        if (cfg.residual_refuse_above is not None
                and worst > float(cfg.residual_refuse_above)):
            raise ValueError(
                f"downfold: worst-q eps_W = {worst:.3e} exceeds the deck's "
                f"residual_refuse_above = {cfg.residual_refuse_above:g}, so "
                f"the small bundle was NOT written.  The retained window "
                f"holds directions the kept centroids do not span; raise "
                f"mu_small (the ceiling at this rcond is "
                f"{sel.eigen_rank_pool}) or widen the window.")
    else:
        print_fn(
            "  [downfold] report_residual = false: the per-q Pythagorean "
            "error bar was NOT computed, and this run therefore has no "
            "answer to 'did the downfold work' that does not require a "
            "reference calculation.  It costs two GEMMs at mu_L per q.")

    # ---- the write --------------------------------------------------------
    with timing.section("downfold.write", announce=True,
                        label="small restart bundle"):
        out_file = _write_small_bundle(
            cfg, geom, small, g0_S, rs.enk_full, psi_S_Y, keep_idx, mu_S,
            mesh_xy, sel=sel, rank_per_q=rank_per_q, eps=eps,
            print_fn=print_fn)

    cent_file = _write_centroid_subset(cfg, keep_idx, out_file,
                                       print_fn=print_fn)

    return DownfoldResult(
        source_file=src, output_file=out_file, mu_large=mu_L, mu_small=mu_S,
        n_q=n_q, n_bands=geom["nb"], keep_idx=keep_idx, selection=sel,
        rank_per_q=rank_per_q, eps_w=eps, norms=norms,
        wrote_nohead=tuple(n for n in small if n.endswith("_nohead")),
        centroid_file=cent_file)


def _announce_residual(eps: dict, print_fn) -> None:
    print_fn(
        "  [downfold/residual] eps_W(q) = sqrt(1 - ||W_S||^2/||W||^2), the "
        "relative error of the DOWNFOLDED OBSERVABLE on the retained "
        "window.  It is exact, not an estimate: the fit is an orthogonal "
        "projection, so Pythagoras holds and the residual needs no "
        "reference calculation and never forms the N x N observable.")
    for name, e in eps.items():
        finite = e[np.isfinite(e)]
        if finite.size == 0:
            print_fn(f"  [downfold/residual] {name}: no finite value "
                     f"(||W||^2 was zero at every q)")
            continue
        print_fn(
            f"  [downfold/residual] {name}: eps_W min/median/max = "
            f"{finite.min():.3e} / {np.median(finite):.3e} / "
            f"{finite.max():.3e} over {finite.size} q  "
            f"(worst at q={int(np.nanargmax(e))})")
    print_fn(
        "  [downfold/residual] q=0's absolute norms are head-dominated and "
        "should not be compared across q; the RATIO is still meaningful "
        "there because the head contaminates both norms identically.")


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

def _write_small_bundle(cfg, geom, small, g0_S, enk_full, psi_S, keep_idx,
                        mu_S, mesh_xy, *, sel, rank_per_q, eps, print_fn):
    out_dir = cfg.output_restart
    tmp_dir = os.path.join(out_dir, "tmp")
    if process_rank() == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    barrier("downfold.mkdir")
    out_file = os.path.join(tmp_dir, f"isdf_tensors_{mu_S}.h5")

    band_slices = (_BandSlices(geom["band_window"])
                   if geom["band_window"] is not None else None)
    policy = read_coulomb_policy_from_h5(
        resolve_restart_file(cfg.source_restart))

    # THE COULOMB POLICY IS INHERITED VERBATIM, and stamped.  The congruence
    # is LINEAR and applied AFTER the Dyson solve, so the head convention
    # (vhead, whead, mc_average_placement) rides through untouched — but two
    # bundles with different head conventions and no stamp is a
    # silent-disagreement bug of exactly the class the stamp exists to
    # prevent, so the small bundle carries the parent's policy string rather
    # than an empty one.
    write_restart_state_to_h5(
        out_file, n_rmu_logical=int(mu_S),
        V_qmunu=small["V_qmunu"], W0_qmunu=small["W0_qmunu"],
        G0_mu_nu=g0_S, enk_full=enk_full,
        mesh=mesh_xy, mode="w",
        kgrid=tuple(int(v) for v in geom["kgrid"]),
        band_slices=band_slices, coulomb_policy=policy)
    write_restart_state_to_h5(
        out_file, n_rmu_logical=int(mu_S), psi_full_y=psi_S,
        mesh=mesh_xy, mode="a")

    # The _nohead twins, when the parent had them.  Same congruence, same
    # writer; they are read-only opt-ins that nothing in-tree writes, so a
    # downfold that dropped them would quietly make an opt-in unavailable.
    for name in ("V_qmunu_nohead", "W0_qmunu_nohead"):
        if name in small:
            _append_munu(out_file, name, small[name], mu_S, mesh_xy)

    if geom["vhead"] is not None or geom["whead"] is not None:
        write_head_scalars_to_h5(
            out_file, vhead=geom["vhead"], whead=geom["whead"],
            omega_grid=geom["omega_grid"])

    _stamp_downfold_provenance(out_file, cfg, geom, keep_idx, mu_S,
                               sel=sel, rank_per_q=rank_per_q, eps=eps)
    barrier("downfold.bundle_written")
    print_fn(f"  [downfold] wrote {out_file}  "
             f"(mu {sel.mu_large} -> {mu_S}, "
             f"{(sel.mu_large / max(mu_S, 1)) ** 2:.0f}x smaller per "
             f"(mu,mu) tensor)")
    return out_file


def _append_munu(filename, name, arr, mu_S, mesh_xy):
    """Append one extra (nq, mu, mu) tensor under its own dataset name."""
    from file_io.slab_io import SlabIO
    shape = tuple(list(arr.shape[:-2]) + [int(mu_S), int(mu_S)])
    with SlabIO(filename, mode="a", mesh=mesh_xy) as io:
        io.create_dataset(name, shape=shape, dtype=arr.dtype)
        io.write_slab(name, arr)


def _stamp_downfold_provenance(filename, cfg, geom, keep_idx, mu_S, *,
                               sel, rank_per_q, eps):
    """Rank-0 group recording what this bundle IS and how it was made.

    A downfolded bundle is indistinguishable from a natively-fitted one by
    shape, and that is the point — but it is NOT the same object, and a
    reader that wants to know can ask.  Everything needed to reproduce the
    compression, or to map an index back to the parent, is here.
    """
    if process_rank() != 0:
        barrier("downfold.provenance")
        return
    with h5py.File(filename, "a") as f:
        if "downfold_provenance" in f:
            del f["downfold_provenance"]
        g = f.create_group("downfold_provenance")
        g.attrs["downfold_provenance_version"] = np.int64(
            DOWNFOLD_PROVENANCE_VERSION)
        g.attrs["mode"] = cfg.mode
        g.attrs["plan"] = cfg.plan
        g.attrs["parent_file"] = str(resolve_restart_file(cfg.source_restart))
        g.attrs["parent_mu"] = np.int64(sel.mu_large)
        g.attrs["mu_small"] = np.int64(mu_S)
        g.attrs["downfold_rcond"] = np.float64(cfg.downfold_rcond)
        g.attrs["downfold_select_tol"] = np.float64(sel.select_tol)
        g.attrs["window"] = np.asarray(
            [cfg.band_range_left[0], cfg.band_range_left[1],
             cfg.band_range_right[0], cfg.band_range_right[1]],
            dtype=np.int64)
        g.attrs["eigen_rank_pool"] = np.int64(sel.eigen_rank_pool)
        g.attrs["select_rank"] = np.int64(sel.select_rank)
        g.attrs["eigen_rank_kept"] = np.int64(sel.eigen_rank_kept)
        g.attrs["r19_anchor"] = R19_ANCHOR
        if geom.get("centroids_charge_md5") is not None:
            g.attrs["parent_centroids_charge_md5"] = geom[
                "centroids_charge_md5"]
        g.create_dataset("keep_idx", data=np.asarray(keep_idx, dtype=np.int64))
        g.create_dataset("retained_rank_per_q",
                         data=np.asarray(rank_per_q, dtype=np.int64))
        for name, e in eps.items():
            g.create_dataset(f"eps_w_{name}",
                             data=np.asarray(e, dtype=np.float64))
    barrier("downfold.provenance")


def _write_centroid_subset(cfg, keep_idx, out_file, *, print_fn):
    """The kept rows of the parent's centroid table, when we were given it.

    The bundle format carries no centroid COORDINATES — only a content hash
    of the table they came from — so a downfolded bundle is complete for
    every BSE consumer without this.  What it is NOT complete for is a fresh
    GW run on the small basis, which needs the points.  Writing them costs a
    slice, and stamping the new file's md5 is what lets the small bundle
    claim a centroid table honestly.  Stamping the PARENT's hash would be a
    lie about which points these tensors describe, so when no parent table is
    given the stamp is simply absent and the run says so.
    """
    if not cfg.parent_centroids_file:
        print_fn(
            "  [downfold] no parent_centroids_file given, so the small "
            "bundle carries no centroids_charge_md5.  It is complete for "
            "every BSE consumer (the bundle format holds no coordinates, "
            "only their hash); a GW run on the small basis would need the "
            "kept coordinates, and naming the parent's table in the input "
            "file is what produces them.")
        return ""
    # Validated on EVERY rank — a refusal raised only on rank 0 would hang
    # the others on the barrier below.  np.loadtxt of a centroid table is
    # host-local and cheap; only the write is rank-gated.
    src = cfg.parent_centroids_file
    rows = np.loadtxt(src)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[0] <= int(np.max(keep_idx)):
        raise ValueError(
            f"downfold: parent_centroids_file {src} has {rows.shape[0]} rows "
            f"but the kept index set reaches {int(np.max(keep_idx))}.  That "
            f"table is not the one the parent bundle's tensors were built "
            f"on, and slicing it would silently attach the wrong "
            f"coordinates to the right numbers.")
    out = os.path.join(os.path.dirname(out_file),
                       f"centroids_frac_{len(keep_idx)}_downfold.txt")
    if process_rank() == 0:
        np.savetxt(out, rows[np.asarray(keep_idx)], fmt="%.12f")
        with open(out, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        with h5py.File(out_file, "a") as f:
            f.attrs["centroids_charge_md5"] = md5
        print_fn(f"  [downfold] wrote {out} ({len(keep_idx)} kept centroids, "
                 f"md5 {md5[:12]}...) and stamped it on the bundle")
    barrier("downfold.centroids")
    return out
