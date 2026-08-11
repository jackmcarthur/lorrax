"""Arbitrary-Q bare-exchange tile V_Q — F-scheme + b26p interpolation backend.

Production port of the validated reference implementation
``runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study/``
``REFERENCE_arbitrary_q_vq.py`` (arbitrary_q_bse.md §§12-13, F_SCHEME_NOTE):
Tikhonov clean → Gaussian SR/LR kernel split → truncated-R stencil on the
cleaned SR tiles + ONE global b26p least-squares model of the long-range
form factors → closed-form assembly at any Q.  Arithmetic is preserved from
the reference (whose acceptance run reproduces the §13 pins to every printed
digit); this module adds the production seams:

  * ``prepare_coarse`` runs on-device q-BATCHED: the coarse q axis is
    sharded ``P(('x','y'))`` over all mesh devices and each device eighs +
    cleans + splits its own q-shard collective-free, in chunks with one
    host round-trip each (``auto`` = this batched native path; the
    distributed-FFI backends ``cusolvermp|slate`` stay available by
    explicit request for tiles too large to eigh replicated).
  * ``fit_lr_model`` (the b26p LSQ) stays HOST/replicated on purpose: the
    normal blocks are (n_b ≤ 10)² per (G_z, q) — a few kilobytes, O(10⁶)
    flops total — and the solved coefficients are (n_b, n_μ) ≈ tens of kB.
    Sharding them buys nothing and costs collectives; the fit inputs
    (``Fch``, ``W``) are per-q n_μ×|𝒢| blocks pulled to host once, offline.
  * ``make_eval_vq`` builds ONE jitted evaluator whose q-DEPENDENT data
    (target Q, the stencil-weight pseudo-inverse, the SR-tile stack, the
    fitted coefficients) are RUNTIME ARGUMENTS — never closure constants —
    so a whole Q-path is a single compile (the per-q-recompile lesson,
    PHASE2_LOG "Per-q recompile elimination").  Output tile is emitted at
    the loader's padded extent, sharded ``P('x','y')``.

THE PIPELINE (per-element math; μ = ISDF centroid, s_μ its fractional
coordinate, K = q+G Cartesian in bohr⁻¹, v = slab-truncated Coulomb):

  stage 1  ``prepare_coarse`` — offline, per coarse q_j, own frame:
    (a) Tikhonov clean WITHOUT forming Z:
            eigh: C_q = R diag(λ) R^H
            g_ε(λ) = λ² / (λ² + (ε_tik·λ_max)²)
            S_q    = R g_ε(λ) R^H                     (Hermitian, ~projector)
        so ζ_c = S_q ζ_stored and V_c = conj(S_q) V conj(S_q) — an analytic
        filter, NOT a hard cut (hard-cut projectors rotate freely inside
        C_q's gapless spectrum — Davis-Kahan; §12.3).
    (b) Gaussian SR/LR split per G channel:
            v_LR(K) = v(K) e^{−K²/4α²},   v_SR = v·(−expm1(−K²/4α²))
            V_SRc(q_j) = conj(S) [V_ref − V_LR] conj(S)
        LR confined to the FIXED Miller superset 𝒢(α) =
        {G : min_{q∈BZ, q_z=0} |q+G|² ≤ 4α² ln(1/ε_LR)}, intersected
        with the G_z channels the b26p model fits (``lr_fit_degrees``:
        an energy cutoff with a hard two-shell floor) — the rest are
        model-zero and contribute exact 0.0 (``lr_gset``).
    (c) phase-factored LR form-factor samples on 𝒢(α):
            F_μ(q_j;G) = e^{+2πi (q_j+G)·s_μ} (S_q ζ̃)_μ(q_j+G)
        (centroid winding phase carried analytically — the g0-winding cure).

  stage 2  ``fit_lr_model`` — ONE weighted LSQ over all coarse samples:
        M_μ(K_∥, G_z) ≈ Σ_b c_b[μ,G_z] (K_x/2α)^p (K_y/2α)^r,
        degree ladder {|G_z|=0:3, 1:2, 2:0, 3:0} → 26 complex coefficients
        per μ on a cell whose criterion stops at |G_z|=3; channels beyond
        the ladder enter at degree 0, one coefficient each
        TOTAL, weight w = v_LR(q+G) (the objective is then exactly
        ‖ΔA‖²_F of the LR tile factor A = ζ̃√v_LR).  Per-q normal blocks
        keep leave-one-out refits honest at O(n_b²).

  stage 3  ``eval_vq`` — cheap, per target Q, no solve / eigh / r_tot:
        w_j(Q) = e^{−2πi Q·R} · pinv(e^{−2πi q_j·R})       (nR7 stencil)
        V(Q)   = Σ_j w_j V_SRc(q_j)
               + conj(A) A^T,  A = e^{−2πi(Q+G)·s_μ} M_μ(Q+G) √v_LR(Q+G)

SCOPE — THE MODEL'S, NOT THE MODULE'S.  The b26p INTERPOLATION above is
slab-only (q_z = 0 coarse grids and slab-separable axes; the per-G_z channels
are exact nowhere else), and that half is ENFORCED by
:func:`slab_scope_violations`, from :func:`build_vq_evaluator` — the MODEL
build — before anything expensive.  It used to fire at load, and moved when
the refit below stopped being slab-bound; ``load_zeta_coarse`` still defaults
to ``require_slab=True`` and only the refit caller passes ``False``.  Both
paths need FULL-BZ stored ζ (nq == nk); IBZ-only storage (the IBZ cascade) is
rejected with a clear error — unfolding ζ through the one canonical SymMaps
sym-action is deferred work, not a parallel helper here.

The ground-truth alternative (``--vq-mode=refit`` in ``bse.exciton_bands``)
— a per-Q ζ refit from htransform full-r wavefunctions — lives in this
module too (``refit_vq``): both are V_Q sources with one calling contract.
NOTHING IN THE REFIT IS 2-D: it contracts with whatever kernel
:func:`make_v_on_set` hands it, which on a ``sys_dim=3`` deck is the
PRODUCER's own Coulomb door, so it is the arbitrary-Q exchange a bulk crystal
runs and not merely a checking mode.  Which band window it fits ζ' on is the
caller's ``--refit-window`` choice, and that choice picks the gate —
:func:`refit_window_view` is where both are written down.

Do NOT "improve" the model with: literal/pinned real-space moments (refuted
twice, §12.2/§13.3), SVD learned multipoles (no low rank, §13.2), hard-cut
cleaning in the fit gauge (the q-fiber IS the cut edge, §13.1), multi-width
GTO ladders (conditioning, §13.2).
"""
from __future__ import annotations

import os
from functools import partial

import h5py
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.fft_helpers import local_fftn3
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from distrib_la import plan as linalg_plan                          # noqa: E402

# pipeline constants (§13.5 production shape; reference values verbatim)
ALPHA = 0.30          # Gaussian split width, 1/bohr; broad optimum ~1.5-2x dq
EPS_TIK = 1e-4        # relative Tikhonov filter width (fit gauge, §13.1)
EPS_LR = 1e-8         # Gaussian weight bound defining the LR G-superset
DEG_B26P = {0: 3, 1: 2, 2: 0, 3: 0}   # in-plane poly DEGREE per |G_z|
RIDGE = 1e-11         # normal-equation ridge (lr_prep.ChannelFit.RIDGE)
RY2MEV = 13605.693

# Which |G_z| channels the long-range model FITS -- see :func:`lr_fit_degrees`.
# ``DEG_B26P`` above is the degree LADDER (how rich the in-plane polynomial is
# per channel); it no longer decides WHICH channels exist.  Channels above its
# top entry enter at degree 0, one complex coefficient each.
E_CUT_FIT = 1.0       # Ry -- default channel-fitting cutoff (validated below)
FIT_SHELL_FLOOR = 2   # hard floor: |G_z| shells always fitted (umklapp roll)
FIT_FLOOR_MARGIN = 1e-6   # lifts the floor off exact equality at |n| = floor


def relF(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# Sharding that is legal for THIS deck's extents.  n_mu is a k-means output
# and nq is the BZ q-count; neither is rounded to the mesh here (bse_io rounds
# its own copy), so both refuse to shard on most decks.  See
# common/sharding_fit.py for the measured evidence and the durable fix.
from common.sharding_fit import fit_sharding as _ns              # noqa: E402
from common.sharding_fit import legal_spec as _legal_spec        # noqa: E402

# The ζ file's DOOR.  TOP-LEVEL name only: ``zeta_loader.loader`` would be
# a past-the-door edge and ``tests/test_layering.py`` fails on those.  The
# ``ensure_on_path()`` above is what makes it resolvable in a bare launch.
from zeta_loader import ZetaLoader                               # noqa: E402


# ===========================================================================
# ζ transport — ONE reader for zeta_q.h5, TWO plans over it
# ===========================================================================
def _zeta_mesh_for_loader(mesh, log_fn=print):
    """``(mesh_or_None, distributed)`` — the ζ transport decision, once.

    ``ZetaLoader``'s data path is SlabIO, which REFUSES at open on a
    stack whose phdf5 FFI is absent (``file_io.slab_io``'s module
    docstring: there is one transport and nothing to demote to).  So the
    decision of whether to hand the loader a mesh has to be taken BEFORE
    constructing it, and it is taken here, once, and announced — because
    "which transport ran" is the single most consequential fact about a
    large run's I/O and a silent fallback is indistinguishable from a
    hang (same reasoning, and the same probe, as
    ``bse_io._bse_slabio_usable``).

    Header-only (``None``) does NOT mean "no reader": the loader still
    owns every metadata read on ``zeta_q.h5``.  It means the ζ TILES come
    back through the local h5py plan, which INVARIANTS row 6 licenses as
    the default — the defect that row records is a family with ONLY a
    local plan, and after this change ``vq_interp`` has both.
    """
    if mesh is None:
        return None, False
    from file_io.slab_io import probe_availability
    ok, stage, reason = probe_availability()
    if not ok:
        log_fn(f"  [vq_interp] SlabIO unavailable at probe stage '{stage}' "
               f"({reason}); reading ζ with the local h5py q-hyperslab plan "
               f"— memory-correct (one q-chunk per rank, no allgather) and "
               f"un-sharded, so every rank reads the whole chunk.")
        return None, False
    return mesh, True


class _ZetaGTiles:
    """The lazy ``(nq, n_mu, ngkmax)`` ζ(G) stack, read through ONE owner.

    ``zeta_q_G`` is 47.8 GB at the converged MoS2 reference and every
    consumer slices it on q, so it is never materialised: this object is
    a HANDLE, not an array.  It replaces the raw ``h5py`` dataset the
    module used to stash in ``zx["ZG"]`` and keeps that dataset's
    indexing surface (``.shape``, ``[q]``, ``[q0:q1]`` → host numpy) so
    no consumer had to change, plus ONE new call that the h5py dataset
    could not express:

    * :meth:`read_q_slab` — the DISTRIBUTED plan.  ``ZetaLoader.load``
      → ``SlabIO.read_slab`` hyperslabs **only this rank's shard** of the
      q-chunk straight into the target sharding.  The call it replaces
      (``device_put_process_local(ds[sl], qb3)``) read the WHOLE chunk
      into host numpy on EVERY rank and then threw away all but its own
      shard: at nq=144 / n_μ=2412 / ngkmax=8603 a 48-q chunk is 15.9 GB,
      per rank.
    * ``__getitem__`` — the LOCAL plan, unchanged in what it returns:
      ``ZetaLoader.read_zeta_G_local(key)`` is the same h5py hyperslab,
      the same host numpy, the same bytes.  It is ALSO the only plan when
      the SlabIO probe declines.

    The layout CONTRACT is read once, from the loader
    (``gvec_components``/``ngk``/``ngkmax``, the sentinel Miller pad,
    ``zeta_q_G[q, :, ngk[q]:] == 0``); this module no longer re-derives
    any of it from raw datasets.  BOTH HANDLES BELONG TO THE LOADER now —
    the collective SlabIO one and the local plan's serial h5py one — and
    :meth:`close` ends them by closing it, instead of this object holding
    a second ``h5py.File`` of its own and the convention that nobody
    drops ``zx`` keeping the rest alive.

    WHY ``__getitem__`` IS NOT A SlabIO READ, AND WHERE THAT NOW LIVES.
    A SlabIO read is COLLECTIVE over the mesh and returns a ``jax.Array``
    whose requested shape must be mesh-divisible under its
    ``partition_spec``: a single-q ``(1, n_mu, ngkmax)`` read cannot be
    q-sharded at all, a replicated one materialises the same bytes on
    every rank plus a device round-trip, and putting a collective behind
    ``ds[q]`` would turn any future rank-0-only diagnostic into a hang
    instead of an error.  That argument is no longer this module's to
    make: it is the documented contract of
    :meth:`zeta_loader.ZetaLoader.read_zeta_G_local` — *local by design,
    per-rank independent, do not make this collective* — because the plan
    it describes moved INTO the door, where the one owner of the file can
    hold it to that promise.  What stays here is the consequence: every
    ``__getitem__`` caller in this module is a replicated host diagnostic
    (``recon``, ``run_gates``, ``run_nulls``); those are the mirrors
    ledger row 64 is about, and replacing them with on-device reductions
    is a diagnostics rewrite, not a transport change.
    """

    def __init__(self, loader: ZetaLoader, *, path: str, distributed: bool):
        self._loader = loader
        self._distributed = bool(distributed)
        if loader.zeta_layout != 'G_flat':
            raise ValueError(
                f"vq_interp needs a G-flat ζ ('zeta_q_G'); {path} has "
                f"zeta_layout={loader.zeta_layout!r}.  Refit with the "
                f"G-flat writer (gw.isdf_fitting).  There is no longer a "
                f"read path for r-space ζ at all: ZetaLoader's disk→G FFT "
                f"+ sphere gather was deleted on 2026-08-07 because no "
                f"writer in the tree emits that layout.")
        self.shape = (int(loader.n_q_on_disk), int(loader.n_rmu_disk),
                      int(loader.n_G_sph_disk))
        self.dtype = np.complex128
        # The header-vs-dataset ngkmax agreement check is GONE from here, not
        # dropped: ``ZetaLoader.__init__`` enforces it at OPEN, which is the
        # only place that can, since BOTH plans that could disagree about
        # ngkmax (header-sized collective, dataset-sized local) are its.

    # -- local plan (host numpy, h5py hyperslab; unchanged semantics) ---
    def __getitem__(self, key):
        """``zeta_q_G[key]`` as host numpy — the loader's serial handle.

        Delegation, not a re-implementation: ``read_zeta_G_local`` returns
        exactly what ``dataset[key]`` returns for any h5py key, and the
        service pins that byte-for-byte against a raw handle.  The
        post-close refusal is the LOADER's now (this used to be a local
        ``self._ds is None`` test with its own message), which is what
        makes "closed" one fact about one owner instead of two objects
        each with a private opinion.
        """
        return self._loader.read_zeta_G_local(key)

    # -- distributed plan (per-rank hyperslab straight into `sharding`) -
    def read_q_slab(self, q_offset: int, q_count: int, *, sharding):
        """``(q_count, n_mu, ngkmax)`` on ``sharding``.

        Distributed: ``ZetaLoader.load`` → ``SlabIO.read_slab`` with a
        per-rank hyperslab.  Local: the h5py chunk placed with
        ``device_put_process_local`` — bit-identical, since both plans
        return the same on-disk elements and neither reduces.
        """
        q_offset, q_count = int(q_offset), int(q_count)
        if self._distributed:
            return self._loader.load(
                q=np.arange(q_offset, q_offset + q_count, dtype=np.int32),
                sharding=sharding.spec)
        return device_put_process_local(
            self[q_offset:q_offset + q_count], sharding)

    # -- ownership -----------------------------------------------------
    def close(self) -> None:
        """Release the ζ handles.  Idempotent; post-close reads REFUSE.

        Both handles are the loader's, so closing it is what ends them —
        and the refusal a later ``ZG[q]`` gets is
        ``ZetaLoader.read_zeta_G_local``'s ("…is closed; its local ζ reads
        are no longer serviceable"), not this class's old private message.
        ``close_zeta_coarse`` still calls ``loader.close()`` right after
        this; ``ZetaLoader.close`` is idempotent, so the second call is a
        no-op and the ownership statement stays true from either end.
        """
        self._loader.close()


# ===========================================================================
# SCOPE — is this deck a slab at all?
# ===========================================================================
def slab_scope_violations(bvec, qfr=None, policy=None) -> list[str]:
    """Every way this deck falls outside ``vq_interp``'s slab scope, in prose.

    Empty list = in scope.  Pure: plain arrays and the parsed Coulomb-policy
    dict in, strings out, no file handles — so the conditions are testable on
    cell geometry alone, which is how they are gated
    (``tests/test_bse_vq_interp_scope.py``).

    THE MODULE IS SLAB-ONLY IN TWO INDEPENDENT WAYS, and until 2026-08-10
    neither said so out loud.

    1. **The kernel is hardwired.**  :func:`v_slab_on_set` — every ``v`` in
       this module, the SR/LR split included — is the Ismail-Beigi 2-D
       truncation ``8π/K² · f2d / Ω`` with ``f2d = 1 − e^{−z_c|K_∥|}
       cos(K_z z_c)``, ``z_c = π/b3_z``.  There is no ``sys_dim`` anywhere in
       this file.  A deck with ``sys_dim = 3`` built its stored ``V_qmunu``
       from the BULK kernel (``vcoul.bulk_3d``, no ``f2d``), so the two are
       different operators and ``run_gates``' ``makeVq_vs_disk`` compares
       apples to oranges.  MEASURED on the ``si_bse_debug`` parent
       (``/pscratch/sd/j/jackm/xd_parent``, nq = 64, n_μ = 936): the slab
       kernel gives **3.218e-01** against a 5e-6 tolerance — the number
       ``PIPELINE_HEALTH.md`` punch row 23 opened with — and rebuilding the
       same tiles with the bulk kernel instead drops it to **4.593e-02**,
       with the remainder attributable to the deck's own
       ``mc_average_vcoul_body = true`` mini-BZ head-slot injection, which
       this module does not model either.  Neither number is evidence about
       the interpolation: ``makeVq_vs_disk`` is an ON-GRID check of the
       stored ζ against the stored tiles and runs before any fit.

    2. **The long-range model's channels only exist on a slab.**  Stage 2
       fits ``M_μ(K_x, K_y)`` once per ``|G_z|`` channel.  That is exact only
       when ``K_z`` is CONSTANT within a channel, which needs ``b3 ∥ z``,
       ``b1, b2 ⊥ z`` AND ``q_z = 0`` on the coarse grid — then
       ``K_z = G_z·|b3|`` identically.  On an fcc cell neither holds:
       ``si_bse_debug``'s ``b3 = 0.6123·(1, −1, 1)`` has in-plane components
       of exactly its own length, so ``run_gates``' ``slab_axes_offdiag``
       reads **1.000e+00** — not a small number that might be tightened away,
       but the ratio a cubic reciprocal lattice takes by construction.  The
       in-plane polynomial would be fitting a ``K_z`` it cannot see.

    So a bulk deck is not a tuning problem, and no change to the fit reaches
    it.  ``--vq-mode ongrid`` (exact at every Q on the BSE grid) and
    ``--vq-mode refit`` (the per-Q ζ refit) are the modes that serve one.

    ``policy`` is :func:`file_io.read_coulomb_policy_from_h5`'s dict, or
    ``None`` for a restart written before the stamp; an unstamped file is
    judged on geometry alone, which is the safe direction — condition 2 is
    the one that cannot be worked around, and it needs no stamp to see.
    """
    out = []
    bvec = np.asarray(bvec, dtype=np.float64)
    sys_dim = (policy or {}).get("sys_dim", "")
    if str(sys_dim).strip() not in ("", "2"):
        out.append(
            f"the restart's Coulomb-policy stamp says sys_dim={sys_dim}, but "
            f"this module builds every v(q+G) with the 2-D Ismail-Beigi slab "
            f"truncation (v_slab_on_set) and has no other kernel; the stored "
            f"V_qmunu was built with the sys_dim={sys_dim} kernel, so the two "
            f"are different operators")
    # ``b3_z`` is the denominator of ``run_gates``' ``slab_axes_offdiag`` AND
    # of the truncation length ``z_c = π/b3_z``; a cell with no z-projection
    # on b3 at all has neither, so it is named rather than divided by.
    b3z = abs(float(bvec[2, 2]))
    if b3z <= 1e-12 * float(np.max(np.abs(bvec))):
        out.append(
            f"b3 has no z-projection to speak of (|b3_z| = {b3z:.3e}), so the "
            f"slab truncation length z_c = π/b3_z is undefined and there is no "
            f"axis for the |G_z| channels to be channels of")
    else:
        offdiag = float(max(np.max(np.abs(bvec[2, :2])),
                            np.max(np.abs(bvec[:2, 2]))) / b3z)
        if offdiag > 1e-12:
            out.append(
                f"the cell's axes are not slab-separable: max|b3_xy|/|b3_z| "
                f"and max|b1_z, b2_z|/|b3_z| reach {offdiag:.3e} "
                f"(slab_axes_offdiag, tolerance 1e-12), so K_z is not constant "
                f"within a |G_z| channel and the per-channel in-plane fit has "
                f"an unmodelled variable")
    if qfr is not None:
        qz = float(np.max(np.abs(np.asarray(qfr, dtype=np.float64)[:, 2])))
        if qz > 1e-12:
            out.append(
                f"the coarse q-grid is not planar: max|q_z| = {qz:.3e}, so "
                f"K_z = q_z·b3_z + G_z·|b3| varies across the samples of one "
                f"|G_z| channel even on slab-separable axes")
    return out


def assert_slab_scope(bvec, qfr=None, policy=None, *, source="") -> None:
    """Refuse an out-of-scope deck by name.  See :func:`slab_scope_violations`.

    THE ONE SITE, and it is :func:`build_vq_evaluator` that calls it — the
    b26p MODEL build, which is the thing that is slab-only — still before
    anything expensive, so the refusal arrives instead of ``build_cq`` + a
    gate battery whose numbers are consequences rather than causes.  The
    refit caller loads ζ with ``require_slab=False`` and never reaches here.
    """
    why = slab_scope_violations(bvec, qfr=qfr, policy=policy)
    if not why:
        return
    where = f" ({source})" if source else ""
    raise ValueError(
        "vq_interp is a SLAB model and this deck is not a slab" + where
        + ".  " + "  ".join(f"({i + 1}) {w}." for i, w in enumerate(why))
        + "  None of these is a tuning problem and no change to the long-range "
          "fit reaches them.  On a bulk deck use `--vq-mode ongrid` (exact, "
          "but only at Q on the BSE grid) or `--vq-mode refit` (the per-Q "
          "zeta refit, exact at ANY Q: it takes the producer's own Coulomb "
          "door on a sys_dim=3 deck -- make_v_on_set -- and the driver refits "
          "every path Q behind an on-grid null).  See the module docstring's "
          "SCOPE note and PIPELINE_HEALTH.md punch row 23, which is this "
          "refusal's measured history.")


# ===========================================================================
# coarse-data loading (reference load_fixture, with paths as arguments)
# ===========================================================================
def load_zeta_coarse(restart_file: str, zeta_file: str, *,
                     mesh: Mesh | None = None, log_fn=print,
                     require_slab: bool = True) -> dict:
    """Load the coarse-grid ζ/ψ/tile data into a plain-dict bundle ``zx``.

    q-LABELING (the two wrap traps, KNOWN_SANDBOX_ERRORS 2026-07-17):
    ``zeta_q.h5 mf_header/kpoints/rk`` stores the UNWRAPPED QE list, while
    the stored ζ spheres are centred on the BGW-WRAPPED q (worth a measured
    155× on the physical interp ladder).  ``np.round`` is round-half-to-even,
    so components at exactly 1/2 need the sphere itself to pick the sign:
    keep the candidate wrap minimising max|q+G|² over the stored sphere
    (``_fix_sphere_wrap``).  Every downstream phase/kernel uses these
    wrapped labels.

    Requires FULL-BZ ζ storage (nq == nk).  IBZ-only storage (the IBZ
    cascade) raises — see the module docstring.

    THE THREE BIG q-STACKS STAY ON DISK.  ``ZG`` (nq, n_μ, ngkmax),
    ``Vqmunu`` and ``W0`` (nq, n_μ, n_μ) are read-only and every consumer
    slices them on the q axis, so they are kept as LAZY handles and
    pulled per q-chunk instead of being materialised per process.  At
    the converged MoS2 reference (nq = 144, n_μ = 2412, ngkmax = 8603) ζ
    alone is **47.8 GB**; four ranks per Perlmutter GPU node (251 GB) cannot
    hold it, which is what confined the exciton driver to ``--vq-mode
    ongrid``.  Lazy, the resident host cost is ψ (3.7 GB) plus one q-chunk.
    Same principle as the ψ(G) host cache in the GW path: large read-only
    caches are pulled per slice, never carried.

    READERS AND OWNERSHIP.  ``zeta_q.h5`` is read by exactly ONE object,
    :class:`zeta_loader.ZetaLoader` — every header/metadata value
    below is one of its attributes, the ζ tiles come through
    :class:`_ZetaGTiles`, which wraps that same loader, and since the V4
    replumb the tiles' serial h5py handle is the loader's too — so this
    module does not open ``zeta_q.h5`` at all any more.  It used to open
    it itself and re-derive the G-flat layout contract
    (per-q ``gvec_components`` padded with the sentinel Miller index,
    ``ngk``, ``ngkmax``, the ``zeta_q_G[q, :, ngk[q]:] == 0``
    guarantee) from raw datasets, which made it a SECOND independent
    reader of a layout only ``ZetaLoader``/``common.coulomb_sphere``
    define.  ``restart_file`` (``psi_full_y`` / ``V_qmunu`` /
    ``W0_qmunu`` / ``enk_full``) has no such owner — it is not a ζ file
    and no loader class covers it — so its handle is still a raw
    ``h5py.File``, but it is now closed explicitly by
    :func:`close_zeta_coarse` rather than kept alive by the convention
    that nobody drops ``zx``.

    ``mesh`` selects the ζ TRANSPORT (see :func:`_zeta_mesh_for_loader`):
    with a mesh whose stack can serve SlabIO, ``prepare_coarse``'s q-chunk
    read becomes a per-rank hyperslab; without one, the local h5py plan
    runs, byte-identical.  It is optional so the host-only diagnostics and
    the fixture tests keep working on a bare checkout.
    """
    # LAZY, and matching ``bse_io``'s own lazy ``from . import vq_interp``:
    # the two modules reference each other and neither may be the one that
    # forces the other at import time.
    from . import bse_io
    zx = {"restart_file": restart_file, "zeta_file": zeta_file}
    # NOT a context manager: the handles outlive this call so the lazy
    # reads below stay serviceable.  ``zx`` owns them; drop them with
    # ``close_zeta_coarse(zx)``.
    fr = h5py.File(restart_file, "r")
    zx["_h5_restart"] = fr
    zx["psi"] = fr["psi_full_y"][()]          # (nk, nb, ns, n_mu) u at centroids
    zx["kgrid"] = fr["kgrid"][()].astype(int)
    # LAZY — disk tiles (gate reference) — UNLESS the file stores the q
    # wedge, in which case the tiles do not exist on disk and the unfold
    # runs once, here, through ``bse_io.restart_munu_full_bz`` (the one
    # seam; see its docstring for why all-at-once is the design and not a
    # shortcut).  On every legacy and full-BZ file this binds the h5py
    # handle exactly as it always did, so the lazy read path is unchanged
    # for every restart file that exists today.
    zx["Vqmunu"] = (bse_io.restart_munu_full_bz(fr["V_qmunu"], "V_qmunu",
                                                restart_file)
                    if bse_io.is_q_wedge(fr["V_qmunu"]) else fr["V_qmunu"])
    # PRESENCE IS NOT PERSISTENCE.  ``gw_init`` allocates a full-size ZERO
    # ``W0_qmunu`` unconditionally (``tagged_arrays.write_restart_state_to_h5``
    # with ``init_W0=True``), so ``"W0_qmunu" in fr`` is true on a run whose
    # ``persist_w0`` never fired and the tiles handed to ``build_hdir`` are
    # zeros of exactly the right shape.  That is the April all-zero-screening
    # mechanism: a plausible excitonic spectrum out of a W that was never
    # written, with every shape check green.  ``tagged_arrays`` already stamps
    # ``W0_ready`` (False for the placeholder, True when real data lands) and
    # ``bse_io`` already gates on it in both of its readers; this path asks
    # the same question instead of asking whether the dataset exists.
    if "W0_qmunu" in fr and bool(fr["W0_qmunu"].attrs.get("W0_ready", False)):
        # Same wedge question as V above, and the GUARD IS UNTOUCHED on
        # purpose: ``test_bse_w0_ready_gate``'s ratchet requires every
        # ``if`` binding ``zx["W0"]`` to test the persisted flag, so the
        # q-storage branch goes on the right-hand side (a conditional
        # EXPRESSION, not a nested ``if``) where it cannot teach that
        # matcher an exception it would then carry forever.
        zx["W0"] = (bse_io.restart_munu_full_bz(fr["W0_qmunu"], "W0_qmunu",
                                                restart_file)
                    if bse_io.is_q_wedge(fr["W0_qmunu"])
                    else fr["W0_qmunu"])   # LAZY — screened tiles (Hdir)
    zx["enk"] = fr["enk_full"][()]            # (nk, nb) Ry
    _mesh_for_loader, _distributed = _zeta_mesh_for_loader(mesh, log_fn=log_fn)
    zl = ZetaLoader(zeta_file, mesh=_mesh_for_loader)
    zx["_zeta_loader"] = zl
    zx["zeta_distributed"] = _distributed
    zx["ZG"] = _ZetaGTiles(zl, path=zeta_file, distributed=_distributed)
    # ``ngk_per_q`` is ``isdf_header/ngk`` (the per-q ζ SPHERE size).
    # ``zl.ngk`` is a DIFFERENT array — ``mf_header/kpoints/ngk``, the
    # WFN's per-k G count, bound by ``bind_mf_attrs``.  Reading the wrong
    # one truncates every sphere silently.
    zx["gvec"] = np.asarray(zl.gvec_components).astype(np.int64)
    zx["ngk"] = np.asarray(zl.ngk_per_q).astype(int)
    fg = np.asarray(zl.fft_grid).astype(int)
    qraw = np.array(zl.kpoints, copy=True)
    zx["adot"] = np.asarray(zl.adot)
    blat = float(np.real(zl.blat))
    # BGW stores bvec in units of blat = 2π/alat; physical bohr⁻¹
    # (|bvec^T g|² in Ry) needs the blat factor (measured: 10.4%
    # makeVq-vs-disk residual without it).
    zx["bvec"] = np.asarray(zl.bvec) * blat
    zx["celvol"] = float(np.real(zl.cell_volume))
    rmu_idx = np.asarray(zl.r_mu_fft_idx).astype(int)
    zx["zeta_cutoff"] = float(zl.zeta_cutoff_ry)
    ifmax = np.asarray(zl.ifmax)
    zx["nk"], zx["nb"], zx["ns"], zx["n_mu"] = zx["psi"].shape
    zx["nq"] = zx["ZG"].shape[0]
    zx["ngkmax"] = zx["ZG"].shape[2]
    zx["nx"], zx["ny"], zx["nz"] = [int(x) for x in fg]
    zx["n_rtot"] = zx["nx"] * zx["ny"] * zx["nz"]
    if zx["nq"] != zx["nk"]:
        raise ValueError(
            f"vq_interp needs FULL-BZ zeta storage: zeta_q.h5 has nq={zx['nq']} "
            f"but the k-grid has nk={zx['nk']} (IBZ cascade active).  "
            f"Regenerate the fit with full-BZ zeta, or wait for the IBZ-zeta "
            f"unfold (deferred; must route through the one SymMaps sym-action).")
    # ── q LABELS FOR A FULL-BZ ζ WRITTEN FROM A SYMMETRY-REDUCED WFN ──────
    # ``mf_header`` is copied verbatim from the WFN, so ``kpoints/rk`` holds
    # the WFN's k-list — the IBZ when the mean-field run used symmetry.  The ζ
    # writer, under ``LORRAX_FORCE_FULL_BZ=1``, writes the FULL BZ:
    # ``_bgw_wrap_q(sym.kvecs_asints) / kgrid`` (gw/isdf_fitting.py, the
    # ``q_irr_frac is None`` branch).  On the MoS2 4x4 deck that is 16 ζ tiles
    # against a 10-row ``rk``, and ``qraw[:nq]`` silently returned 10 rows —
    # surfacing three lines later as the misleading "duplicate k labels in rk
    # list" (job 7882499 cell exb64s).  Reconstruct the writer's own list.
    #
    # THIS IS NOT TAKEN ON TRUST.  ``run_gates`` rebuilds V from ζ at EVERY q
    # and compares it against the stored ``V_qmunu[q]`` at 5e-6
    # (``makeVq_vs_disk_Vqmunu_allq_max``).  A permuted or mis-wrapped q list
    # cannot pass that: each q's ζ would be checked against a different q's
    # stored tile.  Do not run this path with ``LORRAX_SKIP_VQ_GATES=1`` until
    # it has passed once on a given deck.
    if qraw.shape[0] < zx["nq"]:
        _kg = np.asarray(zx["kgrid"], dtype=np.float64)
        _idx = np.stack(np.meshgrid(np.arange(_kg[0]), np.arange(_kg[1]),
                                    np.arange(_kg[2]), indexing="ij"),
                        axis=-1).reshape(-1, 3).astype(np.float64)
        _wrapped = np.where(_idx > _kg[None, :] / 2.0, _idx - _kg[None, :], _idx)
        qfull = _wrapped / _kg[None, :]
        # Necessary condition: every k the mean-field header DOES carry must
        # appear in the reconstruction (catches a transposed grid or the wrong
        # wrap convention, both of which would otherwise reach run_gates as a
        # confusing numerical failure).
        _have = {tuple(np.rint(v * _kg).astype(int) % _kg.astype(int))
                 for v in qfull}
        _missing = [tuple(np.rint(v * _kg).astype(int) % _kg.astype(int))
                    for v in qraw if tuple(np.rint(v * _kg).astype(int)
                                           % _kg.astype(int)) not in _have]
        if _missing:
            raise ValueError(
                f"reconstructed full-BZ q list does not contain "
                f"{len(_missing)} of the {qraw.shape[0]} mf_header k-points "
                f"(e.g. {_missing[0]}); the on-disk q ordering is not the "
                f"C-order wrapped {tuple(int(v) for v in zx['kgrid'])} grid "
                f"this reconstruction assumes.")
        try:
            _first = jax.process_index() == 0
        except Exception:
            _first = True
        if _first:
            print(f"  [vq_interp] zeta_q.h5 holds {zx['nq']} full-BZ q but "
                  f"mf_header/kpoints/rk has only {qraw.shape[0]} (the WFN is "
                  f"symmetry-reduced).  q labels reconstructed as the BGW-"
                  f"wrapped C-order {tuple(int(v) for v in zx['kgrid'])} grid; "
                  f"run_gates' per-q makeVq-vs-disk check verifies it.",
                  flush=True)
        qraw = qfull
    zx["qfr_raw"] = qraw[: zx["nq"]]
    zx["qfr"] = zx["qfr_raw"] - np.round(zx["qfr_raw"])  # BGW-wrapped, pre half-fix
    kg = zx["kgrid"]
    zx["k_int"] = np.rint(zx["qfr_raw"] * kg[None, :]).astype(int) % kg[None, :]
    zx["k_lookup"] = {tuple(v): i for i, v in enumerate(zx["k_int"])}
    assert len(zx["k_lookup"]) == zx["nq"], "duplicate k labels in rk list"
    rx = np.arange(zx["nx"]) / zx["nx"]
    ry = np.arange(zx["ny"]) / zx["ny"]
    rz = np.arange(zx["nz"]) / zx["nz"]
    RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing="ij")
    zx["rfrac"] = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
    dims = np.array([zx["nx"], zx["ny"], zx["nz"]])
    zx["rmu_frac"] = rmu_idx / dims[None, :]     # centroid frac coords s_μ
    zx["rmu_flat"] = ((rmu_idx[:, 0] * zx["ny"]) + rmu_idx[:, 1]) * zx["nz"] \
        + rmu_idx[:, 2]
    zx["nv"] = int(ifmax.ravel()[0])
    assert np.all(ifmax == zx["nv"]), "ifmax not uniform over k"
    _fix_sphere_wrap(zx)
    # SCOPE, before anything expensive.  The stamp is scalar metadata read
    # with serial h5py (safe on every rank, no SlabIO handle), and it is the
    # deck's own record of which Coulomb kernel built ``V_qmunu`` — the one
    # fact this module cannot derive from geometry.  NOT re-announced here:
    # ``bse_io`` already prints ``describe_coulomb_policy_stamp`` once per
    # driver run, and the row-23 log shows that line sitting one screen above
    # the gate failures it explained.  The stamp was never missing; nothing
    # READ it.  So the fix is a refusal that quotes it, not a second copy.
    from file_io import read_coulomb_policy_from_h5
    zx["policy"] = read_coulomb_policy_from_h5(restart_file)
    # ``require_slab=False`` is the REFIT caller, and it is not a bypass of
    # the scope check — it is the observation that the scope check is about a
    # part of this module the refit never touches.  Both slab-only facts
    # (:func:`slab_scope_violations`) are properties of the b26p long-range
    # MODEL and of ``v_slab_on_set``; the refit fits ζ at the target Q and
    # contracts it with the kernel :func:`make_v_on_set` hands it, which on a
    # bulk deck is the producer's own.  The check therefore moved to
    # ``build_vq_evaluator`` — the model build — rather than being weakened,
    # so an interp/both run on a bulk deck refuses exactly as loudly as
    # before, one call later and before anything expensive still.
    if require_slab:
        assert_slab_scope(zx["bvec"], qfr=zx["qfr"], policy=zx["policy"],
                          source=restart_file)
    return zx


def close_zeta_coarse(zx: dict) -> None:
    """Release every handle :func:`load_zeta_coarse` opened.

    EXPLICIT ownership, because the three lazy q-stacks are read-only HANDLES
    and have to outlive the loader call — so ending them cannot be left to
    CPython refcounting on ``zx``.  ``zx`` remains usable for everything
    already materialised (ψ, ε, the q labels, the kernels); only the lazy
    reads (``ZG``, ``Vqmunu``, ``W0``) stop.

    Idempotent, and safe on a partially-built bundle.
    """
    tiles = zx.pop("ZG", None)
    if tiles is not None and hasattr(tiles, "close"):
        tiles.close()
    loader = zx.pop("_zeta_loader", None)
    if loader is not None:
        loader.close()
    fr = zx.pop("_h5_restart", None)
    if fr is not None:
        fr.close()
    zx.pop("Vqmunu", None)
    zx.pop("W0", None)


def _fix_sphere_wrap(zx):
    """(reference ``_fix_sphere_wrap``) Half-boundary wrap disambiguation:
    per q, among the ±1/2 sign candidates keep the one whose sphere fits
    max|q+G|² ≤ cutoff.  No-op on grids without half components."""
    changed = 0
    for q in range(zx["nq"]):
        base = zx["qfr_raw"][q] - np.round(zx["qfr_raw"][q])
        cands = [[]]
        for c in range(3):
            opts = [0.5, -0.5] if abs(abs(base[c]) - 0.5) < 1e-9 else [base[c]]
            cands = [cc + [o] for cc in cands for o in opts]
        n = int(zx["ngk"][q])
        G = zx["gvec"][q][:, :n].astype(np.float64)
        best, bestm = None, None
        for cc in cands:
            qc = np.asarray(cc)
            K = zx["bvec"].T @ (qc[:, None] + G)
            m = float(np.max(np.sum(K * K, axis=0)))
            if bestm is None or m < bestm:
                best, bestm = qc, m
        assert bestm <= zx["zeta_cutoff"] + 1e-9, \
            f"q={q}: no candidate wrap fits the stored sphere"
        if np.max(np.abs(best - zx["qfr"][q])) > 1e-12:
            changed += 1
        zx["qfr"][q] = best
    if changed:
        print(f"  [wrapfix] {changed} of {zx['nq']} q relabeled to the "
              f"sphere-derived center")


# ===========================================================================
# grid / sphere / kernel primitives (reference arithmetic verbatim)
# ===========================================================================
def flat_idx(zx, gv):
    """(3, n) int Miller → flat C-order FFT index."""
    return ((gv[0] % zx["nx"]) * zx["ny"] + gv[1] % zx["ny"]) * zx["nz"] \
        + gv[2] % zx["nz"]


def recon(zx, q):
    """ζ_q(μ, r) in the lab frame on the full FFT grid (gates only)."""
    ZGq = zx["ZG"][q]
    box = np.zeros((zx["n_mu"], zx["n_rtot"]), dtype=np.complex128)
    fi = flat_idx(zx, zx["gvec"][q])
    n = int(zx["ngk"][q])
    box[:, fi[:n]] = ZGq[:, :n]
    R = np.fft.ifftn(box.reshape(zx["n_mu"], zx["nx"], zx["ny"], zx["nz"]),
                     axes=(1, 2, 3), norm="backward"
                     ).reshape(zx["n_mu"], zx["n_rtot"])
    return R * np.exp(2j * np.pi * (zx["rfrac"] @ zx["qfr"][q]))[None, :]


def to_sphere(zx, zr, q):
    """rows(r) → rows(G) on sphere(q) (gates + refit)."""
    ph = np.exp(-2j * np.pi * (zx["rfrac"] @ zx["qfr"][q]))
    box = np.fft.fftn((zr * ph[None, :]).reshape(-1, zx["nx"], zx["ny"],
                                                 zx["nz"]),
                      axes=(1, 2, 3), norm="backward"
                      ).reshape(zr.shape[0], zx["n_rtot"])
    fi = flat_idx(zx, zx["gvec"][q])
    n = int(zx["ngk"][q])
    out = np.zeros((zr.shape[0], zx["ngkmax"]), dtype=np.complex128)
    out[:, :n] = box[:, fi[:n]]
    return out


def v_slab_on_set(zx, qfrac, GS, kind="slab", alpha=None):
    """Slab-truncated Coulomb kernel on an explicit Miller set at momentum
    ``qfrac`` (wrapped fractional), per G channel, K = q+G Cartesian (1/bohr):
        v(K) = 8π / K² · f2d / V_cell,
        f2d  = 1 − exp(−z_c |K_∥|) cos(K_z z_c),   z_c = π / b3_z
    Only the true divergence K² < 1e-12 is zeroed (the q=0 G=0 slot);
    at q ≠ 0 the finite G=0 term is part of the body (measured: zeroing it
    moves makeVq-vs-disk from ~1e-9 to 0.33).  Split (stable expm1;
    vSR+vLR == v to 1e-13, gated):
        slab_lr: v · e^{−K²/4α²}      slab_sr: v · (−expm1(−K²/4α²))

    NOT DEDUPED INTO ``vcoul.Slab2D``, deliberately.  Same FORMULA, different
    ARITHMETIC: the service spells it ``v_reg * fact`` and is bit-compared
    against BGW's pre-port table, this spells it ``8π/K² · f2d / celvol``, and
    the two differ in the last ulp — not a free swap under a 5e-6
    ``makeVq_vs_disk`` gate.  The ``slab_sr``/``slab_lr`` split has no service
    equivalent on an explicit Miller set at all.  The refit's BULK kernel does
    go through the door (:func:`make_v_on_set`).
    """
    K = zx["bvec"].T @ (np.asarray(qfrac)[:, None] + GS.astype(np.float64))
    K2 = np.sum(K * K, axis=0)
    zero = K2 < 1e-12
    K2s = np.where(zero, 1.0, K2)
    zc = np.pi / zx["bvec"][2, 2]
    f2d = 1.0 - np.exp(-zc * np.sqrt(K[0] ** 2 + K[1] ** 2)) \
        * np.cos(K[2] * zc)
    v = 8.0 * np.pi / K2s * f2d / zx["celvol"]
    if kind == "slab_lr":
        v = v * np.exp(-K2 / (4.0 * alpha ** 2))
    elif kind == "slab_sr":
        v = v * (-np.expm1(-K2 / (4.0 * alpha ** 2)))
    return np.where(zero, 0.0, v)


def make_v_on_set(zx, policy, log_fn=print):
    """The kernel the REFIT contracts with — the PRODUCER's, not this module's.

    ``v_slab_on_set`` above is the 2-D Ismail-Beigi truncation, and it is the
    only kernel the b26p long-range model can have, because that model's
    ``|G_z|`` channels only exist on a slab (see
    :func:`slab_scope_violations`).  The **refit** is a different animal: it
    fits ζ at the target Q from the htransform ψ and contracts it with
    ``v`` directly, with no fitted model anywhere in the path.  Nothing in it
    is 2-D except this one call — so on a bulk deck the refit is exact the
    moment the kernel is the one that built the stored tiles.

    That is not an argument, it is a measurement.  Rebuilding the
    ``si_bse_debug`` 4×4×4 960→191 bundle's own ``V_qmunu`` from its own
    stored ζ, at all 64 coarse q (2026-08-10,
    ``/pscratch/sd/j/jackm/xbdense_0810/probe_v3d2.py``):

    | kernel | makeVq-vs-disk relative (min / med / max) | under the 5e-6 gate |
    |---|---|---|
    | ``v_slab_on_set`` (this module's) | 5.8e-3 / 3.8e-2 / 5.1e-1 | 0 of 64 |
    | bulk 8π/K²/Ω, **no** mini-BZ head | 1.4e-14 / 9.2e-3 / 4.8e-2 | 1 of 64 |
    | the door below (bulk + mini-BZ head) | 3.7e-15 / 9.2e-15 / **3.3e-14** | **64 of 64** |

    The middle row is the one the module docstring already predicted and left
    open ("the remainder attributable to the deck's own
    ``mc_average_vcoul_body = true`` mini-BZ head-slot injection, which this
    module does not model either").  It is closed here by not modelling it at
    all: ``gw.compute_vcoul.compute_v_q_per_G`` IS the function the V_q
    writer called, and ``build_v_head_miniBZ_fn_3d`` IS the head it injected,
    so the refit and the producer cannot disagree about a convention — there
    is only one implementation of it.  Note which row is load-bearing: the
    mini-BZ head moves this by six orders of magnitude and the deck's 25 Ry
    ``bare_coulomb_cutoff`` moves it by nothing (the ζ sphere is already
    inside it), so a lane that ported the cutoff and skipped the head would
    have got 9.2e-3 and no idea why.

    ``policy`` is ``file_io.read_coulomb_policy_from_h5``'s dict.  ONLY
    ``sys_dim = 3`` takes the new route; 2-D and unstamped decks keep
    ``v_slab_on_set`` byte for byte, so no slab result anywhere can move.
    """
    sys_dim = str((policy or {}).get("sys_dim", "")).strip()
    if sys_dim != "3":
        return lambda qfrac, GS, kind="slab", alpha=None: v_slab_on_set(
            zx, qfrac, GS, kind, alpha)

    from gw.compute_vcoul import compute_v_q_per_G
    from vcoul import build_v_head_miniBZ_fn_3d

    cutoff = (policy or {}).get("bare_coulomb_cutoff", None)
    cutoff = float(cutoff) if cutoff not in (None, "") else None
    mc = str((policy or {}).get("mc_average_vcoul_body", "")).strip().lower()
    head_fn = (build_v_head_miniBZ_fn_3d(
        np.asarray(zx["kgrid"], dtype=int), zx["bvec"], zx["celvol"])
        if mc in ("true", "1", "yes") else None)
    log_fn(f"  [refit] bulk kernel via the producer's own door: sys_dim=3, "
           f"bare_coulomb_cutoff={cutoff}, mini-BZ head slot="
           f"{head_fn is not None} (mc_average_vcoul_body={mc!r}).  The "
           f"on-grid null below is what certifies this pairing.")

    def _v(qfrac, GS, kind="slab", alpha=None):
        if kind != "slab":
            raise NotImplementedError(
                "the bulk refit kernel has no SR/LR Gaussian split — that "
                "split exists for the b26p long-range model, which is "
                "slab-only and is not on the refit path")
        v = compute_v_q_per_G(
            np.asarray(qfrac, dtype=np.float64)[None, :],
            np.asarray(GS, dtype=np.int32)[None, :, :],
            bvec=zx["bvec"], cell_volume=zx["celvol"], sys_dim=3,
            vcoul_cutoff_ry=cutoff, v_head_fn=head_fn)
        return np.real(np.asarray(v)[0])

    return _v


def v_sphere(zx, q, kind="slab", alpha=None):
    """Kernel on the stored sphere at coarse q.  Returns (v, n_G)."""
    n = int(zx["ngk"][q])
    v = v_slab_on_set(zx, zx["qfr"][q], zx["gvec"][q][:, :n], kind, alpha)
    return v, n


def make_vq(zx, zt, q, kind="slab", alpha=None):
    """V[μν] = Σ_G conj(zt_μ(q+G)) v(q+G) zt_ν(q+G) on sphere(q)."""
    v, n = v_sphere(zx, q, kind, alpha)
    A = zt[:, :n] * np.sqrt(v)[None, :]
    return np.conj(A) @ A.T


def v_sphere_padded(zx, kind="slab", alpha=None):
    """Kernel rows on every stored sphere, zero-padded to ``ngkmax``:
    (nq, ngkmax) with v = 0 beyond ngk[q].  The zero pad makes batched
    A = ZG·√v identical to the per-q truncated ``make_vq`` factor (pad
    channels multiply any stored junk columns to exact zeros)."""
    v_all = np.zeros((zx["nq"], zx["ngkmax"]))
    for q in range(zx["nq"]):
        v, n = v_sphere(zx, q, kind, alpha)
        v_all[q, :n] = v[:n]
    return v_all


def _batched_vq_relF(ZG, v_all, V_ref, q_chunk=48):
    """relF(make_vq(q), V_ref[q]) for every q — the per-q tile rebuild
    V = conj(A)A^T, A = ZG·√v, batched on device in q chunks (same
    arithmetic as the host ``make_vq`` loop it replaces; measured 18.9 s →
    ~2 s at MoS2 12×12)."""

    @jax.jit
    def _chunk(ZG_b, v_b, Vd_b):
        A = ZG_b * jnp.sqrt(v_b)[:, None, :]
        V = jnp.einsum("bmg,bng->bmn", jnp.conj(A), A)
        d = V - Vd_b
        num = jnp.linalg.norm(d.reshape(d.shape[0], -1), axis=1)
        den = jnp.linalg.norm(Vd_b.reshape(Vd_b.shape[0], -1), axis=1)
        return num / den

    nq = ZG.shape[0]
    out = []
    for q0 in range(0, nq, q_chunk):
        sl = slice(q0, min(q0 + q_chunk, nq))
        out.append(np.asarray(jax.device_get(_chunk(
            jnp.asarray(ZG[sl]), jnp.asarray(v_all[sl]),
            jnp.asarray(V_ref[sl])))))
    return np.concatenate(out)


def build_cq(zx, mesh_xy: Mesh, q_chunk=48):
    """C_q Gram rebuild from ψ at centroids (reference ``build_cq``,
    order-robust R-space route, arithmetic verbatim — evaluated on device,
    q-CHUNK-accumulated: P_R = Σ_q e^{2πi q·R} Pk(q) is summed chunkwise
    with a donated accumulator, so peak device residency is the P_R
    intermediate plus ONE ψ/Pk chunk — never full-ψ (9.4 GB at the nb=80
    fit window) + full-Pk):
        C_q[μν] = Σ_{k,mn} conj(ρ_kmn(r_μ)) ρ_kmn(r_ν),
        ρ_kmn(r) = Σ_s conj(u_{m, wrap(k−q), s}(r)) u_{n, k, s}(r)
    with stored cell-periodic spinors at WRAPPED k labels (torus
    convention, no umklapp phases; gate: X^H X == C_q).

    P_R (nR, ns, n_μ, n_μ, ns) is sharded on the (μ, ν) FACE — the same
    ``P('x','y')`` layout every other tile in this module lives on — because
    replicated it is nR·ns²·n_μ²·16 B, i.e. 3.8 GB at the 6×6 / n_μ = 1496
    fixture but **53.6 GB at the converged 12×12 / n_μ = 2412 reference**,
    which no single device holds.  R and q stay unsharded, so BOTH
    contractions (the q-Fourier accumulation and the R-Fourier transform
    back) are device-local: no collectives, only the final host gather.
    The einsums replace the original reshape-matmuls verbatim — a reshape
    that flattens ns·n_μ·n_μ·ns would destroy the face sharding."""
    nq, nb, ns, n_mu = zx["nq"], zx["nb"], zx["ns"], zx["n_mu"]
    kg = zx["kgrid"]
    Rall = np.array([[rx, ry, rz] for rx in range(kg[0])
                     for ry in range(kg[1]) for rz in range(kg[2])])
    Rw = ((Rall + kg // 2) % kg) - (kg // 2)
    nR = len(Rw)
    EqR_np = np.exp(2j * np.pi * (zx["qfr"] @ Rw.T))
    rep = NamedSharding(mesh_xy, P())
    # (μ, ν) face.  n_mu is the raw k-means centroid count, so the face is
    # legal only when it divides the mesh axis; _ns degrades (loudly) when it
    # does not.  Pk carries the k-chunk on the leading axis, P_R the R index —
    # different leading extents, same face, so both are fitted separately.
    face5 = _ns(mesh_xy, P(None, None, "x", "y", None),
                (nR, ns, n_mu, n_mu, ns), "build_cq.P_R")
    face5_k = _ns(mesh_xy, P(None, None, "x", "y", None),
                  (q_chunk, ns, n_mu, n_mu, ns), "build_cq.Pk")
    face3_R = _ns(mesh_xy, P(None, "x", "y"), (nR, n_mu, n_mu), "build_cq.C_R")
    face3 = _ns(mesh_xy, P(None, "x", "y"), (nq, n_mu, n_mu), "build_cq.C_q")

    @partial(jax.jit, donate_argnums=(0,), out_shardings=face5)
    def _pr_acc(P_R, psi_c, EqR_c):
        psiX = jnp.conj(psi_c).transpose(0, 3, 1, 2)
        Pk = jax.lax.with_sharding_constraint(
            jnp.einsum("kmna,knbr->karmb", psiX, psi_c), face5_k)
        return P_R + jnp.einsum("kR,kavmb->Ravmb", EqR_c, Pk)

    @partial(jax.jit, out_shardings=face3)
    def _cq_final(P_R, EqR):
        C_R = jax.lax.with_sharding_constraint(
            jnp.einsum("ravmb,ravmb->rvm", jnp.conj(P_R), P_R), face3_R)
        return jnp.einsum("qr,rvm->qmv", jnp.conj(EqR) / nq, C_R)

    # Allocate the accumulator ALREADY SHARDED.  ``device_put(jnp.zeros(...))``
    # would materialise the whole 53.6 GB on one device first and only then
    # reshard — the allocation that OOMs an 80 GB card at n_μ = 2412.
    P_R = jax.jit(lambda: jnp.zeros((nR, ns, n_mu, n_mu, ns),
                                    dtype=jnp.complex128),
                  out_shardings=face5)()
    # Process-local placement of the host chunks (scorecard AA.1): every
    # rank reads the same file / computes the same phases, so plain
    # ``device_put``'s hidden assert_equal all-gather (P × chunk bytes,
    # once per q-chunk) verifies a tautology.  LORRAX_CHECK_REPLICA=1
    # re-arms the assertion.
    for q0 in range(0, nq, q_chunk):
        sl = slice(q0, min(q0 + q_chunk, nq))
        P_R = _pr_acc(P_R, device_put_process_local(zx["psi"][sl], rep),
                      device_put_process_local(EqR_np[sl], rep))
    # Return C_q as a MESH-SHARDED device array on the (μ, ν) FACE
    # (``face3`` = ``P(None, 'x', 'y')`` — q replicated, μ over 'x', ν over 'y'),
    # the SAME (μ, ν) tiling every other matrix in this module lives on.  The
    # former ``_to_host`` process_allgather materialised the full
    # (nq, n_μ, n_μ) c128 stack on EVERY process (13.4 GB/proc at nq=144,
    # n_μ=2412); the sharded form holds only nq·(n_μ/px)·(n_μ/py)·16 B per
    # device.  Consumers (``prepare_coarse``, ``run_gates``) reshard/gather on
    # DEVICE per chunk — no per-proc host gather.
    return _cq_final(P_R, device_put_process_local(EqR_np, rep))


def kq_index(zx, ki, qi):
    """Index of wrap(k − q) on the stored grid."""
    d = zx["k_int"][ki] - zx["k_int"][qi]
    return zx["k_lookup"][tuple(d % zx["kgrid"])]


def kq_index_of_frac(zx, qfrac):
    """Grid index of a fractional q that must lie ON the coarse grid."""
    kg = zx["kgrid"]
    ki = np.rint(np.asarray(qfrac) * kg).astype(int) % kg
    assert np.max(np.abs(np.asarray(qfrac) * kg - np.rint(np.asarray(qfrac) * kg))) < 1e-8, \
        f"{qfrac} is not on the coarse grid"
    return zx["k_lookup"][tuple(ki)]


def gap_window_pairs(zx, q, nvw=3, ncw=3):
    """Spin-traced BSE exchange rows M_cvk(μ) = Σ_s conj(u_{c,k−q,s})
    u_{v,k,s} at the centroids; top-nvw valence × bottom-ncw conduction ×
    all k → (npair, n_μ).  These contract the tile into the physical
    gap-window block B = M V M^H — the campaign verdict variable."""
    nv = zx["nv"]
    cs = list(range(nv, nv + ncw))
    vs = list(range(nv - nvw, nv))
    rows = np.empty((zx["nk"], ncw, nvw, zx["n_mu"]), dtype=np.complex128)
    for k in range(zx["nk"]):
        kq = kq_index(zx, k, q)
        rows[k] = np.einsum("csm,vsm->cvm",
                            np.conj(zx["psi"][kq][cs]), zx["psi"][k][vs])
    return rows.reshape(-1, zx["n_mu"])


def b_block(x, V):
    """B[p,p'] = Σ_{μν} x[p,μ] V[μν] conj(x[p',ν]).

    K^x = M V M†: the bra (output) index carries the bare vertex and the ket
    (the index a trial vector contracts) carries the conjugate, as the
    transition density <0|ρ̂|Ψ> = Σ A_cvk ψ_ck ψ*_vk requires.
    """
    return x @ V @ np.conj(x).T


# ===========================================================================
# gate battery (reference run_gates; every value printed; any FAIL stops)
# ===========================================================================
def run_gates(zx, C_q):
    ok = True

    def log(k, v, tol=None):
        nonlocal ok
        flag = "" if tol is None else ("  OK" if v <= tol else "  ** FAIL **")
        if tol is not None and v > tol:
            ok = False
        print(f"    [gate] {k:<44s} {v:.3e}{flag}")

    print("  [gates] vq_interp coarse data:")
    n0 = int(zx["ngk"][0])
    zt = to_sphere(zx, recon(zx, 0), 0)
    log("recon_roundtrip_sphere_Gamma",
        relF(zt[:, :n0], zx["ZG"][0][:, :n0]), 1e-13)
    k2max = max(np.max(np.sum((zx["bvec"].T @ (zx["qfr"][q][:, None]
                               + zx["gvec"][q][:, :int(zx["ngk"][q])]
                               .astype(np.float64))) ** 2, axis=0))
                for q in range(zx["nq"]))
    log("sphere_max|q+G|^2_minus_cutoff", max(0.0, k2max - zx["zeta_cutoff"]),
        1e-9)
    vd = _batched_vq_relF(zx["ZG"], v_sphere_padded(zx), zx["Vqmunu"])
    log("makeVq_vs_disk_Vqmunu_allq_max", float(np.max(vd)), 5e-6)
    # X^H X == C_q (torus convention) at q=0 — device Gram, k-CHUNK
    # accumulated (the host loop + serial-BLAS zgemm cost 13.7 s at 12×12;
    # the un-chunked X tensor would be nk·nb²·n_μ·16 B ≈ 14.7 GB at the
    # nb=80 fit window — never materialized)
    q = 0
    kqs = np.array([kq_index(zx, k, q) for k in range(zx["nk"])])

    @partial(jax.jit, donate_argnums=(0,))
    def _xhx_acc(G, psi_kq_c, psi_c):
        X = jnp.einsum("knsm,kMsm->knMm", jnp.conj(psi_kq_c), psi_c)
        Xf = X.reshape(-1, X.shape[-1])
        return G + jnp.conj(Xf).T @ Xf

    G = jnp.zeros((zx["n_mu"], zx["n_mu"]), dtype=jnp.complex128)
    for k0 in range(0, zx["nk"], 16):
        sl = slice(k0, min(k0 + 16, zx["nk"]))
        G = _xhx_acc(G, jnp.asarray(zx["psi"][kqs[sl]]),
                     jnp.asarray(zx["psi"][sl]))
    # C_q is now a device array sharded on the (μ, ν) face; gather ONLY the
    # q=0 slice for this diagnostic (n_μ²·16 B, replicated) — the full stack is
    # never brought to host.
    C_q0 = _to_host(C_q[0]) if isinstance(C_q, jax.Array) else C_q[0]
    log("XHX_vs_Cq_torus_q0",
        relF(np.asarray(jax.device_get(G)), C_q0), 1e-8)
    for q in range(2):
        v, n = v_sphere(zx, q)
        vs, _ = v_sphere(zx, q, kind="slab_sr", alpha=0.63)
        vl, _ = v_sphere(zx, q, kind="slab_lr", alpha=0.63)
        log(f"vSR+vLR==v_q{q}",
            float(np.max(np.abs(vs[:n] + vl[:n] - v[:n]))
                  / max(np.max(np.abs(v[:n])), 1e-300)), 1e-13)
    # Slab-axis separability (per-G_z channels need b3 ∥ z, b1/b2 in-plane).
    # BELT AND BRACES, not the guard: ``build_vq_evaluator`` has already
    # refused a non-slab cell by name (:func:`assert_slab_scope`) before this
    # battery runs, so on any deck that reaches here the line reads 0.000e+00.
    # It stays because a battery that prints every quantity it depends on is
    # how the row-23 attribution was possible at all.  Do NOT read a FAIL here
    # as an interpolation defect: it is cell geometry, and so is the
    # ``makeVq_vs_disk`` residual that accompanies it on a bulk deck (both
    # measured in :func:`slab_scope_violations`).
    bv = zx["bvec"]
    log("slab_axes_offdiag", float(max(np.max(np.abs(bv[2, :2])),
                                       np.max(np.abs(bv[:2, 2])))
                                   / np.abs(bv[2, 2])), 1e-12)
    assert ok, "vq_interp gate battery FAILED — stop (KNOWN_SANDBOX_ERRORS rule)"


# ===========================================================================
# STAGE 1 — offline preparation at the coarse grid points
# ===========================================================================
def lr_fit_degrees(zx, *, e_cut=None, shell_floor=FIT_SHELL_FLOOR):
    """THE CRITERION: which ``|G_z|`` channels the long-range model fits, and
    at what in-plane polynomial degree.  Returns ``{|G_z|: degree}``.  This is
    the ONE site that decides the fitted channel set; everything else
    (:func:`lr_gset`'s trim, :func:`lr_design_blocks`, the mini-BZ head-slot
    guard) reads the dict this returns, so they cannot drift apart.

        fit channel n  iff  (n·|b₃|)² ≤ E_eff,
        E_eff = max(e_cut, (shell_floor·|b₃|)²·(1 + FIT_FLOOR_MARGIN))

    **An energy cutoff, with a hard two-shell floor.**  That shape is the
    owner's ruling of 2026-08-10 — ``docs/architecture/decisions.md``, "The
    long-range channel criterion is an energy cutoff with a two-shell floor",
    which is the entry to cite; this docstring does not mint it — and both
    halves of it are load-bearing.

    *Why an energy cutoff.*  What this replaces was a hardcoded ``|G_z| ≤ 3``
    read off ``DEG_B26P``'s keys — a fixed shell COUNT spanning a
    cell-dependent energy window.  ``|b₃| = 2π/c`` shrinks as a slab's vacuum
    padding grows, so three shells cover 0.69 Ry on the MoS2 reference cell but
    only 0.077 Ry once the vacuum is tripled, while the LR superset's own
    (isotropic, 6.63 Ry) cutoff keeps widening.  Stage 1 subtracts the
    FULL-sphere ``V_LR`` and stage 3 adds back only the FITTED channels, so
    every unfitted channel is weight that is subtracted and never returned:
    measured **17.93 % of the long-range weight lost at 3× vacuum**, silently,
    because ``run_nulls`` tests the sampled ``Fch`` rather than the fitted
    model.  Tying the shell count to an ENERGY makes it follow ``1/|b₃|``
    instead of standing still, which is what closes that gap.

    *Why the floor, in the owner's own terms.*  The requirement is that the fit
    "reliably capture the first at least 2 G shells, because we need to capture
    G=0, which rolls by an umklapp vector at BZ boundaries."  The head slot is
    ``G* = argmin_G |Q+G|``, and at a zone boundary that argmin is NOT ``G=0``:
    it rolls onto a neighbouring reciprocal-lattice vector.  Past the boundary
    along z (``q_z = 0.6``) the argmin lands on ``G = [0,0,-1]``, i.e. the
    ``|G_z| = 1`` channel.  A criterion that captured only the literal first
    shell would leave that channel unfitted, its form factor identically zero,
    and the head magnitude would be multiplied by zero and dropped in silence.
    The floor is what makes the criterion incapable of that, whatever ``e_cut``
    or whatever cell it is handed.

    Two shells rather than one is also what the bulk control demands, and the
    two constraints agree: on Si (``|b₃| = 1.06``, where a 1 Ry cutoff alone
    would keep only ``|G_z| = 0``) the LR weight lost is **48.4 % at a
    zero-shell floor, 1.37 % at one shell, and 0.000 % at two** — two shells is
    the smallest floor that leaves a bulk cell bit-identical to today, and a
    third shell buys nothing there because the superset itself stops at
    ``|G_z| = 2``.  ``FIT_FLOOR_MARGIN`` only lifts the comparison off exact
    floating-point equality at ``|n| = shell_floor``.

    Worth knowing which half is carrying the guarantee where: on a THICK SLAB
    ``|b₃|`` is small, so the cutoff alone already reaches well past the rolled
    channel and the floor never binds.  It is on BULK and thin cells — large
    ``|b₃|``, few shells inside any reasonable energy — that the cutoff would
    strand the roll and the floor is the whole of the protection.  The floor
    costs nothing where it does not bind, which is why it is unconditional
    rather than a regime test.

    *The default, and why it is 1.0 Ry and not the 0.5 Ry sketch.*  The default
    has to sit in ``[0.691, 1.229) Ry`` to reproduce today's channel set
    exactly on the MoS2 reference deck — those bounds are ``(3|b₃|)²`` and
    ``(4|b₃|)²`` there — and it has to hold the 3×-vacuum loss under 1 %.
    **0.5 Ry fails both**: it drops the ``|G_z| = 3`` channel on the reference
    deck (161 → 115 columns, and the reference deck's own loss RISES 0.24 % →
    1.90 %), and it still leaves 2.04 % lost at 3× vacuum.  ``E_CUT_FIT = 1.0``
    is a round value mid-window, bit-identical on both reference decks, and
    measured (7×7 in-plane q sample, the metric of FIX_vq_interp.md §B.3):

        cell              fitted   nG    LR weight lost
        Si bulk           ≤2       123   0.000 %  (floor-dominated; today 0.00 %)
        MoS2 reference    ≤3       161   0.240 %  (today 0.24 % — identical set)
        MoS2 vacuum ×1.5  ≤5       253   0.189 %  (today 2.61 %)
        MoS2 vacuum ×2    ≤7       345   0.164 %  (today 7.17 %)
        MoS2 vacuum ×3    ≤10      483   0.290 %  (today 17.93 %)

    The extra channels enter at degree 0 — one complex coefficient each — so
    stage 2's solve is unchanged in practice; the cost is nG in the stage-3
    GEMM and in the host ``Fch`` block, and it is paid only in the regime that
    needs it (nothing on Si, nothing on the reference slab).
    """
    e_cut = E_CUT_FIT if e_cut is None else float(e_cut)
    if not e_cut > 0.0:
        raise ValueError(f"vq_interp fit cutoff must be positive, got {e_cut}")
    b3 = float(np.linalg.norm(np.asarray(zx["bvec"], dtype=np.float64)[2]))
    e_eff = max(e_cut, (int(shell_floor) * b3) ** 2 * (1.0 + FIT_FLOOR_MARGIN))
    nmax = int(np.floor(np.sqrt(e_eff) / b3))
    return {n: DEG_B26P.get(n, 0) for n in range(nmax + 1)}


def lr_gset(zx, alpha=ALPHA, degrees=None):
    """Fixed global Miller superset of the LR channel: all G with
    min_{q∈BZ, q_z=0} |q+G|² ≤ 4α² ln(1/ε_LR), minimised over a 13×13
    in-plane q sample (reference ``lr_gset``, verbatim), then RESTRICTED to
    the G_z channels the b26p model actually fits.

    THE G_z TRIM.  The cutoff above is ISOTROPIC in K, but the model is not.
    :func:`lr_design_blocks` fits only the channels named in ``degrees``
    (``DEG_B26P`` — |G_z| ≤ 3) and :func:`make_eval_vq` leaves every other
    column of ``M`` at its ``jnp.zeros`` initialisation, so those columns
    carry ``zt = 0``, ``A = 0``, and contribute EXACTLY ``0.0`` to
    ``conj(A) A^T``.  Measured: ``max|A[:, dead]| = 0.000e+00`` over 8 target
    Q on the MoS2 3×3 slab deck — an exact zero, not a small number.

    Carrying them is not free, and how much it costs depends on the CELL.
    ``|G_z|`` runs to ``Kmax/|b3|``, and ``|b3| = 2π/c`` shrinks as the slab's
    vacuum padding grows, so the dead fraction rises with cell height while
    the modelled window ``3·|b3|`` shrinks:

      * Si bulk (a₃ = 7.26 bohr, |b3| = 1.061):  nG = 123, |G_z| ≤ 2,
        **0 dead** — the trim is a no-op, the cutoff already stops inside
        the fitted channels.
      * MoS2 slab (a₃ = 22.68 bohr, |b3| = 0.277): nG = 337, |G_z| ≤ 9,
        **176 dead (52.2%)**.

    52% of the ``(nq, n_μ, nG)`` host ``Fch`` block, 52% of the ``(n_μ, nG)``
    ``A`` matrix and of both of its resharding constraints, and 52% of the
    K-dimension of the LR outer product — all of it multiplying exact zeros.

    THE TRIM IS OFF BY DEFAULT, AND THE NUMBER THAT REFUSED IT IS
    ``5.793e-03``.  The dead columns are dead in ``eval_vq`` — that part is
    proven — but they are NOT dead in :func:`run_nulls`, whose
    ``F_own_rebuild_vs_cleaned_LR_tile_max`` null rebuilds the LR tile from
    the SAMPLED form factors ``Fch`` on the superset and compares it against
    ``Sc·V_LR·Sc`` built from the FULL stored sphere.  That null certifies
    SET COVERAGE, not model fidelity, so dropping columns from the set
    breaks it: 5.793e-03 against a 1e-6 tolerance on the MoS2 3×3 deck.

    Read the two facts together and they say something the tree did not
    record: stage 1 subtracts the FULL-sphere ``V_LR`` and stage 3 adds back
    only the FITTED channels, so the |G_z| > 3 weight is already lost on the
    production path — and ``run_nulls`` cannot see that, because it tests
    ``Fch`` rather than the fitted model.  Trimming does not create the gap;
    it moves the gap into the one instrument that was reporting on it.
    Closing it is a MODEL question (widen ``DEG_B26P``), not a set question,
    and it is owner-ruled work — see FIX_vq_interp.md.

    ``LORRAX_VQ_LR_GZ_TRIM=1`` opts in: bit-identical in ``eval_vq`` to ~2
    ulp, worth 1.69x on the LR outer product at n_μ = 2412 and half the host
    ``Fch`` block, at the price of that null.  The trim is keyed on
    ``degrees`` rather than on the literal ``DEG_B26P`` so the superset and
    the fit cannot drift apart silently; :func:`lr_design_blocks` REFUSES a
    ``degrees`` wider than the one the superset was built with.
    """
    K2max = 4.0 * alpha ** 2 * np.log(1.0 / EPS_LR)
    Kmax = np.sqrt(K2max)
    nmax = [int(np.ceil(Kmax / np.linalg.norm(zx["bvec"][i]))) + 1
            for i in range(3)]
    gr = [np.arange(-n, n + 1) for n in nmax]
    GX, GY, GZ = np.meshgrid(*gr, indexing="ij")
    Gall = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 0)
    ts = np.linspace(-0.5, 0.5, 13, endpoint=False)
    m = np.full(Gall.shape[1], np.inf)
    for tx in ts:
        for ty in ts:
            qf = np.array([tx, ty, 0.0])
            K = zx["bvec"].T @ (qf[:, None] + Gall.astype(np.float64))
            m = np.minimum(m, np.sum(K * K, axis=0))
    keep = m <= K2max
    if os.environ.get("LORRAX_VQ_LR_GZ_TRIM", "0") == "1":
        deg = lr_fit_degrees(zx) if degrees is None else degrees
        keep = keep & np.isin(np.abs(Gall[2]),
                              np.asarray(sorted({abs(int(g)) for g in deg})))
    return np.ascontiguousarray(Gall[:, keep])


def _sphere_slot(zx, q, GS):
    """Miller columns of GS → stored sphere slots at q; −1 where outside
    the sphere (channel is zero in the stored representation; Gaussian
    weight bounded by exp(−cutoff/4α²))."""
    n = int(zx["ngk"][q])
    lut = {tuple(g): i for i, g in enumerate(zx["gvec"][q][:, :n].T)}
    return np.array([lut.get(tuple(g), -1) for g in GS.T])


def _to_host(x):
    """Gather a device array to a full host numpy array on EVERY process,
    whether it is PROCESS-SPANNING or fully addressable.

    ONE line of delegation to :func:`common.collectives.gather_to_host`, which
    owns the three-arm branch (``device_get`` when fully addressable, a local
    shard read when merely replicated, ``process_allgather(tiled=True)`` when
    the shards are genuinely remote) and the incidents behind it.  The wrapper
    is not free-standing: ``tests/test_bse_gather_and_mesh.py`` ratchets it,
    and its four siblings, to REACH that service rather than re-derive it.

    WHAT MAKES IT LOAD-BEARING HERE: the collective arm is reached in a LOOP.
    ``prepare_coarse``'s ζ-clean tiles are q-sharded ``P(('x','y'),...)``
    across the whole mesh, so on a 16-process run every chunk takes it, and
    the ranks stay in lockstep only because that loop is deterministic (same
    nq / q_chunk on every process).
    """
    from common.collectives import gather_to_host
    return gather_to_host(x)


def prepare_coarse(zx, C_q, mesh_xy: Mesh, *, alpha=ALPHA, eps_tik=EPS_TIK,
                   eigh_backend: str = "auto", q_chunk: int = 48,
                   keep_host_mirrors: bool = True, degrees=None,
                   fit_ecut=None):
    """STAGE 1.  Returns the coarse-side bundle ``prep``:
      S        (nq, n_μ, n_μ)  Tikhonov cleaning operators S_q (host copy,
                               nulls only) — ``None`` without host mirrors
      V_SRc    (nq, n_μ, n_μ)  cleaned short-range tiles — DEVICE stack,
                               sharded ``P(None,'x','y')`` (the stencil data)
      V_SRc_np (nq, n_μ, n_μ)  host copy (LOO ladders / metrics) — ``None``
                               without host mirrors
      GS       (3, nG)         fixed LR Miller superset 𝒢(α), G_z-trimmed
                               to the fitted channels (:func:`lr_gset`)
      Fch      (nq, n_μ, nG)   phase-factored cleaned LR form factors (host —
                               fit input only, droppable after stage 2)
      W        (nq, nG)        LSQ weights v_LR(q+G) (head slot → 0)
      gz_cols  {gz: cols}      per-G_z column index into GS

    The per-coarse-q linear algebra is q-BATCHED on device in chunks of
    ``q_chunk``, sharded ``P(('x','y'))`` on the q axis (each device owns its
    q-shard end to end — eigh, S_q = R g_ε(Λ) R^H, V_c = conj(S)(V_ref −
    V_LR)conj(S), ζ̃-cleaning, and the phase-factored superset gather all
    run without collectives), with ONE host round-trip per chunk.  The
    per-q sphere is zero-padded to ``ngkmax`` (pad channels carry v = 0, so
    they contribute exact zeros — same arithmetic as the per-q truncation)
    which keeps the whole stage at ≤2 XLA compiles instead of one per
    distinct sphere size (15 at MoS2 12×12; the ``_clean_split`` census
    finding).  ``eigh_backend='auto'`` resolves to the batched native path —
    see ``distrib_la.Plan`` for when the FFI backends win;
    they are honored per-q if explicitly requested.  n_μ is used at its
    LOGICAL extent here; the jitted evaluator pads on output.

    ``q_chunk`` is a CEILING, lowered to keep the host-side ζ read bounded:
    the chunk is read whole on every process before it is device_put, and at
    the converged reference (n_μ = 2412, ngkmax = 8603) one q is 332 MB, so
    48 q is 15.9 GB per rank — 64 GB per 4-rank Perlmutter node, on top of
    C_q and the two (nq, n_μ, n_μ) host stacks.  It stays a whole number of
    mesh shards so the q axis still device_puts evenly.

    ``keep_host_mirrors=False`` drops ``S`` and ``V_SRc_np``.  They are read
    ONLY by ``run_nulls`` and ``eval_vq_host`` (the LOO ladders), so with the
    diagnostics off they are 2 × nq·n_μ²·16 B of dead weight — 26.8 GB per
    process at the converged reference, which is 107 GB of a 251 GB node once
    four ranks share it, and it is what OOM-kills the node.  ``V_SRc`` is then
    assembled straight from the per-chunk device shards, never via a host
    round-trip.
    """
    nq, n_mu = zx["nq"], zx["n_mu"]
    ngkmax = zx["ngkmax"]
    ndev = int(mesh_xy.devices.size)
    # ONE resolved plan for the whole q loop: platform / compiled-capability
    # / coverage / geometry guards all fire here, before any q is touched
    # (see distrib_la.resolve_backend), and the plan then carries the backend,
    # operand-sharding contract and the batch behaviour to the call site
    # below.  auto|off resolve to the q-batched native path.  Hoisted, not
    # per-chunk: resolution dlopens, and the first FFI call builds a BLACS /
    # cuSOLVERMp context (scorecard L §5).
    eigh_plan = linalg_plan("eigh", mesh_xy, backend=eigh_backend)
    q_chunk = int(min(q_chunk, nq,
                      max(ndev, ndev * (5e9 // (n_mu * ngkmax * 16 * ndev)))))
    assert np.max(np.abs(zx["qfr"][:, 2])) < 1e-12, "slab pipeline needs q_z=0"
    # ``degrees`` reaches the superset builder as well as the fit, so the two
    # cannot disagree about which G_z channels exist (see lr_gset's G_z trim).
    degrees = lr_fit_degrees(zx, e_cut=fit_ecut) if degrees is None else degrees
    GS = lr_gset(zx, alpha, degrees=degrees)
    nG = GS.shape[1]
    # q-batched layout: every device owns whole matrices for its own q.  That
    # needs nq_chunk % ndev == 0, which a deck with fewer BZ q-points than
    # devices can never satisfy (4x4x1 => nq=16, at P=64 there is nothing to
    # split).  _ns degrades to replicated, loudly, in exactly that regime; when
    # the chunk does divide, the spec is returned unchanged and no HLO moves.
    # The RAGGED TAIL is fitted separately: nq % q_chunk can leave a short last
    # chunk that is illegal even when the full ones are legal.
    def _qb(nb_lead):
        return (_ns(mesh_xy, P(("x", "y"), None), (nb_lead, n_mu),
                    "prepare_coarse.qb2"),
                _ns(mesh_xy, P(("x", "y"), None, None), (nb_lead, n_mu, ngkmax),
                    "prepare_coarse.qb3"))

    qb2, qb3 = _qb(q_chunk)

    # ── host geometry/kernel prep (cheap per-q numpy vector math) ────────
    # v rows zero-padded to ngkmax (``v_sphere_padded``): pad channels
    # multiply the (possibly junk) stored ZG columns beyond ngk[q] to exact
    # zeros.  Superset slots outside the sphere map to the appended zero
    # column (slot ngkmax).
    v_ref_all = v_sphere_padded(zx)
    v_lr_all = v_sphere_padded(zx, kind="slab_lr", alpha=alpha)
    idx_all = np.empty((nq, nG), dtype=np.int64)
    W = np.empty((nq, nG))
    n_out = 0
    for q in range(nq):
        idx = _sphere_slot(zx, q, GS)
        n_out += int(np.sum(idx < 0))
        idx_all[q] = np.where(idx < 0, ngkmax, idx)
        W[q] = v_slab_on_set(zx, zx["qfr"][q], GS, kind="slab_lr", alpha=alpha)

    rmu = jnp.asarray(zx["rmu_frac"])                    # (n_μ, 3) trace const
    GS_f = jnp.asarray(GS.T.astype(np.float64))          # (nG, 3) trace const

    # The two batch kernels are built PER DISTINCT CHUNK LENGTH (at most two:
    # the full q_chunk and the ragged tail), because their shardings depend on
    # whether that length divides the device count.  Cached, so the compile
    # count is unchanged when nq % q_chunk == 0 — the common case.
    _KERNELS: dict = {}

    def _kernels(nb_lead):
        if nb_lead in _KERNELS:
            return _KERNELS[nb_lead]
        qb2_b, qb3_b = _qb(nb_lead)
        _KERNELS[nb_lead] = (_make_eigh_batch(qb2_b, qb3_b),
                             _make_clean_split(qb3_b))
        return _KERNELS[nb_lead]

    def _make_eigh_batch(qb2, qb3):
        @partial(jax.jit, out_shardings=(qb2, qb3))
        def _eigh_batch(C):
            C = jax.lax.with_sharding_constraint(C, qb3)
            lam, R = jnp.linalg.eigh(C)
            return lam, R
        return _eigh_batch

    def _clean_split_body(lam, R, ZGq, v_ref, v_lr, idx, qfr_b):
        # S_q = R g_ε(Λ) R^H  (analytic Tikhonov filter of C_q; §12.3)
        g = lam ** 2 / (lam ** 2
                        + (eps_tik * lam.max(axis=1, keepdims=True)) ** 2)
        S = jnp.einsum("bmr,br,bnr->bmn", R, g, jnp.conj(R))
        Sc = jnp.conj(S)
        # V_ref − V_LR from the sphere factors A = ζ̃√v  (V = conj(A)A^T)
        A_ref = ZGq * jnp.sqrt(v_ref)[:, None, :]
        A_lr = ZGq * jnp.sqrt(v_lr)[:, None, :]
        V_delta = jnp.einsum("bmg,bng->bmn", jnp.conj(A_ref), A_ref) \
            - jnp.einsum("bmg,bng->bmn", jnp.conj(A_lr), A_lr)
        V_SRc = Sc @ V_delta @ Sc
        zt = S @ ZGq                     # cleaned ζ̃ on the sphere (rows μ)
        # (c) phase-factored cleaned form factors on the superset
        zt_ext = jnp.concatenate(
            [zt, jnp.zeros((zt.shape[0], n_mu, 1), zt.dtype)], axis=2)
        ztg = jnp.take_along_axis(zt_ext, idx[:, None, :], axis=2)
        qG = qfr_b[:, None, :] + GS_f[None, :, :]        # (b, nG, 3)
        ph = jnp.exp(2j * jnp.pi * jnp.einsum("mi,bgi->bmg", rmu, qG))
        return S, V_SRc, ph * ztg

    def _make_clean_split(qb3):
        return jax.jit(_clean_split_body, out_shardings=(qb3, qb3, qb3))

    S_np = (np.empty((nq, n_mu, n_mu), dtype=np.complex128)
            if keep_host_mirrors else None)
    V_SRc_np = (np.empty((nq, n_mu, n_mu), dtype=np.complex128)
                if keep_host_mirrors else None)
    V_dev_chunks = []
    Fch = np.empty((nq, n_mu, nG), dtype=np.complex128)

    def C_herm(sl):
        """Hermitised C_q, PER CHUNK.  ``C_q`` is a device array sharded on the
        (μ, ν) face (``build_cq`` no longer host-gathers); slicing the q axis
        (replicated) and hermitising stay ON DEVICE — no per-proc host copy of
        the 13.4 GB stack.  ``_eigh_batch``'s ``with_sharding_constraint`` then
        reshards each chunk from the (μ, ν) face to the q-batched ``qb3`` layout.
        (numpy ``C_q`` still works — ``jnp`` ops promote it — preserving the old
        host path for callers/tests that pass a replicated array.)"""
        c = C_q[sl]
        return 0.5 * (c + jnp.conj(jnp.swapaxes(c, -2, -1)))

    for q0 in range(0, nq, q_chunk):
        sl = slice(q0, min(q0 + q_chunk, nq))
        nb_lead = sl.stop - sl.start
        (_eigh_batch, _clean_split) = _kernels(nb_lead)
        qb2, qb3 = _qb(nb_lead)
        if eigh_plan.is_native:
            lam, R = _eigh_batch(C_herm(sl))
        else:
            # Explicit distributed-FFI request.  ``plan.batched`` is the
            # per-q loop + stack this used to spell out, plus the operand
            # reshard to the FFI's P(None,'x','y') contract — one call, and
            # the only thing that changes if a backend ever grows a real
            # batched eigh is which branch inside the plan runs.  The same
            # batched post-eigh pipeline follows either way (single source
            # for the math).
            lam, R = eigh_plan.batched(C_herm(sl))
        # Process-local placement of the host chunks (scorecard AA.1) —
        # same file / same host tables on every rank, so plain
        # ``device_put``'s hidden assert_equal all-gather (5 × per chunk)
        # verifies a tautology.  LORRAX_CHECK_REPLICA=1 re-arms it.
        #
        # ζ is the exception: it is NOT a small host table, it is the
        # 47.8-GB q-stack, and ``device_put_process_local`` still needs
        # the WHOLE q-chunk in host numpy on every rank before it slices
        # out that rank's shard (15.9 GB per rank per chunk at the
        # reference extents).  ``read_q_slab`` asks the ζ reader for the
        # same chunk on the same sharding and, when the deployment can
        # serve SlabIO, each rank hyperslabs only its own shard off disk.
        # Same elements, same layout; it is a transport change.
        S_b, V_b, F_b = _clean_split(
            lam, R,
            zx["ZG"].read_q_slab(sl.start, nb_lead, sharding=qb3),
            device_put_process_local(v_ref_all[sl], qb2),
            device_put_process_local(v_lr_all[sl], qb2),
            device_put_process_local(idx_all[sl], qb2),
            device_put_process_local(zx["qfr"][sl], qb2))
        # ``_to_host``, not device_get: S_b/V_b/F_b are q-sharded qb3 across
        # the whole mesh, so at P>1 their shards span other processes.  This
        # is the deterministic loop ``_to_host``'s docstring relies on.
        if keep_host_mirrors:
            S_np[sl] = _to_host(S_b)
            V_SRc_np[sl] = _to_host(V_b)
        else:
            V_dev_chunks.append(V_b)
        Fch[sl] = _to_host(F_b)
    tail = float(np.exp(-zx["zeta_cutoff"] / (4.0 * alpha ** 2)))
    print(f"  [prep] gset({alpha}) = {nG} G; {n_out} out-of-sphere (q,G) "
          f"channels zero-filled; sphere-tail bound {tail:.1e}"
          f"{'' if keep_host_mirrors else '; host mirrors dropped'}")
    stencil_sh = _ns(mesh_xy, P(None, "x", "y"), (nq, n_mu, n_mu),
                     "prepare_coarse.V_SRc")
    # Host-mirror branch: process-local placement (AA.1; the mirror was
    # assembled identically on every rank from gathered chunks).
    V_SRc_dev = (device_put_process_local(V_SRc_np, stencil_sh)
                 if keep_host_mirrors
                 else jax.jit(lambda *c: jnp.concatenate(c, axis=0),
                              out_shardings=stencil_sh)(*V_dev_chunks))
    del V_dev_chunks
    return {"alpha": alpha, "eps_tik": eps_tik, "GS": GS, "S": S_np,
            "V_SRc": V_SRc_dev, "V_SRc_np": V_SRc_np, "Fch": Fch, "W": W,
            # the degrees the SUPERSET was built with — lr_design_blocks
            # refuses a wider request against it (silent-narrowing guard)
            "lr_gz_degrees": (degrees if os.environ.get(
                "LORRAX_VQ_LR_GZ_TRIM", "0") == "1" else None),
            # THE fitted channel set (:func:`lr_fit_degrees`), always present.
            # Every consumer of "which |G_z| does the model fit?" reads this —
            # the fit itself and the mini-BZ head-slot guard — so the criterion
            # is applied once, here, and never re-derived downstream.
            "fit_degrees": dict(degrees),
            "gz_cols": {int(g): np.where(GS[2] == g)[0]
                        for g in np.unique(GS[2])}}


# ===========================================================================
# STAGE 2 — the global b26p LR fit (host/replicated; see module docstring)
# ===========================================================================
def _poly_spec(d):
    """[(a, b)] with a+b ≤ d, graded order."""
    return [(a, t - a) for t in range(d + 1) for a in range(t + 1)]


def _eval_basis_np(Kpar, spec, alpha):
    """Design matrix (n_samples, nb): (K_x/2α)^p (K_y/2α)^r, real."""
    s = 1.0 / (2.0 * alpha)
    x, y = Kpar[0] * s, Kpar[1] * s
    return np.stack([(x ** a) * (y ** b) if (a or b) else np.ones_like(x)
                     for a, b in spec], 1)


def lr_design_blocks(zx, prep, degrees=None):
    """Per-(G_z, q) weighted normal blocks of the LR fit:
        AtA[gz][q] = Φ^T diag(w) Φ     (nb, nb)   real basis
        AtY[gz][q] = Φ^T diag(w) Y     (nb, n_μ)
    with Φ the in-plane design at the (q, G) samples of channel gz,
    w = v_LR(q+G), Y = Fch samples.  Channels with |gz| absent from
    ``degrees`` are model-zero (dropped).  Per-q blocks make LOO refits
    honest (target's samples excluded) at O(nb²) cost."""
    if degrees is None:
        degrees = (prep.get("lr_gz_degrees") or prep.get("fit_degrees")
                   or DEG_B26P)
    # SILENT-NARROWING GUARD.  When ``prepare_coarse`` trimmed the superset to
    # a degrees dict, asking here for a WIDER one would not raise anywhere: the
    # loop below iterates ``prep['gz_cols']``, so a channel the superset no
    # longer carries is simply never visited and the model quietly loses it.
    # That is the "config key parsed and quietly ignored" failure mode
    # src/bse/STATUS.md:167-170 records as having cost this project days.
    _trim = prep.get("lr_gz_degrees")
    if _trim is not None:
        extra = sorted({abs(int(g)) for g in degrees}
                       - {abs(int(g)) for g in _trim})
        if extra:
            raise ValueError(
                f"lr_design_blocks asked for |G_z| channels {extra}, but "
                f"prepare_coarse built the LR superset trimmed to "
                f"{sorted({abs(int(g)) for g in _trim})} (lr_gset's G_z trim). "
                f"Those channels have no columns in prep['GS'] and would be "
                f"silently dropped from the fit.  Rebuild prepare_coarse with "
                f"the same `degrees`, or set LORRAX_VQ_LR_GZ_TRIM=0.")
    nq = zx["nq"]
    des = {"specs": {}, "AtA": {}, "AtY": {}, "alpha": prep["alpha"]}
    for g, cols in prep["gz_cols"].items():
        if abs(g) not in degrees:
            continue
        spec = _poly_spec(degrees[abs(g)])
        nb = len(spec)
        assert nb <= 0.6 * len(cols) * nq, f"gz={g}: basis under-determined"
        AtA = np.empty((nq, nb, nb))
        AtY = np.empty((nq, nb, zx["n_mu"]), dtype=np.complex128)
        for q in range(nq):
            qG = zx["qfr"][q][:, None] + prep["GS"][:, cols].astype(np.float64)
            Kpar = (zx["bvec"].T @ qG)[:2]
            Phi = _eval_basis_np(Kpar, spec, prep["alpha"])
            w = prep["W"][q][cols]
            Pw = Phi * w[:, None]
            AtA[q] = Phi.T @ Pw
            AtY[q] = Pw.T @ prep["Fch"][q][:, cols].T
        des["specs"][g] = spec
        des["AtA"][g] = AtA
        des["AtY"][g] = AtY
    ncoef = sum(len(s) for s in des["specs"].values())
    print(f"  [fit] LR design: degrees {degrees} -> {ncoef} complex "
          f"coefficients per mu (global)")
    return des


def fit_lr_model(des, exclude=None):
    """STAGE 2 solve: one ridge-stabilised normal solve per G_z channel
    over all coarse q except ``exclude``.  Returns {gz: C (nb, n_μ)} —
    n_μ × 26 complex TOTAL for b26p."""
    nq = next(iter(des["AtA"].values())).shape[0]
    sel = [q for q in range(nq) if q != exclude]
    out = {}
    for g in des["specs"]:
        A = des["AtA"][g][sel].sum(0)
        Y = des["AtY"][g][sel].sum(0)
        A = A + RIDGE * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0])
        out[g] = np.linalg.solve(A, Y)
    return out


# ===========================================================================
# STAGE 3 — the ONE jitted evaluator at arbitrary target Q
# ===========================================================================
def stencil_r7(zx):
    """The campaign's in-plane truncated-R stencil: 7 shortest lattice
    vectors [i, j, 0] in the adot metric."""
    Rall = np.array([[i, j, 0] for i in range(-2, 4) for j in range(-2, 4)])
    d = np.sqrt(np.einsum("ri,ij,rj->r", Rall, zx["adot"], Rall))
    return Rall[np.argsort(d)][:7]


def stencil_pinv(q_train, Rset):
    """Q-INDEPENDENT part of the trigonometric stencil weights:
    pinv(F) with F_ji = e^{−2πi q_j·R_i}.  ``w(Q) = f0(Q) @ pinv(F)`` is
    then evaluated inside the jitted ``eval_vq`` — exact (delta) when Q is
    a training point and Rset resolves the grid."""
    F = np.exp(-2j * np.pi * (np.asarray(q_train) @ np.asarray(Rset).T))
    return np.linalg.pinv(F)


def pack_coeffs(des, coeffs):
    """Fitted b26p coefficients as a jit-friendly tuple, in the fixed
    channel order of ``des['specs']`` (dict order is stable per build)."""
    return tuple(jnp.asarray(coeffs[g]) for g in des["specs"])


_MBZ_DQ_CACHE: dict = {}
_MBZ_DQ_CACHE_MAX = 2


@jax.jit
def _mbz_draw_u(gidx, base_key):
    """``(local, 3)`` uniforms, one per GLOBAL slot index.

    The sharded sobol/threefry idiom: per-sample keys folded off one root
    key, so the draw for a given global slot is the same no matter which
    rank makes it.  MODULE-LEVEL and taking ``base_key`` as a runtime
    argument — the closure this replaced was rebuilt inside
    :func:`minibz_head_vlr` on every call, and JAX keys its trace cache on
    callable identity, so every target Q paid a fresh trace.
    """
    def one(i):
        return jax.random.uniform(jax.random.fold_in(base_key, i),
                                  (3,), dtype=jnp.float64)
    return jax.vmap(one)(gidx)


def _mbz_dq(bvec, kgrid, *, n_q, nsamples, qmc_reps, seed_offset, lo, chunk):
    """This rank's mini-BZ offsets ``δq`` for slab ``[lo, lo+chunk)`` — MEMOISED.

    ``chunk`` IS THE SAME ON EVERY RANK, and that is a cache-contract
    requirement.  It used to be ``hi - lo`` with ``lo = rank*n_kept//nranks``,
    i.e. a rank slab, and that extent is the SHAPE of two jits:
    :func:`_mbz_draw_u` traces at ``(hi-lo,) uint32`` and
    ``vcoul.wrap_points_to_voronoi`` at ``(hi-lo, 3) f64``.  ``n_kept =
    qmc_reps * n_q`` with ``n_q`` an adaptive per-Q clamp, so it does not
    divide the rank count in general, the slabs differed by one band across
    ranks, and each distinct extent was a separate compiled program with a
    separate persistent-cache key — held by some ranks and not others, which
    is the collective-compile deadlock precondition
    (FIX_multislice_cachekey.md §6.1, sibling 1).

    Now the extent is ``chunk = ceil(n_kept/nranks)`` everywhere and the
    caller masks the slots that ran past ``n_kept``.  Those surplus slots are
    ordinary further QMC draws — finite, in range, and simply not counted —
    so nothing here has to defend against them.

    ``lo`` IS DELIBERATELY NOT IN THE KEY.  It is ``rank * chunk``, and both
    of those are constant within a process, so within the only scope this
    dict has it is a function of ``chunk``.  Keeping it in the key would put
    the rank slab back into a cache key for no lookup that could ever
    benefit — a host memo is per-process and cross-rank sharing was never on
    offer — and ``tests/cache_key_lint.py``'s ``rank-cache-key`` rule would
    be right to say so.

    NOTHING here depends on the target Q.  The draws are indexed by global
    slot only; the Voronoi wrap and the ``randlims`` affine map are pure cell
    geometry.  Only the final ``_minibz_kernel_bare(shift_cart, dq, …)`` in
    :func:`minibz_head_vlr` sees Q at all.  Yet the whole construction used to
    be rebuilt per target Q, and it is expensive in exactly the way a Q loop
    cannot amortise: a ``(local, 3)`` threefry draw on device, a 63 MB
    device→host pull, a 2.6M-point host matmul, a device round trip through
    ``wrap_points_to_voronoi`` whose ``(N, 27, 3)`` candidate tensor is 1.7 GB
    at the production sample count, and a second 63 MB pull — measured
    **1.40 s per Q** at ``nsamples=2**18, qmc_reps=10`` on the MoS2 3×3 slab
    deck, independent of ``n_μ``.

    The key carries every input the arrays depend on, INCLUDING ``n_q`` and
    the (rank-invariant) chunk width, so a hit returns the array today's code
    would have built element for element and in the same order.  The
    reduction that consumes it is therefore **bit-identical**, not merely
    close — which is why the cache can be keyed rather than the samples
    re-derived by prefix slicing.

    Bounded to ``_MBZ_DQ_CACHE_MAX`` entries: ``dq`` is ``(chunk, 3)`` f64,
    63 MB per entry at the production sample count divided by the rank count.
    ``n_q`` is BGW's adaptive per-batch count and clamps to ``nsamples`` for
    every Q whose ``|Q+G*|`` is small enough (all 8 probed Q on the reference
    slab deck), so a Q path normally lives in a single entry.
    """
    from vcoul import (minibz_cell_affine, minibz_frac_to_cart,
                       wrap_points_to_voronoi)
    key = (bvec.tobytes(), tuple(int(s) for s in kgrid), int(n_q),
           int(nsamples), int(qmc_reps), int(seed_offset), int(chunk))
    hit = _MBZ_DQ_CACHE.get(key)
    if hit is not None:
        return hit
    base_key = jax.random.PRNGKey(int(seed_offset))
    # ONE extent on every rank: the offset rides in as a VALUE, which is an
    # operand, while the count is the shape.  Same split as the canonical
    # ``jit__multi_slice`` fix (sizes static, offsets dynamic).
    slots = int(lo) + jnp.arange(int(chunk), dtype=jnp.uint32)
    rep = slots // np.uint32(n_q)                       # replicate batch
    loc = slots % np.uint32(n_q)                        # in-batch draw
    gidx = rep * np.uint32(int(nsamples)) + loc         # global draw index
    U = np.asarray(_mbz_draw_u(gidx, base_key), dtype=np.float64)
    # δq mapping THROUGH THE DOOR, not a local re-spelling of it: all three
    # steps are ``vcoul.minibz``'s, which is what makes the convention
    # decidable in one place.  It is a convention with a history — the
    # transposed frac→cart spelling ``U @ bvec.T`` is NOT a fundamental domain
    # of the reciprocal lattice and shipped as a three-month bias in the 3D
    # body head (358bb0b), which is precisely why the service refuses to let
    # each caller write its own.  nmax=3 = BGW ncell.  Bit-identical to the
    # local spellings this replaces (measured over 200 random cells, both
    # helpers exactly equal in float64).
    randcart = minibz_frac_to_cart(U, bvec)
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), jnp.asarray(bvec), nmax=3), dtype=np.float64)
    dq = (minibz_cell_affine(bvec, kgrid) @ wrapped.T).T
    dq[:, 2] = 0.0                                       # 2D slab: qz = 0
    if len(_MBZ_DQ_CACHE) >= _MBZ_DQ_CACHE_MAX:
        _MBZ_DQ_CACHE.pop(next(iter(_MBZ_DQ_CACHE)))
    _MBZ_DQ_CACHE[key] = dq
    return dq


def minibz_head_vlr(zx, prep, Qfrac, *, alpha=None, nsamples=2**18,
                    qmc_reps=10, n_coarse=250_000, seed_offset=0, kgrid=None,
                    moment=False):
    """RANK-PARALLEL mini-BZ CELL AVERAGE of the LR-slab head at target Q.

    Returns ``(gstar, head_val)``, or ``(gstar, head_val, M_ab)`` when
    ``moment=True``:

      ``gstar``    = ``argmin_G |Q+G|²`` over the stored Miller superset
                     ``prep['GS']`` — the umklapp bringing Q nearest Γ.
      ``head_val`` = ``<v_LR(Q+G*)>_mBZ / celvol`` (real), in the SAME units
                     as ``eval_vq``'s stored ``v[gstar]`` point value, so the
                     evaluator injects it by a straight slot replacement.
      ``M_ab``     = ``<v_LR(q) q_a q_b>_mBZ / celvol`` (real symmetric 3x3,
                     ``moment=True`` only), on the SAME kept samples and with
                     ``q = Q+G*+δq`` the full Cartesian momentum.

    THE SCALAR IS THE WRONG MOMENT, AND ``M_ab`` IS THE RIGHT ONE.  The head
    the exchange kernel actually wants averaged is ``v(q)·|q·d|²``, not
    ``v(q)`` — the pair amplitude vanishes linearly at Γ and its square is
    what tames the pole.  Averaging ``v`` alone and evaluating the amplitude
    at the cell CENTRE keeps one direction out of the cell's whole
    distribution and weights the radius at the sample instead of by ``v``;
    the two errors are independent, one angular and one radial, so neither
    is fixable alone (``LT_HEAD_PROBLEM.md`` §2.2).  Because the dipole is a
    property of the transition and not of the integration variable, the true
    average factorises exactly onto ``conj(d_a) M_ab d_b``.  ``M_ab``
    therefore cannot be injected through ``eval_vq`` — that seam is a scalar
    under a square root — and is consumed instead as a rank-three term over
    the transition index (``bse_stack_matvec``'s ``head_tensor``).  The
    scalar return is kept because the pre-tensor arm is the object under
    repair and its baseline must not move.

    2D slab: pure in-plane adaptive MC (the head is a ``|Q|`` cusp, not a
    ``1/q²`` pole — no analytic sphere, BGW ``minibzaverage_2d``).  Only the
    LR channel is averaged; the SR body already carries ``v_SR(Q+G*)`` once
    (arbitrary_q_bse.md §16 no-double-count).  The winding ``e^{-i2θ}`` is
    untouched — it rides the phase-factored ζ̃ model in ``eval_vq``; this
    scalar supplies the magnitude only.

    ONE source of truth for the PHYSICS: the bare LR-slab kernel is
    :func:`vcoul._minibz_kernel_bare` (the same
    ``8π·f2d·e^{−K²/4α²}/|K|²`` GW's Coulomb head uses), the mini-BZ affine
    wrap is :func:`vcoul.wrap_points_to_voronoi` + the same
    ``randlims`` map as :func:`vcoul.minibz_voronoi_batches` (all through
    the service door since the 2026-08-07 replumb), and
    the inscribed-sphere / adaptive-``n_q`` rule matches
    :func:`vcoul.minibz_average` (``minibzaverage.f90:63-75``).

    RANK-PARALLELISM (this routine used to run the FULL serial host Sobol QMC
    redundantly on every process): the estimator is the mean of the bare
    kernel over ``qmc_reps`` replicate batches, each keeping its first
    ``n_q`` mini-BZ δq draws — equivalently a single mean over the
    ``reps·n_q`` kept samples, since ``n_q`` is the same for every rep.  Each
    kept sample carries a GLOBAL slot index whose randomness comes ONLY from
    that index, via ``jax.random.fold_in(PRNGKey(seed), gidx)`` (the sharded
    sobol/threefry idiom: per-sample keys folded off one root key, no
    per-rank reseeding).  Ranks own disjoint contiguous slabs of the
    kept-slot range and evaluate only their slab; the per-rank partial sums
    are all-reduced across processes (``process_allgather`` + sum).  Because
    the sample for a given global slot is identical no matter which rank
    draws it, the result is DETERMINISTIC and rank-count invariant (bit-equal
    up to float summation reorder ~1e-13), just ``1/nranks`` the work.

    ``kgrid`` overrides ``zx['kgrid']`` for the mini-BZ CELL SIZE — the
    ``bse_k_grid`` coarse→fine init passes the FINE k-grid here so the q=0
    exchange head is the (smaller) fine mini-BZ average, not the coarse one
    (the head magnitude scales with the mini-BZ cell area).  That init reaches
    this function only on the ``head_minibz_average`` OPT-IN arm; with the key
    off (the default) it carries the coarse q=0 tile through untouched and
    never calls here (bse_io._interpolate_bse_data_to_grid).  Default None uses
    the stored coarse grid (the exciton_bands Q-path convention, unchanged).
    """
    # The DOOR is the true dependency; the ``gw.coulomb`` / ``gw.vcoul``
    # spellings of these are compat shims.  Everything Q-INDEPENDENT (the
    # draws, the Voronoi wrap, the mini-BZ affine) lives in :func:`_mbz_dq`.
    from vcoul import _minibz_kernel_bare, minibz_inscribed_sphere_r2
    if alpha is None:
        alpha = float(prep["alpha"])
    GS = np.asarray(prep["GS"], dtype=np.float64)          # (3, nG)
    bvec = np.asarray(zx["bvec"], dtype=np.float64)
    celvol = float(zx["celvol"])
    kgrid = (tuple(int(s) for s in zx["kgrid"]) if kgrid is None
             else tuple(int(s) for s in kgrid))
    Qf = np.asarray(Qfrac, dtype=np.float64)
    K = bvec.T @ (Qf[:, None] + GS)                        # (3, nG) cartesian Q+G
    K2 = np.sum(K * K, axis=0)
    gstar = int(np.argmin(K2))
    # The head magnitude is injected onto the LR column ``gstar`` by
    # ``eval_vq`` (``v = where(arange(nG)==gstar, head_val, v)``), and that
    # column's form factor is ``M[:, gstar]`` — which the b26p model leaves at
    # ZERO for any |G_z| it does not fit.  Landing the head on such a column
    # would multiply the cell-averaged magnitude by an identically-zero form
    # factor and drop it silently.  For a q_z=0 slab the argmin is always a
    # G_z=0 slot, so this cannot fire in scope; it fires if the caller leaves
    # that scope.  (Nothing checked this before.)
    # THE UMKLAPP CONSTRAINT (owner ruling 2026-08-10, docs/architecture/
    # decisions.md).  ``gstar`` is the head slot, and at a BZ boundary it is
    # NOT G=0 — it rolls onto a neighbouring reciprocal-lattice vector, which
    # is exactly why :func:`lr_fit_degrees` carries a hard two-shell floor.
    # The check now reads the criterion's own resolved set rather than
    # ``DEG_B26P``'s keys, so widening the criterion widens the guard with it.
    _fitdeg = prep.get("fit_degrees") or DEG_B26P
    if int(abs(GS[2, gstar])) not in _fitdeg:
        raise ValueError(
            f"mini-BZ head slot G*={GS[:, gstar].astype(int).tolist()} is in "
            f"the |G_z|={int(abs(GS[2, gstar]))} channel, which the b26p model "
            f"does not fit (fitted channels={sorted(_fitdeg)}).  Its form "
            f"factor is model-zero, so the head magnitude would be silently "
            f"dropped.  "
            f"Target Q={Qf.tolist()} is outside the slab pipeline's q_z=0 "
            f"scope.")
    shift_cart = K[:, gstar]                               # cartesian Q+G*
    len_shift2 = float(shift_cart @ shift_cart)
    q0sph2 = minibz_inscribed_sphere_r2(bvec, kgrid, is_2d=True)

    # BGW adaptive per-batch sample count (minibzaverage.f90:63-75); the 2D
    # slab head always takes this (non-analytic-sphere) branch.
    if len_shift2 > 1e-12:
        n_q = int(round(n_coarse * 4.0 * q0sph2 / len_shift2))
        n_q = max(1, min(n_q, int(nsamples)))
    else:
        n_q = int(nsamples)

    rank = int(jax.process_index())
    nranks = int(jax.process_count())
    n_kept = int(qmc_reps) * n_q
    # ONE CHUNK WIDTH FOR EVERY RANK — the cache contract.  The old bounds
    # were ``rank*n_kept//nranks`` … ``(rank+1)*n_kept//nranks``: a balanced
    # partition, but a RAGGED one, and the width is the traced shape of
    # ``_mbz_draw_u`` and ``wrap_points_to_voronoi``.  A uniform ceil-width
    # chunk covers exactly the same global slots [0, n_kept) exactly once
    # once the tail is masked, so no sample is gained or lost; what changes
    # is that every rank now compiles ONE program instead of one per slab
    # width, and the ranks past ``n_kept`` compile it too instead of
    # compiling nothing at all (FIX_multislice_cachekey.md §6.1, sibling 1).
    chunk = max(1, -(-n_kept // max(nranks, 1)))
    lo = rank * chunk
    # The mask, not a shorter array: slots at or past ``n_kept`` are drawn
    # and then not counted.  ``keep`` is all-False on a surplus rank, which
    # reproduces the old ``else:`` branch's zeros exactly — and reaches it
    # through the same compiles its peers did.
    keep = (lo + np.arange(chunk)) < n_kept

    # Q-INDEPENDENT and memoised: the draws, the Voronoi wrap and the
    # mini-BZ affine map are pure cell geometry indexed by global slot.
    # Only the kernel evaluation below sees Q.
    dq = _mbz_dq(bvec, kgrid, n_q=n_q, nsamples=nsamples,
                 qmc_reps=qmc_reps, seed_offset=seed_offset,
                 lo=lo, chunk=chunk)
    v, _ = _minibz_kernel_bare(shift_cart, dq, kind="slab_lr",
                               alpha=alpha, zc=float(np.pi / bvec[2, 2]))
    # Host-side masking: ``_minibz_kernel_bare`` is pure numpy, so dropping
    # the padded rows here costs no module and no shape.
    v = v[keep]
    dq = dq[keep]
    local_sum = float(np.sum(v))
    local_cnt = float(v.shape[0])
    if moment:
        K = shift_cart[None, :] + dq                         # (N, 3) momentum
        local_mom = (K * v[:, None]).T @ K                   # (3, 3)
    else:
        local_mom = np.zeros((3, 3))

    if nranks > 1:
        from jax.experimental import multihost_utils
        payload = np.concatenate(
            [np.asarray([local_sum, local_cnt], dtype=np.float64),
             local_mom.reshape(-1)]) if moment else \
            np.asarray([local_sum, local_cnt], dtype=np.float64)
        g = np.asarray(multihost_utils.process_allgather(payload, tiled=False))
        tot_sum = float(g[:, 0].sum())
        tot_cnt = float(g[:, 1].sum())
        tot_mom = g[:, 2:].sum(axis=0).reshape(3, 3) if moment \
            else np.zeros((3, 3))
    else:
        tot_sum, tot_cnt = local_sum, local_cnt
        tot_mom = local_mom
    head_bare = tot_sum / tot_cnt                            # mean over kept
    if moment:
        return gstar, head_bare / celvol, (tot_mom / tot_cnt) / celvol
    return gstar, head_bare / celvol


def make_eval_vq(zx, prep, des, mesh_xy: Mesh, n_rmu_pad: int | None = None,
                 head_minibz_average: bool = False):
    """Build the ONE jitted arbitrary-Q evaluator.

        eval_vq(Qfrac, V_SRc_stack, pinvF, coeffs_tuple) -> V(Q)          (default)
        eval_vq(Qfrac, V_SRc_stack, pinvF, coeffs_tuple, head_val, gstar) (mini-BZ)

    Q-DEPENDENT data enter as RUNTIME ARGUMENTS — target ``Qfrac``, the
    SR-tile stack (whose training subset a LOO caller may swap), the
    stencil pseudo-inverse ``pinvF`` (train-set-dependent), and the fitted
    coefficient tuple (model/LOO-dependent) — so one compile serves every
    Q on a path (per-q-recompile lesson).  Q-INDEPENDENT geometry (the
    Miller superset, per-G_z column indices, polynomial specs, bvec,
    centroid coordinates, the R stencil, slab-kernel constants) is baked
    in as trace constants: it never changes between calls.

    Per-element (module docstring, stage 3):
        w_j(Q)   = e^{−2πi Q·R} pinv(F)                       (nR7)
        V_SR(Q)  = Σ_j w_j V_SRc(q_j)                         (tile AXPYs)
        M_μ(K)   = Φ(K_∥) C[gz]           per exact G_z channel, K = Q+G
        A        = e^{−2πi(Q+G)·s_μ} M_μ √v_LR(Q+G)
        V(Q)     = V_SR + conj(A) A^T

    ``head_minibz_average`` (config ``head_minibz_average``): when True the
    evaluator takes two extra runtime args ``head_val`` (real
    ``<v_LR(Q+G*)>_mBZ / celvol`` from :func:`minibz_head_vlr`) and ``gstar``
    (the ``argmin_G|Q+G|`` slot), and REPLACES the LR channel's ``v[gstar]``
    POINT value with the mini-BZ CELL AVERAGE before building A.  Only the
    magnitude scalar changes: the phase-factored ζ̃ model ``zt`` (which
    carries the 2D winding ``e^{-i2θ}``) and the SR body ``V_SR`` (which
    carries ``v_SR(Q+G*)`` once) are UNTOUCHED — arbitrary_q_bse.md §16
    no-double-count.  Default False keeps the 4-arg point-value evaluator
    bit-identical (the shared body's injection is a ``jnp.where`` gated on a
    sentinel ``gstar=-1``, an exact no-op).

    Output: (n_out, n_out) tile, ``P('x','y')``, n_out = ``n_rmu_pad`` (the
    loader's padded extent; pad rows/cols zero) or the logical n_μ when
    ``n_rmu_pad`` is None.  NO solve, NO eigh, NO r_tot object.
    """
    n_mu = zx["n_mu"]
    n_out = int(n_rmu_pad) if n_rmu_pad is not None else n_mu
    assert n_out >= n_mu
    GS = jnp.asarray(prep["GS"].astype(np.float64))          # (3, nG)
    nG = int(GS.shape[1])
    bvec = jnp.asarray(zx["bvec"])
    rmu = jnp.asarray(zx["rmu_frac"])                        # (n_μ, 3)
    R7 = jnp.asarray(stencil_r7(zx).astype(np.float64))      # (7, 3)
    alpha = float(prep["alpha"])
    zc = float(np.pi / zx["bvec"][2, 2])
    celvol = float(zx["celvol"])
    specs = [(g, tuple(des["specs"][g]),
              jnp.asarray(prep["gz_cols"][g], dtype=jnp.int32))
             for g in des["specs"]]
    # Two different (μ, ν) extents live in this body: the LOGICAL n_mu on the
    # intermediates and the loader's PADDED n_out on the returned tile.  n_out
    # is padded_mu_extent(n_mu, px*py) so it always divides; n_mu is the raw
    # k-means count and usually does not.  Fit them separately — using one
    # sharding for both is what made the intermediates illegal.
    grid_mu = _ns(mesh_xy, P("x", "y"), (n_mu, n_mu), "eval_vq.V_SR(n_mu)")
    grid_out = _ns(mesh_xy, P("x", "y"), (n_out, n_out), "eval_vq.out(n_out)")
    row_x = _ns(mesh_xy, P("x", None), (n_mu, nG), "eval_vq.A_x")
    row_y = _ns(mesh_xy, P("y", None), (n_mu, nG), "eval_vq.A_y")

    def _phi(Kpar, spec):
        s = 1.0 / (2.0 * alpha)
        x, y = Kpar[0] * s, Kpar[1] * s
        return jnp.stack([(x ** a) * (y ** b) if (a or b) else jnp.ones_like(x)
                          for a, b in spec], 1)

    def _body(Qfrac, V_SRc_stack, pinvF, coeffs_tuple, head_val, gstar):
        # ── SR stencil:  w(Q) = e^{−2πi Q·R} pinv(F);  V_SR = Σ_j w_j V_SRc ──
        f0 = jnp.exp(-2j * jnp.pi * (Qfrac @ R7.T))          # (nR,)
        w = f0 @ pinvF                                        # (n_train,)
        V_SR = jnp.tensordot(w, V_SRc_stack, axes=(0, 0))     # (n_μ, n_μ) xy
        V_SR = jax.lax.with_sharding_constraint(V_SR, grid_mu)
        # ── LR model rebuild at K = Q+G (closed form, never interpolated) ──
        K = bvec.T @ (Qfrac[:, None] + GS)                    # (3, nG)
        M = jnp.zeros((n_mu, GS.shape[1]), dtype=jnp.complex128)
        for i, (g, spec, cols) in enumerate(specs):
            Phi = _phi(K[:2][:, cols], spec)                  # (ncol, nb)
            M = M.at[:, cols].set((Phi @ coeffs_tuple[i]).T)
        K2 = jnp.sum(K * K, axis=0)
        zero = K2 < 1e-12                                     # q=0 G=0 slot only
        K2s = jnp.where(zero, 1.0, K2)
        f2d = 1.0 - jnp.exp(-zc * jnp.sqrt(K[0] ** 2 + K[1] ** 2)) \
            * jnp.cos(K[2] * zc)
        v = 8.0 * jnp.pi / K2s * f2d / celvol \
            * jnp.exp(-K2 / (4.0 * alpha ** 2))
        v = jnp.where(zero, 0.0, v)
        # mini-BZ head: replace the LR POINT value at G*=gstar with the
        # cell-averaged <v_LR(Q+G*)>_mBZ.  Sentinel gstar=-1 (off path) →
        # arange==-1 all-False → v unchanged, exact no-op (bit-identical).
        v = jnp.where(jnp.arange(nG) == gstar, head_val, v)
        qG = Qfrac[None, :] + GS.T                            # (nG, 3)
        zt = jnp.exp(-2j * jnp.pi * (rmu @ qG.T)) * M         # (n_μ, nG)
        A = zt * jnp.sqrt(v)[None, :]
        A_x = jax.lax.with_sharding_constraint(A, row_x)
        A_y = jax.lax.with_sharding_constraint(A, row_y)
        V = V_SR + jnp.conj(A_x) @ A_y.T
        V = jax.lax.with_sharding_constraint(V, grid_mu)
        if n_out > n_mu:
            V = jnp.pad(V, ((0, n_out - n_mu), (0, n_out - n_mu)))
        return jax.lax.with_sharding_constraint(V, grid_out)

    if head_minibz_average:
        @partial(jax.jit, out_shardings=grid_out)
        def eval_vq(Qfrac, V_SRc_stack, pinvF, coeffs_tuple, head_val, gstar):
            return _body(Qfrac, V_SRc_stack, pinvF, coeffs_tuple,
                         head_val, gstar)
    else:
        @partial(jax.jit, out_shardings=grid_out)
        def eval_vq(Qfrac, V_SRc_stack, pinvF, coeffs_tuple):
            return _body(Qfrac, V_SRc_stack, pinvF, coeffs_tuple,
                         jnp.asarray(0.0, dtype=jnp.float64),
                         jnp.asarray(-1, dtype=jnp.int32))

    return eval_vq


def build_vq_evaluator(restart_file, mesh_xy: Mesh, n_rmu_pad: int | None = None,
                       *, zeta_file=None, alpha=ALPHA, eps_tik=EPS_TIK,
                       eigh_backend="auto", head_minibz_average=False,
                       run_diagnostics=True, log_fn=print, fit_ecut=None):
    """ONE arbitrary-Q exchange-tile model build (stages 1-3), packaged.

    This is the SINGLE orchestration of the ``vq_interp`` pipeline
    (``load_zeta_coarse`` → ``build_cq`` → gates → ``prepare_coarse`` →
    ``lr_design_blocks`` → ``fit_lr_model`` → nulls → ``make_eval_vq`` +
    stencil pieces).  BOTH the ``exciton_bands`` Q-path driver and the general
    BSE init's ``bse_k_grid`` coarse→fine densification (on its
    ``head_minibz_average`` opt-in arm only) call this — there is no second
    copy of the setup sequence.

    Returns a ``SimpleNamespace`` with every handle the per-Q evaluation needs:
        .zx, .prep, .des, .coeffs        the fitted coarse-side model
        .eval_vq                         the ONE jitted evaluator (Q runtime arg)
        .pinvF                           stencil pseudo-inverse (Q-independent)
        .coeffs_packed                   jit-friendly coefficient tuple
        .head_minibz_average             whether eval_vq takes (head_val, gstar)

    ``zeta_file`` defaults to ``zeta_q.h5`` beside ``restart_file``.  ``eval_vq``
    at a target Q is then a single dispatch (see ``make_eval_vq`` /
    ``minibz_head_vlr``).
    """
    from types import SimpleNamespace
    # The reference gate battery (run_gates / run_nulls) materializes replicated
    # tensors that OOM for large centroid counts (e.g. 1496 recovered-D3h on 16
    # GPU → a 58 GB alloc).  Allow opting out via env; the coarse-fit still runs
    # and the driver's physical on-grid htransform gate still validates.
    if os.environ.get("LORRAX_SKIP_VQ_GATES", "0") == "1":
        run_diagnostics = False
    if zeta_file is None:
        zeta_file = os.path.join(os.path.dirname(restart_file), "zeta_q.h5")
    # ``mesh_xy`` reaches the ζ reader so ``prepare_coarse``'s q-chunk
    # read is a per-rank SlabIO hyperslab rather than a whole-chunk host
    # read on every rank.  The reader announces and falls back to the
    # local h5py plan when the deployment cannot serve SlabIO.
    zx = load_zeta_coarse(restart_file, zeta_file, mesh=mesh_xy,
                          log_fn=log_fn, require_slab=False)
    # THE ONE SITE, moved here from the loader: this function is the b26p
    # model build, and the model is what is slab-only.  Still before anything
    # expensive — ``build_cq`` is the next line.
    assert_slab_scope(zx["bvec"], qfr=zx["qfr"], policy=zx.get("policy"),
                      source=restart_file)
    C_q = build_cq(zx, mesh_xy)
    if run_diagnostics:
        run_gates(zx, C_q)
    # The host mirrors exist for run_nulls / eval_vq_host only — same switch.
    prep = prepare_coarse(zx, C_q, mesh_xy, alpha=alpha, eps_tik=eps_tik,
                          eigh_backend=eigh_backend,
                          keep_host_mirrors=run_diagnostics,
                          fit_ecut=fit_ecut)
    des = lr_design_blocks(zx, prep)
    coeffs = fit_lr_model(des)
    if run_diagnostics:
        run_nulls(zx, prep, des, coeffs)
    eval_vq = make_eval_vq(zx, prep, des, mesh_xy, n_rmu_pad,
                           head_minibz_average=head_minibz_average)
    pinvF = jnp.asarray(stencil_pinv(zx["qfr"], stencil_r7(zx)))
    coeffs_packed = pack_coeffs(des, coeffs)
    return SimpleNamespace(
        zx=zx, prep=prep, des=des, coeffs=coeffs, eval_vq=eval_vq,
        pinvF=pinvF, coeffs_packed=coeffs_packed,
        head_minibz_average=head_minibz_average)


def eval_vq_host(zx, prep, des, coeffs, qfrac, train=None):
    """Host-side reference evaluation (LOO ladders, nulls, tests) — the
    same arithmetic as the jitted evaluator, numpy throughout."""
    if train is None:
        train = list(range(zx["nq"]))
    R7 = stencil_r7(zx)
    pinvF = stencil_pinv(zx["qfr"][train], R7)
    f0 = np.exp(-2j * np.pi * (np.asarray(qfrac) @ R7.T))
    w = f0 @ pinvF
    V_SR = np.tensordot(w, prep["V_SRc_np"][train], axes=(0, 0))
    GS = prep["GS"]
    qf = np.asarray(qfrac, dtype=np.float64)
    M = np.zeros((zx["n_mu"], GS.shape[1]), dtype=np.complex128)
    Kall = zx["bvec"].T @ (qf[:, None] + GS.astype(np.float64))
    for g, spec in des["specs"].items():
        cols = prep["gz_cols"][g]
        Phi = _eval_basis_np(Kall[:2][:, cols], spec, prep["alpha"])
        M[:, cols] = (Phi @ coeffs[g]).T
    v = v_slab_on_set(zx, qf, GS, kind="slab_lr", alpha=prep["alpha"])
    qG = qf[None, :] + GS.T.astype(np.float64)
    zt = np.exp(-2j * np.pi * (zx["rmu_frac"] @ qG.T)) * M
    A = zt * np.sqrt(v)[None, :]
    return V_SR + np.conj(A) @ A.T


# ===========================================================================
# nulls (must hold at machine level before any accuracy number is read)
# ===========================================================================
def run_nulls(zx, prep, des, coeffs):
    ok = True

    def log(k, v, tol):
        nonlocal ok
        flag = "  OK" if v <= tol else "  ** FAIL **"
        if v > tol:
            ok = False
        print(f"    [null] {k:<44s} {v:.3e}{flag}")

    # exact-stencil reproduction: with the FULL R lattice the trig weights
    # are a delta, so "interpolating" to a training point returns its data
    kg = zx["kgrid"]
    Rfull = np.array([[i - kg[0] // 2, j - kg[1] // 2, 0]
                      for i in range(kg[0]) for j in range(kg[1])])
    q0 = 1
    F = np.exp(-2j * np.pi * (zx["qfr"] @ Rfull.T))
    f0 = np.exp(-2j * np.pi * (zx["qfr"][q0] @ Rfull.T))
    w = f0 @ np.linalg.pinv(F)
    log("exact_stencil_VSRc_train_point",
        relF(np.tensordot(w, prep["V_SRc_np"], axes=(0, 0)),
             prep["V_SRc_np"][q0]), 1e-9)
    log("exact_stencil_Fch_train_point",
        relF(np.tensordot(w, prep["Fch"], axes=(0, 0)), prep["Fch"][q0]),
        1e-9)
    # own F rebuild == cleaned LR tile (channel machinery consistency;
    # bounded by the out-of-sphere zero-fill)
    rr = []
    # own F rebuild vs Sc·V_LR·Sc, batched on device in q chunks (same
    # per-q arithmetic as the host loop it replaces; 6.9 s → ~1 s at 12×12)
    v_lr_all = v_sphere_padded(zx, kind="slab_lr", alpha=prep["alpha"])
    GS_f = jnp.asarray(prep["GS"].T.astype(np.float64))
    rmu = jnp.asarray(zx["rmu_frac"])

    @jax.jit
    def _rebuild_chunk(ZG_b, v_b, S_b, Fch_b, W_b, qfr_b):
        A_lr = ZG_b * jnp.sqrt(v_b)[:, None, :]
        VLR = jnp.einsum("bmg,bng->bmn", jnp.conj(A_lr), A_lr)
        Sc = jnp.conj(S_b)
        VLRc = Sc @ VLR @ Sc
        qG = qfr_b[:, None, :] + GS_f[None, :, :]
        zt = jnp.exp(-2j * jnp.pi
                     * jnp.einsum("mi,bgi->bmg", rmu, qG)) * Fch_b
        A = zt * jnp.sqrt(W_b)[:, None, :]
        V = jnp.einsum("bmg,bng->bmn", jnp.conj(A), A)
        d = (V - VLRc).reshape(V.shape[0], -1)
        return (jnp.linalg.norm(d, axis=1)
                / jnp.linalg.norm(VLRc.reshape(VLRc.shape[0], -1), axis=1))

    for q0 in range(0, zx["nq"], 48):
        sl = slice(q0, min(q0 + 48, zx["nq"]))
        rr.extend(np.asarray(jax.device_get(_rebuild_chunk(
            jnp.asarray(zx["ZG"][sl]), jnp.asarray(v_lr_all[sl]),
            jnp.asarray(prep["S"][sl]), jnp.asarray(prep["Fch"][sl]),
            jnp.asarray(prep["W"][sl]), jnp.asarray(zx["qfr"][sl])))))
    log("F_own_rebuild_vs_cleaned_LR_tile_max", float(np.max(rr)), 1e-6)
    assert ok, "vq_interp null battery FAILED — stop"


# ===========================================================================
# exciton swap metric (reference build_hdir / exciton_evs, verbatim: TDA
# gap-window Hamiltonian, direct term from stored W0; only the exchange
# block B is swapped between truth and prediction)
# ===========================================================================
def build_hdir(zx, q0, nvw=3, ncw=3):
    # ABSENT means the restart file's W0 was never persisted — the loader
    # binds this key only when ``W0_ready`` says so.  Say that, rather than
    # letting a bare KeyError stand in for a diagnosis the loader already
    # made.
    if "W0" not in zx:
        raise KeyError(
            "build_hdir needs a persisted W0_qmunu, and the restart file "
            "does not have one: the dataset is missing, or it is the "
            "full-size ZERO placeholder gw_init allocates (W0_ready=False). "
            "The direct term would be identically zero and the exciton "
            "binding would be entirely absent — re-run GW screening before "
            "trusting this diagnostic.")
    kg = zx["kgrid"]
    # THE Q AXIS MUST BE THE FULL BZ, AND THIS IS THE ONLY PLACE THAT SAYS SO.
    # ``qkk`` below is built from ``k_lookup`` and therefore holds FULL-BZ q
    # indices, and ``W0[qkk[k]]`` fancy-indexes by them.  A W0 whose q axis
    # is anything else — a q_irr wedge, a truncated write, a reshaped
    # artifact — does not raise there.  It returns rows belonging to
    # DIFFERENT q's, and this diagnostic then reports a plausible, wrong
    # exciton spectrum: numbers that survive every downstream check because
    # nothing downstream checks this.  The sharded readers in ``bse_io``
    # (``_MunuSlabPlan``, ``_resolve_munu_reader``) already refuse exactly
    # this disagreement; this site was the hole in that set.  Asked of the
    # lazy handle, before any work, so the refusal costs nothing.
    #
    # The extents are read into locals BEFORE the test on purpose: the
    # W0_ready ratchet in ``tests/test_bse_w0_ready_gate.py`` requires
    # every ``if`` whose body names ``zx["W0"]`` to be testing the
    # persisted flag, and it is right to.  This refusal is a different
    # question about the same object, so it keeps the subscript out of
    # the guarded body rather than teaching that matcher an exception.
    _nq_expected = int(np.prod(np.asarray(kg, dtype=np.int64)))
    _nq_w0 = int(zx["W0"].shape[0])
    if _nq_w0 != _nq_expected:
        raise ValueError(
            f"build_hdir: W0_qmunu has {_nq_w0} q rows but the deck's "
            f"k-grid {tuple(int(v) for v in np.asarray(kg))} needs "
            f"{_nq_expected}.  The q lookup below indexes W0 by FULL-BZ "
            f"q, so any other length silently returns the wrong rows "
            f"instead of failing.  A q_irr wedge must be unfolded on read "
            f"(symmetry_maps.qirr_store.read_tensor) before it reaches "
            f"here.")
    cs = list(range(zx["nv"], zx["nv"] + ncw))
    vs = list(range(zx["nv"] - nvw, zx["nv"]))
    npair = zx["nk"] * ncw * nvw
    kqs = np.array([kq_index(zx, k, q0) for k in range(zx["nk"])])
    qkk = np.array([[zx["k_lookup"][tuple((zx["k_int"][k] - zx["k_int"][kp])
                                          % kg)]
                     for kp in range(zx["nk"])] for k in range(zx["nk"])])
    D = np.array([zx["enk"][kqs[k], c] - zx["enk"][k, v]
                  for k in range(zx["nk"]) for c in cs for v in vs])
    psic = np.ascontiguousarray(zx["psi"][:, cs])
    psiv = np.ascontiguousarray(zx["psi"][:, vs])
    psic_kq = psic[kqs]
    # ``zx["W0"]`` is a lazy h5py dataset (see load_zeta_coarse) and the q
    # lookup below is an unsorted permutation, which h5py cannot fancy-index.
    # This diagnostic wants every q anyway, so materialise once here.
    W0 = zx["W0"][()]
    H = np.zeros((npair, npair), dtype=np.complex128)
    bs = ncw * nvw
    for k in range(zx["nk"]):
        Tc = np.einsum("csm,KCsm->KcCm", np.conj(psic_kq[k]), psic_kq,
                       optimize=True)
        Tv = np.einsum("vsm,KVsm->KvVm", psiv[k], np.conj(psiv),
                       optimize=True)
        Wg = W0[qkk[k]]
        blk = np.einsum("KcCm,Kmn,KvVn->KcvCV", Tc, Wg, Tv, optimize=True)
        H[k * bs:(k + 1) * bs] = blk.transpose(1, 2, 0, 3, 4).reshape(bs,
                                                                      npair)
    return D, H / zx["nk"]


def exciton_evs(zx, D, Hdir, B, nstate=4):
    H = np.diag(D).astype(np.complex128) - Hdir + B / zx["nk"]
    H = 0.5 * (H + np.conj(H.T))
    return np.linalg.eigvalsh(H)[:nstate]


# ===========================================================================
# Per-Q ζ REFIT — the compute-don't-interpolate ground truth (§2d / §11.4)
# ===========================================================================
# The production default V_Q source: redo the ISDF fit AT the target
# momentum from htransform-reconstructed full-r wavefunctions.  This is the
# off-grid ground truth the interpolation program has been missing — every
# off-grid accuracy number quoted for eval_vq is scored against THIS.
#
# Per-element (fit conventions == the GW producer, gated by the on-grid
# null refit-vs-stored below):
#     ρ̃_kmn(r) = Σ_s conj(u^{ht}_{m, wrap(k−q), s}(r)) u_{n, k, s}(r)
#         (cell-periodic u at wrapped labels — torus convention, both legs;
#          m-leg from htransform: u_m(r) = Σ_α c_{m,wrap(k−q)}[α] B_full[α](r))
#     C_Q[μν] = Σ_{k,mn} conj(ρ̃(r_μ)) ρ̃(r_ν)
#     Z_Q[μr] = Σ_{k,mn} conj(ρ̃(r_μ)) ρ̃(r)
#     ζ̃_Q    = C_Q⁺ Z_Q                                (THE PRODUCER'S SOLVE:
#               isdf.core.solve_zeta_charge_dense at the charge_zeta_solve
#               and zeta_rcond THIS BUNDLE'S ζ was fitted with, read from
#               its own isdf_header/fit_provenance — see _zeta_solve_of)
#     ζ̃_Q(G) = FFT_r ζ̃_Q  gathered on the sphere |bᵀ(q+G)|² ≤ cutoff
#     V_Q    = Σ_G conj(ζ̃(G)) v(q+G) ζ̃(G)
#
# Scale note: this mode holds full-r ψ for the whole fit window on device
# (~(nk·nb)·(ns·n_rtot)·16 B — ~1 GB on the MoS2 3×3 fixtures) and costs a
# fit-scale GEMM chain per Q.  It is the EXPENSIVE mode by design; the
# fixture-scale target is 1 GPU, minutes per Q.

def read_zeta_fit_provenance(zeta_file: str) -> dict | None:
    """The producer's ζ-fit stamp off ``zeta_q.h5``; ``None`` if absent.

    ``isdf_header/fit_provenance`` is a JSON string written by
    ``gw.gw_init._zeta_fit_provenance`` AFTER ``mark_zeta_done``, so its
    presence means the ζ on disk is both complete and described.  THE BUNDLE
    IS THE TRUTH: it records the EFFECTIVE (post-``LORRAX_ZETA_RCOND``)
    conditioning knobs and the EXACT band windows the fit ran on, which is
    what a refit reproducing that fit needs — the deck beside it may have
    moved, and on a decoupled ``zeta_nband`` run the deck's band window is
    not the ζ-fit window at all.

    Scalar metadata through serial h5py (:func:`file_io.read_isdf_header`),
    safe on every rank before any tensor moves — same contract as
    ``read_coulomb_policy_from_h5``.
    """
    import json

    from file_io.isdf_header import read_isdf_header
    try:
        prov = read_isdf_header(zeta_file).fit_provenance
    except (OSError, KeyError, ValueError):
        return None
    if not prov:
        return None
    try:
        out = json.loads(prov)
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _zeta_solve_of(prov: dict, zeta_file: str) -> tuple[str, float, float]:
    """``(charge_zeta_solve, zeta_rcond, zeta_ridge)`` of the fit on disk.

    The three values ``isdf.core.solve_zeta_charge_dense`` needs to redo the
    producer's ζ solve.  ``zeta_rcond``/``zeta_ridge`` are stored as the
    provenance's own strings (``repr(value)``, or the raw env string when the
    deprecated env twin won — ``isdf.core.deprecated_env_record``), so they
    parse as floats either way and they are the EFFECTIVE values, which is
    the whole point of reading them here instead of off the deck.
    """
    kind = str(prov.get("charge_zeta_solve", "")).strip().lower()
    if kind not in ("rank_truncate", "cholesky"):
        raise SystemExit(
            f"exciton_bands --vq-mode=refit: {zeta_file} records "
            f"charge_zeta_solve={prov.get('charge_zeta_solve')!r}, which is "
            f"not a charge-channel ζ solve this refit can reproduce.  The "
            f"refit must run the SAME solve the producer ran; it will not "
            f"guess one.")
    try:
        rcond = float(prov["zeta_rcond"])
        ridge = float(prov.get("zeta_ridge", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"exciton_bands --vq-mode=refit: {zeta_file}'s fit_provenance "
            f"does not carry a parseable zeta_rcond/zeta_ridge "
            f"({prov.get('zeta_rcond')!r} / {prov.get('zeta_ridge')!r}).  "
            f"Those are the conditioning knobs ζ was fitted under and the "
            f"refit reproduces them exactly; it has no default for them."
        ) from exc
    return kind, rcond, ridge


def _zeta_fit_window_of(prov: dict) -> tuple[int, int] | None:
    """The ABSOLUTE band window ζ was fitted on, or ``None`` when the two
    halves of the pair density do not share one.

    The producer fits ``ρ_mn`` with m from ``band_range_left`` and n from
    ``band_range_right`` — ``(b0, b3)`` and ``(b1, b4)`` on a stock deck.
    The refit fits a SYMMETRIC window (one band axis, both legs), so it can
    only reproduce ζ when those two ranges coincide; when they do not, this
    returns ``None`` and the caller keeps the historical behaviour of fitting
    the whole stored band axis.  On every deck the tile null has ever passed
    on, ``nval == nelec`` and ``ncond == nband − nelec`` make them coincide.
    """
    try:
        bl = [int(v) for v in prov["band_range_left"]]
        br = [int(v) for v in prov["band_range_right"]]
    except (KeyError, TypeError, ValueError):
        return None
    return (bl[0], bl[1]) if bl == br else None


def refit_window_view(zx, band_range, log_fn=print,
                      degeneracy_mode: str = "strict",
                      degeneracy_tol_ry: float | None = None,
                      window_mode: str = "bse"):
    """A band-axis VIEW of ``zx`` on a SUB-WINDOW of the stored band axis.

    THE REFIT DOES NOT HAVE TO FIT THE WHOLE ζ-FIT WINDOW.  ``refit_vq``
    re-fits ζ at the target q from the pair densities of whatever band window
    it is handed; the ζ-fit window is the window that reproduces the
    producer's ζ, and therefore the producer's stored ``V_qmunu`` tiles, but
    it is not the only *legal* window.  For an exciton band structure the
    object being built is a BSE-window object — the exchange kernel contracts
    BSE-window (c, v) pairs and nothing else — so a ζ' fitted on the BSE
    window (plus the htransform's conduction guard bands, which is exactly
    what the deck's ``nval``/``ncond`` name) carries every pair density the
    kernel will ever ask it for.

    WHY THIS EXISTS AT ALL: the Galerkin rank bound.  ``build_fH_R`` needs the
    parent ISDF basis to span ``nk·nb``, i.e. ``n_μ,parent · n_s ≥ nk · nb``.
    On the Si 4×4×4 / 960-centroid lineage that is 1920 ≥ 64·nb, so the ζ-fit
    window (nb = 60, bound 3840) is unreachable and a 20-band window
    (bound 1280) is comfortable.  The window that makes the refit reproduce
    the stored tiles is exactly the window that breaks its htransform leg —
    see ``tests/known_failures/2026-08-10-exciton-bands-offgrid-Q-is-slab-only.md``.

    WHAT IT COSTS, AND WHERE THAT IS PAID.  ζ' ≠ ζ, so the refit will NOT
    reproduce the stored ``V_qmunu`` tiles and :func:`refit_ongrid_null` — a
    TILE-level identity — is not the gate for this mode and must not be run
    with a widened tolerance to make it pass.  The gate moves up one level, to
    the contracted object: on-grid BSE eigenvalues through the refit route
    against the same eigenvalues through the stored tiles.  ``exciton_bands``
    owns that gate because the contraction is its.

    ``band_range`` is the deck's ABSOLUTE band range ``(b_lo, b_hi)`` — the
    same ``(nelec − nval, nelec + ncond)`` ``initialize_wfns`` used.  It is
    located inside the ζ-fit window through the restart's own ``band_window``
    stamp (``tagged_arrays``' five canonical edges), never by assuming the two
    windows share an origin.

    Both edges of the sub-window are degeneracy-checked at
    ``degeneracy_mode`` (default ``strict``): a refit window that cuts a
    Kramers pair fits ζ' on half a multiplet, which is not a subspace of
    anything.  Edges that coincide with the ζ-fit window's own edges are
    boundaries of the calculation, not cuts, and are skipped by
    :func:`common.band_degeneracy.check_band_window` itself.

    ``window_mode`` LABELS the slice, and only the label: the arithmetic is
    the same either way.  ``"bse"`` is the narrowing above — ζ' ≠ ζ, tile null
    off, contracted gate on.  ``"zeta"`` is the OTHER caller (2026-08-11): on a
    bundle whose deck decoupled ``zeta_nband`` from ``nband``, the producer's ζ
    was fitted on a STRICT SUB-WINDOW of the ``psi_full_y`` band axis, so
    reaching the producer's own window is itself a slice — and there the tile
    null is exactly the right gate and stays armed.  Passing the wrong label
    would print the wrong certification story next to a correct number, which
    is the failure mode this argument exists to prevent.
    """
    from common.band_degeneracy import DEGENERACY_TOL_RY, check_band_window

    if window_mode not in ("zeta", "bse"):
        raise SystemExit(
            f"refit_window_view: window_mode={window_mode!r} is not 'zeta' "
            f"or 'bse'.")
    b_lo, b_hi = int(band_range[0]), int(band_range[1])
    fr = zx.get("_h5_restart")
    if fr is None or "band_window" not in fr:
        raise SystemExit(
            f"exciton_bands --refit-window={window_mode}: {zx.get('restart_file')} "
            f"carries no ``band_window`` stamp, so the deck's absolute band "
            f"range ({b_lo}, {b_hi}) cannot be located inside the stored "
            f"psi_full_y's {zx['nb']}-band ζ-fit window.  Without that offset "
            f"the band axis would be sliced from an assumed origin and the "
            f"refit would fit ζ' on the wrong bands — silently, since every "
            f"shape still matches.  The stamp is written by "
            f"tagged_arrays.write_restart_state_to_h5; re-run the GW to get a "
            f"bundle that carries one, or run --refit-window=zeta.")
    bw = [int(v) for v in np.asarray(fr["band_window"]).ravel()]
    z_lo, z_hi = bw[0], bw[4]
    if (z_hi - z_lo) != int(zx["nb"]):
        raise SystemExit(
            f"exciton_bands --refit-window={window_mode}: the restart's band_window "
            f"stamp {tuple(bw)} spans {z_hi - z_lo} bands but psi_full_y "
            f"carries nb={zx['nb']}.  The stamp and the tensor disagree about "
            f"the window they were written under, so no slice of the band "
            f"axis can be trusted.")
    if not (z_lo <= b_lo < b_hi <= z_hi):
        raise SystemExit(
            f"exciton_bands --refit-window={window_mode}: the deck's band window "
            f"[{b_lo}, {b_hi}) is not contained in the bundle's ζ-fit window "
            f"[{z_lo}, {z_hi}).  The refit fits ζ' from the STORED "
            f"psi_full_y, so it can only narrow that window, never reach "
            f"outside it.  Lower the deck's nval/ncond into the stored range.")
    lo, hi = b_lo - z_lo, b_hi - z_lo
    if (hi - lo) == int(zx["nb"]):
        return zx                       # identity slice: the ζ-fit window
    # Degeneracy on the REFIT window's own edges.  ``enk`` is (nk, nb_ζ) Ry in
    # the ζ-fit window's indexing, which is the indexing ``lo``/``hi`` are in.
    check_band_window(
        np.asarray(zx["enk"]), lo, hi,
        tol_ry=(DEGENERACY_TOL_RY if degeneracy_tol_ry is None
                else float(degeneracy_tol_ry)),
        mode=degeneracy_mode,
        where="exciton_bands refit zeta-fit window", log=log_fn)
    zw = dict(zx)
    zw["psi"] = zx["psi"][:, lo:hi]
    zw["enk"] = np.asarray(zx["enk"])[:, lo:hi]
    zw["nb"] = hi - lo
    zw["_refit_window_abs"] = (b_lo, b_hi)
    _shape = (f"ζ' is re-fitted on absolute bands [{b_lo}, {b_hi}) "
              f"({hi - lo} of psi_full_y's {zx['nb']}), so the Galerkin rank "
              f"bound is nk·nb = {zx['nk']}·{hi - lo} = "
              f"{zx['nk'] * (hi - lo)} instead of {zx['nk'] * zx['nb']}.")
    if window_mode == "bse":
        log_fn(f"  [refit] BSE-WINDOW refit: {_shape}  ζ' is NOT the "
               f"producer's ζ and the stored V_qmunu tiles will NOT come "
               f"back — the certification is the contracted on-grid "
               f"eigenvalue gate in exciton_bands, not the tile null.")
    else:
        log_fn(f"  [refit] ζ-FIT-WINDOW refit on a DECOUPLED bundle: "
               f"{_shape}  That window IS the producer's own "
               f"(zeta_nband < nband, read from the ζ file's "
               f"fit_provenance), so ζ' is the producer's ζ and the tile "
               f"null stays armed as the gate.")
    return zw


#: Guard bands the refit's fH window carries ABOVE the ζ-fit window, by
#: default.  Four, from measurement rather than taste: on both probed parents
#: of ``tests/known_failures/2026-08-11-fifth-wall-is-the-f-transform-\
#: shoulder.md`` §2 the bands whose ``min_k |f|/max|f|`` reaches exactly zero
#: are the top FOUR (dp2628n20, nb 20) and the top TWO (p2628n52, nb 52), and
#: the first band that never gets closer to zero than ~4e-4 of max|f| is four
#: below the edge on the wider one.  Same number the ``exciton_bands``
#: guard-band warning has used since the Si degeneracy root-cause, and for a
#: related reason.
REFIT_N_GUARD_DEFAULT = 4


def refit_prepare(input_file: str, mesh_xy: Mesh, zx, log_fn=print,
                  r_chunk: int = 2048, policy=None, keep_idx=None,
                  window_mode: str = "zeta",
                  degeneracy_mode: str = "strict",
                  degeneracy_tol_ry: float | None = None,
                  n_guard: int | None = None):
    """One-time refit state: htransform handles + full-r α-basis.

    THE TWO-WINDOW CONTRACT.  There are two windows here and they are not the
    same window:

      * the **ζ-fit window** — the producer's, exactly.  ζ' is re-fitted at
        the target Q from the pair densities of THIS window and no other, or
        it is not the producer's ζ and ``refit_ongrid_null`` (correctly)
        refuses it.  ``zx_fit`` carries it; ``refit_vq`` fits in it.
      * the **fH window** — the ζ window PLUS ``n_guard`` bands above it.
        ``ctilde``, ``B_at_mu``, ``enk_sigma``, ``B_full`` and ``rank`` all
        live here.

    WHY THEY MUST DIFFER.  ``f(ε)`` is identically zero for ε ≥ ``shift`` and
    the htransform sets ``shift = max_k ε[nb−1]`` — the top band of its OWN
    window.  Pinning the fH window to the ζ window therefore asks
    ``compute_wfns_fi`` for precisely the bands ``fH`` cannot represent: the
    top band vanishes at the k that defines the shift, and ``f`` vanishes to
    second order approaching it, so a whole SHOULDER of bands below the top
    carries a per-cent or less of fH's weight and ``eigh`` fills those slots
    out of fH's null space.  Measured (``2026-08-11-fifth-wall-is-the-f-\
    transform-shoulder.md``): four bands exactly zeroed at three k each on
    ``dp2628n20``, α-space overlap ‖O[m,:]‖ collapsing to 0.23…0.27, and an
    on-grid tile null of 1.267/1.409 against a 5.0e-02 bracket — while the
    representation itself measured 8.9e-15 orthonormality and a 4.3e-15
    Galerkin residual.  The representation was never the defect; the REQUEST
    was.

    WHAT IT COSTS.  The Galerkin capacity bound tightens from ``nk·nb_ζ`` to
    ``nk·(nb_ζ + n_guard) ≤ n_μ·n_s``, and the deck has to be able to LOAD the
    guards (``initialize_wfns`` raises ``nband`` for them and refuses when the
    WFN has no such bands — a guard band past the file arrives as exact zeros
    and nothing downstream would say so).  ``n_guard=0`` restores the
    zero-guard configuration exactly, and is the red twin: it must fail the
    tile null loudly.

    Returns ``rst`` dict:
      zx_fit   the band-axis view of ``zx`` the refit actually fits in — the
          bundle itself under ``window_mode="zeta"``, a sub-window view under
          ``"bse"`` (:func:`refit_window_view`).  EVERY later ``refit_vq``
          call must be handed this, not the caller's ``zx``.
      ctilde, B_at_mu, enk_sigma, kgrid_co — htransform setup on the WIDE
          (fH) window: ``nb_wide = zx_fit["nb"] + n_guard``
      psi_r    (nk·nb_ζ, ns·n_rp)  stored ζ-WINDOW u on the full r-grid,
                                 device, zero-padded on r to a multiple of
                                 r_chunk.  The guard bands are deliberately
                                 NOT here: they shape fH, they are not pair
                                 density.
      B_full   (rank, ns·n_rp)   α-basis on the full r-grid = W_proj ψ_r,
                                 built from the WIDE ψ because ``W_proj`` and
                                 ``ctilde`` are the wide window's
      n_guard, nb_wide, window_abs (ζ), window_abs_fh (wide) — the contract,
          recorded so a reader of ``rst`` never has to infer which is which
      n_rtot, r_chunk, galerkin_rel — bookkeeping + printed residual
      zeta_solve  ``(charge_zeta_solve, zeta_rcond, zeta_ridge)`` of the fit
          on disk — read from the ζ file's own ``isdf_header/fit_provenance``
          and NOT from the deck.  ``refit_vq`` refuses without it.
    """
    # ARGUMENT VALIDATION FIRST — it costs nothing, it needs no file to be
    # readable, and it runs BEFORE the htransform import below, whose module
    # body brings up the communicator stack and the REQUIRED-FFI gate.
    n_guard = REFIT_N_GUARD_DEFAULT if n_guard is None else int(n_guard)
    if n_guard < 0:
        raise SystemExit(
            f"refit_prepare: n_guard={n_guard} is negative.  Guard bands sit "
            f"ABOVE the ζ-fit window; 0 is the zero-guard configuration the "
            f"f-shoulder row convicts, and there is nothing below it.")

    from gw.gw_config import read_lorrax_input
    from bandstructure.htransform import initialize_wfns
    from common.wfn_transforms import iter_psi_rchunk_bandwise

    # THE ζ SOLVE, BEFORE ANYTHING EXPENSIVE.  A refit whose solve does not
    # match the producer's cannot reproduce the producer's tiles no matter how
    # good its ψ representation is (five parents, §4 of
    # tests/known_failures/2026-08-11-narrowed-zeta-window-clears-fh-and-the-\
    # tile-null-still-refuses.md), so the fit stamp is read first and the run
    # refuses here rather than after the htransform.
    _prov = read_zeta_fit_provenance(zx["zeta_file"])
    if _prov is None:
        raise SystemExit(
            f"exciton_bands --vq-mode=refit: {zx['zeta_file']} carries no "
            f"``isdf_header/fit_provenance``, so the ζ solve it was fitted "
            f"with — charge_zeta_solve, zeta_rcond, zeta_ridge — is unknown. "
            f"The refit re-solves the SAME system at the target Q and must "
            f"use the SAME solve: a plain ridged Cholesky keeps the near-null "
            f"directions the producer's rank truncation discarded and V_Q is "
            f"quadratic in them.  The stamp is written by gw.gw_init after "
            f"the fit completes; re-run the ζ fit to get a stamped file, or "
            f"use --vq-mode=ongrid, which needs no refit.")
    _zeta_solve = _zeta_solve_of(_prov, zx["zeta_file"])
    log_fn(f"  [refit] ζ solve = THE PRODUCER'S: {_zeta_solve[0]} at "
           f"zeta_rcond={_zeta_solve[1]:.3e}, zeta_ridge={_zeta_solve[2]:.3e} "
           f"(read from {os.path.basename(zx['zeta_file'])} "
           f"isdf_header/fit_provenance, not from the deck)")
    params = read_lorrax_input(input_file)
    (wfn, sym, meta, _, _S, ctilde, B_at_mu, enk_sigma,
     W_proj) = initialize_wfns(input_file, params, log_fn, mesh_xy=mesh_xy,
                               return_full_proj=True,
                               n_guard_bands=n_guard)
    nk, nb_wide, rank = (int(ctilde.shape[0]), int(ctilde.shape[1]),
                         int(ctilde.shape[2]))
    # ``nb`` is, everywhere below and in every ``refit_vq`` call, the ζ-FIT
    # window's band count — the window ζ' is fitted on and the window the
    # pair densities live in.  ``nb_wide`` is fH's.  They differ by exactly
    # the guards, and conflating them is the defect this contract fixes.
    nb = nb_wide - n_guard
    ns = int(B_at_mu.shape[1])
    # DOWNFOLDED BUNDLE: the htransform leg fits in the PARENT ISDF basis (it
    # is sized by nk·nb, not by mu_S — resolve_isdf_basis), so B_at_mu comes
    # back at the parent's mu while every ζ object here is at the child's.
    # Slice the mu axis to the kept rows, which is the same column slice that
    # defines the bundle's own psi_full_y, and do it HERE so psi_rmu_Y is born
    # at the child's width instead of being trimmed after the contraction.
    if keep_idx is not None:
        ki = np.asarray(keep_idx, dtype=int)
        if int(B_at_mu.shape[2]) != int(zx["n_mu"]):
            B_at_mu = B_at_mu[:, :, jnp.asarray(ki)]
            log_fn(f"  [refit] downfold: B_at_mu mu axis sliced "
                   f"{int(B_at_mu.shape[2])} <- parent basis, to the "
                   f"{len(ki)} kept centroid rows")
    assert nk == zx["nk"] and ns == zx["ns"], \
        (f"htransform window (nk={nk}, ns={ns}) != zeta-fit window "
         f"(nk={zx['nk']}, ns={zx['ns']})")
    # The deck's ABSOLUTE band range — the ζ-FIT window, which is what the
    # deck's nval/ncond name and what ``zx`` carries.  ``initialize_wfns``
    # built ``ctilde`` over ``band_range_fh``, which is this plus the guards.
    nelec = int(wfn.nelec)
    band_range = (nelec - int(params["nval"]), nelec + int(params["ncond"]))
    band_range_fh = (band_range[0], band_range[1] + n_guard)
    if window_mode not in ("zeta", "bse"):
        raise SystemExit(
            f"refit_prepare: window_mode={window_mode!r} is not 'zeta' or "
            f"'bse'.")
    if window_mode == "bse":
        # ζ' on the deck's (narrower) window; the tile null does not apply and
        # the driver's contracted eigenvalue gate is the certification.
        zx = refit_window_view(zx, band_range, log_fn=log_fn,
                               degeneracy_mode=degeneracy_mode,
                               degeneracy_tol_ry=degeneracy_tol_ry,
                               window_mode="bse")
    else:
        # THE ζ-FIT WINDOW IS WHAT THE PRODUCER'S STAMP SAYS IT IS, not what
        # psi_full_y's extent says.  Those were the same object until the
        # 2026-08-11 ``zeta_nband`` decoupling: ``nband`` used to set BOTH the
        # χ0/Σ band-sum top ``b4`` and the top of the ζ fit's own band ranges,
        # so the stored band axis and the ζ-fit window could not differ.  With
        # ``zeta_nband < nband`` the producer fits ζ on a strict SUB-window of
        # the band axis it then stores, and a refit that fitted the whole axis
        # would be fitting a DIFFERENT ζ — with every shape still matching,
        # which is how this would go unnoticed.  The slice is the same one
        # ``--refit-window=bse`` takes; the difference is that this window IS
        # the producer's, so the tile null remains the gate.
        _zw = _zeta_fit_window_of(_prov)
        if _zw is not None and (_zw[1] - _zw[0]) != int(zx["nb"]):
            zx = refit_window_view(zx, _zw, log_fn=log_fn,
                                   degeneracy_mode=degeneracy_mode,
                                   degeneracy_tol_ry=degeneracy_tol_ry,
                                   window_mode="zeta")
        if _zw is not None and (int(band_range[0]), int(band_range[1])) != _zw:
            # LENGTH IS NOT IDENTITY.  The check below compares band COUNTS,
            # so a deck whose window has the right width at the wrong origin
            # would sail through it and fit ζ' on the wrong bands with no
            # shape error anywhere.  The producer wrote down which bands;
            # compare against those.
            raise SystemExit(
                f"exciton_bands --refit-window=zeta: the deck's band window "
                f"[{band_range[0]}, {band_range[1]}) (nelec−nval, nelec+ncond) "
                f"is not the window the producer fitted ζ on, which "
                f"{os.path.basename(zx['zeta_file'])}'s fit_provenance records "
                f"as [{_zw[0]}, {_zw[1]}).  The refit reproduces the "
                f"producer's ζ from the pair densities of the producer's "
                f"window, so it has to BE that window.  Set the deck's "
                f"nval/ncond to reach exactly [{_zw[0]}, {_zw[1]}) — on a "
                f"decoupled bundle (zeta_nband < nband) that is "
                f"nval+ncond = {_zw[1] - _zw[0]}, NOT nband — or run "
                f"--refit-window=bse, which fits ζ' on the deck's window and "
                f"certifies on the contracted object instead.")
    if nb != zx["nb"]:
        raise SystemExit(
            f"exciton_bands --vq-mode=refit: the ζ-FIT window is nb={nb} "
            f"(deck nval+ncond; fH spans {nb_wide} = that plus {n_guard} "
            f"guard band(s)) but the ζ the refit fits against "
            f"carries nb={zx['nb']}.  Under the default "
            f"``--refit-window=zeta`` the refit RE-FITS zeta at the target Q "
            f"from the pair densities of the PRODUCER's window, so it has to "
            f"be that window — a narrower one gives a different zeta and the "
            f"on-grid TILE null would (correctly) refuse it.  Two ways "
            f"forward:\n"
            f"  * set the deck's nval/ncond to the GW run's "
            f"({zx['nb']} bands total); --n-val/--n-cond still choose the BSE "
            f"window inside it.  Needs n_mu,parent*n_s >= nk*(nb + n_guard) = "
            f"{zx['nk'] * (zx['nb'] + n_guard)} for the htransform Galerkin "
            f"leg (the guards are inside the bound too);\n"
            f"  * or ``--refit-window=bse``, which fits ζ' on the DECK's "
            f"window instead (the BSE window plus its guard bands) and "
            f"certifies on the CONTRACTED object — on-grid BSE eigenvalues, "
            f"refit route vs stored route — because a windowed ζ' cannot and "
            f"must not reproduce the stored V_qmunu tiles.")
    fg = tuple(int(x) for x in meta.fft_grid)
    assert fg == (zx["nx"], zx["ny"], zx["nz"]), \
        f"WFN FFT grid {fg} != zeta_q.h5 grid {(zx['nx'], zx['ny'], zx['nz'])}"
    n_rtot = zx["n_rtot"]

    # ── THE TWO-WINDOW CONTRACT, announced ───────────────────────────────
    # Same shape of report and the same thresholds as ``exciton_bands``'s own
    # guard-band block, because it is the same quantity: bands above what the
    # caller reads, kept so the f-shoulder lands on nobody.  Below four is the
    # measured shoulder depth on the probed parents; above sixteen is the
    # documented on-grid cliff, which is not hypothetical (MoS2/640c on-grid
    # |Δε_c| ~1 meV at nband ≤ 48 against ~955 meV at nband 80).
    log_fn(f"  [refit] two-window contract: ζ' is fitted on bands "
           f"[{band_range[0]}, {band_range[1]}) ({nb} bands, the producer's); "
           f"fH spans [{band_range_fh[0]}, {band_range_fh[1]}) ({nb_wide} "
           f"bands) = that plus {n_guard} guard band(s).  Galerkin capacity "
           f"needs n_mu*n_s = {int(zx['n_mu']) * ns} >= nk*nb_fh = "
           f"{nk * nb_wide}"
           + ("" if int(zx["n_mu"]) * ns >= nk * nb_wide else
              "  ** SHORT — build_fH_R's orthonormality gate decides **"))
    if n_guard < 4:
        log_fn(f"  [warn] only {n_guard} guard band(s) above the ζ-fit "
               f"window.  f(ε) ≡ 0 for ε ≥ max_k ε[nb_fh−1], and the dead "
               f"set measured FOUR bands deep on a 20-band window and TWO on "
               f"a 52-band one (2026-08-11-fifth-wall-is-the-f-transform-"
               f"shoulder.md §2), so the top of the ζ window may come back "
               f"from eigh as an arbitrary null-space direction.  The "
               f"f-shoulder gate in compute_wfns_fi is what decides — it "
               f"prints the dead set's actual depth on THIS deck; this is "
               f"the warning that makes it cheap to act on.")
    if n_guard > 16:
        log_fn(f"  [warn] {n_guard} guard bands is a LARGE interp window and "
               f"a large window does not improve — past a system-dependent "
               f"cliff it WRECKS — the recovered on-grid energies (MoS2/640c: "
               f"~1 meV at nband<=48, ~955 meV at nband 80).  Keep the guards "
               f"just deep enough to clear the f-shoulder.")

    # Stream the ψ of the fH (WIDE) window onto the full r-grid (host
    # assembly, one band chunk at a time), then push once to device.
    # ``band_range_fh`` is the window ``initialize_wfns`` built ``ctilde``
    # over, so it is the window ``W_proj`` projects and the ONLY window
    # ``B_full = W_proj @ ψ`` is defined on.  The ζ-window sub-block is sliced
    # back out below and is what ``refit_vq`` contracts; the guard bands exist
    # to shape fH and are deliberately absent from every pair density.
    psi_r_host = np.empty((nk, nb_wide, ns, n_rtot), dtype=np.complex128)
    for bc_range, psi_bc in iter_psi_rchunk_bandwise(
            wfn, sym, meta, mesh_xy, band_range_fh, 0, n_rtot,
            bool(params.get("bispinor", False)), band_chunk_size=16):
        lo = bc_range[0] - band_range_fh[0]
        hi = bc_range[1] - band_range_fh[0]
        # ``_to_host`` (= common.collectives.gather_to_host), NOT device_get.
        # ``iter_psi_rchunk_bandwise`` yields a mesh-sharded global array, so on a MULTI-PROCESS run its
        # shards live on other processes and ``jax.device_get`` raises
        # ("Fetching value for a jax.Array that spans non-addressable
        # devices") — measured at P=4/4-process, job 56612363 step .59, the
        # first time this function ever ran under the driver.  Its three-arm
        # branch is also why the naive repair is wrong: ``process_allgather(
        # tiled=True)`` on the fully-addressable P=1 array would concatenate
        # this process's whole copy and multiply the k axis by P.
        # SLICE TO THE CHUNK'S OWN WIDTH.  ``load_psi_gflat_padded`` pads the
        # band axis up to a multiple of the device count, so a chunk narrower
        # than that comes back WIDER than ``bc_range`` says — 9 requested
        # bands arriving as 12 at four devices.  Every window this function
        # ever saw before the two-window contract had a band count divisible
        # by the device count (52, 24, 20 at P=4), so the last chunk was
        # always already aligned and the pad never materialised; a guard count
        # that makes ``nb_fh`` odd (53, 55, 57) is the first caller to see it,
        # and it arrives as a numpy broadcast error naming two shapes and
        # neither the loader nor the pad.  The pad rows are physically zero,
        # so the slice discards nothing.
        _bc = _to_host(psi_bc)
        psi_r_host[:, lo:hi] = _bc[:, :hi - lo]
        del _bc
    n_rp = ((n_rtot + r_chunk - 1) // r_chunk) * r_chunk
    if n_rp > n_rtot:
        psi_r_host = np.concatenate(
            [psi_r_host, np.zeros((nk, nb_wide, ns, n_rp - n_rtot),
                                  dtype=np.complex128)], axis=3)
    psi_r_fh = jnp.asarray(psi_r_host.reshape(nk * nb_wide, ns * n_rp))
    # α-basis on full r; Galerkin fidelity printed (the refit floor at
    # on-grid q is bounded below by this residual)
    W_proj = jnp.asarray(W_proj)
    # ns folds with r: W_proj columns are (nk·nb_wide); psi_r_fh rows likewise
    # — but the SVD's column space folded (ns, n_mu); on full r the fold is
    # (ns, n_rp), consistent because ψ rows carry (s, r) in C order.
    B_full = W_proj @ psi_r_fh                   # (rank, ns·n_rp)
    rec = jnp.asarray(ctilde.reshape(nk * nb_wide, rank)) @ B_full
    gal = float(jnp.linalg.norm(rec - psi_r_fh) / jnp.linalg.norm(psi_r_fh))
    # TWO RESIDUALS, because the two windows are two objects.  The fH one is
    # what ``B_full`` and ``ctilde`` actually represent; the ζ one is the
    # floor that bounds the tile, because the pair densities the tile
    # contracts live in the ζ window and nowhere else.  Printing only the
    # first would attribute the guards' own (larger, harmless) representation
    # error to the tile; printing only the second would hide a wide window
    # that has stopped spanning.
    _r3 = rec.reshape(nk, nb_wide, ns * n_rp)[:, :nb]
    _p3 = psi_r_fh.reshape(nk, nb_wide, ns * n_rp)[:, :nb]
    gal_zeta = float(jnp.linalg.norm(_r3 - _p3) / jnp.linalg.norm(_p3))
    del rec, _r3, _p3
    log_fn(f"  [refit] Galerkin full-r residual ‖cB−ψ‖/‖ψ‖ = {gal:.3e} over "
           f"the fH window ({nb_wide} bands), {gal_zeta:.3e} over the ζ-fit "
           f"window ({nb} bands) — the ζ one is the refit-vs-stored on-grid "
           f"floor, since only ζ-window pair densities enter the tile")
    # The ζ-WINDOW ψ is what ``refit_vq`` contracts, on BOTH legs: the n leg
    # directly, the stored-m-leg twin by k-index.  Slice it out of the wide
    # host buffer and drop the wide device copy — the guards' job is done the
    # moment ``B_full`` exists.
    del psi_r_fh
    psi_r = jnp.asarray(
        np.ascontiguousarray(psi_r_host[:, :nb]).reshape(nk * nb, ns * n_rp))
    del psi_r_host
    return {"zx_fit": zx, "window_mode": window_mode,
            "window_abs": (int(band_range[0]), int(band_range[1])),
            "window_abs_fh": (int(band_range_fh[0]), int(band_range_fh[1])),
            "n_guard": int(n_guard), "nb_wide": int(nb_wide),
            "zeta_solve": _zeta_solve,
            "ctilde": ctilde, "B_at_mu": B_at_mu, "enk_sigma": enk_sigma,
            "kgrid_co": (int(meta.nkx), int(meta.nky), int(meta.nkz)),
            "psi_r": psi_r, "B_full": B_full, "n_rtot": n_rtot,
            "n_rp": n_rp, "r_chunk": int(r_chunk), "galerkin_rel": gal,
            "galerkin_rel_zeta": gal_zeta,
            "rank": rank,
            "v_on_set": make_v_on_set(zx, policy, log_fn=log_fn)}


def refit_ongrid_null(zx, rst, V_stored, kgrid_vq, mesh_xy, log_fn=print,
                      q_list=None, tol=5e-2):
    """THE gate: refit at a COARSE q must give back the STORED tile.

    CERTIFY WHERE CONSUMED.  Every off-grid exchange tile this driver writes
    comes out of :func:`refit_vq`, and the only place its answer can be
    checked against something independent is a q where the producer already
    wrote one down.  So before a single off-grid number is used, the same
    call is made at on-grid q and compared to ``V_qmunu[q]``.

    The floor is NOT machine precision and must not be set there: the refit
    reaches the target q through the htransform Galerkin representation of ψ,
    whose full-r residual ``refit_prepare`` prints, and the tile error is
    bounded below by it.  ``tol`` is therefore a REGRESSION bracket on the
    refit-vs-stored ratio, not an accuracy claim — the number that matters is
    printed next to it, and a lane reading this should compare it against the
    Galerkin residual, not against zero.  (The kernel itself is separately
    exact: rebuilding the stored tiles from the STORED ζ with the same door
    lands at 3.3e-14, so anything above that is the ψ representation, not the
    Coulomb.)

    NOT APPLICABLE TO A WINDOWED REFIT, and this refuses rather than being
    re-tuned.  ``--refit-window=bse`` fits ζ' on the BSE window, so ζ' ≠ ζ and
    the stored tiles — a full-ζ-fit-window object — are not the answer this
    call is computing.  A lane that reached for ``tol`` here to make that pass
    would be widening a bracket around a comparison that has no reason to
    hold.  The gate for that mode is the contracted eigenvalue certification
    in ``exciton_bands``; see :func:`refit_window_view`.

    Returns the list of (q_index, relative error).
    """
    if rst.get("window_mode", "zeta") != "zeta":
        raise SystemExit(
            f"refit_ongrid_null: this is a TILE-level identity against the "
            f"producer's stored V_qmunu, and this refit state was prepared "
            f"with window_mode={rst.get('window_mode')!r} — ζ' is fitted on "
            f"bands {rst.get('window_abs')}, not on the ζ-fit window, so it "
            f"is a DIFFERENT ζ and the stored tiles are not what it computes. "
            f"Certify the contracted object instead (on-grid BSE eigenvalues, "
            f"refit route vs stored route); do not widen ``tol`` here.")
    kg = np.asarray(kgrid_vq, dtype=int)
    if q_list is None:
        # Γ plus the three coarse q furthest from it — the head-dominated
        # slot and the zone-boundary ones, which fail differently.
        idx = np.stack(np.meshgrid(*[np.arange(k) for k in kg],
                                   indexing="ij"), axis=-1).reshape(-1, 3)
        q_list = [tuple(int(v) for v in idx[0])]
        far = np.argsort(-np.sum(np.minimum(idx, kg[None, :] - idx) ** 2,
                                 axis=1))
        for j in far[:3]:
            q_list.append(tuple(int(v) for v in idx[j]))
    sh = NamedSharding(mesh_xy, P("x", "y"))
    out = []
    worst = 0.0
    for (ix, iy, iz) in q_list:
        qfr = np.array([ix, iy, iz], dtype=np.float64)
        qfr = np.where(qfr > kg / 2.0, qfr - kg, qfr) / kg
        # NORMS ON DEVICE.  ``V_stored`` is the loader's sharded (mu, nu, ...)
        # tensor and on a multi-process run it is not fully addressable, so a
        # host gather here would die on exactly the geometry this driver runs
        # at.  jnp reductions over the (x, y) face give the same number and
        # need no gather.
        V_ref = jax.device_put(
            V_stored[:, :, ix % kg[0], iy % kg[1], iz % kg[2]], sh)
        n_mu = min(int(V_ref.shape[0]), int(zx["n_mu"]))
        V_new = refit_vq(zx, rst, qfr, mesh_xy, log_fn=lambda *_: None)
        pad = np.zeros(tuple(int(v) for v in V_ref.shape), dtype=np.complex128)
        pad[:n_mu, :n_mu] = V_new[:n_mu, :n_mu]
        V_new_d = device_put_process_local(pad, sh)
        rel = float(jnp.linalg.norm(V_new_d - V_ref)
                    / jnp.maximum(jnp.linalg.norm(V_ref), 1e-300))
        out.append(((ix, iy, iz), rel))
        worst = max(worst, rel)
        log_fn(f"  [refit-null] q=({ix},{iy},{iz}) frac="
               f"{np.array2string(qfr, precision=4)}: refit vs stored "
               f"V_qmunu rel = {rel:.3e}")
    log_fn(f"  [refit-null] worst {worst:.3e} over {len(out)} coarse q "
           f"(bracket {tol:.1e}; the Galerkin full-r residual printed above "
           f"is the physical floor, and the kernel's own on-grid null is "
           f"3.3e-14 — so this number is the htransform representation)")
    if not np.isfinite(worst) or worst > tol:
        raise SystemExit(
            f"exciton_bands: the per-Q exchange REFIT does not reproduce the "
            f"stored V_qmunu at on-grid q (worst relative {worst:.3e} against "
            f"the {tol:.1e} bracket).  Every off-grid tile this run would "
            f"write comes from the same call, so nothing downstream of here "
            f"is worth plotting.  Check the Coulomb-policy stamp "
            f"(sys_dim / mc_average_vcoul_body / bare_coulomb_cutoff) against "
            f"the run that wrote the bundle, and the Galerkin residual "
            f"refit_prepare printed — a large residual there is an htransform "
            f"window problem, not a kernel one.")
    return out


def zeta_r_to_sphere_q(zx, zeta, qw, fi):
    """ζ(r) rows → ζ(q+G) rows at the flat FFT slots ``fi``, host.

    THE PRODUCER'S FRAME, and the only place the refit states it::

        ZG_μ(G) = Σ_r e^{−2πi (q+G)·r} ζ_μ(r)

    — the Bloch factor multiplies ζ ON THE r GRID, before the FFT, which is
    exactly what the ζ writer does (``gw.isdf_fitting`` hands ``qvec_frac``
    to ``common.wfn_transforms.accumulate_rchunk_to_gflat``, whose per-q
    phase site is pre-FFT).  This is the device twin of the host
    :func:`to_sphere` four hundred lines up: same factor, same
    ``norm="backward"``, same flat-index gather, differing only in that
    ``to_sphere`` gathers the STORED sphere at a stored q index while this
    takes an explicit slot list, so it also serves an off-grid q.

    It is a named function rather than four lines inside :func:`refit_vq`
    because ``tests/test_refit_frame_convention.py`` pins it against
    :func:`to_sphere` on a random ζ at finite q, and that identity is the one
    the sixth wall broke: the refit used to apply a per-μ centroid winding
    phase ``e^{−2πi q·s_μ}`` to a phase-free FFT instead, which is exact only
    for a ζ that is a delta at s_μ, is an exact no-op at q = 0, and is why Γ
    was clean at 4.7e-06 while every finite q read 1.11–1.17.
    """
    n_mu = int(zeta.shape[0])
    ph_r = np.exp(-2j * np.pi * (zx["rfrac"] @ np.asarray(qw)))
    box = local_fftn3((jnp.asarray(zeta) * jnp.asarray(ph_r)[None, :]).reshape(
        n_mu, zx["nx"], zx["ny"], zx["nz"]), axes=(1, 2, 3), norm="backward")
    # ``_to_host`` (= common.collectives.gather_to_host), NOT device_get, and
    # THE PARITY OF n_μ IS WHY THAT WAS NOT OBVIOUS.  ζ inherits its μ-axis
    # sharding from ``psi_m_mu`` — ``bse_setup``'s ``psi_rmu_Y``, laid out
    # ``P(None, None, None, 'y')`` — so this box is μ-sharded across the
    # mesh and, at P>1, across PROCESSES.  ``jax.device_get`` on that raises
    # "Fetching value for a jax.Array that spans non-addressable (non process
    # local) devices" (measured at 4 GPUs in 4 processes on the μ=960 parent,
    # `qsign_recut_0811/_logs/xb_ctl_parent.log`).
    #
    # It hid for as long as it did because ``sharding_fit`` silently declines
    # a spec the extent cannot satisfy: at the μ=191 child, 191 % 2 != 0, the
    # μ axis is REPLICATED instead, and ``Array._value`` serves a fully
    # replicated array out of the local shard before it ever reaches the
    # addressability check — so the identical call works.  Every P=4 leg the
    # refit path had ever had was the odd-μ child, and odd μ cannot see this
    # by construction.  ``gather_to_host`` covers all three shapes in the one
    # branch it already owns: device_get when fully addressable (P=1),
    # addressable_data(0) when replicated (the odd-μ arm, still no
    # collective), process_allgather(tiled=True) when genuinely sharded.
    # Placement only — the tile is arithmetic on the same values.
    return _to_host(box.reshape(n_mu, zx["n_rtot"])[:, jnp.asarray(fi)])


def _sphere_millers(zx, qw):
    """All Miller G with |bᵀ(qw+G)|² ≤ zeta_cutoff (the fit sphere at qw)."""
    Kmax = np.sqrt(zx["zeta_cutoff"])
    nmax = [int(np.ceil(Kmax / np.linalg.norm(zx["bvec"][i]))) + 1
            for i in range(3)]
    gr = [np.arange(-n, n + 1) for n in nmax]
    GX, GY, GZ = np.meshgrid(*gr, indexing="ij")
    Gall = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 0)
    K2 = np.sum((zx["bvec"].T @ (np.asarray(qw)[:, None]
                                 + Gall.astype(np.float64))) ** 2, axis=0)
    return np.ascontiguousarray(Gall[:, K2 <= zx["zeta_cutoff"]])


_REFIT_KERNELS: dict = {}


def _refit_kernels(nk, nb, ns, n_mu, rank, r_chunk, zeta_solve):
    """Jitted refit chunk kernels, cached on the (shape) signature so every
    refit Q after the first is dispatch-only (per-q-recompile lesson).

    ``zeta_solve`` is the producer's ``(charge_zeta_solve, zeta_rcond,
    zeta_ridge)`` triple for the bundle being refitted (:func:`_zeta_solve_of`).
    It is part of the CACHE KEY, not just of the closure: the ζ solve is a
    different program per (kind, rcond), and a key that omitted it would hand
    the second bundle of a process the first bundle's compiled solve.
    """
    key = (nk, nb, ns, n_mu, rank, r_chunk, tuple(zeta_solve))
    hit = _REFIT_KERNELS.get(key)
    if hit is not None:
        return hit
    _solve_kind, _rcond, _ridge = zeta_solve
    # LAZY, and for the same reason every other import in this module is:
    # ``isdf.core``'s module body reaches ``distrib_la`` and the FFI plan
    # resolver, and this module is imported by host-only diagnostics that
    # never refit.  L1 → L1, so the layer map has nothing to say about it.
    from isdf.core import solve_zeta_charge_dense

    @jax.jit
    def _cq_and_x(psi_m_mu, psi_mu):
        # X[k,m,n,μ] = Σ_s conj(u^{ht}_{m,k−q,s}(μ)) u_{n,k,s}(μ)  (spin-traced)
        X = jnp.einsum("kmsu,knsu->kmnu", jnp.conj(psi_m_mu), psi_mu)
        Xf = X.reshape(nk * nb * nb, n_mu)
        C = jnp.conj(Xf).T @ Xf
        return Xf, 0.5 * (C + jnp.conj(C).T)

    @jax.jit
    def _psi_m_chunk(c_m, B_chunk):
        # ψ^{ht}_{m,k−q,s}(r) = Σ_α c_m[k,α,m] B_full[α,s,r]   (chunk of r)
        return jnp.einsum("kam,asr->kmsr", c_m,
                          B_chunk.reshape(rank, ns, r_chunk))

    @jax.jit
    def _z_chunk(psi_m, psi_chunk, Xf):
        rho = jnp.einsum("kmsr,knsr->kmnr", jnp.conj(psi_m),
                         psi_chunk.reshape(nk, nb, ns, r_chunk))
        return jnp.conj(Xf).T @ rho.reshape(nk * nb * nb, r_chunk)

    @jax.jit
    def _solve_zeta(C, Z):
        # THE PRODUCER'S SOLVE, not a second implementation of it.  This
        # used to be a plain Cholesky with a fixed 1e-14·|tr C| ridge, under
        # a comment citing a private ridged-Cholesky helper of ``isdf.core``
        # — a symbol that has never existed in this tree.  The production
        # charge path is
        # ``replicated_rank_truncate``: an eigh pseudo-inverse that DROPS
        # every direction with λ < zeta_rcond·λ_max.  A ridged Cholesky
        # KEEPS those directions and amplifies them by up to four orders
        # more than the producer's κ cap allows, so ζ' differed from ζ in
        # precisely the near-null subspace — and V_Q is quadratic in ζ.
        # The tile identity ``refit_ongrid_null`` measured the consequence
        # on five parents: worst-tile error monotone in the DISCARDED
        # fraction (4.7 % → 3.289, 7.9 % → 16.0, 39.4 % → 50.9, 58.6 % →
        # 139.9) against a 5.0e-02 bracket, while the htransform Galerkin
        # residual moved eight orders and the tile null did not follow it
        # (tests/known_failures/2026-08-11-narrowed-zeta-window-clears-fh-\
        # and-the-tile-null-still-refuses.md §4).
        return solve_zeta_charge_dense(
            C, Z, charge_zeta_solve=_solve_kind, zeta_rcond=_rcond,
            zeta_ridge=_ridge)

    kernels = (_cq_and_x, _psi_m_chunk, _z_chunk, _solve_zeta)
    _REFIT_KERNELS[key] = kernels
    return kernels


def refit_vq(zx, rst, q_tile_frac, mesh_xy: Mesh, log_fn=print,
             m_leg: str = "htransform"):
    """Ground-truth V at TILE momentum ``q_tile_frac`` via a per-Q ζ refit.

    Tile-momentum labeling matches ``V_qmunu`` / ``eval_vq``: the pair
    density carries conduction at wrap(k − q).  Returns the (n_μ, n_μ)
    host tile (Hermitian by construction).

    ``m_leg``: ``"htransform"`` (production — works at ANY q) or
    ``"stored"`` (on-grid q only: the m-leg is the stored grid ψ at the
    wrapped index — the formulation-null configuration that isolates the
    fit conventions from the htransform representation quality).

    On-grid null: ``refit_vq`` at a coarse q reproduces the stored
    ``V_qmunu[q]`` up to the Galerkin/htransform floor (printed by
    ``refit_prepare``); the driver's refit gate asserts this before any
    off-grid refit number is quoted.
    """
    import time as _time

    t0 = _time.time()
    nk, nb, ns, n_mu = zx["nk"], zx["nb"], zx["ns"], zx["n_mu"]
    rank = rst["rank"]
    r_chunk = rst["r_chunk"]
    qw = np.asarray(q_tile_frac, dtype=np.float64)
    qw = qw - np.round(qw)
    n_rp = rst["n_rp"]
    # THE PRODUCER'S SOLVE OR NOTHING.  ``rst["zeta_solve"]`` is set by
    # ``refit_prepare`` from the ζ file's own fit_provenance; there is no
    # default, because the default this call used to carry (ridged Cholesky)
    # is the defect — a refit that cannot name the fit it is reproducing has
    # no business producing a tile.
    if "zeta_solve" not in rst:
        raise SystemExit(
            "refit_vq: this refit state carries no ``zeta_solve`` — the "
            "(charge_zeta_solve, zeta_rcond, zeta_ridge) the bundle's ζ was "
            "actually fitted with.  ζ' must be solved the way ζ was or it "
            "differs from ζ in the near-null subspace the producer truncated, "
            "and V_Q is quadratic in that difference.  Build the state with "
            "vq_interp.refit_prepare, which reads the triple from the ζ "
            "file's isdf_header/fit_provenance.")
    cq_and_x, psi_m_chunk, z_chunk, solve_zeta = _refit_kernels(
        nk, nb, ns, n_mu, rank, r_chunk, rst["zeta_solve"])
    psi_r = rst["psi_r"].reshape(nk, nb, ns, n_rp)
    if m_leg == "stored":
        kqs = np.array([kq_index_of_frac(zx, k_frac - qw) for k_frac in
                        (zx["k_int"].astype(np.float64)
                         / zx["kgrid"][None, :])])
        psi_m_mu = jnp.asarray(zx["psi"][kqs])
        c_m = None
        psi_m_r = psi_r[jnp.asarray(kqs)]
    else:
        # IMPORTED IN THE BRANCH THAT USES IT.  ``bandstructure.bse_setup``
        # pulls in ``htransform``, whose module body runs
        # ``initialize_communicator_stack()`` and with it the REQUIRED-FFI
        # gate.  The ``"stored"`` leg never calls ``compute_wfns_fi``, so at
        # function scope that import made a leg that needs no htransform
        # refuse on a machine with no built ``liblorrax_ffi_host.so`` — which
        # is every off-cluster bench, and it is what kept the P>1 twin below
        # from being runnable anywhere but a compute node.  Production is
        # unchanged: this branch IS the htransform leg, and ``refit_prepare``
        # has already imported the same stack before any caller gets here.
        from bandstructure.bse_setup import compute_wfns_fi

        # m-leg q list: wrap(k − q) for every coarse k, via htransform
        k_frac = zx["k_int"].astype(np.float64) / zx["kgrid"][None, :]
        qm_list = k_frac - qw[None, :]
        bundle = compute_wfns_fi(
            ctilde=rst["ctilde"], B_at_mu=rst["B_at_mu"],
            enk_sigma=rst["enk_sigma"], kgrid_co=rst["kgrid_co"],
            band_window_fi=(0, nb), mesh_xy=mesh_xy, q_list=qm_list,
            return_coeffs=True)
        psi_m_mu = jnp.asarray(bundle.psi_rmu_Y)      # (nk, nb, ns, n_μ)
        c_m = jnp.asarray(bundle.coeffs_fi)           # (nk, rank, nb)
        psi_m_r = None

    Xf, C = cq_and_x(psi_m_mu, jnp.asarray(zx["psi"]))
    Z_parts = []
    B_full = rst["B_full"].reshape(rank, ns, n_rp)
    psi_r_flat = rst["psi_r"].reshape(nk * nb, ns, n_rp)
    for r0 in range(0, n_rp, r_chunk):
        if m_leg == "stored":
            pm = psi_m_r[:, :, :, r0:r0 + r_chunk]
        else:
            pm = psi_m_chunk(
                c_m,
                B_full[:, :, r0:r0 + r_chunk].reshape(rank, ns * r_chunk))
        Z_parts.append(z_chunk(
            pm,
            psi_r_flat[:, :, r0:r0 + r_chunk].reshape(nk * nb, ns * r_chunk),
            Xf))
    # Z columns are r slots of the PADDED grid; solve then trim the r pad
    # (pad columns are exact zeros — zero ψ ⇒ zero ρ ⇒ zero Z).
    Z = jnp.concatenate(Z_parts, axis=1)              # (n_μ, n_rp)
    zeta = solve_zeta(C, Z)[:, : rst["n_rtot"]]       # (n_μ, n_rtot)
    # periodic-frame ζ̃ → sphere coefficients at qw → V tile, BY THE
    # PRODUCER'S OWN TRANSFORM.  The ζ writer is
    # ``common.wfn_transforms.accumulate_rchunk_to_gflat``, called from
    # ``gw.isdf_fitting`` with ``qvec_frac=q``, and what it computes is
    #
    #     ZG_μ(G) = Σ_r e^{−2πi q·r} ζ_μ(r) e^{−2πi G·r}
    #             = Σ_r e^{−2πi (q+G)·r} ζ_μ(r),
    #
    # i.e. the Bloch factor multiplies ζ ON THE r GRID, before the FFT, so
    # that the transform is taken at q+G and not at G.  ``to_sphere`` and
    # ``recon`` twenty lines up are the same statement in host numpy, which
    # is what makes this the module's one spelling of the frame rather than
    # a second opinion about it.  Both are exact no-ops at q = 0.
    #
    # THIS LINE USED TO CARRY A CENTROID WINDING PHASE INSTEAD —
    # ``e^{−2πi q·s_μ} · FFT_r[ζ_μ](G)``, a per-μ DIAGONAL phase standing in
    # for an r-dependent one.  The substitution is exact only if ζ_μ were a
    # delta at s_μ, and it is not: ζ_μ is a cardinal interpolation function
    # with support across the cell, so the two differ in the G-channel
    # structure and not merely by a phase.  It is why Γ was clean and
    # nothing else was
    # (``tests/known_failures/2026-08-11-two-window-contract-lands-and-the-\
    # sixth-wall-is-finite-q.md`` §5: 4.688e-06 at Γ, 1.11–1.17 at every
    # finite q, with no stored tile reproduced at all).  Measured on
    # ``dp2628n20`` with ``m_leg="stored"`` at four coarse q, ζ' against the
    # STORED ζ on the matched sphere and the tile against the stored tile:
    #
    #   transform                         ζ' relF     tile relF
    #   e^{−2πi q·s_μ}·FFT[ζ]  (was)      1.02–1.10   1.13–1.17
    #   FFT[ζ]  (winding dropped)         1.42–1.76   0.23–0.57
    #   THIS: FFT[e^{−2πi q·r} ζ]         1.4e-06–4.6e-06   2.0e-06–7.9e-06
    #
    # and the winding phase applied ON TOP of this one puts the tile back at
    # 1.27–1.37, which is what says it is spurious rather than merely
    # misplaced.  The magnitudes moved too — |ζ'| alone was 0.20–0.57 wrong —
    # so this was never a phase convention that a sign flip could reach.  The
    # G-sphere (``_sphere_millers`` vs the stored ``gvec``: identical sets,
    # 0 missing either way at every q) and ``v_on_set`` (relF 0.0 against the
    # producer's kernel on the stored sphere) were the other two suspects and
    # are both exonerated by the same leg.  No centroid winding phase is
    # applied anywhere below: ``zx["rmu_frac"]`` is still the right object in
    # the F-scheme, where the same factor is taken OUT of the stored ζ to
    # leave something smooth enough to interpolate — an approximation there
    # ON PURPOSE, and never an identity here.
    GS = _sphere_millers(zx, qw)
    fi = flat_idx(zx, GS)
    zt = zeta_r_to_sphere_q(zx, zeta, qw, fi)
    # The producer's kernel on a bulk deck, this module's slab one otherwise;
    # ``rst["v_on_set"]`` is built once in refit_prepare.  See make_v_on_set.
    v = rst.get("v_on_set", None)
    v = v(qw, GS) if v is not None else v_slab_on_set(zx, qw, GS)
    A = zt * np.sqrt(v)[None, :]
    V = np.conj(A) @ A.T
    log_fn(f"  [refit] q_tile={np.array2string(qw, precision=4)}: "
           f"|G_sphere|={GS.shape[1]}, {_time.time()-t0:.1f}s")
    return V
