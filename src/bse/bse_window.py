"""The BSE band window: which bands are in it, and how its axes are padded.

AUTHORITY RULE — the window a run SOLVED is the window everything downstream
names.  A caller asks for ``n_val``/``n_cond``; this module turns that request
into a real band range, and every number that leaves it is post-clamp and
post-snap.  ``resolve_n_occ`` settles where the gap is, the band-degeneracy
guard may widen the request so that a boundary never cuts a degenerate
multiplet, and ``apply_eqp_and_reslice_bands`` re-asks both questions on the
quasiparticle spectrum when ``--eqp`` moved the energies.  Nothing else in the
BSE is allowed to re-derive a window from an array's shape: the shapes carry
the mesh PAD, not the bands.

That is the second half of what lives here.  The band axes are rounded up to a
mesh multiple, and the pad is not inert — ``H_BSE``'s diagonal is a difference
of band energies, so a zero pad would place spurious transitions BELOW the
absorption onset.  The signed ``PAD_EPS_GUARD_RY`` sentinel, the padding
helper that writes it, and the masks and counts that name the physical block
are all here, together, because they are one convention and drifting them
apart is how the pad becomes visible in a spectrum.

``write_eigenvectors_stream`` is here for the same reason: the window it
declares in ``eigenvectors.h5`` is the resolved one, and the file's ``nv``/
``nc`` header fields have to name real bands that ``dipole.h5`` can be sliced
with.  The writer trims the mesh pad off by COUNT and refuses a declared
window narrower than the one that was solved.
"""
from __future__ import annotations

import os
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from common.band_degeneracy import (DEFAULT_MODE, DEGENERACY_TOL_RY,
                                    check_band_window)
from common.collectives import gather_to_host


def _log0(*a, **k):
    """``print`` on process 0 only.

    The band-window guard emits a four-line block; every rank resolves the
    same window from the same energies, so without this the warning arrives
    64 times at P=64 and reads as 64 different problems.  ``band_degeneracy``
    itself stays pure-numpy and jax-free, which is why the rank filter is
    injected here rather than living in the guard.
    """
    try:
        first = jax.process_index() == 0
    except Exception:
        first = True
    if first:
        print(*a, **k)


# ═══════════════════════════════════════════════════════════════════════
#  The band-pad sentinel — why the ε pad is NOT zero
# ═══════════════════════════════════════════════════════════════════════
# A zero pad is inert for operators LINEAR or BILINEAR in the padded axis
# and a WRONG NUMBER for a diagonalisation.  Both cases are live on the
# BSE (c, v) band axes and they need DIFFERENT fills:
#
#   ψ pad rows -> EXACT ZERO.  Every kernel term (V and W) is bilinear in
#     ψ, so a zero-ψ pad band contributes exactly nothing and, critically,
#     couples the pad block to the physical block with EXACTLY zero
#     off-diagonal.  The pad block decouples; that is what makes the
#     sentinel below safe.
#
#   ε pad entries -> ±PAD_EPS_GUARD_RY.  ``H_BSE``'s diagonal is
#     ΔE(c,v,k) = ε_c(k,c) − ε_v(k,v) (``bse_ring_comm._apply_D_term``,
#     and the same construction in bse_serial / bse_simple /
#     w_omega_chain / bse_preconditioner).  With a ZERO ε pad the pad
#     transitions acquire diagonal energies
#         (c_pad, v_pad) -> 0 − 0        = 0
#         (c_pad, v_real)-> 0 − ε_v      = |ε_v|
#         (c_real, v_pad)-> ε_c − 0      = ε_c
#     and, because ΔE_physical = ε_c − ε_v = ε_c + |ε_v| is LARGER than
#     either of the mixed terms, every one of those spurious transitions
#     sits BELOW the true absorption onset.  An eigensolver asked for the
#     lowest excitons returns them.  Signing the pad ±guard puts every pad
#     transition at ΔE ≳ PAD_EPS_GUARD_RY instead, decoupled and outside
#     any physical window, so the padded modes are dropped BY COUNT
#     (n_cond_pad·n_val_pad − n_cond·n_val per k) and never by value.
#
# Value: 1e3 Ry ≈ 13.6 keV.  This is the constant the exciton-bands driver
# already shipped (it was defined there, and that driver ALSO carried a
# hand-rolled repair of this loader's zero ε_v pad — the repair is now an
# assertion that the loader did it).
#
# FINITE, and that is the point.  The tree has two sentinel families and
# they are chosen on whether the pad SURVIVES the function that writes it:
#
#   * ``psp/dft_operators.py`` uses 1e10 on ``T_diag`` — a preconditioner
#     diagonal and an argsort basis for a selection.  The pad entries do
#     not leave; an absurd value is free.
#   * ``common/wfn_transforms.py:1644-1651`` pads band ENERGIES with
#     ``max(real ε) + 1 Ry`` and its comment says why it is not ∞:
#     "keeps PPM resolvent arithmetic 1/(ω − e + iη) safe under fp
#     warnings".  Those pad energies are KEPT and flow downstream.
#
# The BSE ε pad is the SECOND family, unambiguously: it is stored in the
# bundle as ``data['eps_c']``/``['eps_v']`` and handed to every driver,
# and ``bse_preconditioner`` builds exactly a resolvent from it,
# 1/(ΔE − λ + ε_shift).  So the sentinel must stay finite and in scale.
# 1e3 Ry is seven orders above the widest QP window this code will ever
# see, keeps the float32 KPM leg's ΔE well inside single precision, and
# matching the in-tree BSE value means the loader now AGREES with the one
# BSE driver that had this right rather than moving its numbers.
PAD_EPS_GUARD_RY = 1.0e3


def pad_zone_mask_np(n_cond, n_val, n_cond_pad, n_val_pad, nk, dtype=np.float64):
    """``(1, nc_pad, nv_pad, nk)`` 1.0 on physical transitions, 0.0 on pad.

    THE spelling for "restrict a BSE pair-basis vector to the physical
    block", derived BY COUNT from the logical/padded extents the loader
    puts in the bundle — never by thresholding an energy.  Callers that
    seed a Krylov space with random numbers over the PADDED shape must
    apply this, or they hand the solver a start vector with support in a
    block whose eigenvalues are the ``PAD_EPS_GUARD_RY`` sentinel.

    Vectors built from ψ (dipoles, ``w_omega_chain``'s seed block) already
    have exact zeros there by the bilinearity above and need no mask.
    """
    mask = np.zeros((1, int(n_cond_pad), int(n_val_pad), int(nk)), dtype=dtype)
    mask[:, :int(n_cond), :int(n_val), :] = 1.0
    return mask


def pad_zone_mask(data, dtype=None):
    """:func:`pad_zone_mask_np` read off a loader bundle, as a jax array."""
    nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
    if dtype is None:
        dtype = data["eps_c"].dtype
    return jnp.asarray(pad_zone_mask_np(
        int(data["n_cond"]), int(data["n_val"]),
        int(data["n_cond_pad"]), int(data["n_val_pad"]), nk), dtype=dtype)


def n_pad_transitions(data) -> int:
    """How many pair-basis entries are pad — the COUNT to drop by."""
    nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
    return nk * (int(data["n_cond_pad"]) * int(data["n_val_pad"])
                 - int(data["n_cond"]) * int(data["n_val"]))


def _generate_kpts_grid(nkx: int, nky: int, nkz: int) -> np.ndarray:
    """Monkhorst-Pack style k-point grid in crystal coords [0, 1), C-order.

    Returns ``(nk, 3)``.  Single consumer: ``write_eigenvectors_stream``.
    """
    kpts = []
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                kx = ix / nkx
                ky = iy / nky
                kz = iz / nkz if nkz > 0 else 0.0
                kpts.append([kx, ky, kz])
    return np.array(kpts, dtype=np.float64)


def write_eigenvectors_stream(
    output_file: str,
    eigenvalues: jax.Array,
    eigenvectors: jax.Array,
    n_val: int,
    n_cond: int,
    nkx: int,
    nky: int,
    nkz: int,
    n_write: int,
    use_tda: bool = True,
) -> None:
    # ── WHICH WINDOW DOES THIS FILE DESCRIBE? ─────────────────────────────
    # The LOGICAL, POST-SNAP one.  ``n_val``/``n_cond`` become BGW's ``nv``/
    # ``nc`` header fields, and ``absorption_eigvecs`` slices ``dipole.h5``
    # with exactly those against ``n_occ`` — bands ``[n_occ - nv, n_occ)`` and
    # ``[n_occ, n_occ + nc)``.  So they have to name REAL bands, which means
    # the counts the loader RESOLVED (``data['n_val']`` / ``data['n_cond']``),
    # after ``--band-degeneracy`` widened them at a cut multiplet.
    #
    # Two other numbers get mistaken for them and neither belongs in the file:
    #
    #   * the CLI ``--n-val``/``--n-cond`` REQUEST, which is PRE-snap.  This
    #     was the caller-side defect: ``bse_jax._preview_lanczos`` passed the
    #     request, so on any snapping deck (Si ``--n-cond 4`` snaps to 8) the
    #     dataset was created at the requested ``nc`` and the write of the
    #     real component died with ``TypeError: Can't broadcast``, leaving a
    #     truncated ``eigenvectors.h5`` behind — worse than writing none.
    #
    #   * the mesh-rounded PAD extents ``n_cond_pad``/``n_val_pad``, which is
    #     what the incoming array is actually shaped by.  Those are not bands:
    #     ``pad_zone_mask``'s block is decoupled by construction (ψ pad = 0)
    #     and its amplitudes are exact zero, so writing them would put a
    #     column of zeros in the file under a band label the dipole file has
    #     no matching entry for.  They are dropped BY COUNT here, the same
    #     spelling as ``pad_zone_mask_np`` — never by thresholding a value.
    #
    # Hence: the caller declares the logical counts, and this writer TRIMS the
    # component to them (see ``_to_bgw``).  A writer that instead trusted the
    # array's own shape would silently re-export the pad.
    #
    # AND THE TRIM IS CHECKED, because the two reasons an incoming component is
    # WIDER than the declared window need opposite treatment and cannot be told
    # apart by shape: mesh pad must be dropped, a stale pre-snap count must be
    # REFUSED (the bands it drops are real and carry weight).  What separates
    # them is a value, and it separates them exactly: the pad block is
    # decoupled by construction and its amplitudes are EXACT zero, so a
    # discarded block with any weight in it is not pad.  Trading the old loud
    # ``Can't broadcast`` for a silent truncation would be a worse bug than the
    # one being fixed.
    #
    # ``use_tda`` is written HONESTLY (was hardcoded 1).  TDA eigenvectors arrive
    # as ``(n_write, 1, nc, nv, nk)`` (resonant X only); full-BSE (non-TDA) as
    # ``(n_write, 2, nc, nv, nk)`` = the paired (X, Y) with X^H X - Y^H Y = +1.
    # For non-TDA the resonant X is written to ``eigenvectors`` (so the
    # sum-over-states absorption reads the dominant part) and the coupling Y to a
    # sibling ``eigenvectors_coupling`` dataset (both components persisted).
    # BGW eigenvectors.h5 stores eigenvalues in eV (header text in
    # ``eigenvalues.dat`` says "eig (eV)"; matches BGW's BSE/diag.f90
    # write path).  Our solvers return Ry — convert here so a downstream
    # consumer using BGW conventions reads the right number.
    RYD2EV = 13.6056980659
    # ``gather_to_host``, not ``device_get``: see the ONE-writer note below for
    # why this writer must not assume any particular solver's sharding.  On the
    # replicated Lanczos arrays and on Davidson's already-host eigenvalues the
    # helper degrades to the plain ``device_get`` that was here before.
    eigenvalues = gather_to_host(eigenvalues[:n_write]) * RYD2EV
    # The declared window must FIT inside what the solver actually carried.
    # ``>=`` is the pad (trimmed in ``_to_bgw``); ``<`` is a caller that named
    # more bands than it solved, which h5py would have reported five frames
    # deeper as an unattributable "Can't broadcast".
    n_cond, n_val = int(n_cond), int(n_val)
    _nc_arr, _nv_arr = (int(s) for s in eigenvectors.shape[-3:-1])
    if _nc_arr < n_cond or _nv_arr < n_val:
        raise ValueError(
            f"write_eigenvectors_stream: declared window n_cond={n_cond} "
            f"n_val={n_val} does not fit the eigenvectors' (nc, nv) = "
            f"({_nc_arr}, {_nv_arr}).  Pass the LOADER's resolved counts "
            f"(data['n_cond'] / data['n_val']), not the CLI request — the "
            f"band-degeneracy guard may have widened the window.")
    kpts = _generate_kpts_grid(nkx, nky, nkz)
    nk = kpts.shape[0]
    ns = 1
    nQ = 1
    flavor = 2
    spin_kernel = 3
    bse_hamiltonian_size = ns * nk * n_val * n_cond
    evec_sz = bse_hamiltonian_size

    kpts_fortran = kpts.T.copy()
    exciton_Q_shifts = np.zeros((1, 3), dtype=np.float64)

    def _to_bgw(comp, which):
        # comp: (nc_pad, nv_pad, nk) -> BGW (nk, nc, nv, ns) layout.
        #
        # TRIM FIRST, FLIP SECOND, and the order is load-bearing.  The pad is
        # appended at the TOP of each band axis, while BGW's convention
        # reverses the valence axis (v=0 = highest valence,
        # BSE/input_fi.f90:407) where our internal slice puts v=0 at the
        # deepest valence.  Flipping before the trim would slide the pad
        # columns to the FRONT and then keep them as the "highest valence"
        # bands — zeros written under real band labels, silently.
        comp = np.asarray(comp)
        kept = comp[:n_cond, :n_val, :]
        if comp.shape[0] > n_cond or comp.shape[1] > n_val:
            # Everything outside the kept block, in one number.  The reference
            # is the KEPT block's own scale, so this is a statement about
            # weight and not about units.
            w_kept = float(np.max(np.abs(kept))) if kept.size else 0.0
            w_drop = max(
                float(np.max(np.abs(comp[n_cond:, :, :])))
                if comp.shape[0] > n_cond else 0.0,
                float(np.max(np.abs(comp[:, n_val:, :])))
                if comp.shape[1] > n_val else 0.0)
            if w_drop > 1.0e-10 * max(w_kept, 1.0e-300):
                raise ValueError(
                    f"write_eigenvectors_stream: trimming eigenvector {which} "
                    f"from (nc, nv) = ({comp.shape[0]}, {comp.shape[1]}) down "
                    f"to the declared window ({n_cond}, {n_val}) would DISCARD "
                    f"amplitude {w_drop:.3e} against a kept scale of "
                    f"{w_kept:.3e}.  The mesh pad is exactly zero, so this is "
                    f"not pad: the declared window is narrower than the one "
                    f"that was solved.  Pass the LOADER's resolved counts "
                    f"(data['n_cond'] / data['n_val']) — the band-degeneracy "
                    f"guard widens the CLI request.")
        c = np.transpose(kept, (2, 0, 1))[:, :, ::-1][..., None]
        return c.real, c.imag

    # ── PRE-FLIGHT, before any file exists ────────────────────────────────
    # The declared window is a property of the RUN, not of a state, so state 0
    # settles it for all of them — and settling it here means a wrong window
    # refuses with nothing on disk instead of leaving the truncated
    # ``eigenvectors.h5`` that the pre-fix crash left behind.  Runs on EVERY
    # rank, above the rank-0 gate, for the lockstep reason set out below: the
    # gather is a global collective and the refusal is a function of shapes and
    # values every rank holds identically, so all ranks raise together.
    if n_write > 0:
        _v0 = gather_to_host(eigenvectors[0])
        if use_tda:
            _to_bgw(_v0[0] if np.ndim(_v0) == 4 else _v0, "0 (resonant X)")
        else:
            _to_bgw(_v0[0], "0 (resonant X)")
            _to_bgw(_v0[1], "0 (coupling Y)")

    # ── ONE writer ────────────────────────────────────────────────────────
    # Every rank used to reach the ``h5py.File(output_file, "w")`` below on the
    # same shared path.  MEASURED at P=64 / N_mu=10015 (job 7879470, leg
    # m24x64): the solve finishes and prints its eigenvalues, then the ranks
    # race each other truncating one file and the step exits rc=1 with
    #     OSError: Unable to synchronously create file (file signature not found)
    #     OSError: ... (truncated file: eof = 96, ..., stored_eof = 2048)
    # leaving an eigenvectors.h5 written by whichever rank won.  That is
    # QUALITY_PATTERNS #7 ("P ranks overwrote one output file") in production.
    #
    # WHY ``gather_to_host`` AND NOT ``jax.device_get``.  This comment used to
    # read "``solve_bse_sharded`` pins ``out_shardings=(rep_eig, rep_eig,
    # rep_eig)`` … so eigenvalues/eigenvectors are REPLICATED", and took that as
    # licence to fetch with a bare ``device_get``.  That is true of the Lanczos
    # routes and FALSE of the ``--solver davidson`` route, which returns ``X``
    # on the solve sharding ``P(None,"x","y",None)``; ``--solver davidson
    # --write-eigs`` therefore died on every rank at P>1 with "Fetching value
    # for `jax.Array` that spans non-addressable (non process local) devices".
    # ``bse_lanczos`` now pins the Davidson branch to the same replicated
    # convention (that is the root-cause half of the fix), but a WRITER must not
    # depend on a solver's layout to be correct: any future solver, or any
    # caller that hands this function a solve-sharded array directly, would
    # resurrect the same crash.  ``common.collectives.gather_to_host`` answers
    # "does this process hold all of it?" instead of assuming, and its arms
    # degrade to exactly the ``device_get`` that was here before whenever the
    # array is replicated or single-process — so the 1-GPU path is unchanged.
    #
    # The fetches stay UNGATED on both branches on purpose: ``eigenvectors[i]``
    # slices a GLOBAL array, which dispatches an XLA computation every process
    # must enter, and ``gather_to_host``'s collective arm is chosen on
    # ``is_fully_addressable``/``is_fully_replicated``, both GLOBAL properties —
    # so every rank takes the same arm and the ranks stay in lockstep.  A
    # rank-0-only body would hang the multi-process client rather than fix
    # anything.
    if jax.process_index() != 0:
        for i in range(n_write):
            gather_to_host(eigenvectors[i])
        return

    with h5py.File(output_file, "w") as f:
        f.create_group("mf_header")
        f.create_group("eps_header")
        f.create_group("bse_header")

        exciton_header = f.create_group("exciton_header")
        exciton_header.create_dataset("version", data=1)
        exciton_header.create_dataset("flavor", data=flavor)

        params = exciton_header.create_group("params")
        params.create_dataset("bse_hamiltonian_size", data=bse_hamiltonian_size)
        params.create_dataset("evec_sz", data=evec_sz)
        params.create_dataset("spin_kernel", data=spin_kernel)
        params.create_dataset("nevecs", data=n_write)
        params.create_dataset("ns", data=ns)
        params.create_dataset("nc", data=n_cond)
        params.create_dataset("nv", data=n_val)
        params.create_dataset("use_tda", data=1 if use_tda else 0)

        kpoints = exciton_header.create_group("kpoints")
        kpoints.create_dataset("nk", data=nk)
        kpoints.create_dataset("kpts", data=kpts_fortran)
        kpoints.create_dataset("nQ", data=nQ)
        kpoints.create_dataset("exciton_Q_shifts", data=exciton_Q_shifts.T)

        exciton_data = f.create_group("exciton_data")
        exciton_data.create_dataset("eigenvalues", data=eigenvalues)
        evec_dset = exciton_data.create_dataset(
            "eigenvectors",
            shape=(1, n_write, nk, n_cond, n_val, ns, 2),
            dtype=np.float64,
        )
        coupling_dset = None
        if not use_tda:
            coupling_dset = exciton_data.create_dataset(
                "eigenvectors_coupling",
                shape=(1, n_write, nk, n_cond, n_val, ns, 2),
                dtype=np.float64,
            )

        for i in range(n_write):
            vec = gather_to_host(eigenvectors[i])
            if use_tda:
                # (1, nc, nv, nk) sharded or (nc, nv, nk) unsharded -> resonant X.
                Xc = vec[0] if vec.ndim == 4 else vec
                Yc = None
            else:
                # (2, nc, nv, nk) = paired (X, Y) from the non-TDA solver.
                Xc, Yc = vec[0], vec[1]
            re, im = _to_bgw(Xc, f"{i} (resonant X)")
            evec_dset[0, i, :, :, :, :, 0] = re
            evec_dset[0, i, :, :, :, :, 1] = im
            if Yc is not None:
                re, im = _to_bgw(Yc, f"{i} (coupling Y)")
                coupling_dset[0, i, :, :, :, :, 0] = re
                coupling_dset[0, i, :, :, :, :, 1] = im

    print(f"Wrote {n_write} eigenvectors to {output_file}"
          + ("" if use_tda else " (+ coupling Y)"))


def _pad_axis_to_multiple(x: jax.Array, axis: int, multiple: int,
                          *, fill: float = 0.0) -> tuple[jax.Array, int]:
    """Pad ``axis`` up to a multiple of ``multiple``; return (padded, PADDED extent).

    ``fill`` is the value written into the pad zone and it is NOT a
    detail: ψ must be padded with 0.0 (bilinear ⇒ inert) and ε with
    ±:data:`PAD_EPS_GUARD_RY` (diagonal of a diagonalisation ⇒ a zero pad
    is a wrong number).  See the module-level note on the sentinel.  It is
    keyword-only so that no call site can pick the fill positionally by
    accident — a mis-signed ε guard puts pad transitions BELOW the onset,
    which is the failure this parameter exists to make unspellable.
    """
    size = x.shape[axis]
    pad = (-size) % multiple
    if pad == 0:
        return x, size
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad)
    # Return the PADDED extent (size + pad), not the pre-pad size: every
    # consumer binds this to n_val_pad/n_cond_pad and relies on it being the
    # mesh-rounded value (e.g. bse_ring_comm's `n_cond_pad % px == 0` guard).
    # Previously wrong only when the band count was not already a mesh
    # multiple — invisible on all mesh-divisible validated runs.
    #
    # NOTE the opposite convention to ``runtime.padding.pad_axis_to``,
    # which returns the LOGICAL extent (n_orig) from the same position.
    # Two helpers returning different extents from the same slot is how
    # the bug above happened; do not "unify" them by silently swapping
    # one — the bundle carries BOTH (``n_val``/``n_cond`` logical,
    # ``n_val_pad``/``n_cond_pad`` padded) and consumers must name which
    # they want.  ``tests/test_pad_parity_gates.py`` pins this.
    return jnp.pad(x, pad_width, mode="constant",
                   constant_values=fill), size + pad


def read_bgw_eqp(eqp_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a BerkeleyGW ``eqp1.dat`` file."""

    kpts = []
    e_dft_blocks = []
    e_qp_blocks = []

    with open(eqp_file) as f:
        while True:
            header = f.readline()
            if not header:
                break
            stripped = header.strip()
            if not stripped:
                break
            if stripped.startswith("#"):
                continue
            parts = header.split()
            if len(parts) < 4:
                break
            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            n_bands = int(parts[3])
            kpts.append([kx, ky, kz])

            e_dft_k = []
            e_qp_k = []
            for _ in range(n_bands):
                cols = f.readline().split()
                e_dft_k.append(float(cols[2]))
                e_qp_k.append(float(cols[3]))
            e_dft_blocks.append(e_dft_k)
            e_qp_blocks.append(e_qp_k)

    kpts_ibz = np.array(kpts)
    max_band = max(len(b) for b in e_dft_blocks)
    n_kpts = len(kpts)
    e_dft_ibz = np.full((n_kpts, max_band), np.nan)
    e_qp_ibz = np.full((n_kpts, max_band), np.nan)
    for i in range(n_kpts):
        nb = len(e_dft_blocks[i])
        e_dft_ibz[i, :nb] = e_dft_blocks[i]
        e_qp_ibz[i, :nb] = e_qp_blocks[i]
    return kpts_ibz, e_dft_ibz, e_qp_ibz


def _parse_wfn_path(input_file: str) -> str:
    """Extract ``wfn_file`` from ``cohsex.in`` and resolve relative paths."""

    input_dir = os.path.dirname(os.path.abspath(input_file))
    wfn_file = "WFN.h5"
    with open(input_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            if key.strip() == "wfn_file":
                wfn_file = val.strip()
                break
    if not os.path.isabs(wfn_file):
        wfn_file = os.path.join(input_dir, wfn_file)
    return wfn_file


def resolve_n_occ(
    enk_full: np.ndarray,
    *,
    n_occ: Optional[int] = None,
    input_file: Optional[str] = None,
    fermi_energy: Optional[float] = None,
) -> int:
    """Determine n_occ (count of occupied bands) for BSE band slicing.

    Resolution order:

      1. **Explicit ``n_occ``** — caller knows; return as-is.
      2. **WFN.h5 ``ifmax``** via ``input_file`` (cohsex.in's ``wfn_file``
         entry). Reads ``mf_header/kpoints/ifmax`` directly — authoritative.
      3. **``mean_enk < fermi_energy``** if ``fermi_energy`` is explicitly
         passed (Ry). Caller's responsibility to pass a sane reference.

    Raises ``ValueError`` if none of the above resolves. The previous
    "auto-detect" heuristic (``mean_enk < 0`` or "largest gap") was
    silently broken for systems whose pseudopotential reference puts the
    valence well above zero (most QE setups, e.g. Si): it returned only
    the deepest semicore states. We now require an explicit source.

    Parameters
    ----------
    enk_full : (nk, nb) ndarray (Ry) — DFT eigenvalues per k. Used only
        when ``fermi_energy`` is given as a hint.
    n_occ : int, optional — explicit bypass.
    input_file : str, optional — cohsex.in / nscf.in path. Its
        ``wfn_file`` entry is followed to a WFN.h5 to read ``ifmax``.
    fermi_energy : float, optional (Ry) — explicit Fermi-level hint.
    """
    if n_occ is not None:
        return int(n_occ)

    if input_file is not None:
        try:
            from ffi import _services
            _services.ensure_on_path()
            from wfn_loader import WfnLoader
            wfn_path = _parse_wfn_path(input_file)
            if os.path.exists(wfn_path):
                w = WfnLoader(wfn_path)
                return int(w.nelec)
            else:
                print(f"  [resolve_n_occ] WFN.h5 not found at {wfn_path}; "
                      "trying fermi_energy hint next.")
        except Exception as e:
            print(f"  [resolve_n_occ] WFN.h5 lookup failed "
                  f"({type(e).__name__}: {e}); trying fermi_energy hint next.")

    if fermi_energy is not None:
        mean_enk = np.asarray(np.mean(enk_full, axis=0))
        nb = mean_enk.size
        n_occ_hint = int(np.sum(mean_enk < fermi_energy))
        if 1 <= n_occ_hint <= nb - 1:
            return n_occ_hint

    raise ValueError(
        "Could not determine n_occ. Pass `n_occ=` explicitly, or "
        "`input_file=` pointing to a cohsex.in / nscf.in whose `wfn_file` "
        "resolves to a valid WFN.h5 (where `mf_header/kpoints/ifmax` "
        "gives the count of occupied bands authoritatively)."
    )


def apply_eqp_corrections(
    enk_full: np.ndarray,
    eqp_file: str,
    input_file: Optional[str] = None,
    ry_to_ev: float = 13.6056980659,
) -> np.ndarray:
    """Apply BGW ``eqp1.dat`` corrections to full-BZ DFT eigenvalues."""

    _kpts_ibz, e_dft_ibz, e_qp_ibz = read_bgw_eqp(eqp_file)
    nk_ibz, nb_eqp = e_dft_ibz.shape
    nk_full, nb_full = enk_full.shape
    enk_qp = enk_full.copy()

    if input_file is not None:
        from ffi import _services
        _services.ensure_on_path()
        from wfn_loader import WfnLoader
        from symmetry_maps import SymMaps

        wfn_path = _parse_wfn_path(input_file)
        wfn = WfnLoader(wfn_path)
        sym = SymMaps(wfn)
        assert sym.nk_tot == nk_full
        assert nk_ibz == sym.nk_red

        for ik_full in range(nk_full):
            ik_ibz = sym.irr_idx_k[ik_full]
            for ib in range(min(nb_eqp, nb_full)):
                if not np.isnan(e_qp_ibz[ik_ibz, ib]):
                    enk_qp[ik_full, ib] = e_qp_ibz[ik_ibz, ib] / ry_to_ev
    else:
        enk_full_ev = enk_full * ry_to_ev
        tol_ev = 0.01
        matched = np.zeros(nk_full, dtype=bool)
        for ik_full in range(nk_full):
            best_ibz = -1
            best_err = np.inf
            for ik_ibz in range(nk_ibz):
                n_compare = min(nb_eqp, nb_full)
                mask = ~np.isnan(e_dft_ibz[ik_ibz, :n_compare])
                if not np.any(mask):
                    continue
                err = np.max(
                    np.abs(
                        enk_full_ev[ik_full, :n_compare][mask]
                        - e_dft_ibz[ik_ibz, :n_compare][mask]
                    )
                )
                if err < best_err:
                    best_err = err
                    best_ibz = ik_ibz
            if best_ibz >= 0 and best_err < tol_ev:
                matched[ik_full] = True
                for ib in range(min(nb_eqp, nb_full)):
                    if not np.isnan(e_qp_ibz[best_ibz, ib]):
                        enk_qp[ik_full, ib] = e_qp_ibz[best_ibz, ib] / ry_to_ev

        # ``matched`` used to be written and never read: a k-point whose DFT
        # energies did not match any IBZ block within ``tol_ev`` simply kept
        # its DFT energies, and the run continued as if it were a QP
        # calculation.  That is a silent mean-field/quasiparticle mix -- the
        # exact class of error that makes a cross-code comparison stop being
        # apples-to-apples without anything in the log saying so.  On Si the
        # DFT->QP shift is ~0.6 eV, so a single unmatched k-point moves the
        # excitons it carries by hundreds of meV.  Fail instead.
        if not matched.all():
            bad = np.flatnonzero(~matched)
            raise ValueError(
                f"apply_eqp_corrections: {bad.size} of {nk_full} k-points "
                f"could not be matched to an IBZ block of {eqp_file!r} by "
                f"mean-field energy (tol {tol_ev} eV); first few: "
                f"{bad[:8].tolist()}. Those k-points would silently keep DFT "
                f"energies inside a quasiparticle calculation. Pass "
                f"input_file= so the mapping comes from the symmetry maps "
                f"instead of an energy heuristic, or check that the eqp file "
                f"belongs to this wavefunction.")

    return enk_qp


def apply_eqp_and_reslice_bands(
    restart_file: str,
    eqp_file: str,
    input_file: Optional[str],
    n_val: int,
    n_cond: int,
    n_occ: Optional[int],
    grid_x: int,
    grid_y: int,
    degeneracy_mode: str = DEFAULT_MODE,
    degeneracy_tol_ry: float = DEGENERACY_TOL_RY,
) -> tuple[jax.Array, jax.Array, int]:
    """Apply BGW ``eqp1.dat`` corrections and re-slice the BSE band window.

    Reads ``enk_full`` from ``restart_file``, applies the eqp corrections, then
    RE-resolves n_occ on the CORRECTED energies (QP shifts can move the gap)
    before slicing ``n_val``/``n_cond`` bands around the new n_occ.  The
    valence/conduction energy axes are padded to multiples of (grid_y, grid_x)
    to match the loader's psi band axes.

    ``n_val``/``n_cond`` must be the loader-CLAMPED counts (``data['n_val']`` /
    ``data['n_cond']``), not raw CLI requests, or the slice can run out of
    bounds.  Single-sourced by the sharded --eqp paths (bse_jax._preview_lanczos,
    davidson_absorption, absorption_haydock).

    Returns ``(eps_v_padded, eps_c_padded, n_occ_eff)``.
    """
    with h5py.File(restart_file, "r") as f:
        enk_full_np = np.asarray(f["enk_full"][:])
    enk_full_np = apply_eqp_corrections(enk_full_np, eqp_file, input_file=input_file)
    n_occ_eff = resolve_n_occ(enk_full_np, n_occ=n_occ, input_file=input_file)
    # Degeneracy guard, REPORT-ONLY here by construction.  The loader already
    # snapped the window on the DFT spectrum and ψ has been read at those
    # shapes; this function only re-derives ENERGIES, so widening now would
    # desynchronise eps from psi.  What is new at this seam is that the QP
    # shifts can open or close a near-degeneracy the DFT spectrum did not
    # have, and that is worth a warning even though it cannot be repaired
    # without re-reading ψ with a wider window.
    check_band_window(
        enk_full_np, n_occ_eff - n_val, n_occ_eff + n_cond,
        tol_ry=degeneracy_tol_ry, mode=degeneracy_mode,
        where="apply_eqp_and_reslice_bands (QP-corrected spectrum; re-run "
              "with a wider --n-val/--n-cond to repair)", log=_log0)
    val_idx = np.arange(n_occ_eff - n_val, n_occ_eff)
    cond_idx = np.arange(n_occ_eff, n_occ_eff + n_cond)
    eps_v = jnp.asarray(enk_full_np[:, val_idx])
    eps_c = jnp.asarray(enk_full_np[:, cond_idx])
    # Signed sentinel, as at the loader seam — this REPLACES data['eps_*'],
    # so a zero pad here would re-open the wrong-number path after --eqp.
    eps_v, _ = _pad_axis_to_multiple(eps_v, axis=1, multiple=grid_y,
                                     fill=-PAD_EPS_GUARD_RY)
    eps_c, _ = _pad_axis_to_multiple(eps_c, axis=1, multiple=grid_x,
                                     fill=PAD_EPS_GUARD_RY)
    return eps_v, eps_c, n_occ_eff
