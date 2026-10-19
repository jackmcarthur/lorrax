"""Ansatz-neutral post-processing for the on-shell dynamic Sigma_c.

The pole model owns production of the on-shell body (``ppm_sigma.OnShellSigma``)
and its q->0 head.  Everything after that point -- the request it is evaluated
on, head injection, the reads at DFT energies and the canonical
``sigma_mnk.h5`` write -- is common to every dynamic ansatz and lives here
exactly once.  ``sigma_omega_grid_debug`` keeps the body on the omega grid
instead, a ``(n_omega, nk, nb, nb)`` cube interpolated at the DFT energies
and written as the ``sigma_mnk.h5`` omega cube.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from common.units import RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange


@dataclass(frozen=True)
class OmegaCoverage:
    """WHICH (k, n) the Sigma(omega) grid actually sampled, and what was done.

    Returned beside the at-DFT Sigma_c so that every artifact written from
    it can say which of its cells are measurements and which are endpoint
    values.  Before this existed the two were indistinguishable on disk: the
    exact-origin Na run wrote finite semicore EQP values whose stored
    ``sigC`` is bit-for-shown-digits the ``omega = -5 eV`` endpoint of
    ``sigma_mnk.h5`` rather than Sigma at the state's DFT energy, and the
    only trace was a ``QSGW: 10142 clipped (41.3%)`` line in the log.

    ``mask_kn`` is True where the state WAS sampled.  ``policy`` is the
    fixed value ``"static_omega0"`` (uncovered states use Sigma(omega=0)), so a reader
    knows whether the uncovered cells hold an endpoint value (``clamp``) or
    a non-finite marker (``mask``); ``refuse`` never reaches a consumer.
    ``uncovered_bands`` names them, one ``band N: [lo, hi] eV on k/nk k``
    line per band (``band_partition.coarse_band_report``).
    """

    mask_kn: np.ndarray
    n_uncovered: int
    fraction_uncovered: float
    omega_min_ev: float
    omega_max_ev: float
    policy: str
    uncovered_bands: tuple = ()

    def summary(self) -> str:
        """One line, suitable for an artifact comment or a log."""
        return (f"omega_coverage: {self.n_uncovered} of "
                f"{int(np.asarray(self.mask_kn).size)} "
                f"({100.0 * self.fraction_uncovered:.2f}%) evaluation "
                f"energies outside the sampled grid "
                f"[{self.omega_min_ev:.4f}, {self.omega_max_ev:.4f}] eV; "
                f"out_of_range_policy={self.policy}")


class _OmegaSlabs:
    """``cube[lo:hi]`` along ω for a writer that reads slabs in order, the start carried on device.

    Each read returns ``(slab, lo + size)`` from one program per slab size, so a
    sequential writer (``sigma_output._write_omega_cube``) transfers no slab
    start: the first start is a compiled constant and every later one is the
    previous program's output.  An out-of-order read (a single ω index) places
    its start once per value.  The slabs are the cube's own elements.
    """

    def __init__(self, cube):
        self._cube, self._next, self._start = cube, None, None

    def __call__(self, lo, hi):
        from jax.sharding import NamedSharding, PartitionSpec
        lo, hi = int(lo), int(hi)
        sharding = self._cube.sharding
        replicated = NamedSharding(sharding.mesh, PartitionSpec())
        if lo == 0:
            start = _slab_zero(replicated)()
        elif lo == self._next:
            start = self._start
        else:
            from common.collectives import device_put_process_local_cached
            start = device_put_process_local_cached(np.int32(lo), replicated)
        slab, self._start = _slab_program(sharding, replicated, hi - lo)(self._cube, start)
        self._next = hi
        return slab


@lru_cache(maxsize=None)
def _slab_zero(replicated):
    return jax.jit(lambda: jnp.zeros((), jnp.int32), out_shardings=replicated)


@lru_cache(maxsize=None)
def _slab_program(sharding, replicated, size):
    return jax.jit(lambda cube, start: (jax.lax.dynamic_slice_in_dim(cube, start, size, axis=0),
                                        start + jnp.asarray(size, start.dtype)),
                   out_shardings=(sharding, replicated))


def extract_sigma_diag_logical(
    sigma_c_omega: jax.Array,
    mesh_xy,
    *,
    band_axis=None,
    host: bool = True,
) -> np.ndarray | None:
    """Replicate the Sigma_c(omega) diagonal and restore its logical bands.

    Parameters
    ----------
    sigma_c_omega
        Correlation self-energy with shape
        (n_omega, nk, nb_carrier, nb_carrier).
    mesh_xy
        Process mesh owning the two carrier band faces.
    band_axis : runtime.padding.PaddedAxis or None
        Logical/carrier receipt published by the Sigma producer.

    host
        False on a rank that will not read the diagonal (only the reporting
        rank writes it): the rank enters the collective, as every rank must,
        and makes no host copy.

    Returns
    -------
    np.ndarray or None
        Replicated diagonal with shape (n_omega, nk, nb_logical); None
        with ``host=False``.
    """
    from .qsgw_utils import extract_sigma_diag_replicated
    diagonal = extract_sigma_diag_replicated(sigma_c_omega, mesh_xy)
    if not host:
        return None
    diagonal = np.asarray(diagonal)
    if band_axis is not None:
        from runtime.padding import strip_axis
        diagonal = np.asarray(strip_axis(diagonal, band_axis, axis=-1))
    return diagonal


def add_head_sigma_diag(sigma_c_body, head, *, band_axis=None, donate_body=False):
    """The on-shell body plus the band-diagonal dynamic q->0 head.

    ``head`` is the closed form ``head_correction.HeadSigmaDiag`` (its value at
    every table and its derivative, ``OnShellSigma.with_head``), or ``None``
    when the ansatz has no separate head contribution.  A grid body
    (``sigma_omega_grid_debug``) takes it at every omega
    (:func:`_add_head_sigma_diag_grid`; ``donate_body`` writes it into the
    body's buffer there).
    """
    from .ppm_sigma import is_on_shell
    if not is_on_shell(sigma_c_body):
        return _add_head_sigma_diag_grid(sigma_c_body, head, band_axis=band_axis,
                                         donate_body=donate_body)
    return sigma_c_body.with_head(head, band_axis)


def _add_head_sigma_diag_grid(
    sigma_c_body_omega: jax.Array,
    head_sigma_diag_w_kn_ry: np.ndarray | None,
    *,
    band_axis=None,
    donate_body: bool = False,
) -> jax.Array:
    """Add a band-diagonal dynamic q->0 head to a Sigma_c body cube (the grid).

    Parameters
    ----------
    sigma_c_body_omega
        Body contribution with shape ``(n_omega, nk, nb, nb)`` in Ry.
    head_sigma_diag_w_kn_ry
        Head-only diagonal with shape ``(n_omega, nk, nb)`` in Ry: a host
        array or the closed-form ``head_correction.HeadSigmaDiag``; ``None``
        when the ansatz has no separate head contribution.
    donate_body
        On the band-sharded layout, write the head into the body's buffer
        and delete the body array.  Only a caller that reads the body no
        further may pass True.
    """
    if head_sigma_diag_w_kn_ry is None:
        return sigma_c_body_omega

    from .qsgw_utils import (
        add_band_diag_rows_sharded, add_band_diag_sharded,
        is_band_sharded_sigma_omega)
    # The closed-form head (head_correction.HeadSigmaDiag) on the band-sharded
    # cube: each rank evaluates the rows of its own tile, so the
    # (n_omega, nk, nb) diagonal is on no host.  Every other case takes the
    # array (the replicated layouts hold the whole cube on each device).
    by_rows = (hasattr(head_sigma_diag_w_kn_ry, "band_rows")
               and is_band_sharded_sigma_omega(sigma_c_body_omega))
    head = (head_sigma_diag_w_kn_ry if by_rows
            else np.asarray(head_sigma_diag_w_kn_ry))
    expected_head = tuple(sigma_c_body_omega.shape[:2]) + (
        int(band_axis.logical) if band_axis is not None
        else int(sigma_c_body_omega.shape[2]),)
    if head.shape != expected_head:
        raise ValueError(
            "dynamic Sigma head shape must match the body diagonal: "
            f"head={head.shape}, body={sigma_c_body_omega.shape}")
    if band_axis is not None:
        from runtime.padding import authenticate_axis
        authenticate_axis(
            sigma_c_body_omega, band_axis, axis=-2,
            where="dynamic Sigma body producer")
        authenticate_axis(
            sigma_c_body_omega, band_axis, axis=-1,
            where="dynamic Sigma body producer")
    if by_rows:
        n_w, nk, n_logical = head.shape

        def carrier_rows(start, stop):
            # Carrier rows past the logical bands add exact zeros.
            rows = np.zeros((n_w, nk, stop - start), dtype=np.complex128)
            top = min(stop, n_logical)
            if top > start:
                rows[..., :top - start] = head.band_rows(start, top)
            return rows

        return add_band_diag_rows_sharded(
            sigma_c_body_omega, carrier_rows, donate=donate_body)
    if band_axis is not None:
        # Padded on the host, and only when the carrier is wider: a device
        # pad pulled back by the adder was a second (n_omega, nk, nb) host
        # copy and a device copy per call.
        if int(band_axis.carrier) != int(head.shape[-1]):
            padded = np.zeros(head.shape[:-1] + (int(band_axis.carrier),),
                              dtype=head.dtype)
            padded[..., :head.shape[-1]] = head
            head = padded

    if is_band_sharded_sigma_omega(sigma_c_body_omega):
        return add_band_diag_sharded(
            sigma_c_body_omega, head, donate=donate_body)

    n_w, nk, nb = head.shape
    dense = np.zeros((n_w, nk, nb, nb), dtype=np.complex128)
    idx = np.arange(nb)
    dense[:, :, idx, idx] = head
    return sigma_c_body_omega + jnp.asarray(
        dense, dtype=sigma_c_body_omega.dtype)


def sigma_dft_energies_ev(*, wfn, sym, meta, band_slices) -> np.ndarray:
    """E_DFT ``(nk, nb)`` (eV, absolute) on the Sigma bands: the eqp0 centre."""
    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range,
        band_slices.sigma_range, nspinor=meta.nspinor)
    return np.asarray(enk_dft) * RYD_TO_EV


def sigma_on_shell_request(config, *, e_eval_ev, wfn, sym, meta, band_slices,
                           efermi_ry, print_fn):
    """The on-shell evaluation of one dynamic Sigma call, built and announced once.

    The request (``gw.ppm_sigma.on_shell_request``) holds E_eval
    (``e_eval_ev``, the energies the QSGW ansatz reads, absolute eV) and
    E_DFT, both measured from ``efermi_ry``, the body's omega reference.
    These are the very subtractions :func:`eval_sigma_c_at_dft_energies` and
    the finalize make, so the finalize's reads match the tables bit for bit
    (``qsgw_utils.check_on_shell_reads``).  Its parent rows are the sweep's
    (``gw_init._prepare_parent_wavefunction_plan``).  An SC run with
    ``sigma_out_of_grid = static`` adds the grid-edge census tables
    (``sc_iteration._sc_edge_ambiguity``).
    """
    from .gw_config import QPSolver
    from .ppm_sigma import on_shell_request
    if e_eval_ev is None:
        raise ValueError("an on-shell Sigma needs the energies it is evaluated at (e_qp_ev)")
    edge_window_ev = None
    if (config.qp_solver is QPSolver.SELF_CONSISTENT
            and config.sigma.out_of_grid == "static"):
        from .gw_config import sigma_classification_window_ev
        from .scissor import sc_padded_window_ev
        edge_window_ev = sc_padded_window_ev(*sigma_classification_window_ev(config.sigma))
    efermi_ev = float(efermi_ry) * RYD_TO_EV
    request = on_shell_request(
        e_eval_rel_ev=np.asarray(e_eval_ev, dtype=np.float64) - efermi_ev,
        e_dft_rel_ev=sigma_dft_energies_ev(
            wfn=wfn, sym=sym, meta=meta, band_slices=band_slices) - efermi_ev,
        omega_range_ev=np.asarray(config.omega_grid_ev, dtype=np.float64),
        out_of_grid=config.sigma.out_of_grid,
        irr_idx=np.asarray(sym.irr_idx_k), n_parent=len(wfn.kvecs(k=sym.parent_k_domain)),
        edge_window_ev=edge_window_ev)
    print_fn(f"  Sigma_c on shell: {request.n_table} energy table(s) {request.names}, "
             f"{request.frequencies_ry.size} distinct frequencies")
    return request


def eval_sigma_c_at_dft_energies(
    sigma_c, *,
    config,
    band_slices, wfn, sym, meta,
    print_fn,
    mesh_xy=None,
    band_axis=None,
    efermi_ry=None,
    efermi_provenance=None,
):
    """Read diag(Sigma_c) at every DFT band energy (the on-shell ``dft0`` table).

    A grid body (``sigma_omega_grid_debug``) is interpolated along omega
    instead, on the band mesh ``mesh_xy``.

    ``efermi_ry`` is the reference the Sigma body was built with.
    None keeps the loader's midgap/VBM ``wfn.efermi`` (the insulating PPM
    convention). A metallic caller MUST pass the same fixed-N mu its body
    used: mixing references samples Sigma at energies shifted by
    (mu - efermi_midgap) — measured +2.79 eV on the sodium deck, showing
    up as a spurious ~+2.4 eV near-E_F QP correction.

    ``efermi_provenance`` is what ``gw.efermi.resolve_sigma_efermi_ry``
    said the reference WAS.  Given, it is stamped verbatim.  Omitted, the
    provenance falls back to the ``efermi_ry is None`` proxy below, which
    is right for every caller that has not been routed through the
    resolver yet and WRONG for one that has: the proxy labels any explicit
    reference "fixed-N mu", so an MPA run at ``fermi_reference = midgap``
    would be stamped as a metal's chemical potential.  Pass it.

    Returns ``(sigma_c_at_dft_ev, omega_dft_rel_ev, efermi_dft_ev,
    provenance, coverage)``.  THE PROVENANCE IS RETURNED, not re-derived by
    the caller: this function owns the ``efermi_ry is None`` decision, and
    the writer that stamps the answer into ``sigma_mnk.h5`` must record the
    same one it read with, not a second opinion about it.  The
    :class:`OmegaCoverage` rides beside it for exactly the same reason — the
    at-DFT array cannot say by itself which of its cells were sampled and
    which read Sigma(omega = 0).
    """
    enk_dft_ev = sigma_dft_energies_ev(
        wfn=wfn, sym=sym, meta=meta, band_slices=band_slices)
    vbm_ev = float(wfn.vbm) * RYD_TO_EV
    cbm_ev = float(wfn.cbm) * RYD_TO_EV
    ref_ry = float(wfn.efermi) if efermi_ry is None else float(efermi_ry)
    efermi_dft_ev = ref_ry * RYD_TO_EV
    omega_dft_rel_ev = enk_dft_ev - efermi_dft_ev
    from file_io.sigma_output import (OMEGA_REFERENCE_FIXED_N_MU,
                                      OMEGA_REFERENCE_MIDGAP)
    if efermi_provenance is not None:
        provenance = str(efermi_provenance)
    else:
        provenance = (OMEGA_REFERENCE_MIDGAP if efermi_ry is None
                      else OMEGA_REFERENCE_FIXED_N_MU)
    print_fn(
        f"  omega reference = {efermi_dft_ev:.6f} eV "
        f"({provenance}; VBM={vbm_ev:.6f}, CBM={cbm_ev:.6f})"
    )

    from .band_partition import coarse_band_report
    from .ppm_sigma import is_on_shell
    from .qsgw_utils import check_on_shell_reads, omega_coverage
    grid_ev = np.asarray(config.omega_grid_ev, dtype=np.float64)
    # Uncovered states take Sigma(omega=0), named and reported (owner rule
    # 2026-09-22), as the SC cover rule reads them (qsgw_utils.sigma_eval_omega).
    policy = "static_omega0"
    covered, n_uncovered, frac_uncovered = omega_coverage(
        grid_ev, omega_dft_rel_ev)
    uncovered_bands = tuple(coarse_band_report(
        omega_dft_rel_ev, ~covered, mu_ev=0.0,
        band_offset=band_slices.sigma_range[0]))
    if not is_on_shell(sigma_c):
        # Two omega slots per (k, n) leave the cube; the (n_omega, nk, nb)
        # diagonal is not made on any host (full BZ: 3.7 GB per rank at Fe 20^3).
        from .qsgw_utils import interp_sigma_diag_along_omega
        sigma_c_at_dft_ev = interp_sigma_diag_along_omega(
            sigma_c, mesh_xy, grid_ev, omega_dft_rel_ev,
            band_axis=band_axis, scale=RYD_TO_EV,
            context="Sigma_c at E_DFT (eqp0/eqp1)",
            print_fn=print_fn)
    else:
        # The sweep evaluated Sigma_c at these very energies (an uncovered state
        # at omega = 0, the reader's rule), so it is read, not interpolated.
        check_on_shell_reads(sigma_c.request.table_ev("dft0"),
                             np.where(covered, omega_dft_rel_ev, 0.0),
                             what="Sigma_c at E_DFT")
        if n_uncovered:
            print_fn(f"  omega coverage [Sigma_c at E_DFT (eqp0/eqp1)]: {n_uncovered} of "
                     f"{omega_dft_rel_ev.size} ({100.0 * frac_uncovered:.1f}%) evaluation "
                     f"energies outside [{float(grid_ev[0]):.3f}, {float(grid_ev[-1]):.3f}] "
                     "use Sigma(omega=0).")
        sigma_c_at_dft_ev = sigma_c.diagonal("dft0", band_axis) * RYD_TO_EV

    return (
        sigma_c_at_dft_ev,
        omega_dft_rel_ev,
        efermi_dft_ev,
        provenance,
        OmegaCoverage(mask_kn=covered, n_uncovered=n_uncovered,
                      fraction_uncovered=frac_uncovered,
                      omega_min_ev=float(grid_ev[0]),
                      omega_max_ev=float(grid_ev[-1]),
                      policy=policy, uncovered_bands=uncovered_bands),
    )


def sigma_omega_output_path(config, input_dir: str) -> str:
    """Resolve the canonical dynamic-Sigma output path."""
    out_path = config.paths.sigma_omega_h5_file
    if not os.path.isabs(out_path):
        out_path = os.path.join(input_dir, out_path)
    return out_path


@lru_cache(maxsize=8)
def _ev_tensor_fns(sharding):
    """Σ_total and Σ_c in eV from the Ry pair, as two executables."""
    from file_io.sigma_output import derive_sigma_total

    def total(c_ry, x_ry, h_ev):
        return derive_sigma_total(RYD_TO_EV * c_ry, RYD_TO_EV * x_ry, h_ev)

    return (jax.jit(total, out_shardings=sharding),
            jax.jit(lambda c_ry: RYD_TO_EV * c_ry, out_shardings=sharding))


def write_sigma_omega(
    sigma_c, *,
    sig_x: jax.Array,
    sig_h: jax.Array,
    v_h_scalar: jax.Array | None = None,
    h_transverse: jax.Array | None = None,
    config,
    input_dir: str,
    meta,
    mesh_xy,
    omega_reference_ev,
    omega_reference_provenance,
    eval_energies_rel_ev,
    eval_energies_provenance,
    omega_coverage=None,
    sym=None,
    star_already_selected: bool = False,
    band_extrapolation=None,
    band_axis=None,
    print_fn=None,
) -> str:
    """Write canonical ``sigma_mnk.h5`` for any dynamic Sigma ansatz.

    ``omega_reference_ev`` / ``omega_reference_provenance`` are REQUIRED
    keyword arguments, not defaulted ones: the energies this file stamps
    are relative, and a caller that does not say what it is
    relative to produces the unstamped file audit A2 is about.  Both come
    straight from :func:`eval_sigma_c_at_dft_energies`, which owns the
    choice.

    ``eval_energies_rel_ev`` / ``eval_energies_provenance`` are REQUIRED for
    the same reason, one question further along.  The ω reference says what
    the AXIS is measured from; these say WHERE THIS Σ WAS EVALUATED, which
    is a different fact and was the one the file did not carry.  A from-disk
    reassembly (``eqp_bgw.make_eqp_bgw``) therefore could not distinguish a
    one-shot Sigma (evaluated at E_DFT, so centring eqp1's linearization
    there is correct) from a self-consistent one (evaluated at the converged
    QP spectrum, where centring at E_DFT is a different and wrong
    calculation).  Its docstring says so plainly — "Nothing in the files
    distinguishes the two cases, so this function will not guess" — and the
    guess it then made silently was E_DFT.  ``eval_energies_provenance`` is
    ``"at_e_dft"`` or ``"self_consistent_qp"``, MEASURED by the caller as an
    array comparison rather than inferred from a config key.

    ``omega_coverage`` is the optional :class:`OmegaCoverage` for the at-DFT
    read; given it, the file states how many of its evaluation
    energies were never sampled and what was done about them.

    ``band_extrapolation`` is the optional
    ``{"arrays": {...}, "attrs": {...}}`` the Σ_c band-convergence fit
    produces (``gw.band_extrapolation.spectral_h5_payload``).  None —
    the default and every non-extrapolating run — writes exactly the file
    that was written before the feature existed.
    """
    from file_io import write_sigma_omega_h5
    from .ppm_windows import sigma_regularization_for_config

    if v_h_scalar is None:
        if h_transverse is not None:
            raise ValueError("write_sigma_omega: H_T requires scalar V_H")
        v_h_scalar = sig_h
    if band_axis is not None:
        from runtime.padding import pad_square
        sig_x = pad_square(sig_x, band_axis)
        sig_h = pad_square(sig_h, band_axis)
        v_h_scalar = pad_square(v_h_scalar, band_axis)
        if h_transverse is not None:
            h_transverse = pad_square(h_transverse, band_axis)
    hartree_ev = RYD_TO_EV * sig_h
    if h_transverse is not None:
        # Component-aware files derive their aggregate in output units from
        # the two persisted operands.  This makes the receipt identity exact,
        # rather than relying on floating-point distributivity across a Ry→eV
        # conversion.  Charge-only retains the historical single multiply.
        hartree_ev = (
            RYD_TO_EV * v_h_scalar + RYD_TO_EV * h_transverse)

    out_path = sigma_omega_output_path(config, input_dir)
    # RE-DERIVED, not threaded.  ``resolve_sigma_regularization`` is a pure
    # function of the config the driver already ran against, so calling it
    # here stamps exactly the xi the kernel used -- and a threaded value
    # would be a second thing to keep in step, which is the whole class of
    # defect this seam was consolidated to remove.
    sigma_regularization = sigma_regularization_for_config(config)

    # One derivation of the star tables keeps the producer and consumer of
    # the antiunitary conjugation convention together.
    star = None
    if sym is not None:
        from .kin_ion_io import star_tables
        star = star_tables(sym)

    from .ppm_sigma import is_on_shell
    if not is_on_shell(sigma_c):
        # The omega cube (sigma_omega_grid_debug).
        from .qsgw_utils import is_band_sharded_sigma_omega
        if is_band_sharded_sigma_omega(sigma_c):
            shd = sigma_c.sharding

            # The two eV cubes are made one at a time, when the writer reads
            # them (file_io.sigma_output.DeferredArray): Na 8^3 at 1001 omega
            # held the Ry cube and both eV cubes at once, 3 x 14.12 GiB per rank.
            # With star tables the file keeps k_irr rows, so each eV cube is
            # read as its star-spread ω slice and its retained rows only; a
            # full-BZ file is converted and written ω slab by ω slab.  Either
            # way the values come from the Ry cube's same elements and no full
            # eV cube is made.
            from file_io.sigma_output import DeferredArray
            total_fn, c_fn = _ev_tensor_fns(shd)
            shape = tuple(sigma_c.shape)
            c_ry, x_ry, h_ev = sigma_c, sig_x, hartree_ev

            def _rows(rows):
                rows = jnp.asarray(rows)
                return (jnp.take(c_ry, rows, axis=1), jnp.take(x_ry, rows, axis=0),
                        jnp.take(h_ev, rows, axis=0))

            total_slabs, c_slabs = _OmegaSlabs(c_ry), _OmegaSlabs(c_ry)
            total_ev = DeferredArray(
                shape, lambda: total_fn(c_ry, x_ry, h_ev),
                omega_range=lambda lo, hi: total_fn(total_slabs(lo, hi), x_ry, h_ev),
                take_rows=lambda rows: total_fn(*_rows(rows)))
            sigma_c_ev = DeferredArray(
                shape, lambda: c_fn(c_ry),
                omega_range=lambda lo, hi: c_fn(c_slabs(lo, hi)),
                take_rows=lambda rows: c_fn(_rows(rows)[0]))
            write_sigma_omega_h5(
                out_path, total_ev, omega_ev=config.omega_grid_ev,
                sigma_c_kij_ev=sigma_c_ev,
                sigma_sx_kij_ev=RYD_TO_EV * sig_x,
                hartree_kij_ev=hartree_ev,
                hartree_scalar_kij_ev=(
                    None if h_transverse is None
                    else RYD_TO_EV * v_h_scalar),
                hartree_transverse_kij_ev=(
                    None if h_transverse is None
                    else RYD_TO_EV * h_transverse),
                mesh=mesh_xy, star=star,
                star_already_selected=star_already_selected,
                omega_reference_ev=omega_reference_ev,
                omega_reference_provenance=omega_reference_provenance,
                sigma_regularization=sigma_regularization,
                eval_energies_rel_ev=eval_energies_rel_ev,
                eval_energies_provenance=eval_energies_provenance,
                omega_coverage=omega_coverage,
                band_extrapolation=band_extrapolation,
                band_axis=band_axis,
                print_fn=print_fn,
            )
            return out_path

        write_sigma_omega_h5(
            out_path, None, omega_ev=config.omega_grid_ev,
            sigma_c_kij_ev=RYD_TO_EV * sigma_c,
            sigma_sx_kij_ev=RYD_TO_EV * sig_x,
            hartree_kij_ev=hartree_ev,
            hartree_scalar_kij_ev=(
                None if h_transverse is None
                else RYD_TO_EV * v_h_scalar),
            hartree_transverse_kij_ev=(
                None if h_transverse is None
                else RYD_TO_EV * h_transverse),
            mesh=mesh_xy, star=star,
            star_already_selected=star_already_selected,
            omega_reference_ev=omega_reference_ev,
            omega_reference_provenance=omega_reference_provenance,
            sigma_regularization=sigma_regularization,
            eval_energies_rel_ev=eval_energies_rel_ev,
            eval_energies_provenance=eval_energies_provenance,
            omega_coverage=omega_coverage,
            band_extrapolation=band_extrapolation,
            band_axis=band_axis,
            print_fn=print_fn,
        )
        return out_path

    # The pair in place of an omega cube, and the eqp diagonals the file's
    # reader reassembles from (every rank enters the diagonal read, which is
    # collective).
    request = sigma_c.request
    total_fn, c_fn = _ev_tensor_fns(sigma_c.pair.sharding)
    on_shell = dict(
        names=request.names,
        energies_rel_ev=np.stack([request.table_ev(n) for n in request.names]),
        diag_ev=np.stack([sigma_c.diagonal(n, band_axis) * RYD_TO_EV
                          for n in request.names]),
        dsigma=sigma_c.diagonal("derivative", band_axis))
    write_sigma_omega_h5(
        out_path, total_fn(sigma_c.pair, sig_x, hartree_ev),
        sigma_c_kij_ev=c_fn(sigma_c.pair),
        sigma_sx_kij_ev=RYD_TO_EV * sig_x,
        hartree_kij_ev=hartree_ev,
        hartree_scalar_kij_ev=(
            None if h_transverse is None
            else RYD_TO_EV * v_h_scalar),
        hartree_transverse_kij_ev=(
            None if h_transverse is None
            else RYD_TO_EV * h_transverse),
        mesh=mesh_xy, star=star,
        star_already_selected=star_already_selected,
        omega_reference_ev=omega_reference_ev,
        omega_reference_provenance=omega_reference_provenance,
        sigma_regularization=sigma_regularization,
        eval_energies_rel_ev=eval_energies_rel_ev,
        eval_energies_provenance=eval_energies_provenance,
        omega_coverage=omega_coverage,
        band_extrapolation=band_extrapolation,
        band_axis=band_axis,
        on_shell=on_shell,
        print_fn=print_fn,
    )
    return out_path


__all__ = [
    "OmegaCoverage",
    "add_head_sigma_diag",
    "eval_sigma_c_at_dft_energies",
    "sigma_omega_output_path",
    "write_sigma_omega",
]
