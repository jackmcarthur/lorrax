"""Build charge and current head responses with typed wavefunction transport."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import os
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.shard_map import shard_map
from gw.degen_average import TOL_DEGENERACY_RY


__all__ = [
    "DftVelocityHeadData",
    "InterbandCommutatorHeadData",
    "IterationHeadResponse",
    "IterationHeadSamples",
    "ParallelTransportHeadData",
    "QPVelocity",
    "StaticGaugeHallTransaction",
    "assemble_delta_head_manifold",
    "assemble_head_manifold",
    "build_iteration_head_samples",
    "build_iteration_head_response",
    "build_dft_head_response",
    "covariant_link_derivative",
    "head_s_tensor_sharded",
    "head_wings_sharded",
    "raw_hall_pseudovector_sharded",
    "static_gauge_hall_transaction",
    "static_head_wings_sharded",
    "head_samples_from_s",
    "metal_head_summary",
    "interband_commutator_velocity",
    "finalize_iteration_head_sample",
    "finalize_iteration_head_rows",
    "finalize_iteration_head_samples",
    "iteration_head_sample_terms",
    "load_dft_velocity_head",
    "load_dft_dipole_head",
    "load_interband_commutator_head",
    "load_parallel_transport_head",
    "qp_velocity",
    "reduced_covector_to_cartesian",
    "rotate_velocity_active_to_qp",
    "rotate_velocity_to_qp",
    "report_trs_velocity_parity",
    "trs_velocity_parity_residual",
]

# The band-trace parity residual above which the assembled velocity is
# taken to have an INVERTED time-reversal parity rather than a gauge or
# window artefact.  Calibrated from the failure's own magnitude, not from
# a value that wants to pass: a flipped sign makes
# ``|v(−k) + conj(v(k))| = 2|v|``, i.e. residual 2.0 (the register
# measured exactly ``rel 2.000`` for the same class of error on
# ``dipole_cart``), while a straddled degenerate multiplet at the window
# edge perturbs a trace by the multiplet's own share of it.  1.0 —
# "strictly more than the entire signal" — is the only bar between the two
# that no gauge or truncation artefact can reach.  Anything above the
# ROUNDOFF floor and below this is warned, not refused, because no deck
# has yet measured that floor.
_TRS_VELOCITY_PARITY_BREAK = 1.0
_TRS_VELOCITY_PARITY_FLOOR = 1.0e-6


# Factory results are keyed by the mesh identity and static shape facts.  The
# SC loop calls these functions from Python, so constructing an uncached jit
# in the iteration body would pay a compile for every iteration.
_KERNEL_CACHE: dict[tuple, Callable] = {}


# Bound the only frequency-by-band-pair temporary in the direct wing kernel.
# The full Y/Z outputs are much smaller (three Cartesian rows/columns), and a
# ring step visits every frequency block before circulating its band tile.
# 8 -> 2 (2026-09-21, TaAs 8x8x8 head OOM): the block's (block, nk, nb, nb)
# complex128 weight temporaries are the largest live values of the wing
# kernel once the contraction is bounded (14.4 GB each at block 8, nb 468,
# nk 512; a compiled temp region of 43.9 GiB at the TaAs tile).  Each
# frequency is independent, so the block is a pure memory/throughput dial
# with no effect on values.
_HEAD_WING_FREQUENCY_BLOCK = 2

# The face-layout wing kernel's per-step psi gather: each step gathers one
# block of every rank's mu tile with the rank's own band tile, a
# (nk, ns, block*sqrt(P), nb_full/sqrt(P)) buffer per endpoint, independent of
# how many centroids a rank owns (``head_wing_mu_block`` sizes it from the run
# budget; ``_head_wing_kernel_face``'s docstring has the residency algebra).
#: Share of the stage room the gathered endpoint blocks may take.
_HEAD_WING_ROOM_FRACTION = 0.5
#: Narrowest block: below it the per-step gathers and GEMMs are latency-bound.
_HEAD_WING_MU_MIN = 16


def head_wing_mu_block(*, mu_local, nk, ns, nb_full, n_ends):
    """Centroids per face head-wing step, from the run budget's room.

    One step holds ``n_ends`` gathered endpoint blocks of
    ``16·nk·ns·block·nb_full`` bytes per rank; the widest block whose set fits
    ``_HEAD_WING_ROOM_FRACTION`` of the stage room
    (``common.gpu_utils.device_room_bytes``) wins, at least
    ``_HEAD_WING_MU_MIN`` (or the whole tile when it is narrower) and at most
    the rank's whole mu tile.  Every process enters.
    """
    from common.gpu_utils import device_budget_bytes, device_room_bytes, record_stage_price
    per_mu = 16.0 * int(nk) * int(ns) * int(nb_full) * int(n_ends)
    room = float(device_room_bytes())
    fit = int(_HEAD_WING_ROOM_FRACTION * room // per_mu)
    block = int(min(int(mu_local), max(fit, _HEAD_WING_MU_MIN)))
    record_stage_price(f"head wings, mu block {block}/{int(mu_local)}",
                       device_budget_bytes() - room + block * per_mu)
    return block
# Width three is the incumbent Rydberg velocity.  Width eight has the same
# energy-denominator contract: for a literal long-wave transition derivative
# D^(I,a) = d_q_a M^I|_0 it consumes P^(I,a) = -DeltaE * D^(I,a), flattened
# as (a,I)=(2,4).  It never consumes D itself.  Keeping that distinction at
# this shared boundary prevents a future producer from adding two spurious
# inverse powers of DeltaE.  The width-eight contraction is only the
# first-derivative/first-derivative piece of a generalized CT/TT response;
# second jets, response-weight derivatives and contact terms are assembled by
# the producer, not inferred here.
_HEAD_VERTEX_WIDTHS = (3, 8)


_STATIC_GAUGE_HALL_PRODUCER_ID = (
    "lorrax.static_gauge_hall/full_bz_uniform_gauge_v1")
_STATIC_GAUGE_HALL_TOKEN = object()


@dataclass(frozen=True)
class StaticGaugeHallTransaction:
    """Sealed Hall result from one complete uniform-gauge transaction.

    ``sigma_H`` is the three-component real Hall pseudovector consumed by
    the static response producer.  The fingerprint is copied from the same
    uniform-gauge sweep that supplied ``Gamma_raw``.  A current-only sweep
    authenticates the Hall operator without claiming contact or transfer-jet
    closure; those terms remain an explicit capability decision downstream.

    The large ``Gamma_raw`` band matrix remains sharded over both processor
    axes and is not retained here.  The only replicated product is the
    three-component Hall vector.
    """

    sigma_H: jax.Array
    hamiltonian_config_operator_fingerprint: str
    wfn_fingerprint: str
    band_start: int
    band_stop: int
    nk_tot: int
    producer_id: str
    _producer_token: object

    def __post_init__(self) -> None:
        if self._producer_token is not _STATIC_GAUGE_HALL_TOKEN:
            raise TypeError(
                "StaticGaugeHallTransaction is issued only by "
                "static_gauge_hall_transaction")
        fingerprint = str(
            self.hamiltonian_config_operator_fingerprint).strip()
        if (not fingerprint.startswith("sha256:") or len(fingerprint) != 71
                or any(c not in "0123456789abcdef" for c in fingerprint[7:])):
            raise ValueError(
                "StaticGaugeHallTransaction has an invalid operator hash")
        wfn_sha = str(self.wfn_fingerprint).strip()
        if (len(wfn_sha) != 64
                or any(c not in "0123456789abcdef" for c in wfn_sha)):
            raise ValueError("StaticGaugeHallTransaction has an invalid WFN hash")
        if (int(self.band_start) != 0 or int(self.band_stop) <= 0
                or int(self.nk_tot) <= 0):
            raise ValueError(
                "StaticGaugeHallTransaction requires bands [0,stop) and "
                "nk_tot>0")
        if self.producer_id != _STATIC_GAUGE_HALL_PRODUCER_ID:
            raise ValueError(
                "StaticGaugeHallTransaction has an unknown producer")
        if (tuple(self.sigma_H.shape) != (3,)
                or np.dtype(self.sigma_H.dtype) != np.dtype(np.float64)):
            raise ValueError(
                "StaticGaugeHallTransaction sigma_H must be float64[3]")


def _pad_head_band_manifold(v, e, f, surface, *, mesh: Mesh):
    """Zero-pad a logical head manifold for both processor-grid axes.

    ``nb_logical`` remains the authoritative transition mask in every
    consumer kernel. Padding here is storage only: it makes the two band
    axes legal for ``P('x', 'y')`` without inventing physical states. A
    common multiple is intentional because the wing ring uses one band
    storage extent on both processor axes.

    ``v`` is COMMITTED here to the exact ``P(None, None, 'x', 'y')`` layout
    every caller's kernel declares (``_s_tensor_kernel``,
    ``_head_wing_kernel``, ``_drude_tensor_kernel``).  Every caller builds
    ``v`` via a bare ``jnp.asarray(velocity_cart)`` on a freshly host-read
    dipole array, which JAX places as an UNCOMMITTED, single-device
    ``SingleDeviceSharding`` -- never the mesh at all.  Feeding that
    foreign sharding straight into a ``shard_map``-wrapped ``jax.jit``
    whose OTHER operands (the centroid ψ copies) already carry proper
    ``NamedSharding`` on this same mesh forces GSPMD's auto-reshard
    prologue to reconcile one genuinely off-mesh operand against several
    on-mesh ones -- and on the production MoS2 9x9x1/626-band/mu=5288
    shape (P=16) that reconciliation was measured requesting a single
    81.74 GiB allocation at the ``block_until_ready`` in
    ``build_dft_head_response``, ~10x one full ``(nk,ns,mu,nb)`` ψ copy,
    against a compile-only peak of 5.98 GiB for the SAME kernels when
    every input is ALREADY correctly sharded (2026-08-22 restart-path OOM
    investigation, branch fix/head-fold-streamed-2026-08-22).  This
    reproduces byte-for-byte identically whether ``wfns`` came from a
    fresh zeta fit or a restart load -- the restart loader's ψ contract is
    not at fault; the fresh path only avoids it because the ISDF zeta fit
    upstream OOMs first at production scale, on a different binder, so it
    never reaches this call.  The canonical process-local placement helper
    puts ``v`` on the mesh before it reaches a kernel, so no jit call in this
    module dispatches a foreign sharding or invents a second placement path.
    """
    nb = int(v.shape[-1])
    from runtime.padding import padded_axis
    band_axis = padded_axis(
        nb, mesh, name="QSGW head band carrier",
        specs=((P(None, None, "x", None), 2),
               (P(None, None, None, "y"), 3)))
    nb_padded = band_axis.carrier
    if nb_padded != nb:
        pad = nb_padded - nb
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, pad)))
        e = jnp.pad(e, ((0, 0), (0, pad)))
        f = jnp.pad(f, ((0, 0), (0, pad)))
        surface = jnp.pad(surface, ((0, 0), (0, pad)))
    v = device_put_process_local(
        v, NamedSharding(mesh, P(None, None, "x", "y")))
    return v, e, f, surface


@dataclass(frozen=True)
class ParallelTransportHeadData:
    """Validated, device-resident inputs held across the SC loop."""

    forward_links: jax.Array
    forward_neighbors: np.ndarray
    velocity_dft_cart: jax.Array
    nb_logical: int
    reciprocal_lattice_cart: np.ndarray
    validation: dict[str, float]
    #: ``(n_source, 3, nb_logical)`` link-overlap singular values, descending
    #: along the last axis, host-resident (small: O(nk*nb) real numbers).
    #: Read but NOT consulted by this loader itself -- the D3(a) preflight
    #: in ``sc_iteration.load_head_velocity_source`` reads it from here to
    #: refuse a window edge that cuts a hybridized manifold.  See
    #: ``file_io.parallel_transport.load_link_singular_values``.
    singular_values: np.ndarray
    #: ``(3, nk, nb_storage, nb_storage)`` position operator of the collapsed
    #: (one-point, vacuum) k axes, zero on stencil axes; ``None`` when the k
    #: grid has no collapsed axis.  The connection along a collapsed axis
    #: (``common.parallel_transport.link_stencil_orders``).
    collapsed_position: object = None
    #: Outer band set of ``forward_links`` (and of ``singular_values`` and
    #: ``collapsed_position``), at least ``nb_logical``.  The covariant
    #: derivative of DeltaH runs there and is restricted to the head's
    #: ``nb_logical`` bands; ``velocity_dft_cart`` is the head block.
    #: 0 means the head's own set.
    nb_links: int = 0
    #: ``p`` alone (head block, DFT basis) when the artifact carries it:
    #: the p / V_NL split of the per-map head block (velocity_term_shares).
    velocity_kinetic_cart: object = None
    #: Why the links cannot serve ``D_k DeltaH`` on any map (stencil or
    #: window-hybridization gate); the links are then dropped and every map
    #: runs ``U^dagger v_DFT U`` (:func:`sigma_term_zeroed`).  None: served.
    link_unserved: str | None = None


@dataclass(frozen=True)
class DftVelocityHeadData:
    """The same head inputs minus the finite links.

    ``sc_head_update = dft_velocity`` runs the metallic head chain on the
    exact DFT p-matrix velocity written by
    ``get_dipole_mtxels --parallel-transport`` and NOTHING else from that
    artifact: no links, so no covariant ``DΔH`` correction to the
    velocity, so no dependence on the link/rotation stage.  The velocity is
    still rotated into the current QP basis every iteration by the same
    ``U`` the head carry threads — the approximation is confined to the
    ΔH-induced *change* of the velocity operator, which this mode drops.

    ``forward_links`` is a field, pinned at ``None``, so that every
    consumer can ask one object the same question and branch on the answer
    instead of on the mode string.

    This is the configuration every accepted sodium head number was
    produced in (claims 0180/0181/0189, through
    ``tools/qsgw_head_spectrum.py --dft-velocity-only``).  The covariant
    upgrade is parked on claim 0183.
    """

    velocity_dft_cart: jax.Array
    nb_logical: int
    reciprocal_lattice_cart: np.ndarray
    forward_links: None = None
    forward_neighbors: None = None
    validation: None = None
    #: Set when ``sc_head_update = parallel_transport`` runs on this velocity
    #: because the artifact's links are incomplete: the reason its Sigma
    #: term is zero on every map (:func:`sigma_term_zeroed`).
    link_unserved: str | None = None


def head_storage_extent(mesh: Mesh, nb_head: int) -> int:
    """The head's band carrier: ``nb_head`` padded for ``P(..., 'x', 'y')``.

    The SC map's band ladder (energies, occupations, U) has this carrier,
    so every head operator is stored at it.  The link artifact's carrier
    (``common.parallel_transport.band_storage_extent``, the whole mesh
    product) can be wider: 13 bands are 16 there and 14 here at P4.
    """
    from runtime.padding import padded_axis
    return padded_axis(
        int(nb_head), mesh, name="head band carrier",
        specs=((P(None, None, "x", None), 2),
               (P(None, None, None, "y"), 3))).carrier


def head_band_block(operator, nb_head: int, *, mesh: Mesh, nb_outer: int):
    """The head block ``[..., :nb_head, :nb_head]`` of an outer-set band matrix.

    ``operator`` is ``(..., S_o, S_o)`` at ``P(..., 'x', 'y')`` on the outer
    band set; the result is ``(..., S_h, S_h)`` at the same spec, with
    ``S_h = head_storage_extent(mesh, nb_head)`` and the rows and columns
    past ``nb_head`` exactly zero (the head's padding).  Identity when the
    two sets and their carriers agree.
    """
    storage = head_storage_extent(mesh, int(nb_head))
    if int(nb_outer) == int(nb_head) and int(operator.shape[-1]) == storage:
        return operator
    lead = (None,) * (operator.ndim - 2)
    key = ("head_band_block", id(mesh), int(nb_head), tuple(operator.shape))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        keep = np.arange(storage) < int(nb_head)
        mask = np.asarray(keep[:, None] & keep[None, :])

        def block(x):
            return jnp.where(mask, x[..., :storage, :storage],
                             jnp.zeros((), x.dtype))
        kernel = jax.jit(block, out_shardings=NamedSharding(
            mesh, P(*lead, "x", "y")))
        _KERNEL_CACHE[key] = kernel
    return kernel(operator)


def _ascii_stamp(io, path: str, name: str) -> str:
    """Read the byte-valued int32 provenance stamp through SlabIO."""
    raw = np.asarray(io.read_small(name, dtype=np.int32), dtype=np.int32)
    if raw.ndim != 1 or np.any(raw < 0) or np.any(raw > 255):
        raise ValueError(
            f"{path}: {name} is not a 1-D byte-valued stamp dataset; "
            "regenerate with get_dipole_mtxels")
    try:
        return bytes(raw.astype(np.uint8).tolist()).decode("ascii")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            f"{path}: {name} is not an ASCII SHA-256 stamp; regenerate with "
            "get_dipole_mtxels"
        ) from exc


def parallel_transport_link_state(path: str, *, mesh: Mesh):
    """``(links_complete, singular_values)`` of a link artifact, read cheaply.

    Complete means ``connection_complete`` and ``velocity_validation_complete``
    are both 1 (a velocity-only or interrupted artifact is not).
    ``singular_values`` is ``load_link_singular_values`` on the artifact's
    outer band set, or None when the links are incomplete.  Nothing
    O(nk*nb^2) is read.
    """
    from file_io.parallel_transport import load_link_singular_values
    from file_io.slab_io import SlabIO

    with SlabIO(path, mode="r", mesh=mesh) as io:
        flags = [int(io.read_small(name, dtype=np.int64)) for name in (
            "connection_complete", "velocity_validation_complete")]
        if flags != [1, 1]:
            return False, None
        nb_outer = int(io.read_small("band_stop", dtype=np.int64))
        return True, load_link_singular_values(io, nb_logical=nb_outer)


def load_parallel_transport_head(
    path: str,
    *,
    mesh: Mesh,
    sym,
    wfn,
    meta,
) -> ParallelTransportHeadData:
    """Load and validate the preprocessing artifact without mixed ownership.

    Every cheap provenance/refusal is checked before either O(nk*nb^2)
    dataset is read.  The stored links are manifold-dependent, so a
    strict subset or superset is rejected rather than sliced.

    THE METADATA ARE RANK-0 / SMALL DATASETS, and they are read through
    ``SlabIO.read_small`` — the same HDF5 library instance that reads the
    payloads, in the SAME read-only handle.  Two earlier spellings are
    worth naming because both were defects:

    * ``read_slab(name, shape=())`` — refused before a byte moved ("slab
      shape must be non-empty"): a scalar dataspace has no hyperslab, so
      this loader died on the FIRST scalar and could never reach its own
      refusal list, whatever the artifact contained;
    * a short-lived serial-h5py owner opened and closed ahead of SlabIO —
      correct about ORDERING and still a second HDF5 library instance on a
      file the FFI wrote, which is the cohabitation class audit A1 exists
      to retire (``docs/architecture/slab_io.md#one-owner``).

    One handle, one library, one open.
    """
    from file_io.parallel_transport import (
        COLLAPSED_POSITION_DATASET,
        SCHEMA_VERSION,
        VELOCITY_DFT_DATASET,
        load_full_bz_links,
        load_link_singular_values,
        stored_link_steps,
    )
    from file_io.slab_io import SlabIO
    from common.parallel_transport import (
        band_storage_extent, collapsed_axes, link_stencil,
        link_stencil_orders, wfn_fingerprint)

    int_names = (
        "schema_version",
        "connection_complete",
        "velocity_validation_complete",
        "velocity_validation_passed",
        "band_start",
        "band_stop",
        "effective_nspinor",
        "bispinor",
    )
    validation_names = (
        "atol",
        "rtol",
        "max_abs",
        "max_rel",
        "max_abs_diagonal",
        "max_abs_offdiagonal",
        "transition_relative_l2",
        "transition_overlap_real",
        "transition_overlap_imag",
        "head_response_relative_frobenius",
        "head_response_trace_ratio",
    )
    with SlabIO(path, mode="r", mesh=mesh) as io:
        ints = {name: int(io.read_small(name, dtype=np.int64))
                for name in int_names}
        kgrid = np.asarray(io.read_small("kgrid", dtype=np.int32),
                           dtype=np.int32)
        reciprocal = np.asarray(
            io.read_small("reciprocal_lattice_cart", dtype=np.float64),
            dtype=np.float64,
        )
        fingerprint = _ascii_stamp(
            io, path, "wfn_fingerprint_utf8")

        expected_nb = int(meta.b_id_4_user)
        expected_kgrid = np.asarray(wfn.kgrid, dtype=np.int32)
        expected_reciprocal = (
            np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat))
        refusals = []
        if ints["schema_version"] != int(SCHEMA_VERSION):
            refusals.append(
                f"schema_version={ints['schema_version']}, "
                f"expected {int(SCHEMA_VERSION)} (schema 4 stores links on "
                "the point-group-closed shell of common.parallel_transport."
                "link_stencil; fix: rerun the dipole step, get_dipole_mtxels, "
                "to rebuild the links; sc_head_update = dft_velocity still "
                "reads the old file)"
            )
        if ints["connection_complete"] != 1:
            refusals.append("connection_complete is not 1")
        # The reconstruction error is measured, not judged, here: the head
        # judges it on its Sigma correction (link_correction_bound).
        if ints["velocity_validation_complete"] != 1:
            refusals.append(
                "mandatory finite-link DFT head validation is not complete"
            )
        # The links may run on an outer band set that contains the head's
        # (``get_dipole_mtxels --parallel-transport-bands``): the head reads
        # its block and the velocity gate must have judged at least it.
        try:
            judged = int(io.read_small("velocity_validation_band_stop",
                                       dtype=np.int64))
        except (KeyError, RuntimeError, OSError, ValueError):
            judged = ints["band_stop"]
        if ints["band_start"] != 0 or ints["band_stop"] < expected_nb:
            refusals.append(
                f"band manifold [{ints['band_start']},{ints['band_stop']}) "
                f"does not contain the head manifold [0,{expected_nb})"
            )
        elif judged < expected_nb:
            refusals.append(
                f"the velocity gate judged bands [0,{judged}) only; the head "
                f"needs [0,{expected_nb}) (regenerate with this deck)"
            )
        if ints["effective_nspinor"] != int(meta.nspinor):
            refusals.append(
                f"effective_nspinor={ints['effective_nspinor']} != current "
                f"{int(meta.nspinor)}"
            )
        if bool(ints["bispinor"]) != bool(int(meta.nspinor) == 4):
            refusals.append("bispinor convention differs from current run")
        if not np.array_equal(kgrid, expected_kgrid):
            refusals.append(
                f"kgrid={tuple(kgrid)} != current {tuple(expected_kgrid)}")
        if not np.allclose(reciprocal, expected_reciprocal,
                           rtol=0.0, atol=1.0e-13):
            refusals.append(
                "Cartesian reciprocal lattice differs from the current WFN")
        expected_fingerprint = wfn_fingerprint(wfn)
        if fingerprint != expected_fingerprint:
            refusals.append(
                "WFN fingerprint differs (parallel-transport data are stale "
                "or were generated from another DFT solution)"
            )
        # REFUSE BEFORE THE O(nk*nb^2) READS, still inside the handle.  Every
        # operand above is replicated (a stamp, or this run's own config), so
        # this raises on every rank or on none — and each rank's ``__exit__``
        # then closes the collective handle, which is the ordering
        # ``SlabIO.close`` requires.
        #
        # ALSO before the ``velocity_validation_*`` float read, deliberately:
        # those 11 datasets (``atol``, ``rtol``, ``max_abs``, ...) are only
        # written by ``complete_velocity_validation``, at the END of
        # ``write_parallel_transport_artifact`` — never by
        # ``initialize_parallel_transport_artifact``.  A velocity-only
        # artifact (D2, ``--parallel-transport-velocity-only``) therefore
        # NEVER has them, only ``velocity_validation_complete/passed = 0``
        # (its unconditional init-time stamp).  Reading them before this
        # refusal check crashed with a bare ``KeyError: "...doesn't exist"``
        # on exactly that artifact class instead of the named refusal above
        # (audit finding, 2026-08-23: reproduced live against a real
        # velocity-only artifact through this exact loader,
        # ``runs/Na/02_soc48b_qsgw_mpa/09_dft_velocity_headgate_p16_20260823/
        # veloc_build/parallel_transport_velocity_only.h5``) — the
        # "links-requiring consumer reading a velocity-only artifact must
        # refuse, not crash" contract this artifact-schema split exists to
        # keep.  ``velocity_validation_complete != 1`` is already one of the
        # ``refusals`` above, so by the time this line is reached the floats
        # are guaranteed present.
        if refusals:
            raise ValueError(
                f"{path}: refusing QSGW parallel-transport head:\n  - "
                + "\n  - ".join(refusals)
            )
        validation = {
            key: float(io.read_small(f"velocity_validation_{key}",
                                     dtype=np.float64))
            for key in validation_names
        }
        # The link error of link_correction_bound: the relative L2 on the
        # head's elements (head_velocity_set).  An artifact that predates it
        # carries only the transition (off-diagonal) relative L2.
        try:
            validation["link_relative_error"] = float(io.read_small(
                "velocity_validation_head_set_relative_l2", dtype=np.float64))
        except (KeyError, RuntimeError, OSError, ValueError):
            validation["link_relative_error"] = validation[
                "transition_relative_l2"]

        spec = P(None, None, "x", "y")
        nb_outer = int(ints["band_stop"])
        nb_storage = head_storage_extent(mesh, expected_nb)
        outer_storage = band_storage_extent(mesh, nb_outer)
        large_shape = (3, int(meta.nk_tot), outer_storage, outer_storage)
        # The stored link steps must be this lattice's link_stencil; an
        # artifact from before the shell (three reduced axes only, schema
        # 3) is refused above by its schema.
        stored_steps = stored_link_steps(io)
        want_steps = link_stencil(
            tuple(int(n) for n in expected_kgrid), expected_reciprocal).steps
        if not np.array_equal(stored_steps, want_steps):
            raise ValueError(
                f"GATE pt_link_stencil: {path}: stored link steps "
                f"{stored_steps.tolist()} != link_stencil "
                f"{want_steps.tolist()}; regenerate the artifact with "
                "get_dipole_mtxels")
        forward_neighbors = np.asarray(io.read_slab(
            "full_forward_neighbors",
            shape=(int(meta.nk_tot), int(want_steps.shape[0])),
            partition_spec=P(None, None), as_numpy=True), dtype=np.int64)
        links = load_full_bz_links(
            io, mesh=mesh, nk=int(meta.nk_tot), nb_storage=outer_storage,
            nb_logical=nb_outer,
        )
        velocity = head_band_block(io.read_slab(
            VELOCITY_DFT_DATASET, shape=large_shape, partition_spec=spec
        ), expected_nb, mesh=mesh, nb_outer=nb_outer)
        try:
            from file_io.parallel_transport import VELOCITY_KINETIC_DATASET
            velocity_kinetic = head_band_block(io.read_slab(
                VELOCITY_KINETIC_DATASET, shape=large_shape,
                partition_spec=spec), expected_nb, mesh=mesh,
                nb_outer=nb_outer)
        except (KeyError, RuntimeError, OSError, ValueError):
            velocity_kinetic = None       # an artifact that predates it
        # Small (O(nk*nb) real) host-resident diagnostic, read in the SAME
        # handle as everything above -- one owner, one open, per this
        # loader's own docstring.  Not consulted here; the D3(a) window
        # preflight in ``sc_iteration.load_head_velocity_source`` reads it
        # off the returned object.
        singular_values = load_link_singular_values(io, nb_logical=nb_outer)
        collapsed_position = None
        if collapsed_axes(expected_kgrid):
            # A collapsed axis has no link stencil; its connection is the
            # stored position operator.  An artifact that predates it has
            # neither the stamp nor the dataset and refuses here by name
            # instead of reading a short slab.  The reader's own words are
            # kept so a transport fault is not mistaken for an old file.
            try:
                stored_orders = np.asarray(io.read_small(
                    "link_stencil_orders", dtype=np.int32)).reshape(3)
            except (KeyError, RuntimeError, OSError, ValueError) as exc:
                raise ValueError(
                    f"GATE pt_collapsed_axis_artifact: {path}: the k grid "
                    f"{tuple(int(n) for n in expected_kgrid)} has a collapsed "
                    "axis but link_stencil_orders could not be read "
                    f"({type(exc).__name__}: {exc}).  An artifact that "
                    "predates the collapsed-axis position operator has no "
                    "such stamp; regenerate it with get_dipole_mtxels "
                    "--parallel-transport-out") from exc
            want_orders = link_stencil_orders(expected_kgrid)
            if tuple(int(o) for o in stored_orders) != tuple(want_orders):
                raise ValueError(
                    f"GATE pt_collapsed_axis_artifact: {path}: stored "
                    f"link_stencil_orders "
                    f"{tuple(int(o) for o in stored_orders)} != "
                    f"{want_orders} for kgrid "
                    f"{tuple(int(n) for n in expected_kgrid)}")
            collapsed_position = io.read_slab(
                COLLAPSED_POSITION_DATASET, shape=large_shape,
                partition_spec=spec)

    expected_prefix = (3, int(meta.nk_tot))
    nd = int(forward_neighbors.shape[1])
    if (
        tuple(links.shape[:2]) != (nd, int(meta.nk_tot))
        or tuple(velocity.shape) != expected_prefix + (nb_storage, nb_storage)
        or links.shape[-2] != links.shape[-1]
        or int(links.shape[-1]) < nb_outer
    ):
        raise ValueError(
            f"{path}: large PT dataset shapes are inconsistent: "
            f"links={links.shape}, v={velocity.shape}, neighbors="
            f"{forward_neighbors.shape}, expected prefix "
            f"{expected_prefix} and at least {expected_nb} bands."
        )
    # TIME-REVERSAL PARITY, MEASURED HERE AND NOT ASSUMED DOWNSTREAM.  The
    # head lane differentiates this velocity and adds ``d_k Sigma`` and
    # ``-i[A, Sigma]`` to it, all three terms carrying the SAME odd parity
    # (module docstring, eq. 2) — so a sign error anywhere in that sum is
    # visible in exactly this statistic and invisible in every other gate
    # the artifact carries.  The verdict comes from the run's canonical
    # SymMaps rather than the artifact because time reversal is a property
    # of the DFT solution, and the fingerprint check above has already
    # established that these are the same solution.
    if not hasattr(sym, "trs_allowed"):
        raise ValueError(
            "GATE qsgw_head_needs_measured_trs: load_parallel_transport_"
            "head requires SymMaps.trs_allowed; the supplied symmetry "
            "object has no verdict.")
    trs_measured = bool(sym.trs_allowed)
    report_trs_velocity_parity(
        f"{path}: v^DFT", trs_velocity_parity_residual(
            velocity[..., :expected_nb, :expected_nb],
            kgrid=tuple(int(n) for n in expected_kgrid),
            trs_measured=trs_measured),
        trs_measured=trs_measured)
    return ParallelTransportHeadData(
        forward_links=links,
        forward_neighbors=forward_neighbors,
        velocity_dft_cart=velocity,
        nb_logical=expected_nb,
        reciprocal_lattice_cart=reciprocal,
        validation=validation,
        singular_values=singular_values,
        collapsed_position=collapsed_position,
        nb_links=nb_outer,
        velocity_kinetic_cart=velocity_kinetic,
    )


def load_dft_velocity_head(
    path: str,
    *,
    mesh: Mesh,
    wfn,
    meta,
    config=None,
) -> DftVelocityHeadData:
    """Load the completed exact-DFT velocity stage, and only that stage.

    This is the loader ``tools/qsgw_head_spectrum.py --dft-velocity-only``
    has always used — it lived in that tool until ``sc_head_update =
    dft_velocity`` gave the driver the same route, and it moved here rather
    than being copied so the two cannot drift.

    The key difference from :func:`load_parallel_transport_head` is
    deliberate:

    * ``connection_complete`` / ``velocity_validation_*`` are NOT required.
      The velocity is written and checked by the dipole job on its own; the
      link and velocity-validation stages exist to serve the
      covariant correction this mode does not take.
    Every other provenance refusal the PT loader emits is kept verbatim:
    schema, band manifold, k grid, reciprocal lattice, WFN fingerprint.

    Like the PT loader, the stamps come through ``SlabIO.read_small`` in
    the same read-only handle as the payload — one HDF5 library instance
    per file (``docs/architecture/slab_io.md#one-owner``).
    """
    from common.parallel_transport import band_storage_extent, wfn_fingerprint
    from file_io.parallel_transport import (
        SCHEMA_VERSION,
        VELOCITY_DFT_DATASET,
    )
    from file_io.slab_io import SlabIO

    nb = int(meta.b_id_4_user)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        schema = int(io.read_small("schema_version", dtype=np.int64))
        band_start = int(io.read_small("band_start", dtype=np.int64))
        band_stop = int(io.read_small("band_stop", dtype=np.int64))
        kgrid = np.asarray(io.read_small("kgrid", dtype=np.int32),
                           dtype=np.int32)
        reciprocal = np.asarray(
            io.read_small("reciprocal_lattice_cart", dtype=np.float64),
            dtype=np.float64,
        )
        fingerprint = _ascii_stamp(io, path, "wfn_fingerprint_utf8")
        # DFT+U stamp; an artifact written before V_U has none and is the
        # 'none' operator.  Checked only when the caller names its deck.
        hubbard_got = hubbard_want = None
        if config is not None:
            from file_io.parallel_transport import HUBBARD_PROVENANCE_ATTR
            hubbard_want = expected_hubbard_stamp(
                config, wfn=wfn, fallback_dir=os.path.dirname(os.path.abspath(path)),
                caller="sc_head_update = dft_velocity")
            try:
                hubbard_got = _ascii_stamp(io, path, HUBBARD_PROVENANCE_ATTR)
            except Exception:                      # absent: pre-V_U artifact
                hubbard_got = "none"
        expected_reciprocal = (
            np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat)
        )
        refusals = []
        # Schemas 3 and 4 change only the link-consumer contract (4: the
        # link_stencil shell).  The DFT velocity payload and all provenance
        # fields are schema-2 compatible, and this mode consumes no links.
        if schema not in (2, 3, int(SCHEMA_VERSION)):
            refusals.append(
                f"schema_version={schema}, expected 2, 3 or {SCHEMA_VERSION}")
        if band_start != 0 or band_stop < nb:
            refusals.append(
                f"band manifold [{band_start},{band_stop}) does not contain "
                f"[0,{nb})"
            )
        if not np.array_equal(kgrid, np.asarray(wfn.kgrid, dtype=np.int32)):
            refusals.append("k grid differs from the current WFN")
        if not np.allclose(
            reciprocal, expected_reciprocal, rtol=0.0, atol=1.0e-13
        ):
            refusals.append("reciprocal lattice differs from the current WFN")
        if fingerprint != wfn_fingerprint(wfn):
            refusals.append(
                "WFN fingerprint differs from the velocity artifact")
        if hubbard_want is not None and hubbard_got != hubbard_want:
            refusals.append(
                "DFT+U stamp differs (i[r,V_U] present/absent or different "
                f"U/J/B/occupations): artifact={hubbard_got!r} deck={hubbard_want!r}")
        # Rank-invariant operands, so this refuses everywhere or nowhere —
        # before the (3, nk, nb, nb) read, still inside the handle.
        if refusals:
            raise ValueError(
                f"{path}: refusing DFT velocity stage:\n  - "
                + "\n  - ".join(refusals)
            )
        outer_storage = band_storage_extent(mesh, band_stop)
        velocity = head_band_block(io.read_slab(
            VELOCITY_DFT_DATASET,
            shape=(3, int(meta.nk_tot), outer_storage, outer_storage),
            partition_spec=P(None, None, "x", "y"),
        ), nb, mesh=mesh, nb_outer=band_stop)
    return DftVelocityHeadData(
        velocity_dft_cart=velocity,
        nb_logical=nb,
        reciprocal_lattice_cart=reciprocal,
    )


def load_dft_dipole_head(input_dir, *, mesh: Mesh, wfn, meta, config):
    """Use the authenticated charge dipole for a direct-only metallic head."""
    import os

    if int(meta.b_id_0) != 0:
        raise ValueError("metal direct head requires a band manifold starting at 0")
    velocity = read_authenticated_dipole_velocity(
        os.path.join(input_dir, "dipole.h5"), wfn=wfn, meta=meta,
        config=config, mesh=mesh)
    nb = int(meta.b_id_4_chi_user)
    pad = head_storage_extent(mesh, nb) - nb
    velocity = np.pad(velocity, ((0, 0), (0, 0), (0, pad), (0, pad)))
    return DftVelocityHeadData(
        velocity_dft_cart=device_put_process_local(
            velocity, NamedSharding(mesh, P(None, None, "x", "y"))),
        nb_logical=nb,
        reciprocal_lattice_cart=np.asarray(wfn.bvec) * float(wfn.blat),
    )


def _mesh_xy(mesh: Mesh) -> tuple[str, str]:
    names = tuple(str(a) for a in mesh.axis_names)
    if names != ("x", "y"):
        raise ValueError(
            "QSGW parallel-transport head requires the production ('x','y') "
            f"band mesh, got axes {names!r}."
        )
    return names[0], names[1]


def _signed_fft_rows(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Integer real-space rows in the flat-k service's C ordering."""
    axes = [np.fft.fftfreq(int(n), d=1.0 / int(n)) for n in kgrid]
    rr = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return np.asarray(rr.reshape(-1, 3), dtype=np.float64)


def _cartesian_fft_multipliers(
    kgrid: tuple[int, int, int],
    bvec_cart: np.ndarray,
) -> np.ndarray:
    """Return ``2*pi * d(kappa)/d(k_cart) * R`` as ``(3,nk)``."""
    B = np.asarray(bvec_cart, dtype=np.float64)
    if B.shape != (3, 3):
        raise ValueError(f"bvec_cart must have shape (3,3), got {B.shape}.")
    if abs(float(np.linalg.det(B))) < 1.0e-14:
        raise ValueError(
            "bvec_cart is singular; Cartesian k derivatives are undefined."
        )
    # k_cart_j = sum_i kappa_i B_ij, hence
    # d/dk_cart_j = sum_i (B^-1)_ji d/dkappa_i.
    return 2.0 * np.pi * (np.linalg.inv(B) @ _signed_fft_rows(kgrid).T)


def reduced_covector_to_cartesian(covector_reduced, bvec_cart):
    """Convert a reduced-k covector using LORRAX's row-vector B convention.

    ``k_cart = kappa @ B`` because WFN reciprocal vectors are rows.  Thus
    ``D_cart[j] = sum_i (B^-1)[j,i] D_kappa[i]``.  This is the row-basis
    spelling of the conventional ``B_column^-T`` rule.
    """
    B = np.asarray(bvec_cart, dtype=np.float64)
    if B.shape != (3, 3) or abs(float(np.linalg.det(B))) < 1.0e-14:
        raise ValueError(
            f"bvec_cart must be a nonsingular (3,3) matrix, got {B.shape}."
        )
    A = jnp.asarray(covector_reduced)
    if A.ndim < 1 or int(A.shape[0]) != 3:
        raise ValueError(
            "reduced covector must have a leading Cartesian-component "
            f"axis of extent 3, got {A.shape}."
        )
    return jnp.einsum("ij,j...->i...", np.linalg.inv(B), A, optimize=True)


def _spectral_kernel(mesh: Mesh, kgrid: tuple[int, int, int]) -> Callable:
    """Cached FFT-based Cartesian derivative kernel.

    RETAINED WITHOUT A PRODUCTION CALLER (2026-08-23 retirement sweep,
    D4): this and :func:`_cartesian_fft_multipliers` used to back the
    public ``spectral_cartesian_derivative``/``covariant_cartesian_
    derivative`` pair, both retired below (dead: zero production callers,
    and the Si velocity expeditions measured the split construction they
    implemented -- a separately-FFT-differentiated operator plus a
    finite-link commutator -- as producing a correction with ~0 overlap
    to the true one on real SOC data; see the module docstring's
    ``v_Q = v_DFT + D_link(...)`` note and
    ``reports/metal_head_pt_pipelines_2026-08-23/PLAN.md``).  These two
    stayed: ``tests/multi_device/parallel_transport_profile.py`` imports
    them directly (bypassing the now-deleted public wrapper) for its own
    HLO-rematerialization check.  That test module was ALREADY broken
    before this sweep touched anything -- it also imports
    ``covariant_structured_delta``/``_structured_delta_kernel``, which do
    not exist anywhere in this file and have not since before this
    session (grep-verified); registered separately in
    KNOWN_LORRAX_ISSUES.md rather than repaired here, since untangling it
    means editing a P=4 real-distributed-service gate this sweep cannot
    execute to verify.  Left in place rather than deleted out from under
    that reference, per DISCIPLINE's "grep-verified zero callers in src/
    AND tests/" bar -- this one is not zero in tests/, even though the
    caller is inert.
    """
    from ffi import ffi_dial_key

    key = ("spectral_cart", id(mesh), tuple(kgrid), ffi_dial_key())
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)
    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn

    spec_3d = P(None, None, None, "x", "y")
    component_spec_3d = P(None, None, None, None, "x", "y")
    fft = make_flat_k_fftn(mesh, kgrid, spec_3d, norm="ortho")
    # Batch x/y/z through one inverse-FFT service call.  Besides avoiding
    # three dispatches, this keeps the shared real-space operator resident
    # exactly once in the compiled graph.
    ifft_components = make_flat_k_ifftn(mesh, kgrid, component_spec_3d, norm="ortho")
    out_sharding = NamedSharding(mesh, P(None, None, "x", "y"))

    @jax.jit
    def _kernel(operator_k, multipliers_cart_k):
        operator_R = fft(operator_k)
        weighted_R = operator_R[:, None, :, :] * (
            1j * multipliers_cart_k.T[:, :, None, None]
        )
        deriv = jnp.moveaxis(ifft_components(weighted_R), 1, 0)
        return jax.lax.with_sharding_constraint(deriv, out_sharding)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


#: The head's link-error tolerance: 1 % of its velocity on the elements it
#: reads (``link_correction_bound``), i.e. at most ~2 % of S_aa or of the
#: Drude weight.  A map above it runs with ``D_k DeltaH = 0``
#: (``sigma_term_zeroed``).  Coarse 4-point axes sit below it at their fixed
#: points (Fe 4^3 3.9e-3, Si 4^3 6.2e-3).
HEAD_LINK_RTOL = 1.0e-2


def link_correction_bound(correction, velocity_dft, occupations_kn, *,
                          link_error: float,
                          rtol: float) -> tuple[float, float, float, float]:
    r"""Judge the link error on what the head uses: ``D_k DeltaH``.

    The head's velocity is ``v_DFT + D_k DeltaH``; ``v_DFT`` is exact and
    only the correction goes through the finite links.  The artifact's
    reconstruction of ``v_DFT`` from ``D_k H_DFT`` measures the links'
    relative error ``link_error`` on the elements the head reads
    (``file_io.parallel_transport.head_velocity_set``: transitions and the
    Fermi-surface diagonal), so the head's error is bounded by

        ``link_error * |D_k DeltaH| / |v_DFT|  <=  rtol``

    with both norms on that set (this map's occupations) and ``rtol`` =
    :data:`HEAD_LINK_RTOL`.  Returns
    ``(link_error, ratio, bound, rtol)``: every map logs it in its head block,
    and a map whose bound exceeds ``rtol`` runs with ``D_k DeltaH = 0``
    (:func:`sigma_term_zeroed`).  A DFT-start map 0 has ``DeltaH = 0`` and a
    zero bound.
    """
    from file_io.parallel_transport import head_velocity_set
    nb = int(velocity_dft.shape[-1])
    head_set = head_velocity_set(jnp.asarray(occupations_kn)[:, :nb])[None]
    ratio = float(jax.device_get(jnp.sqrt(
        jnp.sum(jnp.where(head_set, jnp.abs(correction) ** 2, 0.0))
        / jnp.maximum(jnp.sum(jnp.where(head_set, jnp.abs(velocity_dft) ** 2,
                                        0.0)), 1.0e-60))))
    bound = float(link_error) * ratio
    return float(link_error), ratio, bound, float(rtol)


def sigma_term_zeroed(link_unserved: str | None, bound) -> str | None:
    """The parallel_transport head's one rule: why ``D_k DeltaH`` is zero on this map.

    Owner 2026-09-30: the head stays ``U^dagger (v_DFT + D_k DeltaH) U`` for
    the whole run; on a map whose links cannot serve the Sigma term it is
    set to zero, ``U^dagger v_DFT U``, and the next map checks again.  No
    refusal and no other velocity mode.  The links cannot serve when they
    are incomplete or fail the stencil or window-hybridization gate
    (``link_unserved``, fixed for the run) or when this map's
    :func:`link_correction_bound` exceeds :data:`HEAD_LINK_RTOL`.  Returns
    the reason, or None when the term is served.
    """
    if link_unserved:
        return str(link_unserved)
    if bound is None:
        return None
    _, _, value, rtol = bound
    if np.isfinite(value) and value <= rtol:
        return None
    return f"link bound above rtol {rtol:.1e}"


def velocity_term_shares(v_qp, pieces, *, nb_logical, surface_weight_kn=None,
                         energies_kn=None, occupations_kn=None):
    r"""Each velocity term's share of the head, per Cartesian axis.

    ``v_qp = sum_X X`` (QP basis); the share of term ``X`` is
    ``sum w Re(conj(v) X) / sum w |v|^2``, so the shares add to 1.  Metals
    (``surface_weight_kn``): the Fermi-surface weights on the diagonal, i.e.
    the share of the Drude weight ``omega_p^2``.  Insulators: interband
    pairs with ``w = |f_m - f_n| / |E_m - E_n|^3``, the share of the static
    q->0 head ``S_aa(0)``.  Returns ``(names, shares[n_terms, 3])``.
    """
    nb = int(v_qp.shape[-1])
    keep = jnp.arange(nb) < int(nb_logical)
    if surface_weight_kn is not None:
        w = jnp.asarray(surface_weight_kn, dtype=jnp.float64)[:, :nb] * keep
        v = jnp.diagonal(v_qp, axis1=-2, axis2=-1)
        total = jnp.sum(w[None] * jnp.abs(v) ** 2, axis=(1, 2))
        parts = [jnp.sum(w[None] * jnp.real(jnp.conj(v) * jnp.diagonal(
            x, axis1=-2, axis2=-1)), axis=(1, 2)) for _, x in pieces]
    else:
        e = jnp.asarray(energies_kn, dtype=jnp.float64)[:, :nb]
        f = jnp.asarray(occupations_kn, dtype=jnp.float64)[:, :nb]
        dE = jnp.abs(e[:, :, None] - e[:, None, :])
        df = jnp.abs(f[:, :, None] - f[:, None, :])
        live = (dE > 1.0e-6) & (df > 1.0e-10) & keep[:, None] & keep[None, :]
        w = jnp.where(live, df / jnp.where(live, dE, 1.0) ** 3, 0.0)
        total = jnp.sum(w[None] * jnp.abs(v_qp) ** 2, axis=(1, 2, 3))
        parts = [jnp.sum(w[None] * jnp.real(jnp.conj(v_qp) * x), axis=(1, 2, 3))
                 for _, x in pieces]
    shares = np.asarray(jax.device_get(jnp.stack(parts) / jnp.maximum(
        total, 1.0e-300)[None]), dtype=np.float64)
    return tuple(name for name, _ in pieces), shares


def covariant_link_derivative(
    delta_h_dft,
    forward_links,
    forward_neighbors,
    *,
    mesh: Mesh,
    kgrid,
    bvec_cart,
    collapsed_position=None,
):
    """Return the direct finite-link covariant derivative of ``Delta H``.

    Neighbouring operators are transported into the central DFT basis before
    the stencil is applied on the point-group-closed link shell
    (``common.parallel_transport.link_stencil`` owns the steps, weights and
    orders); a collapsed axis takes ``-i[Z_a, Delta H]`` with the stored
    position operator ``collapsed_position``.  This is one gauge-covariant
    discrete object; no separately differentiated Hamiltonian and
    connection commutator have to cancel on a finite grid.
    """
    from common.parallel_transport import (
        link_covariant_derivative,
        link_stencil,
        make_distributed_band_matmul,
    )

    delta = jnp.asarray(delta_h_dft, dtype=jnp.complex128)
    links = jnp.asarray(forward_links, dtype=jnp.complex128)
    grid = tuple(int(n) for n in kgrid)
    reduced = link_covariant_derivative(
        delta,
        links,
        np.asarray(forward_neighbors, dtype=np.int64),
        link_stencil(grid, bvec_cart),
        band_matmul=make_distributed_band_matmul(mesh, n_batch_axes=1),
        collapsed_position=collapsed_position,
    )
    return reduced_covector_to_cartesian(reduced, bvec_cart)


def _active_rotation_kernel(mesh: Mesh, nb_active: int) -> Callable:
    key = ("active_velocity_rotation", id(mesh), int(nb_active))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    from common.parallel_transport import make_distributed_band_matmul

    multiply = make_distributed_band_matmul(mesh, n_batch_axes=2)
    na = int(nb_active)

    @jax.jit
    def _kernel(v, U):
        change = U - jnp.eye(na, dtype=U.dtype)[None]
        change = jnp.broadcast_to(change[None], (3,) + change.shape)
        right = multiply(v[:, :, :, :na], change)
        tmp = v.at[:, :, :, :na].add(right)
        change_h = jnp.swapaxes(jnp.conj(change), -1, -2)
        left = multiply(change_h, tmp[:, :, :na, :])
        return tmp.at[:, :, :na, :].add(left)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def _rotation_kernel(mesh: Mesh) -> Callable:
    key = ("velocity_rotation", id(mesh))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)

    def _right_local(v_row, U_col):
        return jnp.einsum("akim,kmn->akin", v_row, U_col, optimize=True)

    right = shard_map(
        _right_local,
        mesh=mesh,
        in_specs=(P(None, None, "x", None), P(None, None, "y")),
        out_specs=P(None, None, "x", "y"),
        check_vma=False,
    )

    def _left_local(U_free, tmp_col):
        return jnp.einsum("kmp,akmn->akpn", jnp.conj(U_free), tmp_col, optimize=True)

    left = shard_map(
        _left_local,
        mesh=mesh,
        in_specs=(P(None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, None, "x", "y"),
        check_vma=False,
    )

    @jax.jit
    def _kernel(velocity_cart, U):
        # Each contraction gathers one band axis only.  The intermediate
        # and result remain P(component,k,x,y), so no full nb^2 matrix is
        # resident on a rank.
        return left(U, right(velocity_cart, U))

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def rotate_velocity_to_qp(velocity_cart, U_dft_to_qp, *, mesh: Mesh):
    """Return ``U^dagger v_i U`` for all Cartesian components in one jit."""
    return _rotation_kernel(mesh)(velocity_cart, U_dft_to_qp)


def rotate_velocity_active_to_qp(velocity_cart, U_active, *, mesh: Mesh):
    """Apply blockdiag(U_active,I)^H v blockdiag(U_active,I).

    Work scales as O(nb_head * nb_active^2), and no dense full-manifold
    unitary is constructed.
    """
    na = int(U_active.shape[-1])
    if U_active.shape[-2] != na:
        raise ValueError("U_active must be square on its band axes")
    return _active_rotation_kernel(mesh, na)(velocity_cart, U_active)


# ---------------------------------------------------------------------------
# sc_head_update = interband_commutator: the QSGW head velocity without links
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterbandCommutatorHeadData:
    r"""Head inputs of ``sc_head_update = interband_commutator``.

    The exact DFT velocity stage of the ``get_dipole_mtxels
    --parallel-transport-out`` artifact (the same read as
    :class:`DftVelocityHeadData`), authenticated to carry the nonlocal
    commutator.  Each map adds :func:`interband_commutator_velocity` to it.
    No links are read, so no k axis needs a stencil.  ``forward_links`` is
    pinned at ``None`` so the shared head builder takes its no-link path.
    """

    velocity_dft_cart: jax.Array
    nb_logical: int
    reciprocal_lattice_cart: np.ndarray
    #: ``(3, nk, nb_s, nb_s)`` reduced position operator ``Z_a = <m|b_a.r|n>``
    #: of the collapsed (one-point) k axes, zero elsewhere; ``None`` on a
    #: grid with no collapsed axis.  Along such an axis the head uses it
    #: exactly (``-i[Z_a, DeltaH]``) instead of the cross-gap commutator.
    collapsed_position: object = None
    collapsed_axes: tuple = ()
    forward_links: None = None
    forward_neighbors: None = None
    validation: None = None


def load_interband_commutator_head(
    path: str, *, mesh: Mesh, wfn, meta, config,
) -> InterbandCommutatorHeadData:
    """Load the DFT velocity stage and refuse a velocity without V_NL.

    ``W = v/(E_m - E_l)`` is the interband position operator only when
    ``v = i[H, r]`` for the SAME ``H`` whose energies divide it, so the
    velocity must carry ``i[V_NL, r]``.  The producer stamps that as
    ``vnl_included``; a p-only artifact (``--skip-vnl``) or one that predates
    the stamp refuses here.  Every other provenance check is
    :func:`load_dft_velocity_head`'s.
    """
    from common.parallel_transport import band_storage_extent, collapsed_axes
    from file_io.parallel_transport import COLLAPSED_POSITION_DATASET
    from file_io.slab_io import SlabIO

    axes = tuple(int(a) for a in collapsed_axes(wfn.kgrid)) if wfn is not None else ()
    position = None
    with SlabIO(path, mode="r", mesh=mesh) as io:
        try:
            vnl_included = int(io.read_small("vnl_included", dtype=np.int32))
        except (KeyError, RuntimeError, OSError, ValueError) as exc:
            vnl_included = None
            detail = f"{type(exc).__name__}: {exc}"
        if vnl_included == 1 and axes:
            nb_head = int(meta.b_id_4_user)
            nb_outer = max(nb_head, int(io.read_small("band_stop", dtype=np.int64)))
            outer_storage = band_storage_extent(mesh, nb_outer)
            try:
                position = head_band_block(io.read_slab(
                    COLLAPSED_POSITION_DATASET,
                    shape=(3, int(meta.nk_tot), outer_storage, outer_storage),
                    partition_spec=P(None, None, "x", "y")),
                    nb_head, mesh=mesh, nb_outer=nb_outer)
            except (KeyError, RuntimeError, OSError, ValueError) as exc:
                raise ValueError(
                    f"GATE pt_collapsed_axis_artifact: {path}: the k grid "
                    f"{tuple(int(n) for n in wfn.kgrid)} has a collapsed axis "
                    f"but {COLLAPSED_POSITION_DATASET} could not be read "
                    f"({type(exc).__name__}: {exc}); regenerate with "
                    "get_dipole_mtxels --parallel-transport-out "
                    "--parallel-transport-velocity-only") from exc
    if vnl_included != 1:
        got = ("no vnl_included stamp (" + detail + ")"
               if vnl_included is None else f"vnl_included = {vnl_included}")
        raise ValueError(
            "GATE sc_head_interband_commutator_velocity_operator: "
            f"{path}: {got}.\n"
            "  want: the full DFT velocity v = p + i[V_NL, r] (+ SOC), "
            "stamped vnl_included = 1\n"
            "  fix:  regenerate with get_dipole_mtxels "
            "--parallel-transport-out <file> --parallel-transport-velocity-only "
            "(without --skip-vnl)\n"
            "  why:  r_ml = -i v_ml/(E_m - E_l) holds only for the velocity of "
            "the Hamiltonian whose energies divide it; p alone is off by the "
            "nonlocal commutator\n"
            "  doc:  docs/self_consistency.md, 'Interband-commutator head'")
    base = load_dft_velocity_head(
        path, mesh=mesh, wfn=wfn, meta=meta, config=config)
    return InterbandCommutatorHeadData(
        velocity_dft_cart=base.velocity_dft_cart,
        nb_logical=int(base.nb_logical),
        reciprocal_lattice_cart=base.reciprocal_lattice_cart,
        collapsed_position=position,
        collapsed_axes=axes,
    )


def _interband_commutator_kernel(
    mesh: Mesh, *, nb_logical: int, nb_active: int, n_occ: int,
) -> Callable:
    from common.parallel_transport import make_distributed_band_matmul
    from gw.degen_average import TOL_DEGENERACY_RY

    key = ("interband_commutator", id(mesh), int(nb_logical),
           int(nb_active), int(n_occ))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)
    multiply = make_distributed_band_matmul(mesh, n_batch_axes=2)
    out_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    tile_sharding = NamedSharding(mesh, P(None, "x", "y"))
    na, nbl, nv = int(nb_active), int(nb_logical), int(n_occ)
    tol = float(TOL_DEGENERACY_RY)

    @jax.jit
    def _kernel(v, delta_active, tail_diagonal, e_dft, mix_w, mix_z, z_pos):
        nbs = int(v.shape[-1])
        band = jnp.arange(nbs)
        live = band < nbl
        de = jax.lax.with_sharding_constraint(
            e_dft[:, :, None] - e_dft[:, None, :], tile_sharding)
        # Cross-gap pairs only: valence-conduction and conduction-valence.
        occupied = band < nv
        pair = (live[:, None] & live[None, :]
                & (occupied[:, None] != occupied[None, :]))
        keep = pair[None] & (jnp.abs(de) > tol)
        W = jnp.where(keep[None], v / jnp.where(keep, de, 1.0)[None],
                      jnp.zeros((), dtype=v.dtype))
        # Collapsed reduced axes take the exact position operator:
        # W_red = B W_cart, W_red[a] = i Z_a there, W_cart = B^-1 W_red.
        W = (jnp.einsum("ij,j...->i...", mix_w, W)
             + jnp.einsum("ij,j...->i...", mix_z, 1j * z_pos))
        W = jax.lax.with_sharding_constraint(W, out_sharding)
        # [DeltaH, W] with DeltaH = active block (+) diagonal tail.
        t = jnp.where(band[None, :] >= na, tail_diagonal, 0.0)
        C = (t[None, :, :, None] - t[None, :, None, :]) * W
        A = jnp.broadcast_to(delta_active[None], (3,) + delta_active.shape)
        C = C.at[:, :, :na, :].add(multiply(A, W[:, :, :na, :]))
        C = C.at[:, :, :, :na].add(-multiply(W[:, :, :, :na], A))
        C = jax.lax.with_sharding_constraint(C, out_sharding)
        vc = (band[:, None] < nv) & (band[None, :] >= nv) & live[None, :]
        vc = vc[None, None]
        num = jnp.sum(jnp.where(vc, jnp.abs(C) ** 2, 0.0), axis=(1, 2, 3))
        den = jnp.sum(jnp.where(vc, jnp.abs(v) ** 2, 0.0), axis=(1, 2, 3))
        excluded = jnp.sum(pair[None] & ~keep)
        min_kept = jnp.min(jnp.where(keep, jnp.abs(de), jnp.inf))
        return (jax.lax.with_sharding_constraint(v + C, out_sharding),
                (num, den, excluded, min_kept))

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def interband_commutator_velocity(
    velocity_dft_cart,
    delta_h_active,
    tail_diagonal,
    energies_dft_kn_ry,
    *,
    nb_logical: int,
    n_occ: int,
    mesh: Mesh,
    collapsed_position=None,
    collapsed_axes=(),
    reciprocal_lattice_cart=None,
):
    r"""QSGW head velocity ``v + [DeltaH, W]`` in the DFT basis, no links.

    ``W_vc = v_vc/(E_v - E_c)`` is ``i r_vc``, the cross-gap position
    operator of the DFT Hamiltonian (valence ``v < n_occ <= c``), so
    ``[H_DFT, W] = v`` on the valence-conduction blocks and
    ``-i[r^VC, DeltaH] = [DeltaH, W]``.  Split the covariant derivative by
    occupation class, ``D DeltaH = -i[A^VC, DeltaH] + D^class DeltaH``.  The
    valence-conduction block of ``D^class DeltaH`` holds only the cross-gap
    block ``DeltaH_VC``, so the head is exact for any ``DeltaH`` that does
    not mix valence and conduction (a band-diagonal ``DeltaH`` gives
    ``v_vc (E^QP_v - E^QP_c)/(E_v - E_c)``), and its error is first order in
    the cross-gap mixing.  No sum over states is truncated: ``DeltaH`` is
    the active block plus a diagonal tail, so every ``W`` element it meets
    is inside the head manifold.

    Degenerate and near-degenerate manifolds: every same-class pair is
    EXCLUDED, not only exact multiplets.  Inside a class the interband
    ``W`` has no gap below it: near-degenerate pairs make it arbitrarily
    large, and the k derivative of ``DeltaH`` that would cancel it has no
    stencil-free form.  Excluding pairs within ``TOL_DEGENERACY_RY`` alone
    left Si 6x6x6 SOC with a head 8.8x the link head (pairs at 0.19 meV);
    the class rule has no tolerance and every denominator is at least the
    direct gap.  A cross-gap pair within ``TOL_DEGENERACY_RY``
    (``gw.degen_average``) is a closed gap; it is excluded and counted, and
    the SC caller refuses it.

    Collapsed axes (a slab normal, a wire's transverse axes): there the
    cell is not periodic and the connection is the stored position operator
    ``Z_a`` (``common.parallel_transport.link_stencil_orders``), so the
    reduced component ``a`` of ``W`` is ``i Z_a``, full and exact, and the
    cross-gap rule is used only along the periodic axes:
    ``W_cart = B^-1 [B W^VC with rows a replaced by i Z_a]``.

    Shapes: ``velocity_dft_cart`` (3, nk, nb_s, nb_s) ``P(None, None, x, y)``;
    ``delta_h_active`` (nk, na, na); ``tail_diagonal`` and
    ``energies_dft_kn_ry`` (nk, nb_s), Ry.  Work O(3 nk na^2 nb_s); peak
    transients three velocity-sized arrays (v, W, the correction), each
    ``3 nk nb_s^2 16 B / P`` per rank.

    Returns ``(velocity, stats)``; ``stats`` holds, per Cartesian axis, the
    valence-conduction sums ``|[DeltaH, W]_vc|^2`` and ``|v_vc|^2``, the count
    of cross-gap pairs within ``TOL_DEGENERACY_RY``, and the smallest kept
    ``|E_c - E_v|`` (the direct DFT gap of the manifold).
    """
    v = jnp.asarray(velocity_dft_cart, dtype=jnp.complex128)
    delta = jnp.asarray(delta_h_active, dtype=jnp.complex128)
    tail = jnp.asarray(tail_diagonal, dtype=jnp.float64)
    e = jnp.asarray(energies_dft_kn_ry, dtype=jnp.float64)
    if v.ndim != 4 or int(v.shape[0]) != 3 or v.shape[-1] != v.shape[-2]:
        raise ValueError(f"velocity must be (3,nk,nb,nb); got {v.shape}")
    nk, nbs = int(v.shape[1]), int(v.shape[-1])
    if delta.ndim != 3 or delta.shape[0] != nk or delta.shape[1] != delta.shape[2]:
        raise ValueError(f"delta_h_active must be (nk,na,na); got {delta.shape}")
    na = int(delta.shape[-1])
    if tail.shape != (nk, nbs) or e.shape != (nk, nbs):
        raise ValueError(
            f"tail/energies must be ({nk},{nbs}); got {tail.shape}/{e.shape}")
    if not (0 < int(n_occ) < int(nb_logical) <= nbs and na <= nbs):
        raise ValueError(
            "interband commutator head needs 0 < n_occ < nb_logical <= "
            f"nb_storage and na <= nb_storage; got n_occ={n_occ}, "
            f"nb_logical={nb_logical}, nb_storage={nbs}, na={na}")
    axes = tuple(int(a) for a in collapsed_axes)
    if axes:
        if collapsed_position is None or reciprocal_lattice_cart is None:
            raise ValueError(
                "collapsed axes need the stored position operator and the "
                "reciprocal lattice")
        B = np.asarray(reciprocal_lattice_cart, dtype=np.float64)
        Binv = np.linalg.inv(B)
        sel = np.zeros((3, 3))
        sel[axes, axes] = 1.0
        mix_w = Binv @ (np.eye(3) - sel) @ B
        mix_z = Binv @ sel
        z_pos = jnp.asarray(collapsed_position, dtype=jnp.complex128)
        if z_pos.shape != v.shape:
            raise ValueError(
                f"collapsed position {z_pos.shape} != velocity {v.shape}")
    else:
        mix_w, mix_z = np.eye(3), np.zeros((3, 3))
        z_pos = jnp.zeros((3, 1, 1, 1), dtype=jnp.complex128)
    return _interband_commutator_kernel(
        mesh, nb_logical=int(nb_logical), nb_active=na, n_occ=int(n_occ),
    )(v, delta, tail, e, jnp.asarray(mix_w, dtype=jnp.complex128),
      jnp.asarray(mix_z, dtype=jnp.complex128), z_pos)


#: The QSGW velocity file the SC driver writes beside ``dipole.h5``.
QSGW_DIPOLE_FILE = "dipole_qsgw.h5"


def write_qsgw_dipole(path, velocity: QPVelocity, U_active, energies_qp_kn_ry,
                      *, nb_logical: int, mesh: Mesh, print_fn=print) -> None:
    r"""Write one SC map's velocity as a ``dipole.h5`` in its QP basis.

    ``velocity`` is the map's :func:`qp_velocity` (the one the head read),
    ``U_active`` the map's rotation to its input QP states and
    ``energies_qp_kn_ry`` their energies: the same three operands the map's
    head consumed (:func:`build_iteration_head_response`).  The file holds
    ``U^dagger v U`` with ``band_energies = E_QP`` in ``file_io.dipole``'s
    layout, so ``load_dipole_h5`` and every absorption consumer read it
    unchanged; their ``d = v_cv/(E_c - E_v)`` is then the position operator
    between QP states.  ``velocity.label`` names the Sigma term and is
    stamped as the ``velocity`` attribute (docs/self_consistency.md,
    'QSGW dipoles').  COLLECTIVE.
    """
    from common.collectives import gather_to_host
    from file_io.dipole import write_dipole

    v_qp = rotate_velocity_active_to_qp(velocity.dft_cart, U_active, mesh=mesh)
    nb = int(nb_logical)
    energies = energies_qp_kn_ry
    energies = np.asarray(gather_to_host(energies) if isinstance(
        energies, jax.Array) else energies, dtype=np.float64)[:, :nb]
    kmajor = jax.jit(lambda v: jnp.moveaxis(v, 0, 1), out_shardings=NamedSharding(
        mesh, P(None, None, "x", "y")))(v_qp)
    del v_qp
    write_dipole(path, kmajor, energies, mesh=mesh, attrs={
        "nbands": nb, "nk": int(energies.shape[0]), "skip_vnl": 0,
        "basis": "qp", "velocity": velocity.label,
        "note": ("QSGW velocity U^H v U between the SC final map's input QP "
                 "states (gw.qsgw_head.qp_velocity); band_energies are "
                 "their QP energies")})
    print_fn(f"  QSGW dipoles: {os.path.basename(str(path))} "
             f"({int(energies.shape[0])} k x {nb} bands, QP basis, "
             f"{velocity.label})")


def stamp_qsgw_dipole_provenance(path, *, wfn_qp_path, config, wfn, meta,
                                  print_fn=print) -> None:
    r"""Bind ``dipole_qsgw.h5`` to ``WFN_qp.h5`` with ``dipole.h5``'s stamps.

    The file is ``U^\dagger v U`` between the states ``WFN_qp.h5`` stores,
    so it is that WFN's velocity: the WFN identity stamped is WFN_qp's
    fingerprint, and the window, V_NL mode and sign, representation and
    DFT+U stamps are this deck's (:func:`head_dipole_operator_stamps`, the
    set the SC head was authenticated against).  A GW run on WFN_qp with
    the same deck then reads it as its ``dipole.h5`` through
    :func:`read_authenticated_dipole_velocity`; any other WFN refuses it by
    fingerprint.  The Sigma term stays named by the ``velocity`` stamp.
    Rank-0 write; call once ``WFN_qp.h5`` is published.  COLLECTIVE.
    """
    from common.collectives import rank0_transaction
    from psp.get_dipole_mtxels import dipole_provenance

    if int(meta.b_id_0) != 0:
        print_fn(f"  QSGW dipoles: {os.path.basename(str(path))} not bound "
                 f"to WFN_qp (head bands start at {int(meta.b_id_0)}, the "
                 "dipole reader indexes bands from 0)")
        return
    stamps = head_dipole_operator_stamps(
        config, wfn=wfn, meta=meta,
        fallback_dir=os.path.dirname(os.path.abspath(str(path))))

    def _stamp():
        import h5py
        from wfn_loader import WfnLoader
        with h5py.File(str(path), "r+") as h5:
            nb_written = int(h5.attrs["nbands"])
            wfn_qp = WfnLoader(str(wfn_qp_path))
            for key, value in dipole_provenance(
                    wfn=wfn_qp, wfn_path=str(wfn_qp_path),
                    nb_written=nb_written, nspinor=int(wfn_qp.nspinor),
                    **stamps).items():
                h5.attrs[key] = value

    rank0_transaction(path, stage="qsgw_dipole_provenance", write=_stamp)
    print_fn(f"  QSGW dipoles: {os.path.basename(str(path))} bound to "
             f"{os.path.basename(str(wfn_qp_path))} (dipole.h5 provenance "
             "stamps; a GW run on WFN_qp reads it as its dipole.h5)")


def _assemble_kernel(mesh: Mesh, nb_storage: int) -> Callable:
    key = ("assemble_head_manifold", id(mesh), int(nb_storage))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)
    out_sharding = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def _kernel(delta_active, U_active):
        nk, nb_active, _ = delta_active.shape
        delta = jnp.zeros((nk, nb_storage, nb_storage), dtype=jnp.complex128)
        delta = delta.at[:, :nb_active, :nb_active].set(delta_active)
        U = jnp.broadcast_to(
            jnp.eye(nb_storage, dtype=jnp.complex128)[None, :, :],
            (nk, nb_storage, nb_storage),
        )
        U = U.at[:, :nb_active, :nb_active].set(U_active)
        return (
            jax.lax.with_sharding_constraint(delta, out_sharding),
            jax.lax.with_sharding_constraint(U, out_sharding),
        )

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def assemble_head_manifold(
    delta_h_active,
    U_active,
    *,
    nb_storage: int,
    mesh: Mesh,
):
    """Embed the active QSGW block in the full velocity/head manifold.

    The inactive correction is zero and its basis rotation is identity.
    Keeping the full matrix is load-bearing: A-active/inactive commutators
    and high-conduction transitions would both be lost by slicing A down to
    the active Sigma window.
    """
    if delta_h_active.shape != U_active.shape or delta_h_active.ndim != 3:
        raise ValueError(
            "active delta-H and U must be equal-shaped (nk,nb,nb) arrays; "
            f"got {delta_h_active.shape}/{U_active.shape}."
        )
    if delta_h_active.shape[1] != delta_h_active.shape[2]:
        raise ValueError("active delta-H/U matrices must be square.")
    if int(delta_h_active.shape[1]) > int(nb_storage):
        raise ValueError(
            f"active nb={delta_h_active.shape[1]} exceeds head storage nb={nb_storage}."
        )
    return _assemble_kernel(mesh, int(nb_storage))(delta_h_active, U_active)


def _assemble_delta_kernel(mesh: Mesh, nb_storage: int) -> Callable:
    key = ("assemble_delta_head", id(mesh), int(nb_storage))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    out_sharding = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def _kernel(delta_active, tail_diagonal):
        nk, na, _ = delta_active.shape
        delta = jnp.zeros((nk, nb_storage, nb_storage), dtype=jnp.complex128)
        delta = delta.at[:, :na, :na].set(delta_active)
        idx = jnp.arange(na, nb_storage)
        delta = delta.at[:, idx, idx].set(tail_diagonal[:, na:nb_storage])
        return jax.lax.with_sharding_constraint(delta, out_sharding)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def assemble_delta_head_manifold(
    delta_h_active,
    tail_diagonal,
    *,
    nb_storage: int,
    mesh: Mesh,
    nb_logical: int | None = None,
    nb_links: int | None = None,
):
    """Embed active DeltaH and the current diagonal sum-band tail.

    ``nb_links > nb_logical``: the links run on an outer band set
    (``ParallelTransportHeadData.nb_links``), so DeltaH is embedded there.
    Past the head's ``nb_logical`` bands it continues the diagonal scissor
    tail: each k takes its highest head band's shift (the rigid tail law's
    Delta_c when that band is in the tail).
    """
    delta = jnp.asarray(delta_h_active)
    tail = jnp.asarray(tail_diagonal)
    if nb_links is not None and nb_logical is not None:
        # The links' carrier (band_storage_extent, the whole mesh product)
        # can exceed the head's (head_storage_extent) with no outer set.
        from common.parallel_transport import band_storage_extent
        top = int(nb_logical)
        outer_storage = band_storage_extent(mesh, int(nb_links))
        if int(nb_links) > top or outer_storage > int(nb_storage):
            tail = jnp.concatenate([
                tail[:, :top],
                jnp.broadcast_to(tail[:, top - 1:top],
                                 (tail.shape[0], outer_storage - top))], axis=1)
            nb_storage = outer_storage
    if delta.ndim != 3 or delta.shape[-2] != delta.shape[-1]:
        raise ValueError("delta_h_active must be (nk,na,na)")
    if tail.ndim != 2 or tail.shape[0] != delta.shape[0]:
        raise ValueError("tail_diagonal must be (nk,nb_storage)")
    if int(tail.shape[1]) < int(nb_storage):
        raise ValueError(f"tail diagonal extent {tail.shape[1]} < storage {nb_storage}")
    if int(delta.shape[-1]) > int(nb_storage):
        raise ValueError("active DeltaH exceeds the head manifold")
    return _assemble_delta_kernel(mesh, int(nb_storage))(delta, tail)


def _interband_weight(dE, f_diff, z, prefactor):
    r"""Adler-Wiser interband weight ``prefactor f_diff / (dE (z^2 - dE^2))``.

    Every pair reaching this weight is split by more than
    :data:`gw.degen_average.TOL_DEGENERACY_RY`: an exactly degenerate pair
    carries no interband transition.  Its ``dE -> 0`` limit is Fermi-surface
    (intraband) content, ``(-f') |v_nm|^2 / z^2``, which the metallic head
    takes in the multiplet Drude tensor of :func:`head_drude_tensor_sharded`
    with the same weights as the diagonal, so the tensor is invariant under
    rotations inside the multiplet.  Here a 0/1 table that cuts a multiplet
    would give ``1 / (dE z^2)`` at roundoff ``dE``; the mask removes that.
    The clip guards the ``z ~ dE`` resonance of a split pair at real ``z``.
    """
    denom = dE * (z * z - dE * dE)
    return jnp.where(jnp.abs(denom) > 1.0e-16, prefactor * f_diff / denom,
                     jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))


def _head_wing_interband_weight(
    dE, f_diff, z, prefactor, transition,
):
    r"""Adler--Wiser mixed head/body weight in the ``P=-dE*D`` basis.

    Replacing one density-jet leg ``D`` of the direct response by the
    energy-scaled head vertex ``P=-dE*D`` contributes the explicit minus
    below.  Both wing layouts call this owner.  The finite-frequency
    intraband surface term is not a ``D -> P`` substitution and remains the
    separate positive ``pref_surface*surface_weight/z`` contribution in the
    two kernels.
    """
    denom = z * z - dE * dE
    return jnp.where(
        transition & (jnp.abs(denom) > 1.0e-16),
        -prefactor * f_diff / denom,
        jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
    )


def _s_tensor_kernel(mesh: Mesh, *, nb_logical: int,
                     metal_split: bool = False) -> Callable:
    key = (("head_s", id(mesh), int(nb_logical)) if not metal_split
           else ("head_s_metal", id(mesh), int(nb_logical)))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(v_local, e_bra, e_ket, f_bra, f_ket, omegas, prefactor, eta,
               *split):
        nx, ny = v_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        dE = e_bra[:, :, None] - e_ket[:, None, :]
        f_diff = f_ket[:, None, :] - f_bra[:, :, None]
        logical = ((ix[:, None] < nb_logical) & (iy[None, :] < nb_logical))[None, :, :]
        # Sum every energy-ordered band pair split by more than the BGW
        # degeneracy tolerance.  f_diff is SIGNED (MP1 may overshoot [0, 1]),
        # so filtering on f_v-f_c>0 would not implement Adler-Wiser.  A pair
        # inside one multiplet is intraband content (``_interband_weight``).
        transition = logical & (dE >= TOL_DEGENERACY_RY)
        keep = None
        if metal_split:
            # A metal keeps 1 - phi of each pair here; its Fermi-surface share
            # phi (``fermi_surface.intraband_pair_fraction``) goes to D.
            from gw.fermi_surface import intraband_pair_fraction
            d_x, d_y, w_x, w_y, moment = split
            keep = 1.0 - intraband_pair_fraction(
                v_local, dE, d_x, d_y, w_x, w_y, moment, TOL_DEGENERACY_RY)

        def _one(omega):
            z = omega + 1j * eta
            weight = jnp.where(
                transition,
                _interband_weight(dE, f_diff, z, prefactor),
                jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
            )
            if keep is not None:
                weight = weight * keep
            local = jnp.einsum(
                "akij,kij,bkij->ab", jnp.conj(v_local), weight, v_local, optimize=True
            )
            return jax.lax.psum(local, (ax_x, ax_y))

        # Bounded frequency-by-band-pair temporary (mirrors
        # ``_head_wing_kernel``'s ``_HEAD_WING_FREQUENCY_BLOCK`` ring: see
        # the comment at its definition, which names the wing kernel as
        # bounding "the ONLY" such temporary -- this one was the omitted
        # twin).  ``jax.vmap(_one)(omegas)`` batches ``dE``/``weight``
        # (shape ``(n_omega, nk, nx, ny)``) across the FULL omega axis at
        # once; on XLA:GPU this is materialised as a standalone buffer
        # before the reducing einsum, exactly the "global einsum lets XLA
        # select a full-matrix temporary even though the public result is
        # 3x3" failure mode commit d2d6d521 fixed for the Schur fold.
        # Chunking the batch to a fixed block bounds that temporary at
        # ``block`` frequencies regardless of how many omegas a future
        # caller (an N-pole GN-PPM fit, a dense MPA frequency walk) asks
        # for; today's GN-PPM/HL-PPM two-role case pads to one block and
        # costs nothing extra.
        n_omega = omegas.shape[0]
        block = min(_HEAD_WING_FREQUENCY_BLOCK, int(n_omega))
        from runtime.padding import padded_axis
        n_padded = padded_axis(
            int(n_omega), block,
            name="head-wing frequency block carrier").carrier
        pad = n_padded - int(n_omega)
        omega_blocks = jnp.pad(
            omegas, (0, pad), constant_values=jnp.asarray(1.0j, dtype=omegas.dtype)
        ).reshape(-1, block)

        def _block(_carry, omega_block):
            return _carry, jax.vmap(_one)(omega_block)

        _, out_blocks = jax.lax.scan(_block, None, omega_blocks, unroll=1)
        n_vertex = int(v_local.shape[0])
        return out_blocks.reshape(
            n_padded, n_vertex, n_vertex)[:n_omega]

    split_specs = ((P(None, None, "x"), P(None, None, "y"), P(None, "x"),
                    P(None, "y"), P(None, None)) if metal_split else ())
    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),
            P(None, "x"),
            P(None, "y"),
            P(None, "x"),
            P(None, "y"),
            P(None),
            P(),
            P(),
        ) + split_specs,
        out_specs=P(None, None, None),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _head_wing_kernel(
    mesh: Mesh,
    *,
    nb_logical: int,
    include_surface: bool,
    mu_block: int,
    layout: str = "face",
    classes: int = 0,
    anti: bool = False,
) -> Callable:
    """Build the canonical face head-wing kernel with bounded centroid tiles."""
    if layout not in ("face", "axis"):
        raise ValueError(f"_head_wing_kernel requires face or axis layout, got {layout!r}")
    return _head_wing_kernel_face(
        mesh, nb_logical=int(nb_logical), include_surface=bool(include_surface),
        mu_block=int(mu_block), layout=layout, classes=int(classes), anti=bool(anti))


def _head_wing_kernel_face(
    mesh: Mesh,
    *,
    nb_logical: int,
    include_surface: bool,
    mu_block: int,
    layout="face",
    classes: int = 0,
    anti: bool = False,
) -> Callable:
    """Contract velocity and density vertices on the vertex's own pair tiles.

    ``classes = 0``: one k sum, outputs ``Y (n_omega, n_vertex, mu)`` and
    ``Z (n_omega, mu, n_vertex)``.  ``classes = C``: the k axis is the raw
    parents and two ``(C, n_parent)`` count matrices ``class_u``/``class_a``
    weight each parent's contraction into its operation classes (see
    :func:`_head_wings_sharded_face`); outputs gain a class axis, ``Y (n_omega,
    n_vertex, C, mu)`` and ``Z (n_omega, C, mu, n_vertex)``.  ``anti`` adds
    the antiunitary partner contraction, which reads the transposed pair
    weight: an antiunitary child's density is ``rho_ji`` of its parent.

    Residency: the vertex stays ``v[a, k, i_X, j_Y]`` and the pair weight,
    ``dE``, ``f`` and the masks are built on the same ``(i_X, j_Y)`` tile, so
    every pair-indexed value is ``1/P``.  ``Y`` reads the mun faces and ``Z``
    the nmu faces (separate endpoint bundles may differ between the two).
    Per centroid block ``b`` (one ``mu_block`` slice of each
    rank's own mu tile) a pass gathers its two endpoints on the block only,
    ``[k, s, M_b, i_X]`` for the bra and ``[k, s, M_b, j_Y]`` for the ket,
    ``M_b`` being block ``b`` of every mu tile (square mesh: the X and Y mu
    tiles are one partition).  The face's own band axis is one gather; the
    other is the transpose partner's block (``to_transpose_partner``) and
    one gather.  Each rank contracts its pair tile; ``Y`` is reduce-scattered
    over X onto ``mu_X`` and summed over Y, ``Z`` the mirror.  The
    antiunitary term reads the transposed tile, so it also takes the bra
    with bands on Y and the ket with bands on X.
    """
    key = ("head_wings_face", id(mesh), int(nb_logical), bool(include_surface),
           int(mu_block), layout, int(classes), bool(anti))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)
    if int(mesh.shape[ax_x]) != int(mesh.shape[ax_y]):
        raise ValueError(
            "head wing kernel: the pair-tile contraction needs one mu "
            f"partition on X and Y, i.e. a square mesh; got "
            f"{int(mesh.shape[ax_x])}x{int(mesh.shape[ax_y])} (repo "
            "docs/architecture/decisions.md 2026-08-01: square meshes only).")
    from jax.experimental.layout import Layout, with_layout_constraint
    from common.collectives import to_transpose_partner
    from common.wfn_layout import psi_specs
    nmu_spec, mun_spec = psi_specs(layout)
    use_anti = bool(anti) and bool(classes)

    def _local(
        v_local,
        bra_mun_local,
        ket_mun_local,
        bra_nmu_local,
        ket_nmu_local,
        energies,
        occupations,
        surface_weight,
        omegas,
        pref_inter,
        pref_surface,
        eta,
        class_u=None,
        class_a=None,
    ):
        n_vertex, nk, nbx, nby = (int(n) for n in v_local.shape)
        n_class = max(int(classes), 1)
        x0 = jax.lax.axis_index(ax_x) * nbx
        y0 = jax.lax.axis_index(ax_y) * nby
        if layout == "axis":
            # Axis faces hold whole band axes: keep this rank's pair tile rows.
            def _x_rows(a):
                return jax.lax.dynamic_slice_in_dim(a, x0, nbx, axis=1)

            def _y_cols(a):
                return jax.lax.dynamic_slice_in_dim(a, y0, nby, axis=3)
            bra_nmu_local, ket_nmu_local = _x_rows(bra_nmu_local), _x_rows(ket_nmu_local)
            bra_mun_local, ket_mun_local = _y_cols(bra_mun_local), _y_cols(ket_mun_local)

        def _k_sum(a, b, a_anti=None, b_anti=None):
            """``sum_{s,j} a b`` per k, weighted into classes (plain k sum
            when ``classes = 0``)."""
            if not classes:
                return jnp.einsum("ksmj,ksmj->m", a, b)[None]
            out = class_u @ jnp.einsum("ksmj,ksmj->km", a, b)
            if a_anti is not None:
                out = out + class_a @ jnp.einsum("ksmj,ksmj->km", a_anti, b_anti)
            return out

        ix = x0 + jnp.arange(nbx)
        iy = y0 + jnp.arange(nby)
        e_x = jax.lax.dynamic_slice_in_dim(energies, x0, nbx, axis=1)
        e_y = jax.lax.dynamic_slice_in_dim(energies, y0, nby, axis=1)
        f_x = jax.lax.dynamic_slice_in_dim(occupations, x0, nbx, axis=1)
        f_y = jax.lax.dynamic_slice_in_dim(occupations, y0, nby, axis=1)
        logical2d = ((ix[:, None] < nb_logical) & (iy[None, :] < nb_logical))[None, :, :]
        dE = e_x[:, :, None] - e_y[:, None, :]
        f_diff = f_y[:, None, :] - f_x[:, :, None]
        transition = logical2d & (dE > 0.0)
        if include_surface:
            s_x = jax.lax.dynamic_slice_in_dim(surface_weight, x0, nbx, axis=1)
            diagonal = logical2d & (ix[:, None] == iy[None, :])[None, :, :]
            surface_pair = jnp.where(diagonal, s_x[:, :, None], 0.0)
        else:
            surface_pair = jnp.zeros_like(dE)

        n_omega = omegas.shape[0]
        z = omegas + 1j * eta
        inv_z = jnp.where(
            jnp.abs(omegas) > 1.0e-15, 1.0 / z,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        freq_block = min(_HEAD_WING_FREQUENCY_BLOCK, int(n_omega))
        from runtime.padding import padded_axis
        n_omega_padded = padded_axis(
            int(n_omega), freq_block,
            name="face head response frequency carrier").carrier
        freq_pad = n_omega_padded - int(n_omega)
        z_blocks = jnp.pad(
            z, (0, freq_pad),
            constant_values=jnp.asarray(1.0j, dtype=jnp.complex128),
        ).reshape(-1, freq_block)
        inv_z_blocks = jnp.pad(inv_z, (0, freq_pad)).reshape(-1, freq_block)

        def _weighted_stack(contract):
            """Stream omega in bounded blocks of the local pair tile."""
            def _step(_carry, node):
                z_block, inv_z_block = node
                weight = _head_wing_interband_weight(
                    dE[None], f_diff[None],
                    z_block[:, None, None, None], pref_inter,
                    transition[None],
                )
                if include_surface:
                    weight = weight + (
                        pref_surface * inv_z_block[:, None, None, None]
                        * surface_pair[None])
                return _carry, contract(weight)
            _, blocks = jax.lax.scan(
                _step, None, (z_blocks, inv_z_blocks), unroll=1)
            return blocks

        mu_local = int(bra_mun_local.shape[2])
        if int(bra_nmu_local.shape[-1]) != mu_local:
            raise ValueError(
                "head wing kernel: psi_mun/psi_nmu local mu tiles differ "
                f"({mu_local} vs {int(bra_nmu_local.shape[-1])}); the pair-tile "
                "contraction needs one mu partition on X and Y.")
        # Blocks read the faces in place: the last one is clamped to end at
        # mu_local and rewrites its overlap with the same rows (no padded
        # face copies).
        mu_block_ = min(int(mu_block), mu_local)
        n_blocks = -(-mu_local // mu_block_)
        p_side = int(mesh.shape[ax_x])

        # Endpoint blocks ``[k, s, M_b, n]`` with the band tile on X (``_x``)
        # or on Y (``_y``).  A face's own band axis needs one gather; the
        # other band axis is its transpose partner's block, then a gather.
        # The block is pinned row-major and the gather stacks a new leading
        # tile axis, so only the gathered block is reordered: left free, a
        # gather along the mu axis let layout assignment copy each whole
        # resident face (4 x 882 MB on the mu3088 CPU dump; the pin is
        # common.contract_bands.bands_to_contraction_slabs's).
        def _block(a, start, axis):
            t = jax.lax.dynamic_slice_in_dim(a, start, mu_block_, axis=axis)
            return with_layout_constraint(t, Layout(major_to_minor=tuple(range(t.ndim))))

        def _gathered(t, axis_name, order):
            g = jax.lax.all_gather(t, axis_name, axis=0, tiled=False)
            g = jnp.transpose(g, order)
            return g.reshape(*g.shape[:2], -1, g.shape[-1])

        def _mun_y(a, start):
            return _gathered(_block(a, start, 2), ax_x, (1, 2, 0, 3, 4))

        def _mun_x(a, start):
            return _gathered(to_transpose_partner(_block(a, start, 2), p_side),
                             ax_y, (1, 2, 0, 3, 4))

        def _nmu_x(a, start):
            return _gathered(_block(a, start, 3), ax_y, (1, 3, 0, 4, 2))

        def _nmu_y(a, start):
            return _gathered(to_transpose_partner(_block(a, start, 3), p_side),
                             ax_x, (1, 3, 0, 4, 2))

        def _pass(bra, ket, band_x, band_y, contract, scatter_axis, sum_axis,
                  out_shape, dim):
            """One wing: per centroid block, gather the endpoints, contract the
            local pair tile, reduce-scatter the block onto its own mu tile."""
            def _block_step(acc, blk):
                start = jnp.minimum(blk * mu_block_, mu_local - mu_block_)
                ends = (band_x(bra, start), band_y(ket, start))
                if use_anti:
                    ends = ends + (band_y(bra, start), band_x(ket, start))
                part = _weighted_stack(lambda w: contract(w, *ends))
                part = part.reshape(n_omega_padded, *part.shape[2:])
                # M_b is tile-major: the scatter's chunk t is mu tile t's block.
                blk_out = jax.lax.psum(jax.lax.psum_scatter(
                    part, scatter_axis, scatter_dimension=dim, tiled=True), sum_axis)
                return jax.lax.dynamic_update_slice_in_dim(
                    acc, blk_out, start, axis=dim), None
            acc, _ = jax.lax.scan(
                _block_step, jnp.zeros(out_shape, jnp.complex128),
                jnp.arange(n_blocks, dtype=jnp.int32), unroll=1)
            return acc[:n_omega]

        # Y[w,a,m] = sum_{k,s,i,j} conj(v)[a,k,i,j] W[w,k,i,j]
        #                          conj(bra)[k,s,m,i] ket[k,s,m,j]  (mun faces)
        # Z[w,m,b] = sum_{k,s,i,j} bra[k,s,m,i] conj(ket)[k,s,m,j]
        #                          W[w,k,i,j] v[b,k,i,j]            (nmu faces)
        # over this rank's (i_X, j_Y) pair tile, per frequency and vertex:
        # T = conj(v[a]) W (nk, nbx, nby), then i against the bra, then k, s,
        # j against the ket; the largest value is T.  An antiunitary child
        # reads rho_ji: its pair weight is (v W)^T, contracted with the
        # Y-banded bra and the X-banded ket (see _head_wings_sharded_face).
        def _contract_left(weight, bra_x, ket_y, bra_y=None, ket_x=None):
            def _one_frequency(_carry, weight_w):
                rows = []
                for a in range(n_vertex):
                    t = jnp.conj(v_local[a]) * weight_w
                    u = jnp.einsum("ksmi,kij->ksmj", jnp.conj(bra_x), t)
                    u_anti = None
                    if use_anti:
                        u_anti = jnp.einsum(
                            "ksmi,kji->ksmj", jnp.conj(bra_y),
                            v_local[a] * weight_w)
                    rows.append(_k_sum(u, ket_y, u_anti, ket_x))
                return _carry, jnp.stack(rows, axis=0)
            _, y = jax.lax.scan(_one_frequency, None, weight, unroll=1)
            return y

        def _contract_right(weight, bra_x, ket_y, bra_y=None, ket_x=None):
            def _one_frequency(_carry, weight_w):
                cols = []
                for b in range(n_vertex):
                    t = weight_w * v_local[b]
                    u = jnp.einsum("ksmi,kij->ksmj", bra_x, t)
                    u_anti = None
                    if use_anti:
                        u_anti = jnp.einsum(
                            "ksmi,kji->ksmj", bra_y,
                            weight_w * jnp.conj(v_local[b]))
                    cols.append(_k_sum(
                        u, jnp.conj(ket_y), u_anti,
                        None if ket_x is None else jnp.conj(ket_x)))
                return _carry, jnp.stack(cols, axis=-1)
            _, z = jax.lax.scan(_one_frequency, None, weight, unroll=1)
            return z

        Y_x = _pass(bra_mun_local, ket_mun_local, _mun_x, _mun_y, _contract_left,
                    ax_x, ax_y, (n_omega_padded, n_vertex, n_class, mu_local), 3)
        Z_y = _pass(bra_nmu_local, ket_nmu_local, _nmu_x, _nmu_y, _contract_right,
                    ax_y, ax_x, (n_omega_padded, n_class, mu_local, n_vertex), 2)
        if not classes:
            return Y_x[:, :, 0], Z_y[:, 0]
        return Y_x, Z_y

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),   # v_local
            mun_spec,                  # bra_mun_local
            mun_spec,                  # ket_mun_local
            nmu_spec,                  # bra_nmu_local
            nmu_spec,                  # ket_nmu_local
            P(None, None),             # energies (nk, nb_full), replicated
            P(None, None),             # occupations
            P(None, None),             # surface_weight
            P(None),                   # omegas
            P(),
            P(),
            P(),
        ) + ((P(None, None), P(None, None)) if classes else ()),
        out_specs=((P(None, None, "x"), P(None, "y", None)) if not classes
                   else (P(None, None, None, "x"), P(None, None, "y", None))),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _pad_head_band_manifold_to(v, e, f, surface, *, mesh: Mesh, width: int):
    """Like ``_pad_head_band_manifold`` but pads to an EXPLICIT ``width``
    rather than inferring one from ``v``'s own current extent.

    The face wing kernel's contracted operand (``psi_mun``/``psi_nmu``) is
    NOT legally sliceable to an arbitrary logical window (obstacle #3: a
    face-sharded band axis need not be mesh-divisible at that boundary),
    so the kernel contracts ``v``'s pair tiles against the faces' own band
    tiles at the full stored ``nb_full`` width.  ``v``/
    ``e``/``f``/``surface`` must therefore be embedded in that SAME width
    (zero beyond the physical ``[b0,b4)`` extent — safe, since every
    consumer masks on ``nb_logical``, never on ``v``'s own shape) rather
    than the smaller chi0-only padding ``_pad_head_band_manifold`` does
    for the legacy kernel.  ``nb_full`` is already mesh-divisible by
    construction of the two-face carrier, so no further rounding is
    needed here.

    Also applies the fix registered in ``KNOWN_LORRAX_ISSUES.md`` (the
    v-sharding-commit defect on the legacy path): every returned array goes
    through the canonical process-local placement helper onto its declared
    mesh sharding before any kernel sees it, so a foreign
    ``SingleDeviceSharding`` operand never reaches this kernel's
    ``shard_map``.
    """
    nb = int(v.shape[-1])
    if width < nb:
        raise ValueError(
            f"_pad_head_band_manifold_to: width={width} smaller than v's "
            f"own extent {nb}")
    pad = width - nb
    if pad:
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, pad)))
        e = jnp.pad(e, ((0, 0), (0, pad)))
        f = jnp.pad(f, ((0, 0), (0, pad)))
        surface = jnp.pad(surface, ((0, 0), (0, pad)))
    v = device_put_process_local(
        v, NamedSharding(mesh, P(None, None, "x", "y")))
    e = device_put_process_local(e, NamedSharding(mesh, P(None, None)))
    f = device_put_process_local(f, NamedSharding(mesh, P(None, None)))
    surface = device_put_process_local(
        surface, NamedSharding(mesh, P(None, None)))
    return v, e, f, surface


def head_wings_sharded(
    velocity_cart,
    wfns,
    energies_kn_ry,
    occupations_kn,
    omegas_ry,
    *,
    mesh: Mesh,
    nb_logical: int,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    eta_ry: float = 0.0,
    surface_weight_kn=None,
    body_bra_wfns=None,
    body_ket_wfns=None,
):
    """Contract energy-scaled velocity jets with centroid vertices on canonical faces or parents."""
    if getattr(wfns, "layout", None) not in ("face", "axis"):
        raise ValueError("head_wings_sharded requires a canonical face or axis layout")
    return _head_wings_sharded_face(
        velocity_cart, wfns, energies_kn_ry, occupations_kn, omegas_ry,
        mesh=mesh, nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
        nspinor=nspinor, eta_ry=eta_ry, surface_weight_kn=surface_weight_kn,
        body_bra_wfns=body_bra_wfns, body_ket_wfns=body_ket_wfns)


def _head_wings_sharded_face(
    velocity_cart,
    wfns,
    energies_kn_ry,
    occupations_kn,
    omegas_ry,
    *,
    mesh: Mesh,
    nb_logical: int,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    eta_ry: float = 0.0,
    surface_weight_kn=None,
    body_bra_wfns=None,
    body_ket_wfns=None,
    _classes=None,
):
    """Contract the face wings, or on parents-only storage the parents' own.

    Parents-only storage (``wfns.green_parent``, no full-k faces): no child
    face is formed.  A child's density vertex is its parent's at the source
    centroid (conjugated, i.e. ``rho_ji``, on an antiunitary row: the spin
    action and the Bloch phase cancel in ``psi^dagger psi``) and its velocity
    is the polar time-odd image of its parent's (``symmetry_maps.
    unfold_file_wedge_polar_matrix``, conjugated on an antiunitary row), so
    ``Y_child = F Y_parent[perm(mu)]`` with the antiunitary rows' parent
    contraction reading the transposed pair weight.  Each parent is
    contracted once per operation class it feeds and the classes are
    transported once (:meth:`CentroidKUnfoldPlan.operation_classes`).
    """
    if (getattr(wfns, "layout", None) in ("face", "axis") and wfns.psi_mun is None
            and getattr(wfns, "green_parent", None) is not None):
        if body_bra_wfns is not None or body_ket_wfns is not None:
            raise ValueError(
                "head_wings_sharded(layout='face'): separately supplied "
                "endpoint bundles are not combined with parents-only storage.")
        carrier = wfns.green_parent
        classes = carrier.plan.operation_classes()
        v_all = jnp.asarray(velocity_cart, dtype=jnp.complex128)
        if int(v_all.shape[0]) != 3:
            raise ValueError(
                "GATE parent_head_wing_vertex: got a width-"
                f"{int(v_all.shape[0])} vertex on parents-only storage; want "
                "the width-3 Cartesian velocity; why: the parent transport "
                "rotates a polar time-odd vector, and a width-8 (a,I) jet has "
                "no typed action here.")
        r = device_put_process_local(
            classes.parent_rows, NamedSharding(mesh, P(None)))
        parents = _parent_face(carrier, wfns.slices)
        take = lambda a: None if a is None else jnp.take(
            jnp.asarray(a, dtype=jnp.float64), r, axis=0)
        Y_c, Z_c = _head_wings_sharded_face(
            jnp.take(v_all, r, axis=1), parents, take(energies_kn_ry),
            take(occupations_kn), omegas_ry, mesh=mesh, nb_logical=nb_logical,
            nk_tot=nk_tot, nspin=nspin, nspinor=nspinor, eta_ry=eta_ry,
            surface_weight_kn=take(surface_weight_kn), _classes=classes)
        mix = np.asarray(carrier.plan.sym.cartesian_action(
            classes.ops, axial=False, time_odd=True), dtype=np.float64)
        return _wing_transport(mesh)(
            Y_c, Z_c, jnp.asarray(classes.local_perm), jnp.asarray(mix))

    bra_wfns = wfns if body_bra_wfns is None else body_bra_wfns
    ket_wfns = wfns if body_ket_wfns is None else body_ket_wfns

    face_shapes = None
    for endpoint_name, endpoint in (
            ("wfns", wfns), ("body_bra_wfns", bra_wfns),
            ("body_ket_wfns", ket_wfns)):
        if getattr(endpoint, "layout", None) != wfns.layout:
            raise ValueError(
                f"head_wings_sharded(layout='face'): {endpoint_name}.layout "
                f"must be 'face', got {getattr(endpoint, 'layout', None)!r}")
        if endpoint.psi_mun is None or endpoint.psi_nmu is None:
            raise ValueError(
                f"head_wings_sharded(layout='face') requires "
                f"{endpoint_name}.psi_mun and {endpoint_name}.psi_nmu "
                "(got None)")
        nk_mun, s_mun, mu_x, n_mun = endpoint.psi_mun.shape
        nk_nmu, n_nmu, s_nmu, mu_y = endpoint.psi_nmu.shape
        shapes = (int(nk_mun), int(s_mun), int(mu_x), int(n_mun),
                  int(nk_nmu), int(n_nmu), int(s_nmu), int(mu_y))
        if nk_mun != nk_nmu or s_mun != s_nmu or n_mun != n_nmu:
            raise ValueError(
                f"head_wings_sharded(layout='face'): {endpoint_name} face "
                f"axes disagree: psi_mun={endpoint.psi_mun.shape}, "
                f"psi_nmu={endpoint.psi_nmu.shape}")
        if (tuple(endpoint.enk.shape) != (nk_mun, n_mun)
                or tuple(endpoint.occ.shape) != (nk_mun, n_mun)):
            raise ValueError(
                f"head_wings_sharded(layout='face'): {endpoint_name} "
                f"energy/occupation shapes {endpoint.enk.shape}/"
                f"{endpoint.occ.shape} do not match its face k/band axes "
                f"{(nk_mun, n_mun)}")
        if endpoint.slices != wfns.slices:
            raise ValueError(
                f"head_wings_sharded(layout='face'): {endpoint_name}.slices "
                "does not match wfns.slices")
        if face_shapes is None:
            face_shapes = shapes
        elif shapes != face_shapes:
            raise ValueError(
                f"head_wings_sharded(layout='face'): {endpoint_name} face "
                f"shapes {shapes} do not match wfns face shapes "
                f"{face_shapes}")

    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    omega = jnp.atleast_1d(jnp.asarray(omegas_ry, dtype=jnp.complex128))
    if (v.ndim != 4 or int(v.shape[0]) not in _HEAD_VERTEX_WIDTHS
            or v.shape[2] != v.shape[3]):
        raise ValueError(
            "velocity_cart must be (n_vertex,nk,nb,nb) with canonical "
            f"n_vertex in {_HEAD_VERTEX_WIDTHS}; got {v.shape}.")
    if e.shape != f.shape or tuple(e.shape) != tuple(v.shape[1:3]):
        raise ValueError(
            f"energy/occupation shapes {e.shape}/{f.shape} do not match "
            f"velocity (nk,nb)={v.shape[1:3]}.")
    nk_mun, s_mun, _mu_x, n_mun = wfns.psi_mun.shape
    nk_nmu, n_nmu, s_nmu, _mu_y = wfns.psi_nmu.shape
    if nk_mun != int(v.shape[1]) or nk_nmu != int(v.shape[1]):
        raise ValueError(
            "centroid wavefunction k axis does not match the velocity")
    if s_mun != s_nmu:
        raise ValueError(
            f"psi_mun/psi_nmu spinor axes disagree: {s_mun} vs {s_nmu}")
    if n_mun != n_nmu:
        raise ValueError(
            f"psi_mun/psi_nmu band extents disagree: {n_mun} vs {n_nmu}")
    nb_full = int(n_mun)
    if nb_full < int(v.shape[-1]):
        raise ValueError("centroid wavefunctions do not cover the head manifold")
    include_surface = surface_weight_kn is not None
    surface = (
        jnp.asarray(surface_weight_kn, dtype=jnp.float64)
        if include_surface else device_put_process_local(
            np.zeros(e.shape, np.float64), NamedSharding(mesh, P(None, None))))
    if surface.shape != e.shape:
        raise ValueError(
            f"surface_weight_kn shape {surface.shape} does not match {e.shape}.")
    v, e, f, surface = _pad_head_band_manifold_to(
        v, e, f, surface, mesh=mesh, width=nb_full)
    spin_denominator = (
        float(max(int(nspin), 1)) * float(max(int(nspinor), 1)))
    pref_inter = 4.0 / (float(nk_tot) * spin_denominator)
    pref_surface = 2.0 / (float(nk_tot) * spin_denominator)
    class_args = ()
    if _classes is not None:
        anti = np.asarray(_classes.antiunitary, dtype=bool)[:, None]
        rep = NamedSharding(mesh, P(None, None))
        class_args = tuple(device_put_process_local(
            np.where(mask, _classes.counts, 0.0).astype(np.complex128), rep)
            for mask in (~anti, anti))
    with_anti = _classes is not None and bool(np.any(_classes.antiunitary))
    mu_block = head_wing_mu_block(
        mu_local=-(-int(bra_wfns.psi_mun.shape[2]) // int(mesh.shape[_mesh_xy(mesh)[0]])),
        nk=int(bra_wfns.psi_mun.shape[0]), ns=int(bra_wfns.psi_mun.shape[1]),
        nb_full=nb_full, n_ends=4 if with_anti else 2)
    return _head_wing_kernel(
        mesh, nb_logical=int(nb_logical),
        include_surface=bool(include_surface), mu_block=mu_block, layout=wfns.layout,
        classes=0 if _classes is None else int(_classes.ops.size),
        anti=with_anti)(
            v, bra_wfns.psi_mun, ket_wfns.psi_mun,
            bra_wfns.psi_nmu, ket_wfns.psi_nmu,
            e, f, surface, omega,
            jnp.asarray(pref_inter, dtype=jnp.complex128),
            jnp.asarray(pref_surface, dtype=jnp.complex128),
            jnp.asarray(float(eta_ry), dtype=jnp.float64),
            *class_args,
        )


def _parent_face(carrier, slices):
    """The raw-parent carrier as a face bundle whose k axis is the parents."""
    from types import SimpleNamespace
    return SimpleNamespace(
        layout=carrier.layout, psi_mun=carrier.psi_mun, psi_nmu=carrier.psi_nmu,
        enk=carrier.enk, occ=carrier.occ, slices=slices)


def _wing_transport(mesh: Mesh) -> Callable:
    """``sum_c F_c . {Y,Z}_c[perm_c(mu)]`` on each rank's own centroid shard."""
    key = ("head_wing_transport", id(mesh))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    from gw.centroid_k_unfold import CentroidKUnfoldPlan
    move = CentroidKUnfoldPlan.transport_classes

    def _local(Y, Z, perm, mix):
        return (move(Y, perm, class_axis=2, mu_axis=3, mesh_axis="x",
                     mix=mix, mix_axis=1),
                move(Z, perm, class_axis=1, mu_axis=2, mesh_axis="y",
                     mix=mix, mix_axis=3))

    kernel = jax.jit(shard_map(
        _local, mesh=mesh,
        in_specs=(P(None, None, None, "x"), P(None, None, "y", None),
                  P(None, None), P()),
        out_specs=(P(None, None, "x"), P(None, "y", None)), check_vma=False))
    _KERNEL_CACHE[key] = kernel
    return kernel


def static_head_wings_sharded(
    wfns,
    surface_weight_kn,
    *,
    mesh: Mesh,
    nb_logical: int,
    nk_tot: int,
    nspin: int,
    nspinor: int,
):
    """Sum the static density vertex with minus the supplied negative occupation derivative."""
    if getattr(wfns, "layout", None) not in ("face", "axis"):
        raise ValueError("static_head_wings_sharded requires a canonical face or axis layout")
    return _static_head_wings_sharded_face(
        wfns, surface_weight_kn, mesh=mesh, nb_logical=nb_logical,
        nk_tot=nk_tot, nspin=nspin, nspinor=nspinor)


def _static_head_wings_kernel_face(mesh: Mesh, layout="face", classes: bool = False) -> Callable:
    """Cached shard_map kernel: a LOCAL density-weighted band sum per
    face orientation, then one ``psum`` over the mesh axis holding the
    summed band index.  No ring, no gather — see
    :func:`_static_head_wings_sharded_face`'s docstring for why the
    static vertex does not need one (it is diagonal in mu, unlike the
    dynamic wings' genuine (i,j) operator).  ``classes``: the weight is
    ``(C, n_parent, nb)`` per operation class and the outputs ``(C, mu)``."""
    key = ("static_head_wings_face", id(mesh), layout, bool(classes))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)
    from common.wfn_layout import psi_specs
    nmu_spec, mun_spec = psi_specs(layout)
    distributed_bands = int(layout == "face")
    sum_x = lambda value: value
    sum_y = lambda value: value
    if distributed_bands:
        sum_x = lambda value: jax.lax.psum(value, ax_x)
        sum_y = lambda value: jax.lax.psum(value, ax_y)

    def _band_slice(weight_full, coord, width):
        zero = jnp.zeros((), dtype=coord.dtype)
        start = coord * width * distributed_bands
        lead = weight_full.shape[:-1]
        return jax.lax.dynamic_slice(
            weight_full, (zero,) * len(lead) + (start,), lead + (width,))

    def _local(psi_mun_local, psi_nmu_local, weight_full):
        # ``classes``: one weight row block per operation class, summed per
        # class (``c``); otherwise the incumbent single k sum.
        n_y_local = psi_mun_local.shape[-1]
        weight_y = _band_slice(weight_full, jax.lax.axis_index(ax_y), n_y_local)
        density_x = jnp.sum(jnp.square(jnp.abs(psi_mun_local)), axis=1)
        left = sum_y(jnp.einsum("ckn,kmn->cm" if classes else "kn,kmn->m",
                                weight_y, density_x))

        n_x_local = psi_nmu_local.shape[1]
        weight_x = _band_slice(weight_full, jax.lax.axis_index(ax_x), n_x_local)
        density_y = jnp.sum(jnp.square(jnp.abs(psi_nmu_local)), axis=2)
        right = sum_x(jnp.einsum("ckn,knm->cm" if classes else "kn,knm->m",
                                 weight_x, density_y))
        return left, right

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            mun_spec,                  # psi_mun_local
            nmu_spec,                  # psi_nmu_local
            P() if classes else P(None, None),   # weight, replicated
        ),
        out_specs=((P(None, "x"), P(None, "y")) if classes
                   else (P("x"), P("y"))),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _static_head_wings_sharded_face(
    wfns,
    surface_weight_kn,
    *,
    mesh: Mesh,
    nb_logical: int,
    nk_tot: int,
    nspin: int,
    nspinor: int,
):
    """``layout='face'`` body of :func:`static_head_wings_sharded`.

    ``C_mu = (2/(Nk*nspin*nspinor)) sum_kn f'(E_kn)|psi_kn(mu)|^2`` is
    DIAGONAL in mu — no cross-mu (i,j) operator, unlike the dynamic
    wings — so no gather/ring is needed: each rank sums ``|psi|^2`` over
    the band-index fraction it already owns locally, then one ``psum``
    over the mesh axis holding that fraction completes the sum, exactly
    the report's own words ("local |psi|^2 weighted band sums followed
    by a psum").  Like the dynamic face wing, this pays the full
    ``nb_full``-wide sum rather than a windowed one (obstacle #3).
    """
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if surface.ndim != 2:
        raise ValueError(
            f"static head surface weights must be (nk,nb), got {surface.shape}")
    classes = plan = None
    if (wfns.psi_mun is None
            and getattr(wfns, "green_parent", None) is not None):
        # Parents-only storage: a child's |psi|^2 is its parent's at the
        # source centroid (the spin action and Bloch phase cancel, and an
        # antiunitary row conjugates a real density), so each parent's
        # density is weighted by its class members' summed weights and
        # transported once per operation class.
        plan = wfns.green_parent.plan
        classes = plan.operation_classes()
        wfns = _parent_face(wfns.green_parent, wfns.slices)
    if wfns.psi_mun is None or wfns.psi_nmu is None:
        raise ValueError(
            "static_head_wings_sharded(layout='face') requires "
            "wfns.psi_mun and wfns.psi_nmu (got None).")
    nk_mun, _s_mun, _mu_x, n_mun = wfns.psi_mun.shape
    _nk_nmu, n_nmu, _s_nmu, _mu_y = wfns.psi_nmu.shape
    if n_mun != n_nmu:
        raise ValueError(
            f"psi_mun/psi_nmu band extents disagree: {n_mun} vs {n_nmu}")
    nb_full = int(n_mun)
    if not (0 < int(nb_logical) <= nb_full):
        raise ValueError(f"need 0 < nb_logical <= {nb_full}, got {nb_logical}")
    if int(surface.shape[0]) != (nk_mun if plan is None else plan.n_full):
        raise ValueError("centroid wavefunctions do not cover static weights")
    width = int(surface.shape[1])
    if width > nb_full:
        raise ValueError(
            "static head surface weights are wider than the face bundle's "
            f"stored band extent: {width} > {nb_full}")
    if width < nb_full:
        surface = jnp.pad(surface, ((0, 0), (0, nb_full - width)))
    logical = jnp.arange(nb_full)[None, :] < int(nb_logical)
    weight = jnp.where(logical, surface, 0.0)
    if plan is not None:
        # E[c, p, n]: the summed weights of parent p's class-c rows.
        member = lambda labels, n: (
            np.asarray(labels)[None, :] == np.arange(n)[:, None]
        ).astype(np.float64)
        weight = jnp.einsum(
            "cr,pr,rn->cpn", member(classes.class_of_row, classes.ops.size),
            member(plan.irr_idx, plan.n_parent), weight)
    weight = device_put_process_local(
        weight, NamedSharding(mesh, P()))
    prefactor = -2.0 / (
        float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    left, right = _static_head_wings_kernel_face(
        mesh, layout=wfns.layout, classes=plan is not None)(
            wfns.psi_mun, wfns.psi_nmu, weight)
    if plan is not None:
        left, right = _static_wing_transport(mesh)(
            left, right, jnp.asarray(classes.local_perm))
    return prefactor * left, prefactor * right


def _static_wing_transport(mesh: Mesh) -> Callable:
    """``sum_c C_c[perm_c(mu)]`` on each rank's own centroid shard."""
    key = ("static_head_wing_transport", id(mesh))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    from gw.centroid_k_unfold import CentroidKUnfoldPlan
    move = CentroidKUnfoldPlan.transport_classes
    kernel = jax.jit(shard_map(
        lambda left, right, perm: (
            move(left, perm, class_axis=0, mu_axis=1, mesh_axis="x"),
            move(right, perm, class_axis=0, mu_axis=1, mesh_axis="y")),
        mesh=mesh, in_specs=(P(None, "x"), P(None, "y"), P(None, None)),
        out_specs=(P("x"), P("y")), check_vma=False))
    _KERNEL_CACHE[key] = kernel
    return kernel


def _drude_tensor_kernel(mesh: Mesh, *, nb_logical: int,
                         metal_split: bool = False) -> Callable:
    """Compile the Fermi-surface velocity contraction once per band shape.

    With ``metal_split`` each pair enters with its Fermi-surface share
    ``phi`` of ``fermi_surface.intraband_pair_fraction`` (the S kernel keeps
    ``1 - phi``), and the kernel also returns each state's intraband
    velocity spread ``sum_{m != n} phi_nm Re(v_nm^* v_nm^T)``,
    ``(nk, nb, 3, 3)`` sharded over x (``FermiSurfaceIntraband`` splits its
    atoms with it).
    """
    key = ("head_drude", id(mesh), int(nb_logical), bool(metal_split))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(v_local, e_x, e_y, s_x, s_y, prefactor, *split):
        nx, ny = v_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        logical = ((ix[:, None] < nb_logical)
                   & (iy[None, :] < nb_logical))[None, :, :]
        if metal_split:
            from gw.fermi_surface import intraband_pair_fraction
            d_x, d_y, moment = split
            share = jnp.where(logical, intraband_pair_fraction(
                v_local, e_x[:, :, None] - e_y[:, None, :], d_x, d_y,
                s_x, s_y, moment, TOL_DEGENERACY_RY), 0.0)
        else:
            share = (logical & (
                jnp.abs(e_x[:, :, None] - e_y[:, None, :]) < TOL_DEGENERACY_RY)
            ).astype(jnp.float64)
        weight = share * 0.5 * (s_x[:, :, None] + s_y[:, None, :])
        local = prefactor * jnp.einsum(
            "akij,kij,bkij->ab",
            jnp.conj(v_local), weight, v_local, optimize=True,
        )
        tensor = jax.lax.psum(local, (ax_x, ax_y))
        if not metal_split:
            return tensor
        partner = jnp.where((ix[:, None] != iy[None, :])[None], share, 0.0)
        spread = jnp.real(jnp.einsum(
            "akij,kij,bkij->kiab", jnp.conj(v_local),
            partner.astype(v_local.dtype), v_local, optimize=True))
        return tensor, jax.lax.psum(spread, ax_y)

    split_specs = ((P(None, None, "x"), P(None, None, "y"), P(None, None))
                   if metal_split else ())
    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, None, "x", "y"), P(None, "x"), P(None, "y"),
                  P(None, "x"), P(None, "y"), P()) + split_specs,
        out_specs=((P(None, None), P(None, "x", None, None)) if metal_split
                   else P(None, None)),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def head_drude_tensor_sharded(
    velocity_cart,
    surface_weight_kn,
    energies_kn_ry,
    *,
    mesh: Mesh,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    pair_split: "MetalPairSplit | None" = None,
    with_spread: bool = False,
):
    r"""Return the ab-initio Drude tensor ``D_ab`` in Rydberg units.

    .. math::
        D_{ab} = \frac{C}{\Omega N_k} \sum_k \sum_{nm:\,|E_n-E_m|<\delta}
        \bar w_{k,nm}\, v^{a*}_{k,nm} v^b_{k,nm},
        \qquad C = \frac{2}{n_{\rm spin} n_{\rm spinor}},

    the q -> 0, |z| >> v q limit of the intraband density response,
    ``chi = q.D.q / z^2``, so ``omega_p(qhat)^2 = 8 pi qhat.D.qhat`` (Ry^2)
    and free electrons give ``D = 2n``.  ``w`` is the Fermi-surface table
    (``Nk`` times the normalized-zone ``integral delta(E-mu)``) and
    ``wbar = (w_n + w_m)/2``.  The sum runs over each degenerate multiplet
    (``delta = TOL_DEGENERACY_RY``, BGW's TOL_Degeneracy), not only the
    diagonal: the trace ``sum_nm v_nm v_mn`` over a multiplet is invariant
    under rotations inside it, the diagonal alone is not, and those pairs are
    excluded from the interband ``S``.  The velocities include the nonlocal
    pseudopotential; in QSGW they are rotated into the current basis.  No
    fitted or experimental plasma frequency enters.

    ``pair_split`` (every metallic production head) weights every pair by
    its Fermi-surface share ``phi`` (``fermi_surface.intraband_pair_fraction``;
    an exact multiplet has ``phi = 1``);
    ``with_spread`` also returns each state's intraband velocity spread
    ``(nk, nb_storage, 3, 3)`` on the host.
    """
    if with_spread and pair_split is None:
        raise ValueError("the intraband velocity spread needs a metal pair split")
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    energies = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    if v.ndim != 4 or v.shape[0] != 3 or v.shape[2] != v.shape[3]:
        raise ValueError(
            f"velocity_cart must be (3,nk,nb,nb), got {v.shape}.")
    if tuple(surface.shape) != tuple(v.shape[1:3]) or (
            energies.shape != surface.shape):
        raise ValueError(
            f"surface_weight_kn {surface.shape} and energies "
            f"{energies.shape} must match velocity (nk,nb)={v.shape[1:3]}.")
    if not (0 < int(nb_logical) <= int(v.shape[2])):
        raise ValueError(
            f"need 0 < nb_logical <= stored nb, got "
            f"{nb_logical}, {v.shape[2]}.")
    v, energies, _f, surface = _pad_head_band_manifold(
        v, energies, surface, surface, mesh=mesh)
    pref = 2.0 / (
        float(cell_volume)
        * float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    split = () if pair_split is None else pair_split.operands(v)
    result = _drude_tensor_kernel(
        mesh, nb_logical=int(nb_logical), metal_split=pair_split is not None)(
        v, energies, energies, surface, surface,
        jnp.asarray(pref, dtype=jnp.complex128), *split)
    tensor = result if pair_split is None else result[0]
    tensor = 0.5 * (tensor + jnp.conj(tensor.T))
    if not with_spread:
        return tensor
    from common.collectives import gather_to_host
    return tensor, np.asarray(gather_to_host(result[1]))


@dataclass(frozen=True)
class MetalPairSplit:
    """Inputs of the metallic head's Taylor-radius pair split.

    ``diag`` is the real diagonal velocity ``(3, nk, nb_storage)`` on the
    padded head band carrier, replicated (``O(nk nb)`` bytes); ``moment`` the
    q = 0 cell's Coulomb-weighted second moment ``(3, 3)`` in bohr^-2
    (``vcoul.minibz_coulomb_moment``).  Every metallic head kernel (``S``,
    ``D`` and the four-current interband tensor) splits its pairs with
    ``fermi_surface.intraband_pair_fraction`` on these two objects.
    """

    diag: jax.Array
    moment: np.ndarray
    diag_host: np.ndarray

    def operands(self, v_padded):
        if int(self.diag.shape[-1]) != int(v_padded.shape[-1]):
            raise ValueError(
                "metal pair split was built on a different band carrier: "
                f"{self.diag.shape} vs velocity {v_padded.shape}")
        return (self.diag, self.diag,
                jnp.asarray(self.moment, dtype=jnp.float64))

    def host_diag(self, nb_logical):
        return self.diag_host[:, :, :int(nb_logical)]


def _velocity_diagonal_kernel(mesh: Mesh) -> Callable:
    key = ("head_velocity_diag", id(mesh))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(v_local):
        nx, ny = v_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        eq = (ix[:, None] == iy[None, :])[None, None]
        return jax.lax.psum(
            jnp.real(jnp.sum(jnp.where(eq, v_local, 0.0), axis=-1)), ax_y)

    kernel = jax.jit(shard_map(
        _local, mesh=mesh, in_specs=(P(None, None, "x", "y"),),
        out_specs=P(None, None, "x"), check_vma=False))
    _KERNEL_CACHE[key] = kernel
    return kernel


def metal_pair_split(velocity_cart, *, mesh: Mesh, bvec_cart, kgrid,
                     is_2d: bool = False):
    """Build the :class:`MetalPairSplit` of one velocity operator.

    ``is_2d`` reads the pair scale on the slab's in-plane polygon cell
    (``vcoul.minibz_coulomb_moment``)."""
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import minibz_coulomb_moment
    from common.collectives import gather_to_host

    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    e = jnp.zeros(v.shape[1:3], dtype=jnp.float64)
    v, _e, _f, _s = _pad_head_band_manifold(v, e, e, e, mesh=mesh)
    diag = np.asarray(gather_to_host(_velocity_diagonal_kernel(mesh)(v)))
    moment = minibz_coulomb_moment(np.asarray(bvec_cart, dtype=np.float64),
                                   tuple(int(n) for n in kgrid), is_2d=is_2d)
    replicated = device_put_process_local(diag, NamedSharding(mesh, P()))
    return MetalPairSplit(diag=replicated, moment=moment, diag_host=diag)


def head_s_tensor_sharded(
    velocity_cart,
    energies_kn_ry,
    occupations_kn,
    omegas_ry,
    *,
    mesh: Mesh,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    eta_ry: float = 0.0,
    surface_weight_kn=None,
    pair_split: "MetalPairSplit | None" = None,
):
    """Build interband plus optional Drude ``S(omega)`` from current velocity.

    The initial call uses the saved DFT operator.  Later self-consistent calls
    use its covariantly updated and rotated counterpart.

    The contraction runs over every pair in ``[0, nb_logical)`` and uses
    the signed factor ``f_nk - f_mk``.  There is deliberately no integer
    occupied-band boundary in this API.  If ``surface_weight_kn`` is supplied,
    the diagonal-velocity Fermi-surface tensor is added as
    ``D/(omega+i*eta)^2``.  This is the dynamic q->0 intraband limit; the
    strictly static metallic limit has a different order of limits.

    Energies and occupations are passed twice with complementary one-axis
    shardings.  Each rank forms only its local conduction-by-valence tile;
    a two-axis psum reduces the final operator-axis tensor.  The ordinary
    path passes three Cartesian velocities.  A static-gauge producer instead
    flattens its canonical energy-scaled jet
    ``P^(I,a)=-DeltaE*d_q_a M^I|_0`` over ``(a,I)=(2,4)``.  Passing the
    literal transition derivative would be wrong by two powers of the
    interband energy in this kernel's bilinear.  This width-eight contraction
    owns only the first-derivative/first-derivative term; the producer must add
    the independently derived response-weight, second-jet, and contact terms
    before calling a result complete.  No other width is admitted, so this
    shared kernel has exactly the incumbent width-three and packed width-eight
    executable shapes.
    """
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    omega = jnp.atleast_1d(jnp.asarray(omegas_ry, dtype=jnp.complex128))
    if v.ndim != 4 or int(v.shape[0]) not in _HEAD_VERTEX_WIDTHS:
        raise ValueError(
            "velocity_cart must be (n_vertex,nk,nb,nb) with canonical "
            f"n_vertex in {_HEAD_VERTEX_WIDTHS}; got {v.shape}.")
    if e.shape != f.shape or tuple(e.shape) != tuple(v.shape[1:3]):
        raise ValueError(
            f"energy/occupation shapes {e.shape}/{f.shape} do not match "
            f"velocity (nk,nb)={v.shape[1:3]}."
        )
    if v.shape[2] != v.shape[3]:
        raise ValueError("velocity band matrices must be square.")
    if not (0 < int(nb_logical) <= int(v.shape[2])):
        raise ValueError(
            f"need 0 < nb_logical <= stored nb, got "
            f"{nb_logical}, {v.shape[2]}."
        )
    include_surface = surface_weight_kn is not None
    surface = (
        jnp.asarray(surface_weight_kn, dtype=jnp.float64)
        if include_surface else jnp.zeros_like(e))
    if surface.shape != e.shape:
        raise ValueError(
            f"surface_weight_kn shape {surface.shape} does not match {e.shape}.")
    v, e, f, surface = _pad_head_band_manifold(
        v, e, f, surface, mesh=mesh)
    pref = 4.0 / (
        float(cell_volume)
        * float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    metal_split = include_surface and pair_split is not None
    split = ()
    if metal_split:
        d_x, d_y, moment = pair_split.operands(v)
        split = (d_x, d_y, surface, surface, moment)
    interband = _s_tensor_kernel(mesh, nb_logical=int(nb_logical),
                                 metal_split=metal_split)(
        v,
        e,
        e,
        f,
        f,
        omega,
        jnp.asarray(pref, dtype=jnp.complex128),
        jnp.asarray(float(eta_ry), dtype=jnp.float64),
        *split,
    )
    if not include_surface:
        return interband
    if int(v.shape[0]) != 3:
        raise ValueError(
            "packed energy-scaled transition jets do not yet have a derived "
            "metallic Drude completion; surface_weight_kn is admitted only "
            "for the incumbent three-Cartesian-velocity path")
    drude = head_drude_tensor_sharded(
        v,
        surface,
        e,
        mesh=mesh,
        nb_logical=int(nb_logical),
        cell_volume=float(cell_volume),
        nk_tot=int(nk_tot),
        nspin=int(nspin),
        nspinor=int(nspinor),
        pair_split=pair_split,
    )
    z = omega + 1j * jnp.asarray(float(eta_ry), dtype=jnp.float64)
    # The exact static metallic limit is Thomas-Fermi, not the omega->0
    # value of the dynamic Drude expression.  Leave an exact zero-frequency
    # slot untouched here; ``head_samples_from_s`` replaces that slot with
    # the separately averaged TF model when surface weights are present.
    inv_z2 = jnp.where(
        jnp.abs(z) > 1.0e-15,
        1.0 / jnp.square(z),
        jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
    )
    return interband + drude[None, :, :] * inv_z2[:, None, None]


def _raw_hall_kernel(mesh: Mesh, *, nb_logical: int) -> Callable:
    """Distributed occupied-state Berry-overlap contraction.

    This is the band-tiled form of ``orbital_magnetization.cB``.  It returns
    the axial cross product before physical prefactors; no band matrix is
    gathered and only the three-component reduction is replicated.
    """
    key = ("static_gauge_raw_hall", id(mesh), int(nb_logical))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(gamma_local, e_bra, e_ket, f_bra, f_ket, deps_tol):
        nx, ny = gamma_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        logical = (
            (ix[:, None] < nb_logical)
            & (iy[None, :] < nb_logical)
        )[None, :, :]
        dE = e_bra[:, :, None] - e_ket[:, None, :]
        separated = jnp.abs(dE) > deps_tol
        inv_dE2 = jnp.where(
            logical & separated,
            1.0 / jnp.square(jnp.where(separated, dE, 1.0)),
            0.0,
        )
        weight = f_bra[:, :, None] * inv_dE2

        gx, gy, gz = gamma_local
        # Hermiticity gives Gamma_b[m,n] = conj(Gamma_b[n,m]), avoiding a
        # transpose/all-to-all of the band tile.  This is exactly the axial
        # product used by psp.orbital_magnetization.orbital_pieces_at_k.
        cross = jnp.stack((
            gy * jnp.conj(gz) - gz * jnp.conj(gy),
            gz * jnp.conj(gx) - gx * jnp.conj(gz),
            gx * jnp.conj(gy) - gy * jnp.conj(gx),
        ))
        cB_raw = jnp.einsum("akij,kij->a", cross, weight, optimize=True)

        # A degeneracy joining differently occupied states invalidates the
        # ordinary insulating SOS expression; report one small flag rather
        # than silently clipping it into a Hall number.
        unsafe = jnp.any(
            logical
            & (~separated)
            & (jnp.abs(f_bra[:, :, None] - f_ket[:, None, :]) > 1.0e-12))
        return (
            jax.lax.psum(cB_raw, (ax_x, ax_y)),
            jax.lax.psum(unsafe.astype(jnp.int32), (ax_x, ax_y)),
        )

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),
            P(None, "x"),
            P(None, "y"),
            P(None, "x"),
            P(None, "y"),
            P(),
        ),
        out_specs=(P(None), P()),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def raw_hall_pseudovector_sharded(
    gamma_raw,
    energies_kn_ry,
    occupations_kn,
    *,
    mesh: Mesh,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor_wfn: int,
    degeneracy_tolerance_ry: float = 1.0e-10,
):
    r"""Derive the schema's real raw Hall pseudovector from ``Gamma_raw``.

    The accepted raw Breit vertex and physical Pauli velocity are

    ``Gamma_raw = (alpha_FS/2) v_Ry`` and
    ``j = c Gamma_raw = v_Ry/2``.

    Let ``cB`` be the incumbent orbital-magnetization Berry overlap built
    from ``v_Ry``.  With state capacity
    ``C=2/(nspin*nspinor_wfn)``, the immutable-schema convention is

    ``sigma_H_raw = -(alpha_FS*C/(2*Omega_cell)) Im(cB)``.

    This implementation contracts the same transaction's ``Gamma_raw``
    directly, so ``cB_raw=(alpha_FS/2)^2 cB`` and the applied prefactor is
    equivalently ``-C/(Omega_cell*Nk*(alpha_FS/2))``.  The minus sign is the
    documented occupied-Berry/Hall sign; ``static_hall_linear_response``
    later inserts ``CT[a,i] = -i epsilon[b,a,i] sigma_H[b]`` -- the MINUS
    is not independent of this one.  The live Adler--Wiser response
    energy-orders the bra and conjugates the row (``P = -Delta*D``), so its
    linear CT imaginary part is the negative of the occupied-bra Berry
    tensor stored here (``cacc4e07``; oracles
    ``tests/test_qsgw_parallel_transport_head.py::
    test_raw_hall_matches_orbital_cB_owner_and_documented_sign`` and
    ``tests/test_photon_head_sign_oracle.py::
    test_part_a_definitions_share_one_convention``).
    """
    from common.bispinor_init import HALFALPHA

    gamma = jnp.asarray(gamma_raw, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    if gamma.ndim != 4 or gamma.shape[1] != 3:
        raise ValueError(
            "gamma_raw must be (nk,3,nb,nb) from "
            f"sweep_uniform_current_matrix_elements; got {gamma.shape}")
    if gamma.shape[2] != gamma.shape[3]:
        raise ValueError("gamma_raw band matrices must be square")
    if e.shape != f.shape or tuple(e.shape) not in (
            (int(gamma.shape[0]), int(nb_logical)),
            (int(gamma.shape[0]), int(gamma.shape[2]))):
        raise ValueError(
            f"energy/occupation shapes {e.shape}/{f.shape} do not match "
            "the logical or stored band extent of gamma_raw "
            f"{gamma.shape} (nb_logical={int(nb_logical)})")
    if not (0 < int(nb_logical) <= int(gamma.shape[2])):
        raise ValueError(
            f"need 0 < nb_logical <= stored nb={gamma.shape[2]}; "
            f"got {nb_logical}")
    if not np.isfinite(float(cell_volume)) or float(cell_volume) <= 0.0:
        raise ValueError("cell_volume must be positive")
    if int(nk_tot) <= 0 or int(nspin) <= 0 or int(nspinor_wfn) <= 0:
        raise ValueError("nk_tot, nspin, and nspinor_wfn must be positive")
    if int(gamma.shape[0]) != int(nk_tot):
        raise ValueError(
            "raw Hall requires one Gamma_raw row per full-BZ k point: "
            f"gamma_raw nk={int(gamma.shape[0])}, nk_tot={int(nk_tot)}. "
            "IBZ/subset rows must be unfolded through the symmetry service "
            "before the 1/Nk normalization is applied")
    if float(degeneracy_tolerance_ry) <= 0.0:
        raise ValueError("degeneracy_tolerance_ry must be positive")

    # ``sweep_uniform_current_matrix_elements`` stores both band axes at the
    # mesh-divisible carrier extent, while WFN energies/occupations are the
    # logical file manifold.  Use the repository padding owner rather than
    # forcing a producer to manufacture padded electronic states.  The
    # kernel's explicit ``nb_logical`` mask makes these storage rows inert.
    if int(e.shape[1]) != int(gamma.shape[2]):
        from runtime.padding import pad_axis
        e = pad_axis(e, int(gamma.shape[2]), axis=1).array
        f = pad_axis(f, int(gamma.shape[2]), axis=1).array

    # Reuse the incumbent head manifold's one padding/sharding owner.  The
    # transpose is a view putting the replicated component axis first.
    vertex = jnp.transpose(gamma, (1, 0, 2, 3))
    vertex, e, f, _ = _pad_head_band_manifold(
        vertex, e, f, jnp.zeros_like(e), mesh=mesh)
    cB_raw, unsafe = _raw_hall_kernel(
        mesh, nb_logical=int(nb_logical))(
            vertex,
            e,
            e,
            f,
            f,
            jnp.asarray(float(degeneracy_tolerance_ry), dtype=jnp.float64),
        )
    if int(np.asarray(unsafe)):
        raise ValueError(
            "GATE static_gauge_raw_hall_degenerate: differently occupied "
            "states are degenerate within degeneracy_tolerance_ry; the "
            "insulating occupied-state Berry SOS formula is undefined")
    capacity = 2.0 / (float(nspin) * float(nspinor_wfn))
    prefactor = -capacity / (
        float(cell_volume) * float(nk_tot) * float(HALFALPHA))
    return jnp.asarray(
        prefactor * jnp.imag(cB_raw), dtype=jnp.float64)


def static_gauge_hall_transaction(
    uniform_gauge,
    *,
    wfn,
    sym,
    band_start: int,
    band_stop: int,
    mesh: Mesh,
    degeneracy_tolerance_ry: float = 1.0e-10,
) -> StaticGaugeHallTransaction:
    r"""Produce the artifact-ready Hall term from one canonical transaction.

    ``uniform_gauge`` must be the result of
    :func:`common.mtxel_sweep.sweep_uniform_current_matrix_elements`.  Hall
    production consumes its current block.  (The complete sweep,
    ``sweep_uniform_gauge_matrix_elements``, was deleted on 2026-09-02 with
    the rest of the stranded FULL-seam producers; the ``complete`` branch
    below is therefore unreachable and registered in
    KNOWN_LORRAX_ISSUES.md rather than removed inside a dead-code commit
    that must not touch a live producer's contract.)  Exact contact and optional
    transfer-q1/q2 fields are validated when present, but they are not
    materialized merely to reduce Hall: on a realistic band manifold that
    would make a three-number reduction retain many unrelated band matrices.
    A complete response producer must separately require those fields under
    its own capability gate; the charge+Hall model records them as omitted.

    Energies and occupations are read from the same ``WfnLoader`` and unfolded
    from its file wedge through :func:`symmetry_maps.unfold_file_wedge_to_full_bz`.
    Consequently ``Gamma_raw``, energies and occupations all have one row per
    physical full-BZ k before the sole ``1/Nk`` normalization is applied.  No
    driver-local star reconstruction, wavefunction reopen, band-matrix gather,
    FFT, current operator, or second Hall contraction is introduced here.
    """
    from common.mtxel_sweep import (
        UniformGaugeCurrentMatrixElements, UniformGaugeMatrixElements)
    from common.parallel_transport import wfn_fingerprint
    from symmetry_maps import unfold_file_wedge_to_full_bz

    if not isinstance(uniform_gauge, (
            UniformGaugeCurrentMatrixElements, UniformGaugeMatrixElements)):
        raise TypeError(
            "static gauge Hall production requires the canonical "
            "uniform-gauge current transaction")
    complete = isinstance(uniform_gauge, UniformGaugeMatrixElements)

    start, stop = int(band_start), int(band_stop)
    logical = stop - start
    if start != 0 or logical <= 0 or stop > int(wfn.nbands):
        raise ValueError(
            "static gauge Hall band interval must start at band zero and "
            f"satisfy 0 < stop <= WFN.nbands; got [{start},{stop})")
    if int(wfn.nspin) != 1:
        raise ValueError(
            "static gauge Hall transaction currently requires nspin=1: "
            "Gamma_raw has no explicit spin-channel axis")

    gamma = uniform_gauge.gamma_raw
    nk_tot = int(sym.nk_tot)
    if (gamma.ndim != 4 or int(gamma.shape[0]) != nk_tot
            or int(gamma.shape[1]) != 3
            or int(gamma.shape[2]) != int(gamma.shape[3])
            or int(gamma.shape[2]) < logical):
        raise ValueError(
            "canonical static gauge Hall transaction requires full-BZ "
            "Gamma_raw[nk,3,nb,nb] with both band carriers covering the "
            f"logical interval: got {gamma.shape}, nk_tot={nk_tot}, "
            f"logical bands={logical}")
    storage = int(gamma.shape[2])
    if (complete and tuple(uniform_gauge.lambda_raw.shape) != (
            nk_tot, 3, 3, storage, storage)):
        raise ValueError(
            "uniform-gauge Hall transaction has an invalid exact-contact "
            f"shape {uniform_gauge.lambda_raw.shape}")

    fingerprint = str(
        uniform_gauge.hamiltonian_config_operator_fingerprint).strip()
    if (not fingerprint.startswith("sha256:")
            or len(fingerprint) != len("sha256:") + 64
            or any(c not in "0123456789abcdef" for c in fingerprint[7:])):
        raise ValueError(
            "uniform-gauge Hall transaction lacks the canonical "
            "Hamiltonian/config/operator SHA-256 fingerprint")

    energies_file = np.asarray(
        wfn.energies[0, :, start:stop], dtype=np.float64)
    occupations_file = np.asarray(
        wfn.occs[0, :, start:stop], dtype=np.float64)
    if (energies_file.shape != (int(sym.nk_red), logical)
            or occupations_file.shape != energies_file.shape):
        raise ValueError(
            "WFN energy/occupation file-wedge tables do not match the "
            f"requested Hall manifold: {energies_file.shape}/"
            f"{occupations_file.shape}, expected "
            f"{(int(sym.nk_red), logical)}")
    if np.any((occupations_file != 0.0) & (occupations_file != 1.0)):
        raise ValueError(
            "static gauge Hall artifact production is insulating-only and "
            "requires exact 0/1 occupations")
    occupations_above = np.asarray(
        wfn.occs[0, :, stop:], dtype=np.float64)
    if np.any(occupations_above != 0.0):
        raise ValueError(
            "static gauge Hall band interval omits occupied WFN states; "
            "increase band_stop")

    energies_full = unfold_file_wedge_to_full_bz(sym, energies_file)
    occupations_full = unfold_file_wedge_to_full_bz(sym, occupations_file)
    sigma_H = raw_hall_pseudovector_sharded(
        gamma,
        energies_full,
        occupations_full,
        mesh=mesh,
        nb_logical=logical,
        cell_volume=float(wfn.cell_volume),
        nk_tot=nk_tot,
        nspin=int(wfn.nspin),
        nspinor_wfn=int(wfn.nspinor),
        degeneracy_tolerance_ry=float(degeneracy_tolerance_ry),
    )
    return StaticGaugeHallTransaction(
        sigma_H=sigma_H,
        hamiltonian_config_operator_fingerprint=fingerprint,
        wfn_fingerprint=wfn_fingerprint(wfn),
        band_start=start,
        band_stop=stop,
        nk_tot=nk_tot,
        producer_id=_STATIC_GAUGE_HALL_PRODUCER_ID,
        _producer_token=_STATIC_GAUGE_HALL_TOKEN,
    )


def _static_gauge_hall_transaction_from_artifact(
    *, sigma_H, hamiltonian_config_operator_fingerprint: str,
    wfn_fingerprint: str, band_start: int, band_stop: int, nk_tot: int,
    mesh: Mesh,
) -> StaticGaugeHallTransaction:
    """Place a loader-validated Hall vector on the run mesh."""
    sigma = device_put_process_local(
        np.asarray(sigma_H, dtype=np.float64),
        NamedSharding(mesh, P()))
    return StaticGaugeHallTransaction(
        sigma_H=sigma,
        hamiltonian_config_operator_fingerprint=(
            hamiltonian_config_operator_fingerprint),
        wfn_fingerprint=wfn_fingerprint,
        band_start=int(band_start),
        band_stop=int(band_stop),
        nk_tot=int(nk_tot),
        producer_id=_STATIC_GAUGE_HALL_PRODUCER_ID,
        _producer_token=_STATIC_GAUGE_HALL_TOKEN,
    )


@dataclass(frozen=True)
class IterationHeadResponse:
    """Direct head and centroid-sharded wings before the body Schur fold."""

    omegas: tuple[complex, ...]
    S_direct: jax.Array
    Y_x: jax.Array | None
    Z_y: jax.Array | None
    static_kappa2_bohr2: float | None
    static_Y_x: jax.Array | None
    static_Z_y: jax.Array | None
    static_chi_body_gamma: jax.Array | None
    sigma_energies_ry: np.ndarray
    sigma_occupations: np.ndarray
    efermi_ry: float
    #: Metals only: the Fermi-surface (Drude) tensor ``D_ab`` in the S
    #: convention, ``omega_p(qhat)^2 = 8 pi qhat.D.qhat`` (Ry^2).  Reported
    #: by the drivers so every metallic log carries its plasma frequency.
    drude_tensor: np.ndarray | None = None
    #: Metals only: the velocity atoms of the intraband response
    #: (``fermi_surface.FermiSurfaceIntraband``), whose moments are ``N0``
    #: and ``drude_tensor``; the q = 0 cell evaluates it at every sample.
    fermi_surface: object | None = None
    #: ``(names, shares[n_terms, 3], link_bound, sigma_zeroed)`` from
    #: :func:`velocity_term_shares`, :func:`link_correction_bound` and
    #: :func:`sigma_term_zeroed` (the reason, or None), for the per-map head
    #: block (``sc_iteration._record_head_block``).
    velocity_terms: tuple | None = None


@dataclass(frozen=True)
class IterationHeadSamples:
    """Per-iteration q=0 samples plus the matching active QP spectrum."""

    omegas: tuple[complex, ...]
    samples: tuple[object, ...]
    sigma_energies_ry: np.ndarray
    sigma_occupations: np.ndarray
    efermi_ry: float

    def at(self, omega):
        z = complex(omega)
        for known, sample in zip(self.omegas, self.samples):
            if abs(z - known) <= 1.0e-12:
                return sample
        raise KeyError(
            f"QSGW iteration head has no sample at omega={z} Ry; "
            f"available={self.omegas}."
        )


def _metal_intraband(response):
    """The metallic head's intraband atoms, ``None`` on an insulator."""
    if response.drude_tensor is None:
        return None
    if response.fermi_surface is None:
        raise ValueError(
            "metallic head response carries a Drude tensor but no Fermi-surface "
            "intraband model (gw.qsgw_head.metal_intraband_model)")
    return response.fermi_surface


def metal_intraband_model(velocity_cart, surface_weight_kn, energies_kn_ry, *,
                          mesh: Mesh, nb_logical: int, cell_volume: float,
                          nk_tot: int, nspin: int, nspinor: int, bvec_cart,
                          kgrid, is_2d: bool = False):
    """``(D, atoms, split)`` of a metal: one pair split, one surface table.

    The split (:func:`metal_pair_split`) is built once from this velocity and
    handed to every kernel that cuts pairs; ``D`` and each state's intraband
    velocity spread come from the same masked contraction, and the atoms
    (``fermi_surface.FermiSurfaceIntraband``) reproduce ``Re D`` and ``N0``.
    """
    from gw.fermi_surface import FermiSurfaceIntraband

    split = metal_pair_split(velocity_cart, mesh=mesh, bvec_cart=bvec_cart,
                             kgrid=kgrid, is_2d=is_2d)
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    drude, spread = head_drude_tensor_sharded(
        velocity_cart, surface, energies_kn_ry, mesh=mesh,
        nb_logical=int(nb_logical), cell_volume=float(cell_volume),
        nk_tot=int(nk_tot), nspin=int(nspin), nspinor=int(nspinor),
        pair_split=split, with_spread=True)
    drude = np.asarray(drude)
    capacity = 2.0 / (float(max(int(nspin), 1)) * float(max(int(nspinor), 1)))
    atoms = FermiSurfaceIntraband(
        np.asarray(surface_weight_kn, dtype=np.float64)[:, :int(nb_logical)],
        split.diag_host, spread, drude, capacity=capacity,
        cell_volume=float(cell_volume), nk_tot=int(nk_tot))
    # The physical Drude tensor is the phi -> 0 (vanishing cell) limit: only
    # exact multiplets are intraband.  The cell-effective D above approaches
    # it as the cell shrinks (docs/theory/metal-q0-head.md section 2).
    physical = dataclasses.replace(split, moment=np.zeros_like(split.moment))
    atoms.physical_drude = np.asarray(head_drude_tensor_sharded(
        velocity_cart, surface, energies_kn_ry, mesh=mesh,
        nb_logical=int(nb_logical), cell_volume=float(cell_volume),
        nk_tot=int(nk_tot), nspin=int(nspin), nspinor=int(nspinor),
        pair_split=physical))
    return drude, atoms, split


def plasma_frequencies_ev(drude_tensor):
    """Principal ``omega_p = sqrt(8 pi eig(Re D))`` in eV of a Drude tensor."""
    from common import RYD_TO_EV
    D = np.real(np.asarray(drude_tensor))
    return np.sqrt(np.maximum(8.0 * np.pi * np.linalg.eigvalsh(0.5 * (D + D.T)),
                              0.0)) * RYD_TO_EV


def drude_offdiagonal(drude_tensor) -> float:
    """``max |Re D_ab| (a != b) / max |Re D_aa|``: zero on a cubic or
    tetragonal (m || z) crystal, whose point group forbids every
    off-diagonal; a link stencil not closed under that group shows here."""
    D = np.real(np.asarray(drude_tensor))
    off = np.abs(D - np.diag(np.diag(D)))
    return float(np.max(off) / max(float(np.max(np.abs(np.diag(D)))), 1.0e-300))


def drude_report(atoms) -> str:
    """Physical and cell-effective plasma frequencies of one metal head.

    Each prints the principal values (ascending eigenvalues) of the Drude
    tensor, not its x/y/z components; the head block prints the diagonal.
    ``offdiag`` is :func:`drude_offdiagonal` of the physical tensor.
    """
    return ("omega_p physical principal (exact multiplets, phi -> 0) = "
            + "/".join(f"{x:.6f}" for x in plasma_frequencies_ev(atoms.physical_drude))
            + f" eV (offdiag {drude_offdiagonal(atoms.physical_drude):.2e})"
            + "; q=0-cell effective principal (Fermi-surface share phi of near pairs) = "
            + "/".join(f"{x:.4f}" for x in plasma_frequencies_ev(atoms.drude_tensor))
            + " eV")


def _fold_static_kappa2(response, W_body_gamma, cell_volume, mesh):
    """Return kappa_eff^2 after the scalar static wing/body/wing fold."""
    if response.static_kappa2_bohr2 is None:
        return None
    if W_body_gamma is None:
        return response.static_kappa2_bohr2
    if response.static_Y_x is None or response.static_Z_y is None:
        raise ValueError(
            "static body-screened head requested without static density wings")
    from gw.head_correction import fold_cartesian_head_wings_sharded
    direct = jnp.asarray(
        [[-float(response.static_kappa2_bohr2) / (8.0 * np.pi)]],
        dtype=jnp.complex128,
    )
    effective = fold_cartesian_head_wings_sharded(
        direct,
        response.static_Y_x[None, :],
        W_body_gamma,
        response.static_Z_y[:, None],
        float(cell_volume),
        mesh_xy=mesh,
    )[0, 0]
    value = complex(np.asarray(effective))
    scale = max(abs(value.real), 1.0)
    if abs(value.imag) > 1.0e-8 * scale:
        raise ValueError(
            "static Schur effective head is not real: "
            f"f00_eff={value!r}")
    kappa2 = -8.0 * np.pi * value.real
    if not np.isfinite(kappa2) or kappa2 <= 0.0:
        raise ValueError(
            "static Schur fold produced nonphysical screening: "
            f"kappa_eff^2={kappa2!r}")
    return float(kappa2)


def finalize_iteration_head_sample(
    response: IterationHeadResponse,
    omega_index: int,
    W_body_gamma=None,
    *,
    wfn,
    meta,
    config,
    mesh: Mesh,
):
    r"""Finalize one response frequency while its total body W is resident.

    This is the disk-bounded MPA seam: the caller passes total screened
    W_body_gamma, never Wc, and only the replicated 3x3 Schur result
    survives the call. Left and right wings remain independent at complex
    frequency.
    """
    terms = iteration_head_sample_terms(
        response, omega_index, W_body_gamma, meta=meta, config=config, mesh=mesh)
    if not isinstance(terms, dict):
        return terms
    return finalize_iteration_head_rows(
        response, [terms], wfn=wfn, meta=meta, config=config)[0]


def finalize_iteration_head_rows(response, terms, *, wfn, meta, config):
    """Mini-BZ-average several :func:`iteration_head_sample_terms` rows in one call.

    Row ``i`` equals ``finalize_iteration_head_sample`` of its frequency bit
    for bit; the rows share one mini-BZ draw (``head_samples_from_s``).  The
    rows of one call share their fold policy; only a zero-frequency row reads
    its static kappa^2.
    """
    if not terms:
        return ()
    if len({(t["response_kind"], t["source_prefix"]) for t in terms}) != 1:
        raise ValueError("head rows of one call must share their fold policy")
    static = [t["static_kappa2"] for t in terms if abs(t["omega"]) <= 1.0e-14]
    return head_samples_from_s(
        np.stack([np.asarray(t["S_effective"], dtype=np.complex128) for t in terms]),
        tuple(t["omega"] for t in terms),
        wfn=wfn,
        meta=meta,
        config=config,
        static_kappa2_bohr2=static[0] if static else None,
        intraband=_metal_intraband(response),
        response_kind=terms[0]["response_kind"],
        source_prefix=terms[0]["source_prefix"],
    )


def iteration_head_sample_terms(
    response: IterationHeadResponse,
    omega_index: int,
    W_body_gamma=None,
    *,
    meta,
    config,
    mesh: Mesh,
):
    """One frequency's replicated 3x3 head tensor, folded while ``W_body_gamma`` is resident.

    Returns the ``head_correction = off`` sample itself, or a dict of the
    operands :func:`finalize_iteration_head_rows` averages.  Only the 3x3
    result outlives the call.
    """
    from gw.gw_config import HeadCorrection, coerce_head_correction

    policy = coerce_head_correction(
        getattr(config.head, "correction", HeadCorrection.FULL))
    index = int(omega_index)
    if not 0 <= index < len(response.omegas):
        raise IndexError(
            f"head frequency index {index} outside [0,{len(response.omegas)})")
    if policy is HeadCorrection.OFF:
        from gw.head_correction import HeadResponseKind, HeadSample
        return HeadSample(
            vc0=0.0j, wcoul0=0.0j, source="head_correction=off",
            omega=response.omegas[index], S_cart=None,
            response_kind=HeadResponseKind.OFF)
    S_effective = response.S_direct[index]
    use_fold = (
        policy is HeadCorrection.FULL and W_body_gamma is not None)
    if use_fold:
        if response.Y_x is None or response.Z_y is None:
            raise ValueError(
                "body-screened QSGW head requested without head/body wings")
        W = jnp.asarray(W_body_gamma)
        if (
            int(W.shape[-2]) != int(response.Y_x.shape[-1])
            or int(W.shape[-1]) != int(response.Z_y.shape[-2])
        ):
            raise ValueError(
                "QSGW head-wing centroid extents do not match W(Gamma): "
                f"Y={response.Y_x.shape}, W={W.shape}, Z={response.Z_y.shape}")
        from gw.head_correction import fold_cartesian_head_wings_sharded
        S_effective = fold_cartesian_head_wings_sharded(
            response.S_direct[index],
            response.Y_x[index],
            W,
            response.Z_y[index],
            float(meta.cell_volume),
            mesh_xy=mesh,
        )
    intraband = _metal_intraband(response)
    static_kappa2 = None if intraband is not None else response.static_kappa2_bohr2
    if use_fold and abs(response.omegas[index]) <= 1.0e-14:
        static_kappa2 = _fold_static_kappa2(
            response, W_body_gamma, float(meta.cell_volume), mesh)
    return dict(
        omega=complex(response.omegas[index]),
        S_effective=np.asarray(S_effective, dtype=np.complex128),
        static_kappa2=static_kappa2,
        response_kind=("full_local_fields" if use_fold
                       else "direct_irreducible"),
        source_prefix=("head_schur" if use_fold else "head_direct"),
    )


def finalize_iteration_head_samples(
    response: IterationHeadResponse,
    *,
    wfn,
    meta,
    config,
    mesh: Mesh,
    requests=None,
    W_by_role=None,
) -> IterationHeadSamples:
    """Apply the optional body Schur fold and mini-BZ-average the head.

    ``W_by_role`` is the already-screened finite-G/centroid body returned by
    :func:`gw.screening.compute_screening`.  Flat q index zero is Gamma in
    the production C-order convention, and its singular head channel is
    absent, so ``W_by_role[role][0]`` is precisely the body operand required
    by the bordered-Dyson reduction.

    Passing no ``W_by_role`` intentionally produces the direct-head result.
    This keeps the one-shot diagnostic API and X-only path unchanged.
    """
    from gw.gw_config import HeadCorrection, coerce_head_correction

    policy = coerce_head_correction(
        getattr(config.head, "correction", HeadCorrection.FULL))
    if policy is HeadCorrection.OFF:
        from gw.head_correction import HeadResponseKind, HeadSample
        samples = tuple(
            HeadSample(
                vc0=0.0j, wcoul0=0.0j, source="head_correction=off",
                omega=z, S_cart=None,
                response_kind=HeadResponseKind.OFF)
            for z in response.omegas)
        return IterationHeadSamples(
            omegas=response.omegas, samples=samples,
            sigma_energies_ry=response.sigma_energies_ry,
            sigma_occupations=response.sigma_occupations,
            efermi_ry=response.efermi_ry)
    S_effective = response.S_direct
    use_fold = policy is HeadCorrection.FULL and bool(W_by_role)
    if use_fold:
        if response.Y_x is None or response.Z_y is None:
            raise ValueError(
                "body-screened QSGW head requested without head/body wings")
        if requests is None:
            raise ValueError("screening requests are required to match W roles")
        reqs = tuple(requests)
        if len(reqs) != len(response.omegas):
            raise ValueError(
                f"head has {len(response.omegas)} frequencies but screening "
                f"has {len(reqs)} requests")
        W_gamma = []
        for omega, req in zip(response.omegas, reqs):
            if abs(complex(req.omega_ry) - omega) > 1.0e-12:
                raise ValueError(
                    f"head/screening frequency mismatch: {omega} vs "
                    f"{req.omega_ry} ({req.role})")
            try:
                W_role = W_by_role[req.role]
            except KeyError as exc:
                raise KeyError(
                    f"screening did not return required head role {req.role!r}") \
                    from exc
            # q = 0 is its own orbit: the wedge row is the full-zone row.
            from .cohsex_sigma import interaction_operator
            W_gamma.append(interaction_operator(W_role).representative_row(0))
        W_gamma = jnp.stack(W_gamma, axis=0)
        # Hard lifetime boundary (KNOWN_LORRAX_ISSUES.md "the bounded full-
        # head fold still needs a fresh-fit lifetime boundary"): force this
        # tiny (n_omega, mu_X, mu_Y) Gamma extraction eagerly, INSTEAD of
        # letting it stay queued behind whatever the caller does next.  Every
        # other stage this array's inputs pass through (screening.py's
        # chi/Dyson solves) already ends on an explicit
        # ``block_until_ready()``; ``qsgw_head.py`` had none, so this whole
        # module's only synchronization used to be the FIRST host readback
        # in ``head_samples_from_s``, which is why an OOM anywhere upstream
        # of it always surfaced there instead of at its own site.  This does
        # not change the value or its sharding -- only when the allocator is
        # asked to account for it -- so it is a pure scheduling change with
        # no bit-exactness impact.
        jax.block_until_ready(W_gamma)
        from gw.isdf_fitting import mem_probe
        mem_probe("qsgw_head.finalize_head_samples.pre_fold")
        if (
            int(W_gamma.shape[-2]) != int(response.Y_x.shape[-1])
            or int(W_gamma.shape[-1]) != int(response.Z_y.shape[-2])
        ):
            raise ValueError(
                "QSGW head-wing centroid extents do not match W(Gamma): "
                f"Y={response.Y_x.shape}, W={W_gamma.shape}, "
                f"Z={response.Z_y.shape}")
        from gw.head_correction import fold_cartesian_head_wings_sharded
        S_effective = fold_cartesian_head_wings_sharded(
            response.S_direct,
            response.Y_x,
            W_gamma,
            response.Z_y,
            float(meta.cell_volume),
            mesh_xy=mesh,
        )
        jax.block_until_ready(S_effective)
        mem_probe("qsgw_head.finalize_head_samples.post_fold")
    intraband = _metal_intraband(response)
    static_kappa2 = response.static_kappa2_bohr2
    if use_fold and static_kappa2 is not None and any(
            abs(z) <= 1.0e-14 for z in response.omegas):
        static_indices = [
            i for i, z in enumerate(response.omegas) if abs(z) <= 1.0e-14]
        if len(static_indices) != 1:
            raise ValueError(
                "static metallic head requires exactly one z=0 response")
        static_kappa2 = _fold_static_kappa2(
            response, W_gamma[static_indices[0]], float(meta.cell_volume), mesh)
    elif intraband is not None:
        static_kappa2 = None
    samples = head_samples_from_s(
        S_effective,
        response.omegas,
        wfn=wfn,
        meta=meta,
        config=config,
        static_kappa2_bohr2=static_kappa2,
        intraband=intraband,
        response_kind=("full_local_fields" if use_fold
                       else "direct_irreducible"),
        source_prefix=("head_schur" if use_fold else "head_direct"),
    )
    return IterationHeadSamples(
        omegas=response.omegas,
        samples=samples,
        sigma_energies_ry=response.sigma_energies_ry,
        sigma_occupations=response.sigma_occupations,
        efermi_ry=response.efermi_ry,
    )


def _extra_chi_rows(extras, intraband):
    """Every screened row's ``chi_extra`` [Z, n] on one mini-BZ batch ``q`` [n, 3], or None.

    ``extras[i]`` is None, ``("lindhard", z)`` (the Fermi-surface Lindhard
    term at z) or ``("constant", c)``; all Lindhard rows come from one
    program per batch (``FermiSurfaceIntraband.density_response_rows``).
    """
    if all(e is None for e in extras):
        return None
    lindhard = [i for i, e in enumerate(extras) if e is not None and e[0] == "lindhard"]
    zs = [extras[i][1] for i in lindhard]
    constant = np.asarray([e[1] if e is not None and e[0] == "constant" else 0.0
                           for e in extras], dtype=np.complex128)

    def rows(q):
        out = jnp.broadcast_to(jnp.asarray(constant)[:, None], (len(extras), q.shape[0]))
        if lindhard:
            out = out.at[np.asarray(lindhard)].set(intraband.density_response_rows(q, zs))
        return out
    return rows


def head_samples_from_s(
    S_cart_omega,
    omegas_ry,
    *,
    wfn,
    meta,
    config,
    static_kappa2_bohr2: float | None = None,
    response_kind="direct_irreducible",
    source_prefix: str = "qsgw_parallel_transport",
    intraband=None,
) -> tuple[object, ...]:
    """Convert replicated 3x3 S tensors to mini-BZ averaged head samples.

    ``intraband`` (``fermi_surface.FermiSurfaceIntraband``) marks a metal
    whose ``S`` carries the q-first Drude term ``D / z^2``: at every sample
    that term is exchanged for the anisotropic Fermi-surface Lindhard
    response of the cell, evaluated at each q sample, which is Thomas-Fermi
    at ``z = 0`` and Drude for ``|z| >> q u``
    (``docs/theory/metal-q0-head.md``).  ``static_kappa2_bohr2`` names the
    one exact-zero row whose Thomas-Fermi term is the full head's folded
    kappa^2; on a metal that row keeps its interband ``S(0)``.
    """
    from gw.head_correction import (
        HeadResponseKind, HeadSample, resolve_head_override)
    from gw.isdf_fitting import mem_probe
    from gw.vcoul import compute_q0_averages, compute_q0_averages_screened

    # This is the first host readback of ``S_cart_omega`` for callers that
    # do not already sync it (``finalize_iteration_head_samples`` now does,
    # at its own site -- see the lifetime-boundary comment there).  Report
    # what is live HERE too so a caller that skips that boundary (the
    # per-sample ``finalize_iteration_head_sample`` diagnostic entry point,
    # or a future one) still gets an attributable snapshot instead of a bare
    # RESOURCE_EXHAUSTED at this line.
    mem_probe("qsgw_head.head_samples_from_s.pre_readback")
    S_host = np.asarray(S_cart_omega, dtype=np.complex128)
    intraband_drude = None if intraband is None else intraband.drude_tensor
    omegas = tuple(complex(z) for z in np.asarray(omegas_ry).reshape(-1))
    if S_host.shape != (len(omegas), 3, 3):
        raise ValueError(
            f"S_cart_omega must be ({len(omegas)},3,3), got {S_host.shape}."
        )
    params = {
        "vhead": config.head.vhead,
        "whead_0freq": config.head.whead_0freq,
        "whead_imfreq": config.head.whead_imfreq,
    }
    kind = HeadResponseKind(response_kind)
    analytic_sphere = bool(getattr(
        config.head, "analytic_q0_sphere", config.head.head_minibz_average))
    # Pass 1: each row's cell-average operands.  Every row but a Thomas-Fermi
    # kappa^2 row averages v/(1 - v(q.S.q + chi_extra)) on the one mini-BZ
    # draw, so those rows share it in ONE call below (one device copy of the
    # draw, one <v>) instead of one call per frequency.
    rows = []
    for z, S in zip(omegas, S_host):
        override = resolve_head_override(params, z)
        if override is not None:
            rows.append(("override", z, override, None, False))
            continue
        is_static_metal = (
            static_kappa2_bohr2 is not None and abs(z) <= 1.0e-14)
        extra_chi = None
        if intraband is not None and not is_static_metal:
            drude = np.asarray(intraband_drude, dtype=np.complex128)
            if abs(z) > 1.0e-15:
                S = S - drude / (z * z)
            extra_chi = ("lindhard", z)
        elif is_static_metal and intraband is not None:
            # The folded static slot is the z -> 0+ limit of the same cell
            # function: interband q.S(0).q (epsilon_inf) plus Thomas-Fermi,
            # <8 pi / (q.eps_inf.q + kappa^2)>, never <8 pi/(q^2+kappa^2)>.
            extra_chi = ("constant", -float(static_kappa2_bohr2) / (8.0 * np.pi))
        static_from_s = is_static_metal and intraband is not None
        rows.append(("screened" if (static_from_s or not is_static_metal)
                     else "kappa2", z, S, extra_chi, is_static_metal))
    screened = [i for i, r in enumerate(rows) if r[0] == "screened"]
    averages = {}
    if screened:
        vc0, wcoul0 = compute_q0_averages_screened(
            wfn, meta, [rows[i][2] for i in screened],
            extra_chi_rows=_extra_chi_rows([rows[i][3] for i in screened], intraband),
            analytic_sphere=analytic_sphere)
        vc0 = complex(vc0)
        averages.update(zip(screened, ((vc0, w) for w in wcoul0)))
    for i, row in enumerate(rows):
        if row[0] == "kappa2":
            averages[i] = compute_q0_averages(
                wfn, jnp.asarray(0.0, dtype=jnp.float64), meta,
                S_cart=None,
                static_kappa2=jnp.asarray(static_kappa2_bohr2, dtype=jnp.float64),
                analytic_sphere=analytic_sphere, extra_chi=None)
    # Pass 2: read back once per row, in the caller's order.
    out = []
    for i, (label, z, S, _, is_static_metal) in enumerate(rows):
        if label == "override":
            out.append(S)
            continue
        vc0, wc0 = averages[i]
        out.append(
            HeadSample(
                vc0=complex(vc0),
                wcoul0=complex(wc0),
                source=(
                    (f"{source_prefix}_tf"
                     if is_static_metal else source_prefix)
                    if abs(z) <= 1.0e-14
                    else f"{source_prefix}(omega={z} Ry)"
                ),
                omega=z,
                S_cart=(S if label == "screened" else None),
                response_kind=kind,
            )
        )
    return tuple(out)


def _metal_static_head(wfns, surface, occupation_state, omegas, *, mesh,
                       meta, config, nb_logical, nspin, nspinor):
    """The exact z = 0 slot of a metallic head: Thomas-Fermi and its fold.

    ``kappa_TF^2 = 8 pi N(E_F) / Omega`` with ``N(E_F)`` from the same
    Fermi-surface table as the Drude term.  When the head carries wings
    (``wfns`` given) and the plan has an exact static row, the static
    density wings and the static Gamma body (``compute_chi0_matsubara`` at
    ``n = 0``, whose tau factors carry ``-df/dE``) complete the fold.  Both
    metallic head builders take this one owner.
    """
    capacity = 2.0 / (float(max(int(nspin), 1)) * float(max(int(nspinor), 1)))
    kappa2 = (8.0 * np.pi * capacity
              * float(np.sum(np.asarray(surface, dtype=np.float64)))
              / float(meta.nk_tot) / float(meta.cell_volume))
    static_Y_x = static_Z_y = static_chi_body_gamma = None
    if wfns is not None and any(abs(complex(z)) <= 1.0e-14 for z in omegas):
        static_Y_x, static_Z_y = static_head_wings_sharded(
            wfns, surface, mesh=mesh, nb_logical=int(nb_logical),
            nk_tot=int(meta.nk_tot), nspin=int(nspin), nspinor=int(nspinor))
        from gw.w_isdf import compute_chi0_matsubara
        static_chi_body_gamma = compute_chi0_matsubara(
            wfns, meta, mesh, occupation_state=occupation_state,
            nu_indices=(0,),
            rel_tol=float(config.minimax_config.target_error))[0:1]
    return kappa2, static_Y_x, static_Z_y, static_chi_body_gamma


@dataclass(frozen=True)
class QPVelocity:
    r"""One SC map's velocity in the DFT band basis; consumers read U^H v U.

    ``dft_cart`` is ``v_DFT`` plus this map's Sigma term: ``D_k DeltaH``
    from the links (``parallel_transport``), ``[DeltaH, W]``
    (``interband_commutator``), or none (``dft_velocity``, or a map whose
    links cannot serve it).  The head (S, Drude, wings) and
    ``dipole_qsgw.h5`` both take this array and the map's ``U``, so the SC
    screening and the BSE dipoles see one velocity.  ``base`` is ``v_DFT``;
    ``correction`` the Sigma term added (None when absent or zeroed);
    ``bound`` the link bound; ``zeroed`` why the term is zero on this map;
    ``label`` the terms in words (the dipole file's ``velocity`` stamp).
    """

    dft_cart: jax.Array
    base: jax.Array
    correction: jax.Array | None
    bound: tuple | None
    zeroed: str | None
    label: str


def qp_velocity(
    velocity_dft_cart,
    occupations_qp_kn,
    *,
    mesh: Mesh,
    nb_logical: int,
    delta_h_dft=None,
    forward_links=None,
    forward_neighbors=None,
    kgrid: tuple[int, int, int] | None = None,
    bvec_cart=None,
    collapsed_position=None,
    nb_links: int | None = None,
    link_bound: tuple[float, float] | None = None,
    velocity_base_cart=None,
    link_unserved: str | None = None,
) -> QPVelocity:
    """The owner of the SC velocity: ``v_DFT`` plus this map's Sigma term.

    ``forward_links=None`` with no ``velocity_base_cart`` is
    ``sc_head_update = dft_velocity``: ``delta_h_dft`` is unused.  With
    ``velocity_base_cart`` (``interband_commutator``) ``velocity_dft_cart``
    already holds ``v + [DeltaH, W]``.  ``parallel_transport`` drops the
    covariant correction on a map whose links cannot serve it
    (:func:`sigma_term_zeroed`; ``link_unserved`` is the run-long reason
    when the source carries no links).
    """
    v_dft_basis = jnp.asarray(velocity_dft_cart, dtype=jnp.complex128)
    base, correction, bound = v_dft_basis, None, None
    if velocity_base_cart is not None:
        # interband_commutator hands v + [DeltaH, W]: its Sigma term is the
        # difference from the DFT velocity p + i[r, V_NL].
        base = jnp.asarray(velocity_base_cart, dtype=jnp.complex128)
        correction = v_dft_basis - base
    zeroed = sigma_term_zeroed(link_unserved, None)
    if forward_links is not None:
        if forward_neighbors is None:
            raise ValueError(
                "forward_neighbors are required when forward_links are present"
            )
        # On an outer link set the derivative is taken there and restricted
        # to the head's bands (identity when the two sets agree).
        correction = head_band_block(covariant_link_derivative(
            delta_h_dft,
            forward_links,
            forward_neighbors,
            mesh=mesh,
            kgrid=kgrid,
            bvec_cart=bvec_cart,
            collapsed_position=collapsed_position,
        ), int(nb_logical), mesh=mesh,
            nb_outer=int(nb_links or nb_logical))
        if link_bound is not None:
            bound = link_correction_bound(
                correction, v_dft_basis, occupations_qp_kn,
                link_error=link_bound[0],
                rtol=link_bound[1])
        zeroed = sigma_term_zeroed(None, bound)
        if zeroed is None:
            v_dft_basis = v_dft_basis + correction
        else:
            correction = None
    if velocity_base_cart is not None:
        label = ("v_DFT + [DeltaH, W] (interband_commutator; intraband "
                 "connection omitted)")
    elif forward_links is not None and zeroed is None:
        label = "v_DFT + D_k DeltaH (parallel_transport links)"
    elif zeroed is not None:
        label = f"v_DFT (D_k DeltaH zeroed: {zeroed})"
    else:
        label = "v_DFT (dft_velocity: no Sigma term)"
    return QPVelocity(dft_cart=v_dft_basis, base=base, correction=correction,
                      bound=bound, zeroed=zeroed, label=label)


@dataclass(frozen=True)
class HeadVelocityTerms:
    """One map's QP velocity, its p / V_NL / Sigma shares and Drude tensor.

    ``velocity_terms = (names, shares, bound, zeroed)`` feeds the per-map
    head block; ``drude_tensor`` (metals) is the q=0-cell Drude weight;
    ``sigma_occupations`` the occupations the block's gap reads.
    """

    v_qp: jax.Array
    velocity_terms: tuple
    drude_tensor: object
    fermi_surface: object
    pair_split: object
    sigma_occupations: np.ndarray


def head_velocity_terms(
    velocity: QPVelocity,
    U_dft_to_qp,
    energies_qp_kn_ry,
    occupations_qp_kn,
    *,
    surface_weight_qp_kn=None,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
    nb_logical: int,
    wfn,
    meta,
    velocity_kinetic_cart=None,
) -> HeadVelocityTerms:
    """The head block's operands, one owner for the scalar and four-current heads.

    Rotates ``velocity`` into the QP basis, splits it into its p, V_NL and
    Sigma shares (``velocity_term_shares``) and, on a metal, forms the
    Fermi-surface Drude tensor (``metal_intraband_model``).
    """
    v_dft_basis, base = velocity.dft_cart, velocity.base
    correction, bound, zeroed = (velocity.correction, velocity.bound,
                                 velocity.zeroed)
    v_qp = rotate_velocity_active_to_qp(v_dft_basis, U_dft_to_qp, mesh=mesh)
    # The per-map head block: p, V_NL and Sigma shares of this velocity.
    pieces = ([("p", velocity_kinetic_cart),
               ("V_NL", base - velocity_kinetic_cart)]
              if velocity_kinetic_cart is not None else [("p + V_NL", base)])
    if correction is not None:
        pieces.append(("Sigma", correction))
    names, shares = velocity_term_shares(
        v_qp, [(name, rotate_velocity_active_to_qp(
            jnp.asarray(x, dtype=jnp.complex128), U_dft_to_qp, mesh=mesh))
            for name, x in pieces],
        nb_logical=nb_logical, surface_weight_kn=surface_weight_qp_kn,
        energies_kn=energies_qp_kn_ry, occupations_kn=occupations_qp_kn)
    if zeroed is not None:
        names, shares = names + ("Sigma",), np.vstack([shares, np.zeros((1, 3))])
    # Physical state multiplicity belongs to the source WFN.  A
    # kinetic-balance lift changes only the stored spinor representation.
    drude_tensor = fermi_surface = pair_split = None
    if surface_weight_qp_kn is not None:
        drude_tensor, fermi_surface, pair_split = metal_intraband_model(
            v_qp, surface_weight_qp_kn, energies_qp_kn_ry, mesh=mesh,
            nb_logical=nb_logical, cell_volume=float(meta.cell_volume),
            nk_tot=int(meta.nk_tot), nspin=int(wfn.nspin),
            nspinor=int(meta.nspinor_wfnfile), bvec_cart=bvec_cart,
            kgrid=kgrid)
    return HeadVelocityTerms(
        v_qp=v_qp, velocity_terms=(names, shares, bound, zeroed),
        drude_tensor=drude_tensor, fermi_surface=fermi_surface,
        pair_split=pair_split,
        sigma_occupations=np.asarray(occupations_qp_kn, dtype=np.float64))


def build_iteration_head_response(
    velocity: QPVelocity,
    U_dft_to_qp,
    energies_qp_kn_ry,
    occupations_qp_kn,
    omegas_ry,
    *,
    surface_weight_qp_kn=None,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
    nb_logical: int,
    sigma_energies_ry,
    efermi_ry: float,
    wfn,
    meta,
    config,
    wfns_qp=None,
    eta_ry: float | None = None,
    occupation_state=None,
    velocity_kinetic_cart=None,
) -> IterationHeadResponse:
    """Build current-basis direct head and, when requested, its wings.

    ``velocity`` is this map's :func:`qp_velocity`.  Everything downstream
    of it -- the per-iteration rotation into the QP basis, S(z), the Drude
    term, the ISDF wings, the static kappa^2 -- is the SAME code on every
    head route.

    ``occupation_state`` is the map's solved state (``occupations_qp_kn`` is
    its ``f_kn``); the static Gamma body is ``gw.w_isdf.compute_chi0_matsubara``
    at ``n = 0`` on it, which refuses any family but Fermi-Dirac.
    """
    terms = head_velocity_terms(
        velocity, U_dft_to_qp, energies_qp_kn_ry, occupations_qp_kn,
        surface_weight_qp_kn=surface_weight_qp_kn, mesh=mesh, kgrid=kgrid,
        bvec_cart=bvec_cart, nb_logical=nb_logical, wfn=wfn, meta=meta,
        velocity_kinetic_cart=velocity_kinetic_cart)
    v_qp, velocity_terms = terms.v_qp, terms.velocity_terms
    drude_tensor, fermi_surface, pair_split = (
        terms.drude_tensor, terms.fermi_surface, terms.pair_split)
    resolved_eta_ry = (
        float(config.head.wcoul0_eta)
        if eta_ry is None else float(eta_ry)
    )
    normalization_nspinor = int(meta.nspinor_wfnfile)
    S = head_s_tensor_sharded(
        v_qp,
        energies_qp_kn_ry,
        occupations_qp_kn,
        omegas_ry,
        mesh=mesh,
        nb_logical=nb_logical,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
        nspin=int(wfn.nspin),
        nspinor=normalization_nspinor,
        eta_ry=resolved_eta_ry,
        surface_weight_kn=surface_weight_qp_kn,
        pair_split=pair_split,
    )
    Y_x = Z_y = None
    static_Y_x = static_Z_y = static_chi_body_gamma = None
    if wfns_qp is not None:
        Y_x, Z_y = head_wings_sharded(
            v_qp,
            wfns_qp,
            energies_qp_kn_ry,
            occupations_qp_kn,
            omegas_ry,
            mesh=mesh,
            nb_logical=nb_logical,
            nk_tot=int(meta.nk_tot),
            nspin=int(wfn.nspin),
            nspinor=normalization_nspinor,
            eta_ry=resolved_eta_ry,
            surface_weight_kn=surface_weight_qp_kn,
        )
    omegas = tuple(complex(z) for z in np.asarray(omegas_ry).reshape(-1))
    static_Y_x = static_Z_y = static_chi_body_gamma = None
    static_kappa2 = None
    if surface_weight_qp_kn is not None:
        (static_kappa2, static_Y_x, static_Z_y,
         static_chi_body_gamma) = _metal_static_head(
            wfns_qp, surface_weight_qp_kn, occupation_state, omegas,
            mesh=mesh, meta=meta, config=config, nb_logical=nb_logical,
            nspin=int(wfn.nspin), nspinor=normalization_nspinor)
    return IterationHeadResponse(
        omegas=omegas,
        S_direct=S,
        Y_x=Y_x,
        Z_y=Z_y,
        static_kappa2_bohr2=static_kappa2,
        static_Y_x=static_Y_x,
        static_Z_y=static_Z_y,
        static_chi_body_gamma=static_chi_body_gamma,
        sigma_energies_ry=np.asarray(sigma_energies_ry, dtype=np.float64),
        sigma_occupations=np.asarray(occupations_qp_kn, dtype=np.float64)[
            :, : np.shape(sigma_energies_ry)[1]
        ],
        efermi_ry=float(efermi_ry),
        drude_tensor=drude_tensor,
        fermi_surface=fermi_surface,
        velocity_terms=velocity_terms,
    )


def expected_hubbard_stamp(config, *, wfn, fallback_dir, caller) -> str:
    """The DFT+U stamp a velocity artifact must carry for THIS deck.

    One resolver (``psp.hubbard_ops.resolve_hubbard_input``) for producer and
    consumers: it refuses when the WFN's QE schema declares DFT+U and the deck
    names no Hubbard input, so an old p + i[r,V_NL] file cannot silently feed a
    DFT+U head.  Deck-relative paths resolve against ``config.input_dir``.
    """
    from psp.hubbard_ops import hubbard_provenance_for
    return hubbard_provenance_for(
        getattr(config, "hubbard_input", ""),
        getattr(config, "hubbard_occupations", ""),
        wfn=wfn, base_dir=(getattr(config, "input_dir", "") or fallback_dir),
        caller=caller)


def head_dipole_operator_stamps(config, *, wfn, meta, fallback_dir) -> dict:
    """The dipole stamps this deck's head needs, in ``check_dipole_provenance``'s keywords.

    One resolver for the head reader and for the SC's ``dipole_qsgw.h5``
    stamp (:func:`stamp_qsgw_dipole_provenance`): the window, the full
    analytic ``p + i[r, V_NL]`` operator with the resolved V_NL sign, the
    head representation and the DFT+U stamp.
    """
    from psp.get_dipole_mtxels import resolve_vnl_velocity_sign
    from common.four_current_model import resolve_four_current_representation
    representation = resolve_four_current_representation(
        bool(getattr(config, "bispinor", int(meta.nspinor) == 4)),
        getattr(config, "bispinor_gw", "bare_transverse"))
    return dict(
        nval=int(config.nval), ncond=int(config.ncond),
        nband=int(config.nband),
        bispinor=representation.scalar_head_bispinor,
        skip_vnl=False, vnl_mode="analytic",
        vnl_velocity_sign=resolve_vnl_velocity_sign(
            None, config.vnl_velocity_sign),
        hubbard=expected_hubbard_stamp(
            config, wfn=wfn, fallback_dir=fallback_dir,
            caller="dft head dipole velocity"))


def read_authenticated_dipole_velocity(
    dipole_path, *, wfn, meta, config, mesh: Mesh, wfn_fingerprint_binding=None,
):
    """Read file-wedge velocities and restore their polar time-odd full-k action.

    COLLECTIVE over ``mesh``: the parent rows are read through SlabIO, each
    rank its band tile (``file_io.restart_bundle.read_dipole_parent_window``).
    """
    from symmetry_maps import unfold_file_wedge_polar_matrix

    # Fail before the host read and every sharded head allocation.  Shape does
    # not identify a velocity artifact: in particular, a two-spinor dipole and
    # a kinetic-balance four-spinor dipole have the same (3,nk,nb,nb) shape.
    # The producer owns both the stamp grammar and sign resolution; consume
    # those owners directly rather than mirroring either convention here.
    from file_io.restart_bundle import (
        check_dipole_provenance,
    )
    if not check_dipole_provenance(
            dipole_path,
            wfn=wfn,
            wfn_fingerprint_binding=wfn_fingerprint_binding,
            **head_dipole_operator_stamps(
                config, wfn=wfn, meta=meta,
                fallback_dir=os.path.dirname(os.path.abspath(dipole_path)))):
        raise ValueError(
            "GATE dft_head_dipole_provenance: the full head received an "
            "unauthenticated dipole artifact.\n"
            f"  got:  dipole_file = {dipole_path!r}; at least one WFN, "
            "q->0 coverage, VNL, DFT+U, or representation stamp mismatched\n"
            "  want: dipole.h5 regenerated from this run's exact deck\n"
            "  why:  S_direct and the wings must use the same WFN and "
            "velocity operator as the finite-q charge response")
    sym = wfn.symmetry()
    b0, b4 = int(meta.b_id_0), int(meta.b_id_4_chi_user)
    from file_io.restart_bundle import read_dipole_parent_window
    parents = read_dipole_parent_window(
        dipole_path, sym.kirr_fullids, b0, b4, nk_full=sym.nk_tot, mesh=mesh)
    return np.moveaxis(unfold_file_wedge_polar_matrix(sym, parents), 1, 0)


def build_dft_head_response(
    wfns,
    omegas_ry,
    *,
    input_dir: str,
    mesh: Mesh,
    wfn,
    meta,
    config,
    wfn_fingerprint_binding=None,
    wings: bool = True,
    occupation_state=None,
    frozen_parts: dict | None = None,
) -> IterationHeadResponse:
    """Build the one-shot DFT head on exactly the chi0 band manifold.

    This is the non-self-consistent entry to the same sharded direct-head and
    wing kernels used by QSGW.  In particular, both ``S_direct`` and ``Y/Z``
    use ``[b0,b4_chi)``; constructing the scalar from every band in a larger
    dipole file while the body uses ``number_bands_chi`` is refused by shape
    and slicing here rather than silently mixing transition manifolds.

    ``wings=False`` builds the direct head alone (``Y_x = Z_y = None``): the
    ``head_correction = no_local_fields`` response, whose consumer
    (:func:`finalize_iteration_head_sample`) never folds.  ``S_direct``
    needs no time-reversal assumption (``gw.shared_pole_head`` docstring), so
    this is the head an ordered store carries.

    ``occupation_state`` is the metal's fixed-N Fermi-Dirac state on this
    spectrum (the one-shot state, or the DFT state of a map whose head stays
    frozen).  It replaces the bundle's 0/1 table, a step by band index that
    splits degenerate multiplets at the cut, and it adds the intraband
    Fermi-surface response exactly as the QSGW builder does
    (:func:`build_iteration_head_response`): the table of
    :func:`gw.fermi_surface.metal_head_surface_weights` enters ``S(z)`` as
    the Drude term and the wings as their intraband term, and the exact
    ``z = 0`` row takes Thomas-Fermi with its static fold
    (:func:`_metal_static_head`).  ``None`` is the insulating head.

    ``frozen_parts`` is a caller-owned dict that holds the frequency-free
    part of this response between calls on the same ``wfns``: the
    authenticated velocity and, on a metal, the Fermi-surface table, the
    Drude atoms, the pair split and the static head.  Its key is ``wings``,
    whether the plan has an exact static row, and the state's ``(mu,
    occ_hash)``.  A frozen SC head (``sc_head_update = off``) keeps that key
    while its frequency plan follows the QP energy span, so each map then
    builds only S(z) and the wings.
    """
    import os

    z = np.asarray(omegas_ry, dtype=np.complex128).reshape(-1)
    parts_key = (bool(wings), any(abs(complex(v)) <= 1.0e-14 for v in z),
                 None if occupation_state is None else
                 (float(occupation_state.mu_ry), occupation_state.occ_hash))
    parts = None if frozen_parts is None else frozen_parts.get(parts_key)
    dipole_path = os.path.join(input_dir, "dipole.h5")
    b0 = int(meta.b_id_0)
    b4 = int(meta.b_id_4_chi_user)
    nb_logical = b4 - b0
    energies = jnp.asarray(wfns.enk[:, :nb_logical])
    occupations = jnp.asarray(wfns.occ[:, :nb_logical])
    if parts is None:
        if not os.path.exists(dipole_path):
            raise FileNotFoundError(
                "head_correction=full requires dipole.h5 to build the direct "
                f"head and wings; missing {dipole_path}.")
        # The one velocity owner at DeltaH = 0 and U = I: v_DFT, no Sigma
        # term (the SC dft_velocity head on the DFT states).  Held on the
        # host, as the frozen part it is.
        velocity_cart = np.asarray(qp_velocity(
            read_authenticated_dipole_velocity(
                dipole_path, wfn=wfn, meta=meta, config=config, mesh=mesh,
                wfn_fingerprint_binding=wfn_fingerprint_binding),
            occupations, mesh=mesh, nb_logical=nb_logical).dft_cart)
    else:
        velocity_cart = parts["velocity_cart"]
    if velocity_cart.shape[1:] != (
            int(meta.nk_tot), nb_logical, nb_logical):
        raise ValueError(
            "dipole/chi head manifold mismatch: sliced velocity has "
            f"{velocity_cart.shape}, expected "
            f"(3,{int(meta.nk_tot)},{nb_logical},{nb_logical}) for global "
            f"bands [{b0},{b4}).")
    # ``meta.nspinor`` is four for the bispinor representation, whereas
    # response normalization counts the source-WFN states.
    normalization_nspinor = int(meta.nspinor_wfnfile)
    surface = None
    static_kappa2 = None
    static_Y_x = static_Z_y = static_chi_body_gamma = None
    drude_tensor = fermi_surface = pair_split = None
    if occupation_state is not None:
        if b0 != 0:
            raise ValueError(
                "metal head requires a band manifold starting at 0, got "
                f"b_id_0={b0}")
        f_state = np.asarray(occupation_state.f_kn, dtype=np.float64)
        if f_state.shape[0] != int(meta.nk_tot) or f_state.shape[1] < nb_logical:
            raise ValueError(
                "metal head occupation state does not cover the chi head "
                f"manifold: f_kn {f_state.shape}, want "
                f"({int(meta.nk_tot)}, >={nb_logical})")
        occupations = jnp.asarray(f_state[:, :nb_logical])
        if parts is None:
            from .fermi_surface import metal_head_surface_weights
            surface_host = metal_head_surface_weights(
                np.asarray(energies, dtype=np.float64),
                float(occupation_state.mu_ry), sym=wfn.symmetry(),
                kgrid=wfn.kgrid, bvec_cart=_head_bvec(wfn))
            surface = jnp.asarray(surface_host)
            drude_tensor, fermi_surface, pair_split = metal_intraband_model(
                jnp.asarray(velocity_cart), surface, energies, mesh=mesh,
                nb_logical=nb_logical, cell_volume=float(meta.cell_volume),
                nk_tot=int(meta.nk_tot), nspin=int(wfn.nspin),
                nspinor=normalization_nspinor, bvec_cart=_head_bvec(wfn),
                kgrid=wfn.kgrid)
            (static_kappa2, static_Y_x, static_Z_y,
             static_chi_body_gamma) = _metal_static_head(
                wfns if wings else None, surface, occupation_state, z,
                mesh=mesh, meta=meta, config=config, nb_logical=nb_logical,
                nspin=int(wfn.nspin), nspinor=normalization_nspinor)
        else:
            (surface, drude_tensor, fermi_surface, pair_split, static_kappa2,
             static_Y_x, static_Z_y, static_chi_body_gamma) = parts["metal"]
    if parts is None and frozen_parts is not None:
        frozen_parts.clear()
        frozen_parts[parts_key] = dict(
            velocity_cart=velocity_cart,
            metal=(surface, drude_tensor, fermi_surface, pair_split,
                   static_kappa2, static_Y_x, static_Z_y,
                   static_chi_body_gamma))
    S = head_s_tensor_sharded(
        jnp.asarray(velocity_cart), energies, occupations, z,
        mesh=mesh, nb_logical=nb_logical,
        cell_volume=float(meta.cell_volume), nk_tot=int(meta.nk_tot),
        nspin=int(wfn.nspin), nspinor=normalization_nspinor,
        eta_ry=float(config.head.wcoul0_eta), surface_weight_kn=surface,
        pair_split=pair_split)
    Y_x = Z_y = None
    if wings:
        Y_x, Z_y = head_wings_sharded(
            jnp.asarray(velocity_cart), wfns, energies, occupations, z,
            mesh=mesh, nb_logical=nb_logical, nk_tot=int(meta.nk_tot),
            nspin=int(wfn.nspin), nspinor=normalization_nspinor,
            eta_ry=float(config.head.wcoul0_eta), surface_weight_kn=surface)
    # Hard lifetime boundary: this module previously had zero
    # ``block_until_ready`` calls (unlike ``screening.py``'s per-stage
    # discipline), so the direct head/wings built here stayed queued,
    # unattributed, until whatever LATER stage first forced a host
    # readback -- see the matching boundary + comment in
    # ``finalize_iteration_head_samples``.  Pure scheduling change.
    jax.block_until_ready((S, Y_x, Z_y))
    from gw.isdf_fitting import mem_probe
    mem_probe("qsgw_head.build_dft_head_response.post_response")
    e_host = np.asarray(energies)
    n_occ_local = max(0, min(int(meta.nelec) - b0, nb_logical))
    if occupation_state is not None:
        efermi = float(occupation_state.mu_ry)
    elif 0 < n_occ_local < nb_logical:
        efermi = 0.5 * (
            float(np.max(e_host[:, n_occ_local - 1]))
            + float(np.min(e_host[:, n_occ_local])))
    else:
        efermi = 0.0
    return IterationHeadResponse(
        omegas=tuple(complex(value) for value in z),
        S_direct=S, Y_x=Y_x, Z_y=Z_y,
        static_kappa2_bohr2=static_kappa2,
        static_Y_x=static_Y_x, static_Z_y=static_Z_y,
        static_chi_body_gamma=static_chi_body_gamma,
        sigma_energies_ry=e_host[:, :int(meta.nb_sigma)],
        sigma_occupations=np.asarray(occupations)[:, :int(meta.nb_sigma)],
        efermi_ry=efermi, drude_tensor=drude_tensor,
        fermi_surface=fermi_surface)


def _head_bvec(wfn):
    """Cartesian reciprocal basis (rows b_i, bohr^-1) of the Coulomb service."""
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import CoulombGeometry
    return np.asarray(CoulombGeometry.from_wfn(wfn).bvec, dtype=np.float64)


def metal_head_summary(response: IterationHeadResponse, occupation_state) -> str:
    """One log line naming a metallic head's occupations and plasma frequencies."""
    from common import RYD_TO_EV
    kappa = response.static_kappa2_bohr2
    wp = ("omega_p principal = " + "/".join(
        f"{x:.4f}" for x in plasma_frequencies_ev(response.drude_tensor)) + " eV"
        if response.fermi_surface is None else drude_report(response.fermi_surface))
    return (
        "metal head: fixed-N "
        f"{occupation_state.smearing_family} occupations (mu="
        f"{float(occupation_state.mu_ry) * RYD_TO_EV:.6f} eV, occ_hash="
        f"{occupation_state.occ_hash}) plus the tetrahedron Fermi-surface "
        "intraband term; " + wp
        + ("" if kappa is None else f"; kappa_TF^2 = {kappa:.6f} bohr^-2")
        + ("" if response.fermi_surface is None
           else "; intraband cell: " + response.fermi_surface.describe()))


def build_iteration_head_samples(
    delta_h_dft,
    forward_links,
    forward_neighbors,
    velocity_dft_cart,
    U_dft_to_qp,
    energies_qp_kn_ry,
    occupations_qp_kn,
    omegas_ry,
    *,
    surface_weight_qp_kn=None,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
    nb_logical: int,
    sigma_energies_ry,
    efermi_ry: float,
    wfn,
    meta,
    config,
    collapsed_position=None,
) -> IterationHeadSamples:
    """Backward-compatible direct-head builder used by small diagnostics."""
    response = build_iteration_head_response(
        delta_h_dft,
        forward_links,
        forward_neighbors,
        velocity_dft_cart,
        U_dft_to_qp,
        energies_qp_kn_ry,
        occupations_qp_kn,
        omegas_ry,
        surface_weight_qp_kn=surface_weight_qp_kn,
        mesh=mesh,
        kgrid=kgrid,
        bvec_cart=bvec_cart,
        nb_logical=nb_logical,
        sigma_energies_ry=sigma_energies_ry,
        efermi_ry=efermi_ry,
        wfn=wfn,
        meta=meta,
        config=config,
        collapsed_position=collapsed_position,
    )
    return finalize_iteration_head_samples(
        response, wfn=wfn, meta=meta, config=config, mesh=mesh)


def trs_velocity_parity_residual(
    velocity_cart,
    *,
    kgrid: tuple[int, int, int],
    trs_measured: bool | None,
) -> dict[str, float]:
    """Measure ``v_i(−k) = −conj(v_i(k))`` — the module docstring's eq. (2).

    APPLIES EQUALLY to ``v^DFT`` and to the assembled ``v^Q`` — the whole
    content of the derivation above is that the QSGW corrections do not
    change the parity, so this one statistic gates both and a sign error
    in ``d_k Sigma`` or in the ``−i[A, Sigma]`` commutator cannot hide in
    the sum.

    ``trs_measured`` is REQUIRED and has no default.  Pass
    ``SymMaps.trs_allowed`` (the canonical consumer-facing result of the
    spin-density measurement).  ``None`` means the verdict is unavailable,
    and the statistic is then returned with ``verdict = nan`` rather than
    being read as a pass — an unmeasured system is not a TRS system.

    THE VERDICT STATISTIC IS THE BAND TRACE, and the reason is in the
    module docstring's SCOPE paragraph: ``tr v_i(k)`` is invariant under
    any unitary mixing inside the retained band window, so it survives
    both a little-group gauge difference between ``k`` and ``−k`` and the
    Kramers-partner ambiguity of a spinor deck.  Its SENSITIVITY, stated
    because a null on it must not be quoted as coverage it does not have:
    a trace is blind to any parity error whose band matrix is traceless,
    which includes a sign flip confined to the strictly off-diagonal
    transition sector.  ``elementwise_rel`` is returned beside it as a
    STRICTLY STRONGER diagnostic that is only meaningful when the full-BZ
    gauge is known to be pair-coherent, and it is never the verdict.

    Returns
    -------
    dict
        ``trace_rel`` (the verdict statistic), ``trace_abs``,
        ``trace_scale``, ``elementwise_rel`` (diagnostic),
        ``verdict`` — ``1.0`` pass, ``0.0`` fail, ``nan`` not applicable
        (TRS broken or unmeasured, where eq. (2) is not an identity).
    """
    from common.sanity import neg_q_index

    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    grid = tuple(int(n) for n in kgrid)
    nk = int(np.prod(grid))
    if v.ndim != 4 or int(v.shape[0]) != 3 or int(v.shape[1]) != nk:
        raise ValueError(
            "trs_velocity_parity_residual expects (3, nk, nb, nb) with "
            f"nk={nk} for kgrid={grid}; got {tuple(v.shape)}.")
    neg = jnp.asarray(neg_q_index(grid))

    @jax.jit
    def _stats(a):
        mirror = jnp.take(a, neg, axis=1)
        # Band trace: gauge-invariant under any unitary inside the window.
        tr = jnp.trace(a, axis1=-2, axis2=-1)
        tr_mirror = jnp.trace(mirror, axis1=-2, axis2=-1)
        tr_dev = jnp.max(jnp.abs(tr_mirror + jnp.conj(tr)))
        tr_scale = jnp.max(jnp.abs(tr))
        el_dev = jnp.max(jnp.abs(mirror + jnp.conj(a)))
        el_scale = jnp.max(jnp.abs(a))
        return jnp.stack([
            tr_dev.astype(jnp.float64), tr_scale.astype(jnp.float64),
            el_dev.astype(jnp.float64), el_scale.astype(jnp.float64),
        ])

    tr_dev, tr_scale, el_dev, el_scale = (
        float(x) for x in np.asarray(jax.device_get(_stats(v))))
    trace_rel = (tr_dev / tr_scale) if tr_scale > 0.0 else tr_dev
    el_rel = (el_dev / el_scale) if el_scale > 0.0 else el_dev
    verdict = float("nan")
    if trs_measured is not None and bool(trs_measured):
        verdict = 1.0 if trace_rel <= _TRS_VELOCITY_PARITY_BREAK else 0.0
    return {
        "trace_rel": trace_rel,
        "trace_abs": tr_dev,
        "trace_scale": tr_scale,
        "elementwise_rel": el_rel,
        "verdict": verdict,
    }


def report_trs_velocity_parity(
    name: str,
    metrics: dict[str, float],
    *,
    trs_measured: bool | None,
    print_fn=print,
) -> bool:
    """Print the parity verdict; refuse only on an INVERTED parity.

    Three outcomes, and they are deliberately different lines:

    * ``trs_measured`` false or ``None`` — eq. (2) is not an identity for
      this mean field (or nobody measured whether it is), so the number is
      printed as a DIAGNOSTIC with no verdict.  A ferromagnet is expected
      to violate it; that asymmetry is anomalous-velocity physics.
    * measured TRS, residual above ``_TRS_VELOCITY_PARITY_BREAK`` — the
      parity is INVERTED, which is the failure a wrong sign in ``d_k
      Sigma`` or in ``−i[A, Sigma]`` produces, and it refuses through the
      standard :class:`common.sanity.SanityError` route unless the named
      override is set.
    * measured TRS, residual between the roundoff floor and that bar — a
      loud WARNING, not a refusal, because no deck has yet measured this
      statistic's floor and a ceiling derived from nothing is the trap
      ``TASTE.md`` calls calibrating from the value that wants to pass.
    """
    from common import sanity

    # The documented global escape hatch applies here as to every other
    # stage-boundary gate: ``LORRAX_SANITY=0`` skips it entirely.  The
    # NAMED override below is the narrower knob, for an operator who wants
    # this one refusal lifted and every other gate kept.
    if not sanity.sanity_enabled():
        return True
    rel = float(metrics["trace_rel"])
    el = float(metrics["elementwise_rel"])
    detail = (f"tr-parity max|tr v(-k) + conj(tr v(k))|/max|tr v| = "
              f"{rel:.3e} (elementwise diagnostic {el:.3e})")
    if trs_measured is None:
        print_fn(f"  sanity[{name}]: {detail} — time reversal was NOT "
                 f"MEASURED for this WFN, so no verdict is taken (an "
                 f"unmeasured system is not a TRS system).")
        return True
    if not bool(trs_measured):
        print_fn(f"  sanity[{name}]: {detail} — the measured spin density "
                 f"says TIME REVERSAL IS BROKEN, so v(-k) = -conj(v(k)) is "
                 f"NOT an identity here and this number is a diagnostic "
                 f"only.")
        return True
    if rel > _TRS_VELOCITY_PARITY_BREAK:
        sanity.warn(
            f"{name} has an INVERTED time-reversal parity: {detail}, above "
            f"{_TRS_VELOCITY_PARITY_BREAK:.1f} — strictly more than the "
            f"whole signal, which no gauge or band-window artefact can "
            f"reach.  The velocity operator is ODD under time reversal and "
            f"every term of v^Q = v^DFT + d_k Sigma - i[A, Sigma] carries "
            f"that same parity (gw/qsgw_head.py module docstring, eq. 2), "
            f"so this is a sign, not a convention.  Set "
            f"LORRAX_ALLOW_TRS_VELOCITY_PARITY_BREAK=1 to proceed anyway "
            f"and leave a trace.",
            print_fn=print_fn)
        from .gw_config import env_bool
        if not env_bool("LORRAX_ALLOW_TRS_VELOCITY_PARITY_BREAK", False,
                        print_fn=print_fn):
            raise sanity.SanityError(
                f"{name}: time-reversal parity is inverted ({rel:.3e}); "
                f"see gw/qsgw_head.py eq. (2).  Override with "
                f"LORRAX_ALLOW_TRS_VELOCITY_PARITY_BREAK=1.")
        return False
    if rel > _TRS_VELOCITY_PARITY_FLOOR:
        sanity.warn(
            f"{name} time-reversal parity residual is above the roundoff "
            f"floor: {detail} > {_TRS_VELOCITY_PARITY_FLOOR:.1e}.  This "
            f"statistic has NO CALIBRATED CEILING yet — no deck has "
            f"measured its floor — so it warns rather than refuses.  A "
            f"degenerate multiplet straddling the band-window edge is the "
            f"benign explanation; check the window before the physics.",
            print_fn=print_fn)
        return False
    print_fn(f"  sanity[{name}]: {detail} — parity holds.")
    return True
