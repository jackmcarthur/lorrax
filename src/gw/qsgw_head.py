"""Build charge and current head responses with typed wavefunction transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.shard_map import shard_map


__all__ = [
    "DftVelocityHeadData",
    "IterationHeadResponse",
    "IterationHeadSamples",
    "ParallelTransportHeadData",
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
    "finalize_iteration_head_sample",
    "finalize_iteration_head_samples",
    "load_dft_velocity_head",
    "load_parallel_transport_head",
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
_HEAD_WING_FREQUENCY_BLOCK = 8

# Bound the face-layout wing kernel's per-step psi gather (obstacle #3/#5 of
# the low_mem_bands audit, report §"Full q->0 head/body wings").  psi_mun/
# psi_nmu have NO replicated band axis (unlike legacy's psi_xn/psi_yn), so a
# rank cannot read an arbitrary band window for free; instead it gathers the
# FULL band extent for a small MU block at a time via one lax.all_gather per
# block, keeping the transient (nk, ns, nb_full, mu_block)-shaped buffer
# bounded independent of how many centroids this rank owns locally.  See
# ``_head_wing_kernel_face``'s docstring for the full residency algebra.
_HEAD_WING_MU_BLOCK = 64
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


def _ascii_stamp(io, path: str, name: str) -> str:
    """A ``uint8`` provenance stamp, read through SlabIO, as ``str``.

    ``write_attr`` publishes these as ``uint8`` datasets.  The phdf5
    transport's dtype table has no unsigned type, so the read asks for
    ``int32`` and HDF5 widens — the same route
    ``file_io.parallel_transport._decode_i32_text`` already takes for the
    W-av stamps, and the reason it takes it.
    """
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
        SCHEMA_VERSION,
        VELOCITY_DFT_DATASET,
        load_full_bz_links,
        load_link_singular_values,
    )
    from file_io.slab_io import SlabIO
    from common.parallel_transport import band_storage_extent, wfn_fingerprint

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
                f"expected {int(SCHEMA_VERSION)}"
            )
        if ints["connection_complete"] != 1:
            refusals.append("connection_complete is not 1")
        if (
            ints["velocity_validation_complete"] != 1
            or ints["velocity_validation_passed"] != 1
        ):
            refusals.append(
                "mandatory finite-link DFT head validation is not "
                "complete/passing"
            )
        if ints["band_start"] != 0 or ints["band_stop"] != expected_nb:
            refusals.append(
                f"band manifold [{ints['band_start']},{ints['band_stop']}) != "
                f"current full head manifold [0,{expected_nb})"
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

        spec = P(None, None, "x", "y")
        nb_storage = band_storage_extent(mesh, expected_nb)
        large_shape = (3, int(meta.nk_tot), nb_storage, nb_storage)
        forward_neighbors = np.asarray(io.read_slab(
            "full_forward_neighbors", shape=(int(meta.nk_tot), 3),
            partition_spec=P(None, None), as_numpy=True), dtype=np.int64)
        links = load_full_bz_links(
            io, mesh=mesh, nk=int(meta.nk_tot), nb_storage=nb_storage,
            nb_logical=expected_nb,
        )
        velocity = io.read_slab(
            VELOCITY_DFT_DATASET, shape=large_shape, partition_spec=spec
        )
        # Small (O(nk*nb) real) host-resident diagnostic, read in the SAME
        # handle as everything above -- one owner, one open, per this
        # loader's own docstring.  Not consulted here; the D3(a) window
        # preflight in ``sc_iteration.load_head_velocity_source`` reads it
        # off the returned object.
        singular_values = load_link_singular_values(io, nb_logical=expected_nb)

    expected_prefix = (3, int(meta.nk_tot))
    if (
        tuple(links.shape[:2]) != expected_prefix
        or links.shape != velocity.shape
        or links.shape[-2] != links.shape[-1]
        or int(links.shape[-1]) < expected_nb
        or forward_neighbors.shape != (int(meta.nk_tot), 3)
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
    )


def load_dft_velocity_head(
    path: str,
    *,
    mesh: Mesh,
    wfn,
    meta,
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
    nb_storage = band_storage_extent(mesh, nb)
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
        expected_reciprocal = (
            np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat)
        )
        refusals = []
        # Schema 3 changes only the link-consumer validation contract.  The
        # DFT velocity payload and all provenance fields are byte-for-byte
        # schema-2 compatible, and this mode deliberately consumes no links.
        if schema not in (2, int(SCHEMA_VERSION)):
            refusals.append(
                f"schema_version={schema}, expected 2 or {SCHEMA_VERSION}")
        if (band_start, band_stop) != (0, nb):
            refusals.append(
                f"band manifold [{band_start},{band_stop}) != [0,{nb})"
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
        # Rank-invariant operands, so this refuses everywhere or nowhere —
        # before the (3, nk, nb, nb) read, still inside the handle.
        if refusals:
            raise ValueError(
                f"{path}: refusing DFT velocity stage:\n  - "
                + "\n  - ".join(refusals)
            )
        velocity = io.read_slab(
            VELOCITY_DFT_DATASET,
            shape=(3, int(meta.nk_tot), nb_storage, nb_storage),
            partition_spec=P(None, None, "x", "y"),
        )
    return DftVelocityHeadData(
        velocity_dft_cart=velocity,
        nb_logical=nb,
        reciprocal_lattice_cart=reciprocal,
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


def covariant_link_derivative(
    delta_h_dft,
    forward_links,
    forward_neighbors,
    *,
    mesh: Mesh,
    kgrid,
    bvec_cart,
):
    """Return the direct finite-link covariant derivative of ``Delta H``.

    Neighbouring operators are transported into the central DFT basis before
    the fourth-order reduced-coordinate stencil is applied.  This is one
    gauge-covariant discrete object; no separately differentiated Hamiltonian
    and connection commutator have to cancel on a finite grid.
    """
    from common.parallel_transport import (
        fourth_order_covariant_derivative,
        make_distributed_band_matmul,
    )

    delta = jnp.asarray(delta_h_dft, dtype=jnp.complex128)
    links = jnp.asarray(forward_links, dtype=jnp.complex128)
    spacing = 1.0 / np.asarray(tuple(int(n) for n in kgrid), dtype=np.float64)
    reduced = fourth_order_covariant_derivative(
        delta,
        links,
        np.asarray(forward_neighbors, dtype=np.int64),
        spacing,
        band_matmul=make_distributed_band_matmul(mesh, n_batch_axes=1),
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
):
    """Embed active DeltaH and the current diagonal sum-band tail."""
    delta = jnp.asarray(delta_h_active)
    tail = jnp.asarray(tail_diagonal)
    if delta.ndim != 3 or delta.shape[-2] != delta.shape[-1]:
        raise ValueError("delta_h_active must be (nk,na,na)")
    if tail.ndim != 2 or tail.shape[0] != delta.shape[0]:
        raise ValueError("tail_diagonal must be (nk,nb_storage)")
    if int(tail.shape[1]) < int(nb_storage):
        raise ValueError(f"tail diagonal extent {tail.shape[1]} < storage {nb_storage}")
    if int(delta.shape[-1]) > int(nb_storage):
        raise ValueError("active DeltaH exceeds the head manifold")
    return _assemble_delta_kernel(mesh, int(nb_storage))(delta, tail)


def _interband_degenerate_weight(
    dE, f_diff, z, s_avg, prefactor, near_degenerate, *, include_surface: bool,
):
    r"""Adler-Wiser interband weight, continuous through ``dE -> 0``.

    ``prefactor * f_diff / (dE * (z**2 - dE**2))`` has a removable
    singularity at ``dE = 0`` for FIXED, nonzero ``z``: by l'Hopital on the
    numerator (``f_diff -> -f'(E_mid) * dE`` as the two energies coalesce),
    ``f_diff / dE -> 0.5*(s_bra + s_ket)``, where ``s = -f'`` is the
    caller's own MP1 Fermi-surface weight (:func:`gw.efermi.
    mp1_negative_derivative`) -- the SAME divided-difference-to-derivative
    limit :func:`gw.w_isdf.compute_chi0_direct_fractional`'s
    ``_fractional_pair_scan_face`` already takes for its own ``z=0`` diagonal
    limit (``w_isdf.py`` ``diagonal_limit = -0.5*(sa+sb)``; the sign here
    is ``+`` rather than ``-`` because this module's ``f_diff`` is built
    ket-minus-bra where that scan's ``df`` is a-minus-b -- same physical
    limit, opposite index convention).  The ``z``-dependence keeps its
    finite-``z`` form throughout: only ``dE`` is taken to a limit, never
    ``z`` -- a resonance (``z`` near ``dE`` at a NON-degenerate pair) is a
    different singularity and is untouched by this branch.

    ``s_avg`` (``0.5*(s_bra+s_ket)``) and ``near_degenerate`` (the
    resolution-scaled ``|dE|`` test) are precomputed by the caller,
    already broadcast to ``dE``'s shape: neither depends on ``omega``, so
    hoisting them out of the inner per-frequency closure avoids
    recomputing a comparison already fixed by the band-pair tile alone.

    ``include_surface`` is a plain Python ``bool``, closed over at trace
    time by the caller (its kernel factory already keys its compilation
    cache on it): when ``False`` -- no ``-f'`` data, i.e. every insulating
    or fixed-occupation deck -- this compiles to EXACTLY the pre-fix
    ``jnp.where(|denom|>1e-16, regular, 0)`` body; the degenerate branch
    is never lowered into that HLO graph and ``s_avg``/``near_degenerate``
    are unused (may be ``None``).

    SCOPE.  This formula (``prefactor * f_diff / (dE * (z**2 - dE**2))``)
    is ``_s_tensor_kernel``'s ONLY; the wing kernel
    ``_head_wing_kernel_face`` builds a structurally
    DIFFERENT ``F_ij = -pref_inter * f_diff / (z**2 - dE**2)`` -- one fewer
    power of ``dE`` (the mixed density/velocity response), because a wing pairs one velocity leg with one
    dimension-1-in-energy density vertex where the head pairs two
    velocity legs.  At fixed nonzero ``z``, THAT formula is already
    continuous as ``dE -> 0`` (``f_diff -> 0`` at the same order as the
    numerator, no compensating ``dE`` in the denominator to cancel), so
    its own ``|denom|>1e-16`` clip is not this same defect -- it guards a
    ``z ~= dE`` resonance, a different singularity this helper does not
    address.  Do not reuse this helper there without rederiving the limit.
    """
    denom = dE * (z * z - dE * dE)
    zero = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)
    regular = prefactor * f_diff / denom
    clipped = jnp.where(jnp.abs(denom) > 1.0e-16, regular, zero)
    if not include_surface:
        return clipped
    z_ok = jnp.abs(z) > 1.0e-15
    degenerate = prefactor * s_avg / (z * z)
    return jnp.where(near_degenerate & z_ok, degenerate, clipped)


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


def _s_tensor_kernel(
    mesh: Mesh, *, nb_logical: int, include_surface: bool = False,
) -> Callable:
    key = ("head_s", id(mesh), int(nb_logical), bool(include_surface))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(
        v_local, e_bra, e_ket, f_bra, f_ket, s_bra, s_ket, omegas, prefactor, eta,
    ):
        nx, ny = v_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        dE = e_bra[:, :, None] - e_ket[:, None, :]
        f_diff = f_ket[:, None, :] - f_bra[:, :, None]
        logical = ((ix[:, None] < nb_logical) & (iy[None, :] < nb_logical))[None, :, :]
        # Sum every energy-ordered band pair.  f_diff is SIGNED: MP1 is
        # not globally monotone and may overshoot slightly outside [0, 1],
        # so filtering on f_v-f_c>0 would not implement the Adler-Wiser
        # occupation difference.  The historical 0/1 path is unchanged
        # because its energy-ordered nonzero differences are positive.
        transition = logical & (dE > 0.0)
        if include_surface:
            scale = jnp.maximum(
                1.0,
                jnp.maximum(jnp.abs(e_bra)[:, :, None], jnp.abs(e_ket)[:, None, :]),
            )
            near_degenerate = (
                jnp.abs(dE) <= 64.0 * jnp.finfo(jnp.float64).eps * scale)
            s_avg = 0.5 * (s_bra[:, :, None] + s_ket[:, None, :])
        else:
            near_degenerate = None
            s_avg = None

        def _one(omega):
            z = omega + 1j * eta
            weight = jnp.where(
                transition,
                _interband_degenerate_weight(
                    dE, f_diff, z, s_avg, prefactor, near_degenerate,
                    include_surface=include_surface,
                ),
                jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
            )
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

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),
            P(None, "x"),
            P(None, "y"),
            P(None, "x"),
            P(None, "y"),
            P(None, "x"),
            P(None, "y"),
            P(None),
            P(),
            P(),
        ),
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
    layout: str = "face",
) -> Callable:
    """Build the canonical face head-wing kernel with bounded centroid tiles."""
    if layout != "face":
        raise ValueError(f"_head_wing_kernel requires layout='face', got {layout!r}")
    return _head_wing_kernel_face(
        mesh, nb_logical=int(nb_logical), include_surface=bool(include_surface))


def _head_wing_kernel_face(
    mesh: Mesh,
    *,
    nb_logical: int,
    include_surface: bool,
) -> Callable:
    """Contract velocity and density vertices in bounded centroid and frequency tiles."""
    key = ("head_wings_face", id(mesh), int(nb_logical), bool(include_surface))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

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
    ):
        nk = v_local.shape[1]
        v_full = jax.lax.all_gather(v_local, ax_x, axis=2, tiled=True)
        v_full = jax.lax.all_gather(v_full, ax_y, axis=3, tiled=True)
        nb_full = v_full.shape[-1]
        ns = bra_mun_local.shape[1]
        mu_x_local = bra_mun_local.shape[2]
        mu_y_local = bra_nmu_local.shape[-1]

        idx = jnp.arange(nb_full)
        logical1d = idx < nb_logical
        logical2d = (logical1d[:, None] & logical1d[None, :])[None, :, :]
        dE = energies[:, :, None] - energies[:, None, :]
        f_diff = occupations[:, None, :] - occupations[:, :, None]
        transition = logical2d & (dE > 0.0)
        if include_surface:
            diagonal = logical2d & (idx[:, None] == idx[None, :])[None, :, :]
            surface_pair = jnp.where(diagonal, surface_weight[:, :, None], 0.0)
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
            """Stream omega in bounded blocks (mirrors the legacy ring's
            own ``_HEAD_WING_FREQUENCY_BLOCK`` discipline); no cross-block
            accumulation is needed since distinct blocks cover distinct
            frequencies (unlike the legacy ring's cross-RING-STEP carry)."""
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

        # ---- Y_x: mu on X (psi_mun's own axis), gather bands over Y ----
        mu_x_block = min(_HEAD_WING_MU_BLOCK, int(mu_x_local))
        from runtime.padding import padded_axis
        mu_x_axis = padded_axis(
            int(mu_x_local), mu_x_block,
            name="face head left-centroid work carrier")
        mu_x_padded = mu_x_axis.carrier
        n_x_blocks = mu_x_padded // mu_x_block
        bra_mun_padded = jnp.pad(
            bra_mun_local,
            ((0, 0), (0, 0), (0, mu_x_padded - mu_x_local), (0, 0)))
        ket_mun_padded = jnp.pad(
            ket_mun_local,
            ((0, 0), (0, 0), (0, mu_x_padded - mu_x_local), (0, 0)))

        def _x_step(_carry, blk):
            zero = jnp.zeros((), dtype=blk.dtype)
            start = blk * mu_x_block
            bra_tile = jax.lax.dynamic_slice(
                bra_mun_padded, (zero, zero, start, zero),
                (nk, ns, mu_x_block, bra_mun_padded.shape[-1]))
            ket_tile = jax.lax.dynamic_slice(
                ket_mun_padded, (zero, zero, start, zero),
                (nk, ns, mu_x_block, ket_mun_padded.shape[-1]))
            bra_full = jax.lax.all_gather(
                bra_tile, ax_y, axis=3, tiled=True)
            ket_full = jax.lax.all_gather(
                ket_tile, ax_y, axis=3, tiled=True)

            def _contract_left(weight):
                return jnp.einsum(
                    "akij,wkij,ksmi,ksmj->wam",
                    jnp.conj(v_full), weight,
                    jnp.conj(bra_full), ket_full,
                    optimize=True,
                )
            blocks = _weighted_stack(_contract_left)
            return _carry, blocks.reshape(
                n_omega_padded, int(v_full.shape[0]), mu_x_block)

        _, y_chunks = jax.lax.scan(
            _x_step, None, jnp.arange(n_x_blocks, dtype=jnp.int32), unroll=1)
        n_vertex = int(v_full.shape[0])
        Y_x = jnp.moveaxis(y_chunks, 0, 2).reshape(
            n_omega_padded, n_vertex,
            mu_x_padded)[:n_omega, :, :mu_x_local]

        # ---- Z_y: mu on Y (psi_nmu's own axis), gather bands over X ----
        mu_y_block = min(_HEAD_WING_MU_BLOCK, int(mu_y_local))
        mu_y_axis = padded_axis(
            int(mu_y_local), mu_y_block,
            name="face head right-centroid work carrier")
        mu_y_padded = mu_y_axis.carrier
        n_y_blocks = mu_y_padded // mu_y_block
        bra_nmu_padded = jnp.pad(
            bra_nmu_local,
            ((0, 0), (0, 0), (0, 0), (0, mu_y_padded - mu_y_local)))
        ket_nmu_padded = jnp.pad(
            ket_nmu_local,
            ((0, 0), (0, 0), (0, 0), (0, mu_y_padded - mu_y_local)))

        def _y_step(_carry, blk):
            zero = jnp.zeros((), dtype=blk.dtype)
            start = blk * mu_y_block
            bra_tile = jax.lax.dynamic_slice(
                bra_nmu_padded, (zero, zero, zero, start),
                (nk, bra_nmu_padded.shape[1], ns, mu_y_block))
            ket_tile = jax.lax.dynamic_slice(
                ket_nmu_padded, (zero, zero, zero, start),
                (nk, ket_nmu_padded.shape[1], ns, mu_y_block))
            bra_gathered = jax.lax.all_gather(
                bra_tile, ax_x, axis=1, tiled=True)
            ket_gathered = jax.lax.all_gather(
                ket_tile, ax_x, axis=1, tiled=True)
            bra_full = jnp.transpose(bra_gathered, (0, 2, 3, 1))
            ket_full = jnp.transpose(ket_gathered, (0, 2, 3, 1))

            def _contract_right(weight):
                return jnp.einsum(
                    "ksmi,ksmj,wkij,bkij->wmb",
                    bra_full, jnp.conj(ket_full), weight, v_full,
                    optimize=True,
                )
            blocks = _weighted_stack(_contract_right)
            return _carry, blocks.reshape(
                n_omega_padded, mu_y_block, n_vertex)

        _, z_chunks = jax.lax.scan(
            _y_step, None, jnp.arange(n_y_blocks, dtype=jnp.int32), unroll=1)
        Z_y = jnp.moveaxis(z_chunks, 0, 1).reshape(
            n_omega_padded, mu_y_padded,
            n_vertex)[:n_omega, :mu_y_local, :]

        return Y_x, Z_y

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),   # v_local
            P(None, None, "x", "y"),   # bra_mun_local  (PSI_MUN_SPEC)
            P(None, None, "x", "y"),   # ket_mun_local  (PSI_MUN_SPEC)
            P(None, "x", None, "y"),   # bra_nmu_local  (PSI_NMU_SPEC)
            P(None, "x", None, "y"),   # ket_nmu_local  (PSI_NMU_SPEC)
            P(None, None),             # energies (nk, nb_full), replicated
            P(None, None),             # occupations
            P(None, None),             # surface_weight
            P(None),                   # omegas
            P(),
            P(),
            P(),
        ),
        out_specs=(P(None, None, "x"), P(None, "y", None)),
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
    so it is always gathered at its full stored ``nb_full`` width.  ``v``/
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
    if getattr(wfns, "layout", None) != "face":
        raise ValueError("head_wings_sharded requires the canonical face layout")
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
):
    """Stream parent children and pad response tables to the canonical face band extent."""
    if (getattr(wfns, "layout", None) == "face" and wfns.psi_mun is None
            and getattr(wfns, "green_parent", None) is not None):
        # Parents-only storage: the wings are a sum over k, so stream the
        # children of one raw parent at a time (w_isdf.iter_parent_children_
        # faces) and accumulate; no full-k face is ever resident.  The
        # velocity has already been restored by the typed polar action.
        if body_bra_wfns is not None or body_ket_wfns is not None:
            raise ValueError(
                "head_wings_sharded(layout='face'): separately supplied "
                "endpoint bundles are not combined with parents-only storage.")
        from gw.w_isdf import iter_parent_children_faces
        v_all = jnp.asarray(velocity_cart, dtype=jnp.complex128)
        e_all = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
        f_all = jnp.asarray(occupations_kn, dtype=jnp.float64)
        s_all = (None if surface_weight_kn is None
                 else jnp.asarray(surface_weight_kn, dtype=jnp.float64))
        Y_x = Z_y = None
        for rows, child in iter_parent_children_faces(
                wfns.green_parent, mesh, slices=wfns.slices):
            r = device_put_process_local(
                np.asarray(rows, np.int32), NamedSharding(mesh, P(None)))
            y, z = _head_wings_sharded_face(
                jnp.take(v_all, r, axis=1), child, jnp.take(e_all, r, axis=0),
                jnp.take(f_all, r, axis=0), omegas_ry, mesh=mesh,
                nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
                nspinor=nspinor, eta_ry=eta_ry,
                surface_weight_kn=(None if s_all is None
                                   else jnp.take(s_all, r, axis=0)))
            Y_x = y if Y_x is None else Y_x + y
            Z_y = z if Z_y is None else Z_y + z
        return Y_x, Z_y

    bra_wfns = wfns if body_bra_wfns is None else body_bra_wfns
    ket_wfns = wfns if body_ket_wfns is None else body_ket_wfns

    face_shapes = None
    for endpoint_name, endpoint in (
            ("wfns", wfns), ("body_bra_wfns", bra_wfns),
            ("body_ket_wfns", ket_wfns)):
        if getattr(endpoint, "layout", None) != "face":
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
    return _head_wing_kernel(
        mesh, nb_logical=int(nb_logical),
        include_surface=bool(include_surface), layout="face")(
            v, bra_wfns.psi_mun, ket_wfns.psi_mun,
            bra_wfns.psi_nmu, ket_wfns.psi_nmu,
            e, f, surface, omega,
            jnp.asarray(pref_inter, dtype=jnp.complex128),
            jnp.asarray(pref_surface, dtype=jnp.complex128),
            jnp.asarray(float(eta_ry), dtype=jnp.float64),
        )


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
    if getattr(wfns, "layout", None) != "face":
        raise ValueError("static_head_wings_sharded requires the canonical face layout")
    return _static_head_wings_sharded_face(
        wfns, surface_weight_kn, mesh=mesh, nb_logical=nb_logical,
        nk_tot=nk_tot, nspin=nspin, nspinor=nspinor)


def _static_head_wings_kernel_face(mesh: Mesh) -> Callable:
    """Cached shard_map kernel: a LOCAL density-weighted band sum per
    face orientation, then one ``psum`` over the mesh axis holding the
    summed band index.  No ring, no gather — see
    :func:`_static_head_wings_sharded_face`'s docstring for why the
    static vertex does not need one (it is diagonal in mu, unlike the
    dynamic wings' genuine (i,j) operator)."""
    key = ("static_head_wings_face", id(mesh))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(psi_mun_local, psi_nmu_local, weight_full):
        nk = psi_mun_local.shape[0]

        n_y_local = psi_mun_local.shape[-1]
        y_coord = jax.lax.axis_index(ax_y)
        y_zero = jnp.zeros((), dtype=y_coord.dtype)
        y_start = y_coord * n_y_local
        weight_y = jax.lax.dynamic_slice(
            weight_full, (y_zero, y_start), (nk, n_y_local))
        density_x = jnp.sum(jnp.square(jnp.abs(psi_mun_local)), axis=1)
        left = jax.lax.psum(
            jnp.einsum("kn,kmn->m", weight_y, density_x), ax_y)

        n_x_local = psi_nmu_local.shape[1]
        x_coord = jax.lax.axis_index(ax_x)
        x_zero = jnp.zeros((), dtype=x_coord.dtype)
        x_start = x_coord * n_x_local
        weight_x = jax.lax.dynamic_slice(
            weight_full, (x_zero, x_start), (nk, n_x_local))
        density_y = jnp.sum(jnp.square(jnp.abs(psi_nmu_local)), axis=2)
        right = jax.lax.psum(
            jnp.einsum("kn,knm->m", weight_x, density_y), ax_x)
        return left, right

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),   # psi_mun_local  (PSI_MUN_SPEC)
            P(None, "x", None, "y"),   # psi_nmu_local  (PSI_NMU_SPEC)
            P(None, None),             # weight (nk, nb_full), replicated
        ),
        out_specs=(P("x"), P("y")),
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
    if (wfns.psi_mun is None
            and getattr(wfns, "green_parent", None) is not None):
        # Parents-only storage: stream the children star by star (see
        # _head_wings_sharded_face); the static wing is a plain k sum.
        from gw.w_isdf import iter_parent_children_faces
        left = right = None
        for rows, child in iter_parent_children_faces(
                wfns.green_parent, mesh, slices=wfns.slices):
            r = jnp.asarray(rows, dtype=jnp.int32)
            a, b = _static_head_wings_sharded_face(
                child, jnp.take(surface, r, axis=0), mesh=mesh,
                nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
                nspinor=nspinor)
            left = a if left is None else left + a
            right = b if right is None else right + b
        return left, right
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
    if int(surface.shape[0]) != nk_mun:
        raise ValueError("centroid wavefunctions do not cover static weights")
    width = int(surface.shape[1])
    if width > nb_full:
        raise ValueError(
            "static head surface weights are wider than the face bundle's "
            f"stored band extent: {width} > {nb_full}")
    if width < nb_full:
        surface = jnp.pad(surface, ((0, 0), (0, nb_full - width)))
    logical = jnp.arange(nb_full)[None, :] < int(nb_logical)
    weight = device_put_process_local(
        jnp.where(logical, surface, 0.0),
        NamedSharding(mesh, P(None, None)))
    prefactor = -2.0 / (
        float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    left, right = _static_head_wings_kernel_face(mesh)(
        wfns.psi_mun, wfns.psi_nmu, weight)
    return prefactor * left, prefactor * right


def _drude_tensor_kernel(mesh: Mesh, *, nb_logical: int) -> Callable:
    """Compile the Fermi-surface velocity contraction once per band shape."""
    key = ("head_drude", id(mesh), int(nb_logical))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(v_local, surface_weight_x, prefactor):
        nx, ny = v_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        diagonal = (
            (ix[:, None] == iy[None, :])
            & (ix[:, None] < nb_logical)
            & (iy[None, :] < nb_logical)
        )[None, :, :]
        weight = jnp.where(diagonal, surface_weight_x[:, :, None], 0.0)
        local = prefactor * jnp.einsum(
            "akij,kij,bkij->ab",
            jnp.conj(v_local), weight, v_local, optimize=True,
        )
        return jax.lax.psum(local, (ax_x, ax_y))

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, None, "x", "y"), P(None, "x"), P()),
        out_specs=P(None, None),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def head_drude_tensor_sharded(
    velocity_cart,
    surface_weight_kn,
    *,
    mesh: Mesh,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
):
    """Return the ab-initio Drude tensor ``D_ab`` in Rydberg units.

    ``D_ab = C/(Omega Nk) sum_kn (-df/dE) v_a,nn* v_b,nn`` with state
    capacity ``C=2/(nspin*nspinor)``.  Consequently the directional plasma
    frequency in this tree's Rydberg convention is
    ``omega_p(qhat)^2 = 8*pi*qhat.D.qhat``.  The diagonal QSGW velocities
    include the nonlocal-pseudopotential and covariant-rotation terms.  At
    the initial iteration they are exactly the saved DFT ``dH/dk``; only a
    subsequent self-consistent Hamiltonian update makes them QSGW.  No fitted
    or experimental plasma frequency enters.
    """
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if v.ndim != 4 or v.shape[0] != 3 or v.shape[2] != v.shape[3]:
        raise ValueError(
            f"velocity_cart must be (3,nk,nb,nb), got {v.shape}.")
    if tuple(surface.shape) != tuple(v.shape[1:3]):
        raise ValueError(
            f"surface_weight_kn shape {surface.shape} does not match "
            f"velocity (nk,nb)={v.shape[1:3]}.")
    if not (0 < int(nb_logical) <= int(v.shape[2])):
        raise ValueError(
            f"need 0 < nb_logical <= stored nb, got "
            f"{nb_logical}, {v.shape[2]}.")
    v, _e, _f, surface = _pad_head_band_manifold(
        v, surface, surface, surface, mesh=mesh)
    pref = 2.0 / (
        float(cell_volume)
        * float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    tensor = _drude_tensor_kernel(mesh, nb_logical=int(nb_logical))(
        v, surface, jnp.asarray(pref, dtype=jnp.complex128))
    return 0.5 * (tensor + jnp.conj(tensor.T))


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
    interband = _s_tensor_kernel(
        mesh, nb_logical=int(nb_logical), include_surface=bool(include_surface),
    )(
        v,
        e,
        e,
        f,
        f,
        surface,
        surface,
        omega,
        jnp.asarray(pref, dtype=jnp.complex128),
        jnp.asarray(float(eta_ry), dtype=jnp.float64),
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
        mesh=mesh,
        nb_logical=int(nb_logical),
        cell_volume=float(cell_volume),
        nk_tot=int(nk_tot),
        nspin=int(nspin),
        nspinor=int(nspinor),
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
    if (complete
            and ((uniform_gauge.dgamma_dq_raw is None)
                 != (uniform_gauge.d2gamma_dq2_raw is None))):
        raise ValueError(
            "uniform-gauge Hall transaction has only one of its optional "
            "first/second transfer-jet fields")

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
    if (complete and uniform_gauge.dgamma_dq_raw is not None
            and tuple(uniform_gauge.dgamma_dq_raw.shape) != (
                nk_tot, 3, 3, storage, storage)):
        raise ValueError(
            "uniform-gauge Hall transaction has an invalid first transfer "
            f"jet shape {uniform_gauge.dgamma_dq_raw.shape}")
    if (complete and uniform_gauge.d2gamma_dq2_raw is not None
            and tuple(uniform_gauge.d2gamma_dq2_raw.shape) != (
                nk_tot, 3, 3, 3, storage, storage)):
        raise ValueError(
            "uniform-gauge Hall transaction has an invalid second transfer "
            f"jet shape {uniform_gauge.d2gamma_dq2_raw.shape}")

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
    static_kappa2 = response.static_kappa2_bohr2
    if use_fold and abs(response.omegas[index]) <= 1.0e-14:
        static_kappa2 = _fold_static_kappa2(
            response, W_body_gamma, float(meta.cell_volume), mesh)
    return head_samples_from_s(
        S_effective[None, :, :],
        (response.omegas[index],),
        wfn=wfn,
        meta=meta,
        config=config,
        static_kappa2_bohr2=static_kappa2,
        response_kind=("full_local_fields" if use_fold
                       else "direct_irreducible"),
        source_prefix=("head_schur" if use_fold else "head_direct"),
    )[0]


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
            W_gamma.append(W_role[0])
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
    static_kappa2 = response.static_kappa2_bohr2
    if use_fold and static_kappa2 is not None:
        static_indices = [
            i for i, z in enumerate(response.omegas) if abs(z) <= 1.0e-14]
        if len(static_indices) != 1:
            raise ValueError(
                "static metallic head requires exactly one z=0 response")
        static_kappa2 = _fold_static_kappa2(
            response, W_gamma[static_indices[0]], float(meta.cell_volume), mesh)
    samples = head_samples_from_s(
        S_effective,
        response.omegas,
        wfn=wfn,
        meta=meta,
        config=config,
        static_kappa2_bohr2=static_kappa2,
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
) -> tuple[object, ...]:
    """Convert replicated 3x3 S tensors to mini-BZ averaged head samples."""
    from gw.head_correction import (
        HeadResponseKind, HeadSample, resolve_head_override)
    from gw.isdf_fitting import mem_probe
    from gw.vcoul import compute_q0_averages

    # This is the first host readback of ``S_cart_omega`` for callers that
    # do not already sync it (``finalize_iteration_head_samples`` now does,
    # at its own site -- see the lifetime-boundary comment there).  Report
    # what is live HERE too so a caller that skips that boundary (the
    # per-sample ``finalize_iteration_head_sample`` diagnostic entry point,
    # or a future one) still gets an attributable snapshot instead of a bare
    # RESOURCE_EXHAUSTED at this line.
    mem_probe("qsgw_head.head_samples_from_s.pre_readback")
    S_host = np.asarray(S_cart_omega, dtype=np.complex128)
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
    out = []
    kind = HeadResponseKind(response_kind)
    for z, S in zip(omegas, S_host):
        override = resolve_head_override(params, z)
        if override is not None:
            out.append(override)
            continue
        is_static_metal = (
            static_kappa2_bohr2 is not None and abs(z) <= 1.0e-14)
        vc0, wc0 = compute_q0_averages(
            wfn,
            jnp.asarray(0.0, dtype=jnp.float64),
            meta,
            S_cart=None if is_static_metal else S,
            static_kappa2=(
                jnp.asarray(static_kappa2_bohr2, dtype=jnp.float64)
                if is_static_metal else None),
            analytic_sphere=bool(getattr(
                config.head, "analytic_q0_sphere",
                config.head.head_minibz_average)),
        )
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
                S_cart=None if is_static_metal else S,
                response_kind=kind,
            )
        )
    return tuple(out)


def build_iteration_head_response(
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
    wfns_qp=None,
    eta_ry: float | None = None,
) -> IterationHeadResponse:
    """Build current-basis direct head and, when requested, its wings.

    ``forward_links=None`` is ``sc_head_update = dft_velocity``: no link
    manifold is resident, so the covariant ``DΔH`` correction is dropped
    and the bare DFT p-matrix velocity enters.  ``delta_h_dft`` is then
    unused and may be None.  Everything downstream of the velocity —
    the per-iteration rotation into the QP basis, S(z), the Drude term, the
    ISDF wings, the static κ² — is the SAME code on both routes.
    """
    v_dft_basis = jnp.asarray(velocity_dft_cart, dtype=jnp.complex128)
    if forward_links is not None:
        if forward_neighbors is None:
            raise ValueError(
                "forward_neighbors are required when forward_links are present"
            )
        v_dft_basis = v_dft_basis + covariant_link_derivative(
            delta_h_dft,
            forward_links,
            forward_neighbors,
            mesh=mesh,
            kgrid=kgrid,
            bvec_cart=bvec_cart,
        )
    v_qp = rotate_velocity_active_to_qp(v_dft_basis, U_dft_to_qp, mesh=mesh)
    resolved_eta_ry = (
        float(config.head.wcoul0_eta)
        if eta_ry is None else float(eta_ry)
    )
    # Physical state multiplicity belongs to the source WFN.  A
    # kinetic-balance lift changes only the stored spinor representation.
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
    if (
        wfns_qp is not None
        and surface_weight_qp_kn is not None
        and any(abs(z) <= 1.0e-14 for z in omegas)
    ):
        static_Y_x, static_Z_y = static_head_wings_sharded(
            wfns_qp,
            surface_weight_qp_kn,
            mesh=mesh,
            nb_logical=int(nb_logical),
            nk_tot=int(meta.nk_tot),
            nspin=int(wfn.nspin),
            nspinor=normalization_nspinor,
        )
        from gw.w_isdf import compute_chi0_static_fractional_gamma
        static_chi_body_gamma = compute_chi0_static_fractional_gamma(
            wfns_qp,
            energies_qp_kn_ry,
            occupations_qp_kn,
            surface_weight_qp_kn,
            meta,
            mesh,
            nb_logical=int(nb_logical),
        )
    static_kappa2 = None
    if surface_weight_qp_kn is not None:
        capacity = 2.0 / (
            float(max(int(wfn.nspin), 1))
            * float(max(normalization_nspinor, 1)))
        # Tetrahedron weights arrive multiplied by Nk to share the distributed
        # Drude contraction's interface.  Undo that factor for the normalized
        # BZ density of states, then use kappa_TF^2=8*pi*DOS_Ry/Omega.
        dos_ry_per_cell = capacity * float(
            np.sum(np.asarray(surface_weight_qp_kn, dtype=np.float64))) / float(
                meta.nk_tot)
        static_kappa2 = (
            8.0 * np.pi * dos_ry_per_cell / float(meta.cell_volume))
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
    )


def read_authenticated_dipole_velocity(
    dipole_path, *, wfn, meta, config, wfn_fingerprint_binding=None,
):
    """Read file-wedge velocities and restore their polar time-odd full-k action."""
    import h5py
    from symmetry_maps import unfold_file_wedge_polar_matrix

    # Fail before the host read and every sharded head allocation.  Shape does
    # not identify a velocity artifact: in particular, a two-spinor dipole and
    # a kinetic-balance four-spinor dipole have the same (3,nk,nb,nb) shape.
    # The producer owns both the stamp grammar and sign resolution; consume
    # those owners directly rather than mirroring either convention here.
    from psp.get_dipole_mtxels import (
        check_dipole_provenance, resolve_vnl_velocity_sign)
    expected_vnl_sign = resolve_vnl_velocity_sign(
        None, config.vnl_velocity_sign)
    from common.four_current_model import resolve_four_current_representation
    representation = resolve_four_current_representation(
        bool(getattr(config, "bispinor", int(meta.nspinor) == 4)),
        getattr(config, "bispinor_gw", "bare_transverse"))
    if not check_dipole_provenance(
            dipole_path,
            wfn=wfn,
            nval=int(config.nval),
            ncond=int(config.ncond),
            nband=int(config.nband),
            bispinor=representation.scalar_head_bispinor,
            skip_vnl=False,
            vnl_mode="analytic",
            vnl_velocity_sign=expected_vnl_sign,
            wfn_fingerprint_binding=wfn_fingerprint_binding):
        raise ValueError(
            "GATE dft_head_dipole_provenance: the full head received an "
            "unauthenticated dipole artifact.\n"
            f"  got:  dipole_file = {dipole_path!r}; at least one WFN, "
            "q->0 coverage, VNL, or representation stamp mismatched\n"
            "  want: dipole.h5 regenerated from this run's exact deck\n"
            "  why:  S_direct and the wings must use the same WFN and "
            "velocity operator as the finite-q charge response")
    sym = wfn.symmetry()
    b0, b4 = int(meta.b_id_0), int(meta.b_id_4_chi_user)
    with h5py.File(dipole_path, "r") as h5:
        velocity = h5["dipole_cart"]
        if velocity.shape[1] != int(sym.nk_tot):
            raise ValueError(
                "dipole_cart must retain its full-BZ file indexing; "
                f"got {velocity.shape[1]} rows, want {sym.nk_tot}")
        parents = np.stack([
            velocity[:, int(row), b0:b4, b0:b4]
            for row in sym.kirr_fullids])
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
) -> IterationHeadResponse:
    """Build the one-shot DFT head on exactly the chi0 band manifold.

    This is the non-self-consistent entry to the same sharded direct-head and
    wing kernels used by QSGW.  In particular, both ``S_direct`` and ``Y/Z``
    use ``[b0,b4_chi)``; constructing the scalar from every band in a larger
    dipole file while the body uses ``number_bands_chi`` is refused by shape
    and slicing here rather than silently mixing transition manifolds.
    """
    import os

    dipole_path = os.path.join(input_dir, "dipole.h5")
    if not os.path.exists(dipole_path):
        raise FileNotFoundError(
            "head_correction=full requires dipole.h5 to build the direct "
            f"head and wings; missing {dipole_path}.")
    velocity_cart = read_authenticated_dipole_velocity(
        dipole_path, wfn=wfn, meta=meta, config=config,
        wfn_fingerprint_binding=wfn_fingerprint_binding)
    b0 = int(meta.b_id_0)
    b4 = int(meta.b_id_4_chi_user)
    nb_logical = b4 - b0
    energies = jnp.asarray(wfns.enk[:, :nb_logical])
    occupations = jnp.asarray(wfns.occ[:, :nb_logical])
    if velocity_cart.shape[1:] != (
            int(meta.nk_tot), nb_logical, nb_logical):
        raise ValueError(
            "dipole/chi head manifold mismatch: sliced velocity has "
            f"{velocity_cart.shape}, expected "
            f"(3,{int(meta.nk_tot)},{nb_logical},{nb_logical}) for global "
            f"bands [{b0},{b4}).")
    z = np.asarray(omegas_ry, dtype=np.complex128).reshape(-1)
    # ``meta.nspinor`` is four for the bispinor representation, whereas
    # response normalization counts the source-WFN states.
    normalization_nspinor = int(meta.nspinor_wfnfile)
    S = head_s_tensor_sharded(
        jnp.asarray(velocity_cart), energies, occupations, z,
        mesh=mesh, nb_logical=nb_logical,
        cell_volume=float(meta.cell_volume), nk_tot=int(meta.nk_tot),
        nspin=int(wfn.nspin), nspinor=normalization_nspinor,
        eta_ry=float(config.head.wcoul0_eta))
    Y_x, Z_y = head_wings_sharded(
        jnp.asarray(velocity_cart), wfns, energies, occupations, z,
        mesh=mesh, nb_logical=nb_logical, nk_tot=int(meta.nk_tot),
        nspin=int(wfn.nspin), nspinor=normalization_nspinor,
        eta_ry=float(config.head.wcoul0_eta))
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
    if 0 < n_occ_local < nb_logical:
        efermi = 0.5 * (
            float(np.max(e_host[:, n_occ_local - 1]))
            + float(np.min(e_host[:, n_occ_local])))
    else:
        efermi = 0.0
    return IterationHeadResponse(
        omegas=tuple(complex(value) for value in z),
        S_direct=S, Y_x=Y_x, Z_y=Z_y,
        static_kappa2_bohr2=None,
        static_Y_x=None, static_Z_y=None, static_chi_body_gamma=None,
        sigma_energies_ry=e_host[:, :int(meta.nb_sigma)],
        sigma_occupations=np.asarray(occupations)[:, :int(meta.nb_sigma)],
        efermi_ry=efermi)


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
