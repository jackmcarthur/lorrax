"""Finite-link covariant velocity for the self-consistent GW head.

The preprocessing job owns wavefunctions.  This module deliberately does
not: its production inputs are saved nearest-neighbour overlap links, the
independently exact DFT velocity, and the current fixed-DFT-basis QSGW
Hamiltonian.  It implements the single gauge-covariant discrete object

    v_Q = v_DFT + D_link (H_Q - H_DFT)

where neighbouring operators are parallel-transported before a fourth-order
finite-difference stencil is applied.  No separately differentiated H and
Berry-connection commutator have to cancel on a finite grid.  The module
rebuilds the tiny Cartesian S tensor from the resulting band-tiled velocity.
When current centroid wavefunctions are supplied it also builds the two
q-linear head/body wings.  The wings stay centroid-sharded; only the final
``(n_omega, 3, 3)`` Schur-reduced tensor is replicated.

Frequency streaming and the head-finalize lifetime boundary
-------------------------------------------------------------
``_s_tensor_kernel`` (feeding ``S_direct``) and ``_head_wing_kernel``
(feeding ``Y_x``/``Z_y``) both hold a per-omega ``(nk, nx_local, ny_local)``
band-pair temporary while contracting velocities into the tiny head/wing
output.  The wing kernel has always bounded this at
``_HEAD_WING_FREQUENCY_BLOCK`` frequencies per ring step; the S-tensor
kernel did not -- it ran ``jax.vmap(_one)(omegas)`` over the FULL omega
axis, an unbounded twin of the fold's one-device temporary commit
d2d6d521 already fixed (same "XLA selects a full-matrix temporary even
though the public result is tiny" failure mode, on the omega axis instead
of the centroid one).  Fixed 2026-08-22
(fix/head-fold-streamed-2026-08-22): ``_s_tensor_kernel`` now streams
omega in the same block size via ``lax.scan``, so its compiled peak is
flat past one block regardless of how many frequencies a caller (a
GN-N-pole fit, a dense MPA walk) asks for -- verified by
``tests/test_mpa_dynamic_head.py::
test_s_tensor_kernel_temp_size_is_bounded_past_one_frequency_block``.

``qsgw_head.py`` also had zero ``block_until_ready`` calls anywhere,
unlike ``screening.py``'s per-stage sync discipline (chi.exec, W.exec,
...).  Every array this module builds therefore stayed queued and
unattributed on the device stream until the FIRST host readback anywhere
downstream -- historically ``head_samples_from_s``'s
``np.asarray(S_cart_omega)``, which is why an OOM whose true buffer lived
anywhere upstream of it (in this module or earlier) always surfaced at
that one line with no attribution.  ``build_dft_head_response`` and
``finalize_iteration_head_samples`` now each end their own stage on an
explicit ``jax.block_until_ready`` plus a
``gw.isdf_fitting.mem_probe(...)`` call (env-gated on
``LORRAX_MEM_DEBUG=1``, the SAME probe ``isdf_fitting``/``gw_init`` already
use for the zeta-fit/V_q HBM lifecycle -- single source of truth, not a
second memory-diagnostic helper).  This is a pure scheduling change: it
does not alter any value, only when the allocator is charged for it and
where a failure is reported from.

Measured 2026-08-22 (see
``runs/MoS2/86_bgw_lorrax_scaling_20260819/points/k9_c600_integ/lorrax/
attempts/headfold_hlo_probe_20260822/``): at the production MoS2 9x9x1
GN-PPM shape (nk=81, nb_logical=626, n_rmu=5288, n_omega=2, REAL 16-process
4x4 A100 mesh -- matching the failing run exactly), the compiled peak of
``_s_tensor_kernel`` + ``_head_wing_kernel`` + ``fold_cartesian_head_wings_
sharded`` together is ~6 GiB, far short of the 87.77 GB
(``bfc_allocator.cc`` "trying to allocate 81.74GiB") single-buffer request
the production run hit at this same seam.  That request also exceeds the
63.82 GB XLA pool this run reported at startup, so it cannot be explained
by crowding from other live state either -- ONE buffer here is genuinely
too large.  The three kernels audited above are RULED OUT as that buffer's
source by this measurement; it was not isolated further within this
session's scope.  The lifetime-boundary syncs and mem_probe calls above
give the next production attempt (or the next agent) an attributable,
measured live-array snapshot at each stage instead of the single
unattributed sync point this module had before.

``low_mem_bands`` face-layout wings (2026-08-22)
-------------------------------------------------
``head_wings_sharded``/``static_head_wings_sharded`` (feeding ``Y_x``/
``Z_y``/``static_Y_x``/``static_Z_y``) now dispatch on ``wfns.layout``
(``reports/gwjax_low_mem_bands_audit_2026-08-22/report.md``, census rows
6/7): ``layout='legacy'`` is the exact untouched body; ``layout='face'``
routes to ``_head_wings_sharded_face``/``_static_head_wings_sharded_face``,
whose kernels (``_head_wing_kernel_face``, ``_static_head_wings_kernel_
face``) never hold a band-replicated psi copy the way the legacy ring
does — see ``_head_wing_kernel_face``'s own docstring for the bounded
mu-blocked-gather algorithm and its relationship (none) to the separately
registered, still-open v-sharding-reshard 81.74-GiB legacy defect
(``KNOWN_LORRAX_ISSUES.md``).  ``head_s_tensor_sharded`` (feeding
``S_direct``) needs no face port — it never touches psi.

Units and coordinates
---------------------
Hamiltonians and frequencies are Ry.  ``bvec_cart`` has reciprocal lattice
vectors as rows in 1/bohr, matching ``blat * WfnLoader.bvec``.  The
production finite-link stencil differentiates on the reduced-coordinate
kappa grid and ``B^{-1}`` converts that covector to Cartesian k.  There is
no extra hbar conversion in LORRAX's Ry/bohr velocity convention.  (An
FFT-based spectral derivative plus a separate finite-link connection
commutator existed here as a second discretization of ``D_k Sigma``; it is
retired as of the 2026-08-23 retirement sweep — the Si velocity
expeditions measured its correction as ~uncorrelated with the true one on
real SOC data, not merely lower-order, because a finite grid gives the
split no exact product rule to cancel truncation error against.  See
``covariant_link_derivative``, the sole production/gate discretization.)

TIME-REVERSAL PARITY OF THE QSGW VELOCITY — the convention, derived
--------------------------------------------------------------------
Written out here because a head-correction lane depends on it, a wrong
sign in it is SILENT on a TRS-broken deck, and until now only the bare
term's parity was recorded anywhere (``docs/architecture/symmetry_register.md``
§"The three rows that are NOT one operation": ``dipole_cart`` needs a −1
on the antiunitary rows, measured as ``rel 2.000`` without it).

Let ``Theta`` be the antiunitary, and take the full-BZ gauge LORRAX
generates, ``u_n(−k) = Theta u_n(k)`` — which is exactly what
``symmetry_maps.unfold_psi`` builds on a Theta row (``i sigma_y . conj``
for a spinor, plain ``conj`` for a scalar).  Antiunitarity gives
``<Theta a|Theta b> = conj(<a|b>)``, so for any operator with
``Theta O Theta^-1 = s O`` (``s = ±1``) the band matrix obeys, ELEMENTWISE
and with no transpose,

    O_mn(−k) = s * conj(O_mn(k)).                                    (1)

Apply (1) to the three terms of ``v^Q = v^DFT + d_k Sigma − i[A, Sigma]``:

* ``H``, ``Sigma``, ``kin_ion``, ``V_H`` are EVEN (``s = +1``).
* Differentiation flips the parity.  With ``M(−k) = conj(M(k))``, write
  ``g(k) = M(−k)``; then ``d_i g = conj(d_i M)`` and also
  ``d_i g = −(d_i M)(−k)``, hence ``(d_i M)(−k) = −conj((d_i M)(k))``.
  So ``d_k Sigma`` and ``d_k H`` are ODD.
* The Berry connection is EVEN.  ``A_i = i <u_m|d_i u_n>``; the same two
  routes give ``<u_m(−k)|(d_i u_n)(−k)> = −conj(<u_m|d_i u_n>(k))``, and
  the explicit ``i`` turns that into ``A_i(−k) = +conj(A_i(k))``.
* Therefore ``−i[A_i, Sigma](−k) = −i conj([A_i, Sigma](k)) =
  −conj(−i[A_i, Sigma](k))``: the commutator term is ODD too.

**All three terms carry the SAME parity, so the QSGW velocity obeys the
same relation the bare one does:**

    v^Q_i(−k) = − conj( v^Q_i(k) )     [and each term separately].   (2)

Nothing about the correction flips the sign, which is the useful half of
the result: a sign error in ``d_k Sigma`` or in the commutator shows up as
a parity violation of the assembled ``v^Q`` rather than cancelling.

SCOPE, and the two ways (2) stops being an elementwise statement.
(a) It is a statement about the gauge in which ``u_n(−k) = Theta u_n(k)``.
A full-BZ row reached from its IBZ parent by an unrelated SPATIAL
operation is in a gauge related to that one by a little-group rotation, so
the elementwise form holds only up to that band-space unitary — the same
"one consistent row per orbit" question ``symmetry_maps.qgrid_trs`` settles
on the q axis.
(b) With ``Theta^2 = −1`` (spinor / SOC) the partner of band ``n`` at
``−k`` is its KRAMERS partner, and within a degenerate doublet the label
is gauge-arbitrary.
:func:`trs_velocity_parity_residual` therefore reports the band-TRACE
statistic as its verdict — ``tr v_i`` is invariant under any unitary
mixing inside the retained window, so it survives both (a) and (b) — and
the elementwise number only as a diagnostic.

AND IT IS AN IDENTITY ONLY WHERE TIME REVERSAL HOLDS.  On a ferromagnet
``Theta`` is not a symmetry, ``tr v_i(−k) != −tr v_i(k)`` in general (that
asymmetry is the anomalous-velocity physics, not a bug), and the residual
must not be gated.  The measured verdict is ``WfnLoader.trs_holds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map


__all__ = [
    "DftVelocityHeadData",
    "IterationHeadResponse",
    "IterationHeadSamples",
    "ParallelTransportHeadData",
    "StaticGaugeFirstOrderComponent",
    "StaticGaugeHallTransaction",
    "StaticGaugeSecondOrderComponent",
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
    "static_gauge_first_order_component_sharded",
    "static_gauge_second_order_component_sharded",
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
    :class:`gw.head_correction.StaticGaugeHeadResponse`.  The fingerprint is
    copied from the *same* uniform-gauge sweep that supplied ``Gamma_raw``;
    it therefore remains inseparable from the current/contact/transfer-jet
    Hamiltonian identity that a complete response artifact must carry.

    The large ``Gamma_raw`` band matrix remains sharded over both processor
    axes and is not retained here.  The only replicated product is the
    three-component Hall vector.
    """

    sigma_H: jax.Array
    hamiltonian_config_operator_fingerprint: str
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
    never reaches this call.  ``jax.device_put`` here places ``v`` on the
    mesh directly from its single source device, before it ever reaches a
    kernel, so no jit call in this module dispatches a foreign sharding.
    """
    nb = int(v.shape[-1])
    alignment = lcm(int(mesh.shape["x"]), int(mesh.shape["y"]))
    nb_padded = ((nb + alignment - 1) // alignment) * alignment
    if nb_padded != nb:
        pad = nb_padded - nb
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, pad)))
        e = jnp.pad(e, ((0, 0), (0, pad)))
        f = jnp.pad(f, ((0, 0), (0, pad)))
        surface = jnp.pad(surface, ((0, 0), (0, pad)))
    v = jax.device_put(v, NamedSharding(mesh, P(None, None, "x", "y")))
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


@dataclass(frozen=True)
class StaticGaugeFirstOrderComponent:
    r"""Retained-manifold first jet and its authentic ``D1*D1`` response.

    ``energy_scaled_d1_raw[a,I]`` is
    ``P^(I,a)=-DeltaE*D^(I,a)`` in the canonical in-plane ``(a,I)=(2,4)``
    order.  ``S_first_first`` is only the corresponding bilinear response
    term, shaped ``(n_omega,2,2,4,4)``.  It is deliberately not a
    :class:`gw.head_correction.StaticGaugeHeadResponse`: response-weight
    second derivatives, the complete second state/vertex jet, and contact
    terms are absent.

    The band-linear fields retain the mesh-padded band extent.  They are
    derivatives with respect to Cartesian transfer q for the bra ``k-q``
    orientation; only entries below ``nb_logical`` are physical.
    ``retained_connection_cart`` is the same in-plane connection already
    formed for the first jet.  Returning that live value lets the opt-in
    second-order component reuse it without a second link-stencil pass.
    """

    energy_scaled_d1_raw: jax.Array
    S_first_first: jax.Array
    bra_energy_dq_ry: jax.Array
    occupation_difference_dq: jax.Array
    charge_ward_residual: jax.Array
    retained_connection_cart: jax.Array
    nb_logical: int


@dataclass(frozen=True)
class StaticGaugeSecondOrderComponent:
    r"""Retained-manifold insulating second-order bubble component.

    ``transition_d2_raw[pair,I]`` is the literal symmetric second transition
    jet in pair order ``(xx,xy,yy)``.  ``S_second_zero_zero_second`` is its
    second-zero plus zero-second bubble contraction.  ``S_response_weight``
    contains the terms generated by the first and second derivatives of the
    exact incumbent Adler--Wiser weight.  ``S_bubble_second_derivative`` is
    their sum with the symmetrized first-first contribution; it is the
    derivative, not the coefficient of ``q_a q_b`` (the latter is one half).

    This object is deliberately retained-manifold and bubble-only.  It is
    not a :class:`gw.head_correction.StaticGaugeHeadResponse`: complement-
    space state response, metallic occupation derivatives, contact response,
    body lineage, and persistence are absent.
    """

    first_order: StaticGaugeFirstOrderComponent
    transition_d2_raw: jax.Array
    bra_energy_dq2_ry: jax.Array
    occupation_difference_dq2: jax.Array
    S_second_zero_zero_second: jax.Array
    S_response_weight: jax.Array
    S_bubble_second_derivative: jax.Array
    ordered_curvature_residual: jax.Array
    q2_symmetry_residual: jax.Array
    nb_logical: int

    @property
    def S_bubble_q2_coefficient_cart(self) -> jax.Array:
        r"""Retained-bubble coefficient of ``q_a q_b`` in Cartesian form.

        The differentiated response is stored in symmetric pair order
        ``(xx,xy,yy)`` as ``S_bubble_second_derivative``.  Taylor's theorem
        supplies the factor ``1/2``.  This view expands the result to
        ``(n_omega,2,2,4,4)`` so that its axes match the ``S_direct``
        convention of :class:`gw.head_correction.StaticGaugeHeadResponse`::

            Pi_bubble(q) = q_a q_b S_bubble_q2_coefficient_cart[a,b]

        It remains only the retained-manifold bubble contribution.  In
        particular, exposing the correctly normalized coefficient does not
        supply the missing complement-space or contact terms and cannot be
        used by itself to publish a production FULL head.
        """
        hessian = jnp.asarray(self.S_bubble_second_derivative)
        if hessian.ndim != 4 or hessian.shape[1:] != (3, 4, 4):
            raise ValueError(
                "S_bubble_second_derivative must have shape "
                "(n_omega,3,4,4) in pair order (xx,xy,yy); got "
                f"{hessian.shape}")
        coefficient = 0.5 * hessian
        xx, xy, yy = (
            coefficient[:, 0], coefficient[:, 1], coefficient[:, 2])
        return jnp.stack(
            (jnp.stack((xx, xy), axis=1),
             jnp.stack((xy, yy), axis=1)),
            axis=1,
        )


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
    # the artifact carries.  The verdict comes off the WFN rather than the
    # artifact because time reversal is a property of the DFT solution,
    # and the fingerprint check above has already established that these
    # are the same solution.  ``None`` (no measurement on this object) is
    # reported as no-verdict, never as a pass.
    trs_measured = getattr(wfn, "trs_holds", None)
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
    ``_fractional_pair_scan`` already takes for its own ``z=0`` diagonal
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
    is ``_s_tensor_kernel``'s ONLY; the wing kernels
    (``_head_wing_kernel_legacy``/``_face``) build a structurally
    DIFFERENT ``F_ij = pref_inter * f_diff / (z**2 - dE**2)`` -- one fewer
    power of ``dE`` (documented explicitly in ``head_wings_sharded``'s own
    docstring), because a wing pairs one velocity leg with one
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
        n_padded = ((int(n_omega) + block - 1) // block) * block
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


def _static_gauge_bubble_mixed_derivative_kernel(
    mesh: Mesh, *, nb_logical: int,
) -> Callable:
    r"""Differentiate the incumbent insulating head bubble for one ``ab``.

    This is analytic AD of :func:`_s_tensor_kernel`, not a second response
    weight or collective implementation.  The two scalar parameters select
    one ordered pair of physical transfer directions.  Supplying the same
    first jet on both parameters gives a diagonal second derivative; supplying
    different jets gives a mixed derivative.  In both cases the ``s*t``
    coefficient is the physical symmetric second jet.
    """
    key = ("static_gauge_bubble_mixed_d2", id(mesh), int(nb_logical))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    bubble = _s_tensor_kernel(
        mesh, nb_logical=int(nb_logical), include_surface=False)

    def _kernel(
        m0,
        d_left,
        d_right,
        d2_pair,
        energies,
        energy_left,
        energy_right,
        energy_d2_pair,
        occupations,
        omegas,
        prefactor,
        eta,
    ):
        surface = jnp.zeros_like(energies)
        origin = jnp.zeros((2,), dtype=energies.dtype)
        tangent_s = jnp.asarray((1.0, 0.0), dtype=energies.dtype)
        tangent_t = jnp.asarray((0.0, 1.0), dtype=energies.dtype)

        def _evaluate(st, *, vertex_second_only: bool):
            s, t = st[0], st[1]
            if vertex_second_only:
                m_q = m0 + (s * t) * d2_pair
                e_bra = energies
            else:
                m_q = (
                    m0
                    + s * d_left
                    + t * d_right
                    + (s * t) * d2_pair
                )
                e_bra = (
                    energies
                    + s * energy_left
                    + t * energy_right
                    + (s * t) * energy_d2_pair
                )
            delta = e_bra[:, :, None] - energies[:, None, :]
            energy_scaled = -delta[None] * m_q
            return bubble(
                energy_scaled,
                e_bra,
                energies,
                occupations,
                occupations,
                surface,
                surface,
                omegas,
                prefactor,
                eta,
            )

        def _mixed(fun):
            def _along_s(st):
                return jax.jvp(fun, (st,), (tangent_s,))[1]

            return jax.jvp(_along_s, (origin,), (tangent_t,))[1]

        full = _mixed(lambda st: _evaluate(st, vertex_second_only=False))
        second_zero = _mixed(
            lambda st: _evaluate(st, vertex_second_only=True))
        return full, second_zero

    kernel = jax.jit(_kernel)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _head_wing_kernel(
    mesh: Mesh,
    *,
    nb_logical: int,
    include_surface: bool,
    layout: str = "legacy",
) -> Callable:
    """Layout dispatcher.  ``layout='legacy'`` (default) returns the exact
    pre-``low_mem_bands`` kernel, unmoved; ``layout='face'`` returns the
    two-face carrier's bounded band-pair-gather kernel.  Single owner, no
    ``build_G_low_mem``-style fork: this is the ONE call site every caller
    of the direct q-linear wings goes through (see :func:`head_wings_sharded`,
    the only caller)."""
    if layout == "legacy":
        return _head_wing_kernel_legacy(
            mesh, nb_logical=int(nb_logical),
            include_surface=bool(include_surface))
    if layout == "face":
        return _head_wing_kernel_face(
            mesh, nb_logical=int(nb_logical),
            include_surface=bool(include_surface))
    raise ValueError(
        f"_head_wing_kernel: layout must be 'legacy' or 'face', got "
        f"{layout!r}")


def _head_wing_kernel_legacy(
    mesh: Mesh,
    *,
    nb_logical: int,
    include_surface: bool,
) -> Callable:
    r"""Return the cached all-band q-linear wing contraction.

    The velocity is already tiled on both band axes.  A naive wing einsum
    would have to gather one complete band axis because the output centroid
    axis and one velocity-band axis share the same mesh axis.  Instead each
    rank circulates its small velocity tile around that mesh axis.  After one
    ring every local centroid slice has seen every band tile, while no rank
    ever materialises a full ``nb x nb`` velocity matrix.

    UNTOUCHED by ``low_mem_bands`` — this is the exact pre-existing body,
    unmoved; see ``_head_wing_kernel_face`` for the two-face sibling.
    """
    key = ("head_wings", id(mesh), int(nb_logical), bool(include_surface))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)
    px = int(mesh.shape[ax_x])
    py = int(mesh.shape[ax_y])
    perm_x = tuple((i, (i + 1) % px) for i in range(px))
    perm_y = tuple((i, (i + 1) % py) for i in range(py))

    def _frequency_layout(n_omega):
        block = min(_HEAD_WING_FREQUENCY_BLOCK, int(n_omega))
        padded = ((int(n_omega) + block - 1) // block) * block
        return block, padded

    def _local(
        v_local,
        psi_xn_local,
        psi_yn_local,
        energies,
        occupations,
        surface_weight,
        omegas,
        pref_inter,
        pref_surface,
        eta,
    ):
        nk = v_local.shape[1]
        nx, ny = v_local.shape[-2:]
        ns = psi_xn_local.shape[1]
        nmu_x = psi_xn_local.shape[2]
        nmu_y = psi_yn_local.shape[2]
        x_coord = jax.lax.axis_index(ax_x)
        y_coord = jax.lax.axis_index(ax_y)
        zero = jnp.asarray(0, dtype=x_coord.dtype)
        x_start = x_coord * nx
        y_start = y_coord * ny
        z = omegas + 1j * eta
        inv_z = jnp.where(
            jnp.abs(omegas) > 1.0e-15,
            1.0 / z,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        )
        frequency_block, n_omega_padded = _frequency_layout(
            omegas.shape[0])
        frequency_pad = n_omega_padded - int(omegas.shape[0])
        z_blocks = jnp.pad(
            z, (0, frequency_pad),
            constant_values=jnp.asarray(1.0j, dtype=jnp.complex128),
        ).reshape(-1, frequency_block)
        inv_z_blocks = jnp.pad(
            inv_z, (0, frequency_pad)).reshape(-1, frequency_block)
        block_indices = jnp.arange(z_blocks.shape[0], dtype=jnp.int32)

        def _accumulate_frequency_blocks(
            accumulator,
            dE,
            f_diff,
            transition,
            surface_pair,
            contract,
        ):
            def _block(block_acc, node):
                block_index, z_block, inv_z_block = node
                denom = (
                    z_block[:, None, None, None] ** 2
                    - dE[None, :, :, :] ** 2)
                weight = jnp.where(
                    transition[None, :, :, :]
                    & (jnp.abs(denom) > 1.0e-16),
                    pref_inter * f_diff[None, :, :, :] / denom,
                    jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
                )
                if include_surface:
                    weight = weight + (
                        pref_surface
                        * inv_z_block[:, None, None, None]
                        * surface_pair[None, :, :, :])
                contribution = contract(weight)
                start = block_index * frequency_block
                starts = (start,) + (zero,) * (block_acc.ndim - 1)
                sizes = (frequency_block,) + block_acc.shape[1:]
                old = jax.lax.dynamic_slice(block_acc, starts, sizes)
                return jax.lax.dynamic_update_slice(
                    block_acc, old + contribution, starts), None

            return jax.lax.scan(
                _block,
                accumulator,
                (block_indices, z_blocks, inv_z_blocks),
                unroll=1,
            )[0]

        e_low_y = jax.lax.dynamic_slice(
            energies, (zero, y_start), (nk, ny))
        f_low_y = jax.lax.dynamic_slice(
            occupations, (zero, y_start), (nk, ny))
        psi_low_x = jax.lax.dynamic_slice(
            psi_xn_local, (zero, zero, zero, y_start), (nk, ns, nmu_x, ny))

        n_vertex = int(v_local.shape[0])
        y0 = jnp.zeros(
            (n_omega_padded, n_vertex, nmu_x), dtype=jnp.complex128)

        def _left_step(step, carry):
            v_tile, acc = carry
            source_x = jnp.mod(x_coord - step, px)
            high_start = source_x * nx
            e_high = jax.lax.dynamic_slice(
                energies, (zero, high_start), (nk, nx))
            f_high = jax.lax.dynamic_slice(
                occupations, (zero, high_start), (nk, nx))
            psi_high = jax.lax.dynamic_slice(
                psi_xn_local, (zero, zero, zero, high_start),
                (nk, ns, nmu_x, nx))
            dE = e_high[:, :, None] - e_low_y[:, None, :]
            f_diff = f_low_y[:, None, :] - f_high[:, :, None]
            global_high = high_start + jnp.arange(nx)
            global_low = y_start + jnp.arange(ny)
            logical = (
                (global_high[:, None] < nb_logical)
                & (global_low[None, :] < nb_logical)
            )[None, :, :]
            transition = logical & (dE > 0.0)
            surface_pair = jnp.zeros_like(dE)
            if include_surface:
                diagonal = logical & (
                    global_high[:, None] == global_low[None, :]
                )[None, :, :]
                surface_high = jax.lax.dynamic_slice(
                    surface_weight, (zero, high_start), (nk, nx))
                surface_pair = jnp.where(
                    diagonal, surface_high[:, :, None], 0.0)

            def _contract_left(weight):
                # b_ij(mu) remains fused; no nk*nb^2*nmu tensor is stored.
                return jnp.einsum(
                    "akij,wkij,ksmi,ksmj->wam",
                    jnp.conj(v_tile), weight,
                    jnp.conj(psi_high), psi_low_x,
                    optimize=True,
                )

            acc = _accumulate_frequency_blocks(
                acc, dE, f_diff, transition, surface_pair, _contract_left)
            v_next = (
                jax.lax.ppermute(v_tile, ax_x, perm_x)
                if px > 1 else v_tile)
            return v_next, acc

        (unused_v, Y_x), _ = jax.lax.scan(
            lambda carry, step: (_left_step(step, carry), None),
            (v_local, y0), jnp.arange(px, dtype=x_coord.dtype), unroll=1)
        del unused_v
        Y_x = Y_x[:omegas.shape[0]]
        Y_x = jax.lax.psum(Y_x, ax_y)

        e_high_x = jax.lax.dynamic_slice(
            energies, (zero, x_start), (nk, nx))
        f_high_x = jax.lax.dynamic_slice(
            occupations, (zero, x_start), (nk, nx))
        psi_high_y = jax.lax.dynamic_slice(
            psi_yn_local, (zero, zero, zero, x_start), (nk, ns, nmu_y, nx))
        z0 = jnp.zeros(
            (n_omega_padded, nmu_y, n_vertex), dtype=jnp.complex128)

        def _right_step(step, carry):
            v_tile, acc = carry
            source_y = jnp.mod(y_coord - step, py)
            low_start = source_y * ny
            e_low = jax.lax.dynamic_slice(
                energies, (zero, low_start), (nk, ny))
            f_low = jax.lax.dynamic_slice(
                occupations, (zero, low_start), (nk, ny))
            psi_low = jax.lax.dynamic_slice(
                psi_yn_local, (zero, zero, zero, low_start),
                (nk, ns, nmu_y, ny))
            dE = e_high_x[:, :, None] - e_low[:, None, :]
            f_diff = f_low[:, None, :] - f_high_x[:, :, None]
            global_high = x_start + jnp.arange(nx)
            global_low = low_start + jnp.arange(ny)
            logical = (
                (global_high[:, None] < nb_logical)
                & (global_low[None, :] < nb_logical)
            )[None, :, :]
            transition = logical & (dE > 0.0)
            surface_pair = jnp.zeros_like(dE)
            if include_surface:
                diagonal = logical & (
                    global_high[:, None] == global_low[None, :]
                )[None, :, :]
                surface_low = jax.lax.dynamic_slice(
                    surface_weight, (zero, low_start), (nk, ny))
                surface_pair = jnp.where(
                    diagonal, surface_low[:, None, :], 0.0)

            def _contract_right(weight):
                return jnp.einsum(
                    "ksmi,ksmj,wkij,bkij->wmb",
                    psi_high_y, jnp.conj(psi_low), weight, v_tile,
                    optimize=True,
                )

            acc = _accumulate_frequency_blocks(
                acc, dE, f_diff, transition, surface_pair, _contract_right)
            v_next = (
                jax.lax.ppermute(v_tile, ax_y, perm_y)
                if py > 1 else v_tile)
            return v_next, acc

        (unused_v, Z_y), _ = jax.lax.scan(
            lambda carry, step: (_right_step(step, carry), None),
            (v_local, z0), jnp.arange(py, dtype=y_coord.dtype), unroll=1)
        del unused_v
        Z_y = Z_y[:omegas.shape[0]]
        Z_y = jax.lax.psum(Z_y, ax_x)
        return Y_x, Z_y

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),
            P(None, None, "x", None),
            P(None, None, "y", None),
            P(None, None),
            P(None, None),
            P(None, None),
            P(None),
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


def _head_wing_kernel_face(
    mesh: Mesh,
    *,
    nb_logical: int,
    include_surface: bool,
) -> Callable:
    r"""Two-face-carrier q-linear wing contraction (audit report §"Full
    q->0 head/body wings", census rows 6/7).

    WHY THE LEGACY RING DOES NOT PORT, AND THE 10x ALGEBRA
    --------------------------------------------------------
    ``_head_wing_kernel_legacy`` is cheap only because ``psi_xn``/``psi_yn``
    carry a REPLICATED band axis: any rank can read an arbitrary band
    window ``psi_xn_local[..., lo:hi]`` for free, so only the small
    velocity tile (``3 x nk x nx x ny``, independent of mu) needs
    circulating around the ring.  ``psi_mun``/``psi_nmu`` have NO such
    replicated axis — every axis is mesh-sharded — so the identical trick
    is unavailable, exactly the report's own words: "a face layout needs a
    2-D band-pair ring/tile algorithm... cannot be replaced by the
    one-particle G GEMM" (dense ``Gij``/exact-response census row).

    A DIFFERENT confirmed 81.74-GiB defect (registered separately,
    `KNOWN_LORRAX_ISSUES.md`, "the v-sharding fix is not closed") lives on
    the LEGACY-only path and is unrelated to what follows: this module's
    own algebra there was ``10.006x`` ONE FULL ``(nk,ns,mu,nb)`` psi copy,
    the size GSPMD's auto-reshard prologue pays when it reconciles ONE
    off-mesh (``SingleDeviceSharding``) operand — the freshly host-read
    velocity — against SEVERAL already on-mesh psi-sized operands inside a
    single compiled program.  That mechanism needs a legacy-shaped psi
    input (a full band-replicated copy) to reproduce, so it CANNOT recur
    here even in principle: this kernel never holds anything psi-shaped
    that is band-replicated, and every operand it builds is explicitly
    ``jax.device_put`` onto its declared ``NamedSharding`` before use (see
    ``_pad_head_band_manifold_to`` below) — the exact discipline whose
    absence caused that legacy defect.  The residency bound below is
    therefore independent of, not a fix for, that open legacy row.

    THE BOUNDED ALGORITHM
    ----------------------
    Split the work along the axis each face orientation already carries
    for free:

    * mu (the OUTPUT index) never moves.  ``psi_mun``'s mu axis is
      X-sharded — the SAME axis ``Y_x``'s output is sharded on — so a
      rank's own local mu range is already the right output range, no
      communication.  Symmetrically for ``psi_nmu`` (mu on Y) and ``Z_y``.
    * n (the two SUMMED band indices ``i``, ``j`` of ``b_ij(mu) = sum_s
      conj(psi_i(mu)) psi_j(mu)``) is what is missing locally: ``psi_mun``
      only holds a Y-fraction of it.  For a SMALL block of ``_HEAD_WING_
      MU_BLOCK`` mu values at a time, one ``lax.all_gather`` over the
      OTHER mesh axis assembles the full ``nb_full``-wide band vector for
      exactly that block — bounded by ``nk * ns * nb_full * mu_block``,
      independent of how many mu values this rank owns in total.  ``i``
      and ``j`` then read the SAME gathered array (two labels on one
      operand, matching ``b_ij``'s own definition), so no second ring or
      gather is needed for the "other" band index the way a naive
      transcription of the legacy ring might suggest.
    * The (i,j)-operator itself — ``conj(v[a,i,j]) * F_ij(omega)`` — is
      ``nb_full x nb_full``, INDEPENDENT of mu.  It is built once (one
      two-axis ``lax.all_gather`` of the small, already band-mesh-sharded
      velocity) and reused across every mu block, at the SAME
      ``_HEAD_WING_FREQUENCY_BLOCK``-bounded omega streaming the legacy
      kernel already uses (reused unchanged, not re-derived).

    COST, NAMED (matching ``Wavefunctions.band_mask``'s own convention):
    every mu block pays the FULL ``nb_full x nb_full`` (i,j) contraction —
    the same "good correctness bring-up path" obstacle #3 already
    sanctions for masked face operations — so this kernel does
    ``mu_local / _HEAD_WING_MU_BLOCK`` TIMES the (i,j) FLOPs a
    theoretically optimal ring would, in exchange for a residency bound
    that never scales with mu.  Also unlike legacy, ranks sharing an
    output mu range but differing on the OTHER mesh axis (e.g. same X,
    different Y for ``Y_x``) redundantly compute the IDENTICAL answer —
    correct (every rank's own local mu range is complete on its own, no
    ``psum`` needed) but ``P``-axis-fold redundant in FLOPs, a stated
    trade for zero extra communication.

    Peak transient per rank, this kernel only (steady-state persistent
    storage is unaffected — it is still ``psi_mun``/``psi_nmu`` at
    ``2*S/(Px*Py)``, the carrier's own number):
    ``v_full`` (``3*nk*nb_full^2``, gathered once) +
    ``weight`` (``omega_block*nk*nb_full^2``, rebuilt once per mu block) +
    the gathered psi block (``nk*ns*nb_full*mu_block``, small).  None of
    these three terms contains ``mu_local`` or ``mu_pad``.
    """
    key = ("head_wings_face", id(mesh), int(nb_logical), bool(include_surface))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(
        v_local,
        psi_mun_local,
        psi_nmu_local,
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
        ns = psi_mun_local.shape[1]
        mu_x_local = psi_mun_local.shape[2]
        mu_y_local = psi_nmu_local.shape[-1]

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
        n_omega_padded = ((int(n_omega) + freq_block - 1) // freq_block) * freq_block
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
                denom = z_block[:, None, None, None] ** 2 - dE[None] ** 2
                weight = jnp.where(
                    transition[None] & (jnp.abs(denom) > 1.0e-16),
                    pref_inter * f_diff[None] / denom,
                    jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
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
        n_x_blocks = -(-int(mu_x_local) // mu_x_block)
        mu_x_padded = n_x_blocks * mu_x_block
        psi_mun_padded = jnp.pad(
            psi_mun_local,
            ((0, 0), (0, 0), (0, mu_x_padded - mu_x_local), (0, 0)))

        def _x_step(_carry, blk):
            zero = jnp.zeros((), dtype=blk.dtype)
            start = blk * mu_x_block
            tile = jax.lax.dynamic_slice(
                psi_mun_padded, (zero, zero, start, zero),
                (nk, ns, mu_x_block, psi_mun_padded.shape[-1]))
            psi_full = jax.lax.all_gather(tile, ax_y, axis=3, tiled=True)

            def _contract_left(weight):
                return jnp.einsum(
                    "akij,wkij,ksmi,ksmj->wam",
                    jnp.conj(v_full), weight,
                    jnp.conj(psi_full), psi_full,
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
        n_y_blocks = -(-int(mu_y_local) // mu_y_block)
        mu_y_padded = n_y_blocks * mu_y_block
        psi_nmu_padded = jnp.pad(
            psi_nmu_local,
            ((0, 0), (0, 0), (0, 0), (0, mu_y_padded - mu_y_local)))

        def _y_step(_carry, blk):
            zero = jnp.zeros((), dtype=blk.dtype)
            start = blk * mu_y_block
            tile = jax.lax.dynamic_slice(
                psi_nmu_padded, (zero, zero, zero, start),
                (nk, psi_nmu_padded.shape[1], ns, mu_y_block))
            gathered = jax.lax.all_gather(tile, ax_x, axis=1, tiled=True)
            psi_full = jnp.transpose(gathered, (0, 2, 3, 1))

            def _contract_right(weight):
                return jnp.einsum(
                    "ksmi,ksmj,wkij,bkij->wmb",
                    psi_full, jnp.conj(psi_full), weight, v_full,
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
            P(None, None, "x", "y"),   # psi_mun_local  (PSI_MUN_SPEC)
            P(None, "x", None, "y"),   # psi_nmu_local  (PSI_NMU_SPEC)
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
    v-sharding-commit defect on the legacy path): every returned array is
    explicitly ``jax.device_put`` onto its declared mesh sharding before
    any kernel sees it, so a foreign ``SingleDeviceSharding`` operand
    never reaches this kernel's ``shard_map``.
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
    v = jax.device_put(v, NamedSharding(mesh, P(None, None, "x", "y")))
    e = jax.device_put(e, NamedSharding(mesh, P(None, None)))
    f = jax.device_put(f, NamedSharding(mesh, P(None, None)))
    surface = jax.device_put(surface, NamedSharding(mesh, P(None, None)))
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
):
    r"""Build q-linear head/body wings in the current band basis.

    For every energy-ordered interband pair, with
    ``b_ij(mu)=sum_s psi_i(mu)^* psi_j(mu)``, this evaluates

    ``Y[a,mu] = sum conj(v[a,ij]) F_ij b_ij(mu)`` and
    ``Z[mu,b] = sum conj(b_ij(mu)) F_ij v[b,ij]``,

    where ``F_ij = 4(f_j-f_i)/(Nk*nspin*nspinor*(z^2-dE^2))``.  This is
    exactly the normalization paired with ``head_s_tensor_sharded``:
    ``S_direct`` owns ``1/cell_volume`` and the later Schur fold introduces
    the sole additional ``1/cell_volume`` multiplying ``Y W Z``.

    With tetrahedron surface weights, the finite-frequency intraband wings
    ``sum delta(E-mu) v_a b_nn / z`` are included as well.  The strictly
    static metal limit is not obtained by setting ``z=0`` in that expression;
    its head remains on the separate Thomas-Fermi path.

    Every rank owns the same ``(nb/Px) * (nb/Py)`` band-pair tile.  The
    x-sharded and y-sharded centroid wavefunction copies build Y and Z,
    respectively, while band tiles circulate around only the matching mesh
    axis.  Frequencies are blocked inside each ring step, so a tile is sent
    once rather than once per frequency block and no all-frequency pair
    tensor is formed.

    Layout dispatch on ``wfns.layout`` (report §5, single owner): the body
    below, unchanged, is ``layout='legacy'``.  ``layout='face'`` routes to
    :func:`_head_wings_sharded_face`, which returns the SAME
    ``(Y_x, Z_y)`` shapes/sharding/physics — every caller downstream of
    this function is layout-agnostic.  A ``wfns`` with no ``layout``
    attribute at all (pre-two-face-carrier test fixtures that stand in a
    bare ``psi_xn``/``psi_yn``-carrying stub) defaults to ``'legacy'``,
    matching ``Wavefunctions.layout``'s own dataclass default.

    Exactly two operator widths are admitted: the incumbent three Cartesian
    velocity rows and the canonical eight-row packed ``(a,I)`` energy-scaled
    jet ``P^(I,a)=-DeltaE*d_q_a M^I|_0`` (two in-plane q directions by four
    Lorentz fields).  Literal transition derivatives are not accepted by
    this denominator convention.  The width-eight result is the associated
    first-derivative one-leg contribution, not a claim that the complete
    CT/TT response needs no response-weight, second-jet, or contact terms.
    Keeping that boundary closed gives the shared kernel exactly two XLA
    shapes, rather than one compile for every caller-chosen width.
    """
    layout = getattr(wfns, "layout", "legacy")
    if layout == "face":
        return _head_wings_sharded_face(
            velocity_cart, wfns, energies_kn_ry, occupations_kn, omegas_ry,
            mesh=mesh, nb_logical=nb_logical, nk_tot=nk_tot, nspin=nspin,
            nspinor=nspinor, eta_ry=eta_ry,
            surface_weight_kn=surface_weight_kn)
    if layout != "legacy":
        raise ValueError(
            f"head_wings_sharded: wfns.layout must be 'legacy' or 'face', "
            f"got {layout!r}")
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
    if (
        int(wfns.psi_xn.shape[0]) != int(v.shape[1])
        or int(wfns.psi_yn.shape[0]) != int(v.shape[1])
        or int(wfns.psi_xn.shape[1]) != int(wfns.psi_yn.shape[1])
    ):
        raise ValueError(
            "centroid wavefunction k/spinor axes do not match the velocity")
    if (
        int(wfns.psi_xn.shape[-1]) < int(v.shape[-1])
        or int(wfns.psi_yn.shape[-1]) < int(v.shape[-1])
    ):
        raise ValueError("centroid wavefunctions do not cover the head manifold")
    include_surface = surface_weight_kn is not None
    surface = (
        jnp.asarray(surface_weight_kn, dtype=jnp.float64)
        if include_surface else jnp.zeros_like(e))
    if surface.shape != e.shape:
        raise ValueError(
            f"surface_weight_kn shape {surface.shape} does not match {e.shape}.")
    v, e, f, surface = _pad_head_band_manifold(
        v, e, f, surface, mesh=mesh)
    psi_xn = jnp.asarray(wfns.psi_xn)[..., : int(v.shape[-1])]
    psi_yn = jnp.asarray(wfns.psi_yn)[..., : int(v.shape[-1])]
    psi_pad = int(v.shape[-1]) - int(psi_xn.shape[-1])
    if psi_pad:
        psi_xn = jnp.pad(psi_xn, ((0, 0), (0, 0), (0, 0), (0, psi_pad)))
        psi_yn = jnp.pad(psi_yn, ((0, 0), (0, 0), (0, 0), (0, psi_pad)))
    spin_denominator = (
        float(max(int(nspin), 1)) * float(max(int(nspinor), 1)))
    pref_inter = 4.0 / (float(nk_tot) * spin_denominator)
    pref_surface = 2.0 / (float(nk_tot) * spin_denominator)
    return _head_wing_kernel(
        mesh, nb_logical=int(nb_logical),
        include_surface=bool(include_surface))(
            v, psi_xn, psi_yn, e, f, surface, omega,
            jnp.asarray(pref_inter, dtype=jnp.complex128),
            jnp.asarray(pref_surface, dtype=jnp.complex128),
            jnp.asarray(float(eta_ry), dtype=jnp.float64),
        )


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
):
    """``layout='face'`` body of :func:`head_wings_sharded`.  Same physics
    and normalization as the legacy body's docstring; only the carrier and
    kernel differ.  ``psi_mun``/``psi_nmu`` span the bundle's full stored
    ``[b0,b4)`` band extent (obstacle #3), so — unlike the legacy body,
    which truncates ``wfns.psi_xn``/``psi_yn`` down to the velocity's own
    ``nb_pad`` — this pads ``v``/``e``/``f``/``surface`` UP to that full
    extent instead (:func:`_pad_head_band_manifold_to`); the kernel masks
    anything beyond ``nb_logical`` to zero.
    """
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
    if wfns.psi_mun is None or wfns.psi_nmu is None:
        raise ValueError(
            "head_wings_sharded(layout='face') requires wfns.psi_mun and "
            "wfns.psi_nmu (got None) — this bundle is not a face carrier "
            "despite wfns.layout=='face'.")
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
        if include_surface else jnp.zeros_like(e))
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
            v, wfns.psi_mun, wfns.psi_nmu, e, f, surface, omega,
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
    r"""Build the strictly-static intraband centroid wings.

    In the static order of limits the diagonal density vertex survives:

    ``C_mu = (2/(Nk*nspin*nspinor)) sum_kn f'(E_kn)|psi_kn(mu)|^2``.

    ``surface_weight_kn`` is ``-f'`` in the caller's integration scheme, so
    the explicit minus sign below is physical.  The x/y centroid copies stay
    sharded on their respective mesh axes and the spinor axis is summed
    without any component-count special case.

    Layout dispatch on ``wfns.layout`` (report §5): the body below,
    unchanged, is ``layout='legacy'``.  ``layout='face'`` routes to
    :func:`_static_head_wings_sharded_face` — per the audit report, this
    is the EASY wing: a local density-weighted band sum plus one
    ``psum``, no ring/gather needed at all (unlike the dynamic wings),
    because the static vertex is DIAGONAL in mu — no cross-mu operator to
    sweep.  A ``wfns`` with no ``layout`` attribute defaults to
    ``'legacy'`` (see :func:`head_wings_sharded`'s matching note).
    """
    layout = getattr(wfns, "layout", "legacy")
    if layout == "face":
        return _static_head_wings_sharded_face(
            wfns, surface_weight_kn, mesh=mesh, nb_logical=nb_logical,
            nk_tot=nk_tot, nspin=nspin, nspinor=nspinor)
    if layout != "legacy":
        raise ValueError(
            f"static_head_wings_sharded: wfns.layout must be 'legacy' or "
            f"'face', got {layout!r}")
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if surface.ndim != 2:
        raise ValueError(
            f"static head surface weights must be (nk,nb), got {surface.shape}")
    if not (0 < int(nb_logical) <= int(surface.shape[1])):
        raise ValueError(
            f"need 0 < nb_logical <= {surface.shape[1]}, got {nb_logical}")
    if (
        int(wfns.psi_xn.shape[0]) != int(surface.shape[0])
        or int(wfns.psi_yn.shape[0]) != int(surface.shape[0])
        or int(wfns.psi_xn.shape[-1]) < int(surface.shape[1])
        or int(wfns.psi_yn.shape[-1]) < int(surface.shape[1])
    ):
        raise ValueError("centroid wavefunctions do not cover static weights")
    logical = jnp.arange(surface.shape[1])[None, :] < int(nb_logical)
    weight = jnp.where(logical, surface, 0.0)
    psi_x = wfns.psi_xn[..., : int(surface.shape[1])]
    psi_y = wfns.psi_yn[..., : int(surface.shape[1])]
    density_x = jnp.sum(jnp.square(jnp.abs(psi_x)), axis=1)
    density_y = jnp.sum(jnp.square(jnp.abs(psi_y)), axis=1)
    prefactor = -2.0 / (
        float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    with mesh:
        left = jax.lax.with_sharding_constraint(
            prefactor * jnp.einsum("kn,kmn->m", weight, density_x),
            NamedSharding(mesh, P("x")),
        )
        right = jax.lax.with_sharding_constraint(
            prefactor * jnp.einsum("kn,kmn->m", weight, density_y),
            NamedSharding(mesh, P("y")),
        )
    return left, right


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
    weight = jax.device_put(
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


def static_gauge_first_order_component_sharded(
    gamma_raw,
    dgamma_dq_raw,
    forward_links,
    forward_neighbors,
    energies_kn_ry,
    occupations_kn,
    omegas_ry,
    *,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    eta_ry: float = 0.0,
    surface_weight_kn=None,
) -> StaticGaugeFirstOrderComponent:
    r"""Build the insulating retained-manifold ``D1*D1`` gauge component.

    For band-matrix order ``(bra=m, ket=n)``, Berry connection
    ``A_a=i<u_m|d_a u_n>`` and the repository's bra ``k-q`` orientation,

    ``D_a^0=-i A_a`` and
    ``D_a^i=-i (A_a Gamma_i) + Q_{i,a}``.

    The shared width-eight head kernel consumes
    ``P_a^I=-Delta_mn D_a^I``, not ``D``.  Its charge column is populated by
    the exact Ward reduction ``P_a^0=v_a=Gamma_a/(alpha_FS/2)``; the
    finite-link value ``-Delta D_a^0`` is retained only in the reported
    closure residual.  This makes the charge-charge block reduce to the
    incumbent dipole S tensor without inheriting finite-link truncation.

    This first bounded lane is insulating.  Its occupation-difference jet is
    exactly zero.  A metallic ``-df/dE`` table is refused because the current
    production table may be an integrated tetrahedron weight, not a local
    chain-rule derivative.  The explicit q2 vertex is intentionally absent:
    using it before the second state jet exists would mislabel a partial
    second derivative as complete.
    """
    if surface_weight_kn is not None:
        raise ValueError(
            "static gauge first-order component is insulating-only; "
            "metallic occupation derivatives are not yet derived")

    from common.bispinor_init import HALFALPHA
    from common.parallel_transport import (
        band_storage_extent,
        fourth_order_connection,
        make_distributed_band_matmul,
        undersampled_link_axes,
    )
    from runtime.padding import pad_axis

    gamma = jnp.asarray(gamma_raw, dtype=jnp.complex128)
    q1 = jnp.asarray(dgamma_dq_raw, dtype=jnp.complex128)
    links = jnp.asarray(forward_links, dtype=jnp.complex128)
    energies = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    occupations = jnp.asarray(occupations_kn, dtype=jnp.float64)
    grid = tuple(int(n) for n in kgrid)
    logical = int(nb_logical)
    storage = band_storage_extent(mesh, logical)
    nk = int(nk_tot)

    if gamma.shape != (nk, 3, storage, storage):
        raise ValueError(
            "gamma_raw must be the mesh-padded uniform-gauge result with "
            f"shape {(nk, 3, storage, storage)}; got {gamma.shape}")
    if q1.shape != (nk, 3, 3, storage, storage):
        raise ValueError(
            "dgamma_dq_raw must be (nk,current,transfer,band,band) with "
            f"shape {(nk, 3, 3, storage, storage)}; got {q1.shape}")
    if links.shape != (3, nk, storage, storage):
        raise ValueError(
            "forward_links must share the uniform-gauge band manifold; "
            f"expected {(3, nk, storage, storage)}, got {links.shape}")
    if np.shape(forward_neighbors) != (nk, 3):
        raise ValueError(
            f"forward_neighbors must be {(nk, 3)}; got "
            f"{np.shape(forward_neighbors)}")
    if int(np.prod(grid)) != nk:
        raise ValueError(f"kgrid product {int(np.prod(grid))} != nk_tot {nk}")
    short_in_plane = tuple(
        axis for axis in undersampled_link_axes(grid) if axis in ("x", "y"))
    if short_in_plane:
        raise ValueError(
            "static gauge in-plane first jet requires the canonical fourth-"
            f"order link stencil; undersampled axes={short_in_plane}")
    if energies.ndim != 2 or occupations.shape != energies.shape:
        raise ValueError(
            "energies and occupations must be paired (nk,nb) tables; got "
            f"{energies.shape} and {occupations.shape}")
    if int(energies.shape[0]) != nk or int(energies.shape[1]) not in (
            logical, storage):
        raise ValueError(
            "energy/occupation tables must carry the logical or padded head "
            f"manifold ({nk},{logical}|{storage}); got {energies.shape}")
    occ_host = np.asarray(occupations[:, :logical])
    if not np.all((occ_host == 0.0) | (occ_host == 1.0)):
        raise ValueError(
            "static gauge first-order component requires exact insulating "
            "0/1 occupations; metallic/fractional occupation derivatives "
            "are not yet derived")

    if int(energies.shape[1]) == logical:
        energies = pad_axis(energies, storage, axis=1).array
        occupations = pad_axis(occupations, storage, axis=1).array
    if energies.shape != (nk, storage) or occupations.shape != (nk, storage):
        raise AssertionError("canonical band padding did not reach storage")

    spacing = 1.0 / np.asarray(grid, dtype=np.float64)
    connection_reduced = fourth_order_connection(
        links,
        np.asarray(forward_neighbors, dtype=np.int64),
        spacing,
        band_matmul=make_distributed_band_matmul(mesh, n_batch_axes=1),
    )
    connection_cart = reduced_covector_to_cartesian(
        connection_reduced, bvec_cart)[:2]

    # One batched distributed product for all (a,i), not six new kernels.
    gamma_dir = jnp.moveaxis(gamma, 1, 0)
    left = jnp.broadcast_to(
        connection_cart[:, None], (2, 3, nk, storage, storage),
    ).reshape(6, nk, storage, storage)
    right = jnp.broadcast_to(
        gamma_dir[None], (2, 3, nk, storage, storage),
    ).reshape(6, nk, storage, storage)
    a_gamma = make_distributed_band_matmul(mesh, n_batch_axes=2)(
        left, right).reshape(2, 3, nk, storage, storage)
    q_current = jnp.transpose(q1, (2, 1, 0, 3, 4))[:2]
    d_current = -1.0j * a_gamma + q_current

    delta = energies[:, :, None] - energies[:, None, :]
    velocity = gamma_dir / jnp.asarray(HALFALPHA, dtype=jnp.float64)
    p_charge = velocity[:2]
    p_current = -delta[None, None] * d_current
    p = jnp.concatenate((p_charge[:, None], p_current), axis=1)

    d_charge_state = -1.0j * connection_cart
    p_charge_state = -delta[None] * d_charge_state
    f_diff = occupations[:, None, :] - occupations[:, :, None]
    band = jnp.arange(storage)
    transition = (
        (delta > 0.0)
        & (jnp.abs(f_diff) > 1.0e-12)
        & (band[:, None] < logical)
        & (band[None, :] < logical)
    )
    charge_error = jnp.max(jnp.where(
        transition[None], jnp.abs(p_charge_state - p_charge), 0.0))
    charge_scale = jnp.max(jnp.where(
        transition[None], jnp.abs(p_charge), 0.0))
    charge_ward_residual = jnp.where(
        charge_scale > 0.0, charge_error / charge_scale, 0.0)

    p_flat = p.reshape(8, nk, storage, storage)
    s_flat = head_s_tensor_sharded(
        p_flat,
        energies,
        occupations,
        omegas_ry,
        mesh=mesh,
        nb_logical=logical,
        cell_volume=float(cell_volume),
        nk_tot=nk,
        nspin=int(nspin),
        nspinor=int(nspinor),
        eta_ry=float(eta_ry),
    )
    s_first_first = jnp.transpose(
        s_flat.reshape(int(s_flat.shape[0]), 2, 4, 2, 4),
        (0, 1, 3, 2, 4),
    )
    bra_energy_dq = -jnp.real(jnp.diagonal(
        velocity[:2], axis1=-2, axis2=-1))
    return StaticGaugeFirstOrderComponent(
        energy_scaled_d1_raw=p,
        S_first_first=s_first_first,
        bra_energy_dq_ry=bra_energy_dq,
        occupation_difference_dq=jnp.zeros_like(bra_energy_dq),
        charge_ward_residual=charge_ward_residual,
        retained_connection_cart=connection_cart,
        nb_logical=logical,
    )


def static_gauge_second_order_component_sharded(
    gamma_raw,
    dgamma_dq_raw,
    d2gamma_dq2_raw,
    forward_links,
    forward_neighbors,
    energies_kn_ry,
    occupations_kn,
    omegas_ry,
    *,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    eta_ry: float = 0.0,
    surface_weight_kn=None,
) -> StaticGaugeSecondOrderComponent:
    r"""Build the insulating retained-manifold bubble through second q order.

    The bra is at ``k-q``.  With ``A_a=i<u|d_a u>`` and the incumbent
    transported derivative ``C_b(O)=d_b O-i[A_b,O]``, the symmetric retained
    second bra jet is

    ``T_ab = sym_ab(i C_b(A_a) - A_b A_a)``
    ``     = sym_ab(i d_b A_a - A_a A_b)``.

    The current transition jet is therefore

    ``E_i,ab = T_ab Gamma_i - i(A_a Q_i,b + A_b Q_i,a) + Q2_i,ab``.

    ``Q2`` is the true second derivative used with ``q_a q_b/2`` by the
    canonical ICL sweep.  Its transfer indices must already be symmetric at
    roundoff; this consumer measures and refuses asymmetry rather than
    repairing it.

    The response is analytic AD of the exact incumbent head-weight kernel.
    This preserves its transition mask, prefactor, frequency blocking,
    two-axis band tiling, and sole psum implementation.  Only ``(xx,xy,yy)``
    is streamed.  Exact charge ``P=v`` from the first-order Ward reduction is
    retained in the first jet; the finite-link ``T_ab`` remains the explicit
    retained-manifold second charge jet.

    This API is insulating and bubble-only.  It neither consumes metallic
    surface weights nor supplies complement-space, contact, body, or FULL
    screened completion.
    """
    if surface_weight_kn is not None:
        raise ValueError(
            "static gauge second-order component is insulating-only; "
            "metallic f'' occupation response is not derived")

    from common.bispinor_init import HALFALPHA
    from common.parallel_transport import (
        fourth_order_covariant_derivative,
        make_distributed_band_matmul,
    )
    from runtime.padding import pad_axis

    gamma = jnp.asarray(gamma_raw, dtype=jnp.complex128)
    q1 = jnp.asarray(dgamma_dq_raw, dtype=jnp.complex128)
    q2 = jnp.asarray(d2gamma_dq2_raw, dtype=jnp.complex128)
    links = jnp.asarray(forward_links, dtype=jnp.complex128)
    energies = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    occupations = jnp.asarray(occupations_kn, dtype=jnp.float64)
    grid = tuple(int(n) for n in kgrid)
    nk = int(nk_tot)
    logical = int(nb_logical)
    if gamma.ndim != 4 or gamma.shape[-2] != gamma.shape[-1]:
        raise ValueError(
            "gamma_raw must be (nk,current,band,band) before Q2 validation; "
            f"got {gamma.shape}")
    storage = int(gamma.shape[-1])
    expected_q2 = (nk, 3, 3, 3, storage, storage)
    if q2.shape != expected_q2:
        raise ValueError(
            "d2gamma_dq2_raw must be "
            "(nk,current,transfer,transfer,band,band) with shape "
            f"{expected_q2}; got {q2.shape}")

    q2_cart = jnp.transpose(q2, (2, 3, 1, 0, 4, 5))[:2, :2]
    q2_asymmetry = jnp.max(jnp.abs(q2_cart - jnp.swapaxes(q2_cart, 0, 1)))
    q2_scale = jnp.max(jnp.abs(q2_cart))
    q2_symmetry_residual = jnp.where(
        q2_scale > 0.0, q2_asymmetry / q2_scale, 0.0)
    q2_error_host, q2_scale_host = (
        float(x) for x in jax.device_get((q2_asymmetry, q2_scale)))
    q2_tolerance = 512.0 * np.finfo(np.float64).eps * max(
        1.0, q2_scale_host)
    if q2_error_host > q2_tolerance:
        raise ValueError(
            "static gauge Q2 transfer indices are not symmetric: "
            f"max|Q2_ab-Q2_ba|={q2_error_host:.6e}, "
            f"roundoff allowance={q2_tolerance:.6e}; regenerate the "
            "canonical ICL transfer jet rather than averaging this defect")

    first = static_gauge_first_order_component_sharded(
        gamma,
        q1,
        links,
        forward_neighbors,
        energies,
        occupations,
        omegas_ry,
        mesh=mesh,
        kgrid=kgrid,
        bvec_cart=bvec_cart,
        nb_logical=logical,
        cell_volume=float(cell_volume),
        nk_tot=nk,
        nspin=int(nspin),
        nspinor=int(nspinor),
        eta_ry=float(eta_ry),
    )

    if int(energies.shape[1]) == logical:
        energies = pad_axis(energies, storage, axis=1).array
        occupations = pad_axis(occupations, storage, axis=1).array
    if energies.shape != (nk, storage) or occupations.shape != (nk, storage):
        raise ValueError(
            "second-order energy/occupation tables must reach the first-"
            f"order storage shape {(nk, storage)}; got "
            f"{energies.shape}/{occupations.shape}")

    spacing = 1.0 / np.asarray(grid, dtype=np.float64)
    plus = np.asarray(forward_neighbors, dtype=np.int64)
    multiply_one = make_distributed_band_matmul(mesh, n_batch_axes=1)
    multiply_six = make_distributed_band_matmul(mesh, n_batch_axes=2)

    def _multiply_in_six(left, right):
        """Run one bounded block on the first-jet's six-wide executable."""
        count = int(left.shape[0])
        if right.shape != left.shape:
            raise ValueError(
                f"paired component products differ: {left.shape}/{right.shape}")
        if not 0 < count <= 6:
            raise ValueError(
                "second-order band products must be streamed in one "
                f"six-component block; got {count}")
        pad = 6 - count
        if pad:
            zeros = jnp.zeros(
                (pad, nk, storage, storage), dtype=left.dtype)
            left = jnp.concatenate((left, zeros), axis=0)
            right = jnp.concatenate((right, zeros), axis=0)
        return multiply_six(left, right)[:count]

    def _cartesian_covariant_derivatives(operators):
        rows = []
        for operator in operators:
            reduced = fourth_order_covariant_derivative(
                operator,
                links,
                plus,
                spacing,
                band_matmul=multiply_one,
            )
            rows.append(reduced_covector_to_cartesian(
                reduced, bvec_cart)[:2])
        return jnp.stack(rows, axis=0)

    connection = first.retained_connection_cart
    connection_derivative = _cartesian_covariant_derivatives(connection)
    left_b = jnp.broadcast_to(
        connection[None], (2, 2, nk, storage, storage))
    right_a = jnp.broadcast_to(
        connection[:, None], (2, 2, nk, storage, storage))
    connection_product = _multiply_in_six(
        left_b.reshape(4, nk, storage, storage),
        right_a.reshape(4, nk, storage, storage),
    ).reshape(2, 2, nk, storage, storage)
    ordered_t = 1.0j * connection_derivative - connection_product
    retained_t = 0.5 * (ordered_t + jnp.swapaxes(ordered_t, 0, 1))
    curvature_error = jnp.max(jnp.abs(
        ordered_t - jnp.swapaxes(ordered_t, 0, 1)))
    curvature_scale = jnp.max(jnp.abs(ordered_t))
    ordered_curvature_residual = jnp.where(
        curvature_scale > 0.0, curvature_error / curvature_scale, 0.0)

    gamma_dir = jnp.moveaxis(gamma, 1, 0)
    q_current = jnp.transpose(q1, (2, 1, 0, 3, 4))[:2]
    pair_axes = ((0, 0), (0, 1), (1, 1))
    t_pair = jnp.stack(tuple(retained_t[a, b] for a, b in pair_axes))

    # One existing distributed-matmul family owns every T*Gamma and A*Q
    # product.  Stream one retained pair at a time: the first block holds
    # [T_ab*Gamma_i, A_a*Q_i,b], the diagonal reuses its A*Q half, and only
    # xy needs a second (three-valid, six-wide padded) A_b*Q_i,a block.
    # No 21-component band-square input or product carrier is materialized.
    current_d2_rows = []
    for p, (a, b) in enumerate(pair_axes):
        primary_left = jnp.concatenate((
            jnp.broadcast_to(
                t_pair[p][None], (3, nk, storage, storage)),
            jnp.broadcast_to(
                connection[a][None], (3, nk, storage, storage)),
        ), axis=0)
        primary_right = jnp.concatenate((gamma_dir, q_current[b]), axis=0)
        primary = _multiply_in_six(primary_left, primary_right)
        current = primary[:3] - 1.0j * primary[3:]
        if a == b:
            current = current - 1.0j * primary[3:]
        else:
            secondary_left = jnp.broadcast_to(
                connection[b][None], (3, nk, storage, storage))
            secondary = _multiply_in_six(
                secondary_left, q_current[a])
            current = current - 1.0j * secondary
        current_d2_rows.append(current + q2_cart[a, b])
    current_d2 = jnp.stack(current_d2_rows)
    transition_d2 = jnp.concatenate((t_pair[:, None], current_d2), axis=1)

    velocity = gamma_dir / jnp.asarray(HALFALPHA, dtype=jnp.float64)
    velocity_derivative = _cartesian_covariant_derivatives(velocity[:2])
    velocity_hessian = 0.5 * (
        velocity_derivative + jnp.swapaxes(velocity_derivative, 0, 1))
    energy_d2 = jnp.stack(tuple(jnp.real(jnp.diagonal(
        velocity_hessian[a, b], axis1=-2, axis2=-1))
        for a, b in pair_axes))

    delta = energies[:, :, None] - energies[:, None, :]
    nonzero_delta = delta != 0.0
    transition_d1 = jnp.where(
        nonzero_delta[None, None],
        -first.energy_scaled_d1_raw / jnp.where(
            nonzero_delta, delta, jnp.asarray(1.0, dtype=delta.dtype)
        )[None, None],
        jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
    )
    identity = jnp.broadcast_to(
        jnp.eye(storage, dtype=jnp.complex128)[None],
        (nk, storage, storage),
    )
    m0 = jnp.concatenate((identity[None], gamma_dir), axis=0)

    prefactor = jnp.asarray(
        4.0
        / (
            float(cell_volume)
            * float(nk)
            * float(max(int(nspin), 1))
            * float(max(int(nspinor), 1))
        ),
        dtype=jnp.complex128,
    )
    omega = jnp.atleast_1d(jnp.asarray(omegas_ry, dtype=jnp.complex128))
    eta = jnp.asarray(float(eta_ry), dtype=jnp.float64)
    mixed_kernel = _static_gauge_bubble_mixed_derivative_kernel(
        mesh, nb_logical=logical)
    full_pairs = []
    second_zero_pairs = []
    for p, (a, b) in enumerate(pair_axes):
        full, second_zero = mixed_kernel(
            m0,
            transition_d1[a],
            transition_d1[b],
            transition_d2[p],
            energies,
            first.bra_energy_dq_ry[a],
            first.bra_energy_dq_ry[b],
            energy_d2[p],
            occupations,
            omega,
            prefactor,
            eta,
        )
        full_pairs.append(full)
        second_zero_pairs.append(second_zero)
    full_second = jnp.stack(full_pairs, axis=1)
    second_zero = jnp.stack(second_zero_pairs, axis=1)
    first_first = jnp.stack(tuple(
        first.S_first_first[:, a, b]
        + first.S_first_first[:, b, a]
        for a, b in pair_axes
    ), axis=1)
    response_weight = full_second - first_first - second_zero

    return StaticGaugeSecondOrderComponent(
        first_order=first,
        transition_d2_raw=transition_d2,
        bra_energy_dq2_ry=energy_d2,
        occupation_difference_dq2=jnp.zeros_like(energy_d2),
        S_second_zero_zero_second=second_zero,
        S_response_weight=response_weight,
        S_bubble_second_derivative=full_second,
        ordered_curvature_residual=ordered_curvature_residual,
        q2_symmetry_residual=q2_symmetry_residual,
        nb_logical=logical,
    )


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
    later inserts the independent ``+i epsilon[b,a,i]`` CT convention.
    """
    from common.bispinor_init import HALFALPHA

    gamma = jnp.asarray(gamma_raw, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    if gamma.ndim != 4 or gamma.shape[1] != 3:
        raise ValueError(
            "gamma_raw must be (nk,3,nb,nb) from "
            f"sweep_uniform_gauge_matrix_elements; got {gamma.shape}")
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

    # ``sweep_uniform_gauge_matrix_elements`` stores both band axes at the
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
    :func:`common.mtxel_sweep.sweep_uniform_gauge_matrix_elements` with its
    transfer-q2 capability enabled.  That requirement is provenance, not a
    Hall-algebra dependency: it guarantees that the returned fingerprint is
    also the identity of the current/contact/response-jet transaction a
    complete :class:`gw.head_correction.StaticGaugeHeadResponse` will use.

    Energies and occupations are read from the same ``WfnLoader`` and unfolded
    from its file wedge through :func:`symmetry_maps.unfold_file_wedge_to_full_bz`.
    Consequently ``Gamma_raw``, energies and occupations all have one row per
    physical full-BZ k before the sole ``1/Nk`` normalization is applied.  No
    driver-local star reconstruction, wavefunction reopen, band-matrix gather,
    FFT, current operator, or second Hall contraction is introduced here.
    """
    from common.mtxel_sweep import UniformGaugeMatrixElements
    from symmetry_maps import unfold_file_wedge_to_full_bz

    if not isinstance(uniform_gauge, UniformGaugeMatrixElements):
        raise TypeError(
            "static gauge Hall production requires the canonical "
            "UniformGaugeMatrixElements transaction")
    if (uniform_gauge.dgamma_dq_raw is None
            or uniform_gauge.d2gamma_dq2_raw is None):
        raise ValueError(
            "GATE static_gauge_hall_incomplete_transaction: Hall artifact "
            "provenance requires a uniform-gauge sweep with "
            "include_transfer_q2=True so its fingerprint also binds the "
            "response jet")

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
    if tuple(uniform_gauge.lambda_raw.shape) != (
            nk_tot, 3, 3, storage, storage):
        raise ValueError(
            "uniform-gauge Hall transaction has an invalid exact-contact "
            f"shape {uniform_gauge.lambda_raw.shape}")
    if tuple(uniform_gauge.dgamma_dq_raw.shape) != (
            nk_tot, 3, 3, storage, storage):
        raise ValueError(
            "uniform-gauge Hall transaction has an invalid first transfer "
            f"jet shape {uniform_gauge.dgamma_dq_raw.shape}")
    if tuple(uniform_gauge.d2gamma_dq2_raw.shape) != (
            nk_tot, 3, 3, 3, storage, storage):
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
        band_start=start,
        band_stop=stop,
        nk_tot=nk_tot,
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


def build_dft_head_response(
    wfns,
    omegas_ry,
    *,
    input_dir: str,
    mesh: Mesh,
    wfn,
    meta,
    config,
) -> IterationHeadResponse:
    """Build the one-shot DFT head on exactly the chi0 band manifold.

    This is the non-self-consistent entry to the same sharded direct-head and
    wing kernels used by QSGW.  In particular, both ``S_direct`` and ``Y/Z``
    use ``[b0,b4_chi)``; constructing the scalar from every band in a larger
    dipole file while the body uses ``number_bands_chi`` is refused by shape
    and slicing here rather than silently mixing transition manifolds.
    """
    import os
    from common.chi_from_dipole import read_dipole_h5

    dipole_path = os.path.join(input_dir, "dipole.h5")
    if not os.path.exists(dipole_path):
        raise FileNotFoundError(
            "head_correction=full requires dipole.h5 to build the direct "
            f"head and wings; missing {dipole_path}.")
    # Fail before the host read and every sharded head allocation.  Shape does
    # not identify a velocity artifact: in particular, a two-spinor dipole and
    # a kinetic-balance four-spinor dipole have the same (3,nk,nb,nb) shape.
    # The producer owns both the stamp grammar and sign resolution; consume
    # those owners directly rather than mirroring either convention here.
    from psp.get_dipole_mtxels import (
        check_dipole_provenance, resolve_vnl_velocity_sign)
    expected_vnl_sign = resolve_vnl_velocity_sign(
        None, config.vnl_velocity_sign)
    if not check_dipole_provenance(
            dipole_path,
            wfn=wfn,
            nval=int(config.nval),
            ncond=int(config.ncond),
            nband=int(config.nband),
            bispinor=bool(int(meta.nspinor) == 4),
            skip_vnl=False,
            vnl_mode="analytic",
            vnl_velocity_sign=expected_vnl_sign):
        raise ValueError(
            "GATE dft_head_dipole_provenance: head_correction=full refuses "
            "dipole.h5 because its WFN/q→0-coverage/VNL/representation "
            "provenance does not authenticate the charge-response body. "
            "Regenerate the "
            "dipole from this run's deck before building S_direct or wings.")
    velocity_cart, _ = read_dipole_h5(dipole_path)
    b0 = int(meta.b_id_0)
    b4 = int(meta.b_id_4_chi_user)
    nb_logical = b4 - b0
    velocity_cart = np.asarray(velocity_cart)[:, :, b0:b4, b0:b4]
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
    ``WfnLoader.trs_holds`` (the spin-density measurement).  ``None`` means
    the verdict is unavailable, and the statistic is then returned with
    ``verdict = nan`` rather than being read as a pass — an unmeasured
    system is not a TRS system.

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
