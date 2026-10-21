"""Two-point GN/HL-PPM fit and its adapter to shared dynamic Sigma.

The PPM-specific work ends after fitting one ``(Omega, B)`` pair for every
``(q, mu, nu)``.  :func:`compute_sigma_c_ppm_omega_grid` resolves invalid-pole
policy, writes those fields as a finalized one-pole MPA store, and delegates
all dynamic windows, box rules, cache use, tau execution and accumulation to
``gw.mpa.sigma``.  Only the static-COHSEX completion for invalid PPM poles
remains outside that store because it is frequency independent.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common import timing
from common.units import RYD_TO_EV
from .gw_config import DynamicSigmaConfig, PPMConfig
from .minimax_screening import (
    GN_PPM_EXTREME_TAIL_DIVISOR,
    fit_gn_ppm_from_wc_pair,
)
from .ppm_windows import (
    branches_for_omega_grid,
    resolve_sigma_regularization,
)
from runtime.padding import PaddedAxis

if TYPE_CHECKING:
    from .band_extrapolation import BandBracketPlan


@dataclass(frozen=True)
class PPMBuildResult:
    omega_p: float
    Wc0_q: jax.Array          # (nq, μ, μ) static W^c(0) = W(0) − V; the data
                              # seam for the invalid-pole static-COHSEX term
                              # (ppm_invalid_mode="static_limit", BGW mode 3).
                              # Identity: Wc0 = −2·B_q/Ω_q elementwise.
    B_q: jax.Array            # (nq, μ, μ) PPM amplitude: (R_+ + R_-)/2
    Omega_q: jax.Array        # (nq, μ, μ) PPM pole frequency
    valid_mask_q: jax.Array   # (nq, μ, μ)
    unfulfilled_fraction: float
    n_nodes_static: int
    #: The odd residue D = (R_+ - R_-)/2 of the ordered-orientation fit on a
    #: measured-broken-TR deck.  Conduction branches consume R_+ = B + D;
    #: valence branches consume R_- = B - D.  ``None`` is the incumbent
    #: single-residue model.
    B_odd_q: jax.Array | None = None
    probe_hermiticity_residual: float | None = None
    odd_even_residue_ratio: float | None = None
    #: The q wedge the fields live on (a ``symmetry_maps.QirrOperator``
    #: carrying the tables, no values), or ``None``: the full zone (HL, an
    #: unreduced grid).  Every Sigma consumer unfolds them on load.
    q_wedge: object = None


def _residue_for_space(space: str, B_q, B_odd_q=None):
    """The pole residue one Σ branch evolves: ``R_+`` (cond) or ``R_-`` (val).

    ``Σ_c(E) = Σ_occ ψψ†⊙R_-/(E−ε+Ω) + Σ_emp ψψ†⊙R_+/(E−ε−Ω)`` — the
    conduction (empty) A-space carries the +Ω residue, the valence
    (occupied) A-space the −Ω one (derivation §4, verified against the
    imaginary-axis contour in ``tests/test_gnppm_ordered_orientations.py``).
    With ``B_odd_q=None`` both branches get ``B_q`` and nothing about the
    incumbent path changes, not even the object identity.  A zero-valued
    ``B_odd_q`` (the explicit broken-TR debug A/B arm) likewise gives
    ``R_+ = R_- = B_q`` while retaining the odd-channel observability seam.
    """
    if space not in ("cond", "val"):
        raise ValueError(
            f"_residue_for_space: space must be 'cond' or 'val'; got {space!r}")
    if B_odd_q is None:
        return B_q
    return B_q + B_odd_q if space == "cond" else B_q - B_odd_q


@lru_cache(maxsize=16)
def band_diagonal_slots(mesh_xy, ndim: int):
    """Band diagonal of a ``P(..., 'x', 'y')`` band-tiled array, kept in place.

    ``(..., nb, nb) -> (..., nb, p_y)`` at the same spec: the tile that owns
    element ``(n, n)`` writes it into its one slot, every other slot is an
    exact zero, so the sum over the last axis is the diagonal bit for bit and
    nothing moves between ranks.  Needs a square processor mesh.
    """
    from jax import shard_map
    if int(mesh_xy.shape["x"]) != int(mesh_xy.shape["y"]):
        raise ValueError("band-diagonal slots need a square processor mesh")
    spec = P(*((None,) * (ndim - 2)), "x", "y")

    def slots(tile):
        own = jax.lax.axis_index("x") == jax.lax.axis_index("y")
        diag = jnp.diagonal(tile, axis1=-2, axis2=-1)
        return jnp.where(own, diag, jnp.zeros((), tile.dtype))[..., None]

    return shard_map(slots, mesh=mesh_xy, in_specs=spec, out_specs=spec,
                     check_vma=False)


@lru_cache(maxsize=8)
def _diagonal_of_cube_fn(mesh_xy):
    return jax.jit(band_diagonal_slots(mesh_xy, 4))


@lru_cache(maxsize=8)
def _diagonal_sum_fn(mesh_xy):
    return jax.jit(lambda values: jnp.sum(values, axis=-1),
                   out_shardings=NamedSharding(mesh_xy, P()))


@lru_cache(maxsize=8)
def _diagonal_slots_read_fn(mesh_xy):
    return jax.jit(lambda values, s, k, n: jnp.sum(values, axis=-1)[s, k, n],
                   out_shardings=NamedSharding(mesh_xy, P()))


@dataclass(frozen=True)
class BandDiagonalSlots:
    """The band diagonal of a Sigma(omega) cube without the cube.

    ``values`` is ``(n_omega, nk, nb, p_y)`` at ``P(None, None, 'x', 'y')``
    (:func:`band_diagonal_slots`).  The readers return what the same reads of
    the cube return.
    """
    values: jax.Array
    mesh_xy: Mesh

    def logical(self, band_axis=None, *, host=True):
        """``(n_omega, nk, nb_logical)`` on the host (None with ``host=False``)."""
        diagonal = _diagonal_sum_fn(self.mesh_xy)(self.values)
        if not host:
            return None
        diagonal = np.asarray(diagonal)
        if band_axis is not None:
            from runtime.padding import strip_axis
            diagonal = np.asarray(strip_axis(diagonal, band_axis, axis=-1))
        return diagonal

    def at_omega_slots(self, slots_s_kn):
        """``d[slots[s, k, n], k, n]`` on the host, (n_slot, nk, nb_logical)."""
        slots = np.asarray(slots_s_kn)
        k_idx = np.arange(slots.shape[1])[None, :, None]
        n_idx = np.arange(slots.shape[-1])[None, None, :]
        return np.asarray(_diagonal_slots_read_fn(self.mesh_xy)(
            self.values, slots, k_idx, n_idx))

    def block_until_ready(self):
        self.values.block_until_ready()
        return self


#: The on-shell evaluation tables, in slot order (:func:`on_shell_request`).
#: ``pair``: E_eval as the QSGW build reads it (``sigma_out_of_grid``);
#: ``eval0``/``dft0``: E_eval and E_DFT under the eqp readers' rule (an
#: uncovered state reads Sigma(omega = 0)); ``zero``/``edge`` (the SC
#: grid-edge census, ``sigma_out_of_grid = static`` only): omega = 0 and the
#: sampled range's edge nearer each state.  A table equal to an earlier one
#: is not repeated; :meth:`OnShellRequest.slot` resolves the name.
ON_SHELL_TABLES = ("pair", "eval0", "dft0", "zero", "edge")


@dataclass(frozen=True)
class OnShellRequest:
    """Where an on-shell Sigma_c is evaluated: one energy per state and table.

    ``tables_ry`` ``(n_table, nk, nb_logical)`` float64, Ry, measured from the
    Sigma body's omega reference, full BZ, logical Sigma bands; table 0 is
    ``pair``.  ``parent_ry`` is the same tables on the parent k rows the sweep
    evaluates (``centroid_k_unfold.parent_rows``), ``frequencies_ry`` their
    sorted distinct union, the planner's frequency list, and ``parent_index``
    each parent entry's position in it.  ``covered_kn`` marks the states whose
    E_eval lies inside the sampled range ``omega_range_ev``; the derivative
    slot is read only there (Z = 1 elsewhere, as the grid's clamped probes
    gave).
    """
    tables_ry: np.ndarray
    names: tuple
    aliases: dict
    parent_ry: np.ndarray
    frequencies_ry: np.ndarray
    parent_index: np.ndarray
    covered_kn: np.ndarray
    omega_range_ev: tuple

    @property
    def n_table(self) -> int:
        return int(self.tables_ry.shape[0])

    @property
    def n_slot(self) -> int:
        """One value slot per table, then dSigma/domega on ``pair``."""
        return self.n_table + 1

    def slot(self, name: str) -> int:
        """The diagonal slot of table ``name`` (``"derivative"``: the last)."""
        if name == "derivative":
            return self.n_table
        return int(self.aliases[name])

    def table_ev(self, name: str) -> np.ndarray:
        return self.tables_ry[self.slot(name)] * RYD_TO_EV

    def select_k(self, select_rows) -> "OnShellRequest":
        """The request on the k rows ``select_rows`` keeps (a leading-axis selector).

        The parent-row fields are the sweep's and stay as they are."""
        tables = np.moveaxis(np.asarray(select_rows(np.moveaxis(self.tables_ry, 1, 0))), 0, 1)
        return replace(self, tables_ry=tables,
                       covered_kn=np.asarray(select_rows(np.asarray(self.covered_kn))))


def on_shell_request(*, e_eval_rel_ev, e_dft_rel_ev, omega_range_ev, out_of_grid,
                     irr_idx, n_parent, edge_window_ev=None):
    """The on-shell tables for one Sigma call (:class:`OnShellRequest`).

    ``e_eval_rel_ev`` / ``e_dft_rel_ev`` ``(nk, nb)``: the energies the QSGW
    ansatz evaluates Sigma at and the DFT energies, both measured from the
    Sigma body's omega reference.  ``omega_range_ev`` is the sampled range
    (``config.omega_grid_ev``: only its extent is read).  The read rules are
    :func:`gw.qsgw_utils.sigma_eval_omega` for ``pair`` (what H reads) and
    the eqp readers' Sigma(omega = 0) rule for ``eval0``/``dft0``.
    ``irr_idx``/``n_parent``: the full-k -> parent map of the sweep's k plan.
    ``edge_window_ev`` (the SC growth window, ``sigma_out_of_grid = static``):
    add the ``zero`` and ``edge`` tables of the grid-edge census
    (``qsgw_utils.sigma_grid_edge_ambiguity``).

    ``GATE on_shell_star_invariant``: the energies must be the same at every
    k of a star (within 1e-9 Ry), since the wedge Sigma is evaluated at its
    parent's energies and transported to the star.
    """
    from .centroid_k_unfold import parent_rows
    from .qsgw_utils import omega_coverage, sigma_eval_omega
    grid = np.asarray(omega_range_ev, dtype=np.float64)
    e_eval = np.asarray(e_eval_rel_ev, dtype=np.float64)
    e_dft = np.asarray(e_dft_rel_ev, dtype=np.float64)
    if e_eval.shape != e_dft.shape or e_eval.ndim != 2:
        raise ValueError(f"on_shell_request: E_eval {e_eval.shape} and E_DFT "
                         f"{e_dft.shape} must be one (nk, nb) shape")
    lo, hi = float(grid[0]), float(grid[-1])
    if not lo <= 0.0 <= hi:
        raise ValueError("on_shell_request: omega = 0 must lie in the sampled range "
                         f"[{lo:.3f}, {hi:.3f}] eV")
    pair = sigma_eval_omega(grid, e_eval, out_of_grid)[0]
    covered = omega_coverage(grid, e_eval)[0]
    candidates = [("pair", pair), ("eval0", np.where(covered, e_eval, 0.0)),
                  ("dft0", np.where(omega_coverage(grid, e_dft)[0], e_dft, 0.0))]
    if edge_window_ev is not None:
        top = (np.abs(e_eval - max(hi, float(edge_window_ev[1])))
               <= np.abs(e_eval - min(lo, float(edge_window_ev[0]))))
        candidates += [("zero", np.zeros_like(e_eval)), ("edge", np.where(top, hi, lo))]
    tables, names, aliases = [], [], {}
    for name, table in candidates:
        for i, kept in enumerate(tables):
            if np.array_equal(kept, table):
                aliases[name] = i
                break
        else:
            aliases[name] = len(tables)
            tables.append(table)
            names.append(name)
    tables_ry = np.stack(tables) / RYD_TO_EV
    parent = np.asarray(parent_rows(irr_idx, int(n_parent), tables_ry, axis=1))
    drift = float(np.max(np.abs(parent[:, np.asarray(irr_idx)] - tables_ry), initial=0.0))
    if drift > 1.0e-9:
        raise ValueError(
            f"GATE on_shell_star_invariant: the on-shell energies differ by {drift:.3e} Ry "
            "between k points of one star")
    frequencies, index = np.unique(parent, return_inverse=True)
    return OnShellRequest(
        tables_ry=tables_ry, names=tuple(names), aliases=aliases, parent_ry=parent,
        frequencies_ry=frequencies,
        parent_index=np.asarray(index, dtype=np.int32).reshape(parent.shape),
        covered_kn=np.asarray(covered, dtype=bool), omega_range_ev=(lo, hi))


def ordered_window_request(energy_ry, efermi_ry):
    """Exact current-shell request for the ordered instantaneous window.

    Parameters
    ----------
    energy_ry : ndarray (nk, n_active)
        Current absolute electronic shell energies in Ry. Row and column
        matrix samples use these same shells, never a legacy SC grid fallback.
    efermi_ry : float
        Common absolute frequency reference in Ry for both external shells
        and internal Green energies.

    Returns
    -------
    request, receipt : OnShellRequest, dict
        Native request and its single scalar planning domain: the current
        active minimum/maximum with a 1 eV margin. Zero is included in the
        domain if required by the native request contract, but is not added
        as an evaluation shell. All full-band virtual lines remain unchanged.
    """
    energy = np.asarray(energy_ry, np.float64)
    efermi = float(efermi_ry)
    if (energy.ndim != 2 or not energy.size or not np.isfinite(energy).all()
            or not np.isfinite(efermi)):
        raise ValueError("GATE ordered_window_request: finite current shell energies required")
    external = (energy-efermi)*RYD_TO_EV
    margin = 1.0
    actual = (float(external.min()), float(external.max()))
    domain = (min(0.0, actual[0]-margin), max(0.0, actual[1]+margin))
    request = on_shell_request(e_eval_rel_ev=external, e_dft_rel_ev=external,
        omega_range_ev=np.asarray(domain), out_of_grid="cover",
        irr_idx=np.arange(len(energy)), n_parent=len(energy))
    expected = external/RYD_TO_EV
    if (not np.all(request.covered_kn)
            or not np.array_equal(request.tables_ry[request.slot("pair")], expected)
            or any(not np.array_equal(table, expected) for table in request.tables_ry)):
        raise ValueError("GATE ordered_window_request: current shells changed or fell back to static reads")
    receipt = dict(policy="exact_current_active_shells", active_shape=list(energy.shape),
        external_shell_range_ev=list(actual), domain_ev=list(domain), margin_ev=margin,
        frequency_reference_ry=efermi, all_shells_covered=True,
        static_fallback_reads=0, legacy_SC_grid_used=False,
        evaluation_frequency_count=int(request.frequencies_ry.size),
        domain_boundaries_are_evaluation_frequencies=False)
    return request, receipt


def slots_from_diagonal(diag, n_slot_cols):
    """``(..., nb)`` -> the :func:`band_diagonal_slots` layout ``(..., nb, p_y)``.

    Band ``n`` lives on the tile row block ``n // (nb / p_y)``, whose diagonal
    tile writes it into column ``n // (nb / p_y)``; every other column is an
    exact zero.  An elementwise select, so it fuses into its consumer.
    """
    nb = int(diag.shape[-1])
    block = nb // int(n_slot_cols)
    owner = (jnp.arange(nb)[:, None] // block) == jnp.arange(int(n_slot_cols))[None, :]
    return jnp.where(owner, diag[..., :, None], jnp.zeros((), diag.dtype))


def _diag_matrix(diag):
    """``(..., nb)`` -> ``(..., nb, nb)`` with ``diag`` on the band diagonal (fused select)."""
    nb = int(diag.shape[-1])
    eye = jnp.arange(nb)[:, None] == jnp.arange(nb)[None, :]
    return jnp.where(eye, diag[..., :, None], jnp.zeros((), diag.dtype))


@lru_cache(maxsize=8)
def _on_shell_head_fn(mesh_xy):
    """Add a band-diagonal head to an on-shell Sigma: the pair's diagonal and
    every value slot get the head's value at their table, the derivative slot
    its derivative.  ``head`` is ``(n_slot, nk, nb_carrier)`` replicated;
    ``pair_head`` its pair-table row twice."""
    spec = NamedSharding(mesh_xy, P(None, None, "x", "y"))

    def add(pair, slots, head, pair_head):
        slots = slots + slots_from_diagonal(head, slots.shape[-1])
        return (None if pair is None else pair + _diag_matrix(pair_head)), slots
    return jax.jit(add, out_shardings=(spec, spec))


@dataclass(frozen=True)
class OnShellSigma:
    """Sigma_c on shell: the QSGW pair and the eqp readers' diagonals, no omega axis.

    ``pair`` ``(2, nk, nb, nb)`` at ``P(None, None, 'x', 'y')``:
    ``pair[0, k, m, n] = Sigma_mn(k, omega_km)`` and ``pair[1, k, m, n] =
    Sigma_mn(k, omega_kn)`` on ``request``'s ``pair`` table; ``None`` for a
    diagonal-only twin.  ``slots`` ``(n_slot, nk, nb, p_y)`` at the same spec
    (:func:`band_diagonal_slots` layout): the band diagonal on each table, then
    dSigma_nn/domega on ``pair``.  Every array is Ry (the derivative is
    dimensionless) on the full BZ and the padded band carrier.
    """
    pair: "jax.Array | None"
    slots: jax.Array
    request: OnShellRequest
    mesh_xy: Mesh
    #: The ``(n_slot, nk, nb_carrier)`` host diagonals, read once (:meth:`diagonal`).
    _diagonals: "np.ndarray | None" = dataclasses.field(
        default=None, init=False, repr=False, compare=False)

    @property
    def shape(self):
        """The pair's shape (its slot axis leads)."""
        return tuple(self.pair.shape) if self.pair is not None else tuple(self.slots.shape)

    @property
    def dtype(self):
        return self.slots.dtype

    def diagonal(self, name: str, band_axis=None):
        """``(nk, nb_logical)`` on the host: table ``name``'s diagonal, or ``"derivative"``.

        Every slot's diagonal is read in one collective the first time any is
        asked for (every rank enters it) and served from the host after."""
        if self._diagonals is None:
            object.__setattr__(self, "_diagonals",
                               np.asarray(_diagonal_sum_fn(self.mesh_xy)(self.slots)))
        value = self._diagonals[self.request.slot(name)]
        if band_axis is not None:
            from runtime.padding import strip_axis
            value = np.asarray(strip_axis(value, band_axis, axis=-1))
        return value

    def with_head(self, head, band_axis=None):
        """Add the band-diagonal q->0 head (``head_correction.HeadSigmaDiag``)."""
        if head is None:
            return self
        from common.collectives import device_put_process_local
        values, deriv = [], None
        for name in self.request.names:
            v, d = head.on_shell(self.request.tables_ry[self.request.slot(name)])
            values.append(v)
            if name == "pair":
                deriv = d
        rows = np.stack(values + [deriv])
        carrier = int(self.slots.shape[2])
        if rows.shape[-1] != carrier:
            rows = np.pad(rows, ((0, 0), (0, 0), (0, carrier - rows.shape[-1])))
        rep = NamedSharding(self.mesh_xy, P())
        head_rows = device_put_process_local(np.asarray(rows, np.complex128), rep)
        pair_head = device_put_process_local(
            np.ascontiguousarray(np.broadcast_to(rows[:1], (2,) + rows.shape[1:])), rep)
        pair, slots = _on_shell_head_fn(self.mesh_xy)(
            self.pair, self.slots, head_rows, pair_head)
        return replace(self, pair=pair, slots=slots)

    def select_k(self, select_rows) -> "OnShellSigma":
        """On the k rows ``select_rows`` keeps (a leading-axis selector)."""
        def rows(a):
            return None if a is None else jnp.moveaxis(
                select_rows(jnp.moveaxis(a, 1, 0)), 0, 1)
        return replace(self, pair=rows(self.pair), slots=rows(self.slots),
                       request=self.request.select_k(select_rows))

    def __add__(self, other: "OnShellSigma") -> "OnShellSigma":
        """The sum of two evaluations of one request (the four-current classes)."""
        return add_on_shell_term(self, other)

    def __sub__(self, other: "OnShellSigma") -> "OnShellSigma":
        """The difference of two evaluations of one request (the ordered-residue twin)."""
        if other.request is not self.request:
            raise ValueError("OnShellSigma: a difference needs one request")
        sub = _on_shell_difference_fn(self.mesh_xy)
        return replace(self, slots=sub(self.slots, other.slots),
                       pair=None if self.pair is None else sub(self.pair, other.pair))

    def block_until_ready(self):
        jax.block_until_ready((self.pair, self.slots))
        return self


@lru_cache(maxsize=8)
def _on_shell_difference_fn(mesh_xy):
    return jax.jit(lambda a, b: a - b,
                   out_shardings=NamedSharding(mesh_xy, P(None, None, "x", "y")))


def is_on_shell(sigma) -> bool:
    """Whether ``sigma`` is the on-shell evaluation; else the omega-grid cube
    (``sigma_omega_grid_debug``)."""
    return isinstance(sigma, OnShellSigma)


@dataclass(frozen=True)
class BandCountCube:
    """The cumulative band-count points of the on-shell Sigma_c, served one at a time.

    The sweep keeps, on the FILE wedge, the pair of the first and of the last
    count (``matrices``, each ``(2, n_wedge_k, nb, nb)``) and the diagonal
    slots of EVERY count (``slot_diagonals``, each ``(n_slot, n_wedge_k, nb,
    p_y)``).  That is what the band extrapolation reads: the fit takes each
    count's diagonals (:meth:`diagonal`), and its matrix combines N1 and N3
    only, N2 carrying weight zero (``SpectralShellFit.weights``); the
    derivative and the eqp tables of the extrapolated Sigma combine every
    count's slots.  :meth:`point` returns a matrix count on the full BZ as an
    :class:`OnShellSigma`.

    On the omega grid (``request`` None, ``sigma_omega_grid_debug``) the
    matrices are ``(n_omega, n_wedge_k, nb, nb)``, ``diagonals`` holds the
    band diagonal of every count between the first and the last, and
    :meth:`point` returns ``(n_omega, nk, nb, nb)`` at ``P(None, None, 'x',
    'y')``.  The ``(n_count, n_omega, nk, nb, nb)`` cube is never made (Na
    8^3 at 1001 omega: 42.4 GiB per rank in one allocation at P4).

    ``fresh``: ``unfold`` returns a new buffer.  Where the wedge is the full
    BZ it is false and a matrix point IS the stored array.
    """
    matrices: list
    slot_diagonals: list
    #: wedge pair -> full BZ; wedge diagonal slots -> full BZ.
    unfold: Callable
    unfold_rows: Callable
    nk: int
    fresh: bool
    mesh_xy: Mesh
    #: The on-shell evaluation, or None (the omega grid).
    request: "OnShellRequest | None"
    #: Added to every point: ``(nk, nb, nb)``, the PPM static-limit term (the
    #: same at every energy), or an :class:`OnShellSigma` of the four-current
    #: classes and constant that are not extrapolated
    #: (``gw.mpa.sector_sigma.compute_sector_sigma``); on the grid that is
    #: ``(n_omega, nk, nb, nb)`` on the full BZ.
    term: "jax.Array | OnShellSigma | None" = None
    #: Grid only: the band diagonal of each count between the first and last.
    diagonals: list = dataclasses.field(default_factory=list)
    #: The counts whose matrix ``matrices`` holds, ascending
    #: (``band_extrapolation_checkpoints``: the main plan's N1 and N3 among
    #: the union's counts); ``()`` is the first and the last.
    matrix_index: tuple = ()

    @property
    def shape(self):
        if self.request is None:
            n_omega, _, nb_i, nb_j = self.matrices[0].shape
            return (len(self.matrices) + len(self.diagonals), n_omega, int(self.nk), nb_i, nb_j)
        _, nk, nb_i, nb_j = self.matrices[0].shape
        return (len(self.slot_diagonals), 2, int(self.nk), nb_i, nb_j)

    def diagonal_only(self, i: int) -> bool:
        if self.matrix_index:
            return int(i) not in self.matrix_index
        return 0 < int(i) < self.shape[0] - 1

    def _matrix(self, i: int):
        if self.matrix_index:
            return self.matrices[self.matrix_index.index(int(i))]
        return self.matrices[0 if int(i) == 0 else -1]

    def restrict(self, indices) -> "BandCountCube":
        """The counts ``indices`` (ascending) as a cube of their own.

        The first and the last of them must carry their matrix; the counts
        between are served by their diagonal.  On shell only (the union sweep
        of ``band_extrapolation_checkpoints`` keeps the matrices of its main
        plan's N1 and N3).  The members are shared, not copied.
        """
        idx = tuple(int(i) for i in indices)
        if self.request is None:
            raise ValueError("BandCountCube.restrict: on-shell cubes only")
        if list(idx) != sorted(set(idx)) or not idx:
            raise ValueError(f"BandCountCube.restrict: indices {idx} must ascend")
        if self.diagonal_only(idx[0]) or self.diagonal_only(idx[-1]):
            raise ValueError(
                f"BandCountCube.restrict: counts {idx[0]} and {idx[-1]} must keep "
                f"their matrix; kept {self.matrix_index or (0, self.shape[0] - 1)}")
        ends = (idx[0],) if len(idx) == 1 else (idx[0], idx[-1])
        return dataclasses.replace(
            self, matrices=[self._matrix(i) for i in ends], diagonals=[],
            slot_diagonals=[self.slot_diagonals[i] for i in idx], matrix_index=())

    def point(self, i: int) -> OnShellSigma:
        if self.diagonal_only(i):
            raise ValueError(
                f"BandCountCube: count {i} is kept on its band diagonal "
                f"only; read it with diagonal({i})")
        value = self.unfold(self._matrix(i))
        if self.request is None:
            if self.term is None:
                return value
            return _add_static_term_fn(value.sharding, self.fresh)(value, self.term)
        return self._point(i, value)

    def diagonal(self, i: int) -> OnShellSigma:
        """Count ``i``'s diagonal slots on the full BZ (a diagonal-only point).

        On the grid: the count's band diagonal, without its cube."""
        if self.request is None:
            return self._grid_diagonal(i)
        return self._point(i, None)

    def _point(self, i: int, pair) -> OnShellSigma:
        point = OnShellSigma(pair=pair, slots=self.unfold_rows(self.slot_diagonals[int(i)]),
                             request=self.request, mesh_xy=self.mesh_xy)
        return point if self.term is None else add_on_shell_term(point, self.term)

    def _grid_diagonal(self, i: int) -> "BandDiagonalSlots":
        slots = (self.diagonals[int(i) - 1] if self.diagonal_only(i) else
                 _diagonal_of_cube_fn(self.mesh_xy)(self._matrix(i)))
        slots = self.unfold_rows(slots)
        if self.term is not None:
            slots = _add_term_slots_fn(self.mesh_xy, self.term.ndim)(slots, self.term)
        return BandDiagonalSlots(slots, self.mesh_xy)

    def block_until_ready(self):
        jax.block_until_ready((self.matrices, self.diagonals, self.slot_diagonals))
        return self

    def release(self):
        """Drop the stored counts once the extrapolation has consumed them
        (without symmetry they are the points themselves)."""
        self.matrices.clear()
        self.diagonals.clear()
        self.slot_diagonals.clear()

    def __sub__(self, other):
        if self.term is not None or other.term is not None:
            raise ValueError(
                "BandCountCube: a difference is taken before the static term")
        return replace(
            self,
            matrices=[a - b for a, b in zip(self.matrices, other.matrices)],
            diagonals=[a - b for a, b in zip(self.diagonals, other.diagonals)],
            slot_diagonals=[a - b for a, b in zip(self.slot_diagonals,
                                                  other.slot_diagonals)])

    def __add__(self, other):
        """The sum of two sweeps' counts on one plan (the causal branches of one sweep)."""
        if self.term is not None or other.term is not None:
            raise ValueError("BandCountCube: a sum is taken before the static term")
        if other.request is not self.request:
            raise ValueError("BandCountCube: a sum needs one on-shell request (or none)")
        return replace(
            self,
            matrices=[a + b for a, b in zip(self.matrices, other.matrices)],
            diagonals=[a + b for a, b in zip(self.diagonals, other.diagonals)],
            slot_diagonals=[a + b for a, b in zip(self.slot_diagonals,
                                                  other.slot_diagonals)])


@lru_cache(maxsize=8)
def _on_shell_static_term_fn(mesh_xy, n_table):
    """An omega-independent ``(nk, nb, nb)`` term on an on-shell Sigma: both pair
    slots and every value slot take it, the derivative slot nothing."""
    term_slots = band_diagonal_slots(mesh_xy, 3)
    spec = NamedSharding(mesh_xy, P(None, None, "x", "y"))

    def add(pair, slots, term):
        d = term_slots(term)[None]
        slots = slots + jnp.concatenate(
            [jnp.broadcast_to(d, (n_table,) + d.shape[1:]),
             jnp.zeros_like(d)], axis=0)
        return (None if pair is None else pair + term[None]), slots
    return jax.jit(add, out_shardings=(spec, spec))


@lru_cache(maxsize=8)
def _on_shell_sum_fn(mesh_xy):
    spec = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    return jax.jit(lambda a, b: None if a is None else a + b, out_shardings=spec)


def add_on_shell_term(point: OnShellSigma, term) -> OnShellSigma:
    """``point + term``: a static ``(nk, nb, nb)`` term or an on-shell one."""
    if isinstance(term, OnShellSigma):
        if term.request is not point.request:
            raise ValueError("on-shell term: evaluated on a different request")
        add = _on_shell_sum_fn(point.mesh_xy)
        return replace(point, slots=add(point.slots, term.slots),
                       pair=(None if point.pair is None else add(point.pair, term.pair)))
    if term.ndim != 3:
        raise ValueError(f"on-shell term: a static (nk, nb, nb) term, got {term.shape}")
    pair, slots = _on_shell_static_term_fn(point.mesh_xy, point.request.n_table)(
        point.pair, point.slots, term)
    return replace(point, pair=pair, slots=slots)


@lru_cache(maxsize=8)
def _add_term_slots_fn(mesh_xy, ndim=3):
    """Diagonal slots of ``sigma + term`` from those of ``sigma``: the owning
    slot adds the term's diagonal, the zero slots add zero.  ``ndim``: the
    term's, 3 (one ω-independent term) or 4 (one per ω)."""
    term_slots = band_diagonal_slots(mesh_xy, ndim)
    lead = (None,) * (4 - ndim)
    return jax.jit(lambda slots, term: slots + term_slots(term)[lead],
                   out_shardings=NamedSharding(mesh_xy, P(None, None, "x", "y")))


@dataclass(frozen=True)
class SigmaOmegaResult:
    # A bracketed sum (every PPM run; MPA under band extrapolation) carries a
    # :class:`BandCountCube`: point ``i`` is Σ_c summed over bands
    # ``[0, band_counts[i])``, so the last point is the ordinary full-band
    # Σ_c.  One point in the ordinary case (``band_counts == (nband,)``), 3
    # under ``use_band_extrapolation``.  An unbracketed MPA sum carries the
    # :class:`OnShellSigma` itself, its arrays at P(None, None, 'x', 'y'); a
    # grid one (``sigma_omega_grid_debug``) the ``(n_omega, nk, nb, nb)``
    # array, band-tiled the same way (qsgw_utils.is_band_sharded_sigma_omega).
    sigma_c_kij: 'OnShellSigma | BandCountCube | jax.Array'
    #: Logical Sigma band window and its square mesh carrier.  Consumers keep
    #: the carrier while either band axis is sharded and derive every mask or
    #: slice from this receipt.
    band_axis: PaddedAxis | None = None
    #: Optional exact ordered-residue contribution on the SAME request:
    #: ``Sigma_c[B,D] - Sigma_c[B,D=0]``.  GN-PPM carries this beside the
    #: result in ``PPMOutputs`` because of its band-bracket pipeline; MPA has
    #: no bracket tail and carries it here.  ``None`` is the TRS/incumbent
    #: route and preserves its object graph.
    sigma_c_odd_kij: 'OnShellSigma | jax.Array | None' = None
    #: The LOGICAL band count each leading-axis element sums to.  Aligned
    #: with ``sigma_c_kij``'s axis 0; the extrapolation reads both together
    #: and nothing else needs either.
    band_counts: tuple[int, ...] = ()
    #: ``(n_bracket, nk, nb, nb)`` Ry, or ``None``.  The CUMULATIVE Coulomb-hole
    #: half of the ``ppm_invalid_mode="static_limit"`` term at each of
    #: ``band_counts`` — the part of Σ_c that is a static Coulomb hole and is
    #: therefore folded in as a CONSTANT rather than extrapolated.  Present
    #: only when the extrapolation is running (``len(band_counts) > 1``) and
    #: the run has invalid poles to treat.  It is NOT a component to be added
    #: to ``sigma_c_kij`` — that fold already happened; it is the evidence for
    #: how much of ``S_inf`` was never extrapolated
    #: (``band_extrapolation.static_limit_tail_ruling``).
    static_coh_at_counts: np.ndarray | None = None
    #: Measured fit-level ``max|D|/max|B|`` for an ordered MPA store.  Kept
    #: here so the shared driver report does not re-read or re-fit poles.
    odd_even_residue_ratio: float | None = None
    #: The energy (Ry) the omega axis of THIS body is measured from: the
    #: current bundle's VBM or midgap (``_prepare_sigma_state``).  The
    #: finalizer must read Sigma back in this frame; without it the lookup
    #: fell back to ``wfn.efermi`` (the DFT midgap), off by 0.5-1.1 eV on
    #: CrI3 16x16 SC (audit 2026-09-23 item 2).  None on non-PPM results.
    efermi_ry: float | None = None
    #: The omega grid of a grid result (``sigma_omega_grid_debug``): the
    #: cube's leading axis, Ry and eV.  None on shell.
    omega_ry: np.ndarray | None = None
    omega_ev: np.ndarray | None = None
    #: ``keep_branches`` sweeps only: ``{"cond": Sigma_p, "val": Sigma_h}``, the
    #: dynamic body of each causal branch (``gw.mpa.sigma.BRANCH_SPACES``), each
    #: of ``sigma_c_kij``'s type before any static term: Sigma_c = Sigma_p +
    #: Sigma_h (+ the PPM invalid-pole static term, + the caller's head).
    #: ``None`` is the incumbent result (docs/theory/td_agw_dynamical.md §1.5).
    branches: dict | None = None




# ---------------------------------------------------------------------------
#  Physics-state prep — single jit that collapses the scattered trace-time
#  jnp operations the driver used to emit (Fermi level, band masks, PPM
#  pole masks, invalid-count tallies).
# ---------------------------------------------------------------------------

class _SigmaPhysicsState(NamedTuple):
    efermi: jax.Array          # scalar
    E_cond: jax.Array          # (nk, nb_full)  max(enk - efermi, 0)
    H_val: jax.Array           # (nk, nb_full)  max(efermi - enk, 0)
    cond_mask: jax.Array       # (nk, nb_full)  bool
    val_mask: jax.Array        # (nk, nb_full)  bool
    B_corr: jax.Array          # (nq, μ, μ)     c128, ready-to-contract B_q
    Omega_abs: jax.Array       # (nq, μ, μ)     f64,  max(Re Ω_q, 0)
    B_mask: jax.Array          # (nq, μ, μ)     bool, B_mask_raw & valid
    invalid_mask: jax.Array    # (nq, μ, μ)     bool, logical modes with Ω²<0
    n_total_modes: jax.Array   # scalar int64
    n_invalid: jax.Array       # scalar int64


def assert_gapped_occupations_for_ppm(occ_full, *, print_fn=print) -> int:
    """Refuse a GN/HL-PPM Σ whose occupation table has a Fermi-crossing band.

    Returns the number of crossing bands (always 0 on the accepted path), so
    a caller or a test reads a NUMBER rather than the absence of an
    exception.

    WHAT IS MEASURED, AND WHY THE MATERIAL CHECK DOES NOT ANSWER IT.
    ``gw_config.validate_material_inputs`` already refuses GN-PPM on a WFN
    whose occupations are fractional (``GATE gn_ppm_refuses_metals``, owner
    ruling 2026-09-17: GN-PPM is never allowed for metals).  A WFN written
    with integer occupations can still carry a band that crosses the Fermi
    level of the step table, and that is a property of the spectrum.  So the
    measurement here is on the occupation table:

        band n crosses E_F  <=>  occ[:, n] > 0.5 is not constant over k.

    ``wavefunction_bundle._build_occ`` fills ``occ`` as ``(enk <= efermi)``,
    exactly 0.0/1.0, so on a gapped system every band is uniformly occupied
    or uniformly empty and this is exactly zero.  A crossing band is the one
    thing that makes it nonzero, and it needs no tolerance: the predicate is
    over an integer table.

    WHAT GOES WRONG IF IT RUNS ANYWAY, which is why this refuses rather than
    warns.  ``_prepare_sigma_state`` builds ``vbm = max(enk | occupied)`` and
    ``cbm = min(enk | empty)``; with a crossing band ``vbm > cbm``, so the
    "midgap" ``0.5*(vbm+cbm)`` is not in any gap and the ``fermi_reference``
    the deck chose is meaningless.  It then clips
    ``E_cond = max(enk - efermi, 0)`` and ``H_val = max(efermi - enk, 0)``, so
    a band on the wrong side of that pseudo-Fermi level cannot even be
    REPRESENTED — its dynamic denominator collapses to the ω = 0 edge.  Every
    array keeps its shape and the run completes.  Measured on Na bcc SOC 48b
    (`reports/occupation_threshold_all_paths_2026-08-16/evidence/probe_na.log`,
    JID 57138992): MP1 gives f in [-0.002194, +1.031587] with 150
    negative-lobe and 12 over-one (k,n) entries, and the step split assigns
    every one of them fully to one branch with weight 1.

    This is the SCOPE LIMIT the module has documented in prose since the
    ``TODO(metal-greens)`` at ``_prepare_sigma_state`` was written, enforced.
    The port itself — feeding the iteration's ``OccupationState`` into
    ``branches_for_omega_grid`` as ``cond_weight = 1-f`` / ``val_weight = f``
    — remains open work; a documented limitation that nothing enforces is a
    limitation only the reader has.
    """
    occ = np.asarray(jax.device_get(occ_full))
    if occ.ndim != 2:
        raise ValueError(
            f"assert_gapped_occupations_for_ppm: expected (nk, nb) "
            f"occupations, got shape {occ.shape}")
    filled = occ > 0.5
    crossing = np.flatnonzero(filled.any(axis=0) & ~filled.all(axis=0))
    if crossing.size == 0:
        return 0
    # No override: GN-PPM is not allowed for metals (owner ruling
    # 2026-09-17), and the former crossing-bands environment debugging
    # hatch was the one way to run it on a Fermi-crossing spectrum.
    raise ValueError(
        f"GATE ppm_sigma_gapped_occupations: this spectrum has "
        f"{crossing.size} Fermi-crossing band(s) — occupied at some k and "
        f"empty at others — at band index/indices "
        f"{crossing.tolist()[:12]}"
        + (" (first 12 shown)" if crossing.size > 12 else "") + ".\n"
        f"  got:  a metallic occupation table on the GN/HL plasmon-pole "
        f"Sigma driver, whose band split is a hard occ > 0.5 step\n"
        f"  want: a gapped spectrum, i.e. every band uniformly occupied or "
        f"uniformly empty over k\n"
        f"  fix:  GN/HL-PPM is not allowed for metals (owner ruling "
        f"2026-09-17): run this system with compute_mode = mpa and "
        f"sigma_w_model = shared_pole on a WFN with its fractional "
        f"occupations and occ_smearing_width_ry; or narrow the band window "
        f"so no band crosses E_F\n"
        f"  why:  with a crossing band, vbm > cbm, so the 'midgap' Fermi "
        f"reference this driver derives is not in any gap, and E_cond/H_val "
        f"are clipped at zero so a wrong-side band cannot be represented.  "
        f"Nothing about that changes an array shape or the exit code.\n"
        f"  doc:  docs/theory/metallic-mpa-screening.md")


def ppm_fermi_frame(enk_full, occ_full, use_midgap):
    """The origin of the PPM Sigma(omega) grid: the current spectrum's VBM, or
    ``0.5*(VBM + CBM)`` when ``use_midgap`` and a state is empty.

    The one owner of that rule: ``_prepare_sigma_state`` builds the Sigma
    frame with it, and ``efermi.sigma_frame_mu_ev`` (shared by the one-shot
    and the SC loop) calls it on the same energies to judge grid coverage in
    that frame.
    """
    occ_mask = occ_full > 0.5
    vbm = jnp.max(jnp.where(occ_mask, enk_full, -1.0e30))
    cbm = jnp.min(jnp.where(~occ_mask, enk_full, 1.0e30))
    midgap_candidate = jnp.where(jnp.any(~occ_mask), 0.5 * (vbm + cbm), vbm)
    return jnp.where(use_midgap, midgap_candidate, vbm)


@jax.jit
def _prepare_sigma_state(
    enk_full: jax.Array,
    occ_full: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    valid_mask_q: jax.Array,
    use_midgap: jax.Array,
    keep_invalid: jax.Array,
) -> _SigmaPhysicsState:
    """Derive Fermi level + derived energy/PPM arrays in one fused trace.

    Replaces ~9 eager jnp ops previously emitted at trace time by the sigma
    driver.  ``use_midgap`` is a traced bool scalar; the caller passes
    ``jnp.asarray(fermi_reference == 'midgap')``.  ``valid_mask_q`` is always
    a real bool array (the caller substitutes ``jnp.ones_like(...)`` when
    no mask is available), so the helper doesn't branch on None.

    ``keep_invalid`` is a traced bool implementing ``ppm_invalid_mode`` (BGW
    ``invalid_gpp_mode``) for poles with fitted ``Omega^2 < 0``: False = drop
    them from the τ-pole sum (``B_mask &= valid``; BGW mode 0 / "zero", and
    also the pole-sum half of "static_limit" / BGW mode 3 — the caller adds
    the analytic static-COHSEX term for the modes flagged by
    ``invalid_mask``); True = keep the fit's fallback pole at
    ``fallback_omega`` (default 2 Ry; BGW mode 2 / "2ry").

    μ-pad safety is structural, not per-consumer: pad modes are born DEAD
    at the fit (``fit_gn_ppm_from_wc_pair(n_mu_logical=...)`` zeroes their
    Ω, hence B = 0 and valid = False), so ``B_mask_raw = Ω > 1e-14``
    excludes them here — and in every other Ω/B consumer — with no mask
    argument (docs/architecture/padding.md#pad-extent-invariance).
    """
    # TODO(metal-greens): the finite-occupation Green's-function/Sigma
    # decomposition needs particle/hole weights f and 1-f, not f > 0.5.
    occ_mask = occ_full > 0.5
    unocc_mask = ~occ_mask
    efermi = ppm_fermi_frame(enk_full, occ_full, use_midgap)

    E_cond = jnp.maximum(enk_full - efermi, 0.0)
    H_val = jnp.maximum(efermi - enk_full, 0.0)

    Omega_abs = jnp.maximum(jnp.real(Omega_q), 0.0).astype(jnp.float64)
    B_corr = jnp.asarray(B_q, dtype=jnp.complex128)
    B_mask_raw = Omega_abs > 1.0e-14
    valid = jnp.asarray(valid_mask_q, dtype=bool)
    # ppm_invalid_mode: keep_invalid=False drops Omega^2<0 poles (BGW mode 0);
    # keep_invalid=True keeps the fit's fallback pole (BGW mode 2).
    B_mask = B_mask_raw & (valid | keep_invalid)
    invalid_mask = B_mask_raw & (~valid)

    return _SigmaPhysicsState(
        efermi=efermi,
        E_cond=E_cond, H_val=H_val,
        cond_mask=unocc_mask, val_mask=occ_mask,
        B_corr=B_corr, Omega_abs=Omega_abs, B_mask=B_mask,
        invalid_mask=invalid_mask,
        n_total_modes=jnp.sum(B_mask_raw, dtype=jnp.int64),
        n_invalid=jnp.sum(invalid_mask, dtype=jnp.int64),
    )


# ---------------------------------------------------------------------------
#  PPM construction
# ---------------------------------------------------------------------------

@jax.jit
def _probe_antihermitian_residual(W_q):
    """``max_q |(W_q - W_q^H) z|_inf / max_q |W_q z|_inf`` for one fixed probe.

    ``z`` has unit-modulus, golden-ratio phases. ``W^H z`` is
    ``conj(conj(z)^T W)``, so each q costs two mat-vecs and a row reduction,
    and W is never transposed.  The exact entrywise ``max|W_q - W_q^H|``
    transposed every sharded q block and took ~5 s per SC map on CrI3 16x16
    at P64 (P2-J, 2026-09-24).  An anti-Hermitian part of W is seen at the
    same relative size: the ratio is 0 for a Hermitian W.
    """
    n = W_q.shape[-1]
    phase = jnp.mod(jnp.arange(n, dtype=jnp.float64) * 0.6180339887498949, 1.0)
    z = jnp.exp(2j * jnp.pi * phase).astype(W_q.dtype)
    w_z = jnp.einsum("qij,j->qi", W_q, z)
    wh_z = jnp.conj(jnp.einsum("j,qji->qi", jnp.conj(z), W_q))
    scale = jnp.max(jnp.abs(w_z))
    return jnp.where(scale > 0.0, jnp.max(jnp.abs(w_z - wh_z)) / scale, 0.0)


def fit_ppm(
    W0_q: jax.Array,
    Wprobe_q: jax.Array,
    V_q: jax.Array,
    probe_omega: complex,
    mesh_xy: Mesh,
    *,
    fallback_omega: float = 2.0,
    n_nodes_static: int = 0,
    print_fn=None,
    model_label: str = "PPM",
    n_mu_logical: int,
    mu_active_mask=None,
    q_neg_index: np.ndarray | None = None,
    coarsen_extreme_tails: bool = False,
    ordered_orientations: bool = False,
    q_star=None,
    q_partner=None,
    q_wedge=None,
) -> PPMBuildResult:
    """Fit two-point PPM pole parameters from precomputed W(0) and W(probe).

    Model-agnostic over the pole-fit ansatz: the same algebra serves
    both Godby-Needs (purely imaginary ``probe_omega = i·ωp``) and
    Hybertsen-Louie (real ``probe_omega = Ω`` above all transitions).

    ``ordered_orientations=True`` (GN on a measured-broken-TR deck) splits
    the probe into Hermitian and anti-Hermitian halves and returns the odd
    residue ``D`` in ``B_odd_q``.

    All input arrays are flat-q (nq, μ, μ).  Returns PPMBuildResult with
    B_q, Omega_q, valid_mask_q sharded as P(None, 'x', 'y').

    ``n_mu_logical`` (REQUIRED, = ``meta.n_rmu``): logical centroid
    count.  The fitted tensors keep the padded extent, but pad modes are
    born DEAD (Ω = B = 0, valid = False) and the ``unfulfilled``
    fraction counts logical modes only — see ``fit_gn_ppm_from_wc_pair``.
    ``q_neg_index`` is the public symmetry service's canonical full-grid
    involution, passed from the driver rather than rebuilt at this layer.  It
    is required only when ``coarsen_extreme_tails`` is true.

    ``coarsen_extreme_tails=True`` is the fixed user-ruled GN-only policy.  It
    preserves affected ``Wc(0)`` elements but changes their finite poles and
    ``1/z^2`` moments, so it is not strict BGW pole parity.  It is explicit
    here because the same algebra also fits HL poles, whose real-axis
    two-point model must not silently inherit this GN policy.
    """
    import time as _t
    z = complex(probe_omega)
    t0 = _t.perf_counter()

    Wc0_q = W0_q - V_q
    Wci_q = Wprobe_q - V_q
    fit = fit_gn_ppm_from_wc_pair(
         Wc0_q, Wci_q, z, fallback_omega=float(fallback_omega),
         n_mu_logical=int(n_mu_logical),
         mu_active_mask=mu_active_mask,
         q_neg_index=q_neg_index,
         q_star=q_star, q_partner=q_partner,
         coarsen_extreme_tails=bool(coarsen_extreme_tails),
         ordered_orientations=bool(ordered_orientations),
         print_fn=print_fn if print_fn is not None else print)

    q_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    Omega = jax.lax.with_sharding_constraint(
        jnp.asarray(fit.omega_qmunu), q_shard)
    B = jax.lax.with_sharding_constraint(jnp.asarray(fit.B_qmunu), q_shard)
    B_odd = (None if fit.B_odd_qmunu is None else
             jax.lax.with_sharding_constraint(
                 jnp.asarray(fit.B_odd_qmunu), q_shard))
    valid_mask = jax.lax.with_sharding_constraint(
        jnp.asarray(fit.valid_qmunu), q_shard)
    Wc0_q = jax.lax.with_sharding_constraint(Wc0_q, q_shard)
    t1 = _t.perf_counter()

    probe_hermiticity_residual = None
    odd_even_residue_ratio = None
    if B_odd is not None:
        d_max = float(jax.device_get(jnp.max(jnp.abs(B_odd))))
        b_max = float(jax.device_get(jnp.max(jnp.abs(B))))
        odd_even_residue_ratio = d_max / b_max if b_max > 0.0 else d_max
        probe_hermiticity_residual = float(jax.device_get(
            _probe_antihermitian_residual(Wprobe_q)))
        if print_fn is not None:
            print_fn(
                f"  {model_label} ORDERED residues (measured-broken-TR deck): "
                f"R± = B ± D; max|D|/max|B| = "
                f"{odd_even_residue_ratio:.3e}; W(iω_p) probe Hermiticity "
                f"residual = {probe_hermiticity_residual:.3e}.")

    # Deck-level ε_H measurement (env-gated observability; channel-
    # hermiticity memo §1.3/§3.5): the Laplace-family symmetry diagnostics
    # (σ_R symmetric / σ_I antisymmetric, check L1) hold only to the PPM
    # ω_p in PPMBuildResult historically meant the imaginary-axis magnitude;
    # carry the probe magnitude there for diagnostics.  Downstream Σ kernels
    # consume only B_q, Omega_q (the *fitted* pole frequency), so the probe
    # magnitude is for logging / restart provenance only.
    probe_mag = float(abs(z))

    if print_fn is not None:
        kind = "iωp" if abs(z.real) < 1.0e-12 else "Ω"
        print_fn(
            f"  {model_label} fit: {t1-t0:.2f}s, {kind}={probe_mag:.4f} Ry, "
            f"unfulfilled={100.0 * fit.unfulfilled_fraction:.2f}%")
        print_fn(
            f"  {model_label} pole census: valid={fit.n_valid}, "
            f"Omega=[{fit.omega_min_raw:.8e}, {fit.omega_max_raw:.8e}] Ry, "
            f"min |Wc(0)-Wc(probe)|/max(|Wc(0)|,|Wc(probe)|)="
            f"{fit.pair_relative_separation_min:.8e}")
        if coarsen_extreme_tails:
            budget = fit.n_valid // GN_PPM_EXTREME_TAIL_DIVISOR
            print_fn(
                "  GN user-ruled fitted-pole tail policy: "
                f"low={fit.n_tail_low}/{budget}, "
                f"high={fit.n_tail_high}/{budget}; "
                f"[{fit.omega_min_raw:.8e}, {fit.omega_max_raw:.8e}] -> "
                f"[{fit.omega_min_after:.8e}, {fit.omega_max_after:.8e}] Ry; "
                f"anchor={fit.tail_anchor_omega:.8e} Ry; boundary key groups "
                "are atomic before orbit closure; physical partner orbits "
                "remain unsplit and closure may further undershoot"
            )
            print_fn(
                "  GN tail semantics: affected Wc(0) is preserved by "
                "B'=-Wc(0)*Omega'/2; the 1/z^2 moment changes; this is not "
                "BGW finite-pole parity; exact panes remain downstream."
            )

    return PPMBuildResult(
        omega_p=probe_mag,
        Wc0_q=Wc0_q,
        B_q=B,
        Omega_q=Omega,
        valid_mask_q=valid_mask,
        unfulfilled_fraction=fit.unfulfilled_fraction,
        n_nodes_static=n_nodes_static,
        B_odd_q=B_odd,
        probe_hermiticity_residual=probe_hermiticity_residual,
        odd_even_residue_ratio=odd_even_residue_ratio,
        q_wedge=q_wedge,
    )


# ---------------------------------------------------------------------------
#  The held pole-fit plan and the screening update.  The GW driver
#  (``ppm_pipeline.compute_ppm_sigma_pipeline``) and the TD-aGW refresh
#  (docs/theory/td_agw_dynamical.md Eq. 6) run ONE fit: :func:`fit_pole_model`.
# ---------------------------------------------------------------------------

def pole_fit_plan(V_q, probe_omega, mesh_xy, *, n_mu_logical: int, fallback_omega: float,
                  ordered: bool, coarsen_extreme_tails: bool, q_neg_index=None,
                  mu_active_mask=None, wedge=None, n_nodes_static: int = 0,
                  model_label: str = "GN-PPM", solve_extent: int | None = None) -> dict:
    """Everything the two-point fit reads except its two W samples, held for a run.

    Parameters
    ----------
    V_q : (nq, mu, mu) at ``P(None, 'x', 'y')``, Ry — the bare V on the fit's q
        set (the wedge rows for a wedge fit; the full zone otherwise).
    probe_omega : the second support, ``1j * omega_p`` (GN) or ``omega_p`` (HL), Ry.
    n_mu_logical, fallback_omega, ordered, coarsen_extreme_tails, q_neg_index,
    mu_active_mask, n_nodes_static, model_label : :func:`fit_ppm`'s options
        (``ordered`` = ``ordered_orientations``: a measured-broken-TR deck).
    wedge : ``dict(q_star=, q_partner=, q_wedge=)`` of a wedge fit, None for the
        full zone (a general TD drive breaks the point group: full zone).
    solve_extent : the centroid extent the Dyson solves of
        :func:`screening_update` run at, ``meta.mu_solve_extent`` (the whole
        packed carrier when the run's pads interleave per shard); default
        ``n_mu_logical`` (a canonical, suffix-padded carrier).

    Returns a plain dict: ``V_q``, ``probe_omega``, ``mesh_xy``, the centroid
    carrier ``axis`` of the Dyson solves (:func:`screening_update`) and the
    fit's keyword arguments ``fit``.
    """
    from runtime.padding import padded_mu_axis
    return dict(
        V_q=V_q, probe_omega=complex(probe_omega), mesh_xy=mesh_xy,
        axis=padded_mu_axis(int(n_mu_logical if solve_extent is None else solve_extent),
                            mesh_xy),
        fit=dict(fallback_omega=float(fallback_omega), n_nodes_static=int(n_nodes_static),
                 model_label=str(model_label), n_mu_logical=int(n_mu_logical),
                 mu_active_mask=mu_active_mask, q_neg_index=q_neg_index,
                 coarsen_extreme_tails=bool(coarsen_extreme_tails),
                 ordered_orientations=bool(ordered), **dict(wedge or {})))


def fit_pole_model(W_static_q, W_probe_q, plan: dict, *, print_fn=None) -> PPMBuildResult:
    """The pole model of W sampled at the plan's two supports, 0 and ``probe_omega``.

    :func:`fit_ppm` on ``(W_static_q, W_probe_q, plan['V_q'])``: per element
    ``(q, mu, nu)`` the pole Omega, the even residue B = (R_+ + R_-)/2 and, on an
    ordered plan, the odd residue D = (R_+ - R_-)/2
    (docs/theory/gn-ppm-nonhermitian.md §3).  Shapes and sharding are the
    inputs': (nq, mu, mu) at ``P(None, 'x', 'y')``, Ry.
    """
    return fit_ppm(W_static_q, W_probe_q, plan["V_q"], plan["probe_omega"], plan["mesh_xy"],
                   print_fn=print_fn, **plan["fit"])


def screening_update(W0_at_supports, dchi_at_supports, plan: dict, *, print_fn=None):
    """W(z_s, t) = [1 - W0(z_s) dchi0(z_s, t)]^-1 W0(z_s) at both supports, then the pole fit.

    docs/theory/td_agw_dynamical.md Eq. (6): ``gw.w_isdf.solve_w``'s Dyson
    solve with the run's W0 in the place of v (``meta=None``, ``pref = 1``:
    dchi0 is physical, Ry^-1), followed by :func:`fit_pole_model`, the fit the
    GW driver runs.  No file is read or written; nothing is re-planned.

    Parameters
    ----------
    W0_at_supports : (W0(0), W0(probe)) — the held screened interactions, each
        (nq, mu, mu) at ``P(None, 'x', 'y')``, Ry, on the plan's q set.
    dchi_at_supports : (dchi0(0), dchi0(probe)), same layout, Ry^-1.  DONATED
        to the solves.  With V in W0's place and the full chi0 in dchi0's, this
        is the GW run's own screening at the two supports.
    plan : :func:`pole_fit_plan`.

    Returns
    -------
    (poles, W) : the :class:`PPMBuildResult` (Omega_q, B_q, B_odd_q, valid_mask_q,
        Wc0_q) and the tuple (W(0, t), W(probe, t)).
    """
    from .w_isdf import solve_w
    W0s, dchis = tuple(W0_at_supports), tuple(dchi_at_supports)
    if len(W0s) != 2 or len(dchis) != 2:
        raise ValueError("GATE screening_update_supports: got "
                         f"{len(W0s)} W0 and {len(dchis)} dchi0 samples; want the two "
                         "supports 0 and probe_omega; fix: pass one of each, in that order")
    W = tuple(solve_w(w0, dchi, None, plan["mesh_xy"], pref=1.0, axis=plan["axis"])
              for w0, dchi in zip(W0s, dchis))
    return fit_pole_model(W[0], W[1], plan, print_fn=print_fn), W


# ---------------------------------------------------------------------------
#  Sigma convolution — the device-side τ loop.  Its host-side counterpart
#  (window construction) lives in ppm_windows; the two halves share no state
#  beyond the window list itself.
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Sigma band-window mesh padding
# ---------------------------------------------------------------------------

def pad_sigma_window(psi_proj_xr, psi_proj_yn, mesh_xy):
    """Zero-pad both projected-band operands to one square Sigma carrier.

    The returned third value is the :class:`runtime.padding.PaddedAxis`
    receipt.  Production calls the same owner directly in ``gw.mpa.sigma``;
    this small adapter remains the projection primitive's focused test seam.
    """
    from runtime.padding import pad_to_axis, padded_axis

    tag = padded_axis(
        int(psi_proj_xr.shape[1]), mesh_xy, name="Sigma band window",
        specs=(
            (P(None, "x"), 1),
            (P(None, None, None, "y"), 3),
        ))
    return (
        pad_to_axis(psi_proj_xr, tag, axis=1),
        pad_to_axis(psi_proj_yn, tag, axis=3),
        tag,
    )


def sigma_band_axis(nb_proj: int, mesh_xy, *, ansatz: str):
    """Return the square mesh carrier receipt for a Sigma band window.

    Both matrix axes share one tag so Hermitisation remains square even on a
    rectangular mesh.  The runtime owner derives the least common compatible
    divisor from the two actual specs.
    """
    from runtime.padding import padded_axis

    return padded_axis(
        int(nb_proj), mesh_xy, name=f"{ansatz} Sigma band window",
        specs=(
            (P(None, "x"), 1),
            (P(None, None, None, "y"), 3),
        ))


def _compute_invalid_static_sigma(
    wfns,
    Wc0_q: jax.Array,
    invalid_mask: jax.Array,
    meta,
    mesh_xy: Mesh,
    occupation_state=None,
    q_wedge=None,
    coulomb_hole: bool = True,
) -> np.ndarray:
    """Static-COHSEX Σ for the invalid PPM poles (BGW ``invalid_gpp_mode=3``).

    BGW's default treatment of a pole with fitted ``Ω² < 0`` sets
    ``ω̃ → 1/TOL_ZERO`` (mtxel_cor.f90:788/838), which is the Ω→∞ limit of
    the full dynamical pole: for that mode's ``W_static = W^c(0)·mask``,

        occupied   l:  ssx → −I_ε,  sch → −½·I_ε   ⇒  −W_static + ½·W_static
        unoccupied l:                sch → −½·I_ε   ⇒            + ½·W_static

    i.e. the mode is treated within static COHSEX: a screened-exchange
    term over occupied states plus the Coulomb-hole over the full RI
    window.  (Ω→∞ can NOT be pushed through the τ-integral — ``B ∝ Ω``
    makes ``B·e^{−iΩτ}`` non-integrable — hence this analytic,
    ω-independent term instead.)  Equivalently, per intermediate state:
    occ → −½·W^c(0) (= B/Ω), unocc → +½·W^c(0) — the exact Ω→∞ limit of
    the two-branch pole sum ``B/(ω−E_l∓Ω)``.

    The static Σ kernels (``gw.cohsex_sigma``: the Σ τ node at τ = 0) with the
    masked static ``W^c(0)`` as the interaction:

        Σ_static = sigma_sx(G_occ, W_static) + sigma_coh(W_static − 0)
                 = −⟨G_occ·W_static⟩ + ½·⟨G_RI·W_static⟩

    μ-pad safety is inherited from ``invalid_mask`` (pad modes are born
    dead at the fit, so they are never flagged invalid and ``W_static`` is
    exactly zero there).

    Returns the replicated host tensor (nk, nb_sigma, nb_sigma) in Ry,
    to be added to Σ_c at EVERY ω (the term is ω-independent).

    ``occupation_state`` is the iteration's
    :class:`gw.efermi.OccupationState` (``None`` ⇒ the integer projector,
    bit-exact insulating behaviour).  It matters here for the same reason
    it matters in ``cohsex_sigma``: the screened-exchange half of this
    term runs over the OCCUPIED manifold, so on a metal the Fermi-shell
    bands must enter with their fractional weights.

    ``coulomb_hole = False`` returns the screened-exchange half alone (the
    population-dependent part of a split band sum, :class:`InternalBands`).
    """
    from .cohsex_sigma import build_Gij
    Gij = build_Gij(meta, mesh_xy, occupation_state)
    sigma_sx, sigma_coh, W_static = _invalid_static_kernels(
        wfns, Wc0_q, invalid_mask, meta, mesh_xy, q_wedge)
    sx = _band_sigma_host(sigma_sx(wfns, Gij, W_static), wfns)
    if not coulomb_hole:
        return sx
    coh = _band_sigma_host(sigma_coh(wfns, W_static, None), wfns)
    return sx + coh


def _invalid_static_kernels(wfns, Wc0_q, invalid_mask, meta, mesh_xy, q_wedge):
    """The static Σ kernels (``cohsex_sigma``'s SX and COH) and the masked ``W^c(0)``
    as a Hermitian interaction on the fit's q wedge (conj rule), or the full zone."""
    import dataclasses as _dc
    from .cohsex_sigma import _face_kwargs, _make_cohsex_kernels
    sigma_sx, sigma_coh = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, int(meta.nk_tot), **_face_kwargs(wfns))
    W_static = jnp.where(
        jnp.asarray(invalid_mask, dtype=bool),
        jnp.asarray(Wc0_q, dtype=jnp.complex128),
        jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
    )
    if q_wedge is not None:
        W_static = _dc.replace(q_wedge, values=W_static, trs_rule="conj")
    return sigma_sx, sigma_coh, W_static


def _band_sigma_host(sigma_kij, wfns):
    """A static Σ ``(nk, carrier, carrier)`` on the host, cut to the QP window."""
    from common.collectives import gather_to_host
    nb = int(wfns.slices.nb_sigma)
    return np.asarray(gather_to_host(sigma_kij), dtype=np.complex128)[:, :nb, :nb]


def _invalid_static_coh_by_bracket(
    wfns,
    Wc0_q: jax.Array,
    invalid_mask: jax.Array,
    meta,
    mesh_xy: Mesh,
    brackets,
    q_wedge=None,
) -> np.ndarray:
    """The static-limit term's OWN band-count series, one point per bracket.

    WHY THIS EXISTS — THE ONE PLACE THE PPM-ONLY GUARD CANNOT REACH.
    ``gw.sigma_dispatch``'s guard keeps the band extrapolation away from a
    static Coulomb hole because the ``1/N → 0`` limit is wrong for one (it
    ANTI-converges: 94.9 → 288.2 meV MAE as nband goes 60 → 124, overshooting
    BerkeleyGW's exact closure by ~340 meV — ``gw.band_extrapolation``'s module
    docstring owns that measurement).  That guard is per-``compute_mode``, and
    ``ppm_invalid_mode = "static_limit"`` — the SHIPPING DEFAULT — puts a
    static Coulomb hole inside a Σ whose ``compute_mode`` genuinely IS
    ``gn_ppm``.  The guard cannot see it because it is per-MODE, one logical
    ISDF mode at a time, underneath the seam the guard checks.

    WHAT IS AND IS NOT BAND-COUNT DEPENDENT.  The static-limit term is
    ``Σ_static = −⟨G_occ·W_static⟩ + ½⟨G_RI·W_static⟩``.  The first half runs
    over OCCUPIED states through ``Gij`` and is band-count independent for any
    ``nband ≥ nelec``.  The second — the Coulomb hole — runs over ``s.full``
    with no occupation projector, so it carries exactly the slowly convergent
    unoccupied tail this module exists to worry about.  Only the second half is
    measured here, and that is not a simplification: the extrapolation is an
    AFFINE estimator with ``sum(c) == 1``, so any band-count-INDEPENDENT part
    passes through it unchanged and cancels identically out of
    ``S_inf − S(N₃)``.  The occupied half therefore cannot contribute to the
    diagnostic even in principle.

    IT IS FREE, WHICH IS WHY IT CAN BE ON BY DEFAULT.  ``G_RI`` is a plain band
    sum, so the brackets PARTITION it the same way they partition Σ_c, and
    ``n_brk`` contractions over disjoint sub-ranges cost the same total flops
    as the one contraction over the whole range that runs anyway.  The price is
    one extra pass of the COH channel, not ``n_brk`` extra passes.

    Returns the replicated host tensor ``(n_brk, nk, nb_sigma, nb_sigma)`` in
    Ry, holding the DISJOINT per-bracket contributions — the caller cumulates
    them to get Σ_COH at each of the plan's band counts, in the same order and
    by the same rule that turns the Σ_c brackets into band counts.
    """
    _, sigma_coh, W_static = _invalid_static_kernels(
        wfns, Wc0_q, invalid_mask, meta, mesh_xy, q_wedge)
    out = [_band_sigma_host(sigma_coh(wfns, W_static, None,
                                      ri_bands=(int(lo), int(hi))), wfns)
           for lo, hi in brackets]
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
#  Top-level sigma driver
# ---------------------------------------------------------------------------

def _resolve_ppm_band_plan(wfns, meta, plan, *, where):
    """One band-bracket validation seam for both transition executors."""
    from .band_extrapolation import (
        assert_brackets_match_ols_abscissae, trivial_plan)

    s = wfns.slices
    if plan is None:
        plan = trivial_plan(
            int(s.nb_sigma_sum), int(s.b2 - s.b0),
            int(meta.b_id_4_sigma_user or s.b4) - int(s.b0))
    assert_brackets_match_ols_abscissae(
        plan, s, meta=meta, where=where)
    return plan




@jax.jit
def _ppm_as_one_pole_store_fields(
    state: _SigmaPhysicsState,
    B_odd_q=None,
):
    """Policy-resolved PPM poles in the MPA ``(p,q,mu,nu)`` convention.

    Both ansatzes use

    ``Wc(z) = 2*Omega*B/(z**2 - Omega**2)`` and therefore
    ``Wc(0) = -2*B/Omega``.  No residue rescaling or sign flip is needed.
    ``state.B_mask`` has already applied ``ppm_invalid_mode``; absent modes
    are encoded by the MPA store's ordinary zero-residue convention and their
    frequency is zeroed as well.  An ordered GN fit's ``B_odd_q=D`` is masked
    by the same policy and stored beside ``B`` so the shared executor consumes
    ``R_+=B+D`` on conduction rows and ``R_-=B-D`` on valence rows.  GN/HL
    poles retained here are positive real frequencies, i.e. causal MPA poles
    with zero fitted width.
    """
    live = jnp.asarray(state.B_mask, dtype=bool)
    if B_odd_q is not None and tuple(B_odd_q.shape) != tuple(state.B_corr.shape):
        raise ValueError(
            "ordered PPM B_odd_q must match B_q; got "
            f"{tuple(B_odd_q.shape)} and {tuple(state.B_corr.shape)}")
    Omega = jnp.where(live, state.Omega_abs, 0.0).astype(jnp.complex128)
    B = jnp.where(live, state.B_corr, 0.0 + 0.0j).astype(jnp.complex128)
    D = (None if B_odd_q is None else
         jnp.where(live, jnp.asarray(B_odd_q), 0.0 + 0.0j)
         .astype(jnp.complex128)[None, ...])
    return Omega[None, ...], B[None, ...], D


def _add_static_ppm_term(
        sigma_c_kij, sigma_static_host, mesh_xy, band_axis: PaddedAxis):
    """Add one static invalid-pole term to every cumulative band count."""
    from common.collectives import device_put_process_local
    from runtime.padding import pad_square

    static_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    sigma_static_host = pad_square(
        jnp.asarray(sigma_static_host, dtype=jnp.complex128), band_axis)
    static = device_put_process_local(
        np.asarray(sigma_static_host, dtype=np.complex128), static_sharding)
    return replace(sigma_c_kij, term=static)


@lru_cache(maxsize=8)
def _add_static_term_fn(sharding, donate):
    """``sigma + term``, an ω-independent term broadcast over omega; ``donate``: in the
    point's buffer."""
    return jax.jit(lambda sigma, term: sigma + (term if term.ndim == sigma.ndim
                                                else term[None, ...]),
                   out_shardings=sharding,
                   donate_argnums=(0,) if donate else ())


def compute_sigma_c_ppm_omega_grid(
    wfns,
    ppm,
    meta,
    mesh_xy: Mesh,
    *,
    ppm_cfg: PPMConfig,
    sigma_cfg: DynamicSigmaConfig,
    mpa_cfg,
    ansatz: str,
    screening_diagrams,
    occupation_state=None,
    plan: 'BandBracketPlan | None' = None,
    fixed_quadrature_session=None,
    request_for=None,
    omega_grid_ry: np.ndarray | None = None,
    populations=None,
    internal: 'InternalBands | None' = None,
    keep_branches=False,
    print_fn=print,
    particle_weights=None,
) -> SigmaOmegaResult:
    """Compute GN/HL-PPM Sigma_c through the shared MPA dynamic route.

    ``populations`` ``(nk, nb_full)`` real, on ``wfns``' bands (``slices.full``):
    the band populations of a non-equilibrium state of the gapped system
    (TD-aGW's instantaneous window, docs/theory/td_agw_dynamical.md Eq. 3; the
    frozen bands carry their step occupations).  The causal branches then
    carry the weights f (valence) and 1 - f (conduction) on unclipped
    energies, as ``gw.mpa.sigma._branches`` does with an occupation state, and
    the invalid-pole static term's occupied projector is diag(f).  ``None`` is
    the incumbent 0/1 step split, bit-for-bit.  The step table ``wfns.occ``
    still fixes the gap check and the omega frame.

    ``internal`` (:class:`InternalBands`, with ``populations`` only) restricts
    the intermediate-state band sum: both causal branches and the invalid-pole
    term's occupied projector run over its bands, and the windows are planned
    on their energies.  ``None`` is the whole sum.

    ``particle_weights`` (``(nk, nb_full)`` real, with ``populations`` only): the
    conduction branch's weights where they are not 1 - f (a non-diagonal
    internal line's, :func:`_window_lines`); ``populations`` stays the
    valence weight and the occupied projector's.  None is 1 - f.

    The PPM fit remains the two-point algebra in :func:`fit_ppm`.  This stage
    applies the established invalid-pole policy, hands that result to the
    executor in memory as a one-pole source, and then delegates window construction,
    denominator-box rules, cache lookup, pole batching, the tau executor and
    omega accumulation to :func:`gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`.

    ``plan`` retains the PPM band-convergence contract: disjoint brackets are
    evaluated by the shared spatial kernel and returned as cumulative band
    counts.  The optional static-COHSEX invalid-pole term remains separate
    from the dynamic store and is added exactly once to each cumulative count.

    ``request_for(efermi_ry)`` returns the on-shell request at this body's
    omega reference (``dynamic_sigma.sigma_on_shell_request``); the reference
    is the bundle's VBM/midgap resolved here.  ``omega_grid_ry`` in its place
    evaluates Sigma_c on that grid (``sigma_omega_grid_debug``).

    ``keep_branches``: the result also carries each causal branch's dynamic
    body from the same sweep (``SigmaOmegaResult.branches``; the invalid-pole
    static term is not in either, so Sigma_c = cond + val + static term).
    False is the incumbent call, bit for bit.
    """
    if (request_for is None) == (omega_grid_ry is None):
        raise ValueError("compute_sigma_c_ppm_omega_grid: pass request_for (on shell) "
                         "or omega_grid_ry (the grid), not both")
    from .mpa.sigma import MemoryPoleSource, compute_sigma_c_mpa_omega_grid

    s = wfns.slices
    plan = _resolve_ppm_band_plan(
        wfns, meta, plan, where="ppm_sigma shared MPA bracket partition")

    enk_full = wfns.enk[:, s.full]
    occ_full = wfns.occ[:, s.full]
    if omega_grid_ry is not None:
        omega_grid_ry = np.asarray(omega_grid_ry, dtype=np.float64)
        if omega_grid_ry.ndim != 1 or not omega_grid_ry.size:
            raise ValueError("omega_grid_ry must be a nonempty vector")
    if int(meta.nk_tot) != int(enk_full.shape[0]):
        raise ValueError(
            "enk_full shape mismatch: expected first dim "
            f"{int(meta.nk_tot)}, got {enk_full.shape[0]}")

    invalid_mode = str(ppm_cfg.invalid_mode).strip().lower()
    if invalid_mode == "imaginary":
        raise NotImplementedError(
            "ppm_invalid_mode='imaginary' (BGW mode 1) needs a "
            "complex-Omega path.")
    if invalid_mode not in (
            "zero", "skip", "2ry", "static_limit", "infinity"):
        raise ValueError(
            "ppm_invalid_mode must be zero/skip/2ry/static_limit/infinity; "
            f"got {invalid_mode!r}")
    keep_invalid = invalid_mode == "2ry"
    invalid_static = invalid_mode in ("static_limit", "infinity")

    assert_gapped_occupations_for_ppm(occ_full, print_fn=print_fn)
    valid_mask = ppm.valid_mask_q
    if valid_mask is None:
        valid_mask = jnp.ones(ppm.Omega_q.shape, dtype=bool)
    with timing.section("sigma.state"):
        state = _prepare_sigma_state(
            enk_full, occ_full, ppm.B_q, ppm.Omega_q, valid_mask,
            jnp.asarray(sigma_cfg.fermi_reference == "midgap", dtype=bool),
            jnp.asarray(keep_invalid, dtype=bool))

    regularization_width_ry = (
        float(sigma_cfg.regularization_ev) / RYD_TO_EV)
    ansatz_name = str(getattr(ansatz, "value", ansatz)).strip().lower()
    resolved_xi = resolve_sigma_regularization(
        requested_ry=regularization_width_ry, ansatz=ansatz_name)
    print_fn(resolved_xi.describe())
    regularization_width_ry = resolved_xi.resolved_ry

    n_total_modes, n_invalid = map(
        int, jax.device_get((state.n_total_modes, state.n_invalid)))
    stars = None if ppm.q_wedge is None else ppm.q_wedge.star_sizes()
    if n_invalid:
        n_invalid_q = np.asarray(jax.device_get(
            jnp.sum(state.invalid_mask, axis=(1, 2), dtype=jnp.int64)))
        if stars is not None:
            # The census of the full-zone field: each wedge row counts once
            # per q of its star.
            n_total_q = np.asarray(jax.device_get(
                jnp.sum(state.B_mask | state.invalid_mask, axis=(1, 2),
                        dtype=jnp.int64)))
            n_invalid = int(np.dot(n_invalid_q, stars))
            n_total_modes = int(np.dot(n_total_q, stars))
        print_fn(
            f"  PPM invalid modes: {n_invalid}/{n_total_modes} "
            f"({100.0 * n_invalid / max(n_total_modes, 1):.2f}%)")
        print_fn(
            f"  PPM invalid modes per {'q' if stars is None else 'wedge q'}: "
            f"min={int(n_invalid_q.min())} max={int(n_invalid_q.max())} "
            f"counts={np.array2string(n_invalid_q, max_line_width=100, threshold=64)}")

    sigma_static_host = None
    static_coh_at_counts = None
    if internal is not None and populations is None:
        raise ValueError(
            "GATE ppm_sigma_internal: got an internal-band restriction without band "
            "populations; want populations (the split is of a population sum); fix: "
            "pass populations with internal")
    if particle_weights is not None and populations is None:
        raise ValueError(
            "GATE ppm_sigma_particle_weights: got particle weights without band "
            "populations; want both branch weights; fix: pass populations with them")
    if populations is not None:
        if occupation_state is not None:
            raise ValueError(
                "GATE ppm_sigma_populations: got both an occupation state and band "
                "populations; want one occupation source; fix: pass populations only "
                "(a GN/HL-PPM deck carries no metallic occupation state)")
        occupation_state = _population_state(
            populations if internal is None else internal.weights(populations),
            meta, enk_full.shape)
    if invalid_static and n_invalid:
        sigma_static_host = _compute_invalid_static_sigma(
            wfns, ppm.Wc0_q, state.invalid_mask, meta, mesh_xy,
            occupation_state=occupation_state, q_wedge=ppm.q_wedge,
            coulomb_hole=internal is None or internal.coulomb_hole)
        print_fn(
            "  PPM invalid modes -> static COHSEX: max|Sigma_static| = "
            f"{float(np.max(np.abs(sigma_static_host))) * RYD_TO_EV:.4f} eV")
        if plan.n_brackets > 1:
            static_coh_at_counts = np.cumsum(
                _invalid_static_coh_by_bracket(
                    wfns, ppm.Wc0_q, state.invalid_mask, meta, mesh_xy,
                    plan.bounds, q_wedge=ppm.q_wedge),
                axis=0)

    Omega_p, B_p, B_odd_p = _ppm_as_one_pole_store_fields(
        state, ppm.B_odd_q)
    if getattr(meta, 'mu_basis', None) is not None:
        # The store keeps the canonical centroid order (grid-agnostic).
        Omega_p, B_p = (meta.mu_basis.unpack_operator(Omega_p),
                        meta.mu_basis.unpack_operator(B_p))
        if B_odd_p is not None:
            B_odd_p = meta.mu_basis.unpack_operator(B_odd_p)
    diagram_value = str(getattr(
        screening_diagrams, "value", screening_diagrams))
    # In memory, no store (owner 2026-09-23): the executor reads the same
    # sharded fields a store round trip would have returned.
    poles = MemoryPoleSource(
        Omega_p, B_p, B_odd_p,
        n_mu_logical=int(meta.n_rmu), mesh_xy=mesh_xy, q_wedge=ppm.q_wedge,
        provenance={
            "fit_protocol": "two_point_ppm",
            "pole_model": ansatz_name,
            "ppm_invalid_mode": invalid_mode,
            "screening_diagrams": diagram_value,
            "probe_frequency_ry": float(ppm.omega_p),
            "unfulfilled_fraction": float(ppm.unfulfilled_fraction),
        })
    del Omega_p, B_p, B_odd_p
    print_fn(
        f"  {ansatz_name} fit -> in-memory one-pole source (no store); "
        f"invalid policy={invalid_mode}")

    efermi_ry = float(jax.device_get(state.efermi))
    # The windows are planned on the states' own energies (on the grid, on it).
    request = None if request_for is None else request_for(efermi_ry)
    omega_req = omega_grid_ry if request is None else request.frequencies_ry
    if populations is None:
        branches = branches_for_omega_grid(
            omega_req,
            E_cond=state.E_cond,
            H_val=state.H_val,
            cond_mask=state.cond_mask,
            val_mask=state.val_mask)
    else:
        branches = _population_branches(omega_req, enk_full, state.efermi, populations,
                                        internal=None if internal is None else internal.mask,
                                        particle_weights=particle_weights)
    result = compute_sigma_c_mpa_omega_grid(
        wfns, poles, meta, mesh_xy,
        request=request,
        omega_grid_ry=omega_grid_ry,
        efermi_ry=efermi_ry,
        regularization_width_ry=regularization_width_ry,
        edge_factor=float(sigma_cfg.window_edge_factor),
        quadrature_eps=float(sigma_cfg.quadrature_eps),
        omega_grid_step_ry=float(sigma_cfg.omega_step_ev) / RYD_TO_EV,
        pole_batch_size=int(mpa_cfg.pole_batch_size),
        expected_screening_diagrams=screening_diagrams,
        sigma_branches=branches,
        band_brackets=plan.bounds,
        band_counts=plan.counts,
        fixed_quadrature_session=fixed_quadrature_session,
        # The same fitted box rules as full frequency (owner 2026-09-22): the
        # analytic fixed-height line rule needed 416 vs 206 (cond:resonant) and
        # 320 vs 178 (val:resonant) nodes on comparable zero-damping boxes
        # (CrI3 16x16 GN-PPM 824 tau nodes vs TaAs MPA).
        analytic_line=False,
        # No D=0 twin on the GN/HL route: the odd-reference Sigma was a
        # diagnostic-only second sweep, removed by owner decision 2026-09-23.
        odd_reference=False,
        keep_branches=keep_branches,
        print_fn=print_fn)
    sigma_c_kij = result.sigma_c_kij
    if sigma_static_host is not None:
        sigma_c_kij = _add_static_ppm_term(
            sigma_c_kij, sigma_static_host, mesh_xy, result.band_axis)
    return SigmaOmegaResult(
        sigma_c_kij=sigma_c_kij,
        band_axis=result.band_axis,
        sigma_c_odd_kij=result.sigma_c_odd_kij,
        band_counts=result.band_counts,
        static_coh_at_counts=static_coh_at_counts,
        odd_even_residue_ratio=ppm.odd_even_residue_ratio,
        efermi_ry=efermi_ry,
        omega_ry=result.omega_ry,
        omega_ev=result.omega_ev,
        branches=result.branches)


# ---------------------------------------------------------------------------
#  Band populations of a non-equilibrium state, and the Σ τ sweep on a window
#  (TD-aGW, docs/theory/td_agw_dynamical.md §1.2-1.4 and §9 item 5)
# ---------------------------------------------------------------------------

def _population_state(populations, meta, shape):
    """The occupation projector's duck-typed state for band populations ``f`` (nk, nb).

    ``gw.cohsex_sigma.build_Gij`` reads ``f_kn`` and ``n_electrons`` (capacity
    weighted); the count is the populations' own, so the projector refuses a
    population outside the Sigma window (its Hartree-density guard).
    """
    from types import SimpleNamespace
    from .cohsex_sigma import _spin_capacity
    f = np.asarray(jax.device_get(populations), dtype=np.float64)
    if f.shape != tuple(shape):
        raise ValueError(f"GATE ppm_sigma_population_shape: got {f.shape}; want {tuple(shape)} "
                         "(the wfns' full band table); fix: pass every band's population")
    return SimpleNamespace(f_kn=f, n_electrons=_spin_capacity(meta) * float(np.sum(f))
                           / float(meta.nk_tot))


def _population_branches(omega_ry, enk_full, efermi, populations, internal=None,
                         particle_weights=None):
    """Causal branches of band populations: valence weight f, conduction 1 - f, energies unclipped.

    ``gw.mpa.sigma._branches``' occupation-state form on the PPM frame
    ``efermi``: a hole in a valence band sits on the conduction branch at its
    own (negative) energy, which the step split's clipped ``E_cond`` could not
    represent.  ``internal`` (nk, nb) bool keeps only those intermediate states
    in both branches (:class:`InternalBands`); None keeps every band.
    ``particle_weights`` (nk, nb) real: the conduction (particle) branch's
    weights where they are not 1 - f — a non-diagonal internal line's
    (:func:`_window_lines`); None is 1 - f.
    A branch keeps a band where its own weight is nonzero.
    """
    f = jnp.asarray(populations, dtype=jnp.float64)
    cw = 1.0 - f if particle_weights is None else jnp.asarray(particle_weights, jnp.float64)
    energy = jnp.asarray(enk_full, dtype=jnp.float64) - efermi
    if internal is None:
        return branches_for_omega_grid(
            omega_ry, E_cond=energy, H_val=-energy, cond_mask=(cw != 0.0), val_mask=(f != 0.0),
            cond_weight=cw, val_weight=f)
    keep = jnp.asarray(internal, dtype=bool)
    return branches_for_omega_grid(
        omega_ry, E_cond=energy, H_val=-energy, cond_mask=(cw != 0.0) & keep,
        val_mask=(f != 0.0) & keep, cond_weight=jnp.where(keep, cw, 0.0),
        val_weight=jnp.where(keep, f, 0.0))


@dataclass(frozen=True)
class InternalBands:
    """A restriction of the Σ sweep's intermediate-state band sum (TD-aGW, td_agw_dynamical.md §1.6).

    At fixed screening each causal branch is linear in the internal weights
    (particle 1 - f_n, hole f_n), and so is the invalid-pole term's screened
    exchange (occupied projector diag f), so Σ[f] = Σ[internal] + Σ[rest]
    exactly.  ``mask`` (nk, nb_full) bool selects the internal bands n of both
    branches and of that projector; the windows are planned on their energies
    alone.  ``coulomb_hole``: whether the invalid-pole term's Coulomb hole
    (population independent, over every band) is in this part; exactly one
    part of a split carries it.
    """
    mask: np.ndarray
    coulomb_hole: bool

    def weights(self, populations):
        """The populations seen by the occupied projector: f on the internal bands, 0 elsewhere."""
        f = np.asarray(jax.device_get(populations), np.float64)
        return np.where(np.broadcast_to(np.asarray(self.mask, bool), f.shape), f, 0.0)


def window_sigma_plan(config, *, head=None, session=None, frequencies_abs_ry=None,
                      window_only: bool = False) -> dict:
    """The held plan of :func:`window_sigma_c` from a GW run's config.

    ``head``: the run's ``head_correction.HeadSigmaDiag`` (frozen q -> 0 head,
    evaluated at the instantaneous energies and populations), or None.
    ``session``: the fixed quadrature session dict (``mpa.sigma``'s
    ``fixed_quadrature_session``: rules fitted once and reused while the
    windows stay inside their padded boxes), or None (a one-shot plan, as the
    GW run's).  ``frequencies_abs_ry``: absolute frequencies (Ry) the windows
    are planned on in addition to the window's own energies (the GW run's
    on-shell list reproduces its window plan exactly), or None.
    ``window_only``: the windows are planned on the window's own on-shell
    energies alone.  The rows of the Sigma band prefix outside the window,
    which the sweep evaluates and nobody reads (§9 item 6), take the energy
    of the window's first band at their k, so they add no frequency.
    """
    if window_only and frequencies_abs_ry is not None:
        raise ValueError(
            "GATE window_sigma_plan_frequencies: got window_only with an added frequency "
            "list; want one planning rule; fix: drop frequencies_abs_ry")
    return dict(ppm_cfg=config.ppm, sigma_cfg=config.sigma, mpa_cfg=config.mpa,
                ansatz=config.compute_mode, screening_diagrams=config.screening.diagrams,
                omega_grid_ry=np.asarray(config.omega_grid_ry, np.float64),
                omega_range_ev=np.asarray(config.omega_grid_ev, np.float64),
                out_of_grid=config.sigma.out_of_grid, head=head, session=session,
                frequencies_abs_ry=(None if frequencies_abs_ry is None else
                                    np.asarray(frequencies_abs_ry, np.float64)),
                window_only=bool(window_only))


@lru_cache(maxsize=8)
def _window_rotation_fn(b_lo, n_a, nmu_sharding, mun_sharding, full_nmu, full_mun):
    """psi_A -> psi_A U on both faces, every other band untouched (Eq. 2)."""
    win = slice(int(b_lo), int(b_lo) + int(n_a))

    def rotate(psi_nmu, psi_mun, U, keep):
        keep_n = keep.astype(psi_nmu.dtype)
        rot_n = jnp.einsum("kmsu,kmi->kisu", psi_nmu[:, win], U.astype(psi_nmu.dtype))
        rot_m = jnp.einsum("ksum,kmi->ksui", psi_mun[..., win], U.astype(psi_mun.dtype))
        psi_nmu = (psi_nmu * keep_n[None, :, None, None]).at[:, win].set(rot_n)
        psi_mun = (psi_mun * keep_n[None, None, None, :]).at[..., win].set(rot_m)
        return psi_nmu, psi_mun
    return jax.jit(rotate, out_shardings=(nmu_sharding, mun_sharding))


def _rotate_faces(psi_nmu, psi_mun, U, b_lo, keep):
    fn = _window_rotation_fn(int(b_lo), int(U.shape[-1]), psi_nmu.sharding, psi_mun.sharding,
                             tuple(psi_nmu.shape), tuple(psi_mun.shape))
    return fn(psi_nmu, psi_mun, jnp.asarray(U), jnp.asarray(keep))


#: The internal-line pairings of a non-diagonal line (``window_sigma_c``'s ``pairing``,
#: td_agw_dynamical.md §1.5, Eq. 8c): ``symmetric``, 1/2 {g(e'), C} in both tables, or
#: ``vertex``, each external index with the internal index on its own vertex.
LINE_PAIRINGS = ("symmetric", "vertex")


@lru_cache(maxsize=8)
def _line_insert_fn(b_lo, n_a, n_blocks, n_pad, nmu_spec, mun_spec, mesh_xy, full_nmu, full_mun):
    """Both layouts with other panels' window columns inserted after the window (Eq. 8c).

    [0, b_hi) of ``base`` | columns [b_lo, b_hi) of each of the ``n_blocks`` extra
    panels | n_pad zero columns | [b_hi, nb) of ``base``, b_hi = b_lo + n_a: the extra
    internal columns of a non-diagonal line (:func:`_window_lines`).  Both layouts hold
    non-conjugated orbitals.
    """
    lo, hi = int(b_lo), int(b_lo) + int(n_a)
    out = (NamedSharding(mesh_xy, nmu_spec), NamedSharding(mesh_xy, mun_spec))

    def build(base_nmu, base_mun, extra_nmu, extra_mun):
        pad_n = jnp.zeros(base_nmu.shape[:1] + (int(n_pad),) + base_nmu.shape[2:], base_nmu.dtype)
        pad_m = jnp.zeros(base_mun.shape[:3] + (int(n_pad),), base_mun.dtype)
        nmu = jnp.concatenate([base_nmu[:, :hi]] + [x[:, lo:hi] for x in extra_nmu]
                              + [pad_n, base_nmu[:, hi:]], axis=1)
        mun = jnp.concatenate([base_mun[..., :hi]] + [x[..., lo:hi] for x in extra_mun]
                              + [pad_m, base_mun[..., hi:]], axis=-1)
        return nmu, mun
    return jax.jit(build, out_shardings=out)


def _insert_columns(base, extras, b_lo, n_a, mesh_xy):
    """(nmu, mun, n_pad) of :func:`_line_insert_fn` for face pairs ``base`` and ``extras``;
    the ``n_pad`` zero columns keep the band axis a multiple of both mesh axes (the faces
    shard it over x and y)."""
    import math
    div = math.lcm(*(int(n) for n in mesh_xy.devices.shape))
    n_pad = (-(int(base[0].shape[1]) + len(extras) * int(n_a))) % div
    fn = _line_insert_fn(int(b_lo), int(n_a), len(extras), int(n_pad), base[0].sharding.spec,
                         base[1].sharding.spec, mesh_xy, tuple(base[0].shape),
                         tuple(base[1].shape))
    nmu, mun = fn(base[0], base[1], [x[0] for x in extras], [x[1] for x in extras])
    return nmu, mun, int(n_pad)


def _line_live(C):
    """(nk, n_A) bool: state m is live on the line C when max_n |C_mn| clears the
    occupation floor (``gw.efermi.band_in_occupation_window``, the one band-inclusion
    predicate).  At a diagonal C it is the populations' liveness."""
    from .efermi import band_in_occupation_window
    return np.asarray(band_in_occupation_window(np.max(np.abs(C), axis=-1)), bool)


def _line_head(head, table_ry, win, C_h, C_p):
    """The q -> 0 head of a non-diagonal line on the window (Ry): (val, cond, static).

    At q -> 0 the pair densities are band diagonal, so the head pairs each external
    index with the internal index on its own vertex (m = i, n = j): with line C a
    branch's head is C_ij h_i in the first table and C_ij h_j in the second, h its value
    at unit weight (:func:`_head_branch_rows`: the hole part at f = 1, the particle part
    at f = 0), in either pairing; its static-W limit is C_h s(1) + (1 - C_h) s(0).  At a
    diagonal C it is the populations' head.  val, cond (2, nk, n, n); static (nk, n, n).
    """
    n = C_h.shape[-1]
    occ = np.asarray(head.occupations)
    h_one, _, s_one = _head_branch_rows(replace(head, occupations=np.ones_like(occ)),
                                        table_ry, win)
    _, p_zero, s_zero = _head_branch_rows(replace(head, occupations=np.zeros_like(occ)),
                                          table_ry, win)
    two = lambda C, h: np.stack([C * h[:, :, None], C * h[:, None, :]])
    static = C_h * s_one[:, :, None] + (np.eye(n)[None] - C_h) * s_zero[:, :, None]
    return two(C_h, h_one), two(C_p, p_zero), static


def _window_lines(rho, pairing, *, line_carrier, line_sweep, census, head_rows, faces, b_lo,
                  n_a, nk, enk_t, occ, pop, plan, block4, block3, first, static_limit,
                  static_kernels, mesh_xy):
    """``window_sigma_c``'s outputs for a non-diagonal internal line (§1.5, Eq. 8c).

    The hole line C_h = rho' and the particle line C_p = 1 - rho' of the window, in the
    instantaneous basis of ``faces`` (psi' = psi U), enter the tau sweep as G = L diag(w)
    R^H with no occupation in w: w is 1 on a branch's live states (:func:`_line_live`),
    0 elsewhere.  The Sigma box needs live states in both branches of every sweep, so
    the particle line is the identity block (L = R = psi') minus the hole panels:
    G_L[1 - rho'] = psi' w psi'^H - psi' w rho' psi'^H.  ``vertex``: two sweeps, G_L
    (window columns L = psi', R = psi' rho') and G_R (L = psi' rho', R = psi'), each with
    an identity block of n_A columns inserted after the window (hole weight 0, particle
    +1; the window columns hole +1, particle -1); Sigma_ij(e_i) of both branches from
    G_L, Sigma_ij(e_j) from G_R.  ``symmetric``: one sweep, the G_L and G_R columns and
    the identity block side by side (3 n_A internal columns; hole 1/2, 1/2, 0; particle
    -1/2, -1/2, 1), both tables 1/2 (G_L + G_R).  The inserted columns sit inside the
    Sigma window, so the hole sweep's occupied projector, which gives the static term and
    the static limit, is psi' rho' psi'^H.  The head is :func:`_line_head`'s.  Each sweep
    has its own held session (``plan['line_sessions']``).
    """
    from common.collectives import gather_to_host
    from .cohsex_sigma import build_Gij
    win = slice(b_lo, b_lo + n_a)
    b_hi = b_lo + n_a
    nb = int(pop.shape[1])
    lines = dict(val=rho, cond=np.eye(n_a)[None] - rho)
    live = {name: _line_live(C).astype(np.float64) for name, C in lines.items()}
    sessions = None if plan.get("session") is None else plan.setdefault("line_sessions", {})
    plan_for = lambda key: plan if sessions is None else dict(
        plan, session=sessions.setdefault(key, {}))
    keep = np.ones(int(faces[0].shape[1]))
    hole = _rotate_faces(faces[0], faces[1], rho, b_lo, keep)        # psi' rho' on the window
    if pairing == "vertex":
        # (side, extra face blocks, panel source, hole weights, particle weights) per block
        todo = [("L", [faces], (None, "right")), ("R", [faces], ("left", None))]
        w_h, w_p = [1.0, 0.0], [-1.0, 1.0]
    else:
        todo = [("S", [hole, faces], (None, "right"))]
        w_h, w_p = [0.5, 0.5, 0.0], [-0.5, -0.5, 1.0]
    tabs, runs, src = {}, [], None
    e_w = enk_t[:, win]
    for side, extras, which in todo:
        nmu, mun, n_pad = _insert_columns(faces, extras, b_lo, n_a, mesh_xy)
        n_blk = len(extras)
        n_ins = n_blk * n_a + n_pad
        # the panel side carries psi' rho' on the window and psi' on every inserted block
        p_nmu, p_mun, _ = _insert_columns(hole, [faces] * n_blk, b_lo, n_a, mesh_xy)
        if side == "S":                  # faces [psi', psi' rho', psi']: R = [psi' rho', psi', psi']
            panels = (None, p_nmu)
        else:
            panels = (None, p_nmu) if which[1] == "right" else (p_mun, None)
        ins = lambda a, blks: np.concatenate([a[:, :b_hi]] + blks + [a[:, b_hi:]], axis=1)
        enk_l = ins(enk_t, [e_w] * n_blk + [np.repeat(e_w[:, :1], n_pad, axis=1)])
        occ_l = ins(occ, [occ[:, win]] * n_blk + [np.repeat(occ[:, b_lo:b_lo + 1], n_pad, 1)])
        mask = np.zeros((nk, nb + n_ins), bool)
        val_w = np.zeros((nk, nb + n_ins))
        cond_w = np.zeros((nk, nb + n_ins))
        for blk in range(n_blk + 1):
            cols = slice(b_lo, b_hi) if blk == 0 else slice(b_hi + (blk - 1) * n_a,
                                                             b_hi + blk * n_a)
            mask[:, cols] = True
            val_w[:, cols] = w_h[blk] * live["val"]
            cond_w[:, cols] = w_p[blk] * live["cond"]
        ib = InternalBands(mask=mask, coulomb_hole=False)
        ln = line_carrier((nmu, mun), panels, n_ins, enk_l, occ_l)
        res, req = line_sweep(ln, val_w, cond_w, ib, plan_for(f"line_{side}"),
                              keep_branches=True)
        runs.append(dict(name=side, sweep=census(), efermi_ry=float(res.efermi_ry),
                         nb_sigma=int(ln[2].nb_sigma), n_insert=int(n_ins)))
        t = {name: np.array(gather_to_host(block4(first(res.branches[name]).pair)))
             for name in ("val", "cond")}
        for name in ("val", "cond"):
            if side == "R":
                tabs[name][1] = t[name][1]
            else:
                tabs[name] = t[name]
        if src is None:
            term = getattr(res.sigma_c_kij, "term", None)
            tabs["static"] = (np.zeros((nk, n_a, n_a), np.complex128) if term is None else
                              np.asarray(gather_to_host(block3(term))))
            src = dict(line=ln, pop=val_w, ib=ib, request=req, efermi_ry=float(res.efermi_ry))
    frames = {r["efermi_ry"] for r in runs}
    if len(frames) != 1:
        raise ValueError(f"GATE window_sigma_lines: the line sweeps' omega frames differ "
                         f"({sorted(frames)} Ry)")
    efermi, req = src["efermi_ry"], src["request"]
    hr = head_rows(efermi, enk_t, pop, b_hi)
    h_static = np.zeros((nk, n_a, n_a), np.complex128)
    if hr is not None:
        hv, hc, h_static = _line_head(hr, req.tables_ry[req.slot("pair")], win,
                                      lines["val"], lines["cond"])
        tabs["val"] = tabs["val"] + hv
        tabs["cond"] = tabs["cond"] + hc
    pair = tabs["cond"] + tabs["val"] + tabs["static"][None]
    sw = [r["sweep"] for r in runs]
    out = dict(pair=pair, diag=np.einsum("kii->ki", pair[0]), efermi_ry=efermi, request=req,
               carrier=dict(slices=src["line"][2], nb_sigma=runs[0]["nb_sigma"]),
               branch_pair=tabs,
               sweep=dict(n_windows=sum(x["n_windows"] for x in sw),
                          window_tau_pairs=sum(x["window_tau_pairs"] for x in sw),
                          distinct_tau=sum(x["distinct_tau"] for x in sw),
                          windows=[(f"{r['name']}:{n_}", c_) for r in runs
                                   for n_, c_ in r["sweep"]["windows"]],
                          held=all(x["held"] for x in sw), rebuilds=sum(x["rebuilds"] for x in sw),
                          lines=[dict(name=r["name"], n_insert=r["n_insert"], **r["sweep"])
                                 for r in runs]),
               full_rho=dict(pairing=pairing, n_insert=int(runs[0]["n_insert"]),
                             max_offdiag=float(np.max(np.abs(
                                 rho - np.einsum("kii->ki", rho)[:, :, None] * np.eye(n_a)[None]))),
                             live=dict(val=int(np.sum(live["val"])), cond=int(np.sum(live["cond"])))))
    if static_limit:
        wfns_t, meta_t = src["line"][0], src["line"][1]
        sigma_sx, _, W_all = static_kernels(wfns_t, meta_t)
        state = _population_state(src["ib"].weights(src["pop"]), meta_t, src["pop"].shape)
        sx = np.asarray(gather_to_host(block3(sigma_sx(wfns_t, build_Gij(meta_t, mesh_xy, state),
                                                       W_all))))
        # the head's static limit is a matrix on a non-diagonal line: it is folded into sx
        out["static_limit"] = dict(sx=sx + h_static, head=np.zeros((nk, n_a)))
    return out


def _window_rho(window: dict, U, f, internal, omit_frozen):
    """The window's rho' (instantaneous basis) of a full-ρ internal line, Hermitized, or None.

    Refused unless the sweep is the active part's (the frozen bands' populations never
    change) and the populations are rho''s diagonal.
    """
    rho = window.get("rho")
    if rho is None:
        return None
    if internal != "active" or omit_frozen:
        raise ValueError(
            f"GATE window_sigma_full_rho: got rho with internal={internal!r}, omit_frozen="
            f"{bool(omit_frozen)}; want internal='active' (the full-ρ internal line is the "
            "window's own bands); fix: sweep the frozen part on the populations")
    rho = np.asarray(rho, np.complex128)
    if rho.shape != U.shape:
        raise ValueError(f"GATE window_sigma_shapes: got rho {rho.shape}; want {U.shape}")
    rh = np.conj(np.swapaxes(rho, -1, -2))
    scale = max(1.0, float(np.max(np.abs(rho))))
    if float(np.max(np.abs(rho - rh))) > 1e-10 * scale:
        raise ValueError("GATE window_sigma_rho_hermitian: rho' is not Hermitian (max|rho - "
                         f"rho^H| = {float(np.max(np.abs(rho - rh))):.3e}); fix: pass the state's "
                         "density matrix in the instantaneous basis")
    rho = 0.5 * (rho + rh)
    d = np.real(np.einsum("kii->ki", rho))
    if float(np.max(np.abs(d - f))) > 1e-10:
        raise ValueError("GATE window_sigma_rho_populations: f is not diag(rho') (max diff "
                         f"{float(np.max(np.abs(d - f))):.3e}); fix: pass f = diag(rho')")
    return rho


def _with_frequencies(request: OnShellRequest, extra_ry) -> OnShellRequest:
    """The request planned on its own energies united with ``extra_ry`` (Ry, its frame).

    The sweep reads each parent-row table entry at ``parent_index`` of the
    united list (``OnShellRequest.parent_ry``)."""
    freqs = np.unique(np.concatenate([request.frequencies_ry, np.asarray(extra_ry, np.float64)]))
    index = np.searchsorted(freqs, request.parent_ry).astype(np.int32)
    if not np.array_equal(freqs[index], request.parent_ry):
        raise ValueError("GATE window_sigma_frequencies: a table energy is not on the united list")
    return replace(request, frequencies_ry=freqs, parent_index=index)


@lru_cache(maxsize=16)
def _window_block_fn(mesh_xy, b_lo, n_a, ndim):
    """The window block [b_lo, b_lo + n_a)^2 of a band-sharded Sigma, replicated.

    ``ndim`` 4: ``(lead, nk, nb, nb)`` (an on-shell pair, a Sigma(omega) cube);
    3: ``(nk, nb, nb)`` (a static term).  The block is a few KB, so it moves to
    every device instead of the whole Sigma window to every host.
    """
    win = slice(int(b_lo), int(b_lo) + int(n_a))
    rep = NamedSharding(mesh_xy, P())
    if int(ndim) == 4:
        return jax.jit(lambda a: a[:, :, win, win], out_shardings=rep)
    return jax.jit(lambda a: a[:, win, win], out_shardings=rep)


def _head_branch_rows(head, table_ry, win):
    """The q -> 0 head on the window at one on-shell table, split by its occupation weights.

    ``head`` (``gw.head_correction.HeadSigmaDiag``) is sum_p pref_p [f/(d + z_p) +
    (1 - f)/(d - z_p)], linear in the occupation f of its own band: the hole part
    f A is f times its value at f = 1, the particle part (1 - f) B is 1 - f times its
    value at f = 0, both from ``HeadSigmaDiag.on_shell`` (the readers' one
    arithmetic).  ``static`` is its static-W limit, sum_p pref_p (2 f - 1)/z_p (the
    head's COHSEX: -f W_h(0) + W_h(0)/2 per cell).  Host (nk, n_A) each, Ry.
    """
    e = np.asarray(head.enk_ry)[:, win]
    f = np.asarray(head.occupations)[:, win]
    t = np.asarray(table_ry)[:, win]
    hole = f * replace(head, enk_ry=e, occupations=np.ones_like(f)).on_shell(t)[0]
    particle = (1.0 - f) * replace(head, enk_ry=e, occupations=np.zeros_like(f)).on_shell(t)[0]
    pref = np.asarray(head.residues_ry) / (float(head.cell_volume) * float(head.nk_tot))
    z = np.asarray(head.poles_ry, np.complex128).reshape(-1)
    static = (2.0 * f - 1.0) * complex(np.sum(pref / z)) if z.size else np.zeros_like(hole)
    return hole, particle, static


def add_window_parts(a: dict, b: dict) -> dict:
    """The window Sigma of a split band sum: :func:`window_sigma_c`'s outputs of two parts added.

    ``a`` and ``b`` are one window's outputs on complementary ``internal`` sets
    (:class:`InternalBands`: the frozen part held from t0 and a refresh's active
    part).  Every table adds (``pair``, ``diag``, ``onshell``, ``omega``, each
    ``branch_pair`` and ``static_limit`` table); the request, carrier and frame
    are ``b``'s (the frames must agree); ``sweep`` is ``b``'s with both parts'
    censuses under ``parts``.
    """
    if float(a["efermi_ry"]) != float(b["efermi_ry"]):
        raise ValueError("GATE window_sigma_parts: the parts' omega frames differ "
                         f"({a['efermi_ry']!r} vs {b['efermi_ry']!r} Ry)")
    out = dict(b)
    for key in ("pair", "diag", "onshell", "omega"):
        if (key in a) != (key in b):
            raise ValueError(f"GATE window_sigma_parts: {key!r} in one part only")
        if key in b:
            out[key] = a[key] + b[key]
    for key in ("branch_pair", "static_limit"):
        if (key in a) != (key in b):
            raise ValueError(f"GATE window_sigma_parts: {key!r} in one part only")
        if key in b:
            out[key] = {name: a[key][name] + b[key][name] for name in b[key]}
    out["sweep"] = dict(b.get("sweep") or {}, parts=[a.get("sweep"), b.get("sweep")])
    return out


def shared_pole_window_source(model: dict, meta, *, efermi_ry: float, linalg) -> dict:
    """The ``poles`` argument of :func:`window_sigma_c` for a scalar shared-pole model.

    ``model`` is the screening handle (``path``, ``identity``, ``digest``, ``K``) of
    ``gw.shared_pole_screening.screen_shared_poles``; ``meta`` the bundle whose
    ``shared_pole_recipe`` and ``shared_pole_capacity`` built it (the Σ stage's
    eta gate and model validation read them); ``efermi_ry`` the omega frame of the
    run's shared-pole Σ stage (``efermi.resolve_sigma_efermi_ry``); ``linalg`` the
    deck's resolved dense layout (``gw_config.linalg_resolution(...).layout``),
    which places the W(τ) synthesis.
    """
    return dict(member="shared_pole", model=dict(model), meta=meta, efermi_ry=float(efermi_ry),
                representation=model.get("representation", "scalar"),
                linalg=linalg)


def fresh_shared_pole_capacity(meta, mesh_xy):
    """A shallow copy of ``meta`` with its own shared-pole capacity ledger, no stage live.

    The ledger (``gw.shared_pole_recipe.CapacityLedger``) prices one map's phases
    under unique stage names; a window Sigma or a static-W call after the GW run
    (TD-aGW, every refresh) is its own phase, so each gets a fresh ledger with no
    upstream reservation live (the run's bank and constructor are gone by then).
    """
    import copy
    from common.gpu_utils import device_budget_bytes
    from .shared_pole_recipe import CapacityLedger
    out = copy.copy(meta)
    out.shared_pole_capacity = CapacityLedger(out, mesh_xy=mesh_xy,
                                              device_budget_bytes=int(device_budget_bytes()))
    out.shared_pole_capacity.live_stages = ()
    return out


def _shared_pole_window_run(wfns_t, source, meta_t, mesh_xy, plan, pop, internal_bands, *,
                            request_for, omega_grid_ry, keep_branches, print_fn, particle=None):
    """One window sweep of a shared-pole model: the run's shared-pole Σ stage on the window carrier.

    ``gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`` (``sigma_w_model =
    "shared_pole"``) with the causal branches of the band populations ``pop``
    (:func:`_population_branches`, restricted to ``internal_bands``; ``particle``
    the conduction weights where they are not 1 - pop, a non-diagonal line's),
    the run's ω frame, broadening and quadrature, and the plan's held session.
    Returns the executor's result with ``efermi_ry`` set to the frame.
    """
    from .mpa.sigma import compute_sigma_c_mpa_omega_grid
    model, efermi = source["model"], float(source["efermi_ry"])
    cfg = plan["sigma_cfg"]
    request = None if request_for is None else request_for(efermi)
    omega = omega_grid_ry if request is None else request.frequencies_ry
    s = wfns_t.slices
    # The shared-pole executor's branches run over the Σ band sum (``slices.sigma_sum``,
    # gw.mpa.sigma._branches), which the faces carry; ``pop`` is on ``slices.full``.
    nb_all = int(np.asarray(jax.device_get(wfns_t.enk)).shape[1])
    cols = np.arange(nb_all)[s.full]
    sel = np.searchsorted(cols, np.arange(nb_all)[s.sigma_sum])
    w = pop if internal_bands is None else internal_bands.weights(pop)
    cw = None if particle is None else (
        np.asarray(particle, np.float64) if internal_bands is None
        else internal_bands.weights(particle))
    branches = _population_branches(
        omega, np.asarray(jax.device_get(wfns_t.enk), np.float64)[:, s.sigma_sum], efermi,
        np.asarray(w)[:, sel],
        internal=None if internal_bands is None else
        np.broadcast_to(np.asarray(internal_bands.mask, bool), np.shape(w))[:, sel],
        particle_weights=None if cw is None else np.asarray(cw)[:, sel])
    res = compute_sigma_c_mpa_omega_grid(
        wfns_t, model["path"], meta_t, mesh_xy, sigma_w_model="shared_pole",
        fit_identity=model["identity"], fit_digest=model["digest"], request=request,
        omega_grid_ry=None if request is not None else omega_grid_ry, efermi_ry=efermi,
        regularization_width_ry=float(cfg.regularization_ev) / RYD_TO_EV,
        edge_factor=float(cfg.window_edge_factor), quadrature_eps=float(cfg.quadrature_eps),
        omega_grid_step_ry=float(cfg.omega_step_ev) / RYD_TO_EV,
        pole_batch_size=int(plan["mpa_cfg"].pole_batch_size),
        expected_screening_diagrams=plan["screening_diagrams"], sigma_branches=branches,
        fixed_quadrature_session=plan.get("session"), linalg=source.get("linalg"),
        keep_branches=keep_branches, print_fn=print_fn)
    return replace(res, efermi_ry=efermi)


def ordered_window_spectral_tail(pair_first, pair_last, e_state_ev, reference):
    """One-sided complex tail: S3_ab(E_b)+(S3-S1)_ab(E_b) r(E_b).

    ``pair_first/last[2,nk,nA,nA]`` are small host Ry tables from the SAME
    current sweep, at the reference fit's N1 and N3. Slot0 samples row E_a,
    slot1 column E_b. ``e_state_ev[nk,nA]`` uses the reference DFT/Weyl
    ladder's absolute eV zero. ``reference`` holds the native equilibrium
    SpectralShellFit, its authenticated input-frame identifier, and an
    explicit uniform ladder_shift_ev. Only the shape/ladder is frozen;
    amplitudes and denominator energies update. No shell arithmetic average,
    Hermitian projection or rate correction is made here.
    """
    from .band_extrapolation import SpectralShellFit, spectral_tail_ratio
    fit = reference.get("fit")
    if not isinstance(fit, SpectralShellFit) or not reference.get("input_frame"):
        raise ValueError("GATE ordered_sigma_tail: native fit and authenticated input frame required")
    if fit.n_failed:
        raise ValueError("GATE ordered_sigma_tail: reference fit has states without a tail")
    shift = float(reference["ladder_shift_ev"])
    first, last = np.asarray(pair_first), np.asarray(pair_last)
    e = np.asarray(e_state_ev, np.float64)
    if (e.ndim != 2 or first.shape != (2, e.shape[0], e.shape[1], e.shape[1])
            or last.shape != first.shape or not np.isfinite(shift)
            or not np.isfinite(first).all() or not np.isfinite(last).all()):
        raise ValueError("GATE ordered_sigma_tail: finite matched row/column tables required")
    ratio = spectral_tail_ratio(fit.ladder, fit.counts, e-shift,
        beta=fit.beta, omega_ev=fit.omega_ev)
    weight = np.stack((np.broadcast_to(ratio[:, :, None], e.shape+(e.shape[1],)),
                       np.broadcast_to(ratio[:, None, :], e.shape+(e.shape[1],))))
    out = last+(last-first)*weight
    return out, dict(input_frame=reference["input_frame"],
        counts=np.asarray(fit.counts).tolist(), beta=float(fit.beta), omega_ev=float(fit.omega_ev),
        ladder_shift_ev=shift, ratio=ratio.tolist(), max_delta_ry=float(np.max(abs(out-last))),
        approximation="frozen_equilibrium_shell_shape_live_complex_amplitude",
        shell_ordering="slot0_row_slot1_column_no_arithmetic_average",
        destination="full_virtual_hermitian_material_self_energy_only")


def _ordered_tail_brackets(reference, *, b_lo, hi, n_insert, nb):
    """Partition the live ordered line while retaining active, N1 and N3 matrices."""
    counts = np.asarray(reference["fit"].counts)
    if (counts.shape != (3,) or not (hi < counts[0] < counts[1] < counts[2] <= nb)
            or b_lo < 0 or n_insert < hi-b_lo):
        raise ValueError("GATE ordered_sigma_tail: three empty-tail cuts must lie above active window")
    brackets = [(b_lo, hi+hi-b_lo)]  # active rho and inserted I panels; pad omitted
    labels = [hi-b_lo]
    if b_lo:
        brackets.append((0, b_lo))
        labels.append(hi)
    lo = hi
    for cut in counts:
        end = nb if cut == counts[-1] else int(cut)
        brackets.append((lo+n_insert, end+n_insert))
        labels.append(int(cut))
        lo = int(cut)
    first = len(brackets)-3
    return tuple(brackets), tuple(labels), (0, first, len(brackets)-1), first


def _ordered_window_lines(source, plan, *, mesh_xy, faces, rho, U, b_lo, n_a,
                          enk, occ, pop, line_carrier, print_fn):
    """Full microscopic retarded branches, active rates and all-band virtual MH.

    One native band-bracket sweep retains both the active intermediate
    contribution and the full virtual sum, at the same compact real
    destination bands. The left electronic panel is psi rho, so every
    phase belongs to the right internal index. An inserted identity active
    panel represents I-rho without diagonalizing the density.
    """
    from common.collectives import gather_to_host
    from .mpa.sigma import ordered_shared_pole_sigma
    from .ppm_windows import _SigmaBranch

    nk, nb = pop.shape
    hi = int(b_lo) + int(n_a)
    # Both faces remain the actual instantaneous orbitals; only the Green
    # left panel receives rho. Frozen spectators have diagonal 0/1 lines.
    hole = _rotate_faces(*faces, rho, b_lo, np.ones(faces[0].shape[1]))
    enlarged_n, enlarged_m, n_pad = _insert_columns(faces, [faces], b_lo, n_a, mesh_xy)
    _, left_m, _ = _insert_columns(hole, [faces], b_lo, n_a, mesh_xy)
    n_insert = int(n_a) + n_pad
    insert = lambda a, x, pad: np.concatenate((a[:, :hi], x, pad, a[:, hi:]), axis=1)
    energy = insert(enk, enk[:, b_lo:hi], np.repeat(enk[:, b_lo:b_lo+1], n_pad, axis=1))
    occ_l = insert(occ, occ[:, b_lo:hi], np.zeros((nk, n_pad)))
    wfns_t, meta_t, slices_t, _ = line_carrier((enlarged_n, enlarged_m),
        (left_m, None), n_insert, energy, occ_l)
    efermi = float(source["efermi_ry"])
    external = enk[:, b_lo:hi] - efermi
    request, request_receipt = ordered_window_request(enk[:, b_lo:hi], efermi)
    hole_weight = insert(pop, np.zeros((nk, n_a)), np.zeros((nk, n_pad)))
    part_weight = insert(1.0-pop, np.ones((nk, n_a)), np.zeros((nk, n_pad)))
    # Preserve every nonzero coherence. The negative particle panel is
    # -rho, NOT I-rho: even a fully occupied column must cancel its
    # inserted identity column exactly.
    live_h = np.any(np.abs(rho) != 0.0, axis=-1)
    hole_weight[:, b_lo:hi] = live_h
    part_weight[:, b_lo:hi] = -live_h.astype(np.float64)
    # All spectator and active phases use the SAME explicit full energy
    # table. The sign is a line coefficient, never a pole/sign projection.
    branches = tuple(_SigmaBranch(tag=f"ordered:{space}", E_A=energy-efermi,
        base_mask_A=weight != 0.0, space=space, neg_omega_half=False,
        omega_abs=np.abs(request.frequencies_ry), omega_idx=np.arange(request.frequencies_ry.size),
        band_weight=weight) for space, weight in (("cond", part_weight), ("val", hole_weight)))
    brackets = [(b_lo, hi+n_a)]
    if b_lo:
        brackets.append((0, b_lo))
    if hi+n_insert < nb+n_insert:
        brackets.append((hi+n_insert, nb+n_insert))
    tail = plan.get("spectral_tail")
    matrix_counts = band_counts = tail_first = None
    if tail is not None:
        logical = int(source["meta"].b_id_4_sigma_user)-int(source["meta"].b_id_0)
        if int(tail["fit"].counts[-1]) != logical:
            raise ValueError("GATE ordered_sigma_tail: reference N3 differs from current Sigma band support")
        brackets, band_counts, matrix_counts, tail_first = _ordered_tail_brackets(
            tail, b_lo=b_lo, hi=hi, n_insert=n_insert, nb=nb)
    cfg = plan["sigma_cfg"]
    meta_sigma = fresh_shared_pole_capacity(meta_t, mesh_xy)
    # All ranks read live room with current full-band faces/model resident;
    # a fresh ledger alone has no inherited baseline or agreed phase room.
    meta_sigma.shared_pole_capacity.at_phase()
    res, receipt = ordered_shared_pole_sigma(wfns_t, source["model"],
        meta_sigma, mesh_xy, request=request,
        sigma_branches=branches, regularization_width_ry=float(cfg.regularization_ev)/RYD_TO_EV,
        quadrature_eps=float(cfg.quadrature_eps), projection_bands=(b_lo, hi),
        band_brackets=tuple(brackets), band_counts=band_counts,
        band_matrix_counts=matrix_counts, print_fn=print_fn)
    block = _window_block_fn(mesh_xy, 0, n_a, 4)
    both = {}
    for name, count in (("active", 0), ("full", len(brackets)-1)):
        both[name] = {space: np.asarray(gather_to_host(block(res.branches[space].point(count).pair)))
                      for space in ("cond", "val")}
    tail_receipt = None
    if tail is not None:
        tail_receipt = {}
        for space in ("cond", "val"):
            first_pair = np.asarray(gather_to_host(block(res.branches[space].point(tail_first).pair)))
            both["full"][space], tail_receipt[space] = ordered_window_spectral_tail(
                first_pair, both["full"][space], enk[:, b_lo:hi]*RYD_TO_EV, tail)
    head = plan.get("head")
    if head is not None:
        head = replace(head, enk_ry=enk[:, b_lo:hi], occupations=np.ones((nk, n_a)),
                       efermi_ry=efermi)
        head_pair = {space: np.zeros((2, nk, n_a, n_a), complex) for space in ("cond", "val")}
        for slot in range(2):
            for i in range(n_a):
                energies = (np.repeat(external[:, i:i+1], n_a, axis=1) if slot == 0 else external)
                h = np.conj(head.on_shell(energies)[0])
                p = replace(head, occupations=np.zeros((nk, n_a))).on_shell(energies)[0]
                head_pair["val"][slot, :, i] = rho[:, i] * h
                head_pair["cond"][slot, :, i] = (np.eye(n_a)[None]-rho)[:, i] * p
        for count in both.values():
            for space in head_pair:
                count[space] = count[space] + head_pair[space]
    # Existing downstream column algebra expects an ADVANCED hole pair.
    # This exact small-table conversion recovers Rh_column once, with no
    # arithmetic shell average and no additional principal-value term.
    compatibility = lambda t: dict(cond=t["cond"],
        val=np.conj(t["val"][::-1].swapaxes(-1, -2)), static=np.zeros((nk, n_a, n_a), complex))
    active, full = (compatibility(both[key]) for key in ("active", "full"))
    total = both["full"]["cond"] + both["full"]["val"]
    receipt.update(internal_bands=int(nb), real_destinations=int(n_a),
        external_request=request_receipt,
        rate_intermediates="active_only", virtual_intermediates="all",
        ordered_internal_phase="right_index", head="frozen_reference_current_electronic_lines",
        bulk_evaluation_eta_ry=float(cfg.regularization_ev)/RYD_TO_EV,
        head_pole_imaginary_parts_ry=([] if head is None else
            np.asarray(head.poles_ry, np.complex128).imag.tolist()),
        head_width_convention="frozen_source_poles_no_added_bulk_eta",
        full_density=True, spatial_projection=False,
        bracket_intervals=[list(x) for x in brackets], spectral_tail=tail_receipt)
    return dict(pair=total, diag=np.einsum("kii->ki", total[1]),
        onshell=None, branch_pair=full, active_branch_pair=active,
        static=np.zeros((nk, n_a, n_a), complex), retarded_pair=both["full"],
        request=request, efermi_ry=efermi, native_receipt=receipt,
        carrier=dict(slices=slices_t, nb_sigma=int(n_a)),
        sweep=dict(n_windows=receipt["n_windows"], window_tau_pairs=receipt["window_tau_pairs"],
            distinct_tau=receipt["window_tau_pairs"], held=False, rebuilds=0,
            windows=[(x["name"], x["node_count"]) for x in receipt["windows"]]),
        full_rho=dict(pairing="vertex", n_insert=n_insert), basis=U)


def window_sigma_c(wfns, poles, window: dict, plan: dict, *, meta, mesh_xy, omega_ry=None,
                   omit_frozen: bool = False, branches: bool = False,
                   static_limit: bool = False, internal: str | None = None,
                   keep_frozen=None, static_limit_wc0=None, pairing: str = "symmetric",
                   print_fn=None) -> dict:
    """Sigma^c of a band window at an instantaneous state, from a pole model (Ry).

    docs/theory/td_agw_dynamical.md §1.2-1.5 (Eqs. 2-3, 8) and §9 item 5.  The GW
    run's own Σ τ sweep (:func:`compute_sigma_c_ppm_omega_grid` and the shared
    executor ``gw.mpa.sigma``) on a carrier that differs from the run's only
    in the window: its faces are rotated, psi_A -> psi_A U (one n_A x n_A
    rotation per k), its energies are the instantaneous e and its branch
    weights the populations f (valence f, conduction 1 - f).  Every other
    band keeps its equilibrium energy and step occupation, and G(tau) of
    every band is composed at every node from the stored centroid faces;
    no G(tau) is stored.  No file is read or written; only the window block
    of each result leaves the devices.

    Parameters
    ----------
    wfns : the GW run's ``Wavefunctions`` bundle (raw-parent faces of every band
        at the centroids, ``green_parent``; energies ``enk`` and step
        occupations ``occ``, Ry).  Held for the run, never modified.  Its parent
        rows must be the full zone (a general drive breaks the point group).
    poles : :class:`PPMBuildResult` (:func:`fit_pole_model` / :func:`screening_update`), or a
        shared-pole source, :func:`shared_pole_window_source`: the scalar shared-pole model
        of a GW run (or of a refit at other occupations) swept by the run's own
        shared-pole Σ stage (``gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`` with
        ``sigma_w_model = "shared_pole"``); it has no invalid modes, so its
        ``branch_pair['static']`` is zero.
    window : ``dict(b_lo, U, e, f)`` — the active window [b_lo, b_lo + n_A) on the
        bundle's band axis: U (nk, n_A, n_A) the instantaneous orbitals'
        columns in the band basis, e (nk, n_A) their energies (Ry, the
        bundle's absolute frame), f (nk, n_A) their populations; full-k rows
        in the bundle's k order.  Optional ``rho`` (nk, n_A, n_A): the window's
        density matrix rho' = U^H rho U in the instantaneous basis (f its
        diagonal; ``internal = "active"`` only): the internal line is then the
        full-ρ one, hole rho'_mn and particle (1 - rho')_mn, contracted as G = L
        diag(w) R^H on the panels psi' and psi' C with the energy pairing
        ``pairing`` (td_agw_dynamical.md §1.5, Eqs. 8a and 8c; :func:`_window_lines`).
        At a diagonal rho' it is the populations' sweep to rounding.  Optional
        ``e_frozen`` (nk, nb_full) Ry, the bundle's absolute frame: the internal
        energies of the bands outside the window (default the bundle's ``enk``).
    plan : :func:`window_sigma_plan`.
    meta, mesh_xy : the run's.
    omega_ry : optional (n_omega,) frequencies measured from the body's omega
        reference (returned as ``efermi_ry``): the matrix Sigma^c_ij(omega)
        is also evaluated there (the grid route).
    omit_frozen : CONTROL VARIANT ONLY — the frozen bands are dropped from G.
    branches : also return each causal branch's pair tables from the SAME sweep
        (``keep_branches``): ``branch_pair`` = dict(cond, val, static), cond the
        particle branch (weight 1 - f, residue R_+, retarded broadening), val the
        hole branch (weight f, R_-, time-ordered broadening: E - e + Omega - i eta),
        each with its part of the head (:func:`_head_branch_rows`), and static the
        invalid-pole static term's block (nk, n_A, n_A), Hermitian and frequency
        independent: pair = cond + val + static to rounding.
    static_limit : also return ``static_limit`` = dict(sx, head): the static-W limit
        of the populations' screened exchange, sx (nk, n_A, n_A) = the window block
        of -<G_occ W^c(0)> with W^c(0) = W(0) - V on every mode (the invalid-pole
        kernels with no mode masked), and head (nk, n_A) its head part
        (:func:`_head_branch_rows`).  At fixed screening the populations enter the
        static limit of the GN-PPM Sigma only through these (the COH half and the
        D/Omega term are population independent).
    internal : None (every band of the Σ sum as an intermediate state: the
        incumbent), ``"active"`` (the window's own bands [b_lo, b_hi), with the
        head and the invalid-pole term's screened exchange of their populations)
        or ``"frozen"`` (every other band, with the invalid-pole term's Coulomb
        hole; no head).  At fixed poles the two parts add up to the whole sum
        (:class:`InternalBands`), and with ``dress = "none"`` the frozen part
        never changes: the frozen-W refresh sweeps the active part only
        (td_agw_dynamical.md §1.5-1.6).
    keep_frozen : with ``internal = "frozen"`` only: (nb_full,) bool, the frozen
        bands kept in the sum (a truncation; its error is measured, not bounded).
    pairing : :data:`LINE_PAIRINGS`, read with ``rho`` only: ``"symmetric"`` (1/2
        {g(e'), C} in both tables, Eq. 8a) or ``"vertex"`` (Sigma_ij(e_i) from G_L,
        Sigma_ij(e_j) from G_R, Eq. 8c).  With ``rho`` the head's static limit is a
        matrix and sits in ``static_limit['sx']`` (``head`` zero).
    static_limit_wc0 : with ``static_limit`` only: the W^c(0) of the static limit,
        (nq, mu, mu) as ``poles.Wc0_q`` (default ``poles.Wc0_q``; required for a
        shared-pole source, whose static value is its model's,
        ``gw.mpa.sigma.shared_pole_static_wc``).  A refreshed-screening collision term
        sweeps the poles of W(t) and subtracts the static limit of the run's W0, the
        part K drho already holds (td_agw_dynamical.md §7.1).

    Returns
    -------
    dict with ``pair`` (2, nk, n_A, n_A): Sigma^c_ij(e_i), Sigma^c_ij(e_j) in the
    instantaneous basis (body, invalid-pole static term and head);
    ``onshell`` (nk, n_A, n_A): U Sigma~ U^H with Sigma~ = 1/2 (pair_0 + pair_1), the
    complex on-shell operator of Eq. 8 in the band basis (no Hermitian
    projection; its Hermitian part is the operator of sigma-on-shell.md);
    ``diag`` (nk, n_A): Sigma^c_ii(e_i); ``omega`` (n_omega, nk, n_A, n_A) in the band
    basis when ``omega_ry`` is given; ``efermi_ry``; ``request`` (the on-shell
    tables); ``carrier`` = dict(slices, nb_sigma), the rotated carrier's band
    slices and the Σ window [0, nb_sigma) the sweep ran on; ``branch_pair`` /
    ``static_limit`` on request.  ``pair``,
    ``branch_pair`` and ``static_limit`` are in the instantaneous basis (the
    columns of U); ``onshell`` and ``omega`` in the band basis.  ``sweep``: the
    on-shell sweep's window plan (``gw.quadrature_log``): ``n_windows``,
    ``window_tau_pairs``, ``distinct_tau``, ``windows`` [(name, n_tau)].  With ``rho``, the
    line sweeps' totals and ``lines`` (each sweep's census); ``full_rho`` (or None)
    = dict(pairing, n_insert, max_offdiag, live).

    The Sigma band window is [0, b_lo + n_A): the invalid-pole static term's
    occupied projector (``cohsex_sigma.build_Gij``) needs every occupied band
    inside it.  Projecting only the window is design item 6 (separate encode
    and decode faces).
    """
    from .wavefunction_bundle import BandSlices
    p = print if print_fn is None else print_fn
    quiet = (lambda *a, **k: None) if print_fn is None else print_fn
    carrier = wfns.green_parent
    kplan = None if carrier is None else carrier.plan
    nk = int(meta.nk_tot)
    if kplan is None or int(kplan.n_parent) != nk or not np.array_equal(
            np.asarray(kplan.irr_idx), np.arange(nk)):
        raise ValueError(
            "GATE window_sigma_full_zone: got a bundle whose parents are not the full zone "
            f"(n_parent={None if kplan is None else int(kplan.n_parent)}, nk={nk}); want "
            "one parent per k (a general drive breaks the point group); fix: run the GW "
            "parent on an identity-only grid, as runs/CrI3/516")
    s = wfns.slices
    b_lo = int(window["b_lo"])
    U = np.asarray(window["U"], np.complex128)
    e = np.asarray(window["e"], np.float64)
    f = np.asarray(window["f"], np.float64)
    n_a = int(U.shape[-1])
    b_hi = b_lo + n_a
    if U.shape != (nk, n_a, n_a) or e.shape != (nk, n_a) or f.shape != (nk, n_a):
        raise ValueError(f"GATE window_sigma_shapes: got U {U.shape}, e {e.shape}, f {f.shape}; "
                         f"want ({nk}, {n_a}, {n_a}) and ({nk}, {n_a})")
    if not (int(s.b2) - int(s.b0) <= b_hi <= int(s.nb_sigma_sum)) or b_lo < 0:
        raise ValueError(f"GATE window_sigma_bands: window [{b_lo}, {b_hi}) must close above "
                         f"the LUMO ({int(s.b2) - int(s.b0)}) inside the Sigma sum "
                         f"({int(s.nb_sigma_sum)})")
    win = slice(b_lo, b_hi)
    enk = np.asarray(jax.device_get(wfns.enk), np.float64)
    occ = np.asarray(jax.device_get(wfns.occ), np.float64)
    enk_t = enk.copy()
    if window.get("e_frozen") is not None:
        e_fz = np.asarray(window["e_frozen"], np.float64)
        if e_fz.shape != enk.shape:
            raise ValueError(f"GATE window_sigma_shapes: got e_frozen {e_fz.shape}; want "
                             f"{enk.shape} (the bundle's energy table)")
        enk_t = e_fz.copy()
    enk_t[:, win] = e
    pop = occ[:, s.full].copy()
    pop[:, win] = f
    keep = np.ones(int(carrier.psi_nmu.shape[1]))
    if omit_frozen:
        keep[:] = 0.0
        keep[win] = 1.0
    internal_bands = None
    if internal is not None:
        if internal not in ("active", "frozen"):
            raise ValueError(f"GATE window_sigma_internal: got {internal!r}; want None, "
                             "'active' or 'frozen'")
        active = np.zeros(pop.shape, bool)
        active[:, win] = True
        mask = active if internal == "active" else ~active
        if keep_frozen is not None:
            if internal != "frozen":
                raise ValueError("GATE window_sigma_keep_frozen: a truncation of the frozen "
                                 "sum needs internal='frozen'")
            mask = mask & np.broadcast_to(np.asarray(keep_frozen, bool), pop.shape)
        internal_bands = InternalBands(mask=mask, coulomb_hole=(internal == "frozen"))
    general = (isinstance(poles, dict) and
               poles.get("representation", poles.get("model", {}).get("representation")) == "ordered_complex")
    rho_w = _window_rho(window, U, f, "active" if general else internal, omit_frozen)
    if rho_w is not None and (pairing not in LINE_PAIRINGS or omega_ry is not None
                              or not branches):
        raise ValueError(
            f"GATE window_sigma_full_rho: got rho with pairing={pairing!r}, omega_ry "
            f"{'set' if omega_ry is not None else 'None'}, branches={bool(branches)}; want "
            f"pairing in {LINE_PAIRINGS}, no omega grid, branches=True (the line sweeps "
            "return their branch tables); fix: pass branches=True and drop omega_ry")
    psi_nmu, psi_mun = _rotate_faces(carrier.psi_nmu, carrier.psi_mun, U, b_lo, keep)
    rep = NamedSharding(mesh_xy, P(None, None))
    from common.collectives import device_put_process_local
    put = lambda a: device_put_process_local(np.ascontiguousarray(a), rep)
    rows = np.asarray(kplan.parent_full_rows if kplan.parent_full_rows is not None
                      else np.arange(nk))
    spm = isinstance(poles, dict) and poles.get("member") == "shared_pole"
    if spm and static_limit and static_limit_wc0 is None:
        raise ValueError("GATE window_sigma_static_wc0: got a shared-pole source without "
                         "static_limit_wc0; want the model's W^c(0) (gw.mpa.sigma."
                         "shared_pole_static_wc); fix: pass static_limit_wc0")
    meta_src = poles.get("meta", meta) if spm else meta
    head = None if internal == "frozen" else plan.get("head")

    def line_carrier(faces, panels=(None, None), n_ins=0, enk_l=enk_t, occ_l=None):
        """(wfns_t, meta_t, slices_t, enk_l): one sweep's carrier, the rotated faces with
        ``n_ins`` columns inserted after the window and the line's G panels."""
        occ_c = carrier.occ if occ_l is None else put(occ_l[rows])
        carrier_t = replace(carrier, psi_nmu=faces[0], psi_mun=faces[1],
                            enk=put(enk_l[rows]), occ=occ_c,
                            green_left=panels[0], green_right=panels[1])
        if n_ins:
            top = lambda edge: (int(edge) or int(s.b4))
            chi = top(s.b4_chi)
            slices_t = BandSlices.from_band_edges(
                s.b0, min(int(s.b1), int(s.b0) + b_hi), s.b2, int(s.b0) + b_hi + n_ins,
                int(s.b4) + n_ins, b4_chi=chi + (n_ins if chi > int(s.b0) + b_hi else 0),
                b4_sigma=top(s.b4_sigma) + n_ins, b4_logical=top(s.b4_logical) + n_ins)
        else:
            slices_t = BandSlices.from_band_edges(
                s.b0, min(int(s.b1), int(s.b0) + b_hi), s.b2, int(s.b0) + b_hi, s.b4,
                b4_chi=s.b4_chi, b4_sigma=s.b4_sigma, b4_logical=s.b4_logical)
        # The Sigma stage reads the parent carrier only; full-k faces, if the
        # bundle holds any, would be the unrotated window and are dropped.
        wfns_t = replace(wfns, enk=put(enk_l), slices=slices_t, green_parent=carrier_t,
                         psi_nmu=None, psi_mun=None)
        if occ_l is not None:
            wfns_t = replace(wfns_t, occ=put(occ_l))
        meta_t = dataclasses.replace(meta_src, b_id_3=int(meta.b_id_0) + b_hi + n_ins)
        if n_ins:
            lo = int(meta.b_id_0) + b_hi
            shift = lambda edge: int(edge) + (n_ins if int(edge) > lo else 0)
            meta_t = dataclasses.replace(
                meta_t, b_id_4=int(meta_src.b_id_4) + n_ins,
                b_id_4_user=int(meta_src.b_id_4_user or meta_src.b_id_4) + n_ins,
                b_id_4_chi=shift(meta_src.b_id_4_chi), b_id_4_sigma=shift(meta_src.b_id_4_sigma))
        if spm:
            # The shared-pole Σ stage reads the map's recipe and census, set on the run's
            # metadata as attributes outside its fields (gw_jax), which dataclasses.replace
            # does not carry (its __post_init__ rederives nb_sigma and the band edges from
            # b_id_3); its capacity ledger is this call's own.
            for name in ("shared_pole_recipe", "shared_pole_census", "shared_pole_state_identity"):
                if hasattr(meta_src, name):
                    setattr(meta_t, name, getattr(meta_src, name))
            meta_t = fresh_shared_pole_capacity(meta_t, mesh_xy)
        return wfns_t, meta_t, slices_t, enk_l

    if general:
        if rho_w is None or pairing != "vertex" or omit_frozen or static_limit or keep_frozen is not None:
            raise ValueError("GATE ordered_window_sigma: full rho, vertex ordering and full virtual "
                             "sum required; static-W subtraction and frozen-band truncation are not this member")
        return _ordered_window_lines(poles, plan, mesh_xy=mesh_xy, faces=(psi_nmu, psi_mun),
            rho=rho_w, U=U, b_lo=b_lo, n_a=n_a, enk=enk_t, occ=occ, pop=pop,
            line_carrier=line_carrier, print_fn=quiet)

    def line_sweep(line, pop_l, part_l, ib_l, plan_l, omega_grid=None, keep_branches=False):
        """One Σ sweep on a line carrier: on shell (``omega_grid`` None) or on that grid
        (the debug route).  Returns (result, the on-shell request or None)."""
        wfns_t, meta_t, slices_t, enk_l = line
        nb_sig = int(slices_t.nb_sigma)
        built = {}

        def request_for(efermi_ry):
            e_rel = (enk_l[:, :nb_sig] - float(efermi_ry)) * RYD_TO_EV
            if plan_l.get("window_only"):
                e_rel = e_rel.copy()
                e_rel[:, :b_lo] = e_rel[:, b_lo:b_lo + 1]
            req = on_shell_request(e_eval_rel_ev=e_rel, e_dft_rel_ev=e_rel,
                                   omega_range_ev=plan_l["omega_range_ev"],
                                   out_of_grid=plan_l["out_of_grid"],
                                   irr_idx=np.asarray(kplan.irr_idx), n_parent=int(kplan.n_parent))
            if plan_l.get("frequencies_abs_ry") is not None:
                req = _with_frequencies(req, plan_l["frequencies_abs_ry"] - float(efermi_ry))
            built["request"] = req
            return req

        if spm:
            # each Sigma evaluation (the on-shell pair, the grid) is its own capacity phase:
            # the ledger's stage names are unique per ledger, so every call gets a fresh one
            res = _shared_pole_window_run(
                wfns_t, poles, fresh_shared_pole_capacity(meta_t, mesh_xy), mesh_xy, plan_l,
                pop_l, ib_l, request_for=request_for if omega_grid is None else None,
                omega_grid_ry=omega_grid, keep_branches=keep_branches, print_fn=quiet,
                particle=part_l)
        else:
            res = compute_sigma_c_ppm_omega_grid(
                wfns_t, poles, meta_t, mesh_xy, ppm_cfg=plan_l["ppm_cfg"],
                sigma_cfg=plan_l["sigma_cfg"], mpa_cfg=plan_l["mpa_cfg"],
                ansatz=plan_l["ansatz"], screening_diagrams=plan_l["screening_diagrams"],
                plan=None, fixed_quadrature_session=plan_l.get("session"),
                request_for=request_for if omega_grid is None else None,
                omega_grid_ry=omega_grid, populations=pop_l, internal=ib_l,
                keep_branches=keep_branches, print_fn=quiet, particle_weights=part_l)
        return res, built.get("request")

    def census():
        from .quadrature_log import sigma_plan
        geometry = sigma_plan()[0] or {}
        return dict(n_windows=int(geometry.get("n_windows", 0)),
                    window_tau_pairs=int(geometry.get("window_tau_pairs", 0)),
                    distinct_tau=int(geometry.get("distinct_tau_count", 0)),
                    windows=[(str(w["name"]), int(w["node_count"]))
                             for b in geometry.get("branches", ()) for w in b["windows"]],
                    held=bool(geometry.get("sc_fixed_quadrature", False)),
                    rebuilds=int(geometry.get("sc_fixed_rebuilds_this_iteration", 0)))

    def head_rows(efermi_ry, enk_l, pop_l, nb_sig):
        return None if head is None else replace(
            head, enk_ry=enk_l[:, :nb_sig], occupations=pop_l[:, :nb_sig],
            efermi_ry=float(efermi_ry))

    from common.collectives import gather_to_host
    block4 = _window_block_fn(mesh_xy, b_lo, n_a, 4)
    block3 = _window_block_fn(mesh_xy, b_lo, n_a, 3)
    first = lambda x: x.point(0) if hasattr(x, "point") else x
    Uh = np.conj(np.swapaxes(U, -1, -2))
    if rho_w is not None:
        out = _window_lines(
            rho_w, pairing, line_carrier=line_carrier, line_sweep=line_sweep, census=census,
            head_rows=head_rows, faces=(psi_nmu, psi_mun), b_lo=b_lo, n_a=n_a, nk=nk,
            enk_t=enk_t, occ=occ, pop=pop, plan=plan, mesh_xy=mesh_xy, block4=block4,
            block3=block3, first=first, static_limit=static_limit,
            static_kernels=lambda wfns_t, meta_t: _invalid_static_kernels(
                wfns_t, poles.Wc0_q if static_limit_wc0 is None else static_limit_wc0, True,
                meta_t, mesh_xy, None if spm else poles.q_wedge))
        S = 0.5 * (out["pair"][0] + out["pair"][1])
        out.update(onshell=U @ S @ Uh)
        p(f"  window Sigma_c: bands [{b_lo}, {b_hi}) on the Sigma window "
          f"[0, {out['carrier']['nb_sigma']}), max|Sigma_ii(e_i)| = "
          f"{float(np.max(np.abs(out['diag']))) * RYD_TO_EV:.6f} eV, branches kept, internal "
          f"bands {internal}, full-rho internal line ({pairing} pairing, "
          f"{len(out['sweep']['lines'])} sweeps), {out['sweep']['distinct_tau']} tau in "
          f"{out['sweep']['n_windows']} windows")
        return out
    line = line_carrier((psi_nmu, psi_mun))
    wfns_t, meta_t, slices_t, _ = line
    nb_sig = int(slices_t.nb_sigma)
    res, request = line_sweep(line, pop, None, internal_bands, plan, keep_branches=bool(branches))
    sweep = census()
    hr = head_rows(res.efermi_ry, enk_t, pop, nb_sig)
    # The pole executor returns a band-count cube (count 0: the window's sum); the
    # shared-pole executor (gw.mpa.sigma) its OnShellSigma directly.
    point = first(res.sigma_c_kij).with_head(hr)
    pair = np.asarray(gather_to_host(block4(point.pair)))
    diag = np.einsum("kii->ki", pair[0])
    S = 0.5 * (pair[0] + pair[1])
    out = dict(pair=pair, diag=diag, onshell=U @ S @ Uh, efermi_ry=float(res.efermi_ry),
               request=request, carrier=dict(slices=slices_t, nb_sigma=nb_sig),
               sweep=sweep, full_rho=None)
    if branches or static_limit:
        table = request.tables_ry[request.slot("pair")]
        hole, particle, head_static = (
            _head_branch_rows(hr, table, win) if hr is not None else
            (np.zeros((nk, n_a)),) * 3)
    if branches:
        from .mpa.sigma import BRANCH_SPACES
        tabs = {space: np.array(gather_to_host(block4(first(res.branches[space]).pair)))
                for space in BRANCH_SPACES}
        idx = np.arange(n_a)
        tabs["val"][:, :, idx, idx] += hole[None]
        tabs["cond"][:, :, idx, idx] += particle[None]
        term = getattr(res.sigma_c_kij, "term", None)
        tabs["static"] = (np.zeros((nk, n_a, n_a), np.complex128) if term is None else
                          np.asarray(gather_to_host(block3(term))))
        out["branch_pair"] = tabs
    if static_limit:
        from .cohsex_sigma import build_Gij
        sigma_sx, _, W_all = _invalid_static_kernels(
            wfns_t, poles.Wc0_q if static_limit_wc0 is None else static_limit_wc0, True,
            meta_t, mesh_xy, None if spm else poles.q_wedge)
        state = _population_state(pop if internal_bands is None else
                                  internal_bands.weights(pop), meta_t, pop.shape)
        sx = sigma_sx(wfns_t, build_Gij(meta_t, mesh_xy, state), W_all)
        out["static_limit"] = dict(sx=np.asarray(gather_to_host(block3(sx))), head=head_static)
    if omega_ry is not None:
        grid = np.asarray(omega_ry, np.float64)
        res_w, _ = line_sweep(line, pop, None, internal_bands, plan, omega_grid=grid)
        cube = first(res_w.sigma_c_kij)
        Sw = np.asarray(gather_to_host(block4(cube)))
        hr_w = head_rows(res_w.efermi_ry, enk_t, pop, nb_sig)
        if hr_w is not None:
            hw = replace(hr_w, omega_grid_ry=grid).band_rows(b_lo, b_hi)
            Sw = Sw + np.einsum("wki,ij->wkij", hw, np.eye(n_a))
        out["omega"] = U[None] @ Sw @ Uh[None]
    p(f"  window Sigma_c: bands [{b_lo}, {b_hi}) on the Sigma window [0, {nb_sig}), "
      f"max|Sigma_ii(e_i)| = {float(np.max(np.abs(diag))) * RYD_TO_EV:.6f} eV"
      + (" (control variant: frozen bands omitted from G)" if omit_frozen else "")
      + (", branches kept" if branches else "")
      + ("" if internal is None else f", internal bands {internal}")
      + f", {sweep['distinct_tau']} tau in {sweep['n_windows']} windows")
    return out
