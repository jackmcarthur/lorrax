"""Tier-0 fit-free full-frequency correlation-self-energy oracle.

This is the maintained form of the direct-SoS referee (sandbox claim 0363):
ordered-pair Adler--Wiser chi0, the production distributed Dyson solve,
canonical q-star unfold, and numerical contour deformation.  No MPA sample
or pole store is accepted by this module.  The route is deliberately an
O(N^4) oracle and is not eligible to become the production frequency model.

The large objects retain their natural two-dimensional mesh layout.  In
particular chi and W are ``P(None, 'x', 'y')``.  The incumbent diagonal
contour contraction is weighted before its mesh psum, so its replicated
result is one scalar per external target.  The selected-block seam below
instead builds one bounded external-frequency tile of the weighted centroid
operator and routes its band projection through
``common.contract_bands.contract_bands_block_reshard``.  That shared owner
reduce-scatters the square result directly to ``P(None, None, 'x', 'y')``;
the block route must never replace it with a replicated projection.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from .gw_config import INTERNAL_FF_CD_RESPONSE_WIDTH_EV


PAIR_TILE = 4
TARGET_TILE = 4
INTERMEDIATE_BAND_TILE = 4
EXTERNAL_FREQUENCY_TILE = 4
FREQUENCY_BATCH = 4
MATRIX_CHECKPOINT_FREQUENCIES = 16
IMAG_ORIGIN_LIMIT_EV = 1.0e-10
CENTER_SHIFT_EV = 1.0e-10
REAL_MAX_EV = 70.0
REAL_STEP_EV = 0.25
REAL_COARSE_STEP_EV = 0.50
REAL_HARD_MAX_EV = 250.0
IMAG_MAX_EV = 100.0
IMAG_MAP_SCALE_EV = 2.5
IMAG_FINE_INTERVALS = 64
CD_CONTROL_TOL_MEV = 0.5
RESPONSE_WIDTHS_EV = (INTERNAL_FF_CD_RESPONSE_WIDTH_EV,)

# Checkpoint compatibility follows numerical semantics, not incidental source
# bytes.  Bump this value whenever the ordered-pair response, Dyson/unfold, or
# weighted contour accumulator changes meaning.  Head-only, diagnostics, and
# comment changes deliberately do not invalidate an expensive body resume.
BODY_ACCUMULATOR_SEMANTIC_EPOCH = (
    "internal-ff-cd-body-v2:ordered-pair-dyson-qstar-contour")
CHECKPOINT_SCHEMA = 2
MATRIX_CHECKPOINT_SCHEMA = 1
ARRAY_RECEIPT_SCHEME = "numpy-c-order-sha256-v1"
STAGE_TIMING_KEYS = (
    "dyson_solve_wall_seconds",
    "q_unfold_wall_seconds",
    "contract_host_checks_wall_seconds",
)


@dataclass(frozen=True)
class InternalFFResult:
    sigma_c_diag_ev: np.ndarray | None
    efermi_ev: float
    artifact_path: str
    sigma_c_body_omega_ry: jax.Array | None = None
    head_sigma_diag_w_kn_ry: jax.Array | None = None
    sigma_band_axis: object | None = None


def real_grid(width_ev: float, *, max_ev: float = REAL_MAX_EV) -> np.ndarray:
    """Uniform residue line at the requested physical broadening.

    ``max_ev`` is normally derived from the run's complete target/intermediate
    energy-difference table.  The exported default exists only for small
    deterministic tests; it is not the production coverage decision.
    """
    width_ev = float(width_ev)
    if width_ev not in RESPONSE_WIDTHS_EV:
        raise ValueError(
            f"internal_ff_cd response width {width_ev} eV is not in the "
            f"fixed oracle policy {RESPONSE_WIDTHS_EV}")
    max_ev = float(max_ev)
    if not (np.isfinite(max_ev) and 0.0 < max_ev <= REAL_HARD_MAX_EV):
        raise ValueError(
            f"internal_ff_cd real coverage must lie in (0, "
            f"{REAL_HARD_MAX_EV}] eV, got {max_ev!r}")
    n = int(np.ceil(max_ev / REAL_STEP_EV - 1.0e-12))
    return REAL_STEP_EV * np.arange(n + 1, dtype=np.float64)


def imag_grid(n_intervals: int = IMAG_FINE_INTERVALS) -> np.ndarray:
    """Nested tangent-mapped imaginary grid including 0 and the tail edge."""
    n_intervals = int(n_intervals)
    if n_intervals <= 0:
        raise ValueError("internal_ff_cd imaginary intervals must be positive")
    theta_max = np.arctan(IMAG_MAX_EV / IMAG_MAP_SCALE_EV)
    theta = np.linspace(0.0, theta_max, n_intervals + 1)
    grid = IMAG_MAP_SCALE_EV * np.tan(theta)
    grid[0], grid[-1] = 0.0, IMAG_MAX_EV
    return grid


def _coarse_real_grid(fine: np.ndarray, width_ev: float) -> np.ndarray:
    del width_ev
    stride = int(round(REAL_COARSE_STEP_EV / REAL_STEP_EV))
    if stride <= 1 or not np.isclose(
            stride * REAL_STEP_EV, REAL_COARSE_STEP_EV):
        raise AssertionError("real CD fine/coarse spacings are not nested")
    idx = np.arange(0, fine.size, stride, dtype=np.int32)
    if idx[-1] != fine.size - 1:
        idx = np.append(idx, fine.size - 1).astype(np.int32)
    return idx


def _coarse_imag_grid(fine: np.ndarray) -> np.ndarray:
    if fine.size != IMAG_FINE_INTERVALS + 1:
        raise ValueError(
            "internal_ff_cd coarse imaginary certificate requires the "
            f"canonical {IMAG_FINE_INTERVALS + 1}-node fine grid")
    return np.arange(0, fine.size, 2, dtype=np.int32)


def _real_coverage_max(required_max_ev: float) -> float:
    """Return an interpolation-safe, coarse-grid-aligned real ceiling."""
    required_max_ev = float(required_max_ev)
    if not (np.isfinite(required_max_ev) and required_max_ev >= 0.0):
        raise ValueError(
            "internal_ff_cd residue coverage must be finite and nonnegative, "
            f"got {required_max_ev!r}")
    # Linear interpolation needs a node strictly above every residue energy.
    # Align that guard node to both the 0.25 eV fine and 0.50 eV control grids.
    max_ev = REAL_COARSE_STEP_EV * np.ceil(
        (required_max_ev + REAL_STEP_EV) / REAL_COARSE_STEP_EV)
    if max_ev > REAL_HARD_MAX_EV + 1.0e-12:
        raise ValueError(
            "internal_ff_cd interpolation-safe real W coverage would require "
            f"{max_ev:.6f} eV for a {required_max_ev:.6f} eV residue, beyond "
            f"the explicit {REAL_HARD_MAX_EV:.6f} eV oracle guard")
    return float(max_ev)


def _control_certificate(*, real_fine: np.ndarray,
                         real_coarse: np.ndarray,
                         imag_fine: np.ndarray,
                         imag_coarse: np.ndarray,
                         imag_tail: np.ndarray,
                         tolerance_mev: float = CD_CONTROL_TOL_MEV):
    """Certify each quadrature control separately, without cancellation."""
    deltas = {
        "real_fine_minus_coarse_ev": np.asarray(real_fine - real_coarse),
        "imag_fine_minus_coarse_ev": np.asarray(imag_fine - imag_coarse),
        "imag_full_minus_tail_ev": np.asarray(imag_fine - imag_tail),
    }
    component_abs_mev = np.stack([
        1000.0 * np.abs(delta.real) for delta in deltas.values()
    ] + [
        1000.0 * np.abs(delta.imag) for delta in deltas.values()
    ])
    worst_mev = np.max(component_abs_mev, axis=0)
    resolved = worst_mev <= float(tolerance_mev)
    return deltas, worst_mev, resolved


def _panel_bounds(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.empty_like(grid)
    ends = np.empty_like(grid)
    starts[0] = 0.0
    ends[0] = 0.5 * (grid[0] + grid[1])
    starts[1:] = 0.5 * (grid[:-1] + grid[1:])
    ends[1:-1] = 0.5 * (grid[1:-1] + grid[2:])
    ends[-1] = 0.5 * (-grid[-2] + 3.0 * grid[-1])
    return starts, ends


def _direct_pair_scan(psi_x_a, psi_y_a, psi_x_b, psi_y_b,
                      energy_a, energy_b, occ_a, occ_b, surface_a,
                      surface_b, z_values, *, nb_logical: int, tile: int):
    """Exact referee ordered-pair Adler--Wiser sum."""
    nk, nspinor, nmu_x, nb = psi_x_a.shape
    nmu_y = psi_y_a.shape[2]
    nb_pad = ((int(nb) + tile - 1) // tile) * tile
    pad = nb_pad - int(nb)
    pad4 = ((0, 0), (0, 0), (0, 0), (0, pad))
    pad2 = ((0, 0), (0, pad))
    pa_x_full, pb_x_full = jnp.pad(psi_x_a, pad4), jnp.pad(psi_x_b, pad4)
    pa_y_full, pb_y_full = jnp.pad(psi_y_a, pad4), jnp.pad(psi_y_b, pad4)
    ea_full, eb_full = jnp.pad(energy_a, pad2), jnp.pad(energy_b, pad2)
    fa_full, fb_full = jnp.pad(occ_a, pad2), jnp.pad(occ_b, pad2)
    sa_full, sb_full = jnp.pad(surface_a, pad2), jnp.pad(surface_b, pad2)
    z = jnp.asarray(z_values, jnp.complex128)
    ntiles = nb_pad // tile

    def pair_tile(accumulator, flat_index):
        ia = (flat_index // ntiles) * tile
        ib = (flat_index % ntiles) * tile
        pxa = jax.lax.dynamic_slice(pa_x_full, (0, 0, 0, ia),
                                    (nk, nspinor, nmu_x, tile))
        pxb = jax.lax.dynamic_slice(pb_x_full, (0, 0, 0, ib),
                                    (nk, nspinor, nmu_x, tile))
        pya = jax.lax.dynamic_slice(pa_y_full, (0, 0, 0, ia),
                                    (nk, nspinor, nmu_y, tile))
        pyb = jax.lax.dynamic_slice(pb_y_full, (0, 0, 0, ib),
                                    (nk, nspinor, nmu_y, tile))
        ea = jax.lax.dynamic_slice(ea_full, (0, ia), (nk, tile))
        eb = jax.lax.dynamic_slice(eb_full, (0, ib), (nk, tile))
        fa = jax.lax.dynamic_slice(fa_full, (0, ia), (nk, tile))
        fb = jax.lax.dynamic_slice(fb_full, (0, ib), (nk, tile))
        sa = jax.lax.dynamic_slice(sa_full, (0, ia), (nk, tile))
        sb = jax.lax.dynamic_slice(sb_full, (0, ib), (nk, tile))
        de = ea[:, :, None] - eb[:, None, :]
        df = fa[:, :, None] - fb[:, None, :]
        scale = jnp.maximum(1.0, jnp.maximum(abs(ea[:, :, None]),
                                             abs(eb[:, None, :])))
        separated = abs(de) > 64.0 * jnp.finfo(jnp.float64).eps * scale
        static = jnp.where(
            separated, df / jnp.where(separated, de, 1.0),
            -0.5 * (sa[:, :, None] + sb[:, None, :]))
        dynamic = df[None] / (de[None] + z[:, None, None, None])
        weights = jnp.where((abs(z) <= 1.0e-30)[:, None, None, None],
                            static[None], dynamic)
        ga, gb = ia + jnp.arange(tile), ib + jnp.arange(tile)
        logical = ((ga[:, None] < int(nb_logical))
                   & (gb[None, :] < int(nb_logical)))[None, None]
        weights = jnp.where(logical, weights, 0.0)
        dx = jnp.einsum("ksma,ksmb->kmab", pxa, jnp.conj(pxb), optimize=True)
        dy = jnp.einsum("ksna,ksnb->knab", pya, jnp.conj(pyb), optimize=True)
        add = jnp.einsum("zkab,kmab,knab->zmn", weights, dx,
                         jnp.conj(dy), optimize=True)
        return accumulator + add, None

    zero = jnp.zeros((z.size, nmu_x, nmu_y), jnp.complex128)
    chi, _ = jax.lax.scan(pair_tile, zero,
                          jnp.arange(ntiles * ntiles), unroll=1)
    return chi / jnp.sqrt(jnp.asarray(nk, jnp.float64))


def make_direct_kernel(mesh: Mesh, *, nb_logical: int, tile: int = PAIR_TILE):
    from common.shard_map import shard_map
    from .wavefunction_bundle import PSI_XN_SPEC, PSI_YN_SPEC

    def local(psi_xn, psi_yn, kminusq, energies, occupations, surface, z):
        pbx, pby = jnp.take(psi_xn, kminusq, axis=0), jnp.take(psi_yn, kminusq, axis=0)
        return _direct_pair_scan(
            psi_xn, psi_yn, pbx, pby, energies, jnp.take(energies, kminusq, axis=0),
            occupations, jnp.take(occupations, kminusq, axis=0), surface,
            jnp.take(surface, kminusq, axis=0), z,
            nb_logical=nb_logical, tile=tile)

    return jax.jit(shard_map(
        local, mesh=mesh,
        in_specs=(PSI_XN_SPEC, PSI_YN_SPEC, P(None), P(None, None),
                  P(None, None), P(None, None), P(None)),
        out_specs=P(None, "x", "y"), check_vma=False))


def make_weighted_contract_kernel(mesh: Mesh, *, n_targets: int,
                                  inner_stop: int,
                                  tile: int = TARGET_TILE):
    """Contract and reduce one W frequency without a spectral-history cube."""
    from common.shard_map import shard_map
    from .wavefunction_bundle import PSI_XN_SPEC, PSI_YN_SPEC

    n_pad = ((int(n_targets) + tile - 1) // tile) * tile
    ntiles = n_pad // tile

    def local(psi_xn, psi_yn, wc, target_k, target_b, kmq, coeff_flat):
        nb = int(psi_xn.shape[-1])
        # Coefficients have no spatial/centroid axes.  Keep that distinction
        # visible to sharding audits by transporting the (target,q*band)
        # table as a replicated rank-2 scalar table, then exposing q/band
        # only inside this local contraction.
        coeff = coeff_flat.reshape(n_targets, kmq.shape[1], nb)
        tk = jnp.pad(target_k, (0, n_pad - n_targets))
        tb = jnp.pad(target_b, (0, n_pad - n_targets))
        kmap = jnp.pad(kmq, ((0, n_pad - n_targets), (0, 0)))
        cfull = jnp.pad(coeff, ((0, n_pad - n_targets), (0, 0), (0, 0)))
        valid = jnp.arange(n_pad) < n_targets
        out0 = jnp.zeros((n_pad,), jnp.complex128)

        def target_tile(out, it):
            lo = it * tile
            tki = jax.lax.dynamic_slice(tk, (lo,), (tile,))
            tbi = jax.lax.dynamic_slice(tb, (lo,), (tile,))
            kmi = jax.lax.dynamic_slice(kmap, (lo, 0), (tile, kmq.shape[1]))
            ci = jax.lax.dynamic_slice(cfull, (lo, 0, 0),
                                       (tile, coeff.shape[1], coeff.shape[2]))
            vi = jax.lax.dynamic_slice(valid, (lo,), (tile,))
            tx_rows = jnp.take(psi_xn, tki, axis=0)
            ty_rows = jnp.take(psi_yn, tki, axis=0)
            tx = jax.vmap(lambda row, band: row[:, :, band])(tx_rows, tbi)
            ty = jax.vmap(lambda row, band: row[:, :, band])(ty_rows, tbi)
            # The k-q gather retains one target and one q axis while the
            # band sum remains the innermost logical axis.
            ix = jnp.take(psi_xn, kmi.reshape(-1), axis=0).reshape(
                tile, kmi.shape[1], psi_xn.shape[1], psi_xn.shape[2], nb)
            iy = jnp.take(psi_yn, kmi.reshape(-1), axis=0).reshape(
                tile, kmi.shape[1], psi_yn.shape[1], psi_yn.shape[2], nb)
            dx = jnp.einsum("tsu,tqsum->tqmu", jnp.conj(tx), ix, optimize=True)
            dy = jnp.einsum("tsu,tqsum->tqmu", jnp.conj(ty), iy, optimize=True)
            logical_n = (jnp.arange(nb) < int(inner_stop))[None, None, :]
            ci = jnp.where(logical_n, ci, 0.0)
            partial = jnp.einsum(
                "tqn,tqna,qab,tqnb->t", ci, dx, wc, jnp.conj(dy),
                optimize=True)
            total = jax.lax.psum(partial, ("x", "y"))
            total = jnp.where(vi, total, 0.0)
            return jax.lax.dynamic_update_slice(out, total, (lo,)), None

        out, _ = jax.lax.scan(target_tile, out0, jnp.arange(ntiles), unroll=1)
        return out[:n_targets]

    mapped = jax.jit(shard_map(
        local, mesh=mesh,
        in_specs=(PSI_XN_SPEC, PSI_YN_SPEC, P(None, "x", "y"),
                  P(None), P(None), P(None, None), P(None, None)),
        out_specs=P(None), check_vma=False))

    def apply(psi_xn, psi_yn, wc, target_k, target_b, kmq, coeff):
        if coeff.ndim != 3:
            raise ValueError(
                f"weighted contour coefficients must be (target,q,band), "
                f"got {coeff.shape}")
        return mapped(
            psi_xn, psi_yn, wc, target_k, target_b, kmq,
            coeff.reshape(coeff.shape[0], coeff.shape[1] * coeff.shape[2]))

    return apply


# Descriptor fields shared by the host quadrature planner and the device
# weighted-operator builder.  A fixed-width numeric row is intentional: one
# compiled block kernel serves real fine/coarse and imaginary fine/coarse/tail
# rules without a Python callback or a second contraction pathway.
_RULE_KIND = 0
_RULE_LEFT = 1
_RULE_CENTER = 2
_RULE_RIGHT = 3
_RULE_PANEL_START = 4
_RULE_PANEL_END = 5
_RULE_HAS_LEFT = 6
_RULE_HAS_RIGHT = 7
_RULE_SIZE = 8
_RULE_REAL = 0
_RULE_IMAGINARY = 1


def _real_node_descriptor(grid: np.ndarray, iw: int) -> np.ndarray:
    """Return the device rule for one piecewise-linear real-axis node."""
    grid = np.asarray(grid, dtype=np.float64)
    iw = int(iw)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("real contour grid must be strictly increasing")
    if not 0 <= iw < grid.size:
        raise IndexError(f"real contour node {iw} outside [0,{grid.size})")
    row = np.zeros(_RULE_SIZE, dtype=np.float64)
    row[_RULE_KIND] = _RULE_REAL
    row[_RULE_CENTER] = grid[iw]
    row[_RULE_LEFT] = grid[iw - 1] if iw else grid[iw]
    row[_RULE_RIGHT] = grid[iw + 1] if iw + 1 < grid.size else grid[iw]
    row[_RULE_HAS_LEFT] = float(iw > 0)
    row[_RULE_HAS_RIGHT] = float(iw + 1 < grid.size)
    return row


def _imaginary_node_descriptor(grid: np.ndarray, iw: int) -> np.ndarray:
    """Return the exact incumbent atan-panel rule for one imaginary node."""
    grid = np.asarray(grid, dtype=np.float64)
    iw = int(iw)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("imaginary contour grid must be strictly increasing")
    if not 0 <= iw < grid.size:
        raise IndexError(f"imaginary contour node {iw} outside [0,{grid.size})")
    starts, ends = _panel_bounds(grid)
    row = np.zeros(_RULE_SIZE, dtype=np.float64)
    row[_RULE_KIND] = _RULE_IMAGINARY
    row[_RULE_PANEL_START] = starts[iw]
    row[_RULE_PANEL_END] = ends[iw]
    return row


def _device_contour_weight(rule, x_signed, occupations):
    """Evaluate one contour basis function on a bounded state tile."""
    x_abs = jnp.abs(x_signed)

    def real_weight(_):
        left = rule[_RULE_LEFT]
        center = rule[_RULE_CENTER]
        right = rule[_RULE_RIGHT]
        has_left = rule[_RULE_HAS_LEFT] > 0.5
        has_right = rule[_RULE_HAS_RIGHT] > 0.5
        left_den = jnp.where(has_left, center - left, 1.0)
        right_den = jnp.where(has_right, right - center, 1.0)
        left_arm = jnp.where(
            has_left & (x_abs >= left) & (x_abs <= center),
            (x_abs - left) / left_den, 0.0)
        right_arm = jnp.where(
            has_right & (x_abs >= center) & (x_abs <= right),
            (right - x_abs) / right_den, 0.0)
        hat = jnp.maximum(left_arm, right_arm)
        residue_sign = jnp.where(
            x_signed >= 0.0, -(1.0 - occupations), occupations)
        return residue_sign * hat

    def imaginary_weight(_):
        return (
            jnp.arctan(rule[_RULE_PANEL_END] / x_signed)
            - jnp.arctan(rule[_RULE_PANEL_START] / x_signed)
        ) / jnp.pi

    return jax.lax.cond(
        jnp.asarray(rule[_RULE_KIND], jnp.int32) == _RULE_REAL,
        real_weight, imaginary_weight, operand=None)


def _make_weighted_block_operator_kernel(
        mesh: Mesh, *, n_target_k: int, inner_stop: int,
        omega_tile: int = EXTERNAL_FREQUENCY_TILE,
        inner_tile: int = INTERMEDIATE_BAND_TILE):
    """Build one weighted centroid-operator tile for selected-k blocks.

    For one already-resident full-q ``Wc(z_j)`` this forms

    ``O[e,k,s,mu,s',nu] = sum_q,l c_j(e,k,q,l) psi_l psi_l^* Wc_j``.

    The external-frequency extent is a fixed small tile and the intermediate
    band sum is a ``lax.scan`` over ``inner_tile`` states.  The result retains
    ``P(None,None,None,'x',None,'y')``; its only intended consumer is the
    canonical band projector.  No chi, Dyson, q-star, or symmetry logic lives
    here.
    """
    from common.shard_map import shard_map
    from .wavefunction_bundle import PSI_XN_SPEC, PSI_YN_SPEC

    n_target_k = int(n_target_k)
    inner_stop = int(inner_stop)
    omega_tile = int(omega_tile)
    inner_tile = int(inner_tile)
    if min(n_target_k, inner_stop, omega_tile, inner_tile) <= 0:
        raise ValueError(
            "weighted block operator extents and tiles must be positive")
    inner_pad = ((inner_stop + inner_tile - 1) // inner_tile) * inner_tile
    n_inner_tiles = inner_pad // inner_tile

    def local(psi_xn, psi_yn, wc, kmq, energies_ev, occupations,
              omega_abs_ev, omega_valid, rule):
        nb = int(psi_xn.shape[-1])
        if inner_stop > nb:
            raise ValueError(
                f"weighted block inner_stop={inner_stop} exceeds "
                f"wavefunction carrier {nb}")
        band_pad = inner_pad - nb
        if band_pad > 0:
            psi_x = jnp.pad(psi_xn, ((0, 0), (0, 0), (0, 0),
                                     (0, band_pad)))
            psi_y = jnp.pad(psi_yn, ((0, 0), (0, 0), (0, 0),
                                     (0, band_pad)))
            energy = jnp.pad(energies_ev, ((0, 0), (0, band_pad)))
            occ = jnp.pad(occupations, ((0, 0), (0, band_pad)))
        else:
            psi_x, psi_y = psi_xn, psi_yn
            energy, occ = energies_ev, occupations

        nspin = int(psi_x.shape[1])
        nmu_x, nmu_y = int(psi_x.shape[2]), int(psi_y.shape[2])
        nq = int(wc.shape[0])
        zero = jnp.zeros(
            (omega_tile, n_target_k, nspin, nmu_x, nspin, nmu_y),
            dtype=jnp.complex128)

        def add_state_tile(accumulator, flat_index):
            iq = flat_index // n_inner_tiles
            il = (flat_index % n_inner_tiles) * inner_tile
            i0 = jnp.asarray(0, dtype=il.dtype)
            kmi = jax.lax.dynamic_index_in_dim(
                kmq, iq, axis=1, keepdims=False)
            px = jnp.take(psi_x, kmi, axis=0)
            py = jnp.take(psi_y, kmi, axis=0)
            px = jax.lax.dynamic_slice(
                px, (i0, i0, i0, il),
                (n_target_k, nspin, nmu_x, inner_tile))
            py = jax.lax.dynamic_slice(
                py, (i0, i0, i0, il),
                (n_target_k, nspin, nmu_y, inner_tile))
            ek = jnp.take(energy, kmi, axis=0)
            fk = jnp.take(occ, kmi, axis=0)
            ek = jax.lax.dynamic_slice(
                ek, (i0, il), (n_target_k, inner_tile))
            fk = jax.lax.dynamic_slice(
                fk, (i0, il), (n_target_k, inner_tile))
            x_signed = (
                omega_abs_ev[:, None, None] - ek[None, :, :]
                + CENTER_SHIFT_EV)
            coeff = _device_contour_weight(rule, x_signed, fk[None, :, :])
            logical_l = (
                il + jnp.arange(inner_tile, dtype=jnp.int32) < inner_stop)
            coeff = jnp.where(
                omega_valid[:, None, None] & logical_l[None, None, :],
                coeff, 0.0)
            density = jnp.einsum(
                "ksal,ktbl,ekl->eksatb", px, jnp.conj(py), coeff,
                optimize=True)
            wq = jax.lax.dynamic_index_in_dim(
                wc, iq, axis=0, keepdims=False)
            return accumulator + density * wq[None, None, None, :, None, :], None

        result, _ = jax.lax.scan(
            add_state_tile, zero,
            jnp.arange(nq * n_inner_tiles, dtype=jnp.int32), unroll=1)
        return result

    return jax.jit(shard_map(
        local, mesh=mesh,
        in_specs=(PSI_XN_SPEC, PSI_YN_SPEC, P(None, "x", "y"),
                  P(None, None), P(None, None), P(None, None), P(None),
                  P(None), P(None)),
        out_specs=P(None, None, None, "x", None, "y"),
        check_vma=False))


def _make_weighted_block_contract_kernel(
        mesh: Mesh, *, n_target_k: int, inner_stop: int,
        omega_tile: int = EXTERNAL_FREQUENCY_TILE,
        inner_tile: int = INTERMEDIATE_BAND_TILE):
    """Compose the one weighted-operator builder with the shared projector.

    ``psi_left`` and ``psi_right`` are the already-selected, zero-padded
    square Sigma target window.  The returned tile is
    ``(omega_tile,k,m,n)`` at ``P(None,None,'x','y')``.  The two-stage
    reduce-scatter and its divisibility checks remain owned solely by
    :mod:`common.contract_bands`.
    """
    from common.contract_bands import contract_bands_block_reshard

    build_operator = _make_weighted_block_operator_kernel(
        mesh, n_target_k=n_target_k, inner_stop=inner_stop,
        omega_tile=omega_tile, inner_tile=inner_tile)
    project = contract_bands_block_reshard(mesh, extra="leading")

    @jax.jit
    def apply(psi_xn, psi_yn, psi_left, psi_right, wc, kmq,
              energies_ev, occupations, omega_abs_ev, omega_valid, rule):
        operator = build_operator(
            psi_xn, psi_yn, wc, kmq, energies_ev, occupations,
            omega_abs_ev, omega_valid, rule)
        return project(psi_left, operator, psi_right)

    return apply


def _make_matrix_accumulator_kernels(
        mesh: Mesh, *, matrix_shape: tuple[int, int, int, int],
        head_shape: tuple[int, int, int]):
    """Create distributed zero/add/pin operations for one selected cube."""
    matrix_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    # Shard the analytic diagonal by its one band axis.  Replication over y
    # makes the x-owned band block available on the diagonal matrix rank
    # (x == y) without ever replicating the whole O(nomega*nk*nb) carrier.
    head_sharding = NamedSharding(mesh, P(None, None, "x"))

    make_matrix_zero = jax.jit(
        lambda: jnp.zeros(matrix_shape, dtype=jnp.complex128),
        out_shardings=matrix_sharding)
    make_head_zero = jax.jit(
        lambda: jnp.zeros(head_shape, dtype=jnp.complex128),
        out_shardings=head_sharding)

    @jax.jit
    def add_matrix_tile(accumulator, tile, omega_lo):
        zero = jnp.asarray(0, dtype=omega_lo.dtype)
        start = (omega_lo, zero, zero, zero)
        return jax.lax.dynamic_update_slice(
            accumulator,
            jax.lax.dynamic_slice(
                accumulator, start, tile.shape) + tile,
            start)

    add_matrix_tile = jax.jit(
        add_matrix_tile, out_shardings=matrix_sharding)

    @jax.jit
    def add_head_sample(accumulator, omega_abs_ev, omega_valid,
                        energies_ev, occupations, wc0_ry, rule,
                        inverse_volume_nk):
        x_signed = (
            omega_abs_ev[:, None, None] - energies_ev[None, :, :]
            + CENTER_SHIFT_EV)
        coefficient = _device_contour_weight(
            rule, x_signed, occupations[None, :, :])
        coefficient = jnp.where(
            omega_valid[:, None, None], coefficient, 0.0)
        # The body path consumes S=-Wc.  The Gamma scalar follows the same
        # convention and differs only by its analytic volume normalization.
        return accumulator - coefficient * wc0_ry * inverse_volume_nk

    add_head_sample = jax.jit(
        add_head_sample, out_shardings=head_sharding)

    @jax.jit
    def pin_head_on_shell(head, omega_rel_ev, energy_rel_ev,
                          static_head_ry, band_valid):
        """Minimum-norm two-node correction reproducing eta=0 on shell."""
        n_omega = int(head.shape[0])
        hi = jnp.searchsorted(omega_rel_ev, energy_rel_ev, side="right")
        hi = jnp.clip(hi, 1, n_omega - 1)
        lo = hi - 1
        w_hi = ((energy_rel_ev - omega_rel_ev[lo])
                / (omega_rel_ev[hi] - omega_rel_ev[lo]))
        w_hi = jnp.where(energy_rel_ev <= omega_rel_ev[0], 0.0, w_hi)
        w_hi = jnp.where(energy_rel_ev >= omega_rel_ev[-1], 1.0, w_hi)
        w_lo = 1.0 - w_hi
        current = (
            jnp.take_along_axis(head, lo[None], axis=0)[0] * w_lo
            + jnp.take_along_axis(head, hi[None], axis=0)[0] * w_hi)
        delta = jnp.where(band_valid[None], static_head_ry - current, 0.0)
        denom = jnp.maximum(w_lo * w_lo + w_hi * w_hi,
                            jnp.finfo(jnp.float64).tiny)
        omega_index = jnp.arange(n_omega)[:, None, None]
        correction = delta[None] * (
            jnp.where(omega_index == lo[None], w_lo[None] / denom[None], 0.0)
            + jnp.where(omega_index == hi[None],
                        w_hi[None] / denom[None], 0.0))
        return head + correction

    pin_head_on_shell = jax.jit(
        pin_head_on_shell, out_shardings=head_sharding)
    return (make_matrix_zero, make_head_zero, add_matrix_tile,
            add_head_sample, pin_head_on_shell)


def _real_coefficients(grid: np.ndarray, iw: int, x_abs: np.ndarray,
                       sign: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid, x_abs, side="right") - 1
    idx = np.clip(idx, 0, grid.size - 2)
    frac = (x_abs - grid[idx]) / (grid[idx + 1] - grid[idx])
    return sign * (np.where(idx == iw, 1.0 - frac, 0.0)
                   + np.where(idx + 1 == iw, frac, 0.0))


def _imag_coefficients(grid: np.ndarray, iw: int, x_signed: np.ndarray) -> np.ndarray:
    starts, ends = _panel_bounds(grid)
    return (np.arctan(ends[iw] / x_signed)
            - np.arctan(starts[iw] / x_signed)) / np.pi


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _array_receipt(array, *, dtype=None) -> dict:
    """Compact exact receipt for a replicated host array; never gathers JAX."""
    values = np.asarray(array, dtype=dtype)
    if values.dtype.hasobject:
        raise TypeError("internal_ff_cd cannot receipt an object array")
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(ARRAY_RECEIPT_SCHEME.encode("ascii"))
    digest.update(b"\0")
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.view(np.uint8))
    return {
        "scheme": ARRAY_RECEIPT_SCHEME,
        "dtype": values.dtype.str,
        "shape": [int(v) for v in values.shape],
        "sha256": digest.hexdigest(),
    }


def _require_charge_zeta_identity(receipt) -> dict:
    if not isinstance(receipt, dict) or set(receipt) != {"scheme", "digest"}:
        raise ValueError(
            "internal_ff_cd requires the canonical two-field "
            "charge_zeta_identity receipt; legacy/unstamped ISDF state "
            "cannot support an exact body checkpoint")
    out = {"scheme": str(receipt["scheme"]),
           "digest": str(receipt["digest"])}
    if not out["scheme"] or not out["digest"]:
        raise ValueError(
            "internal_ff_cd charge_zeta_identity fields must be nonempty")
    return out


def _require_coulomb_policy_receipt(receipt) -> str:
    from file_io import COULOMB_POLICY_KEYS, format_coulomb_policy

    if not isinstance(receipt, dict):
        raise ValueError(
            "internal_ff_cd requires the Coulomb policy stamped with the "
            "resident V; a legacy/unstamped restart cannot resume exactly")
    expected, actual = set(COULOMB_POLICY_KEYS), set(receipt)
    if actual != expected:
        raise ValueError(
            "internal_ff_cd Coulomb receipt keys differ from the current "
            f"owner schema: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}")
    return format_coulomb_policy(receipt)


def _body_checkpoint_provenance(
        *, energies_ry, state, charge_zeta_identity,
        coulomb_policy_receipt, q_mapping, centroid_indices,
        nb_chi_logical, nb_sigma_logical, band_carrier_storage,
        b0, V_q, nmu_logical, nk, mesh_xy) -> dict:
    """Receipts for every available numerical input to the body accumulator."""
    nb_physical = max(int(nb_chi_logical), int(nb_sigma_logical))
    energies = np.asarray(energies_ry, np.float64)
    occupations = np.asarray(state.f_kn, np.float64)
    if energies.shape[0] != int(nk) or occupations.shape != energies.shape:
        raise ValueError(
            "internal_ff_cd energy/occupation carrier mismatch while "
            f"building checkpoint provenance: {energies.shape} vs "
            f"{occupations.shape}, nk={nk}")
    if energies.shape[1] != int(band_carrier_storage):
        raise ValueError(
            "internal_ff_cd energy carrier changed before checkpoint "
            f"provenance: {energies.shape[1]} != {band_carrier_storage}")
    q_receipts = {
        str(name): _array_receipt(values)
        for name, values in sorted(q_mapping.items())
    }
    return {
        "body_accumulator_semantic_epoch": BODY_ACCUMULATOR_SEMANTIC_EPOCH,
        "bands": {
            "b0": int(b0),
            "number_bands_chi": int(nb_chi_logical),
            "number_bands_sigma": int(nb_sigma_logical),
            # This is a carrier compatibility gate, not a physical count.
            "band_carrier_storage": int(band_carrier_storage),
        },
        "energies_ry": _array_receipt(
            energies[:, :nb_physical], dtype=np.float64),
        "occupation": {
            "family": str(state.smearing_family),
            "width_ry": float(state.smearing_width_ry),
            "mu_ry": float(state.mu_ry),
            "owner_hash": str(state.occ_hash),
            "physical_f_kn": _array_receipt(
                occupations[:, :nb_physical], dtype=np.float64),
        },
        "charge_zeta_identity": _require_charge_zeta_identity(
            charge_zeta_identity),
        "centroid_indices": _array_receipt(
            centroid_indices, dtype=np.int64),
        "q_mapping": q_receipts,
        "coulomb": {
            "construction_policy": _require_coulomb_policy_receipt(
                coulomb_policy_receipt),
            # V is distributed and is deliberately neither gathered nor
            # duplicated for hashing.  The construction receipts above bind
            # its zeta/source and policy; these fields bind its live carrier.
            "shape": [int(v) for v in V_q.shape],
            "dtype": str(V_q.dtype),
            "nmu_logical": int(nmu_logical),
        },
        # Reduction topology affects floating-point order.  Even equal carrier
        # extents therefore cannot move a checkpoint between P4 and P16.
        "mesh_shape": {
            str(axis): int(mesh_xy.shape[axis]) for axis in mesh_xy.axis_names
        },
        "nk": int(nk),
        "algorithm_parameters": {
            "pair_tile": PAIR_TILE,
            "target_tile": TARGET_TILE,
            "frequency_batch": FREQUENCY_BATCH,
            "center_shift_ev": CENTER_SHIFT_EV,
            "imag_origin_limit_ev": IMAG_ORIGIN_LIMIT_EV,
        },
    }


def _checkpoint_identity(*, kind, grid, width_ev, target_k, target_b,
                         body_provenance, w_observer_identity=None):
    identity = {
        "schema": CHECKPOINT_SCHEMA,
        "route": "internal_ff_cd",
        "kind": str(kind),
        "grid": _array_receipt(grid, dtype=np.float64),
        "eta_w_ev": None if width_ev is None else float(width_ev),
        "target_k": _array_receipt(target_k, dtype=np.int32),
        "target_b": _array_receipt(target_b, dtype=np.int32),
        "body_provenance": body_provenance,
    }
    if w_observer_identity is not None:
        identity["w_observer_identity"] = str(w_observer_identity)
    return identity


def _matrix_output_identity(identity: dict, *, omega_rel_ev,
                            sigma_band_axis, nk: int) -> dict:
    """Bind the scalar W-prefix identity to its selected output carrier."""
    out = dict(identity)
    out.update({
        "matrix_checkpoint_schema": MATRIX_CHECKPOINT_SCHEMA,
        "output": "selected_matrix_block",
        "omega_rel_ev": _array_receipt(omega_rel_ev, dtype=np.float64),
        "full_bz_k_rows": int(nk),
        "sigma_band_axis": {
            "logical": int(sigma_band_axis.logical),
            "carrier": int(sigma_band_axis.carrier),
            "partition": "P(None,None,x,y)",
        },
        "external_frequency_tile": EXTERNAL_FREQUENCY_TILE,
        "intermediate_band_tile": INTERMEDIATE_BAND_TILE,
    })
    return out


def _load_checkpoint(path: Path, identity, n_targets, n_accumulators, *,
                     return_stage_timings: bool = False):
    empty_stages = {key: 0.0 for key in STAGE_TIMING_KEYS}
    if not path.exists():
        result = (0, [np.zeros(n_targets, np.complex128)
                      for _ in range(n_accumulators)], 0.0, 0.0)
        return result + (empty_stages,) if return_stage_timings else result
    with np.load(path, allow_pickle=False) as data:
        stamped = json.loads(str(np.asarray(data["identity_json"])[()]))
        if stamped != identity:
            old_schema = stamped.get("schema", "absent")
            raise ValueError(
                f"internal_ff_cd checkpoint {path} has stale identity; "
                f"checkpoint schema={old_schema!r}, required schema="
                f"{identity.get('schema')!r}. Delete or move the incomplete "
                "run variant rather than mixing numerical semantics, grids, "
                "occupations, Coulomb/ISDF receipts, q maps, band carriers, "
                "or targets. Automatic prefix migration is intentionally "
                "unavailable without an old-grid prefix and exact-zero "
                "padded-orbital proof.")
        completed = int(np.asarray(data["completed"])[()])
        accum = np.asarray(data["accumulators"], np.complex128)
        if accum.shape != (n_accumulators, n_targets):
            raise ValueError(
                f"internal_ff_cd checkpoint {path} accumulator shape "
                f"{accum.shape} != {(n_accumulators, n_targets)}")
        grid_n = int(identity["grid_n"])
        if completed < 0 or completed > grid_n or (
                completed % FREQUENCY_BATCH != 0 and completed != grid_n):
            raise ValueError(
                f"internal_ff_cd checkpoint {path} has invalid completed="
                f"{completed}")
        if not np.all(np.isfinite(accum)):
            raise ValueError(
                f"internal_ff_cd checkpoint {path} has nonfinite accumulator")
        result = (completed, [row.copy() for row in accum],
                  float(np.asarray(data["chi_wall_seconds"])[()]),
                  float(np.asarray(data["solve_contract_wall_seconds"])[()]))
        if not all(np.isfinite(value) and value >= 0.0 for value in result[2:]):
            raise ValueError(
                f"internal_ff_cd checkpoint {path} has invalid aggregate "
                "wall timing")
        missing_stages = [key for key in STAGE_TIMING_KEYS if key not in data]
        if return_stage_timings and missing_stages:
            raise ValueError(
                f"internal_ff_cd checkpoint {path} lacks schema-2 stage "
                f"timings {missing_stages}")
        stages = {
            key: (float(np.asarray(data[key])[()]) if key in data else 0.0)
            for key in STAGE_TIMING_KEYS
        }
        if not all(np.isfinite(value) and value >= 0.0
                   for value in stages.values()):
            raise ValueError(
                f"internal_ff_cd checkpoint {path} has invalid stage timing")
        return result + (stages,) if return_stage_timings else result


def _save_checkpoint(path: Path, identity, completed, accumulators,
                     chi_wall, solve_contract_wall, *, stage_timings=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    stages = ({key: 0.0 for key in STAGE_TIMING_KEYS}
              if stage_timings is None else {
                  key: float(stage_timings[key]) for key in STAGE_TIMING_KEYS
              })
    with tmp.open("wb") as stream:
        np.savez(
            stream,
            identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
            completed=np.asarray(int(completed), np.int64),
            accumulators=np.stack(accumulators),
            chi_wall_seconds=np.asarray(float(chi_wall), np.float64),
            solve_contract_wall_seconds=np.asarray(
                float(solve_contract_wall), np.float64),
            **{key: np.asarray(value, np.float64)
               for key, value in stages.items()})
    os.replace(tmp, path)


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _matrix_checkpoint_paths(base_path: Path) -> tuple[Path, tuple[Path, Path]]:
    base = Path(base_path)
    return (
        Path(str(base) + ".current.json"),
        (Path(str(base) + ".slot0.h5"), Path(str(base) + ".slot1.h5")),
    )


def _read_matrix_checkpoint_pointer(base_path: Path):
    """Read one atomic identity pointer and require all hosts saw one image."""
    from common.collectives import all_gather_processes

    pointer_path, _ = _matrix_checkpoint_paths(base_path)
    raw = pointer_path.read_bytes() if pointer_path.exists() else b""
    gathered = np.asarray(all_gather_processes(
        np.frombuffer(hashlib.sha256(raw).digest(), dtype=np.uint8)),
        dtype=np.uint8)
    if np.any(gathered != gathered[:1]):
        raise RuntimeError(
            "internal_ff_cd matrix checkpoint pointer changed while ranks "
            f"were opening it: {pointer_path}")
    if not raw:
        return None
    try:
        pointer = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"internal_ff_cd matrix checkpoint pointer is corrupt: "
            f"{pointer_path}") from exc
    required = {"schema", "slot", "generation", "completed", "identity"}
    if set(pointer) != required:
        raise ValueError(
            "internal_ff_cd matrix checkpoint pointer keys differ from the "
            f"schema: got {sorted(pointer)}, required {sorted(required)}")
    if (int(pointer["schema"]) != MATRIX_CHECKPOINT_SCHEMA
            or int(pointer["slot"]) not in (0, 1)):
        raise ValueError(
            f"internal_ff_cd matrix checkpoint pointer is invalid: {pointer}")
    return pointer


def _atomic_write_matrix_pointer(pointer_path: Path, pointer: dict) -> None:
    """Publish the only mutable checkpoint byte after its slot commits."""
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(pointer_path) + f".tmp.{os.getpid()}")
    payload = (_canonical_json(pointer) + "\n").encode("utf-8")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, pointer_path)
    directory_fd = os.open(pointer_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _save_matrix_checkpoint(
        base_path: Path, identity: dict, completed: int,
        accumulators: tuple[jax.Array, ...],
        head_accumulators: tuple[jax.Array, ...], *, mesh: Mesh,
        logical_matrix_shape: tuple[int, int, int, int],
        logical_head_shape: tuple[int, int, int],
        chi_wall: float = 0.0, solve_contract_wall: float = 0.0,
        stage_timings: dict | None = None) -> dict:
    """Commit an all-P matrix prefix to an inactive SlabIO slot.

    The active file is immutable.  Only after the inactive SlabIO handle has
    closed (and therefore stamped its commit receipt) does rank zero replace
    the tiny identity pointer.  At no point is a matrix accumulator converted
    to NumPy or gathered on a host.
    """
    from common.collectives import rank0_transaction
    from file_io.slab_io import SlabIO

    if not accumulators or len(accumulators) != len(head_accumulators):
        raise ValueError(
            "internal_ff_cd matrix checkpoint requires matching nonempty "
            "body/head accumulator tuples")
    grid_n = int(identity.get("grid_n", -1))
    completed = int(completed)
    if completed < 0 or completed > grid_n:
        raise ValueError(
            f"internal_ff_cd matrix checkpoint completed={completed} outside "
            f"[0,{grid_n}]")
    logical_matrix_shape = tuple(int(v) for v in logical_matrix_shape)
    logical_head_shape = tuple(int(v) for v in logical_head_shape)
    body_shapes = {tuple(int(v) for v in value.shape)
                   for value in accumulators}
    head_shapes = {tuple(int(v) for v in value.shape)
                   for value in head_accumulators}
    if len(body_shapes) != 1 or len(head_shapes) != 1:
        raise ValueError("internal_ff_cd checkpoint accumulator shapes differ")
    matrix_carrier, head_carrier = next(iter(body_shapes)), next(iter(head_shapes))
    if (len(matrix_carrier) != 4 or len(head_carrier) != 3
            or any(a > b for a, b in zip(logical_matrix_shape, matrix_carrier))
            or any(a > b for a, b in zip(logical_head_shape, head_carrier))):
        raise ValueError(
            "internal_ff_cd checkpoint logical/carrier shapes are invalid: "
            f"matrix {logical_matrix_shape}/{matrix_carrier}, "
            f"head {logical_head_shape}/{head_carrier}")

    pointer_path, slot_paths = _matrix_checkpoint_paths(base_path)
    current = _read_matrix_checkpoint_pointer(base_path)
    if current is not None and current["identity"] != identity:
        raise ValueError(
            "internal_ff_cd matrix checkpoint identity changed; start a new "
            f"run variant rather than overwriting its slots: {base_path}")
    if current is not None and completed < int(current["completed"]):
        raise ValueError(
            "internal_ff_cd matrix checkpoint cannot move backwards: "
            f"completed={completed} after {int(current['completed'])}")
    active = -1 if current is None else int(current["slot"])
    generation = 0 if current is None else int(current["generation"]) + 1
    inactive = 0 if active != 0 else 1
    slot_path = slot_paths[inactive]
    identity_bytes = _canonical_json(identity).encode("utf-8")
    identity_sha256 = hashlib.sha256(identity_bytes).digest()
    stages = ({key: 0.0 for key in STAGE_TIMING_KEYS}
              if stage_timings is None else {
                  key: float(stage_timings[key]) for key in STAGE_TIMING_KEYS
              })

    rank0_transaction(
        pointer_path, stage="internal_ff_cd.matrix_checkpoint_directory",
        write=lambda: pointer_path.parent.mkdir(parents=True, exist_ok=True))
    with SlabIO(slot_path, mode="w", mesh=mesh) as io:
        for index, value in enumerate(accumulators):
            name = f"body_accumulator_{index}"
            io.create_dataset(
                name, shape=logical_matrix_shape, dtype=np.complex128)
            io.write_slab(
                name, value, valid_shape=logical_matrix_shape,
                dtype=np.complex128)
        for index, value in enumerate(head_accumulators):
            name = f"head_accumulator_{index}"
            io.create_dataset(
                name, shape=logical_head_shape, dtype=np.complex128)
            io.write_slab(
                name, value, valid_shape=logical_head_shape,
                dtype=np.complex128)
        io.write_attr("matrix_checkpoint_schema", np.int64(
            MATRIX_CHECKPOINT_SCHEMA))
        io.write_attr(
            "identity_sha256_bytes", np.frombuffer(
                identity_sha256, dtype=np.uint8).astype(np.int32))
        io.write_attr("generation", np.int64(generation))
        io.write_attr("completed", np.int64(completed))
        io.write_attr("n_accumulators", np.int64(len(accumulators)))
        io.write_attr("chi_wall_seconds", np.float64(chi_wall))
        io.write_attr(
            "solve_contract_wall_seconds", np.float64(solve_contract_wall))
        for key, value in stages.items():
            io.write_attr(key, np.float64(value))

    pointer = {
        "schema": MATRIX_CHECKPOINT_SCHEMA,
        "slot": inactive,
        "generation": generation,
        "completed": completed,
        "identity": identity,
    }
    rank0_transaction(
        pointer_path, stage="internal_ff_cd.matrix_checkpoint_publish",
        write=lambda: _atomic_write_matrix_pointer(pointer_path, pointer))
    return pointer


def _load_matrix_checkpoint(
        base_path: Path, identity: dict, n_accumulators: int, *, mesh: Mesh,
        matrix_carrier_shape: tuple[int, int, int, int],
        head_carrier_shape: tuple[int, int, int],
        logical_matrix_shape: tuple[int, int, int, int],
        logical_head_shape: tuple[int, int, int],
        return_stage_timings: bool = False):
    """Read only the slot named by the atomic pointer, directly to shards."""
    from file_io.slab_io import SlabIO

    pointer = _read_matrix_checkpoint_pointer(base_path)
    if pointer is None:
        return None
    if pointer["identity"] != identity:
        raise ValueError(
            "internal_ff_cd matrix checkpoint identity changed; start a new "
            f"run variant rather than mixing prefixes: {base_path}")
    identity_sha256 = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")).digest()
    pointer_path, slot_paths = _matrix_checkpoint_paths(base_path)
    del pointer_path
    slot_path = slot_paths[int(pointer["slot"])]
    with SlabIO(slot_path, mode="r", mesh=mesh) as io:
        stamped_digest = bytes(np.asarray(
            io.read_small("identity_sha256_bytes"), dtype=np.uint8))
        generation = int(io.read_small("generation"))
        completed = int(io.read_small("completed"))
        stamped_n = int(io.read_small("n_accumulators"))
        if stamped_digest != identity_sha256:
            raise ValueError(
                f"internal_ff_cd matrix slot identity mismatch: {slot_path}")
        if (generation != int(pointer["generation"])
                or completed != int(pointer["completed"])):
            raise ValueError(
                "internal_ff_cd matrix pointer/slot generation mismatch: "
                f"{pointer} vs generation={generation},completed={completed}")
        if stamped_n != int(n_accumulators):
            raise ValueError(
                f"internal_ff_cd matrix slot has {stamped_n} accumulators; "
                f"expected {n_accumulators}")
        body = tuple(io.read_slab(
            f"body_accumulator_{index}", shape=matrix_carrier_shape,
            valid_shape=logical_matrix_shape, mesh=mesh,
            partition_spec=P(None, None, "x", "y"), dtype=np.complex128)
                     for index in range(stamped_n))
        head = tuple(io.read_slab(
            f"head_accumulator_{index}", shape=head_carrier_shape,
            valid_shape=logical_head_shape, mesh=mesh,
            partition_spec=P(None, None, "x"), dtype=np.complex128)
                     for index in range(stamped_n))
        chi_wall = float(io.read_small("chi_wall_seconds"))
        solve_contract_wall = float(io.read_small(
            "solve_contract_wall_seconds"))
        stages = {
            key: float(io.read_small(key)) for key in STAGE_TIMING_KEYS
        }
    values = (completed, body, head, chi_wall, solve_contract_wall)
    return values + (stages,) if return_stage_timings else values


def _prepare_head_response_context(
        wfns, *, state, config, meta, mesh, sym, wfn, band_slices,
        nb_chi_logical):
    """Prepare the one q->0 response owner for static and streamed samples."""
    from common.collectives import device_put_process_local
    from .fermi_surface import star_symmetrize_weights, tetrahedron_delta_weights
    from .qsgw_head import load_dft_velocity_head

    nk = int(meta.nk_tot)
    surface = tetrahedron_delta_weights(
        np.asarray(wfns.enk), np.asarray(sym.unfolded_kpts),
        tuple(int(x) for x in wfn.kgrid), float(state.mu_ry))
    surface = star_symmetrize_weights(surface, np.asarray(sym.irr_idx_k))
    surface_kn = jnp.asarray(surface * nk, dtype=jnp.float64)
    velocity = load_dft_velocity_head(
        config.paths.parallel_transport_file, mesh=mesh, wfn=wfn, meta=meta)
    nb_logical = int(nb_chi_logical)
    nb_storage = int(band_slices.nb_full)
    if int(wfns.enk.shape[1]) != nb_storage:
        raise ValueError(
            "internal_ff_cd head carrier mismatch: BandSlices storage "
            f"extent is {nb_storage}, wfns.enk has {wfns.enk.shape[1]}")
    if tuple(velocity.velocity_dft_cart.shape[-2:]) != (
            nb_storage, nb_storage):
        raise ValueError(
            "internal_ff_cd velocity carrier mismatch: expected padded "
            f"matrix extent {nb_storage}, got "
            f"{tuple(velocity.velocity_dft_cart.shape[-2:])}")
    if nb_logical > int(velocity.nb_logical):
        raise ValueError(
            "internal_ff_cd head response requests "
            f"number_bands_chi={nb_logical}, but the velocity store covers only "
            f"{int(velocity.nb_logical)} bands")
    identity = np.broadcast_to(
        np.eye(nb_storage, dtype=np.complex128)[None],
        (nk, nb_storage, nb_storage)).copy()
    U = device_put_process_local(
        identity, NamedSharding(mesh, P(None, "x", "y")))
    return {
        "surface_kn": surface_kn,
        "velocity": velocity,
        "U": U,
        "nb_logical": nb_logical,
        "nb_storage": nb_storage,
    }


def _build_head_response(
        context, z_ry, *, wfns, state, config, meta, mesh, wfn):
    """Build dynamic head/wings at exactly the streamed body frequencies."""
    from .qsgw_head import build_iteration_head_response

    nb_storage = int(context["nb_storage"])
    velocity = context["velocity"]
    return build_iteration_head_response(
        None, None, velocity.velocity_dft_cart, context["U"],
        wfns.enk[:, :nb_storage], state.f_kn[:, :nb_storage],
        np.asarray(z_ry, np.complex128),
        surface_weight_qp_kn=context["surface_kn"][:, :nb_storage], mesh=mesh,
        kgrid=tuple(int(x) for x in wfn.kgrid),
        bvec_cart=velocity.reciprocal_lattice_cart,
        nb_logical=int(context["nb_logical"]),
        sigma_energies_ry=np.asarray(wfns.enk[:, wfns.slices.sigma]),
        efermi_ry=float(state.mu_ry), wfn=wfn, meta=meta, config=config,
        wfns_qp=wfns, eta_ry=0.0)


def _compute_head_diag_ev(wfns, target_k, target_b, *, state, config, meta,
                          mesh, sym, wfn, V_q, band_slices,
                          nb_chi_logical, print_fn, context=None):
    """The referee's pole-free static metallic head and eta=0 half residue."""
    from .qsgw_head import finalize_iteration_head_sample
    from .w_isdf import solve_w

    nk = int(meta.nk_tot)
    if context is None:
        context = _prepare_head_response_context(
            wfns, state=state, config=config, meta=meta, mesh=mesh, sym=sym,
            wfn=wfn, band_slices=band_slices,
            nb_chi_logical=nb_chi_logical)
    response = _build_head_response(
        context, np.asarray([0.0 + 0.0j], np.complex128), wfns=wfns,
        state=state, config=config, meta=meta, mesh=mesh, wfn=wfn)
    # The production head owns the tetrahedron surface convention.  Its
    # body matrix is solved directly, exactly as in the referee route.
    head_chi = response.static_chi_body_gamma.copy()
    w_gamma = solve_w(
        V_q[:1], head_chi, meta, mesh,
        dyson_solver="distributed")[0]
    sample = finalize_iteration_head_sample(
        response, 0, w_gamma, wfn=wfn, meta=meta, config=config, mesh=mesh)
    wc0 = complex(sample.wcoul0) - complex(sample.vc0)
    f_target = np.asarray(state.f_kn)[target_k, target_b]
    sigma = -(2.0 * f_target - 1.0) * wc0 * RYD_TO_EV / (
        2.0 * float(meta.cell_volume) * nk)
    print_fn(
        f"  internal_ff_cd head: source={sample.source}, "
        f"Wc0={wc0.real:+.9e}{wc0.imag:+.9e}i Ry, eta_W=0 exactly")
    return np.asarray(sigma, np.complex128), {
        "source": sample.source,
        "vc0": [float(complex(sample.vc0).real), float(complex(sample.vc0).imag)],
        "wcoul0": [float(complex(sample.wcoul0).real),
                    float(complex(sample.wcoul0).imag)],
        "wc0": [float(wc0.real), float(wc0.imag)],
        "eta_w_ev": 0.0,
        "formula": "-(2f-1)*(wcoul0-vc0)*Ry_to_eV/(2*cell_volume*Nk)",
    }


def _target_labels(sym, target_k: np.ndarray, target_b: np.ndarray,
                   band_offset: int) -> list[dict]:
    kfrac = np.asarray(sym.unfolded_kpts, dtype=np.float64)
    return [{
        "target_index": int(i),
        "k_full_index": int(k),
        "k_crystal": [float(v) for v in kfrac[int(k)]],
        "band_one_based": int(band_offset + b + 1),
    } for i, (k, b) in enumerate(zip(target_k, target_b))]


def compute_internal_ff_cd(
    wfns,
    V_q,
    *,
    config,
    meta,
    mesh_xy: Mesh,
    sym,
    wfn,
    band_slices,
    centroid_indices,
    input_dir: str,
    occupation_state=None,
    charge_zeta_identity=None,
    coulomb_policy_receipt=None,
    print_fn: Callable = print,
) -> InternalFFResult:
    """Compute the Tier-0 on-shell diagonal and its refusal-grade ledger.

    The external target set is the file wedge (``kirr_fullids``) times the
    complete Sigma evaluation window.  The returned array is unfolded to
    the driver's full k set.  A failed quadrature-control cell is written to
    the ledger and then refused by name; it is never hidden by cancellation
    or a mean.
    """
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    from symmetry_maps import unfold_isdf_operator
    from .efermi import OccupationState, mp1_negative_derivative
    from .gw_config import InternalFFCDOutput
    from .v_q_g_flat import _resolve_ibz_q_list
    from .w_isdf import solve_w

    t_start = time.perf_counter()
    rank = int(jax.process_index())
    nk = int(meta.nk_tot)
    b0 = int(band_slices.b0)
    nb_storage = int(band_slices.nb_full)
    nb_chi_logical = int(meta.b_id_4_chi_user) - b0
    nb_sigma_logical = int(meta.b_id_4_sigma_user) - b0
    if int(wfns.enk.shape[1]) != nb_storage:
        raise ValueError(
            "internal_ff_cd band carrier mismatch: "
            f"BandSlices.nb_full={nb_storage}, wfns.enk={wfns.enk.shape}")
    for name, value in (("number_bands_chi", nb_chi_logical),
                        ("number_bands_sigma", nb_sigma_logical)):
        if not 0 < value <= nb_storage:
            raise ValueError(
                f"internal_ff_cd {name}={value} is outside padded carrier "
                f"extent {nb_storage}")
    if occupation_state is None:
        occupation_state = OccupationState.solve_mp1(
            wfns.enk, np.full(nk, 1.0 / nk), float(wfn.num_electrons),
            float(config.occ_broadening_ry),
            state_capacity=float(spin_degeneracy_factor(wfn)))
    state = occupation_state
    selected_matrix = (
        config.sigma.internal_ff_cd_output
        is InternalFFCDOutput.SELECTED_MATRIX_BLOCK)
    surface = mp1_negative_derivative(
        wfns.enk, float(state.mu_ry), float(state.smearing_width_ry))

    q_full = np.asarray(sym.q_irr_full_idx, np.int32)
    kq_map_full = np.asarray(sym.kqfull_map, np.int32)
    if kq_map_full.shape != (nk, nk):
        raise ValueError(
            f"internal_ff_cd expected full k-q map {(nk, nk)}, got "
            f"{kq_map_full.shape}")
    kmq_wedge = np.asarray(kq_map_full[:, q_full].T, np.int32)
    (qint, qfrac, irr_idx, sym_idx, sym_perm, l_table,
     use_ibz) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=np.asarray(centroid_indices),
        kgrid=tuple(int(x) for x in wfn.kgrid),
        fft_grid=tuple(int(x) for x in wfn.fft_grid),
        context="Tier-0 internal_ff_cd")
    if not use_ibz or len(qint) != len(q_full):
        raise ValueError(
            "internal_ff_cd requires the canonical closed q wedge; "
            f"resolved use_ibz={use_ibz}, rows={len(qint)}, "
            f"SymMaps rows={len(q_full)}")
    v_wedge = jnp.take(V_q, jnp.asarray(q_full), axis=0)
    v_wedge.block_until_ready()

    k_wedge = np.asarray(sym.kirr_fullids, np.int32)
    sigma_bands = np.arange(
        int(band_slices.sigma.start or 0), int(band_slices.sigma.stop),
        dtype=np.int32)
    q_rows = np.arange(nk, dtype=np.int32)
    energies_ev = np.asarray(wfns.enk, np.float64) * RYD_TO_EV
    occupations = np.asarray(state.f_kn, np.float64)

    if selected_matrix:
        if getattr(wfns, "layout", "legacy") != "legacy":
            raise ValueError(
                "internal_ff_cd selected_matrix_block currently requires "
                "the legacy wavefunction carrier used by its direct chi0; "
                "a face-layout port must extend the same block projector, "
                "not introduce a second Green-function owner")
        if int(mesh_xy.shape["x"]) != int(mesh_xy.shape["y"]):
            raise ValueError(
                "internal_ff_cd selected_matrix_block requires the square "
                "2D processor grid used by its band-diagonal head injection; "
                f"got mesh {mesh_xy.shape}")
        from common.collectives import device_put_process_local
        from runtime.padding import pad_to_axis
        from .ppm_sigma import sigma_band_axis

        target_k = np.arange(nk, dtype=np.int32)
        target_b = sigma_bands.copy()
        n_targets = int(nk)
        kmq_target = np.asarray(kq_map_full, np.int32)
        omega_rel_ev = np.asarray(config.omega_grid_ev, dtype=np.float64)
        if (omega_rel_ev.ndim != 1 or omega_rel_ev.size < 2
                or np.any(np.diff(omega_rel_ev) <= 0.0)):
            raise ValueError(
                "internal_ff_cd selected_matrix_block requires a strictly "
                "increasing Sigma omega grid with at least two values")
        omega_abs_logical_ev = omega_rel_ev + float(state.mu_ry) * RYD_TO_EV
        omega_carrier_n = (
            (omega_rel_ev.size + EXTERNAL_FREQUENCY_TILE - 1)
            // EXTERNAL_FREQUENCY_TILE * EXTERNAL_FREQUENCY_TILE)
        omega_abs_ev = np.pad(
            omega_abs_logical_ev,
            (0, omega_carrier_n - omega_rel_ev.size), mode="edge")
        omega_valid = np.arange(omega_carrier_n) < omega_rel_ev.size
        max_required = float(np.max(np.abs(
            omega_abs_logical_ev[:, None, None]
            - energies_ev[None, :, :nb_sigma_logical])))
        real_max_ev = _real_coverage_max(max_required)
        x_signed = x_abs = residue_sign = None

        sigma_axis = sigma_band_axis(
            int(sigma_bands.size), mesh_xy, ansatz="internal_ff_cd")
        psi_proj_xr = pad_to_axis(
            wfns.xr(band_slices.sigma), sigma_axis, axis=1)
        psi_proj_yn = pad_to_axis(
            wfns.yn(band_slices.sigma), sigma_axis, axis=3)
        matrix_shape = (
            int(omega_carrier_n), nk, int(sigma_axis.carrier),
            int(sigma_axis.carrier))
        logical_matrix_shape = (
            int(omega_rel_ev.size), nk, int(sigma_axis.logical),
            int(sigma_axis.logical))
        head_shape = (
            int(omega_carrier_n), nk, int(sigma_axis.carrier))
        logical_head_shape = (
            int(omega_rel_ev.size), nk, int(sigma_axis.logical))
        block_contract = _make_weighted_block_contract_kernel(
            mesh_xy, n_target_k=nk, inner_stop=nb_sigma_logical)
        (make_matrix_zero, make_head_zero, add_matrix_tile,
         add_head_sample, pin_head_on_shell) = _make_matrix_accumulator_kernels(
             mesh_xy, matrix_shape=matrix_shape, head_shape=head_shape)
        head_energy_host = np.pad(
            energies_ev[:, sigma_bands],
            ((0, 0), (0, sigma_axis.carrier - sigma_axis.logical)))
        head_occ_host = np.pad(
            occupations[:, sigma_bands],
            ((0, 0), (0, sigma_axis.carrier - sigma_axis.logical)))
        head_energy_ev = device_put_process_local(
            head_energy_host, NamedSharding(mesh_xy, P(None, "x")))
        head_occupations = device_put_process_local(
            head_occ_host, NamedSharding(mesh_xy, P(None, "x")))
        omega_abs_device = jnp.asarray(omega_abs_ev, dtype=jnp.float64)
        omega_valid_device = jnp.asarray(omega_valid)
        head_context = _prepare_head_response_context(
            wfns, state=state, config=config, meta=meta, mesh=mesh_xy,
            sym=sym, wfn=wfn, band_slices=band_slices,
            nb_chi_logical=nb_chi_logical)
        gamma_rows = np.flatnonzero(q_full == 0)
        if gamma_rows.size != 1:
            raise ValueError(
                "internal_ff_cd selected_matrix_block requires exactly one "
                f"Gamma row in the q wedge, got {gamma_rows.tolist()}")
        gamma_row = int(gamma_rows[0])
    else:
        target_k = np.repeat(k_wedge, sigma_bands.size)
        target_b = np.tile(sigma_bands, k_wedge.size)
        n_targets = int(target_k.size)
        kmq_target = np.stack([
            kq_map_full[int(k), q_rows] for k in target_k]).astype(np.int32)
        target_e_ev = energies_ev[target_k, target_b]
        inner_e_ev = energies_ev[kmq_target]
        inner_f = occupations[kmq_target]
        x_signed = target_e_ev[:, None, None] - inner_e_ev + CENTER_SHIFT_EV
        x_abs = abs(x_signed)
        max_required = float(np.max(x_abs[:, :, :nb_sigma_logical]))
        real_max_ev = _real_coverage_max(max_required)
        residue_sign = np.where(
            x_signed >= 0.0, -(1.0 - inner_f), inner_f)

    direct = make_direct_kernel(
        mesh_xy, nb_logical=nb_chi_logical)
    contract = (None if selected_matrix else make_weighted_contract_kernel(
        mesh_xy, n_targets=n_targets, inner_stop=nb_sigma_logical))
    n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
    body_provenance = _body_checkpoint_provenance(
        energies_ry=wfns.enk, state=state,
        charge_zeta_identity=charge_zeta_identity,
        coulomb_policy_receipt=coulomb_policy_receipt,
        q_mapping={
            "q_full": q_full,
            "kmq_wedge": kmq_wedge,
            "kmq_target": kmq_target,
            "irr_idx": irr_idx,
            "sym_idx": sym_idx,
            "sym_perm": sym_perm,
            "l_table": l_table,
            "qfrac": qfrac,
        },
        centroid_indices=centroid_indices,
        nb_chi_logical=nb_chi_logical,
        nb_sigma_logical=nb_sigma_logical,
        band_carrier_storage=nb_storage,
        b0=b0, V_q=V_q, nmu_logical=meta.n_rmu, nk=nk,
        mesh_xy=mesh_xy)

    checkpoint_dir = Path(input_dir) / "internal_ff_cd_checkpoints"
    observer = None
    observer_identity = None
    observer_receipt = None
    if config.sigma.w_observer:
        # Lazy by contract: the default-off oracle does not import the
        # observer, create its kernel, or execute an additional JAX op.
        from vcoul import CoulombGeometry
        from .internal_ff_w_observer import (
            open_w_observer, plan_w_observer)

        real_arm_plans = []
        for width in RESPONSE_WIDTHS_EV:
            planned_grid = real_grid(width, max_ev=real_max_ev)
            planned_z = (planned_grid + 1j * width) / RYD_TO_EV
            real_arm_plans.append({
                "name": f"real_eta_{width:.8f}",
                "requested_z_ry": planned_z,
                "evaluated_z_ry": planned_z,
            })
        planned_igrid = imag_grid()
        observer_spec = plan_w_observer(
            input_dir=input_dir, real_arms=real_arm_plans,
            imag_grid={
                "name": "imaginary",
                "requested_z_ry": 1j * planned_igrid / RYD_TO_EV,
                "evaluated_z_ry": 1j * np.maximum(
                    planned_igrid, IMAG_ORIGIN_LIMIT_EV) / RYD_TO_EV,
            },
            q_full=q_full, q_irr_frac=qfrac,
            bvec_cart=CoulombGeometry.from_wfn(wfn).bvec,
            nmu_logical=meta.n_rmu,
            centroid_identity=body_provenance["centroid_indices"],
            body_provenance=body_provenance)
        observer_identity = observer_spec.identity_digest
        # Authenticate every pre-existing body prefix before mode="w" may
        # create observer bytes.  In particular, an old default-off CD
        # checkpoint refuses here because it lacks the observer identity.
        preexisting_completed = []
        for width in RESPONSE_WIDTHS_EV:
            planned_grid = real_grid(width, max_ev=real_max_ev)
            planned_identity = _checkpoint_identity(
                kind="real", grid=planned_grid, width_ev=width,
                target_k=target_k, target_b=target_b,
                body_provenance=body_provenance,
                w_observer_identity=observer_identity)
            planned_identity["grid_n"] = int(planned_grid.size)
            if selected_matrix:
                planned_identity = _matrix_output_identity(
                    planned_identity, omega_rel_ev=omega_rel_ev,
                    sigma_band_axis=sigma_axis, nk=nk)
                planned_checkpoint = (
                    checkpoint_dir / f"real_eta_{width:.8f}.matrix")
                pointer = _read_matrix_checkpoint_pointer(planned_checkpoint)
                if pointer is None:
                    preexisting_completed.append(0)
                else:
                    if pointer["identity"] != planned_identity:
                        raise ValueError(
                            "internal_ff_cd selected checkpoint identity "
                            f"changed before W-observer open: {planned_checkpoint}")
                    preexisting_completed.append(int(pointer["completed"]))
            else:
                planned_checkpoint = (
                    checkpoint_dir / f"real_eta_{width:.8f}.npz")
                preexisting_completed.append(_load_checkpoint(
                    planned_checkpoint, planned_identity, n_targets, 2)[0])
        planned_identity = _checkpoint_identity(
            kind="imaginary", grid=planned_igrid, width_ev=None,
            target_k=target_k, target_b=target_b,
            body_provenance=body_provenance,
            w_observer_identity=observer_identity)
        planned_identity["grid_n"] = int(planned_igrid.size)
        if selected_matrix:
            planned_identity = _matrix_output_identity(
                planned_identity, omega_rel_ev=omega_rel_ev,
                sigma_band_axis=sigma_axis, nk=nk)
            planned_checkpoint = checkpoint_dir / "imaginary.matrix"
            pointer = _read_matrix_checkpoint_pointer(planned_checkpoint)
            if pointer is None:
                preexisting_completed.append(0)
            else:
                if pointer["identity"] != planned_identity:
                    raise ValueError(
                        "internal_ff_cd selected checkpoint identity changed "
                        f"before W-observer open: {planned_checkpoint}")
                preexisting_completed.append(int(pointer["completed"]))
        else:
            preexisting_completed.append(_load_checkpoint(
                checkpoint_dir / "imaginary.npz", planned_identity,
                n_targets, 3)[0])
        if any(preexisting_completed) and not (
                Path(observer_spec.payload_path).exists()
                and Path(observer_spec.sidecar_path).exists()):
            raise ValueError(
                "internal_ff_cd checkpoints have consumed W frequencies but "
                "the matching W observer transaction is absent; missing W "
                "cannot be reconstructed. Start a new run variant.")
        observer = open_w_observer(
            observer_spec, mesh_xy=mesh_xy, v_wedge=v_wedge)
        observer_receipt = observer.artifact_receipt

    def frequency_batch(z_ry, coefficient_rows, global_frequency_index=None,
                        selected_state=None):
        """Evaluate a fixed-size referee frequency batch.

        The direct pair scan is substantially more efficient with its
        replicated frequency axis batched.  W is still solved and consumed
        one frequency at a time, so no additional ``N_mu^2`` object is ever
        replicated or retained as spectral history.
        """
        z = jnp.asarray(z_ry, dtype=jnp.complex128)
        head_response = None
        if selected_matrix:
            head_response = _build_head_response(
                head_context, np.asarray(z_ry, np.complex128), wfns=wfns,
                state=state, config=config, meta=meta, mesh=mesh_xy, wfn=wfn)
        chi_rows = [
            direct(wfns.psi_xn, wfns.psi_yn, jnp.asarray(kmq_wedge[i]),
                   wfns.enk, state.f_kn, surface, z)
            for i in range(q_full.size)]
        chi_bq = jnp.stack(chi_rows, axis=1)
        chi_bq.block_until_ready()
        chi_completed_at = time.perf_counter()
        stage_seconds = {key: 0.0 for key in STAGE_TIMING_KEYS}
        outputs = []
        if selected_matrix:
            if selected_state is None:
                raise ValueError(
                    "selected_matrix_block frequency batch lacks its "
                    "distributed accumulators")
            selected_body = list(selected_state[0])
            selected_head = list(selected_state[1])
        if observer is not None and (
                global_frequency_index is None
                or len(global_frequency_index) != len(coefficient_rows)):
            raise ValueError(
                "W observer requires one global index per frequency row")
        for jb, coefficients in enumerate(coefficient_rows):
            try:
                t_stage = time.perf_counter()
                w_wedge = solve_w(
                    v_wedge, chi_bq[jb].copy(), meta, mesh_xy,
                    dyson_solver="distributed")
                w_wedge.block_until_ready()
                stage_seconds["dyson_solve_wall_seconds"] += (
                    time.perf_counter() - t_stage)
                t_stage = time.perf_counter()
                wc_wedge = w_wedge - v_wedge
                if observer is not None:
                    observer.observe(global_frequency_index[jb], wc_wedge)
                    # The observer owns its timings; do not charge its action
                    # or enqueue latency to the production q-unfold stage.
                    t_stage = time.perf_counter()
                wc_full = unfold_isdf_operator(
                    wc_wedge, irr_idx=irr_idx, sym_idx=sym_idx,
                    sym_perm=sym_perm, L_table=l_table, q_irr_frac=qfrac,
                    mesh_xy=mesh_xy, n_sym_spatial=n_sym_spatial,
                    trs_rule="pair_transpose")
                wc_full.block_until_ready()
                stage_seconds["q_unfold_wall_seconds"] += (
                    time.perf_counter() - t_stage)
                # CD spectral convention S=(I-epsilon^-1)V = -Wc.
                t_stage = time.perf_counter()
                row = []
                if selected_matrix:
                    from .qsgw_head import finalize_iteration_head_sample
                    sample = finalize_iteration_head_sample(
                        head_response, jb, w_wedge[gamma_row],
                        wfn=wfn, meta=meta, config=config, mesh=mesh_xy)
                    wc0_ry = complex(sample.wcoul0) - complex(sample.vc0)
                    for accumulator_index, rule in coefficients:
                        for omega_lo in range(
                                0, omega_carrier_n,
                                EXTERNAL_FREQUENCY_TILE):
                            omega_hi = omega_lo + EXTERNAL_FREQUENCY_TILE
                            value = -block_contract(
                                wfns.psi_xn, wfns.psi_yn,
                                psi_proj_xr, psi_proj_yn, wc_full,
                                jnp.asarray(kmq_target),
                                jnp.asarray(energies_ev, jnp.float64),
                                jnp.asarray(occupations, jnp.float64),
                                omega_abs_device[omega_lo:omega_hi],
                                omega_valid_device[omega_lo:omega_hi],
                                jnp.asarray(rule, jnp.float64))
                            selected_body[accumulator_index] = add_matrix_tile(
                                selected_body[accumulator_index], value / nk,
                                jnp.asarray(omega_lo, jnp.int32))
                        selected_head[accumulator_index] = add_head_sample(
                            selected_head[accumulator_index],
                            omega_abs_device, omega_valid_device,
                            head_energy_ev, head_occupations,
                            jnp.asarray(wc0_ry, jnp.complex128),
                            jnp.asarray(rule, jnp.float64),
                            jnp.asarray(
                                1.0 / (float(meta.cell_volume) * nk),
                                jnp.float64))
                else:
                    for coeff in coefficients:
                        value = -contract(
                            wfns.psi_xn, wfns.psi_yn, wc_full,
                            jnp.asarray(target_k), jnp.asarray(target_b),
                            jnp.asarray(kmq_target),
                            jnp.asarray(coeff, jnp.float64))
                        host = np.asarray(value)
                        if not np.all(np.isfinite(host)):
                            bad = int(np.flatnonzero(~np.isfinite(host))[0])
                            raise FloatingPointError(
                                "internal_ff_cd nonfinite contracted stream "
                                f"at target {bad}")
                        row.append(host * (RYD_TO_EV / nk))
                outputs.append(row)
                del w_wedge, wc_wedge, wc_full
                stage_seconds["contract_host_checks_wall_seconds"] += (
                    time.perf_counter() - t_stage)
            except BaseException:
                if observer is not None:
                    observer.close(body_complete=False)
                raise
        selected_out = ((tuple(selected_body), tuple(selected_head))
                        if selected_matrix else None)
        return outputs, selected_out, chi_completed_at, stage_seconds

    real_results = []
    real_head_results = []
    grid_records = []
    final_coarse_real = None
    final_coarse_real_head = None
    for width in RESPONSE_WIDTHS_EV:
        arm_name = f"real_eta_{width:.8f}"
        arm_start = (0 if observer is None else
                     int(observer.spec.arm(arm_name)["start"]))
        grid = real_grid(width, max_ev=real_max_ev)
        coarse = (_coarse_real_grid(grid, width)
                  if width == RESPONSE_WIDTHS_EV[-1] else None)
        coarse_lookup = ({int(i): j for j, i in enumerate(coarse)}
                         if coarse is not None else {})
        identity = _checkpoint_identity(
            kind="real", grid=grid, width_ev=width,
            target_k=target_k, target_b=target_b,
            body_provenance=body_provenance,
            w_observer_identity=observer_identity)
        identity["grid_n"] = int(grid.size)
        if selected_matrix:
            identity = _matrix_output_identity(
                identity, omega_rel_ev=omega_rel_ev,
                sigma_band_axis=sigma_axis, nk=nk)
            checkpoint = checkpoint_dir / f"real_eta_{width:.8f}.matrix"
            loaded_matrix = _load_matrix_checkpoint(
                checkpoint, identity, 2, mesh=mesh_xy,
                matrix_carrier_shape=matrix_shape,
                head_carrier_shape=head_shape,
                logical_matrix_shape=logical_matrix_shape,
                logical_head_shape=logical_head_shape,
                return_stage_timings=True)
            if loaded_matrix is None:
                completed = 0
                loaded = (make_matrix_zero(), make_matrix_zero())
                loaded_head = (make_head_zero(), make_head_zero())
                chi_wall = solve_contract_wall = 0.0
                stage_wall = {key: 0.0 for key in STAGE_TIMING_KEYS}
            else:
                (completed, loaded, loaded_head, chi_wall,
                 solve_contract_wall, stage_wall) = loaded_matrix
        else:
            checkpoint = checkpoint_dir / f"real_eta_{width:.8f}.npz"
            (completed, loaded, chi_wall, solve_contract_wall,
             stage_wall) = _load_checkpoint(
                checkpoint, identity, n_targets, 2,
                return_stage_timings=True)
        if observer is not None:
            observer.require_checkpoint_prefix(arm_name, completed)
        accum, coarse_accum = loaded
        if selected_matrix:
            head_accum, coarse_head_accum = loaded_head
        if rank == 0 and completed:
            print_fn(
                f"  internal_ff_cd resume real eta_W={width:g} eV: "
                f"{completed}/{len(grid)} frequencies from {checkpoint}")
        t_arm = time.perf_counter()
        for lo in range(completed, grid.size, FREQUENCY_BATCH):
            hi = min(lo + FREQUENCY_BATCH, grid.size)
            coefficient_rows = []
            for iw in range(lo, hi):
                coeff = (_real_node_descriptor(grid, iw) if selected_matrix
                         else _real_coefficients(
                             grid, iw, x_abs, residue_sign))
                coeffs = ([(0, coeff)] if selected_matrix else [coeff])
                if iw in coarse_lookup:
                    cj = coarse_lookup[iw]
                    cgrid = grid[coarse]
                    ccoeff = (
                        _real_node_descriptor(cgrid, cj) if selected_matrix
                        else _real_coefficients(
                            cgrid, cj, x_abs, residue_sign))
                    coeffs.append((1, ccoeff) if selected_matrix else ccoeff)
                coefficient_rows.append(coeffs)
            t_batch = time.perf_counter()
            z_batch = (grid[lo:hi] / RYD_TO_EV
                       + 1j * (width / RYD_TO_EV))
            if observer is None:
                values, selected_state, t_after_chi, stage_batch = (
                    frequency_batch(
                        z_batch, coefficient_rows,
                        selected_state=(
                            ((accum, coarse_accum),
                             (head_accum, coarse_head_accum)))
                        if selected_matrix else None))
            else:
                values, selected_state, t_after_chi, stage_batch = frequency_batch(
                    z_batch, coefficient_rows,
                    np.arange(arm_start + lo, arm_start + hi, dtype=np.int64),
                    selected_state=(
                        ((accum, coarse_accum),
                         (head_accum, coarse_head_accum)))
                    if selected_matrix else None)
            chi_wall += t_after_chi - t_batch
            solve_contract_wall += time.perf_counter() - t_after_chi
            for key in STAGE_TIMING_KEYS:
                stage_wall[key] += stage_batch[key]
            if selected_matrix:
                ((accum, coarse_accum),
                 (head_accum, coarse_head_accum)) = selected_state
            else:
                for jb, iw in enumerate(range(lo, hi)):
                    accum += values[jb][0]
                    if iw in coarse_lookup:
                        coarse_accum += values[jb][1]
            publish_prefix = (
                not selected_matrix
                or hi == len(grid)
                or hi % MATRIX_CHECKPOINT_FREQUENCIES == 0)
            if observer is not None and publish_prefix:
                observer.commit_prefix(arm_name, hi)
            if selected_matrix and publish_prefix:
                _save_matrix_checkpoint(
                    checkpoint, identity, hi, (accum, coarse_accum),
                    (head_accum, coarse_head_accum), mesh=mesh_xy,
                    logical_matrix_shape=logical_matrix_shape,
                    logical_head_shape=logical_head_shape,
                    chi_wall=chi_wall,
                    solve_contract_wall=solve_contract_wall,
                    stage_timings=stage_wall)
            elif not selected_matrix and rank == 0:
                _save_checkpoint(
                    checkpoint, identity, hi, (accum, coarse_accum),
                    chi_wall, solve_contract_wall,
                    stage_timings=stage_wall)
            if rank == 0 and (hi == len(grid) or hi % 50 < FREQUENCY_BATCH):
                print_fn(
                    f"  internal_ff_cd real eta_W={width:g} eV: "
                    f"{hi}/{len(grid)} frequencies")
        real_results.append(accum)
        final_coarse_real = coarse_accum if coarse is not None else final_coarse_real
        if selected_matrix:
            real_head_results.append(head_accum)
            final_coarse_real_head = (
                coarse_head_accum if coarse is not None
                else final_coarse_real_head)
        grid_records.append({
            "kind": "real", "eta_w_ev": width, "n": int(grid.size),
            "min_ev": float(grid[0]), "max_ev": float(grid[-1]),
            "sha256": _hash_array(grid),
            "frequency_batch": FREQUENCY_BATCH,
            "checkpoint": str(checkpoint),
            "resumed_at_frequency": completed,
            "chi_wall_seconds": chi_wall,
            "solve_contract_wall_seconds": solve_contract_wall,
            **stage_wall,
            "solve_contract_unattributed_wall_seconds": float(
                solve_contract_wall - sum(stage_wall.values())),
            "wall_seconds": time.perf_counter() - t_arm,
        })

    arm_name = "imaginary"
    arm_start = (0 if observer is None else
                 int(observer.spec.arm(arm_name)["start"]))
    igrid = imag_grid()
    icoarse = _coarse_imag_grid(igrid)
    icoarse_lookup = {int(i): j for j, i in enumerate(icoarse)}
    itail = igrid[igrid <= 60.0 + 1.0e-12]
    tail_panel_max_ev = float(_panel_bounds(itail)[1][-1])
    identity = _checkpoint_identity(
        kind="imaginary", grid=igrid, width_ev=None,
        target_k=target_k, target_b=target_b,
        body_provenance=body_provenance,
        w_observer_identity=observer_identity)
    identity["grid_n"] = int(igrid.size)
    if selected_matrix:
        identity = _matrix_output_identity(
            identity, omega_rel_ev=omega_rel_ev,
            sigma_band_axis=sigma_axis, nk=nk)
        checkpoint = checkpoint_dir / "imaginary.matrix"
        loaded_matrix = _load_matrix_checkpoint(
            checkpoint, identity, 3, mesh=mesh_xy,
            matrix_carrier_shape=matrix_shape,
            head_carrier_shape=head_shape,
            logical_matrix_shape=logical_matrix_shape,
            logical_head_shape=logical_head_shape,
            return_stage_timings=True)
        if loaded_matrix is None:
            completed = 0
            loaded = (make_matrix_zero(), make_matrix_zero(),
                      make_matrix_zero())
            loaded_head = (make_head_zero(), make_head_zero(),
                           make_head_zero())
            chi_wall = solve_contract_wall = 0.0
            stage_wall = {key: 0.0 for key in STAGE_TIMING_KEYS}
        else:
            (completed, loaded, loaded_head, chi_wall,
             solve_contract_wall, stage_wall) = loaded_matrix
    else:
        checkpoint = checkpoint_dir / "imaginary.npz"
        (completed, loaded, chi_wall, solve_contract_wall,
         stage_wall) = _load_checkpoint(
            checkpoint, identity, n_targets, 3,
            return_stage_timings=True)
    if observer is not None:
        observer.require_checkpoint_prefix(arm_name, completed)
    imag_accum, imag_coarse, imag_tail = loaded
    if selected_matrix:
        imag_head, imag_coarse_head, imag_tail_head = loaded_head
    if rank == 0 and completed:
        print_fn(
            f"  internal_ff_cd resume imaginary: {completed}/{len(igrid)} "
            f"frequencies from {checkpoint}")
    t_arm = time.perf_counter()
    for lo in range(completed, igrid.size, FREQUENCY_BATCH):
        hi = min(lo + FREQUENCY_BATCH, igrid.size)
        coefficient_rows = []
        for iw in range(lo, hi):
            coeff = (_imaginary_node_descriptor(igrid, iw)
                     if selected_matrix else
                     _imag_coefficients(igrid, iw, x_signed))
            coeffs = ([(0, coeff)] if selected_matrix else [coeff])
            if iw in icoarse_lookup:
                cj = icoarse_lookup[iw]
                cgrid = igrid[icoarse]
                ccoeff = (_imaginary_node_descriptor(cgrid, cj)
                          if selected_matrix else
                          _imag_coefficients(cgrid, cj, x_signed))
                coeffs.append((1, ccoeff) if selected_matrix else ccoeff)
            if iw < itail.size:
                tcoeff = (_imaginary_node_descriptor(itail, iw)
                          if selected_matrix else
                          _imag_coefficients(itail, iw, x_signed))
                coeffs.append((2, tcoeff) if selected_matrix else tcoeff)
            coefficient_rows.append(coeffs)
        z_imag_ev = np.maximum(igrid[lo:hi], IMAG_ORIGIN_LIMIT_EV)
        t_batch = time.perf_counter()
        z_batch = 1j * z_imag_ev / RYD_TO_EV
        if observer is None:
            values, selected_state, t_after_chi, stage_batch = frequency_batch(
                z_batch, coefficient_rows,
                selected_state=(
                    ((imag_accum, imag_coarse, imag_tail),
                     (imag_head, imag_coarse_head, imag_tail_head)))
                if selected_matrix else None)
        else:
            values, selected_state, t_after_chi, stage_batch = frequency_batch(
                z_batch, coefficient_rows,
                np.arange(arm_start + lo, arm_start + hi, dtype=np.int64),
                selected_state=(
                    ((imag_accum, imag_coarse, imag_tail),
                     (imag_head, imag_coarse_head, imag_tail_head)))
                if selected_matrix else None)
        chi_wall += t_after_chi - t_batch
        solve_contract_wall += time.perf_counter() - t_after_chi
        for key in STAGE_TIMING_KEYS:
            stage_wall[key] += stage_batch[key]
        if selected_matrix:
            ((imag_accum, imag_coarse, imag_tail),
             (imag_head, imag_coarse_head, imag_tail_head)) = selected_state
        else:
            for jb, iw in enumerate(range(lo, hi)):
                imag_accum += values[jb][0]
                offset = 1
                if iw in icoarse_lookup:
                    imag_coarse += values[jb][offset]
                    offset += 1
                if iw < itail.size:
                    imag_tail += values[jb][offset]
        publish_prefix = (
            not selected_matrix
            or hi == len(igrid)
            or hi % MATRIX_CHECKPOINT_FREQUENCIES == 0)
        if observer is not None and publish_prefix:
            observer.commit_prefix(arm_name, hi)
        if selected_matrix and publish_prefix:
            _save_matrix_checkpoint(
                checkpoint, identity, hi,
                (imag_accum, imag_coarse, imag_tail),
                (imag_head, imag_coarse_head, imag_tail_head), mesh=mesh_xy,
                logical_matrix_shape=logical_matrix_shape,
                logical_head_shape=logical_head_shape,
                chi_wall=chi_wall, solve_contract_wall=solve_contract_wall,
                stage_timings=stage_wall)
        elif not selected_matrix and rank == 0:
            _save_checkpoint(
                checkpoint, identity, hi,
                (imag_accum, imag_coarse, imag_tail),
                chi_wall, solve_contract_wall,
                stage_timings=stage_wall)
        if rank == 0 and (hi == len(igrid) or hi % 50 < FREQUENCY_BATCH):
            print_fn(f"  internal_ff_cd imaginary: {hi}/{len(igrid)} frequencies")
    grid_records.append({
        "kind": "imaginary", "eta_w_ev": 0.0, "n": int(igrid.size),
        "min_ev": float(igrid[0]), "max_ev": float(igrid[-1]),
        "origin_limit_ev": IMAG_ORIGIN_LIMIT_EV,
        "sha256": _hash_array(igrid),
        "frequency_batch": FREQUENCY_BATCH,
        "checkpoint": str(checkpoint),
        "resumed_at_frequency": completed,
        "chi_wall_seconds": chi_wall,
        "solve_contract_wall_seconds": solve_contract_wall,
        **stage_wall,
        "solve_contract_unattributed_wall_seconds": float(
            solve_contract_wall - sum(stage_wall.values())),
        "wall_seconds": time.perf_counter() - t_arm,
    })
    if observer is not None:
        observer.close(body_complete=True)

    if selected_matrix:
        head_target_k = np.repeat(np.arange(nk, dtype=np.int32),
                                  sigma_bands.size)
        head_target_b = np.tile(sigma_bands, nk)
        static_head_ev_flat, head_record = _compute_head_diag_ev(
            wfns, head_target_k, head_target_b, state=state, config=config,
            meta=meta, mesh=mesh_xy, sym=sym, wfn=wfn, V_q=V_q,
            band_slices=band_slices, nb_chi_logical=nb_chi_logical,
            print_fn=print_fn, context=head_context)
        static_head_host = np.pad(
            static_head_ev_flat.reshape(nk, sigma_bands.size) / RYD_TO_EV,
            ((0, 0), (0, sigma_axis.carrier - sigma_axis.logical)))
        static_head = device_put_process_local(
            static_head_host, NamedSharding(mesh_xy, P(None, "x")))
        energy_rel_ev = head_energy_ev - float(state.mu_ry) * RYD_TO_EV
        band_valid = jnp.arange(sigma_axis.carrier) < sigma_axis.logical
        omega_rel_device = jnp.asarray(omega_rel_ev, jnp.float64)

        def pin(value):
            return pin_head_on_shell(
                value[:omega_rel_ev.size], omega_rel_device,
                energy_rel_ev, static_head, band_valid)

        fine_head = pin(real_head_results[-1] + imag_head)
        coarse_real_head = pin(final_coarse_real_head + imag_head)
        coarse_imag_head = pin(real_head_results[-1] + imag_coarse_head)
        tail_imag_head = pin(real_head_results[-1] + imag_tail_head)
        fine_real = real_results[-1][:omega_rel_ev.size]
        coarse_real = final_coarse_real[:omega_rel_ev.size]
        fine_imag = imag_accum[:omega_rel_ev.size]
        coarse_imag = imag_coarse[:omega_rel_ev.size]
        tail_imag = imag_tail[:omega_rel_ev.size]

        def component_max_mev(value):
            maximum = jnp.maximum(
                jnp.max(jnp.abs(jnp.real(value))),
                jnp.max(jnp.abs(jnp.imag(value))))
            return float(np.asarray(maximum)) * RYD_TO_EV * 1000.0

        controls = {
            "body_real_fine_minus_coarse_mev": component_max_mev(
                fine_real - coarse_real),
            "body_imag_fine_minus_coarse_mev": component_max_mev(
                fine_imag - coarse_imag),
            "body_imag_full_minus_tail_mev": component_max_mev(
                fine_imag - tail_imag),
            "head_real_fine_minus_coarse_mev": component_max_mev(
                fine_head - coarse_real_head),
            "head_imag_fine_minus_coarse_mev": component_max_mev(
                fine_head - coarse_imag_head),
            "head_imag_full_minus_tail_mev": component_max_mev(
                fine_head - tail_imag_head),
        }
        worst_control_mev = max(controls.values())
        body = fine_real + fine_imag
        dynamic_head = fine_head
        finite = bool(np.asarray(jnp.all(jnp.isfinite(body)))) and bool(
            np.asarray(jnp.all(jnp.isfinite(dynamic_head))))
        if not finite:
            raise FloatingPointError(
                "internal_ff_cd selected_matrix_block produced a nonfinite "
                "body or head shard")

        artifact = {
            "schema": 3,
            "route": "internal_ff_cd",
            "output": "selected_matrix_block",
            "status": ("pass" if worst_control_mev <= CD_CONTROL_TOL_MEV
                       else "unresolved"),
            "jobid": os.environ.get("SLURM_JOB_ID", "unknown"),
            "source_commit": os.environ.get(
                "LORRAX_SOURCE_COMMIT", "working-tree"),
            "wall_seconds": time.perf_counter() - t_start,
            "checkpoint_provenance": {
                "schema": MATRIX_CHECKPOINT_SCHEMA,
                **body_provenance,
            },
            "sharding": {
                "chi": "P(None,x,y)",
                "W": "P(None,x,y)",
                "sigma_body": "P(None,None,x,y)",
                "sigma_head": "P(None,None,x) analytic diagonal",
                "replicated_matrix_cube_per_process": False,
                "full_bz_k_rows": nk,
                "sigma_band_logical": int(sigma_axis.logical),
                "sigma_band_carrier": int(sigma_axis.carrier),
            },
            "omega": {
                "reference": "fixed-N MP1 mu",
                "reference_ev": float(state.mu_ry) * RYD_TO_EV,
                "n": int(omega_rel_ev.size),
                "min_ev": float(omega_rel_ev[0]),
                "max_ev": float(omega_rel_ev[-1]),
                "sha256": _hash_array(omega_rel_ev),
            },
            "coverage": {
                "real_max_ev": real_max_ev,
                "residue_required_max_ev": max_required,
                "imag_max_ev": IMAG_MAX_EV,
                "imag_tail_last_sample_ev": float(itail[-1]),
                "imag_tail_panel_max_ev": tail_panel_max_ev,
            },
            "certificate": {
                "tolerance_mev": CD_CONTROL_TOL_MEV,
                "rule": "max absolute real/imag component, body and head "
                        "certified independently without cancellation",
                "worst_component_mev": worst_control_mev,
                **controls,
            },
            "head": {
                **head_record,
                "dynamic_samples": "same streamed real/imag W batches",
                "on_shell_pin": "minimum-norm correction on the two "
                                 "bracketing omega nodes reproduces the "
                                 "existing eta=0 half-residue exactly",
            },
            "grids": grid_records,
        }
        if observer_receipt is not None:
            artifact["w_observer"] = observer_receipt
        artifact_path = os.path.join(input_dir, "internal_ff_cd.json")
        if rank == 0:
            Path(artifact_path).write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(
            "internal_ff_cd_selected_matrix_artifact")
        if worst_control_mev > CD_CONTROL_TOL_MEV:
            raise RuntimeError(
                "internal_ff_cd selected_matrix_block componentwise "
                f"quadrature certificate is {worst_control_mev:.6f} meV, "
                f"above {CD_CONTROL_TOL_MEV:g} meV; artifact: "
                f"{artifact_path}")
        return InternalFFResult(
            sigma_c_diag_ev=None,
            efermi_ev=float(state.mu_ry) * RYD_TO_EV,
            artifact_path=artifact_path,
            sigma_c_body_omega_ry=body,
            head_sigma_diag_w_kn_ry=dynamic_head,
            sigma_band_axis=sigma_axis)

    head_ev, head_record = _compute_head_diag_ev(
        wfns, target_k, target_b, state=state, config=config, meta=meta,
        mesh=mesh_xy, sym=sym, wfn=wfn, V_q=V_q,
        band_slices=band_slices, nb_chi_logical=nb_chi_logical,
        print_fn=print_fn)
    totals = real_results[-1] + imag_accum + head_ev
    controls, control_max_mev, contracts = _control_certificate(
        real_fine=real_results[-1], real_coarse=final_coarse_real,
        imag_fine=imag_accum, imag_coarse=imag_coarse,
        imag_tail=imag_tail)
    unresolved = np.flatnonzero(~contracts)
    labels = _target_labels(sym, target_k, target_b, band_slices.b0)
    records = []
    for i, label in enumerate(labels):
        records.append({
            **label,
            "resolved": bool(contracts[i]),
            "control_tolerance_mev": CD_CONTROL_TOL_MEV,
            "control_max_component_mev": float(control_max_mev[i]),
            "response_width_ev": float(RESPONSE_WIDTHS_EV[-1]),
            "sigma_c_ev": [float(totals[i].real),
                            float(totals[i].imag)],
            "body_residue_ev": [float(real_results[-1][i].real),
                                  float(real_results[-1][i].imag)],
            "body_imag_axis_ev": [float(imag_accum[i].real),
                                    float(imag_accum[i].imag)],
            "head_ev": [float(head_ev[i].real), float(head_ev[i].imag)],
            "real_fine_minus_coarse_mev": [
                float(1000.0 * controls[
                    "real_fine_minus_coarse_ev"][i].real),
                float(1000.0 * controls[
                    "real_fine_minus_coarse_ev"][i].imag)],
            "imag_fine_minus_coarse_mev": [
                float(1000.0 * controls[
                    "imag_fine_minus_coarse_ev"][i].real),
                float(1000.0 * controls[
                    "imag_fine_minus_coarse_ev"][i].imag)],
            "imag_full_minus_tail_mev": [
                float(1000.0 * controls[
                    "imag_full_minus_tail_ev"][i].real),
                float(1000.0 * controls[
                    "imag_full_minus_tail_ev"][i].imag)],
        })
    artifact = {
        "schema": 2,
        "route": "internal_ff_cd",
        "status": "pass" if unresolved.size == 0 else "unresolved",
        "jobid": os.environ.get("SLURM_JOB_ID", "unknown"),
        "source_commit": os.environ.get("LORRAX_SOURCE_COMMIT", "working-tree"),
        "wall_seconds": time.perf_counter() - t_start,
        "checkpoint_provenance": {
            "schema": CHECKPOINT_SCHEMA,
            **body_provenance,
        },
        "sharding": {
            "chi": "P(None,x,y)", "W": "P(None,x,y)",
            "replicated_nmu2_per_process": False,
            "contract_output": "one complex scalar per external target",
        },
        "stamps": {
            "occupation_hash": state.occ_hash,
            "occupation_family": "mp1",
            "occupation_width_ry": float(state.smearing_width_ry),
            "mu_ry": float(state.mu_ry),
            "mc_average_vcoul_body": bool(config.head.mc_average_vcoul_body),
            "bgw_metal_q0_treatment": config.head.bgw_metal_q0_treatment,
            "number_bands_chi": nb_chi_logical,
            "number_bands_sigma": nb_sigma_logical,
            "band_carrier_storage": nb_storage,
            "q_wedge": int(q_full.size), "q_full": nk,
        },
        "coverage": {
            "real_max_ev": real_max_ev,
            "residue_required_max_ev": max_required,
            "imag_max_ev": IMAG_MAX_EV,
            "imag_tail_last_sample_ev": float(itail[-1]),
            "imag_tail_panel_max_ev": tail_panel_max_ev,
        },
        "certificate": {
            "tolerance_mev": CD_CONTROL_TOL_MEV,
            "rule": "max absolute real/imag component of each independent "
                    "real-grid, imaginary-grid, and imaginary-tail control",
        },
        "grids": grid_records,
        "head": head_record,
        "unresolved_target_indices": unresolved.tolist(),
        "records": records,
    }
    if observer_receipt is not None:
        artifact["w_observer"] = observer_receipt
    artifact_path = os.path.join(input_dir, "internal_ff_cd.json")
    if rank == 0:
        Path(artifact_path).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices("internal_ff_cd_artifact")
    if unresolved.size:
        names = [
            f"k={labels[i]['k_crystal']}, band={labels[i]['band_one_based']} "
            f"(real-grid={1000.0 * controls['real_fine_minus_coarse_ev'][i].real:+.3f}"
            f"{1000.0 * controls['real_fine_minus_coarse_ev'][i].imag:+.3f}i, "
            f"imag-grid={1000.0 * controls['imag_fine_minus_coarse_ev'][i].real:+.3f}"
            f"{1000.0 * controls['imag_fine_minus_coarse_ev'][i].imag:+.3f}i, "
            f"imag-tail={1000.0 * controls['imag_full_minus_tail_ev'][i].real:+.3f}"
            f"{1000.0 * controls['imag_full_minus_tail_ev'][i].imag:+.3f}i meV)"
            for i in unresolved]
        raise RuntimeError(
            "internal_ff_cd componentwise quadrature certificate exceeded "
            f"{CD_CONTROL_TOL_MEV:g} meV for "
            f"{len(names)} cells; they are NAMED UNRESOLVED and no mean or "
            "full-deck claim is emitted. Artifact: " + artifact_path + "\n  "
            + "\n  ".join(names))

    sigma_irr = totals.reshape(k_wedge.size, sigma_bands.size)
    sigma_full = sigma_irr[np.asarray(sym.irr_idx_k, dtype=np.int32)]
    return InternalFFResult(
        sigma_c_diag_ev=np.asarray(sigma_full),
        efermi_ev=float(state.mu_ry) * RYD_TO_EV,
        artifact_path=artifact_path)


__all__ = [
    "InternalFFResult", "compute_internal_ff_cd", "make_direct_kernel",
    "make_weighted_contract_kernel", "real_grid", "imag_grid",
    "FREQUENCY_BATCH", "RESPONSE_WIDTHS_EV", "REAL_MAX_EV", "IMAG_MAX_EV",
    "REAL_STEP_EV", "REAL_COARSE_STEP_EV", "REAL_HARD_MAX_EV",
    "IMAG_FINE_INTERVALS", "CD_CONTROL_TOL_MEV",
]
