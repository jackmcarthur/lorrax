"""Tier-0 fit-free full-frequency correlation-self-energy oracle.

This is the maintained form of the direct-SoS referee (sandbox claim 0363):
ordered-pair Adler--Wiser chi0, the production distributed Dyson solve,
canonical q-star unfold, and numerical contour deformation.  No MPA sample
or pole store is accepted by this module.  The route is deliberately an
O(N^4) oracle and is not eligible to become the production frequency model.

The large objects retain their natural two-dimensional mesh layout.  In
particular chi and W are ``P(None, 'x', 'y')``.  The contour contraction is
weighted before its mesh psum, so its replicated result is one scalar per
external target; neither an N_mu by N_mu matrix nor the target-q-band
spectral history is materialized on a process.
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
FREQUENCY_BATCH = 4
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
ARRAY_RECEIPT_SCHEME = "numpy-c-order-sha256-v1"
STAGE_TIMING_KEYS = (
    "dyson_solve_wall_seconds",
    "q_unfold_wall_seconds",
    "contract_host_checks_wall_seconds",
)


@dataclass(frozen=True)
class InternalFFResult:
    sigma_c_diag_ev: np.ndarray
    efermi_ev: float
    artifact_path: str


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


def _compute_head_diag_ev(wfns, target_k, target_b, *, state, config, meta,
                          mesh, sym, wfn, V_q, band_slices,
                          nb_chi_logical, velocity_path, print_fn):
    """The referee's pole-free static metallic head and eta=0 half residue."""
    from common.collectives import device_put_process_local
    from common.kq_mapping import kminq_idx_for_iq
    from .fermi_surface import star_symmetrize_weights, tetrahedron_delta_weights
    from .qsgw_head import (build_iteration_head_response,
                            finalize_iteration_head_sample,
                            load_dft_velocity_head)
    from .w_isdf import solve_w

    nk = int(meta.nk_tot)
    surface = tetrahedron_delta_weights(
        np.asarray(wfns.enk), np.asarray(sym.unfolded_kpts),
        tuple(int(x) for x in wfn.kgrid), float(state.mu_ry))
    surface = star_symmetrize_weights(surface, np.asarray(sym.irr_idx_k))
    surface_kn = jnp.asarray(surface * nk, dtype=jnp.float64)
    velocity = load_dft_velocity_head(
        velocity_path, mesh=mesh, wfn=wfn, meta=meta)
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
    response = build_iteration_head_response(
        delta_h_dft=None,
        forward_links=None,
        forward_neighbors=None,
        velocity_dft_cart=velocity.velocity_dft_cart,
        U_dft_to_qp=U,
        energies_qp_kn_ry=wfns.enk[:, :nb_storage],
        occupations_qp_kn=state.f_kn[:, :nb_storage],
        omegas_ry=np.asarray([0.0 + 0.0j], np.complex128),
        surface_weight_qp_kn=surface_kn[:, :nb_storage], mesh=mesh,
        kgrid=tuple(int(x) for x in wfn.kgrid),
        bvec_cart=velocity.reciprocal_lattice_cart,
        nb_logical=nb_logical,
        sigma_energies_ry=np.asarray(wfns.enk[:, wfns.slices.sigma]),
        efermi_ry=float(state.mu_ry), wfn=wfn, meta=meta, config=config,
        wfns_qp=wfns, eta_ry=0.0)
    gamma_map = kminq_idx_for_iq(sym, 0)[None, :]
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
    from file_io.paths import resolve_input_path
    from symmetry_maps import unfold_isdf_operator
    from .efermi import OccupationState, mp1_negative_derivative
    from .qsgw_head import preflight_dft_velocity_head
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
    velocity_path = resolve_input_path(
        input_dir, config.paths.parallel_transport_file)
    preflight_dft_velocity_head(
        velocity_path, mesh=mesh_xy, wfn=wfn, meta=meta)
    if rank == 0:
        print_fn(
            "  internal_ff_cd preflight: authenticated exact-DFT velocity "
            f"stage at {velocity_path}")
    if occupation_state is None:
        occupation_state = OccupationState.solve_mp1(
            wfns.enk, np.full(nk, 1.0 / nk), float(wfn.num_electrons),
            float(config.occ_broadening_ry),
            state_capacity=float(spin_degeneracy_factor(wfn)))
    state = occupation_state
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
    target_k = np.repeat(k_wedge, sigma_bands.size)
    target_b = np.tile(sigma_bands, k_wedge.size)
    n_targets = int(target_k.size)
    q_rows = np.arange(nk, dtype=np.int32)
    kmq_target = np.stack([
        kq_map_full[int(k), q_rows] for k in target_k]).astype(np.int32)
    energies_ev = np.asarray(wfns.enk, np.float64) * RYD_TO_EV
    occupations = np.asarray(state.f_kn, np.float64)
    target_e_ev = energies_ev[target_k, target_b]
    inner_e_ev = energies_ev[kmq_target]
    inner_f = occupations[kmq_target]
    x_signed = target_e_ev[:, None, None] - inner_e_ev + CENTER_SHIFT_EV
    x_abs = abs(x_signed)
    max_required = float(np.max(x_abs[:, :, :nb_sigma_logical]))
    real_max_ev = _real_coverage_max(max_required)
    residue_sign = np.where(x_signed >= 0.0, -(1.0 - inner_f), inner_f)

    direct = make_direct_kernel(
        mesh_xy, nb_logical=nb_chi_logical)
    contract = make_weighted_contract_kernel(
        mesh_xy, n_targets=n_targets,
        inner_stop=nb_sigma_logical)
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

    def frequency_batch(z_ry, coefficient_rows, global_frequency_index=None):
        """Evaluate a fixed-size referee frequency batch.

        The direct pair scan is substantially more efficient with its
        replicated frequency axis batched.  W is still solved and consumed
        one frequency at a time, so no additional ``N_mu^2`` object is ever
        replicated or retained as spectral history.
        """
        z = jnp.asarray(z_ry, dtype=jnp.complex128)
        chi_rows = [
            direct(wfns.psi_xn, wfns.psi_yn, jnp.asarray(kmq_wedge[i]),
                   wfns.enk, state.f_kn, surface, z)
            for i in range(q_full.size)]
        chi_bq = jnp.stack(chi_rows, axis=1)
        chi_bq.block_until_ready()
        chi_completed_at = time.perf_counter()
        stage_seconds = {key: 0.0 for key in STAGE_TIMING_KEYS}
        outputs = []
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
                            "internal_ff_cd nonfinite contracted stream at "
                            f"target {bad}")
                    row.append(host * (RYD_TO_EV / nk))
                outputs.append(row)
                del w_wedge, wc_wedge, wc_full
                stage_seconds["contract_host_checks_wall_seconds"] += (
                    time.perf_counter() - t_stage)
            except BaseException:
                if observer is not None:
                    observer.close(body_complete=False)
                raise
        return outputs, chi_completed_at, stage_seconds

    real_results = []
    grid_records = []
    final_coarse_real = None
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
        checkpoint = checkpoint_dir / f"real_eta_{width:.8f}.npz"
        (completed, loaded, chi_wall, solve_contract_wall,
         stage_wall) = _load_checkpoint(
            checkpoint, identity, n_targets, 2,
            return_stage_timings=True)
        if observer is not None:
            observer.require_checkpoint_prefix(arm_name, completed)
        accum, coarse_accum = loaded
        if rank == 0 and completed:
            print_fn(
                f"  internal_ff_cd resume real eta_W={width:g} eV: "
                f"{completed}/{len(grid)} frequencies from {checkpoint}")
        t_arm = time.perf_counter()
        for lo in range(completed, grid.size, FREQUENCY_BATCH):
            hi = min(lo + FREQUENCY_BATCH, grid.size)
            coefficient_rows = []
            for iw in range(lo, hi):
                coeff = _real_coefficients(grid, iw, x_abs, residue_sign)
                coeffs = [coeff]
                if iw in coarse_lookup:
                    cj = coarse_lookup[iw]
                    cgrid = grid[coarse]
                    ccoeff = _real_coefficients(
                        cgrid, cj, x_abs, residue_sign)
                    coeffs.append(ccoeff)
                coefficient_rows.append(coeffs)
            t_batch = time.perf_counter()
            z_batch = (grid[lo:hi] / RYD_TO_EV
                       + 1j * (width / RYD_TO_EV))
            if observer is None:
                values, t_after_chi, stage_batch = frequency_batch(
                    z_batch, coefficient_rows)
            else:
                values, t_after_chi, stage_batch = frequency_batch(
                    z_batch, coefficient_rows,
                    np.arange(arm_start + lo, arm_start + hi, dtype=np.int64))
            chi_wall += t_after_chi - t_batch
            solve_contract_wall += time.perf_counter() - t_after_chi
            for key in STAGE_TIMING_KEYS:
                stage_wall[key] += stage_batch[key]
            for jb, iw in enumerate(range(lo, hi)):
                accum += values[jb][0]
                if iw in coarse_lookup:
                    coarse_accum += values[jb][1]
            if observer is not None:
                observer.commit_prefix(arm_name, hi)
            if rank == 0:
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
    checkpoint = checkpoint_dir / "imaginary.npz"
    (completed, loaded, chi_wall, solve_contract_wall,
     stage_wall) = _load_checkpoint(
        checkpoint, identity, n_targets, 3,
        return_stage_timings=True)
    if observer is not None:
        observer.require_checkpoint_prefix(arm_name, completed)
    imag_accum, imag_coarse, imag_tail = loaded
    if rank == 0 and completed:
        print_fn(
            f"  internal_ff_cd resume imaginary: {completed}/{len(igrid)} "
            f"frequencies from {checkpoint}")
    t_arm = time.perf_counter()
    for lo in range(completed, igrid.size, FREQUENCY_BATCH):
        hi = min(lo + FREQUENCY_BATCH, igrid.size)
        coefficient_rows = []
        for iw in range(lo, hi):
            coeff = _imag_coefficients(igrid, iw, x_signed)
            coeffs = [coeff]
            if iw in icoarse_lookup:
                cj = icoarse_lookup[iw]
                cgrid = igrid[icoarse]
                ccoeff = _imag_coefficients(cgrid, cj, x_signed)
                coeffs.append(ccoeff)
            if iw < itail.size:
                tcoeff = _imag_coefficients(itail, iw, x_signed)
                coeffs.append(tcoeff)
            coefficient_rows.append(coeffs)
        z_imag_ev = np.maximum(igrid[lo:hi], IMAG_ORIGIN_LIMIT_EV)
        t_batch = time.perf_counter()
        z_batch = 1j * z_imag_ev / RYD_TO_EV
        if observer is None:
            values, t_after_chi, stage_batch = frequency_batch(
                z_batch, coefficient_rows)
        else:
            values, t_after_chi, stage_batch = frequency_batch(
                z_batch, coefficient_rows,
                np.arange(arm_start + lo, arm_start + hi, dtype=np.int64))
        chi_wall += t_after_chi - t_batch
        solve_contract_wall += time.perf_counter() - t_after_chi
        for key in STAGE_TIMING_KEYS:
            stage_wall[key] += stage_batch[key]
        for jb, iw in enumerate(range(lo, hi)):
            imag_accum += values[jb][0]
            offset = 1
            if iw in icoarse_lookup:
                imag_coarse += values[jb][offset]
                offset += 1
            if iw < itail.size:
                imag_tail += values[jb][offset]
        if observer is not None:
            observer.commit_prefix(arm_name, hi)
        if rank == 0:
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

    head_ev, head_record = _compute_head_diag_ev(
        wfns, target_k, target_b, state=state, config=config, meta=meta,
        mesh=mesh_xy, sym=sym, wfn=wfn, V_q=V_q,
        band_slices=band_slices, nb_chi_logical=nb_chi_logical,
        velocity_path=velocity_path, print_fn=print_fn)
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
