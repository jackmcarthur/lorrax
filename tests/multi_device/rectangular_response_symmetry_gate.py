"""Real-P4 gate for rectangular ordered-response symmetry and packing.

The symmetry service remains representation-neutral.  This caller-boundary
gate supplies unequal charge/transverse bases, checks the service against a
direct NumPy oracle on unitary and antiunitary rows, then passes the resulting
CT/TC tiles through the incumbent packed four-current response adapter.

Run only through the Perlmutter compute harness::

    lx run -N 1 -G 4 -n 1 python3 -u \
      tests/multi_device/rectangular_response_symmetry_gate.py --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_TESTS))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402


_IRR = np.asarray([0, 1, 0, 1], dtype=np.int32)
_SYM = np.asarray([0, 1, 2, 3], dtype=np.int32)
_Q_IRR = np.asarray([
    [0.25, 0.125, 0.0],
    [-0.125, 0.25, 0.0],
], dtype=np.float64)
_N_SYM_SPATIAL = 2


def _endpoint_tables(padded, logical, spatial_perm, spatial_wrap):
    """Build an explicit spatial/TRS table with an invariant padded tail."""
    identity = np.arange(int(padded), dtype=np.int32)
    perm = np.broadcast_to(identity, (4, int(padded))).copy()
    perm[1, :logical] = np.asarray(spatial_perm, dtype=np.int32)
    perm[2] = perm[0]
    perm[3] = perm[1]
    wraps = np.zeros((4, int(padded), 3), dtype=np.int32)
    wraps[1, :logical] = np.asarray(spatial_wrap, dtype=np.int32)
    wraps[3] = wraps[1]
    return perm, wraps


def _direct_rectangular(
        forward, reverse, *, left_perm, left_wrap, right_perm, right_wrap,
        logical_left, logical_right):
    """Per-element ordered CT<->TC oracle, with no service/JAX calls."""
    n_left, n_right = forward.shape[-2:]
    out = np.zeros((len(_IRR), n_left, n_right), dtype=np.complex128)
    for iq, (parent, row) in enumerate(zip(_IRR, _SYM)):
        parent, row = int(parent), int(row)
        lp = left_perm[row]
        rp = right_perm[row]
        phi_left = np.exp(
            2j * np.pi * (left_wrap[row] @ _Q_IRR[parent]))
        phi_right = np.exp(
            2j * np.pi * (right_wrap[row] @ _Q_IRR[parent]))
        if row >= _N_SYM_SPATIAL:
            # The antiunitary ordered response uses the independently formed
            # reverse orientation, not a same-tile shape inference.
            block = reverse[parent].T[np.ix_(lp, rp)]
            block = (np.conj(phi_left)[:, None] * block
                     * phi_right[None, :])
        else:
            block = forward[parent][np.ix_(lp, rp)]
            block = (phi_left[:, None] * block
                     * np.conj(phi_right)[None, :])
        block[logical_left:, :] = 0.0
        block[:, logical_right:] = 0.0
        out[iq] = block
    return out


def _direct_square(value, perm, wraps):
    """Historical same-basis/conjugating action, independently evaluated."""
    out = np.empty((len(_IRR), value.shape[-1], value.shape[-1]),
                   dtype=np.complex128)
    for iq, (parent, row) in enumerate(zip(_IRR, _SYM)):
        parent, row = int(parent), int(row)
        p = perm[row]
        phase = np.exp(2j * np.pi * (wraps[row] @ _Q_IRR[parent]))
        block = value[parent][np.ix_(p, p)]
        block = phase[:, None] * block * np.conj(phase)[None, :]
        out[iq] = np.conj(block) if row >= _N_SYM_SPATIAL else block
    return out


def _put(value, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(
        np.asarray(value), NamedSharding(mesh, P(None, "x", "y")))


def check_rectangular_response_symmetry(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from gw.photon_layout import (
        PhotonBasisLayout, pack_photon_response_tiles,
        unpack_photon_response_tiles)
    from symmetry_maps import unfold_isdf_operator

    if tuple(mesh.devices.shape) != (2, 2):
        raise AssertionError(f"gate requires a 2x2 mesh; got {mesh.devices.shape}")
    n_charge, p_charge = 10, 12
    n_transverse, p_transverse = 6, 8
    charge_perm, charge_wrap = _endpoint_tables(
        p_charge, n_charge,
        [1, 0, 3, 2, 5, 4, 7, 6, 9, 8],
        [[0, 0, 0], [1, 0, 0], [0, -1, 0], [1, -1, 0],
         [0, 0, 0], [-1, 0, 0], [0, 1, 0], [-1, 1, 0],
         [1, 0, 0], [0, -1, 0]])
    transverse_perm, transverse_wrap = _endpoint_tables(
        p_transverse, n_transverse,
        [2, 0, 1, 5, 3, 4],
        [[0, 0, 0], [0, 1, 0], [-1, 0, 0], [1, 1, 0],
         [0, -1, 0], [1, 0, 0]])

    rng = np.random.default_rng(20260825)
    ct_ibz = (rng.standard_normal((2, p_charge, p_transverse))
              + 1j * rng.standard_normal((2, p_charge, p_transverse)))
    tc_ibz = (rng.standard_normal((2, p_transverse, p_charge))
              + 1j * rng.standard_normal((2, p_transverse, p_charge)))
    # Padding is deliberately hostile at input.  A pass requires structural
    # output masks on both independently sharded endpoint axes.
    ct_ibz[:, n_charge:, :] = 41.0 + 43.0j
    ct_ibz[:, :, n_transverse:] = 47.0 + 53.0j
    tc_ibz[:, n_transverse:, :] = 59.0 + 61.0j
    tc_ibz[:, :, n_charge:] = 67.0 + 71.0j
    ct_dev, tc_dev = _put(ct_ibz, mesh), _put(tc_ibz, mesh)

    common = dict(
        irr_idx=_IRR, sym_idx=_SYM, q_irr_frac=_Q_IRR,
        mesh_xy=mesh, n_sym_spatial=_N_SYM_SPATIAL,
        trs_rule="pair_transpose")
    ct = unfold_isdf_operator(
        ct_dev, sym_perm=charge_perm, L_table=charge_wrap,
        right_sym_perm=transverse_perm, right_L_table=transverse_wrap,
        left_logical_extent=n_charge,
        right_logical_extent=n_transverse,
        trs_pair_q_ibz=tc_dev, **common)
    tc = unfold_isdf_operator(
        tc_dev, sym_perm=transverse_perm, L_table=transverse_wrap,
        right_sym_perm=charge_perm, right_L_table=charge_wrap,
        left_logical_extent=n_transverse,
        right_logical_extent=n_charge,
        trs_pair_q_ibz=ct_dev, **common)
    ct.block_until_ready()
    tc.block_until_ready()
    ct_host, tc_host = np.asarray(ct), np.asarray(tc)
    ct_ref = _direct_rectangular(
        ct_ibz, tc_ibz, left_perm=charge_perm, left_wrap=charge_wrap,
        right_perm=transverse_perm, right_wrap=transverse_wrap,
        logical_left=n_charge, logical_right=n_transverse)
    tc_ref = _direct_rectangular(
        tc_ibz, ct_ibz, left_perm=transverse_perm,
        left_wrap=transverse_wrap, right_perm=charge_perm,
        right_wrap=charge_wrap, logical_left=n_transverse,
        logical_right=n_charge)
    np.testing.assert_allclose(ct_host, ct_ref, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(tc_host, tc_ref, rtol=1.0e-13, atol=1.0e-13)
    wanted = NamedSharding(mesh, P(None, "x", "y"))
    for label, tile in (("CT", ct), ("TC", tc)):
        if not tile.sharding.is_equivalent_to(wanted, 3):
            raise AssertionError(f"{label} sharding is {tile.sharding!r}")

    anti_rows = np.flatnonzero(_SYM >= _N_SYM_SPATIAL)
    wrong_ct = _direct_rectangular(
        ct_ibz, np.swapaxes(ct_ibz, -1, -2),
        left_perm=charge_perm, left_wrap=charge_wrap,
        right_perm=transverse_perm, right_wrap=transverse_wrap,
        logical_left=n_charge, logical_right=n_transverse)
    wrong_scale = max(float(np.max(np.abs(ct_ref[anti_rows]))), 1.0e-300)
    wrong_partner_rel = float(np.max(
        np.abs(ct_ref[anti_rows] - wrong_ct[anti_rows]))) / wrong_scale
    if wrong_partner_rel < 0.1:
        raise AssertionError(
            f"antiunitary CT result did not distinguish its TC partner: "
            f"{wrong_partner_rel:.3e}")

    # A rectangular carrier cannot silently take the historical square-table
    # default.  Its right basis must be explicit even when extents happen to
    # agree on some production deck.
    try:
        unfold_isdf_operator(
            ct_dev, sym_perm=charge_perm, L_table=charge_wrap, **common)
    except ValueError as exc:
        if "right extent" not in str(exc):
            raise
    else:
        raise AssertionError("rectangular CT accepted one square table")

    # Red/green compatibility with every unchanged square call: no new
    # argument is supplied and the old conjugating action matches its direct
    # oracle on the same unitary/antiunitary map.
    square_ibz = (rng.standard_normal((2, p_charge, p_charge))
                  + 1j * rng.standard_normal((2, p_charge, p_charge)))
    square = unfold_isdf_operator(
        _put(square_ibz, mesh), irr_idx=_IRR, sym_idx=_SYM,
        sym_perm=charge_perm, L_table=charge_wrap, q_irr_frac=_Q_IRR,
        mesh_xy=mesh, n_sym_spatial=_N_SYM_SPATIAL)
    square.block_until_ready()
    square_ref = _direct_square(square_ibz, charge_perm, charge_wrap)
    np.testing.assert_allclose(
        np.asarray(square), square_ref, rtol=1.0e-13, atol=1.0e-13)

    # The GW boundary owns the packed C⊕T representation.  The service tile
    # drops into that incumbent path without a copy, alternate packer, or
    # reinterpretation of endpoint axes.
    layout = PhotonBasisLayout.from_centroid_extents(
        n_charge, n_transverse, mesh)
    packed = pack_photon_response_tiles(
        {(0, 1): ct, (1, 0): tc}, len(_IRR), layout, mesh,
        dtype=jnp.complex128)
    views = unpack_photon_response_tiles(packed, layout, mesh)
    np.testing.assert_array_equal(np.asarray(views[0][1]), ct_host)
    np.testing.assert_array_equal(np.asarray(views[1][0]), tc_host)
    np.testing.assert_array_equal(np.asarray(views[0][0]), 0.0)

    pad_max = max(
        float(np.max(np.abs(ct_host[:, n_charge:, :]))),
        float(np.max(np.abs(ct_host[:, :, n_transverse:]))),
        float(np.max(np.abs(tc_host[:, n_transverse:, :]))),
        float(np.max(np.abs(tc_host[:, :, n_charge:]))))
    return {
        "ct_shape": tuple(int(v) for v in ct.shape),
        "tc_shape": tuple(int(v) for v in tc.shape),
        "pad_max": pad_max,
        "ct_oracle_max_abs": float(np.max(np.abs(ct_host - ct_ref))),
        "tc_oracle_max_abs": float(np.max(np.abs(tc_host - tc_ref))),
        "square_oracle_max_abs": float(
            np.max(np.abs(np.asarray(square) - square_ref))),
        "wrong_partner_rel": wrong_partner_rel,
        "packed_shape": tuple(int(v) for v in packed.shape),
    }


def _main():
    import jax
    from jax.sharding import Mesh

    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    args = parser.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    devices = np.asarray(jax.devices())
    if devices.size != px * py:
        raise AssertionError(
            f"gate requested {px * py} devices, found {devices.size}")
    mesh = Mesh(devices.reshape(px, py), ("x", "y"))
    result = check_rectangular_response_symmetry(mesh)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    print(
        f"PASS rectangular_response_symmetry[{args.mesh}] "
        f"commit={commit} {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
