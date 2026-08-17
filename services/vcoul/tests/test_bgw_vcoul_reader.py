"""BGW ``write_vcoul`` grammar and exact integer-G mapping."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vcoul import fill_v_sphere_for_q, read_bgw_vcoul


FIXTURE = Path(__file__).with_name("data") / "bgw_vcoul_sigma_w60_excerpt.txt"


def test_streaming_reader_retains_first_sigma_q_walk_and_hashes_bytes():
    table = read_bgw_vcoul(FIXTURE)

    assert table.sha256 == "93ccab74af14772faa44e2424343629767ecb0180754a43e581d62952f9e907e"
    assert table.q_fracs.shape == (2, 3)
    assert table.n_G == 6
    assert table.q0_vcoul_raw() == 12699.491
    np.testing.assert_array_equal(
        table.G_miller_per_q[1],
        np.asarray([[0, 0, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.int32))


def test_direct_sphere_mapping_preserves_bgw_order_independently():
    table = read_bgw_vcoul(FIXTURE, compute_sha256=False)
    requested = np.asarray(
        [[-1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.int32)
    got = fill_v_sphere_for_q(
        table, (0.0, 0.0, 0.125), requested, 2.0,
        sym_mats_k=None, require_exact_sphere=True)
    np.testing.assert_array_equal(
        got, np.asarray([22.782782, 1298.6185, 26.502419]) / 2.0)
