"""Head-order coverage for the TRS-reduced SOC tutorial (host metadata only)."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from bse.bsemat_export import bgw_kernel_q_list, pair_head_momenta
from symmetry_maps import bgw_integer_q_to_fractional


def test_full_restart_grid_covers_trs_wedge_and_preserves_explicit_template():
    """A spatial WFN header cannot expand this TRS wedge; all81 pairs need the full grid."""
    fixture = Path(__file__).parent / 'fixtures/bsemat_mos2_trs_qgrid.json'
    h = json.loads(fixture.read_text())
    grid = np.asarray(h['grid'], dtype=np.int64)
    kpts, mtrx, bdot = (np.asarray(h[key]) for key in ('kpoints', 'mtrx', 'bdot'))
    native_wedge = np.asarray(h['native_eps_q'])
    incomplete = bgw_kernel_q_list(native_wedge, mtrx, bdot)
    assert len(incomplete) == 6
    with pytest.raises(ValueError, match=r'bsemat_q_list: pair \(0,1\)'):
        pair_head_momenta(kpts, bdot, incomplete)

    # The standalone carrier uses the owner's canonical full grid, not an
    # unverified q -> -q action inferred from the spatial WFN header.
    integer_grid = np.asarray(list(itertools.product(*(range(n) for n in grid))))
    full_q = bgw_integer_q_to_fractional(integer_grid, grid)
    complete = bgw_kernel_q_list(full_q, mtrx, bdot)
    assert len(complete) == 9
    heads, lengths = pair_head_momenta(kpts, bdot, complete)
    shifts = np.asarray(list(itertools.product(range(-2, 3), repeat=3)))
    for i, j in itertools.product(range(9), repeat=2):
        delta = kpts[i] - kpts[j]
        residual = heads[i, j] - delta
        assert np.max(np.abs(residual - np.rint(residual))) < 1e-12
        images = delta[None] - shifts
        independent_min = np.min(np.einsum('qi,ij,qj->q', images, bdot, images))
        assert np.isclose(lengths[i, j] ** 2, independent_min, atol=1e-12)

    # An explicit BerkeleyGW template retains its recorded q order and ties.
    template = bgw_kernel_q_list(np.asarray(h['bgw_reference_q']), mtrx, bdot)
    assert np.array_equal(template, np.asarray(h['expected_reference_kernel_q']))
    pair_head_momenta(kpts, bdot, template)
