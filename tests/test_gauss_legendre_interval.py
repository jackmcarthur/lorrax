"""Focused contract for the shared host Gauss--Legendre interval owner."""
import numpy as np
import pytest

from common.gauss_legendre import (
    GAUSS_LEGENDRE_INTERVAL_PROVENANCE,
    gauss_legendre_interval,
)
from vcoul import gauss_legendre_interval as provider_interval_rule


def test_interval_rule_is_exact_through_the_gauss_degree_and_immutable():
    assert gauss_legendre_interval is provider_interval_rule
    nodes, weights = gauss_legendre_interval(4, -0.7, 2.3)
    for degree in range(8):
        got = np.sum(weights * nodes**degree)
        expected = (2.3**(degree + 1) - (-0.7)**(degree + 1)) / (degree + 1)
        np.testing.assert_allclose(got, expected, rtol=2.0e-14, atol=2.0e-14)
    assert not nodes.flags.writeable and not weights.flags.writeable
    assert "leggauss" in GAUSS_LEGENDRE_INTERVAL_PROVENANCE


@pytest.mark.parametrize(
    "args", [(0, 0.0, 1.0), (True, 0.0, 1.0), (3.5, 0.0, 1.0),
             (3, 1.0, 1.0), (3, np.nan, 1.0)])
def test_interval_rule_refuses_invalid_geometry(args):
    with pytest.raises(ValueError):
        gauss_legendre_interval(*args)
