import os
import sys

# Ensure repo root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cohsex_isdf import read_cohsex_input


def test_read_cohsex_test_in():
    params = read_cohsex_input('cohsex_test.in')
    assert params['wfn_file'] == 'WFNsmall.h5'
    assert params['restart'] is True
