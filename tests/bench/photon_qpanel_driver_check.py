"""Canonical SC/BSE driver checks with every coupled q parent in a separate panel."""
from unittest.mock import patch
from tests.hsuite.chain import main
with patch('gw.response_bank.photon_response_q_width',return_value=1):
    raise SystemExit(main())
