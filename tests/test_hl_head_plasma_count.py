"""The HL-PPM head plasma frequency counts electrons, not bands (CPU).

``meta.nelec`` is the occupied-band count, half the electron count on a
spin-degenerate scalar WFN, so ω_p² = 16π N_e / V read from it was half the
BGW value. The head source and the config are stubs; the HL arm is the
production one.
"""
from types import SimpleNamespace

import numpy as np

from ffi import _services

_services.ensure_on_path()

from gw import head_correction, ppm_pipeline  # noqa: E402


def test_hl_head_reads_the_wfn_electron_count():
    sample = head_correction.HeadSample(
        vc0=10.0 + 0j, wcoul0=1.0 + 0j, source="stub", omega=0j)
    source = SimpleNamespace(at=lambda omega: sample)
    config = SimpleNamespace(
        ppm=SimpleNamespace(head_omega_h_ry=None),
        compute_mode=SimpleNamespace(ppm_model="hl"))
    volume, n_electrons = 270.0, 8.0          # Si: 8 electrons in 4 bands
    meta = SimpleNamespace(nelec=4, cell_volume=volume)
    got = ppm_pipeline._fit_head_correction(
        source, config=config, meta=meta, num_electrons=n_electrons,
        probe_omega=0j, print_fn=lambda *_: None)
    want = head_correction.fit_head_hl_analytic(
        vc0=10.0, wcoul0_static=1.0,
        omega_p_sq_ry=16.0 * np.pi * n_electrons / volume)
    assert got.omega_h_sq == want.omega_h_sq
