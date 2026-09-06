"""Collective-compile gate for a density sweep with unequal local work."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_uneven_density_work_reaches_the_reduction(monkeypatch):
    """Six unit-occupied Bloch states give two electrons over three k rows."""
    from gw import kin_ion_io
    import jax
    from common.collectives import resolve_mesh

    if jax.process_count() != 4:
        pytest.skip("requires four processes with one GPU each")
    wfn = SimpleNamespace(
        physical_density_band_stop=2, occupation_state_capacity=1.0,
        cell_volume=1.0,
        physical_density_occupations=lambda *, k, unit_as_none: None)
    meta = SimpleNamespace(fft_grid=(2, 2, 2), nspinor=4)

    def load_box(wfn, meta, ik, b_hi, *, b_lo, bispinor, bispinor_lift):
        box = np.zeros((b_hi - b_lo, 4, 2, 2, 2), dtype=np.complex128)
        box[:, 0, 0, 0, 0] = 1.0
        return jax.device_put(box, jax.local_devices()[0])

    monkeypatch.setattr(kin_ion_io, "load_kpoint_fftbox_local", load_box)
    got = kin_ion_io.build_valence_density_distributed(
        wfn, SimpleNamespace(nk_tot=3), meta, mesh=resolve_mesh(),
        include_dirac_current=True, print_fn=print)
    assert kin_ion_io.rho_work_items(3, 2, 4) == [
        (0, 0, 1), (0, 1, 2), (1, 0, 1),
        (1, 1, 2), (2, 0, 1), (2, 1, 2)]
    np.testing.assert_allclose(got[0], 2.0, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(got[1:], np.zeros((3, 2, 2, 2)))
