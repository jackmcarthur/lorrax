import types
import sys
import os
import numpy as np
from numpy.polynomial.laguerre import laggauss

# Ensure repository root is on the module search path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Provide a minimal CuPy stub so modules depending on CuPy import without error
cupy_stub = types.ModuleType('cupy')
for name in ['exp', 'sqrt', 'linspace', 'asarray', 'sum', 'zeros', 'arange']:
    setattr(cupy_stub, name, getattr(np, name))
# minimal CUDA runtime interface
cupy_stub.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 0))
cupy_stub.complex128 = np.complex128
sys.modules.setdefault('cupy', cupy_stub)

from common import wfnreader


def ctsp_chi(A_v, A_c, valence, conduction, z, tau, w, E_gap):
    """Evaluate chi(omega=0) using the CTSP quadrature."""
    chi = 0.0
    dv = np.exp(-z * tau[:, None] * (valence.max() - valence[None, :]))
    dc = np.exp(-z * tau[:, None] * (conduction[None, :] - conduction.min()))
    rho_v = dv @ A_v
    rho_c = dc @ A_c
    chi = np.sum(-2 * z * w * np.exp(-(z * E_gap - 1) * tau) * rho_c * rho_v)
    return chi


def direct_chi(A_v, A_c, valence, conduction):
    """Direct evaluation of chi(omega=0)."""
    denom = conduction[None, :] - valence[:, None]
    return -2 * np.sum(A_v[:, None] * A_c[None, :] / denom)


def test_ctsp_matches_direct():
    wfn = wfnreader.WFNReader('examples/cohsex_test/WFNsmall.h5')
    energies = wfn.energies[0, 0]
    valence = energies[: wfn.nelec]
    conduction = energies[wfn.nelec :]

    # Define CTSP grid based on the energies from this k-point
    class _W: pass
    val_win = _W(); val_win.start_energy = float(valence.min()); val_win.end_energy = float(valence.max())
    cond_win = _W(); cond_win.start_energy = float(conduction.min()); cond_win.end_energy = float(conduction.max())
    E_gap = cond_win.start_energy - val_win.end_energy
    E_bw = cond_win.end_energy - val_win.start_energy
    z = np.sqrt(E_bw / E_gap)

    # Use a reasonably dense Gauss-Laguerre grid
    n_tau = 50
    tau, w = laggauss(n_tau)

    test_sets = [
        (np.ones_like(valence), np.ones_like(conduction)),
        (np.arange(len(valence)), np.arange(len(conduction)) + 2),
        (np.random.default_rng(0).normal(size=len(valence)), np.random.default_rng(1).normal(size=len(conduction))),
    ]

    for A_v, A_c in test_sets:
        direct = direct_chi(A_v, A_c, valence, conduction)
        ctsp = ctsp_chi(A_v, A_c, valence, conduction, z, tau, w, E_gap)
        assert np.isclose(ctsp, direct, rtol=1e-2, atol=1e-6)

