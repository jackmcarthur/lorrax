import os
import sys
import types
import numpy as np

# Ensure repository root is on the module search path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Provide a minimal CuPy stub so modules depending on CuPy import without error
cupy_stub = types.ModuleType("cupy")
for name in [
    "exp",
    "sqrt",
    "linspace",
    "asarray",
    "sum",
    "zeros",
    "arange",
]:
    setattr(cupy_stub, name, getattr(np, name))
cupy_stub.linalg = types.SimpleNamespace(norm=np.linalg.norm)
cupy_stub.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 0))
cupy_stub.complex128 = np.complex128
cupy_stub.float64 = np.float64
sys.modules.setdefault("cupy", cupy_stub)

import wfnreader
from get_windows import get_window_info


def test_ctsp_vs_direct_range_of_bands():
    """Compare CTSP quadrature to the direct energy-sum for several coefficients."""

    wfn = wfnreader.WFNReader("WFN.h5")

    # Pull the first window pair as w_isdf would
    win = get_window_info(0.01, wfn)[0]

    energies = wfn.energies[0, 0]
    val_slice = slice(wfn.nelec - 26, wfn.nelec)
    cond_slice = slice(wfn.nelec, wfn.nelec + 10)
    E_v = energies[val_slice]
    E_c = energies[cond_slice]

    z = win.z_lm
    tau = win.tau_i
    w = win.w_i
    E_c_min = win.cond_window.start_energy
    E_v_max = win.val_window.end_energy
    E_gap = E_c_min - E_v_max

    def direct(A_v, A_c):
        denom = E_c[np.newaxis, :] - E_v[:, np.newaxis]
        return -1.0 * np.sum(A_v[:, None] * A_c[None, :] / denom)

    def ctsp(A_v, A_c):
        rho_m = np.exp(-z * tau[:, None] * (E_c - E_c_min)) @ A_c
        rhobar = np.exp(-z * tau[:, None] * (E_v_max - E_v)) @ A_v
        factors = np.exp(-(z * E_gap - 1.0) * tau)
        return -1.0 * z * np.sum(w * factors * rho_m * rhobar)

    rng = np.random.default_rng(0)
    coeff_sets = [
        (np.ones_like(E_v), np.ones_like(E_c)),
        (np.linspace(1.0, 2.0, len(E_v)), np.linspace(1.0, 2.0, len(E_c))),
        (rng.random(len(E_v)), rng.random(len(E_c))),
    ]

    for A_v, A_c in coeff_sets:
        chi_dir = direct(A_v, A_c)
        chi_ctsp = ctsp(A_v, A_c)
        assert np.isclose(chi_ctsp, chi_dir, rtol=5e-2)
        print(f"chi_dir: {chi_dir:.5f}, chi_ctsp: {chi_ctsp:.5f}")

test_ctsp_vs_direct_range_of_bands()
