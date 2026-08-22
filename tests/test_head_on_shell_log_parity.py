"""The GN-PPM head log scalar must come from the head KERNEL, not a twin.

Register row ``src/gw/ppm_pipeline.py:193-201``: the concise log printed
``-R_h/(Omega_h * V * N_k)`` while the kernel and the named
``sig_c_head(Edft).Re`` output column evaluate ``+R_h/(Omega_h * V * N_k)``
on shell.  Measured on the Si 6x6x6 two-update controls (JID 57243214):
log ``-0.8071 eV`` against ``+0.807048 eV`` in ``sigma_freq_debug.dat``
for the same occupied state.

These cells are the A/B that licensed deleting the duplicate formula:

* ``test_on_shell_helper_matches_the_kernel_on_shell`` -- the helper agrees
  with a full ``compute_ppm_head_sigma_diag`` evaluation at the on-shell
  frequency of a real occupied band, to 1e-12 relative.  This is the
  identity the log line now inherits.
* ``test_on_shell_helper_is_the_old_formula_with_the_sign_repaired`` -- the
  MAGNITUDE is unchanged from the deleted expression and only the sign
  moved, so the change is a sign repair and not a re-derivation.
* ``test_ppm_pipeline_does_not_restate_the_head_closed_form`` -- source
  gate: the driver must not grow a third spelling.

Pure numpy/host: no mesh, no devices.  Scope is the scalar on the log
line and its relation to the head kernel; it says nothing about the head
FIT (``fit_head_ppm``) that produced ``R_h`` / ``Omega_h``.
"""
import pathlib

import numpy as np
import pytest

from gw.head_correction import (
    HeadGNParams,
    compute_ppm_head_sigma_diag,
    on_shell_occupied_head_sigma_ry,
)

_CELL_VOLUME = 270.011
_NK_TOT = 216


def _head(R_h=0.65, omega_h=0.98):
    return HeadGNParams(
        omega_h_sq=omega_h ** 2, omega_h=omega_h,
        B_h=-2.0 * R_h * omega_h, R_h=R_h,
        wc_head_0=25.0, wc_head_iwp=5.0, vc0=100.0, omega_p=1.1)


def test_on_shell_helper_matches_the_kernel_on_shell():
    """The helper == the production kernel at omega = eps - E_F, occupied."""
    head = _head()
    efermi = 0.31
    # One occupied band at a NON-trivial energy, and the grid point that
    # sits exactly on its shell.  If the helper had baked in the wrong
    # delta this would move; it does not, because delta = 0 either way.
    eps = 0.12
    diag = compute_ppm_head_sigma_diag(
        head,
        omega_grid_ry=np.array([eps - efermi]),
        enk_ry=np.array([[eps]]),
        efermi_ry=efermi,
        n_occ=1,
        cell_volume=_CELL_VOLUME,
        nk_tot=_NK_TOT,
    )
    kernel_value = float(np.real(diag[0, 0, 0]))
    helper = on_shell_occupied_head_sigma_ry(
        head, cell_volume=_CELL_VOLUME, nk_tot=_NK_TOT)
    assert kernel_value != 0.0
    assert helper == pytest.approx(kernel_value, rel=1e-12), (
        "the log scalar no longer tracks the tensor the finalizer injects")


def test_on_shell_helper_is_the_old_formula_with_the_sign_repaired():
    """|value| unchanged; sign flipped.  A sign repair, not a re-derivation."""
    head = _head()
    deleted_expression = (
        -head.R_h / (head.omega_h * _CELL_VOLUME * _NK_TOT))
    helper = on_shell_occupied_head_sigma_ry(
        head, cell_volume=_CELL_VOLUME, nk_tot=_NK_TOT)
    assert helper == pytest.approx(-deleted_expression, rel=1e-9)
    assert abs(helper) == pytest.approx(abs(deleted_expression), rel=1e-9)
    # And the sign is the one the tensor carries for R_h > 0.
    assert helper > 0.0


def test_on_shell_helper_is_zero_for_a_degenerate_head():
    """Same degenerate-head answer as the kernel: 0, not a division blow-up."""
    head = _head(R_h=0.0)
    assert on_shell_occupied_head_sigma_ry(
        head, cell_volume=_CELL_VOLUME, nk_tot=_NK_TOT) == 0.0


def test_ppm_pipeline_does_not_restate_the_head_closed_form():
    """Source gate: no third spelling of R_h/(Omega_h * V * N_k) in the driver."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "ppm_pipeline.py").read_text()
    # Comments narrate the repaired formula on purpose; strip them before
    # matching (TASTE 17 -- a grep hit in a comment is not a fact about code).
    code = "\n".join(
        line.split("#", 1)[0] for line in src.splitlines())
    assert "head_gn.R_h" not in code, (
        "ppm_pipeline re-derived the head closed form; call "
        "head_correction.on_shell_occupied_head_sigma_ry instead")
    assert "on_shell_occupied_head_sigma_ry(" in code
