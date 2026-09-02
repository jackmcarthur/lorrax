"""P=4 GN-PPM Fermi-seam gate on the ordered-residue toy.

The two far endpoints reproduce the CrI3 ``[-90, +20] eV`` conditioning
geometry without making a large cube.  The seven inner points sample both
one-sided limits at 0.136 eV spacing.  The old mixed HGL/exact executor has a
fixed jump on this geometry; the shared causal box executor is smooth for
both the measured-broken-TR residues and its forced-``D=0`` red twin.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, ROOT)
from tests.multi_device import gnppm_box_ordered_sigma_p4 as base  # noqa: E402


np = base.np
jax = base.jax
Mesh = base.Mesh
multihost_utils = base.multihost_utils


_DEBUG_ENV = "LORRAX_DEBUG_GN_ODD_RESIDUE_OFF"
_RYD_TO_EV = 13.605693122994
_STEP_RY = 0.01
_OMEGA_RY = np.asarray(
    [
        -90.0 / _RYD_TO_EV,
        -3.0 * _STEP_RY,
        -2.0 * _STEP_RY,
        -_STEP_RY,
        0.0,
        _STEP_RY,
        2.0 * _STEP_RY,
        3.0 * _STEP_RY,
        20.0 / _RYD_TO_EV,
    ],
    dtype=np.float64,
)


def _execute(wfns, ppm, meta, mesh, path, cache_dir):
    ppm_cfg = base.SimpleNamespace(invalid_mode="zero")
    sigma_cfg = base.SimpleNamespace(
        regularization_ev=0.25,
        regularization_floor_ev=None,
        window_edge_factor=1.5,
        fermi_reference="midgap",
        # The CrI3-wide boxes' measured runtime-noise floor reaches 4.3e-7;
        # the production 5%-of-eps gate therefore requires eps >= 8.6e-6.
        quadrature_eps=2.0e-5,
        quadrature_reduction_seconds=120.0,
        omega_step_ev=_STEP_RY * _RYD_TO_EV,
    )
    mpa_cfg = base.SimpleNamespace(sigma_max_nodes=512, pole_batch_size=1)
    result = base.compute_sigma_c_ppm_omega_grid(
        wfns,
        ppm,
        meta,
        mesh,
        ppm_cfg=ppm_cfg,
        sigma_cfg=sigma_cfg,
        mpa_cfg=mpa_cfg,
        omega_grid_ry=_OMEGA_RY,
        ansatz="gn_ppm",
        fit_store_path=path,
        screening_diagrams=base.ScreeningDiagrams.W_RPA,
        quadrature_cache_dir=cache_dir,
        print_fn=print if jax.process_index() == 0 else lambda *_a, **_k: None,
    )
    cube = np.asarray(
        multihost_utils.process_allgather(result.sigma_c_kij, tiled=True)
    )
    # PPM always carries the one-entry cumulative band-bracket axis.
    if cube.ndim == 5:
        if cube.shape[0] != 1:
            base._fail(f"expected one band bracket, got shape {cube.shape}")
        cube = cube[0]
    return cube


def _fermi_metrics(cube):
    # The inner grid is [-3h,-2h,-h,0,h,2h,3h].  Quadratic one-sided
    # extrapolation removes ordinary slope and curvature, leaving a direct
    # estimator of a fixed discontinuity at zero.
    left = 3.0 * cube[3] - 3.0 * cube[2] + cube[1]
    right = 3.0 * cube[5] - 3.0 * cube[6] + cube[7]
    zero = cube[4]
    scale = max(float(np.max(np.abs(cube[1:8]))), 1.0)
    return (
        float(np.max(np.abs(left - right))),
        float(np.max(np.abs(zero - left))),
        float(np.max(np.abs(zero - right))),
        scale,
    )


def main():
    if jax.process_count() != 4 or jax.device_count() != 4:
        base._fail(
            "requires exactly four ranks and four global GPUs; got "
            f"{jax.process_count()} ranks and {jax.device_count()} devices"
        )
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    root = os.environ.get("GNPPM_FERMI_SEAM_GATE_DIR", os.getcwd())
    cache_dir = os.path.join(root, "uniform_rule_cache")
    ordered_path = os.path.join(root, "ordered_fermi_seam_one_pole.h5")
    debug_path = os.path.join(root, "debug_even_fermi_seam_one_pole.h5")
    if jax.process_index() == 0:
        os.makedirs(root, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        for path in (ordered_path, debug_path):
            if os.path.isfile(path):
                os.remove(path)
    base.barrier("gnppm_box_fermi_seam_clean")

    with mesh:
        wfns, ppm, meta = base._fixture(mesh)
        os.environ.pop(_DEBUG_ENV, None)
        ordered = _execute(wfns, ppm, meta, mesh, ordered_path, cache_dir)
        os.environ[_DEBUG_ENV] = "1"
        debug = _execute(wfns, ppm, meta, mesh, debug_path, cache_dir)
        os.environ.pop(_DEBUG_ENV, None)

    rows = (("ordered", ordered), ("D=0", debug))
    for name, cube in rows:
        jump, zero_left, zero_right, scale = _fermi_metrics(cube)
        tolerance = 2.0e-4 * scale
        if max(jump, zero_left, zero_right) > tolerance:
            base._fail(
                f"{name} Sigma is discontinuous at E_F: "
                f"one_sided_jump={jump:.12e}, zero_left={zero_left:.12e}, "
                f"zero_right={zero_right:.12e}, tolerance={tolerance:.12e}"
            )

    base.barrier("gnppm_box_fermi_seam_done")
    if jax.process_index() == 0:
        ordered_metrics = _fermi_metrics(ordered)
        debug_metrics = _fermi_metrics(debug)
        print(
            "[gnppm_box_fermi_seam] PASS "
            f"ordered_jump={ordered_metrics[0]:.12e} "
            f"ordered_zero_left={ordered_metrics[1]:.12e} "
            f"ordered_zero_right={ordered_metrics[2]:.12e} "
            f"debug_jump={debug_metrics[0]:.12e} "
            f"debug_zero_left={debug_metrics[1]:.12e} "
            f"debug_zero_right={debug_metrics[2]:.12e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    base.finalize_process(main())
