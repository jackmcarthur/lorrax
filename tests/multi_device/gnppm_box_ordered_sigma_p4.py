"""P=4 production gate for ordered GN residues through MPA box Sigma.

The reference cube was evaluated at ``origin/main@c1020045`` before the
box-path merge, with the incumbent PPM Sigma executor at target error 1e-10.
The box executor uses 2e-6, the tightest target admitted by its explicit
float32-runtime noise budget; the comparison tolerance is subordinate to
that certificate.
It is the full printed complex128 result of the seed-20260902 synthetic
broken-TR fixture, not a value copied from either implementation under test.
The evidence log and byte hash are recorded in the cx-rebase-box report.

Run only through the Perlmutter four-rank launcher.  This cell writes the
algebraic one-pole store, plans denominator boxes, executes the shared MPA
tau kernel with R+ on conduction branches and R- on valence branches, then
repeats the exact execution with ``LORRAX_DEBUG_GN_ODD_RESIDUE_OFF=1``.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
for _service in ("distrib_la", "minimax", "symmetry_maps", "wfn_loader"):
    sys.path.insert(0, os.path.join(ROOT, "services", _service, "src"))

from runtime import finalize_process, initialize_communicator_stack  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import barrier, device_put_process_local  # noqa: E402
from gw.gw_config import ScreeningDiagrams  # noqa: E402
from gw.ppm_sigma import PPMBuildResult, compute_sigma_c_ppm_omega_grid  # noqa: E402
from gw.wavefunction_bundle import BandSlices, Wavefunctions  # noqa: E402


_DEBUG_ENV = "LORRAX_DEBUG_GN_ODD_RESIDUE_OFF"
_EXPECTED_C1020045 = np.asarray([[[[[
    16.298442737528113 - 8.930530112033209j,
    11.870725419498351 - 6.219679921482220j,
    7.822521025624061 - 1.0664032131855106j,
    -3.094335859888907 - 14.144742210927202j,
], [
    15.030710499908743 + 0.5365615169446869j,
    5.175762593811499 - 7.845902123686349j,
    8.797995878208460 + 1.3073517603645806j,
    -3.303438680522252 - 8.721500161834832j,
], [
    11.402297467063656 - 8.146469926428324j,
    3.694126330934712 + 4.985794151976762j,
    9.843495811615831 - 0.448446891240210j,
    -0.14708530515050278 - 8.832730099390780j,
], [
    10.339809071011622 - 5.450787748402775j,
    4.727293463801406 - 6.240391061427903j,
    0.7174882490951893 - 4.485915413050049j,
    -3.7112296854327775 - 4.0421940140260375j,
]]]]], dtype=np.complex128)


def _fail(message):
    raise RuntimeError(f"[gnppm_box_ordered rank={jax.process_index()}] {message}")


def _fixture(mesh):
    rng = np.random.default_rng(20260902)
    nk, nb, nmu, ns, nocc = 1, 4, 4, 1, 2
    psi_xn = (
        rng.normal(size=(nk, ns, nmu, nb))
        + 1j * rng.normal(size=(nk, ns, nmu, nb))
    ).astype(np.complex128)
    psi_yn = (
        rng.normal(size=(nk, ns, nmu, nb))
        + 1j * rng.normal(size=(nk, ns, nmu, nb))
    ).astype(np.complex128)
    psi_xr = np.transpose(psi_xn, (0, 3, 1, 2)).copy()
    psi_yr = np.transpose(psi_yn, (0, 3, 1, 2)).copy()
    slices = BandSlices.from_band_edges(
        0, 0, nocc, nb, nb, b4_chi=nb, b4_sigma=nb)
    enk = np.asarray([[-1.0, -0.6, 0.7, 1.2]], dtype=np.float64)
    occ = np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    def put(value, spec):
        return device_put_process_local(value, NamedSharding(mesh, spec))

    wfns = Wavefunctions(
        psi_xn=put(psi_xn, P(None, None, "x", None)),
        psi_xr=put(psi_xr, P(None, None, None, "x")),
        psi_yr=put(psi_yr, P(None, None, None, "y")),
        psi_yn=put(psi_yn, P(None, None, "y", None)),
        enk=put(enk, P(None, None)),
        occ=put(occ, P(None, None)),
        slices=slices,
    )

    def hermitian():
        matrix = (
            rng.normal(size=(nmu, nmu))
            + 1j * rng.normal(size=(nmu, nmu)))
        return 0.5 * (matrix + matrix.conj().T)

    residue_plus = hermitian()
    residue_minus = hermitian()
    residue_even = 0.5 * (residue_plus + residue_minus)
    residue_odd = 0.5 * (residue_plus - residue_minus)
    omega = 1.5 + 0.3 * rng.uniform(size=(nmu, nmu))
    omega = 0.5 * (omega + omega.T)
    ppm = PPMBuildResult(
        omega_p=2.0,
        Wc0_q=put((-2.0 * residue_even / omega)[None], P(None, "x", "y")),
        B_q=put(residue_even[None], P(None, "x", "y")),
        Omega_q=put(omega[None], P(None, "x", "y")),
        valid_mask_q=put(
            np.ones((1, nmu, nmu), dtype=bool), P(None, "x", "y")),
        unfulfilled_fraction=0.0,
        n_nodes_static=1,
        B_odd_q=put(residue_odd[None], P(None, "x", "y")),
    )
    meta = SimpleNamespace(
        nk_tot=1, nkx=1, nky=1, nkz=1, n_rmu=nmu,
        b_id_4_sigma_user=nb)
    return wfns, ppm, meta


def _execute(wfns, ppm, meta, mesh, path, cache_dir):
    ppm_cfg = SimpleNamespace(invalid_mode="zero")
    sigma_cfg = SimpleNamespace(
        regularization_ev=0.25,
        regularization_floor_ev=0.0,
        window_edge_factor=1.5,
        fermi_reference="midgap",
        quadrature_eps=2.0e-6,
        quadrature_reduction_seconds=120.0,
        omega_step_ev=0.1,
    )
    mpa_cfg = SimpleNamespace(sigma_max_nodes=512, pole_batch_size=1)
    result = compute_sigma_c_ppm_omega_grid(
        wfns, ppm, meta, mesh,
        ppm_cfg=ppm_cfg,
        sigma_cfg=sigma_cfg,
        mpa_cfg=mpa_cfg,
        omega_grid_ry=np.asarray([0.05]),
        ansatz="gn_ppm",
        fit_store_path=path,
        screening_diagrams=ScreeningDiagrams.W_RPA,
        quadrature_cache_dir=cache_dir,
        print_fn=print if jax.process_index() == 0 else lambda *_a, **_k: None,
    )
    cube = np.asarray(multihost_utils.process_allgather(
        result.sigma_c_kij, tiled=True))
    odd = np.asarray(multihost_utils.process_allgather(
        result.sigma_c_odd_kij, tiled=True))
    return cube, odd


def main():
    if jax.process_count() != 4 or jax.device_count() != 4:
        _fail(
            "requires exactly four ranks and four global GPUs; got "
            f"{jax.process_count()} ranks and {jax.device_count()} devices")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    root = os.environ.get("GNPPM_BOX_GATE_DIR", os.getcwd())
    cache_dir = os.path.join(root, "uniform_rule_cache")
    ordered_path = os.path.join(root, "ordered_one_pole.h5")
    debug_path = os.path.join(root, "debug_even_one_pole.h5")
    if jax.process_index() == 0:
        os.makedirs(root, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        for path in (ordered_path, debug_path):
            if os.path.isfile(path):
                os.remove(path)
    barrier("gnppm_box_ordered_clean")

    with mesh:
        wfns, ppm, meta = _fixture(mesh)
        os.environ.pop(_DEBUG_ENV, None)
        ordered, odd = _execute(
            wfns, ppm, meta, mesh, ordered_path, cache_dir)
        os.environ[_DEBUG_ENV] = "1"
        debug, debug_odd = _execute(
            wfns, ppm, meta, mesh, debug_path, cache_dir)
        os.environ.pop(_DEBUG_ENV, None)

    delta_ref = float(np.max(np.abs(ordered - _EXPECTED_C1020045)))
    delta_debug = float(np.max(np.abs(ordered - debug)))
    odd_scale = float(np.max(np.abs(odd)))
    debug_odd_scale = float(np.max(np.abs(debug_odd)))
    scale = float(np.max(np.abs(_EXPECTED_C1020045)))
    if delta_ref > 5.0e-7 * max(scale, 1.0):
        _fail(
            "ordered box Sigma moved from c1020045 beyond its 5e-7 "
            f"relative gate: max|delta|={delta_ref:.12e}, scale={scale:.12e}")
    if not odd_scale > 1.0e-6 * max(float(np.max(np.abs(ordered))), 1.0):
        _fail(f"ordered odd Sigma is vacuous: max|Sigma_odd|={odd_scale:.12e}")
    if not delta_debug > 1.0e-6 * max(float(np.max(np.abs(ordered))), 1.0):
        _fail(f"debug D=0 arm did not flip Sigma: max|delta|={delta_debug:.12e}")
    if debug_odd_scale != 0.0:
        _fail(
            "debug D=0 arm did not close its exact odd twin to zero: "
            f"max|Sigma_odd|={debug_odd_scale:.12e}")

    barrier("gnppm_box_ordered_done")
    if jax.process_index() == 0:
        print(
            "[gnppm_box_ordered] PASS "
            f"max_delta_c1020045={delta_ref:.12e} "
            f"max_ordered_odd={odd_scale:.12e} "
            f"max_debug_flip={delta_debug:.12e} "
            f"max_debug_odd={debug_odd_scale:.12e}",
            flush=True)
    return 0


if __name__ == "__main__":
    finalize_process(main())
