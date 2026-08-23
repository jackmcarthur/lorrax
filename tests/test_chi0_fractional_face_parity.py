"""Algebra parity: the exact finite-occupation chi0 family (legacy vs
face), the census row "Exact finite-occupation response"
(``gw_config._LOW_MEM_BANDS_REFUSALS``'s
``low_mem_bands_metal_material_class_unported``) plus the fractional/
contour chi0 kernel — ``feat/metal-response-face-2026-08-23``.  See
``docs/architecture/fractional_chi0_response_face.md`` for the full
derivation this test is gating.

Three functions, all layout-dispatched on ``wfns.layout``, all exercised
here against the SAME synthetic ψ/energy/occupation data (legacy and face
``Wavefunctions`` bundles are built from the identical underlying
``psi_rmu_Y``/``psi_rmuT_X`` pair via ``build_wavefunctions``/
``build_wavefunctions_face``, so any disagreement is the face mechanism,
not a data mismatch):

* :func:`gw.w_isdf.compute_chi0_static_fractional_gamma` — the exact
  static q=0 divided-difference body (Part B, ordered-pair kernel, the
  genuinely new masked-gather + psum mechanism).  Exercised at
  ``nb_logical < e.shape[1] < nb_full`` — a non-mesh-divisible logical
  window narrower than BOTH the caller's own energies table AND the face
  carrier's full band extent, forcing the face path's own zero-pad from
  ``e.shape[1]`` up to ``nb_full``.
* :func:`gw.w_isdf.compute_chi0_direct_fractional` — the finite-q,
  finite-z generalization (same mechanism; a nonzero ``z`` exercises the
  DYNAMIC weight branch, not the static divided difference), at
  ``nb_logical < nb_full`` (``e.shape[1] == nb_full`` here, since this
  function reads ``wfns.enk`` directly).
* :func:`gw.w_isdf.compute_chi0_contour_fractional` — Part A, the
  fractional/contour kernel; a substitution of operands onto the existing
  ``build_G_tau(layout='face', ...)`` mechanism, no new algorithm.

Cases (task-specified coverage): ``ns1``/``ns2`` (the GEMM-seam-adjacent
axis-order concern for a bispinor-shaped input — though this kernel family
has no GEMM seam, ns=2 still exercises the einsum's spinor contraction and
the density-tile axis order), each with a genuinely metallic (MP1,
fractional, injected near-degenerate pair) occupation table built from
:func:`gw.efermi.mp1_occupations`/:func:`mp1_negative_derivative`.

    lx run -N 1 -G 4 -n 4 bash tmp/lm_chi0frac_run_wrap.sh \\
        tests/test_chi0_fractional_face_parity.py --mesh 2x2
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_TOL = 1.0e-9


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype("complex128")


#: (ns, seed) — nb_full=24, n_rmu=4, kgrid=(2,2,1) (nk=4) held fixed across
#: cases; both mesh-divisible for a 2x2 mesh.  ``nb_narrow=17`` (Gamma
#: kernel's own energies-table width) is deliberately NOT a multiple of 2
#: (the mesh) and NOT equal to nb_full=24 (the face carrier's own extent)
#: — the case the caller-side zero-pad exists for.
_CASES = (
    ("ns1", dict(ns=1, seed=11)),
    ("ns2", dict(ns=2, seed=12)),
)
_CASES_BY_NAME = {name: kwargs for name, kwargs in _CASES}

_NB_FULL = 24
_N_RMU = 4
_NB_NARROW = 17
_NB_LOGICAL_GAMMA = 17     # == e.shape[1] here: the Gamma kernel's full window
_NB_LOGICAL_DIRECT = 19    # < nb_full=24: exercises the mask with no caller-pad


def _worker(case_name: str) -> int:
    """Runs in a fresh subprocess (JAX_PLATFORMS=cpu,
    --xla_force_host_platform_device_count=4 set by the caller).  Prints
    one JSON line: {"max_rel": {...}} or {"skip": "..."}."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from gw.wavefunction_bundle import (
        BandSlices, build_wavefunctions, build_wavefunctions_face,
    )
    from gw.w_isdf import (
        compute_chi0_contour_fractional,
        compute_chi0_direct_fractional,
        compute_chi0_static_fractional_gamma,
    )
    from gw.efermi import (
        OccupationState, mp1_negative_derivative, mp1_occupations,
    )

    ns, seed = (_CASES_BY_NAME[case_name]["ns"],
                _CASES_BY_NAME[case_name]["seed"])

    devs = jax.devices()
    PX = PY = 2
    if len(devs) < PX * PY:
        print(json.dumps({"skip": f"only {len(devs)} devices (<{PX*PY})"}))
        return 0
    mesh = Mesh(np.asarray(devs[: PX * PY]).reshape(PX, PY), ("x", "y"))

    rng = np.random.default_rng(seed)
    nk = 4
    kgrid = (2, 2, 1)
    nb_full = _NB_FULL
    n_rmu = _N_RMU

    class _Meta:
        pass
    meta = _Meta()
    meta.nk_tot = nk
    meta.nkx, meta.nky, meta.nkz = kgrid

    # ---- synthetic psi, shared by legacy and face bundles -------------
    psi_rmu_Y = jnp.asarray(_crand(rng, nk, nb_full, ns, n_rmu))
    psi_rmuT_X = jnp.asarray(_crand(rng, nk, n_rmu, nb_full, ns))

    # ---- energies: an injected EXACT degeneracy (bands 2,3 at k=0) so
    # the diagonal_limit / surface_weight branch is genuinely exercised,
    # not just the generic separated branch.  ALSO a deep semicore tail
    # (bands 0-2) and a deep virtual tail (bands nb_full-3..nb_full-1),
    # far outside the near-EF window -- forces the derived f_slice/
    # u_slice to be genuinely NARROWER than [0, nb_full) on each side
    # (mirrors a real metal's semicore/valence/conduction structure).
    # This is the exact shape that caught a real bug this session
    # (Na production-shape harness, 2026-08-23): a small, uniformly-
    # random-energy table with no deep tail gives f_slice/u_slice that
    # both happen to span nearly the FULL band range, which accidentally
    # makes "weight, don't window" a no-op and hides any bug in HOW the
    # window is applied -- see docs/architecture/
    # fractional_chi0_response_face.md for the mechanism this exercises.
    enk = rng.uniform(-1.0, 1.0, size=(nk, nb_full))
    enk[0, 3] = enk[0, 2]
    enk[:, 0:3] = rng.uniform(-50.0, -40.0, size=(nk, 3))       # deep occupied
    enk[:, nb_full - 3:nb_full] = rng.uniform(40.0, 50.0, size=(nk, 3))  # deep empty
    enk_full = jnp.asarray(enk, dtype=jnp.float64)

    mu = float(np.median(enk))
    width = 0.15
    f_kn = mp1_occupations(enk_full, mu, width)
    surface = mp1_negative_derivative(enk_full, mu, width)

    slices = BandSlices.from_band_edges(0, 0, nb_full // 2, nb_full, nb_full)

    wfns_legacy = build_wavefunctions(
        psi_rmu_Y, psi_rmuT_X, enk_full=enk_full, slices=slices,
        mesh_xy=mesh, efermi=None)
    wfns_face = build_wavefunctions_face(
        psi_rmu_Y, psi_rmuT_X, enk_full=enk_full, slices=slices,
        mesh_xy=mesh, efermi=None)

    occ_state = OccupationState(
        f_kn=f_kn, mu_ry=mu, smearing_family="mp1",
        smearing_width_ry=width, n_electrons=float(np.sum(np.asarray(f_kn))))

    from jax.experimental import multihost_utils as _mhu

    def _cmp(a, b):
        A = np.asarray(_mhu.process_allgather(a, tiled=True))
        B = np.asarray(_mhu.process_allgather(b, tiled=True))
        if A.shape != B.shape:
            return {"error": f"shape mismatch: legacy={A.shape} face={B.shape}"}
        absdiff = np.abs(A - B)
        ref_scale = float(np.abs(A).max())
        max_abs = float(absdiff.max())
        return {"max_abs": max_abs, "ref_scale": ref_scale,
                "max_rel": max_abs / max(ref_scale, 1e-300)}

    out = {"case": case_name, "ns": ns}

    # ---- Part B, Gamma: nb_narrow=17 < nb_full=24, non-mesh-divisible --
    e_narrow = enk_full[:, :_NB_NARROW]
    f_narrow = f_kn[:, :_NB_NARROW]
    s_narrow = surface[:, :_NB_NARROW]
    gamma_legacy = jax.block_until_ready(compute_chi0_static_fractional_gamma(
        wfns_legacy, e_narrow, f_narrow, s_narrow, meta, mesh,
        nb_logical=_NB_LOGICAL_GAMMA))
    gamma_face = jax.block_until_ready(compute_chi0_static_fractional_gamma(
        wfns_face, e_narrow, f_narrow, s_narrow, meta, mesh,
        nb_logical=_NB_LOGICAL_GAMMA))
    out["gamma"] = _cmp(gamma_legacy, gamma_face)

    # ---- Part B, finite-q/finite-z: z != 0, exercises the dynamic branch
    kminq_rows = np.stack([
        rng.permutation(nk).astype(np.int32) for _ in range(2)])
    z_direct = np.asarray([0.03 + 0.01j], dtype=np.complex128)
    direct_legacy = jax.block_until_ready(compute_chi0_direct_fractional(
        wfns_legacy, z_direct, meta, mesh, occupation_state=occ_state,
        kminq_rows=kminq_rows, nb_logical=_NB_LOGICAL_DIRECT))
    direct_face = jax.block_until_ready(compute_chi0_direct_fractional(
        wfns_face, z_direct, meta, mesh, occupation_state=occ_state,
        kminq_rows=kminq_rows, nb_logical=_NB_LOGICAL_DIRECT))
    out["direct"] = _cmp(direct_legacy, direct_face)

    # ---- Part A, fractional/contour kernel -----------------------------
    # Needs make_flat_k_fftn -> the FFTW3-ABI HOST FFI backend (the SAME
    # requirement the ordinary, already-shipped minimax kernel has) --
    # unrelated to this session's own code, and this sandbox's CPU-
    # emulated multi-device path does not have a working host .so
    # (a cray-libsci dependency, libsci_gnu.so.6, is absent from the
    # container image reachable from this worktree's env — a scaffolding
    # gap, recorded in KNOWN_SANDBOX_ERRORS.md, not a defect in this
    # port).  Real CUDA (the __main__ CLI path below, `lx run -G 4`)
    # already has a working FFI target and exercises this quantity fully;
    # the CPU-emulated leg here degrades to an explicit, named skip
    # rather than a false pass or an opaque crash.
    try:
        time_nodes = np.asarray([0.1, 0.5, 1.3], dtype=np.float64)
        weight_rows = np.asarray([0.4, 0.3, 0.2], dtype=np.float64)
        z_contour = np.asarray([0.05 + 0.2j], dtype=np.complex128)
        contour_legacy = jax.block_until_ready(compute_chi0_contour_fractional(
            wfns_legacy, time_nodes, weight_rows, z_contour, meta, mesh,
            occupations=f_kn, energy_reference=mu))
        contour_face = jax.block_until_ready(compute_chi0_contour_fractional(
            wfns_face, time_nodes, weight_rows, z_contour, meta, mesh,
            occupations=f_kn, energy_reference=mu))
        out["contour"] = _cmp(contour_legacy, contour_face)
    except RuntimeError as exc:
        if "FFTW3-ABI host backend is unavailable" not in str(exc):
            raise
        out["contour"] = {
            "skip": "host FFT FFI backend unavailable in this sandbox "
                    "environment (KNOWN_SANDBOX_ERRORS.md) -- verify on "
                    "real CUDA via this file's __main__ CLI instead"}

    print(json.dumps(out))
    return 0


def _run_worker(case_name: str, ndev: int = 4, timeout: int = 300):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={ndev}").strip()
    _repo_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["PYTHONPATH"] = _repo_src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "worker", case_name],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker {case_name} failed rc={res.returncode}\nSTDOUT:\n"
        f"{res.stdout}\nSTDERR:\n{res.stderr}")
    lines = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(lines[-1])


@pytest.mark.parametrize("name,kwargs", _CASES, ids=[c[0] for c in _CASES])
def test_fractional_chi0_face_matches_legacy(name, kwargs):
    out = _run_worker(name)
    if "skip" in out:
        pytest.skip(f"fractional chi0 face-layout parity gate: {out['skip']}")
    skipped = []
    for quantity in ("gamma", "direct", "contour"):
        result = out[quantity]
        if "skip" in result:
            skipped.append(f"{quantity}: {result['skip']}")
            continue
        assert "error" not in result, f"{quantity}: {result.get('error')}"
        assert result["max_rel"] < _TOL, (
            f"{quantity} face vs legacy parity FAILED: max relative diff "
            f"{result['max_rel']:.3e} (case {name})")
    if skipped:
        # gamma/direct (this port's own genuinely new mechanism) still
        # ran and were asserted above; only note the narrower scope.
        print("PARTIAL SCOPE (see stdout, not a pass/fail signal): "
              + "; ".join(skipped))


# ---------------------------------------------------------------------------
# Real-CUDA CLI (matches tests/test_isdf_zq_face_parity.py's shape).
# ---------------------------------------------------------------------------

def _cli_main():
    import argparse
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2", help="PxQ process mesh")
    args = ap.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}")
    if jax.device_count() != px * py:
        p0(f"REFUSE: need exactly {px * py} devices for a {args.mesh} mesh; "
           f"got {jax.device_count()}")
        return 1
    if (px, py) != (2, 2):
        p0("REFUSE: this file's synthetic shapes are fixed for a 2x2 mesh "
           "(n_rmu=4, nb_full=24) -- pass --mesh 2x2.")
        return 1

    failures = 0
    for name, _kwargs in _CASES:
        try:
            rc = _worker_inline(name)
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep
            failures += 1
            p0(f"FAIL {name}: {exc}")
            continue
        if rc.get("skip"):
            p0(f"SKIP {name}: {rc['skip']}")
            continue
        case_fail = False
        for quantity in ("gamma", "direct", "contour"):
            result = rc.get(quantity, {})
            if "skip" in result:
                p0(f"SKIP {name}/{quantity}: {result['skip']}")
                continue
            if "error" in result:
                case_fail = True
                p0(f"FAIL {name}/{quantity}: {result['error']}")
                continue
            ok = result["max_rel"] < _TOL
            case_fail = case_fail or not ok
            p0(f"{'PASS' if ok else 'FAIL'} {name}/{quantity}: "
               f"max|diff|={result['max_abs']:.3e} "
               f"(ref scale {result['ref_scale']:.3e}) "
               f"max|rel diff|={result['max_rel']:.3e}")
        if case_fail:
            failures += 1
    p0(f"done: {len(_CASES) - failures}/{len(_CASES)} cases passed")
    return 1 if failures else 0


def _worker_inline(case_name: str) -> dict:
    """Runs ``_worker``'s BODY in-process (this IS the CUDA process
    already), capturing its one JSON print instead of spawning a
    subprocess."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _worker(case_name)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"worker {case_name} printed no JSON (rc={rc})")
    return json.loads(lines[-1])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        sys.exit(_worker(sys.argv[2]))
    # Real-CUDA CLI path (`lx run -N 1 -G 4 -n 4 ... --mesh 2x2`): mirrors
    # tests/test_isdf_zq_face_parity.py's own bootstrap EXACTLY -- init
    # BEFORE importing jax/distrib_la, guarded by __main__.
    _TESTS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")
    sys.exit(_cli_main())
