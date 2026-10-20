"""Bounded native P4 software proof; NOT material/SC/Pauli qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
import warnings

REVISION = "38dbea1665518cb13018b972bf6dd2a059910ed5"


def fixture(mesh):
    import jax
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.wfn_layout import PSI_NMU_SPEC, PSI_MUN_SPEC
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.wavefunction_bundle import BandSlices, ParentGreenCarrier, Wavefunctions
    from symmetry_maps import SymMaps

    rng = np.random.default_rng(1993)
    grid, coarse, nb, ns, mu = (8, 8, 1), (4, 4, 1), 4, 2, 4
    nk = int(np.prod(grid))
    coords = np.indices(grid).reshape(3, -1).T
    kf = coords/np.array(grid)
    facts = SimpleNamespace(trs_holds=False, ntran=1, kgrid=np.array(grid),
        nkpts=nk, kpoints=kf, sym_matrices=np.eye(3, dtype=np.int64)[None], shift=np.zeros(3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sym = SymMaps(facts)
    centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.int32)
    plan = build_centroid_k_unfold_plan(sym, centroids, (4, 4, 4), mesh,
                                       nspinor=ns, parent_k_frac=kf)
    perm = np.asarray(plan.layout.axis.pack_host(np.arange(mu)[None], axis=1))[0]
    psi = .2*(rng.normal(size=(nk, nb, ns, mu))+1j*rng.normal(size=(nk, nb, ns, mu)))
    psi = psi[..., perm]
    energy = rng.uniform(-.8, .9, (nk, nb))
    occ = rng.uniform(.1, .9, (nk, nb))
    rotation, _ = np.linalg.qr(rng.normal(size=(nk, nb, nb))+1j*rng.normal(size=(nk, nb, nb)))
    rho = (rotation*np.array([0., .2, .8, 1.])[None, None])@rotation.conj().swapaxes(-1, -2)
    put = lambda a, spec: device_put_process_local(np.ascontiguousarray(a), NamedSharding(mesh, spec))
    pn, pm = put(psi, PSI_NMU_SPEC), put(psi.transpose(0, 2, 3, 1), PSI_MUN_SPEC)
    ed, fd = put(energy, P()), put(occ, P())
    carrier = ParentGreenCarrier(psi_nmu=pn, psi_mun=pm, enk=ed, occ=fd, plan=plan, layout="face")
    wfns = Wavefunctions(enk=ed, occ=fd, slices=BandSlices.from_band_edges(
        0, 0, 2, nb, nb, b4_logical=nb), green_parent=carrier, layout="face")
    return locals()


def literal_crop(value, grid, target):
    """Explicit real-lattice triangular window, independent of production FFT."""
    import numpy as np
    lattice = np.fft.ifftn(value.reshape(*grid, *value.shape[1:]),
                           axes=(0, 1, 2), norm="ortho")
    out = np.zeros((*target, *value.shape[1:]), complex)
    for cell in np.ndindex(*grid):
        weight = 1.
        for r, n, m in zip(cell, grid, target):
            signed = r if r <= (n-1)//2 else r-n
            if n != m:
                weight *= max(1-abs(signed)/(m/2), 0.)
        slot = tuple(r % m for r, m in zip(cell, target))
        out[slot] += weight*lattice[cell]
    return np.fft.fftn(out, axes=(0, 1, 2), norm="ortho").reshape(
        int(np.prod(target)), *value.shape[1:])*np.sqrt(np.prod(target)/np.prod(grid))


def response_stage(c):
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream, _get_chi_fractional_contour_kernel_face
    from gw.response_bank import full_density_bare_moments

    mesh, grid, coarse = c["mesh"], c["grid"], c["coarse"]
    nk, nb, mu, ns = (c[k] for k in ("nk", "nb", "mu", "ns"))
    pm, pn, energy, psi, rho = (c[k] for k in ("pm", "pn", "energy", "psi", "rho"))
    ph = density_left_endpoint(mesh, pm.shape)(pm, c["put"](rho, P()))
    times = np.array([.13+.1j, -.07+.2j])
    operands = (times, np.eye(2, dtype=complex), pm, pn, ph, c["ed"])
    def stream(**kw):
        return full_density_response_stream(mesh, grid, (nk, nb, mu, ns), n_outputs=2, **kw)
    default = np.asarray(gather_to_host(stream(q_ids=range(nk))(*operands)))
    explicit = np.asarray(gather_to_host(stream(q_ids=range(nk), response_window="sharp")(*operands)))
    np.testing.assert_array_equal(default.view(np.uint64), explicit.view(np.uint64))
    smoothed = np.asarray(gather_to_host(stream(q_ids=range(16), qgrid=coarse,
                                                 response_window="fejer")(*operands)))
    # Literal band/spin transition sums with rho LEFT of right-index phases.
    weighted = np.einsum("kasr,kab->kbsr", psi, rho)
    direct = np.zeros_like(default)
    moments = [np.zeros((nk, mu, mu), complex) for _ in range(2)]
    for iq, q in enumerate(c["coords"]):
        partner = np.ravel_multi_index(((c["coords"]-q) % grid).T, grid)
        vertex = np.einsum("kbsr,kdsr->kbdr", psi, psi[partner].conj())
        amplitude = (np.einsum("kbsr,kdsr->kbdr", psi, weighted[partner].conj())
                     -np.einsum("kbsr,kdsr->kbdr", weighted, psi[partner].conj()))
        gap = energy[:, :, None]-energy[partner, None, :]
        for it, t in enumerate(times):
            direct[iq, it] = np.einsum("kbdm,kbdn,kbd->mn", amplitude, vertex.conj(),
                                        np.exp(-gap*t))/np.sqrt(nk)
        moments[0][iq] = np.einsum("kbdm,kbdn->mn", amplitude, vertex.conj())/np.sqrt(nk)
        moments[1][iq] = np.einsum("kbdm,kbdn,kbd->mn", amplitude, vertex.conj(), gap)/np.sqrt(nk)
    np.testing.assert_allclose(default, direct, rtol=3e-11, atol=3e-13)
    expected = literal_crop(direct, grid, coarse)
    np.testing.assert_allclose(smoothed, expected, rtol=3e-11, atol=3e-13)
    meta = SimpleNamespace(nkx=8, nky=8, nkz=1, qgrid=coarse, qgrid_response_window="fejer")
    actual_m = full_density_bare_moments((pm, pn, ph, c["ed"]), meta,
                                         mesh_xy=mesh, q_ids=range(16))
    errors = []
    for actual, full in zip(actual_m, moments):
        actual = np.asarray(gather_to_host(actual)); expected_m = literal_crop(full, grid, coarse)
        np.testing.assert_allclose(actual, expected_m, rtol=3e-11, atol=3e-13)
        errors.append(float(np.max(abs(actual-expected_m))))
    # Standard scalar-SPM contour seam, independently of the coherent builder.
    scalar_args = (np.array([.13, .2]), np.eye(2, dtype=complex), pm, pn,
                   c["ed"], c["fd"], 1-c["fd"], np.array(0.))
    scalar_full = _get_chi_fractional_contour_kernel_face(mesh, grid, 2,
        (nk, nb, mu, ns), ordered=True, selected_q=tuple(range(nk)))
    scalar_coarse = _get_chi_fractional_contour_kernel_face(mesh, grid, 2,
        (nk, nb, mu, ns), ordered=True, selected_q=tuple(range(16)),
        qgrid=coarse, response_window="fejer")
    sf = np.asarray(gather_to_host(scalar_full(*scalar_args)))
    sc = np.asarray(gather_to_host(scalar_coarse(*scalar_args)))
    scalar_expected = literal_crop(sf, grid, coarse)
    np.testing.assert_allclose(sc, scalar_expected, rtol=3e-11, atol=3e-13)
    return dict(default_sharp_bitwise=True, raw_prefactor_electronic_Nk=nk,
        max_full_literal_error=float(np.max(abs(default-direct))),
        max_fejer_literal_error=float(np.max(abs(smoothed-expected))), moment_errors=errors,
        max_scalar_crop_error=float(np.max(abs(sc-scalar_expected))), density_noncommuting=True)


def sigma_stage(c):
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from gw.ppm_tau_kernel import _get_sigma_kij_kernel, sigma_subtile_operands, sigma_kconv_tables
    from gw.wavefunction_bundle import parent_sigma_operands, sigma_face_kernel_kwargs
    from symmetry_maps import QirrOperator

    # An explicitly declared arbitrary complex W(tau) node, not a fitted bank.
    rng, mesh = c["rng"], c["mesh"]
    w = .15*(rng.normal(size=(16, 4, 4))+1j*rng.normal(size=(16, 4, 4)))
    wd = c["put"](w, P(None, "x", "y")); wedge = QirrOperator.whole_zone(wd)
    args = parent_sigma_operands(c["wfns"])
    spatial = sigma_subtile_operands(*args[:4], mesh_xy=mesh)
    kernel = _get_sigma_kij_kernel(mesh_xy=mesh, kgrid=c["grid"], qgrid=c["coarse"],
        q_wedge=wedge, **sigma_face_kernel_kwargs(c["wfns"]))
    t, ref = .21-.04j, .13
    actual = np.asarray(gather_to_host(kernel(*spatial, args[4], args[5], ref, t,
        wd, None, wedge.with_load(mesh).load, sigma_kconv_tables(mesh, c["plan"]))))
    expected = np.zeros_like(actual)
    for iq, q in enumerate(np.indices(c["coarse"]).reshape(3, -1).T*2):
        partner = np.ravel_multi_index(((c["coords"]-q) % c["grid"]).T, c["grid"])
        vertex = np.einsum("kisr,kbsr->kibr", c["psi"].conj(), c["psi"][partner])
        weight = c["occ"][partner]*np.exp(-1j*(c["energy"][partner]-ref)*t)
        expected -= np.einsum("kibr,rs,kjbs,kb->kij", vertex, w[iq], vertex.conj(), weight)/16
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-12)
    return dict(max_complex_matrix_error=float(np.max(abs(actual-expected))), q_weight=1/16,
        full_matrix_row_and_column_entries=True, scope="single_tau_node_not_full_retarded_Sigma")


def factory_stage(c):
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from ffi import ffi_dial_key
    from common.collectives import gather_to_host
    from gw.shared_pole_screening import _general_frequency_programs
    factory = _general_frequency_programs(c["mesh"], P(None, "x", "y"), ffi_dial_key())
    assert factory is _general_frequency_programs(c["mesh"], P(None, "x", "y"), ffi_dial_key())
    def random(shape):
        return c["rng"].normal(size=shape)+1j*c["rng"].normal(size=shape)
    host = lambda a: np.asarray(gather_to_host(a))
    put = lambda a: c["put"](a, P(None, "x", "y"))
    maximum = 0.
    for current in range(2):
        mu, lam = np.array([1+2j, 2+3j])+current, np.array([3+4j, 4+5j])-current
        ql, il, qr, wr = random((2, 4, 2)), random((2, 2, 4)), random((2, 4, 2)), random((2, 4, 2))
        args = (c["put"](mu, P()), c["put"](lam, P()), *(put(a) for a in (ql, il, qr, wr)))
        got = factory[0](*args)
        a, b = il@qr, ql.conj().swapaxes(-1, -2)@wr
        E, A = -(a-b)/(mu[:, None]-lam[None]), -(mu[:, None]*a-lam[None]*b)/(mu[:, None]-lam[None])
        for actual, exact in zip(got, (E, A, il, wr)):
            np.testing.assert_allclose(host(actual), exact, rtol=3e-11, atol=3e-12)
        qi, c1, c2 = random((2, 4, 2)), random((2, 4, 4)), random((2, 4, 4))
        augmented = factory[1](*args[:2], *got, *args[2:], put(qi), put(c1), put(c2))
        # p=2 tile joins interleave the finite/infinity coordinate blocks.
        qh = qi.conj().swapaxes(-1, -2)
        ei, ef, ci, bi = qh@wr, il@qi, c1@qi, qh@c1
        plain = (np.concatenate((np.concatenate((E, ef), -1), np.concatenate((ei, qh@ci), -1)), -2),
            np.concatenate((np.concatenate((A, mu[None, :, None]*ef-ql.conj().swapaxes(-1, -2)@ci), -1),
                            np.concatenate((ei*lam[None, None]-bi@qr, qh@(c2@qi)), -1)), -2),
            np.concatenate((il, bi), -2), np.concatenate((wr, ci), -1))
        order = np.array([0, 2, 1, 3])
        expected = (plain[0][:, order][:, :, order], plain[1][:, order][:, :, order],
                    plain[2][:, order], plain[3][:, :, order])
        for actual, exact in zip(augmented, expected):
            actual = host(actual)
            np.testing.assert_allclose(actual, exact, rtol=3e-11, atol=3e-12)
            maximum = max(maximum, float(np.max(abs(actual-exact))))
    return dict(current_input_sets=2, same_cached_program_objects=True,
                max_pencil_infinity_error=maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=120.)
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 120:
        raise ValueError("wall cap must be <=120 seconds")
    if os.environ.get("LEG_HEAD") != REVISION:
        raise ValueError("exact launcher revision mismatch")
    watchdog = threading.Timer(args.max_wall_seconds, lambda: os._exit(124))
    watchdog.daemon = True; watchdog.start()
    import gw.gw_jax as driver  # Collective native bootstrap BEFORE NumPy/JAX arrays.
    import runtime
    import jax
    import numpy as np
    from common.collectives import rank0_transaction
    from ffi.fft import kconv_backend
    root = Path(args.source_root).resolve()
    import gw.w_isdf, gw.ppm_tau_kernel, gw.shared_pole_screening
    modules = (driver, gw.w_isdf, gw.ppm_tau_kernel, gw.shared_pole_screening)
    origins = {m.__name__: str(Path(m.__file__).resolve()) for m in modules}
    if any(root not in Path(p).parents for p in origins.values()):
        raise ValueError("source module origin mismatch")
    mesh = driver.RUNTIME.mesh
    if (jax.process_count() != 4 or jax.local_device_count() != 1 or mesh.devices.shape != (2, 2)
            or any(d.platform != "gpu" for d in jax.devices()) or not jax.config.x64_enabled
            or kconv_backend(mesh, (8, 8, 1)) != "mathdx"):
        raise ValueError("requires native CUDA P4 2x2 FP64; no reference FFT route")
    out = Path(args.out).resolve()
    rank0_transaction(out, stage="fejer/new_output", write=lambda: out.mkdir(parents=True, exist_ok=False))
    report = dict(status="RUNNING_SOFTWARE_ONLY", revision=REVISION, module_origins=origins,
        archive_sha256=os.environ.get("LEG_SOURCE_ARCHIVE_SHA256"),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        material_admitted=False, SC_performed=False, TD_propagated_fs=0,
        fixture=dict(kgrid=[8, 8, 1], qgrid=[4, 4, 1], bands=4, spinor=2, mu=4),
        stages={}, max_wall_seconds=args.max_wall_seconds, max_active_GPUh=4*args.max_wall_seconds/3600)
    def save():
        def write():
            tmp = out/"receipt.tmp.json"
            tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            os.replace(tmp, out/"receipt.json")
        rank0_transaction(out, stage="fejer/receipt", write=write)
    started = time.monotonic(); rc = 0
    try:
        c = fixture(mesh)
        for name, fn in (("native_chi_moments", response_stage), ("native_coarse_Sigma_tau", sigma_stage),
                         ("native_cached_pencil_infinity", factory_stage)):
            report["phase"] = name; save()
            if jax.process_index() == 0: print("BEGIN", name, flush=True)
            before = time.monotonic(); result = fn(c)
            report["stages"][name] = dict(seconds=time.monotonic()-before, result=result)
            save()
        report["status"] = "PASS_SOFTWARE_ONLY_NOT_MATERIAL_OR_CONSERVING_CLOSURE"
    except BaseException as error:
        # A rank-local native failure must not enter a fresh I/O collective
        # while its peers may still execute a different device program.
        print(f"Fejer P4 rank {jax.process_index()} FAILED: {error!r}\n"+
              traceback.format_exc(), flush=True)
        raise
    report["elapsed_seconds"] = time.monotonic()-started
    save(); watchdog.cancel()
    runtime.finalize_process(rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
