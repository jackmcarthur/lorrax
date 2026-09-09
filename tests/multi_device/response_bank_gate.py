"""P4/P16 planted bank equations and bounded-stream parity.

Run as a script under lx, one rank per GPU; argv[1] is an evidence directory.
All dense NumPy oracles are tiny and execute on the compute node only.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from runtime import initialize_communicator_stack, finalize_process

initialize_communicator_stack()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import resolve_mesh, gather_to_host
from gw.response_bank import response_algebra
from gw.w_isdf import _get_chi_fractional_contour_kernel_face
from gw.wavefunction_bundle import PSI_MUN_SPEC, PSI_NMU_SPEC


def main():
    mesh = resolve_mesh()
    assert mesh.shape["x"] > 1 and mesh.shape["y"] > 1
    root = Path(sys.argv[1])
    face = NamedSharding(mesh, P(None, "x", "y"))
    rng = np.random.default_rng(731)
    n, b, k = 8, 3, 5
    meta = SimpleNamespace(nk_tot=64, nspin=1, nspinor_wfnfile=1)

    def put(a, sharding=face):
        return jax.make_array_from_callback(a.shape, sharding, lambda ix: a[ix])

    def error(a, expected):
        expected = put(np.asarray(expected))
        return float(gather_to_host(jnp.linalg.norm(a - expected)
                                   / jnp.linalg.norm(expected)))

    h = np.eye(n)[None] + 0.05 * rng.normal(size=(b, n, n))
    h = (h + h.swapaxes(-1, -2)) / 2
    h = h.astype(np.complex128)
    v = h @ h
    c = (rng.normal(size=(b, n, k))
         + 1j * rng.normal(size=(b, n, k))) * 0.08
    adj = c.conj().swapaxes(-1, -2)
    poles = np.arange(1., k + 1.)
    z = 0.7 + 0.4j
    s = z * z
    a = (c / (s - poles)) @ adj
    ad = (-c / (s - poles) ** 2) @ adj
    a0 = c @ adj
    a1 = (c * poles) @ adj
    # Independent transition-space RPA resolvent (Woodbury identity).
    t = np.diag(poles)[None] + adj @ v @ c
    r = np.linalg.inv(s * np.eye(k)[None] - t)
    expected = v @ c @ r @ adj @ v
    expected_d = -v @ c @ r @ r @ adj @ v
    m1 = v @ c @ adj @ v / 2
    m3 = v @ c @ t @ adj @ v / 2
    results = {}
    for layout in ("local", "distributed"):
        samples, moments, receipt = response_algebra(
            meta, {"linalg": layout}, mesh_xy=mesh, n=n)
        pref = receipt["prefactor"]
        operands = (put(h), put(a / pref), put(ad / pref))
        executable = samples.lower(*operands).compile()
        w, wd = executable(*operands)
        got_m1, got_m3 = moments(put(h), put(a0), put(a1))
        row = dict(value=error(w, expected), derivative=error(wd, expected_d),
                   M1=error(got_m1, m1), M3=error(got_m3, m3))
        assert max(row.values()) < 1e-11, (layout, row)
        wrong, _ = samples(put(h), put(-a / pref), put(ad / pref))
        row["red_sign"] = error(wrong, expected)
        row["red_factor_two"] = error(got_m1 * 2, m1)
        wrong, _ = samples(put(h), put(a), put(ad))
        row["red_double_prefactor"] = error(wrong, expected)
        assert min(row[key] for key in row if key.startswith("red_")) > 0.1
        assert w.sharding.spec == P(None, "x", "y")
        row["memory"] = str(executable.memory_analysis())
        row["plan"] = receipt
        results[layout] = row
        if jax.process_index() == 0:
            (root / f"dyson_{layout}.hlo").write_text(executable.as_text())

    # Same fractional one-particle stream, selected noncontiguous parents.
    # Complex moment weights exercise both Keldysh orientations at t=0.
    nk, nb = 8, 8
    psi = (rng.normal(size=(nk, 1, n, nb))
           + 1j * rng.normal(size=(nk, 1, n, nb))) / 8
    right = psi.conj().transpose(0, 3, 1, 2)
    energy = np.broadcast_to(np.linspace(-1, 2, nb), (nk, nb)).copy()
    f = 1 / (1 + np.exp(energy / 0.3))
    rep = NamedSharding(mesh, P())
    args = (put(np.array([0., .3, 1.]), rep),
            put(np.array([[1., .4j, .2], [.2j, 1., .7j]]), rep),
            put(psi, NamedSharding(mesh, PSI_MUN_SPEC)),
            put(right, NamedSharding(mesh, PSI_NMU_SPEC)),
            put(energy, rep), put(f, rep), put(1-f, rep),
            put(np.array(0.), rep))
    full = _get_chi_fractional_contour_kernel_face(mesh, (2, 2, 2), 2,
                                                  (nk, nb, n, 1))
    selected = _get_chi_fractional_contour_kernel_face(
        mesh, (2, 2, 2), 2, (nk, nb, n, 1), selected_q=(0, 3, 7))
    reference = jnp.stack(full(*args), axis=1)[jnp.array([0, 3, 7])]
    executable = selected.lower(*args).compile()
    actual = executable(*args)
    relative = float(gather_to_host(jnp.linalg.norm(actual-reference)
                                   / jnp.linalg.norm(reference)))
    assert relative < 1e-11, relative
    assert actual.sharding.spec == P(None, None, "x", "y")
    results["selected_stream"] = dict(relative=relative,
        shape=list(actual.shape), memory=str(executable.memory_analysis()))
    if jax.process_index() == 0:
        (root / "selected_stream.hlo").write_text(executable.as_text())
        (root / "receipt.json").write_text(json.dumps(dict(
            status="PASS", checks=results,
            source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            job=os.getenv("SLURM_JOB_ID"), step=os.getenv("SLURM_STEP_ID"),
            processes=jax.process_count()), indent=2))
    print("IBANK planted algebra and selected stream PASS", flush=True)


if __name__ == "__main__":
    main()
    finalize_process()
