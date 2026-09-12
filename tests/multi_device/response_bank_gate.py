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
from gw.response_bank import response_algebra, exact_bare_moments, _coulomb_algebra
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
        root_v = _coulomb_algebra(mesh, n, n, layout)
        got_h, got_hi, negative, ranks = root_v(put(v))
        row["coulomb_root"] = error(got_h, h)
        row["coulomb_inverse"] = error(got_hi, np.linalg.inv(h))
        assert not bool(negative) and np.all(np.asarray(ranks) == n)
        singular_h = h.copy()
        singular_h[:, -2:, :] = 0
        singular_h[:, :, -2:] = 0
        got_h, got_hi, negative, ranks = root_v(put(singular_h @ singular_h))
        row["coulomb_supported_root"] = error(got_h, singular_h)
        row["coulomb_supported_inverse"] = error(got_hi, np.linalg.pinv(singular_h))
        assert not bool(negative) and np.all(np.asarray(ranks) == n - 2)
        bad_v = singular_h @ singular_h
        bad_v[:, -1, -1] = -0.1
        _, _, negative, _ = root_v(put(bad_v))
        assert bool(negative), "negative Coulomb eigenvalue was accepted"
        assert max(row[key] for key in row if key.startswith("coulomb_")) < 1e-11
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

    carry_kernel = _get_chi_fractional_contour_kernel_face(
        mesh, (2,2,2), 2, (nk,nb,n,1), selected_q=(0,3,7), bank_carry=True)
    carry_shard = NamedSharding(mesh,P(None,None,"x","y"))
    carry = put(np.zeros((2,3,n,n),complex),carry_shard)
    carry_executable = carry_kernel.lower(*args,carry).compile()
    carried = carry_executable(*args,carry)
    twice = carry_executable(*args,carried)
    carry_error = float(gather_to_host(jnp.linalg.norm(jnp.swapaxes(twice,0,1)-2*reference)
                                      /jnp.linalg.norm(2*reference)))
    assert carry_error < 1e-11, carry_error
    assert carry_executable.memory_analysis().alias_size_in_bytes > 0
    results["donated_carry"] = dict(relative=carry_error,
        memory=str(carry_executable.memory_analysis()))

    # Remote Laplace cell, independently summed over lower/upper band pairs.
    real_psi = psi.real.astype(np.complex128)
    lower = np.arange(nb) < 3
    upper = ~lower
    times = np.array([0., .3, 1.])
    laplace = _get_chi_fractional_contour_kernel_face(
        mesh, (2, 2, 2), 1, (nk, nb, n, 1), selected_q=(0,),
        pair_mode="laplace")
    lap_args = (put(times, rep), put(np.ones((1, 3), complex), rep),
        put(real_psi, NamedSharding(mesh, PSI_MUN_SPEC)),
        put(real_psi.transpose(0, 3, 1, 2), NamedSharding(mesh, PSI_NMU_SPEC)),
        put(energy, rep), put(np.stack([f*lower, (1-f)*lower]), rep),
        put(np.stack([(1-f)*upper, f*upper]), rep), put(np.zeros(2), rep))
    lap_expected = np.zeros((1, 1, n, n), complex)
    for ik in range(nk):
        for i in range(3):
            for j in range(3, nb):
                pair = real_psi[ik, 0, :, i] * real_psi[ik, 0, :, j]
                lap_expected[0, 0] += ((f[ik, i]-f[ik, j])
                    * np.exp(-(energy[ik, j]-energy[ik, i])*times).sum()
                    * np.outer(pair, pair.conj()) * 2 / np.sqrt(nk))
    lap_error = error(laplace(*lap_args)[:, 0], lap_expected[:, 0])
    assert lap_error < 1e-11, lap_error
    results["laplace_cell"] = dict(relative=lap_error)

    # Independent explicit band-pair oracle only on this tiny planted test.
    # Real orbitals make the TRS-even response exact without a star fixture.
    real_psi = psi.real.astype(np.complex128)
    wfns = SimpleNamespace(layout="face", green_parent=None,
        psi_mun=put(real_psi, NamedSharding(mesh, PSI_MUN_SPEC)),
        psi_nmu=put(real_psi.transpose(0, 3, 1, 2),
                    NamedSharding(mesh, PSI_NMU_SPEC)),
        enk=put(energy, rep), occ=put(f, rep),
        slices=SimpleNamespace(nb_full=nb, b0=0, b4_logical=5))
    moment_meta = SimpleNamespace(nkx=2, nky=2, nkz=2, nk_tot=nk,
        nspin=1, nspinor=1, nspinor_wfnfile=1, b_id_4_chi_user=5,
        mu_basis=SimpleNamespace(n_packed=n))
    execute = lambda kernel, args, stage: kernel(*args)
    a0, a1, census = exact_bare_moments(wfns, moment_meta,
        mesh_xy=mesh, q_ids=(0,), execute=execute)
    truth = []
    for power in (1, 3):
        value = np.zeros((1, n, n), dtype=np.complex128)
        for ik in range(nk):
            for i in range(5):
                for j in range(5):
                    pair = real_psi[ik, 0, :, i] * real_psi[ik, 0, :, j]
                    value[0] += ((f[ik, i]-f[ik, j])
                        * (energy[ik, j]-energy[ik, i])**power
                        * np.outer(pair, pair.conj()))
        truth.append(value * 2 / nk)
    moment_errors = dict(A0=error(a0, truth[0]), A1=error(a1, truth[1]))
    assert max(moment_errors.values()) < 1e-11, moment_errors
    wfns.occ = put((f > .5).astype(float), rep)
    wrong0, _, _ = exact_bare_moments(wfns, moment_meta,
        mesh_xy=mesh, q_ids=(0,), execute=execute)
    moment_errors["red_occupation"] = error(wrong0, truth[0])
    assert moment_errors["red_occupation"] > 1e-3, moment_errors
    results["six_correlations"] = dict(**moment_errors, census=census)
    # MP1 permits slight occupation overshoot. Preserve the supplied state
    # exactly; clamping to FD bounds would change its response moments.
    overshoot = f.copy()
    overshoot[:, 0], overshoot[:, 4] = 1.02, -0.02
    wfns.occ = put(overshoot, rep)
    mp0, _, _ = exact_bare_moments(wfns, moment_meta,
        mesh_xy=mesh, q_ids=(0,), execute=execute)
    mp_truth = np.zeros((1, n, n), complex)
    for ik in range(nk):
        for i in range(5):
            for j in range(5):
                pair = real_psi[ik, 0, :, i] * real_psi[ik, 0, :, j]
                mp_truth[0] += ((overshoot[ik, i]-overshoot[ik, j])
                    * (energy[ik, j]-energy[ik, i])
                    * np.outer(pair, pair.conj()) * 2 / nk)
    mp_error = error(mp0, mp_truth)
    assert mp_error < 1e-11, mp_error
    results["supplied_occupation_overshoot"] = dict(A0=mp_error)
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
