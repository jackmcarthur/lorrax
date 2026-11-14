"""The convective carriers of a real solid satisfy the L = 1 commutator identity (P4 script).

For every full-BZ k of a WFN and every Cartesian channel a, the loader's
carrier ``convective_a`` (field from the run's own UPF files, checked
against ``kin_ion.h5``) gives

    <Psi^(a)_m| alpha^a |Psi^(a)_n> = HALFALPHA <m| 2(k+G)_a + d_a V_NL |n>
                                    = HALFALPHA <m| i[H, r_a] |n>,

and the increment over the raw lift is HALFALPHA <m|d_a V_NL|n> alone
(the kinetic vertex misses exactly that).  The right side is formed
independently on each k's own sphere (``psp.vnl_ops.vnl_velocity_matrix``).

GPU, four ranks, phdf5 backend (the production read)::

    lx run -N 1 -G 4 -n 4 -- python3 tests/test_convective_carrier_solid.py \\
        --wfn WFN.h5 --pseudo-dir . --kin-ion kin_ion.h5 --bands 0 16

CPU, one process, eager backend (``--eager``; ``--kin-ion`` may be omitted,
then the files' own stamp is used)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 tests/test_convective_carrier_solid.py \\
        --eager --wfn tests/hsuite/fixture/WFN.h5 --pseudo-dir tests/hsuite/fixture

Prints ``CONVECTIVE_CARRIER_SOLID PASS`` with the worst errors.  Under
pytest nothing is collected.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--kin-ion", default=None)
    ap.add_argument("--bands", nargs=2, type=int, default=(0, 8))
    ap.add_argument("--eager", action="store_true")
    o = ap.parse_args()
    import numpy as np
    if o.eager:
        import jax
        jax.config.update("jax_enable_x64", True)
        from jax.sharding import Mesh
        mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    else:
        import runtime
        mesh = runtime.initialize_communicator_stack().mesh
        import jax
    import jax.numpy as jnp
    from common.bispinor_init import HALFALPHA
    from common.collectives import gather_to_host
    from common.gamma_matrices import gamma1, gamma2, gamma3
    from psp.pseudos import load_pseudopotentials, pseudopotential_md5_stamp
    from psp.vnl_ops import (build_vnl_kdata_from_kvec, build_vnl_setup_from_pseudo_dir,
                             convective_field, vnl_velocity_matrix)
    from wfn_loader import WfnLoader
    if o.kin_ion:
        from file_io import read_kin_ion_provenance
        stored = read_kin_ion_provenance(o.kin_ion).get("pseudopotential_md5")
    else:
        stored = pseudopotential_md5_stamp(load_pseudopotentials(o.pseudo_dir))
    host = lambda x: np.asarray(x) if o.eager else np.asarray(gather_to_host(x))
    bands = tuple(o.bands)
    worst_vertex = worst_inc = 0.0
    nl_seen = 0.0
    with WfnLoader(o.wfn, mesh=mesh, backend="eager" if o.eager else "phdf5") as wfn:
        setup, identity = build_vnl_setup_from_pseudo_dir(wfn, o.pseudo_dir, kin_ion_md5=stored)
        wfn.attach_current_vertex_field(convective_field(setup), identity=identity)
        B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=float)
        nk = int(wfn.symmetry().nk_tot)
        for k in range(nk):
            rows = [k]
            psi2 = host(wfn.load(bands=bands, k=rows))[0]
            raw = host(wfn.load(bands=bands, k=rows, bispinor=True, bispinor_lift="raw"))[0]
            g = np.asarray(wfn.gvecs(k=rows))[0]
            kv = np.asarray(wfn.kvecs(k=rows))[0]
            n = int(np.asarray(wfn.ngk_valid(k=rows))[0])
            kd = build_vnl_kdata_from_kvec(kv, g[:n], setup, compute_dZ=True)
            nb = bands[1] - bands[0]
            p = jnp.asarray(psi2[:nb, :, :n])
            v_nl = np.asarray(vnl_velocity_matrix(p, kd.Z, kd.dZ, setup.E_super))
            K = (g[:n] + kv) @ B
            for a, alpha in zip((1, 2, 3), (gamma1, gamma2, gamma3)):
                c = host(wfn.load(bands=bands, k=rows, bispinor=True,
                                  bispinor_lift=f"convective_{a}"))[0][:nb, :, :n]
                r = raw[:nb, :, :n]
                vertex = np.einsum("msg,st,ntg->mn", np.conj(c), np.asarray(alpha), c)
                p2 = np.einsum("msg,g,nsg->mn", np.conj(psi2[:nb, :, :n]), 2.0 * K[:, a - 1],
                               psi2[:nb, :, :n])
                want = HALFALPHA * (p2 + v_nl[a - 1])
                scale = HALFALPHA * 2.0 * float(np.abs(K).max())
                worst_vertex = max(worst_vertex, float(np.abs(vertex - want).max()) / scale)
                d = c[:, 2:] - r[:, 2:]
                sa = np.asarray(alpha)[:2, 2:]
                inc = (np.einsum("msg,st,ntg->mn", np.conj(r[:, :2]), sa, d)
                       + np.einsum("msg,st,ntg->mn", np.conj(d), sa.conj().T, r[:, :2]))
                gn = float(np.max(np.linalg.norm(d.reshape(nb, -1), axis=1), initial=0.0))
                if gn > 0:
                    worst_inc = max(worst_inc, float(np.abs(inc - HALFALPHA * v_nl[a - 1]).max()) / gn)
                nl_seen = max(nl_seen, HALFALPHA * float(np.abs(v_nl[a - 1]).max()) / scale)
    ok = worst_vertex < 1e-11 and worst_inc < 1e-11 and nl_seen > 1e3 * max(worst_vertex, 1e-16)
    if jax.process_index() == 0:
        print(f"CONVECTIVE_CARRIER_SOLID {'PASS' if ok else 'FAIL'}: {nk} k, bands {bands}; "
              f"vertex - HALFALPHA<2(k+G)+dV> {worst_vertex:.1e} of HALFALPHA max|2(k+G)|; "
              f"increment - HALFALPHA<dV> {worst_inc:.1e} of ||gamma||; "
              f"nonlocal part {nl_seen:.1e} of the scale", flush=True)
    if not o.eager:
        import runtime
        runtime.finalize_process(0 if ok else 1)
    elif not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
