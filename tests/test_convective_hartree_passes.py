"""The transverse direct field of the convective vertex: three carrier passes, P4 script.

``gw.hartree.direct_field_matrices`` on the hsuite H2 fixture (or any nspinor = 2 WFN with its UPF
directory), with the convective field attached to the loader at scale s (the attached callable times s):

1. s = 0: the charge matrix <m|V_H|n> and H_T = <m|alpha.A|n> equal the run WITHOUT a field (the one-pass
   kinetic path) to round-off: the three passes (J_a from carrier a, A_a on carrier a) add up to the one.
2. H_T(s) is quadratic in s (pair currents are linear in the field, H_T bilinear in them): the third
   difference over s = 0, 1, 2, 3 vanishes to round-off, and the change H_T(1) - H_T(0) is resolved.
3. V_H does not move with s (every carrier has the same large block).

GPU, four ranks::

    lx run -N 1 -G 4 -n 4 -- python3 tests/test_convective_hartree_passes.py \\
        --wfn tests/hsuite/fixture/WFN.h5 --pseudo-dir tests/hsuite/fixture

Prints ``CONVECTIVE_HARTREE PASS`` with the figures.  Under pytest nothing is collected.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--nb", type=int, default=6)
    o = ap.parse_args()
    import runtime
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from types import SimpleNamespace
    from common.collectives import gather_to_host
    from gw.hartree import direct_field_matrices
    from psp.pseudos import load_pseudopotentials, pseudopotential_md5_stamp
    from psp.vnl_ops import build_vnl_setup_from_pseudo_dir, convective_field
    from wfn_loader import WfnLoader
    host = lambda x: np.asarray(gather_to_host(x))

    def fields(wfn):
        out = direct_field_matrices(
            wfn, wfn.symmetry(), SimpleNamespace(nspinor=4, npol=4, fft_grid=wfn.fft_grid),
            truncation_2d=False, nb=o.nb, mesh=mesh, include_transverse=True,
            return_sharded=True)
        return host(out.charge), host(out.transverse)

    with WfnLoader(o.wfn, mesh=mesh) as wfn:
        v0, t0 = fields(wfn)                                    # no field: the kinetic one-pass path
    stamp = pseudopotential_md5_stamp(load_pseudopotentials(o.pseudo_dir))
    V, T = {}, {}
    for s in (0.0, 1.0, 2.0, 3.0):
        with WfnLoader(o.wfn, mesh=mesh) as wfn:
            setup, identity = build_vnl_setup_from_pseudo_dir(wfn, o.pseudo_dir, kin_ion_md5=stamp)
            base = convective_field(setup)
            wfn.attach_current_vertex_field(lambda *a, b=base, s=s: b(*a) * s, identity=identity)
            V[s], T[s] = fields(wfn)
    tscale = float(np.abs(t0).max())
    vscale = float(np.abs(v0).max())
    e_t0 = float(np.abs(T[0.0] - t0).max())
    e_v0 = max(float(np.abs(V[s] - v0).max()) for s in V)
    d01 = float(np.abs(T[1.0] - T[0.0]).max())
    third = float(np.abs(T[3.0] - 3 * T[2.0] + 3 * T[1.0] - T[0.0]).max())
    ok = (e_t0 <= 1e-12 * tscale and e_v0 <= 1e-12 * vscale and d01 > 1e3 * max(e_t0, 1e-300)
          and third <= 1e-6 * d01)
    if jax.process_index() == 0:
        print(f"CONVECTIVE_HARTREE {'PASS' if ok else 'FAIL'}: |H_T(s=0) - H_T(no field)| = {e_t0:.3e} "
              f"of max|H_T| {tscale:.3e} Ry; max_s |V_H(s) - V_H| = {e_v0:.3e} of {vscale:.3e} Ry; "
              f"|H_T(1) - H_T(0)| = {d01:.3e} Ry; third difference / change = {third / max(d01, 1e-300):.2e}",
              flush=True)
    runtime.finalize_process(0 if ok else 1)


if __name__ == "__main__":
    main()
