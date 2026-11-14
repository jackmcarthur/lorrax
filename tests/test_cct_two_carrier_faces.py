"""The parent Gram with faces from two carriers, against independent NumPy normal equations.

The convective current vertex (docs: breit_salvage plan, one four-spinor per
Cartesian channel) fits terms whose two endpoints carry different four-spinor
carriers, Ψ^(i) at μ and Ψ^(j) at ν.  :func:`isdf.core.c_q_from_psi_sm` takes
the two faces as separate arguments; this script checks that carrier A as
``psi_mun_parent`` and carrier B as ``psi_nmu_parent`` give the normal
equations of tests/test_cct_parent_stream.py built from the two-carrier
projectors D_w(μ,ν) = Σ_n w_n Ψ^A_n(μ) Ψ^B_n(ν)† (A at μ, B at ν), with and without the LR + conj(−q) completion, for scalar and current
vertex pairs, on the unreduced (trivial) view and on a reduced view (there
the check is mechanical: a fixed-axis carrier is not covariant).

No native library is needed: four host devices, the XLA k-convolution
(``ffi.fft.xla_reference``) and the XLA band GEMM::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 LORRAX_BANDS_GEMM_FFI=0 \\
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
    python3 tests/test_cct_two_carrier_faces.py

Under pytest nothing is collected.
"""
import numpy as np


def two_carrier_reference(psi_a, psi_b, wl, wr, plan, kgrid, gl=0, gr=0):
    """NumPy normal equations with carrier ``psi_a`` at μ and ``psi_b`` at ν.

    ``psi_a, psi_b (n_parent, ns, mu, nb)``; returns ``(nk, mu, mu)``.  The
    typed unfold and the vertex placement are those of
    ``tests/test_cct_parent_stream.numpy_reference``; only the projector
    differs, ``D(μ,ν) = Σ_n w_n A_n(μ) conj(B_n(ν))``.
    """
    from common.gamma_matrices import gamma_perm_phase_host
    nk, ns, mu = (plan.n_full, plan.nspinor, plan.n_centroid_packed)

    def projector(w):
        return np.einsum('pamn,n,pbvn->pambv', psi_a, w, psi_b.conj())

    def unfold(d):
        out = np.empty((nk, ns, mu, mu, ns), complex)
        for k in range(nk):
            p, op = (int(plan.irr_idx[k]), int(plan.sym_idx[k]))
            idx = plan.sym_perm[op]
            x = np.take(np.take(d[p], idx, axis=1), idx, axis=3)
            phase = np.exp(2j * np.pi * (plan.L_table[op] @ plan.k_parent_frac[p]))
            x = x * phase[None, :, None, None] * phase.conj()[None, None, None, :]
            if op >= plan.n_sym_spatial:
                x = x.conj()
            u = plan.spin_action_full[k]
            out[k] = np.einsum('ac,cmen,be->amnb', u, x, u.conj()).conj()
        return out
    left, right = unfold(projector(wl)), unfold(projector(wr))
    if ns == 4:
        (pl, fl), (pr, fr) = map(gamma_perm_phase_host, (gl, gr))
        right = (np.take(np.take(right, pl, axis=1), pr, axis=4)
                 * fl[None, :, None, None, None] * fr[None, None, None, None, :])
    shape = tuple(kgrid)
    z = (np.fft.ifftn(left.reshape(shape + left.shape[1:]), axes=(0, 1, 2), norm='forward').conj()
         * np.fft.ifftn(right.reshape(shape + right.shape[1:]), axes=(0, 1, 2), norm='forward')).sum(axis=(3, 6))
    return np.fft.fftn(z, axes=(0, 1, 2), norm='forward').reshape(nk, mu, mu)


def _plan(view, mesh, layout, rng, kgrid, ns):
    from gw.centroid_k_unfold import CentroidKUnfoldPlan
    nk, mu = int(np.prod(kgrid)), layout.axis.n_padded
    if view == 'trivial':
        irr, sym = np.arange(nk, dtype=np.int32), np.zeros(nk, np.int32)
        canon = np.arange(19)[None, :]
        wraps = np.zeros((1, mu, 3))
        u = np.tile(np.eye(ns, dtype=complex), (nk, 1, 1))
        kp = np.stack(np.meshgrid(*[np.arange(n) / n for n in kgrid], indexing='ij'), -1).reshape(-1, 3)
        n_spatial = 1
    else:
        irr = np.array([0, 1, 2, 0, 1, 2], np.int32)
        sym = np.array([0, 0, 0, 1, 2, 3], np.int32)
        canon = np.tile(np.arange(19), (4, 1))
        for i in range(0, 18, 2):
            canon[1:, i:i + 2] = [i + 1, i]
        wraps = rng.integers(-1, 2, (4, mu, 3)).astype(np.float64)
        wraps[0] = 0
        wraps[:, ~layout.axis.active_mask] = 0
        u = []
        for k in range(nk):
            z = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
            z = np.block([[z, np.zeros((2, 2))], [np.zeros((2, 2)), (-1 if k % 2 else 1) * z]])
            u.append(np.eye(ns, dtype=complex) if k < 3 else z)
        u, kp, n_spatial = np.asarray(u), rng.random((3, 3)), 2
    return CentroidKUnfoldPlan(mesh, layout, irr, sym, layout.axis.pack_permutations_host(canon),
                               wraps, kp, u, n_spatial, ns)


def main():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from ffi.fft import xla_reference
    from common.collectives import device_put_process_local, gather_to_host
    from common.grouped_layout import build_square_grouped_shard_layout
    from isdf.core import c_q_from_psi_sm
    from distrib_la import gemm_plan
    from symmetry_maps import q_negation_index
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    rng = np.random.default_rng(5)
    kgrid, nb, ns = (3, 2, 1), 8, 4
    neg = np.asarray(q_negation_index(kgrid), np.int32)
    layout = build_square_grouped_shard_layout(np.arange(19) // 2, (2, 2))
    mu = layout.axis.n_padded
    put = lambda a, spec: device_put_process_local(a, NamedSharding(mesh, spec))
    wl = np.array([1, 1, 1, 1, 0, 0, 0, 0.0])
    wr = np.array([0, 0, 1, 1, 1, 1, 1, 1.0])
    worst, gap = 0.0, np.inf
    for view in ('trivial', 'reduced'):
        plan = _plan(view, mesh, layout, rng, kgrid, ns)
        faces = []
        for _ in range(2):
            psi = (rng.normal(size=(plan.n_parent, ns, mu, nb))
                   + 1j * rng.normal(size=(plan.n_parent, ns, mu, nb)))
            psi[:, :, ~layout.axis.active_mask, :] = 0
            faces.append(psi)
        psi_a, psi_b = faces
        with xla_reference():
            gemm = gemm_plan(mesh, m=ns * mu, n=ns * mu, k=nb, nq=plan.n_parent,
                             dtype=np.complex128, layout='face', warmup=False)
            for gl, gr in ((0, 0), (1, 1), (1, 2), (2, 3)):
                ref = two_carrier_reference(psi_a, psi_b, wl, wr, plan, kgrid, gl, gr)
                one = two_carrier_reference(psi_a, psi_a, wl, wr, plan, kgrid, gl, gr)
                scale = float(np.abs(ref).max())
                for q_neg, want in ((None, ref), (neg, ref + np.conj(ref[neg]))):
                    got = np.asarray(gather_to_host(c_q_from_psi_sm(
                        put(psi_a, P(None, None, 'x', 'y')),
                        put(psi_b.transpose(0, 3, 1, 2), P(None, 'x', None, 'y')),
                        wl, wr, k_unfold_plan=plan, kgrid=kgrid, mesh_xy=mesh, gemm=gemm,
                        gamma_L=gl, gamma_R=gr, q_neg_idx=q_neg)))
                    err = float(np.abs(got - want).max()) / scale
                    worst = max(worst, err)
                    print(f"{view:8s} vertices ({gl},{gr}) completion={q_neg is not None}: "
                          f"relative error {err:.2e}", flush=True)
                gap = min(gap, float(np.abs(ref - one).max()) / scale)
    assert worst < 1e-12, worst
    assert gap > 1e-2, f"two-carrier and one-carrier references agree ({gap}): the check is blind"
    print(f"CCT_TWO_CARRIER PASS: worst relative error {worst:.1e}; "
          f"two- vs one-carrier references differ by >= {gap:.2f}", flush=True)


if __name__ == '__main__':
    main()
