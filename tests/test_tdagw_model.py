"""The TD-aGW interaction model against an independent dense kernel, with red twins.

``tdagw.model.build_td_agw_apply`` on a synthetic bundle (k grid 3x2x1, a
4-band window, two spinor components, 7 centroids and 13 G vectors closed
under inversion with ζ̃(−G) = ζ̃*(G), so both the μ and the G carrier are
padded) is compared with a NumPy evaluation of
the kernel of ``docs/theory/td_agw.md`` that shares no code with it
(``_dense_reference``: explicit six-index sums, W(q) from an independent
numpy FFT of W_R).

Runs as a script on a 2x2 mesh.  cpu, one process with four host devices,
every k-convolution built inside ``ffi.fft.xla_reference()``::

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/test_tdagw_model.py

CUDA, four ranks with one GPU each, the router's own routes::

    lx run -N 1 -G 4 -n 4 -- python3 tests/test_tdagw_model.py

Under pytest it skips.  Gates (G) and red twins (R); each twin feeds a
deliberately wrong operator to the SAME checker as the gate it shadows and
must fail it:

G1  apply == dense reference, SEX + Hartree, SEX only, Hartree only    rel <= 1e-12
G2  dense_td_kernel(apply) == dense reference                           rel <= 1e-12
G3  Hermitian δρ_k gives Hermitian ΔΣ_k (W(q) Hermitian, V̄ real)        rel <= 1e-12
G4  δρ = 0 gives exactly 0;  G5 complex linearity                       rel <= 1e-12
G6  both terms == Hartree + SEX applied separately                      rel <= 1e-12
G7  a batch of three δρ == three single applications                    rel <= 1e-12
G8  the BSE identity: Hartree == bse_stack_matvec 'rpa' (ε = 0, V_q0 = V̄) and
    −SEX == 'bse' − 'rpa' on the same window                            rel <= 1e-12
G10 the TDGW seam: the bridge's ΔΣ (TDGW k order, Ha) == ½ K δρ with K permuted
    as the exporter permutes it, for a random TDGW k order               rel <= 1e-12
G10b the bridge's TDA mode == TDGW's TDA kernel semantics (ΔΣ_cv = ½ K_cv,c'v' δρ_c'v',
    ΔΣ_vc = ΔΣ_cv†, other blocks 0)                                      rel <= 1e-12
R1  one orbital leg conjugated (ψ_Y → ψ_Y*) fails G1
G9  keeping G = 0 in v(G) adds exactly the rank-one term
    (w_s/N_k) M v₀ ζ̃*(0)ζ̃(0) M† δρ                                     rel <= 1e-12
R2  G = 0 kept in v(G) fails G1
R3  W(−q) in place of W(q) (wrong convolution direction) fails G1
R4  a non-Hermitian W(q) fails G3
"""
import sys

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np

TOL = 1e-12
KGRID = (3, 2, 1)
NB, NS, NMU, NG = 4, 2, 7, 13


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _synthetic(rng):
    """Host operands: ψ (nk, nb, ns, nμ), W(q) Hermitian per q (nμ, nμ, kx, ky, kz),
    ζ̃ (nμ, nG) conjugation-symmetric on an inversion-closed G set, v (nG,) > 0 with
    v[g0] = 0, and the G = 0 slot g0."""
    nk = int(np.prod(KGRID))
    c = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    psi = c(nk, NB, NS, NMU)
    A = c(NMU, NMU, *KGRID)
    Wq = 0.5 * (A + np.conj(np.swapaxes(A, 0, 1)))
    # G = {0, +g_1..g_6, -g_1..g_6}: ζ̃(-g) = ζ̃*(g) and v(-g) = v(g), the structure of
    # the q = 0 fit, whose conjugation-closed normal equations make ζ_0(r) real.
    half = (NG - 1) // 2
    zeta = c(NMU, NG)
    zeta[:, 0] = zeta[:, 0].real
    zeta[:, 1 + half:] = np.conj(zeta[:, 1:1 + half])
    v = rng.random(NG) + 0.1
    v[1 + half:] = v[1:1 + half]
    g0 = 0
    v[g0] = 0.0
    return psi, Wq, zeta, v, g0


def _dense_reference(psi, Wq, zeta, v, *, sex=True, hartree=True):
    """K[k,n,m,k',n',m'] of docs/theory/td_agw.md, explicit sums (no LORRAX code)."""
    nk, nb, ns, _ = psi.shape
    w_s = 2.0 if ns == 1 else 1.0
    ijk = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1)
    dq = (ijk[:, None, :] - ijk[None, :, :]) % np.asarray(KGRID)
    qflat = np.ravel_multi_index((dq[..., 0], dq[..., 1], dq[..., 2]), KGRID)    # (k, k') -> k-k'
    Wkk = Wq.reshape(Wq.shape[0], Wq.shape[1], -1)[:, :, qflat]                 # (μ, ν, k, k')
    K = np.zeros((nk, nb, nb, nk, nb, nb), dtype=np.complex128)
    if sex:
        # A[k,n,k',n',μ] = Σ_t ψ*_{nk,t}(μ) ψ_{n'k',t}(μ);  B[k,m,k',m',ν] = Σ_s ψ_{mk,s}(ν) ψ*_{m'k',s}(ν)
        A = np.einsum("kntu,KNtu->knKNu", np.conj(psi), psi)
        B = np.einsum("kmsv,KMsv->kmKMv", psi, np.conj(psi))
        K -= np.einsum("knKNu,uvkK,kmKMv->knmKNM", A, Wkk, B) / nk
    if hartree:
        M = np.einsum("knsu,kmsu->knmu", np.conj(psi), psi)
        Vbar = np.einsum("uG,G,vG->uv", np.conj(zeta), v, zeta)
        K += w_s / nk * np.einsum("knmu,uv,KNMv->knmKNM", M, Vbar, np.conj(M))
    return K


def run_checks(mesh):
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from ffi import fft as F
    from runtime.padding import padded_mu_extent, round_up
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from tdagw.model import (ZETA_G_SPEC, V_G_SPEC, assemble_td_model, build_td_agw_apply,
                             dense_td_kernel, pad_td_state, strip_td_state, td_state_sharding)

    on_cpu = F.kconv_backend(mesh) != "mathdx"
    arm = F.xla_reference if on_cpu else (lambda: __import__("contextlib").nullcontext())
    rng = np.random.default_rng(20261016)
    psi, Wq, zeta, v, g0 = _synthetic(rng)
    nk = int(np.prod(KGRID))
    p = int(np.prod(mesh.devices.shape))
    nmu_pad = padded_mu_extent(NMU, mesh)
    nG_pad = round_up(NG, p)
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))

    def model_of(psi_h, Wq_h, v_h):
        psi_p = np.zeros((nk, NB, NS, nmu_pad), complex); psi_p[..., :NMU] = psi_h
        W_p = np.zeros((nmu_pad, nmu_pad) + KGRID, complex); W_p[:NMU, :NMU] = Wq_h
        W_R = np.fft.ifftn(W_p, axes=(2, 3, 4), norm="ortho")
        z_p = np.zeros((nmu_pad, nG_pad), complex); z_p[:NMU, :NG] = zeta
        v_p = np.zeros(nG_pad); v_p[:NG] = v_h
        return assemble_td_model(
            put(psi_p, P(None, None, None, "x")), put(W_R, P("x", "y", None, None, None)),
            put(z_p, ZETA_G_SPEC), put(v_p, V_G_SPEC), mesh_xy=mesh, kgrid=KGRID,
            k_frac=np.zeros((nk, 3)), band_lo=0, n_bands=NB, n_rmu=NMU)

    st = td_state_sharding(mesh)
    model = model_of(psi, Wq, v)

    def apply_of(mdl, **terms):
        with arm():
            fn = build_td_agw_apply(mesh, mdl, **terms)
        def call(x):
            xd = put(pad_td_state(x, mdl), st.spec)
            with arm():
                return strip_td_state(gather(fn(xd)), mdl)
        return call

    c = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    x = c(3, nk, NB, NB)
    K_both = _dense_reference(psi, Wq, zeta, v)
    K_sex = _dense_reference(psi, Wq, zeta, v, hartree=False)
    K_h = _dense_reference(psi, Wq, zeta, v, sex=False)
    ref = lambda K, x: np.einsum("knmKNM,bKNM->bknm", K, x)
    f_both, f_sex, f_h = apply_of(model), apply_of(model, hartree=False), apply_of(model, sex=False)
    out = {}
    y = f_both(x)
    out["G1 both == dense"] = _rel(y, ref(K_both, x))
    out["G1 sex == dense"] = _rel(f_sex(x), ref(K_sex, x))
    out["G1 hartree == dense"] = _rel(f_h(x), ref(K_h, x))
    with arm():
        Kd = dense_td_kernel(build_td_agw_apply(mesh, model), model, mesh_xy=mesh, batch=16)
    out["G2 dense_td_kernel == dense"] = _rel(Kd, K_both)
    xh = 0.5 * (x + np.conj(np.swapaxes(x, -1, -2)))
    yh = f_both(xh)
    out["G3 hermitian in -> out"] = float(np.max(np.abs(yh - np.conj(np.swapaxes(yh, -1, -2))))
                                          / np.max(np.abs(yh)))
    z = f_both(np.zeros_like(x))
    out["G4 zero -> zero (max |y|)"] = float(np.max(np.abs(z)))
    a, b = 0.3 - 1.7j, -2.1 + 0.4j
    x2 = c(3, nk, NB, NB)
    out["G5 linearity"] = _rel(f_both(a * x + b * x2), a * y + b * f_both(x2))
    out["G6 both == hartree + sex"] = _rel(y, f_h(x) + f_sex(x))
    single = np.concatenate([f_both(x[i:i + 1]) for i in range(3)])
    out["G7 batch == singles"] = _rel(y, single)

    # G8: the BSE's own matvec on the same window (c = v = window, X = δρ).
    sh_X = NamedSharding(mesh, P(None, "x", "y", None))
    Z = np.zeros((nmu_pad, nG_pad), complex); Z[:NMU, :NG] = zeta
    vp = np.zeros(nG_pad); vp[:NG] = v
    Vbar = np.einsum("uG,G,vG->uv", np.conj(Z), vp, Z)
    eps0 = put(np.zeros((nk, int(model["nb_pad"]))), P(None, None))
    Xb = put(np.moveaxis(pad_td_state(x, model), 1, -1), sh_X.spec)
    bse_args = (model["psi_X"], model["psi_Y"], model["psi_X"], model["psi_Y"], eps0, eps0,
                model["W_R"], put(Vbar, P("x", "y")), model["M"])
    with arm():
        rpa = gather(build_bse_stack_matvec(mesh, *KGRID, kernel="rpa")(Xb, *bse_args))
        bse = gather(build_bse_stack_matvec(mesh, *KGRID, kernel="bse")(Xb, *bse_args))
    to_td = lambda a: strip_td_state(np.moveaxis(a, -1, 1), model)
    out["G8 hartree == BSE rpa (eps=0, V_q0=Vbar)"] = _rel(f_h(x), to_td(rpa))
    out["G8 sex == BSE bse - rpa"] = _rel(f_sex(x), to_td(bse - rpa))

    # G10: the TDGW seam.  TDGW holds δρ in its WFN's k order and wants ΔΣ in Ha; the
    # bridge (make_tdgw_dsigma) and the exporter (kernel_export: K[rows][:, :, :, rows])
    # must apply the same permutation, or (b) and (c) of the validation diverge.
    from tdagw.tdgw_bridge import make_tdgw_dsigma
    rows = rng.permutation(nk)                      # model row of each "TDGW" k point
    with arm():
        seam = make_tdgw_dsigma(build_td_agw_apply(mesh, model), model, mesh, rows)
        d_seam = seam(x[0])
    K_tdgw = K_both[rows][:, :, :, rows]
    out["G10 bridge seam == exporter seam (Ha)"] = _rel(
        d_seam, 0.5 * np.einsum("knmKNM,KNM->knm", K_tdgw, x[0]))
    # G10b: the bridge's TDA mode is TDGW's TDA kernel semantics: ΔΣ_cv = ½ K_{cv,c'v'} δρ_{c'v'},
    # ΔΣ_vc = ΔΣ_cv†, all other blocks 0 (the first NB // 2 window bands are valence).
    nv = NB // 2
    cs, vs = slice(nv, NB), slice(0, nv)
    with arm():
        seam_tda = make_tdgw_dsigma(build_td_agw_apply(mesh, model), model, mesh, rows,
                                    tda_n_val=nv)
        d_tda = seam_tda(x[0])
    y_cv = 0.5 * np.einsum("kcvKCV,KCV->kcv", K_tdgw[:, cs, vs][:, :, :, :, cs, vs], x[0][:, cs, vs])
    ref_tda = np.zeros_like(x[0])
    ref_tda[:, cs, vs] = y_cv
    ref_tda[:, vs, cs] = np.conj(np.swapaxes(y_cv, -1, -2))
    out["G10b bridge TDA seam == TDGW TDA semantics"] = _rel(d_tda, ref_tda)

    # Red twins: each must FAIL the checker of the gate it shadows.
    twins = {}
    bad = dict(model)
    bad["psi_Y"] = jax.jit(jnp.conj, out_shardings=model["psi_Y"].sharding)(model["psi_Y"])
    twins["R1 conj on one leg (G1 sex)"] = _rel(apply_of(bad, hartree=False)(x), ref(K_sex, x))
    v0 = 0.75
    v_g0 = v.copy(); v_g0[g0] = v0
    y_g0 = apply_of(model_of(psi, Wq, v_g0), sex=False)(x)
    twins["R2 G=0 kept (G1 hartree)"] = _rel(y_g0, ref(K_h, x))
    M = np.einsum("knsu,kmsu->knmu", np.conj(psi), psi)
    head = v0 * np.outer(np.conj(zeta[:, g0]), zeta[:, g0])
    rank1 = (1.0 if NS == 2 else 2.0) / nk * np.einsum(
        "knmu,uv,KNMv,bKNM->bknm", M, head, np.conj(M), x)
    out["G9 G=0 difference == rank-one term"] = _rel(y_g0 - f_h(x), rank1)
    neg = [(-np.arange(n)) % n for n in KGRID]
    Wq_minus = Wq[:, :, neg[0]][:, :, :, neg[1]][:, :, :, :, neg[2]]
    twins["R3 W(-q) (G1 sex)"] = _rel(apply_of(model_of(psi, Wq_minus, v), hartree=False)(x),
                                      ref(K_sex, x))
    A = c(NMU, NMU, *KGRID)
    yh_bad = apply_of(model_of(psi, A, v))(xh)
    twins["R4 non-Hermitian W (G3)"] = float(
        np.max(np.abs(yh_bad - np.conj(np.swapaxes(yh_bad, -1, -2)))) / np.max(np.abs(yh_bad)))
    return out, twins


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    mesh = stack.mesh
    if tuple(mesh.devices.shape) != (2, 2):
        print(f"SKIP: needs a 2x2 mesh, got {tuple(mesh.devices.shape)}", flush=True)
        runtime.finalize_process(0)
        return 0
    out, twins = run_checks(mesh)
    gate_tol = {k: (0.0 if k.startswith("G4") else TOL) for k in out}
    bad = [k for k, v in out.items() if not v <= gate_tol[k]]
    caught = {k: v > 1e-6 for k, v in twins.items()}
    if jax.process_index() == 0:
        for k, v in out.items():
            print(f"{'ok  ' if v <= gate_tol[k] else 'FAIL'} {k:46s} {v:.3e}", flush=True)
        for k, v in twins.items():
            print(f"{'ok  ' if caught[k] else 'FAIL'} {k:46s} {v:.3e} (twin must exceed 1e-6)",
                  flush=True)
        print(f"TDAGW_MODEL_GATE {'PASS' if not bad and all(caught.values()) else 'FAIL'}: "
              f"{len(out) - len(bad)}/{len(out)} gates, {sum(caught.values())}/{len(caught)} "
              f"red twins caught, on {jax.device_count()} devices "
              f"({jax.devices()[0].platform})", flush=True)
    rc = 1 if (bad or not all(caught.values())) else 0
    runtime.finalize_process(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
