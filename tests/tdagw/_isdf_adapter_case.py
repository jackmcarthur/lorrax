"""The propagator's ISDF adapter on a synthetic TDMODEL model (child of test_prop_isdf_adapter_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_isdf_adapter_case.py

Prints one JSON line, with the B10 collective tables: the collectives the
adapter's compiled program adds to the model's own, for the adapter and for
the twin that reshards by one sharding constraint.  The synthetic operands and the explicit-sum kernel
follow tests/test_tdagw_model.py (TDMODEL's _synthetic and _dense_reference,
copied so this file does not import another test module):
K[k,n,m,k',n',m'] in Ry on the model's flat C-order k grid, SEX
-(1/N_k) A W B and Hartree (w_s/N_k) M Vbar M^H.
"""
import json

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

KGRID = (3, 2, 1)
NB, NS, NMU, NG = 4, 2, 7, 13


def synthetic(rng):
    nk = int(np.prod(KGRID))
    c = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    psi = c(nk, NB, NS, NMU)
    A = c(NMU, NMU, *KGRID)
    Wq = 0.5 * (A + np.conj(np.swapaxes(A, 0, 1)))
    half = (NG - 1) // 2
    zeta = c(NMU, NG)
    zeta[:, 0] = zeta[:, 0].real
    zeta[:, 1 + half:] = np.conj(zeta[:, 1:1 + half])
    v = rng.random(NG) + 0.1
    v[1 + half:] = v[1:1 + half]
    v[0] = 0.0
    return psi, Wq, zeta, v


def dense_reference(psi, Wq, zeta, v, *, sex=True, hartree=True):
    nk, nb, ns, _ = psi.shape
    w_s = 2.0 if ns == 1 else 1.0
    ijk = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1)
    dq = (ijk[:, None, :] - ijk[None, :, :]) % np.asarray(KGRID)
    qflat = np.ravel_multi_index((dq[..., 0], dq[..., 1], dq[..., 2]), KGRID)
    Wkk = Wq.reshape(Wq.shape[0], Wq.shape[1], -1)[:, :, qflat]
    K = np.zeros((nk, nb, nb, nk, nb, nb), dtype=np.complex128)
    if sex:
        A = np.einsum("kntu,KNtu->knKNu", np.conj(psi), psi)
        B = np.einsum("kmsv,KMsv->kmKMv", psi, np.conj(psi))
        K -= np.einsum("knKNu,uvkK,kmKMv->knmKNM", A, Wkk, B) / nk
    if hartree:
        M = np.einsum("knsu,kmsu->knmu", np.conj(psi), psi)
        Vbar = np.einsum("uG,G,vG->uv", np.conj(zeta), v, zeta)
        K += w_s / nk * np.einsum("knmu,uv,KNMv->knmKNM", M, Vbar, np.conj(M))
    return K


def main():
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from ffi import fft as F
    from runtime.padding import padded_mu_extent, round_up
    from common.collectives import resolve_mesh
    from tdagw.model import ZETA_G_SPEC, V_G_SPEC, assemble_td_model
    from tdagw.self_energy import isdf_model
    from tdagw import propagate as tp

    mesh = resolve_mesh()
    rng = np.random.default_rng(20261016)
    psi, Wq, zeta, v = synthetic(rng)
    nk = int(np.prod(KGRID))
    p = int(np.prod(mesh.devices.shape))
    nmu_pad = padded_mu_extent(NMU, mesh)
    nG_pad = round_up(NG, p)
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    ijk = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1)
    k_frac_model = ijk / np.asarray(KGRID, float)

    psi_p = np.zeros((nk, NB, NS, nmu_pad), complex); psi_p[..., :NMU] = psi
    W_p = np.zeros((nmu_pad, nmu_pad) + KGRID, complex); W_p[:NMU, :NMU] = Wq
    W_R = np.fft.ifftn(W_p, axes=(2, 3, 4), norm="ortho")
    z_p = np.zeros((nmu_pad, nG_pad), complex); z_p[:NMU, :NG] = zeta
    v_p = np.zeros(nG_pad); v_p[:NG] = v
    model = assemble_td_model(
        put(psi_p, P(None, None, None, "x")), put(W_R, P("x", "y", None, None, None)),
        put(z_p, ZETA_G_SPEC), put(v_p, V_G_SPEC), mesh_xy=mesh, kgrid=KGRID,
        k_frac=k_frac_model, band_lo=0, n_bands=NB, n_rmu=NMU)

    # the propagator's k order: a random permutation of the grid, shifted by lattice vectors
    order = rng.permutation(nk)
    k_hdr = k_frac_model[order] + rng.integers(-1, 2, size=(nk, 3)) * np.array([1, 1, 0])
    nk_pad = round_up(nk, p)
    ntr = 2
    occ = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)
    rho_eq_h = np.zeros((nk_pad, NB, NB), complex); rho_eq_h[:nk] = occ
    g = rng.standard_normal((ntr, nk, NB, NB)) + 1j * rng.standard_normal((ntr, nk, NB, NB))
    rho_h = np.zeros((ntr, nk_pad, NB, NB), complex)
    rho_h[:, :nk] = occ + 0.1 * (g + np.conj(np.swapaxes(g, -1, -2)))
    rho_eq = tp._place(rho_eq_h, mesh, 0)
    rho = tp.place_state(rho_h, mesh)
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))

    out = {}
    for terms in ("both", "sex", "hartree"):
        sex, hartree = terms in ("both", "sex"), terms in ("both", "hartree")
        K = dense_reference(psi, Wq, zeta, v, sex=sex, hartree=hartree)
        Kp = K[order][:, :, :, order]                       # propagator k order
        drho = rho_h[:, :nk] - rho_eq_h[None, :nk]
        ref = 0.5 * np.einsum("knmKNM,tKNM->tknm", Kp, drho)  # Ha
        for tag, req, scale in (("gate", rho_eq, 1.0), ("twin_no_halving", rho_eq, 2.0),
                                ("twin_rho_eq_zero", tp._place(np.zeros_like(rho_eq_h), mesh, 0), 1.0)):
            sm = isdf_model(model, mesh, k_frac=k_hdr, rho_eq=req, sex=sex, hartree=hartree)
            with F.xla_reference():
                y = jax.jit(lambda prm, r: sm.apply(prm, r, 0.0))(sm.params, rho)
            got = scale * gather(y)
            rel = float(np.max(np.abs(got[:, :nk] - ref)) / np.max(np.abs(ref)))
            pad = float(np.max(np.abs(got[:, nk:]), initial=0.0))
            out[f"{terms}:{tag}"] = dict(rel=rel, pad=pad)
    # TDGW TDA semantics (nv = 2 occupied rows): dSigma_cv = 1/2 K_{cv,c'v'} drho_{c'v'}, vc = cv^H, rest 0
    nv = 2
    K = dense_reference(psi, Wq, zeta, v)[order][:, :, :, order]
    drho = rho_h[:, :nk] - rho_eq_h[None, :nk]
    x = np.zeros_like(drho); x[:, :, nv:, :nv] = drho[:, :, nv:, :nv]
    y_cv = 0.5 * np.einsum("knmKNM,tKNM->tknm", K, x)[:, :, nv:, :nv]
    ref = np.zeros_like(drho); ref[:, :, nv:, :nv] = y_cv
    ref[:, :, :nv, nv:] = np.conj(np.swapaxes(y_cv, -1, -2))
    for tag, tda in (("tda", nv), ("twin_tda_full", None)):
        sm = isdf_model(model, mesh, k_frac=k_hdr, rho_eq=rho_eq, tda_n_val=tda)
        with F.xla_reference():
            y = jax.jit(lambda prm, r: sm.apply(prm, r, 0.0))(sm.params, rho)
        got = gather(y)
        out[f"both:{tag}"] = dict(rel=float(np.max(np.abs(got[:, :nk] - ref)) / np.max(np.abs(ref))),
                                  pad=float(np.max(np.abs(got[:, nk:]), initial=0.0)))
    out["mesh"] = list(mesh.devices.shape)
    out["nk_pad"] = nk_pad
    out.update(_collective_tables(mesh, model, sm_both(model, mesh, k_hdr, rho_eq), rho,
                                  ntr, nk))
    print("RESULT " + json.dumps(out), flush=True)


def sm_both(model, mesh, k_hdr, rho_eq):
    from tdagw.self_energy import isdf_model
    return isdf_model(model, mesh, k_frac=k_hdr, rho_eq=rho_eq)


def _twin_constraint_apply(static, params, rho, t):
    """The adapter as it was: one sharding constraint between the two layouts."""
    import jax.numpy as jnp
    from tdagw.model import build_td_agw_apply, td_state_sharding
    from tdagw.self_energy import _ISDF_OPERANDS
    mesh, kgrid, nspinor, nb_pad, rows, perm, nk, sex, hartree, _ = static
    ops, rho_eq = params
    d = rho - rho_eq[None]
    nb, nk_pad = d.shape[-1], d.shape[1]
    d = jnp.pad(d, ((0, 0), (0, 0), (0, nb_pad - nb), (0, nb_pad - nb)))
    d = jax.lax.with_sharding_constraint(d, td_state_sharding(mesh))
    x = jnp.take(d, jnp.asarray(perm), axis=1)
    y = build_td_agw_apply(mesh, dict(zip(_ISDF_OPERANDS, ops), kgrid=kgrid, nspinor=nspinor),
                           sex=sex, hartree=hartree)(x)
    y = jnp.take(y, jnp.asarray(rows), axis=1)[..., :nb, :nb]
    return 0.5 * jnp.pad(y, ((0, 0), (0, nk_pad - nk), (0, 0), (0, 0)))


def _collectives(hlo):
    """[(kind, result shape)] of every collective in an optimized HLO module."""
    import re
    rows = []
    for line in hlo.splitlines():
        m = re.search(r"\b(all-gather|all-to-all|all-reduce|collective-permute|"
                      r"reduce-scatter)(-start)?\(", line)
        if m and "=" in line.split(m.group(0))[0]:
            shape = re.sub(r"\{[^}]*\}", "", line.split("=", 1)[1].split(m.group(0))[0]).strip()
            rows.append((m.group(1), shape))
    return rows


def _collective_tables(mesh, model, sm, rho, ntr, nk):
    """B10: the adapter adds exactly the movers' four all_to_all to the model's own collectives."""
    from collections import Counter
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi import fft as F
    from tdagw.model import build_td_agw_apply, td_state_sharding
    from tdagw.self_energy import _ISDF_OPERANDS
    kspec = NamedSharding(mesh, P(None, ("x", "y"), None, None))

    def table(fn, *args):
        with F.xla_reference():
            return Counter(_collectives(jax.jit(fn).lower(*args).compile().as_text()))

    ops = sm.params[0]
    nb_pad = int(model["nb_pad"])
    x_face = jax.device_put(np.zeros((ntr, nk, nb_pad, nb_pad), complex), td_state_sharding(mesh))
    m_apply = build_td_agw_apply(mesh, dict(zip(_ISDF_OPERANDS, ops), kgrid=model["kgrid"],
                                            nspinor=model["nspinor"]))
    model_t = table(lambda x: m_apply(x), x_face)
    static = sm.apply.args[0]
    pinned = lambda f: (lambda prm, r: jax.lax.with_sharding_constraint(f(prm, r), kspec))
    adapter_t = table(pinned(lambda prm, r: sm.apply(prm, r, 0.0)), sm.params, rho)
    twin_t = table(pinned(lambda prm, r: _twin_constraint_apply(static, prm, r, 0.0)),
                   sm.params, rho)

    def extra(t):
        add, gone = t - model_t, model_t - t
        return dict(added=sorted(f"{k}:{s}" for (k, s), n in add.items() for _ in range(n)),
                    removed=sorted(f"{k}:{s}" for (k, s), n in gone.items() for _ in range(n)))
    return {"b10:adapter": extra(adapter_t), "b10:twin_constraint": extra(twin_t)}


if __name__ == "__main__":
    main()
