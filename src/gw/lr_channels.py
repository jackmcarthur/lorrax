"""The dressed long-range channels of the static W, at every coarse q.

What the SR/LR W densifier (``bse.bse_densify``, srlr, the only W model)
needs from GW, computed beside the static Dyson solve because χ₀ exists only
there.  With the long-range channels L of ``vcoul.lr_channels`` (plane waves
q + G_a, kernel v_a = v(q+G_a)) and the ISDF rows

    F_aμ = ζ_q(μ, G_a)/√Ω,        V = V_b + Fᴴ diag(v) F,

the static W = (1 − Vχ)⁻¹V splits exactly at every q (Schur complement):

    W = J + F̃ᴴ D F̃,   J = (1 − V_b χ)⁻¹ V_b,   F̃ = F (1 − χV_b)⁻¹,
    P = F̃ χ Fᴴ,       D = [diag(1/v) − P]⁻¹ = (1 − diag(v) P)⁻¹ diag(v).

F̃ (n_L × N_μ) are the SR-dressed long-wavelength form factors and P (n_L × n_L)
the SR-dressed proper polarizability of the channels; D is the screened
long-range interaction (for one channel, D = v/(1 − vP), the screened head).
J is never formed.  With A = 1 − χV, A⁻¹ = 1 + χW and Woodbury on
1 − χV_b = A + χFᴴ diag(v) F,

    G = F + (Fχ) W,   Q = G (χFᴴ),   F̃ = (1 + Q diag(v))⁻¹ G,   P = (1 + Q diag(v))⁻¹ Q,

so the cost is three skinny products (n_L rows) against χ₀ and W, and χ₀'s
two are taken before the Dyson solve donates it.  At Γ the literal G = 0
channel has v = 0 (V(0) is the body), so it carries no LR term there.

Symmetry.  F̃ rows transform as ζ rows (one leg, with the channel's G and the
nonsymmorphic τ phase), and P as c_a c_b* P on the channel pair; the BSE
unfolds them through ``symmetry_maps``.  Files keep the canonical centroid
order; the restart group is ``W0_lr`` (``file_io.tagged_arrays.write_lr_channels_to_h5``).
"""
from __future__ import annotations

import os

import numpy as np
from common.log import get_logger

log = get_logger(__name__)



def slab_shells(bvec) -> int:
    """|G_z| shells of the slab channel set: the vq_interp long-range criterion.

    One owner for "which G_z channels are long-range": the energy cutoff with
    a two-shell floor of ``bse.vq_interp.lr_fit_degrees`` (decisions.md
    2026-08-10).
    """
    from bse.vq_interp import lr_fit_degrees
    return max(lr_fit_degrees({"bvec": np.asarray(bvec, dtype=np.float64)}))


def dress(F, Fchi, chiFh, FchiW, v):
    """The Woodbury form of the Schur split: ``(F̃, P)`` from three skinny products.

    Parameters
    ----------
    F : (n_q, n_L, N) complex
        Bare channel rows ζ_q(μ, G_a)/√Ω.
    Fchi, chiFh, FchiW : (n_q, n_L, N), (n_q, N, n_L), (n_q, n_L, N)
        F χ, χ Fᴴ and (F χ) W at the same q.
    v : (n_q, n_L) float
        Channel kernel v(q + G_a), Ry·bohr³.

    Returns
    -------
    Ft : (n_q, n_L, N) — F̃ = (1 + Q diag v)⁻¹ (F + FχW)
    P  : (n_q, n_L, n_L) — (1 + Q diag v)⁻¹ Q, Q = (F + FχW) χFᴴ; Hermitian
    """
    G = F + FchiW
    Q = G @ chiFh
    A = np.eye(Q.shape[-1])[None] + Q * v[:, None, :]
    Ft = np.linalg.solve(A, G)
    P = np.linalg.solve(A, Q)
    return Ft, 0.5 * (P + np.conj(np.swapaxes(P, -1, -2)))


def read_channel_rows(zeta_path, q_frac, *, sys_dim, kgrid, mc_average_vcoul_body):
    """Bare channel rows F (n_q, n_L, n_rmu) in canonical order, and the channel table.

    One host hyperslab of n_L columns per q from ``zeta_q_G`` (the q rows of
    the file are the solve's q list); a channel outside the stored sphere is
    a refusal, never a zero.  The lattice (blat·bvec) and Ω are the file's
    own mean-field header.  The channel kernel is the value V carries: the
    head slot takes ``v_q_g_flat.v_head_fn_in_V`` (the mini-BZ average on a
    3D ``mc_average_vcoul_body`` deck), so V_b = V − Fᴴ diag(v) F is V
    without those slots.  Returns ``(F, channels, n_z, head_kernel)``.
    """
    from .v_q_g_flat import v_head_fn_in_V
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import lr_channels
    from zeta_loader import ZetaLoader

    with ZetaLoader(zeta_path) as zl:
        bvec = np.asarray(zl.bvec, dtype=np.float64) * float(zl.blat)
        cell_volume = float(zl.cell_volume)
        n_z = slab_shells(bvec) if int(sys_dim) == 2 else 0
        vfn = v_head_fn_in_V(mc_average_vcoul_body=mc_average_vcoul_body,
                             sys_dim=sys_dim, kgrid=tuple(kgrid), bvec=bvec,
                             cell_volume=cell_volume)
        ch = lr_channels(q_frac, bvec=bvec, sys_dim=sys_dim, n_z=n_z,
                         head_fn=None if vfn is None else (lambda K: cell_volume * vfn(K)))
        gcomp = np.asarray(zl.gvec_components, dtype=np.int64)
        ngk = np.asarray(zl.ngk_per_q, dtype=np.int64)
        n_rmu = int(zl.n_rmu)
        if gcomp.shape[0] != len(q_frac):
            raise ValueError(
                f"GATE lr_channel_q_rows: {zeta_path} holds {gcomp.shape[0]} q "
                f"rows; want the solve's {len(q_frac)}; why: the channel rows "
                "would pair with another q's χ₀ and W.")
        F = np.empty((len(q_frac), ch.gvec.shape[1], n_rmu), np.complex128)
        for qi in range(len(q_frac)):
            lut = {tuple(g): i for i, g in enumerate(gcomp[qi, :, :ngk[qi]].T)}
            slots = [lut.get(tuple(int(x) for x in g)) for g in ch.gvec[qi]]
            if any(s is None for s in slots):
                raise ValueError(
                    f"GATE lr_channel_in_sphere: q {qi} channel G "
                    f"{ch.gvec[qi].tolist()} not all in the ζ sphere; want "
                    "every long-range channel stored; fix: raise the ζ cutoff.")
            order = np.argsort(slots)
            cols = zl.read_zeta_G_local((qi, slice(None), list(np.asarray(slots)[order])))
            F[qi][order] = np.asarray(cols).T
    return (F / np.sqrt(cell_volume), ch, n_z,
            "point" if vfn is None else "minibz_average")


def begin(chi0_q, *, config, meta, sym, centroid_indices, wedge, mesh_xy):
    """Read the channel rows and take Fχ, χFᴴ before the Dyson solve donates χ₀.

    Returns the hand-off for :func:`finish`, or ``None`` when this run has no
    long-range channels (sys_dim not 2/3, no ζ file, an unreduced q list that
    is not the file's).  χ₀ (n_q, μ, μ) is the solve's operand at
    ``P(None,'x','y')`` in the packed centroid order.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from .v_q_g_flat import _resolve_ibz_q_list

    sys_dim = int(getattr(meta, "sys_dim", 0) or 0)
    zeta_path = os.path.join(getattr(config, "input_dir", "") or ".", "tmp", "zeta_q.h5")
    if sys_dim not in (2, 3) or not os.path.isfile(zeta_path):
        return None
    if wedge is not None:
        q_frac = np.asarray(wedge[0]["q_irr_frac"], dtype=np.float64)
    else:
        _, q_frac, *_rest, reduced = _resolve_ibz_q_list(
            sym=sym, centroid_indices=centroid_indices, kgrid=tuple(meta.kgrid),
            fft_grid=tuple(meta.fft_grid), context="W long-range channels")
        if reduced:
            return None
        q_frac = np.asarray(q_frac, dtype=np.float64)
    F, ch, n_z, head_kernel = read_channel_rows(
        zeta_path, q_frac, sys_dim=sys_dim, kgrid=meta.kgrid,
        mc_average_vcoul_body=bool(getattr(config.head, "mc_average_vcoul_body", False)))
    n_rmu = F.shape[-1]
    basis = getattr(meta, "mu_basis", None)
    rep = NamedSharding(mesh_xy, P())
    # The rows are host data (n_q·n_L·N_μ·16 B, n_L ≤ 9), packed on the host
    # and carried as two μ-sharded copies, μ on x and on y (the g0_X/g0_Y
    # pattern), so each product contracts in χ₀'s and W's (μ_X, ν_Y) layout.
    if basis is not None and not basis.is_identity:
        F_run = basis.pack_host(F, axis=2)
    else:
        F_run = np.zeros(F.shape[:2] + (int(chi0_q.shape[-1]),), np.complex128)
        F_run[..., :n_rmu] = F
    if int(F_run.shape[-1]) != int(chi0_q.shape[-1]):
        raise ValueError(
            f"lr_channels: channel rows carry {F_run.shape[-1]} centroids, χ₀ "
            f"{chi0_q.shape[-1]}")
    from common.collectives import device_put_process_local
    F_x = device_put_process_local(F_run, NamedSharding(mesh_xy, P(None, None, "x")))
    F_y = device_put_process_local(F_run, NamedSharding(mesh_xy, P(None, None, "y")))
    # The skinny results (n_q, n_L, N_μ) come back replicated for the host dressing.
    fchi = jax.jit(lambda f, c: jnp.einsum("qam,qmn->qan", f, c), out_shardings=rep)
    chif = jax.jit(lambda c, f: jnp.einsum("qmn,qan->qma", c, jnp.conj(f)), out_shardings=rep)
    # The physical χ is the solve's prefactor times the stored χ₀:
    # W = (1 − pref·Vχ₀)⁻¹V (gw.w_isdf.solve_w).
    from .w_isdf import _w_solve_pref_scalar
    pref = _w_solve_pref_scalar(meta)
    Fchi = pref * np.asarray(fchi(F_x, chi0_q))
    chiFh = pref * np.asarray(chif(chi0_q, F_y))
    return dict(F_run=F_run, Fchi=Fchi, chiFh=chiFh, ch=ch, n_z=n_z,
                head_kernel=head_kernel,
                q_frac=q_frac, n_rmu=n_rmu, sys_dim=sys_dim, basis=basis,
                mesh_xy=mesh_xy)


def finish(state, W_q):
    """(F χ) W, the Woodbury dressing, and the deposit for the restart writer."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh_xy = state["mesh_xy"]
    fw = jax.jit(lambda f, w: jnp.einsum("qam,qmn->qan", f, w),
                 out_shardings=NamedSharding(mesh_xy, P()))
    from common.collectives import device_put_process_local
    FchiW = np.asarray(fw(device_put_process_local(
        state["Fchi"], NamedSharding(mesh_xy, P(None, None, "x"))), W_q))
    Ft, Pm = dress(state["F_run"], state["Fchi"], state["chiFh"], FchiW,
                   state["ch"].v)
    # P = F χ_b Fᴴ, χ_b = (1 − χV_b)⁻¹χ the response to the positive V_b, is
    # negative definite for the NSD static χ₀ (MoS₂ 9×9: max λ_max/|λ_min| =
    # −9·10⁻⁵).  Γ is excluded: there the G = 0 channel's P vanishes by
    # orthogonality, sign undetermined.  A χ without the Dyson prefactor gave
    # +4·10³, a bare-v split of a mini-BZ-averaged V +1.  The GW run goes on;
    # only W0_lr is withheld, so an srlr BSE refuses on this restart by name.
    lam = np.linalg.eigvalsh(Pm)
    off = np.linalg.norm(state["q_frac"], axis=1) > 1e-12
    ratio = lam[:, -1] / np.maximum(np.abs(lam[:, 0]), 1e-300)
    worst = float(np.max(ratio[off])) if off.any() else float("nan")
    if worst > 1e-3:
        import warnings
        warnings.warn(
            f"GATE lr_channels_negative_P: got max λ_max/|λ_min| = {worst:.3e} "
            "for W0's dressed channel polarizability P at q != 0; want <= 1e-3 "
            "(P <= 0); why: χ₀, V and W disagree (prefactor, kernel or q "
            "rows), so F~ and P would not split this W0.  W0_lr is not "
            "written; a kgrid_fine BSE (srlr W interpolation) will refuse on "
            "this restart.",
            RuntimeWarning, stacklevel=2)
        return
    basis = state["basis"]
    if basis is not None and not basis.is_identity:
        Ft = basis.unpack_host(Ft, axis=2)
    from .restart_q_storage import deposit_lr_channels
    deposit_lr_channels(dict(
        Ft=Ft[..., :state["n_rmu"]], P=Pm, gvec=state["ch"].gvec,
        v=state["ch"].v, q_frac=state["q_frac"], n_z=int(state["n_z"]),
        sys_dim=int(state["sys_dim"]), head_kernel=state["head_kernel"]))
    log.debug(f"  W long-range channels: {Pm.shape[-1]} per q on {Pm.shape[0]} q "
          f"(sys_dim {state['sys_dim']}, |G_z| <= {state['n_z']}); P definiteness "
          f"max λ_max/|λ_min| {worst:.3e}; dressed form factors F~ and P "
          f"deposited for the restart")


# ---------------------------------------------------------------------------
# The smooth long-range model between coarse q (consumed by bse.bse_densify)
# ---------------------------------------------------------------------------
#: Window radii as fractions of the BZ inscribed radius q_ins: the model owns
#: W for |q| < r_a, hands it to the interpolant by |q| = r_b (quintic
#: smoothstep), and is fitted on 0 < |q| < r_b.  r_b < q_ins keeps every
#: windowed q on one WS slot.  MoS₂ 9×9 (q_ins 0.607 bohr⁻¹): 0.20 → 0.55.
WINDOW_INNER = 1.0 / 3.0
WINDOW_OUTER = 0.9
#: Highest polynomial degree of the F̃ fit; lowered until the monomial count
#: is at most half the fitted q (a determined least-squares problem).
MAX_DEG_F = 4
#: Highest degree in |q|² of the radial P fit.
MAX_DEG_P = 3


def window(qn, r_a, r_b):
    """w(|q|): 1 below r_a, quintic smoothstep 1 → 0 on [r_a, r_b], 0 above."""
    x = np.clip((np.asarray(qn, dtype=np.float64) - r_a) / (r_b - r_a), 0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def _monomials(q, deg, scale):
    """All monomials of total degree ≤ deg in the periodic components of q/scale."""
    from itertools import combinations_with_replacement
    x = np.asarray(q, dtype=np.float64) / scale
    cols = [np.ones(len(x))]
    for d in range(1, deg + 1):
        for c in combinations_with_replacement(range(x.shape[1]), d):
            cols.append(np.prod(x[:, list(c)], axis=1))
    return np.stack(cols, axis=1)


def fit_model(qc, Ft, P, *, r_mu, S_cart, c0, sys_dim, r_fit):
    """Fit the smooth long-range model on the coarse q with 0 < |q| < r_fit.

    * F̃_aμ(q) e^{+iq·r_μ} — complex polynomial in Cartesian q (in-plane for a
      slab), one per (a, μ); the Bloch phase of the home-cell centroid is
      removed first so what is fitted is smooth.
    * P_ab(q) = |q|^{m_ab} y_ab(|q|²), y a polynomial in |q|², m = 2 on the
      literal-G = 0 channel's row and column (charge neutrality), anchored
      by one least-squares row to the head tensor: y_00(0) = S̄ (S's mean
      diagonal over the periodic directions), y_0a(0) = 0.  A soft anchor,
      because the ISDF-projected P_00/q² need not tend to S exactly (MoS₂
      9×9: hard anchor −0.2434, soft −0.2368, 12×12 truth −0.2393 at
      |q| = 0.101 bohr⁻¹).

    qc (n_q, 3) Cartesian WS-representative q, bohr⁻¹; Ft (n_q, n_L, N);
    P (n_q, n_L, n_L); r_mu (N, 3) Cartesian centroids, bohr; S_cart (3, 3);
    c0 the G = 0 channel index.  Returns the model (a dict of coefficients)
    and the fit residuals.
    """
    dim = 2 if int(sys_dim) == 2 else 3
    qn = np.linalg.norm(qc, axis=1)
    sel = np.flatnonzero((qn > 1e-9) & (qn < r_fit))
    if sel.size < 2:
        raise ValueError(
            f"GATE srlr_fit_points: {sel.size} coarse q inside the fit radius "
            f"{r_fit:.3f} bohr^-1; want >= 2; why: no long-range model can be "
            "fitted; fix: a denser coarse GW grid.")
    deg = MAX_DEG_F
    while deg > 0 and _monomials(qc[:1, :dim], deg, 1.0).shape[1] > sel.size // 2:
        deg -= 1
    A = _monomials(qc[sel, :dim], deg, r_fit)
    ph = np.exp(1j * (qc[sel] @ r_mu.T))                       # (n_fit, N)
    Y = (Ft[sel] * ph[:, None, :]).reshape(sel.size, -1)
    cF = np.linalg.lstsq(A, Y, rcond=None)[0]
    res_F = np.linalg.norm(A @ cF - Y) / max(np.linalg.norm(Y), 1e-300)

    n_L = P.shape[-1]
    q2 = qn[sel] ** 2
    shells = np.unique(np.round(qn[sel], 6)).size
    degP = max(0, min(MAX_DEG_P, shells - 1))
    m = np.zeros((n_L, n_L), int)
    m[c0, :] = m[:, c0] = 2
    Pf = P[sel].reshape(sel.size, -1)
    rad = np.stack([q2 ** j for j in range(degP + 1)], axis=1)  # (n_fit, degP+1)
    cP = np.zeros((degP + 1, n_L * n_L), np.complex128)
    S_iso = float(np.trace(np.real(S_cart)[:dim, :dim])) / dim
    for mm in (0, 2):
        cols = np.flatnonzero(m.ravel() == mm)
        A_P, b_P = rad, Pf[:, cols] / q2[:, None] ** (mm // 2)
        if mm == 2:     # one anchor row: y(0) = S on the head, 0 on its wings
            A_P = np.vstack([A_P, np.eye(1, degP + 1)])
            b_P = np.vstack([b_P, (cols == c0 * n_L + c0)[None] * S_iso])
        cP[:, cols] = np.linalg.lstsq(A_P, b_P, rcond=None)[0]
    model = dict(cF=cF, deg=deg, cP=cP, degP=degP, m=m, c0=int(c0),
                 r_fit=float(r_fit), r_mu=r_mu, dim=dim, n_L=n_L, N=Ft.shape[-1])
    Pm = model_P(model, qc[sel])
    res_P = np.linalg.norm(Pm - P[sel]) / np.linalg.norm(P[sel])
    return model, dict(n_fit=int(sel.size), deg_F=deg, deg_P=degP,
                       relres_F=float(res_F), relres_P=float(res_P))


def model_P(model, qc):
    """P_m(q) (n, n_L, n_L) from :func:`fit_model`'s coefficients."""
    q2 = np.sum(np.asarray(qc, dtype=np.float64) ** 2, axis=1)
    rad = np.stack([q2 ** j for j in range(model["degP"] + 1)], axis=1)
    Pm = (rad @ model["cP"]) * q2[:, None] ** (model["m"].ravel()[None, :] // 2)
    Pm = Pm.reshape(-1, model["n_L"], model["n_L"])
    return 0.5 * (Pm + np.conj(np.swapaxes(Pm, -1, -2)))


def model_rows(model, qc):
    """F̃_m(q) (n, n_L, N) from :func:`fit_model`'s coefficients."""
    qc = np.asarray(qc, dtype=np.float64)
    A = _monomials(qc[:, :model["dim"]], model["deg"], model["r_fit"])
    F = (A @ model["cF"]).reshape(len(qc), model["n_L"], model["N"])
    return F * np.exp(-1j * (qc @ model["r_mu"].T))[:, None, :]


def screened_lr(P, v):
    """D = [diag(1/v) − P]⁻¹ = (1 − diag(v) P)⁻¹ diag(v), (n, n_L, n_L); v = 0 allowed."""
    n = P.shape[-1]
    return np.linalg.solve(np.eye(n)[None] - v[:, :, None] * P,
                           v[:, :, None] * np.eye(n)[None])
