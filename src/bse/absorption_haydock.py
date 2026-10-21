"""ε₂(ω) via Haydock continued fraction — eigenvector-free route.

Mirrors BGW's ``BSE/haydock.f90 → iterate.f90 → absh.f90`` pathway:

  - seed Lanczos at the dipole vector ``|d^α⟩``, one per polarisation
    α ∈ {x, y, z};  3-block recursion shares the BSE matvec
  - iterate ``n_iter`` steps without reorthogonalisation, accumulating
    ``α_n`` and ``β_n``  (``β_n`` already factored out of the next vector)
  - at evaluation time, build ε₂(ω) from the backward continued fraction:

        g(z) = 1 / (z − α_1 − β_1²/(z − α_2 − β_2²/(z − α_3 − …)))
        ε_2^α(ω) = (16π² / (V_cell · N_k · n_spin · n_spinor))
                   × ‖d^α‖² · (−Im g(ω+iη) / π)

The Haydock trick: never form a single eigenvector; spectrum lives in
the (α_n, β_n) recurrence.

THIS ROUTE IS THE CONVENTION-FREE WITNESS, and it needed no change when
the two eigenvector routes were adjudicated on 2026-08-09.  It never
forms an ``A``, so it cannot conjugate one wrongly: it evaluates
``⟨d|(z − H)⁻¹|d⟩`` with the seed ``|d⟩ = Σ_t d_t |t⟩`` taken straight
from ``slice_dipole_to_bse_window`` (correct, because
``⟨t|r̂|0⟩ = ⟨ck|r̂|vk⟩ = d_t``) and the Lanczos inner product
``⟨s|H|s⟩`` conjugating the bra as usual (on a ``bse_k_grid`` bundle the
seed is the bundle's own fine-gauge dipoles, ``window_dipoles``).  Its
spectral weights are
therefore ``|⟨S|d⟩|² = |Σ_t conj(A_t) d_t|²``, which is exactly
``|⟨0|r̂|S⟩|²`` for the contraction in
``absorption_common.exciton_dipole_projections``.  Any sum-over-states
ε₂ that disagrees with this route disagrees with the resolvent.

Convention on ``n_spin`` vs ``n_spinor``: BGW's ``n_spin`` is 2 only for
collinear spin-polarised calcs.  For our spinor calcs ``n_spin = 1`` and
``n_spinor = 2``.
"""
from __future__ import annotations

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import argparse
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import get_donated_kfft_kminor

from .absorption_common import (RYD2EV, build_dipole_vector_bse, jdos_from_transitions, kramers_kronig_eps1, window_dipoles, write_absorption_dat, write_absorption_h5)
from .bse_head import head_shifted_eps_c
from .bse_window import _parse_wfn_path
from .bse_io import (load_bse_data_from_restart_sharded)
from file_io.restart_bundle import (_find_restart_file)
from .bse_ring_comm import create_mesh_2d, make_bse_shardings
from .bse_stack_matvec import build_bse_stack_matvec


#: Cartesian polarization letters, in the order of the dipole's first axis and
#: of the ``absorption_haydock_b<1|2|3>_eh.dat`` file suffixes.
POLARIZATIONS = "xyz"


def parse_polarizations(text: str) -> tuple[int, ...]:
    """``'xyz'``, ``'x'``, ``'xz'`` ... -> Cartesian indices, in order, no repeats."""
    pols = tuple(POLARIZATIONS.index(ch) for ch in str(text).lower()
                 if ch in POLARIZATIONS)
    if not pols or len(set(pols)) != len(pols) or len(pols) != len(str(text)):
        raise ValueError(
            f"GATE haydock_polarizations: got {text!r}; want a non-empty subset "
            f"of {POLARIZATIONS!r} without repeats (e.g. 'x' for cubic crystals "
            f"or in-plane light, where the other two recursions repeat it)")
    return pols


def haydock_recursion_block(matvec_block, d_block, n_iter, constrain=lambda x: x):
    """Run a block Haydock recurrence (no reorthogonalisation).

    Parameters
    ----------
    matvec_block : (block, n_cond, n_val, nk) → (block, n_cond, n_val, nk)
        Applies the BSE Hamiltonian to a block of trial vectors.
    d_block : jax.Array, shape (block, n_cond, n_val, nk)
        Seed dipole vectors |d^α⟩, one per polarization (block = 1..3).
    n_iter : int
        Number of recurrence steps.
    constrain : the trial layout constraint (``sh.X``), applied to the carried
        Lanczos vectors so the loop body keeps them on the transition tiles:
        without it the propagator is free to carry them replicated, which costs
        a gather of every vector per step.

    Returns
    -------
    alphas : (block, n_iter) float — α_n = ⟨s_n|H|s_n⟩.
    betas  : (block, n_iter) float — β_n = ‖r_n‖ for r_n = (H−α_n)s_n − β_{n−1} s_{n−1}.
    norms  : (block,) float        — ‖d^α‖.
    """
    flat_shape = (d_block.shape[0], -1)
    d_flat = d_block.reshape(flat_shape)
    norms = jnp.linalg.norm(d_flat, axis=1)                                  # (block,)
    bcast = (-1,) + (1,) * (d_block.ndim - 1)
    s_curr = constrain(d_block / norms.reshape(bcast))
    s_prev = constrain(jnp.zeros_like(s_curr))

    def body(carry, _):
        s_curr, s_prev, b_curr = carry
        Hs = constrain(matvec_block(s_curr))
        a = jnp.einsum("a...,a...->a", s_curr.conj(), Hs).real               # (block,)
        a_b = a.reshape(bcast)
        b_b = b_curr.reshape(bcast)
        r = constrain(Hs - a_b * s_curr - b_b * s_prev)
        b_next = jnp.sqrt(jnp.einsum("a...,a...->a", r.conj(), r).real)    # (block,)
        s_next = constrain(r / b_next.reshape(bcast))
        return (s_next, s_curr, b_next), (a, b_next)

    init_b = jnp.zeros(d_block.shape[0], dtype=jnp.float64)
    (_, _, _), (alphas_T, betas_T) = lax.scan(
        body, (s_curr, s_prev, init_b), None, length=n_iter)
    # scan stacks along axis 0 → transpose to (block, n_iter)
    return alphas_T.T, betas_T.T, norms


def ensure_W_R(data, mesh_xy):
    """``data["W_R"] = ifftn_ortho(W_q)`` once per payload, with ``W_q`` DONATED.

    The same memoised donating transform ``bse_lanczos.solve_bse_sharded`` runs
    (``fft_helpers.get_donated_kfft_kminor``), so a payload the eigensolve has
    already transformed (``keep_W_R=True``) is reused as is, and a fresh one
    holds ONE (μ, ν, k) tile per rank instead of W_q and W_R together.
    """
    if data.get("W_R") is not None:
        return data["W_R"]
    sh = make_bse_shardings(mesh_xy)
    ifft = get_donated_kfft_kminor(
        mesh_xy, tuple(int(n) for n in data["W_q"].shape[-3:]), sh.W.spec,
        kind="ifftn", norm="ortho")
    data["W_R"] = ifft(data["W_q"])
    data["W_q"] = None
    return data["W_R"]


def haydock_program(mesh_xy, kgrid, n_iter):
    """The jitted recursion ``(d_seed, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c,
    eps_v, W_R, V_q0, M) -> (alphas, betas, norms)``: every device array an
    argument, the stack matvec inside one ``lax.scan`` of ``n_iter`` steps."""
    sh = make_bse_shardings(mesh_xy)
    matvec_stack = build_bse_stack_matvec(mesh_xy, *kgrid, kernel="bse")
    rep = NamedSharding(mesh_xy, P())

    @partial(
        jax.jit,
        in_shardings=(
            sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
            sh.eps, sh.eps, sh.W, sh.V, sh.M,
        ),
        out_shardings=(rep, rep, rep),
    )
    def _run(d_seed, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M):
        def matvec_block(X_block):
            return matvec_stack(
                jax.lax.with_sharding_constraint(X_block, sh.X),
                psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M)

        return haydock_recursion_block(
            matvec_block, d_seed, n_iter,
            constrain=lambda x: jax.lax.with_sharding_constraint(x, sh.X))

    return _run


def haydock_coefficients(data, mesh_xy, d_block_np, n_iter):
    """``(alphas, betas, norms)`` (host) of the Haydock recursion on a loaded payload.

    ``d_block_np`` is the ``(n_pol, n_cond_pad, n_val_pad, nk)`` seed block
    (``build_dipole_vector_bse`` rows of the requested polarizations).  It is
    placed straight onto the trial layout ``sh.X`` (no whole copy on one
    device), and every device array enters :func:`haydock_program` as an
    argument.  ``data["W_R"]`` is used when present (the eigensolve's), else
    built once by :func:`ensure_W_R`.
    """
    sh = make_bse_shardings(mesh_xy)
    kgrid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    W_R = ensure_W_R(data, mesh_xy)
    d_np = np.ascontiguousarray(d_block_np, dtype=np.complex128)
    d_seed = jax.make_array_from_callback(d_np.shape, sh.X, lambda idx: d_np[idx])
    alphas, betas, norms = haydock_program(mesh_xy, kgrid, int(n_iter))(
        d_seed, data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
        head_shifted_eps_c(data["eps_c"], data), data["eps_v"], W_R,
        data["V_q0"], data["M"])
    return np.asarray(alphas), np.asarray(betas), np.asarray(norms)


def write_haydock_outputs(*, out_prefix, pols, alphas, betas, norms, d_alpha,
                          de_cv, V_cell, nk, eta_eV, n_spin, n_spinor,
                          omega_min_eV, omega_max_eV, n_omega, no_eps1,
                          n_iter, metadata=None):
    """ε₂ (continued fraction), the IP JDOS, optionally ε₁ (Kramers-Kronig), and
    the per-polarization ``.dat`` files plus ``<out_prefix>.h5``; rank 0 writes.

    ``pols`` are the Cartesian indices of the recursions run (rows of
    ``alphas``); the arrays in the h5 carry one column per entry and the
    ``polarizations`` attribute names them.
    """
    omegas_eV = np.linspace(omega_min_eV, omega_max_eV, n_omega)
    omegas_Ry = omegas_eV / RYD2EV
    eta_Ry = eta_eV / RYD2EV
    eps2 = absorption_from_haydock(
        omegas_Ry, eta_Ry, alphas, betas, norms, V_cell, nk, n_spin, n_spinor)
    jdos = jdos_from_transitions(
        np.asarray(d_alpha)[list(pols)], de_cv, omegas_Ry, V_cell, nk, eta_Ry,
        n_spin, n_spinor)
    if no_eps1:
        eps1 = np.ones_like(eps2)
    else:
        eps1 = np.stack([kramers_kronig_eps1(omegas_Ry, eps2[:, a])
                         for a in range(eps2.shape[1])], axis=1)
    # ONE writer.  The (α, β) are replicated, so every rank holds the same
    # spectrum; at P>1 four ranks truncating the same h5 collided ("Unable to
    # synchronously create file (file signature not found)", rc 137 on
    # 2026-09-24's P4 gate run, before and after the stack-matvec move).
    if jax.process_index() != 0:
        return
    out_prefix_p = Path(out_prefix)
    for col, a in enumerate(pols):
        out_dat = out_prefix_p.with_name(f"absorption_haydock_b{a + 1}_eh.dat")
        header = [
            "BSE absorption — Haydock continued-fraction route",
            f"polarization: b{a + 1} ({POLARIZATIONS[a]})",
            f"V_cell = {V_cell:.6f} bohr^3,  N_k = {nk}",
            f"eta = {eta_eV:.4f} eV,  n_spin = {n_spin},  n_spinor = {n_spinor}",
            f"n_iter = {n_iter},  ‖d‖ = {norms[col]:.6e}",
        ]
        write_absorption_dat(out_dat, omegas_eV, eps2[:, col], eps1[:, col],
                             jdos[:, col], header)
        print(f"[haydock] wrote {out_dat}")
    out_h5 = out_prefix_p.with_name(f"{out_prefix_p.name}.h5")
    write_absorption_h5(
        out_h5,
        omegas_eV=omegas_eV,
        eps2_3pol=eps2, eps1_3pol=eps1, jdos_3pol=jdos,
        haydock_alphas=alphas, haydock_betas=betas, haydock_norms=norms,
        metadata={
            "route": "haydock",
            "V_cell_bohr3": V_cell,
            "N_k": nk,
            "eta_eV": eta_eV,
            "n_spin": n_spin,
            "n_spinor": n_spinor,
            "n_iter": n_iter,
            "polarizations": "".join(POLARIZATIONS[a] for a in pols),
            **(metadata or {}),
        },
    )
    print(f"[haydock] wrote {out_h5}")


def absorption_from_haydock(
    omegas_Ry, eta_Ry, alphas, betas, norms,
    V_cell, n_k, n_spin, n_spinor,
):
    """Build ε₂(ω) from the Haydock (α_n, β_n) coefficients.

    One independent continued fraction per row of ``alphas`` (polarisation).
    """
    alphas = np.asarray(alphas)                                              # (n_pol, n_iter)
    betas = np.asarray(betas)
    norms = np.asarray(norms)
    z = omegas_Ry + 1j * eta_Ry
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    n_pol, n_iter = alphas.shape
    eps2 = np.zeros((omegas_Ry.size, n_pol), dtype=np.float64)
    for a in range(n_pol):
        cf = np.zeros_like(z)
        for n in range(n_iter - 1, 0, -1):
            cf = (betas[a, n - 1] ** 2) / (z - alphas[a, n] - cf)
        g = 1.0 / (z - alphas[a, 0] - cf)
        eps2[:, a] = -pref * (norms[a] ** 2) * g.imag / np.pi
    return eps2


def run_haydock(
    *,
    input_file: str,
    n_val: int,
    n_cond: int,
    n_occ: int,
    eqp_file: str | None,
    dipole_file: str,
    dipole_position_fix: bool = False,
    V_cell: float,
    n_iter: int,
    eta_eV: float,
    n_spin: int,
    n_spinor: int,
    omega_min_eV: float,
    omega_max_eV: float,
    n_omega: int,
    out_prefix: str,
    no_eps1: bool,
    pols: tuple[int, ...] = (0, 1, 2),
):
    """End-to-end Haydock absorption: load BSE data, run the block recursion over
    the requested polarizations, evaluate the continued fraction, write outputs.
    ``bse.bse_jax --haydock-iter`` runs the same recursion on the eigensolve's
    payload in the same process."""
    restart_file = _find_restart_file(input_file)
    n_devices = jax.device_count()
    if n_devices < 2:
        raise RuntimeError(
            "Haydock route currently requires the sharded matvec (≥2 devices).")
    mesh_xy = create_mesh_2d()
    grid_x, grid_y = mesh_xy.devices.shape

    print(f"[haydock] loading BSE data from {restart_file}")
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
        input_file=input_file, n_occ=n_occ, eqp_file=eqp_file,
        velocity_file=dipole_file,
        velocity_position_fix=dipole_position_fix,
        q0_head="diagonal",           # operands via bse_head.head_shifted_eps_c
    )
    # Resolve n_occ for the downstream dipole slice — same source as the
    # loader used (WFN.h5 ifmax → largest-gap fallback).
    from file_io.restart_bundle import read_metadata
    enk_full_np = read_metadata(restart_file)["energies"]
    from .bse_io import resolve_n_occ
    n_occ = resolve_n_occ(enk_full_np, n_occ=n_occ, input_file=input_file)
    if eqp_file is not None and not data["densified"]:
        # (A bse_k_grid bundle took the QP energies inside its htransform.)
        # Re-slice the band window AND re-resolve n_occ on the eqp-corrected
        # energies (a QP shift can move the gap). Overwrite n_occ so the
        # downstream dipole slice below uses the corrected value too, rather
        # than the stale pre-correction n_occ resolved above.
        from .bse_io import apply_eqp_and_reslice_bands
        data["eps_v"], data["eps_c"], n_occ = apply_eqp_and_reslice_bands(
            restart_file, eqp_file, input_file,
            int(data["n_val"]), int(data["n_cond"]), n_occ, grid_x, grid_y)

    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    # The window the loader RESOLVED — clamped to the file and then widened by
    # ``--band-degeneracy`` at a cut multiplet.  The dipole slice below has to
    # agree with the Hamiltonian's window band-for-band, so it reads these and
    # not the ``n_val``/``n_cond`` REQUEST: on a snapping deck the request cuts
    # the dipole over a narrower, differently-placed window and
    # ``build_dipole_vector_bse`` then zero-pads the slots the BSE window is
    # really using.  That is silently wrong absorption, not a crash — the same
    # pre-snap assumption ``bse_jax``'s ``--write-eigs`` path made loudly.
    n_val_eff = int(data["n_val"])
    n_cond_eff = int(data["n_cond"])
    print(f"[haydock] mesh {grid_x}x{grid_y}, problem "
          f"{nc_pad} cond × {nv_pad} val × {nk} k = {nc_pad * nv_pad * nk}")

    d_alpha, de_cv = window_dipoles(data, dipole_file, n_occ=n_occ,
                                    wfn_path=_parse_wfn_path(input_file))
    print(f"[haydock] dipole sliced → (3, nk={nk}, nc={n_cond_eff}, "
          f"nv={n_val_eff})")
    d_block_np = build_dipole_vector_bse(d_alpha, n_cond_pad=nc_pad, n_val_pad=nv_pad)

    print(f"[haydock] running {len(pols)}-block recurrence × {n_iter} iters "
          f"(polarizations {''.join(POLARIZATIONS[a] for a in pols)}) …")
    alphas, betas, norms = haydock_coefficients(
        data, mesh_xy, d_block_np[list(pols)], n_iter)
    print(f"[haydock] α_1 = {alphas[:, 0]}  (Ry)")
    print(f"[haydock] β_1 = {betas[:, 0]}  (Ry)")
    print(f"[haydock] ‖d‖ = {norms}")

    write_haydock_outputs(
        out_prefix=out_prefix, pols=pols, alphas=alphas, betas=betas,
        norms=norms, d_alpha=d_alpha, de_cv=de_cv, V_cell=V_cell, nk=nk,
        eta_eV=eta_eV, n_spin=n_spin, n_spinor=n_spinor,
        omega_min_eV=omega_min_eV, omega_max_eV=omega_max_eV, n_omega=n_omega,
        no_eps1=no_eps1, n_iter=n_iter,
        metadata=({"dipole_route": data["dipoles"]["route"],
                   "dipole_source": data["dipoles"]["source"],
                   "dipole_on_grid_rel": data["dipoles"]["on_grid_rel"]}
                  if data["densified"] else None))


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False,
        description="Haydock-route BSE absorption (continued fraction; no eigenvectors)",
    )
    p.add_argument("-i", "--input", required=True,
                   help="cohsex.in (BSE input; restart bundle resolved from it)")
    p.add_argument("--n-val", type=int, required=True)
    p.add_argument("--n-cond", type=int, required=True)
    p.add_argument("--n-occ", type=int, required=True)
    p.add_argument("--eqp", default=None,
                   help="BGW eqp1.dat for QP corrections (optional)")
    p.add_argument("--dipole", default="dipole.h5",
                   help="dipole.h5 from psp.get_dipole_mtxels (default: "
                        "dipole.h5); under bse_k_grid the deck (coarse) WFN's, "
                        "carried through the htransform into the fine states")
    p.add_argument("--dipole-position-fix", action="store_true",
                   help="bse_k_grid with --eqp: exact <c|r|v> of the fine QP "
                        "states instead of v/dE_RQ (bse_jax --dipole-position-fix)")
    p.add_argument("--V-cell", type=float, required=True,
                   help="Unit-cell volume (bohr^3)")
    p.add_argument("--n-iter", type=int, default=200,
                   help="Haydock iterations per polarisation (default 200)")
    p.add_argument("--n-spin", type=int, default=1,
                   help="Spin multiplicity (2 only for collinear spin-polarised)")
    p.add_argument("--n-spinor", type=int, default=2,
                   help="Spinor components (2 for spinor calcs, default 2)")
    p.add_argument("--eta-eV", type=float, default=0.1)
    p.add_argument("--omega-min-eV", type=float, default=0.0)
    p.add_argument("--omega-max-eV", type=float, default=15.0)
    p.add_argument("--n-omega", type=int, default=1500)
    p.add_argument("--out-prefix", default="absorption_haydock")
    p.add_argument("--no-eps1", action="store_true",
                   help="Skip Kramers-Kronig eps1 (faster)")
    p.add_argument("--pol", default=POLARIZATIONS,
                   help="Cartesian polarizations to recurse, a subset of 'xyz' "
                        "(default all three).  Each costs one BSE matvec per "
                        "step: 'x' alone is the whole spectrum of a cubic "
                        "crystal and of in-plane light on a hexagonal one")
    args = p.parse_args(argv)
    try:
        pols = parse_polarizations(args.pol)
    except ValueError as exc:
        p.error(str(exc))

    run_haydock(
        input_file=args.input,
        n_val=args.n_val, n_cond=args.n_cond, n_occ=args.n_occ,
        eqp_file=args.eqp, dipole_file=args.dipole,
        dipole_position_fix=args.dipole_position_fix,
        V_cell=args.V_cell, n_iter=args.n_iter,
        eta_eV=args.eta_eV, n_spin=args.n_spin, n_spinor=args.n_spinor,
        omega_min_eV=args.omega_min_eV, omega_max_eV=args.omega_max_eV,
        n_omega=args.n_omega,
        out_prefix=args.out_prefix, no_eps1=args.no_eps1, pols=pols,
    )


if __name__ == "__main__":
    raise SystemExit(main())
