"""Full microscopic native shared-pole MGKBA snapshot runtime.

The primary joint modes retain every centroid, q and virtual band. Archived
projected helpers below remain explicit prototype diagnostics, not a fallback
for propagation. Energies use Ry until the Ha propagation boundary.
"""
from __future__ import annotations

import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from tdagw import collision as C
from tdagw import keldysh_rational as kr
from tdagw import propagate as tp
from tdagw import snapshot_screening as SS
from tdagw.dynsig import (collision_basis_state, collision_snapshot_state,
                         slow_basis_step, _track_bands)
from tdagw.self_energy import held_term, with_held_term


def prepare_native_source(source, mesh):
    """Authenticate a full-grid native orbital/Coulomb source without projection.

    The returned ``psi`` is the source's existing centroid face, not a
    density-channel vertex tensor. Large orbital and bare-Coulomb arrays
    remain distributed on all ranks. Only scalar metadata and the small
    full-band energy/occupation tables are gathered.
    """
    from common.collectives import gather_to_host
    from symmetry_maps import QirrOperator
    wfns, meta = source["wfns"], source["meta"]
    carrier = wfns.green_parent
    if carrier is None:
        raise ValueError("GATE td_native_parent_carrier: explicit full-grid parent faces required")
    kgrid = tuple(int(x) for x in meta.kgrid)
    nk = int(np.prod(kgrid))
    rows = carrier.plan
    frac = np.asarray(rows.k_parent_frac, float)
    from gw.centroid_k_unfold import authenticate_canonical_full_grid
    qsym = source.get("screening_inputs", {}).get("kwargs", {}).get("sym", getattr(rows, "sym", None))
    flat_grid = authenticate_canonical_full_grid(rows, kgrid, sym=qsym)
    if (frac.shape != (nk, 3) or not np.array_equal(np.asarray(rows.irr_idx), np.arange(nk))
            or (rows.parent_full_rows is not None and
                not np.array_equal(np.asarray(rows.parent_full_rows), np.arange(nk)))):
        raise ValueError("GATE td_native_full_grid: full identity parent k order required")
    if int(wfns.slices.b0) != 0:
        raise ValueError("GATE td_native_full_bands: electronic support must start at zero")
    logical = int(wfns.slices.b4_logical)
    nb_chi = min(int(wfns.slices.b4_chi), logical)
    nb_sigma = min(int(wfns.slices.b4_sigma or nb_chi), logical)
    nb = max(nb_chi, nb_sigma)
    V = source.get("V_q", source.get("screening_inputs", {}).get("V_q"))
    if V is None:
        raise ValueError("GATE td_native_bare_coulomb: source must retain its full bare V")
    interaction = QirrOperator.of(V)
    if (not interaction.is_whole_zone() or interaction.n_full != nk or
            not np.array_equal(np.asarray(interaction.full_rows), np.arange(nk))):
        raise ValueError("GATE td_native_coulomb_grid: full identity q carrier required")
    psi = carrier.projection_faces()[0]
    V = interaction.values
    if not isinstance(V, jax.Array) or V.ndim != 3 or V.shape[-1] != psi.shape[-1]:
        raise ValueError("GATE td_native_centroids: distributed full-mu Coulomb/orbital faces required")
    nspin = int(getattr(meta, "nspin", 1))
    nspinor = int(getattr(meta, "nspinor_wfnfile", psi.shape[2]))
    if min(nspin, nspinor) < 1:
        raise ValueError("GATE td_native_spin_weight: positive file spin multiplicities required")
    return dict(energy=np.asarray(gather_to_host(wfns.enk))[:, :nb],
        occupations=np.asarray(gather_to_host(wfns.occ))[:, :nb], k_frac=frac,
        kgrid=kgrid, nk=nk, nb=nb, nb_chi=nb_chi, nb_sigma=nb_sigma,
        mesh=mesh, nmu_logical=int(meta.n_rmu), spin_weight=2.0/(nspin*nspinor),
        psi=psi, V_q=V, head=source.get("head"),
        efermi_ry=float(source.get("efermi_ry") or 0.0), native_full_microscopic=True,
        flat_grid=flat_grid)


def make_native_snapshot_evaluator(source, prepared, options):
    """Bind full-mu current screening to the production native Sigma window.

    Every iterate receives the same explicit block-embedded instantaneous
    state in response and Sigma. ``joint_frozen`` reuses the equilibrium W
    handle exactly; electronic lines and the full virtual energies still
    update. Only active-band matrix outputs leave the devices.
    """
    from gw.shared_pole_screening import screen_general_shared_poles
    from gw.ppm_sigma import shared_pole_window_source, window_sigma_c, window_sigma_plan
    from gw.shared_pole_recipe import parse_support_sites
    from common.units import RYD_TO_EV
    from dataclasses import replace
    import os
    start, na = int(options["b_lo"]), int(options["n_active"])
    active = slice(start, start+na)
    inp = source.get("screening_inputs")
    if inp is None:
        raise ValueError("GATE td_native_screening_source: source must retain native shared-pole W inputs")
    mode = options.get("screening", "joint")
    if mode not in ("joint", "frozen"):
        raise ValueError("GATE td_native_screening: expected joint or frozen")
    scratch = options.get("spm_scratch")
    if not scratch:
        raise ValueError("GATE td_native_screening_scratch: unique model output directory required")
    cache = dict(n=0)
    plan = window_sigma_plan(source["config"], head=source.get("head"), window_only=True)
    screening_config = inp["config"]
    supports = parse_support_sites(options.get("spm_support_sites", ""))
    if supports is not None:
        screening_config = replace(screening_config,
            sigma=replace(screening_config.sigma, w_support_sites_ev=supports["text"]))

    def evaluate(basis, *, equilibrium=False, fit_bases=None):
        wall = time.perf_counter()
        U, e, rho = (np.asarray(basis[k]) for k in ("U", "e", "rho"))
        energy = prepared["energy"].copy()
        energy[:, :start] += (e[:, 0]-energy[:, start])[:, None]
        energy[:, start+na:] += (e[:, -1]-energy[:, start+na-1])[:, None]
        energy[:, active] = e
        cache["n"] += 1
        if mode == "joint" or equilibrium:
            snapshot = dict(U_active=U, E_active_ry=e, rho_active=rho,
                active_slice=active, energy_ry=energy,
                gamma_ry=float(options.get("adiabatic_ry",
                    float(options.get("adiabatic_ev", 0.05))/RYD_TO_EV)),
                eta_ry=float(source["config"].sigma.regularization_ev)/RYD_TO_EV,
                spm_support_sites_ev=options.get("spm_support_sites", ""),
                support_reads_ev=source.get("recipe_inputs", {}).get("support_reads_ev"),
                fit_tolerance=float(options.get("fit_tolerance", 0.02)))
            # The native builder owns support/direction/K admission. No
            # projected rank, prototype pole order or sample count enters.
            built = screen_general_shared_poles(inp["wfns"], inp["V_q"], inp["meta"], screening_config,
                mesh_xy=prepared["mesh"], sym=inp["kwargs"]["sym"], snapshot=snapshot,
                run_dir=os.path.abspath(scratch), label=f"snapshot_{cache['n']:06d}",
                fit_basis=fit_bases, print_fn=options.get("say", print))
            handle = shared_pole_window_source(built["shared_pole"], built["meta"],
                efermi_ry=prepared["efermi_ry"], linalg=source.get("linalg"))
            screening_receipt = built["receipt"]
            if equilibrium:
                cache.update(equilibrium=handle, equilibrium_receipt=screening_receipt)
        else:
            if "equilibrium" not in cache:
                raise ValueError("GATE td_native_equilibrium: initialize equilibrium W before frozen control")
            handle, screening_receipt = cache["equilibrium"], cache["equilibrium_receipt"]
        window = dict(b_lo=start, U=U, e=e, rho=rho,
                      f=np.diagonal(rho, axis1=-2, axis2=-1).real, e_frozen=energy)
        out = window_sigma_c(source["wfns"], handle, window, plan, meta=source["meta"],
            mesh_xy=prepared["mesh"], branches=True, pairing="vertex", static_limit=False,
            print_fn=options.get("say", print))
        active_pair, full_pair = out["active_branch_pair"], out["branch_pair"]
        gamma, injection, _ = C.branch_operators(active_pair["cond"], active_pair["val"])
        _, _, mh = C.branch_operators(full_pair["cond"], full_pair["val"])
        Uh = U.conj().swapaxes(-1, -2)
        totals = {key: U @ np.asarray(value) @ Uh for key, value in
                  (("gamma", gamma), ("injection", injection), ("m_h", mh))}
        totals["static"] = np.zeros_like(totals["m_h"])
        totals.update(screening_receipts=screening_receipt,
            native_receipt=out["native_receipt"], raw=out,
            wall_s=time.perf_counter()-wall)
        return totals

    return evaluate, cache


def prepare_projected_source(source, mesh, *, rank=8):
    """Prepare distributed Coulomb factors and a streaming q vertex factory.

    ``source`` is ``dynsig.gw_window_source(screening=True)`` on a full
    unreduced grid. ``rank`` bounds the Coulomb coordinates, not the band
    sum. Returned ``vertices(q)`` yields ``T[k,b,d,r]`` and a projection
    receipt. Psi has shape ``[k,band,spin,mu]`` and is sharded on mu over
    both mesh axes. V remains ``[q,mu,nu]`` on ``P(None,'x','y')``.
    No all-q vertex cache or host centroid face is constructed.
    """
    from common.collectives import device_put_process_local, gather_to_host
    from tdagw.model import flat_k_rows

    wfns, meta = source["wfns"], source["meta"]
    carrier = wfns.green_parent
    if carrier is None:
        raise ValueError("GATE td_snapshot_parent_carrier: explicit full-grid parent carrier required")
    faces = carrier.projection_faces()[0] if hasattr(carrier, "projection_faces") else carrier.psi_nmu
    plan = carrier.plan
    kgrid = tuple(int(x) for x in meta.kgrid)
    nk = int(np.prod(kgrid))
    k_frac = np.asarray(plan.k_parent_frac, float)
    irr = np.asarray(getattr(plan, "irr_idx", np.arange(nk)))
    if k_frac.shape != (nk, 3) or not np.array_equal(irr, np.arange(nk)):
        raise ValueError("GATE td_snapshot_full_grid: full unreduced parent k grid required")
    parent_rows = getattr(plan, "parent_full_rows", None)
    if parent_rows is not None and not np.array_equal(parent_rows, np.arange(nk)):
        raise ValueError("GATE td_snapshot_parent_rows: parent carrier must retain bundle row order")
    if int(wfns.slices.b0) != 0:
        raise ValueError("GATE td_snapshot_full_bands: electronic support must start at band zero")
    logical = getattr(wfns.slices, "b4_logical", None)
    nb_chi = int(wfns.slices.b4_chi)
    nb_sigma = int(getattr(wfns.slices, "b4_sigma", 0) or nb_chi)
    logical = int(logical or max(nb_chi, nb_sigma))
    nb_chi, nb_sigma = min(nb_chi, logical), min(nb_sigma, logical)
    nb = max(nb_chi, nb_sigma)
    V = source.get("V_q")
    if V is None:
        V = source.get("screening_inputs", {}).get("V_q")
    if V is None:
        raise ValueError("GATE td_snapshot_bare_coulomb: source has no bare Coulomb operator")
    from symmetry_maps import QirrOperator
    interaction = QirrOperator.of(V)
    if (not interaction.is_whole_zone() or interaction.n_full != nk
            or not np.array_equal(np.asarray(interaction.full_rows), np.arange(nk))):
        raise ValueError("GATE td_snapshot_coulomb_full_grid: whole-zone identity q carrier required")
    V = interaction.values
    if not isinstance(V, jax.Array) or V.ndim != 3 or V.shape[0] != nk:
        raise ValueError("GATE td_snapshot_bare_coulomb: distributed full-q Coulomb array required")
    nmu = int(V.shape[-1])
    if V.shape[-2] != nmu or faces.shape[-1] != nmu:
        raise ValueError("GATE td_snapshot_centroids: Coulomb and orbital centroid axes differ")
    rank = min(int(rank), int(getattr(meta, "n_rmu", nmu)))
    if rank < 1:
        raise ValueError("GATE td_snapshot_density_rank: positive Coulomb rank required")
    axes = ("x", "y")
    ndev = int(mesh.size)
    nmu_pad = ((nmu + ndev - 1) // ndev) * ndev
    psi_sh = NamedSharding(mesh, P(None, None, None, axes))
    v_sh = NamedSharding(mesh, P(None, "x", "y"))
    factor_sh = NamedSharding(mesh, P(axes, None))
    rep = NamedSharding(mesh, P())
    psi = jax.jit(lambda p: jnp.pad(p[:, :nb],
        ((0, 0), (0, 0), (0, 0), (0, nmu_pad - nmu))), out_shardings=psi_sh)(faces)
    V = jax.jit(lambda v: jnp.pad(v,
        ((0, 0), (0, nmu_pad - nmu), (0, nmu_pad - nmu))), out_shardings=v_sh)(V)
    rng = np.random.default_rng(1729)
    # One real sketch for every q preserves conjugate q/-q pairing.  Use
    # the principal inverse square root below rather than eigenvector
    # columns, whose arbitrary phases would obscure that relation.
    omega_h = rng.normal(size=(nmu_pad, rank)).astype(np.complex128)
    omega_h[nmu:] = 0.0
    omega = device_put_process_local(omega_h, NamedSharding(mesh, P("y", None)))

    def factor(v, sketch, iq, iq_partner, conjugate):
        tile = jax.lax.dynamic_index_in_dim(v, iq, axis=0, keepdims=False)
        y = tile @ sketch
        partner_tile = jax.lax.dynamic_index_in_dim(v, iq_partner, axis=0, keepdims=False)
        partner_y = partner_tile @ sketch
        pairing_error = jnp.linalg.norm(partner_y - jnp.conj(y)) / jnp.maximum(
            jnp.linalg.norm(partner_y), 1e-300)
        gram = jnp.conj(sketch.T) @ y
        gram = 0.5 * (gram + jnp.conj(gram.T))
        values, vectors = jnp.linalg.eigh(gram)
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1e-300)
        keep = values > 1e-12 * scale
        inverse = jnp.where(keep, 1.0 / jnp.sqrt(jnp.maximum(values, 1e-300)), 0.0)
        invroot = (vectors * inverse[None]) @ jnp.conj(vectors.T)
        f = y @ invroot
        f = jnp.where(conjugate, jnp.conj(f), f)
        total = jnp.real(jnp.trace(tile))
        captured = jnp.sum(jnp.abs(f) ** 2)
        return f, jnp.stack((jnp.min(values) / scale, jnp.sum(keep),
                            (total - captured) / jnp.maximum(jnp.abs(total), 1e-300), pairing_error))

    factor_fn = jax.jit(factor, out_shardings=(factor_sh, rep))

    def project(p, f, partner):
        # A k scan bounds the three-operand contraction's orbital/factor
        # intermediate. Vectorizing all k would keep nk copies alive.
        zero = jnp.zeros((p.shape[0], p.shape[1], p.shape[1], f.shape[1]), p.dtype)

        def one_k(local, k):
            block = jnp.einsum("bsu,dsu,ur->bdr", p[k], jnp.conj(p[partner[k]]),
                               jnp.conj(f), optimize=True)
            return local.at[k].set(block), None

        local = jax.lax.scan(one_k, zero, jnp.arange(p.shape[0]))[0]
        return jax.lax.psum(local, axes)

    project_fn = jax.jit(jax.shard_map(project, mesh=mesh,
        in_specs=(P(None, None, None, axes), P(axes, None), P()),
        out_specs=P(), check_vma=False))
    flat = np.asarray(flat_k_rows(k_frac, kgrid), int)
    inverse_flat = np.argsort(flat)
    ijk = np.stack(np.unravel_index(flat, kgrid), axis=1)
    qijk = np.stack(np.unravel_index(np.arange(nk), kgrid), axis=1)
    q_negative = np.ravel_multi_index(tuple(((-qijk) % kgrid).T), kgrid)
    kmq = np.stack([inverse_flat[np.ravel_multi_index(tuple(((ijk - q) % kgrid).T), kgrid)]
                    for q in qijk])

    def vertices(iq):
        representative = min(int(iq), int(q_negative[iq]))
        f, stats = factor_fn(V, omega, jnp.asarray(representative, jnp.int32),
            jnp.asarray(q_negative[representative], jnp.int32), jnp.asarray(iq != representative))
        stats = np.asarray(gather_to_host(stats))
        if not np.isfinite(stats).all() or stats[0] < -1e-10:
            raise ValueError("GATE td_snapshot_coulomb_psd: sketch has nonfinite/negative metric")
        if stats[3] > 1e-10:
            raise ValueError(f"GATE td_snapshot_coulomb_reciprocity: q/-q projected bare "
                             f"interaction mismatch {stats[3]:.3g}")
        partner = device_put_process_local(kmq[iq], rep)
        T = np.asarray(gather_to_host(project_fn(psi, f, partner)))
        return T, dict(q=int(iq), requested_rank=rank, effective_rank=int(stats[1]),
                       omitted_coulomb_trace_fraction=float(stats[2]),
                       coulomb_pairing_error=float(stats[3]))

    nspin = int(getattr(meta, "nspin", 1))
    nspinor_file = int(getattr(meta, "nspinor_wfnfile", psi.shape[2]))
    if nspin < 1 or nspinor_file < 1:
        raise ValueError("GATE td_snapshot_spin_weight: positive file spin multiplicities required")
    return dict(vertices=vertices, kmq=kmq, q_negative=q_negative,
                energy=np.asarray(gather_to_host(wfns.enk))[:, :nb],
                occupations=np.asarray(gather_to_host(wfns.occ))[:, :nb],
                k_frac=k_frac, kgrid=kgrid, nk=nk, nb=nb, nb_chi=nb_chi,
                nb_sigma=nb_sigma, rank=rank, mesh=mesh, nmu_logical=int(getattr(meta, "n_rmu", nmu)),
                spin_weight=2.0 / (nspin * nspinor_file),
                psi=psi, V_q=V, omega=omega, head=source.get("head"),
                efermi_ry=float(source.get("efermi_ry") or 0.0))


def rotate_pair_vertices(vertices, U, partner, active):
    """Rotate both charge-pair legs, retaining every frozen electronic band.

    ``vertices[k,b,d,r]`` is one q block; ``U[k,a,a]`` rotates the active
    block only, with ``psi' = psi U``.  ``partner[k]`` selects k-q.
    """
    out = np.array(vertices, copy=True)
    out[:, active] = np.einsum("kab,kacr->kbcr", U, out[:, active], optimize=True)
    out[:, :, active] = np.einsum("kdcr,kce->kder", out[:, :, active],
                                 U[partner].conj(), optimize=True)
    return out


def authenticate_orbital_gauge(prepared, model, active):
    """Check the actual sampled active orbitals against an ISDF mean-field model.

    The k ordering and centroid padding can differ; the physical orbital
    gauge cannot.  The scalar norm is evaluated collectively from device
    operands. Generic models without ISDF orbital operands are identified
    explicitly in the returned receipt.
    """
    from tdagw.model import flat_k_rows
    from tdagw.self_energy import _isdf_apply
    if getattr(model.apply, "func", None) is not _isdf_apply:
        return dict(checked=False, reason="model has no ISDF orbital operands")
    model_psi = model.params[0][0]
    nmu, nk = prepared["nmu_logical"], prepared["nk"]
    na = int(active.stop) - int(active.start)
    if (model_psi.shape[0] != nk or model_psi.shape[1] < na or model_psi.shape[-1] < nmu
            or model_psi.shape[2] != prepared["psi"].shape[2]):
        raise ValueError("GATE td_snapshot_orbital_gauge: source and ISDF model support differ")
    rows = np.asarray(flat_k_rows(prepared["k_frac"], prepared["kgrid"]), int)
    flat_order = np.argsort(rows)

    def gauge_error(source_psi, reference_psi):
        a = source_psi[jnp.asarray(flat_order), active, :, :nmu]
        b = reference_psi[:, :na, :, :nmu]
        return jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-300)

    error = float(jax.jit(gauge_error)(prepared["psi"], model_psi))
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError(f"GATE td_snapshot_orbital_gauge: relative source/model orbital "
                         f"difference {error:.3g}; identical sampled orbital gauge required")
    return dict(checked=True, relative_error=error, tolerance=1e-10)


def _hf_coherent_apply(inner, params, rho, t):
    """Bare HF increment plus the chosen constant one-body baseline, Ha."""
    inner_params, baseline, rho_eq, bare_head = params
    return inner(inner_params, rho, t) + baseline - bare_head * (rho - rho_eq[None])


def bare_hf_coherent_model(model, prepared, rho_eq, baseline, *, bare_head_ha=0.0,
                           explicit_hf_model=None):
    """Replace screened exchange by bare V using the existing TD kernel.

    ``model`` is the authenticated full ISDF TD model. ``baseline`` is
    ``(ntr,nk_pad,na,na)`` Ha in propagator k sharding. Bare Fock uses
    source V on the same centroid gauge and the existing routed inverse
    k FFT. The supplied reference W0 is never subtracted from reduced Wc0.
    An explicitly supplied ``explicit_hf_model`` is for independent plain
    array models whose bare-HF contract is owned by their caller.
    """
    from tdagw.self_energy import SelfEnergyModel, _isdf_apply
    from tdagw.dynsig import make_dynsig_kernels, _w_R, WR_SPEC
    if explicit_hf_model is not None:
        bare = explicit_hf_model
    else:
        if getattr(model.apply, "func", None) is not _isdf_apply:
            raise ValueError("GATE td_snapshot_bare_hf_model: full ISDF model or explicit bare-HF model required")
        static = model.apply.args[0]
        if static[-1] is not None:
            raise ValueError("GATE td_snapshot_bare_hf_model: TDA projection cannot carry full-density HF")
        if int(prepared["psi"].shape[2]) > 2:
            raise ValueError("GATE td_snapshot_bare_hf_model: transverse four-current HF increment not implemented")
        from bse.bse_preconditioner import exchange_spin_weight
        if prepared["spin_weight"] != exchange_spin_weight(int(static[2])):
            raise ValueError("GATE td_snapshot_bare_hf_model: TD Hartree spin multiplicity differs from GW source")
        operands, reference = model.params
        extent = int(operands[3].shape[0])
        V = prepared["V_q"]
        if extent > V.shape[-1]:
            extra = extent - int(V.shape[-1])
            V = jax.jit(lambda v: jnp.pad(v, ((0, 0), (0, extra), (0, extra))))(V)
        V = V[:, :extent, :extent]
        kern = make_dynsig_kernels(prepared["mesh"], prepared["kgrid"])
        bare_R = jax.jit(lambda v: _w_R(v, kern),
            out_shardings=NamedSharding(prepared["mesh"], WR_SPEC))(V)
        bare_operands = list(operands)
        bare_operands[3] = bare_R
        bare_static = (*static[:7], True, True, None)
        bare = SelfEnergyModel(apply=functools.partial(_isdf_apply, bare_static),
            params=(tuple(bare_operands), reference), linear=True, name="ISDF bare Hartree-Fock increment")
    return SelfEnergyModel(apply=functools.partial(_hf_coherent_apply, bare.apply),
        params=(bare.params, baseline, rho_eq, float(bare_head_ha)), linear=bare.linear,
        name=f"{bare.name} + constant one-body baseline")


def hf_reference_window(source, prepared, active):
    """Authenticated hHF[rho_eq] in source/bundle k order, Ry."""
    from common.collectives import gather_to_host
    reference = source.get("hf_reference")
    if reference is None or reference.get("basis") != "source_wfn":
        raise ValueError("GATE td_snapshot_hf_reference: rerun GW source with HF capture seam")
    from gw.sigma_dispatch import SIGMA_KSET_FULL_BZ
    if reference.get("kset") != SIGMA_KSET_FULL_BZ:
        raise ValueError("GATE td_snapshot_hf_reference: HF matrices must cover the full BZ")
    lo, hi = int(reference["band_start"]), int(reference["band_stop"])
    if active.start < lo or active.stop > hi:
        raise ValueError("GATE td_snapshot_hf_reference: authenticated HF matrices do not cover active window")
    win = slice(active.start - lo, active.stop - lo)
    parts = [np.asarray(gather_to_host(reference[key]))[:, win, win]
             for key in ("kin_ion_ry", "hartree_ry", "exchange_ry")]
    shape = (prepared["nk"], active.stop - active.start, active.stop - active.start)
    if any(part.shape != shape or not np.isfinite(part).all() for part in parts):
        raise ValueError("GATE td_snapshot_hf_reference: inconsistent/nonfinite one-body/Hartree/Fock blocks")
    H = sum(parts)
    if np.max(abs(H - H.conj().swapaxes(-1, -2))) > 1e-9:
        raise ValueError("GATE td_snapshot_hf_reference: non-Hermitian HF baseline")
    return H, float(reference.get("bare_head_coefficient_ry", 0.0))


def snapshot_supports(energy, options):
    """Resolve explicit disjoint upper-half-plane fit/check supports, Ry."""
    if "fit_sites" in options and "check_sites" in options:
        return np.asarray(options["fit_sites"], complex), np.asarray(options["check_sites"], complex)
    span = max(float(np.ptp(energy)), float(options.get("support_span_ry", 0.0)), 1e-3)
    count = int(options.get("support_count", 192))
    height = float(options.get("support_height_ry", max(0.02 * span, 1e-3)))
    low = min(span * 1e-4, 0.01 / 13.605693122994)
    positive = np.geomspace(max(low, 1e-9), 1.1 * span, max(2, count // 2))
    fit = np.concatenate((-positive[::-1], positive)) + 1j * height
    probe = np.geomspace(max(1.41 * low, 1e-9), 1.07 * span, max(4, count // 8))
    check = np.concatenate((-probe[::-1], probe)) + 1.37j * height
    return fit, check


def make_snapshot_evaluator(prepared, options):
    """Make an all-q projected GW evaluator and its equilibrium W cache.

    Returned ``evaluate(basis, equilibrium=False)`` uses bundle-order
    ``basis={U,e,rho}``, with active eigenvalues in Ry and full active rho
    in that eigenbasis.  It embeds frozen 0/1 density and edge-scissored
    energies, streams q vertices, jointly screens W unless frozen, and
    returns full-matrix active tables in the fixed band basis, in Ry.
    The cache stores reduced W realizations and rank-square static Wc0,
    never all-q pair vertices or full centroid matrices.
    """
    from tdagw.snapshot_collision import q_sigma_operators
    from common.collectives import device_put_process_local, gather_to_host

    start, na = int(options["b_lo"]), int(options["n_active"])
    active = slice(start, start + na)
    nk = prepared["nk"]
    energy0, occupations = prepared["energy"], prepared["occupations"]
    frozen_energy = np.asarray(options.get("frozen_energy", energy0))
    fit_sites, check_sites = snapshot_supports(frozen_energy, options)
    eta = float(options.get("eta_ry", 0.01))
    gamma = float(options.get("adiabatic_ry", 0.01))
    mode = options.get("screening", "joint")
    if mode not in ("joint", "frozen"):
        raise ValueError("GATE td_snapshot_screening: expected joint or frozen")
    cache = {}
    mesh = prepared["mesh"]
    rep = NamedSharding(mesh, P())
    vertex_sh = NamedSharding(mesh, P(("x", "y"), None, None, None))
    sample_projected = SS.make_projected_sampler(mesh, band_block=int(options.get("band_block", 32)))
    n_chi, n_sigma = prepared["nb_chi"], prepared["nb_sigma"]
    nk_pad = ((nk + mesh.size - 1) // mesh.size) * mesh.size
    replicated = lambda value: device_put_process_local(np.asarray(value), rep)

    def evaluate(basis, *, equilibrium=False, fit_bases=None):
        U, external, rho_a = (np.asarray(basis[k]) for k in ("U", "e", "rho"))
        energy = frozen_energy.copy()
        if options.get("frozen_edge_updates", True):
            energy = energy0.copy()
            energy[:, :start] += (external[:, 0] - energy0[:, start])[:, None]
            energy[:, start + na:] += (external[:, -1] - energy0[:, start + na - 1])[:, None]
        energy[:, active] = external
        density = SS.embed_active_density(occupations, rho_a, start)
        energy_dev, density_dev = (replicated(x) for x in
                                  (energy[:, :n_chi], density[:, :n_chi, :n_chi]))
        totals = {key: np.zeros((nk, na, na), complex)
                  for key in ("gamma", "injection", "m_h", "static")}
        receipts = []
        current_screens = {}
        for iq in range(nk):
            tq = time.perf_counter()
            T, projection = prepared["vertices"](iq)
            partner = prepared["kmq"][iq]
            T = rotate_pair_vertices(T, U, partner, active)
            t_project = time.perf_counter()
            if (equilibrium or mode == "joint") and options.get("statistics") == "lossless_bath":
                negative = int(prepared["q_negative"][iq])
                screen = SS.lossless_bath_snapshot(T[:, :n_chi, :n_chi],
                    energy[:, :n_chi], density[:, :n_chi, :n_chi], partner,
                    fit_sites, check_sites, spin_weight=prepared["spin_weight"],
                    max_order=int(options.get("max_order", 48)),
                    fit_tolerance=float(options.get("fit_tolerance", 0.02)),
                    max_pencil_bytes=int(options.get("max_pencil_bytes", 128 * 1024 ** 2)),
                    representative=current_screens.get(negative) if negative < iq else None,
                    self_negative=negative == iq)
                current_screens[iq] = screen
            elif equilibrium or mode == "joint":
                Tdev = device_put_process_local(np.pad(T[:, :n_chi, :n_chi],
                    ((0, nk_pad - nk), (0, 0), (0, 0), (0, 0))), vertex_sh)
                partner_dev = replicated(partner)
                def sampler(sites):
                    return np.asarray(gather_to_host(sample_projected(
                        Tdev, energy_dev, density_dev, partner_dev, replicated(sites),
                        replicated(gamma), replicated(prepared["spin_weight"]))))
                negative = int(prepared["q_negative"][iq])
                if options.get("statistics") == "retarded_reference":
                    screen = SS.retarded_reference_snapshot(sampler, fit_sites, check_sites,
                        max_order=int(options.get("max_order", 48)),
                        fit_tolerance=float(options.get("fit_tolerance", 0.02)),
                        max_pencil_bytes=int(options.get("max_pencil_bytes", 128 * 1024 ** 2)),
                        representative=current_screens.get(negative) if negative < iq else None,
                        self_negative=negative == iq, collision_eta=eta,
                        fit_basis=(fit_bases.setdefault(iq, {}) if fit_bases is not None
                                   and negative >= iq else None))
                elif negative < iq:
                    screen = SS.screen_snapshot_partner(current_screens[negative], sampler, check_sites)
                else:
                    screen = SS.screen_snapshot(sampler, fit_sites, check_sites,
                        max_order=int(options.get("max_order", 48)),
                        fit_tolerance=float(options.get("fit_tolerance", 1e-3)),
                        max_projection_order=int(options.get("max_projection_order", 400)),
                        max_pencil_bytes=int(options.get("max_pencil_bytes", 128 * 1024 ** 2)),
                        real_fit_sites=fit_sites.real if options.get("real_axis_fit", False) else None,
                        self_negative=negative == iq)
                current_screens[iq] = screen
            else:
                if iq not in cache:
                    raise ValueError("GATE td_snapshot_equilibrium: evaluate equilibrium before frozen W")
                screen = cache[iq][0]
            if equilibrium:
                cache[iq] = (screen, kr.evaluate(screen["correlation"], 0.0))
            if iq not in cache:
                raise ValueError("GATE td_snapshot_equilibrium: static Wc0 baseline missing")
            t_screen = time.perf_counter()
            rows = q_sigma_operators(T[:, active, :n_sigma].conj(),
                energy[partner, :n_sigma], density[partner, :n_sigma, :n_sigma],
                external, dict(retarded=screen["correlation"], lesser=screen["lesser"]),
                eta=eta, active=active, static_wc0=cache[iq][1])
            for key in totals:
                totals[key] += rows[key] / nk
            receipt = screen["receipt"]
            receipts.append(dict(projection,
                fit_order=int(receipt["fit_order"]),
                bubble_check_error=float(receipt["bubble_check_error"]),
                screening_check_error=float(receipt["screening_check_error"]),
                noise_check_error=float(receipt["noise_check_error"]),
                derived_by_crossing=bool(receipt.get("derived_by_crossing", False)),
                crossing_noise_error=(float(receipt["noise_check_error"])
                                      if receipt.get("derived_by_crossing", False) else 0.0),
                self_negative_crossing=bool(receipt.get("self_negative_crossing", False)),
                fit_method=receipt.get("fit_method"),
                training_refit_error=receipt.get("training_refit_error"),
                fit_pencil_shape=receipt.get("fit_pencil_shape"),
                fit_pencil_bytes=receipt.get("fit_pencil_bytes"),
                screening_density_approximation=receipt.get("screening_density_approximation"),
                discarded_screening_coherence_norm=receipt.get("discarded_screening_coherence_norm"),
                noise_approximation=receipt.get("noise_approximation"),
                noise_check_kind=receipt.get("noise_check_kind"),
                collective_normalization_error=receipt.get("collective_normalization_error"),
                minimum_noise_residue=receipt.get("minimum_noise_residue"),
                zero_gap_noise=receipt.get("zero_gap_noise"),
                zero_gap_statistics=receipt.get("zero_gap_statistics"),
                zero_gap_transition_count=receipt.get("zero_gap_transition_count"),
                zero_gap_unequal_population_count=receipt.get("zero_gap_unequal_population_count"),
                zero_gap_equal_population_static_count=receipt.get("zero_gap_equal_population_static_count"),
                zero_gap_response_rank=receipt.get("zero_gap_response_rank"),
                zero_gap_response_roundoff_residual=receipt.get("zero_gap_response_roundoff_residual"),
                zero_gap_discarded_roundoff_response_norm=receipt.get("zero_gap_discarded_roundoff_response_norm"),
                near_zero_unequal_population_count=receipt.get("near_zero_unequal_population_count"),
                minimum_nonzero_transition_gap_ry=receipt.get("minimum_nonzero_transition_gap_ry"),
                screening_gap_resolution_ry=receipt.get("screening_gap_resolution_ry"),
                screening_gap_projection_count=receipt.get("screening_gap_projection_count"),
                maximum_screening_gap_adjustment_ry=receipt.get("maximum_screening_gap_adjustment_ry"),
                dense_workspace_bytes=receipt.get("dense_workspace_bytes"),
                lag_decay_used=receipt.get("lag_decay_used"),
                screening_population_roundoff_correction=receipt.get("screening_population_roundoff_correction"),
                preliminary_transition_projection=receipt.get("preliminary_transition_projection"),
                original_transition_counts=receipt.get("original_transition_counts"),
                projected_transition_counts=receipt.get("projected_transition_counts"),
                final_screened_model=receipt.get("final_screened_model"),
                reference_zero_frequency_count=receipt.get("reference_zero_frequency_count"),
                reference_signed_gain_pole_count=receipt.get("reference_signed_gain_pole_count"),
                reference_complex_residue_pole_count=receipt.get("reference_complex_residue_pole_count"),
                reference_maximum_pole_growth_ry=receipt.get("reference_maximum_pole_growth_ry"),
                reference_collision_eta_ry=receipt.get("reference_collision_eta_ry"),
                reference_laplace_margin_ry=receipt.get("reference_laplace_margin_ry"),
                reference_support_margin_ry=receipt.get("reference_support_margin_ry"),
                reference_modal_condition=receipt.get("reference_modal_condition"),
                reference_modal_error=receipt.get("reference_modal_error"),
                polarization_fit_basis=receipt.get("polarization_fit_basis"),
                polarization_fit_basis_reused=receipt.get("polarization_fit_basis_reused"),
                project_s=t_project - tq, screening_s=t_screen - t_project,
                sigma_s=time.perf_counter() - t_screen))
        Uh = U.conj().swapaxes(-1, -2)
        if prepared.get("head") is not None:
            from dataclasses import replace
            from gw.ppm_sigma import _line_head
            head = replace(prepared["head"], enk_ry=energy,
                           occupations=np.real(np.diagonal(density, axis1=-2, axis2=-1)),
                           efermi_ry=prepared["efermi_ry"])
            hv, hc, hs = _line_head(head, energy - prepared["efermi_ry"], active,
                                    rho_a, np.eye(na)[None] - rho_a)
            gh, dh, mh = C.branch_operators(hc, hv)
            for key, value in (("gamma", gh), ("injection", dh), ("m_h", mh), ("static", hs)):
                totals[key] += np.asarray(value)
        totals = {key: U @ val @ Uh for key, val in totals.items()}
        totals["screening_receipts"] = receipts
        return totals

    return evaluate, cache


def snapshot_collision(source, ops, *, mesh, k_frac, b_lo, ntr, model,
                       options=None, say=print, record=None):
    """Native full-microscopic collision adapter with full-density snapshots.

    The keyword bundle ``options`` owns numerical controls in eV:
    ``spm_support_sites, spm_scratch, fit_tolerance, adiabatic_ev,
    eta_ev, snapshot_steps, snapshot_tol_ev,
    snapshot_mixing, snapshot_acceleration, screening, statistics, form, hermitian, policy, basis``.
    ``retarded_reference`` fits full-mu full-rho PR/WR with the native
    shared-frequency constructor, and uses equilibrium-style T=0 reference
    branches without an independent fluctuation-noise model. All centroid
    ports, q points and virtual bands are retained. Electronic lines keep
    the complete active density. The source GW recipe, not a projected
    rank or prototype pole-order setting, owns frequency approximation.
    ``hamiltonian='hf'`` (default) uses the authenticated absolute
    ``hHF[rho]+MH[rho,H]`` and solves its coherent correction already at
    t0. The supplied DFT density is then explicitly a prepared quench.
    ``hamiltonian='reference'`` uses ``diag(e_ref)+delta_HF+MH-MH_eq``:
    its constant one-body counterterm anchors the supplied reference.
    Both use bare Hartree/Fock increments; neither subtracts a reduced
    static Wc term from a different full screened reference kernel.

    ``screening`` is ``joint`` or ``joint_frozen``; both use the same
    full microscopic equilibrium screening, then only ``joint`` rebuilds it inside
    each snapshot iterate.  Return keys match ``dynsig.window_collision``:
    diss, held, refresh, state, t0, eq. All ranks call this adapter.
    """
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from tdagw.model import flat_k_rows
    from tdagw.tdgw_io import HA_IN_EV

    options = dict(options or {})
    statistics = options.get("statistics", "unspecified")
    if statistics != "retarded_reference":
        raise ValueError("GATE td_collision_statistics: native full-microscopic joint modes require "
                         "retarded_reference; finite_lag/lossless_bath helpers are prototype diagnostics only")
    if options.get("density_rank") is not None:
        raise ValueError("GATE td_native_density_rank: spatial rank projection is not a native joint control")
    statistical_status = dict(statistics="retarded_reference", screened_equilibrium_fdt="not_a_nonequilibrium_statistics_claim",
             physical_lossless_statistics="not_requested_retarded_vacuum_reference",
             screening_density_approximation="full_matrix_ordered_full_microscopic_snapshot")
    form, hermitian = options.get("form", "subtracted"), options.get("hermitian", "dynamic")
    mode = options.get("screening", "joint")
    convention = options.get("hamiltonian", "hf")
    if convention not in ("hf", "reference"):
        raise ValueError("GATE td_snapshot_hamiltonian: expected hf or reference")
    if mode not in ("joint", "joint_frozen") or form not in C.FORMS or hermitian not in C.HERMITIAN_MODES:
        raise ValueError("GATE td_snapshot_options: invalid screening, collision form or Hermitian mode")
    if options.get("basis", "instantaneous") != "instantaneous":
        raise ValueError("GATE td_snapshot_basis: joint snapshots require the full-H instantaneous basis")
    if model is None:
        raise ValueError("GATE td_snapshot_model: propagator reference mean-field model required")
    t_setup = time.perf_counter()
    prepared = prepare_native_source(source, mesh)
    nk, na, nk_pad = prepared["nk"], int(ops.e_qp.shape[-1]), int(ops.e_qp.shape[0])
    start, active = int(b_lo), slice(int(b_lo), int(b_lo) + na)
    if start < 0 or start + na > prepared["nb"]:
        raise ValueError("GATE td_snapshot_window: propagated window outside full electronic support")
    if start + na > min(prepared["nb_chi"], prepared["nb_sigma"]):
        raise ValueError("GATE td_snapshot_window_support: active window must lie in both GW band sums")
    td_rows = np.asarray(flat_k_rows(np.asarray(k_frac), prepared["kgrid"]), int)
    bundle_rows = np.asarray(flat_k_rows(prepared["k_frac"], prepared["kgrid"]), int)
    if len(td_rows) != nk or not np.array_equal(np.sort(td_rows), np.arange(nk)):
        raise ValueError("GATE td_snapshot_td_grid: propagated k grid must be the same full zone")
    bundle_of = np.argsort(bundle_rows)[td_rows]
    to_bundle = lambda x: np.asarray(x)[np.argsort(bundle_of)]
    gauge = authenticate_orbital_gauge(prepared, model, active)
    rho_eq_h = np.asarray(gather_to_host(ops.rho_eq))[:nk]
    if not np.allclose(np.diagonal(rho_eq_h, axis1=-2, axis2=-1).real,
                       prepared["occupations"][bundle_of, active], rtol=0.0, atol=1e-12):
        raise ValueError("GATE td_snapshot_reference: source and propagator occupations differ")
    ry_ha = RYD_TO_EV / HA_IN_EV
    e_ref = to_bundle(np.asarray(gather_to_host(ops.e_qp))[:nk]) / ry_ha
    hf_eq, bare_head = hf_reference_window(source, prepared, active)
    put = lambda blocks: tp.place_window_blocks(blocks, mesh, nk_pad)
    h_ref = np.einsum("ki,ij->kij", e_ref, np.eye(na)).astype(complex)
    offset = np.zeros_like(hf_eq) if convention == "reference" else hf_eq - h_ref
    baseline = put(np.stack([offset[bundle_of]] * ntr) * ry_ha)
    model = bare_hf_coherent_model(model, prepared, ops.rho_eq, baseline,
        bare_head_ha=bare_head * ry_ha, explicit_hf_model=options.get("hf_model"))
    frozen_energy = prepared["energy"].copy()
    frozen_energy[:, :start] += (e_ref[:, 0] - frozen_energy[:, start])[:, None]
    frozen_energy[:, start + na:] += (e_ref[:, -1] - frozen_energy[:, start + na - 1])[:, None]
    evaluator_opts = dict(options, b_lo=start, n_active=na, say=say,
        frozen_energy=frozen_energy, screening="frozen" if mode == "joint_frozen" else "joint")
    evaluate, screening_cache = make_native_snapshot_evaluator(source, prepared, evaluator_opts)
    held0 = None if hermitian == "none" else put(np.zeros((ntr, nk, na, na), complex))
    rho0 = put(np.broadcast_to(rho_eq_h, (ntr, nk, na, na)))
    slow_apply = model.apply if hermitian == "none" else with_held_term(model, None).apply
    occupations_dev = jax.jit(lambda r: jnp.real(jnp.diagonal(r, axis1=-2, axis2=-1)))(ops.rho_eq)
    step = slow_basis_step(slow_apply, ops.rho_eq, ops.e_qp, occupations_dev)

    def bases_at(rho, params, t):
        U, E, _, _, _ = collision_basis_state(step, params, rho, t)
        U_h = np.asarray(gather_to_host(U))[:, :nk]
        E_h = np.asarray(gather_to_host(E))[:, :nk]
        rho_h = np.asarray(gather_to_host(rho))[:, :nk]
        bases = []
        for tr in range(ntr):
            Ub, Eb = to_bundle(U_h[tr]), to_bundle(E_h[tr]) / ry_ha
            rb = to_bundle(rho_h[tr])
            rp = Ub.conj().swapaxes(-1, -2) @ rb @ Ub
            Ub, Eb, _ = _track_bands(Ub, Eb, np.diagonal(rp, axis1=-2, axis2=-1).real)
            rp = Ub.conj().swapaxes(-1, -2) @ rb @ Ub
            bases.append(dict(U=Ub, e=Eb, rho=rp))
        return bases

    params0 = model.params if hermitian == "none" else (model.params, held0)
    bases0 = bases_at(rho0, params0, 0.0)
    cold_snapshot = dict(iterations=1, residual=0.0, converged=True)
    cold_fit_bases = {} if statistics == "retarded_reference" else None
    if convention == "hf" and hermitian != "none":
        def cold(candidate):
            bases = bases_at(rho0, (model.params, candidate), 0.0)
            row = (evaluate(bases[0], equilibrium=True, fit_bases=cold_fit_bases)
                   if cold_fit_bases is not None else evaluate(bases[0], equilibrium=True))
            target = put(np.stack([row["m_h"][bundle_of]] * ntr) * ry_ha)
            return target, (row, bases)

        held0, (eq, bases0), cold_snapshot = collision_snapshot_state(cold, held0,
            steps=int(options.get("snapshot_steps", 32)),
            tol_ha=float(options.get("snapshot_tol_ev", 1e-4)) / HA_IN_EV,
            mixing=float(options.get("snapshot_mixing", 0.7)),
            acceleration=options.get("snapshot_acceleration", "anderson"))
    else:
        eq = evaluate(bases0[0], equilibrium=True)
    keys = ("gamma", "injection", "m_h", "static")
    eq_rows = {key: np.stack([eq[key][bundle_of]] * ntr) for key in keys}
    gamma_eq, injection_eq = (put(eq_rows[key] * ry_ha) for key in ("gamma", "injection"))
    program = tp.collision_program(mesh, raw=form == "raw")
    initial = program(gamma_eq, injection_eq, rho0, ops.rho_eq, gamma_eq, injection_eq, ops.w_k)
    jax.block_until_ready(initial["diss"])
    fs = 1000.0 / 24.188843265857
    policy = options.get("policy") or C.RefreshPolicy()
    pops = jax.jit(lambda r: jnp.diagonal(r, axis1=-2, axis2=-1).real)

    def diagnostics(out):
        outgoing, incoming = np.asarray(out["out"]) * fs, np.asarray(out["inn"]) * fs
        flux = 0.5 * (abs(outgoing) + abs(incoming))
        residual = outgoing + incoming
        relative = np.divide(abs(residual), flux, out=np.zeros_like(flux), where=flux > 0)
        return dict(trace=np.asarray(out["trace"]).tolist(),
            number_residual_per_fs=residual.tolist(), number_residual_rel=relative.tolist(),
            out_per_fs=outgoing.tolist(), in_per_fs=incoming.tolist(),
            min_rate_mev=float(out["min_rate"]) * HA_IN_EV * 1e3,
            min_injection_mev=float(out["min_injection"]) * HA_IN_EV * 1e3,
            min_blocking_mev=float(out["min_blocking"]) * HA_IN_EV * 1e3,
            rho_min=float(out["rho_min"]), rho_max=float(out["rho_max"]),
            c_eq_diag=float(out["c_eq_diag"]), c_eq_offdiag=float(out["c_eq_offdiag"]))

    t0 = dict(n=0, t_fs=0.0, why="t0", screening=mode, form=form, hermitian=hermitian,
              **statistical_status,
              spatial_projection=False, nmu=prepared["nmu_logical"], electronic_bands=prepared["nb"],
              q_receipts=eq["screening_receipts"], setup_s=time.perf_counter() - t_setup,
              orbital_gauge=gauge,
              hamiltonian_convention=("hHF[rho]+MH[rho,H]" if convention == "hf" else
                  "diag(e_ref)+delta_HF[rho]+MH[rho,H]-MH_eq"),
              hamiltonian=convention, snapshot=cold_snapshot,
              initial_state="DFT-prepared density quench" if convention == "hf" else "anchored reference density",
              **diagnostics(initial))
    t0["wall_s"] = t0["setup_s"]
    state = dict(n=0, t_last=0.0, f_last=np.asarray(gather_to_host(pops(rho0)))[:, :nk],
                 trace_last=np.asarray(initial["trace"]), history=[t0],
                 basis_last=bases0, snapshot_held=held0, hf_reference_ry=hf_eq)
    mh0 = eq["m_h"] if hermitian != "none" else np.zeros_like(hf_eq)
    state["one_body_counterterm_ry"] = h_ref - hf_eq - mh0 if convention == "reference" else np.zeros_like(hf_eq)
    H0 = (bases0[0]["U"] * bases0[0]["e"][..., None, :]) @ bases0[0]["U"].conj().swapaxes(-1, -2)
    req_b = to_bundle(rho_eq_h)
    t0["prepared_density_commutator_ha"] = float(np.max(np.linalg.norm(
        H0 @ req_b - req_b @ H0, axis=(-2, -1)))) * ry_ha
    say(f"  native joint snapshot t0: {prepared['nmu_logical']} mu, {prepared['nb']} bands, {nk} q; "
        f"{mode}; setup {t0['setup_s']:.2f} s")
    if record is not None:
        record(t0)

    def refresh(t, rho, ops_now, params):
        populations = np.asarray(gather_to_host(pops(rho)))[:, :nk]
        due, df, why = C.refresh_due(policy, t, state["t_last"], populations, state["f_last"])
        if not due:
            return None
        wall = time.perf_counter()
        screening_iterations = []
        fit_bases = [{} for _ in range(ntr)] if statistics == "retarded_reference" else None
        residual_history = []

        def iterate(candidate):
            candidate_params = params if hermitian == "none" else held_term(params)[1](candidate)
            bases = bases_at(rho, candidate_params, t)
            rows = ([evaluate(basis, fit_bases=fit_bases[trial]) for trial, basis in enumerate(bases)]
                    if fit_bases is not None else [evaluate(basis) for basis in bases])
            screening_iterations.append(dict(
                screening=[row["screening_receipts"] for row in rows],
                native_sigma=[row["native_receipt"] for row in rows]))
            target = candidate
            if hermitian != "none":
                mh = np.stack([row["m_h"][bundle_of] for row in rows])
                reference = eq_rows["m_h"] if convention == "reference" else np.zeros_like(mh)
                target = put(np.asarray(C.hermitian_update(mh, reference)) * ry_ha)
            return target, (rows, bases)

        predictor = jnp.zeros_like(rho) if hermitian == "none" else held_term(params)[0]
        try:
            held, (rows, bases), snapshot = collision_snapshot_state(iterate, predictor,
                steps=int(options.get("snapshot_steps", 32)),
                tol_ha=float(options.get("snapshot_tol_ev", 1e-4)) / HA_IN_EV,
                mixing=float(options.get("snapshot_mixing", 0.7)),
                acceleration=options.get("snapshot_acceleration", "anderson"),
                residual_history=residual_history)
        except ValueError as error:
            state["snapshot_failure"] = dict(t_fs=float(t) / fs, error=str(error),
                                              screening_iterations=screening_iterations,
                                              residuals_ha=residual_history)
            raise
        snapshot["screening_iterations"] = screening_iterations
        t_swept = time.perf_counter()
        gamma, injection = (put(np.stack([row[key][bundle_of] for row in rows]) * ry_ha)
                            for key in ("gamma", "injection"))
        out = program(gamma, injection, rho, ops_now.rho_eq, gamma_eq, injection_eq, ops_now.w_k)
        jax.block_until_ready(out["diss"])
        t_built = time.perf_counter()
        new_params, delta = params, None
        if hermitian != "none":
            previous, setter = held_term(params)
            new_params, delta = setter(held), held - previous
        t_hermitian = time.perf_counter()
        trace = np.asarray(out["trace"])
        elapsed = (float(t) - state["t_last"]) / fs
        rec = dict(n=state["n"] + 1, t_fs=float(t) / fs, df=float(df), why=why,
                   **statistical_status,
                   screening=mode, wall_s=time.perf_counter() - wall, snapshot=snapshot,
                   sweep_s=t_swept - wall, build_s=t_built - t_swept,
                   hermitian_s=t_hermitian - t_built,
                   held_max_mev=float(jnp.max(jnp.abs(held))) * HA_IN_EV * 1e3,
                   d_trace=(trace - state["trace_last"]).tolist(),
                   trace_drift_per_fs=((trace - state["trace_last"]) / elapsed).tolist(),
                   q_receipts=[row["screening_receipts"] for row in rows], **diagnostics(out))
        state.update(n=rec["n"], t_last=float(t), f_last=populations, trace_last=trace,
                     basis_last=bases, snapshot_held=held)
        state["history"].append(rec)
        say(f"  joint snapshot refresh {rec['n']}: t {rec['t_fs']:.5f} fs, "
            f"{snapshot['iterations']} sweeps, residual {snapshot['residual']:.3e} Ha, "
            f"wall {rec['wall_s']:.2f} s")
        if record is not None:
            record(rec)
        return dict(ops=ops_now._replace(collision=out["diss"]), params=new_params, delta=delta)

    return dict(diss=initial["diss"], held=held0, refresh=refresh, state=state, t0=t0, eq=eq, model=model,
                c_eq=initial["c_eq"], operators_eq=(gamma_eq, injection_eq),
                prepared=prepared, screening=screening_cache, evaluate=evaluate,
                basis_of=bases_at, bundle_of=bundle_of)
