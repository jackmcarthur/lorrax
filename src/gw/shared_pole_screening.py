"""Deck-driven shared real-pole screening (DESIGN §3, rulings15–16).

The response, constructor and persistence owners implement their own stages.
This helper authenticates the current map and passes handles between stages:
files, or the device-resident bank and SC model when they fit. Samples,
directions and poles are rebuilt across changed maps; the SC owner may retain
time nodes after current-domain certification.
"""
from pathlib import Path
import dataclasses
from functools import partial
import hashlib
import json
import shutil
import time

import numpy as np
from common import timing


def active_eta_verification_sites(native_sites, energy, active, eta_ry, *, native_fit_count=0):
    """Keep native IDs, append active-height training and independent holds.

    Energies/sites are Ry. This is numerical geometry, NOT a bound on
    full complex Sigma error. Every native fit/held site stays byte-identical
    in the original order; full-Sigma refinement remains independently needed.
    """
    native=np.asarray(native_sites,dtype=np.complex128)
    energy=np.asarray(energy,float)
    if (native.ndim!=1 or energy.ndim!=2 or not np.isfinite(native).all()
            or not np.isfinite(energy).all() or not np.isfinite(eta_ry) or eta_ry<=0
            or not 0<=active.start<active.stop<=energy.shape[1]
            or not 0<=native_fit_count<=native.size):
        raise ValueError('GATE shared_general_verification: finite native sites and active energies required')
    span=float(np.ptp(energy[:,active]))
    sites=list(native)
    fit=list(range(int(native_fit_count))); held=list(range(int(native_fit_count),len(native)))
    collisions=[]
    for role,fractions in (('fit',(0.,.5,1.)),('held',(.25,.75))):
        own,other=(fit,held) if role=='fit' else (held,fit)
        for fraction in fractions:
            z=complex(fraction*span,eta_ry)
            matches=[i for i,value in enumerate(sites) if value==z]
            if any(i in other for i in matches):
                collisions.append(dict(role=role,fraction=fraction,reason='opposite-role existing site'))
            elif matches:
                if matches[0] not in own:
                    own.append(matches[0])
            else:
                own.append(len(sites));sites.append(z)
    extra=np.asarray(sites[len(native):],complex)
    receipt=dict(native_count=len(native),extra_count=len(extra),eta_ry=float(eta_ry),
        active_bands=[active.start,active.stop],active_transition_span_ry=span,
        extra_real_sites_ry=extra.real.tolist(),native_ids_unchanged=True,
        fit_positions=fit,held_positions=held,collisions=collisions,collapsed_span=span==0.,
        scope='training/verification geometry only; not a Sigma error certificate',
        required_refinement='full complex Sigma including real/offdiagonal and collision operators')
    return np.asarray(sites,complex),receipt


def screen_general_shared_poles(wfns, V_q, meta, config, *, mesh_xy, sym, snapshot,
                                 run_dir, label, fit_basis=None, print_fn=print):
    """Full microscopic instantaneous-rho W, with general shared frequency poles.

    ``snapshot`` supplies ``U_active[k,a,a]``, ``E_active_ry[k,a]``,
    ``rho_active[k,a,a]`` in that instantaneous basis, ``active_slice`` and
    optional full ``energy_ry[k,b]``. ``gamma_ry`` is the declared lag decay;
    ``eta_ry`` is the electronic retarded evaluation height. Every native
    centroid, screening band and full-grid q is retained. Native support,
    direction-width and pole-budget controls come from the source recipe,
    never from the projected TD prototype's rank/order controls.

    General tangential interpolation has independent full-port factors and
    unchanged complex poles. Opposite q models are related algebraically;
    self-negative q models use an explicit half-weight crossing mixture.
    ``fit_basis`` optionally holds only interpolation directions/reduction
    spaces within one inner Hamiltonian solve. Current response, residues,
    collective poles and all held checks are recomputed on every call.
    """
    import copy
    import math
    import sys
    import jax
    import jax.numpy as jnp
    import distrib_la
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host, rank0_transaction
    from common.contract_bands import merge_spin_centroid
    from symmetry_maps import QirrOperator
    from .wavefunction_bundle import rotate_wavefunctions
    from .greens_function_kernel import density_left_endpoint
    from .centroid_k_unfold import authenticate_canonical_full_grid
    from .response_bank import (full_density_sample_rules, full_density_bare_moments,
                                 _coulomb_algebra, dense_layout, response_algebra,
                                 operator_digest, response_blocks, dielectric_physical_metrics,
                                 dielectric_inverse_bounds)
    from .w_isdf import full_density_response_stream
    from .shared_pole_directions import _sample_point
    from .shared_pole_execution import face_matmul
    from .shared_pole_pencil import general_tangential_pencil, tile_join, tile_block, join_vectors
    from .shared_pole_constructor import (general_pencil_batch_plan, collective_general_pencil_batch,
                                          general_pencil_workspace_bytes)
    from .shared_pole_recipe import bind_shared_pole_census, resolve_shared_pole_recipe
    from types import SimpleNamespace
    from file_io.shared_pole_store import write_general_shared_pole_model
    started = time.monotonic()
    phases, phase_start = {}, {}
    def begin(name):
        phase_start[name] = time.monotonic()
        if jax.process_index() == 0:
            print_fn(f"General shared-pole W [{label}] BEGIN {name}")
            sys.stdout.flush()
    def end(name, operands=None):
        if operands is not None:
            jax.block_until_ready(operands)
        phases[name] = time.monotonic()-phase_start.pop(name)
        if jax.process_index() == 0:
            print_fn(f"General shared-pole W [{label}] END {name} {phases[name]:.3f}s")
            sys.stdout.flush()
    begin("setup")
    current_meta = copy.copy(meta)
    source_recipe = current_meta.shared_pole_recipe
    if fit_basis is None:
        fit_basis = {}
    previous_direction_bytes = sum(a.size*a.dtype.itemsize for a in fit_basis.get("directions", {}).values())
    nk = int(meta.nk_tot)
    grid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    qgrid = tuple(int(n) for n in (getattr(meta, "qgrid", None) or grid))
    nq = int(np.prod(qgrid))
    sym = getattr(meta, "q_sym", sym)
    if not np.array_equal(np.asarray(sym.q_irr_full_idx), np.arange(nq)):
        raise ValueError("GATE shared_general_full_q: every current q must be an explicit parent")
    qcoords = np.asarray(sym.q_irr_kgrid_int)
    canonical_qcoords = np.indices(qgrid).reshape(3, -1).T
    if (qcoords.shape != (nq, 3) or qcoords.dtype.kind not in "iu"
            or not np.array_equal(qcoords % np.asarray(qgrid), canonical_qcoords)):
        raise ValueError("GATE shared_general_full_q: canonical unshifted bosonic q coordinates required")
    active = snapshot["active_slice"]
    if not isinstance(active, slice):
        active = slice(*active)
    energy = np.asarray(snapshot.get("energy_ry", gather_to_host(wfns.enk)), float).copy()
    energy[:, active] = np.asarray(snapshot["E_active_ry"], float)
    occupations = np.asarray(gather_to_host(wfns.occ), float)
    rho = np.eye(energy.shape[1], dtype=complex)[None]*occupations[:, :, None]
    rho[:, active, active] = np.asarray(snapshot["rho_active"], complex)
    if (not np.isfinite(rho).all() or np.max(abs(rho-rho.conj().swapaxes(-1, -2))) > 1e-10):
        raise ValueError("GATE shared_general_density: finite Hermitian full density required")
    current = rotate_wavefunctions(wfns, jnp.asarray(snapshot["U_active"]),
        enk_active_new=jnp.asarray(energy[:, active]), enk_base=jnp.asarray(energy),
        efermi=None, mesh_xy=mesh_xy, active_slice=active, occupations=occupations)
    census = meta.shared_pole_census
    bind_shared_pole_census(current, current_meta,
        occupation_state=SimpleNamespace(f_kn=np.real(rho.diagonal(axis1=-2, axis2=-1)),
            mu_ry=float(census["mu_ry"]), smearing_family="instantaneous-full-density-diagonal-census",
            smearing_width_ry=0.0),
        trs_allowed=False, state_capacity=float(census["state_capacity"]),
        kweights=np.asarray(census["k_weights"]))
    resolved = resolve_shared_pole_recipe(config, current, current_meta, mesh_xy=mesh_xy,
        print_fn=print_fn, support_reads_ev=snapshot.get("support_reads_ev", source_recipe["support_reads_ev"]))
    # Geometry and selection are numerical spaces held only within this
    # physical refresh. Current census and scientific identity stay current.
    recipe = dict(fit_basis.setdefault("recipe", resolved), census=current_meta.shared_pole_census)
    general_gates = dict(version="full-density-general-shared-pole-gates-v6",
        source_policy_version=recipe.get("recipe_version", recipe.get("version")),
        scalar_provider='coherent_endpoint_blocks_public_minimax_v1',
        eta_training_verification='active_span_3train_2interlaced_held_v1',
        numerical_spaces='two-sided-descriptor-svd',
        eigensolver='current-descriptor-balanced-ordinary-geev',
        frequency_equilibration='unit-aware-combined-pencil-Ruiz-infinity-8',
        frequency_scale='max_current_full_requested_energy_span_eta',
        descriptor_rank_tolerance=float(recipe['bank_rule_tolerance']),
        full_microscopic_support=True, finite_density_and_residues=True,
        scalar_rule_tolerance=float(recipe["bank_rule_tolerance"]),
        full_matrix_fit_held_tolerance=float(snapshot.get("fit_tolerance", .02)),
        modal_reconstruction_tolerance=1e-9, laplace_margin_ry=1e-12,
        capacity="explicit-full-map-and-host-pencil-bound", store="sha256-complete-full-q")
    recipe.update(source_recipe_hash=recipe["recipe_hash"], source_passive_gate_hash=recipe["gate_hash"],
        source_passive_gate_version=recipe["gate_version"], gate_version=general_gates["version"],
        construction_representation="general-two-sided-full-density",
        inherited_passive_gram_and_passivity="NOT_APPLICABLE", general_gates=general_gates,
        gate_hash=hashlib.sha256(_json(general_gates).encode()).hexdigest())
    recipe["recipe_hash"] = hashlib.sha256(_json({k:v for k,v in recipe.items() if k != "recipe_hash"}).encode()).hexdigest()
    current_meta.shared_pole_recipe = recipe
    current_meta.shared_pole_capacity.live_stages = ()
    carrier = current.green_parent
    if carrier is None or carrier.psi_mun.shape[0] != nk:
        raise ValueError("GATE shared_general_full_k: explicit full-grid wavefunction parents required")
    plan = carrier.plan
    if (not np.array_equal(np.asarray(plan.irr_idx), np.arange(nk))
            or plan.parent_full_rows is None
            or not np.array_equal(np.asarray(plan.parent_full_rows), np.arange(nk))):
        raise ValueError("GATE shared_general_full_k: native flat-grid identity row order required")
    grid_receipt = authenticate_canonical_full_grid(
        plan, grid, sym=sym if qgrid == grid else None)
    grid_receipt.update(response_grid=list(qgrid), n_q=nq, response_q_order="canonical-C")
    pm, pn = carrier.psi_mun, carrier.psi_nmu
    layout = carrier.layout
    ns, nmu, nb = map(int, pm.shape[1:])
    stop = min(int(meta.b_id_4_chi_user), int(wfns.slices.b4_logical))-int(wfns.slices.b0)
    if int(wfns.slices.b0) != 0 or stop < active.stop:
        raise ValueError("GATE shared_general_bands: complete screening support must contain active window")
    # Carrier padding/out-of-chi bands are exact-zero operands, not a band
    # cutoff chosen for this algorithm. All requested physical chi bands stay.
    band_mask = jnp.arange(nb) < stop
    pm, pn = pm*band_mask[None, None, None], pn*band_mask[None, :, None, None]
    rho[:, stop:, :] = 0; rho[:, :, stop:] = 0
    v = QirrOperator.of(V_q)
    if not v.is_whole_zone() or v.n_full != nq:
        raise ValueError("GATE shared_general_full_q: full native Coulomb operator required")
    V = v.at_rows(np.arange(nq))
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    rep = NamedSharding(mesh_xy, P())
    mm = face_matmul(mesh_xy)
    fit_ids, held_ids = list(map(int, recipe["fit_ids"])), list(map(int, recipe["held_ids"]))
    sites = np.asarray([_sample_point(recipe, i) for i in fit_ids+held_ids], complex)
    # Fit the observed continuation scale, while keeping independent
    # interlaced holds and EVERY original native fit/held position intact.
    eta = float(snapshot["eta_ry"])
    sites,verification_geometry=active_eta_verification_sites(sites,energy,active,eta,
        native_fit_count=len(fit_ids))
    fit_positions=verification_geometry['fit_positions']
    held_positions=verification_geometry['held_positions']
    recipe['general_support_geometry']=dict(sites_ry=[[float(z.real),float(z.imag)] for z in sites],
        fit_positions=fit_positions,held_positions=held_positions,
        training_width_policy='native imaginary_width / line_direction_cap by support type')
    widths = [int(recipe["imaginary_width"]) if z.real == 0 else
              int(recipe.get("line_direction_cap") or nmu) for z in sites[fit_positions]]
    recipe['general_support_geometry']['training_widths']=widths
    general_gates['support_geometry_hash']=hashlib.sha256(_json(recipe['general_support_geometry']).encode()).hexdigest()
    recipe['gate_hash']=hashlib.sha256(_json(general_gates).encode()).hexdigest()
    recipe['recipe_hash']=hashlib.sha256(_json({k:v for k,v in recipe.items() if k!='recipe_hash'}).encode()).hexdigest()
    side_bound = sum(extent for z, extent in zip(sites[fit_positions], widths) if z.real != 0)
    side_bound += sum(widths)+2*int(recipe["infinity_width"])
    ranks = int(mesh_xy.size)
    bank_bytes = 16*nq*len(sites)*nmu*nmu//ranks
    action_bytes = 16*nq*4*nmu*side_bound//ranks
    pencil_bytes = 32*nq*side_bound*side_bound//ranks
    green_bytes = 16*nk*(ns*nmu)**2*6//ranks
    price = current_meta.shared_pole_capacity.reserve("general_full_map",
        resident_bytes_per_rank=bank_bytes+action_bytes+pencil_bytes,
        workspace_bytes_per_rank=green_bytes+bank_bytes)
    current_meta.shared_pole_capacity.live_stages = (price["stage"],)
    ph = density_left_endpoint(mesh_xy, pm.shape, layout=layout)(pm, jnp.asarray(rho))
    energy_d = jnp.asarray(energy)
    roots = _coulomb_algebra(mesh_xy, nmu, int(meta.n_rmu),
                              dense_layout(meta, config, mesh_xy, nmu))(V)
    H, _, negative, _ = roots
    if bool(jnp.any(negative)):
        raise ValueError("GATE shared_general_coulomb: bare V has resolved negative spectrum")
    dyson, _, _, algebra = response_algebra(current_meta, config, mesh_xy=mesh_xy, n=nmu, ordered=True)
    gamma = float(snapshot.get("gamma_ry", snapshot.get("adiabatic_ry", 0)))
    tolerance = float(snapshot.get("fit_tolerance", .02))
    if len(fit_ids) < 4:
        raise ValueError("GATE shared_general_supports: at least four native fitted supports required")
    end("setup", (ph, H))
    begin("planning")
    response_active = snapshot.get('response_active_slice', active)
    if not isinstance(response_active,slice):
        response_active = slice(*response_active)
    if not 0 <= response_active.start <= active.start < active.stop <= response_active.stop <= stop:
        raise ValueError('GATE response_blocks_active: numerical response group must contain the coherent window')
    band = np.arange(nb)[None]
    amask = (band >= response_active.start)&(band < response_active.stop)
    live = band < stop
    f_support = np.where(amask,1.,occupations)*live
    u_support = np.where(amask,1.,1.-occupations)*live
    blocks = response_blocks(energy,f_support,u_support,(response_active.start,response_active.stop),
                             allow_crossing=True)
    rules, scalar_cache = [], fit_basis.setdefault('scalar_rules',{})
    # Do not force native high-height supports to use the narrow eta-height
    # continuation grid. Partition by actual height and by noncrossing status;
    # every original site remains an independently checked physical target.
    heights = np.floor(np.log2((sites.imag+gamma)/np.min(sites.imag+gamma))/2).astype(int)
    for bi, block in enumerate(blocks):
        lm, rm = block['u'] != 0, block['f'] != 0
        block['bounds'] = [float(energy[lm].min()),float(energy[lm].max()),
                           float(energy[rm].min()),float(energy[rm].max())]
        # Only spectator columns with EXACT 0/1 density may be skipped.
        # Every propagated coherent column stays live regardless of its
        # diagonal occupation; rho need not commute with current energies.
        coherent = (band >= active.start)&(band < active.stop)
        left_support = lm & ((coherent | (occupations != 1)) if block['name'][1]=='A' else True)
        right_support = rm & ((coherent | (occupations != 0)) if block['name'][0]=='A' else True)
        ranges = []
        for support in (left_support,right_support):
            nonempty = support.any(axis=1)
            lower = np.where(nonempty,np.argmax(support,axis=1),0)
            upper = np.where(nonempty,nb-np.argmax(support[:,::-1],axis=1),0)
            ranges.append(np.stack([lower,upper]))
        block['band_ranges'] = np.asarray(ranges,np.int32)
        remote = (np.abs(sites.real) < block['lo']) if block['lo'] > 0 else np.zeros(len(sites),bool)
        groups = sorted(set(zip(remote.tolist(),heights.tolist())))
        for is_remote,height in groups:
            ids = np.flatnonzero((remote == is_remote)&(heights == height))
            # Noncrossing must also survive outward hold padding; if near
            # the boundary, the general bounded primitive handles it exactly.
            key = (block['name'],tuple(ids))
            if jax.process_index()==0:
                print_fn(f"General shared-pole W [{label}] plan block={block['name']} "
                         f"interval_Ry={[block['lo'],block['hi']]} sites={len(ids)} "
                         f"height_bin={height} noncrossing={is_remote}")
                sys.stdout.flush()
            planned = full_density_sample_rules(energy[:,:stop],sites[ids],gamma_ry=gamma,
                tolerance=float(recipe['bank_rule_tolerance']), interval_ry=(block['lo'],block['hi']),
                paired=True,held=scalar_cache.setdefault(key,{}))
            for rule in planned:
                rules.append(dict(rule,block_index=bi,block=block['name'],
                                  members=[int(ids[j]) for j in rule['members']]))
    end("planning")
    if jax.process_index() == 0:
        print_fn(f"General shared-pole W [{label}] scalar plan: {len(rules)} panels, "
                 f"{sum(len(r['times']) for r in rules)} Green pairs; "
                 f"max error={max(r['sampled_error'] for r in rules):.3e}, "
                 f"mass={max(r['coefficient_mass'] for r in rules):.3e}; "
                 "Hankel workspace bound=4GiB, panel count bound=32")
        sys.stdout.flush()
    begin("stream")
    polarizations = [None]*len(sites)
    for panel_index, rule in enumerate(rules):
        panel_start = time.monotonic()
        block = blocks[rule['block_index']]
        if block['name'][1] == 'A':
            left = (pm-ph)*jnp.asarray(block['u'])[:,None,None]
        else:
            left = pm*jnp.asarray(block['u'])[:,None,None]
        right = (ph if block['name'][0] == 'A' else pm)*jnp.asarray(block['f'])[:,None,None]
        stream = full_density_response_stream(mesh_xy, grid, (nk, nb, nmu, ns),
            q_ids=np.arange(nq), n_outputs=len(rule["members"]), layout=layout, qgrid=qgrid,
            rule_reference_ry=rule["reference_ry"],pair_bounds_ry=block['bounds'],
            pair_band_ranges=block['band_ranges'])
        raw = stream(jnp.asarray(rule["times"]), jnp.asarray(rule["weights"]), left, pn, right, energy_d)
        for j, sid in enumerate(rule["members"]):
            value = raw[:, j]
            polarizations[sid] = value if polarizations[sid] is None else polarizations[sid]+value
        # Bound asynchronous live panels to the accumulator plus CURRENT
        # raw result, rather than queueing every full-microscopic bank.
        jax.block_until_ready(polarizations)
        del raw, value,left,right
        if jax.process_index() == 0 and (panel_index == 0 or (panel_index+1) % 4 == 0
                                        or panel_index+1 == len(rules)):
            print_fn(f"General shared-pole W [{label}] stream panel {panel_index+1}/{len(rules)} "
                     f"nodes={len(rule['times'])} wall={time.monotonic()-panel_start:.3f}s "
                     f"elapsed={time.monotonic()-phase_start['stream']:.3f}s")
            sys.stdout.flush()
    end("stream", polarizations)
    # Minimax panels add POLARIZATIONS, not screened W. Never add independent
    # Dyson solutions when more than one panel contributes to the same site.
    begin("dyson")
    values = [dyson.value(H, value) for value in polarizations]
    del polarizations
    end("dyson", values)
    sample_wall = time.monotonic()-started
    put = lambda a, sh=rep: device_put_process_local(np.asarray(a), sh)
    panel_side = int(mesh_xy.shape["y"])
    extent = lambda n: max(panel_side, math.ceil(int(n)/panel_side)*panel_side)
    eig = distrib_la.plan("eigh", mesh_xy, n=nmu)
    directions = fit_basis.setdefault("directions", {})
    begin("directions")
    coords = np.unravel_index(np.arange(nq), qgrid)
    minus = np.ravel_multi_index(tuple((-c) % n for c, n in zip(coords, qgrid)), qgrid)
    left_states, right_states = [], []
    for training_index,i in enumerate(fit_positions):
        W = values[i]
        width = (int(recipe["imaginary_width"]) if sites[i].real == 0
                 else int(recipe.get("line_direction_cap") or nmu))
        if i not in directions:
            # Both sides use singular directions: -Herm(W) is not PSD for
            # a general coherent/gain input and must not be used as a Gram.
            source = W if training_index % 2 else jax.jit(lambda w: jnp.conj(jnp.swapaxes(w, -1, -2)),
                                             out_shardings=face)(W)
            q, _ = distrib_la.right_singular_vectors(source, float(recipe["direction_cutoff"]),
                eigh_plan=eig, column_extent=extent, max_rank=width,
                multiplet_tol=float(recipe["multiplet_relative_tolerance"]))
            directions[i] = q
        q = directions[i]
        if training_index % 2:
            right_states.append((sites[i], q, mm(W, q)))
        else:
            left_states.append((sites[i], q, mm(q, W, transa="C")))
        if sites[i].real != 0:
            # Exact crossing supplies the opposite frequency using the
            # independently current partner sample. No additional Dyson
            # sample, equilibrium transpose relation or PSD premise enters.
            qm = jnp.conj(q[minus])
            wm = jnp.conj(W[minus])
            if training_index % 2:
                left_states.append((-sites[i].conjugate(), qm, mm(qm, wm, transa="C")))
            else:
                right_states.append((-sites[i].conjugate(), qm, mm(wm, qm)))
    def joined(states, left):
        zs = join_vectors([np.full(s[1].shape[-1], s[0]) for s in states], panel_side)
        return (put(zs), tile_join([s[1] for s in states], -1, face),
                tile_join([s[2] for s in states], -2 if left else -1, face))
    mu, ql, il = joined(left_states, True)
    lam, qr, wr = joined(right_states, False)
    end("directions", (ql, il, qr, wr))
    begin("pencil")
    E, A, B, C = jax.jit(lambda mu, lam, ql, il, qr, wr: general_tangential_pencil(
        (mu, ql, il), (lam, qr, wr), matmul=mm), out_shardings=(face, face, face, face))(mu, lam, ql, il, qr, wr)
    # General infinity rows constrain the exact 1/z and 1/z² coefficients
    # on the retained infinity directions. Full-port moment defects are
    # measured below; finite direction/reduction accuracy does not make
    # those full-matrix coefficients exact.
    # They are not forced to be Hermitian/even for a coherent finite-band
    # projected density algebra. Gamma shifts X2 by -i gamma X1 once.
    x1, x2 = full_density_bare_moments((pm, pn, ph, energy_d), meta,
                                      mesh_xy=mesh_xy, q_ids=np.arange(nq))
    pref = float(algebra["prefactor"])
    c1 = mm(V, mm(pref*x1, V))
    c2 = mm(V, mm(pref*(x2-1j*gamma*x1), V))+mm(V, mm(pref*x1, mm(V, mm(pref*x1, V))))
    moment_scale = max(float(np.ptp(energy[:, :stop])), eta)
    if "infinity" not in directions:
        c2_scaled = c2/moment_scale
        # Union of both moment input AND output ranges. This is a numerical
        # direction Gram, not Hermitization/positivity repair of W or rho.
        normal = (mm(c1, c1, transb="C")+mm(c1, c1, transa="C")
                  +mm(c2_scaled, c2_scaled, transb="C")+mm(c2_scaled, c2_scaled, transa="C"))
        normal = jax.jit(lambda a: .5*(a+jnp.conj(jnp.swapaxes(a, -1, -2))),
                         out_shardings=face)(normal)
        directions["infinity"], _ = distrib_la.leading_eigenvectors(normal,
            int(recipe["infinity_width"]), eigh_plan=eig, column_extent=extent,
            rcond=max(nmu*np.finfo(float).eps, float(recipe["direction_cutoff"])**2),
            multiplet_tol=float(recipe["multiplet_relative_tolerance"]))
    qi = directions["infinity"]
    def add_infinity(mu, lam, E, A, B, C, ql, il, qr, wr, qi, c1, c2):
        ei = mm(qi, wr, transa="C")
        ef = mm(il, qi)
        ci = mm(c1, qi)
        bi = mm(qi, c1, transa="C")
        eii = mm(qi, ci, transa="C")
        aii = mm(qi, mm(c2, qi), transa="C")
        ai = ei*lam[None, None]-mm(bi, qr)
        af = mu[None, :, None]*ef-mm(ql, ci, transa="C")
        return (tile_block(((E, ef), (ei, eii)), face),
                tile_block(((A, af), (ai, aii)), face),
                tile_join((B, bi), -2, face), tile_join((C, ci), -1, face))
    E, A, B, C = jax.jit(add_infinity, out_shardings=(face,)*4)(mu, lam, E, A, B, C, ql, il, qr, wr, qi, c1, c2)
    # Infinity conditions have one extra energy unit per endpoint relative
    # to finite divided differences. Match those units BEFORE numerical
    # rank selection, respecting the actual tiled join order on each axis.
    coordinate_units=(join_vectors((np.ones(mu.shape[0]),np.full(qi.shape[-1],1/moment_scale)),panel_side),
                      join_vectors((np.ones(lam.shape[0]),np.full(qi.shape[-1],1/moment_scale)),panel_side))
    frequency_coordinates=dict(frequency_scale_ry=moment_scale,
        infinity_width=int(qi.shape[-1]),tile_extent=panel_side,
        left_units_sha256=hashlib.sha256(coordinate_units[0].tobytes()).hexdigest(),
        right_units_sha256=hashlib.sha256(coordinate_units[1].tobytes()).hexdigest())
    recipe['frequency_coordinates']=frequency_coordinates
    recipe['recipe_hash']=hashlib.sha256(_json({k:v for k,v in recipe.items() if k!='recipe_hash'}).encode()).hexdigest()
    positive_cap = recipe.get("pole_budget")
    cap = 2*int(positive_cap) if positive_cap is not None else min(E.shape[-2:])
    # Source recipe is the native accuracy policy. Host workspace is an
    # explicit numerical admission bound, not a pole budget chosen to fit RAM.
    workspace = int(snapshot.get("host_workspace_bytes", 4*1024**3))
    nl, nr = map(int, E.shape[-2:])
    host_price = general_pencil_workspace_bytes((nl,nr))
    if host_price > workspace:
        raise ValueError(f"GATE shared_general_capacity: reduced host pencil {nl}x{nr} "
                         f"needs {host_price}>{workspace} bytes before any host gather")
    end("pencil", (E, A, B, C))
    begin("hostreduce")
    spaces = fit_basis.setdefault("spaces", {})
    banks, receipts = {}, []
    stop_after=snapshot.get('stop_after_model_q')
    if stop_after is not None and int(stop_after)!=0:
        raise ValueError('GATE shared_general_diagnostic: only stop_after_model_q=0 is supported')
    parents=[0] if stop_after is not None else [iq for iq in range(nq) if iq <= int(minus[iq])]
    position=0
    while position<len(parents):
        retained=sum(a.nbytes for space in spaces.values() for a in space.values())
        plan=general_pencil_batch_plan([(nl,nr)]*min(jax.process_count(),len(parents)-position),
            max_order=cap,max_workspace_bytes=workspace,retained_host_bytes=retained)
        ids=parents[position:position+plan['batch_size']]
        batch_start=time.monotonic()
        if jax.process_index()==0:
            print_fn(f"General shared-pole W [{label}] host batch parents={ids} "
                     f"shape={nl}x{nr} cap={cap} peak_bytes={plan['host_workspace_bytes']}")
            sys.stdout.flush()
        # All ranks execute EVERY slice/gather before the pure host owners
        # reduce their assigned parents concurrently. Never gather in a
        # rank-conditional lambda or mix independent host modal gauges.
        pencils=[(np.asarray(gather_to_host(E[iq])),np.asarray(gather_to_host(A[iq]))) for iq in ids]
        models=collective_general_pencil_batch(pencils,max_order=cap,max_workspace_bytes=workspace,
            rank_tolerance=float(recipe['bank_rule_tolerance']),
            frequency_scale_ry=moment_scale,coordinate_units=coordinate_units,
            fixed_spaces=[spaces.get(iq) for iq in ids],retained_host_bytes=retained)
        for batch_index,(iq,model) in enumerate(zip(ids,models)):
            spaces.setdefault(iq, model["spaces"])
            order = len(model["poles_ry"])
            carrier_order = extent(order)
            padding = carrier_order-order
            omap = np.pad(model["output_map"], ((0, 0), (0, padding)))
            imap = np.pad(model["input_map"], ((0, 0), (0, padding)))
            l = mm(C[iq:iq+1], put(omap[None], face))
            r = mm(B[iq:iq+1], put(imap[None], face), transa="C")
            p = np.pad(model["poles_ry"], (0, padding))
            partner = int(minus[iq])
            if partner == iq:
                l = tile_join((.5*l, -.5*jnp.conj(l)), -1, face)
                r = tile_join((r, jnp.conj(r)), -1, face)
                p = join_vectors((p, -p.conj()), panel_side)
            banks[iq] = (l, r, p)
            if partner != iq:
                banks[partner] = (-jnp.conj(l), jnp.conj(r), -p.conj())
            receipts.append(dict(q=iq, partner=partner, **model["receipt"]))
            growth=float(np.max(p.imag,initial=-np.inf))
            bad_growth=eta<=growth+1e-12 or float(np.min(sites.imag))<=growth+1e-12
            if bad_growth or stop_after is not None:
                if jax.process_index()==0:
                    print_fn(f'General shared-pole W [{label}] early diagnostic q={iq} '
                             f'order={order} growth_Ry={growth:.9g} eta_margin_Ry={eta-growth:.9g}')
                    sys.stdout.flush()
                # All reductions/gathers happen on EVERY rank before the
                # root-only I/O transaction. No microscopic factor is read
                # on the host: only O(K) norms and reduced-frequency arrays.
                ln=np.asarray(gather_to_host(jnp.sqrt(jnp.sum(abs(l)**2,axis=(-3,-2))))).reshape(-1)
                rn=np.asarray(gather_to_host(jnp.sqrt(jnp.sum(abs(r)**2,axis=(-3,-2))))).reshape(-1)
                sample_errors=[]
                for j,z in enumerate(sites):
                    actual=mm(l/put((z-p)[None,None]),r,transb='C')
                    target=values[j][iq:iq+1]
                    sample_errors.append(float(jnp.linalg.norm(actual-target)/
                        jnp.maximum(jnp.linalg.norm(target),1e-300)))
                moment_errors=[]
                for power,target in ((0,c1[iq:iq+1]),(1,c2[iq:iq+1])):
                    actual=mm(l*put((p**power)[None,None]),r,transb='C')
                    moment_errors.append(float(jnp.linalg.norm(actual-target)/
                        jnp.maximum(jnp.linalg.norm(target),1e-300)))
                root=Path(run_dir)/f'{label}_general_shared_pole'
                path=root/f'q_{iq:03d}_diagnostic.npz'
                report=dict(status='refused_laplace' if bad_growth else 'unadmitted_partial',
                    admitted=False,sigma_computed=False,q=iq,partner=partner,
                    reduced_shape=[nl,nr],model_order=order,bank_columns=len(p),
                    max_growth_ry=growth,eta_ry=eta,eta_margin_ry=eta-growth,
                    support_margin_ry=float(np.min(sites.imag))-growth,
                    scalar_tolerance=float(recipe['bank_rule_tolerance']),fit_tolerance=tolerance,
                    sample_errors=sample_errors,moment_errors=moment_errors,
                    direct_dyson_probes_file=f'q_{iq:03d}_direct_probes.json' if bad_growth else None,
                    model_receipt=model['receipt'],phase_seconds=phases,
                    elapsed_seconds=time.monotonic()-started,artifact=str(path),
                    full_microscopic_host_payload=False,n_mu=int(meta.n_rmu),n_q=nq,screening_bands=stop,
                    response_active_bands=[response_active.start,response_active.stop])
                report['verification_geometry']=verification_geometry
                report['frequency_coordinates']=frequency_coordinates
                arrays=dict(E=pencils[batch_index][0],A=pencils[batch_index][1],
                    poles_ry=model['poles_ry'],output_map=model['output_map'],input_map=model['input_map'],
                    space_right=model['spaces']['right'],space_left=model['spaces']['left'],
                    bank_poles_ry=p,left_column_norm=ln,right_column_norm=rn,residue_norm_bound=ln*rn,
                    energy_ry=energy,U_active=np.asarray(snapshot['U_active']),rho_active=np.asarray(snapshot['rho_active']),
                    sites_ry=sites,fit_positions=np.asarray(fit_positions,np.int32),
                    held_positions=np.asarray(held_positions,np.int32),
                    frequency_scale_ry=np.array(moment_scale),
                    left_coordinate_units=coordinate_units[0],right_coordinate_units=coordinate_units[1],
                    gamma_ry=np.array(gamma),eta_ry=np.array(eta))
                # Full-port actions contract ON DEVICE into frequency-only
                # Grams. Their host bound includes existing batch inputs,
                # canonical maps/spaces and the two optional held actions.
                selected=[j for j in held_positions if j>=len(fit_ids)+len(held_ids)][:2]
                if not selected:
                    selected=held_positions[:2]
                action_bytes=16*(nl*nl+nr*nr+len(selected)*nl*nr)
                base_bytes=retained+sum(e.nbytes+a.nbytes for e,a in pencils)
                base_bytes+=sum(sum(m[k].nbytes for k in ('poles_ry','output_map','input_map'))+
                                sum(a.nbytes for a in m['spaces'].values()) for m in models)
                report['frequency_capture_host_bytes']=base_bytes+action_bytes
                report['frequency_gram_scope']='unsymmetrized_reduced_transfer_CXB'
                report['bank_self_negative_mixture']=partner==iq
                if base_bytes+action_bytes<=workspace:
                    arrays['GC']=np.asarray(gather_to_host(mm(C[iq:iq+1],C[iq:iq+1],transa='C')))[0]
                    arrays['GB']=np.asarray(gather_to_host(mm(B[iq:iq+1],B[iq:iq+1],transb='C')))[0]
                    mixed=[];exact_norm=[]
                    for j in selected:
                        target=values[j][iq:iq+1]
                        mixed.append(np.asarray(gather_to_host(mm(C[iq:iq+1],
                            mm(target,B[iq:iq+1],transb='C'),transa='C')))[0])
                        exact_norm.append(float(jnp.sum(abs(target)**2)))
                    arrays.update(probe_sites_ry=sites[selected],probe_mixed_J=np.asarray(mixed),
                                  probe_exact_norm_sq=np.asarray(exact_norm))
                else:
                    report['frequency_action_grams_omitted']='explicit host workspace bound'
                def save_diagnostic():
                    root.mkdir(parents=True,exist_ok=True)
                    if path.exists():
                        raise FileExistsError(f'diagnostic artifact exists: {path}')
                    np.savez(path,**arrays)
                    path.with_suffix('.json').write_text(_json(report))
                rank0_transaction(path,stage='shared_general.frequency_diagnostic',write=save_diagnostic)
                # The reduced model is now durably committed BEFORE any new
                # microscopic probe can stall/refuse or exhaust the job cap.
                probes=[]
                if bad_growth and len(p):
                    suspect=p[int(np.argmax(p.imag))]
                    offset=max(eta,.001*abs(suspect))
                    probe_sites=[suspect+1j*offset,suspect+offset+1j*offset]
                    if stop_after is not None:
                        probe_sites.insert(0,suspect)
                        # All native matrices/identities remain face-sharded
                        # inside these scalar-returning diagnostic programs.
                        basic_metrics=jax.jit(lambda e,m: dielectric_physical_metrics(e,m,matmul=mm),
                            in_shardings=(face,rep),out_shardings=rep)
                        inverse_metrics=jax.jit(lambda e,m,b: dielectric_physical_metrics(e,m,matmul=mm,inverse=b),
                            in_shardings=(face,rep,face),out_shardings=rep)
                    for probe_index,z in enumerate(probe_sites):
                        probe_start=time.monotonic()
                        raw_probe=None
                        row=dict(z_ry=[float(z.real),float(z.imag)],at_fitted_pole=z==suspect,
                                 gamma_ry=gamma,additional_collision_eta_ry=0.)
                        if jax.process_index()==0:
                            print_fn(f'General shared-pole W [{label}] direct microscopic dielectric/Dyson probe '
                                     f'q={iq} z_Ry={z}; basic artifact committed')
                            sys.stdout.flush()
                        try:
                            for block in blocks:
                                left=((pm-ph) if block['name'][1]=='A' else pm)*jnp.asarray(block['u'])[:,None,None]
                                right=(ph if block['name'][0]=='A' else pm)*jnp.asarray(block['f'])[:,None,None]
                                prules=full_density_sample_rules(energy[:,:stop],np.array([z]),gamma_ry=gamma,
                                    tolerance=float(recipe['bank_rule_tolerance']),interval_ry=(block['lo'],block['hi']),paired=True)
                                for rule in prules:
                                    stream=full_density_response_stream(mesh_xy,grid,(nk,nb,nmu,ns),q_ids=[iq],
                                        n_outputs=1,layout=layout,qgrid=qgrid,rule_reference_ry=rule['reference_ry'],pair_bounds_ry=block['bounds'],
                                        pair_band_ranges=block['band_ranges'])
                                    part=stream(jnp.asarray(rule['times']),jnp.asarray(rule['weights']),left,pn,right,energy_d)[:,0]
                                    raw_probe=part if raw_probe is None else raw_probe+part
                            if stop_after is not None:
                                # No inverse or fitted-pole division precedes
                                # the durable exact dielectric norm witness.
                                h=H[iq:iq+1]
                                epsilon=dyson.dielectric_matrix(h,raw_probe)
                                mask=np.asarray(meta.mu_basis.active_mask,bool)
                                dimension=int(mask.sum())
                                basic={k:float(v) for k,v in basic_metrics(epsilon,jnp.asarray(mask)).items()}
                                basic.update(q=iq,z_ry=row['z_ry'],at_fitted_pole=z==suspect,
                                    logical_dimension=dimension,packed_dimension=nmu,gamma_ry=gamma,
                                    physical_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                                    scalar_rule_tolerance=float(recipe['bank_rule_tolerance']),
                                    prefactor=float(algebra['prefactor']),
                                    epsilon_definition='I-prefactor*H*P_raw(z)*H',
                                    additional_collision_eta_ry=0.,admitted=False,sigma_computed=False,
                                    elapsed_seconds=time.monotonic()-probe_start)
                                basic_path=root/f'q_{iq:03d}_dielectric_{probe_index:02d}_basic.json'
                                def save_basic():
                                    if basic_path.exists():raise FileExistsError(basic_path)
                                    basic_path.write_text(_json(basic))
                                rank0_transaction(basic_path,stage='shared_general.dielectric_basic',write=save_basic)
                                row['dielectric_basic_file']=basic_path.name
                                try:
                                    eps,eps_inverse=dyson.dielectric(h,raw_probe)
                                    metrics={k:float(v) for k,v in inverse_metrics(
                                        eps,jnp.asarray(mask),eps_inverse).items()}
                                    bounds=dielectric_inverse_bounds(metrics['epsilon_fro'],metrics['inverse_fro'],
                                        metrics['inverse_residual_fro'],dimension,
                                        padding_defect=max(metrics['padding_defect'],metrics['inverse_padding_defect']))
                                    inverse_record=dict(basic,**metrics,inverse_bounds=bounds,
                                        elapsed_seconds=time.monotonic()-probe_start)
                                except Exception as error:
                                    inverse_record=dict(basic,inverse_error=str(error),
                                        elapsed_seconds=time.monotonic()-probe_start)
                                inverse_path=root/f'q_{iq:03d}_dielectric_{probe_index:02d}_inverse.json'
                                def save_inverse():
                                    if inverse_path.exists():raise FileExistsError(inverse_path)
                                    inverse_path.write_text(_json(inverse_record))
                                rank0_transaction(inverse_path,stage='shared_general.dielectric_inverse',write=save_inverse)
                                row['dielectric_inverse_file']=inverse_path.name
                            if z!=suspect:
                                exact=dyson.value(H[iq:iq+1],raw_probe)
                                fitted=mm(l/put((z-p)[None,None]),r,transb='C')
                                row.update(exact_w_norm=float(jnp.linalg.norm(exact)),
                                    fitted_w_norm=float(jnp.linalg.norm(fitted)),relative_w_defect=float(jnp.linalg.norm(fitted-exact)/
                                        jnp.maximum(jnp.linalg.norm(exact),1e-300)))
                        except Exception as error:
                            row['error']=str(error)
                        row['elapsed_seconds']=time.monotonic()-probe_start
                        probes.append(row)
                    probe_path=root/f'q_{iq:03d}_direct_probes.json'
                    def save_probes():
                        if probe_path.exists():
                            raise FileExistsError(f'probe artifact exists: {probe_path}')
                        probe_path.write_text(_json(dict(admitted=False,sigma_computed=False,q=iq,probes=probes)))
                    rank0_transaction(probe_path,stage='shared_general.direct_probes',write=save_probes)
                if bad_growth:
                    raise ValueError(f'GATE shared_general_laplace: q={iq} growth={growth} not below eta/supports; '
                                     f'frequency-only diagnostic retained at {path}')
                raise RuntimeError(f'GATE shared_general_diagnostic_stop: unadmitted q={iq} partial model; '
                                   f'no Sigma computed; diagnostic retained at {path}')
        jax.block_until_ready([banks[iq][:2] for iq in ids])
        if jax.process_index()==0:
            print_fn(f"General shared-pole W [{label}] host batch complete parents={ids} "
                     f"orders={[len(m['poles_ry']) for m in models]} wall={time.monotonic()-batch_start:.3f}s")
            sys.stdout.flush()
        del pencils,models,model,omap,imap
        position+=len(ids)
    end("hostreduce", [b[:2] for b in banks.values()])
    # Every current full-matrix sample checks the FINAL bank, including the
    # mirrored partner and self-negative convex crossing mixture.
    begin("checks")
    checks, moment_checks = [], []
    for iq in range(nq):
        l, r, p = banks[iq]
        growth = float(np.max(p.imag, initial=-np.inf))
        if eta <= growth+1e-12 or np.min(sites.imag) <= growth+1e-12:
            raise ValueError(f"GATE shared_general_laplace: q={iq} growth={growth} not below eta/supports")
        for power, target in ((0, c1[iq:iq+1]), (1, c2[iq:iq+1])):
            actual = mm(l*put((p**power)[None, None]), r, transb="C")
            moment_checks.append(dict(q=iq, power=power,
                defect=float(jnp.linalg.norm(actual-target)/jnp.maximum(jnp.linalg.norm(target), 1e-300))))
        for j, z in enumerate(sites):
            got = mm(l/put((z-p)[None, None]), r, transb="C")
            target = values[j][iq:iq+1]
            error = float(jnp.linalg.norm(got-target)/jnp.maximum(jnp.linalg.norm(target), 1e-300))
            checks.append(dict(q=iq, support=j, held=j in held_positions, error=error))
            if not np.isfinite(error) or error > tolerance:
                raise ValueError(f"GATE shared_general_held_w: q={iq} support={j} relative error {error:.6g}>{tolerance}")
    end("checks")
    begin("persist")
    identity = dict(iteration_id=str(label), energies=hashlib.sha256(energy.tobytes()).hexdigest(),
        occupations=hashlib.sha256(rho.tobytes()).hexdigest(),
        hamiltonian=hashlib.sha256((energy.tobytes()+np.asarray(snapshot["U_active"]).tobytes())).hexdigest(),
        wavefunctions=operator_digest(merge_spin_centroid(pm, 1, 2), mesh_xy, cache=False),
        centroids=hashlib.sha256(np.asarray(meta.mu_basis.canonical_indices).tobytes()).hexdigest())
    if qgrid != grid:
        identity.update(electronic_grid=list(grid), response_grid=list(qgrid))
    tables = _shared_pole_tables(meta, sym, meta.mu_basis.canonical_indices)
    root = Path(run_dir)/f"{label}_general_shared_pole"
    if jax.process_index() == 0:
        root.mkdir(parents=True, exist_ok=True)
    path = root/"model.h5"
    # All physical banks are known before persistence. Compact only exact
    # unused file columns; never discard a fitted pole to meet storage cost.
    kmax = extent(max(len(bank[2]) for bank in banks.values()))
    for iq in range(nq):
        l, r, p = banks.pop(iq)
        header = write_general_shared_pole_model(str(path), l[:, :, None], r[:, :, None],
            put(p[None]), [len(p)], q_span=(iq, iq+1), meta=current_meta, tables=tables,
            recipe=recipe, receipts=dict(identity=identity), kmax=kmax)
    end("persist")
    receipt = dict(construction="full-microscopic-general-shared-pole", n_mu=int(meta.n_rmu),
        n_q=nq, n_k=nk, electronic_grid=list(grid), response_grid=list(qgrid),
        screening_bands=stop, poles=receipts, checks=checks, moment_checks=moment_checks, sample_seconds=sample_wall,
        seconds=time.monotonic()-started, response_rules=[{k:v for k,v in r.items() if k not in ("times", "weights")}
                                                       for r in rules],
        input_density_coherence_norm=float(np.linalg.norm(rho-np.eye(nb)[None]*rho.diagonal(axis1=1, axis2=2)[:, :, None])),
        algebra=algebra, spatial_projection=False, screening_density="full_matrix",
        reference_statistics="retarded_reference", general_gates=general_gates)
    receipt["capacity"] = price
    receipt['verification_geometry']=verification_geometry
    receipt['frequency_coordinates']=frequency_coordinates
    receipt['response_active_bands'] = [response_active.start,response_active.stop]
    receipt['response_blocks'] = [dict(name=b['name'],interval_ry=[b['lo'],b['hi']],
                                      endpoint_bounds_ry=b['bounds'],
                                      structural_band_ranges=b['band_ranges'].tolist()) for b in blocks]
    receipt['quadrature_work'] = dict(ordered_green_pairs=sum(len(r['times']) for r in rules),
        product_greens_per_node=2,output_node_updates=sum(len(r['times'])*len(r['members']) for r in rules),
        largest_output_count=max(len(r['members']) for r in rules),
        rule_groups=len(rules),scalar_provider='exact_endpoint_blocks_public_minimax_owners')
    receipt["phase_seconds"] = phases
    receipt["physical_grid_order"] = grid_receipt
    receipt["native_positive_pole_budget"] = positive_cap
    receipt["general_signed_pole_budget"] = int(cap)
    receipt["stored_Kmax"] = int(kmax)
    receipt["stored_payload_bytes"] = 32*nq*int(meta.n_rmu)*kmax+16*nq*kmax
    receipt["reduced_host_pencil_shape"] = [nl, nr]
    receipt["reduced_host_workspace_bytes_per_rank"] = host_price
    receipt["held_reduced_host_basis_bytes_per_rank"] = sum(
        a.nbytes for space in spaces.values() for a in space.values())
    retained = sum(a.size*a.dtype.itemsize for a in directions.values())
    kept = current_meta.shared_pole_capacity.reserve("general_held_numeric_basis",
        resident_bytes_per_rank=max(0, retained-previous_direction_bytes)//ranks,
        workspace_bytes_per_rank=0)
    current_meta.shared_pole_capacity.live_stages = (kept["stage"],)
    receipt["held_direction_bytes_per_rank"] = retained//ranks
    if jax.process_index() == 0:
        (root/"receipt.json").write_text(_json(receipt))
        print_fn(f"General shared-pole W: {nq}q, {meta.n_rmu}mu, {stop}bands; {receipt['seconds']:.3f}s")
    return dict(shared_pole=dict(path=str(path), identity=identity, digest=header["digest"],
                                 K=header["K"], header=header, representation="ordered_complex"),
                meta=current_meta, receipt=receipt)


def _print_eigh_stack_routes(print_fn):
    """Record the route distrib_la chose for each new eigh stack shape."""
    import distrib_la
    from lxkit import memory
    for line in distrib_la.new_stack_routes():
        stack, decision = line.split(": ", 1)
        route, _, reason = decision.partition(" ")
        memory.record_route(f"distrib_la {stack}", route.rstrip(","), reason.strip(" ,"))


def _json(value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, complex):
            return dict(real=x.real, imag=x.imag)
        from file_io.shared_pole_store import ResidentSectorModel
        if isinstance(x, ResidentSectorModel):
            return str(x)
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=encode, sort_keys=True, allow_nan=False)


def _authenticated_constructor_resume(root, identity, recipe, *, photon=False, coulomb_sha256=None):
    """Authenticate a complete producer bank before preserving it.

    This is deliberately narrower than restart: it resumes only the missing
    constructor after the producer receipts and store validator bind the exact
    current map identity, resolved recipe, response convention and final commit.
    Photon moments are in the bank receipt; scalar moments have a second receipt,
    and both bind the bare V digest ``coulomb_sha256`` (``_coulomb_resource``).
    Any other partial directory follows the existing remove-and-rebuild path,
    including a bank built against another V digest (another P, older code).
    """
    from file_io.shared_pole_store import validate_shared_pole_bank

    receipt_paths = ((root / 'bank_receipt.json',) if photon else
                     (root / 'bank_receipt.json', root / 'moments_receipt.json'))
    bank_path = root / 'bank.h5'
    required = (*receipt_paths, bank_path)
    if not all(path.is_file() for path in required):
        return False
    try:
        bank_receipt = json.loads(receipt_paths[0].read_text())
        moments_receipt = None if photon else json.loads(receipt_paths[1].read_text())
    except (OSError, ValueError):
        return False
    if bank_receipt.get('identity') != identity or bank_receipt.get('completion') is not True:
        return False
    if photon:
        if (bank_receipt.get('stage') != 'photon'
                or bank_receipt.get('bank_complete') is not True):
            return False
    elif (moments_receipt.get('identity') != identity
          or moments_receipt.get('completion') is not True
          or moments_receipt.get('bank_complete') is not True
          or bank_receipt.get('coulomb_identity') != moments_receipt.get('coulomb_identity')
          or (bank_receipt.get('coulomb_identity') or {}).get('sha256') != coulomb_sha256):
        return False
    try:
        header = validate_shared_pole_bank(
            bank_path, expected_identity=identity, mesh_xy=None,
            require_complete=True, expected_recipe=recipe, collective=False)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return ('photon_layout' in header) == photon


# What the sector constructor publishes into a map directory.
_SECTOR_OUTPUTS = ('sectors.json', 'CC.h5', 'TT.h5', 'CT_C.h5', 'CT_T.h5')


def _published_sector_handle(root, identity):
    """The sector handle an interrupted run published for this map, or None.

    A run killed after its constructor leaves ``sectors.json``. Its handle is
    reused when the manifest binds this map's identity (label, WFN, energies,
    occupations, centroids, recipe_hash) and every model and the constant it
    names is a file in this directory; Sigma's manifest validation then
    authenticates each digest on every rank. Models that were device-resident
    died with the run, so that manifest gives None and the caller rebuilds. A
    manifest of another identity refuses: it is not this map's private state.
    """
    manifest = root / 'sectors.json'
    try:
        header = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return None
    if header.get('identity') != identity:
        raise ValueError(f'GATE shared_pole_output: sector manifest at {manifest} binds another '
                         'map, recipe or WFN; use a fresh run directory')
    paths = [row.get('path') for row in header.get('sectors', {}).values()]
    paths.append(header.get('constant', {}).get('path'))
    own = root.resolve()
    if len(paths) != 5 or not all(isinstance(p, str) and Path(p).parent == own
                                  and Path(p).is_file() for p in paths):
        return None
    return dict(path=str(manifest.resolve()), identity=header['identity'],
                digest=header['digest'], representation=header['representation'],
                sectors=header['sectors'], constant=header['constant'])


def _mark_photon_head(handle, config):
    """Name the direct four-current Γ head on a photon handle when heads are on."""
    from .gw_config import HeadCorrection, uses_direct_bispinor_shared_pole_head
    if config.head.correction is HeadCorrection.OFF:
        return
    if not uses_direct_bispinor_shared_pole_head(config):
        raise ValueError("GATE shared_pole_photon_head: unsupported photon Γ policy")
    handle["direct_photon_head"] = "first_order_cc_ct_tc_tt"


def shared_pole_identity(wfns, meta, *, label, wfn, binding, centroid_indices):
    """Bind logical current energies/occupations and their wavefunction source.

    SC supplies the current Hamiltonian/rotation identity from its owner;
    a DFT fingerprint alone cannot authenticate rotated wavefunctions.
    Array hashes here cover only replicated small energy/occupation tables.
    """
    from common.parallel_transport import fingerprint_from_binding, wfn_fingerprint
    from .response_bank import response_weights
    census = response_weights(wfns, meta)[-1]
    source = wfn_fingerprint(wfn) if binding is None else fingerprint_from_binding(binding, wfn)
    state = getattr(meta, "shared_pole_state_identity", None)
    if str(label).startswith("sc_"):
        if not isinstance(state, dict) or any(not isinstance(state.get(k), str)
                or not state[k] for k in ("hamiltonian", "wavefunctions")):
            raise ValueError("GATE shared_pole_state_identity: current SC Hamiltonian/rotation identity missing")
    else:
        state = dict(wavefunctions=source, hamiltonian=hashlib.sha256(
            (source + census["energy_sha256"]).encode()).hexdigest())
    recipe = meta.shared_pole_recipe
    identity = dict(iteration_id=str(label), hamiltonian=state["hamiltonian"],
        wavefunctions=state["wavefunctions"], energies=census["energy_sha256"],
        occupations=census["occupation_sha256"],
        centroids=hashlib.sha256(np.asarray(centroid_indices, np.int64).tobytes()).hexdigest(),
        recipe_hash=recipe["recipe_hash"], gate_hash=recipe["gate_hash"])
    if str(label).startswith("sc_"):
        # The SC state labels do not name the WFN (bind_shared_pole_sc_identity),
        # so an SC map's bank, model and sector manifest bind the source WFN's
        # fingerprint here (the dipole provenance's): a rerun reuses them only
        # on the same WFN, never on energies alone.
        identity["wfn"] = source
    return identity


def _coulomb_resource(value, meta, sym, mesh_xy):
    """The bare V parents' resource: the on-device operator, named by its token and digest.

    ``value`` is the packed bare V (a ``QirrOperator`` on the run's q wedge,
    or a full-q face [Q,mu_p,mu_p]), Ry; its wedge rows are the parents.  The
    response owner roots the parents straight from this operator
    (``response_bank._coulomb_roots``); no copy is staged on disk (54.6 GB and
    41 s at Ni 20^3 when it was).  The identity is a device digest of the
    packed parents, the same on every rank, so a bank's receipts and a
    constructor resume bind the operator's values.  The packed layout follows
    the mesh, so a bank built at another P does not match and is rebuilt.
    """
    from symmetry_maps import QirrOperator
    from .response_bank import _operator_token, operator_digest
    basis = meta.mu_basis
    qids = np.asarray(sym.q_irr_full_idx, np.int64)
    op = QirrOperator.of(value)
    nq = int(np.prod(getattr(meta, "qgrid", None) or (meta.nkx, meta.nky, meta.nkz)))
    if op.n_full != nq or tuple(op.values.shape[1:]) != (basis.n_packed, basis.n_packed):
        raise ValueError("GATE shared_pole_coulomb: expected the packed bare V")
    # The parents' rows are one array for the run (the wedge's own values, or
    # one gather of a whole-zone operator), so the held roots and the digest
    # are reused by every SC map.
    held = _PARENT_ROWS.get("V")
    if held is None or held[0] is not op.values or held[1] != tuple(qids.tolist()):
        held = _PARENT_ROWS["V"] = (op.values, tuple(qids.tolist()), op.at_rows(qids))
    rows = held[2]
    return dict(path=None, dataset=None, basis="canonical", q_irr_full_idx=qids.tolist(),
                sha256=operator_digest(rows, mesh_xy), operator=_operator_token(rows))


#: The bare V of this run and its parent rows (one entry).
_PARENT_ROWS: dict = {}


def _bank_residence(meta, config, *, mesh_xy, sym, root, label, photon, mu_bases=None):
    """Keep this map's bank on the devices when it fits, else in host memory.

    The bank is written once and read back once by the constructor; the file
    exists only because the producer is frequency-major and the constructor
    parent-major. It stays resident when (1) the resident payload R and one
    complete read copy fit in half of the available device budget (R4's
    q-local margin), and (2) the constructor's own route admission, with R
    live, still selects local parents, so residency never changes the route.
    ``photon`` is the photon layout of a full bispinor bank (else None); its
    sector constructor route must likewise be unchanged. Otherwise the payload
    goes to the host tier when it takes at most half of this process's
    host budget (``host_bytes_per_process``), else to one file per rank
    (both SlabIO's streamed tier); the devices then hold only the span being
    read. The shared scratch file is left for a W export (``write_w`` re-reads
    the bank after the constructor), a requested distributed layout, and a
    per-rank file the disk or quota refuses on any rank.
    Returns ``(payload or None, receipt)``; a device payload's R is reserved
    (``receipt["stage"]``) and the caller keeps it in ``ledger.live_stages``
    until it is released.
    """
    from common.gpu_utils import host_bytes_per_process
    from file_io.shared_pole_store import ResidentBankPayload, shared_pole_bank_payload_bytes
    from .gw_config import linalg_resolution
    from .shared_pole_constructor import constructor_route

    from .shared_pole_directions import line_width_bounds
    ledger = meta.shared_pole_capacity
    ordered = bool(photon) or not bool(sym.trs_allowed)
    nq = len(np.asarray(sym.q_irr_full_idx))
    R = shared_pole_bank_payload_bytes(meta, recipe=meta.shared_pole_recipe,
                                       ordered=ordered, nq=nq, mesh_xy=mesh_xy,
                                       line_widths=line_width_bounds(meta, mesh_xy=mesh_xy,
                                           photon_bases=None if photon is None else mu_bases),
                                       photon_extent=None if photon is None else photon.packed_extent,
                                       photon_bases=None if photon is None else mu_bases)
    receipt = dict(residence="file", payload_bytes_per_rank=R,
                   payload_bytes_total=R * int(mesh_xy.size))
    if config.debug.write_w or linalg_resolution(
            {"linalg": config.backend.linalg}).layout != "local":
        receipt["reason"] = "write_w export or distributed layout"
        return None, receipt
    carrier = meta.mu_basis.n_canonical if photon is None else photon.packed_extent
    bank_label = str(root / "bank.h5")

    def host(reason):
        receipt["half_host_budget_bytes_per_rank"] = int(host_bytes_per_process()) // 2
        if R > receipt["half_host_budget_bytes_per_rank"]:
            # Read once by the same ranks in the same layout: one file per rank
            # (SlabIO's streamed tier), not the shared bank.h5.
            receipt.update(residence="rank_file", reason=reason + "; payload exceeds half the "
                           "host budget; per-rank files")
            return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label,
                                       memory_kind="file", root=root), receipt
        receipt.update(residence="host", reason=reason + "; host tier")
        return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label,
                                   memory_kind="host", root=root), receipt

    both = ledger.preview(resident_bytes_per_rank=2 * R, workspace_bytes_per_rank=0,
                          concurrent_with=())
    receipt["half_budget_bytes_per_rank"] = both["available_device_bytes_per_rank"] // 2
    if both["aggregate_bytes_per_rank"] > receipt["half_budget_bytes_per_rank"]:
        return host("payload and one read copy exceed half the device budget")
    if photon is not None:
        from .shared_pole_sectors import sector_execution
        route = lambda upstream: sector_execution(meta, config, mu_bases, nq,
                                                  mesh_xy=mesh_xy, upstream=upstream)[0]
        without = route(())
    stage = f"bank_resident.{label}"
    ledger.reserve(stage, resident_bytes_per_rank=R, workspace_bytes_per_rank=0,
                   concurrent_with=())
    if photon is not None:
        # The sector constructor must keep the layout it would have chosen
        # without the payload live, so residency never changes the route.
        execution, wanted = route((stage,)), without
    else:
        execution, wanted = constructor_route(
            meta, config, meta.shared_pole_recipe, mesh_xy=mesh_xy, ledger=ledger,
            upstream=(stage,), ordered=ordered, odd_moments=ordered, nq=nq)[0], "local"
    if execution != wanted:
        return host("constructor would change route with the payload live")
    receipt.update(residence="device", stage=stage,
                   reason="payload, one read copy and the unchanged constructor route fit")
    return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label), receipt


#: The device-resident scalar model of this process's latest SC map (at most
#: one). It lives from its constructor through the map's head and Sigma and,
#: on the accepted final map, the W0 persist; the next map's entry
#: (``sc_iteration.gw_iteration_map``, before the wavefunction rotation and
#: the Hartree rebuild, which no planner prices it against) or the end of the
#: SC run releases it (:func:`release_resident_model`).
_RESIDENT_MODEL: list = []


def release_resident_model():
    """Release the latest SC map's device-resident scalar or CC sector model, if any."""
    while _RESIDENT_MODEL:
        _RESIDENT_MODEL.pop().release()


def hold_resident_model(model, meta, nbytes, *, stage):
    """Keep an SC map's resident CC sector model live, as the scalar model is.

    ``mpa.sector_sigma.compute_sector_sigma`` releases TT and CT after Sigma
    and hands CC here: the accepted final map's W0 persist reads it
    (``mpa.sector_sigma.sector_static_wc``).  Its ``nbytes`` stay reserved
    as ``stage``; :func:`release_resident_model` releases it.
    """
    ledger = meta.shared_pole_capacity
    ledger.reserve(stage, resident_bytes_per_rank=int(nbytes),
                   workspace_bytes_per_rank=0, concurrent_with=ledger.live_stages)
    ledger.live_stages = (*ledger.live_stages, stage)
    _RESIDENT_MODEL.append(model)


def _scalar_model_residence(meta, nq, width, *, mesh_xy, root, identity):
    """Keep an SC map's scalar model on the devices when it fits.

    The model file of an SC map has three readers in the same process: the
    head, Sigma and, on the accepted final map, the W0 persist
    (``gw.mpa.sigma.shared_pole_static_wc``); no later run reads it (SC
    scratch never serves a restart). It stays resident under the sector
    models' admission (``shared_pole_store.admit_resident_model``): R, its
    bytes at the stored K extent's bound (``model_column_bound`` of the
    constructor's column width), and one copy fit in half of the device
    budget beside the live stages. The constructor has finished when this is
    asked, so its route cannot change; Sigma and the W0 persist price R, so
    their results match the file route bit for bit where their panel schedule
    is unchanged. Otherwise model.h5 is written as before. Returns ``(model or None, receipt)``; a resident R is reserved as
    ``receipt["stage"]`` and the model is held for :func:`release_resident_model`.
    """
    from file_io.shared_pole_store import (ResidentSectorModel, admit_resident_model,
                                           model_column_bound)
    ledger = meta.shared_pole_capacity
    R = ResidentSectorModel.payload_bytes(mesh_xy, nq, meta.mu_basis.n_canonical,
                                          model_column_bound(meta, width))
    receipt = admit_resident_model(
        ledger, R, f"scalar_model.{identity['iteration_id']}.{len(ledger.entries)}",
        ledger.live_stages)
    if receipt["residence"] != "device":
        return None, receipt
    model = ResidentSectorModel(mesh_xy, label=str(root / "model.h5"))
    _RESIDENT_MODEL.append(model)
    return model, receipt


def _release_bank_file(path):
    """Unlink this run's link to a committed map's file-tier bank.

    At production shape the bank is the run's largest file:
    N_q (2 N_dense + N_moments) N_mu^2 16 B plus the line panels, 1.26e12 B at
    Fe 20^3 (1062 parents, 1796 centroids). A resume directory's bank may be
    hard-linked from another attempt; only this link goes, and the link count
    is returned with the bytes so the receipt says whether space was freed.
    Returns ``None`` when there is no file.  Rank 0 alone looks: a per-rank
    ``exists()`` raced its unlink, and a late rank skipped the release while
    the others waited in it (the Na SC hang on a cold cache, CPU P4).
    """
    import os
    from common.collectives import rank0_transaction

    def unlink():
        if not os.path.exists(path):
            return None
        stat = os.stat(path)
        os.unlink(path)
        return dict(bytes=int(stat.st_size), links=int(stat.st_nlink))
    return rank0_transaction(path, stage="shared_pole.bank.release", write=unlink,
                             return_value=True)


#: The per-map scratch generation ``screen_shared_poles`` creates under an SC
#: label (bank, Coulomb staging, constant and receipts).
_MANAGED_SCRATCH = r"sc_[0-9]{4}_shared_pole"
#: Set once this process has swept earlier processes' streamed stores (prepare_output).
_SWEPT: list = []


def retain_iteration_scratch(run_dir, label, *, print_fn=print):
    """Collectively keep only map ``label``'s shared-pole scratch generation.

    Each SC map screens into its own ``sc_NNNN_shared_pole/``, and a file-tier
    bank there is the payload of ``shared_pole_bank_payload_bytes`` (about
    0.27 TB on Fe 8^3), so without this the scratch grows linearly in maps.  The caller invokes it
    only after the current map's model has been built, consumed by Sigma and
    passed the Sigma gates, so a failure keeps the last usable generation, and
    the current one survives convergence for constructor resume.  Only exact
    managed names are eligible; scanning rather than removing only ``N-1``
    also clears stale later maps of a longer earlier run.  This is
    ``gw.mpa.model.retain_iteration_artifacts``'s rule, through the same
    removal owner.  Run-lifetime artifacts (the photon route's static
    reference) live beside the generations, never inside one.
    """
    import os
    from .qsgw_utils import remove_managed

    root = os.path.abspath(os.fspath(run_dir))
    removed = remove_managed(
        root, _MANAGED_SCRATCH, keep=[os.path.join(root, f"{label}_shared_pole")],
        barrier_tag=f"shared_pole.scratch.retain.{label}", print_fn=print_fn)
    if removed:
        print_fn(f"  shared-pole scratch: retained {label}; discarded: "
                 + ", ".join(sorted(removed)))
    return tuple(sorted(removed))


def _shared_pole_tables(meta, sym, centroid_indices):
    """Build raw-parent tables through the canonical symmetry service."""
    from symmetry_maps import (QirrTables, centroid_source_map_and_wrap,
                               bgw_integer_q_to_fractional)
    perm, wraps = centroid_source_map_and_wrap(
        np.asarray(centroid_indices), sym.sym_matrices, sym.translations,
        np.asarray(meta.fft_grid), extend_trs=True)
    qt = QirrTables(irr_idx_q=sym.irr_idx_q, sym_idx_q=sym.sym_idx_q,
        q_irr_frac=bgw_integer_q_to_fractional(
            sym.q_irr_kgrid_int, getattr(meta, "qgrid", None) or (meta.nkx, meta.nky, meta.nkz)),
        sym_perm=perm, L_table=wraps, n_sym_spatial=len(sym.sym_matrices))
    return dict(qirr=qt, q_irr_full_idx=np.asarray(sym.q_irr_full_idx, np.int64), sym=sym)


def compile_w_model_ahead(config, meta, sym, mesh_xy):
    """Submit the scalar shared-pole W model's shape-fixed compiles before the zeta fit.

    H = V^(1/2) of the bare V's parents takes its shape from the centroid
    basis and the q wedge, and the scalar head's pole fit from the deck's
    ``n_poles``; both compile on helper threads
    (``common.jax_compile_cache.compile_ahead``) while the zeta fit and V run,
    instead of when the bank and the head first need them.
    """
    from .gw_config import HeadCorrection, uses_full_bispinor_shared_pole
    if config.sigma.w_model != "shared_pole" or uses_full_bispinor_shared_pole(config):
        return
    sym = getattr(meta, "q_sym", sym)
    from .response_bank import compile_coulomb_roots_ahead
    compile_coulomb_roots_ahead(meta, config, mesh_xy, len(sym.q_irr_full_idx))
    if config.head.correction is not HeadCorrection.OFF:
        from .mpa.fit_driver import compile_scalar_fit_ahead
        compile_scalar_fit_ahead(config.mpa.n_poles, config.mpa.pole_solver)


def screen_shared_poles(wfns, V_q, meta, config, *, mesh_xy, sym,
                        centroid_indices, run_dir, label, wfn,
                        wfn_fingerprint_binding, tensors_filename, occupation_state, print_fn,
                        head_resolver=None, mpa_plan=None, iteration_head_response=None,
                        material_class=None, wfns_transverse=None,
                        bispinor_v_q_path=None, mu_bases=None,
                        photon_g0_vectors=None, photon_head_cache=None,
                        photon_head_state=None):
    """Build current W; only one-shot models may use ISDF restart membership.

    SC labels own separate map scratch. ``restart`` may restore the invariant
    ISDF basis, but never skips the current response or W construction.
    """
    sym = getattr(meta, "q_sym", sym)
    source_wfn = None
    from .gw_config import uses_full_bispinor_shared_pole
    photon = uses_full_bispinor_shared_pole(config)
    if photon and int(meta.nspinor) != 4:
        raise ValueError("GATE shared_pole_sectors: full_shared_pole requires four-spinor metadata")
    photon_layout = None
    if photon:
        from .photon_layout import PhotonBasisLayout
        if wfns_transverse is None or bispinor_v_q_path is None or mu_bases is None:
            raise ValueError("GATE shared_pole_sectors: both current-map endpoint bundles and photon V are required")
        photon_layout = PhotonBasisLayout.from_centroid_extents(
            mu_bases[0].n_logical, mu_bases[1].n_logical, mesh_xy)
    if config.debug.write_w or config.write_poles:
        source_wfn = getattr(wfn, "path", None)
        if source_wfn is None or not str(source_wfn).strip():
            raise ValueError("GATE shared_pole_output: write_w/write_poles require the source WFN path before screening")
    # A time-reversal-broken metal runs on this route (METAL 2026-09-16): the
    # ordered bank keeps both particle-hole orientations with the odd channel
    # (8d3ad5fe) and its kernels return the physical orientation FT_q[chi]
    # (86307cff). GATE mpa_ordered_metal stays on the MPA route, which fits one
    # residue with no odd channel (gw.mpa.model._require_metal_time_reversal).
    if material_class == "metal" and not bool(getattr(sym, "trs_allowed", True)):
        print_fn("  shared-pole screening: time-reversal-broken METAL; ordered bank "
                 "(both orientations, odd channel) with fractional occupations")
    from .ppm_pipeline import observing_window_source, publish_window_source
    if not photon and observing_window_source():
        # The held inputs of TD-aGW's shared-pole refit (this stage re-run at the
        # window's populations, tdagw.dynsig.spm_screening); one empty list test
        # when no ppm_pipeline.observe_window_sigma_source block is open.
        publish_window_source(screening_inputs=dict(
            wfns=wfns, V_q=V_q, meta=meta, config=config, kwargs=dict(
                mesh_xy=mesh_xy, sym=sym, centroid_indices=centroid_indices, run_dir=run_dir,
                label=label, wfn=wfn, wfn_fingerprint_binding=wfn_fingerprint_binding,
                tensors_filename=tensors_filename, occupation_state=occupation_state,
                print_fn=print_fn, head_resolver=head_resolver, mpa_plan=mpa_plan,
                iteration_head_response=iteration_head_response,
                material_class=material_class)))
    with timing.section("spole.screening_setup"):
        from file_io.shared_pole_store import initialize_shared_pole_bank
        from file_io.tagged_arrays import register_shared_pole_restart_member
        from .shared_pole_recipe import shared_pole_restart_handle
        from .shared_pole_constructor import construct_shared_poles
        from .w_isdf import produce_w_bank, compute_response_moments

        started = time.monotonic()
        recipe, ledger = meta.shared_pole_recipe, meta.shared_pole_capacity
        # This is the top-level map boundary. All upstream V/psi carriers are
        # inherited; no newly allocated bank/constructor arrays exist yet.
        # (The SC map entry already released the previous map's resident model.)
        ledger.live_stages = ()
        if occupation_state is not None:
            from common.collectives import replicate_to_mesh
            occupations = np.asarray(occupation_state.f_kn, np.float64)
            if occupations.shape != wfns.occ.shape:
                raise ValueError("GATE shared_pole_occupations: supplied current state does not match carrier")
            wfns = dataclasses.replace(wfns, occ=replicate_to_mesh(occupations, mesh_xy))
            if photon:
                wfns_transverse = dataclasses.replace(wfns_transverse, occ=wfns.occ)
        identity = shared_pole_identity(wfns, meta, label=label, wfn=wfn,
            binding=wfn_fingerprint_binding, centroid_indices=centroid_indices)
        sc_scratch = str(label).startswith("sc_")
        if config.restart and tensors_filename is not None and not sc_scratch and not photon:
            handle = shared_pole_restart_handle(tensors_filename,
                expected_identity=identity, meta=meta, mesh_xy=mesh_xy, print_fn=print_fn)
            if handle is not None:
                result = dict(shared_pole=handle)
                from .gw_config import HeadCorrection
                if config.head.correction is not HeadCorrection.OFF:
                    from .shared_pole_head import build_shared_pole_head
                    head, iteration_head = build_shared_pole_head(
                        handle, None, V_q, wfns, meta, config, mesh_xy=mesh_xy, wfn=wfn,
                        response=iteration_head_response, head_resolver=head_resolver,
                        plan=mpa_plan, material_class=material_class, occupation_state=occupation_state)
                    result.update(mpa_head=head, iteration_head=iteration_head)
                # The export is written only once the map is complete, so an
                # export on disk always implies a finished head.  Under SC
                # this branch is unreachable (``sc_scratch`` skips restart).
                if config.debug.write_w or config.write_poles:
                    from file_io.shared_pole_store import export_shared_pole_outputs
                    with timing.section("spole.outputs"):
                        export_shared_pole_outputs(handle, meta=meta, config=config,
                            mesh_xy=mesh_xy, source_wfn=source_wfn,
                            run_dir=run_dir, label=label, print_fn=print_fn,
                            tables=_shared_pole_tables(meta, sym, centroid_indices))
                return result
        root = Path(run_dir).resolve() / (str(label) + "_shared_pole")
        # The bare V parents (a collective device digest, before the rank-0 resume
        # decision): a complete bank resumes only when built against these values.
        coulomb = None if photon else _coulomb_resource(V_q, meta, sym, mesh_xy)
        from common.collectives import rank0_transaction
        from file_io.commit_state import assert_committed

        def prepare_output():
            # Only rank zero reads the small completion marker, on the compute
            # node. The transaction owner broadcasts any refusal to every rank.
            # An interrupted run's directory keeps what authenticates for this
            # map and loses the rest: a published sector handle is reused, a
            # complete bank resumes the constructor, anything else is rebuilt.
            import h5py
            from file_io.slab_io import remove_stale_streamed_banks
            # A per-rank streamed store is unlinked as soon as it is opened, so its bytes
            # die with the process, SIGKILL included; only older code, or a kill between
            # open and unlink, leaves files, so the first map of a process removes every
            # generation's leftovers before it creates any.
            stale = remove_stale_streamed_banks(
                [root / "streamed_bank"] if _SWEPT else
                [p / "streamed_bank" for p in Path(run_dir).resolve().glob("*_shared_pole")])
            _SWEPT.append(True)
            if stale:
                print_fn("WARNING shared-pole output: removed streamed stores left by an earlier "
                         "process: " + " ".join(stale))
            model = root / "model.h5"
            if photon and (root / 'sectors.json').exists():
                if _published_sector_handle(root, identity) is not None:
                    print_fn(f"shared-pole output: authenticated sector manifest retained at {root}; reusing it")
                    return "published"
            complete, bound = False, None
            if model.exists():
                try:
                    with h5py.File(model, "r") as h5:
                        assert_committed(h5, path=model)
                        complete = "final_commit" in h5
                        raw = h5["header_json"][()] if complete else b"{}"
                    bound = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw)).get("identity")
                except (OSError, ValueError, KeyError, AttributeError):
                    complete = False
            if complete and sc_scratch and not photon and bound == identity:
                # An SC map's committed model of this very map (the file tier
                # persists it): Sigma reads it; nothing upstream is rebuilt.
                print_fn(f"shared-pole output: committed model of this map retained at {model}; reusing it")
                return "committed"
            if complete:
                print_fn(f"shared-pole output: complete model retained at {model}; refusing rebuild")
                raise ValueError(f"GATE shared_pole_output: complete model {model}; use its compatible restart member or a fresh run directory")
            if _authenticated_constructor_resume(root, identity, recipe, photon=photon,
                                                 coulomb_sha256=None if photon else coulomb['sha256']):
                print_fn(f"shared-pole output: authenticated complete bank retained at {root}; resuming constructor")
                stale = [name for name in (_SECTOR_OUTPUTS if photon else ())
                         if (root / name).exists()]
                for name in stale:
                    (root / name).unlink()
                if stale:
                    print_fn(f"WARNING shared-pole output: removed {' '.join(stale)} from {root}; "
                             "rebuilding this map's constructor")
                return True
            if root.exists():
                removed = ' '.join(sorted(path.name for path in root.iterdir()))
                print_fn(f"WARNING shared-pole output: removed partial directory {root} "
                         f"({removed}); rebuilding this map")
                shutil.rmtree(root)
            else:
                print_fn(f"shared-pole output: creating new directory {root}")
            root.mkdir(parents=True)
            return False

        resume_constructor = rank0_transaction(
            root, stage="shared_pole.prepare_output", write=prepare_output,
            return_value=True)
        if resume_constructor == "published":
            from common.collectives import agree_io_error
            handle, error = None, None
            try:
                handle = _published_sector_handle(root, identity)
                if handle is None:
                    raise ValueError(f"GATE shared_pole_output: sector manifest at {root} changed during reuse")
            except (OSError, ValueError) as exc:
                error = exc
            agree_io_error(error, path=root / 'sectors.json', stage='shared_pole.published_handle')
            _mark_photon_head(handle, config)
            return dict(shared_pole=handle)
        tables = _shared_pole_tables(meta, sym, centroid_indices)
    if resume_constructor == "committed":
        coulomb = None
    with timing.section("spole.bank_setup"):
        resident, residence = (None, dict(residence="file", reason=(
            "committed model reused" if resume_constructor == "committed" else "authenticated resume")))
        if not resume_constructor:
            resident, residence = _bank_residence(meta, config, mesh_xy=mesh_xy, sym=sym,
                                                  root=root, label=label,
                                                  photon=photon_layout if photon else None,
                                                  mu_bases=mu_bases)
        print_fn(f"shared-pole bank residence: {residence['residence']}; "
                 f"{residence['payload_bytes_per_rank'] / 2**30:.3f} GiB/rank, "
                 f"{residence['payload_bytes_total'] / 2**30:.2f} GiB total; {residence['reason']}"
                 if 'payload_bytes_per_rank' in residence else
                 f"shared-pole bank residence: file; {residence['reason']}")
        if "stage" in residence:
            ledger.live_stages = (residence["stage"],)
        bank = dict(path=str(root / "bank.h5") if resident is None else resident,
                    identity=identity, tables=tables, coulomb=coulomb, root=str(root))
        if photon:
            bank.update(photon_layout=photon_layout, mu_bases=mu_bases,
                        bispinor_v_q_path=bispinor_v_q_path,
                        sector_tables=(tables, _shared_pole_tables(
                            meta, sym, mu_bases[1].canonical_indices)))
        if not resume_constructor:
            initialize_shared_pole_bank(bank["path"], meta=meta, tables=tables,
                recipe=recipe, identity=identity, mesh_xy=mesh_xy,
                **(dict(photon_layout=photon_layout, mu_bases=mu_bases) if photon else {}))
            if not getattr(bank["path"], "fits", True):
                # The per-rank files could not reserve their bytes on some rank
                # (agreed on every rank): the shared scratch file, as before.
                bank["path"].release()
                bank["path"], resident = str(root / "bank.h5"), None
                residence = dict(residence, residence="file", reason="per-rank files refused "
                                 "by the filesystem (capacity); shared scratch file")
                print_fn(f"shared-pole bank residence: file; {residence['reason']}")
                initialize_shared_pole_bank(bank["path"], meta=meta, tables=tables,
                    recipe=recipe, identity=identity, mesh_xy=mesh_xy,
                    **(dict(photon_layout=photon_layout, mu_bases=mu_bases) if photon else {}))
        receipts = dict(identity=identity)
        if resume_constructor is True:
            receipts['bank'] = json.loads((root / 'bank_receipt.json').read_text())
            if not photon:
                receipts['moments'] = json.loads((root / 'moments_receipt.json').read_text())
            receipts['constructor_resume'] = dict(
                status='AUTHENTICATED_COMPLETE_BANK', source=str(root / 'bank.h5'))
        recorded = set()

        def record(stage, receipt):
            # EVERY RANK LEAVES THIS CALL THE SAME WAY.  A bare
            # ``process_index() == 0`` write raises on rank 0 alone (quota,
            # EIO on purge-eligible scratch) while ranks 1..P-1 walk into
            # the next collective and hang to walltime (INVARIANTS 21).
            # ``rank0_transaction`` does the same serial write and
            # broadcasts its verdict, exactly as the sibling write above.
            receipts[stage] = receipt
            recorded.add(stage)
            path = root / (stage + "_receipt.json")
            rank0_transaction(
                path, stage=f"shared_pole.receipt.{stage}",
                write=lambda: path.write_text(_json(receipt) + "\n"))
            print_fn(f"shared-pole {stage}: completion={receipt.get('completion', receipt.get('status'))}; "
                     f"seconds={receipt.get('seconds', {})}")
    with timing.section("spole.bank"):
        if resume_constructor:
            print_fn('shared-pole bank: not produced; '
                     + ('committed model reused' if resume_constructor == 'committed'
                        else 'authenticated complete producer artifact reused'))
        elif photon:
            from .response_bank import compute_photon_bank
            record("bank", compute_photon_bank(wfns, wfns_transverse, meta, config,
                mesh_xy=mesh_xy, sym=sym, mu_bases=mu_bases, layout=photon_layout,
                occupation_state=occupation_state, sample_plan=recipe, bank_io=bank,
                wfn=wfn, photon_g0_vectors=photon_g0_vectors,
                wfn_fingerprint_binding=wfn_fingerprint_binding,
                photon_head_cache=photon_head_cache,
                photon_head_state=photon_head_state, print_fn=print_fn))
        else:
            record("bank", produce_w_bank(wfns, meta, config, mesh_xy=mesh_xy,
                sym=sym, sample_plan=recipe, bank_io=bank, print_fn=print_fn))
    if not photon and not resume_constructor:
        with timing.section("spole.moments", announce=True,
                            label="shared-pole response moments"):
            record("moments", compute_response_moments(wfns, meta, config,
                mesh_xy=mesh_xy, sym=sym, bank_io=bank))
        moments_seconds = receipts["moments"].get("seconds", {})
        print_fn(f"Response quadrature: moments {receipts['moments'].get('correlation_count', 0)} "
                 f"correlations in {len(receipts['moments'].get('q_batches', ()))} q batch(es) "
                 f"of <= {receipts['moments'].get('q_width', 0)} parents; "
                 f"{moments_seconds.get('total', 0.0):.2f} s ("
                 + " ".join(f"{name}={moments_seconds[name]:.2f}" for name in
                            ("correlations", "coulomb", "dyson", "diagnostics", "io")
                            if name in moments_seconds) + ")")
    # The constructor owns scratch reads, actual pencil planning and the
    # final writer. It must query its own native workspace at the actual R.
    # W/dW and M1/M3 are distinct keyed datasets in the same scratch file.
    if resume_constructor == "committed":
        from file_io.shared_pole_store import validate_shared_pole_model
        model_path = root / "model.h5"
        result = dict(model=str(model_path), status="COMMITTED_MODEL_REUSED",
                      model_header=validate_shared_pole_model(
                          model_path, expected_identity=identity, mesh_xy=mesh_xy,
                          capacity=ledger),
                      model_residence=dict(residence="file", reason="committed model reused"))
    elif photon:
        from .shared_pole_sectors import construct_sector_poles
        result = construct_sector_poles(bank, meta, config,
            mesh_xy=mesh_xy, output=str(root / "model.h5"), print_fn=print_fn)
        from lxkit import memory
        from .shared_pole_execution import route_summary
        rows = result.get("execution") or ()
        if rows:
            routes = [(row["sector"], row) for row in rows] + [("CT", rows[0]["joint"])]
            for name, row in routes:
                memory.record_route(f"shared-pole sector constructor {name}", row["mode"],
                                    route_summary(row["mode"], row), warning=row.get("warning"))
            _print_eigh_stack_routes(print_fn)
    else:
        # An SC map keeps its model on the devices when it fits; an export
        # (write_w, write_poles) reads model.h5 and a one-shot registers it
        # as a restart member, so those write it.  On the bank's file tier
        # the model is persisted too: a rerun reuses it instead of the bank,
        # which is released below.
        model_rule = None
        if (sc_scratch and resident is not None
                and not (config.debug.write_w or config.write_poles)):
            model_rule = partial(_scalar_model_residence, meta, mesh_xy=mesh_xy,
                                 root=root, identity=identity)
        with timing.section("spole.constructor", announce=True,
                            label="shared-pole constructor"):
            result = construct_shared_poles(bank, bank, meta, config,
                mesh_xy=mesh_xy, output=str(root / "model.h5"), residence=model_rule)
        walls = dict(result.get("seconds", {}))
        rounds = walls.pop("rounds", 0)
        from lxkit import memory
        from .shared_pole_execution import route_summary
        route = result.get("execution")
        if route is not None:
            memory.record_route("shared-pole constructor", route["mode"],
                                route_summary(route["mode"], route),
                                warning=route.get("warning"))
        print_fn("Shared-pole constructor: "
                 + f"{rounds} round(s); seconds "
                 + " ".join(f"{name}={value:.2f}" for name, value in
                            sorted(walls.items(), key=lambda item: -item[1])))
        _print_eigh_stack_routes(print_fn)
    if resident is not None:
        # The model is committed; the constructor was the bank's last reader.
        resident.release()
        ledger.live_stages = ()
    elif not photon and not config.debug.write_w:
        # The file tier likewise: model.h5 is committed, and only a write_w
        # export (and the photon Sigma's constant) would read the bank again.
        released = _release_bank_file(root / "bank.h5")
        if released is not None:
            residence = dict(residence, released=released)
            print_fn(f"shared-pole bank: scratch file released after the constructor "
                     f"({released['bytes'] / 2**30:.2f} GiB, {released['links']} link(s))")
    receipts["bank_residence"] = residence
    with timing.section("spole.screening_finalize", announce=True,
                        label="shared-pole receipts and head"):
        record("constructor", result)
        models = result["model_residence"]
        if "payload_bytes_per_rank" in models:
            print_fn(f"shared-pole {'sector models' if photon else 'model'}: {models['residence']}; "
                     f"{models['payload_bytes_per_rank'] / 2**30:.3f} GiB/rank"
                     f"{' bound' if photon else ''}; {models['reason']}")
        header = None if photon else result["model_header"]
        handle = (result['handle'] if photon else
                  dict(path=result["model"], identity=identity,
                       digest=header["digest"], K=list(header["K"])))
        if not photon and models.get("stage"):
            # Sigma and the final-map W0 persist read it; the next map releases it.
            handle["model_stage"] = models["stage"]
        if photon and sc_scratch and handle.get("model_stage"):
            # Sigma releases TT and CT; CC stays for the final-map W0 persist.
            handle["hold_charge_model"] = True
        # Device-resident sector models stay live until Sigma releases them.
        ledger.live_stages = (handle['model_stage'],) if handle.get('model_stage') else ()
        result = dict(shared_pole=handle)
        from .gw_config import HeadCorrection
        if photon:
            _mark_photon_head(handle, config)
        elif config.head.correction is not HeadCorrection.OFF:
            from .shared_pole_head import build_shared_pole_head
            head, iteration_head = build_shared_pole_head(
                handle, header, V_q, wfns, meta, config, mesh_xy=mesh_xy, wfn=wfn,
                response=iteration_head_response, head_resolver=head_resolver,
                plan=mpa_plan, material_class=material_class, occupation_state=occupation_state)
            result.update(mpa_head=head, iteration_head=iteration_head)
            record("head", head)
        if (config.debug.write_w or config.write_poles) and not photon:
            from file_io.shared_pole_store import export_shared_pole_outputs
            with timing.section("spole.outputs"):
                receipts["outputs"] = export_shared_pole_outputs(handle, meta=meta,
                    config=config, mesh_xy=mesh_xy, source_wfn=source_wfn,
                    run_dir=run_dir, label=label, tables=tables, print_fn=print_fn)
        # The member lives in the restart file, so it is registered exactly when
        # that file is written (one owner: gw_output.restart_tensor_writes_enabled).
        from .gw_output import restart_tensor_writes_enabled
        if (tensors_filename is not None and not sc_scratch and not photon
                and restart_tensor_writes_enabled(config, tensors_filename)):
            receipts["restart_member"] = register_shared_pole_restart_member(
                tensors_filename, handle["path"], expected_identity=identity,
                mesh_xy=mesh_xy, capacity=ledger)
        receipts["seconds"] = time.monotonic() - started
        receipts["handle"] = handle
        summary = root / "construction_receipt.json"
        # A recorded stage is already in its own ``<stage>_receipt.json``; the
        # summary names that file instead of encoding the stage a second time.
        compact = {key: (dict(file=key + "_receipt.json") if key in recorded else value)
                   for key, value in receipts.items()}
        rank0_transaction(
            summary, stage="shared_pole.construction_receipt",
            write=lambda: summary.write_text(_json(compact) + "\n"))
        return result
