"""Prepare atom-local RHSs for the existing four-component charge ISDF fit.

The source uses the normalized graph U=[I;X](I+X†X)^(-1/2). Compact
atomic fields taper the served large block and derive its small block from
the same gradient; their implicit Pauli precursor is R^-1(w R delta_phi).
Both sampled endpoints use that same reconstructed carrier. An explicit
full-WFN Lowdin convention rotates all endpoints by one factor measured
from the actual served four-spinor overlap.
Atomic density RHSs are projected onto Y_lm, then solved
by the smooth fit's existing factor in ``ZetaG.contract_v``.  This module
owns stage assembly, never symmetry transport, band-pair storage or FFTs.
"""
from __future__ import annotations

import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np


SCHEMA = "lorrax.isdf_augmentation.v1"
AUGMENTED_CHARGE_PAYLOAD_SCHEMA = "smooth_plus_local_delta_v1"


def _normalized_cache_control(control):
    """Normalized-field controls exclude the independent local Fourier artifact."""
    return {key:value for key,value in control.items()
            if key not in ('local_coulomb_fourier_file','local_coulomb_fourier_sha256')}


def read_augmentation_manifest(directory, *, load_raw_parent=True):
    """Authenticate one manifest and its small matched atomic sidecars.

    Parameters
    ----------
    directory : path
        Directory containing manifest.json. Species paths are relative to
        that directory. Numerical quadratures belong to this immutable
        artifact, rather than additional GW input switches.
    load_raw_parent : bool
        Keep True for fitting and restart authentication. False is solely
        for preparing the raw-parent artifact: authenticate the same
        species inputs while permitting the raw file/SHA pair to be absent.

    Returns
    -------
    dict
        Manifest fields, loaded species tables, and a content identity.
        Frozen core orbitals remain diagnostics; no core bands are added.
    """
    from psp.atomic_reconstruction import load_atomic_reconstruction, upf_identity

    if not isinstance(load_raw_parent,bool):
        raise TypeError("load_raw_parent must be a preparation-only boolean")
    root = Path(directory).resolve()
    path = root / "manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema") != SCHEMA
            or manifest.get("carrier") != "normalized_rkb"
            or manifest.get("frozen_core_policy") != "reconstruct_valence_only"):
        raise ValueError("augmentation manifest must declare normalized_rkb and reconstruct_valence_only")
    required = ("species", "radial", "angular", "cache", "runtime")
    if any(not isinstance(manifest.get(key), dict) for key in required):
        raise ValueError("augmentation manifest requires species/radial/angular/cache/runtime dictionaries")
    local_keys = {'local_coulomb_fourier_file','local_coulomb_fourier_sha256'}
    present = local_keys.intersection(manifest['cache'])
    if present and present != local_keys:
        raise ValueError("prepared local Coulomb Fourier cache requires explicit file/SHA pair")
    normal_control = _normalized_cache_control(manifest['cache'])
    if 'charge_fit' in manifest and manifest['charge_fit'] != {'conditioning':'unit_diagonal'}:
        raise ValueError("explicit charge_fit must contain exactly conditioning=unit_diagonal")
    if 'charge_metric' in manifest:
        accepted = ({'smooth_neutral_cross':'onsite'},
                    {'smooth_neutral_cross':'onsite','moment_enrichment':'served_monopole'})
        metric = manifest['charge_metric']
        positive = isinstance(metric, dict) and metric.get('body_metric') == 'physical_low_local_high'
        if positive:
            control = metric.get('periodic_compensation_cache')
            if (set(metric) != {'body_metric','moment_enrichment','periodic_compensation_cache'}
                    or metric['moment_enrichment'] != 'served_monopole'
                    or not isinstance(control, dict) or set(control) != {'file','file_sha256'}
                    or not isinstance(control['file'], str) or not control['file']
                    or not isinstance(control['file_sha256'], str)
                    or len(control['file_sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in control['file_sha256'])):
                raise ValueError('positive charge metric requires exactly delta/exact-M0 and a pinned periodic compensation file')
        elif metric not in accepted:
            raise ValueError("explicit charge_metric must contain exactly smooth_neutral_cross=onsite and optional moment_enrichment=served_monopole")
        if 'interpolation_degree' not in manifest['radial']:
            raise ValueError("onsite smooth-neutral charge cross requires radial interpolation_degree")
    enrich = manifest.get('charge_metric',{}).get('moment_enrichment') == 'served_monopole'
    served = manifest.get('served_moments')
    served_overlap = manifest.get('overlap',{}).get('mode') == 'full_wfn_lowdin'
    if (enrich or served_overlap) and served is None:
        raise ValueError("full-WFN served overlap or monopole enrichment requires an explicit served_moments artifact table")
    if served is not None:
        species_keys = {'species_files','species_sha256'}
        full_keys = species_keys | {'raw_parent_file','raw_parent_sha256'}
        accepted_keys = (full_keys,) if load_raw_parent else (species_keys,full_keys)
        if (not isinstance(served,dict) or set(served) not in accepted_keys
                or not isinstance(served['species_files'],dict)
                or not isinstance(served['species_sha256'],dict)
                or set(served['species_files']) != set(manifest['species'])
                or set(served['species_sha256']) != set(manifest['species'])
                or not isinstance(manifest['cache'].get('species_files'),dict)
                or manifest.get('overlap',{}).get('mode') != 'full_wfn_lowdin'):
            raise ValueError("served_moments requires exact species files/SHA coverage, cached normalized fields and full_wfn_lowdin")
    digest = hashlib.sha256(raw)
    tables = {}
    channel_control = manifest.get('partial_wave_channels')
    if channel_control is not None and (not isinstance(channel_control,dict)
            or channel_control.get('mode') != 'native_oncv_reference_channels'
            or not isinstance(channel_control.get('lmax'),int)
            or isinstance(channel_control['lmax'],bool) or channel_control['lmax'] < 0):
        raise ValueError("partial_wave_channels must explicitly name native_oncv_reference_channels and integer lmax")
    for label, entry in sorted(manifest["species"].items(), key=lambda item: int(item[0])):
        z = int(label)
        if z < 1 or not isinstance(entry, dict):
            raise ValueError("augmentation species must be keyed by positive atomic number")
        upf = (root / entry["source_upf"]).resolve()
        sidecar = (root / entry["reconstruction"]).resolve()
        data = load_atomic_reconstruction(sidecar, upf)
        source = upf_identity(upf)
        recorded = data["metadata"]["operator_comparison"].get("source", {})
        for key in ("source_sha256", "operator_sha256", "generator_input_sha256"):
            if recorded.get(key) != source[key]:
                raise ValueError(f"augmentation species {z}: authenticated {key} differs from source UPF")
        if (source["pseudo_type"] != "NC" or source["relativistic"] != "full"
                or source["has_so"].lower() not in ("t", "true")):
            raise ValueError("four-component augmentation requires a fully relativistic NC source UPF")
        if "atomic_number" in data["metadata"] and int(data["metadata"]["atomic_number"]) != z:
            raise ValueError("augmentation manifest species differs from atomic sidecar")
        if int(source['atomic_number']) != z:
            raise ValueError("augmentation manifest species differs from source atomic reference")
        selected = data['metadata'].get('radial_channel_selection')
        if channel_control is None and selected is not None:
            raise ValueError("a channel-selected sidecar requires explicit manifest partial_wave_channels")
        if channel_control is not None:
            # Only PP_RELBETA records define the native operator inventory;
            # independent scattering channels and diagnostic core waves do
            # not create an NC reference absent from the supplied UPF.
            native = set()
            for row in source['spin_channels']:
                if 'lll' not in row or 'jjj' not in row:
                    continue
                l,j2 = int(row['lll']),int(round(2*float(row['jjj'])))
                if j2 == 2*l+1:
                    kappa = -l-1
                elif l > 0 and j2 == 2*l-1:
                    kappa = l
                else:
                    raise ValueError("source UPF has an invalid native relativistic projector channel")
                if l <= channel_control['lmax']:
                    native.add((l,kappa))
            actual = set(zip(map(int,data['l']),map(int,data['kappa'])))
            declared = (set((int(row['l']),int(row['kappa'])) for row in selected.get('retained_channels',[]))
                        if isinstance(selected,dict) else set())
            if (not isinstance(selected,dict) or not native or actual != native or declared != actual
                    or selected.get('mode') != channel_control['mode']
                    or selected.get('lmax') != channel_control['lmax']
                    or any(len(str(selected.get(key,''))) != 64
                           for key in ('parent_payload_sha256','parent_metadata_sha256'))):
                raise ValueError(f"species {z} selected sidecar does not match explicit native UPF partial-wave channels")
        digest.update(str(z).encode())
        digest.update(source["source_sha256"].encode())
        digest.update(source['frozen_configuration_sha256'].encode())
        digest.update(data["metadata"]["payload_sha256"].encode())
        # The NPZ metadata includes the frozen atomic configuration and
        # generator/operator certificate; authenticate those bytes too.
        digest.update(json.dumps(data["metadata"], sort_keys=True).encode())
        tables[z] = data
    if not tables:
        raise ValueError("augmentation manifest contains no species")
    cache_files = manifest['cache'].get('species_files')
    cached = None
    if cache_files is not None:
        from psp.augmentation_cache import load_normalized_cache
        if (not isinstance(cache_files,dict)
                or set(cache_files) != set(manifest['species'])):
            raise ValueError("explicit normalized cache files must cover exactly the manifest species")
        cached = {}
        support = float(manifest['radial']['support_radius'])
        for z,data in tables.items():
            cache_path = (root/cache_files[str(z)]).resolve()
            cached[z] = load_normalized_cache(cache_path,data,normal_control,support_radius=support)
            digest.update(hashlib.sha256(cache_path.read_bytes()).digest())
    fourier_caches = None
    fourier_cache_hashes = {}
    if 'fourier_cache' in manifest:
        from psp.atomic_fourier_cache import load_atomic_fourier_caches
        control = manifest['fourier_cache']
        if (not isinstance(control,dict) or not isinstance(control.get('species_files'),dict)
                or set(control['species_files']) != set(manifest['species'])):
            raise ValueError("explicit atomic Fourier files must cover exactly the manifest species")
        fourier_caches = {}
        for z,data in tables.items():
            cache_path = (root/control['species_files'][str(z)]).resolve()
            fourier_caches[z] = load_atomic_fourier_caches(cache_path,data,control)
            fourier_cache_hashes[str(z)] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            digest.update(bytes.fromhex(fourier_cache_hashes[str(z)]))
    projection_binding = None
    if fourier_caches is not None:
        from isdf.atomic_moments import raw_parent_projection_binding
        projection_binding = raw_parent_projection_binding(tables,fourier_cache_hashes)
    served_caches = raw_moments = None
    if served is not None:
        from isdf.atomic_moments import load_served_moment_cache, load_raw_parent_moments
        served_caches = {}
        for z in tables:
            cache_path = (root/served['species_files'][str(z)]).resolve()
            sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            if sha != served['species_sha256'][str(z)]:
                raise ValueError(f"served moment species {z} file identity mismatch")
            norm_path = (root/cache_files[str(z)]).resolve()
            served_caches[z] = load_served_moment_cache(cache_path,
                normalized_cache_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                support_radius=float(manifest['radial']['support_radius']))
            digest.update(bytes.fromhex(sha))
        if load_raw_parent:
            raw_path = (root/served['raw_parent_file']).resolve()
            # Authenticate the file and its internal payload now. Preparation
            # independently reconstructs this binding from the actual WFN.
            with np.load(raw_path,allow_pickle=False) as archive:
                binding = json.loads(str(archive['metadata_json']))['binding']
            raw_moments = load_raw_parent_moments(raw_path,expected_binding=binding,
                expected_file_sha256=served['raw_parent_sha256'])
            if (raw_moments.get('atom_C') is not None
                    and binding.get('projection_binding') != projection_binding):
                raise ValueError("prepared raw C disagrees with the authenticated atomic tables or Fourier duals")
            import importlib.util
            for owner,sha in binding['source_identity']['owner_sources_sha256'].items():
                if hashlib.sha256(Path(importlib.util.find_spec(owner).origin).read_bytes()).hexdigest() != sha:
                    raise ValueError(f"raw served-moment owner identity mismatch: {owner}")
            digest.update(bytes.fromhex(served['raw_parent_sha256']))
    prepared_cache = None
    if present:
        from isdf.coulomb_fourier_cache import load_coulomb_fourier_cache
        local_path = (root/manifest['cache']['local_coulomb_fourier_file']).resolve()
        local_sha = manifest['cache']['local_coulomb_fourier_sha256']
        prepared_cache = load_coulomb_fourier_cache(local_path,expected_file_sha256=local_sha)
        digest.update(bytes.fromhex(local_sha))
    return dict(manifest,tables=tables,normalized_caches=cached,
                fourier_caches=fourier_caches,served_moment_caches=served_caches,
                raw_parent_moments=raw_moments,prepared_fourier_cache=prepared_cache,
                raw_parent_projection_binding=projection_binding,
                identity=digest.hexdigest(),directory=str(root))


def augmentation_identity(directory):
    """Content identity for fitting and restart provenance, after authentication."""
    return read_augmentation_manifest(directory)["identity"]


def _radial_grid(control):
    from psp.atomic_reconstruction import log_radial_weights

    if "radius" in control:
        r = np.asarray(control["radius"], dtype=np.float64)
        w = np.asarray(control["weights_dr"], dtype=np.float64)
    else:
        if control.get("kind") != "log_simpson":
            raise ValueError("radial manifest must contain explicit radius/weights_dr or kind=log_simpson")
        r = np.geomspace(float(control["r_min"]), float(control["r_max"]), int(control["points"]))
        w = log_radial_weights(r)
    support = float(control["support_radius"])
    if (r.ndim != 1 or w.shape != r.shape or len(r) < 4
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(w))
            or np.any(r <= 0) or np.any(w <= 0) or np.any(np.diff(r) <= 0)
            or support < r[-1]):
        raise ValueError("augmentation radial quadrature requires positive ordered bohr radii and dr weights")
    return r, w, support


def _orbit_angular_quadrature(control, cartesian_rotations):
    """Average a Lebedev rule over actual crystal rotations, then merge duplicates."""
    from scipy.integrate import lebedev_rule
    from scipy.spatial import cKDTree
    from scipy.special import sph_harm_y

    order = int(control["lebedev_order"])
    lmax = int(control["lmax"])
    if lmax < 0 or order < 2*lmax:
        raise ValueError("Lebedev polynomial order must cover twice the retained density lmax")
    xyz, weights = lebedev_rule(order)
    directions = np.asarray(xyz, dtype=np.float64).T
    weights = np.asarray(weights, dtype=np.float64)
    matrices = np.asarray(cartesian_rotations,dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3,3) or not len(matrices):
        raise ValueError("angular orbit closure requires the symmetry owner's Cartesian operation rows")
    if np.max(np.abs(matrices @ matrices.transpose(0, 2, 1)-np.eye(3))) > 2e-10:
        raise ValueError("crystal Cartesian rotations are inconsistent with augmentation lattice rows")
    full = np.concatenate([directions @ op for op in matrices])
    full_weights = np.tile(weights/len(matrices), len(matrices))
    tree = cKDTree(full)
    used = np.zeros(len(full), dtype=bool)
    rows, totals = [], []
    for i in range(len(full)):
        if used[i]:
            continue
        same = np.asarray(tree.query_ball_point(full[i], 2e-12), dtype=int)
        if np.max(np.linalg.norm(full[same]-full[i], axis=1)) > 2e-12:
            raise ValueError("ambiguous angular orbit merge")
        used[same] = True
        rows.append(full[i])
        totals.append(np.sum(full_weights[same]))
    directions, weights = np.asarray(rows), np.asarray(totals)
    lm = np.asarray([(l, m) for l in range(lmax+1) for m in range(-l, l+1)], dtype=np.int32)
    theta = np.arccos(np.clip(directions[:, 2], -1, 1))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    harmonics = np.asarray([sph_harm_y(l, m, theta, phi) for l, m in lm])
    error = float(np.max(np.abs((harmonics.conj()*weights) @ harmonics.T-np.eye(len(lm)))))
    tolerance = float(control["orthogonality_tolerance"])
    if error > tolerance or abs(np.sum(weights)-4*np.pi) > tolerance:
        raise ValueError(f"angular quadrature harmonic Gram residual {error:.3e} exceeds {tolerance:.3e}")
    return directions, weights, lm, harmonics, error


def _certify_spheres(centers_frac, lattice, support):
    """Certify every atom and lattice image using a singular-value distance bound."""
    sigma = float(np.linalg.svd(lattice, compute_uv=False)[-1])
    width = int(np.ceil(2*float(support)/sigma))+2
    offsets = np.asarray([(i, j, k) for i in range(-width, width+1)
                          for j in range(-width, width+1) for k in range(-width, width+1)])
    minimum = np.inf
    for a, center in enumerate(centers_frac):
        for b, other in enumerate(centers_frac):
            distance = np.linalg.norm((other-center+offsets) @ lattice, axis=1)
            if a == b:
                distance[np.all(offsets == 0, axis=1)] = np.inf
            minimum = min(minimum, float(distance.min()))
    if not np.isfinite(minimum) or minimum <= 2*support*(1+2e-12):
        raise ValueError(f"augmentation spheres overlap: closest atom/image {minimum:.12g} bohr, diameter {2*support:.12g}")
    return minimum


def _normalized_caches(tables, control, support, *, validated_caches=None):
    """Use the atomic cache owner and retain explicit norm/gradient tail checks."""
    from psp.augmentation_cache import (build_normalized_cache,
                                        normalized_cache_tail_diagnostics)

    caches, tails = {}, {}
    for z, data in tables.items():
        if float(data['r'][-1]) > support:
            raise ValueError(f"species {z} pre-lift reconstruction radius exceeds compact sphere")
        if validated_caches is not None:
            if z not in validated_caches:
                raise ValueError(f"species {z} is missing its explicit validated normalized cache")
            cache = validated_caches[z]
        else:
            cache = build_normalized_cache(data, control, support_radius=support)
        tail = normalized_cache_tail_diagnostics(cache, support_radius=support)
        tolerance = float(control['tail_relative_tolerance'])
        gradient_tolerance = float(control.get('tail_gradient_relative_tolerance', tolerance))
        if (not np.isfinite((tolerance, gradient_tolerance)).all()
                or tolerance < 0 or gradient_tolerance < 0):
            raise ValueError("normalized-cache tail tolerances must be finite and nonnegative")
        if tail['norm'] > tolerance or tail['gradient'] > gradient_tolerance:
            raise ValueError(f"species {z} normalized-RKB tail {tail} exceeds manifest controls")
        caches[z], tails[z] = cache, tail
    return caches, tails


def _put(array, mesh, spec):
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding
    return device_put_process_local(np.asarray(array), NamedSharding(mesh, spec))


def _atomic_fourier_table(data, wavevectors, geometry, mesh, *,
                          radial_cache=None, delta_overlap=False):
    """Build only addressable G shards through the canonical placement owner.

    Atomic transform/dual definitions remain with their existing owners.
    This avoids computing and staging every other process's large angular
    table on each host. The callback is a pure function of global metadata
    and the owner's authenticated shard indices.
    """
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_tiles
    from psp.augmentation_spinors import spinor_function_labels
    from psp.augmented_samples import atomic_projection_table
    from psp.reconstruction_overlap import atomic_delta_overlap_table

    K = np.asarray(wavevectors,dtype=np.float64)
    if K.ndim != 3 or K.shape[-1] != 3:
        raise ValueError("atomic Fourier placement requires (parent,G,3) momenta")
    nf = len(spinor_function_labels(data['l'],data['kappa']))
    shape = (len(K),nf,2,K.shape[1])
    factory = atomic_delta_overlap_table if delta_overlap else atomic_projection_table
    def tile(index):
        parents,functions,spin,gslots = index
        values = np.asarray([factory(data,momenta,center_cart=geometry['center_cart'],
            cell_volume=geometry['cell_volume'],normalized_rkb_source=True,
            radial_cache=radial_cache) for momenta in K[parents,gslots]])
        return values[:,functions,spin,:]
    return device_put_process_tiles(shape,NamedSharding(mesh,P(None,None,None,('x','y'))),tile)


def _face_complement(nmu, mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import transpose_xy

    # Moving spin to a replicated prefix changes no data distribution.
    transposed = jax.jit(lambda a: a.transpose(0, 2, 1, 3),
        out_shardings=NamedSharding(mesh, P(None, None, 'x', 'y')))(nmu)
    return transpose_xy(transposed, mesh)


def _preserve_for_donating_fit(value):
    """Own bitwise complex operands before another fit donates their source.

    Two separate conjugation dispatches prevent identity/copy elimination.
    The first preserves its input and creates the intermediate; the second
    donates that intermediate. Source and one additional buffer coexist,
    on exactly the input sharding, with no host or reciprocal gather.
    """
    import jax
    import jax.numpy as jnp

    if np.dtype(value.dtype) != np.dtype(np.complex128):
        raise ValueError('preserved fitting operands must be complex128')
    saved = jax.jit(jnp.conj,out_shardings=value.sharding)(value)
    saved.block_until_ready()
    saved = jax.jit(jnp.conj,out_shardings=value.sharding,donate_argnums=0)(saved)
    saved.block_until_ready()
    return saved


def _tile_kernels(mesh, pc, bc):
    """Fixed parent/band tiles; only bounded projector coefficients replicate."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    XY = ('x', 'y')
    z0 = jnp.int32(0)
    gs, fs, cs = P(None, None, None, XY), P(None, 'x', None, 'y'), P(None, XY, None)
    def compile(fn, ins, outs, donate=()):
        return jax.jit(shard_map(fn, mesh=mesh, in_specs=ins, out_specs=outs,
                                check_vma=False), donate_argnums=donate)
    def source(a, p0, b0):
        return jax.lax.dynamic_slice(a, (p0, b0, z0, z0), (pc, bc, a.shape[2], a.shape[3]))
    def face(a, p0, b0):
        local = jax.lax.dynamic_slice(a, (p0, z0, z0, z0), (pc, a.shape[1], 4, a.shape[3]))
        indices = b0+jnp.arange(bc)-jax.lax.axis_index('x')*a.shape[1]
        valid = (indices >= 0)&(indices < a.shape[1])
        tile = jnp.take(local,jnp.clip(indices,0,a.shape[1]-1),axis=1)*valid[None,:,None,None]
        return jax.lax.psum_scatter(tile,'x',scatter_dimension=1,tiled=True)
    def update_face(a, value, p0, b0):
        value = jax.lax.all_gather(value,'x',axis=1,tiled=True)
        indices = b0+jnp.arange(bc)-jax.lax.axis_index('x')*a.shape[1]
        valid = (indices >= 0)&(indices < a.shape[1])
        indices = jnp.where(valid,indices,a.shape[1])
        local = jax.lax.dynamic_slice(a,(p0,z0,z0,z0),(pc,a.shape[1],4,a.shape[3]))
        local = local.at[:,indices].set(value,mode='drop')
        return jax.lax.dynamic_update_slice(a,local,(p0,z0,z0,z0))
    def store_coeff(a, value, p0, b0):
        rank = jax.lax.axis_index('x')*py+jax.lax.axis_index('y')
        indices = b0+jnp.arange(bc)-rank*a.shape[1]
        valid = (indices >= 0)&(indices < a.shape[1])
        indices = jnp.where(valid,indices,a.shape[1])
        local = jax.lax.dynamic_slice(a,(p0,z0,z0),(pc,a.shape[1],a.shape[2]))
        local = local.at[:,indices].set(value,mode='drop')
        return jax.lax.dynamic_update_slice(a,local,(p0,z0,z0))
    def read_coeff(a, p0, b0):
        rank = jax.lax.axis_index('x')*py+jax.lax.axis_index('y')
        local = jax.lax.dynamic_slice(a,(p0,z0,z0),(pc,a.shape[1],a.shape[2]))
        indices = b0+jnp.arange(bc)-rank*a.shape[1]
        valid = (indices >= 0)&(indices < a.shape[1])
        tile = jnp.take(local,jnp.clip(indices,0,a.shape[1]-1),axis=1)*valid[None,:,None]
        return jax.lax.psum(tile,XY)
    return dict(source=compile(source, (gs, P(), P()), gs),
                face=compile(face, (fs, P(), P()), fs),
                update_face=compile(update_face, (fs, fs, P(), P()), fs, (0,)),
                store_coeff=compile(store_coeff, (cs, P(), P(), P()), cs, (0,)),
                read_coeff=compile(read_coeff, (cs, P(), P()), P()))


def _point_phase_kernel(mesh):
    """One bounded parent/G-shard phase packet, reused over band batches."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    return jax.jit(shard_map(
        lambda K,points:jnp.exp(1j*jnp.einsum('pgi,mi->pgm',K,points)),mesh=mesh,
        in_specs=(P(None,('x','y'),None),P()),
        out_specs=P(None,('x','y'),None),check_vma=False))


def _point_samples_kernel(mesh, pc, bc, npoint, g_block, fft_points):
    """Sparse physical-G DFT on a bounded packet, with canonical face output."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    gs, fs = P(None, None, None, ('x', 'y')), P(None, 'x', None, 'y')
    z0 = jnp.int32(0)
    if npoint % py:
        raise ValueError("packed atomic point extent must divide Y face")
    def sample(source, phase_table, live):
        ng = source.shape[-1]
        width = min(int(g_block), ng)
        blocks = (ng+width-1)//width
        pad = blocks*width-ng
        source = jnp.pad(source, ((0,0),(0,0),(0,0),(0,pad)))
        phase_table = jnp.pad(phase_table,((0,0),(0,pad),(0,0)),constant_values=1.)
        zero = jnp.zeros((pc, bc, 4, npoint), jnp.complex128)
        def add(acc, step):
            c = jax.lax.dynamic_slice(source, (z0,z0,z0,step*width), (pc,bc,4,width))
            phase = jax.lax.dynamic_slice(phase_table,(z0,step*width,z0),(pc,width,npoint))
            return acc+jnp.einsum('pnsg,pgm->pnsm', c, phase), None
        partial = jax.lax.scan(add, zero, jnp.arange(blocks,dtype=jnp.int32), unroll=1)[0]
        total = jax.lax.psum(partial, ('x', 'y'))/np.sqrt(float(fft_points))
        total *= live[None,None,None,:]
        return jax.lax.dynamic_slice(total,
                    (z0,jax.lax.axis_index('x')*(bc//px),z0,jax.lax.axis_index('y')*(npoint//py)),
                    (pc,bc//px,4,npoint//py))
    return jax.jit(shard_map(sample, mesh=mesh,
        in_specs=(gs,P(None,('x','y'),None),P()),
        out_specs=fs, check_vma=False))


def _smooth_point_faces(source, wavevectors, points, active, mesh, *, lattice,
                        parent_chunk, band_chunk, g_block, fft_points):
    """Sample one already selected reciprocal carrier with the blocked DFT owner."""
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P

    npar,nb = map(int,source.shape[:2])
    pc,bc,mu = int(parent_chunk),int(band_chunk),len(points)
    kernels = _tile_kernels(mesh,pc,bc)
    faces = jnp.zeros((npar,nb,4,mu),jnp.complex128,
                      device=NamedSharding(mesh,P(None,'x',None,'y')))
    sampler = _point_samples_kernel(mesh,pc,bc,mu,g_block,fft_points)
    phase_kernel = _point_phase_kernel(mesh)
    cart,live = _put(points @ lattice,mesh,P()),_put(active.astype(float),mesh,P())
    for p0 in range(0,npar,pc):
        K = _put(wavevectors[p0:p0+pc],mesh,P(None,('x','y'),None))
        phase = phase_kernel(K,cart)
        for b0 in range(0,nb,bc):
            tile = sampler(kernels['source'](source,jnp.int32(p0),jnp.int32(b0)),phase,live)
            faces = kernels['update_face'](faces,tile,jnp.int32(p0),jnp.int32(b0))
        faces.block_until_ready()
    return faces


def _sample_geometry(points, active, centers, lattice, caches, atom_types, kfrac, support, scale):
    """Bounded host delta tables; samples use sqrt(Omega/Nfft) U delta_phi."""
    from psp.augmented_samples import atomic_image_geometry
    from psp.augmentation_spinors import evaluate_normalized_delta

    for center, z in zip(centers, atom_types):
        relative, images = atomic_image_geometry(points, center, lattice)
        inside = (np.linalg.norm(relative, axis=1) <= support) & active
        # The evaluator owns spin-angular degeneracy and channel order.
        probe = evaluate_normalized_delta(caches[int(z)], np.zeros((1,3)))
        delta = np.zeros((probe.shape[0], 4, len(points)), dtype=np.complex128)
        if np.any(inside):
            delta[..., inside] = evaluate_normalized_delta(caches[int(z)], relative[inside])*scale
        phase = np.exp(2j*np.pi*np.einsum('pi,mi->pm', kfrac, images))*active[None,:]
        yield delta, phase


def _packet_plan(plan, points, first=None):
    from .centroid_k_unfold import build_centroid_k_unfold_plan

    right = build_centroid_k_unfold_plan(plan.sym, points, plan.fft_grid, plan.mesh_xy,
        nspinor=4, parent_k_frac=plan.k_parent_frac, coordinate_kind='fractional',
        layout=None if first is None else first.layout)
    if first is None:
        return right
    for name in ('sym_perm', 'L_table'):
        if not np.array_equal(getattr(first, name), getattr(right, name)):
            raise ValueError(f"atomic radial packets change canonical {name}; choose sphere/packet geometry with one authenticated transport table")
    return first


def _angular_bucket_tables(point_plan, na, rp, weights_y, py):
    """Group each local point shard in canonical atom/radius/direction order.

    The packed point axis represents exactly ``na * rp * nang`` logical
    points, with direction fastest. The common point plan authenticates its
    coordinates; this map checks the logical ordering used by compression.
    All shards use one static bucket width, including wholly ghost shards.
    """
    packed = np.asarray(point_plan.layout.axis.packed_to_canonical)
    active = np.asarray(point_plan.layout.axis.active_mask, dtype=bool)
    weights_y = np.asarray(weights_y)
    if weights_y.ndim != 2 or min(weights_y.shape) < 1 or not np.isfinite(weights_y).all():
        raise ValueError("angular compression requires finite harmonic/direction weights")
    nang = int(weights_y.shape[1])
    logical = int(na)*int(rp)*nang
    if (min(int(na),int(rp),int(py)) < 1 or packed.ndim != 1
            or active.shape != packed.shape or len(packed)%int(py)
            or int(point_plan.layout.axis.n_logical) != logical
            or not np.array_equal(np.sort(packed[active]),np.arange(logical))):
        raise ValueError("angular compression point axis must contain exactly atom/radius/direction rows")
    local_points, rows = len(packed)//int(py), int(na)*int(rp)
    groups = []
    for y in range(int(py)):
        labels = packed[y*local_points:(y+1)*local_points]
        live = active[y*local_points:(y+1)*local_points]
        groups.append([np.flatnonzero(live & (labels//nang == row)) for row in range(rows)])
    width = max(1,max(len(indices) for shard in groups for indices in shard))
    indices = np.zeros((int(py),rows,width),np.int32)
    angular = np.zeros((int(py),weights_y.shape[0],rows,width),weights_y.dtype)
    for y, shard in enumerate(groups):
        for row, columns in enumerate(shard):
            indices[y,row,:len(columns)] = columns
            angular[y,:,row,:len(columns)] = weights_y[:,packed[y*local_points+columns]%nang]
    return indices,angular


def _angular_compression_workspace_bytes(nq, nmu, na, nh, rp, nang, px):
    """Bound local bucket gather, pre-scatter result and index/weight tables.

    A shard can own every direction of a radial row. Price that common
    worst-case width before constructing the actual packed packet plan.
    The full input RHS and stored output are charged by their own operands.
    """
    rows, mu_local = int(na)*int(rp), int(nmu)//int(px)
    gather = 16.*int(nq)*mu_local*rows*int(nang)
    # The GW mesh is square, so the feature-Y pad uses the same extent.
    features = ((rows*int(nh)+int(px)-1)//int(px))*int(px)
    partial = 16.*int(nq)*mu_local*features
    tables = rows*int(nang)*(4.+16.*int(nh))
    return dict(bucket_gather=gather,pre_scatter_result=partial,
                index_and_angular_tables=tables,total=gather+partial+tables)


def _compress_rhs_kernel(mesh, point_plan, na, nh, rp, weights_y):
    """Project local angular buckets; psum_scatter keeps features over Y."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    py = int(mesh.shape['y'])
    indices,angular = _angular_bucket_tables(point_plan,na,rp,weights_y,py)
    if int(nh) != int(angular.shape[1]):
        raise ValueError("angular compression harmonic extent disagrees with weights")
    nf = na*nh*rp
    nfp = ((nf+py-1)//py)*py
    indices = _put(indices, mesh, P('y',None,None))
    angular = _put(angular, mesh, P('y',None,None,None))
    def project(z, columns, weights):
        values = jnp.take(z,columns[0],axis=-1)
        partial = jnp.einsum('qmfj,hfj->qmfh',values,weights[0])
        partial = partial.reshape((z.shape[0],z.shape[1],na,rp,nh))
        partial = partial.transpose(0,1,2,4,3).reshape((z.shape[0],z.shape[1],nf))
        partial = jnp.pad(partial,((0,0),(0,0),(0,nfp-nf)))
        return jax.lax.psum_scatter(partial, 'y', scatter_dimension=2, tiled=True)
    kernel = jax.jit(shard_map(project, mesh=mesh,
        in_specs=(P(None,'x','y'), P('y',None,None), P('y',None,None,None)),
        out_specs=P(None,'x','y'), check_vma=False))
    return lambda z: kernel(z, indices, angular), nf, nfp


def _orbital_norm_kernels(mesh):
    """Charge diagnostics; local-grid result is a quadrature estimate, never a renormalization."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    def smooth(a):
        return jax.lax.psum(jnp.sum(abs(a)**2,axis=(2,3)),('x','y'))
    def correction(ps,ae,weights):
        delta = ae-ps
        density = 2*jnp.real(jnp.conj(ps)*delta)+abs(delta)**2
        return jax.lax.psum(jnp.einsum('pnsm,m->pn',density,weights),'y')
    source = jax.jit(shard_map(smooth,mesh=mesh,in_specs=P(None,None,None,('x','y')),
                             out_specs=P(),check_vma=False))
    local = jax.jit(shard_map(correction,mesh=mesh,
        in_specs=(P(None,'x',None,'y'),P(None,'x',None,'y'),P('y')),
        out_specs=P(None,'x'),check_vma=False))
    return source,local


def _band_rotation_kernel(mesh, pc, layout, output_bands, public_start, physical_stop):
    """Rotate bounded parent packets, gathering only a packet's old band axis.

    The complete WFN factor acts before selecting the public fitting columns.
    Thus bands outside the public fitting window still contribute to those
    columns. Source G slots never gather; face/atomic bands gather only within
    one parent packet. Transport columns beyond the requested physical stop
    are exactly zero, including the public face padding.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    from psp.reconstruction_overlap import rotate_band_rows

    axes = dict(source=None, face='x', coefficients=('x','y'))
    specs = dict(source=P(None,None,None,('x','y')),
                 face=P(None,'x',None,'y'), coefficients=P(None,('x','y'),None))
    if layout not in specs:
        raise ValueError("unknown augmentation band-rotation layout")
    axis, spec = axes[layout], specs[layout]
    owners = 1 if axis is None else (int(mesh.shape['x']) if axis == 'x' else int(mesh.size))
    if int(output_bands)%owners:
        raise ValueError("public reconstruction band carrier must divide its band-owner axis")
    width = int(output_bands)//owners
    z0 = jnp.int32(0)
    def rotate(values, factor):
        out = jnp.zeros((values.shape[0],width,*values.shape[2:]), values.dtype)
        def packet(acc, step):
            p0 = step*pc
            rows = jax.lax.dynamic_slice_in_dim(values,p0,pc,axis=0)
            if axis is not None:
                rows = jax.lax.all_gather(rows,axis,axis=1,tiled=True)
            A = jax.lax.dynamic_slice_in_dim(factor,p0,pc,axis=0)
            # The common owner supplies the complex band-row orientation.
            rows = rotate_band_rows(rows,A,band_axis=1)
            wanted = public_start+jnp.arange(output_bands)
            selected = jnp.take(rows,wanted,axis=1,mode='clip')
            selected *= (wanted < physical_stop).reshape((1,output_bands)+(1,)*(rows.ndim-2))
            if axis is not None:
                rank = (jax.lax.axis_index('x') if axis == 'x' else
                        jax.lax.axis_index('x')*int(mesh.shape['y'])+jax.lax.axis_index('y'))
                selected = jax.lax.dynamic_slice_in_dim(selected,rank*width,width,axis=1)
            return jax.lax.dynamic_update_slice_in_dim(acc,selected,p0,axis=0),None
        return jax.lax.scan(packet,out,jnp.arange(values.shape[0]//pc,dtype=jnp.int32),unroll=1)[0]
    return jax.jit(shard_map(rotate,mesh=mesh,in_specs=(spec,P()),
                            out_specs=spec,check_vma=False))


def _full_wfn_rotation(smooth, nmu, coefficients, geometry, mesh):
    """Measure the actual served four-spinor Gram and rotate one full WFN.

    Only bounded parent packets of the small C/D/Gram arrays reach the host.
    The reciprocal wavefunctions remain G-sharded. Authenticated prepared
    four-component D and B describe the same served field as the samples;
    the coarse local-density quadrature never supplies the overlap metric.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    from common.collectives import gather_to_host
    from psp.reconstruction_overlap import reconstruction_gram,lowdin_factor

    npar,nb = map(int,smooth.shape[:2])
    pc = int(geometry['parent_chunk'])
    kernels = _tile_kernels(mesh,pc,nb)
    def source_gram(a):
        return jax.lax.psum(jnp.einsum('pnsg,pmsg->pnm',a.conj(),a),('x','y'))
    gram_kernel = jax.jit(shard_map(source_gram,mesh=mesh,
        in_specs=P(None,None,None,('x','y')),out_specs=P(),check_vma=False))
    atomic_grams = [geometry['served_caches'][int(z)]['B'] for z in geometry['atom_types']]
    raw_D = geometry['raw_served_D']
    physical_bands = int(geometry['physical_bands'])
    if (len(raw_D) != len(coefficients) or any(
            d.shape != (npar,physical_bands,len(B)) or not np.isfinite(d).all()
            for d,B in zip(raw_D,atomic_grams))):
        raise ValueError("prepared served overlap rows/channels disagree with the full-WFN carrier")
    receipts, factors = [], []
    source_grams = []
    raw_coefficients = [[] for _ in coefficients]
    raw_delta_overlaps = [[] for _ in coefficients]
    for p0 in range(0,npar,pc):
        source = kernels['source'](smooth,jnp.int32(p0),jnp.int32(0))
        gram = gather_to_host(gram_kernel(source))
        source_grams.append(gram)
        for atom in range(len(coefficients)):
            coeff = kernels['read_coeff'](coefficients[atom],jnp.int32(p0),jnp.int32(0))
            c_host = gather_to_host(coeff)
            d_host = np.pad(raw_D[atom][p0:p0+pc],((0,0),(0,nb-physical_bands),(0,0)))
            raw_coefficients[atom].append(c_host)
            raw_delta_overlaps[atom].append(d_host)
            gram = reconstruction_gram(gram,c_host,d_host,atomic_grams[atom])
        receipt = lowdin_factor(gram,physical_bands=int(geometry['physical_bands']))
        factors.append(receipt.pop('inverse_sqrt'))
        receipts.append(receipt)
    factor_host = np.concatenate(factors,axis=0)
    factor = _put(factor_host,mesh,P())
    args = (mesh,pc,geometry['output_bands'],geometry['public_start'],geometry['physical_stop'])
    source_rotation = _band_rotation_kernel(args[0],args[1],'source',*args[2:])
    coefficient_rotation = _band_rotation_kernel(args[0],args[1],'coefficients',*args[2:])
    smooth = source_rotation(smooth,factor)
    if nmu is not None:
        nmu = _band_rotation_kernel(args[0],args[1],'face',*args[2:])(nmu,factor)
    coefficients = [coefficient_rotation(c,factor) for c in coefficients]
    smooth.block_until_ready()
    receipt = {key:np.concatenate([r[key] for r in receipts],axis=0)
               for key in receipts[0] if isinstance(receipts[0][key],np.ndarray)}
    receipt.update(physical_bands=int(geometry['physical_bands']),inverse_sqrt=factor_host,
                   convention='full_wfn_lowdin_effective_vertex',
                   overlap_operator='actual_served_four_spinor')
    receipt.update(source_gram=np.concatenate(source_grams),
        atomic_coefficients=tuple(np.concatenate(c) for c in raw_coefficients),
        delta_overlaps=tuple(np.concatenate(d) for d in raw_delta_overlaps),
        atomic_delta_grams=tuple(atomic_grams))
    return smooth,nmu,coefficients,receipt


def _rhs_storage_kernels(mesh, q_indices, qpad, na, nh, nr, rp, feature_count):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    fs, qs = P(None,'x','y'), P(('x','y'),None,None)
    z0 = jnp.int32(0)
    index = _put(np.asarray(q_indices, np.int32), mesh, P())
    def select(z, ids):
        z = jnp.take(z, ids, axis=0)
        return jnp.pad(z, ((0,qpad-len(q_indices)),(0,0),(0,0)))
    select = jax.jit(shard_map(select, mesh=mesh, in_specs=(fs,P()), out_specs=fs, check_vma=False))
    def store(out, packet, r0):
        values = packet[...,:feature_count].reshape(packet.shape[0],packet.shape[1],na,nh,rp)
        view = out.reshape(out.shape[0],out.shape[1],na,nh,nr)
        view = jax.lax.dynamic_update_slice(view, values, (z0,z0,z0,z0,r0))
        return view.reshape(out.shape)
    store = jax.jit(shard_map(store, mesh=mesh, in_specs=(qs,qs,P()), out_specs=qs,
                             check_vma=False), donate_argnums=(0,))
    return lambda z: select(z,index), store


def _local_rhs_workspace_bytes(nparent, nq, nmu, npoint, mesh_size, *, retain_smooth=False,
                               nq_accumulator=None, primitive_count=1):
    """Live rectangular charge buffers before selected-q angular projection.

    The AE endpoint output remains live while the PS quarter scan holds its
    accumulator, one scalar FFI result and their sum: four full-q scalar
    panels. Subtraction and LR+RL completion fit within the same four-panel
    bound. Each Pauli-quarter parent projector has four spin entries; its
    left and right outputs plus the next GEMM/layout temporary need three
    such parent panels. The native resident k-parent convolution keeps its
    FFT row banks in shared memory. Persistent q-owned RHS and solved local
    coefficients belong to the separate resident terms in the stage ledger.
    Retaining the smooth scalar RHS adds one live PS result during delta
    completion and another completion/quarter temporary: six scalar panels
    bound both independently completed outputs without relying on aliasing.
    """
    scalar = 16.*int(nq)*int(nmu)*int(npoint)/int(mesh_size)
    parent = 16.*4*int(nparent)*int(nmu)*int(npoint)/int(mesh_size)
    if int(primitive_count) < 1:
        raise ValueError("RHS workspace requires at least one vertex primitive")
    # Previous delta/PS totals coexist with the next primitive scan.
    panels = (6 if retain_smooth else 4)+(2 if retain_smooth else 1)*int(primitive_count>1)
    if nq_accumulator is None:
        endpoint_bytes, full_panels = panels*scalar, panels
        accumulator_rows = int(nq)
    else:
        accumulator_rows = int(nq_accumulator)
        if not 1 <= accumulator_rows <= int(nq):
            raise ValueError("indexed RHS accumulator must name a nonempty full-q subset")
        # Native output and a possible native-layout/gather temporary retain
        # full-q size. Every quarter sum and both completions retain only
        # the selected/negative-partner union; no native ABI is changed.
        endpoint_bytes = 2*scalar+panels*scalar*accumulator_rows/int(nq)
        full_panels = 2
    return dict(full_q_scalar=scalar, full_q_scalar_panels=full_panels,
                accumulator_q_rows=accumulator_rows, accumulator_scalar_panels=panels,
                endpoint_and_quarter_outputs=endpoint_bytes,
                parent_projectors_and_layout=3*parent, total=endpoint_bytes+3*parent)


def _auxiliary_field_kernel(mesh, geometry, packed_tables, scale):
    """Place signed functional fields on the canonical band-X/point-Y face.

    Coefficients and overlaps use (parent,band,atomic function), band-sharded
    over XY. Only those small tables gather over Y; no reciprocal source
    or full point field gathers. Packed tables have (sign,2*function,4,point).
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    from isdf.atomic_moments import evaluate_auxiliary_charge

    table = _put(packed_tables,mesh,P(None,None,None,'y'))
    def evaluate(c,d,t):
        c = jax.lax.all_gather(c,'y',axis=1,tiled=True)
        d = jax.lax.all_gather(d,'y',axis=1,tiled=True)
        return evaluate_auxiliary_charge(c,d,dict(geometry,signed_tables=t),
                                        grid_sample_scale=scale,array_api=jnp)
    kernel = jax.jit(shard_map(evaluate,mesh=mesh,
        in_specs=(P(None,('x','y'),None),P(None,('x','y'),None),P(None,None,None,'y')),
        out_specs=(P(None,'x',None,'y'),P(None,'x',None,'y')),check_vma=False))
    return lambda c,d: kernel(c,d,table)


def _served_monopole_rhs(plan, faces, coefficients, overlaps, geometry, mesh,
                         source_capture=None):
    """Integrate the authenticated signed charge functional with the fit owner.

    The fixed GL8/Lebedev26 auxiliary geometry is an exact κ-covariant
    functional, not another physical density quadrature. Coefficients and
    overlaps have already received the same full-WFN band rotation.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.staged_reshard import face_to_batch_reshard
    from isdf.atomic_moments import auxiliary_charge_geometry, make_auxiliary_monopole_compressor
    from isdf.local_rhs import local_density_rhs

    atoms = [auxiliary_charge_geometry(geometry['caches'][int(z)]) for z in geometry['atom_types']]
    points = np.concatenate([center+atom['relative_points'] @ np.linalg.inv(geometry['lattice'])
                             for center,atom in zip(geometry['centers'],atoms)])
    right = _packet_plan(plan,points)
    axis = right.layout.axis
    weights = np.zeros((len(atoms),len(points)),np.float64)
    npar,nb = coefficients[0].shape[:2]
    fs = P(None,'x',None,'y')
    plus = jnp.zeros((npar,nb,4,right.n_centroid_packed),jnp.complex128,
                     device=NamedSharding(mesh,fs))
    minus = jnp.zeros_like(plus)
    start = 0
    for atom,(aux,c,d) in enumerate(zip(atoms,coefficients,overlaps)):
        stop = start+len(aux['relative_points'])
        weights[atom,start:stop] = aux['integration_weights']
        table = np.zeros((*aux['signed_tables'].shape[:3],len(points)),np.complex128)
        table[...,start:stop] = aux['signed_tables']
        packed = axis.pack_host(table,axis=3,fill_value=0.)
        a,b = _auxiliary_field_kernel(mesh,aux,packed,geometry['scale'])(c,d)
        plus,minus = plus+a,minus+b
        start = stop
    if source_capture is not None:
        from isdf.atomic_hartree import make_occupied_point_trace, make_occupied_density_projection
        trace = make_occupied_point_trace(right, source_capture['public_occupations'],
            source_capture['full_kweights'], cell_volume=source_capture['cell_volume'],
            spin_degeneracy=source_capture['spin_degeneracy'])
        physical = (trace(plus)-trace(minus))[None]
        py = int(mesh.shape['y'])
        local_points = int(right.n_centroid_packed)//py
        columns = np.broadcast_to(np.arange(local_points, dtype=np.int32),
                                  (py, 1, local_points)).copy()
        packed_weights = axis.pack_host(weights, axis=1, fill_value=0.)
        table = packed_weights.reshape(len(atoms), py, local_points).transpose(1,0,2)[:, :, None]
        project_source = make_occupied_density_projection(mesh, columns, table,
                                                          output_shape=(len(atoms),))
        source_capture['exact_monopole'] = project_source(physical)/np.sqrt(4*np.pi)
        source_capture['exact_monopole'].block_until_ready()
    compressor = make_auxiliary_monopole_compressor(mesh,right,weights)
    rhs = local_density_rhs(centroid_faces=faces,
        atom_ae_faces=(plus,_face_complement(plus,mesh)),
        atom_ps_faces=(minus,_face_complement(minus,mesh)),left_plan=plan,right_plan=right,
        weight_l=geometry['weight_l'],weight_r=geometry['weight_r'],kgrid=geometry['kgrid'],
        mesh_xy=mesh,q_neg_idx=geometry['qneg'],q_indices=geometry['indexed_q'])
    ids = (geometry['q_indices'] if geometry['indexed_q'] is None
           else np.arange(len(geometry['q_indices'])))
    selector,_ = _rhs_storage_kernels(mesh,ids,geometry['qpad'],len(atoms),1,1,1,len(atoms))
    rhs = face_to_batch_reshard(mesh)(selector(compressor['compress'](rhs)))
    rhs = jax.jit(lambda a: a[...,:len(atoms)],
                  out_shardings=NamedSharding(mesh,P(('x','y'),None,None)))(rhs)
    return rhs,dict(carrier=atoms[0]['carrier'],radial_points=8,angular_order=7,
                    auxiliary_radius=.7,points_per_atom=[len(a['relative_points']) for a in atoms],
                    signed_factor_receipts=[a['factor_receipts'] for a in atoms])


def _hartree_source_request(request, *, wfn, plan, public_range):
    """Resolve physical source occupations from the WFN, independent of fit loss."""
    import hashlib

    if request is None:
        return None
    if (not isinstance(request, dict)
            or set(request) != {'occupations', 'full_kweights', 'spin_degeneracy'}
            or plan.nspinor != 4 or isinstance(request['spin_degeneracy'], (bool,np.bool_))
            or request['spin_degeneracy'] != 1.
            or float(wfn.occupation_state_capacity) != 1.):
        raise ValueError('Hartree source request requires canonical normalized4 physical occupations')
    nb, stop = int(wfn.nbands), int(wfn.physical_density_band_stop)
    domain = 'full_bz' if plan.sym.parent_k_domain == 'full_bz' else 'file'
    values = wfn.physical_density_occupations(k=domain, unit_as_none=True)
    canonical = np.zeros((plan.n_parent, nb), np.float64)
    if values is None:
        canonical[:, :stop] = 1.
    elif np.asarray(values).shape == (plan.n_parent, stop):
        canonical[:, :stop] = values
    else:
        raise ValueError('Hartree source WFN occupation rows differ from the actual raw-parent domain')
    supplied = canonical if request['occupations'] is None else np.asarray(request['occupations'])
    if (supplied.shape != canonical.shape or np.iscomplexobj(supplied)
            or not np.isfinite(supplied).all() or np.any(supplied < 0) or np.any(supplied > 1)
            or not np.array_equal(supplied, canonical)):
        raise ValueError('Hartree source occupations differ from the canonical physical WFN source')
    lo, hi = map(int, public_range)
    if np.any(canonical[:, :lo]) or np.any(canonical[:, hi:]):
        raise ValueError('Hartree source occupied support lies outside the public SAME150 fitting carrier')
    full_values = wfn.physical_density_occupations(k='full_bz', unit_as_none=True)
    full_values = np.ones((plan.n_full, stop)) if full_values is None else np.asarray(full_values)
    if (full_values.shape != (plan.n_full, stop)
            or not np.array_equal(canonical[np.asarray(plan.irr_idx), :stop], full_values)):
        raise ValueError('Hartree source occupations disagree with the canonical full-zone parent unfold')
    canonical_weights = np.full(plan.n_full, 1./plan.n_full, np.float64)
    weights = (canonical_weights if request['full_kweights'] is None
               else np.asarray(request['full_kweights']))
    if (weights.shape != canonical_weights.shape or np.iscomplexobj(weights)
            or not np.isfinite(weights).all() or not np.array_equal(weights, canonical_weights)):
        raise ValueError('Hartree source requires the canonical uniform full-zone density weights')
    parent_weights = np.bincount(np.asarray(plan.irr_idx), weights=weights,
                                 minlength=plan.n_parent)
    return dict(occupations=canonical, full_kweights=weights, parent_kweights=parent_weights,
        spin_degeneracy=1., public_band_range=(lo, hi),
        occupations_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
        full_kweights_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        electron_count=float(np.sum(canonical*parent_weights[:, None])),
        cell_volume=float(wfn.cell_volume))


def _hartree_atomic_feature_binding(state):
    import hashlib
    return dict(feature_order=['delta','PS','exact_Y00'],
        radius_sha256=hashlib.sha256(np.asarray(state['radius'],np.float64).tobytes()).hexdigest(),
        weights_dr_sha256=hashlib.sha256(np.asarray(state['weights_dr'],np.float64).tobytes()).hexdigest(),
        lm_sha256=hashlib.sha256(np.asarray(state['lm'],np.int64).tobytes()).hexdigest(),
        centers_cart_sha256=hashlib.sha256(np.asarray(state['centers_cart'],np.float64).tobytes()).hexdigest(),
        fft_points=int(state['fft_points']),cell_volume=float(state['cell_volume']),
        support_radius=float(state['support_radius']))


def _physical_full_wfn_frame_binding(factor, physical_bands):
    """Exact physical factor payload; transport ghost padding has no identity."""
    import hashlib

    a = np.asarray(factor)
    nb = int(physical_bands)
    if (a.ndim != 3 or nb < 1 or min(a.shape[1:]) < nb
            or not np.isfinite(a[:, :nb, :nb]).all()):
        raise ValueError('Hartree source requires the finite full physical WFN factor')
    physical = np.asarray(a[:, :nb, :nb],np.complex128)
    return dict(full150_frame_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
        physical_frame_shape=list(physical.shape),physical_frame_band_domain=[0,nb],
        physical_frame_convention='same full-WFN inverse_sqrt columns on reciprocal and atomic band rows')


def charge_hartree_response_metadata(state, functional):
    """Bind a real physical functional to this actual captured source/frame.

    Called after the canonical Hartree numerical owner prepares its potential
    and local response. It performs no second density, Poisson solve or fit.
    """
    import hashlib
    import json
    import jax

    source = state.get('hartree_source')
    if (source is None or 'exact_monopole' not in source
            or functional.get('operator') != 'ordinary_3D_periodic_full_FFT_G0_zero'
            or functional.get('neutral_mean_policy') != 'subtract_free_space_neutral_cell_mean'
            or functional.get('local_feature_order') != ('delta','PS','exact_Y00')):
        raise ValueError('Hartree response requires this actual physical source and exact local operator')
    binding = source['source_binding']
    digest = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if digest(binding) != source['source_identity']:
        raise ValueError('Hartree source/frame identity differs from its actual captured binding')
    potential = np.asarray(jax.device_get(functional['smooth_potential']))
    local = np.asarray(jax.device_get(functional['local_response']))
    if (potential.dtype != np.dtype(np.float64) or potential.shape != (1,*binding['fft_grid'])
            or not np.isfinite(potential).all() or local.ndim != 2 or local.shape[0] != 1
            or not np.isfinite(local).all()):
        raise ValueError('Hartree functional payload differs from the actual source/grid')
    atomic = _hartree_atomic_feature_binding(state)
    potential_binding = dict(source_identity=source['source_identity'],
        operator=functional['operator'],neutral_mean_policy=functional['neutral_mean_policy'],
        potential_payload_sha256=hashlib.sha256(potential.tobytes()).hexdigest(),
        local_response_sha256=hashlib.sha256(np.asarray(local,np.complex128).tobytes()).hexdigest(),
        atomic_feature_binding=atomic)
    return dict(source_identity=source['source_identity'],source_binding=binding,
        potential_identity=digest(potential_binding),potential_binding=potential_binding)


def _bind_charge_q0_source(zeta, state):
    """Authenticate the response's physical source/feature order before solving."""
    import hashlib
    import json

    metadata = getattr(zeta,'q0_response_metadata',None)
    if metadata is None:
        return
    source = state.get('hartree_source')
    digest = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (source is None or metadata.get('source_identity') != source['source_identity']
            or metadata.get('source_binding') != source['source_binding']
            or digest(source['source_binding']) != source['source_identity']
            or source['source_binding']['augmentation_identity'] != state['identity']
            or metadata.get('potential_binding',{}).get('atomic_feature_binding') != _hartree_atomic_feature_binding(state)
            or digest(metadata.get('potential_binding')) != metadata.get('potential_identity')):
        raise ValueError('Hartree q0 response source/frame or atomic feature binding differs from the actual augmentation')
    zeta.q0_response_source_identity = source['source_identity']


def _current_augmentation_request(request, *, wfn, meta, plan, sym, public_range):
    """Validate an internal current family against its authoritative typed basis.

    This prepares operands only. Solved-current star and public finite-q head
    admission remain owned by the canonical current family and kernel bundle.
    """
    from common.gamma_matrices import current_fit_terms
    from symmetry_maps import select_current_basis
    if request is None:
        return None
    required = {'meta','plan','centroid_indices','band_range_left','band_range_right',
                'current_basis_rows','q_full_indices'}
    if not isinstance(request,dict) or set(request) != required:
        raise ValueError('current augmentation request requires one complete typed three-current family')
    mt, pt = request['meta'], request['plan']
    if int(mt.nspinor) != 4 or pt.nspinor != 4 or pt.n_parent != plan.n_parent:
        raise ValueError('current augmentation requires the same physical four-spinor parents')
    for name in ('fft_grid','kgrid','n_rtot','nk_tot','cell_volume'):
        if not np.array_equal(np.asarray(getattr(mt,name)),np.asarray(getattr(meta,name))):
            raise ValueError(f'current augmentation {name} differs from the shared physical source')
    for name in ('fft_grid','kgrid','cell_volume'):
        if not np.array_equal(np.asarray(getattr(meta,name)),np.asarray(getattr(wfn,name))):
            raise ValueError(f'current augmentation {name} disagrees with the authoritative WFN')
    if pt.mesh_xy != plan.mesh_xy or pt.sym is not plan.sym or pt.sym is not sym:
        raise ValueError('current augmentation must use the same physical symmetry and mesh owners')
    for name in ('irr_idx','sym_idx','k_parent_frac','spin_action_full','spatial_ops','translations',
                 'fft_grid','parent_full_rows'):
        if not np.array_equal(np.asarray(getattr(pt,name)),np.asarray(getattr(plan,name))):
            raise ValueError(f'current augmentation raw-parent transport {name} differs')
    kind = getattr(pt,'coordinate_kind','fft_indices')
    basis = getattr(mt,'mu_basis',None)
    points = np.asarray(request['centroid_indices'])
    if (basis is None or kind != basis.coordinate_kind
            or not np.array_equal(points,basis.canonical_indices)
            or not np.array_equal(pt.layout.axis.packed_to_canonical,basis.layout.axis.packed_to_canonical)
            or not np.array_equal(pt.layout.axis.active_mask,basis.layout.axis.active_mask)):
        raise ValueError('current augmentation points/plan disagree with the authoritative typed T basis')
    left, right = tuple(request['band_range_left']), tuple(request['band_range_right'])
    if any(len(bounds)!=2 or any(int(v)!=v for v in bounds) for bounds in (left,right)):
        raise ValueError('current augmentation fitting windows must be integer band intervals')
    served_stop = min(public_range[1],int(meta.b_id_4_user),int(mt.b_id_4_user))
    if (min(left[0],right[0]) != public_range[0]
            or any(not public_range[0] <= bounds[0] < bounds[1] <= served_stop
                   for bounds in (left,right))):
        raise ValueError('GATE current_public_band_coverage: T windows exceed shared physical samples; separately cropped full-WFN output required')
    rows = request['current_basis_rows']
    rows = None if rows is None else np.asarray(rows,dtype=np.complex128)
    # Resolve from the physical WFN group, as gw_init does. A reduced
    # computational q table cannot choose a different component basis.
    _, selected_rows = select_current_basis(wfn.symmetry())
    for resolved_rows in (getattr(meta,'current_basis_rows',None),
                          getattr(mt,'current_basis_rows',None),selected_rows):
        if ((rows is None) != (resolved_rows is None)
                or rows is not None and not np.array_equal(rows,np.asarray(resolved_rows))):
            raise ValueError('current augmentation basis rows differ from the canonical physical-group selection')
    terms = tuple(current_fit_terms(v,rows) for v in (1,2,3))
    qrows = np.asarray(request['q_full_indices'])
    if (qrows.ndim != 1 or not len(qrows) or not np.issubdtype(qrows.dtype,np.integer)
            or len(np.unique(qrows)) != len(qrows)
            or np.any(qrows < 0) or np.any(qrows >= int(meta.nk_tot))):
        raise ValueError('current augmentation requires unique canonical full-q rows')
    return dict(meta=mt,plan=pt,points=points,coordinate_kind=kind,band_range_left=left,
                band_range_right=right,current_basis_rows=rows,vertex_terms=terms,
                q_full_indices=np.asarray(qrows,dtype=np.int32))


def _current_contract_workspace_bytes(*, qpad, mu, atoms, harmonics, radial_points,
                                      g_tile, shards):
    """Bound the separate q-owned current contraction phase, not point work.

    The paired raw and solved operands remain resident. One channel's LU
    scratch may coexist with all three solved operands. Fourier and pair
    kernels use the actual store tile; no shape-derived tile is substituted.
    Default provider field/centroid tiles are fixed internal execution sizes.
    """
    q = int(qpad)/int(shards)
    na, nh, nr, mu, gt = map(int,(atoms,harmonics,radial_points,mu,g_tile))
    mt, ft = 64, 128
    mup = ((mu+mt-1)//mt)*mt
    cloud = 16*q*mu*na*nh*nr
    matrix = 16*q*mu*mu
    moment = 16*q*mu*na*nh*2
    radial_row = 16*q*mu*nr
    degree = int(np.sqrt(nh))
    # Three q-owned input pairs plus three solved pairs; one paired solve
    # transient is separate. Bounded field gathers never retain half-clouds.
    paired_operands = 12*cloud
    # The canonical signed LU solve owns two paired working RHS banks and
    # its factor/layout workspace in addition to input and solved outputs.
    solve_scratch = 4*cloud+2*matrix
    moments = 3*moment
    moment_scratch = 2*moments+2*radial_row
    # Six existing V accumulators, six logical correction outputs and the
    # six-pair padded scan stack are simultaneously addressed at onsite.
    matrices = 15*matrix+6*16*q*mup*mup
    gathered = 16*q*mt*2*3*nr
    fields = 16*q*mt*ft
    onsite_scratch = 4*gathered+18*fields+4*6*16*q*mt*mt
    # Bound every common VSH radial factor bucket, including width-two
    # sectors and both delta/g factors, and their sparse angular metadata.
    factor_tables = 16*3*(degree+1)*(2*nr+4)**2+64*9*nh
    fourier_tables = (8*q*degree*gt*nr+16*q*na*nh*gt+8*q*degree*2*gt)
    tile = 16*q*mu*gt
    # Six delta/g outputs, their two running scans, source/smooth/current
    # physical tiles, pair operands and worst-case retained shell columns.
    # The full-G shell bound also covers the canonical short head bundle.
    fourier_and_pair_scratch = 32*tile
    return dict(paired_operands=paired_operands,solve_scratch=solve_scratch,
        moments=moments,moment_scratch=moment_scratch,matrices=matrices,
        onsite_scratch=onsite_scratch,factor_tables=factor_tables,
        fourier_tables=fourier_tables,fourier_and_pair_scratch=fourier_and_pair_scratch,
        total=sum((paired_operands,solve_scratch,moments,moment_scratch,matrices,
                   onsite_scratch,factor_tables,fourier_tables,fourier_and_pair_scratch)))


def prepare_augmentation(*, wfn, sym, meta, cfg, mesh_xy, plan,
                         centroid_indices, parent_psi, parent_faces,
                         band_range_left, band_range_right, print_fn=print,
                         write_ibz_only=True, public_band_range=None, artifact=None,
                         charge_fit_weights=None, current_request=None,
                         hartree_source_request=None):
    """Correct persistent samples and assemble the atom-local charge RHS.

    Run this before the smooth fit's conjugation/donation. The raw mode
    retains its reciprocal carrier; explicit full-WFN Lowdin returns a
    replacement carrier in state['parent_psi']. Returned canonical faces have the
    incumbent ``(nmu,mun)`` shapes/shardings. State retains only the q-owned
    raw RHS, small geometry and provenance for ``attach_local_augmentation``.
    The typed input plan owns the sample coordinates, including any explicit
    fractional atomic samples. The state adds the atom-local charge RHS.
    ``artifact`` may be the same strictly authenticated reader result from
    this fresh invocation. Supplying it avoids duplicate file reads without
    retaining global state; fitting callers never supply an unvalidated
    manifest dictionary. The default authenticates the files here.
    ``charge_fit_weights`` is the already resolved fitting policy. The same
    shared endpoint factory supplies the radial and auxiliary moment RHSs;
    this stage does not independently infer occupations or configuration.
    ``current_request`` is an internal complete-family operand request. It
    samples its authoritative T basis once with the same full-WFN factor,
    shares the atomic right endpoints, and retains three unit-loss delta/PS
    RHSs. It does not enable current fitting or establish solved-star admission.
    ``hartree_source_request`` captures a physical occupied source before
    donation, using the same packets and full-WFN factor. Its occupations
    come from the WFN, independently of the normal-equation loss weights.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.gpu_utils import device_budget_bytes, warn_over_budget
    from common import timing
    from common.staged_reshard import face_to_batch_reshard
    from isdf.local_rhs import local_density_rhs
    from psp.augmented_samples import (make_atomic_projection,
                                      make_sample_correction,build_projection_radial_cache)
    from .isdf_fitting import fitting_band_weights, fitting_weight_options
    from symmetry_maps import q_negation_index

    with timing.section('augmentation.manifest_and_cache_read'):
        if artifact is None:
            artifact = read_augmentation_manifest(cfg.paths.atomic_reconstruction_dir)
    overlap_mode = artifact.get('overlap',{}).get('mode','none')
    if overlap_mode not in ('none','full_wfn_lowdin'):
        raise ValueError(f"unsupported explicit reconstruction overlap mode {overlap_mode!r}")
    if int(meta.nspinor) != 4 or plan.nspinor != 4:
        raise ValueError("atomic ISDF reconstruction currently requires the normalized four-component charge carrier")
    if parent_psi is None or parent_psi.psi_G is None:
        raise ValueError("atomic augmentation requires the resident raw-parent reciprocal carrier")
    smooth = parent_psi.psi_G
    npar, nb, ns, ng = map(int, smooth.shape)
    coordinate_kind = getattr(plan,'coordinate_kind','fft_indices')
    if coordinate_kind not in ('fft_indices','fractional'):
        raise ValueError("atomic sampling requires a typed FFT-index or fractional centroid plan")
    if parent_faces is None and coordinate_kind != 'fractional':
        raise ValueError("FFT-index augmentation requires its canonical smooth parent faces")
    face_bands = nb if parent_faces is None else int(parent_faces[0].shape[1])
    public_range = (tuple(map(int,parent_psi.band_range)) if public_band_range is None
                    else tuple(map(int,public_band_range)))
    public_start = public_range[0]-int(parent_psi.band_range[0])
    public_bands = public_range[1]-public_range[0]
    if public_start < 0 or public_bands < 1 or public_start+public_bands > face_bands:
        raise ValueError("augmentation public band interval must lie inside the authenticated loaded face interval")
    raw_k = np.asarray(parent_psi.kvecs_frac,dtype=np.float64)
    plan_k = np.asarray(plan.k_parent_frac,dtype=np.float64)
    same_k = raw_k.shape == plan_k.shape and np.allclose(raw_k,plan_k,rtol=0.,atol=2e-12)
    bad_faces = (parent_faces is not None and (
        (int(parent_faces[0].shape[0]),int(parent_faces[0].shape[2])) != (npar,4)
        or int(parent_faces[0].shape[-1]) != int(plan.n_centroid_packed) or face_bands > nb))
    if ns != 4 or npar != plan.n_parent or not same_k or bad_faces:
        difference = np.max(np.abs(raw_k-plan_k)) if raw_k.shape == plan_k.shape else np.inf
        raise ValueError(f"augmentation source/faces/raw-parent plan disagree: source {smooth.shape}, "
                         f"faces {None if parent_faces is None else parent_faces[0].shape}, plan parents/spin {plan.n_parent}/{plan.nspinor}, "
                         f"raw/plan k shapes {raw_k.shape}/{plan_k.shape}, max absolute difference {difference:.3e}")
    fit_origin = min(int(band_range_left[0]),int(band_range_right[0]))
    if public_range[0] != fit_origin:
        raise ValueError("augmentation public band origin must equal the smooth fitting window origin")
    if overlap_mode == 'full_wfn_lowdin':
        physical_bands = int(wfn.nbands)
        if (int(artifact['overlap'].get('bands',-1)) != physical_bands
                or int(parent_psi.band_range[0]) != 0
                or face_bands < physical_bands or nb < physical_bands):
            raise ValueError("full_wfn_lowdin requires every available WFN band, loaded from band zero")
        if (artifact.get('served_moment_caches') is None
                or artifact.get('raw_parent_moments') is None):
            raise ValueError("full_wfn_lowdin requires authenticated prepared overlap of the actual served four-spinor")
    elif fit_origin != parent_psi.band_range[0]:
        raise ValueError("augmentation band weights must use the same raw-parent band origin as the smooth fit")
    current = _current_augmentation_request(current_request,wfn=wfn,meta=meta,plan=plan,sym=sym,public_range=public_range)
    # Density identity names physical bands; reciprocal/output storage keeps
    # its independent mesh carrier. Source occupations are padded below.
    source_range = (public_range[0], min(public_range[1], int(wfn.nbands),
        int(getattr(meta, 'b_id_4_user', public_range[1]))))
    source_capture = _hartree_source_request(hartree_source_request, wfn=wfn, plan=plan,
                                             public_range=source_range)
    if source_capture is not None and (overlap_mode != 'full_wfn_lowdin'
            or (artifact.get('charge_metric', {}).get('smooth_neutral_cross') != 'onsite'
                and artifact.get('charge_metric', {}).get('body_metric') != 'physical_low_local_high')
            or artifact.get('charge_metric', {}).get('moment_enrichment') != 'served_monopole'):
        raise ValueError('Hartree source capture requires SAME150 served overlap, onsite cross and exact served monopoles')
    if current is not None and overlap_mode != 'full_wfn_lowdin':
        raise ValueError('current augmentation requires the actual served full150 physical overlap')
    if current is not None and 'interpolation_degree' not in artifact['radial']:
        raise ValueError('current augmentation requires the explicit physical density interpolant')
    cached_coefficients = (artifact.get('raw_parent_moments') or {}).get('atom_C')
    if cached_coefficients is not None and overlap_mode != 'full_wfn_lowdin':
        raise ValueError("prepared full-window atomic coefficients require the actual full-WFN overlap stage")
    lattice = float(wfn.alat)*np.asarray(wfn.avec, dtype=np.float64)
    centers = np.asarray(wfn.atom_crys, dtype=np.float64)%1.
    atom_types = np.asarray(wfn.atom_types, dtype=int)
    if centers.shape != (len(atom_types),3) or not set(atom_types).issubset(artifact['tables']):
        raise ValueError("augmentation manifest does not authenticate every WFN atomic species")
    radius, weights, support = _radial_grid(artifact['radial'])
    norm_radial = weights*radius**2
    if 'interpolation_degree' in artifact['radial']:
        from isdf.augmentation import radial_coulomb_metric_interpolated
        norm_radial = radial_coulomb_metric_interpolated(radius,np.array([0]),support_radius=support,
            interpolation_degree=artifact['radial']['interpolation_degree'],
            quadrature_order=artifact['radial'].get('quadrature_order'))['moments'][0]
    nearest = _certify_spheres(centers, lattice, support)
    with timing.section('augmentation.spinor_cache'):
        caches, tails = _normalized_caches(artifact['tables'],_normalized_cache_control(artifact['cache']),support,
                                          validated_caches=artifact.get('normalized_caches'))
    directions, angles_w, lm, Y, angular_error = _orbit_angular_quadrature(
        artifact['angular'],np.asarray(sym.R_cart)[:int(plan.n_sym_spatial)])
    na, nh, nr = len(centers), len(lm), len(radius)
    control, Ptot = artifact['runtime'], int(mesh_xy.size)
    pc, bc = gcd(npar,int(control['parent_chunk'])), gcd(nb,int(control['band_chunk']))
    rp, gblock = int(control['radial_packet']), int(control['g_block'])
    if pc < 1 or bc < Ptot or bc%Ptot or rp < 1 or nr%rp or gblock < 1:
        raise ValueError("fixed augmentation chunks must divide parent/band/radial axes and band_chunk must be P-compatible")
    if write_ibz_only and getattr(sym,'q_irr_full_idx',None) is not None:
        from .qgrid_symmetry import resolve_qgrid_symmetry_tables
        coordinate_kwargs = {'coordinate_kind':coordinate_kind} if coordinate_kind == 'fractional' else {}
        resolved = resolve_qgrid_symmetry_tables(sym=sym,centroid_indices=centroid_indices,
            fft_grid=meta.fft_grid,translations=wfn.translations,
            context='atomic charge RHS IBZ write',announce_fallback=True,**coordinate_kwargs)
        write_ibz_only = bool(resolved.use_ibz)
    q_indices = (np.asarray(sym.q_irr_full_idx,dtype=int) if write_ibz_only
                 and getattr(sym,'q_irr_full_idx',None) is not None else np.arange(meta.nk_tot))
    qneg = q_negation_index(meta.kgrid) if tuple(band_range_left) != tuple(band_range_right) else None
    indexed_q = q_indices if len(q_indices) < int(meta.nk_tot) else None
    q_union = None
    if indexed_q is not None:
        from isdf.local_rhs import _selected_q_layout
        q_union, _, _ = _selected_q_layout(indexed_q,meta.kgrid,qneg)
    qpad = ((len(q_indices)+Ptot-1)//Ptot)*Ptot
    mu = int(plan.n_centroid_packed)
    rhs_bytes = 16*qpad*mu*na*nh*nr/Ptot
    reciprocal = float(wfn.blat)*np.asarray(wfn.bvec)
    positive_body = artifact.get('charge_metric',{}).get('body_metric') == 'physical_low_local_high'
    if positive_body and int(meta.sys_dim) != 3:
        raise ValueError('positive local-high charge metric requires periodic bulk 3D')
    periodic_plan = None
    if positive_body:
        from symmetry_maps import bgw_integer_q_to_fractional
        from isdf.positive_charge_metric import plan_positive_periodic_action
        qfull = np.indices(tuple(meta.kgrid)).reshape(3, -1).T
        periodic_control = artifact['charge_metric']['periodic_compensation_cache']
        periodic_path = (Path(artifact['directory'])/periodic_control['file']).resolve()
        geometry = dict(reciprocal_rows_bohr_inverse=reciprocal,
            cell_volume_bohr3=float(meta.cell_volume), atom_centres_bohr=centers@lattice,
            operator_q_fractional=bgw_integer_q_to_fractional(qfull[q_indices], meta.kgrid),
            support_radius_bohr=float(support))
        periodic_plan = plan_positive_periodic_action(mesh_xy, geometry=geometry, lm=lm,
            centroid_basis=meta.mu_basis, fft_points=int(meta.n_rtot),
            cache_path=periodic_path, cache_file_sha256=periodic_control['file_sha256'])
    onsite_cross = artifact.get('charge_metric',{}).get('smooth_neutral_cross') == 'onsite'
    moment_enrichment = artifact.get('charge_metric',{}).get('moment_enrichment') == 'served_monopole'
    rhs_copies = 4 if onsite_cross else 2
    face_bytes = 16*npar*nb*4*mu/Ptot
    source_bytes = 16*np.prod(smooth.shape)/Ptot
    factor_v_bytes = 16*qpad*mu*mu/Ptot
    # Raw rhs and its solved coefficients coexist during contraction;
    # the onsite mode has both delta and PS endpoint tables. Its combined
    # provider operand replaces the raw pair before coefficient solving,
    # so four single-endpoint tables bound raw+concatenated or RHS+solved.
    # factor, V, four faces and one smooth G tile are priced explicitly.
    # A conservative point-packing bound adds one full symmetry-orbit
    # padding allowance per mesh shard before the exact plan is built.
    point_bound = na*rp*len(directions)+Ptot*len(plan.spatial_ops)
    point_faces = 16*npar*nb*4*point_bound/Ptot
    point_workspace = _local_rhs_workspace_bytes(
        npar,meta.nk_tot,mu,point_bound,Ptot,retain_smooth=onsite_cross,
        nq_accumulator=None if q_union is None else len(q_union))
    from ffi.fft import kconv_backend, pair_resident_refusal
    convolution_backend = kconv_backend(mesh_xy)
    transform_scratch = 0.
    if convolution_backend == 'mathdx':
        if pair_resident_refusal(meta.kgrid):
            # The canonical staged handler uses scalar-spin tiles with at
            # most2048 spatial columns: typed left/right loads, transforms,
            # product and running sum. Its full output is charged above.
            columns = min(2048, (mu//int(mesh_xy.shape['x']))*
                          ((point_bound+int(mesh_xy.shape['y'])-1)//int(mesh_xy.shape['y'])))
            transform_scratch = 6.*16*int(meta.nk_tot)*columns
    else:
        # The canonical CPU plan uses full open-spin unfold/FFT banks.
        # Bound both four-spin-entry quarter banks and their load, layout,
        # transformed/conjugated intermediates independently of aliasing.
        transform_scratch = 24.*point_workspace['full_q_scalar']
    point_workspace['convolution_transform_scratch'] = transform_scratch
    point_workspace['total'] += transform_scratch
    angular_workspace = _angular_compression_workspace_bytes(
        len(q_indices),mu,na,nh,rp,len(directions),int(mesh_xy.shape['x']))
    dft_tile = 16*pc*bc*4*point_bound
    # The original phase plus its G-block padding fit in two padded
    # packets. Parent-loop completion below bounds their lifetime.
    ng_rank = ng//Ptot
    phase_width = min(gblock,ng_rank)
    phase_g = ((ng_rank+phase_width-1)//phase_width)*phase_width
    phase_bytes = 32*pc*phase_g*max(point_bound,mu)
    smooth_tile = 16*qpad*mu*min(ng,gblock)/Ptot
    overlap_bytes = (3*source_bytes+16*npar*nb*nb*3
                     +sum(32*npar*nb*np.sum(2*abs(artifact['tables'][int(z)]['kappa']))/Ptot for z in atom_types)
                     if overlap_mode == 'full_wfn_lowdin' else 0.)
    # Prepared physical D is resident host data even without the auxiliary
    # monopole stage. Charge its full per-process payload conservatively in
    # admission; device copies/rotation belong to the separate moment term.
    prepared_overlap_host_bytes = (sum(d.nbytes for d in artifact['raw_parent_moments']['atom_D'])
                                   if overlap_mode == 'full_wfn_lowdin' else 0.)
    prepared_projection_host_bytes = (sum(c.nbytes for c in cached_coefficients)
                                     if cached_coefficients is not None else 0.)
    moment_bytes = 0.
    if moment_enrichment:
        # Cached full-band D_R and both its rotated carrier and signed field
        # tables are small. The functional is assembled after the physical
        # radial packets; price its distinct full native output and union
        # panels conservatively even when their lifetimes cannot overlap.
        aux_points = na*208+Ptot*len(plan.spatial_ops)
        aux_workspace = _local_rhs_workspace_bytes(npar,meta.nk_tot,mu,aux_points,Ptot,
            nq_accumulator=None if q_union is None else len(q_union))
        aux_scratch = 0.
        if convolution_backend == 'mathdx':
            if pair_resident_refusal(meta.kgrid):
                aux_columns = min(2048,(mu//int(mesh_xy.shape['x']))*
                                  ((aux_points+int(mesh_xy.shape['y'])-1)//int(mesh_xy.shape['y'])))
                aux_scratch = 6.*16*int(meta.nk_tot)*aux_columns
        else:
            aux_scratch = 24.*aux_workspace['full_q_scalar']
        aux_workspace['convolution_transform_scratch'] = aux_scratch
        aux_workspace['total'] += aux_scratch
        aux_functions = sum(len(artifact['served_moment_caches'][int(z)]['labels']) for z in atom_types)
        moment_bytes = (aux_workspace['total']+64*npar*nb*aux_points/Ptot
            +32*npar*nb*aux_functions/Ptot+64*4*aux_functions*aux_points/int(mesh_xy.shape['y'])
            +32*qpad*mu*na/Ptot)
    current_price = 0.
    if current is not None:
        from isdf.local_rhs import _selected_q_layout
        cmu = int(current['plan'].n_centroid_packed)
        cq = current['q_full_indices']
        cpad = ((len(cq)+Ptot-1)//Ptot)*Ptot
        cqneg = q_negation_index(meta.kgrid) if current['band_range_left'] != current['band_range_right'] else None
        cindexed = cq if len(cq)<int(meta.nk_tot) else None
        cunion = None if cindexed is None else _selected_q_layout(cindexed,meta.kgrid,cqneg)[0]
        cw = _local_rhs_workspace_bytes(npar,meta.nk_tot,cmu,point_bound,Ptot,retain_smooth=True,
            nq_accumulator=None if cunion is None else len(cunion),primitive_count=max(map(len,current['vertex_terms'])))
        if convolution_backend == 'mathdx':
            ccolumns = min(2048,(cmu//int(mesh_xy.shape['x']))*((point_bound+int(mesh_xy.shape['y'])-1)//int(mesh_xy.shape['y'])))
            cscratch = 6.*16*int(meta.nk_tot)*ccolumns if pair_resident_refusal(meta.kgrid) else 0.
        else:
            cscratch = 24.*cw['full_q_scalar']
        cw['convolution_transform_scratch']=cscratch;cw['total']+=cscratch
        ca = _angular_compression_workspace_bytes(len(cq),cmu,na,nh,rp,len(directions),int(mesh_xy.shape['x']))
        craw = 16*cpad*cmu*na*nh*nr/Ptot
        # Preparation retains six raw delta/PS operands and the protected
        # current source/faces. Factor/paired solves and V/Fourier scratch
        # run only after point/quarter/angular workspaces have been released.
        # Their separately authenticated phases are priced before fit/attach.
        current_price = (6*craw+4*16*npar*nb*4*cmu/Ptot+source_bytes
            +max(0.,cw['total']-point_workspace['total'])+max(0.,ca['total']-angular_workspace['total'])
            +max(0.,32*pc*phase_g*cmu-phase_bytes)+max(0.,16*pc*bc*4*cmu-dft_tile))
        current.update(qpad=cpad,qneg=cqneg,indexed_q=cindexed,workspace=cw,raw_rhs_bytes=craw,
                       price=current_price,angular_workspace=ca)
    price = (source_bytes+4*face_bytes+rhs_copies*rhs_bytes+3*factor_v_bytes+4*point_faces
             +point_workspace['total']+dft_tile+phase_bytes+smooth_tile+overlap_bytes+moment_bytes
             +prepared_overlap_host_bytes+prepared_projection_host_bytes+angular_workspace['total']+current_price)
    if periodic_plan is not None:
        price += periodic_plan['resident_bound_bytes_per_rank']
    hartree_source_bytes = 0.
    if source_capture is not None:
        # Canonical local density scan: one additional band-owned G carrier,
        # two FFT boxes plus transform/absolute-square intermediates for its
        # local bands, and small replicated physical source outputs. No new
        # source projection, full band-pair cloud or FFT implementation.
        hartree_source_bytes = (source_bytes+6.*16*(public_bands+Ptot)*4*int(meta.n_rtot)/Ptot
            +6.*16*int(meta.n_rtot)+4.*16*na*nh*nr
            +8.*npar*point_bound/int(mesh_xy.shape['y']))
        price += hartree_source_bytes
    if current is not None:
        # These arrays are local to this procedural preparation and are not
        # returned/captured by the state. Its RHS completion blocks the last
        # use before returning; distributed lifetime tests pin that boundary.
        packet_transients = (4*point_faces+point_workspace['total']+dft_tile+phase_bytes
                             +moment_bytes+angular_workspace['total'])
        shared_persistent = price-current_price-packet_transients
        current_persistent = 4*16*npar*nb*4*cmu/Ptot+source_bytes
        current.update(phase_prices=dict(preparation=price,
            shared_charge_persistent=shared_persistent,
            protected_current_operands=current_persistent,
            released_packet_workspaces=packet_transients),
            fit_resident_extra_bytes_per_rank=shared_persistent+current_persistent+6*craw)
    budget = float(device_budget_bytes())
    if price > budget:
        warn_over_budget('isdf.augmentation',price,budget)
        raise ValueError(f"augmentation manifest requires {price/1e9:.3f} GB/rank, budget {budget/1e9:.3f}; "
                         "reduce radial/angular controls or implement a streamed local coefficient representation")
    print_fn(f"  augmentation: q={len(q_indices)}/{meta.nk_tot}, radial={nr}, lm={nh}, angular={len(directions)}, "
             f"packet={rp}, parents={pc}, bands={bc}; raw RHS {rhs_bytes/1e9:.3f} GB/rank, "
             f"resident estimate {price/1e9:.3f} GB/rank / budget {budget/1e9:.3f}")
    gs, fs, cs, qs = P(None,None,None,('x','y')), P(None,'x',None,'y'), P(None,('x','y'),None), P(('x','y'),None,None)
    if smooth.sharding != NamedSharding(mesh_xy,gs):
        raise ValueError("augmentation source must use the incumbent G-over-XY layout")
    kfrac = np.asarray(plan.k_parent_frac)
    gv = np.asarray(wfn.gvecs(k=sym.parent_k_domain), dtype=np.float64)
    if gv.shape[0] != npar or gv.shape[-1] != 3 or gv.shape[1] > ng:
        raise ValueError("raw signed G vectors disagree with augmentation source")
    gv = np.pad(gv,((0,0),(0,ng-gv.shape[1]),(0,0)))
    wavevectors = (gv+kfrac[:,None,:]) @ reciprocal
    counts = np.asarray(wfn.ngk_valid(k=sym.parent_k_domain),dtype=int)
    if counts.shape != (npar,) or np.any(counts < 1) or np.any(counts > ng):
        raise ValueError("augmentation raw-parent G logical counts disagree with source")
    wavevectors[np.arange(ng)[None,:] >= counts[:,None]] = 0.
    raw_served_D = None
    if overlap_mode == 'full_wfn_lowdin':
        from isdf.atomic_moments import raw_parent_moment_binding
        expected = raw_parent_moment_binding(wfn,k_parent_frac=kfrac,gvecs=gv,ngk_valid=counts,
            centers_cart=centers @ lattice,atom_types=atom_types,cell_volume=float(meta.cell_volume),
            physical_bands=int(wfn.nbands),
            served_cache_sha256_by_species=artifact['served_moments']['species_sha256'],
            projection_binding=(artifact['raw_parent_projection_binding']
                                if cached_coefficients is not None else None))
        raw = artifact['raw_parent_moments']
        if raw['metadata']['binding'] != expected:
            raise ValueError("raw served-moment cache disagrees with the actual WFN, parent/G geometry or full band window")
        if moment_enrichment:
            raw_served_D = [_put(np.pad(d,((0,0),(0,nb-d.shape[1]),(0,0))),mesh_xy,cs)
                            for d in raw['atom_D']]
    with timing.section('augmentation.projection_cache'):
        projection_caches = {}
        Kmax = float(np.max(np.linalg.norm(wavevectors,axis=-1)))*(1+1e-12)
        if artifact.get('fourier_caches') is not None:
            for z,pair in artifact['fourier_caches'].items():
                if any(float(table['momentum'][-1]) < Kmax for table in pair.values()):
                    raise ValueError("explicit atomic Fourier cache does not cover the WFN momenta")
                projection_caches[z] = pair['projection']
        elif 'projection_momentum_points' in artifact['cache']:
            for z in set(atom_types):
                projection_caches[int(z)] = build_projection_radial_cache(artifact['tables'][int(z)],
                    momentum_max=Kmax,momentum_points=int(artifact['cache']['projection_momentum_points']),
                    relative_tolerance=float(artifact['cache'].get('projection_relative_tolerance',1e-10)),
                    absolute_tolerance=float(artifact['cache'].get('projection_absolute_tolerance',1e-12)),
                    validation_points=int(artifact['cache'].get('projection_validation_points',64)))
    kernels = _tile_kernels(mesh_xy,pc,bc)
    project = make_atomic_projection(mesh_xy)
    correct = make_sample_correction(mesh_xy,face_sharding=NamedSharding(mesh_xy,fs))
    scale = np.sqrt(float(meta.cell_volume)/int(meta.n_rtot))
    mu_points = np.asarray(centroid_indices,dtype=np.float64)
    if coordinate_kind == 'fft_indices':
        mu_points = mu_points/np.asarray(meta.fft_grid)
    mu_points = plan.layout.axis.pack_host(mu_points,axis=0,fill_value=0.)
    active = np.asarray(plan.layout.axis.active_mask)
    coefficients = []
    phase_kernel = _point_phase_kernel(mesh_xy)
    nmu = None if parent_faces is None else parent_faces[0]
    if parent_faces is None and overlap_mode != 'full_wfn_lowdin':
        with timing.section('augmentation.fractional_mu_samples'):
            nmu = _smooth_point_faces(smooth,wavevectors,mu_points,active,mesh_xy,lattice=lattice,
                parent_chunk=pc,band_chunk=bc,g_block=gblock,fft_points=meta.n_rtot)
    if nmu is not None and face_bands < nb:
        # The G-slot carrier pads bands to the whole mesh; public faces
        # need only X divisibility and keep their existing smaller extent.
        # Internal zero padding permits one fixed P-compatible band tile.
        nmu = jax.jit(lambda a: jnp.pad(a,((0,0),(0,nb-face_bands),(0,0),(0,0))),
                      out_shardings=NamedSharding(mesh_xy,fs))(nmu)
    with timing.section('augmentation.project_and_initial_samples'):
        if cached_coefficients is not None:
            # Exact unrotated full150 projections, authenticated against the
            # actual WFN and dual owners above. The same A acts before crop.
            coefficients = [_put(np.pad(c,((0,0),(0,nb-c.shape[1]),(0,0))),mesh_xy,cs)
                            for c in cached_coefficients]
        delta_iter = (() if cached_coefficients is not None else
                      _sample_geometry(mu_points,active,centers,lattice,caches,atom_types,kfrac,support,scale))
        for atom,(delta,phase) in enumerate(delta_iter):
            z, data = int(atom_types[atom]), artifact['tables'][int(atom_types[atom])]
            delta_device = _put(delta,mesh_xy,P(None,None,'y'))
            coeff = jnp.zeros((npar,nb,len(delta)),jnp.complex128,device=NamedSharding(mesh_xy,cs))
            for p0 in range(0,npar,pc):
                table = _atomic_fourier_table(data,wavevectors[p0:p0+pc],
                    dict(center_cart=centers[atom] @ lattice,cell_volume=meta.cell_volume),
                    mesh_xy,radial_cache=projection_caches.get(z))
                phase_device = _put(phase[p0:p0+pc],mesh_xy,P(None,'y'))
                for b0 in range(0,nb,bc):
                    source = kernels['source'](smooth,jnp.int32(p0),jnp.int32(b0))[:,:,:2]
                    tile_coeff = project(source,table)
                    coeff = kernels['store_coeff'](coeff,tile_coeff,jnp.int32(p0),jnp.int32(b0))
                    if overlap_mode == 'none':
                        face = kernels['face'](nmu,jnp.int32(p0),jnp.int32(b0))
                        face = correct(face,tile_coeff,delta_device,phase_device)
                        nmu = kernels['update_face'](nmu,face,jnp.int32(p0),jnp.int32(b0))
            coefficients.append(coeff)
    overlap_receipt = replacement_parent = None
    with timing.section('augmentation.full_wfn_overlap_and_samples'):
        if overlap_mode == 'full_wfn_lowdin':
            output_bands = ((public_bands+Ptot-1)//Ptot)*Ptot
            physical_stop = min(public_range[1],int(getattr(meta,'b_id_4_user',public_range[1])))
            geometry = dict(parent_chunk=pc,tables=artifact['tables'],atom_types=atom_types,
                wavevectors=wavevectors,centers_cart=centers @ lattice,cell_volume=meta.cell_volume,
                physical_bands=physical_bands,output_bands=output_bands,public_start=public_start,
                physical_stop=physical_stop,served_caches=artifact['served_moment_caches'],
                raw_served_D=artifact['raw_parent_moments']['atom_D'])
            smooth,nmu,coefficients,overlap_receipt = _full_wfn_rotation(
                smooth,nmu,coefficients,geometry,mesh_xy)
            if raw_served_D is not None:
                rotation = _band_rotation_kernel(mesh_xy,pc,'coefficients',output_bands,public_start,physical_stop)
                factor = _put(overlap_receipt['inverse_sqrt'],mesh_xy,P())
                raw_served_D = [rotation(d,factor) for d in raw_served_D]
            replacement_parent = parent_psi._replace(psi_G=smooth,band_range=public_range,faces=None)
            # The full factor acted before this public selection. The existing
            # band-window weights now refer to the original fitting origin.
            nb,public_start = output_bands,0
            face_bands = public_bands
            bc = gcd(nb,int(control['band_chunk']))
            if bc < Ptot or bc%Ptot:
                raise ValueError("rotated public band carrier must admit the fixed mesh-compatible band chunk")
            kernels = _tile_kernels(mesh_xy,pc,bc)
            if nmu is None:
                # The full physical Gram/factor was measured first. Sampling
                # the selected G carrier avoids unused full-WFN point bands;
                # its rotation owner has already zeroed every public ghost.
                with timing.section('augmentation.fractional_mu_samples'):
                    nmu = _smooth_point_faces(smooth,wavevectors,mu_points,active,mesh_xy,lattice=lattice,
                        parent_chunk=pc,band_chunk=bc,g_block=gblock,fft_points=meta.n_rtot)
            delta_iter = _sample_geometry(mu_points,active,centers,lattice,caches,atom_types,kfrac,support,scale)
            for atom,(delta,phase) in enumerate(delta_iter):
                delta_device = _put(delta,mesh_xy,P(None,None,'y'))
                for p0 in range(0,npar,pc):
                    phase_device = _put(phase[p0:p0+pc],mesh_xy,P(None,'y'))
                    for b0 in range(0,nb,bc):
                        tile_coeff = kernels['read_coeff'](coefficients[atom],jnp.int32(p0),jnp.int32(b0))
                        face = kernels['face'](nmu,jnp.int32(p0),jnp.int32(b0))
                        face = correct(face,tile_coeff,delta_device,phase_device)
                        nmu = kernels['update_face'](nmu,face,jnp.int32(p0),jnp.int32(b0))
            print_fn(f"  augmentation explicit full-WFN Lowdin: {physical_bands} physical bands, "
                     f"min metric eigenvalue {np.min(overlap_receipt['eigenvalue_min']):.6e}, "
                     f"max restored-Gram error {np.max(overlap_receipt['factor_isometry_error']):.3e}; "
                     "same factor on reciprocal, atomic and sample carriers, DFT energy labels retained")
    corrected_faces = (nmu,_face_complement(nmu,mesh_xy))
    if source_capture is not None:
        import hashlib
        import json
        from common.wfn_layout import band_sphere_spec
        from gw.qsgw_density import rho_from_wfns

        lo, hi = source_capture['public_band_range']
        source_capture['public_occupations'] = np.pad(
            source_capture['occupations'][:, lo:hi], ((0,0),(0,nb-(hi-lo))))
        with timing.section('augmentation.occupied_smooth_density'):
            source_G = jax.jit(lambda a: a,
                out_shardings=NamedSharding(mesh_xy,band_sphere_spec()))(smooth)
            source_capture['smooth_density'] = rho_from_wfns(
                source_G, source_capture['public_occupations'],source_capture['parent_kweights'],
                mesh=mesh_xy,box_index=parent_psi.sphere_index,fft_grid=meta.fft_grid,
                cell_volume=float(meta.cell_volume),spin_degeneracy=1.,sym=sym,
                sym_perm=sym.fft_grid_pullback(sym.active_symmetry_rows,tuple(meta.fft_grid)),
                print_fn=print_fn)
            source_capture['smooth_density'].block_until_ready()
            del source_G
        binding = dict(schema='lorrax.augmentation_occupied_source.v1',
            augmentation_identity=artifact['identity'],prepared_raw_binding=expected,
            physical_bands=int(wfn.nbands),public_band_range=[lo,hi],
            **_physical_full_wfn_frame_binding(overlap_receipt['inverse_sqrt'],int(wfn.nbands)),
            occupations_sha256=source_capture['occupations_sha256'],
            full_kweights_sha256=source_capture['full_kweights_sha256'],
            spin_degeneracy=1.,fft_grid=list(map(int,meta.fft_grid)),
            cell_volume=float(meta.cell_volume),source_frame_policy='same_actual_served_four_spinor_full_WFN_Lowdin')
        source_capture['source_binding'] = binding
        source_capture['source_identity'] = hashlib.sha256(
            json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        source_capture['local_ps_density'] = jnp.zeros((1,na,nh,nr),jnp.complex128,
                                                       device=NamedSharding(mesh_xy,P()))
        source_capture['local_delta_density'] = jnp.zeros_like(source_capture['local_ps_density'])
    if current is not None:
        cpplan = current['plan']
        same_points = (current['coordinate_kind']==coordinate_kind
            and np.array_equal(current['points'],np.asarray(centroid_indices))
            and np.array_equal(cpplan.layout.axis.packed_to_canonical,plan.layout.axis.packed_to_canonical))
        if same_points:
            with timing.section('augmentation.current_preserved_operands'):
                current['parent_faces'] = tuple(_preserve_for_donating_fit(a) for a in corrected_faces)
        else:
            cpoints = np.asarray(current['points'],dtype=np.float64)
            if current['coordinate_kind']=='fft_indices':
                cpoints = cpoints/np.asarray(meta.fft_grid)
            cpoints = cpplan.layout.axis.pack_host(cpoints,axis=0,fill_value=0.)
            cactive = np.asarray(cpplan.layout.axis.active_mask)
            with timing.section('augmentation.current_T_samples'):
                cnmu = _smooth_point_faces(smooth,wavevectors,cpoints,cactive,mesh_xy,lattice=lattice,
                    parent_chunk=pc,band_chunk=bc,g_block=gblock,fft_points=meta.n_rtot)
                for atom,(delta,phase) in enumerate(_sample_geometry(cpoints,cactive,centers,lattice,caches,atom_types,kfrac,support,scale)):
                    delta_device=_put(delta,mesh_xy,P(None,None,'y'))
                    for p0 in range(0,npar,pc):
                        phase_device=_put(phase[p0:p0+pc],mesh_xy,P(None,'y'))
                        for b0 in range(0,nb,bc):
                            tile_coeff=kernels['read_coeff'](coefficients[atom],jnp.int32(p0),jnp.int32(b0))
                            face=kernels['face'](cnmu,jnp.int32(p0),jnp.int32(b0))
                            cnmu=kernels['update_face'](cnmu,correct(face,tile_coeff,delta_device,phase_device),jnp.int32(p0),jnp.int32(b0))
                current['parent_faces']=(cnmu,_face_complement(cnmu,mesh_xy))
        with timing.section('augmentation.current_preserved_operands'):
            current['parent_psi'] = replacement_parent._replace(psi_G=_preserve_for_donating_fit(smooth))
        current['preserved_source_bytes_per_rank'] = source_bytes
        current['same_point_faces_preserved'] = same_points
        current['weight_l'],current['weight_r']=fitting_band_weights(nb,current['band_range_left'],current['band_range_right'])
        cshape=(current['qpad'],int(cpplan.n_centroid_packed),na*nh*nr)
        current['rhs']=[jnp.zeros(cshape,jnp.complex128,device=NamedSharding(mesh_xy,qs)) for _ in range(3)]
        current['smooth_rhs']=[jnp.zeros(cshape,jnp.complex128,device=NamedSharding(mesh_xy,qs)) for _ in range(3)]
        current['selector']=current['store_rhs']=None
    norm_source,norm_local = _orbital_norm_kernels(mesh_xy)
    smooth_norm = norm_source(smooth)
    charge_delta = jnp.zeros((npar,nb),jnp.float64,device=NamedSharding(mesh_xy,P(None,'x')))
    weight_options = fitting_weight_options(charge_fit_weights)
    weight_l,weight_r = fitting_band_weights(nb,band_range_left,band_range_right,
                                           **weight_options)
    rhs = jnp.zeros((qpad,mu,na*nh*nr),jnp.complex128,device=NamedSharding(mesh_xy,qs))
    smooth_rhs = (jnp.zeros(rhs.shape,jnp.complex128,device=NamedSharding(mesh_xy,qs))
                  if onsite_cross else None)
    right_plan = compressor = selector = store_rhs = sampler = None
    occupied_trace = occupied_projection = None
    to_q = face_to_batch_reshard(mesh_xy)
    with timing.section('augmentation.atomic_density_rhs'):
        for r0 in range(0,nr,rp):
            relative = radius[r0:r0+rp,None,None]*directions[None,:,:]
            # Keep center+dr unwrapped. These are physical Bloch densities,
            # not cell-periodic values: wrapping before Y_lm projection would
            # lose exp(+2pi i q.L) on every boundary-crossing density sample.
            # The canonical service authenticates periodic membership and
            # computes wraps against the original unwrapped source positions.
            points = (centers[:,None,None,:]+relative[None,:,:,:] @ np.linalg.inv(lattice)).reshape(-1,3)
            right_plan = _packet_plan(plan,points,right_plan)
            points = right_plan.layout.axis.pack_host(points,axis=0,fill_value=0.)
            live = np.asarray(right_plan.layout.axis.active_mask)
            npoint = int(right_plan.n_centroid_packed)
            if sampler is None:
                sampler = _point_samples_kernel(mesh_xy,pc,bc,npoint,gblock,meta.n_rtot)
                compressor,nf,nfp = _compress_rhs_kernel(mesh_xy,right_plan,na,nh,rp,Y.conj()*angles_w)
                packet_q = q_indices if indexed_q is None else np.arange(len(q_indices))
                selector,store_rhs = _rhs_storage_kernels(mesh_xy,packet_q,qpad,na,nh,nr,rp,nf)
                if source_capture is not None:
                    from isdf.atomic_hartree import make_occupied_point_trace,make_occupied_density_projection
                    occupied_trace = make_occupied_point_trace(right_plan,source_capture['public_occupations'],
                        source_capture['full_kweights'],cell_volume=float(meta.cell_volume),spin_degeneracy=1.)
                    indices,angular = _angular_bucket_tables(right_plan,na,rp,Y.conj()*angles_w,int(mesh_xy.shape['y']))
                    occupied_projection = make_occupied_density_projection(
                        mesh_xy,indices,angular,output_shape=(na,rp,nh))
            ps = jnp.zeros((npar,nb,4,npoint),jnp.complex128,device=NamedSharding(mesh_xy,fs))
            ae = jnp.zeros((npar,nb,4,npoint),jnp.complex128,device=NamedSharding(mesh_xy,fs))
            cart_device,live_device = _put(points @ lattice,mesh_xy,P()),_put(live.astype(float),mesh_xy,P())
            for p0 in range(0,npar,pc):
                K = _put(wavevectors[p0:p0+pc],mesh_xy,P(None,('x','y'),None))
                phases = phase_kernel(K,cart_device)
                for b0 in range(0,nb,bc):
                    tile = sampler(kernels['source'](smooth,jnp.int32(p0),jnp.int32(b0)),phases,live_device)
                    ps = kernels['update_face'](ps,tile,jnp.int32(p0),jnp.int32(b0))
                    ae = kernels['update_face'](ae,tile,jnp.int32(p0),jnp.int32(b0))
                ae.block_until_ready()
            # The two independent accumulators prevent donation aliasing the
            # smooth endpoint, including XLA's elimination of an a+0 copy.
            delta_iter = _sample_geometry(points,live,centers,lattice,caches,atom_types,kfrac,support,scale)
            for atom,(delta,phase) in enumerate(delta_iter):
                delta_device = _put(delta,mesh_xy,P(None,None,'y'))
                for p0 in range(0,npar,pc):
                    phase_device = _put(phase[p0:p0+pc],mesh_xy,P(None,'y'))
                    for b0 in range(0,nb,bc):
                        tile_coeff = kernels['read_coeff'](coefficients[atom],jnp.int32(p0),jnp.int32(b0))
                        face = kernels['face'](ae,jnp.int32(p0),jnp.int32(b0))
                        face = correct(face,tile_coeff,delta_device,phase_device)
                        ae = kernels['update_face'](ae,face,jnp.int32(p0),jnp.int32(b0))
            volume_weights = np.broadcast_to(norm_radial[r0:r0+rp][None,:,None]
                                *angles_w[None,None,:],(na,rp,len(directions))).reshape(-1)
            volume_weights = right_plan.layout.axis.pack_host(volume_weights,axis=0,fill_value=0.)
            volume_weights *= int(meta.n_rtot)/float(meta.cell_volume)
            charge_delta = charge_delta+norm_local(ps,ae,_put(volume_weights,mesh_xy,P('y')))
            if source_capture is not None:
                with timing.section('augmentation.occupied_atomic_density'):
                    ps_density,ae_density = occupied_trace(ps),occupied_trace(ae)
                    density = jax.jit(lambda p,a: jnp.stack((p,a-p)),
                        out_shardings=NamedSharding(mesh_xy,P(None,'y')))(ps_density,ae_density)
                    projected = occupied_projection(density).transpose(0,1,3,2)
                    source_capture['local_ps_density'] = source_capture['local_ps_density'].at[:,:,:,r0:r0+rp].set(projected[:1])
                    source_capture['local_delta_density'] = source_capture['local_delta_density'].at[:,:,:,r0:r0+rp].set(projected[1:])
                    jax.block_until_ready((source_capture['local_ps_density'],source_capture['local_delta_density']))
                    del density,projected,ps_density,ae_density
            ae_faces,ps_faces = (ae,_face_complement(ae,mesh_xy)),(ps,_face_complement(ps,mesh_xy))
            local = local_density_rhs(centroid_faces=corrected_faces,
                        atom_ae_faces=ae_faces,atom_ps_faces=ps_faces,left_plan=plan,right_plan=right_plan,
                        weight_l=weight_l,weight_r=weight_r,kgrid=meta.kgrid,mesh_xy=mesh_xy,q_neg_idx=qneg,
                        return_smooth=onsite_cross,q_indices=indexed_q)
            if onsite_cross:
                local, local_smooth = local
            # Q selection commutes with this q-independent angular map.
            # Complete LR+RL above, then project only the rows the fit keeps.
            packet = to_q(compressor(selector(local)))
            rhs = store_rhs(rhs,packet,jnp.int32(r0))
            rhs.block_until_ready()
            if onsite_cross:
                smooth_packet = to_q(compressor(selector(local_smooth)))
                smooth_rhs = store_rhs(smooth_rhs,smooth_packet,jnp.int32(r0))
                smooth_rhs.block_until_ready()
                del local_smooth, smooth_packet
            # Only compressed charge operands survive into the current
            # scans. The exact same AE/PS right faces serve every channel.
            del local, packet
            if current is not None:
                if current['selector'] is None:
                    cids=current['q_full_indices'] if current['indexed_q'] is None else np.arange(len(current['q_full_indices']))
                    current['selector'],current['store_rhs']=_rhs_storage_kernels(mesh_xy,cids,current['qpad'],na,nh,nr,rp,nf)
                for channel,terms in enumerate(current['vertex_terms']):
                    cd,csmooth=local_density_rhs(centroid_faces=current['parent_faces'],
                        atom_ae_faces=ae_faces,atom_ps_faces=ps_faces,
                        left_plan=current['plan'],right_plan=right_plan,weight_l=current['weight_l'],weight_r=current['weight_r'],
                        kgrid=meta.kgrid,mesh_xy=mesh_xy,q_neg_idx=current['qneg'],q_indices=current['indexed_q'],
                        vertex_terms=terms,return_smooth=True)
                    for key,value in (('rhs',cd),('smooth_rhs',csmooth)):
                        cpacket=to_q(compressor(current['selector'](value)))
                        current[key][channel]=current['store_rhs'](current[key][channel],cpacket,jnp.int32(r0))
                        current[key][channel].block_until_ready()
                    del cd,csmooth,cpacket,value
            # The next endpoint scan must not retain the previous full-q
            # rectangular output. Only the compressed q-owner table survives.
            del ae_faces,ps_faces
    monopole_rhs = moment_receipt = None
    if moment_enrichment:
        with timing.section('augmentation.served_monopole_rhs'):
            monopole_rhs,moment_receipt = _served_monopole_rhs(plan,corrected_faces,coefficients,raw_served_D,
                dict(caches=artifact['served_moment_caches'],atom_types=atom_types,centers=centers,
                     lattice=lattice,scale=scale,weight_l=weight_l,weight_r=weight_r,kgrid=meta.kgrid,
                     qneg=qneg,indexed_q=indexed_q,q_indices=q_indices,qpad=qpad),mesh_xy,
                     source_capture=source_capture)
            monopole_rhs.block_until_ready()
    state = dict(rhs=rhs,radius=radius,weights_dr=weights,lm=lm,centers_cart=centers @ lattice,
        q_full_indices=q_indices,reciprocal=reciprocal,cell_volume=float(meta.cell_volume),
        fft_points=int(meta.n_rtot),support_radius=support,kgrid=tuple(meta.kgrid),sym=sym,
        identity=artifact['identity'],tail_relative_norm=tails,angular_gram_error=angular_error,
        nearest_atom_image=nearest,raw_rhs_bytes_per_rank=rhs_bytes,resident_estimate_bytes_per_rank=price)
    if current is not None:
        for key in ('selector','store_rhs','weight_l','weight_r'):
            del current[key]
        current.update(radius=radius,weights_dr=weights,lm=lm,centers_cart=centers@lattice,
            reciprocal=reciprocal,cell_volume=float(meta.cell_volume),fft_points=int(meta.n_rtot),
            support_radius=support,minimum_atom_image_distance=nearest,identity=artifact['identity'],
            sym=sym,kgrid=tuple(meta.kgrid),
            shared_physical_bands=int(wfn.nbands),shared_public_band_range=public_range,
            source_frame_policy='same_actual_served_four_spinor_full_WFN_Lowdin',unit_endpoint_loss=True)
        current.update({key:artifact['radial'][key]
                       for key in ('interpolation_degree','quadrature_order','fourier_points')
                       if key in artifact['radial']})
        if overlap_receipt is not None:
            current['overlap_receipt']=overlap_receipt
        state['current']=current
    state['local_rhs_workspace_bytes_per_rank'] = point_workspace
    if source_capture is not None:
        for key in ('public_occupations','occupations','full_kweights','parent_kweights'):
            del source_capture[key]
        source_capture['resident_bound_bytes_per_rank'] = hartree_source_bytes
        state['hartree_source'] = source_capture
    state['angular_compression_workspace_bytes_per_rank'] = angular_workspace
    state['prepared_served_overlap_host_bytes_per_process'] = prepared_overlap_host_bytes
    state['resident_rhs_copies'] = rhs_copies
    state['indexed_local_q_union'] = q_union
    if artifact.get('prepared_fourier_cache') is not None:
        state['prepared_fourier_cache'] = artifact['prepared_fourier_cache']
    if onsite_cross:
        state['smooth_rhs'] = smooth_rhs
        state['smooth_neutral_cross'] = 'onsite'
    if moment_enrichment:
        state['monopole_rhs'] = monopole_rhs
        state['moment_enrichment'] = 'served_monopole'
        state['served_moment_receipt'] = moment_receipt
        state['served_moment_workspace_bytes_per_rank'] = moment_bytes
        state['served_moment_local_workspace_bytes_per_rank'] = aux_workspace
    state['charge_factor_equilibration'] = artifact.get('charge_fit',{}).get('conditioning')
    state['charge_fit_weights'] = dict(weight_options) if weight_options else None
    if positive_body:
        state['body_metric'] = 'physical_low_local_high'
        control = dict(artifact['charge_metric']['periodic_compensation_cache'])
        control['file'] = str((Path(artifact['directory'])/control['file']).resolve())
        state['periodic_compensation_cache'] = control
        state['periodic_plan'] = periodic_plan
        state['periodic_action_workspace'] = periodic_plan['receipt']
    state['prepared_atomic_projection_host_bytes_per_process'] = prepared_projection_host_bytes
    state['atomic_projection_source'] = ('prepared_full_window_v2' if cached_coefficients is not None
                                         else 'canonical_runtime_projection_v1')
    state['orbital_norm_change_estimate'] = charge_delta
    state['smooth_orbital_norm'] = smooth_norm
    state['orbital_norm_quadrature_convention'] = ('local_density_interpolant'
        if 'interpolation_degree' in artifact['radial'] else 'nodal_dr_quadrature')
    if overlap_receipt is not None:
        state['overlap_receipt'] = overlap_receipt
        overlap_receipt['k_parent_frac'] = np.asarray(plan.k_parent_frac)
        state['parent_psi'] = replacement_parent
    source_origin = public_range[0] if replacement_parent is not None else int(parent_psi.band_range[0])
    diagnostic_bands = min(face_bands,int(getattr(meta,'b_id_4_user',source_origin+face_bands))-source_origin)
    norm_change = float(jax.device_get(jnp.max(abs(charge_delta[:,:diagnostic_bands]))))
    corrected_norm = jax.jit(lambda a,b: a+b)(smooth_norm,charge_delta)
    norm_defect = float(jax.device_get(jnp.max(abs(corrected_norm[:,:diagnostic_bands]-1.))))
    state['maximum_orbital_norm_change_estimate'] = norm_change
    state['maximum_corrected_norm_defect_estimate'] = norm_defect
    for key in ('interpolation_degree','quadrature_order','fourier_points'):
        if key in artifact['radial']:
            state[key] = artifact['radial'][key]
    print_fn(f"  augmentation RHS complete: normalized-tail norms {tails}, angular Gram {angular_error:.3e}; "
             f"reciprocal source convention {overlap_mode}")
    print_fn(f"  augmentation orbital norm diagnostic (coarse atomic quadrature): max change {norm_change:.6e}, "
             f"max |S_nn-1| {norm_defect:.6e}; overlap convention {overlap_mode}")
    if public_start != 0 or public_bands < nb:
        nmu = jax.jit(lambda a: a[:,public_start:public_start+public_bands],
                     out_shardings=NamedSharding(mesh_xy,fs))(nmu)
        corrected_faces = (nmu,_face_complement(nmu,mesh_xy))
    return corrected_faces,state


def attach_local_augmentation(zeta_g, state, *, body_contract=None):
    """Bind the saved local RHS to the smooth fit's factor and exact G sphere."""
    from isdf.atomic_coulomb import radial_coulomb_provider
    from symmetry_maps import bgw_integer_q_to_fractional

    if state['rhs'].shape[:2] != (zeta_g.store.Q_pad,zeta_g.store.mu_pad):
        raise ValueError("augmentation selected q/packed-mu carrier disagrees with fitted smooth ZetaG")
    _bind_charge_q0_source(zeta_g,state)
    sym = state['sym']
    qfull = np.asarray(sym.kvecs_asints)
    qfrac = bgw_integer_q_to_fractional(qfull[state['q_full_indices']],state['kgrid'])
    gv = np.asarray(zeta_g.gvec_components).transpose(0,2,1)
    kg = (gv+qfrac[:,None,:]) @ state['reciprocal']
    radial_controls = {key:state[key] for key in ('interpolation_degree','quadrature_order','fourier_points') if key in state}
    if state.get('body_metric') == 'physical_low_local_high':
        from isdf.coulomb_fourier_cache import load_periodic_compensation_cache
        from isdf.positive_charge_metric import positive_radial_coulomb_provider
        if (not isinstance(body_contract, dict)
                or set(body_contract) != {'sys_dim','bare_coulomb_cutoff_ry'}
                or body_contract['sys_dim'] != 3
                or not np.isfinite(body_contract['bare_coulomb_cutoff_ry'])
                or body_contract['bare_coulomb_cutoff_ry'] <= 0
                or state.get('moment_enrichment') != 'served_monopole'
                or 'monopole_rhs' not in state or 'smooth_rhs' in state):
            raise ValueError('positive charge requires the canonical bulk body contract and delta/exact-M0 operands')
        control = state['periodic_compensation_cache']
        geometry = dict(reciprocal_rows_bohr_inverse=state['reciprocal'],
            cell_volume_bohr3=state['cell_volume'], atom_centres_bohr=state['centers_cart'],
            operator_q_fractional=qfrac, support_radius_bohr=state['support_radius'])
        cache = load_periodic_compensation_cache(control['file'], mesh=zeta_g.mesh,
            expected_file_sha256=control['file_sha256'], geometry=geometry, lm=state['lm'])
        if 'prepared_fourier_cache' in state:
            radial_controls['prepared_cache'] = state['prepared_fourier_cache']
        provider = positive_radial_coulomb_provider(zeta_g, state['rhs'],
            monopole_rhs=state['monopole_rhs'], radius=state['radius'],
            weights_dr=state['weights_dr'], lm=state['lm'], centers_cart=state['centers_cart'],
            q_frac=qfrac, gvec_components=np.asarray(zeta_g.gvec_components), q_plus_G_cart=kg,
            cell_volume=state['cell_volume'], fft_points=state['fft_points'],
            support_radius=state['support_radius'], minimum_atom_image_distance=state['nearest_atom_image'],
            body_cutoff_ry=body_contract['bare_coulomb_cutoff_ry'], periodic_cache=cache,
            periodic_plan=state['periodic_plan'],
            **radial_controls)
        provider['identity'] = state['identity']
        zeta_g.local_augmentation = provider
        del state['rhs'], state['monopole_rhs']
        return zeta_g
    if 'smooth_rhs' in state:
        if state.get('smooth_neutral_cross') != 'onsite' or state['smooth_rhs'].shape != state['rhs'].shape:
            raise ValueError("onsite smooth RHS must use the same selected q/packed-mu/local axes")
        radial_controls['smooth_rhs'] = state['smooth_rhs']
    if 'monopole_rhs' in state:
        if state.get('moment_enrichment') != 'served_monopole' or 'smooth_rhs' not in state:
            raise ValueError("served monopole RHS requires the explicit onsite smooth-neutral charge metric")
        radial_controls['monopole_rhs'] = state['monopole_rhs']
    if 'prepared_fourier_cache' in state:
        radial_controls['prepared_cache'] = state['prepared_fourier_cache']
    provider = radial_coulomb_provider(zeta_g,state['rhs'],radius=state['radius'],weights_dr=state['weights_dr'],
        lm=state['lm'],centers_cart=state['centers_cart'],q_plus_G_cart=kg,
        cell_volume=state['cell_volume'],fft_points=state['fft_points'],support_radius=state['support_radius'],
        **radial_controls)
    provider['identity'] = state['identity']
    zeta_g.local_augmentation = provider
    if 'smooth_rhs' in state:
        # The combined provider now owns both raw endpoints. Do not retain
        # separate state references alongside its concatenated RHS and solve.
        # Geometry, pricing and already-saved overlap diagnostics stay intact.
        del state['rhs'], state['smooth_rhs']
        state.pop('monopole_rhs',None)
    return zeta_g


def attach_current_augmentation(zetas, state, *, body_contract):
    """Consume a complete current operand into its OWN three fitted factors.

    ``state`` is ``prepare_augmentation(...,current_request=...)['current']``.
    ``body_contract`` contains the canonical reciprocal prefactor, Cartesian
    head, physical body mask and authenticated public body-head slot bundle.
    The physical current writer and solved-star admission are separate gates;
    this internal no-file seam does not enable the public GW current flow.
    """
    from isdf.atomic_breit import radial_breit_providers
    from symmetry_maps import bgw_integer_q_to_fractional

    required = {'reciprocal_prefactor','head_cartesian','body_mask','body_head_slots'}
    if (not isinstance(zetas,dict) or set(zetas)!={1,2,3} or not isinstance(body_contract,dict)
            or set(body_contract)!=required or body_contract['body_head_slots'] is None):
        raise ValueError('current augmentation requires complete OWN3 fits and the canonical body/head contract')
    if (not isinstance(state,dict)
            or state.get('source_frame_policy')!='same_actual_served_four_spinor_full_WFN_Lowdin'
            or state.get('unit_endpoint_loss') is not True
            or not isinstance(state.get('rhs'),list) or len(state['rhs'])!=3
            or not isinstance(state.get('smooth_rhs'),list) or len(state['smooth_rhs'])!=3):
        raise ValueError('current augmentation requires three prepared unit-loss delta/PS operands')
    channels = tuple(zetas[v] for v in (1,2,3))
    for v,z in zip((1,2,3),channels):
        if (z.mesh is not state['plan'].mesh_xy or z.mu_basis is not state['meta'].mu_basis
                or getattr(z,'fit_vertex_mu_L',None)!=v
                or getattr(z,'fit_augmentation_identity',None)!=state['identity']
                or not np.array_equal(getattr(z,'fit_q_full_indices',None),state['q_full_indices'])
                or z.store.mu_pad!=state['plan'].n_centroid_packed
                or z.store.Q!=len(state['q_full_indices'])):
            raise ValueError('current augmentation q/basis/OWN-factor/source binding differs from its fit')
    from common.gpu_utils import device_budget_bytes,warn_over_budget
    workspace=_current_contract_workspace_bytes(qpad=channels[0].store.Q_pad,
        mu=channels[0].store.mu_pad,atoms=len(state['centers_cart']),harmonics=len(state['lm']),
        radial_points=len(state['radius']),g_tile=channels[0].store.g_tile,shards=channels[0].mesh.size)
    phase=state['phase_prices']
    contraction_price=(phase['shared_charge_persistent']+phase['protected_current_operands']+workspace['total'])
    budget=float(device_budget_bytes())
    if contraction_price>budget:
        warn_over_budget('isdf.current_augmentation_contract',contraction_price,budget)
        raise ValueError('current augmentation contraction exceeds its q-owned device memory budget')
    state['contraction_workspace_bytes_per_rank']=workspace
    phase['contraction']=contraction_price
    qfrac = bgw_integer_q_to_fractional(
        np.asarray(state['sym'].kvecs_asints)[state['q_full_indices']],state['kgrid'])
    gv = np.asarray(channels[0].gvec_components).transpose(0,2,1)
    kg = (gv+qfrac[:,None])@state['reciprocal']
    controls={key:state[key] for key in ('interpolation_degree','quadrature_order','fourier_points') if key in state}
    providers=radial_breit_providers(channels,state['rhs'],smooth_rhs=state['smooth_rhs'],
        body_metric='compensated',radius=state['radius'],lm=state['lm'],centers_cart=state['centers_cart'],
        q_plus_G_cart=kg,cell_volume=state['cell_volume'],fft_points=state['fft_points'],
        support_radius=state['support_radius'],minimum_atom_image_distance=state['minimum_atom_image_distance'],
        current_basis_rows=state['current_basis_rows'],**body_contract,**controls)
    # Concatenation must finish before the six raw operands are released.
    # The subsequent OWN3 solve coexists with six paired clouds, not eighteen.
    for provider in providers:
        provider['rhs'].block_until_ready()
    for z,provider in zip(channels,providers):
        provider['identity']=state['identity']
        z.local_augmentation=provider
    del state['rhs'],state['smooth_rhs']
    return zetas
