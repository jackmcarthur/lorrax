"""Exact source-positive W to signed first-order W for verification only.

This is not a refreshed-screening constructor and is not the production
``joint_frozen`` initialization policy. No response fit or pole repair occurs.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def _signed_reference_faces(faces, counts, minus, *, mesh_xy, width):
    """Expand ``L R†/(z²-Ω²)`` with the ordered -q transpose.

    ``faces=(L_X,R_Y,Omega²)`` has shapes ``[q,mu_p,1,Kp]`` (complex
    Ry**(3/2)) and ``[q,Kp]`` (real Ry²). This algebra deliberately accepts
    independent physical L/R to expose the negative-sector factor SWAP.
    Returns native writer X faces, complex Ry poles and active-prefix counts.
    All factor reshards stay distributed; only small q/pole metadata replicate.
    The caller admits the complete conversion workspace before invoking it.
    """
    left, right, poles2 = faces
    counts, minus = np.asarray(counts, np.int64), np.asarray(minus, np.int32)
    signed_counts = counts+counts[minus]
    if width < int(signed_counts.max(initial=0)) or left.shape != right.shape:
        raise ValueError("GATE source_frozen_conversion: invalid factor/count extent")
    if poles2.shape != (len(counts), left.shape[-1]):
        raise ValueError("GATE source_frozen_conversion: pole/factor shapes differ")
    out = NamedSharding(mesh_xy, P(None, "x", None, "y"))
    rep = NamedSharding(mesh_xy, P())

    @jax.jit(out_shardings=(out, out, rep, rep))
    def expand(l, r, lam):
        column = jnp.arange(width)[None]
        negative = column >= jnp.asarray(counts)[:, None]
        index = jnp.where(negative, column-jnp.asarray(counts)[:, None], column)
        index = jnp.clip(index, 0, l.shape[-1]-1)
        take = lambda a: jnp.take_along_axis(a, index[:, None, None], axis=-1)
        lp, rp = take(l), take(r)
        # -(L_-q R_-q†)^T = -conj(R_-q) conj(L_-q)†.
        ln, rn = -jnp.conj(take(r[minus])), jnp.conj(take(l[minus]))
        pos_lam = jnp.take_along_axis(lam, index, axis=-1)
        neg_lam = jnp.take_along_axis(lam[minus], index, axis=-1)
        omega = jnp.sqrt(jnp.where(negative, neg_lam, pos_lam))
        active = column < jnp.asarray(signed_counts)[:, None]
        scale = jnp.where(active, 1/jnp.sqrt(2*omega), 0.)
        a = jnp.where(negative[:, None, None], ln, lp)*scale[:, None, None]
        b = jnp.where(negative[:, None, None], rn, rp)*scale[:, None, None]
        poles = jnp.where(active, jnp.where(negative, -omega, omega)+0j, 0j)
        return a, b, poles, jnp.asarray(signed_counts)

    return expand(left, right, poles2)


def source_frozen_ordered_model(model, meta, *, mesh_xy, tables, path):
    """Persist an exact SOURCE-FROZEN positive-to-signed model adapter.

    Parameters
    ----------
    model : dict
        Authenticated source positive shared-pole handle; original stays untouched.
    meta, mesh_xy, tables : native bundles
        Same packed mu basis/full-q tables and a fresh capacity ledger.
    path : str
        New absent general-model filename. Existing output is never overwritten.

    Returns
    -------
    dict
        Standard ``ordered_complex`` model handle plus explicit source provenance.
        Logical active columns are compacted BEFORE mesh padding. No mu-sized
        device array is gathered to a host, fitted, clipped, or spatially reduced.
    """
    from common.collectives import gather_to_host, rank0_transaction
    from file_io.shared_pole_store import (
        face_width, open_shared_pole_model, read_shared_pole_faces,
        validate_shared_pole_model, write_general_shared_pole_model,
    )
    from gw.mpa.sigma import _shared_pole_panel_realizer
    from symmetry_maps import q_negation_index

    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    header = validate_shared_pole_model(model["path"], expected_identity=model["identity"],
        mesh_xy=mesh_xy, capacity=ledger)
    if header["digest"] != model["digest"]:
        raise ValueError("GATE source_frozen_identity: source handle/store digest differ")
    nq = int(header["n_q_full"])
    if (header["representation"] != "scalar-ordered-ph" or header["n_q_irr"] != nq
            or not np.array_equal(header["q_irr_full_idx"], np.arange(nq))
            or not np.array_equal(header["qirr"]["irr_idx_q"], np.arange(nq))):
        raise ValueError("GATE source_frozen_grid: explicit ordered full-q source required")
    # A latent store with a nontrivial spatial realization cannot just be
    # relabeled as a general full-q model. This initial adapter refuses it.
    realizer = _shared_pole_panel_realizer(meta, header, (0, nq), mesh_xy=mesh_xy)
    if not realizer.is_identity:
        raise ValueError("GATE source_frozen_realization: source realization is not identity")
    target = Path(path)
    def prepare_output():
        if target.exists():
            raise FileExistsError(str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
    def check_output():
        if target.exists():
            raise FileExistsError(str(target))
    rank0_transaction(target, stage="source_frozen/absent", write=check_output)
    minus = np.asarray(q_negation_index(tuple(header["grid"])), np.int32)
    counts = np.asarray(header["K"], np.int64)
    signed_counts = counts+counts[minus]
    kmax = int(signed_counts.max(initial=0))
    if not kmax:
        raise ValueError("GATE source_frozen_empty: no physical source poles")
    width = face_width(mesh_xy, kmax)
    source_width = face_width(mesh_xy, int(header["Kmax"]))
    nrank, m = int(mesh_xy.size), int(meta.mu_basis.n_packed)
    source_bytes = 32*nq*m*source_width//nrank
    signed_bytes = 32*nq*m*width//nrank
    # Conservative simultaneous inputs/output/gather/reshard/write envelopes.
    stage = "source_frozen.exact_conversion"
    price = ledger.reserve(stage, resident_bytes_per_rank=source_bytes+signed_bytes,
        workspace_bytes_per_rank=2*signed_bytes+64*nq*width,
        concurrent_with=ambient)
    ledger.live_stages = (*ambient, stage)
    try:
        rank0_transaction(target, stage="source_frozen/output", write=prepare_output)
        with open_shared_pole_model(model["path"], mesh_xy=mesh_xy) as io:
            lx, ry, poles2, _ = read_shared_pole_faces(io, (0, nq), meta=meta, header=header)
        pole_host = np.asarray(gather_to_host(poles2))
        active = np.arange(source_width)[None] < counts[:, None]
        if not np.all(np.isfinite(pole_host)) or not np.all(pole_host[active] > 0):
            raise ValueError("GATE source_frozen_poles: real strictly-positive source poles required")
        left, right, poles, _ = _signed_reference_faces((lx, ry, poles2), counts, minus,
            mesh_xy=mesh_xy, width=width)
        identity = dict(model["identity"], source_frozen_adapter="signed-first-order-v1",
                        source_positive_digest=model["digest"])
        receipt = dict(identity=identity, control="SOURCE-FROZEN EQUILIBRIUM W",
            source_path=str(model["path"]), source_digest=model["digest"],
            source_representation=header["representation"], positive_K=counts.tolist(),
            signed_K=signed_counts.tolist(), capacity=price,
            normalization="p+=Omega(q),R+=B(q)/(2Omega(q)); p-=-Omega(-q),R-=-B(-q)^T/(2Omega(-q))",
            source_realization="authenticated identity", refreshed_screening=False)
        final = write_general_shared_pole_model(str(target), left, right, poles, signed_counts,
            q_span=(0, nq), meta=meta, tables=tables, recipe=dict(header["recipe"],
                source_frozen_adapter="signed-first-order-v1"), receipts=receipt, kmax=kmax)
        jax.block_until_ready((left, right, poles))
        return dict(path=str(target), identity=identity, digest=final["digest"],
            K=final["K"], representation="ordered_complex", source_frozen_receipt=receipt)
    finally:
        ledger.live_stages = ambient


