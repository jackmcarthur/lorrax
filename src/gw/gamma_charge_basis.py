"""Γ real charge endpoints from the authenticated complex-G sphere.

Only geometry and all-P map construction live here.  Density and operator
transforms use ``contour_reference``'s existing endpoint owners.  A real
charge basis grants no time-reversal permission and does not make magnetic
screening symmetric, real, or even in frequency.
"""
from functools import lru_cache
import hashlib
import json

import numpy as np

__all__ = ["gamma_charge_basis_metadata", "gamma_charge_basis"]


def gamma_charge_basis_metadata(sphere, *, row=0):
    """Bind a Γ ``SphereSet`` row to O(M) cosine/sine map metadata.

    Parameters
    ----------
    sphere : SphereSet
        Canonical integer G order, physical ``ngk`` and fractional q.  Pad
        slots are excluded.  The selected q must be exactly zero.
    row : int
        Explicit sphere row; integer-image or finite-q reinterpretation is
        not performed.

    Returns
    -------
    metadata : numpy.ndarray
        Int32 ``[M,3]`` representative/partner/kind rows.  Kinds are virtual,
        identity G0, cosine and positive sine.  This small table may be
        replicated; no dense host map is built.
    receipt : dict
        JSON geometry/basis convention, exact G-negation and metadata hash.
        It is a geometry receipt, not a numerical unitarity certificate.
    """
    from gw.mixed_basis_pair_convolution import SphereSet
    from gw.plane_wave_lehmann import _g_negation

    if not isinstance(sphere, SphereSet):
        raise TypeError("Γ charge endpoints require the canonical SphereSet")
    if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
        raise ValueError("Γ charge sphere row must be an integer")
    row = int(row)
    if not 0 <= row < sphere.n or not np.array_equal(sphere.frac[row], np.zeros(3)):
        raise ValueError("Γ charge endpoints require an explicit zero-q sphere row")
    count, carrier = int(sphere.ngk[row]), int(sphere.width)
    g = np.asarray(sphere.gvecs[row, :count])
    limit = np.iinfo(np.int32).max
    if np.any(g < -limit) or np.any(g > limit):
        raise ValueError("Γ charge integer G coordinates exceed the canonical index range")
    negation = _g_negation(g, carrier)
    zero = np.flatnonzero(np.all(g == 0, axis=1))
    if len(zero) != 1:
        raise ValueError("Γ charge endpoints require one physical G0")
    metadata = np.zeros((carrier, 3), np.int32)
    metadata[int(zero[0])] = (int(zero[0]), int(zero[0]), 1)
    for i, vector in enumerate(g):
        j = int(negation[i])
        if tuple(vector) > tuple(-vector):
            metadata[i] = (i, j, 2)
            metadata[j] = (i, j, 3)
    receipt = dict(
        schema="gamma-real-charge-basis-v1", q_fractional=[0., 0., 0.],
        source_row=row, physical_g_count=count, carrier=carrier,
        retained_g=g.tolist(), g_negation=negation.tolist(),
        row_metadata=metadata.tolist(),
        convention="phi_G=exp(+iG.r)/sqrtOmega; cos=(phi_G+phi_-G)/sqrt2; "
                   "sin=(-i*phi_G+i*phi_-G)/sqrt2; signed representative lex(G)>lex(-G)",
        endpoint_convention="rho_R=rho_G Udagger; W_R=conj(U) W_G U.T; "
                            "W_G=U.T W_R conj(U); no normalization inserted",
        ordered_partner="inverse(SAME-z W_R.T)=P_Gneg W_G.T P_Gneg; not causal conjugation",
        scope="Γ geometry only; zero virtual rows/columns, G0 identity; no TR, "
              "symmetric/even-frequency, head, fit or physical-model admission")
    receipt["signature"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return metadata, receipt


@lru_cache(maxsize=32)
def _map_program(mesh, carrier):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import transpose_xy
    from common.shard_map import shard_map

    nx, ny = int(mesh.shape["x"]), int(mesh.shape["y"])
    local_rows, local_cols = carrier // nx, carrier // ny
    face = NamedSharding(mesh, P(None, "x", "y"))
    replica = NamedSharding(mesh, P())

    def tile(metadata):
        rows = jax.lax.axis_index("x") * local_rows + jnp.arange(local_rows)
        cols = jax.lax.axis_index("y") * local_cols + jnp.arange(local_cols)
        first, second, kind = metadata[rows].T
        half = jnp.asarray(1. / np.sqrt(2.), jnp.complex128)
        left = jnp.where(kind == 1, 1., jnp.where(kind == 2, half,
                         jnp.where(kind == 3, -1j * half, 0.)))
        right = jnp.where(kind == 2, half, jnp.where(kind == 3, 1j * half, 0.))
        value = ((cols[None] == first[:, None]) * left[:, None]
                 + (cols[None] == second[:, None]) * right[:, None])
        return value[None]

    mapped = shard_map(tile, mesh=mesh, in_specs=P(),
                       out_specs=P(None, "x", "y"))

    def maps(metadata):
        value = mapped(metadata)
        return value, transpose_xy(value.conj(), mesh)

    return jax.jit(maps, in_shardings=replica, out_shardings=(face, face))


def gamma_charge_basis(sphere, *, mesh, row=0):
    """Materialize U and U† only on their all-P matrix tiles.

    All world ranks call with identical typed sphere metadata.  A small
    geometry-hash agreement precedes metadata replication.  Returned maps
    are complex128 ``[1,M,M]`` at ``P(None,'x','y')`` on a square mesh.
    Programs are cached by mesh/carrier; geometry remains a runtime input.

    Use ``project_density_endpoints(rho_G,Udagger)`` and
    ``lift_interaction_endpoints(W_G,U,prefactor=1)`` in the existing source
    owner, independently for both actual density directions.  U† restores
    operators through the same lift.  This function transforms no field,
    inserts no physical prefactor and certifies no pole-model accuracy.
    """
    import jax
    from common.collectives import all_gather_processes, replicate_to_mesh

    metadata, receipt = gamma_charge_basis_metadata(sphere, row=row)
    if (tuple(mesh.axis_names) != ("x", "y") or len(mesh.shape) != 2
            or int(mesh.shape["x"]) != int(mesh.shape["y"])
            or receipt["carrier"] % int(mesh.shape["x"])):
        raise ValueError("Γ charge maps require a square x/y mesh and divisible carrier")
    devices = tuple(mesh.devices.flat)
    world_devices = tuple(jax.devices())
    if len(devices) != len(world_devices) or set(devices) != set(world_devices):
        raise ValueError("Γ charge map mesh must cover every global JAX device exactly once")
    if not jax.config.x64_enabled:
        raise ValueError("Γ charge maps require the canonical complex128 runtime")
    signature = np.frombuffer(bytes.fromhex(receipt["signature"]), np.uint8).copy()
    copies = np.asarray(all_gather_processes(signature)).reshape(-1, len(signature))
    if not np.all(copies == copies[0]):
        raise ValueError("Γ charge metadata differs between world ranks")
    value, adjoint = _map_program(mesh, receipt["carrier"])(
        replicate_to_mesh(metadata, mesh))
    receipt = dict(receipt, map_spec="PartitionSpec(None, 'x', 'y')",
                   global_map_bytes=receipt["carrier"] ** 2 * 16,
                   replicated_metadata_bytes=int(metadata.nbytes),
                   global_devices=int(mesh.devices.size))
    return value, adjoint, receipt
