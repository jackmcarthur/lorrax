"""Finite-plane-wave Γ Lehmann response from bounded native transition tiles.

This reference owner composes canonical ortho wavefunction transforms,
backward density FFTs, the shared ordered Lehmann weights, and distrib_la's
bounded weighted face contraction. It is a step-occupied Γ reference, not a
metal or general-q adapter. The caller owns archive authentication, native
coverage, typed parent-to-child unfolding and the retained-G alias proof.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from .lehmann_response import lehmann_pair_weights
from .plane_wave_screening import chi_lehmann_sum_scale

__all__ = ["gamma_transition_vertices", "GammaLehmannResponse"]


def _require_spec(value, mesh, spec, label):
    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, NamedSharding) or sharding.mesh != mesh or sharding.spec != spec:
        raise ValueError(f"{label} requires {spec}, got {sharding!r}")


@lru_cache(maxsize=None)
def _transition_kernel(mesh, grid, n_occupied):
    from common.fft_helpers import make_sharded_fftn_3d
    from common.wfn_transforms import _sphere_gather
    box_spec = P(None, ("x", "y"), None, None, None, None)
    fft = make_sharded_fftn_3d(mesh, box_spec, box_spec, norm="backward")

    def value(occupied, empty, sphere_index):
        # ψ(r) carries the canonical ortho normalization. Its density's
        # backward FFT is the conventional coefficient convolution M(G).
        density = jnp.sum(occupied[:, None].conj() * empty[:, :, None], axis=3)
        flat = density.reshape((1, empty.shape[1] * n_occupied, 1) + grid)
        modes = _sphere_gather(fft(flat), sphere_index)
        return modes.reshape(empty.shape[1], n_occupied, modes.shape[-1])

    return jax.jit(value, out_shardings=NamedSharding(mesh, P(("x", "y"), None, None)))


def gamma_transition_vertices(occupied_rbox, empty_rbox, sphere_index, *, mesh,
                               occupied_replication_bound_bytes):
    """One full-k native tile: M_ab(G) with a occupied and b empty.

    Occupied factors are an explicitly bounded replicated one-k tile;
    empty factors and the returned ``[b,a,G]`` remain band-sharded over all
    P ranks. Inputs are ordinary full-Bloch ψ(r) from the canonical owner
    at the SAME loader-paired k. At Γ their Bloch phases cancel; no phase
    removal, time-reversal identity or independent k table is introduced.
    """
    a, b = occupied_rbox, empty_rbox
    if (a.ndim != 6 or b.ndim != 6 or a.shape[0] != 1 or b.shape[0] != 1
            or a.shape[2:] != b.shape[2:] or a.shape[1] < 1):
        raise ValueError("Γ transition vertices require compatible one-k occupied/empty rboxes")
    _require_spec(a, mesh, P(None, None, None, None, None, None), "bounded occupied tile")
    _require_spec(b, mesh, P(None, ("x", "y"), None, None, None, None), "empty tile")
    if (int(occupied_replication_bound_bytes) < 1
            or a.size * a.dtype.itemsize > int(occupied_replication_bound_bytes)):
        raise ValueError("Γ occupied tile exceeds its explicitly admitted replication bound")
    if sphere_index.ndim != 2 or sphere_index.shape[0] != 1:
        raise ValueError("Γ transition vertices require the one-row retained-G sphere index")
    return _transition_kernel(mesh, tuple(int(v) for v in a.shape[3:]), int(a.shape[1]))(
        a, b, jnp.asarray(sphere_index, dtype=jnp.int32))


def _g_negation(gvecs, carrier):
    g = np.asarray(gvecs)
    if (g.ndim != 2 or g.shape[1] != 3 or g.dtype.kind not in "iu"
            or len(g) < 1 or carrier < len(g) or len(set(map(tuple, g))) != len(g)):
        raise ValueError("Γ response needs unique integer retained G vectors")
    lookup = {tuple(v): i for i, v in enumerate(g)}
    try:
        permutation = np.asarray([lookup[tuple(-v)] for v in g], np.int32)
    except KeyError as exc:
        raise ValueError("Γ response retained-G sphere must close exactly under negation") from exc
    if (not np.array_equal(g[permutation], -g)
            or not np.array_equal(permutation[permutation], np.arange(len(g)))):
        raise ValueError("Γ response G-negation failed its exact involution proof")
    return np.concatenate((permutation, np.arange(len(g), carrier, dtype=np.int32)))


@lru_cache(maxsize=None)
def _response_kernel(mesh, panel_bytes, with_derivative):
    from distrib_la import panel_matmul
    from common.staged_reshard import permute_sharded_axis
    face = NamedSharding(mesh, P(None, "x", "y"))

    def value(left, right, de, df, z, gneg):
        forward = lehmann_pair_weights(de, df, z[None], with_derivative=with_derivative)
        reverse = lehmann_pair_weights(-de, -df, z[None], with_derivative=with_derivative)
        outputs = []
        for i in range(2 if with_derivative else 1):
            ordinary = panel_matmul(left, right, mesh=mesh, panel_bytes=panel_bytes,
                                    weights=forward[i][None])
            # M_ba(G)=conj M_ab(-G). Conjugate the small output, then
            # permute both endpoints. No reverse or weighted full-M copy
            # is retained. Only the replicated weights are conjugated.
            partner = panel_matmul(left, right, mesh=mesh, panel_bytes=panel_bytes,
                                   weights=reverse[i].conj()[None]).conj()
            partner = permute_sharded_axis(partner, 1, gneg, mesh, P(None, "x", "y"))
            partner = permute_sharded_axis(partner, 2, gneg, mesh, P(None, "x", "y"))
            outputs.append(ordinary + partner)
        return tuple(outputs) if with_derivative else outputs[0]

    return jax.jit(value, out_shardings=(face, face) if with_derivative else face)


class GammaLehmannResponse:
    """All-native, step-occupied Γ pair bank, held on two all-P faces.

    ``vertices[b,k,a,G]`` is band-major, with padded empty-band axis on XY;
    occupied energies have shape ``[k,a]``, empty energies/validity ``[k,b]``.
    Only complete physical occupied and empty spectra are admissible. The
    caller's coverage receipt authenticates that precondition. Invalid and
    retained-G pad coefficients must be EXACT zero; both direction weights
    vanish on invalid bands. The Γ endpoint reversal uses an exact G-negation
    involution, including interleaved complex densities without TR symmetry.
    """

    def __init__(self, vertices, energies_occupied, energies_empty, valid_empty, *,
                 gvecs, mesh, cell_volume, n_spin=1, n_spinor=1,
                 panel_bytes=64 << 20):
        _require_spec(vertices, mesh, P(("x", "y"), None, None, None), "Γ transition bank")
        if vertices.ndim != 4 or vertices.dtype != jnp.complex128:
            raise ValueError("Γ transition bank needs complex128 [b,k,a,G] axes")
        B, K, A, M = (int(v) for v in vertices.shape)
        ea, eb, valid = np.asarray(energies_occupied), np.asarray(energies_empty), np.asarray(valid_empty)
        if (ea.shape != (K, A) or eb.shape != (K, B) or valid.shape != (K, B)
                or valid.dtype != np.bool_ or not np.all(valid.any(axis=1))
                or not np.all(np.isfinite(ea)) or not np.all(np.isfinite(eb[valid]))
                or int(panel_bytes) < 1):
            raise ValueError("Γ response requires finite physical energies and explicit boolean native validity")
        if not float(np.min(eb[valid])) > float(np.max(ea)):
            raise ValueError("Γ step response requires a positive occupied-empty gap")
        gneg = _g_negation(gvecs, M)
        self.mesh, self.panel_bytes = mesh, int(panel_bytes)
        self.scale = chi_lehmann_sum_scale(cell_volume=cell_volume, n_k=K,
                                          n_spin=n_spin, n_spinor=n_spinor)
        rep = NamedSharding(mesh, P())
        mask = valid.T[:, :, None] & np.ones((1, 1, A), bool)
        de = np.where(mask, ea[None] - eb.T[:, :, None], 0.0).reshape(-1)
        df = mask.astype(np.float64).reshape(-1)
        from common.collectives import replicate_to_mesh
        self.de, self.df = replicate_to_mesh(de, mesh), replicate_to_mesh(df, mesh)
        self.gneg = replicate_to_mesh(gneg, mesh)
        zero = jax.jit(lambda v: jnp.max(jnp.abs(jnp.where(
            (~mask)[..., None] | (jnp.arange(M) >= len(gvecs))[None, None, None], v, 0))),
            out_shardings=rep)(vertices)
        if float(np.asarray(jax.device_get(zero))) != 0.0:
            raise ValueError("Γ transition bank has a nonzero native-band or retained-G ghost")
        face = NamedSharding(mesh, P(None, "x", "y"))
        T = B * K * A
        from common.staged_reshard import band_to_product_r_reshard
        from common.collectives import transpose_xy
        flat = jax.jit(lambda v: v.reshape(1, T, 1, M), out_shardings=NamedSharding(
            mesh, P(None, ("x", "y"), None, None)))(vertices)
        ordinary = band_to_product_r_reshard(mesh, face=True)(flat)
        right = jax.jit(lambda v: v.reshape(1, T, M), out_shardings=face)(ordinary)
        self.left = jax.jit(lambda v: transpose_xy(v, mesh), out_shardings=face)(right)
        self.right = jax.jit(lambda v: v.conj(), out_shardings=face)(right)
        del flat, ordinary, right
        self.left.block_until_ready(); self.right.block_until_ready()
        self.receipt = dict(scope="gamma_step_occupied", shape=[B, K, A, M],
            physical_empty_by_k=valid.sum(axis=1).tolist(), physical_occupied_per_k=A,
            physical_pair_count=int(valid.sum()) * A, transition_carrier=T,
            scale=self.scale, normalization="M_raw_coefficient_convolution; spin/(Omega*Nk)",
            reverse_rule="M_ba(G)=conj(M_ab(-G))", g_negation=gneg.tolist(),
            pair_faces_spec=[None, "x", "y"],
            resident_pair_faces_bytes_global=2 * T * M * vertices.dtype.itemsize,
            replicated_energy_weight_metadata_bytes=de.nbytes + df.nbytes + gneg.nbytes,
            panel_bytes_per_rank=self.panel_bytes, raw_vertices_retained=False)

    def evaluate(self, z_values, *, with_derivative=False):
        """Physical χ and optionally dχ/ds at finite upper-half-plane sites.

        The fixed two face banks are reused one frequency at a time; neither
        an NF-weighted transition copy nor an NF-broadcast operand exists.
        Returned small operators have leading z and one Γ q row.
        """
        z = np.asarray(z_values, np.complex128)
        if (z.ndim != 1 or not len(z) or not np.all(np.isfinite(z))
                or np.any(z.imag <= 0)):
            raise ValueError("Γ Lehmann sites must be finite, nonempty, upper-half-plane values")
        kernel = _response_kernel(self.mesh, self.panel_bytes, bool(with_derivative))
        values, derivatives = [], []
        for site in z:
            result = kernel(self.left, self.right, self.de, self.df,
                            jnp.asarray(site), self.gneg)
            if with_derivative:
                value, derivative = result
                derivatives.append(derivative * self.scale)
            else:
                value = result
            values.append(value * self.scale)
        face = NamedSharding(self.mesh, P(None, None, "x", "y"))
        stack = jax.jit(lambda *a: jnp.stack(a), out_shardings=face)
        answer = stack(*values)
        return (answer, stack(*derivatives)) if with_derivative else answer
