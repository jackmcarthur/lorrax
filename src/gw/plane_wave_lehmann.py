"""Finite-plane-wave Lehmann response from bounded native transition tiles.

This reference owner composes canonical ortho wavefunction transforms,
backward density FFTs, the shared ordered Lehmann weights, and distrib_la's
bounded weighted face contraction. Γ step banks and explicit ordered tiles
share those owners. The caller owns archive authentication, native
coverage, typed parent-to-child unfolding and the retained-G alias proof.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from .lehmann_response import lehmann_pair_weights
from .plane_wave_screening import chi_lehmann_sum_scale

__all__ = ["gamma_transition_vertices", "transition_vertices",
           "GammaLehmannResponse", "OrderedLehmannPair"]


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


@lru_cache(maxsize=None)
def _q_transition_kernel(mesh, grid, n_left, conjugate_density):
    from common.fft_helpers import make_sharded_fftn_3d
    from common.wfn_transforms import _sphere_gather, apply_bloch_phase
    box_spec = P(None, ("x", "y"), None, None, None, None)
    fft = make_sharded_fftn_3d(mesh, box_spec, box_spec, norm="backward")

    def value(left, right, sphere_index, q):
        density = jnp.sum(left[:, None].conj() * right[:, :, None], axis=3)
        if conjugate_density:
            density = density.conj()
        flat = density.reshape((1, right.shape[1] * n_left, 1) + grid)
        flat = apply_bloch_phase(flat, q[None], grid, sign=-1)
        modes = _sphere_gather(fft(flat), sphere_index)
        return modes.reshape(right.shape[1], n_left, modes.shape[-1])

    return jax.jit(value, out_shardings=NamedSharding(mesh, P(("x", "y"), None, None)))


def transition_vertices(left_rbox, right_rbox, sphere_index, *, mesh,
                        q_frac, left_k_frac, right_k_frac,
                        left_replication_bound_bytes, conjugate_density=False):
    """One directed finite-q tile, ``FFT[e^-iqr sum_s conj(ψa) ψb]``.

    ``left`` is an explicitly bounded replicated one-k role tile; ``right``
    and the returned ``[b,a,G]`` remain band-sharded over all P. Ordinary
    full-Bloch fields and their EXACT loader-paired k representatives are
    required. Their momentum difference must equal q modulo an integer G.
    That integer wrap remains in the density FFT; it is never discarded.

    ``conjugate_density=True`` constructs the opposite ordered density
    from the same small-left/large-right schedule, before removal of its
    declared q phase. It does not infer a time-reversal or minus-q response.
    """
    a, b = left_rbox, right_rbox
    if (a.ndim != 6 or b.ndim != 6 or a.shape[0] != 1 or b.shape[0] != 1
            or a.shape[2:] != b.shape[2:] or min(a.shape[1], b.shape[1]) < 1):
        raise ValueError("directed transition vertices require compatible one-k rboxes")
    _require_spec(a, mesh, P(None, None, None, None, None, None), "bounded left tile")
    _require_spec(b, mesh, P(None, ("x", "y"), None, None, None, None), "right tile")
    if (a.dtype != jnp.complex128 or b.dtype != jnp.complex128
            or int(left_replication_bound_bytes) < 1
            or a.size * a.dtype.itemsize > int(left_replication_bound_bytes)):
        raise ValueError("directed left tile requires complex128 within its explicit replication bound")
    if sphere_index.ndim != 2 or sphere_index.shape[0] != 1:
        raise ValueError("directed transition vertices require one retained-G sphere row")
    q, ka, kb = (np.asarray(v, np.float64) for v in (q_frac, left_k_frac, right_k_frac))
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (q, ka, kb)):
        raise ValueError("directed transitions require finite three-component paired k/q representatives")
    difference = (ka - kb if conjugate_density else kb - ka) - q
    if np.max(np.abs(difference - np.rint(difference))) > 1e-12:
        raise ValueError("directed transition momentum does not close by an integer reciprocal wrap")
    if not isinstance(conjugate_density, (bool, np.bool_)):
        raise ValueError("conjugate_density must be an explicit boolean ordered-density choice")
    return _q_transition_kernel(mesh, tuple(map(int, a.shape[3:])), int(a.shape[1]),
                                bool(conjugate_density))(
        a, b, jnp.asarray(sphere_index, dtype=jnp.int32), jnp.asarray(q))


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


@lru_cache(maxsize=None)
def _ordered_response_kernel(mesh, panel_bytes, with_derivative):
    from distrib_la import panel_matmul
    face = NamedSharding(mesh, P(None, "x", "y"))

    def value(left, right, de, df, z):
        weights = lehmann_pair_weights(de, df, z[None], with_derivative=with_derivative)
        outputs = tuple(panel_matmul(left, right, mesh=mesh, panel_bytes=panel_bytes,
                                    weights=weights[i][None])
                        for i in range(2 if with_derivative else 1))
        return outputs if with_derivative else outputs[0]

    return jax.jit(value, out_shardings=(face, face) if with_derivative else face)


@lru_cache(maxsize=None)
def _transition_face_programs(mesh):
    """Reuse the same transport callables across bounded native tiles."""
    from common.collectives import transpose_xy
    face = NamedSharding(mesh, P(None, "x", "y"))
    def flatten(v):
        B, K, A, M = map(int, v.shape)
        return v.reshape(1, B * K * A, 1, M)
    return (jax.jit(flatten, out_shardings=NamedSharding(
                mesh, P(None, ("x", "y"), None, None))),
            jax.jit(lambda v: v.reshape(1, v.shape[1], v.shape[-1]), out_shardings=face),
            jax.jit(lambda v: transpose_xy(v, mesh), out_shardings=face),
            jax.jit(lambda v: v.conj(), out_shardings=face))


def _transition_faces(vertices, mesh):
    """The shared volume-preserving raw-band to two all-P face conversion."""
    from common.staged_reshard import band_to_product_r_reshard
    flatten, reshape, transpose, conjugate = _transition_face_programs(mesh)
    flat = flatten(vertices)
    ordinary = band_to_product_r_reshard(mesh, face=True)(flat)
    right = reshape(ordinary)
    left = transpose(right)
    right = conjugate(right)
    left.block_until_ready(); right.block_until_ready()
    return left, right


@lru_cache(maxsize=None)
def _ordered_vertex_guard(mesh, physical_g_count):
    """Exact ghost/finite guard with a dynamic replicated pair mask."""
    def guard(v, valid):
        ghost_mask = (~valid)[..., None] | (
            jnp.arange(v.shape[-1]) >= physical_g_count)[None, None, None]
        return jnp.stack((jnp.max(jnp.abs(jnp.where(ghost_mask, v, 0))),
                          jnp.all(jnp.isfinite(v)).astype(jnp.float64)))
    return jax.jit(guard, in_shardings=(NamedSharding(mesh,
        P(("x", "y"), None, None, None)), NamedSharding(mesh, P())),
        out_shardings=NamedSharding(mesh, P()))


class OrderedLehmannPair:
    """One explicit directed native-pair tile on two all-P density faces.

    ``vertices[b,k,a,G]`` and replicated ``de``, ``df``, ``pair_valid``
    have the same [b,k,a] order. The caller supplies de=Ea-Eb and df=fa-fb
    for the physical ordered density represented by its vertices. No gap,
    occupation threshold, k partner, reverse q or complete-spectrum claim
    is inferred. Bounded metallic streams can include every finite FD tail.

    ``n_k`` is the full physical normalization census (defaults to the tile's
    k extent); a one-k streaming tile must declare the full-zone count.
    Retained-G/native ghosts must be exact zero, including df-zero physical
    pairs. This owner never accepts a nonzero ghost just because its weight
    would vanish. The original Γ bank uses the same face conversion.
    """

    def __init__(self, vertices, energy_difference, occupation_difference, pair_valid,
                 *, mesh, cell_volume, n_k=None, n_spin=1, n_spinor=1,
                 physical_g_count=None, panel_bytes=64 << 20):
        _require_spec(vertices, mesh, P(("x", "y"), None, None, None), "ordered transition tile")
        if vertices.ndim != 4 or vertices.dtype != jnp.complex128 or min(vertices.shape) < 1:
            raise ValueError("ordered transition tile needs complex128 [b,k,a,G] axes")
        B, K, A, M = map(int, vertices.shape)
        de, df, valid = (np.asarray(v) for v in (energy_difference, occupation_difference, pair_valid))
        if (de.shape != (B, K, A) or df.shape != de.shape or valid.shape != de.shape
                or valid.dtype != np.bool_ or de.dtype.kind != "f" or df.dtype.kind != "f"
                or not np.all(np.isfinite(de[valid])) or not np.all(np.isfinite(df[valid]))
                or np.any(np.abs(df[valid]) > 1.)):
            raise ValueError("ordered transitions need finite physical de/df and explicit boolean pair validity")
        Nk = K if n_k is None else n_k
        Ng = M if physical_g_count is None else physical_g_count
        if (isinstance(Nk, (bool, np.bool_)) or not isinstance(Nk, (int, np.integer))
                or Nk < K or isinstance(Ng, (bool, np.bool_))
                or not isinstance(Ng, (int, np.integer)) or not 1 <= Ng <= M
                or isinstance(panel_bytes, (bool, np.bool_)) or int(panel_bytes) < 1):
            raise ValueError("ordered transitions require explicit positive physical k/G extents and panel bound")
        self.mesh, self.panel_bytes = mesh, int(panel_bytes)
        self.scale = chi_lehmann_sum_scale(cell_volume=cell_volume, n_k=int(Nk),
                                          n_spin=n_spin, n_spinor=n_spinor)
        from common.collectives import replicate_to_mesh
        self.de = replicate_to_mesh(np.where(valid, de, 0.).astype(np.float64).reshape(-1), mesh)
        self.df = replicate_to_mesh(np.where(valid, df, 0.).astype(np.float64).reshape(-1), mesh)
        ghost, finite = np.asarray(jax.device_get(_ordered_vertex_guard(mesh, int(Ng))(
            vertices, replicate_to_mesh(valid, mesh))))
        if float(ghost) != 0.:
            raise ValueError("ordered transition tile has a nonzero native-pair or retained-G ghost")
        if not bool(finite):
            raise ValueError("ordered transition tile has a nonfinite density coefficient")
        self.left, self.right = _transition_faces(vertices, mesh)
        self.physical_g_count = int(Ng)
        self.receipt = dict(scope="explicit_ordered_pair_tile", shape=[B, K, A, M],
            physical_pair_count=int(valid.sum()), nonzero_df_pair_count=int(np.count_nonzero(df[valid])),
            full_k_normalization=int(Nk), physical_g_count=int(Ng), scale=self.scale,
            normalization="M_raw_coefficient_convolution; spin/(Omega*Nk)",
            pair_faces_spec=[None, "x", "y"], resident_pair_faces_bytes_global=2*B*K*A*M*16,
            panel_bytes_per_rank=self.panel_bytes, derivative_variable="s=z^2")

    @classmethod
    def from_density_face(cls, density, energy_difference, occupation_difference,
                          pair_valid, *, mesh, physical_prefactor, endpoint_valid,
                          normalization, panel_bytes=64 << 20, donate_density=False):
        """Reuse one ordinary all-P density face without a full raw bank.

        ``density[1,T,M]`` is at ``P(None,'x','y')``. Flat de/df and explicit
        boolean validity have T entries, in precisely that density order.
        The endpoint mask has M entries; interleaved centroid pads are not
        reinterpreted as a prefix. The caller supplies a finite positive
        physical prefactor and its normalization label. Raw centroid χ
        uses ``1/centroid_response_denominator(Nk)``; the existing Dyson
        owner supplies its remaining spin/k normalization.

        At an I/O seam the centroid owner must already have packed a
        canonical logical file axis. This method does not read, repack or
        infer a basis. It transposes the ordinary face over all processors,
        waits for that output, then conjugates its original storage.
        ``donate_density=True`` explicitly invalidates the caller's density
        array and keeps only the two final faces, rather than three banks.
        """
        _require_spec(density, mesh, P(None, "x", "y"), "ordinary density face")
        if (density.ndim != 3 or density.shape[0] != 1 or min(density.shape) < 1
                or density.dtype != jnp.complex128):
            raise ValueError("ordinary density face requires complex128 [1,T,M]")
        T, M = map(int, density.shape[1:])
        de, df, valid = (np.asarray(v) for v in (energy_difference, occupation_difference, pair_valid))
        endpoints = np.asarray(endpoint_valid)
        if (de.shape != (T,) or df.shape != (T,) or valid.shape != (T,)
                or valid.dtype != np.bool_ or de.dtype.kind != "f" or df.dtype.kind != "f"
                or not np.all(np.isfinite(de[valid])) or not np.all(np.isfinite(df[valid]))
                or np.any(np.abs(df[valid]) > 1.) or endpoints.shape != (M,)
                or endpoints.dtype != np.bool_ or not np.any(endpoints)):
            raise ValueError("ordinary density face needs finite physical de/df and explicit pair/endpoint validity")
        if (not isinstance(donate_density, (bool, np.bool_)) or not isinstance(normalization, str)
                or not normalization.strip() or isinstance(panel_bytes, (bool, np.bool_))
                or int(panel_bytes) < 1):
            raise ValueError("density face requires an explicit normalization, panel bound and boolean donation choice")
        scale = float(np.asarray(jax.device_get(physical_prefactor)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("density face physical prefactor must be finite and positive")

        def guard(v):
            mask = (~valid)[None, :, None] | (~endpoints)[None, None, :]
            return jnp.stack((jnp.max(jnp.abs(jnp.where(mask, v, 0))),
                              jnp.all(jnp.isfinite(v)).astype(jnp.float64)))
        ghost, finite = np.asarray(jax.device_get(jax.jit(guard,
            out_shardings=NamedSharding(mesh, P()))(density)))
        if ghost != 0.:
            raise ValueError("ordinary density face has a nonzero physical-pair or endpoint ghost")
        if not bool(finite):
            raise ValueError("ordinary density face has a nonfinite coefficient")
        from common.collectives import replicate_to_mesh, transpose_xy
        face = NamedSharding(mesh, P(None, "x", "y"))
        out = object.__new__(cls)
        out.mesh, out.panel_bytes, out.scale = mesh, int(panel_bytes), scale
        out.de = replicate_to_mesh(np.where(valid, de, 0.).astype(np.float64), mesh)
        out.df = replicate_to_mesh(np.where(valid, df, 0.).astype(np.float64), mesh)
        out.left = jax.jit(lambda v: transpose_xy(v, mesh), out_shardings=face)(density)
        out.left.block_until_ready()
        conjugate = jax.jit(lambda v: v.conj(), out_shardings=face,
                            donate_argnums=(0,) if donate_density else ())
        out.right = conjugate(density)
        out.right.block_until_ready()
        out.physical_g_count = int(endpoints.sum())
        out.endpoint_valid = endpoints.copy()
        out.receipt = dict(scope="explicit_ordinary_density_face", density_face_shape=[1, T, M],
            physical_pair_count=int(valid.sum()), nonzero_df_pair_count=int(np.count_nonzero(df[valid])),
            physical_endpoint_count=int(endpoints.sum()), scale=scale, normalization=normalization,
            pair_faces_spec=[None, "x", "y"], resident_pair_faces_bytes_global=2*T*M*16,
            raw_density_donated=bool(donate_density), panel_bytes_per_rank=out.panel_bytes,
            derivative_variable="s=z^2")
        return out

    def _evaluate(self, z_values, with_derivative, gneg=None):
        z = np.asarray(z_values, np.complex128)
        if z.ndim != 1 or not len(z) or not np.all(np.isfinite(z)) or np.any(z.imag <= 0):
            raise ValueError("ordered Lehmann sites must be finite, nonempty, upper-half-plane values")
        values, derivatives = [], []
        for site in z:
            args = (self.left, self.right, self.de, self.df, jnp.asarray(site))
            if gneg is None:
                result = _ordered_response_kernel(self.mesh, self.panel_bytes, bool(with_derivative))(*args)
            else:
                result = _response_kernel(self.mesh, self.panel_bytes, bool(with_derivative))(*args, gneg)
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

    def evaluate(self, z_values, *, with_derivative=False):
        """One directed χ contribution and optional exact dχ/ds."""
        return self._evaluate(z_values, with_derivative)

    def evaluate_gamma_pair(self, z_values, *, gvecs, with_derivative=False):
        """Add the Γ reverse transition from the SAME paired-k density.

        The caller must cover each unordered pair exactly once. This is not
        a finite-q transport identity; general q requires both actual directed
        density tiles and uses :meth:`evaluate` for each.
        """
        if len(gvecs) != self.physical_g_count:
            raise ValueError("Γ paired tile G census differs from its physical-G mask")
        if hasattr(self, "endpoint_valid") and not np.array_equal(
                self.endpoint_valid, np.arange(len(self.endpoint_valid)) < self.physical_g_count):
            raise ValueError("Γ G-negation cannot reinterpret an interleaved endpoint mask as a prefix")
        from common.collectives import replicate_to_mesh
        gneg = replicate_to_mesh(_g_negation(gvecs, int(self.left.shape[-2])), self.mesh)
        return self._evaluate(z_values, with_derivative, gneg)

    def evaluate_same_k_pair(self, z_values, *, endpoint_negation=None, with_derivative=False):
        """The two directions of an explicitly same-k unordered pair bank.

        Real-space centroid endpoints use the identity permutation and the
        conjugated density. PW endpoints require their authenticated G-negation
        permutation. This operation supplies no finite-q or TR identity.
        """
        M = int(self.left.shape[-2])
        permutation = np.arange(M, dtype=np.int32) if endpoint_negation is None else np.asarray(endpoint_negation)
        if (permutation.shape != (M,) or permutation.dtype.kind not in "iu"
                or not np.array_equal(np.sort(permutation), np.arange(M))
                or not np.array_equal(permutation[permutation], np.arange(M))):
            raise ValueError("same-k paired endpoints require an exact integer negation involution")
        if hasattr(self, "endpoint_valid") and not np.array_equal(
                self.endpoint_valid[permutation], self.endpoint_valid):
            raise ValueError("same-k negation changes physical endpoint validity")
        from common.collectives import replicate_to_mesh
        return self._evaluate(z_values, with_derivative,
                              replicate_to_mesh(permutation.astype(np.int32), self.mesh))


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
        T = B * K * A
        self.left, self.right = _transition_faces(vertices, mesh)
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
