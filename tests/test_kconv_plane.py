"""The route-G plane door of the k-convolution router (cpu leg) against the chain it replaced.

``ffi.fft.make_fused_conv_kplane(D, F)`` must equal, bit for bit, the old
route-G chain -- the XLA Bloch phase, moveaxis and L/R split of the D-plane
FFT output followed by the parent door on the identity plan -- and the literal
circular k sum.  Red twin: the Bloch phase rolled by one k must miss.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from ffi import fft as F

KG = (3, 2, 2)


def _operands(ns, seed=0):
    rng = np.random.default_rng(seed + ns)
    nk, g, c, p = int(np.prod(KG)), 3, 2, 5
    D = rng.normal(size=(nk, g, ns, 2 * c, ns, p)) + 1j * rng.normal(size=(nk, g, ns, 2 * c, ns, p))
    kfrac = np.stack(np.unravel_index(np.arange(nk), KG), 1) / np.asarray(KG, float)
    x = rng.uniform(size=(g, p))
    Fb = np.exp(-2j * np.pi * kfrac[:, :1, None] * x[None]) / np.sqrt(7.0)    # (nk, g, p)
    return D, Fb, nk, g, c, p


def _identity_tables(nk, ns, c, r, kg=KG):
    """The ten typed tables of the identity plan (every k its own parent)."""
    eye = np.broadcast_to(np.eye(ns * ns, dtype=np.complex128), (nk, ns * ns, ns * ns)).copy()
    return tuple(jnp.asarray(a) for a in (
        np.arange(nk, dtype=np.int32), np.zeros(nk, np.int32),
        np.arange(c, dtype=np.int32)[None], np.arange(r, dtype=np.int32)[None],
        np.zeros((1, c, 3)), np.zeros((1, r, 3)),
        np.stack(np.unravel_index(np.arange(nk), kg), 1) / np.asarray(kg, float),
        np.zeros(nk, np.int32), eye, eye))


def _old_chain(D, Fb, ns, c, perm, phase):
    """Route G before the plane door: phase, moveaxis, split, parent door."""
    nk, g, _, _, _, p = D.shape
    d = jnp.asarray(D) * jnp.asarray(Fb)[:, :, None, None, None, :]
    Dk = jnp.moveaxis(d, 1, 4).reshape(nk, ns, 2 * c, ns, g * p)
    kern = F._plan_kparent(KG, ns, perm, phase, perm, phase,
                           F.conv_kpair_scale("forward", nk))
    return kern(Dk[:, :, :c], Dk[:, :, c:], _identity_tables(nk, ns, c, g * p))


def _literal(D, Fb, ns, c, perm, phase, kg=KG):
    """U_q(m, n) = Σ_k Σ_ab φ_a φ_b conj P^L_{k,ab}(m,n) P^R_{k+q,πa πb}(m,n)."""
    nk, g, _, _, _, p = D.shape
    Pk = np.conj(D * Fb[:, :, None, None, None, :])                    # (k, g, a, 2c, b, p)
    Pk = np.moveaxis(Pk, 1, 4).reshape(nk, ns, 2 * c, ns, g * p)
    L, R = Pk[:, :, :c], Pk[:, :, c:]
    R = R[:, perm][:, :, :, perm] * phase[None, :, None, None, None] * phase[None, None, None, :, None]
    idx = np.stack(np.unravel_index(np.arange(nk), kg), 1)
    out = np.zeros((nk, c, g * p), np.complex128)
    for q in range(nk):
        kq = np.ravel_multi_index(((idx + idx[q]) % np.asarray(kg)).T, kg)
        out[q] = np.einsum("kambn,kambn->mn", L.conj(), R[kq])
    return out


def test_plane_door_matches_old_route_g_chain_bitwise():
    """cpu leg: bitwise equal to the parent-door chain; 1e-12 of the literal sum; red twin fires."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    assert F.kconv_backend(mesh) == "plan"
    for ns in (1, 2, 4):
        D, Fb, nk, g, c, p = _operands(ns)
        perm = np.roll(np.arange(ns), 1) if ns > 1 else np.arange(ns)
        phase = np.asarray([1, 1j, -1, -1j][:ns])
        door = F.make_fused_conv_kplane(mesh, KG, ns, perm_l=perm, phase_l=phase,
                                        perm_r=perm, phase_r=phase)
        U = np.asarray(door(jnp.asarray(D), jnp.asarray(Fb)))
        old = np.asarray(_old_chain(D, Fb, ns, c, perm, phase))
        ref = _literal(D, Fb, ns, c, perm, phase)
        assert U.shape == (nk, c, g * p)
        assert np.array_equal(U, old), (ns, float(np.max(np.abs(U - old))))
        assert np.max(np.abs(U - ref)) <= 1e-12 * np.max(np.abs(ref)), ns
        red = np.asarray(door(jnp.asarray(D), jnp.asarray(np.roll(Fb, 1, axis=0))))
        assert np.max(np.abs(red - ref)) > 1e-3 * np.max(np.abs(ref)), ns


def test_plane_door_refuses_a_bad_layout():
    """An odd slot axis or a phase of the wrong shape is a ValueError at trace time."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    door = F.make_fused_conv_kplane(mesh, KG, 2, perm_l=np.arange(2), phase_l=np.ones(2),
                                    perm_r=np.arange(2), phase_r=np.ones(2))
    D, Fb, nk, g, c, p = _operands(2)
    for bad_d, bad_f in ((D[:, :, :, :3], Fb), (D, Fb[:, :, :-1])):
        try:
            door(jnp.asarray(bad_d), jnp.asarray(bad_f))
        except ValueError as exc:
            assert "k-conv plane" in str(exc)
        else:
            raise AssertionError("bad plane operands were accepted")
