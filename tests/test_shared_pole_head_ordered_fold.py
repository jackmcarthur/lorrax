"""The full Gamma head on an ordered (time-reversal-broken) shared-pole store.

``gw.shared_pole_head`` folds ``S + Y W Z / Omega`` with the store's signed
body and the Lehmann wings of every ordered pair.  Checked here on CPU:

* wings: the face kernel equals a direct numpy Lehmann sum over all ordered
  pairs on a random TRS-broken system (the old energy-ordered even weight
  ``-c f A / (z^2 - dE^2)`` misses it); ``Y(-z) = Z(z)^T``;
* body: the ordered Gamma body equals the store normalization
  ``sum_p b b^H/(2W(z-W)) - conj(b) b^T/(2W(z+W))``, reduces to the TRS
  ``b (z^2 - Lambda)^-1 b^H`` for real ``b``, and obeys ``W(-z) = W(z)^T``;
* four-component carrier (the ``bare_transverse`` restricted-kinetic-balance
  store, capacity counted on the two-component source WFN): the wings are the
  spin-traced Lehmann sum over all four components, are invariant under a
  global U(4) rotation of the components, and with zero small components
  equal the two-component carrier's, so the fold needs no refusal there.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

NK, NS, MU, NB, NOCC = 2, 2, 6, 8, 3
C = 4.0 / (NK * 1 * NS)          # pref_inter, nspin = 1, nspinor = 2
OMEGAS = np.array([0.4j, 0.15 + 0.6j, 1.3j, -0.15 - 0.6j, -0.4j])


def _meshes():
    out = [Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))]
    if len(jax.devices()) >= 4:
        out.append(Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y")))
    return out


def _system(seed, trs):
    rng = np.random.default_rng(seed)
    e = np.sort(rng.uniform(-1.5, 1.5, (NK, NB)), axis=1)
    e[:, NOCC:] += 0.4                                   # an insulating gap
    f = np.zeros((NK, NB)); f[:, :NOCC] = 1.0
    psi = rng.normal(size=(NK, NS, MU, NB))
    D = rng.normal(size=(3, NK, NB, NB))
    if not trs:
        psi = psi + 1j * rng.normal(size=psi.shape)
        D = D + 1j * rng.normal(size=D.shape)
    D = 0.5 * (D + np.conj(np.swapaxes(D, -1, -2)))       # Hermitian position vertex
    dE = e[:, :, None] - e[:, None, :]
    v = -dE[None] * D                                    # energy-scaled head vertex P
    return e, f, psi.astype(np.complex128), D.astype(np.complex128), v.astype(np.complex128)


def _kernel_wings(mesh, e, f, psi, v):
    from gw.qsgw_head import head_wings_sharded
    wfns = SimpleNamespace(layout="face", psi_mun=jnp.asarray(psi),
                           psi_nmu=jnp.asarray(psi.transpose(0, 3, 1, 2)),
                           enk=jnp.asarray(e), occ=jnp.asarray(f), slices=None,
                           green_parent=None)
    Y, Z = head_wings_sharded(jnp.asarray(v), wfns, jnp.asarray(e), jnp.asarray(f),
                              jnp.asarray(OMEGAS), mesh=mesh, nb_logical=NB, nk_tot=NK,
                              nspin=1, nspinor=NS, eta_ry=0.0)
    return np.asarray(Y), np.asarray(Z)


def _lehmann_wings(e, f, psi, D):
    """Direct chi0 wing: every ordered pair x != y, (f_y - f_x) conj(D_xy) rho_xy / (z - dE)."""
    rho = np.einsum("ksmx,ksmy->kxym", np.conj(psi), psi)
    dE = e[:, :, None] - e[:, None, :]
    fd = f[:, None, :] - f[:, :, None]
    off = ~np.eye(NB, dtype=bool)[None]
    Y = np.zeros((len(OMEGAS), 3, MU), complex); Z = np.zeros((len(OMEGAS), MU, 3), complex)
    for w, z in enumerate(OMEGAS):
        g = np.where(off, 0.5 * C * fd / (z - dE + (~off)), 0.0)
        Y[w] = np.einsum("akxy,kxy,kxym->am", np.conj(D), g, rho)
        Z[w] = np.einsum("akxy,kxy,kxym->ma", D, g, np.conj(rho))
    return Y, Z


def _old_even_wings(e, f, psi, v):
    """The pre-2026-10-19 kernel: energy-ordered pairs, -c f A / (z^2 - dE^2)."""
    rho = np.einsum("ksmx,ksmy->kxym", np.conj(psi), psi)
    dE = e[:, :, None] - e[:, None, :]
    fd = f[:, None, :] - f[:, :, None]
    Y = np.zeros((len(OMEGAS), 3, MU), complex)
    for w, z in enumerate(OMEGAS):
        g = np.where(dE > 0, -C * fd / (z * z - dE * dE), 0.0)
        Y[w] = np.einsum("akxy,kxy,kxym->am", np.conj(v), g, rho)
    return Y


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_ordered_wings_equal_the_direct_lehmann_sum():
    """Wings equal a direct Lehmann sum over all ordered pairs (the even TRS weight misses it; Y(-z) = Z(z)^T), also for the four-component carrier."""
    # --- wings_trs_broken_match_direct_lehmann
    for mesh in _meshes():
        e, f, psi, D, v = _system(2, trs=False)
        Y, Z = _kernel_wings(mesh, e, f, psi, v)
        Y_ref, Z_ref = _lehmann_wings(e, f, psi, D)
        assert _rel(Y, Y_ref) < 1e-12, _rel(Y, Y_ref)
        assert _rel(Z, Z_ref) < 1e-12, _rel(Z, Z_ref)
        # The case is sensitive: the even TRS weight misses the odd-in-z wing here.
        assert _rel(_old_even_wings(e, f, psi, v), Y_ref) > 1e-2
        # Y(-z) = Z(z)^T (OMEGAS[3] = -OMEGAS[1], OMEGAS[4] = -OMEGAS[0]).
        for neg, pos in ((3, 1), (4, 0)):
            assert _rel(Y[neg], Z[pos].T) < 1e-12

    # --- four_component_carrier_wings_are_spin_traced
    for mesh in _meshes():
        e, f, psi2, D, v = _system(4, trs=False)
        rng = np.random.default_rng(5)
        psi4 = (rng.normal(size=(NK, 4, MU, NB))
                + 1j * rng.normal(size=(NK, 4, MU, NB))).astype(np.complex128)
        Y, Z = _kernel_wings(mesh, e, f, psi4, v)
        Y_ref, Z_ref = _lehmann_wings(e, f, psi4, D)
        assert _rel(Y, Y_ref) < 1e-12, _rel(Y, Y_ref)
        assert _rel(Z, Z_ref) < 1e-12, _rel(Z, Z_ref)
        # psi^dagger psi is invariant under one U(4) acting on every state.
        u, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        Y_u, Z_u = _kernel_wings(mesh, e, f, np.einsum("st,ktmn->ksmn", u, psi4), v)
        assert _rel(Y_u, Y) < 1e-12 and _rel(Z_u, Z) < 1e-12
        # Zero small components: the two-component carrier's wings.
        lifted = np.concatenate([psi2, np.zeros_like(psi2)], axis=1)
        Y_l, Z_l = _kernel_wings(mesh, e, f, lifted, v)
        Y_2, Z_2 = _kernel_wings(mesh, e, f, psi2, v)
        assert _rel(Y_l, Y_2) < 1e-12 and _rel(Z_l, Z_2) < 1e-12


def _body(mesh, ordered, z, b, poles, counts):
    import distrib_la
    from gw.shared_pole_head import _gamma_body
    face = NamedSharding(mesh, P(None, "x", "y"))
    rep = NamedSharding(mesh, P())
    n = b.shape[1]
    ev = _gamma_body(mesh, ("off", distrib_la.ROUTE_BATCH_RESHARD), ordered)
    out = ev(jnp.asarray(z, jnp.complex128), jax.device_put(jnp.asarray(b), face),
             jax.device_put(jnp.asarray(poles), rep), jax.device_put(jnp.asarray(counts), rep),
             jax.device_put(jnp.zeros((1, n, n), jnp.complex128), face))
    return np.asarray(out)[0]

