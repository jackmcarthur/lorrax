"""The full Gamma head on an ordered (time-reversal-broken) shared-pole store.

``gw.shared_pole_head`` folds ``S + Y W Z / Omega`` with the store's signed
body and the Lehmann wings of every ordered pair.  Checked here on CPU:

* wings: the face kernel equals a direct numpy Lehmann sum over all ordered
  pairs on a random TRS-broken system, and on a time-reversal-even system
  (real psi, real position vertex) the old energy-ordered even weight
  ``-c f A / (z^2 - dE^2)`` to roundoff; ``Y(-z) = Z(z)^T``;
* body: the ordered Gamma body equals the store normalization
  ``sum_p b b^H/(2W(z-W)) - conj(b) b^T/(2W(z+W))``, reduces to the TRS
  ``b (z^2 - Lambda)^-1 b^H`` for real ``b``, and obeys ``W(-z) = W(z)^T``;
* gate: full is admitted on an ordered insulator and refused on an ordered metal.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_wings_trs_even_agree_with_old_weight(mesh):
    e, f, psi, D, v = _system(1, trs=True)
    Y, Z = _kernel_wings(mesh, e, f, psi, v)
    Y_old = _old_even_wings(e, f, psi, v)
    Y_ref, Z_ref = _lehmann_wings(e, f, psi, D)
    assert _rel(Y, Y_old) < 1e-12, _rel(Y, Y_old)
    assert _rel(Y_ref, Y_old) < 1e-12
    assert _rel(Z, Z_ref) < 1e-12


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_wings_trs_broken_match_direct_lehmann(mesh):
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


@pytest.mark.parametrize("mesh", _meshes(), ids=lambda m: f"P{m.size}")
def test_ordered_body_is_store_normalization(mesh):
    rng = np.random.default_rng(3)
    n, K, k = 8, 6, 5                                    # the sixth pole is inactive
    b = rng.normal(size=(1, n, K)) + 1j * rng.normal(size=(1, n, K))
    poles = np.concatenate([rng.uniform(0.2, 4.0, (1, k)), np.zeros((1, K - k))], axis=1)
    counts = np.array([k])
    W = np.sqrt(poles[0, :k]); bb = b[0, :, :k]
    for z in (0.7j, 0.3 + 1.1j):
        got = _body(mesh, True, z, b, poles, counts)
        ref = ((bb / (2 * W * (z - W))) @ bb.conj().T
               - (bb.conj() / (2 * W * (z + W))) @ bb.T)
        assert _rel(got, ref) < 1e-12
        assert _rel(_body(mesh, True, -z, b, poles, counts), got.T) < 1e-12
    # Real b: the ordered form is the TRS-even b (z^2 - Lambda)^-1 b^H.
    br = b.real.astype(np.complex128)
    for z in (0.7j, 0.3 + 1.1j):
        trs = _body(mesh, False, z, br, poles, counts)
        ref = (br[0, :, :k] / (z * z - W * W)) @ br[0, :, :k].T
        assert _rel(trs, ref) < 1e-12
        assert _rel(_body(mesh, True, z, br, poles, counts), trs) < 1e-12


def test_gate_admits_ordered_insulator_refuses_ordered_metal():
    from gw.gw_config import HeadCorrection
    from gw.shared_pole_head import refuse_unsupported_shared_pole_head
    config = SimpleNamespace(sigma=SimpleNamespace(w_model="shared_pole"),
                             head=SimpleNamespace(correction=HeadCorrection.FULL),
                             bispinor=False)
    refuse_unsupported_shared_pole_head(config, trs_allowed=False, nspinor=2,
                                        material_class="insulator")
    refuse_unsupported_shared_pole_head(config, trs_allowed=True, nspinor=2,
                                        material_class="metal")
    with pytest.raises(ValueError, match="GATE shared_pole_head_ordered"):
        refuse_unsupported_shared_pole_head(config, trs_allowed=False, nspinor=2,
                                            material_class="metal")
