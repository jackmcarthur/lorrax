"""``sc_head_update = interband_commutator``: kernel, exactness, grammar.

1. The kernel equals a dense numpy ``v + [DeltaH, W]`` on a padded,
   mesh-non-divisible manifold with an exactly degenerate pair.
2. The exactness statement, against finite differences of a tight-binding
   ``H(k)`` that share no code with the kernel: a ``DeltaH`` with no
   valence-conduction block gives the exact valence-conduction velocity,
   and a cross-gap mixing ``epsilon`` gives an error linear in ``epsilon``
   (the negative control).
3. Grammar: the value parses, a metal refuses by name, the loader refuses a
   velocity without the nonlocal commutator, and dispatch never reaches the
   link loader or its stencil preflight.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from gw import qsgw_head
from gw.degen_average import TOL_DEGENERACY_RY
from gw.gw_config import (
    HEAD_UPDATES, INSULATOR_HEAD_UPDATES, LorraxConfig,
    validate_material_inputs)
from gw.qsgw_head import interband_commutator_velocity
from gw.sc_iteration import load_head_velocity_source

jax.config.update("jax_enable_x64", True)


def _mesh():
    devices = np.asarray(jax.devices())
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _hermitian(rng, *shape):
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))


def _reference(v, delta_active, tail, e, nb_logical):
    nk, nbs = e.shape
    na = delta_active.shape[-1]
    de = e[:, :, None] - e[:, None, :]
    live = np.arange(nbs) < nb_logical
    pair = live[:, None] & live[None, :] & ~np.eye(nbs, dtype=bool)
    keep = pair[None] & (np.abs(de) > TOL_DEGENERACY_RY)
    W = np.where(keep[None], v / np.where(keep, de, 1.0)[None], 0.0)
    dH = np.zeros((nk, nbs, nbs), dtype=np.complex128)
    dH[:, :na, :na] = delta_active
    idx = np.arange(na, nbs)
    dH[:, idx, idx] = tail[:, na:]
    C = np.einsum("kml,akln->akmn", dH, W) - np.einsum("akml,kln->akmn", W, dH)
    return v + C, keep, pair


# ---------------------------------------------------------------------------
# 1. Kernel against the dense formula
# ---------------------------------------------------------------------------

def test_kernel_matches_dense_commutator_with_padding_and_a_degenerate_pair():
    rng = np.random.default_rng(2509)
    nk, nbl, nbs, na, n_occ = 3, 7, 8, 5, 3
    e = np.sort(rng.uniform(-1.0, 1.0, size=(nk, nbs)), axis=1)
    e[:, 1] = e[:, 2] + 0.25 * TOL_DEGENERACY_RY  # one exact multiplet
    e[:, nbl:] = e[:, :1]                         # padding energy collides
    v = _hermitian(rng, 3, nk, nbs, nbs)
    v[..., nbl:, :] = 0.0
    v[..., :, nbl:] = 0.0
    delta = _hermitian(rng, nk, na, na)
    tail = rng.normal(size=(nk, nbs))
    got, (num, den, excluded, min_kept) = interband_commutator_velocity(
        jnp.asarray(v), jnp.asarray(delta), jnp.asarray(tail), jnp.asarray(e),
        nb_logical=nbl, n_occ=n_occ, mesh=_mesh())
    ref, keep, pair = _reference(v, delta, tail, e, nbl)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12)
    # The multiplet pair is excluded in both orders at every k.
    assert int(excluded) == int(np.sum(pair[None] & ~keep)) == 2 * nk
    de = np.abs(e[:, :, None] - e[:, None, :])
    assert float(min_kept) == pytest.approx(float(np.min(de[keep])))
    vc = np.zeros((nbs, nbs), dtype=bool)
    vc[:n_occ, n_occ:nbl] = True
    C = ref - v
    np.testing.assert_allclose(
        np.asarray(num), np.sum(np.abs(C[:, :, vc]) ** 2, axis=(1, 2)),
        rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(den), np.sum(np.abs(v[:, :, vc]) ** 2, axis=(1, 2)),
        rtol=1e-12)


def test_band_diagonal_delta_h_renormalizes_every_nondegenerate_pair():
    rng = np.random.default_rng(7)
    nk, nb, n_occ = 2, 6, 2
    e = np.sort(rng.uniform(-1.0, 1.0, size=(nk, nb)), axis=1)
    shift = rng.normal(size=(nk, nb))
    v = _hermitian(rng, 3, nk, nb, nb)
    delta = np.zeros((nk, nb, nb), dtype=np.complex128)
    delta[:, np.arange(nb), np.arange(nb)] = shift
    got, _ = interband_commutator_velocity(
        jnp.asarray(v), jnp.asarray(delta), jnp.asarray(shift), jnp.asarray(e),
        nb_logical=nb, n_occ=n_occ, mesh=_mesh())
    eq = e + shift
    ratio = (eq[:, :, None] - eq[:, None, :]) / np.where(
        np.eye(nb, dtype=bool)[None], 1.0, e[:, :, None] - e[:, None, :])
    want = np.where(np.eye(nb, dtype=bool)[None, None], v, v * ratio[None])
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Exactness, against finite differences of a tight-binding H(k)
# ---------------------------------------------------------------------------

def _tb_model(rng, nb):
    h0 = _hermitian(rng, nb, nb) + np.diag(np.linspace(-3.0, 3.0, nb))
    c = [_hermitian(rng, nb, nb) * 0.3 for _ in range(3)]
    s = [_hermitian(rng, nb, nb) * 0.3 for _ in range(3)]

    def H(k):
        return h0 + sum(np.cos(k[a]) * c[a] + np.sin(k[a]) * s[a]
                        for a in range(3))

    def dH(k):
        return np.stack([-np.sin(k[a]) * c[a] + np.cos(k[a]) * s[a]
                         for a in range(3)])
    return H, dH


def _delta_h_orbital(H, X, Y, Z, eps, n_occ):
    """Lattice-periodic DeltaH(k) = P_V X P_V + P_C Y P_C + eps (P_V Z P_C + h.c.)."""
    def D(k):
        _, U = np.linalg.eigh(H(k))
        PV = U[:, :n_occ] @ U[:, :n_occ].conj().T
        PC = np.eye(len(PV)) - PV
        mix = PV @ Z(k) @ PC
        return PV @ X(k) @ PV + PC @ Y(k) @ PC + eps * (mix + mix.conj().T)
    return D


def _vc_error(eps, *, seed=31):
    rng = np.random.default_rng(seed)
    nb, n_occ, nk, h = 6, 3, 3, 1.0e-4
    H, dH = _tb_model(rng, nb)
    gx, gy, gz = (_hermitian(rng, nb, nb) for _ in range(3))
    X = lambda k: 0.2 * gx * np.cos(k[0]) + 0.1 * gy        # noqa: E731
    Y = lambda k: 0.3 * gy * np.sin(k[1]) + 0.2 * gz        # noqa: E731
    Z = lambda k: gz * np.cos(k[2]) + gx                    # noqa: E731
    D = _delta_h_orbital(H, X, Y, Z, eps, n_occ)
    ks = rng.uniform(-np.pi, np.pi, size=(nk, 3))
    v = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    e = np.zeros((nk, nb))
    delta = np.zeros((nk, nb, nb), dtype=np.complex128)
    exact = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    for ik, k in enumerate(ks):
        e[ik], U = np.linalg.eigh(H(k))
        v[:, ik] = np.einsum("im,aij,jn->amn", U.conj(), dH(k), U)
        delta[ik] = U.conj().T @ D(k) @ U
        for a in range(3):
            dk = np.zeros(3)
            dk[a] = h
            dD = (D(k + dk) - D(k - dk)) / (2.0 * h)
            exact[a, ik] = v[a, ik] + U.conj().T @ dD @ U
    got, _ = interband_commutator_velocity(
        jnp.asarray(v), jnp.asarray(delta), jnp.zeros((nk, nb)),
        jnp.asarray(e), nb_logical=nb, n_occ=n_occ, mesh=_mesh())
    got = np.asarray(got)
    vc = (slice(None), slice(None), slice(0, n_occ), slice(n_occ, nb))
    return (np.linalg.norm(got[vc] - exact[vc])
            / np.linalg.norm(exact[vc] - v[vc]))


def test_no_cross_gap_block_means_an_exact_valence_conduction_velocity():
    # Finite differences at h = 1e-4 leave O(h^2) ~ 1e-8.
    assert _vc_error(0.0) < 1.0e-6


def test_cross_gap_mixing_error_is_first_order_in_the_mixing():
    # Negative control: the same check fires once DeltaH mixes the classes,
    # and the error doubles with the mixing.
    e1, e2 = _vc_error(0.02), _vc_error(0.04)
    assert e1 > 1.0e-3
    assert 1.7 < e2 / e1 < 2.3


# ---------------------------------------------------------------------------
# 3. Grammar, refusals and dispatch
# ---------------------------------------------------------------------------

_BASE = """\
[cohsex]
sys_dim = 3
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
qp_solver = self_consistent
"""


def _config(tmp_path, extra: str = ""):
    path = tmp_path / "ich.in"
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


def test_the_value_parses_and_joins_the_head_vocabulary(tmp_path):
    assert INSULATOR_HEAD_UPDATES == ("interband_commutator",)
    assert "interband_commutator" in HEAD_UPDATES
    cfg = _config(tmp_path, "sc_head_update = interband_commutator\n")
    assert cfg.sc.head_update == "interband_commutator"


def test_a_typo_refuses_and_names_the_new_value(tmp_path):
    with pytest.raises(ValueError, match="interband_commutator"):
        _config(tmp_path, "sc_head_update = interband_commutators\n")


def test_a_metal_refuses_by_name(tmp_path):
    cfg = _config(tmp_path, "compute_mode = mpa\n"
                            "sc_head_update = interband_commutator\n")
    with pytest.raises(
            ValueError,
            match="GATE sc_head_interband_commutator_insulator_only"):
        validate_material_inputs(cfg, "metal")


def _stub(kgrid, mode):
    config = SimpleNamespace(
        sc=SimpleNamespace(head_update=mode), do_G0=True,
        paths=SimpleNamespace(parallel_transport_file="pt.h5"))
    wfn = SimpleNamespace(energies=np.zeros((1, 4)), kgrid=kgrid)
    meta = SimpleNamespace(b_id_4_user=4)
    return config, wfn, meta


def test_dispatch_reads_no_links_and_needs_no_stencil(tmp_path, monkeypatch):
    sentinel = SimpleNamespace(nb_logical=4)
    monkeypatch.setattr(
        qsgw_head, "load_interband_commutator_head",
        lambda *a, **k: sentinel)

    def _trap(*a, **k):
        raise AssertionError("link loader reached")
    monkeypatch.setattr(qsgw_head, "load_parallel_transport_head", _trap)
    kgrid = (2, 2, 1)  # a two-point axis: no link stencil exists
    config, wfn, meta = _stub(kgrid, "interband_commutator")
    got = load_head_velocity_source(
        config, str(tmp_path), mesh=None, sym=SimpleNamespace(trs_allowed=True),
        wfn=wfn, meta=meta, material_class="insulator",
        print_fn=lambda *a, **k: None)
    assert got is sentinel
    # Control: the same grid refuses the link route before any read.
    config, wfn, meta = _stub(kgrid, "parallel_transport")
    with pytest.raises(ValueError, match="pt_head_stencil_unsupported"):
        load_head_velocity_source(
            config, str(tmp_path), mesh=None,
            sym=SimpleNamespace(trs_allowed=True), wfn=wfn, meta=meta,
            material_class="insulator", print_fn=lambda *a, **k: None)


class _FakeSlab:
    def __init__(self, stamp):
        self.stamp = stamp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read_small(self, name, dtype=None):
        if self.stamp is None:
            raise KeyError(name)
        return np.asarray(self.stamp, dtype=np.int32)


@pytest.mark.parametrize("stamp", [0, None])
def test_the_loader_refuses_a_velocity_without_the_nonlocal_commutator(
        monkeypatch, stamp):
    import file_io.slab_io as slab_io

    monkeypatch.setattr(slab_io, "SlabIO", lambda *a, **k: _FakeSlab(stamp))

    def _trap(*a, **k):
        raise AssertionError("velocity read before the operator stamp")
    monkeypatch.setattr(qsgw_head, "load_dft_velocity_head", _trap)
    with pytest.raises(
            ValueError,
            match="GATE sc_head_interband_commutator_velocity_operator"):
        qsgw_head.load_interband_commutator_head(
            "pt.h5", mesh=None, wfn=None, meta=None, config=None)
