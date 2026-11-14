"""The loader's convective carriers ``convective_1..3`` (stage A loader hook).

CPU, pytest, eager WfnLoader on the hsuite H2 fixture.  With the field of
``psp.vnl_ops.convective_field`` attached:

* ``raw`` and ``large_block`` are bit for bit what a loader without the field
  returns (the default run does not move);
* ``convective_a`` equals ``raw`` plus (HALFALPHA/2) sigma^a gamma_a, so its
  q = 0 channel-a vertex is HALFALPHA <m|2(k+G)_a + d_a V_NL|n> (the plan's
  test 3.1, on the loader's carrier), with the large block untouched;
* ``load``, ``load_process_local`` and ``unfold_parent_to_full_k`` (the
  child path of the centroid faces and the route-G zeta source) return the
  same convective carrier;
* without a field a convective carrier refuses, a second field of another
  identity refuses, and the carrier names stamp their own provenance while
  ``lift_to_4spinor`` never accepts them.

Not covered here (needs the native library): the phdf5 backend, whose lift
goes through the same ``_apply_bispinor_lift``.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "hsuite" / "fixture"
STAMP = "H=" + hashlib.md5((FIXTURE / "H.upf").read_bytes()).hexdigest()


def _loader():
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh
    from wfn_loader import WfnLoader
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    return WfnLoader(FIXTURE / "WFN.h5", mesh=mesh, backend="eager")


def _attached(wfn):
    from psp.vnl_ops import build_vnl_setup_from_pseudo_dir, convective_field
    setup, identity = build_vnl_setup_from_pseudo_dir(wfn, str(FIXTURE), kin_ion_md5=STAMP)
    wfn.attach_current_vertex_field(convective_field(setup), identity=identity)
    return setup


def test_convective_carriers_from_the_loader():
    pytest.importorskip("xsdata")
    import jax.numpy as jnp
    from common.bispinor_init import HALFALPHA
    from common.gamma_matrices import gamma1, gamma2, gamma3
    from psp.vnl_ops import build_vnl_kdata_from_kvec, vnl_velocity_matrix
    rows = [0, 7, 24]                                    # full-BZ rows
    with _loader() as bare:
        ref = {lift: np.asarray(bare.load(bands=(0, 4), k=rows, bispinor=True, bispinor_lift=lift))
               for lift in ("raw", "large_block")}
        with pytest.raises(ValueError, match="GATE convective_vertex_field"):
            bare.load(bands=(0, 4), k=rows, bispinor=True, bispinor_lift="convective_1")
    with _loader() as wfn:
        setup = _attached(wfn)
        for lift, want in ref.items():
            assert np.array_equal(np.asarray(
                wfn.load(bands=(0, 4), k=rows, bispinor=True, bispinor_lift=lift)), want)
        psi2 = np.asarray(wfn.load(bands=(0, 4), k=rows))
        gvecs, kvecs, ngk = (np.asarray(wfn.gvecs(k=rows)), np.asarray(wfn.kvecs(k=rows)),
                             np.asarray(wfn.ngk_valid(k=rows)))
        B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=float)
        sym = wfn.symmetry()
        worst, nl = 0.0, 0.0
        for a, alpha in zip((1, 2, 3), (gamma1, gamma2, gamma3)):
            name = f"convective_{a}"
            conv = np.asarray(wfn.load(bands=(0, 4), k=rows, bispinor=True, bispinor_lift=name))
            assert np.array_equal(conv[:, :, :2], ref["raw"][:, :, :2])
            local = np.asarray(wfn.load_process_local(bands=(0, 4), k=rows, bispinor=True,
                                                      bispinor_lift=name))
            np.testing.assert_array_equal(local, conv)
            for i, k in enumerate(rows):
                n = int(ngk[i])
                parent = int(np.asarray(sym.irr_idx_k)[k])
                from wfn_loader import IBZRows
                from jax.sharding import PartitionSpec as P
                ppsi = wfn.load(bands=(0, 4), k=IBZRows((parent,)),
                                sharding=P(None, ("x", "y"), None, None))
                child = np.asarray(wfn.unfold_parent_to_full_k(
                    ppsi, parent=parent, full_k=k, bispinor=True, bispinor_lift=name))
                np.testing.assert_allclose(child[0], conv[i], rtol=0,
                                           atol=1e-14 * np.abs(conv[i]).max())
                kd = build_vnl_kdata_from_kvec(kvecs[i], gvecs[i, :n], setup, compute_dZ=True)
                p = jnp.asarray(psi2[i, :, :, :n])
                v_nl = np.asarray(vnl_velocity_matrix(p, kd.Z, kd.dZ, setup.E_super))[a - 1]
                K = (gvecs[i, :n] + kvecs[i]) @ B
                p2 = np.einsum("msg,g,nsg->mn", np.conj(psi2[i, :, :, :n]), 2.0 * K[:, a - 1],
                               psi2[i, :, :, :n])
                c = conv[i, :, :, :n]
                vertex = np.einsum("msg,st,ntg->mn", np.conj(c), np.asarray(alpha), c)
                want = HALFALPHA * (p2 + v_nl)
                # Cauchy-Schwarz scale of the vertex (unit states): HALFALPHA max|2(k+G)|;
                # a channel can vanish by symmetry at a k and have no scale of its own.
                scale = HALFALPHA * 2.0 * float(np.abs(K).max())
                err = float(np.abs(vertex - want).max()) / scale
                worst = max(worst, err)
                nl = max(nl, HALFALPHA * float(np.abs(v_nl).max()) / scale)
                assert err < 1e-11, (a, k, err)
                # The increment alone against HALFALPHA <d_a V_NL>, on the nonlocal
                # term's own scale (the Cauchy-Schwarz bound of gamma).
                # alpha^a couples the blocks only, so the increment d of the small block
                # enters the vertex linearly: psi_L,m+ sigma^a d_n + d_m+ sigma^a psi_L,n
                # (formed directly: subtracting two vertices would cancel at 1e-18).
                r = ref["raw"][i, :, :, :n]
                d = c[:, 2:] - r[:, 2:]
                sa = np.asarray(alpha)[:2, 2:]
                inc = (np.einsum("msg,st,ntg->mn", np.conj(r[:, :2]), sa, d)
                       + np.einsum("msg,st,ntg->mn", np.conj(d), sa.conj().T, r[:, :2]))
                gamma_norm = float(np.max(np.linalg.norm(d.reshape(d.shape[0], -1), axis=1)))
                np.testing.assert_allclose(inc, HALFALPHA * v_nl, rtol=0, atol=1e-12 * gamma_norm)
                assert not np.any(conv[i, :, :, n:])
        print(f"q = 0 vertex of the loader's convective carriers: worst {worst:.1e}; "
              f"the nonlocal part is {nl:.1e} of the scale")
        assert nl > 1e3 * max(worst, 1e-16), (nl, worst)      # the check sees the term


def test_identity_and_provenance():
    pytest.importorskip("xsdata")
    from common.bispinor_init import (carrier_provenance, kinetic_balance_lift_provenance,
                                      lift_to_4spinor)
    with _loader() as wfn:
        _attached(wfn)
        _attached(wfn)                                   # the same field again: accepted
        with pytest.raises(ValueError, match="already attached"):
            wfn.attach_current_vertex_field(lambda *a: None, identity="H=other")
    names = [carrier_provenance(f"convective_{a}") for a in (1, 2, 3)]
    assert len(set(names)) == 3 and all("i[V_NL,r_" in n for n in names)
    assert carrier_provenance("raw") == kinetic_balance_lift_provenance("raw")
    with pytest.raises(ValueError, match="unknown kinetic-balance representation"):
        kinetic_balance_lift_provenance("convective_1")
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="unknown kinetic-balance representation"):
        lift_to_4spinor(jnp.zeros((1, 1, 2, 3), complex), jnp.zeros((1, 3, 3)), jnp.zeros((1, 3)),
                        jnp.eye(3), representation="convective_1")
