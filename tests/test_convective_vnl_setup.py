"""The convective vertex's in-process V_NL: the kin_ion setup, from the run's own files.

CPU, pytest, the hsuite H2 fixture (fully relativistic H, nspinor = 2) read
by the eager loader.  ``psp.vnl_ops.build_vnl_setup_from_pseudo_dir`` builds
the same ``VNLSetup`` as ``gw.kin_ion_io`` (bit for bit) and refuses another
or an absent ``kin_ion.h5`` stamp and an empty directory;
``psp.vnl_ops.convective_field`` is sigma^a i[V_NL, r_a] psi_L of
``convective_velocity_field`` on each k's own sphere, zero on the pad rows.
The deck key ``pseudo_dir`` resolves against the input file.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "hsuite" / "fixture"


def _loader():
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh
    from wfn_loader import WfnLoader
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    return WfnLoader(FIXTURE / "WFN.h5", mesh=mesh, backend="eager")


def test_pseudo_dir_key(tmp_path):
    from gw.gw_config import LorraxConfig
    deck = ("[cohsex]\nsys_dim = 3\ncompute_mode = tangential\nqp_solver = one_shot_dft\n"
            "number_bands = 30\nnval = 4\nncond = 4\nbispinor = true\n")
    path = tmp_path / "gw.in"
    for extra, want in (("", ""), ("pseudo_dir = pp\n", str(tmp_path / "pp")),
                        ("pseudo_dir = /abs/pp\n", "/abs/pp")):
        path.write_text(deck + extra)
        cfg = LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                           runtime_platform="cpu", print_fn=lambda *_a: None)
        assert cfg.pseudo_dir == want


def test_setup_is_the_hamiltonians_and_refusals(tmp_path):
    pytest.importorskip("xsdata")
    from psp.pseudos import load_pseudopotentials
    from psp.vnl_ops import build_vnl_setup, build_vnl_setup_from_pseudo_dir
    stamp = "H=" + hashlib.md5((FIXTURE / "H.upf").read_bytes()).hexdigest()
    with _loader() as wfn:
        setup, identity = build_vnl_setup_from_pseudo_dir(wfn, str(FIXTURE), kin_ion_md5=stamp)
        assert identity == stamp and setup.soc
        # gw.kin_ion_io's call, on the same WFN and files.
        ref = build_vnl_setup(wfn, wfn.symmetry(), None, load_pseudopotentials(str(FIXTURE)),
                              nspinor=int(wfn.nspinor))
        for name in ("E_super", "G_table", "Gp_table", "row_beta_idx", "row_l", "row_m", "row_tau"):
            assert np.array_equal(np.asarray(getattr(setup, name)), np.asarray(getattr(ref, name))), name
        assert (setup.q_max, setup.n_q, setup.total_R) == (ref.q_max, ref.n_q, ref.total_R)
        for stored in (None, stamp.replace(stamp[-4:], "0000")):
            with pytest.raises(ValueError, match="GATE convective_vertex_pseudopotentials"):
                build_vnl_setup_from_pseudo_dir(wfn, str(FIXTURE), kin_ion_md5=stored)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no \\*.upf"):
            build_vnl_setup_from_pseudo_dir(wfn, str(empty), kin_ion_md5=stamp)


def test_field_on_the_loader_rows():
    pytest.importorskip("xsdata")
    import jax.numpy as jnp
    from wfn_loader import IBZRows
    from common.gamma_matrices import paulis
    from psp.vnl_ops import (build_vnl_kdata_from_kvec, build_vnl_setup_from_pseudo_dir,
                             convective_field, vnl_velocity_matrix)
    stamp = "H=" + hashlib.md5((FIXTURE / "H.upf").read_bytes()).hexdigest()
    with _loader() as wfn:
        setup, _ = build_vnl_setup_from_pseudo_dir(wfn, str(FIXTURE), kin_ion_md5=stamp)
        rows = IBZRows((0, 3))
        psi = wfn.load(bands=(0, 4), k=rows)
        gvecs, kvecs, ngk = (np.asarray(wfn.gvecs(k=rows)), np.asarray(wfn.kvecs(k=rows)),
                             np.asarray(wfn.ngk_valid(k=rows)))
        field = convective_field(setup)
        for a in (1, 2, 3):
            sg = np.asarray(field(psi, gvecs, kvecs, ngk, a))
            assert sg.shape == psi.shape
            for i in range(2):
                n = int(ngk[i])
                assert not np.any(sg[i, :, :, n:])                     # pad rows
                kd = build_vnl_kdata_from_kvec(kvecs[i], gvecs[i, :n], setup, compute_dZ=True)
                p = jnp.asarray(psi[i, :, :, :n])
                v = np.asarray(vnl_velocity_matrix(p, kd.Z, kd.dZ, setup.E_super))[a - 1]
                # sigma^a sigma^a = 1: <psi_m| sigma^a (sigma^a gamma_a)_n> = <m|d_a V_NL|n>.
                got = np.einsum("msg,st,ntg->mn", np.conj(np.asarray(p)),
                                np.asarray(paulis[a - 1]), sg[i, :, :, :n])
                # Round-off of two G orderings (ngkmax with pad vs the sphere), on the
                # Cauchy-Schwarz scale |<m|sigma gamma_n>| <= ||gamma_n|| (some blocks
                # vanish by symmetry and have no scale of their own).
                scale = float(np.max(np.linalg.norm(sg[i].reshape(sg.shape[1], -1), axis=1)))
                np.testing.assert_allclose(got, v, rtol=0, atol=1e-12 * scale)
                if a == 1 and i == 0:
                    assert np.abs(v).max() > 1e-3 * scale          # not a vacuous block
