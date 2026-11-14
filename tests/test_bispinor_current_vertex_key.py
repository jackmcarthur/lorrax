"""The current-vertex key of a bispinor run and the pseudopotential identity it needs.

CPU, pytest.  ``bispinor_current_vertex`` (docs/input_reference.md): the
default ``kinetic`` leaves the stamp and the carrier of the parent commit
unchanged (the same object); ``convective`` refuses where it cannot act and,
until stage A reads the per-channel carriers, everywhere
(``GATE bispinor_current_vertex_stage``).  ``kin_ion.h5``'s UPF stamp
(``pseudopotential_md5``) and its refusal on another or an absent stamp; the
refusal of ultrasoft/PAW files; the carrier's increment sits in the small
block alone with the scale HALFALPHA / 2.
"""

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gw.gw_config import LorraxConfig

FIXTURE = Path(__file__).parent / "hsuite" / "fixture" / "H.upf"

_BISPINOR = """[cohsex]
sys_dim = 3
compute_mode = tangential
qp_solver = one_shot_dft
number_bands = 30
nval = 4
ncond = 4
bispinor = true
"""


def _config(tmp_path, text):
    path = tmp_path / "gw.in"
    path.write_text(text)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform="cpu",
                                       print_fn=lambda *_a: None)


def test_key_default_stamp_and_gates(tmp_path):
    from gw.gw_config import bispinor_transverse_terms_stamp, bispinor_current_vertex_stamp
    default = _config(tmp_path, _BISPINOR)
    assert default.bispinor_current_vertex == "kinetic"
    assert bispinor_current_vertex_stamp(default) is None
    # The parent commit's stamp, character for character.
    assert bispinor_transverse_terms_stamp(default) == "hartree=on; exchange=on; semicore=none"
    named = _config(tmp_path, _BISPINOR + "bispinor_current_vertex = kinetic\n")
    assert bispinor_transverse_terms_stamp(named) == bispinor_transverse_terms_stamp(default)
    # The stamp a convective run will carry (the gate keeps it from running today).
    as_convective = SimpleNamespace(**{**vars(default), "bispinor_current_vertex": "convective"})
    assert bispinor_transverse_terms_stamp(as_convective) == (
        "hartree=on; exchange=on; semicore=none; vertex=convective_vnl_v1")

    with pytest.raises(ValueError, match="GATE bispinor_current_vertex: unknown"):
        _config(tmp_path, _BISPINOR + "bispinor_current_vertex = velocity\n")
    for deck in (_BISPINOR.replace("bispinor = true", "bispinor = false"),
                 _BISPINOR + "bispinor_gw = full_tangential\n",
                 _BISPINOR + "restart = true\n"):
        with pytest.raises(ValueError, match="GATE bispinor_current_vertex_route"):
            _config(tmp_path, deck + "bispinor_current_vertex = convective\n")
    with pytest.raises(ValueError, match="GATE bispinor_current_vertex_stage"):
        _config(tmp_path, _BISPINOR + "bispinor_current_vertex = convective\n")


def test_pseudopotential_stamp_and_refusals(tmp_path):
    from psp.pseudos import (load_pseudopotentials, pseudopotential_md5_stamp,
                             refuse_pseudopotential_mismatch,
                             refuse_nonseparable_for_convective)
    pytest.importorskip("xsdata")
    here = tmp_path / "a"
    here.mkdir()
    shutil.copy(FIXTURE, here / "H.upf")
    pseudos = load_pseudopotentials(str(here))
    stamp = pseudopotential_md5_stamp(pseudos)
    assert stamp == "H=" + hashlib.md5(FIXTURE.read_bytes()).hexdigest()
    refuse_pseudopotential_mismatch(stamp, pseudos, consumer="test")
    refuse_pseudopotential_mismatch(stamp.encode(), pseudos, consumer="test")   # h5py bytes
    with pytest.raises(ValueError, match="GATE convective_vertex_pseudopotentials"):
        refuse_pseudopotential_mismatch(None, pseudos, consumer="test")         # old kin_ion.h5
    other = tmp_path / "b"
    other.mkdir()
    (other / "H.upf").write_text(FIXTURE.read_text() + "\n<!-- swapped D -->\n")
    swapped = load_pseudopotentials(str(other))
    with pytest.raises(ValueError, match="GATE convective_vertex_pseudopotentials"):
        refuse_pseudopotential_mismatch(stamp, swapped, consumer="test")
    in_memory = {"H": SimpleNamespace(pp_header=pseudos["H"].pp_header)}
    with pytest.raises(ValueError, match="GATE convective_vertex_pseudopotentials"):
        refuse_pseudopotential_mismatch("H=unknown", in_memory, consumer="test")

    refuse_nonseparable_for_convective(pseudos)
    for header in (SimpleNamespace(pseudo_type="US", is_ultrasoft="T", is_paw="F"),
                   SimpleNamespace(pseudo_type="PAW", is_ultrasoft="F", is_paw=True)):
        with pytest.raises(ValueError, match="GATE convective_vertex_pseudo_type"):
            refuse_nonseparable_for_convective({"X": SimpleNamespace(pp_header=header)})


def test_carrier_kinetic_is_the_raw_lift_and_convective_adds_to_the_small_block():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from common.bispinor_init import HALFALPHA, convective_carrier, lift_to_4spinor
    from common.gamma_matrices import gamma1, gamma2, gamma3, paulis
    rng = np.random.default_rng(3)
    nb, ng = 3, 17
    psi = rng.normal(size=(1, nb, 2, ng)) + 1j * rng.normal(size=(1, nb, 2, ng))
    raw = lift_to_4spinor(jnp.asarray(psi), jnp.asarray(rng.integers(-2, 3, (1, ng, 3)).astype(float)),
                          jnp.asarray(rng.random((1, 3))), jnp.eye(3) * 0.7, representation="raw")
    gamma = rng.normal(size=(3, 1, nb, 2, ng)) + 1j * rng.normal(size=(3, 1, nb, 2, ng))
    sg = jnp.einsum("aij,aknjg->aknig", paulis, jnp.asarray(gamma))
    for a, alpha in zip((1, 2, 3), (gamma1, gamma2, gamma3)):
        assert convective_carrier(raw, sg, a, vertex="kinetic") is raw
        conv = convective_carrier(raw, sg, a, vertex="convective")
        assert np.array_equal(np.asarray(conv[..., :2, :]), np.asarray(raw[..., :2, :]))
        # Psi^(a)+ alpha^a Psi^(a) = Psi+ alpha^a Psi + (HALFALPHA/2)(psi+ gamma_a + gamma_a+ psi)
        pair = lambda X: np.einsum("knsg,st,kmtg->knm", np.conj(X), np.asarray(alpha), X)
        added = 0.5 * HALFALPHA * (np.einsum("knsg,kmsg->knm", np.conj(psi), gamma[a - 1])
                                   + np.einsum("knsg,kmsg->knm", np.conj(gamma[a - 1]), psi))
        got = pair(np.asarray(conv)) - pair(np.asarray(raw))
        np.testing.assert_allclose(got, added, rtol=0, atol=1e-14 * np.abs(added).max())
    with pytest.raises(ValueError, match="label must be 1, 2 or 3"):
        convective_carrier(raw, sg, 0, vertex="convective")
