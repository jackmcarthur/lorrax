"""The unnamed ``sc_head_update`` resolves alike on two-component and bare_transverse decks.

CPU, pytest; deck parse only.  On the Pauli carrier the bare_transverse
charge head is the two-spinor dipole and links, the scalar object, so an
unnamed key takes the same head (``parallel_transport`` where the link file
exists, else ``dft_velocity`` from ``dipole.h5``, else ``off``).  A named
value is kept, and the bare_transverse metal admits its own default.
"""
from dataclasses import replace

from gw.gw_config import LorraxConfig, uses_metal_direct_drude_head

_SC = """[cohsex]
sys_dim = 3
compute_mode = tangential
qp_solver = self_consistent
number_bands = 30
nval = 4
ncond = 4
"""


def _head(tmp_path, extra):
    path = tmp_path / "gw.in"
    path.write_text(_SC + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform="cpu",
                                       print_fn=lambda *_a: None)


def test_unnamed_head_update_is_the_same_on_both_representations(tmp_path):
    decks = ("bispinor = false\n", "bispinor = true\nbispinor_gw = bare_transverse\n")
    for deck in decks:
        assert _head(tmp_path, deck).sc.head_update == "off"     # no file at all
    (tmp_path / "dipole.h5").write_bytes(b"")
    assert [_head(tmp_path, d).sc.head_update for d in decks] == ["dft_velocity"] * 2
    (tmp_path / "parallel_transport.h5").write_bytes(b"")
    resolved = [_head(tmp_path, d) for d in decks]
    assert [c.sc.head_update for c in resolved] == ["parallel_transport"] * 2
    assert _head(tmp_path, decks[1] + "sc_head_update = off\n").sc.head_update == "off"
    # A bare_transverse metal (Fermi-Dirac width given) takes dft_velocity
    # unnamed; parallel_transport only when named, and then it is admitted.
    metal = "occ_smearing_width_ry = 0.01\n"
    assert _head(tmp_path, decks[0] + metal).sc.head_update == "parallel_transport"
    assert _head(tmp_path, decks[1] + metal).sc.head_update == "dft_velocity"
    named = _head(tmp_path, decks[1] + metal + "sc_head_update = parallel_transport\n")
    assert named.sc.head_update == "parallel_transport"
    assert uses_metal_direct_drude_head(named)
    assert uses_metal_direct_drude_head(
        replace(resolved[1], sc=replace(resolved[1].sc, head_update="dft_velocity")))
