"""``GATE onsite_breit_keys``: ``onsite_breit_double_counting = paw`` needs the smooth transverse direct field.

``paw`` adds ``h^dir_AE - h^dir_PS`` to the transverse direct-field slot, on
the premise that the slot already holds the smooth ``<m|alpha.A[J]|n>``, whose
on-site part ``h^dir_PS`` models.  ``bispinor_transverse_hartree = false``
makes that smooth field an exact zero (an array, not ``None``), so the
runtime pairing check in ``gw.sigma_dispatch.onsite_breit_into_transverse``
passes and the run would subtract a pseudo direct term it does not have.
The parse gate refuses the pair by name.
"""
import pytest

from gw.gw_config import LorraxConfig

ON = ("bispinor = true\nonsite_breit = true\nonsite_breit_files = Ni.onsite_breit.json\n"
      "onsite_breit_double_counting = paw\n")


def _config(tmp_path, extra=""):
    path = tmp_path / "gw.in"
    path.write_text("[cohsex]\nsys_dim = 3\nnumber_bands = 30\nnval = 4\nncond = 4\n" + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False, runtime_platform="cpu",
                                        print_fn=lambda *_a, **_k: None)


def test_paw_with_the_smooth_direct_field_is_served(tmp_path):
    for exchange in ("paw", "off"):
        c = _config(tmp_path, ON + f"onsite_breit_exchange = {exchange}\n")
        assert c.bispinor and c.bispinor_transverse_hartree
        assert (c.onsite_breit_double_counting, c.onsite_breit_exchange) == ("paw", exchange)


def test_paw_without_the_smooth_direct_field_refuses_by_name(tmp_path):
    for exchange in ("paw", "off"):
        with pytest.raises(ValueError, match="GATE onsite_breit_keys") as err:
            _config(tmp_path, ON + f"onsite_breit_exchange = {exchange}\n"
                    "bispinor_transverse_hartree = false\n")
        text = str(err.value)
        for part in ("got:", "want:", "why:", "fix:", "bispinor_transverse_hartree = false",
                     "onsite_breit_double_counting = paw"):
            assert part in text, part


def test_the_switch_alone_is_untouched(tmp_path):
    c = _config(tmp_path, "bispinor = true\nbispinor_transverse_hartree = false\n")
    assert c.bispinor and not c.bispinor_transverse_hartree and not c.onsite_breit
