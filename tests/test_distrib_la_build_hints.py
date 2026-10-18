"""The rebuild command a missing handler's refusal names is the one that builds its library.

``loader.probe_target`` on a loaded leg that does not export a target's handler returns
``lxkit.probe.missing_symbol``'s reason with ``loader.target_build_hint``: ROCM's ELPA eigh names
``config/frontier/build_elpa_hip.sh`` (then ``build_ffi_rocm.sh``), the SLATE targets keep
``config/frontier/build_slate_hip.sh``; CUDA has no per-target override.  CPU only: the leg is a
stand-in object exporting no handler symbol, so no device and no FFI library are needed.
"""
import pytest

from distrib_la import loader

ELPA = "config/frontier/build_elpa_hip.sh"
SLATE = "config/frontier/build_slate_hip.sh"
FFI_ROCM = "config/frontier/build_ffi_rocm.sh"


class _LegWithoutHandlers:
    """A loaded ROCM leg built without any handler: every ``hasattr(lib, sym)`` is False."""


@pytest.fixture
def rocm_leg_without_handlers(monkeypatch):
    monkeypatch.setattr(loader, "get_lib", lambda platform: _LegWithoutHandlers())
    monkeypatch.setattr(loader._native, "borrowed_reason", lambda platform, target: "")
    monkeypatch.setitem(loader._LIB_PATHS, "ROCM", "/stand-in/liblorrax_ffi_rocm.so")


@pytest.mark.parametrize("target", sorted(loader._ROCM_ELPA_SYMBOLS))
def test_missing_elpa_refusal_names_the_elpa_build_script(rocm_leg_without_handlers, target):
    usable, why = loader.probe_target(target, "ROCM")
    assert not usable
    assert f"built WITHOUT the {target} handler" in why
    assert ELPA in why and FFI_ROCM in why and SLATE not in why
    assert why.index(ELPA) < why.index(FFI_ROCM)          # ELPA first, then the leg


SLATE_TARGETS = sorted(t for t in loader._ROCM_TARGET_SYMBOLS if t.startswith("lorrax_slate_"))


def test_rocm_has_slate_targets():
    assert "lorrax_slate_eigh" in SLATE_TARGETS
    assert set(loader._ROCM_SLATE_LU_SYMBOLS) <= set(SLATE_TARGETS)


@pytest.mark.parametrize("target", SLATE_TARGETS)
def test_missing_slate_refusal_names_the_slate_build_script(rocm_leg_without_handlers, target):
    usable, why = loader.probe_target(target, "ROCM")
    assert not usable
    assert f"built WITHOUT the {target} handler" in why
    assert SLATE in why and FFI_ROCM in why and ELPA not in why


def test_only_rocm_elpa_targets_are_overridden():
    assert set(loader._TARGET_BUILD_HINTS) == {("ROCM", t) for t in loader._ROCM_ELPA_SYMBOLS}
    for platform in ("CUDA", "cpu"):
        for t in loader._TARGETS[platform]:
            assert loader.target_build_hint(platform, t) == loader._BUILD_HINTS[platform]
