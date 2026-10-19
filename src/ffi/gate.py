"""LORRAX's binding of :mod:`lxkit.gate`: the same resolver, probed by ``ffi_loader``.

Every name is lxkit's own object (one announcement set, one platform key,
one grammar); :class:`Gate` only defaults its probe to
``ffi.common.ffi_loader.probe_target``.  Contract:
``docs/dev/ffi_gate_contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import lxkit as _lx

MODE_SPELLINGS = _lx.MODE_SPELLINGS
MODE_HELP = _lx.MODE_HELP
FFI_PLATFORM_MAP = _lx.FFI_PLATFORM_MAP
rank_id = _lx.rank_id
rank0 = _lx.rank0
announce_once = _lx.announce_once
reset_gate_state = _lx.reset_gate_state
device_vendor = _lx.device_vendor
mesh_ffi_platform = _lx.mesh_ffi_platform
platform_from_env = _lx.platform_from_env
dial_key = _lx.dial_key

__all__ = ["Gate", "MODE_SPELLINGS", "MODE_HELP", "FFI_PLATFORM_MAP",
           "rank_id", "rank0", "announce_once", "reset_gate_state",
           "device_vendor", "mesh_ffi_platform", "platform_from_env", "dial_key"]


def _probe(target: str, platform: str) -> tuple[bool, str]:
    from ffi.common import ffi_loader
    return ffi_loader.probe_target(target, platform)


@dataclass(frozen=True)
class Gate(_lx.Gate):
    """:class:`lxkit.gate.Gate` probed through LORRAX's own library loader."""

    probe: Callable[[str, str], tuple[bool, str]] | None = _probe
