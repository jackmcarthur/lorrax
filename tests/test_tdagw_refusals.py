"""TD-aGW refusals carry GATE / got / want / why / fix, and the kernel writer refuses header codes by name.

CPU-only and device-free.  The model's reasons live in one table (``tdagw.model._WHY``), so a gate added
without a reason fails ``test_every_model_gate_has_a_reason``.
"""
import os
import re

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np
import pytest

from tdagw import kernel_export as KE
from tdagw import model as M

MODEL_SRC = os.path.join(os.path.dirname(M.__file__), "model.py")
FORMAT = re.compile(r"^GATE [a-z0-9_]+: got: .+; want: .+; why: .+; fix: .+")


def test_every_model_gate_has_a_reason():
    gates = set(re.findall(r'_refuse\(\s*"([a-z0-9_]+)"', open(MODEL_SRC).read()))
    assert gates, "no _refuse call found"
    assert sorted(gates - set(M._WHY)) == []
    assert sorted(set(M._WHY) - gates) == []          # no reason without its gate


def test_model_refusal_format():
    with pytest.raises(ValueError) as e:
        M._refuse("td_window_gap", "bands [0, 4) with n_occ=8", "b_lo < n_occ < b_hi", "fix it")
    assert FORMAT.match(str(e.value)) and M._WHY["td_window_gap"] in str(e.value)


@pytest.mark.parametrize("kw, gate", [(dict(sys_dim=1, nspinor=2), "tdgw_kernel_truncation"),
                                      (dict(sys_dim=2, nspinor=4), "tdgw_kernel_type")])
def test_kernel_header_codes_refuse_before_writing(tmp_path, kw, gate):
    K = np.zeros((1, 2, 2, 1, 2, 2), complex)
    path = tmp_path / "kernel.h5"
    with pytest.raises(ValueError) as e:
        KE.write_tdgw_kernel(str(path), K, kpts=np.zeros((1, 3)), ibndmin=1, ivbm=1, **kw)
    assert str(e.value).startswith(f"GATE {gate}:") and FORMAT.match(str(e.value))
    assert not path.exists()
    KE.write_tdgw_kernel(str(path), K, kpts=np.zeros((1, 3)), ibndmin=1, ivbm=1,
                         sys_dim=2, nspinor=2)            # the control writes
    assert path.exists()


@pytest.mark.parametrize("call", [
    lambda p: KE.write_tdgw_kernel(p, np.zeros((1, 2, 2, 1, 2, 3)), kpts=np.zeros((1, 3)),
                                   ibndmin=1, ivbm=1, sys_dim=2, nspinor=2),
    lambda p: KE.write_tdgw_kernel(p, np.zeros((1, 2, 2, 1, 2, 2)), kpts=np.zeros((2, 3)),
                                   ibndmin=1, ivbm=1, sys_dim=2, nspinor=2),
    lambda p: KE.tdgw_file_band_order(1, 2, 2),
])
def test_kernel_export_refusals_carry_why(tmp_path, call):
    with pytest.raises(ValueError) as e:
        call(str(tmp_path / "k.h5"))
    assert FORMAT.match(str(e.value))


@pytest.mark.parametrize("ibndmin, nb, ivbm", [(129, 4, 130), (100, 60, 130), (88, 56, 130), (1, 2, 1)])
def test_kernel_band_order_is_the_tdgw_io_definition(ibndmin, nb, ivbm):
    from tdagw.tdgw_io import kernel_file_bands
    order = KE.tdgw_file_band_order(ibndmin, nb, ivbm)
    n_v = ivbm - ibndmin + 1
    fb = kernel_file_bands(ivbm, n_v, nb - n_v)
    bands = ibndmin + np.arange(nb)
    assert np.array_equal(fb[order], bands)                      # file slot of each window band
    v = bands <= ivbm                                            # the closed form it replaced
    assert np.array_equal(order, np.where(v, ivbm - bands, n_v + bands - ivbm - 1))
