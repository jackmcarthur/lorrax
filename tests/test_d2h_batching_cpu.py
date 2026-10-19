"""Host reads batched or kept on device (CPU, seconds).

A checked solve's status costs every rank one read of its failure flag, and its
error table is read only when that flag is set (one attempt: the refusal); the bank's reciprocity
census is read once at the receipt; the store's verdicts are read once per call.
"""
import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _prints(status, n):
    from distrib_la import _result_check as rc
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc._notices((status[0], np.asarray(status[1])), "eigh", n, np.complex128, "test")
    return err.getvalue()


@pytest.mark.parametrize("rows,named", [
    ([1e-16, 1e-16], 0),           # accepted: nothing printed
    ([1e-3, 1e-16], 1),            # failed its check: one refusal per failed matrix
    ([np.inf, np.inf], 1),
])
def test_refusal_names_each_failed_matrix(rows, named):
    from distrib_la import _result_check as rc
    table = jnp.asarray([rows, [1e-16, 1e-16]])      # a stack of two matrices, the second accepted
    printed = _prints((jnp.asarray(bool(named)), table), 64)
    assert printed.count("GATE distrib_la_result_check") == named


def test_status_is_read_once_per_array(monkeypatch):
    """One read of the failure flag per status array; the table is never read while it passed."""
    from distrib_la import _result_check as rc
    rc._FAILED.clear()
    reads = []
    real = jax.device_get
    monkeypatch.setattr(jax, "device_get", lambda x: reads.append(1) or real(x))
    status = (jnp.asarray(False), jnp.full((2,), 1e-16))
    for _ in range(3):
        assert rc._flag(status) is False
    assert reads == [1]


def test_raise_if_failed_reads_the_table_only_on_failure(monkeypatch):
    from distrib_la import _result_check as rc
    rc._FAILED.clear()
    rc.raise_if_failed((jnp.asarray(False), jnp.full((2,), np.inf)), "eigh", 64, np.complex128, "ok")
    with pytest.raises(ValueError, match="GATE distrib_la_result_check: eigh n=64 at bad"):
        rc.raise_if_failed((jnp.asarray(True), jnp.full((2,), 1e-3)), "eigh", 64, np.complex128, "bad")


def test_reciprocity_rows_resolve_in_order_from_one_read():
    from gw.response_bank import _resolve_reciprocity
    receipt = {"_reciprocity_pending": [
        (jnp.asarray([0.5, 0.25]), [(dict(q_parent=0, q_full=0, u_ry=1.0), 1)]),
        (jnp.asarray([0.125]), [(dict(q_parent=3, q_full=7, u_ry=2.0), 0)])]}
    _resolve_reciprocity(receipt)
    assert "_reciprocity_pending" not in receipt
    assert receipt["reciprocity_real_supports"] == [
        dict(q_parent=0, q_full=0, u_ry=1.0, defect=0.25),
        dict(q_parent=3, q_full=7, u_ry=2.0, defect=0.125)]
    _resolve_reciprocity(receipt)          # idempotent without pending rows


def test_store_verdicts_refuse_the_first_bad_entry():
    # The bank's finite verdicts moved to the writer (SlabIO write_slab
    # require_finite) and the resident bank's deferred flags
    # (tests/test_resident_bank_finite_cpu.py); the sentinel verdicts stay here.
    from file_io import shared_pole_store as sps
    sps._refuse_invalid([True, jnp.asarray(True)])
    with pytest.raises(Exception, match="invalid inactive sentinel"):
        sps._refuse_invalid([True, jnp.asarray(True), jnp.asarray(False)])
    with pytest.raises(Exception, match="invalid inactive sentinel"):
        sps._refuse_invalid([False])
