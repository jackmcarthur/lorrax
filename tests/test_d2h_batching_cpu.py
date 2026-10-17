"""Host reads batched or kept on device (CPU, seconds).

A checked solve's status costs rank 0 one read (failure and notice bits together),
and its error table is read only when a notice prints; the bank's reciprocity
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
        rc._notices((status[0], np.asarray(status[1])), "eigh", n, np.complex128, "test", final=True)
    return bool(err.getvalue())


@pytest.mark.parametrize("rows", [
    [[1e-16, 1e-16], [np.nan, np.nan]],          # accepted first attempt: no notice
    [[1e-3, 1e-16], [1e-16, 1e-16]],             # first attempt failed, retry passed: notice
    [[np.nan, np.nan], [np.nan, np.nan]],        # nothing ran
    [[np.inf, np.inf], [1e-3, 1e-3]],            # every attempt failed: refusal
])
def test_notice_bit_matches_the_printed_notices(rows):
    from distrib_la import _result_check as rc
    rc._FAILED.clear(); rc._NOTICE.clear()
    n = 64
    table = jnp.asarray([rows, rows[::-1]])      # a stack of two matrices
    failed = jnp.asarray(bool(np.all(np.asarray(rows)[-1] > rc.accept(n))))
    status = (failed, table)
    got_failed, notice = rc._status(status, n, np.complex128)
    assert notice == _prints(status, n)          # the host table takes the original path
    assert got_failed == bool(failed)


def test_status_is_read_once_per_array(monkeypatch):
    from distrib_la import _result_check as rc
    rc._FAILED.clear(); rc._NOTICE.clear()
    reads = []
    real = jax.device_get
    monkeypatch.setattr(jax, "device_get", lambda x: reads.append(1) or real(x))
    status = (jnp.asarray(False), jnp.full((2, 2), 1e-16))
    for _ in range(3):
        assert rc._flag(status, 64, np.complex128) is False
    assert rc._flag(status) is False
    assert reads == []          # the packed word is read with np.asarray, once, not device_get
    other = (status[0], jnp.full((2, 2), 1.0))       # same flag, another table
    assert rc._status(other, 64, np.complex128)[1] is True


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


def test_notices_without_a_read_word_still_read_the_table():
    """No memo (no _flag/_read_status before): the table is read and the notice prints."""
    from distrib_la import _result_check as rc
    rc._FAILED.clear(); rc._NOTICE.clear()
    status = (jnp.asarray(False), jnp.asarray([[1e-3, 1e-3], [1e-16, 1e-16]]))
    assert _prints(status, 64)


def test_one_word_covers_every_table_of_a_program():
    from distrib_la import _result_check as rc
    rc._FAILED.clear(); rc._NOTICE.clear()
    quiet = jnp.asarray([[1e-16, 1e-16], [np.nan, np.nan]])
    loud = jnp.asarray([[1e-3, 1e-3], [1e-16, 1e-16]])
    failed = jnp.asarray(False)
    assert rc._read_status(failed, (quiet, loud), [(64, np.complex128)] * 2) is False
    assert rc._memo(rc._NOTICE, quiet) is False and rc._memo(rc._NOTICE, loud) is True
    assert rc._flag((failed, (quiet, loud))) is False          # memo: no second read
    assert not _prints((failed, quiet), 64) and _prints((failed, loud), 64)
