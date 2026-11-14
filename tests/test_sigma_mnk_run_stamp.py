"""``sigma_mnk.h5`` carries the run-level stamps as ROOT ATTRIBUTES.

CPU, pytest.  ``write_sigma_omega_h5`` wrote the ``common.provenance``
identities (the bispinor carrier, ``bispinor_transverse_terms``) through
``SlabIO.write_attr``, which despite its name lands a small root DATASET; a
four-GPU leg found no attribute.  The writer now queues them with
``io.stamp_dataset_attrs("/", ...)``.  SlabIO needs the native tile path to
open a file, so this test drives the two pieces it relies on: the writer's
call (read from its source) and the deferred publication that lands a "/"
stamp as root attributes (``file_io._slab_io_ffi._apply_dataset_attrs``, the
rank-0 reopen both transports use), read back with h5py.  The four-GPU leg
re-checks the written ``sigma_mnk.h5``.
"""

import ast
from pathlib import Path


def test_writer_queues_run_provenance_on_the_root_group():
    src = (Path(__file__).resolve().parents[1] / "src" / "file_io"
           / "sigma_output.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "write_sigma_omega_h5")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)]
    stamps = [c for c in calls if c.func.attr == "stamp_dataset_attrs"
              and isinstance(c.args[0], ast.Constant) and c.args[0].value == "/"]
    assert len(stamps) == 1
    # No run identity goes through write_attr (a dataset) any more.
    names = {ast.unparse(c.args[0]) for c in calls if c.func.attr == "write_attr"}
    assert "key" not in names


def test_root_stamp_lands_as_root_attributes(tmp_path):
    import h5py
    from file_io._slab_io_ffi import _apply_dataset_attrs, _host_attr_value
    stamp = "hartree=off; exchange=on; semicore=excluded_bands:16"
    path = tmp_path / "sigma_mnk.h5"
    with h5py.File(path, "w") as h5:
        h5["sigma_sx_kij_ev"] = [[0.0]]
    with h5py.File(path, "a") as h5:
        _apply_dataset_attrs(h5, [("/", {
            "four_current_carrier": "pauli_charge_rkb_current_v1",
            "bispinor_transverse_terms": stamp})])
    with h5py.File(path, "r") as h5:
        got = {k: (v.decode() if isinstance(v, bytes) else str(v))
               for k, v in h5.attrs.items()}
        assert got["bispinor_transverse_terms"] == stamp
        assert got["four_current_carrier"] == "pauli_charge_rkb_current_v1"
        assert "bispinor_transverse_terms" not in h5
    assert _host_attr_value(stamp) is not None
