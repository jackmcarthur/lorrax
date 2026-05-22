"""Smoke test: cohsex.in schema completeness.

Pending Blitz #2 (consensus.md §#2): introduce ``src/gw/_cohsex_schema.py``
with a ``Field(name, type, default, doc)`` table covering all parser keys.

Once Blitz #2 lands:
  - Remove the ``pytest.skip`` call below.
  - Import ``_cohsex_schema`` (or wherever the schema module lands).
  - Assert ``set(parser._DEFAULTS) == set(documented_keys) | deprecated_keys``.

The ``deprecated_keys`` allow-list (per consensus Q4) should cover at least:
  ``output_file``, ``use_chunked_isdf``, ``sigma_debug_split_contrib``,
  ``write_no_head_vw``.
"""
import pytest


@pytest.mark.skip(reason="Pending Blitz #2 — cohsex schema module not yet implemented")
def test_schema_keys_match_parser_defaults():
    """set(parser._DEFAULTS) == set(documented_keys) | deprecated_keys."""
    from gw._cohsex_schema import _FIELDS, _DEPRECATED_KEYS  # noqa: F401  # type: ignore[import]
    from gw.cohsex_input import read_lorrax_input  # noqa: F401  # type: ignore[import]

    documented_keys = {f.name for f in _FIELDS}
    deprecated_keys = set(_DEPRECATED_KEYS)

    # Import the parser to get its live _DEFAULTS dict.
    import importlib
    parser_mod = importlib.import_module("gw.cohsex_input")
    parser_defaults = set(getattr(parser_mod, "_DEFAULTS", {}).keys())

    assert parser_defaults == documented_keys | deprecated_keys, (
        f"Schema mismatch.\n"
        f"  In parser but not schema: {parser_defaults - (documented_keys | deprecated_keys)}\n"
        f"  In schema but not parser: {(documented_keys | deprecated_keys) - parser_defaults}"
    )
