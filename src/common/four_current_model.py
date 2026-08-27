"""Names shared by four-current configuration and artifact provenance."""

PAULI_REFERENCE_BARE_TRANSVERSE_MODEL = "pauli_reference_bare_transverse"


def is_pauli_reference_model(value) -> bool:
    """Recognize the mixed Pauli-charge/raw4-current model selector."""
    return str(getattr(value, "value", value)).strip().lower() == (
        PAULI_REFERENCE_BARE_TRANSVERSE_MODEL)


RAW_KINETIC_BALANCE_CHARGE_REPRESENTATION = (
    "raw_kinetic_balance_identity_charge_v1")
PAULI_TWO_SPINOR_CHARGE_REPRESENTATION = (
    "qe_normalized_pauli_two_spinor_charge_v1")
SOURCE_WFN_CHARGE_REPRESENTATION = "source_wfn_normalized_charge_v1"
RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION = (
    "raw_kinetic_balance_alpha_spatial_current_v1")
