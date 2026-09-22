"""The remat give-up line is parsed and reported once, not once per rank."""
from runtime.pjrt_log_filter import (
    parse_rematerialization_notice, rematerialization_banner)

REAL = (b"W0921 20:33:15.157615   53186 hlo_rematerialization.cc:3233] Can't reduce "
        b"memory use below 60.51GiB (64969051109 bytes) by rematerialization; only "
        b"reduced to 65.54GiB (70376241720 bytes), down from 65.54GiB "
        b"(70376241720 bytes) originally\n")


def test_parses_the_real_production_line():
    assert parse_rematerialization_notice(REAL) == (
        64969051109, 70376241720, 70376241720)


def test_ignores_other_stderr():
    assert parse_rematerialization_notice(b"W0921 pjrt_executable.cc:1] other\n") is None


def test_banner_names_the_zero_reduction_and_the_flag():
    text = rematerialization_banner(*parse_rematerialization_notice(REAL)).decode()
    assert "freed NOTHING" in text
    assert "--xla_disable_hlo_passes=rematerialization" in text
