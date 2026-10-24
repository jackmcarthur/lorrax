"""CPU cache identity keeps target/platform bytes and removes only device IDs."""
from common.jax_compile_cache import _cpu_target_machine


def message(key, payload):
    size, encoded = len(payload), []
    while size >= 128:
        encoded.append(128 + size % 128)
        size //= 128
    return bytes([key, *encoded, size]) + payload


TYPE = b'type.googleapis.com/xla.CpuTopologyProto'
TARGET = message(0x2a, message(0x0a, b'x86_64') + message(0x1a, b'+avx2,' * 40))
UNKNOWN_CPU = b'\x38\x05'
UNKNOWN_ANY = b'\x1d\x00\x11\x22\x33'
PLATFORM = b'\x08\x02\x12\x03cpu\x1a\x02v1\x58\x3a'


def envelope(cpu, *, type_url=TYPE, platform=PLATFORM):
    packed = message(0x0a, type_url) + UNKNOWN_ANY + message(0x12, cpu)
    return platform + message(0x4a, packed)


def test_cpu_rank_records_share_one_key_without_losing_other_fields():
    wanted = TARGET + UNKNOWN_CPU
    for device in (b'\x0a\x00', b'\x0a\x02\x10\x01',
                   b'\x0a\x04\x10\x03\x18\x02'):
        raw = device + TARGET + UNKNOWN_CPU
        assert _cpu_target_machine(raw) == wanted
        assert _cpu_target_machine(envelope(raw)) == envelope(wanted)
    # A target feature, platform version or unknown field still changes identity.
    original = _cpu_target_machine(envelope(b'\x0a\x00' + wanted))
    for changed in (envelope(b'\x0a\x00' + wanted.replace(b'avx2', b'avx3')),
                    envelope(b'\x0a\x00' + wanted, platform=PLATFORM.replace(b'v1', b'v2')),
                    envelope(b'\x0a\x00' + wanted + b'\x40\x07')):
        assert _cpu_target_machine(changed) != original


def test_unknown_or_malformed_topology_retains_jax_identity():
    raw = b'\x0a\x02\x10\x01' + TARGET
    wrapped = envelope(raw)
    for unknown in (b'', b'not a topology', b'\x0a\x00',
                    envelope(raw, type_url=b'type.googleapis.com/other.Topology'),
                    envelope(raw, platform=PLATFORM.replace(b'cpu', b'gpu')),
                    raw[:-1], wrapped[:-1], wrapped + b'\x80',
                    envelope(b'\x0a\x01\x80' + TARGET),
                    envelope(b'\x0a\x00\x2a\x01\x80'),
                    wrapped + message(0x4a, b'')):
        assert _cpu_target_machine(unknown) == unknown
