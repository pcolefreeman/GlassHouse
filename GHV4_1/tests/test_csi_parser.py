"""Unit tests for CSI parsing and amplitude computation.

All tests use synthetic data — no hardware or serial port required.
"""

from __future__ import annotations

import math
import sys
import os

# Ensure the python/ directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from serial_csi_reader import (
    parse_csi_line,
    compute_amplitudes,
    format_amplitude_summary,
)


# ---------------------------------------------------------------------------
# parse_csi_line tests
# ---------------------------------------------------------------------------


def test_parse_valid_csi_line():
    """Construct a valid CSV line and verify all parsed fields."""
    line = "CSI_DATA,42,24:6F:28:AA:BB:CC,-55,8,3 -4 10 20 -1 0 7 -8"
    result = parse_csi_line(line)

    assert result is not None
    assert result["seq"] == 42
    assert result["mac"] == "24:6F:28:AA:BB:CC"
    assert result["rssi"] == -55
    assert result["data_len"] == 8
    assert result["raw_bytes"] == [3, -4, 10, 20, -1, 0, 7, -8]


def test_parse_ignores_non_csi_lines():
    """Boot messages, empty lines, and garbage should return None."""
    assert parse_csi_line("") is None
    assert parse_csi_line("=== CSI Receiver Starting ===") is None
    assert parse_csi_line("WiFi channel set to 11: OK") is None
    assert parse_csi_line("random noise garbage") is None
    assert parse_csi_line("CSI_DATA,bad") is None  # too few fields
    assert parse_csi_line("\r\n") is None


# ---------------------------------------------------------------------------
# compute_amplitudes tests
# ---------------------------------------------------------------------------


def test_compute_amplitudes_basic():
    """Known byte pairs -> verify amplitude = sqrt(imag^2 + real^2)."""
    # Pairs: (imag=3, real=4) -> 5.0,  (imag=0, real=1) -> 1.0
    raw = [3, 4, 0, 1]
    amps = compute_amplitudes(raw, skip_first_word=False)

    assert len(amps) == 2
    assert math.isclose(amps[0], 5.0)
    assert math.isclose(amps[1], 1.0)


def test_compute_amplitudes_skips_first_word():
    """When skip_first_word=True, first 4 bytes are discarded."""
    # First 4 bytes: garbage.  Remaining: (imag=6, real=8) -> 10.0
    raw = [99, 99, 99, 99, 6, 8]
    amps = compute_amplitudes(raw, skip_first_word=True)

    assert len(amps) == 1
    assert math.isclose(amps[0], 10.0)


def test_compute_amplitudes_no_skip():
    """When skip_first_word=False, all bytes are processed."""
    raw = [99, 99, 99, 99, 6, 8]
    amps = compute_amplitudes(raw, skip_first_word=False)

    # 3 pairs: (99,99), (99,99), (6,8)
    assert len(amps) == 3
    assert math.isclose(amps[0], math.sqrt(99**2 + 99**2))
    assert math.isclose(amps[2], 10.0)


def test_compute_amplitudes_signed_bytes():
    """Values > 127 should be interpreted as negative signed int8.

    The firmware prints signed int8_t values directly, but if raw
    unsigned bytes are ever fed in (0-255 range), _to_signed8 converts
    them: 255 -> -1, 200 -> -56, 128 -> -128.
    """
    # imag=200 -> -56, real=200 -> -56 -> sqrt(56^2 + 56^2) = 56*sqrt(2)
    raw = [200, 200]
    amps = compute_amplitudes(raw, skip_first_word=False)

    assert len(amps) == 1
    expected = math.sqrt(56**2 + 56**2)
    assert math.isclose(amps[0], expected, rel_tol=1e-9)

    # Also test with already-signed values (negative ints from firmware)
    raw_signed = [-56, -56]
    amps_signed = compute_amplitudes(raw_signed, skip_first_word=False)
    assert math.isclose(amps_signed[0], expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# format_amplitude_summary tests
# ---------------------------------------------------------------------------


def test_format_amplitude_summary():
    """Summary string should contain mean and max values."""
    amps = [5.0, 10.0, 15.0, 20.0]
    summary = format_amplitude_summary(amps)

    assert "mean=" in summary
    assert "max=" in summary
    # Mean = 12.5, Max = 20.0
    assert "12.5" in summary
    assert "20.0" in summary


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


def test_full_pipeline():
    """End-to-end: raw CSV line -> parse -> compute -> format.

    Verifies the full chain produces sensible output.
    """
    # Simulate a real CSI line with 8 bytes (4 subcarriers)
    # Pairs: (3,4)=5  (0,5)=5  (-3,4)=5  (6,8)=10
    line = "CSI_DATA,100,24:6F:28:AA:BB:CC,-42,8,3 4 0 5 -3 4 6 8"

    parsed = parse_csi_line(line)
    assert parsed is not None
    assert parsed["seq"] == 100

    amps = compute_amplitudes(parsed["raw_bytes"])
    assert len(amps) == 4
    assert math.isclose(amps[0], 5.0)
    assert math.isclose(amps[1], 5.0)
    assert math.isclose(amps[2], 5.0)
    assert math.isclose(amps[3], 10.0)

    summary = format_amplitude_summary(amps)
    assert "mean=" in summary
    assert "max=" in summary
    # Mean = 6.25, max = 10.0
    assert "6.2" in summary
    assert "10.0" in summary


# ---------------------------------------------------------------------------
# S02 multi-link format tests
# ---------------------------------------------------------------------------


def test_parse_multi_link_csi_line():
    """Valid S02 format line with all fields including tx_node, rx_node, link_id."""
    line = "CSI_DATA,7,A,B,AB,-48,8,3 -4 10 20 -1 0 7 -8"
    result = parse_csi_line(line)

    assert result is not None
    assert result["seq"] == 7
    assert result["tx_node"] == "A"
    assert result["rx_node"] == "B"
    assert result["link_id"] == "AB"
    assert result["rssi"] == -48
    assert result["data_len"] == 8
    assert result["raw_bytes"] == [3, -4, 10, 20, -1, 0, 7, -8]
    # S02 format must NOT have a 'mac' key
    assert "mac" not in result


def test_parse_multi_link_all_link_ids():
    """All 6 link IDs (AB, AC, AD, BC, BD, CD) parse correctly."""
    links = [
        ("A", "B", "AB"),
        ("A", "C", "AC"),
        ("A", "D", "AD"),
        ("B", "C", "BC"),
        ("B", "D", "BD"),
        ("C", "D", "CD"),
    ]
    for tx, rx, link_id in links:
        line = f"CSI_DATA,1,{tx},{rx},{link_id},-50,4,1 2 3 4"
        result = parse_csi_line(line)
        assert result is not None, f"Failed to parse link {link_id}"
        assert result["tx_node"] == tx
        assert result["rx_node"] == rx
        assert result["link_id"] == link_id
        assert result["rssi"] == -50
        assert result["data_len"] == 4
        assert result["raw_bytes"] == [1, 2, 3, 4]


def test_parse_multi_link_preserves_s01_compat():
    """Existing S01 format lines still parse correctly after the S02 changes.

    Re-asserts the same contract as test_parse_valid_csi_line — S01 lines
    with a MAC address in field[2] must continue to produce 'mac' key.
    """
    line = "CSI_DATA,42,24:6F:28:AA:BB:CC,-55,8,3 -4 10 20 -1 0 7 -8"
    result = parse_csi_line(line)

    assert result is not None
    assert result["seq"] == 42
    assert result["mac"] == "24:6F:28:AA:BB:CC"
    assert result["rssi"] == -55
    assert result["data_len"] == 8
    assert result["raw_bytes"] == [3, -4, 10, 20, -1, 0, 7, -8]
    # S01 format must NOT have S02-specific keys
    assert "tx_node" not in result
    assert "rx_node" not in result
    assert "link_id" not in result


def test_parse_multi_link_amplitude_pipeline():
    """Full pipeline: S02 CSV -> parse -> compute_amplitudes -> verify."""
    # Pairs: (imag=3, real=4) = 5.0, (imag=0, real=5) = 5.0
    line = "CSI_DATA,200,C,D,CD,-62,4,3 4 0 5"
    result = parse_csi_line(line)

    assert result is not None
    assert result["link_id"] == "CD"
    assert result["seq"] == 200

    amps = compute_amplitudes(result["raw_bytes"])
    assert len(amps) == 2
    assert math.isclose(amps[0], 5.0)
    assert math.isclose(amps[1], 5.0)

    summary = format_amplitude_summary(amps)
    assert "mean=5.0" in summary
    assert "max=5.0" in summary


def test_parse_multi_link_edge_cases():
    """Edge cases: empty CSI bytes, very large seq, negative rssi."""
    # Empty CSI bytes
    line_empty = "CSI_DATA,0,A,C,AC,-90,0,"
    result = parse_csi_line(line_empty)
    assert result is not None
    assert result["raw_bytes"] == []
    assert result["data_len"] == 0
    assert result["link_id"] == "AC"

    # Very large seq number
    line_big_seq = "CSI_DATA,999999999,B,D,BD,-30,2,10 20"
    result2 = parse_csi_line(line_big_seq)
    assert result2 is not None
    assert result2["seq"] == 999999999
    assert result2["link_id"] == "BD"

    # Very negative rssi
    line_neg_rssi = "CSI_DATA,5,A,D,AD,-127,4,1 2 3 4"
    result3 = parse_csi_line(line_neg_rssi)
    assert result3 is not None
    assert result3["rssi"] == -127
    assert result3["link_id"] == "AD"
