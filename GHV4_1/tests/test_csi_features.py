"""Unit tests for CSI feature extraction — subcarrier selection and turbulence.

All tests use synthetic data — no hardware or serial port required.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# Ensure the python/ directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from csi_features import (
    SELECTED_SUBCARRIERS,
    _EPSILON,
    compute_turbulence,
    select_subcarriers,
)
from serial_csi_reader import compute_amplitudes, parse_csi_line


# ---------------------------------------------------------------------------
# select_subcarriers tests
# ---------------------------------------------------------------------------


class TestSelectSubcarriers:
    """Tests for subcarrier selection from full amplitude arrays."""

    def test_selects_correct_indices(self):
        """Values at the selected indices should be returned in order."""
        # Create 64 amplitudes where value == index for easy verification
        amplitudes = [float(i) for i in range(64)]
        result = select_subcarriers(amplitudes)

        assert len(result) == len(SELECTED_SUBCARRIERS)
        for i, idx in enumerate(SELECTED_SUBCARRIERS):
            assert result[i] == float(idx), (
                f"Index {i}: expected {float(idx)}, got {result[i]}"
            )

    def test_returns_numpy_array(self):
        """Return type must be a numpy float64 array."""
        amplitudes = [1.0] * 64
        result = select_subcarriers(amplitudes)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_custom_indices(self):
        """Custom index list overrides the default subcarrier set."""
        amplitudes = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = select_subcarriers(amplitudes, indices=[0, 2, 4])

        np.testing.assert_array_equal(result, [10.0, 30.0, 50.0])

    def test_out_of_range_indices_skipped(self):
        """Indices beyond amplitude array length are silently dropped."""
        amplitudes = [1.0, 2.0, 3.0]  # only 3 values
        result = select_subcarriers(amplitudes, indices=[0, 1, 2, 99, 100])

        assert len(result) == 3
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_empty_amplitudes(self):
        """Empty amplitude list produces empty array."""
        result = select_subcarriers([], indices=[0, 1, 2])
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_all_indices_out_of_range(self):
        """If all indices are out of range, return empty array."""
        amplitudes = [1.0, 2.0]
        result = select_subcarriers(amplitudes, indices=[50, 51, 52])
        assert len(result) == 0

    def test_default_indices_match_constant(self):
        """Default selection uses the SELECTED_SUBCARRIERS constant."""
        amplitudes = [float(i) for i in range(64)]
        result_default = select_subcarriers(amplitudes)
        result_explicit = select_subcarriers(amplitudes, indices=SELECTED_SUBCARRIERS)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_short_amplitude_array(self):
        """Firmware may produce fewer than 64 subcarriers — handle gracefully."""
        # Only 20 subcarriers: indices 20+ should be dropped from default set
        amplitudes = [5.0] * 20
        result = select_subcarriers(amplitudes)

        # From SELECTED_SUBCARRIERS, only [12, 14, 16, 18] are < 20
        expected_count = sum(1 for idx in SELECTED_SUBCARRIERS if idx < 20)
        assert len(result) == expected_count
        assert all(v == 5.0 for v in result)

    def test_selected_subcarriers_are_in_valid_range(self):
        """Verify the constant contains only valid subcarrier indices.

        Valid range: 11–51, excluding DC at 32.  Guard bands: 0–10, 53–63.
        """
        for idx in SELECTED_SUBCARRIERS:
            assert 11 <= idx <= 52, f"Index {idx} outside valid range 11–52"
            assert idx != 32, f"Index 32 (DC null) should not be selected"

    def test_selected_subcarriers_sorted(self):
        """The constant should be sorted for deterministic extraction."""
        assert SELECTED_SUBCARRIERS == sorted(SELECTED_SUBCARRIERS)

    def test_selected_subcarriers_count(self):
        """ESPectre defaults use 12 subcarrier indices."""
        assert len(SELECTED_SUBCARRIERS) == 12


# ---------------------------------------------------------------------------
# compute_turbulence tests
# ---------------------------------------------------------------------------


class TestComputeTurbulence:
    """Tests for CV-normalized turbulence computation."""

    def test_uniform_amplitudes_zero_turbulence(self):
        """All-equal amplitudes → std=0 → turbulence=0."""
        selected = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
        turb = compute_turbulence(selected)
        assert turb == 0.0

    def test_known_cv(self):
        """Verify CV against manually computed value.

        Values: [2, 4, 6, 8]
        Mean = 5.0
        Std = sqrt(((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4) = sqrt(5) ≈ 2.236
        CV = std / mean = sqrt(5) / 5 ≈ 0.4472
        """
        selected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
        turb = compute_turbulence(selected)

        expected_std = math.sqrt(5.0)
        expected_cv = expected_std / 5.0
        assert math.isclose(turb, expected_cv, rel_tol=1e-9)

    def test_empty_array_returns_zero(self):
        """Empty input → 0.0, not an error."""
        turb = compute_turbulence(np.array([], dtype=np.float64))
        assert turb == 0.0

    def test_single_element_zero_turbulence(self):
        """Single value → std=0 → turbulence=0."""
        turb = compute_turbulence(np.array([42.0], dtype=np.float64))
        assert turb == 0.0

    def test_zero_mean_guard(self):
        """All-zero amplitudes → mean < epsilon → return 0.0."""
        selected = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        turb = compute_turbulence(selected)
        assert turb == 0.0

    def test_near_zero_mean_guard(self):
        """Near-zero mean (below epsilon) → return 0.0, no division error."""
        tiny = _EPSILON / 10.0
        selected = np.array([tiny, tiny, tiny], dtype=np.float64)
        turb = compute_turbulence(selected)
        assert turb == 0.0

    def test_above_epsilon_mean(self):
        """Mean just above epsilon should compute normally, not return 0."""
        val = _EPSILON * 100  # well above epsilon but still small
        selected = np.array([val, val * 2, val * 3], dtype=np.float64)
        turb = compute_turbulence(selected)
        assert turb > 0.0  # non-uniform values should produce positive CV

    def test_returns_float(self):
        """Return type must be a plain Python float."""
        selected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        turb = compute_turbulence(selected)
        assert isinstance(turb, float)

    def test_turbulence_nonnegative(self):
        """CV is always non-negative (std and mean are both non-negative)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            vals = rng.uniform(0.1, 100.0, size=12)
            turb = compute_turbulence(vals)
            assert turb >= 0.0

    def test_high_variation_high_turbulence(self):
        """Wide spread in amplitudes should produce higher turbulence
        than narrow spread."""
        narrow = np.array([9.0, 10.0, 11.0, 10.0], dtype=np.float64)
        wide = np.array([1.0, 20.0, 3.0, 18.0], dtype=np.float64)

        turb_narrow = compute_turbulence(narrow)
        turb_wide = compute_turbulence(wide)

        assert turb_wide > turb_narrow


# ---------------------------------------------------------------------------
# Integration: parser → amplitudes → select → turbulence
# ---------------------------------------------------------------------------


class TestFullFeaturePipeline:
    """End-to-end: CSI CSV line → parse → amplitudes → select → turbulence."""

    def _make_csi_line(self, byte_values: list[int], link_id: str = "AB") -> str:
        """Build a synthetic S02-format CSI line."""
        tx, rx = link_id[0], link_id[1]
        bytes_str = " ".join(str(b) for b in byte_values)
        return f"CSI_DATA,1,{tx},{rx},{link_id},-50,{len(byte_values)},{bytes_str}"

    def _make_64_subcarrier_bytes(self, amplitude: float = 10.0) -> list[int]:
        """Generate 128 raw bytes (64 subcarrier pairs) with uniform amplitude.

        Sets real=round(amplitude), imag=0 for every subcarrier.
        Amplitude = sqrt(0² + real²) = |real|.
        """
        real = int(round(amplitude))
        # Clamp to signed int8 range
        real = max(-128, min(127, real))
        pairs: list[int] = []
        for _ in range(64):
            pairs.extend([0, real])  # [imag, real]
        return pairs

    def test_pipeline_produces_turbulence(self):
        """Full pipeline on uniform data should produce near-zero turbulence."""
        raw_bytes = self._make_64_subcarrier_bytes(amplitude=15.0)
        line = self._make_csi_line(raw_bytes)

        parsed = parse_csi_line(line)
        assert parsed is not None

        amps = compute_amplitudes(parsed["raw_bytes"])
        assert len(amps) == 64

        selected = select_subcarriers(amps)
        assert len(selected) == 12  # all 12 default indices are within range

        turb = compute_turbulence(selected)
        # Uniform amplitudes → turbulence ≈ 0
        assert math.isclose(turb, 0.0, abs_tol=1e-9)

    def test_pipeline_varied_amplitudes(self):
        """Non-uniform amplitudes should produce positive turbulence."""
        # Create bytes with alternating amplitudes: subcarrier i gets
        # amplitude (5 + i%10), producing variation across selected indices
        pairs: list[int] = []
        for i in range(64):
            real = 5 + (i % 10)
            pairs.extend([0, real])  # imag=0, real varies

        line = self._make_csi_line(pairs)
        parsed = parse_csi_line(line)
        assert parsed is not None

        amps = compute_amplitudes(parsed["raw_bytes"])
        selected = select_subcarriers(amps)
        turb = compute_turbulence(selected)

        assert turb > 0.0, "Non-uniform amplitudes should produce positive turbulence"

    def test_pipeline_s01_format(self):
        """Pipeline also works with S01 (legacy) format lines."""
        raw_bytes = self._make_64_subcarrier_bytes(amplitude=20.0)
        bytes_str = " ".join(str(b) for b in raw_bytes)
        line = f"CSI_DATA,100,AA:BB:CC:DD:EE:FF,-42,{len(raw_bytes)},{bytes_str}"

        parsed = parse_csi_line(line)
        assert parsed is not None
        assert "mac" in parsed

        amps = compute_amplitudes(parsed["raw_bytes"])
        selected = select_subcarriers(amps)
        turb = compute_turbulence(selected)

        assert math.isclose(turb, 0.0, abs_tol=1e-9)
