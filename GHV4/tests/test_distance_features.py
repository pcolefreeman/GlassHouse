"""Tests for distance_features — snap CSI → ML feature vector."""
import numpy as np
import pytest
from ghv4.distance_features import (
    snap_csi_to_complex,
    extract_snap_features,
    pair_features,
    FEATURE_NAMES,
)
from ghv4.config import NULL_SUBCARRIER_INDICES, VALID_SUBCARRIER_COUNT


def _make_csi_bytes(n_subcarriers=128):
    """Generate synthetic int8 I/Q CSI bytes (imag, real per subcarrier)."""
    rng = np.random.default_rng(42)
    iq = rng.integers(-127, 127, size=n_subcarriers * 2, dtype=np.int8)
    return bytes(iq)


class TestSnapCsiToComplex:
    def test_returns_121_complex(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        assert csi.shape == (VALID_SUBCARRIER_COUNT,)
        assert np.iscomplexobj(csi)

    def test_null_subcarriers_excluded(self):
        # All-zero CSI → all zeros, but length still 121
        csi = snap_csi_to_complex(bytes(256))
        assert csi.shape == (VALID_SUBCARRIER_COUNT,)
        assert np.all(csi == 0)

    def test_rejects_short_buffer(self):
        assert snap_csi_to_complex(bytes(100)) is None

    def test_byte_order_imag_real(self):
        """ESP32 CSI: byte[0]=imag, byte[1]=real for subcarrier 0."""
        buf = bytes([5, 10] + [0] * 254)  # sub0: imag=5, real=10
        csi = snap_csi_to_complex(buf)
        # sub0 is in NULL set, so check sub3 (first valid after nulls)
        buf2 = bytes([0] * 6 + [7, 3] + [0] * 248)  # sub3: imag=7, real=3
        csi2 = snap_csi_to_complex(buf2)
        # sub3 is index 0 in valid array (subs 0,1,2 are null)
        assert csi2[0] == complex(3, 7)


class TestExtractSnapFeatures:
    def test_output_length(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        feats = extract_snap_features(csi)
        # 121 amp_norm + 121 phase = 242
        assert len(feats) == VALID_SUBCARRIER_COUNT * 2

    def test_amp_norm_range(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        feats = extract_snap_features(csi)
        amp_norm = feats[:VALID_SUBCARRIER_COUNT]
        assert all(0.0 <= v <= 1.0 for v in amp_norm)

    def test_phase_range(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        feats = extract_snap_features(csi)
        phase = feats[VALID_SUBCARRIER_COUNT:]
        # Scaled by pi → range [-1, 1]
        assert all(-1.0 <= v <= 1.0 for v in phase)


class TestPairFeatures:
    def test_output_length_484(self):
        fwd_csi = snap_csi_to_complex(_make_csi_bytes(128))
        rev_csi = snap_csi_to_complex(
            bytes(np.random.default_rng(99).integers(-127, 127, 256, dtype=np.int8))
        )
        vec = pair_features(fwd_csi, rev_csi)
        assert len(vec) == 484  # 242 fwd + 242 rev

    def test_feature_names_match_length(self):
        assert len(FEATURE_NAMES) == 484


class TestFeatureNames:
    def test_prefix_structure(self):
        # First 121 should be fwd_amp_norm_*, next 121 fwd_phase_*
        assert FEATURE_NAMES[0].startswith("fwd_amp_norm_")
        assert FEATURE_NAMES[121].startswith("fwd_phase_")
        assert FEATURE_NAMES[242].startswith("rev_amp_norm_")
        assert FEATURE_NAMES[363].startswith("rev_phase_")
