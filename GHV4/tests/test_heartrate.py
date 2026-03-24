"""Tests for ghv4.heartrate — CSI heart rate analysis.

NOTE: ghv4/heartrate.py does not exist yet. These tests are stubs
that will be implemented once the source module is created. See task spec
GlassHouseRepo-7407 for planned coverage.
"""
import pytest


pytestmark = pytest.mark.skip(
    reason="ghv4.heartrate module does not exist yet"
)


class TestHeartRateAnalyzerPlaceholder:
    def test_module_importable(self):
        """heartrate should be importable."""
        from ghv4 import heartrate  # noqa: F401

    def test_synthetic_heartbeat_detection(self):
        """Known 1.0 Hz periodic signal should detect ~60 BPM."""
        pass

    def test_flat_signal_no_heartbeat(self):
        """Constant signal should yield no heart rate."""
        pass

    def test_noise_only_no_heartbeat(self):
        """White noise should not produce a confident heart rate."""
        pass

    def test_multiple_frequencies_picks_strongest(self):
        """Signal with breathing + heartbeat should pick heartbeat band."""
        pass

    def test_out_of_range_frequency_rejected(self):
        """Frequency outside 0.8-2.5 Hz should be rejected."""
        pass
