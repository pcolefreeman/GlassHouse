"""Unit tests for CSIMUSICEstimator.

Tests call private methods directly (_csi_to_complex, _music_delay,
_mdl_order, _first_peak_tau) for deterministic synchronous coverage.
The compute-worker-thread integration (collect → get_distances) is
tested via a brief sleep after collect(); this is acceptable here because
the worker does < 10 ms of work per pair.
"""
import struct
import time
import numpy as np
import pytest


# ── _csi_to_complex ──────────────────────────────────────────────────────────

def test_csi_to_complex_converts_int8_pairs():
    """ESP32 format: interleaved int8 (imag, real) per subcarrier."""
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    # 128 subcarriers × 2 bytes; first subcarrier: imag=10, real=20
    raw = np.zeros(256, dtype=np.int8)
    raw[0] = 10  # imag
    raw[1] = 20  # real
    result = CSIMUSICEstimator._csi_to_complex(raw.tobytes())
    assert result is not None
    # Subcarrier 0 is a null index — NOT in the output; check length only
    assert len(result) == 121


def test_csi_to_complex_rejects_short_input():
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    assert CSIMUSICEstimator._csi_to_complex(bytes(255)) is None  # < 256


def test_csi_to_complex_output_length():
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    raw = bytes(256)
    result = CSIMUSICEstimator._csi_to_complex(raw)
    assert result is not None
    assert len(result) == 121  # 128 - 7 null subcarriers


# ── _first_peak_tau ───────────────────────────────────────────────────────────

def test_first_peak_tau_finds_first_peak():
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    tau_grid = np.linspace(0, 100e-9, 1000)
    pseudo = np.ones(1000)
    # Insert a single sharp peak at index 237
    pseudo[237] = 100.0
    result = CSIMUSICEstimator._first_peak_tau(pseudo, tau_grid)
    assert result is not None
    assert abs(result - tau_grid[237]) < 1e-12


def test_first_peak_tau_returns_none_when_flat():
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    tau_grid = np.linspace(0, 100e-9, 100)
    pseudo = np.ones(100)  # all equal — no peak above mean
    assert CSIMUSICEstimator._first_peak_tau(pseudo, tau_grid) is None


# ── _mdl_order ────────────────────────────────────────────────────────────────

def test_mdl_order_returns_small_value_for_single_source():
    """A rank-1 signal should produce L=1 or L=2 at most."""
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    K, N = 121, 30
    # Build a rank-1 covariance matrix
    a = np.exp(-1j * 2 * np.pi * np.random.rand(K))
    R = np.outer(a, a.conj()) * 50.0 + np.eye(K) * 1.0
    eigenvalues = np.linalg.eigvalsh(R)
    L = CSIMUSICEstimator._mdl_order(eigenvalues, N, K)
    assert 1 <= L <= 3


# ── _music_delay (synthetic H with known τ) ───────────────────────────────────

def _make_synthetic_H(tau_true: float, K: int, N: int,
                      snr_db: float = 20.0) -> np.ndarray:
    """
    Build a (K, N) complex array H where each column is a(τ_true) + noise.
    H has rank ≈ 1 (single-path channel). SNR = signal_power / noise_power.
    """
    from ghv3_1.spacing_estimator import SUBCARRIER_FREQS
    a = np.exp(-1j * 2 * np.pi * SUBCARRIER_FREQS * tau_true)   # (K,)
    signal_power = 10 ** (snr_db / 10.0)
    noise_std    = 1.0
    signal_std   = np.sqrt(signal_power)
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal((K, N)) + 1j * rng.standard_normal((K, N))) * noise_std / np.sqrt(2)
    H = np.outer(a, np.ones(N)) * signal_std + noise
    return H


def test_music_delay_known_tau_7_1m():
    """
    7.1 m → τ_true ≈ 23.7 ns. MUSIC should return τ within ±3 ns (±0.9 m).
    This test is probabilistic; fixed rng seed (42) keeps it deterministic.
    """
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    from scipy.constants import c as C
    tau_true = 7.1 / C
    H = _make_synthetic_H(tau_true, K=121, N=30, snr_db=20.0)
    est = CSIMUSICEstimator.__new__(CSIMUSICEstimator)  # bypass __init__ thread start
    tau_est = est._music_delay(H)
    assert tau_est is not None
    assert abs(tau_est - tau_true) < 3e-9


def test_music_delay_returns_none_for_single_snapshot():
    """N=1 covariance matrix is rank-1 → eigendecompose degenerates, return None."""
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    H = np.ones((121, 1), dtype=complex)
    est = CSIMUSICEstimator.__new__(CSIMUSICEstimator)
    assert est._music_delay(H) is None


# ── collect / get_distances (thread integration) ──────────────────────────────

def _make_csi_bytes_for_tau(tau: float) -> bytes:
    """Produce a 256-byte CSI buffer whose complex values approximate a(τ)."""
    from ghv3_1.spacing_estimator import SUBCARRIER_FREQS, VALID_INDICES
    a = np.exp(-1j * 2 * np.pi * SUBCARRIER_FREQS * tau)
    # Fill 128-subcarrier array; null indices → (0+0j)
    full = np.zeros(128, dtype=complex)
    for i, k in enumerate(VALID_INDICES):
        full[k] = a[i]
    # Encode as int8 (imag, real) pairs
    raw = np.zeros(256, dtype=np.int8)
    scale = 60.0  # keep values within int8 range
    for k in range(128):
        raw[2 * k]     = int(np.clip(full[k].imag * scale, -127, 127))
        raw[2 * k + 1] = int(np.clip(full[k].real * scale, -127, 127))
    return raw.tobytes()


def test_get_distances_empty_initially():
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()
    assert est.get_distances() == {}


def test_reset_all_clears_state():
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    from scipy.constants import c as C
    est = CSIMUSICEstimator()
    csi = _make_csi_bytes_for_tau(7.1 / C)
    for _ in range(5):
        est.collect(1, 2, csi)
    est.reset_all()
    assert est.get_distances() == {}
    with est._lock:
        assert len(est._H) == 0
        assert len(est._enqueued) == 0


def test_collect_accumulates_snapshots():
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    from ghv3_1.config import MUSIC_MIN_SNAP as MIN_SNAP
    from scipy.constants import c as C
    est = CSIMUSICEstimator()
    csi = _make_csi_bytes_for_tau(7.1 / C)
    for i in range(MIN_SNAP - 1):
        est.collect(1, 2, csi)
    with est._lock:
        assert len(est._H.get((1, 2), [])) == MIN_SNAP - 1
    # No compute job queued yet (both directions not filled)
    assert est.get_distances() == {}


def test_collect_triggers_compute_when_both_directions_filled():
    """
    After both (reporter=1,peer=2) and (reporter=2,peer=1) have >= MIN_SNAP
    snapshots, a compute job is enqueued and get_distances() eventually
    returns a positive distance for pair '1-2'.
    """
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    from ghv3_1.config import MUSIC_MIN_SNAP as MIN_SNAP
    from scipy.constants import c as C
    est = CSIMUSICEstimator()
    tau = 7.1 / C
    csi = _make_csi_bytes_for_tau(tau)
    for _ in range(MIN_SNAP):
        est.collect(1, 2, csi)
        est.collect(2, 1, csi)
    # Wait for worker thread to complete (~50 ms is generous for 121×121 eig)
    for _ in range(20):
        if est.get_distances():
            break
        time.sleep(0.05)
    d = est.get_distances()
    assert '1-2' in d
    assert d['1-2'] > 0
