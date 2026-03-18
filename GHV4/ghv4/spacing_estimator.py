# GHV4/ghv4/spacing_estimator.py
"""spacing_estimator.py — MUSIC-only shouter spacing estimation for GHV4.

Public API:
    CSIMUSICEstimator()
    .collect(reporter_id, peer_id, csi_bytes) — feed CSI snapshot
    .get_distances()                          — snapshot dict {"1-2": dist_m, ...}
    .reset_all()                              — clear all buffers

    SpacingEstimator(spacing_path, music_estimator)
    .start()                  — start daemon writer thread
    .get_distances()          — delegates to CSIMUSICEstimator
    ._maybe_write()           — rate-limited atomic JSON write
"""
import json
import os
import queue
import threading
import time
from typing import Dict

import logging

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

_log = logging.getLogger("ghv4.spacing_estimator")

from ghv4.config import (
    PAIR_KEYS,
    NULL_SUBCARRIER_INDICES,
    SUBCARRIERS,
    MUSIC_TAU_MAX_S,
    MUSIC_TAU_STEPS,
    MUSIC_MIN_SNAP,
    MUSIC_MAX_SNAP,
    CSI_NOISE_FLOOR,
)

_PAIR_INDICES = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# MUSIC CSI ranging constants
CH6_CENTER_HZ         = 2_437_000_000.0
SUBCARRIER_SPACING_HZ = 312_500.0
VALID_INDICES         = [k for k in range(SUBCARRIERS) if k not in NULL_SUBCARRIER_INDICES]
SUBCARRIER_FREQS      = np.array([
    CH6_CENTER_HZ + (k - 64) * SUBCARRIER_SPACING_HZ for k in VALID_INDICES
])


class CSIMUSICEstimator:
    """
    Collects per-pair CSI snapshots from [0xEE][0xFF] serial frames and
    computes offset-free pairwise distances via MUSIC super-resolution CIR.
    Thread-safe. get_distances() returns a dict {"1-2": dist_m, ...}.
    """

    def __init__(self) -> None:
        self._H: dict = {}                         # (reporter, peer) → list[np.ndarray]
        self._distances: dict = {}
        self._enqueued: set = set()                # pairs with pending compute job
        self._reset_gen: int = 0                   # incremented on reset_all()
        self._lock = threading.Lock()
        self._compute_queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._compute_loop, daemon=True, name="MUSICCompute"
        )
        self._worker.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def collect(self, reporter_id: int, peer_id: int, csi_bytes: bytes) -> None:
        """Called from serial dispatch thread for each [0xEE][0xFF] frame."""
        vec = self._csi_to_complex(csi_bytes)
        if vec is None:
            return
        if np.max(np.abs(vec)) < CSI_NOISE_FLOOR:
            _log.debug("MUSIC collect: pair (%d,%d) rejected — below noise floor", reporter_id, peer_id)
            return
        key  = (reporter_id, peer_id)
        rev  = (peer_id, reporter_id)
        pair_norm = (min(reporter_id, peer_id), max(reporter_id, peer_id))
        with self._lock:
            if key not in self._H:
                self._H[key] = []
            if len(self._H[key]) < MUSIC_MAX_SNAP:
                self._H[key].append(vec)
                _log.debug("MUSIC collect: pair (%d,%d) now %d/%d snapshots",
                           reporter_id, peer_id, len(self._H[key]), MUSIC_MAX_SNAP)
            if pair_norm not in self._enqueued:
                n_fwd = len(self._H.get(key, []))
                n_rev = len(self._H.get(rev, []))
                if n_fwd >= MUSIC_MIN_SNAP and n_rev >= MUSIC_MIN_SNAP:
                    H_fwd = list(self._H[key])
                    H_rev = list(self._H.get(rev, []))
                    gen   = self._reset_gen
                    self._enqueued.add(pair_norm)
                    self._compute_queue.put(
                        (reporter_id, peer_id, H_fwd, H_rev, gen)
                    )
                    _log.info("MUSIC compute queued: pair (%d-%d) fwd=%d rev=%d",
                              min(reporter_id, peer_id), max(reporter_id, peer_id), n_fwd, n_rev)

    def get_distances(self) -> dict:
        """Thread-safe snapshot of all computed MUSIC distances."""
        with self._lock:
            return dict(self._distances)

    def reset_all(self) -> None:
        """
        Clear all snapshot buffers and distances. Call when ranging phase resets.
        Increments _reset_gen so any in-flight compute jobs discard results.
        """
        with self._lock:
            self._H.clear()
            self._distances.clear()
            self._enqueued.clear()
            self._reset_gen += 1

    def reset_pair(self, i: int, j: int) -> None:
        pair_norm = (min(i, j), max(i, j))
        with self._lock:
            self._H.pop((i, j), None)
            self._H.pop((j, i), None)
            self._distances.pop(f"{i}-{j}", None)
            self._distances.pop(f"{j}-{i}", None)
            self._enqueued.discard(pair_norm)
            self._reset_gen += 1

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_loop(self) -> None:
        while True:
            reporter_id, peer_id, H_fwd, H_rev, job_gen = self._compute_queue.get()
            with self._lock:
                current_gen = self._reset_gen
            if job_gen != current_gen:
                continue
            pair_key = f"{min(reporter_id, peer_id)}-{max(reporter_id, peer_id)}"
            H_ij = np.array(H_fwd).T   # (121, N)
            H_ji = np.array(H_rev).T   # (121, N)
            tau_ij = self._music_delay(H_ij)
            tau_ji = self._music_delay(H_ji)
            if tau_ij is None or tau_ji is None:
                reason = "tau_ij=None" if tau_ij is None else "tau_ji=None"
                _log.warning("MUSIC failed: pair (%s) %s", pair_key, reason)
                continue
            tau_avg = (tau_ij + tau_ji) / 2.0
            d = float(SPEED_OF_LIGHT * tau_avg)
            _log.info("MUSIC result: pair (%s) tau=%.2f ns d=%.2f m", pair_key, tau_avg * 1e9, d)
            with self._lock:
                if self._reset_gen == job_gen:
                    self._distances[pair_key] = round(d, 2)

    def _music_delay(self, H: np.ndarray):
        """
        MUSIC delay estimator.
        H: (K, N) complex array — K=121 subcarriers, N snapshots.
        Returns estimated delay in seconds, or None on failure.
        """
        K, N = H.shape
        if N < 2:
            return None
        R = (H @ H.conj().T) / N
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(R)
        except np.linalg.LinAlgError:
            _log.warning("MUSIC: eigh failed (singular covariance)")
            return None
        L = self._mdl_order(eigenvalues, N, K)
        if L < 1:
            _log.warning("MUSIC: MDL returned order %d, rejecting", L)
            return None
        L = min(L, K // 2)
        E_noise = eigenvectors[:, :K - L]
        tau_grid = np.linspace(0, MUSIC_TAU_MAX_S, MUSIC_TAU_STEPS)
        phase_matrix = np.exp(
            -1j * 2 * np.pi * np.outer(SUBCARRIER_FREQS, tau_grid)
        )
        proj   = E_noise.conj().T @ phase_matrix
        pseudo = 1.0 / np.sum(np.abs(proj) ** 2, axis=0)
        tau = self._first_peak_tau(pseudo, tau_grid)
        if tau is not None and (tau < 0 or tau > MUSIC_TAU_MAX_S):
            _log.warning("MUSIC: tau=%.2f ns out of range, rejecting", tau * 1e9)
            return None
        return tau

    @staticmethod
    def _mdl_order(eigenvalues: np.ndarray, N: int, K: int) -> int:
        """MDL model order selection. Returns estimated number of signal components."""
        eigenvalues = eigenvalues[::-1]   # descending
        best_L, best_score = 1, np.inf
        for L in range(1, K // 2):
            noise_eigs = eigenvalues[L:]
            if len(noise_eigs) == 0:
                break
            geom  = np.exp(np.mean(np.log(np.maximum(noise_eigs, 1e-12))))
            arith = np.mean(noise_eigs)
            if geom <= 0 or arith <= 0:
                continue
            log_lik = -(K - L) * N * np.log(geom / arith)
            penalty = 0.5 * L * (2 * K - L) * np.log(N)
            score   = -log_lik + penalty
            if score < best_score:
                best_score, best_L = score, L
        return best_L

    @staticmethod
    def _first_peak_tau(pseudo: np.ndarray, tau_grid: np.ndarray):
        """Returns τ of the dominant (global maximum) peak of the pseudospectrum.

        Prior implementation returned the first local maximum above the mean,
        which consistently selected spurious near-zero-delay sidelobes in
        reverberant indoor environments instead of the true propagation peak.
        The dominant peak is physically the most likely direct-path arrival.
        Boundary maxima (index 0 or last) are rejected as tau_grid artefacts.
        """
        peak_idx = int(np.argmax(pseudo))
        if peak_idx == 0 or peak_idx == len(pseudo) - 1:
            return None  # peak at tau_grid boundary — reject as artefact
        return float(tau_grid[peak_idx])

    @staticmethod
    def _csi_to_complex(csi_bytes: bytes):
        """
        Convert raw ESP32 CSI bytes to complex128 array of length 121.
        ESP32 format: interleaved int8 pairs (imaginary, real) per subcarrier.
        With 128 subcarriers × 2 bytes each = 256 bytes minimum required.
        """
        if len(csi_bytes) < SUBCARRIERS * 2:
            return None
        raw  = np.frombuffer(csi_bytes[:SUBCARRIERS * 2], dtype=np.int8)
        imag = raw[0::2].astype(np.float64)
        real = raw[1::2].astype(np.float64)
        full = real + 1j * imag               # (128,)
        return full[VALID_INDICES]             # (121,)


class SpacingEstimator:
    """MUSIC-only spacing estimator. Writes spacing.json from CSIMUSICEstimator.

    Preserves the public interface (start, get_distances, feed) so that UI code
    and serial_io continue to work without changes. The feed() method is now a
    no-op since RSSI distance calculation has been removed — [0xCC][0xDD] frames
    are consumed and discarded at the serial layer.
    """

    def __init__(self, spacing_path: str = "spacing.json",
                 music_estimator=None):
        self._path             = spacing_path
        self._music_estimator  = music_estimator
        self._last_write       = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SpacingWriter"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()

    def feed(self, frame: dict) -> None:
        """No-op. Retained for interface compatibility."""
        pass

    def get_distances(self) -> Dict[str, float]:
        """Thread-safe snapshot. Returns MUSIC distances only."""
        if self._music_estimator is None:
            return {}
        return self._music_estimator.get_distances()

    def get_rssi_values(self) -> Dict[str, float]:
        """No-op stub. RSSI distance removed in GHV4. Returns empty dict."""
        return {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Periodically write spacing.json with MUSIC distances."""
        while True:
            time.sleep(1.0)
            self._maybe_write()

    def _maybe_write(self) -> None:
        """Write spacing.json atomically, rate-limited to <=1 write/second."""
        now = time.time()
        if now - self._last_write < 1.0:
            return
        self._last_write = now

        distances = self.get_distances()
        if not distances:
            return

        pairs = {}
        for key in PAIR_KEYS:
            if key not in distances:
                continue
            pairs[key] = {
                "distance_m": round(distances[key], 2),
                "source": "music",
            }
        out = {
            "version": 2,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pairs":   pairs,
        }

        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, self._path)
