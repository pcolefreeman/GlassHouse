# GHV3.1/ghv3_1/spacing_estimator.py
"""spacing_estimator.py — RSSI-based shouter spacing estimation for GHV3.

Public API:
    SpacingEstimator(spacing_path, config_path)
    .start()                  — start daemon thread
    .feed(frame)              — enqueue a ranging frame dict {'payload': bytes}
    .get_distances()          — snapshot dict {"1-2": dist_m, ...}
    ._process(frame)          — exposed for unit testing (bypass queue)
    ._distance(rssi_pair)     — log-distance path loss model
    ._maybe_write()           — rate-limited atomic JSON write
"""
import json
import os
import queue
import struct
import threading
import time
from typing import Dict, Optional

import logging

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

_log = logging.getLogger("ghv3_1.spacing_estimator")

from ghv3_1.config import (
    PAIR_KEYS,
    NULL_SUBCARRIER_INDICES,
    SUBCARRIERS,
    DEFAULT_RSSI_N,
    DEFAULT_RSSI_REF_DBM,
    DEFAULT_RSSI_D0_M,
    MUSIC_TAU_MAX_S,
    MUSIC_TAU_STEPS,
    MUSIC_MIN_SNAP,
    MUSIC_MAX_SNAP,
    CSI_NOISE_FLOOR,
)

_PAIR_INDICES = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
# Fallback config when ranging_config.json is missing — uses config.py defaults
_DEFAULT_CONFIG = {
    "n": DEFAULT_RSSI_N,             # 2.5 (generic)
    "rssi_ref_dbm": DEFAULT_RSSI_REF_DBM,  # -40.0 (generic)
    "d0_m": DEFAULT_RSSI_D0_M,      # 1.0
}
ALPHA = 0.1
MIN_SAMPLES = 1

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
    Thread-safe. get_distances() returns a dict compatible with SpacingEstimator.
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
        """Returns τ of first local maximum above the mean, or None."""
        threshold = np.mean(pseudo)
        for i in range(1, len(pseudo) - 1):
            if (pseudo[i] > threshold
                    and pseudo[i] >= pseudo[i - 1]
                    and pseudo[i] >= pseudo[i + 1]):
                return float(tau_grid[i])
        return None

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
    """Consumes [0xCC][0xDD] ranging frames; produces spacing.json."""

    def __init__(self, spacing_path: str = "spacing.json",
                 config_path: str = "ranging_config.json",
                 music_estimator=None):
        self._path             = spacing_path
        self._config_path      = config_path
        self._config           = self._load_config(config_path)
        self._music_estimator  = music_estimator
        # 1-indexed; row=reporter, col=peer; index 0 unused pad
        self._rssi  = np.zeros((5, 5), dtype=float)
        self._count = np.zeros((5, 5), dtype=int)
        self._ranging_queue: queue.Queue = queue.Queue()
        self._last_write = 0.0
        self._lock = threading.Lock()
        self._prev_source: dict = {}  # pair_key → "rssi" | "music"
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SpacingEstimator"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread.start()

    def feed(self, frame: dict) -> None:
        """Non-blocking. Called from GlassHouseV3 dispatch loop."""
        self._ranging_queue.put(frame)

    def get_distances(self) -> Dict[str, float]:
        """Thread-safe snapshot. MUSIC distances take precedence over RSSI."""
        rssi_dist  = self._get_rssi_distances()
        music_dist = self._music_estimator.get_distances() if self._music_estimator else {}
        merged = {**rssi_dist, **music_dist}
        # Log only on source transitions (not every 200ms call)
        for k, d in merged.items():
            src = "music" if k in music_dist else "rssi"
            if self._prev_source.get(k) != src:
                _log.info("Distance %s source changed: %s -> %s (d=%.2f m)",
                          k, self._prev_source.get(k, "none"), src, d)
                self._prev_source[k] = src
        return merged

    def get_rssi_values(self) -> Dict[str, float]:
        """Thread-safe snapshot of per-pair average EMA RSSI (dBm)."""
        with self._lock:
            result = {}
            for key, (i, j) in zip(PAIR_KEYS, _PAIR_INDICES):
                if min(int(self._count[i][j]), int(self._count[j][i])) >= MIN_SAMPLES:
                    result[key] = (self._rssi[i][j] + self._rssi[j][i]) / 2.0
            return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_rssi_distances(self) -> Dict[str, float]:
        """RSSI-only distances. Called by get_distances()."""
        with self._lock:
            result = {}
            for key, (i, j) in zip(PAIR_KEYS, _PAIR_INDICES):
                samples = min(int(self._count[i][j]), int(self._count[j][i]))
                if samples >= MIN_SAMPLES:
                    rssi_pair = (self._rssi[i][j] + self._rssi[j][i]) / 2.0
                    result[key] = self._distance(rssi_pair)
            return dict(result)

    def _run(self) -> None:
        while True:
            frame = self._ranging_queue.get()
            self._config = self._load_config(self._config_path)  # hot-reload on each frame
            self._process(frame)

    def _process(self, frame: dict) -> None:
        """Parse frame payload and update RSSI/count arrays. Called from daemon thread."""
        payload = frame.get('payload', b'')
        if len(payload) < 12:
            return
        ver, reporter_id = struct.unpack_from('<BB', payload, 0)
        peer_rssi  = list(struct.unpack_from('<5b', payload, 2))
        peer_count = list(struct.unpack_from('<5B', payload, 7))

        with self._lock:
            for peer_id in range(1, 5):
                if peer_id == reporter_id:
                    continue
                if peer_count[peer_id] == 0:
                    continue  # shouter has no data for this peer
                new_rssi = float(peer_rssi[peer_id])
                if self._count[reporter_id][peer_id] == 0:
                    # First observation — initialise without blending
                    self._rssi[reporter_id][peer_id] = new_rssi
                else:
                    self._rssi[reporter_id][peer_id] = (
                        (1.0 - ALPHA) * self._rssi[reporter_id][peer_id]
                        + ALPHA * new_rssi
                    )
                self._count[reporter_id][peer_id] += 1

        # Compute merged distances OUTSIDE the lock, then pass to writer
        merged = self.get_distances()
        self._maybe_write(merged)

    def _distance(self, rssi_pair: float) -> float:
        """Log-distance path loss: d = d0 × 10^((rssi_ref − rssi) / (10n))."""
        cfg = self._config
        n, rssi_ref, d0 = cfg["n"], cfg["rssi_ref_dbm"], cfg["d0_m"]
        return d0 * (10 ** ((rssi_ref - rssi_pair) / (10.0 * n)))

    def _maybe_write(self, merged_distances: dict = None) -> None:
        """Write spacing.json atomically, rate-limited to <=1 write/second."""
        now = time.time()
        if now - self._last_write < 1.0:
            return
        self._last_write = now

        # If no merged distances provided, compute them (for direct callers)
        if merged_distances is None:
            merged_distances = self.get_distances()

        # Determine source for each pair
        music_dist = self._music_estimator.get_distances() if self._music_estimator else {}

        with self._lock:
            pairs = {}
            for key, (i, j) in zip(PAIR_KEYS, _PAIR_INDICES):
                if key not in merged_distances:
                    continue
                samples = min(int(self._count[i][j]), int(self._count[j][i]))
                rssi_pair = (self._rssi[i][j] + self._rssi[j][i]) / 2.0 if samples >= MIN_SAMPLES else None
                source = "music" if key in music_dist else "rssi"
                pairs[key] = {
                    "distance_m": round(merged_distances[key], 2),
                    "source": source,
                    "rssi_avg": round(rssi_pair, 1) if rssi_pair is not None else None,
                    "samples": samples,
                }
            out = {
                "version": 1,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pairs":   pairs,
                "config":  self._config,
            }

        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, self._path)

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            return {
                "n":           float(cfg.get("n",           _DEFAULT_CONFIG["n"])),
                "rssi_ref_dbm": float(cfg.get("rssi_ref_dbm", _DEFAULT_CONFIG["rssi_ref_dbm"])),
                "d0_m":        float(cfg.get("d0_m",        _DEFAULT_CONFIG["d0_m"])),
            }
        except FileNotFoundError:
            return dict(_DEFAULT_CONFIG)
