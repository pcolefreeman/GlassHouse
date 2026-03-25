"""GHV5 shared constants — single source of truth for all modules."""

from pathlib import Path

# ── Project paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
# ── Serial ─────────────────────────────────────────────────────
BAUD_RATE = 921_600

# ── Frame magic bytes ──────────────────────────────────────────
MAGIC_LISTENER = bytes([0xAA, 0x55])
MAGIC_SHOUTER  = bytes([0xBB, 0xDD])
MAGIC_CSI_SNAP = bytes([0xEE, 0xFF])

# ── Frame header sizes (after magic bytes consumed) ────────────
LISTENER_HDR_SIZE = 20
SHOUTER_HDR_SIZE  = 29
CSI_SNAP_HDR_SIZE = 6  # offsetof(csi_snap_pkt_t, csi) minus 2-byte magic

# ── CSI geometry ───────────────────────────────────────────────
SUBCARRIERS = 128
NULL_SUBCARRIER_INDICES = frozenset({0, 1, 2, 32, 63, 64, 65})

# Phase-diff null indices — includes neighbors of null subcarriers
# because pdiff[i] = phase[i+1] - phase[i] is undefined when either
# endpoint is null.
NULL_PDIFF_INDICES = frozenset({0, 1, 2, 31, 32, 62, 63, 64, 65})

# ── Timing ─────────────────────────────────────────────────────
BUCKET_MS = 200
POLL_INTERVAL_MIN_MS = 50
SPACING_JSON_REFRESH_S = 5

# ── Shouters & grid ────────────────────────────────────────────
ACTIVE_SHOUTER_IDS = [1, 2, 3, 4]
PAIR_KEYS = ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]
CELL_LABELS = [f"r{r}c{c}" for r in range(3) for c in range(3)]
GRID_POS = {i: (i // 3, i % 3) for i in range(9)}

# ── Feature contract ──────────────────────────────────────────
SPACING_FEATURE_NAMES = [f"dist_{k}" for k in PAIR_KEYS]

# ── WiFi RF ──────────────────────────────────────────────────
WIFI_CHANNEL6_CENTER_HZ   = 2_437_000_000.0
WIFI_SUBCARRIER_SPACING_HZ = 312_500.0

# ── MUSIC estimator ───────────────────────────────────────────
MUSIC_TAU_MAX_S = 100e-9      # 30 m search range
MUSIC_TAU_STEPS = 1000         # 0.1 ns → ~3 cm steps
MUSIC_MIN_SNAP = 15            # min snapshots per direction to run MUSIC
MUSIC_MAX_SNAP = 35            # matches N_SNAP in firmware (DRAM limit on ESP32)
CSI_NOISE_FLOOR = 2            # min max(abs(csi)) to accept snapshot; tunable

# ── UI constants ───────────────────────────────────────────────
MAX_LOG_LINES = 500

# ── Pi Display ────────────────────────────────────────────────
PI_DISPLAY_FPS         = 10                  # 10 Hz redraw
PI_DISPLAY_BG          = (13, 13, 13)        # #0d0d0d (matches viz.py dark bg)
PI_CELL_ACTIVE         = (255, 107, 53)      # #FF6B35 rescue orange (matches viz.py)
PI_CELL_INACTIVE       = (26, 26, 26)        # #1a1a1a (matches viz.py)
PI_CELL_BORDER         = (68, 68, 68)        # #444444
PI_TEXT_ACTIVE         = (255, 255, 255)     # white
PI_TEXT_INACTIVE       = (102, 102, 102)     # #666666
PI_SCREEN_SIZE         = (800, 480)          # Standard Pi 7" DSI LCD

# SAR / Pi display layout (shared by breathing.py and pi_display.py)
SAR_DISPLAY_TITLE_H    = 44
SAR_DISPLAY_STATUS_H   = 40
SAR_DISPLAY_GRID_PAD   = 24
SAR_DISPLAY_CELL_GAP   = 4
SAR_DISPLAY_MARGIN     = 14

# ── EDA constants ──────────────────────────────────────────────
META_COLS = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col", "activity"]
EXPECTED_COLS = 5134

# ── Signal hardening (RuView-inspired) ───────────────────────
HAMPEL_WINDOW = 11                        # Sliding window size for Hampel filter
HAMPEL_THRESHOLD = 3.0                    # MAD multiplier for outlier rejection
COHERENCE_THRESHOLD = 0.3                 # Min coherence score to accept frame
SUBCARRIER_TOP_K = 30                     # Number of subcarriers to select per path
SUBCARRIER_MIN_K = 10                     # Never select fewer than this many subcarriers

# ── Heart rate detection ─────────────────────────────────────
HEARTRATE_BAND_HZ = (0.8, 2.0)           # Heart rate frequency range
HEARTRATE_CONFIDENCE_THRESHOLD = 0.2      # Lower than breathing (0.3) — weaker signal
HEARTRATE_PEAK_PROMINENCE = 0.05          # FFT peak prominence threshold

# ── Presence scoring ────────────────────────────────────────
PRESENCE_LOGSIGMOID_SCALE = 2.0           # Steepness of log-sigmoid mapping
PRESENCE_LOGSIGMOID_MIDPOINT = 0.5        # log1p(var) midpoint for sigmoid
PRESENCE_RANK_DIVISOR = 2.0               # Normalizer: (mean/median - 1) / divisor → 0-1

# ── Breathing detection ───────────────────────────────────────
BREATHING_WINDOW_S    = 15
BREATHING_SNAP_HZ     = 20
BREATHING_WINDOW_N    = int(BREATHING_WINDOW_S * BREATHING_SNAP_HZ)  # 600 frames
BREATHING_SLIDE_N     = 20        # 20 frames at 20 Hz = 1s between updates
BREATHING_BAND_HZ     = (0.1, 0.5)
BREATHING_NPAIRS      = 10
BREATHING_CONFIDENCE_THRESHOLD = 0.30  # raised from 0.05; contrast makes empty room 0%
BREATHING_CONTRAST_CEILING    = 3.0   # contrast ratio at which confidence saturates to 1.0
                                       # (contrast = path_snr_eig / median_all_snr_eig)
BREATHING_MIN_PATHS_FOR_CONTRAST = 3   # need 3+ paths for meaningful median; fewer → phase only
BREATHING_MIN_PATHS_TOTAL     = 2      # Absolute minimum active paths required to attempt a guess

# ── Detection hardening (anti-ghost, temporal persistence) ────
BREATHING_CONFIDENCE_BETA     = 0.3    # EMA smoothing factor for per-path confidence (~3s time constant)
BREATHING_CONFIRM_WINDOWS     = 3      # Consecutive above-threshold windows to confirm detection
BREATHING_RELEASE_WINDOWS     = 2      # Consecutive below-threshold windows to release detection
BREATHING_BASELINE_ALPHA      = 0.05   # EMA factor for per-path amplitude baseline (~20 window TC)
BREATHING_BASELINE_WARMUP     = 10     # Windows before per-path baseline is trusted
BREATHING_STALE_TIMEOUT_S     = 30     # Seconds of no data before resetting path state

# Path-to-cell mapping: which grid cells each shouter↔shouter path crosses.
# Keys are (min_id, max_id) tuples for undirected shouter pairs.
# Shouter corners: S1=BL, S2=TL, S3=TR, S4=BR.
BREATHING_PATH_MAP = {
    (1, 2): ["r2c0", "r1c0", "r0c0"],  # S1(BL)↔S2(TL) = left edge
    (1, 3): ["r2c0", "r1c1", "r0c2"],  # S1(BL)↔S3(TR) = BL→TR diagonal
    (1, 4): ["r2c0", "r2c1", "r2c2"],  # S1(BL)↔S4(BR) = bottom edge
    (2, 3): ["r0c0", "r0c1", "r0c2"],  # S2(TL)↔S3(TR) = top edge
    (2, 4): ["r0c0", "r1c1", "r2c2"],  # S2(TL)↔S4(BR) = TL→BR diagonal
    (3, 4): ["r0c2", "r1c2", "r2c2"],  # S3(TR)↔S4(BR) = right edge
}
