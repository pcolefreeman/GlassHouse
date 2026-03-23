"""GHV4 shared constants — single source of truth for all modules."""

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

# ── Shouters & grid ────────────────────────────────────────────
ACTIVE_SHOUTER_IDS = [1, 2, 3, 4]
PAIR_KEYS = ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]
CELL_LABELS = [f"r{r}c{c}" for r in range(3) for c in range(3)]
GRID_POS = {i: (i // 3, i % 3) for i in range(9)}

# ── Feature contract ──────────────────────────────────────────
SPACING_FEATURE_NAMES = [f"dist_{k}" for k in PAIR_KEYS]

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

# ── EDA constants ──────────────────────────────────────────────
META_COLS = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col", "activity"]
EXPECTED_COLS = 5134

# ---------------------------------------------------------------------------
# ML Distance Estimation
# ---------------------------------------------------------------------------
VALID_SUBCARRIER_COUNT = SUBCARRIERS - len(NULL_SUBCARRIER_INDICES)  # 121
DISTANCE_FEATURE_COUNT = VALID_SUBCARRIER_COUNT * 2 * 2             # 484

CALIBRATION_WINDOW_S   = 30       # seconds of snap collection
CALIBRATION_EXTENSION_S = 15      # extension if insufficient data
CALIBRATION_MAX_EXTENSIONS = 2    # max number of extensions
CALIBRATION_MIN_PAIRS  = 10       # min matched fwd/rev pairs per shouter pair

DISTANCE_MODEL_DIR     = "distance_models"
DISTANCE_MAX_TREES     = 200      # max estimators per GB/RF model (Pi memory budget)

# ── Breathing detection ───────────────────────────────────────
BREATHING_WINDOW_S    = 30
BREATHING_WINDOW_N    = int(BREATHING_WINDOW_S / (BUCKET_MS / 1000))  # 150 frames
BREATHING_SLIDE_N     = 10        # 10 frames × BUCKET_MS = 2000 ms between updates
BREATHING_BAND_HZ     = (0.1, 0.5)
BREATHING_NPAIRS      = 10
BREATHING_CONFIDENCE_THRESHOLD = 0.3

# Path-to-cell mapping: which grid cells each shouter path crosses.
# Keys are shouter IDs (1-4), values are lists of cell labels.
# Default assumes shouters at 4 corners, listener roughly central.
BREATHING_PATH_MAP = {
    1: ["r0c0", "r1c1"],   # top-left corner
    2: ["r0c2", "r1c1"],   # top-right corner
    3: ["r2c0", "r1c1"],   # bottom-left corner
    4: ["r2c2", "r1c1"],   # bottom-right corner
}
