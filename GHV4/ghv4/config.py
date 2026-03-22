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

# ── EDA constants ──────────────────────────────────────────────
META_COLS = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col"]
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
