"""GHV5 shared constants — breathing detection only."""

# ── Serial ─────────────────────────────────────────────────────
BAUD_RATE = 921_600

# ── Frame magic bytes ──────────────────────────────────────────
MAGIC_LISTENER = bytes([0xAA, 0x55])
MAGIC_SHOUTER  = bytes([0xBB, 0xDD])
MAGIC_CSI_SNAP = bytes([0xEE, 0xFF])

# ── Frame header sizes (after magic bytes consumed) ────────────
LISTENER_HDR_SIZE = 20
SHOUTER_HDR_SIZE  = 29
CSI_SNAP_HDR_SIZE = 6

# ── CSI geometry ───────────────────────────────────────────────
SUBCARRIERS = 128
NULL_SUBCARRIER_INDICES = frozenset({0, 1, 2, 32, 63, 64, 65})

# ── Grid ───────────────────────────────────────────────────────
CELL_LABELS = [f"r{r}c{c}" for r in range(3) for c in range(3)]

# ── Pi Display ────────────────────────────────────────────────
PI_DISPLAY_FPS         = 10
PI_DISPLAY_BG          = (13, 13, 13)
PI_CELL_ACTIVE         = (255, 107, 53)
PI_CELL_INACTIVE       = (26, 26, 26)
PI_CELL_BORDER         = (68, 68, 68)
PI_TEXT_ACTIVE         = (255, 255, 255)
PI_TEXT_INACTIVE       = (102, 102, 102)
PI_SCREEN_SIZE         = (800, 480)

# ── Breathing detection ───────────────────────────────────────
BREATHING_WINDOW_S    = 30
BREATHING_SNAP_HZ     = 20
BREATHING_WINDOW_N    = int(BREATHING_WINDOW_S * BREATHING_SNAP_HZ)  # 600 frames
BREATHING_SLIDE_N     = 20        # 20 frames at 20 Hz = 1s between updates
BREATHING_BAND_HZ     = (0.1, 0.5)
BREATHING_NPAIRS      = 10
BREATHING_CONFIDENCE_THRESHOLD = 0.05
BREATHING_PCA_COMPONENTS = 3     # PCA components retained before FFT
PRESENCE_VARIANCE_MIDPOINT  = 50.0   # sigmoid center (tune with --log-level DEBUG)
PRESENCE_VARIANCE_STEEPNESS = 0.5    # sigmoid steepness

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
