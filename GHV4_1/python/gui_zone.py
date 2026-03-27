"""
Zone Display GUI — renders the detection area as a 2D rectangle with
real-time zone highlighting and presence status.

Uses pygame for cross-platform rendering.  The module can be imported
without pygame installed (lazy import in ZoneDisplay.__init__).

Room layout (looking down):
    A (top-left)  ----  B (top-right)
    |        \\ /        |
    |         X         |
    |        / \\        |
    C (bot-left)  ----  D (bot-right)

Quadrants:
    Q1 = top-left  (near A)
    Q2 = top-right (near B)
    Q3 = bot-left  (near C)
    Q4 = bot-right (near D)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from presence_detector import RoomState
    from zone_detector import Zone, ZoneResult

# ---------------------------------------------------------------------------
# Constants — importable without pygame
# ---------------------------------------------------------------------------

#: Window dimensions (pixels)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

#: Detection area rectangle (within the window)
AREA_LEFT = 60
AREA_TOP = 80
AREA_WIDTH = 480
AREA_HEIGHT = 400

#: Colors (R, G, B)
COLOR_BG = (30, 30, 35)
COLOR_GRID = (80, 80, 90)
COLOR_GRID_DIVIDER = (60, 60, 70)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 150)
COLOR_TEXT_LABEL = (180, 180, 190)
COLOR_NODE = (100, 180, 255)
COLOR_LINK_IDLE = (50, 50, 60)
COLOR_LINK_MOTION = (255, 160, 40)
COLOR_EMPTY = (40, 60, 40)
COLOR_OCCUPIED = (200, 60, 60)
COLOR_ZONE_ACTIVE = (60, 180, 100, 120)  # semi-transparent green
COLOR_ZONE_INACTIVE = (40, 40, 48)
COLOR_CONFIDENCE_HIGH = (60, 200, 120)
COLOR_CONFIDENCE_LOW = (200, 180, 60)
COLOR_SIDEBAR_BG = (38, 38, 44)
COLOR_FPS = (100, 100, 110)

#: Node positions (corners of the detection area) — pixel coordinates
#: Computed relative to area rectangle at runtime by _compute_layout()
NODE_NAMES = ["A", "B", "C", "D"]

#: Link definitions — pairs of node names
LINK_DEFS = [
    ("A", "B"),  # top edge
    ("C", "D"),  # bottom edge
    ("A", "C"),  # left edge
    ("B", "D"),  # right edge
    ("A", "D"),  # diagonal
    ("B", "C"),  # diagonal
]

#: Maps link tuple to canonical link_id
LINK_TO_ID = {
    ("A", "B"): "AB",
    ("C", "D"): "CD",
    ("A", "C"): "AC",
    ("B", "D"): "BD",
    ("A", "D"): "AD",
    ("B", "C"): "BC",
}

#: Quadrant label positions relative to area (fractional)
ZONE_LABELS = {
    "Q1": (0.25, 0.25),
    "Q2": (0.75, 0.25),
    "Q3": (0.25, 0.75),
    "Q4": (0.75, 0.75),
}

#: Sidebar position
SIDEBAR_LEFT = AREA_LEFT + AREA_WIDTH + 30
SIDEBAR_TOP = AREA_TOP
SIDEBAR_WIDTH = 200


def _compute_layout() -> dict[str, tuple[int, int]]:
    """Compute node pixel positions from the area rectangle."""
    return {
        "A": (AREA_LEFT, AREA_TOP),
        "B": (AREA_LEFT + AREA_WIDTH, AREA_TOP),
        "C": (AREA_LEFT, AREA_TOP + AREA_HEIGHT),
        "D": (AREA_LEFT + AREA_WIDTH, AREA_TOP + AREA_HEIGHT),
    }


# ---------------------------------------------------------------------------
# ZoneDisplay — pygame renderer
# ---------------------------------------------------------------------------


class ZoneDisplay:
    """Real-time zone visualization using pygame.

    Create with ``ZoneDisplay()`` then call ``update()`` each frame.
    Call ``close()`` to shut down cleanly.

    Raises ImportError on construction if pygame is not installed.
    """

    def __init__(self, title: str = "CSI Zone Monitor") -> None:
        # Lazy import — module can be imported without pygame
        import pygame as _pg

        self._pg = _pg
        self._pg.init()

        self._screen = self._pg.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            _pg.RESIZABLE,
        )
        self._pg.display.set_caption(title)

        self._font_large = self._pg.font.SysFont("consolas", 28, bold=True)
        self._font_medium = self._pg.font.SysFont("consolas", 20)
        self._font_small = self._pg.font.SysFont("consolas", 15)
        self._font_node = self._pg.font.SysFont("consolas", 22, bold=True)
        self._font_zone = self._pg.font.SysFont("consolas", 32, bold=True)

        self._node_positions = _compute_layout()
        self._clock = self._pg.Clock()
        self._fps = 0.0

    def update(
        self,
        room_state: RoomState,
        zone_result: ZoneResult,
        link_states: dict[str, dict],
        fps: float = 0.0,
    ) -> bool:
        """Redraw the display with current detection state.

        Args:
            room_state: Current OCCUPIED/EMPTY state.
            zone_result: ZoneResult from ZoneDetector.estimate().
            link_states: Dict from PresenceEngine.get_link_states().
            fps: Frames per second to display (0 to hide).

        Returns:
            False if the user closed the window (quit event), True otherwise.
        """
        pg = self._pg

        # Handle events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return False

        self._screen.fill(COLOR_BG)

        self._draw_title(room_state)
        self._draw_zones(room_state, zone_result)
        self._draw_links(link_states)
        self._draw_nodes()
        self._draw_zone_labels(zone_result)
        self._draw_sidebar(room_state, zone_result, link_states)
        if fps > 0:
            self._draw_fps(fps)

        pg.display.flip()
        self._clock.tick(30)  # Cap at 30 FPS for display updates
        return True

    def close(self) -> None:
        """Shut down pygame display."""
        self._pg.quit()

    # -- drawing helpers ----------------------------------------------------

    def _draw_title(self, room_state: RoomState) -> None:
        """Draw the status header at the top."""
        from presence_detector import RoomState as RS

        if room_state == RS.OCCUPIED:
            color = COLOR_OCCUPIED
            text = "● OCCUPIED"
        else:
            color = COLOR_EMPTY
            text = "○ EMPTY"

        surf = self._font_large.render(text, True, color)
        self._screen.blit(surf, (AREA_LEFT, 20))

        # Subtitle
        sub = self._font_small.render(
            "CSI Zone Monitor — Real-time Presence & Zone Detection",
            True,
            COLOR_TEXT_DIM,
        )
        self._screen.blit(sub, (AREA_LEFT, 52))

    def _draw_zones(self, room_state: RoomState, zone_result: ZoneResult) -> None:
        """Draw the 4 quadrant rectangles with zone highlighting."""
        from presence_detector import RoomState as RS
        from zone_detector import Zone

        pg = self._pg
        half_w = AREA_WIDTH // 2
        half_h = AREA_HEIGHT // 2

        zone_rects = {
            Zone.Q1: pg.Rect(AREA_LEFT, AREA_TOP, half_w, half_h),
            Zone.Q2: pg.Rect(AREA_LEFT + half_w, AREA_TOP, half_w, half_h),
            Zone.Q3: pg.Rect(AREA_LEFT, AREA_TOP + half_h, half_w, half_h),
            Zone.Q4: pg.Rect(AREA_LEFT + half_w, AREA_TOP + half_h, half_w, half_h),
        }

        for zone, rect in zone_rects.items():
            if (
                room_state == RS.OCCUPIED
                and zone_result.zone == zone
            ):
                # Active zone — highlighted
                highlight = pg.Surface((rect.width, rect.height), pg.SRCALPHA)
                highlight.fill(COLOR_ZONE_ACTIVE)
                self._screen.blit(highlight, rect.topleft)
            else:
                pg.draw.rect(self._screen, COLOR_ZONE_INACTIVE, rect)

            # Zone border
            pg.draw.rect(self._screen, COLOR_GRID_DIVIDER, rect, 1)

        # Outer border
        outer = pg.Rect(AREA_LEFT, AREA_TOP, AREA_WIDTH, AREA_HEIGHT)
        pg.draw.rect(self._screen, COLOR_GRID, outer, 2)

    def _draw_links(self, link_states: dict[str, dict]) -> None:
        """Draw links between nodes — colored by state."""
        pg = self._pg

        for n1, n2 in LINK_DEFS:
            link_id = LINK_TO_ID[(n1, n2)]
            p1 = self._node_positions[n1]
            p2 = self._node_positions[n2]

            state_info = link_states.get(link_id, {})
            is_motion = state_info.get("state") == "MOTION"

            color = COLOR_LINK_MOTION if is_motion else COLOR_LINK_IDLE
            width = 3 if is_motion else 1

            pg.draw.line(self._screen, color, p1, p2, width)

    def _draw_nodes(self) -> None:
        """Draw node circles at corners with labels."""
        pg = self._pg

        for name, pos in self._node_positions.items():
            pg.draw.circle(self._screen, COLOR_NODE, pos, 12)
            pg.draw.circle(self._screen, COLOR_BG, pos, 9)

            # Label offset — push labels outside the area
            ox, oy = 0, 0
            if name == "A":
                ox, oy = -20, -22
            elif name == "B":
                ox, oy = 10, -22
            elif name == "C":
                ox, oy = -20, 10
            elif name == "D":
                ox, oy = 10, 10

            label = self._font_node.render(name, True, COLOR_NODE)
            self._screen.blit(label, (pos[0] + ox, pos[1] + oy))

    def _draw_zone_labels(self, zone_result: ZoneResult) -> None:
        """Draw Q1-Q4 labels centered in each quadrant."""
        from zone_detector import Zone

        for zone in Zone:
            frac_x, frac_y = ZONE_LABELS[zone.value]
            cx = AREA_LEFT + int(frac_x * AREA_WIDTH)
            cy = AREA_TOP + int(frac_y * AREA_HEIGHT)

            is_active = zone_result.zone == zone

            if is_active:
                color = COLOR_TEXT
                font = self._font_zone
            else:
                color = COLOR_TEXT_DIM
                font = self._font_medium

            label = font.render(zone.value, True, color)
            rect = label.get_rect(center=(cx, cy))
            self._screen.blit(label, rect)

            # Show score below label
            score = zone_result.scores.get(zone, 0.0)
            score_text = self._font_small.render(
                f"{score:.4f}", True, COLOR_TEXT_DIM
            )
            score_rect = score_text.get_rect(center=(cx, cy + 24))
            self._screen.blit(score_text, score_rect)

    def _draw_sidebar(
        self,
        room_state: RoomState,
        zone_result: ZoneResult,
        link_states: dict[str, dict],
    ) -> None:
        """Draw the right sidebar with per-link detail and confidence."""
        pg = self._pg
        from presence_detector import RoomState as RS

        # Sidebar background
        sidebar_rect = pg.Rect(
            SIDEBAR_LEFT, SIDEBAR_TOP, SIDEBAR_WIDTH, AREA_HEIGHT
        )
        pg.draw.rect(self._screen, COLOR_SIDEBAR_BG, sidebar_rect)
        pg.draw.rect(self._screen, COLOR_GRID_DIVIDER, sidebar_rect, 1)

        y = SIDEBAR_TOP + 10
        pad = SIDEBAR_LEFT + 10

        # Zone info
        header = self._font_medium.render("Zone Info", True, COLOR_TEXT_LABEL)
        self._screen.blit(header, (pad, y))
        y += 28

        if zone_result.zone is not None:
            zone_text = self._font_medium.render(
                f"Zone: {zone_result.zone.value}", True, COLOR_TEXT
            )
            self._screen.blit(zone_text, (pad, y))
            y += 24

            if zone_result.confidence == float("inf"):
                conf_str = "∞"
            else:
                conf_str = f"{zone_result.confidence:.1f}x"

            conf_color = (
                COLOR_CONFIDENCE_HIGH
                if zone_result.confidence > 2.0
                else COLOR_CONFIDENCE_LOW
            )
            conf_text = self._font_small.render(
                f"Confidence: {conf_str}", True, conf_color
            )
            self._screen.blit(conf_text, (pad, y))
            y += 24
        else:
            no_zone = self._font_small.render("No zone", True, COLOR_TEXT_DIM)
            self._screen.blit(no_zone, (pad, y))
            y += 24

        y += 16

        # Link states
        header2 = self._font_medium.render("Link States", True, COLOR_TEXT_LABEL)
        self._screen.blit(header2, (pad, y))
        y += 28

        for link_id in ["AB", "AC", "AD", "BC", "BD", "CD"]:
            info = link_states.get(link_id, {})
            state_str = info.get("state", "?")
            variance = info.get("variance", 0.0)
            wf = info.get("window_full", False)

            if state_str == "MOTION":
                color = COLOR_LINK_MOTION
                indicator = "●"
            else:
                color = COLOR_TEXT_DIM
                indicator = "○"

            line_text = f"{indicator} {link_id}  {state_str:<6s} v={variance:.5f}"
            if not wf:
                line_text += " [fill]"

            text_surf = self._font_small.render(line_text, True, color)
            self._screen.blit(text_surf, (pad, y))
            y += 20

    def _draw_fps(self, fps: float) -> None:
        """Draw FPS counter in bottom-right corner."""
        fps_text = self._font_small.render(f"{fps:.0f} FPS", True, COLOR_FPS)
        self._screen.blit(
            fps_text,
            (WINDOW_WIDTH - 70, WINDOW_HEIGHT - 25),
        )
