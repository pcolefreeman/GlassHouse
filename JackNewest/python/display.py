"""Simple Pygame display for zone localization demo."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zone_detector import ZoneResult

# On Raspberry Pi (Linux framebuffer), use fbcon for direct LCD output.
# On Windows/macOS, let SDL pick the native video driver.
import sys
if sys.platform == "linux" and os.environ.get("SDL_VIDEODRIVER") is None:
    os.environ.setdefault("SDL_VIDEODRIVER", "fbcon")

import pygame

# Room layout colors
BG_COLOR = (30, 30, 30)
EMPTY_COLOR = (60, 60, 60)
ZONE_COLOR = (0, 180, 80)
TEXT_COLOR = (255, 255, 255)
GRID_COLOR = (100, 100, 100)

# Zone quadrant screen positions (x, y, w, h) for a 320x240 display
ZONE_RECTS = {
    "Q1": (10, 10, 145, 105),    # top-left
    "Q2": (165, 10, 145, 105),   # top-right
    "Q3": (10, 125, 145, 105),   # bottom-left
    "Q4": (165, 125, 145, 105),  # bottom-right
}


class Display:
    """Renders zone localization state on an LCD via Pygame."""

    def __init__(self, width: int = 320, height: int = 240) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("GlassHouse v2")
        self._font = pygame.font.SysFont("monospace", 20)
        self._small_font = pygame.font.SysFont("monospace", 14)

    def update(self, result: ZoneResult, occupied: bool) -> None:
        self._screen.fill(BG_COLOR)

        for zone_name, rect in ZONE_RECTS.items():
            color = EMPTY_COLOR
            if occupied and result.zone is not None and result.zone.value == zone_name:
                color = ZONE_COLOR
            pygame.draw.rect(self._screen, color, rect)
            pygame.draw.rect(self._screen, GRID_COLOR, rect, 1)
            label = self._small_font.render(zone_name, True, TEXT_COLOR)
            self._screen.blit(label, (rect[0] + 5, rect[1] + 5))

        # Status text
        status = "OCCUPIED" if occupied else "EMPTY"
        zone_str = result.zone.value if result.zone else "---"
        text = self._font.render(f"{status} | Zone: {zone_str}", True, TEXT_COLOR)
        self._screen.blit(text, (10, 240 - 30))

        pygame.display.flip()

        # Process events to prevent OS hang
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

    def close(self) -> None:
        pygame.quit()
