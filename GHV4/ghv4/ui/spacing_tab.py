"""spacing_tab.py — Shouter distance cards for GHV4 UI."""
from __future__ import annotations

import threading

import customtkinter as ctk

from ghv4.config import PAIR_KEYS

_GREEN_TXT = "#1a7a38"


class SpacingCards(ctk.CTkFrame):
    """Six distance pair cards arranged in a 2x3 grid."""

    def __init__(self, parent, label: str = "SHOUTER DISTANCES", **kwargs):
        super().__init__(parent, **kwargs)
        ctk.CTkLabel(self, text=label, font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(fill="x", padx=8, pady=(0, 8))
        for col_i in range(3):
            grid.grid_columnconfigure(col_i, weight=1)

        self._labels: dict[str, ctk.CTkLabel] = {}
        self._lock = threading.Lock()

        for i, key in enumerate(PAIR_KEYS):
            cell = ctk.CTkFrame(grid, fg_color="#f0f4f8", corner_radius=6)
            cell.grid(row=i // 3, column=i % 3, padx=4, pady=3, sticky="ew")
            ctk.CTkLabel(
                cell, text=key, font=("", 10, "bold"), text_color="#555555"
            ).pack(side="left", padx=(8, 4), pady=4)
            val_lbl = ctk.CTkLabel(
                cell, text="--", font=("Courier", 11), text_color="#aaaaaa"
            )
            val_lbl.pack(side="left", padx=(0, 8), pady=4)
            self._labels[key] = val_lbl

    def update_distances(self, distances: dict, rssi_values: dict | None = None) -> None:
        """Refresh card values. Thread-safe."""
        with self._lock:
            for key, lbl in self._labels.items():
                d = distances.get(key)
                if d is None:
                    lbl.configure(text="--", text_color="#aaaaaa")
                else:
                    rssi_str = ""
                    if rssi_values:
                        r = rssi_values.get(key)
                        if r is not None:
                            rssi_str = f" ({r:.0f})"
                    lbl.configure(text=f"{d:.2f} m{rssi_str}", text_color=_GREEN_TXT)
