"""app.py — GHV4 application shell and tab router."""
from __future__ import annotations

import datetime
import os
import sys
import traceback

import customtkinter as ctk

from ghv4.ui.capture_tab import CaptureTab
from ghv4.ui.debug_tab import ListenerDebugTab, ShouterDebugTab


def _install_crash_logger() -> None:
    """Write unhandled exceptions to ghv4_crash.log."""
    if getattr(sys, "frozen", False):
        log_dir = os.path.dirname(sys.executable)
    else:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, "ghv4_crash.log")

    _original = sys.excepthook

    def _handler(exc_type, exc_value, exc_tb):
        with open(log_path, "a") as f:
            f.write(f"\n--- {datetime.datetime.now()} ---\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        _original(exc_type, exc_value, exc_tb)

    sys.excepthook = _handler


class App(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title("GlassHouse V3.1 — Data Collection")
        self.resizable(True, True)

        tabs = ctk.CTkTabview(self, anchor="nw")
        tabs.pack(fill="both", expand=True)

        cap_frame = tabs.add("  Capture  ")
        dbg_frame = tabs.add("  Debug  ")

        # Sub-tabs for debug
        dbg_tabs = ctk.CTkTabview(dbg_frame, anchor="nw")
        dbg_tabs.pack(fill="both", expand=True)
        lst_frame = dbg_tabs.add("  Listener  ")
        sht_frame = dbg_tabs.add("  Shouter  ")

        # Mount tabs
        self._capture_tab = CaptureTab(cap_frame)
        self._capture_tab.pack(fill="both", expand=True)

        self._listener_tab = ListenerDebugTab(lst_frame)
        self._listener_tab.pack(fill="both", expand=True)

        self._shouter_tab = ShouterDebugTab(sht_frame)
        self._shouter_tab.pack(fill="both", expand=True)

        # Wire ranging-restart detection to capture tab MUSIC reset
        self._listener_tab.set_reset_music_callback(self._capture_tab.reset_music)

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self) -> None:
        """Stop all threads and close serial ports."""
        self._capture_tab.stop()
        self._listener_tab.stop()
        self._shouter_tab.stop()
        self.destroy()


def main():
    _install_crash_logger()
    app = App()
    app.mainloop()
