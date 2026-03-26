"""widgets.py — Reusable UI components for GHV4."""
from __future__ import annotations

import queue

import customtkinter as ctk

from ghv5.config import MAX_LOG_LINES


class PortDropdown(ctk.CTkFrame):
    """COM port combobox with Refresh button."""

    def __init__(self, parent, label: str = "PORT", **kwargs):
        super().__init__(parent, **kwargs)
        ctk.CTkLabel(self, text=label, font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 8))
        self._var = ctk.StringVar(value="COM3")
        self._combo = ctk.CTkComboBox(row, variable=self._var, width=140)
        self._combo.pack(side="left")
        ctk.CTkButton(
            row, text="Refresh", width=70, command=self.refresh
        ).pack(side="left", padx=(8, 0))

    @property
    def port(self) -> str:
        return self._var.get().strip()

    @port.setter
    def port(self, value: str) -> None:
        self._var.set(value)

    @property
    def var(self) -> ctk.StringVar:
        return self._var

    @property
    def combo(self) -> ctk.CTkComboBox:
        return self._combo

    def refresh(self) -> None:
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
        self._combo.configure(values=ports)
        if not self._var.get() and ports:
            self._var.set(ports[0])

    def refresh_with_ports(self, ports: list[str]) -> None:
        """Refresh from an externally-fetched port list (avoids redundant enumeration)."""
        self._combo.configure(values=ports)
        if not self._var.get() and ports:
            self._var.set(ports[0])


class LogPanel(ctk.CTkFrame):
    """Scrollable text area with thread-safe append and line limit enforcement."""

    def __init__(self, parent, label: str = "LOG", height: int = 130, **kwargs):
        super().__init__(parent, **kwargs)
        ctk.CTkLabel(self, text=label, font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        self._textbox = ctk.CTkTextbox(
            self, height=height, font=("Courier", 11), state="disabled"
        )
        self._textbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def append(self, msg: str, tag: str | None = None) -> None:
        """Append a line, enforce MAX_LOG_LINES, scroll to end."""
        tb = self._textbox
        tb.configure(state="normal")
        if tag and hasattr(tb, '_textbox'):
            tb._textbox.insert("end", msg + "\n", tag)
        else:
            tb.insert("end", msg + "\n")
        line_count = int(tb.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            tb.delete("1.0", f"{line_count - MAX_LOG_LINES + 1}.0")
        tb.configure(state="disabled")
        tb.see("end")

    @property
    def textbox(self) -> ctk.CTkTextbox:
        return self._textbox


class StatusLabel(ctk.CTkLabel):
    """Colored status text label."""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("text", "Not connected")
        kwargs.setdefault("text_color", "#888888")
        kwargs.setdefault("font", ("", 11))
        super().__init__(parent, **kwargs)

    def set_status(self, text: str, color: str = "#888888") -> None:
        self.configure(text=text, text_color=color)
