# ghv2_ui.py
"""GlassHouseV2 data collection GUI.

Usage:
    python ghv2_ui.py

Build:
    pyinstaller GHV2_Collector.spec
"""
from __future__ import annotations

import datetime
import os
import queue
import sys
import threading
import time
import traceback

import customtkinter as ctk
from tkinter import filedialog

import csi_parser
import GlassHouseV2 as ghv2
from ghv2_ui_logic import build_label, first_cell, validate_depth, validate_width, validate_zone


# ── Crash logger ─────────────────────────────────────────────────────────────

def _install_crash_logger() -> None:
    """Write unhandled exceptions to ghv2_ui_crash.log before the window opens."""
    if getattr(sys, "frozen", False):
        log_dir = os.path.dirname(sys.executable)
    else:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, "ghv2_ui_crash.log")

    _original = sys.excepthook

    def _handler(exc_type, exc_value, exc_tb):
        with open(log_path, "a") as f:
            f.write(f"\n--- {datetime.datetime.now()} ---\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        _original(exc_type, exc_value, exc_tb)

    sys.excepthook = _handler


_install_crash_logger()


# ── Path helper ───────────────────────────────────────────────────────────────

def _resolve_output_dir(raw: str) -> str:
    """Make relative paths absolute, anchored to the exe/script directory.

    When the .exe is double-clicked, Python's CWD is unpredictable (often the
    user's home folder). Anchoring to the exe directory ensures that
    'data/processed' always resolves next to the executable.
    """
    if os.path.isabs(raw):
        return raw
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, raw)


# ── Colours ───────────────────────────────────────────────────────────────────

_MAX_LOG_LINES = 500

_GREEN_BG  = "#e6f5ec"
_GREEN_BD  = "#2d9a4a"
_GREEN_TXT = "#1a7a38"
_RED_BG    = "#fde8e8"
_NORMAL_BD = "#d0d8e4"


# ── BackgroundCaptureThread ───────────────────────────────────────────────────

class BackgroundCaptureThread(threading.Thread):
    """Runs SerialReader + CSVWriter from GlassHouseV2 in the background.

    Communicates with the UI via log_queue:
      - str  → append text to the log panel
      - None → sentinel: capture is done, re-enable Start button
    """

    BAUD_RATE = 921600

    def __init__(
        self,
        port: str,
        output_path: str,
        meta: dict,
        log_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True, name="BackgroundCapture")
        self._port        = port
        self._output_path = output_path
        self._meta        = meta
        self._log_queue   = log_queue
        self._stop_event  = threading.Event()

    def stop(self) -> None:
        """Signal the capture loop to stop cleanly."""
        self._stop_event.set()

    def _log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def run(self) -> None:
        import serial

        frame_queue: queue.Queue = queue.Queue()
        ser = None
        try:
            os.makedirs(
                os.path.dirname(os.path.abspath(self._output_path)),
                exist_ok=True,
            )
            ser = serial.Serial(self._port, self.BAUD_RATE, timeout=1)
            self._log(f"[GHV2] {self._port}  →  {self._output_path}")
            self._log(
                f"[GHV2] label={self._meta['label']}  "
                f"zone={self._meta['zone_id']}  "
                f"row={self._meta['grid_row']}  "
                f"col={self._meta['grid_col']}"
            )

            with open(self._output_path, "w", newline="") as f_out:
                reader = ghv2.SerialReader(ser, frame_queue)
                writer = ghv2.CSVWriter(frame_queue, f_out)
                reader.start()
                writer.start()

                while not self._stop_event.is_set():
                    time.sleep(csi_parser.BUCKET_MS / 1000.0)
                    frame_queue.put(("flush", dict(self._meta)))
                    # Auto-stop if the serial reader died (e.g. device disconnected)
                    if not reader.is_alive():
                        self._log("[GHV2] Serial reader stopped — capture ended.")
                        break

                # ── clean shutdown ────────────────────────────────────────
                reader.stop()
                frame_queue.put(("flush", dict(self._meta)))  # final bucket
                frame_queue.put(None)                          # CSVWriter exit
                writer.join(timeout=2)
                if writer.is_alive():
                    self._log(
                        "[GHV2] WARNING: CSV writer did not finish in time"
                        " — file may be incomplete."
                    )

            self._log(f"[GHV2] Saved to {self._output_path}")

        except Exception:
            self._log(f"[ERROR] {traceback.format_exc()}")
        finally:
            if ser and ser.is_open:
                ser.close()
            self._log_queue.put(self)  # sentinel → UI re-enables Start (identity check)


# ── App ───────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        # set_appearance_mode / set_default_color_theme MUST come before super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title("GlassHouse V2 — Data Collection")
        self.resizable(False, False)

        self._selected: set[tuple[int, int]] = set()
        self._cell_btns: dict[tuple[int, int], ctk.CTkButton] = {}
        self._log_queue: queue.Queue = queue.Queue()
        self._capture_thread: BackgroundCaptureThread | None = None

        self._build_ui()
        self._refresh_ports()
        self._poll_log()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        # ── Serial Port ───────────────────────────────────────────────────────
        pf = ctk.CTkFrame(self)
        pf.pack(fill="x", **pad)
        ctk.CTkLabel(pf, text="SERIAL PORT", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        port_row = ctk.CTkFrame(pf, fg_color="transparent")
        port_row.pack(fill="x", padx=8, pady=(0, 8))
        self._port_var = ctk.StringVar(value="COM3")
        self._port_combo = ctk.CTkComboBox(port_row, variable=self._port_var, width=140)
        self._port_combo.pack(side="left")
        ctk.CTkButton(
            port_row, text="Refresh", width=80, command=self._refresh_ports
        ).pack(side="left", padx=(8, 0))

        # ── Output Folder ─────────────────────────────────────────────────────
        ff = ctk.CTkFrame(self)
        ff.pack(fill="x", **pad)
        ctk.CTkLabel(ff, text="OUTPUT FOLDER", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        folder_row = ctk.CTkFrame(ff, fg_color="transparent")
        folder_row.pack(fill="x", padx=8, pady=(0, 8))
        self._folder_var = ctk.StringVar(value="data/processed")
        self._folder_entry = ctk.CTkEntry(
            folder_row, textvariable=self._folder_var, width=320
        )
        self._folder_entry.pack(side="left")
        ctk.CTkButton(
            folder_row, text="Browse…", width=80, command=self._browse_folder
        ).pack(side="left", padx=(8, 0))

        # ── Area Dimensions & Zone ────────────────────────────────────────────
        df = ctk.CTkFrame(self)
        df.pack(fill="x", **pad)
        ctk.CTkLabel(df, text="AREA DIMENSIONS & ZONE", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        dims_row = ctk.CTkFrame(df, fg_color="transparent")
        dims_row.pack(fill="x", padx=8, pady=(0, 8))
        self._width_var  = ctk.StringVar(value="")
        self._depth_var  = ctk.StringVar(value="")
        self._zone_var   = ctk.StringVar(value="0")
        self._width_entry = ctk.CTkEntry(
            dims_row, textvariable=self._width_var, width=90, placeholder_text="Width (m)"
        )
        self._depth_entry = ctk.CTkEntry(
            dims_row, textvariable=self._depth_var, width=90, placeholder_text="Depth (m)"
        )
        self._zone_entry = ctk.CTkEntry(
            dims_row, textvariable=self._zone_var, width=90, placeholder_text="Zone ID"
        )
        self._width_entry.pack(side="left")
        self._depth_entry.pack(side="left", padx=(8, 0))
        self._zone_entry.pack(side="left",  padx=(8, 0))

        # ── Occupancy Grid ────────────────────────────────────────────────────
        gf = ctk.CTkFrame(self)
        gf.pack(fill="x", **pad)
        ctk.CTkLabel(gf, text="OCCUPANCY LABEL", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        grid_body = ctk.CTkFrame(gf, fg_color="transparent")
        grid_body.pack(fill="x", padx=8, pady=(0, 8))

        cell_frame = ctk.CTkFrame(grid_body, fg_color="transparent")
        cell_frame.pack(side="left")
        for r in range(3):
            for c in range(3):
                btn = ctk.CTkButton(
                    cell_frame,
                    text=f"r{r}c{c}",
                    width=56, height=56,
                    fg_color="white",
                    text_color="#999999",
                    border_width=2,
                    border_color=_NORMAL_BD,
                    corner_radius=8,
                    hover_color="#f0f0f0",
                    command=lambda row=r, col=c: self._toggle_cell(row, col),
                )
                btn.grid(row=r, column=c, padx=3, pady=3)
                self._cell_btns[(r, c)] = btn
        ctk.CTkButton(
            cell_frame,
            text="Clear (Empty)",
            width=188, height=28,
            fg_color="#f0f0f0",
            text_color="#555555",
            hover_color="#e0e0e0",
            border_width=1,
            border_color="#cccccc",
            command=self._clear_cells,
        ).grid(row=3, column=0, columnspan=3, pady=(6, 0))

        info_col = ctk.CTkFrame(grid_body, fg_color="transparent")
        info_col.pack(side="left", padx=(16, 0), anchor="n")

        ctk.CTkLabel(info_col, text="Generated Label", font=("", 11)).pack(anchor="w")
        self._label_display = ctk.CTkLabel(
            info_col,
            text="empty",
            width=140,
            anchor="w",
            text_color=_GREEN_TXT,
            fg_color=_GREEN_BG,
            corner_radius=6,
        )
        self._label_display.pack(anchor="w", ipadx=6, ipady=4)

        ctk.CTkLabel(info_col, text="Row / Col (auto)", font=("", 11)).pack(
            anchor="w", pady=(10, 0)
        )
        rc_row = ctk.CTkFrame(info_col, fg_color="transparent")
        rc_row.pack(anchor="w")
        self._row_display = ctk.CTkLabel(rc_row, text="0", width=50, anchor="center",
                                         fg_color="#eeeeee", corner_radius=6)
        self._col_display = ctk.CTkLabel(rc_row, text="0", width=50, anchor="center",
                                         fg_color="#eeeeee", corner_radius=6)
        self._row_display.pack(side="left")
        self._col_display.pack(side="left", padx=(6, 0))
        ctk.CTkLabel(
            info_col, text="from first selected cell", font=("", 10), text_color="#aaaaaa"
        ).pack(anchor="w")

        # ── Start / Stop ──────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=6)
        self._start_btn = ctk.CTkButton(
            btn_frame,
            text="▶  Start Capture",
            fg_color="#2d9a4a",
            hover_color="#1f7a38",
            text_color="white",
            font=("", 13, "bold"),
            command=self._start_capture,
        )
        self._stop_btn = ctk.CTkButton(
            btn_frame,
            text="■  Stop",
            fg_color="#e0e8f0",
            hover_color="#e0e8f0",
            text_color="#aaaaaa",
            font=("", 13, "bold"),
            state="disabled",
            command=self._stop_capture,
        )
        self._start_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._stop_btn.pack(side="left",  expand=True, fill="x", padx=(4, 0))

        # ── Log ───────────────────────────────────────────────────────────────
        lf = ctk.CTkFrame(self)
        lf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ctk.CTkLabel(lf, text="LOG", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        self._log_text = ctk.CTkTextbox(
            lf, height=130, font=("Courier", 11), state="disabled"
        )
        self._log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _refresh_ports(self) -> None:
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
        self._port_combo.configure(values=ports)
        if not self._port_var.get() and ports:
            self._port_var.set(ports[0])

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self._folder_var.set(path)

    def _toggle_cell(self, row: int, col: int) -> None:
        key = (row, col)
        if key in self._selected:
            self._selected.discard(key)
            self._cell_btns[key].configure(
                fg_color="white", text_color="#999999", border_color=_NORMAL_BD
            )
        else:
            self._selected.add(key)
            self._cell_btns[key].configure(
                fg_color=_GREEN_BG, text_color=_GREEN_TXT, border_color=_GREEN_BD
            )
        self._sync_label()

    def _clear_cells(self) -> None:
        for key in list(self._selected):
            self._cell_btns[key].configure(
                fg_color="white", text_color="#999999", border_color=_NORMAL_BD
            )
        self._selected.clear()
        self._sync_label()

    def _sync_label(self) -> None:
        self._label_display.configure(text=build_label(self._selected))
        r, c = first_cell(self._selected)
        self._row_display.configure(text=str(r))
        self._col_display.configure(text=str(c))

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_and_collect(self) -> dict | None:
        """Validate all fields. Highlight invalid ones red. Return args dict or None."""
        ok = True

        port = self._port_var.get().strip()
        if not port:
            self._port_combo.configure(border_color="#e74c3c")
            ok = False
        else:
            self._port_combo.configure(border_color=_NORMAL_BD)

        width_s = self._width_var.get().strip()
        depth_s = self._depth_var.get().strip()
        zone_s  = self._zone_var.get().strip()

        width = None
        if width_s:
            width = validate_width(width_s)
            self._width_entry.configure(fg_color="white" if width is not None else _RED_BG)
            if width is None:
                ok = False
        else:
            self._width_entry.configure(fg_color="white")

        depth = None
        if depth_s:
            depth = validate_depth(depth_s)
            self._depth_entry.configure(fg_color="white" if depth is not None else _RED_BG)
            if depth is None:
                ok = False
        else:
            self._depth_entry.configure(fg_color="white")

        # Both dimensions must be provided together — one alone is silently dropped from
        # the filename, which is confusing.
        if (width is None) != (depth is None):
            self._width_entry.configure(fg_color=_RED_BG if width is None else "white")
            self._depth_entry.configure(fg_color=_RED_BG if depth is None else "white")
            ok = False

        zone = validate_zone(zone_s) if zone_s else 0
        if zone is None:
            self._zone_entry.configure(fg_color=_RED_BG)
            ok = False
        else:
            self._zone_entry.configure(fg_color="white")

        if not ok:
            return None

        r, c = first_cell(self._selected)
        return {
            "port":     port,
            "out_dir":  _resolve_output_dir(self._folder_var.get().strip()),
            "width":    width,
            "depth":    depth,
            "label":    build_label(self._selected),
            "zone_id":  zone,
            "grid_row": r,
            "grid_col": c,
        }

    # ── Capture control ───────────────────────────────────────────────────────

    def _start_capture(self) -> None:
        args = self._validate_and_collect()
        if args is None:
            return

        output_path = ghv2.build_output_filename(
            args["out_dir"], args["width"], args["depth"]
        )
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._append_log(f"--- Capture started {ts} ---")

        meta = {k: args[k] for k in ("label", "zone_id", "grid_row", "grid_col")}
        self._capture_thread = BackgroundCaptureThread(
            port=args["port"],
            output_path=output_path,
            meta=meta,
            log_queue=self._log_queue,
        )
        self._capture_thread.start()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(
            state="normal",
            fg_color="#c0392b",
            hover_color="#9b2820",
            text_color="white",
        )

    def _stop_capture(self) -> None:
        if self._capture_thread:
            self._capture_thread.stop()

    def _reset_buttons(self) -> None:
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(
            state="disabled",
            fg_color="#e0e8f0",
            hover_color="#e0e8f0",
            text_color="#aaaaaa",
        )
        self._capture_thread = None

    # ── Log polling ───────────────────────────────────────────────────────────

    def _poll_log(self) -> None:
        try:
            while True:
                item = self._log_queue.get_nowait()
                if isinstance(item, str):
                    self._append_log(item)
                elif item is self._capture_thread:
                    self._reset_buttons()
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _append_log(self, msg: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        line_count = int(self._log_text.index("end-1c").split(".")[0])
        if line_count > _MAX_LOG_LINES:
            self._log_text.delete("1.0", f"{line_count - _MAX_LOG_LINES + 1}.0")
        self._log_text.configure(state="disabled")
        self._log_text.see("end")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
