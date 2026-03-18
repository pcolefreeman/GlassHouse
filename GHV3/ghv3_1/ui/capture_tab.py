"""capture_tab.py -- Data collection tab for GHV3.1 UI."""
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

from ghv3_1 import csi_parser
from ghv3_1 import serial_io
from ghv3_1.config import BAUD_RATE, BUCKET_MS, MAX_LOG_LINES
from ghv3_1.cell_logic import build_label, first_cell, validate_zone
from ghv3_1.ui.spacing_tab import SpacingCards


# -- Colours -------------------------------------------------------------------

_GREEN_BG  = "#e6f5ec"
_GREEN_BD  = "#2d9a4a"
_GREEN_TXT = "#1a7a38"
_RED_BG    = "#fde8e8"
_NORMAL_BD = "#d0d8e4"


# -- Path helper ---------------------------------------------------------------

def _resolve_output_dir(raw: str) -> str:
    """Make relative paths absolute, anchored to the exe/script directory."""
    if os.path.isabs(raw):
        return raw
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, raw)


# -- BackgroundCaptureThread ---------------------------------------------------

class BackgroundCaptureThread(threading.Thread):
    """Runs SerialReader + CSVWriter from serial_io in the background."""

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
        self._log_queue         = log_queue
        self._stop_event        = threading.Event()
        self._spacing_estimator = None
        self._music_estimator   = None

    def stop(self) -> None:
        self._stop_event.set()

    def reset_music(self) -> None:
        """Reset MUSIC snapshot buffers. Call before a new ranging phase."""
        if self._music_estimator is not None:
            self._music_estimator.reset_all()

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
            from ghv3_1.spacing_estimator import SpacingEstimator, CSIMUSICEstimator

            ser = serial.Serial(self._port, BAUD_RATE, timeout=1)
            self._log(f"[GHV3] {self._port}  ->  {self._output_path}")
            self._log(
                f"[GHV3] label={self._meta['label']}  "
                f"zone={self._meta['zone_id']}  "
                f"row={self._meta['grid_row']}  "
                f"col={self._meta['grid_col']}  "
                f"activity={self._meta.get('activity', 'none')}"
            )

            music_est = CSIMUSICEstimator()
            self._music_estimator = music_est
            spacing_est = SpacingEstimator(
                spacing_path=os.path.join(
                    os.path.dirname(os.path.abspath(self._output_path)), "spacing.json"
                ),
                music_estimator=music_est,
            )
            spacing_est.start()
            self._spacing_estimator = spacing_est

            with open(self._output_path, "w", newline="") as f_out:
                reader = serial_io.SerialReader(ser, frame_queue, music_estimator=music_est)
                writer = serial_io.CSVWriter(frame_queue, f_out, spacing_estimator=spacing_est)
                reader.start()
                writer.start()

                while not self._stop_event.is_set():
                    time.sleep(BUCKET_MS / 1000.0)
                    frame_queue.put(("flush", dict(self._meta)))
                    if not reader.is_alive():
                        self._log("[GHV3] Serial reader stopped -- capture ended.")
                        break

                reader.stop()
                frame_queue.put(("flush", dict(self._meta)))
                frame_queue.put(None)
                writer.join(timeout=2)
                if writer.is_alive():
                    self._log(
                        "[GHV3] WARNING: CSV writer did not finish in time"
                        " -- file may be incomplete."
                    )

            self._log(f"[GHV3] Saved to {self._output_path}")

        except Exception:
            self._log(f"[ERROR] {traceback.format_exc()}")
        finally:
            if ser and ser.is_open:
                ser.close()
            self._log_queue.put(self)


# -- CaptureTab ----------------------------------------------------------------

class CaptureTab(ctk.CTkFrame):
    """Data collection tab: port selection, grid labels, capture controls, log."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Capture state
        self._selected: set[tuple[int, int]] = set()
        self._cell_btns: dict[tuple[int, int], ctk.CTkButton] = {}
        self._activity: str = "none"
        self._activity_btns: dict[str, ctk.CTkButton] = {}
        self._log_queue: queue.Queue = queue.Queue()
        self._capture_thread: BackgroundCaptureThread | None = None
        self._capture_end_time: float | None = None

        self._build_ui()
        self._refresh_ports()
        self._poll_log()
        self._update_distances()

    # -- UI construction -------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        # Serial Port
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

        # Output Folder
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
            folder_row, text="Browse...", width=80, command=self._browse_folder
        ).pack(side="left", padx=(8, 0))

        # Zone
        df = ctk.CTkFrame(self)
        df.pack(fill="x", **pad)
        ctk.CTkLabel(df, text="ZONE", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        dims_row = ctk.CTkFrame(df, fg_color="transparent")
        dims_row.pack(fill="x", padx=8, pady=(0, 8))
        self._zone_var   = ctk.StringVar(value="0")
        self._zone_entry = ctk.CTkEntry(
            dims_row, textvariable=self._zone_var, width=90, placeholder_text="Zone ID"
        )
        self._zone_entry.pack(side="left")

        # Occupancy Grid
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

        # Activity
        af = ctk.CTkFrame(self)
        af.pack(fill="x", **pad)
        ctk.CTkLabel(af, text="ACTIVITY", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        act_row = ctk.CTkFrame(af, fg_color="transparent")
        act_row.pack(fill="x", padx=8, pady=(0, 8))
        for activity in ("sitting", "standing", "moving", "covered"):
            btn = ctk.CTkButton(
                act_row,
                text=activity.capitalize(),
                width=90, height=32,
                fg_color="white",
                text_color="#999999",
                border_width=2,
                border_color=_NORMAL_BD,
                corner_radius=8,
                hover_color="#f0f0f0",
                command=lambda a=activity: self._toggle_activity(a),
            )
            btn.pack(side="left", padx=(0, 6))
            self._activity_btns[activity] = btn

        # Run Duration
        dur_frame = ctk.CTkFrame(self, fg_color="transparent")
        dur_frame.pack(fill="x", padx=12, pady=(2, 2))
        ctk.CTkLabel(dur_frame, text="Run Duration (s):", font=("", 11)).pack(side="left")
        self._duration_var = ctk.StringVar(value="")
        self._duration_entry = ctk.CTkEntry(
            dur_frame, textvariable=self._duration_var, width=80, placeholder_text="inf"
        )
        self._duration_entry.pack(side="left", padx=(8, 0))
        ctk.CTkLabel(
            dur_frame, text="(blank = run until stopped)", font=("", 10), text_color="#aaaaaa"
        ).pack(side="left", padx=(8, 0))
        self._countdown_label = ctk.CTkLabel(
            dur_frame, text="", width=110, font=("Courier", 11, "bold"), text_color=_GREEN_TXT
        )
        self._countdown_label.pack(side="right")

        # Start / Stop
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=6)
        self._start_btn = ctk.CTkButton(
            btn_frame,
            text="Start Capture",
            fg_color="#2d9a4a",
            hover_color="#1f7a38",
            text_color="white",
            font=("", 13, "bold"),
            command=self._start_capture,
        )
        self._stop_btn = ctk.CTkButton(
            btn_frame,
            text="Stop",
            fg_color="#e0e8f0",
            hover_color="#e0e8f0",
            text_color="#aaaaaa",
            font=("", 13, "bold"),
            state="disabled",
            command=self._stop_capture,
        )
        self._start_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._stop_btn.pack(side="left",  expand=True, fill="x", padx=(4, 0))

        # Shouter distances
        self._spacing_cards = SpacingCards(self)
        self._spacing_cards.pack(fill="x", padx=12, pady=(0, 6))

        # Log
        lf = ctk.CTkFrame(self)
        lf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ctk.CTkLabel(lf, text="LOG", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        self._log_text = ctk.CTkTextbox(
            lf, height=130, font=("Courier", 11), state="disabled"
        )
        self._log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # -- Port helpers ----------------------------------------------------------

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

    # -- Cell grid callbacks ---------------------------------------------------

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

    def _toggle_activity(self, activity: str) -> None:
        if self._activity == activity:
            # Deselect — revert to default
            self._activity = "none"
            self._activity_btns[activity].configure(
                fg_color="white", text_color="#999999", border_color=_NORMAL_BD
            )
        else:
            # Deselect previous
            if self._activity in self._activity_btns:
                self._activity_btns[self._activity].configure(
                    fg_color="white", text_color="#999999", border_color=_NORMAL_BD
                )
            self._activity = activity
            self._activity_btns[activity].configure(
                fg_color=_GREEN_BG, text_color=_GREEN_TXT, border_color=_GREEN_BD
            )

    def _sync_label(self) -> None:
        self._label_display.configure(text=build_label(self._selected))
        r, c = first_cell(self._selected)
        self._row_display.configure(text=str(r))
        self._col_display.configure(text=str(c))

    # -- Validation & capture control ------------------------------------------

    def _validate_and_collect(self) -> dict | None:
        ok = True

        port = self._port_var.get().strip()
        if not port:
            self._port_combo.configure(border_color="#e74c3c")
            ok = False
        else:
            self._port_combo.configure(border_color=_NORMAL_BD)

        zone_s  = self._zone_var.get().strip()

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
            "label":    build_label(self._selected),
            "zone_id":  zone,
            "grid_row": r,
            "grid_col": c,
            "activity": self._activity,
        }

    def _start_capture(self) -> None:
        args = self._validate_and_collect()
        if args is None:
            return

        # Parse optional duration
        dur_s_str = self._duration_var.get().strip()
        dur_s: float | None = None
        if dur_s_str:
            try:
                dur_s = float(dur_s_str)
                if dur_s <= 0:
                    raise ValueError
                self._duration_entry.configure(fg_color="white")
            except ValueError:
                self._duration_entry.configure(fg_color=_RED_BG)
                return
        else:
            self._duration_entry.configure(fg_color="white")

        output_path = serial_io.build_output_filename(
            args["out_dir"], None, None
        )
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._append_log(f"--- Capture started {ts} ---")
        if dur_s is not None:
            self._append_log(f"[GHV3] Auto-stop in {dur_s:.0f}s")

        meta = {k: args[k] for k in ("label", "zone_id", "grid_row", "grid_col", "activity")}
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

        if dur_s is not None:
            self._capture_end_time = time.time() + dur_s
            self._tick_countdown()
        else:
            self._capture_end_time = None
            self._countdown_label.configure(text="")

    def _stop_capture(self) -> None:
        if self._capture_thread:
            self._capture_thread.stop()

    def _tick_countdown(self) -> None:
        if self._capture_end_time is None or self._capture_thread is None:
            self._countdown_label.configure(text="")
            return
        remaining = self._capture_end_time - time.time()
        if remaining <= 0:
            self._countdown_label.configure(text="Stopping...")
            self._capture_end_time = None
            self._stop_capture()
            return
        self._countdown_label.configure(text=f"{remaining:.0f}s remaining")
        self.after(500, self._tick_countdown)

    def _reset_buttons(self) -> None:
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(
            state="disabled",
            fg_color="#e0e8f0",
            hover_color="#e0e8f0",
            text_color="#aaaaaa",
        )
        self._capture_thread = None
        self._capture_end_time = None
        self._countdown_label.configure(text="")

    # -- Log polling -----------------------------------------------------------

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
        if line_count > MAX_LOG_LINES:
            self._log_text.delete("1.0", f"{line_count - MAX_LOG_LINES + 1}.0")
        self._log_text.configure(state="disabled")
        self._log_text.see("end")

    # -- Distance display ------------------------------------------------------

    def _update_distances(self) -> None:
        """Refresh the capture tab's SpacingCards from the active estimator."""
        est = None
        if self._capture_thread is not None:
            est = self._capture_thread._spacing_estimator

        dists = est.get_distances() if est is not None else {}
        rssis = est.get_rssi_values() if est is not None else {}
        self._spacing_cards.update_distances(dists, rssis)

        self.after(1000, self._update_distances)

    # -- Public API ------------------------------------------------------------

    def reset_music(self) -> None:
        """Reset MUSIC snapshot buffers on the active capture thread."""
        if self._capture_thread is not None:
            self._capture_thread.reset_music()

    def stop(self) -> None:
        """Stop the capture thread if running."""
        if self._capture_thread:
            self._capture_thread.stop()
