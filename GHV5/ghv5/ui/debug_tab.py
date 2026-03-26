"""debug_tab.py — Listener and shouter debug monitors for GHV4 UI."""
from __future__ import annotations

import os
import queue
import re
import struct
import sys
import threading
import time
import traceback
from collections import deque

import customtkinter as ctk

from ghv5 import csi_parser
from ghv5.config import BAUD_RATE, PAIR_KEYS, MAX_LOG_LINES, CSI_SNAP_HDR_SIZE
from ghv5.ui.widgets import LogPanel, PortDropdown, StatusLabel
from ghv5.ui.spacing_tab import SpacingCards


# ── Colours ───────────────────────────────────────────────────────────────────

_GREEN_BG  = "#e6f5ec"
_GREEN_BD  = "#2d9a4a"
_GREEN_TXT = "#1a7a38"
_RED_BG    = "#fde8e8"
_NORMAL_BD = "#d0d8e4"

# Regex for [LST] HELLO lines
_HELLO_RE = re.compile(
    r'\[LST\]\s+HELLO\s+sid=(\d+)\s+\(MAC-assigned\)\s+IP=([\d.]+)\s+MAC=([0-9A-Fa-f:]{17})',
    re.IGNORECASE,
)

# Table column definitions for the shouter debug table
_TBL_COLS = [
    ("ID",        50),
    ("MAC",       150),
    ("IP",        110),
    ("Last Seen", 80),
    ("RSSI",      55),
    ("NF",        55),
    ("Hits",      55),
    ("Misses",    55),
    ("Miss %",    60),
    ("Re-HELLO",  65),
]


# ── ListenerDebugThread ──────────────────────────────────────────────────────

class ListenerDebugThread(threading.Thread):
    """Monitors a listener COM port.

    Parses both binary frames ([0xAA][0x55], [0xBB][0xDD]) and text lines
    ([LST] HELLO ...) and puts typed dicts into debug_queue.

    Queue item types:
      {'type': 'status',         'msg': str}
      {'type': 'done'}
      {'type': 'lst_text',       'line': str}
      {'type': 'hello',          'sid': int, 'ip': str, 'mac': str, 'ts': float}
      {'type': 'listener_frame', 'frame': dict, 'ts': float}
      {'type': 'shouter_frame',  'frame': dict, 'ts': float}
      {'type': 'snap_frame',     'snap': dict, 'ts': float}
      {'type': 'ranging_start'}
    """

    def __init__(self, port: str, debug_queue: queue.Queue) -> None:
        super().__init__(daemon=True, name="ListenerDebug")
        self._port        = port
        self._queue       = debug_queue
        self._stop_event  = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import serial as _serial
        ser = None
        try:
            ser = _serial.Serial(self._port, BAUD_RATE, timeout=0.5)
            self._queue.put({'type': 'status', 'msg': f'[DBG] Listener connected on {self._port}'})
            while not self._stop_event.is_set():
                self._read_one(ser)
        except Exception:
            self._queue.put({'type': 'status', 'msg': f'[ERROR] {traceback.format_exc()}'})
        finally:
            if ser and ser.is_open:
                ser.close()
            self._queue.put({'type': 'done'})

    def _read_one(self, ser) -> None:
        b = ser.read(1)
        if not b:
            return
        b0 = b[0]

        if b0 == 0xAA:
            b1 = ser.read(1)
            if not b1 or b1[0] != 0x55:
                return
            hdr = ser.read(20)
            if len(hdr) < 20:
                return
            csi_len = struct.unpack_from('<H', hdr, 18)[0]
            csi = ser.read(csi_len)
            frame = csi_parser.parse_listener_frame(b'\xAA\x55' + hdr + csi, 0)
            if frame:
                self._queue.put({'type': 'listener_frame', 'frame': frame, 'ts': time.time()})

        elif b0 == 0xEE:                          # must be BEFORE >= 0x20
            b1 = ser.read(1)
            if not b1 or b1[0] != 0xFF:
                return
            # Read the 6-byte header (after magic): ver(1) reporter(1) peer(1) seq(1) csi_len(2)
            hdr = ser.read(CSI_SNAP_HDR_SIZE)
            if len(hdr) < CSI_SNAP_HDR_SIZE:
                return
            csi_len = struct.unpack_from('<H', hdr, 4)[0]
            csi = ser.read(csi_len)
            snap = csi_parser.parse_csi_snap_frame(hdr + csi)
            if snap:
                self._queue.put({'type': 'snap_frame', 'snap': snap, 'ts': time.time()})

        elif b0 == 0xBB:
            b1 = ser.read(1)
            if not b1 or b1[0] != 0xDD:
                return
            hdr = ser.read(29)
            if len(hdr) < 29:
                return
            csi_len = struct.unpack_from('<H', hdr, 27)[0]
            csi = ser.read(csi_len)
            frame = csi_parser.parse_shouter_frame(b'\xBB\xDD' + hdr + csi, 0)
            if frame:
                self._queue.put({'type': 'shouter_frame', 'frame': frame, 'ts': time.time()})

        elif b0 >= 0x20:
            # Printable — read to end-of-line
            rest = bytearray()
            while not self._stop_event.is_set():
                ch = ser.read(1)
                if not ch or ch[0] == ord('\n'):
                    break
                if ch[0] != ord('\r'):
                    rest.append(ch[0])
            # Suppress binary garbage (bootloader output, desync'd frame bytes).
            # All legitimate [LST] messages are pure ASCII (<=0x7E).
            if b0 > 0x7E or any(c > 0x7E for c in rest):
                return
            line = chr(b0) + rest.decode('ascii', errors='replace')
            if not line.lstrip('\r').startswith('['):
                return
            if line.strip():
                m = _HELLO_RE.search(line)
                if m:
                    self._queue.put({
                        'type': 'hello',
                        'sid':  int(m.group(1)),
                        'ip':   m.group(2),
                        'mac':  m.group(3).lower(),
                        'ts':   time.time(),
                    })
                if 'starting ranging' in line.lower():
                    self._queue.put({'type': 'ranging_start'})
                self._queue.put({'type': 'lst_text', 'line': line})


# ── ShouterDebugThread ───────────────────────────────────────────────────────

class ShouterDebugThread(threading.Thread):
    """Monitors a shouter COM port.

    Reads text lines and puts them into debug_queue.

    Queue item types:
      {'type': 'status',   'msg': str}
      {'type': 'done'}
      {'type': 'sht_line', 'line': str, 'ts': float}
    """

    def __init__(self, port: str, debug_queue: queue.Queue) -> None:
        super().__init__(daemon=True, name="ShouterDebug")
        self._port        = port
        self._queue       = debug_queue
        self._stop_event  = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import serial as _serial
        ser = None
        try:
            ser = _serial.Serial(self._port, BAUD_RATE, timeout=0.5)
            self._queue.put({'type': 'status', 'msg': f'[DBG] Shouter connected on {self._port}'})
            while not self._stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue
                # Suppress binary garbage (ESP32 bootloader bytes at wrong baud).
                # All legitimate [SHT] messages are pure ASCII (<=0x7E).
                if any(b > 0x7E for b in raw):
                    continue
                line = raw.decode('ascii', errors='replace').rstrip()
                if line:
                    self._queue.put({'type': 'sht_line', 'line': line, 'ts': time.time()})
        except Exception:
            self._queue.put({'type': 'status', 'msg': f'[ERROR] {traceback.format_exc()}'})
        finally:
            if ser and ser.is_open:
                ser.close()
            self._queue.put({'type': 'done'})


# ── ListenerDebugTab ─────────────────────────────────────────────────────────

class ListenerDebugTab(ctk.CTkFrame):
    """Listener debug panel — shouter table, spacing cards, stats, log."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Thread state
        self._debug_queue: queue.Queue = queue.Queue()
        self._listener_debug_thread: ListenerDebugThread | None = None

        # Shouter states with lock (race condition fix per spec Section 9)
        self._shouter_states_lock = threading.Lock()
        self._shouter_states: dict[int, dict] = {
            i: {
                'mac': '--', 'ip': '--', 'last_ts': 0.0,
                'rssi': None, 'nf': None,
                'hits': 0, 'misses': 0, 'reconnects': 0,
            }
            for i in range(1, 5)
        }
        self._tbl_labels: dict[int, dict[str, ctk.CTkLabel]] = {}

        # Frame counters
        self._lst_frame_times: deque = deque(maxlen=100)
        self._sht_frame_times: deque = deque(maxlen=100)
        self._lst_total = 0
        self._sht_total = 0
        self._snap_total = 0

        # MUSIC estimator (created immediately so snap frames can be collected)
        from ghv5.spacing_estimator import CSIMUSICEstimator
        self._music_estimator = CSIMUSICEstimator()

        self._reset_music_cb = None  # callable, set by App to reset capture tab MUSIC

        # Debug log file (opened on Connect, closed on Disconnect)
        self._lst_log_file = None

        self._build_ui()
        self._poll_debug_queue()
        self._update_debug_table()
        self._update_distances()

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        # Connection row
        cf = ctk.CTkFrame(self)
        cf.pack(fill="x", **pad)
        ctk.CTkLabel(cf, text="LISTENER PORT", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        conn_row = ctk.CTkFrame(cf, fg_color="transparent")
        conn_row.pack(fill="x", padx=8, pady=(0, 8))
        self._lst_port_var = ctk.StringVar(value="COM3")
        self._lst_port_combo = ctk.CTkComboBox(conn_row, variable=self._lst_port_var, width=140)
        self._lst_port_combo.pack(side="left")
        ctk.CTkButton(
            conn_row, text="Refresh", width=70, command=self._refresh_debug_ports
        ).pack(side="left", padx=(8, 0))
        self._lst_connect_btn = ctk.CTkButton(
            conn_row, text="Connect", width=80,
            fg_color="#2d9a4a", hover_color="#1f7a38", text_color="white",
            command=self._connect_listener_debug,
        )
        self._lst_connect_btn.pack(side="left", padx=(8, 0))
        self._lst_disconnect_btn = ctk.CTkButton(
            conn_row, text="Disconnect", width=90,
            fg_color="#e0e8f0", hover_color="#e0e8f0", text_color="#aaaaaa",
            state="disabled",
            command=self._disconnect_listener_debug,
        )
        self._lst_disconnect_btn.pack(side="left", padx=(6, 0))

        self._lst_status_label = ctk.CTkLabel(
            self, text="Not connected", text_color="#888888", font=("", 11)
        )
        self._lst_status_label.pack(anchor="w", padx=14, pady=(0, 4))

        # Shouter table
        tf = ctk.CTkFrame(self)
        tf.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(tf, text="VISIBLE SHOUTERS", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        tbl_frame = ctk.CTkFrame(tf, fg_color="transparent")
        tbl_frame.pack(fill="x", padx=8, pady=(0, 8))

        # Header row
        for col_idx, (col_name, col_w) in enumerate(_TBL_COLS):
            ctk.CTkLabel(
                tbl_frame, text=col_name, width=col_w,
                font=("", 10, "bold"), text_color="#555555",
                fg_color="#e8edf2", corner_radius=4, anchor="center",
            ).grid(row=0, column=col_idx, padx=2, pady=(0, 2), sticky="ew")

        # Data rows (1 per shouter ID)
        for row_idx, sid in enumerate(range(1, 5), start=1):
            self._tbl_labels[sid] = {}
            defaults = {
                'id':        str(sid),
                'mac':       '--',
                'ip':        '--',
                'last_seen': 'never',
                'rssi':      '--',
                'nf':        '--',
                'hits':      '0',
                'misses':    '0',
                'miss_pct':  '--',
                'reconnects':'0',
            }
            col_keys = ['id', 'mac', 'ip', 'last_seen', 'rssi', 'nf',
                        'hits', 'misses', 'miss_pct', 'reconnects']
            bg = "#f9f9f9" if row_idx % 2 == 0 else "white"
            for col_idx, (key, (_, col_w)) in enumerate(zip(col_keys, _TBL_COLS)):
                lbl = ctk.CTkLabel(
                    tbl_frame,
                    text=defaults[key],
                    width=col_w,
                    font=("Courier", 11),
                    fg_color=bg,
                    corner_radius=3,
                    anchor="center",
                )
                lbl.grid(row=row_idx, column=col_idx, padx=2, pady=1, sticky="ew")
                self._tbl_labels[sid][key] = lbl

        # Shouter distances (SpacingCards widget)
        self._spacing_cards = SpacingCards(self)
        self._spacing_cards.pack(fill="x", padx=12, pady=(0, 6))

        # Also keep a flat dict for the legacy _update_distances path
        self._dist_labels: dict[str, ctk.CTkLabel] = {}
        # Wire the SpacingCards internal labels into _dist_labels for direct access
        for key in PAIR_KEYS:
            self._dist_labels[key] = self._spacing_cards._labels[key]

        # Stats bar
        sf = ctk.CTkFrame(self, fg_color="#f0f4f8")
        sf.pack(fill="x", padx=12, pady=(0, 6))
        stats_inner = ctk.CTkFrame(sf, fg_color="transparent")
        stats_inner.pack(fill="x", padx=8, pady=4)

        self._lst_fps_label   = ctk.CTkLabel(stats_inner, text="Listener CSI: 0.0 fps",
                                              font=("Courier", 11), text_color="#444")
        self._sht_fps_label   = ctk.CTkLabel(stats_inner, text="Shouter poll: 0.0 fps",
                                              font=("Courier", 11), text_color="#444")
        self._lst_total_label = ctk.CTkLabel(stats_inner, text="LST frames: 0",
                                              font=("Courier", 11), text_color="#444")
        self._sht_total_label = ctk.CTkLabel(stats_inner, text="SHT frames: 0",
                                              font=("Courier", 11), text_color="#444")
        self._lst_fps_label.pack(side="left", padx=(0, 16))
        self._sht_fps_label.pack(side="left", padx=(0, 16))
        self._lst_total_label.pack(side="left", padx=(0, 16))
        self._sht_total_label.pack(side="left")

        # Log panel for [LST] text output
        lf = ctk.CTkFrame(self)
        lf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ctk.CTkLabel(lf, text="LISTENER LOG", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        self._lst_log_text = ctk.CTkTextbox(
            lf, height=120, font=("Courier", 11), state="disabled"
        )
        self._lst_log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── Port refresh ─────────────────────────────────────────────────────────

    def _refresh_debug_ports(self) -> None:
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
        self._lst_port_combo.configure(values=ports)
        if ports and not self._lst_port_var.get():
            self._lst_port_var.set(ports[0])

    # ── Connect / Disconnect ─────────────────────────────────────────────────

    def _connect_listener_debug(self) -> None:
        if self._listener_debug_thread and self._listener_debug_thread.is_alive():
            return
        port = self._lst_port_var.get().strip()
        if not port:
            return
        # Open debug log file
        if getattr(sys, "frozen", False):
            log_dir = os.path.dirname(sys.executable)
        else:
            log_dir = os.path.dirname(os.path.abspath(__file__))
        ts = time.strftime("%H%M%S")
        log_path = os.path.join(log_dir, f"listener_debug_{ts}.txt")
        try:
            self._lst_log_file = open(log_path, "w", encoding="utf-8", errors="replace")
            self._lst_log_file.write(
                f"GHV4 Listener Debug Log — {port} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "=" * 60 + "\n\n"
            )
            self._lst_log_file.flush()
        except OSError:
            self._lst_log_file = None
        self._listener_debug_thread = ListenerDebugThread(port, self._debug_queue)
        self._listener_debug_thread.start()
        self._lst_connect_btn.configure(state="disabled")
        self._lst_disconnect_btn.configure(
            state="normal", fg_color="#c0392b", hover_color="#9b2820", text_color="white"
        )
        msg = f"Connecting to {port}..."
        if self._lst_log_file:
            msg += f"  (log: {os.path.basename(log_path)})"
        self._lst_status_label.configure(text=msg, text_color="#888888")

    def _disconnect_listener_debug(self) -> None:
        if self._listener_debug_thread:
            self._listener_debug_thread.stop()

    def _reset_listener_debug_buttons(self) -> None:
        self._lst_connect_btn.configure(state="normal")
        self._lst_disconnect_btn.configure(
            state="disabled", fg_color="#e0e8f0", hover_color="#e0e8f0", text_color="#aaaaaa"
        )
        self._lst_status_label.configure(text="Disconnected", text_color="#888888")
        self._listener_debug_thread = None
        if self._lst_log_file:
            try:
                self._lst_log_file.close()
            except OSError:
                pass
            self._lst_log_file = None

    # ── Debug queue polling ──────────────────────────────────────────────────

    def _poll_debug_queue(self) -> None:
        try:
            while True:
                item = self._debug_queue.get_nowait()
                t = item.get('type')

                if t == 'status':
                    msg = item['msg']
                    if self._listener_debug_thread and self._listener_debug_thread.is_alive():
                        self._lst_status_label.configure(text=msg, text_color="#444444")
                    if '[DBG] Listener' in msg or '[LST]' in msg:
                        self._append_lst_log(msg)
                        self._lst_status_label.configure(text=msg, text_color="#1a7a38")
                    else:
                        self._append_lst_log(msg)

                elif t == 'done':
                    if self._listener_debug_thread and not self._listener_debug_thread.is_alive():
                        self._reset_listener_debug_buttons()

                elif t == 'hello':
                    sid = item['sid']
                    if 1 <= sid <= 4:
                        with self._shouter_states_lock:
                            s = self._shouter_states[sid]
                            s['mac']       = item['mac']
                            s['ip']        = item['ip']
                            s['last_ts']   = item['ts']
                            s['reconnects'] += 1
                    self._lst_status_label.configure(
                        text=f"HELLO: sid={item['sid']} MAC={item['mac']}",
                        text_color="#1a7a38",
                    )

                elif t == 'lst_text':
                    self._append_lst_log(item['line'])

                elif t == 'listener_frame':
                    self._lst_total += 1
                    self._lst_frame_times.append(item['ts'])

                elif t == 'shouter_frame':
                    frame = item['frame']
                    sid   = frame.get('shouter_id', 0)
                    flags = frame.get('flags', 0)
                    self._sht_total += 1
                    self._sht_frame_times.append(item['ts'])
                    if 1 <= sid <= 4:
                        with self._shouter_states_lock:
                            s = self._shouter_states[sid]
                            s['last_ts'] = item['ts']
                            if flags & 0x01:  # HIT
                                s['hits'] += 1
                                s['rssi'] = frame.get('poll_rssi')
                                s['nf']   = frame.get('poll_noise_floor')
                                if s['mac'] == '--':
                                    s['mac'] = frame.get('mac', '--')
                            else:             # MISS
                                s['misses'] += 1

                elif t == 'snap_frame':
                    snap = item['snap']
                    self._snap_total += 1
                    self._music_estimator.collect(
                        snap['reporter_id'], snap['peer_id'], snap['csi']
                    )
                    if self._snap_total % 10 == 1:
                        self._append_lst_log(
                            f"[MUSIC] snap #{self._snap_total}: "
                            f"reporter={snap['reporter_id']} peer={snap['peer_id']}"
                        )

                elif t == 'ranging_start':
                    self._append_lst_log("[DIAG] Ranging phase restarted — resetting MUSIC buffers")
                    self._music_estimator.reset_all()
                    if self._reset_music_cb is not None:
                        self._reset_music_cb()

        except queue.Empty:
            pass
        self.after(100, self._poll_debug_queue)

    # ── Debug table update ───────────────────────────────────────────────────

    def _update_debug_table(self) -> None:
        now = time.time()

        with self._shouter_states_lock:
            for sid in range(1, 5):
                s    = self._shouter_states[sid]
                lbls = self._tbl_labels[sid]

                # Last seen
                if s['last_ts'] == 0.0:
                    last_str = "never"
                else:
                    delta = now - s['last_ts']
                    if delta < 2.0:
                        last_str = f"{delta:.1f}s"
                        lbls['last_seen'].configure(text_color=_GREEN_TXT)
                    elif delta < 10.0:
                        last_str = f"{delta:.1f}s"
                        lbls['last_seen'].configure(text_color="#e67e00")
                    else:
                        last_str = f"{delta:.0f}s"
                        lbls['last_seen'].configure(text_color="#c0392b")

                total = s['hits'] + s['misses']
                miss_pct = f"{100*s['misses']/total:.1f}%" if total > 0 else "--"

                lbls['mac'].configure(text=s['mac'])
                lbls['ip'].configure(text=s['ip'])
                lbls['last_seen'].configure(text=last_str)
                lbls['rssi'].configure(text=str(s['rssi']) if s['rssi'] is not None else '--')
                lbls['nf'].configure(text=str(s['nf'])   if s['nf']   is not None else '--')
                lbls['hits'].configure(text=str(s['hits']))
                lbls['misses'].configure(text=str(s['misses']))
                lbls['miss_pct'].configure(text=miss_pct)
                lbls['reconnects'].configure(text=str(s['reconnects']))

                # Highlight row green if recently seen
                if s['last_ts'] and (now - s['last_ts']) < 2.0:
                    for lbl in lbls.values():
                        lbl.configure(fg_color="#e6f5ec")
                elif s['last_ts'] == 0.0:
                    bg = "#f9f9f9" if (sid % 2 == 0) else "white"
                    for lbl in lbls.values():
                        lbl.configure(fg_color=bg)

        # Update FPS stats
        cutoff = now - 3.0
        lst_recent = sum(1 for t in self._lst_frame_times if t > cutoff)
        sht_recent = sum(1 for t in self._sht_frame_times if t > cutoff)
        self._lst_fps_label.configure(text=f"Listener CSI: {lst_recent/3:.1f} fps")
        self._sht_fps_label.configure(text=f"Shouter poll: {sht_recent/3:.1f} fps")
        self._lst_total_label.configure(text=f"LST frames: {self._lst_total}")
        self._sht_total_label.configure(text=f"SHT frames: {self._sht_total}  MUSIC snaps: {self._snap_total}")

        self.after(500, self._update_debug_table)

    # ── Distance display ─────────────────────────────────────────────────────

    def _update_distances(self) -> None:
        # Show MUSIC distances only — RSSI estimates suppressed
        dists = self._music_estimator.get_distances()

        for key, lbl in self._dist_labels.items():
            d = dists.get(key)
            if d is None:
                lbl.configure(text="--", text_color="#aaaaaa")
            else:
                lbl.configure(text=f"{d:.2f} m", text_color=_GREEN_TXT)

        self.after(1000, self._update_distances)

    # ── Log helper ───────────────────────────────────────────────────────────

    def _append_lst_log(self, msg: str) -> None:
        msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._lst_log_text.configure(state="normal")
        self._lst_log_text.insert("end", msg + "\n")
        line_count = int(self._lst_log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self._lst_log_text.delete("1.0", f"{line_count - MAX_LOG_LINES + 1}.0")
        self._lst_log_text.configure(state="disabled")
        self._lst_log_text.see("end")
        if self._lst_log_file:
            try:
                self._lst_log_file.write(msg + "\n")
                self._lst_log_file.flush()
            except OSError:
                pass

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def debug_queue(self) -> queue.Queue:
        """Expose queue for external components that need to read it."""
        return self._debug_queue

    @property
    def spacing_estimator(self):
        """SpacingEstimator accessor (ranging frames removed in GHV4)."""
        return None

    def set_reset_music_callback(self, cb) -> None:
        """Register a callback invoked when ranging restarts."""
        self._reset_music_cb = cb

    def stop(self) -> None:
        """Stop the listener thread and close log file."""
        if self._listener_debug_thread:
            self._listener_debug_thread.stop()
        if self._lst_log_file:
            try:
                self._lst_log_file.close()
            except OSError:
                pass
            self._lst_log_file = None


# ── ShouterDebugTab ──────────────────────────────────────────────────────────

class ShouterDebugTab(ctk.CTkFrame):
    """Shouter debug panel — color-coded event log with connect/disconnect."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Thread state
        self._debug_queue: queue.Queue = queue.Queue()
        self._shouter_debug_thread: ShouterDebugThread | None = None

        # Debug log file (opened on Connect, closed on Disconnect)
        self._sht_log_file = None

        self._build_ui()
        self._poll_debug_queue()

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}

        # Connection row
        cf = ctk.CTkFrame(self)
        cf.pack(fill="x", **pad)
        ctk.CTkLabel(cf, text="SHOUTER PORT", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        conn_row = ctk.CTkFrame(cf, fg_color="transparent")
        conn_row.pack(fill="x", padx=8, pady=(0, 8))
        self._sht_port_var = ctk.StringVar(value="COM3")
        self._sht_port_combo = ctk.CTkComboBox(conn_row, variable=self._sht_port_var, width=140)
        self._sht_port_combo.pack(side="left")
        ctk.CTkButton(
            conn_row, text="Refresh", width=70, command=self._refresh_debug_ports
        ).pack(side="left", padx=(8, 0))
        self._sht_connect_btn = ctk.CTkButton(
            conn_row, text="Connect", width=80,
            fg_color="#2d9a4a", hover_color="#1f7a38", text_color="white",
            command=self._connect_shouter_debug,
        )
        self._sht_connect_btn.pack(side="left", padx=(8, 0))
        self._sht_disconnect_btn = ctk.CTkButton(
            conn_row, text="Disconnect", width=90,
            fg_color="#e0e8f0", hover_color="#e0e8f0", text_color="#aaaaaa",
            state="disabled",
            command=self._disconnect_shouter_debug,
        )
        self._sht_disconnect_btn.pack(side="left", padx=(6, 0))

        self._sht_status_label = ctk.CTkLabel(
            self, text="Not connected", text_color="#888888", font=("", 11)
        )
        self._sht_status_label.pack(anchor="w", padx=14, pady=(0, 4))

        # Legend
        leg = ctk.CTkFrame(self, fg_color="#f0f4f8")
        leg.pack(fill="x", padx=12, pady=(0, 6))
        leg_row = ctk.CTkFrame(leg, fg_color="transparent")
        leg_row.pack(fill="x", padx=8, pady=4)
        legend_items = [
            ("POLL",  "#1565c0"),
            ("SHOUT", "#e65100"),
            ("HELLO", "#1a7a38"),
            ("WiFi",  "#6a1e8a"),
            ("ERROR", "#c0392b"),
        ]
        for tag, color in legend_items:
            ctk.CTkLabel(
                leg_row, text=f"  {tag}", text_color=color, font=("", 11),
            ).pack(side="left", padx=(0, 14))

        # Event log
        lf = ctk.CTkFrame(self)
        lf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ctk.CTkLabel(lf, text="SHOUTER EVENT LOG", font=("", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 2)
        )
        self._sht_log_text = ctk.CTkTextbox(
            lf, height=300, font=("Courier", 11), state="disabled"
        )
        self._sht_log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Configure color tags on the underlying tk.Text widget
        tb = self._sht_log_text._textbox
        tb.tag_configure("poll",  foreground="#1565c0")
        tb.tag_configure("shout", foreground="#e65100")
        tb.tag_configure("hello", foreground="#1a7a38")
        tb.tag_configure("wifi",  foreground="#6a1e8a")
        tb.tag_configure("error", foreground="#c0392b")

    # ── Port refresh ─────────────────────────────────────────────────────────

    def _refresh_debug_ports(self) -> None:
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
        self._sht_port_combo.configure(values=ports)
        if ports and not self._sht_port_var.get():
            self._sht_port_var.set(ports[0])

    # ── Connect / Disconnect ─────────────────────────────────────────────────

    def _connect_shouter_debug(self) -> None:
        if self._shouter_debug_thread and self._shouter_debug_thread.is_alive():
            return
        port = self._sht_port_var.get().strip()
        if not port:
            return
        # Open debug log file
        if getattr(sys, "frozen", False):
            log_dir = os.path.dirname(sys.executable)
        else:
            log_dir = os.path.dirname(os.path.abspath(__file__))
        ts = time.strftime("%H%M%S")
        log_path = os.path.join(log_dir, f"shouter_debug_{ts}.txt")
        try:
            self._sht_log_file = open(log_path, "w", encoding="utf-8", errors="replace")
            self._sht_log_file.write(
                f"GHV4 Shouter Debug Log — {port} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "=" * 60 + "\n\n"
            )
            self._sht_log_file.flush()
        except OSError:
            self._sht_log_file = None
        self._shouter_debug_thread = ShouterDebugThread(port, self._debug_queue)
        self._shouter_debug_thread.start()
        self._sht_connect_btn.configure(state="disabled")
        self._sht_disconnect_btn.configure(
            state="normal", fg_color="#c0392b", hover_color="#9b2820", text_color="white"
        )
        msg = f"Connecting to {port}..."
        if self._sht_log_file:
            msg += f"  (log: {os.path.basename(log_path)})"
        self._sht_status_label.configure(text=msg, text_color="#888888")

    def _disconnect_shouter_debug(self) -> None:
        if self._shouter_debug_thread:
            self._shouter_debug_thread.stop()

    def _reset_shouter_debug_buttons(self) -> None:
        self._sht_connect_btn.configure(state="normal")
        self._sht_disconnect_btn.configure(
            state="disabled", fg_color="#e0e8f0", hover_color="#e0e8f0", text_color="#aaaaaa"
        )
        self._sht_status_label.configure(text="Disconnected", text_color="#888888")
        self._shouter_debug_thread = None
        if self._sht_log_file:
            try:
                self._sht_log_file.close()
            except OSError:
                pass
            self._sht_log_file = None

    # ── Debug queue polling ──────────────────────────────────────────────────

    def _poll_debug_queue(self) -> None:
        try:
            while True:
                item = self._debug_queue.get_nowait()
                t = item.get('type')

                if t == 'status':
                    msg = item['msg']
                    if self._shouter_debug_thread and self._shouter_debug_thread.is_alive():
                        self._sht_status_label.configure(text=msg, text_color="#444444")
                    if '[DBG] Shouter' in msg or '[SHT]' in msg:
                        self._append_sht_log(msg)
                        self._sht_status_label.configure(text=msg, text_color="#1a7a38")
                    else:
                        self._append_sht_log(msg)

                elif t == 'done':
                    if self._shouter_debug_thread and not self._shouter_debug_thread.is_alive():
                        self._reset_shouter_debug_buttons()

                elif t == 'sht_line':
                    self._append_sht_log(item['line'])

        except queue.Empty:
            pass
        self.after(100, self._poll_debug_queue)

    # ── Log helper ───────────────────────────────────────────────────────────

    def _append_sht_log(self, line: str) -> None:
        # Prepend wall-clock timestamp
        line = f"[{time.strftime('%H:%M:%S')}] {line}"
        # Determine color tag
        lo = line.lower()
        if '[sht] poll' in lo:
            tag = "poll"
        elif '[sht] shout' in lo:
            tag = "shout"
        elif '[sht] hello' in lo or 'hello sent' in lo:
            tag = "hello"
        elif 'wifi' in lo or 'connect' in lo or 'reconnect' in lo:
            tag = "wifi"
        elif 'error' in lo or 'fatal' in lo:
            tag = "error"
        else:
            tag = None

        tb = self._sht_log_text._textbox
        tb.configure(state="normal")
        if tag:
            tb.insert("end", line + "\n", tag)
        else:
            tb.insert("end", line + "\n")
        line_count = int(tb.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            tb.delete("1.0", f"{line_count - MAX_LOG_LINES + 1}.0")
        tb.configure(state="disabled")
        tb.see("end")
        if self._sht_log_file:
            try:
                self._sht_log_file.write(line + "\n")
                self._sht_log_file.flush()
            except OSError:
                pass

    # ── Public API ───────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop the shouter thread and close log file."""
        if self._shouter_debug_thread:
            self._shouter_debug_thread.stop()
        if self._sht_log_file:
            try:
                self._sht_log_file.close()
            except OSError:
                pass
            self._sht_log_file = None
