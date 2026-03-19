# Firmware — ESP32 Protocol & Gotchas
<!-- last verified: 2026-03-19 -->

## Serial Frame Types (listener COM port → PC)
```
[0xAA][0x55]  — listener CSI frame   magic(2) + 20-byte header + csi[N]
[0xBB][0xDD]  — shouter poll frame   magic(2) + 29-byte header + csi[N]
[0xCC][0xDD]  — ranging report       magic(2) + 12-byte payload (fixed, no CSI)
[0xEE][0xFF]  — CSI snapshot fwd     magic(2) + csi_snap_pkt_t payload (reporter, peer, seq, csi[N])
text          — [LST] debug lines    pure ASCII, newline-terminated
```

## UDP Packet Types (WiFi, not serial)
```
[BB][FA]  hello_pkt_t       shouter → listener  (10 bytes)
[BB][CC]  poll_pkt_t        listener → shouter  (108 bytes, includes 96-byte pad)
[BB][EE]  response_pkt_t    shouter → listener  (404 bytes, includes up to 384-byte CSI)
[BB][A0]  peer_info_pkt_t   listener → shouters (32 bytes, during ranging)
[BB][A1]  range_req_pkt_t   listener → shouter  (7 bytes, per-beacon-round)
[BB][A2]  range_bcn_pkt_t   shouter → broadcast (8 bytes, during ranging)
[BB][A3]  ranging_rpt_pkt_t shouter → listener  (14 bytes, RSSI report)
[BB][A4]  csi_snap_pkt_t    shouter → listener  (up to 392 bytes, one per CSI snapshot)
```

## Frame Protocol
- `[0xAA][0x55]` — listener frame (20-byte header after magic)
- `[0xBB][0xDD]` — shouter frame (29-byte header after magic)
- `[0xCC][0xDD]` — ranging frame (routed to SpacingEstimator)
- 128 subcarriers; **null indices `{0,1,2,32,63,64,65}` must be dropped** before feature extraction

## Gotchas

- **`shouter_csi_cb` MAC matching always fails in STA mode** — `wifi_csi_info_t.mac` is the
  AP/listener BSSID, not the transmitting shouter's MAC. Peer RSSI is captured in
  `on_esp_now_recv` via `recv_info->rx_ctrl->rssi` — true P2P RSSI, not AP-relayed.
  Dispatch order in `loop()`: `[0xA0]` PEER_INFO → `[0xA1]` RANGE_REQ → `[0xBB][0xCC]` POLL.
  The `[0xA2]` UDP RANGE_BCN handler was removed (2026-03-16) when ESP-NOW replaced UDP beacons.
- **`ranging_done` resets on shouter disconnect** — after the 2026-03-16 firmware fix, the
  listener fires `ARDUINO_EVENT_WIFI_AP_STADISCONNECTED` on disconnect, sets `ranging_done = false`,
  and re-ranges once all 4 reconnect. `ranging_done` is `static volatile bool` at file scope
  (not `static bool` inside `loop()`). If re-ranging never fires, check the event handler is
  registered in `setup()` and that `shouter_mac[]` was populated by a prior HELLO.
- **SHOUTER DISTANCES cards stay `--` (firmware causes)** —
  (2) **All 4 shouters must run the same firmware** — bidirectional RSSI requires `min(count[i→j], count[j→i]) >= 1`;
  if any shouter runs old firmware it never sends `[BB][A3]`, its direction stays 0, every pair involving it stays `--`.
  (3) **`[BB][A3]` must be sent BEFORE CSI snapshots in ShouterV3.ino** (fixed 2026-03-17) — sending it after
  90 × ~392-byte snapshot packets overflows the listener's UDP RX queue and silently drops `[BB][A3]`.
  Fix already applied: ranging_rpt is now sent immediately after `[BB][EE]`, before the snap loop.
  See also: Python-side cause (1) in `ghv4/CLAUDE.md`.
- **Ranging requires all 4 shouters — no timeout fallback** — `run_ranging_phase()` fires only
  when `registered_shouter_count == 4`. If one shouter permanently fails to associate (hardware
  fault, wrong `SHOUTER_ID` flashed), distances stay `--` indefinitely. Fix: resolve the hardware
  issue, then power-cycle the listener.
- **WiFi event handler lambdas in ESP32 Arduino use empty capture `[]`** — file-scope variables
  (`shouter_mac[]`, `shouter_ready[]`, `ranging_done`) are accessed directly, not captured.
  Using `[&]` or `[=]` for file-scope vars is a compile error. Cross-task bools must be
  `volatile` at file scope to prevent compiler register-caching across task boundaries.
- Shouter firmware sends ranging frames only during a discrete ranging phase (logged as `[LST] Starting ranging phase` / `[LST] Ranging phase complete`), not continuously
- Shouter serial port outputs **text only** (`[SHT]` lines) — no binary frames. Listener serial port outputs binary frames (`[0xAA][0x55]`, `[0xBB][0xDD]`, `[0xCC][0xDD]`) mixed with `[LST]` text.
- **ESP-NOW init sequence** — `esp_now_init()` must be called after `connect_and_register()`
  (WiFi STA fully connected). Do NOT call it again on WiFi dropout/reconnect — it persists.
  Broadcast MAC must be registered via `esp_now_add_peer` before any `esp_now_send` or send
  silently returns `ESP_ERR_ESPNOW_NOT_FOUND`. Use `bcast_peer.channel = 0` (not 6) to avoid
  `ESP_ERR_ESPNOW_CHAN`. `on_esp_now_recv` runs in WiFi task context (Core 0) — use
  `portENTER_CRITICAL` (not ISR variant). `ifidx = WIFI_IF_STA` required in STA mode.
- **Passive background beacons** — Shouters send 1 ESP-NOW beacon/second from `loop()` (after
  ranging phase) to keep peer RSSI estimates live. Causes ~7–10% miss rate increase vs 0%.
  Interval is `last_passive_bcn_ms >= 1000` in ShouterV3.ino; increase to 2000 if misses climb.
- **Test room geometry** — perfect 25ft square. Shouter corners: 1=bottom-left, 2=top-left,
  3=top-right, 4=bottom-right. Sides (7.62m): 1-2, 2-3, 3-4, 4-1. Diagonals (10.78m): 1-3, 2-4.
- **`ranging_completed_ms` must be in globals** — declaring it inside or after `run_ranging_phase()` in
  `firmware/ListenerV3/ListenerV3.ino` causes "not declared in this scope". Place it in the file-scope globals block near `ranging_done`.
- **RSSI-based ranging accuracy** — Log-distance path loss model; `ranging_config.json` hot-reloads
  on each frame (no restart needed). Current calibration: `n=2.16`, `rssi_ref_dbm=-26.2` (two-point,
  anchored on side=7.62m and diagonal=10.78m of 25ft test room, 2026-03-17). RSSI has ±1–2m indoor
  error regardless of calibration — values reflect relative ordering more than absolute meters.
- **`SNAP_DRAIN_MS = 350` drain window in `poll_all_shouters()`** — without this, all 4 shouters burst 90 snaps concurrently and overflow the listener's UDP RX buffer (~8–16 KB), dropping ~85% of snap packets. MUSIC requires ≥15 snaps per direction; without the drain window it never triggers. If MUSIC distances stop appearing, check this value first. Fixed 2026-03-17.
- **CSI MUSIC ranging — fully implemented, not yet producing distances** — spec at
  `docs/superpowers/specs/2026-03-16-music-csi-ranging-design.md`; implementation plan at
  `docs/superpowers/plans/2026-03-16-music-csi-ranging.md`. Replaces RSSI scalar with
  MUSIC super-resolution CIR; offset-free (`d = c × τ`); bidirectional CFO cancellation via
  averaging τ_ij and τ_ji; `CSIMUSICEstimator` class in `ghv4/spacing_estimator.py`. MAC
  attribution solved by callback ordering: `shouter_csi_cb` (ISR, Core 0) always completes
  before `on_esp_now_recv` (WiFi task, Core 0) — do NOT move either callback off Core 0.
- **`CSI_SNAP_HEADER_SIZE = 6` in Python, not 8** — `offsetof(csi_snap_pkt_t, csi) = 8` in C
  (magic-inclusive), but `parse_csi_snap_frame` receives a buffer AFTER the 2 magic bytes are
  consumed by the dispatcher, so the pre-CSI header is only 6 bytes. The spec originally had 8;
  corrected in both spec and plan 2026-03-16.
- **ESP32 CSI byte format** — int8 imaginary first, then int8 real, per subcarrier (2 bytes
  each). 128 subcarriers × 2 = 256 bytes minimum for HT20. With `ltf_merge_en=true`,
  `info->len` may exceed 256 bytes; only the first 256 are needed for 128 subcarriers.
- **`bcn_seq=0xFF` passive beacon sentinel already in firmware** — no firmware change needed;
  the guard `if (bcn->bcn_seq == 0xFF) return;` in `on_esp_now_recv` is sufficient to skip
  background beacons during CSI snapshot collection.
- **Shouter distances off by 3-5m** — RSSI log-distance model with current calibration
  (`n=2.16, rssi_ref=-26.2`) still has significant indoor error. RSSI is being phased out
  in favor of CSI MUSIC ranging (`d = c * tau`, no calibration needed). Diagnostics plan
  at `docs/superpowers/plans/2026-03-17-music-ranging-diagnostics.md`.
- **Never call `Serial.printf` inside `portENTER_CRITICAL`** — `Serial.printf` is blocking I/O;
  calling it with interrupts disabled triggers ESP32 watchdog timeout. Capture values inside
  the critical section, print after `portEXIT_CRITICAL`.
