# Firmware — ESP32 Protocol & Gotchas
<!-- last verified: 2026-03-24 (continuous beacons implemented, ranging disabled) -->

## Important
- **Canonical firmware is in `GHV4/firmware/`** — `ListenerV4/` and `ShouterV4/` directories. `GHV3/firmware/` contains identical copies but is not the active working set.
- **12 firmware improvements implemented (2026-03-22)** — see `docs/superpowers/plans/2026-03-21-firmware-12-improvements.md` for the full plan. All tasks complete; hardware testing deferred.
- **Design spec** at `~/.claude/plans/transient-leaping-rabin.md` — detailed design with review fixes applied.
- **Continuous shouter beacons (implemented, hardware test pending)** — 10 Hz ESP-NOW beacons
  from each shouter for breathing/heart rate CSI. Replaces one-shot ranging for SAR mode.
  Spec: `docs/superpowers/specs/2026-03-23-continuous-shouter-beacons-design.md`.
  Plan: `docs/superpowers/plans/2026-03-23-continuous-shouter-beacons.md`.

## Serial Frame Types (listener COM port → PC)
```
[0xAA][0x55]  — listener CSI frame   magic(2) + 20-byte header + csi[N]
[0xBB][0xDD]  — shouter poll frame   magic(2) + 29-byte header + csi[N]
[0xEE][0xFF]  — CSI snapshot fwd     magic(2) + csi_snap_pkt_t payload (reporter, peer, seq, csi[N])
text          — [LST] debug lines    pure ASCII, newline-terminated
```

## UDP Packet Types (WiFi, not serial)
```
[BB][FA]  hello_pkt_t       shouter → listener  (10 bytes)
[BB][CC]  poll_pkt_t        listener → shouter  (108 bytes, includes 96-byte pad)
                            target_id=0xFF for broadcast poll (USE_BROADCAST_POLL=1)
[BB][EE]  response_pkt_t    shouter → listener  (404 bytes, includes up to 384-byte CSI)
[BB][A0]  peer_info_pkt_t   listener → shouters (32 bytes, during ranging)
[BB][A1]  range_req_pkt_t   listener → shouter  (7 bytes, per-beacon-round)
[BB][A2]  range_bcn_pkt_t   shouter → broadcast (8 bytes, during ranging)
[BB][A3]  ranging_rpt_pkt_t — REMOVED (RSSI ranging dead code; struct and send/receive paths deleted)
[BB][A4]  csi_snap_pkt_t    shouter → listener  (up to 392 bytes, one per CSI snapshot)
[BB][A5]  ack_pkt_t         bidirectional       (6 bytes; ack_type=0xFA for HELLO ACK, 0xA0 for PEER_INFO ACK)
```

## Frame Protocol
- `[0xAA][0x55]` — listener frame (20-byte header after magic)
- `[0xBB][0xDD]` — shouter frame (29-byte header after magic)
- `[0xEE][0xFF]` — CSI snapshot frame (6-byte header after magic + csi[N])
- 128 subcarriers; **null indices `{0,1,2,32,63,64,65}` must be dropped** before feature extraction

## Gotchas

- **`shouter_csi_cb` MAC matching always fails in STA mode** — `wifi_csi_info_t.mac` is the
  AP/listener BSSID, not the transmitting shouter's MAC. Peer RSSI is captured in
  `on_esp_now_recv` via `recv_info->rx_ctrl->rssi` — true P2P RSSI, not AP-relayed.
  Dispatch order in `loop()`: `[0xA5]` ACK → `[0xA0]` PEER_INFO → `[0xA1]` RANGE_REQ → `[0xBB][0xCC]` POLL.
  The `[0xA2]` UDP RANGE_BCN handler was removed (2026-03-16) when ESP-NOW replaced UDP beacons.
- **`ranging_done` resets on shouter disconnect** — the listener fires
  `ARDUINO_EVENT_WIFI_AP_STADISCONNECTED` on disconnect, sets `ranging_done = false`,
  and resets the ranging state machine to `RNG_IDLE`. If ranging was in progress, it is aborted.
  `ranging_done` is `static volatile bool` at file scope. If re-ranging never fires, check the
  event handler is registered in `setup()` and that `shouter_mac[]` was populated by a prior HELLO.
- **`[BB][A3]` ranging report removed** — RSSI-based peer ranging was dead code in GHV4
  (Python discarded these frames). The `ranging_rpt_pkt_t` struct, shouter send path, and
  listener receive/emit path have all been deleted. The peer_table and ESP-NOW callback
  remain (used by CSI snapshot collection for MUSIC distance estimation).
- **Ranging is non-blocking (state machine)** — `advance_ranging()` replaces the old blocking
  `run_ranging_phase()`. Each `loop()` iteration advances one state: `RNG_IDLE` → `RNG_SEND_PEER_INFO`
  → `RNG_WAIT_PEER_ACK` → `RNG_BEACON_ROUND` → `RNG_WAIT_BEACONS` → `RNG_DRAIN_SNAPS` →
  `RNG_NEXT_SHOUTER` → `RNG_COMPLETE`. CSI polling continues during ranging — no 11-second pause.
  Dynamic snap drain exits on expected count (105), silence timeout (500ms), or hard cap (3s).
- **Fatal errors auto-restart** — all `while(1) delay(1000)` halts replaced with
  `delay(3000); ESP.restart();`. The 3s delay allows the error message to print to serial.
  Applies to CSI config failure (listener + shouter), WiFi connect timeout, ESP-NOW init, and
  broadcast peer registration.
- **MAC-based ID assignment** — shouters no longer need `SHOUTER_ID` compiled in. The listener
  has a `known_macs[4][6]` lookup table; on HELLO, it derives the ID from the MAC and sends it
  back via HELLO ACK (`[BB][A5]`). The shouter stores `my_id` from the ACK. Unknown MACs get the
  next available slot as fallback for board replacement. Polls received before ID assignment are
  skipped. MAC table: `{68:FE:71:90:60:A0}→1, {68:FE:71:90:68:14}→2, {68:FE:71:90:6B:90}→3, {20:E7:C8:EC:F5:DC}→4`.
- **Broadcast polling** — controlled by `#define USE_BROADCAST_POLL 1` in ListenerV4.ino.
  Listener sends one poll with `target_id=0xFF` to `192.168.4.255`. Each shouter staggers its
  response by `(my_id - 1) * STAGGER_MS` (default 40ms). Cycle time ~200ms vs ~420ms sequential.
  Set `USE_BROADCAST_POLL 0` to fall back to sequential polling. Shouter extracts poll response
  into `send_poll_response()` helper; stagger uses `stagger_pending`/`stagger_target_ms`/`stagger_poll`.
- **CSI timestamp matching** — shouter selects the ring buffer entry with the smallest age
  (`now_ms - rx_timestamp_ms`) instead of always using the newest. Both use `esp_timer_get_time()/1000`
  as the clock source for consistency with the ISR callback.
- **Ring buffer overflow counters** — both listener and shouter track `csi_overflow_count`
  (volatile, ISR-written). Emitted every 100 poll cycles as `[LST] csi_overflow=N` or
  `[SHT] csi_overflow=N`. Listener drops frames when ring is full; shouter overwrites oldest.
- **Health monitoring** — listener tracks `consecutive_miss[id]` per shouter. After 10
  consecutive poll misses, emits `[LST] WARN shouter %d: %d consecutive misses` (one-shot,
  resets on next hit).
- **Listener SPOF detection** — shouter tracks `last_poll_rx_ms`. If no polls received for 10s,
  emits `[SHT] WARN no polls for N ms — listener may be down` (one-shot, resets on next poll).
- **Ranging disabled (SAR mode)** — `advance_ranging()` and the `rng_state != RNG_IDLE` gate
  are commented out in `ListenerV4.ino loop()`. Polls run unconditionally from startup.
  The ranging state machine, enum, variables, and WiFi event handler remain in source but are
  inactive. To re-enable ranging: uncomment both lines and restore `delay(15)` in
  `ShouterV4 send_poll_response()` snap drain loop.
- **Passive background beacons removed** — the 1 Hz ESP-NOW passive beacons that caused 7-10%
  miss rate increase have been deleted from `loop()`. The `bcn_seq==0xFF` guard in `on_esp_now_recv`
  is retained as defensive code for backwards compatibility with old firmware peers.
- **WiFi event handler lambdas in ESP32 Arduino use empty capture `[]`** — file-scope variables
  (`shouter_mac[]`, `shouter_ready[]`, `ranging_done`, `rng_state`) are accessed directly, not captured.
  Using `[&]` or `[=]` for file-scope vars is a compile error. Cross-task bools must be
  `volatile` at file scope to prevent compiler register-caching across task boundaries.
- Shouter serial port outputs **text only** (`[SHT]` lines) — no binary frames. Listener serial port outputs binary frames (`[0xAA][0x55]`, `[0xBB][0xDD]`, `[0xEE][0xFF]`) mixed with `[LST]` text.
- **ESP-NOW init sequence** — `esp_now_init()` must be called after `connect_and_register()`
  (WiFi STA fully connected). Do NOT call it again on WiFi dropout/reconnect — it persists.
  Broadcast MAC must be registered via `esp_now_add_peer` before any `esp_now_send` or send
  silently returns `ESP_ERR_ESPNOW_NOT_FOUND`. Use `bcast_peer.channel = 0` (not 6) to avoid
  `ESP_ERR_ESPNOW_CHAN`. `on_esp_now_recv` runs in WiFi task context (Core 0) — use
  `portENTER_CRITICAL` (not ISR variant). `ifidx = WIFI_IF_STA` required in STA mode.
- **Test room geometry** — perfect 25ft square. Shouter corners: 1=bottom-left, 2=top-left,
  3=top-right, 4=bottom-right. Sides (7.62m): 1-2, 2-3, 3-4, 4-1. Diagonals (10.78m): 1-3, 2-4.
- **RSSI-based ranging accuracy** — Log-distance path loss model; `ranging_config.json` hot-reloads
  on each frame (no restart needed). Current calibration: `n=2.16`, `rssi_ref_dbm=-26.2` (two-point,
  anchored on side=7.62m and diagonal=10.78m of 25ft test room, 2026-03-17). RSSI has ±1–2m indoor
  error regardless of calibration — values reflect relative ordering more than absolute meters.
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
- **Never call `Serial.printf` inside `portENTER_CRITICAL`** — `Serial.printf` is blocking I/O;
  calling it with interrupts disabled triggers ESP32 watchdog timeout. Capture values inside
  the critical section, print after `portEXIT_CRITICAL`.
