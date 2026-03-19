```markdown
# 20260319T135326_session.md

## High‑level overview  

| Component | Role | Radio mode | Primary network function |
|-----------|------|------------|--------------------------|
| **ListenerV4** (`ListenerV4.ino`) | *Central coordinator* (the “listener”) | Wi‑Fi **AP** (soft‑AP) + UDP server on port 3333 | Discovers shouters, polls them, collects CSI snapshots, runs a ranging phase, emits serial frames that are later parsed by a host PC. |
| **ShouterV4** (`ShouterV4.ino`) | *Peripheral* (the “shouter”) | Wi‑Fi **STA** + ESP‑NOW for peer‑to‑peer beacons | Registers with the listener, receives POLL packets, replies with a response packet (containing the latest CSI and a ranging report), and streams CSI snapshots back to the listener. |

Both sketches share **`../GHV4Protocol.h`** which defines the binary packet formats (`HELLO`, `POLL`, `RANGE_REQ`, `RESP`, `CSI_SNAP`, …) and a set of magic numbers / field layouts. All timing, sequence numbers, and buffer layouts are therefore identical on both sides, enabling a strict request‑response hand‑shake.

---  

## Communication flow – step by step  

### 1. Network set‑up  

| Listener | Shouter |
|----------|---------|
| `WiFi.mode(WIFI_AP)` → `softAP(SSID, nullptr, CHANNEL)` | `WiFi.mode(WIFI_STA)` → connects to the same SSID |
| Starts a UDP server on **LISTENER_PORT (3333)** | Starts a UDP client on **SHOUTER_PORT (3334)** (outgoing to the listener’s soft‑AP IP) |
| Enables CSI capture (`esp_wifi_set_csi(true)`) | Enables CSI capture (`esp_wifi_set_csi(true)`) and registers a CSI ISR (`shouter_csi_cb`) |

Both devices now share the same Wi‑Fi channel (the AP channel). The shouter also brings up ESP‑NOW for low‑latency, peer‑to‑peer beacons that are *not* routed through the AP.

---  

### 2. Discovery – **HELLO**  

* **Shouter** sends a `HELLO_MAGIC` UDP packet to the listener (`LISTENER_IP:LISTENER_PORT`).  
* **Listener** receives the packet in `handle_incoming_udp()`.  
  * It extracts `shouter_id`, stores the source IP/MAC in `shouter_ip[]` / `shouter_mac[]`, marks `shouter_ready[] = true`, and prints a debug line.  

Result: every discovered shouter gets a stable entry indexed by its logical ID (1‑4). The listener knows *who* is present and can address it directly later.

---  

### 3. Polling – **POLL → RESP**  

1. **Listener** iterates over all registered shouters (`poll_all_shouters()`).
2. For each `id` it builds a `poll_pkt_t` and sends it to `shouter_ip[id]:SHOUTER_PORT`.
3. The shouter receives the POLL, builds a `response_pkt_t` and sends it back on the same UDP socket.
   * The response contains:
     * `tx_seq`, `tx_ms`, `poll_seq`, `shouter_id`
     * `poll_rssi` / `poll_noise_floor` (derived from the *latest* CSI entry)
     * Optional `csi[]` payload (the most recent CSI snapshot)
   * If no response arrives within `POLL_TIMEOUT_MS`, the listener emits a “miss” frame (`emit_shouter_frame(nullptr, id, false, …)`).
4. After a successful reply, the listener *drains* the CSI ring buffer for **SNAP_DRAIN_MS** to capture any CSI packets the shouter may burst out immediately after its response. This prevents the next POLL from colliding with the previous shouter’s CSI flood.

**Key synchronization points**

* `current_poll_seq` is updated in the listener *before* each POLL and copied into the transmitted packet. The shouter validates `resp.poll_seq == poll_seq` before accepting a packet, eliminating stale replies.
* All accesses to the CSI ring buffer in the listener are protected by `lst_ring_mux` (IRAM‑safe spinlock).  
* In the shouter, `get_latest_csi()` uses `(ring_write - 1) % RING_SIZE` to fetch the newest entry safely.

---  

### 4. Ranging phase – **PEER_INFO → RANGE_REQ → RANGE_BCN → RANGE_RPT**  

When *all four* shouters have been registered for at least `RANGING_STABILITY_MS` and no recent dropout has occurred (`RANGING_COOLDOWN_MS`), the listener starts a **ranging phase**:

1. **PEER_INFO** packets are broadcast to *all* shouters (`peer_info_pkt_t`).  
   * They carry the MAC addresses of the four logical peers (including the sender’s own MAC).  
   * This lets each shouter fill its `peer_table[]` (used later for RSSI EMA).  

2. **RANGE_REQ** packets are sent *sequentially* to each shouter (target_id = 1…4).  
   * Payload: `n_beacons`, `interval_ms`, etc.  
   * The shouter replies by sending **ESP‑NOW beacons** (`range_bcn_pkt_t`) on the broadcast ESP‑NOW channel.  
   * Each beacon is transmitted after a short `delay(rr->interval_ms)`.  

3. While the shouter is transmitting those beacons, its CSI ISR keeps filling the `csi_snap_buf[][N_SNAP]`. When the listener later receives a **CSI_SNAP** packet (`magic = CSI_SNAP_MAGIC`), it forwards the payload to the host PC via a special serial frame (`emit_csi_snap_frame`).  

4. After each beacon burst the listener calls `poll_all_shouters()` again **once** to collect the **RANGE_RPT** (`RANGE_RPT_MAGIC`) that each shouter sends back *immediately* after its POLL‑response. This packet contains the EMA‑smoothed RSSI values taken from `peer_table[]`.  

5. The ranging phase finishes when all four beacons have been requested and processed. The listener prints a summary (`snap_frames_emitted`) and records `ranging_completed_ms`.

---  

### 5. Continuous listening – **loop()**  

* The listener’s `loop()` runs at a minimum interval of `POLL_INTERVAL_MIN_MS` (≈ 50 ms).  
* If no shouters are registered yet it only drains the CSI buffer and looks for stray HELLO packets.  
* Once four shouters are stable it either runs the ranging phase (once) or proceeds to regular polling of each shouter.  
* Throughout the loop it also **drain`s** the CSI ring buffer and calls `handle_incoming_udp()` to process any stray HELLO or response packets that arrive outside the polling windows.

---  

## Protocol details – how the two sketches stay in sync  

| Packet type | Direction | Magic | Payload (key fields) | When it is sent | How the peer reacts |
|-------------|-----------|-------|----------------------|----------------|---------------------|
| `HELLO_MAGIC` | Shouter → Listener | `0xAA 0x55` | `shouter_id` | Immediately after Wi‑Fi connect (once) | Listener registers the shouter (IP/MAC) |
| `POLL_MAGIC` | Listener → Shouter | `0xAA 0x55` (same magic) | `target_id`, `poll_seq`, `listener_ms` | Every polling cycle (per shouter) | Shouter replies with `RESP_MAGIC` (or a miss) |
| `RESP_MAGIC` | Shouter → Listener | `0xBB 0xA3` (RANGE_RPT) **or** `0xBB 0xA4` (CSI_SNAP) **or** `0xBB 0xA5` (response) | `tx_seq`, `tx_ms`, `poll_seq`, `shouter_id`, optional CSI payload | Immediately after POLL is received (or after ranging) | Listener emits a serial frame |
| `CSI_SNAP_MAGIC` | Shouter → Listener | `0xBB 0xA4` | Full `csi_snap_pkt_t` (reporter_id, peer_id, snap_seq, CSI payload) | During ranging phase, after each beacon | Listener forwards to host PC |
| `RANGE_BCN_MAGIC` | Shouter (via ESP‑NOW) → *other* shouters | `0xBB 0xA1` (range_req) → beacons | `shouter_id`, `bcn_seq` | Periodic passive beacons (1 Hz) | Other shouters update their RSSI tables |
| `PEER_INFO_MAGIC` | Listener → Shouter | `0xBB 0xA0` | List of peer MACs & IDs | Once, before ranging starts | Shouter populates its `peer_table[]` |

All packets are **binary**, little‑endian where indicated, and include a 2‑byte magic prefix that the receiver checks before parsing the rest. The protocol is deliberately *stateless* apart from the sequence numbers (`poll_seq`, `snap_seq`) and the timestamps (`listener_ms`, `tx_ms`) that are monotonically increased by each side.

---  

## Critical‑section & ISR design  

| Component | Protection mechanism | Reason |
|-----------|----------------------|--------|
| Listener CSI ring buffer (`lst_ring_*`) | `lst_ring_mux` (spinlock) – used in ISR (`portENTER_CRITICAL_ISR`) and in `loop()` (`portENTER_CRITICAL`) | ISR must not block; only copy data, no heap, no Serial. |
| Shouter CSI ring buffer (`ring_*`) | `ring_mux` (spinlock) – ISR variant for `shouter_csi_cb` | Same constraints – keep ISR short. |
| Shared data accessed by both tasks (e.g., `peer_table[]`) | `peer_mux` (normal spinlock) – taken in task context (`on_esp_now_recv`) | Guarantees atomic updates when ESP‑NOW callback fires. |
| `csi_snap_buf[][N_SNAP]` | `snap_mux` (spinlock) – taken when a shouter writes or clears its snapshot count | Prevent race between ISR (writing snapshots) and the polling thread (reading them). |
| ESP‑NOW send / receive callbacks | `portENTER_CRITICAL` (task‑level) – safe on dual‑core ESP32‑WROOM‑UE | Avoids race with the ISR that may be writing to the same buffers. |

All critical sections are deliberately *short* (a few `memcpy` / atomic variable updates). This design keeps the system responsive and avoids deadlocks, but it does impose a strict ordering: ISR → critical section → exit → task can then safely read the data.

---  

## Timing & back‑pressure considerations  

| Parameter | Meaning | Typical value | Effect |
|-----------|---------|---------------|--------|
| `POLL_TIMEOUT_MS` | Max time to wait for a POLL response | 100 ms (was 50 ms) | Gives shouters enough time to finish CSI capture and ESP‑NOW beacon bursts. |
| `SNAP_DRAIN_MS` | How long the listener stays in “drain mode” after a POLL response | 2000 ms | Prevents the listener’s UDP RX queue from overflowing when a shouter bursts many CSI snapshots. |
| `INTER_SHOUTER_GAP_MS` | Minimum idle time between consecutive POLLs | 5 ms | Allows the Wi‑Fi channel to settle and avoids collisions between successive POLLs. |
| `RANGING_STABILITY_MS` | Minimum idle time after the last HELLO before starting ranging | 5000 ms | Guarantees that all shouters have settled (no recent disconnects). |
| `RANGING_COOLDOWN_MS` | Minimum time since the last ranging phase finished before another can start | 30000 ms | Prevents rapid re‑ranging on temporary drop‑outs. |
| `CSI_SNAP_MAX` / `SHOUTER_CSI_MAX` | Maximum CSI payload length accepted by each side | Defined in protocol header | Controls how many bytes are copied into the response packet; overflow is clipped. |

The listener deliberately **blocks** only for the short `POLL_TIMEOUT_MS` window and for the *drain* period. All other work (Serial emission, ESP‑NOW beaconing) happens in the background via `delay()`/`yield()`. This design works well when the number of shouters is small (4) and the CSI payload size is bounded, but it can become a bottleneck if the number of peers or the CSI payload size grows.

---  

## Where the two sketches interact – a concise sequence diagram  

```
[Listener]                [Shouter]                     (Wi‑Fi/ESP‑NOW)
---------------------------------------------------------------------------
1. AP up, UDP listen      2. Connect to AP
3. Enable CSI            4. Enable CSI
5. Send HELLO ------------> (UDP) -->  (Listener receives)
6. Register shouter      7. Store IP/MAC
8. Periodic POLL --------> (UDP) -->  (Shouter receives)
9. Build RESP ------------> (UDP) -->  (Listener receives)
10. Emit serial frame   11. (Optional) ESP‑NOW beacon
12. Drain CSI buffer    13. (During drain) send CSI_SNAP packets
14. When all 4 stable --> 15. Send PEER_INFO (UDP)
16. Send RANGE_REQ -----> (UDP) -->  (Shouter receives)
17. Send ESP‑NOW beacons --> (ESP‑NOW) -->  (Neighbors receive)
18. Shouter captures CSI --> stores in csi_snap_buf
19. After beacons -----> 20. Build RANGE_RPT (UDP) --> (Listener receives)
21. Emit ranging frame  22. (Loop continues)
```

The **ESP‑NOW beacons** are the only *non‑UDP* path used by the shouter; they deliver true peer‑to‑peer RSSI measurements that the listener later packs into `RANGE_RPT`. All other traffic (HELLO, POLL, RESP, CSI_SNAP) travels over UDP between the soft‑AP interface of the listener and the STA interface of the shouter.

---  

## Potential issues & suggested improvements  

| Area | Observation | Suggested fix / mitigation |
|--------|-------------|-----------------------------|
| **Blocking delays** | The listener uses `delay()`/`delayMicroseconds()` and busy‑waits (`while (millis() < deadline)`). On a busy host this can freeze the serial output and make debugging harder. | Replace busy‑wait loops with `yield()` or a non‑blocking state machine. Keep only the minimal `delay()` required for pacing snapshot emission. |
| **UDP packet loss** | The listener’s UDP receive buffer is only ~8‑16 KB. During the CSI‑snapshot burst a shouter can emit > 30 KB, leading to dropped packets. | Increase the socket receive buffer (`udp.setRxBufferSize()`) if the underlying lwIP permits, or further spread the snapshot emission over a longer interval (e.g., increase the `delay(15)` to 20 ms). |
| **Race conditions between ESP‑NOW and CSI snapshots** | `on_esp_now_recv()` reads the latest CSI entry via `get_latest_csi()` while the CSI ISR may still be filling `csi_snap_buf[]`. If the ISR updates the buffer just after the callback reads it, the snapshot could be stale or partially written. | Protect the snapshot copy with `snap_mux` *inside* `on_esp_now_recv()` (the code already does this) and add a short “checksum” or sequence number to each `csi_snap_pkt_t` so the listener can discard out‑of‑order packets. |
| **ESP‑NOW channel drift** | The shouter uses the current STA channel for ESP‑NOW (`bcast_peer.channel = 0`). If the STA later changes channel (e.g., due to a Wi‑Fi reconnection), the ESP‑NOW packets may be sent on a different channel than the intended receivers, causing missed beacons. | Force the ESP‑NOW peer to stay on a fixed channel by calling `esp_wifi_set_channel(<fixed_channel>, WIFI_SECOND_CHAN_NONE)` before `esp_now_init()`, or re‑add the peer after any Wi‑Fi channel change. |
| **Limited snapshot capacity** | `csi_snap_buf[5][N_SNAP]` can hold at most `5 × N_SNAP` entries. With `N_SNAP = 35` the total size is ~67 KB, which is close to the DRAM budget on the ESP32‑WROOM‑UE. If the shouter’s CSI capture rate spikes (e.g., due to high traffic), the ring can overflow and snapshots are silently dropped. | Dynamically allocate `N_SNAP` based on the available heap at runtime, or implement a “drop‑oldest” policy where the oldest entry is overwritten when `csi_snap_count[sid]` reaches `N_SNAP`. Additionally, reduce `SHOUTER_CSI_MAX` if the CSI payload does not need the full 392‑byte maximum. |
| **Watchdog timeout on long loops** | The listener’s `loop()` contains several `while (millis() < deadline)` loops that can run for up to `POLL_TIMEOUT_MS` (100 ms) plus the drain interval (`SNAP_DRAIN_MS = 2000 ms`). If a shouter misbehaves and never sends a response, the listener may stay in that loop for seconds, potentially feeding the watchdog with a “soft‑reset”. | Replace the busy‑wait with a non‑blocking state machine that increments a timeout counter and aborts after a configurable number of attempts, emitting a “miss” frame and moving on. |
| **Serial output contention** | `Serial.printf()` is called from both ISR‑safe contexts (e.g., after `on_esp_now_recv`) and from normal task code. When called from an ISR‑protected section, it can enable interrupts only briefly, but prolonged prints can still trigger the watchdog if interrupts are disabled for too long. | Buffer the debug messages in a small ring buffer and emit them from the main `loop()` after all critical sections have been released. Alternatively, wrap each `Serial.printf()` with a watchdog‑friendly timeout (`if (millis() - start > 100) break;`). |
| **Hard‑coded magic numbers & protocol versioning** | The protocol relies on a handful of magic constants (`HELLO_MAGIC_0/1`, `RESP_MAGIC_0/1`, etc.). If a future revision changes the packet layout, old binaries will misinterpret new packets, leading to silent drops. | Introduce a 1‑byte protocol version field as the first byte after the magic pair, and have both sides reject packets with an unknown version. This makes forward compatibility easier. |
| **Power‑up sequencing** | The listener’s `setup()` calls `WiFi.softAP()` before enabling CSI (`esp_wifi_set_csi(true)`). If the CSI hardware needs the AP to be fully stable, the order could cause `esp_wifi_set_csi()` to fail on some ESP‑IDF versions. | Move the CSI configuration to *after* the AP is started and the soft‑AP IP is assigned, then retry the CSI enable with a retry loop. |
| **Memory fragmentation** | The listener uses several large static buffers (`lst_ring[LST_RING_SIZE]`, `csi_snap_pkt_t` copies, etc.) that are allocated at compile time. If the project later adds more global arrays, the static RAM may become fragmented, causing `malloc` failures in the CSI ISR. | Prefer static allocation for all per‑packet structures and avoid `malloc` inside ISR or tight‑loop code. If dynamic allocation is unavoidable, pool the buffers upfront and reuse them. |
| **Testing & verification** | The current codebase lacks a unit‑test harness for the packet parsing logic (e.g., `handle_incoming_udp`). This makes regression bugs hard to catch before flashing many boards. | Add a host‑side Python script that can inject serialized packets over a virtual COM port or UDP socket, exercising each magic type and validating that the firmware’s parsing logic behaves as expected. |
| **Security considerations** | The AP is open (`softAP(SSID, nullptr)`) and all UDP traffic is unauthenticated. An attacker on the same Wi‑Fi channel could inject fake HELLO/POLL packets, causing the listener to index spurious shouters or to emit false ranging reports. | Add a simple shared secret (e.g., a 4‑byte nonce) appended to each packet and verify it on receipt. For higher security, switch to WPA2‑PSK with a strong passphrase and use WPA2‑Enterprise for management traffic. |

---  

### Additional Recommendations for a Robust Deployment  

1. **Graceful degradation** – When the CSI snapshot buffer overflows, the shouter should still emit the `RANGE_RPT` (RSSI) packet; RSSI is the most critical data for the listener’s tracking algorithm.  
2. **Dynamic poll‑interval scaling** – If the listener detects repeated timeouts from a particular shouter, it can increase `POLL_TIMEOUT_MS` for that shouter only, rather than globally.  
3. **LED / GPIO diagnostics** – Map a GPIO to a LED that toggles on each major state transition (HELLO received, POLL sent, RESP received, ranging phase start). This provides a quick visual sanity check during field deployments.  
4. **OTA update path** – Since both sketches share the same Wi‑Fi interface, consider reserving a separate UDP port (e.g., 4444) for OTA commands, allowing the listener to push new firmware to the shouters without resetting the whole AP.  
5. **Logging to external storage** – For debugging large‑scale deployments, route the debug strings to an SD‑card file system (`/sd_log.txt`) instead of relying solely on the serial console, which can be overwhelmed by high‑frequency prints.  
6. **Use of sequence numbers in CSI snapshots** – Append a monotonically increasing `snap_seq` field to each `csi_snap_pkt_t`. The listener can then verify packet order and drop duplicates that may arise from ring‑buffer wrap‑around.  
7. **Channel fallback for ESP‑NOW** – If the STA interface changes channel due to a background scan, schedule a periodic re‑initialisation of the ESP‑NOW broadcast peer on the new channel to avoid losing beacon delivery.  

---  

### Summary  

The **ListenerV4** and **ShouterV4** sketches implement a tightly coupled, time‑synchronized ranging protocol that uses a combination of UDP for control‑plane messages and ESP‑NOW for low‑latency RSSI beacons. Their design deliberately keeps critical sections short, uses static ring buffers, and relies on magic‑number‑based packet validation to keep the two sides in lockstep.

The main challenges stem from **timing pressure** (especially during CSI‑snapshot bursts), **shared‑resource contention** (CSI buffers, ESP‑NOW peer tables), and **limited DRAM** on the shouter when storing many snapshots. Addressing the points in the table above—particularly improving timeout handling, adding versioning, tightening ESP‑NOW channel stability, and introducing graceful degradation—will make the system more resilient to network jitter, firmware upgrades, and scaling to larger numbers of shouters.