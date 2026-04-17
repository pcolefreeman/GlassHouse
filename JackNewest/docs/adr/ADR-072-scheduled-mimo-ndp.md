# ADR-072: Coordinated MIMO-like Scheduled NDP Injection with Cross-Node Time Sync

## 1. Status

**Proposed — blocked on prerequisite (Layer 1 time sync).**

- Author: Claude Opus 4.7
- Date: 2026-04-16
- Supersedes (when active): simple unscheduled periodic NDP from sibling Lane B
- Depends on: a new cross-node time-sync service that does not yet exist in firmware
- Related: ADR-029 (NDP sensing injection stub), ADR-060 (peer MAC filter),
  ADR-070 (ESP-NOW peer sensing — sibling), ADR-071 (WiFi-to-host bridge — sibling)

---

## 2. Context

GlassHouse v2 deploys a 4-node WiFi-CSI mesh (plus coordinator) on ESP32-S3 using
ESP-IDF. Each perimeter node:

- Runs in promiscuous mode and captures CSI for any frame whose source MAC is on
  its peer whitelist (`firmware/perimeter/main/csi_collector.c:170-205`).
- Can inject a minimal 24-byte NDP-like null-data frame via
  `csi_inject_ndp_frame()` (`firmware/perimeter/main/csi_collector.c:444-480`)
  using `esp_wifi_80211_tx(WIFI_IF_STA, ...)`.
- Labels its serialized frames with a monotonic per-node sequence counter
  (`firmware/perimeter/main/csi_collector.c:142-143` — `uint32_t seq = s_sequence++`),
  which is **local-only**: each node's counter is independent and not aligned
  with any other node.
- Identifies itself via `g_nvs_config.node_id`
  (`firmware/perimeter/main/nvs_config.h:30`, loaded by `nvs_config_load()`).

Basic coordination patterns already exist in this firmware. The heartbeat module
in `firmware/perimeter/main/heartbeat.c:17-47` fires a periodic timer that sends
a 1-byte UDP ping (`0xAA`) to the coordinator; per the Lane-E brief, deployed
variants of this pattern use a staggered phase offset of
`(node_id - 1) * (interval_ms / 4)` so nodes do not transmit simultaneously.
That existing stagger is **open-loop** — it relies on each node's independent
FreeRTOS/`esp_timer` clock and has no shared time reference. It proves the
architecture tolerates per-node scheduled TX, but it is not precise enough
(drift over minutes is many ms) to use as the scheduling primitive for a
sensing TDMA grid.

Sibling Lane B is activating a simple **unscheduled** 50 Hz periodic NDP from
each node. That works for raw I/Q throughput but does not solve the labeling
problem below, because multiple peers can fire inside one callback window.

Sibling ADR-070 proposes moving peer sensing traffic to ESP-NOW broadcasts;
sibling ADR-071 proposes a WiFi-to-host data path replacing USB-Serial/JTAG.
This ADR is orthogonal to the data-egress path (ADR-071) and interacts with
the TX medium choice (ADR-070) — see §8 and §12.

---

## 3. Problem

Three concrete problems motivate this ADR:

**(a) Blended per-peer CSI.**
When `wifi_csi_callback` fires at `csi_collector.c:164`, the only evidence of
which peer transmitted the probe is the source MAC matched at
`csi_collector.c:172-177`. If two or more peers transmit probes within the
same CSI callback window, the ESP32-S3 driver surfaces *one* callback per
received frame but the receiver has no way to reason about **relative** timing
(i.e. "peer A transmitted at t=0.0 ms, peer B at t=0.1 ms"). The resulting
per-(tx, rx)-pair CSI stream rate is unknown and non-uniform.

**(b) No shared time reference.**
The per-node `s_sequence` counter at `csi_collector.c:142-143` is independent.
There is no firmware-level concept of "slot 17 on node 1 corresponds to
slot 17 on node 2." Without this, coordinated TDMA is impossible; the best
we have is the open-loop heartbeat stagger (§2), which drifts ~20 ppm per
the ESP32-S3 crystal spec — i.e. ~1 ms drift in ~50 s.

**(c) Multistatic-radar inference impossible.**
Useful multistatic / MIMO-like techniques (bistatic range, Doppler separation
by TX geometry, angle-of-arrival from cross-node phase alignment) all require
knowing *which* transmitter produced *which* CSI snapshot with sub-symbol
timing precision. Today's firmware cannot produce that labeling.

---

## 4. Decision

Introduce **two layers**, shipped in order:

- **Layer 1 (L1) — Cross-node time sync** targeting **≤ 1 ms** median offset
  between any two perimeter nodes, with an explicit re-sync cadence that
  bounds drift from the ESP32-S3 crystal (~20 ppm).
- **Layer 2 (L2) — Slotted NDP schedule** layered on L1: a periodic TDMA frame
  in which exactly one node's `csi_inject_ndp_frame()`
  (`csi_collector.c:444-480`) fires in each slot, with a guard band wide enough
  to absorb residual L1 sync error.

**L2 is inert until L1 ships.** L2 code paths compile but do nothing until an
L1 status flag reports "synced." Before that, nodes fall back to Lane B's
unscheduled 50 Hz periodic NDP. This makes the ADR safe to land incrementally.

---

## 5. Architecture (ASCII)

TDMA frame: 20 ms period, 4 nodes, 5 ms per slot → 50 Hz total NDP rate
(each node TX at 50 Hz / 4 = 12.5 Hz, but every receiver observes 50 Hz total).

```
 Frame N                                   Frame N+1
 |<----------------- 20 ms ---------------->|
 +---------+---------+---------+---------+  +---------+ ...
 | slot 0  | slot 1  | slot 2  | slot 3  |  | slot 0  |
 | node 1  | node 2  | node 3  | node 4  |  | node 1  |
 |  TX NDP |  TX NDP |  TX NDP |  TX NDP |  |  TX NDP |
 +---------+---------+---------+---------+  +---------+
  0ms      5ms       10ms      15ms      20ms

 Per-slot (5 ms) breakdown:
 +-------+----------------------+-----------+
 | guard |  NDP TX + airtime    |  guard    |
 |  1 ms |  ~24us preamble +    |   ~1 ms   |
 |       |  driver queue jitter |           |
 +-------+----------------------+-----------+

 Receivers (all 3 non-TX nodes in each slot) run wifi_csi_callback
 and tag the frame with (tx_node = slot_owner, rx_node = self.node_id).
```

Slot ownership is strictly `slot_id = (node_id - 1) mod 4`. A receiver that
misses the first slot of a frame can still infer `tx_node` from
`slot_number mod 4 + 1`, cross-checked against `info->mac` against the peer
MAC table populated via `csi_collector_add_peer()` (`csi_collector.c:257-274`).

---

## 6. Layer 1 — Time-Sync Design

### 6.1 Options considered

**Option 1A: Coordinator-as-master, timestamp in Vendor Specific IE (VSIE) or
periodic UDP broadcast.**
The coordinator (`firmware/coordinator/main/main.c`) already hosts a SoftAP
(`CSI_NET_V2`, channel 1, `main.c:19-22`) to which all perimeter nodes
associate, and it already receives per-node UDP packets on port 4210. A new
periodic "time beacon" — either injected as a VSIE in the SoftAP's beacon
frame or broadcast as a tiny UDP packet from the coordinator on a known port
— would carry the coordinator's `esp_timer_get_time()` (microseconds since
boot) plus a monotonic beacon sequence number. Each perimeter node
timestamps beacon **reception** against its own `esp_timer_get_time()` and
computes an offset. Expected precision: **0.3–1.5 ms** after a simple PI
filter over 5–10 beacons, dominated by SoftAP beacon jitter (~0.5 ms) and
lwIP/UDP stack delay variance (UDP path).

**Option 1B: SNTP over SoftAP.**
The coordinator runs an SNTP server; perimeter nodes run LwIP's SNTP
client (`esp_netif_sntp_*`). SNTP is designed for wall-clock time (seconds)
and its typical precision over a LAN is ~10 ms without active filtering.
Reaching 1 ms would require a non-trivial custom implementation on top of
the SNTP framing. Also, SNTP binds to the `time()` wall clock, which is
harder to reason about than a monotonic timer for TDMA slot math.

**Option 1C: IEEE 1588 (PTP).**
Gold standard (sub-microsecond) but requires hardware timestamping support.
ESP32-S3 WiFi hardware does **not** expose PTP-grade TX/RX timestamping on
802.11 frames; a pure-software PTP over WiFi would devolve into Option 1A.

### 6.2 Decision — 1A: Coordinator-as-master, UDP time beacon

**Why 1A:**

- Re-uses an existing trust boundary (coordinator is already the TX/RX hub).
- Single clock domain (coordinator's `esp_timer_get_time()`) → no wall-clock
  ambiguity; TDMA math stays in microseconds from boot.
- Target precision (≤1 ms) is comfortably achievable with a UDP beacon at
  10 Hz plus a one-pole smoothing filter on offset.
- No new hardware dependency; no PTP stack to port.
- VSIE variant (injecting timestamp into SoftAP beacons) is a viable
  **future tightening** if the UDP path is too jittery — it bypasses lwIP
  entirely and lands in the RX callback directly, plausibly reaching
  ~200 µs.

**Concrete parameters:**

| Parameter | Value |
|---|---|
| Beacon rate | 10 Hz (every 100 ms) |
| Beacon payload | `{uint32_t magic, uint32_t seq, uint64_t coord_us}` |
| Re-sync cadence | Continuous (beacon at 10 Hz, filter updates every beacon) |
| Filter | 1-pole IIR on `(coord_us − local_us)`, alpha = 0.2 |
| Drift bound between beacons | 20 ppm × 100 ms = 2 µs (negligible vs ~1 ms target) |
| Sync-healthy flag | Set once 5 consecutive beacons arrive within ±2 ms of filtered offset |
| Sync-lost flag | Cleared after 3 consecutive missing beacons (300 ms) |

When sync-lost, L2 degrades to open-loop (pre-computed schedule drifts at
~20 ppm — 1 ms per 50 s — and receivers rely on MAC labeling until a
beacon resumes).

---

## 7. Layer 2 — Slotted NDP Schedule

### 7.1 TDMA frame

- **Frame period:** 20 ms (`T_frame`)
- **Slot count:** 4 (equals `g_nvs_config.tdm_node_count` today;
  `nvs_config.h:37`)
- **Slot duration:** 5 ms (`T_slot = T_frame / 4`)
- **Guard band:** 1 ms leading + 1 ms trailing per slot (final design; P1
  ships with 2 ms + 2 ms for safety)
- **Active TX window:** 3 ms in P2 (5 − 2 guards), 1 ms in P1
- **Total NDP rate observed by each receiver:** 4 × (1 / 20 ms) = 200 Hz
  worst-case if all 4 slots fire; **50 Hz per-(tx,rx)-pair** effective after
  RX rate-limiting at `CSI_MIN_SEND_INTERVAL_US` (`csi_collector.c:61`)

### 7.2 node_id → slot_id mapping

```
slot_id = (g_nvs_config.node_id - 1) mod 4        // for node_id ∈ {1..4}
slot_phase_us = slot_id * (T_slot_us)             // = slot_id * 5000
next_tx_us = (floor(sync_now_us / T_frame_us) + 1) * T_frame_us + slot_phase_us
```

The transmitter arms a single-shot `esp_timer` for `next_tx_us`, fires
`csi_inject_ndp_frame()` (`csi_collector.c:444-480`), and immediately
re-arms for `next_tx_us + T_frame_us`.

### 7.3 Guard time and time-sync error budget

L1 targets ≤1 ms offset. P1 uses 2 ms guards at each slot boundary, so
two neighbor slots can each be 1 ms off without colliding. Once L1 is
measured to hold ≤1 ms for an hour (acceptance in §11), P2 tightens
guards to 1 ms each.

### 7.4 Receiver-side labeling

On `wifi_csi_callback` (`csi_collector.c:164`), the receiver computes:

```
rx_us = esp_timer_get_time()
slot_number = floor((rx_us + local_offset_us) / T_slot_us)
inferred_tx_node = (slot_number mod 4) + 1
actual_tx_node = peer_node_ids[ index_where(peer_macs[i] == info->mac) ]
if inferred_tx_node != actual_tx_node: label_mismatch_count++
```

The **MAC-derived** `actual_tx_node` is authoritative (it is already used at
`csi_collector.c:172-177`). The slot-derived `inferred_tx_node` is a
diagnostic check: a mismatch rate > 1 % signals sync drift or slot collision
and should flip L2 to a degraded mode.

### 7.5 Interaction with Lane B's SAR_AMP stream

Lane B activates a new 48-sample batched SAR_AMP frame at magic `0xC5110007`
(distinct from the existing `CSI_MAGIC` used by `csi_serialize_frame` at
`csi_collector.c:126-127`). That stream is an **egress** path from receiver
to coordinator — it is unaffected by L2 because L2 only controls *who
transmits a probe and when*. The new magic differentiates batched SAR
frames from per-CSI-callback frames on the downstream decoder, so L2 adds
no new decoder work.

Collision risk: Lane B's unscheduled 50 Hz NDP must be **disabled** when L2
is active (i.e. when `sync_healthy == true`). This is a one-bit gate.

---

## 8. Backward Compatibility

- **Ambient CSI unchanged.** Promiscuous-mode CSI on ambient (non-peer-NDP)
  traffic still fires and still passes the MAC whitelist
  (`csi_collector.c:170-205`). L2 only *adds* scheduled probe frames; it
  does not suppress ambient CSI.
- **Lane B is subsumed when L2 activates.** Lane B's simple 50 Hz periodic
  NDP is equivalent to L2 running with `tdm_node_count = 1, guard = 0`,
  i.e. a degenerate schedule. Once `sync_healthy` latches true, the
  per-node timer switches from "fire every 20 ms regardless" to "fire at
  `next_tx_us` per the schedule."
- **ADR-071 (WiFi-to-host) independence.** L2 produces the same
  ADR-018-format CSI frames (`csi_collector.c:82-95`) that ADR-071
  forwards. L2 only changes frame **labeling timing**, not frame format.
- **ADR-070 (ESP-NOW) interaction.** See §12.

---

## 9. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **ESP32-S3 crystal drift (~20 ppm).** 1 ms drift per 50 s free-running. | High | Medium | L1 beacon at 10 Hz → drift between beacons = 2 µs, well inside budget. |
| R2 | **Missed sync beacon** due to WiFi congestion / lwIP buffer pressure. | Medium | Medium | 3-strike rule (§6.2): lose sync after 300 ms; revert L2 to degraded (MAC-only) labeling until beacon returns. |
| R3 | **Slot boundary vs ambient CSI race.** A non-peer frame arriving mid-slot can cause the RX callback queue to back up and bleed a probe into the next slot's window. | Medium | Medium | `CSI_MIN_SEND_INTERVAL_US` rate limit (`csi_collector.c:61`) already back-pressures egress; P1's 4 ms guard absorbs a 1–2 ms stall. |
| R4 | **Frame-rate vs MTU budget.** 50 Hz × 4 RX nodes × ~208 B ADR-018 frame = **41.6 KB/s per RX node egress**. Lane B pushes this up with 48-sample batches. SoftAP has ~3 Mbit/s usable at channel 1 — headroom ≈ ×70. Confirmed OK but headroom shrinks if the 4-node mesh scales. | Low | Low | Rate-limit at `csi_collector.c:223` already caps per-peer egress. Scale probe rate down linearly with node count if mesh grows. |
| R5 | **Coordinator as single point of failure** for L1. If the coordinator dies, all nodes lose sync within 300 ms. | Medium | High | P2 adds "sticky sync" — after losing the coordinator, nodes free-run on last-known offset for up to 60 s (drift bound: 20 ppm × 60 s = 1.2 ms, still inside 2 ms guard). A future ADR can add master-election among perimeter nodes. |
| R6 | **Channel semantics collision with ADR-070 (ESP-NOW).** ESP-NOW uses its own channel state; scheduling across mixed SoftAP/ESP-NOW channels may be impossible. See §12. | Medium | High | Defer L2 + ADR-070 composition until both are in P1. |

---

## 10. Migration Phases

- **P0 — L1 only.**
  Ship coordinator UDP time beacon + perimeter sync client. No L2 behavior
  change. **Success criterion:** median inter-node offset ≤ 1 ms measured
  continuously for 1 hour across 4 nodes; `sync_lost` events < 5 per hour.
- **P1 — L2 with 2 ms guard per side.**
  Enable slotted NDP when `sync_healthy`. Receivers log slot-vs-MAC label
  match rate. Success: label match ≥ 99 %; zero observed slot collisions
  over 1 hour.
- **P2 — Tighten guard to 1 ms + sticky-sync fallback (R5).**
- **P3 — Multistatic inference layer (OUT OF SCOPE of this ADR).**

---

## 11. Acceptance Criteria

| Layer | Metric | Target |
|---|---|---|
| L1 | Median pairwise offset (any two nodes) | ≤ 1 ms |
| L1 | P99 pairwise offset | ≤ 2 ms |
| L1 | Uptime with `sync_healthy = true` | ≥ 99 % over 1 hour |
| L2 | Slot-start jitter (TX time vs scheduled `next_tx_us`) | ≤ 500 µs |
| L2 | (tx,rx) labeling accuracy (slot-derived vs MAC-derived) | ≥ 99 % |
| L2 | Zero overlap between any two nodes' TX windows | 0 observed collisions / 1 h |

---

## 12. Rejected Alternatives

**GPS PPS (1 pulse-per-second).**
Rejected. The SAR deployment target is **indoor** search-and-rescue; GPS
fixes are unreliable or absent inside structures. Using GPS would kill the
primary use case.

**Wire-based sync (1-wire bus / shared SYNC GPIO).**
Rejected. The "toss-and-run" deployment pattern of these nodes means
physical cabling between units is impractical at deploy time. A SYNC wire
also re-introduces the physical deployment friction that the mesh was
designed to eliminate.

**Pure open-loop stagger (scale up heartbeat.c:17-47 pattern to sensing).**
Rejected. The heartbeat stagger is fine at 1 Hz where 20 ppm drift is
irrelevant, but at 50 Hz sensing with 5 ms slots, uncorrected drift
crosses a slot boundary in ~50 s. Unusable without L1.

**Channel-hopping as de facto TDMA** (rely on `csi_hop_next_channel` at
`csi_collector.c:370-393`).
Rejected. Channel hopping already exists for diversity
(`csi_collector.c:66-73`), but it does not select **which node** is TXing —
it only changes **which channel** all nodes listen on. Orthogonal to the
labeling problem.

---

## 13. Open Question — Interaction with ADR-070 (ESP-NOW)

ADR-070 proposes moving peer sensing TX from `esp_wifi_80211_tx` on the
station interface to ESP-NOW broadcasts. ESP-NOW and SoftAP share the WiFi
radio but have distinct channel state machines: under ESP-NOW, the node's
channel is set via `esp_now_peer_info_t.channel`, and mid-frame channel
switches are not guaranteed to preserve TX timing. If ADR-070 lands before
L2, the L2 scheduler must fire `esp_now_send()` rather than
`esp_wifi_80211_tx()`, and the per-peer channel field must either (a) all
equal the SoftAP channel or (b) include a channel-switch guard in the slot
budget. This ADR does **not** resolve that; it is an explicit open question
that must be closed before L2 ships if ADR-070 is already active.

---

## 14. References

- `firmware/perimeter/main/csi_collector.c:142-143` — per-node sequence counter
- `firmware/perimeter/main/csi_collector.c:164-244` — `wifi_csi_callback`, peer MAC filter
- `firmware/perimeter/main/csi_collector.c:444-480` — `csi_inject_ndp_frame()`
- `firmware/perimeter/main/csi_collector.c:257-274` — `csi_collector_add_peer()`
- `firmware/perimeter/main/heartbeat.c:17-47` — existing staggered periodic TX
- `firmware/perimeter/main/nvs_config.h:30,37` — `node_id`, `tdm_node_count`
- `firmware/coordinator/main/main.c:19-23` — SoftAP channel/port config
- ESP-IDF WiFi API: `esp_wifi_80211_tx`, `esp_timer_get_time`
- ESP32-S3 datasheet — 40 MHz crystal, typical ±20 ppm
- ADR-029 — NDP sensing injection stub (predecessor)
- ADR-060 — peer MAC filter (predecessor)
- ADR-070 — ESP-NOW peer sensing migration (sibling, see §13)
- ADR-071 — WiFi-to-host data bridge (sibling, orthogonal)
- IEEE 1588-2019 (PTP) — for context in §6.1 Option 1C
