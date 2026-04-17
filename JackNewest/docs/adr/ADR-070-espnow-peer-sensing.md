# ADR-070: Migrate peer sensing traffic from SoftAP+UDP to ESP-NOW

## 1. Status

**Proposed.**

- Author: Claude Opus 4.7
- Date: 2026-04-16
- Supersedes: (none — extends the transport layer only)
- Series: this ADR is part of a 3-ADR cluster covering the v2.1 transport rework.
  See sibling **ADR-071** (WiFi-to-host bridge) and sibling **ADR-072** (MIMO /
  time-sync).

---

## 2. Context

GlassHouse v2 is a 4-node perimeter WiFi-CSI sensor mesh on ESP32-S3
(ESP-IDF) used for search-and-rescue (SAR) indoor sensing. Today:

- The coordinator runs a **SoftAP** on channel 1 with SSID `CSI_NET_V2`
  (`firmware/coordinator/main/main.c:19-22`, `main.c:98-114`) and a
  UDP listener on port 4210 (`main.c:23`, `main.c:142-182`).
- Each perimeter node **associates as a STA to that SoftAP** and sends
  CSI/vitals/I-Q/heartbeat/link-report packets over a standard IPv4+UDP
  socket (`firmware/perimeter/main/stream_sender.c:36-57`,
  `stream_sender.c:109-122`).
- CSI callbacks run in promiscuous mode
  (`firmware/perimeter/main/csi_collector.c:304-312`) and are filtered by a
  multi-MAC peer whitelist
  (`csi_collector.c:31-34`, `csi_collector.c:170-186`). Only frames whose
  source MAC matches a configured peer produce a link report via
  `link_reporter_record()` (`csi_collector.c:199`,
  `firmware/perimeter/main/link_reporter.c:62-100`).
- The coordinator’s UDP listener forwards each packet, COBS-encoded, to the
  host over USB-Serial/JTAG (`main.c:164-181`). `MAX_PKT_SIZE = 512`
  (`main.c:31`). Oversized packets are silently dropped (`main.c:167`).

### Traffic sources on each perimeter node (all UDP today)

| Packet                 | Magic         | Size       | File:line                                              | Typical rate |
|------------------------|---------------|-----------:|--------------------------------------------------------|--------------|
| CSI ADR-018 frame      | `0xC5110001`  | 20 + N·2·2 | `csi_collector.c:96-159`, `csi_collector.h:15-21`      | ≤50 Hz (rate-limited, see below) |
| Vitals packet          | `0xC5110002`  | 32 B exact | `edge_processing.h:28`, `edge_processing.h:96-111`     | 1 Hz (`vital_interval_ms` default 1000) |
| Feature vector (ADR-069)| `0xC5110003` | 48 B exact | `edge_processing.h:114-125`                            | 1-10 Hz |
| Fused vitals (ADR-063) | `0xC5110004`  | 48 B exact | `edge_processing.h:128-154`                            | 1 Hz |
| Compressed frame       | `0xC5110005`  | variable   | `edge_processing.h:29`                                 | event-driven |
| Raw I/Q stream         | `0xC5110006`  | 8 + ≤1024  | `edge_processing.c:718-742`, `edge_processing.h:30-37` | burst |
| Heartbeat ping         | 1 byte `0xAA` | 1 B        | `firmware/perimeter/main/heartbeat.c:17-22`            | configurable (typical 1-5 Hz) |
| Link report            | `0x01` (type) | 10 B packed| `link_reporter.c:33-42`, `link_reporter.c:127-130`     | configurable |

### Key numbers grounded in the code

- **Rate limit on CSI upload: `CSI_MIN_SEND_INTERVAL_US = 20 ms`** → **50 Hz
  hard cap** on off-device CSI sends (`csi_collector.c:61`,
  `csi_collector.c:222-236`). This exists because in promiscuous mode the
  CSI callback fires 100-500+ Hz and lwIP exhausts its pbuf pool (ENOMEM),
  crashing the node.
- **ENOMEM backoff is 100 ms** with suppressed-send counting
  (`stream_sender.c:32-33`, `stream_sender.c:88-107`) — i.e. when the
  lwIP buffers are exhausted we stop sending entirely for 100 ms. This is
  observable in the field as a periodic rate collapse.
- **Coordinator MAX_PKT_SIZE = 512 B** (`main.c:31`). Anything larger is
  dropped at `recvfrom`, which caps the raw-I/Q packet to ~500 B payload
  even though the serializer allocates up to `CSI_MAX_FRAME_SIZE = 20 +
  4·256·2 = 2068 B` (`csi_collector.h:21`). In practice the edge pipeline
  already enforces `EDGE_MAX_IQ_BYTES = 1024` (`edge_processing.h:34`),
  so the transport, not the DSP, is the upper limit.
- **Observed off-device rate: ~20 Hz per channel** (`firmware/perimeter/README.md:15`,
  `README.md:113`). This is well below the 50 Hz cap — i.e. lwIP, not CSI
  generation, is the current bottleneck.

### Topology assumption the current design bakes in

Every perimeter node must associate to the coordinator’s SoftAP on channel
1 before any packet can flow. If the coordinator reboots or the STA
association drops, all sensing traffic stops until reassociation completes
(typically seconds). This is fine for a fixed lab install but poor for a
SAR deployment.

---

## 3. Problem

For SAR deployment (responders drop nodes into an unknown building and
expect high-rate sensing within a few seconds), the current
UDP-over-SoftAP transport has four concrete limiters:

1. **Startup dependency**: every perimeter node is useless until the
   coordinator’s SoftAP is up and the STA has DHCP’d. No sensing during
   coordinator reboot.
2. **Link-loss amplification**: a single dropped association cascades into
   seconds of dead sensor output (re-assoc + DHCP + ARP). For SAR, where
   we already assume flaky RF, this is the wrong failure mode.
3. **lwIP pbuf pressure**: the `CSI_MIN_SEND_INTERVAL_US = 20 ms` cap and
   the ENOMEM cooldown
   (`csi_collector.c:61`, `stream_sender.c:88-107`) show lwIP is the
   bottleneck on CSI throughput, not airtime or CPU. Every frame pays for
   IP+UDP headers, ARP, and lwIP queueing.
4. **Per-packet overhead**: UDP over 802.11 adds 802.11 MAC (24 B) +
   LLC/SNAP (8 B) + IP (20 B) + UDP (8 B) = **~60 B of framing per
   packet**. For a 32 B vitals packet that’s ~188% overhead. ESP-NOW is
   a single 802.11 action-vendor frame with only the MAC header.

Benchmarks reported by ESP-IDF community testing put ESP-NOW unicast
end-to-end latency at ~3 ms vs UDP-over-SoftAP at ~15 ms, and we should
expect similar for broadcast.

---

## 4. Decision

> **Replace per-peer UDP sensing traffic with ESP-NOW broadcasts.**
> **Keep UDP (or its successor per sibling ADR-071) only for the
> coordinator-to-host bridge.**

Concretely:

- Each perimeter node sends CSI/vitals/I-Q/feature/fused/heartbeat/link-report
  packets as ESP-NOW frames. Broadcast by default (one TX reaches all
  peers and the coordinator simultaneously).
- The coordinator receives ESP-NOW frames in its `esp_now_register_recv_cb`
  and COBS-forwards them to the host exactly as it does today for UDP
  (`main.c:164-181`). The USB-Serial/JTAG leg is unchanged — see sibling
  ADR-071 for proposals to replace that leg.
- The SoftAP is retained **only** to keep `esp_wifi_set_promiscuous(true)`
  and CSI capture working on a known channel. Perimeter STAs **no longer
  associate**; they stay unassociated on the same channel. (See §8
  “Channel management” and “coexistence” risks.)
- `stream_sender.{c,h}` is replaced / wrapped by `esp_now_sender.{c,h}`
  exposing the same `int stream_sender_send(const uint8_t *data, size_t
  len)` signature, so `csi_collector.c:224`, `edge_processing.c:741`, etc.
  do not change.

---

## 5. Architecture diagram

### Before (today, at commit `ffc103c`)

```
           perimeter node N1 (STA)               perimeter node N2 (STA)
           ┌──────────────────┐                  ┌──────────────────┐
           │ csi_collector    │                  │ csi_collector    │
           │ promisc CSI cb   │                  │ promisc CSI cb   │
           │   └─ filter MAC  │                  │   └─ filter MAC  │
           │ edge_processing  │                  │ edge_processing  │
           │ stream_sender.c  │                  │ stream_sender.c  │
           │ sendto() ─► UDP  │                  │ sendto() ─► UDP  │
           └────────┬─────────┘                  └────────┬─────────┘
                    │ 802.11 STA->AP                       │
                    │ (assoc + DHCP required)             │
                    ▼                                      ▼
                ┌────────────────────── coordinator (SoftAP) ──────────────┐
                │ SSID CSI_NET_V2, ch 1, WPA2  (main.c:98-114)             │
                │ UDP :4210  (main.c:23, 142-182)                          │
                │ COBS-encode + USB-Serial/JTAG FIFO  (main.c:47-89,       │
                │ 164-181)                                                 │
                └──────────────────────────┬───────────────────────────────┘
                                           ▼
                                   host (laptop) / Python
```

Observations flow N1↔N2 *only* via CSI on received 802.11 frames; every
useful sensor byte must still round-trip through the coordinator as UDP.

### After (proposed, ADR-070)

```
    ┌──────────────────┐  ESP-NOW broadcast    ┌──────────────────┐
    │ perimeter N1     │◄───────ff:ff:ff:ff:ff:ff──────►│ perimeter N2 │
    │ esp_now_sender.c │                       │ esp_now_sender.c │
    └─────────┬────────┘                       └─────────┬────────┘
              │                                          │
              │            (every node hears every       │
              │             other node's CSI-trigger     │
              │             probe AND payload directly)  │
              ▼                                          ▼
              ┌────────── coordinator (SoftAP, STA not assoc) ──────────┐
              │ esp_now_recv_cb  → COBS → USB-Serial/JTAG               │
              │ (SoftAP still up for ch-lock + CSI promisc, no DHCP)    │
              └───────────────────────────┬─────────────────────────────┘
                                          ▼
                                  host (laptop) / Python
```

All sensing is peer-to-peer; the coordinator becomes a pure "sniff and
forward to USB" node with no routing duties.

---

## 6. Interface design

### 6.1 File layout

- **New**: `firmware/perimeter/main/esp_now_sender.c` + `esp_now_sender.h`
- **Delete** (Phase C, see §7): `firmware/perimeter/main/stream_sender.c/.h`
- **Unchanged**: `csi_collector.c`, `edge_processing.c`, `heartbeat.c`,
  `link_reporter.c` — they continue to call `stream_sender_send()`. The
  symbol is re-exported by `esp_now_sender.c` for ABI compatibility.

Proposed header signature:

```
int  esp_now_sender_init(void);
int  esp_now_sender_add_peer(const uint8_t mac[6]);  // called 3x at boot
int  esp_now_sender_send_broadcast(const uint8_t *data, size_t len);
int  esp_now_sender_send_unicast(const uint8_t mac[6], const uint8_t *d, size_t n);

// Back-compat shim (same signature as stream_sender.c:80):
int  stream_sender_send(const uint8_t *data, size_t len);
```

### 6.2 ESP-NOW peer registration flow

At boot, each node registers:

1. The **broadcast MAC** `ff:ff:ff:ff:ff:ff` as a peer — required by
   ESP-NOW even for broadcast TX.
2. Each of the **other 3 perimeter MACs** from NVS (`nvs_config.h:57`
   already stores `peer_macs[4][6]` and `peer_count`; the existing
   provisioning script populates this — see
   `firmware/perimeter/provision.py`).
3. The **coordinator MAC** as an explicit peer (for optional unicast
   "urgent" frames; see §6.3).

Failure modes: if a peer MAC is not registered, `esp_now_send` returns
`ESP_ERR_ESPNOW_NOT_FOUND` and no frame leaves. The init path must fail
loudly rather than silently drop.

### 6.3 Broadcast vs unicast tradeoff

| | Broadcast (`ff:ff:ff:ff:ff:ff`) | Unicast per-peer |
|---|---|---|
| TX events per frame | 1 | N_peers (3) |
| Ack / retry | **no** | yes (up to 3 retries at MAC layer) |
| Reliability | best-effort only | reliable within range |
| Airtime | minimal | N× broadcast |
| CSI-trigger side-effect | 1 TX → every neighbor captures CSI from that frame | N TXs → N CSI events per neighbor pair, but each is only from the intended recipient’s MAC |
| Fits the SAR model | yes (one-to-many drop-in network) | no (assumes known topology) |

**Decision**: use broadcast for routine sensor data (vitals, CSI, I-Q,
heartbeats), and reserve unicast for **control frames** (OTA, WASM upload
tokens, SAR-mode enable) where reliability matters more than airtime.

### 6.4 Magic-number → ESP-NOW mapping

ESP-NOW carries an arbitrary payload up to 250 B inside a single vendor
action frame. Each sender prepends the existing on-wire format unchanged:

| Magic         | Today’s size | Fits in 250 B? | Strategy |
|---------------|-------------:|---------------:|----------|
| `0xC5110001` CSI (`csi_collector.h:15`)        | 20 + iq_len, cap ~500 B (coord MAX_PKT_SIZE)  | **Often no** for full 52-subcarrier HT20 (20+208=228 B fits; 20+104=124 fits; HT40 does **not**) | Fragment or route over UDP fallback |
| `0xC5110002` Vitals (32 B, `edge_processing.h:111`) | 32 B | yes | native ESP-NOW |
| `0xC5110003` Feature (48 B) | 48 B | yes | native |
| `0xC5110004` Fused vitals (48 B, `edge_processing.h:154`) | 48 B | yes | native |
| `0xC5110005` Compressed | variable | usually yes | native, error-check len |
| `0xC5110006` Raw I/Q (8 + iq_len, up to 1032 B)| up to 1032 B | **no for full bursts** | Fragment |
| `0xAA` heartbeat (1 B) | 1 B | trivially yes | native |
| `0x01` link report (10 B)| 10 B | yes | native |

> **Note on “SAR_AMP_MAGIC”.** The task brief references a `SAR_AMP_MAGIC`
> packet type. A `grep` over the worktree at commit `ffc103c` finds **no
> occurrence** of `SAR_AMP_MAGIC`. **Assumption flagged explicitly**: either
> this is a proposed/future magic number owned by a sibling ADR (likely
> ADR-072), or it is a stale name from an earlier spec. This ADR treats
> it as reserved and allocates `0xC5110007` for it in the wire-format
> table, to be confirmed in sibling ADR-072.

### 6.5 MTU constraint — 250 B hard limit

ESP-NOW’s max payload per frame is **250 bytes** (ESP-IDF
`esp_now_send()` constraint; see §11 references). Two consequences:

1. **CSI ADR-018 frames for wide bandwidth do not fit**. HT20 with 52
   subcarriers × 2 streams × 2 bytes I/Q = 208 B + 20 B header = 228 B
   (fits), but HT40 or higher subcarrier counts do not.
2. **Raw I/Q bursts (up to 1032 B) do not fit** in a single frame.

Two possible strategies; this ADR prescribes **both**, gated by payload
size:

- **Sub-250-B payloads** → single ESP-NOW frame. Applies to all vitals/
  feature/fused/heartbeat/link-report traffic unconditionally.
- **Over-250-B payloads** → either:
  - **(Preferred)** split into numbered fragments with a 4-B fragment
    header (magic `0xC5110FFF`, seq, fragment_index, total_fragments),
    reassembled at the coordinator. This keeps the code path uniform.
  - **(Fallback)** keep UDP-over-SoftAP for these specific frames during
    Phase B of migration (see §7).

---

## 7. Migration phases

### Phase A — Dual-stack: ESP-NOW added alongside UDP (non-breaking)

- Add `esp_now_sender.c` with the same `stream_sender_send()` symbol
  guarded by a build-time flag `CONFIG_TRANSPORT_ESPNOW` (default **n**).
- SoftAP remains, STA assoc remains, UDP remains the primary path.
- Coordinator gains an `esp_now_recv_cb` that COBS-forwards exactly like
  the UDP bridge.
- **Acceptance criterion (A)**: with `CONFIG_TRANSPORT_ESPNOW=y` on all 4
  perimeter nodes and the coordinator, host-side decoder produces the
  same decoded vitals/CSI stream as the UDP build over a 10-minute soak,
  with **≥95% packet parity** and **zero USB-side COBS errors**.

### Phase B — ESP-NOW default, UDP fallback only for >250-B frames

- Flip Kconfig default to `CONFIG_TRANSPORT_ESPNOW=y`.
- `esp_now_sender_send` returns an ENOTSUP-like error for payload
  `len > 250` and the caller falls back to the legacy `stream_sender`
  UDP path for that single frame.
- STA association to SoftAP is **still used** but only for those large
  frames.
- **Acceptance criterion (B)**: ≥99% of *all* frames leave via ESP-NOW;
  <1% use UDP fallback. Median end-to-end latency
  (perimeter-sensor-event → host-USB-byte) drops by ≥5 ms relative to
  Phase A measurements.

### Phase C — Fragmentation added, UDP sensing path deleted

- Implement §6.5 fragmentation in `esp_now_sender.c` and
  reassembly in the coordinator.
- Remove `stream_sender.c/.h`. Remove STA assoc on perimeter nodes.
- SoftAP on coordinator is retained *only* for CSI channel lock and
  (optionally) for the host-bridge path described in sibling ADR-071.
- **Acceptance criterion (C)**: perimeter node with **no STA assoc** runs
  for 60 min, produces the full packet taxonomy to the host, reassembles
  all fragmented frames with <0.1% fragment loss, and survives
  coordinator reboot with <1 s gap in vitals output (vs today’s ~5-10 s).

---

## 8. Risk register

| # | Risk | Likelihood | Severity | Mitigation |
|---|------|-----------|----------|------------|
| R1 | **ESP-NOW 250-B MTU vs existing frame sizes.** Raw I/Q (up to 1032 B, `edge_processing.h:34`) and wide-bandwidth CSI frames will not fit. | **High** | High | Phase B keeps UDP fallback for >250-B frames; Phase C implements fragmentation. Reject this ADR entirely if fragmentation is rejected. |
| R2 | **Interaction with promiscuous mode + CSI callback.** Today CSI fires on *any* received frame including ESP-NOW broadcasts (`csi_collector.c:304-312`). Every ESP-NOW TX by node N1 will itself generate a CSI callback on N2/N3/N4. Volume increases. | **High** | Med | The `CSI_MIN_SEND_INTERVAL_US=20 ms` rate limit (`csi_collector.c:61`) still applies, but the ENOMEM cooldown is lwIP-scoped and ESP-NOW bypasses lwIP — so the current rate limit logic should actually become *less* pressured, not more. Still, revisit `CSI_MIN_SEND_INTERVAL_US` with measurement in Phase A. |
| R3 | **Channel management.** ESP-NOW requires all peers on the same primary channel. Today `csi_collector.c:286-299` auto-detects the AP channel, but the channel-hop feature (`csi_collector.c:64-79`, `csi_collector.c:370-440`) will break peer reachability mid-hop. | **High** | High | Until ADR-072 resolves TDM+hopping, **disable channel hopping when `CONFIG_TRANSPORT_ESPNOW=y`** (enforce `hop_count=1`). Document in Kconfig help. |
| R4 | **Memory overhead of ESP-NOW peer state.** ESP-NOW allocates ~80 B per registered peer plus ~2 KB for the ESP-NOW task. For 4 perimeter + 1 coordinator + 1 broadcast = 6 peers, budget ≈ 2.5 KB heap. | **Low** | Low | Document in §9 acceptance criteria. Falls well under any per-node budget. |
| R5 | **ESP-NOW + SoftAP coexistence on the same interface.** ESP-NOW can run in AP, STA, or APSTA mode; it shares the radio with the interface. **Assumption**: per ESP-IDF docs, ESP-NOW in APSTA mode on the coordinator will coexist with the SoftAP, as long as the primary channel matches. This is stated as an assumption to verify in Phase A. | Med | High | Phase A acceptance test explicitly exercises simultaneous SoftAP-serving + ESP-NOW-receiving on the coordinator. If coexistence fails, fall back to: retain SoftAP on coordinator only for channel lock (no STA clients), or split coordinator into two ESP32s (one AP-only, one ESP-NOW-only). |
| R6 | **Lost reliability on broadcast.** Broadcast ESP-NOW has no MAC-layer retry. Real-world loss with 4 nodes at SAR-realistic RSSI (−75 to −85 dBm) could be several %. | Med | Med | Vitals and features are 1 Hz and loss-tolerant. For CSI, the 50 Hz cap already implies loss tolerance. Application-layer sequence numbers are already in place (`csi_collector.c:49`, `csi_collector.c:142`) so loss is observable downstream. |
| R7 | **Coordinator reboot no longer disturbs sensing, but loses all in-flight buffered frames.** No fix needed — this is an improvement over today — but it changes the observable pattern for the host. | Low | Low | Document in host-side decoder. See sibling ADR-071. |

---

## 9. Acceptance criteria

The migration is considered successful when **all** of the following hold
on a 4-perimeter + 1-coordinator bench deployment:

1. **Off-device sustained rate ≥ 40 Hz** per perimeter node for CSI
   (current: ~20 Hz effective, `README.md:15`/`:113`; theoretical cap
   today: 50 Hz per `CSI_MIN_SEND_INTERVAL_US`). Target = 40 Hz gives
   us a 2× improvement with ≥20% headroom.
2. **End-to-end latency p95 ≤ 5 ms** measured from `esp_timer_get_time()`
   at `wifi_csi_callback` entry (`csi_collector.c:164`) to the first
   byte appearing on the host USB serial (current estimate: ~15-20 ms).
3. **Memory budget**: ESP-NOW state must fit in **< 4 KB additional
   heap** per node (measured via `heap_caps_get_free_size(MALLOC_CAP_8BIT)`
   delta pre- and post-`esp_now_sender_init`).
4. **WiFi channel occupancy**: total airtime on the primary CSI channel
   (Tx + broadcast retries) must decrease by ≥15% vs the UDP baseline,
   measured with a sniffer.
5. **Link-loss recovery**: coordinator is rebooted; perimeter nodes
   continue sending ESP-NOW broadcasts; the host-side decoder is
   back-in-sync within **≤ 1.5 s** of coordinator re-boot (vs today’s
   ~5-10 s for SoftAP re-association).
6. **Zero increase in ENOMEM events** on perimeter nodes (grep
   `stream_sender.c:116` log) during a 1-hour soak.

---

## 10. Rejected alternatives

### 10.1 Stay on UDP but batch multiple sensor records per datagram

- **Idea**: pack N vitals or CSI frames into one 1-KB UDP packet to amortize
  IP+UDP overhead.
- **Why rejected**: (a) the coordinator’s `MAX_PKT_SIZE=512`
  (`main.c:31`) would need to be raised and every oversize check audited;
  (b) lwIP pbuf pressure is roughly per-packet, not per-byte, so
  batching would help, but lwIP ARP resolve and assoc recovery times
  remain the pathological failure modes; (c) batching increases
  worst-case latency (wait for batch fill) which is the opposite of
  what SAR needs; (d) none of this removes the STA-assoc startup
  dependency.

### 10.2 Use BLE Mesh instead of ESP-NOW

- **Idea**: switch peer transport to BLE Mesh (also supported on
  ESP32-S3).
- **Why rejected**: (a) BLE Mesh throughput is ~1 KB/s per node, an
  order of magnitude below what the current ~20 Hz CSI stream
  (~4 KB/s per node, raw I/Q burst up to 20 KB/s) needs; (b) the
  *entire point* of this system is CSI derived from 802.11 traffic —
  moving sensor traffic to BLE doesn’t generate the 802.11 frames that
  trigger CSI in the first place, so we’d need ESP-NOW (or something
  like it) anyway to keep the sensing substrate working; (c) BLE Mesh
  adds a second radio stack to maintain for zero sensing benefit.

### 10.3 (Considered, also rejected) 802.11 raw 80211_tx action frames

- **Idea**: use `esp_wifi_80211_tx()` (already used by the NDP stub,
  `csi_collector.c:444-480`) directly.
- **Why rejected**: no encryption, no peer management, no send-callback,
  no PMK — we’d be re-implementing ESP-NOW poorly. The NDP stub exists
  purely to inject a 24-B probe; it is not a data transport.

---

## 11. References

### ESP-IDF documentation
- ESP-NOW API reference (Espressif docs, v5.x):
  https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/network/esp_now.html
- ESP-NOW vendor-specific action frame format, 250-B payload limit, peer
  table size, coexistence with Wi-Fi AP/STA modes: same source.

### Sibling ADRs in this v2.1 transport series
- **Sibling ADR-071** — WiFi-to-host bridge (replaces the USB-Serial/JTAG
  + COBS leg in `main.c:47-89`, `main.c:164-181`).
- **Sibling ADR-072** — MIMO / time-sync (owns the TDM schedule in
  `nvs_config.h:32-37`, the channel-hop table in
  `csi_collector.c:64-79`, and likely the `SAR_AMP_MAGIC` wire type
  flagged as assumption in §6.4).

### Prior ADRs referenced in the codebase (by grep)
- ADR-018 CSI binary frame format — `csi_collector.c:3`,
  `csi_collector.h`
- ADR-029 channel hopping + TDM — `csi_collector.c:8-12`,
  `nvs_config.c:43-44`
- ADR-039 edge processing / vitals packets — `edge_processing.h:28-30`,
  `nvs_config.c:54-81`
- ADR-060 per-node MAC filter + channel override — `csi_collector.c:27`,
  `nvs_config.c:99-102`
- ADR-063 fused vitals 48-B packet — `edge_processing.h:127-154`
- ADR-069 feature-vector 48-B packet — `edge_processing.h:113-125`

### Grounding file:line citations used above

- `firmware/coordinator/main/main.c:19-22` — SoftAP SSID / channel
- `firmware/coordinator/main/main.c:23` — UDP port 4210
- `firmware/coordinator/main/main.c:31` — `MAX_PKT_SIZE = 512`
- `firmware/coordinator/main/main.c:47-89` — COBS + USB-Serial/JTAG FIFO
- `firmware/coordinator/main/main.c:98-114` — SoftAP init
- `firmware/coordinator/main/main.c:142-182` — UDP bridge task
- `firmware/perimeter/main/stream_sender.c:32-33` — ENOMEM cooldown
- `firmware/perimeter/main/stream_sender.c:36-57` — UDP socket init
- `firmware/perimeter/main/stream_sender.c:80-126` — send path with backoff
- `firmware/perimeter/main/csi_collector.c:27` — NVS config hook
- `firmware/perimeter/main/csi_collector.c:31-34` — peer MAC whitelist storage
- `firmware/perimeter/main/csi_collector.c:61` — `CSI_MIN_SEND_INTERVAL_US = 20 ms`
- `firmware/perimeter/main/csi_collector.c:96-159` — ADR-018 serializer
- `firmware/perimeter/main/csi_collector.c:164-244` — CSI callback
- `firmware/perimeter/main/csi_collector.c:222-236` — rate-limited UDP send
- `firmware/perimeter/main/csi_collector.c:304-312` — promiscuous mode setup
- `firmware/perimeter/main/csi_collector.c:444-480` — NDP inject stub
- `firmware/perimeter/main/csi_collector.h:15-21` — magic + CSI_MAX_FRAME_SIZE
- `firmware/perimeter/main/heartbeat.c:17-22` — 1-byte heartbeat
- `firmware/perimeter/main/link_reporter.c:33-42` — 10-B link report struct
- `firmware/perimeter/main/link_reporter.c:62-100` — record path
- `firmware/perimeter/main/link_reporter.c:127-130` — sendto for reports
- `firmware/perimeter/main/nvs_config.h:54-57` — `peer_count`, `peer_macs[4][6]`
- `firmware/perimeter/main/nvs_config.c:310-342` — NVS peer MAC load
- `firmware/perimeter/main/edge_processing.h:28-30` — vitals/compressed/I-Q magics
- `firmware/perimeter/main/edge_processing.h:34` — `EDGE_MAX_IQ_BYTES = 1024`
- `firmware/perimeter/main/edge_processing.h:96-111` — 32-B vitals struct
- `firmware/perimeter/main/edge_processing.h:113-125` — 48-B feature struct
- `firmware/perimeter/main/edge_processing.h:127-154` — 48-B fused-vitals struct
- `firmware/perimeter/main/edge_processing.c:718-742` — I/Q packet sender
- `firmware/perimeter/main/Kconfig.projbuild:10-21` — current UDP target config
- `firmware/perimeter/main/CMakeLists.txt:1-8` — current source list
- `firmware/perimeter/README.md:15,113` — observed ~20 Hz off-device rate
