# ADR-071: Replace USB-Serial/JTAG Bridge with WiFi-to-Host Forwarding

## 1. Status

**Proposed** — design-only, no implementation in this changeset.

- **Author:** Claude Opus 4.7
- **Date:** 2026-04-16
- **Supersedes (partially):** the coordinator bridge pathway described implicitly by the USB-Serial/JTAG code path in `firmware/coordinator/main/main.c`.
- **Siblings:** ADR-070 (ESP-NOW peering), ADR-072 (MIMO / time-sync).

---

## 2. Context

The GlassHouse v2 mesh is a 4-node (plus coordinator) WiFi-CSI search-and-rescue sensor array. Perimeter nodes detect presence / vitals / motion and forward packets over WiFi UDP (port 4210) to a coordinator running a SoftAP (`CSI_NET_V2`, channel 1). The coordinator today is a **pure transparent bridge**: it receives the UDP datagram, COBS-encodes it, and ships it over USB-Serial/JTAG to a tethered laptop.

Current bridge (see `firmware/coordinator/main/main.c:135-182` — `udp_bridge_task`):

```
recvfrom(sock, rx_buf, ...) -> cobs_encode(rx_buf, ...) -> usb_jtag_write_raw(...)
```

Host side is `glasshouse-capture/python/serial_receiver.py:44-63` — `SerialReceiver.read_packets()` — which reads the serial stream, splits on the `0x00` COBS delimiter, and yields decoded packets. `glasshouse-capture/python/frame_decoder.py:31` — `parse_packet()` — does the wire-format parsing and is downstream of whatever transport delivers the bytes.

### Throughput ceiling

- USB-Serial/JTAG is configured at 921600 baud (`main.c:30`). Theoretical line rate: ~92 KB/s (with 8-N-1 framing, ~92,160 bytes/sec).
- **Observed effective rate: ~10 packets/sec** (reported by field tests through `debug/capture.py`).
- Assuming an average payload around 200 B (mix of vitals=32 B, feature=48 B, fused=48 B, CSI up to ~300 B, plus COBS overhead of ~0.4%), 10 pkt/s corresponds to **~2 KB/s — roughly 2% of line rate**. The bottleneck is not raw bandwidth; it is the per-packet cost of the `usb_serial_jtag_ll_write_txfifo` + `usb_serial_jtag_ll_txfifo_flush` loop in `main.c:47-61` (5 retry attempts with `vTaskDelay(1)`, yielding ~1–2 ms per packet *if* the host hasn't stalled; much worse when the host buffer fills).
- Additionally, `main.c:47-61` silently drops packets after 5 retries to avoid blocking the receive loop — we therefore lose frames without any signal upstream.

### Deployment constraint

- In the SAR concept of operations, the coordinator is "tossed" at a rubble pile / structure perimeter by a responder who then operates a laptop from a safe standoff (tens of meters).
- A 10 m USB tether is impractical and a trip hazard on unstable ground. USB-to-fiber extenders exist but add cost, weight, and another failure point.
- ESP32-S3 supports `WIFI_MODE_APSTA` (concurrent SoftAP + Station). The coordinator can continue to host the sensing SoftAP (for perimeter node traffic) while *also* associating as a STA client to an operator WiFi (hotspot on the responder's laptop, or a deployed field AP) and forwarding frames over UDP to the host.

---

## 3. Problem

Three distinct problems motivate this ADR:

### 3a. Per-packet USB/COBS overhead dominates at small packet sizes

The `udp_bridge_task` in `main.c:164-181` executes, per packet:
1. `recvfrom` (userspace)
2. `cobs_encode` (O(n), in-place)
3. `usb_jtag_write_raw` with up to 5 retries and a `vTaskDelay(1)` between each (`main.c:51-59`)
4. `usb_serial_jtag_ll_txfifo_flush` per iteration

For small frames (vitals=32 B, link=10 B, heartbeat=1 B from `frame_decoder.py:48`), the flush/retry fixed cost swamps the byte-transfer cost. Measured ~10 pkt/s ceiling matches a ~100 ms/packet effective budget, which is two orders of magnitude worse than the line-rate prediction.

### 3b. USB tether incompatible with "toss-and-run" SAR deployment

The coordinator is intended to be positioned physically near the incident scene — on top of rubble, through a window, on a remote tripod. A USB-A cable of useful length (>5 m) is at the edge of the spec for USB 2.0 full-speed without an active repeater, and every meter of cable in an SAR environment is one more snag point. The current architecture **forces a tether where none should exist**.

### 3c. Single USB cable is a single point of failure

A pulled, stepped-on, or dust-fouled USB connector on the coordinator kills the entire data path. WiFi STA association fails gracefully and retries; USB does not. In the current design there is no redundancy — one cable is the bridge.

---

## 4. Decision

**Replace the USB-CDC bridge with a WiFi forwarding mode.** The coordinator continues to run the sensing SoftAP (`CSI_NET_V2`, channel 1). Additionally, it joins an operator-WiFi network as a **STA** and forwards every incoming perimeter frame as a UDP datagram to a configured host listener. The host-side `SerialReceiver` is replaced by a `UDPReceiver` that preserves the `read_packets()` generator contract so that `frame_decoder.py` and `debug/capture.py` remain unchanged.

The USB bridge path is **not removed immediately**; it is retained as a build-time fallback for bench development where no operator WiFi exists (see Migration Phases).

---

## 5. Architecture Diagram

### Before (current, ffc103c)

```
  Perimeter x4 ──UDP/WiFi──▶  Coordinator (SoftAP)
                                  │
                                  ▼
                         cobs_encode + 0x00
                                  │
                                  ▼
                         usb_serial_jtag_ll_write_txfifo
                                  │ USB 921600
                                  ▼
                    Laptop  /dev/ttyUSB0 or COMx
                                  │
                                  ▼
                      SerialReceiver.read_packets()
                                  │
                                  ▼
                        frame_decoder.parse_packet()
```

### After (proposed)

```
  Perimeter x4 ──UDP/WiFi (CSI_NET_V2, ch 1)──▶  Coordinator
                                                  ├── AP iface (ch 1)
                                                  └── STA iface (ch N, operator WiFi)
                                                          │
                                                          │ sendto(host_ip, UDP/4211)
                                                          ▼
                                          Operator WiFi (hotspot or field AP)
                                                          │
                                                          ▼
                                                 Laptop (no cable)
                                                          │
                                                          ▼
                                          UDPReceiver.read_packets()
                                                          │
                                                          ▼
                                          frame_decoder.parse_packet()   ◀── unchanged
```

---

## 6. Interface Design

### 6.1 Coordinator side

**Dual-interface concurrent operation.** ESP-IDF supports `WIFI_MODE_APSTA` (see `esp_wifi_set_mode(WIFI_MODE_APSTA)` — this is a documented ESP-IDF capability; currently `main.c:109` uses `WIFI_MODE_AP`). Both interfaces share a single radio, and as a consequence **both must operate on the same channel** — this is called out in the Risk Register (§10).

The existing `wifi_init_softap` in `main.c:92-114` becomes `wifi_init_apsta`, which additionally calls `esp_wifi_set_config(WIFI_IF_STA, &sta_cfg)` with an operator SSID/PSK stored in NVS. NVS keys: `opwifi.ssid`, `opwifi.psk`, `opwifi.host_ip`, `opwifi.host_port`. A serial CLI (over UART1, which is already the log channel per `main.c:27-29`) provisions these before field deployment.

**What replaces `usb_jtag_write_raw`:** a UDP send on the STA interface:

```c
// Pseudocode replacement for main.c:47-61 + main.c:171
sendto(fwd_sock, rx_buf, len, 0, (struct sockaddr *)&host_addr, sizeof(host_addr));
```

No COBS wrapping — UDP datagrams are already framed (see §7).

### 6.2 Host discovery

Three options, in decreasing order of field robustness:

| Option | Pros | Cons |
|---|---|---|
| **Static IP** (coord stores `host_ip` in NVS; laptop configured with a fixed IP on operator WiFi) | Deterministic. No DNS/multicast. Works on hostile networks. | Requires per-deployment provisioning. |
| **mDNS / DNS-SD** (coord queries `_glasshouse._udp.local`, laptop advertises it) | Zero-config. Survives DHCP renewals. | mDNS is flaky over some APs (especially phone hotspots on iOS/Android), and ESP-IDF `mdns` component adds ~15 KB flash. |
| **DHCP option magic** (laptop runs a DHCP server offering a vendor-specific option pointing at itself) | Coord auto-discovers. | Only works when the laptop is also the DHCP server (i.e. laptop-hosted hotspot), not on a field AP. |

**Recommended:** static IP as the baseline (it always works), mDNS as a convenience layer on top. The coord tries mDNS first, falls back to the NVS-stored static IP if discovery fails within ~3 s.

### 6.3 Host side

Replace `glasshouse-capture/python/serial_receiver.py` behaviourally with a new `glasshouse-capture/python/udp_receiver.py`:

```python
# Not implementation — this is the interface contract for §11 acceptance.
class UDPReceiver:
    def __init__(self, bind_host: str = "0.0.0.0", bind_port: int = 4211) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def read_packets(self) -> Generator[bytes, None, None]: ...  # SAME signature
```

Because `SerialReceiver.read_packets()` at `serial_receiver.py:44` yields `bytes` (COBS-decoded UDP payloads), and `UDPReceiver.read_packets()` would yield `bytes` (raw UDP payloads, which are the **same** thing — the original perimeter payload), the downstream pipeline in `debug/capture.py:46` (`for packet in receiver.read_packets(): ... parse_packet(packet)`) is unchanged.

`capture.py` would gain one flag: `--transport {serial,udp}` (default `udp` post-migration). Total host-side delta is expected to be **under 100 LOC** (one new file + one argparse flag + a factory function).

### 6.4 Fallback to USB bridge

A compile-time flag `CONFIG_GH_BRIDGE_MODE` (menuconfig) selects:
- `GH_BRIDGE_USB` — current `usb_jtag_write_raw` path (preserved verbatim from `main.c:47-61`).
- `GH_BRIDGE_WIFI` — new `sendto` path.
- `GH_BRIDGE_BOTH` — dual-output for migration-phase testing; write to USB FIFO *and* UDP. Useful to A/B-test packet loss.

Bench development keeps `GH_BRIDGE_USB` as the default for now. CI / release builds flip to `GH_BRIDGE_WIFI` at Phase 2.

---

## 7. MTU / Fragmentation

WiFi UDP MTU is 1500 B (less IP+UDP headers → 1472 B payload). GlassHouse frames are small:

| Frame type | Size | Source |
|---|---|---|
| heartbeat | 1 B | `frame_decoder.py:48` |
| link | 10 B | `frame_decoder.py:163` |
| vitals | 32 B | `frame_decoder.py:80` (`VITALS_PKT_SIZE`) |
| feature | 48 B | `frame_decoder.py:97` |
| fused | 48 B | `frame_decoder.py:112` |
| CSI | ~20–300 B typical, up to `MAX_PKT_SIZE=512` | `main.c:31`, `frame_decoder.py:56` |
| I/Q | variable, capped at 512 B | `main.c:31` |

Every frame fits in a single UDP datagram (hard cap = `MAX_PKT_SIZE=512` B from `main.c:31`). **No fragmentation required.**

**Should COBS be retained?** No. COBS exists (`main.c:66-89`) solely to delimit frames in a byte stream where `0x00` cannot appear mid-frame. UDP is already a datagram protocol: each `recvfrom` call returns exactly one sender-side `sendto` payload. Keeping COBS in the UDP path would be pure overhead. **Drop COBS in `GH_BRIDGE_WIFI` mode.**

This means `SerialReceiver`'s `decode_cobs` (`serial_receiver.py:11-13`) is also dropped in the UDP path — `UDPReceiver.read_packets()` yields raw payload bytes directly to `parse_packet`.

---

## 8. Security Considerations

### Threat model (field SAR)

The operator WiFi network will be one of:
- (A) Laptop-hosted hotspot (ad-hoc, single responder) — most common.
- (B) Deployed field AP with WPA2-PSK — multi-responder ops.
- (C) Open public WiFi — hopefully never, but possible in disaster zones.

### Threats

| Threat | Description | Mitigation in this ADR |
|---|---|---|
| **Eavesdropping** | Sensor data (presence, vitals) is arguably sensitive — a hostile actor sniffing WiFi could detect survivor locations. | WPA2-PSK on operator WiFi is the baseline. For production, frame payloads should be AES-GCM'd with a per-deployment key. **Out of scope** for this ADR (feasibility first). |
| **Coordinator MAC spoofing** | An attacker on the operator WiFi could impersonate the coord's IP/MAC and inject fake frames to the host listener. | Bind host listener to the coord's expected IP + simple per-frame sequence counter + optional HMAC (AuthN). **Out of scope** for this ADR; flagged for production. |
| **Replay** | An attacker records coord→host UDP traffic and replays it to fake old telemetry. | Per-frame monotonic sequence (already present in most frame types — see `frame_decoder.py:59` for CSI `seq`, `:99` for feature `seq`) + timestamp-based rejection on the host. **Out of scope** for this ADR. |
| **Denial of service** | Flood the coord's STA interface with traffic and starve the SoftAP. | WiFi firmware is priority-per-interface; quantify impact in Phase 1 testing. |
| **Operator hotspot is open** | Responder laptop broadcasts SSID in the clear to avoid provisioning friction. | Explicitly require WPA2-PSK in the deployment SOP; ship the firmware refusing to join open networks unless `CONFIG_GH_ALLOW_OPEN_WIFI=y`. |

This ADR does **not** mandate payload-layer encryption. It is a feasibility decision. A follow-on ADR should cover AuthN/AuthE (candidate: ADR-073).

---

## 9. Migration Phases

### Phase 1 — Dual transport (USB + WiFi)
- Build the firmware with `GH_BRIDGE_BOTH` and ship to one dev kit.
- Every frame goes out both USB and UDP.
- Host runs both receivers and diffs the streams.
- **Acceptance:** ≥ 99% of frames arrive on both transports in a 10-minute static test; UDP matches USB byte-for-byte after COBS decode; **UDP-path packet rate ≥ 5× USB-path packet rate** on the same hardware (expected: the USB ceiling is the bottleneck, not the perimeter).

### Phase 2 — WiFi default, USB fallback
- `GH_BRIDGE_WIFI` default; USB path compiled in but inactive at runtime.
- `python/udp_receiver.py` and a `--transport udp` flag shipped to `debug/capture.py`.
- **Acceptance:** One field trial (~20 m standoff, operator hotspot) sustains ≥ 50 pkt/s aggregate from 4 perimeter nodes for 10 minutes with < 2% loss; USB fallback compiles and runs on a bench build.

### Phase 3 — USB path removed
- `main.c` no longer imports `hal/usb_serial_jtag_ll.h`; `cobs_encode` and `usb_jtag_write_raw` deleted from the coordinator (they may survive on perimeter firmware for debug consoles, separately).
- `serial_receiver.py` deprecated with a clear upgrade note; eventually removed one minor version later.
- **Acceptance:** Two successful end-to-end deployments (simulated SAR drill) with WiFi-only transport; no regressions in `parse_packet` output.

---

## 10. Risk Register

| # | Risk | Likelihood | Severity | Notes |
|---|---|---|---|---|
| R1 | **AP + STA forced to same channel.** ESP32-S3 has a single radio; `WIFI_MODE_APSTA` must use one channel for both. If the operator WiFi is on channel 6 or 11, the sensing SoftAP is forced off ch 1 — possibly disrupting perimeter association and channel-dependent CSI calibration. | **High** | **High** | Mitigation options: (a) dynamically adopt the operator channel at boot, re-announce SoftAP on that channel (perimeters must rescan); (b) require operators to configure the field AP on ch 1; (c) accept the channel change and add a re-calibration step to ADR-072. Prefer (a) with (b) as SOP fallback. |
| R2 | **Operator WiFi congestion drops frames.** Consumer phone hotspots buffer/prioritize unpredictably; a 50–100 pkt/s UDP stream can be silently rate-limited. | Medium | Medium | Per-frame sequence counter already exists in most types; host-side loss monitor flags drops > 2%. If chronic, switch operator transport to a dedicated field AP (bigger hammer) or add an application-layer retransmit for link frames only. |
| R3 | **Power draw increase.** AP+STA concurrent draws more than AP-only; WiFi TX on two associations is ~30–40% more average current (empirical ESP32-S3 range; to be measured). Coord on battery will have shorter runtime. | Medium | Low–Medium | Measure in Phase 1. If impact > 25%, add a `power_save` mode for the STA interface (`WIFI_PS_MIN_MODEM`) which the ESP-IDF supports on the STA iface without affecting AP. |
| R4 | **Latency jitter on operator WiFi.** USB latency is sub-ms and monotone; WiFi latency is 2–50 ms typical with occasional 200+ ms spikes on busy networks. Some vitals-timing features assume low latency. | Medium | Low | Vitals end-to-end is already multi-hop WiFi perimeter→coord (see ADR-072 for time-sync). Adding one more hop is small in absolute terms. Flag any vitals-pipeline ADR that assumes < 5 ms host-latency for review. |
| R5 | **Discovery / IP-assignment reliability.** DHCP from a field AP might never lease an IP; mDNS might be blocked; static IP requires provisioning. In a time-pressured SAR deployment, "laptop can't see coord" is a catastrophic UX failure. | Medium | **High** | Triple-layer fallback (mDNS → static IP from NVS → LED status indicating WiFi-joined-but-host-unreachable). A physical button on the coord triggers a "broadcast heartbeat to 255.255.255.255:4211" mode so any listener on the same subnet can find it. |
| R6 | **COBS-removal regression.** Any host-side code path that assumes framing (possibly untested third-party consumers) will break. | Low | Low | Kept COBS retained in USB fallback; `udp_receiver.py` and `serial_receiver.py` coexist during migration. |

---

## 11. Acceptance Criteria

The design is considered successful when:

1. **Sustained throughput:** Coordinator forwards ≥ 50 pkt/s aggregate for 10 min continuous with < 2% packet loss on the operator WiFi (this is a 5× improvement over the current ~10 pkt/s USB ceiling). Byte rate target: ≥ 20 KB/s.
2. **End-to-end latency (perimeter recvfrom → host `read_packets` yield):** median ≤ 15 ms, p99 ≤ 100 ms on a healthy operator network.
3. **Host-side code delta:** ≤ 100 LOC net new (one new `udp_receiver.py` + a `--transport` flag in `capture.py` + a factory). No changes to `frame_decoder.py`.
4. **Decoder preserved:** All frame types in `frame_decoder.py:31-183` parse identically when fed from `UDPReceiver.read_packets()` vs `SerialReceiver.read_packets()` (diff of parsed-record streams on the same input is empty modulo timing fields).
5. **USB fallback remains buildable:** `GH_BRIDGE_USB` compiles cleanly and flashes; one end-to-end test per release cycle exercises it.
6. **Range:** Coord→laptop line-of-sight range ≥ 15 m on a typical 2.4 GHz operator hotspot (sanity check; 802.11n datasheets comfortably support this, but confirm with field measurement).

---

## 12. Rejected Alternatives

### 12a. Ethernet (PoE)
Gigabit Ethernet would cleanly solve the throughput and reliability problems. **Rejected** because it reintroduces cabling, which the SAR use-case explicitly excludes (§3b). ESP32-S3 also has no on-chip Ethernet MAC without an external PHY (W5500 or LAN8720), increasing BOM cost and board complexity. Defensible for a fixed-site variant; not for the field deployment this ADR targets.

### 12b. Cellular / LTE Modem (e.g. SIM7600)
A cellular modem would let the coord reach a cloud endpoint independent of responder infrastructure. **Rejected** because (a) BOM cost (~$40 module + SIM), (b) average power draw (~300–600 mW idle vs. ~150 mW for WiFi STA), (c) disaster-zone cell coverage is unreliable exactly when SAR happens, (d) latency is worse (cellular RTT typically 50–150 ms), (e) subscription/SIM management overhead per deployment. Defensible for a "persistent monitoring" variant shipping telemetry back to HQ; out of scope here.

### 12c. Direct LoRa-to-host
LoRa would give multi-km range with no infrastructure. **Rejected** because of the bandwidth ceiling: LoRa SF7-BW500 tops out at ~11 kbps, and the perimeter mesh produces tens of KB/s of CSI + I/Q. The useful SAR signal (vitals+link+fused at reduced rate) fits in LoRa, but losing raw CSI would be a regression against the full architecture. Defensible as a supplementary "summary uplink" (see potential future ADR), not as a replacement for the bulk-data bridge.

---

## 13. References

### Grounding files (this worktree, ffc103c baseline)
- `firmware/coordinator/main/main.c:47-61` — `usb_jtag_write_raw`, the direct USB FIFO write with retry.
- `firmware/coordinator/main/main.c:66-89` — `cobs_encode`, the in-firmware COBS encoder.
- `firmware/coordinator/main/main.c:92-114` — `wifi_init_softap`, the current AP-only init that this ADR converts to `APSTA`.
- `firmware/coordinator/main/main.c:135-182` — `udp_bridge_task`, the end-to-end bridge loop with its observed ~10 pkt/s ceiling.
- `firmware/coordinator/main/main.c:34-42` — `uart1_vprintf`, the log redirect that keeps USB-CDC clean for COBS; unchanged by this ADR.
- `glasshouse-capture/python/serial_receiver.py:44-63` — `SerialReceiver.read_packets()`, the generator contract to preserve.
- `glasshouse-capture/python/frame_decoder.py:31-183` — `parse_packet`, unchanged under this ADR.
- `glasshouse-capture/debug/capture.py:22-78` — `capture()`, the top-level capture entrypoint that gains a `--transport` flag.

### Sibling ADRs
- **ADR-070** (ESP-NOW peering) — may alter the perimeter→coord hop, but not the coord→host hop this ADR concerns. Orthogonal; no conflict expected.
- **ADR-072** (MIMO / time-sync) — time-sync assumptions may tighten the latency budget for the coord→host hop; R4 above is the relevant coupling point.

### ESP-IDF documentation
- `esp_wifi_set_mode(WIFI_MODE_APSTA)` — ESP-IDF WiFi Driver API, "AP+STA Coexistence" section. Confirms that AP and STA must share a channel and that the STA association dictates the channel when both are active.
