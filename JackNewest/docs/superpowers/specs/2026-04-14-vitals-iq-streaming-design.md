# Vitals Timer Fix + I/Q Streaming Design

**Date:** 2026-04-14
**Status:** Approved
**Demo deadline:** 2026-04-24

## Problem

GlassHouse v2 receives ~3 vitals packets per 30-second session instead of the expected ~30 (1 Hz default). Root cause: `send_vitals_packet()` is called inside `process_frame()`, which only executes when a CSI frame is popped from the ring buffer by the edge DSP task. Two failure modes:

1. **Tier 0**: `edge_task` never starts, so vitals are never sent.
2. **Tier 1-2 with sparse CSI**: The timer gate inside `process_frame()` depends on frame arrival rate.

Additionally, compressed frames (266-1281 bytes) and feature vectors (48 bytes) go through `stream_sender`, triggering ENOMEM backoff that suppresses vitals delivery.

The firmware's on-device DSP (zero-crossing BPM, 12.8s window) has fundamental accuracy limits. Python-side FFT with 60s windows and cross-node fusion can achieve ~2-3x better BPM accuracy.

## Solution Overview

Three coordinated changes:

| Phase | Scope | LOC |
|-------|-------|-----|
| Phase A: Vitals Timer | Firmware (perimeter) | ~35 |
| Phase B: I/Q Streaming | Firmware (perimeter) | ~20 |
| Phase C: Python DSP | Python | ~187 |

## Phase A: Independent Vitals Timer

### Changes to `firmware/perimeter/main/edge_processing.c`

**Add an `esp_timer`** in `edge_processing_init()` that fires every `vital_interval_ms` (default 1000ms). Place timer creation **before** the tier-0 early return so vitals are sent at all tiers.

```
edge_processing_init()
    ...existing state reset (lines ~995-1042)...
    NEW: esp_timer_create(vitals_timer_cb, vital_interval_ms)
    NEW: esp_timer_start_periodic(...)
    if (tier == 0) return ESP_OK;   // timer still runs
    ...create edge_task on Core 1...
```

**Timer callback** (`vitals_timer_cb`):
1. Enter critical section (spinlock)
2. Snapshot vitals state: `s_motion_energy`, `s_presence_detected`, `s_breathing_bpm`, `s_heartrate_bpm`, `s_fall_detected`, `s_latest_rssi`, `s_persons[]`
3. Exit critical section
4. Build and send vitals packet from snapshot
5. Set `s_pkt_valid = true` (for WASM dispatch compatibility)

**Remove from `process_frame()`**: Lines 853-871 (vitals send gate, feature vector send, `s_last_vitals_send_us` tracking). Remove `send_feature_vector()` call entirely.

### Changes to `firmware/perimeter/main/stream_sender.c`

**Add a FreeRTOS mutex** (`s_send_mutex`) to protect:
- `s_backoff_until_us` (64-bit, tearable on 32-bit bus)
- `s_enomem_suppressed` counter
- `sendto()` call serialization

```c
static SemaphoreHandle_t s_send_mutex;

// In stream_sender_init():
s_send_mutex = xSemaphoreCreateMutex();

// In stream_sender_send():
xSemaphoreTake(s_send_mutex, portMAX_DELAY);
// ...existing backoff check + sendto logic...
xSemaphoreGive(s_send_mutex);
```

### Thread Safety Analysis

| Variable | Written by | Read by timer | Protection |
|----------|-----------|---------------|------------|
| `s_motion_energy` (float) | `process_frame()` Core 1 | `vitals_timer_cb` Core 0 | portENTER_CRITICAL spinlock |
| `s_presence_detected` (bool) | `process_frame()` Core 1 | `vitals_timer_cb` Core 0 | portENTER_CRITICAL spinlock |
| `s_breathing_bpm` (float) | `process_frame()` Core 1 | `vitals_timer_cb` Core 0 | portENTER_CRITICAL spinlock |
| `s_heartrate_bpm` (float) | `process_frame()` Core 1 | `vitals_timer_cb` Core 0 | portENTER_CRITICAL spinlock |
| `s_persons[]` (struct array) | `process_frame()` Core 1 | `vitals_timer_cb` Core 0 | portENTER_CRITICAL spinlock |
| `s_pkt_valid` (volatile bool) | `vitals_timer_cb` Core 0 | `process_frame()` Core 1 | portENTER_CRITICAL spinlock |
| `s_backoff_until_us` (int64) | `stream_sender_send()` | `stream_sender_send()` | Mutex |

### Init Ordering Requirement

`stream_sender_init_with()` is called at `main.c:160`, before `edge_processing_init()` at `main.c:213`. The mutex is created inside `stream_sender_init_with()`, so it is guaranteed to exist before the vitals timer starts. **This ordering must not be changed.**

### WASM Dispatch (Step 14)

Step 14 (`process_frame()` lines 874-892) is **preserved**. It reads `s_pkt_valid`, which is now set by `vitals_timer_cb` under the same spinlock. On startup, `s_pkt_valid` remains false until the first timer fire (up to 1 second). WASM dispatch is silently skipped during this window. This is acceptable — WASM modules need several frames of data before producing meaningful output anyway.

### Pre-existing Note: Magic 0xC5110004 Collision

`EDGE_FUSED_MAGIC` (edge_processing.h) and `WASM_OUTPUT_MAGIC` (wasm_runtime.h) both use `0xC5110004`. This is a pre-existing issue not introduced by this spec. Python currently doesn't parse either packet type. If fused vitals or WASM output packets need parsing in the future, this collision must be resolved first.

### Tier 0 Behavior

Timer fires, but all DSP values remain at init (0.0, false). Vitals packets arrive with zeros. Python sees node is alive. This is correct.

## Phase B: Raw I/Q Streaming

### New Packet Type

**Magic:** `EDGE_IQ_MAGIC = 0xC5110006` (added to `edge_processing.h`)

```
I/Q Stream Packet (variable length, max 264 bytes)
Offset  Size  Field
0..3    4     Magic (0xC5110006, little-endian)
4       1     node_id
5       1     channel
6..7    2     iq_len (little-endian u16, actual I/Q byte count)
8..N    var   Raw I/Q data (iq_len bytes, max 256 for HT40)
```

Variable `iq_len` because ESP32 CSI returns different sizes: HT20=128B, HT40=256B, legacy=64B.

### Sending Logic

In `process_frame()`, replace `send_compressed_frame()` (tier >= 2, lines 849-851) with rate-limited I/Q streaming:

```c
#define IQ_STREAM_DIVIDER 2  // send every 2nd frame -> ~10 Hz at 20 Hz CSI

static uint8_t s_iq_divider = 0;
if (++s_iq_divider >= IQ_STREAM_DIVIDER) {
    s_iq_divider = 0;
    send_iq_packet(slot->iq_data, slot->iq_len, slot->channel);
}
```

`send_iq_packet()` builds the header and calls `stream_sender_send()` (mutex-protected).

### Removed

- `send_compressed_frame()` call — no longer invoked (function body can remain as dead code or be removed)
- `send_feature_vector()` call — already removed in Phase A

### Bandwidth

| Metric | Value |
|--------|-------|
| Packet size | 8 + iq_len (typ. 136B for HT20, max 264B for HT40) |
| Rate per node | ~10 Hz |
| 4 nodes total | ~8 KB/s |
| Serial capacity | ~88 KB/s |
| Utilization | ~9% |

### Rate Divider Note

The divider counts frames processed by `edge_task`. If CSI frames are sparse, effective output rate drops proportionally. This is acceptable — no I/Q data means nothing to send.

## Phase C: Python DSP Pipeline

### New File: `python/iq_processor.py`

```python
class IQProcessor:
    def __init__(self, sample_rate: float = 10.0, window_sec: float = 60.0)
    def feed(self, node_id: int, channel: int, iq_data: bytes) -> None
    def get_vitals(self) -> dict
```

### Phase Extraction

```python
def feed(self, node_id, channel, iq_data):
    # I/Q layout: interleaved [I0, Q0, I1, Q1, ...] as int8 pairs.
    # Confirmed by WASM dispatch in edge_processing.c lines 877-879:
    #   i_val = (int8_t)slot->iq_data[sc * 2];
    #   q_val = (int8_t)slot->iq_data[sc * 2 + 1];
    n_sub = len(iq_data) // 2
    i_vals = np.array([np.int8(b) for b in iq_data[0::2]], dtype=np.float32)
    q_vals = np.array([np.int8(b) for b in iq_data[1::2]], dtype=np.float32)
    phase = np.arctan2(q_vals, i_vals)
    self._buffers[node_id].append(phase)
```

### Phase Unwrapping (Critical)

Raw `atan2` wraps at +/-pi. Discontinuities corrupt FFT. Unwrapping is applied along the **time axis** before spectral analysis:

```python
phase_matrix = np.array(buffer)        # shape (N_frames, N_subcarriers)
unwrapped = np.unwrap(phase_matrix, axis=0)  # unwrap along time
```

### BPM Estimation

Uses `scipy.signal.welch()` (Welch PSD) for robustness against noise:

```python
def _estimate_bpm(self, phase_series, band_hz, fs):
    freqs, psd = scipy.signal.welch(phase_series, fs=fs, nperseg=256)
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    peak_idx = np.argmax(psd[mask])
    peak_freq = freqs[mask][peak_idx]
    peak_power = psd[mask][peak_idx]
    noise_floor = np.median(psd[mask])
    snr = peak_power / max(noise_floor, 1e-10)
    return peak_freq * 60.0, snr  # BPM, SNR
```

**Frequency bands:**
- Breathing: 0.1-0.5 Hz (6-30 BPM)
- Heart rate: 0.8-2.0 Hz (48-120 BPM)

**Resolution:** At 10 Hz sample rate, nperseg=256 gives ~0.039 Hz per bin (~2.3 BPM). Sufficient for detecting breathing presence. Display BPM as approximate ("~16 BPM").

### Cross-Node Fusion

Per-node BPM estimates weighted by SNR:

```python
def _fuse_estimates(self, estimates):
    # estimates: list of (bpm, snr) per node
    total_snr = sum(snr for _, snr in estimates if snr > 3.0)
    if total_snr == 0:
        return 0.0, 0.0
    fused = sum(bpm * snr for bpm, snr in estimates if snr > 3.0) / total_snr
    confidence = min(total_snr / 40.0, 1.0)
    return fused, confidence
```

SNR threshold of 3.0 filters out nodes with no clear spectral peak.

### Memory

600 frames x 96 subcarriers x 4 bytes x 4 nodes = ~900 KB. Fine on desktop.

### Integration

**`python/link_aggregator.py`** — new branch in `feed()`:

```python
_IQ_MAGIC = b'\x06\x00\x11\xC5'  # 0xC5110006 little-endian

elif len(packet) >= 8 and packet[:4] == self._IQ_MAGIC:
    self._parse_iq(packet)
```

`_parse_iq()` extracts node_id, channel, iq_len, and raw bytes from the header, then stores them in a public accessor (`latest_iq` property or similar). `main.py` reads this after each packet batch and passes to `IQProcessor.feed()`.

**`python/main.py`** — instantiate `IQProcessor`, wire the data flow:
1. After each packet batch from `link_aggregator`, check for new I/Q data
2. Pass raw I/Q bytes to `iq_processor.feed(node_id, channel, iq_data)`
3. Every ~1s, call `iq_processor.get_vitals()` and display BPM in debug output

### Dependencies

Add to `requirements.txt` (at repo root — create if it doesn't exist, or append to existing):
```
numpy>=1.24
scipy>=1.10
```

### Main Loop Blocking

`get_vitals()` runs Welch PSD on 600x96 matrix. Expected time: 5-20ms. The OS serial buffer (4-16 KB) absorbs incoming packets during this window. At ~8 KB/s I/Q throughput, 20ms of blocking accumulates ~160 bytes — well within buffer capacity.

## Testing Strategy

### Firmware (Phase A+B)

- Verify vitals packets arrive at 1 Hz with a capture tool
- Verify I/Q packets arrive at ~10 Hz per active node
- Verify no ENOMEM backoff with compressed frames removed
- Count packets per node over 30s — expect ~30 vitals + ~300 I/Q

### Python (Phase C)

- Unit test: feed synthetic I/Q with known breathing frequency, verify BPM output
- Unit test: phase unwrapping handles +/-pi boundaries correctly
- Integration test: feed capture data, verify IQProcessor doesn't crash
- Hardware test: live session, verify BPM displayed in debug output

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Firmware flash fails | Phase A is independent — can ship vitals timer alone |
| I/Q packets don't arrive | Fall back to firmware vitals (now reliable with timer fix) |
| FFT BPM inaccurate | Display as "~N BPM" with confidence score |
| scipy not installable | Degrade gracefully — IQProcessor returns empty dict |
| ENOMEM at 10 Hz I/Q | Reduce to 5 Hz via IQ_STREAM_DIVIDER=4 |

## Timeline

| Days | Task |
|------|------|
| 1-2 | Phase A: Vitals timer + stream_sender mutex |
| 3-4 | Phase B: I/Q packet type + firmware flash |
| 5-7 | Phase C: IQProcessor + integration |
| 8-9 | Hardware testing + tuning |
| 10 | Demo rehearsal (April 24) |
