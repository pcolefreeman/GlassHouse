/**
 * @file sar_amp_sender.h
 * @brief Batched per-peer CSI mean-amplitude stream (magic 0xC5110007).
 *
 * Off-device CSI delivery needs the mean amplitude per (self, peer) pair to
 * be reliable and high-rate. Sending one UDP datagram per CSI callback
 * (~50 Hz * N peers) wastes payload overhead and stresses lwIP pbufs.
 *
 * This module batches 48 amplitude samples per peer into a single
 * fixed-size 208-byte packet that is flushed either:
 *   1. When the per-peer ring reaches SAR_AMP_BATCH_SIZE samples.
 *   2. When more than 1 second has elapsed since the oldest unsent sample.
 *
 * Wire format (little-endian, packed):
 *
 *     uint32  magic           = SAR_AMP_MAGIC (0xC5110007)   // offset 0
 *     uint8   node_id         = this device's node ID         // offset 4
 *     uint8   peer_id         = the peer whose link this is   // offset 5
 *     uint16  n_samples       = SAR_AMP_BATCH_SIZE (48)       // offset 6
 *     uint32  batch_start_us  = timestamp of amps[0]          // offset 8
 *     uint32  interval_us     = avg interval between samples  // offset 12
 *     float   amps[48]                                        // offset 16..207
 *   = 208 bytes total.
 *
 * The push API is designed to be called from the WiFi CSI callback context
 * (see csi_collector.c), so the critical section is minimal and the flush
 * happens lazily.
 */

#ifndef SAR_AMP_SENDER_H
#define SAR_AMP_SENDER_H

#include <stdint.h>
#include "edge_processing.h"  /* SAR_AMP_MAGIC */

/** Max simultaneous peers (matches MAX_PEER_MACS in csi_collector.c). */
#define SAR_MAX_PEERS      4

/** Samples per flush. 48 floats = 192 bytes of payload data per peer. */
#define SAR_AMP_BATCH_SIZE 48

/** Wire packet — exactly 208 bytes, little-endian, packed. */
typedef struct __attribute__((packed)) {
    uint32_t magic;            /**< SAR_AMP_MAGIC = 0xC5110007. */
    uint8_t  node_id;          /**< Transmitting node's ID. */
    uint8_t  peer_id;          /**< Peer node ID this batch corresponds to. */
    uint16_t n_samples;        /**< Valid samples in amps[] (<= SAR_AMP_BATCH_SIZE). */
    uint32_t batch_start_us;   /**< esp_timer_get_time() of amps[0], low 32 bits. */
    uint32_t interval_us;      /**< (batch_end - batch_start) / (n-1), or 0. */
    float    amps[SAR_AMP_BATCH_SIZE]; /**< Mean amplitudes, one per CSI callback. */
} sar_amp_pkt_t;

_Static_assert(sizeof(sar_amp_pkt_t) == 208, "sar_amp_pkt_t must be 208 bytes");

/**
 * Push one mean-amplitude sample for a peer into its ring.
 *
 * Thread-safety: safe to call from WiFi CSI callback context. Uses a
 * minimal critical section to protect the per-peer ring state.
 *
 * @param peer_id   Peer's node ID (1..SAR_MAX_PEERS). Samples for peer_id
 *                  outside [1, SAR_MAX_PEERS] are silently dropped.
 * @param mean_amp  Mean CSI amplitude across subcarriers for this frame.
 * @param ts_us     esp_timer_get_time() at the moment of capture.
 */
void sar_amp_push(uint8_t peer_id, float mean_amp, int64_t ts_us);

/**
 * Best-effort flush of any peer whose buffer has aged past the 1 s
 * timeout. Exposed for tests / tools — the normal flow flushes
 * lazily inside sar_amp_push().
 */
void sar_amp_check_timeouts(void);

/**
 * Start a periodic esp_timer that calls sar_amp_check_timeouts() every
 * `period_ms` milliseconds. Fixes the starvation failure mode where peers
 * with rare pushes never hit the opportunistic timeout check.
 *
 * Safe to call once after sar_amp_sender is ready. Re-calling is a no-op.
 *
 * @param period_ms  Sweep cadence in milliseconds (e.g., 500).
 */
void sar_amp_tick_start(uint32_t period_ms);

#endif /* SAR_AMP_SENDER_H */
