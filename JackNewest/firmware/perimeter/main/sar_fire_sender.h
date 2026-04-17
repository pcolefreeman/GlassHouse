/**
 * @file sar_fire_sender.h
 * @brief Firehose per-subcarrier amplitude stream (SAR Option D).
 *
 * Emits one UDP packet per CSI callback containing the full per-subcarrier
 * amplitude vector (up to 64 subcarriers, uint8 each). This preserves the
 * room-independent temporal structure at physiological bands that collapse-
 * to-scalar features (link variance, mean_amp) destroy.
 *
 * Rate-limited per peer at ~50 Hz via SAR_FIRE_MIN_INTERVAL_US. At 4 nodes ×
 * 3 peers × 50 Hz × 80 B = 48 KB/s aggregate on USB (≈52% of 92 KB/s).
 *
 * Wire magic: 0xC5110008 (next after SAR_AMP_MAGIC=0xC5110007).
 */

#ifndef SAR_FIRE_SENDER_H
#define SAR_FIRE_SENDER_H

#include <stdint.h>

/** Magic for the firehose per-subcarrier packet. */
#define SAR_FIRE_MAGIC           0xC5110008u

/** Max subcarriers carried per packet. Matches typical HT20 CSI profile. */
#define SAR_FIRE_MAX_SUBCAR      64

/** Per-peer min interval between SAR_FIRE emissions (µs). 20000 = 50 Hz. */
#define SAR_FIRE_MIN_INTERVAL_US 20000

/**
 * Per-callback firehose packet.
 *
 * 16-byte header + 64 uint8 amplitudes = 80 bytes total.
 */
typedef struct __attribute__((packed)) {
    uint32_t magic;                         /**< SAR_FIRE_MAGIC. */
    uint8_t  node_id;                       /**< This reporter's NVS node_id. */
    uint8_t  peer_id;                       /**< Source node of this CSI trigger. */
    uint16_t n_subcar;                      /**< Valid entries in amps[]. */
    uint32_t ts_us;                         /**< esp_timer_get_time() low 32 bits. */
    int8_t   rssi;                          /**< Link RSSI (dBm). */
    int8_t   noise_floor;                   /**< Noise floor (dBm). */
    uint16_t reserved;                      /**< Pad to 4-byte align. */
    uint8_t  amps[SAR_FIRE_MAX_SUBCAR];     /**< Per-subcarrier uint8 amplitude. */
} sar_fire_pkt_t;

_Static_assert(sizeof(sar_fire_pkt_t) == 80, "sar_fire_pkt_t must be 80 bytes");

/**
 * Emit a firehose packet for one CSI callback.
 *
 * Per-peer rate-limited: calls faster than SAR_FIRE_MIN_INTERVAL_US since the
 * last successful emission for the same peer are dropped silently.
 *
 * Safe to call from WiFi CSI callback context (single-producer per core).
 *
 * @param peer_id      Source node ID (1..4). Out-of-range peers are dropped.
 * @param amps         Per-subcarrier amplitude bytes (0..255 saturating).
 * @param n_subcar     Number of valid entries in amps[] (≤ SAR_FIRE_MAX_SUBCAR).
 * @param rssi         Link RSSI from rx_ctrl.
 * @param noise_floor  Noise floor from rx_ctrl.
 * @param ts_us        Timestamp from esp_timer_get_time().
 */
void sar_fire_push(uint8_t peer_id,
                   const uint8_t *amps,
                   uint16_t n_subcar,
                   int8_t rssi,
                   int8_t noise_floor,
                   int64_t ts_us);

#endif /* SAR_FIRE_SENDER_H */
