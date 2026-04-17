/**
 * @file sar_fire_sender.c
 * @brief Implementation of SAR firehose per-subcarrier stream (Option D).
 *
 * Stateless: one packet per CSI callback per peer (subject to rate limit).
 * No ring buffer, no timer — each call either emits or drops immediately.
 * This eliminates the starvation failure mode we saw with sar_amp_sender.c
 * where rarely-pushed peer rings never flushed.
 *
 * The only per-peer state is the last-emit timestamp used for rate limiting.
 */

#include "sar_fire_sender.h"
#include "nvs_config.h"
#include "stream_sender.h"

#include <string.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"

extern nvs_config_t g_nvs_config;

static const char *TAG = "sar_fire";

/** Last-emit timestamps per peer_id index [1..SAR_FIRE_MAX_PEERS]. */
#define SAR_FIRE_MAX_PEERS 4
static int64_t s_last_emit_us[SAR_FIRE_MAX_PEERS] = {0, 0, 0, 0};

/** Spinlock for rate-limit state. Cheap because WiFi CSI callback is
 *  single-producer and contention is negligible. */
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;

void sar_fire_push(uint8_t peer_id,
                   const uint8_t *amps,
                   uint16_t n_subcar,
                   int8_t rssi,
                   int8_t noise_floor,
                   int64_t ts_us)
{
    if (peer_id == 0 || peer_id > SAR_FIRE_MAX_PEERS) {
        return;
    }
    if (amps == NULL || n_subcar == 0) {
        return;
    }

    const uint8_t idx = peer_id - 1;

    /* Per-peer rate limit — drop if last emit too recent. */
    bool allow;
    portENTER_CRITICAL(&s_mux);
    if ((ts_us - s_last_emit_us[idx]) >= SAR_FIRE_MIN_INTERVAL_US) {
        s_last_emit_us[idx] = ts_us;
        allow = true;
    } else {
        allow = false;
    }
    portEXIT_CRITICAL(&s_mux);

    if (!allow) {
        return;
    }

    /* Build packet on stack — no heap. */
    sar_fire_pkt_t pkt;
    memset(&pkt, 0, sizeof(pkt));

    pkt.magic       = SAR_FIRE_MAGIC;
    pkt.node_id     = g_nvs_config.node_id;
    pkt.peer_id     = peer_id;
    pkt.n_subcar    = (n_subcar > SAR_FIRE_MAX_SUBCAR) ? SAR_FIRE_MAX_SUBCAR : n_subcar;
    pkt.ts_us       = (uint32_t)(ts_us & 0xFFFFFFFFu);
    pkt.rssi        = rssi;
    pkt.noise_floor = noise_floor;
    pkt.reserved    = 0;

    memcpy(pkt.amps, amps, pkt.n_subcar);
    /* amps[n_subcar..MAX-1] stay zero from memset above. */

    int ret = stream_sender_send((const uint8_t *)&pkt, sizeof(pkt));
    if (ret <= 0) {
        /* Terse — don't spam at 50 Hz. */
        static uint32_t s_fail_count = 0;
        s_fail_count++;
        if ((s_fail_count % 100) == 1) {
            ESP_LOGW(TAG, "stream_sender_send failed (fail #%lu, peer %u)",
                     (unsigned long)s_fail_count, (unsigned)peer_id);
        }
    }
}
