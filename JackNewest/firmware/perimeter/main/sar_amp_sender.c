/**
 * @file sar_amp_sender.c
 * @brief Implementation of per-peer batched amplitude sender (Lane B Fix 3).
 *
 * One 208-byte UDP datagram per (peer, batch-of-48-samples). Flushes when
 * the ring is full OR when the oldest sample is more than 1 second old.
 *
 * Called directly from csi_collector.c's WiFi CSI callback, so hot path
 * stays lock-lite: per-peer portMUX spinlock protects only the ring
 * state machine, and the UDP send happens after the critical section
 * using a stack-local copy of the packet.
 */

#include "sar_amp_sender.h"
#include "nvs_config.h"
#include "stream_sender.h"

#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"

/* Access to g_nvs_config for node_id. */
extern nvs_config_t g_nvs_config;

static const char *TAG = "sar_amp";

/** Flush peer ring if oldest sample is older than this. */
#define SAR_AMP_TIMEOUT_US (1 * 1000 * 1000)  /* 1 second */

/**
 * Per-peer ring. Index into s_rings[] is (peer_id - 1) when peer_id
 * falls in [1, SAR_MAX_PEERS]. Everything else is dropped.
 *
 * Invariants:
 *   - count <= SAR_AMP_BATCH_SIZE
 *   - if count > 0, ts_first_us is the esp_timer_get_time() of amps[0].
 *   - amps[0..count-1] and ts[0..count-1] are valid.
 */
typedef struct {
    float    amps[SAR_AMP_BATCH_SIZE];
    int64_t  ts[SAR_AMP_BATCH_SIZE];
    uint16_t count;
    int64_t  ts_first_us;
} sar_peer_ring_t;

static sar_peer_ring_t s_rings[SAR_MAX_PEERS];

/* ESP-IDF portMUX — cheap spinlock, safe from WiFi callback context.
 * One mutex covers all peers; contention is negligible because the
 * WiFi CSI callback is single-producer per core. */
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;

/**
 * Serialize a ring's contents into a sar_amp_pkt_t. Does NOT clear the
 * ring — the caller is expected to reset ring->count = 0 inside the
 * critical section and release the lock before calling this.
 *
 * @param copy_amps  Local copy of the amps array (post-CS snapshot).
 * @param copy_ts    Local copy of the timestamps array.
 * @param n          Number of valid entries in copy_amps / copy_ts.
 * @param peer_id    Peer ID to stamp into the packet.
 * @param pkt        Output packet (caller-allocated).
 */
static void build_packet(const float *copy_amps,
                         const int64_t *copy_ts,
                         uint16_t n,
                         uint8_t peer_id,
                         sar_amp_pkt_t *pkt)
{
    memset(pkt, 0, sizeof(*pkt));
    pkt->magic   = SAR_AMP_MAGIC;
    pkt->node_id = g_nvs_config.node_id;
    pkt->peer_id = peer_id;
    pkt->n_samples = n;

    if (n > 0) {
        /* Low 32 bits of the int64 us timestamp — enough for an interval
         * reference, and wraps every ~71 minutes which is harmless for
         * the host decoder (it treats these as relative markers). */
        pkt->batch_start_us = (uint32_t)(copy_ts[0] & 0xFFFFFFFFu);
    }

    if (n >= 2) {
        int64_t span = copy_ts[n - 1] - copy_ts[0];
        if (span > 0) {
            pkt->interval_us = (uint32_t)(span / (int64_t)(n - 1));
        }
    }

    memcpy(pkt->amps, copy_amps, n * sizeof(float));
    /* amps[n..BATCH_SIZE-1] stay zero from the memset above. */
}

/**
 * Flush one peer if its ring is either full or has timed out.
 * Caller must NOT hold s_mux when calling — this function handles the
 * critical section itself.
 */
static void flush_peer_if_ready(uint8_t peer_id, int64_t now_us, bool force)
{
    if (peer_id == 0 || peer_id > SAR_MAX_PEERS) {
        return;
    }
    const uint8_t idx = peer_id - 1;

    float    local_amps[SAR_AMP_BATCH_SIZE];
    int64_t  local_ts[SAR_AMP_BATCH_SIZE];
    uint16_t n = 0;

    portENTER_CRITICAL(&s_mux);
    sar_peer_ring_t *r = &s_rings[idx];
    bool should_flush = false;
    if (r->count >= SAR_AMP_BATCH_SIZE) {
        should_flush = true;
    } else if (r->count > 0 && (now_us - r->ts_first_us) >= SAR_AMP_TIMEOUT_US) {
        should_flush = true;
    } else if (force && r->count > 0) {
        should_flush = true;
    }

    if (should_flush) {
        n = r->count;
        memcpy(local_amps, r->amps, n * sizeof(float));
        memcpy(local_ts,   r->ts,   n * sizeof(int64_t));
        r->count = 0;
        r->ts_first_us = 0;
    }
    portEXIT_CRITICAL(&s_mux);

    if (n == 0) {
        return;
    }

    sar_amp_pkt_t pkt;
    build_packet(local_amps, local_ts, n, peer_id, &pkt);

    int ret = stream_sender_send((const uint8_t *)&pkt, sizeof(pkt));
    if (ret <= 0) {
        /* Do not spam logs — keep it to a terse warn at most. */
        ESP_LOGW(TAG, "UDP send failed for peer %u (n=%u)",
                 (unsigned)peer_id, (unsigned)n);
    }
}

void sar_amp_push(uint8_t peer_id, float mean_amp, int64_t ts_us)
{
    if (peer_id == 0 || peer_id > SAR_MAX_PEERS) {
        return;
    }
    const uint8_t idx = peer_id - 1;

    bool full_flush_needed = false;

    portENTER_CRITICAL(&s_mux);
    sar_peer_ring_t *r = &s_rings[idx];
    if (r->count == 0) {
        r->ts_first_us = ts_us;
    }
    if (r->count < SAR_AMP_BATCH_SIZE) {
        r->amps[r->count] = mean_amp;
        r->ts[r->count]   = ts_us;
        r->count++;
    }
    if (r->count >= SAR_AMP_BATCH_SIZE) {
        full_flush_needed = true;
    }
    portEXIT_CRITICAL(&s_mux);

    if (full_flush_needed) {
        flush_peer_if_ready(peer_id, ts_us, false);
        return;
    }

    /* Opportunistic timeout check — only the pushing peer, cheap. */
    int64_t oldest_delta;
    portENTER_CRITICAL(&s_mux);
    oldest_delta = (r->count > 0) ? (ts_us - r->ts_first_us) : 0;
    portEXIT_CRITICAL(&s_mux);
    if (oldest_delta >= SAR_AMP_TIMEOUT_US) {
        flush_peer_if_ready(peer_id, ts_us, false);
    }
}

void sar_amp_check_timeouts(void)
{
    int64_t now = esp_timer_get_time();
    for (uint8_t pid = 1; pid <= SAR_MAX_PEERS; pid++) {
        flush_peer_if_ready(pid, now, false);
    }
}

static esp_timer_handle_t s_tick_timer = NULL;

static void sar_amp_tick_cb(void *arg)
{
    (void)arg;
    sar_amp_check_timeouts();
}

void sar_amp_tick_start(uint32_t period_ms)
{
    if (s_tick_timer != NULL) {
        return;  /* already started — idempotent */
    }
    esp_timer_create_args_t args = {
        .callback = sar_amp_tick_cb,
        .name = "sar_amp_tick",
        .dispatch_method = ESP_TIMER_TASK,  /* sendto needs task context, not ISR */
    };
    if (esp_timer_create(&args, &s_tick_timer) != ESP_OK) {
        ESP_LOGW(TAG, "sar_amp_tick_start: timer create failed");
        s_tick_timer = NULL;
        return;
    }
    if (esp_timer_start_periodic(s_tick_timer, (uint64_t)period_ms * 1000) != ESP_OK) {
        ESP_LOGW(TAG, "sar_amp_tick_start: timer start failed");
        esp_timer_delete(s_tick_timer);
        s_tick_timer = NULL;
        return;
    }
    ESP_LOGI(TAG, "sar_amp timeout sweep: %lu ms", (unsigned long)period_ms);
}
