/**
 * @file stream_sender.c
 * @brief UDP stream sender for CSI frames.
 *
 * Opens a UDP socket and sends serialized ADR-018 frames to the aggregator.
 */

#include "stream_sender.h"

#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "sdkconfig.h"

static const char *TAG = "stream_sender";

static int s_sock = -1;
static struct sockaddr_in s_dest_addr;

/**
 * ENOMEM backoff state.
 * When sendto fails with ENOMEM (errno 12), we suppress further sends for
 * a cooldown period to let lwIP reclaim packet buffers.  Without this,
 * rapid-fire CSI callbacks can exhaust the pbuf pool and crash the device.
 */
static SemaphoreHandle_t s_send_mutex;       /* Serializes stream_sender_send() */
static int64_t s_backoff_until_us = 0;       /* esp_timer timestamp to resume */

/*
 * SAR quick-win fix 6: shorten the ENOMEM suppression window.
 *
 * Rationale: at the 50 Hz CSI target a new frame arrives every 20 ms.  A
 * 100 ms cooldown drops 5 consecutive frames for every transient pbuf
 * exhaustion, which dominates the observed 0.4–0.8 Hz off-device rate.
 * Paired with the raised LWIP pbuf pool sizes in sdkconfig.defaults
 * (CONFIG_LWIP_UDP_RECVMBOX_SIZE=32, CONFIG_LWIP_TCPIP_RECVMBOX_SIZE=64),
 * a 10 ms window still lets lwIP reclaim buffers without stalling the
 * stream.  Non-SAR builds keep the 100 ms behavior.
 */
#if CONFIG_SAR_MODE
#define ENOMEM_COOLDOWN_MS  10
#else
#define ENOMEM_COOLDOWN_MS  100
#endif
#define ENOMEM_LOG_INTERVAL 50               /* log every Nth suppressed send */
static uint32_t s_enomem_suppressed = 0;
static uint32_t s_enomem_total_events = 0;   /* every ENOMEM trigger (not suppressions) */

static int sender_init_internal(const char *ip, uint16_t port)
{
    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket: errno %d", errno);
        return -1;
    }

    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    s_dest_addr.sin_family = AF_INET;
    s_dest_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip, &s_dest_addr.sin_addr) <= 0) {
        ESP_LOGE(TAG, "Invalid target IP: %s", ip);
        close(s_sock);
        s_sock = -1;
        return -1;
    }

    ESP_LOGI(TAG, "UDP sender initialized: %s:%d", ip, port);
    return 0;
}

int stream_sender_init(void)
{
    return sender_init_internal(CONFIG_CSI_TARGET_IP, CONFIG_CSI_TARGET_PORT);
}

int stream_sender_init_with(const char *ip, uint16_t port)
{
    int rc = sender_init_internal(ip, port);
    if (rc != 0) return rc;

    s_send_mutex = xSemaphoreCreateMutex();
    if (s_send_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create send mutex");
        close(s_sock);
        s_sock = -1;
        return ESP_ERR_NO_MEM;
    }

    return 0;
}

int stream_sender_send(const uint8_t *data, size_t len)
{
    if (s_sock < 0) {
        return -1;
    }

    xSemaphoreTake(s_send_mutex, portMAX_DELAY);

    /* ENOMEM backoff: if we recently exhausted lwIP buffers, skip sends
     * until the cooldown expires.  This prevents the cascade of failed
     * sendto calls that leads to a guru meditation crash. */
    if (s_backoff_until_us > 0) {
        int64_t now = esp_timer_get_time();
        if (now < s_backoff_until_us) {
            s_enomem_suppressed++;
            if ((s_enomem_suppressed % ENOMEM_LOG_INTERVAL) == 1) {
                ESP_LOGW(TAG, "sendto suppressed (ENOMEM backoff, %lu dropped)",
                         (unsigned long)s_enomem_suppressed);
            }
            xSemaphoreGive(s_send_mutex);
            return -1;
        }
        /* Cooldown expired — resume sending */
        ESP_LOGI(TAG, "ENOMEM backoff expired, resuming sends (%lu were suppressed)",
                 (unsigned long)s_enomem_suppressed);
        s_backoff_until_us = 0;
        s_enomem_suppressed = 0;
    }

    int sent = sendto(s_sock, data, len, 0,
                      (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
    if (sent < 0) {
        if (errno == ENOMEM) {
            /* Start backoff to let lwIP reclaim buffers.  SAR quick-win fix 6:
             * every ENOMEM event is logged via ESP_LOGW (routes to UART1). */
            s_backoff_until_us = esp_timer_get_time() +
                                 (int64_t)ENOMEM_COOLDOWN_MS * 1000;
            s_enomem_total_events++;
            ESP_LOGW(TAG,
                     "sendto ENOMEM #%lu — backing off for %d ms (len=%u)",
                     (unsigned long)s_enomem_total_events,
                     ENOMEM_COOLDOWN_MS, (unsigned)len);
        } else {
            ESP_LOGW(TAG, "sendto failed: errno %d", errno);
        }
        xSemaphoreGive(s_send_mutex);
        return -1;
    }

    xSemaphoreGive(s_send_mutex);
    return sent;
}

void stream_sender_deinit(void)
{
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
        ESP_LOGI(TAG, "UDP sender closed");
    }
}
