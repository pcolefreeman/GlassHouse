// firmware/coordinator/main/main.c
// GlassHouse v2 Coordinator — SoftAP + UDP-to-COBS-serial bridge

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "esp_event.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "driver/uart.h"
#include "hal/usb_serial_jtag_ll.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "coordinator";

#define SOFTAP_SSID       "CSI_NET_V2"
#define SOFTAP_PASS       "glasshouse"
#define SOFTAP_CHANNEL    1
#define SOFTAP_MAX_CONN   5
#define UDP_PORT           4210
// Data output goes via USB-Serial/JTAG (the built-in USB on ESP32-S3).
// We write directly to the hardware FIFO to avoid disrupting the USB connection.
// Log output is redirected to UART1 to avoid corrupting the COBS stream.
#define LOG_UART_NUM       UART_NUM_1
#define LOG_UART_TX_PIN    17
#define LOG_UART_RX_PIN    18
#define UART_BAUD          921600
/* SAR quick-win fix 5: raised from 512 → 2100 to accommodate full ADR-018
 * CSI frames (legacy 512 silently dropped legit traffic; observed CSI rate
 * dropped to 0.4–0.8 Hz per node as a result). */
#define MAX_PKT_SIZE       2100
/* Drop logging cadence: warn on every Nth drop *or* every N seconds,
 * whichever comes first. */
#define DROP_LOG_EVERY_N       100
#define DROP_LOG_EVERY_US      (60LL * 1000LL * 1000LL)  /* 60 s */

// --- Log redirect: send ESP_LOG output to UART1 instead of USB-CDC ---
static int uart1_vprintf(const char *fmt, va_list args)
{
    char buf[256];
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    if (len > 0) {
        uart_write_bytes(LOG_UART_NUM, buf, len);
    }
    return len;
}

// --- Raw USB-Serial/JTAG FIFO write (bypasses driver to avoid USB disconnect) ---
// Keep retries low so a stalled USB host doesn't block the UDP receive loop.
// If the host isn't reading (port closed), drop the packet quickly and move on.
static size_t usb_jtag_write_raw(const uint8_t *data, size_t len)
{
    size_t offset = 0;
    int retries = 0;
    while (offset < len && retries < 5) {
        uint32_t written = usb_serial_jtag_ll_write_txfifo(data + offset, len - offset);
        usb_serial_jtag_ll_txfifo_flush();
        offset += written;
        if (offset < len) {
            vTaskDelay(1); // yield if FIFO was full, let USB drain
            retries++;
        }
    }
    return offset;
}

// --- COBS encoding ---
// Encodes `src` (len bytes) into `dst`. Returns encoded length.
// `dst` must be at least len + len/254 + 1 bytes.
static size_t cobs_encode(const uint8_t *src, size_t len, uint8_t *dst)
{
    size_t read_idx = 0, write_idx = 1, code_idx = 0;
    uint8_t code = 1;

    while (read_idx < len) {
        if (src[read_idx] == 0x00) {
            dst[code_idx] = code;
            code = 1;
            code_idx = write_idx++;
        } else {
            dst[write_idx++] = src[read_idx];
            code++;
            if (code == 0xFF) {
                dst[code_idx] = code;
                code = 1;
                code_idx = write_idx++;
            }
        }
        read_idx++;
    }
    dst[code_idx] = code;
    return write_idx;
}

// --- WiFi SoftAP setup ---
static void wifi_init_softap(void)
{
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = SOFTAP_SSID,
            .ssid_len = strlen(SOFTAP_SSID),
            .channel = SOFTAP_CHANNEL,
            .password = SOFTAP_PASS,
            .max_connection = SOFTAP_MAX_CONN,
            .authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "SoftAP started: SSID=%s CH=%d", SOFTAP_SSID, SOFTAP_CHANNEL);
}

// --- UART setup ---
static void uart_init(void)
{
    // Redirect ESP log output to UART1 so USB-CDC stays clean for COBS data
    uart_config_t log_cfg = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_param_config(LOG_UART_NUM, &log_cfg));
    ESP_ERROR_CHECK(uart_set_pin(LOG_UART_NUM, LOG_UART_TX_PIN, LOG_UART_RX_PIN,
                                  UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(LOG_UART_NUM, 1024, 1024, 0, NULL, 0));
    esp_log_set_vprintf((vprintf_like_t)&uart1_vprintf);
}

// --- UDP listener task ---
static void udp_bridge_task(void *pvParameters)
{
    /* rx_buf sized to MAX_PKT_SIZE (raised to 2100 for full ADR-018 frames).
     * cobs_buf has worst-case COBS expansion: +1 header per 254 bytes + delimiter. */
    static uint8_t rx_buf[MAX_PKT_SIZE];
    static uint8_t cobs_buf[MAX_PKT_SIZE + MAX_PKT_SIZE / 254 + 2];
    uint32_t pkt_count = 0;
    uint32_t usb_write_fail_count = 0;

    /* SAR quick-win fix 5: track and periodically log oversized-drop events
     * (previously silent per AC-5). */
    uint32_t oversize_drop_count = 0;
    uint32_t oversize_drops_since_log = 0;
    uint32_t oversize_max_seen = 0;
    int64_t  last_drop_log_us = 0;

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(UDP_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed: %d", errno);
        vTaskDelete(NULL);
        return;
    }

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        ESP_LOGE(TAG, "Socket bind failed: %d", errno);
        close(sock);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "UDP bridge listening on port %d", UDP_PORT);

    while (1) {
        int len = recvfrom(sock, rx_buf, sizeof(rx_buf), 0, NULL, NULL);
        if (len <= 0) continue;
        if ((size_t)len > MAX_PKT_SIZE) {
            /* SAR quick-win fix 5: no longer silent — count drops, track the
             * largest oversized packet seen, and warn via ESP_LOGW (routed to
             * UART1 by uart1_vprintf) every DROP_LOG_EVERY_N drops or every
             * DROP_LOG_EVERY_US, whichever comes first. */
            oversize_drop_count++;
            oversize_drops_since_log++;
            if ((uint32_t)len > oversize_max_seen) oversize_max_seen = (uint32_t)len;

            int64_t now = esp_timer_get_time();
            bool count_gate = (oversize_drops_since_log >= DROP_LOG_EVERY_N);
            bool time_gate  = (last_drop_log_us != 0) &&
                              ((now - last_drop_log_us) >= DROP_LOG_EVERY_US);
            if (count_gate || time_gate || last_drop_log_us == 0) {
                ESP_LOGW(TAG,
                    "oversize drop: len=%d > MAX_PKT_SIZE=%d (total=%lu, since_last=%lu, max_seen=%lu)",
                    len, MAX_PKT_SIZE,
                    (unsigned long)oversize_drop_count,
                    (unsigned long)oversize_drops_since_log,
                    (unsigned long)oversize_max_seen);
                oversize_drops_since_log = 0;
                last_drop_log_us = now;
            }
            continue;
        }

        size_t cobs_len = cobs_encode(rx_buf, (size_t)len, cobs_buf);
        cobs_buf[cobs_len] = 0x00; // COBS delimiter
        size_t written = usb_jtag_write_raw(cobs_buf, cobs_len + 1);
        if (written < cobs_len + 1) {
            usb_write_fail_count++;
        }

        pkt_count++;
        if (pkt_count % 100 == 0) {
            ESP_LOGI(TAG, "Forwarded %lu packets (%lu USB write failures)",
                     (unsigned long)pkt_count, (unsigned long)usb_write_fail_count);
        }
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    uart_init();
    wifi_init_softap();

    xTaskCreate(udp_bridge_task, "udp_bridge", 4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "GlassHouse v2 Coordinator ready");
}
