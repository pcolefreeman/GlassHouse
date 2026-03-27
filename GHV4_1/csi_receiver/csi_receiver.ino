/*
 * CSI Receiver Firmware
 * 
 * Receives ESP-NOW packets from the sender ESP32 and captures CSI
 * (Channel State Information) data using ESP-IDF APIs called from
 * Arduino framework. Outputs CSI data as CSV lines to USB serial
 * at 921600 baud for Python processing.
 *
 * Architecture:
 *   - WiFi initialized in STA mode (no connection to any AP)
 *   - Promiscuous mode enabled to sniff all packets on channel 11
 *   - CSI callback captures LLTF subcarrier data per received packet
 *   - Callback copies data to global buffer (no Serial in callback)
 *   - loop() prints buffered CSI as CSV when data is ready
 *
 * CSV Output Format:
 *   CSI_DATA,<seq>,<sender_mac>,<rssi>,<data_len>,<b0> <b1> <b2> ...
 *   (first 4 bytes of CSI data are skipped — first_word_invalid on ESP32)
 *
 * Hardware: ESP32-WROOM
 * Board Package: arduino-esp32 (v2.x or v3.x)
 */

#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_err.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#define SERIAL_BAUD       921600
#define WIFI_CHANNEL      11
#define CSI_BUF_SIZE      384   // max CSI data bytes (generous for LLTF)

// Sender factory MAC (Board 3) — MUST match the actual MAC of the sender board
static const uint8_t SENDER_MAC[6] = {0x68, 0xFE, 0x71, 0x90, 0x6B, 0x90};

// ---------------------------------------------------------------------------
// Global CSI data buffer (written by callback, read by loop)
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t  mac[6];
    int8_t   rssi;
    uint16_t data_len;
    uint8_t  data[CSI_BUF_SIZE];
    bool     first_word_invalid;
} csi_frame_t;

static volatile bool    csi_data_ready = false;
static csi_frame_t      csi_frame;
static uint32_t         seq_counter = 0;

// ---------------------------------------------------------------------------
// CSI Callback — runs in WiFi task context, keep it minimal
// ---------------------------------------------------------------------------

static volatile uint32_t debug_pkt_count = 0;  // count all CSI callbacks
static volatile uint8_t last_seen_mac[6] = {0}; // last MAC seen (for debug)

static void csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
    if (info == NULL || info->buf == NULL || info->len == 0) {
        return;
    }

    // Debug: count every packet and save the MAC
    debug_pkt_count++;
    memcpy((void *)last_seen_mac, info->mac, 6);

    // DEBUG: temporarily accept ALL packets to find sender
    // Filter by sender MAC address to ignore ambient traffic
    // if (memcmp(info->mac, SENDER_MAC, 6) != 0) {
    //     return;
    // }

    // If a previous frame hasn't been consumed yet, drop this one
    // (avoids corrupting buffer while loop() is reading it)
    if (csi_data_ready) {
        return;
    }

    // Copy metadata
    memcpy((void *)csi_frame.mac, info->mac, 6);
    csi_frame.rssi = info->rx_ctrl.rssi;
    csi_frame.first_word_invalid = info->first_word_invalid;

    // Copy CSI data bytes
    uint16_t copy_len = info->len;
    if (copy_len > CSI_BUF_SIZE) {
        copy_len = CSI_BUF_SIZE;
    }
    csi_frame.data_len = copy_len;
    memcpy((void *)csi_frame.data, info->buf, copy_len);

    // Signal loop() that data is ready
    csi_data_ready = true;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);  // let serial stabilize

    Serial.println("=== CSI Receiver Starting ===");

    // Initialize WiFi in STA mode — do NOT call WiFi.begin()
    WiFi.mode(WIFI_STA);
    Serial.println("WiFi mode: STA (no connection)");

    // Disconnect from any AP (safety measure)
    WiFi.disconnect();

    // Set fixed WiFi channel
    esp_err_t err;
    err = esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    Serial.printf("WiFi channel set to %d: %s\n", WIFI_CHANNEL,
                  err == ESP_OK ? "OK" : "FAIL");

    // Enable promiscuous mode — required to receive CSI from
    // ESP-NOW broadcasts when we're not connected to any AP
    err = esp_wifi_set_promiscuous(true);
    Serial.printf("Promiscuous mode: %s\n",
                  err == ESP_OK ? "ENABLED" : "FAIL");

    // Configure CSI capture — LLTF only (no HT-LTF on ESP32-WROOM
    // without an active HT connection)
    wifi_csi_config_t csi_config;
    csi_config.lltf_en           = true;   // Legacy Long Training Field
    csi_config.htltf_en          = false;   // No HT-LTF (not connected)
    csi_config.stbc_htltf2_en    = false;   // No STBC HT-LTF
    csi_config.ltf_merge_en      = true;    // Merge sub-carrier data
    csi_config.channel_filter_en = false;   // Don't filter by channel
    csi_config.manu_scale        = false;   // No manual scaling

    err = esp_wifi_set_csi_config(&csi_config);
    Serial.printf("CSI config (LLTF enabled): %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Register CSI callback
    err = esp_wifi_set_csi_rx_cb(&csi_rx_callback, NULL);
    Serial.printf("CSI callback registered: %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Enable CSI collection
    err = esp_wifi_set_csi(true);
    Serial.printf("CSI collection: %s\n",
                  err == ESP_OK ? "ENABLED" : "FAIL");

    // Initialize ESP-NOW — required so that incoming ESP-NOW frames
    // from the sender trigger the CSI callback
    err = esp_now_init();
    Serial.printf("ESP-NOW init: %s\n", err == ESP_OK ? "OK" : "FAIL");

    // Add sender as ESP-NOW peer so its frames are processed
    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, SENDER_MAC, 6);
    peer.channel = WIFI_CHANNEL;
    peer.encrypt = false;
    err = esp_now_add_peer(&peer);
    Serial.printf("Sender peer added: %s\n", err == ESP_OK ? "OK" : "FAIL");

    // Debug: verify actual channel after all init
    uint8_t primary_ch;
    wifi_second_chan_t second_ch;
    esp_wifi_get_channel(&primary_ch, &second_ch);
    Serial.printf("Actual WiFi channel: %d\n", primary_ch);

    // Print expected sender MAC
    Serial.printf("Filtering for sender MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  SENDER_MAC[0], SENDER_MAC[1], SENDER_MAC[2],
                  SENDER_MAC[3], SENDER_MAC[4], SENDER_MAC[5]);

    Serial.println("=== CSI Receiver Ready — waiting for packets ===");
}

// ---------------------------------------------------------------------------
// Loop — print CSI data from global buffer when ready
// ---------------------------------------------------------------------------

void loop() {
    loop_debug();  // print debug stats every 3s

    if (!csi_data_ready) {
        return;  // nothing to do — tight poll
    }

    // Determine offset: skip first 4 bytes if first_word_invalid
    // (hardware limitation on ESP32 — first 4 bytes of CSI buffer are garbage)
    uint16_t offset = csi_frame.first_word_invalid ? 4 : 0;

    // Safety check: make sure we have data beyond the offset
    if (csi_frame.data_len <= offset) {
        csi_data_ready = false;
        return;
    }

    // Build CSV line: CSI_DATA,seq,mac,rssi,valid_len,byte0 byte1 byte2 ...
    Serial.printf("CSI_DATA,%u,%02X:%02X:%02X:%02X:%02X:%02X,%d,%u,",
                  seq_counter,
                  csi_frame.mac[0], csi_frame.mac[1], csi_frame.mac[2],
                  csi_frame.mac[3], csi_frame.mac[4], csi_frame.mac[5],
                  csi_frame.rssi,
                  csi_frame.data_len - offset);

    // Print CSI bytes (skipping first_word_invalid bytes)
    for (uint16_t i = offset; i < csi_frame.data_len; i++) {
        if (i > offset) {
            Serial.print(' ');
        }
        Serial.print((int8_t)csi_frame.data[i]);
    }
    Serial.println();

    seq_counter++;
    csi_data_ready = false;
}

// Debug: print packet stats every 3 seconds
static unsigned long last_debug_ms = 0;
void loop_debug() {
    unsigned long now = millis();
    if (now - last_debug_ms >= 3000) {
        last_debug_ms = now;
        Serial.printf("[DEBUG] Total CSI callbacks: %u | Last MAC seen: %02X:%02X:%02X:%02X:%02X:%02X\n",
                      debug_pkt_count,
                      last_seen_mac[0], last_seen_mac[1], last_seen_mac[2],
                      last_seen_mac[3], last_seen_mac[4], last_seen_mac[5]);
    }
}
