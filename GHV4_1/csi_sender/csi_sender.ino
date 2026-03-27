/*
 * CSI Sender Firmware
 * 
 * Broadcasts ESP-NOW packets at ~20 Hz (50ms interval) on WiFi
 * channel 11 so the paired receiver ESP32 can capture CSI data
 * from these transmissions.
 *
 * Architecture:
 *   - WiFi initialized in STA mode (no connection to any AP)
 *   - Custom MAC address set to match receiver's SENDER_MAC filter
 *   - ESP-NOW initialized with broadcast peer
 *   - loop() sends a small payload every 50ms using millis()
 *
 * Payload: 4-byte sequence counter (uint32_t). The content doesn't
 * matter for CSI — the receiver extracts CSI from the physical layer
 * of any received packet. The payload just makes each packet unique.
 *
 * Hardware: ESP32-WROOM
 * Board Package: arduino-esp32 (v2.x or v3.x)
 *
 * NOTE: This project uses 5 ESP32-WROOM modules total:
 *   - 4 perimeter nodes (used in S02+ for round-robin TX)
 *   - 1 coordinator
 *   For this slice (S01), only 2 are used: this sender and one receiver.
 */

#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#define SERIAL_BAUD       115200     // sender only needs basic debug output
#define WIFI_CHANNEL      11         // must match receiver
#define TX_INTERVAL_MS    50         // ~20 Hz broadcast rate

// Custom MAC address — MUST match SENDER_MAC in csi_receiver.ino
// This is {0x24, 0x6F, 0x28, 0xAA, 0xBB, 0xCC}
static uint8_t CUSTOM_MAC[6] = {0x24, 0x6F, 0x28, 0xAA, 0xBB, 0xCC};

// Broadcast address for ESP-NOW (all 0xFF = broadcast to all peers)
static uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

static uint32_t seq_counter = 0;
static unsigned long last_tx_time = 0;

// ---------------------------------------------------------------------------
// ESP-NOW send callback — runs quickly, Serial.println is OK here
// ---------------------------------------------------------------------------

void on_data_sent(const esp_now_send_info_t *tx_info, esp_now_send_status_t status) {
    // Intentionally minimal — just log success/fail
    if (status != ESP_NOW_SEND_SUCCESS) {
        Serial.printf("ESP-NOW TX FAIL at seq %u\n", seq_counter - 1);
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);  // let serial stabilize

    Serial.println("=== CSI Sender Starting ===");

    // Initialize WiFi in STA mode — do NOT call WiFi.begin()
    WiFi.mode(WIFI_STA);
    Serial.println("WiFi mode: STA (no connection)");

    // Disconnect from any AP (safety measure)
    WiFi.disconnect();

    // Set custom MAC address so receiver can filter by known MAC
    esp_err_t err;
    err = esp_wifi_set_mac(WIFI_IF_STA, CUSTOM_MAC);
    Serial.printf("Custom MAC set: %s\n", err == ESP_OK ? "OK" : "FAIL");

    // Set fixed WiFi channel — must match receiver
    err = esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    Serial.printf("WiFi channel set to %d: %s\n", WIFI_CHANNEL,
                  err == ESP_OK ? "OK" : "FAIL");

    // Initialize ESP-NOW
    err = esp_now_init();
    Serial.printf("ESP-NOW init: %s\n", err == ESP_OK ? "OK" : "FAIL");

    // Register send callback
    esp_now_register_send_cb(on_data_sent);

    // Add broadcast peer
    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, BROADCAST_ADDR, 6);
    peer.channel = WIFI_CHANNEL;
    peer.encrypt = false;

    err = esp_now_add_peer(&peer);
    Serial.printf("Broadcast peer added: %s\n", err == ESP_OK ? "OK" : "FAIL");

    // Print own MAC for verification
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    Serial.printf("Own MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    Serial.printf("TX rate: %d Hz (every %d ms)\n",
                  1000 / TX_INTERVAL_MS, TX_INTERVAL_MS);

    Serial.println("=== CSI Sender Ready — broadcasting ===");
}

// ---------------------------------------------------------------------------
// Loop — broadcast ESP-NOW packet every TX_INTERVAL_MS
// ---------------------------------------------------------------------------

void loop() {
    unsigned long now = millis();

    if (now - last_tx_time < TX_INTERVAL_MS) {
        return;  // not time yet
    }
    last_tx_time = now;

    // Payload is just the sequence counter — content doesn't matter
    // for CSI, but it makes packets unique and aids debugging
    uint8_t payload[4];
    memcpy(payload, &seq_counter, sizeof(seq_counter));

    esp_err_t result = esp_now_send(BROADCAST_ADDR, payload, sizeof(payload));

    if (result == ESP_OK) {
        // Only print every 20th packet to avoid flooding serial
        // (at 20 Hz, this prints once per second)
        if (seq_counter % 20 == 0) {
            Serial.printf("ESP-NOW TX: seq=%u (OK)\n", seq_counter);
        }
    } else {
        Serial.printf("ESP-NOW TX: seq=%u (ERR: %d)\n", seq_counter, result);
    }

    seq_counter++;
}
