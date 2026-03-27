/*
 * Perimeter Node Firmware — Multi-Node Round-Robin CSI Network
 *
 * Unified firmware for all 4 perimeter ESP32 nodes (A, B, C, D).
 * Each node operates in two modes depending on whose turn it is:
 *
 *   TX mode (my turn): Broadcast CSI_TX stimulus packets so other
 *     nodes capture CSI from this node's transmission.
 *
 *   RX mode (another node's turn): Capture CSI from the transmitting
 *     node's stimulus packet, then relay the CSI data to the coordinator
 *     via unicast ESP-NOW (CSI_REPORT).
 *
 * Protocol:
 *   1. Coordinator broadcasts TURN_CMD [0x01, node_id] to designate
 *      which node transmits next.
 *   2. Designated node broadcasts CSI_TX [0x02, node_id, seq_hi, seq_lo, ...]
 *      after a brief settling delay.
 *   3. Other nodes capture CSI from that stimulus, then unicast
 *      CSI_REPORT [0x03, tx_id, rx_id, rssi, len_hi, len_lo, csi...]
 *      to the coordinator.
 *
 * Flash instructions:
 *   Change NODE_ID below before flashing each board:
 *     0 = Node A, 1 = Node B, 2 = Node C, 3 = Node D
 *
 * Hardware: ESP32-WROOM
 * Board Package: arduino-esp32 (v2.x or v3.x)
 */

#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_err.h"

// ===========================================================================
// *** CHANGE THIS BEFORE FLASHING EACH BOARD ***
// 0 = Node A, 1 = Node B, 2 = Node C, 3 = Node D
// ===========================================================================
#define NODE_ID  0

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#define SERIAL_BAUD       115200     // debug output only (coordinator does USB)
#define WIFI_CHANNEL      11
#define CSI_BUF_SIZE      384        // max CSI data bytes
#define NUM_NODES         4
#define TX_BURST_COUNT    3          // stimulus packets per turn
#define TX_BURST_DELAY_MS 5          // ms between stimulus packets
#define SETTLE_DELAY_MS   10         // ms after TURN_CMD before transmitting

// Message type bytes — must match coordinator exactly
#define MSG_TURN_CMD    0x01
#define MSG_CSI_TX      0x02
#define MSG_CSI_REPORT  0x03

// ---------------------------------------------------------------------------
// MAC addresses — must match coordinator and all perimeter nodes
// ---------------------------------------------------------------------------

// Per-node custom MACs:  A=01, B=02, C=03, D=04
static const uint8_t NODE_MACS[NUM_NODES][6] = {
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x01},  // Node A (ID 0)
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x02},  // Node B (ID 1)
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x03},  // Node C (ID 2)
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x04},  // Node D (ID 3)
};

// Coordinator MAC — unicast target for CSI_REPORT
// String form: 24:6F:28:AA:00:00
static const uint8_t COORDINATOR_MAC[6] = {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x00};

// Broadcast address for ESP-NOW (hearing TURN commands + sending CSI_TX)
static const uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ---------------------------------------------------------------------------
// Node identity (derived from NODE_ID at compile time)
// ---------------------------------------------------------------------------

static uint8_t my_mac[6];
static const char NODE_LABELS[NUM_NODES] = {'A', 'B', 'C', 'D'};

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

static volatile bool   is_my_turn      = false;   // set by ESP-NOW recv callback
static volatile uint8_t current_tx_id  = 0xFF;    // who is currently transmitting
static uint16_t        tx_seq_counter  = 0;        // sequence for CSI_TX packets

// ---------------------------------------------------------------------------
// CSI capture buffer (written by callback, read by loop)
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t  sender_mac[6];
    int8_t   rssi;
    uint16_t data_len;
    uint8_t  data[CSI_BUF_SIZE];
    bool     first_word_invalid;
    uint8_t  tx_node_id;   // which node was transmitting (from current_tx_id)
} csi_frame_t;

static volatile bool  csi_data_ready = false;
static csi_frame_t    csi_frame;

// ---------------------------------------------------------------------------
// CSI Callback — runs in WiFi task context, ZERO I/O allowed
// ---------------------------------------------------------------------------

static void csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
    if (info == NULL || info->buf == NULL || info->len == 0) {
        return;
    }

    // Only capture CSI when we are in RX mode (another node's turn)
    if (is_my_turn || current_tx_id == 0xFF) {
        return;
    }

    // Filter by sender MAC: must be a known perimeter node, not self
    bool from_known_node = false;
    for (int i = 0; i < NUM_NODES; i++) {
        if (i == NODE_ID) continue;  // skip own MAC
        if (memcmp(info->mac, NODE_MACS[i], 6) == 0) {
            from_known_node = true;
            break;
        }
    }
    if (!from_known_node) {
        return;  // ignore coordinator, self, and ambient traffic
    }

    // If previous frame hasn't been consumed, drop this one
    if (csi_data_ready) {
        return;
    }

    // Copy metadata
    memcpy((void *)csi_frame.sender_mac, info->mac, 6);
    csi_frame.rssi = info->rx_ctrl.rssi;
    csi_frame.first_word_invalid = info->first_word_invalid;
    csi_frame.tx_node_id = current_tx_id;

    // Copy CSI data
    uint16_t copy_len = info->len;
    if (copy_len > CSI_BUF_SIZE) {
        copy_len = CSI_BUF_SIZE;
    }
    csi_frame.data_len = copy_len;
    memcpy((void *)csi_frame.data, info->buf, copy_len);

    // Signal loop()
    csi_data_ready = true;
}

// ---------------------------------------------------------------------------
// ESP-NOW receive callback — runs in WiFi task context, minimal work only
// ---------------------------------------------------------------------------

static volatile bool  turn_cmd_received = false;
static volatile bool  turn_is_mine      = false;
static volatile uint8_t turn_node_id    = 0xFF;

void on_espnow_recv(const uint8_t *mac, const uint8_t *data, int len) {
    if (data == NULL || len < 1) return;

    uint8_t msg_type = data[0];

    if (msg_type == MSG_TURN_CMD && len >= 2) {
        // TURN_CMD: [0x01, node_id]
        uint8_t target_id = data[1];
        turn_node_id = target_id;
        current_tx_id = target_id;

        if (target_id == NODE_ID) {
            // It's my turn to transmit
            turn_is_mine = true;
            is_my_turn = true;
        } else {
            // Another node's turn — I'm a receiver
            turn_is_mine = false;
            is_my_turn = false;
        }
        turn_cmd_received = true;
    }
    // CSI_TX packets (0x02) are received over-the-air and generate CSI
    // via the CSI callback — we don't need to parse them in ESP-NOW recv
}

// ---------------------------------------------------------------------------
// Build and send CSI_REPORT to coordinator
// ---------------------------------------------------------------------------

static void send_csi_report(const csi_frame_t *frame) {
    // Skip first_word_invalid bytes (same as S01)
    uint16_t offset = frame->first_word_invalid ? 4 : 0;
    if (frame->data_len <= offset) return;

    uint16_t valid_len = frame->data_len - offset;

    // ESP-NOW max payload is 250 bytes
    // Header: [type, tx_id, rx_id, rssi, len_hi, len_lo] = 6 bytes
    // Max CSI data: 250 - 6 = 244 bytes
    uint16_t csi_copy_len = valid_len;
    if (csi_copy_len > 244) {
        csi_copy_len = 244;
    }

    uint8_t report[250];
    report[0] = MSG_CSI_REPORT;        // 0x03
    report[1] = frame->tx_node_id;     // who was transmitting
    report[2] = (uint8_t)NODE_ID;      // who captured (me)
    report[3] = (uint8_t)frame->rssi;  // RSSI (signed, cast to uint8)
    report[4] = (uint8_t)(csi_copy_len >> 8);   // data_len high byte
    report[5] = (uint8_t)(csi_copy_len & 0xFF); // data_len low byte

    // Copy CSI data bytes (skipping first_word_invalid)
    memcpy(&report[6], &frame->data[offset], csi_copy_len);

    uint16_t total_len = 6 + csi_copy_len;

    // Unicast to coordinator
    esp_now_send(COORDINATOR_MAC, report, total_len);
}

// ---------------------------------------------------------------------------
// Broadcast CSI_TX stimulus packet
// ---------------------------------------------------------------------------

static void send_csi_tx_stimulus() {
    uint8_t pkt[8];
    pkt[0] = MSG_CSI_TX;                        // 0x02
    pkt[1] = (uint8_t)NODE_ID;                  // who is transmitting
    pkt[2] = (uint8_t)(tx_seq_counter >> 8);     // seq high byte
    pkt[3] = (uint8_t)(tx_seq_counter & 0xFF);   // seq low byte
    pkt[4] = 0;
    pkt[5] = 0;
    pkt[6] = 0;
    pkt[7] = 0;

    esp_now_send(BROADCAST_ADDR, pkt, sizeof(pkt));
    tx_seq_counter++;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);

    Serial.printf("=== Perimeter Node %c (ID %d) Starting ===\n",
                  NODE_LABELS[NODE_ID], NODE_ID);

    // Set own MAC from the node table
    memcpy(my_mac, NODE_MACS[NODE_ID], 6);

    // Initialize WiFi in STA mode — no AP connection
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    Serial.println("WiFi mode: STA (no connection)");

    // Set custom MAC address
    esp_err_t err;
    err = esp_wifi_set_mac(WIFI_IF_STA, my_mac);
    Serial.printf("Custom MAC set (%02X:%02X:%02X:%02X:%02X:%02X): %s\n",
                  my_mac[0], my_mac[1], my_mac[2],
                  my_mac[3], my_mac[4], my_mac[5],
                  err == ESP_OK ? "OK" : "FAIL");

    // Set WiFi channel
    err = esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    Serial.printf("WiFi channel %d: %s\n", WIFI_CHANNEL,
                  err == ESP_OK ? "OK" : "FAIL");

    // Enable promiscuous mode — required for CSI capture
    err = esp_wifi_set_promiscuous(true);
    Serial.printf("Promiscuous mode: %s\n",
                  err == ESP_OK ? "ENABLED" : "FAIL");

    // Configure CSI — LLTF only (same as S01 receiver)
    wifi_csi_config_t csi_config;
    csi_config.lltf_en           = true;
    csi_config.htltf_en          = false;
    csi_config.stbc_htltf2_en    = false;
    csi_config.ltf_merge_en      = true;
    csi_config.channel_filter_en = false;
    csi_config.manu_scale        = false;

    err = esp_wifi_set_csi_config(&csi_config);
    Serial.printf("CSI config (LLTF): %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Register CSI callback
    err = esp_wifi_set_csi_rx_cb(&csi_rx_callback, NULL);
    Serial.printf("CSI callback: %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Enable CSI collection
    err = esp_wifi_set_csi(true);
    Serial.printf("CSI collection: %s\n",
                  err == ESP_OK ? "ENABLED" : "FAIL");

    // Initialize ESP-NOW
    err = esp_now_init();
    Serial.printf("ESP-NOW init: %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Register ESP-NOW receive callback
    esp_now_register_recv_cb(on_espnow_recv);
    Serial.println("ESP-NOW recv callback: registered");

    // Add broadcast peer (for hearing TURN commands and sending CSI_TX)
    esp_now_peer_info_t bcast_peer;
    memset(&bcast_peer, 0, sizeof(bcast_peer));
    memcpy(bcast_peer.peer_addr, BROADCAST_ADDR, 6);
    bcast_peer.channel = WIFI_CHANNEL;
    bcast_peer.encrypt = false;

    err = esp_now_add_peer(&bcast_peer);
    Serial.printf("Broadcast peer: %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Add coordinator as unicast peer (for sending CSI_REPORT)
    esp_now_peer_info_t coord_peer;
    memset(&coord_peer, 0, sizeof(coord_peer));
    memcpy(coord_peer.peer_addr, COORDINATOR_MAC, 6);
    coord_peer.channel = WIFI_CHANNEL;
    coord_peer.encrypt = false;

    err = esp_now_add_peer(&coord_peer);
    Serial.printf("Coordinator peer (%02X:%02X:%02X:%02X:%02X:%02X): %s\n",
                  COORDINATOR_MAC[0], COORDINATOR_MAC[1], COORDINATOR_MAC[2],
                  COORDINATOR_MAC[3], COORDINATOR_MAC[4], COORDINATOR_MAC[5],
                  err == ESP_OK ? "OK" : "FAIL");

    Serial.printf("=== Perimeter Node %c Ready — waiting for TURN commands ===\n",
                  NODE_LABELS[NODE_ID]);
}

// ---------------------------------------------------------------------------
// Loop — handle TX turns and relay captured CSI to coordinator
// ---------------------------------------------------------------------------

void loop() {
    // ---- Handle TURN command ----
    if (turn_cmd_received) {
        turn_cmd_received = false;

        if (turn_is_mine) {
            // My turn: settle delay, then send TX stimulus burst
            delay(SETTLE_DELAY_MS);

            for (int i = 0; i < TX_BURST_COUNT; i++) {
                send_csi_tx_stimulus();
                if (i < TX_BURST_COUNT - 1) {
                    delay(TX_BURST_DELAY_MS);
                }
            }

            // Debug: log every 20th turn
            if (tx_seq_counter % 20 < TX_BURST_COUNT) {
                Serial.printf("TX turn: sent %d stimulus pkts, seq=%u\n",
                              TX_BURST_COUNT, tx_seq_counter);
            }
        }
        // If not my turn, we're now in RX mode — CSI callback will fire
    }

    // ---- Relay captured CSI to coordinator ----
    if (csi_data_ready) {
        send_csi_report(&csi_frame);

        // Debug: log occasionally
        static uint32_t report_count = 0;
        report_count++;
        if (report_count % 20 == 0) {
            Serial.printf("CSI relay: tx=%c rx=%c rssi=%d (report #%u)\n",
                          NODE_LABELS[csi_frame.tx_node_id],
                          NODE_LABELS[NODE_ID],
                          csi_frame.rssi,
                          report_count);
        }

        csi_data_ready = false;
    }
}
