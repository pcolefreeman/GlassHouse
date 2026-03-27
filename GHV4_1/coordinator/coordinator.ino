/*
 * Coordinator Firmware — Multi-Node Round-Robin CSI Network
 *
 * Master orchestrator that runs the round-robin state machine:
 *   1. Broadcasts TURN_CMD to designate which perimeter node transmits
 *   2. Receives CSI_REPORT packets from perimeter nodes via ESP-NOW
 *   3. Prints extended CSV lines to USB serial at 921600 baud
 *
 * The coordinator does NOT capture CSI itself — it receives pre-captured
 * CSI data from perimeter nodes via ESP-NOW unicast.
 *
 * Round-robin cycle (5 Hz = 200ms):
 *   Slot 0 (Node A) → Slot 1 (Node B) → Slot 2 (Node C) → Slot 3 (Node D)
 *   Each slot: broadcast TURN_CMD, wait up to 50ms for CSI_REPORTs,
 *   print received reports as CSV, advance to next slot.
 *
 * Serial CSV format (extended, for multi-link parsing):
 *   CSI_DATA,<seq>,<tx_node>,<rx_node>,<link_id>,<rssi>,<data_len>,<b0> <b1> ...
 *
 * Own MAC: 24:6F:28:AA:00:00
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
#define NUM_NODES         4
#define SLOT_TIMEOUT_MS   50       // max wait per slot (50ms × 4 = 200ms = 5 Hz)
#define MAX_REPORTS_PER_SLOT 3     // max CSI_REPORTs expected per slot
#define CSI_BUF_SIZE      244      // max CSI data bytes per report (250 - 6 header)

// Message type bytes — must match perimeter_node exactly
#define MSG_TURN_CMD    0x01
#define MSG_CSI_TX      0x02
#define MSG_CSI_REPORT  0x03

// ---------------------------------------------------------------------------
// MAC addresses — must match perimeter_node.ino exactly
// ---------------------------------------------------------------------------

// Coordinator own MAC: 24:6F:28:AA:00:00
static uint8_t COORDINATOR_MAC[6] = {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x00};

// Per-node custom MACs: A=01, B=02, C=03, D=04
static const uint8_t NODE_MACS[NUM_NODES][6] = {
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x01},  // Node A (ID 0)
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x02},  // Node B (ID 1)
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x03},  // Node C (ID 2)
    {0x24, 0x6F, 0x28, 0xAA, 0x00, 0x04},  // Node D (ID 3)
};

// Broadcast address for sending TURN commands
static const uint8_t BROADCAST_ADDR[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Node labels for CSV output
static const char NODE_LABELS[NUM_NODES] = {'A', 'B', 'C', 'D'};

// ---------------------------------------------------------------------------
// CSI Report buffer — written by ESP-NOW recv callback, read by loop()
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t  tx_node_id;      // who was transmitting
    uint8_t  rx_node_id;      // who captured the CSI
    int8_t   rssi;            // signal strength
    uint16_t data_len;        // CSI data byte count
    uint8_t  data[CSI_BUF_SIZE];  // CSI subcarrier data
} csi_report_t;

// Ring buffer for incoming CSI reports (callback writes, loop reads)
#define REPORT_RING_SIZE  12   // 3 per slot × 4 slots = 12 max
static volatile csi_report_t report_ring[REPORT_RING_SIZE];
static volatile uint8_t ring_write_idx = 0;
static volatile uint8_t ring_read_idx  = 0;

// ---------------------------------------------------------------------------
// ESP-NOW receive callback — runs in WiFi task, ZERO Serial I/O
// ---------------------------------------------------------------------------

void on_espnow_recv(const uint8_t *mac, const uint8_t *data, int len) {
    if (data == NULL || len < 1) return;

    uint8_t msg_type = data[0];

    if (msg_type == MSG_CSI_REPORT && len >= 6) {
        // CSI_REPORT: [0x03, tx_id, rx_id, rssi, len_hi, len_lo, csi...]
        // Parse header
        uint8_t  tx_id   = data[1];
        uint8_t  rx_id   = data[2];
        int8_t   rssi    = (int8_t)data[3];
        uint16_t csi_len = ((uint16_t)data[4] << 8) | data[5];

        // Validate
        if (tx_id >= NUM_NODES || rx_id >= NUM_NODES) return;
        uint16_t available = (uint16_t)(len - 6);
        if (csi_len > available) csi_len = available;
        if (csi_len > CSI_BUF_SIZE) csi_len = CSI_BUF_SIZE;

        // Check ring buffer space (if full, drop this report)
        uint8_t next_write = (ring_write_idx + 1) % REPORT_RING_SIZE;
        if (next_write == ring_read_idx) {
            return;  // ring full — drop oldest-unread is too risky, just drop new
        }

        // Write to ring buffer
        volatile csi_report_t *slot = &report_ring[ring_write_idx];
        slot->tx_node_id = tx_id;
        slot->rx_node_id = rx_id;
        slot->rssi       = rssi;
        slot->data_len   = csi_len;
        memcpy((void *)slot->data, &data[6], csi_len);

        // Advance write index (must be last — acts as memory barrier intent)
        ring_write_idx = next_write;
    }
    // Ignore other message types (TURN_CMD echoes, CSI_TX)
}

// ---------------------------------------------------------------------------
// CSV Output Helpers
// ---------------------------------------------------------------------------

static uint32_t csv_seq_counter = 0;

// Build alphabetically ordered link ID from tx/rx node IDs
// e.g., tx=B(1), rx=A(0) → "AB"  (A < B alphabetically)
static void get_link_id(uint8_t tx_id, uint8_t rx_id, char *out) {
    char tx_label = NODE_LABELS[tx_id];
    char rx_label = NODE_LABELS[rx_id];

    if (tx_label <= rx_label) {
        out[0] = tx_label;
        out[1] = rx_label;
    } else {
        out[0] = rx_label;
        out[1] = tx_label;
    }
    out[2] = '\0';
}

// Print one CSI report as extended CSV line
static void print_csv_report(const csi_report_t *report) {
    char link_id[3];
    get_link_id(report->tx_node_id, report->rx_node_id, link_id);

    // CSI_DATA,seq,tx_node,rx_node,link_id,rssi,data_len,bytes...
    Serial.printf("CSI_DATA,%u,%c,%c,%s,%d,%u,",
                  csv_seq_counter,
                  NODE_LABELS[report->tx_node_id],
                  NODE_LABELS[report->rx_node_id],
                  link_id,
                  report->rssi,
                  report->data_len);

    // Print CSI bytes as space-separated signed int8 values
    for (uint16_t i = 0; i < report->data_len; i++) {
        if (i > 0) {
            Serial.print(' ');
        }
        Serial.print((int8_t)report->data[i]);
    }
    Serial.println();

    csv_seq_counter++;
}

// ---------------------------------------------------------------------------
// Round-robin state machine
// ---------------------------------------------------------------------------

static uint8_t       current_slot     = 0;     // 0=A, 1=B, 2=C, 3=D
static unsigned long  slot_start_time  = 0;
static bool           slot_turn_sent   = false;
static uint32_t       cycle_counter    = 0;

// Send TURN_CMD broadcast to designate which node transmits
static void send_turn_cmd(uint8_t node_id) {
    uint8_t cmd[2];
    cmd[0] = MSG_TURN_CMD;   // 0x01
    cmd[1] = node_id;

    esp_now_send(BROADCAST_ADDR, cmd, sizeof(cmd));
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);  // let serial stabilize

    Serial.println("=== Coordinator Starting ===");

    // Initialize WiFi in STA mode — no AP connection
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    Serial.println("WiFi mode: STA (no connection)");

    // Set custom MAC address: 24:6F:28:AA:00:00
    esp_err_t err;
    err = esp_wifi_set_mac(WIFI_IF_STA, COORDINATOR_MAC);
    Serial.printf("Custom MAC set (%02X:%02X:%02X:%02X:%02X:%02X): %s\n",
                  COORDINATOR_MAC[0], COORDINATOR_MAC[1], COORDINATOR_MAC[2],
                  COORDINATOR_MAC[3], COORDINATOR_MAC[4], COORDINATOR_MAC[5],
                  err == ESP_OK ? "OK" : "FAIL");

    // Set WiFi channel — must match all perimeter nodes
    err = esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    Serial.printf("WiFi channel %d: %s\n", WIFI_CHANNEL,
                  err == ESP_OK ? "OK" : "FAIL");

    // Initialize ESP-NOW
    err = esp_now_init();
    Serial.printf("ESP-NOW init: %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Register ESP-NOW receive callback
    esp_now_register_recv_cb(on_espnow_recv);
    Serial.println("ESP-NOW recv callback: registered");

    // Add broadcast peer (for sending TURN commands)
    esp_now_peer_info_t bcast_peer;
    memset(&bcast_peer, 0, sizeof(bcast_peer));
    memcpy(bcast_peer.peer_addr, BROADCAST_ADDR, 6);
    bcast_peer.channel = WIFI_CHANNEL;
    bcast_peer.encrypt = false;

    err = esp_now_add_peer(&bcast_peer);
    Serial.printf("Broadcast peer: %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Add 4 unicast peers (perimeter nodes) — good practice for receiving
    for (int i = 0; i < NUM_NODES; i++) {
        esp_now_peer_info_t node_peer;
        memset(&node_peer, 0, sizeof(node_peer));
        memcpy(node_peer.peer_addr, NODE_MACS[i], 6);
        node_peer.channel = WIFI_CHANNEL;
        node_peer.encrypt = false;

        err = esp_now_add_peer(&node_peer);
        Serial.printf("Node %c peer (%02X:%02X:%02X:%02X:%02X:%02X): %s\n",
                      NODE_LABELS[i],
                      NODE_MACS[i][0], NODE_MACS[i][1], NODE_MACS[i][2],
                      NODE_MACS[i][3], NODE_MACS[i][4], NODE_MACS[i][5],
                      err == ESP_OK ? "OK" : "FAIL");
    }

    // Print configuration summary
    Serial.printf("Round-robin: %d nodes, %d ms/slot, %d Hz cycle\n",
                  NUM_NODES, SLOT_TIMEOUT_MS, 1000 / (NUM_NODES * SLOT_TIMEOUT_MS));
    Serial.printf("Serial baud: %d\n", SERIAL_BAUD);

    Serial.println("=== Coordinator Ready — starting round-robin ===");

    // Start first slot
    slot_start_time = millis();
    slot_turn_sent = false;
}

// ---------------------------------------------------------------------------
// Loop — round-robin state machine + CSV output
// ---------------------------------------------------------------------------

void loop() {
    unsigned long now = millis();

    // ---- Phase 1: Send TURN command at start of slot ----
    if (!slot_turn_sent) {
        send_turn_cmd(current_slot);
        slot_turn_sent = true;
        slot_start_time = now;
    }

    // ---- Phase 2: Drain ring buffer — print any buffered CSI reports ----
    while (ring_read_idx != ring_write_idx) {
        // Copy from volatile ring buffer to local struct
        csi_report_t local_report;
        volatile csi_report_t *rp = &report_ring[ring_read_idx];

        local_report.tx_node_id = rp->tx_node_id;
        local_report.rx_node_id = rp->rx_node_id;
        local_report.rssi       = rp->rssi;
        local_report.data_len   = rp->data_len;
        memcpy(local_report.data, (const void *)rp->data, rp->data_len);

        // Advance read index
        ring_read_idx = (ring_read_idx + 1) % REPORT_RING_SIZE;

        // Print CSV line
        print_csv_report(&local_report);
    }

    // ---- Phase 3: Check slot timeout → advance to next slot ----
    if (now - slot_start_time >= SLOT_TIMEOUT_MS) {
        // Advance to next node
        current_slot = (current_slot + 1) % NUM_NODES;
        slot_turn_sent = false;

        // Log cycle completion (every full cycle through all 4 nodes)
        if (current_slot == 0) {
            cycle_counter++;
            // Print cycle heartbeat every 25 cycles (~5 seconds at 5 Hz)
            if (cycle_counter % 25 == 0) {
                Serial.printf("# cycle=%u seq=%u\n", cycle_counter, csv_seq_counter);
            }
        }
    }
}
