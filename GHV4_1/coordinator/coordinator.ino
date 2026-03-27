/*
 * Coordinator Firmware — Multi-Node Round-Robin CSI Network
 *
 * Master orchestrator that runs the round-robin state machine:
 *   1. Broadcasts TURN_CMD to designate which perimeter node transmits
 *   2. Receives CSI_REPORT packets from perimeter nodes via UDP
 *   3. Prints extended CSV lines to USB serial at 921600 baud
 *
 * The coordinator does NOT capture CSI itself — it receives pre-captured
 * CSI data from perimeter nodes via UDP unicast/broadcast.
 *
 * Transport: WiFi SoftAP + UDP (replaced ESP-NOW in M002)
 *   - Coordinator runs as AP (SSID: CSI_NET, channel 11)
 *   - Perimeter nodes connect as STA clients
 *   - Turn commands broadcast on UDP port 4211
 *   - CSI reports received on UDP port 4210
 *
 * Round-robin cycle (5 Hz = 200ms):
 *   Slot 0 (Node A) → Slot 1 (Node B) → Slot 2 (Node C) → Slot 3 (Node D)
 *   Each slot: broadcast TURN_CMD, wait up to 50ms for CSI_REPORTs,
 *   print received reports as CSV, advance to next slot.
 *
 * Serial CSV format (extended, for multi-link parsing):
 *   CSI_DATA,<seq>,<tx_node>,<rx_node>,<link_id>,<rssi>,<data_len>,<b0> <b1> ...
 *
 * Coordinator factory MAC (Board 5): 68:FE:71:90:66:A8
 * Hardware: ESP32-WROOM
 * Board Package: arduino-esp32 (v2.x or v3.x)
 */

#include <WiFi.h>
#include <WiFiUdp.h>
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

// SoftAP configuration
#define AP_SSID          "CSI_NET"
#define AP_PASSWORD      "csi12345"
#define AP_MAX_CONN      4

// UDP configuration
#define UDP_PORT          4210     // CSI reports: perimeter → coordinator
#define UDP_BCAST_PORT    4211     // Turn commands: coordinator → all nodes

// ---------------------------------------------------------------------------
// MAC addresses — factory MACs, must match perimeter_node.ino exactly
// ---------------------------------------------------------------------------

// Coordinator factory MAC (Board 5): 68:FE:71:90:66:A8
// No custom MAC override — using real hardware MACs
static const uint8_t COORDINATOR_MAC[6] = {0x68, 0xFE, 0x71, 0x90, 0x66, 0xA8};

// Per-node factory MACs — must match perimeter_node.ino exactly
static const uint8_t NODE_MACS[NUM_NODES][6] = {
    {0x68, 0xFE, 0x71, 0x90, 0x68, 0x14},  // Node A (ID 0) — Board 2, Top-Left
    {0x68, 0xFE, 0x71, 0x90, 0x6B, 0x90},  // Node B (ID 1) — Board 3, Top-Right
    {0x68, 0xFE, 0x71, 0x90, 0x60, 0xA0},  // Node C (ID 2) — Board 1, Bottom-Left
    {0x20, 0xE7, 0xC8, 0xEC, 0xF5, 0xDC},  // Node D (ID 3) — Board 4, Bottom-Right
};

// Node labels for CSV output
static const char NODE_LABELS[NUM_NODES] = {'A', 'B', 'C', 'D'};

// ---------------------------------------------------------------------------
// UDP instances
// ---------------------------------------------------------------------------

WiFiUDP udpReport;    // Listens on UDP_PORT (4210) for incoming CSI reports
WiFiUDP udpCmd;       // Sends turn commands to broadcast on UDP_BCAST_PORT (4211)

// SoftAP broadcast address
static const IPAddress broadcastIP(192, 168, 4, 255);

// ---------------------------------------------------------------------------
// CSI Report buffer — written by UDP receiver in loop(), read by CSV printer
// ---------------------------------------------------------------------------

typedef struct {
    uint8_t  tx_node_id;      // who was transmitting
    uint8_t  rx_node_id;      // who captured the CSI
    int8_t   rssi;            // signal strength
    uint16_t data_len;        // CSI data byte count
    uint8_t  data[CSI_BUF_SIZE];  // CSI subcarrier data
} csi_report_t;

// Ring buffer for incoming CSI reports
#define REPORT_RING_SIZE  12   // 3 per slot × 4 slots = 12 max
static csi_report_t report_ring[REPORT_RING_SIZE];
static uint8_t ring_write_idx = 0;
static uint8_t ring_read_idx  = 0;

// ---------------------------------------------------------------------------
// UDP CSI Report receiver — called from loop(), parses binary report packets
// ---------------------------------------------------------------------------

// Temporary packet buffer for UDP reads
static uint8_t udp_pkt_buf[6 + CSI_BUF_SIZE];  // header (6) + max CSI data

static void receive_udp_reports() {
    int packetSize;
    while ((packetSize = udpReport.parsePacket()) > 0) {
        if (packetSize < 6 || packetSize > (int)sizeof(udp_pkt_buf)) {
            // Flush invalid packet
            while (udpReport.available()) udpReport.read();
            continue;
        }

        int bytesRead = udpReport.read(udp_pkt_buf, packetSize);
        if (bytesRead < 6) continue;

        uint8_t msg_type = udp_pkt_buf[0];

        if (msg_type == MSG_CSI_REPORT) {
            // CSI_REPORT: [0x03, tx_id, rx_id, rssi, len_hi, len_lo, csi...]
            uint8_t  tx_id   = udp_pkt_buf[1];
            uint8_t  rx_id   = udp_pkt_buf[2];
            int8_t   rssi    = (int8_t)udp_pkt_buf[3];
            uint16_t csi_len = ((uint16_t)udp_pkt_buf[4] << 8) | udp_pkt_buf[5];

            // Validate
            if (tx_id >= NUM_NODES || rx_id >= NUM_NODES) continue;
            uint16_t available = (uint16_t)(bytesRead - 6);
            if (csi_len > available) csi_len = available;
            if (csi_len > CSI_BUF_SIZE) csi_len = CSI_BUF_SIZE;

            // Check ring buffer space (if full, drop this report)
            uint8_t next_write = (ring_write_idx + 1) % REPORT_RING_SIZE;
            if (next_write == ring_read_idx) {
                continue;  // ring full — drop new report
            }

            // Write to ring buffer
            csi_report_t *slot = &report_ring[ring_write_idx];
            slot->tx_node_id = tx_id;
            slot->rx_node_id = rx_id;
            slot->rssi       = rssi;
            slot->data_len   = csi_len;
            memcpy(slot->data, &udp_pkt_buf[6], csi_len);

            // Advance write index
            ring_write_idx = next_write;
        }
        // Ignore other message types
    }
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

// Send TURN_CMD broadcast via UDP to designate which node transmits
static void send_turn_cmd(uint8_t node_id) {
    uint8_t cmd[2];
    cmd[0] = MSG_TURN_CMD;   // 0x01
    cmd[1] = node_id;

    udpCmd.beginPacket(broadcastIP, UDP_BCAST_PORT);
    udpCmd.write(cmd, sizeof(cmd));
    udpCmd.endPacket();
}

// ---------------------------------------------------------------------------
// Node registration — MAC-to-ID tracking and network readiness
// ---------------------------------------------------------------------------

// Bitmask tracking which nodes are currently connected (bit 0=A, 1=B, 2=C, 3=D)
static uint8_t nodes_connected_mask = 0;

// Compare a STA MAC against NODE_MACS[0..3], return index (0-3) or -1 if unknown
static int match_sta_mac(const uint8_t *mac) {
    for (int i = 0; i < NUM_NODES; i++) {
        if (memcmp(mac, NODE_MACS[i], 6) == 0) {
            return i;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// WiFi Event Handler — node registration with MAC-to-ID tracking
// ---------------------------------------------------------------------------

void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
    switch (event) {
        case ARDUINO_EVENT_WIFI_AP_STACONNECTED: {
            const uint8_t *mac = info.wifi_ap_staconnected.mac;
            int idx = match_sta_mac(mac);
            if (idx >= 0) {
                nodes_connected_mask |= (1 << idx);
                Serial.printf("# Node %c connected (MAC: %02X:%02X:%02X:%02X:%02X:%02X)\n",
                              NODE_LABELS[idx],
                              mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
                if (nodes_connected_mask == 0x0F) {
                    Serial.println("# All 4 nodes connected — network ready");
                }
            } else {
                Serial.printf("# Unknown STA connected (MAC: %02X:%02X:%02X:%02X:%02X:%02X)\n",
                              mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
            }
            break;
        }
        case ARDUINO_EVENT_WIFI_AP_STADISCONNECTED: {
            const uint8_t *mac = info.wifi_ap_stadisconnected.mac;
            int idx = match_sta_mac(mac);
            if (idx >= 0) {
                nodes_connected_mask &= ~(1 << idx);
                Serial.printf("# Node %c disconnected (MAC: %02X:%02X:%02X:%02X:%02X:%02X)\n",
                              NODE_LABELS[idx],
                              mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
            } else {
                Serial.printf("# Unknown STA disconnected (MAC: %02X:%02X:%02X:%02X:%02X:%02X)\n",
                              mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
            }
            break;
        }
        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);  // let serial stabilize

    Serial.println("=== Coordinator Starting (SoftAP + UDP) ===");

    // Register WiFi event handler for STA connect/disconnect
    WiFi.onEvent(onWiFiEvent);

    // Configure SoftAP: SSID, password, channel, hidden=false, max_connection=4
    WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL, 0, AP_MAX_CONN);
    delay(100);  // let AP stabilize

    Serial.printf("SoftAP SSID: %s\n", AP_SSID);
    Serial.printf("SoftAP IP: %s\n", WiFi.softAPIP().toString().c_str());
    Serial.printf("SoftAP Channel: %d\n", WIFI_CHANNEL);

    // Print coordinator MAC (factory)
    Serial.printf("Coordinator MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  COORDINATOR_MAC[0], COORDINATOR_MAC[1], COORDINATOR_MAC[2],
                  COORDINATOR_MAC[3], COORDINATOR_MAC[4], COORDINATOR_MAC[5]);

    // Initialize UDP sockets
    udpReport.begin(UDP_PORT);
    Serial.printf("UDP report listener on port %d: OK\n", UDP_PORT);

    // The command socket doesn't need to listen; it just sends broadcasts
    udpCmd.begin(UDP_BCAST_PORT);
    Serial.printf("UDP command sender on port %d: OK\n", UDP_BCAST_PORT);

    // Print registered node MACs for reference
    for (int i = 0; i < NUM_NODES; i++) {
        Serial.printf("Node %c MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                      NODE_LABELS[i],
                      NODE_MACS[i][0], NODE_MACS[i][1], NODE_MACS[i][2],
                      NODE_MACS[i][3], NODE_MACS[i][4], NODE_MACS[i][5]);
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

    // ---- Phase 2: Receive UDP reports into ring buffer ----
    receive_udp_reports();

    // ---- Phase 3: Drain ring buffer — print any buffered CSI reports ----
    while (ring_read_idx != ring_write_idx) {
        // Copy from ring buffer to local struct
        csi_report_t local_report;
        csi_report_t *rp = &report_ring[ring_read_idx];

        local_report.tx_node_id = rp->tx_node_id;
        local_report.rx_node_id = rp->rx_node_id;
        local_report.rssi       = rp->rssi;
        local_report.data_len   = rp->data_len;
        memcpy(local_report.data, rp->data, rp->data_len);

        // Advance read index
        ring_read_idx = (ring_read_idx + 1) % REPORT_RING_SIZE;

        // Print CSV line
        print_csv_report(&local_report);
    }

    // ---- Phase 4: Check slot timeout → advance to next slot ----
    if (now - slot_start_time >= SLOT_TIMEOUT_MS) {
        // Advance to next node
        current_slot = (current_slot + 1) % NUM_NODES;
        slot_turn_sent = false;

        // Log cycle completion (every full cycle through all 4 nodes)
        if (current_slot == 0) {
            cycle_counter++;
            // Print cycle heartbeat every 25 cycles (~5 seconds at 5 Hz)
            if (cycle_counter % 25 == 0) {
                Serial.printf("# cycle=%u seq=%u nodes=%u/4\n",
                              cycle_counter, csv_seq_counter,
                              (unsigned)__builtin_popcount(nodes_connected_mask));
            }
        }
    }
}
