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
 *     via UDP unicast (CSI_REPORT).
 *
 * Transport: WiFi STA + UDP (replaced ESP-NOW in M002)
 *   - Coordinator runs as AP (SSID: CSI_NET, channel 11)
 *   - Perimeter nodes connect as STA clients
 *   - Turn commands received on UDP broadcast port 4211
 *   - CSI reports sent to coordinator at 192.168.4.1:4210
 *   - CSI_TX stimulus broadcast to 192.168.4.255:4211
 *
 * Protocol:
 *   1. Coordinator broadcasts TURN_CMD [0x01, node_id] via UDP to designate
 *      which node transmits next.
 *   2. Designated node broadcasts CSI_TX [0x02, node_id, seq_hi, seq_lo, ...]
 *      via UDP after a brief settling delay.
 *   3. Other nodes capture CSI from that stimulus, then unicast
 *      CSI_REPORT [0x03, tx_id, rx_id, rssi, len_hi, len_lo, csi...]
 *      to the coordinator via UDP.
 *
 * Flash instructions:
 *   Change NODE_ID below before flashing each board:
 *     0 = Node A, 1 = Node B, 2 = Node C, 3 = Node D
 *
 * Hardware: ESP32-WROOM
 * Board Package: arduino-esp32 (v2.x or v3.x)
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "esp_err.h"

// ===========================================================================
// *** CHANGE THIS BEFORE FLASHING EACH BOARD ***
// 0 = TL, 1 = TR, 2 = BL, 3 = BR
// ===========================================================================
#define NODE_ID  1

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
// Network configuration — must match coordinator exactly
// ---------------------------------------------------------------------------

#define CSI_NET_SSID     "CSI_NET"
#define CSI_NET_PASS     "csi12345"

#define UDP_REPORT_PORT  4210        // CSI reports: perimeter → coordinator
#define UDP_CMD_PORT     4211        // Turn commands: coordinator → all nodes
                                     // Also used for CSI_TX stimulus broadcast

// Coordinator AP IP (always 192.168.4.1 for ESP32 SoftAP)
static const IPAddress COORDINATOR_IP(192, 168, 4, 1);

// Subnet broadcast address for UDP broadcasts
static const IPAddress BROADCAST_IP(192, 168, 4, 255);

// WiFi connection parameters
#define WIFI_CONNECT_TIMEOUT_MS   10000   // 10 seconds initial connect timeout
#define WIFI_RECONNECT_DELAY_MS   2000    // 2 seconds between reconnect attempts

// ---------------------------------------------------------------------------
// MAC addresses — factory MACs, must match coordinator.ino exactly
// ---------------------------------------------------------------------------

// Per-node factory MACs — must match coordinator.ino exactly
static const uint8_t NODE_MACS[NUM_NODES][6] = {
    {0x68, 0xFE, 0x71, 0x90, 0x68, 0x14},  // Node A (ID 0) — Board 2, Top-Left
    {0x68, 0xFE, 0x71, 0x90, 0x6B, 0x90},  // Node B (ID 1) — Board 3, Top-Right
    {0x68, 0xFE, 0x71, 0x90, 0x60, 0xA0},  // Node C (ID 2) — Board 1, Bottom-Left
    {0x20, 0xE7, 0xC8, 0xEC, 0xF5, 0xDC},  // Node D (ID 3) — Board 4, Bottom-Right
};

// ---------------------------------------------------------------------------
// UDP instances
// ---------------------------------------------------------------------------

WiFiUDP reportUdp;    // Sends CSI reports to coordinator on UDP_REPORT_PORT
WiFiUDP cmdUdp;       // Listens for turn commands on UDP_CMD_PORT

// ---------------------------------------------------------------------------
// Node identity (derived from NODE_ID at compile time)
// ---------------------------------------------------------------------------

static uint8_t my_mac[6];
static const char NODE_LABELS[NUM_NODES] = {'A', 'B', 'C', 'D'};

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

static volatile bool   is_my_turn      = false;   // set by turn command handler
static volatile uint8_t current_tx_id  = 0xFF;    // who is currently transmitting
static uint16_t        tx_seq_counter  = 0;        // sequence for CSI_TX packets

// WiFi reconnect state
static unsigned long   last_reconnect_attempt = 0;
static bool            wifi_connected         = false;

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
// Turn command state (set by UDP handler in loop())
// ---------------------------------------------------------------------------

static bool     turn_cmd_received = false;
static bool     turn_is_mine      = false;
static uint8_t  turn_node_id      = 0xFF;

// ---------------------------------------------------------------------------
// Check for turn commands via UDP on port 4211
// ---------------------------------------------------------------------------

static void check_turn_commands() {
    int packetSize = cmdUdp.parsePacket();
    if (packetSize < 2) return;

    uint8_t buf[8];
    int bytesRead = cmdUdp.read(buf, sizeof(buf));
    if (bytesRead < 2) return;

    uint8_t msg_type = buf[0];

    if (msg_type == MSG_TURN_CMD) {
        // TURN_CMD: [0x01, node_id]
        uint8_t target_id = buf[1];
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
}

// ---------------------------------------------------------------------------
// Build and send CSI_REPORT to coordinator via UDP
// ---------------------------------------------------------------------------

static void send_csi_report(const csi_frame_t *frame) {
    // Skip first_word_invalid bytes (same as S01)
    uint16_t offset = frame->first_word_invalid ? 4 : 0;
    if (frame->data_len <= offset) return;

    uint16_t valid_len = frame->data_len - offset;

    // UDP packet: header (6 bytes) + CSI data
    // Header: [type, tx_id, rx_id, rssi, len_hi, len_lo]
    // Max CSI data: 244 bytes (same limit as before for consistency)
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

    // Unicast to coordinator via UDP
    reportUdp.beginPacket(COORDINATOR_IP, UDP_REPORT_PORT);
    reportUdp.write(report, total_len);
    reportUdp.endPacket();
}

// ---------------------------------------------------------------------------
// Broadcast CSI_TX stimulus packet via UDP
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

    // Broadcast on the subnet so other nodes' promiscuous CSI capture sees it
    reportUdp.beginPacket(BROADCAST_IP, UDP_CMD_PORT);
    reportUdp.write(pkt, sizeof(pkt));
    reportUdp.endPacket();

    tx_seq_counter++;
}

// ---------------------------------------------------------------------------
// WiFi connection management
// ---------------------------------------------------------------------------

static void wifi_connect() {
    Serial.printf("Connecting to AP: %s ...\n", CSI_NET_SSID);
    WiFi.begin(CSI_NET_SSID, CSI_NET_PASS);

    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - start > WIFI_CONNECT_TIMEOUT_MS) {
            Serial.println("WiFi connect timeout — will retry in loop");
            return;
        }
        delay(250);
        Serial.print(".");
    }
    Serial.println();
    Serial.printf("Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    wifi_connected = true;

    // Start UDP listeners after WiFi is connected
    cmdUdp.begin(UDP_CMD_PORT);
    Serial.printf("UDP command listener on port %d: OK\n", UDP_CMD_PORT);

    // Report socket — used for sending, but bind it too
    reportUdp.begin(0);  // ephemeral port for sending
    Serial.println("UDP report sender: OK");
}

static void wifi_check_reconnect() {
    if (WiFi.status() == WL_CONNECTED) {
        if (!wifi_connected) {
            // Reconnected
            wifi_connected = true;
            Serial.printf("Reconnected! IP: %s\n", WiFi.localIP().toString().c_str());

            // Re-start UDP listeners
            cmdUdp.begin(UDP_CMD_PORT);
            reportUdp.begin(0);
        }
        return;
    }

    // Not connected — attempt reconnect with backoff
    wifi_connected = false;
    unsigned long now = millis();
    if (now - last_reconnect_attempt < WIFI_RECONNECT_DELAY_MS) {
        return;  // wait before retrying
    }
    last_reconnect_attempt = now;

    Serial.println("WiFi disconnected — attempting reconnect...");
    WiFi.disconnect();
    WiFi.begin(CSI_NET_SSID, CSI_NET_PASS);
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);

    Serial.printf("=== Perimeter Node %c (ID %d) Starting (STA + UDP) ===\n",
                  NODE_LABELS[NODE_ID], NODE_ID);

    // Set own MAC from the node table (for identity reference)
    memcpy(my_mac, NODE_MACS[NODE_ID], 6);

    // Initialize WiFi in STA mode and connect to coordinator's AP
    WiFi.mode(WIFI_STA);

    // Print factory MAC
    Serial.printf("Node %c MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  NODE_LABELS[NODE_ID],
                  my_mac[0], my_mac[1], my_mac[2],
                  my_mac[3], my_mac[4], my_mac[5]);

    // Connect to coordinator's SoftAP
    wifi_connect();

    // Set WiFi channel explicitly (matches coordinator AP channel)
    esp_err_t err;
    err = esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    Serial.printf("WiFi channel %d: %s\n", WIFI_CHANNEL,
                  err == ESP_OK ? "OK" : "FAIL");

    // Enable promiscuous mode — required for CSI capture
    err = esp_wifi_set_promiscuous(true);
    Serial.printf("Promiscuous mode: %s\n",
                  err == ESP_OK ? "ENABLED" : "FAIL");

    // Configure CSI — LLTF + HT-LTF for better subcarrier resolution
    wifi_csi_config_t csi_config;
    csi_config.lltf_en           = true;
    csi_config.htltf_en          = true;
    csi_config.stbc_htltf2_en    = true;
    csi_config.ltf_merge_en      = false;
    csi_config.channel_filter_en = false;
    csi_config.manu_scale        = false;

    err = esp_wifi_set_csi_config(&csi_config);
    Serial.printf("CSI config (LLTF+HT-LTF): %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Register CSI callback
    err = esp_wifi_set_csi_rx_cb(&csi_rx_callback, NULL);
    Serial.printf("CSI callback: %s\n",
                  err == ESP_OK ? "OK" : "FAIL");

    // Enable CSI collection
    err = esp_wifi_set_csi(true);
    Serial.printf("CSI collection: %s\n",
                  err == ESP_OK ? "ENABLED" : "FAIL");

    Serial.printf("=== Perimeter Node %c Ready — waiting for TURN commands ===\n",
                  NODE_LABELS[NODE_ID]);
}

// ---------------------------------------------------------------------------
// Loop — handle TX turns and relay captured CSI to coordinator
// ---------------------------------------------------------------------------

void loop() {
    // ---- WiFi reconnect check ----
    wifi_check_reconnect();

    // Don't process commands or send reports if not connected
    if (!wifi_connected) {
        delay(100);
        return;
    }

    // ---- Check for TURN commands via UDP ----
    check_turn_commands();

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
