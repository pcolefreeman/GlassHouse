#include "esp_wifi.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include "GHV2Protocol.h"

// ── CSI ring buffer ────────────────────────────────────────────────────────────
#define RING_SIZE 2

struct CsiEntry {
    uint8_t  bytes[SHOUTER_CSI_MAX];
    uint16_t len;
    uint32_t rx_timestamp_ms;
    int8_t   rssi;
    int8_t   noise_floor;
};

static CsiEntry           ring[RING_SIZE];
static volatile int       ring_write = 0;
static volatile int       ring_count = 0;
static portMUX_TYPE       ring_mux   = portMUX_INITIALIZER_UNLOCKED;

// Called by WiFi driver — keep short, no heap allocation
// IMPORTANT: millis() is NOT IRAM-safe and must NOT be called here.
//            Use esp_timer_get_time() (returns microseconds) and convert to ms.
void IRAM_ATTR shouter_csi_cb(void* ctx, wifi_csi_info_t* info) {
    if (!info || !info->buf || info->len <= 0) return;
    uint16_t copy_len = (info->len <= SHOUTER_CSI_MAX) ? info->len : SHOUTER_CSI_MAX;
    portENTER_CRITICAL_ISR(&ring_mux);
    int idx = ring_write % RING_SIZE;
    memcpy(ring[idx].bytes, info->buf, copy_len);
    ring[idx].len             = copy_len;
    ring[idx].rx_timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);  // µs → ms, IRAM-safe
    ring[idx].rssi            = info->rx_ctrl.rssi;
    ring[idx].noise_floor     = info->rx_ctrl.noise_floor;
    ring_write++;
    if (ring_count < RING_SIZE) ring_count++;
    portEXIT_CRITICAL_ISR(&ring_mux);
}

// Returns most-recently captured entry. Returns false if ring is empty.
// NOTE: Do NOT use (ring_write - 1 + RING_SIZE) % RING_SIZE — that formula wraps
//       incorrectly once ring_write > RING_SIZE. Use (ring_write - 1) % RING_SIZE.
bool get_latest_csi(CsiEntry* out) {
    portENTER_CRITICAL(&ring_mux);
    if (ring_count == 0) { portEXIT_CRITICAL(&ring_mux); return false; }
    int idx = (ring_write - 1) % RING_SIZE;
    memcpy(out, &ring[idx], sizeof(CsiEntry));
    portEXIT_CRITICAL(&ring_mux);
    return true;
}

// ── Per-device configuration — change SHOUTER_ID before flashing each board ──
#define SHOUTER_ID    1          // 1, 2, 3, or 4 — change before flashing each board
static_assert(SHOUTER_ID >= 1 && SHOUTER_ID <= 4, "SHOUTER_ID must be 1, 2, 3, or 4");
#define SSID          "CSI_PRIVATE_AP"
// NOTE: AP is open (no WPA2 password). WiFi.softAP(SSID, nullptr, CHANNEL) on listener
//       side explicitly passes nullptr for the password to keep the embedded mesh simple.
//       Do NOT deploy on an untrusted network.
#define LISTENER_IP   "192.168.4.1"
#define LISTENER_PORT 3333
#define SHOUTER_PORT  3334

WiFiUDP udp;

void send_hello() {
    hello_pkt_t pkt;
    memset(&pkt, 0, sizeof(pkt));
    pkt.magic[0]   = HELLO_MAGIC_0;
    pkt.magic[1]   = HELLO_MAGIC_1;
    pkt.ver        = 1;
    pkt.shouter_id = SHOUTER_ID;
    WiFi.macAddress(pkt.src_mac);   // fills 6-byte MAC directly
    udp.beginPacket(LISTENER_IP, LISTENER_PORT);
    udp.write((uint8_t*)&pkt, sizeof(pkt));
    udp.endPacket();
    Serial.println("[SHT] HELLO sent");
}

void connect_and_register() {
    WiFi.mode(WIFI_STA);
    WiFi.begin(SSID);
    Serial.print("[SHT] Connecting");
    unsigned long t = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - t > 15000) {
            Serial.println("\n[SHT] FATAL: WiFi connect timeout — check AP is up");
            while (1) delay(1000);
        }
        delay(250);
        Serial.print(".");
    }
    Serial.printf("\n[SHT] Connected, IP=%s\n", WiFi.localIP().toString().c_str());
    send_hello();
}

void setup() {
    Serial.begin(921600);
    udp.begin(SHOUTER_PORT);   // MUST be before connect_and_register() → send_hello()

    wifi_csi_config_t cfg = {};
    cfg.lltf_en           = true;
    cfg.htltf_en          = true;
    cfg.stbc_htltf2_en    = true;
    cfg.ltf_merge_en      = true;
    cfg.channel_filter_en = false;
    cfg.manu_scale        = false;
    if (esp_wifi_set_csi_config(&cfg) != ESP_OK ||
        esp_wifi_set_csi_rx_cb(shouter_csi_cb, NULL) != ESP_OK ||
        esp_wifi_set_csi(true) != ESP_OK) {
        Serial.println("[SHT] FATAL: CSI enable failed");
        while (1) delay(1000);
    }
    Serial.println("[SHT] CSI capture enabled");

    connect_and_register();
}

static uint32_t tx_seq = 1;  // starts at 1; 0 is never used in real responses

void loop() {
    // WiFi dropout recovery
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[SHT] WiFi dropped, reconnecting...");
        WiFi.reconnect();
        unsigned long t = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - t < 10000) delay(250);
        if (WiFi.status() == WL_CONNECTED) send_hello();
        return;
    }

    int pkt_size = udp.parsePacket();
    if (pkt_size < (int)sizeof(poll_pkt_t)) return;  // nothing or too small

    poll_pkt_t poll;
    udp.read((uint8_t*)&poll, sizeof(poll));

    // Validate magic and target
    if (poll.magic[0] != POLL_MAGIC_0 || poll.magic[1] != POLL_MAGIC_1) return;
    if (poll.target_id != SHOUTER_ID) return;

    // Grab latest CSI (or empty if ring buffer has no entries yet).
    // Design note: we always use the newest entry. The spec (Section 4.3) suggests
    // selecting the entry with rx_timestamp closest to the poll send time; with
    // RING_SIZE=2 and a 50ms poll cycle the newest entry is always the best candidate,
    // so this simplification is intentional. Revisit if RING_SIZE or cycle rate changes.
    CsiEntry csi_e;
    bool has_csi = get_latest_csi(&csi_e);

    // Build response
    response_pkt_t resp;
    memset(&resp, 0, sizeof(resp));
    resp.magic[0]   = RESP_MAGIC_0;
    resp.magic[1]   = RESP_MAGIC_1;
    resp.ver        = 1;
    resp.shouter_id = SHOUTER_ID;
    resp.tx_seq     = tx_seq;
    resp.tx_ms      = millis();
    resp.poll_seq   = poll.poll_seq;
    if (has_csi) {
        resp.poll_rssi        = csi_e.rssi;
        resp.poll_noise_floor = csi_e.noise_floor;
        resp.csi_len          = csi_e.len;
        memcpy(resp.csi, csi_e.bytes, csi_e.len);
    }
    // else csi_len = 0 (ring buffer empty) — already zeroed by memset

    udp.beginPacket(LISTENER_IP, LISTENER_PORT);
    udp.write((uint8_t*)&resp, sizeof(resp));
    udp.endPacket();
    tx_seq++;
}
