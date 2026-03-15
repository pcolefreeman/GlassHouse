#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "GHV2Protocol.h"

// ── Configuration ──────────────────────────────────────────────────────────────
#define SSID              "CSI_PRIVATE_AP"
#define CHANNEL           6
#define LISTENER_PORT     3333
#define SHOUTER_PORT      3334
#define POLL_TIMEOUT_MS   50
#define POLL_INTERVAL_MIN_MS 50

// ── Shouter registry ───────────────────────────────────────────────────────────
static IPAddress shouter_ip[5];          // indices 1–4; [0] unused
static uint8_t   shouter_mac[5][6];      // MAC per shouter ID, from HELLO
static bool      shouter_ready[5] = {};  // true once HELLO received

// ── Globals ────────────────────────────────────────────────────────────────────
WiFiUDP  udp;
volatile uint32_t current_poll_seq = 0;
uint32_t poll_seq = 0;

// ── Listener CSI ring buffer — callback stores here; loop drains to Serial ────
#define LST_RING_SIZE 4

struct LstCsiEntry {
    uint8_t  mac[6];
    uint32_t snap_seq;
    uint32_t ts_ms;
    int8_t   rssi;
    int8_t   noise_floor;
    uint16_t csi_len;
    uint8_t  csi[CSI_MAX_BYTES];
};

static LstCsiEntry        lst_ring[LST_RING_SIZE];
static volatile int       lst_ring_write = 0;
static volatile int       lst_ring_read  = 0;
static portMUX_TYPE       lst_ring_mux   = portMUX_INITIALIZER_UNLOCKED;

// IRAM_ATTR: only stores to ring buffer; no heap, no FreeRTOS calls, no Serial.
void IRAM_ATTR listener_csi_cb(void* ctx, wifi_csi_info_t* info) {
    if (!info || !info->buf || info->len <= 0) return;
    uint16_t copy_len = (info->len <= CSI_MAX_BYTES) ? info->len : CSI_MAX_BYTES;
    portENTER_CRITICAL_ISR(&lst_ring_mux);
    int idx = lst_ring_write % LST_RING_SIZE;
    memcpy(lst_ring[idx].mac, info->mac, 6);
    lst_ring[idx].snap_seq    = current_poll_seq;
    lst_ring[idx].ts_ms       = (uint32_t)(esp_timer_get_time() / 1000);  // IRAM-safe
    lst_ring[idx].rssi        = info->rx_ctrl.rssi;
    lst_ring[idx].noise_floor = info->rx_ctrl.noise_floor;
    lst_ring[idx].csi_len     = copy_len;
    memcpy(lst_ring[idx].csi, info->buf, copy_len);
    lst_ring_write++;
    portEXIT_CRITICAL_ISR(&lst_ring_mux);
}

// ── [0xAA][0x55] Serial emission ──────────────────────────────────────────────
// Layout: magic(2) ver(1) flags(1) ts_ms(4) rssi(1) nf(1) mac(6) poll_seq(4) csi_len(2) csi[N]
// Called from task context ONLY — Serial.write uses FreeRTOS semaphores.
void emit_listener_frame(const LstCsiEntry* e) {
    Serial.write(SER_A_MAGIC_0);
    Serial.write(SER_A_MAGIC_1);
    Serial.write((uint8_t)1);                    // ver
    Serial.write((uint8_t)0x00);                 // flags = listener-side
    Serial.write((uint8_t*)&e->ts_ms, 4);        // timestamp_ms
    Serial.write((uint8_t)e->rssi);
    Serial.write((uint8_t)e->noise_floor);
    Serial.write(e->mac, 6);                     // transmitter MAC
    Serial.write((uint8_t*)&e->snap_seq, 4);     // poll_seq
    Serial.write((uint8_t*)&e->csi_len, 2);
    Serial.write(e->csi, e->csi_len);
}

// Drain all pending entries from the listener CSI ring buffer. Call from loop().
void drain_listener_csi() {
    while (true) {
        portENTER_CRITICAL(&lst_ring_mux);
        if (lst_ring_read == lst_ring_write) { portEXIT_CRITICAL(&lst_ring_mux); break; }
        int idx = lst_ring_read % LST_RING_SIZE;
        LstCsiEntry e = lst_ring[idx];
        lst_ring_read++;
        portEXIT_CRITICAL(&lst_ring_mux);
        emit_listener_frame(&e);
    }
}

// ── [0xBB][0xDD] Serial emission ──────────────────────────────────────────────
// Layout: magic(2) ver(1) flags(1) listener_ms(4) tx_seq(4) tx_ms(4) shouter_id(1)
//         poll_seq(4) poll_rssi(1) poll_nf(1) mac(6) csi_len(2) csi[N]
void emit_shouter_frame(const response_pkt_t* resp, uint8_t id,
                        bool is_hit, uint32_t listener_ms) {
    uint8_t  flags = is_hit ? 0x01 : 0x00;
    uint32_t tx_s  = is_hit ? resp->tx_seq          : 0;
    uint32_t tx_m  = is_hit ? resp->tx_ms           : 0;
    uint32_t p_seq = is_hit ? resp->poll_seq        : poll_seq;
    int8_t   p_rss = is_hit ? resp->poll_rssi       : 0;
    int8_t   p_nf  = is_hit ? resp->poll_noise_floor: 0;
    uint16_t clen  = is_hit ? resp->csi_len         : 0;
    uint8_t* mac   = shouter_mac[id];

    Serial.write(SER_B_MAGIC_0);
    Serial.write(SER_B_MAGIC_1);
    Serial.write((uint8_t)1);                // ver
    Serial.write(flags);
    Serial.write((uint8_t*)&listener_ms, 4);
    Serial.write((uint8_t*)&tx_s, 4);
    Serial.write((uint8_t*)&tx_m, 4);
    Serial.write(id);
    Serial.write((uint8_t*)&p_seq, 4);
    Serial.write((uint8_t)p_rss);
    Serial.write((uint8_t)p_nf);
    Serial.write(mac, 6);
    Serial.write((uint8_t*)&clen, 2);
    if (clen > 0 && is_hit) Serial.write(resp->csi, clen);
}

// ── HELLO + response dispatcher ───────────────────────────────────────────────
// Call during poll-wait window to handle HELLO re-registrations.
// Returns pointer to response data if a RESP packet was read; nullptr otherwise.
// pkt_out must be a caller-allocated response_pkt_t.
bool handle_incoming_udp(response_pkt_t* pkt_out) {
    int sz = udp.parsePacket();
    if (sz < 2) return false;

    uint8_t magic[2];
    udp.read(magic, 2);

    if (magic[0] == HELLO_MAGIC_0 && magic[1] == HELLO_MAGIC_1) {
        uint8_t buf[sizeof(hello_pkt_t) - 2];
        udp.read(buf, sizeof(buf));
        uint8_t sid = buf[1];  // shouter_id
        if (sid >= 1 && sid <= 4) {
            shouter_ip[sid] = udp.remoteIP();
            memcpy(shouter_mac[sid], buf + 2, 6);
            shouter_ready[sid] = true;
            // NOTE: udp.remoteIP().toString() returns a temporary String; capture before c_str().
            String remote_str = udp.remoteIP().toString();
            Serial.printf("[LST] HELLO sid=%d IP=%s MAC=%02X:%02X:%02X:%02X:%02X:%02X\n",
                sid, remote_str.c_str(),
                buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
        }
        udp.flush();  // consume any remaining bytes in this UDP packet
        return false;
    }

    if (magic[0] == RESP_MAGIC_0 && magic[1] == RESP_MAGIC_1) {
        pkt_out->magic[0] = magic[0];
        pkt_out->magic[1] = magic[1];
        int remaining = (int)sizeof(response_pkt_t) - 2;
        udp.read(((uint8_t*)pkt_out) + 2, remaining);
        return true;
    }

    udp.flush();
    return false;
}

void setup() {
    Serial.begin(921600);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(SSID, nullptr, CHANNEL);
    Serial.printf("[LST] AP up: SSID=%s ch=%d IP=%s\n",
        SSID, CHANNEL, WiFi.softAPIP().toString().c_str());
    udp.begin(LISTENER_PORT);
    Serial.println("[LST] UDP listening on 3333");

    // Enforce MCS0_LGI on AP interface — SDK ≥ 2.0.0 required
    if (esp_wifi_config_80211_tx_rate(WIFI_IF_AP, WIFI_PHY_RATE_MCS0_LGI) != ESP_OK)
        Serial.println("[LST] WARNING: tx_rate config failed");

    wifi_csi_config_t cfg = {};
    cfg.lltf_en           = true;
    cfg.htltf_en          = true;
    cfg.stbc_htltf2_en    = true;
    cfg.ltf_merge_en      = true;
    cfg.channel_filter_en = false;
    cfg.manu_scale        = false;
    if (esp_wifi_set_csi_config(&cfg) != ESP_OK ||
        esp_wifi_set_csi_rx_cb(listener_csi_cb, NULL) != ESP_OK ||
        esp_wifi_set_csi(true) != ESP_OK) {
        Serial.println("[LST] FATAL: CSI enable failed");
        while (1) delay(1000);
    }
    Serial.println("[LST] CSI capture enabled");
}

static unsigned long last_cycle_ms = 0;

void loop() {
    unsigned long now = millis();
    if (now - last_cycle_ms < POLL_INTERVAL_MIN_MS) {
        response_pkt_t dummy;
        handle_incoming_udp(&dummy);  // handle HELLO packets during inter-cycle gap
        drain_listener_csi();         // drain buffered CSI frames during idle time
        return;
    }
    // NOTE: last_cycle_ms is set at the END of the poll loop (after all shouters polled),
    //       not here. Setting it here would make the interval measure from poll *start*;
    //       setting it at the end ensures POLL_INTERVAL_MIN_MS of idle time after the
    //       last poll completes, preventing back-to-back cycles when polling takes < 50ms.

    bool polled_any = false;
    for (uint8_t id = 1; id <= 4; id++) {
        if (!shouter_ready[id]) continue;
        polled_any = true;

        // Set global BEFORE sending poll — CSI callback reads it.
        // portENTER_CRITICAL ensures the write is visible to Core 0 (CSI callback).
        portENTER_CRITICAL(&lst_ring_mux);
        current_poll_seq = poll_seq;
        portEXIT_CRITICAL(&lst_ring_mux);

        // Build + send poll
        poll_pkt_t pkt;
        memset(&pkt, 0, sizeof(pkt));
        pkt.magic[0]    = POLL_MAGIC_0;
        pkt.magic[1]    = POLL_MAGIC_1;
        pkt.ver         = 1;
        pkt.target_id   = id;
        pkt.poll_seq    = poll_seq;
        pkt.listener_ms = millis();
        memset(pkt.pad, 0xA5, POLL_PAD_SIZE);
        udp.beginPacket(shouter_ip[id], SHOUTER_PORT);
        udp.write((uint8_t*)&pkt, sizeof(pkt));
        udp.endPacket();

        // Wait for response; drain CSI ring buffer while waiting
        unsigned long deadline = millis() + POLL_TIMEOUT_MS;
        bool got_response = false;
        response_pkt_t resp;
        while (millis() < deadline) {
            drain_listener_csi();
            if (handle_incoming_udp(&resp)) {
                // Discard stale responses (wrong poll_seq or wrong shouter)
                if (resp.poll_seq != poll_seq || resp.shouter_id != id) continue;
                got_response = true;
                emit_shouter_frame(&resp, id, true, millis());
                break;
            }
            delayMicroseconds(100);
        }

        if (!got_response) {
            emit_shouter_frame(nullptr, id, false, millis());
        }
    }

    drain_listener_csi();  // final drain after all shouters polled this cycle

    // Only advance poll_seq and update last_cycle_ms if we actually polled someone.
    // This prevents the seq counter from racing ahead during boot / before any HELLO.
    if (polled_any) {
        poll_seq++;
        last_cycle_ms = millis();
    }
}
