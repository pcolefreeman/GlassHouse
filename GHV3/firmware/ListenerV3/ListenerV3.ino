#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "../GHV3Protocol.h"

// ── Configuration ──────────────────────────────────────────────────────────────
#define SSID              "CSI_PRIVATE_AP"
#define CHANNEL           6
#define LISTENER_PORT     3333
#define SHOUTER_PORT      3334
#define POLL_TIMEOUT_MS          100   // increased from 50ms — allows for 802.11 CSMA/CA backoff with 4 shouters
#define POLL_INTERVAL_MIN_MS      50
#define INTER_SHOUTER_GAP_MS       5   // guard interval between consecutive shouter polls to clear the channel
#define RANGING_COOLDOWN_MS    30000   // minimum ms between ranging phases; prevents re-ranging on brief dropout
#define RANGING_STABILITY_MS    5000   // all 4 shouters must be registered this long before ranging starts
#define SNAP_DRAIN_MS           2000   // per-shouter drain window: 35 snaps × 3 peers × 15ms + margin

// ── Shouter registry ───────────────────────────────────────────────────────────
static IPAddress shouter_ip[5];          // indices 1–4; [0] unused
static uint8_t   shouter_mac[5][6];      // MAC per shouter ID, from HELLO
static bool      shouter_ready[5] = {};  // true once HELLO received

// ── Globals ────────────────────────────────────────────────────────────────────
WiFiUDP  udp;
volatile uint32_t current_poll_seq = 0;
uint32_t poll_seq = 0;
static volatile bool ranging_done = false;
static unsigned long ranging_completed_ms = 0;  // millis() when last ranging phase finished
static unsigned long last_hello_ms = 0;         // millis() of most recent HELLO from any shouter
static uint16_t snap_frames_emitted = 0;

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

// Serial frame C: [0xCC][0xDD] + 12-byte payload (ranging_rpt_pkt_t bytes 2–13)
void emit_ranging_frame(const ranging_rpt_pkt_t *rpt) {
    uint8_t buf[14];
    buf[0] = SER_C_MAGIC_0;    // 0xCC
    buf[1] = SER_C_MAGIC_1;    // 0xDD
    // Bytes 2–13: copy bytes [2..13] of ranging_rpt_pkt_t (skip the 2-byte magic)
    memcpy(buf + 2, ((const uint8_t *)rpt) + 2, 12);
    Serial.write(buf, 14);
}

// Serial frame D: [0xEE][0xFF] + ver(1)+reporter_id(1)+peer_id(1)+bcn_seq(1)+csi_len(2)+csi[N]
// pkt->magic[2] ([BB][A4]) is NOT forwarded — Python parser expects 6-byte header after magic.
void emit_csi_snap_frame(const csi_snap_pkt_t *pkt) {
    uint16_t payload_len = (uint16_t)(offsetof(csi_snap_pkt_t, csi) - 2 + pkt->csi_len);
    Serial.write(SER_D_MAGIC_0);
    Serial.write(SER_D_MAGIC_1);
    Serial.write(((const uint8_t*)pkt) + 2, payload_len);
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
            last_hello_ms = millis();
            // NOTE: udp.remoteIP().toString() returns a temporary String; capture before c_str().
            String remote_str = udp.remoteIP().toString();
            Serial.printf("[LST] HELLO sid=%d IP=%s MAC=%02X:%02X:%02X:%02X:%02X:%02X\n",
                sid, remote_str.c_str(),
                buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);
        }
        udp.flush();  // consume any remaining bytes in this UDP packet
        return false;
    }

    // RANGE_RPT_MAGIC [0xBB][0xA3] — shouter's ranging report
    if (magic[0] == RANGE_RPT_MAGIC_0 && magic[1] == RANGE_RPT_MAGIC_1) {
        ranging_rpt_pkt_t rpt;
        rpt.magic[0] = magic[0];
        rpt.magic[1] = magic[1];
        udp.read(((uint8_t*)&rpt) + 2, sizeof(ranging_rpt_pkt_t) - 2);
        emit_ranging_frame(&rpt);
        return false;   // MUST return false — do NOT treat as poll hit
        // Returning true here would skip the real response_pkt_t and emit a MISS.
    }

    // CSI_SNAP_MAGIC [0xBB][0xA4] — shouter's CSI snapshot for MUSIC ranging
    if (magic[0] == CSI_SNAP_MAGIC_0 && magic[1] == CSI_SNAP_MAGIC_1) {
        if (sz < 2 + (int)(offsetof(csi_snap_pkt_t, csi))) {
            udp.flush();
            return false;
        }
        csi_snap_pkt_t snap;
        snap.magic[0] = magic[0];
        snap.magic[1] = magic[1];
        int remaining = sz - 2;
        if (remaining > (int)(sizeof(csi_snap_pkt_t) - 2))
            remaining = (int)(sizeof(csi_snap_pkt_t) - 2);
        udp.read(((uint8_t*)&snap) + 2, remaining);
        emit_csi_snap_frame(&snap);
        snap_frames_emitted++;
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

// ── poll_all_shouters — used by run_ranging_phase ─────────────────────────────
void poll_all_shouters() {
    for (uint8_t id = 1; id <= 4; id++) {
        if (!shouter_ready[id]) continue;
        Serial.printf("[LST] poll_all: polling shouter %d\n", id);

        portENTER_CRITICAL(&lst_ring_mux);
        current_poll_seq = poll_seq;
        portEXIT_CRITICAL(&lst_ring_mux);

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

        unsigned long deadline = millis() + POLL_TIMEOUT_MS;
        bool got_response = false;
        response_pkt_t resp;
        while (millis() < deadline) {
            drain_listener_csi();
            if (handle_incoming_udp(&resp)) {
                if (resp.poll_seq != poll_seq || resp.shouter_id != id) continue;
                got_response = true;
                emit_shouter_frame(&resp, id, true, millis());
                break;
            }
            delay(1);  // yield to FreeRTOS scheduler — feeds TWDT
        }
        if (!got_response) {
            emit_shouter_frame(nullptr, id, false, millis());
        }
        // Drain snap packets from this shouter before polling the next.
        // Shouter sends N_SNAP × 3ms ≈ 270ms of UDP after its poll response.
        // Without this window all 4 shouters burst concurrently and overflow
        // the listener UDP RX buffer (~8–16 KB), dropping ~85% of snap packets.
        unsigned long drain_end = millis() + SNAP_DRAIN_MS;
        while (millis() < drain_end) {
            response_pkt_t dummy;
            handle_incoming_udp(&dummy);
            drain_listener_csi();
            yield();  // yield without 1ms floor — process snaps as fast as possible
        }
        unsigned long gap_end = millis() + INTER_SHOUTER_GAP_MS;
        while (millis() < gap_end) { drain_listener_csi(); yield(); }
    }
    poll_seq++;
}

// run_ranging_phase — called once after all shouters have registered.
// Sends PEER_INFO to all shouters, then sequentially requests ranging beacons.
void run_ranging_phase() {
    Serial.println("[LST] Starting ranging phase");
    snap_frames_emitted = 0;

    // Build and send peer_info_pkt_t to each registered shouter
    peer_info_pkt_t pi;
    pi.magic[0] = PEER_INFO_MAGIC_0;
    pi.magic[1] = PEER_INFO_MAGIC_1;
    pi.ver      = 1;
    pi.n_peers  = 4;
    for (int i = 0; i < 4; i++) {
        pi.peers[i].shouter_id = (uint8_t)(i + 1);        // 1-indexed
        memcpy(pi.peers[i].mac, shouter_mac[i + 1], 6);
    }
    for (int s = 1; s <= 4; s++) {
        udp.beginPacket(shouter_ip[s], SHOUTER_PORT);
        udp.write((uint8_t *)&pi, sizeof(pi));
        udp.endPacket();
    }
    delay(50);

    // Sequential ranging: one shouter beacons at a time
    const uint8_t  N_BCN      = 35;   // matches N_SNAP=35 on shouter (DRAM limit)
    const uint16_t BCN_MS     = 20;   // unchanged
    const uint16_t MARGIN_MS  = 50;   // unchanged

    for (int beacon_id = 1; beacon_id <= 4; beacon_id++) {
        Serial.printf("[LST] Ranging: requesting beacons from shouter %d\n", beacon_id);
        range_req_pkt_t rr;
        rr.magic[0]    = RANGE_REQ_MAGIC_0;
        rr.magic[1]    = RANGE_REQ_MAGIC_1;
        rr.ver         = 1;
        rr.target_id   = beacon_id;
        rr.n_beacons   = N_BCN;
        rr.interval_ms = BCN_MS;
        udp.beginPacket(shouter_ip[beacon_id], SHOUTER_PORT);
        udp.write((uint8_t *)&rr, sizeof(rr));
        udp.endPacket();
        delay((uint32_t)N_BCN * BCN_MS + MARGIN_MS);
        Serial.printf("[LST] Ranging: beacon %d delay done, polling\n", beacon_id);
        // One normal poll cycle collects ranging_rpt_pkt_t from all shouters
        poll_all_shouters();
        Serial.printf("[LST] Ranging: beacon %d complete, snaps=%d\n", beacon_id, snap_frames_emitted);
    }
    Serial.printf("[LST] Ranging done: %d snap frames emitted\n", snap_frames_emitted);
    ranging_completed_ms = millis();
}

void setup() {
    Serial.begin(921600);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(SSID, nullptr, CHANNEL);
    Serial.printf("[LST] AP up: SSID=%s ch=%d IP=%s\n",
        SSID, CHANNEL, WiFi.softAPIP().toString().c_str());
    udp.begin(LISTENER_PORT);
    Serial.println("[LST] UDP listening on 3333");

    WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info) {
        const uint8_t *mac = info.wifi_ap_stadisconnected.mac;
        for (int i = 1; i <= 4; i++) {
            if (memcmp(shouter_mac[i], mac, 6) == 0) {
                shouter_ready[i] = false;
                ranging_done = false;
                Serial.printf("[LST] Shouter %d disconnected, ranging reset\n", i);
                break;
            }
        }
    }, ARDUINO_EVENT_WIFI_AP_STADISCONNECTED);

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

    // Count registered shouters
    uint8_t registered_shouter_count = 0;
    for (int i = 1; i <= 4; i++) {
        if (shouter_ready[i]) registered_shouter_count++;
    }

    // One-shot ranging phase once all 4 shouters register.
    // Stability window: wait RANGING_STABILITY_MS after the last HELLO to ensure
    // all shouters are stable before starting ranging.
    // Cooldown prevents re-ranging on brief dropout/reconnect cycles.
    if (!ranging_done && registered_shouter_count == 4 &&
        (millis() - ranging_completed_ms >= RANGING_COOLDOWN_MS) &&
        (millis() - last_hello_ms >= RANGING_STABILITY_MS)) {
        Serial.println("[LST] All 4 shouters stable — starting ranging");
        ranging_done = true;
        run_ranging_phase();
    }
    
    if (registered_shouter_count < 4) {
        Serial.printf("[LST] Waiting for shouters (%d/4 registered)\n", registered_shouter_count);
        last_cycle_ms = millis();
        return;
    }

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

        // Guard interval: let the channel clear before polling the next shouter.
        // Draining CSI here is useful work to fill the gap.
        unsigned long gap_end = millis() + INTER_SHOUTER_GAP_MS;
        while (millis() < gap_end) drain_listener_csi();
    }

    drain_listener_csi();  // final drain after all shouters polled this cycle

    // Always update last_cycle_ms so the inter-cycle gap (above) opens up and
    // handle_incoming_udp() runs — this is the only path that processes HELLO packets.
    // Without this, last_cycle_ms stays 0 forever when no shouters are registered and
    // HELLO packets are never consumed.
    // Only advance poll_seq when we actually polled someone.
    if (polled_any) {
        poll_seq++;
    }
    last_cycle_ms = millis();
}
