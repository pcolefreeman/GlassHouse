#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "../GHV4Protocol.h"

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
#define USE_BROADCAST_POLL  1     // 1 = broadcast + stagger; 0 = sequential (fallback)
#define STAGGER_MS         40     // per-shouter stagger delay for broadcast polling

// ── Shouter registry ───────────────────────────────────────────────────────────
static IPAddress shouter_ip[5];          // indices 1–4; [0] unused
static uint8_t   shouter_mac[5][6];      // MAC per shouter ID, from HELLO
static bool      shouter_ready[5] = {};  // true once HELLO received

// MAC → ID lookup table. Populate with actual MACs from `esptool.py read_mac`.
// If a MAC doesn't match, assign next available ID as fallback.
static const uint8_t known_macs[4][6] = {
    {0x68, 0xFE, 0x71, 0x90, 0x60, 0xA0},  // ID 1
    {0x68, 0xFE, 0x71, 0x90, 0x68, 0x14},  // ID 2
    {0x68, 0xFE, 0x71, 0x90, 0x6B, 0x90},  // ID 3
    {0x20, 0xE7, 0xC8, 0xEC, 0xF5, 0xDC},  // ID 4
};

uint8_t mac_to_id(const uint8_t mac[6]) {
    for (int i = 0; i < 4; i++) {
        if (memcmp(known_macs[i], mac, 6) == 0) return (uint8_t)(i + 1);
    }
    // Fallback: assign next available ID for unknown MACs (board replacement)
    for (int i = 1; i <= 4; i++) {
        if (!shouter_ready[i]) return (uint8_t)i;
    }
    return 0;  // all slots full
}

// ── Globals ────────────────────────────────────────────────────────────────────
WiFiUDP  udp;
volatile uint32_t current_poll_seq = 0;
uint32_t poll_seq = 0;
static volatile bool ranging_done = false;
static unsigned long ranging_completed_ms = 0;  // millis() when last ranging phase finished
static unsigned long last_hello_ms = 0;         // millis() of most recent HELLO from any shouter
static uint16_t snap_frames_emitted = 0;
static volatile uint32_t csi_overflow_count = 0;
static uint32_t lst_poll_count = 0;
static uint16_t consecutive_miss[5] = {};  // index 1-4; tracks consecutive poll misses
static bool     miss_warned[5] = {};       // true once warning emitted; reset on hit

// ── Ranging state machine ────────────────────────────────────────────────────
enum RangingState {
    RNG_IDLE, RNG_SEND_PEER_INFO, RNG_WAIT_PEER_ACK,
    RNG_BEACON_ROUND, RNG_WAIT_BEACONS, RNG_DRAIN_SNAPS,
    RNG_NEXT_SHOUTER, RNG_COMPLETE
};
static volatile RangingState rng_state = RNG_IDLE;  // volatile: written by WiFi event handler (Core 0), read by loop() (Core 1)
static uint8_t  rng_current_shouter = 0;
static uint32_t rng_state_entered_ms = 0;

// Ranging constants (moved from run_ranging_phase)
static const uint8_t  RNG_N_BCN     = 35;
static const uint16_t RNG_BCN_MS    = 20;
static const uint16_t RNG_MARGIN_MS = 50;

// Dynamic snap drain tracking (#9)
static uint16_t snap_drain_count = 0;
static uint32_t last_snap_ms = 0;
static const uint16_t SNAP_EXPECTED_MAX = 105;  // N_BCN * 3 peers
static const uint32_t SNAP_SILENCE_MS   = 500;
static const uint32_t SNAP_HARD_CAP_MS  = 3000;

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
    if (lst_ring_write - lst_ring_read >= LST_RING_SIZE) {
        csi_overflow_count++;
        portEXIT_CRITICAL_ISR(&lst_ring_mux);
        return;  // ring full — skip this frame
    }
    int idx = lst_ring_write % LST_RING_SIZE;
    memcpy(lst_ring[idx].mac, info->mac, 6);
    lst_ring[idx].snap_seq    = current_poll_seq;
    lst_ring[idx].ts_ms       = (uint32_t)(esp_timer_get_time() / 1000);
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
        uint8_t *src_mac = buf + 2;  // MAC starts at offset 2 in buf (after ver, shouter_id)
        uint8_t sid = mac_to_id(src_mac);
        if (sid == 0) {
            Serial.printf("[LST] WARN unknown MAC %02X:%02X:%02X:%02X:%02X:%02X — no slots available\n",
                src_mac[0], src_mac[1], src_mac[2], src_mac[3], src_mac[4], src_mac[5]);
            udp.flush();
            return false;
        }
        if (sid >= 1 && sid <= 4) {
            shouter_ip[sid] = udp.remoteIP();
            memcpy(shouter_mac[sid], src_mac, 6);
            shouter_ready[sid] = true;
            last_hello_ms = millis();
            // NOTE: udp.remoteIP().toString() returns a temporary String; capture before c_str().
            String remote_str = udp.remoteIP().toString();
            Serial.printf("[LST] HELLO sid=%d (MAC-assigned) IP=%s MAC=%02X:%02X:%02X:%02X:%02X:%02X\n",
                sid, remote_str.c_str(),
                src_mac[0], src_mac[1], src_mac[2], src_mac[3], src_mac[4], src_mac[5]);
            // Send HELLO ACK with assigned ID
            ack_pkt_t ack = {};
            ack.magic[0]    = ACK_MAGIC_0;
            ack.magic[1]    = ACK_MAGIC_1;
            ack.ack_type    = HELLO_MAGIC_1;  // 0xFA
            ack.assigned_id = sid;
            ack.ack_seq     = 0;
            udp.beginPacket(udp.remoteIP(), SHOUTER_PORT);
            udp.write((uint8_t*)&ack, sizeof(ack));
            udp.endPacket();
        }
        udp.flush();  // consume any remaining bytes in this UDP packet
        return false;
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

    // ACK_MAGIC [0xBB][0xA5] — shouter acknowledges PEER_INFO
    if (magic[0] == ACK_MAGIC_0 && magic[1] == ACK_MAGIC_1) {
        ack_pkt_t ack;
        ack.magic[0] = magic[0];
        ack.magic[1] = magic[1];
        udp.read(((uint8_t*)&ack) + 2, sizeof(ack_pkt_t) - 2);
        if (ack.ack_type == PEER_INFO_MAGIC_1) {  // 0xA0
            Serial.printf("[LST] PEER_INFO ACK from shouter (type=0x%02X)\n", ack.ack_type);
        }
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

// ── Non-blocking ranging state machine ────────────────────────────────────────
void advance_ranging() {
    uint32_t now = millis();

    switch (rng_state) {
    case RNG_IDLE: {
        // Check trigger conditions
        uint8_t cnt = 0;
        for (int i = 1; i <= 4; i++) if (shouter_ready[i]) cnt++;
        if (!ranging_done && cnt == 4 &&
            (now - ranging_completed_ms >= RANGING_COOLDOWN_MS) &&
            (now - last_hello_ms >= RANGING_STABILITY_MS)) {
            Serial.println("[LST] All 4 shouters stable — starting ranging (non-blocking)");
            rng_state = RNG_SEND_PEER_INFO;
            rng_state_entered_ms = now;
            snap_frames_emitted = 0;
        }
        break;
    }

    case RNG_SEND_PEER_INFO: {
        peer_info_pkt_t pi;
        pi.magic[0] = PEER_INFO_MAGIC_0;
        pi.magic[1] = PEER_INFO_MAGIC_1;
        pi.ver      = 1;
        pi.n_peers  = 4;
        for (int i = 0; i < 4; i++) {
            pi.peers[i].shouter_id = (uint8_t)(i + 1);
            memcpy(pi.peers[i].mac, shouter_mac[i + 1], 6);
        }
        for (int s = 1; s <= 4; s++) {
            udp.beginPacket(shouter_ip[s], SHOUTER_PORT);
            udp.write((uint8_t *)&pi, sizeof(pi));
            udp.endPacket();
        }
        rng_state = RNG_WAIT_PEER_ACK;
        rng_state_entered_ms = now;
        Serial.println("[LST] PEER_INFO sent to all shouters, waiting 200ms for processing");
        break;
    }

    case RNG_WAIT_PEER_ACK: {
        // Best-effort wait: give shouters 200ms to process PEER_INFO before beaconing.
        // ACKs are logged by handle_incoming_udp but not counted here — proceeding is unconditional.
        if (now - rng_state_entered_ms >= 200) {
            rng_current_shouter = 1;
            rng_state = RNG_BEACON_ROUND;
            rng_state_entered_ms = now;
        }
        break;
    }

    case RNG_BEACON_ROUND: {
        Serial.printf("[LST] Ranging: requesting beacons from shouter %d\n", rng_current_shouter);
        range_req_pkt_t rr;
        rr.magic[0]    = RANGE_REQ_MAGIC_0;
        rr.magic[1]    = RANGE_REQ_MAGIC_1;
        rr.ver         = 1;
        rr.target_id   = rng_current_shouter;
        rr.n_beacons   = RNG_N_BCN;
        rr.interval_ms = RNG_BCN_MS;
        udp.beginPacket(shouter_ip[rng_current_shouter], SHOUTER_PORT);
        udp.write((uint8_t *)&rr, sizeof(rr));
        udp.endPacket();
        rng_state = RNG_WAIT_BEACONS;
        rng_state_entered_ms = now;
        break;
    }

    case RNG_WAIT_BEACONS: {
        uint32_t wait_ms = (uint32_t)RNG_N_BCN * RNG_BCN_MS + RNG_MARGIN_MS;
        if (now - rng_state_entered_ms >= wait_ms) {
            rng_state = RNG_DRAIN_SNAPS;
            rng_state_entered_ms = now;
            snap_drain_count = 0;
            last_snap_ms = now;
            Serial.printf("[LST] Ranging: beacon %d wait done, draining snaps\n", rng_current_shouter);
        }
        break;
    }

    case RNG_DRAIN_SNAPS: {
        static uint16_t drain_start_snaps = 0;
        if (snap_drain_count == 0) {
            drain_start_snaps = snap_frames_emitted;
        }
        uint16_t new_snaps = snap_frames_emitted - drain_start_snaps;
        if (new_snaps > snap_drain_count) {
            snap_drain_count = new_snaps;
            last_snap_ms = now;
        }
        bool done = false;
        if (snap_drain_count >= SNAP_EXPECTED_MAX) done = true;
        if (now - last_snap_ms >= SNAP_SILENCE_MS) done = true;
        if (now - rng_state_entered_ms >= SNAP_HARD_CAP_MS) done = true;
        if (done) {
            Serial.printf("[LST] Ranging: drained %d snaps for shouter %d\n",
                snap_drain_count, rng_current_shouter);
            rng_state = RNG_NEXT_SHOUTER;
        }
        break;
    }

    case RNG_NEXT_SHOUTER: {
        rng_current_shouter++;
        if (rng_current_shouter <= 4) {
            rng_state = RNG_BEACON_ROUND;
            rng_state_entered_ms = now;
        } else {
            rng_state = RNG_COMPLETE;
        }
        break;
    }

    case RNG_COMPLETE: {
        ranging_done = true;
        ranging_completed_ms = now;
        Serial.printf("[LST] Ranging done (non-blocking): %d snap frames emitted\n", snap_frames_emitted);
        rng_state = RNG_IDLE;
        break;
    }
    }
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
                if (rng_state != RNG_IDLE) {
                    rng_state = RNG_IDLE;
                    Serial.printf("[LST] Ranging aborted — shouter %d disconnected\n", i);
                } else {
                    Serial.printf("[LST] Shouter %d disconnected, ranging reset\n", i);
                }
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
        delay(3000); ESP.restart();
    }
    Serial.println("[LST] CSI capture enabled");
}

static unsigned long last_cycle_ms = 0;

void loop() {
    unsigned long now = millis();

    // Always handle incoming UDP and drain CSI — these run in every state
    response_pkt_t dummy;
    handle_incoming_udp(&dummy);
    drain_listener_csi();

    // Advance ranging state machine (one step per iteration)
    advance_ranging();

    // Normal polling only when ranging is idle
    if (rng_state != RNG_IDLE) return;

    // Rate limit polling
    if (now - last_cycle_ms < POLL_INTERVAL_MIN_MS) return;

    // Count registered shouters
    uint8_t registered_shouter_count = 0;
    for (int i = 1; i <= 4; i++) {
        if (shouter_ready[i]) registered_shouter_count++;
    }

    if (registered_shouter_count < 4) {
        Serial.printf("[LST] Waiting for shouters (%d/4 registered)\n", registered_shouter_count);
        last_cycle_ms = millis();
        return;
    }

#if USE_BROADCAST_POLL
    // Broadcast poll — single packet, staggered responses
    portENTER_CRITICAL(&lst_ring_mux);
    current_poll_seq = poll_seq;
    portEXIT_CRITICAL(&lst_ring_mux);

    poll_pkt_t pkt;
    memset(&pkt, 0, sizeof(pkt));
    pkt.magic[0]    = POLL_MAGIC_0;
    pkt.magic[1]    = POLL_MAGIC_1;
    pkt.ver         = 1;
    pkt.target_id   = 0xFF;  // broadcast sentinel
    pkt.poll_seq    = poll_seq;
    pkt.listener_ms = millis();
    memset(pkt.pad, 0xA5, POLL_PAD_SIZE);
    udp.beginPacket(IPAddress(192, 168, 4, 255), SHOUTER_PORT);
    udp.write((uint8_t*)&pkt, sizeof(pkt));
    udp.endPacket();

    // Collect staggered responses
    bool got[5] = {};
    unsigned long window_end = millis() + 4 * STAGGER_MS + POLL_TIMEOUT_MS;
    while (millis() < window_end) {
        drain_listener_csi();
        response_pkt_t resp;
        if (handle_incoming_udp(&resp)) {
            uint8_t sid = resp.shouter_id;
            if (resp.poll_seq == poll_seq && sid >= 1 && sid <= 4 && !got[sid]) {
                got[sid] = true;
                emit_shouter_frame(&resp, sid, true, millis());
                consecutive_miss[sid] = 0;
                miss_warned[sid] = false;
            }
        }
        delayMicroseconds(100);
    }
    // Emit misses for shouters that didn't respond
    for (uint8_t id = 1; id <= 4; id++) {
        if (!shouter_ready[id]) continue;
        if (!got[id]) {
            emit_shouter_frame(nullptr, id, false, millis());
            consecutive_miss[id]++;
            if (consecutive_miss[id] >= 10 && !miss_warned[id]) {
                Serial.printf("[LST] WARN shouter %d: %d consecutive misses\n",
                    id, consecutive_miss[id]);
                miss_warned[id] = true;
            }
        }
    }
    bool polled_any = true;

#else
    bool polled_any = false;
    for (uint8_t id = 1; id <= 4; id++) {
        if (!shouter_ready[id]) continue;
        polled_any = true;

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
                consecutive_miss[id] = 0;
                miss_warned[id] = false;
                break;
            }
            delayMicroseconds(100);
        }

        if (!got_response) {
            emit_shouter_frame(nullptr, id, false, millis());
            consecutive_miss[id]++;
            if (consecutive_miss[id] >= 10 && !miss_warned[id]) {
                Serial.printf("[LST] WARN shouter %d: %d consecutive misses\n",
                    id, consecutive_miss[id]);
                miss_warned[id] = true;
            }
        }

        unsigned long gap_end = millis() + INTER_SHOUTER_GAP_MS;
        while (millis() < gap_end) drain_listener_csi();
    }
#endif

    drain_listener_csi();

    if (polled_any) {
        poll_seq++;
        lst_poll_count++;                    // #7: overflow emission counter
        if (lst_poll_count % 100 == 0) {
            Serial.printf("[LST] csi_overflow=%lu\n", (unsigned long)csi_overflow_count);
        }
    }
    last_cycle_ms = millis();
}
