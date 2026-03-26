#include "esp_wifi.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include "../GHV4Protocol.h"
#include "esp_now.h"

// Safety guard: on_esp_now_recv uses portENTER_CRITICAL which is unsafe on single-core builds.
#if CONFIG_FREERTOS_UNICORE
#error "on_esp_now_recv uses portENTER_CRITICAL — unsafe on single-core FreeRTOS build"
#endif

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
static volatile uint32_t  csi_overflow_count = 0;
static uint32_t           sht_resp_count = 0;
static uint32_t           bcn_rx_count[5] = {};  // beacons received per peer ID (indices 1-4)

// ── CSI snapshot buffer (ranging phase) ───────────────────────────────────
#define N_SNAP 35
// csi_snap_buf[peer_id][snap_seq]; indices 1–4 valid, index 0 unused.
// Only 3 of 4 peer slots populated (a shouter never beacons to itself).
// Total: 5 × 35 × sizeof(CsiEntry) ≈ 5 × 35 × 392 = 68,600 bytes (~67 KB).
// N_SNAP=60 overflowed DRAM by 42616 bytes; max safe is ~38.
static CsiEntry csi_snap_buf[5][N_SNAP];
static uint8_t  csi_snap_count[5] = {};
static portMUX_TYPE snap_mux = portMUX_INITIALIZER_UNLOCKED;

// ── Peer RSSI table (populated from PEER_INFO, updated by CSI callback) ──────
struct PeerEntry {
    uint8_t mac[6];
    int8_t  rssi;    // EMA-smoothed, dBm; initialised to 0
    uint8_t count;   // observations, capped at 255
    bool    valid;   // true once PEER_INFO received for this slot
};
static PeerEntry    peer_table[5] = {};  // index 1–4; index 0 unused
static portMUX_TYPE peer_mux = portMUX_INITIALIZER_UNLOCKED;

// Called by WiFi driver — keep short, no heap allocation
// IMPORTANT: millis() is NOT IRAM-safe and must NOT be called here.
//            Use esp_timer_get_time() (returns microseconds) and convert to ms.
void IRAM_ATTR shouter_csi_cb(void* ctx, wifi_csi_info_t* info) {
    if (!info || !info->buf || info->len <= 0) return;
    uint16_t copy_len = (info->len <= SHOUTER_CSI_MAX) ? info->len : SHOUTER_CSI_MAX;
    portENTER_CRITICAL_ISR(&ring_mux);
    if (ring_count >= RING_SIZE) {
        csi_overflow_count++;  // track overwrite
    }
    int idx = ring_write % RING_SIZE;
    memcpy(ring[idx].bytes, info->buf, copy_len);
    ring[idx].len             = copy_len;
    ring[idx].rx_timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);
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

// ── Per-device ID — assigned at runtime by listener via HELLO ACK (MAC-based) ──
static uint8_t my_id = 0;  // assigned by listener via HELLO ACK; 0 = not yet assigned
#define SSID          "CSI_PRIVATE_AP"
// NOTE: AP is open (no WPA2 password). WiFi.softAP(SSID, nullptr, CHANNEL) on listener
//       side explicitly passes nullptr for the password to keep the embedded mesh simple.
//       Do NOT deploy on an untrusted network.
#define LISTENER_IP   "192.168.4.1"
#define LISTENER_PORT 3333
#define SHOUTER_PORT  3334
#define STAGGER_MS         40

WiFiUDP udp;

void send_hello() {
    hello_pkt_t pkt;
    memset(&pkt, 0, sizeof(pkt));
    pkt.magic[0]   = HELLO_MAGIC_0;
    pkt.magic[1]   = HELLO_MAGIC_1;
    pkt.ver        = 1;
    pkt.shouter_id = my_id;  // 0 until assigned; listener uses MAC not this field
    WiFi.macAddress(pkt.src_mac);
    udp.beginPacket(LISTENER_IP, LISTENER_PORT);
    udp.write((uint8_t*)&pkt, sizeof(pkt));
    udp.endPacket();
    uint8_t mac[6];
    WiFi.macAddress(mac);
    Serial.printf("[SHT] HELLO sent (MAC=%02X:%02X:%02X:%02X:%02X:%02X)\n",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void connect_and_register() {
    // WiFi.mode() already called in setup(); reconnects from loop() don't need to re-set it
    WiFi.begin(SSID);
    Serial.print("[SHT] Connecting");
    unsigned long t = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - t > 15000) {
            Serial.println("\n[SHT] FATAL: WiFi connect timeout — check AP is up");
            delay(3000); ESP.restart();
        }
        delay(250);
        Serial.print(".");
    }
    Serial.printf("\n[SHT] Connected, IP=%s\n", WiFi.localIP().toString().c_str());
    send_hello();
}

// ESP-NOW receive callback — fires in WiFi task context (Core 0), not ISR.
// Captures true peer-to-peer RSSI from ranging beacons sent by peer shouters.
// Uses portENTER_CRITICAL (task variant) — safe on dual-core ESP32-WROOM-UE.
void on_esp_now_recv(const esp_now_recv_info_t *recv_info,
                     const uint8_t *data, int data_len) {
    if (data_len < (int)sizeof(range_bcn_pkt_t)) return;
    const range_bcn_pkt_t *bcn = (const range_bcn_pkt_t *)data;
    if (bcn->magic[0] != RANGE_BCN_MAGIC_0 ||
        bcn->magic[1] != RANGE_BCN_MAGIC_1) return;
    uint8_t sid = bcn->shouter_id;
    if (sid < 1 || sid > 4 || sid == my_id) return;
    bcn_rx_count[sid]++;
    int8_t rssi = (int8_t)recv_info->rx_ctrl->rssi;
    portENTER_CRITICAL(&peer_mux);
    if (peer_table[sid].valid) {
        if (peer_table[sid].count == 0) {
            peer_table[sid].rssi = rssi;           // first sample: no EMA blend
        } else {
            peer_table[sid].rssi = (int8_t)(
                (7 * (int)peer_table[sid].rssi + (int)rssi) / 8
            );
        }
        if (peer_table[sid].count < 255) peer_table[sid].count++;
    }
    portEXIT_CRITICAL(&peer_mux);
    // Passive background beacons (bcn_seq=0xFF) update peer_table RSSI above but
    // are not part of a structured ranging window — skip CSI snapshots for them.
    if (bcn->bcn_seq == 0xFF) return;
    // Snapshot the CSI that shouter_csi_cb stored for this exact ESP-NOW frame.
    // Callback ordering: shouter_csi_cb (ISR, Core 0) always completes before
    // on_esp_now_recv (WiFi task, Core 0) resumes. get_latest_csi() returns the
    // entry written by that ISR.
    CsiEntry snap;
    if (get_latest_csi(&snap)) {
        uint8_t idx;
        bool stored = false;
        portENTER_CRITICAL(&snap_mux);
        idx = csi_snap_count[sid];
        if (idx < N_SNAP) {
            csi_snap_buf[sid][idx] = snap;
            csi_snap_count[sid]++;
            stored = true;
        }
        portEXIT_CRITICAL(&snap_mux);
        // Print OUTSIDE critical section — Serial.printf is blocking I/O
        // and will trigger watchdog timeout if called with interrupts disabled.
        if (stored && (idx % 10 == 0)) {
            Serial.printf("[SHT] snap peer=%d seq=%d len=%d\n", sid, idx, snap.len);
        }
    } else {
        Serial.printf("[SHT] snap peer=%d MISS (no CSI in ring buffer)\n", sid);
    }
}

void setup() {
    Serial.begin(921600);

    // WiFi driver must be initialized before udp.begin() or any esp_wifi_* IDF calls;
    // otherwise the lwIP semaphores are NULL and FreeRTOS asserts in xQueueSemaphoreTake.
    WiFi.mode(WIFI_STA);

    // CSI config — requires WiFi driver to be up (after WiFi.mode())
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
        delay(3000); ESP.restart();
    }
    Serial.println("[SHT] CSI capture enabled");

    udp.begin(SHOUTER_PORT);   // safe — WiFi driver initialized above; must be before send_hello()

    connect_and_register();

    // ESP-NOW init — must be after WiFi STA is connected (connect_and_register completed).
    // Called once only; persists across WiFi reconnects.
    if (esp_now_init() != ESP_OK) {
        Serial.println("[SHT] FATAL: esp_now_init failed");
        delay(3000); ESP.restart();
    }
    esp_now_register_recv_cb(on_esp_now_recv);

    // Broadcast MAC must be registered as a peer before esp_now_send will accept it.
    // Without esp_now_add_peer, send returns ESP_ERR_ESPNOW_NOT_FOUND silently.
    // ifidx = WIFI_IF_STA because shouter is in STA mode.
    {
        esp_now_peer_info_t bcast_peer = {};
        memset(bcast_peer.peer_addr, 0xFF, 6);
        bcast_peer.channel = 0;        // 0 = use current STA channel (avoids ESP_ERR_ESPNOW_CHAN)
        bcast_peer.encrypt = false;
        bcast_peer.ifidx   = WIFI_IF_STA;
        if (esp_now_add_peer(&bcast_peer) != ESP_OK) {
            Serial.println("[SHT] FATAL: esp_now_add_peer(broadcast) failed");
            delay(3000); ESP.restart();
        }
    }
    Serial.println("[SHT] ESP-NOW ready");
    // NOTE: esp_now_init() must NOT be called again on WiFi dropout/reconnect.
    // ESP-NOW peers persist across reconnects. The WiFi dropout recovery path
    // in loop() calls WiFi.reconnect() only — do not add esp_now_init() there.
}

static uint8_t  udp_buf[512];
static uint32_t tx_seq = 1;       // starts at 1; 0 is never used in real responses
static uint32_t last_poll_rx_ms = 0;  // millis() of last poll received; 0 = never
static bool listener_warning_sent = false;
static bool     stagger_pending = false;
static uint32_t stagger_target_ms = 0;
static poll_pkt_t stagger_poll;  // saved poll for delayed response

// ── Poll response helper ────────────────────────────────────────────────────
// Extracted from loop() so broadcast-staggered polls can reuse the same path.
void send_poll_response(poll_pkt_t *poll) {
    last_poll_rx_ms = millis();
    listener_warning_sent = false;
    Serial.printf("[SHT] POLL seq=%lu\n", (unsigned long)poll->poll_seq);

    // Select CSI entry closest to current time (timestamp-based matching).
    CsiEntry csi_e;
    bool has_csi = false;
    portENTER_CRITICAL(&ring_mux);
    if (ring_count > 0) {
        uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
        int best = 0;
        uint32_t best_age = UINT32_MAX;
        int n = (ring_count < RING_SIZE) ? ring_count : RING_SIZE;
        for (int i = 0; i < n; i++) {
            uint32_t age = now_ms - ring[i].rx_timestamp_ms;
            if (age < best_age) { best_age = age; best = i; }
        }
        memcpy(&csi_e, &ring[best], sizeof(CsiEntry));
        has_csi = true;
    }
    portEXIT_CRITICAL(&ring_mux);

    // Build response
    response_pkt_t resp;
    memset(&resp, 0, sizeof(resp));
    resp.magic[0]   = RESP_MAGIC_0;
    resp.magic[1]   = RESP_MAGIC_1;
    resp.ver        = 1;
    resp.shouter_id = my_id;
    resp.tx_seq     = tx_seq;
    resp.tx_ms      = millis();
    resp.poll_seq   = poll->poll_seq;
    if (has_csi) {
        resp.poll_rssi        = csi_e.rssi;
        resp.poll_noise_floor = csi_e.noise_floor;
        resp.csi_len          = csi_e.len;
        memcpy(resp.csi, csi_e.bytes, csi_e.len);
    }

    udp.beginPacket(LISTENER_IP, LISTENER_PORT);
    udp.write((uint8_t*)&resp, sizeof(resp));
    udp.endPacket();
    Serial.printf("[SHT] SHOUT tx=%lu poll=%lu csi=%u\n",
        (unsigned long)tx_seq, (unsigned long)poll->poll_seq,
        (unsigned)resp.csi_len);
    tx_seq++;
    sht_resp_count++;
    if (sht_resp_count % 100 == 0) {
        Serial.printf("[SHT] csi_overflow=%lu\n", (unsigned long)csi_overflow_count);
        Serial.printf("[SHT] bcn_rx: S1=%lu S2=%lu S3=%lu S4=%lu\n",
            (unsigned long)bcn_rx_count[1], (unsigned long)bcn_rx_count[2],
            (unsigned long)bcn_rx_count[3], (unsigned long)bcn_rx_count[4]);
    }

    // Transmit buffered CSI snapshots to listener.
    // Rotate drain start peer using poll's stagger offset to prevent S1 from
    // always getting first UDP bandwidth.  Without this, every shouter drains
    // peer 1 first, giving S1 paths a systematic fill-rate advantage.
    uint8_t drain_start = poll->pad[0] % 4;  // same rotation offset as stagger
    for (uint8_t step = 0; step < 4; step++) {
        uint8_t peer = ((drain_start + step) % 4) + 1;  // 1-based, rotated
        if (peer == my_id) continue;
        portENTER_CRITICAL(&snap_mux);
        uint8_t n = csi_snap_count[peer];
        csi_snap_count[peer] = 0;
        portEXIT_CRITICAL(&snap_mux);
        for (uint8_t s = 0; s < n; s++) {
            portENTER_CRITICAL(&snap_mux);
            CsiEntry e = csi_snap_buf[peer][s];
            portEXIT_CRITICAL(&snap_mux);
            csi_snap_pkt_t pkt;
            pkt.magic[0]     = CSI_SNAP_MAGIC_0;
            pkt.magic[1]     = CSI_SNAP_MAGIC_1;
            pkt.ver          = 1;
            pkt.reporter_id  = my_id;
            pkt.peer_id      = peer;
            pkt.snap_seq     = s;
            pkt.csi_len      = e.len < CSI_SNAP_MAX ? e.len : CSI_SNAP_MAX;
            memcpy(pkt.csi, e.bytes, pkt.csi_len);
            udp.beginPacket(LISTENER_IP, LISTENER_PORT);
            udp.write((uint8_t*)&pkt,
                      (uint16_t)(offsetof(csi_snap_pkt_t, csi) + pkt.csi_len));
            udp.endPacket();
            delay(2);  // was delay(15) — only ~6 snaps between polls now (vs 105 during ranging)
        }
    }
}

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

    // HELLO retry — if connected but not polled within 5 s, re-register with listener.
    // Guards against dropped HELLO UDP packets on initial connect.
    static uint32_t last_hello_retry_ms = 0;
    if (last_poll_rx_ms == 0 && millis() - last_hello_retry_ms >= 5000) {
        send_hello();
        last_hello_retry_ms = millis();
    }

    // Listener SPOF detection — warn if no polls received for 10 seconds
    if (last_poll_rx_ms > 0 && millis() - last_poll_rx_ms > 10000 &&
        !listener_warning_sent) {
        Serial.printf("[SHT] WARN no polls for %lu ms — listener may be down\n",
            (unsigned long)(millis() - last_poll_rx_ms));
        listener_warning_sent = true;
    }

    // Continuous ESP-NOW beacons for SAR breathing/heart rate detection (~20 Hz)
    // Jitter: random interval 42-58ms (base 50ms ± 8ms) to reduce ESP-NOW collisions
    // between shouters that would otherwise beacon at synchronized intervals.
    static uint32_t last_beacon_ms = 0;
    static uint32_t cont_bcn_seq = 0;
    static uint32_t next_beacon_interval = 50;  // ms until next beacon
    if (my_id > 0 && millis() - last_beacon_ms >= next_beacon_interval) {
        last_beacon_ms = millis();
        next_beacon_interval = 42 + (esp_random() % 17);  // 42-58ms (50 ± 8ms jitter)
        range_bcn_pkt_t bcn;
        bcn.magic[0]   = RANGE_BCN_MAGIC_0;
        bcn.magic[1]   = RANGE_BCN_MAGIC_1;
        bcn.ver        = 1;
        bcn.shouter_id = my_id;
        bcn.bcn_seq    = (uint8_t)(cont_bcn_seq & 0xFE);  // 0-254 even; never 0xFF
        cont_bcn_seq++;
        static const uint8_t ESPNOW_BROADCAST[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
        esp_now_send(ESPNOW_BROADCAST, (uint8_t *)&bcn, sizeof(bcn));
    }

    // Process staggered broadcast response
    if (stagger_pending && millis() >= stagger_target_ms) {
        stagger_pending = false;
        send_poll_response(&stagger_poll);
    }

    int pkt_len = udp.parsePacket();
    if (pkt_len < 2) return;

    udp.read(udp_buf, sizeof(udp_buf));

    // ACK_MAGIC [0xBB][0xA5] — listener acknowledges HELLO (carries assigned ID)
    if (pkt_len >= (int)sizeof(ack_pkt_t)) {
        ack_pkt_t *ack = (ack_pkt_t *)udp_buf;
        if (ack->magic[0] == ACK_MAGIC_0 && ack->magic[1] == ACK_MAGIC_1) {
            if (ack->ack_type == HELLO_MAGIC_1 && ack->assigned_id >= 1 && ack->assigned_id <= 4) {
                my_id = ack->assigned_id;
                Serial.printf("[SHT] HELLO ACK received, my_id=%d\n", my_id);
            }
            return;  // consumed
        }
    }

    // PEER_INFO_MAGIC [0xBB][0xA0] — listener sends our peers' MACs and IDs
    if (pkt_len >= (int)sizeof(peer_info_pkt_t)) {
        peer_info_pkt_t *pi = (peer_info_pkt_t *)udp_buf;
        if (pi->magic[0] == PEER_INFO_MAGIC_0 && pi->magic[1] == PEER_INFO_MAGIC_1) {
            portENTER_CRITICAL(&peer_mux);   // task context — not ISR variant
            for (int k = 0; k < pi->n_peers && k < 4; k++) {
                uint8_t sid = pi->peers[k].shouter_id;
                if (sid < 1 || sid > 4 || sid == my_id) continue;
                memcpy(peer_table[sid].mac, pi->peers[k].mac, 6);
                peer_table[sid].rssi  = 0;
                peer_table[sid].count = 0;
                peer_table[sid].valid = true;
            }
            portEXIT_CRITICAL(&peer_mux);
            portENTER_CRITICAL(&snap_mux);
            for (int k = 0; k < pi->n_peers && k < 4; k++) {
                uint8_t sid = pi->peers[k].shouter_id;
                if (sid < 1 || sid > 4 || sid == my_id) continue;
                csi_snap_count[sid] = 0;
            }
            portEXIT_CRITICAL(&snap_mux);
            // Send ACK for PEER_INFO
            ack_pkt_t ack = {};
            ack.magic[0]    = ACK_MAGIC_0;
            ack.magic[1]    = ACK_MAGIC_1;
            ack.ack_type    = PEER_INFO_MAGIC_1;  // 0xA0
            ack.assigned_id = 0;
            ack.ack_seq     = 0;
            udp.beginPacket(LISTENER_IP, LISTENER_PORT);
            udp.write((uint8_t*)&ack, sizeof(ack));
            udp.endPacket();
            Serial.println("[SHT] PEER_INFO ACK sent");
            return;  // consumed
        }
    }

    // RANGE_REQ_MAGIC [0xBB][0xA1] — listener wants us to beacon N times
    if (pkt_len >= (int)sizeof(range_req_pkt_t)) {
        range_req_pkt_t *rr = (range_req_pkt_t *)udp_buf;
        if (rr->magic[0] == RANGE_REQ_MAGIC_0 && rr->magic[1] == RANGE_REQ_MAGIC_1) {
            if (rr->target_id != my_id) return;   // not for us
            range_bcn_pkt_t bcn;
            bcn.magic[0]   = RANGE_BCN_MAGIC_0;
            bcn.magic[1]   = RANGE_BCN_MAGIC_1;
            bcn.ver        = 1;
            bcn.shouter_id = my_id;
            for (uint8_t b = 0; b < rr->n_beacons; b++) {
                bcn.bcn_seq = b;
                // ESP-NOW broadcast — bypasses AP, gives true peer-to-peer RSSI at receivers.
                // Broadcast MAC was registered via esp_now_add_peer in setup().
                static const uint8_t ESPNOW_BROADCAST[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
                esp_now_send(ESPNOW_BROADCAST, (uint8_t *)&bcn, sizeof(bcn));
                delay(rr->interval_ms);
            }
            return;
        }
    }

    // Validate poll magic and target
    if (pkt_len < (int)sizeof(poll_pkt_t)) return;
    poll_pkt_t *poll = (poll_pkt_t *)udp_buf;
    if (poll->magic[0] != POLL_MAGIC_0 || poll->magic[1] != POLL_MAGIC_1) return;
    if (my_id == 0) {
        Serial.println("[SHT] Poll received but no ID assigned yet — skipping");
        return;
    }
    if (poll->target_id != my_id && poll->target_id != 0xFF) return;

    // For broadcast polls, stagger response using rotation offset from listener.
    // pad[0] contains the rotation offset (0-3); shouter computes its position
    // in the rotated order: position = (my_id - 1 - offset + 4) % 4.
    // Position 0 responds immediately; others stagger by position * STAGGER_MS.
    if (poll->target_id == 0xFF) {
        uint8_t offset = poll->pad[0] % 4;
        uint8_t position = ((my_id - 1) - offset + 4) % 4;
        if (position > 0) {
            stagger_pending = true;
            stagger_target_ms = millis() + (uint32_t)position * STAGGER_MS;
            memcpy(&stagger_poll, poll, sizeof(poll_pkt_t));
            return;  // response sent from loop() after stagger delay
        }
    }
    // Direct poll or first in rotated broadcast order — respond immediately
    send_poll_response(poll);
}
