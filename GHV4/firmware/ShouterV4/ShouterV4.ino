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
#define SHOUTER_ID    4          // 1, 2, 3, or 4 — change before flashing each board
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
    // WiFi.mode() already called in setup(); reconnects from loop() don't need to re-set it
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
    if (sid < 1 || sid > 4 || sid == SHOUTER_ID) return;
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
        if (stored) {
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
        while (1) delay(1000);
    }
    Serial.println("[SHT] CSI capture enabled");

    udp.begin(SHOUTER_PORT);   // safe — WiFi driver initialized above; must be before send_hello()

    connect_and_register();

    // ESP-NOW init — must be after WiFi STA is connected (connect_and_register completed).
    // Called once only; persists across WiFi reconnects.
    if (esp_now_init() != ESP_OK) {
        Serial.println("[SHT] FATAL: esp_now_init failed");
        while (1) delay(1000);
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
            while (1) delay(1000);
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

    // Passive background beacon — 1 per second, fires regardless of ranging phase.
    // Peers receive via on_esp_now_recv and update their peer_table EMA continuously.
    // Ranging reports already sent on every poll, so Python EMA tracks live RSSI.
    // bcn_seq=0xFF is a sentinel (ignored by receiver); no collision risk at 1 Hz.
    static uint32_t last_passive_bcn_ms = 0;
    if (millis() - last_passive_bcn_ms >= 1000) {
        range_bcn_pkt_t bcn = {};
        bcn.magic[0]   = RANGE_BCN_MAGIC_0;
        bcn.magic[1]   = RANGE_BCN_MAGIC_1;
        bcn.ver        = 1;
        bcn.shouter_id = SHOUTER_ID;
        bcn.bcn_seq    = 0xFF;  // passive background beacon
        static const uint8_t BCAST[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
        esp_now_send(BCAST, (uint8_t *)&bcn, sizeof(bcn));
        last_passive_bcn_ms = millis();
    }

    int pkt_len = udp.parsePacket();
    if (pkt_len < 2) return;

    udp.read(udp_buf, sizeof(udp_buf));

    // PEER_INFO_MAGIC [0xBB][0xA0] — listener sends our peers' MACs and IDs
    if (pkt_len >= (int)sizeof(peer_info_pkt_t)) {
        peer_info_pkt_t *pi = (peer_info_pkt_t *)udp_buf;
        if (pi->magic[0] == PEER_INFO_MAGIC_0 && pi->magic[1] == PEER_INFO_MAGIC_1) {
            portENTER_CRITICAL(&peer_mux);   // task context — not ISR variant
            for (int k = 0; k < pi->n_peers && k < 4; k++) {
                uint8_t sid = pi->peers[k].shouter_id;
                if (sid < 1 || sid > 4 || sid == SHOUTER_ID) continue;
                memcpy(peer_table[sid].mac, pi->peers[k].mac, 6);
                peer_table[sid].rssi  = 0;
                peer_table[sid].count = 0;
                peer_table[sid].valid = true;
            }
            portEXIT_CRITICAL(&peer_mux);
            portENTER_CRITICAL(&snap_mux);
            for (int k = 0; k < pi->n_peers && k < 4; k++) {
                uint8_t sid = pi->peers[k].shouter_id;
                if (sid < 1 || sid > 4 || sid == SHOUTER_ID) continue;
                csi_snap_count[sid] = 0;
            }
            portEXIT_CRITICAL(&snap_mux);
            return;  // consumed
        }
    }

    // RANGE_REQ_MAGIC [0xBB][0xA1] — listener wants us to beacon N times
    if (pkt_len >= (int)sizeof(range_req_pkt_t)) {
        range_req_pkt_t *rr = (range_req_pkt_t *)udp_buf;
        if (rr->magic[0] == RANGE_REQ_MAGIC_0 && rr->magic[1] == RANGE_REQ_MAGIC_1) {
            if (rr->target_id != SHOUTER_ID) return;   // not for us
            range_bcn_pkt_t bcn;
            bcn.magic[0]   = RANGE_BCN_MAGIC_0;
            bcn.magic[1]   = RANGE_BCN_MAGIC_1;
            bcn.ver        = 1;
            bcn.shouter_id = SHOUTER_ID;
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
    if (poll->target_id != SHOUTER_ID) return;
    last_poll_rx_ms = millis();
    Serial.printf("[SHT] POLL seq=%lu\n", (unsigned long)poll->poll_seq);

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
    resp.poll_seq   = poll->poll_seq;
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
    Serial.printf("[SHT] SHOUT tx=%lu poll=%lu csi=%u\n",
        (unsigned long)tx_seq, (unsigned long)poll->poll_seq,
        (unsigned)resp.csi_len);
    tx_seq++;

    // Send ranging report BEFORE CSI snapshots. [BB][A3] is only 14 bytes and must
    // arrive at the listener; the snapshot burst (up to 90 × ~392 bytes) can flood
    // the listener's UDP RX queue and silently drop any packet sent after it.
    // Sending [BB][A3] first guarantees the RSSI data reaches the listener regardless
    // of whether the snapshot batch overflows the queue.
    ranging_rpt_pkt_t rpt = {};
    rpt.magic[0]   = RANGE_RPT_MAGIC_0;
    rpt.magic[1]   = RANGE_RPT_MAGIC_1;
    rpt.ver        = 1;
    rpt.shouter_id = SHOUTER_ID;
    portENTER_CRITICAL(&peer_mux);
    for (int i = 1; i <= 4; i++) {
        if (peer_table[i].valid) {
            rpt.peer_rssi[i]  = peer_table[i].rssi;
            rpt.peer_count[i] = peer_table[i].count;
        }
    }
    portEXIT_CRITICAL(&peer_mux);
    udp.beginPacket(LISTENER_IP, LISTENER_PORT);
    udp.write((uint8_t *)&rpt, sizeof(rpt));
    udp.endPacket();

    // Transmit buffered CSI snapshots to listener. The listener's handle_incoming_udp
    // drains these during the poll-wait window for the next shouter in the cycle.
    // Rate: up to 3 peers × 30 snaps × 392 bytes = ~35 KB. Some snaps may be dropped
    // if the listener RX queue fills — MUSIC ranging degrades gracefully; RSSI is safe
    // because [BB][A3] was already sent above.
    for (uint8_t peer = 1; peer <= 4; peer++) {
        if (peer == SHOUTER_ID) continue;
        portENTER_CRITICAL(&snap_mux);
        uint8_t n = csi_snap_count[peer];
        csi_snap_count[peer] = 0;  // clear after reading — snaps are one-shot per ranging phase
        portEXIT_CRITICAL(&snap_mux);
        for (uint8_t s = 0; s < n; s++) {
            portENTER_CRITICAL(&snap_mux);
            CsiEntry e = csi_snap_buf[peer][s];
            portEXIT_CRITICAL(&snap_mux);
            csi_snap_pkt_t pkt;
            pkt.magic[0]     = CSI_SNAP_MAGIC_0;
            pkt.magic[1]     = CSI_SNAP_MAGIC_1;
            pkt.ver          = 1;
            pkt.reporter_id  = SHOUTER_ID;
            pkt.peer_id      = peer;
            pkt.snap_seq     = s;
            pkt.csi_len      = e.len < CSI_SNAP_MAX ? e.len : CSI_SNAP_MAX;
            memcpy(pkt.csi, e.bytes, pkt.csi_len);
            udp.beginPacket(LISTENER_IP, LISTENER_PORT);
            udp.write((uint8_t*)&pkt,
                      (uint16_t)(offsetof(csi_snap_pkt_t, csi) + pkt.csi_len));
            udp.endPacket();
            delay(15);  // Pace snapshot packets — 15ms > 4.25ms serial TX floor with good margin
        }
    }
}
