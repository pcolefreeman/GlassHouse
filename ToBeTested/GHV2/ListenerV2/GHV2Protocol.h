#pragma once
#include <stdint.h>

// ── Magic bytes ───────────────────────────────────────────────────────────────
#define POLL_MAGIC_0    0xBB
#define POLL_MAGIC_1    0xCC
#define RESP_MAGIC_0    0xBB
#define RESP_MAGIC_1    0xEE
#define HELLO_MAGIC_0   0xBB
#define HELLO_MAGIC_1   0xFA
#define SER_A_MAGIC_0   0xAA
#define SER_A_MAGIC_1   0x55
#define SER_B_MAGIC_0   0xBB
#define SER_B_MAGIC_1   0xDD

// ── Sizes ─────────────────────────────────────────────────────────────────────
#define POLL_PAD_SIZE     96
#define CSI_MAX_BYTES    384
#define SHOUTER_CSI_MAX  384

// ── UDP packets ───────────────────────────────────────────────────────────────

// hello_pkt_t — shouter → listener on connect (10 bytes)
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];     // [HELLO_MAGIC_0, HELLO_MAGIC_1] = [0xBB, 0xFA]
    uint8_t  ver;          // = 1
    uint8_t  shouter_id;   // 1–4
    uint8_t  src_mac[6];   // shouter's own MAC from WiFi.macAddress()
} hello_pkt_t;             // 2+1+1+6 = 10 bytes

// poll_pkt_t — listener → shouter (108 bytes)
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];           // [POLL_MAGIC_0, POLL_MAGIC_1] = [0xBB, 0xCC]
    uint8_t  ver;                // = 1
    uint8_t  target_id;          // 1–4
    uint32_t poll_seq;           // monotonically increasing, little-endian
    uint32_t listener_ms;        // listener millis() at send time
    uint8_t  pad[POLL_PAD_SIZE]; // 0xA5 fill — forces 802.11n HT frame encoding
} poll_pkt_t;                    // 2+1+1+4+4+96 = 108 bytes

// response_pkt_t — shouter → listener (404 bytes)
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];          // [RESP_MAGIC_0, RESP_MAGIC_1] = [0xBB, 0xEE]
    uint8_t  ver;               // = 1
    uint8_t  shouter_id;        // 1–4
    uint32_t tx_seq;            // starts at 1; 0 never used
    uint32_t tx_ms;             // shouter millis() when response was built
    uint32_t poll_seq;          // echoed from poll_pkt_t.poll_seq
    int8_t   poll_rssi;         // RSSI shouter observed for poll frame (dBm)
    int8_t   poll_noise_floor;  // shouter noise floor (dBm)
    uint16_t csi_len;           // ≤ SHOUTER_CSI_MAX; 0 = ring buffer empty
    uint8_t  csi[SHOUTER_CSI_MAX]; // I/Q pairs
} response_pkt_t;               // 2+1+1+4+4+4+1+1+2+384 = 404 bytes

/*
 * Serial frame A: [0xAA][0x55]   — emitted by listener CSI callback
 *   magic(2) ver(1) flags(1) ts_ms(4) rssi(1) nf(1) mac(6) poll_seq(4) csi_len(2) csi[N]
 *   Header after magic: 20 bytes.  Max total: 406 bytes.
 *
 * Serial frame B: [0xBB][0xDD]   — emitted by listener after each poll (HIT or MISS)
 *   magic(2) ver(1) flags(1) listener_ms(4) tx_seq(4) tx_ms(4) shouter_id(1)
 *   poll_seq(4) poll_rssi(1) poll_nf(1) mac(6) csi_len(2) csi[N]
 *   Header after magic: 29 bytes.  Max total: 415 bytes.
 *   flags: 0x01 = HIT (valid response), 0x00 = MISS (timeout)
 */
