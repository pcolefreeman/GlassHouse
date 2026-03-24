// GHV4/ShouterV4/GHV4Protocol.h
#pragma once
#include <stdint.h>

// ── Inherited from GHV2Protocol.h (unchanged) ────────────────────────────────
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

#define POLL_PAD_SIZE     96
#define CSI_MAX_BYTES    384
#define SHOUTER_CSI_MAX  384

typedef struct __attribute__((packed)) {
    uint8_t  magic[2];
    uint8_t  ver;
    uint8_t  shouter_id;
    uint8_t  src_mac[6];
} hello_pkt_t;             // 10 bytes

typedef struct __attribute__((packed)) {
    uint8_t  magic[2];
    uint8_t  ver;
    uint8_t  target_id;
    uint32_t poll_seq;
    uint32_t listener_ms;
    uint8_t  pad[POLL_PAD_SIZE];
} poll_pkt_t;              // 108 bytes

typedef struct __attribute__((packed)) {
    uint8_t  magic[2];
    uint8_t  ver;
    uint8_t  shouter_id;
    uint32_t tx_seq;
    uint32_t tx_ms;
    uint32_t poll_seq;
    int8_t   poll_rssi;
    int8_t   poll_noise_floor;
    uint16_t csi_len;
    uint8_t  csi[SHOUTER_CSI_MAX];
} response_pkt_t;          // 404 bytes

// ── GHV3 additions ────────────────────────────────────────────────────────────
#define PEER_INFO_MAGIC_0  0xBB
#define PEER_INFO_MAGIC_1  0xA0
#define RANGE_REQ_MAGIC_0  0xBB
#define RANGE_REQ_MAGIC_1  0xA1
#define RANGE_BCN_MAGIC_0  0xBB
#define RANGE_BCN_MAGIC_1  0xA2

// peer_info_pkt_t — listener → each shouter (32 bytes)
typedef struct __attribute__((packed)) {
    uint8_t magic[2];       // [0xBB][0xA0]
    uint8_t ver;            // = 1
    uint8_t n_peers;        // 4
    struct {
        uint8_t shouter_id;
        uint8_t mac[6];
    } peers[4];             // 4 × 7 = 28 bytes
} peer_info_pkt_t;          // 2+1+1+28 = 32 bytes

// range_req_pkt_t — listener → target shouter (7 bytes)
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];      // [0xBB][0xA1]
    uint8_t  ver;           // = 1
    uint8_t  target_id;
    uint8_t  n_beacons;     // default 10
    uint16_t interval_ms;   // default 20
} range_req_pkt_t;          // 7 bytes

// range_bcn_pkt_t — shouter → broadcast (8 bytes)
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];      // [0xBB][0xA2]
    uint8_t  ver;           // = 1
    uint8_t  shouter_id;
    uint32_t bcn_seq;
} range_bcn_pkt_t;          // 8 bytes

// ── ACK packet ───────────────────────────────────────────────────────────
#define ACK_MAGIC_0  0xBB
#define ACK_MAGIC_1  0xA5

// ack_pkt_t — bidirectional (6 bytes)
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];      // [0xBB][0xA5]
    uint8_t  ack_type;      // magic[1] of packet being ACKed (0xFA=HELLO, 0xA0=PEER_INFO)
    uint8_t  assigned_id;   // only for HELLO ACK; 0 otherwise
    uint16_t ack_seq;       // seq from original packet (0 if N/A)
} ack_pkt_t;                // 6 bytes

// ── MUSIC CSI snapshot ─────────────────────────────────────────────────────
#define CSI_SNAP_MAGIC_0  0xBB
#define CSI_SNAP_MAGIC_1  0xA4
#define SER_D_MAGIC_0     0xEE
#define SER_D_MAGIC_1     0xFF
// CSI_SNAP_MAX matches SHOUTER_CSI_MAX so no snapshot is silently truncated.
// offsetof(csi_snap_pkt_t, csi) = 8 bytes:
//   magic(2) + ver(1) + reporter_id(1) + peer_id(1) + snap_seq(1) + csi_len(2) = 8
#define CSI_SNAP_MAX      SHOUTER_CSI_MAX   // 384 bytes

typedef struct __attribute__((packed)) {
    uint8_t  magic[2];       // [0xBB][0xA4]
    uint8_t  ver;            // = 1
    uint8_t  reporter_id;    // shouter that captured this CSI
    uint8_t  peer_id;        // shouter that sent the beacon
    uint8_t  snap_seq;       // 0..N_BCN-1
    uint16_t csi_len;        // bytes in csi[]
    uint8_t  csi[CSI_SNAP_MAX];
} csi_snap_pkt_t;            // 8 + 384 = 392 bytes max

// ── Per-path CSI baseline (stored in SPIFFS) ─────────────────────────────────
// Stores average empty-room CSI amplitude per subcarrier for one shouter pair.
// Used by SAR breathing/presence detection to subtract static environment.
// 128 subcarriers × 2 bytes (amplitude) = 256 bytes of CSI data.
#define BASELINE_N_SUBCARRIERS  128

typedef struct __attribute__((packed)) {
    uint8_t  reporter_id;    // shouter that captured the CSI
    uint8_t  peer_id;        // shouter that sent the beacon
    uint16_t n_samples;      // number of frames averaged into this baseline
    uint32_t timestamp_ms;   // millis() when baseline was captured
    float    amplitude[BASELINE_N_SUBCARRIERS];  // mean |CSI| per subcarrier (float for averaging)
} csi_baseline_t;            // 2 + 2 + 4 + 128*4 = 520 bytes
