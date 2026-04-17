"""Single source of truth for GlassHouse v2 wire-format frame decoding.

Supports all six GlassHouse frame magics (from firmware/perimeter/main/
edge_processing.h and csi_collector.h), the 10-byte link_reporter frame,
and the single-byte 0xAA heartbeat from heartbeat.c.

Wire byte order: little-endian (ESP32-S3 native).
"""

from __future__ import annotations

import struct
from typing import Any


# --- Magic numbers (u32 LE on wire = low byte first in hex) ---
# 0xC5110001 -> b'\x01\x00\x11\xc5'
CSI_MAGIC_LE        = b'\x01\x00\x11\xc5'  # csi_collector.h:15   — raw CSI frame (ADR-018)
VITALS_MAGIC_LE     = b'\x02\x00\x11\xc5'  # edge_processing.h:28
FEATURE_MAGIC_LE    = b'\x03\x00\x11\xc5'  # edge_processing.h:114
FUSED_MAGIC_LE      = b'\x04\x00\x11\xc5'  # edge_processing.h:128
COMPRESSED_MAGIC_LE = b'\x05\x00\x11\xc5'  # edge_processing.h:29
IQ_MAGIC_LE         = b'\x06\x00\x11\xc5'  # edge_processing.h:30
SAR_AMP_MAGIC_LE    = b'\x07\x00\x11\xc5'  # edge_processing.h (Lane B) — batched amplitudes
SAR_FIRE_MAGIC_LE   = b'\x08\x00\x11\xc5'  # sar_fire_sender.h (Option D) — per-subcarrier firehose

CSI_HEADER_SIZE     = 20  # csi_collector.h:18
VITALS_PKT_SIZE     = 32  # edge_processing.h:111 (_Static_assert)
FEATURE_PKT_SIZE    = 48  # edge_processing.h:125
FUSED_PKT_SIZE      = 48  # edge_processing.h:154
SAR_AMP_PKT_SIZE    = 208 # sar_amp_sender.h (_Static_assert)
SAR_AMP_BATCH_SIZE  = 48  # sar_amp_sender.h
SAR_FIRE_HDR_SIZE   = 16  # sar_fire_sender.h (16 header + up to 64 amps)
SAR_FIRE_MAX_SUBCAR = 64
SAR_FIRE_PKT_SIZE   = 80  # sar_fire_sender.h (_Static_assert)


def parse_packet(packet: bytes) -> dict[str, Any]:
    """Decode one COBS-unwrapped UDP payload into a typed record.

    Returns a dict with 'type' set to one of:
        'heartbeat' | 'csi' | 'vitals' | 'feature' | 'fused' |
        'compressed' | 'iq' | 'link' | 'vitals_short' | 'unknown'

    Dispatch order (see module docstring):
      1. 1-byte 0xAA heartbeat (from perimeter heartbeat.c:22-23)
      2. 4-byte magic match (C5110001..0006)
      3. 10-byte link_reporter frame (packet[0]==0x01 && len==10)
      4. fallback: 'unknown'
    """
    rec: dict[str, Any] = {"raw": packet.hex()}
    n = len(packet)

    # 1) Perimeter heartbeat — heartbeat.c sends a single 0xAA byte via UDP.
    if n == 1 and packet[0] == 0xAA:
        rec.update({"type": "heartbeat", "len": 1})
        return rec

    # 2) Magic dispatch. Only meaningful for frames >= 4 bytes.
    if n >= 4:
        magic4 = packet[:4]

        if magic4 == CSI_MAGIC_LE and n >= CSI_HEADER_SIZE:
            # csi_collector.c:126-157 — ADR-018 header layout
            node_id, n_ant, n_subcar, freq_mhz, seq, rssi, noise_floor = struct.unpack_from(
                '<BBHIIbb', packet, 4
            )
            iq_actual = n - CSI_HEADER_SIZE
            iq_declared = n_subcar * 2 * max(n_ant, 1)
            rec.update({
                "type": "csi",
                "node_id": node_id,
                "n_antennas": n_ant,
                "n_subcarriers": n_subcar,
                "freq_mhz": freq_mhz,
                "seq": seq,
                "rssi": rssi,
                "noise_floor": noise_floor,
                "iq_bytes": iq_actual,
                "iq_bytes_declared": iq_declared,
                "len": n,
            })
            return rec

        if magic4 == VITALS_MAGIC_LE:
            # edge_vitals_pkt_t — edge_processing.h:96-109 (32 bytes)
            if n >= VITALS_PKT_SIZE:
                flags = packet[5]
                energy = struct.unpack_from('<f', packet, 16)[0]
                rec.update({
                    "type": "vitals",
                    "node_id": packet[4],
                    "flags": flags,
                    "presence": bool(flags & 0x01),
                    "fall_bit": bool(flags & 0x02),
                    "motion_bit": bool(flags & 0x04),
                    "motion_energy": round(energy, 6),
                    "len": n,
                })
            else:
                rec.update({"type": "vitals_short", "len": n})
            return rec

        if magic4 == FEATURE_MAGIC_LE and n >= FEATURE_PKT_SIZE:
            # edge_feature_pkt_t — edge_processing.h:116-123 (48 bytes, ADR-069)
            node_id, _reserved, seq = struct.unpack_from('<BBH', packet, 4)
            ts_us = struct.unpack_from('<q', packet, 8)[0]
            features = struct.unpack_from('<8f', packet, 16)
            rec.update({
                "type": "feature",
                "node_id": node_id,
                "seq": seq,
                "timestamp_us": ts_us,
                "features": [round(f, 6) for f in features],
                "len": n,
            })
            return rec

        if magic4 == FUSED_MAGIC_LE and n >= FUSED_PKT_SIZE:
            # edge_fused_vitals_pkt_t — edge_processing.h:130-152 (48 bytes, ADR-063)
            flags = packet[5]
            fusion_conf = packet[11]
            motion_energy = struct.unpack_from('<f', packet, 12)[0]
            presence_score = struct.unpack_from('<f', packet, 16)[0]
            mmwave_hr = struct.unpack_from('<f', packet, 24)[0]
            mmwave_br = struct.unpack_from('<f', packet, 28)[0]
            mmwave_dist = struct.unpack_from('<f', packet, 32)[0]
            rec.update({
                "type": "fused",
                "node_id": packet[4],
                "flags": flags,
                "presence": bool(flags & 0x01),
                "mmwave_present": bool(flags & 0x08),
                "fusion_confidence": fusion_conf,
                "motion_energy": round(motion_energy, 6),
                "presence_score": round(presence_score, 6),
                "mmwave_hr_bpm": round(mmwave_hr, 3),
                "mmwave_br_bpm": round(mmwave_br, 3),
                "mmwave_distance_cm": round(mmwave_dist, 2),
                "len": n,
            })
            return rec

        if magic4 == COMPRESSED_MAGIC_LE:
            # ADR-069 delta-compressed CSI. Layout not mechanically verified here
            # — record magic + length until a firmware sample arrives.
            rec.update({
                "type": "compressed",
                "node_id": packet[4] if n > 4 else None,
                "len": n,
            })
            return rec

        if magic4 == IQ_MAGIC_LE and n >= 8:
            # Matches existing debug/capture.py:86-96 layout
            node_id = packet[4]
            channel = packet[5]
            iq_len = struct.unpack_from('<H', packet, 6)[0]
            rec.update({
                "type": "iq",
                "node_id": node_id,
                "channel": channel,
                "iq_len": iq_len,
                "len": n,
            })
            return rec

        if magic4 == SAR_AMP_MAGIC_LE and n >= SAR_AMP_PKT_SIZE:
            # sar_amp_pkt_t — sar_amp_sender.h (208 bytes, Lane B batched SAR amps)
            # Layout: <4s B B H I I 48f>  (magic, node_id, peer_id, n_samples,
            #          batch_start_us, interval_us, amps[48])
            node_id, peer_id, n_samples = struct.unpack_from('<BBH', packet, 4)
            batch_start_us, interval_us = struct.unpack_from('<II', packet, 8)
            amps = struct.unpack_from('<%df' % SAR_AMP_BATCH_SIZE, packet, 16)
            rec.update({
                "type": "sar_amp",
                "node_id": node_id,
                "peer_id": peer_id,
                "n_samples": n_samples,
                "batch_start_us": batch_start_us,
                "interval_us": interval_us,
                "amps": [round(a, 6) for a in amps[:n_samples]],
                "len": n,
            })
            return rec

        if magic4 == SAR_FIRE_MAGIC_LE and n >= SAR_FIRE_PKT_SIZE:
            # sar_fire_pkt_t — sar_fire_sender.h (Option D firehose, 80 bytes)
            # Layout: magic(4) node_id(1) peer_id(1) n_subcar(2) ts_us(4)
            #         rssi(i8) noise_floor(i8) reserved(2) amps[64 u8]
            node_id, peer_id, n_subcar = struct.unpack_from('<BBH', packet, 4)
            ts_us = struct.unpack_from('<I', packet, 8)[0]
            rssi, noise_floor = struct.unpack_from('<bb', packet, 12)
            n_actual = min(n_subcar, SAR_FIRE_MAX_SUBCAR, n - SAR_FIRE_HDR_SIZE)
            amps = list(struct.unpack_from('<%dB' % n_actual, packet, SAR_FIRE_HDR_SIZE))
            rec.update({
                "type": "sar_fire",
                "node_id": node_id,
                "peer_id": peer_id,
                "n_subcar": n_subcar,
                "ts_us": ts_us,
                "rssi": rssi,
                "noise_floor": noise_floor,
                "amps": amps,
                "len": n,
            })
            return rec

    # 3) link_reporter frame: exactly 10 bytes, byte 0 == 0x01, layout '<BBBfBH'
    # link_reporter.c:146. NOT a magic-family frame — distinguishable by len.
    if n == 10 and packet[0] == 0x01:
        try:
            _, node, partner, variance, state, count = struct.unpack('<BBBfBH', packet[:10])
            lo, hi = min(node, partner), max(node, partner)
            rec.update({
                "type": "link",
                "link": f"{lo}{hi}",
                "node": node,
                "partner": partner,
                "variance": round(variance, 6),
                "state": int(state),
                "count": count,
                "len": n,
            })
            return rec
        except struct.error:
            pass  # fall through to unknown

    # 4) Fallback
    rec.update({"type": "unknown", "len": n})
    return rec
