"""GlassHouse v2 main loop — serial -> aggregator -> zone -> display."""

from __future__ import annotations

import argparse
import sys
import time

from python.serial_receiver import SerialReceiver
from python.link_aggregator import LinkAggregator
from python.zone_detector import ZoneDetector, build_live_zone_tracker
from python.display import Display

try:
    from python.iq_processor import IQProcessor
    _HAS_IQ = True
except ImportError:
    _HAS_IQ = False


def main(port: str, baud: int, headless: bool = False, debug: bool = False) -> None:
    receiver = SerialReceiver(port=port, baud=baud)
    aggregator = LinkAggregator()
    detector = ZoneDetector(link_states_fn=aggregator.get_link_states)
    tracker = build_live_zone_tracker()
    display = None if headless else Display()

    if _HAS_IQ:
        iq_proc = IQProcessor()
    else:
        iq_proc = None
        print("WARNING: numpy/scipy not installed — I/Q DSP disabled")

    last_print = 0.0
    last_vitals_print = 0.0
    PRINT_INTERVAL = 0.2  # Throttle console output to 5 Hz
    pkt_count = 0
    vitals_count = 0
    iq_feed_count = 0
    OCC_WINDOW = 7    # Sliding window size for debounce
    OCC_THRESH = 3    # Require 3 of last 7 frames with zone detection
    occ_history: list[bool] = []

    try:
        for packet in receiver.read_packets():
            pkt_count += 1
            # Track vitals packets
            if len(packet) >= 4 and packet[:4] == b'\x02\x00\x11\xC5':
                vitals_count += 1
                if debug:
                    import struct
                    now = time.monotonic()
                    if (now - last_vitals_print) >= 1.0:
                        last_vitals_print = now
                        flags = packet[5] if len(packet) > 5 else 0
                        presence = bool(flags & 0x01)
                        motion_e = struct.unpack_from('<f', packet, 16)[0] if len(packet) >= 20 else 0.0
                        print(f"  VITALS: presence={presence} motion_e={motion_e:.4f} "
                              f"flags=0x{flags:02x} (pkts={pkt_count} vitals={vitals_count})")
            aggregator.feed(packet)

            # Drain I/Q frames and feed to DSP processor
            if iq_proc is not None:
                for node_id, channel, iq_data in aggregator.drain_iq():
                    iq_proc.feed(node_id, channel, iq_data)
                    iq_feed_count += 1
                    if iq_feed_count % 10 == 0:  # ~1s at 10 Hz
                        vitals_iq = iq_proc.get_vitals()
                        br = vitals_iq["breathing_bpm"]
                        br_c = vitals_iq["breathing_confidence"]
                        if br > 0 and debug:
                            print(f"  IQ-DSP: breathing ~{br:.0f} BPM (conf={br_c:.2f})")

            if aggregator.links_updated():
                raw_zone = detector.estimate()
                # Sliding-window debounce: require OCC_THRESH of last
                # OCC_WINDOW frames to declare occupied.  Handles bursty
                # link report delivery where consecutive hits are rare.
                occ_history.append(raw_zone.zone is not None)
                if len(occ_history) > OCC_WINDOW:
                    occ_history.pop(0)
                occupied = sum(occ_history) >= OCC_THRESH
                stable = tracker.update(raw_zone, occupied=occupied)

                if display:
                    display.update(stable, occupied)
                else:
                    now = time.monotonic()
                    if (now - last_print) >= PRINT_INTERVAL:
                        last_print = now
                        zone_str = stable.zone.value if stable.zone else "---"
                        status = "OCC" if occupied else "EMPTY"
                        line = f"[{status}] Zone: {zone_str}  conf={stable.confidence:.2f}"

                        if debug:
                            # Show per-link variance and rolling baseline
                            link_states = aggregator.get_link_states()
                            baselines = detector._baselines
                            links_info = []
                            for lid in sorted(link_states):
                                ls = link_states[lid]
                                v = ls['variance']
                                b = baselines.get(lid, 0.0)
                                ratio = v / max(b, 0.01)
                                wf = ls.get('window_full', False)
                                if ratio > detector._DEVIATION_MULT:
                                    flag = "*"  # spike
                                elif (lid in {'23', '34'} and b >= detector._ABSORPTION_FLOOR
                                      and (not wf or v < b * detector._ABSORPTION_MULT)):
                                    flag = "v"  # absorption
                                else:
                                    flag = " "
                                links_info.append(
                                    f"{lid}:{v:.1f}/{b:.1f}={ratio:.1f}{flag}"
                                )
                            e_str = f" E={aggregator.motion_energy:.2f}"
                            line += e_str + f"  | {' '.join(links_info)}"

                        print(line)

    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if display:
            display.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GlassHouse v2")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Show per-link state in console output")
    args = parser.parse_args()
    main(args.port, args.baud, args.headless, args.debug)
