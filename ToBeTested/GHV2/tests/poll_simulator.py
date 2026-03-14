"""Simulates 5 polls to a specific shouter and prints the responses."""
import socket, struct, time, sys

SHOUTER_IP   = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.2"
LISTENER_BIND_IP = ""  # bind to all interfaces
LISTENER_PORT    = 3333
SHOUTER_PORT     = 3334

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTENER_BIND_IP, LISTENER_PORT))
sock.settimeout(0.5)

for i in range(5):
    # poll_pkt_t: magic(2) ver(1) target_id(1) poll_seq(4) listener_ms(4) pad(96)
    poll = struct.pack("<2sBBII96s",
        b'\xBB\xCC', 1, 1, i, int(time.time() * 1000) & 0xFFFFFFFF, b'\xA5' * 96)
    sock.sendto(poll, (SHOUTER_IP, SHOUTER_PORT))
    try:
        data, _ = sock.recvfrom(500)
        assert data[:2] == b'\xBB\xEE', f"Bad magic: {data[:2].hex()}"
        # response_pkt_t layout (packed): magic(2) ver(1) shouter_id(1)
        #   tx_seq(4@4) tx_ms(4@8) poll_seq(4@12) poll_rssi(1@16) poll_nf(1@17)
        #   csi_len(2@18) csi[N@20]
        tx_seq,    = struct.unpack_from("<I", data, 4)
        poll_echo, = struct.unpack_from("<I", data, 12)
        csi_len,   = struct.unpack_from("<H", data, 18)
        print(f"Poll {i}: tx_seq={tx_seq}  poll_seq={poll_echo}  csi_len={csi_len}")
    except socket.timeout:
        print(f"Poll {i}: TIMEOUT")
    time.sleep(0.1)
