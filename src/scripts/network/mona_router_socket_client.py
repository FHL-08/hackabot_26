import argparse
import socket
import time


def recv_line(sock, timeout):
    sock.settimeout(timeout)
    buf = ""
    while "\n" not in buf:
        data = sock.recv(4096)
        if not data:
            raise ConnectionError("Socket closed by bot")
        buf += data.decode("utf-8", errors="replace")
    line, _ = buf.split("\n", 1)
    return line.strip()


def main():
    parser = argparse.ArgumentParser(description="MONA router socket echo client")
    parser.add_argument("--host", default="192.168.0.240", help="Bot IP")
    parser.add_argument("--port", type=int, default=5003, help="Bot TCP port")
    parser.add_argument("--count", type=int, default=5, help="Number of messages")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between messages")
    parser.add_argument("--timeout", type=float, default=2.0, help="Receive timeout")
    args = parser.parse_args()

    print(f"[CLIENT] Connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=3.0) as sock:
        print("[CLIENT] Connected")
        for seq in range(args.count):
            msg = f"hello_{seq}"
            t0 = time.monotonic()
            sock.sendall((msg + "\n").encode("utf-8"))
            echo = recv_line(sock, timeout=args.timeout)
            dt_ms = (time.monotonic() - t0) * 1000.0
            ok = echo == msg
            print(f"[CLIENT] seq={seq} ok={ok} rtt_ms={dt_ms:.1f} sent={msg!r} recv={echo!r}")
            time.sleep(args.delay)

    print("[CLIENT] Done")


if __name__ == "__main__":
    main()
