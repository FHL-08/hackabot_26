import argparse
import socket
import threading
import time
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send wheel velocity test commands directly to MONA bots over TCP."
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address for the test server.")
    parser.add_argument("--port", type=int, default=5005,
                        help="Bind port for the test server.")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Command send rate in Hz.")
    parser.add_argument(
        "--mode",
        choices=["broadcast", "bot"],
        default="broadcast",
        help="broadcast: send six-float format for all bots; bot: send bot(id,l,r).",
    )
    parser.add_argument(
        "--bot-id", type=int, choices=[1, 2, 3], default=1, help="Bot ID for bot mode.")
    parser.add_argument("--left", type=float, default=0.0,
                        help="Left wheel target for bot mode.")
    parser.add_argument("--right", type=float, default=0.0,
                        help="Right wheel target for bot mode.")
    parser.add_argument("--b1-left", type=float, default=0.0)
    parser.add_argument("--b1-right", type=float, default=0.0)
    parser.add_argument("--b2-left", type=float, default=0.0)
    parser.add_argument("--b2-right", type=float, default=0.0)
    parser.add_argument("--b3-left", type=float, default=0.0)
    parser.add_argument("--b3-right", type=float, default=0.0)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> str:
    if args.mode == "bot":
        return f"bot({args.bot_id},{args.left:.6f},{args.right:.6f})"

    return (
        f"{args.b1_left:.6f} {args.b1_right:.6f} "
        f"{args.b2_left:.6f} {args.b2_right:.6f} "
        f"{args.b3_left:.6f} {args.b3_right:.6f}"
    )


def accept_loop(server: socket.socket, clients: List[Tuple[socket.socket, Tuple[str, int]]], lock: threading.Lock) -> None:
    while True:
        conn, addr = server.accept()
        conn.settimeout(0.1)
        with lock:
            clients.append((conn, addr))
        print(f"[connect] {addr[0]}:{addr[1]}")


def main() -> None:
    args = parse_args()
    if args.hz <= 0:
        raise ValueError("--hz must be > 0")

    period_s = 1.0 / args.hz
    command = build_command(args)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(8)

    clients: List[Tuple[socket.socket, Tuple[str, int]]] = []
    clients_lock = threading.Lock()

    accept_thread = threading.Thread(
        target=accept_loop,
        args=(server, clients, clients_lock),
        daemon=True,
    )
    accept_thread.start()

    print(f"[server] listening on {args.host}:{args.port}")
    print(f"[server] mode={args.mode} rate={args.hz:.2f}Hz")
    print(f"[server] command: {command}")
    print("[server] press Ctrl+C to stop")

    try:
        while True:
            line = (command + "\n").encode("utf-8")
            with clients_lock:
                snapshot = list(clients)

            disconnected: List[Tuple[socket.socket, Tuple[str, int]]] = []
            for conn, addr in snapshot:
                try:
                    conn.sendall(line)
                except OSError:
                    disconnected.append((conn, addr))

            if disconnected:
                with clients_lock:
                    for dead_conn, dead_addr in disconnected:
                        try:
                            dead_conn.close()
                        except OSError:
                            pass
                        clients[:] = [
                            c for c in clients if c[0] is not dead_conn]
                        print(f"[disconnect] {dead_addr[0]}:{dead_addr[1]}")

            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\n[server] stopping")
    finally:
        with clients_lock:
            for conn, _ in clients:
                try:
                    conn.close()
                except OSError:
                    pass
            clients.clear()
        server.close()


if __name__ == "__main__":
    main()
