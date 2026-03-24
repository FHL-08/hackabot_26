import argparse
import json
import socket
import time


DEFAULT_SERVER_IP = "192.168.137.1"
DEFAULT_SERVER_PORT = 5001
DEFAULT_COUNT = 5
DEFAULT_DELAY = 0.5


class TcpJsonClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.settimeout(5)

    def send_packet(self, packet):
        message = json.dumps(packet) + "\n"
        self.sock.sendall(message.encode("utf-8"))

    def recv_packet(self):
        while "\n" not in self.buffer:
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("Connection closed by server")
            self.buffer += data.decode("utf-8")

        line, self.buffer = self.buffer.split("\n", 1)
        return json.loads(line)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


def parse_args():
    parser = argparse.ArgumentParser(description="Wi-Fi JSON echo client")
    parser.add_argument("--host", default=DEFAULT_SERVER_IP, help="Server IP address")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help="Server port")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Messages to send")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between messages (sec)")
    parser.add_argument("--message", default="hello_from_client", help="Message payload")
    return parser.parse_args()


def main():
    args = parse_args()

    client = TcpJsonClient(args.host, args.port)
    client.connect()
    print(f"[CLIENT] Connected to {args.host}:{args.port}")

    try:
        for seq in range(args.count):
            packet = {
                "type": "echo",
                "seq": seq,
                "message": f"{args.message}_{seq}"
            }
            client.send_packet(packet)
            reply = client.recv_packet()
            print(f"[CLIENT] Reply: {reply}")
            time.sleep(args.delay)
    finally:
        client.close()
        print("[CLIENT] Closed")


if __name__ == "__main__":
    main()
