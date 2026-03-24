import socket
import subprocess
import time


HOST = "0.0.0.0"
PORT = 5002
EXPECTED_SSID = "TP-Link_6C24"
PING_INTERVAL_SECONDS = 1.0
PING_TIMEOUT_SECONDS = 3.0
WAIT_LOG_SECONDS = 5.0


def run_command(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=False,
    )
    return result.returncode, result.stdout


def get_wifi_ssid():
    code, stdout = run_command(["netsh", "wlan", "show", "interfaces"])
    if code != 0:
        return None

    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("SSID") and "BSSID" not in stripped:
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def get_local_ipv4s():
    code, stdout = run_command(["ipconfig"])
    if code != 0:
        return []

    ips = []
    for line in stdout.splitlines():
        if "IPv4 Address" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                ip = parts[1].strip()
                if ip and not ip.startswith("127."):
                    ips.append(ip)
    return sorted(set(ips))


def send_line(conn, text):
    conn.sendall((text + "\n").encode("utf-8"))


def run_client_session(conn, addr):
    print(f"[SERVER] Client connected: {addr}")
    conn.settimeout(0.1)

    buffer = ""
    seq = 0
    next_ping_at = time.monotonic()
    pending = {}

    while True:
        now = time.monotonic()

        if now >= next_ping_at:
            msg = f"PING {seq} {int(time.time() * 1000)}"
            try:
                send_line(conn, msg)
            except OSError:
                print("[SERVER] Client disconnected while sending")
                break
            pending[msg] = now
            print(f"[SERVER] -> {msg}")
            seq += 1
            next_ping_at = now + PING_INTERVAL_SECONDS

        try:
            data = conn.recv(4096)
        except socket.timeout:
            data = None
        except ConnectionResetError:
            print("[SERVER] Connection reset by peer")
            break

        if data == b"":
            print("[SERVER] Client disconnected")
            break

        if data:
            buffer += data.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                if line in pending:
                    rtt_ms = (time.monotonic() - pending.pop(line)) * 1000.0
                    print(f"[SERVER] <- ECHO OK: {line} (rtt={rtt_ms:.1f} ms)")
                else:
                    print(f"[SERVER] <- {line}")

        now = time.monotonic()
        for msg, sent_at in list(pending.items()):
            if now - sent_at > PING_TIMEOUT_SECONDS:
                print(f"[SERVER] !! Timeout waiting for echo: {msg}")
                del pending[msg]


def main():
    ssid = get_wifi_ssid()
    local_ips = get_local_ipv4s()

    print(f"[SERVER] Wi-Fi SSID: {ssid}")
    if ssid != EXPECTED_SSID:
        print(f"[SERVER] Warning: expected SSID {EXPECTED_SSID}, got {ssid}")
    print(f"[SERVER] Local IPv4s: {', '.join(local_ips) if local_ips else '(none)'}")
    print("[SERVER] Put your laptop Wi-Fi IPv4 into SERVER_IP in mona_hotspot_bot_client.ino")
    print(f"[SERVER] Listening on {HOST}:{PORT}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(WAIT_LOG_SECONDS)

    try:
        while True:
            print("[SERVER] Waiting for bot connection...")
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            if addr[0] in local_ips:
                print(f"[SERVER] Ignoring local self-connection from {addr}")
                conn.close()
                continue

            try:
                run_client_session(conn, addr)
            finally:
                conn.close()
    finally:
        server.close()


if __name__ == "__main__":
    main()
