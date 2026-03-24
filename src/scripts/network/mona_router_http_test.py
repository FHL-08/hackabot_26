import argparse
import time
import urllib.parse
import urllib.request


def http_get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body


def main():
    parser = argparse.ArgumentParser(description="MONA HTTP router connectivity test")
    parser.add_argument("--bot-ip", default="192.168.0.240", help="Bot IP address")
    parser.add_argument("--count", type=int, default=5, help="Number of requests")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    base = f"http://{args.bot_ip}"
    print(f"[HTTP-TEST] Target: {base}")

    # Basic reachability check.
    status, body = http_get(f"{base}/ping", timeout=args.timeout)
    print(f"[HTTP-TEST] /ping -> status={status} body={body!r}")

    # Repeated echo test.
    for seq in range(args.count):
        msg = f"hello_{seq}"
        qs = urllib.parse.urlencode({"msg": msg})
        t0 = time.monotonic()
        status, body = http_get(f"{base}/echo?{qs}", timeout=args.timeout)
        dt_ms = (time.monotonic() - t0) * 1000.0
        ok = body == msg
        print(
            f"[HTTP-TEST] /echo seq={seq} status={status} ok={ok} "
            f"rtt_ms={dt_ms:.1f} recv={body!r}"
        )
        time.sleep(args.delay)

    status, body = http_get(f"{base}/status", timeout=args.timeout)
    print(f"[HTTP-TEST] /status -> status={status} body={body}")


if __name__ == "__main__":
    main()
