import subprocess

TARGET_SSID = "TP-Link_6C24"


def run_command(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=False
    )
    return result.returncode, result.stdout, result.stderr


def get_current_wifi_info():
    code, stdout, stderr = run_command(["netsh", "wlan", "show", "interfaces"])
    if code != 0:
        raise RuntimeError(stderr.strip() or "Failed to read Wi-Fi interface info.")

    state = None
    ssid = None

    for line in stdout.splitlines():
        stripped = line.strip()

        if stripped.startswith("State"):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                state = parts[1].strip()

        if stripped.startswith("SSID") and "BSSID" not in stripped:
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                ssid = parts[1].strip()

    return state, ssid, stdout


def main():
    state, ssid, raw = get_current_wifi_info()

    print(f"Current state: {state}")
    print(f"Current SSID: {ssid}")

    if state and ssid and state.lower() == "connected" and ssid == TARGET_SSID:
        print("TEST PASSED")
    else:
        print("TEST FAILED")
        print("\nFull interface output:\n")
        print(raw)


if __name__ == "__main__":
    main()