import subprocess
import tempfile
import time
import os
from xml.sax.saxutils import escape

SSID = "TP-Link_6C24"
PASSWORD = "17346559"

AUTHENTICATION = "WPA2PSK"
ENCRYPTION = "AES"


def run_command(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=False
    )
    return result.returncode, result.stdout, result.stderr


def build_wifi_profile_xml(ssid, password):
    ssid_xml = escape(ssid)
    password_xml = escape(password)

    return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid_xml}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid_xml}</name>
        </SSID>
        <nonBroadcast>false</nonBroadcast>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>{AUTHENTICATION}</authentication>
                <encryption>{ENCRYPTION}</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password_xml}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>
"""


def get_current_wifi_info():
    code, stdout, stderr = run_command(["netsh", "wlan", "show", "interfaces"])
    if code != 0:
        return None

    state = None
    ssid = None

    for line in stdout.splitlines():
        stripped = line.strip()

        if stripped.startswith("State"):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                state = parts[1].strip()

        # Avoid matching BSSID
        if stripped.startswith("SSID") and not stripped.startswith("SSID name") and "BSSID" not in stripped:
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                ssid = parts[1].strip()

    return {"state": state, "ssid": ssid, "raw": stdout}


def main():
    profile_xml = build_wifi_profile_xml(SSID, PASSWORD)

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        profile_path = f.name
        f.write(profile_xml)

    try:
        print(f"Adding Wi-Fi profile for {SSID}...")
        code, stdout, stderr = run_command([
            "netsh", "wlan", "add", "profile",
            f"filename={profile_path}",
            "user=current"
        ])
        print(stdout.strip())
        if stderr.strip():
            print(stderr.strip())

        if code != 0:
            raise RuntimeError("Failed to add Wi-Fi profile.")

        print(f"Connecting to {SSID}...")
        code, stdout, stderr = run_command([
            "netsh", "wlan", "connect",
            f"name={SSID}",
            f"ssid={SSID}"
        ])
        print(stdout.strip())
        if stderr.strip():
            print(stderr.strip())

        if code != 0:
            raise RuntimeError("Failed to start Wi-Fi connection.")

        # Wait a bit for the connection to settle
        connected = False
        for _ in range(10):
            time.sleep(1)
            info = get_current_wifi_info()
            if info and info["state"] and info["ssid"]:
                print(f"State: {info['state']} | SSID: {info['ssid']}")
                if info["state"].lower() == "connected" and info["ssid"] == SSID:
                    connected = True
                    break

        if connected:
            print(f"Success: connected to {SSID}")
        else:
            print("Did not confirm a successful connection.")
            info = get_current_wifi_info()
            if info:
                print(info["raw"])

    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)


if __name__ == "__main__":
    main()