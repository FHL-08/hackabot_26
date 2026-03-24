# hackabot_26
The ice9 swarm challenge.

## Project layout

- `src/` keeps core runtime entrypoints and shared modules.
- Main runtime entrypoints in `src/`: `laptop_server.py`, `tracker_sender.py`, `swarm_orchestrator.py`.
- `src/scripts/vision/`, `src/scripts/network/`, `src/scripts/simulation/` contain grouped script implementations.
- Run grouped scripts as modules, for example: `python -m src.scripts.vision.generate_markers`.
- `src/data/` contains checked-in baseline calibration assets.
- `src/generated/` contains generated outputs and is ignored by git.
- `firmware/execute_server_commands/execute_server_commands.ino` contains the MONA robot firmware used with this Python control stack.

## Network configuration (important)

This project does not require `TP-Link_6C24` specifically.
`TP-Link_6C24` is only the current default in code and can be changed.

If you are using a different Wi-Fi network:

- Put the laptop and all MONA robots on the same LAN.
- Update firmware Wi-Fi credentials in `firmware/execute_server_commands/execute_server_commands.ino`:
	- `WIFI_SSID`
	- `WIFI_PASSWORD`
- Update robot target host in the same file:
	- `LAPTOP_HOST` should be the laptop IP on that network.
- Optional warning text in laptop runtime can be changed in `src/laptop_server.py`:
	- `DEFAULT_EXPECTED_SSID` is only used for warning messages.

In short: any network is fine as long as IP/ports match and all devices can reach each other.

## Environment setup (Windows PowerShell)

1. Create a virtual environment (replace `3.12` with your installed Python version):

```powershell
py -3.12 -m venv venv
```

2. Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Arduino setup for MONA robots (required)

Before flashing `firmware/execute_server_commands/execute_server_commands.ino`, install the Arduino dependencies below.

1. Install Arduino IDE 2.x.

2. For MONA-ESP board setup, follow the instructions here:
	- `https://github.com/ICE9-Robotics/MONA_ESP_lib/tree/main`

3. Add the ESP32WebServer library to Arduino IDE:
	- Open this page in your browser:
	  - `https://github.com/Pedroalbuquerque/ESP32WebServer/tree/master`
	- On that page, click the green `Code` button and choose `Download ZIP`.
	- In Arduino IDE, go to `Sketch -> Include Library -> Add .ZIP Library...`.
	- Browse to the downloaded ZIP file, select it, and install.

4. Restart Arduino IDE after installing libraries.

5. Select board and port, then upload:
	- Open `firmware/execute_server_commands/execute_server_commands.ino`
	- Select your ESP32-compatible MONA board in `Tools -> Board`
	- Select the correct serial port in `Tools -> Port`
	- Click `Upload`

If library includes fail during compile, verify `ESP32WebServer` is installed in Arduino IDE via the ZIP library flow above.

## Runtime command (auto from tracker)

From the repository root, run:

```powershell
python src/laptop_server.py --auto-from-tracker
```

Your longer command is valid too, but `src/` is required if you run from the repo root:

```powershell
python src/laptop_server.py --auto-from-tracker --host 0.0.0.0 --port 5005 --tracker-host 0.0.0.0 --tracker-port 5000
```

Default runtime values in `laptop_server.py` are:

- `--host 0.0.0.0` (bot command server bind address)
- `--port 5005` (bot command TCP port)
- `--tracker-host 0.0.0.0` (tracker listener bind address)
- `--tracker-port 5000` (tracker listener TCP port)

## Tracker sender

Run tracker sender in a second terminal:

```powershell
python src/tracker_sender.py --server-ip 127.0.0.1 --server-port 5000
```

Tracker sender destination defaults:

- `--server-ip 127.0.0.1`
- `--server-port 5000`

## Quick robot control test (without laptop_server.py)

Use this lightweight test server to send wheel velocity commands directly to robots over Wi-Fi:

```powershell
python -m src.scripts.network.wheel_velocity_test_server --host 0.0.0.0 --port 5005 --mode bot --bot-id 1 --left 8 --right 8 --hz 10
```

The robot firmware already understands this command format (`bot(id,left,right)`), so you can verify motor control without running tracker or MPC.

Send one command stream for all three robots using six-float broadcast format:

```powershell
python -m src.scripts.network.wheel_velocity_test_server --host 0.0.0.0 --port 5005 --mode broadcast --b1-left 8 --b1-right 8 --b2-left 0 --b2-right 0 --b3-left 0 --b3-right 0 --hz 10
```

Notes:

- Keep `LAPTOP_HOST` and `LAPTOP_PORT` in firmware aligned with this script's `--host/--port` target.
- Put laptop and robots on the same Wi-Fi LAN.
- Stop the test server before running `laptop_server.py` again to avoid port conflicts.

### Host and tracker-host guidance

- Usually keep `--host` and `--tracker-host` the same on the laptop server process.
- `0.0.0.0` means "listen on all local interfaces". This is usually the best bind value.
- If you want to restrict interfaces, set both to the laptop's specific LAN IPv4 (for example `192.168.0.23`).

Important distinction:

- `--host` and `--tracker-host` are bind addresses for `laptop_server.py`.
- `tracker_sender.py` must connect to a real destination address, not `0.0.0.0`.
- Use `tracker_sender.py --server-ip ... --server-port ...` to set destination without editing code.

You can find your laptop IPv4 in PowerShell with:

```powershell
ipconfig
```

### How to choose ports

- Current defaults are by project convention: tracker on `5000`, bot commands on `5005`.
- You can change them to any free TCP ports, but both ends must match.
- If you change `--tracker-port` on `laptop_server.py`, pass the same port to `tracker_sender.py --server-port`.
- If a port is busy, pick another and retry.

Check if a port is already in use:

```powershell
netstat -ano | findstr :5000
netstat -ano | findstr :5005
```

To change them, pass different CLI values, for example:

```powershell
python src/laptop_server.py --auto-from-tracker --host 192.168.0.23 --port 6005 --tracker-host 192.168.0.23 --tracker-port 6000
```

## ArUco marker mapping

The runtime currently maps marker IDs like this:

- `12 -> r1`
- `9 -> r2`
- `4 -> r3`

This mapping is defined in `MARKER_TO_ROBOT` inside `src/laptop_server.py`.
To change marker IDs, edit that dictionary and keep the robot names `r1`, `r2`, `r3` consistent with the rest of the control stack.
